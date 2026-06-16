// SPDX-License-Identifier: AGPL-3.0-only

//! FibQuant fidelity spike (Step 1 of `docs/design/fibquant-kv-compression.md`).
//!
//! Two modes:
//! 1. *No args* — synthetic self-check: sample random Gaussian `d`-vectors
//!    (the codec normalizes, so by FibQuant's universality the source seen is
//!    always the spherical-Beta `f_{d,k}`), encode→decode at several rates, and
//!    print per-coordinate reconstruction MSE + vector cosine vs rate. Confirms
//!    the codebook/codec reproduce the paper's rate–distortion *trend*.
//! 2. *Path arg* — real KV: load a dumped KV container (format `FKV1`), sweep
//!    `(k, N)` rates, compress K and V, and print the attention-output cosine
//!    similarity (paper Eq. 3) — the success metric (≥0.95 at the chosen rate).
//!
//! `FKV1` container (all little-endian):
//!   magic[4]="FKV1" | u32 d | u32 nkv | u32 nq | u32 T
//!   K: T*nkv*d bf16 | V: T*nkv*d bf16 | Q: nq*d bf16

use std::env;

use atlas_quant::fibquant::{FibQuantCodec, Rotation, attention_output_cosine, mean_vector_cosine};
use half::bf16;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

const SEED: u64 = 0xF1B0_0A0B_7041;

/// Pick the rotation from `FIB_ROT` (`hadamard` reuses Atlas's WHT kernel path;
/// default `haar` is the dense paper rotation). Decides whether the eventual
/// `.cu` kernel can reuse `wht_bf16` or must upload a d×d rotation buffer.
fn build_codec(d: usize, k: usize, n: usize) -> FibQuantCodec {
    match std::env::var("FIB_ROT").as_deref() {
        Ok("hadamard") => FibQuantCodec::new_with_rotation(d, k, n, Rotation::hadamard(d)),
        _ => FibQuantCodec::new(d, k, n, SEED),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        run_real_kv(&args[1]);
    } else {
        run_synthetic();
    }
}

fn run_synthetic() {
    let d = 256usize; // A3B head_dim
    let rates: &[(usize, usize)] = &[(2, 16), (2, 64), (2, 256), (4, 256), (4, 1024), (8, 256)];
    println!("== FibQuant synthetic self-check (d={d}, source = spherical-Beta) ==");
    println!(
        "{:>4} {:>5} {:>9} {:>11} {:>10} {:>10}",
        "k", "N", "rate(b)", "compress", "mse/coord", "vec_cos"
    );
    let m = 1024usize; // vectors
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);
    for &(k, n) in rates {
        let codec = FibQuantCodec::new(d, k, n, SEED);
        let mut mse_sum = 0.0f64;
        let mut cos_sum = 0.0f64;
        for _ in 0..m {
            let x: Vec<f64> = (0..d).map(|_| rng.sample(StandardNormal)).collect();
            // Codec normalizes internally, so any input distribution exercises
            // the spherical-Beta source (FibQuant universality, Theorem 1).
            let enc = codec.encode_vec(&x);
            let xh = codec.decode_vec(&enc);
            let e: f64 = x.iter().zip(xh.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            mse_sum += e / d as f64;
            let mut no = 0.0;
            let mut nr = 0.0;
            let mut dot = 0.0;
            for j in 0..d {
                no += x[j] * x[j];
                nr += xh[j] * xh[j];
                dot += x[j] * xh[j];
            }
            no = no.sqrt();
            nr = nr.sqrt();
            if no > 1e-20 && nr > 1e-20 {
                cos_sum += dot / (no * nr);
            }
        }
        println!(
            "{:>4} {:>5} {:>9.3} {:>10.1}× {:>10.2e} {:>10.4}",
            k,
            n,
            codec.rate_bits(),
            codec.compression_vs_fp16(),
            mse_sum / m as f64,
            cos_sum / m as f64
        );
    }
    println!("\nExpect: mse/coord falls as rate rises; vec_cos → 1.0 at high rate.");
}

fn run_real_kv(path: &str) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("failed to read {path}: {e}");
        std::process::exit(1);
    });
    let (d, nkv, nq, t, kf, vf, qf) = parse_fkv1(&bytes);
    println!("== FibQuant real-KV fidelity ({path}) ==");
    println!("d={d} nkv={nkv} nq={nq} T={t} (rate sweep over K and V)");
    let rates: &[(usize, usize)] = &[(2, 16), (2, 64), (2, 256), (4, 256), (4, 1024)];
    println!(
        "{:>4} {:>6} {:>9} {:>11} {:>10} {:>12}",
        "k", "N", "rate(b)", "compress", "k_vec_cos", "attn_cos"
    );
    for &(k, n) in rates {
        let codec = build_codec(d, k, n);
        let kh = codec.decode_tensor(&codec.encode_tensor(&kf));
        let vh = codec.decode_tensor(&codec.encode_tensor(&vf));
        let k_cos = mean_vector_cosine(&kf, &kh, d);
        let attn = attention_output_cosine(&kf, &vf, &kh, &vh, &qf, t, nkv, nq, d);
        println!(
            "{:>4} {:>6} {:>9.3} {:>10.1}× {:>10.4} {:>12.4}",
            k,
            n,
            codec.rate_bits(),
            codec.compression_vs_fp16(),
            k_cos,
            attn
        );
    }
    println!("\nSuccess target: attn_cos ≥ 0.95 at the chosen 4×/8× rate.");
}

/// Parse `FKV1` → `(d, nkv, nq, T, k[T*nkv*d], v[..], q[nq*d])` as `f64`.
fn parse_fkv1(b: &[u8]) -> (usize, usize, usize, usize, Vec<f64>, Vec<f64>, Vec<f64>) {
    assert!(b.len() >= 20 && &b[0..4] == b"FKV1", "bad FKV1 magic");
    let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
    let d = u32_at(4);
    let nkv = u32_at(8);
    let nq = u32_at(12);
    let t = u32_at(16);
    let kv_count = t * nkv * d;
    let q_count = nq * d;
    let bf16_vec = |start: usize, count: usize| -> Vec<f64> {
        (0..count)
            .map(|i| {
                let o = start + i * 2;
                bf16::from_le_bytes([b[o], b[o + 1]]).to_f32() as f64
            })
            .collect()
    };
    let body = 20;
    let ksz = kv_count * 2;
    let vsz = kv_count * 2;
    let k = bf16_vec(body, kv_count);
    let v = bf16_vec(body + ksz, kv_count);
    let q = bf16_vec(body + ksz + vsz, q_count);
    (d, nkv, nq, t, k, v, q)
}
