// SPDX-License-Identifier: AGPL-3.0-only
//! GATE B for the chunked-scan GDN prototype (the #1 prefill lever).
//!
//! Launches `gated_delta_rule_prefill_chunk64` (the FFMA C=64 prototype) on the
//! GPU and validates it against:
//!   (1) the CPU recurrent SSOT (per-token delta-rule, matches the live kernel
//!       math — no clamp, linear gate, rsqrt(d) scale) at small + boundary
//!       seq_lens, and
//!   (2) the LIVE `gated_delta_rule_prefill_persistent_wy4` kernel directly at
//!       the wy32 crash length 27731 (kernel-vs-kernel, fast — bounds-safety +
//!       correctness without the slow CPU ref). This is also GATE A part 2.
//!
//! Run: cargo run -p spark-model --release --example gdn_chunk64_gateb
//! (needs a GB10 GPU; ~30s incl. the 27731 launch.)

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const HEAD_REPEAT: usize = NV / NK;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.f() }
}

struct Inp {
    q: Vec<bf16>, // [t][nk][kd]
    k: Vec<bf16>,
    v: Vec<bf16>,    // [t][nv][vd]
    gate: Vec<f32>,  // [t][nv]
    beta: Vec<f32>,
    h0: Vec<f32>, // [nv][kd][vd]
}
fn gen_inp(t: usize, seed: u64) -> Inp {
    let mut r = Lcg(seed);
    Inp {
        q: (0..t * NK * KD).map(|_| bf16::from_f64(r.r(-0.5, 0.5))).collect(),
        k: (0..t * NK * KD).map(|_| bf16::from_f64(r.r(-0.5, 0.5))).collect(),
        v: (0..t * NV * VD).map(|_| bf16::from_f64(r.r(-0.5, 0.5))).collect(),
        gate: (0..t * NV).map(|_| r.r(0.80, 0.999) as f32).collect(),
        beta: (0..t * NV).map(|_| r.r(0.0, 1.0) as f32).collect(),
        h0: (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect(),
    }
}

/// Per-token recurrent SSOT (matches the live prefill kernel: no clamp).
fn recurrent_ref(inp: &Inp, t: usize) -> Vec<bf16> {
    let scale = (KD as f32).powf(-0.5);
    let mut h = inp.h0.clone();
    let mut out = vec![bf16::ZERO; t * NV * VD];
    for vh in 0..NV {
        let kh = vh / HEAD_REPEAT;
        for ti in 0..t {
            let g = inp.gate[ti * NV + vh];
            let bt = inp.beta[ti * NV + vh];
            for tid in 0..VD {
                let mut hk = 0.0f32;
                for j in 0..KD {
                    hk += h[(vh * KD + j) * VD + tid] * inp.k[(ti * NK + kh) * KD + j].to_f32();
                }
                let vnew = (inp.v[(ti * NV + vh) * VD + tid].to_f32() - g * hk) * bt;
                let mut qd = 0.0f32;
                for j in 0..KD {
                    let idx = (vh * KD + j) * VD + tid;
                    let hn = g * h[idx] + inp.k[(ti * NK + kh) * KD + j].to_f32() * vnew;
                    h[idx] = hn;
                    qd += hn * inp.q[(ti * NK + kh) * KD + j].to_f32();
                }
                out[(ti * NV + vh) * VD + tid] = bf16::from_f32(qd * scale);
            }
        }
    }
    out
}

fn up_bf16(gpu: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}
fn up_f32(gpu: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}
fn down_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<bf16>> {
    let mut bytes = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut bytes)?;
    Ok(bytes.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect())
}

#[allow(clippy::too_many_arguments)]
fn launch_gdn(
    gpu: &dyn GpuBackend, kernel: spark_runtime::gpu::KernelHandle, smem: u32, t: u32,
    h: DevicePtr, q: DevicePtr, k: DevicePtr, v: DevicePtr, g: DevicePtr, b: DevicePtr, o: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([NV as u32, 1, 1]).block([128, 1, 1]).shared_mem(smem)
        .arg_ptr(h).arg_ptr(q).arg_ptr(k).arg_ptr(v).arg_ptr(g).arg_ptr(b).arg_ptr(o)
        .arg_u32(1).arg_u32(t).arg_u32(NK as u32).arg_u32(NV as u32)
        .arg_u32(KD as u32).arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32).arg_u32((NV * VD) as u32).arg_u32(NV as u32)
        .launch(0)
}

fn compare(a: &[bf16], b: &[bf16]) -> (f32, f64) {
    let (mut maxd, mut dot, mut na, mut nb) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (xf, yf) = (x.to_f32(), y.to_f32());
        maxd = maxd.max((xf - yf).abs());
        dot += (xf * yf) as f64; na += (xf * xf) as f64; nb += (yf * yf) as f64;
    }
    (maxd, dot / (na.sqrt() * nb.sqrt() + 1e-12))
}

fn run_chunk64(gpu: &dyn GpuBackend, kc: spark_runtime::gpu::KernelHandle, inp: &Inp, t: usize) -> Result<Vec<bf16>> {
    let (q, k, v) = (up_bf16(gpu, &inp.q)?, up_bf16(gpu, &inp.k)?, up_bf16(gpu, &inp.v)?);
    let (g, b, h) = (up_f32(gpu, &inp.gate)?, up_f32(gpu, &inp.beta)?, up_f32(gpu, &inp.h0)?);
    let o = gpu.alloc(t * NV * VD * 2)?;
    launch_gdn(gpu, kc, 98560, t as u32, h, q, k, v, g, b, o)?;
    gpu.synchronize(0)?;
    let out = down_bf16(gpu, o, t * NV * VD)?;
    for p in [q, k, v, g, b, h, o] { let _ = gpu.free(p); }
    Ok(out)
}
fn run_wy4(gpu: &dyn GpuBackend, kw: spark_runtime::gpu::KernelHandle, inp: &Inp, t: usize) -> Result<Vec<bf16>> {
    let (q, k, v) = (up_bf16(gpu, &inp.q)?, up_bf16(gpu, &inp.k)?, up_bf16(gpu, &inp.v)?);
    let (g, b, h) = (up_f32(gpu, &inp.gate)?, up_f32(gpu, &inp.beta)?, up_f32(gpu, &inp.h0)?);
    let o = gpu.alloc(t * NV * VD * 2)?;
    launch_gdn(gpu, kw, 69688, t as u32, h, q, k, v, g, b, o)?;
    gpu.synchronize(0)?;
    let out = down_bf16(gpu, o, t * NV * VD)?;
    for p in [q, k, v, g, b, h, o] { let _ = gpu.free(p); }
    Ok(out)
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_config("qwen3_6_moe", 2048).expect("no ptx set");
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let kc = gpu.kernel("gated_delta_rule_chunk64", "gated_delta_rule_prefill_chunk64")?;
    let kw = gpu.kernel("gated_delta_rule_persistent", "gated_delta_rule_prefill_persistent_wy4")?;
    eprintln!("chunk64 handle={} wy4 handle={}", kc.0, kw.0);

    let mut all_ok = true;
    // (1) vs CPU recurrent SSOT + vs wy4, at small/boundary seq_lens.
    for &t in &[64usize, 98, 100, 128, 200] {
        let inp = gen_inp(t, 0xABCD ^ t as u64);
        let refr = recurrent_ref(&inp, t);
        let c = run_chunk64(gpu, kc, &inp, t)?;
        let w = run_wy4(gpu, kw, &inp, t)?;
        let (md_cr, cos_cr) = compare(&c, &refr);
        let (md_cw, cos_cw) = compare(&c, &w);
        let (md_wr, cos_wr) = compare(&w, &refr);
        let ok = cos_cr >= 0.9999 && md_cr < 0.02 && cos_cw >= 0.9999;
        all_ok &= ok;
        eprintln!("t={t:5}  chunk64-vs-ref: max={md_cr:.4} cos={cos_cr:.6} | chunk64-vs-wy4: max={md_cw:.4} cos={cos_cw:.6} | wy4-vs-ref: max={md_wr:.4} cos={cos_wr:.6}  {}",
            if ok { "PASS" } else { "FAIL" });
    }
    // (2) wy32 crash length: chunk64 vs wy4 directly (bounds-safety + correctness; no slow CPU ref).
    for &t in &[27731usize, 27732] {
        let inp = gen_inp(t, 0xBEEF ^ t as u64);
        let c = run_chunk64(gpu, kc, &inp, t)?;
        let w = run_wy4(gpu, kw, &inp, t)?;
        let finite = c.iter().all(|x| x.to_f32().is_finite());
        let (md, cos) = compare(&c, &w);
        // bf16-H drift grows max-abs at long ctx (precision under GATE-C test);
        // GATE B here = STRUCTURAL correctness (cos + finite + bounds-safe).
        let ok = finite && cos >= 0.999;
        all_ok &= ok;
        eprintln!("t={t:5}  chunk64-vs-wy4: max={md:.4} cos={cos:.6} finite={finite}  {} (wy32-len; max=bf16-H, GATE-C decides)",
            if ok { "PASS" } else { "FAIL" });
    }
    eprintln!("\nGATE B (structural): {}", if all_ok { "PASS ✅" } else { "FAIL ❌" });

    // ── SPEED: time chunk64 (TC) vs wy4 at a long context ──
    let tt = 16384usize;
    let inp = gen_inp(tt, 0x5151);
    let timeit = |kern: spark_runtime::gpu::KernelHandle, smem: u32| -> Result<f64> {
        let (q, k, v) = (up_bf16(gpu, &inp.q)?, up_bf16(gpu, &inp.k)?, up_bf16(gpu, &inp.v)?);
        let (g, bb, h) = (up_f32(gpu, &inp.gate)?, up_f32(gpu, &inp.beta)?, up_f32(gpu, &inp.h0)?);
        let o = gpu.alloc(tt * NV * VD * 2)?;
        launch_gdn(gpu, kern, smem, tt as u32, h, q, k, v, g, bb, o)?; gpu.synchronize(0)?; // warmup
        let t0 = std::time::Instant::now();
        let iters = 5;
        for _ in 0..iters { launch_gdn(gpu, kern, smem, tt as u32, h, q, k, v, g, bb, o)?; }
        gpu.synchronize(0)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        for p in [q, k, v, g, bb, h, o] { let _ = gpu.free(p); }
        Ok(ms)
    };
    let t_c = timeit(kc, 98560)?;
    let t_w = timeit(kw, 69688)?;
    eprintln!("SPEED @t={tt}: chunk64(TC)={t_c:.2}ms  wy4={t_w:.2}ms  speedup={:.2}x", t_w / t_c);

    if !all_ok { std::process::exit(1); }
    Ok(())
}
