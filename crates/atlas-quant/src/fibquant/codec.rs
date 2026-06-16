// SPDX-License-Identifier: AGPL-3.0-only

//! FibQuant encoder/decoder over a `d`-dimensional vector.
//!
//! Pipeline (arXiv:2605.11478 §3.1): store the L2 norm `ν = ‖x‖`, rotate the
//! unit vector by the shared orthogonal `Π`, split into `d/k` blocks, and store
//! the nearest codebook index per block. Decode reverses the lookup and applies
//! `Πᵀ`. The orthogonality of `Π` is what lets a fused kernel attend in rotated
//! space (`Q·K = (ΠQ)·(ΠK)`); here we run the full reference path in `f64`.

use crate::fibquant::codebook::{Codebook, build as build_codebook};
use crate::fibquant::rotate::Rotation;
use rand::SeedableRng;

/// One cached vector's compressed representation: `f64` norm (quantized to fp16
/// in production) + `d/k` codebook indices.
#[derive(Clone, Debug)]
pub struct EncodedVec {
    pub norm: f64,
    pub indices: Vec<u32>,
}

/// The full FibQuant codec: a fixed rotation + a precomputed codebook.
pub struct FibQuantCodec {
    pub d: usize,
    pub k: usize,
    pub rotation: Rotation,
    pub codebook: Codebook,
}

impl FibQuantCodec {
    /// Build a deterministic codec for `(d, k, N)` from `seed`.
    /// Same `(d,k,N,seed)` ⟹ identical rotation + codebook everywhere.
    pub fn new(d: usize, k: usize, n: usize, seed: u64) -> Self {
        assert!(d > 0 && k > 0 && k <= d, "need 0 < k <= d");
        let nblocks = d / k;
        assert!(nblocks * k == d, "k must divide d (d={d}, k={k})");
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let rotation = Rotation::from_seed(d, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5);
        let codebook = build_codebook(d, k, n, &mut rng);
        Self {
            d,
            k,
            rotation,
            codebook,
        }
    }

    /// Bits per coordinate of the payload (norm side-channel excluded): `log2(N)/k`.
    pub fn rate_bits(&self) -> f64 {
        (self.codebook.n as f64).log2() / self.k as f64
    }

    /// Compression vs fp16 payload: `16 / rate`.
    pub fn compression_vs_fp16(&self) -> f64 {
        16.0 / self.rate_bits()
    }

    /// Encode one `d`-length vector → `{norm, indices}`.
    pub fn encode_vec(&self, x: &[f64]) -> EncodedVec {
        debug_assert_eq!(x.len(), self.d);
        let mut nrm = 0.0;
        for v in x {
            nrm += v * v;
        }
        nrm = nrm.sqrt();
        if nrm <= 1e-30 {
            return EncodedVec {
                norm: 0.0,
                indices: vec![0; self.d / self.k],
            };
        }
        let inv = 1.0 / nrm;
        let unit: Vec<f64> = x.iter().map(|v| v * inv).collect();
        let mut rotated = vec![0.0; self.d];
        self.rotation.apply(&unit, &mut rotated);
        let nblocks = self.d / self.k;
        let mut indices = Vec::with_capacity(nblocks);
        for m in 0..nblocks {
            let block = &rotated[m * self.k..m * self.k + self.k];
            indices.push(self.codebook.nearest(block).0 as u32);
        }
        EncodedVec { norm: nrm, indices }
    }

    /// Decode one encoded vector → `d`-length reconstruction `x̂ = ν·Πᵀ·ŷ`.
    pub fn decode_vec(&self, enc: &EncodedVec) -> Vec<f64> {
        let nblocks = self.d / self.k;
        let mut y_hat = vec![0.0; self.d];
        for m in 0..nblocks {
            let idx = enc.indices[m] as usize;
            let row = &self.codebook.words[idx * self.k..idx * self.k + self.k];
            y_hat[m * self.k..m * self.k + self.k].copy_from_slice(row);
        }
        let mut out = vec![0.0; self.d];
        self.rotation.apply_transpose(&y_hat, &mut out);
        for v in out.iter_mut() {
            *v *= enc.norm;
        }
        out
    }

    /// Encode a `[n_vecs, d]` tensor (row-major) → per-vector norms + indices.
    pub fn encode_tensor(&self, t: &[f64]) -> Vec<EncodedVec> {
        let n_vecs = t.len() / self.d;
        debug_assert_eq!(n_vecs * self.d, t.len());
        (0..n_vecs)
            .map(|i| self.encode_vec(&t[i * self.d..i * self.d + self.d]))
            .collect()
    }

    /// Decode a batch back to `[n_vecs, d]`.
    pub fn decode_tensor(&self, encs: &[EncodedVec]) -> Vec<f64> {
        let mut out = vec![0.0; encs.len() * self.d];
        for (i, e) in encs.iter().enumerate() {
            let row = self.decode_vec(e);
            out[i * self.d..i * self.d + self.d].copy_from_slice(&row);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_recovers_unit_vector_at_high_rate() {
        let codec = FibQuantCodec::new(64, 2, 256, 42); // b=4 (4×)
        let x: Vec<f64> = (0..64).map(|i| (i as f64).sin() * 0.7).collect();
        let enc = codec.encode_vec(&x);
        let x_hat = codec.decode_vec(&enc);
        let nrm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let err = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        // At 4× the relative error should be small.
        assert!(err / nrm < 0.2, "rel err {} too large", err / nrm);
    }
}
