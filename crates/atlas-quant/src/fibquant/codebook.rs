// SPDX-License-Identifier: AGPL-3.0-only

//! Radial–angular codebook construction for the spherical-Beta source
//! `f_{d,k}` (FibQuant, arXiv:2605.11478 §3.3 + Appendix A).
//!
//! A codeword is `c_n = r_n · u_n`: a Beta-quantile radius paired with a
//! quasi-uniform direction, then refined by multi-restart Lloyd–Max on samples
//! from `f_{d,k}`. The codebook is a **precomputed constant** (no calibration)
//! reused across every layer/head/prompt/token — the universality of Theorem 1.

use rand::Rng;
use rand_distr::StandardNormal;

use crate::fibquant::directions::direction_set;
use crate::fibquant::special::inverse_beta_cdf;

/// Shared offline codebook: `N` codewords of dimension `k`, row-major `N*k`.
#[derive(Clone)]
pub struct Codebook {
    pub k: usize,
    pub n: usize,
    pub words: Vec<f64>,
}

impl Codebook {
    /// Nearest codeword to `block` (squared distance, `len == k`).
    /// Brute force — fine offline and in the fidelity spike (encode cost is
    /// paid at cache-fill, not decode).
    pub fn nearest(&self, block: &[f64]) -> (usize, f64) {
        debug_assert_eq!(block.len(), self.k);
        let mut best = 0usize;
        let mut best_d = f64::MAX;
        for n in 0..self.n {
            let row = &self.words[n * self.k..n * self.k + self.k];
            let mut d = 0.0;
            for j in 0..self.k {
                let diff = row[j] - block[j];
                d += diff * diff;
            }
            if d < best_d {
                best_d = d;
                best = n;
            }
        }
        (best, best_d)
    }
}

/// Beta-quantile radii (Eq. 5): `r_n = √BetaInv(q_n; k/2, β_{d,k})` with
/// midpoint quantiles `q_n = (n+½)/N`. For `k=2` this reproduces the closed
/// form `√(1−(1−q)^{4/d})` (Eq. 6) — checked in tests.
pub(crate) fn radii(d: usize, k: usize, n: usize) -> Vec<f64> {
    let a = k as f64 / 2.0;
    let beta = (k as f64 / (k as f64 + 2.0)) * ((d - k - 2) as f64 / 2.0) + 1.0;
    let mut out = vec![0.0; n];
    for (i, o) in out.iter_mut().enumerate() {
        let q = (i as f64 + 0.5) / (n as f64);
        *o = inverse_beta_cdf(a, beta, q).sqrt();
    }
    out
}

/// Draw one sample from the spherical-Beta marginal `f_{d,k}`: a uniform point
/// on `S^{d-1}` (Gaussian → normalize), take its first `k` coordinates.
pub(crate) fn sample_spherical_beta(d: usize, k: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut g = vec![0.0f64; d];
    for v in g.iter_mut() {
        *v = rng.sample(StandardNormal);
    }
    let nrm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nrm = if nrm > 1e-12 { nrm } else { 1.0 };
    let mut s = vec![0.0; k];
    for j in 0..k {
        s[j] = g[j] / nrm;
    }
    s
}

/// Random `k×k` Haar orthogonal via Gram–Schmidt (small `k`).
fn haar_k(k: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut m = vec![0.0f64; k * k];
    for v in m.iter_mut() {
        *v = rng.sample(StandardNormal);
    }
    for c in 0..k {
        for p in 0..c {
            let mut dot = 0.0;
            for r in 0..k {
                dot += m[r * k + c] * m[r * k + p];
            }
            for r in 0..k {
                m[r * k + c] -= dot * m[r * k + p];
            }
        }
        let mut nrm = 0.0;
        for r in 0..k {
            nrm += m[r * k + c] * m[r * k + c];
        }
        nrm = nrm.sqrt();
        let inv = if nrm > 1e-12 { 1.0 / nrm } else { 1.0 };
        for r in 0..k {
            m[r * k + c] *= inv;
        }
    }
    m
}

/// Build the codebook: `radii × directions` init, then multi-restart Lloyd–Max
/// on `f_{d,k}` samples (Appendix A, Table 2: `M=30N`, `R=4`, `T=25`).
pub fn build(d: usize, k: usize, n: usize, rng: &mut impl Rng) -> Codebook {
    let rad = radii(d, k, n);
    let dirs = direction_set(k, n);
    // Deterministic init: c_i = r_i · u_i.
    let mut init = vec![0.0; n * k];
    for i in 0..n {
        for j in 0..k {
            init[i * k + j] = rad[i] * dirs[i * k + j];
        }
    }

    let m = 30 * n;
    let samples: Vec<Vec<f64>> = (0..m).map(|_| sample_spherical_beta(d, k, rng)).collect();

    const RESTARTS: usize = 4;
    const ITERS: usize = 25;
    let mut best = init.clone();
    let mut best_mse = f64::MAX;
    for _ in 0..RESTARTS {
        let q = haar_k(k, rng);
        let mut words = init.clone();
        // Rotate init codebook by Q.
        for i in 0..n {
            let mut rotated = vec![0.0; k];
            for r in 0..k {
                let mut s = 0.0;
                for c in 0..k {
                    s += q[r * k + c] * words[i * k + c];
                }
                rotated[r] = s;
            }
            words[i * k..i * k + k].copy_from_slice(&rotated);
        }
        let mut mse = f64::MAX;
        for _it in 0..ITERS {
            let mut sums = vec![0.0; n * k];
            let mut counts = vec![0u32; n];
            let mut total = 0.0f64;
            let cb = Codebook {
                k,
                n,
                words: words.clone(),
            };
            for s in &samples {
                let (idx, dist) = cb.nearest(s);
                total += dist;
                counts[idx] += 1;
                for j in 0..k {
                    sums[idx * k + j] += s[j];
                }
            }
            // Centroid update + empty-cell repair (split the busiest cell).
            for i in 0..n {
                if counts[i] > 0 {
                    let c = counts[i] as f64;
                    for j in 0..k {
                        words[i * k + j] = sums[i * k + j] / c;
                    }
                }
            }
            for i in 0..n {
                if counts[i] == 0 {
                    // Find busiest cell and split it.
                    let busy = counts
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, c)| c)
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    for j in 0..k {
                        let g: f64 = rng.sample(StandardNormal);
                        words[i * k + j] = words[busy * k + j] + g * 1e-3;
                    }
                }
            }
            mse = total / (m as f64);
        }
        if mse < best_mse {
            best_mse = mse;
            best = words;
        }
    }
    Codebook { k, n, words: best }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn k2_radii_match_closed_form() {
        // Eq. 6: r = √(1−(1−q)^{4/d}).
        let d = 64;
        let r = radii(d, 2, 8);
        for (i, &ri) in r.iter().enumerate() {
            let q = (i as f64 + 0.5) / 8.0;
            let expect = (1.0 - (1.0 - q).powf(4.0 / d as f64)).sqrt();
            assert!((ri - expect).abs() < 1e-9, "radius {i} {ri} vs {expect}");
        }
    }

    #[test]
    fn codebook_mse_decreases_with_n() {
        let d = 64;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let cb_small = build(d, 2, 16, &mut rng);
        let cb_big = build(d, 2, 256, &mut rng);
        let mut rng2 = ChaCha8Rng::seed_from_u64(99);
        let mut mse_s = 0.0;
        let mut mse_b = 0.0;
        for _ in 0..2000 {
            let s = sample_spherical_beta(d, 2, &mut rng2);
            mse_s += cb_small.nearest(&s).1;
            mse_b += cb_big.nearest(&s).1;
        }
        assert!(mse_b < mse_s, "bigger codebook must beat smaller");
    }
}
