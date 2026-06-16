// SPDX-License-Identifier: AGPL-3.0-only

//! The shared Haar-random orthogonal rotation `Π`.
//!
//! `Π` is generated **once** from a fixed seed and reused across every layer,
//! head, prompt and token (FibQuant's source-agnostic universality, Theorem 1
//! of arXiv:2605.11478). Orthogonality is what makes the fused kernel feasible:
//! `Q·K = (ΠQ)·(ΠK)`, so `K` is stored compressed in rotated space, `Q` is
//! rotated once per query, and `K` decompression needs no inverse rotation.
//!
//! Construction: fill a `d×d` matrix with iid standard normals (deterministic
//! `ChaCha8` seed), then orthonormalize via modified Gram–Schmidt. The result
//! is a Haar-uniform orthogonal matrix. `f64` throughout (offline construction
//! + fidelity spike); the eventual CUDA constant is emitted from the same seed.

use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// A `d×d` orthogonal matrix, stored row-major.
#[derive(Clone)]
pub struct Rotation {
    /// Number of rows / columns.
    pub d: usize,
    /// Row-major `d*d` entries.
    pub mat: Vec<f64>,
}

impl Rotation {
    /// Build the normalized Walsh–Hadamard transform (Sylvester construction),
    /// `H_d / √d`. Orthogonal, deterministic, no storage needed on GPU (Atlas's
    /// `wht_bf16` kernel implements the same transform via the butterfly). Used
    /// to test whether a FibQuant codebook holds fidelity under WHT instead of a
    /// dense Haar matrix — WHT would let the kernel reuse the existing WHT
    /// infrastructure rather than upload a d×d (up to 256 KB) rotation buffer.
    pub fn hadamard(d: usize) -> Self {
        assert!(d.is_power_of_two(), "Hadamard needs a power-of-two d");
        let mut h = vec![0.0f64; d * d];
        h[0] = 1.0; // H_1
        let mut size = 1;
        while size < d {
            // H_{2n} = [[H_n, H_n], [H_n, -H_n]], reading the top-left H_n.
            for r in 0..size {
                for c in 0..size {
                    let v = h[r * d + c];
                    h[r * d + (size + c)] = v;
                    h[(size + r) * d + c] = v;
                    h[(size + r) * d + (size + c)] = -v;
                }
            }
            size *= 2;
        }
        let inv_sqrt = 1.0 / (d as f64).sqrt();
        for v in h.iter_mut() {
            *v *= inv_sqrt;
        }
        Self { d, mat: h }
    }

    /// Build a deterministic Haar-random orthogonal matrix from `seed`.
    /// Same seed ⟹ identical matrix everywhere (spike, host, CUDA constant).
    pub fn from_seed(d: usize, seed: u64) -> Self {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut a = vec![0.0f64; d * d];
        for v in a.iter_mut() {
            *v = rng.sample(StandardNormal);
        }
        // Modified Gram–Schmidt orthonormalization of the columns.
        for j in 0..d {
            // Normalize column j against already-fixed columns 0..j.
            for i in 0..j {
                let dot = col_dot(&a, d, j, i);
                col_axpy(&mut a, d, j, i, -dot);
            }
            let nrm = col_norm(&a, d, j);
            // A fresh Gaussian column is non-zero with probability 1; guard the
            // degenerate case by re-drawing until it has positive norm.
            let nrm = if nrm > 1e-12 {
                nrm
            } else {
                'redraw: loop {
                    for r in 0..d {
                        a[r * d + j] = rng.sample(StandardNormal);
                    }
                    let n2 = col_norm(&a, d, j);
                    if n2 > 1e-12 {
                        break 'redraw n2;
                    }
                }
            };
            let inv = 1.0 / nrm;
            for r in 0..d {
                a[r * d + j] *= inv;
            }
        }
        Self { d, mat: a }
    }

    /// Apply `Π` to a unit vector `u` (`len == d`), writing into `out` (`len == d`).
    /// `out = Π u` where `Π` is row-major → `out[i] = Σ_j mat[i*d+j] * u[j]`.
    pub fn apply(&self, u: &[f64], out: &mut [f64]) {
        debug_assert_eq!(u.len(), self.d);
        debug_assert_eq!(out.len(), self.d);
        let d = self.d;
        for (i, out_i) in out.iter_mut().enumerate() {
            let row = &self.mat[i * d..i * d + d];
            let mut s = 0.0;
            for (&uj, &mj) in u.iter().zip(row.iter()) {
                s += uj * mj;
            }
            *out_i = s;
        }
    }

    /// Apply `Πᵀ` (transpose) — used by the host-side reference decoder.
    /// `out[i] = Σ_j mat[j*d+i] * u[j]`, accumulated column-major to avoid index math.
    pub fn apply_transpose(&self, u: &[f64], out: &mut [f64]) {
        debug_assert_eq!(u.len(), self.d);
        debug_assert_eq!(out.len(), self.d);
        let d = self.d;
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for (j, &uj) in u.iter().enumerate() {
            if uj == 0.0 {
                continue;
            }
            let row_j = &self.mat[j * d..j * d + d];
            for (o, &mj) in out.iter_mut().zip(row_j.iter()) {
                *o += uj * mj;
            }
        }
    }
}

fn col_dot(a: &[f64], d: usize, j: usize, i: usize) -> f64 {
    let mut s = 0.0;
    for r in 0..d {
        s += a[r * d + j] * a[r * d + i];
    }
    s
}

fn col_norm(a: &[f64], d: usize, j: usize) -> f64 {
    let mut s = 0.0;
    for r in 0..d {
        let v = a[r * d + j];
        s += v * v;
    }
    s.sqrt()
}

fn col_axpy(a: &mut [f64], d: usize, j: usize, i: usize, alpha: f64) {
    for r in 0..d {
        a[r * d + j] += alpha * a[r * d + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_orthogonal() {
        let rot = Rotation::from_seed(64, 0x5eed);
        let d = rot.d;
        // ΠᵀΠ = I: check a few columns are orthonormal.
        for j in 0..d {
            let n = col_norm(&rot.mat, d, j);
            assert!((n - 1.0).abs() < 1e-9, "col {j} norm {n}");
        }
        for j in 0..d {
            for i in (j + 1)..d {
                let dot = col_dot(&rot.mat, d, j, i);
                assert!(dot.abs() < 1e-9, "cols {j},{i} dot {dot}");
            }
        }
        // apply then apply_transpose should be identity.
        let u: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0).sin()).collect();
        let nu: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        let unit: Vec<f64> = u.iter().map(|v| v / nu).collect();
        let mut rot_u = vec![0.0; d];
        rot.apply(&unit, &mut rot_u);
        let mut back = vec![0.0; d];
        rot.apply_transpose(&rot_u, &mut back);
        for i in 0..d {
            assert!((back[i] - unit[i]).abs() < 1e-9, "ΠᵀΠ u != u at {i}");
        }
    }
}
