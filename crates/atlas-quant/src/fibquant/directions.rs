// SPDX-License-Identifier: AGPL-3.0-only

//! Quasi-uniform direction sets on `S^{k-1}` for the FibQuant codebook init.
//!
//! Geometry only — no calibration. The angular component of the spherical-Beta
//! source is uniform, so the direction init is a quasi-uniform spherical
//! packing; Lloyd–Max (`codebook.rs`) then polishes it to the finite-`N` source.
//!
//! - `k = 2`: planar Fibonacci spiral (golden-angle sunflower).
//! - `k = 3`: Fibonacci sphere (equal-area latitude bands + golden azimuth).
//! - `k ≥ 4`: Roberts–Kronecker rank-one sequence mapped through `Φ⁻¹`.

use std::f64::consts::PI;

use crate::fibquant::special::erfinv;

const GOLDEN_FRAC: f64 = 0.381_966_011_250_105_1; // 1 - 1/φ

/// Build `n_dirs` unit directions on `S^{k-1}`, returned row-major (`n_dirs*k`).
pub(crate) fn direction_set(k: usize, n_dirs: usize) -> Vec<f64> {
    match k {
        2 => fibonacci_spiral(n_dirs),
        3 => fibonacci_sphere(n_dirs),
        _ => roberts_kronecker(k, n_dirs),
    }
}

/// k=2: golden-angle sunflower on the circle.
fn fibonacci_spiral(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * 2];
    for i in 0..n {
        let theta = 2.0 * PI * GOLDEN_FRAC * (i as f64);
        out[i * 2] = theta.cos();
        out[i * 2 + 1] = theta.sin();
    }
    out
}

/// k=3: Fibonacci sphere — equal-area latitude bands, golden-angle azimuth.
/// Points are already unit length (latitude radius × azimuth + y).
fn fibonacci_sphere(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * 3];
    let golden = 2.0 * PI * GOLDEN_FRAC;
    for i in 0..n {
        let y = 1.0 - (i as f64 + 0.5) * 2.0 / (n as f64);
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * (i as f64);
        out[i * 3] = r * theta.cos();
        out[i * 3 + 1] = y;
        out[i * 3 + 2] = r * theta.sin();
    }
    out
}

/// k≥4: Roberts–Kronecker rank-one sequence `{(n+½) φ_k^{-j}}` through `Φ⁻¹`,
/// normalized onto `S^{k-1}`. `φ_k` is the positive root of `x^{k+1} = x + 1`.
fn roberts_kronecker(k: usize, n: usize) -> Vec<f64> {
    let phi_k = plastic_root(k as f64);
    let mut out = vec![0.0; n * k];
    for i in 0..n {
        let mut v = vec![0.0; k];
        for j in 1..=k {
            let xi = (((i as f64) + 0.5) * phi_k.powi(-(j as i32))).fract();
            // `Φ⁻¹(ξ) = √2 · erfinv(2ξ − 1)`.
            v[j - 1] = (2.0_f64).sqrt() * erfinv(2.0 * xi - 1.0);
        }
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nrm = if nrm > 1e-12 { nrm } else { 1.0 };
        for j in 0..k {
            out[i * k + j] = v[j] / nrm;
        }
    }
    out
}

/// Positive real root of `x^{k+1} − x − 1 = 0` (generalized golden ratio).
/// Newton from 1.5; converges in <15 steps for all k ≥ 1.
fn plastic_root(k: f64) -> f64 {
    let mut x = 1.5_f64;
    for _ in 0..50 {
        let f = x.powf(k + 1.0) - x - 1.0;
        let fp = (k + 1.0) * x.powf(k) - 1.0;
        if fp.abs() < 1e-18 {
            break;
        }
        let nx = x - f / fp;
        if (nx - x).abs() < 1e-15 {
            return nx;
        }
        x = nx;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spiral_and_sphere_are_unit() {
        let s2 = fibonacci_spiral(64);
        for i in 0..64 {
            let nrm = (s2[i * 2] * s2[i * 2] + s2[i * 2 + 1] * s2[i * 2 + 1]).sqrt();
            assert!((nrm - 1.0).abs() < 1e-12);
        }
        let s3 = fibonacci_sphere(64);
        for i in 0..64 {
            let nrm = (s3[i * 3].powi(2) + s3[i * 3 + 1].powi(2) + s3[i * 3 + 2].powi(2)).sqrt();
            assert!((nrm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn roberts_kronecker_is_unit() {
        let rk = roberts_kronecker(4, 64);
        for i in 0..64 {
            let nrm = (0..4).map(|j| rk[i * 4 + j].powi(2)).sum::<f64>().sqrt();
            assert!((nrm - 1.0).abs() < 1e-9, "rk dir {i} norm {nrm}");
        }
    }

    #[test]
    fn plastic_root_checks() {
        // k=1 ⇒ golden ratio φ.
        assert!((plastic_root(1.0) - (1.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-9);
        // satisfies x^{k+1} = x + 1.
        for k in [2.0_f64, 3.0, 4.0] {
            let x = plastic_root(k);
            assert!((x.powf(k + 1.0) - x - 1.0).abs() < 1e-9);
        }
    }
}
