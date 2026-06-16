// SPDX-License-Identifier: AGPL-3.0-only

//! Special functions used to build the FibQuant radial–angular codebook.
//!
//! These are pure-Rust, dependency-free reimplementations of the standard
//! numerical-recipes formulas (Lanczos `ln Γ`, regularized incomplete beta
//! `I_x(a,b)` + its inverse, and `erf`/`erfinv`). They are only used *offline*
//! to precompute the codebook (and in the fidelity spike); they never run on
//! the decode path.
//!
//! All math is done in `f64`. The codebook is the canonical object induced by
//! the random-access normalize–rotate interface (spherical-Beta source,
//! `f_{d,k}`); see `docs/design/fibquant-kv-compression.md` and
//! <https://arxiv.org/abs/2605.11478>.

use std::f64::consts::PI;

/// Natural log of the Gamma function (Lanczos, g=7, n=9).
/// Accurate to ~1e-15 for x > 0. Uses the reflection formula for x < 0.5.
pub(crate) fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1-x) = π / sin(πx) ⇒ lnΓ(x) = lnπ − ln(sin(πx)·Γ(1-x)).
        PI.ln() - ((PI * x).sin() * (ln_gamma(1.0 - x)).exp()).ln()
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &ci) in C.iter().enumerate().skip(1) {
            a += ci / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Continued fraction for the incomplete beta function (Lentz's method).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const FP_MIN: f64 = 1e-300;
    const EPS: f64 = 1e-15;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FP_MIN {
        d = FP_MIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200 {
        let m2 = 2 * m;
        let aa = m as f64 * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FP_MIN {
            d = FP_MIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FP_MIN {
            c = FP_MIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FP_MIN {
            d = FP_MIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FP_MIN {
            c = FP_MIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a,b)` in `0..=1`.
pub(crate) fn beta_i(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lnbeta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let bt = (lnbeta + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Inverse of the regularized incomplete beta: find `x` with `I_x(a,b) = p`.
/// Monotone bisection on `[0,1]` — robust, ~100 iterations for full f64.
/// Only called at offline codebook construction (cheap, N times).
pub(crate) fn inverse_beta_cdf(a: f64, b: f64, p: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&p), "beta quantile out of range");
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    let mut x = 0.5;
    for _ in 0..100 {
        let fx = beta_i(a, b, x);
        if fx < p {
            lo = x;
        } else {
            hi = x;
        }
        let mid = 0.5 * (lo + hi);
        if (mid - x).abs() < 1e-15 || lo == hi {
            return mid;
        }
        x = mid;
    }
    x
}

/// Complementary error function (Chebyshev approximation, Numerical Recipes).
pub(crate) fn erfc(x: f64) -> f64 {
    const NCOEFF: [f64; 10] = [
        -1.265_512_23,
        1.000_023_68,
        0.374_091_96,
        0.096_784_18,
        -0.186_288_06,
        0.278_868_07,
        -1.135_203_98,
        1.488_515_87,
        -0.822_152_23,
        0.170_872_77,
    ];
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let mut poly = NCOEFF[NCOEFF.len() - 1];
    for &c in NCOEFF.iter().rev().skip(1) {
        poly = poly * t + c;
    }
    // `poly` already includes the −1.26551223 constant term, so the exponent is
    // `−z² + poly` (Numerical Recipes erfc nested form).
    let ans = t * (-z * z + poly).exp();
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// Error function.
pub(crate) fn erf(x: f64) -> f64 {
    1.0 - erfc(x)
}

/// Inverse error function, via monotone bisection on `erf`.
/// `erf` is strictly increasing with `erf(±8) ≈ ±(1 − 1e-29)`, so `[-10, 10]`
/// brackets any `p ∈ (-1, 1)`. Only called at offline codebook construction
/// (Roberts–Kronecker direction mapping); ~60 `erf` evals per call is cheap.
pub(crate) fn erfinv(p: f64) -> f64 {
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if p == 0.0 {
        return 0.0;
    }
    let mut lo = -10.0f64;
    let mut hi = 10.0f64;
    let mut x = 0.0;
    for _ in 0..80 {
        let fx = erf(x);
        if fx < p {
            lo = x;
        } else {
            hi = x;
        }
        let mid = 0.5 * (lo + hi);
        if (mid - x).abs() < 1e-16 || lo == hi {
            return mid;
        }
        x = mid;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_known_values() {
        assert!((ln_gamma(1.0)).abs() < 1e-12); // Γ(1)=1
        assert!((ln_gamma(2.0)).abs() < 1e-12); // Γ(2)=1
        assert!((ln_gamma(0.5) - (PI.sqrt()).ln()).abs() < 1e-12); // Γ(1/2)=√π
        assert!((ln_gamma(10.0) - (362_880.0_f64).ln()).abs() < 1e-9); // Γ(10)=9!
    }

    #[test]
    fn beta_i_endpoints_and_symmetry() {
        assert!(beta_i(2.0, 3.0, 0.0).abs() < 1e-12);
        assert!((beta_i(2.0, 3.0, 1.0) - 1.0).abs() < 1e-12);
        // I_x(a,b) + I_{1-x}(b,a) = 1.
        for &x in &[0.1, 0.25, 0.5, 0.73] {
            let s = beta_i(2.0, 3.0, x) + beta_i(3.0, 2.0, 1.0 - x);
            assert!((s - 1.0).abs() < 1e-10, "symmetry broke at {x}: {s}");
        }
    }

    #[test]
    fn inverse_beta_round_trips() {
        for &p in &[0.05, 0.2, 0.5, 0.8, 0.95] {
            let x = inverse_beta_cdf(3.0, 5.0, p);
            let back = beta_i(3.0, 5.0, x);
            assert!(
                (back - p).abs() < 1e-9,
                "round-trip p={p} x={x} back={back}"
            );
        }
    }

    #[test]
    fn erf_and_erfinv_round_trip() {
        // NR erfc is ~1e-7 accurate, so the round-trip floor is ~1e-7 (ample for
        // the Roberts–Kronecker direction mapping, which Lloyd–Max refines).
        for &z in &[-1.2f64, -0.4, 0.37, 0.9, 1.3] {
            let p = erf(z);
            let z2 = erfinv(p);
            assert!((z - z2).abs() < 1e-6, "erf/erfinv round-trip {z} -> {z2}");
        }
        // Known value: erfinv(0.5) ≈ 0.4769362762.
        assert!((erfinv(0.5) - 0.476_936_276).abs() < 1e-6);
        assert!(erfinv(0.0).abs() < 1e-6);
    }
}
