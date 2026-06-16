// SPDX-License-Identifier: AGPL-3.0-only

//! Fidelity metrics for the FibQuant spike — the downstream loss the paper
//! reports (Eq. 3): attention-output cosine similarity between the reference
//! `softmax(QKᵀ/√d)V` and the same attention run over reconstructed `K̂, V̂`.
//!
//! Layouts (row-major `f64`):
//! - `k`, `v`: `[T, n_kv_heads, d]`.
//! - `q`: `[n_q_heads, d]` (one query token; GQA groups `nq/nkv` query heads
//!   per KV head).
//!
//! This mirrors the host-side attention math in `kvflash_pager.rs`
//! (`attention_block_weights`): per query head, attend to that head's KV group.

/// Softmax over a slice, in place, with the standard max-subtraction trick.
fn softmax_inplace(scores: &mut [f64]) {
    if scores.is_empty() {
        return;
    }
    let mut mx = f64::NEG_INFINITY;
    for &s in scores.iter() {
        if s > mx {
            mx = s;
        }
    }
    if !mx.is_finite() {
        return;
    }
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - mx).exp();
        sum += *s;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
    }
}

/// Mean attention-output cosine similarity over query heads.
/// Returns `1.0` when `T == 0` (vacuous) and skips degenerate outputs.
#[allow(clippy::too_many_arguments)] // workspace-policy-allowed; 9 plain scalars for one matmul.
pub fn attention_output_cosine(
    k_ref: &[f64],
    v_ref: &[f64],
    k_hat: &[f64],
    v_hat: &[f64],
    q: &[f64],
    t: usize,
    nkv: usize,
    nq: usize,
    d: usize,
) -> f64 {
    if t == 0 || nkv == 0 || nq == 0 || d == 0 {
        return 1.0;
    }
    let gqa = nq / nkv;
    let inv_sqrt = 1.0 / (d as f64).sqrt();
    let mut cos_sum = 0.0;
    let mut count = 0;
    for h in 0..nq {
        let g = h / gqa;
        let mut s_ref = vec![0.0; t];
        let mut s_hat = vec![0.0; t];
        for tok in 0..t {
            let kr = &k_ref[(tok * nkv + g) * d..(tok * nkv + g + 1) * d];
            let kh = &k_hat[(tok * nkv + g) * d..(tok * nkv + g + 1) * d];
            let qh = &q[h * d..(h + 1) * d];
            let mut dr = 0.0;
            let mut dh = 0.0;
            for j in 0..d {
                dr += qh[j] * kr[j];
                dh += qh[j] * kh[j];
            }
            s_ref[tok] = dr * inv_sqrt;
            s_hat[tok] = dh * inv_sqrt;
        }
        softmax_inplace(&mut s_ref);
        softmax_inplace(&mut s_hat);
        let mut o_ref = vec![0.0; d];
        let mut o_hat = vec![0.0; d];
        for tok in 0..t {
            let vr = &v_ref[(tok * nkv + g) * d..(tok * nkv + g + 1) * d];
            let vh = &v_hat[(tok * nkv + g) * d..(tok * nkv + g + 1) * d];
            for j in 0..d {
                o_ref[j] += s_ref[tok] * vr[j];
                o_hat[j] += s_hat[tok] * vh[j];
            }
        }
        let dot: f64 = o_ref.iter().zip(o_hat.iter()).map(|(a, b)| a * b).sum();
        let nr: f64 = o_ref.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nh: f64 = o_hat.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nr > 1e-20 && nh > 1e-20 {
            cos_sum += dot / (nr * nh);
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        cos_sum / count as f64
    }
}

/// Mean per-vector cosine `⟨x, x̂⟩ / (‖x‖·‖x̂‖)` over a `[n_vecs, d]` tensor.
/// Vectors with near-zero norm (sink / all-zero) are skipped.
pub fn mean_vector_cosine(orig: &[f64], recon: &[f64], d: usize) -> f64 {
    if d == 0 || orig.is_empty() {
        return 1.0;
    }
    debug_assert_eq!(orig.len(), recon.len());
    let n_vecs = orig.len() / d;
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..n_vecs {
        let (mut dot, mut no, mut nr) = (0.0, 0.0, 0.0);
        for j in 0..d {
            let a = orig[i * d + j];
            let b = recon[i * d + j];
            dot += a * b;
            no += a * a;
            nr += b * b;
        }
        no = no.sqrt();
        nr = nr.sqrt();
        if no > 1e-20 && nr > 1e-20 {
            sum += dot / (no * nr);
            count += 1;
        }
    }
    if count == 0 { 1.0 } else { sum / count as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_kv_gives_cosine_one() {
        let d = 8;
        let nkv = 2;
        let nq = 4;
        let t = 5;
        let k: Vec<f64> = (0..t * nkv * d).map(|i| (i as f64).sin()).collect();
        let v: Vec<f64> = (0..t * nkv * d).map(|i| (i as f64).cos()).collect();
        let q: Vec<f64> = (0..nq * d).map(|i| (i as f64 * 0.3).sin()).collect();
        let c = attention_output_cosine(&k, &v, &k, &v, &q, t, nkv, nq, d);
        assert!((c - 1.0).abs() < 1e-9, "identical KV cosine {c}");
    }
}
