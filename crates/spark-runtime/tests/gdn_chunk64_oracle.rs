// SPDX-License-Identifier: AGPL-3.0-only

//! Chunk-64 GDN prefill ORACLE (the load-bearing foundation for the
//! chunked-scan GDN rewrite — the #1 prefill lever vs vLLM's FLA).
//!
//! Two CPU references over the SAME live-kernel math (read from
//! `gated_delta_rule_persistent.cu::gated_delta_rule_prefill_persistent_wy4`
//! + `parity_gdn.rs` decode reference):
//!
//!   `recurrent_ref` — the per-token delta-rule recurrence, the SSOT:
//!       hk   = <h, k_t>
//!       vnew = (v_t - g_t·hk)·beta_t
//!       h    = g_t·h + vnew·k_t        (per k-row, per v-element)
//!       out  = <h, q_t>·rsqrt(k_dim)   (uses UPDATED h)
//!     NO gate clamp (the live PREFILL kernel does not clamp — only decode
//!     does), NO in-kernel l2norm (applied upstream), per-token gate/beta.
//!
//!   `chunk64_ref` — the chunked-parallel form the new CUDA kernel will use,
//!     processing C=64 tokens at once. Gate decay is computed in LOG space:
//!     gate[] is a LINEAR decay factor, but the product of 64 sub-1 gates
//!     underflows, so we use exp(cumsum(log g)). exp(gc_i - gc_l) ==
//!     Π_{m=l+1}^{i} g_m EXACTLY — numerically stable, mathematically
//!     identical to the linear products wy4 uses at C=4. (This is the bug
//!     the design review missed in BOTH directions: the design forgot the
//!     initial log(); the review wrongly said use raw linear products.)
//!
//! GATE A part 1 (this file, CPU-only, no GPU): chunk64_ref == recurrent_ref
//! in f64. Validates the chunk decomposition math BEFORE any CUDA is written.
//! (Part 2 — recurrent_ref vs the LIVE wy4 kernel in bf16 — lands in a
//! follow-up that launches the kernel; this file proves the math first.)

#[derive(Clone, Copy)]
struct Dims {
    t: usize,
    nk: usize,
    nv: usize,
    kd: usize,
    vd: usize,
}
impl Dims {
    fn head_repeat(&self) -> usize {
        self.nv / self.nk
    }
}

/// Deterministic LCG → reproducible inputs without an rng dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        // Numerical Recipes LCG; take top bits → [0,1).
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

struct Inputs {
    q: Vec<f64>, // [t][nk][kd]
    k: Vec<f64>,
    v: Vec<f64>,    // [t][nv][vd]
    gate: Vec<f64>, // [t][nv]
    beta: Vec<f64>,
    h0: Vec<f64>, // [nv][kd][vd]
}

fn gen_inputs(d: Dims, seed: u64) -> Inputs {
    let mut r = Lcg(seed);
    let q = (0..d.t * d.nk * d.kd).map(|_| r.range(-0.5, 0.5)).collect();
    let k = (0..d.t * d.nk * d.kd).map(|_| r.range(-0.5, 0.5)).collect();
    let v = (0..d.t * d.nv * d.vd).map(|_| r.range(-0.5, 0.5)).collect();
    // Realistic GDN gates: close to 1 (slow decay); always > 0 so log() is finite.
    let gate = (0..d.t * d.nv).map(|_| r.range(0.80, 0.999)).collect();
    let beta = (0..d.t * d.nv).map(|_| r.range(0.0, 1.0)).collect();
    let h0 = (0..d.nv * d.kd * d.vd).map(|_| r.range(-0.1, 0.1)).collect();
    Inputs { q, k, v, gate, beta, h0 }
}

#[inline]
fn qk(idx_t: usize, kh: usize, j: usize, d: Dims) -> usize {
    (idx_t * d.nk + kh) * d.kd + j
}
#[inline]
fn vid(idx_t: usize, vh: usize, tid: usize, d: Dims) -> usize {
    (idx_t * d.nv + vh) * d.vd + tid
}
#[inline]
fn hid(vh: usize, j: usize, tid: usize, d: Dims) -> usize {
    (vh * d.kd + j) * d.vd + tid
}

/// Per-token recurrence — the token-equality SSOT.
fn recurrent_ref(inp: &Inputs, d: Dims) -> (Vec<f64>, Vec<f64>) {
    let scale = (d.kd as f64).powf(-0.5);
    let mut h = inp.h0.clone();
    let mut out = vec![0.0f64; d.t * d.nv * d.vd];
    for vh in 0..d.nv {
        let kh = vh / d.head_repeat();
        for t in 0..d.t {
            let g = inp.gate[t * d.nv + vh];
            let bt = inp.beta[t * d.nv + vh];
            for tid in 0..d.vd {
                let mut hk = 0.0;
                for j in 0..d.kd {
                    hk += h[hid(vh, j, tid, d)] * inp.k[qk(t, kh, j, d)];
                }
                let vnew = (inp.v[vid(t, vh, tid, d)] - g * hk) * bt;
                let mut qdot = 0.0;
                for j in 0..d.kd {
                    let idx = hid(vh, j, tid, d);
                    let hn = g * h[idx] + inp.k[qk(t, kh, j, d)] * vnew;
                    h[idx] = hn;
                    qdot += hn * inp.q[qk(t, kh, j, d)];
                }
                out[vid(t, vh, tid, d)] = qdot * scale;
            }
        }
    }
    (out, h)
}

/// Chunked-parallel form (C=64), log-of-linear gate decay. Must equal
/// `recurrent_ref` exactly (in f64).
fn chunk64_ref(inp: &Inputs, d: Dims) -> (Vec<f64>, Vec<f64>) {
    const C: usize = 64;
    let scale = (d.kd as f64).powf(-0.5);
    let mut h = inp.h0.clone();
    let mut out = vec![0.0f64; d.t * d.nv * d.vd];
    let dot = |a_t: usize, ah: usize, b_t: usize, bh: usize, src_a: &[f64], src_b: &[f64]| -> f64 {
        let mut s = 0.0;
        for j in 0..d.kd {
            s += src_a[qk(a_t, ah, j, d)] * src_b[qk(b_t, bh, j, d)];
        }
        s
    };
    for vh in 0..d.nv {
        let kh = vh / d.head_repeat();
        let mut cs = 0;
        while cs < d.t {
            let ce = C.min(d.t - cs);
            // inclusive log-cumsum of linear gates
            let mut gc = vec![0.0f64; ce];
            let mut acc = 0.0;
            for i in 0..ce {
                acc += inp.gate[(cs + i) * d.nv + vh].ln();
                gc[i] = acc;
            }
            for tid in 0..d.vd {
                // forward-substitution solve for u (== per-token vnew)
                let mut u = vec![0.0f64; ce];
                for i in 0..ce {
                    let bi = inp.beta[(cs + i) * d.nv + vh];
                    let mut hk = 0.0;
                    for j in 0..d.kd {
                        hk += h[hid(vh, j, tid, d)] * inp.k[qk(cs + i, kh, j, d)];
                    }
                    let mut ri = bi * inp.v[vid(cs + i, vh, tid, d)] - bi * gc[i].exp() * hk;
                    for l in 0..i {
                        let kk = dot(cs + l, kh, cs + i, kh, &inp.k, &inp.k);
                        ri -= bi * (gc[i] - gc[l]).exp() * kk * u[l];
                    }
                    u[i] = ri;
                }
                // outputs: out_i = scale·[exp(gc_i)<H0,q_i> + Σ_{l<=i} exp(gc_i-gc_l) u_l <k_l,q_i>]
                for i in 0..ce {
                    let mut hq = 0.0;
                    for j in 0..d.kd {
                        hq += h[hid(vh, j, tid, d)] * inp.q[qk(cs + i, kh, j, d)];
                    }
                    let mut o = gc[i].exp() * hq;
                    for l in 0..=i {
                        let kq = dot(cs + l, kh, cs + i, kh, &inp.k, &inp.q);
                        o += (gc[i] - gc[l]).exp() * u[l] * kq;
                    }
                    out[vid(cs + i, vh, tid, d)] = o * scale;
                }
                // state update: H = exp(gc_last)·H0 + Σ_l exp(gc_last-gc_l) u_l k_l
                let last = ce - 1;
                for j in 0..d.kd {
                    let idx = hid(vh, j, tid, d);
                    let mut hv = gc[last].exp() * h[idx];
                    for l in 0..ce {
                        hv += (gc[last] - gc[l]).exp() * u[l] * inp.k[qk(cs + l, kh, j, d)];
                    }
                    h[idx] = hv;
                }
            }
            cs += ce;
        }
    }
    (out, h)
}

/// FLA-style 3-pass DECOMPOSITION (the multi-kernel structure):
///   recompute_w_u (parallel over chunks, H-independent): solve (I+L)U = βV and
///       (I+L)W = β·exp(gc)·K  by forward-substitution. L[i][l]=β_i·exp(gc_i-gc_l)·<k_l,k_i>.
///   chunk_delta_h (serial over chunks): S_{c+1}=exp(gc_last)·S_c + Σ_i exp(gc_last-gc_i)·u_i·k_i.
///   chunk_fwd_o (parallel over chunks): O_i = (exp(gc_i)·<S_c,q_i> + Σ_{l<=i} exp(gc_i-gc_l)·<k_l,q_i>·u_l)·scale.
///   where u_i = U_i - <W_i, S_c>  (the only S-coupling; U,W are H-independent → parallel).
/// Must equal `recurrent_ref` exactly (u = U - W·S reduces to the recurrent vnew).
fn fla_decomposed_ref(inp: &Inputs, d: Dims) -> (Vec<f64>, Vec<f64>) {
    const C: usize = 64;
    let scale = (d.kd as f64).powf(-0.5);
    let mut h = inp.h0.clone(); // S[k][v] per vh via hid(vh,k,v)
    let mut out = vec![0.0f64; d.t * d.nv * d.vd];
    let kdot = |a_t: usize, b_t: usize, kh: usize, sb: &[f64]| -> f64 {
        let mut s = 0.0;
        for j in 0..d.kd { s += inp.k[qk(a_t, kh, j, d)] * sb[qk(b_t, kh, j, d)]; }
        s
    };
    for vh in 0..d.nv {
        let kh = vh / d.head_repeat();
        let mut cs = 0;
        while cs < d.t {
            let ce = C.min(d.t - cs);
            let mut gc = vec![0.0f64; ce];
            let mut acc = 0.0;
            for i in 0..ce { acc += inp.gate[(cs + i) * d.nv + vh].ln(); gc[i] = acc; }
            // recompute_w_u: U[i][v] (ce×vd), W[i][k] (ce×kd) via forward-sub (== T·βV, T·β·exp(gc)·K)
            let mut uu = vec![0.0f64; ce * d.vd];
            let mut ww = vec![0.0f64; ce * d.kd];
            for i in 0..ce {
                let bi = inp.beta[(cs + i) * d.nv + vh];
                for v in 0..d.vd { uu[i * d.vd + v] = bi * inp.v[vid(cs + i, vh, v, d)]; }
                let egci = gc[i].exp();
                for k in 0..d.kd { ww[i * d.kd + k] = bi * egci * inp.k[qk(cs + i, kh, k, d)]; }
                for l in 0..i {
                    let lil = bi * (gc[i] - gc[l]).exp() * kdot(cs + l, cs + i, kh, &inp.k);
                    for v in 0..d.vd { uu[i * d.vd + v] -= lil * uu[l * d.vd + v]; }
                    for k in 0..d.kd { ww[i * d.kd + k] -= lil * ww[l * d.kd + k]; }
                }
            }
            // u_full[i][v] = U[i][v] - <W[i], S[:,v]>  (entry state S = h[vh])
            let mut uf = vec![0.0f64; ce * d.vd];
            for i in 0..ce {
                for v in 0..d.vd {
                    let mut s = uu[i * d.vd + v];
                    for k in 0..d.kd { s -= ww[i * d.kd + k] * h[hid(vh, k, v, d)]; }
                    uf[i * d.vd + v] = s;
                }
            }
            // chunk_fwd_o (uses entry S + u_full)
            for i in 0..ce {
                let egci = gc[i].exp();
                for v in 0..d.vd {
                    let mut o = 0.0;
                    for k in 0..d.kd { o += egci * h[hid(vh, k, v, d)] * inp.q[qk(cs + i, kh, k, d)]; }
                    for l in 0..=i {
                        let kqd = kdot(cs + l, cs + i, kh, &inp.q);
                        o += (gc[i] - gc[l]).exp() * kqd * uf[l * d.vd + v];
                    }
                    out[vid(cs + i, vh, v, d)] = o * scale;
                }
            }
            // chunk_delta_h (serial state update, uses entry S + u_full)
            let last = ce - 1;
            let dl = gc[last];
            for k in 0..d.kd {
                for v in 0..d.vd {
                    let mut hv = dl.exp() * h[hid(vh, k, v, d)];
                    for i in 0..ce {
                        hv += (dl - gc[i]).exp() * uf[i * d.vd + v] * inp.k[qk(cs + i, kh, k, d)];
                    }
                    h[hid(vh, k, v, d)] = hv;
                }
            }
            cs += ce;
        }
    }
    (out, h)
}

#[test]
fn fla_decomposed_matches_recurrent_oracle() {
    let dims_base = |t| Dims { t, nk: 2, nv: 4, kd: 8, vd: 8 };
    for &t in &[1usize, 2, 63, 64, 65, 100, 128, 129, 192, 200] {
        let d = dims_base(t);
        let inp = gen_inputs(d, 0x1234_5678 ^ (t as u64));
        let (o_rec, h_rec) = recurrent_ref(&inp, d);
        let (o_fla, h_fla) = fla_decomposed_ref(&inp, d);
        let mut max_o = 0.0f64;
        for (a, b) in o_rec.iter().zip(&o_fla) { max_o = max_o.max((a - b).abs()); }
        let mut max_h = 0.0f64;
        for (a, b) in h_rec.iter().zip(&h_fla) { max_h = max_h.max((a - b).abs()); }
        assert!(
            max_o < 1e-9 && max_h < 1e-9,
            "seq_len={t}: FLA-decomposed vs recurrent diverged — out={max_o:e} h={max_h:e}"
        );
    }
}

#[test]
fn chunk64_matches_recurrent_oracle() {
    let dims_base = |t| Dims { t, nk: 2, nv: 4, kd: 8, vd: 8 };
    // seq_lens spanning full chunks + every partial-final-chunk remainder class.
    for &t in &[1usize, 2, 63, 64, 65, 100, 128, 129, 192, 200] {
        let d = dims_base(t);
        let inp = gen_inputs(d, 0x1234_5678 ^ (t as u64));
        let (o_rec, h_rec) = recurrent_ref(&inp, d);
        let (o_chk, h_chk) = chunk64_ref(&inp, d);

        let mut max_o = 0.0f64;
        for (a, b) in o_rec.iter().zip(&o_chk) {
            max_o = max_o.max((a - b).abs());
        }
        let mut max_h = 0.0f64;
        for (a, b) in h_rec.iter().zip(&h_chk) {
            max_h = max_h.max((a - b).abs());
        }
        // Pure-f64 math equivalence: differences are roundoff only.
        assert!(
            max_o < 1e-9 && max_h < 1e-9,
            "seq_len={t}: chunk64 vs recurrent diverged — out={max_o:e} h={max_h:e}"
        );
    }
}
