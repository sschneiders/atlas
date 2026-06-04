// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN prefill — FLA-style MULTI-KERNEL decomposition (the path to beat
// vLLM). The single-fused chunk64 kernel was boxed in (serial-per-chunk in one
// CTA → 0.38-0.69x). FLA's speed comes from splitting into passes where the BIG
// matmuls are PARALLEL over all chunks (full 48-SM occupancy) and only a small
// state-passing is serial:
//   recompute_w_u (THIS, parallel over chunks×heads, H-independent):
//       solve (I+L)U = βV, (I+L)W = β·exp(gc)·K  via forward-substitution.
//       L[i][l] = β_i·exp(gc_i-gc_l)·<k_l,k_i> (l<i), Gram built on tensor cores.
//   chunk_delta_h (serial over chunks): S_{c+1}=exp(gc_last)S_c + K̃ᵀU - K̃ᵀ(WS).
//   chunk_fwd_o (parallel over chunks): O = Q̃·S + tril(Q̃·Kᵀ)·U.
// Math validated == recurrent SSOT by the CPU oracle
// (crates/spark-runtime/tests/gdn_chunk64_oracle.rs :: fla_decomposed_ref).
//
// gate[] LINEAR decay (NO exp, NO clamp on prefill); chunk decay in LOG space.
// GB10 sm_121: mma.sync.m16n8k16 BF16 (no wgmma/TMA, ldmatrix broken).

#include <cuda_bf16.h>

#define K_DIM 128
#define V_DIM 128
#define CHUNK 64

// C[m][n] = Σ_k A[m][k]·B[n][k], M=64, K=K_DIM, N=NTC*8. A/B row-major bf16 smem;
// 128 threads = 4 warps (16 M-rows each). NSTRIDE = C row-stride. (SSOT helper.)
template <int NTC, int NSTRIDE, bool OutBf16>
__device__ __forceinline__ void mma_gram(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, void* __restrict__ C
) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    const unsigned grp = lane >> 2;
    const unsigned q = lane & 3;
    const unsigned warp_m = warp * 16;
    const unsigned short* sA = (const unsigned short*)A;
    const unsigned short* sB = (const unsigned short*)B;
    float acc[NTC][4];
    #pragma unroll
    for (int nt = 0; nt < NTC; nt++) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
    #pragma unroll
    for (unsigned ks = 0; ks < K_DIM; ks += 16) {
        unsigned fr0 = warp_m + grp, fr1 = fr0 + 8;
        unsigned fc0 = ks + q * 2, fc1 = fc0 + 8;
        unsigned a0 = *(const unsigned*)&sA[fr0 * K_DIM + fc0];
        unsigned a1 = *(const unsigned*)&sA[fr1 * K_DIM + fc0];
        unsigned a2 = *(const unsigned*)&sA[fr0 * K_DIM + fc1];
        unsigned a3 = *(const unsigned*)&sA[fr1 * K_DIM + fc1];
        #pragma unroll
        for (int nt = 0; nt < NTC; nt++) {
            unsigned nc = nt * 8 + grp;
            unsigned k0 = ks + q * 2, k1 = k0 + 8;
            unsigned b0 = ((unsigned)sB[nc * K_DIM + k0 + 1] << 16) | (unsigned)sB[nc * K_DIM + k0];
            unsigned b1 = ((unsigned)sB[nc * K_DIM + k1 + 1] << 16) | (unsigned)sB[nc * K_DIM + k1];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }
    #pragma unroll
    for (int nt = 0; nt < NTC; nt++) {
        unsigned n0 = nt * 8 + q * 2, n1 = n0 + 1;
        unsigned m0 = warp_m + grp, m1 = m0 + 8;
        if (OutBf16) {
            __nv_bfloat16* Cb = (__nv_bfloat16*)C;
            Cb[m0 * NSTRIDE + n0] = __float2bfloat16(acc[nt][0]);
            Cb[m0 * NSTRIDE + n1] = __float2bfloat16(acc[nt][1]);
            Cb[m1 * NSTRIDE + n0] = __float2bfloat16(acc[nt][2]);
            Cb[m1 * NSTRIDE + n1] = __float2bfloat16(acc[nt][3]);
        } else {
            float* Cf = (float*)C;
            Cf[m0 * NSTRIDE + n0] = acc[nt][0];
            Cf[m0 * NSTRIDE + n1] = acc[nt][1];
            Cf[m1 * NSTRIDE + n0] = acc[nt][2];
            Cf[m1 * NSTRIDE + n1] = acc[nt][3];
        }
    }
}

// ── KERNEL 1: recompute_w_u ──────────────────────────────────────────────
// Grid: (NT, num_v_heads, batch)  Block: (128,1,1).  One CTA per (chunk, head).
// Outputs (f32, layout [(b*NT+c)*nv+vh][CHUNK][·]):
//   U_out: [.. ][CHUNK][V_DIM]   = T·(βV)
//   W_out: [.. ][CHUNK][K_DIM]   = T·(β·exp(gc)·K)
// where T=(I+L)⁻¹ applied by forward-substitution (parallel over the V/K cols).
// smem: sk_bf(16K) + kk(16K f32) + L(16K f32) + gc(256) ≈ 48.25KB.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_recompute_wu(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ W_out,   // bf16 — per-chunk intermediate, fed to TC matmuls in #2/#3
    __nv_bfloat16* __restrict__ U_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,   // NT = ceil(seq_len/CHUNK)
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int c = blockIdx.x;     // chunk index
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (c >= num_chunks || vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sk = (__nv_bfloat16*)smem_raw;       // [CHUNK*K_DIM] bf16
    float* kk = (float*)(sk + CHUNK * K_DIM);           // [CHUNK*CHUNK] f32 Gram
    float* L = kk + CHUNK * CHUNK;                      // [CHUNK*CHUNK] f32 decay-weighted strict-lower
    float* gc = L + CHUNK * CHUNK;                      // [CHUNK]

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        sk[i * K_DIM + j] = (i < ce)
            ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + j]
            : __float2bfloat16(0.0f);
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            acc += logf(gate[(unsigned long long)(cs + i) * gb_stride + vh]);
            gc[i] = acc;
        }
    }
    __syncthreads();

    mma_gram<8, CHUNK, false>(sk, sk, kk);   // kk[l][i] = <k_l,k_i>
    __syncthreads();

    // L[i][l] = β_i·exp(gc_i-gc_l)·<k_l,k_i>  for l<i ; 0 otherwise.  (kk symmetric)
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += 128) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l < i) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            L[i * CHUNK + l] = bi * expf(gc[i] - gc[l]) * kk[l * CHUNK + i];
        } else {
            L[i * CHUNK + l] = 0.0f;
        }
    }
    __syncthreads();

    const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

    // Pass 1: U[:,v] forward-sub (one thread per v-element).  U_i = β_i·V_i - Σ_{l<i} L[i][l]·U_l
    if (tid < v_dim) {
        float u[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            float ui = bi * (float)value[(unsigned long long)(cs + i) * v_stride + vh * v_dim + tid];
            for (unsigned int l = 0; l < i; l++) ui -= L[i * CHUNK + l] * u[l];
            u[i] = ui;
            U_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(ui);
        }
    }
    // Pass 2: W[:,k] forward-sub (one thread per k-element).  W_i = β_i·exp(gc_i)·K_i - Σ_{l<i} L[i][l]·W_l
    if (tid < k_dim) {
        float w[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            float wi = bi * expf(gc[i]) * (float)sk[i * K_DIM + tid];
            for (unsigned int l = 0; l < i; l++) wi -= L[i * CHUNK + l] * w[l];
            w[i] = wi;
            W_out[base * CHUNK * K_DIM + i * k_dim + tid] = __float2bfloat16(wi);
        }
    }
}

// ── KERNEL 2: chunk_delta_h ──────────────────────────────────────────────
// The SERIAL state-passing spine — PRECISION-CRITICAL, so S stays f32 and its
// matmuls are fp32-FFMA (NOT bf16-TC: bf16-S drift fails token-equality).
// Grid: (num_v_heads, batch). One CTA per head, serial over chunks. 128 threads
// = v-columns; thread tid owns the WHOLE state column S[:,tid] RESIDENT IN
// REGISTERS (Sreg[K_DIM]) across all chunks — loaded once, updated in-register
// per chunk, stored once. This kills the per-chunk smem read/write of the 64KB
// f32 state (~98k smem ops/thread over a 16k ctx) that bottlenecked the prior
// smem-resident design at ~5.8 TFLOP/s (latency-bound at 4 warps, not FLOP-bound).
// Freed smem also lets W and K live in SEPARATE buffers → one fewer sync/chunk.
// (V-tiling for occupancy REGRESSED — 2 CTAs/head redundantly reload W + re-run
// the serial loop. bf16x2-TC for the matmuls is the precision-sensitive next lever.)
// Per chunk c (entry S_c): store S_c → S_out; uc = U_c - W_c·S_c → uc_out;
// S_{c+1} = exp(gc_last)·S_c + Σ_i exp(gc_last-gc_i)·uc_i·k_i.
// smem: Wb(16K bf16) + Kb(16K bf16) + Ub(16K bf16) + gc = 49408 B.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_delta_h(
    float* __restrict__ h_state,          // [nv][K][V] per (b,vh): entry state IN, final state OUT
    const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    float* __restrict__ S_out,            // [(b*NT+c)*nv+vh][K][V] per-chunk ENTRY states
    __nv_bfloat16* __restrict__ uc_out,   // [(b*NT+c)*nv+vh][C][V] corrected values
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Wb = (__nv_bfloat16*)smem_raw;                 // [CHUNK*K_DIM] bf16
    __nv_bfloat16* Kb = Wb + CHUNK * K_DIM;                       // [CHUNK*K_DIM] bf16
    __nv_bfloat16* Ub = Kb + CHUNK * K_DIM;                       // [CHUNK*V_DIM] bf16
    float* gc = (float*)(Ub + CHUNK * V_DIM);                     // [CHUNK]

    // State column S[:,tid] resident in registers for this thread's whole lifetime.
    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[K_DIM];
    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) Sreg[k] = H[k * V_DIM + tid];

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

        // Store entry state S_c (thread tid owns column tid).
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = Sreg[k];
        for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
            unsigned int i = idx / k_dim, k = idx % k_dim;
            bool live = i < ce;
            Wb[i * K_DIM + k] = live ? W_in[base * CHUNK * K_DIM + i * k_dim + k] : __float2bfloat16(0.0f);
            Kb[i * K_DIM + k] = live
                ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + k]
                : __float2bfloat16(0.0f);
        }
        for (unsigned int idx = tid; idx < CHUNK * v_dim; idx += 128) {
            unsigned int i = idx / v_dim, v = idx % v_dim;
            Ub[i * V_DIM + v] = (i < ce) ? U_in[base * CHUNK * V_DIM + i * v_dim + v] : __float2bfloat16(0.0f);
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (unsigned int i = 0; i < ce; i++) {
                acc += logf(gate[(unsigned long long)(cs + i) * gb_stride + vh]);
                gc[i] = acc;
            }
        }
        __syncthreads();

        // uc_i = U_i - W_i·S   (W·S contracts over k against the register state column)
        float duc[CHUNK];
        const float dl = gc[ce - 1];
        const float edl = expf(dl);
        for (unsigned int i = 0; i < ce; i++) {
            float ws = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < K_DIM; k++)
                ws += (float)Wb[i * K_DIM + k] * Sreg[k];
            float uci = (float)Ub[i * V_DIM + tid] - ws;
            uc_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(uci);
            duc[i] = expf(dl - gc[i]) * uci;   // decayed corrected-value, once per i
        }
        // S_{c+1} = edl·S + Σ_i duc_i·k_i   (in-register update, no smem state traffic)
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) {
            float hv = edl * Sreg[k];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kb[i * K_DIM + k];   // pure MAC inner loop
            Sreg[k] = hv;
        }
        __syncthreads();   // Wb/Kb/Ub reused next chunk
    }

    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) H[k * V_DIM + tid] = Sreg[k];
}

// ── KERNEL 3: chunk_fwd_o ────────────────────────────────────────────────
// The PARALLEL output pass. Grid: (NT, num_v_heads, batch). One CTA per (chunk,head).
// O_i = (exp(gc_i)·<S_c[:,v],q_i> + Σ_{l<=i} exp(gc_i-gc_l)·<k_l,q_i>·uc_l[v])·rsqrt(d).
// kq=<q_i,k_l> Gram on tensor cores; S_c read bf16 (TERMINAL output → no compounding,
// like wy4's bf16 output rounding → precision-safe). Output layout matches wy4.
// smem: sq(16K)+sk(16K)+kq(16K f32)+ucb(16K)+Sb(32K bf16)+gc = 96.25KB.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_fwd_o(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ S_in,       // [(b*NT+c)*nv+vh][K][V] entry states (from #2)
    const __nv_bfloat16* __restrict__ uc_in,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride
) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (c >= num_chunks || vh >= num_v_heads || b >= batch_size) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);
    const unsigned long long out_base = ((unsigned long long)(b * seq_len) * num_v_heads + vh) * v_dim;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sq = (__nv_bfloat16*)smem_raw;          // [CHUNK*K_DIM]
    __nv_bfloat16* sk = sq + CHUNK * K_DIM;                // [CHUNK*K_DIM]
    float* kq = (float*)(sk + CHUNK * K_DIM);              // [CHUNK*CHUNK]
    __nv_bfloat16* ucb = (__nv_bfloat16*)(kq + CHUNK * CHUNK); // [CHUNK*V_DIM]
    __nv_bfloat16* Sb = ucb + CHUNK * V_DIM;               // [K_DIM*V_DIM] bf16 (S_c)
    float* gc = (float*)(Sb + K_DIM * V_DIM);              // [CHUNK]

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        if (i < ce) {
            unsigned long long off = (unsigned long long)(cs + i) * qk_stride + kh * k_dim + j;
            sq[i * K_DIM + j] = query[off];
            sk[i * K_DIM + j] = key[off];
        } else {
            sq[i * K_DIM + j] = __float2bfloat16(0.0f);
            sk[i * K_DIM + j] = __float2bfloat16(0.0f);
        }
    }
    for (unsigned int idx = tid; idx < CHUNK * v_dim; idx += 128) {
        unsigned int i = idx / v_dim, v = idx % v_dim;
        ucb[i * V_DIM + v] = (i < ce) ? uc_in[base * CHUNK * V_DIM + i * v_dim + v] : __float2bfloat16(0.0f);
    }
    for (unsigned int idx = tid; idx < K_DIM * V_DIM; idx += 128)
        Sb[idx] = __float2bfloat16(S_in[base * K_DIM * V_DIM + idx]);
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            acc += logf(gate[(unsigned long long)(cs + i) * gb_stride + vh]);
            gc[i] = acc;
        }
    }
    __syncthreads();

    mma_gram<8, CHUNK, false>(sq, sk, kq);   // kq[i][l] = <q_i, k_l>
    __syncthreads();

    // Fold the intra-chunk decay into the Gram ONCE: kq[i][l] ← exp(gc_i-gc_l)·<q_i,k_l>
    // (was: expf(gc_i-gc_l) recomputed per v-column = v_dim× redundant transcendentals).
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += 128) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l <= i) kq[p] = expf(gc[i] - gc[l]) * kq[p];
    }
    __syncthreads();

    if (tid < v_dim) {
        for (unsigned int i = 0; i < ce; i++) {
            float t1 = 0.0f;
            for (unsigned int k = 0; k < k_dim; k++)
                t1 += (float)Sb[k * V_DIM + tid] * (float)sq[i * K_DIM + k];
            t1 *= expf(gc[i]);
            float t2 = 0.0f;
            for (unsigned int l = 0; l <= i; l++)
                t2 += kq[i * CHUNK + l] * (float)ucb[l * V_DIM + tid];   // pure MAC inner loop
            output[out_base + (unsigned long long)(cs + i) * num_v_heads * v_dim + tid] =
                __float2bfloat16((t1 + t2) * inv_sqrt_d);
        }
    }
}
