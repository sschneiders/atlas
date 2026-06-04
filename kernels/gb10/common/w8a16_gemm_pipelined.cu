// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 Pipelined Dequant+GEMM — Fix-A tensor-core rewrite.
//
// C[M,N] = A[M,K] (BF16 activations) * dequant(B[N,K] (FP8 E4M3 weights))
//
// Same math + same number formats as the production `w8a16_gemm`, but with
// a much larger 128×128 output tile (8 warps) so each CTA re-streams far
// less of the B weight matrix from DRAM, plus a 2-stage cp.async software
// pipeline that hides global-load latency behind the tensor-core MMAs.
//
// FP8-E4M3 weight format (2D block-scaled), identical to w8a16_gemm:
//   B:           [N, K] uint8 — one byte per weight (FP8 E4M3)
//   block_scale: [N/128, K/128] FP32 — per-block scale factor
//
// Two-level FP32 accumulation (PRESERVED EXACTLY from w8a16_gemm — holds a
// deep-layer FP8 precision floor):
//   inner_acc accumulates MMA outputs across the 8 K_STEPs of one 128-K
//   block, where smem_B holds UNSCALED BF16-cast E4M3 weights (lossless,
//   E4M3 has 3 mantissa bits ⊂ BF16's 7). At each 128-K-block boundary:
//       outer_acc += inner_acc * block_scale[n_block, k_block]; inner_acc = 0
//   The scale is applied ONCE per block on the FP32 accumulator — never
//   per-element, never folded into BF16.
//
// Pipeline (Stage 2): a 2-deep cp.async software pipeline prefetches the
// NEXT K-step's A tile (BF16, contiguous K-run) and RAW FP8 B bytes (per-N
// contiguous K-run) into double-buffered smem while the MMAs consume the
// current K-step. cp.async.cg (sm_80+) is correct on sm_121; TMA /
// cp.async.bulk are AVOIDED (they silently corrupt on sm_121).
//
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.
//
// Grid: (ceil(N/128), ceil(M/128), 1), Block: (256, 1, 1) = 8 warps.
// Static shared memory only (~24.7 KB), under the 48 KB limit.

#include <cuda_bf16.h>
#include "e4m3_lut.cuh"

#define PM_M_TILE 128
#define PM_N_TILE 128
#define PM_K_STEP 16
#define PM_PAD 2
// A-tile smem row stride MUST be a multiple of 8 BF16 (16 bytes) so every
// 16-B cp.async chunk lands on a 16-byte boundary (cp.async.cg requires
// 16-byte alignment — a 36-byte PAD=2 stride misaligns odd rows and faults
// with CUDA_ERROR_MISALIGNED_ADDRESS). 16 real K-cols + 8 pad = 24 BF16 =
// 48 bytes (multiple of 16); the pad also breaks shared-memory bank
// conflicts on the MMA's u32 reads.
#define PM_A_STRIDE 24                        // 16 K-cols + 8 pad, mult. of 8 BF16
#define PM_FP8_BLOCK 128
#define PM_WARPS 8
#define PM_THREADS (PM_WARPS * 32)            // 256
#define PM_N_TILES_PER_WARP (PM_N_TILE / 8)   // 16 m16n8k16 N-tiles per warp
#define PM_STAGES 2                           // double-buffered cp.async pipeline

// cp.async 16-byte (cg = cache-global) copy: smem <- global. sm_80+; correct
// on sm_121 (unlike TMA / cp.async.bulk). Requires 16-byte-aligned addresses.
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// MMA over one resident K_STEP (16 K-elements). Accumulates into inner[16][4]
// (one m16n8k16 per N-tile). Byte-for-byte identical fragment layout to
// w8a16_gemm's helper, just generalised from 8 to 16 N-tiles.
__device__ __forceinline__ void pm_mma_kstep(
    const __nv_bfloat16* smem_A,   // [PM_M_TILE][PM_A_STRIDE]
    const __nv_bfloat16* smem_B,   // [PM_K_STEP][PM_N_TILE + PM_PAD]
    float inner[PM_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = PM_A_STRIDE;
    const unsigned int b_stride = PM_N_TILE + PM_PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;
    unsigned int frag_c0 = tid * 2;
    unsigned int frag_c1 = tid * 2 + 8;

    unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
    unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
    unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
    unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

    #pragma unroll
    for (int n_tile = 0; n_tile < PM_N_TILES_PER_WARP; n_tile++) {
        unsigned int n_col = n_tile * 8 + group_id;
        unsigned int k0 = tid * 2;
        unsigned int k1 = tid * 2 + 8;

        unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k0 * b_stride + n_col];
        unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k1 * b_stride + n_col];

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(inner[n_tile][0]), "=f"(inner[n_tile][1]),
              "=f"(inner[n_tile][2]), "=f"(inner[n_tile][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(inner[n_tile][0]), "f"(inner[n_tile][1]),
              "f"(inner[n_tile][2]), "f"(inner[n_tile][3])
        );
    }
}

/// W8A16 pipelined GEMM: B[N,K] row-major FP8 E4M3 with 2D block scales.
/// 128×128 tile, 8 warps, 2-stage cp.async prefetch pipeline.
extern "C" __global__ void w8a16_gemm_pipelined(
    const __nv_bfloat16* __restrict__ A,            // [M, K] BF16 activations
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,           // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                   // [M, N] BF16 output
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * PM_M_TILE;
    const unsigned int cta_n = blockIdx.x * PM_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;   // 8 warps × 16 = 128 M-rows
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // Double-buffered smem (~24.7 KB total). cp.async destinations (smem_A,
    // smem_Braw) are __align__(16) with 16-byte-aligned row strides so every
    // 16-B cp.async.cg chunk is naturally aligned. smem_B (MMA-ready [k][n]
    // BF16) is written by scalar dequant stores so its alignment is free.
    __shared__ __align__(16) __nv_bfloat16 smem_A[PM_STAGES][PM_M_TILE][PM_A_STRIDE];
    __shared__ __nv_bfloat16 smem_B[PM_STAGES][PM_K_STEP][PM_N_TILE + PM_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[PM_STAGES][PM_N_TILE][PM_K_STEP];

    // Two-level FP32 accumulation (see file header — PRESERVED EXACTLY).
    float inner_acc[PM_N_TILES_PER_WARP][4];
    float outer_acc[PM_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < PM_N_TILES_PER_WARP; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int k_blocks = K / PM_FP8_BLOCK;
    const unsigned int k_steps_per_block = PM_FP8_BLOCK / PM_K_STEP;
    const unsigned int n_block = cta_n / PM_FP8_BLOCK;
    const unsigned int n_steps = (K + PM_K_STEP - 1) / PM_K_STEP;

    // A-tile cp.async: 128 rows × 16 K-cols BF16. Each row = 16 BF16 = 32 B =
    // two 16-B chunks. 256 chunks total → 1 chunk/thread (256 threads).
    const unsigned int a_chunks = (PM_M_TILE * PM_K_STEP) / 8;     // 256

    // Issue cp.async loads for K-step `step` into double-buffer `stage`. The
    // copies are contiguous along K (the contiguous global axis for both A
    // [M,K] and B [N,K]) so each 16-B cp.async pulls a real run. Bounds /
    // K-tail fall back to a masked scalar copy (cp.async cannot predicate).
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        unsigned int k_base = step * PM_K_STEP;

        // ── A: contiguous 16-B (8 BF16) chunks along K ──
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += PM_THREADS) {
            unsigned int row = (c * 8) / PM_K_STEP;          // 0..127
            unsigned int col = (c * 8) % PM_K_STEP;          // 0 or 8
            unsigned int gr = cta_m + row;
            unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (gr < M && gc + 8 <= K) {
                cp_async_cg_16(dst, &A[(unsigned long long)gr * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gcol = gc + e;
                    dst[e] = (gr < M && gcol < K) ? A[(unsigned long long)gr * K + gcol]
                                                  : __float2bfloat16(0.0f);
                }
            }
        }

        // ── B raw: one contiguous 16-B run of K-bytes per N-row ──
        // smem_Braw[stage][n][k] mirrors global B[n, k_base + k] contiguously.
        #pragma unroll
        for (unsigned int nrow = threadIdx.x; nrow < PM_N_TILE; nrow += PM_THREADS) {
            unsigned int gn = cta_n + nrow;
            unsigned char* dst = &smem_Braw[stage][nrow][0];
            if (gn < N && k_base + PM_K_STEP <= K) {
                cp_async_cg_16(dst, &B[(unsigned long long)gn * K + k_base]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < PM_K_STEP; e++) {
                    unsigned int gk = k_base + e;
                    dst[e] = (gn < N && gk < K) ? B[(unsigned long long)gn * K + gk] : 0;
                }
            }
        }
        cp_async_commit();
    };

    // LUT-dequant just-arrived raw B for `stage` into the [k][n] BF16 layout
    // the MMA reads. Raw is [n][k]; MMA wants [k][n]. No scale here (folded on
    // the FP32 accumulator at the block boundary).
    auto dequant_B = [&](unsigned int stage) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < PM_K_STEP * PM_N_TILE; idx += PM_THREADS) {
            unsigned int k = idx / PM_N_TILE;     // 0..15
            unsigned int n = idx % PM_N_TILE;     // 0..127
            unsigned char wb = smem_Braw[stage][n][k];
            smem_B[stage][k][n] = __float2bfloat16(E4M3_LUT[wb]);
        }
    };

    // ── Software-pipelined main loop ──
    // Prologue: prefetch K-step 0 into stage 0.
    prefetch(0, 0);
    unsigned int k_step_in_block = 0;

    for (unsigned int step = 0; step < n_steps; step++) {
        unsigned int cur = step & 1;
        unsigned int nxt = (step + 1) & 1;

        // Prefetch the NEXT K-step before consuming the current one, so its
        // global loads overlap the dequant + MMA below.
        if (step + 1 < n_steps) {
            prefetch(step + 1, nxt);
            // Two groups committed (current + next); wait until ≤1 remains in
            // flight → the CURRENT stage's copy has completed.
            cp_async_wait_group<1>();
        } else {
            // Last step: only the current group is in flight.
            cp_async_wait_group<0>();
        }
        __syncthreads();   // raw B for `cur` is now resident for all threads

        dequant_B(cur);
        __syncthreads();   // smem_B[cur] BF16 fully written before MMA reads it

        pm_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                     inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();   // done reading smem_*[cur]; safe for next reuse

        // K_BLOCK boundary: fold scaled inner into outer, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = (step * PM_K_STEP) / PM_FP8_BLOCK;
            const float scale = block_scale[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM_N_TILES_PER_WARP; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
                inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }

    // Fold any incomplete trailing K_BLOCK (only when K % FP8_BLOCK != 0).
    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / PM_FP8_BLOCK;
        const float scale = block_scale[n_block * k_blocks + k_block];
        #pragma unroll
        for (int i = 0; i < PM_N_TILES_PER_WARP; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }

    // ── Store C tile: f32 outer accumulators → BF16 output ──
    #pragma unroll
    for (int n_tile = 0; n_tile < PM_N_TILES_PER_WARP; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < M && col0 < N) C[row0 * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
        if (row0 < M && col1 < N) C[row0 * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        if (row1 < M && col0 < N) C[row1 * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
        if (row1 < M && col1 < N) C[row1 * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
    }
}
