// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMV Fused — dual projection + silu-input variants.
//
// Reduces shared expert kernels from 4 to 2 per layer (saves 96 launches total):
//   Before: gate (1) + up (1) + silu_mul (1) + down (1) = 4 per layer × 48 = 192
//   After:  gate_up_dual (1) + silu_down (1) = 2 per layer × 48 = 96
//
// w4a16_gemv_dual: blockIdx.z selects projection 0 vs 1.
//   Both projections share the same BF16 input A[1, K].
//   Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
//
// w4a16_gemv_silu_input: reads gate_out + up_out BF16 vectors, computes
//   silu(gate)*up inline as activation, then GEMV with NVFP4 down weights.
//   Eliminates separate silu_mul kernel entirely.
//   Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
__device__ __forceinline__ unsigned char scl_enc_fp8(float v) {
    if (v != v) return 0x7F;                 // NaN
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}

#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_FUSED_W4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ── W4A16 GEMV Dual Projection ──
//
// blockIdx.z = 0: first projection (gate), blockIdx.z = 1: second (up).
// Both read same shared BF16 input A[1, K] with different NVFP4 weights.
// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual(
    const __nv_bfloat16* __restrict__ A,           // [1, K] shared input
    const unsigned char* __restrict__ B1_packed,    // [N, K/2] proj 0 weights
    const unsigned char* __restrict__ B1_scale,     // [N, K/GROUP_SIZE] proj 0
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,                 // [1, N] proj 0 output
    const unsigned char* __restrict__ B2_packed,    // [N, K/2] proj 1 weights
    const unsigned char* __restrict__ B2_scale,     // [N, K/GROUP_SIZE] proj 1
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,                 // [1, N] proj 1 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? B1_packed : B2_packed;
    const unsigned char* B_scale = proj == 0 ? B1_scale : B2_scale;
    float scale2 = proj == 0 ? scale2_1 : scale2_2;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    // Process 32 K-values per iteration: one 16-byte (uint4) weight load
    // holds 32 packed FP4 weights, paired with 4× uint4 BF16 activation loads.
    // Wider weight loads issue fewer, larger memory transactions — the key win
    // for this weight-bound (W4A16) GEMV on RDNA3.5 where memory-level
    // parallelism, not compute, is the limiter. 32 spans exactly 2 GROUP_SIZE=16
    // scale groups. K is a multiple of 32 for this model (5120 / 17408).
    const unsigned int K32 = K / 32;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k32 = lane; k32 < K32; k32 += threads_per_out) {
        const unsigned int base_k = k32 * 32;

        // 32 BF16 activations = 4× uint4 (each uint4 = 8 BF16)
        uint4 a0 = ((const uint4*)A)[k32 * 4 + 0];
        uint4 a1 = ((const uint4*)A)[k32 * 4 + 1];
        uint4 a2 = ((const uint4*)A)[k32 * 4 + 2];
        uint4 a3 = ((const uint4*)A)[k32 * 4 + 3];
        const unsigned int a_raw[16] = {
            a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
            a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};

        // 16 packed weight bytes (= 32 FP4) as one uint4
        uint4 w_packed = *(const uint4*)(
            B_packed + (unsigned long long)n * half_K + (unsigned long long)k32 * 16);
        const unsigned int wp[4] = {w_packed.x, w_packed.y, w_packed.z, w_packed.w};

        // Two scale groups (first 16 K-values, second 16 K-values)
        unsigned int sg0 = base_k / GROUP_SIZE;
        unsigned char sb0 = B_scale[(unsigned long long)n * num_groups + sg0];
        unsigned char sb1 = B_scale[(unsigned long long)n * num_groups + sg0 + 1];
        float scale_a = scl_fp8(sb0) * scale2;
        float scale_b = scl_fp8(sb1) * scale2;

        #pragma unroll
        for (int q = 0; q < 4; q++) {
            unsigned int packed4 = wp[q];
            // q in {0,1} -> first scale group; {2,3} -> second
            float scale = (q < 2) ? scale_a : scale_b;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
                float w_lo = s_lut[byte_val & 0xF] * scale;
                float w_hi = s_lut[byte_val >> 4] * scale;

                unsigned int araw = a_raw[q * 4 + b];
                __nv_bfloat16 a_lo, a_hi;
                *(unsigned short*)&a_lo = (unsigned short)(araw & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(araw >> 16);
                acc += __bfloat162float(a_lo) * w_lo;
                acc += __bfloat162float(a_hi) * w_hi;
            }
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFULL, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ── W4A16 GEMV with SiLU-fused Input ──
//
// Reads gate_out[K] and up_out[K] BF16, computes silu(gate)*up inline
// as the activation, then GEMV with NVFP4 down weights.
// Eliminates the separate silu_mul kernel entirely.
// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_silu_input(
    const __nv_bfloat16* __restrict__ gate_out,    // [1, K] gate proj output
    const __nv_bfloat16* __restrict__ up_out,      // [1, K] up proj output
    const unsigned char* __restrict__ B_packed,     // [N, K/2] down weights
    const unsigned char* __restrict__ B_scale,      // [N, K/GROUP_SIZE]
    const float scale2,
    __nv_bfloat16* __restrict__ C,                  // [1, N] output
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    // Process 32 K-values per iteration (see w4a16_gemv_dual): one uint4 weight
    // load (32 FP4) + 4× uint4 gate + 4× uint4 up activation loads. Wider weight
    // loads improve memory-level parallelism on this weight-bound down-proj GEMV.
    const unsigned int K32 = K / 32;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k32 = lane; k32 < K32; k32 += threads_per_out) {
        const unsigned int base_k = k32 * 32;

        uint4 g0 = ((const uint4*)gate_out)[k32 * 4 + 0];
        uint4 g1 = ((const uint4*)gate_out)[k32 * 4 + 1];
        uint4 g2 = ((const uint4*)gate_out)[k32 * 4 + 2];
        uint4 g3 = ((const uint4*)gate_out)[k32 * 4 + 3];
        uint4 u0 = ((const uint4*)up_out)[k32 * 4 + 0];
        uint4 u1 = ((const uint4*)up_out)[k32 * 4 + 1];
        uint4 u2 = ((const uint4*)up_out)[k32 * 4 + 2];
        uint4 u3 = ((const uint4*)up_out)[k32 * 4 + 3];
        const unsigned int g_raw[16] = {
            g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w,
            g2.x, g2.y, g2.z, g2.w, g3.x, g3.y, g3.z, g3.w};
        const unsigned int u_raw[16] = {
            u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w,
            u2.x, u2.y, u2.z, u2.w, u3.x, u3.y, u3.z, u3.w};

        uint4 w_packed = *(const uint4*)(
            B_packed + (unsigned long long)n * half_K + (unsigned long long)k32 * 16);
        const unsigned int wp[4] = {w_packed.x, w_packed.y, w_packed.z, w_packed.w};

        unsigned int sg0 = base_k / GROUP_SIZE;
        float scale_a = scl_fp8(B_scale[(unsigned long long)n * num_groups + sg0]) * scale2;
        float scale_b = scl_fp8(B_scale[(unsigned long long)n * num_groups + sg0 + 1]) * scale2;

        #pragma unroll
        for (int q = 0; q < 4; q++) {
            unsigned int packed4 = wp[q];
            float scale = (q < 2) ? scale_a : scale_b;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
                float w_lo = s_lut[byte_val & 0xF] * scale;
                float w_hi = s_lut[byte_val >> 4] * scale;

                unsigned int graw = g_raw[q * 4 + b];
                unsigned int uraw = u_raw[q * 4 + b];
                __nv_bfloat16 g_lo, g_hi, u_lo, u_hi;
                *(unsigned short*)&g_lo = (unsigned short)(graw & 0xFFFF);
                *(unsigned short*)&g_hi = (unsigned short)(graw >> 16);
                *(unsigned short*)&u_lo = (unsigned short)(uraw & 0xFFFF);
                *(unsigned short*)&u_hi = (unsigned short)(uraw >> 16);
                float gf_lo = __bfloat162float(g_lo);
                float gf_hi = __bfloat162float(g_hi);

                // SiLU(gate) * up
                float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
                float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

                acc += a_lo * w_lo;
                acc += a_hi * w_hi;
            }
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFULL, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}
