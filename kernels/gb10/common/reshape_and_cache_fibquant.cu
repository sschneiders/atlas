// SPDX-License-Identifier: AGPL-3.0-only

// FibQuant KV cache write kernel (arXiv:2605.11478).
//
// Input K/V are already Walsh-Hadamard rotated (the host write path applies
// `wht_bf16` before this kernel — same bookend as the turbo dtypes, gated by
// `is_wht_rotated()`). This kernel then: computes the per-vector L2 norm,
// normalizes, and quantizes each k-dim block to the nearest shared codebook
// codeword (the spherical-Beta f_{d,k} optimal codebook). Stores:
//   { bf16 norm (2 B), head_dim/FIB_K 1-byte indices } per (token, kv_head).
//
// Memory layout per block (K and V separately):
//   vector (token t, kv_head h) at offset (t*num_kv_heads + h) * (2 + head_dim/FIB_K)
//   bf16 norm at that offset; indices immediately after.
//
// Grid: (num_tokens * num_kv_heads, 1, 1)   Block: (128, 1, 1)
// One CTA per vector: the first (head_dim/FIB_K) threads each own one k-block;
// they cooperatively reduce the vector L2 norm, then each searches the codebook.
//
// The codebook is NOT embedded: it is built on the host from `atlas-quant` for
// the layer's actual head_dim (spherical-Beta f_{d,k}), uploaded once at model
// init, and passed in as the trailing `fibq_codebook` device pointer. Only the
// geometry (FIB_K=4, FIB_N=256) is compile-time — so any head_dim works.

#include <cuda_bf16.h>

#ifndef FIB_K
#define FIB_K 4
#endif
#ifndef FIB_N
#define FIB_N 256
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Block-reduce sum of `val` across the first `n` threads (n ≤ blockDim.x).
__device__ __forceinline__ float fib_block_sum(float val, unsigned int tid) {
    __shared__ float sm[128];
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    if ((tid & (WARP_SIZE - 1)) == 0) sm[tid >> 5] = val;
    __syncthreads();
    unsigned int wid = tid >> 5;
    if (wid == 0) {
        unsigned int nwarps = blockDim.x >> 5;
        float v = (tid < nwarps) ? sm[tid] : 0.0f;
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (tid == 0) sm[0] = v;
    }
    __syncthreads();
    return sm[0];
}

extern "C" __global__ void reshape_and_cache_flash_fibquant(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const float* __restrict__ fibq_codebook
) {
    const unsigned int vec_idx = blockIdx.x;
    const unsigned int token_idx = vec_idx / num_kv_heads;
    const unsigned int kv_head = vec_idx % num_kv_heads;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int nblocks = head_dim / FIB_K;   // 64 (hd256) / 32 (hd128)
    const unsigned int payload = 2u + nblocks;        // bytes per vector

    // Stage the 4 KB codebook to shared memory (every thread indexes it
    // data-dependently during the search — __constant__ would serialise).
    __shared__ float cb_smem[FIB_N * FIB_K];
    for (unsigned int i = tid; i < FIB_N * FIB_K; i += blockDim.x)
        cb_smem[i] = fibq_codebook[i];
    __syncthreads();

    // Each of the first `nblocks` threads loads its k-block of WHT(K)/WHT(V).
    float kf[FIB_K], vf[FIB_K];
    float ksq = 0.0f, vsq = 0.0f;
    if (tid < nblocks) {
        const __nv_bfloat16* kp = key + (unsigned long long)token_idx * key_stride
                                 + (unsigned long long)kv_head * head_dim + tid * FIB_K;
        const __nv_bfloat16* vp = value + (unsigned long long)token_idx * value_stride
                                 + (unsigned long long)kv_head * head_dim + tid * FIB_K;
        #pragma unroll
        for (int j = 0; j < FIB_K; j++) {
            kf[j] = __bfloat162float(kp[j]);
            vf[j] = __bfloat162float(vp[j]);
            ksq += kf[j] * kf[j];
            vsq += vf[j] * vf[j];
        }
    } else {
        #pragma unroll
        for (int j = 0; j < FIB_K; j++) { kf[j] = 0.0f; vf[j] = 0.0f; }
    }

    float knorm = sqrtf(fib_block_sum(ksq, tid));
    float vnorm = sqrtf(fib_block_sum(vsq, tid));
    float kinv = (knorm > 1e-20f) ? (1.0f / knorm) : 0.0f;
    float vinv = (vnorm > 1e-20f) ? (1.0f / vnorm) : 0.0f;

    const unsigned int block_idx = (unsigned int)(slot / (long long)block_size);
    const unsigned int block_offset = (unsigned int)(slot % (long long)block_size);
    const unsigned long long vec_off =
        (unsigned long long)block_idx * block_stride_bytes
        + ((unsigned long long)block_offset * num_kv_heads + kv_head) * payload;

    if (tid < nblocks) {
        // Quantize the unit-vector block against the unit-ball codebook.
        float ku[FIB_K], vu[FIB_K];
        #pragma unroll
        for (int j = 0; j < FIB_K; j++) { ku[j] = kf[j] * kinv; vu[j] = vf[j] * vinv; }

        unsigned int kbest = 0, vbest = 0;
        float kbest_d = 1e30f, vbest_d = 1e30f;
        for (unsigned int c = 0; c < FIB_N; c++) {
            const float* cw = &cb_smem[c * FIB_K];
            float kd = 0.0f, vd = 0.0f;
            #pragma unroll
            for (int j = 0; j < FIB_K; j++) {
                float dk = ku[j] - cw[j]; kd += dk * dk;
                float dv = vu[j] - cw[j]; vd += dv * dv;
            }
            if (kd < kbest_d) { kbest_d = kd; kbest = c; }
            if (vd < vbest_d) { vbest_d = vd; vbest = c; }
        }
        k_cache[vec_off + 2u + tid] = (unsigned char)kbest;
        v_cache[vec_off + 2u + tid] = (unsigned char)vbest;
    }

    // Thread 0 writes the bf16 norm header for K and V.
    if (tid == 0) {
        *((__nv_bfloat16*)(k_cache + vec_off)) = __float2bfloat16(knorm);
        *((__nv_bfloat16*)(v_cache + vec_off)) = __float2bfloat16(vnorm);
    }
}
