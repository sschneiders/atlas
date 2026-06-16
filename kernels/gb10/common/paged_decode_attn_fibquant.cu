// SPDX-License-Identifier: AGPL-3.0-only

// Paged Decode Attention — FibQuant (WHT + spherical-Beta vector codebook).
//
// Mirror of paged_decode_attn_turbo4.cu with the K/V read path replaced: FibQuant
// stores {bf16 norm, 1-byte codebook indices} per vector (no separate scale
// section), so the dequant is a codebook gather × norm instead of nibble × FP8
// group scale. Q is loaded already WHT-rotated and the output is iWHT'd by the
// host bookends (same as turbo — `is_wht_rotated()` is true), so this kernel
// changes ONLY the K/V load+dequant lines vs the FP8/turbo structure.
//
// Memory layout per block (K or V separately):
//   vector (token t, head h) at (t*num_kv_heads + h) * (2 + head_dim/FIB_K)
//   bf16 norm at that offset; head_dim/FIB_K index bytes immediately after.
//
// Grid: (num_q_heads, num_seqs, 1)   Block: (256, 1, 1)
// v1 embeds the hd=256 codebook (A3B target).

#include <cuda_bf16.h>
#include "fibquant_codebook_256.cuh"

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define NUM_WARPS 8
#define BC 4

__device__ __forceinline__ void unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// Gather `VEC_BF16` reconstructed elements for one lane: the vector's bf16 norm
// (shared across lanes) × the codebook codewords selected by this lane's
// `VEC_BF16/FIB_K` indices. `payload` points at the vector's norm byte.
__device__ __forceinline__ void fibquant_dequant(
    const unsigned char* payload,
    unsigned int lane_id,
    const float* cb_smem,
    float* out
) {
    float norm = __bfloat162float(*((const __nv_bfloat16*)payload));
    const unsigned char* idx = payload + 2;
    const unsigned int nidx = VEC_BF16 / FIB_K;
    const unsigned int base = lane_id * nidx;
    #pragma unroll
    for (int b = 0; b < (int)nidx; b++) {
        const float* cw = &cb_smem[(unsigned int)idx[base + b] * FIB_K];
        #pragma unroll
        for (int j = 0; j < FIB_K; j++) out[b * FIB_K + j] = cw[j] * norm;
    }
}

extern "C" __global__ void paged_decode_attn_fibquant(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;
    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Stage the 4 KB FibQuant codebook to shared memory (data-dependent gather).
    __shared__ float cb_smem[FIB_N * FIB_K];
    for (unsigned int i = tid; i < FIB_N * FIB_K; i += blockDim.x)
        cb_smem[i] = FIB_CODEBOOK[i];
    __syncthreads();

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    const unsigned int nblocks = head_dim / FIB_K;
    const unsigned int payload = 2u + nblocks;
    const unsigned int token_payload_stride = num_kv_heads * payload;
    const unsigned int kv_payload_off = kv_head * payload;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (already WHT-rotated by the host bookend).
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16 / 2; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    unsigned int chunk_size = (seq_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * block_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * block_stride_bytes;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        for (; processed < aligned_count; processed += BC) {
            float k_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                unsigned int p = block_offset + processed + b;
                const unsigned char* kp = k_block + p * token_payload_stride + kv_payload_off;
                fibquant_dequant(kp, lane_id, cb_smem, k_vals[b]);
            }

            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) dot += q_reg[i] * k_vals[b][i];
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot * inv_sqrt_d;
            }

            float m_new = m;
            #pragma unroll
            for (int b = 0; b < BC; b++) m_new = fmaxf(m_new, scores[b]);
            float exp_old = __expf(m - m_new);
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) o_reg[i] *= exp_old;
            l *= exp_old;

            float exp_factors[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l += exp_factors[b];
            }
            m = m_new;

            float v_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                if (exp_factors[b] > 1e-3f) {
                    unsigned int p = block_offset + processed + b;
                    const unsigned char* vp = v_block + p * token_payload_stride + kv_payload_off;
                    fibquant_dequant(vp, lane_id, cb_smem, v_vals[b]);
                } else {
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) v_vals[b][i] = 0.0f;
                }
            }
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) o_reg[i] += ef * v_vals[b][i];
            }
        }

        for (; processed < batch_count; processed++) {
            unsigned int p = block_offset + processed;
            const unsigned char* kp = k_block + p * token_payload_stride + kv_payload_off;
            float k_tmp[VEC_BF16];
            fibquant_dequant(kp, lane_id, cb_smem, k_tmp);

            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) dot += q_reg[i] * k_tmp[i];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            float v_tmp[VEC_BF16] = {0};
            if (exp_new > 1e-3f) {
                const unsigned char* vp = v_block + p * token_payload_stride + kv_payload_off;
                fibquant_dequant(vp, lane_id, cb_smem, v_tmp);
            }
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
            m = m_new;
        }

        pos += batch_count;
    }

    // Tree-based inter-warp reduction (identical to FP8/turbo).
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) smem_o[warp_id][vec_offset_bf16 + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (unsigned int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset_bf16 + i] =
                        smem_o[warp_id][vec_offset_bf16 + i] * scale_me +
                        smem_o[other][vec_offset_bf16 + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_BF16 / 2; i++) {
            float v0 = smem_o[0][vec_offset_bf16 + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset_bf16 + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}
