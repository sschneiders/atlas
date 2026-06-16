// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — FibQuant KV cache variant.
//
// Reads {bf16 norm, 1-byte codebook indices} per vector from the paged cache,
// gathers `codebook[index] × norm` into BF16 shared-memory tiles, then runs
// Flash Attention over contiguous BF16 Q (the shared `prefill_paged_compute.cuh`
// — only the K/V load path differs from the FP8 variant). Q is WHT-rotated and
// the output iWHT'd by the host bookends (same as the turbo dtypes).
//
// Grid: (num_q_heads, ceil(q_len/BR), 1)  Block: (128 or 256, 1, 1)
// v1 embeds the hd=256 codebook (A3B target).

#include <cuda_bf16.h>
#include "fibquant_codebook_256.cuh"

// Stage the 4 KB FibQuant codebook to shared memory once per CTA so the
// data-dependent gather in LOAD_KV_TILE reads from smem, not __constant__.
// Declared `extern` to avoid clashing with the compute header's smem layout:
// the compute header uses dynamically-sized shared memory, and this static
// smem is separate.
#define KERNEL_PREAMBLE \
    __shared__ float fibq_cb_smem[FIB_N * FIB_K]; \
    for (unsigned int _i = threadIdx.x; _i < FIB_N * FIB_K; _i += blockDim.x) \
        fibq_cb_smem[_i] = FIB_CODEBOOK[_i]; \
    __syncthreads();

// FibQuant tile loader: gather `codebook[index] × norm` → BF16 into smem.
// `_col` is the element offset within the head_dim vector (a multiple of 8);
// 8 elements = (8 / FIB_K) codebook blocks. The vector's norm (bf16) is at the
// start of its {norm, indices} payload.
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned int _payload = 2u + head_dim / FIB_K; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const unsigned char* _vec = (const unsigned char*)(cache) \
                    + (unsigned long long)_pb * fibq_block_stride \
                    + ((unsigned long long)_bo * num_kv_heads + (kvh)) * _payload; \
                float _norm = __bfloat162float(*((const __nv_bfloat16*)_vec)); \
                const unsigned char* _idx = _vec + 2; \
                __nv_bfloat16 _v[8]; \
                for (int _b = 0; _b < 8 / FIB_K; _b++) { \
                    const float* _cw = &fibq_cb_smem[(unsigned int)_idx[_col / FIB_K + _b] * FIB_K]; \
                    for (int _j = 0; _j < FIB_K; _j++) \
                        _v[_b * FIB_K + _j] = __float2bfloat16(_cw[_j] * _norm); \
                } \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_fibquant
#define K_CACHE_TYPE const void* __restrict__
#define V_CACHE_TYPE const void* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const unsigned long long fibq_block_stride

#include "prefill_paged_compute.cuh"
