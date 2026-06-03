// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention for HDIM=512 (Gemma-4 full-attention) — BF16 KV.
// HIP/gfx1151 port. cp.async replaced by synchronous uint4 smem copies.
//
// Grid: (num_q_heads, ceil(q_len/32), 1)   Block: (256, 1, 1)
// Required dynamic shared memory: 105,344 bytes (Q/K/V 32768 each + P 2560 +
//   S 4096 + m/l 256 + resc 128). The WMMA port adds an FP32 score-staging
//   buffer (smem_S) and a per-row rescale buffer (smem_resc) vs the NVIDIA
//   register-softmax version's 101,120 bytes — caller .shared_mem() MUST be
//   raised to 105,344. @human-review: confirm the dispatch passes 105,344 and
//   that gfx1151 permits >100 KB dynamic LDS for this (compile-only) Gemma-4
//   path; the dense Qwen3.6-27B model uses HDIM=256 (prefill_paged_compute.cuh).

#include <cuda_bf16.h>

// 1D BF16 tile loader: synchronous uint4 copy paged cache → smem (row stride = HDIM_512).
#define LOAD_KV_TILE_512(cache, bt, smem_ptr, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM_512 / 8; \
        const unsigned long long _ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _i = (t); _i < TILE_CHUNKS_512; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _gm = (const void*)( \
                    (cache) + _pb * _ps + _bo * _rs + (kvh) * head_dim + _col); \
                *((uint4*)&(smem_ptr)[_row * HDIM_512 + _col]) = *((const uint4*)_gm); \
            } else { \
                *((uint4*)&(smem_ptr)[_row * HDIM_512 + _col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_512
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d
#define KERNEL_PREAMBLE /* nothing */

#include "prefill_paged_compute_512.cuh"
