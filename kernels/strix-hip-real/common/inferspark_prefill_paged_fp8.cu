// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — FP8 E4M3 KV cache variant. HIP/gfx1151 port.
//
// Reads FP8 K/V from paged cache, dequantizes to BF16 in shared memory, then
// runs the WMMA Flash Attention compute. K/V use distinct dequant scales
// (k_scale / v_scale) selected by comparing the cache pointer to K_cache.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Software E4M3 -> FP32 decode (standard E4M3; matches the NVIDIA reference).
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_storage_t b, float scale) {
    return __float2bfloat16(scl_fp8((unsigned char)b) * scale);
}

// FP8 tile loader: manual load + dequant + 16-byte smem store.
// Scale chosen per-call: k_scale for K loads, v_scale for V loads.
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const float _sc = ((const void*)(cache) == (const void*)K_cache) ? k_scale : v_scale; \
        const unsigned int _cpr = HDIM / 8; \
        for (unsigned int _i = (t); _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                __nv_bfloat16 _v[8]; \
                for (int _j = 0; _j < 8; _j++) \
                    _v[_j] = fp8_to_bf16(_base[_j], _sc); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_fp8
#define K_CACHE_TYPE const void* __restrict__
#define V_CACHE_TYPE const void* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const float k_scale \
    , const float v_scale \
    , const unsigned long long fp8_cache_stride
#define KERNEL_PREAMBLE /* nothing */

#include "prefill_paged_compute.cuh"
