# Native ROCm/HIP build path (no SCALE) — gfx1151 / Strix Halo

Goal: run Atlas on AMD GPUs **without SCALE**, by compiling the existing CUDA
kernels with `hipcc` and replacing SCALE's `libcuda.so` with a thin HIP shim.
Status: **build path validated; full-model serve pending build-system wiring +
a GPU window.** See `../../../.claude` memory `project_hip_port_strix` for the
blow-by-blow.

## Two layers, both validated on silicon (2026-06-03)

### 1. Kernels → hipcc
`hipcc` compiles the unmodified `.cu` directly using the CUDA→HIP **compat
headers** in `compat/` (forward `cuda_*.h` → `hip/*`, alias `__nv_*` types) +
force-included `hip/hip_runtime.h` + a tree-wide warp-mask widen sed (HIP
`__shfl_*_sync` need a 64-bit mask) + `__activemask()` captured as 64-bit.

Result on the full kernel set: **72/92 compile clean, 0 misc failures.** The
remaining 20 are tensor-core kernels using NVIDIA `mma.sync` PTX — hand-ported
to AMD WMMA (`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`). `HipTarget` in
`build_target.rs` runs this recipe (`hipcc -x hip --genco`).

WMMA C-fragment layout (empirically mapped on gfx1151): for lane `l`, vgpr
element `e` → output `row = 2*e + (l>>4)`, `col = l & 15`. A-load `a[i]=A[l&15][i]`,
B-load `b[k]=B[k][l&15]`. The ported `w4a16_gemm` (see `ported/`) matches a CPU
dequant-GEMM reference to within 1 bf16 ULP (0/4096 cells worse).

### 2. Runtime → libcuda→HIP shim
The runtime is unchanged (cudarc, `-lcuda`). `libcuda_hip_shim.cpp` re-exports
the **33** CUDA driver symbols the binary imports and maps each to HIP
(`cuModuleLoadData`→`hipModuleLoadData`, `cuLaunchKernel`→`hipModuleLaunchKernel`,
`cuMemAlloc_v2`→`hipMalloc`, graphs/streams/events 1:1). Build it as `libcuda.so`,
put it first on the loader path:
```
hipcc -shared -fPIC libcuda_hip_shim.cpp -o libcuda.so -I/opt/rocm/<ver>/include
```

## Remaining to a running model
1. Port the dense prefill MMA kernels to WMMA (recipe proven on `w4a16_gemm`):
   `dense_gemm_tc`, one `inferspark_prefill` attention kernel, `w8a16_gemm(_t)`.
2. Wire `HipTarget` into `build.rs` (stage `compat/` → `ATLAS_HIP_COMPAT_INCLUDE`,
   build the sed'd source mirror, emit `.co` objects into the registry).
3. `cargo build` spark-server; link/LD against the shim `libcuda.so` + `libamdhip64`.
4. Serve Qwen3.6-27B-dense; validate coherence; measure decode tok/s (target 20).

Decode needs **zero** WMMA ports (all GEMV/attn-decode/SSM-decode kernels are in
the clean/mechanical buckets) — the MMA work only affects prefill/TTFT.
