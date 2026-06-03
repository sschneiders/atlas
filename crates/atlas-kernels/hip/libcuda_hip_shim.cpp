// SPDX-License-Identifier: AGPL-3.0-only
//
// libcuda → HIP shim. Re-exports the exact 33 CUDA driver-API symbols the
// Atlas spark binary imports (via cudarc) and implements each over HIP/ROCm.
// Built as `libcuda.so` and placed FIRST on the loader path so the unchanged
// cudarc runtime drives AMD GPUs natively — no SCALE.
//
// CUDA driver ABI ↔ HIP type compatibility:
//   CUdeviceptr (u64) ↔ hipDeviceptr_t (void*)   — cast
//   CUstream/CUmodule/CUfunction/CUevent/CUgraph(Exec) ↔ hip* — opaque ptrs
//   CUresult ↔ hipError_t — success==0 matches; error enums differ but cudarc
//   checks success and formats via cuGetErrorString (mapped to hipGetErrorString).
#include <hip/hip_runtime.h>
#include <cstring>

typedef unsigned long long CUdeviceptr;
extern "C" {

// ── init / context ────────────────────────────────────────────────────
int cuCtxGetCurrent(void** pctx)            { return hipCtxGetCurrent((hipCtx_t*)pctx); }
int cuCtxSetCurrent(void* ctx)              { return hipCtxSetCurrent((hipCtx_t)ctx); }

// ── errors ────────────────────────────────────────────────────────────
int cuGetErrorName(int err, const char** s)   { *s = hipGetErrorName((hipError_t)err);   return 0; }
int cuGetErrorString(int err, const char** s) { *s = hipGetErrorString((hipError_t)err); return 0; }

// ── memory ────────────────────────────────────────────────────────────
int cuMemAlloc_v2(CUdeviceptr* dptr, size_t n)      { return hipMalloc((void**)dptr, n); }
int cuMemFree_v2(CUdeviceptr dptr)                  { return hipFree((void*)dptr); }
int cuMemAllocHost_v2(void** pp, size_t n)          { return hipHostMalloc(pp, n, 0); }
int cuMemFreeHost(void* p)                          { return hipHostFree(p); }
int cuMemAllocManaged(CUdeviceptr* dptr, size_t n, unsigned flags)
                                                    { return hipMallocManaged((void**)dptr, n, flags); }
int cuMemGetInfo_v2(size_t* free, size_t* total)    { return hipMemGetInfo(free, total); }

int cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src, size_t n, void* s)
                              { return hipMemcpyHtoDAsync((hipDeviceptr_t)dst, (void*)src, n, (hipStream_t)s); }
int cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr src, size_t n, void* s)
                              { return hipMemcpyDtoHAsync(dst, (hipDeviceptr_t)src, n, (hipStream_t)s); }
int cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t n, void* s)
                              { return hipMemcpyDtoDAsync((hipDeviceptr_t)dst, (hipDeviceptr_t)src, n, (hipStream_t)s); }
int cuMemsetD8Async(CUdeviceptr dst, unsigned char uc, size_t n, void* s)
                              { return hipMemsetD8Async((hipDeviceptr_t)dst, uc, n, (hipStream_t)s); }
int cuMemsetD32Async(CUdeviceptr dst, unsigned int ui, size_t n, void* s)
                              { return hipMemsetD32Async((hipDeviceptr_t)dst, ui, n, (hipStream_t)s); }

// ── modules / kernels ─────────────────────────────────────────────────
int cuModuleLoadData(void** m, const void* image)          { return hipModuleLoadData((hipModule_t*)m, image); }
int cuModuleGetFunction(void** f, void* m, const char* nm) { return hipModuleGetFunction((hipFunction_t*)f, (hipModule_t)m, nm); }
int cuModuleUnload(void* m)                                { return hipModuleUnload((hipModule_t)m); }
// On NVIDIA this opts a module function into >48KB dynamic shared mem. On AMD
// the LDS is sized from hipModuleLaunchKernel's shared-mem arg (up to the
// RDNA 64KB cap), so no opt-in is needed — success no-op.
int cuFuncSetAttribute(void* f, int attr, int val)         { (void)f;(void)attr;(void)val; return 0; }
int cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz,
                   unsigned shmem, void* stream, void** params, void** extra) {
  return hipModuleLaunchKernel((hipFunction_t)f, gx, gy, gz, bx, by, bz,
                               shmem, (hipStream_t)stream, params, extra);
}

// ── streams ───────────────────────────────────────────────────────────
int cuStreamCreate(void** s, unsigned flags)        { return hipStreamCreateWithFlags((hipStream_t*)s, flags); }
int cuStreamSynchronize(void* s)                    { return hipStreamSynchronize((hipStream_t)s); }
int cuStreamWaitEvent(void* s, void* e, unsigned f) { return hipStreamWaitEvent((hipStream_t)s, (hipEvent_t)e, f); }
int cuStreamBeginCapture(void* s, int mode)         { return hipStreamBeginCapture((hipStream_t)s, (hipStreamCaptureMode)mode); }
int cuStreamEndCapture(void* s, void** pgraph)      { return hipStreamEndCapture((hipStream_t)s, (hipGraph_t*)pgraph); }

// ── events ────────────────────────────────────────────────────────────
int cuEventCreate(void** e, unsigned flags) { return hipEventCreateWithFlags((hipEvent_t*)e, flags); }
int cuEventDestroy_v2(void* e)              { return hipEventDestroy((hipEvent_t)e); }
int cuEventRecord(void* e, void* s)         { return hipEventRecord((hipEvent_t)e, (hipStream_t)s); }
int cuEventSynchronize(void* e)             { return hipEventSynchronize((hipEvent_t)e); }

// ── CUDA graphs ───────────────────────────────────────────────────────
// cudarc's cuGraphInstantiate (legacy arity): (exec*, graph, errNode*, logBuf, bufSize)
int cuGraphInstantiate(void** pexec, void* graph, void** errNode, char* logBuf, size_t bufSize) {
  (void)errNode; (void)logBuf; (void)bufSize;
  return hipGraphInstantiate((hipGraphExec_t*)pexec, (hipGraph_t)graph, nullptr, nullptr, 0);
}
int cuGraphLaunch(void* exec, void* s)  { return hipGraphLaunch((hipGraphExec_t)exec, (hipStream_t)s); }
int cuGraphExecDestroy(void* exec)      { return hipGraphExecDestroy((hipGraphExec_t)exec); }
int cuGraphDestroy(void* graph)         { return hipGraphDestroy((hipGraph_t)graph); }

int cuGraphInstantiateWithFlags(void** pexec, void* graph, unsigned long long flags){ return hipGraphInstantiateWithFlags((hipGraphExec_t*)pexec,(hipGraph_t)graph,flags); }

} // extern "C"
