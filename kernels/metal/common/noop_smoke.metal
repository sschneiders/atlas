// SPDX-License-Identifier: AGPL-3.0-only
//
// Placeholder kernel for Phase 2/3: lets the build pipeline (xcrun metal
// → metallib → include_bytes!) round-trip end-to-end before any real
// kernel ports land. Phase 7 replaces this with rms_norm / gemv /
// attention etc. The noop is referenced by the smoke test in
// `crates/spark-runtime/src/metal_backend.rs` once the runtime backend
// is implemented (Phase 4).

#include <metal_stdlib>
using namespace metal;

kernel void noop_smoke(
    device float *out [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        out[gid] = 0.0f;
    }
}
