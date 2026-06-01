#!/bin/bash
cd /workspace/atlas
SH=$HOME/scale171/scale-1.7.1-Linux
export ATLAS_FORCE_GLOBAL_GDN=1
export ATLAS_W4A16_VARIANT=v1
export ATLAS_NO_FP8_PREDEQUANT=1
export ATLAS_DIAG_GEMMA4=1
export ATLAS_DUMP_LAYER_NORM=1
export ATLAS_DUMP_GDN=1
# export ATLAS_W4A16_NOPIPE=1
# SCALE bundles its own ROCm 7.2.3 runtime (the CWSR queue-create fix lives here).
# Put SCALE libs FIRST so /opt/rocm cannot shadow the fixed libhsa-runtime64.
export LD_LIBRARY_PATH="$SH/targets/gfx1151/lib:$SH/lib"
export PATH="$SH/targets/gfx1151/bin:$PATH"
echo "=== serve start $(date) ==="
exec target/release/spark serve "${1:-Qwen/Qwen3.6-27B-FP8}" --port 8081 --max-seq-len 4096 --gpu-memory-utilization 0.70 --kv-cache-dtype bf16 --kv-high-precision-layers max --max-batch-size 4
