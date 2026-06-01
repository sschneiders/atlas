#!/bin/bash
cd /workspace/atlas
. "$HOME/.cargo/env"
export SCALE_HOME="$HOME/scale171/scale-1.7.1-Linux"
export ATLAS_TARGET_HW=strix
export ATLAS_TARGET_MODEL=qwen3.6-27b
export ATLAS_TARGET_QUANT=nvfp4
export CUDA_PATH="$SCALE_HOME/targets/gfx1151"
export CUDA_HOME="$SCALE_HOME/targets/gfx1151"
export PATH="$SCALE_HOME/targets/gfx1151/bin:/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:$SCALE_HOME/targets/gfx1151/lib:$LD_LIBRARY_PATH"
export CUDARC_CUDA_VERSION=12080
echo "=== build start $(date) ==="
echo "nvcc resolves to: $(command -v nvcc)"
rm -rf target/release/build/atlas-kernels-* target/release/build/spark-storage-*
cargo build --release -p spark-server --no-default-features --features cuda
echo "BUILD_EXIT=$?"
echo "=== build end $(date) ==="
