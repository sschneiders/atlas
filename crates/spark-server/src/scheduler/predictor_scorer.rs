// SPDX-License-Identifier: AGPL-3.0-only

//! HSS-`Predictor`-backed relevance scorer for KVFlash residency (deep
//! recall). Wraps [`spark_storage::Predictor`]: per-block relevance is
//! `Q_proj · A_g` (the per-token low-rank K, max-reduced per block by the
//! `predictor_score` kernel), aggregated as a mean across the attention
//! layers. Implements [`KvFlashScorer`] so the pager's score-driven
//! `reselect` recalls contextually-relevant paged-out chunks instead of
//! dropping them under LRU.
//!
//! Why the HSS Predictor and not a drafter model: Atlas's fused attention
//! kernels don't expose the weight matrix a drafter would need, but the
//! Predictor produces the same Q·K_lowrank relevance signal self-containedly
//! (it owns its projection matrix P + low-rank K store A_g). See
//! `docs/design/kvflash-port.md` "PredictorScorer (deep recall)".
//!
//! Step 3 (this file) handles BF16 KV: the paged-out K bytes are already
//! BF16 and are uploaded + projected directly. FP8 dequant (the A3B's
//! default KV dtype) is Step 4.

use std::ffi::c_void;

use anyhow::Result;

use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kvflash_scorer::KvFlashScorer;
use spark_storage::ModelDims;
use spark_storage::cuda_min::{DeviceBuffer, copy_d_to_h_async, stream_sync};
use spark_storage::predictor::{Predictor, PredictorDims};

/// Low-rank projection dimension. Higher = sharper relevance, more HBM for
/// A_g. 32 matches the HSS default and is ample for chunk-granularity
/// ranking.
pub(super) const PREDICTOR_RANK: usize = 32;
/// Seed for the fixed random projection matrix P (deterministic; any value
/// works — the projection is a lossless Johnson-Lindenstrauss-style sketch).
const PREDICTOR_SEED: u64 = 0x5eed_5eed;

/// `KvFlashScorer` backed by the HSS `Predictor`. Holds the device-side
/// captured Q, the projected Q, the per-layer block-score scratch, and a
/// K-upload scratch. `stream` is stashed from `capture_q` (the model's decode
/// stream) so projection + scoring stay ordered with the model's Q write.
pub struct PredictorScorer {
    predictor: Predictor,
    q_capture: DeviceBuffer, // [num_q_heads, head_dim] BF16 — stashed decode Q
    q_proj: DeviceBuffer,    // [num_q_heads, r] BF16 — projected Q
    block_scores: DeviceBuffer, // [max_blocks] f32 — one layer's per-block scores
    score_host: Vec<f32>,    // host mirror of `block_scores`
    k_scratch: DeviceBuffer, // [num_kv_heads, block_size, head_dim] BF16 — K upload
    stream: u64,
    dims: PredictorDims,
    /// Per logical block: has it been projected into A_g? Grows with the
    /// sequence; the pager's reselect refreshes only `false` resident blocks.
    projected: Vec<bool>,
    /// Expected BF16 K byte count for one layer's block (dtype auto-detect:
    /// a mismatch means FP8/other, handled in Step 4).
    bf16_k_bytes: usize,
}

impl PredictorScorer {
    /// Construct on the scheduler thread (CUDA ctx already bound). Uses the
    /// default stream (0) at build time; `capture_q` later rebinds `stream` to
    /// the model's decode stream for hot-path ordering.
    pub fn new(model: ModelDims) -> Result<Self> {
        Self::new_with_rank(model, PREDICTOR_RANK)
    }

    pub fn new_with_rank(model: ModelDims, rank: usize) -> Result<Self> {
        let dims = PredictorDims {
            num_layers: model.num_layers as usize,
            num_q_heads: model.num_q_heads as usize,
            num_kv_heads: model.num_kv_heads as usize,
            head_dim: model.head_dim as usize,
            r: rank,
            block_size: model.block_size as usize,
            max_blocks: model.max_blocks_per_layer as usize,
        };
        dims.validate()?;
        let stream = 0u64;
        let predictor = Predictor::new_on_stream(stream, dims, PREDICTOR_SEED)?;
        let q_capture = DeviceBuffer::new(dims.num_q_heads * dims.head_dim * 2)?;
        let q_proj = DeviceBuffer::new(dims.num_q_heads * dims.r * 2)?;
        let block_scores = DeviceBuffer::new(dims.max_blocks * 4)?;
        let score_host = vec![0.0f32; dims.max_blocks];
        // K block layout: [num_kv_heads, block_size, head_dim] BF16.
        let k_scratch = DeviceBuffer::new(dims.num_kv_heads * dims.block_size * dims.head_dim * 2)?;
        let bf16_k_bytes = dims.num_kv_heads * dims.block_size * dims.head_dim * 2;
        Ok(Self {
            predictor,
            q_capture,
            q_proj,
            block_scores,
            score_host,
            k_scratch,
            stream,
            dims,
            projected: Vec::new(),
            bf16_k_bytes,
        })
    }

    /// Upload one BF16 K block to the device scratch and project it into A_g
    /// at the logical `block_id` slot. No-op for non-BF16 K (FP8 = Step 4).
    fn project_bf16(&mut self, layer: usize, block_id: u32, k_host: &[u8], gpu: &dyn GpuBackend) {
        if k_host.len() != self.bf16_k_bytes {
            // FP8 / asymmetric dtype — dequant lands in Step 4. Skip so the
            // BF16-only Step 3 stays correct (A_g slot left uninitialised for
            // this block; its score will be ~0, never recalled).
            return;
        }
        // Sync H2D: blocks until the device write is complete, so the
        // projection kernel (on `self.stream`) is guaranteed to read the
        // uploaded bytes regardless of cross-stream ordering. Sync (not async)
        // also removes any host-source lifetime hazard — `k_host` may be a
        // short-lived per-iteration read in the pager's refresh loop.
        if let Err(e) = gpu.copy_h2d(k_host, DevicePtr(self.k_scratch.ptr)) {
            tracing::warn!("kvflash PredictorScorer: K upload failed (l={layer}): {e}");
            return;
        }
        if let Err(e) = self.predictor.project_kv_block_on_stream(
            self.stream,
            layer,
            block_id as usize,
            self.k_scratch.ptr,
        ) {
            tracing::warn!(
                "kvflash PredictorScorer: project failed (l={layer}, b={block_id}): {e}"
            );
        }
    }
}

impl KvFlashScorer for PredictorScorer {
    fn name(&self) -> &'static str {
        "predictor"
    }

    fn capture_q(
        &mut self,
        q: DevicePtr,
        _num_q_heads: u32,
        _head_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) {
        // Rebind to the model's decode stream so projection + scoring stay
        // ordered with the Q write (same-stream → no extra sync needed).
        self.stream = stream;
        let bytes = self.dims.num_q_heads * self.dims.head_dim * 2;
        if let Err(e) = gpu.copy_d2d_async(q, DevicePtr(self.q_capture.ptr), bytes, stream) {
            tracing::warn!("kvflash PredictorScorer: Q capture failed: {e}");
        }
    }

    fn project_evicted_block(
        &mut self,
        layer: usize,
        block_id: u32,
        k_host: &[u8],
        gpu: &dyn GpuBackend,
    ) {
        self.project_bf16(layer, block_id, k_host, gpu);
    }

    fn is_projected(&self, block_id: u32) -> bool {
        self.projected
            .get(block_id as usize)
            .copied()
            .unwrap_or(false)
    }

    fn mark_projected(&mut self, block_id: u32) {
        let idx = block_id as usize;
        if idx >= self.projected.len() {
            self.projected.resize(idx + 1, false);
        }
        self.projected[idx] = true;
    }

    fn score_chunks(&mut self, num_chunks: usize) -> Vec<f32> {
        let stream = self.stream;
        let n = num_chunks.min(self.dims.max_blocks);
        // 1. Project the captured decode Q into low-rank space.
        if let Err(e) =
            self.predictor
                .project_q_on_stream(stream, self.q_capture.ptr, self.q_proj.ptr)
        {
            tracing::warn!("kvflash PredictorScorer: project_q failed: {e}");
            return vec![0.0; num_chunks];
        }
        // 2. Score every block per layer; accumulate a per-layer mean across
        //    attention layers. `score_blocks` reads the layer's contiguous A_g
        //    region (per-token, max-reduced per block by the kernel).
        let mut acc = vec![0.0f32; n];
        let layer_stride_bytes = self.dims.max_blocks * self.dims.per_layer_block_floats() * 2;
        let base = self.predictor.a_g_dev_ptr();
        let inv = 1.0 / self.dims.num_layers as f32;
        for layer in 0..self.dims.num_layers {
            let layer_a_g = base + (layer * layer_stride_bytes) as u64;
            if let Err(e) = self.predictor.score_blocks_on_stream(
                stream,
                self.q_proj.ptr,
                layer_a_g,
                self.block_scores.ptr,
                self.dims.max_blocks,
            ) {
                tracing::warn!("kvflash PredictorScorer: score (l={layer}) failed: {e}");
                continue;
            }
            if let Err(e) = copy_d_to_h_async(
                self.score_host.as_mut_ptr() as *mut c_void,
                self.block_scores.ptr,
                self.dims.max_blocks * 4,
                stream,
            ) {
                tracing::warn!("kvflash PredictorScorer: score readback failed: {e}");
                continue;
            }
            let _ = stream_sync(stream);
            for (acc_i, s) in acc.iter_mut().zip(self.score_host.iter()) {
                *acc_i += s * inv;
            }
        }
        // Blocks beyond max_blocks are unscoreable (A_g is bounded) → score 0.
        acc.resize(num_chunks, 0.0);
        acc
    }
}
