// SPDX-License-Identifier: AGPL-3.0-only

//! #110: fully-batched mixed step (M decode + N prefill chunks) dispatch.

#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::traits::{MixedBatchResult, Model, PrefillSlice, SequenceState};

impl TransformerModel {
    /// #110: fully-batched mixed step (M decode + N prefill chunks).
    ///
    /// The previous trait DEFAULT (`Model::mixed_forward_batch`) ran the
    /// decode half via `decode_batch` (which hardcodes `default_stream` in
    /// `decode_batch_compute_main`) and the prefill half via
    /// `prefill_batch_chunk` on the caller's separate `prefill_stream`, with
    /// NO event ordering them. Both halves carve the SAME singleton
    /// `buffers.scratch()` (decode metadata at +32768) and `buffers.logits()`.
    /// At concurrency >= 4 the prefill half's chunk-0 `zero_all` + MoE top-k
    /// staging clobbered the decode half's metadata/logits mid-flight ->
    /// torn KV-slot read -> CUDA_ERROR_ILLEGAL_ACCESS (700), or silently
    /// corrupted decode logits. (Same class fixed for the single-prefill
    /// fused path on 2026-05-10; never propagated here.)
    ///
    /// Fix: force BOTH halves onto `default_stream` so they serialize
    /// (decode completes — including its scratch-metadata attention reads
    /// and LM head — before the prefill half reuses those buffers), and copy
    /// the decode logits into a private staging buffer before the prefill
    /// half overwrites `logits`. The two halves cannot overlap anyway (shared
    /// arena), so single-stream costs nothing vs the old serial default.
    pub(super) fn mixed_forward_batch_dispatch(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_streams: &mut [PrefillSlice<'_>],
        _stream: u64,
    ) -> Result<MixedBatchResult> {
        let stream = self.gpu.default_stream();
        let decode_logits = if !decode_tokens.is_empty() {
            let raw = self.decode_batch(decode_tokens, decode_seqs, stream)?;
            if raw != DevicePtr::NULL {
                // Decode logits are [n_decode, vocab] BF16, contiguous from
                // row 0. Stage them out of `buffers.logits()` (same stream,
                // ordered after the decode LM head) before the prefill half's
                // zero_all / finalize_last overwrite that buffer.
                let staging = self.buffers.decode_logits_staging();
                let bytes = decode_tokens.len() * self.config.vocab_size * 2;
                self.gpu.copy_d2d_async(raw, staging, bytes, stream)?;
                staging
            } else {
                raw
            }
        } else {
            DevicePtr::NULL
        };
        let prefill_logits = self.prefill_batch_chunk(prefill_streams, stream)?;
        Ok(MixedBatchResult {
            decode_logits,
            prefill_logits,
        })
    }
}
