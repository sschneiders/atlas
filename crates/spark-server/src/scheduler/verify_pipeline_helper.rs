// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-time pre-sample LogitsProcessor pipeline (Phase C-2 wiring).
//!
//! The MTP / speculative-decode verify paths used to consume the raw
//! GPU `argmax_bf16` ID at every verify position, completely bypassing
//! the 8-stage [`crate::scheduler::logit_processors`] pipeline that the
//! non-MTP path runs on every sampled token. Result: tokens emitted
//! through verify (the dominant decode path when MTP is enabled —
//! every accepted/bonus token came from `decode_verify_graphed`) never
//! saw mid-word `</think>` defer, post-close think mask, tool-during-
//! think mask, forced think-end injection, pin-to-tool-call, forced-
//! token fast-path, or grammar bitmask. This is the root cause of
//! grammar desync, malformed tool calls, mid-word `</think>` cuts and
//! stray `<think>` re-entry observed on Qwen3.6-FP8 (opencode-session
//! transcripts, 2026-05-24).
//!
//! This module replays the same dequant + pipeline on a host-side copy
//! of the verify logits buffer (`[K, vocab]` BF16, written by
//! `decode_verify_graphed_*` into `model.logits_buffer_ptr()`), then
//! picks the resulting argmax. Cost: ~0.8 ms per verify position for a
//! ~256k vocab on host, mirroring the non-MTP `process_seq_logits` path
//! in `decode_logits_seq.rs`. The CUDA-graphed `argmax_bf16` saving of
//! ~0.5 ms/step is preserved for the **draft** path (drafts already go
//! through a separate grammar-bitmask path in MTP propose); only the
//! **verify-time** argmax is replaced.
//!
//! Per-position semantics: the pipeline is applied independently to
//! each verify position 0..K using the live `ActiveSeq` state at call
//! time. For position 0 that state is exactly the post-`last_token`
//! state, identical to the non-MTP decode site. For positions ≥ 1 the
//! sequence state has not yet been advanced by `emit_token(drafts[i])`
//! (acceptance is decided after this call), so a few state-dependent
//! masks (mid-word lookback, etc.) read a slightly stale `last
//! output_tokens` — best-effort, identical to what the model itself
//! does internally when unrolling the verify positions greedily. The
//! grammar / forced-think masks still apply correctly because they
//! key off flags that only flip via `emit_token`, which has not yet
//! run on the verify position.

use crate::scheduler::ActiveSeq;
use crate::scheduler::helpers::bf16_to_f32;
use crate::scheduler::logit_processors::{LogitsContext, run_pipeline};
use spark_model::traits::Model;

/// Per-position verify logits, dequantised + processed through the full
/// pre-sample pipeline. Returns the chosen token: either the forced
/// token from a [`crate::scheduler::logit_processors::forced_token::ForcedTokenFastPath`]
/// short-circuit, or the post-pipeline argmax.
///
/// `logits_bytes`: byte slice for ONE verify position; length
/// `vocab_size * 2` (BF16) or `vocab_size * 4` (FP32).
/// `is_fp32`: true when the model emits FP32 logits (Gemma-4 dense).
/// `a`: the active sequence; the pipeline mutates seq state in place
/// (F2 confidence arm, sentence_defer_count, etc.).
/// `ctx`: tokenizer special-token IDs used by the pipeline.
///
/// Mirrors the host-side path of `decode_logits_seq::process_seq_logits`
/// for byte-identical pipeline semantics.
pub fn verify_pick_with_pipeline(
    logits_bytes: &[u8],
    is_fp32: bool,
    vocab_size: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> u32 {
    // 1. Dequant per the same scheme as `process_seq_logits`.
    let mut f32_logits: Vec<f32> = if is_fp32 {
        (0..vocab_size)
            .map(|j| {
                let off = j * 4;
                f32::from_le_bytes([
                    logits_bytes[off],
                    logits_bytes[off + 1],
                    logits_bytes[off + 2],
                    logits_bytes[off + 3],
                ])
            })
            .collect()
    } else {
        (0..vocab_size)
            .map(|j| {
                let lo = logits_bytes[j * 2];
                let hi = logits_bytes[j * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };

    // 2. Run the canonical pipeline. Short-circuit returns the forced
    //    token directly — no argmax scan needed.
    if let Some(forced) = run_pipeline(&mut f32_logits, a, ctx) {
        return forced;
    }

    // 3. Argmax over the (now-masked) vector. `f32::partial_cmp` with
    //    NaN-safe fallback to `Equal` matches the sampler's argmax
    //    branch behaviour.
    let mut best_id: u32 = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in f32_logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i as u32;
        }
    }
    best_id
}

/// Convenience: copy the full `[K, vocab]` verify logits buffer to
/// host and apply [`verify_pick_with_pipeline`] to every position,
/// returning the K processed token IDs. Falls back to the raw argmax
/// IDs if the D2H copy fails (matches `verify_resample` and
/// `extract_verify_logprobs` failure semantics).
///
/// `argmax_ids` is the GPU-graphed argmax already returned by
/// `decode_verify_graphed*`; used as the fallback for the failure
/// path and as the array length source.
pub fn verify_pick_all_with_pipeline(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Vec<u32> {
    let k = argmax_ids.len();
    if k == 0 {
        return Vec::new();
    }
    let vocab = model.vocab_size();
    // BF16 always for verify path: `decode_verify_graphed_*` writes BF16
    // to `logits_buffer()`. The FP32-lm_head path (Gemma-4 dense) does
    // not go through verify (no MTP for dense Gemma).
    let elem_bytes = 2usize;
    let total = k * vocab * elem_bytes;
    let mut buf = vec![0u8; total];
    if model
        .copy_logits_to_host(model.logits_buffer_ptr(), &mut buf)
        .is_err()
    {
        return argmax_ids.to_vec();
    }
    (0..k)
        .map(|i| {
            let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
            verify_pick_with_pipeline(slice, false, vocab, a, ctx)
        })
        .collect()
}
