// SPDX-License-Identifier: AGPL-3.0-only

//! Content-phase token handling for `process_decode_logits` — the
//! non-thinking branch of the per-token decode loop. Extracted from
//! `decode_logits_step.rs` to keep that file ≤500 LoC.
//!
//! Runs once per sampled token while the sequence is *outside*
//! `<think>…</think>`. The content-loop / inter-tool-prose / post-think
//! content-cap watchdogs that used to live here were removed 2026-06-12
//! for vLLM parity: a turn ends only on EOS, client stop strings,
//! `max_tokens`, or tool-call end.

use super::*;

/// Handle one sampled token that lands in the content phase (model is
/// not inside `<think>`). Mutates `a` in place: decrements the
/// generation budget and advances content bookkeeping.
pub fn handle_content_token(a: &mut ActiveSeq, _model: &dyn Model) {
    a.consume_generation_budget();
    a.content_started = true;
    // think_just_ended is a one-shot: it was set when the prior
    // token was `</think>`; clear it now that we've emitted the
    // first content token (which Change 3b's mask pinned to
    // tool_call_start_token when require_tool_call was set).
    a.think_just_ended = false;
}
