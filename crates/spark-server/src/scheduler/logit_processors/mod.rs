// SPDX-License-Identifier: AGPL-3.0-only

//! Composable pre-sample logits pipeline.
//!
//! Pre-sample logit transformations used to live as a ~200-line inline
//! block inside `decode_logits_seq::process_seq_logits`. The blob
//! hard-coded ordering, made per-stage opt-out impossible, and
//! prevented moving individual stages to GPU. This module decomposes
//! it into eight per-stage [`LogitsProcessor`] impls plus a thin
//! pipeline driver. Stage order matches the pre-refactor monolith;
//! semantics are byte-identical (verified via the integration tests
//! in [`pipeline_tests`]).
//!
//! ## Stage order
//!
//! 1. [`f2_confidence::F2ConfidenceEarlyStop`] — arms `force_end_thinking`
//!    when top-1 probability stays ≥ 0.95 for the configured run.
//! 2. [`mid_word::MidWordThinkEndMask`] — suppresses `</think>` when
//!    the previous token decoded to mid-word text.
//! 3. [`post_close::PostCloseThinkMask`] — after `</think>` fires,
//!    masks `</think>` + `<think>` so the model can't re-enter.
//! 4. [`tool_during_think::ToolCallDuringThinkingMask`] — masks
//!    `<tool_call>` during thinking; biases it down on tool-loop.
//! 5. [`forced_think_end::ForcedThinkEndInjector`] — when budget
//!    + sentence-boundary policy says inject, blanket-mask to `</think>`.
//! 6. [`pin_tool_call::PinToToolCallStart`] — one-shot pin to
//!    `<tool_call>` immediately after `</think>` when require_tool_call.
//! 7. [`forced_token::ForcedTokenFastPath`] — when grammar admits
//!    exactly one next token, short-circuit pipeline + sampling.
//! 8. [`grammar_bitmask::GrammarBitmaskApply`] — apply grammar's
//!    next-token bitmask.
//!
//! ## Out of scope
//!
//! Adaptive-sampling entropy observation runs after this pipeline; it
//! decides sampling policy (greedy gate, effective temperature), not
//! logit transforms. The final `sample_with_params_history` call is
//! also downstream.

use crate::scheduler::ActiveSeq;

pub mod adadec_diag;
pub mod f2_confidence;
pub mod forced_think_end;
pub mod forced_token;
pub mod grammar_bitmask;
pub mod mid_word;
pub mod pin_tool_call;
pub mod post_close;
pub mod tool_during_think;

#[cfg(test)]
mod pipeline_tests;

/// Per-step environment passed to every processor. Holds tokenizer-
/// special tokens the pipeline cares about. `Copy` so it threads
/// through cheaply.
#[derive(Debug, Clone, Copy)]
pub struct LogitsContext {
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_end_token: Option<u32>,
}

/// Outcome of one [`LogitsProcessor::apply`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorOutcome {
    /// Logits modified in place (or no change). Continue pipeline.
    Continue,
    /// Pipeline short-circuit: emit this token directly with no
    /// further masking, grammar advance, or sampling.
    EmitToken(u32),
}

/// One stage of the pre-sample pipeline. Implementations are pure-CPU
/// today; a future GPU-resident bitmask kernel can implement this
/// trait too without changing the driver.
pub trait LogitsProcessor: Send + Sync {
    /// Apply this stage's transform to `logits`. May read+mutate
    /// `seq` state (e.g. F2 sets `seq.force_end_thinking`; grammar
    /// stages mutate `seq.grammar_state`).
    fn apply(
        &self,
        logits: &mut [f32],
        seq: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome;

    /// Stable identifier for tracing + future per-request enable/
    /// disable. Static, no allocation.
    fn name(&self) -> &'static str;

    /// vLLM convention: `true` if this stage never alters which token
    /// wins under argmax (e.g. additive bias preserving ordering).
    /// Currently advisory — reserved for future GPU-batched skip paths.
    fn is_argmax_invariant(&self) -> bool {
        false
    }
}

/// Run the canonical pipeline. Returns `Some(token)` when any stage
/// short-circuited via [`ProcessorOutcome::EmitToken`]; `None`
/// otherwise (caller proceeds to sampling).
pub fn run_pipeline(
    logits: &mut [f32],
    seq: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Option<u32> {
    let stages: [&dyn LogitsProcessor; 9] = [
        &f2_confidence::F2ConfidenceEarlyStop,
        &mid_word::MidWordThinkEndMask,
        &post_close::PostCloseThinkMask,
        &tool_during_think::ToolCallDuringThinkingMask,
        &forced_think_end::ForcedThinkEndInjector,
        &pin_tool_call::PinToToolCallStart,
        &forced_token::ForcedTokenFastPath,
        &grammar_bitmask::GrammarBitmaskApply,
        // AdaDec Phase 1 diagnostic — observes the post-grammar-bitmask
        // distribution, never mutates. No-op when ATLAS_ADADEC_DIAGNOSTIC
        // env var is unset.
        &adadec_diag::AdaDecDiagnostic,
    ];
    for stage in stages.iter() {
        match stage.apply(logits, seq, ctx) {
            ProcessorOutcome::Continue => {}
            ProcessorOutcome::EmitToken(tok) => return Some(tok),
        }
    }
    None
}
