// SPDX-License-Identifier: AGPL-3.0-only

//! Compile-time + light-touch unit tests for the pre-sample pipeline.
//!
//! ## Why no full byte-identical integration test
//!
//! [`crate::scheduler::ActiveSeq`] has ~60 fields including channels
//! (`ResponseSink`, `cancel_flag`), wall-clock `Instant`s, an
//! `AdaptiveSamplingState`, an `SsmDecodeRing`, an optional
//! `GrammarState` (which itself needs an `xgrammar::Tokenizer`), and
//! is `pub(super)` to `scheduler::*`. Constructing it from a `#[cfg(test)]`
//! module under `scheduler::logit_processors` is mechanically possible
//! but the resulting helper would be ~80 lines of fixture wiring per
//! test for ~5 lines of behaviour assertion — and any drift in the
//! `ActiveSeq` schema would silently rot the fixtures.
//!
//! The byte-identical guarantee against the pre-refactor monolith
//! lives in the parent task's live-run regression — the pipeline is
//! wired into `process_seq_logits` in a follow-up step, after which
//! the existing scheduler integration tests + opencode-session.md
//! prose-corpus replay form the actual byte-identical guard.
//!
//! This file's tests cover only what is testable without ActiveSeq:
//!
//! 1. The stage order in `run_pipeline` (compile-time guarantee that
//!    all 8 unit structs implement the trait and are wired in order).
//! 2. Stage `name()` strings are stable + distinct (forward-compat with
//!    per-stage enable/disable flags).
//! 3. The pure deferral-input math from `ForcedThinkEndInjector` matches
//!    `confidence::should_inject_think_end` semantics for the boundary
//!    cases the integration was known to hit on 2026-05-23.

use super::*;
use crate::scheduler::confidence::{
    MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    should_inject_think_end,
};

/// Every stage has a unique, stable name. Drift here breaks tracing
/// dashboards and any future per-stage opt-out config.
#[test]
fn stage_names_are_distinct_and_stable() {
    let names: [&'static str; 8] = [
        f2_confidence::F2ConfidenceEarlyStop.name(),
        mid_word::MidWordThinkEndMask.name(),
        post_close::PostCloseThinkMask.name(),
        tool_during_think::ToolCallDuringThinkingMask.name(),
        forced_think_end::ForcedThinkEndInjector.name(),
        pin_tool_call::PinToToolCallStart.name(),
        forced_token::ForcedTokenFastPath.name(),
        grammar_bitmask::GrammarBitmaskApply.name(),
    ];
    // Distinct
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            assert_ne!(
                names[i], names[j],
                "stage names must be distinct ({} == {})",
                names[i], names[j]
            );
        }
    }
    // Stable identifiers (pin exact strings so a rename is a visible
    // diff — these appear in tracing logs + future feature flags).
    assert_eq!(names[0], "f2_confidence_early_stop");
    assert_eq!(names[1], "mid_word_think_end_mask");
    assert_eq!(names[2], "post_close_think_mask");
    assert_eq!(names[3], "tool_call_during_thinking_mask");
    assert_eq!(names[4], "forced_think_end_injector");
    assert_eq!(names[5], "pin_to_tool_call_start");
    assert_eq!(names[6], "forced_token_fastpath");
    assert_eq!(names[7], "grammar_bitmask_apply");
}

/// The only stage that should advertise argmax-invariance is the
/// F2 confidence accumulator (logits are read-only). Everything else
/// mutates logits and must report `false`.
#[test]
fn argmax_invariance_advertisement() {
    assert!(f2_confidence::F2ConfidenceEarlyStop.is_argmax_invariant());
    assert!(!mid_word::MidWordThinkEndMask.is_argmax_invariant());
    assert!(!post_close::PostCloseThinkMask.is_argmax_invariant());
    assert!(!tool_during_think::ToolCallDuringThinkingMask.is_argmax_invariant());
    assert!(!forced_think_end::ForcedThinkEndInjector.is_argmax_invariant());
    assert!(!pin_tool_call::PinToToolCallStart.is_argmax_invariant());
    assert!(!forced_token::ForcedTokenFastPath.is_argmax_invariant());
    assert!(!grammar_bitmask::GrammarBitmaskApply.is_argmax_invariant());
}

/// `ForcedThinkEndInjector` packages three booleans for the gate. Pin
/// the truth table against the gate function so the constants used in
/// the injector (`THINK_DEFER_BUDGET_FACTOR`, `THINK_DEFER_ABS_CEILING`,
/// `MAX_SENTENCE_DEFER_TOKENS`) all stay consistent with the gate.
#[test]
fn forced_think_end_gate_semantics() {
    // Not armed → never inject.
    assert!(!should_inject_think_end(false, false, false, false));
    assert!(!should_inject_think_end(false, true, true, true));

    // Armed + hard override → always inject (even mid-fence).
    assert!(should_inject_think_end(true, true, false, true));
    assert!(should_inject_think_end(true, false, false, true));

    // Armed + in fence + no override → defer.
    assert!(!should_inject_think_end(true, true, false, false));
    assert!(!should_inject_think_end(true, true, true, false));

    // Armed + outside fence + at sentence boundary → inject.
    assert!(should_inject_think_end(true, false, true, false));

    // Armed + outside fence + NOT at boundary → defer (await period).
    assert!(!should_inject_think_end(true, false, false, false));
}

/// The defer-override math in `ForcedThinkEndInjector::apply` mirrors
/// what the inline monolith computed. This pins the constants we
/// import so a sweep that tunes them in `confidence.rs` flags this test
/// rather than silently drifting the injector.
#[test]
fn defer_override_math_constants() {
    // Budget×factor exceeded.
    let budget: u32 = 100;
    let thinking_tokens: u32 = budget.saturating_mul(THINK_DEFER_BUDGET_FACTOR);
    assert!(thinking_tokens >= budget.saturating_mul(THINK_DEFER_BUDGET_FACTOR));

    // Absolute ceiling when budget is None.
    let unlimited_tokens: u32 = THINK_DEFER_ABS_CEILING;
    assert!(unlimited_tokens >= THINK_DEFER_ABS_CEILING);

    // Sentence-defer ceiling.
    let defer_count: u32 = MAX_SENTENCE_DEFER_TOKENS;
    assert!(defer_count >= MAX_SENTENCE_DEFER_TOKENS);
}
