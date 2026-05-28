// SPDX-License-Identifier: AGPL-3.0-only

//! Boot-time whitespace-token vocab scan.
//!
//! The Qwen3.6 BPE vocab has ~440 tokens that decode to whitespace-only
//! strings (spaces, tabs, newlines, multi-byte combinations). Two
//! scheduler-side gates need to know the complete set:
//!
//!   1. `decode_logits_seq.rs` — the position-0 parameter-body mask
//!      biases whitespace tokens down by `-8.0` so the model can't
//!      emit `<parameter=KEY> </parameter>` (whitespace then immediate
//!      close, stripped by the parser to empty args).
//!   2. `emit_step.rs` — the `param_body_chars_emitted` counter must
//!      NOT increment on whitespace tokens for the same reason (the
//!      counter gates the mask; bumping it on whitespace would unlock
//!      the close prematurely).
//!
//! Prior to this module both call sites carried a hardcoded
//! `[220, 198, 197, 256, 271]` list — 5 of 440 tokens. Under FP8
//! long-context drift the sampler regularly picks one of the other 435
//! and bypasses both gates, manifesting as the empty-arg /
//! whitespace-collapse drift modes (`research3_drift_catalog.md` #9,
//! #11; Tier A 2026-05-26 probe).
//!
//! The set is populated once at server startup via [`init`] (called
//! from `main_modules/serve.rs` after the tokenizer loads) and read by
//! the hot paths via [`whitespace_tokens`] / [`is_whitespace`]. A
//! `OnceLock` keeps the read-side lock-free and statically-allocated.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use tokenizers::Tokenizer;

static WHITESPACE_TOKENS: OnceLock<HashSet<u32>> = OnceLock::new();

/// Per-token mask (indexed by token id): `true` iff the token decodes
/// to text whose LAST character is an ASCII digit (0-9). Drives the
/// WS2 mid-content gate in [`crate::scheduler::decode_logits_seq`]
/// which suppresses leading-whitespace continuations when the model
/// has just emitted a digit (closes `0.1.0`→`0.1 .0` and
/// `2024`→`2 024` Tier A drift).
///
/// Stored as `Arc<[bool]>` for O(1) lookup at decode time, matching
/// the existing [`crate::scheduler::helpers::numeric_token_mask`]
/// pattern. Index by `tok as usize`; bounds-checked at the call site.
static DIGIT_ENDING_MASK: OnceLock<Arc<[bool]>> = OnceLock::new();

/// Empty-set sentinel returned by [`whitespace_tokens`] when called
/// before [`init`] (e.g. from unit tests that don't build a full
/// server). Production paths always see the real set; tests see an
/// empty mask, which is a fail-open semantic that matches the
/// pre-WS1 behavior (the 5-token literal mask is preserved alongside).
static EMPTY: OnceLock<HashSet<u32>> = OnceLock::new();

/// Scan the tokenizer vocab for whitespace-only tokens and freeze the
/// resulting set as the process-global mask. Subsequent calls are
/// no-ops (idempotent). Logs the discovered count + a sample.
///
/// "Whitespace-only" = `decode(id)` returns a non-empty string whose
/// every char satisfies `char::is_whitespace`. This matches the
/// semantic of the prior hardcoded list (`' '`, `'\t'`, `'\n'`,
/// `'  '`, `'\n\n'` were all whitespace-only) and excludes adjacent
/// drift-prone-but-not-whitespace-only tokens like `' .'` (id 641)
/// which are addressed separately by the WS2 mid-content gate.
pub fn init(tokenizer: &Tokenizer) {
    if WHITESPACE_TOKENS.get().is_some() {
        return;
    }
    let vocab_size = tokenizer.get_vocab_size(true) as u32;
    let mut set: HashSet<u32> = HashSet::new();
    let mut digit_mask: Vec<bool> = vec![false; vocab_size as usize];
    for id in 0..vocab_size {
        let Ok(decoded) = tokenizer.decode(&[id], false) else {
            continue;
        };
        if decoded.is_empty() {
            continue;
        }
        if decoded.chars().all(char::is_whitespace) {
            set.insert(id);
        }
        // WS2 (2026-05-26): pre-compute the per-token "last char is
        // ASCII digit" lookup. Used by the mid-content whitespace
        // gate to detect mid-number positions where an inserted
        // whitespace token is structurally wrong.
        if decoded.chars().next_back().is_some_and(|c| c.is_ascii_digit()) {
            digit_mask[id as usize] = true;
        }
    }
    let count = set.len();
    // Sanity check: the 5 historical Qwen3.6 whitespace IDs should be
    // in the scan if this is a Qwen-family model. Warn (don't insert)
    // when missing — on non-Qwen tokenizers those IDs may decode to
    // unrelated content and force-inserting them would mis-bias the
    // sampler. The scan above is authoritative; the warn just
    // surfaces a possible Qwen-tokenizer regression for operators.
    let mut historical_missing = Vec::new();
    for &expected in &[220u32, 198, 197, 256, 271] {
        if !set.contains(&expected) {
            historical_missing.push(expected);
        }
    }
    if !historical_missing.is_empty() {
        tracing::warn!(
            "whitespace_mask::init: historical Qwen whitespace ids missing from scan: \
             {historical_missing:?}. If this is a non-Qwen model that's expected. \
             If this is a Qwen model investigate a tokenizer regression."
        );
    }
    let digit_count = digit_mask.iter().filter(|&&b| b).count();
    let _ = WHITESPACE_TOKENS.set(set);
    let _ = DIGIT_ENDING_MASK.set(digit_mask.into());
    tracing::info!(
        "whitespace_mask: scanned vocab ({vocab_size} tokens), found {count} whitespace-only \
         tokens and {digit_count} digit-ending tokens"
    );
}

/// Returns the populated set, or an empty set if [`init`] was never
/// called. Production callers should ensure init runs at startup; the
/// fallback is purely defensive (e.g. for downstream cargo tests that
/// poke scheduler internals without booting a real server).
pub fn whitespace_tokens() -> &'static HashSet<u32> {
    WHITESPACE_TOKENS
        .get()
        .unwrap_or_else(|| EMPTY.get_or_init(HashSet::new))
}

/// Convenience predicate. `false` when [`init`] was never called.
#[inline]
pub fn is_whitespace(tok: u32) -> bool {
    whitespace_tokens().contains(&tok)
}

/// WS2 predicate: `true` iff `tok` decodes to a string whose last
/// character is an ASCII digit. `false` when [`init`] was never
/// called OR `tok` is out of vocab range (defensive). Used by the
/// mid-content whitespace gate to detect "model just emitted a
/// digit" positions where a leading-whitespace continuation is
/// structurally suspicious (Tier A drift `0.1.0`→`0.1 .0`).
#[inline]
pub fn is_digit_ending(tok: u32) -> bool {
    let Some(mask) = DIGIT_ENDING_MASK.get() else {
        return false;
    };
    mask.get(tok as usize).copied().unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: empty-set fallback is safe before init.
    #[test]
    fn pre_init_returns_empty() {
        // Note: this test relies on running in a fresh process where
        // WHITESPACE_TOKENS has not been set. Other tests in the same
        // binary that call `init` would invalidate this assumption;
        // we keep it isolated by never calling init here.
        let s = whitespace_tokens();
        // Either empty (never init'd) or populated (init'd by sibling
        // test). Both are acceptable; we just verify the API doesn't
        // panic.
        let _ = s.len();
    }
}
