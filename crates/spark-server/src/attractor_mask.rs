// SPDX-License-Identifier: AGPL-3.0-only

//! Boot-time scan of known FP8 drift "attractor" tokens.
//!
//! Drift catalog `research3_drift_catalog.md` documents 15 patterns;
//! several (notably #7 "`lean://` prefix attractor") are caused by a
//! specific BPE token winning at the FIRST character of a tool
//! parameter body under FP8 long-context drift. The model fixates on
//! that token even when the user never asked for anything Lean-related,
//! then continues the parameter content with the wrong concept rooted.
//!
//! Mechanism: at param-body position 0 inside a tool argument, suppress
//! these known-attractor token IDs by `-8.0` (same magnitude as the
//! whitespace position-0 mask). Mid-content occurrences are NOT
//! suppressed (Lean is a real language; legitimate uses must survive).
//!
//! Design: model-agnostic via boot-time tokenizer encode. The list of
//! attractor STRINGS is currently hardcoded here as a starting set;
//! a future refinement moves it to `MODEL.toml [behavior].attractors`
//! so per-model overrides are possible without rebuilding.

use std::collections::HashSet;
use std::sync::OnceLock;

use tokenizers::Tokenizer;

static ATTRACTOR_TOKENS: OnceLock<HashSet<u32>> = OnceLock::new();
static EMPTY: OnceLock<HashSet<u32>> = OnceLock::new();

/// Known FP8 drift attractor STRINGS. Each is tokenized at boot using
/// the live tokenizer; the resulting token ids land in
/// [`attractor_tokens`].
///
/// - `lean`, ` lean` — drift #7 `lean://` prefix attractor (catalog).
///   Lowercase only; `Lean` (capital) is a real language we don't
///   want to suppress.
const ATTRACTOR_STRINGS: &[&str] = &[
    "lean",
    " lean",
];

/// Scan the tokenizer for the [`ATTRACTOR_STRINGS`] and freeze the
/// resulting token id set. Idempotent. Logs the discovered ids.
///
/// For each string we take the FIRST token id of its tokenization —
/// the attractor fires on the first token, after which a different
/// gate (rep_penalty, B1 margin, etc.) catches the continuation.
pub fn init(tokenizer: &Tokenizer) {
    if ATTRACTOR_TOKENS.get().is_some() {
        return;
    }
    let mut set: HashSet<u32> = HashSet::new();
    for &s in ATTRACTOR_STRINGS {
        let Ok(enc) = tokenizer.encode(s, false) else {
            tracing::warn!("attractor_mask: failed to encode {s:?}");
            continue;
        };
        let ids = enc.get_ids();
        if let Some(&id) = ids.first() {
            set.insert(id);
            tracing::info!(
                "attractor_mask: registered {s:?} -> id {id} (full tokenization: {:?})",
                ids
            );
        }
    }
    let _ = ATTRACTOR_TOKENS.set(set);
}

/// Returns the populated set, or an empty set if [`init`] was never
/// called. Production callers should ensure init runs at startup.
pub fn attractor_tokens() -> &'static HashSet<u32> {
    ATTRACTOR_TOKENS
        .get()
        .unwrap_or_else(|| EMPTY.get_or_init(HashSet::new))
}

/// Convenience predicate; `false` before init.
#[inline]
pub fn is_attractor(tok: u32) -> bool {
    attractor_tokens().contains(&tok)
}
