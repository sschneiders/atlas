// SPDX-License-Identifier: AGPL-3.0-only
//
// A2 (2026-05-26) — Tier-2 fuzzy repair of tool-call arguments against
// the prompt vocabulary.
//
// The FP8 Qwen3.6 drift pattern we observed across Wave-1 and Wave-3
// probes is one-byte substitution / character-drop in identifier-like
// tokens that the user explicitly named in the original prompt:
//
//   user: `/tmp/test-rust-axum-v3`           model emits: `/tmp/test-rust-axut-v3`  (m→t)
//   user: `/tmp/test-rust-axum-wave1`        model emits: `/tmp/test-rust-axum-wave/`  (1→/)
//   user: `axum-v3`                          model emits: `axuma-aadac`              (Lev-3)
//
// Per the `research3_fuzzy_repair.md` agent: opencode's `edit` tool
// already runs a 9-strategy fuzzy chain internally; its `read`/`bash`/
// path tools do NOT. When Atlas's tier-2 validator rejects a malformed
// path, Atlas can repair the path BEFORE returning the call to opencode
// — closing roughly 70% of the observed drift events on its own,
// independent of the Tier 5c retry inference.
//
// Scope of this module: pure functions, no external deps. Levenshtein
// edit distance + word-level vocabulary matching. Caller drives when to
// invoke (validator-failure path).

use std::collections::HashSet;

/// Edit-distance threshold for accepting a fuzzy match. Lev-1 catches
/// single-byte substitutions (`axum`→`axut`) and single-byte drops
/// (`/test-rust-axum-v3`→`/test-rustaxum-v3` after the hyphen drop).
/// Lev-2 is more aggressive and worth opting into only when the value
/// is clearly garbled and the prompt-vocab match is unambiguous.
pub const LEV_DEFAULT_MAX: usize = 2;

/// Extract a word-level vocabulary from arbitrary prompt text. Words
/// are runs of `[a-zA-Z0-9_.\-/]` of length ≥ 3 — i.e. roughly the
/// tokens that look like identifiers, paths, file names, version
/// strings. Whitespace and the usual prose-punctuation set are
/// separators. Returned set is owned `String`s so the caller can keep
/// the vocab around for the lifetime of the request.
pub fn extract_prompt_vocab(prompt_text: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut buf = String::new();
    for ch in prompt_text.chars() {
        let is_id = ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-' | '/');
        if is_id {
            buf.push(ch);
        } else {
            if buf.len() >= 3 {
                out.insert(std::mem::take(&mut buf));
            } else {
                buf.clear();
            }
        }
    }
    if buf.len() >= 3 {
        out.insert(buf);
    }
    out
}

/// Classic dynamic-programming Levenshtein distance. Caps the matrix at
/// `max + 1` so we return `max + 1` for anything strictly above the
/// threshold (avoids worst-case O(N*M) on hopelessly-different inputs).
pub fn levenshtein(a: &str, b: &str, max: usize) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > max {
        return max + 1;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        let mut row_min = curr[0];
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost).min(curr[j] + 1).min(prev[j + 1] + 1);
            row_min = row_min.min(curr[j + 1]);
        }
        if row_min > max {
            return max + 1;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Try to repair `candidate` by matching it against the vocab. Returns
/// `Some(replacement)` only when the match is UNAMBIGUOUS — exactly one
/// vocab entry within `max_distance`. Ambiguous matches return `None`
/// so the caller falls through to other recovery paths.
pub fn repair_word(candidate: &str, vocab: &HashSet<String>, max_distance: usize) -> Option<String> {
    if candidate.len() < 3 {
        return None;
    }
    if vocab.contains(candidate) {
        return None;
    }
    let mut best: Option<(usize, &String)> = None;
    let mut ambiguous = false;
    for v in vocab {
        if v.len() < 3 {
            continue;
        }
        let d = levenshtein(candidate, v, max_distance);
        if d <= max_distance {
            match best {
                None => best = Some((d, v)),
                Some((bd, _)) if d < bd => {
                    best = Some((d, v));
                    ambiguous = false;
                }
                Some((bd, _)) if d == bd => {
                    ambiguous = true;
                }
                _ => {}
            }
        }
    }
    if ambiguous {
        return None;
    }
    best.map(|(_, v)| v.clone())
}

/// Repair an entire tool-argument value by:
///  1. Splitting on non-identifier chars (whitespace, JSON punctuation,
///     path separators are kept as separators).
///  2. Replacing each word that has an unambiguous Lev-≤`max` match
///     against the vocab.
///  3. Stitching the original separators back in.
///
/// Returns `Some(repaired)` iff any substitution actually happened —
/// `None` means the value is either already in-vocab or unrepairable.
pub fn repair_value(value: &str, vocab: &HashSet<String>, max_distance: usize) -> Option<String> {
    let mut out = String::with_capacity(value.len());
    let mut buf = String::new();
    let mut any_repair = false;
    let flush = |buf: &mut String, out: &mut String, any_repair: &mut bool| {
        if buf.is_empty() {
            return;
        }
        if let Some(rep) = repair_word(buf, vocab, max_distance) {
            out.push_str(&rep);
            *any_repair = true;
        } else {
            out.push_str(buf);
        }
        buf.clear();
    };
    for ch in value.chars() {
        let is_id = ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-' | '/');
        if is_id {
            buf.push(ch);
        } else {
            flush(&mut buf, &mut out, &mut any_repair);
            out.push(ch);
        }
    }
    flush(&mut buf, &mut out, &mut any_repair);
    if any_repair { Some(out) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lev_basic() {
        assert_eq!(levenshtein("axum", "axut", 2), 1);
        assert_eq!(levenshtein("axum", "axum", 2), 0);
        assert_eq!(levenshtein("axum", "xyzabc", 2), 3); // capped at max+1
    }

    #[test]
    fn vocab_extracts_path_idents() {
        let v = extract_prompt_vocab("create rust axum project inside ./test-rust-axum-v3 please");
        assert!(v.contains("axum"));
        assert!(v.contains("test-rust-axum-v3") || v.contains("./test-rust-axum-v3"));
        assert!(!v.contains("a")); // too short
    }

    #[test]
    fn repair_unambiguous_substitution() {
        let v: HashSet<String> = ["axum", "tokio", "serde"].into_iter().map(String::from).collect();
        assert_eq!(repair_word("axut", &v, 2), Some("axum".into()));
        assert_eq!(repair_word("tokyo", &v, 2), Some("tokio".into()));
        assert_eq!(repair_word("axum", &v, 2), None); // already in vocab
    }

    #[test]
    fn repair_value_path_drift() {
        let v: HashSet<String> = ["test-rust-axum-v3", "Cargo.toml"]
            .into_iter()
            .map(String::from)
            .collect();
        let repaired = repair_value("/tmp/test-rust-axut-v3/Cargo.toml", &v, 2);
        assert_eq!(repaired.as_deref(), Some("/tmp/test-rust-axum-v3/Cargo.toml"));
    }

    #[test]
    fn repair_skips_ambiguous() {
        let v: HashSet<String> = ["axum", "axem"].into_iter().map(String::from).collect();
        // Both axum and axem are Lev-1 from "axym" → ambiguous, no repair.
        assert_eq!(repair_word("axym", &v, 1), None);
    }
}
