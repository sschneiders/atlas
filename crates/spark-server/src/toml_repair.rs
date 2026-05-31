// SPDX-License-Identifier: AGPL-3.0-only

//! SC1 (2026-05-26, /loop iter 2) — server-side post-process repair of
//! Cargo.toml content emitted by the FP8 model in `write` tool calls.
//!
//! The C1 harness (N=10) shows ~50% of attempted Cargo.toml writes are
//! structurally broken at the TOML-parse level — drift modes catalogued
//! per inspection of runs/run_sm1_*.json:
//!
//! - **Newline collapse** (drift #11): `[package] name = "X" version =
//!   "Y"` on one line. Found in run_sm1_1, run_sm1_8.
//! - **Preamble hallucination** (drift #16, new): model prepends a fake
//!   compiler warning or error message before the actual TOML content.
//!   E.g. run_sm1_4: `cargo:warning: This is not valid TOML — it's a
//!   Cargo.toml file\n[package]\n…`. run_sm1_7: `hipify-error[hipify-
//!   perl]: …:1:0-23: hipErrorUnknown…`.
//! - **Total garbage**: starts with a stray byte (`d[package]…`) and
//!   degenerates into invalid syntax. Not repairable.
//!
//! Repair pipeline (each step is conservative — produce a candidate,
//! re-parse, accept only if it now parses):
//!   1. Strip preamble: find the first `[` AT THE START of a line and
//!      discard everything before it.
//!   2. Insert newlines between adjacent statements: regex inserts a
//!      `\n` between `]` and `\w`, between `"` and `\w+\s*=`, between
//!      `}` and `\w+\s*=`, between digit and `[`.
//!   3. Try the strip + the regex fix together.
//!
//! Conservative semantics: original content is returned unchanged if
//! it's already valid OR if no repair produces valid TOML.

// No regex dependency — tool_parser/parse_single_a.rs:40 documents the
// project preference for hand-rolled pattern matching. Repair is done
// via a single-pass char walker below.

/// Try to make `content` parse as TOML. Returns `Some(repaired)` iff
/// the original was invalid AND a repair produced valid TOML.
/// `None` means either (a) original was already valid, or (b) no
/// repair worked. Both leave the original content intact at the call
/// site.
pub fn try_repair_toml(content: &str) -> Option<String> {
    // Step 0: already valid? — nothing to do.
    if toml::from_str::<toml::Value>(content).is_ok() {
        return None;
    }

    let candidates = generate_repair_candidates(content);
    for cand in candidates {
        if cand == content {
            continue;
        }
        if toml::from_str::<toml::Value>(&cand).is_ok() {
            tracing::info!(
                "SC1 toml_repair: repaired Cargo.toml from {} chars to {} chars (orig was unparseable)",
                content.len(),
                cand.len()
            );
            return Some(cand);
        }
    }
    None
}

/// Build a list of repair candidates from most-conservative to most-
/// aggressive. Each is independently re-validated by the caller.
fn generate_repair_candidates(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    // (a) Preamble strip
    if let Some(stripped) = strip_preamble(content) {
        out.push(stripped.clone());
        // (a)+(b) preamble strip THEN newline insertion
        out.push(insert_newlines(&stripped));
    }
    // (b) Newline insertion only
    out.push(insert_newlines(content));
    out
}

/// Find the first `[` at the start of a line (`\n[` or beginning of
/// file) and discard everything before. Returns None if no such
/// position exists. Defensive: don't strip more than a few hundred
/// bytes of preamble — if the model wrote pages of garbage, the
/// rest is probably also garbage.
fn strip_preamble(content: &str) -> Option<String> {
    if content.starts_with('[') {
        return None;
    }
    let bytes = content.as_bytes();
    let mut pos = None;
    let limit = std::cmp::min(bytes.len(), 1024);
    for i in 0..limit {
        if bytes[i] == b'[' && (i == 0 || bytes[i - 1] == b'\n') {
            pos = Some(i);
            break;
        }
    }
    pos.map(|p| content[p..].to_string())
}

/// Heuristic newline insertion at common TOML drift boundaries.
/// Single-pass char walker — no regex. Conservative: only inserts
/// `\n` when the previous char is a known TOML terminator AND the
/// upcoming run looks like an identifier followed by `=` (a new
/// key/value statement) or a `[…]` section header.
///
/// The terminators we recognize and the transition that triggers
/// insertion:
///   `]` + ident= → `]\nident=`        (section header → first key)
///   `"` + ident= → `"\nident=`        (quoted value → next key, only when separated by whitespace)
///   `}` + ident= → `}\nident=`        (inline-table close → next key)
///   `]` + `[`    → `]\n\n[`           (section header → next section)
///   `"` + `[`    → `"\n\n[`           (quoted value → next section)
///   digit + `[`  → `digit\n\n[`       (unquoted number → next section)
///
/// We track string state so we don't insert newlines INSIDE TOML
/// strings (where `]` and `"` can appear as content).
fn insert_newlines(content: &str) -> String {
    let bytes = content.as_bytes();
    let mut out = String::with_capacity(bytes.len() + 32);
    let mut in_string = false;
    let mut prev_significant: Option<u8> = None;
    // Tracks whether we've seen at least one whitespace byte since
    // the last non-whitespace byte. Used to distinguish
    // `truetokio` (no break — DON'T insert newline) from
    // `true tokio` (break — insert newline before "tokio" if it
    // starts a key=value statement).
    let mut ws_since_prev_sig = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];

        // Track in-string. Only `"` toggles (single-quote strings
        // are rare in TOML and we don't handle multi-line `"""`).
        if c == b'"' && !is_escaped(bytes, i) {
            in_string = !in_string;
        }

        // Outside strings, check if THIS position is a transition
        // that needs a newline inserted BEFORE it.
        if !in_string {
            let needs_nl = match (prev_significant, c) {
                // Section open after section close, quoted value,
                // or digit — emit blank-line separator.
                (Some(b']'), b'[') | (Some(b'"'), b'[') | (Some(b'0'..=b'9'), b'[') => Some(2),
                // Identifier-key start after a structural close
                // (`]`/`"`/`}`) — emit single newline.
                (Some(b']' | b'"' | b'}'), x) if is_ident_start(x) => {
                    peek_key(bytes, i).then_some(1)
                }
                // Identifier-key start after an alphanumeric prev
                // AND a whitespace gap — covers `true tokio=…`,
                // `42 axum=…`, where the prev value was a bare
                // bool / number / unquoted identifier.
                (Some(p), x)
                    if (p.is_ascii_alphanumeric())
                        && is_ident_start(x)
                        && ws_since_prev_sig =>
                {
                    peek_key(bytes, i).then_some(1)
                }
                _ => None,
            };
            if let Some(n) = needs_nl {
                if !out.ends_with('\n') {
                    for _ in 0..n {
                        out.push('\n');
                    }
                } else if n == 2 && !out.ends_with("\n\n") {
                    out.push('\n');
                }
            }
        }

        out.push(c as char);

        // Update prev_significant — skip whitespace so transitions
        // like `]   name=` still trigger.
        if c.is_ascii_whitespace() {
            ws_since_prev_sig = true;
        } else {
            prev_significant = Some(c);
            ws_since_prev_sig = false;
        }
        i += 1;
    }
    out
}

fn is_escaped(bytes: &[u8], i: usize) -> bool {
    // Count backslashes preceding bytes[i]; odd count = escaped.
    let mut count = 0usize;
    let mut j = i;
    while j > 0 && bytes[j - 1] == b'\\' {
        count += 1;
        j -= 1;
    }
    count % 2 == 1
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

/// Peek forward from `bytes[i]` to see whether the upcoming run
/// looks like `ident\s*=` (a TOML key/value statement start).
/// Allows `.` and `-` in the identifier (e.g. `tokio.workspace`,
/// `serde-json`). Cap the scan at 64 bytes — well past any
/// realistic key length.
fn peek_key(bytes: &[u8], i: usize) -> bool {
    let mut j = i;
    let end = std::cmp::min(bytes.len(), i + 64);
    let mut saw_ident = false;
    while j < end {
        let c = bytes[j];
        if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.' {
            saw_ident = true;
            j += 1;
        } else if c == b' ' || c == b'\t' {
            j += 1;
        } else if c == b'=' {
            return saw_ident;
        } else {
            return false;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_valid() {
        let valid = "[package]\nname = \"foo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n";
        assert_eq!(try_repair_toml(valid), None);
    }

    #[test]
    fn newline_collapse_run1() {
        // Real run_sm1_1.json content.
        let broken = r#"[package] name = "harness-sm1-r" version = "0.1.0" edition = "2024"

[dependencies] axum = { version = "=0.8", features=["json"] } serde_json.workspace-features=true tokio.workspace=true"#;
        let repaired = try_repair_toml(broken);
        assert!(repaired.is_some(), "expected repair on run1 collapse");
        // Sanity: repaired version parses
        let r = repaired.unwrap();
        assert!(toml::from_str::<toml::Value>(&r).is_ok());
    }

    #[test]
    fn preamble_strip_run4() {
        // Real run_sm1_4.json content (model prepended a fake warning).
        let broken = r#"cargo:warning: This is not valid TOML - it's a Cargo.toml file
[package]
name = "harness"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.8"
"#;
        let repaired = try_repair_toml(broken);
        assert!(repaired.is_some(), "expected repair on run4 preamble");
        let r = repaired.unwrap();
        assert!(r.starts_with("[package]"));
        assert!(toml::from_str::<toml::Value>(&r).is_ok());
    }

    #[test]
    fn preamble_strip_run7() {
        // Real run_sm1_7.json content (fake hipify error).
        let broken = r#"hipify-error[hipify-perl]: /tmp/harness-sm1-r7/Cargo.toml:1:0-23: hipErrorUnknown (unhandled error) [action] Run `--force` if you want this transformation to proceed despite unhandled errors (not recommended)."#;
        // Run 7 had ONLY the preamble — no real TOML after it. Strip
        // would fail (no `[` at start of line). Expect None.
        let repaired = try_repair_toml(broken);
        assert!(
            repaired.is_none(),
            "no TOML body after preamble → unrepairable"
        );
    }

    #[test]
    fn unrepairable_total_garbage() {
        // run_sm1_6 content: starts with stray `d`, mid-line broken
        // braces. Newline insertion can't fix nested-brace garbage.
        let broken = r#"d[package]name = "axum-ping-pong"version = "0.1.0"edition = "2021"[dependencies]axum = { version = "0.8", features = ["json"] }tokio  { version   , features   ]"#;
        // May or may not be reparable depending on aggressive regex —
        // the test asserts the function returns SOMETHING valid OR
        // None, and never panics.
        let _ = try_repair_toml(broken);
    }

    #[test]
    fn probe_r105_content_leak_missing_eq() {
        // r105 shape: collapsed one-line + `version.workspace true` (no =)
        // + trailing `</content>` XML leak.
        let broken = r#"[package] name = "pingpong" version.workspace true edition = "2024" [dependencies] axum = "0.8"</content>"#;
        let r = try_repair_toml(broken);
        eprintln!("PROBE r105 => {:?}", r);
    }

    #[test]
    fn probe_r110_content_leak() {
        // r110 shape: collapsed one-line + trailing `</content>` XML leak,
        // axum="0".
        let broken = r#"[package] name = "srv" version = "0.1.0" edition = "2024" [dependencies] axum = "0"</content>"#;
        let r = try_repair_toml(broken);
        eprintln!("PROBE r110 => {:?}", r);
    }

    #[test]
    fn probe_r4_pure_collapse() {
        // r4 shape: pure newline collapse, no XML leak. SC1 should fix.
        let broken = r#"[package] name = "x" version = "0.1.0" edition = "2024" [dependencies] axum = { version = "0.8", features=["json"] }"#;
        let r = try_repair_toml(broken);
        eprintln!("PROBE r4 => {:?}", r);
    }

    #[test]
    fn newline_collapse_run8() {
        // run_sm1_8.json — partial collapse (mid-line `version = "X" edition = "Y"`).
        let broken = r#"[package]
name = "ping-pong"
version = "0.1.0" edition = "2024"

[dependencies] axum = { version = "0", features=["json"] }
"#;
        let repaired = try_repair_toml(broken);
        assert!(repaired.is_some(), "expected repair on partial collapse");
        assert!(toml::from_str::<toml::Value>(&repaired.unwrap()).is_ok());
    }
}
