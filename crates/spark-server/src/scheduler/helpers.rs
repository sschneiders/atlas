// SPDX-License-Identifier: AGPL-3.0-only

//! Helpers: BF16 conversion, hard-stop registry, sampling defaults.
//!
//! The loop-detection machinery that used to live here (thinking-loop /
//! content-loop / fuzzy-repeat detectors, `WatchdogParams`, the
//! numeric/boundary token masks, `ATLAS_DISABLE_WATCHDOGS`) was removed
//! 2026-06-12 for vLLM parity: a turn ends only on EOS, client stop
//! strings, `max_tokens`, or tool-call end.

/// Convert two little-endian BF16 bytes to f32.
#[inline]
pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
}

/// Global hard-stop token for ChatML role boundaries (`<|im_start|>`).
///
/// Set once at startup from `main.rs::set_im_start_hard_stop` when the
/// tokenizer exposes `<|im_start|>` as a single token id (Qwen3.5/3.6 family
/// tokenizers: id 248045). Read from `emit_token` to bail out of the turn
/// regardless of tool-call / min_tokens suppression — otherwise
/// the model can sample `<|im_start|>`, have it silently swallowed as a
/// suppressed EOS, and continue emitting the following role literal
/// (`user` / `assistant`, plain BPE tokens) which DO stream to the client.
///
/// 0 = unset / no hard-stop (non-Qwen tokenizers). The value is checked
/// with `load(Ordering::Relaxed)` on the emit path — no atomicity contract
/// beyond "set once before the first request lands", which is guaranteed
/// by the main.rs init ordering.
static IM_START_HARD_STOP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Install the ChatML role-boundary hard-stop. Called once from `main.rs`
/// at startup when `<|im_start|>` resolves to a single token id. Noop when
/// called with 0.
pub fn set_im_start_hard_stop(id: u32) {
    IM_START_HARD_STOP.store(id, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
pub fn im_start_hard_stop() -> Option<u32> {
    let id = IM_START_HARD_STOP.load(std::sync::atomic::Ordering::Relaxed);
    if id == 0 { None } else { Some(id) }
}

/// Fix B (2026-06-05): global hard-stop token for the `<tool_response>` control
/// token. Set once at startup from `tokenizer_runtime.rs` when `<tool_response>`
/// resolves to a single token id; mirrors the `<|im_start|>` hard-stop above.
/// 0 = unset / no hard-stop.
static TOOL_RESPONSE_HARD_STOP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub fn set_tool_response_hard_stop(id: u32) {
    TOOL_RESPONSE_HARD_STOP.store(id, std::sync::atomic::Ordering::Relaxed);
}
pub fn tool_response_hard_stop() -> Option<u32> {
    let id = TOOL_RESPONSE_HARD_STOP.load(std::sync::atomic::Ordering::Relaxed);
    if id == 0 { None } else { Some(id) }
}
// ── Sampling defaults (SSOT) ────────────────────────────────────────────────
// All SamplingParams constructors reference these constants. Change here, not
// at each call site.
pub const DEFAULT_LZ_PENALTY: f32 = 0.0;
pub const DEFAULT_DRY_MULTIPLIER: f32 = 0.0;
pub const DEFAULT_DRY_BASE: f32 = 1.75;
// Was 2 (oobabooga's reference value, optimised for free-form text).
// Bumped to 3 (2026-04-25) because at allowed_length=2 the DRY sampler
// penalises legitimate code micro-repetition (consecutive `(`, `,`,
// indentation, two-line `let x =` patterns) and breaks tool-call JSON
// emission. allowed_length=3 still catches the bash-fence
// "Running: …Executing: …" attractor (which spans 6+ tokens) while
// letting normal source-code patterns through. Per Agent 8 SOTA
// research, this matches the consensus for code workloads.
pub const DEFAULT_DRY_ALLOWED_LENGTH: u32 = 3;

// ── Grammar forced-token fast-path (xgrammar Tier 3b) ───────────────────────

/// Resolved kill-switch for the grammar forced-token (Coalescence)
/// fast-path. Computed once on first read from the environment.
///
/// The fast-path emits a token directly — skipping the model sample and
/// the vocab-wide bitmask fill — only when the active tool-call grammar
/// admits exactly one legal next token (xgrammar's `forced_token`
/// guarantees a single-bit mask). Output is therefore bit-identical to
/// the sampled path, so the fast-path is **on by default**.
///
/// `ATLAS_DISABLE_FORCED_TOKEN=1` (or `true`) forces it off — a
/// kill-switch should a future grammar/matcher regression ever make the
/// forced-token guarantee unsafe.
static FORCED_TOKEN_FASTPATH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Pure parse of the `ATLAS_DISABLE_FORCED_TOKEN` env value into the
/// resolved "fast-path enabled" boolean. Split out of
/// [`forced_token_fastpath_enabled`] so the parsing rule is unit-testable
/// without touching the process-wide `OnceLock`.
///
/// `None` (env unset) → enabled. A truthy value (`"1"` / `"true"`,
/// case-insensitive, surrounding whitespace ignored) → disabled.
/// Everything else (empty, `"0"`, `"false"`, junk) → enabled.
fn parse_forced_token_fastpath(env: Option<&str>) -> bool {
    match env {
        Some(v) => {
            let v = v.trim();
            !(v == "1" || v.eq_ignore_ascii_case("true"))
        }
        None => true,
    }
}

/// Resolve a "default-ON, explicit-disable" env flag. `None` (unset) → ON.
/// A falsy value (`"0"` / `"false"`, case-insensitive, trimmed) → OFF.
/// Everything else (`"1"`, `"true"`, junk) → ON. Mirrors the disable-idiom
/// of [`parse_forced_token_fastpath`].
fn env_flag_default_on(name: &str) -> bool {
    match std::env::var(name).ok().as_deref().map(str::trim) {
        Some(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
        None => true,
    }
}

/// Fix B (2026-06-05, baked default ON): hard-stop on the <tool_response>
/// control token (a token the model must never generate). Treated like a
/// stop token. Kill-switch: `ATLAS_TOOL_RESPONSE_STOP=0`/`false`.
pub fn tool_response_stop_enabled() -> bool {
    env_flag_default_on("ATLAS_TOOL_RESPONSE_STOP")
}

/// Tool-completion guard (2026-06-13, default ON; kill-switch
/// `ATLAS_TOOL_COMPLETION_GUARD=0`/`false`).
///
/// OPINIONATED, EXPERIMENTATION-DRIVEN departure from strict vLLM parity.
/// In `tool_choice="auto"` vLLM lets the model emit EOS at any time; a
/// turn that states an intended action then samples `<|im_end|>` without
/// emitting the tool call is "valid" and ends the turn. opencode masks
/// this by auto-continuing on a no-tool-call turn; Zed does not, so it
/// stops the agentic session prematurely (observed live 2026-06-13:
/// Qwen3.6-35B-A3B-FP8 emitted "Let me read native.rs…" then EOS, no tool
/// call, even at temp 0.3). This guard makes Atlas robust to clients that
/// do NOT auto-continue, the way opencode's orchestration is: in a
/// tool-active turn that has produced content but not yet a tool call, a
/// bare EOS is suppressed so the model continues to the tool call it was
/// about to emit — strictly BOUNDED to a small token window so a turn that
/// genuinely has nothing more to do still stops (no hallucinated-transcript
/// trap; see the bounded design in `emit_step::tool_completion_guard_*`).
/// This is a deliberate product choice, not a correctness requirement;
/// disable it for strict-parity deployments.
pub fn tool_completion_guard_enabled() -> bool {
    env_flag_default_on("ATLAS_TOOL_COMPLETION_GUARD")
}

/// Whether the grammar forced-token fast-path is enabled (default
/// `true`; disabled by `ATLAS_DISABLE_FORCED_TOKEN=1`/`true`).
pub fn forced_token_fastpath_enabled() -> bool {
    *FORCED_TOKEN_FASTPATH.get_or_init(|| {
        parse_forced_token_fastpath(std::env::var("ATLAS_DISABLE_FORCED_TOKEN").ok().as_deref())
    })
}

/// Per-token mid-word mask (2026-05-24): `mask[id]` is true iff the
/// token decodes to text whose LAST character is alphanumeric — i.e.
/// emitting `</think>` (or any sentence-end punctuation) right after
/// this token would split a word.
///
/// Used by the forced-`</think>` injector (client/CLI thinking budgets)
/// to defer the close to a word boundary rather than splitting a word.
///
/// Fail-open: never set → suppression is skipped and the model
/// retains full freedom to terminate thinking at any token.
static MID_WORD_TOKEN_MASK: std::sync::OnceLock<std::sync::Arc<[bool]>> =
    std::sync::OnceLock::new();

/// Set once at startup from the resolved tokenizer. Idempotent.
pub fn set_mid_word_token_mask(mask: std::sync::Arc<[bool]>) {
    let _ = MID_WORD_TOKEN_MASK.set(mask);
}

/// Read the mid-word token mask. `None` until `set_mid_word_token_mask`
/// runs — callers must treat `None` as "no mid-word info available"
/// and skip the suppression.
pub fn mid_word_token_mask() -> Option<std::sync::Arc<[bool]>> {
    MID_WORD_TOKEN_MASK.get().cloned()
}

/// `mask[id] == true` iff token `id` decodes to text ending in a
/// well-formed generation boundary — a newline, or sentence-ending
/// punctuation (`.`, `!`, `?`) optionally followed by a closing quote
/// or whitespace. Built once at startup from the tokenizer; drives the
/// forced-`</think>` injector's sentence-boundary deferral.
/// Fail-open: never set → the injector closes at the budget edge.
static BOUNDARY_TOKEN_MASK: std::sync::OnceLock<std::sync::Arc<[bool]>> =
    std::sync::OnceLock::new();

/// Set once at startup from the resolved tokenizer. Idempotent.
pub fn set_boundary_token_mask(mask: std::sync::Arc<[bool]>) {
    let _ = BOUNDARY_TOKEN_MASK.set(mask);
}

/// Read the boundary-token mask. `None` until `set_boundary_token_mask`
/// runs — callers must treat `None` as "no boundary info available".
pub fn boundary_token_mask() -> Option<std::sync::Arc<[bool]>> {
    BOUNDARY_TOKEN_MASK.get().cloned()
}

// ── vLLM-parity repetition detection (SamplingParams.repetition_detection) ──
//
// Port of vLLM's `v1/core/sched/utils.py::_has_repeating_pattern` +
// the `check_stop` repetition branch (vLLM >= v0.17.0). OPT-IN per
// request only — there is no server default and no heuristic gating;
// when the request carries no `repetition_detection`, this code never
// runs (vLLM parity).

/// vLLM `_has_repeating_pattern`: compares the last `pattern_len`
/// tokens against the preceding `min_count - 1` repetitions of the
/// same length. End-anchored — historic patterns the model has moved
/// past never match. Caller must ensure
/// `tokens.len() >= pattern_len * min_count`.
#[inline]
fn has_repeating_pattern_anchored(tokens: &[u32], pattern_len: usize, min_count: usize) -> bool {
    let n = tokens.len();
    for offset_in_window in 1..=pattern_len {
        let target = tokens[n - offset_in_window];
        for m in 1..min_count {
            let idx = n - (pattern_len * m + offset_in_window);
            if tokens[idx] != target {
                return false;
            }
        }
    }
    true
}

/// vLLM `check_sequence_repetition`: scan pattern lengths
/// `[max(min_pattern_size, 1) ..= max_pattern_size]` for an
/// end-anchored repeat of `min_count` copies. `max_pattern_size == 0`
/// disables (vLLM semantics; enforced again here so a non-validated
/// caller cannot enable a zero-width scan).
pub fn detect_sequence_repetition(
    tokens: &[u32],
    params: &crate::openai::RepetitionDetectionParams,
) -> bool {
    if params.max_pattern_size == 0 || params.min_count < 2 {
        return false;
    }
    let min_count = params.min_count as usize;
    let period_min = (params.min_pattern_size as usize).max(1);
    let period_max = params.max_pattern_size as usize;
    let n = tokens.len();
    for pattern_len in period_min..=period_max {
        if pattern_len * min_count > n {
            return false;
        }
        if has_repeating_pattern_anchored(tokens, pattern_len, min_count) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod repetition_tests {
    use super::detect_sequence_repetition;
    use crate::openai::RepetitionDetectionParams;

    fn params(min_p: u32, max_p: u32, min_c: u32) -> RepetitionDetectionParams {
        RepetitionDetectionParams {
            min_pattern_size: min_p,
            max_pattern_size: max_p,
            min_count: min_c,
        }
    }

    #[test]
    fn disabled_when_max_pattern_zero() {
        let toks = [1, 2, 1, 2, 1, 2, 1, 2];
        assert!(!detect_sequence_repetition(&toks, &params(0, 0, 3)));
    }

    #[test]
    fn detects_period_two_loop() {
        let toks = [9, 9, 9, 1, 2, 1, 2, 1, 2];
        assert!(detect_sequence_repetition(&toks, &params(1, 4, 3)));
    }

    #[test]
    fn end_anchored_ignores_historic_pattern() {
        // The period-2 repeat is followed by fresh non-repeating output:
        // an end-anchored detector must NOT fire (vLLM semantics).
        let toks = [1, 2, 1, 2, 1, 2, 7, 8, 9, 10, 11];
        assert!(!detect_sequence_repetition(&toks, &params(1, 4, 3)));
    }

    #[test]
    fn respects_min_count() {
        // Only 2 copies of the period-3 pattern at the tail.
        let toks = [4, 5, 6, 4, 5, 6];
        assert!(detect_sequence_repetition(&toks, &params(1, 4, 2)));
        assert!(!detect_sequence_repetition(&toks, &params(1, 4, 3)));
    }

    #[test]
    fn too_short_sequence_never_fires() {
        let toks = [1, 2];
        assert!(!detect_sequence_repetition(&toks, &params(1, 64, 3)));
    }

    #[test]
    fn period_one_token_run() {
        let toks = [3, 7, 7, 7, 7];
        assert!(detect_sequence_repetition(&toks, &params(1, 2, 4)));
    }
}
