// SPDX-License-Identifier: AGPL-3.0-only
//
// Tier 5c (2026-05-26) — schema-validation re-roll.
//
// When `validate_tool_calls` flags a hard error AND `MODEL.toml
// [behavior].tool_retry = true` (the per-model default; overrideable
// per request via `ATLAS_TOOL_RETRY` env var), this module exposes the
// one-shot retry primitive used by both the blocking endpoint
// (`chat_blocking::build_choice_message`) and the streaming endpoint
// (`chat_stream::handle_done`).
//
// Retry prompt shape (2026-05-26 tuning):
//   `original_prompt_tokens` + correction_nudge_tokens
//                              ^^^^^^^^^^^^^^^^^^^^^^^
// The model's failed-attempt output is intentionally NOT included.
// Initial trace (atlas-gb10:wave1-stream5c-arc opencode probe) showed
// that with temp=0 greedy + failed output in context, the retry locks
// into reproducing the same drifted token sequence (e.g. a leading
// digit + closing brace contamination from prior tool-response text).
// A clean correction prompt at temp=0 puts the model in a fresh
// inference path. The nudge text already references "the exact target
// path the user requested in the original message" so the model has a
// pointer back to user intent without seeing its own bad emission.

use crate::AppState;
use crate::api::inference_types::{GrammarSpec, InferenceRequest};
use crate::tool_parser;

/// Build the retry prompt, submit one InferenceRequest::Blocking, parse
/// the response, run the same validation pipeline. Returns the valid
/// tool calls from the retry, or `None` if the retry could not produce
/// a passing tool call.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn attempt_tool_retry(
    state: &AppState,
    original_prompt_tokens: &[u32],
    errors_summary: &str,
    tools: &[tool_parser::ToolDefinition],
    cwd_hint: Option<&str>,
    grammar_spec: Option<GrammarSpec>,
    max_tokens: usize,
    timeout_at: Option<std::time::Instant>,
) -> Option<Vec<tool_parser::ToolCall>> {
    // The correction nudge: tokenize raw text and append after the
    // original prompt — no need to re-render the entire chat template,
    // which is expensive and would also drop the original from the
    // prompt-cache window. The failed-attempt output is NOT included
    // (see module-level doc): greedy reproduces it.
    let correction = format!(
        "<|im_end|>\n<|im_start|>user\n\
         Your previous tool call had a validation error: {errors_summary}\n\
         Retry exactly once. Use proper qwen3_coder XML: \
         `<parameter=NAME>VALUE</parameter>` blocks inside \
         `<tool_call>\\n<function=NAME>\\n…\\n</function>\\n</tool_call>`. \
         Use the exact target path the user requested in the original message. \
         Do not include XML attribute syntax (e.g. `filePath=\"…\"`); \
         do not embed `<tool_call>` text inside a parameter value.\n\
         <|im_end|>\n<|im_start|>assistant\n<think>\n",
    );
    let correction_tokens = match state.tokenizer.encode(&correction) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("Tier 5c: failed to tokenize correction nudge: {e}");
            return None;
        }
    };

    let mut retry_prompt: Vec<u32> =
        Vec::with_capacity(original_prompt_tokens.len() + correction_tokens.len());
    retry_prompt.extend_from_slice(original_prompt_tokens);
    retry_prompt.extend_from_slice(&correction_tokens);

    let (tx, rx) = tokio::sync::oneshot::channel();
    let request = InferenceRequest::Blocking {
        prompt_tokens: std::sync::Arc::new(retry_prompt),
        session_hash: 0,
        image_pixels: Vec::new(),
        max_tokens,
        min_tokens: 0,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias: Vec::new(),
        stop_tokens: Vec::new(),
        enable_thinking: true,
        thinking_budget: Some(512),
        repetition_detection: None,
        require_tool_call: true,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec,
        seed: Some(1),
        top_logprobs: None,
        timeout_at,
        response_tx: tx,
    };

    if state.request_tx.send(request).await.is_err() {
        tracing::warn!("Tier 5c: scheduler queue full — falling back to original error");
        return None;
    }

    let response = match rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            tracing::warn!("Tier 5c: retry inference error: {e}");
            return None;
        }
        Err(_) => {
            tracing::warn!("Tier 5c: retry cancelled");
            return None;
        }
    };

    let retry_text = state.tokenizer.decode(&response.output_tokens).ok()?;
    let (_, mut retry_calls) = tool_parser::parse_tool_calls(&retry_text);
    if retry_calls.is_empty() {
        return None;
    }
    tool_parser::backfill_required_params(&mut retry_calls, tools);
    if state
        .tool_call_parser
        .as_ref()
        .is_some_and(|p| p.wants_typed_arguments())
    {
        tool_parser::coerce_all(&mut retry_calls, tools);
    }
    if let Some(cwd) = cwd_hint {
        tool_parser::normalize_paths(&mut retry_calls, cwd);
    }
    let retry_validated = tool_parser::validate_tool_calls(retry_calls, tools);
    if retry_validated.valid.is_empty() {
        for err in &retry_validated.errors {
            tracing::warn!("Tier 5c: retry tool call ALSO failed: {err}");
        }
        return None;
    }
    if !retry_validated.errors.is_empty() {
        tracing::info!(
            "Tier 5c: retry produced {} valid + {} invalid tool-call(s); using the valid ones",
            retry_validated.valid.len(),
            retry_validated.errors.len()
        );
    }
    Some(retry_validated.valid)
}

/// Returns true when Tier 5c retry should fire for this server. Source
/// of truth: `MODEL.toml [behavior].tool_retry` (compiled into
/// `state.behavior.tool_retry` at build time, default `true`). The
/// `ATLAS_TOOL_RETRY` env var, if set, OVERRIDES the MODEL.toml default
/// — useful for A/B testing without rebuilding.
pub(crate) fn tool_retry_enabled(state: &AppState) -> bool {
    if let Ok(v) = std::env::var("ATLAS_TOOL_RETRY") {
        let truthy = v == "1" || v.eq_ignore_ascii_case("true");
        let falsy = v == "0" || v.eq_ignore_ascii_case("false");
        if truthy {
            return true;
        }
        if falsy {
            return false;
        }
    }
    state.behavior.tool_retry
}

/// A2 (2026-05-26) — Tier-2 fuzzy repair against the prompt vocabulary.
/// Cheaper than `attempt_tool_retry` (no inference round-trip) and
/// catches the dominant FP8 drift mode where the model substituted a
/// single byte in an identifier the user explicitly named (axum→axut,
/// hyphen drops). Runs BEFORE the inference retry — if it returns
/// `Some(repaired_calls)`, the retry is skipped entirely.
///
/// Returns `Some(repaired_calls)` only when EVERY originally-failed
/// call has a successful repair AND the repaired set re-validates.
/// Otherwise `None` so the caller can still fall through to the
/// inference retry.
pub(crate) fn attempt_fuzzy_repair(
    failed_calls: &[tool_parser::ToolCall],
    prompt_vocab: &std::collections::HashSet<String>,
    tools: &[tool_parser::ToolDefinition],
    cwd_hint: Option<&str>,
) -> Option<Vec<tool_parser::ToolCall>> {
    let mut repaired: Vec<tool_parser::ToolCall> = Vec::with_capacity(failed_calls.len());
    let mut any_repair = false;
    for tc in failed_calls {
        // Parse the JSON arguments; if it isn't an object we skip
        // — fuzzy repair only knows how to walk fields.
        let Ok(serde_json::Value::Object(map)) =
            serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
        else {
            repaired.push(tc.clone());
            continue;
        };
        let mut repaired_map = serde_json::Map::with_capacity(map.len());
        let mut tc_repaired = false;
        for (k, v) in map.iter() {
            match v {
                serde_json::Value::String(s) => {
                    if let Some(rep) = tool_parser::fuzzy_repair::repair_value(
                        s,
                        prompt_vocab,
                        tool_parser::fuzzy_repair::LEV_DEFAULT_MAX,
                    ) {
                        repaired_map.insert(k.clone(), serde_json::Value::String(rep));
                        tc_repaired = true;
                    } else {
                        repaired_map.insert(k.clone(), v.clone());
                    }
                }
                _ => {
                    repaired_map.insert(k.clone(), v.clone());
                }
            }
        }
        if tc_repaired {
            any_repair = true;
        }
        let new_args = serde_json::Value::Object(repaired_map).to_string();
        repaired.push(tool_parser::ToolCall {
            id: tc.id.clone(),
            call_type: tc.call_type.clone(),
            function: tool_parser::FunctionCall {
                name: tc.function.name.clone(),
                arguments: new_args,
            },
        });
    }
    if !any_repair {
        return None;
    }
    // Re-run the validation pipeline on the repaired set.
    let mut to_validate = repaired;
    if let Some(cwd) = cwd_hint {
        tool_parser::normalize_paths(&mut to_validate, cwd);
    }
    let revalidated = tool_parser::validate_tool_calls(to_validate, tools);
    if revalidated.valid.is_empty() || !revalidated.errors.is_empty() {
        return None;
    }
    tracing::info!(
        "A2 fuzzy_repair: rescued {} tool-call(s) via prompt-vocab Levenshtein match — no inference retry needed",
        revalidated.valid.len()
    );
    Some(revalidated.valid)
}

/// A2-AO (2026-05-26, /loop iteration 1) — ALWAYS-ON variant of fuzzy
/// repair. Runs in the parser pipeline BEFORE validation on every
/// successfully-parsed tool call. Catches drift #1 ("axum→axut",
/// "axum-r4→axu m/r4", "test-rust-axum→test/rust/axum") where the
/// path is structurally valid but semantically wrong — these never
/// trigger a hard validation error so the regular `attempt_fuzzy_repair`
/// (which gates on hard-error) never runs.
///
/// Conservatism: each value is repaired only when `repair_value`
/// returns Some (i.e. an unambiguous Lev≤2 match in prompt vocab
/// AND at least one word in the value got substituted). Already-in-
/// vocab values, ambiguous matches, and out-of-vocab tokens are left
/// alone. A single fire is logged at info; a periodic SUMMARY is left
/// for a future iteration if traffic gets high.
///
/// Mutates `calls` in place. No return value (no semantic decision
/// flows from "did anything change?"). The fuzzy-repaired calls then
/// proceed into the regular validate_tool_calls pipeline; any
/// remaining errors flow to Tier 5c as before.
pub(crate) fn apply_fuzzy_repair_inplace(
    calls: &mut [tool_parser::ToolCall],
    prompt_vocab: &std::collections::HashSet<String>,
) {
    let mut repairs_made = 0usize;
    for tc in calls.iter_mut() {
        let Ok(serde_json::Value::Object(map)) =
            serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
        else {
            continue;
        };
        let mut repaired_map = serde_json::Map::with_capacity(map.len());
        let mut tc_repaired = false;
        for (k, v) in map.iter() {
            match v {
                serde_json::Value::String(s) => {
                    if let Some(rep) = tool_parser::fuzzy_repair::repair_value(
                        s,
                        prompt_vocab,
                        tool_parser::fuzzy_repair::LEV_DEFAULT_MAX,
                    ) {
                        if rep != *s {
                            tracing::info!(
                                "A2-AO: {} field '{k}' repaired: {s:?} -> {rep:?}",
                                tc.function.name
                            );
                            repairs_made += 1;
                            tc_repaired = true;
                        }
                        repaired_map.insert(k.clone(), serde_json::Value::String(rep));
                    } else {
                        repaired_map.insert(k.clone(), v.clone());
                    }
                }
                _ => {
                    repaired_map.insert(k.clone(), v.clone());
                }
            }
        }
        if tc_repaired {
            tc.function.arguments = serde_json::Value::Object(repaired_map).to_string();
        }
    }
    if repairs_made > 0 {
        tracing::info!("A2-AO: total {repairs_made} field repair(s) this call");
    }
}
