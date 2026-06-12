// SPDX-License-Identifier: AGPL-3.0-only

//! emit_token + compile_grammar_state + StartPrefillResult enum.

use super::*;

/// Emit a token for an active sequence (stream + bookkeeping).
///
/// Per OpenAI spec, stop/EOS tokens are NOT streamed to the client —
/// the returned text must not contain the stop sequence. The token is
/// still recorded in output_tokens for accurate token counting.
///
/// When `logprobs` is Some, the logprobs data is accumulated for blocking
/// responses and sent via `StreamEvent::TokenWithLogprobs` for streaming.
pub fn emit_token(a: &mut ActiveSeq, tok: u32, logprobs: Option<crate::api::TokenLogprobs>) {
    // Cooperative cancellation from the streaming pipeline. The
    // stream-side loop guards (Bug-2 name-run cap, F11 within-dedup,
    // F44 perm-fail, loop-watchdog) flip this flag when they decide
    // the response should end. Treat it like an EOS: finalise now so
    // `handle_done` runs with the proper `tool_loop_capped` /
    // `finish_reason="length"` machinery, instead of letting the
    // model keep emitting tokens that just get suppressed.
    if let Some(ref f) = a.cancel_flag
        && f.load(std::sync::atomic::Ordering::Acquire)
    {
        a.finished = true;
        return;
    }

    // ChatML role-boundary HARD stop (`<|im_start|>`).
    //
    // Handled BEFORE grammar advance / EOS suppression: if the model
    // hallucinated a `<|im_start|>` mid-turn, we must end the turn regardless
    // of grammar / require_tool_call / min_tokens. The regular EOS path at
    // line ~3020 honors `suppress_eos`, which is true while a tool-call
    // grammar is active — so if we fell through to it, the tokenizer would
    // strip `<|im_start|>` (special-token) but the following role literal
    // (`user` / `assistant` — regular tokens) would stream to the client,
    // poisoning its context and causing the observed multi-turn drift /
    // "file was corrupted" hallucinations in opencode.
    if let Some(ims) = im_start_hard_stop()
        && tok == ims
    {
        // Push the hard-stop token to output_tokens so lifecycle.rs reports
        // `finish_reason="stop"` (because `<|im_start|>` is registered in
        // `eos_tokens` at startup — see tokenizer_runtime.rs::im_start_id).
        // Without this push, `last_tok = output_tokens.last()` is the prior
        // content token, lifecycle's `is_eos` check fails, and the response
        // is mis-reported as `finish_reason="length"` (Bug 3 from OpenClaw
        // 2026-05-08 session: "Done: 13 tokens (length) despite max_tokens=
        // 8192" — clients then misinterpret the truncation as a real
        // length-limit hit and either retry or surface a wrong error).
        // The streamed-text path strips stop tokens server-side, so the
        // client never sees the literal `<|im_start|>` bytes.
        a.output_tokens.push(tok);
        a.finished = true;
        tracing::debug!(
            "<|im_start|> hard-stop fired (id={ims}); ending turn before grammar/suppress_eos"
        );
        return;
    }

    // Fix B (2026-06-05, kill-switch): <tool_response> hard stop — the model must
    // never generate this control token; if it does (post-tool-call runaway), end
    // the turn. Mirrors the <|im_start|> hard stop above.
    if tool_response_stop_enabled()
        && let Some(trs) = tool_response_hard_stop()
        && tok == trs
    {
        a.output_tokens.push(tok);
        a.finished = true;
        tracing::debug!("<tool_response> hard-stop fired (id={trs}); ending turn");
        return;
    }

    // Spontaneous <think>: model generates <think> even when thinking was not
    // requested. Enter thinking mode so EOS is suppressed and thinking content
    // is stripped. This handles MTP bootstrap/verify paths.
    if !a.inside_thinking && a.think_start_token == Some(tok) {
        a.inside_thinking = true;
        a.think_ended = false;
        a.think_skip_count = 0;
        a.thinking_budget = a.spontaneous_think_budget;
        tracing::debug!("Spontaneous <think> detected in emit_token, entering thinking mode");
        return; // don't emit <think> as content
    }

    // Silently skip </think> tokens outside thinking mode (same as process_decode_logits).
    if !a.inside_thinking && a.think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
        }
        return;
    }

    // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
    // Guard with !inside_thinking — tool calls inside thinking are spurious.
    if a.require_tool_call && a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }

    // Tool-body / parameter-body state machine.
    //
    // SM1 (2026-05-26): extracted from inline-in-emit_token to the free
    // function `update_tool_param_state` so the regular non-MTP decode
    // path (`decode_logits_step.rs`) can call it too. Previously the
    // state was ONLY updated when `emit_token` ran — which happens from
    // spec/verify paths but NOT from `process_decode_logits`. With
    // mtp=false (Qwen3.6 baseline), the state machine never ran and
    // every dependent gate (A1 rep-penalty toggle, B1 margin detector)
    // was silently dead code. (The pos-0 close-tag/AM1 logit-bias that
    // also depended on this state was removed 2026-06-03; the state is
    // still required for A1/B1 and the adadec_diag dump.)
    update_tool_param_state(a, tok);

    // Mark a tool call complete on `</tool_call>` (outside thinking) so the
    // EOS path below can allow a natural end-of-turn.
    if a.tool_call_end_token == Some(tok) && !a.inside_thinking {
        a.tool_call_completed = true;
    }

    // Advance grammar state with the emitted token — skip while the
    // sequence is inside `<think>`…`</think>` so the matcher only
    // sees the final-output token stream.
    let mut disengage_grammar = false;
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        let advanced = gs.accept_token(tok);
        if !advanced {
            // Grammar/model disagreement (BUG#2 class: e.g. a merged BPE token
            // like `><` or a `</X` content run the qwen3_coder value rule
            // forbids, often surfaced via an under-masked MTP draft). The token
            // is already a legitimate model emission; the matcher is now
            // desynced. Previously we set `a.finished = true` here — a
            // CATASTROPHIC cliff that lost the ENTIRE agentic turn on a single
            // refused token (root cause of the opencode webserver_ok gap).
            // Instead, DISENGAGE the grammar for the remainder of this response
            // and continue decoding UNCONSTRAINED — exactly what vLLM (the 10/10
            // reference) does by parsing tools post-hoc. Atlas's server-side
            // tool parser still extracts tool calls from the emitted text, so
            // the structural guarantee is gracefully traded for turn survival.
            tracing::warn!(
                tok,
                output_len = a.output_tokens.len(),
                "gs.accept_token returned false — grammar/model disagreement; disengaging grammar for the remainder of this response (free decode + post-hoc tool parse) instead of aborting the turn."
            );
            disengage_grammar = true;
        }
    }
    if disengage_grammar {
        // Drop the matcher: subsequent decode steps see `grammar_state == None`
        // and decode unconstrained. Set after the `ref mut gs` borrow ends.
        a.grammar_state = None;
    }

    // Accumulate logprobs data for blocking responses.
    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    a.output_tokens.push(tok);
    check_repetition_stop(a);

    // Detect </think> transition. Track thinking token count for budget
    // enforcement. vLLM parity (2026-06-12): thinking tokens consume the
    // generation budget like any other output token — `max_tokens` is the
    // bound on the whole turn, reasoning included. (They were previously
    // "free", which made an unbudgeted thinking phase unbounded.)
    if a.inside_thinking {
        a.consume_generation_budget();
        if a.think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.think_ended = true;
            // One-shot for the next decode step: pin to
            // tool_call_start_token if require_tool_call (Change 3b).
            a.think_just_ended = true;
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        } else {
            a.thinking_tokens += 1;
            if let Some(budget) = a.thinking_budget
                && a.thinking_tokens >= budget
                && !a.force_end_thinking
            {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(
                    "Thinking budget exhausted ({budget} tokens), arming </think>; \
                     deferring to next sentence boundary"
                );
            }
        }
    } else {
        a.consume_generation_budget();
        // Clear think_just_ended one-shot now that we've consumed the
        // token after </think>.
        a.think_just_ended = false;
    }

    // EOS handling. vLLM parity (2026-06-12): in `tool_choice="auto"` a
    // natural EOS always ends the turn — the trigger-gated grammar's
    // `is_terminated()` never becomes true in auto mode, and suppressing
    // EOS on it trapped the model in hallucinated-transcript runaways
    // (it could neither act nor stop). EOS is suppressed only when the
    // request explicitly requires a tool call that has not happened yet
    // (`tool_choice="required"`/specific — vLLM also forces the call
    // there) or while `min_tokens` is unmet (also a vLLM feature).
    let legacy_suppresses_eos = a.require_tool_call;
    let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
    let suppress_eos = legacy_suppresses_eos || min_tokens_suppresses;

    if a.eos_tokens.contains(&tok) && !suppress_eos {
        a.finished = true;
        return;
    }
    if a.eos_tokens.contains(&tok) && suppress_eos {
        // EOS suppressed: grammar not terminated, legacy tool call not yet seen,
        // or min_tokens not reached. Don't stop — let the model continue generating.
        return;
    }
    // OPENCODE FIX: see process_decode_logits — same gate. Suppress streaming
    // of spontaneous-thinking content so it doesn't pollute opencode's history.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if let ResponseSink::Streaming(ref tx) = a.sink
        && !suppress_stream
    {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        // Discriminate transient backpressure (channel full) from a real
        // consumer-drop (channel closed). The previous `try_send().is_err()`
        // collapsed the two and silently terminated the seq with
        // `finish_reason="length"` whenever the SSE consumer momentarily
        // stalled — surfaced as "request stops half-way" in Open WebUI.
        match tx.try_send(event) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!("Streaming receiver dropped, finishing seq");
                a.finished = true;
                return;
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                if let Err(e) = tx.blocking_send(event) {
                    tracing::error!("Streaming send failed during backpressure: {e}");
                    a.finished = true;
                    return;
                }
            }
        }
    }
    if a.remaining == 0 {
        tracing::info!(
            "emit_token: remaining=0, output_tokens={}, thinking_tokens={}",
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
}

// F72 (byte-level partial-trigger anchor) was removed in F73 / fix42.
// The sampler-side anchor hung the server in production; the broken
// envelope is now recovered at the streaming-sanitizer + parser
// layer. xgrammar's non-anchored TagDispatch limitation is pinned
// for documentation by
// `grammar.rs::tests::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`.

/// Compile a grammar state from a grammar specification + engine.
///
/// Returns `Some(GrammarState)` if compilation succeeds, `None` otherwise
/// (logging a warning on failure so the request falls back to legacy tool_call
/// suppression). Called once per request during prefill.
pub fn compile_grammar_state(
    engine: &mut Option<GrammarEngine>,
    grammar_spec: &Option<GrammarSpec>,
    eos_tokens: &[u32],
) -> Option<GrammarState> {
    let spec = grammar_spec.as_ref()?;
    let engine = engine.as_mut()?;

    // F69 (2026-04-29): symmetric dispatch via the trait. The parser
    // is the single source of truth for both runtime parsing and
    // grammar compilation; no string match keyed on `parser_name`.
    // Mistral's default trait impl returns `None`, which we treat as
    // "no constraint, fall through to unconstrained decoding."
    let compiled = match spec {
        GrammarSpec::ToolCall {
            tools,
            parser,
            use_triggers,
        } => match parser.compile_tool_grammar(engine, tools, *use_triggers) {
            Some(result) => result,
            None => {
                tracing::debug!(
                    "Grammar: parser '{}' opted out of constrained decoding for this request",
                    parser.name(),
                );
                return None;
            }
        },
        GrammarSpec::JsonObject => engine.compile_json_grammar(),
        GrammarSpec::JsonSchema { schema } => engine.compile_json_schema(schema),
    };

    let label = match spec {
        GrammarSpec::ToolCall { parser, tools, .. } => {
            format!("parser={}, tools={}", parser.name(), tools.len())
        }
        GrammarSpec::JsonObject => "response_format=json_object".to_string(),
        GrammarSpec::JsonSchema { .. } => "response_format=json_schema".to_string(),
    };

    match compiled {
        Ok(grammar) => {
            let vocab_size = engine.vocab_size();
            match GrammarState::new(&grammar, vocab_size) {
                Ok(state) => {
                    tracing::info!("Grammar constrained decoding active: {label}");
                    // Exempt the model's stop/EOS tokens from grammar refusal
                    // so a legitimate end-of-turn token cannot desync the NPDA
                    // and truncate the response (see GrammarState::accept_token).
                    Some(state.with_stop_tokens(eos_tokens))
                }
                Err(e) => {
                    tracing::warn!("Grammar state creation failed: {e}");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!("Grammar compilation failed: {e}");
            None
        }
    }
}

/// Result of starting a chunked prefill.
pub enum StartPrefillResult {
    /// Prompt fit in one chunk → ready for decode.
    Active(ActiveSeq),
    /// Prompt needs more chunks → add to prefilling[].
    InProgress(PrefillInProgress),
    /// Completed during first chunk (EOS on first token).
    Finished,
}

// Tool-body / parameter-body state machine, hoisted out of
// `emit_token` (SM1, 2026-05-26).
//
// Both speculative-decoding paths (`verify_k2_step`, `verify_k4_step`,
// `verify_dflash_step`, `spec_step`) and the regular non-spec decode
// path (`decode_logits_step::process_decode_logits`) call this on
// every emitted token so the state machine stays in sync with
// `a.output_tokens`. The previous inline version was unreachable
// from the non-spec path, leaving the close-tag mask, AM1 attractor
// suppression, B1 margin detector, and A1 penalty toggle all silently
// dead.
//
// **Slice semantics**: this function does NOT assume `tok` has been
// pushed onto `a.output_tokens` or that it has not. It auto-detects
// from `a.output_tokens.last()` and slices accordingly:
//  - `emit_token` calls this BEFORE pushing → `last()` is the prior
//    token, lookback uses the full slice.
//  - `decode_logits_step::process_decode_logits` calls this AFTER
//    pushing → `last()` is `tok`, lookback excludes the trailing
//    entry so the search for `<parameter=KEY>` ending at the current
//    `>` is correct in both cases.
//
// State mutations:
//  - `a.inside_tool_body`         set on `<tool_call>`, cleared on `</tool_call>`
//  - `a.inside_parameter_body`    set on `<parameter=KEY>` close `>`, cleared on `</`
//  - `a.param_body_chars_emitted` ++ per non-close body token
//
// (The MAX_TOOL_BODY_TOKENS envelope-stuck guard was removed 2026-06-12
// for vLLM parity — a never-closing envelope now burns to max_tokens,
// exactly as it would on vLLM, and the post-hoc parser salvages.)
//
// Token IDs are Qwen3.6 byte-level BPE (verified via /tokenize 2026-05-25):
//   27 = `<`, 28 = `=`, 29 = `>`, 510 = `</`, 15704 = `parameter`.

/// vLLM-parity repetition stop — the `check_stop` repetition branch
/// (`v1/core/sched/utils.py`, vLLM >= v0.17.0). Called after each
/// committed token from BOTH decode paths (`emit_token` for MTP/verify,
/// `process_decode_logits` for non-MTP — same SSOT pattern as
/// `update_tool_param_state`). Inert unless the request opted in via
/// `repetition_detection`; honors `min_tokens` like vLLM's check order.
pub fn check_repetition_stop(a: &mut ActiveSeq) {
    let Some(params) = a.repetition_detection else {
        return;
    };
    if a.finished || a.output_tokens.len() < a.min_tokens {
        return;
    }
    if crate::scheduler::helpers::detect_sequence_repetition(&a.output_tokens, &params) {
        tracing::info!(
            output_len = a.output_tokens.len(),
            max_pattern_size = params.max_pattern_size,
            min_count = params.min_count,
            "repetition detected (vLLM-parity repetition_detection); ending \
             generation with finish_reason=\"repetition\""
        );
        a.finish_reason_override = Some("repetition");
        a.finished = true;
    }
}

pub fn update_tool_param_state(a: &mut ActiveSeq, tok: u32) {
    if a.inside_thinking {
        return;
    }
    if a.tool_call_start_token == Some(tok) {
        a.inside_tool_body = true;
        return;
    }
    if a.tool_call_end_token == Some(tok) {
        a.inside_tool_body = false;
        a.inside_parameter_body = false;
        a.param_body_chars_emitted = 0;
        return;
    }
    if !a.inside_tool_body {
        return;
    }

    const TOK_LT: u32 = 27;
    const TOK_PARAMETER: u32 = 15704;
    const TOK_EQ: u32 = 28;
    const TOK_GT: u32 = 29;
    const TOK_LT_SLASH: u32 = 510;

    if a.inside_parameter_body {
        if tok == TOK_LT_SLASH {
            // Start of `</parameter>` close-tag — exit body.
            a.inside_parameter_body = false;
            a.param_body_chars_emitted = 0;
        } else {
            // Any non-close body token advances the counter. The
            // position-0 mask in `decode_logits_seq.rs` (close-tag +
            // AM1 attractor) fires only while this counter is 0, so it
            // deactivates after the first emitted body token.
            a.param_body_chars_emitted = a.param_body_chars_emitted.saturating_add(1);
        }
        return;
    }

    // Not yet inside_parameter_body: scan for `<parameter=KEY>` opener
    // ending at this `>` (29). Lookback 8 tokens for `[27, 15704, 28]`
    // signature without an intervening close.
    if tok != TOK_GT {
        return;
    }
    // Auto-detect whether `tok` is already in output_tokens (caller
    // pushed) or not (caller has not yet pushed). The signature search
    // must NOT include `tok` itself — the lookback is "what came
    // BEFORE this `>`".
    let n = a.output_tokens.len();
    let n_for_lookback = if n > 0 && a.output_tokens[n - 1] == tok {
        n - 1
    } else {
        n
    };
    if n_for_lookback < 3 {
        return;
    }
    let start = n_for_lookback.saturating_sub(8);
    let window = &a.output_tokens[start..n_for_lookback];
    let mut sig_idx: Option<usize> = None;
    for i in 0..window.len().saturating_sub(2) {
        if window[i] == TOK_LT && window[i + 1] == TOK_PARAMETER && window[i + 2] == TOK_EQ {
            sig_idx = Some(i + 3);
        }
    }
    let Some(after_eq) = sig_idx else { return };
    // Check no intervening `</` or `>` in the KEY span between
    // `<parameter=` and the current `>`.
    let body_segment = &window[after_eq..];
    let intervening_close = body_segment
        .iter()
        .any(|&t| t == TOK_LT_SLASH || t == TOK_GT);
    if !intervening_close {
        a.inside_parameter_body = true;
        a.param_body_chars_emitted = 0;
    }
}

// SM1 unit tests deferred: ActiveSeq has 60+ fields and no public
// constructor; building a test instance requires more boilerplate
// than the state machine itself. Live-verification post-deploy is via
// the A1 rep-penalty toggle / B1 margin-detector behaviour.
