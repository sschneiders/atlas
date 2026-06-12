// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits: post-decode logits processing.

use super::*;

/// DIAG (ATLAS_DECODE_TIMING=1): localize the host-path decode cost. Splits the
/// per-token wall into `copy` (D2H of the full 248k-vocab logits + the GPU
/// forward-wait absorbed by that sync) vs `sample` (the host scalar loops over
/// 248k: BF16→FP32 expand + penalties + masks + argmax). Emits a 100-token
/// running summary. Zero-cost when the env var is unset (OnceLock-gated).
fn decode_timing_record(copy_us: u64, sample_us: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("ATLAS_DECODE_TIMING").is_ok()) {
        return;
    }
    static COPY: AtomicU64 = AtomicU64::new(0);
    static SAMPLE: AtomicU64 = AtomicU64::new(0);
    static CNT: AtomicU64 = AtomicU64::new(0);
    COPY.fetch_add(copy_us, Ordering::Relaxed);
    SAMPLE.fetch_add(sample_us, Ordering::Relaxed);
    let n = CNT.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(100) {
        let c = COPY.swap(0, Ordering::Relaxed);
        let s = SAMPLE.swap(0, Ordering::Relaxed);
        CNT.store(0, Ordering::Relaxed);
        tracing::info!(
            "DECODE_TIMING (last 100 host-path tokens): copy+fwd-wait={:.2}ms/tok sample(248k host)={:.2}ms/tok",
            c as f64 / 100_000.0,
            s as f64 / 100_000.0,
        );
    }
}

/// Sample and process decode logits for all active sequences.
///
/// Factored out of `step_decode_only` so that `mixed_forward` can reuse
/// the same sampling + token-processing logic without duplication (SSOT).
/// `logits` must point to `[n, vocab_size]` BF16 on device where n = active.len().
pub fn process_decode_logits(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    logits: DevicePtr,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
) {
    let n = active.len();

    // Grammar bitmask is CPU-side, so any sequence with active grammar forces
    // the host-side sampling path for its logits slice.
    let any_grammar = active.iter().any(|a| a.grammar_state.is_some());
    let any_logprobs = active.iter().any(|a| a.top_logprobs.is_some());
    // FP32 lm_head models (Gemma-4 dense) MUST use the host-side path —
    // `argmax_batch` assumes BF16 layout and would interpret 4-byte FP32
    // values as 2-byte BF16 pairs, returning garbage tokens.
    let model_logits_fp32 = model.decode_logits_fp32();
    let needs_host_logits = active
        .iter()
        .any(|a| a.inside_thinking || a.think_ended || a.grammar_state.is_some())
        || any_logprobs
        || model_logits_fp32;

    let new_tokens: Vec<(u32, Option<crate::api::TokenLogprobs>)> =
        if active.iter().all(|a| a.temperature == 0.0) && !any_grammar && !needs_host_logits {
            // Fast path: all greedy, no grammar, no thinking — GPU argmax for the full batch.
            match model.argmax_batch(logits, n, 0) {
                Ok(t) => t.into_iter().map(|tok| (tok, None)).collect(),
                Err(e) => {
                    tracing::error!("argmax_batch error: {e:#}");
                    for mut a in active.drain(..) {
                        send_error(model, &mut a, &format!("{e:#}"));
                    }
                    return;
                }
            }
        } else {
            // Host-side path: copy all batch logits to host, sample per-sequence.
            // Required when any sequence has temperature > 0 or grammar constraints.
            let vocab_size = model.vocab_size();
            // FP32 lm_head dispatch (Gemma-4 dense + ATLAS_GEMMA4_FP32_LMHEAD=1).
            // When the model writes FP32 logits to its decode-logits buffer, we
            // copy 4 bytes/element and skip the BF16→FP32 expansion. Earlier
            // bisection at model.rs:1192-1201 incorrectly concluded FP32 lm_head
            // had no effect on Gemma-4 because this dispatch was never wired —
            // the scheduler always read the (stale) BF16 logits buffer.
            // FP32 lm_head dispatch (Gemma-4 dense). When `use_fp32_logits` is
            // on, the per-token decode lm_head writes 4 bytes/element. The
            // passed `logits` pointer is whatever the most-recent forward
            // returned — that's already the correct buffer (prefill or decode).
            // We just need to read it with the matching width.
            let logits_fp32 = model.decode_logits_fp32();
            let elem_bytes = if logits_fp32 { 4 } else { 2 };
            let t_copy = std::time::Instant::now();
            let mut buf = vec![0u8; n * vocab_size * elem_bytes];
            if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
                tracing::error!("copy_logits_to_host error: {e:#}");
                for mut a in active.drain(..) {
                    send_error(model, &mut a, &format!("{e:#}"));
                }
                return;
            }
            let copy_us = t_copy.elapsed().as_micros() as u64;
            // SSOT: build the same `LogitsContext` the verify path passes
            // into `run_pipeline`, so `process_seq_logits` and the MTP
            // verify path share one pipeline-stage signature instead of
            // two divergent arg lists. `think_start_token` lives on the
            // per-seq `ActiveSeq` (read inside the pipeline stages), so it
            // is intentionally not carried in the context.
            let ctx = crate::scheduler::logit_processors::LogitsContext {
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
            };
            let t_sample = std::time::Instant::now();
            let sampled: Vec<(u32, Option<crate::api::TokenLogprobs>)> = active
                .iter_mut()
                .enumerate()
                .map(|(i, a)| {
                    process_seq_logits(
                        model,
                        a,
                        &buf,
                        i,
                        vocab_size,
                        elem_bytes,
                        logits_fp32,
                        &ctx,
                        adaptive_sampling,
                    )
                })
                .collect();
            decode_timing_record(copy_us, t_sample.elapsed().as_micros() as u64);
            sampled
        };
    let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let token_ids: Vec<u32> = new_tokens.iter().map(|(t, _)| *t).collect();
        tracing::debug!(
            "DECODE: n={n} step={step_ms:.1}ms ({:.1} tok/s) tokens={:?}",
            1000.0 * n as f64 / step_ms,
            token_ids,
        );
    }

    let now = Instant::now();
    for (i, (tok, logprobs)) in new_tokens.into_iter().enumerate() {
        let a = &mut active[i];
        a.last_token = tok;
        a.last_token_time = now;

        // Fix B (2026-06-05, kill-switch): <tool_response> hard stop. This decode
        // path has no `<|im_start|>` hard-stop block (that lives only in
        // emit_step.rs), so add the guard at the earliest safe point in the
        // per-token handler — before grammar advance / EOS handling. The model
        // must never generate this control token; if it does (post-tool-call
        // runaway), end the turn. Uses `continue` (loop body), not `return`.
        if tool_response_stop_enabled()
            && let Some(trs) = tool_response_hard_stop()
            && tok == trs
        {
            a.output_tokens.push(tok);
            a.finished = true;
            tracing::debug!("<tool_response> hard-stop fired (id={trs}); ending turn");
            continue;
        }

        // Spontaneous <think>: model generates <think> even when thinking
        // was not requested. Enter thinking mode so the thinking content
        // is stripped. Matches vLLM's behavior of always parsing
        // <think>...</think> regardless of enable_thinking setting.
        if !a.inside_thinking && think_start_token == Some(tok) {
            a.inside_thinking = true;
            a.think_ended = false; // reset so </think> detection path works
            a.think_skip_count = 0;
            a.thinking_budget = a.spontaneous_think_budget;
            tracing::debug!("Spontaneous <think> detected, entering thinking mode");
            continue; // don't emit <think> as content
        }

        // Silently skip </think> tokens outside thinking mode.
        // At long context (37k+), models degenerate into repeating </think>.
        // Skip up to 50 occurrences, then force-stop. This gives cached
        // prompts a chance to produce content while limiting degenerate loops.
        if !a.inside_thinking && think_end_token == Some(tok) {
            a.think_skip_count += 1;
            if a.think_skip_count >= 50 {
                a.finished = true;
            }
            continue;
        }
        // Reset skip counter when a real content token is generated.
        if a.think_ended {
            a.think_skip_count = 0;
        }

        // Advance grammar state with the sampled token — but only
        // once thinking is finished, because thinking tokens are
        // stripped from the API output and should not consume grammar
        // slots (matches the bitmask-skip in the sampler above).
        if !a.inside_thinking
            && let Some(ref mut gs) = a.grammar_state
        {
            gs.accept_token(tok);
        }

        // vLLM parity (2026-06-12): thinking tokens consume the generation
        // budget like any other output token — `max_tokens` bounds the
        // whole turn, reasoning included.
        if a.inside_thinking {
            a.consume_generation_budget();
            if think_end_token == Some(tok) {
                a.inside_thinking = false;
                a.force_end_thinking = false;
                a.sentence_defer_count = 0;

                a.in_code_fence = false;
                a.think_ended = true;
                // One-shot: pin the next sampled token to the
                // tool-call-start token if the request requires a
                // tool call (Change 3b). Cleared in the `else`
                // branch below on the next emit.
                a.think_just_ended = true;
            } else {
                a.thinking_tokens += 1;
                // Track ``` code-fence parity within the thinking block:
                // each fence token flips in/out of a fenced code span.
                // The F2 confidence early-stop (process_seq_logits) is
                // suppressed while `in_code_fence` — code is near-
                // deterministic (high top-1 prob) but that is NOT a
                // "done reasoning" signal; braking here truncates the
                // model mid-statement. THINK_LOOP (below) deliberately
                // stays active even inside fences: it catches
                // *repeating* fence-narration, not one coherent block.
                a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
                // Set force_end_thinking when budget exhausted (picked up next iteration)
                if let Some(budget) = a.thinking_budget
                    && a.thinking_tokens >= budget
                    && !a.force_end_thinking
                {
                    a.force_end_thinking = true;
                    a.sentence_defer_count = 0;
                    tracing::info!(
                        "Thinking budget exhausted ({budget} tokens), arming </think>; \
                         deferring up to {MAX_SENTENCE_DEFER_TOKENS} tokens for sentence boundary"
                    );
                }
            }
        } else {
            // Content-phase token: budget bookkeeping + the content-loop
            // and inter-tool-prose watchdogs. Extracted to
            // `decode_logits_content.rs` to keep this file ≤500 LoC.
            // `model` is threaded through so a watchdog rollback can
            // restore SSM recurrent state on hybrid models (Phase-C).
            handle_content_token(a, model);
        }

        // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
        // Guard with !inside_thinking — a <tool_call> inside thinking is spurious
        // and must not clear require_tool_call (which would allow premature EOS).
        if a.require_tool_call && tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.require_tool_call = false;
            a.tool_call_opened = true;
        }
        // Safety: if require_tool_call is still set after 512 tokens, the model
        // isn't generating a tool call (grammar may have failed to compile).
        // Clear the flag so EOS is no longer suppressed — prevents infinite gen.
        if a.require_tool_call && a.output_tokens.len() > 512 {
            tracing::warn!(
                "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
            );
            a.require_tool_call = false;
        }

        // Accumulate logprobs data for blocking responses.
        if let Some(lp) = logprobs {
            a.logprobs_data.push(lp);
        }

        // </tool_call> stop: in legacy mode (no grammar), stop after first tool call.
        // When grammar is active, allow the model to generate multiple tool calls —
        // the grammar controls when EOS is valid.
        if tool_call_end_token == Some(tok) && !a.inside_thinking {
            a.output_tokens.push(tok);
            // Fix A (2026-06-05): mark the tool call complete so the EOS-escape
            // gate (below) can lift suppression. Inert unless
            // `tool_eos_escape_enabled()` (default OFF).
            a.tool_call_completed = true;
            if let ResponseSink::Streaming(ref tx) = a.sink {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                match tx.try_send(event) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        tracing::warn!(
                            "Streaming receiver dropped during tool_call_end, finishing sequence"
                        );
                        a.finished = true;
                        continue;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                        if let Err(e) = tx.blocking_send(event) {
                            tracing::error!(
                                "Streaming send failed during tool_call_end backpressure: {e}"
                            );
                            a.finished = true;
                            continue;
                        }
                    }
                }
            }
            if a.grammar_state.is_none() {
                // Legacy mode: one tool call per response
                a.finished = true;
            }
            // Mirror finish_sequence (lines ~3445-3448): keep
            // `inside_tool_body` and the grammar FSM in sync with the
            // emitted token stream. The `continue;` below skips the
            // `emit_token()` path that would normally do this, so
            // without these two lines the flag stays `true` for all
            // subsequent prose tokens — sampler penalties stay
            // disabled for the rest of the response, and the grammar
            // bitmask drifts out of sync with the actual emission.
            // Root-caused 2026-04-26 (8-agent sweep, F1).
            a.inside_tool_body = false;
            if let Some(ref mut gs) = a.grammar_state {
                gs.accept_token(tok);
            }
            // F9 companion (2026-04-26): clear `think_ended` at every
            // </tool_call> boundary so legitimate post-tool
            // re-thinking is allowed. F9 masks <think> when
            // `think_ended=true`, but between tool calls the model
            // SHOULD be allowed to re-think (MiniMax-M2 / Qwen3.6
            // pattern per project_minimax_m27_final.md). F10's
            // watchdog-fire counter still applies — repeated
            // re-thinking that loops will decay its budget.
            a.think_ended = false;
            continue;
        }

        // EOS handling. vLLM parity (2026-06-12): a natural EOS always ends
        // the turn except when the request explicitly requires a tool call
        // that has not happened yet (`tool_choice="required"`/specific —
        // vLLM also forces the call there) or while `min_tokens` is unmet
        // (also a vLLM feature). The auto-mode grammar suppression, the
        // EOS-escape kill-switch, the in-thinking suppression, and the
        // POST_THINK_MIN_CONTENT guard were all removed — each had been
        // observed forcing the model past its natural stop into template
        // artefacts or hallucinated transcripts.
        let legacy_suppresses_eos = a.require_tool_call;
        let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
        let suppress_eos = legacy_suppresses_eos || min_tokens_suppresses;

        if a.eos_tokens.contains(&tok) && !suppress_eos {
            // Stop/EOS token: do NOT stream to client (OpenAI spec: returned text
            // must not contain the stop sequence). The token is still added to
            // output_tokens for correct token count; the API layer strips the
            // decoded text for blocking responses.
            a.output_tokens.push(tok);
            crate::scheduler::emit_step::update_tool_param_state(a, tok);
            a.finished = true;
        } else if a.eos_tokens.contains(&tok) && suppress_eos {
            // EOS suppressed: grammar not terminated or legacy tool call not yet seen.
            // Don't stop, don't stream the EOS — the model must keep generating.
            // Don't add to output_tokens (EOS is discarded).
        } else {
            a.output_tokens.push(tok);
            // SM1 (2026-05-26): drive the tool-body / parameter-body
            // state machine from the non-spec decode path. Previously
            // only spec/verify paths called this (via emit_token),
            // leaving every dependent gate (close-tag mask, AM1, B1,
            // A1) silently dead under `mtp=false`.
            crate::scheduler::emit_step::update_tool_param_state(a, tok);
            // #155 iter3: block-aligned Marconi checkpoint on the
            // non-MTP decode path (live SSM state is canonical here).
            if !a.inside_thinking {
                model.decode_marconi_checkpoint(&mut a.seq);
            }
            // OPENCODE FIX: when the model spontaneously emits `<think>` even
            // though the request didn't ask for thinking (`enable_thinking=false`),
            // the `<think>` open token itself is suppressed (line ~1356), but
            // the thinking-content tokens that follow MUST also be kept off the
            // wire — otherwise opencode persists them as `assistant.content` and
            // on the next turn the model sees its own past garbage (fake
            // `<function=…>`, fake `<tool_response>`) as a "format example" and
            // continues the pattern. Tokens stay in `output_tokens` for the
            // blocking response path's reasoning_content extraction.
            let suppress_stream = a.inside_thinking && !a.enable_thinking;
            if let ResponseSink::Streaming(ref tx) = a.sink
                && !suppress_stream
            {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                match tx.try_send(event) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        tracing::debug!(
                            "Streaming receiver dropped (decode_logits), finishing seq"
                        );
                        a.finished = true;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                        if let Err(e) = tx.blocking_send(event) {
                            tracing::error!(
                                "Streaming send failed during backpressure (decode_logits): {e}"
                            );
                            a.finished = true;
                        }
                    }
                }
            }
            if a.remaining == 0 {
                tracing::info!(
                    "process_decode_logits: remaining=0, output_tokens={}, thinking_tokens={}",
                    a.output_tokens.len(),
                    a.thinking_tokens
                );
                a.finished = true;
            }
            // Grammar termination = end of sequence. With `stop_after_first=true`
            // (tool_choice="required"), the structural-tag matcher transitions
            // to its terminal state right after the single tool call closes.
            // The model's free distribution past that point can be degenerate
            // (Nemotron-Super-120B emits a `</parameter>` loop and never
            // samples EOS naturally). Finish here instead of letting it run.
            if a.grammar_state
                .as_ref()
                .is_some_and(|gs| gs.is_terminated())
            {
                a.finished = true;
            }

            // Check request timeout.
            if !a.finished
                && let Some(deadline) = a.timeout_at
                && Instant::now() >= deadline
            {
                tracing::warn!("Request timeout after {:?}", a.request_start.elapsed());
                a.finished = true;
            }
        }
    }
}
