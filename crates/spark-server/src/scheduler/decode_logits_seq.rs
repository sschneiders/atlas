// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits per-sequence helper (extracted to keep parent file ≤500 LoC).

use super::*;

/// B1 (2026-05-26): periodic summary of low-margin drift events.
///
/// Same shape as `verify_k2_step::k2_record_outcome`: an atomic counter
/// of low-margin firings since the last summary, plus a total. Every
/// `B1_SUMMARY_PERIOD` events we emit a single WARN line and reset.
/// Production observability for the FP8 low-margin argmax-flip regime
/// without spamming per-position trace.
const B1_SUMMARY_PERIOD: u64 = 100;
static B1_LOW_MARGIN_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn b1_record_low_margin(margin: f32, top1: u32, top2: u32) {
    let n = B1_LOW_MARGIN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    // Per-event trace — TRACE level by design (23.7% of long-ctx
    // positions trip this; INFO would spam). Power-user diagnostic:
    // `RUST_LOG=spark::scheduler::decode_logits_seq=trace`.
    tracing::trace!(
        "B1 low margin: gap={margin:.3} top1={top1} top2={top2}"
    );
    if n.is_multiple_of(B1_SUMMARY_PERIOD) {
        tracing::warn!(
            "B1 drift gauge: {n} low-margin (<1.5 logprobs) decode positions \
             observed inside parameter bodies. \
             FP8 numerical noise is in the argmax-flip regime — consider \
             reviewing tool-arg outputs for whitespace / digit-collapse drift."
        );
    }
}

/// Process logits for a single active sequence: dequant, adjust, sample, return token + optional logprobs.
#[allow(clippy::too_many_arguments)]
/// ATLAS_FORCE_TEMP_ZERO=1 — diagnostic mode that bypasses all drift
/// mitigation (WS1/AM1/WS2/A4/B1/C4) and just returns argmax of raw
/// logits. Used together with VLLM_FORCE_TEMP_ZERO on vLLM for
/// apples-to-apples layer-cosine comparison.
pub(crate) fn force_temp_zero_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ATLAS_FORCE_TEMP_ZERO")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

pub fn process_seq_logits(
    _model: &dyn Model,
    a: &mut ActiveSeq,
    buf: &[u8],
    i: usize,
    vocab_size: usize,
    elem_bytes: usize,
    logits_fp32: bool,
    think_end_token: Option<u32>,
    _think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    _tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
) -> (u32, Option<crate::api::TokenLogprobs>) {
    let slice = &buf[i * vocab_size * elem_bytes..(i + 1) * vocab_size * elem_bytes];
    let mut f32_logits: Vec<f32> = if logits_fp32 {
        // Direct FP32: 4 bytes/element little-endian.
        (0..vocab_size)
            .map(|j| {
                let off = j * 4;
                f32::from_le_bytes([slice[off], slice[off + 1], slice[off + 2], slice[off + 3]])
            })
            .collect()
    } else {
        // BF16 → FP32 expansion.
        (0..vocab_size)
            .map(|j| {
                let lo = slice[j * 2];
                let hi = slice[j * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };

    // ATLAS_FORCE_TEMP_ZERO short-circuit: pure argmax on raw logits,
    // no drift patches, no penalties, no biases. Matches vLLM at
    // temperature=0 (and VLLM_FORCE_TEMP_ZERO=1).
    if force_temp_zero_enabled() {
        let mut best_idx: u32 = 0;
        let mut best_val: f32 = f32::NEG_INFINITY;
        for (j, &v) in f32_logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = j as u32;
            }
        }
        let logprobs = a
            .top_logprobs
            .map(|k| extract_logprobs_from_f32(&f32_logits, best_idx, k as usize));
        return (best_idx, logprobs);
    }

    // F2: Confidence-based early stop during thinking.
    // When top-1 prob >= 0.95 for 30 consecutive tokens, force </think>.
    // Only kicks in after 400 thinking tokens — the model needs room to
    // plan (numbered lists, step-by-step reasoning have high per-token
    // confidence but are NOT signs the model is done thinking).
    // Previous thresholds (200 tokens, 10 consecutive) were too aggressive
    // and caused premature thinking termination in agentic coding sessions.
    //
    // Code-fence handling: a ``` block inside the reasoning is even
    // MORE confident than prose (Python/JSON syntax is near-
    // deterministic: `def`/`(`/`:`/indent/`return`), so 30 consecutive
    // ≥0.95 tokens trips trivially while the model is *productively*
    // drafting code. We still ARM the brake here (a model can ramble in
    // code forever — it must eventually stop), but the forced </think>
    // injection is DEFERRED until the fence closes (see
    // `should_inject_think_end` at the injection site below), so the
    // boundary lands cleanly right after the code block instead of
    // splitting a statement. The token-period THINK_LOOP watchdog
    // (decode_logits_step) also stays active in fences.
    if !crate::scheduler::helpers::disable_watchdogs()
        && a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_tokens >= 400
        && crate::scheduler::helpers::watchdog_params().confidence_early_stop
    {
        let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = f32_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let confident = sum_exp > 0.0 && 1.0 / sum_exp >= 0.95;
        let (run, force_end) = confidence_run_step(confident, a.consecutive_confident);
        a.consecutive_confident = run;
        if force_end {
            a.force_end_thinking = true;
            a.sentence_defer_count = 0;
            tracing::info!(
                "Confidence early stop armed: top-1 prob >= 0.95 for {} tokens (after {} thinking tokens){}",
                crate::scheduler::helpers::watchdog_params().confidence_run_length,
                a.thinking_tokens,
                if a.in_code_fence {
                    " — deferred until ``` fence closes"
                } else {
                    " — deferring until next sentence boundary"
                }
            );
        }
    }

    // Mid-word `</think>` defer (2026-05-24): while INSIDE thinking,
    // suppress `</think>` if the previously emitted token decodes to
    // text ending in alphanumeric — i.e. the model is mid-word and
    // closing thinking now would yield "creating thep" / "ping/pong en"
    // style cuts. Observed in opencode-session.md 2026-05-24: 8/8
    // thinking blocks ended mid-word on Qwen3.6-FP8.
    //
    // FP8 precision drift biases `</think>` logit upward by enough to
    // flip word-continuation losers at low margin. This guard restores
    // word boundaries without papering over the model's natural
    // decision to end thinking (the very next token after the word
    // closes, the model is free to sample `</think>` again — and the
    // continuation tokens (space / punctuation / newline) cap the
    // defer at one extra token most of the time).
    //
    // SSOT: the mid-word mask is the same source the tokenizer-runtime
    // build loop produces, paralleling boundary_token_mask. Fail-open:
    // mask absent → no suppression.
    if !crate::scheduler::helpers::disable_watchdogs()
        && a.inside_thinking
        && let Some(end_tok) = think_end_token
        && let Some(prev_tok) = a.output_tokens.last().copied()
        && let Some(mask) = crate::scheduler::helpers::mid_word_token_mask()
        && mask.get(prev_tok as usize).copied().unwrap_or(false)
    {
        let end_idx = end_tok as usize;
        if end_idx < f32_logits.len() {
            f32_logits[end_idx] = f32::NEG_INFINITY;
        }
    }

    // After thinking is done, suppress the </think> token to prevent
    // degenerate loops where the model generates hundreds of </think>.
    if a.think_ended {
        if let Some(end_tok) = think_end_token {
            let end_idx = end_tok as usize;
            if end_idx < f32_logits.len() {
                f32_logits[end_idx] = f32::NEG_INFINITY;
            }
        }
        // F9 (2026-04-26): symmetric mask for the START token.
        // Once `think_ended` is true (watchdog forced close OR
        // model emitted </think> naturally), the model must not
        // re-enter thinking in the same response. Without this
        // mask, the spontaneous-<think> re-entry path at the
        // emit site flips `inside_thinking=true` again on any
        // sampled <think>, and the watchdog fires again ~8s
        // later — observed three rapid re-entries on
        // 2026-04-26 fix29 logs. arXiv evidence: s1
        // (2501.19393), DeepSeek-R1, Qwen3 (2505.09388),
        // Production Repetition (2512.04419) all mask the
        // open token after first close. Chain-of-Draft
        // (2502.18600) ablates penalty stacking (12% drop) vs
        // hard masking (94% drop) — masking dominates.
        if let Some(start_tok) = a.think_start_token {
            let start_idx = start_tok as usize;
            if start_idx < f32_logits.len() {
                f32_logits[start_idx] = f32::NEG_INFINITY;
            }
        }
    }

    // Suppress <tool_call> during thinking (prevents KV cache contamination
    // from think-leak bug) AND when tool call loop detected (≥4 identical
    // calls — see api.rs:548). For the loop case, use a STRONG NEGATIVE
    // BIAS (−12.0) instead of `-inf` so the model can still escape if its
    // evidence for a tool call is overwhelming (e.g. user explicitly says
    // "actually run the tests"). For thinking, hard-mask remains: tool
    // calls inside <think> are unparsable per the (canonical) qwen3_coder
    // dialect, so they must be physically blocked.
    if a.inside_thinking {
        if let Some(tc_start) = tool_call_start_token {
            let idx = tc_start as usize;
            if idx < f32_logits.len() {
                f32_logits[idx] = f32::NEG_INFINITY;
            }
        }
    } else if a.suppress_tool_call
        && let Some(tc_start) = tool_call_start_token
    {
        let idx = tc_start as usize;
        if idx < f32_logits.len() {
            f32_logits[idx] -= 12.0;
        }
    }

    // Force </think> when budget exhausted OR confidence early stop
    // triggered — but DEFER while inside a ``` code fence so the
    // injection never splits a code statement (2026-05-17 thinkbrake
    // fix). The fence closes within a bounded number of tokens, then
    // this fires cleanly at the block boundary.
    // Bound the in-fence deferral: a model that writes its whole answer
    // as a code block inside <think> never closes the fence, so an
    // unbounded defer traps the deliverable in reasoning. Past
    // THINK_DEFER_BUDGET_FACTOR× the budget (or the absolute ceiling
    // when budget is None), inject </think> even mid-fence.
    // Compute the three deferral inputs for the injection gate.
    //   1. `defer_hard_override`: legacy budget-overrun ceiling — at
    //      3× thinking_budget or the absolute 2048 ceiling, inject
    //      regardless of fence/boundary state. Now ALSO fires when
    //      the sentence-boundary defer has been ticking for
    //      MAX_SENTENCE_DEFER_TOKENS steps without finding a period
    //      (model is dumping unpunctuated content — digits, identifiers,
    //      code-without-fences).
    //   2. `at_sentence_boundary`: previously-emitted token decoded to
    //      text ending in `.`/`!`/`?`/`\n`. The boundary mask is
    //      built at startup (`scheduler::helpers::boundary_token_mask`)
    //      and is the same mask Phase-C rollback uses, so the
    //      boundary semantics stay consistent across the codebase.
    //      Fail-open: when the mask is absent or the previous-token
    //      lookup misses, treat as "not at boundary" — the
    //      MAX_SENTENCE_DEFER_TOKENS ceiling guarantees forward
    //      progress.
    let at_sentence_boundary = a
        .output_tokens
        .last()
        .copied()
        .and_then(|prev_tok| {
            crate::scheduler::helpers::boundary_token_mask()
                .as_deref()
                .and_then(|m| m.get(prev_tok as usize).copied())
        })
        .unwrap_or(false);
    let defer_hard_override = match a.thinking_budget {
        Some(b) => a.thinking_tokens >= b.saturating_mul(THINK_DEFER_BUDGET_FACTOR),
        None => a.thinking_tokens >= THINK_DEFER_ABS_CEILING,
    } || a.sentence_defer_count >= MAX_SENTENCE_DEFER_TOKENS;
    if a.inside_thinking
        && should_inject_think_end(
            a.force_end_thinking,
            a.in_code_fence,
            at_sentence_boundary,
            defer_hard_override,
        )
        && let Some(end_tok) = think_end_token
    {
        let end_idx = end_tok as usize;
        if end_idx < f32_logits.len() {
            for logit in f32_logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            f32_logits[end_idx] = 0.0;
        }
    } else if a.inside_thinking && a.force_end_thinking {
        // Armed but deferring this step — tick the counter so the
        // MAX_SENTENCE_DEFER_TOKENS ceiling stays bounded.
        a.sentence_defer_count = a.sentence_defer_count.saturating_add(1);
    }

    // Change 3b: one-shot pin-to-tool-call-start.
    // When the previous token was `</think>` AND the request
    // requires a tool call AND no tool-call has been opened yet,
    // mask all logits to -inf except `tool_call_start_token`.
    // This prevents architectures like MiniMax M2 (which always
    // thinks via the chat template) from wandering into prose
    // after `</think>` instead of emitting the structured tool
    // call. Models that don't have `require_tool_call` set
    // (i.e. the request didn't pass tools) skip this entirely.
    if a.think_just_ended
        && a.require_tool_call
        && !a.tool_call_opened
        && !a.inside_thinking
        && let Some(start_tok) = tool_call_start_token
    {
        let idx = start_tok as usize;
        if idx < f32_logits.len() {
            for logit in f32_logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            f32_logits[idx] = 0.0;
            tracing::debug!("Forced tool_call_start_token after </think> (require_tool_call set)");
        }
    }

    // F70 (2026-04-29, attempted): canonical-opener anchor
    // bias was REVERTED. xgrammar's TagDispatch is non-anchored
    // (verified by
    // `grammar.rs::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`),
    // and a flat +2.5 logit boost on `tool_call_start_token`
    // pushes the model into the tool body too aggressively —
    // observed live: 1-tool prompts produce
    // `tool_calls[0].function.arguments = {"command":""}`
    // because the model rushes through the envelope with no
    // parameter values. The proper fix is byte-level partial
    // trigger anchoring (mask trigger-breaker tokens only when
    // a partial-match suffix is actually present in recent
    // output) but that's a follow-up — for now we accept the
    // "model occasionally drifts on stressed prompts"
    // limitation and rely on F26/F2 to terminate the response
    // cleanly when it happens.

    // ── Forced-token fast-path (xgrammar Tier 3b, Coalescence) ──
    // When the active tool-call grammar admits exactly one legal next
    // token, the model sample is redundant: the token is determined.
    // `forced_token()` returns `Some(id)` ONLY when the authoritative
    // next-token bitmask has a single set bit — so emitting `id`
    // directly is bit-identical to sampling from an all-but-`id`-masked
    // logit vector (every other token would be `-inf`). We skip the
    // O(vocab) bitmask fill *and* the O(vocab) CPU sampling scan for
    // these positions; this is the big win for structured tool-call
    // scaffolding (literal `<function=`, `</parameter>`, JSON
    // punctuation emit with no sampling work).
    //
    // GUARDS — the fast-path fires only when ALL hold:
    //  * not inside `<think>` — thinking is unconstrained (mirrors the
    //    bitmask-skip below; thinking tokens never advance the grammar).
    //  * the request actually has an active grammar (`grammar_state`).
    //  * the kill-switch is on (default; `ATLAS_DISABLE_FORCED_TOKEN`).
    //  * `top_logprobs` is NOT requested — logprobs are extracted from
    //    the model's logit distribution; the fast-path never builds it.
    //    Falling through to the normal masked-sample path keeps logprobs
    //    byte-identical (the all-but-one mask makes the sample return
    //    the same forced token anyway, so output is unchanged).
    //
    // The returned forced token still flows through the SAME caller
    // accounting as a sampled token — `decode_logits_step` pushes it to
    // `output_tokens`, calls `gs.accept_token`, runs stop-token / EOS /
    // streaming handling — so all downstream state is identical.
    // Tier-1 (Epoch 1+2) gate: do NOT use the forced-token fast-path
    // when we're inside a parameter body that has emitted zero content
    // tokens — the fast-path returns the grammar's sole legal token
    // directly without applying logit_bias, which bypasses our
    // anti-empty-parameter mask on token 510 (`</`). Per
    // `bench/fp8_dgx2_drift/research_synthesis.md`, this is exactly
    // the over-restrictive fastpath case A7 flagged at
    // `decode_logits_seq.rs:307-317`.
    let tier1_active = a.inside_parameter_body && a.param_body_chars_emitted == 0;
    if !a.inside_thinking
        && a.top_logprobs.is_none()
        && !tier1_active
        && crate::scheduler::helpers::forced_token_fastpath_enabled()
        && let Some(ref mut gs) = a.grammar_state
        && let Some(forced) = gs.forced_token()
    {
        // `forced` is the sole grammar-legal token; `forced_token`
        // returns only non-negative vocab ids (it reads them off the
        // packed bitmask). Emit directly — no mask fill, no sample.
        return (forced as u32, None);
    }

    // Apply grammar bitmask BEFORE sampling — but NOT during
    // `<think>`…`</think>`. Thinking is free-form reasoning that
    // is stripped from the final API response, so forcing it
    // through a JSON-tool-call grammar produces garbage
    // punctuation streams (observed with opencode: the assistant
    // thinking channel filled with `!.,),,,***` before the
    // model recovered after `</think>`).
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
        && gs.fill_bitmask()
    {
        gs.apply_bitmask_to_logits(&mut f32_logits);
    }

    // AdaDec Phase 1 diagnostic (no-op when ATLAS_ADADEC_DIAGNOSTIC unset).
    // Observes Shannon entropy of the post-grammar-bitmask distribution
    // from the MAIN decode path — pairs with the matching call in
    // `run_pipeline` (verify path) so we get coverage across both code
    // paths.
    crate::scheduler::logit_processors::adadec_diag::log_step(
        &f32_logits,
        a,
        "decode",
    );

    // F72 (byte-level partial-trigger anchor) was removed — see
    // F73 / fix42. The sampler-side anchor hung the server in
    // production despite passing isolated unit tests; the
    // model's broken-envelope case is now recovered at the
    // streaming-sanitizer + parser layer (F73 + F71). The
    // xgrammar non-anchored TagDispatch limitation is pinned
    // by `grammar.rs::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`
    // for documentation only.

    // ── Adaptive sampling: update zone, observe entropy, check greedy gate ──
    // Disabled by default (--adaptive-sampling flag). Each call scans the
    // full vocab (262k) on CPU: entropy O(V) exp+log, greedy gate O(V) exp.
    // Cost: ~300-400µs per token → 2-3x throughput regression when enabled.
    let greedy_gate = if adaptive_sampling {
        a.adaptive.update_zone(
            a.tool_call_opened,
            a.inside_thinking,
            a.grammar_state.is_some(),
        );
        a.adaptive.observe_entropy(&f32_logits);
        a.adaptive.update_lz_ratio(&a.output_tokens);
        a.adaptive.should_use_greedy(&f32_logits)
    } else {
        false
    };
    let effective_temp = if adaptive_sampling {
        a.adaptive.effective_temperature()
    } else {
        a.temperature
    };

    // Unified sampling path: stochastic OR greedy (temp==0 or
    // adaptive greedy_gate) both go through
    // `sample_with_params_history`. The function applies all
    // configured penalties (repetition / presence / frequency /
    // LZ / DRY) and logit_bias BEFORE the temperature decision,
    // so greedy argmax respects MODEL.toml-configured penalties
    // — matching HF Transformers / vLLM / llama.cpp behavior.
    //
    // The earlier "Pure greedy argmax — NO penalties" bypass
    // here was the load-bearing bug for Gemma-4-31B's greedy
    // fib failure: `MODEL.toml` configures rep_penalty=1.1 but
    // the bypass dropped it. After this change, the configured
    // penalty applies at temp=0.
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    // Force temp=0 for greedy_gate path (adaptive override) so
    // sample_with_params_seeded takes the post-penalty argmax
    // branch instead of running the full stochastic pipeline.
    let sampling_temp = if greedy_gate { 0.0 } else { effective_temp };
    // Advance seed per token for deterministic but varying randomness.
    let step_seed = a.seed.map(|s| s.wrapping_add(a.output_tokens.len() as u64));
    // A1 (2026-05-26): the prior Phase-3.1 logic (2026-04-25) zeroed
    // ALL penalties inside the tool-call body to avoid penalising
    // legitimate JSON structural repetition (`":"`, `","`, key
    // tokens). Live Wave-1+3 traces showed this was the dominant
    // root cause of the worst opencode attractors: runaway bash
    // commands with mismatched parens (`...) > ~/dir_setup.txt) >>
    // dir_done=true...` accumulating ~MB strings), `lean://` prefix
    // loops, and same-tool-call repetition. Without rep_penalty
    // inside the body, the model can emit the same degenerate
    // token cluster indefinitely.
    //
    // Per `research3_drift_catalog.md` top-3 attack: restore
    // penalties inside tool body at full strength. The original
    // JSON-structure concern is theoretically real but empirically
    // a non-issue at rep_pen=1.10 / window=256 — JSON tokens have
    // strong logit margins from XGrammar and the model's training,
    // so a 9% soft downweight does not flip their selection. The
    // runaway-attractor failure mode is the dominant cost.
    //
    // DRY stays zeroed inside the body: DRY's n-gram heuristic
    // (window 2-4 tokens) does meaningfully fight short JSON
    // repetitions and the win-to-cost trade-off is closer there.
    let in_tool = a.inside_tool_body && !a.inside_thinking;

    // Tier-1 (Epoch 1+2) parameter-body byte-counter mask: when the
    // model is inside `<parameter=KEY>…</parameter>` body AND no
    // CONTENT tokens have been emitted yet, suppress (a) token 510
    // (`</`, first token of `</parameter>` close), AND (b) the common
    // whitespace-only tokens 220 (` `), 198 (`\n`), 197 (`\t`), 256
    // (`  `), 271 (`\n\n`). The Epoch-2 v54 trial showed the model
    // bypassed the close-only mask by emitting a whitespace token
    // first (which the parser's `.trim()` strips at
    // `tool_parser/parse_single_b.rs:105`, yielding empty args after
    // the model "successfully" emits `</parameter>` on the next
    // token). Masking the common whitespace cluster forces the first
    // body token to be non-whitespace content.
    //
    // The Qwen3.6 vocab has many multi-byte whitespace tokens beyond
    // these 5, so this is not bulletproof — but it covers the
    // empirically-most-likely tokens the sampler picks in low-margin
    // distributions under FP8 drift. A future Tier 1b would do a
    // full vocab scan for whitespace-only tokens at boot.
    //
    // Bias of -8.0 is firm but not infinite — if the model has VERY
    // strong evidence (which it shouldn't given the structural
    // intent), the close can still win. State tracking lives in
    // `emit_step.rs` flag-flip block.
    let mut logit_bias_local = a.logit_bias.clone();
    if a.inside_parameter_body && a.param_body_chars_emitted == 0 {
        // Close-tag opener `</`
        logit_bias_local.push((510u32, -8.0f32));
        // WS1 (2026-05-26): full vocab-scanned whitespace set. Was a
        // 5-token literal `[220, 198, 197, 256, 271]`; the Qwen3.6 vocab
        // has ~440 whitespace-only tokens, so the historical mask let
        // ~98% of whitespace BPEs through. crate::whitespace_mask::init
        // runs once at boot and populates a process-global HashSet via
        // OnceLock; the read here is lock-free. Empty set if init
        // didn't run (e.g. unit tests) — preserves fail-open semantics.
        // Model-agnostic: any tokenizer's whitespace-only tokens land
        // in the set; non-Qwen models scan their own vocab.
        for &ws_tok in crate::whitespace_mask::whitespace_tokens() {
            logit_bias_local.push((ws_tok, -8.0f32));
        }
        // AM1 (2026-05-26): drift #7 "`lean://` prefix attractor"
        // suppression. At position 0 of a tool-param body, FP8 noise
        // routinely flips the first-content token to `lean` (id 2588)
        // or ` lean` (id 15192) on Qwen3.6. Same -8.0 firmness as the
        // close-tag opener; only triggers at position 0 so legitimate
        // mid-content `lean` survives.
        for &att_tok in crate::attractor_mask::attractor_tokens() {
            logit_bias_local.push((att_tok, -8.0f32));
        }
        // Diagnostic: log when the position-0 mask is active. INFO
        // level so it shows up under default RUST_LOG. One line per
        // tool-param body entry — not spammy. Helps verify the gate
        // fires when we expect.
        tracing::info!(
            "ws1/am1 mask active at param_body pos=0: {} ws + {} attractor + close-`</`",
            crate::whitespace_mask::whitespace_tokens().len(),
            crate::attractor_mask::attractor_tokens().len(),
        );
    } else if a.inside_parameter_body
        && a.param_body_chars_emitted > 0
        && a.output_tokens
            .last()
            .copied()
            .is_some_and(crate::whitespace_mask::is_digit_ending)
    {
        // WS2 (2026-05-26): mid-content whitespace gate.
        //
        // The Tier A drift mode (`0.1.0`→`0.1 .0`, `2024`→`2 024`) fires
        // mid-content where the position-0 WS1 mask never triggers. FP8
        // long-context noise (~23.7% of decode positions have top-1↔top-2
        // gap < 1.5 logprobs in long agentic context — see
        // bench/fp8_dgx2_drift/research_C1_results.md) lets a whitespace
        // continuation flip a low-margin number-continuation argmax.
        //
        // We trigger ONLY when the model just emitted a digit-ending token
        // (precomputed `is_digit_ending` bit per token id, built from
        // tokenizer at boot — model-agnostic). The bias is `-3.0`, much
        // lighter than the WS1 `-8.0` close-mask, because legitimate
        // whitespace-after-digit happens in math/code ("3 + 5", "10 items").
        // -3.0 is enough to flip an FP8-noise-driven low-margin selection
        // back to the higher-confidence digit/punct continuation but does
        // NOT override a genuinely-high-margin whitespace choice.
        for &ws_tok in crate::whitespace_mask::whitespace_tokens() {
            logit_bias_local.push((ws_tok, -3.0f32));
        }
    }
    // A4 (2026-05-26) POST_THINK_MIN_REASONING floor: suppress the
    // `</think>` token until at least MIN_REASONING_TOKENS thinking
    // tokens have been emitted. Closes the reasoning-collapse cascade
    // documented in research2_probe_forensics.md (reasoning_content
    // length decays 233→0 chars over 14 assistant turns). When the
    // model emits a vanishingly short `<think>` block, the downstream
    // tool emission lacks the planning context and drifts to phantom
    // paths / leaked control characters.
    //
    // Bias is -8.0 (firm but not infinite — same magnitude as the
    // empty-parameter mask above). If reasoning_budget is explicitly
    // set very small (e.g. <16) the request opted out of meaningful
    // thinking and the floor doesn't apply.
    const MIN_REASONING_TOKENS: u32 = 16;
    if a.inside_thinking
        && a.thinking_tokens < MIN_REASONING_TOKENS
        && a.thinking_budget.unwrap_or(MIN_REASONING_TOKENS) >= MIN_REASONING_TOKENS
        && let Some(end_tok) = think_end_token
    {
        logit_bias_local.push((end_tok, -8.0f32));
    }

    // B1 (2026-05-26): margin-ratio drift detector.
    //
    // We scan `f32_logits` for top-1 and top-2 (pre-internal-penalty,
    // pre-bias) and record the gap. In long-context FP8 decode
    // 23.7% of decode positions have gap<1.5 logprobs (see
    // bench/fp8_dgx2_drift/research_C1_results.md) — exactly the
    // regime where FP8 numerical noise flips a low-margin argmax.
    //
    // The detector is always-on (cost: one O(V) scan per token,
    // ~250k cmps ≈ 50µs on host f32 buffer). Output: optional
    // per-position TRACE event + periodic summary WARN when the
    // low-margin RATE exceeds threshold over a window (mirrors the
    // K2 drift gauge pattern in `verify_k2_step::k2_record_outcome`).
    //
    // Top-2 is also used by the C4v1 fallback below: when both
    // - we're inside a parameter body content position (mid-decode),
    // - and the margin is below `LOW_MARGIN_THRESHOLD`,
    // C4v1 weakens the natural top-1 bias so the sampler can land
    // on top-2 instead — breaking the deterministic FP8-flip
    // attractor without an expensive BF16 reverify forward pass.
    let (margin_top1_idx, margin_top1_val, margin_top2_idx, margin_top2_val) = {
        let mut t1_idx = 0u32;
        let mut t1_val = f32::NEG_INFINITY;
        let mut t2_idx = 0u32;
        let mut t2_val = f32::NEG_INFINITY;
        for (idx, &v) in f32_logits.iter().enumerate() {
            if v > t1_val {
                t2_val = t1_val;
                t2_idx = t1_idx;
                t1_val = v;
                t1_idx = idx as u32;
            } else if v > t2_val {
                t2_val = v;
                t2_idx = idx as u32;
            }
        }
        (t1_idx, t1_val, t2_idx, t2_val)
    };
    let margin = margin_top1_val - margin_top2_val;
    const LOW_MARGIN_THRESHOLD: f32 = 1.5;
    let low_margin_in_body =
        a.inside_parameter_body && a.param_body_chars_emitted > 0 && margin < LOW_MARGIN_THRESHOLD;
    if low_margin_in_body {
        b1_record_low_margin(margin, margin_top1_idx, margin_top2_idx);
    }

    // C4v1 (2026-05-26): DISABLED 2026-05-26 PM after the first live
    // probe showed the top-2 lift introduced a new drift mode
    // (reasoning text emitted inside `<parameter=content>` body
    // instead of legitimate TOML/code).
    //
    // Hypothesis: the C1 inference that "at low margin, top-1 and
    // top-2 are FP8 vs BF16 candidates" was based on a KL-symmetry
    // argument, not a direct measurement. In practice the top-2 is
    // often a SEMANTICALLY-different continuation that BF16 would
    // also have rejected; lifting it adds noise instead of correcting
    // FP8 drift. Need a per-token BF16 forward to verify which side
    // of the boundary is "right" before re-enabling.
    //
    // B1 (the detector above) stays active for visibility — it logs
    // low-margin positions without taking action. The C4v1 lift will
    // come back once we have a BF16 reference signal to gate it (the
    // real QSpec design from agent 2's research).
    //
    // Code retained below as a `false &&` block so the diff is easy
    // to revert.
    #[allow(clippy::overly_complex_bool_expr)]
    if false && low_margin_in_body {
        const C4_LIFT_FRACTION: f32 = 0.5;
        let lift = (LOW_MARGIN_THRESHOLD - margin) * C4_LIFT_FRACTION;
        if lift > 0.0 {
            logit_bias_local.push((margin_top2_idx, lift));
            tracing::trace!(
                "C4v1: lifting top2={margin_top2_idx} by +{lift:.3} (margin={margin:.3})"
            );
        }
    }

    // WS-mask diagnostic (#222): when ATLAS_WS_MASK_DIAG=1 and we're inside
    // a tool-param body, measure whether our whitespace/attractor logit_bias
    // DEMOTES a token the model actually wanted. Compares pre-mask argmax
    // (raw f32_logits) vs post-mask argmax (raw + logit_bias_local). If the
    // model's top-1 was a whitespace token and the bias flips it away, that
    // is the mechanism collapsing newlines inside TOML/code content. vLLM
    // never masks whitespace, so any flip here is a pure Atlas-vs-vLLM
    // divergence. Observe-only; does not change sampling.
    if a.inside_parameter_body
        && std::env::var("ATLAS_WS_MASK_DIAG").ok().as_deref() == Some("1")
        && !logit_bias_local.is_empty()
    {
        let pre_argmax = f32_logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            })
            .0;
        // Apply the local bias into a scratch copy to find post-mask argmax.
        let mut biased = f32_logits.clone();
        for &(tok, b) in &logit_bias_local {
            if let Some(slot) = biased.get_mut(tok as usize) {
                *slot += b;
            }
        }
        let post_argmax = biased
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            })
            .0;
        if pre_argmax != post_argmax
            && crate::whitespace_mask::is_whitespace(pre_argmax as u32)
        {
            tracing::info!(
                "WS_MASK_DIAG: param-body flip — model wanted ws tok {} (logit {:.3}) \
                 but mask demoted it to tok {} (logit {:.3}); chars_emitted={}",
                pre_argmax,
                f32_logits[pre_argmax],
                post_argmax,
                f32_logits[post_argmax],
                a.param_body_chars_emitted,
            );
        }
    }

    // Snapshot the applied bias for the logit dump BEFORE it's moved into
    // SamplingParams (only clones when ATLAS_LOGIT_DUMP is set).
    let dump_bias: Option<Vec<(u32, f32)>> = if super::logit_dump::enabled() {
        Some(logit_bias_local.clone())
    } else {
        None
    };

    let sampled = sample_with_params_history(
        f32_bytes,
        &SamplingParams {
            temperature: sampling_temp,
            top_k: a.top_k,
            top_p: a.top_p,
            top_n_sigma: a.top_n_sigma,
            min_p: a.min_p,
            logit_bias: logit_bias_local,
            // A1 (2026-05-26): full penalty INSIDE tool body too.
            // Stops attractor patterns (mismatched parens runaway,
            // `lean://` prefix loop, same-tool-call repetition).
            repetition_penalty: a.repetition_penalty,
            repetition_penalty_window: a.repetition_penalty_window,
            presence_penalty: a.presence_penalty,
            frequency_penalty: a.frequency_penalty,
            lz_penalty: if a.grammar_state.is_some() {
                0.0
            } else {
                a.lz_penalty
            },
            // DRY stays disabled inside tool body. DRY's n-gram
            // heuristic (window 2-4 tokens) DOES meaningfully
            // fight short JSON repetitions (`","`, `":"`), and
            // unlike rep_penalty's 9% soft downweight, DRY's
            // exponential schedule can flip token selection.
            dry_multiplier: if in_tool { 0.0 } else { a.dry_multiplier },
            dry_base: a.dry_base,
            dry_allowed_length: a.dry_allowed_length,
            dry_sequence_breakers: a.dry_sequence_breakers.clone(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: step_seed,
        },
        &a.output_tokens,
    );

    // Complete per-step logit dump (#222): ATLAS_LOGIT_DUMP=<file>. Captures
    // raw top-K + every applied bias + sampled, for Atlas↔vLLM divergence
    // analysis. Inert unless the env var is set.
    if let Some(bias) = dump_bias.as_ref() {
        super::logit_dump::record(
            a.output_tokens.len(),
            a.inside_parameter_body,
            a.param_body_chars_emitted as usize,
            &f32_logits,
            bias,
            sampled,
        );
    }

    // Extract top-K logprobs from f32_logits if requested.
    let logprobs = a
        .top_logprobs
        .map(|k| extract_logprobs_from_f32(&f32_logits, sampled, k as usize));
    (sampled, logprobs)
}
