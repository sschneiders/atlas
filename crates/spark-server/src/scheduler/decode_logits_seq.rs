// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits per-sequence helper (extracted to keep parent file ≤500 LoC).

use super::*;

/// Process logits for a single active sequence: dequant, adjust, sample, return token + optional logprobs.
#[allow(clippy::too_many_arguments)]
pub fn process_seq_logits(
    _model: &dyn Model,
    a: &mut ActiveSeq,
    buf: &[u8],
    i: usize,
    vocab_size: usize,
    elem_bytes: usize,
    logits_fp32: bool,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
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

    // ── Pre-sample logits pipeline (Phase C-2 wiring, 2026-05-24) ──
    // The 8 pre-sample stages (F2 confidence arm, mid-word `</think>`
    // defer, post-close think mask, tool-call-during-thinking mask,
    // forced think-end injector, pin-to-tool-call-start, forced-token
    // fast-path, grammar bitmask) used to live as a ~290-line inline
    // block here. They are now composable [`LogitsProcessor`] impls in
    // `scheduler::logit_processors`. Semantics are byte-identical: each
    // processor was ported byte-for-byte from this site and is pinned
    // by `pipeline_tests::stage_names_are_distinct_and_stable`. See
    // `logit_processors/mod.rs` for the canonical stage order.
    //
    // F70 (2026-04-29, attempted): canonical-opener anchor bias was
    // REVERTED. xgrammar's TagDispatch is non-anchored (verified by
    // `grammar.rs::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`),
    // and a flat +2.5 logit boost on `tool_call_start_token` pushed
    // the model into the tool body too aggressively — observed live:
    // 1-tool prompts produced `tool_calls[0].function.arguments =
    // {"command":""}` because the model rushed through the envelope
    // with no parameter values. The proper fix is byte-level partial
    // trigger anchoring (mask trigger-breaker tokens only when a
    // partial-match suffix is actually present in recent output) but
    // that's a follow-up — for now we accept the "model occasionally
    // drifts on stressed prompts" limitation and rely on F26/F2 to
    // terminate the response cleanly when it happens.
    let ctx = crate::scheduler::logit_processors::LogitsContext {
        think_end_token,
        think_start_token,
        tool_call_start_token,
        tool_call_end_token,
    };
    if let Some(forced) =
        crate::scheduler::logit_processors::run_pipeline(&mut f32_logits, a, &ctx)
    {
        // Forced-token fast-path short-circuit: bit-identical to
        // sampling from an all-but-`forced`-masked logit vector. The
        // caller (`decode_logits_step`) still runs the SAME post-emit
        // accounting (output_tokens push, grammar advance, stop-token
        // handling), so all downstream state stays identical.
        return (forced, None);
    }

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
    // Phase-gated sampler scoping (P3.1, 2026-04-25):
    // inside the tool-call body (between `<tool_call>` and
    // `</tool_call>`) the JSON we emit is dense with
    // legitimate short repetitions — `":"`, `","`, key
    // tokens — that DRY/presence_penalty/frequency_penalty
    // would otherwise penalise, breaking schema validity.
    // XGrammar already guarantees structural correctness
    // here; penalties only add noise. Outside the tool
    // body (free text + `<think>`) the full preset
    // applies: this is where prose loops actually live.
    let in_tool = a.inside_tool_body && !a.inside_thinking;
    let sampled = sample_with_params_history(
        f32_bytes,
        &SamplingParams {
            temperature: sampling_temp,
            top_k: a.top_k,
            top_p: a.top_p,
            top_n_sigma: a.top_n_sigma,
            min_p: a.min_p,
            logit_bias: a.logit_bias.clone(),
            repetition_penalty: if in_tool { 1.0 } else { a.repetition_penalty },
            repetition_penalty_window: a.repetition_penalty_window,
            presence_penalty: if in_tool { 0.0 } else { a.presence_penalty },
            frequency_penalty: if in_tool { 0.0 } else { a.frequency_penalty },
            lz_penalty: if a.grammar_state.is_some() {
                0.0
            } else {
                a.lz_penalty
            },
            // DRY: same logic. Outside the tool body it
            // remains active to dampen `<think>` fence-narration
            // attractors. Inside the body, disabled — JSON
            // patterns repeat and that's correct.
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

    // Extract top-K logprobs from f32_logits if requested.
    let logprobs = a
        .top_logprobs
        .map(|k| extract_logprobs_from_f32(&f32_logits, sampled, k as usize));
    (sampled, logprobs)
}
