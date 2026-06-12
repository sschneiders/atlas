// SPDX-License-Identifier: AGPL-3.0-only
//
// Helpers for the four `DetectorOutput` variants emitted by the
// streaming tool-call detector. Shared by both `handle_token` (mid-
// stream `process()` outputs) and `handle_done` (end-of-stream
// `flush()` outputs).

use axum::response::sse::Event;

use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::super::stream_guards::flush_content_sanitizer;
use super::ctx::StreamCtx;
use super::state::{PendingRetry, StreamState};

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

/// Tier 5c (2026-05-26): emit `chunk_json` to either the client SSE
/// stream OR a per-tool-call-index buffer in `StreamState`. When tool
/// retry is enabled we hold all tool_call SSE chunks until
/// `handle_tool_call_delta` runs validation; on pass the buffered chunks
/// flush to the client, on fail they're discarded and the retry fires
/// at `handle_done`. When tool retry is disabled this is a direct emit
/// (preserves the existing real-time streaming behaviour).
fn emit_or_buffer_tool_chunk(
    state: &mut StreamState,
    ctx: &StreamCtx,
    idx: usize,
    chunk_json: String,
    sse_events: &mut SseVec,
) {
    if ctx.tool_retry_enabled {
        state
            .buffered_tool_chunks
            .entry(idx)
            .or_default()
            .push(chunk_json);
    } else {
        sse_events.push(Ok(Event::default().data(chunk_json)));
    }
}

/// Flush all buffered SSE chunks for tool-call `idx` into `sse_events`.
/// No-op when retry is disabled (chunks were emitted directly).
fn flush_buffered_tool_chunks(state: &mut StreamState, idx: usize, sse_events: &mut SseVec) {
    if let Some(chunks) = state.buffered_tool_chunks.remove(&idx) {
        for chunk_json in chunks {
            sse_events.push(Ok(Event::default().data(chunk_json)));
        }
    }
}

/// Drop all buffered SSE chunks for tool-call `idx` without emitting.
/// Called when validation fails and we're going to fire a Tier 5c retry.
fn drop_buffered_tool_chunks(state: &mut StreamState, idx: usize) {
    state.buffered_tool_chunks.remove(&idx);
}

/// `DetectorOutput::ToolCall(tc, idx)`: complete tool call.
pub(super) fn handle_complete_tool_call(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc: &mut tool_parser::ToolCall,
    tc_idx: usize,
    sse_events: &mut SseVec,
) {
    // Content → Tool boundary: flush sanitiser tail.
    let pre_tool_tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if !pre_tool_tail.is_empty() {
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, pre_tool_tail)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
        ));
    }
    tool_parser::backfill_required_params(std::slice::from_mut(tc), &ctx.tool_defs_for_backfill);
    if ctx.wants_typed_arguments {
        tool_parser::coerce_all(std::slice::from_mut(tc), &ctx.tool_defs_for_backfill);
    }
    if let Some(ref cwd) = ctx.cwd_for_normalize {
        tool_parser::normalize_paths(std::slice::from_mut(tc), cwd);
    }
    let validation = tool_parser::validate_single_tool_call(tc, &ctx.tool_defs_for_backfill);
    let is_soft = validation
        .as_ref()
        .err()
        .map(|e| e.contains("non-empty"))
        .unwrap_or(false);
    if let Err(e) = &validation
        && !is_soft
    {
        tracing::warn!(
            tool = %tc.function.name,
            "tool call validation error (hard): {e}; replacing with content and ending"
        );
        let msg = format!("[atlas] Tool call rejected: {e}");
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, msg)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
        ));
        state.stop_string_triggered = true;
    } else if let Err(e) = &validation {
        // Soft validation error (empty required string) — emit the tool
        // call as the model produced it and let opencode's per-tool
        // schema surface its own actionable error. See
        // `handle_tool_call_delta` for the rationale.
        tracing::warn!(
            tool = %tc.function.name,
            "tool call validation error (soft): {e}; passing through to opencode"
        );
        let preview: String = tc.function.arguments.chars().take(120).collect();
        let s = if tc.function.arguments.len() > preview.len() {
            "…"
        } else {
            ""
        };
        tracing::info!("Tool call: {}({preview}{s})", tc.function.name);
        crate::metrics::TOOL_CALLS_TOTAL.inc();
        let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, tc, tc_idx);
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&start).unwrap_or_default())
        ));
        let frag = ChatCompletionChunk::tool_call_args_fragment(
            &ctx.model,
            &ctx.id,
            tc_idx,
            &tc.function.arguments,
        );
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
        ));
    } else {
        // Successful complete-call path — log + metric to match the
        // blocking and incremental-streaming paths.
        let preview: String = tc.function.arguments.chars().take(120).collect();
        let s = if tc.function.arguments.len() > preview.len() {
            "…"
        } else {
            ""
        };
        tracing::info!("Tool call: {}({preview}{s})", tc.function.name);
        crate::metrics::TOOL_CALLS_TOTAL.inc();
        let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, tc, tc_idx);
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&start).unwrap_or_default())
        ));
        let frag = ChatCompletionChunk::tool_call_args_fragment(
            &ctx.model,
            &ctx.id,
            tc_idx,
            &tc.function.arguments,
        );
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
        ));
    }
}

/// `DetectorOutput::ToolCallStart` — incremental: emit header now.
pub(super) fn handle_tool_call_start(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc_id: String,
    name: String,
    idx: usize,
    sse_events: &mut SseVec,
) {
    let pre_tool_tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if !pre_tool_tail.is_empty() {
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, pre_tool_tail)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
        ));
    }
    state
        .streaming_tool_args
        .insert(idx, (name.clone(), String::new()));
    let tc = tool_parser::ToolCall {
        id: tc_id,
        call_type: "function".to_string(),
        function: tool_parser::FunctionCall {
            name,
            arguments: String::new(),
        },
    };
    let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, &tc, idx);
    let start_json = serde_json::to_string(&start).unwrap_or_default();
    emit_or_buffer_tool_chunk(state, ctx, idx, start_json, sse_events);
}

/// `DetectorOutput::ToolCallDelta` — incremental: append args.
///
/// For qwen3_coder XML the streaming detector emits a single Delta with
/// the full parsed-and-canonicalised JSON arguments at the `</tool_call>`
/// boundary (see `streaming_impl.rs::process` line ~67 — args can't be
/// streamed character-by-character because XML parameter blocks must
/// finish before they convert to JSON). This is the natural spot to run
/// the same `backfill_required_params` + `validate_single_tool_call`
/// chain that the complete-tool-call path runs at `handle_complete_tool_call`,
/// so that streaming and non-streaming responses behave identically.
///
/// Without this, a model that emits `<function=NAME></function>` with no
/// `<parameter=>` blocks (observed under qwen3_coder + multi-turn agentic
/// loops with 21 tools, OpenClaw 2026.5.7) streams literal `"{}"` to the
/// client even when required parameters are declared in the schema —
/// while the non-streaming path would have backfilled `{"required_key": ""}`
/// and at least logged a warning. Issue #40 (iromu) called out this
/// "Opencode breaks tool calling more often" symptom.
pub(super) fn handle_tool_call_delta(
    state: &mut StreamState,
    ctx: &StreamCtx,
    args: String,
    idx: usize,
    sse_events: &mut SseVec,
) {
    let mut emit_args = args.clone();
    if let Some(entry) = state.streaming_tool_args.get_mut(&idx) {
        let name = entry.0.clone();
        let mut tc = tool_parser::ToolCall {
            id: format!("call_{:016x}", idx),
            call_type: "function".into(),
            function: tool_parser::FunctionCall {
                name: name.clone(),
                arguments: args.clone(),
            },
        };
        tool_parser::backfill_required_params(
            std::slice::from_mut(&mut tc),
            &ctx.tool_defs_for_backfill,
        );
        if ctx.wants_typed_arguments {
            tool_parser::coerce_all(std::slice::from_mut(&mut tc), &ctx.tool_defs_for_backfill);
        }
        if let Some(ref cwd) = ctx.cwd_for_normalize {
            tool_parser::normalize_paths(std::slice::from_mut(&mut tc), cwd);
        }
        if let Err(e) = tool_parser::validate_single_tool_call(&tc, &ctx.tool_defs_for_backfill) {
            // Mid-stream validation rejections used to emit a `[atlas] Tool
            // call rejected: …` content chunk and trip `stop_string_triggered`
            // — but `handle_tool_call_start` had already emitted the
            // `tool_calls[idx]` header to opencode, so suppressing the args
            // delta left opencode mid-call with no completion. opencode then
            // reported `SchemaError(Missing key)`, a less actionable error
            // than its own per-tool schema check (e.g. "The argument 'file'
            // cannot be empty. Received ''").
            //
            // Empty-required-string failures (most common: F78 path tools,
            // 2026-05-25 shell tools) are recoverable: emit the args delta
            // as the model produced them and let opencode's per-tool schema
            // surface its own actionable error to the model on the next
            // turn. Hard failures (unknown tool name, args not valid JSON)
            // still bail with a content chunk because they cannot be made
            // into a complete tool call at all.
            let is_soft = e.contains("non-empty");
            if is_soft {
                tracing::warn!(
                    tool = %name,
                    "tool call validation error (stream Δ, soft): {e}; passing through so opencode can surface its own per-tool schema error"
                );
                emit_args = tc.function.arguments.clone();
                entry.1.push_str(&emit_args);
            } else if ctx.tool_retry_enabled {
                // Tier 5c (2026-05-26): drop the buffered start + args
                // chunks for this idx, record the failure context, and
                // signal the scheduler to stop. `handle_done` will see
                // `pending_retry` and fire the retry inference; if the
                // retry produces a valid call we emit it in place of
                // the failed call, so the client never sees the bad one.
                tracing::warn!(
                    tool = %name,
                    "tool call validation error (stream Δ, hard, retry pending): {e}"
                );
                // Release the `entry` borrow on `state.streaming_tool_args`
                // before mutating the buffered-chunks + pending_retry on
                // `state` (the borrow checker rejects two simultaneous
                // mutable borrows of `state`). Capture what we still need.
                entry.1.push_str(&args);
                let errors_summary = e.to_string();
                drop_buffered_tool_chunks(state, idx);
                state.pending_retry = Some(PendingRetry {
                    errors_summary,
                    failed_idx: idx,
                });
                state.stop_string_triggered = true;
                state
                    .cancel_flag
                    .store(true, std::sync::atomic::Ordering::Release);
                return;
            } else {
                tracing::warn!(
                    tool = %name,
                    "tool call validation error (stream Δ, hard): {e}; replacing with content and ending"
                );
                let msg = format!("[atlas] Tool call rejected: {e}");
                let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, msg);
                sse_events.push(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                ));
                state.stop_string_triggered = true;
                entry.1.push_str(&args);
                return;
            }
        } else {
            emit_args = tc.function.arguments.clone();
            entry.1.push_str(&emit_args);
        }
    } else if !args.is_empty() {
        // No prior ToolCallStart for this idx — keep legacy passthrough.
    }
    if !emit_args.is_empty() {
        let frag =
            ChatCompletionChunk::tool_call_args_fragment(&ctx.model, &ctx.id, idx, &emit_args);
        let frag_json = serde_json::to_string(&frag).unwrap_or_default();
        // Either flush previously-buffered start + this args chunk
        // together (success path under retry), or emit directly (retry
        // disabled). When retry is disabled the start chunk was already
        // emitted in real time, so `emit_or_buffer_tool_chunk` just adds
        // the args chunk.
        emit_or_buffer_tool_chunk(state, ctx, idx, frag_json, sse_events);
        if ctx.tool_retry_enabled {
            flush_buffered_tool_chunks(state, idx, sse_events);
        }
    }
}

/// `DetectorOutput::ToolCallArgsFragment` — live-streaming: a ready-to-forward
/// slice of `function.arguments` the detector already coerced (XML) or sliced
/// (JSON). Append it verbatim to the accumulated args and emit it directly as an
/// OpenAI `tool_calls[idx].function.arguments` fragment — NO coercion or
/// validation (the detector did that per-field). If no prior `ToolCallStart`
/// created the accumulator entry for `idx`, the fragment is dropped (the header
/// must precede its arguments).
pub(super) fn handle_tool_call_args_fragment(
    state: &mut StreamState,
    ctx: &StreamCtx,
    fragment: String,
    idx: usize,
    sse_events: &mut SseVec,
) {
    let Some(entry) = state.streaming_tool_args.get_mut(&idx) else {
        return;
    };
    entry.1.push_str(&fragment);
    let frag = ChatCompletionChunk::tool_call_args_fragment(&ctx.model, &ctx.id, idx, &fragment);
    sse_events.push(Ok(
        Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
    ));
}

/// `DetectorOutput::ToolCallEnd` — close out the streaming accumulator
/// for `idx` and log the completed call. The dedup / name-run / total
/// caps that used to end the response here were removed 2026-06-12 for
/// vLLM parity (the server never vetoes a tool call).
pub(super) fn handle_tool_call_end(state: &mut StreamState, _ctx: &StreamCtx, idx: usize) {
    if let Some((name, args_json)) = state.streaming_tool_args.remove(&idx) {
        // Successful streaming tool call — log + metric to match the
        // blocking and complete-call paths.
        let preview: String = args_json.chars().take(120).collect();
        let s = if args_json.len() > preview.len() {
            "…"
        } else {
            ""
        };
        tracing::info!("Tool call: {name}({preview}{s})");
        crate::metrics::TOOL_CALLS_TOTAL.inc();
    }
}
