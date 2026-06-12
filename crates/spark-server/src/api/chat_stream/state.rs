// SPDX-License-Identifier: AGPL-3.0-only
//
// Mutable per-stream state captured by the `flat_map` closure in
// `chat_stream.rs`. Lifted out of that closure so each `StreamEvent`
// arm can be extracted to a free function (`handle_token`,
// `handle_done`, `handle_error`) that takes `&mut StreamState` plus
// any additional non-state arguments.
//
// Read-only context (`Arc<AppState>`, model name, tool defs, ...) is
// passed via `StreamCtx` (see `ctx.rs`) so the helpers don't need to
// duplicate two dozen function-parameter slots.

use std::collections::HashMap;

use crate::tool_parser;

pub(super) struct StreamState {
    /// Token IDs accumulated since the last reset (cleared at the
    /// `</think>` boundary so post-thinking content decodes cleanly).
    pub(super) all_toks: Vec<u32>,
    /// Byte offset into the thinking-phase decoded text already
    /// emitted as `reasoning_chunk` deltas.
    pub(super) emitted: usize,
    /// Lazy streaming-decoder over the content phase (post-thinking).
    pub(super) content_decoder: Option<crate::tokenizer::StreamingDecoder<'static>>,
    /// Buffer used for stop-string matching across delta boundaries.
    pub(super) accumulated_content: String,
    /// Number of bytes of `accumulated_content` already forwarded to
    /// the client. The vLLM-style hold-back (see `handle_token`) keeps
    /// the last `max(stop_string_len) - 1` bytes back until either a
    /// match completes or the stream finalises, so the emitted prefix
    /// can lag behind the accumulator. Used to compute the next delta
    /// slice without re-emitting bytes.
    pub(super) stop_string_emitted_len: usize,
    /// Mirror of the post-sanitizer content stream; used by the
    /// post-stream refusal classifier and the `--dump` synthesiser.
    pub(super) refusal_scan_buf: String,
    /// Flips true on first stop-string match or on watchdog/dedup
    /// trip; suppresses further content emissions.
    pub(super) stop_string_triggered: bool,
    /// Sanitiser state: suppressing content while waiting for a
    /// matching `</parameter>` close after an orphan `<parameter=`.
    pub(super) suppressing_param_leak: bool,
    /// Sanitiser state: currently inside a tool-call envelope opener
    /// (e.g. `<minimax:tool_call>`); inner `<invoke ...>` etc. are
    /// legitimate content while this is true.
    pub(super) inside_envelope: bool,
    /// Mirror of `inside_envelope` for the reasoning sanitiser.
    pub(super) reasoning_inside_envelope: bool,
    /// Tag-scan buffer for the content sanitiser.
    pub(super) tag_scan_buf: String,
    /// Sanitiser state for reasoning content (parallel to
    /// `suppressing_param_leak` above).
    pub(super) reasoning_suppressing_leak: bool,
    /// Tag-scan buffer for the reasoning sanitiser.
    pub(super) reasoning_tag_scan_buf: String,
    /// Set true when a fenced/XML tool intent is salvaged into a
    /// synthetic `tool_call` so the Done arm picks the right
    /// `finish_reason`.
    pub(super) salvaged_tool_call: bool,
    /// Per-streaming-toolcall accumulator keyed by `oa_idx`.
    /// Holds (name, args_so_far) until `ToolCallEnd` logs the call.
    pub(super) streaming_tool_args: HashMap<usize, (String, String)>,
    /// Cooperative cancellation flag shared with the scheduler. Flipped
    /// true on a forced-stop condition (stop-string match, tool-call
    /// validation hard-fail); the scheduler reads it in
    /// `emit_step::emit_token` and finalises the sequence. Without
    /// this, `stop_string_triggered` only suppresses output and the
    /// scheduler keeps generating until natural EOS / max_tokens.
    pub(super) cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Streaming tool-call detector (`Some` iff `tools_active`).
    pub(super) detector: Option<tool_parser::StreamingToolDetector>,
    /// True iff the reasoning/`<think>` phase has finished. Starts
    /// `true` when the request did not enable thinking.
    pub(super) thinking_done: bool,
    /// Dead after the tool-call retry stack was removed (`tool_retry_enabled`
    /// is now constant `false`, so chunks are always streamed in real time
    /// and this map stays empty). Retained so the buffering helpers in
    /// `tool_handlers.rs` still type-check.
    pub(super) buffered_tool_chunks: std::collections::HashMap<usize, Vec<String>>,
    /// Dead after the tool-call retry stack was removed; never set now that
    /// `tool_retry_enabled` is constant `false`.
    pub(super) pending_retry: Option<PendingRetry>,
    /// `return_token_ids`: sampled token IDs not yet attached to an
    /// emitted chunk. One ID is pushed per `handle_token` call (== one
    /// sampled token == one increment of `usage.completion_tokens`),
    /// then drained onto the next client-visible chunk. The sum of all
    /// drained IDs across the stream therefore equals
    /// `completion_tokens` exactly. Stays empty unless the request
    /// opted in, so it costs nothing on the default path.
    pub(super) pending_token_ids: Vec<u32>,
}

/// Carrier for the (now-removed) tool-call retry path. Never constructed
/// anymore, but retained so `pending_retry`'s type still resolves.
pub(super) struct PendingRetry {
    pub(super) errors_summary: String,
    pub(super) failed_idx: usize,
}

impl StreamState {
    pub(super) fn new(
        tools_active: bool,
        enable_thinking: bool,
        cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
        tool_defs: Vec<tool_parser::ToolDefinition>,
    ) -> Self {
        Self {
            all_toks: Vec::new(),
            emitted: 0,
            content_decoder: None,
            accumulated_content: String::new(),
            stop_string_emitted_len: 0,
            refusal_scan_buf: String::new(),
            stop_string_triggered: false,
            suppressing_param_leak: false,
            inside_envelope: false,
            reasoning_inside_envelope: false,
            tag_scan_buf: String::new(),
            reasoning_suppressing_leak: false,
            reasoning_tag_scan_buf: String::new(),
            salvaged_tool_call: false,
            streaming_tool_args: HashMap::new(),
            cancel_flag,
            detector: if tools_active {
                Some(tool_parser::StreamingToolDetector::new_with_tools(
                    tool_defs,
                ))
            } else {
                None
            },
            thinking_done: !enable_thinking,
            buffered_tool_chunks: HashMap::new(),
            pending_retry: None,
            pending_token_ids: Vec::new(),
        }
    }
}
