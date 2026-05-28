// SPDX-License-Identifier: AGPL-3.0-only
//
// Read-only context shared by every `StreamEvent` arm. Owned by the
// `flat_map` closure so the per-event handlers can borrow it
// immutably alongside `&mut StreamState`.

use std::collections::HashSet;
use std::sync::Arc;

use crate::AppState;
use crate::tool_parser;

pub(super) struct StreamCtx {
    pub(super) state: Arc<AppState>,
    pub(super) model: String,
    pub(super) id: String,
    pub(super) prompt_len: usize,
    pub(super) enable_thinking: bool,
    pub(super) tool_defs_for_backfill: Vec<tool_parser::ToolDefinition>,
    pub(super) cwd_for_normalize: Option<String>,
    pub(super) stop_strings: Vec<String>,
    /// Tier 5c (2026-05-26): original-request context needed by
    /// `handle_done` to fire the post-stream retry. `tool_retry_enabled`
    /// is cached from `ATLAS_TOOL_RETRY` at request setup (avoid the
    /// env-var lookup per token); `prompt_tokens` is the rendered chat
    /// prompt (used as the base for the retry prompt); `grammar_spec`
    /// keeps the retry under the same xgrammar tag constraints as the
    /// original generation.
    pub(super) tool_retry_enabled: bool,
    /// Arc-wrapped so the per-event closure + spawned retry task can
    /// share the read-only token slice without duplicating ~40 KB of
    /// `Vec<u32>` on every refresh. `attempt_tool_retry` takes `&[u32]`
    /// so deref-coercion through Arc → Vec → slice is free at the call
    /// site.
    pub(super) prompt_tokens: Arc<Vec<u32>>,
    /// A2-AO (2026-05-26, /loop iter 1): cached prompt vocabulary
    /// (~identifiers from the rendered chat prompt) for always-on
    /// fuzzy repair in `handle_tool_call_delta`. Computed once per
    /// request from the decoded prompt tokens; cheap reads per call.
    pub(super) prompt_vocab: Arc<HashSet<String>>,
    pub(super) grammar_spec: Option<crate::api::inference_types::GrammarSpec>,
    pub(super) max_tokens: usize,
    pub(super) timeout_at: Option<std::time::Instant>,
    /// Number of trailing bytes to hold back from each streaming delta
    /// while stop-string matching is active. vLLM mirrors this in
    /// `IncrementalDetokenizer.update`: when stop strings are
    /// configured, the last `max(len(s) for s in stop_strings) - 1`
    /// bytes of the accumulated text are withheld so a stop string
    /// that lands across two decoded chunks (e.g. `<|im_st` + `art|>`)
    /// is never emitted as a partial leak. Zero when `stop_strings` is
    /// empty (existing behaviour preserved).
    pub(super) stop_string_buffer_len: usize,
    pub(super) leak_markers: tool_parser::LeakMarkers,
    /// PR 73 type coercion: whether the active tool parser wants
    /// schema-driven type coercion applied to parsed arguments
    /// (string → integer/boolean/array/object). True for qwen3_xml.
    pub(super) wants_typed_arguments: bool,
    pub(super) max_tool_calls_per_response: usize,
    pub(super) req_stream_include_usage: bool,
    pub(super) req_ctx: Option<crate::rate_limiter::RequestContext>,
    pub(super) dump_seq: Option<u64>,
}
