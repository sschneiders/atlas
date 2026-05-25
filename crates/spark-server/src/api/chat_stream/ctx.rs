// SPDX-License-Identifier: AGPL-3.0-only
//
// Read-only context shared by every `StreamEvent` arm. Owned by the
// `flat_map` closure so the per-event handlers can borrow it
// immutably alongside `&mut StreamState`.

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
