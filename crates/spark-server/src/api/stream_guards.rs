// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming-side sanitizer flush helper.
//!
//! The other guards that used to live here (`bump_f12_tool_call_count`
//! per-response tool-call cap, `check_loop_watchdog` repeating-line
//! detector) were removed 2026-06-12 for vLLM parity — the server no
//! longer force-ends a response it judges repetitive or tool-heavy.

use crate::tool_parser;

/// Flush anything held in the streaming sanitizer's tail buffer at
/// stream end. Drops content if tag-suppression is still active (no
/// close arrived) or if the remaining bytes look like an incomplete
/// tag opener.
pub fn flush_content_sanitizer(
    tag_scan_buf: &mut String,
    suppressing_param_leak: &mut bool,
    markers: &tool_parser::LeakMarkers,
) -> String {
    if *suppressing_param_leak {
        tag_scan_buf.clear();
        *suppressing_param_leak = false;
        return String::new();
    }
    if tag_scan_buf.is_empty() {
        return String::new();
    }
    let tag_max: usize = markers
        .orphan_open
        .iter()
        .chain(markers.close.iter())
        .map(|t| t.len())
        .max()
        .unwrap_or(0);
    let final_text = std::mem::take(tag_scan_buf);
    let looks_like_partial_tag = {
        let t = final_text.trim_end();
        tag_max > 0 && t.starts_with('<') && !t.contains(char::is_whitespace) && t.len() < tag_max
    };
    if looks_like_partial_tag {
        String::new()
    } else {
        final_text
    }
}
