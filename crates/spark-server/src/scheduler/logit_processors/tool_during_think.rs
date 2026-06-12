// SPDX-License-Identifier: AGPL-3.0-only

//! Suppress `<tool_call>` during thinking.
//!
//! Tool calls inside `<think>` are unparsable per the canonical
//! qwen3_coder dialect, so the opener is hard-masked (`-inf`) while
//! `inside_thinking`. (The tool-loop `-12.0` bias branch was removed
//! 2026-06-12 with the request-level loop detector — vLLM parity.)

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct ToolCallDuringThinkingMask;

impl LogitsProcessor for ToolCallDuringThinkingMask {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if a.inside_thinking
            && let Some(tc_start) = ctx.tool_call_start_token
        {
            let idx = tc_start as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "tool_call_during_thinking_mask"
    }
}
