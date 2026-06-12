// SPDX-License-Identifier: AGPL-3.0-only
//
// Resolve `(enable_thinking, thinking_budget)` for a single
// request. Precedence (highest wins):
//   1. `--disable-thinking` CLI flag (forces OFF for every request)
//   2. Request body (`reasoning_effort`, `thinking.budget_tokens`, …)
//   3. MODEL.toml `[behavior].thinking_default`
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;

pub(super) fn resolve_thinking(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
    tools_active: bool,
) -> (bool, Option<u32>) {
    if state.disable_thinking {
        return (false, None);
    }
    let (et, tb) = req.resolve_thinking(state.behavior.thinking_default);
    // `thinking_in_tools=false` is the MODEL.toml DEFAULT for tool-
    // active turns: it suppresses thinking when the client is silent.
    let et = if tools_active
        && !state.behavior.thinking_in_tools
        && !req.thinking_explicitly_requested()
    {
        false
    } else {
        et
    };
    // vLLM parity (2026-06-12): no server-imposed thinking budget. A
    // budget is enforced only when the client requests one
    // (`thinking_token_budget` / `thinking.budget_tokens`) or the
    // operator sets one (CLI `--max-thinking-budget` / MODEL.toml
    // `[behavior].max_thinking_budget`, 0 = unset). The previous 90%-of-
    // max_tokens implicit cap is gone — thinking tokens now consume the
    // generation budget directly, so `max_tokens` is the natural bound.
    let budget = if et {
        tb.or(match state.behavior.max_thinking_budget {
            0 => None,
            b => Some(b),
        })
    } else {
        None
    };
    (et, budget)
}
