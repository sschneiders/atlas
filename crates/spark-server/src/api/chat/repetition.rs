//
// Resolve operator-default repetition detection for a single request.
// Precedence (highest wins):
//   1. Request body (`ChatCompletionRequest.repetition_detection`)
//   2. CLI `--repetition-detection` (already merged into
//      `state.behavior.repetition_detection` at serve time)
//   3. MODEL.toml `[behavior.repetition_detection]` (also in
//      `state.behavior`)
//   4. off (`None`)
//
// Mirrors `thinking::resolve_thinking` in shape. Lifted out of
// `chat::chat_completions_inner` (wave 4g).

use std::sync::Arc;

use crate::AppState;
use crate::openai::{ChatCompletionRequest, RepetitionDetectionParams};

pub(super) fn resolve_repetition_detection(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
) -> Option<RepetitionDetectionParams> {
    req.repetition_detection
        .or(state.behavior.repetition_detection)
}
