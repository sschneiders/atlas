// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::fuzzy_match::fuzzy_match_tool_name;
use super::*;

/// Fix tool call arguments: schema-aware type coercion + backfill missing params.
///
/// The qwen3_coder XML format emits all parameter values as raw text. This function:
/// 1. **Type coercion**: Converts string values to the schema-expected type
///    (number, boolean, integer, object, array). Prevents "expected number,
///    received string" errors from clients like OpenCode.
/// 2. **Backfill**: Adds empty strings for missing required string parameters.
///    Prevents cascading error loops from missing params.
///
/// Matches vLLM's qwen3coder_tool_parser behavior (schema-aware type conversion).
///
/// Resolves the effective type from a JSON schema property, handling `anyOf`/`oneOf`
/// wrappers (e.g., Pydantic v2's `Optional[int]` → `{"anyOf": [{"type":"integer"},{"type":"null"}]}`).
fn resolve_schema_type(schema: &serde_json::Value) -> Option<&str> {
    // Direct "type" field
    if let Some(t) = schema.get("type").and_then(|t| t.as_str()) {
        return Some(t);
    }
    // anyOf / oneOf: pick first non-null type
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema.get(key).and_then(|v| v.as_array()) {
            for variant in variants {
                if let Some(t) = variant.get("type").and_then(|t| t.as_str())
                    && t != "null"
                {
                    return Some(t);
                }
            }
        }
    }
    None
}

pub fn backfill_required_params(calls: &mut [ToolCall], tools: &[ToolDefinition]) {
    for call in calls.iter_mut() {
        let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
            continue;
        };
        let Some(ref params_schema) = tool_def.function.parameters else {
            continue;
        };
        let required = params_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        let properties = params_schema.get("properties").and_then(|p| p.as_object());
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;

        // 1. Coerce existing parameters to schema-expected types.
        if let Some(props) = properties {
            for (key, value) in args.iter_mut() {
                let expected_type = props.get(key).and_then(|p| resolve_schema_type(p));
                if let (Some(expected), serde_json::Value::String(s)) = (expected_type, &value) {
                    let coerced = match expected {
                        "number" => s.parse::<f64>().ok().map(|n| {
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(n)
                                    .unwrap_or(serde_json::Number::from(0)),
                            )
                        }),
                        "integer" => s
                            .parse::<i64>()
                            .ok()
                            .map(|n| serde_json::Value::Number(n.into())),
                        "boolean" => match s.to_lowercase().as_str() {
                            "true" | "1" | "yes" => Some(serde_json::Value::Bool(true)),
                            "false" | "0" | "no" => Some(serde_json::Value::Bool(false)),
                            _ => None,
                        },
                        "object" | "array" => serde_json::from_str(s).ok(),
                        _ => None, // "string" or unknown — keep as-is
                    };
                    if let Some(new_val) = coerced {
                        *value = new_val;
                        changed = true;
                    }
                }
            }
        }

        // 2. Normalize parameter names to match the schema.
        // The model sometimes emits camelCase (filePath) when the schema
        // defines snake_case (file_path), or vice versa. This is a known
        // Qwen3-Coder issue (vLLM #35347, llama.cpp #19382).
        if let Some(props) = properties {
            // Build case-insensitive lookup: "filepath" → "file_path" (schema name)
            let schema_normalized: std::collections::HashMap<String, &str> = props
                .keys()
                .map(|k| (k.to_lowercase().replace('_', ""), k.as_str()))
                .collect();

            let keys_to_fix: Vec<(String, String)> = args
                .keys()
                .filter(|k| !props.contains_key(*k))
                .filter_map(|k| {
                    let norm = k.to_lowercase().replace('_', "");
                    schema_normalized
                        .get(&norm)
                        .map(|schema_key| (k.clone(), schema_key.to_string()))
                })
                .collect();

            for (wrong_key, right_key) in keys_to_fix {
                if let Some(val) = args.remove(&wrong_key) {
                    args.entry(right_key).or_insert(val);
                    changed = true;
                }
            }
        }

        // 3. Backfill missing required string parameters.
        for key in &required {
            if !args.contains_key(*key) {
                let is_string = properties
                    .and_then(|p| p.get(*key))
                    .and_then(|v| v.get("type"))
                    .and_then(|t| t.as_str())
                    .is_none_or(|t| t == "string"); // default to string if no schema
                if is_string {
                    args.insert(key.to_string(), serde_json::Value::String(String::new()));
                    changed = true;
                }
            }
        }

        // 4. Auto-fill empty parameters with sensible defaults.
        // The model often generates empty required fields. Instead of rejecting
        // (which causes error loops), fill them with context-derived defaults.
        let func_name = call.function.name.clone();
        for key in &required {
            if let Some(serde_json::Value::String(val)) = args.get(*key) {
                if !val.trim().is_empty() {
                    continue;
                }
                let auto_val = match *key {
                    "description" => {
                        if let Some(serde_json::Value::String(cmd)) = args.get("command") {
                            if cmd.len() > 50 {
                                format!("Run: {}...", &cmd[..47])
                            } else {
                                format!("Run: {cmd}")
                            }
                        } else {
                            format!("{func_name} operation")
                        }
                    }
                    "filePath" | "file_path" => {
                        // Can't guess the path — leave empty so validation catches it
                        continue;
                    }
                    "oldString" | "old_string" => {
                        // Can't guess what to replace — leave empty
                        continue;
                    }
                    _ => continue,
                };
                args.insert(key.to_string(), serde_json::Value::String(auto_val));
                changed = true;
            }
        }

        if changed && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args)) {
            call.function.arguments = new_args;
        }
    }
}

/// Check if a tool call has empty required parameters that can't be auto-filled.
/// Returns the names of empty required params, or empty vec if all are filled.
pub fn find_empty_required_params(call: &ToolCall, tools: &[ToolDefinition]) -> Vec<String> {
    let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
        return Vec::new();
    };
    let Some(ref params_schema) = tool_def.function.parameters else {
        return Vec::new();
    };
    let required = params_schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let args: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&call.function.arguments).unwrap_or_default();
    let mut empty = Vec::new();
    for key in &required {
        match args.get(key.as_str()) {
            None => empty.push(key.clone()),
            Some(serde_json::Value::String(s)) if s.trim().is_empty() => empty.push(key.clone()),
            Some(serde_json::Value::Null) => empty.push(key.clone()),
            _ => {}
        }
    }
    empty
}

/// Normalize file paths in tool call arguments to be relative to the working directory.
///
/// OPENCODE BUG FIX (2026-04-22): the previous behaviour stripped the leading `/`
/// of any absolute path NOT under cwd, mangling user-intended paths like
/// `/tmp/calc-test16/calc.py` into `tmp/calc-test16/calc.py`. opencode then
/// resolved that relative path under `Instance.directory` (= `$HOME`), so the
/// file ended up at `$HOME/tmp/calc-test16/calc.py` instead of
/// `/tmp/calc-test16/`. The model spent 8+ turns trying to "fix" the directory
/// before the user noticed.
///
/// New behaviour:
/// - Paths under cwd → made relative (still helpful for Claude-Code-style clients)
/// - Paths starting with `/` but NOT under cwd → **PASS THROUGH UNCHANGED**.
///   The model knew what it wanted (e.g. user said "put it in /tmp/..."); we
///   should not second-guess. If it really is wrong, the filesystem op will
///   fail with a clear error and the model can self-correct.
/// - Already relative paths → unchanged
/// FP8 path-drift recovery: when a file-write's `filePath` is unusable —
/// empty, a bare directory, or a drifted/hallucinated absolute path OUTSIDE
/// cwd (e.g. `/tmp/harness-mtpr6/Cargo.toml` with a truncated dir, or
/// `/tmp/pure_axioms.txt`) — infer the intended in-project path from a
/// recognizable basename or the CONTENT shape and return it (relative to
/// cwd). Returns None when `cur` is already a sane in-project path. The
/// model's content is correct; only the path string drifted (low-margin FP8
/// token flips on arbitrary path strings) — this recovers intent, it never
/// invents content. Gated by ATLAS_WRITE_PATH_RECOVERY (PCND opt-in).
fn recover_drifted_write_path(cur: &str, content: &str, cwd: &str) -> Option<String> {
    let cwd_trim = cwd.trim_end_matches('/');
    let c = cur.trim().trim_start_matches('=').trim().trim_matches('"');
    let bn = c.rsplit('/').next().unwrap_or("");
    let in_cwd = !c.is_empty() && c.starts_with(&format!("{cwd_trim}/"));
    let is_dir = c.is_empty() || c.ends_with('/') || c == cwd_trim || c == cwd;
    let sane_file = !bn.is_empty() && bn.contains('.');
    // Already a usable in-project path (relative, or cwd-rooted) → leave it.
    if !is_dir && sane_file && (in_cwd || !c.starts_with('/')) {
        return None;
    }
    // 1) Drifted DIR but recognizable basename → canonical in-project location.
    if sane_file {
        let low = bn.to_ascii_lowercase();
        if low == "cargo.toml" {
            return Some("Cargo.toml".into());
        }
        if low == "main.rs" {
            return Some("src/main.rs".into());
        }
        if low == "lib.rs" {
            return Some("src/lib.rs".into());
        }
        if low.ends_with(".rs") {
            return Some(format!("src/{bn}"));
        }
        if low.ends_with(".toml") || low.ends_with(".lock") {
            return Some(bn.to_string());
        }
    }
    // 2) Empty / bare-dir / unrecognizable path → infer from content shape.
    classify_path_from_content(content).map(str::to_string)
}

/// Map a file's CONTENT shape to its canonical in-project path:
/// `[package]`/`[dependencies]` → `Cargo.toml`; `fn main(` →
/// `src/main.rs`. Returns `None` when the content carries no
/// recognizable project-file signature.
///
/// Single source of truth for content-based path inference, shared by
/// `recover_drifted_write_path` (drifted-write-path recovery in
/// `normalize_paths`) and the fenced-code tool salvage (narrate-then-
/// tool recovery in `crate::tool_salvage`). It never invents content —
/// it only classifies content the model already produced.
pub(crate) fn classify_path_from_content(content: &str) -> Option<&'static str> {
    if content.contains("[package]") || content.contains("[dependencies]") {
        return Some("Cargo.toml");
    }
    if content.contains("fn main(") {
        return Some("src/main.rs");
    }
    None
}

/// A line that is *only* a markdown code fence: ```` ``` ```` optionally
/// followed by a language tag (```` ```rust ````). Code lines that merely
/// *contain* a fence in a string literal (`let s = "```";`) or a doc
/// comment (`/// ``` `) do NOT match — the trimmed line must start with the
/// fence and carry nothing but an identifier after it.
fn is_code_fence_line(line: &str) -> bool {
    let t = line.trim();
    t.starts_with("```")
        && t[3..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '+' || c == '-')
}

/// Whether a fence line carries a language tag (```` ```rust ````) vs being
/// a bare fence (```` ``` ````). Opening fences usually carry a tag; closing
/// fences are bare — used to disambiguate a lone fence.
fn fence_has_lang(line: &str) -> bool {
    !line.trim()[3..].is_empty()
}

/// Strip markdown-fence contamination from a file-write `content`.
///
/// FP8 failure mode (2026-05-29, harness run 1): the model emits a
/// `write()` whose `content` is its entire markdown answer — the file
/// body wrapped in a ```` ```lang ```` fence and/or trailed by prose
/// ("This creates…", a bare ```` ``` ````, "**Principles Applied:**").
/// The stray fence + prose makes the written `.rs`/`.toml` fail to
/// compile (`unknown start of token: backtick`). This recovers just the
/// file body the model produced — it never invents content.
///
/// Structures handled:
///  - **Trailing contamination** (no leading fence, a bare closing fence
///    with code before it): truncate at the closing fence.
///  - **Fully wrapped** / **intro + fenced block** (an opening ```lang
///    fence): keep the body between the opening fence and the next fence.
///
/// Applies to code files only — markdown/text targets (where ```` ``` ````
/// is legitimate) are skipped. Returns the cleaned body when contamination
/// was found and removed, else `None`.
fn strip_markdown_fence_contamination(content: &str, file_path: &str) -> Option<String> {
    let lower = file_path.to_ascii_lowercase();
    const SKIP_EXT: &[&str] = &[".md", ".markdown", ".mdx", ".rst", ".txt"];
    if SKIP_EXT.iter().any(|e| lower.ends_with(e)) {
        return None;
    }

    let lines: Vec<&str> = content.lines().collect();
    let fences: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| is_code_fence_line(l))
        .map(|(i, _)| i)
        .collect();
    if fences.is_empty() {
        return None; // clean content — no fences to strip
    }

    let f0 = fences[0];
    let first_nonempty = lines.iter().position(|l| !l.trim().is_empty());
    // f0 is an OPENING fence when it carries a language tag OR is the very
    // first content line; otherwise it is a bare CLOSING fence sitting
    // after the real code (the trailing-contamination case).
    let f0_is_opening = fence_has_lang(lines[f0]) || first_nonempty == Some(f0);

    let body: Vec<&str> = if f0_is_opening {
        let body_start = f0 + 1;
        let close = fences.iter().copied().find(|&i| i >= body_start);
        match close {
            Some(ci) => lines[body_start..ci].to_vec(),
            None => lines[body_start..].to_vec(),
        }
    } else {
        // Bare closing fence with code before it → keep everything before.
        lines[..f0].to_vec()
    };

    let cleaned = body.join("\n");
    let cleaned = cleaned.trim_matches('\n');
    if cleaned.is_empty() {
        return None; // never emit an empty file
    }
    let cleaned = format!("{cleaned}\n");
    if cleaned.trim_end() == content.trim_end() {
        None // nothing actually removed
    } else {
        Some(cleaned)
    }
}

pub fn normalize_paths(calls: &mut [ToolCall], cwd: &str) {
    // Common parameter names that contain file paths
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path", "file"];
    let cwd_slash = if cwd.ends_with('/') {
        cwd.to_string()
    } else {
        format!("{cwd}/")
    };

    for call in calls.iter_mut() {
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                // Long-context FP8 drift mode (2026-05-28): the model
                // sometimes emits the value with XML-attribute-style
                // framing — `="/tmp/x/main.rs"` instead of `/tmp/x/main.rs`.
                // The qwen3_coder grammar accepts the literal `=` and quotes
                // as part of the parameter body. Strip them here so the
                // downstream path-shape check and write dispatch see a
                // clean path. vLLM's tool_parser does similar leniency.
                let trimmed = path.trim();
                let mut sanitized: &str = trimmed;
                if let Some(rest) = sanitized.strip_prefix('=') {
                    sanitized = rest.trim_start();
                }
                // FP8 drift (2026-05-29, fencecontent run 1): the model
                // sometimes leaks a JSON-fragment-shaped value like
                // `"/tmp/x/Cargo.toml",` — the path wrapped in quotes with a
                // trailing comma. Drop trailing commas/whitespace first so the
                // surrounding-quote strip below sees a clean `"…"`; otherwise
                // the file is created with the quotes+comma literally in its
                // name and the project never builds.
                sanitized = sanitized.trim_end_matches([',', ' ', '\t']);
                if sanitized.len() >= 2
                    && sanitized.starts_with('"')
                    && sanitized.ends_with('"')
                {
                    sanitized = &sanitized[1..sanitized.len() - 1];
                }
                if sanitized != path.as_str() {
                    args.insert(
                        key.to_string(),
                        serde_json::Value::String(sanitized.to_string()),
                    );
                    changed = true;
                }
                // Re-read after possible sanitization
                let Some(serde_json::Value::String(path)) = args.get(*key) else {
                    continue;
                };
                if !path.starts_with('/') {
                    continue; // Already relative — leave it
                }
                if !path.starts_with(&cwd_slash) {
                    // Absolute path NOT under cwd — pass through verbatim. The
                    // user explicitly asked for this location (e.g. "/tmp/..."),
                    // and trimming `/` here breaks downstream clients that
                    // resolve relative paths against THEIR own working dir.
                    continue;
                }
                let new_path = path[cwd_slash.len()..].to_string();
                if new_path != *path && !new_path.is_empty() {
                    args.insert(key.to_string(), serde_json::Value::String(new_path));
                    changed = true;
                }
            }
        }
        // FP8 path-drift recovery for file writes (env-gated, PCND opt-in).
        if std::env::var("ATLAS_WRITE_PATH_RECOVERY").as_deref() == Ok("1")
            && matches!(
                call.function.name.as_str(),
                "write" | "Write" | "edit" | "Edit"
            )
        {
            let content = args
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let cur = ["filePath", "file_path", "path", "file"]
                .iter()
                .find_map(|k| args.get(*k).and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string();
            if let Some(rp) = recover_drifted_write_path(&cur, &content, cwd) {
                for k in ["file_path", "path", "file"] {
                    args.remove(k);
                }
                tracing::warn!(
                    "ATLAS_WRITE_PATH_RECOVERY: drifted write path {:?} → {:?} (content-inferred)",
                    cur,
                    rp
                );
                args.insert("filePath".to_string(), serde_json::Value::String(rp));
                changed = true;
            }
            // Markdown-fence contamination recovery: the model sometimes
            // dumps its full markdown answer (```lang-wrapped body + stray
            // closing fence + trailing prose) into the write content,
            // producing an uncompilable .rs/.toml. Strip it for code files.
            // Re-read the (possibly path-recovered) target so the extension
            // gate sees the canonical path.
            let fp = ["filePath", "file_path", "path", "file"]
                .iter()
                .find_map(|k| args.get(*k).and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string();
            if let Some(clean) = strip_markdown_fence_contamination(&content, &fp) {
                tracing::warn!(
                    "ATLAS_WRITE_PATH_RECOVERY: stripped markdown-fence contamination from {:?} content ({} → {} bytes)",
                    fp,
                    content.len(),
                    clean.len()
                );
                args.insert("content".to_string(), serde_json::Value::String(clean));
                changed = true;
            }
        }
        if changed && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args)) {
            call.function.arguments = new_args;
        }
    }
}

// ── Tool call validation ──

/// Result of validating a batch of tool calls against their schemas.
pub struct ValidatedToolCalls {
    /// Tool calls that passed all validations.
    pub valid: Vec<ToolCall>,
    /// Human-readable error messages for invalid calls.
    /// These should be injected into the response content so the model
    /// sees clear, actionable feedback instead of cryptic client errors.
    pub errors: Vec<String>,
}

/// Validate tool calls against their schemas. Returns valid calls and
/// error messages for invalid ones.
///
/// Checks:
/// 1. Tool name exists in definitions
/// 2. Arguments are valid JSON
/// 3. Required string params are non-empty
/// 4. file_path params don't look like directories (end with `/`)
pub fn validate_tool_calls(
    mut calls: Vec<ToolCall>,
    tools: &[ToolDefinition],
) -> ValidatedToolCalls {
    let mut valid = Vec::new();
    let mut errors = Vec::new();

    for call in &mut calls {
        // Fuzzy name repair: if model hallucinates a close-but-wrong name,
        // map to the closest available tool (NVFP4 models often drop prefixes
        // like "get_" or use abbreviations like "weather" for "get_weather").
        if tools.iter().all(|t| t.function.name != call.function.name)
            && let Some(best) = fuzzy_match_tool_name(&call.function.name, tools)
        {
            tracing::info!(
                "Fuzzy tool name repair: '{}' -> '{}'",
                call.function.name,
                best
            );
            call.function.name = best;
        }
        match validate_single_tool_call(call, tools) {
            Ok(()) => valid.push(call.clone()),
            Err(msg) => errors.push(msg),
        }
    }

    ValidatedToolCalls { valid, errors }
}

/// Validate a single tool call. Returns `Ok(())` if valid,
/// `Err(error_message)` with a clear, actionable error if invalid.
pub fn validate_single_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), String> {
    let name = &call.function.name;

    // 1. Check tool name exists
    let tool_def = tools.iter().find(|t| t.function.name == *name);
    if tool_def.is_none() {
        let available: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        return Err(format!(
            "Error: Unknown tool '{}'. Available tools: {}",
            name,
            available.join(", ")
        ));
    }
    let tool_def = tool_def.unwrap();

    // 2. Check arguments are valid JSON
    let args: serde_json::Map<String, serde_json::Value> =
        match serde_json::from_str(&call.function.arguments) {
            Ok(a) => a,
            Err(_) => {
                return Err(format!(
                    "Error: {} arguments must be valid JSON. Got: {}",
                    name,
                    &call.function.arguments[..call.function.arguments.len().min(100)]
                ));
            }
        };

    // 3. Check required params are present. Do NOT enforce non-empty strings —
    // that is the client's schema concern. Empty-string rejection here broke
    // Theia IDE's getWorkspaceFileList, which legitimately passes path="".
    if let Some(ref params_schema) = tool_def.function.parameters {
        let required: Vec<&str> = params_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        for key in &required {
            if args.get(*key).is_none() {
                return Err(format!(
                    "Error: {} requires parameter '{}' but it was not provided.",
                    name, key
                ));
            }
        }
    }

    // 4. Path-specific validation for file tools
    const FILE_TOOLS: &[&str] = &["Write", "write", "Edit", "edit", "Read", "read"];
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path"];
    // F78 (2026-04-30): file MUTATION tools must have a non-empty
    // path. Live opencode session
    // `ses_2215a95d6ffe6gAzHMBrcXqGXX` looped 11 turns because the
    // model emitted `{"content":"...","filePath":""}` (the model
    // self-truncated the content string and grammar-completed
    // filePath with the empty default). opencode's Write tool
    // returned EISDIR; the model retried with the same empty path.
    // Rejecting here turns the malformed tool_call into a no-op so
    // the response falls through to text — the model gets a single
    // chance to recover instead of opencode echoing EISDIR forever.
    // Read/Glob/LS keep the lenient behavior (Theia's
    // getWorkspaceFileList legitimately passes path="").
    const WRITE_FAMILY: &[&str] = &[
        "Write",
        "write",
        "Edit",
        "edit",
        "MultiEdit",
        "multiEdit",
        "multi_edit",
    ];
    if WRITE_FAMILY.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                let trimmed = path.trim();
                if trimmed.is_empty() {
                    // #211 option-B diagnostic (env-gated): pinpoint the
                    // empty_path drift — generation vs parse. Logs the full
                    // post-parse arg shape (keys + per-value lengths). An
                    // empty filePath alongside a large `content` is the
                    // self-truncation generation pattern (F78); filePath
                    // absent ⇒ omission; a path under an unexpected key ⇒
                    // parser. Inert unless ATLAS_TOOLCALL_DEBUG=1.
                    if std::env::var("ATLAS_TOOLCALL_DEBUG").as_deref() == Ok("1") {
                        let shape: Vec<String> = args
                            .iter()
                            .map(|(k, v)| match v {
                                serde_json::Value::String(s) => {
                                    format!("{k}=str(len={})", s.len())
                                }
                                other => format!("{k}={}", other),
                            })
                            .collect();
                        tracing::warn!(
                            tool = %name, empty_key = %key,
                            "ATLAS_TOOLCALL_DEBUG empty-path arg shape: [{}]",
                            shape.join(", ")
                        );
                    }
                    return Err(format!(
                        "Error: {name} requires a non-empty '{key}'. \
                             Got empty string — provide an absolute path \
                             like '/tmp/calc-test75/Cargo.toml'."
                    ));
                }
                // Long-context FP8 drift mode: model occasionally emits
                // the value with XML-attribute-style framing — e.g.
                // `<parameter=filePath>="/tmp/x/main.rs"</parameter>`
                // — leaking the `="..."` shape into the value. Strip a
                // leading `=` and a single pair of surrounding ASCII
                // double-quotes before the path-shape check so these
                // drifted-but-recoverable calls still resolve. vLLM's
                // tool_parser does similar leniency.
                // opencode resolves write paths against the agent cwd
                // (`--dir`), so bare RELATIVE filenames like `Cargo.toml`
                // or `src/main.rs` are legitimate — vLLM accepts them and
                // the model emits them constantly. The previous rule
                // required a `/`, `./`, or `../` prefix and rejected
                // `Cargo.toml`, which made opencode loop on rejections and
                // abandon the task. Accept any non-empty path EXCEPT ones
                // carrying shell metacharacters / whitespace, which signal
                // a leaked command (e.g. `created && ls -R`) rather than a
                // real path — those we still reject (also closes CWE-78
                // command-leak-as-path).
                const SHELL_META: &[char] =
                    &[' ', '\t', '\n', '\r', '&', '|', ';', '`', '$', '<', '>', '(', ')', '*', '?'];
                let looks_like_command = trimmed.contains(SHELL_META);
                if looks_like_command || trimmed.len() < 3 {
                    return Err(format!(
                        "Error: {name} '{key}' must be a filesystem path (absolute or relative \
                         to the working directory), at least 3 chars, with no shell \
                         metacharacters or whitespace. Got {path:?}."
                    ));
                }
            }
        }
    }
    // Shell-execution tools must have a non-empty command. Mirrors F78
    // for the Write family. Without this, the `any_text` qwen3_coder
    // body grammar (2026-05-25) accepts an immediately-closed parameter
    // `<parameter=command></parameter>`; opencode's bash handler then
    // returns "The argument 'file' cannot be empty. Received ''" and
    // the model burns to max_tokens retrying the same empty call.
    // Previously the `json_schema` body grammar combined with
    // `enforce_min_length_on_required_strings` (`grammar/schema.rs`)
    // enforced min_length 1 at the FSM level; lifting that check to
    // the validator post-parse keeps the same invariant while letting
    // the grammar body be `any_text` (native XML wire format).
    const SHELL_FAMILY: &[&str] = &[
        "bash", "Bash", "shell", "Shell", "exec", "Exec", "run", "Run",
        "execute", "Execute", "terminal", "Terminal",
    ];
    const CMD_KEYS: &[&str] = &["command", "cmd", "script", "code"];
    if SHELL_FAMILY.contains(&name.as_str()) {
        for key in CMD_KEYS {
            if let Some(serde_json::Value::String(cmd)) = args.get(*key)
                && (cmd.trim().is_empty() || cmd.trim().len() < 2)
            {
                return Err(format!(
                    "Error: {name} requires a non-empty '{key}'. \
                         Got empty string — provide the shell command \
                         to execute, e.g. 'ls /tmp'."
                ));
            }
        }
    }
    if FILE_TOOLS.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                if path.ends_with('/') {
                    return Err(format!(
                        "Error: {} file_path must be a FILE, not a directory. Got '{}'. Use e.g. '{}/index.ts'",
                        name,
                        path,
                        path.trim_end_matches('/')
                    ));
                }
                // Check if it looks like just a directory name (no extension, no dots, no uppercase)
                // Allow extensionless files like LICENSE, Makefile, Dockerfile, Cargo.lock etc.
                if !path.is_empty()
                    && !path.contains('.')
                    && !path.contains('/')
                    && path
                        .chars()
                        .all(|c| c.is_lowercase() || c == '-' || c == '_')
                {
                    return Err(format!(
                        "Error: {} file_path '{}' looks like a directory. Add a filename, e.g. '{}/index.ts'",
                        name, path, path
                    ));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod fence_contamination_tests {
    use super::strip_markdown_fence_contamination as strip;

    #[test]
    fn trailing_fence_and_prose_truncated() {
        // Harness run-1 shape: clean code, then a bare closing fence, then
        // explanatory prose (with backticks) and a Principles footer.
        let content = "use axum::Router;\n\nfn main() {}\n```\n\nThis creates a server that:\n- reads `PORT`\n\n**Principles Applied:**\n- SBIO\n";
        let out = strip(content, "src/main.rs").expect("must strip");
        assert_eq!(out, "use axum::Router;\n\nfn main() {}\n");
        assert!(!out.contains("```"));
        assert!(!out.contains("Principles"));
    }

    #[test]
    fn fully_wrapped_fence_unwrapped() {
        let content = "```rust\nfn main() { println!(\"hi\"); }\n```\n";
        let out = strip(content, "src/main.rs").expect("must unwrap");
        assert_eq!(out, "fn main() { println!(\"hi\"); }\n");
    }

    #[test]
    fn intro_then_fenced_block_keeps_only_code() {
        let content = "Here is the file:\n```rust\nfn main() {}\n```\nLet me know!\n";
        let out = strip(content, "src/main.rs").expect("must extract fenced block");
        assert_eq!(out, "fn main() {}\n");
    }

    #[test]
    fn clean_code_untouched() {
        let content = "fn main() {\n    let x = 1;\n}\n";
        assert!(strip(content, "src/main.rs").is_none());
    }

    #[test]
    fn code_with_fence_in_string_literal_untouched() {
        // A line that merely contains ``` inside a string is not a fence line.
        let content = "fn main() {\n    let s = \"```\";\n    println!(\"{}\", s);\n}\n";
        assert!(strip(content, "src/main.rs").is_none());
    }

    #[test]
    fn markdown_target_skipped() {
        // README.md legitimately contains fenced code — never strip.
        let content = "# Title\n\n```rust\nfn main() {}\n```\n\nmore docs\n";
        assert!(strip(content, "README.md").is_none());
    }

    #[test]
    fn toml_trailing_fence_truncated() {
        let content = "[package]\nname = \"x\"\nversion = \"0.1.0\"\n```\n\nThat is the manifest.\n";
        let out = strip(content, "Cargo.toml").expect("must strip");
        assert_eq!(out, "[package]\nname = \"x\"\nversion = \"0.1.0\"\n");
    }

    #[test]
    fn malformed_quoted_comma_filepath_sanitized() {
        // FP8 drift: filePath value leaked as a JSON fragment `"…/Cargo.toml",`
        // (surrounding quotes + trailing comma). The unconditional path
        // sanitizer must clean it to a cwd-relative `Cargo.toml` so the file
        // lands with a usable name. Env-independent (the quote/comma strip is
        // not gated by ATLAS_WRITE_PATH_RECOVERY).
        use crate::tool_parser::{FunctionCall, ToolCall};
        let mut calls = vec![ToolCall {
            id: "x".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "write".into(),
                arguments: serde_json::json!({
                    "filePath": "\"/tmp/proj/Cargo.toml\",",
                    "content": "[package]\nname = \"x\"\n"
                })
                .to_string(),
            },
        }];
        super::normalize_paths(&mut calls, "/tmp/proj");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["filePath"], "Cargo.toml");
    }
}
