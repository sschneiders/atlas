# KVFlash port — design and implementation plan

**Status:** in progress (branch chain `feat/kvflash-*`).
**Source:** [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub) `optimizations/kvflash/` (Apache-2.0).
**License note:** Atlas is AGPL-3.0-only; lucebox-hub is Apache-2.0. This is a **reimplementation
of a mechanism** (Atlas is Rust + CUDA PTX, lucebox is C++/Python/CUDA), not a code import.
Apache-2.0 → AGPL-3.0 inclusion is one-way compatible; `deny.toml` already permits Apache-2.0
dependencies. Attribution to the lucebox-hub authors and the FlashMemory paper (arXiv 2606.09079)
lives in this file and the implementation PRs.

## What KVFlash is

FlashMemory-style (arXiv 2606.09079) decode-time KV paging. The attention KV cache is allocated
at a fixed **pool** size (e.g. 1024–4096 tokens) instead of `max_ctx`; cold chunks page to **host
RAM** bit-exact and recallable; a slot-validity mask is uploaded before every compute; a **reselect**
loop repages the pool every τ decoded tokens using a relevance scorer (a small Qwen3-0.6B drafter,
LRU fallback). Net effect: the GPU footprint of the attention KV cache is a hard `O(pool)`
constant regardless of logical context length, and decode speed stops depending on context length.

Measured upstream (Qwen3.6-27B Q4_K_M, RTX 3090, Q8_0 KV): 256K context at 38.6 tok/s with 72 MiB
resident KV (vs 13.1 tok/s / 4608 MiB full cache); needle recall 88–100% at 6% residency.

## Locked decisions (this port)

1. **Scope:** full KVFlash — paging + LRU + drafter scorer + cross-tokenizer scorer + spec-decode
   integration.
2. **Architecture:** extend the existing `HighSpeedSwap` (HSS) orchestrator rather than build a
   parallel system (SSOT). Concretely: generalize HSS's `backend` field from `IoUringBackend` to
   `B: StorageBackend` and add a `HostRamBackend`.
3. **Validity mask:** surgical — overload `block_table` entries with the existing `dummy_kv_block`
   sentinel so the resident-set is conveyed by the block table itself. No new kernel arg in the
   MVP; an explicit per-block mask kernel variant is a documented follow-up.
4. **Hardware/model targets:** `gb10` hardware; `qwen3.6-27b` and `qwen3.6-35b-a3b` model targets.
   Other archs/hw deferred.
5. **License:** reimplementation of the concept; no lucebox code copied.
6. **Granularity:** keep Atlas's 16-token KV blocks (4 Atlas blocks = 1 KVFlash 64-token chunk).
   No new `--block-size` knob.

## Component map (KVFlash → Atlas, verified)

| KVFlash concept | Atlas integration point | Location | Fit |
|---|---|---|---|
| Fixed GPU slot pool | `ScratchPool` (HSS) + `PagedKvCache` free-list | `spark-storage/src/scratch_pool.rs:53`; `spark-runtime/src/kv_cache.rs:432` | direct |
| Cold chunks in host RAM | **new `HostRamBackend: StorageBackend`** (HSS backend today is `IoUringBackend` to disk) | `spark-storage/src/backend/mod.rs:28`; `high_speed_swap.rs:29,84` | new impl, existing trait |
| Bit-exact block move | `read_block` / `write_block` D2H/H2D | `spark-runtime/src/kv_cache/paged_impl.rs:449-482` | direct; valid for all 16 `KvCacheDtype` |
| Slot-validity (surgical) | `block_table` + `dummy_kv_block` sentinel | `spark-model/src/model/impl_b1.rs:95`; `traits.rs:77` | reuse existing sentinel |
| Reselect loop every τ | `EvictionPolicy::rank` + HSS tile loop | `spark-storage/src/eviction.rs:66`; `high_speed_swap/impl_more.rs:183-311` | direct |
| Relevance scorer | `Predictor` (Q·K_lr r=32, advisory, BF16-K) | `spark-storage/src/predictor.rs:63` | direct (LRU mode) |
| Drafter-as-scorer | **new `load_kvflash_scorer()`** mirroring `load_dflash_drafter` | `spark-server/src/main_modules/serve_phases/weights.rs:103`; `spark-model/src/weight_loader/dflash_loader.rs:3` | pattern exists |
| CLI flag + env var | `ServeArgs` (clap); manual dual-check idiom | `spark-server/src/cli.rs:22-491`, `:156` (`--dflash`), `:455` (`ATLAS_FAST_LOAD`) | add `--kvflash` family |
| Per-model policy | `[kvflash]` MODEL.toml → `KvflashConfig` (mirrors `[dflash]`→`DflashConfig`) | `atlas-kernels/src/lib.rs:321`; `atlas-kernels/build_parse.rs:365` | pattern exists |
| Slot-mapped spec verify | `verify_dflash_step` (slot-mapped, EP=2) | `spark-server/src/scheduler/verify_dflash_step.rs:17` | reuse template |
| Per-request repage state | `ActiveSeq` + `SequenceState.block_table` / `disk_block_ids` | `spark-server/src/scheduler/types.rs:88`; `spark-model/src/traits.rs:77,157` | add fields |
| Kernel variant registration | `.cu` stem → module + dispatch match arm | `atlas-kernels/build.rs:868`; `spark-model/src/layers/qwen3_attention/init_kernel_dispatch.rs:28` | file-convention |

## Architectural fork (resolved → extend HSS)

KVFlash's *defining* memory win is that **cache tensors are allocated at pool size, not `max_ctx`**
("that allocation delta IS the memory saving"). Atlas's `PagedKvCache` already bounds *total* VRAM
via a shared free-list, but a single long-context request still pins `logical_ctx / block_size`
blocks. KVFlash makes a single request's resident footprint pool-bounded regardless of logical
context. The port is therefore meaningful and distinct from Atlas's existing pager.

`HighSpeedSwap` is currently hardwired to `IoUringBackend` (`high_speed_swap.rs:29`) and constructed
with one at `:84`. PR1 generalizes the `backend` field to `B: StorageBackend` and ships a
`HostRamBackend` (pinned-host ring buffer, reusing `gpu.alloc_host_pinned`). The HSS orchestrator,
predictor, scratch pool, eviction policy, and tiled attention are reused unchanged — the only
variable is where cold bytes land (disk vs host RAM).

## Phased branch chain

All branches chain off the previous one (branch N+1 is based on branch N). `main` stays in sync
with upstream; all KVFlash artifacts live on the branch chain.

### branch1 — `feat/kvflash-1-hostram-backend`
- Generalize `HighSpeedSwap.backend: IoUringBackend` → `backend: Box<dyn StorageBackend>`.
  (Box, not a generic param: HSS is installed via a `thread_local!` holding a concrete type at
  `high_speed_swap.rs:226`, so `<B>` would break the thread-local. Dyn-dispatch on `read`/
  `write` is negligible — called a few times per step behind ms-scale I/O.)
- Refactor `new_on_stream` so a caller can inject an arbitrary backend; keep an IoUringBackend
  default constructor so existing callers compile unchanged.
- Add `HostRamBackend` in `spark-storage/src/backend/host_ram.rs`: an in-process store of
  pinned host bytes keyed by `GroupKey` (mirrors `PosixBackend`'s bounce-buffer pattern at
  `backend/posix.rs:16`); `read()` issues `copy_h_to_d_async` on the supplied stream;
  `write_from_host()` stores the bytes (no `Layout` / files needed).
- Bit-exact round-trip tests: a pure-host store test (no GPU) plus a full H2D round-trip
  (`#[ignore = "requires GPU"]`, mirroring `posix.rs:111`).
- `PagedKvCache` is NOT touched in PR1 — under "extend HSS" the pool is HSS's `ScratchPool`
  (already pool-sized at `cfg.resident_blocks`). The KVFlash pager that bridges residency to
  per-request KV is introduced in PR3. PR1 is pure `spark-storage`, no CUDA kernel changes.

### branch2 — `feat/kvflash-2-slot-mask`
- Surgical validity: define a `KVFLASH_INVALID_BLOCK` sentinel (reuse the `dummy_kv_block` pattern
  at `impl_b1.rs:95`). Resident-set = the entries of `block_table` that are not the sentinel.
- Add a `fill_slot_validity` helper that rewrites `block_table` entries for paged-out chunks to the
  sentinel before each compute; restore on page-in.
- Relocate-and-replay argmax equivalence unit test (KVFlash gate B/C, ≤1% argmax flips).
- No new kernel arg in MVP; documented follow-up: explicit `paged_decode_attn_masked.cu` variant.

### branch3 — `feat/kvflash-3-reselect-lru`
- `kvflash_maybe_reselect` wired into the decode step at scheduler granularity τ (default 64).
- LRU-only policy via `EvictionPolicy` (`eviction.rs:66`) — no drafter yet.
- CLI flags next to `--dflash` (`cli.rs:156`): `--kvflash <tokens|auto>`, `--kvflash-tau <N>`,
  `--kvflash-policy {drafter,lru}` (default `lru` until PR4). Env: `ATLAS_KVFLASH`,
  `ATLAS_KVFLASH_TAU`, `ATLAS_KVFLASH_MAX_POOL`. (Atlas-prefixed env vars, not `DFLASH_*`.)
- Per-arch gating: qwen35 (gb10/qwen3.6-27b) first; qwen35moe (gb10/qwen3.6-35b-a3b) inherits.
- `auto` sizing: half of free VRAM after weights/reserves at the model's KV density, capped at
  16384 tokens and `--max-ctx`, floored at a protected minimum.

### branch4 — `feat/kvflash-4-drafter-scorer`
- `load_kvflash_scorer()` in `serve_phases/weights.rs` mirroring `load_dflash_drafter`
  (`weights.rs:103`): reads a small Qwen3-0.6B-style drafter, returns a scorer handle.
- `KvFlashScorer` trait (mirror of lucebox's) with `DrafterScorer` and `LruScorer` impls.
- Auto-attach when a drafter is present; `--kvflash-policy drafter` selects it.
- Reuse `Predictor`'s BF16-K dequant path (`spark-runtime/src/kv_dequant.rs`) for quant layers.

### branch5 — `feat/kvflash-5-spec-decode`
- Slot-mapped pool verify in `scheduler/verify_dflash_step.rs` (mirror of the existing slot-mapped
  verify at `:17`). Rejected drafts need no rollback: the `pos < base_pos` validity rule excludes
  their slots until replay rewrites them.
- Acceptance-parity test: pooled vs full cache (upstream measured 15.4–15.6% vs 15.3%).

### branch6 — `feat/kvflash-6-model-toml-crosstok`
- `[kvflash]` MODEL.tomL section → `KvflashConfig` runtime struct (mirrors `DflashConfig`,
  `atlas-kernels/src/lib.rs:321`); build-time parse in `build_parse.rs` mirroring `parse_dflash`
  (`:365`).
- Add `[kvflash]` to `kernels/gb10/qwen3.6-27b/MODEL.toml` and
  `kernels/gb10/qwen3.6-35b-a3b/MODEL.toml`.
- Cross-tokenizer scorer (`KvFlashCrossTokScorer`) for non-qwen targets: detokenize target ids →
  re-tokenize for drafter → score → map back by char spans. (Implemented but untuned for the two
  qwen targets it's not strictly needed; included for parity and future archs.)
- Pooled chunked prefill (qwen35) — prompts larger than the pool prefill in 16-token-block batches
  at constant VRAM.

## Validation

- Port KVFlash test gates A–F (`server/test/test_kvflash.cpp`): baseline KV bytes; relocation proof
  (shuffled placement, teacher-forced argmax ≤1% flips); live paging bit-exact roundtrip (≥90% KV
  cut); eviction-then-recall continuity; NIAH needle recall at 6–9% residency.
- Acceptance-parity test for DFlash spec decode pooled vs full cache.
- Run `tests/single_gpu_suite.py` slice on a CUDA host; confirm no NaN on FP8/NVFP4 KV paths
  (AGENTS.md failure mode: "BF16 paged cache routed into an FP8 kernel, silent NaN").
- Local checks before each branch: `cargo fmt --all -- --check`,
  `ATLAS_SKIP_BUILD=1 cargo clippy --workspace --tests --all-features -- -Dwarnings`,
  `bash scripts/check-license-headers.sh`, `typos`.

## Environment caveat

This host (Windows) has no `nvcc`. Branches are validated with `ATLAS_SKIP_BUILD=1 cargo clippy`
and `cargo fmt` only. Real CUDA compilation, the bit-exact/argmax/NIAH gates, and the spec-decode
acceptance test require a CUDA-capable host. Each branch's commit message records which gates have
run and which remain pending.

## Attribution

KVFlash mechanism: Luce-Org/lucebox-hub (`optimizations/kvflash/`), Apache-2.0.
Underlying algorithm: FlashMemory, arXiv:2606.09079.
Reimplemented for Atlas under AGPL-3.0-only; no lucebox source code copied.

---

# NEXT-SESSION HANDOFF — PredictorScorer (deep recall)

**Read this first.** KVFlash works (flat decode, the headline benefit). The one
remaining functional gap is **deep recall** — under LRU, content outside the
recent tail is dropped from attention, so long-document retrieval fails (the
validation test's shallow-needle MISS). The fix is a relevance scorer that
recalls relevant paged-out chunks. This section is the complete, self-contained
brief to implement it.

## Current state (verified on the gb10)

- Decode-only flatness **0.92** across 512–8192 tokens with `--kvflash 1024 --kvflash-compact`
  (cap-compaction makes attention O(pool) immediately; the 7537 cliff is fixed).
- 9 branches pushed, all compile clean on **real CUDA** (gb10, aarch64, kernel
  wildcard). Branch tip: `feat/kvflash-9-drafter-scorer`.
- The scorer-driven residency mechanism is **complete and unit-tested**:
  `page_in` (recall), `plan_reselect` (pure-logic score-driven planner),
  `reselect`, `maybe_reselect` (τ-cadence gate). It is **dormant** — no scorer
  is attached, so the pager is pure-LRU. Attaching ANY `KvFlashScorer` that
  produces real per-chunk relevance scores makes residency score-driven and
  restores deep recall.

## gb10 access (the validation loop is LIVE — use it)

The dev's Windows machine has passwordless SSH to the gb10. **Just use `ssh gb10`.**
- gb10: `192.168.1.123` (`gx10-98db`, aarch64, CUDA 13), user `sascha_schneiders`.
- Repo on gb10: `~/dev/public/atlas` (same fork: `origin` = `sschneiders/atlas`).
- SSH alias `gb10` is in the Windows `~/.ssh/config`; key at
  `~/Documents/keys/gb10/gb10_key` (Windows side). Test with:
  `ssh -o BatchMode=yes gb10 "cd ~/dev/public/atlas && git rev-parse --abbrev-ref HEAD"`
- Workflow: edit on Windows → `git push` → `ssh gb10 "cd ~/dev/public/atlas && git pull && cargo build --release"`
  → run server + `tests/test_kvflash_validation.py` → read real output.
- **Wildcard kernel rebuild** (needed to run the A3B): the build script
  `/tmp/kv_wild.sh` on the gb10 sets `ATLAS_TARGET_MODEL=* ATLAS_TARGET_QUANT=*`.
  The default `cargo build` compiles only `qwen3-next-80b-a3b` (won't load the A3B).
  Most PTX is cached, so wildcard rebuilds are ~30s, not 15-30 min.
- **ssh quoting gotcha**: PowerShell → ssh → bash quoting is brutal. For anything
  non-trivial, write a script file to the gb10 first (`ssh gb10 'cat > /tmp/x.sh
  << "SCRIPT" ... SCRIPT'`) then run it. Avoid `$()`, nested `"`, and `export VAR=*`
  (globs!); prefix assignments `VAR=* cmd` do NOT glob and are safe.

## The drafter = reuse the HSS Predictor (NOT a drafter model)

The original plan (run Qwen3-0.6B + extract its attention) is **wrong for Atlas**:
its attention kernels are fused and don't expose the weight matrix. Atlas already
has the right mechanism — the **HSS `Predictor`** (`crates/spark-storage/src/predictor.rs`),
which scores blocks as `Q_proj · K_lowrank` per-block relevance. It's the same
relevance signal lucebox's drafter produces, without a second model or attention
extraction. **Use it.**

Predictor API (verified, standalone-constructible):
- `Predictor::new_on_stream(stream, dims: PredictorDims, seed) -> Result<Self>` — loads
  its own PTX (`q_lowrank_project`, `kv_lowrank_project`, `predictor_score`), allocates
  the projection matrix P + the low-rank K store `A_g`. Self-contained — does NOT need
  the HighSpeedSwap orchestrator. `PredictorDims { num_layers, num_q_heads, num_kv_heads,
  head_dim, r (rank, try 32), block_size, max_blocks }`.
- `project_kv_block_on_stream(stream, layer, block_id, k_block_dev)` — projects ONE K
  block (BF16) into `A_g`. **K must be BF16** — the A3B's KV is FP8, so dequant first
  (see below).
- `project_q_on_stream(stream, q_dev, q_proj_dev)` — Q is `[num_q_heads, head_dim]` BF16.
- `score_blocks_on_stream(stream, q_proj_dev, a_g_layer_ptr, block_scores_dev, max_blocks)`
  — fills `block_scores_dev[block] = relevance` for ONE layer. Aggregate across the
  attention layers (mean) to get per-chunk scores.

## The integration (5 steps; each gets a gb10 compile + recall check)

**Step 1 — Q-capture hook (spark-model). THE first step, deepest change.**
The scorer needs the current decode Q. The decode Q is internal to the attention
layers. Add a hook: one attention layer (or a chosen layer) writes its per-step Q
(`[num_q_heads, head_dim]` BF16) to a pager-owned device buffer. Concretely: add a
device buffer to the pager (or a thread-local), have the attention layer's decode
path copy its Q there each step (`gpu.copy_d2d` or the Q is already on device — just
stash the ptr). Then `kvflash_step` / the scorer reads it. Look at
`crates/spark-model/src/layers/qwen3_attention/decode/run_paged_decode.rs` for where Q
is materialized per step.

**Step 2 — `A_g` population during eviction (spark-runtime pager).**
In `kvflash_pager.rs::page_out_one`, after `kv_cache.read_block(layer, physical, gpu)`
(which already reads K for the host store), ALSO project it into the Predictor's `A_g`
via `project_kv_block_on_stream`. This is FREE — the K is already being read for
eviction. `A_g` then holds the low-rank K of every paged-out block, which is exactly
what the scorer scores to decide recall. (Always-resident blocks like the tail are
never in `A_g` — fine, they don't need scoring; they're kept anyway.) For FP8 KV,
dequant the read-back K to BF16 before projecting (Step 4).

**Step 3 — `PredictorScorer` (spark-server).**
New struct in `crates/spark-server/src/main_modules/serve_phases/` (near
`load_kvflash_scorer` — it already builds a `WeightStore`-based scorer skeleton;
replace/extend). Holds a `spark_storage::Predictor` + scratch buffers
(`q_proj_dev`, `block_scores_dev`). Implements `KvFlashScorer`:
`score_chunks` → `project_q` (using the stashed Q from Step 1) → loop layers:
`score_blocks_on_stream` → aggregate to per-chunk mean → return `Vec<f32>`.
Constructed in `install_kvflash` (needs the model dims — already available via
`Model::kv_cache_dims()`), attached via `spark_runtime::kvflash_pager::set_scorer`.

**Step 4 — FP8 dequant (the A3B's KV is FP8).**
`PagedKvCache::read_block` returns raw FP8 bytes; the Predictor needs BF16 K. Reuse
`crates/spark-runtime/src/kv_dequant.rs` (host-side dequant to BF16 — already used by
the HSS predictor path, "Phase 6.2.c" per its docstring). Dequant in `page_out_one`
before `project_kv_block`.

**Step 5 — wire + validate.**
Attach the scorer in `install_kvflash`. `maybe_reselect` already invokes `reselect`
every τ steps when a scorer is attached (PR9) — no change needed there. Run
`tests/test_kvflash_validation.py`: success = the **shallow-needle MISS → HIT**
(relevant early-context chunk recalled) AND decode flatness stays ~0.92 (no
regression from the projection/scoring overhead).

## Gotchas / invariants

- **`KvFlashScorer` is `Send`**; the Predictor must be movable to the scheduler thread
  (it holds device ptrs — likely `Send` like the rest of spark-storage; verify on first
  gb10 compile).
- The scorer is called from `reselect`, which holds `&mut self` (pager) + `&mut kv_cache`
  + `gpu`. The scorer (`self.scorer`) borrowing + `kv_cache` borrowing are distinct
  objects — no aliasing, but watch the borrow in `reselect` (the existing code already
  threads this correctly; mirror it).
- **Don't regress the cap-compaction (PR8)**: attention must stay O(pool). The scorer
  only decides WHICH pool-sized set is resident; it must not change the attention span.
- The `DrafterScorer`/`CrossTokScorer` stubs in `kvflash_scorer.rs` and the
  `load_kvflash_scorer` skeleton can be replaced/repurposed for the PredictorScorer —
  they're dormant LRU-fallbacks.
- `KvflashConfig.tau` (default 64) controls reselect cadence; the scorer runs every τ
  decoded tokens. Projection/scoring cost must stay <~15% of decode or raise τ.

## Validation command (on the gb10, after wiring)

```bash
# rebuild (wildcard, ~30s cached):
ssh gb10 'cd ~/dev/public/atlas && /tmp/kv_wild.sh 2>&1 | tail -3'
# start with the scorer (once --kvflash-policy implies scorer attach):
ssh gb10 'cd ~/dev/public/atlas && target/release/spark serve Qwen/Qwen3.6-35B-A3B-FP8 --kvflash 1024 --kvflash-compact --port 8888 > /tmp/kv_srv.log 2>&1 &'
# run the test:
ssh gb10 'cd ~/dev/public/atlas && python3 tests/test_kvflash_validation.py --url http://localhost:8888 --pool 1024'
```
**Success**: `needle recall: {'shallow': 'HIT', ...}` (was MISS) + `decode flatness ~0.9`.

---

# SESSION 2 RESULTS — PredictorScorer wired, flatness held, recall blocked

Branch tip: `feat/kvflash-9-drafter-scorer`. Steps 1-3 landed + compile-clean
on the gb10 (real CUDA, `cargo clippy -Dwarnings` across spark-runtime /
spark-model / spark-server). Step 4 (FP8 dequant) is NOT done — the A3B was
validated with `--kv-cache-dtype bf16` instead (BF16 KV path in the scorer).

## What works

- **Decode flatness 0.92-0.93 held** with the scorer attached (BF16 KV,
  `--kvflash 1024 --kvflash-compact`). This is the headline KVFlash benefit
  and it is NOT regressed by the scorer. A/B-confirmed: same config with
  `ATLAS_KVFLASH_NO_SCORER=1` (scorer disabled) is 0.93; with scorer 0.92.
- The full plumbing is live and dormant-safe: Q-capture hook, eviction-time
  projection into A_g, resident-block refresh, score-driven reselect,
  PredictorScorer attached in `install_kvflash`. Smoke test passes.

## Recall is BLOCKED — root cause diagnosed (NOT a plumbing bug)

`shallow` (and `deep`) needle stay **MISS** with the scorer. Diagnosis:

- The Predictor's per-block scores are **uniformly ~30-40** across ALL blocks
  (filler + needle). The argmax jumps around randomly and is **never the
  needle block**. So `plan_reselect` recalls the wrong paged-out chunks.
- This is **not** the cross-stream `k_scratch` race (that was real and is
  fixed — `stream_sync` after each projection kernel — but the score
  distribution was already uniform before and after). Verified by the
  `score_chunks argmax` debug log.
- It is **not** the capture layer (tried layer 0 and layer 9 — same uniform
  distribution).
- It is the **fundamental reactive-recall limitation**: the decode Q only
  aligns with a paged-out chunk if the model is already "attending toward"
  it. Once the needle is paged out the model can't attend to it, so the
  decode Q doesn't reflect it, so `Q·K_lowrank` doesn't rank it high, so it
  isn't recalled. The HSS Predictor was designed for proactive *eviction*
  (which blocks to drop); it does not provide a signal that can *find*
  unseen content. Q-driven relevance cannot anticipate future relevance.

## What it took to keep flatness at 0.92 (three fixes on top of step 3)

1. **Cap per-reselect swap to 8** (`reselect`, NOT `plan_reselect` — the
   pure planner + its 4 unit tests are untouched). `plan_reselect` aims for
   the full top-pool each call, but scores move with each decode Q so an
   unbounded swap churned 50/50 blocks every τ → 5x decode slowdown
   (flatness 0.22). Cap → gradual convergence → flatness restored.
2. **Zero A_g at construction** (cuMemAlloc doesn't zero) so unprojected
   blocks score exactly 0 instead of garbage (which churned on noise).
3. **`first_reselect_pending`**: fire reselect on the FIRST decode step
   (right after the prefill question) in addition to every τ, so a relevant
   paged-out chunk is recalled before the answer is generated.
4. **`mark_projected` in the eviction path** so a recalled block isn't
   re-projected every reselect.
5. (Race fix) **`stream_sync` after each projection kernel** so each block's
   kernel reads its own `k_scratch`.

## Status of the 5 steps

- Step 1 (Q-capture): DONE.
- Step 2 (A_g population at eviction): DONE (uses logical block id).
- Step 3 (PredictorScorer): DONE (BF16 path).
- Step 4 (FP8 dequant): NOT DONE. The scorer's `project_bf16` skips non-BF16
  K (score ~0). For the A3B's default FP8 KV, dequant is needed. The FP8
  scale lives on the attention layer (`Qwen3AttentionLayer::effective_fp8_scales`
  → `self.attn.k_scale`), NOT on the cache — needs a bridge (capture via the
  Q-capture hook, or a model accessor) to reach the scorer.
- Step 5 (wire + validate): wiring DONE; validation PARTIAL (flatness ✓,
  recall ✗).

## Next steps for recall (the open problem)

The Q-driven Predictor score can't find unseen paged-out content. Options to
explore, in rough order of promise:

1. **Proactive diverse retention** instead of reactive recall: keep a
   rolling diverse sample of paged-out chunks resident (not just Q-relevant
   ones) so the needle has a chance to be in-window when asked. Trade some
   pool slots for recall coverage.
2. **On-demand re-prefetch**: when the model's decode logits show
   uncertainty / a "looking-back" pattern (e.g. lookback lens spike),
   re-prefetch a span of paged-out chunks. Heuristic, not Q-driven.
3. **Score normalization / rank tuning**: the uniform ~30-40 scores suggest
   the dot product is magnitude-dominated. A cosine-like score (normalize
   A_g per block) or higher rank *might* discriminate — but r=64 already
   costs ~1.7-3.4 GB of A_g on the A3B, and the chicken-and-egg argument
   above suggests the signal isn't there to find. Low promise.
4. **Re-examine the FlashMemory paper's recall mechanism** (arXiv:2606.09079)
   — KVFlash is based on it; it may use a non-Q-driven recall we haven't
   ported.

## gb10 validation commands (current)

```bash
# wildcard rebuild (the default cargo build only does qwen3-next-80b-a3b):
ssh gb10 'cd ~/dev/public/atlas && PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=* ATLAS_TARGET_QUANT=* cargo build --release 2>&1 | tail -3'
# A3B + scorer + BF16 KV (FP8 needs step 4):
ssh gb10 'cd ~/dev/public/atlas && pkill -f "spark serve"; nohup target/release/spark serve Qwen/Qwen3.6-35B-A3B-FP8 --kvflash 1024 --kvflash-compact --kv-cache-dtype bf16 --port 8888 > /tmp/kv_srv.log 2>&1 &'
ssh gb10 'cd ~/dev/public/atlas && python3 tests/test_kvflash_validation.py --url http://localhost:8888 --pool 1024'
# A/B isolation (scorer off = pure LRU baseline):
ssh gb10 'cd ~/dev/public/atlas && ATLAS_KVFLASH_NO_SCORER=1 target/release/spark serve ...'
```
NB: ssh quoting from PowerShell is brutal — write scripts to /tmp on the
gb10 (scp a .sh) instead of inline heredocs. nvcc is at `/usr/local/cuda/bin`
but NOT on PATH in non-interactive ssh shells, so prefix `PATH=...:$PATH`
for any build (cudarc's build script needs `nvcc --version`).

