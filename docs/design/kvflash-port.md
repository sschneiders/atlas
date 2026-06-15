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
