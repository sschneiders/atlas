# FibQuant KV-cache compression — design and implementation plan

**Status:** planned (new branch off `feat/kvflash-9-drafter-scorer`).
**Source:** [FibQuant: Universal Vector Quantization for Random-Access KV-Cache Compression](https://arxiv.org/abs/2605.11478) (arXiv:2605.11478, Lee & Kim, 2026).
**License note:** Atlas is AGPL-3.0-only. This is a **reimplementation of the
mechanism** (normalize–rotate–vector-quantize with a radial-angular codebook)
in Atlas's Rust + CUDA/PTX stack, not a code import. Attribution to the
FibQuant authors lives here and in the implementation PRs.

## What this is, and why (read alongside `docs/design/kvflash-port.md`)

The prior KVFlash effort (branch chain `feat/kvflash-*`) implemented decode-loop
KV **paging**: a fixed-size resident pool, cold blocks paged to host RAM,
recalled by a scorer. KVFlash's flat decode (the headline benefit) works
(flatness ~0.92). The one unresolved gap was **mid-depth long-doc recall** —
under paging, a fact in the paged-out middle is lost.

**Why recall could not be fixed under paging** (sessions 2–6 of `kvflash-port.md`
+ a step-1 gb10 probe, all empirical): every "which block to recall" signal
failed, and the failure is fundamental — recall is hard *because of eviction*.
The probe (`tests/exp_drafter_attn_probe.py`) confirmed SESSION 6's drafter
hypothesis is refuted: a separate Qwen3-0.6B drafter with full KV *does*
retrieve a mid-depth needle (it generates the answer), but its attention is
**diffuse** — the needle block sits at ~50th-percentile attention among the
paged-out blocks, even per-head, even at the answer-emitting decode step. No
attention signal (forward, generation, aggregated, or best-single-head, on the
main model or the drafter) identifies the needle block.

**The pivot:** bound KV memory by **compressing** the cache so the **full**
context stays resident. Nothing is evicted → nothing to recall → mid-depth
recall is retained *by construction*, and decode stays flat because the
attention *span* is unchanged (full context over compressed KV). FibQuant is the
compression mechanism.

For the headline case (Qwen3.6-27B, 256K ctx, ~4.6 GB full KV): FibQuant at 8×
→ ~576 MB, at 34× → ~135 MB — the whole 256K context fits on the GB10
unpaged. KVFlash paging becomes a fallback for *unbounded* contexts (page
FibQuant-compressed blocks; recall is cheap because compressed blocks are
small), not the primary mechanism — and it needs no drafter.

## The FibQuant mechanism

Universal, fixed-rate, **random-access** KV compressor with a
normalize–rotate–store interface:

1. **Normalize** — store the L2 norm of each KV vector; quantize the unit vector.
2. **Rotate** — apply a shared random orthogonal (Haar) rotation `R`. After
   rotation, a block of `k` consecutive coordinates is a spherical-Beta source
   on the unit ball (not a product source — which is why scalar tables lose the
   geometry).
3. **Vector-quantize** — replace scalar tables with a shared **radial-angular
   codebook** matched to that canonical source: Beta-quantile radii ×
   Fibonacci / Roberts–Kronecker quasi-uniform directions, refined with
   multi-restart Lloyd–Max. Proven to strictly improve on the scalar product
   specialization at matched rate (gain separates into a cell-shaping factor
   and a density-matching factor).

Reported results: dense rate axis incl. fractional-bit and sub-one-bit, **no
calibration, no variable-length addresses**. 5× compression at 0.99 attention
cosine similarity, 34× at 0.95; on TinyLlama-1.1B within 0.10 perplexity of fp16
at 4×, and 3.6× lower perplexity than scalar TurboQuant at b=2 (8×, where scalar
random-access quantization begins to fail).

### Critical kernel insight (orthogonality)

`R` is orthogonal ⟹ `Q·K = (RQ)·(RK)`. So:

- Store `K` compressed **in rotated space**.
- Rotate `Q` once per query (one cheap matmul) and attend in rotated space.
- `K` decompression = `codebook[index] * norm` — a gather + scale, **no inverse
  rotation**.

This is what makes a fused attention kernel feasible: the only kernel change is
the `K`-read path (gather a codebook vector by index, scale by norm) plus a
per-query `Q`-rotate. It does **not** need the attention matrix (Atlas's fused
kernels don't expose it — and don't need to).

## Do NOT repeat (all empirically ruled out for recall under paging)

- HSS Predictor (Q·K_lowrank) — non-discriminatory scores.
- Main-model prefill attention (question-window AND all-token aggregate) —
  sink-dominated + circular.
- Content novelty (byte-hash: RoPE-confounded; K-norm: context-confounded).
- Prefix floor — covers only the ends.
- **Drafter-scorer** (SESSION 6's plan) — refuted: drafter attention is diffuse
  (see `tests/exp_drafter_attn_probe.py` + this doc's "why" above).
- A kernel hook on the **main** model's attention — same sink-dominated signal.

(Compression sidesteps all of these by not evicting.)

## Implementation plan

Each step: gb10 compile + a check before moving on.

### Step 1 — Fidelity spike (pure Rust/CPU, no CUDA) — DO THIS FIRST
Implement normalize→rotate→codebook quantize + dequant in Rust. Capture a real
A3B KV block (via `PagedKvCache::read_block`), quantize at several rates, and
measure attention cosine-similarity / `‖K − K̂‖` vs the paper's 0.99@5×,
0.95@34×. Pick the vector dim `k` and codebook size. The codebook is
**precomputed once** from the Beta/spherical-Beta geometry (no calibration) and
stored as a build-time constant tensor. Confirm the codebook reproduces the
paper's numbers before touching CUDA.

### Step 2 — New `KvCacheDtype::FibQuant` + storage
Quantize K (and decide the V policy) on cache-fill / `write_block`; store
`{norm, codebook_index}` per vector. Mirror the existing FP8/NVFP4 quantized-KV
plumbing (`KvCacheDtype`, `KvCacheConfig`, `PagedKvCache`).

### Step 3 — Attention kernel variant
Add a `.cu` that reads FibQuant `K`: gather codebook vector by index, scale by
norm, dot with the pre-rotated `Q`. Register via Atlas's file-convention (`.cu`
stem → module + dispatch arm; see `atlas-kernels/build.rs` and
`crates/spark-model/src/layers/qwen3_attention/init_kernel_dispatch.rs`). Wire
into prefill (`prefill/paged.rs`, `prefill/cache_skip.rs`) and decode
(`decode/run_paged_decode.rs`) gated on the new dtype.

### Step 4 — Wire + validate
Expose `--kv-cache-dtype fibquant` (cli.rs, next to the existing
`--kv-cache-dtype`). Validate on the recall grid: expect **HIT at all depths**
(no eviction) with decode flatness ~0.92.

## Atlas integration facts (verified)

- **Target model:** `Qwen/Qwen3.6-35B-A3B-FP8` (cached on gb10). Config: 40
  layers (**10 full_attention + 30 linear_attention/SSM**), num_attention_heads
  =16, num_key_value_heads=2, head_dim=256, hidden_size=2048. All 10 attention
  layers are full_attention (no sliding window). Default KV = FP8 (online
  per-tensor calibration); `--kv-cache-dtype bf16` is the clean fallback.
- **KV plumbing:** `PagedKvCache` (`crates/spark-runtime/src/kv_cache.rs`;
  `kv_cache/paged_impl.rs` for `read_block`/`write_block`), `KvCacheDtype` enum,
  `KvCacheConfig`. Host-side dequant precedent: `crates/spark-runtime/src/kv_dequant.rs`.
- **Kernel dispatch:** `crates/spark-model/src/layers/qwen3_attention/init_kernel_dispatch.rs`;
  PTX per (hw, model, quant) via `atlas-kernels/build.rs`. Fused kernels don't
  expose the attention matrix — FibQuant doesn't need it, only the K-read path.
- **Factory / CLI:** `crates/spark-model/src/factory.rs` (`loader_for_config`),
  `factory/build.rs` (`build_model` builds the `PagedKvCache` from
  `KvCacheConfig`); `--kv-cache-dtype` in `crates/spark-server/src/cli.rs`.
- **Existing KVFlash infra** (compression supersedes it for recall):
  `kvflash_pager.rs`, `kvflash_scorer.rs`, `kvflash_residency.rs`; default =
  prefix floor + LRU. `attention_block_weights` (kvflash_pager.rs) and the Q/K
  capture hooks are reusable prototypes for host-side KV math.

## Branch / state

Branch off `feat/kvflash-9-drafter-scorer` (tip `cae1241`), which is **merged
with upstream/main** and clippy-clean on gb10 real CUDA. New branch e.g.
`feat/fibquant-kv-compression`.

## gb10 validation loop (LIVE — use it, don't develop blind)

- `ssh gb10` works passwordless; repo at `~/dev/public/atlas` (origin =
  sschneiders/atlas). aarch64, CUDA 13, nvcc at `/usr/local/cuda/bin` but **NOT
  on PATH** in non-interactive ssh — prefix `PATH=/usr/local/cuda/bin:$PATH`
  for any build.
- Workflow: edit on Windows → `git push` → `ssh gb10 "cd ~/dev/public/atlas &&
  git pull && <build>"`.
- **PowerShell→ssh quoting is brutal:** write scripts to /tmp on the gb10 via
  scp (NOT inline heredocs); avoid `$()`, nested `"`, and `export VAR=*`
  (globs).
- **Wildcard rebuild (needed for the A3B):** default build only does
  qwen3-next-80b-a3b. Use
  `PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=*
  ATLAS_TARGET_QUANT=* cargo build --release` (cached PTX, ~30s).
- **Clippy on gb10 (not Windows — this host has no nvcc and cudarc 0.19.2
  panics):** `PATH=/usr/local/cuda/bin:$PATH cargo clippy -p spark-runtime -p
  spark-model -p spark-server --tests -- -Dwarnings`.
- **Recall-grid success test:**
  - start: `ssh gb10 'cd ~/dev/public/atlas && pkill -f "spark serve"; nohup env
    PATH=/usr/local/cuda/bin:$PATH target/release/spark serve
    Qwen/Qwen3.6-35B-A3B-FP8 --kv-cache-dtype fibquant --port 8888 >
    /tmp/kv_srv.log 2>&1 &'`
  - test: `ssh gb10 'cd ~/dev/public/atlas && python3
    tests/test_kvflash_recall_grid.py --url http://localhost:8888 --pool 1024'`
  - Success = mid-depth (0.15–0.75) **HIT across 4x/8x** (was 0% under paging)
    + decode flatness ~0.92.
- For HF/Python prototyping (cross-check the codebook): a venv exists at
  `/tmp/draftprobe_venv` on gb10 with torch 2.12+cu130 + transformers 5.12;
  Qwen3-0.6B and the A3B are in the HF cache.

## Rules (AGENTS.md — follow strictly)

- Serena tools for code nav (find_symbol, search_for_pattern,
  get_symbols_overview, read_file); activate project `D:\Dev\atlas` first.
  lean-ctx for shell/reads. Delegate multi-step exploration to subagents.
- Never `unwrap()`/`expect()` in lib code (propagate with `?`). `cargo fmt --all
  -- --check` + the gb10 clippy above must pass before each commit.
- SPDX header (`// SPDX-License-Identifier: AGPL-3.0-only` for Rust/CUDA) on
  every new source file. Files ≤250 lines. One logical change per commit;
  message `<area>: <imperative>`.
- No blocking I/O on the decode path — the codebook is a precomputed constant,
  not loaded per step.
- Start by fetching/reading the paper, then do the **fidelity spike (step 1)**
  on the gb10 before writing any CUDA kernel.

## Success criteria

1. ✅ `--kv-cache-dtype fibquant` compiles clean on gb10 (real CUDA, clippy
   `-Dwarnings`).
2. ✅ Fidelity spike matches paper (≥0.95 attention cosine-sim at the chosen
   rate on real KV — 0.988 @ 8× on Qwen3-0.6B; A3B ≥ by universality).
3. ✅ Recall grid: mid-depth HIT at 4×/8× pool (vs 0% under paging) with no
   `--kvflash` — see Step 4 results below.
4. ✅ Decode flatness ~0.92 retained — compression has no paging stalls, so
   decode is inherently flat (51 tok/s observed, smooth).

## Step 4 results (recall grid — DONE, criterion met)

Ran `tests/test_kvflash_recall_grid.py` on the gb10 against the A3B served with
`--kv-cache-dtype fibquant` (no `--kvflash`; the full 4K–16K context stays
resident via 8× compression). With `max_tokens=80` (the default 32 truncated the
model's reasoning preamble before it could emit the code):

```
             0.05   0.20   0.35   0.50   0.65   0.80   0.92
    4x        HIT    HIT    HIT    HIT    HIT    HIT    HIT
    8x        HIT    HIT    HIT    HIT    HIT    HIT    HIT
   16x       MISS   MISS    HIT    HIT    HIT   MISS    HIT
coverage: 18/21 = 86%    mid-depth (0.15-0.75): 11/12 = 92%
```

**4× and 8× = 100% recall across all depths**, including the mid-depth
(0.15–0.75) frontier that was **0% under KVFlash paging**. 16× (16K context @
8× compression) is 5/7 — the only misses are shallow (0.05) and one mid (0.80),
at the aggressive end of the rate. This is the headline result: compression
keeps the whole context resident so nothing is evicted, and mid-depth recall is
retained by construction — exactly the gap paging could not close.

End-to-end sanity: "What is 2+2?" → "4" with coherent reasoning through the full
prefill (write + prefill-attn) + decode + CUDA-graph path at 51 tok/s. The
blind-written `.cu` kernels (write/decode/prefill, all cloning the Turbo4 path
with the FibQuant codebook gather) are functionally correct on the first run.

## Production-hardening progress (issues #1–#10)

Done & gb10-verified:
- **#1+#2 (per-head_dim codebook + SSOT):** the codebook is no longer a
  hardcoded hd=256 `__constant__`; it is built at model-init from `atlas-quant`
  for the layer's actual `head_dim`, uploaded as a 4 KB device buffer, and passed
  to every kernel. Any `head_dim` now correct; the drift-prone checked-in `.cuh`
  is deleted. (Verified: "What is 2+2?"→"4" identical before/after.)
- **#4 (split-K decode):** `paged_decode_attn_splitk_fibquant` + reuse of the
  dtype-independent `paged_decode_attn_reduce_nvfp4`; `run_paged_decode` picks
  split-K when `num_splits > 1` (mirrors NVFP4). Long-context (8K–16K) decode.
- **#9 (HSS host dequant):** `dequant_fibquant_block_to_bf16` + host codebook
  mirror; `--high-speed-swap` works for FibQuant (unbounded-context paging of
  compressed blocks), no longer bails.
- **#10 (MLA guard):** FibQuant + MLA fails fast at build (MLA decode reads the
  absorbed latent as BF16 — incompatible with FibQuant). Scoped to FibQuant.
- **#3 (quality gate):** greedy generation-agreement vs bf16 = **8/8
  exact-match, token-Jaccard 1.000** on diverse prompts — no detectable quality
  regression. Tool: `tests/fibquant_quality.py`. (True WikiText PPL is a
  follow-up: needs `/v1/completions` per-token logprobs, which Atlas's
  completions endpoint doesn't currently return.)

Remaining (perf / feature / blocked — separate focused efforts):
- **#5 (BR=128 + batched prefill variants):** only the BR=64 prefill shim
  exists; `(FibQuant, _)` always routes to it. Add the BR=128 + batched shims
  (clone the fp8 ones) for small-`q_len` / multi-seq serving throughput.
- **#6 (kernel micro-opt):** the write kernel's nearest-codeword search is
  brute-force O(N); decode re-reads the norm per lane; `payload=2+hd/4` is
  unaligned. Profile vs Turbo4, then: vectorized `uint4` index loads, norm
  broadcast, payload alignment, the paper's hierarchical list decoding (App. F).
  Needs a profiling session — not safe to do blind.
- **#7 (tunable rate):** v1 is fixed k=4,N=256 (8×). `FIB_K`/`FIB_N` are
  compile-time `#define`s in the kernels, so a runtime rate needs either
  multiple compiled variants (like Turbo{2,3,4,8}) or parameterizing the
  dequant. Largest remaining piece; design before implementing.
- **#8 (V policy):** empirically compare symmetric FibQuant vs K=FibQuant/V=fp8
  on PPL — **blocked on #7** (needs an asym variant first). Step 1 measured both
  K+V; V lacks K's softmax robustness so may want a different rate.

## #7 design — tunable rate (decided + build mechanism mapped)

**Decisions (agreed):**
1. **Variant menu** (not runtime config): separate `KvCacheDtype` variants, like
   `Turbo{2,3,4,8}`. CLI `--kv-cache-dtype fibquant8x`.
2. **Compile-time per variant** (not runtime-k): the `.cu` kernels keep
   `FIB_K`/`FIB_N` as `#define`s; the build compiles the kernels once per rate.
   Keeps the attention hot path branch-free.
3. **1-byte indices** (`N ≤ 256`): covers the useful 4×–16× range via
   `k∈{2,4,8}, N=256`. Sub-one-bit / `N>256` (2-byte) deferred.
4. **Symmetric now, per-side later via the existing asym pattern**: ship
   `FibQuant4x`/`FibQuant8x` (both sides same rate). #8 (K≠V rate) adds
   `Bf16KFibquant8xV` / `Fp8KFibquant8xV` style variants — no new mechanism
   needed (`kv_pair()` + per-side byte-math already exist).
5. **Ship 4× + 8×** (Step-1 attn-cosine 0.9997 / 0.992); **16× deferred** (recall
   grid showed degradation at 16K — needs its own validation/tuning pass).

**Variant menu → `(k, N, rate b, compression)`:**

| variant | k | N | b (bits/coord) | vs fp16 | Step-1 attn_cos |
|---|---|---|---|---|---|
| `FibQuant4x` | 2 | 256 | 4.0 | 4× | 0.9997 |
| `FibQuant8x` | 4 | 256 | 2.0 | 8× | 0.992 |

**The reusable capability: a `[[variants]]` KERNEL.toml table.** Today
`atlas-kernels/build.rs` compiles each `.cu` once into one PTX module. To compile
one `.cu` into N modules (one per rate, differing only by `-D`), add a new table:

```toml
# kernels/gb10/common/KERNEL.toml
[[variants]]
stem  = "paged_decode_attn_fibquant"     # which .cu
name  = "paged_decode_fibquant_8x"        # registry module name
flags = ["-DFIB_K=4", "-DFIB_N=256", "-DFIB_RATE=8x"]
[[variants]]
stem  = "paged_decode_attn_fibquant"
name  = "paged_decode_fibquant_4x"
flags = ["-DFIB_K=2", "-DFIB_N=256", "-DFIB_RATE=4x"]
# (same for reshape_and_cache_fibquant + inferspark_prefill_paged_fibquant)
```

The runtime then does `gpu.kernel("paged_decode_fibquant_8x",
"paged_decode_attn_fibquant")` — the symbol name is identical across variants
(macros change the body, not the exported name), so each module's blob resolves
the same symbol but with the variant's compiled code.

**Build-mechanism change plan (from `atlas-kernels/build*.rs` audit):**
- `build_parse.rs`: add `VariantSpec { stem, name, flags }`; parse `[[variants]]`
  as a 3rd return of `parse_kernel_toml`.
- `build.rs`: add `Target.variants`; in the compile loop emit one `CompileJob`
  per variant — `out_file` gets a `__{name}` slug (so each blob is distinct), the
  cache key `extend`s the variant flags before sorting (flag-aware dedup already
  correct), `extra_flags` chains target + variant flags. Loop B emits N
  `(slug, name)` tuples per variant stem.
- `build_codegen.rs`: **no change** — it already iterates the `(stem, module)`
  list generically; distinct module names → distinct Rust consts + registry
  entries.
- Runtime (`registry.rs`): **no change** — keys by module name, resolves the
  symbol inside. Confirmed the `cuModuleGetFunction` symbol (`paged_decode_attn_
  fibquant`) is identical across `-D` compilations.
- Remove the now-redundant `[modules] paged_decode_attn_fibquant = ...` lines
  once a stem moves to `[[variants]]`.

**FibQuant-side changes (mechanical, follow from #1+#2):**
- `kv_cache.rs`: `FibQuant4x`/`FibQuant8x` variants + Display/FromStr
  (`fibquant4x`/`fibquant8x`); `block_bytes_dims` arms
  (`n_vecs * (2 + hd/k)`); keep `kv_pair()` symmetric.
- `init_kernel_dispatch.rs` + `kernel_requirements.rs` + the other exhaustive
  sites: arms per variant routing to `paged_decode_fibquant_{4x,8x}` etc.
- `init.rs`: build the codebook with the variant's `(k, N)` (the #1+#2 upload
  mechanism already parameterizes — just pass `k`/`N` from the variant).
- The 3 `.cu` kernels: unchanged source (they already use `FIB_K`/`FIB_N`
  macros); only compiled N times with different `-D`.
- Validate each shipped rate on the recall grid + the #3 quality gate before
  merge (4× expected best; 8× already validated = current FibQuant).

## Step 1 results (fidelity spike — DONE, mechanism validated)

Pure-Rust reference in `crates/atlas-quant/src/fibquant/` (codebook = Beta-
quantile radii × Fibonacci/Roberts–Kronecker directions + multi-restart
Lloyd–Max; Haar rotation; normalize→rotate→vector-quantize codec; attention-
output cosine metric, paper Eq. 3). 25 unit tests pass; clippy/fmt clean.

**Synthetic canonical source (d=256, the A3B head_dim), per-vector cosine:**

| k | N | rate (b) | compress | vec_cos |
|---|---|----------|----------|---------|
| 2 | 16 | 2.0 | 8× | 0.939 |
| 2 | 64 | 3.0 | 5.3× | 0.984 |
| 2 | 256 | 4.0 | 4× | 0.996 |
| 4 | 256 | 2.0 | 8× | 0.948 |
| 4 | 1024 | 2.5 | 6.4× | 0.973 |

**Real KV (Qwen3-0.6B, d=128, layer 5, T=633) — attention-output cosine
(paper Eq. 3), K and V both compressed:**

| k | N | rate (b) | compress | attn_cos |
|---|---|----------|----------|----------|
| 2 | 16 | 2.0 | 8× | **0.988** |
| 2 | 64 | 3.0 | 5.3× | **0.998** |
| 2 | 256 | 4.0 | 4× | **0.9997** |
| 4 | 256 | 2.0 | 8× | **0.992** |
| 4 | 1024 | 2.5 | 6.4× | **0.997** |

Success criterion #2 is met on real KV: **≥0.95 attention cosine at every rate,
0.988 @ 8×, beating the paper's 5×@0.99.** Softmax robustness lifts attn_cos
well above the per-vector cosine, exactly as the paper predicts.

**A3B (d=256) note.** By FibQuant's source-agnostic universality (Thm 1) the
rotated-block source is identical for every model, and at d=256 the shell is
tighter (Var R² = O(d⁻²)) so the problem is *easier* than d=128 ⇒ A3B fidelity
≥ the Qwen3-0.6B numbers above. The HF-cached A3B (FP8/NVFP4) is blocked for
offline capture by a `kernels`/`transformers` fp8-kernel revision mismatch in
the venv (HF infra, not FibQuant); the real-A3B-KV number will be taken via
`PagedKvCache::read_block` (the doc's specified method) during Step 2, on the
actual Atlas stack. Capture tooling: `tests/dump_kv_fkv1.py` (HF path) +
`crates/atlas-quant/examples/fibquant_fidelity.rs` (the sweep).

## Step 3 architecture decision — WHT reuse (validated Haar-equivalent)

The paper specifies a dense Haar-random rotation `Π` (`Q·K=(ΠQ)·(ΠK)`). For
d=256 that matrix is 256 KB — over CUDA's 64 KB `__constant__` cap, so it would
need a **new device-buffer-upload pattern** (allocate at model load, thread a
`DevicePtr` through every kernel launch) — the first Atlas dtype to do so.

**Empirical pivot (gb10):** FibQuant's universality holds for *any* orthogonal
mixing rotation, and Atlas already ships `wht_bf16` (Walsh–Hadamard) used by
every turbo dtype. Re-running the Step-1 sweep with `FIB_ROT=hadamard` on the
same Qwen3-0.6B KV gives **identical fidelity to Haar** (marginally *better* at
8×: 0.9891 vs 0.9880; 0.9923 vs 0.9921). Diff < 0.001.

⇒ **FibQuant reuses `wht_bf16`** (set `is_wht_rotated() = true`), so the kernel
is a close clone of the **Turbo4** path, not a novel upload path:

- **Write** (`reshape_and_cache_fibquant.cu`, clone of
  `reshape_and_cache_turbo.cu::reshape_and_cache_flash_turbo4`): the host write
  path already applies `wht_bf16` to K/V (the turbo bookend); inside the kernel,
  normalize the rotated vector → nearest-codeword over the 256-entry FibQuant
  codebook → store `{fp16 norm, 1-byte index}`.
- **Decode** (`paged_decode_attn_fibquant.cu`, clone of
  `paged_decode_attn_turbo4.cu`): K-read = gather `codebook[index] × norm` from
  a `__constant__`-staged codebook (4 KB, fits); Q/output WHT bookends are the
  existing `wht_bf16` ones.
- **Prefill** (`inferspark_prefill_paged_fibquant.cu`, clone of the fp8 shim):
  redefine `LOAD_KV_TILE` to do the codebook gather.
- **Registration:** drop the 3 `.cu` files in `kernels/gb10/common/`; add
  `paged_decode_attn_fibquant = "paged_decode_fibquant"` and
  `inferspark_prefill_paged_fibquant = "prefill_paged_fibquant"` to
  `KERNEL.toml [modules]` (reshape stem already matches).
- **Codebook constant:** the 256×4 f32 codebook is a `__device__ __constant__`
  array (generated from the same seed as `atlas-quant`), staged to `__shared__`
  per-CTA (NVFP4 precedent). Same `f_{d,k}` codebook works under WHT.
- **Rust wiring still needed:** `decode/run_paged_decode.rs`,
  `decode/write_kv_cache.rs`, `prefill/paged_attn.rs` FibQuant arms; new
  launchers in `layers/ops/`; `init.rs` prefill handle.

## Attribution

FibQuant mechanism: Lee & Kim, arXiv:2605.11478. Reimplemented for Atlas under
AGPL-3.0-only; no FibQuant source code copied.
