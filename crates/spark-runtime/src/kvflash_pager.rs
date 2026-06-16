// SPDX-License-Identifier: AGPL-3.0-only

//! Thread-local decode-loop KV pager for KVFlash (PR7).
//!
//! Holds, per active decode slot on the scheduler thread, a residency mask
//! ([`KvflashResidency`]) plus a host-RAM store for paged-out logical KV
//! blocks. After each `decode_batch`, the model's `kvflash_step` dispatch
//! calls [`evict_to_capacity`]: while the slot's resident block count exceeds
//! the pool cap, the oldest (lowest-index) non-protected resident block is
//! read back per layer ([`PagedKvCache::read_block`]), copied into the host
//! store, its GPU block returned to the free pool
//! ([`PagedKvCache::return_evicted_block`]), and the slot's `block_table`
//! entry rewritten to the zeroed `dummy_kv_block`. The forward path already
//! resolves missing blocks to the zeroed dummy via the existing
//! `unwrap_or(dummy_kv_block)` convention at the attention launch sites, so
//! NO forward-path edits are required — paged-out blocks simply contribute ~0
//! to attention (KVFlash's `--no-mask` mode).
//!
//! MVP scope: LRU eviction to a per-slot host store only. NO page-in/recall,
//! NO drafter (later PRs). This makes resident KV pool-bounded — the testable
//! headline benefit (decode speed stays flat as context grows).
//!
//! This is the runtime primitive; the model-trait seam + scheduler wiring live
//! in spark-model / spark-server. Mirrors `spark_storage::high_speed_swap`'s
//! thread-local `install_local` / `with_local` pattern (the scheduler thread
//! installs once after `bind_gpu_to_thread`; per-step callers reach it via
//! `with_local`). See `docs/design/kvflash-port.md` PR7.

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::Result;

use crate::gpu::{DevicePtr, GpuBackend};
use crate::kv_cache::PagedKvCache;
use crate::kvflash_config::KvflashConfig;
use crate::kvflash_residency::KvflashResidency;

/// Per-layer `(k_bytes, v_bytes)` for one logical block read back from the GPU.
type HostKvLayer = (Vec<u8>, Vec<u8>);

/// Host-RAM store for one slot's paged-out blocks: `logical_block -> one
/// `(k, v)` pair per layer`, indexed by layer.
type HostStore = HashMap<u32, Vec<HostKvLayer>>;

/// Per-slot pager state (one per active decode request on this thread).
struct SlotState {
    /// Residency + protection bitmap for this request's logical blocks.
    /// NOTE: only the SINK (block 0) is stored in the residency's `protected`
    /// bitmap. The trailing decode window is a SLIDING window — recomputed
    /// each eviction pick from `protected_tail_blocks` + the current total —
    /// so it tracks the growing `block_table` without needing `unprotect`.
    residency: KvflashResidency,
    /// Host-RAM store for paged-out blocks (see [`HostStore`]).
    host_store: HostStore,
    /// The zeroed dummy block the forward path reads for paged-out slots.
    dummy_block: u32,
    /// Number of logical blocks at the trailing edge of the sequence that
    /// are dynamically protected from eviction (causal-continuity window).
    /// A block `idx` is protected iff `idx == 0` (sink) OR
    /// `idx >= total() - protected_tail_blocks`.
    protected_tail_blocks: u32,
    /// Decode steps since the last score-driven reselect on this slot. When it
    /// reaches `cfg.tau` AND a scorer is attached, [`KvflashPager::reselect`]
    /// runs and the counter resets.
    steps_since_reselect: u32,
}

/// The decode-loop pager. Installed thread-local after `bind_gpu_to_thread`.
pub struct KvflashPager {
    cfg: KvflashConfig,
    block_size: u32,
    num_layers: usize,
    /// Cached `cfg.compact` for the thread-local [`compact_enabled`] fast path
    /// (mirrors `block_size` / `num_layers` being cached at install time so
    /// the per-step decode sites do not re-read the config).
    compact: bool,
    /// Optional chunk-relevance scorer. When present, [`KvflashPager::reselect`]
    /// is score-driven (recall relevant paged-out chunks, evict irrelevant
    /// resident ones). When absent, the pager is pure-LRU (recency-only) — the
    /// documented MVP quality limitation. Attached via [`set_scorer`] /
    /// [`KvflashPager::set_scorer`].
    scorer: Option<Box<dyn crate::kvflash_scorer::KvFlashScorer>>,
    slots: HashMap<usize, SlotState>,
}

impl KvflashPager {
    /// Construct a pager with the resolved config + KV cache geometry. The
    /// geometry is cached at install (from the model's `PagedKvCache`) so
    /// the per-step eviction loop does not re-lock the cache just to read dims.
    pub fn new(cfg: KvflashConfig, block_size: u32, num_layers: usize) -> Self {
        let compact = cfg.compact;
        Self {
            cfg,
            block_size,
            num_layers,
            compact,
            scorer: None,
            slots: HashMap::new(),
        }
    }

    /// Attach a chunk-relevance scorer. Once attached, [`Self::reselect`]
    /// becomes score-driven (recall relevant paged-out chunks). Until a real
    /// drafter forward is wired, the scorer's `score_chunks` may still return
    /// LRU-equivalent scores — correctness is preserved either way.
    pub fn set_scorer(&mut self, scorer: Box<dyn crate::kvflash_scorer::KvFlashScorer>) {
        self.scorer = Some(scorer);
    }

    /// True iff a scorer is attached (score-driven reselect available).
    pub fn has_scorer(&self) -> bool {
        self.scorer.is_some()
    }

    /// Forward the per-step decode Query to the attached scorer (if any).
    /// Called from the model's chosen attention layer each decode step so the
    /// scorer's later [`KvFlashScorer::score_chunks`] can rank chunks by
    /// relevance to the current query. No-op when no scorer is attached
    /// (recency/LRU residency) — the scorer's `capture_q` default is itself a
    /// no-op, so this is doubly inert for recency-only scorers.
    pub fn capture_q(
        &mut self,
        q: DevicePtr,
        num_q_heads: u32,
        head_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) {
        if let Some(s) = self.scorer.as_mut() {
            s.capture_q(q, num_q_heads, head_dim, gpu, stream);
        }
    }

    /// The resolved KVFlash config (read-only accessor for callers that need
    /// `pool_tokens` / `protected_tail_blocks` / `tau` / `policy`).
    pub fn cfg(&self) -> &KvflashConfig {
        &self.cfg
    }

    /// Cached KV cache block size in tokens.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Cached number of KV cache layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Cached `cfg.compact` — true iff block-table compaction (PR8) is enabled.
    /// Decode attention sites read this (via the thread-local
    /// [`compact_enabled`]) to decide whether to build the resident-only view.
    pub fn compact(&self) -> bool {
        self.compact
    }

    /// Resident pool cap in BLOCKS (floor of `pool_tokens / block_size`).
    /// `resident_count` is driven down to at most this by [`Self::evict_to_capacity`].
    pub fn pool_blocks(&self) -> usize {
        // `block_size` originates from the live `PagedKvCache` at install time
        // and is always >= 1 by construction (a real KV-cache config value),
        // so this division never panics.
        self.cfg.pool_tokens / self.block_size as usize
    }

    /// Register a decode slot with the pager. Creates a residency with all
    /// `num_logical_blocks` resident; only the sink (logical block 0) is
    /// marked protected in the residency. The trailing decode window is a
    /// SLIDING window — recomputed each eviction pick from
    /// `protected_tail_blocks` + the current total — so it tracks the growing
    /// `block_table` without needing `unprotect`. Idempotent per slot.
    pub fn begin_request(
        &mut self,
        slot: usize,
        num_logical_blocks: u32,
        dummy_block: u32,
        protected_tail_blocks: u32,
    ) {
        if self.slots.contains_key(&slot) {
            return;
        }
        let n = num_logical_blocks as usize;
        let mut residency = KvflashResidency::new(n);
        // Sink block 0 is always resident (FlashMemory always-resident floor).
        residency.protect(0);
        // NOTE: the trailing window is NOT protect()ed here — it slides as the
        // sequence grows, so it is derived dynamically at eviction time.
        self.slots.insert(
            slot,
            SlotState {
                residency,
                host_store: HashMap::new(),
                dummy_block,
                protected_tail_blocks,
                steps_since_reselect: 0,
            },
        );
    }

    /// Sync the slot's residency to a grown `block_table` length: any new
    /// logical blocks (indices `old_total .. new_len`) appended by the decode
    /// loop since the last step are marked resident. The trailing protection
    /// window is recomputed dynamically at eviction time (see
    /// [`Self::pick_eviction_victims`]), so it slides automatically — no
    /// explicit re-protection needed here. No-op for an unregistered slot or
    /// when the length has not grown.
    pub fn sync_to_len(&mut self, slot: usize, new_len: usize) {
        if let Some(st) = self.slots.get_mut(&slot) {
            st.residency.grow(new_len);
        }
    }

    /// Drop a slot's pager state. The `host_store` is dropped = host RAM
    /// freed. Safe to call for a slot that was never registered (no-op).
    pub fn end_request(&mut self, slot: usize) {
        self.slots.remove(&slot);
    }

    /// True iff [`Self::begin_request`] has registered `slot` (and
    /// [`Self::end_request`] has not since removed it).
    pub fn slot_state_exists(&self, slot: usize) -> bool {
        self.slots.contains_key(&slot)
    }

    /// Pure-logic eviction planner: returns the logical block indices that
    /// [`Self::evict_to_capacity`] WOULD page out, in eviction order, WITHOUT
    /// touching the GPU. Under LRU (the [`crate::kvflash_scorer::LruScorer`]'s
    /// ascending-score convention) the victim is the LOWEST-INDEX non-protected
    /// resident block. Eviction stops once `resident_count` would reach
    /// `pool_blocks`; if protected blocks dominate, fewer victims than the
    /// deficit are returned (protected blocks are never evicted). Testable
    /// without a GPU.
    pub fn pick_eviction_victims(&self, slot: usize, pool_blocks: usize) -> Vec<u32> {
        let mut victims = Vec::new();
        let Some(st) = self.slots.get(&slot) else {
            return victims;
        };
        let total = st.residency.total();
        // Dynamic trailing window: the last `protected_tail_blocks` logical
        // blocks are protected (causal-continuity window). Slides as `total`
        // grows, so a block that was tail-protected at one step becomes
        // evictable once enough newer blocks append behind it.
        let tail_start = total.saturating_sub(st.protected_tail_blocks as usize);
        let mut resident = st.residency.resident_count();
        if resident <= pool_blocks {
            return victims;
        }
        // Oldest = lowest index = lowest LruScorer score = first to evict.
        for idx in 0..total {
            if resident <= pool_blocks {
                break;
            }
            if !st.residency.is_resident(idx) {
                continue;
            }
            // Protected: sink (block 0, stored in the residency bitmap) OR
            // inside the current sliding tail window.
            if st.residency.is_protected(idx) || idx >= tail_start {
                continue;
            }
            victims.push(idx as u32);
            resident -= 1;
        }
        victims
    }

    /// The core eviction loop. While the slot's resident block count exceeds
    /// `pool_blocks`: pick the LRU victim set via [`Self::pick_eviction_victims`],
    /// then for each victim read its per-layer K/V back to host
    /// ([`PagedKvCache::read_block`]), store it in the host store, free the GPU
    /// block ([`PagedKvCache::return_evicted_block`]), rewrite
    /// `block_table[logical] = dummy_block`, and mark the block paged-out.
    /// Returns the number of blocks evicted. No-op (returns 0) when the slot
    /// is already at/under cap or not registered.
    ///
    /// `physical_block` is read from `block_table[logical]` BEFORE it is
    /// overwritten with the dummy — the caller's table is the source of truth
    /// for the current physical mapping.
    pub fn evict_to_capacity(
        &mut self,
        slot: usize,
        block_table: &mut [u32],
        pool_blocks: usize,
        kv_cache: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<usize> {
        // Self-sync: grow the residency to the current block_table length so
        // blocks appended by the decode loop since the last step are tracked
        // (resident) before eviction planning. Cheap no-op when unchanged.
        self.sync_to_len(slot, block_table.len());
        let victims = self.pick_eviction_victims(slot, pool_blocks);
        if victims.is_empty() {
            return Ok(0);
        }
        let mut evicted = 0usize;
        for logical in victims {
            let l = logical as usize;
            if l >= block_table.len() {
                break;
            }
            let physical = block_table[l];
            // Read K/V per layer + project into the scorer's low-rank store
            // so the block stays scoreable for recall after page-out.
            let layers_kv = self.read_block_layers(logical, physical, kv_cache, gpu)?;
            // Free the GPU block (bypasses ref-counting: the pager owns it).
            kv_cache.return_evicted_block(physical);
            // Stash the host copy + repoint the table at the zeroed dummy.
            let dummy = match self.slots.get_mut(&slot) {
                Some(st) => {
                    st.host_store.insert(logical, layers_kv);
                    st.residency.mark_paged_out(l);
                    st.dummy_block
                }
                None => break,
            };
            block_table[l] = dummy;
            evicted += 1;
        }
        Ok(evicted)
    }

    /// Evict a single logical block (one iteration of the eviction loop, factored
    /// out so [`Self::reselect`] can reuse it). Returns true if evicted.
    fn page_out_one(
        &mut self,
        slot: usize,
        logical: u32,
        block_table: &mut [u32],
        kv_cache: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<bool> {
        let l = logical as usize;
        if l >= block_table.len() {
            return Ok(false);
        }
        let physical = block_table[l];
        // Read K/V per layer + project into the scorer's low-rank store.
        let layers_kv = self.read_block_layers(logical, physical, kv_cache, gpu)?;
        kv_cache.return_evicted_block(physical);
        let dummy = match self.slots.get_mut(&slot) {
            Some(st) => {
                st.host_store.insert(logical, layers_kv);
                st.residency.mark_paged_out(l);
                st.dummy_block
            }
            None => return Ok(false),
        };
        block_table[l] = dummy;
        Ok(true)
    }

    /// Read one physical block's per-layer K/V back from the GPU into host
    /// buffers, and (if a scorer is attached) project each layer's K into the
    /// scorer's low-rank K store so the block stays scoreable for recall after
    /// it is paged out. The projection is the relevance signal for recall: a
    /// paged-out block whose low-rank K was captured here can later be ranked
    /// by the scorer and paged back in. No-op projection when no scorer is
    /// attached (pure-LRU). Shared by [`Self::evict_to_capacity`] and
    /// [`Self::page_out_one`] (SSOT for the read-back path).
    fn read_block_layers(
        &mut self,
        logical: u32,
        physical: u32,
        kv_cache: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<Vec<HostKvLayer>> {
        let num_layers = self.num_layers;
        let mut layers_kv = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            layers_kv.push(kv_cache.read_block(layer, physical, gpu)?);
        }
        if let Some(s) = self.scorer.as_mut() {
            for (layer, (k, _v)) in layers_kv.iter().enumerate() {
                s.project_evicted_block(layer, logical, k, gpu);
            }
        }
        Ok(layers_kv)
    }

    /// Page a single logical block BACK INTO the GPU pool (recall). Allocates a
    /// fresh GPU block, restores its per-layer K/V from the host store
    /// ([`PagedKvCache::write_block`]), rewrites `block_table[logical]` to the
    /// new physical, and marks the block resident. The inverse of eviction.
    /// Returns true if recalled; false if the block wasn't paged out, the slot
    /// is unknown, or the KV cache has no free block to spare.
    pub fn page_in(
        &mut self,
        slot: usize,
        logical: u32,
        block_table: &mut [u32],
        kv_cache: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<bool> {
        let l = logical as usize;
        if l >= block_table.len() {
            return Ok(false);
        }
        // Pull the host copy out of the slot (removed from the host store).
        let layers_kv = match self.slots.get_mut(&slot) {
            Some(st) => match st.host_store.remove(&logical) {
                Some(v) => v,
                None => return Ok(false), // wasn't paged out
            },
            None => return Ok(false),
        };
        // Alloc a fresh GPU block + restore K/V per layer. If the cache is full,
        // put the host copy back and bail (retry next reselect).
        let physical = match kv_cache.try_alloc_block() {
            Some(p) => p,
            None => {
                if let Some(st) = self.slots.get_mut(&slot) {
                    st.host_store.insert(logical, layers_kv);
                }
                return Ok(false);
            }
        };
        for (layer, (k, v)) in layers_kv.iter().enumerate() {
            kv_cache.write_block(layer, physical, k, v, gpu)?;
        }
        block_table[l] = physical;
        if let Some(st) = self.slots.get_mut(&slot) {
            st.residency.mark_resident(l);
        }
        Ok(true)
    }

    /// Score-driven reselect: converge the slot's resident set toward the
    /// top-`pool_blocks` chunks by relevance (plus protected sink+tail). Evicts
    /// low-score resident non-protected blocks and recalls high-score paged-out
    /// blocks. No-op (returns `(0, 0)`) when no scorer is attached or the slot
    /// is unknown — so with no scorer the pager stays pure-LRU via
    /// [`Self::evict_to_capacity`]. Returns `(n_evicted, n_recalled)`.
    pub fn reselect(
        &mut self,
        slot: usize,
        block_table: &mut [u32],
        pool_blocks: usize,
        kv_cache: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<(usize, usize)> {
        self.sync_to_len(slot, block_table.len());
        let total = match self.slots.get(&slot) {
            Some(st) => st.residency.total(),
            None => return Ok((0, 0)),
        };
        if total == 0 {
            return Ok((0, 0));
        }
        // 1. Refresh A_g: project resident blocks not yet in the scorer's
        //    low-rank K store (new decode tokens since the last reselect) so
        //    they rank alongside paged-out blocks (which were projected at
        //    eviction). Without this, resident blocks would score as 0 and
        //    plan_reselect would thrash the whole pool every τ steps.
        let num_layers = self.num_layers;
        let resident_blocks: Vec<(u32, u32)> = match self.slots.get(&slot) {
            Some(st) => (0..total as u32)
                .filter(|l| {
                    let li = *l as usize;
                    st.residency.is_resident(li) && li < block_table.len()
                })
                .map(|l| (l, block_table[l as usize]))
                .collect(),
            None => return Ok((0, 0)),
        };
        if let Some(scorer) = self.scorer.as_mut() {
            for (logical, physical) in resident_blocks {
                if scorer.is_projected(logical) {
                    continue;
                }
                for layer in 0..num_layers {
                    let (k, _v) = kv_cache.read_block(layer, physical, gpu)?;
                    scorer.project_evicted_block(layer, logical, &k, gpu);
                }
                scorer.mark_projected(logical);
            }
        }
        // 2. Score all materialized chunks (resident or host-backed).
        let scores = match self.scorer.as_mut() {
            Some(s) => s.score_chunks(total),
            None => return Ok((0, 0)),
        };
        // 2. Plan (pure logic; reads residency + protected_tail_blocks).
        let (evict, recall) = {
            let st = match self.slots.get(&slot) {
                Some(s) => s,
                None => return Ok((0, 0)),
            };
            plan_reselect(
                &st.residency,
                st.protected_tail_blocks,
                pool_blocks,
                &scores,
            )
        };
        // 3. Execute: evict first (frees GPU capacity), then recall.
        let mut n_evict = 0usize;
        for &logical in &evict {
            if self.page_out_one(slot, logical, block_table, kv_cache, gpu)? {
                n_evict += 1;
            }
        }
        let mut n_recall = 0usize;
        for &logical in &recall {
            if self.page_in(slot, logical, block_table, kv_cache, gpu)? {
                n_recall += 1;
            }
        }
        Ok((n_evict, n_recall))
    }

    /// τ-cadence gate around [`Self::reselect`]. Increments the per-slot step
    /// counter; when it reaches `cfg.tau` AND a scorer is attached, runs a
    /// score-driven reselect and resets the counter. With no scorer attached
    /// this is a cheap no-op (counter still increments but never fires), so the
    /// MVP stays pure-LRU via [`Self::evict_to_capacity`]. Called every decode
    /// step by the scheduler after the LRU catch-up eviction.
    pub fn maybe_reselect(
        &mut self,
        slot: usize,
        block_table: &mut [u32],
        pool_blocks: usize,
        kv_cache: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let tau = self.cfg.tau;
        let has_scorer = self.scorer.is_some();
        let fire = match self.slots.get_mut(&slot) {
            Some(st) => {
                st.steps_since_reselect = st.steps_since_reselect.saturating_add(1);
                if st.steps_since_reselect >= tau && has_scorer {
                    st.steps_since_reselect = 0;
                    true
                } else {
                    false
                }
            }
            None => false,
        };
        if fire {
            self.reselect(slot, block_table, pool_blocks, kv_cache, gpu)?;
        }
        Ok(())
    }
}

/// Pure-logic reselect planner. Given per-chunk relevance `scores` (higher =
/// more relevant — any [`crate::kvflash_scorer::KvFlashScorer`]'s output),
/// decide which logical blocks to EVICT (currently resident, non-protected,
/// outside the desired top-set) and which to RECALL (currently paged-out,
/// inside the desired top-set), so the resident set converges toward the
/// top-`pool_blocks` chunks by score plus the protected sink+tail window.
/// Testable without a GPU; [`KvflashPager::reselect`] executes the plan.
pub fn plan_reselect(
    residency: &crate::kvflash_residency::KvflashResidency,
    protected_tail_blocks: u32,
    pool_blocks: usize,
    scores: &[f32],
) -> (Vec<u32>, Vec<u32>) {
    use std::collections::HashSet;
    let total = residency.total();
    let n = total.min(scores.len());
    let tail_start = total.saturating_sub(protected_tail_blocks as usize);
    let is_protected = |idx: usize| residency.is_protected(idx) || idx >= tail_start;
    // Non-protected candidates (resident OR paged-out), each with score + flag.
    let mut candidates: Vec<(usize, f32, bool)> = (0..n)
        .filter(|&idx| !is_protected(idx))
        .map(|idx| (idx, scores[idx], residency.is_resident(idx)))
        .collect();
    let num_protected = (0..n).filter(|&i| is_protected(i)).count();
    let non_protected_capacity = pool_blocks.saturating_sub(num_protected);
    // Desired resident among non-protected: top `non_protected_capacity` by score.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let desired: HashSet<usize> = candidates
        .iter()
        .take(non_protected_capacity)
        .map(|(idx, _, _)| *idx)
        .collect();
    let (mut evict, mut recall) = (Vec::new(), Vec::new());
    for (idx, _score, is_resident) in candidates {
        if is_resident && !desired.contains(&idx) {
            evict.push(idx as u32);
        } else if !is_resident && desired.contains(&idx) {
            recall.push(idx as u32);
        }
    }
    (evict, recall)
}

// ── Thread-local installation for the scheduler thread (mirrors
//    spark_storage::high_speed_swap's install_local / with_local) ──
//
// The scheduler thread, after `bind_gpu_to_thread`, calls [`install`] to
// register the pager. The model's per-decode-step `kvflash_step` dispatch
// then accesses it via [`with_local`]. The pager's host-store allocations
// live as long as the thread; cleanup happens on thread exit (or explicit
// drop via [`uninstall`]).

thread_local! {
    static LOCAL: RefCell<Option<KvflashPager>> = const { RefCell::new(None) };
}

/// Install the pager on the current thread. Idempotent (overwrites any prior
/// installation, dropping it).
pub fn install(cfg: KvflashConfig, block_size: u32, num_layers: usize) {
    let pager = KvflashPager::new(cfg, block_size, num_layers);
    LOCAL.with(|cell| {
        *cell.borrow_mut() = Some(pager);
    });
}

/// True iff [`install`] has populated this thread's slot.
pub fn is_active() -> bool {
    LOCAL.with(|cell| cell.borrow().is_some())
}

/// Clear this thread's pager (for shutdown / between tests).
pub fn uninstall() {
    LOCAL.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Run `f` with a `&mut KvflashPager` if installed; returns `None` if not.
/// Mirrors `spark_storage::with_local`.
pub fn with_local<R>(f: impl FnOnce(&mut KvflashPager) -> Result<R>) -> Option<Result<R>> {
    LOCAL.with(|cell| cell.borrow_mut().as_mut().map(f))
}

/// True iff slot `slot` is registered with the thread-local pager. `false`
/// when no pager is installed. Used by the model's `kvflash_step` dispatch to
/// lazily [`begin_request`] a slot on its first step (avoids admission-point
/// surgery in the scheduler).
pub fn slot_state_exists(slot: usize) -> bool {
    LOCAL.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|p| p.slot_state_exists(slot))
            .unwrap_or(false)
    })
}

/// The installed pager's pool cap in blocks, or `None` when no pager is
/// installed. Convenience for the model dispatch so it does not need to reach
/// into the config directly.
/// True iff a scorer is attached to the thread-local pager (score-driven
/// reselect available). `false` when no pager is installed.
pub fn has_scorer() -> bool {
    LOCAL.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|p| p.has_scorer())
            .unwrap_or(false)
    })
}

/// Attach a chunk-relevance scorer to the thread-local pager. No-op when no
/// pager is installed. Once attached, the scheduler's periodic reselect
/// (every `tau` decoded tokens) becomes score-driven: relevant paged-out
/// chunks are recalled, irrelevant resident ones evicted.
pub fn set_scorer(scorer: Box<dyn crate::kvflash_scorer::KvFlashScorer>) {
    let _ = with_local(|p| {
        p.set_scorer(scorer);
        Ok(())
    });
}

/// Forward the per-step decode Q to the thread-local pager's scorer. Called
/// from the model's chosen attention layer each decode step so the scorer's
/// later `score_chunks` can rank chunks by relevance to the current query.
/// No-op when no pager / no scorer is installed.
pub fn capture_q(q: DevicePtr, num_q_heads: u32, head_dim: u32, gpu: &dyn GpuBackend, stream: u64) {
    let _ = with_local(|p| {
        p.capture_q(q, num_q_heads, head_dim, gpu, stream);
        Ok(())
    });
}

/// Run a score-driven [`KvflashPager::reselect`] on the thread-local pager for
/// `slot`. Returns `Some((n_evicted, n_recalled))` if a pager is installed,
/// else `None`. No-op when no scorer is attached (returns `Some((0,0))`).
pub fn reselect(
    slot: usize,
    block_table: &mut [u32],
    pool_blocks: usize,
    kv_cache: &mut PagedKvCache,
    gpu: &dyn GpuBackend,
) -> Option<Result<(usize, usize)>> {
    with_local(|p| p.reselect(slot, block_table, pool_blocks, kv_cache, gpu))
}

/// τ-cadence gate around [`reselect`] on the thread-local pager. Called every
/// decode step by the scheduler; fires a score-driven reselect every `tau`
/// steps when a scorer is attached, else a cheap no-op. Returns `Some(())` if
/// a pager is installed.
pub fn maybe_reselect(
    slot: usize,
    block_table: &mut [u32],
    pool_blocks: usize,
    kv_cache: &mut PagedKvCache,
    gpu: &dyn GpuBackend,
) -> Option<Result<()>> {
    with_local(|p| p.maybe_reselect(slot, block_table, pool_blocks, kv_cache, gpu))
}

pub fn pool_blocks() -> Option<usize> {
    LOCAL.with(|cell| cell.borrow().as_ref().map(|p| p.pool_blocks()))
}

/// The installed pager's `protected_tail_blocks` (from its config), or `None`
/// when no pager is installed. Used by the model's `kvflash_begin` dispatch.
pub fn protected_tail_blocks() -> Option<u32> {
    LOCAL.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|p| p.cfg().protected_tail_blocks)
    })
}

/// True iff block-table compaction (PR8) is enabled on the installed pager.
/// Returns `false` when no pager is installed. Decode attention sites call
/// this to gate the resident-only block-table build (off by default — no
/// behavior change unless `--kvflash-compact` is set).
pub fn compact_enabled() -> bool {
    LOCAL.with(|cell| cell.borrow().as_ref().map(|p| p.compact()).unwrap_or(false))
}

/// Register a decode slot on the thread-local pager. No-op when no pager is
/// installed. Thin wrapper over [`KvflashPager::begin_request`].
pub fn begin_request(
    slot: usize,
    num_logical_blocks: u32,
    dummy_block: u32,
    protected_tail_blocks: u32,
) {
    if let Some(res) = with_local(|p| {
        p.begin_request(slot, num_logical_blocks, dummy_block, protected_tail_blocks);
        Ok(())
    }) && let Err(e) = res
    {
        tracing::error!("kvflash begin_request: {e:#}");
    }
}

/// Drop a decode slot's pager state on the thread-local pager. No-op when no
/// pager is installed. Thin wrapper over [`KvflashPager::end_request`].
pub fn end_request(slot: usize) {
    if let Some(res) = with_local(|p| {
        p.end_request(slot);
        Ok(())
    }) && let Err(e) = res
    {
        tracing::error!("kvflash end_request: {e:#}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvflash_config::{KvflashConfig, KvflashPolicy};

    fn cfg() -> KvflashConfig {
        KvflashConfig {
            pool_tokens: 64, // 4 blocks at block_size 16
            tau: 16,
            policy: KvflashPolicy::Lru,
            protected_tail_blocks: 2,
            compact: false,
        }
    }

    #[test]
    fn begin_request_sink_protected_tail_dynamic() {
        let mut p = KvflashPager::new(cfg(), 16, 4);
        p.begin_request(0, 10, 999, 2);
        let st = p.slots.get(&0).expect("slot registered");
        // Only the sink (block 0) is in the residency's protected bitmap.
        assert!(st.residency.is_protected(0));
        // The trailing window is NOT permanently marked — it slides and is
        // enforced dynamically at eviction time (see sliding_tail test).
        assert!(!st.residency.is_protected(8));
        assert!(!st.residency.is_protected(9));
        // all start resident.
        assert_eq!(st.residency.resident_count(), 10);
        assert_eq!(st.dummy_block, 999);
        assert_eq!(st.protected_tail_blocks, 2);
        // The dynamic tail is still respected by the eviction planner:
        // 10 resident, cap 4, tail=2 (blocks 8,9) + sink (0) protected.
        // Evictable: 1..=7. Need to evict 6 → 1,2,3,4,5,6 (7 stays resident).
        assert_eq!(p.pick_eviction_victims(0, 4), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn sliding_tail_unprotects_old_tail_on_growth() {
        // Sequence grows: a block that was tail-protected becomes evictable
        // once enough newer blocks append behind it.
        let mut p = KvflashPager::new(cfg(), 16, 4);
        p.begin_request(0, 4, 999, 2); // tail = blocks 2,3 (4-2)
        // Cap 1: resident=4, evict down to 1. Sink(0) + tail(2,3) protected.
        // Evictable: block 1 only. Evict 1 → 3 victims needed but only 1
        // evictable, so just [1] (protected blocks dominate).
        assert_eq!(p.pick_eviction_victims(0, 1), vec![1]);
        // Grow to 8 blocks (decode appended 4 more). tail now = blocks 6,7.
        // Block 2 (formerly tail) is now evictable. Resident=8, cap=1, evict 7.
        // Protected: sink 0 + tail 6,7. Evictable resident: 1,2,3,4,5 → evict 5.
        p.sync_to_len(0, 8);
        assert_eq!(p.slots[&0].residency.total(), 8);
        assert_eq!(p.pick_eviction_victims(0, 1), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sync_to_len_grows_residency_with_new_resident_blocks() {
        let mut p = KvflashPager::new(cfg(), 16, 4);
        p.begin_request(0, 3, 999, 1);
        assert_eq!(p.slots[&0].residency.total(), 3);
        assert_eq!(p.slots[&0].residency.resident_count(), 3);
        // Decode appends a 4th logical block.
        p.sync_to_len(0, 4);
        assert_eq!(p.slots[&0].residency.total(), 4);
        assert_eq!(p.slots[&0].residency.resident_count(), 4);
        assert!(p.slots[&0].residency.is_resident(3), "new block resident");
        // No-op when not grown.
        p.sync_to_len(0, 4);
        assert_eq!(p.slots[&0].residency.total(), 4);
        // Unregistered slot is a no-op (no panic).
        p.sync_to_len(99, 100);
    }

    #[test]
    fn begin_request_is_idempotent() {
        let mut p = KvflashPager::new(cfg(), 16, 4);
        p.begin_request(0, 10, 999, 2);
        // second registration for the same slot is ignored.
        p.begin_request(0, 20, 888, 1);
        let st = p.slots.get(&0).expect("slot");
        assert_eq!(st.residency.total(), 10, "first registration preserved");
        assert_eq!(st.dummy_block, 999);
    }

    #[test]
    fn pick_victims_lru_lowest_index_non_protected() {
        let mut p = KvflashPager::new(cfg(), 16, 4);
        // pool_blocks = 64 / 16 = 4.
        p.begin_request(0, 10, 999, 2);
        // 10 resident, cap 4 -> evict 6. Protected: 0 (sink), 8,9 (tail).
        // Evictable non-protected resident: 1..=7 (7 blocks). LRU = lowest 6.
        let v = p.pick_eviction_victims(0, 4);
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn pick_victims_none_when_under_cap() {
        let mut p = KvflashPager::new(cfg(), 16, 4);
        p.begin_request(0, 3, 999, 2);
        // 3 resident <= cap 4 -> nothing to evict.
        assert!(p.pick_eviction_victims(0, 4).is_empty());
    }

    #[test]
    fn pick_victims_clamps_when_protected_dominant() {
        let mut p = KvflashPager::new(cfg(), 16, 4);
        // protect ALL blocks (tail covers everything): nothing evictable.
        p.begin_request(0, 10, 999, 10);
        let v = p.pick_eviction_victims(0, 1);
        assert!(v.is_empty(), "all-protected -> no victims even over cap");
    }

    #[test]
    fn pick_victims_unknown_slot_is_empty() {
        let p = KvflashPager::new(cfg(), 16, 4);
        assert!(p.pick_eviction_victims(7, 1).is_empty());
    }

    #[test]
    fn end_request_drops_state() {
        let mut p = KvflashPager::new(cfg(), 16, 4);
        p.begin_request(0, 10, 999, 2);
        assert!(p.slot_state_exists(0));
        p.end_request(0);
        assert!(!p.slot_state_exists(0));
        // end of an unregistered slot is a no-op.
        p.end_request(123);
    }

    #[test]
    fn pool_blocks_is_tokens_over_block_size() {
        let p = KvflashPager::new(cfg(), 16, 4);
        assert_eq!(p.pool_blocks(), 4); // 64 / 16
        let p2 = KvflashPager::new(cfg(), 8, 4);
        assert_eq!(p2.pool_blocks(), 8); // 64 / 8
    }

    #[test]
    fn thread_local_install_is_active_pool_blocks_uninstall() {
        // Clean slate (another test may have left state on this thread).
        uninstall();
        assert!(!is_active());
        assert_eq!(pool_blocks(), None);
        install(cfg(), 16, 4);
        assert!(is_active());
        assert_eq!(pool_blocks(), Some(4));
        assert_eq!(protected_tail_blocks(), Some(2));
        assert!(!slot_state_exists(0));
        begin_request(0, 10, 999, 2);
        assert!(slot_state_exists(0));
        end_request(0);
        assert!(!slot_state_exists(0));
        uninstall();
        assert!(!is_active());
        assert_eq!(pool_blocks(), None);
    }

    // ── plan_reselect (the score-driven residency decision, pure logic) ──

    #[test]
    fn plan_reselect_recalls_high_score_paged_out_evicts_low_score_resident() {
        // 6 blocks: 0 sink (protected). Non-protected: 1,2,3,4,5. tail=2 -> blocks 4,5 protected.
        // So candidates = {1,2,3}. pool_blocks=2, num_protected=3 (0,4,5) -> non_protected_capacity = 0.
        // Hmm that's degenerate; use a bigger setup.
        let mut r = crate::kvflash_residency::KvflashResidency::new(8);
        r.protect(0);
        // Mark block 2 paged out (a "deep" chunk); rest resident.
        r.mark_paged_out(2);
        // protected_tail_blocks = 2 -> blocks 6,7 protected. candidates = {1,2,3,4,5}.
        // scores: make block 2 (paged-out) HIGH, block 1 (resident) LOW.
        let scores = [0.0, 0.1, 9.0, 5.0, 5.0, 5.0, 0.0, 0.0];
        let (evict, recall) = plan_reselect(&r, 2, 4, &scores);
        // non_protected_capacity = 4 - 3 (protected: 0,6,7) = 1. Top-1 by score = block 2.
        // So desired={2}. Block 2 is paged-out & desired -> recall. Resident non-protected
        // not desired (1,3,4,5) -> evict.
        assert!(
            recall.contains(&2),
            "high-score paged-out block recalled: {recall:?}"
        );
        assert!(
            evict.contains(&1),
            "low-score resident block evicted: {evict:?}"
        );
    }

    #[test]
    fn plan_reselect_no_recall_when_nothing_paged_out() {
        let r = crate::kvflash_residency::KvflashResidency::new(6);
        let scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (evict, recall) = plan_reselect(&r, 0, 3, &scores);
        assert!(recall.is_empty(), "nothing paged out -> nothing to recall");
        // All 6 resident, pool 3 -> 3 to evict (the lowest non-protected scores).
        assert!(!evict.is_empty());
    }

    #[test]
    fn plan_reselect_protected_never_evicted() {
        let mut r = crate::kvflash_residency::KvflashResidency::new(6);
        r.protect(0);
        r.protect(3); // an explicit protect
        let scores = [9.0, 0.0, 0.0, 9.0, 0.0, 0.0];
        let (evict, _recall) = plan_reselect(&r, 1, 1, &scores);
        assert!(
            !evict.contains(&0) && !evict.contains(&3),
            "protected blocks never evicted: {evict:?}"
        );
    }

    #[test]
    fn plan_reselect_converges_to_topk_by_score() {
        // 10 blocks all resident, no protection beyond sink (protected_tail=0).
        // pool=3 -> keep top-3 by score (block 0 protected always kept).
        let mut r = crate::kvflash_residency::KvflashResidency::new(10);
        r.protect(0);
        let scores = [0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0];
        let (evict, recall) = plan_reselect(&r, 0, 3, &scores);
        assert!(recall.is_empty()); // nothing paged out
        // non_protected_capacity = 3 - 1 (sink) = 2. Top-2 by score: blocks 1 (9.0), 3 (8.0).
        // The other non-protected resident (2,4,5,6,7,8,9) -> evict (7 blocks).
        assert_eq!(evict.len(), 7);
        assert!(
            !evict.contains(&1) && !evict.contains(&3),
            "top-score blocks kept"
        );
    }
}
