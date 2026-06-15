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

use crate::gpu::GpuBackend;
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
}

/// The decode-loop pager. Installed thread-local after `bind_gpu_to_thread`.
pub struct KvflashPager {
    cfg: KvflashConfig,
    block_size: u32,
    num_layers: usize,
    slots: HashMap<usize, SlotState>,
}

impl KvflashPager {
    /// Construct a pager with the resolved config + KV cache geometry. The
    /// geometry is cached at install (from the model's `PagedKvCache`) so the
    /// per-step eviction loop does not re-lock the cache just to read dims.
    pub fn new(cfg: KvflashConfig, block_size: u32, num_layers: usize) -> Self {
        Self {
            cfg,
            block_size,
            num_layers,
            slots: HashMap::new(),
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
        let num_layers = self.num_layers;
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
            // Read K/V per layer from GPU into host buffers.
            let mut layers_kv = Vec::with_capacity(num_layers);
            for layer in 0..num_layers {
                layers_kv.push(kv_cache.read_block(layer, physical, gpu)?);
            }
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
}
