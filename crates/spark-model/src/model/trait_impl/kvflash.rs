// SPDX-License-Identifier: AGPL-3.0-only

//! KVFlash decode-loop dispatch (PR7). Thin wrappers over the thread-local
//! pager in `spark_runtime::kvflash_pager`. The trait impl in `mod.rs`
//! delegates `kvflash_begin` / `kvflash_step` / `kvflash_end` / `kv_cache_dims`
//! to the `*_dispatch` helpers here.
//!
//! `kvflash_step_dispatch` is the hot path: it locks the KV cache, computes
//! the pool cap, and runs `evict_to_capacity` via the pager's `with_local`.
//! `kvflash_begin_dispatch` is invoked lazily on a slot's first step (no
//! separate scheduler admission-point surgery). All three are no-ops when the
//! pager is not installed, so non-KVFlash users pay only the `is_active()`
//! thread-local check.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// `(block_size, num_layers)` from the live KV cache. Consumed by the
    /// scheduler's `install_kvflash` to construct the thread-local pager.
    pub(super) fn kv_cache_dims_dispatch(&self) -> Option<(u32, usize)> {
        let kv = self.kv_cache.lock();
        Some((kv.block_size() as u32, kv.num_layers()))
    }

    /// Lazily register a decode slot with the pager on its first step. Reads
    /// the dummy block + the config's `protected_tail_blocks`; the slot's
    /// logical block count comes from `seq.block_table.len()`. No-op when the
    /// pager is not installed.
    pub(super) fn kvflash_begin_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        if !spark_runtime::kvflash_pager::is_active() {
            return Ok(());
        }
        let dummy = self.dummy_kv_block;
        let num_logical_blocks = seq.block_table.len() as u32;
        let protected_tail = spark_runtime::kvflash_pager::protected_tail_blocks().unwrap_or(0);
        spark_runtime::kvflash_pager::begin_request(
            seq.slot_idx,
            num_logical_blocks,
            dummy,
            protected_tail,
        );
        Ok(())
    }

    /// Per-decode-step eviction. Locks the KV cache, computes the pool cap
    /// (`pool_tokens / block_size`), and runs `evict_to_capacity` via the
    /// pager's `with_local`. Lazily registers the slot on its first step. The
    /// `&mut PagedKvCache` is extracted from the guard BEFORE the closure so
    /// the closure can capture it alongside `&mut seq.block_table` without a
    /// borrow conflict (mirrors the lock-then-pass pattern in
    /// `save_sequence_state_dispatch`).
    pub(super) fn kvflash_step_dispatch(
        &self,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<()> {
        if !spark_runtime::kvflash_pager::is_active() {
            return Ok(());
        }
        let slot = seq.slot_idx;
        // Lazy begin: register the slot on its first step so the scheduler
        // needs no separate admission-point call site.
        if !spark_runtime::kvflash_pager::slot_state_exists(slot) {
            self.kvflash_begin_dispatch(seq)?;
        }
        let pool_blocks = spark_runtime::kvflash_pager::pool_blocks().unwrap_or(usize::MAX);
        // Lock + deref BEFORE the closure so the closure captures a plain
        // `&mut PagedKvCache` (not the guard), avoiding a double-indirection
        // borrow inside the FnOnce.
        let mut kv = self.kv_cache.lock();
        let kv_ref: &mut spark_runtime::kv_cache::PagedKvCache = &mut *kv;
        let gpu = self.gpu.as_ref();
        if let Some(res) = spark_runtime::kvflash_pager::with_local(|pager| {
            pager.evict_to_capacity(slot, &mut seq.block_table, pool_blocks, kv_ref, gpu)
        }) {
            res?;
        }
        Ok(())
    }

    /// Drop a slot's pager state (frees the host store). No-op when the pager
    /// is not installed. Called by the scheduler on sequence finish.
    pub(super) fn kvflash_end_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        if !spark_runtime::kvflash_pager::is_active() {
            return Ok(());
        }
        spark_runtime::kvflash_pager::end_request(seq.slot_idx);
        Ok(())
    }
}
