// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 9 — non-last chunk: save SSM snapshot at chunked-prefill block
//! boundaries (Marconi intermediate checkpoint). On future partial
//! prefix hits, restoring from the deepest intermediate checkpoint
//! avoids recomputing SSM for the entire prefix.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(in crate::model) fn prefill_b_save_checkpoint(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        if self.ssm_checkpoint_interval == 0 || !self.ssm_snapshots.is_enabled() {
            return Ok(());
        }
        let bs = kv_cache.block_size();
        let end_token = chunk_start + chunk_len;
        let end_block = end_token / bs;
        if end_block == 0 || !end_block.is_multiple_of(self.ssm_checkpoint_interval) {
            return Ok(());
        }
        // Stale-V cap (mirrors finalize_last): never checkpoint-cache a block
        // past the contiguous fully-written-KV prefix. If this boundary's
        // blocks aren't all KV-valid yet, skip the intermediate insert rather
        // than cache stale V.
        if seq.kv_valid_tokens / bs < end_block {
            tracing::debug!(
                "Skip intermediate checkpoint at block {end_block}: \
                 kv_valid_tokens={} only covers {} complete blocks",
                seq.kv_valid_tokens,
                seq.kv_valid_tokens / bs,
            );
            return Ok(());
        }
        if std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("ckpt_save@{end_token}"),
            );
        }

        let snap_result = match self.ssm_snapshots.save(
            seq.slot_idx,
            seq.session_hash,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        ) {
            Ok(Some(id)) => Some(id),
            Ok(None) => {
                // Pool exhausted — try to reclaim from cache
                if self
                    .ssm_snapshots
                    .reclaim_from_cache(self.prefix_cache.as_ref(), kv_cache)
                {
                    self.ssm_snapshots
                        .save(
                            seq.slot_idx,
                            seq.session_hash,
                            &self.ssm_pool,
                            self.gpu.as_ref(),
                            stream,
                        )
                        .ok()
                        .flatten()
                } else {
                    tracing::warn!(
                        "SSM snapshot pool exhausted and no evictable cached entries — \
                         dropping checkpoint for this chunk. Long-context prefix-cache \
                         hits will recompute SSM state. Consider raising \
                         --ssm-cache-slots."
                    );
                    None
                }
            }
            Err(e) => {
                tracing::warn!("SSM snapshot save error: {e}");
                None
            }
        };
        let Some(snap_id) = snap_result else {
            return Ok(());
        };

        let boundary_tokens = &tokens[..end_token];
        // Phase 6.3 sliding-window: when HSS is engaged AND sliding has begun
        // (hss_window_start > 0), the front of the prefix is no longer
        // represented by physical HBM blocks — the rolling-window slice
        // would mis-represent the cached entry. Skip the prefix-cache insert
        // in that case; the SSM snapshot is freed to avoid leaks.
        let skip_boundary_insert = seq.hss_window_start() > 0 || end_block > seq.block_table.len();
        if skip_boundary_insert {
            self.ssm_snapshots.free(snap_id);
            return Ok(());
        }
        let boundary_blocks = &seq.block_table[..end_block];
        // Vision chunks: skip both the radix insert and the SSM snapshot
        // attach — the placeholder token stream is identical for distinct
        // images of the same prompt, so a future hit would resurrect the
        // prior image's state.
        if self.tokens_have_vision_pad(boundary_tokens) {
            self.ssm_snapshots.free(snap_id);
            return Ok(());
        }
        let boundary_disk = if seq.disk_block_ids.len() >= end_block {
            &seq.disk_block_ids[..end_block]
        } else {
            &[][..]
        };
        // #110 RC3 fix: do NOT insert radix tree nodes for the mid-prefill
        // checkpoint boundary. Those nodes were created with
        // matched_tokens=end_token => is_seq_owned=false => ref_count=1, which
        // makes them immediately evictable WHILE this sequence is still
        // mid-prefill (more chunks pending) and its device block-table —
        // uploaded delta-only — still points at those physical blocks. Under
        // block pressure (concurrency>=4 + deep multi-chunk prefills, the
        // #110 regime) the next chunk's ensure_blocks->evict force-zeroes and
        // reallocates a live block => use-after-evict / batch=4 brick.
        //
        // The tree nodes were redundant: the SSM snapshot is found via the
        // independent SsmSnapshotIndex (insert_intermediate_snapshot below),
        // keyed by hash_token_prefix and gated only by token_count <=
        // matched_tokens — it does not read tree nodes. The matched path that
        // lets a future warm hit reach this boundary is supplied by the
        // full-range leaf insert in cache_sequence after this sequence
        // finishes. Dropping the insert removes the evictable-live-block
        // window with zero refcount-accounting change (no leak): the seq's
        // blocks stay pinned by their alloc ref in seq.block_table and only
        // become radix-visible at cache_sequence, after the seq is no longer
        // in-flight.
        if let Some(old) = self.prefix_cache.insert_intermediate_snapshot(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            snap_id,
            seq.session_hash,
            end_token,
        ) {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "Intermediate SSM checkpoint saved at token {} (snapshot_id {}, block {})",
            end_token,
            snap_id,
            end_block,
        );
        Ok(())
    }
}
