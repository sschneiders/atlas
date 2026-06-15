// SPDX-License-Identifier: AGPL-3.0-only

//! Slot-space validity planner for pool-aware speculative verify (KVFlash).
//!
//! Under KVFlash, the DFlash verify step becomes slot-mapped: each draft
//! token's KV write targets its pool slot via per-token `kv_write_rows`, and
//! attention runs over a slot-space validity mask built from
//! [`KvflashResidency`]. This module is the pure-logic planner that produces
//! that mask. It does NOT touch CUDA — it only maps absolute token positions
//! to logical blocks and asks the residency tracker whether each block is
//! resident in the GPU pool.
//!
//! Validity rule (see `docs/design/kvflash-port.md` PR5): a verify-batch
//! position is VALID iff its logical block is resident in the GPU pool. A
//! draft token landing on a NON-resident block reads the zeroed
//! `dummy_kv_block` sentinel and is rejected by `decode_verify` — the desired
//! fallback. Because rejected drafts' slots are simply excluded by the
//! `pos < base_pos` rule until the next replay rewrites them, the accept-prefix
//! and `seq_len`/`seq.tokens` rollback in the non-KVFlash verify path is
//! SKIPPED under kvflash (the pool's validity mask handles rejection).
//! Acceptance parity target: 15.4–15.6% pooled vs 15.3% full cache (lucebox
//! measurement).
//!
//! This is UNVALIDATED SCAFFOLDING (no CUDA host): the planner is wired only
//! behind the runtime-validation gate (see `docs/design/kvflash-port.md` PR5).

use crate::kvflash_residency::KvflashResidency;

/// One entry per token in the verify batch: its logical block index and
/// whether the slot-space mask considers it valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotValidity {
    /// Logical block index owning this token (`token_pos / block_size`).
    pub logical_block: u32,
    /// True iff the mask admits this position: its logical block is resident
    /// in the GPU pool. A `false` entry means the token reads the zeroed
    /// sentinel and will be rejected by `decode_verify`.
    pub valid: bool,
}

/// Plans the slot-space validity for a single pool-aware verify step.
///
/// The verify batch is `[base_pos, base_pos+1, ..., base_pos+num_drafts]`
/// (γ+1 tokens: the last-verified prefix anchor + γ drafts). Under KVFlash
/// each token's KV write targets its pool slot and attention runs over a
/// validity mask built per [`KvflashResidency`]. The rule (see
/// `docs/design/kvflash-port.md` PR5):
///   - token 0 (the prefix anchor) is valid iff its block is resident;
///   - draft token `i ≥ 1` is valid iff its block is resident — the in-flight
///     draft's own slot is written by `kv_write_rows` before attention, so a
///     resident block covers it by construction;
///   - a draft landing on a NON-resident block can't be verified against real
///     K/V — `decode_verify` sees the zeroed sentinel and rejects it, which is
///     the desired fallback (rejected drafts need no rollback).
///
/// This planner is pure: it produces the mask the verify forward will upload
/// plus a count of how many drafts land on resident blocks (a quality
/// diagnostic). It does NOT touch CUDA.
///
/// # Precondition
///
/// `block_size > 0`. It is a programmer error (block_size comes from
/// [`KvflashConfig`]) to pass `0`; `new` asserts this in debug builds and the
/// arithmetic methods treat `0` as `1` defensively so they never divide by
/// zero in release builds.
///
/// [`KvflashConfig`]: crate::kvflash_config::KvflashConfig
pub struct KvflashVerifyPlan {
    /// Absolute position of the prefix anchor (last-verified token).
    pub base_pos: u64,
    /// γ (number of draft tokens; the batch is γ+1 tokens including the anchor).
    pub num_drafts: usize,
    /// KV cache block size (tokens per logical block; default 16).
    pub block_size: u32,
}

impl KvflashVerifyPlan {
    /// Construct a plan for a verify batch of `num_drafts + 1` tokens starting
    /// at `base_pos`.
    ///
    /// Infallible; see the struct-level precondition on `block_size`.
    pub fn new(base_pos: u64, num_drafts: usize, block_size: u32) -> Self {
        debug_assert!(
            block_size > 0,
            "block_size must be > 0 (programmer error; comes from KvflashConfig)"
        );
        Self {
            base_pos,
            num_drafts,
            block_size,
        }
    }

    /// Logical block index owning `token_pos` (`token_pos / block_size`).
    ///
    /// Safe-by-construction: a `block_size` of `0` (a programmer error) is
    /// treated as `1` so this never divides by zero.
    pub fn logical_block(&self, token_pos: u64) -> u32 {
        // Defensive: block_size > 0 is a precondition, but guard against a
        // division-by-zero in release builds where the debug_assert is off.
        let bs = self.block_size.max(1) as u64;
        (token_pos / bs) as u32
    }

    /// Per-token validity for the whole verify batch
    /// `[base_pos ..= base_pos + num_drafts]`, applied to `residency`.
    ///
    /// The returned `Vec` has length `num_drafts + 1`: index 0 is the prefix
    /// anchor, indices `1..=num_drafts` are the drafts. `valid` is
    /// `residency.is_resident(logical_block)`; out-of-bounds blocks (beyond
    /// what the request has materialized) return `false`, which is the
    /// correct validity for an unmaterialized block.
    pub fn slot_validity(&self, residency: &KvflashResidency) -> Vec<SlotValidity> {
        (0..=self.num_drafts)
            .map(|i| {
                // saturating_add: base_pos + i must fit u64; saturating is the
                // safe fallback (num_drafts is tiny in practice).
                let token_pos = self.base_pos.saturating_add(i as u64);
                let logical_block = self.logical_block(token_pos);
                SlotValidity {
                    logical_block,
                    valid: residency.is_resident(logical_block as usize),
                }
            })
            .collect()
    }

    /// Count of draft tokens (indices `1..=num_drafts` in the batch) whose
    /// logical block is resident — i.e. verified against real K/V. A quality
    /// diagnostic for the residency policy's verify-step hit rate.
    pub fn resident_draft_count(&self, residency: &KvflashResidency) -> usize {
        self.slot_validity(residency)
            .iter()
            .skip(1) // skip the prefix anchor at index 0
            .filter(|v| v.valid)
            .count()
    }

    /// Number of distinct logical blocks spanned by the verify batch.
    ///
    /// Computed in `u128` so `base_pos + num_drafts` cannot overflow. A batch
    /// with `num_drafts == 0` (just the anchor) spans exactly one block.
    pub fn spanned_block_count(&self) -> usize {
        let bs = self.block_size.max(1) as u128;
        let first = self.base_pos as u128 / bs;
        let last = (self.base_pos as u128 + self.num_drafts as u128) / bs;
        // last >= first always (base_pos + num_drafts >= base_pos), so this
        // subtract is safe; the `.max(1)` is belt-and-suspenders.
        ((last - first + 1) as usize).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_validity_all_resident() {
        // base_pos=32, block_size=16 → anchor on block 2; all 8 blocks resident.
        let plan = KvflashVerifyPlan::new(32, 4, 16);
        let residency = KvflashResidency::new(8);
        let sv = plan.slot_validity(&residency);
        assert_eq!(sv.len(), 5); // num_drafts + 1
        assert!(
            sv.iter().all(|v| v.valid),
            "every position should be resident"
        );
        assert_eq!(sv[0].logical_block, 2); // 32 / 16 = 2
    }

    #[test]
    fn batch_validity_paged_out_middle() {
        // base_pos=14, block_size=16 → batch 14..=18 spans blocks 0 and 1.
        // anchor (pos 14) + draft 0 (pos 15) on block 0; drafts 1-3 (pos
        // 16,17,18) on block 1. Page out block 1 → those drafts are invalid.
        let plan = KvflashVerifyPlan::new(14, 4, 16);
        let mut residency = KvflashResidency::new(4);
        assert!(residency.mark_paged_out(1));
        let sv = plan.slot_validity(&residency);
        assert_eq!(sv.len(), 5);
        // block 0 entries stay resident.
        assert!(sv[0].valid, "anchor (block 0) should be resident");
        assert!(sv[1].valid, "draft 0 (block 0) should be resident");
        assert_eq!(sv[0].logical_block, 0);
        assert_eq!(sv[1].logical_block, 0);
        // block 1 entries are paged out.
        assert!(!sv[2].valid, "draft 1 (block 1) should be paged out");
        assert!(!sv[3].valid, "draft 2 (block 1) should be paged out");
        assert!(!sv[4].valid, "draft 3 (block 1) should be paged out");
        assert_eq!(sv[2].logical_block, 1);
    }

    #[test]
    fn resident_draft_count_matches() {
        let plan = KvflashVerifyPlan::new(14, 4, 16);
        // Partial: block 1 paged out → only draft 0 (block 0) is resident.
        let mut residency = KvflashResidency::new(4);
        assert!(residency.mark_paged_out(1));
        assert_eq!(plan.resident_draft_count(&residency), 1);
        // All-resident → all 4 drafts resident (anchor at index 0 is skipped).
        let residency_all = KvflashResidency::new(4);
        assert_eq!(plan.resident_draft_count(&residency_all), 4);
    }

    #[test]
    fn spanned_block_count() {
        // Fully inside one block: positions 4..=7, all block 0.
        let p_one = KvflashVerifyPlan::new(4, 3, 16);
        assert_eq!(p_one.spanned_block_count(), 1);
        // Crossing a block boundary: positions 14..=18, blocks 0 and 1.
        let p_two = KvflashVerifyPlan::new(14, 4, 16);
        assert_eq!(p_two.spanned_block_count(), 2);
        // Empty drafts (anchor only) spans one block.
        let p_anchor = KvflashVerifyPlan::new(100, 0, 16);
        assert_eq!(p_anchor.spanned_block_count(), 1);
    }

    #[test]
    fn logical_block_indexing() {
        let plan = KvflashVerifyPlan::new(0, 4, 16);
        // Block boundaries.
        assert_eq!(plan.logical_block(0), 0);
        assert_eq!(plan.logical_block(15), 0);
        assert_eq!(plan.logical_block(16), 1);
        assert_eq!(plan.logical_block(31), 1);
        assert_eq!(plan.logical_block(32), 2);
        // base_pos at a block boundary.
        let plan_at_boundary = KvflashVerifyPlan::new(16, 2, 16);
        assert_eq!(plan_at_boundary.logical_block(16), 1);
        // base_pos mid-block.
        let plan_mid = KvflashVerifyPlan::new(20, 2, 16);
        assert_eq!(plan_mid.logical_block(20), 1); // 20 / 16 = 1
        assert_eq!(plan_mid.logical_block(31), 1);
        assert_eq!(plan_mid.logical_block(32), 2);
    }

    #[test]
    fn empty_drafts_batch() {
        // num_drafts == 0 → batch is just the anchor, length 1.
        let plan = KvflashVerifyPlan::new(100, 0, 16);
        let residency = KvflashResidency::new(8);
        let sv = plan.slot_validity(&residency);
        assert_eq!(sv.len(), 1);
        assert!(sv[0].valid);
        assert_eq!(sv[0].logical_block, 6); // 100 / 16 = 6
        assert_eq!(plan.resident_draft_count(&residency), 0); // no drafts
        assert_eq!(plan.spanned_block_count(), 1);
    }
}
