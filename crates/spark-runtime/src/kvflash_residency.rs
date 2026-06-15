// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request KV residency tracker for KVFlash (surgical slot-validity mask).
//!
//! Tracks which logical KV blocks (block_size=16 token granularity) are
//! currently resident in the GPU pool vs paged out to the host-RAM backend.
//! `apply_to_block_table` rewrites a kernel-bound block table so that paged-out
//! logical blocks point at the (zeroed) `dummy_kv_block` sentinel — reusing
//! Atlas's existing `unwrap_or(dummy_kv_block)` convention at ~15 attention
//! launch sites. The zeroed sentinel contributes ~0 to attention (KVFlash's
//! `--no-mask` mode, measured ~1% argmax flips upstream).
//!
//! Two classes of block are PROTECTED from eviction:
//!   - sinks: the first logical block(s) of the sequence (FlashMemory always-
//!     resident floor), and
//!   - the trailing decode window (the most recent logical blocks, needed for
//!     causal continuity of in-flight generation).

/// A packed residency & protection bitmap for the logical blocks of one request.
pub struct KvflashResidency {
    resident: Vec<bool>,   // true = logical block currently in the GPU pool
    protected: Vec<bool>,  // true = never pageable (sink or trailing window)
    resident_count: usize, // number of `resident` entries that are true
}

impl KvflashResidency {
    /// Create a residency tracker for `num_blocks` logical blocks.
    ///
    /// A fresh request has every block it has written resident in the GPU
    /// pool, so all entries start `resident=true`, `protected=false`, and
    /// `resident_count == num_blocks`. `num_blocks == 0` yields an empty
    /// (but usable) residency.
    pub fn new(num_blocks: usize) -> Self {
        Self {
            resident: vec![true; num_blocks],
            protected: vec![false; num_blocks],
            resident_count: num_blocks,
        }
    }

    /// Number of logical block slots tracked (resident + paged out).
    pub fn total(&self) -> usize {
        self.resident.len()
    }

    /// Number of logical blocks currently resident in the GPU pool.
    pub fn resident_count(&self) -> usize {
        self.resident_count
    }

    /// Grow the tracker to track `new_len` logical blocks. New entries
    /// (indices `old_total .. new_len`) are marked resident (freshly written
    /// by the decode loop) and unprotected. Already-tracked entries —
    /// including any previously paged-out interior blocks — are untouched.
    /// No-op when `new_len <= total()` (sequences do not shrink mid-generation
    /// on the happy path; rollback is handled separately by the caller).
    ///
    /// Used by the KVFlash pager to track a growing `block_table` across
    /// decode steps (the table appends a new logical block every `block_size`
    /// generated tokens).
    pub fn grow(&mut self, new_len: usize) {
        let old_len = self.resident.len();
        if new_len <= old_len {
            return;
        }
        let extra = new_len - old_len;
        self.resident.resize(new_len, true);
        self.protected.resize(new_len, false);
        self.resident_count += extra;
    }

    /// Mark block `idx` as never-pageable (a sink or trailing-window block).
    ///
    /// Idempotent. Out-of-bounds indices are a silent no-op.
    pub fn protect(&mut self, idx: usize) {
        if idx < self.protected.len() {
            self.protected[idx] = true;
        }
    }

    /// Mark every block in `[start, end_exclusive)` as never-pageable.
    ///
    /// Convenience for protecting the trailing decode window and the sinks.
    /// The range is clamped to `[0, total())`; an inverted or empty range is
    /// a no-op and never panics.
    pub fn protect_range(&mut self, start: usize, end_exclusive: usize) {
        let end = end_exclusive.min(self.protected.len());
        if start >= end {
            return;
        }
        self.protected[start..end].fill(true);
    }

    /// Returns `true` if `idx` is protected from eviction. Out-of-bounds → `false`.
    pub fn is_protected(&self, idx: usize) -> bool {
        self.protected.get(idx).copied().unwrap_or(false)
    }

    /// Page block `idx` back into the GPU pool.
    ///
    /// Idempotent; updates `resident_count`. Out-of-bounds indices are a
    /// silent no-op.
    pub fn mark_resident(&mut self, idx: usize) {
        if idx < self.resident.len() && !self.resident[idx] {
            self.resident[idx] = true;
            self.resident_count += 1;
        }
    }

    /// Page block `idx` out of the GPU pool.
    ///
    /// Returns `true` if the block was resident and is now paged out. Returns
    /// `false` if the block is PROTECTED (never page out), already paged out,
    /// or out of bounds. Updates `resident_count` on success.
    pub fn mark_paged_out(&mut self, idx: usize) -> bool {
        // Invariant: `resident.len() == protected.len()` (allocated together
        // in `new()`, never resized), so the bounds check below covers both.
        if idx >= self.resident.len() {
            return false;
        }
        if !self.resident[idx] || self.protected[idx] {
            return false;
        }
        self.resident[idx] = false;
        self.resident_count -= 1;
        true
    }

    /// Returns `true` if `idx` is currently resident in the GPU pool.
    /// Out-of-bounds → `false`.
    pub fn is_resident(&self, idx: usize) -> bool {
        self.resident.get(idx).copied().unwrap_or(false)
    }

    /// Indices of all logical blocks currently paged out, in ascending order.
    pub fn paged_out_indices(&self) -> Vec<usize> {
        self.resident
            .iter()
            .enumerate()
            .filter(|&(_, &r)| !r)
            .map(|(i, _)| i)
            .collect()
    }

    /// Rewrite `block_table` so paged-out logical blocks point at `sentinel`.
    ///
    /// For each logical block index `i` in `0..min(total(), block_table.len())`:
    /// if the block is NOT resident, `block_table[i]` is set to `sentinel`
    /// (Atlas's zeroed `dummy_kv_block`, which contributes ~0 to attention via
    /// the existing `unwrap_or(dummy_kv_block)` convention at the attention
    /// launch sites). Resident entries are left untouched. Entries beyond
    /// `total()` are also left untouched — the caller may pass a table longer
    /// than the residency.
    pub fn apply_to_block_table(&self, block_table: &mut [u32], sentinel: u32) {
        // Zip stops at the shorter operand, so this naturally covers only the
        // overlapping prefix `0..min(total(), block_table.len())`. Entries
        // beyond `total()` (a longer table) are left untouched.
        for (entry, resident) in block_table.iter_mut().zip(self.resident.iter()) {
            if !*resident {
                *entry = sentinel;
            }
        }
    }
}

impl Default for KvflashResidency {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_request_all_resident() {
        let r = KvflashResidency::new(6);
        assert_eq!(r.total(), 6);
        assert_eq!(r.resident_count(), 6);
        for i in 0..6 {
            assert!(r.is_resident(i), "block {i} should start resident");
            assert!(!r.is_protected(i), "block {i} should start unprotected");
        }
        assert!(r.paged_out_indices().is_empty());

        // num_blocks == 0 is allowed and usable.
        let empty = KvflashResidency::new(0);
        assert_eq!(empty.total(), 0);
        assert_eq!(empty.resident_count(), 0);
        assert!(empty.paged_out_indices().is_empty());
    }

    #[test]
    fn mark_paged_out_basic() {
        let mut r = KvflashResidency::new(5);
        assert!(r.mark_paged_out(2));
        assert_eq!(r.resident_count(), 4);
        assert!(!r.is_resident(2));
        assert_eq!(r.paged_out_indices(), vec![2]);

        // paging out an already-paged-out block returns false (no-op).
        assert!(!r.mark_paged_out(2));
        assert_eq!(r.resident_count(), 4);

        // page back in restores it; mark_resident is idempotent.
        r.mark_resident(2);
        assert!(r.is_resident(2));
        assert_eq!(r.resident_count(), 5);
        assert!(r.paged_out_indices().is_empty());
        r.mark_resident(2);
        assert_eq!(r.resident_count(), 5);
    }

    #[test]
    fn protected_blocks_refuse_eviction() {
        let mut r = KvflashResidency::new(8);
        r.protect(0);
        r.protect(7);
        assert!(r.is_protected(0));
        assert!(r.is_protected(7));
        assert!(!r.is_protected(4));

        // protected blocks can never be paged out and stay resident.
        assert!(!r.mark_paged_out(0));
        assert!(r.is_resident(0));
        assert!(!r.mark_paged_out(7));
        assert!(r.is_resident(7));
        assert_eq!(r.resident_count(), 8);

        // a non-protected middle block still pages out fine.
        assert!(r.mark_paged_out(4));
        assert_eq!(r.resident_count(), 7);
    }

    #[test]
    fn protect_range_clamps() {
        let mut r = KvflashResidency::new(5);
        // protect the trailing 3: indices 2, 3, 4.
        r.protect_range(2, 5);
        assert!(!r.is_protected(0));
        assert!(!r.is_protected(1));
        assert!(r.is_protected(2));
        assert!(r.is_protected(3));
        assert!(r.is_protected(4));

        // end_exclusive beyond total clamps to total; never panics.
        r.protect_range(4, 100);
        assert!(r.is_protected(4));

        // inverted / empty range is a no-op.
        r.protect_range(4, 1);
        assert_eq!(r.resident_count(), 5);

        // protect_range on an empty residency is safe.
        let mut e = KvflashResidency::new(0);
        e.protect_range(0, 10);
        assert_eq!(e.total(), 0);
    }

    #[test]
    fn apply_to_block_table_rewrites_only_non_resident() {
        let mut r = KvflashResidency::new(6);
        r.protect(0);
        assert!(r.mark_paged_out(2));
        assert!(r.mark_paged_out(4));

        let mut block_table = [10u32, 11, 12, 13, 14, 15];
        r.apply_to_block_table(&mut block_table, 999);
        assert_eq!(block_table, [10, 11, 999, 13, 999, 15]);

        // a table longer than the residency leaves the trailing entries untouched.
        let mut r2 = KvflashResidency::new(6);
        r2.protect(0);
        r2.mark_paged_out(2);
        r2.mark_paged_out(4);
        let mut long = [10u32, 11, 12, 13, 14, 15, 16, 17];
        r2.apply_to_block_table(&mut long, 999);
        assert_eq!(long, [10, 11, 999, 13, 999, 15, 16, 17]);
    }

    #[test]
    fn apply_respects_bounds() {
        // total == 0 applied to any block_table is a no-op.
        let empty = KvflashResidency::new(0);
        let mut table = [1u32, 2, 3, 4];
        empty.apply_to_block_table(&mut table, 999);
        assert_eq!(table, [1, 2, 3, 4]);

        // residency larger than the table: only the overlapping prefix rewrites.
        let mut r = KvflashResidency::new(10);
        r.mark_paged_out(0);
        r.mark_paged_out(1);
        let mut small = [100u32, 200];
        r.apply_to_block_table(&mut small, 0);
        assert_eq!(small, [0, 0]);
    }

    #[test]
    fn out_of_bounds_ops_are_safe() {
        let mut r = KvflashResidency::new(3);

        // out-of-bounds reads return false, never panic.
        assert!(!r.is_resident(999));
        assert!(!r.is_protected(999));
        assert!(!r.paged_out_indices().contains(&999));

        // out-of-bounds writes are no-ops / return false; never panic.
        r.protect(999);
        assert!(!r.is_protected(999));
        r.mark_resident(999);
        assert_eq!(r.resident_count(), 3);
        assert!(!r.mark_paged_out(999));
        assert_eq!(r.resident_count(), 3);
        r.protect_range(999, 2000);

        // in-bounds ops still work correctly after the no-ops.
        assert!(r.mark_paged_out(1));
        assert!(!r.is_resident(1));
        assert_eq!(r.resident_count(), 2);
    }

    #[test]
    fn default_is_empty() {
        let r = KvflashResidency::default();
        assert_eq!(r.total(), 0);
        assert_eq!(r.resident_count(), 0);
    }
}
