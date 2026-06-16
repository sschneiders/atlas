// SPDX-License-Identifier: AGPL-3.0-only

//! Block-table compaction for KVFlash decode attention (PR8).
//!
//! When KVFlash pages a logical KV block out to host RAM, the pager rewrites
//! that logical block's entry in `seq.block_table` to the zeroed
//! `dummy_kv_block` sentinel (see `kvflash_pager`). The default decode path
//! (the PR7 dummy-mask MVP) still hands the FULL `block_table` to
//! `paged_decode_attn`, so the kernel iterates over O(context) blocks —
//! reading the zeroed dummy for each paged-out block — yielding no compute
//! saving (and a net slowdown from eviction overhead).
//!
//! This module produces the COMPACT view: drop the dummy (paged-out) entries
//! AND cap to `max_blocks` (sink + recent tail) so the kernel iterates over
//! only the resident pool (O(pool)), with a matching reduced `seq_len`. RoPE
//! is baked into the cached K at write time (`fused_k_norm_rope_cache_write_bf16`
//! / the MRoPE variant) and the decode kernel takes NO positions array, so
//! dropping/reordering resident blocks is position-safe — each cached K already
//! encodes its absolute rotation and attention scores are per-slot independent.
//!
//! The write `slot` (where the new token's K/V lands) is decoupled from the
//! read `seq_len` and must NOT be compacted; callers leave it pointing at the
//! real resident tail block.

/// Build the compacted (resident-only) block table + reduced `seq_len` for one
/// decode step, CAPPED to `max_blocks` so attention is O(pool) regardless of
/// how many blocks are currently resident.
///
/// - `block_table`: the per-seq logical→physical map. Paged-out entries have
///   already been rewritten to `dummy` by the pager.
/// - `seq_len_plus_one`: the kernel's attended length convention is
///   `seq.seq_len + 1` (includes the token currently being generated).
/// - `block_size`: tokens per KV block (16 on gb10).
/// - `dummy`: the zeroed sentinel block id (`dummy_kv_block`).
/// - `max_blocks`: the resident-pool cap in BLOCKS (= pool_tokens / block_size).
///   The compacted table contains at most this many entries: the sink
///   (`block_table[0]`, always resident) + the most recent `max_blocks - 1`
///   resident blocks (the trailing decode window). Excess resident blocks
///   (e.g. the ~400 still-resident blocks right after a long prefill, before
///   eviction has caught up) are dropped from the ATTENTION view so the kernel
///   iterates O(pool) immediately — they remain in GPU memory until the pager
///   frees them, but are not attended to.
///
/// Returns `Some((capped_physicals, reduced_seq_len))` when compaction would
/// shrink the table (≥1 block paged out OR resident > max_blocks), else `None`
/// (caller uses the original full table + seq_len — compaction is the identity
/// there, so skipping it avoids needless work).
///
/// `capped_physicals` are in logical order with the tail block (highest logical
/// index) always last. The caller uploads them as the kernel's block_table and
/// passes `reduced_seq_len` as the kernel's seq_len.
pub fn compact_for_attention(
    block_table: &[u32],
    seq_len_plus_one: usize,
    block_size: usize,
    dummy: u32,
    max_blocks: usize,
) -> Option<(Vec<u32>, usize)> {
    if block_table.is_empty() || block_size == 0 || max_blocks == 0 {
        return None;
    }
    // Resident physicals in logical order (drop dummy/paged-out).
    let resident: Vec<u32> = block_table
        .iter()
        .copied()
        .filter(|&p| p != dummy)
        .collect();
    // No compaction needed: nothing paged out AND within the pool cap.
    if resident.len() == block_table.len() && resident.len() <= max_blocks {
        return None;
    }
    // Cap to the sink (resident[0], always block 0) + the most recent
    // (max_blocks - 1) resident blocks. If resident fits within the cap, keep
    // all of it; the cap only bites when resident > max_blocks (e.g. right
    // after a long prefill, before eviction has freed the cold blocks).
    let capped: Vec<u32> = if resident.len() <= max_blocks {
        resident
    } else if max_blocks == 1 {
        vec![*resident.last()?]
    } else {
        let take = max_blocks - 1;
        let mut v = Vec::with_capacity(max_blocks);
        v.push(resident[0]); // sink
        v.extend_from_slice(&resident[resident.len() - take..]);
        v
    };
    // tail_valid = valid tokens in the tail (current) block under the seq_len+1
    // convention. The tail is logical block (n_logical - 1); every block before
    // it is complete, so tokens before the tail = (n_logical - 1) * block_size
    // and the tail holds the remainder. Clamp to [1, block_size]. NB: computed
    // against the REAL logical layout (block_table.len()), not the capped view.
    let n_logical = block_table.len();
    let before_tail = (n_logical - 1) * block_size;
    let tail_valid = seq_len_plus_one
        .saturating_sub(before_tail)
        .clamp(1, block_size);
    let reduced_seq_len = capped.len().saturating_sub(1) * block_size + tail_valid;
    Some((capped, reduced_seq_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tab(xs: &[u32]) -> Vec<u32> {
        xs.to_vec()
    }

    // Large cap so these exercise the paged-out-drop path, not the cap.
    const BIG: usize = 1000;

    #[test]
    fn no_compaction_when_nothing_paged_out_and_under_cap() {
        let bt = tab(&[10, 11, 12, 13]);
        assert_eq!(compact_for_attention(&bt, 65, 16, 999, BIG), None);
    }

    #[test]
    fn no_compaction_for_empty_table() {
        assert_eq!(compact_for_attention(&[], 1, 16, 999, BIG), None);
    }

    #[test]
    fn drops_paged_out_entries_preserves_order() {
        let bt = tab(&[10, 11, 999, 13, 999, 15]);
        let (resident, rlen) = compact_for_attention(&bt, 96, 16, 999, BIG).unwrap();
        assert_eq!(resident, vec![10, 11, 13, 15]);
        assert_eq!(rlen, 4 * 16);
    }

    #[test]
    fn reduced_seq_len_handles_partial_tail() {
        let bt = tab(&[10, 999, 12, 13, 14, 15]);
        let (resident, rlen) = compact_for_attention(&bt, 81, 16, 999, BIG).unwrap();
        assert_eq!(resident.len(), 5);
        assert_eq!(rlen, 65); // (5-1)*16 + 1
    }

    #[test]
    fn all_paged_except_sink_and_tail() {
        let bt = tab(&[100, 999, 999, 999, 999, 999, 999, 108]);
        let (resident, rlen) = compact_for_attention(&bt, 113, 16, 999, BIG).unwrap();
        assert_eq!(resident, vec![100, 108]);
        assert_eq!(rlen, 17); // (2-1)*16 + 1
    }

    #[test]
    fn block_size_zero_is_safe() {
        let bt = tab(&[10, 11]);
        assert_eq!(compact_for_attention(&bt, 5, 0, 999, BIG), None);
    }

    // ── Capping (O(pool) attention even when resident > pool) ──

    #[test]
    fn cap_keeps_sink_plus_recent_tail_when_resident_exceeds_pool() {
        // 8 logical blocks ALL resident (nothing paged out), pool cap = 4.
        // Capped view = sink (block 0) + last 3 resident (blocks 5,6,7).
        let bt = tab(&[10, 11, 12, 13, 14, 15, 16, 17]);
        let (capped, rlen) = compact_for_attention(&bt, 129, 16, 999, 4).unwrap();
        assert_eq!(capped, vec![10, 15, 16, 17]); // sink + last 3
        // tail_valid = 129 - 7*16 = 17 -> clamp 16. reduced = (4-1)*16 + 16 = 64.
        assert_eq!(rlen, 64);
    }

    #[test]
    fn cap_engages_even_when_nothing_paged_out() {
        // Right after a long prefill: many blocks resident, none paged out yet,
        // but attention must STILL be O(pool).
        let bt: Vec<u32> = (0..472).map(|i| 1000 + i).collect(); // 472 resident, no dummies
        let (capped, rlen) = compact_for_attention(&bt, 472 * 16 + 1, 16, 9999, 64).unwrap();
        assert_eq!(capped.len(), 64); // capped to pool
        assert_eq!(*capped.first().unwrap(), 1000); // sink (block 0)
        assert_eq!(*capped.last().unwrap(), 1000 + 471); // tail (last block)
        assert!(rlen <= 64 * 16); // bounded by ~pool, NOT 472 blocks
    }

    #[test]
    fn cap_with_paged_out_middle() {
        // 10 logical: block 0 sink, blocks 3-6 paged out, tail resident.
        // pool cap = 4 → keep sink + last 3 resident.
        let bt = tab(&[100, 101, 102, 999, 999, 999, 999, 107, 108, 109]);
        let (capped, _rlen) = compact_for_attention(&bt, 161, 16, 999, 4).unwrap();
        // resident = [100,101,102,107,108,109]; cap 4 → sink(100) + last 3 (107,108,109).
        assert_eq!(capped, vec![100, 107, 108, 109]);
    }

    #[test]
    fn cap_one_keeps_only_tail() {
        let bt = tab(&[10, 11, 12, 13]);
        let (capped, _rlen) = compact_for_attention(&bt, 65, 16, 999, 1).unwrap();
        assert_eq!(capped, vec![13]); // only the tail
    }

    #[test]
    fn max_blocks_zero_is_safe() {
        let bt = tab(&[10, 11]);
        assert_eq!(compact_for_attention(&bt, 5, 16, 999, 0), None);
    }
}
