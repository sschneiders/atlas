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
//! so the kernel iterates over only the resident blocks (O(pool)), and a
//! matching reduced `seq_len`. RoPE is baked into the cached K at write time
//! (`fused_k_norm_rope_cache_write_bf16` / the MRoPE variant) and the decode
//! kernel takes NO positions array, so dropping/reordering resident blocks is
//! position-safe — each cached K already encodes its absolute rotation and
//! attention scores are per-slot independent.
//!
//! The write `slot` (where the new token's K/V lands) is decoupled from the
//! read `seq_len` and must NOT be compacted; callers leave it pointing at the
//! real resident tail block.

/// Build the compacted (resident-only) block table + reduced `seq_len` for one
/// decode step.
///
/// - `block_table`: the per-seq logical→physical map. Paged-out entries have
///   already been rewritten to `dummy` by the pager.
/// - `seq_len_plus_one`: the kernel's attended length convention is
///   `seq.seq_len + 1` (includes the token currently being generated).
/// - `block_size`: tokens per KV block (16 on gb10).
/// - `dummy`: the zeroed sentinel block id (`dummy_kv_block`).
///
/// Returns `Some((resident_physicals, reduced_seq_len))` when at least one
/// block is paged out (compaction actually shrinks the table), or `None` when
/// nothing is paged out (caller should use the original full table + seq_len —
/// compaction is the identity there, so skipping it avoids needless work).
///
/// `resident_physicals` are the physical block ids in logical order with
/// paged-out entries dropped; the LAST entry is always the tail block (the
/// highest logical index, which the pager protects from eviction). The caller
/// uploads them as the kernel's block_table and passes `reduced_seq_len` as
/// the kernel's seq_len.
pub fn compact_for_attention(
    block_table: &[u32],
    seq_len_plus_one: usize,
    block_size: usize,
    dummy: u32,
) -> Option<(Vec<u32>, usize)> {
    if block_table.is_empty() || block_size == 0 {
        return None;
    }
    // Resident physicals in logical order (drop dummy/paged-out).
    let resident: Vec<u32> = block_table
        .iter()
        .copied()
        .filter(|&p| p != dummy)
        .collect();
    // Nothing paged out → no compaction. (Also catches the all-resident case,
    // where compaction would be a wasteful identity.)
    if resident.len() == block_table.len() {
        return None;
    }
    // tail_valid = valid tokens in the tail (current) block under the
    // seq_len+1 convention. The tail is logical block (n_logical - 1); every
    // block before it is complete (a block is appended only when the previous
    // fills), so the tokens before the tail = (n_logical - 1) * block_size and
    // the tail holds the remainder. Clamp to [1, block_size] for safety
    // (tail always has at least the token being generated).
    let n_logical = block_table.len();
    let before_tail = (n_logical - 1) * block_size;
    let tail_valid = seq_len_plus_one
        .saturating_sub(before_tail)
        .clamp(1, block_size);
    // reduced_seq_len = (R - 1) complete resident blocks + tail_valid.
    // The R-1 non-tail resident blocks are all complete; the tail (last entry
    // of `resident`) carries tail_valid tokens.
    let reduced_seq_len = resident.len().saturating_sub(1) * block_size + tail_valid;
    Some((resident, reduced_seq_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tab(xs: &[u32]) -> Vec<u32> {
        xs.to_vec()
    }

    #[test]
    fn no_compaction_when_nothing_paged_out() {
        // All blocks resident → None (identity, skip).
        let bt = tab(&[10, 11, 12, 13]);
        assert_eq!(compact_for_attention(&bt, 65, 16, 999), None);
    }

    #[test]
    fn no_compaction_for_empty_table() {
        assert_eq!(compact_for_attention(&[], 1, 16, 999), None);
    }

    #[test]
    fn drops_paged_out_entries_preserves_order() {
        // 6 logical blocks; blocks 2 and 4 paged out (== dummy 999).
        let bt = tab(&[10, 11, 999, 13, 999, 15]);
        // seq_len+1 = 65 → tail (block 5) valid = 65 - 5*16 = 65-80 → saturate.
        // Use a seq_len+1 consistent with 6 blocks: e.g. 80 (tail block full).
        let (resident, rlen) = compact_for_attention(&bt, 96, 16, 999).unwrap();
        assert_eq!(resident, vec![10, 11, 13, 15]); // dummy dropped, order kept
        assert_eq!(rlen, 4 * 16); // 4 resident blocks, tail full (96 % 16 == 0)
    }

    #[test]
    fn reduced_seq_len_handles_partial_tail() {
        // 6 logical blocks (0..5), one paged out. seq_len+1 = 81:
        // tokens before tail = 5*16 = 80; tail_valid = 81 - 80 = 1.
        let bt = tab(&[10, 999, 12, 13, 14, 15]);
        let (resident, rlen) = compact_for_attention(&bt, 81, 16, 999).unwrap();
        assert_eq!(resident.len(), 5);
        // (5 resident - 1) complete + tail_valid(1) = 4*16 + 1 = 65.
        assert_eq!(rlen, 65);
    }

    #[test]
    fn reduced_seq_len_identity_when_nothing_paged_matches_seq_len_plus_one() {
        // Sanity: with nothing paged, the formula would yield (n-1)*bs + tail_valid
        // == seq_len+1. We return None in that case, but verify the math by
        // forcing one paged block then checking the all-but-one case.
        let bt = tab(&[10, 11, 999]); // 3 logical, one paged
        let seq_len_plus_one = 49; // 3 blocks: 2 complete (32) + tail 17? clamp to 16.
        let (resident, rlen) = compact_for_attention(&bt, seq_len_plus_one, 16, 999).unwrap();
        assert_eq!(resident, vec![10, 11]);
        // (2-1)*16 + tail_valid. tail_valid = 49 - 2*16 = 49-32 = 17 → clamp 16.
        assert_eq!(rlen, 1 * 16 + 16);
    }

    #[test]
    fn tail_is_always_last_resident_entry() {
        // The current/tail block (highest logical idx) is protected → always
        // present and last in the compacted table.
        let bt = tab(&[999, 999, 999, 999, 42]); // only the tail resident
        let (resident, _rlen) = compact_for_attention(&bt, 81, 16, 999).unwrap();
        assert_eq!(resident, vec![42]);
        assert_eq!(*resident.last().unwrap(), 42);
    }

    #[test]
    fn all_paged_except_sink_and_tail() {
        // Sink (block 0) + tail (block 7) resident; middle paged out.
        let bt = tab(&[100, 999, 999, 999, 999, 999, 999, 108]);
        let (resident, rlen) = compact_for_attention(&bt, 113, 16, 999).unwrap();
        assert_eq!(resident, vec![100, 108]);
        // (2-1)*16 + tail_valid(113 - 7*16 = 113-112 = 1) = 16 + 1 = 17.
        assert_eq!(rlen, 17);
    }

    #[test]
    fn block_size_zero_is_safe() {
        let bt = tab(&[10, 11]);
        assert_eq!(compact_for_attention(&bt, 5, 0, 999), None);
    }
}
