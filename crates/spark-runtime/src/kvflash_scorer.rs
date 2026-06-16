// SPDX-License-Identifier: AGPL-3.0-only

//! Chunk-relevance scorers for KVFlash residency (mirrors lucebox's
//! `KvFlashScorer` seam). The pager's reselect loop (PR3 decode-loop
//! integration) calls `score_chunks` every τ decoded tokens to rank all
//! materialized chunks; the top-`pool` chunks stay resident, the rest page
//! out to the host-RAM backend. See docs/design/kvflash-port.md PR4.

use crate::gpu::{DevicePtr, GpuBackend};
use crate::weights::WeightStore;

/// Chunk-relevance policy interface. Object-safe: `&mut self` (the drafter
/// forward mutates its own KV state), no generics.
pub trait KvFlashScorer: Send {
    /// Return a relevance score for every logical chunk in `[0, num_chunks)`.
    /// Higher = more likely to stay resident. The Vec length MUST equal
    /// `num_chunks`.
    fn score_chunks(&mut self, num_chunks: usize) -> Vec<f32>;
    /// Lowercase policy name for logs ("lru" / "drafter").
    fn name(&self) -> &'static str;
    /// Capture the current decode-step Query for later scoring. Called once
    /// per decode step from the chosen attention layer's decode path, BEFORE
    /// [`KvFlashScorer::score_chunks`] runs in the pager's reselect loop on
    /// the same step. The default is a no-op, so recency/LRU scorers ignore
    /// it; a relevance scorer copies the device-side Q into its own buffer
    /// here and reads it back in `score_chunks`. `q` is BF16
    /// `[num_q_heads, head_dim]` on device.
    fn capture_q(
        &mut self,
        _q: DevicePtr,
        _num_q_heads: u32,
        _head_dim: u32,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) {
    }
}

/// Recency-only scorer (the default and the fallback when no drafter is
/// present). Produces a monotonically increasing score by chunk index — chunk
/// 0 (oldest) gets the lowest score, the most recent chunks get the highest —
/// which combined with the pager's eviction yields recency-only residency.
/// No state, no allocations beyond the score Vec.
#[derive(Default)]
pub struct LruScorer;

impl LruScorer {
    pub fn new() -> Self {
        Self
    }
}

impl KvFlashScorer for LruScorer {
    fn score_chunks(&mut self, num_chunks: usize) -> Vec<f32> {
        (0..num_chunks).map(|i| i as f32).collect()
    }
    fn name(&self) -> &'static str {
        "lru"
    }
}

/// Drafter-backed scorer (PR4 skeleton). Holds a loaded small drafter
/// (Qwen3-0.6B-class) WeightStore; `score_chunks` will run the drafter's tail
/// attention as the indexer query and return per-chunk relevance (chunk means
/// of the drafter's attention over the last query — the FlashMemory LSA loop).
///
/// RUNTIME-VALIDATION GATE: the actual drafter forward needs the full
/// spark-model drafter-head pipeline + an active CUDA context and is NOT yet
/// wired. Until wired, `score_chunks` falls back to LRU ordering so correctness
/// is preserved (drafter-driven residency is a quality optimization, not a
/// correctness requirement — the pool is still a hard VRAM cap either way).
pub struct DrafterScorer {
    /// Loaded drafter weights. Consumed by the forward impl once wired.
    #[allow(dead_code)]
    store: WeightStore,
    /// Drafter `hidden_size` from config.json; used by the forward.
    #[allow(dead_code)]
    hidden_size: usize,
    /// Drafter `num_hidden_layers`; used by the forward.
    #[allow(dead_code)]
    num_layers: usize,
}

impl DrafterScorer {
    pub fn new(store: WeightStore, hidden_size: usize, num_layers: usize) -> Self {
        Self {
            store,
            hidden_size,
            num_layers,
        }
    }
    /// Accessors for the (future) forward impl + tests.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

impl KvFlashScorer for DrafterScorer {
    fn score_chunks(&mut self, num_chunks: usize) -> Vec<f32> {
        // TODO(kvflash PR4 runtime-validation): run the drafter forward over
        // the current context, take the tail-attention distribution as the
        // indexer query, and produce per-chunk relevance scores (mean
        // attention weight per 64-token chunk, mapped to Atlas's 16-token
        // block granularity ×4). Until then, LRU fallback preserves
        // correctness.
        tracing::debug!(
            "kvflash DrafterScorer forward not yet wired (hidden={}, layers={}): LRU fallback",
            self.hidden_size,
            self.num_layers
        );
        (0..num_chunks).map(|i| i as f32).collect()
    }
    fn name(&self) -> &'static str {
        "drafter"
    }
}

/// Cross-tokenizer drafter scorer for non-qwen targets (laguna, gemma4).
///
/// Relevance is a property of the TEXT, not the tokenizer: this scorer
/// detokenizes the target's history with the TARGET tokenizer, re-tokenizes
/// for the drafter's tokenizer, scores, and maps scores back to chunk
/// boundaries by character spans (lucebox's `KvFlashCrossTokScorer`).
///
/// RUNTIME-VALIDATION GATE: the detokenize/re-tokenize step needs both
/// tokenizers wired (spark-server's TokenizerRuntime + the drafter's) and is
/// NOT yet implemented. `score_chunks` falls back to LRU ordering so
/// correctness is preserved. Qwen3.6 targets do NOT need this scorer — they
/// feed target ids to the drafter directly (same tokenizer family).
pub struct CrossTokScorer {
    #[allow(dead_code)]
    store: WeightStore,
    #[allow(dead_code)]
    hidden_size: usize,
}

impl CrossTokScorer {
    pub fn new(store: WeightStore, hidden_size: usize) -> Self {
        Self { store, hidden_size }
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl KvFlashScorer for CrossTokScorer {
    fn score_chunks(&mut self, num_chunks: usize) -> Vec<f32> {
        // TODO(kvflash PR6 runtime-validation): detokenize target ids ->
        // re-tokenize for drafter -> score -> map back by char spans. Until
        // then, LRU fallback preserves correctness.
        tracing::debug!(
            "kvflash CrossTokScorer not yet wired (hidden={}): LRU fallback",
            self.hidden_size
        );
        (0..num_chunks).map(|i| i as f32).collect()
    }
    fn name(&self) -> &'static str {
        "cross-tok"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_scores_ascending_by_index() {
        let mut scorer = LruScorer::new();
        let scores = scorer.score_chunks(5);
        assert_eq!(scores, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn lru_score_length_matches_input() {
        let mut scorer = LruScorer::new();
        assert!(scorer.score_chunks(0).is_empty());
        assert_eq!(scorer.score_chunks(1).len(), 1);
        assert_eq!(scorer.score_chunks(64).len(), 64);
    }

    #[test]
    fn lru_name() {
        assert_eq!(LruScorer::new().name(), "lru");
    }

    #[test]
    fn drafter_falls_back_to_lru_ordering() {
        // WeightStore::empty() is the documented testing constructor
        // (weights.rs). Until the forward is wired, DrafterScorer must
        // return ascending (LRU) scores so correctness is preserved.
        let mut scorer = DrafterScorer::new(WeightStore::empty(), 1024, 28);
        assert_eq!(scorer.name(), "drafter");
        assert_eq!(scorer.hidden_size(), 1024);
        assert_eq!(scorer.num_layers(), 28);
        let scores = scorer.score_chunks(4);
        assert_eq!(scores, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn crosstok_falls_back_to_lru_ordering() {
        // WeightStore::empty() is the documented testing constructor.
        // Until the detokenize/re-tokenize bridge is wired, CrossTokScorer
        // must return ascending (LRU) scores so correctness is preserved.
        let mut scorer = CrossTokScorer::new(WeightStore::empty(), 1024);
        assert_eq!(scorer.name(), "cross-tok");
        assert_eq!(scorer.hidden_size(), 1024);
        let scores = scorer.score_chunks(4);
        assert_eq!(scores, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn object_safety_boxed_dyn() {
        // A Box<dyn KvFlashScorer> works for both impls (object safety).
        let lru: Box<dyn KvFlashScorer> = Box::new(LruScorer::new());
        let drafter: Box<dyn KvFlashScorer> =
            Box::new(DrafterScorer::new(WeightStore::empty(), 512, 12));
        assert_eq!(lru.name(), "lru");
        assert_eq!(drafter.name(), "drafter");
    }
}
