// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime config for the KVFlash decode-time KV pager.
//!
//! KVFlash keeps a small resident pool of logical KV blocks in HBM and pages
//! the rest to a host-RAM ([`spark_storage::HostRamBackend`]) backend, reselecting
//! which blocks stay resident every `tau` decoded tokens. This module holds the
//! resolved, validated config consumed by the scheduler + decode loop. The CLI
//! surface (clap flags, env-var fallback) lives in `spark-server::cli`; the
//! builder that turns `--kvflash auto` / env into a concrete [`KvflashConfig`]
//! lives in `spark-server::main_modules::serve_phases::build`.
//!
//! `spark-runtime` is intentionally clap-free, so [`KvflashPolicy`] is a plain
//! enum with `Display` + `FromStr` impls; `spark-server` defines its own
//! `KvflashPolicyArg` (clap `ValueEnum`) and converts into this enum.
//!
//! See `docs/design/kvflash-port.md` PR3 for the integration plan. The
//! per-step decode-loop paging call sites (decode_a.rs, decode_b.rs, prefill_*,
//! verify_*) are the runtime-validation-gated remainder and are intentionally
//! out of scope for this PR.

use anyhow::{Result, bail};
use std::fmt;
use std::str::FromStr;

/// Residency policy. `Lru` is recency-only (no drafter); `Drafter` lazy-loads
/// a small drafter as the relevance scorer (PR4). Defaults to [`KvflashPolicy::Lru`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvflashPolicy {
    #[default]
    Lru,
    Drafter,
}

impl fmt::Display for KvflashPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvflashPolicy::Lru => f.write_str("lru"),
            KvflashPolicy::Drafter => f.write_str("drafter"),
        }
    }
}

impl FromStr for KvflashPolicy {
    type Err = anyhow::Error;

    /// Parse `"lru"` / `"drafter"` (case-insensitive). Any other value is an
    /// error — there is no implicit default here (PCND: callers must pass an
    /// explicit, recognized policy).
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lru" => Ok(KvflashPolicy::Lru),
            "drafter" => Ok(KvflashPolicy::Drafter),
            other => bail!("unknown kvflash policy '{other}' (expected: lru | drafter)"),
        }
    }
}

/// Resolved KVFlash config.
///
/// No `"auto"` survives to this struct — the builder in
/// `spark-server::main_modules::serve_phases::build` converts `"auto"` / env /
/// `--max-ctx` clamp into a concrete [`KvflashConfig::pool_tokens`] (rounded up
/// to a multiple of `block_size`). Mirrors the shape of
/// `spark_storage::HighSpeedSwapConfig`: all fields required (PCND), and
/// [`KvflashConfig::validate`] enforces cross-field invariants.
#[derive(Clone, Debug)]
pub struct KvflashConfig {
    /// Resident pool size in TOKENS (already resolved from `"auto"` / env /
    /// `--max-ctx` clamp). Rounded up to a multiple of `block_size` by the
    /// builder.
    pub pool_tokens: usize,
    /// Reselect interval floor (decoded tokens between residency reselections).
    pub tau: u32,
    /// Residency policy.
    pub policy: KvflashPolicy,
    /// Logical blocks protected from eviction at the tail (for causal
    /// continuity of in-flight generation). In BLOCKS, not tokens.
    pub protected_tail_blocks: u32,
    /// Enable block-table compaction (PR8): when true, the decode attention
    /// sites drop paged-out (dummy) entries from the per-seq block table and
    /// pass a matching reduced `seq_len`, so the kernel iterates over only the
    /// resident pool (O(pool)) instead of the full context (O(ctx)). Off by
    /// default — experimental; validate output correctness vs the dummy-mask
    /// MVP before trusting it. No new invariant (carried by [`Self::validate`]
    /// for completeness; a bool needs no cross-field check).
    pub compact: bool,
}

impl KvflashConfig {
    /// Enforce cross-field invariants. Returns `Ok(())` on success.
    pub fn validate(&self) -> Result<()> {
        if self.pool_tokens == 0 {
            bail!("--kvflash pool_tokens must be > 0");
        }
        if self.tau == 0 {
            bail!("--kvflash-tau must be > 0");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_lru() {
        assert_eq!(KvflashPolicy::default(), KvflashPolicy::Lru);
    }

    #[test]
    fn policy_display_fromstr_roundtrip() {
        for p in [KvflashPolicy::Lru, KvflashPolicy::Drafter] {
            let s = p.to_string();
            let back: KvflashPolicy = s.parse().expect("round-trip");
            assert_eq!(p, back, "Display/FromStr round-trip for {p:?}");
        }
    }

    #[test]
    fn policy_fromstr_case_insensitive_and_trimmed() {
        assert_eq!("LRU".parse::<KvflashPolicy>().unwrap(), KvflashPolicy::Lru);
        assert_eq!(
            "  Drafter  ".parse::<KvflashPolicy>().unwrap(),
            KvflashPolicy::Drafter
        );
    }

    #[test]
    fn policy_fromstr_rejects_unknown() {
        assert!("greedy".parse::<KvflashPolicy>().is_err());
        assert!("".parse::<KvflashPolicy>().is_err());
    }

    #[test]
    fn validate_rejects_zero_pool_tokens() {
        let cfg = KvflashConfig {
            pool_tokens: 0,
            tau: 64,
            policy: KvflashPolicy::Lru,
            protected_tail_blocks: 4,
            compact: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_tau() {
        let cfg = KvflashConfig {
            pool_tokens: 1024,
            tau: 0,
            policy: KvflashPolicy::Lru,
            protected_tail_blocks: 4,
            compact: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_nonzero_fields() {
        let cfg = KvflashConfig {
            pool_tokens: 1024,
            tau: 64,
            policy: KvflashPolicy::Drafter,
            protected_tail_blocks: 4,
            compact: false,
        };
        assert!(cfg.validate().is_ok());
    }
}
