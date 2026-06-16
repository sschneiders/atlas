// SPDX-License-Identifier: AGPL-3.0-only

//! Model factory call, prefix-cache + high-speed-swap setup, and the
//! rank > 0 EP worker entry point.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn build_prefix_cache(
    args: &cli::ServeArgs,
) -> Box<dyn spark_runtime::prefix_cache::PrefixCache> {
    if args.enable_prefix_caching {
        if args.high_speed_swap {
            tracing::info!(
                "Prefix caching: ENABLED (radix tree, with --high-speed-swap disk-side refcounts)"
            );
        } else {
            tracing::info!("Prefix caching: ENABLED (radix tree)");
        }
        Box::new(spark_runtime::radix_tree::RadixTree::new())
    } else {
        tracing::info!("Prefix caching: disabled");
        Box::new(spark_runtime::prefix_cache::NoPrefixCaching)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_model(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    store: &spark_runtime::weights::WeightStore,
    gpu: Box<dyn spark_runtime::gpu::GpuBackend>,
    max_batch_tokens: usize,
    kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    inference_reserve: usize,
    layer_dtypes: Vec<spark_runtime::kv_cache::KvCacheDtype>,
    hss_cache_blocks_per_seq: Option<u32>,
    prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
    comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    dflash_args: Option<spark_model::factory::DflashBuildArgs<'_>>,
) -> Result<Box<dyn spark_model::traits::Model>> {
    let mtp_quant: spark_model::layers::MtpQuantization = args
        .mtp_quantization
        .parse()
        .context("Invalid --mtp-quantization value")?;
    spark_model::factory::build_model(
        config.clone(),
        store,
        gpu,
        max_batch_tokens,
        args.block_size,
        args.max_seq_len,
        args.max_batch_size,
        mtp_quant,
        args.speculative || args.dflash,
        prefix_cache,
        args.mtp_vocab,
        comm,
        args.self_speculative || args.ngram_speculative,
        if args.dflash {
            args.dflash_gamma.saturating_sub(1).max(1)
        } else {
            args.num_drafts
        },
        kv_dtype,
        inference_reserve,
        args.gpu_memory_utilization,
        args.ssm_cache_slots,
        layer_dtypes,
        args.ssm_checkpoint_interval,
        hss_cache_blocks_per_seq,
        dflash_args,
    )
    .context("Failed to build model")
}

pub(crate) fn build_high_speed_swap_config(
    args: &cli::ServeArgs,
) -> Result<Option<spark_storage::HighSpeedSwapConfig>> {
    if !args.high_speed_swap {
        return Ok(None);
    }
    let dir = args
        .high_speed_swap_dir
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("/var/tmp/atlas-hsw"));
    let bytes_gb = args.high_speed_swap_gb.unwrap_or(64);
    let resident_blocks = args.high_speed_swap_resident_blocks.unwrap_or(8192);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        anyhow::bail!(
            "--high-speed-swap: failed to create dir {}: {e}",
            dir.display()
        );
    }
    let cfg = spark_storage::HighSpeedSwapConfig {
        dir,
        bytes: bytes_gb * (1 << 30),
        resident_blocks,
        rank: args.high_speed_swap_rank,
        qd: args.high_speed_swap_qd,
        graph: args.high_speed_swap_graph.unwrap_or(true),
        projection_seed: 0xCAFE_F00D,
    };
    cfg.validate()?;
    Ok(Some(cfg))
}

pub(crate) fn validate_head_high_speed_swap(
    args: &cli::ServeArgs,
    early_high_speed_swap_cfg: &Option<spark_storage::HighSpeedSwapConfig>,
    swap_space_gb: usize,
) -> Result<Option<spark_storage::HighSpeedSwapConfig>> {
    let Some(cfg) = early_high_speed_swap_cfg.as_ref() else {
        return Ok(None);
    };
    if swap_space_gb > 0
        && cfg.dir.canonicalize().ok().as_deref()
            == std::path::Path::new("/tmp/atlas-swap")
                .canonicalize()
                .ok()
                .as_deref()
    {
        let _ = args;
        anyhow::bail!(
            "--high-speed-swap-dir must not be /tmp/atlas-swap (already used \
             by --swap-space-gb sequence-level fallback)"
        );
    }
    tracing::info!(
        "--high-speed-swap enabled: dir={}, budget={} GiB, scratch={} blocks, \
         rank={}, qd={}, graph={}",
        cfg.dir.display(),
        cfg.bytes / (1 << 30),
        cfg.resident_blocks,
        cfg.rank,
        cfg.qd,
        cfg.graph,
    );
    Ok(Some(cfg.clone()))
}

/// Pure auto-sizing arithmetic, factored out so it is unit-testable without a
/// CUDA context. Returns the resident pool size in TOKENS (NOT yet rounded to a
/// multiple of `block_size` — the caller rounds).
///
/// When `bytes_per_kv_token == 0` the per-token KV cost is unknown (the builder
/// does not yet receive model dims), so VRAM-proportional sizing is impossible
/// and we fall back to the configured caps (`min(max_pool, max_ctx)`, floored at
/// a protected minimum). A future PR that passes the real `bytes_per_kv_token`
/// gets true `free_vram / bytes_per_kv_token` sizing for free.
pub(crate) fn resolve_auto_pool(
    free_vram_bytes: usize,
    bytes_per_kv_token: usize,
    max_pool: usize,
    max_ctx: usize,
) -> usize {
    const PROTECTED_MIN: usize = 512;
    if bytes_per_kv_token == 0 {
        return max_pool.min(max_ctx).max(PROTECTED_MIN);
    }
    let computed = free_vram_bytes / bytes_per_kv_token;
    computed.min(max_pool).min(max_ctx).max(PROTECTED_MIN)
}

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
}

/// Parse `ATLAS_*` truthy env vars: `1`/`true`/`on`/`yes` (case-insensitive)
/// → `true`; `0`/`false`/`off`/`no` → `false`; anything else → `None` (kept
/// absent so the CLI flag default wins). Manual dual-check idiom (no
/// `#[arg(env=...)]` in this codebase — mirrors `env_u32` / `env_usize`).
fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => Some(true),
            "0" | "false" | "off" | "no" => Some(false),
            _ => None,
        })
}

/// Build the resolved KVFlash config from CLI flags + env fallback. Returns
/// `Ok(None)` when KVFlash is not requested (neither `--kvflash` nor
/// `ATLAS_KVFLASH` set). Mirrors `build_high_speed_swap_config`'s shape: takes
/// only `&args`, resolves "auto"/env into a concrete `pool_tokens`, rounds up to
/// a multiple of `block_size`, and validates.
pub(crate) fn build_kvflash_config(
    args: &cli::ServeArgs,
) -> Result<Option<spark_runtime::KvflashConfig>> {
    // Source the pool spec: explicit --kvflash flag, else ATLAS_KVFLASH env
    // (manual dual-check idiom — no #[arg(env=...)] in this codebase).
    let pool_spec = match args.kvflash.as_deref() {
        Some(s) => Some(s.to_string()),
        None => std::env::var("ATLAS_KVFLASH")
            .ok()
            .filter(|s| !s.is_empty()),
    };
    let Some(spec) = pool_spec else {
        return Ok(None);
    };

    // tau / max_pool / policy: flag defaults, overridable by their env vars.
    let tau = env_u32("ATLAS_KVFLASH_TAU").unwrap_or(args.kvflash_tau);
    let max_pool = env_usize("ATLAS_KVFLASH_MAX_POOL").unwrap_or(args.kvflash_max_pool);
    let policy = match std::env::var("ATLAS_KVFLASH_POLICY")
        .ok()
        .filter(|s| !s.is_empty())
    {
        Some(s) => s.parse::<spark_runtime::KvflashPolicy>()?,
        None => args.kvflash_policy.into(),
    };
    // PR8 block-table compaction: explicit --kvflash-compact flag, else env
    // fallback (manual dual-check idiom, same as tau/max_pool above).
    let compact = env_bool("ATLAS_KVFLASH_COMPACT").unwrap_or(args.kvflash_compact);

    let pool_tokens = resolve_pool_tokens(&spec, max_pool, args.max_seq_len, args.block_size)?;
    let cfg = spark_runtime::KvflashConfig {
        pool_tokens,
        tau,
        policy,
        protected_tail_blocks: args.kvflash_protected_tail_blocks,
        compact,
    };
    cfg.validate()?;
    Ok(Some(cfg))
}

/// Resolve the pool token count from the spec string (`"auto"` or a numeric
/// token count), then round up to a multiple of `block_size`.
fn resolve_pool_tokens(
    spec: &str,
    max_pool: usize,
    max_ctx: usize,
    block_size: usize,
) -> Result<usize> {
    let raw = match spec.trim().to_ascii_lowercase().as_str() {
        "auto" => {
            let (free_vram, _total) = spark_storage::cuda_min::mem_info()
                .context("--kvflash auto requires an initialized GPU context")?;
            // bytes_per_kv_token is unknown at this build site (model dims are
            // not passed to the builder yet); resolve_auto_pool falls back to
            // the configured caps. A future PR passes the real per-token cost.
            resolve_auto_pool(free_vram, 0, max_pool, max_ctx)
        }
        other => other
            .parse::<usize>()
            .with_context(|| format!("--kvflash expected a token count or 'auto', got '{other}'"))?
            .min(max_ctx),
    };
    // Round up to a multiple of block_size (KV cache block granularity), keeping
    // at least one block.
    Ok(raw.div_ceil(block_size).max(1) * block_size)
}

/// Validate + log the resolved KVFlash config. Mirrors
/// `validate_head_high_speed_swap`: logs an info summary line and returns the
/// config unchanged. `Ok(None)` when KVFlash is disabled.
pub(crate) fn validate_kvflash(
    args: &cli::ServeArgs,
    cfg: &Option<spark_runtime::KvflashConfig>,
) -> Result<Option<spark_runtime::KvflashConfig>> {
    let _ = args;
    let Some(cfg) = cfg.as_ref() else {
        return Ok(None);
    };
    tracing::info!(
        "--kvflash enabled: pool_tokens={}, tau={}, policy={}, protected_tail_blocks={}",
        cfg.pool_tokens,
        cfg.tau,
        cfg.policy,
        cfg.protected_tail_blocks,
    );
    Ok(Some(cfg.clone()))
}

pub(crate) fn maybe_run_ep_worker(
    args: &cli::ServeArgs,
    model: &mut Option<Box<dyn spark_model::traits::Model>>,
    early_high_speed_swap_cfg: &Option<spark_storage::HighSpeedSwapConfig>,
) -> Result<bool> {
    if args.rank == 0 {
        return Ok(false);
    }
    let rank = args.rank;
    let model_owned = model.take().expect("EP worker requires owned model");
    let model_has_proposer = model_owned.has_proposer();
    if !args.speculative && !args.self_speculative && !args.ngram_speculative && model_has_proposer
    {
        let override_set = matches!(
            std::env::var("ATLAS_ALLOW_SPEC_MISMATCH").as_deref(),
            Ok("1") | Ok("true")
        );
        if !override_set {
            anyhow::bail!(
                "EP worker (rank {rank}) started WITHOUT any --speculative flag, \
                 but this checkpoint has MTP weights and the head will likely use them. \
                 Mirror the head's --speculative / --mtp-quantization / --num-drafts \
                 flags here, or set ATLAS_ALLOW_SPEC_MISMATCH=1 if the head is also \
                 non-speculative."
            );
        }
        tracing::warn!(
            "EP worker (rank {rank}) running WITHOUT speculative flags but \
             ATLAS_ALLOW_SPEC_MISMATCH=1 — head must NOT issue MTP commands."
        );
    } else if !model_has_proposer
        && !args.speculative
        && !args.self_speculative
        && !args.ngram_speculative
    {
        tracing::info!(
            "EP worker (rank {rank}): checkpoint has no MTP weights; \
             spec-mismatch guard auto-skipped (head can't use MTP either)."
        );
    }
    let worker_hss_cfg = early_high_speed_swap_cfg.clone();
    // Copy primitives out of `args` so the worker thread (which is
    // `'static`) doesn't capture the function-scoped `&ServeArgs` ref.
    let max_batch_size = args.max_batch_size;
    let handle = std::thread::spawn(move || {
        model_owned
            .bind_gpu_to_thread()
            .expect("Failed to bind GPU to EP worker thread");
        if let Some(cfg) = worker_hss_cfg {
            match model_owned.high_speed_swap_dims() {
                Some(dims) => {
                    if let Err(e) = spark_storage::install_local(rank as u64, cfg, dims) {
                        tracing::error!(
                            "EP worker (rank {rank}): --high-speed-swap install failed: {e:#}"
                        );
                    } else {
                        tracing::info!(
                            "EP worker (rank {rank}): --high-speed-swap orchestrator installed"
                        );
                    }
                }
                None => {
                    tracing::warn!(
                        "EP worker (rank {rank}): --high-speed-swap requested but model \
                         does not expose high_speed_swap_dims; skipping install"
                    );
                }
            }
        }
        // Slots vec sized to match the head's scheduler `max_batch_size`.
        // Pre-allocate every slot. The head only emits `0xFFFFFFF1`
        // (free+realloc) on lifecycle events — sequence finish/error —
        // not on first use, so a fresh `prefill_a_step` for slot N
        // arrives as `0xFFFFFFF0` with no prior alloc broadcast. Under v1
        // (max_batch_size=1) this is just slot 0, matching the legacy
        // behavior. Under v2 (max_batch_size>1) every slot must be
        // populated up front for the same reason.
        //
        // Both ranks' SSM pools start with the same free-list ordering
        // (see ssm_pool.rs: `(0..max_slots).rev().collect()` + `pop()`),
        // so pre-allocating in `0..max_batch_size` order on the worker
        // means `slots[i].slot_idx == i` — matching the slot ids the
        // head's `alloc_sequence` returns for its Nth claim.
        let mut slots: Vec<Option<spark_model::traits::SequenceState>> =
            (0..max_batch_size).map(|_| None).collect();
        for slot in slots.iter_mut() {
            *slot = Some(
                model_owned
                    .alloc_sequence()
                    .expect("Failed to allocate EP worker sequence"),
            );
        }
        tracing::info!(
            "EP worker ready (rank {rank}, {} slots), waiting for commands",
            slots.len()
        );
        loop {
            match model_owned.ep_worker_step(&mut slots) {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => {
                    tracing::error!("EP worker error: {e:#}");
                    break;
                }
            }
        }
        for slot in slots.iter_mut() {
            if let Some(seq) = slot.as_mut() {
                let _ = model_owned.free_sequence(seq);
            }
        }
        tracing::info!("EP worker stopped (rank {rank})");
    });
    handle.join().expect("EP worker thread panicked");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_pool_vram_proportional_when_cost_known() {
        // 2 GiB free, 256 bytes/token KV → 8192 tokens before clamps.
        let pool = resolve_auto_pool(2 * (1 << 30), 256, 16384, 32768);
        assert_eq!(pool, 8192);
    }

    #[test]
    fn auto_pool_clamps_to_max_pool_and_max_ctx() {
        // huge free VRAM, small caps → clamped to the tighter cap.
        assert_eq!(resolve_auto_pool(usize::MAX, 256, 4096, 32768), 4096);
        assert_eq!(resolve_auto_pool(usize::MAX, 32768, 16384, 4096), 4096);
    }

    #[test]
    fn auto_pool_floors_at_protected_min() {
        // tiny free VRAM → protected minimum of 512.
        let pool = resolve_auto_pool(1000, 256, 16384, 32768);
        assert_eq!(pool, 512);
    }

    #[test]
    fn auto_pool_unknown_cost_falls_back_to_caps() {
        // bytes_per_kv_token == 0 → can't size from VRAM; use caps.
        assert_eq!(resolve_auto_pool(1 << 30, 0, 8192, 32768), 8192);
        // floored at protected min when caps are tiny.
        assert_eq!(resolve_auto_pool(1 << 30, 0, 0, 0), 512);
    }

    #[test]
    fn env_helpers_parse_valid_and_reject_invalid() {
        // std::env::set_var / remove_var are unsafe as of edition 2024
        // (not signal-/thread-safe), so the mutations live in an unsafe block.
        unsafe {
            std::env::set_var("ATLAS_TEST_KVFLASH_U32", "128");
            assert_eq!(env_u32("ATLAS_TEST_KVFLASH_U32"), Some(128));

            std::env::set_var("ATLAS_TEST_KVFLASH_U32", "not-a-num");
            assert_eq!(env_u32("ATLAS_TEST_KVFLASH_U32"), None);

            std::env::set_var("ATLAS_TEST_KVFLASH_U32", "");
            assert_eq!(env_u32("ATLAS_TEST_KVFLASH_U32"), None);

            std::env::remove_var("ATLAS_TEST_KVFLASH_U32");
            assert_eq!(env_u32("ATLAS_TEST_KVFLASH_U32"), None);

            std::env::set_var("ATLAS_TEST_KVFLASH_USIZE", "4096");
            assert_eq!(env_usize("ATLAS_TEST_KVFLASH_USIZE"), Some(4096));
            std::env::remove_var("ATLAS_TEST_KVFLASH_USIZE");
        }
    }
}
