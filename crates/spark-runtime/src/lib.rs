// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

pub mod buffers;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(unix)]
pub mod fast_weights;
pub mod gpu;
pub mod kernel_args;
pub mod kernel_audit;
pub mod kv_cache;
pub mod kv_dequant;
pub mod kv_spill;
pub mod kvflash_compact;
pub mod kvflash_config;
pub mod kvflash_pager;
pub mod kvflash_residency;
pub mod kvflash_scorer;
pub mod kvflash_verify;

// Re-export the KVFlash config types at the crate root so spark-server can
// reference them as `spark_runtime::KvflashConfig` / `KvflashPolicy` (mirrors
// how other top-level runtime types are surfaced).
pub use kvflash_config::{KvflashConfig, KvflashPolicy};
#[cfg(feature = "metal")]
pub mod metal_backend;
pub mod prefix_cache;
pub mod radix_tree;
pub mod sampler;
pub mod weights;
