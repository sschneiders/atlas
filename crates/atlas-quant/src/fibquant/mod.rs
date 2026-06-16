// SPDX-License-Identifier: AGPL-3.0-only

//! FibQuant KV-cache compression — host-side fidelity implementation.
//!
//! Universal fixed-rate random-access KV compressor (Lee & Kim, arXiv:2605.11478):
//! normalize each vector, apply a shared Haar-random orthogonal rotation `Π`,
//! split into `k`-blocks, and store the nearest shared codebook index per block.
//! The codebook is built once from the spherical-Beta source `f_{d,k}` induced
//! by the normalize–rotate interface (Beta-quantile radii × Fibonacci /
//! Roberts–Kronecker directions, polished by multi-restart Lloyd–Max) — no
//! calibration. See `docs/design/fibquant-kv-compression.md`.
//!
//! This crate is the pure-Rust reference used to (a) reproduce the paper's
//! attention-cosine numbers before any CUDA work, and (b) become the SSOT for
//! the eventual `KvCacheDtype::FibQuant` + `.cu` kernel. It runs on CPU only;
//! the decode path never blocks on I/O (the codebook is a precomputed constant).

pub mod codebook;
pub mod codec;
pub mod directions;
pub mod metric;
pub mod rotate;
pub mod special;

pub use codebook::Codebook;
pub use codec::{EncodedVec, FibQuantCodec};
pub use metric::{attention_output_cosine, mean_vector_cosine};
pub use rotate::Rotation;
