// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

pub mod capabilities;
pub mod compute;
pub mod config;
pub mod dtype;
pub mod error;
pub mod target;
pub mod tensor;

// CUDA-only modules: rely on `cudarc` and the NVIDIA driver. Gated so the
// crate compiles on hosts without a CUDA toolchain (e.g. Apple Silicon).
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod kernel;
#[cfg(feature = "cuda")]
pub mod registry;
#[cfg(feature = "cuda")]
pub mod stream;
