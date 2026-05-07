// SPDX-License-Identifier: AGPL-3.0-only

//! Apple Metal GPU backend.
//!
//! Production implementation lives in subsequent phases — this module exists
//! at Phase 1 only to reserve the namespace and prove `--features metal`
//! compiles cleanly on Apple Silicon without the CUDA toolchain.
