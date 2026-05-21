// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    // Register the `atlas_scale` cfg so `#[cfg(atlas_scale)]` does not
    // trip the `unexpected_cfgs` lint (this crate is `#![deny(warnings)]`).
    println!("cargo:rustc-check-cfg=cfg(atlas_scale)");

    if matches!(
        std::env::var("ATLAS_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    ) {
        return;
    }

    // `atlas_scale` selects SCALE/AMD (gfx1151) codepaths over NVIDIA
    // ones — e.g. SCALE's libcuda exports `cuGraphInstantiate` with the
    // 3-arg flags ABI and no `cuGraphInstantiateWithFlags` alias. Driven
    // by the same `ATLAS_TARGET_HW` signal the atlas-kernels build uses.
    if std::env::var("ATLAS_TARGET_HW").as_deref() == Ok("strix") {
        println!("cargo:rustc-cfg=atlas_scale");
    }

    // libcuda is only needed when the cuda feature is on (i.e. when
    // AtlasCudaBackend is compiled in). The metal feature build on
    // Apple Silicon must not request -lcuda.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    // Link libcuda for AtlasCudaBackend's raw CUDA driver API calls.
    // On NVIDIA the driver stub resolves to the installed driver at
    // runtime; on SCALE it resolves to SCALE's libcuda.
    println!("cargo:rustc-link-lib=dylib=cuda");

    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        // NVIDIA CUDA toolkits place libs under lib64/; SCALE's
        // gfx1151 target uses lib/. Search both so either resolves.
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
        println!("cargo:rustc-link-search=native={cuda_path}/lib");
        println!("cargo:rustc-link-search=native={cuda_path}/lib/stubs");
    }
    // Standard CUDA locations
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
}
