// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    // `atlas_scale` mirrors spark-runtime's build.rs cfg: it marks the SCALE/
    // AMD (gfx1151) targets (`strix`, `strix-hip`). On these the GPU-visible
    // pool is a unified APU GTT (~60 GB) that cannot hold the FP8 source
    // checkpoint co-resident with its NVFP4 requant result, so the weight
    // loader frees each FP8 source tensor right after requant (see
    // `quantized_from_fp8`). NVIDIA targets leave the cfg unset and keep the
    // current resident-source behavior byte-for-byte.
    println!("cargo:rustc-check-cfg=cfg(atlas_scale)");
    if std::env::var("ATLAS_TARGET_HW")
        .as_deref()
        .map(|hw| hw.starts_with("strix"))
        .unwrap_or(false)
    {
        println!("cargo:rustc-cfg=atlas_scale");
    }
}
