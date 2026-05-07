// SPDX-License-Identifier: AGPL-3.0-only

//! Hardware-agnostic compute target abstraction.
//!
//! Defines the [`ComputeTarget`] trait: the contract that any GPU compilation
//! and runtime target must satisfy (NVIDIA/PTX, AMD/HSACO, Apple/Metal, etc.).
//!
//! Atlas is designed so that:
//! - **Build time**: kernel source files are compiled by a target-specific
//!   compiler into a target-specific binary format (PTX, SPIR-V, metallib).
//! - **Runtime**: the binary modules are loaded via `GpuBackend::kernel()`
//!   and executed via `GpuBackend::launch()`.
//!
//! This module covers the **build-time** contract. The runtime contract is
//! defined by `GpuBackend` in spark-runtime.
//!
//! # Extending to new hardware
//!
//! 1. Create a `HARDWARE.toml` in `kernels/<hw>/` with `vendor = "<vendor>"`.
//! 2. Implement [`ComputeTarget`] for the new vendor.
//! 3. Write kernel source files in the vendor's language (`.cu`, `.metal`, `.cl`).
//! 4. Implement `GpuBackend` in spark-runtime for the vendor's runtime API.

use std::path::{Path, PathBuf};

/// Vendor identifier parsed from `HARDWARE.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vendor {
    /// NVIDIA CUDA — compiles `.cu` → PTX via nvcc.
    Nvidia,
    /// AMD ROCm — compiles `.cu`/`.hip` → HSACO via hipcc (future).
    Amd,
    /// Apple Metal — compiles `.metal` → metallib via xcrun (future).
    Apple,
    /// Intel oneAPI — compiles `.cl`/`.sycl` → SPIR-V via icpx (future).
    Intel,
}

impl Vendor {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "nvidia" | "cuda" => Some(Self::Nvidia),
            "amd" | "rocm" | "hip" => Some(Self::Amd),
            "apple" | "metal" => Some(Self::Apple),
            "intel" | "oneapi" | "sycl" => Some(Self::Intel),
            _ => None,
        }
    }
}

impl std::fmt::Display for Vendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvidia => write!(f, "nvidia"),
            Self::Amd => write!(f, "amd"),
            Self::Apple => write!(f, "apple"),
            Self::Intel => write!(f, "intel"),
        }
    }
}

/// Build-time compilation target contract.
///
/// Each hardware vendor implements this trait to describe how kernel source
/// files are compiled into loadable binary modules. The build script
/// (`atlas-kernels/build.rs`) uses this trait to compile kernels without
/// knowing the specific compiler or binary format.
///
/// # Lifecycle
///
/// ```text
/// [build.rs]                                [runtime]
///
/// .cu / .metal / .cl                        GpuBackend::new(modules)
///        │                                         │
///        ▼                                         ▼
///  ComputeTarget::compile()              GpuBackend::kernel(name, fn)
///        │                                         │
///        ▼                                         ▼
///   .ptx / .metallib / .spv              GpuBackend::launch(handle, ...)
///        │
///        ▼
///  include_str!() / include_bytes!()
///        │
///        ▼
///  Embedded in binary as &str / &[u8]
/// ```
pub trait ComputeTarget {
    /// File extension for kernel source files (without dot).
    ///
    /// Examples: `"cu"` (NVIDIA), `"metal"` (Apple), `"cl"` (OpenCL).
    fn source_extension(&self) -> &str;

    /// File extension for compiled kernel modules (without dot).
    ///
    /// Examples: `"ptx"` (NVIDIA), `"metallib"` (Apple), `"spv"` (SPIR-V).
    fn output_extension(&self) -> &str;

    /// Whether compiled output is text (can use `include_str!`) or binary
    /// (must use `include_bytes!`).
    ///
    /// PTX is text; SPIR-V and metallib are binary.
    fn output_is_text(&self) -> bool;

    /// Find the compiler executable path.
    ///
    /// Returns `None` if the compiler is not installed.
    fn find_compiler(&self) -> Option<PathBuf>;

    /// Compile a single kernel source file to the target binary format.
    ///
    /// - `source`: path to the kernel source file (e.g., `rms_norm.cu`)
    /// - `output`: path to write the compiled output (e.g., `rms_norm.ptx`)
    /// - `arch`: target architecture string from HARDWARE.toml (e.g., `"sm_121f"`)
    /// - `extra_flags`: additional compiler flags from KERNEL.toml
    ///
    /// Returns `Ok(())` on success, `Err` with compiler output on failure.
    fn compile(
        &self,
        source: &Path,
        output: &Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String>;

    /// Hardware vendor for this target.
    fn vendor(&self) -> Vendor;
}

/// NVIDIA CUDA compilation target: `.cu` → PTX via `nvcc`.
///
/// This is the only concrete implementation today. Other vendors can be
/// added by implementing [`ComputeTarget`] and wiring into `build.rs`.
pub struct NvidiaTarget {
    nvcc_path: PathBuf,
}

impl NvidiaTarget {
    /// Create a new NVIDIA target, locating nvcc from CUDA_HOME or PATH.
    pub fn new() -> Option<Self> {
        let nvcc = find_nvcc()?;
        Some(Self { nvcc_path: nvcc })
    }

    /// Create with an explicit nvcc path.
    pub fn with_compiler(nvcc_path: PathBuf) -> Self {
        Self { nvcc_path }
    }
}

impl ComputeTarget for NvidiaTarget {
    fn source_extension(&self) -> &str {
        "cu"
    }

    fn output_extension(&self) -> &str {
        "ptx"
    }

    fn output_is_text(&self) -> bool {
        true // PTX is human-readable text
    }

    fn find_compiler(&self) -> Option<PathBuf> {
        Some(self.nvcc_path.clone())
    }

    fn compile(
        &self,
        source: &Path,
        output: &Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        let arch_flag = format!("-arch={arch}");
        let mut args = vec!["--ptx".to_string(), arch_flag, "-O3".to_string()];
        args.extend(extra_flags.iter().cloned());
        args.push(source.to_str().unwrap().to_string());
        args.push("-o".to_string());
        args.push(output.to_str().unwrap().to_string());

        let result = std::process::Command::new(&self.nvcc_path)
            .args(&args)
            .output()
            .map_err(|e| format!("Failed to run nvcc: {e}"))?;

        if result.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&result.stderr);
            Err(format!(
                "nvcc --ptx failed for {}: {}",
                source.display(),
                stderr
            ))
        }
    }

    fn vendor(&self) -> Vendor {
        Vendor::Nvidia
    }
}

/// Locate nvcc from CUDA_HOME, CUDA_PATH, or standard install locations.
fn find_nvcc() -> Option<PathBuf> {
    // Check CUDA_HOME / CUDA_PATH environment variables
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(dir) = std::env::var(var) {
            let nvcc = PathBuf::from(dir).join("bin/nvcc");
            if nvcc.exists() {
                return Some(nvcc);
            }
        }
    }
    // Check standard install locations
    for path in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda-12.0/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ] {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    // Check PATH
    which_in_path("nvcc")
}

fn which_in_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(name))
            .find(|p| p.exists())
    })
}

/// Resolve the appropriate [`ComputeTarget`] from a HARDWARE.toml vendor field.
///
/// Falls back to [`NvidiaTarget`] if no vendor is specified (backward compat).
pub fn target_for_vendor(vendor: Option<&str>) -> Box<dyn ComputeTarget> {
    match vendor.and_then(Vendor::parse) {
        Some(Vendor::Nvidia) | None => {
            Box::new(NvidiaTarget::new().expect("nvcc not found — install CUDA toolkit"))
        }
        Some(Vendor::Apple) => Box::new(
            AppleTarget::new()
                .expect("xcrun not found — install Xcode Command Line Tools (xcode-select --install)"),
        ),
        Some(v) => panic!(
            "Compute target '{v}' is not yet implemented. \
             Supported: 'nvidia', 'apple'."
        ),
    }
}

/// Apple Metal compilation target: `.metal` → `metallib` via the
/// two-step `xcrun -sdk macosx metal -c {src} -o {tmp}.air` then
/// `xcrun -sdk macosx metallib {tmp}.air -o {dst}.metallib` pipeline.
pub struct AppleTarget {
    xcrun_path: PathBuf,
}

impl AppleTarget {
    /// Locate `xcrun`. Returns `None` when Xcode Command Line Tools are
    /// missing — the caller decides whether that's fatal.
    pub fn new() -> Option<Self> {
        find_xcrun().map(|xcrun_path| Self { xcrun_path })
    }

    /// Construct with an explicit `xcrun` path (escape hatch for unusual
    /// toolchain layouts).
    pub fn with_compiler(xcrun_path: PathBuf) -> Self {
        Self { xcrun_path }
    }
}

impl ComputeTarget for AppleTarget {
    fn source_extension(&self) -> &str {
        "metal"
    }

    fn output_extension(&self) -> &str {
        "metallib"
    }

    fn output_is_text(&self) -> bool {
        false
    }

    fn find_compiler(&self) -> Option<PathBuf> {
        Some(self.xcrun_path.clone())
    }

    fn compile(
        &self,
        source: &Path,
        output: &Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        // Stage 1: source.metal → tmp.air
        let air_path = output.with_extension("air");
        let mut metal_args: Vec<String> =
            vec!["-sdk".into(), "macosx".into(), "metal".into()];
        if !arch.is_empty() {
            metal_args.push(format!("-std={arch}"));
        }
        metal_args.push("-c".into());
        metal_args.push("-O3".into());
        metal_args.extend(extra_flags.iter().cloned());
        metal_args.push(source.to_str().unwrap().into());
        metal_args.push("-o".into());
        metal_args.push(air_path.to_str().unwrap().into());

        let result = std::process::Command::new(&self.xcrun_path)
            .args(&metal_args)
            .output()
            .map_err(|e| format!("Failed to run xcrun metal: {e}"))?;
        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(format!(
                "xcrun metal failed for {}: {}",
                source.display(),
                stderr
            ));
        }

        // Stage 2: tmp.air → output.metallib
        let result = std::process::Command::new(&self.xcrun_path)
            .args([
                "-sdk",
                "macosx",
                "metallib",
                air_path.to_str().unwrap(),
                "-o",
                output.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("Failed to run xcrun metallib: {e}"))?;
        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(format!(
                "xcrun metallib failed for {}: {}",
                source.display(),
                stderr
            ));
        }
        Ok(())
    }

    fn vendor(&self) -> Vendor {
        Vendor::Apple
    }
}

/// Locate xcrun via the canonical path or PATH fallback.
fn find_xcrun() -> Option<PathBuf> {
    let canonical = PathBuf::from("/usr/bin/xcrun");
    if canonical.exists() {
        return Some(canonical);
    }
    which_in_path("xcrun")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_from_str() {
        assert_eq!(Vendor::parse("nvidia"), Some(Vendor::Nvidia));
        assert_eq!(Vendor::parse("CUDA"), Some(Vendor::Nvidia));
        assert_eq!(Vendor::parse("amd"), Some(Vendor::Amd));
        assert_eq!(Vendor::parse("rocm"), Some(Vendor::Amd));
        assert_eq!(Vendor::parse("apple"), Some(Vendor::Apple));
        assert_eq!(Vendor::parse("metal"), Some(Vendor::Apple));
        assert_eq!(Vendor::parse("intel"), Some(Vendor::Intel));
        assert_eq!(Vendor::parse("unknown"), None);
    }

    #[test]
    fn nvidia_target_extensions() {
        if let Some(target) = NvidiaTarget::new() {
            assert_eq!(target.source_extension(), "cu");
            assert_eq!(target.output_extension(), "ptx");
            assert!(target.output_is_text());
            assert_eq!(target.vendor(), Vendor::Nvidia);
        }
    }
}
