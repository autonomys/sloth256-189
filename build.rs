#![allow(unused_imports)]

extern crate cc;

use std::env;
use std::path::{Path, PathBuf};
use which::which;

#[cfg(target_env = "msvc")]
fn assembly(files: &mut Vec<PathBuf>) {
    files.push(PathBuf::from("src/win64/mod256-189-x86_64.asm"))
}

#[cfg(not(target_env = "msvc"))]
fn assembly(files: &mut Vec<PathBuf>) {
    files.push(PathBuf::from("src/assembly.S"))
}

fn main() {
    // Set CC environment variable to choose alternative C compiler.
    // Optimization level depends on whether or not --release is passed
    // or implied.
    let mut cc = cc::Build::new();
    let mut files = Vec::new();

    files.push(PathBuf::from("src/sloth256_189.c"));

    // account for cross-compilation
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if target_arch.eq("x86_64") {
        assembly(&mut files);
    }
    match (cfg!(feature = "portable"), cfg!(feature = "force-adx")) {
        (true, false) => {
            println!("Compiling in portable mode without ISA extensions");
            cc.define("__SLOTH_PORTABLE__", None);
        }
        (false, true) => {
            if target_arch.eq("x86_64") {
                println!("Enabling ADX support via `force-adx` feature");
                cc.define("__ADX__", None);
            } else {
                println!("`force-adx` is ignored for non-x86_64 targets");
            }
        }
        (false, false) => {
            #[cfg(target_arch = "x86_64")]
            if target_arch.eq("x86_64") && std::is_x86_feature_detected!("adx")
            {
                println!("Enabling ADX because it was detected on the host");
                cc.define("__ADX__", None);
            }
        }
        (true, true) => panic!(
            "Cannot compile with both `portable` and `force-adx` features"
        ),
    }
    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin-memcpy")
        .flag_if_supported("-Wno-unused-command-line-argument");
    if !cfg!(debug_assertions) {
        cc.opt_level(3);
    }
    cc.files(&files).compile("libsloth256_189.a");

    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    let nvcc = match env::var("NVCC") {
        Ok(var) => Ok(PathBuf::from(var)),
        Err(_) => which::which("nvcc"),
    };
    match nvcc {
        Err(_) => (),
        Ok(p) => {
            let nvhome = p.parent().unwrap().parent().unwrap();

            cc::Build::new()
                .cuda(true)
                .files(vec![PathBuf::from("src/sloth256_189.cu")])
                .compile("libsloth256_189_cuda.a");

            if nvhome.is_dir() {
                let str = nvhome.to_str().unwrap();
                #[cfg(target_os = "linux")]
                println!(
                    "cargo:rustc-link-search=native={}/targets/{}-linux/lib",
                    str,
                    target_arch.as_str()
                );
                #[cfg(target_os = "windows")]
                println!("cargo:rustc-link-search=native={}\\lib\\x64", str);
            }
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-cfg=feature=\"cuda\"");
        }
    }
}
