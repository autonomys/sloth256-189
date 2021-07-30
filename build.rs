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
    #[cfg(target_env = "msvc")]
    if !env::var("CC").is_ok() && which::which("clang-cl").is_ok() {
        env::set_var("CC", "clang-cl");
    }
    let mut cc = cc::Build::new();
    let mut files = vec![PathBuf::from("src/sloth256_189.c")];

    // account for cross-compilation
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    match cfg!(feature = "no-asm") {
        false => {
            if target_arch.eq("x86_64") {
                assembly(&mut files);
            }
        }
        true => {
            println!("Compiling without assembly module");
            cc.define("__SLOTH256_189_NO_ASM__", None);
        }
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
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };
    if nvcc.is_ok() {
        cc::Build::new()
            .cuda(true)
            .cudart("static")
            .file("src/sloth256_189.cu")
            .compile("libsloth256_189_cuda.a");

        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }
}
