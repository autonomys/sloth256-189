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
    match nvcc {
        Err(_) => (),
        Ok(nvcc) => {
            cc::Build::new()
                .cuda(true)
                .files(vec![PathBuf::from("src/sloth256_189.cu")])
                .compile("libsloth256_189_cuda.a");

            println!("cargo:rustc-cfg=feature=\"cuda\"");

            // Always add -lcudart and try to figure out -L search path.
            // If the latter fails, it's on user to specify one by setting
            // RUSTFLAGS environment variable.
            #[allow(unused_mut, unused_assignments)]
            let mut libtst = false;
            let mut libdir = nvcc.to_path_buf();
            libdir.pop();
            libdir.pop();
            #[cfg(target_os = "linux")]
            {
                libdir.push("targets");
                libdir.push(target_arch.to_owned() + "-linux");
                libdir.push("lib");
                libtst = true;
            }
            #[cfg(target_env = "msvc")]
            match target_arch.as_str() {
                "x86_64" => {
                    libdir.push("lib");
                    libdir.push("x64");
                    libtst = true;
                }
                "x86" => {
                    libdir.push("lib");
                    libdir.push("Win32");
                    libtst = true;
                }
                _ => libtst = false,
            }
            if libtst && libdir.is_dir() {
                println!(
                    "cargo:rustc-link-search=native={}",
                    libdir.to_str().unwrap()
                );
            }
            println!("cargo:rustc-link-lib=cudart_static");
        }
    }
}
