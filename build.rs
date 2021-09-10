use std::env;
use std::path::PathBuf;

#[cfg(target_env = "msvc")]
fn get_assembly_file() -> PathBuf {
    PathBuf::from("src/win64/mod256-189-x86_64.asm")
}

#[cfg(not(target_env = "msvc"))]
fn get_assembly_file() -> PathBuf {
    PathBuf::from("src/assembly.S")
}

fn main() {
    // Set CC environment variable to choose alternative C compiler.
    // Optimization level depends on whether or not --release is passed
    // or implied.
    #[cfg(target_env = "msvc")]
    if env::var("CC").is_err() && which::which("clang-cl").is_ok() {
        env::set_var("CC", "clang-cl");
    }
    let mut cc = cc::Build::new();
    let mut files = vec![PathBuf::from("src/sloth256_189.c")];

    // account for cross-compilation
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    if cfg!(feature = "no-asm") {
        println!("Compiling without assembly module");
        cc.define("__SLOTH256_189_NO_ASM__", None);
    } else if target_arch.eq("x86_64") {
        files.push(get_assembly_file());
    }

    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin-memcpy")
        .flag_if_supported("-Wno-unused-command-line-argument");

    if !cfg!(debug_assertions) {
        cc.opt_level(3);
    }

    cc.files(&files).compile("libsloth256_189.a");

    if target_os == "windows" && !cfg!(target_env = "msvc") {
        return;
    }
    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    if which::which(env::var("NVCC").as_deref().unwrap_or("nvcc")).is_ok() {
        cc::Build::new()
            .cuda(true)
            .cudart("static")
            .file("src/gpu/ptx.cu")
            .compile("libsloth256_189_cuda.a");

        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }
}
