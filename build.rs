use std::env;
use std::path::PathBuf;

#[cfg(target_env = "msvc")]
fn get_assembly_file() -> PathBuf {
    PathBuf::from("src/cpu/win64/mod256-189-x86_64.asm")
}

#[cfg(not(target_env = "msvc"))]
fn get_assembly_file() -> PathBuf {
    PathBuf::from("src/cpu/assembly.S")
}

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        // Skip everything when building docs on docs.rs
        return;
    }

    // Set CC environment variable to choose alternative C compiler.
    // Optimization level depends on whether or not --release is passed
    // or implied.
    #[cfg(target_env = "msvc")]
    if env::var("CC").is_err() && which::which("clang-cl").is_ok() {
        env::set_var("CC", "clang-cl");
    }
    let mut cc = cc::Build::new();
    cc.extra_warnings(true);
    cc.warnings_into_errors(true);
    let mut files = vec![PathBuf::from("src/cpu/sloth256_189.c")];

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

    cc.files(&files).compile("sloth256_189");

    if target_os == "windows" && !cfg!(target_env = "msvc") {
        return;
    }

    if cfg!(feature = "cuda") {
        cc::Build::new()
            .cuda(true)
            .cudart("static")
            .extra_warnings(true)
            .warnings_into_errors(true)
            .file("src/cuda/ptx.cu")
            .compile("sloth256_189_cuda");
    }

    if cfg!(feature = "opencl") {
        env::var("DEP_OPENMP_FLAG")
            .unwrap()
            .split(' ')
            .for_each(|f| {
                cc.flag(f);
            });

        println!("cargo:rustc-link-lib=OpenCL");
        println!("cargo:rustc-link-search=/opt/amdgpu-pro/lib64/");

        let mut cuda_include: String = "".to_string();
        if env::var("CUDA_PATH").is_ok() {
            let cuda_path = env::var("CUDA_PATH").unwrap();
            println!("cargo:rustc-link-search={}/lib/x64", cuda_path);

            cuda_include = cuda_path + "/include";
        }

        cc::Build::new()
            .cpp(true)
            .flag_if_supported("-pthread")
            .flag_if_supported("-fopenmp")
            .flag_if_supported("/openmp")
            .flag_if_supported("-std:c++17")
            .flag_if_supported("/EHsc")
            .flag_if_supported("-std=c++17")
            .include("/opt/amdgpu-pro/include/")
            .include("/opt/intel/inteloneapi/compiler/latest/linux/include/sycl/")
            .include(cuda_include)
            .file("src/opencl/opencl.cpp")
            .compile("sloth256_189_opencl");
    }
}
