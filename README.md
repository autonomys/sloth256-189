<div align="center">
  <h1><code>sloth256-189</code></h1>
  <strong>Encoder/decoder for the <a href="https://subspace.network/">Subspace Network Blockchain</a> based on the <a href="https://eprint.iacr.org/2015/366">SLOTH permutation</a></strong>
</div>

[![CI](https://github.com/subspace/sloth256-189/actions/workflows/ci.yaml/badge.svg)](https://github.com/subspace/sloth256-189/actions/workflows/ci.yaml)
[![Crates.io](https://img.shields.io/crates/v/sloth256-189)](https://crates.io/crates/sloth256-189)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.rs/sloth256-189)

This is an adaptation of [SLOTH](https://eprint.iacr.org/2015/366) (slow-timed hash function) into a time-asymmetric
permutation using a standard CBC block cipher.

This library consists of 3 major implementations:
- CPU
- CUDA (requires `cuda` feature to be enabled)
- OpenCL (requires `opencl` feature to be enabled)

WebAssembly (`no_std`) is also supported, but `llvm-ar` command needs to be available for `build.rs` to succeed in that case.

### CPU

CPU Implementation contains 3 flavors:
* optimized assembly-assisted implementation for x86-64 processors with ADX ISA extension (Linux, macOS and Windows)
* any 64-bt platform with support for `__int128` C type (modern GCC/Clang, but not MSVC)
* fallback for other platforms

For more details, [README.md](src/cpu/README.md) under `src/cpu` can be referred.

### CUDA

CUDA implementation is heavily using low-level PTX code to achieve the maximum performance.

Details of the CUDA implementation can be found in [README.md](src/cuda/README.md) under `src/cuda`.

### OpenCL

OpenCL is a universal implementation that supports AMD (ROCm), Nvidia (CUDA) and Intel GPUs, including support for
multiple GPU vendors on the same machine at the same time.

Details of the OpenCL implementation can be found in [mod.rs](src/opencl/mod.rs) under `src/opencl`.

### How to test

Simply run `cargo test` at the root folder

- use `cargo test --features=cuda` for enabling cuda tests (CUDA toolkit must be installed on your device)
- use `cargo test --features=opencl` for enabling opencl tests (openCL drivers must be installed on your device)

### How to benchmark

Simply run `cargo bench` at the root folder

- use `cargo bench --features=cuda` for enabling cuda tests (CUDA toolkit must be installed on your device)
- use `cargo bench --features=opencl` for enabling opencl tests (openCL drivers must be installed on your device)

### License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in cc-rs by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
