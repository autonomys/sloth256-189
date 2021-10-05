<div align="center">
  <h1><code>sloth256-189 - CUDA</code></h1>
  <strong>Encoder/decoder for the <a href="https://subspace.network/">Subspace Network Blockchain</a> based on the <a href="https://eprint.iacr.org/2015/366">SLOTH permutation</a></strong>
</div>

Below is the summary of the CUDA implementation of sloth256-189

---

- `encode_ptx.h` is the PTX library for the CUDA functions
- `ptx.cu` is the caller of the PTX library functions, and handling various errors
- `mod.rs` is the file that communicates the CUDA code with Rust.

---

GPUs are efficient for processing batches of data, hence, the CUDA implementation of `encode` function
takes batch of pieces to encode. `ptx.cu` handles all the memory allocations dynamically (if there is not enough memory
on the device, pieces will be split into smaller batches), to achieve the best performance and handle edge cases.

---

For Supranational's Performance Report on CUDA implementation of Sloth256-189, please refer to 
[Supranational-GPU-Performance-Report.pdf](Supranational-GPU-Performance-Report.pdf)

---

CUDA/PTX code is following the same base-algorithm with cpu. Please refer to [README.md](../cpu/README.md) and [Documentation.md](../cpu/Documentation.md) in 
`src/cpu` for more in-depth explanations of the base-algorithm.
