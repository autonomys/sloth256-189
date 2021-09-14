#![warn(rust_2018_idioms, missing_debug_implementations, missing_docs)]
//! A Rust wrapper around both C/assembly and CUDA implementation of Sloth permutation suggested in
//! <https://eprint.iacr.org/2015/366>, extended for a proof-of-replication, and instantiated for
//! 2**256-189 modulus used in Subspace Network.
//!
//! Universal CPU implementation is always available, for CUDA support make sure to enable `cuda`
//! feature and have CUDA toolchain installed.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
