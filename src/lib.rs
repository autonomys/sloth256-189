#![warn(rust_2018_idioms, missing_debug_implementations, missing_docs)]
//! A Rust wrapper around both C/assembly and CUDA implementation of Sloth suggested in
//! https://eprint.iacr.org/2015/366, extended for a proof-of-replication,
//! and instantiated for 2**256-189 modulus.

pub mod cpu;
pub mod cuda;
