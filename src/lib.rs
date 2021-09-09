#![warn(rust_2018_idioms, missing_debug_implementations, missing_docs)]
//! A Rust wrapper around C/assembly implementation of Sloth suggested in
//! https://eprint.iacr.org/2015/366, extended for a proof-of-replication,
//! and instantiated for 2**256-189 modulus.

#[cfg(test)]
mod tests;

mod cpu_sloth;
mod gpu_sloth;

use cpu_sloth::decode;
use cpu_sloth::encode;

use gpu_sloth::check_cuda;
use gpu_sloth::gpu_encode;
