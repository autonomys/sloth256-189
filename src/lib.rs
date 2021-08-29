#![warn(rust_2018_idioms, missing_debug_implementations, missing_docs)]
//! A Rust wrapper around C/assembly implementation of Sloth suggested in
//! https://eprint.iacr.org/2015/366, extended for a proof-of-replication,
//! and instantiated for 2**256-189 modulus.

#[cfg(test)]
mod tests;

use std::error::Error;
use std::fmt;

extern "C" {
    fn sloth256_189_encode(inout: *mut u8, len: usize, iv_: *const u8, layers: usize) -> bool;
    fn sloth256_189_decode(inout: *mut u8, len: usize, iv_: *const u8, layers: usize);
}

/// Data bigger than the prime, this is not supported
#[derive(Debug, Copy, Clone)]
pub struct DataBiggerThanPrime;

impl fmt::Display for DataBiggerThanPrime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Data bigger than the prime, this is not supported")
    }
}

impl Error for DataBiggerThanPrime {}

/// Sequentially encodes a 4096 byte piece s.t. a minimum amount of
/// wall clock time elapses
pub fn encode(
    piece: &mut [u8; 4096],
    iv: [u8; 32],
    layers: usize,
) -> Result<(), DataBiggerThanPrime> {
    unsafe {
        if sloth256_189_encode(piece.as_mut_ptr(), piece.len(), iv.as_ptr(), layers) {
            return Err(DataBiggerThanPrime);
        }
    };
    Ok(())
}

/// Sequentially decodes a 4096 byte encoding in time << encode time
pub fn decode(piece: &mut [u8; 4096], iv: [u8; 32], layers: usize) {
    unsafe { sloth256_189_decode(piece.as_mut_ptr(), piece.len(), iv.as_ptr(), layers) };
}
