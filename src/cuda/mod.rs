//! A Rust wrapper around CUDA-PTX implementation of Sloth suggested in
//! https://eprint.iacr.org/2015/366. Uses `ptx.cu` for the caller functions and kernels (this file
//! contains high level CUDA code, not ptx code, but a caller for ptx code),
//! all the inner functions are present in `encode_ptx.h` as low-level PTX code

use std::error::Error;
use std::fmt;

#[cfg(test)]
mod tests;

/// Some CUDA Error occurred
#[derive(Debug, Copy, Clone)]
pub struct CudaError;

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Some CUDA Error occurred. GPU operation failed with error code..."
        )
        // TODO
        // configure this for different error types
    }
}

impl Error for CudaError {}

// importing the functions from .c files
extern "C" {
    fn sloth256_189_cuda_batch_encode(
        inout: *mut u8,
        len: usize,
        iv_: *const u8,
        layers: usize,
    ) -> bool;
    fn detect_cuda() -> bool;
}

/// checks if CUDA is available in the system
pub fn check_cuda() -> bool {
    unsafe { detect_cuda() }
}

/// Sequentially encodes a batch of pieces using CUDA
pub fn cuda_encode(piece: &mut Vec<u8>, iv: &[u8], layers: usize) -> Result<(), CudaError> {
    assert_eq!(piece.len() % (1024 * 4096), 0); // at least 1024 piece should be sent to GPU for batch
    unsafe {
        if sloth256_189_cuda_batch_encode(piece.as_mut_ptr(), piece.len(), iv.as_ptr(), layers) {
            return Err(CudaError);
        }
    };
    Ok(())
}
