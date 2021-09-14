//! A Rust wrapper around CUDA-PTX implementation of Sloth suggested in
//! https://eprint.iacr.org/2015/366. Uses `ptx.cu` for the caller functions and kernels (this file
//! contains high level CUDA code, not ptx code, but a caller for ptx code),
//! all the inner functions are present in `encode_ptx.h` as low-level PTX code

use thiserror::Error;

#[cfg(test)]
mod tests;

/// CUDA error handler
#[derive(Debug, Error)]
pub enum CudaError {
    /// CudaStatus is returning 1 in case of any error happens,
    /// but we want to inspect the error deeper than that, so in other words:
    /// this means there is an error happening somewhere that we didn't think of
    #[error("CudaStatus returned 1 (this should not happen).")]
    Generic,
    /// trying to allocate memory depending on the free memory currently available on the GPU
    #[error("Required Memory for PieceArray could not be allocated in the GPU.")]
    CudaMallocPiece,
    /// same for IV
    #[error("Required Memory for IV could not be allocated in the GPU.")]
    CudaMallocIV,
    /// trying to copy the PieceArray to the GPU, if malloc is successful, this shouldn't be a problem
    #[error("PieceArray could not be copied to the GPU.")]
    CudaMemcpyPiece,
    /// same for IV
    #[error("IV could not be copied to the GPU.")]
    CudaMemcpyIV,
    /// This indicates, during the kernel launch, something bad happened
    #[error("Something went wrong with the Kernel Launch.")]
    KernelLaunch,
    /// means: CudaSynchronize was not successful, probably related to Kernel
    #[error("Kernel did not finish correctly.")]
    CudaSynchronize,
    /// during the copy of PieceArray back to the CPU, something bad happened, this is highly unlikely
    #[error("PieceArray could not be copied back to the CPU")]
    MemcpyPieceToHost,
}

// importing the functions from .c files
extern "C" {
    fn sloth256_189_cuda_batch_encode(
        inout: *mut u8,
        len: usize,
        iv_: *const u8,
        layers: usize,
    ) -> u8;
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
        let return_code =
            sloth256_189_cuda_batch_encode(piece.as_mut_ptr(), piece.len(), iv.as_ptr(), layers);
        return match return_code {
            0 => Ok(()),
            1 => Err(CudaError::Generic),
            2 => Err(CudaError::CudaMallocPiece),
            3 => Err(CudaError::CudaMallocIV),
            4 => Err(CudaError::CudaMemcpyPiece),
            5 => Err(CudaError::CudaMemcpyIV),
            6 => Err(CudaError::KernelLaunch),
            7 => Err(CudaError::CudaSynchronize),
            8 => Err(CudaError::MemcpyPieceToHost),
            _ => unreachable!("there is no such error code being returned"),
        };
    };
}
