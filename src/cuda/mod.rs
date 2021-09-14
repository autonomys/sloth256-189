//! CUDA (PTX) implementation

use thiserror::Error;

/// CUDA encoding errors
#[derive(Debug, Error)]
pub enum EncodeError {
    /// Pieces argument is invalid, must be multiple of 1024 4096-bytes pieces
    #[error(
        "Pieces argument is invalid, must be multiple of 1024 4096-bytes pieces, {0} bytes given"
    )]
    InvalidPieces(usize),
    /// IVs argument is invalid, must be multiple of 1024 32-bytes IVs
    #[error("IVs argument is invalid, must be multiple of 1024 32-bytes pieces, {0} bytes given")]
    InvalidIVs(usize),
    /// Failed to get memory info, likely means there is no compatible GPU available
    #[error("Failed to get memory info")]
    CudaMemGetInfo,
    /// Not enough memory available on GPU for pieces
    #[error("Not enough memory available on GPU for pieces")]
    CudaMallocPieces,
    /// Not enough memory available on GPU for IVs
    #[error("Not enough memory available on GPU for IVs")]
    CudaMallocIVs,
    /// Failed to copy pieces to GPU
    #[error("Failed to copy pieces to GPU")]
    CudaMemcpyPieces,
    /// Failed to copy IVs to GPU
    #[error("Failed to copy IVs to GPU")]
    CudaMemcpyIVs,
    /// Kernel launch error
    #[error("Kernel launch error")]
    KernelLaunch,
    /// Kernel did not finish correctly (CudaSynchronize)
    #[error("Kernel did not finish correctly (CudaSynchronize)")]
    CudaSynchronize,
    /// Failed to copy encoded piece back to the CPU
    #[error("Failed to copy encoded piece back to the CPU")]
    MemcpyPieceToHost,
}

// importing the functions from .c files
mod ffi {
    extern "C" {
        pub(super) fn is_cuda_available() -> bool;
        pub(super) fn sloth256_189_cuda_batch_encode(
            inout: *mut u8,
            len: usize,
            iv_: *const u8,
            layers: usize,
        ) -> u8;
    }
}

/// Checks if compatible CUDA GPU is available on the system
pub fn is_cuda_available() -> bool {
    unsafe { ffi::is_cuda_available() }
}

/// Sequentially encodes a batch of pieces using CUDA.
///
/// NOTE: This encode function works on batches of pieces and IVs that should be multiple of 1024.
/// For smaller batches or encoding of individual pieces use CPU implementation.
pub fn encode(pieces: &mut [u8], ivs: &[u8], layers: usize) -> Result<(), EncodeError> {
    if pieces.len() % (1024 * 4096) != 0 {
        return Err(EncodeError::InvalidPieces(pieces.len()));
    }
    if ivs.len() % (1024 * 32) != 0 {
        return Err(EncodeError::InvalidPieces(ivs.len()));
    }

    let return_code = unsafe {
        ffi::sloth256_189_cuda_batch_encode(pieces.as_mut_ptr(), pieces.len(), ivs.as_ptr(), layers)
    };
    return match return_code {
        0 => Ok(()),
        1 => Err(EncodeError::CudaMemGetInfo),
        2 => Err(EncodeError::CudaMallocPieces),
        3 => Err(EncodeError::CudaMallocIVs),
        4 => Err(EncodeError::CudaMemcpyPieces),
        5 => Err(EncodeError::CudaMemcpyIVs),
        6 => Err(EncodeError::KernelLaunch),
        7 => Err(EncodeError::CudaSynchronize),
        8 => Err(EncodeError::MemcpyPieceToHost),
        _ => unreachable!("there is no such error code being returned"),
    };
}
