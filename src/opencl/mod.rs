//! OpenCL implementation

use thiserror::Error;
//use std::slice;

/// OpenCL encoding errors
#[derive(Debug, Error)]
pub enum OpenCLEncodeError {
    /// Pieces argument is invalid, must be multiple of 1024 4096-bytes pieces
    #[error(
        "Pieces argument is invalid, must be multiple of 1024 4096-bytes pieces, {0} bytes given"
    )]
    InvalidPieces(usize),
    /// IVs argument is invalid, must be multiple of 1024 32-bytes IVs
    #[error("IVs argument is invalid, must be multiple of 1024 32-bytes pieces, {0} bytes given")]
    InvalidIVs(usize),
    /// Number of pieces should be the same as number of IVs
    #[error("Number of pieces should be the same as number of IVs, {0} pieces and {1} IVs given")]
    InvalidPiecesIVs(usize, usize),
    /// OpenCL API returned an error code
    #[error("OpenCL API error: {0}")]
    OpenCLError(i32),
    /// OpenCL could not find any compatible device on the specified platform
    /// 2026 = No devices, 2027 = No Nvidia GPUs, 2028 = No AMD GPUs, 2029 = No Intel GPUs,
    /// 2035 = Pinned memory couldn't be allocated because no Nvidia GPU was found on the system
    /// 2036 = There was no previously allocated pinned memory
    #[error("No OpenCL compatible device could be found")]
    OpenCLDeviceNotFound(i32),
    /// One of the OpenCL kernel files could not be found
    #[error("One of the OpenCL kernel (.cl) files could not be found")]
    OpenCLKernelFileNotFound(),
}

// importing the functions from .c files
mod ffi {
    extern "C" {
        pub(super) fn sloth256_189_opencl_batch_encode(
            inout: *mut u8,
            len: usize,
            iv: *const u8,
            layers: usize,
            instances: *const u8,
        ) -> i32;

        pub(super) fn sloth256_189_opencl_init(error: &mut i32) -> *const u8;

        pub(super) fn sloth256_189_pinned_alloc(
            instances: *const u8,
            size: usize,
            error: &mut i32,
        ) -> *mut u8;

        pub(super) fn sloth256_189_pinned_free(instances: *const u8) -> i32;

        pub(super) fn sloth256_189_opencl_cleanup(instances: *const u8) -> i32;
    }
}

/// Allocate pinned and aligned host memory. This makes memory copy operations on NVIDIA GPUs faster
/// and allignment is required for zero-copy buffers for Intel integrated graphics devices
/// Call this function after calling the init function
pub fn pinned_memory_alloc(instance: *const u8, size: usize) -> Result<Vec<u8>, OpenCLEncodeError> {
    let mut return_code: i32 = 0;

    let pointer = unsafe { ffi::sloth256_189_pinned_alloc(instance, size, &mut return_code) };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(return_code));
    }

    let pointer_as_vec = unsafe { Vec::from_raw_parts(pointer, size, size) };

    return Ok(pointer_as_vec);
}

/// Free the pinned memory allocated with pinned_memory_alloc function above. A std::mem::forget({vector})
/// call is required since the vector in Rust will be using memory freed in C.
/// Call this function before calling the cleanup function
pub fn pinned_memory_free(instances: *const u8) -> Result<(), OpenCLEncodeError> {
    let return_code = unsafe { ffi::sloth256_189_pinned_free(instances) };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(return_code));
    }

    return Ok(());
}

/// Initializes the encode kernel for a given device. In essence performs the tasks that must be done only once at the start.
/// Returns a pointer to the initialized instance which should be passed to the encode and cleanup functions.
/// Calling this once at the start is sufficient.
pub fn initialize() -> Result<*const u8, OpenCLEncodeError> {
    let mut return_code: i32 = 0;
    let instances = unsafe { ffi::sloth256_189_opencl_init(&mut return_code) };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(return_code));
    }

    return Ok(instances);
}

/// Cleans up the resources allocated in the initialization of the encode kernel.
/// Calling this once at the end is sufficient.
pub fn cleanup(instances: *const u8) -> Result<(), OpenCLEncodeError> {
    let return_code = unsafe { ffi::sloth256_189_opencl_cleanup(instances) };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(return_code));
    }

    return Ok(());
}

/// Sequentially encodes a batch of pieces using OpenCL.
///
/// NOTE: This encode function works on batches of pieces and IVs that should be multiple of 1024.
/// For smaller batches or encoding of individual pieces use CPU implementation.
pub fn encode(
    pieces: &mut [u8],
    ivs: &[u8],
    layers: usize,
    instances: *const u8,
) -> Result<(), OpenCLEncodeError> {
    if pieces.len() % (1024 * 4096) != 0 {
        return Err(OpenCLEncodeError::InvalidPieces(pieces.len()));
    }
    if ivs.len() % (1024 * 32) != 0 {
        return Err(OpenCLEncodeError::InvalidIVs(ivs.len()));
    }
    if pieces.len() / 4096 != ivs.len() / 32 {
        return Err(OpenCLEncodeError::InvalidPiecesIVs(
            pieces.len() / 4096,
            ivs.len() / 32,
        ));
    }

    let return_code = unsafe {
        ffi::sloth256_189_opencl_batch_encode(
            pieces.as_mut_ptr(),
            pieces.len(),
            ivs.as_ptr(),
            layers,
            instances,
        )
    };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(return_code));
    }

    return Ok(());
}
