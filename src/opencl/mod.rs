//! OpenCL implementation

use thiserror::Error;

mod opencl_error_codes;
use opencl_error_codes::get_opencl_error_string;

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
    OpenCLError(String),
    /// OpenCL could not find any compatible device on the specified platform
    /// 2026 = No devices, 2027 = No Nvidia GPUs, 2028 = No AMD GPUs, 2029 = No Intel GPUs,
    /// 2035 = Pinned memory couldn't be allocated because no Nvidia GPU was found on the system
    /// 2036 = There was no previously allocated pinned memory
    #[error("No OpenCL compatible device could be found")]
    OpenCLDeviceNotFound(i32),
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

        pub(super) fn sloth256_189_opencl_init(
            error: &mut i32,
            encode_cl: *const i8,
            nvidia_specific_cl: *const i8,
            mod256_189_cu: *const i8,
            non_nvidia_cl: *const i8,
        ) -> *const u8;

        pub(super) fn sloth256_189_opencl_determine_factors(
            size: usize,
            layers: usize,
            instances: *const u8,
        ) -> i32;

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
/// ///
/// Unfortunately, when allocating pinned memory with OpenCL, a device buffer is also created
/// in addition to the buffer created on the host. Then, during the encoding process, yet another
/// buffer is created to write the contents of piece data. Using pinned memory without creating
/// another buffer during the encode process and writing the contents of piece data to the device
/// buffer created initially when allocating pinned memory is possible, without any wasted device
/// memory, but it results in an overall slower encoding time since there's extra memory allocation.
/// See: https://stackoverflow.com/questions/42011504/why-does-clcreatebuffer-with-cl-mem-alloc-host-ptr-use-discrete-device-memory
///
/// Therefore we can only use around 1/3 of the available amount of device memory at maximum.
pub fn pinned_memory_alloc(instance: *const u8, size: usize) -> Result<Vec<u8>, OpenCLEncodeError> {
    let mut return_code: i32 = 0;

    let pointer = unsafe { ffi::sloth256_189_pinned_alloc(instance, size, &mut return_code) };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(get_opencl_error_string(
            return_code,
        )));
    }

    let pointer_as_vec = unsafe { Vec::from_raw_parts(pointer, size, size) };

    return Ok(pointer_as_vec);
}

/// Free the pinned memory allocated with pinned_memory_alloc function above.
/// A std::mem::forget({vector}) call is required since the vector in Rust will
/// be using memory freed in C.
/// Call this function before calling the cleanup function
pub fn pinned_memory_free(instances: *const u8) -> Result<(), OpenCLEncodeError> {
    let return_code = unsafe { ffi::sloth256_189_pinned_free(instances) };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(get_opencl_error_string(
            return_code,
        )));
    }

    return Ok(());
}

/// Initializes the encode kernel for a given device. In essence performs
/// the tasks that must be done only once at the start.
/// Returns a pointer to the initialized instance which should be passed
/// to the encode and cleanup functions.
/// Calling this once at the start is sufficient.
pub fn initialize() -> Result<*const u8, OpenCLEncodeError> {
    let encode_cl = include_bytes!("encode.cl");
    let nvidia_specific_cl = include_bytes!("nvidia_specific.cl");
    let mod256_189_cu = include_bytes!("mod256-189.cu");
    let non_nvidia_cl = include_bytes!("non_nvidia.cl");

    let encode_cl_str = String::from_utf8_lossy(encode_cl).to_string() + &"\0".to_string();

    let nvidia_specific_cl_str =
        String::from_utf8_lossy(nvidia_specific_cl).to_string() + &"\0".to_string();

    let mod256_189_cu_str = String::from_utf8_lossy(mod256_189_cu).to_string() + &"\0".to_string();

    let non_nvidia_cl_str = String::from_utf8_lossy(non_nvidia_cl).to_string() + &"\0".to_string();

    let mut return_code: i32 = 0;
    let instances = unsafe {
        ffi::sloth256_189_opencl_init(
            &mut return_code,
            encode_cl_str.as_ptr() as *const i8,
            nvidia_specific_cl_str.as_ptr() as *const i8,
            mod256_189_cu_str.as_ptr() as *const i8,
            non_nvidia_cl_str.as_ptr() as *const i8,
        )
    };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(get_opencl_error_string(
            return_code,
        )));
    }

    return Ok(instances);
}

/// Determine the work division configuration between the CPU and the OpenCL compatible
/// devices for a given size and number of layers.
/// Call this function after initialization and if the encoding size or number of layers
/// change.
pub fn determine_work_division_configuration(
    size: usize,
    layers: usize,
    instances: *const u8,
) -> Result<(), OpenCLEncodeError> {
    // Ensure that the given size is valid
    if size % (1024 * 4096) != 0 {
        return Err(OpenCLEncodeError::InvalidPieces(size));
    }

    let return_code =
        unsafe { ffi::sloth256_189_opencl_determine_factors(size, layers, instances) };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(get_opencl_error_string(
            return_code,
        )));
    }

    return Ok(());
}

/// Cleans up the resources allocated in the initialization of the encode kernel.
/// Calling this once at the end is sufficient.
pub fn cleanup(instances: *const u8) -> Result<(), OpenCLEncodeError> {
    let return_code = unsafe { ffi::sloth256_189_opencl_cleanup(instances) };

    if return_code != 0 {
        return Err(OpenCLEncodeError::OpenCLError(get_opencl_error_string(
            return_code,
        )));
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
        return Err(OpenCLEncodeError::OpenCLError(get_opencl_error_string(
            return_code,
        )));
    }

    return Ok(());
}
