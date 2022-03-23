//! Universal OpenCL GPU implementation, supports cards both AMD, Nvidia and Intel.
//!
//! It supports all major GPU vendors, including support for multiple GPU, including support for
//! multiple GPU vendors on the same machine at the same time.
//!
//! ### Linux support
//! In all cases for OpenCL support to build (not needed in runtime), OpenCL header files and
//! library are necessary, can be installed on Ubuntu as follows:
//! ```bash
//! sudo apt-get install ocl-icd-opencl-dev
//! ```
//!
//! #### AMD
//! Mesa OpenCL support is not compatible for AMD GPUs, you need to install ROCm-based OpenCL ICD.
//!
//! This can be done by downloading downloading and installing AMD driver installer from
//! <https://www.amd.com/en/support>, installing will look something like this:
//! ```bash
//! sudo dpkg -i amdgpu-install_21.50.2.50002-1_all.deb
//! ```
//! After which you should have `amdgpu-install` command, use it to install OpenCL ICD as follows:
//! ```bash
//! sudo amdgpu-install --no-dkms --usecase=opencl
//! ```
//!
//! ^ will only install OpenCL ICD and will keep Open Source kernel graphics driver as it was.
//!
//! Removal of all of the above is possible with:
//! ```bash
//! sudo amdgpu-install --uninstall
//! sudo apt-get purge amdgpu-install
//! ```
//!
//! #### Nvidia
//! Proprietary driver with CUDA toolchain is needed, Open Source drivers are not supported.
//!
//! #### Intel
//! Thankfully, for Intel GPU Open Source ICD works great:
//! ```bash
//! sudo apt-get install intel-opencl-icd
//! ```

use std::borrow::Cow;
use std::mem;
use std::os::raw::c_char;
use thiserror::Error;

const ENCODE_CL: &str = concat!(include_str!("encode.cl"), "\0");
const NVIDIA_SPECIFIC_CL: &str = concat!(include_str!("nvidia_specific.cl"), "\0");
const MOD256_189_CU: &str = concat!(include_str!("mod256-189.cu"), "\0");
const NON_NVIDIA_CL: &str = concat!(include_str!("non_nvidia.cl"), "\0");

/// OpenCL encoding errors
#[derive(Debug, Error)]
pub enum OpenCLEncodeError {
    /// Pieces argument is invalid, must be multiple of 4096-bytes pieces
    #[error("Pieces argument is invalid, must be multiple of 4096-bytes pieces, {0} bytes given")]
    InvalidPieces(usize),
    /// IVs argument is invalid, must be multiple of 32-bytes IVs
    #[error("IVs argument is invalid, must be multiple of 32-bytes pieces, {0} bytes given")]
    InvalidIVs(usize),
    /// Number of pieces should be the same as number of IVs
    #[error("Number of pieces should be the same as number of IVs, {0} pieces and {1} IVs given")]
    InvalidPiecesIVs(usize, usize),
    /// OpenCL API returned an error code
    #[error("OpenCL API error: {0}")]
    OpenCLError(Cow<'static, str>),
    /// OpenCL could not find any compatible device on the specified platform
    /// 2026 = No devices, 2027 = No Nvidia GPUs, 2028 = No AMD GPUs, 2029 = No Intel GPUs,
    /// 2035 = Pinned memory couldn't be allocated because no Nvidia GPU was found on the system
    /// 2036 = There was no previously allocated pinned memory
    #[error("No OpenCL compatible device could be found")]
    OpenCLDeviceNotFound(i32),
}

impl OpenCLEncodeError {
    /// Returns `Ok` in case code is not an error
    fn from_return_code(return_code: i32) -> Result<(), Self> {
        let error_string = match return_code {
            0 => {
                // No error
                return Ok(());
            }

            // runtime errors
            -1 => "CL_DEVICE_NOT_FOUND".into(),
            -2 => "CL_DEVICE_NOT_AVAILABLE".into(),
            -3 => "CL_COMPILER_NOT_AVAILABLE".into(),
            -4 => "CL_MEM_OBJECT_ALLOCATION_FAILURE".into(),
            -5 => "CL_OUT_OF_RESOURCES".into(),
            -6 => "CL_OUT_OF_HOST_MEMORY".into(),
            -7 => "CL_PROFILING_INFO_NOT_AVAILABLE".into(),
            -8 => "CL_MEM_COPY_OVERLAP".into(),
            -9 => "CL_IMAGE_FORMAT_MISMATCH".into(),
            -10 => "CL_IMAGE_FORMAT_NOT_SUPPORTED".into(),
            -11 => "CL_BUILD_PROGRAM_FAILURE".into(),
            -12 => "CL_MAP_FAILURE".into(),
            -13 => "CL_MISALIGNED_SUB_BUFFER_OFFSET".into(),
            -14 => "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ".into(),
            -15 => "CL_COMPILE_PROGRAM_FAILURE".into(),
            -16 => "CL_LINKER_NOT_AVAILABLE".into(),
            -17 => "CL_LINK_PROGRAM_FAILURE".into(),
            -18 => "CL_DEVICE_PARTITION_FAILED".into(),
            -19 => "CL_KERNEL_ARG_INFO_NOT_AVAILABLE".into(),

            // compile time errors
            -30 => "CL_INVALID_VALUE".into(),
            -31 => "CL_INVALID_DEVICE_TYPE".into(),
            -32 => "CL_INVALID_PLATFORM".into(),
            -33 => "CL_INVALID_DEVICE".into(),
            -34 => "CL_INVALID_CONTEXT".into(),
            -35 => "CL_INVALID_QUEUE_PROPERTIES".into(),
            -36 => "CL_INVALID_COMMAND_QUEUE".into(),
            -37 => "CL_INVALID_HOST_PTR".into(),
            -38 => "CL_INVALID_MEM_OBJECT".into(),
            -39 => "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR".into(),
            -40 => "CL_INVALID_IMAGE_SIZE".into(),
            -41 => "CL_INVALID_SAMPLER".into(),
            -42 => "CL_INVALID_BINARY".into(),
            -43 => "CL_INVALID_BUILD_OPTIONS".into(),
            -44 => "CL_INVALID_PROGRAM".into(),
            -45 => "CL_INVALID_PROGRAM_EXECUTABLE".into(),
            -46 => "CL_INVALID_KERNEL_NAME".into(),
            -47 => "CL_INVALID_KERNEL_DEFINITION".into(),
            -48 => "CL_INVALID_KERNEL".into(),
            -49 => "CL_INVALID_ARG_INDEX".into(),
            -50 => "CL_INVALID_ARG_VALUE".into(),
            -51 => "CL_INVALID_ARG_SIZE".into(),
            -52 => "CL_INVALID_KERNEL_ARGS".into(),
            -53 => "CL_INVALID_WORK_DIMENSION".into(),
            -54 => "CL_INVALID_WORK_GROUP_SIZE".into(),
            -55 => "CL_INVALID_WORK_ITEM_SIZE".into(),
            -56 => "CL_INVALID_GLOBAL_OFFSET".into(),
            -57 => "CL_INVALID_EVENT_WAIT_LIST".into(),
            -58 => "CL_INVALID_EVENT".into(),
            -59 => "CL_INVALID_OPERATION".into(),
            -60 => "CL_INVALID_GL_OBJECT".into(),
            -61 => "CL_INVALID_BUFFER_SIZE".into(),
            -62 => "CL_INVALID_MIP_LEVEL".into(),
            -63 => "CL_INVALID_GLOBAL_WORK_SIZE".into(),
            -64 => "CL_INVALID_PROPERTY".into(),
            -65 => "CL_INVALID_IMAGE_DESCRIPTOR".into(),
            -66 => "CL_INVALID_COMPILER_OPTIONS".into(),
            -67 => "CL_INVALID_LINKER_OPTIONS".into(),
            -68 => "CL_INVALID_DEVICE_PARTITION_COUNT".into(),

            // sloth256-189 encoding-specific errors
            2026 => "SLOTH_NO_OPENCL_COMPATIBLE_GPUS".into(),
            2027 => "SLOTH_NO_OPENCL_COMPATIBLE_NVIDIA_GPUS".into(),
            2028 => "SLOTH_NO_OPENCL_COMPATIBLE_AMD_GPUS".into(),
            2029 => "SLOTH_NO_OPENCL_COMPATIBLE_INTEL_GPUS".into(),

            // Should never happen since the caller Rust
            // function makes sure that there are more than 1024 pieces
            2031 => "SLOTH_PIECES_NOT_MULTIPLE_OF_1024".into(),

            // Pinned memory allocation fails if
            // there's no OpenCL compatible Nvidia GPUs
            2035 => "SLOTH_PINNED_MEMORY_ALLOCATION_FAILURE".into(),

            // There was no pinned memory allocated previously
            // so no memory to free
            2036 => "SLOTH_NO_ALLOCATED_PINNED_MEMORY".into(),

            // The work division between the CPU and the OpenCL compatible
            // devices were not yet determined.
            // Run the "determine_work_division_configuration" function before
            // encoding.
            2037 => "SLOTH_DEVICE_WORK_DIVISION_NOT_DETERMINED".into(),

            code => format!("Unknown OpenCL error {}", code).into(),
        };

        Err(Self::OpenCLError(error_string))
    }
}

// importing the functions from .c files
mod ffi {
    use std::os::raw::{c_char, c_int, c_uchar};

    #[repr(C)]
    pub(super) struct EncodeOpenCLInstances {
        _data: [u8; 0],
    }

    extern "C" {
        pub(super) fn sloth256_189_opencl_batch_encode(
            inout: *mut c_uchar,
            len: usize,
            iv: *const c_uchar,
            layers: usize,
            instances: *const EncodeOpenCLInstances,
        ) -> c_int;

        pub(super) fn sloth256_189_opencl_init(
            error: &mut c_int,
            encode_cl: *const c_char,
            nvidia_specific_cl: *const c_char,
            mod256_189_cu: *const c_char,
            non_nvidia_cl: *const c_char,
        ) -> *const EncodeOpenCLInstances;

        pub(super) fn sloth256_189_opencl_determine_factors(
            size: usize,
            layers: usize,
            instances: *const EncodeOpenCLInstances,
        ) -> c_int;

        // Pinned memory allocation is not exposed due to low performance improvement (1-2%) and
        // API complexity
        // pub(super) fn sloth256_189_pinned_alloc_supported(
        //     instances: *const EncodeOpenCLInstances,
        // ) -> bool;
        //
        // pub(super) fn sloth256_189_pinned_alloc(
        //     instances: *const EncodeOpenCLInstances,
        //     size: usize,
        //     error: &mut c_int,
        // ) -> *mut u8;
        //
        // pub(super) fn sloth256_189_pinned_free(instances: *const EncodeOpenCLInstances) -> c_int;

        pub(super) fn sloth256_189_opencl_cleanup(instances: *const EncodeOpenCLInstances)
            -> c_int;
    }
}

/// Batch to be encoded on GPU
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct OpenClBatch {
    /// Batch size (of pieces) in bytes (should be multiple of 4096)
    pub size: usize,
    /// Number of encoding layers
    pub layers: usize,
}

/// OpenCL codec
#[derive(Debug)]
pub struct OpenClEncoder {
    batch: Option<OpenClBatch>,
    instances: *const ffi::EncodeOpenCLInstances,
}

impl OpenClEncoder {
    /// Create new OpenCL codec instance for batch encoding on GPU.
    ///
    /// Batch information can be provided upfront to determine load distribution and do necessary
    /// memory allocation.
    pub fn new(batch: Option<OpenClBatch>) -> Result<Self, OpenCLEncodeError> {
        if let Some(batch) = &batch {
            // Ensure that the given size is valid
            if batch.size % 4096 != 0 {
                return Err(OpenCLEncodeError::InvalidPieces(batch.size));
            }
        }

        let mut return_code: i32 = 0;
        let instances = unsafe {
            ffi::sloth256_189_opencl_init(
                &mut return_code,
                ENCODE_CL.as_ptr() as *const c_char,
                NVIDIA_SPECIFIC_CL.as_ptr() as *const c_char,
                MOD256_189_CU.as_ptr() as *const c_char,
                NON_NVIDIA_CL.as_ptr() as *const c_char,
            )
        };

        OpenCLEncodeError::from_return_code(return_code)?;

        let mut encoder = Self {
            batch: None,
            instances,
        };

        if let Some(batch) = batch {
            encoder.recalculate_work_division_configuration(batch)?;

            encoder.batch.replace(batch);
        }

        Ok(encoder)
    }

    /// Sequentially encodes a batch of pieces using OpenCL.
    ///
    /// NOTE: This encode function works on batches of pieces and IVs.
    ///
    /// For smaller batches or encoding of individual pieces use CPU implementation.
    pub fn encode(
        &mut self,
        pieces: &mut [u8],
        ivs: &[u8],
        layers: usize,
    ) -> Result<(), OpenCLEncodeError> {
        if pieces.len() % 4096 != 0 {
            return Err(OpenCLEncodeError::InvalidPieces(pieces.len()));
        }

        if ivs.len() % 32 != 0 {
            return Err(OpenCLEncodeError::InvalidIVs(ivs.len()));
        }

        if pieces.len() / 4096 != ivs.len() / 32 {
            return Err(OpenCLEncodeError::InvalidPiecesIVs(
                pieces.len() / 4096,
                ivs.len() / 32,
            ));
        }

        let batch = OpenClBatch {
            size: pieces.len() / 4096,
            layers,
        };

        if self.batch != Some(batch) {
            self.recalculate_work_division_configuration(batch)?;
        }

        let return_code = unsafe {
            ffi::sloth256_189_opencl_batch_encode(
                pieces.as_mut_ptr(),
                pieces.len(),
                ivs.as_ptr(),
                layers,
                self.instances,
            )
        };

        OpenCLEncodeError::from_return_code(return_code)
    }

    /// Cleans up the resources allocated in the initialization of the encode kernel.
    ///
    /// Prefer this over `drop()` because `drop()` will panic in case of error.
    ///
    /// NOTE: In case error is returned, memory used for kernel initialization might be leaked.
    pub fn destroy(self) -> Result<(), OpenCLEncodeError> {
        let return_code = unsafe { ffi::sloth256_189_opencl_cleanup(self.instances) };

        // We don't want to run `Drop::drop` after this
        mem::forget(self);

        OpenCLEncodeError::from_return_code(return_code)
    }

    /// Determine the work division configuration between the CPU and the OpenCL compatible
    /// devices for a given size and number of layers.
    ///
    /// Call this function after initialization and if the encoding size or number of layers
    /// change.
    fn recalculate_work_division_configuration(
        &mut self,
        batch: OpenClBatch,
    ) -> Result<(), OpenCLEncodeError> {
        // Ensure that the given size is valid
        if batch.size % 4096 != 0 {
            return Err(OpenCLEncodeError::InvalidPieces(batch.size));
        }

        let return_code = unsafe {
            ffi::sloth256_189_opencl_determine_factors(batch.size, batch.layers, self.instances)
        };

        OpenCLEncodeError::from_return_code(return_code)
    }
}

impl Drop for OpenClEncoder {
    fn drop(&mut self) {
        let return_code = unsafe { ffi::sloth256_189_opencl_cleanup(self.instances) };

        OpenCLEncodeError::from_return_code(return_code).unwrap();
    }
}
