pub fn get_opencl_error_string(error_code: i32) -> String {
    match error_code {
        // runtime errors
        -1 => "CL_DEVICE_NOT_FOUND".to_string(),
        -2 => "CL_DEVICE_NOT_AVAILABLE".to_string(),
        -3 => "CL_COMPILER_NOT_AVAILABLE".to_string(),
        -4 => "CL_MEM_OBJECT_ALLOCATION_FAILURE".to_string(),
        -5 => "CL_OUT_OF_RESOURCES".to_string(),
        -6 => "CL_OUT_OF_HOST_MEMORY".to_string(),
        -7 => "CL_PROFILING_INFO_NOT_AVAILABLE".to_string(),
        -8 => "CL_MEM_COPY_OVERLAP".to_string(),
        -9 => "CL_IMAGE_FORMAT_MISMATCH".to_string(),
        -10 => "CL_IMAGE_FORMAT_NOT_SUPPORTED".to_string(),
        -11 => "CL_BUILD_PROGRAM_FAILURE".to_string(),
        -12 => "CL_MAP_FAILURE".to_string(),
        -13 => "CL_MISALIGNED_SUB_BUFFER_OFFSET".to_string(),
        -14 => "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ".to_string(),
        -15 => "CL_COMPILE_PROGRAM_FAILURE".to_string(),
        -16 => "CL_LINKER_NOT_AVAILABLE".to_string(),
        -17 => "CL_LINK_PROGRAM_FAILURE".to_string(),
        -18 => "CL_DEVICE_PARTITION_FAILED".to_string(),
        -19 => "CL_KERNEL_ARG_INFO_NOT_AVAILABLE".to_string(),

        // compile time errors
        -30 => "CL_INVALID_VALUE".to_string(),
        -31 => "CL_INVALID_DEVICE_TYPE".to_string(),
        -32 => "CL_INVALID_PLATFORM".to_string(),
        -33 => "CL_INVALID_DEVICE".to_string(),
        -34 => "CL_INVALID_CONTEXT".to_string(),
        -35 => "CL_INVALID_QUEUE_PROPERTIES".to_string(),
        -36 => "CL_INVALID_COMMAND_QUEUE".to_string(),
        -37 => "CL_INVALID_HOST_PTR".to_string(),
        -38 => "CL_INVALID_MEM_OBJECT".to_string(),
        -39 => "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR".to_string(),
        -40 => "CL_INVALID_IMAGE_SIZE".to_string(),
        -41 => "CL_INVALID_SAMPLER".to_string(),
        -42 => "CL_INVALID_BINARY".to_string(),
        -43 => "CL_INVALID_BUILD_OPTIONS".to_string(),
        -44 => "CL_INVALID_PROGRAM".to_string(),
        -45 => "CL_INVALID_PROGRAM_EXECUTABLE".to_string(),
        -46 => "CL_INVALID_KERNEL_NAME".to_string(),
        -47 => "CL_INVALID_KERNEL_DEFINITION".to_string(),
        -48 => "CL_INVALID_KERNEL".to_string(),
        -49 => "CL_INVALID_ARG_INDEX".to_string(),
        -50 => "CL_INVALID_ARG_VALUE".to_string(),
        -51 => "CL_INVALID_ARG_SIZE".to_string(),
        -52 => "CL_INVALID_KERNEL_ARGS".to_string(),
        -53 => "CL_INVALID_WORK_DIMENSION".to_string(),
        -54 => "CL_INVALID_WORK_GROUP_SIZE".to_string(),
        -55 => "CL_INVALID_WORK_ITEM_SIZE".to_string(),
        -56 => "CL_INVALID_GLOBAL_OFFSET".to_string(),
        -57 => "CL_INVALID_EVENT_WAIT_LIST".to_string(),
        -58 => "CL_INVALID_EVENT".to_string(),
        -59 => "CL_INVALID_OPERATION".to_string(),
        -60 => "CL_INVALID_GL_OBJECT".to_string(),
        -61 => "CL_INVALID_BUFFER_SIZE".to_string(),
        -62 => "CL_INVALID_MIP_LEVEL".to_string(),
        -63 => "CL_INVALID_GLOBAL_WORK_SIZE".to_string(),
        -64 => "CL_INVALID_PROPERTY".to_string(),
        -65 => "CL_INVALID_IMAGE_DESCRIPTOR".to_string(),
        -66 => "CL_INVALID_COMPILER_OPTIONS".to_string(),
        -67 => "CL_INVALID_LINKER_OPTIONS".to_string(),
        -68 => "CL_INVALID_DEVICE_PARTITION_COUNT".to_string(),

        // sloth256-189 encoding-specific errors
        2026 => "SLOTH_NO_OPENCL_COMPATIBLE_GPUS".to_string(),
        2027 => "SLOTH_NO_OPENCL_COMPATIBLE_NVIDIA_GPUS".to_string(),
        2028 => "SLOTH_NO_OPENCL_COMPATIBLE_AMD_GPUS".to_string(),
        2029 => "SLOTH_NO_OPENCL_COMPATIBLE_INTEL_GPUS".to_string(),

        // Should never happen since the caller Rust
        // function makes sure that there are more than 1024 pieces
        2031 => "SLOTH_PIECES_NOT_MULTIPLE_OF_1024".to_string(),

        // Pinned memory allocation fails if
        // there's no OpenCL compatible Nvidia GPUs
        2035 => "SLOTH_PINNED_MEMORY_ALLOCATION_FAILURE".to_string(),

        // There was no pinned memory allocated previously
        // so no memory to free
        2036 => "SLOTH_NO_ALLOCATED_PINNED_MEMORY".to_string(),

        // The work division between the CPU and the OpenCL compatible
        // devices were not yet determined.
        // Run the "determine_work_division_configuration" function before
        // encoding.
        2037 => "SLOTH_DEVICE_WORK_DIVISION_NOT_DETERMINED".to_string(),

        _ => "Unknown OpenCL error".to_string(),
    }
}
