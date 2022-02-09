#include <assert.h>
#include <omp.h>

#include <algorithm>
#include <thread>
#include <chrono>

#include "./opencl_utils.hpp"

#define OCL_ERR_CHECK(X, Y) if ((X) != CL_SUCCESS) return (Y)

enum Platform { NVIDIA, AMD, INTEL };

#define MAX_NUM_COMMAND_QUEUES 64
#define MAX_NUM_COMMAND_QUEUES_INTEL 4

#define ONE_GB 1073741824llu // in bytes

struct EncodeOpenCLInstance {
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue cq[MAX_NUM_COMMAND_QUEUES];
    Platform platform;
};

struct EncodeOpenCLInstances {
    EncodeOpenCLInstance* instances;
    size_t num_instances;
    cl_mem pinned;
    unsigned char* c_pinned;
    int pinned_pos;
};

extern "C" {
    void sloth256_189_encode(unsigned char*, size_t, const unsigned char*, size_t);
}

// Encode a portion of the data with multiple CPU cores/threads.
// Use a lower amount than the maximum amount to not make other
// encoding threads lag
void sloth256_189_cpu_encode_parallel(unsigned char* inout, 
                                      size_t total_size, 
                                      const unsigned char* iv, 
                                      size_t layers) {

    size_t num_pieces = total_size / 4096;

    unsigned int num_threads = std::thread::hardware_concurrency();

    // Code left in to time the encoding of data on the CPU and various different
    // platforms. Uncomment the below line of code in addition to the code in
    // lines 62-65 to enable reporting of CPU encoding times.
    /*auto start = std::chrono::high_resolution_clock::now();*/

    #pragma omp parallel for num_threads(num_threads / 2 - 1)
    for (long long int i = 0; i < (long long int)num_pieces; i++) {
        sloth256_189_encode(inout + 4096 * (size_t)i, 4096, iv + 32 * (size_t)i, layers);
    }

    /*auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total elapsed time for CPU encode: " 
              << (std::chrono::duration<double>(end - start)).count() 
              << " seconds" << std::endl;*/
}

// Allocate pinned memory bound to one of the Nvidia instances.
// In contrast to CUDA, pinned memory in OpenCL cannot exist on its own
// and must be "bound" to a platform. We bind it to an Nvidia GPU
// and the function returns an error if there are no Nvidia GPUs present.
extern "C"
unsigned char* sloth256_189_pinned_alloc(EncodeOpenCLInstances* instances, size_t size, cl_int& err) {
    EncodeOpenCLInstance* instance = NULL;
    for (size_t i = 0; i < instances->num_instances; i++) {
        if (instances->instances[i].platform == NVIDIA) {
            instance = instances->instances + i;
            instances->pinned_pos = (int)i;
            break;
        }
    }
    if (instance == NULL) {
        err = SLOTH_PINNED_MEMORY_ALLOCATION_FAILURE;
        return NULL;
    }

    instances->pinned = clCreateBuffer(instance->context, 
                                       CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
                                       size, NULL, &err); 
    OCL_ERR_CHECK(err, instances->c_pinned);
    
    // Obtain a pointer which points to the beginning of the pinned memory buffer.
    // This is done so that we can write into the buffer in a usual way, rather
    // than using more OpenCL API calls.
    instances->c_pinned = (unsigned char*)clEnqueueMapBuffer(instance->cq[0], 
                                                             instances->pinned, 
                                                             CL_TRUE, 
                                                             CL_MAP_WRITE | CL_MAP_READ, 
                                                             0, size, 0, NULL, 
                                                             NULL, &err); 
    OCL_ERR_CHECK(err, instances->c_pinned);

    return instances->c_pinned;
}

extern "C"
cl_int sloth256_189_pinned_free(EncodeOpenCLInstances* instances) {
    if (instances->pinned_pos == -1)
        return SLOTH_NO_ALLOCATED_PINNED_MEMORY;

    EncodeOpenCLInstance* instance = instances->instances + instances->pinned_pos;

    cl_int err;

    err = clEnqueueUnmapMemObject(instance->cq[0], instances->pinned, 
                                  (void*)instances->c_pinned, 0, NULL, NULL); 
    OCL_ERR_CHECK(err, err);

    instances->c_pinned = NULL;

    err = clReleaseMemObject(instances->pinned); 
    OCL_ERR_CHECK(err, err);

    err = clFlush(instance->cq[0]); 
    OCL_ERR_CHECK(err, err);

    err = clFinish(instance->cq[0]); 
    OCL_ERR_CHECK(err, err);

    return CL_SUCCESS;
}

// Perform the initialization steps for all available OpenCL compatible GPUs belonging
// to either Nvidia, AMD or Intel.
extern "C"
EncodeOpenCLInstances* sloth256_189_opencl_init(cl_int& err, 
                                                char* encode_cl, 
                                                char* nvidia_specific_cl, 
                                                char* mod256_189_cu, 
                                                char* non_nvidia_cl) {

    EncodeOpenCLInstances* instances = (EncodeOpenCLInstances*)malloc(sizeof(EncodeOpenCLInstances));

    std::vector<cl_device_id> all_devices = getAllGPUs(err); 
    OCL_ERR_CHECK(err, instances);

    instances->num_instances = all_devices.size();
    instances->instances = (EncodeOpenCLInstance*)malloc(sizeof(EncodeOpenCLInstance) * instances->num_instances);

    size_t k = 0;
    for (int platform_i = 0; platform_i < 3; platform_i++) {
        Platform platform = (Platform)platform_i;
        std::vector<cl_device_id> devices;
        switch (platform) {
            case NVIDIA:
                devices = getAllNvidiaGPUs(err); 
                if (err != SLOTH_NO_OPENCL_COMPATIBLE_NVIDIA_GPUS)
                    OCL_ERR_CHECK(err, instances);
                break;
            case AMD:
                devices = getAllAMDGPUs(err); 
                if (err != SLOTH_NO_OPENCL_COMPATIBLE_AMD_GPUS) 
                    OCL_ERR_CHECK(err, instances);
                break;
            case INTEL:
                devices = getAllIntelGPUs(err); 
                if (err != SLOTH_NO_OPENCL_COMPATIBLE_INTEL_GPUS) 
                    OCL_ERR_CHECK(err, instances);
                break;
            default:
                // Should never happen since functions in opencl_utils.hpp only get 
                // GPUs from only Nvidia, AMD or Intel platforms
                assert(false);
                break;
        }
        err = CL_SUCCESS;

        for (size_t j = 0; j < devices.size(); j++) {
            EncodeOpenCLInstance* instance = instances->instances + k;
            instance->platform = platform;

            instance->device = devices[j];

            instance->context = clCreateContext(NULL, 1, &instance->device, NULL, NULL, &err); 
            OCL_ERR_CHECK(err, instances);

            // Combine the relevant headers and kernel files together into a single string
            std::string header_str = "";
            if (instance->platform == NVIDIA) {
                std::string ptx_str = std::string(mod256_189_cu);
                header_str = ptx_str + "\n" + std::string(nvidia_specific_cl);
            }
            else {
                header_str = std::string(non_nvidia_cl);
            }
            std::string kernel_str = std::string(encode_cl); 
            OCL_ERR_CHECK(err, instances);

            std::string merged_str = header_str + "\n" + kernel_str;
            size_t kernel_size = merged_str.size();
            const char* c_kernel_str = merged_str.c_str();

            instance->program = clCreateProgramWithSource(instance->context, 1, 
                                                          (const char**)&c_kernel_str, 
                                                          (const size_t*)&kernel_size, 
                                                          &err); 
            OCL_ERR_CHECK(err, instances);

            // Compile the OpenCL kernel
            std::string options = "";
            switch (instance->platform) {
                case NVIDIA:
                    err = clBuildProgram(instance->program, 1, &instance->device, options.data(), NULL, NULL); 
                    if (err != CL_SUCCESS) {
                        printProgramBuildLog(err, instance->device, instance->program); 
                        return instances;
                    }
                    break;
                case AMD:
                    size_t amd_wavefront_size;
                    clGetDeviceInfo(instance->device, 
                                    CL_DEVICE_WAVEFRONT_WIDTH_AMD, 
                                    sizeof(size_t), 
                                    (void*)&amd_wavefront_size, NULL);                
                    OCL_ERR_CHECK(err, instances);
                    options += "-D __AMD_GPU__ -D VCC_T=u" + std::to_string(amd_wavefront_size);
                    clBuildProgram(instance->program, 1, &instance->device, options.data(), NULL, NULL); 
                    if (err != CL_SUCCESS) {
                        printProgramBuildLog(err, instance->device, instance->program); 
                        return instances;
                    }
                    break;
                case INTEL:
                    options += "-D __INTEL_GPU__";
                    clBuildProgram(instance->program, 1, &instance->device, options.data(), NULL, NULL); 
                    if (err != CL_SUCCESS) {
                        printProgramBuildLog(err, instance->device, instance->program); 
                        return instances;
                    }
                    break;
                default:
                    // should never happen since functions in opencl_utils.hpp only get 
                    // GPUs from only Nvidia, AMD or Intel platforms
                    assert(false);
                    break;
            }

            instance->kernel = clCreateKernel(instance->program, "sloth256_189_encode_ocl", &err); 
            OCL_ERR_CHECK(err, instances);

            // Nvidia kernel uses dynamically allocated shared memory. We specify the size of it here.
            if (instance->platform == NVIDIA) {
                size_t local_size = 128;
                err = clSetKernelArg(instance->kernel, 3, 7*sizeof(unsigned int)*local_size, NULL); 
                OCL_ERR_CHECK(err, instances);
            }

            // Create multiple command queues (or streams in CUDA terms) to run data transfers and
            // the encoding kernels asynchronously in relation to each other
            for (size_t i = 0; i < MAX_NUM_COMMAND_QUEUES; i++) {
                cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
                instance->cq[i] = clCreateCommandQueueWithProperties(instance->context, 
                                                                     instance->device, 
                                                                     properties, &err); 
                OCL_ERR_CHECK(err, instances);
            }

            k++;
        }
    }

    instances->pinned = NULL;
    instances->c_pinned = NULL;
    instances->pinned_pos = -1;

    return instances;
}

// De-allocate and free all resources allocated to OpenCL in the initialization
extern "C"
cl_int sloth256_189_opencl_cleanup(EncodeOpenCLInstances* instances) {

    cl_int err;

    for (size_t i = 0; i < instances->num_instances; i++) {
        EncodeOpenCLInstance* instance = instances->instances + i;

        err = clReleaseKernel(instance->kernel); 
        OCL_ERR_CHECK(err, err);

        err = clReleaseProgram(instance->program); 
        OCL_ERR_CHECK(err, err);

        for (size_t j = 0; j < MAX_NUM_COMMAND_QUEUES; j++) {
            err = clReleaseCommandQueue(instance->cq[j]); 
            OCL_ERR_CHECK(err, err);
        }

        err = clReleaseContext(instance->context); 
        OCL_ERR_CHECK(err, err);
    }

    free(instances->instances);
    free(instances);

    return CL_SUCCESS;
}

// Perform encoding on the delegated part of data with a single GPU
void sloth256_189_opencl_batch_encode_instance(unsigned char* inout_offset, 
                                               size_t encode_size, 
                                               const unsigned char* iv_offset, 
                                               size_t layers, 
                                               EncodeOpenCLInstance* instance, 
                                               cl_int& err) {

    err = clSetKernelArg(instance->kernel, 2, sizeof(cl_uint), (void*)&layers); 
    if (err != CL_SUCCESS) return;

    // Set the local size (block size in CUDA terms)
    size_t local_size;
    switch (instance->platform) {
        case NVIDIA:
            local_size = 128;
            break;
        case AMD:
            local_size = 64;
            break;
        case INTEL:
            local_size = 64;
            break;
        default:
            // should never happen since functions in opencl_utils.hpp only get 
            // GPUs from only Nvidia, AMD or Intel platforms
            assert(false);
            break;
    }

    // Query the maximum amount of global memory available for a GPU
    cl_ulong device_global_memory;
    err = clGetDeviceInfo(instance->device, 
                          CL_DEVICE_GLOBAL_MEM_SIZE, 
                          sizeof(cl_ulong), 
                          (void*)&device_global_memory, 
                          NULL); 
    if (err != CL_SUCCESS) return;

    // Whether the GPU has a unified memory system with host and itself
    // An Intel GPU can be either integrated (and have unified memory)
    // or discrete (and not have unified memory)
    cl_bool unified_memory = false;
    switch (instance->platform) {
        case NVIDIA:
            unified_memory = false;
            break;
        case AMD:
            unified_memory = true; 
            // CL_DEVICE_HOST_UNIFIED_MEMORY is false, but CL_MEM_USE_HOST_PTR works.
            // is this consistent across all AMD GPUs? other memory copy methods
            // take much longer
            break;
        case INTEL:
            err = clGetDeviceInfo(instance->device, 
                                  CL_DEVICE_HOST_UNIFIED_MEMORY, 
                                  sizeof(cl_bool), 
                                  (void*)&unified_memory, 
                                  NULL); 
            if (err != CL_SUCCESS) return;
            if (unified_memory)                
                device_global_memory = ONE_GB; 
                // if "unified_memory" is true, then it is an Intel integrated GPU
                // using CL_MEM_USE_HOST_PTR is dangerous and might lead to an elusive error since
                // it's not guaranteed for the allocated memory in Rust to be aligned to 4096 bytes
                // we sacrifice an additional 1~ GB of host memory and treat it as a discrete GPU                             
            break;
        default:
            // should never happen since functions in opencl_utils.hpp only get 
            // GPUs from only Nvidia, AMD or Intel platforms
            assert(false);
            break;
    }

    // Code left in to time the encoding of data on the CPU and various different
    // platforms. Uncomment the below line of code in addition to the code in
    // lines 560-564 to enable reporting of GPU encoding times for all GPU platforms available.
    /*auto start = std::chrono::high_resolution_clock::now();*/

    // Start the encoding process
    size_t remaining_size = encode_size, num_command_queues = 1;
    while (remaining_size != 0) {

        // If the remaining size, is larger than the capacity of the GPU's memory,
        // we encode in batches
        size_t next_size;
        if (instance->platform == INTEL && (bool)unified_memory == true) {
        // Intel integrated GPU
            if (remaining_size > device_global_memory) 
                next_size = device_global_memory;
            else 
                next_size = remaining_size;
        }
        else { 
        // Any discrete GPU
            if (remaining_size > device_global_memory)
                // Batch size is in multiples of one GB
                next_size = (device_global_memory / ONE_GB) * ONE_GB; 
            else 
                next_size = remaining_size;
        }

        size_t num_pieces = next_size / 4096;

        size_t copy_offset = encode_size - remaining_size;

        // Perform different encoding steps for AMD GPUs for maximum performance
        // We can just use the memory as if it's zero-copy and let the AMD driver
        // figure how to best do memory transfers.
        if (instance->platform == AMD) {
            // Create a buffer for IVs on device
            cl_mem ivbuf = clCreateBuffer(instance->context, 
                                          CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          next_size / 128, 
                                          (void*)(iv_offset + copy_offset / 128), 
                                          &err); 
            if (err != CL_SUCCESS) return;

            // Create a buffer for pieces on device
            cl_mem piecebuf = clCreateBuffer(instance->context, 
                                             CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
                                             next_size, 
                                             (void*)(inout_offset + copy_offset), 
                                             &err); 
            if (err != CL_SUCCESS) return;

            // Set the kernel arguments as the newly created buffers
            err = clSetKernelArg(instance->kernel, 0, sizeof(unsigned char*), (void*)&piecebuf); 
            if (err != CL_SUCCESS) return;

            err = clSetKernelArg(instance->kernel, 1, sizeof(unsigned char*), (void*)&ivbuf); 
            if (err != CL_SUCCESS) return;

            // Run the encode kernel
            size_t global_size = num_pieces;
            err = clEnqueueNDRangeKernel(instance->cq[0], instance->kernel, 
                                         1, NULL, &global_size, 
                                         &local_size, 0, NULL, NULL);
            if (err != CL_SUCCESS) return;

            // Free the buffers allocated for IVs and pieces
            err = clReleaseMemObject(piecebuf); 
            if (err != CL_SUCCESS) return;

            err = clReleaseMemObject(ivbuf); 
            if (err != CL_SUCCESS) return;

            remaining_size -= next_size;

            continue;
        }

        // Create a buffer for IVs on device
        cl_mem ivbuf = clCreateBuffer(instance->context, CL_MEM_READ_ONLY, next_size / 128, NULL, &err); 
        if (err != CL_SUCCESS) return;

        // Create a buffer for pieces on device
        cl_mem piecebuf = clCreateBuffer(instance->context, CL_MEM_READ_WRITE, next_size, NULL, &err); 
        if (err != CL_SUCCESS) return;

        // Set the kernel arguments as the newly created buffers
        err = clSetKernelArg(instance->kernel, 0, sizeof(unsigned char*), (void*)&piecebuf); 
        if (err != CL_SUCCESS) return;

        err = clSetKernelArg(instance->kernel, 1, sizeof(unsigned char*), (void*)&ivbuf); 
        if (err != CL_SUCCESS) return;

        if (instance->platform == NVIDIA) {
            num_command_queues = MAX_NUM_COMMAND_QUEUES;
        } 
        else {
            // Intel GPU
            num_command_queues = MAX_NUM_COMMAND_QUEUES_INTEL;
        }

        // Calculate the number of pieces per each thread block
        // And determine the number of comamnd queues that will be used
        size_t divided_num_pieces = num_pieces / local_size;
        // Should never happen since the caller Rust 
        // function makes sure that there are more than 1024 pieces
        if (divided_num_pieces == 0) { 
            err = SLOTH_PIECES_NOT_MULTIPLE_OF_1024; 
            return; 
        }
        while (divided_num_pieces % num_command_queues != 0) {
            num_command_queues /= 2;
        }
        size_t divided_size = next_size / num_command_queues;

        // Also should never happen
        if (divided_size < 4096 * num_command_queues) {
            assert(false);
            return;
        }

        size_t global_size = num_pieces / num_command_queues;

        // Copy IVs to the GPU in one go
        err = clEnqueueWriteBuffer(instance->cq[0], ivbuf, CL_TRUE, 0, 
                                   next_size / 128, 
                                   (const void*)(iv_offset + copy_offset / 128), 
                                   0, NULL, NULL); 
        if (err != CL_SUCCESS) return;

        // Copy the piece data to the GPU in multiple iterations
        for (size_t i = 0; i < num_command_queues; i++) {
            err = clEnqueueWriteBuffer(instance->cq[i], piecebuf, CL_FALSE, 
                                       i * divided_size, divided_size, 
                                       (const void*)(inout_offset + divided_size * i + copy_offset), 
                                       0, NULL, NULL); 
            if (err != CL_SUCCESS) return;
        }

        // Run the encode kernel in multiple iterations
        // Note: The encode kernel and the data transfers will run asynchronously
        // in relation to each other since we use multiple comamnd queues

        for (size_t i = 0; i < num_command_queues; i++) {
            size_t offset = i * global_size;
            err = clEnqueueNDRangeKernel(instance->cq[i], 
                                         instance->kernel, 
                                         1, &offset, &global_size, 
                                         &local_size, 0, NULL, NULL); 
            if (err != CL_SUCCESS) return;
        }

        // Copy the piece data back to the host in multiple iterations
        for (size_t i = 0; i < num_command_queues; i++) {
            err = clEnqueueReadBuffer(instance->cq[i], piecebuf, CL_FALSE,
                                      i * divided_size, divided_size,
                                      (void*)(inout_offset + divided_size * i + copy_offset),
                                      0, NULL, NULL);
            if (err != CL_SUCCESS) return;
        }

        // Free the buffers allocated for IVs and pieces
        err = clReleaseMemObject(piecebuf); 
        if (err != CL_SUCCESS) return;

        err = clReleaseMemObject(ivbuf);
        if (err != CL_SUCCESS) return;

        remaining_size -= next_size;
    }

    err = clFlush(instance->cq[num_command_queues - 1]); 
    if (err != CL_SUCCESS) return;

    err = clFinish(instance->cq[num_command_queues - 1]);
    if (err != CL_SUCCESS) return;

    /*auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total elapsed time for GPU encode on platform " 
              << (int)instance->platform << ": " 
              << (std::chrono::duration<double>(end - start)).count() 
              << " seconds" << std::endl;*/
}

// Values that determine how much of the work is delegated to the respective platforms
// For example, on a system where there are only Nvidia GPUs and CPUs, and the CPU and NVIDIA
// factors are 80.0f and 20.0f respectively, 80% of the data will be encoded by the Nvidia GPUs
// and 20% of the data will be encoded by the CPUs
//
// Currently we hand-tune the factors for each system. The commented out codes above time the
// execution of the encode operation on each platform to give us a hint on how to tune the factors
#define CPU_FACTOR 0.75f
#define NVIDIA_FACTOR 100.0f
#define AMD_FACTOR 15.0f
#define INTEL_FACTOR 1.3f

// Note: The work is sent to the CPU directly, not through any OpenCL CPU drivers
// Therefore having or not having OpenCL CPU drivers installed will not have an impact on the encoding
// process

// Delegates work to the CPU and to each OpenCL compatible GPU based on the FACTORS above
// Launches a thread for the CPU and each OpenCL compatible GPU for concurrent queueing
// of commands
extern "C"
cl_int sloth256_189_opencl_batch_encode(unsigned char* inout, 
                                        size_t total_size, 
                                        unsigned char* iv, 
                                        size_t layers, 
                                        EncodeOpenCLInstances* instances) {

    // Create as many error code variables as the amount of OpenCL compatible GPUs
    cl_int* errs = (cl_int*)malloc(sizeof(cl_int) * instances->num_instances);
    for (size_t i = 0; i < instances->num_instances; i++)
        errs[i] = 0;

    // A thread for the CPU and each GPU
    std::vector<std::thread> threads(instances->num_instances + 1);

    // Count the number of GPUs belonging to each platform
    size_t num_nvidia = 0, num_intel = 0, num_amd = 0;
    for (size_t i = 0; i < instances->num_instances; i++) {
        switch (instances->instances[i].platform) {
            case NVIDIA:
                num_nvidia++;
                break;
            case AMD:
                num_amd++;
                break;
            case INTEL:
                num_intel++;
                break;
            default:
                // should never happen since functions in opencl_utils.hpp only get 
                // GPUs from only Nvidia, AMD or Intel platforms
                assert(false);
                break;
        }
    }

    // Calculate how many multiples of 1024 pieces the CPU and each GPU will encode
    // This number is determined based on the ratio between the factor of the platform
    // the GPU belongs to and the sum of all the factors (total_factor) each device of a platform has.
    size_t* sizes = (size_t*)malloc(sizeof(size_t) * (instances->num_instances + 1));
    double div = (double)(total_size / 1024 / 4096);
    double total_factor = CPU_FACTOR + 
                          NVIDIA_FACTOR * num_nvidia + 
                          AMD_FACTOR * num_amd + 
                          INTEL_FACTOR * num_intel;

    // In the case that the total number of pieces was not divided fully,
    // the remaining size is stored in the leftover variable and assigned to
    // the first GPU device present
    size_t leftover = total_size;

    for (size_t i = 0; i < instances->num_instances; i++) {
        switch (instances->instances[i].platform) {
            case NVIDIA:
                sizes[i] = (size_t)(div * NVIDIA_FACTOR / total_factor) * 1024 * 4096;
                break;
            case AMD:
                sizes[i] = (size_t)(div * AMD_FACTOR / total_factor) * 1024 * 4096;
                break;
            case INTEL:
                sizes[i] = (size_t)(div * INTEL_FACTOR / total_factor) * 1024 * 4096;
                break;
            default:
                // should never happen since functions in opencl_utils.hpp only get 
                // GPUs from only Nvidia, AMD or Intel platforms
                assert(false);
                break;
        }
        leftover -= sizes[i];
    }
    sizes[instances->num_instances] = (size_t)(div * CPU_FACTOR / total_factor) * 1024 * 4096;
    leftover -= sizes[instances->num_instances];
    // Assign the leftover to the first device present with assigned size > 0
    bool assigned = false;
    for (size_t i = 0; i < instances->num_instances + 1; i++) {
        if (sizes[i] > 0) {
            sizes[i] += leftover;
            assigned = true;
            break;
        }    
    }
    // If no device is present with assigned size > 0, assign the leftover to the first device
    if (!assigned) {
        sizes[0] += leftover;
    }

    // Encode the delegated amount of data for each GPU in a separate thread
    size_t offset = 0;
    for (size_t i = 0; i < instances->num_instances; i++) {
        if (sizes[i] == 0)
            continue;
        threads[i] = std::thread(sloth256_189_opencl_batch_encode_instance, inout + offset, 
                                 sizes[i], iv + offset / 128, layers, 
                                 instances->instances + i, std::ref(errs[i]));
        offset += sizes[i];
    }

    // Encode the delegated amount of data for the CPU in a separate thread
    if (sizes[instances->num_instances] != 0) {
        threads[instances->num_instances] = std::thread(sloth256_189_cpu_encode_parallel, 
                                                        inout + offset, 
                                                        sizes[instances->num_instances], 
                                                        iv + offset / 128, layers);
    }

    // Wait for all threads to finish their work 
    // i.e wait for the CPU and the GPUs to all 
    // finish encoding their part of the data
    for(std::thread& thread: threads) {
        if (thread.joinable())
            thread.join();
    }

    // Check if there were any OpenCL errors between all
    // the GPU threads
    cl_int final_err = 0;
    for(size_t i = 0; i < instances->num_instances; i++) {
        final_err |= errs[i];
    }

    return final_err;
}
