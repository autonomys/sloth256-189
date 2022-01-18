#include "./opencl_utils.hpp"

#include <algorithm>
#include <omp.h>
#include <thread>
#include <chrono>

extern "C" {
    void sloth256_189_encode(unsigned char*, size_t, const unsigned char*, size_t);
}

void sloth256_189_cpu_encode_parallel(unsigned char* inout, size_t total_size, const unsigned char* iv, size_t layers) {
    size_t num_pieces = total_size / 4096;

    unsigned int num_threads = std::thread::hardware_concurrency();

    //auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(num_threads / 2 - 1)
    for (long long int i = 0; i < (long long int)num_pieces; i++) {
        sloth256_189_encode(inout + 4096 * (size_t)i, 4096, iv + 32 * (size_t)i, layers);
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //std::cout << "Total elapsed time for CPU encode: " << (std::chrono::duration<double>(end - start)).count() << " seconds" << std::endl;
}

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

extern "C"
unsigned char* sloth256_189_pinned_alloc(EncodeOpenCLInstances* instances, size_t size, cl_int& err) {
    EncodeOpenCLInstance* instance = NULL;
    for (size_t i = 0; i < instances->num_instances; i++) {
        if (instances->instances[i].platform == NVIDIA) {
            instance = instances->instances + i;
            instances->pinned_pos = i;
            break;
        }
    }
    if (instance == NULL) {
        err = (cl_int)2035;
        return NULL;
    }

    instances->pinned = clCreateBuffer(instance->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err); OCL_ERR_CHECK(err, instances->c_pinned);
    instances->c_pinned = (unsigned char*)clEnqueueMapBuffer(instance->cq[0], instances->pinned, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, size, 0, NULL, NULL, &err); OCL_ERR_CHECK(err, instances->c_pinned);

    return instances->c_pinned;
}

extern "C"
cl_int sloth256_189_pinned_free(EncodeOpenCLInstances* instances) {
    if (instances->pinned_pos == -1)
        return (cl_int)2036;

    EncodeOpenCLInstance* instance = instances->instances + instances->pinned_pos;

    cl_int err;

    err = clEnqueueUnmapMemObject(instance->cq[0], instances->pinned, (void*)instances->c_pinned, 0, NULL, NULL); OCL_ERR_CHECK(err, err);
    instances->c_pinned = NULL;
    err = clReleaseMemObject(instances->pinned); OCL_ERR_CHECK(err, err);
    err = clFlush(instance->cq[0]); OCL_ERR_CHECK(err, err);
    err = clFinish(instance->cq[0]); OCL_ERR_CHECK(err, err);

    return CL_SUCCESS;
}

extern "C"
EncodeOpenCLInstances* sloth256_189_opencl_init(cl_int& err) {
    EncodeOpenCLInstances* instances = (EncodeOpenCLInstances*)malloc(sizeof(EncodeOpenCLInstances));

    std::vector<cl_device_id> all_devices = getAllGPUs(err); OCL_ERR_CHECK(err, instances);
    instances->num_instances = all_devices.size();
    instances->instances = (EncodeOpenCLInstance*)malloc(sizeof(EncodeOpenCLInstance) * instances->num_instances);

    size_t k = 0;
    for (int platform_i = 0; platform_i < 3; platform_i++) {
        Platform platform = (Platform)platform_i;
        std::vector<cl_device_id> devices;
        switch (platform) {
            case NVIDIA:
                devices = getAllNvidiaGPUs(err); if (err != 2027) OCL_ERR_CHECK(err, instances);
                break;
            case AMD:
                devices = getAllAMDGPUs(err); if (err != 2028) OCL_ERR_CHECK(err, instances);
                break;
            case INTEL:
                devices = getAllIntelGPUs(err); if (err != 2029) OCL_ERR_CHECK(err, instances);
                break;
        }
        err = CL_SUCCESS;

        for (size_t j = 0; j < devices.size(); j++) {
            EncodeOpenCLInstance* instance = instances->instances + k;
            instance->platform = platform;

            instance->device = devices[j];

            instance->context = clCreateContext(NULL, 1, &instance->device, NULL, NULL, &err); OCL_ERR_CHECK(err, instances);

            std::string encode_path = "src/opencl";
            std::string header_str = "";
            if (instance->platform == NVIDIA) {
                std::string ptx_str = read_from_file(encode_path + "/mod256-189.cu", err); OCL_ERR_CHECK(err, instances);
                header_str = ptx_str + "\n" + read_from_file(encode_path + "/nvidia_specific.cl", err); OCL_ERR_CHECK(err, instances);
            }
            else {
                header_str = read_from_file(encode_path + "/non_nvidia.cl", err); OCL_ERR_CHECK(err, instances);
            }
            std::string kernel_str = read_from_file(encode_path + "/encode.cl", err); OCL_ERR_CHECK(err, instances);
            std::string merged_str = header_str + "\n" + kernel_str;
            size_t kernel_size = merged_str.size();
            const char* c_kernel_str = merged_str.c_str();

            instance->program = clCreateProgramWithSource(instance->context, 1, (const char**)&c_kernel_str, (const size_t*)&kernel_size, &err); OCL_ERR_CHECK(err, instances);

            std::string options = "";
            switch (instance->platform) {
                case NVIDIA:
                    err = clBuildProgram(instance->program, 1, &instance->device, options.data(), NULL, NULL); if (err != CL_SUCCESS) {printProgramBuildLog(err, instance->device, instance->program); return instances; }
                    break;
                case AMD:
                    size_t amd_wavefront_size;
                    clGetDeviceInfo(instance->device, CL_DEVICE_WAVEFRONT_WIDTH_AMD, sizeof(size_t), (void*)&amd_wavefront_size, NULL); OCL_ERR_CHECK(err, instances);
                    options += "-D __AMD_GPU__ -D VCC_T=u" + std::to_string(amd_wavefront_size);
                    clBuildProgram(instance->program, 1, &instance->device, options.data(), NULL, NULL); if (err != CL_SUCCESS) {printProgramBuildLog(err, instance->device, instance->program); return instances; }
                    break;
                case INTEL:
                    options += "-D __INTEL_GPU__";
                    clBuildProgram(instance->program, 1, &instance->device, options.data(), NULL, NULL); if (err != CL_SUCCESS) {printProgramBuildLog(err, instance->device, instance->program); return instances; }
                    break;
            }

            instance->kernel = clCreateKernel(instance->program, "sloth256_189_encode_ocl", &err); OCL_ERR_CHECK(err, instances);
            if (instance->platform == NVIDIA) {
                size_t local_size = 128;
                err = clSetKernelArg(instance->kernel, 3, 7*sizeof(unsigned int)*local_size, NULL); OCL_ERR_CHECK(err, instances);
            }

            for (size_t i = 0; i < MAX_NUM_COMMAND_QUEUES; i++) {
                cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
                instance->cq[i] = clCreateCommandQueueWithProperties(instance->context, instance->device, properties, &err); OCL_ERR_CHECK(err, instances);
            }

            k++;
        }
    }

    instances->pinned = NULL;
    instances->c_pinned = NULL;
    instances->pinned_pos = -1;

    return instances;
}

extern "C"
cl_int sloth256_189_opencl_cleanup(EncodeOpenCLInstances* instances) {

    cl_int err;

    for (size_t i = 0; i < instances->num_instances; i++) {
        EncodeOpenCLInstance* instance = instances->instances + i;

        err = clReleaseKernel(instance->kernel); OCL_ERR_CHECK(err, err);
        err = clReleaseProgram(instance->program); OCL_ERR_CHECK(err, err);
        for (size_t j = 0; j < MAX_NUM_COMMAND_QUEUES; j++) {
            err = clReleaseCommandQueue(instance->cq[j]); OCL_ERR_CHECK(err, err);
        }

        err = clReleaseContext(instance->context); OCL_ERR_CHECK(err, err);
    }

    free(instances->instances);
    free(instances);

    return CL_SUCCESS;
}

void sloth256_189_opencl_batch_encode_instance(unsigned char* inout_offset, size_t encode_size, const unsigned char* iv_offset, size_t layers, EncodeOpenCLInstance* instance, cl_int& err) {

    err = clSetKernelArg(instance->kernel, 2, sizeof(cl_uint), (void*)&layers); if (err != CL_SUCCESS) return;

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
    }

    cl_ulong device_global_memory;
    err = clGetDeviceInfo(instance->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), (void*)&device_global_memory, NULL); if (err != CL_SUCCESS) return;

    cl_bool unified_memory = false;
    switch (instance->platform) {
        case NVIDIA:
            unified_memory = false;
            break;
        case AMD:
            unified_memory = true; // CL_DEVICE_HOST_UNIFIED_MEMORY is false, but CL_MEM_USE_HOST_PTR works.
                                   // is this consistent across all AMD GPUs? other memory copy methods
                                   // take much longer
            break;
        case INTEL:
            err = clGetDeviceInfo(instance->device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), (void*)&unified_memory, NULL); if (err != CL_SUCCESS) return;
            if (unified_memory)                // if "unified_memory" is true, then it is an Intel integrated GPU
                device_global_memory = ONE_GB; // using CL_MEM_USE_HOST_PTR is dangerous and might lead to an elusive error since
                                               // it's not guaranteed for the allocated memory in Rust to be aligned to 4096 bytes
                                               // we sacrifice an additional 1~ GB of host memory and treat it as a discrete GPU
            break;
    }

    //auto start = std::chrono::high_resolution_clock::now();

    size_t remaining_size = encode_size, num_command_queues = 1;
    while (remaining_size != 0) {
        size_t next_size;
        if (instance->platform == INTEL && unified_memory == true) { // Intel integrated GPU
            if (remaining_size > device_global_memory) next_size = device_global_memory;
            else next_size = remaining_size;
        }
        else { // Any discrete GPU
            if (remaining_size > device_global_memory) next_size = (device_global_memory / ONE_GB) * ONE_GB; // multiple of 1 GB
            else next_size = remaining_size;
        }

        size_t num_pieces = next_size / 4096;

        size_t copy_offset = encode_size - remaining_size;

        if (instance->platform == AMD) {
            cl_mem ivbuf = clCreateBuffer(instance->context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, next_size / 128, (void*)(iv_offset + copy_offset / 128), &err); if (err != CL_SUCCESS) return;
            cl_mem piecebuf = clCreateBuffer(instance->context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, next_size, (void*)(inout_offset + copy_offset), &err); if (err != CL_SUCCESS) return;

            err = clSetKernelArg(instance->kernel, 0, sizeof(unsigned char*), (void*)&piecebuf); if (err != CL_SUCCESS) return;
            err = clSetKernelArg(instance->kernel, 1, sizeof(unsigned char*), (void*)&ivbuf); if (err != CL_SUCCESS) return;

            size_t global_size = num_pieces;
            err = clEnqueueNDRangeKernel(instance->cq[0], instance->kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); if (err != CL_SUCCESS) return;

            err = clReleaseMemObject(piecebuf); if (err != CL_SUCCESS) return;
            err = clReleaseMemObject(ivbuf); if (err != CL_SUCCESS) return;
            remaining_size -= next_size;
            continue;
        }

        cl_mem ivbuf = clCreateBuffer(instance->context, CL_MEM_READ_ONLY, next_size / 128, NULL, &err); if (err != CL_SUCCESS) return;
        cl_mem piecebuf = clCreateBuffer(instance->context, CL_MEM_READ_WRITE, next_size, NULL, &err); if (err != CL_SUCCESS) return;

        err = clSetKernelArg(instance->kernel, 0, sizeof(unsigned char*), (void*)&piecebuf); if (err != CL_SUCCESS) return;
        err = clSetKernelArg(instance->kernel, 1, sizeof(unsigned char*), (void*)&ivbuf); if (err != CL_SUCCESS) return;

        if (instance->platform == NVIDIA)
            num_command_queues = MAX_NUM_COMMAND_QUEUES;
        else // Intel
            num_command_queues = MAX_NUM_COMMAND_QUEUES_INTEL;

        size_t divided_num_pieces = num_pieces / local_size;
        if (divided_num_pieces == 0) { err = (cl_int)2031; return; } // should never happen since the caller Rust function makes sure that there are more than 1024 pieces
        while (divided_num_pieces % num_command_queues != 0) {
            num_command_queues /= 2;
        }
        size_t divided_size = next_size / num_command_queues;
        if (divided_size < 4096 * num_command_queues) { err = (cl_int)2032; return; } // also should never happen

        size_t global_size = num_pieces / num_command_queues;

        err = clEnqueueWriteBuffer(instance->cq[0], ivbuf, CL_TRUE, 0, next_size / 128, (const void*)(iv_offset + copy_offset / 128), 0, NULL, NULL); if (err != CL_SUCCESS) return;

        for (size_t i = 0; i < num_command_queues; i++) {
            err = clEnqueueWriteBuffer(instance->cq[i], piecebuf, CL_FALSE, i * divided_size, divided_size, (const void*)(inout_offset + divided_size * i + copy_offset), 0, NULL, NULL); if (err != CL_SUCCESS) return;
        }

        //cl_event event; double duration = 0;
        for (size_t i = 0; i < num_command_queues; i++) {
            size_t offset = i * global_size;
            err = clEnqueueNDRangeKernel(instance->cq[i], instance->kernel, 1, &offset, &global_size, &local_size, 0, NULL, NULL); if (err != CL_SUCCESS) return;
            //duration += getEventElapsedTime(event, instance->cq[i]);
        }
        //printf("Elapsed time for encode kernel: %0.f milliseconds\n", duration / 1000000);

        for (size_t i = 0; i < num_command_queues; i++) {
            err = clEnqueueReadBuffer(instance->cq[i], piecebuf, CL_FALSE, i * divided_size, divided_size, (void*)(inout_offset + divided_size * i + copy_offset), 0, NULL, NULL); if (err != CL_SUCCESS) return;
        }

        err = clReleaseMemObject(piecebuf); if (err != CL_SUCCESS) return;
        err = clReleaseMemObject(ivbuf); if (err != CL_SUCCESS) return;

        remaining_size -= next_size;
    }

    err = clFlush(instance->cq[num_command_queues - 1]); if (err != CL_SUCCESS) return;
    err = clFinish(instance->cq[num_command_queues - 1]); if (err != CL_SUCCESS) return;

    //auto end = std::chrono::high_resolution_clock::now();
    //std::cout << "Total elapsed time for GPU encode on platform " << (int)instance->platform << ": " << (std::chrono::duration<double>(end - start)).count() << " seconds" << std::endl;
}

#define CPU_FACTOR 0.75f
#define NVIDIA_FACTOR 100.0f
#define AMD_FACTOR 15.0f
#define INTEL_FACTOR 1.3f

extern "C"
cl_int sloth256_189_opencl_batch_encode(unsigned char* inout, size_t total_size, unsigned char* iv, size_t layers, EncodeOpenCLInstances* instances) {

    cl_int* errs = (cl_int*)malloc(sizeof(cl_int) * instances->num_instances);
    for (size_t i = 0; i < instances->num_instances; i++)
        errs[i] = 0;

    std::vector<std::thread> threads(instances->num_instances + 1);

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
        }
    }

    size_t* sizes = (size_t*)malloc(sizeof(size_t) * (instances->num_instances + 1));
    double div = total_size / 1024 / 4096;
    double total_factor = CPU_FACTOR + NVIDIA_FACTOR * num_nvidia + AMD_FACTOR * num_amd + INTEL_FACTOR * num_intel;
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
        }
        leftover -= sizes[i];
    }
    sizes[instances->num_instances] = (size_t)(div * CPU_FACTOR / total_factor) * 1024 * 4096;
    leftover -= sizes[instances->num_instances];
    sizes[0] += leftover;

    size_t offset = 0;
    for (size_t i = 0; i < instances->num_instances; i++) {
        if (sizes[i] == 0)
            continue;
        threads[i] = std::thread(sloth256_189_opencl_batch_encode_instance, inout + offset, sizes[i], iv + offset / 128, layers, instances->instances + i, std::ref(errs[i]));
        offset += sizes[i];
    }

    if (sizes[instances->num_instances] != 0) {
        threads[instances->num_instances] = std::thread(sloth256_189_cpu_encode_parallel, inout + offset, sizes[instances->num_instances], iv + offset / 128, layers);
    }

    for(std::thread& thread: threads) {
        if (thread.joinable())
            thread.join();
    }

    cl_int final_err = 0;
    for(size_t i = 0; i < instances->num_instances; i++) {
        final_err |= errs[i];
    }

    return final_err;
}
