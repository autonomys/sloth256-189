#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <bitset>
#include <string>

#include "encode_ptx.h"

#define NUM_THREADS 256
#define NUM_BLOCKS 1024


extern "C" bool is_cuda_available()
{
    cudaDeviceProp prop;
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess;
}

extern "C" int sloth256_189_cuda_batch_encode(unsigned int piece[], size_t len,
                             const unsigned int iv[32], size_t layers)
{
    // for handling potential CUDA errors
    int cudaStatus;

    size_t block_count, thread_count;
    // there is `size` in the variable name, since len does not represent
    // the piece count, but instead the size of the piece_array (in bytes)
    size_t remaining_piece_size = len;
    // in bytes
    size_t processed_piece_size = 0;

    // 1 thread is responsible from 1 piece,
    // 1 thread handles 4096 bytes
    // 256 threads handle 1048576 bytes, or 2**20 bytes
    thread_count = NUM_THREADS;

    // 8GB as Bytes
    // allocating more than 8GB would be overkill, this is an upper-limit set for high-end GPUs.
    // we will tweak this down below with respect to the current available device.
    unsigned long long round_size = len;

    size_t free_mem, total_mem;

    // Getting free and total memory of the device
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        return 1;
    }

    //printf("\nFree memory on this device is: %llu Bytes\n", free_mem);
    //printf("Total memory on this device is: %llu Bytes\n", total_mem);  // we are not using this, but it is fancy :)

    // if device does not have enough free memory
    while (round_size > free_mem) {
        // make the memory requirement smaller
        round_size /= 2;
    }

    // Unfortunately, cudaMalloc does not return an error when size is 0
    if (round_size == 0) {
        return 2;
    }
    //printf("Picked the default amount of memory to be allocated in each round as: %llu Bytes\n", round_size);

    // pointers for device
    u256 *d_piece = 0;
    u256 *d_iv = 0;

    // We want to keep thread_count at 256 for CUDA reasons so we are manipulating block_count instead.
    // (round_size >> 12) -> (round_size / 4096) -> piece_count
    // (piece_count / thread_count) -> how many blocks there should be
    block_count = (round_size / 4096) / thread_count;

    //printf("Trying to allocate %llu Bytes\n", round_size);
    // This might fail, due to User may have opened a program that heavily utilizes the GPU
    cudaStatus = cudaMalloc(&d_piece, round_size);

    // If fails, reduce the requirement
    while (cudaStatus != cudaSuccess) {
        cudaStatus = cudaMalloc(&d_piece, round_size);
        round_size /= 2;
        block_count /= 2;
    }

    // IV occupies 32 bytes, piece occupies 4096 bytes.
    cudaStatus = cudaMalloc(&d_iv, (round_size / 4096 * 32 ));

    if (cudaStatus != cudaSuccess) {
        return 3;
    }

    while (true) {
        // Computing the next range of pieces to be processed, and copying them into GPU memory
        //
        // The reason for the extra division by 4 is: we are doing pointer arithmetic here. Type of `piece` and `iv`
        // are unsigned int, and unsigned int is allocating 4 bytes. So actually, iv+1 reaches to next unsigned int,
        // which is 4 bytes later and we have computed the actual size. We have to divide our computations by 4 in here.
        cudaStatus = cudaMemcpy(d_piece, (piece + (processed_piece_size / 4)),
                                round_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            cudaStatus = 4;
            break;
        }
        cudaStatus = cudaMemcpy(d_iv, (iv + (processed_piece_size / 512)),
                                (round_size / 128), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            cudaStatus = 5;
            break;
        }

         // Calling the kernel, we cast (unsigned int) to suppress warning of possible data loss
        sloth256_189_encode_cuda<<<(unsigned int)block_count, (unsigned int)thread_count>>>(d_piece, d_iv, layers);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            cudaStatus = 6;
            break;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            cudaStatus = 7;
            break;
        }

        cudaStatus = cudaMemcpy((piece + (processed_piece_size / 4)), d_piece,
                                round_size, cudaMemcpyDeviceToHost);
        // Copy back the result to host, again extra division by 4 because of pointer arithmetic
        if (cudaStatus != cudaSuccess) {
            cudaStatus = 8;
            break;
        }

        processed_piece_size += round_size;
        remaining_piece_size -= round_size;

        if (remaining_piece_size == 0) {
            break;
        }
    }

    // cudaStatus is 0 if there is no error and other numbers for specific errors that we inspected for
    return cudaStatus;
}
