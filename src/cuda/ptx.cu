#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <bitset>
#include <string>

#include "encode_ptx.h"

#define NUM_THREADS 256
#define NUM_BLOCKS 1024
#define EIGHT_GB_IN_BYTES 8589934592


extern "C" bool is_cuda_available()
{
    cudaDeviceProp prop;
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess;
}

extern "C" int sloth256_189_cuda_batch_encode(unsigned int piece[], size_t len,
                             const unsigned int iv[32], size_t layers)
{   // len also represents how many bytes in piece[]

    int cudaStatus;  // for handling potential CUDA errors

    int block_count, thread_count;
    unsigned remaining_piece_size = len;  // there is `size` in the variable name, since len does not represent
    // the piece count, but instead the size of the piece_array (in bytes)
    unsigned processed_piece_size = 0;  // in bytes

    thread_count = NUM_THREADS;  // 1 thread is responsible from 1 piece,
    // 1 thread handles 4096 bytes
    // 256 threads handle 1048576 bytes, or 2**20 bytes

    unsigned long long default_round_size = EIGHT_GB_IN_BYTES;  // 8GB as Bytes
    // allocating more than 8GB would be overkill, this is an upper-limit set for high-end GPUs.
    // we will tweak this down below with respect to the current available device.

    unsigned long long to_be_processed_size, free_mem, total_mem;

    // Getting free and total memory of the device
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
      return 1;
    }

    //printf("\nFree memory on this device is: %llu Bytes\n", free_mem);
    //printf("Total memory on this device is: %llu Bytes\n", total_mem);  // we are not using this, but it is fancy :)

    while (default_round_size > free_mem) {  // if device does not have enough free memory
        default_round_size /= 2;  // make the memory requirement smaller
    }

    //printf("Picked the default amount of memory to be allocated in each round as: %llu Bytes\n", default_round_size);

    block_count = (default_round_size / 4096) / thread_count;  // we want to keep thread_count at 256 for CUDA reasons
    // so we are manipulating block_count instead.
    // (to_be_processed_size >> 12) -> (to_be_processed_size / 4096) -> piece_count
    // (piece_count / thread_count) -> how many blocks there should be

    u256 *d_piece, *d_iv;  // pointers for device

    while (true) {  // don't panic, this is not an endless loop :)
        to_be_processed_size = default_round_size;  // at the start of the each turn, use the default size

        // it could be that, remaining_piece_size could be less than the default size
        if (remaining_piece_size < (to_be_processed_size)) {  // so we will adjust our worker count accordingly

            block_count = (remaining_piece_size / 4096) / thread_count;
            // since each thread will handle 4096 bytes, the above equation should make sense
            // important note in here: the above division should not produce a remainder
            // `thread_count` will be 256. During load balancing, send multiples of
            // 256 pieces to the GPU to be safe, so that the above division will not have any remainder

            to_be_processed_size = block_count * thread_count * 4096;  // update our variable
        }

        //printf("Trying to allocate %llu Bytes\n", to_be_processed_size);
        cudaStatus = cudaMalloc(&d_piece, to_be_processed_size);  // trying to allocate memory
        // this might fail, due to User may have opened a program that heavily utilizes the GPU

        while ((cudaStatus != cudaSuccess) && to_be_processed_size != 0) {  // if fails, reduce the requirement
            cudaStatus = cudaMalloc(&d_piece, to_be_processed_size);
            to_be_processed_size /= 2;
            block_count /= 2;
        }

        if (to_be_processed_size == 0) {  // unfortunately, cudaMalloc does not return an error when size is 0
            cudaStatus = 2;  // cudaMalloc failed!
            break;
        }

        cudaStatus = cudaMalloc(&d_iv, (to_be_processed_size / 128 ));
        // iv occupies 32 bytes, piece occupies 4096 bytes.
        // Instead of dividing the size into 4096, then multiplying it with 32, we can divide into 128.
        if (cudaStatus != cudaSuccess) {
            cudaStatus = 3;  // cudaMalloc failed!
            break;
        }

        // computing the next range of pieces to be processed, and copying them into GPU memory
        cudaStatus = cudaMemcpy(d_piece, (piece + (processed_piece_size / 4)),
                                to_be_processed_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            cudaStatus = 4;  // cudaMemcpy failed!
            break;
        }
        cudaStatus = cudaMemcpy(d_iv, (iv + (processed_piece_size / 512)),
                                (to_be_processed_size / 128), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            cudaStatus = 5;  // cudaMemcpy failed!
            break;
        }
        // the reason for the extra division by 4 is:
        // we are doing pointer arithmetic here. Type of `piece` and `iv` are unsigned int, and unsigned int
        // is allocating 4 bytes. So actually, iv+1 reaches to next unsigned int, which is 4 bytes later
        // and we have computed the actual size. We have to divide our computations by 4 in here

        sloth256_189_encode_cuda<<<block_count, thread_count>>>(d_piece, d_iv);  // calling the kernel

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            // Kernel launch failed
            cudaStatus = 6;
            break;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            // cudaDeviceSynchronize returned error code %d, CUDA operation did not finish correctly,
            // returning back to CPU...
            cudaStatus = 7;
            break;
        }

        cudaStatus = cudaMemcpy((piece + (processed_piece_size / 4)), d_piece,
                                to_be_processed_size, cudaMemcpyDeviceToHost);
        // copy back the result to host, again extra division by 4 because of pointer arithmetic
        if (cudaStatus != cudaSuccess) {
            cudaStatus = 8;  // cudaMemcpy failed!
            break;
        }

        processed_piece_size += to_be_processed_size;  // update the processed_piece_size
        remaining_piece_size -= to_be_processed_size;  // likewise :)

        if (remaining_piece_size == 0) {  // successful!
            break;  // Hurry! Get out of the loop
        }
    }

    cudaFree(d_piece);  // clean-up
    cudaFree(d_iv);  // clean-up

    return cudaStatus;  // cudaStatus is 0 if there is no error, 1 if there is error
    // and other numbers for specific errors that we inspected for
}
