#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <bitset>
#include <string>

#include "encode_ptx.h"

using namespace std;

#define CUDA_FATAL(expr) do {				\
    cudaError_t code = expr;				\
    if (code != cudaSuccess) {				\
        cerr << #expr << "@" << __LINE__ << " failed: "	\
             << cudaGetErrorString(code) << endl;	\
	exit(1);					\
    }							\
} while(0)


extern "C" bool detect_cuda()
{
    cudaDeviceProp prop;
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess;
}

extern "C" bool test_batches(unsigned int piece[], size_t len,
                             const unsigned int iv[32], size_t layers)
{
    int block_amount, thread_amount;
    unsigned remaining_piece_size = len;  // there is `size` in the variable name, since len does not represent
    // the piece count, but instead the size of the piece_array (in bytes)
    unsigned processed_piece_size = 0;  // in bytes

    // getting the optimal amount of blocks and threads for throughput
    CUDA_FATAL(cudaOccupancyMaxPotentialBlockSize(&block_amount, &thread_amount, sloth256_189_encode));
    // this actually returns the minimum required block_amount for achieving maximum occupancy.
    // So we can tweak the block_amount up, but it is best to not touch thread_amount

    block_amount = 262144;  // 2**18 = 262144 => 1GB / 4KB
    // by setting block_amount to 262144, we are going to process 1GB of data per round
    // 1GB should be ok for the most of the GPUs out there.
    // this can be tweaked with respect to the GPU model in the future
    // but we should also consider what happens if user is using another app,
    // and does not have enough memory in their GPU for our assumption

    unsigned to_be_processed_size = (block_amount * thread_amount) << 12;  // total worker * piece_size (4096)
    // instead of multiplying with 4096, we can shift by 12
    // this is the size of pieces to be processed (in bytes)

    u256* d_piece;
    u256* d_iv;

    while (true)
    {
        // it could be that, remaining_piece_size could be less than the optimal amount of work to be processed
        if (remaining_piece_size < (to_be_processed_size))  // so we will adjust our worker amount accordingly
        {
            block_amount = remaining_piece_size / thread_amount;
            // important note in here: the above division should not produce a remainder
            // `thread_amount` will be at most 1024, so during load balancing, send multiples of
            // 1024 pieces to the GPU, to guarantee that the above division will not have any remainder
            // it is not a good idea to play with thread_amount, since these should be multiple of 32

            to_be_processed_size = (block_amount * thread_amount) << 12;  // update our variable
        }

        cudaMalloc(&d_piece, to_be_processed_size);
        cudaMalloc(&d_iv, (to_be_processed_size >> 7));  // iv occupies 32 bytes, piece occupies 4096 bytes.
        // Instead of dividing len into 4096, then multiplying it with 32, we can simply shift by 7.

        // computing the next range of pieces to be processed, and copying them to GPU memory
        cudaMemcpy(d_piece, (piece + (processed_piece_size >> 2)), to_be_processed_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_iv, (iv + (processed_piece_size >> 9)), (to_be_processed_size >> 7), cudaMemcpyHostToDevice);
        // the reason for the extra shift by 2 is:
        // we are doing pointer arithmetic here. Type of `piece` and `iv` are unsigned int, and unsigned int
        // is allocating 4 bytes. So actually, iv+1 reaches to next unsigned int, which is 4 bytes later
        // and we have computed the actual size. We have to divide our computations by 4 in here

        sloth256_189_encode<<<block_amount, thread_amount>>>(d_piece, d_iv);  // calling the kernel

        if (cudaDeviceSynchronize() == cudaSuccess)
        {
            cudaMemcpy((piece + (processed_piece_size >> 2)), d_piece, to_be_processed_size, cudaMemcpyDeviceToHost);
            // copy back the result to host
        }

        processed_piece_size += to_be_processed_size;  // update the processed_piece_size
        remaining_piece_size -= to_be_processed_size;  // likewise :)

        if (remaining_piece_size == 0)
        {
            cudaFree(d_piece);  // clean-up
            cudaFree(d_iv);  // clean-up

            return true;  // escape from the loop, and end the function
        }
    }
}


extern "C" bool test_1x1_cuda(unsigned int piece[], size_t len,
                              const unsigned int iv[32], size_t layers)
{
    u256* d_piece;
    u256* d_iv;

    cudaMalloc(&d_piece, len);
    cudaMalloc(&d_iv, 32);

    cudaMemcpy(d_piece, piece, len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, iv, 32, cudaMemcpyHostToDevice);

    sloth256_189_encode<<<1, 1 >>>(d_piece, d_iv);  // calling the kernel

    if (cudaDeviceSynchronize() == cudaSuccess)
        cudaMemcpy(piece, d_piece, len, cudaMemcpyDeviceToHost);

    cudaFree(d_piece);
    cudaFree(d_iv);

    return true;
}




#ifdef STANDALONE_DEMO
int main()
{
// creating problem with nsight, can remove this part - BEGIN
    cudaDeviceProp prop;
    CUDA_FATAL(cudaGetDeviceProperties(&prop, 0));
    cout << prop.name << endl;
    cout << "Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Clock rate: " << prop.clockRate << "kHz" << endl;
    cout << "Memory clock rate: " << prop.memoryClockRate << "kHz" << endl;
    cout << "L2 cache size: " << prop.l2CacheSize << endl;
    cout << "Shared Memory: " << prop.sharedMemPerBlock << endl;
    // creating problem with nsight, can remove this part - END


    // instead of below, we can give any number to blockSize and minGridSize like this:
    /*
    int blockSize = 256;
    int minGridSize = 30;
    */

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch
    CUDA_FATAL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                                  encode_ptx_test));  // creating problem with nsight, can remove this part also

    cout << "kernel<<<" << minGridSize << ", " << blockSize << ">>>, \n";  // shows the parameters for max-occupancy

	u32* piece = (u32*)malloc(sizeof(u32) * 8 * 128 * minGridSize * blockSize);  // allocates memory on the CPU for the piece
	u32* d_piece_ptx, * d_expanded_iv_ptx;  // creating device pointers

    cudaMalloc(&d_piece_ptx, sizeof(u32) * 8 * 128 * minGridSize * blockSize);  // allocates memory on the GPU for the piece
    cudaMalloc(&d_expanded_iv_ptx, sizeof(u32) * 8 * minGridSize * blockSize);  // allocates memory on the GPU for the expanded_iv
	// since expanded_iv will be static for a farmer, this does not need to be copied from CPU everytime, it can be hardcoded to GPU

    cudaMemset(d_piece_ptx, 5u, sizeof(u32) * 8 * 128 * minGridSize * blockSize);  // setting all values inside piece as 5
    cudaMemset(d_expanded_iv_ptx, 3u, sizeof(u32) * 8 * minGridSize * blockSize);  // setting all values inside expanded_iv as 3

    encode_ptx_test<<<minGridSize, blockSize >>>(d_piece_ptx, d_expanded_iv_ptx);  // calling the kernel

    cudaMemcpy(piece, d_piece_ptx, sizeof(u32) * 8 * 128 * minGridSize * blockSize, cudaMemcpyDeviceToHost);  // copying the result back to CPU

	cudaDeviceSynchronize();  // wait for GPU operations to finish

    cout << "Operation successful!\n";

	// FOR DEBUGGING THE OUTPUT (prints the piece in hexadecimal)
	/*unsigned char* piece_byte_ptr = (unsigned char*)piece;
	for (int i = 0; i < 128 * 32; i++)
	{
		unsigned number = (unsigned)piece_byte_ptr[i];

		if (number == 0)
		{
			cout << "00";
		}
		else if (number < 16)
		{
			cout << "0";
			cout << hex << number;
		}
		else
		{
			cout << hex << number;
		}

		if (i % 32 == 31)
			cout << endl;
	}*/

    return 0;
}
#endif
