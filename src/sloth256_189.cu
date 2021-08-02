#include <cuda.h>

#include "sloth256_189.c"

__global__ void test_1x1_kernel(unsigned char *piece_n_iv, size_t len,
                                size_t layers)
{
    (void)sloth256_189_encode(piece_n_iv, len, piece_n_iv+len, layers);
}

extern "C" void test_1x1_cuda(unsigned char piece[], size_t len,
                              const unsigned char iv[32], size_t layers)
{
    unsigned char *piece_n_iv;

    cudaMalloc(&piece_n_iv, len+32);
    cudaMemcpy(piece_n_iv, piece, len, cudaMemcpyHostToDevice);
    cudaMemcpy(piece_n_iv+len, iv, 32, cudaMemcpyHostToDevice);

    test_1x1_kernel<<<1, 1>>>(piece_n_iv, len, layers);

    cudaDeviceSynchronize();

    cudaMemcpy(piece, piece_n_iv, len, cudaMemcpyDeviceToHost);
}

extern "C" bool detect_cuda()
{
    cudaDeviceProp prop;
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess;
}
