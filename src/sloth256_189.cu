#include <cuda.h>

__global__ void test_1x1_kernel(unsigned char *piece_n_iv, size_t len,
                                size_t layers);
#ifdef __CUDA_ARCH__
# include "sloth256_189.c"

__global__ void test_1x1_kernel(unsigned char *piece_n_iv, size_t len,
                                size_t layers)
{
    (void)sloth256_189_encode(piece_n_iv, len, piece_n_iv+len, layers);
}
#endif

extern "C" bool test_1x1_cuda(unsigned char piece[], size_t len,
                              const unsigned char iv[32], size_t layers)
{
    unsigned char *piece_n_iv;

    if (cudaMalloc(&piece_n_iv, len+32) != cudaSuccess)
        return false;

    cudaMemcpy(piece_n_iv, piece, len, cudaMemcpyHostToDevice);
    cudaMemcpy(piece_n_iv+len, iv, 32, cudaMemcpyHostToDevice);

    test_1x1_kernel<<<1, 1>>>(piece_n_iv, len, layers);

    if (cudaDeviceSynchronize() == cudaSuccess)
        cudaMemcpy(piece, piece_n_iv, len, cudaMemcpyDeviceToHost);

    cudaFree(piece_n_iv);

    return true;
}

extern "C" bool detect_cuda()
{
    cudaDeviceProp prop;
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess;
}

#ifdef STANDALONE_DEMO
# include <cstdint>

__global__ void demo(unsigned char *flat, size_t layers);
__device__ uint64_t gpu_ticks;

# ifdef __CUDA_ARCH__
__global__ void demo(unsigned char *flat, size_t layers)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;

    unsigned char *iv = flat + (64+4096)*x;
    unsigned char *piece = iv + 64;

    uint64_t start, end;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    sloth256_189_encode(piece, 4096, iv, layers);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (x == 0) gpu_ticks = end - start;
}
# else
#  include <iostream>
using namespace std;

int main(int argc, const char *argv[])
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties(&prop, 0) failed\n");
        exit(1);
    }
    cout << prop.name << endl;
    cout << "Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Clock rate: " << prop.clockRate << "kHz" << endl;
    cout << "Memory clock rate: " << prop.memoryClockRate << "kHz" << endl;
    cout << "L2 cache size: " << prop.l2CacheSize << endl;
    cout << "Shared Memory: " << prop.sharedMemPerBlock << endl;

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, demo);
    cout << "kernel<<<" << minGridSize << ", " << blockSize << ">>>, ";

    size_t layers = 1;
    if (argc > 3)
        layers = atoi(argv[3]);

    if (argc > 2) {
        int x;
        if ((x = atoi(argv[2])) > 0) blockSize = x;
        if ((x = atoi(argv[1])) > 0) minGridSize = x;
    } else if (argc > 1) {
        int x;
        if ((x = atoi(argv[1])) > 0) minGridSize = x;
    } else {
        blockSize = 32;
        minGridSize = 1;
    }
    cout << "starting <<<" << minGridSize << ", " << blockSize << ">>>" << endl;

    size_t sz = minGridSize * blockSize * (64+4096);
    unsigned char *flat;

    if (!prop.integrated) {
        cudaMalloc(&flat, sz);
        cudaMemset(flat, 5, sz);
        cudaMemset(flat, 3, 32);
    } else {
        cudaMallocManaged(&flat, sz);
        memset(flat, 5, sz);
        memset(flat, 3, 32);
    }

    demo<<<minGridSize, blockSize>>>(flat, layers);

    cudaDeviceSynchronize();

    cudaFree(flat);

    uint64_t ticks;
    cudaMemcpyFromSymbol(&ticks, gpu_ticks, sizeof(ticks));
    cout << "sloth256_189_encode: " << ticks << endl;
}
# endif
#endif
