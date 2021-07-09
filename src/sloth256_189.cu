#include <cuda.h>

extern "C" bool detect_cuda()
{
    cudaDeviceProp prop;
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess;
}
