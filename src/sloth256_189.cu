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

#define STANDALONE_DEMO  // required if we want to compile .cu file, this is letting the below code become an 'entry' point
#ifdef STANDALONE_DEMO
# include <cstdint>

__global__ void demo(unsigned char *flat, size_t layers);
__device__ uint64_t gpu_ticks;

# ifdef __CUDA_ARCH__  // unnecessary for the trial of a standalone .cu compilation
__global__ void demo(unsigned char *flat, size_t layers)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;  // global ID of threads stored in `int x`

    unsigned char *iv = flat + (64+4096)*x;  // getting the respective IV from the flattened array
    unsigned char *piece = iv + 64;  // although IV is 32 bytes, we allocated 64 bytes for padding
    // so we have to add 64 to our pointer to move it to the beginning of the piece

    uint64_t start, end;  // for timing

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));  // for timing, done in assembly
    sloth256_189_encode(piece, 4096, iv, layers);  // computation of actual encoding is done in here
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));  // for timing, done in assembly
    if (x == 0) gpu_ticks = end - start;  // only the first thread computes this
}
# else
#  include <iostream>
using namespace std;

// unnecessary for the trial of a standalone .cu compilation - BEGIN
#define CUDA_FATAL(expr) do {				\
    cudaError_t code = expr;				\
    if (code != cudaSuccess) {				\
        cerr << #expr << "@" << __LINE__ << " failed: "	\
             << cudaGetErrorString(code) << endl;	\
	exit(1);					\
    }							\
} while(0)
// // unnecessary for the trial of a standalone .cu compilation - END
int main(int argc, const char *argv[])
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
                                                  demo));  // creating problem with nsight, can remove this part also

    cout << "kernel<<<" << minGridSize << ", " << blockSize << ">>>, ";  // shows the parameters for max-occupancy

    size_t layers = 1;
    // unnecessary if we are not going to use command-line arguments - BEGIN
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
    // unnecessary if we are not going to use command-line arguments - BEGIN
    cout << "starting <<<" << minGridSize << ", " << blockSize << ">>>" << endl;  // shows the parameters for actual kernel run

    size_t sz = minGridSize * blockSize * (64+4096);  // 64 for IV, 4096 for piece, multiplied for each thread
    unsigned char *flat;


    // creates problem in nsight, just include the code in `else` part, and remove everything else from the below block
    // below block - START
    if (!prop.integrated) {
        CUDA_FATAL(cudaMalloc(&flat, sz));
        cudaMemset(flat, 5, sz);
        cudaMemset(flat, 3, 32);
    } else {
        CUDA_FATAL(cudaMallocManaged(&flat, sz));
        memset(flat, 5, sz);
        memset(flat, 3, 32);
    }
    // below block - END
    // here is the code inside the else block extracted as nsight safe version
    /*
    (cudaMallocManaged(&flat, sz));
    memset(flat, 5, sz);
    memset(flat, 3, 32);
    */

    demo<<<minGridSize, blockSize>>>(flat, layers);

    cudaDeviceSynchronize();

    cudaFree(flat);

    uint64_t ticks;  // for timing
    cudaMemcpyFromSymbol(&ticks, gpu_ticks, sizeof(ticks)); // copies tick amount to host
    cout << "sloth256_189_encode: " << ticks << endl;  // printing ticks in host
}
# endif
#endif
