// Two different versions of the encode kernel

// First one is for Nvidia where shared memory is used
// The second one is for Intel Integrated GPU and AMD GPU which doesn't
// use shared memory

#ifdef __NV_CL_C_VERSION
__kernel void sloth256_189_encode_ocl(__global uchar* inout,
                                      __global const uchar* _iv,
                                      uint layers, 
                                      // dynamically allocated local memory
                                      // shared in CUDA terms
                                      __local uint scratchpad[]) {

    size_t x = get_global_id(0);

    __global uchar* piece = inout + x * 4096;
    __global uchar* iv = _iv + x * 32;

    sloth256_189_encode((limb_t*)piece, 4096, (limb_t*)iv, layers, scratchpad);
}
#else
__kernel void sloth256_189_encode_ocl(__global uchar* inout,
                                      __global const uchar* _iv,
                                      uint layers) {
                                          
    size_t x = get_global_id(0);

    __global uchar* piece = inout + x * 4096;
    __global uchar* iv = _iv + x * 32;

    sloth256_189_encode((__global limb_t*)piece, 4096, 
                        (__global limb_t*)iv, layers);
}
#endif
