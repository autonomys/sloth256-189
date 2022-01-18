#ifdef __NV_CL_C_VERSION         										  // dynamically allocated
__kernel void sloth256_189_encode_ocl(__global uchar* inout, __global const uchar* _iv, uint layers, __local uint scratchpad[]) {

	size_t x = get_global_id(0);

    __global uchar* piece = inout + x * 4096;
    __global uchar* iv = _iv + x * 32;

	sloth256_189_encode((limb_t*)piece, 4096, (limb_t*)iv, layers, scratchpad);
}
#else
//#define WORK_GRP_SIZE_X 64
//__attribute__((reqd_work_group_size(WORK_GRP_SIZE_X, 1, 1)))
__kernel void sloth256_189_encode_ocl(__global uchar* inout, __global const uchar* _iv, uint layers) {

    size_t x = get_global_id(0);

    __global uchar* piece = inout + x * 4096;
    __global uchar* iv = _iv + x * 32;

    sloth256_189_encode((__global limb_t*)piece, 4096, (__global limb_t*)iv, layers);
}
#endif
