#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned int u32;
typedef u32 u256[8];

__device__ __forceinline__ void mul_reduce_256(u256 out, const u256 x, const u256 y)
{
	u256 temp_hi;

	// x * y
	asm("{\n\t"

		// x * y0
		"mul.lo.u32      %0, %16, %24;"
		"mul.hi.u32      %1, %16, %24;"
		"mul.lo.u32      %2, %18, %24;"
		"mul.hi.u32      %3, %18, %24;"
		"mul.lo.u32      %4, %20, %24;"
		"mul.hi.u32      %5, %20, %24;"
		"mul.lo.u32      %6, %22, %24;"
		"mul.hi.u32      %7, %22, %24;"

		"mad.lo.cc.u32   %1, %17, %24, %1;"
		"madc.hi.cc.u32  %2, %17, %24, %2;"
		"madc.lo.cc.u32  %3, %19, %24, %3;"
		"madc.hi.cc.u32  %4, %19, %24, %4;"
		"madc.lo.cc.u32  %5, %21, %24, %5;"
		"madc.hi.cc.u32  %6, %21, %24, %6;"
		"madc.lo.cc.u32  %7, %23, %24, %7;"
		"madc.hi.u32     %8, %23, %24, 0;"

		// x * y1
		"mad.lo.cc.u32   %1, %16, %25, %1;"
		"madc.hi.cc.u32  %2, %16, %25, %2;"
		"madc.lo.cc.u32  %3, %18, %25, %3;"
		"madc.hi.cc.u32  %4, %18, %25, %4;"
		"madc.lo.cc.u32  %5, %20, %25, %5;"
		"madc.hi.cc.u32  %6, %20, %25, %6;"
		"madc.lo.cc.u32  %7, %22, %25, %7;"
		"madc.hi.cc.u32  %8, %22, %25, %8;"
		"addc.u32        %9, 0, 0;"

		"mad.lo.cc.u32   %2, %17, %25, %2;"
		"madc.hi.cc.u32  %3, %17, %25, %3;"
		"madc.lo.cc.u32  %4, %19, %25, %4;"
		"madc.hi.cc.u32  %5, %19, %25, %5;"
		"madc.lo.cc.u32  %6, %21, %25, %6;"
		"madc.hi.cc.u32  %7, %21, %25, %7;"
		"madc.lo.cc.u32  %8, %23, %25, %8;"
		"madc.hi.u32     %9, %23, %25, %9;"

		// x * y2
		"mad.lo.cc.u32   %2, %16, %26, %2;"
		"madc.hi.cc.u32  %3, %16, %26, %3;"
		"madc.lo.cc.u32  %4, %18, %26, %4;"
		"madc.hi.cc.u32  %5, %18, %26, %5;"
		"madc.lo.cc.u32  %6, %20, %26, %6;"
		"madc.hi.cc.u32  %7, %20, %26, %7;"
		"madc.lo.cc.u32  %8, %22, %26, %8;"
		"madc.hi.cc.u32  %9, %22, %26, %9;"
		"addc.u32        %10, 0, 0;"

		"mad.lo.cc.u32   %3, %17, %26, %3;"
		"madc.hi.cc.u32  %4, %17, %26, %4;"
		"madc.lo.cc.u32  %5, %19, %26, %5;"
		"madc.hi.cc.u32  %6, %19, %26, %6;"
		"madc.lo.cc.u32  %7, %21, %26, %7;"
		"madc.hi.cc.u32  %8, %21, %26, %8;"
		"madc.lo.cc.u32  %9, %23, %26, %9;"
		"madc.hi.u32     %10, %23, %26, %10;"

		// x * y3
		"mad.lo.cc.u32   %3, %16, %27, %3;"
		"madc.hi.cc.u32  %4, %16, %27, %4;"
		"madc.lo.cc.u32  %5, %18, %27, %5;"
		"madc.hi.cc.u32  %6, %18, %27, %6;"
		"madc.lo.cc.u32  %7, %20, %27, %7;"
		"madc.hi.cc.u32  %8, %20, %27, %8;"
		"madc.lo.cc.u32  %9, %22, %27, %9;"
		"madc.hi.cc.u32  %10, %22, %27, %10;"
		"addc.u32        %11, 0, 0;"

		"mad.lo.cc.u32   %4, %17, %27, %4;"
		"madc.hi.cc.u32  %5, %17, %27, %5;"
		"madc.lo.cc.u32  %6, %19, %27, %6;"
		"madc.hi.cc.u32  %7, %19, %27, %7;"
		"madc.lo.cc.u32  %8, %21, %27, %8;"
		"madc.hi.cc.u32  %9, %21, %27, %9;"
		"madc.lo.cc.u32  %10, %23, %27, %10;"
		"madc.hi.u32     %11, %23, %27, %11;"

		// x * y4
		"mad.lo.cc.u32   %4, %16, %28, %4;"
		"madc.hi.cc.u32  %5, %16, %28, %5;"
		"madc.lo.cc.u32  %6, %18, %28, %6;"
		"madc.hi.cc.u32  %7, %18, %28, %7;"
		"madc.lo.cc.u32  %8, %20, %28, %8;"
		"madc.hi.cc.u32  %9, %20, %28, %9;"
		"madc.lo.cc.u32  %10, %22, %28, %10;"
		"madc.hi.cc.u32  %11, %22, %28, %11;"
		"addc.u32        %12, 0, 0;"

		"mad.lo.cc.u32   %5, %17, %28, %5;"
		"madc.hi.cc.u32  %6, %17, %28, %6;"
		"madc.lo.cc.u32  %7, %19, %28, %7;"
		"madc.hi.cc.u32  %8, %19, %28, %8;"
		"madc.lo.cc.u32  %9, %21, %28, %9;"
		"madc.hi.cc.u32  %10, %21, %28, %10;"
		"madc.lo.cc.u32  %11, %23, %28, %11;"
		"madc.hi.u32     %12, %23, %28, %12;"

		// x * y5
		"mad.lo.cc.u32   %5, %16, %29, %5;"
		"madc.hi.cc.u32  %6, %16, %29, %6;"
		"madc.lo.cc.u32  %7, %18, %29, %7;"
		"madc.hi.cc.u32  %8, %18, %29, %8;"
		"madc.lo.cc.u32  %9, %20, %29, %9;"
		"madc.hi.cc.u32  %10, %20, %29, %10;"
		"madc.lo.cc.u32  %11, %22, %29, %11;"
		"madc.hi.cc.u32  %12, %22, %29, %12;"
		"addc.u32        %13, 0, 0;"

		"mad.lo.cc.u32   %6, %17, %29, %6;"
		"madc.hi.cc.u32  %7, %17, %29, %7;"
		"madc.lo.cc.u32  %8, %19, %29, %8;"
		"madc.hi.cc.u32  %9, %19, %29, %9;"
		"madc.lo.cc.u32  %10, %21, %29, %10;"
		"madc.hi.cc.u32  %11, %21, %29, %11;"
		"madc.lo.cc.u32  %12, %23, %29, %12;"
		"madc.hi.u32     %13, %23, %29, %13;"

		// x * y6
		"mad.lo.cc.u32   %6, %16, %30, %6;"
		"madc.hi.cc.u32  %7, %16, %30, %7;"
		"madc.lo.cc.u32  %8, %18, %30, %8;"
		"madc.hi.cc.u32  %9, %18, %30, %9;"
		"madc.lo.cc.u32  %10, %20, %30, %10;"
		"madc.hi.cc.u32  %11, %20, %30, %11;"
		"madc.lo.cc.u32  %12, %22, %30, %12;"
		"madc.hi.cc.u32  %13, %22, %30, %13;"
		"addc.u32        %14, 0, 0;"

		"mad.lo.cc.u32   %7, %17, %30, %7;"
		"madc.hi.cc.u32  %8, %17, %30, %8;"
		"madc.lo.cc.u32  %9, %19, %30, %9;"
		"madc.hi.cc.u32  %10, %19, %30, %10;"
		"madc.lo.cc.u32  %11, %21, %30, %11;"
		"madc.hi.cc.u32  %12, %21, %30, %12;"
		"madc.lo.cc.u32  %13, %23, %30, %13;"
		"madc.hi.u32     %14, %23, %30, %14;"

		// x * y7
		"mad.lo.cc.u32   %7, %16, %31, %7;"
		"madc.hi.cc.u32  %8, %16, %31, %8;"
		"madc.lo.cc.u32  %9, %18, %31, %9;"
		"madc.hi.cc.u32  %10, %18, %31, %10;"
		"madc.lo.cc.u32  %11, %20, %31, %11;"
		"madc.hi.cc.u32  %12, %20, %31, %12;"
		"madc.lo.cc.u32  %13, %22, %31, %13;"
		"madc.hi.cc.u32  %14, %22, %31, %14;"
		"addc.u32        %15, 0, 0;"

		"mad.lo.cc.u32   %8, %17, %31, %8;"
		"madc.hi.cc.u32  %9, %17, %31, %9;"
		"madc.lo.cc.u32  %10, %19, %31, %10;"
		"madc.hi.cc.u32  %11, %19, %31, %11;"
		"madc.lo.cc.u32  %12, %21, %31, %12;"
		"madc.hi.cc.u32  %13, %21, %31, %13;"
		"madc.lo.cc.u32  %14, %23, %31, %14;"
		"madc.hi.u32     %15, %23, %31, %15;"

		"}"
		: "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7]), "=r"(temp_hi[0]), "=r"(temp_hi[1]), "=r"(temp_hi[2]), "=r"(temp_hi[3]), "=r"(temp_hi[4]), "=r"(temp_hi[5]), "=r"(temp_hi[6]), "=r"(temp_hi[7])
		: "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]), "r"(y[0]), "r"(y[1]), "r"(y[2]), "r"(y[3]), "r"(y[4]), "r"(y[5]), "r"(y[6]), "r"(y[7]));

	u32 temp_hi8;

	// lazy reduction

	//out_hi[256] * 189 + out_low[256]
	asm("{\n\t"

		"mad.lo.cc.u32   %0, %9, 189, %17;"
		"madc.hi.cc.u32  %1, %9, 189, %18;"
		"madc.lo.cc.u32  %2, %11, 189, %19;"
		"madc.hi.cc.u32  %3, %11, 189, %20;"
		"madc.lo.cc.u32  %4, %13, 189, %21;"
		"madc.hi.cc.u32  %5, %13, 189, %22;"
		"madc.lo.cc.u32  %6, %15, 189, %23;"
		"madc.hi.cc.u32  %7, %15, 189, %24;"
		"addc.u32        %8, 0, 0;"

		"mad.lo.cc.u32   %1, %10, 189, %1;"
		"madc.hi.cc.u32  %2, %10, 189, %2;"
		"madc.lo.cc.u32  %3, %12, 189, %3;"
		"madc.hi.cc.u32  %4, %12, 189, %4;"
		"madc.lo.cc.u32  %5, %14, 189, %5;"
		"madc.hi.cc.u32  %6, %14, 189, %6;"
		"madc.lo.cc.u32  %7, %16, 189, %7;"
		"madc.hi.u32     %8, %16, 189, %8;"

		"}"
		: "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7]), "=r"(temp_hi8)
		: "r"(temp_hi[0]), "r"(temp_hi[1]), "r"(temp_hi[2]), "r"(temp_hi[3]), "r"(temp_hi[4]), "r"(temp_hi[5]), "r"(temp_hi[6]), "r"(temp_hi[7]), "r"(out[0]), "r"(out[1]), "r"(out[2]), "r"(out[3]), "r"(out[4]), "r"(out[5]), "r"(out[6]), "r"(out[7]));

	u32 temp_hi1;

	//out_hi[8] * 189 + out_low[256]
	asm("{\n\t"

		"mad.lo.cc.u32   %0, %9, 189, %10;"
		"addc.cc.u32     %1, %11, 0;"
		"addc.cc.u32     %2, %12, 0;"
		"addc.cc.u32     %3, %13, 0;"
		"addc.cc.u32     %4, %14, 0;"
		"addc.cc.u32     %5, %15, 0;"
		"addc.cc.u32     %6, %16, 0;"
		"addc.cc.u32     %7, %17, 0;"
		"addc.u32        %8, 0, 0;"

		"}"
		: "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7]), "=r"(temp_hi1)
		: "r"(temp_hi8), "r"(out[0]), "r"(out[1]), "r"(out[2]), "r"(out[3]), "r"(out[4]), "r"(out[5]), "r"(out[6]), "r"(out[7]));

	//out_hi[1] * 189 + out_low[256]
	if (temp_hi1 & 1)
	{
		asm("{\n\t"

			"add.cc.u32      %0, %8, 189;"
			"addc.cc.u32     %1, %9, 0;"
			"addc.cc.u32     %2, %10, 0;"
			"addc.cc.u32     %3, %11, 0;"
			"addc.cc.u32     %4, %12, 0;"
			"addc.cc.u32     %5, %13, 0;"
			"addc.cc.u32     %6, %14, 0;"
			"addc.u32        %7, %15, 0;"

			"}"
			: "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
			: "r"(out[0]), "r"(out[1]), "r"(out[2]), "r"(out[3]), "r"(out[4]), "r"(out[5]), "r"(out[6]), "r"(out[7]));
	}

}

__device__ __forceinline__ void square_mul_reduce_256(u256 out, const u256 x, unsigned count, const u256 y)
{   // squares x, multiplies is with itself `count` times, then multiplies this result with `y` in the end.
	// all operations are done in modulo prime
	u256 temp;

	mul_reduce_256(temp, x, x);  // temp = x * x

	while (--count)
		mul_reduce_256(temp, temp, temp);  // temp = temp * temp

	mul_reduce_256(out, temp, y);  // out = temp * y
}

__device__ __forceinline__ bool check_ge_prime(const u256 x)
{  // checks if prime is greater than prime
	if ((x[7] >= 4294967295) & (x[6] >= 4294967295) & (x[5] >= 4294967295) & (x[4] >= 4294967295)
		& (x[3] >= 4294967295) & (x[2] >= 4294967295) & (x[1] >= 4294967295) & (x[0] >= 4294967107))
	{
		return true;
	}

	return false;
}

__device__ __forceinline__ bool check_eq_x_y(const u256 x, const u256 y)
{  // checks if x and y are equal
	if ((x[7] == y[7]) & (x[6] == y[6]) & (x[5] == y[5]) & (x[4] == y[4])
		& (x[3] == y[3]) & (x[2] == y[2]) & (x[1] == y[1]) & (x[0] == y[0]))
	{
		return true;
	}

	return false;
}

__device__ __forceinline__ bool check_odd(const u256 x)
{  // checks if x is odd
	if (x[0] & 1)
		return true;

	return false;
}

__device__ __forceinline__ bool check_even(const u256 x)
{  // checks if x is even
	if (x[0] & 1)
		return false;

	return true;
}

__device__ __forceinline__ void x_minus_prime(u256 x)
{  // x = x - p
	asm("{\n\t"

		"sub.cc.u32     %0, %8, 4294967107;"
		"subc.cc.u32    %1, %9, 4294967295;"
		"subc.cc.u32    %2, %10, 4294967295;"
		"subc.cc.u32    %3, %11, 4294967295;"
		"subc.cc.u32    %4, %12, 4294967295;"
		"subc.cc.u32    %5, %13, 4294967295;"
		"subc.cc.u32    %6, %14, 4294967295;"
		"subc.u32       %7, %15, 4294967295;"

		"}"
		: "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]), "=r"(x[4]), "=r"(x[5]), "=r"(x[6]), "=r"(x[7])
		: "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));
}

__device__ __forceinline__ void prime_minus_x(u256 x)
{  // x = p - x
	asm("{\n\t"

		"sub.cc.u32     %0, 4294967107, %8;"
		"subc.cc.u32    %1, 4294967295, %9;"
		"subc.cc.u32    %2, 4294967295, %10;"
		"subc.cc.u32    %3, 4294967295, %11;"
		"subc.cc.u32    %4, 4294967295, %12;"
		"subc.cc.u32    %5, 4294967295, %13;"
		"subc.cc.u32    %6, 4294967295, %14;"
		"subc.u32       %7, 4294967295, %15;"

		"}"
		: "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]), "=r"(x[4]), "=r"(x[5]), "=r"(x[6]), "=r"(x[7])
		: "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));
}

__device__ __forceinline__ void xor_x_y(u256 out, u256 x, u256 y)
{  // x = x^y
	out[0] = x[0] ^ y[0];
	out[1] = x[1] ^ y[1];
	out[2] = x[2] ^ y[2];
	out[3] = x[3] ^ y[3];
	out[4] = x[4] ^ y[4];
	out[5] = x[5] ^ y[5];
	out[6] = x[6] ^ y[6];
	out[7] = x[7] ^ y[7];
}

__device__ __forceinline__ void eq_x_y(u256 x, const u256 y)
{  // x = y
	x[0] = y[0];
	x[1] = y[1];
	x[2] = y[2];
	x[3] = y[3];
	x[4] = y[4];
	x[5] = y[5];
	x[6] = y[6];
	x[7] = y[7];
}

__device__ __forceinline__ void addition_chain_reduce_256(u256 out, const u256 x)
{
	u256 temp;

	square_mul_reduce_256(out, x, 1, x);
	square_mul_reduce_256(temp, out, 1, x);
	square_mul_reduce_256(out, temp, 3, temp);
	square_mul_reduce_256(out, out, 1, x);
	square_mul_reduce_256(out, out, 7, out);
	square_mul_reduce_256(out, out, 14, out);
	square_mul_reduce_256(out, out, 3, temp);
	square_mul_reduce_256(out, out, 31, out);
	square_mul_reduce_256(out, out, 62, out);
	square_mul_reduce_256(out, out, 124, out);
	square_mul_reduce_256(out, out, 2, x);
	square_mul_reduce_256(out, out, 4, x);

	if (check_ge_prime(out))
		x_minus_prime(out);
}

__device__ __forceinline__ void sqrt_permutation_ptx(u256 out, u256 x)
{   // finds the square root of x in SLOTH setting

	addition_chain_reduce_256(out, x);  // out = pow(x, (prime+1)/4, prime)

	if (check_odd(out)) {  // if the number is odd, we have to negate the square-root
		prime_minus_x(out);
	}

	// check if square of x is the same with input (this check is necessary,
	// since there are some numbers that do not have square roots,
	// but their negations do have, read the SLOTH paper for more details).
	// Then negate the result, and assign this to variable `neg`.
	// So, if our computed square_root's square equals to input, this means
	// the input indeed has a square root and we found it
	u256 check_candidate;
	mul_reduce_256(check_candidate, out, out);

	// because of lazy-reduction in mul_reduce, we have to check the result is greater than prime
	if (check_ge_prime(check_candidate)) {
		x_minus_prime(check_candidate);  // if so, apply the reduction. One check is enough, lazy-reduction guarantees that
	}


	if (check_eq_x_y(check_candidate, x)) {}  // means we have found the correct square root
	else {
		if (check_even(out)) {
			prime_minus_x(out);  // negate the square-root accordingly with it being even-odd.
		}
	}
}

__global__ void encode_ptx(u256* piece, u256* nonce, u256* farmer_id)
{    // in this version, expanded_iv will be computed from nonce array and farmer_id

	int global_idx = threadIdx.x + blockIdx.x * blockDim.x;  // global index of the thread

	u256 feedback;
	xor_x_y(feedback, nonce[global_idx], *farmer_id);  // computation of the expanded_iv from nonce array

	for (int i = 0; i < 128; i++)  // actual sloth_encoding
	{
		xor_x_y(feedback, piece[i + global_idx * 128], feedback);
		sqrt_permutation_ptx(piece[i + global_idx * 128], feedback);
		eq_x_y(feedback, piece[i + global_idx * 128]);
	}
}

__global__ void sloth256_189_encode_cuda(u256* piece_array, u256* expanded_iv, size_t layers)
{   // in this version, expanded_iv will be given directly

	int global_idx = threadIdx.x + blockIdx.x * blockDim.x;  // global index of the thread

	u256 feedback;
	eq_x_y(feedback, expanded_iv[global_idx]);  // getting the related feedback from the expanded_iv array

    for (int x = 0; x < layers; x++)
    {
        for (int i = 0; i < 128; i++)  // actual sloth_encoding
        {
            u32* chunk_ptr = piece_array[i + global_idx * 128];  // getting the related piece from piece_array
            // the pointer created from the right side is a regular pointer (32-bytes).
            // we could have replaced all the below `chunk_ptr`s with `piece_array[i + global_idx * 128]`
            // like it was in the `encode_ptx` function.
            // this is an optimization to eliminate the re-computations of the same pointer

            xor_x_y(feedback, chunk_ptr, feedback);
            sqrt_permutation_ptx(chunk_ptr, feedback);
            eq_x_y(feedback, chunk_ptr);
        }
	}
}
