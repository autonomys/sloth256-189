#ifndef __OPENCL_C_VERSION__
#include <stdlib.h>

#if defined(__GNUC__)
# ifndef alloca
#  define alloca(s) __builtin_alloca(s)
# endif
#elif defined(__sun)
# include <alloca.h>
#elif defined(_WIN32)
# include <malloc.h>
# ifndef alloca
#  define alloca(s) _alloca(s)
# endif
#endif
#elif defined(__SIZE_TYPE__) && !defined(_OPENCL_BASE_H_)
typedef __SIZE_TYPE__ size_t;
#endif

#ifdef __UINTPTR_TYPE__
typedef __UINTPTR_TYPE__ uptr_t;
#else
typedef const void *uptr_t;
#endif

#if !defined(restrict)
# if !defined(__STDC_VERSION__) || __STDC_VERSION__<199901
#  if defined(__GNUC__) && __GNUC__>=2
#   define restrict __restrict__
#  elif defined(_MSC_VER)
#   define restrict __restrict
#  else
#   define restrict
#  endif
# endif
#endif

#if !defined(inline) && !defined(__cplusplus)
# if !defined(__STDC_VERSION__) || __STDC_VERSION__<199901
#  if defined(__GNUC__) && __GNUC__>=2
#   define inline __inline__
#  elif defined(_MSC_VER)
#   define inline __inline
#  else
#   define inline
#  endif
# endif
#endif

#if defined(__CUDA_ARCH__)
# define static static __device__
# ifndef __LITTLE_ENDIAN__
#  define __LITTLE_ENDIAN__ 1
# endif
#endif

#if defined(_LP64) || __SIZEOF_LONG__-0==8
typedef unsigned long limb_t;
#elif defined(_WIN64) || \
      defined(__x86_64__) || defined(__aarch64__) || \
      defined(__mips64) || defined(__ia64) || \
      (defined(__VMS) && !defined(__vax)) /* these named 64-bits can be ILP32 */
typedef unsigned long long limb_t;
#else
typedef unsigned int limb_t;
#endif

#define LIMB_T_BITS (8*sizeof(limb_t))

typedef limb_t vec256[256/LIMB_T_BITS];
typedef limb_t bool_t;

static inline void limbs_from_le_bytes(limb_t *restrict ret,
                                       const unsigned char *in, size_t n)
{
    limb_t limb = 0;

    while(n--) {
        limb <<= 8;
        limb |= in[n];
        /*
         * 'if (n % sizeof(limb_t) == 0)' is omitted because it's cheaper
         * to perform redundant stores than to pay penalty for
         * mispredicted branch. Besides, some compilers unroll the
         * loop and remove redundant stores to 'restict'-ed storage...
         */
        ret[n / sizeof(limb_t)] = limb;
    }
}

static inline void le_bytes_from_limbs(unsigned char *out, const limb_t *in,
                                       size_t n)
{
    const union {
        long one;
        char little;
    } is_endian = { 1 };
    limb_t limb;
    size_t i, j, r;

    if ((uptr_t)out == (uptr_t)in && is_endian.little)
        return;

    r = n % sizeof(limb_t);
    n /= sizeof(limb_t);

    for(i = 0; i < n; i++) {
        for (limb = in[i], j = 0; j < sizeof(limb_t); j++, limb >>= 8)
            *out++ = (unsigned char)limb;
    }
    if (r) {
        for (limb = in[i], j = 0; j < r; j++, limb >>= 8)
            *out++ = (unsigned char)limb;
    }
}

static inline bool_t is_zero(limb_t l)
{   return (~l & (l - 1)) >> (LIMB_T_BITS - 1);   }

static inline bool_t vec_is_equal(const void *a, const void *b, size_t num)
{
    const limb_t *ap = (const limb_t *)a;
    const limb_t *bp = (const limb_t *)b;
    limb_t acc;
    size_t i;

    num /= sizeof(limb_t);

    for (acc = 0, i = 0; i < num; i++)
        acc |= ap[i] ^ bp[i];

    return is_zero(acc);
}

#if !defined(__SLOTH256_189_NO_ASM__) && !defined(__CUDA_ARCH__) && \
    (defined(__x86_64__) || defined(_M_AMD64) || defined(_M_X64))

void sqrx_n_mul_mod_256_189(vec256 out, const vec256 a, size_t count,
                                        const vec256 b);
void mulx_mod_256_189(vec256 out, const vec256 a, const vec256 b);
void sqrx_mod_256_189(vec256 out, const vec256 a);
void redc_mod_256_189(vec256 out, const vec256 a);
void cneg_mod_256_189(vec256 out, const vec256 a, bool_t cbit);
bool_t xor_n_check_mod_256_189(vec256 out, const vec256 a, const vec256 b);

# if defined(__GNUC__)
static inline void __cpuidex(int info[4], int eax, int ecx)
{
    int ebx, edx;

    asm ("cpuid"
         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
         : "a"(eax), "c"(ecx) : "cc");
    info[0] = eax;
    info[1] = ebx;
    info[2] = ecx;
    info[3] = edx;
}
#  define __x86_64__cpuidex __cpuidex
# elif defined(_MSC_VER)
#  include <intrin.h>
#  define __x86_64__cpuidex __cpuidex
# endif

#if 0
static void sqr_n_mul(vec256 out, const vec256 a, size_t count,
                                  const vec256 b)
{
    vec256 t;

    while(count--) {
        sqrx_mod_256_189(t, a);
        a = t;
    }
    mulx_mod_256_189(out, t, b);
}
#endif

/*
 * |out| = |inp|**((|mod|+1)/4)%|mod|, where |mod| is 2**256-189.
 * ~8.700 cycles on Coffee Lake, ~9.500 - on Rocket Lake:-(
 */
static bool_t sqrtx_mod_256_189(vec256 out, const vec256 inp)
{
    vec256 x, y;
    bool_t neg;

#define sqr_n_mul sqrx_n_mul_mod_256_189
    sqr_n_mul(x, inp, 1, inp);  /* 0x3 */
    sqr_n_mul(y, x, 1, inp);    /* 0x7 */
    sqr_n_mul(x, y, 3, y);      /* 0x3f */
    sqr_n_mul(x, x, 1, inp);    /* 0x7f */
    sqr_n_mul(x, x, 7, x);      /* 0x3fff */
    sqr_n_mul(x, x, 14, x);     /* 0xfffffff */
    sqr_n_mul(x, x, 3, y);      /* 0x7fffffff */
    sqr_n_mul(x, x, 31, x);     /* 0x3fffffffffffffff */
    sqr_n_mul(x, x, 62, x);     /* 0xfffffffffffffffffffffffffffffff */
    sqr_n_mul(x, x, 124, x);    /* 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff */
    sqr_n_mul(x, x, 2, inp);    /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd */
    sqr_n_mul(x, x, 4, inp);    /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd1 */
#undef sqr_n_mul

    redc_mod_256_189(x, x);
    mulx_mod_256_189(y, x, x);  /* mulx is in cache, sqrx isn't */
    neg = vec_is_equal(y, inp, sizeof(y)) ^ 1;
    cneg_mod_256_189(out, x, neg ^ (bool_t)(x[0]&1));

    return neg;
}

static void squarex_mod_256_189(vec256 out, const vec256 inp)
{
    bool_t neg = (bool_t)(inp[0]&1);

    sqrx_mod_256_189(out, inp);
    redc_mod_256_189(out, out);
    cneg_mod_256_189(out, out, neg);
}

#else   /* no-asm */

static void cneg_mod_256_189(vec256 out, const vec256 a, bool_t cbit)
{
    limb_t mask, carry;
    size_t i;

    for (mask =0, i = 0; i < sizeof(vec256)/sizeof(limb_t); i++)
        mask |= a[i];

    mask = cbit & (is_zero(mask) ^ 1);
    mask = 0 - mask;

    out[0] = a[0] + (189 & mask); carry = out[0] < (189 & mask);
    for (i = 1; i < sizeof(vec256)/sizeof(limb_t); i++)
        out[i] = a[i] + carry, carry = out[i] < carry;
    for (i = 0; i < sizeof(vec256)/sizeof(limb_t); i++)
        out[i] ^= mask;
    out[0] += (1 & mask); carry = out[0] < (1 & mask);
    for (i = 1; i < sizeof(vec256)/sizeof(limb_t); i++)
        out[i] += carry, carry = out[i] < carry;
}

static bool_t xor_n_check_mod_256_189(vec256 out, const vec256 a,
                                                  const vec256 b)
{
    size_t i;
    limb_t carry;

    for (i = 0; i < sizeof(vec256)/sizeof(limb_t); i++)
        out[i] = a[i] ^ b[i];

    carry = (out[0] + 189) < 189;
    for (i = 1; i < sizeof(vec256)/sizeof(limb_t); i++)
        carry = (out[i] + carry) < carry;

    return (bool_t)carry;
}

#endif

#if (__SIZEOF_INT128__-0==16 || __SIZEOF_LONG_LONG__-0==16) && \
    !defined(__CUDA_ARCH__)

#if __SIZEOF_LONG_LONG__-0==16
typedef unsigned long long u128;
typedef unsigned long u64;
#else
typedef __uint128_t u128;
typedef unsigned long long u64;
#endif

typedef u64 fe51[5];    /* NB! First limb is 52 bits */

#ifdef __OPENCL_C_VERSION__
# define MASK51 0x7ffffffffffff
# define MASK52 0xfffffffffffff
#else
static const u64 MASK51 = 0x7ffffffffffff,
                 MASK52 = 0xfffffffffffff;
#endif

static void to_fe51(fe51 out, const vec256 in)
{
    out[0]  = (in[0])                 & MASK52;
    out[1]  = (in[0]>>52 | in[1]<<12) & MASK51;
    out[2]  = (in[1]>>39 | in[2]<<25) & MASK51;
    out[3]  = (in[2]>>26 | in[3]<<38) & MASK51;
    out[4]  = (in[3]>>13);
}

static void from_fe51(vec256 out, const fe51 in)
{
    u64 h0 = in[0],
        h1 = in[1],
        h2 = in[2],
        h3 = in[3],
        h4 = in[4];
    u64 q;
    u128 t;

    /* compare to the modulus */
    q = (h0 + 189) >> 52;
    q = (h1 + q) >> 51;
    q = (h2 + q) >> 51;
    q = (h3 + q) >> 51;
    q = (h4 + q) >> 51;

    /* full reduction */
    h0 += 189 * q;

    t  = (u128)h0;
    t += (u128)h1<<52;  out[0] = (u64)t; t >>= 64;
    t += (u128)h2<<39;  out[1] = (u64)t; t >>= 64;
    t += (u128)h3<<26;  out[2] = (u64)t; t >>= 64;
    t += (u128)h4<<13;  out[3] = (u64)t;
}

static void mul_mod_256_189(fe51 h, const fe51 f, const fe51 g)
{
    u128 h0, h1, h2, h3, h4;
    u64  g0, g1, g2, g3, g4, f_i;

    f_i = f[0];
    h0 = (u128)f_i * (g0 = g[0]);
    h1 = (u128)f_i * (g1 = g[1]);
    h2 = (u128)f_i * (g2 = g[2]);
    h3 = (u128)f_i * (g3 = g[3]);
    h4 = (u128)f_i * (g4 = g[4]);

    f_i = f[1];
    h0 += (u128)f_i * ((g4 *= 189) << 1);
    h1 += (u128)f_i * g0;
    h2 += (u128)f_i * (g1 <<= 1);
    h3 += (u128)f_i * (g2 <<= 1);
    h4 += (u128)f_i * (g3 <<= 1);

    f_i = f[2];
    h0 += (u128)f_i * (g3 *= 189);
    h1 += (u128)f_i * g4;
    h2 += (u128)f_i * g0;
    h3 += (u128)f_i * g1;
    h4 += (u128)f_i * g2;

    f_i = f[3];
    h0 += (u128)f_i * (g2 *= 189);
    h1 += (u128)f_i * (g3 >>= 1);
    h2 += (u128)f_i * g4;
    h3 += (u128)f_i * g0;
    h4 += (u128)f_i * g1;

    f_i = f[4];
    h0 += (u128)f_i * (g1 *= 189);
    h1 += (u128)f_i * (g2 >>= 1);
    h2 += (u128)f_i * g3;
    h3 += (u128)f_i * g4;
    h4 += (u128)f_i * g0;

    /* partial [lazy] reduction */
    h3 += (u64)(h2 >> 51);      g2 = (u64)h2 & MASK51;
    h1 += (u64)(h0 >> 52);      g0 = (u64)h0 & MASK52;

    h4 += (u64)(h3 >> 51);      g3 = (u64)h3 & MASK51;
    g2 += (u64)(h1 >> 51);      g1 = (u64)h1 & MASK51;

    g0 += (u64)(h4 >> 51)*189;  h[4] = (u64)h4 & MASK51;
    h[3] = g3 + (g2 >> 51);     h[2] = g2 & MASK51;
    h[1] = g1 + (g0 >> 52);     h[0] = g0 & MASK52;
}

static void sqr_mod_256_189(fe51 h, const fe51 f)
{
#ifdef __OPTIMIZE_SIZE__
    mul_mod_256_189(h, f, f);
#else   /* ~50% faster sqrt */
    u128 h0, h1, h2, h3, h4;
    u64  g0, g1, g2, g3, g4;
    u64      f1, f2, f3, f4;

    g0 = f[0];
    h0 = (u128)g0 * g0; g0 <<= 1;
    h1 = (u128)g0 * (g1 = f1 = f[1]);
    h2 = (u128)g0 * (g2 = f2 = f[2]);
    h3 = (u128)g0 * (g3 = f3 = f[3]);
    h4 = (u128)g0 * (g4 = f4 = f[4]);

    h2 += (u128)f1 * (g1 <<= 1);
    h3 += (u128)g1 * (g2 <<= 1);
    h4 += (u128)g1 * (g3 <<= 1);

    h0 += (u128)g1 * ((g4 *= 189) << 1);
    h1 += (u128)g2 * g4;
    h2 += (u128)g3 * g4;
    h3 += (u128)f4 * g4;
    h4 += (u128)f2 * g2;

    h0 += (u128)g2 * (g3 *= 189);
    h1 += (u128)f3 * (g3 >> 1);

    /* partial [lazy] reduction */
    h3 += (u64)(h2 >> 51);      g2 = (u64)h2 & MASK51;
    h1 += (u64)(h0 >> 52);      g0 = (u64)h0 & MASK52;

    h4 += (u64)(h3 >> 51);      g3 = (u64)h3 & MASK51;
    g2 += (u64)(h1 >> 51);      g1 = (u64)h1 & MASK51;

    g0 += (u64)(h4 >> 51)*189;  h[4] = (u64)h4 & MASK51;
    h[3] = g3 + (g2 >> 51);     h[2] = g2 & MASK51;
    h[1] = g1 + (g0 >> 52);     h[0] = g0 & MASK52;
#endif
}

static void sqr_n_mul_fe51(fe51 out, const fe51 a, size_t count,
                                     const fe51 b)
{
    fe51 t;

    while(count--) {
        sqr_mod_256_189(t, a);
        a = t;
    }
    mul_mod_256_189(out, t, b);
}

/*
 * |out| = |inp|**((|mod|+1)/4)%|mod|, where |mod| is 2**256-189.
 * ~11.300 cycles on contemporary Intel processors with clang-10.
 */
static bool_t sqrt_mod_256_189(vec256 out, const vec256 inp)
{
    fe51 x, y, z;
    vec256 sqr, ret;
    bool_t neg;

    to_fe51(z, inp);

#define sqr_n_mul sqr_n_mul_fe51
    sqr_n_mul(x, z, 1, z);      /* 0x3 */
    sqr_n_mul(y, x, 1, z);      /* 0x7 */
    sqr_n_mul(x, y, 3, y);      /* 0x3f */
    sqr_n_mul(x, x, 1, z);      /* 0x7f */
    sqr_n_mul(x, x, 7, x);      /* 0x3fff */
    sqr_n_mul(x, x, 14, x);     /* 0xfffffff */
    sqr_n_mul(x, x, 3, y);      /* 0x7fffffff */
    sqr_n_mul(x, x, 31, x);     /* 0x3fffffffffffffff */
    sqr_n_mul(x, x, 62, x);     /* 0xfffffffffffffffffffffffffffffff */
    sqr_n_mul(x, x, 124, x);    /* 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff */
    sqr_n_mul(x, x, 2, z);      /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd */
    sqr_n_mul(x, x, 4, z);      /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd1 */
#undef sqr_n_mul

    mul_mod_256_189(y, x, x);
    from_fe51(ret, x);
    from_fe51(sqr, y);
    neg = vec_is_equal(sqr, inp, sizeof(sqr)) ^ 1;
    cneg_mod_256_189(out, ret, neg ^ (bool_t)(ret[0]&1));

    return neg;
}

static void square_mod_256_189(vec256 out, const vec256 inp)
{
    bool_t neg = (bool_t)(inp[0]&1);
    fe51 x;

    to_fe51(x, inp);
    sqr_mod_256_189(x, x);
    from_fe51(out, x);
    cneg_mod_256_189(out, out, neg);
}

#else

#if defined(_WIN64) && defined(_MSC_VER) && !defined(__CUDA_ARCH__)
# pragma message(__FILE__": consider building with %CC% set to 'clang-cl'")
#endif

#if defined(_LP64) || __SIZEOF_LONG__-0==8
typedef unsigned long u64;
#else
typedef unsigned long long u64;
#endif
typedef unsigned int u32;

typedef u32 fe26[10];           /* 9x26+22=256 */

#if defined(__OPENCL_C_VERSION__) || defined(__CUDA_ARCH__)
# define MASK26 0x3ffffff
# define MASK22 0x3fffff
#else
static const u32 MASK26 = 0x3ffffff,
                 MASK22 = 0x3fffff;
#endif

static void to_fe26(fe26 out, const void *in_)
{
    const u32 *in = (const u32 *)in_;
    const union {
        long one;
        char little;
    } is_endian = { 1 };
    const size_t adj = (is_endian.little^1) & (sizeof(limb_t)==8);

    out[0] = (in[0^adj])                     & MASK26;
    out[1] = (in[0^adj]>>26 | in[1^adj]<<6)  & MASK26;
    out[2] = (in[1^adj]>>20 | in[2^adj]<<12) & MASK26;
    out[3] = (in[2^adj]>>14 | in[3^adj]<<18) & MASK26;
    out[4] = (in[3^adj]>>8  | in[4^adj]<<24) & MASK26;
    out[5] = (in[4^adj]>>2)                  & MASK26;
    out[6] = (in[4^adj]>>28 | in[5^adj]<<4)  & MASK26;
    out[7] = (in[5^adj]>>22 | in[6^adj]<<10) & MASK26;
    out[8] = (in[6^adj]>>16 | in[7^adj]<<16) & MASK26;
    out[9] = (in[7^adj]>>10);
}

static void from_fe26(void *out_, const fe26 in)
{
    u32 h0, q, *out = (u32 *)out_;
    u64 t;
    const union {
        long one;
        char little;
    } is_endian = { 1 };
    const size_t adj = (is_endian.little^1) & (sizeof(limb_t)==8);

    /* compare to the modulus */
    q = ((h0 = in[0]) + 189) >> 26;
    q = (in[1] + q) >> 26;
    q = (in[2] + q) >> 26;
    q = (in[3] + q) >> 26;
    q = (in[4] + q) >> 26;
    q = (in[5] + q) >> 26;
    q = (in[6] + q) >> 26;
    q = (in[7] + q) >> 26;
    q = (in[8] + q) >> 26;
    q = (in[9] + q) >> 22;

    /* full reduction */
    h0 += 189 * q;

    t  = (u64)h0;
    t += (u64)in[1]<<26;    out[0^adj] = (u32)t;    t >>= 32;
    t += (u64)in[2]<<20;    out[1^adj] = (u32)t;    t >>= 32;
    t += (u64)in[3]<<14;    out[2^adj] = (u32)t;    t >>= 32;
    t += (u64)in[4]<<8;     out[3^adj] = (u32)t;    t >>= 32;
    t += (u64)in[5]<<2;
    t += (u64)in[6]<<28;    out[4^adj] = (u32)t;    t >>= 32;
    t += (u64)in[7]<<22;    out[5^adj] = (u32)t;    t >>= 32;
    t += (u64)in[8]<<16;    out[6^adj] = (u32)t;    t >>= 32;
    t += (u64)in[9]<<10;    out[7^adj] = (u32)t;
}

static void mul_mod_256_189(fe26 h, const fe26 f, const fe26 g_)
{
#ifdef __OPTIMIZE_SIZE__
    u64 H[10], temp;
    u32 g[16], f_i;
    size_t i;

    f_i  = f[0];
    H[0] = (u64)f_i * (g[0] = g_[0]);
    H[1] = (u64)f_i * (g[1] = g_[1]);
    H[2] = (u64)f_i * (g[2] = g_[2]);
    H[3] = (u64)f_i * (g[3] = g_[3]);
    H[4] = (u64)f_i * (g[4] = g_[4]);
    H[5] = (u64)f_i * (g[5] = g_[5]);
    H[6] = (u64)f_i * (g[6] = g_[6]);
    H[7] = (u64)f_i * (g[7] = g_[7]);
    H[8] = (u64)f_i * (g[8] = g_[8]);
    H[9] = (u64)f_i * (g[9] = g_[9]);

    for (i=1; i<10; i++) {
        f_i   = f[i];               temp  = (u64)g[10-i] * (189<<4);
        H[0] += (u64)f_i * (g[(0-i)&0xf]  = (u32)temp & MASK26);
        H[1] += (u64)f_i * (g[(1-i)&0xf] += (u32)(temp >> 26));
        H[2] += (u64)f_i * g[(2-i)&0xf];
        H[3] += (u64)f_i * g[(3-i)&0xf];
        H[4] += (u64)f_i * g[(4-i)&0xf];
        H[5] += (u64)f_i * g[(5-i)&0xf];
        H[6] += (u64)f_i * g[(6-i)&0xf];
        H[7] += (u64)f_i * g[(7-i)&0xf];
        H[8] += (u64)f_i * g[(8-i)&0xf];
        H[9] += (u64)f_i * g[9-i];
    }
#else
    u64 H[10], temp;
    u32 g[10], f_i;

    f_i   = f[0];
    H[0]  = (u64)f_i * (g[0] = g_[0]);
    H[1]  = (u64)f_i * (g[1] = g_[1]);
    H[2]  = (u64)f_i * (g[2] = g_[2]);
    H[3]  = (u64)f_i * (g[3] = g_[3]);
    H[4]  = (u64)f_i * (g[4] = g_[4]);
    H[5]  = (u64)f_i * (g[5] = g_[5]);
    H[6]  = (u64)f_i * (g[6] = g_[6]);
    H[7]  = (u64)f_i * (g[7] = g_[7]);
    H[8]  = (u64)f_i * (g[8] = g_[8]);
    H[9]  = (u64)f_i * (g[9] = g_[9]);

    f_i   = f[1];       temp  = (u64)g[9] * (189<<4);
    H[0] += (u64)f_i * (g[9]  = (u32)temp & MASK26);
    H[1] += (u64)f_i * (g[0] += (u32)(temp >> 26));
    H[2] += (u64)f_i * g[1];
    H[3] += (u64)f_i * g[2];
    H[4] += (u64)f_i * g[3];
    H[5] += (u64)f_i * g[4];
    H[6] += (u64)f_i * g[5];
    H[7] += (u64)f_i * g[6];
    H[8] += (u64)f_i * g[7];
    H[9] += (u64)f_i * g[8];

    f_i   = f[2];       temp  = (u64)g[8] * (189<<4);
    H[0] += (u64)f_i * (g[8]  = (u32)temp & MASK26);
    H[1] += (u64)f_i * (g[9] += (u32)(temp >> 26));
    H[2] += (u64)f_i * g[0];
    H[3] += (u64)f_i * g[1];
    H[4] += (u64)f_i * g[2];
    H[5] += (u64)f_i * g[3];
    H[6] += (u64)f_i * g[4];
    H[7] += (u64)f_i * g[5];
    H[8] += (u64)f_i * g[6];
    H[9] += (u64)f_i * g[7];

    f_i   = f[3];       temp  = (u64)g[7] * (189<<4);
    H[0] += (u64)f_i * (g[7]  = (u32)temp & MASK26);
    H[1] += (u64)f_i * (g[8] += (u32)(temp >> 26));
    H[2] += (u64)f_i * g[9];
    H[3] += (u64)f_i * g[0];
    H[4] += (u64)f_i * g[1];
    H[5] += (u64)f_i * g[2];
    H[6] += (u64)f_i * g[3];
    H[7] += (u64)f_i * g[4];
    H[8] += (u64)f_i * g[5];
    H[9] += (u64)f_i * g[6];

    f_i   = f[4];       temp  = (u64)g[6] * (189<<4);
    H[0] += (u64)f_i * (g[6]  = (u32)temp & MASK26);
    H[1] += (u64)f_i * (g[7] += (u32)(temp >> 26));
    H[2] += (u64)f_i * g[8];
    H[3] += (u64)f_i * g[9];
    H[4] += (u64)f_i * g[0];
    H[5] += (u64)f_i * g[1];
    H[6] += (u64)f_i * g[2];
    H[7] += (u64)f_i * g[3];
    H[8] += (u64)f_i * g[4];
    H[9] += (u64)f_i * g[5];

    f_i   = f[5];       temp  = (u64)g[5] * (189<<4);
    H[0] += (u64)f_i * (g[5]  = (u32)temp & MASK26);
    H[1] += (u64)f_i * (g[6] += (u32)(temp >> 26));
    H[2] += (u64)f_i * g[7];
    H[3] += (u64)f_i * g[8];
    H[4] += (u64)f_i * g[9];
    H[5] += (u64)f_i * g[0];
    H[6] += (u64)f_i * g[1];
    H[7] += (u64)f_i * g[2];
    H[8] += (u64)f_i * g[3];
    H[9] += (u64)f_i * g[4];

    f_i   = f[6];       temp  = (u64)g[4] * (189<<4);
    H[0] += (u64)f_i * (g[4]  = (u32)temp & MASK26);
    H[1] += (u64)f_i * (g[5] += (u32)(temp >> 26));
    H[2] += (u64)f_i * g[6];
    H[3] += (u64)f_i * g[7];
    H[4] += (u64)f_i * g[8];
    H[5] += (u64)f_i * g[9];
    H[6] += (u64)f_i * g[0];
    H[7] += (u64)f_i * g[1];
    H[8] += (u64)f_i * g[2];
    H[9] += (u64)f_i * g[3];

    f_i   = f[7];       temp  = (u64)g[3] * (189<<4);
    H[0] += (u64)f_i * (g[3]  = (u32)temp & MASK26);
    H[1] += (u64)f_i * (g[4] += (u32)(temp >> 26));
    H[2] += (u64)f_i * g[5];
    H[3] += (u64)f_i * g[6];
    H[4] += (u64)f_i * g[7];
    H[5] += (u64)f_i * g[8];
    H[6] += (u64)f_i * g[9];
    H[7] += (u64)f_i * g[0];
    H[8] += (u64)f_i * g[1];
    H[9] += (u64)f_i * g[2];

    f_i   = f[8];       temp  = (u64)g[2] * (189<<4);
    H[0] += (u64)f_i * (g[2]  = (u32)temp & MASK26);
    H[1] += (u64)f_i * (g[3] += (u32)(temp >> 26));
    H[2] += (u64)f_i * g[4];
    H[3] += (u64)f_i * g[5];
    H[4] += (u64)f_i * g[6];
    H[5] += (u64)f_i * g[7];
    H[6] += (u64)f_i * g[8];
    H[7] += (u64)f_i * g[9];
    H[8] += (u64)f_i * g[0];
    H[9] += (u64)f_i * g[1];

    f_i   = f[9];       temp  = (u64)g[1] * (189<<4);
    H[0] += (u64)f_i * (g[1]  = (u32)temp & MASK26);
    H[1] += (u64)f_i * (g[2] += (u32)(temp >> 26));
    H[2] += (u64)f_i * g[3];
    H[3] += (u64)f_i * g[4];
    H[4] += (u64)f_i * g[5];
    H[5] += (u64)f_i * g[6];
    H[6] += (u64)f_i * g[7];
    H[7] += (u64)f_i * g[8];
    H[8] += (u64)f_i * g[9];
    H[9] += (u64)f_i * g[0];
#endif

    /* partial [lazy] reduction */
    H[1] += H[0] >> 26;         g[0] = (u32)H[0] & MASK26;
    H[3] += H[2] >> 26;         g[2] = (u32)H[2] & MASK26;
    H[5] += H[4] >> 26;         g[4] = (u32)H[4] & MASK26;
    H[7] += H[6] >> 26;         g[6] = (u32)H[6] & MASK26;
    H[9] += H[8] >> 26;         g[8] = (u32)H[8] & MASK26;

    H[0] = g[0] + (H[9] >> 22)*189; h[9] = (u32)H[9] & MASK22;

    g[2] += (u32)(H[1] >> 26);  h[1] = (u32)H[1] & MASK26;
    g[4] += (u32)(H[3] >> 26);  h[3] = (u32)H[3] & MASK26;
    g[6] += (u32)(H[5] >> 26);  h[5] = (u32)H[5] & MASK26;
    g[8] += (u32)(H[7] >> 26);  h[7] = (u32)H[7] & MASK26;

    h[1] += (u32)(H[0] >> 26);  h[0] = (u32)H[0] & MASK26;
    h[3] += g[2] >> 26;         h[2] = g[2]      & MASK26;
    h[5] += g[4] >> 26;         h[4] = g[4]      & MASK26;
    h[7] += g[6] >> 26;         h[6] = g[6]      & MASK26;
    h[9] += g[8] >> 26;         h[8] = g[8]      & MASK26;
}

static void sqr_mod_256_189(fe26 h, const fe26 f)
{
#ifdef __OPTIMIZE_SIZE__
    mul_mod_256_189(h, f, f);
#else   /* ~35% faster sqrt */
    u64 H[10], temp;
    u32 g[10], hi;

    g[0] = f[0];
    H[0] = (u64)g[0] * g[0];

    g[1] = f[1];
    H[2] = (u64)g[1] * g[1];        g[1] <<= 1;
    H[3] = (u64)g[1] * (g[2] = f[2]);
    H[4] = (u64)g[1] * (g[3] = f[3]);
    H[5] = (u64)g[1] * (g[4] = f[4]);
    H[6] = (u64)g[1] * (g[5] = f[5]);
    H[7] = (u64)g[1] * (g[6] = f[6]);
    H[8] = (u64)g[1] * (g[7] = f[7]);
    H[9] = (u64)g[1] * (g[8] = f[8]);
                       (g[9] = f[9]);

    H[4] += (u64)g[2] * g[2];       g[2] <<= 1;
    H[5] += (u64)g[2] * g[3];
    H[6] += (u64)g[2] * g[4];
    H[7] += (u64)g[2] * g[5];
    H[8] += (u64)g[2] * g[6];
    H[9] += (u64)g[2] * g[7];

    H[6] += (u64)g[3] * g[3];       g[3] <<= 1;
    H[7] += (u64)g[3] * g[4];
    H[8] += (u64)g[3] * g[5];
    H[9] += (u64)g[3] * g[6];

    H[8] += (u64)g[4] * g[4];       g[4] <<= 1;
    H[9] += (u64)g[4] * g[5];       g[5] <<= 1;
                                    g[6] <<= 1;
                                    g[7] <<= 1;
                                    g[8] <<= 1;

    temp  = (u64)g[9] * (189<<4);   hi = g[0] + (u32)(temp >> 26);
    H[1]  = (u64)g[1] * hi;
    H[2] += (u64)g[2] * hi;
    H[3] += (u64)g[3] * hi;
    H[4] += (u64)g[4] * hi;
    H[5] += (u64)g[5] * hi;
    H[6] += (u64)g[6] * hi;
    H[7] += (u64)g[7] * hi;
    H[8] += (u64)g[8] * hi;
    H[9] += (u64)g[9] * (hi + g[0]);
    H[0] += (u64)g[1] * (g[0] = (u32)temp & MASK26);
    H[7] += (u64)g[8] * g[0];
    H[8] += (u64)g[9] * g[0];

    temp  = (u64)g[8] * (189<<3);   hi = (u32)(temp >> 26);
    H[0] += (u64)g[2] * (g[1] = (u32)temp & MASK26);
    H[5] += (u64)g[7] * g[1];       g[8] >>= 1;
    H[6] += (u64)g[8] * g[1];
    H[7] += (u64)g[8] * hi;
    H[1] += (u64)g[2] * (g[0] += hi);
    H[2] += (u64)g[3] * g[0];
    H[3] += (u64)g[4] * g[0];
    H[4] += (u64)g[5] * g[0];
    H[5] += (u64)g[6] * g[0];
    H[6] += (u64)g[7] * g[0];

    temp  = (u64)g[7] * (189<<3);   hi = (u32)(temp >> 26);
    H[0] += (u64)g[3] * (g[2] = (u32)temp & MASK26);
    H[3] += (u64)g[6] * g[2];       g[7] >>= 1;
    H[4] += (u64)g[7] * g[2];
    H[5] += (u64)g[7] * hi;
    H[1] += (u64)g[3] * (g[1] += hi);
    H[2] += (u64)g[4] * g[1];
    H[3] += (u64)g[5] * g[1];
    H[4] += (u64)g[6] * g[1];

    temp  = (u64)g[6] * (189<<3);   hi = (u32)(temp >> 26);
    H[0] += (u64)g[4] * (g[3] = (u32)temp & MASK26);
    H[1] += (u64)g[5] * g[3];       g[6] >>= 1;
    H[2] += (u64)g[6] * g[3];
    H[3] += (u64)g[6] * hi;
    H[1] += (u64)g[4] * (g[2] += hi);
    H[2] += (u64)g[5] * g[2];

    temp  = (u64)g[5] * (189<<2);
    H[0] += (u64)g[5] * ((u32)temp & MASK26);
    H[1] += (u64)g[5] * ((u32)(temp >> 26));

    /* partial [lazy] reduction */
    H[1] += H[0] >> 26;             g[0] = (u32)H[0] & MASK26;
    H[3] += H[2] >> 26;             g[2] = (u32)H[2] & MASK26;
    H[5] += H[4] >> 26;             g[4] = (u32)H[4] & MASK26;
    H[7] += H[6] >> 26;             g[6] = (u32)H[6] & MASK26;
    H[9] += H[8] >> 26;             g[8] = (u32)H[8] & MASK26;

    H[0] = g[0] + (H[9] >> 22)*189; h[9] = (u32)H[9] & MASK22;

    g[2] += (u32)(H[1] >> 26);      h[1] = (u32)H[1] & MASK26;
    g[4] += (u32)(H[3] >> 26);      h[3] = (u32)H[3] & MASK26;
    g[6] += (u32)(H[5] >> 26);      h[5] = (u32)H[5] & MASK26;
    g[8] += (u32)(H[7] >> 26);      h[7] = (u32)H[7] & MASK26;

    h[1] += (u32)(H[0] >> 26);      h[0] = (u32)H[0] & MASK26;
    h[3] += g[2] >> 26;             h[2] = g[2]      & MASK26;
    h[5] += g[4] >> 26;             h[4] = g[4]      & MASK26;
    h[7] += g[6] >> 26;             h[6] = g[6]      & MASK26;
    h[9] += g[8] >> 26;             h[8] = g[8]      & MASK26;
#endif
}

static void sqr_n_mul_fe26(fe26 out, const fe26 a, size_t count,
                                       const fe26 b)
{
    fe26 t;

    while(count--) {
        sqr_mod_256_189(t, a);
        a = t;
    }
    mul_mod_256_189(out, t, b);
}

static bool_t sqrt_mod_256_189(vec256 out, const vec256 inp)
{
    fe26 x, y, z;
    vec256 sqr, ret;
    bool_t neg;

    to_fe26(z, inp);

#define sqr_n_mul sqr_n_mul_fe26
    sqr_n_mul(x, z, 1, z);      /* 0x3 */
    sqr_n_mul(y, x, 1, z);      /* 0x7 */
    sqr_n_mul(x, y, 3, y);      /* 0x3f */
    sqr_n_mul(x, x, 1, z);      /* 0x7f */
    sqr_n_mul(x, x, 7, x);      /* 0x3fff */
    sqr_n_mul(x, x, 14, x);     /* 0xfffffff */
    sqr_n_mul(x, x, 3, y);      /* 0x7fffffff */
    sqr_n_mul(x, x, 31, x);     /* 0x3fffffffffffffff */
    sqr_n_mul(x, x, 62, x);     /* 0xfffffffffffffffffffffffffffffff */
    sqr_n_mul(x, x, 124, x);    /* 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff */
    sqr_n_mul(x, x, 2, z);      /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd */
    sqr_n_mul(x, x, 4, z);      /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd1 */
#undef sqr_n_mul

    mul_mod_256_189(y, x, x);
    from_fe26(ret, x);
    from_fe26(sqr, y);
    neg = vec_is_equal(sqr, inp, sizeof(sqr)) ^ 1;
    cneg_mod_256_189(out, ret, neg ^ (bool_t)(ret[0]&1));

    return neg;
}

static void square_mod_256_189(vec256 out, const vec256 inp)
{
    bool_t neg = (bool_t)(inp[0]&1);
    fe26 x;

    to_fe26(x, inp);
    sqr_mod_256_189(x, x);
    from_fe26(out, x);
    cneg_mod_256_189(out, out, neg);
}

#endif

#ifdef __x86_64__cpuidex
# if defined(__clang__) /* apparently fails to recognize "rbx" as clobbered */
__attribute__((optnone))
# endif
static int is_adx_avaiable()
{
    static volatile int xfeat = 0;
    int info[4], ebx;

    if ((ebx = xfeat) == 0) {
        ebx = 1;
        __cpuidex(info, 0, 0);
        if (info[0] >= 7) {
            __cpuidex(info, 7, 0);
            ebx |= info[1];
        }
        xfeat = ebx;
    }
    return (ebx >> 19) & 1;
}
#endif

#ifdef __CUDA_ARCH__
static
#endif
int sloth256_189_encode(unsigned char *inout, size_t len,
                        const unsigned char iv_[32], size_t layers)
{
    bool_t ret = 0;
    size_t i;
    limb_t iv[32/sizeof(limb_t)], *feedback = iv;
    limb_t *block = (limb_t *)inout;
#if !defined(__OPENCL_C_VERSION__) && !defined(__CUDA_ARCH__)
    const union {
        long one;
        char little;
    } is_endian = { 1 };

    if (!is_endian.little || (size_t)inout%sizeof(limb_t) != 0) {
        len &= ((size_t)0-32);
        block = (limb_t *)alloca(len);
        limbs_from_le_bytes(block, inout, len);
    }
#elif defined(__LITTLE_ENDIAN__) || (__BYTE_ORDER__-0 == 1234)
    /* assert((size_t)inout%sizeof(limb_t) == 0); */
#else
# error "unsupported platform"
#endif

    limbs_from_le_bytes(iv, iv_, 32);

    len /= sizeof(limb_t);

#ifdef __x86_64__cpuidex
    if (is_adx_avaiable())
        while (layers--) {
            for (i = 0; i < len; i += 32/sizeof(limb_t)) {
                ret |= xor_n_check_mod_256_189(block+i, block+i, feedback);
                sqrtx_mod_256_189(block+i, block+i);
                feedback = block+i;
            }
        }
    else
#endif
    while (layers--) {
        for (i = 0; i < len; i += 32/sizeof(limb_t)) {
            ret |= xor_n_check_mod_256_189(block+i, block+i, feedback);
            sqrt_mod_256_189(block+i, block+i);
            feedback = block+i;
        }
    }

    if ((uptr_t)block != (uptr_t)inout)
        le_bytes_from_limbs(inout, block, len*sizeof(limb_t));

    return (int)ret;
}

#ifdef __CUDA_ARCH__
static
#endif
void sloth256_189_decode(unsigned char *inout, size_t len,
                         const unsigned char iv_[32], size_t layers)
{
    size_t i;
    limb_t iv[32/sizeof(limb_t)];
    limb_t *block = (limb_t *)inout;
#if !defined(__OPENCL_C_VERSION__) && !defined(__CUDA_ARCH__)
    const union {
        long one;
        char little;
    } is_endian = { 1 };

    if (!is_endian.little || (size_t)inout%sizeof(limb_t) != 0) {
        len &= ((size_t)0-32);
        block = (limb_t *)alloca(len);
        limbs_from_le_bytes(block, inout, len);
    }
#elif defined(__LITTLE_ENDIAN__) || (__BYTE_ORDER__-0 == 1234)
    /* assert((size_t)inout%sizeof(limb_t) == 0); */
#else
# error "unsupported platform"
#endif

    limbs_from_le_bytes(iv, iv_, 32);

    len /= sizeof(limb_t);

#ifdef __x86_64__cpuidex
    if (is_adx_avaiable())
        while (1) {
            for (i = len; i -= 32/sizeof(limb_t);) {
                squarex_mod_256_189(block+i, block+i);
                (void)xor_n_check_mod_256_189(block+i, block+i,
                                          block+i-32/sizeof(limb_t));
            }
            squarex_mod_256_189(block, block);
            if (--layers == 0)
                break;
            (void)xor_n_check_mod_256_189(block, block,
                                          block+len-32/sizeof(limb_t));
        }
    else
#endif
    while (1) {
        for (i = len; i -= 32/sizeof(limb_t);) {
            square_mod_256_189(block+i, block+i);
            (void)xor_n_check_mod_256_189(block+i, block+i,
                                          block+i-32/sizeof(limb_t));
        }
        square_mod_256_189(block, block);
        if (--layers == 0)
            break;
        (void)xor_n_check_mod_256_189(block, block,
                                      block+len-32/sizeof(limb_t));
    }
    (void)xor_n_check_mod_256_189(block, block, iv);

    if ((uptr_t)block != (uptr_t)inout)
        le_bytes_from_limbs(inout, block, len*sizeof(limb_t));
}
