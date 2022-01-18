#ifdef __UINTPTR_TYPE__
typedef __UINTPTR_TYPE__ uptr_t;
#else
typedef const void* uptr_t;
#endif

typedef uint limb_t;

#define LIMB_T_BITS (8*sizeof(limb_t))

typedef limb_t vec256[256 / LIMB_T_BITS];
//typedef limb_t bool_t;

static inline void private_limbs_from_global(limb_t* ret, const __global limb_t* in, uchar n)
{
    limb_t limb = 0;

    while (n--) {
        //limb <<= 8;
        //limb |= in[n];
        /*
         * 'if (n % sizeof(limb_t) == 0)' is omitted because it's cheaper
         * to perform redundant stores than to pay penalty for
         * mispredicted branch. Besides, some compilers unroll the
         * loop and remove redundant stores to 'restict'-ed storage...
         */
        //ret[n / sizeof(limb_t)] = limb;
        ret[n] = in[n];
    }
}

static inline void private_limbs_to_global(__global limb_t* out, const limb_t* in, uchar n)
{
    /*limb_t limb;
    uint i, j, r;

    r = n % sizeof(limb_t);
    n /= sizeof(limb_t);

    for (i = 0; i < n; i++) {
        for (limb = in[i], j = 0; j < sizeof(limb_t); j++, limb >>= 8)
            *out++ = (unsigned char)limb;
    }*/
    /*if (r) {
        for (limb = in[i], j = 0; j < r; j++, limb >>= 8)
            *out++ = (unsigned char)limb;
    }*/

    while (n--) {

        out[n] = in[n];
    }
}

static inline bool is_zero(limb_t l)
{
    return (~l & (l - 1)) >> (LIMB_T_BITS - 1);
}

static inline bool vec_is_equal(const limb_t* a, const limb_t* b, uchar num)
{
    limb_t acc;
    uchar i;

    num /= sizeof(limb_t);

    for (acc = 0, i = 0; i < num; i++)
        acc |= a[i] ^ b[i];

    return is_zero(acc);
}

static void cneg_mod_256_189(vec256 out, const vec256 a, const bool cbit)
{
    limb_t mask, carry;
    uchar i;

    for (mask = 0, i = 0; i < sizeof(vec256) / sizeof(limb_t); i++)
        mask |= a[i];

    mask = cbit & (is_zero(mask) ^ 1);
    mask = 0 - mask;

    out[0] = a[0] + (189 & mask); carry = out[0] < (189 & mask);
    for (i = 1; i < sizeof(vec256) / sizeof(limb_t); i++)
        out[i] = a[i] + carry, carry = out[i] < carry;
    for (i = 0; i < sizeof(vec256) / sizeof(limb_t); i++)
        out[i] ^= mask;
    out[0] += (1 & mask); carry = out[0] < (1 & mask);
    for (i = 1; i < sizeof(vec256) / sizeof(limb_t); i++)
        out[i] += carry, carry = out[i] < carry;
}

static bool xor_n_check_mod_256_189(vec256 out, const vec256 a, const vec256 b)
{
    uchar i;
    bool carry;

    for (i = 0; i < sizeof(vec256) / sizeof(limb_t); i++)
        out[i] = a[i] ^ b[i];

    carry = (out[0] + 189) < 189;
    for (i = 1; i < sizeof(vec256) / sizeof(limb_t); i++)
        carry = (out[i] + carry) < carry;

    return carry;
}

typedef uint u32;
typedef ulong u64;
typedef u32 fe26[10];           /* 9x26+22=256 */

#define MASK26 0x3ffffff
#define MASK22 0x3fffff

static void to_fe26(fe26 out, const void* in_)
{
    const u32* in = (const u32*)in_;

    out[0] = (in[0]) & MASK26;
    out[1] = (in[0] >> 26 | in[1] << 6) & MASK26;
    out[2] = (in[1] >> 20 | in[2] << 12) & MASK26;
    out[3] = (in[2] >> 14 | in[3] << 18) & MASK26;
    out[4] = (in[3] >> 8 | in[4 ] << 24) & MASK26;
    out[5] = (in[4] >> 2) & MASK26;
    out[6] = (in[4] >> 28 | in[5] << 4) & MASK26;
    out[7] = (in[5] >> 22 | in[6] << 10) & MASK26;
    out[8] = (in[6] >> 16 | in[7] << 16) & MASK26;
    out[9] = (in[7] >> 10);
}

static void from_fe26(void* out_, const fe26 in)
{
    u32 h0, q, * out = (u32*)out_;
    u64 t;

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

    t = (u64)h0;
    t += (u64)in[1] << 26;    out[0] = (u32)t;    t >>= 32;
    t += (u64)in[2] << 20;    out[1] = (u32)t;    t >>= 32;
    t += (u64)in[3] << 14;    out[2] = (u32)t;    t >>= 32;
    t += (u64)in[4] << 8;     out[3] = (u32)t;    t >>= 32;
    t += (u64)in[5] << 2;
    t += (u64)in[6] << 28;    out[4] = (u32)t;    t >>= 32;
    t += (u64)in[7] << 22;    out[5] = (u32)t;    t >>= 32;
    t += (u64)in[8] << 16;    out[6] = (u32)t;    t >>= 32;
    t += (u64)in[9] << 10;    out[7] = (u32)t;
}

#if defined(__INTEL_GPU__)
inline u64 mul(u32 x, u32 y)
{
    return (u64)x * y;
}

inline u64 mad(u64 z, u32 x, u32 y)
{
    return (u64)x * y + z;
}
#elif defined(__AMD_GPU__)
inline u64 mul(u32 x, u32 y)
{
    union { u64 W; u32 w[2]; } z;
    __asm__("v_mul_lo_u32 %0, %1, %2"
                 : "=v"(z.w[0]) : "v"(x), "v"(y));
    __asm__("v_mul_hi_u32 %0, %1, %2"
                 : "=v"(z.w[1]) : "v"(x), "v"(y));
    return z.W;
}

inline u64 mad(u64 z, u32 x, u32 y)
{
    VCC_T junk;

    __asm__("v_mad_u64_u32 %0, %1, %2, %3, %0"
                 : "+v" (z), "=s" (junk) : "v" (x), "v" (y));
    return z;
}
#endif

inline void mul_mod_256_189(fe26 h, const fe26 f, const fe26 g_)
{
    u64 H[10], temp;
    u32 g[10], f_i;

    f_i = f[0];
    H[0] = mul(f_i, g[0] = g_[0]);
    H[1] = mul(f_i, g[1] = g_[1]);
    H[2] = mul(f_i, g[2] = g_[2]);
    H[3] = mul(f_i, g[3] = g_[3]);
    H[4] = mul(f_i, g[4] = g_[4]);
    H[5] = mul(f_i, g[5] = g_[5]);
    H[6] = mul(f_i, g[6] = g_[6]);
    H[7] = mul(f_i, g[7] = g_[7]);
    H[8] = mul(f_i, g[8] = g_[8]);
    H[9] = mul(f_i, g[9] = g_[9]);

    f_i = f[1];           temp = mul(g[9], 189 << 4);
    H[0] = mad(H[0], f_i, (g[9] = (u32)temp & MASK26));
    H[1] = mad(H[1], f_i, (g[0] += (u32)(temp >> 26)));
    H[2] = mad(H[2], f_i, g[1]);
    H[3] = mad(H[3], f_i, g[2]);
    H[4] = mad(H[4], f_i, g[3]);
    H[5] = mad(H[5], f_i, g[4]);
    H[6] = mad(H[6], f_i, g[5]);
    H[7] = mad(H[7], f_i, g[6]);
    H[8] = mad(H[8], f_i, g[7]);
    H[9] = mad(H[9], f_i, g[8]);

    f_i = f[2];           temp = mul(g[8], 189 << 4);
    H[0] = mad(H[0], f_i, (g[8] = (u32)temp & MASK26));
    H[1] = mad(H[1], f_i, (g[9] += (u32)(temp >> 26)));
    H[2] = mad(H[2], f_i, g[0]);
    H[3] = mad(H[3], f_i, g[1]);
    H[4] = mad(H[4], f_i, g[2]);
    H[5] = mad(H[5], f_i, g[3]);
    H[6] = mad(H[6], f_i, g[4]);
    H[7] = mad(H[7], f_i, g[5]);
    H[8] = mad(H[8], f_i, g[6]);
    H[9] = mad(H[9], f_i, g[7]);

    f_i = f[3];           temp = mul(g[7], 189 << 4);
    H[0] = mad(H[0], f_i, (g[7] = (u32)temp & MASK26));
    H[1] = mad(H[1], f_i, (g[8] += (u32)(temp >> 26)));
    H[2] = mad(H[2], f_i, g[9]);
    H[3] = mad(H[3], f_i, g[0]);
    H[4] = mad(H[4], f_i, g[1]);
    H[5] = mad(H[5], f_i, g[2]);
    H[6] = mad(H[6], f_i, g[3]);
    H[7] = mad(H[7], f_i, g[4]);
    H[8] = mad(H[8], f_i, g[5]);
    H[9] = mad(H[9], f_i, g[6]);

    f_i = f[4];           temp = mul(g[6], 189 << 4);
    H[0] = mad(H[0], f_i, (g[6] = (u32)temp & MASK26));
    H[1] = mad(H[1], f_i, (g[7] += (u32)(temp >> 26)));
    H[2] = mad(H[2], f_i, g[8]);
    H[3] = mad(H[3], f_i, g[9]);
    H[4] = mad(H[4], f_i, g[0]);
    H[5] = mad(H[5], f_i, g[1]);
    H[6] = mad(H[6], f_i, g[2]);
    H[7] = mad(H[7], f_i, g[3]);
    H[8] = mad(H[8], f_i, g[4]);
    H[9] = mad(H[9], f_i, g[5]);

    f_i = f[5];           temp = mul(g[5], 189 << 4);
    H[0] = mad(H[0], f_i, (g[5] = (u32)temp & MASK26));
    H[1] = mad(H[1], f_i, (g[6] += (u32)(temp >> 26)));
    H[2] = mad(H[2], f_i, g[7]);
    H[3] = mad(H[3], f_i, g[8]);
    H[4] = mad(H[4], f_i, g[9]);
    H[5] = mad(H[5], f_i, g[0]);
    H[6] = mad(H[6], f_i, g[1]);
    H[7] = mad(H[7], f_i, g[2]);
    H[8] = mad(H[8], f_i, g[3]);
    H[9] = mad(H[9], f_i, g[4]);

    f_i = f[6];           temp = mul(g[4], 189 << 4);
    H[0] = mad(H[0], f_i, (g[4] = (u32)temp & MASK26));
    H[1] = mad(H[1], f_i, (g[5] += (u32)(temp >> 26)));
    H[2] = mad(H[2], f_i, g[6]);
    H[3] = mad(H[3], f_i, g[7]);
    H[4] = mad(H[4], f_i, g[8]);
    H[5] = mad(H[5], f_i, g[9]);
    H[6] = mad(H[6], f_i, g[0]);
    H[7] = mad(H[7], f_i, g[1]);
    H[8] = mad(H[8], f_i, g[2]);
    H[9] = mad(H[9], f_i, g[3]);

    f_i = f[7];           temp = mul(g[3], 189 << 4);
    H[0] = mad(H[0], f_i, (g[3] = (u32)temp & MASK26));
    H[1] = mad(H[1], f_i, (g[4] += (u32)(temp >> 26)));
    H[2] = mad(H[2], f_i, g[5]);
    H[3] = mad(H[3], f_i, g[6]);
    H[4] = mad(H[4], f_i, g[7]);
    H[5] = mad(H[5], f_i, g[8]);
    H[6] = mad(H[6], f_i, g[9]);
    H[7] = mad(H[7], f_i, g[0]);
    H[8] = mad(H[8], f_i, g[1]);
    H[9] = mad(H[9], f_i, g[2]);

    f_i = f[8];           temp = mul(g[2], 189 << 4);
    H[0] = mad(H[0], f_i, (g[2] = (u32)temp & MASK26));
    H[1] = mad(H[1], f_i, (g[3] += (u32)(temp >> 26)));
    H[2] = mad(H[2], f_i, g[4]);
    H[3] = mad(H[3], f_i, g[5]);
    H[4] = mad(H[4], f_i, g[6]);
    H[5] = mad(H[5], f_i, g[7]);
    H[6] = mad(H[6], f_i, g[8]);
    H[7] = mad(H[7], f_i, g[9]);
    H[8] = mad(H[8], f_i, g[0]);
    H[9] = mad(H[9], f_i, g[1]);

    f_i = f[9];           temp = mul(g[1], 189 << 4);
    H[0] = mad(H[0], f_i, (g[1] = (u32)temp & MASK26));
    H[1] = mad(H[1], f_i, (g[2] += (u32)(temp >> 26)));
    H[2] = mad(H[2], f_i, g[3]);
    H[3] = mad(H[3], f_i, g[4]);
    H[4] = mad(H[4], f_i, g[5]);
    H[5] = mad(H[5], f_i, g[6]);
    H[6] = mad(H[6], f_i, g[7]);
    H[7] = mad(H[7], f_i, g[8]);
    H[8] = mad(H[8], f_i, g[9]);
    H[9] = mad(H[9], f_i, g[0]);

    /* partial [lazy] reduction */
    H[1] += H[0] >> 26;         g[0] = (u32)H[0] & MASK26;
    H[3] += H[2] >> 26;         g[2] = (u32)H[2] & MASK26;
    H[5] += H[4] >> 26;         g[4] = (u32)H[4] & MASK26;
    H[7] += H[6] >> 26;         g[6] = (u32)H[6] & MASK26;
    H[9] += H[8] >> 26;         g[8] = (u32)H[8] & MASK26;

    H[0] = g[0] + (H[9] >> 22) * 189; h[9] = (u32)H[9] & MASK22;

    g[2] += (u32)(H[1] >> 26);  h[1] = (u32)H[1] & MASK26;
    g[4] += (u32)(H[3] >> 26);  h[3] = (u32)H[3] & MASK26;
    g[6] += (u32)(H[5] >> 26);  h[5] = (u32)H[5] & MASK26;
    g[8] += (u32)(H[7] >> 26);  h[7] = (u32)H[7] & MASK26;

    h[1] += (u32)(H[0] >> 26);  h[0] = (u32)H[0] & MASK26;
    h[3] += g[2] >> 26;         h[2] = g[2] & MASK26;
    h[5] += g[4] >> 26;         h[4] = g[4] & MASK26;
    h[7] += g[6] >> 26;         h[6] = g[6] & MASK26;
    h[9] += g[8] >> 26;         h[8] = g[8] & MASK26;
}

static void sqr_mod_256_189(fe26 h, const fe26 f)
{
    u64 H[10], temp;
    u32 g[10], hi;

    g[0] = f[0];
    H[0] = mul(g[0], g[0]);

    g[1] = f[1];
    H[2] = mul(g[1], g[1]);         g[1] <<= 1;
    H[3] = mul(g[1], g[2] = f[2]);
    H[4] = mul(g[1], g[3] = f[3]);
    H[5] = mul(g[1], g[4] = f[4]);
    H[6] = mul(g[1], g[5] = f[5]);
    H[7] = mul(g[1], g[6] = f[6]);
    H[8] = mul(g[1], g[7] = f[7]);
    H[9] = mul(g[1], g[8] = f[8]);
    (g[9] = f[9]);

    H[4] = mad(H[4], g[2], g[2]);   g[2] <<= 1;
    H[5] = mad(H[5], g[2], g[3]);
    H[6] = mad(H[6], g[2], g[4]);
    H[7] = mad(H[7], g[2], g[5]);
    H[8] = mad(H[8], g[2], g[6]);
    H[9] = mad(H[9], g[2], g[7]);

    H[6] = mad(H[6], g[3], g[3]);       g[3] <<= 1;
    H[7] = mad(H[7], g[3], g[4]);
    H[8] = mad(H[8], g[3], g[5]);
    H[9] = mad(H[9], g[3], g[6]);

    H[8] = mad(H[8], g[4], g[4]);       g[4] <<= 1;
    H[9] = mad(H[9], g[4], g[5]);       g[5] <<= 1;
    g[6] <<= 1;
    g[7] <<= 1;
    g[8] <<= 1;

    temp = mul(g[9], (189 << 4));   hi = g[0] + (u32)(temp >> 26);
    H[1] = mul(g[1], hi);
    H[2] = mad(H[2], g[2], hi);
    H[3] = mad(H[3], g[3], hi);
    H[4] = mad(H[4], g[4], hi);
    H[5] = mad(H[5], g[5], hi);
    H[6] = mad(H[6], g[6], hi);
    H[7] = mad(H[7], g[7], hi);
    H[8] = mad(H[8], g[8], hi);
    H[9] = mad(H[9], g[9], hi + g[0]);
    H[0] = mad(H[0], g[1], (g[0] = (u32)temp & MASK26));
    H[7] = mad(H[7], g[8], g[0]);
    H[8] = mad(H[8], g[9], g[0]);

    temp = mul(g[8], (189 << 3));   hi = (u32)(temp >> 26);
    H[0] = mad(H[0], g[2], (g[1] = (u32)temp & MASK26));
    H[5] = mad(H[5], g[7], g[1]);   g[8] >>= 1;
    H[6] = mad(H[6], g[8], g[1]);
    H[7] = mad(H[7], g[8], hi);
    H[1] = mad(H[1], g[2], (g[0] += hi));
    H[2] = mad(H[2], g[3], g[0]);
    H[3] = mad(H[3], g[4], g[0]);
    H[4] = mad(H[4], g[5], g[0]);
    H[5] = mad(H[5], g[6], g[0]);
    H[6] = mad(H[6], g[7], g[0]);

    temp = mul(g[7], (189 << 3));   hi = (u32)(temp >> 26);
    H[0] = mad(H[0], g[3], (g[2] = (u32)temp & MASK26));
    H[3] = mad(H[3], g[6], g[2]);   g[7] >>= 1;
    H[4] = mad(H[4], g[7], g[2]);
    H[5] = mad(H[5], g[7], hi);
    H[1] = mad(H[1], g[3], (g[1] += hi));
    H[2] = mad(H[2], g[4], g[1]);
    H[3] = mad(H[3], g[5], g[1]);
    H[4] = mad(H[4], g[6], g[1]);

    temp = mul(g[6], (189 << 3));   hi = (u32)(temp >> 26);
    H[0] = mad(H[0], g[4], (g[3] = (u32)temp & MASK26));
    H[1] = mad(H[1], g[5], g[3]);   g[6] >>= 1;
    H[2] = mad(H[2], g[6], g[3]);
    H[3] = mad(H[3], g[6], hi);
    H[1] = mad(H[1], g[4], (g[2] += hi));
    H[2] = mad(H[2], g[5], g[2]);

    temp = mul(g[5], (189 << 2));
    H[0] = mad(H[0], g[5], ((u32)temp & MASK26));
    H[1] = mad(H[1], g[5], ((u32)(temp >> 26)));

    /* partial [lazy] reduction */
    H[1] += H[0] >> 26;             g[0] = (u32)H[0] & MASK26;
    H[3] += H[2] >> 26;             g[2] = (u32)H[2] & MASK26;
    H[5] += H[4] >> 26;             g[4] = (u32)H[4] & MASK26;
    H[7] += H[6] >> 26;             g[6] = (u32)H[6] & MASK26;
    H[9] += H[8] >> 26;             g[8] = (u32)H[8] & MASK26;

    H[0] = g[0] + (H[9] >> 22) * 189; h[9] = (u32)H[9] & MASK22;

    g[2] += (u32)(H[1] >> 26);      h[1] = (u32)H[1] & MASK26;
    g[4] += (u32)(H[3] >> 26);      h[3] = (u32)H[3] & MASK26;
    g[6] += (u32)(H[5] >> 26);      h[5] = (u32)H[5] & MASK26;
    g[8] += (u32)(H[7] >> 26);      h[7] = (u32)H[7] & MASK26;

    h[1] += (u32)(H[0] >> 26);      h[0] = (u32)H[0] & MASK26;
    h[3] += g[2] >> 26;             h[2] = g[2] & MASK26;
    h[5] += g[4] >> 26;             h[4] = g[4] & MASK26;
    h[7] += g[6] >> 26;             h[6] = g[6] & MASK26;
    h[9] += g[8] >> 26;             h[8] = g[8] & MASK26;
}

inline void fe26_copy(fe26 out, const fe26 in)
{
    for (uchar i = 0; i < 10; i++)
        out[i] = in[i];
}

inline void fe26_copy_val(fe26 out, const fe26 in1, const fe26 in2, const bool val)
{
    for (uchar i = 0; i < 10; i++)
        out[i] = val ? in1[i] : in2[i];
}

static void sqr_n_mul_fe26(fe26 out, fe26 a, uchar count, const fe26 b)
{
    while (count--)
        sqr_mod_256_189(a, a);

    mul_mod_256_189(out, a, b);
}

__constant uchar loop_args[] = { 0x3, 0x3, 0x6, 0x3, 0xe, 0x3, 0x1e, 0x3, 0x3e, 0x7c, 0xf8, 0x5, 0x9 };

static bool sqrt_mod_256_189(vec256 out, const vec256 inp)
{
    fe26 x, z;
    vec256 sqr, ret;
    bool neg;

    to_fe26(z, inp);
    fe26_copy(x, z);

    fe26 var;
    for (uchar i = 0; i < 13; i++) {
        fe26_copy_val(var, z, x, loop_args[i] & 1);
        sqr_n_mul_fe26(x, x, loop_args[i] >> 1, var);
    }

    //mul_mod_256_189(z, x, x);
    sqr_mod_256_189(z, x);
    from_fe26(ret, x);
    from_fe26(sqr, z);
    neg = vec_is_equal(sqr, inp, sizeof(sqr)) ^ 1;
    cneg_mod_256_189(out, ret, neg ^ (ret[0] & 1));

    return neg;
}

int sloth256_189_encode(__global limb_t* inout, u32 len, const __global limb_t iv[32], u32 layers)
{
    bool ret = 0;
    u32 i = 0;
    limb_t feedback[32 / sizeof(limb_t)], block[32 / sizeof(limb_t)];

    private_limbs_from_global(block, iv, 32 / sizeof(limb_t));

    while (layers--) {

        for (i = 0; i < len / sizeof(limb_t); i += 32 / sizeof(limb_t)) {
            private_limbs_from_global(feedback, inout + i, 32 / sizeof(limb_t));
            ret |= xor_n_check_mod_256_189(block, block, feedback);
            sqrt_mod_256_189(block, block);
            private_limbs_to_global(inout + i, block, 32 / sizeof(limb_t));
        }
    }

    return (int)ret;
}
