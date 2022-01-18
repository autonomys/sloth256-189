static inline void vec_copy(void *ret, const void *a, size_t num)
{
    limb_t *rp = (limb_t *)ret;
    const limb_t *ap = (const limb_t *)a;
    size_t i;

    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = ap[i];
}

inline void t_op_eq(struct vec256_t* x, const vec256 a) {
    
    vec_copy(x->v, a, sizeof(vec256));
}

inline void set_off(struct vec256_shared_t* v, uint idx) { // idx = 0
    
    v->v_off = get_local_id(0) * 7 * sizeof(uint);
}

inline void shared_t_to_vec256_t(struct vec256_t* x, const struct vec256_shared_t a, __local uint scratchpad[]) {

    x->v[0] = a.v0;
    x->v[1] = scratchpad[a.v_off/sizeof(uint) + 0];
    x->v[2] = scratchpad[a.v_off/sizeof(uint) + 1];
    x->v[3] = scratchpad[a.v_off/sizeof(uint) + 2];
    x->v[4] = scratchpad[a.v_off/sizeof(uint) + 3];
    x->v[5] = scratchpad[a.v_off/sizeof(uint) + 4];
    x->v[6] = scratchpad[a.v_off/sizeof(uint) + 5];
    x->v[7] = scratchpad[a.v_off/sizeof(uint) + 6];
}

inline void shared_t_op_eq(struct vec256_shared_t* x, const struct vec256_t a, __local uint scratchpad[]) {

    barrier(CLK_LOCAL_MEM_FENCE);

    x->v0                                 = a.v[0];
    scratchpad[x->v_off/sizeof(uint) + 0] = a.v[1];
    scratchpad[x->v_off/sizeof(uint) + 1] = a.v[2];
    scratchpad[x->v_off/sizeof(uint) + 2] = a.v[3];
    scratchpad[x->v_off/sizeof(uint) + 3] = a.v[4];
    scratchpad[x->v_off/sizeof(uint) + 4] = a.v[5];
    scratchpad[x->v_off/sizeof(uint) + 5] = a.v[6];
    scratchpad[x->v_off/sizeof(uint) + 6] = a.v[7];

    barrier(CLK_LOCAL_MEM_FENCE);
}

static struct vec256_t sqrt_mod_256_189(const struct vec256_shared_t inp, __local uint scratchpad[])
{
    struct vec256_t x;

#define sqr_n_mul  sqr_n_mul_mod_256_189_val_or_self
    shared_t_to_vec256_t(&x, inp, scratchpad);
    t_op_eq(&x, sqr_n_mul(x, 1, 1, inp, scratchpad).l);  /* 0x3 */
    t_op_eq(&x, sqr_n_mul(x, 1, 1, inp, scratchpad).l);    /* 0x7 */
    t_op_eq(&x, sqr_n_mul(x, 3, 0, inp, scratchpad).l);    /* 0x3f */
    t_op_eq(&x, sqr_n_mul(x, 1, 1, inp, scratchpad).l);    /* 0x7f */
    t_op_eq(&x, sqr_n_mul(x, 7, 0, inp, scratchpad).l);    /* 0x3fff */
    t_op_eq(&x, sqr_n_mul(x, 1, 1, inp, scratchpad).l);    /* 0x7fff */
    t_op_eq(&x, sqr_n_mul(x, 15, 0, inp, scratchpad).l);   /* 0x3fffffff */
    t_op_eq(&x, sqr_n_mul(x, 1, 1, inp, scratchpad).l);    /* 0x7fffffff */
    t_op_eq(&x, sqr_n_mul(x, 31, 0, inp, scratchpad).l);   /* 0x3fffffffffffffff */
    t_op_eq(&x, sqr_n_mul(x, 62, 0, inp, scratchpad).l);   /* 0xfffffffffffffffffffffffffffffff */
    t_op_eq(&x, sqr_n_mul(x, 124, 0, inp, scratchpad).l);  /* 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff */
    t_op_eq(&x, sqr_n_mul(x, 2, 1, inp, scratchpad).l);    /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd */
    t_op_eq(&x, sqr_n_mul(x, 4, 1, inp, scratchpad).l);    /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd1 */

#undef sqr_n_mul

    t_op_eq(&x, redc_mod_256_189_val(x).l);
    //x = redc_mod_256_189_val(x);
    return sqr_n_cneg_mod_256_189_val(x, inp, scratchpad);
}

bool sloth256_189_encode(limb_t *piece, size_t len,
                         const limb_t *iv, size_t layers, __local uint scratchpad[])
{
    bool ret = 0;
    struct vec256_t x;
    struct vec256_shared_t inp; set_off(&inp, 0);

    len /= sizeof(limb_t);
    t_op_eq(&x, vec256_load_global(iv).l);

    shared_t_op_eq(&inp, x, scratchpad);

    while (layers--) {
        for (size_t i = 0; i < len; i += 32/sizeof(limb_t)) {
            t_op_eq(&x, vec256_load_global(piece + i).l);
            
            t_op_eq(&x, xor_256_val(x, inp, scratchpad).l);
            
            shared_t_op_eq(&inp, x, scratchpad);
            ret |= check_mod_256_189_val(x);

            t_op_eq(&x, sqrt_mod_256_189(inp, scratchpad).l);

            shared_t_op_eq(&inp, x, scratchpad);
            vec256_store_global(x, piece + i);
        }
    }

    return ret;
}