# sloth256_189.c Documentation

This is the documentation of [sloth256_189.c](sloth256_189.c)
- `sloth256_189.c`: general purpose low-level c code. A very efficient and optimized implementation of sloth encoding/decoding.

---

### sloth256_189.c

- ***sloth256_189_encode:***

    *High level summary:* this function is performing the encoding operation given in SLOTH.

    ```c
    int sloth256_189_encode(unsigned char *inout, size_t len,
                            const unsigned char iv_[32], size_t layers)
    ```

    Now, let's take a look what is does on the high-level. 

    ```cpp
    while (layers--) {
        for (i = 0; i < len; i += 32/sizeof(limb_t)) {
            ret |= xor_n_check_mod_256_189(block+i, block+i, feedback);
            sqrtx_mod_256_189(block+i, block+i);
            feedback = block+i;
        }
    ```

    For pseudo-code correspondent, here is the more readable version:

    ```cpp
    while (layers--) {  // for each layer
        for (i = 0; i < len; i += 1) {  // each i corresponds to a 256-bit chunk
            block[i] = xor(block[i], feedback);  // XORing block[i] and feedback
            block[i] = sqrtx_mod_256_189(block[i]);  // square_root of block[i]
            feedback = block[i];  // update the feedback
        }
    ```

    Now, let's dive a bit deeper. `sloth256_189_encode` calls:

    1. `xor_n_check_mod_256_189`
    2. Then calls `sqrtx_mod_256_189`.
    3. finally, updates the `feedback`.
    <br><br>
  - ***xor_n_check_mod_256_189:***

      *High-level summary:* this function is applying xor operation to its parameters (a and b), then writes the output to its first parameter (out).

      ```c
      static bool_t xor_n_check_mod_256_189(vec256 out,
                                            const vec256 a, const vec256 b)
      ```

      The reason for there is a `check` inside the function name is: after the xor operation, the result may be larger than our modulo (prime), and we always want our results to be reduced in modulo prime.

      Let's take a look at the code from high-level perspective:

      ```c
      static bool_t xor_n_check_mod_256_189(vec256 out, const vec256 a,
                                                        const vec256 b)
      {
          size_t i;
          limb_t carry;

              // this is the part that performs XOR operation between parameters a and b
          for (i = 0; i < sizeof(vec256)/sizeof(limb_t); i++)
              out[i] = a[i] ^ b[i];

              // this is the part that checks for if the result is greated than the prime
          carry = (out[0] + 189) < 189;  // the reason we are checking
              // for the least significant part being smaller than 189 is the following:
              // prime = 2^256 - 189. So if we add 189 to the least significant limb,
              // and the result is smaller than 189, that means an overflow happened
              // which means, our least significant limb is greater than the prime's
              // least significant bit. So we use a carry to inspect other limbs as well
              // if all of them happen to be greater than the prime's limbs
              // that means our number is greater than the prime itself.
          for (i = 1; i < sizeof(vec256)/sizeof(limb_t); i++)
              carry = (out[i] + carry) < carry;


          return (bool_t)carry;
      }
      ```

  - ***sqrtx_mod_256_189:***

      *High level summary:* this function is taking the square root of its parameter (inp), and writes the result to its first parameter (out)

      ```c
      static bool_t sqrtx_mod_256_189(vec256 out, const vec256 inp)
      ```

      The whole code:

      ```c
      static bool_t sqrtx_mod_256_189(vec256 out, const vec256 inp)
      {
          vec256 x, y;
          bool_t neg;

      // addition chain - ignore the hexadecimals down there for now
      // details will be explained later, but basically what this does is:
      // x = (input ^ ((p+1)/ 4)) % p
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

              // since the result is not fully-reduced (lazy-reduce is used)
              // another reduce is necessary in the end of addition chain
          redc_mod_256_189(x, x);

              // take the square of x, write the result into y
          mulx_mod_256_189(y, x, x);  /* mulx is in cache, sqrx isn't */

              // check if square of x is the same with input (this check is necessary,
              // since there are some numbers that do not have square roots,
              // but their negations do have, read the SLOTH paper for more details).
              // Then negate the result, and assign this to variable `neg`.
              // So, if our computed square_root's square equals to input, this means
              // the input indeed has a square root and we found it. And in this case
              // `neg` will be equal to 0. Otherwise, it will be equal to 1.
          neg = vec_is_equal(y, inp, sizeof(y)) ^ 1;

              // actually the above comment was not fully correct. Even for the numbers
              // that do not have square root, we assign a square root for them.
              // since a number will have 2 square roots, one of these square roots
              // is assigned to the original numbers negation (read SLOTH).
              // This assignment is based on square root being odd or even
              // so we apply a logic (neg ^ number.isOdd())
              // and accordingly, negate the square root or not
          cneg_mod_256_189(out, x, neg ^ (bool_t)(x[0]&1));

          return neg;
      }
      ```

      Inner parts of the `sqrtx_mod_256_189`:

      - ***sqr_n_mul (a.k.a. sqrx_n_mul_mod_256_189):***

          *High level summary:* this function takes the square of its last parameter (b), and multiplies this result with itself `count` times (like taking power of it), and finally multiplies this result with its second parameter (a), in modulo p, and writes the final result to its first parameter (out).

            ```c
            void sqrx_n_mul_mod_256_189(vec256 out, const vec256 a,
                                        size_t count, const vec256 b);
            ```

      - ***redc_mod_256_189:***

          *High level summary:* this function reduces its last parameter (a), in modulo prime, and writes the result back into its first parameter (out).

          ```c
          void redc_mod_256_189(vec256 out, const vec256 a);
          ```

      - ***mulx_mod_256_189:***

          *High level summary:* this function multiplies its last two parameters (a, b) in modulo p, and writes the result back to its first parameter (out).

          ```c
          void mulx_mod_256_189(vec256 out, const vec256 a, const vec256 b);
          ```

      - ***vec_is_equal:***

          *High level summary:* this function checks whether two vectors are equal. The last parameter defines till which point this check will be applied. Maybe these two vectors are not equal, in this case, only the first (amount will be determined by the last parameter) limbs will be compared.

          ```c
          static inline bool_t vec_is_equal(const void *a, const void *b,
                                                                              size_t num)
          ```

          The code:

          ```c
          static inline bool_t vec_is_equal(const void *a, const void *b,
                                                                              size_t num)
          {
                  // pointer to a and b
              const limb_t *ap = (const limb_t *)a;
              const limb_t *bp = (const limb_t *)b;
              limb_t acc;
              size_t i;

                  // determine how many limbs will be checked for comparison
              num /= sizeof(limb_t);

                  // if two limbs are equal, `ap[i] ^ bp[i]` will return 0.
                  // this result will be `|=` to the final result.
                  // It makes sense, since for any non-equality among the limbs
                  // the operation `ap[i] ^ bp[i]` will return 1, and this result
                  // will be used in `acc |= 1`, and the result will never be 0 again
              for (acc = 0, i = 0; i < num; i++)
                  acc |= ap[i] ^ bp[i];

                  // another very optimized function to check if limb is full of 0's
              return is_zero(acc);
          }
          ```

          - ***is_zero:***

              *High level summary:* this function checks whether the given limb is zero or not

              ```c
              static inline bool_t is_zero(limb_t l)
              ```

              The code:

              ```c
              static inline bool_t is_zero(limb_t l)
              {
                      return (~l & (l - 1)) >> (LIMB_T_BITS - 1);
              }
              ```

              Negating all the bits in the limb, then `and`ing this with the `limb - 1`, and lastly taking the most significant bit of this. Notice that, it is only possible return 1 in this setting, if the input is zero. Because if the MSB needs to be 1, this means the MSB result of the `~l & (l-1)` needs to be 1. And for this, both `~l` and `l-1` 's MSBs needs to be 1.

              For `~l`'s MSB to be 1, `l`'s MSB needs to be 0. If `l`'s MSB is 0, then `l-1`'s MSB can only be 1, if there is underflow. For `l` to underflow with `l-1`, `l` needs to be 0.

              - Examples (consider limb is 4 bit):
                  - limb: 0001
                      - ~limb: 1110
                      - limb-1: 0000
                      - ~limb & limb-1 : 0000
                      - MSB of result: 0
                  - limb: 1110
                      - ~limb: 0001
                      - limb-1: 1101
                      - ~limb & limb-1: 0001
                      - MSB of result: 0
                  - limb: 1000
                      - ~limb: 0111
                      - limb-1: 0111
                      - ~limb & limb-1: 0111
                      - MSB of result: 0
                  - limb: 0000
                      - ~limb: 1111
                      - limb-1: 1111
                      - ~limb & limb-1: 1111
                      - MSB of result: 1

      - ***cneg_mod_256_189:***

          *High level summary:* this function probably negates its second parameter (a), and writes the result into its first parameter (out). Do not know yet what the third parameter is doing.

          ```c
          static void cneg_mod_256_189(vec256 out, const vec256 a, bool_t cbit)
          ```
