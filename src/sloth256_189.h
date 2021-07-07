#ifdef __SIZE_TYPE__
typedef __SIZE_TYPE__ size_t;
#else
#include <stddef.h>
#endif

#ifdef __UINT8_TYPE__
typedef __UINT8_TYPE__  uint8_t;
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#elif defined(__BLST_CGO__)
typedef _Bool bool; /* it's assumed that cgo calls modern enough compiler */
#elif defined(__STDC_VERSION__) && __STDC_VERSION__>=199901
# define bool _Bool
#else
# define bool int
#endif

bool sloth256_189_encode(uint8_t *inout, size_t len, const uint8_t iv_[32],
                         size_t layers);
void sloth256_189_decode(uint8_t *inout, size_t len, const uint8_t iv_[32],
                         size_t layers);
#ifdef __cplusplus
}
#endif
