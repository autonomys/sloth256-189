#if defined(__x86_64) || defined(__x86_64__)
# if defined(__ELF__)
#  include "elf/mod256-189-x86_64.s"
# elif defined(_WIN64) || defined(__CYGWIN__)
#  include "coff/mod256-189-x86_64.s"
# elif defined(__APPLE__)
#  include "mach-o/mod256-189-x86_64.s"
# endif
#else
# error "unsupported platform"
#endif
