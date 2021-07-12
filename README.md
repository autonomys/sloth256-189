spartan-sloth:
```
Software/Encode-single   time:   [2.0037 ms 2.0051 ms 2.0065 ms]
Software/Encode-parallel time:   [401.25 us 403.03 us 404.78 us]
Software/Decode          time:   [45.105 us 45.223 us 45.336 us]
```

sloth256-189-wip:
```
Software/Encode-single   time:   [278.35 us 278.42 us 278.49 us]  7.2x
Software/Encode-parallel time:   [49.085 us 49.243 us 49.421 us]  8.2x
Software/Decode          time:   [1.5111 us 1.5115 us 1.5120 us]  30x
```

`sqrt_mod_256_189/square_mod_256_189` comes in three "flavours":
```
#if detected(__ADX__)
 x86_64 assembly-assisted code path, executed on processors with ADX
 ISA extension.
 Field element is represented as 4 64-bit limbs. Amount of limb
 multiplication operations is 4*4+5.
#elif __SIZEOF_INT128__-0==16
 Any 64-bit platform with compiler that supports __int128 type, such
 as contemporary gcc or clang, but not Microsoft C.
 Field element is represented as 5 limbs, 1st is 52- and the rest are
 51-bit values. Amount of limb multiplication operations is 5*5+5.
#else
 Fallback for everything else. Presumably even suitable for OpenCL.
 Field element is represented as 12 limbs, a mixture of 22- and 21-bit
 values [with the last one being 19 bits]. Amount of limb multiplication
 operations is 12*12+14. Sensible to vectorize on 32-bit platforms.
#endif
```
Abovementioned amounts of limb multiplication operations don't directly
translate to performance, but can be used as crude first-order
approximations. As far as sqrt operation goes, all thee code paths follow
the same pattern:
```
raise input to ((2**256-189)+1)/4'th power [with dedicated addition chain]
square the result and compare to the input
conditionally negate the result accordingly
```
ADX code path is selected at run time. For testing purposes assembly
support can be suppressed with `--features no-asm` at cargo command
line. Otherwise `build.rs` adds `assembly.S` or
`win64/mod256-189-x86_64.asm` to build sequence depending on whether or
not one uses a non-Microsoft or Microsoft compiler. In former case the
`assembly.S` makes further choice among `coff` (used by MinGW), `elf` (used
by Linux, *BSD, Solaris) or `mach-o` (used by MacOS) executable formats.

The assembly modules are generated from single Perl script,
`src/mod256-189-x86_64.pl`, by executing `src/refresh.sh` on a system
with Perl. It's assumed that universal build-time dependency on Perl is
undesired. Most notably Perl is not customarily installed on Windows.
Otherwise this step could be moved to `build.rs`, in which case one
would remove all subdirectories in `src` and have `build.rs` recreate
the assembly modules during the build phase.

Field multiplication is performed approximately as following:
```
uint512 temp = a * b
temp = lo256(temp) + hi256(temp)*189
temp = lo256(temp) + hi8(temp)*189
return lo256(temp) + hi1(temp)*189
```
"Approximately" refers to the fact that in sparse representations
`lo256()` and `hi256()` are not exact, nor explicit in the
implementation.
