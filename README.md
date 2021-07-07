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
#if defined(__ADX__) && !defined(__SLOTH_PORTABLE__)
 x86_64 assembly-assisted code path, executed on processors with ADX
 ISA extenstion.
 Field element is represented as 4 64-bit limbs.
#elif __SIZEOF_INT128__-0==16
 Any 64-bit platform with compiler that supports __int128 type, such
 as comtemporary gcc or clang, not not Microsoft C.
 Field element is represented as 5 limbs, 1st is 52- and the rest are
 51-bit values.
#else
 Fallback for everything else.
 Field element is represtented as 12 limbs, mixture of 22- and 21-bit
 values [with last one being 19-bits].
#endif
```
All three code paths use the same pattern for the sqrt operation:
```
raise input to ((2**256-189)+1)/4'th power [with dedicated addition chain]
square the result and compare to the input
conditionally negate the result accordingly
```
Code path is selected at compile time in `<top>/build.rs`, which detects
ADX extension availability, and adds `assembly.S` or
`win64/mod256-189-x86_64.asm` to build sequence depending on whether or
not one uses non-Microsoft or Microsoft compiler. In former case the
`assembly.S` makes the choice among `coff` (used by MinGW), `elf` (used
by Linux, *BSD, Solaris) or `mach-o` (used by MacOS) executable formats.
The ADX detection can be suppressed with `--features portable` at cargo
command line.
