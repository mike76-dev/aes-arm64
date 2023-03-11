# aes-arm64
The latest models of Raspberry Pi have a nice ARMv8 CPU. Sadly enough, this CPU is lacking the set of cryptographic instructions, which is present in the other ARMv8 processors. Therefore, Raspberry Pi does not support hardware AES acceleration.
Presented here are three software AES implementations.

1. aes_scalar.cpp

This is the scalar implementation, which can be compiled on any platform, not only ARM.

2. aes_neon.cpp

This implementation uses the NEON SIMD instructions of ARMv8.

3. aes_bitslice.cpp

This implementation uses the bitslicing algorithm, which allows a parallel processing of 8 encryptions/decryptions.

## Benchmarks

Compiled with **gcc 11.3.0** for Ubuntu using **-O3** optimization level.
Executed on **Raspberry Pi 4B**, 8GB RAM.
Each benchmark included 100,000,000 encryption rounds done on 8 data blocks.

- aes_scalar:   92 seconds
- aes_neon:     52 seconds
- aes_bitslice: 11 seconds

For comparison, the aes_scalar benchmark executed on an Intel Core i5 CPU took 24 seconds.
