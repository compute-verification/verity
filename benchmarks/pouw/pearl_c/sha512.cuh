// Pearl-C's names for the GPU's one SHA-512 (verity/ml/kernels/sha512.cuh), every word big-endian: the
// round constants, the IV, one compression of 16 message words (w is the schedule's ring, overwritten; sx::compress<0, 0>,
// straight-line with funnel shifts) and the byte-order helpers that load a message word from two little-endian u32.
// hash_h3.cuh (Pearl-C's -h3), pearl_c4.cu (Pearl-C4's seeds and noise lines), pearl_c_sm120's pd and compute-accounting's
// masked_gemm/sha_noise.cuh and filler/pfill.cuh include it; RS_K and rs_rotr serve the round forms of their own.
#pragma once
#include <cstdint>

#include "../../../verity/ml/kernels/sha512.cuh"

#ifndef DEV
#define DEV __device__ __forceinline__
#endif

#define RS_K sx::K_DEV
DEV uint64_t rs_rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
DEV uint32_t rs_be32(uint32_t x) { return sx::bswap32(x); }
// the big-endian 64-bit message word of 8 bytes loaded as two little-endian u32
DEV uint64_t rs_be64(uint32_t lo, uint32_t hi) { return sx::be64(lo, hi); }
DEV void rs_iv(uint64_t* st) { sx::iv(st); }
DEV void rs_compress(uint64_t* st, uint64_t* w) { sx::compress<0, 0>(st, w); }
