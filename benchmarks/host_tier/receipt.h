// The host CPUs' side of loop.cu, compiled by g++ (receipt.cpp): the mask expanded from z, the VNNI sketch of
// (block ^ mask), the SHA-256 leaf over 16 blocks' sketches, and the adversary's four-stream DRAM read.
#pragma once
#include <cstddef>
#include <cstdint>

extern const uint8_t SEED[32];
extern int sha_ni;     // SHA-256 on the SHA extensions (1) or portable (0); a hypervisor may trap the extensions
extern int pf_bytes;   // the sketch's software prefetch distance a stream, bytes (0: none)
void sha256(const uint8_t* m, size_t n, uint8_t out[32]);
void cpu_keys(int8_t* K, int ks);  // K[l * 256 + j], l < ks
// the 4 KiB mask a receipt XORs into every block, expanded from all of z: M = the 512 words
// splitmix64(h ^ (0x6d00000000 + i)), little-endian, with h = seed64 folded with z[0..15] by h = splitmix64(h ^ z[t])
void cpu_mask(const uint32_t* z, uint8_t* M);
// 16 blocks of 4 KiB at x (64-byte aligned) under mask M (64-byte aligned), ks in {1, 2, 4, 16}: sketches sk[16 * ks]
// (block-major) and the group's leaf
void cpu_group(const uint8_t* x, const uint8_t* M, uint64_t zseq, uint64_t group, const int8_t* K, int ks,
               uint8_t leaf[32], int32_t* sk);
// The same for the 16 groups group0 .. group0 + 15 at x (1 MiB), read as four 256 KiB streams: leaves[32 * 16] and
// sk[256 * ks], each group's as cpu_group gives it
void cpu_chunk(const uint8_t* x, const uint8_t* M, uint64_t zseq, uint64_t group0, const int8_t* K, int ks,
               uint8_t* leaves, int32_t* sk);
uint64_t adv_chunk(const void* p, size_t lines);  // sum of `lines` 64-byte lines read as four streams
