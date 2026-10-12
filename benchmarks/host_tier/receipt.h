// The host CPUs' side of loop.cu, compiled by g++ (receipt.cpp): the VNNI sketch of (block ^ z), the SHA-256 leaf over
// 16 blocks' sketches, and the adversary's four-stream DRAM read.
#pragma once
#include <cstddef>
#include <cstdint>

extern const uint8_t SEED[32];
void sha256(const uint8_t* m, size_t n, uint8_t out[32]);
void cpu_keys(int8_t* K, int ks);  // K[l * 256 + j], l < ks
// 16 blocks of 4 KiB at x (64-byte aligned) under z: sketches sk[16 * ks] (block-major) and the group's leaf
void cpu_group(const uint8_t* x, const uint32_t* z, uint64_t zseq, uint64_t group, const int8_t* K, int ks,
               uint8_t leaf[32], int32_t* sk);
uint64_t adv_chunk(const void* p, size_t lines);  // sum of `lines` 64-byte lines read as four streams
