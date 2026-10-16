// pfill.cuh's host tile built with g++ (tests/test_pfill_ref.py), so its receipts can be held to pfill_ref.py on a machine
// with no GPU; dcap.cu's smix and dseed as dcap.cu defines them.
//   g++ -O2 -std=c++17 -shared -fPIC -o pfill_host.so pfill_host.cpp
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#define __host__
#define __device__
#define __forceinline__ inline
#define __constant__ static const

inline uint64_t smix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
inline uint64_t dseed(uint64_t c0, uint64_t c1, uint64_t tile) { return smix(smix(c0 ^ smix(tile)) ^ c1); }

#include "pfill.cuh"

extern "C" uint64_t pfill_ref(uint64_t c0, uint64_t c1, uint64_t tile, int nc, int steps, int kind) {
  return dfill_ref_kind(c0, c1, tile, nc, steps, kind);
}
extern "C" uint32_t pfill_e2m1x8(uint32_t byte) { return pf_e2m1x8(byte); }
extern "C" uint32_t pfill_s8x4(uint32_t nib, int kind) { return kind == PF_BITS ? pf_s8x4<PF_BITS>(nib) : pf_s8x4<PF_S8>(nib); }
