// Honest's per-word epilogue on sm_90a for the SHA lane's combiners. An IMMA m16n8k32 s8 a step, its output a fresh
// value (so a block's first word needs no copy), and over a block of M steps one of: nothing (base); every word stored
// to shared memory (lead); the XOR of the block's words (xor); or the XOR of its words each rotated by a distinct public
// amount, the first by 0 (xorrot), the block's four XORs a thread stored. Built to SASS only and counted by
// xor_epilogue_sass.py; never launched.
#include <cstdint>

enum Mode { BASE, LEAD, XOR, XORROT };

__device__ __forceinline__ void imma(uint32_t (&d)[4], const uint32_t (&c)[4], const uint4 a, const uint2 b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

template <int M, Mode MODE>
__global__ void __launch_bounds__(32) probe(const uint4* __restrict__ A, const uint2* __restrict__ B,
                                            uint32_t* __restrict__ out, int blocks) {
  __shared__ uint32_t sm[16 * 4 * 32];
  uint32_t c[4] = {0, 0, 0, 0};
  const int lane = threadIdx.x;
  for (int blk = 0; blk < blocks; ++blk) {
    uint32_t x[4] = {0, 0, 0, 0};
#pragma unroll
    for (int t = 0; t < M; ++t) {
      uint32_t d[4];
      imma(d, c, A[(blk * M + t) * 32 + lane], B[(blk * M + t) * 32 + lane]);
#pragma unroll
      for (int k = 0; k < 4; ++k) c[k] = d[k];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        if constexpr (MODE == LEAD) {
          sm[(t * 4 + k) * 32 + lane] = c[k];
        } else if constexpr (MODE == XOR) {
          x[k] = t == 0 ? c[k] : x[k] ^ c[k];
        } else if constexpr (MODE == XORROT) {
          const uint32_t w = t == 0 ? c[k] : __funnelshift_l(c[k], c[k], (7 * t) & 31);
          x[k] = t == 0 ? w : x[k] ^ w;
        }
      }
    }
    if constexpr (MODE == XOR || MODE == XORROT) {
#pragma unroll
      for (int k = 0; k < 4; ++k) sm[k * 32 + lane] = x[k];
    }
    __syncwarp();
  }
  out[lane] = c[0] ^ c[1] ^ c[2] ^ c[3] ^ sm[lane] ^ sm[lane + 32];
}

#define INST(M, MODE) template __global__ void probe<M, MODE>(const uint4*, const uint2*, uint32_t*, int);
INST(16, BASE)
INST(16, LEAD)
INST(16, XOR)
INST(16, XORROT)
INST(11, BASE)
INST(11, XOR)
INST(11, XORROT)
