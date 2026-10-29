// M_key·y a loop trip, for counting its SASS: the committer's own mask_roll in the H200's split form (native_rows.cu, the byte
// table in shared memory), the loop body a salt's 24 loads, the product and its 16 stores. MASK_UNROLL=1: the same product
// with its loop over r unrolled, so each byte's shift is static. Disassembled by mask_sass.sh; mask_check.cu runs mask_unrolled.
#include "native_rows.cu"

#if MASK_UNROLL
DEV void mask_unrolled(const uint4* T, const u64* Y, u32* m) {
#pragma unroll
  for (int w = 0; w < 16; ++w) m[w] = 0;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    u32 A[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) A[i] = 0;
    const int sh = 56 - 8 * (r & 7);
#pragma unroll
    for (int q = 0; q < 12; ++q) {
      const u32 v = (u32)((r < 8 ? Y[2 * q] : Y[2 * q + 1]) >> sh) & 0xFFu;
      const uint4* p = T + 17 * v + q;
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const uint4 u = p[i];
        A[4 * i] ^= u.x; A[4 * i + 1] ^= u.y; A[4 * i + 2] ^= u.z; A[4 * i + 3] ^= u.w;
      }
    }
    const int d = r >> 2, s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= __funnelshift_r(A[w + d], A[w + d + 1], s);
  }
}
#endif

extern "C" __global__ void probe(const u64* __restrict__ ys, u32* __restrict__ out, const uint8_t* __restrict__ Tn, int n) {
  extern __shared__ uint4 T8[];
  load_t8(Tn, T8);
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 1
  for (int s = 0; s < n; ++s) {
    const size_t k = t * n + s;
    u64 Y[24];
#pragma unroll
    for (int i = 0; i < 24; ++i) Y[i] = ys[24 * k + i];
    u32 mk[16];
#if MASK_UNROLL
    mask_unrolled(T8, Y, mk);
#else
    mask_roll<true>(T8, Y, mk);
#endif
#pragma unroll
    for (int w = 0; w < 16; ++w) out[16 * k + w] = mk[w];
  }
}
