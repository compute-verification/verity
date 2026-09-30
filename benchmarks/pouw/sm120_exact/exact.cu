// Where v2's unpromoted chain is exact on sm_120 (RTX PRO 6000), for `no-aligned-exact-region/sm120-unpromoted`
// (internal/pouw/ttout-restatements.md §7), and §5's forming on the card for the unit's operands.
//
// A step of word (i, j) at atom t is exact when the atom's FP32 output equals its accumulator plus the exact sum of its 32
// products.  Every E4M3 value is an integer times 2^-9, so every product and every accumulator on this chain is an integer
// times 2^-18 below 2^49: the test is one int64 comparison.
#include <cstdint>

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, const float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"
               "{%10,%11,%12,%13};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Two FP32 values to two E4M3 codes (RNE, saturating), `lo` in the low byte; -0 and negative underflow give 0x00, as the
// census twin's `enc` does.
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  uint32_t v = r;
  if ((v & 0xFFu) == 0x80u) v &= 0xFF00u;
  if ((v & 0xFF00u) == 0x8000u) v &= 0x00FFu;
  return v;
}

// §5's forming: out[r][l] = E4M3(atom(C[r][l]; E[r] . F[l])), E rows x 32 and F k x 32 codes, one m16n8k32 per 16 x 8
// block of the output and one warp per block.  rows % 16 == 0, k % 8 == 0.
extern "C" __global__ void form_codes(const float* __restrict__ C, const uint8_t* __restrict__ E, const uint8_t* __restrict__ F,
                                      uint8_t* __restrict__ out, int rows, int k) {
  const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const long long blk = static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int kb = k >> 3;
  if (blk >= static_cast<long long>(rows >> 4) * kb) return;
  const int r0 = static_cast<int>(blk / kb) * 16, l0 = static_cast<int>(blk % kb) * 8;
  uint32_t a[4];
  a[0] = *reinterpret_cast<const uint32_t*>(E + static_cast<size_t>(r0 + gid) * 32 + tig * 4);
  a[1] = *reinterpret_cast<const uint32_t*>(E + static_cast<size_t>(r0 + gid + 8) * 32 + tig * 4);
  a[2] = *reinterpret_cast<const uint32_t*>(E + static_cast<size_t>(r0 + gid) * 32 + 16 + tig * 4);
  a[3] = *reinterpret_cast<const uint32_t*>(E + static_cast<size_t>(r0 + gid + 8) * 32 + 16 + tig * 4);
  const uint32_t b0 = *reinterpret_cast<const uint32_t*>(F + static_cast<size_t>(l0 + gid) * 32 + tig * 4);
  const uint32_t b1 = *reinterpret_cast<const uint32_t*>(F + static_cast<size_t>(l0 + gid) * 32 + 16 + tig * 4);
  const float2 c01 = *reinterpret_cast<const float2*>(C + static_cast<size_t>(r0 + gid) * k + l0 + tig * 2);
  const float2 c23 = *reinterpret_cast<const float2*>(C + static_cast<size_t>(r0 + gid + 8) * k + l0 + tig * 2);
  const float c[4] = {c01.x, c01.y, c23.x, c23.y};
  float d[4];
  mma_e4m3(d, a, b0, b1, c);
  *reinterpret_cast<uint16_t*>(out + static_cast<size_t>(r0 + gid) * k + l0 + tig * 2) = static_cast<uint16_t>(e4m3x2(d[0], d[1]));
  *reinterpret_cast<uint16_t*>(out + static_cast<size_t>(r0 + gid + 8) * k + l0 + tig * 2) = static_cast<uint16_t>(e4m3x2(d[2], d[3]));
}

__device__ __forceinline__ int dec(uint32_t c) {
  const int e = (c >> 3) & 15, m = c & 7;
  const int mag = e ? (8 | m) << (e - 1) : m;
  return (c & 0x80u) ? -mag : mag;
}

__device__ __forceinline__ long long fix18(float x) { return __float2ll_rz(x * 262144.0f); }

constexpr int FM = 64, FN = 64, FK = 128;

// The unpromoted chain (one FP32 accumulator over all of k, as sm120_chain's chain_unpromoted) on a 64 x 64 tile per CTA,
// 4 warps of 32 x 32, and each step's exactness: flags[w][i][j] bit b is atom 32w + b of word (i, j); out[i][j] is the
// final word.  m, n % 64 == 0, k % 1024 == 0.
extern "C" __global__ void __launch_bounds__(128) chain_flags(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                              uint32_t* __restrict__ flags, uint32_t* __restrict__ out, int m,
                                                              int n, int k) {
  __shared__ __align__(16) uint8_t sA[FM * FK];
  __shared__ __align__(16) uint8_t sB[FN * FK];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.y * FM, n0 = blockIdx.x * FN;
  float acc[2][4][4];
  uint32_t fl[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) { acc[i][j][e] = 0.f; fl[i][j][e] = 0u; }

  for (int kt = 0; kt < k / FK; ++kt) {
    __syncthreads();
    for (int c = tid; c < FM * FK / 16; c += 128) {
      const int row = c >> 3, ch = c & 7;
      reinterpret_cast<uint4*>(sA)[c] = *reinterpret_cast<const uint4*>(A + static_cast<size_t>(m0 + row) * k + kt * FK + ch * 16);
      reinterpret_cast<uint4*>(sB)[c] = *reinterpret_cast<const uint4*>(B + static_cast<size_t>(n0 + row) * k + kt * FK + ch * 16);
    }
    __syncthreads();
    for (int st = 0; st < FK / 32; ++st) {
      const int t = kt * (FK / 32) + st, o = st * 32;
      uint32_t a[2][4], b[4][2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int r = wm * 32 + mi * 16 + gid;
        a[mi][0] = *reinterpret_cast<const uint32_t*>(sA + r * FK + o + tig * 4);
        a[mi][1] = *reinterpret_cast<const uint32_t*>(sA + (r + 8) * FK + o + tig * 4);
        a[mi][2] = *reinterpret_cast<const uint32_t*>(sA + r * FK + o + 16 + tig * 4);
        a[mi][3] = *reinterpret_cast<const uint32_t*>(sA + (r + 8) * FK + o + 16 + tig * 4);
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int r = wn * 32 + ni * 8 + gid;
        b[ni][0] = *reinterpret_cast<const uint32_t*>(sB + r * FK + o + tig * 4);
        b[ni][1] = *reinterpret_cast<const uint32_t*>(sB + r * FK + o + 16 + tig * 4);
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int ra = wm * 32 + mi * 16 + gid;
        int va0[32], va1[32];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint32_t w0 = *reinterpret_cast<const uint32_t*>(sA + ra * FK + o + q * 4);
          const uint32_t w1 = *reinterpret_cast<const uint32_t*>(sA + (ra + 8) * FK + o + q * 4);
#pragma unroll
          for (int by = 0; by < 4; ++by) { va0[q * 4 + by] = dec((w0 >> (8 * by)) & 0xFFu); va1[q * 4 + by] = dec((w1 >> (8 * by)) & 0xFFu); }
        }
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          float prev[4] = {acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]};
          mma_e4m3(acc[mi][ni], a[mi], b[ni][0], b[ni][1], acc[mi][ni]);
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int cb = wn * 32 + ni * 8 + tig * 2 + e;
            long long s0 = 0, s1 = 0;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const uint32_t w = *reinterpret_cast<const uint32_t*>(sB + cb * FK + o + q * 4);
#pragma unroll
              for (int by = 0; by < 4; ++by) {
                const long long vb = dec((w >> (8 * by)) & 0xFFu);
                s0 += va0[q * 4 + by] * vb;
                s1 += va1[q * 4 + by] * vb;
              }
            }
            const uint32_t bit = 1u << (t & 31);
            if (fix18(acc[mi][ni][e]) == fix18(prev[e]) + s0) fl[mi][ni][e] |= bit;
            if (fix18(acc[mi][ni][2 + e]) == fix18(prev[2 + e]) + s1) fl[mi][ni][2 + e] |= bit;
          }
        }
      }
      if ((t & 31) == 31) {
        const size_t plane = static_cast<size_t>(t >> 5) * m * n;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int row = m0 + wm * 32 + mi * 16 + gid + h * 8, col = n0 + wn * 32 + ni * 8 + tig * 2;
              *reinterpret_cast<uint2*>(flags + plane + static_cast<size_t>(row) * n + col) = make_uint2(fl[mi][ni][2 * h], fl[mi][ni][2 * h + 1]);
              fl[mi][ni][2 * h] = 0u;
              fl[mi][ni][2 * h + 1] = 0u;
            }
      }
    }
  }
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = m0 + wm * 32 + mi * 16 + gid + h * 8, col = n0 + wn * 32 + ni * 8 + tig * 2;
        *reinterpret_cast<uint2*>(out + static_cast<size_t>(row) * n + col) =
            make_uint2(__float_as_uint(acc[mi][ni][2 * h]), __float_as_uint(acc[mi][ni][2 * h + 1]));
      }
}

// For the 32-atom windows starting at t0 .. t0 + nt - 1, word (i, j) exact on every atom of [t, t + 32): with tr = 0, row
// bitsets, M[t - t0][i][j / 32] bit j % 32; with tr = 1, column bitsets, M[t - t0][j][i / 32] bit i % 32.  One warp per 32
// consecutive words of a row (tr = 0) or a column (tr = 1); T / 32 <= 16 flag planes.  m, n % 32 == 0, m * n % blockDim == 0.
extern "C" __global__ void window_bits(const uint32_t* __restrict__ flags, uint32_t* __restrict__ M, int m, int n, int T,
                                       int t0, int nt, int tr) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(m) * n) return;
  const size_t i = tr ? idx % m : idx / n, j = tr ? idx / m : idx % n, word = i * n + j;
  const int planes = T >> 5;
  uint32_t f[17];
#pragma unroll
  for (int p = 0; p < 16; ++p) f[p] = p < planes ? flags[static_cast<size_t>(p) * m * n + word] : 0u;
  f[16] = 0u;
  const size_t dst = tr ? j * (m >> 5) + (i >> 5) : i * (n >> 5) + (j >> 5), plane = static_cast<size_t>(m) * (n >> 5);
  for (int t = t0; t < t0 + nt; ++t) {
    bool ok = false;
    if (t + 32 <= T) {
      const int p = t >> 5, s = t & 31;
      uint32_t lo = f[0], hi = f[1];
#pragma unroll
      for (int q = 0; q < 16; ++q)
        if (q == p) { lo = f[q]; hi = f[q + 1]; }
      const uint64_t x = (static_cast<uint64_t>(hi) << 32 | lo) >> s;
      ok = static_cast<uint32_t>(x) == 0xFFFFFFFFu;
    }
    const uint32_t mask = __ballot_sync(0xFFFFFFFFu, ok);
    if ((threadIdx.x & 31) == 0) M[static_cast<size_t>(t - t0) * plane + dst] = mask;
  }
}
