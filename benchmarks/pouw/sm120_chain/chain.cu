// Pearl-C's ticket chain on sm_120 (RTX PRO 6000), and the attacker's variants of it, for timing one against the other.
//
// C~[i][j] = the chain of A[i] . B[j] (E4M3 codes, both stored as rows along k) in mma.sync kind::f8f6f4 m16n8k32 atoms:
//   MODE 0 (honest)      every 128 products a fresh +0 accumulator chains four atoms, then one FP32 nearest-even add
//                        promotes it into the total (Pearl-C's C~, G = 4);
//   MODE 1 (unpromoted)  one accumulator chains every atom (the cheat that skips the promotions; its word equals C~ only
//                        where no promotion rounds);
//   MODE 2 (int8)        the same kernel on mma.sync m16n8k32 s8 (codes read as int8, s32 accumulate): a rate probe whose
//                        output is exact, so it is gated;
//   MODE 3 (f16acc)      mma.sync m16n8k32 e4m3 with an FP16 accumulator: a rate probe only (its words are not Pearl-C's).
// One CTA computes a 128 x 128 tile with 8 warps of 64 x 32; k in steps of 128 bytes through a 3-stage cp.async ring of
// swizzled 128 x 128-byte tiles (96 KiB); fragments by ldmatrix.  m, n and k are multiples of 128.
#include <cstdint>

constexpr int BM = 128, BN = 128, BK = 128, STAGES = 3, THREADS = 256;
constexpr int TILE = BM * BK;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_wait_1() { asm volatile("cp.async.wait_group 1;"); }
__device__ __forceinline__ void ldm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// Byte offset of (row, 16-byte chunk) in a 128 x 128-byte tile, chunks XOR-swizzled by row so ldmatrix is conflict-free.
__device__ __forceinline__ uint32_t swz(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, const float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"
               "{%10,%11,%12,%13};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void mma_s8(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16acc(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void load_tile(uint8_t* s, const uint8_t* g, int ld, int tid) {
#pragma unroll
  for (int i = 0; i < TILE / 16 / THREADS; ++i) {
    int c = tid + i * THREADS, row = c >> 3, ch = c & 7;
    cp_async16(smem_u32(s + swz(row, ch)), g + static_cast<size_t>(row) * ld + ch * 16);
  }
}

template <int MODE>
__device__ __forceinline__ void chain(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, uint32_t* __restrict__ out,
                                      int m, int n, int k) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, KT = k / BK;
  const uint8_t* gA = A + static_cast<size_t>(m0) * k;
  const uint8_t* gB = B + static_cast<size_t>(n0) * k;

  float acc[4][4][4], tot[4][4][4];
  int32_t iacc[4][4][4];
  uint32_t hacc[4][4][2];
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc[i][j][e] = 0.f; tot[i][j][e] = 0.f; iacc[i][j][e] = 0;
        if (e < 2) hacc[i][j][e] = 0u;
      }

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) {
      load_tile(smem + s * 2 * TILE, gA + s * BK, k, tid);
      load_tile(smem + s * 2 * TILE + TILE, gB + s * BK, k, tid);
    }
    cp_commit();
  }

  for (int kt = 0; kt < KT; ++kt) {
    cp_wait_1();
    __syncthreads();
    const int nk = kt + STAGES - 1;
    if (nk < KT) {
      uint8_t* s = smem + (nk % STAGES) * 2 * TILE;
      load_tile(s, gA + static_cast<size_t>(nk) * BK, k, tid);
      load_tile(s + TILE, gB + static_cast<size_t>(nk) * BK, k, tid);
    }
    cp_commit();
    const uint32_t sA = smem_u32(smem + (kt % STAGES) * 2 * TILE), sB = sA + TILE;

#pragma unroll
    for (int st = 0; st < BK / 32; ++st) {
      uint32_t a[4][4], b[4][2];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) ldm_x4(sA + swz(wm * 64 + mi * 16 + (lane & 15), st * 2 + (lane >> 4)), a[mi]);
#pragma unroll
      for (int nj = 0; nj < 2; ++nj) {
        uint32_t r[4];
        ldm_x4(sB + swz(wn * 32 + nj * 16 + ((lane >> 4) << 3) + (lane & 7), st * 2 + ((lane >> 3) & 1)), r);
        b[2 * nj][0] = r[0]; b[2 * nj][1] = r[1]; b[2 * nj + 1][0] = r[2]; b[2 * nj + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          if constexpr (MODE == 0) {
            if (st == 0) mma_e4m3(acc[mi][ni], a[mi], b[ni][0], b[ni][1], zero);
            else mma_e4m3(acc[mi][ni], a[mi], b[ni][0], b[ni][1], acc[mi][ni]);
          } else if constexpr (MODE == 1) {
            mma_e4m3(acc[mi][ni], a[mi], b[ni][0], b[ni][1], acc[mi][ni]);
          } else if constexpr (MODE == 2) {
            mma_s8(iacc[mi][ni], a[mi], b[ni][0], b[ni][1]);
          } else {
            mma_f16acc(hacc[mi][ni], a[mi], b[ni][0], b[ni][1]);
          }
        }
    }
    if constexpr (MODE == 0) {
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
          for (int e = 0; e < 4; ++e) tot[mi][ni][e] = __fadd_rn(tot[mi][ni][e], acc[mi][ni][e]);
    }
  }

  const int gid = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = m0 + wm * 64 + mi * 16 + gid + h * 8, col = n0 + wn * 32 + ni * 8 + tig * 2;
        uint32_t w0, w1;
        if constexpr (MODE == 0) { w0 = __float_as_uint(tot[mi][ni][2 * h]); w1 = __float_as_uint(tot[mi][ni][2 * h + 1]); }
        else if constexpr (MODE == 1) { w0 = __float_as_uint(acc[mi][ni][2 * h]); w1 = __float_as_uint(acc[mi][ni][2 * h + 1]); }
        else if constexpr (MODE == 2) { w0 = static_cast<uint32_t>(iacc[mi][ni][2 * h]); w1 = static_cast<uint32_t>(iacc[mi][ni][2 * h + 1]); }
        else { w0 = hacc[mi][ni][h] & 0xFFFFu; w1 = hacc[mi][ni][h] >> 16; }
        *reinterpret_cast<uint2*>(out + static_cast<size_t>(row) * n + col) = make_uint2(w0, w1);
      }
}

#define ENTRY(name, mode)                                                                                            \
  extern "C" __global__ void __launch_bounds__(THREADS, 1)                                                           \
      name(const uint8_t* A, const uint8_t* B, uint32_t* out, int m, int n, int k) { chain<mode>(A, B, out, m, n, k); }
ENTRY(chain_honest, 0)
ENTRY(chain_unpromoted, 1)
ENTRY(chain_int8, 2)
ENTRY(chain_f16acc, 3)

// A read-only L2 flush: a sum over a buffer larger than L2, written to one word so it is not elided.
extern "C" __global__ void l2_flush(const uint4* __restrict__ p, size_t n, uint32_t* sink) {
  uint32_t s = 0;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    uint4 v = p[i];
    s ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (s == 0x9E3779B9u) *sink = s;
}
