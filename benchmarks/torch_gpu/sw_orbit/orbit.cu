// One exact Strassen level of the int8 GEMM on sm_90a, built from CUTLASS 3.x sm90 pieces (orbit.py is its reference):
// O [M, N] s32 = A [M, K] s8 · B [N, K] s8ᵀ, both row-major, through seven products of half size on wrapped int8 forms.
//
//   forms        A's or B's seven wrapped forms in ORDER, side by side along K (wide [R/2, 7 K/2] s8), each byte the form
//                mod 256, and per slot the number of carries (bytes whose exact form left [−128, 127]); a carry plane
//                [7, R/2, K/2] s8 (c in {−1, 0, 1}) when asked, and for A its carries as a list a row
//   batched      CUTLASS's cooperative TMA GEMM over L = 7: the products P [7, M/2, N/2] s32 (the unfused orbit)
//   assemble     O's four quadrants from P (the unfused orbit's last pass)
//   fused        the products one after another through one CUTLASS TMA pipeline, as one GEMM of shape (M/2, N/2, 7 K/2)
//                over the wide forms; the consumers hold three 128 x 128 accumulators, switch them every K/2 and store
//                each quadrant of O straight from registers once its last product is in
//   corr         the carries' corrections added into O on the CUDA cores (B's from lists made at load, A's from
//                forms' lists), which makes the orbit exact
//   corr4        B's carries' corrections mod 4 only, bit-sliced 32 rows a word: what a prover needs for O's bit 9 alone
//                (timing only)
//   plain        CUTLASS's cooperative TMA GEMM on A and B, the baseline
//
// The products are on wrapped forms: O is A Bᵀ exactly when no form carried (|a|, |b| ≤ 63 everywhere, say); otherwise
// each carry's 256 c column is still to add (orbit.py's `products`), which corr does.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/pipeline/pipeline.hpp"

namespace orbit_impl {
using namespace cute;

constexpr int kSlots = 7;

// ---------------------------------------------------------------------------------------------------------------------
// Forms. Slot s is Strassen's product ORDER[s] = (2, 4, 1, 6, 7, 3, 5)[s]; a form is X(i0, j0) + sign · X(i1, j1), or one
// block (sign 0), with X(i, j) the block of the i-th half of the rows and the j-th half of K.
struct Form {
  int i0, j0, i1, j1, sign;
};
__host__ __device__ constexpr Form form_of(int which, int s) {
  constexpr Form a[kSlots] = {{1, 0, 1, 1, 1}, {1, 1, 0, 0, 0}, {0, 0, 1, 1, 1}, {1, 0, 0, 0, -1},
                              {0, 1, 1, 1, -1}, {0, 0, 0, 0, 0}, {0, 0, 0, 1, 1}};
  constexpr Form b[kSlots] = {{0, 0, 0, 0, 0}, {0, 1, 0, 0, -1}, {0, 0, 1, 1, 1}, {0, 0, 1, 0, 1},
                              {0, 1, 1, 1, 1}, {1, 0, 1, 1, -1}, {1, 1, 0, 0, 0}};
  return which ? b[s] : a[s];
}

// A's carries as lists: per half-grid row r, up to kAcap entries at alist[r · kAcap], each (s << 16) | (neg << 15) | k,
// their number in acount[r] (which may pass kAcap: the call is then refused and its rows uncorrected).
constexpr int kAcap = 1024;
constexpr int kFormsThreads = 128;

__device__ __forceinline__ unsigned acarry_entry(int s, int k, bool neg) {
  return (unsigned(s) << 16) | (neg ? 0x8000u : 0u) | unsigned(k);
}

// One thread a 16-byte column chunk of one row of the half grid: the four blocks' chunks in, the seven forms' out. Block
// (blockIdx.x, blockIdx.y) covers chunks [128 blockIdx.x, +128) of row blockIdx.y.
// x [R, K] s8 row-major; wide [R/2, 7 Kh] s8; counts[7] u64 (added to); carry [7, R/2, Kh] s8 or null; alist/acount
// (A's lists) or null.
template <int Which>
__global__ void __launch_bounds__(kFormsThreads) forms_kernel(const int8_t* __restrict__ x, int8_t* __restrict__ wide,
                                                              unsigned long long* counts, int8_t* __restrict__ carry,
                                                              unsigned* __restrict__ alist, int* __restrict__ acount,
                                                              int rh, int kh) {
  __shared__ unsigned bcnt[kSlots];
  __shared__ unsigned blist[512];
  __shared__ unsigned bn, bbase;
  if (threadIdx.x < kSlots) bcnt[threadIdx.x] = 0;
  if (threadIdx.x == 0) bn = 0;
  __syncthreads();
  int chunks = kh / 16;
  int r = blockIdx.y, ci = blockIdx.x * kFormsThreads + threadIdx.x;
  bool live = ci < chunks;
  int c = live ? ci * 16 : 0;
  int64_t k = 2LL * kh;
  uint4 blk[2][2] = {};
  if (live) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        blk[i][j] = *reinterpret_cast<const uint4*>(x + (int64_t)(i * rh + r) * k + j * kh + c);
  }
  unsigned cnt[kSlots];
#pragma unroll
  for (int s = 0; s < kSlots; ++s) {
    const Form f = form_of(Which, s);
    uint4 a = blk[f.i0][f.j0], b = blk[f.i1][f.j1], w = a, ov = make_uint4(0, 0, 0, 0);
    if (f.sign) {
      const unsigned* pa = &a.x;
      const unsigned* pb = &b.x;
      unsigned* pw = &w.x;
      unsigned* po = &ov.x;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        unsigned wr = f.sign > 0 ? __vadd4(pa[q], pb[q]) : __vsub4(pa[q], pb[q]);
        unsigned sat = f.sign > 0 ? __vaddss4(pa[q], pb[q]) : __vsubss4(pa[q], pb[q]);
        pw[q] = wr;
        po[q] = __vcmpne4(wr, sat);
      }
    }
    unsigned n = __popc(ov.x) + __popc(ov.y) + __popc(ov.z) + __popc(ov.w);
    cnt[s] = live ? n / 8 : 0;
    if (live) {
      *reinterpret_cast<uint4*>(wide + (int64_t)r * kSlots * kh + (int64_t)s * kh + c) = w;
      if (carry) {
        // an overflowed byte's exact value has a's sign: c = −1 where a < 0, else +1
        const unsigned* pa = &a.x;
        const unsigned* po = &ov.x;
        uint4 cv;
        unsigned* pc = &cv.x;
#pragma unroll
        for (int q = 0; q < 4; ++q) pc[q] = po[q] & (__vcmplts4(pa[q], 0u) | 0x01010101u);
        *reinterpret_cast<uint4*>(carry + ((int64_t)s * rh + r) * kh + c) = cv;
      }
      if (alist && n) {
        const unsigned* pa = &a.x;
        const unsigned* po = &ov.x;
        for (int q = 0; q < 4; ++q)
          for (int b = 0; b < 4; ++b)
            if ((po[q] >> (8 * b)) & 0xffu) {
              unsigned at = atomicAdd(&bn, 1u);
              if (at < 512) blist[at] = acarry_entry(s, c + 4 * q + b, (pa[q] >> (8 * b + 7)) & 1u);
            }
      }
    }
  }
#pragma unroll
  for (int s = 0; s < kSlots; ++s) {
    unsigned v = cnt[s];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0 && v) atomicAdd(&bcnt[s], v);
  }
  __syncthreads();
  if (threadIdx.x < kSlots && bcnt[threadIdx.x]) atomicAdd(counts + threadIdx.x, (unsigned long long)bcnt[threadIdx.x]);
  if (alist) {
    unsigned total = bn;
    if (threadIdx.x == 0 && total) bbase = unsigned(atomicAdd(acount + r, int(total > 512 ? kAcap + 1 : total)));
    __syncthreads();
    if (total && total <= 512)
      for (unsigned i = threadIdx.x; i < total; i += kFormsThreads)
        if (bbase + i < kAcap) alist[(int64_t)r * kAcap + bbase + i] = blist[i];
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The unfused orbit's assembly: O's quadrants from P [7, Mh, Nh] s32 (slots in ORDER: M2 M4 M1 M6 M7 M3 M5), int32 mod 2^32.
__device__ __forceinline__ int add32(int a, int b) { return int(unsigned(a) + unsigned(b)); }
__device__ __forceinline__ int sub32(int a, int b) { return int(unsigned(a) - unsigned(b)); }

__global__ void assemble_kernel(const int* __restrict__ p, int* __restrict__ o, int mh, int nh) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int q4 = nh / 4;
  if (idx >= (long long)mh * q4) return;
  int r = int(idx / q4), c = int(idx % q4) * 4;
  int64_t plane = (int64_t)mh * nh, at = (int64_t)r * nh + c;
  int4 m[kSlots];
#pragma unroll
  for (int s = 0; s < kSlots; ++s) m[s] = *reinterpret_cast<const int4*>(p + s * plane + at);
  const int *m2 = &m[0].x, *m4 = &m[1].x, *m1 = &m[2].x, *m6 = &m[3].x, *m7 = &m[4].x, *m3 = &m[5].x, *m5 = &m[6].x;
  int4 q00, q01, q10, q11;
  int *o00 = &q00.x, *o01 = &q01.x, *o10 = &q10.x, *o11 = &q11.x;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    o00[e] = add32(sub32(add32(m1[e], m4[e]), m5[e]), m7[e]);
    o01[e] = add32(m3[e], m5[e]);
    o10[e] = add32(m2[e], m4[e]);
    o11[e] = add32(add32(sub32(m1[e], m2[e]), m3[e]), m6[e]);
  }
  int64_t n = 2LL * nh;
  *reinterpret_cast<int4*>(o + (int64_t)r * n + c) = q00;
  *reinterpret_cast<int4*>(o + (int64_t)r * n + nh + c) = q01;
  *reinterpret_cast<int4*>(o + (int64_t)(mh + r) * n + c) = q10;
  *reinterpret_cast<int4*>(o + (int64_t)(mh + r) * n + nh + c) = q11;
}

// ---------------------------------------------------------------------------------------------------------------------
// The carries' corrections, exact: O_q += 256 · Σ_s σ(q, s) · (wA_s · cB_sᵀ + cA_s · F_B,sᵀ) over O's quadrants q, with F_B
// B's exact forms (wB + 256 cB).
//   B's side: per slot s and half-grid column n, B's carries as k's, bidx[bptr[s][n] .. bmid[s][n]) the +1 ones and
//   [bmid[s][n] .. bptr[s][n + 1]) the −1 ones (built from B at load). A's forms of slots 1..5 (those whose B form is a
//   sum) are staged in shared memory transposed, 4 rows' bytes a word biased by 128, so one 32-bit add sums two rows in
//   16-bit lanes; the bias comes off once a column.
//   A's side: per row the list forms_a wrote, each entry adding ±F_B,s[n, k] (ft [7][Kh][Nh] s16, from B at load) at
//   every n.
// A CTA owns 4G rows of the half grid and walks all Nh columns, G lanes a column (4 rows each), so each element of O is
// read and written once, by one thread. A row whose list overflowed sets *flag (its corrections are then incomplete).
constexpr int kCorrThreads = 512;
constexpr int kCorrSlots = 5;  // slots 1..5
constexpr int kCorrPad = 8;

// σ(q, s) + 1 in 2 bits a slot, for q = 00, 01, 10, 11 (O00 = M1 + M4 − M5 + M7, O01 = M3 + M5, O10 = M2 + M4,
// O11 = M1 − M2 + M3 + M6; slots in ORDER: M2 M4 M1 M6 M7 M3 M5)
__host__ __device__ constexpr int sigma(int q, int s) {
  return q == 0   ? (s == 1 || s == 2 || s == 4 ? 1 : s == 6 ? -1 : 0)
         : q == 1 ? (s == 5 || s == 6 ? 1 : 0)
         : q == 2 ? (s == 0 || s == 1 ? 1 : 0)
                  : (s == 2 || s == 3 || s == 5 ? 1 : s == 0 ? -1 : 0);
}
__host__ __device__ constexpr unsigned sigma_packed(int q) {
  unsigned v = 0;
  for (int s = 0; s < kSlots; ++s) v |= unsigned(sigma(q, s) + 1) << (2 * s);
  return v;
}
__device__ __forceinline__ int sigma_rt(unsigned packed, int s) { return int((packed >> (2 * s)) & 3u) - 1; }

// Store false (timing only): the corrections are computed but O isn't touched; each thread's xor of them goes to
// o[its global index], so nothing is elided.
template <int G, bool Store>
__global__ void __launch_bounds__(kCorrThreads, 1)
    corr_kernel(const int8_t* __restrict__ wide_a, const int* __restrict__ bptr, const int* __restrict__ bmid,
                const uint16_t* __restrict__ bidx, const unsigned* __restrict__ alist, const int* __restrict__ acount,
                const int16_t* __restrict__ ft, int32_t* __restrict__ o, int* flag, int mh, int nh, int kh) {
  constexpr int R = 4 * G, CW = 32 / G;
  extern __shared__ uint4 sm4[];
  unsigned* sm = reinterpret_cast<unsigned*>(sm4);
  const int ld = kh + kCorrPad;
  const int m0 = blockIdx.x * R;
  const int64_t wld = (int64_t)kSlots * kh;
  for (int t = 0; t < kCorrSlots; ++t) {
    const int s = t + 1, k4s = kh / 4;
    for (int idx = threadIdx.x; idx < k4s * G; idx += kCorrThreads) {
      int k4 = idx % k4s, q = idx / k4s;
      unsigned r[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        r[i] = *reinterpret_cast<const unsigned*>(wide_a + (int64_t)(m0 + 4 * q + i) * wld + (int64_t)s * kh + 4 * k4);
      unsigned t0 = __byte_perm(r[0], r[1], 0x5140), t1 = __byte_perm(r[2], r[3], 0x5140);
      unsigned t2 = __byte_perm(r[0], r[1], 0x7362), t3 = __byte_perm(r[2], r[3], 0x7362);
      uint4 w = make_uint4(__byte_perm(t0, t1, 0x5410) ^ 0x80808080u, __byte_perm(t0, t1, 0x7632) ^ 0x80808080u,
                           __byte_perm(t2, t3, 0x5410) ^ 0x80808080u, __byte_perm(t2, t3, 0x7632) ^ 0x80808080u);
      *reinterpret_cast<uint4*>(sm + (t * G + q) * ld + 4 * k4) = w;
    }
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / G, q = lane % G;
  constexpr unsigned kSig0 = sigma_packed(0), kSig1 = sigma_packed(1), kSig2 = sigma_packed(2), kSig3 = sigma_packed(3);
  int na[4];
  bool over = false;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int c = acount[m0 + 4 * q + i];
    over |= c > kAcap;
    na[i] = min(c, kAcap);
  }
  if (over) atomicOr(flag, 1);
  unsigned chk = 0;
  for (int n = warp * CW + g; n < nh; n += (kCorrThreads / 32) * CW) {
    int acc[4][4] = {};
#pragma unroll
    for (int t = 0; t < kCorrSlots; ++t) {
      const int s = t + 1;
      const unsigned* row = sm + (t * G + q) * ld;
      const int beg = bptr[s * (nh + 1) + n], mid = bmid[s * nh + n], end = bptr[s * (nh + 1) + n + 1];
      int p[4] = {};
      for (int side = 0; side < 2; ++side) {
        const int e0 = side ? mid : beg, e1 = side ? end : mid;
        for (int c0 = e0; c0 < e1; c0 += 128) {
          const int c1 = min(c0 + 128, e1);
          unsigned lo = 0, hi = 0;
          int e = c0;
          for (; e + 1 < c1; e += 2) {
            unsigned x0 = row[bidx[e]], x1 = row[bidx[e + 1]];
            lo += (x0 & 0x00ff00ffu) + (x1 & 0x00ff00ffu);
            hi += ((x0 >> 8) & 0x00ff00ffu) + ((x1 >> 8) & 0x00ff00ffu);
          }
          if (e < c1) {
            unsigned x0 = row[bidx[e]];
            lo += x0 & 0x00ff00ffu;
            hi += (x0 >> 8) & 0x00ff00ffu;
          }
          // at most 128 biased bytes a lane: no lane passes 2^15, so the lanes separate exactly
          const int l0 = int(lo & 0xffffu), l2 = int(lo >> 16), l1 = int(hi & 0xffffu), l3 = int(hi >> 16);
          if (side) {
            p[0] -= l0, p[1] -= l1, p[2] -= l2, p[3] -= l3;
          } else {
            p[0] += l0, p[1] += l1, p[2] += l2, p[3] += l3;
          }
        }
      }
      const int bias = 128 * ((mid - beg) - (end - mid));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int v = p[i] - bias;
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          if (sigma(qq, s) > 0) acc[qq][i] += v;
          if (sigma(qq, s) < 0) acc[qq][i] -= v;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned* L = alist + (int64_t)(m0 + 4 * q + i) * kAcap;
      for (int e = 0; e < na[i]; ++e) {
        const unsigned v = L[e];
        const int s = int(v >> 16), k = int(v & 0x7fffu);
        int f = ft[((int64_t)s * kh + k) * nh + n];
        if (v & 0x8000u) f = -f;
        acc[0][i] += sigma_rt(kSig0, s) * f;
        acc[1][i] += sigma_rt(kSig1, s) * f;
        acc[2][i] += sigma_rt(kSig2, s) * f;
        acc[3][i] += sigma_rt(kSig3, s) * f;
      }
    }
#pragma unroll
    for (int qq = 0; qq < 4; ++qq)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (Store) {
          if (acc[qq][i]) {
            int32_t* at = o + (int64_t)((qq >> 1) * mh + m0 + 4 * q + i) * (2LL * nh) + (qq & 1) * nh + n;
            *at = int(unsigned(*at) + (unsigned(acc[qq][i]) << 8));
          }
        } else {
          chk = (chk ^ unsigned(acc[qq][i])) * 0x9e3779b1u;
        }
  }
  if (!Store) o[(int64_t)blockIdx.x * kCorrThreads + threadIdx.x] = int(chk);
}

// corr_kernel with a thread a column and all 16 rows of its CTA: A's staged forms are 16 rows' bytes a k (one 128-bit
// shared load a carry, four words of two 16-bit lanes each), so a carry costs about half the instructions it does with
// 4 rows a thread. Needs 5 Kh 16 bytes of shared memory: Kh ≤ 2,048 on the H200.
constexpr int kCorr16Threads = 512;

template <bool Store>
__global__ void __launch_bounds__(kCorr16Threads, 1)
    corr16_kernel(const int8_t* __restrict__ wide_a, const int* __restrict__ bptr, const int* __restrict__ bmid,
                  const uint16_t* __restrict__ bidx, const unsigned* __restrict__ alist, const int* __restrict__ acount,
                  const int16_t* __restrict__ ft, int32_t* __restrict__ o, int* flag, int mh, int nh, int kh) {
  extern __shared__ uint4 sm4[];  // [5][kh]: word w of k holds rows 4w..4w+3 at k, biased by 128
  unsigned* sm = reinterpret_cast<unsigned*>(sm4);
  const int m0 = blockIdx.x * 16, k4s = kh / 4;
  const int64_t wld = (int64_t)kSlots * kh;
  for (int idx = threadIdx.x; idx < kCorrSlots * k4s * 4; idx += kCorr16Threads) {
    const int w = idx & 3, k4 = (idx >> 2) % k4s, t = (idx >> 2) / k4s;
    unsigned r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      r[i] = *reinterpret_cast<const unsigned*>(wide_a + (int64_t)(m0 + 4 * w + i) * wld + (int64_t)(t + 1) * kh + 4 * k4);
    unsigned t0 = __byte_perm(r[0], r[1], 0x5140), t1 = __byte_perm(r[2], r[3], 0x5140);
    unsigned t2 = __byte_perm(r[0], r[1], 0x7362), t3 = __byte_perm(r[2], r[3], 0x7362);
    unsigned* base = sm + ((int64_t)t * kh + 4 * k4) * 4 + w;
    base[0] = __byte_perm(t0, t1, 0x5410) ^ 0x80808080u;
    base[4] = __byte_perm(t0, t1, 0x7632) ^ 0x80808080u;
    base[8] = __byte_perm(t2, t3, 0x5410) ^ 0x80808080u;
    base[12] = __byte_perm(t2, t3, 0x7632) ^ 0x80808080u;
  }
  __syncthreads();
  constexpr unsigned kSig0 = sigma_packed(0), kSig1 = sigma_packed(1), kSig2 = sigma_packed(2), kSig3 = sigma_packed(3);
  if (threadIdx.x < 16 && acount[m0 + threadIdx.x] > kAcap) atomicOr(flag, 1);
  unsigned chk = 0;
  for (int n = threadIdx.x; n < nh; n += kCorr16Threads) {
    int acc[4][16] = {};
#pragma unroll
    for (int t = 0; t < kCorrSlots; ++t) {
      const int s = t + 1;
      const uint4* row = sm4 + (int64_t)t * kh;
      const int beg = bptr[s * (nh + 1) + n], mid = bmid[s * nh + n], end = bptr[s * (nh + 1) + n + 1];
      int p[16] = {};
#pragma unroll
      for (int side = 0; side < 2; ++side) {
        const int e0 = side ? mid : beg, e1 = side ? end : mid;
        for (int c0 = e0; c0 < e1; c0 += 128) {
          const int c1 = min(c0 + 128, e1);
          unsigned lo[4] = {}, hi[4] = {};
          int e = c0;
          for (; e + 1 < c1; e += 2) {
            const uint4 x0 = row[bidx[e]], x1 = row[bidx[e + 1]];
            const unsigned* a = &x0.x;
            const unsigned* b = &x1.x;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
              lo[w] += (a[w] & 0x00ff00ffu) + (b[w] & 0x00ff00ffu);
              hi[w] += __byte_perm(a[w], 0, 0x4341) + __byte_perm(b[w], 0, 0x4341);
            }
          }
          if (e < c1) {
            const uint4 x0 = row[bidx[e]];
            const unsigned* a = &x0.x;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
              lo[w] += a[w] & 0x00ff00ffu;
              hi[w] += __byte_perm(a[w], 0, 0x4341);
            }
          }
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            const int l0 = int(lo[w] & 0xffffu), l2 = int(lo[w] >> 16), l1 = int(hi[w] & 0xffffu), l3 = int(hi[w] >> 16);
            if (side) {
              p[4 * w] -= l0, p[4 * w + 1] -= l1, p[4 * w + 2] -= l2, p[4 * w + 3] -= l3;
            } else {
              p[4 * w] += l0, p[4 * w + 1] += l1, p[4 * w + 2] += l2, p[4 * w + 3] += l3;
            }
          }
        }
      }
      const int bias = 128 * ((mid - beg) - (end - mid));
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int v = p[i] - bias;
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          if (sigma(qq, s) > 0) acc[qq][i] += v;
          if (sigma(qq, s) < 0) acc[qq][i] -= v;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const unsigned* L = alist + (int64_t)(m0 + i) * kAcap;
      const int na = min(acount[m0 + i], kAcap);
      for (int e = 0; e < na; ++e) {
        const unsigned v = L[e];
        const int s = int(v >> 16), k = int(v & 0x7fffu);
        int f = ft[((int64_t)s * kh + k) * nh + n];
        if (v & 0x8000u) f = -f;
        acc[0][i] += sigma_rt(kSig0, s) * f;
        acc[1][i] += sigma_rt(kSig1, s) * f;
        acc[2][i] += sigma_rt(kSig2, s) * f;
        acc[3][i] += sigma_rt(kSig3, s) * f;
      }
    }
#pragma unroll
    for (int qq = 0; qq < 4; ++qq)
#pragma unroll
      for (int i = 0; i < 16; ++i)
        if (Store) {
          if (acc[qq][i]) {
            int32_t* at = o + (int64_t)((qq >> 1) * mh + m0 + i) * (2LL * nh) + (qq & 1) * nh + n;
            *at = int(unsigned(*at) + (unsigned(acc[qq][i]) << 8));
          }
        } else {
          chk = (chk ^ unsigned(acc[qq][i])) * 0x9e3779b1u;
        }
  }
  if (!Store) o[(int64_t)blockIdx.x * kCorr16Threads + threadIdx.x] = int(chk);
}

// The attacker's correction (timing only): bit 9 of O needs only O mod 2^10, so of B's carries' corrections only their
// value mod 4. A's forms of slots 1..5 are staged as two bit planes (bits 0 and 1), 32 rows a word, and each carry adds
// (or subtracts) its k's planes mod 4 with three logic ops. out [4][2][mh/32][nh] u32: per quadrant the two planes.
constexpr int kC4Threads = 512;

__device__ __forceinline__ void add4(unsigned& s0, unsigned& s1, unsigned x0, unsigned x1) {
  unsigned c = s0 & x0;
  s0 ^= x0;
  s1 ^= x1 ^ c;
}
__device__ __forceinline__ void sub4(unsigned& s0, unsigned& s1, unsigned x0, unsigned x1) {
  unsigned b = ~s0 & x0;
  s0 ^= x0;
  s1 ^= x1 ^ b;
}

__global__ void __launch_bounds__(kC4Threads, 1)
    corr4_kernel(const int8_t* __restrict__ wide_a, const int* __restrict__ bptr, const int* __restrict__ bmid,
                 const uint16_t* __restrict__ bidx, unsigned* __restrict__ out, int mh, int nh, int kh) {
  extern __shared__ uint4 sm4[];
  uint2* sm = reinterpret_cast<uint2*>(sm4);  // [5][kh]: (bit 0, bit 1) of 32 rows
  const int m0 = blockIdx.x * 32, warp = threadIdx.x / 32, lane = threadIdx.x % 32, k4s = kh / 4;
  const int64_t wld = (int64_t)kSlots * kh;
  for (int j = warp; j < kCorrSlots * k4s; j += kC4Threads / 32) {
    const int t = j / k4s, k4 = j % k4s;
    const unsigned v = *reinterpret_cast<const unsigned*>(wide_a + (int64_t)(m0 + lane) * wld + (int64_t)(t + 1) * kh + 4 * k4);
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      unsigned p0 = __ballot_sync(0xffffffffu, (v >> (8 * b)) & 1u), p1 = __ballot_sync(0xffffffffu, (v >> (8 * b)) & 2u);
      if (lane == b) sm[t * kh + 4 * k4 + b] = make_uint2(p0, p1);
    }
  }
  __syncthreads();
  const int rows32 = mh / 32;
  for (int n = threadIdx.x; n < nh; n += kC4Threads) {
    unsigned q0[4] = {}, q1[4] = {};
#pragma unroll
    for (int t = 0; t < kCorrSlots; ++t) {
      const int s = t + 1;
      const uint2* row = sm + t * kh;
      const int beg = bptr[s * (nh + 1) + n], mid = bmid[s * nh + n], end = bptr[s * (nh + 1) + n + 1];
      unsigned a0 = 0, a1 = 0;
      for (int e = beg; e < mid; ++e) {
        const uint2 x = row[bidx[e]];
        add4(a0, a1, x.x, x.y);
      }
      for (int e = mid; e < end; ++e) {
        const uint2 x = row[bidx[e]];
        sub4(a0, a1, x.x, x.y);
      }
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        if (sigma(qq, s) > 0) add4(q0[qq], q1[qq], a0, a1);
        if (sigma(qq, s) < 0) sub4(q0[qq], q1[qq], a0, a1);
      }
    }
#pragma unroll
    for (int qq = 0; qq < 4; ++qq) {
      out[((int64_t)(qq * 2) * rows32 + blockIdx.x) * nh + n] = q0[qq];
      out[((int64_t)(qq * 2 + 1) * rows32 + blockIdx.x) * nh + n] = q1[qq];
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// SM clock: clock64 against the global timer over a short spin, from one thread.
__global__ void clock_kernel(unsigned long long* out, long long spin) {
  unsigned long long g0, g1;
  long long c0 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  while (clock64() - c0 < spin) {
  }
  long long c1 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  out[0] = (unsigned long long)(c1 - c0);
  out[1] = g1 - g0;
}

// ---------------------------------------------------------------------------------------------------------------------
// CUTLASS GEMMs: s8 A row-major (K-major), s8 B column-major (N x K, K-major), s32 D row-major, no source.
using RowMajor = cutlass::layout::RowMajor;
using ColMajor = cutlass::layout::ColumnMajor;

template <class TileShape, class ClusterShape>
struct CutlassGemm {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, int32_t, int32_t, void, RowMajor, 4, int32_t, RowMajor, 4,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, int8_t, RowMajor, 16, int8_t, ColMajor, 16, int32_t, TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideD = typename Kernel::StrideD;

  Gemm gemm;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;

  // (m, n, k, l) with A at a (lda, batch stride sa), B at b (ldb, sb), D at d (ldd, sd)
  int run(int m, int n, int k, int l, const int8_t* a, int64_t lda, int64_t sa, const int8_t* b, int64_t ldb,
          int64_t sb, int32_t* d, int64_t ldd, int64_t sd, cudaStream_t st) {
    StrideA dA = make_stride(lda, Int<1>{}, sa);
    StrideB dB = make_stride(ldb, Int<1>{}, sb);
    StrideD dD = make_stride(ldd, Int<1>{}, sd);
    typename Gemm::Arguments args{
        l > 1 ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, l},
        {a, dA, b, dB},
        {{1, 0}, nullptr, dD, d, dD}};
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 101;
    size_t ws = Gemm::get_workspace_size(args);
    if (ws > workspace_bytes) {
      if (workspace) cudaFree(workspace);
      if (cudaMalloc(&workspace, ws) != cudaSuccess) return 102;
      workspace_bytes = ws;
    }
    if (gemm.initialize(args, workspace, st) != cutlass::Status::kSuccess) return 103;
    return gemm.run(st) == cutlass::Status::kSuccess ? 0 : 104;
  }
};

using Plain256 = CutlassGemm<Shape<_128, _256, _128>, Shape<_1, _1, _1>>;
using Plain256c2 = CutlassGemm<Shape<_128, _256, _128>, Shape<_2, _1, _1>>;
using Plain128 = CutlassGemm<Shape<_128, _128, _128>, Shape<_1, _1, _1>>;

// ---------------------------------------------------------------------------------------------------------------------
// The fused orbit: CUTLASS's sm90 TMA warp-specialized mainloop (its producer `load` and consumer `mma`), driven over the
// seven products of each 128 x 128 tile of the half grid by a persistent kernel of its own.
using FusedTile = Shape<_128, _128, _128>;
using FusedMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, int8_t, RowMajor, 16, int8_t, ColMajor, 16, int32_t, FusedTile,
    Shape<_1, _1, _1>, cutlass::gemm::collective::StageCountAutoCarveout<1024>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using FusedPipeline = typename FusedMainloop::MainloopPipeline;
using FusedTiledMma = typename FusedMainloop::TiledMma;

struct FusedParams {
  typename FusedMainloop::Params mainloop;
  int32_t* d;
  int64_t ldd;
  int mh, nh, kt;            // half grid rows and columns, k-tiles of 128 a product
  int tiles_m, tiles_n, group;
};

struct FusedSmem {
  typename FusedMainloop::TensorStorage tensors;
  alignas(16) typename FusedMainloop::PipelineStorage pipeline;
};

constexpr int kFusedThreads = 384;

__device__ __forceinline__ void tile_coord(int t, FusedParams const& p, int& tm, int& tn) {
  // groups of `group` tile columns, the rows fastest within a group, so a round of tiles shares B's columns
  int per = p.group * p.tiles_m;
  int g = t / per, r = t % per;
  int gn = min(p.group, p.tiles_n - g * p.group);
  tn = g * p.group + r % gn;
  tm = r / gn;
}

__global__ void __launch_bounds__(kFusedThreads, 1) orbit_fused_kernel(CUTLASS_GRID_CONSTANT FusedParams const p) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(128) char smem_raw[];
  FusedSmem& smem = *reinterpret_cast<FusedSmem*>(smem_raw);
  int thread_idx = threadIdx.x;
  int warp_group = thread_idx / 128;
  int warp_in_group = (thread_idx / 32) % 4;
  int lane = thread_idx % 32;
  if (thread_idx == 0) FusedMainloop::prefetch_tma_descriptors(p.mainloop);

  typename FusedPipeline::Params pp;
  if (warp_group == 0 && warp_in_group == 0) pp.role = FusedPipeline::ThreadCategory::Producer;
  if (warp_group >= 1) pp.role = FusedPipeline::ThreadCategory::Consumer;
  pp.is_leader = (thread_idx % 128) == 0;
  pp.num_consumers = 256;
  pp.num_producers = 1;
  pp.transaction_bytes = p.mainloop.tma_transaction_bytes;
  FusedPipeline pipeline(smem.pipeline, pp, Shape<_1, _1, _1>{});
  __syncthreads();

  FusedMainloop collective;
  int tiles = p.tiles_m * p.tiles_n;
  auto problem = make_shape(p.mh, p.nh, kSlots * p.kt * 128, 1);
  auto load_inputs = collective.load_init(problem, p.mainloop);
  int ktiles = kSlots * p.kt;

  if (warp_group == 0) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    if (warp_in_group == 0) {
      auto state = cutlass::make_producer_start_state<FusedPipeline>();
      auto gA = get<0>(load_inputs);
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        int tm, tn;
        tile_coord(t, p, tm, tn);
        auto blk = make_coord(tm, tn, _, 0);
        auto k_iter = cute::make_coord_iterator(idx2crd(0, shape<3>(gA)), shape<3>(gA));
        collective.load(p.mainloop, pipeline, state, load_inputs, blk, k_iter, ktiles, lane, 0u, smem.tensors);
        state.advance(ktiles);
      }
      collective.load_tail(pipeline, state);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    int mma_thread_idx = thread_idx - 128;
    FusedTiledMma tiled_mma;
    typename FusedMainloop::PipelineState cstate;
    auto r0 = partition_fragment_C(tiled_mma, take<0, 2>(FusedTile{}));
    auto r1 = partition_fragment_C(tiled_mma, take<0, 2>(FusedTile{}));
    auto r2 = partition_fragment_C(tiled_mma, take<0, 2>(FusedTile{}));
    Tensor cD = make_identity_tensor(make_shape(Int<128>{}, Int<128>{}));
    Tensor tCcD = tiled_mma.get_thread_slice(mma_thread_idx).partition_C(cD);
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      int tm, tn;
      tile_coord(t, p, tm, tn);
      auto product = [&](auto& acc) {
        collective.mma(pipeline, cstate, acc, p.kt, mma_thread_idx, smem.tensors, p.mainloop);
        collective.mma_tail(pipeline, cstate, p.kt);
        cstate.advance(p.kt);
      };
      // O's quadrant (qi, qj) of this tile = x + sign · y
      auto store = [&](auto const& x, auto const& y, int sign, int qi, int qj) {
        int row0 = tm * 128, col0 = tn * 128;
        int32_t* base = p.d + (int64_t)(qi * p.mh + row0) * p.ldd + qj * p.nh + col0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(x); i += 2) {
          auto c = tCcD(i);
          int m = get<0>(c), n = get<1>(c);
          int v0 = sign > 0 ? add32(x(i), y(i)) : sub32(x(i), y(i));
          int v1 = sign > 0 ? add32(x(i + 1), y(i + 1)) : sub32(x(i + 1), y(i + 1));
          if (row0 + m < p.mh && col0 + n < p.nh)
            *reinterpret_cast<int2*>(base + (int64_t)m * p.ldd + n) = make_int2(v0, v1);
        }
      };
      product(r0);  // M2
      product(r1);  // M4
      store(r0, r1, 1, 1, 0);  // O10 = M2 + M4
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(r0); ++i) r0(i) = sub32(0, r0(i));
      product(r2);  // M1
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(r0); ++i) {
        r0(i) = add32(r0(i), r2(i));
        r1(i) = add32(r1(i), r2(i));
      }
      product(r2);  // M6
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(r0); ++i) r0(i) = add32(r0(i), r2(i));
      product(r2);  // M7
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(r1); ++i) r1(i) = add32(r1(i), r2(i));
      product(r2);  // M3
      store(r0, r2, 1, 1, 1);  // O11 = M1 − M2 + M6 + M3
      product(r0);  // M5
      store(r2, r0, 1, 0, 1);  // O01 = M3 + M5
      store(r1, r0, -1, 0, 0);  // O00 = M4 + M1 + M7 − M5
    }
  }
#endif
}

int fused_smem_bytes() { return int(sizeof(FusedSmem)); }

// ---------------------------------------------------------------------------------------------------------------------
int sm_count() {
  static int n = 0;
  if (!n) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}

int launch_forms(const int8_t* x, int8_t* wide, unsigned long long* counts, int8_t* carry, unsigned* alist, int* acount,
                 int r, int k, int which, cudaStream_t st) {
  if (r % 2 || k % 32 || k / 2 >= 32768 || r / 2 > 65535) return 201;
  int rh = r / 2, kh = k / 2;
  dim3 grid(unsigned((kh / 16 + kFormsThreads - 1) / kFormsThreads), unsigned(rh));
  if (acount && cudaMemsetAsync(acount, 0, sizeof(int) * rh, st) != cudaSuccess) return 203;
  if (which)
    forms_kernel<1><<<grid, kFormsThreads, 0, st>>>(x, wide, counts, carry, nullptr, nullptr, rh, kh);
  else
    forms_kernel<0><<<grid, kFormsThreads, 0, st>>>(x, wide, counts, carry, alist, acount, rh, kh);
  return cudaGetLastError() == cudaSuccess ? 0 : 202;
}

int launch_fused(const int8_t* a_wide, const int8_t* b_wide, int32_t* d, int m, int n, int k, int group, int ctas,
                 cudaStream_t st) {
  if (m % 2 || n % 2 || k % 256) return 301;
  int mh = m / 2, nh = n / 2, kh = k / 2;
  using StrideA = typename FusedMainloop::StrideA;
  using StrideB = typename FusedMainloop::StrideB;
  int64_t ld = (int64_t)kSlots * kh;
  typename FusedMainloop::Arguments margs{a_wide, StrideA(make_stride(ld, Int<1>{}, ld * mh)), b_wide,
                                          StrideB(make_stride(ld, Int<1>{}, ld * nh))};
  auto problem = make_shape(mh, nh, kSlots * kh, 1);
  if (!FusedMainloop::can_implement(problem, margs)) return 302;
  FusedParams p;
  p.mainloop = FusedMainloop::to_underlying_arguments(problem, margs, nullptr);
  p.d = d;
  p.ldd = n;
  p.mh = mh;
  p.nh = nh;
  p.kt = kh / 128;
  p.tiles_m = (mh + 127) / 128;
  p.tiles_n = (nh + 127) / 128;
  p.group = group > 0 ? group : 8;
  int tiles = p.tiles_m * p.tiles_n;
  int grid = ctas > 0 ? ctas : sm_count();
  if (grid > tiles) grid = tiles;
  static bool attr = false;
  int smem = fused_smem_bytes();
  if (!attr) {
    if (cudaFuncSetAttribute(orbit_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
      return 303;
    attr = true;
  }
  orbit_fused_kernel<<<grid, kFusedThreads, smem, st>>>(p);
  return cudaGetLastError() == cudaSuccess ? 0 : 304;
}

template <int G, bool Store>
int launch_corr_g(const int8_t* wide_a, const int* bptr, const int* bmid, const uint16_t* bidx, const unsigned* alist,
                  const int* acount, const int16_t* ft, int32_t* o, int* flag, int mh, int nh, int kh, cudaStream_t st) {
  if (mh % (4 * G)) return 802;
  int smem = int(sizeof(unsigned)) * kCorrSlots * G * (kh + kCorrPad);
  static int attr = 0;
  if (attr < smem) {
    if (cudaFuncSetAttribute(corr_kernel<G, Store>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
      return 803;
    attr = smem;
  }
  corr_kernel<G, Store><<<mh / (4 * G), kCorrThreads, smem, st>>>(wide_a, bptr, bmid, bidx, alist, acount, ft, o, flag,
                                                                   mh, nh, kh);
  return cudaGetLastError() == cudaSuccess ? 0 : 804;
}

template <bool Store>
int launch_corr_s(const int8_t* wide_a, const int* bptr, const int* bmid, const uint16_t* bidx, const unsigned* alist,
                  const int* acount, const int16_t* ft, int32_t* o, int* flag, int mh, int nh, int kh, cudaStream_t st) {
  if (kh <= 2048) return launch_corr_g<4, Store>(wide_a, bptr, bmid, bidx, alist, acount, ft, o, flag, mh, nh, kh, st);
  if (kh <= 4096) return launch_corr_g<2, Store>(wide_a, bptr, bmid, bidx, alist, acount, ft, o, flag, mh, nh, kh, st);
  if (kh <= 8192) return launch_corr_g<1, Store>(wide_a, bptr, bmid, bidx, alist, acount, ft, o, flag, mh, nh, kh, st);
  return 801;
}

template <bool Store>
int launch_corr16(const int8_t* wide_a, const int* bptr, const int* bmid, const uint16_t* bidx, const unsigned* alist,
                  const int* acount, const int16_t* ft, int32_t* o, int* flag, int mh, int nh, int kh, cudaStream_t st) {
  int smem = int(sizeof(uint4)) * kCorrSlots * kh;
  static int attr = 0;
  if (attr < smem) {
    if (cudaFuncSetAttribute(corr16_kernel<Store>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
      return 805;
    attr = smem;
  }
  corr16_kernel<Store><<<mh / 16, kCorr16Threads, smem, st>>>(wide_a, bptr, bmid, bidx, alist, acount, ft, o, flag, mh,
                                                               nh, kh);
  return cudaGetLastError() == cudaSuccess ? 0 : 806;
}

// mode bit 0: update O (else compute only, timing); bit 1: the 4-rows-a-thread kernel even where corr16 fits
int launch_corr(const int8_t* wide_a, const int* bptr, const int* bmid, const uint16_t* bidx, const unsigned* alist,
                const int* acount, const int16_t* ft, int32_t* o, int* flag, int m, int n, int k, int mode,
                cudaStream_t st) {
  int mh = m / 2, nh = n / 2, kh = k / 2;
  bool store = mode & 1;
  if (kh % 4) return 801;
  if (!(mode & 2) && kh <= 2048 && mh % 16 == 0)
    return store ? launch_corr16<true>(wide_a, bptr, bmid, bidx, alist, acount, ft, o, flag, mh, nh, kh, st)
                 : launch_corr16<false>(wide_a, bptr, bmid, bidx, alist, acount, ft, o, flag, mh, nh, kh, st);
  return store ? launch_corr_s<true>(wide_a, bptr, bmid, bidx, alist, acount, ft, o, flag, mh, nh, kh, st)
               : launch_corr_s<false>(wide_a, bptr, bmid, bidx, alist, acount, ft, o, flag, mh, nh, kh, st);
}

int launch_corr4(const int8_t* wide_a, const int* bptr, const int* bmid, const uint16_t* bidx, unsigned* out, int m,
                 int n, int k, cudaStream_t st) {
  int mh = m / 2, nh = n / 2, kh = k / 2;
  if (mh % 32 || kh % 4) return 811;
  int smem = int(sizeof(uint2)) * kCorrSlots * kh;
  if (smem > 227 * 1024) return 812;
  static int attr = 0;
  if (attr < smem) {
    if (cudaFuncSetAttribute(corr4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return 813;
    attr = smem;
  }
  corr4_kernel<<<mh / 32, kC4Threads, smem, st>>>(wide_a, bptr, bmid, bidx, out, mh, nh, kh);
  return cudaGetLastError() == cudaSuccess ? 0 : 814;
}

Plain256& plain256() {
  static Plain256 g;
  return g;
}
Plain256c2& plain256c2() {
  static Plain256c2 g;
  return g;
}
Plain128& plain128() {
  static Plain128 g;
  return g;
}
Plain256& batched256() {
  static Plain256 g;
  return g;
}

}  // namespace orbit_impl

using namespace orbit_impl;

// ---------------------------------------------------------------------------------------------------------------------
// The library's C surface (bench.py's, by ctypes). Every call is issued on `st` and returns 0, or nonzero on refusal.
extern "C" {

int orbit_info(int* out) {
  out[0] = fused_smem_bytes();
  out[1] = FusedMainloop::DispatchPolicy::Stages;
  out[2] = kFusedThreads;
  out[3] = sm_count();
  return 0;
}

// x [r, k] s8 -> wide [r/2, 7 k/2] s8, counts[7] u64 added to, carry [7, r/2, k/2] s8 when not null; which 0: A, 1: B.
// For A, alist [r/2, 1024] u32 and acount [r/2] s32 (zeroed here) when not null: A's carries a row, as corr reads them.
int orbit_forms(const int8_t* x, int8_t* wide, unsigned long long* counts, int8_t* carry, unsigned* alist, int* acount,
                int r, int k, int which, cudaStream_t st) {
  return launch_forms(x, wide, counts, carry, alist, acount, r, k, which, st);
}

// O [m, n] s32 += the carries' corrections (see corr_kernel): B's lists bptr [7, n/2 + 1], bmid [7, n/2] s32, bidx u16;
// A's alist/acount from orbit_forms; ft [7, k/2, n/2] s16 B's exact forms. *flag set if a row's list overflowed.
// mode bit 0: O updated (else computed only, for timing); bit 1: corr_kernel's 4 rows a thread even where corr16 fits.
int orbit_corr(const int8_t* a_wide, const int* bptr, const int* bmid, const uint16_t* bidx, const unsigned* alist,
               const int* acount, const int16_t* ft, int32_t* o, int* flag, int m, int n, int k, int mode,
               cudaStream_t st) {
  return launch_corr(a_wide, bptr, bmid, bidx, alist, acount, ft, o, flag, m, n, k, mode, st);
}

// out [4][2][m/64][n/2] u32 = B's carries' corrections mod 4 as bit planes (corr4_kernel; timing only)
int orbit_corr4(const int8_t* a_wide, const int* bptr, const int* bmid, const uint16_t* bidx, unsigned* out, int m, int n,
                int k, cudaStream_t st) {
  return launch_corr4(a_wide, bptr, bmid, bidx, out, m, n, k, st);
}

// P [7, m/2, n/2] s32 = the seven products of the wide forms, CUTLASS batched (L = 7)
int orbit_batched(const int8_t* a_wide, const int8_t* b_wide, int32_t* prod, int m, int n, int k, cudaStream_t st) {
  int mh = m / 2, nh = n / 2, kh = k / 2;
  int64_t ld = (int64_t)kSlots * kh;
  return batched256().run(mh, nh, kh, kSlots, a_wide, ld, kh, b_wide, ld, kh, prod, nh, (int64_t)mh * nh, st);
}

// O [m, n] s32 from P
int orbit_assemble(const int32_t* prod, int32_t* o, int m, int n, cudaStream_t st) {
  int mh = m / 2, nh = n / 2;
  if (nh % 4) return 401;
  long long threads = (long long)mh * (nh / 4);
  assemble_kernel<<<unsigned((threads + 255) / 256), 256, 0, st>>>(prod, o, mh, nh);
  return cudaGetLastError() == cudaSuccess ? 0 : 402;
}

// O [m, n] s32 from the wide forms, fused; group: tile columns a raster group (0: 8); ctas: persistent CTAs (0: every SM)
int orbit_fused(const int8_t* a_wide, const int8_t* b_wide, int32_t* o, int m, int n, int k, int group, int ctas,
                cudaStream_t st) {
  return launch_fused(a_wide, b_wide, o, m, n, k, group, ctas, st);
}

// D [m, n] s32 = A Bᵀ by CUTLASS: variant 0 128x256, 1 128x256 in clusters of 2 along M, 2 128x128
int orbit_plain(const int8_t* a, const int8_t* b, int32_t* d, int m, int n, int k, int variant, cudaStream_t st) {
  switch (variant) {
    case 0: return plain256().run(m, n, k, 1, a, k, (int64_t)m * k, b, k, (int64_t)n * k, d, n, (int64_t)m * n, st);
    case 1: return plain256c2().run(m, n, k, 1, a, k, (int64_t)m * k, b, k, (int64_t)n * k, d, n, (int64_t)m * n, st);
    case 2: return plain128().run(m, n, k, 1, a, k, (int64_t)m * k, b, k, (int64_t)n * k, d, n, (int64_t)m * n, st);
  }
  return 501;
}

// out[0] SM clocks and out[1] global-timer nanoseconds over a spin of `spin` clocks on one SM
int orbit_clock(unsigned long long* out, long long spin, cudaStream_t st) {
  clock_kernel<<<1, 1, 0, st>>>(out, spin);
  return cudaGetLastError() == cudaSuccess ? 0 : 601;
}

// ---------------------------------------------------------------------------------------------------------------------
// ext_gemm.h's entry (benchmarks/pouw/accel_v31): D = A Bᵀ exactly, through the fused orbit and the carries'
// corrections. B's forms, their carry lists and exact forms are made at each load (on the host, from the device's carry
// plane); a call whose A lists overflowed (more than 1024 carries in a half-grid row) is recomputed by the plain CUTLASS
// GEMM, so every D it returns is exact.
struct ExtCtx {
  int m, n, k;
  int8_t* a_wide = nullptr;
  int8_t* b_wide = nullptr;
  unsigned long long* counts = nullptr;  // [2][7]: A's, B's
  unsigned* alist = nullptr;
  int* acount = nullptr;
  int* flag = nullptr;
  int* bptr = nullptr;
  int* bmid = nullptr;
  uint16_t* bidx = nullptr;
  int16_t* ft = nullptr;
  int8_t* b = nullptr;  // B itself, for a call the orbit refuses
};

void ext_release(ExtCtx* h) {
  cudaFree(h->a_wide);
  cudaFree(h->b_wide);
  cudaFree(h->counts);
  cudaFree(h->alist);
  cudaFree(h->acount);
  cudaFree(h->flag);
  cudaFree(h->bptr);
  cudaFree(h->bmid);
  cudaFree(h->bidx);
  cudaFree(h->ft);
  cudaFree(h->b);
}

void* v31_ext_init(int m, int n, int k) {
  if (m < 2 || m % 32 || n % 256 || k % 256 || k / 2 > 8192) return nullptr;
  ExtCtx* h = new ExtCtx{m, n, k};
  size_t mh = m / 2, nh = n / 2, kh = k / 2;
  bool ok = cudaMalloc(&h->a_wide, mh * kSlots * kh) == cudaSuccess &&
            cudaMalloc(&h->b_wide, nh * kSlots * kh) == cudaSuccess &&
            cudaMalloc(&h->counts, 2 * kSlots * sizeof(unsigned long long)) == cudaSuccess &&
            cudaMalloc(&h->alist, mh * kAcap * sizeof(unsigned)) == cudaSuccess &&
            cudaMalloc(&h->acount, mh * sizeof(int)) == cudaSuccess && cudaMalloc(&h->flag, sizeof(int)) == cudaSuccess &&
            cudaMalloc(&h->bptr, kSlots * (nh + 1) * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&h->bmid, kSlots * nh * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&h->ft, (size_t)kSlots * kh * nh * sizeof(int16_t)) == cudaSuccess &&
            cudaMalloc(&h->b, (size_t)n * k) == cudaSuccess;
  if (!ok) {
    ext_release(h);
    delete h;
    return nullptr;
  }
  return h;
}

int v31_ext_load(void* hv, const int8_t* b, cudaStream_t st) {
  ExtCtx* h = static_cast<ExtCtx*>(hv);
  size_t nh = h->n / 2, kh = h->k / 2, plane = nh * kh;
  int8_t* carry = nullptr;
  if (cudaMalloc(&carry, kSlots * plane) != cudaSuccess) return 701;
  int rc = cudaMemsetAsync(h->counts + kSlots, 0, kSlots * sizeof(unsigned long long), st) == cudaSuccess &&
                   cudaMemcpyAsync(h->b, b, (size_t)h->n * h->k, cudaMemcpyDeviceToDevice, st) == cudaSuccess
               ? 0
               : 702;
  if (!rc) rc = launch_forms(b, h->b_wide, h->counts + kSlots, carry, nullptr, nullptr, h->n, h->k, 1, st);
  int8_t* hw = rc ? nullptr : new int8_t[kSlots * plane];
  int8_t* hc = rc ? nullptr : new int8_t[kSlots * plane];
  if (!rc && (cudaMemcpyAsync(hw, h->b_wide, kSlots * plane, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
              cudaMemcpyAsync(hc, carry, kSlots * plane, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
              cudaStreamSynchronize(st) != cudaSuccess))
    rc = 703;
  cudaFree(carry);
  if (!rc) {
    // B's lists: per slot and column, the +1 carries' k's, then the −1 ones'
    int* ptr = new int[kSlots * (nh + 1)];
    int* mid = new int[kSlots * nh];
    size_t total = 0;
    for (size_t i = 0; i < kSlots * plane; ++i) total += hc[i] != 0;
    uint16_t* idx = new uint16_t[total ? total : 1];
    int16_t* ft = new int16_t[kSlots * plane];
    size_t at = 0;
    for (int s = 0; s < kSlots; ++s)
      for (size_t c = 0; c < nh; ++c) {
        const int8_t* cr = hc + (size_t)s * plane + c * kh;
        const int8_t* wr = hw + c * kSlots * kh + (size_t)s * kh;
        ptr[s * (nh + 1) + c] = int(at);
        for (size_t j = 0; j < kh; ++j)
          if (cr[j] > 0) idx[at++] = uint16_t(j);
        mid[s * nh + c] = int(at);
        for (size_t j = 0; j < kh; ++j)
          if (cr[j] < 0) idx[at++] = uint16_t(j);
        if (c + 1 == nh) ptr[s * (nh + 1) + nh] = int(at);
        for (size_t j = 0; j < kh; ++j) ft[((size_t)s * kh + j) * nh + c] = int16_t(wr[j] + 256 * cr[j]);
      }
    cudaFree(h->bidx);
    h->bidx = nullptr;
    if (cudaMalloc(&h->bidx, (total ? total : 1) * sizeof(uint16_t)) != cudaSuccess ||
        cudaMemcpy(h->bidx, idx, total * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(h->bptr, ptr, kSlots * (nh + 1) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(h->bmid, mid, kSlots * nh * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(h->ft, ft, kSlots * plane * sizeof(int16_t), cudaMemcpyHostToDevice) != cudaSuccess)
      rc = 704;
    delete[] ptr;
    delete[] mid;
    delete[] idx;
    delete[] ft;
  }
  delete[] hw;
  delete[] hc;
  return rc;
}

int v31_ext_gemm(void* hv, const int8_t* a, int32_t* d, cudaStream_t st) {
  ExtCtx* h = static_cast<ExtCtx*>(hv);
  if (cudaMemsetAsync(h->counts, 0, kSlots * sizeof(unsigned long long), st) != cudaSuccess ||
      cudaMemsetAsync(h->flag, 0, sizeof(int), st) != cudaSuccess)
    return 711;
  int rc = launch_forms(a, h->a_wide, h->counts, nullptr, h->alist, h->acount, h->m, h->k, 0, st);
  if (!rc) rc = launch_fused(h->a_wide, h->b_wide, d, h->m, h->n, h->k, 8, 0, st);
  if (!rc) rc = launch_corr(h->a_wide, h->bptr, h->bmid, h->bidx, h->alist, h->acount, h->ft, d, h->flag, h->m, h->n,
                            h->k, 1, st);
  if (rc) return rc;
  int flag = 0;
  if (cudaMemcpyAsync(&flag, h->flag, sizeof flag, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
      cudaStreamSynchronize(st) != cudaSuccess)
    return 712;
  if (flag && plain256().run(h->m, h->n, h->k, 1, a, h->k, (int64_t)h->m * h->k, h->b, h->k, (int64_t)h->n * h->k, d,
                             h->n, (int64_t)h->m * h->n, st))
    return 715;
  return 0;
}

void v31_ext_free(void* hv) {
  ExtCtx* h = static_cast<ExtCtx*>(hv);
  if (!h) return;
  ext_release(h);
  delete h;
}

}  // extern "C"
