// What an overflow correction costs inside an s8 Strassen leaf, in W1 writes (32 s8 MACs at the card's peak) per
// (noise row, nonzero): hardness_premise's timing (natural.py's `--alu`).
//
//   leaf_corr FIXTURE_DIR OUT_DIR [--m M] [--reps R] [--peak MAC_PER_SM_CLK]
//
// FIXTURE_DIR is leaf_fixture.py's: the leaf's B operand lo (n × K int8, a natural down projection's Strassen W-form's low
// digit) and its overflow digit hi as per-column (k, hi) lists. E, the noise side (m × K int8), is drawn here. Arms:
//   leaf      E · lo on mma.sync m16n8k32 s8: 128 × 128 tiles of 8 warps (64 × 32 each), a 3-stage cp.async ring of
//             64-byte k-tiles, swizzled ldmatrix; the epilogue stores M^T (n × m int32, a column's rows contiguous).
//   leaf+imad the same leaf, and in its epilogue 256 · E · hi for every overflowing entry of the tile's columns: a warp
//             a column, a lane 4 rows, one 32-bit load of E^T (K × m) per (k, lane) and per row a PRMT and an IMAD.
//   leaf+pack the same, with E^T stored biased (E + 128 as bytes) and summed two rows a register in 16-bit lanes:
//             per (k, lane) a load, two PRMTs and two IMADs (signs kept apart, the bias taken off once a column).
//   corr+imad / corr+pack   the correction alone, in place on M^T, over the columns that have overflow.
//   empty     a kernel that does nothing on corr's grid (its launch cost).
//   peak      mma.sync m16n8k32 s8 from registers: the card's s8 rate, the W1 unit (--peak overrides it, e.g. the H200's
//             wgmma rate when this runs there on mma.sync).
// The gate runs before any timing: every arm's output, written over 0xA5, matches the CPU's E · (lo + 256 hi) on 64
// rows × all columns (E · lo for `leaf`), and the poison alone fails it.
#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../../gpu_kit/bench.cuh"

typedef uint32_t u32;

constexpr int BM = 128, BN = 128, BK = 64, STAGES = 3, THREADS = 256;
constexpr int TILE_BYTES = BM * BK;
constexpr int SMEM_MAIN = STAGES * 2 * TILE_BYTES;
constexpr int TP = BM + 4;  // the epilogue's int32 [64 columns][TP]: fragment stores and 16-byte reads both conflict-free
constexpr int SMEM_EPI = 64 * TP * 4;
constexpr int SMEM = SMEM_MAIN > SMEM_EPI ? SMEM_MAIN : SMEM_EPI;
constexpr int PEAK_CHAINS = 8;

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void mma(int* c, const u32* a, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// byte b of x sign-extended (PTX prmt's default mode: a selector nibble's bit 3 replicates the byte's sign)
template <u32 SEL>
__device__ __forceinline__ int sext(u32 x) {
  u32 r;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(x), "n"(SEL));
  return (int)r;
}
// a 64-byte k-tile row's 16-byte chunk, XOR-swizzled so ldmatrix's 8 rows land on 8 distinct 16-byte bank groups
__device__ __forceinline__ int swz(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ void load_stage(u32 sa, u32 sb, const int8_t* __restrict__ E, const int8_t* __restrict__ L,
                                           int m0, int n0, int k0, int K, int tid) {
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int idx = tid + i * THREADS, row = idx >> 2, ch = idx & 3;
    cp_async16(sa + swz(row, ch), E + (size_t)(m0 + row) * K + k0 + ch * 16);
    cp_async16(sb + swz(row, ch), L + (size_t)(n0 + row) * K + k0 + ch * 16);
  }
}

// hi is public (W's form), so everything about it is a table: per entry E^T's row offset in bytes, and per column pack's
// two constants
struct Hi {
  const u32* Et;             // E^T, K × m bytes as words (CORR 2: biased, E + 128)
  const int* colptr;         // n + 1
  const longlong2* ent;      // per overflowing (k, j): {k · m, hi}
  const int2* colc;          // per column: {B, B + 128 Σ hi}, B = 255 Σ_{hi < 0} |hi|
};

// v (rows m0 + 4 lane .. + 3 of column j) += 256 · Σ_k E[row, k] hi[k, j]
template <int CORR>
__device__ __forceinline__ void correct(int4& v, int j, int m0, int lane, const Hi& h) {
  const int beg = __ldg(h.colptr + j), end = __ldg(h.colptr + j + 1);
  if (beg == end) return;
  const char* col = reinterpret_cast<const char*>(h.Et + (m0 >> 2) + lane);
  if constexpr (CORR == 1) {
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma unroll 2
    for (int p = beg; p < end; p++) {
      const longlong2 e = __ldg(h.ent + p);
      const int d = (int)e.y;
      const u32 x = __ldg(reinterpret_cast<const u32*>(col + e.x));
      a0 += d * sext<0x8880>(x);
      a1 += d * sext<0x9991>(x);
      a2 += d * sext<0xAAA2>(x);
      a3 += d * sext<0xBBB3>(x);
    }
    v.x += a0 * 256;
    v.y += a1 * 256;
    v.z += a2 * 256;
    v.w += a3 * 256;
  } else {
    // Two rows a register in 16-bit lanes: a lane ends at B + Σ hi·(E + 128) = Σ_{hi>0} hi·u + Σ_{hi<0} |hi|·(255 − u),
    // in [0, 255 Σ|hi|] ⊂ [0, 65536) (checked on the host), so the 32-bit sum, exact mod 2^32 whatever it borrows on
    // the way, holds both lanes exactly; one IMAD a register a nonzero, no branch on hi's sign
    const int2 c = __ldg(h.colc + j);
    const u32 b2 = (u32)c.x * 0x10001u;
    u32 p01 = b2, p23 = b2;
#pragma unroll 2
    for (int p = beg; p < end; p++) {
      const longlong2 e = __ldg(h.ent + p);
      const u32 d = (u32)e.y;
      const u32 x = __ldg(reinterpret_cast<const u32*>(col + e.x));
      p01 += d * __byte_perm(x, 0, 0x4140);
      p23 += d * __byte_perm(x, 0, 0x4342);
    }
    v.x += ((int)(p01 & 0xffff) - c.y) * 256;
    v.y += ((int)(p01 >> 16) - c.y) * 256;
    v.z += ((int)(p23 & 0xffff) - c.y) * 256;
    v.w += ((int)(p23 >> 16) - c.y) * 256;
  }
}

// The batched corrections (CORR 3: int8 E; CORR 4: E in [−1, 2], D-OUT's level-1 noise forms, 2 bits a row): a lane
// group of 8 lanes a column, a lane 16 rows (m0 + 16 l8 ..), so a warp takes 4 columns at once. The column's entries are
// loaded across the group's lanes and broadcast by SHFL, and 4 nonzeros' E^T loads are in flight before any is used.
struct HiB {
  const void* Et;      // BITS 8: E^T biased bytes (E + 128), K × m; BITS 2: E + 1 in 2 bits, 16 rows a word (row 4s + t
                       // at bit 8t + 2s), K × m / 16 words
  const int* colptr;   // n + 1
  const u32* ent;      // per entry: (k · m / 16) << 8 | (hi & 255)
  const int2* colc;    // per column: {B, B + c Σ hi}: BITS 8 B = 255 Σ_{hi<0}|hi|, c = 128; BITS 2 B = 3 Σ_{hi<0}|hi|, c = 1
};

template <int BITS>
__device__ __forceinline__ void correct16(int4 (&v)[4], int j, int m0, int lane, const HiB& h) {
  const int l8 = lane & 7;
  const int beg = j >= 0 ? __ldg(h.colptr + j) : 0, cnt = j >= 0 ? __ldg(h.colptr + j + 1) - beg : 0;
  const int maxc = __reduce_max_sync(0xffffffffu, cnt);
  if (maxc == 0) return;
  const int2 c = j >= 0 ? __ldg(h.colc + j) : make_int2(0, 0);
  constexpr int W = BITS == 8 ? 8 : 4;
  u32 acc[W];
  const u32 b = BITS == 8 ? (u32)c.x * 0x10001u : (u32)c.x * 0x01010101u;
#pragma unroll
  for (int w = 0; w < W; w++) acc[w] = b;
  const size_t base = (size_t)(m0 >> 4) + l8;
  u32 mine = 0;
#pragma unroll 1
  for (int q0 = 0; q0 < maxc; q0 += 4) {
    if ((q0 & 7) == 0) mine = q0 + l8 < cnt ? __ldg(h.ent + beg + q0 + l8) : 0u;  // 0: offset 0, hi 0
    u32 e[4];
#pragma unroll
    for (int i = 0; i < 4; i++) e[i] = __shfl_sync(0xffffffffu, mine, (q0 & 7) + i, 8);
    if constexpr (BITS == 8) {
      uint4 x[4];
#pragma unroll
      for (int i = 0; i < 4; i++) x[i] = __ldg(reinterpret_cast<const uint4*>(h.Et) + base + (e[i] >> 8));
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const u32 d = (u32)(int)(int8_t)(e[i] & 255);
        const u32 xs[4] = {x[i].x, x[i].y, x[i].z, x[i].w};
#pragma unroll
        for (int w = 0; w < 4; w++) {
          acc[2 * w] += d * __byte_perm(xs[w], 0, 0x4140);
          acc[2 * w + 1] += d * __byte_perm(xs[w], 0, 0x4342);
        }
      }
    } else {
      u32 x[4];
#pragma unroll
      for (int i = 0; i < 4; i++) x[i] = __ldg(reinterpret_cast<const u32*>(h.Et) + base + (e[i] >> 8));
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const u32 d = (u32)(int)(int8_t)(e[i] & 255);
#pragma unroll
        for (int s = 0; s < 4; s++) acc[s] += d * ((x[i] >> (2 * s)) & 0x03030303u);
      }
    }
  }
#pragma unroll
  for (int w = 0; w < 4; w++) {
    if constexpr (BITS == 8) {
      v[w].x += ((int)(acc[2 * w] & 0xffff) - c.y) * 256;
      v[w].y += ((int)(acc[2 * w] >> 16) - c.y) * 256;
      v[w].z += ((int)(acc[2 * w + 1] & 0xffff) - c.y) * 256;
      v[w].w += ((int)(acc[2 * w + 1] >> 16) - c.y) * 256;
    } else {
      v[w].x += ((int)(acc[w] & 0xff) - c.y) * 256;
      v[w].y += ((int)((acc[w] >> 8) & 0xff) - c.y) * 256;
      v[w].z += ((int)((acc[w] >> 16) & 0xff) - c.y) * 256;
      v[w].w += ((int)(acc[w] >> 24) - c.y) * 256;
    }
  }
}

// The deep-batched corrections (CORR 5: int8 E, 8 nonzeros a batch; CORR 6: 2-bit forms, 16): as correct16, but a
// batch's E^T loads are all in flight before any is used, and the tile t (shared or global; null: none) is read and
// written only after the last batch. Batches past the warp's longest column are skipped (maxc is warp-uniform).
template <int BITS, int NB>
__device__ __forceinline__ void correct16b(int4* t, int j, int m0, int lane, const HiB& h) {
  const int l8 = lane & 7;
  const int beg = j >= 0 ? __ldg(h.colptr + j) : 0, cnt = j >= 0 ? __ldg(h.colptr + j + 1) - beg : 0;
  const int maxc = __reduce_max_sync(0xffffffffu, cnt);
  if (maxc == 0) return;
  const int2 c = j >= 0 ? __ldg(h.colc + j) : make_int2(0, 0);
  constexpr int W = BITS == 8 ? 8 : 4;
  u32 acc[W];
  const u32 b = BITS == 8 ? (u32)c.x * 0x10001u : (u32)c.x * 0x01010101u;
#pragma unroll
  for (int w = 0; w < W; w++) acc[w] = b;
  const size_t base = (size_t)(m0 >> 4) + l8;
#pragma unroll 1
  for (int q0 = 0; q0 < maxc; q0 += NB) {
    u32 mine[(NB + 7) / 8];
#pragma unroll
    for (int r = 0; r < (NB + 7) / 8; r++) {
      const int q = q0 + 8 * r + l8;
      mine[r] = q < cnt ? __ldg(h.ent + beg + q) : 0u;  // 0: offset 0, hi 0
    }
    u32 e[NB];
#pragma unroll
    for (int i = 0; i < NB; i++) e[i] = __shfl_sync(0xffffffffu, mine[i >> 3], i & 7, 8);
    if constexpr (BITS == 8) {
      uint4 x[NB];
#pragma unroll
      for (int i = 0; i < NB; i++)
        if (q0 + i < maxc) x[i] = __ldg(reinterpret_cast<const uint4*>(h.Et) + base + (e[i] >> 8));
#pragma unroll
      for (int i = 0; i < NB; i++)
        if (q0 + i < maxc) {
          const u32 d = (u32)(int)(int8_t)(e[i] & 255);
          const u32 xs[4] = {x[i].x, x[i].y, x[i].z, x[i].w};
#pragma unroll
          for (int w = 0; w < 4; w++) {
            acc[2 * w] += d * __byte_perm(xs[w], 0, 0x4140);
            acc[2 * w + 1] += d * __byte_perm(xs[w], 0, 0x4342);
          }
        }
    } else {
      u32 x[NB];
#pragma unroll
      for (int i = 0; i < NB; i++)
        if (q0 + i < maxc) x[i] = __ldg(reinterpret_cast<const u32*>(h.Et) + base + (e[i] >> 8));
#pragma unroll
      for (int i = 0; i < NB; i++)
        if (q0 + i < maxc) {
          const u32 d = (u32)(int)(int8_t)(e[i] & 255);
#pragma unroll
          for (int s = 0; s < 4; s++) acc[s] += d * ((x[i] >> (2 * s)) & 0x03030303u);
        }
    }
  }
  if (!t) return;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    int4 v = t[w];
    if constexpr (BITS == 8) {
      v.x += ((int)(acc[2 * w] & 0xffff) - c.y) * 256;
      v.y += ((int)(acc[2 * w] >> 16) - c.y) * 256;
      v.z += ((int)(acc[2 * w + 1] & 0xffff) - c.y) * 256;
      v.w += ((int)(acc[2 * w + 1] >> 16) - c.y) * 256;
    } else {
      v.x += ((int)(acc[w] & 0xff) - c.y) * 256;
      v.y += ((int)((acc[w] >> 8) & 0xff) - c.y) * 256;
      v.z += ((int)((acc[w] >> 16) & 0xff) - c.y) * 256;
      v.w += ((int)(acc[w] >> 24) - c.y) * 256;
    }
    t[w] = v;
  }
}

template <int CORR>
__global__ void __launch_bounds__(THREADS, 2) leaf_kernel(const int8_t* __restrict__ E, const int8_t* __restrict__ L, int m,
                                                          int K, int* __restrict__ Mt, Hi h, HiB hb) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const u32 base = smem_u32(smem);
  int acc[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int b = 0; b < 4; b++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[a][b][c] = 0;
  const int KT = K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) load_stage(base + s * 2 * TILE_BYTES, base + s * 2 * TILE_BYTES + TILE_BYTES, E, L, m0, n0, s * BK, K, tid);
    cp_commit();
  }
  const int a_row = wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8, a_ch = lane >> 4;
  const int b_row = wn * 32 + (lane & 7) + (lane >> 4) * 8, b_ch = (lane >> 3) & 1;
#pragma unroll 1
  for (int kt = 0; kt < KT; kt++) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int nk = kt + STAGES - 1;
    if (nk < KT) {
      const int s = nk % STAGES;
      load_stage(base + s * 2 * TILE_BYTES, base + s * 2 * TILE_BYTES + TILE_BYTES, E, L, m0, n0, nk * BK, K, tid);
    }
    cp_commit();
    const u32 sa = base + (kt % STAGES) * 2 * TILE_BYTES, sb = sa + TILE_BYTES;
#pragma unroll
    for (int s = 0; s < 2; s++) {
      u32 a[4][4], b[4][2];
#pragma unroll
      for (int mt = 0; mt < 4; mt++) ldsm4(sa + swz(a_row + mt * 16, a_ch + 2 * s), a[mt][0], a[mt][1], a[mt][2], a[mt][3]);
#pragma unroll
      for (int p = 0; p < 2; p++)
        ldsm4(sb + swz(b_row + p * 16, b_ch + 2 * s), b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1]);
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++) mma(acc[mt][nt], a[mt], b[nt][0], b[nt][1]);
    }
  }
  cp_wait<0>();
  __syncthreads();
  int* T = reinterpret_cast<int*>(smem);
  const int g = lane >> 2, q = lane & 3;
#pragma unroll
  for (int half = 0; half < 2; half++) {
    if ((wn >> 1) == half) {
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          const int r = wm * 64 + mt * 16 + g, c = (wn & 1) * 32 + nt * 8 + 2 * q;
          T[c * TP + r] = acc[mt][nt][0];
          T[(c + 1) * TP + r] = acc[mt][nt][1];
          T[c * TP + r + 8] = acc[mt][nt][2];
          T[(c + 1) * TP + r + 8] = acc[mt][nt][3];
        }
    }
    __syncthreads();
    if constexpr (CORR >= 5) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
      constexpr int NB2 = 12;  // sm_90a spills at 16 under the 128-register bound
#else
      constexpr int NB2 = 16;
#endif
#pragma unroll 1
      for (int round = 0; round < 2; round++) {
        const int c = warp * 8 + round * 4 + (lane >> 3), j = n0 + half * 64 + c;
        correct16b<CORR == 5 ? 8 : 2, CORR == 5 ? 8 : NB2>(reinterpret_cast<int4*>(&T[c * TP + 16 * (lane & 7)]), j,
                                                           m0, lane, hb);
      }
      __syncwarp();
    } else if constexpr (CORR >= 3) {
      // the warp's own 8 columns, 4 at a time, corrected in T (a lane 16 rows) before they are stored
#pragma unroll 1
      for (int round = 0; round < 2; round++) {
        const int c = warp * 8 + round * 4 + (lane >> 3), j = n0 + half * 64 + c;
        int4 v[4];
        int4* t = reinterpret_cast<int4*>(&T[c * TP + 16 * (lane & 7)]);
#pragma unroll
        for (int w = 0; w < 4; w++) v[w] = t[w];
        correct16<CORR == 3 ? 8 : 2>(v, j, m0, lane, hb);
#pragma unroll
        for (int w = 0; w < 4; w++) t[w] = v[w];
      }
      __syncwarp();
    }
#pragma unroll 1
    for (int cc = 0; cc < 8; cc++) {
      const int c = warp * 8 + cc, j = n0 + half * 64 + c;
      int4 v = *reinterpret_cast<const int4*>(&T[c * TP + 4 * lane]);
      if constexpr (CORR == 1 || CORR == 2) correct<CORR>(v, j, m0, lane, h);
      *reinterpret_cast<int4*>(&Mt[(size_t)j * m + m0 + 4 * lane]) = v;
    }
    __syncthreads();
  }
}

// the correction alone: block (row tile, 8 overflowing columns), a warp a column
template <int CORR>
__global__ void __launch_bounds__(THREADS) corr_kernel(int* __restrict__ Mt, int m, const int* __restrict__ cols, int ncols,
                                                       Hi h) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int m0 = blockIdx.x * BM, ci = blockIdx.y * 8 + warp;
  if (ci >= ncols) return;
  const int j = __ldg(cols + ci);
  int4* p = reinterpret_cast<int4*>(&Mt[(size_t)j * m + m0 + 4 * lane]);
  int4 v = *p;
  correct<CORR>(v, j, m0, lane, h);
  *p = v;
}

// the batched correction alone: block (row tile, 32 overflowing columns), a warp 4 columns, a lane 16 rows
template <int BITS>
__global__ void __launch_bounds__(THREADS) corr16_kernel(int* __restrict__ Mt, int m, const int* __restrict__ cols, int ncols,
                                                         HiB h) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int m0 = blockIdx.x * BM, ci = blockIdx.y * 32 + warp * 4 + (lane >> 3);
  const int j = ci < ncols ? __ldg(cols + ci) : -1;
  int4 v[4];
  int4* p = j >= 0 ? reinterpret_cast<int4*>(&Mt[(size_t)j * m + m0 + 16 * (lane & 7)]) : nullptr;
#pragma unroll
  for (int w = 0; w < 4; w++) v[w] = p ? p[w] : make_int4(0, 0, 0, 0);
  correct16<BITS>(v, j, m0, lane, h);
  if (p)
#pragma unroll
    for (int w = 0; w < 4; w++) p[w] = v[w];
}

template <int BITS, int NB>
__global__ void __launch_bounds__(THREADS) corr16b_kernel(int* __restrict__ Mt, int m, const int* __restrict__ cols,
                                                          int ncols, HiB h) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int m0 = blockIdx.x * BM, ci = blockIdx.y * 32 + warp * 4 + (lane >> 3);
  const int j = ci < ncols ? __ldg(cols + ci) : -1;
  correct16b<BITS, NB>(j >= 0 ? reinterpret_cast<int4*>(&Mt[(size_t)j * m + m0 + 16 * (lane & 7)]) : nullptr, j, m0,
                       lane, h);
}

__global__ void empty_kernel(int* out) {
  if (out && threadIdx.x == 1u << 30) out[0] = 0;
}

__global__ void __launch_bounds__(THREADS) peak_kernel(int iters, u32 seed, int* __restrict__ out) {
  const u32 t = seed + threadIdx.x;
  const u32 a[4] = {t * 0x9E3779B1u, t * 0x85EBCA77u, t * 0xC2B2AE3Du, t * 0x27D4EB2Fu};
  const u32 b0 = t * 0x165667B1u, b1 = t * 0xD3A2646Cu;
  int acc[PEAK_CHAINS][4];
#pragma unroll
  for (int c = 0; c < PEAK_CHAINS; c++) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0;
#pragma unroll 1
  for (int i = 0; i < iters; i++)
#pragma unroll
    for (int c = 0; c < PEAK_CHAINS; c++) mma(acc[c], a, b0, b1);
  int s = 0;
#pragma unroll
  for (int c = 0; c < PEAK_CHAINS; c++) s += acc[c][0] ^ acc[c][1] ^ acc[c][2] ^ acc[c][3];
  out[blockIdx.x * THREADS + threadIdx.x] = s;
}

// ---------------------------------------------------------------- host

template <class T>
static std::vector<T> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    fprintf(stderr, "cannot read %s\n", path.c_str());
    exit(2);
  }
  const size_t n = (size_t)f.tellg();
  std::vector<T> v(n / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)n);
  return v;
}

template <class T>
static T* dput(const std::vector<T>& h) {
  T* d = nullptr;
  GK_CK(cudaMalloc(&d, h.size() * sizeof(T)));
  GK_CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

static double median_of(const gk::Timing& t, int arm, bool mhz) {
  std::vector<double> v;
  for (const gk::Item& it : t.items)
    if (it.arm == arm) v.push_back(mhz ? (double)it.sm_mhz() : it.ms / it.calls);
  return gk::median(v);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: leaf_corr FIXTURE_DIR OUT_DIR [--m M] [--reps R] [--peak MAC_PER_SM_CLK] [--corrs 1,..,6]\n");
    return 2;
  }
  const std::string fx = argv[1], out = argv[2];
  int m_arg = 0, reps = 15;
  double peak_override = 0;
  std::vector<int> CS = {1, 2, 3, 4, 5, 6};
  for (int i = 3; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--m")) m_arg = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--peak")) peak_override = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--corrs")) {
      CS.clear();
      for (const char* p = argv[i + 1]; *p; p++)
        if (*p >= '1' && *p <= '6') CS.push_back(*p - '0');
    } else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  if (CS.empty()) {
    fprintf(stderr, "--corrs names no correction\n");
    return 2;
  }
  std::vector<int8_t> lo = read_bin<int8_t>(fx + "/lo.bin");
  std::vector<int> colptr = read_bin<int>(fx + "/colptr.bin"), ent = read_bin<int>(fx + "/ent.bin");
  const int n = (int)colptr.size() - 1, K = (int)(lo.size() / n);
  const long long nnz = (long long)ent.size();
  if ((size_t)n * K != lo.size() || n % BN || K % BK || colptr[n] != nnz) {
    fprintf(stderr, "bad fixture: n %d K %d nnz %lld\n", n, K, nnz);
    return 2;
  }
  std::vector<int> cols;
  long long max_abs = 0;
  for (int j = 0; j < n; j++) {
    long long s = 0;
    for (int p = colptr[j]; p < colptr[j + 1]; p++) s += std::abs((int)(int8_t)(ent[p] & 255));
    max_abs = std::max(max_abs, s);
    if (colptr[j + 1] > colptr[j]) cols.push_back(j);
  }
  if (max_abs * 255 >= 65536 || max_abs * 3 >= 256) {
    fprintf(stderr, "a column's sum of |hi| (%lld) overflows pack's 16-bit or bits2's 8-bit lanes\n", max_abs);
    return 2;
  }

  int dev = 0;
  GK_CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  GK_CK(cudaGetDeviceProperties(&prop, dev));
  const int sms = prop.multiProcessorCount;
  GK_CK(cudaFuncSetAttribute(leaf_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  GK_CK(cudaFuncSetAttribute(leaf_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  GK_CK(cudaFuncSetAttribute(leaf_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  GK_CK(cudaFuncSetAttribute(leaf_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  GK_CK(cudaFuncSetAttribute(leaf_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  GK_CK(cudaFuncSetAttribute(leaf_kernel<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  GK_CK(cudaFuncSetAttribute(leaf_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  int occ = 0;
  GK_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, leaf_kernel<1>, THREADS, SMEM));
  // m: two full waves of 128 × 128 tiles over the n / 128 column tiles
  const int m = m_arg ? m_arg : BM * std::max(1, (2 * sms * occ + n / BN - 1) / (n / BN));
  if (m % BM) {
    fprintf(stderr, "m %d is not a multiple of %d\n", m, BM);
    return 2;
  }

  // E: D-OUT's level-1 noise forms (sums and differences of two 1-bit quadrants), uniform over {−1, 0, 1, 2}
  std::vector<int8_t> E((size_t)m * K), Et((size_t)K * m), Etb((size_t)K * m);
  std::vector<u32> Et2((size_t)K * (m / 16), 0u);
  uint64_t st = 0x2545F4914F6CDD1Dull;
  for (auto& x : E) {
    st ^= st << 13, st ^= st >> 7, st ^= st << 17;
    x = (int8_t)((int)((st >> 24) & 3) - 1);
  }
  for (int i = 0; i < m; i++)
    for (int k = 0; k < K; k++) {
      const int8_t e = E[(size_t)i * K + k];
      Et[(size_t)k * m + i] = e;
      Etb[(size_t)k * m + i] = (int8_t)(e ^ 0x80);
      const int r = i & 15;
      Et2[(size_t)k * (m / 16) + i / 16] |= (u32)(e + 1) << (8 * (r & 3) + 2 * (r >> 2));
    }
  int8_t *dE = dput(E), *dL = dput(lo), *dEt = dput(Et), *dEtb = dput(Etb);
  u32* dEt2 = dput(Et2);
  if ((long long)K * (m / 16) >= (1LL << 23)) {
    fprintf(stderr, "K · m / 16 overflows the batched entries' 23-bit offsets\n");
    return 2;
  }
  std::vector<u32> ent3(ent.size());
  for (size_t p = 0; p < ent.size(); p++) ent3[p] = ((u32)(ent[p] >> 8) * (u32)(m / 16)) << 8 | (u32)(ent[p] & 255);
  std::vector<int2> colc2(n);
  for (int j = 0; j < n; j++) {
    int neg = 0, sum = 0;
    for (int p = colptr[j]; p < colptr[j + 1]; p++) {
      const int d = (int8_t)(ent[p] & 255);
      sum += d;
      if (d < 0) neg -= d;
    }
    colc2[j] = make_int2(3 * neg, 3 * neg + sum);
  }
  u32* dent3 = dput(ent3);
  int2* dcolc2 = dput(colc2);
  std::vector<longlong2> ent2(ent.size());
  for (size_t p = 0; p < ent.size(); p++)
    ent2[p] = make_longlong2((long long)(ent[p] >> 8) * m, (long long)(int8_t)(ent[p] & 255));
  std::vector<int2> colc(n);
  for (int j = 0; j < n; j++) {
    int neg = 0, sum = 0;
    for (int p = colptr[j]; p < colptr[j + 1]; p++) {
      const int d = (int8_t)(ent[p] & 255);
      sum += d;
      if (d < 0) neg -= d;
    }
    colc[j] = make_int2(255 * neg, 255 * neg + 128 * sum);
  }
  int *dcolptr = dput(colptr), *dcols = dput(cols);
  longlong2* dent = dput(ent2);
  int2* dcolc = dput(colc);
  int *dM = nullptr, *dS = nullptr, *dpeak = nullptr;
  const size_t mbytes = (size_t)n * m * 4;
  GK_CK(cudaMalloc(&dM, mbytes));
  GK_CK(cudaMalloc(&dS, mbytes));
  const Hi h1{reinterpret_cast<const u32*>(dEt), dcolptr, dent, dcolc},
      h2{reinterpret_cast<const u32*>(dEtb), dcolptr, dent, dcolc};
  const HiB hb8{dEtb, dcolptr, dent3, dcolc}, hb2{dEt2, dcolptr, dent3, dcolc2};
  cudaStream_t s;
  GK_CK(cudaStreamCreate(&s));
  const dim3 lgrid(n / BN, m / BM), cgrid(m / BM, (unsigned)((cols.size() + 7) / 8)),
      cgrid16(m / BM, (unsigned)((cols.size() + 31) / 32));
  const int peak_blocks = sms * 4, peak_iters = 4096;
  GK_CK(cudaMalloc(&dpeak, (size_t)peak_blocks * THREADS * 4));

  // the CPU's reference on 64 rows (all residues mod 128 that 131·t reaches), every column
  const int NR = 64;
  std::vector<int> rows(NR);
  for (int t = 0; t < NR; t++) rows[t] = (int)((131LL * t + 7) % m);
  std::vector<long long> ref_lo((size_t)NR * n), ref_full((size_t)NR * n);
  for (int t = 0; t < NR; t++) {
    const int8_t* e = &E[(size_t)rows[t] * K];
    for (int j = 0; j < n; j++) {
      const int8_t* l = &lo[(size_t)j * K];
      long long a = 0;
      for (int k = 0; k < K; k++) a += (int)e[k] * (int)l[k];
      long long c = 0;
      for (int p = colptr[j]; p < colptr[j + 1]; p++) c += (long long)(int8_t)(ent[p] & 255) * e[ent[p] >> 8];
      ref_lo[(size_t)t * n + j] = a;
      ref_full[(size_t)t * n + j] = a + 256 * c;
    }
  }
  auto matches = [&](const std::vector<long long>& ref) {
    return [&, ref_p = &ref](const void* hptr, size_t) {
      const int* Mh = static_cast<const int*>(hptr);
      for (int t = 0; t < NR; t++)
        for (int j = 0; j < n; j++)
          if ((long long)Mh[(size_t)j * m + rows[t]] != (*ref_p)[(size_t)t * n + j]) return false;
      return true;
    };
  };
  auto leaf = [&](int corr, int* M, cudaStream_t st_) {
    if (corr == 0) leaf_kernel<0><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h1, hb8);
    if (corr == 1) leaf_kernel<1><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h1, hb8);
    if (corr == 2) leaf_kernel<2><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h2, hb8);
    if (corr == 3) leaf_kernel<3><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h2, hb8);
    if (corr == 4) leaf_kernel<4><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h2, hb2);
    if (corr == 5) leaf_kernel<5><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h2, hb8);
    if (corr == 6) leaf_kernel<6><<<lgrid, THREADS, SMEM, st_>>>(dE, dL, m, K, M, h2, hb2);
  };
  auto corr = [&](int c, int* M, cudaStream_t st_) {
    if (c == 1) corr_kernel<1><<<cgrid, THREADS, 0, st_>>>(M, m, dcols, (int)cols.size(), h1);
    if (c == 2) corr_kernel<2><<<cgrid, THREADS, 0, st_>>>(M, m, dcols, (int)cols.size(), h2);
    if (c == 3) corr16_kernel<8><<<cgrid16, THREADS, 0, st_>>>(M, m, dcols, (int)cols.size(), hb8);
    if (c == 4) corr16_kernel<2><<<cgrid16, THREADS, 0, st_>>>(M, m, dcols, (int)cols.size(), hb2);
    if (c == 5) corr16b_kernel<8, 8><<<cgrid16, THREADS, 0, st_>>>(M, m, dcols, (int)cols.size(), hb8);
    if (c == 6) corr16b_kernel<2, 16><<<cgrid16, THREADS, 0, st_>>>(M, m, dcols, (int)cols.size(), hb2);
  };
  const char* names[7] = {"", "imad", "pack", "pack16", "bits2", "pack16x8", "bits2x16"};

  bool ok = true;
  struct GateRow { std::string name; gk::PoisonGate g; };
  std::vector<GateRow> gates;
  gates.push_back({"leaf", gk::poison_gate(dM, mbytes, s, [&](cudaStream_t st_) { leaf(0, dM, st_); }, matches(ref_lo))});
  for (int c : CS) {
    gates.push_back({std::string("leaf+") + names[c],
                     gk::poison_gate(dM, mbytes, s, [&, c](cudaStream_t st_) { leaf(c, dM, st_); }, matches(ref_full))});
    // the correction applied to the uncorrected leaf's output; the control writes the leaf's output only
    auto arm = [&, c](cudaStream_t st_) { leaf(0, dM, st_); corr(c, dM, st_); };
    auto ctl = [&](cudaStream_t st_) { leaf(0, dM, st_); };
    gates.push_back({std::string("corr+") + names[c], gk::poison_gate(dM, mbytes, s, arm, matches(ref_full), ctl)});
  }
  for (auto& r : gates) {
    printf("{\"gate\": \"%s\", \"accepted\": %s, \"control_rejected\": %s, \"poison_left\": %zu}\n", r.name.c_str(),
           r.g.accepted ? "true" : "false", r.g.control_rejected ? "true" : "false", r.g.left);
    ok &= r.g.ok();
  }
  fflush(stdout);
  if (!ok) {
    fprintf(stderr, "gate failed: no timing\n");
    return 3;
  }

  GK_CK(cudaMemcpy(dS, dM, mbytes, cudaMemcpyDeviceToDevice));
  gk::Nvml nv(dev);
  std::vector<gk::Arm> arms = {{"leaf", [&](cudaStream_t st_, int k) { for (int i = 0; i < k; i++) leaf(0, dM, st_); }}};
  for (int c : CS)
    arms.push_back({std::string("leaf+") + names[c],
                    [&, c](cudaStream_t st_, int k) { for (int i = 0; i < k; i++) leaf(c, dM, st_); }});
  for (int c : CS)
    arms.push_back({std::string("corr+") + names[c],
                    [&, c](cudaStream_t st_, int k) { for (int i = 0; i < k; i++) corr(c, dS, st_); }});
  arms.push_back({"empty", [&](cudaStream_t st_, int k) {
                    for (int i = 0; i < k; i++) empty_kernel<<<cgrid, THREADS, 0, st_>>>(nullptr);
                  }});
  arms.push_back({"empty16", [&](cudaStream_t st_, int k) {
                    for (int i = 0; i < k; i++) empty_kernel<<<cgrid16, THREADS, 0, st_>>>(nullptr);
                  }});
  arms.push_back({"peak", [&](cudaStream_t st_, int k) {
                    for (int i = 0; i < k; i++) peak_kernel<<<peak_blocks, THREADS, 0, st_>>>(peak_iters, 1u + i, dpeak);
                  }});
  auto arm_index = [&](const std::string& name) {
    for (size_t a = 0; a < arms.size(); a++)
      if (arms[a].name == name) return (int)a;
    fprintf(stderr, "no arm %s\n", name.c_str());
    exit(2);
  };
  gk::Timing t = gk::timed_items(arms, reps, s, nv);
  GK_CK(cudaGetLastError());
  gk::summary(t, arms);
  {
    FILE* f = fopen((out + "/items.jsonl").c_str(), "w");
    if (f) {
      gk::items_jsonl(t, arms, f);
      fclose(f);
    }
  }
  // per arm: median ms and SM clock; cycles = ms · MHz · 1e3 (per SM)
  std::vector<double> ms(arms.size()), mhz(arms.size()), cyc(arms.size());
  for (size_t a = 0; a < arms.size(); a++) {
    ms[a] = median_of(t, (int)a, false);
    mhz[a] = median_of(t, (int)a, true);
    cyc[a] = ms[a] * mhz[a] * 1e3;
  }
  const double peak_macs = (double)peak_blocks * (THREADS / 32) * peak_iters * PEAK_CHAINS * 16 * 8 * 32;
  const int P = arm_index("peak"), L0 = arm_index("leaf");
  const double peak_measured = peak_macs / (cyc[P] * sms);
  const double peak = peak_override > 0 ? peak_override : peak_measured;  // MAC per SM-clock: W1 per SM-clock
  const double leaf_macs = (double)m * n * K, pairs = (double)m * nnz;
  auto writes = [&](double cycles) { return cycles * sms * peak / (32.0 * pairs); };
  // the same by time alone, each arm at its own clock, against the peak arm's MAC a ms (no NVML needed)
  auto writes_t = [&](double dms) { return dms * (peak_macs / ms[P]) / (32.0 * pairs); };
  printf(
      "{\"result\": \"leaf_corr\", \"gpu\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"occupancy\": %d, \"m\": %d, \"n\": %d, "
      "\"K\": %d, \"nnz\": %lld, \"cols\": %zu, \"pairs\": %.0f, \"steady\": %s, \"peak_mac_per_sm_clk_measured\": %.2f, "
      "\"peak_mac_per_sm_clk\": %.2f",
      prop.name, prop.major, prop.minor, sms, occ, m, n, K, nnz, cols.size(), pairs, t.steady ? "true" : "false",
      peak_measured, peak);
  for (size_t a = 0; a < arms.size(); a++)
    printf(", \"%s\": {\"ms\": %.6f, \"sm_mhz\": %.0f, \"cycles\": %.1f}", arms[a].name.c_str(), ms[a], mhz[a], cyc[a]);
  printf(", \"leaf_efficiency\": %.4f, \"leaf_efficiency_by_time\": %.4f", leaf_macs / (cyc[L0] * sms * peak),
         leaf_macs / (ms[L0] * peak_macs / ms[P]));
  // writes a (row, nonzero): a leaf+ arm's marginal over the leaf, a corr+ arm alone and less its grid's empty launch
  for (int by_time = 0; by_time < 2; by_time++) {
    auto price = [&](int a, int b) {
      return by_time ? writes_t(ms[a] - (b >= 0 ? ms[b] : 0)) : writes(cyc[a] - (b >= 0 ? cyc[b] : 0));
    };
    printf(", \"%s\": {", by_time ? "writes_by_time" : "writes");
    for (int c : CS) {
      const int lc = arm_index(std::string("leaf+") + names[c]), cc = arm_index(std::string("corr+") + names[c]);
      const int e = arm_index(c <= 2 ? "empty" : "empty16");
      printf("%s\"leaf+%s_marginal\": %.4f, \"corr+%s\": %.4f, \"corr+%s_less_launch\": %.4f", c != CS[0] ? ", " : "",
             names[c], price(lc, L0), names[c], price(cc, -1), names[c], price(cc, e));
    }
    printf("}");
  }
  printf("}\n");
  return 0;
}
