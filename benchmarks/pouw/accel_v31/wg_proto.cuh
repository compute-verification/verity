// ACCEL's protocol GEMM (v31.cu's gemm_kernel<true>) on the H200's warpgroup MMAs: wgmma.linear's mainloop
// (benchmarks/torch_gpu/s8_linear/s8_wgmma.cu: persistent, warp-specialized, TMA with SWIZZLE_128B, wgmma m64nNk32
// .s32.s8.s8, 128 x BN tiles of two consumer warpgroups) with the receipt's values, bit for bit gemm_kernel<true>'s:
//   T_s     after the main k-tile that ends segment s, each cell's low 16 bits of X W'^T over the first s L inner indices,
//           to the words buffer in v31's layout ([128 x 128 tile][s][cell], u16)
//   O       T_S + 256 dO - Z_X U^T - V Y_W^T mod 2^32, row-major s32
// The peel's tail rows P_A, P_B (form_a_kernel's and form_b_kernel's, blocks of L') go through the same accumulator, ahead
// of the main k-tiles, in Horner order: the blocks of the top digit D first (from zero), then acc *= 128 and digit D - 1's
// blocks, ..., acc *= -128 after digit 1, then the shift-0 blocks (-Z_0 U^T and -V Y_0^T), so acc holds the tail
// tau = -Z_X U^T - V Y_W^T when the main k-tiles start and tau + T_S after them. Each segment's word is the low half of
// acc - tau: tau's low halves are kept (CM 0: packed in 64 registers a thread at BN 256; CM 1: in an L2-resident scratch
// slot of the CTA's, read back at each segment end), so a tail costs (dz + dy) L' inner indices on the tensor cores and
// no second accumulator.
//   mode 0  the plain product X W'^T to O (the bench's `plain` arm)
//   mode 1  the protocol's GEMM (diag 1: no segment stores, 2: no tail, 4: no dO read; timing only)
//   mode 2  wgmma.linear itself on A, B: bf16(RN(RN(f32(acc) s_a) s_w)) words, the `wgl` arm, gamma's baseline
// Integer wgmma accumulates mod 2^32 (no .satfinite), as the receipt's O is taken.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "../../torch_gpu/s8_linear/s8_acc.h"

namespace wgp {

#define WGD __device__ __forceinline__

constexpr int BK = 128, TM = 128, SMEM_BUDGET = 220 * 1024, MAX_STAGES = 8, TW = 128;   // TW: v31's word tile

template <int BN> struct Tile {
  static_assert(BN == 128 || BN == 256, "128 x 128 or 128 x 256 tiles");
  static constexpr int NWG = 2, NT = 128 * (NWG + 1), NA = BN / 2;
  static constexpr uint32_t A_BYTES = TM * BK, STAGE = (TM + BN) * BK;
  static constexpr int STAGES = SMEM_BUDGET / STAGE < MAX_STAGES ? SMEM_BUDGET / STAGE : MAX_STAGES;
  static constexpr int SMEM = 1024 + STAGES * STAGE + 16 * STAGES;
};

struct alignas(128) Args {
  CUtensorMap a, b, pa, pb;   // X or A (boxes of 128 rows), W' or B (BN rows), P_A (128 rows), P_B (BN rows)
  uint16_t* words;
  const int* dO;
  int* O;                     // mode 0, 1: s32 cells; mode 2: bf16 words
  uint32_t* cbuf;             // CM 1: a slot of TM x BN / 2 words a CTA
  const float *sa, *sw;       // mode 2's scales
  int M, N, K, L, S, lp, dz, dy, TT, group, mode, diag;
};

WGD uint32_t smem_at(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
WGD void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
WGD void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
WGD void mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }
WGD void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WGP_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WGP_WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
WGD void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
WGD void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
WGD void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
template <int R> WGD void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> WGD void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
#define WGP_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WGP_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WGP_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

WGD uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
// d = A B + (sd ? d : 0), one k32 atom of a 64 x N tile: accumulator i of thread x is row 16 (x / 32) + (x % 32) / 4 +
// 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
template <int N> WGD void imma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  if constexpr (N == 256)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
                 ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 " ACC64_S
                 ", %64, %65, p;\n}" : ACC64_R : "l"(da), "l"(db), "r"(sd));
}
template <int NA> WGD void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}

// the tail's k-tiles in Horner order: digit e of block b (0 for blocks 0 and 1, then Z's digits, then Y's), the top digit
// D = max(dz, dy) - 1 first; step i < TT is k-tile kt of P's columns, and `end` says the step closes its digit's group
WGD int digit_of(int b, int dz) { return b < 2 ? 0 : (b <= dz ? b - 1 : b - dz); }
struct TailStep { int kt; int e; bool end; };
WGD TailStep tail_step(int i, int lp, int dz, int dy) {
  const int kpb = lp / BK, nb = dz + dy, D = (dz > dy ? dz : dy) - 1;
  for (int e = D; e >= 0; --e) {
    int n = 0;
    for (int b = 0; b < nb; ++b) n += digit_of(b, dz) == e;
    const int len = n * kpb;
    if (i < len) {
      int seen = 0;
      for (int b = 0; b < nb; ++b)
        if (digit_of(b, dz) == e) {
          if (i < seen + kpb) return TailStep{b * kpb + (i - seen), e, i == len - 1};
          seen += kpb;
        }
    }
    i -= len;
  }
  return TailStep{0, 0, false};
}

WGD uint32_t bf16_pair(int32_t a, int32_t b, float sr, float c0, float c1) {
  const float ya = __fmul_rn(__fmul_rn(__int2float_rn(a), sr), c0), yb = __fmul_rn(__fmul_rn(__int2float_rn(b), sr), c1);
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(ya)) | (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(yb)) << 16;
}

WGD void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

template <int BN, int CM> __device__ void body(const Args& p) {
  using T = Tile<BN>;
  constexpr int NWG = T::NWG, STAGES = T::STAGES, NA = T::NA;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t base = (smem_at(smem) + 1023) & ~1023u, bars = base + STAGES * T::STAGE;
  const int M = p.M, N = p.N, KT = p.K / BK, tiles_m = M / TM, tiles_n = N / BN, tiles = tiles_m * tiles_n;
  const bool proto = p.mode == 1;
  const int TT = proto ? p.TT : 0, steps = TT + KT;
  const int tid = threadIdx.x, wg = tid / 128;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (STAGES + s), NWG * 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (wg == NWG) {
    regs_dec<40>();
    if (tid == NWG * 128) {
      tma_prefetch(&p.a);
      tma_prefetch(&p.b);
      if (TT) { tma_prefetch(&p.pa); tma_prefetch(&p.pb); }
      int s = 0;
      uint32_t ph = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        int tm, tn;
        raster(t, tiles_m, tiles_n, p.group, tm, tn);
        for (int i = 0; i < steps; i++) {
          const uint32_t st = base + s * T::STAGE, full = bars + 8 * s;
          mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          mbar_expect_tx(full, T::STAGE);
          if (i < TT) {
            const int kt = tail_step(i, p.lp, p.dz, p.dy).kt;
            tma_load(st, &p.pa, full, kt * BK, tm * TM);
            tma_load(st + T::A_BYTES, &p.pb, full, kt * BK, tn * BN);
          } else {
            tma_load(st, &p.a, full, (i - TT) * BK, tm * TM);
            tma_load(st + T::A_BYTES, &p.b, full, (i - TT) * BK, tn * BN);
          }
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
      }
    }
    asm volatile("exit;");
  }
  regs_inc<232>();

  const int x = tid % 128, lane = tid % 32;
  const int rloc = wg * 64 + 16 * (x / 32) + (x % 32) / 4, cq = 2 * (x % 4);   // the thread's first row and column in the tile
  int s = 0;
  uint32_t ph = 0;
  int32_t d[NA];
  uint32_t cp[CM == 0 ? NA / 2 : 1];
  uint32_t* cslot = CM == 1 ? p.cbuf + (size_t)blockIdx.x * (TM * BN / 2) : nullptr;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    int tm, tn;
    raster(t, tiles_m, tiles_n, p.group, tm, tn);
    int pend = -1;   // the stage whose release waits on its MMAs
    for (int i = 0; i < steps; i++) {
      mbar_wait_warp(bars + 8 * s, ph);
      const uint32_t a0 = base + s * T::STAGE + wg * 64 * BK, b0 = base + s * T::STAGE + T::A_BYTES;
      WGP_FENCE;
      imma<BN>(d, desc_sw128(a0), desc_sw128(b0), i > 0);
      imma<BN>(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
      imma<BN>(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
      imma<BN>(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
      WGP_COMMIT;
      WGP_WAIT(1);
      if (pend >= 0 && lane == 0) mbar_arrive(bars + 8 * (STAGES + pend));
      pend = s;
      if (++s == STAGES) { s = 0; ph ^= 1; }
      // the events that read the accumulators: a tail digit's end, and a segment's end
      int scale = 0;
      bool capture = false;
      int seg = 0;
      if (i < TT) {
        const TailStep ts = tail_step(i, p.lp, p.dz, p.dy);
        if (ts.end) {
          if (ts.e > 1) scale = 128;
          else if (ts.e == 1) scale = -128;
          else capture = true;
        }
      } else if (proto && !(p.diag & 1)) {
        const int kt = i - TT;
        if (((kt + 1) * BK) % p.L == 0 || kt + 1 == KT) seg = ((kt + 1) * BK + p.L - 1) / p.L;
      }
      if (!scale && !capture && !seg) continue;
      WGP_WAIT(0);
      acc_fence<NA>(d);
      if (lane == 0) mbar_arrive(bars + 8 * (STAGES + pend));
      pend = -1;
      if (scale) {
#pragma unroll
        for (int j = 0; j < NA; j++) d[j] = (int32_t)((uint32_t)d[j] * (uint32_t)scale);
      } else if (capture) {
#pragma unroll
        for (int j = 0; j < NA / 2; j++) {
          const uint32_t w = __byte_perm((uint32_t)d[2 * j], (uint32_t)d[2 * j + 1], 0x5410);
          if constexpr (CM == 0) cp[j] = w;
          else cslot[(size_t)j * (NWG * 128) + tid] = w;
        }
      } else {
        // T_s: the low halves of acc - tau, a pair of columns a 32-bit store, into the 128 x 128 word tiles the tile covers
        const size_t S = (size_t)p.S;
#pragma unroll
        for (int j = 0; j < NA / 4; j++) {
          const int c = 8 * j + cq, sub = c / TW, cc = c % TW;
          const size_t wt = (size_t)tm * (N / TW) + (size_t)tn * (BN / TW) + sub;
          uint32_t* w = reinterpret_cast<uint32_t*>(p.words + (wt * S + (size_t)(seg - 1)) * (TW * TW));
#pragma unroll
          for (int h = 0; h < 2; h++) {
            uint32_t c2 = 0;
            if (TT) {
              if constexpr (CM == 0) c2 = cp[2 * j + h];
              else c2 = cslot[(size_t)(2 * j + h) * (NWG * 128) + tid];
            }
            const uint32_t lo = (uint32_t)d[4 * j + 2 * h] - c2, hi = (uint32_t)d[4 * j + 2 * h + 1] - (c2 >> 16);
            w[((rloc + 8 * h) * TW + cc) >> 1] = __byte_perm(lo, hi, 0x5410);
          }
        }
      }
    }
    if (pend >= 0) {
      WGP_WAIT(0);
      acc_fence<NA>(d);
      if (lane == 0) mbar_arrive(bars + 8 * (STAGES + pend));
    }

    const int row = tm * TM + rloc, col0 = tn * BN + cq;
    if (p.mode == 2) {
      const float r0 = p.sa[row], r1 = p.sa[row + 8];
      uint32_t* y0 = reinterpret_cast<uint32_t*>(p.O) + ((size_t)row * N) / 2;
      uint32_t* y1 = reinterpret_cast<uint32_t*>(p.O) + ((size_t)(row + 8) * N) / 2;
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        const float2 c = *reinterpret_cast<const float2*>(p.sw + col);
        y0[col / 2] = bf16_pair(d[4 * j], d[4 * j + 1], r0, c.x, c.y);
        y1[col / 2] = bf16_pair(d[4 * j + 2], d[4 * j + 3], r1, c.x, c.y);
      }
      continue;
    }
    const bool add = proto && !(p.diag & 4);
#pragma unroll
    for (int j = 0; j < NA / 4; j++) {
      const size_t q0 = (size_t)row * N + col0 + 8 * j, q1 = q0 + (size_t)8 * N;
      int2 lo = make_int2(d[4 * j], d[4 * j + 1]), hi = make_int2(d[4 * j + 2], d[4 * j + 3]);
      if (add) {
        const int2 e0 = __ldg(reinterpret_cast<const int2*>(p.dO + q0)), e1 = __ldg(reinterpret_cast<const int2*>(p.dO + q1));
        lo.x = (int)((uint32_t)lo.x + ((uint32_t)e0.x << 8)); lo.y = (int)((uint32_t)lo.y + ((uint32_t)e0.y << 8));
        hi.x = (int)((uint32_t)hi.x + ((uint32_t)e1.x << 8)); hi.y = (int)((uint32_t)hi.y + ((uint32_t)e1.y << 8));
      }
      *reinterpret_cast<int2*>(p.O + q0) = lo;
      *reinterpret_cast<int2*>(p.O + q1) = hi;
    }
  }
}

template <int BN, int CM>
__global__ void __launch_bounds__(Tile<BN>::NT, 1) gemm(const __grid_constant__ Args p) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  body<BN, CM>(p);
#endif
}

}  // namespace wgp
