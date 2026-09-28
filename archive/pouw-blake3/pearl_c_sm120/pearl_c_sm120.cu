// Pearl-C on the RTX PRO 6000 Blackwell (sm_120a): the kernels of `pearl-c-sm120-v1` (verity/core/protocols/pouw/schemes/
// pearl_c.py on pearl_c_device.SM120, hashing -h1 and -h2) and the plain FP8 GEMMs, for run.py and arm.py.  Every kernel takes one Args by value:
// 16 pointers p[] and 16 ints i[].  sm_120 has no wgmma, TMA multicast or tcgen05: the products are warp-level mma.sync
// (kind::f8f6f4 m16n8k32 E4M3 for every FP8 atom, m16n8k16 BF16 for the peel) fed by ldmatrix from shared memory that
// cp.async fills (TMA on the 128 x 128 GEMMs: mainloop_sm120.cuh), within 99 KB of shared memory per block.
//
// Layout (fixture.py's `rowmajor`): every operand is a row-major byte matrix with its depth contiguous, byte k of row r at
// r * depth + k.  So are A' (m x k) and B~ (n x k), the model's own FP8 layout; E and E' (rows x 32); the F lines (k x 32, line t
// is row t) and their transposes F^T (32 x k, and 128 x k with rows 32-127 zero for G); and the peel factors P_A (m x 128 bytes)
// and P_B (n x 128), 64 BF16 words a row.  In shared memory each 16-byte chunk c of row r sits at chunk c ^ ((r >> 1) & 3) of
// 64-byte rows (c ^ (r & 7) of 128-byte rows), so every ldmatrix phase reads eight distinct bank groups.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "../pearl_c/hash_h2.cuh"
#include "../pearl_c/hash_h3s.cuh"
#include "../pouw_hash/pouw_hash.cuh"
#include "mainloop_sm120.cuh"

// ---------------------------------------------------------------- formats (IEEE, nearest even, no FTZ)
DEV float bf2f(uint32_t b) { return __uint_as_float((b & 0xFFFFu) << 16); }
DEV uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
DEV uint16_t bmul(uint16_t a, uint16_t b) { return f2bf(__fmul_rn(bf2f(a), bf2f(b))); }
DEV uint16_t bdiv(uint16_t a, uint16_t b) { return f2bf(__fdiv_rn(bf2f(a), bf2f(b))); }
DEV uint8_t e4m3(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); }
DEV float e4m3f(uint8_t c) { return __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3))); }
DEV uint16_t e4m3bf(uint8_t c) { return f2bf(e4m3f(c)); }

DEV uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}

// ---------------------------------------------------------------- cp.async, ldmatrix, mma.sync
DEV uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa(s)), "l"(g) : "memory"); }
DEV void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEV void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
DEV void ldsm4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEV uint32_t ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

// d += a b: one E4M3 atom, k = 32 (A: rows g, g + 8 x k 4t.., 16 + 4t..; B: column g x k 4t.., 16 + 4t..; D: rows g, g + 8 x
// columns 2t, 2t + 1, with g = lane / 4, t = lane % 4)
DEV void qmma(float* d, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// d += a b: one BF16 step, k = 16, the same byte layout as the atom's
DEV void hmma(float* d, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------- the GEMM
// A (M x K) and B (N x K) row-major E4M3.  One CTA of 8 warps per TM x BN tile (gemm_body's: the 128 x 128 GEMMs are
// ml_body's, below): 64 x 64 (warp tile 16 x 32) for decode's few rows, where 128 x 128 tiles leave most SMs idle, or 128 x 32
// (16 x 32) for the 32-column words
// (A'F_B, B~F_A, G) at M a multiple of 128; persistent over tiles in groups of `i[3]` tile
// rows; a 4-deep cp.async pipeline of 64-byte stages (two atoms each).  G is the promotion period: every 32 G products (G / 2
// stages) from +0, promoted by one FADD into the FP32 total; G = 0 is one accumulator over all of K (no promotion).  Every
// output's chain is the same at either tile.  Modes:
//   PLAIN   y = bf16(total sa sb)
//   CT      stores C~ (FP32)
//   PEARLC  stores C~, then four k16 BF16 mma of P_A P_B^T into the total, stores U and y = bf16(U ia ib)
//   PEARLC_S  PEARLC without storing C~ or U (only y): the hash-free control for a fused epilogue's digests
//   PEARLC_B  PEARLC_S with the -h1 message digests of C~ and U from registers (ml_body only)
enum { M_PLAIN, M_CT, M_PEARLC, M_PEARLC_S, M_PEARLC_B };
constexpr int BK = 64, STAGES = 4, NT = 256, PEEL_ROW = 128;
template <int MODE, int TM, int BN> __host__ __device__ constexpr int gemm_smem() {
  return STAGES * (TM + BN) * BK + (MODE == M_PEARLC || MODE == M_PEARLC_S ? (TM + BN) * PEEL_ROW : 0);
}

DEV uint32_t swz64(int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }
DEV uint32_t swz128(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

// ROWS rows of 64 bytes at g (row stride ld) into s
template <int ROWS> DEV void stage_rows(uint8_t* s, const uint8_t* g, int ld, int tid) {
#pragma unroll
  for (int j = 0; j < (ROWS * 4 + NT - 1) / NT; j++) {
    const int idx = tid + j * NT, r = idx >> 2, c = idx & 3;
    if (ROWS * 4 % NT == 0 || idx < ROWS * 4) cp16(s + swz64(r, c), g + (size_t)r * ld + 16 * c);
  }
}
// ROWS peel rows of 128 bytes at g into s
template <int ROWS> DEV void peel_rows(uint8_t* s, const uint8_t* g, int tid) {
#pragma unroll
  for (int j = 0; j < ROWS * 8 / NT; j++) {
    const int idx = tid + j * NT, r = idx >> 3, c = idx & 7;
    cp16(s + swz128(r, c), g + (size_t)r * PEEL_ROW + 16 * c);
  }
}

template <int N> struct Int { static constexpr int value = N; };

DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

DEV void store2(float* p, float a, float b) { *reinterpret_cast<float2*>(p) = make_float2(a, b); }
DEV void store_y(uint16_t* p, float a, float b, float sr, float sc0, float sc1) {
  *reinterpret_cast<uint32_t*>(p) = f2bf(__fmul_rn(__fmul_rn(a, sr), sc0)) | (uint32_t)f2bf(__fmul_rn(__fmul_rn(b, sr), sc1)) << 16;
}

template <int MODE, int TM, int BN, int G> DEV void gemm_body(const Args& a) {
  constexpr int WN = 32, WARPS_N = BN / WN, WARPS_M = 8 / WARPS_N, WM = TM / WARPS_M, MI = WM / 16, NI = WN / 8;
  constexpr bool PEEL = MODE == M_PEARLC || MODE == M_PEARLC_S;
  constexpr bool STORE_CU = MODE == M_PEARLC || MODE == M_CT, PROMOTE = G > 0;
  constexpr int PERIOD = PROMOTE ? G / 2 : 1;
  static_assert(G == 0 || G % 2 == 0, "a promotion window is whole 64-byte stages");
  constexpr uint32_t A_ST = TM * BK, B_ST = BN * BK;
  static_assert(WARPS_M * WARPS_N == 8 && MI >= 1 && NI % 2 == 0, "8 warps of 16k x 32 columns");
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = smem + STAGES * A_ST;
  uint8_t* sPA = sB + STAGES * B_ST;
  uint8_t* sPB = sPA + TM * PEEL_ROW;
  const int M = a.i[0], N = a.i[1], K = a.i[2], group = a.i[3];
  const int tiles_m = M / TM, tiles_n = N / BN, tiles = tiles_m * tiles_n, KB = K / BK;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
  const uint8_t* A = reinterpret_cast<const uint8_t*>(a.p[0]);
  const uint8_t* B = reinterpret_cast<const uint8_t*>(a.p[1]);
  float* C = reinterpret_cast<float*>(a.p[2]);
  float* U = reinterpret_cast<float*>(a.p[3]);
  uint16_t* Y = reinterpret_cast<uint16_t*>(a.p[4]);
  const float* rs = reinterpret_cast<const float*>(a.p[7]);
  const float* cs = reinterpret_cast<const float*>(a.p[8]);
  // ldmatrix.x4 row addresses: A's four 8 x 16-byte matrices are (rows 0-7, 8-15) x (k 0-15, 16-31) in fragment order, B's
  // (k 0-15, 16-31) x (columns 0-7, 8-15) of two n8 blocks
  const int a_row = wm * WM + (lane & 15), a_half = lane >> 4;
  const int b_row = wn * WN + (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;
  const int g = lane >> 2, q = lane & 3;

  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    int tm, tn;
    raster(t, tiles_m, tiles_n, group, tm, tn);
    const uint8_t* ga = A + (size_t)tm * TM * K;
    const uint8_t* gb = B + (size_t)tn * BN * K;
    if constexpr (PEEL) {
      peel_rows<TM>(sPA, reinterpret_cast<const uint8_t*>(a.p[5]) + (size_t)tm * TM * PEEL_ROW, tid);
      peel_rows<BN>(sPB, reinterpret_cast<const uint8_t*>(a.p[6]) + (size_t)tn * BN * PEEL_ROW, tid);
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < KB) {
        stage_rows<TM>(sA + s * A_ST, ga + s * BK, K, tid);
        stage_rows<BN>(sB + s * B_ST, gb + s * BK, K, tid);
      }
      cp_commit();
    }
    float acc[MI][NI][4], total[MI][NI][4];
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++)
#pragma unroll
        for (int e = 0; e < 4; e++) { acc[mi][ni][e] = 0.0f; total[mi][ni][e] = 0.0f; }

    for (int kb = 0; kb < KB; kb++) {
      cp_wait<STAGES - 2>();
      __syncthreads();
      {
        const int nk = kb + STAGES - 1, s = nk % STAGES;
        if (nk < KB) {
          stage_rows<TM>(sA + s * A_ST, ga + (size_t)nk * BK, K, tid);
          stage_rows<BN>(sB + s * B_ST, gb + (size_t)nk * BK, K, tid);
        }
        cp_commit();
      }
      if (PROMOTE && kb % PERIOD == 0) {
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
          for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mi][ni][e] = 0.0f;
      }
      const uint32_t a0 = sa(sA + (kb % STAGES) * A_ST), b0 = sa(sB + (kb % STAGES) * B_ST);
#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        uint32_t fa[MI][4], fb[NI][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) ldsm4(fa[mi], a0 + swz64(a_row + 16 * mi, 2 * kk + a_half));
#pragma unroll
        for (int nj = 0; nj < NI / 2; nj++) {
          uint32_t r[4];
          ldsm4(r, b0 + swz64(b_row + 16 * nj, 2 * kk + b_half));
          fb[2 * nj][0] = r[0]; fb[2 * nj][1] = r[1]; fb[2 * nj + 1][0] = r[2]; fb[2 * nj + 1][1] = r[3];
        }
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
          for (int ni = 0; ni < NI; ni++) qmma(acc[mi][ni], fa[mi], fb[ni]);
      }
      if (PROMOTE && kb % PERIOD == PERIOD - 1) {
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
          for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) total[mi][ni][e] = __fadd_rn(total[mi][ni][e], acc[mi][ni][e]);
      }
    }
    if constexpr (!PROMOTE) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
          for (int e = 0; e < 4; e++) total[mi][ni][e] = acc[mi][ni][e];
    }
    cp_wait<0>();
    __syncthreads();

    const int row0 = tm * TM + wm * WM + g, col0 = tn * BN + wn * WN + 2 * q;
    if constexpr (MODE == M_PLAIN) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        const int row = row0 + 16 * mi;
        const float r0 = rs[row], r1 = rs[row + 8];
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          const int col = col0 + 8 * ni;
          const float c0 = cs[col], c1 = cs[col + 1];
          store_y(Y + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1], r0, c0, c1);
          store_y(Y + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3], r1, c0, c1);
        }
      }
    } else if constexpr (STORE_CU) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          const int row = row0 + 16 * mi, col = col0 + 8 * ni;
          store2(C + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1]);
          store2(C + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3]);
        }
    }
    if constexpr (PEEL) {
      const uint32_t pa0 = sa(sPA), pb0 = sa(sPB);
#pragma unroll
      for (int s = 0; s < 4; s++) {
        uint32_t fa[MI][4], fb[NI][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) ldsm4(fa[mi], pa0 + swz128(a_row + 16 * mi, 2 * s + a_half));
#pragma unroll
        for (int nj = 0; nj < NI / 2; nj++) {
          uint32_t r[4];
          ldsm4(r, pb0 + swz128(b_row + 16 * nj, 2 * s + b_half));
          fb[2 * nj][0] = r[0]; fb[2 * nj][1] = r[1]; fb[2 * nj + 1][0] = r[2]; fb[2 * nj + 1][1] = r[3];
        }
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
          for (int ni = 0; ni < NI; ni++) hmma(total[mi][ni], fa[mi], fb[ni]);
      }
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        const int row = row0 + 16 * mi;
        const float r0 = rs[row], r1 = rs[row + 8];
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          const int col = col0 + 8 * ni;
          const float c0 = cs[col], c1 = cs[col + 1];
          if constexpr (STORE_CU) {
            store2(U + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1]);
            store2(U + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3]);
          }
          store_y(Y + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1], r0, c0, c1);
          store_y(Y + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3], r1, c0, c1);
        }
      }
    }
    __syncthreads();
  }
}

// The 128 x 128 GEMMs on bc-fb55a759's mainloop (mainloop_sm120.cuh: TMA into 3 stages of 128 bytes of depth, a producer
// warpgroup, 8 consumer warps of 64 x 32, the chain word for word gemm_body's), with gemm_body's epilogues; PEARLC_B's is GPU 2's
// fused one (bc-7442ca43's gemm_pearlc_b: each (row, 32-column chunk)'s message of C~ and U hashed from registers by
// pouw_hash's quad_msg_digest_b3, BLAKE3 keyed by blake3("pearl-c/h1/message"), into p9 in hash_msg_v's layout), its 1 KB of
// scratch per warp in the tail stage's P_A rows once every consumer has read its peel fragments.  p10 the four tensor maps
// (ml::Maps; P_A, P_B the peel's, else A, B again), p11 the message key.
constexpr int ML_NT = 384;
template <int G> constexpr int ML_RP = G ? 24 : 40;   // the producer's registers: the mainloop's recommended v1 and v2
template <int G> constexpr int ML_U = G ? 1 : 2;      // stages per K-loop iteration (v1 at 2 spills under nvcc 13.0)
template <int MODE, int G> DEV void ml_body(const Args& a) {
  constexpr bool PEEL = MODE >= M_PEARLC, HASH = MODE == M_PEARLC_B;
  constexpr bool STORE_C = MODE == M_CT || MODE == M_PEARLC, STORE_U = MODE == M_PEARLC;
  using L = ml::Mainloop<128, 128, G, 3, PEEL, 1, ML_RP<G>>;
  constexpr int MI = L::MI, NI = L::NI;
  static_assert(L::NT == ML_NT && MI == 4 && NI == 4, "gemm_body's 128 x 128 fragments: 8 warps of 64 x 32");
  extern __shared__ __align__(1024) uint8_t ml_smem[];
  const int M = a.i[0], N = a.i[1], K = a.i[2], group = a.i[3];
  const int tiles_m = M / 128, tiles_n = N / 128;
  L ml(ml_smem);
  ml.roles(reinterpret_cast<const ml::Maps*>(a.p[10]), M, N, K, group);
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    ml::raster(t, tiles_m, tiles_n, group, tm, tn);
    float acc[MI][NI][4], total[MI][NI][4];
    ml.template tile<ML_U<G>, true>(K, acc, total);
    // a block boundary (never taken): scheduled among the chain's last MMAs, the plain epilogue spills v1
    if (MODE == M_PLAIN && K < 0) __trap();
    uint32_t tid;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));   // the epilogue's indices afresh each tile (hoisted, they spill)
    const int warp = tid / 32, lane = tid % 32, wm = warp / L::WARPS_N, wn = warp % L::WARPS_N;
    const int a_row = wm * L::WM + (lane & 15), a_half = lane >> 4;
    const int b_row = wn * L::WN + (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;
    const int g = lane >> 2, q = lane & 3;
    const int row0 = tm * 128 + wm * L::WM + g, col0 = tn * 128 + wn * L::WN + 2 * q;
    float* C = reinterpret_cast<float*>(a.p[2]);   // read here, from the parameter bank, not held through the mainloop
    float* U = reinterpret_cast<float*>(a.p[3]);
    uint16_t* Y = reinterpret_cast<uint16_t*>(a.p[4]);
    const float* rs = reinterpret_cast<const float*>(a.p[7]);
    const float* cs = reinterpret_cast<const float*>(a.p[8]);
    if constexpr (MODE == M_PLAIN) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        const int row = row0 + 16 * mi;
        const float r0 = rs[row], r1 = rs[row + 8];
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          const int col = col0 + 8 * ni;
          const float c0 = cs[col], c1 = cs[col + 1];
          store_y(Y + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1], r0, c0, c1);
          store_y(Y + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3], r1, c0, c1);
        }
      }
    } else if constexpr (STORE_C) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          const int row = row0 + 16 * mi, col = col0 + 8 * ni;
          store2(C + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1]);
          store2(C + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3]);
        }
    }
    if constexpr (PEEL) {
      if constexpr (HASH) {
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
          for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mi][ni][e] = total[mi][ni][e];   // C~, beside U
      }
      const uint32_t pa0 = ml.tail_wait(), pb0 = pa0 + L::A_BYTES;
      // with C~ held beside U, one step's fragments at a time (hoisting all four spills v1)
#pragma unroll(HASH ? 1 : 4)
      for (int s = 0; s < 4; s++) {
        uint32_t fa[MI][4], fb[NI][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) ldsm4(fa[mi], pa0 + swz128(a_row + 16 * mi, 2 * s + a_half));
#pragma unroll
        for (int nj = 0; nj < NI / 2; nj++) {
          uint32_t r[4];
          ldsm4(r, pb0 + swz128(b_row + 16 * nj, 2 * s + b_half));
          fb[2 * nj][0] = r[0]; fb[2 * nj][1] = r[1]; fb[2 * nj + 1][0] = r[2]; fb[2 * nj + 1][1] = r[3];
        }
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
          for (int ni = 0; ni < NI; ni++) hmma(total[mi][ni], fa[mi], fb[ni]);
      }
      if constexpr (!HASH) ml.tail_release();
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        const int row = row0 + 16 * mi;
        const float r0 = rs[row], r1 = rs[row + 8];
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          const int col = col0 + 8 * ni;
          const float c0 = cs[col], c1 = cs[col + 1];
          if constexpr (STORE_U) {
            store2(U + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1]);
            store2(U + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3]);
          }
          store_y(Y + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1], r0, c0, c1);
          store_y(Y + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3], r1, c0, c1);
        }
      }
      if constexpr (HASH) {
        asm volatile("bar.sync 1, %0;" ::"n"(L::NCT) : "memory");   // every consumer's peel reads of the tail are done
        uint64_t* scratch = reinterpret_cast<uint64_t*>(ml_smem + (pa0 - ml.base)) + 128 * warp;
        const uint32_t* kp = reinterpret_cast<const uint32_t*>(a.p[11]);
        uint32_t key[8];
        asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(key[0]), "=r"(key[1]), "=r"(key[2]), "=r"(key[3]) : "l"(kp));
        asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(key[4]), "=r"(key[5]), "=r"(key[6]), "=r"(key[7]) : "l"(kp + 4));
        uint8_t* const dg = reinterpret_cast<uint8_t*>(a.p[9]) + (size_t)(tn * 128 + wn * L::WN) / 32 * 32;
        const int per = N / 32, row = tm * 128 + wm * L::WM;
        auto region = [&](auto pc) {
          constexpr int p = decltype(pc)::value;
          auto frag = [&](int arr, int j, int i) { return arr ? total[2 * p + (j >> 2)][j & 3][i] : acc[2 * p + (j >> 2)][j & 3][i]; };
          uint32_t h[8];
          ph::quad_msg_digest_b3<2>(frag, key, scratch, h);
          uint4* o = reinterpret_cast<uint4*>(dg + (size_t)(row + ph::pearlc::narrow_row(p, lane)) * per * 32);
          o[0] = make_uint4(h[0], h[1], h[2], h[3]);
          o[1] = make_uint4(h[4], h[5], h[6], h[7]);
        };
        region(Int<0>{});
        region(Int<1>{});
        ml.tail_release();
      }
    }
  }
}

// G4: pearl-c-sm120-v1 (pearl_c_device.SM120); _u: unpromoted (SM120_UNPROMOTED); gemm_fast is the plain GEMM unpromoted;
// _d: the 64 x 64 tile; gemm_words: the 128 x 32 tile; ML_GEMM: the 128 x 128 tile on the mainloop, _s y alone, _b fused
#define ML_GEMM(NAME, MODE, G)                                                                                \
  extern "C" __global__ void __launch_bounds__(ML_NT, 1) NAME(const Args a) { ml_body<MODE, G>(a); } \
  extern "C" __global__ void NAME##_smem(int* out) { *out = ml::Mainloop<128, 128, G, 3, true, 1, ML_RP<G>>::SMEM; }
#define GEMM(NAME, MODE, TM, BN, G)                                                                                 \
  extern "C" __global__ void __launch_bounds__(NT, 1) NAME(const Args a) { gemm_body<MODE, TM, BN, G>(a); } \
  extern "C" __global__ void NAME##_smem(int* out) { *out = gemm_smem<MODE, TM, BN>(); }
ML_GEMM(gemm_plain, M_PLAIN, 4)
ML_GEMM(gemm_fast, M_PLAIN, 0)
ML_GEMM(gemm_ct, M_CT, 4)
ML_GEMM(gemm_pearlc, M_PEARLC, 4)
ML_GEMM(gemm_ct_u, M_CT, 0)
ML_GEMM(gemm_pearlc_u, M_PEARLC, 0)
ML_GEMM(gemm_pearlc_s, M_PEARLC_S, 4)
ML_GEMM(gemm_pearlc_s_u, M_PEARLC_S, 0)
ML_GEMM(gemm_pearlc_b, M_PEARLC_B, 4)
ML_GEMM(gemm_pearlc_b_u, M_PEARLC_B, 0)
GEMM(gemm_ct_d, M_CT, 64, 64, 4)
GEMM(gemm_pearlc_d, M_PEARLC, 64, 64, 4)
GEMM(gemm_ct_d_u, M_CT, 64, 64, 0)
GEMM(gemm_pearlc_d_u, M_PEARLC, 64, 64, 0)
GEMM(gemm_words, M_CT, 128, 32, 4)
GEMM(gemm_words_u, M_CT, 128, 32, 0)
// The FP32 words of a 32-column product at decode's few rows (A'F_B at M not a multiple of 128): one warp per m16 x n8
// output tile over all of K, the chain in the GEMM's order (G-atom promotion windows from +0, or one accumulator at G = 0),
// fragments loaded straight from global memory through a ring of D atoms in registers, so D atoms' loads are in flight
// behind the chain.  A block's 4 warps are one m16 row's 4 n8 tiles and share its A rows.  p0 A (M x K), p1 B (N x K), p2 out
// (M x N FP32); i0 M, i1 N, i2 K (multiples of 16, 8, 32 D)
template <int G> DEV void words_body(const Args& a) {
  constexpr int D = 16;
  static_assert(D % (G > 0 ? G : 1) == 0, "the ring holds whole promotion windows");
  const int M = a.i[0], N = a.i[1], K = a.i[2], tn = N / 8, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
  const int w = blockIdx.x * 4 + threadIdx.x / 32;
  if (w >= M / 16 * tn) return;
  const int r0 = w / tn * 16, c0 = w % tn * 8, J = K / 32;
  const uint8_t* A0 = reinterpret_cast<const uint8_t*>(a.p[0]) + (size_t)(r0 + g) * K + 4 * q;
  const uint8_t* A1 = A0 + (size_t)8 * K;
  const uint8_t* B0 = reinterpret_cast<const uint8_t*>(a.p[1]) + (size_t)(c0 + g) * K + 4 * q;
  uint32_t fa[D][4], fb[D][2];
  auto load = [&](int j, uint32_t* x, uint32_t* y) {
    const int kk = 32 * j;
    x[0] = ld32(A0 + kk); x[1] = ld32(A1 + kk); x[2] = ld32(A0 + kk + 16); x[3] = ld32(A1 + kk + 16);
    y[0] = ld32(B0 + kk); y[1] = ld32(B0 + kk + 16);
  };
#pragma unroll
  for (int d = 0; d < D; d++) load(d, fa[d], fb[d]);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}, total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int j0 = 0; j0 < J; j0 += D) {
#pragma unroll
    for (int d = 0; d < D; d++) {
      if constexpr (G > 0) {
        if (d % G == 0)
          for (int e = 0; e < 4; e++) acc[e] = 0.0f;
      }
      qmma(acc, fa[d], fb[d]);
      if constexpr (G > 0) {
        if (d % G == G - 1)
          for (int e = 0; e < 4; e++) total[e] = __fadd_rn(total[e], acc[e]);
      }
      load(min(j0 + D + d, J - 1), fa[d], fb[d]);
    }
  }
  float* out = reinterpret_cast<float*>(a.p[2]) + (size_t)(r0 + g) * N + c0 + 2 * q;
  const float* v = G > 0 ? total : acc;
  store2(out, v[0], v[1]);
  store2(out + (size_t)8 * N, v[2], v[3]);
}
extern "C" __global__ void __launch_bounds__(128) words(const Args a) { words_body<4>(a); }
extern "C" __global__ void __launch_bounds__(128) words_u(const Args a) { words_body<0>(a); }

// ---------------------------------------------------------------- noise lines (Pearl's line rule; hash.cuh's BLAKE3)
DEV uint32_t isqrt(uint32_t v) {
  uint32_t r = (uint32_t)sqrt((double)v);
  while ((uint64_t)r * r > v) r--;
  while ((uint64_t)(r + 1) * (r + 1) <= v) r++;
  return r;
}

// The position of column t of k in form_s5's K32 order (an involution within each aligned 32)
DEV int k32(int t) { return (t & ~31) | (t >> 1 & 3) << 3 | (t >> 3 & 3) << 1 | (t & 1); }

// p0 the line key (8 words), p1 codes (count x 32, row-major), p2 transposed (32 x count) or 0, p3 transposed into 128 x count
// (rows 32-127 left as they are) or 0, both in K32 order along count (the F lines, count = k); i0 side, i1 factor, i2 count,
// i3 bf16(32 x the line's L2 norm): 0x4600 for Pearl's 256, 0x4400 for 16.  i4 lines keyed per row (-h3's E_A,
// `pearl_c.PearlC.encode`): line t < i4 is line 0 under the noise-line subkey of its own seed, p4 + 32 t (b3_seeds' mode 1);
// the rest, line t under p0.  lines_s is -h3s's (`pearl_c.sample_line` on an S512Seed): the raw bytes H_line (hash_h3s.cuh's
// s512_line_raw) in place of BLAKE3's, and p4 the rows' line keys themselves (s512_seeds' p2)
template <bool S512> DEV void lines_body(const Args a) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x, count = a.i[2];
  if (line >= count) return;
  const bool own = line < a.i[4];
  uint32_t kw[8], h[8];
  if (own && !S512) {
    uint32_t seed[8];
    load_key(a.p[4] + (uint64_t)line * 32, seed);
    b3_short<23>(seed, [&](int j) { return (uint32_t)NOISE_LINE_LABEL[j]; }, kw);
  } else {
    load_key(own ? a.p[4] + (uint64_t)line * 32 : a.p[0], kw);
  }
  const uint32_t at = own ? 0u : (uint32_t)line;
  if (S512) {
    s512_line_raw(kw, (uint32_t)a.i[0], (uint32_t)a.i[1], at, h);
  } else {
    uint32_t msg[16];
#pragma unroll
    for (int i = 0; i < 16; i++) msg[i] = 0;
    msg[0] = (uint32_t)a.i[0] | (uint32_t)a.i[1] << 8 | (at & 0xFFFF) << 16;
    msg[1] = at >> 16;
    blake3_block(kw, msg, h);
  }
  int xs[32];
  uint32_t sumsq = 0;
#pragma unroll
  for (int j = 0; j < 32; j++) {
    const uint32_t b = (h[j / 4] >> (8 * (j % 4))) & 0xFF;
    xs[j] = (b >> 7 ? -1 : 1) * (int)((b & 0x7F) + 1);
    sumsq += xs[j] * xs[j];
  }
  const uint16_t scale = bdiv((uint16_t)a.i[3], f2bf((float)isqrt(sumsq * 1024)));
  uint8_t c[32];
#pragma unroll
  for (int j = 0; j < 32; j++) c[j] = e4m3(bf2f(bmul(f2bf((float)xs[j]), scale)));
  uint32_t w[8];
#pragma unroll
  for (int i = 0; i < 8; i++) w[i] = c[4 * i] | c[4 * i + 1] << 8 | c[4 * i + 2] << 16 | (uint32_t)c[4 * i + 3] << 24;
  uint4* out = reinterpret_cast<uint4*>(a.p[1] + (size_t)line * 32);
  out[0] = make_uint4(w[0], w[1], w[2], w[3]);
  out[1] = make_uint4(w[4], w[5], w[6], w[7]);
#pragma unroll
  for (int s = 2; s < 4; s++)
    if (a.p[s]) {
      uint8_t* t = reinterpret_cast<uint8_t*>(a.p[s]);
#pragma unroll
      for (int j = 0; j < 32; j++) t[(size_t)j * count + k32(line)] = c[j];
    }
}
extern "C" __global__ void __launch_bounds__(128) lines(const Args a) { lines_body<false>(a); }
extern "C" __global__ void __launch_bounds__(128) lines_s(const Args a) { lines_body<true>(a); }

// ---------------------------------------------------------------- forming v1 (cheap-binding §5)
constexpr uint32_t S5_LINE = 0x3AB504F3;   // f32(sqrt(32) / (256 * 16))
constexpr float S5_FLOOR = 0x1p-32f;

// one warp per FP32 row, one pass: s = max|x|, rho (32 partial FMA chains, chain c over positions 8c + 256i in order of i,
// then the xor butterfly), alpha, beta_row.  Lane l reads x[256i + 4l ..] and x[256i + 128 + 4l ..], so chain c's position is
// the first (c < 16) or second (c >= 16) vector's x of lane 2 (c mod 16).  p0 rows (rows x k words), p1 alpha, p2 beta,
// p3 1/alpha (glue), p4 (s, rho) per row; i0 rows, i1 k (a multiple of 256)
extern "C" __global__ void __launch_bounds__(256) stats_s5(const Args a) {
  const int row = (blockIdx.x * 256 + threadIdx.x) / 32, lane = threadIdx.x % 32, rows = a.i[0], k = a.i[1];
  if (row >= rows) return;
  const float4* x = reinterpret_cast<const float4*>(a.p[0]) + (size_t)row * (k / 4);
  const int src = 2 * (lane & 15);
  float s = 0.0f, part = 0.0f;
#pragma unroll 4
  for (int j = lane; j < k / 4; j += 64) {
    const float4 v = x[j], w = x[j + 32];
    s = fmaxf(s, fmaxf(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))),
                       fmaxf(fmaxf(fabsf(w.x), fabsf(w.y)), fmaxf(fabsf(w.z), fabsf(w.w)))));
    const float lo = __shfl_sync(~0u, v.x, src), hi = __shfl_sync(~0u, w.x, src), e = lane < 16 ? lo : hi;
    part = __fmaf_rn(e, e, part);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { s = fmaxf(s, __shfl_xor_sync(~0u, s, o)); part = __fadd_rn(part, __shfl_xor_sync(~0u, part, o)); }
  if (lane) return;
  const float rho = __fsqrt_rn(__fdiv_rn(part, (float)((k + 7) / 8)));
  const float alpha = __fdiv_rn(448.0f, fmaxf(__fadd_rn(s, __fmul_rn(4.0f, rho)), S5_FLOOR));
  reinterpret_cast<float*>(a.p[1])[row] = alpha;
  reinterpret_cast<float*>(a.p[2])[row] = __fmul_rn(__fmul_rn(rho, alpha), __uint_as_float(S5_LINE));
  reinterpret_cast<float*>(a.p[3])[row] = __frcp_rn(alpha);
  reinterpret_cast<float2*>(a.p[4])[row] = make_float2(s, rho);
}

#include "h2_rows_stats.cuh"   // -h2's rows form `s`, stats_s5's outputs from hash_rows_b3s's loads (after S5_LINE, S5_FLOOR)

// E' = e4m3(beta_row E) per row.  p0 E codes (rows x 32), p1 beta (f32), p2 out E' codes (rows x 32); i0 rows
extern "C" __global__ void eprime(const Args a) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.i[0]) return;
  const uint8_t* E = reinterpret_cast<const uint8_t*>(a.p[0]) + (size_t)row * 32;
  const float beta = reinterpret_cast<const float*>(a.p[1])[row];
  uint32_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < 32; j++) w[j / 4] |= (uint32_t)e4m3(__fmul_rn(beta, e4m3f(E[j]))) << (8 * (j % 4));
  uint4* out = reinterpret_cast<uint4*>(a.p[2] + (size_t)row * 32);
  out[0] = make_uint4(w[0], w[1], w[2], w[3]);
  out[1] = make_uint4(w[4], w[5], w[6], w[7]);
}

// four E4M3 codes in memory order (a at the lowest byte): two cvt.rn.satfinite.e4m3x2.f32 on live pairs, each converting its
// first source into the upper byte
DEV uint32_t e4m3x4(float a, float b, float c, float d) {
  uint32_t w;
  asm("{\n .reg .b16 lo, hi;\n cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      " mov.b32 %0, {lo, hi};\n}"
      : "=r"(w)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return w;
}

// X~ = e4m3(D), D = the atom with alpha x as its accumulator over E' and the F lines: per warp 16 rows x 128 of depth, sixteen
// m16n8k32 atoms (A = the rows' E', B = eight consecutive F lines, C = alpha x).  p0 rows (FP32), p1 alpha, p2 E' codes
// (rows x 32), p3 F lines (k x 32), p4 out codes (rows x k, in K32 order); i0 rows, i1 k.  grid (k / 128, rows / 64), 128
// threads.  The block's 64 x 128 FP32 tile is staged by cp.async first (a warp's copy is one row's 512 bytes), so every row
// is read whole and at once.
//
// K32 order: within each aligned 32 of k, byte 8q + 2b + s holds column 8b + 2q + s (b, q < 4, s < 2), the transpose of the
// 4 x 4 grid of column pairs.  It is the accumulators' order: over an aligned 32, lane q's pairs of atoms b = 0 .. 3 are
// adjacent, so form_s5 casts a row's 8 codes there (two e4m3x4) and stores them with one 64-bit store.  Each k32 group stays
// in its own 32 bytes, so a product over k is unchanged when both its operands are in K32 order: A'B~ (form_s5's codes) and
// A'F_B, B~F_A, F_B F_A (lines' transposed F lines).  form_s5_scalar is the one-code cast with 16-bit stores in the same
// order, its gate: both write the same codes.
constexpr int FORM_ROW = 160;   // shared bytes per staged row of codes: 128, padded so a half-warp's 64-bit stores hit 32 banks
constexpr int FORM_X = 544;     // shared bytes per staged FP32 row: 512, padded so a half-warp's float2 reads hit 32 banks
//
// MODE 0 is form_s5's loop and 1 form_s5_scalar's; ROW the staged codes' row stride (form_s5_r144's 144 puts two lanes of a
// half-warp's 64-bit stores on each bank, timed beside it).  With RATE (form_rate*) the block takes tile blockIdx.x of the
// (rows / 64) x (k / 128) grid in row-major order and runs the loop i[2] times over it, so the loop's price per code is
// measured in the mix it runs in without the global traffic that bounds form_s5; an empty asm with a memory clobber per
// repeat keeps the compiler from hoisting or merging repeats.  Modes 0 and 1 then write the same codes as their kernels;
// 2 is 0 with each 32-bit word's cast replaced by one of its floats' bits, and 3 is 2 without x (the atoms, F's loads, the
// stores).
template <int MODE, bool RATE, int ROW = FORM_ROW> DEV void form_body(const Args& a, uint8_t* sO, uint8_t* sX) {
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4, k = a.i[1];
  const int bx = RATE ? blockIdx.x % (k / 128) : blockIdx.x, by = RATE ? blockIdx.x / (k / 128) : blockIdx.y;
  const int t0 = bx * 128, r0 = by * 64, rw = r0 + 16 * warp;
  const uint8_t* E = reinterpret_cast<const uint8_t*>(a.p[2]);
  const uint8_t* F = reinterpret_cast<const uint8_t*>(a.p[3]);
  const uint8_t* x = reinterpret_cast<const uint8_t*>(a.p[0]) + ((size_t)r0 * k + t0) * 4;
  const float* alpha = reinterpret_cast<const float*>(a.p[1]);
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int r = warp + 4 * j;
    cp16(sX + r * FORM_X + 16 * lane, x + (size_t)r * k * 4 + 16 * lane);
  }
  cp_commit();
  const uint32_t fa[4] = {ld32(E + (size_t)(rw + g) * 32 + 4 * q), ld32(E + (size_t)(rw + g + 8) * 32 + 4 * q),
                          ld32(E + (size_t)(rw + g) * 32 + 16 + 4 * q), ld32(E + (size_t)(rw + g + 8) * 32 + 16 + 4 * q)};
  const float al0 = alpha[rw + g], al1 = alpha[rw + g + 8];
  cp_wait<0>();
  __syncthreads();
  uint8_t* o0 = sO + (16 * warp + g) * ROW;
  uint8_t* o1 = o0 + 8 * ROW;
  // atom j of the block's 16: B is lines t0 + 8j + g, and lane q's accumulators are columns 8j + 2q, 8j + 2q + 1
  const uint8_t* x0 = sX + (16 * warp + g) * FORM_X + 8 * q;
  const uint8_t* x1 = x0 + 8 * FORM_X;
  const uint8_t* lines = F + (size_t)(t0 + g) * 32 + 4 * q;
  const int reps = RATE ? a.i[2] : 1;
  for (int rep = 0; rep < reps; rep++) {
    float b0 = al0, b1 = al1;
    if (RATE) asm volatile("" : "+f"(b0), "+f"(b1) : : "memory");
    if (MODE != 1) {
#pragma unroll 2
      for (int p = 0; p < 4; p++) {
        float d[4][4];
#pragma unroll
        for (int b = 0; b < 4; b++) {
          const int j = 4 * p + b;
          const uint32_t f[2] = {ld32(lines + 256 * j), ld32(lines + 256 * j + 16)};
          if (MODE == 3) {
            d[b][0] = d[b][1] = b0;
            d[b][2] = d[b][3] = b1;
          } else {
            const float2 v0 = *reinterpret_cast<const float2*>(x0 + 32 * j), v1 = *reinterpret_cast<const float2*>(x1 + 32 * j);
            d[b][0] = __fmul_rn(b0, v0.x); d[b][1] = __fmul_rn(b0, v0.y); d[b][2] = __fmul_rn(b1, v1.x); d[b][3] = __fmul_rn(b1, v1.y);
          }
          qmma(d[b], fa, f);
        }
        const int c = 32 * p + 8 * q;
        if (MODE == 0) {
          *reinterpret_cast<uint2*>(o0 + c) = make_uint2(e4m3x4(d[0][0], d[0][1], d[1][0], d[1][1]),
                                                         e4m3x4(d[2][0], d[2][1], d[3][0], d[3][1]));
          *reinterpret_cast<uint2*>(o1 + c) = make_uint2(e4m3x4(d[0][2], d[0][3], d[1][2], d[1][3]),
                                                         e4m3x4(d[2][2], d[2][3], d[3][2], d[3][3]));
        } else {
          *reinterpret_cast<uint2*>(o0 + c) = make_uint2(__float_as_uint(d[0][0]), __float_as_uint(d[2][0]));
          *reinterpret_cast<uint2*>(o1 + c) = make_uint2(__float_as_uint(d[1][2]), __float_as_uint(d[3][2]));
        }
      }
    } else {
#pragma unroll 4
      for (int j = 0; j < 16; j++) {
        const uint32_t f[2] = {ld32(lines + 256 * j), ld32(lines + 256 * j + 16)};
        const float2 v0 = *reinterpret_cast<const float2*>(x0 + 32 * j), v1 = *reinterpret_cast<const float2*>(x1 + 32 * j);
        float d[4] = {__fmul_rn(b0, v0.x), __fmul_rn(b0, v0.y), __fmul_rn(b1, v1.x), __fmul_rn(b1, v1.y)};
        qmma(d, fa, f);
        const int c = 32 * (j / 4) + 8 * q + 2 * (j % 4);
        *reinterpret_cast<uint16_t*>(o0 + c) = e4m3(d[0]) | (uint16_t)e4m3(d[1]) << 8;
        *reinterpret_cast<uint16_t*>(o1 + c) = e4m3(d[2]) | (uint16_t)e4m3(d[3]) << 8;
      }
    }
  }
  __syncthreads();
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[4]);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int idx = tid + 128 * j, r = idx / 8, c = idx % 8;
    *reinterpret_cast<uint4*>(out + (size_t)(r0 + r) * k + t0 + 16 * c) = *reinterpret_cast<const uint4*>(sO + r * ROW + 16 * c);
  }
}
// form_body<MODE> as a kernel, p0 .. p4 and i0, i1 as form_s5's; with RATE (form_rate*) its loop repeated i[2] times per block,
// grid any number of blocks up to (rows / 64) * (k / 128)
#define FORM_KERNEL(NAME, MODE, RATE, ROW)                                         \
  extern "C" __global__ void __launch_bounds__(128) NAME(const Args a) {          \
    __shared__ __align__(16) uint8_t sO[64 * ROW];                                 \
    __shared__ __align__(16) uint8_t sX[64 * FORM_X];                              \
    form_body<MODE, RATE, ROW>(a, sO, sX);                                         \
  }
FORM_KERNEL(form_s5, 0, false, FORM_ROW)
FORM_KERNEL(form_s5_r144, 0, false, 144)
FORM_KERNEL(form_s5_scalar, 1, false, FORM_ROW)
FORM_KERNEL(form_rate, 0, true, FORM_ROW)
FORM_KERNEL(form_rate_r144, 0, true, 144)
FORM_KERNEL(form_rate_scalar, 1, true, FORM_ROW)
FORM_KERNEL(form_rate_nocast, 2, true, FORM_ROW)
FORM_KERNEL(form_rate_nocast_r144, 2, true, 144)
FORM_KERNEL(form_rate_atoms, 3, true, FORM_ROW)

// the adopted quantizer (the model's FP8 path with PoUW off): codes = e4m3(alpha x) row-major, alpha from stats_s5.
// p0 rows (FP32), p1 alpha, p2 out codes; i0 rows, i1 k.  One thread per 16 elements
extern "C" __global__ void quant_s5(const Args a) {
  const int k = a.i[1];
  const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= (size_t)a.i[0] * (k / 16)) return;
  const int row = (int)(g / (k / 16)), t = (int)(g % (k / 16)) * 16;
  const float al = reinterpret_cast<const float*>(a.p[1])[row];
  const float4* x = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(a.p[0]) + (size_t)row * k + t);
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float4 v = x[i];
    o[i] = e4m3(__fmul_rn(al, v.x)) | e4m3(__fmul_rn(al, v.y)) << 8 | e4m3(__fmul_rn(al, v.z)) << 16 | (uint32_t)e4m3(__fmul_rn(al, v.w)) << 24;
  }
  *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a.p[2]) + (size_t)row * k + t) = make_uint4(o[0], o[1], o[2], o[3]);
}

// ---------------------------------------------------------------- the plain baseline's quantize and inputs
// one CTA per FP32 row: s = 448 / amax, codes = fp8(x s) row-major, glue scale amax / 448.  p0 X (rows x k), p1 out codes,
// p2 out glue (f32); i0 rows, i1 k
extern "C" __global__ void __launch_bounds__(256) plainq(const Args a) {
  __shared__ float red[8];
  const int row = blockIdx.x, k = a.i[1], tid = threadIdx.x;
  const float4* X = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(a.p[0]) + (size_t)row * k);
  float amax = 0.0f;
  for (int g = tid; g < k / 4; g += 256) {
    const float4 v = X[g];
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
  if (tid % 32 == 0) red[tid / 32] = amax;
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int i = 1; i < 8; i++) amax = fmaxf(amax, red[i]);
  const float s = amax > 0.0f ? __fdiv_rn(448.0f, amax) : 1.0f;
  if (tid == 0) reinterpret_cast<float*>(a.p[2])[row] = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 1.0f;
  uint32_t* out = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(a.p[1]) + (size_t)row * k);
  for (int g = tid; g < k / 4; g += 256) {
    const float4 v = X[g];
    out[g] = e4m3(__fmul_rn(v.x, s)) | e4m3(__fmul_rn(v.y, s)) << 8 | e4m3(__fmul_rn(v.z, s)) << 16 | (uint32_t)e4m3(__fmul_rn(v.w, s)) << 24;
  }
}

// timing inputs for v1: FP32 rows, x = (mix32 byte, -128 -> -127) 2^(-4 - h mod 5) with 16 random low mantissa bits when nonzero.
// p0 out; i0 rows, i1 k, i2 seed
extern "C" __global__ void gen_f32(const Args a) {
  const size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= (size_t)a.i[0] * a.i[1]) return;
  const uint32_t h = mix32((uint32_t)a.i[2] ^ mix32((uint32_t)w * 2u + 1u)), b = h & 0xFF;
  const int code = (int)(int8_t)(b == 0x80 ? 0x81 : b);
  const float x = __fmul_rn((float)code, __uint_as_float((123u - (h >> 8) % 5) << 23));
  reinterpret_cast<uint32_t*>(a.p[0])[w] = code ? __float_as_uint(x) | (h >> 16) : 0;
}

// ---------------------------------------------------------------- the clean-up's factors (row-major, 128 bytes a row)
// P_A rows: [bf16(E'_A) || RNE_bf16(A'F_B)].  p0 E'_A codes (rows x 32), p2 AF words (rows x 32), p3 out; i0 rows
extern "C" __global__ void pa(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] * 64) return;
  const int row = t / 64, e = t % 64;
  const uint16_t v = e < 32 ? e4m3bf(reinterpret_cast<const uint8_t*>(a.p[0])[row * 32 + e])
                            : f2bf(reinterpret_cast<const float*>(a.p[2])[row * 32 + e - 32]);
  reinterpret_cast<uint16_t*>(a.p[3])[(size_t)row * 64 + e] = v;
}

// P_B rows: [RNE_bf16(FP32 add(fma chain from +0 over p of be_p G_pq, -BF_jq)) || -be], be = bf16(E'_B).
// p0 E'_B codes, p2 BF words (rows x 32), p3 G words (32 x 32, row p), p4 out; i0 rows
extern "C" __global__ void pb(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] * 32) return;
  const int row = t / 32, q = t % 32;
  const uint8_t* E = reinterpret_cast<const uint8_t*>(a.p[0]) + (size_t)row * 32;
  const float* G = reinterpret_cast<const float*>(a.p[3]);
  float acc = 0.0f;
  for (int p = 0; p < 32; p++) acc = __fmaf_rn(bf2f(e4m3bf(E[p])), G[p * 32 + q], acc);
  const uint16_t mid = f2bf(__fadd_rn(acc, -reinterpret_cast<const float*>(a.p[2])[row * 32 + q]));
  uint16_t* out = reinterpret_cast<uint16_t*>(a.p[4]) + (size_t)row * 64;
  out[q] = mid;
  out[32 + q] = e4m3bf(E[q]) ^ 0x8000;
}

// ---------------------------------------------------------------- -h1: E_A's line key from A's committed root
// The noise is keyed on the device's own commitment of A (pearl_c.PearlC.noise): seed_A = H_"pearl-c/v0/seed-A"(salt ‖ root_A ‖
// root_B ‖ u64 index), H_l(x) = BLAKE3(x; key = BLAKE3(l)), and E_A's line key is keyed(seed_A, "pearl/v4/FP8/noise-line")
// (pearl_kw._line_key).  p0 root_A (the last node of A's tree), p1 the unit's block: salt (32) ‖ root_B (32) ‖ u64 index ‖
// BLAKE3("pearl-c/v0/seed-A") (32), p2 out: E_A's line key (32 B), p3 out: seed_A (32 B) or 0.  One thread.
__constant__ uint8_t LINE_LABEL[23] = {'p', 'e', 'a', 'r', 'l', '/', 'v', '4', '/', 'F', 'P', '8', '/',
                                       'n', 'o', 'i', 's', 'e', '-', 'l', 'i', 'n', 'e'};
extern "C" __global__ void __launch_bounds__(32) seed_line_a(const Args a) {
  if (threadIdx.x) return;
  const uint8_t* ra = reinterpret_cast<const uint8_t*>(a.p[0]);
  const uint8_t* unit = reinterpret_cast<const uint8_t*>(a.p[1]);
  uint32_t key[8], seed[8], h[8];
  load_key(a.p[1] + 72, key);
  b3_short<104>(key, [&](int j) { return (uint32_t)(j < 32 ? unit[j] : j < 64 ? ra[j - 32] : unit[j - 32]); }, seed);
  b3_short<23>(seed, [&](int j) { return (uint32_t)LINE_LABEL[j]; }, h);
  store_cv(a.p[2], h);
  if (a.p[3]) store_cv(a.p[3], seed);
}

// words differing between two buffers: p0, p1 (u32), p2 counter (u64); i0 count (in words, < 2^31 per launch)
extern "C" __global__ void diff_words(const Args a) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const bool d = i < (size_t)a.i[0] && reinterpret_cast<const uint32_t*>(a.p[0])[i] != reinterpret_cast<const uint32_t*>(a.p[1])[i];
  const unsigned n = __syncthreads_count(d);
  if (threadIdx.x == 0 && n) atomicAdd(reinterpret_cast<unsigned long long*>(a.p[2]), (unsigned long long)n);
}
