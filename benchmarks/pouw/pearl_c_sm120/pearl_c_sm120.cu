// Pearl-C on the RTX PRO 6000 Blackwell (sm_120a): the kernels of `pearl-c-sm120-v1` (protocols/pouw/verity_pouw/schemes/
// pearl_c.py on pearl_c_device.SM120) and the plain FP8 GEMMs, for run.py and arm.py.  Every kernel takes one Args by value:
// 16 pointers p[] and 16 ints i[].  sm_120 has no wgmma, TMA multicast or tcgen05: the products are warp-level mma.sync
// (kind::f8f6f4 m16n8k32 E4M3 for every FP8 atom, m16n8k16 BF16 for the peel) fed by ldmatrix from shared memory that
// cp.async fills, within 99 KB of shared memory per block.
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

#include "../pouw_hash/pouw_hash.cuh"

struct Args { uint64_t p[16]; int32_t i[16]; };

#define DEV __device__ __forceinline__

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
// A (M x K) and B (N x K) row-major E4M3.  One CTA of 8 warps per TM x BN tile: 128 x 128 (warp tile 64 x 32), 64 x 64
// (16 x 32) for decode's few rows, where 128 x 128 tiles leave most SMs idle, or 128 x 32 (16 x 32) for the 32-column words
// (A'F_B, B~F_A, G) at M a multiple of 128; persistent over tiles in groups of `i[3]` tile
// rows; a 4-deep cp.async pipeline of 64-byte stages (two atoms each).  G is the promotion period: every 32 G products (G / 2
// stages) from +0, promoted by one FADD into the FP32 total; G = 0 is one accumulator over all of K (no promotion).  Every
// output's chain is the same at either tile.  Modes:
//   PLAIN   y = bf16(total sa sb)
//   CT      stores C~ (FP32)
//   PEARLC  stores C~, then four k16 BF16 mma of P_A P_B^T into the total, stores U and y = bf16(U ia ib)
//   PEARLC_S  PEARLC without storing C~ or U (only y): the hash-free control of PEARLC_H
//   PEARLC_H  PEARLC_S, hashing C~ and U from registers instead of storing them: each (row, 32-column chunk)'s message
//             digest of C~ and U (TurboSHAKE128, D = 0x21, into p9 in hash_msg's layout). C~ is kept in the promotion
//             accumulators through the peel; each warp's scratch is its share of the pipeline's stages, free after the main
//             loop.  128 x 128 tiles only (a 64 x 32 warp tile)
//   PEARLC_B  PEARLC_H under hashing format -h1: BLAKE3 keyed by blake3("pearl-c/h1/message") (32 bytes at p10), one chunk of
//             four blocks per message, into p9 in hash.cuh's hash_msg layout
enum { M_PLAIN, M_CT, M_PEARLC, M_PEARLC_S, M_PEARLC_H, M_PEARLC_B };
constexpr int BK = 64, STAGES = 4, NT = 256, PEEL_ROW = 128;
template <int MODE, int TM, int BN> __host__ __device__ constexpr int gemm_smem() {
  return STAGES * (TM + BN) * BK + (MODE == M_PEARLC || MODE >= M_PEARLC_S ? (TM + BN) * PEEL_ROW : 0);
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
  constexpr bool PEEL = MODE == M_PEARLC || MODE >= M_PEARLC_S, HASH = MODE == M_PEARLC_H || MODE == M_PEARLC_B;
  constexpr bool STORE_CU = MODE == M_PEARLC || MODE == M_CT, PROMOTE = G > 0;
  constexpr int PERIOD = PROMOTE ? G / 2 : 1;
  static_assert(G == 0 || G % 2 == 0, "a promotion window is whole 64-byte stages");
  constexpr uint32_t A_ST = TM * BK, B_ST = BN * BK;
  static_assert(WARPS_M * WARPS_N == 8 && MI >= 1 && NI % 2 == 0, "8 warps of 16k x 32 columns");
  constexpr int SCRATCH = WM * WN * 4;  // a warp's share of the stages: the quad transpose's scratch
  static_assert(!HASH || (MI == 4 && NI == 4 && 8 * SCRATCH <= STAGES * (A_ST + B_ST)), "a hashed warp tile is 64 x 32");
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
    uint8_t* const scratch = smem + SCRATCH * warp;
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
      if constexpr (HASH) {
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
          for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mi][ni][e] = total[mi][ni][e];
      }
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
    if constexpr (HASH) {
      uint8_t* dg = reinterpret_cast<uint8_t*>(a.p[9]) + (size_t)(tn * BN + wn * WN) / 32 * 32;
      const int per = N / 32, row = tm * TM + wm * WM;
      auto region = [&](auto pc) {
        constexpr int p = decltype(pc)::value;
        auto frag = [&](int arr, int j, int i) { return arr ? total[2 * p + (j >> 2)][j & 3][i] : acc[2 * p + (j >> 2)][j & 3][i]; };
        uint8_t* const o = dg + (size_t)(row + ph::pearlc::narrow_row(p, lane)) * per * 32;
        if constexpr (MODE == M_PEARLC_H) {
          uint64_t h[4];
          ph::quad_msg_digest<2>(frag, ph::pearlc::D_MESSAGE, reinterpret_cast<uint64_t*>(scratch), h);
          reinterpret_cast<ulonglong2*>(o)[0] = make_ulonglong2(h[0], h[1]);
          reinterpret_cast<ulonglong2*>(o)[1] = make_ulonglong2(h[2], h[3]);
        } else {
          const uint4 k0 = reinterpret_cast<const uint4*>(a.p[10])[0], k1 = reinterpret_cast<const uint4*>(a.p[10])[1];
          const uint32_t key[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
          uint32_t h[8];
          ph::quad_msg_digest_b3<2>(frag, key, reinterpret_cast<uint64_t*>(scratch), h);
          reinterpret_cast<uint4*>(o)[0] = make_uint4(h[0], h[1], h[2], h[3]);
          reinterpret_cast<uint4*>(o)[1] = make_uint4(h[4], h[5], h[6], h[7]);
        }
      };
      region(Int<0>{});
      region(Int<1>{});
    }
    __syncthreads();
  }
}

// G4: pearl-c-sm120-v1 (pearl_c_device.SM120); _u: unpromoted (SM120_UNPROMOTED); gemm_fast is the plain GEMM unpromoted;
// _d: the 64 x 64 tile; gemm_words: the 128 x 32 tile
#define GEMM(NAME, MODE, TM, BN, G)                                                                                 \
  extern "C" __global__ void __launch_bounds__(NT, 1) NAME(const Args a) { gemm_body<MODE, TM, BN, G>(a); } \
  extern "C" __global__ void NAME##_smem(int* out) { *out = gemm_smem<MODE, TM, BN>(); }
GEMM(gemm_plain, M_PLAIN, 128, 128, 4)
GEMM(gemm_fast, M_PLAIN, 128, 128, 0)
GEMM(gemm_ct, M_CT, 128, 128, 4)
GEMM(gemm_pearlc, M_PEARLC, 128, 128, 4)
GEMM(gemm_ct_u, M_CT, 128, 128, 0)
GEMM(gemm_pearlc_u, M_PEARLC, 128, 128, 0)
GEMM(gemm_ct_d, M_CT, 64, 64, 4)
GEMM(gemm_pearlc_d, M_PEARLC, 64, 64, 4)
GEMM(gemm_ct_d_u, M_CT, 64, 64, 0)
GEMM(gemm_pearlc_d_u, M_PEARLC, 64, 64, 0)
GEMM(gemm_words, M_CT, 128, 32, 4)
GEMM(gemm_words_u, M_CT, 128, 32, 0)
GEMM(gemm_pearlc_s, M_PEARLC_S, 128, 128, 4)
GEMM(gemm_pearlc_s_u, M_PEARLC_S, 128, 128, 0)
GEMM(gemm_pearlc_h, M_PEARLC_H, 128, 128, 4)
GEMM(gemm_pearlc_h_u, M_PEARLC_H, 128, 128, 0)
GEMM(gemm_pearlc_b, M_PEARLC_B, 128, 128, 4)
GEMM(gemm_pearlc_b_u, M_PEARLC_B, 128, 128, 0)

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

// ---------------------------------------------------------------- noise lines (Pearl's line rule)
__device__ const uint32_t B3_IV[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
DEV uint32_t rotr(uint32_t x, int n) { return __funnelshift_r(x, x, n); }
DEV void b3g(uint32_t* v, int a, int b, int c, int d, uint32_t x, uint32_t y) {
  v[a] = v[a] + v[b] + x; v[d] = rotr(v[d] ^ v[a], 16); v[c] = v[c] + v[d]; v[b] = rotr(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + y; v[d] = rotr(v[d] ^ v[a], 8); v[c] = v[c] + v[d]; v[b] = rotr(v[b] ^ v[c], 7);
}
// one keyed BLAKE3 hash of a single 64-byte block (CHUNK_START | CHUNK_END | ROOT | KEYED_HASH), the first 32 bytes
DEV void blake3_block(const uint32_t* key, const uint32_t* msg, uint32_t* out) {
  constexpr int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t v[16], m[16], t[16];
#pragma unroll
  for (int i = 0; i < 8; i++) { v[i] = key[i]; v[i + 8] = i < 4 ? B3_IV[i] : 0; }
  v[14] = 64; v[15] = 1 | 2 | 8 | 16;
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = msg[i];
#pragma unroll
  for (int r = 0; r < 7; r++) {
    b3g(v, 0, 4, 8, 12, m[0], m[1]); b3g(v, 1, 5, 9, 13, m[2], m[3]); b3g(v, 2, 6, 10, 14, m[4], m[5]); b3g(v, 3, 7, 11, 15, m[6], m[7]);
    b3g(v, 0, 5, 10, 15, m[8], m[9]); b3g(v, 1, 6, 11, 12, m[10], m[11]); b3g(v, 2, 7, 8, 13, m[12], m[13]); b3g(v, 3, 4, 9, 14, m[14], m[15]);
#pragma unroll
    for (int i = 0; i < 16; i++) t[i] = m[PERM[i]];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = t[i];
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

DEV uint32_t isqrt(uint32_t v) {
  uint32_t r = (uint32_t)sqrt((double)v);
  while ((uint64_t)r * r > v) r--;
  while ((uint64_t)(r + 1) * (r + 1) <= v) r++;
  return r;
}

// p0 the line key (8 words), p1 codes (count x 32, row-major), p2 transposed (32 x count) or 0, p3 transposed into 128 x count
// (rows 32-127 left as they are) or 0; i0 side, i1 factor, i2 count, i3 bf16(32 x the line's L2 norm): 0x4600 for Pearl's 256,
// 0x4400 for 16
extern "C" __global__ void __launch_bounds__(128) lines(const Args a) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x, count = a.i[2];
  if (line >= count) return;
  const uint32_t* key = reinterpret_cast<const uint32_t*>(a.p[0]);
  uint32_t kw[8], msg[16], h[8];
#pragma unroll
  for (int i = 0; i < 8; i++) kw[i] = key[i];
#pragma unroll
  for (int i = 0; i < 16; i++) msg[i] = 0;
  msg[0] = (uint32_t)a.i[0] | (uint32_t)a.i[1] << 8 | ((uint32_t)line & 0xFFFF) << 16;
  msg[1] = (uint32_t)line >> 16;
  blake3_block(kw, msg, h);
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
      for (int j = 0; j < 32; j++) t[(size_t)j * count + line] = c[j];
    }
}

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

// X~ = e4m3(D), D = the atom with alpha x as its accumulator over E' and the F lines: per warp 16 rows x 128 of depth, sixteen
// m16n8k32 atoms (A = the rows' E', B = eight F lines, C = alpha x).  p0 rows (FP32), p1 alpha, p2 E' codes (rows x 32), p3 F
// lines (k x 32), p4 out codes (rows x k); i0 rows, i1 k.  grid (k / 128, rows / 64), 128 threads.  The block's 64 x 128 FP32
// tile is staged by cp.async first (a warp's copy is one row's 512 bytes), so every row is read whole and at once.
constexpr int FORM_ROW = 144;   // shared bytes per staged row: 128 codes, padded so the quads' stores hit distinct banks
constexpr int FORM_X = 544;     // shared bytes per staged FP32 row: 512, padded so a half-warp's float2 reads hit 32 banks
extern "C" __global__ void __launch_bounds__(128) form_s5(const Args a) {
  __shared__ __align__(16) uint8_t sO[64 * FORM_ROW];
  __shared__ __align__(16) uint8_t sX[64 * FORM_X];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4, k = a.i[1];
  const int t0 = blockIdx.x * 128, r0 = blockIdx.y * 64, rw = r0 + 16 * warp;
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
  const uint8_t* x0 = sX + (16 * warp + g) * FORM_X + 8 * q;
  const uint8_t* x1 = x0 + 8 * FORM_X;
#pragma unroll 4
  for (int nb = 0; nb < 16; nb++) {
    const uint8_t* line = F + (size_t)(t0 + 8 * nb + g) * 32 + 4 * q;
    const uint32_t fb[2] = {ld32(line), ld32(line + 16)};
    const float2 v0 = *reinterpret_cast<const float2*>(x0 + 32 * nb), v1 = *reinterpret_cast<const float2*>(x1 + 32 * nb);
    float d[4] = {__fmul_rn(al0, v0.x), __fmul_rn(al0, v0.y), __fmul_rn(al1, v1.x), __fmul_rn(al1, v1.y)};
    qmma(d, fa, fb);
    const int c = 8 * nb + 2 * q;
    *reinterpret_cast<uint16_t*>(sO + (16 * warp + g) * FORM_ROW + c) = e4m3(d[0]) | (uint16_t)e4m3(d[1]) << 8;
    *reinterpret_cast<uint16_t*>(sO + (16 * warp + g + 8) * FORM_ROW + c) = e4m3(d[2]) | (uint16_t)e4m3(d[3]) << 8;
  }
  __syncthreads();
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[4]);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int idx = tid + 128 * j, r = idx / 8, c = idx % 8;
    *reinterpret_cast<uint4*>(out + (size_t)(r0 + r) * k + t0 + 16 * c) = *reinterpret_cast<const uint4*>(sO + r * FORM_ROW + 16 * c);
  }
}

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

// ---------------------------------------------------------------- hashing: TurboSHAKE128 (Keccak-p[1600, 12], rate 168)
// #449's device code, unfused: the digests read C~ and U back from global memory after the GEMM.
DEV uint64_t rol64(uint64_t x, int n) { return n ? (x << n) | (x >> (64 - n)) : x; }
template <int LUT> DEV uint32_t lop3_32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(LUT));
  return r;
}
template <int LUT> DEV uint64_t lop3_64(uint64_t a, uint64_t b, uint64_t c) {
  return (uint64_t)lop3_32<LUT>((uint32_t)(a >> 32), (uint32_t)(b >> 32), (uint32_t)(c >> 32)) << 32 |
         lop3_32<LUT>((uint32_t)a, (uint32_t)b, (uint32_t)c);
}
DEV uint64_t x3(uint64_t a, uint64_t b, uint64_t c) { return lop3_64<0x96>(a, b, c); }
DEV uint64_t chi(uint64_t a, uint64_t b, uint64_t c) { return lop3_64<0xD2>(a, b, c); }
DEV void keccak12(uint64_t* s) {
  constexpr uint64_t RC[12] = {0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
                               0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
                               0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
  constexpr int rotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
  constexpr int piln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
#pragma unroll
  for (int r = 0; r < 12; r++) {
    uint64_t bc[5];
#pragma unroll
    for (int i = 0; i < 5; i++) bc[i] = x3(x3(s[i], s[i + 5], s[i + 10]), s[i + 15], s[i + 20]);
#pragma unroll
    for (int i = 0; i < 5; i++) {
      const uint64_t p = bc[(i + 4) % 5], q = rol64(bc[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) s[j + i] = x3(s[j + i], p, q);
    }
    uint64_t t = s[1];
#pragma unroll
    for (int i = 0; i < 24; i++) { const int j = piln[i]; const uint64_t x = s[j]; s[j] = rol64(t, rotc[i]); t = x; }
#pragma unroll
    for (int j = 0; j < 25; j += 5) {
      const uint64_t b0 = s[j], b1 = s[j + 1], b2 = s[j + 2], b3 = s[j + 3], b4 = s[j + 4];
      s[j] = chi(b0, b1, b2); s[j + 1] = chi(b1, b2, b3); s[j + 2] = chi(b2, b3, b4); s[j + 3] = chi(b3, b4, b0); s[j + 4] = chi(b4, b0, b1);
    }
    s[0] ^= RC[r];
  }
}
// TurboSHAKE128 of NL little-endian 64-bit lanes (lane(i) = message bytes 8i..8i+7), 32 bytes out
template <int NL, uint64_t DS, class Lane> DEV void turboshake(const Lane& lane, uint64_t* out) {
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; i++) s[i] = 0;
#pragma unroll 1
  for (int b = 0; b < NL / 21; b++) {
#pragma unroll
    for (int w = 0; w < 21; w++) s[w] ^= lane(b * 21 + w);
    keccak12(s);
  }
  constexpr int rem = NL % 21;
#pragma unroll
  for (int w = 0; w < rem; w++) s[w] ^= lane(NL / 21 * 21 + w);
  s[rem] ^= DS;
  s[20] ^= 0x8000000000000000ULL;
  keccak12(s);
#pragma unroll
  for (int i = 0; i < 4; i++) out[i] = s[i];
}
DEV void store32(uint8_t* p, const uint64_t* h) {
  reinterpret_cast<ulonglong2*>(p)[0] = make_ulonglong2(h[0], h[1]);
  reinterpret_cast<ulonglong2*>(p)[1] = make_ulonglong2(h[2], h[3]);
}

// TurboSHAKE128 of a warp's 32 messages, one a lane, each NL little-endian 64-bit lanes, staged 16 lanes (128 bytes) at a time
// through `ring` (the warp's 4 KB): chunk c of message s is the 8 16-byte pieces at chunk(c) + s * stride.  A load covers 4
// messages' 128 bytes, and the next chunk's loads are in flight while this one is absorbed.  Piece p of message s sits at slot
// p ^ (s & 7) of its 128-byte row, so the stores and the lanes' reads spread over the banks.  Messages past `live` read zeros.
template <int NL, uint64_t DS, class Chunk>
DEV void turboshake_warp(const Chunk& chunk, size_t stride, int live, uint4* ring, uint64_t* out) {
  static_assert(NL % 16 == 0, "whole chunks");
  const int lane = threadIdx.x % 32;
  uint4 pre[8];
  auto fetch = [&](int c) {
    const uint4* base = chunk(c);
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const int s = lane / 8 + 4 * r;
      pre[r] = s < live ? base[s * stride + lane % 8] : make_uint4(0, 0, 0, 0);
    }
  };
  auto stage = [&]() {
    __syncwarp();
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const int s = lane / 8 + 4 * r;
      ring[s * 8 + ((lane % 8) ^ (s & 7))] = pre[r];
    }
    __syncwarp();
  };
  const uint64_t* mine = reinterpret_cast<const uint64_t*>(ring + lane * 8);
  auto word = [&](int i) {
    if ((i & 15) == 0) {
      stage();
      if (i + 16 < NL) fetch((i >> 4) + 1);
    }
    return mine[2 * (((i & 15) >> 1) ^ (lane & 7)) + (i & 1)];
  };
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; i++) s[i] = 0;
  fetch(0);
#pragma unroll 1
  for (int b = 0; b < NL / 21; b++) {
#pragma unroll
    for (int w = 0; w < 21; w++) s[w] ^= word(b * 21 + w);
    keccak12(s);
  }
  constexpr int rem = NL % 21;
#pragma unroll
  for (int w = 0; w < rem; w++) s[w] ^= word(NL / 21 * 21 + w);
  s[rem] ^= DS;
  s[20] ^= 0x8000000000000000ULL;
  keccak12(s);
#pragma unroll
  for (int i = 0; i < 4; i++) out[i] = s[i];
}

// message digests: TurboSHAKE128(le32 C~ of the 32-column chunk || le32 U of the chunk, D = 0x21) per (row, chunk); message t
// is C~'s and U's 128 bytes at 128 t.  A warp hashes 32 consecutive messages.
// p0 C~ (rows x cols), p1 U, p2 out (rows x cols/32 x 32 B); i0 rows, i1 cols
extern "C" __global__ void __launch_bounds__(128) hash_msg(const Args a) {
  __shared__ uint4 ring[4][256];
  const size_t total = (size_t)a.i[0] * (a.i[1] / 32), t0 = (size_t)(blockIdx.x * 4 + threadIdx.x / 32) * 32, t = t0 + threadIdx.x % 32;
  if (t0 >= total) return;
  const uint4* c = reinterpret_cast<const uint4*>(a.p[0]) + t0 * 8;
  const uint4* u = reinterpret_cast<const uint4*>(a.p[1]) + t0 * 8;
  uint64_t h[4];
  turboshake_warp<32, 0x21>([&](int ch) { return ch ? u : c; }, 8, (int)min((size_t)32, total - t0), ring[threadIdx.x / 32], h);
  if (t < total) store32(reinterpret_cast<uint8_t*>(a.p[2]) + t * 32, h);
}

// tile leaves: TurboSHAKE128 of a 64 x 64 tile's 128 message digests (rows in order, 2 chunks each), D = 0x22.
// p0 digests, p1 out (tiles row-major x 32 B); i0 rows, i1 cols
extern "C" __global__ void __launch_bounds__(64) hash_leaf(const Args a) {
  const int cols = a.i[1], tn = cols / 64, per = cols / 32;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] / 64 * tn) return;
  const int ti = t / tn, tj = t % tn;
  const uint64_t* dg = reinterpret_cast<const uint64_t*>(a.p[0]);
  uint64_t h[4];
  turboshake<512, 0x22>([&](int i) { return dg[((size_t)(64 * ti + i / 8) * per + 2 * tj + (i / 4) % 2) * 4 + i % 4]; }, h);
  store32(reinterpret_cast<uint8_t*>(a.p[1]) + (size_t)t * 32, h);
}

// A's commitment, modeled: TurboSHAKE128 of each 1,024-byte segment of an FP32 row, D = 0x23.  The rows are contiguous and
// whole segments, so segment t is the 1,024 bytes at 1,024 t; a warp hashes 32 consecutive segments.  p0 rows, p1 out
// (rows x segs x 32 B); i0 rows, i1 row bytes (a multiple of 1,024)
extern "C" __global__ void __launch_bounds__(128) hash_rows_f32(const Args a) {
  __shared__ uint4 ring[4][256];
  const size_t total = (size_t)a.i[0] * (a.i[1] / 1024), t0 = (size_t)(blockIdx.x * 4 + threadIdx.x / 32) * 32, t = t0 + threadIdx.x % 32;
  if (t0 >= total) return;
  const uint4* src = reinterpret_cast<const uint4*>(a.p[0]) + t0 * 64;
  uint64_t h[4];
  turboshake_warp<128, 0x23>([&](int ch) { return src + 8 * ch; }, 64, (int)min((size_t)32, total - t0), ring[threadIdx.x / 32], h);
  if (t < total) store32(reinterpret_cast<uint8_t*>(a.p[1]) + t * 32, h);
}

// a label of n <= 64 bytes as one message block (zero-padded, little-endian words)
DEV void label_block(const char* s, int n, uint32_t w[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) {
    uint32_t x = 0;
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (4 * i + j < n) x |= (uint32_t)(uint8_t)s[4 * i + j] << (8 * j);
    w[i] = x;
  }
}

// Pearl-C's seed_A once A's tree has its root: keyed(blake3("pearl-c/v0/seed-A"), salt ‖ root_A ‖ root_B ‖ index u64), two
// blocks, then from it E_A's line key keyed(seed_A, "pearl/v4/FP8/noise-line") and the jackpot key keyed(seed_A,
// "pearl/v4/FP8/jackpot"), one block each (`pearl_c`'s seeds, `pearl_kw._line_key` and `subkey`), on one thread.
// p0 out (seed_A ‖ line key ‖ jackpot key, 96 B), p1 blake3("pearl-c/v0/seed-A"), p2 salt, p3 root_A, p4 root_B (32 B
// each, 4-byte aligned); i0 the matmul's index
extern "C" __global__ void __launch_bounds__(32) seed_a(const Args a) {
  if (threadIdx.x) return;
  using namespace ph;
  const uint32_t* key = reinterpret_cast<const uint32_t*>(a.p[1]);
  const uint32_t* salt = reinterpret_cast<const uint32_t*>(a.p[2]);
  const uint32_t* root_a = reinterpret_cast<const uint32_t*>(a.p[3]);
  const uint32_t* root_b = reinterpret_cast<const uint32_t*>(a.p[4]);
  uint32_t m[16], cv[8], seed[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = key[i], m[i] = salt[i], m[8 + i] = root_a[i];
  blake3_compress(cv, m, 0, 64, B3_KEYED_HASH | B3_CHUNK_START, cv);
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = i < 8 ? root_b[i] : 0;
  m[8] = (uint32_t)a.i[0];
  blake3_compress(cv, m, 0, 40, B3_KEYED_HASH | B3_CHUNK_END | B3_ROOT, seed);
  uint32_t* out = reinterpret_cast<uint32_t*>(a.p[0]);
  const char line[] = "pearl/v4/FP8/noise-line", jackpot[] = "pearl/v4/FP8/jackpot";
  uint32_t h[8];
  label_block(line, sizeof line - 1, m);
  blake3_compress(seed, m, 0, sizeof line - 1, B3_KEYED_HASH | B3_CHUNK_START | B3_CHUNK_END | B3_ROOT, h);
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = seed[i], out[8 + i] = h[i];
  label_block(jackpot, sizeof jackpot - 1, m);
  blake3_compress(seed, m, 0, sizeof jackpot - 1, B3_KEYED_HASH | B3_CHUNK_START | B3_CHUNK_END | B3_ROOT, h);
#pragma unroll
  for (int i = 0; i < 8; i++) out[16 + i] = h[i];
}

// words differing between two buffers: p0, p1 (u32), p2 counter (u64); i0 count (in words, < 2^31 per launch)
extern "C" __global__ void diff_words(const Args a) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const bool d = i < (size_t)a.i[0] && reinterpret_cast<const uint32_t*>(a.p[0])[i] != reinterpret_cast<const uint32_t*>(a.p[1])[i];
  const unsigned n = __syncthreads_count(d);
  if (threadIdx.x == 0 && n) atomicAdd(reinterpret_cast<unsigned long long*>(a.p[2]), (unsigned long long)n);
}
