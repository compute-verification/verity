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
// A (M x K) and B (N x K) row-major E4M3.  One CTA of 8 warps per 128 x BN tile (warp tile 64 x 32, or 16 x 32 when BN = 32),
// persistent over tiles in groups of `i[3]` tile rows; a 4-deep cp.async pipeline of 64-byte stages (two atoms each).  G is
// the promotion period: every 32 G products (G / 2 stages) from +0, promoted by one FADD into the FP32 total; G = 0 is one
// accumulator over all of K (no promotion).  Modes:
//   PLAIN   y = bf16(total sa sb)
//   CT      stores C~ (FP32)
//   PEARLC  stores C~, then four k16 BF16 mma of P_A P_B^T into the total, stores U and y = bf16(U ia ib)
//   WORDS   BN = 32, stores the FP32 words (A'F_B, B~F_A, G)
enum { M_PLAIN, M_CT, M_PEARLC, M_WORDS };
constexpr int BM = 128, BK = 64, STAGES = 4, NT = 256, PEEL_ROW = 128;
template <int MODE, int BN> __host__ __device__ constexpr int gemm_smem() {
  return STAGES * (BM + BN) * BK + (MODE == M_PEARLC ? (BM + BN) * PEEL_ROW : 0);
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

DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

DEV void store2(float* p, float a, float b) { *reinterpret_cast<float2*>(p) = make_float2(a, b); }
DEV void store_y(uint16_t* p, float a, float b, float sr, float sc0, float sc1) {
  *reinterpret_cast<uint32_t*>(p) = f2bf(__fmul_rn(__fmul_rn(a, sr), sc0)) | (uint32_t)f2bf(__fmul_rn(__fmul_rn(b, sr), sc1)) << 16;
}

template <int MODE, int BN, int G> DEV void gemm_body(const Args& a) {
  constexpr int WN = 32, WARPS_N = BN / WN, WARPS_M = 8 / WARPS_N, WM = BM / WARPS_M, MI = WM / 16, NI = WN / 8;
  constexpr bool PEEL = MODE == M_PEARLC, PROMOTE = G > 0;
  constexpr int PERIOD = PROMOTE ? G / 2 : 1;
  static_assert(G == 0 || G % 2 == 0, "a promotion window is whole 64-byte stages");
  constexpr uint32_t A_ST = BM * BK, B_ST = BN * BK;
  static_assert(WARPS_M * WARPS_N == 8 && MI >= 1 && NI % 2 == 0, "8 warps of 16k x 32 columns");
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = smem + STAGES * A_ST;
  uint8_t* sPA = sB + STAGES * B_ST;
  uint8_t* sPB = sPA + BM * PEEL_ROW;
  const int M = a.i[0], N = a.i[1], K = a.i[2], group = a.i[3];
  const int tiles_m = M / BM, tiles_n = N / BN, tiles = tiles_m * tiles_n, KB = K / BK;
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
    const uint8_t* ga = A + (size_t)tm * BM * K;
    const uint8_t* gb = B + (size_t)tn * BN * K;
    if constexpr (PEEL) {
      peel_rows<BM>(sPA, reinterpret_cast<const uint8_t*>(a.p[5]) + (size_t)tm * BM * PEEL_ROW, tid);
      peel_rows<BN>(sPB, reinterpret_cast<const uint8_t*>(a.p[6]) + (size_t)tn * BN * PEEL_ROW, tid);
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < KB) {
        stage_rows<BM>(sA + s * A_ST, ga + s * BK, K, tid);
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
          stage_rows<BM>(sA + s * A_ST, ga + (size_t)nk * BK, K, tid);
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

    const int row0 = tm * BM + wm * WM + g, col0 = tn * BN + wn * WN + 2 * q;
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
    } else {
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
          store2(U + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1]);
          store2(U + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3]);
          store_y(Y + (size_t)row * N + col, total[mi][ni][0], total[mi][ni][1], r0, c0, c1);
          store_y(Y + (size_t)(row + 8) * N + col, total[mi][ni][2], total[mi][ni][3], r1, c0, c1);
        }
      }
    }
    __syncthreads();
  }
}

// G4: pearl-c-sm120-v1 (pearl_c_device.SM120); _u: unpromoted (SM120_UNPROMOTED); gemm_fast is the plain GEMM unpromoted
#define GEMM(NAME, MODE, BN, G)                                                                                 \
  extern "C" __global__ void __launch_bounds__(NT, 1) NAME(const Args a) { gemm_body<MODE, BN, G>(a); } \
  extern "C" __global__ void NAME##_smem(int* out) { *out = gemm_smem<MODE, BN>(); }
GEMM(gemm_plain, M_PLAIN, 128, 4)
GEMM(gemm_fast, M_PLAIN, 128, 0)
GEMM(gemm_ct, M_CT, 128, 4)
GEMM(gemm_pearlc, M_PEARLC, 128, 4)
GEMM(gemm_words, M_WORDS, 32, 4)
GEMM(gemm_ct_u, M_CT, 128, 0)
GEMM(gemm_pearlc_u, M_PEARLC, 128, 0)
GEMM(gemm_words_u, M_WORDS, 32, 0)

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

// one warp per FP32 row: s = max|x|, rho (32 partial FMA chains over every 8th position, then the xor butterfly), alpha,
// beta_row.  p0 rows (rows x k words), p1 alpha, p2 beta, p3 1/alpha (glue), p4 (s, rho) per row; i0 rows, i1 k
extern "C" __global__ void __launch_bounds__(256) stats_s5(const Args a) {
  const int row = (blockIdx.x * 256 + threadIdx.x) / 32, lane = threadIdx.x % 32, rows = a.i[0], k = a.i[1];
  if (row >= rows) return;
  const float* x = reinterpret_cast<const float*>(a.p[0]) + (size_t)row * k;
  float s = 0.0f, part = 0.0f;
  for (int j = lane * 4; j < k; j += 128) {
    const float4 v = *reinterpret_cast<const float4*>(x + j);
    s = fmaxf(s, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
  }
  for (int p = 8 * lane; p < k; p += 256) part = __fmaf_rn(x[p], x[p], part);
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
// lines (k x 32), p4 out codes (rows x k); i0 rows, i1 k.  grid (k / 128, rows / 64), 128 threads
constexpr int FORM_ROW = 144;   // shared bytes per staged row: 128 codes, padded so the quads' stores hit distinct banks
extern "C" __global__ void __launch_bounds__(128) form_s5(const Args a) {
  __shared__ __align__(16) uint8_t sO[64 * FORM_ROW];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4, k = a.i[1];
  const int t0 = blockIdx.x * 128, r0 = blockIdx.y * 64, rw = r0 + 16 * warp;
  const uint8_t* E = reinterpret_cast<const uint8_t*>(a.p[2]);
  const uint8_t* F = reinterpret_cast<const uint8_t*>(a.p[3]);
  const float* x = reinterpret_cast<const float*>(a.p[0]);
  const float* alpha = reinterpret_cast<const float*>(a.p[1]);
  const uint32_t fa[4] = {ld32(E + (size_t)(rw + g) * 32 + 4 * q), ld32(E + (size_t)(rw + g + 8) * 32 + 4 * q),
                          ld32(E + (size_t)(rw + g) * 32 + 16 + 4 * q), ld32(E + (size_t)(rw + g + 8) * 32 + 16 + 4 * q)};
  const float al0 = alpha[rw + g], al1 = alpha[rw + g + 8];
  const float* x0 = x + (size_t)(rw + g) * k + t0 + 2 * q;
  const float* x1 = x0 + (size_t)8 * k;
#pragma unroll 4
  for (int nb = 0; nb < 16; nb++) {
    const uint8_t* line = F + (size_t)(t0 + 8 * nb + g) * 32 + 4 * q;
    const uint32_t fb[2] = {ld32(line), ld32(line + 16)};
    const float2 v0 = *reinterpret_cast<const float2*>(x0 + 8 * nb), v1 = *reinterpret_cast<const float2*>(x1 + 8 * nb);
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

// message digests: TurboSHAKE128(le32 C~ of the 32-column chunk || le32 U of the chunk, D = 0x21) per (row, chunk).
// p0 C~ (rows x cols), p1 U, p2 out (rows x cols/32 x 32 B); i0 rows, i1 cols
extern "C" __global__ void __launch_bounds__(128) hash_msg(const Args a) {
  const int cols = a.i[1], per = cols / 32;
  const size_t m = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= (size_t)a.i[0] * per) return;
  const uint64_t* c = reinterpret_cast<const uint64_t*>(a.p[0]) + m * 16;
  const uint64_t* u = reinterpret_cast<const uint64_t*>(a.p[1]) + m * 16;
  uint64_t h[4];
  turboshake<32, 0x21>([&](int i) { return i < 16 ? c[i] : u[i - 16]; }, h);
  store32(reinterpret_cast<uint8_t*>(a.p[2]) + m * 32, h);
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

// A's commitment, modeled: TurboSHAKE128 of each 1,024-byte segment of an FP32 row, D = 0x23.  p0 rows, p1 out (rows x segs x
// 32 B); i0 rows, i1 row bytes
extern "C" __global__ void __launch_bounds__(128) hash_rows_f32(const Args a) {
  constexpr int NL = 128;
  const int rs = a.i[1], segs = rs / (8 * NL);
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)a.i[0] * segs) return;
  const uint64_t* src = reinterpret_cast<const uint64_t*>(a.p[0] + (t / segs) * rs + (t % segs) * 8 * NL);
  uint64_t h[4];
  turboshake<NL, 0x23>([&](int i) { return src[i]; }, h);
  store32(reinterpret_cast<uint8_t*>(a.p[1]) + t * 32, h);
}

// words differing between two buffers: p0, p1 (u32), p2 counter (u64); i0 count (in words, < 2^31 per launch)
extern "C" __global__ void diff_words(const Args a) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const bool d = i < (size_t)a.i[0] && reinterpret_cast<const uint32_t*>(a.p[0])[i] != reinterpret_cast<const uint32_t*>(a.p[1])[i];
  const unsigned n = __syncthreads_count(d);
  if (threadIdx.x == 0 && n) atomicAdd(reinterpret_cast<unsigned long long*>(a.p[2]), (unsigned long long)n);
}
