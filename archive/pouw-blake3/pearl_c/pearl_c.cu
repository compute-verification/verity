// Pearl-C on the H100 (sm_90a): the kernels of `pearl-c-h100-v0` (verity/core/protocols/pouw/schemes/pearl_c.py) and the
// plain FP8 baseline, for run.py.  Every kernel takes one Args by value: 16 pointers p[] and 16 ints i[].
//
// Tiled operand layout (both GEMM operands, the forming inputs E and F, and the peel factors): rows are cut into blocks of
// H rows (H = 128, or 32 for the 32-row F^T operand), depth into 32-byte chunks; chunk (rb, kc) is H x 32 bytes at
// ((rb * (depth / 32)) + kc) * H * 32, and byte k of row r inside it is at (r >> 3) * 256 + (k >> 4) * 128 + (r & 7) * 16 +
// (k & 15): K-major without swizzle, core matrices 8 rows x 16 B, LBO 128, SBO 256.  One chunk is one k32 E4M3 atom's (or
// one k16 BF16 step's) operand for H rows, and 4 consecutive chunks (a 128-deep stage) are contiguous.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "acc.h"

#include "hash.cuh"   // Args, DEV, BLAKE3 and the hashing kernels (also built alone for sm_120a)
#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

// ---------------------------------------------------------------- formats (IEEE, nearest even, no FTZ)
DEV float bf2f(uint32_t b) { return __uint_as_float((b & 0xFFFFu) << 16); }
DEV uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
DEV uint16_t bfma(uint16_t a, uint16_t b, uint16_t c) {
  return __bfloat16_as_ushort(__hfma(__ushort_as_bfloat16(a), __ushort_as_bfloat16(b), __ushort_as_bfloat16(c)));
}
DEV uint16_t bmul(uint16_t a, uint16_t b) { return f2bf(__fmul_rn(bf2f(a), bf2f(b))); }
DEV uint16_t bdiv(uint16_t a, uint16_t b) { return f2bf(__fdiv_rn(bf2f(a), bf2f(b))); }
DEV uint16_t bmax(uint16_t a, uint16_t b) { return bf2f(a) >= bf2f(b) ? a : b; }
DEV uint8_t e4m3(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); }
// two codes from one F2FP (both operands live): lo in the low byte
DEV uint16_t e4m3x2(float lo, float hi) { return __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3); }
DEV float e4m3f(uint8_t c) { return __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3))); }
DEV uint16_t e4m3bf(uint8_t c) { return f2bf(e4m3f(c)); }

constexpr uint16_t BF_QMAX = 0x43E0;       // 448
constexpr uint16_t BF_DELTA_R = 0x40B5;    // bf16(f32(delta sqrt r)), delta = 1, r = 32
constexpr uint16_t BF_DELTA_STD = 0x38B5;  // bf16(f32(delta sqrt r / 256^2))
constexpr uint16_t BF_FLOOR = 0x2F80;      // 2^-32

DEV size_t tiled(int row, int k, int H, int depth) {
  return ((size_t)(row / H) * (depth / 32) + k / 32) * (H * 32) + ((row % H) >> 3) * 256 + ((k & 31) >> 4) * 128 + (row & 7) * 16 +
         (k & 15);
}

DEV uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}

// ---------------------------------------------------------------- mbarriers, bulk copies, wgmma
DEV uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mb_init(uint64_t* b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(n) : "memory"); }
DEV void mb_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sa(b)) : "memory"); }
DEV void mb_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(bytes) : "memory");
}
DEV void mb_wait(uint64_t* b, uint32_t parity) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W;}" ::"r"(sa(b)), "r"(parity) : "memory");
}
DEV void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}
DEV uint64_t desc(uint32_t addr) {
  return (uint64_t)((addr >> 4) & 0x3FFF) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
}

template <int BN, int SD> DEV void qmma(float* d, uint64_t da, uint64_t db) {
  if constexpr (BN == 128)
    asm volatile("{.reg .pred p; setp.ne.b32 p, %66, 0; wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " ACC64_S
                 ", %64, %65, p, 1, 1;}" : ACC64_F : "l"(da), "l"(db), "n"(SD));
  else
    asm volatile("{.reg .pred p; setp.ne.b32 p, %18, 0; wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 " ACC16_S
                 ", %16, %17, p, 1, 1;}" : ACC16_F : "l"(da), "l"(db), "n"(SD));
}
DEV void qmma_rt(float* d, uint64_t da, uint64_t db, uint32_t sd) {
  asm volatile("{.reg .pred p; setp.ne.b32 p, %66, 0; wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " ACC64_S
               ", %64, %65, p, 1, 1;}" : ACC64_F : "l"(da), "l"(db), "r"(sd));
}
DEV void hmma(float* d, uint64_t da, uint64_t db) {
  asm volatile("{.reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " ACC64_S
               ", %64, %65, p, 1, 1, 0, 0;}" : ACC64_F : "l"(da), "l"(db));
}

// ---------------------------------------------------------------- the GEMM
// A (M x K) and B (N x K) tiled with H = 128 (B: H = BN), E4M3.  One CTA: consumer warpgroups 0 and 1 (rows 0-63, 64-127 of a
// 128 x BN tile), warpgroup 2's first thread the producer (1-D bulk copies of 16 KB stages, STAGES deep).  Persistent over tiles in
// groups of `i[3]` tile rows.  Modes:
//   PLAIN   every 128-deep block (four atoms) from +0, promoted by one FADD into the FP32 total (G4); y = bf16(total sa sb)
//   FAST    one accumulator over all of K (no promotion); y as PLAIN
//   CT      G4, stores C~ (FP32)
//   PEARLC  G4, stores C~, then four k16 BF16 wgmma of P_A P_B^T into the total, stores U and y = bf16(U ia ib)
//   WORDS   G4 with BN = 32, stores the FP32 words (A'F_B, B~F_A, G)
enum { M_PLAIN, M_FAST, M_CT, M_PEARLC, M_WORDS };
constexpr int STAGES = 5, BM = 128, BK = 128, NWG = 2;
template <int MODE, int BN> __host__ __device__ constexpr int gemm_smem() {
  return STAGES * (BM + BN) * BK + (MODE == M_PEARLC ? 2 * BM * 128 : 0) + (2 * STAGES + 2) * 8;
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

template <int MODE, int BN> DEV void gemm_body(const Args& a) {
  constexpr int NA = BN / 2;
  constexpr uint32_t A_BYTES = BM * BK, B_BYTES = BN * BK;
  constexpr bool PEEL = MODE == M_PEARLC;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = smem + STAGES * A_BYTES;
  uint8_t* sP = sB + STAGES * B_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(sP + (PEEL ? 2 * BM * 128 : 0));
  uint64_t* empty = full + STAGES;
  uint64_t* peel_full = empty + STAGES;
  uint64_t* peel_empty = peel_full + 1;
  const int M = a.i[0], N = a.i[1], K = a.i[2], group = a.i[3];
  const int tiles_m = M / BM, tiles_n = N / BN, tiles = tiles_m * tiles_n, KB = K / BK;
  const int tid = threadIdx.x, wg = tid / 128;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mb_init(&full[s], 1); mb_init(&empty[s], NWG * 4); }
    mb_init(peel_full, 1);
    mb_init(peel_empty, NWG * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (wg == NWG) {
    if (tid != NWG * 128) return;
    const uint8_t* A = reinterpret_cast<const uint8_t*>(a.p[0]);
    const uint8_t* B = reinterpret_cast<const uint8_t*>(a.p[1]);
    int stage = 0;
    uint32_t phase = 0, pphase = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      int tm, tn;
      raster(t, tiles_m, tiles_n, group, tm, tn);
      const uint8_t* ga = A + (size_t)tm * (K / 32) * (BM * 32);
      const uint8_t* gb = B + (size_t)tn * (K / 32) * (BN * 32);
      for (int kb = 0; kb < KB; kb++) {
        mb_wait(&empty[stage], phase ^ 1);
        mb_expect(&full[stage], A_BYTES + B_BYTES);
        bulk_g2s(sA + stage * A_BYTES, ga + (size_t)kb * A_BYTES, A_BYTES, &full[stage]);
        bulk_g2s(sB + stage * B_BYTES, gb + (size_t)kb * B_BYTES, B_BYTES, &full[stage]);
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
      }
      if constexpr (PEEL) {
        mb_wait(peel_empty, pphase ^ 1);
        mb_expect(peel_full, 2 * BM * 128);
        bulk_g2s(sP, reinterpret_cast<const uint8_t*>(a.p[5]) + (size_t)tm * BM * 128, BM * 128, peel_full);
        bulk_g2s(sP + BM * 128, reinterpret_cast<const uint8_t*>(a.p[6]) + (size_t)tn * BM * 128, BM * 128, peel_full);
        pphase ^= 1;
      }
    }
    return;
  }

  const int x = tid % 128, lane = tid % 32;
  float* C = reinterpret_cast<float*>(a.p[2]);
  float* U = reinterpret_cast<float*>(a.p[3]);
  uint16_t* Y = reinterpret_cast<uint16_t*>(a.p[4]);
  const float* rs = reinterpret_cast<const float*>(a.p[7]);
  const float* cs = reinterpret_cast<const float*>(a.p[8]);
  int stage = 0;
  uint32_t phase = 0, pphase = 0;
  float d[NA], total[NA];
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    int tm, tn;
    raster(t, tiles_m, tiles_n, group, tm, tn);
    if constexpr (MODE == M_FAST) {
      for (int kb = 0; kb < KB; kb++) {
        mb_wait(&full[stage], phase);
        const uint32_t a0 = sa(sA + stage * A_BYTES) + wg * 2048, b0 = sa(sB + stage * B_BYTES);
        WG_FENCE;
        qmma_rt(d, desc(a0), desc(b0), kb > 0);
        qmma<BN, 1>(d, desc(a0 + BM * 32), desc(b0 + BN * 32));
        qmma<BN, 1>(d, desc(a0 + 2 * BM * 32), desc(b0 + 2 * BN * 32));
        qmma<BN, 1>(d, desc(a0 + 3 * BM * 32), desc(b0 + 3 * BN * 32));
        WG_COMMIT;
        WG_WAIT(1);
        if (kb > 0 && lane == 0) mb_arrive(&empty[stage == 0 ? STAGES - 1 : stage - 1]);
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
      }
      WG_WAIT(0);
      if (lane == 0) mb_arrive(&empty[stage == 0 ? STAGES - 1 : stage - 1]);
#pragma unroll
      for (int i = 0; i < NA; i++) total[i] = d[i];
    } else {
#pragma unroll
      for (int i = 0; i < NA; i++) total[i] = 0.0f;
      for (int kb = 0; kb < KB; kb++) {
        mb_wait(&full[stage], phase);
        const uint32_t a0 = sa(sA + stage * A_BYTES) + wg * 2048, b0 = sa(sB + stage * B_BYTES);
        WG_FENCE;
        qmma<BN, 0>(d, desc(a0), desc(b0));
        qmma<BN, 1>(d, desc(a0 + BM * 32), desc(b0 + BN * 32));
        qmma<BN, 1>(d, desc(a0 + 2 * BM * 32), desc(b0 + 2 * BN * 32));
        qmma<BN, 1>(d, desc(a0 + 3 * BM * 32), desc(b0 + 3 * BN * 32));
        WG_COMMIT;
        WG_WAIT(0);
        if (lane == 0) mb_arrive(&empty[stage]);
#pragma unroll
        for (int i = 0; i < NA; i++) total[i] = __fadd_rn(total[i], d[i]);
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
      }
    }

    const int row = tm * BM + wg * 64 + 16 * (x / 32) + (x % 32) / 4, col0 = tn * BN + 2 * (x % 4);
    if constexpr (MODE == M_PLAIN || MODE == M_FAST) {
      const float r0 = rs[row], r1 = rs[row + 8];
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        const float c0 = cs[col], c1 = cs[col + 1];
        store_y(Y + (size_t)row * N + col, total[4 * j], total[4 * j + 1], r0, c0, c1);
        store_y(Y + (size_t)(row + 8) * N + col, total[4 * j + 2], total[4 * j + 3], r1, c0, c1);
      }
    } else {
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        store2(C + (size_t)row * N + col, total[4 * j], total[4 * j + 1]);
        store2(C + (size_t)(row + 8) * N + col, total[4 * j + 2], total[4 * j + 3]);
      }
    }
    if constexpr (PEEL) {
      mb_wait(peel_full, pphase);
      const uint32_t pa0 = sa(sP) + wg * 2048, pb0 = sa(sP + BM * 128);
      WG_FENCE;
      hmma(total, desc(pa0), desc(pb0));
      hmma(total, desc(pa0 + 4096), desc(pb0 + 4096));
      hmma(total, desc(pa0 + 2 * 4096), desc(pb0 + 2 * 4096));
      hmma(total, desc(pa0 + 3 * 4096), desc(pb0 + 3 * 4096));
      WG_COMMIT;
      WG_WAIT(0);
      if (lane == 0) mb_arrive(peel_empty);
      pphase ^= 1;
      const float r0 = rs[row], r1 = rs[row + 8];
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        const float c0 = cs[col], c1 = cs[col + 1];
        store2(U + (size_t)row * N + col, total[4 * j], total[4 * j + 1]);
        store2(U + (size_t)(row + 8) * N + col, total[4 * j + 2], total[4 * j + 3]);
        store_y(Y + (size_t)row * N + col, total[4 * j], total[4 * j + 1], r0, c0, c1);
        store_y(Y + (size_t)(row + 8) * N + col, total[4 * j + 2], total[4 * j + 3], r1, c0, c1);
      }
    }
  }
}

#define GEMM(NAME, MODE, BN)                                                                                 \
  extern "C" __global__ void __launch_bounds__(384, 1) NAME(const Args a) { gemm_body<MODE, BN>(a); } \
  extern "C" __global__ void NAME##_smem(int* out) { *out = gemm_smem<MODE, BN>(); }
GEMM(gemm_plain, M_PLAIN, 128)
GEMM(gemm_fast, M_FAST, 128)
GEMM(gemm_ct, M_CT, 128)
GEMM(gemm_pearlc, M_PEARLC, 128)
GEMM(gemm_words, M_WORDS, 32)

// ---------------------------------------------------------------- noise lines (Pearl's line rule)
DEV uint32_t isqrt(uint32_t v) {
  uint32_t r = (uint32_t)sqrt((double)v);
  while ((uint64_t)r * r > v) r--;
  while ((uint64_t)(r + 1) * (r + 1) <= v) r++;
  return r;
}

// p0 the line key (8 words), p1 codes (count x 32, row-major), p2 rows tiled (H = 128, depth 32) or 0, p3 transposed tiled
// (32 x count, H = 32) or 0, p4 transposed tiled (128 x count, H = 128, rows 32-127 left as they are) or 0; i0 side, i1 factor,
// i2 count, i3 bf16(32 x the line's L2 norm): 0x4600 for Pearl's 256, 0x4400 for 16
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
  if (a.p[2]) {
    uint8_t* t = reinterpret_cast<uint8_t*>(a.p[2]);
    *reinterpret_cast<uint4*>(t + tiled(line, 0, 128, 32)) = make_uint4(w[0], w[1], w[2], w[3]);
    *reinterpret_cast<uint4*>(t + tiled(line, 16, 128, 32)) = make_uint4(w[4], w[5], w[6], w[7]);
  }
  if (a.p[3]) {
    uint8_t* t = reinterpret_cast<uint8_t*>(a.p[3]);
#pragma unroll
    for (int j = 0; j < 32; j++) t[tiled(j, line, 32, count)] = c[j];
  }
  if (a.p[4]) {
    uint8_t* t = reinterpret_cast<uint8_t*>(a.p[4]);
#pragma unroll
    for (int j = 0; j < 32; j++) t[tiled(j, line, 128, count)] = c[j];
  }
}

// ---------------------------------------------------------------- forming v0 (#311's, as ported)
DEV int cmp_mid(uint32_t c, uint64_t total, uint32_t k, int E) {
  // sign of mid(c, c + 1)^2 - total 2^E / k, exactly: mid = (2S + 1) 2^(q - 1) for the code's significand S and value S 2^q
  const int ec = (c >> 7) & 0xFF;
  const uint64_t S = ec ? 128 + (c & 0x7F) : (c & 0x7F);
  const int q = ec ? ec - 134 : -133;
  const unsigned __int128 L = (unsigned __int128)((2 * S + 1) * (2 * S + 1)) * k;
  const int D = E - 2 * q + 2;
  if (D >= 0) {
    if (D > 64) return -1;
    const unsigned __int128 R = (unsigned __int128)total << D;
    return L < R ? -1 : L > R ? 1 : 0;
  }
  if (-D > 60) return 1;
  const unsigned __int128 LL = L << (-D);
  return LL < total ? -1 : LL > total ? 1 : 0;
}
DEV uint32_t bf16_sqrt_exact(uint64_t total, uint32_t k, int E) {
  const double y = fmax(sqrt(ldexp((double)total, E) / (double)k), ldexp(1.0, -140));
  uint32_t code = f2bf((float)y);
  for (int it = 0; it < 64; it++) {
    bool lo_ok = code == 0, hi_ok;
    if (!lo_ok) { const int c = cmp_mid(code - 1, total, k, E); lo_ok = c < 0 || (c == 0 && !(code & 1)); }
    { const int c = cmp_mid(code, total, k, E); hi_ok = c > 0 || (c == 0 && !(code & 1)); }
    if (lo_ok && hi_ok) break;
    code += lo_ok ? 1 : -1;
  }
  return code;
}

// one warp per FP10 row (k codes, then k/8 BF16 scales): Pearl's exact_norms, then noisy_quantize's alpha and beta.
// p0 rows, p1 alpha (u16), p2 beta (u16), p3 1/alpha (f32, glue), p4 l2 | linf << 16 (u32); i0 rows, i1 k
extern "C" __global__ void __launch_bounds__(256) norms(const Args a) {
  const int row = (blockIdx.x * 256 + threadIdx.x) / 32, lane = threadIdx.x % 32, rows = a.i[0], k = a.i[1];
  if (row >= rows) return;
  const uint8_t* X = reinterpret_cast<const uint8_t*>(a.p[0]) + (size_t)row * (k + k / 4);
  const uint16_t* sc = reinterpret_cast<const uint16_t*>(X + k);
  int emax = 0;
  float linf = 0.0f;
  for (int b = lane; b < k / 8; b += 32) {
    const uint2 w = *reinterpret_cast<const uint2*>(X + 8 * b);
    int amax = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) amax = max(amax, abs((int)(int8_t)(((j < 4 ? w.x : w.y) >> (8 * (j % 4))) & 0xFF)));
    const uint32_t s = sc[b], e = (s & 0x7FFF) >> 7, m = (s & 0x7F) + (e ? 128 : 0);
    if (amax && m) emax = max(emax, 2 * (int)max(e, 1u));
    linf = fmaxf(linf, __fmul_rn((float)amax, bf2f(s & 0x7FFF)));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { emax = max(emax, __shfl_xor_sync(~0u, emax, o)); linf = fmaxf(linf, __shfl_xor_sync(~0u, linf, o)); }
  const int wl2 = k > 1 ? 27 - (32 - __clz(k - 1)) : 27;
  uint64_t total = 0;
  for (int b = lane; b < k / 8; b += 32) {
    const uint2 w = *reinterpret_cast<const uint2*>(X + 8 * b);
    uint64_t sq = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) { const int v = (int8_t)(((j < 4 ? w.x : w.y) >> (8 * (j % 4))) & 0xFF); sq += v * v; }
    const uint32_t s = sc[b], e = (s & 0x7FFF) >> 7, m = (s & 0x7F) + (e ? 128 : 0);
    const int sh = max(emax - 2 * (int)max(e, 1u), 0);
    if (sh < 64) total += ((uint64_t)m * m * sq << wl2) >> sh;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) total += __shfl_xor_sync(~0u, total, o);
  if (lane) return;
  uint32_t l2 = 0;
  if (total) l2 = (bf16_sqrt_exact(total, k, emax - 268 - wl2) + 2) & ~3u & 0xFFFF;
  const uint16_t linf_b = f2bf(linf);
  const uint16_t l2f = bmax(l2, BF_FLOOR), linff = bmax(linf_b, BF_FLOOR);
  const uint16_t alpha = bdiv(BF_QMAX, bfma(BF_DELTA_R, l2f, linff));
  const uint16_t beta = bmul(bmul(alpha, l2f), BF_DELTA_STD);
  reinterpret_cast<uint16_t*>(a.p[1])[row] = alpha;
  reinterpret_cast<uint16_t*>(a.p[2])[row] = beta;
  reinterpret_cast<float*>(a.p[3])[row] = __frcp_rn(bf2f(alpha));
  reinterpret_cast<uint32_t*>(a.p[4])[row] = l2 | (uint32_t)linf_b << 16;
}

// X~ = fp8(clamp(fma(alpha, X, bf16(beta bf16(N))), +-448)) with N = E F^T on the H100 atom, per 64 rows x 128 of depth.
// p0 FP10 rows, p1 alpha, p2 beta, p3 E tiled (H = 128, depth 32), p4 F tiled (H = 128, depth 32; lines are rows),
// p5 out codes tiled (H = 128, depth k); i0 rows, i1 k.  grid (k / 128, rows / 64), 128 threads
extern "C" __global__ void __launch_bounds__(128) form(const Args a) {
  __shared__ __align__(128) uint8_t sE[2048], sF[4096], sX[64 * 128], sO[4 * 2048];
  __shared__ __align__(16) uint16_t sS[64 * 16];
  __shared__ uint16_t sAl[64], sBe[64];
  const int tid = threadIdx.x, t0 = blockIdx.x * 128, r0 = blockIdx.y * 64, k = a.i[1], rs = k + k / 4;
  const uint8_t* X = reinterpret_cast<const uint8_t*>(a.p[0]);
  const uint4* E = reinterpret_cast<const uint4*>(a.p[3] + (size_t)(r0 / 128) * 4096 + (r0 % 128) / 64 * 2048);
  const uint4* F = reinterpret_cast<const uint4*>(a.p[4] + (size_t)(t0 / 128) * 4096);
  reinterpret_cast<uint4*>(sE)[tid] = E[tid];
  reinterpret_cast<uint4*>(sF)[tid] = F[tid];
  reinterpret_cast<uint4*>(sF)[tid + 128] = F[tid + 128];
  for (int j = tid; j < 512; j += 128)
    reinterpret_cast<uint4*>(sX)[j] = *reinterpret_cast<const uint4*>(X + (size_t)(r0 + j / 8) * rs + t0 + 16 * (j % 8));
  reinterpret_cast<uint4*>(sS)[tid] = *reinterpret_cast<const uint4*>(X + (size_t)(r0 + tid / 2) * rs + k + t0 / 4 + 16 * (tid % 2));
  if (tid < 64) {
    sAl[tid] = reinterpret_cast<const uint16_t*>(a.p[1])[r0 + tid];
    sBe[tid] = reinterpret_cast<const uint16_t*>(a.p[2])[r0 + tid];
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  float d[64];
  WG_FENCE;
  qmma<128, 0>(d, desc(sa(sE)), desc(sa(sF)));
  WG_COMMIT;
  WG_WAIT(0);
#pragma unroll
  for (int i = 0; i < 64; i++) {
    const int r = 16 * (tid / 32) + (tid % 32) / 4 + 8 * ((i / 2) % 2), t = 8 * (i / 4) + 2 * (tid % 4) + i % 2;
    const uint16_t xb = f2bf(__fmul_rn((float)(int8_t)sX[r * 128 + t], bf2f(sS[r * 16 + t / 8])));
    const uint16_t nv = f2bf(__fadd_rn(d[i], 0.0f));
    const uint16_t f = bfma(sAl[r], xb, bmul(sBe[r], nv));
    sO[(t / 32) * 2048 + (r >> 3) * 256 + ((t & 31) >> 4) * 128 + (r & 7) * 16 + (t & 15)] = e4m3(fmaxf(fminf(bf2f(f), 448.0f), -448.0f));
  }
  __syncthreads();
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[5]);
#pragma unroll
  for (int c = 0; c < 4; c++)
    reinterpret_cast<uint4*>(out + ((size_t)(r0 / 128) * (k / 32) + t0 / 32 + c) * 4096 + (r0 % 128) / 64 * 2048)[tid] =
        reinterpret_cast<const uint4*>(sO + c * 2048)[tid];
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

// E' = e4m3(beta_row E) per row.  p0 E codes (rows x 32), p1 beta (f32), p2 out E' codes (rows x 32), p3 out E' tiled (H = 128,
// depth 32); i0 rows
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
  uint8_t* t = reinterpret_cast<uint8_t*>(a.p[3]);
  *reinterpret_cast<uint4*>(t + tiled(row, 0, 128, 32)) = make_uint4(w[0], w[1], w[2], w[3]);
  *reinterpret_cast<uint4*>(t + tiled(row, 16, 128, 32)) = make_uint4(w[4], w[5], w[6], w[7]);
}

// X~ = e4m3(D), D = the atom with alpha x as its accumulator (scale-d = 1) over E' and the F lines, per 64 rows x 128 of depth.
// p0 rows (FP32), p1 alpha, p3 E' tiled, p4 F tiled (H = 128, depth 32; lines are rows), p5 out codes tiled (H = 128, depth k);
// i0 rows, i1 k.  grid (k / 128, rows / 64), 128 threads
extern "C" __global__ void __launch_bounds__(128) form_s5(const Args a) {
  __shared__ __align__(128) uint8_t sE[2048], sF[4096], sO[4 * 2048];
  const int tid = threadIdx.x, t0 = blockIdx.x * 128, r0 = blockIdx.y * 64, k = a.i[1];
  const uint4* E = reinterpret_cast<const uint4*>(a.p[3] + (size_t)(r0 / 128) * 4096 + (r0 % 128) / 64 * 2048);
  const uint4* F = reinterpret_cast<const uint4*>(a.p[4] + (size_t)(t0 / 128) * 4096);
  reinterpret_cast<uint4*>(sE)[tid] = E[tid];
  reinterpret_cast<uint4*>(sF)[tid] = F[tid];
  reinterpret_cast<uint4*>(sF)[tid + 128] = F[tid + 128];
  const float* x = reinterpret_cast<const float*>(a.p[0]);
  const float* alpha = reinterpret_cast<const float*>(a.p[1]);
  const int rA = 16 * (tid / 32) + (tid % 32) / 4, cA = 2 * (tid % 4);
  const float al0 = alpha[r0 + rA], al1 = alpha[r0 + rA + 8];
  float d[64];
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const float2 v0 = *reinterpret_cast<const float2*>(x + (size_t)(r0 + rA) * k + t0 + 8 * j + cA);
    const float2 v1 = *reinterpret_cast<const float2*>(x + (size_t)(r0 + rA + 8) * k + t0 + 8 * j + cA);
    d[4 * j] = __fmul_rn(al0, v0.x); d[4 * j + 1] = __fmul_rn(al0, v0.y);
    d[4 * j + 2] = __fmul_rn(al1, v1.x); d[4 * j + 3] = __fmul_rn(al1, v1.y);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  WG_FENCE;
  qmma<128, 1>(d, desc(sa(sE)), desc(sa(sF)));
  WG_COMMIT;
  WG_WAIT(0);
  // codes i and i + 1 are columns t, t + 1 of row r (r = rA + 8q, t = 8m + cA): one F2FP and one 16-bit store at the
  // tiled offset, a per-thread base plus a constant, so the cast is the statement's (two codes per F2FP)
  const int base = (rA >> 3) * 256 + (rA & 7) * 16 + cA;
#pragma unroll
  for (int i = 0; i < 64; i += 2) {
    const int q = (i / 2) % 2, m = i / 4;
    *reinterpret_cast<uint16_t*>(sO + base + (m / 4) * 2048 + q * 256 + ((m % 4) / 2) * 128 + (m % 2) * 8) =
        e4m3x2(d[i], d[i + 1]);
  }
  __syncthreads();
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[5]);
#pragma unroll
  for (int c = 0; c < 4; c++)
    reinterpret_cast<uint4*>(out + ((size_t)(r0 / 128) * (k / 32) + t0 / 32 + c) * 4096 + (r0 % 128) / 64 * 2048)[tid] =
        reinterpret_cast<const uint4*>(sO + c * 2048)[tid];
}

// the adopted quantizer (the model's FP8 path with PoUW off): codes = e4m3(alpha x) tiled (H = 128), alpha from stats_s5.
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
    o[i] = e4m3x2(__fmul_rn(al, v.x), __fmul_rn(al, v.y)) | (uint32_t)e4m3x2(__fmul_rn(al, v.z), __fmul_rn(al, v.w)) << 16;
  }
  *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a.p[2]) + tiled(row, t, 128, k)) = make_uint4(o[0], o[1], o[2], o[3]);
}

// ---------------------------------------------------------------- the plain baseline's quantize and inputs
// one CTA per row (BF16 when i2 = 0, FP32 when i2 = 1): s = 448 / amax, codes = fp8(x s) tiled (H = 128), glue scale amax / 448.
// p0 X (rows x k), p1 out codes, p2 out glue (f32); i0 rows, i1 k, i2 input type
DEV void load16(const uint8_t* row, int g, bool f32in, float* v) {
  if (f32in) {
    const float4* p = reinterpret_cast<const float4*>(row) + 4 * g;
#pragma unroll
    for (int i = 0; i < 4; i++) { const float4 w = p[i]; v[4 * i] = w.x; v[4 * i + 1] = w.y; v[4 * i + 2] = w.z; v[4 * i + 3] = w.w; }
  } else {
    const uint4* p = reinterpret_cast<const uint4*>(row) + 2 * g;
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const uint4 w = p[i];
      const uint32_t u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; j++) { v[8 * i + 2 * j] = bf2f(u[j]); v[8 * i + 2 * j + 1] = bf2f(u[j] >> 16); }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256) plainq(const Args a) {
  __shared__ float red[8];
  const int row = blockIdx.x, k = a.i[1], tid = threadIdx.x;
  const bool f32in = a.i[2] != 0;
  const uint8_t* X = reinterpret_cast<const uint8_t*>(a.p[0]) + (size_t)row * k * (f32in ? 4 : 2);
  float amax = 0.0f, v[16];
  for (int g = tid; g < k / 16; g += 256) {
    load16(X, g, f32in, v);
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));
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
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[1]);
  for (int g = tid; g < k / 16; g += 256) {
    load16(X, g, f32in, v);
    uint32_t o[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 16; i++) o[i / 4] |= (uint32_t)e4m3(__fmul_rn(v[i], s)) << (8 * (i % 4));
    *reinterpret_cast<uint4*>(out + tiled(row, 16 * g, 128, k)) = make_uint4(o[0], o[1], o[2], o[3]);
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

// X = RNE_bf16(code scale) per element: p0 FP10 rows, p1 out (rows x k bf16); i0 rows, i1 k.  One thread per 8-code block
extern "C" __global__ void open_rows(const Args a) {
  const int k = a.i[1];
  const size_t blk = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (blk >= (size_t)a.i[0] * (k / 8)) return;
  const size_t row = blk / (k / 8), b = blk % (k / 8);
  const uint8_t* X = reinterpret_cast<const uint8_t*>(a.p[0]) + row * (k + k / 4);
  const uint2 w = *reinterpret_cast<const uint2*>(X + 8 * b);
  const float s = bf2f(reinterpret_cast<const uint16_t*>(X + k)[b]);
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint32_t lo = f2bf(__fmul_rn((float)(int8_t)(((2 * i < 4 ? w.x : w.y) >> (8 * ((2 * i) % 4))) & 0xFF), s));
    const uint32_t hi = f2bf(__fmul_rn((float)(int8_t)(((2 * i + 1 < 4 ? w.x : w.y) >> (8 * ((2 * i + 1) % 4))) & 0xFF), s));
    o[i] = lo | hi << 16;
  }
  *reinterpret_cast<uint4*>(a.p[1] + (row * k + 8 * b) * 2) = make_uint4(o[0], o[1], o[2], o[3]);
}

// timing inputs: FP10 rows whose codes are mix32 bytes (-128 -> -127) and scales 2^(-4 - h mod 5).  p0 out; i0 rows, i1 k, i2 seed
extern "C" __global__ void gen_fp10(const Args a) {
  const int k = a.i[1], rs = k + k / 4;
  const size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= (size_t)a.i[0] * rs / 4) return;
  const size_t row = w * 4 / rs, off = w * 4 % rs;
  uint32_t o;
  if (off < (size_t)k) {
    o = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      uint32_t b = mix32((uint32_t)a.i[2] ^ mix32((uint32_t)(row * rs + off + j))) & 0xFF;
      o |= (b == 0x80 ? 0x81 : b) << (8 * j);
    }
  } else {
    const uint32_t h0 = mix32((uint32_t)a.i[2] ^ mix32((uint32_t)(row * rs + off))), h1 = mix32((uint32_t)a.i[2] ^ mix32((uint32_t)(row * rs + off + 2)));
    o = (123 - h0 % 5) << 7 | (123 - h1 % 5) << 23;
  }
  reinterpret_cast<uint32_t*>(a.p[0])[w] = o;
}

// ---------------------------------------------------------------- the clean-up's factors
// the peel's scaled noise: bf16(beta bf16(E)) for v0 (i1 = 0; beta BF16), bf16(E') for v1 (i1 = 1; p0 holds E', beta unused)
DEV uint16_t scaled(const Args& a, int row, int p) {
  const uint8_t e = reinterpret_cast<const uint8_t*>(a.p[0])[row * 32 + p];
  return a.i[1] ? e4m3bf(e) : bmul(reinterpret_cast<const uint16_t*>(a.p[1])[row], e4m3bf(e));
}

// P_A rows (tiled, H = 128, 128 bytes): [scaled noise || RNE_bf16(A'F_B)].  p0 E_A (or E'_A) codes (rows x 32), p1 beta (u16),
// p2 AF words (rows x 32), p3 out; i0 rows, i1 forming
extern "C" __global__ void pa(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] * 64) return;
  const int row = t / 64, e = t % 64;
  const uint16_t v = e < 32 ? scaled(a, row, e) : f2bf(reinterpret_cast<const float*>(a.p[2])[row * 32 + e - 32]);
  *reinterpret_cast<uint16_t*>(a.p[3] + tiled(row, 2 * e, 128, 128)) = v;
}

// P_B rows: [RNE_bf16(FP32 add(fma chain from +0 over p of be_p G_pq, -BF_jq)) || -be], be the scaled noise.
// p0 E_B (or E'_B) codes, p1 beta, p2 BF words (rows x 32), p3 G words (32 x 32, row p), p4 out; i0 rows, i1 forming
extern "C" __global__ void pb(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] * 32) return;
  const int row = t / 32, q = t % 32;
  const float* G = reinterpret_cast<const float*>(a.p[3]);
  float acc = 0.0f;
  for (int p = 0; p < 32; p++) acc = __fmaf_rn(bf2f(scaled(a, row, p)), G[p * 32 + q], acc);
  const uint16_t mid = f2bf(__fadd_rn(acc, -reinterpret_cast<const float*>(a.p[2])[row * 32 + q]));
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[4]);
  *reinterpret_cast<uint16_t*>(out + tiled(row, 2 * q, 128, 128)) = mid;
  *reinterpret_cast<uint16_t*>(out + tiled(row, 2 * (32 + q), 128, 128)) = scaled(a, row, q) ^ 0x8000;
}

// words differing between two buffers: p0, p1 (u32), p2 counter (u64); i0 count (in words, < 2^31 per launch)
extern "C" __global__ void diff_words(const Args a) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const bool d = i < (size_t)a.i[0] && reinterpret_cast<const uint32_t*>(a.p[0])[i] != reinterpret_cast<const uint32_t*>(a.p[1])[i];
  const unsigned n = __syncthreads_count(d);
  if (threadIdx.x == 0 && n) atomicAdd(reinterpret_cast<unsigned long long*>(a.p[2]), (unsigned long long)n);
}

// The correctly rounded divide, reciprocal and square root in the forms the kernels use: the only sequences whose .FTZ FP32 ops
// build.sh admits (ftz_gate.py pins them), and run.py's check holds every word to the correctly rounded value on subnormal
// inputs and outputs.  p0 x, p1 y, p2 out (6 words per element: x / y, 448 / x, x / 448, 1 / x, sqrt x, bf16(x) / bf16(y));
// i0 elements
extern "C" __global__ void __launch_bounds__(128) ieee_ops(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0]) return;
  const float x = reinterpret_cast<const float*>(a.p[0])[t], y = reinterpret_cast<const float*>(a.p[1])[t];
  uint32_t* o = reinterpret_cast<uint32_t*>(a.p[2]) + 6 * (size_t)t;
  o[0] = __float_as_uint(__fdiv_rn(x, y));
  o[1] = __float_as_uint(__fdiv_rn(448.0f, x));
  o[2] = __float_as_uint(__fdiv_rn(x, 448.0f));
  o[3] = __float_as_uint(__frcp_rn(x));
  o[4] = __float_as_uint(__fsqrt_rn(x));
  o[5] = bdiv(f2bf(x), f2bf(y));
}

// 448 / x alone, the form norms' bdiv(BF_QMAX, ·) takes, for which ptxas specialises the divide's slow path to the constant
// dividend: its own pinned sequence.  p0 x, p1 out; i0 elements
extern "C" __global__ void __launch_bounds__(128) ieee_q(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0]) return;
  reinterpret_cast<uint32_t*>(a.p[1])[t] = __float_as_uint(__fdiv_rn(448.0f, reinterpret_cast<const float*>(a.p[0])[t]));
}
