// Pearl-C on the H100 (sm_90a): the kernels of `pearl-c-h100-v0` (protocols/pouw/verity_pouw/schemes/pearl_c.py) and the
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

struct Args { uint64_t p[16]; int32_t i[16]; };

#define DEV __device__ __forceinline__
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
DEV float e4m3f(uint8_t c) { return __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3))); }
DEV uint16_t e4m3bf(uint8_t c) { return f2bf(e4m3f(c)); }

constexpr uint16_t BF_QMAX = 0x43E0;       // 448
constexpr uint16_t BF_DELTA_R = 0x40B5;    // bf16(f32(delta sqrt r)), delta = 1, r = 32
constexpr uint16_t BF_DELTA_STD = 0x38B5;  // bf16(f32(delta sqrt r / 256^2))
constexpr uint16_t BF_FLOOR = 0x2F80;      // 2^-32
constexpr uint16_t BF_NORM = 0x4600;       // 256 * 32

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

// p0 the line key (8 words), p1 codes (count x 32, row-major), p2 rows tiled (H = 128, depth 32) or 0, p3 transposed tiled
// (32 x count, H = 32) or 0, p4 transposed tiled (128 x count, H = 128, rows 32-127 left as they are) or 0; i0 side, i1 factor,
// i2 count
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
  const uint16_t scale = bdiv(BF_NORM, f2bf((float)isqrt(sumsq * 1024)));
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

// ---------------------------------------------------------------- the plain baseline's quantize and inputs
// one CTA per BF16 row: s = 448 / amax, codes = fp8(x s) tiled (H = 128), glue scale amax / 448.  p0 X (rows x k bf16),
// p1 out codes, p2 out glue (f32); i0 rows, i1 k
extern "C" __global__ void __launch_bounds__(256) plainq(const Args a) {
  __shared__ float red[8];
  const int row = blockIdx.x, k = a.i[1], tid = threadIdx.x;
  const uint4* X = reinterpret_cast<const uint4*>(a.p[0] + (size_t)row * k * 2);
  float amax = 0.0f;
  for (int j = tid; j < k / 8; j += 256) {
    const uint4 w = X[j];
    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; i++) amax = fmaxf(amax, fmaxf(fabsf(bf2f(v[i])), fabsf(bf2f(v[i] >> 16))));
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
    const uint4 w0 = X[2 * g], w1 = X[2 * g + 1];
    const uint32_t v[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    uint32_t o[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 16; i++) o[i / 4] |= (uint32_t)e4m3(__fmul_rn(bf2f(v[i / 2] >> (16 * (i % 2))), s)) << (8 * (i % 4));
    *reinterpret_cast<uint4*>(out + tiled(row, 16 * g, 128, k)) = make_uint4(o[0], o[1], o[2], o[3]);
  }
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
// P_A rows (tiled, H = 128, 128 bytes): [bf16(beta_A bf16(E_A)) || RNE_bf16(A'F_B)].  p0 E_A codes (rows x 32), p1 beta (u16),
// p2 AF words (rows x 32), p3 out; i0 rows
extern "C" __global__ void pa(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] * 64) return;
  const int row = t / 64, e = t % 64;
  uint16_t v;
  if (e < 32) v = bmul(reinterpret_cast<const uint16_t*>(a.p[1])[row], e4m3bf(reinterpret_cast<const uint8_t*>(a.p[0])[row * 32 + e]));
  else v = f2bf(reinterpret_cast<const float*>(a.p[2])[row * 32 + e - 32]);
  *reinterpret_cast<uint16_t*>(a.p[3] + tiled(row, 2 * e, 128, 128)) = v;
}

// P_B rows: [RNE_bf16(FP32 add(fma chain from +0 over p of be_p G_pq, -BF_jq)) || -be], be_p = bf16(beta_B bf16(E_B,p)).
// p0 E_B codes, p1 beta, p2 BF words (rows x 32), p3 G words (32 x 32, row p), p4 out; i0 rows
extern "C" __global__ void pb(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] * 32) return;
  const int row = t / 32, q = t % 32;
  const uint8_t* E = reinterpret_cast<const uint8_t*>(a.p[0]) + row * 32;
  const uint16_t beta = reinterpret_cast<const uint16_t*>(a.p[1])[row];
  const float* G = reinterpret_cast<const float*>(a.p[3]);
  float acc = 0.0f;
  for (int p = 0; p < 32; p++) acc = __fmaf_rn(bf2f(bmul(beta, e4m3bf(E[p]))), G[p * 32 + q], acc);
  const uint16_t mid = f2bf(__fadd_rn(acc, -reinterpret_cast<const float*>(a.p[2])[row * 32 + q]));
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[4]);
  *reinterpret_cast<uint16_t*>(out + tiled(row, 2 * q, 128, 128)) = mid;
  *reinterpret_cast<uint16_t*>(out + tiled(row, 2 * (32 + q), 128, 128)) = bmul(beta, e4m3bf(E[q])) ^ 0x8000;
}

// ---------------------------------------------------------------- hashing: TurboSHAKE128 (Keccak-p[1600, 12], rate 168)
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

// A's commitment, modeled: TurboSHAKE128 of each 1,280-byte segment of an FP10 row, D = 0x23.  p0 rows, p1 out (rows x segs x 32 B);
// i0 rows, i1 k
extern "C" __global__ void __launch_bounds__(128) hash_rows(const Args a) {
  const int k = a.i[1], rs = k + k / 4, segs = rs / 1280;
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)a.i[0] * segs) return;
  const uint64_t* src = reinterpret_cast<const uint64_t*>(a.p[0] + (t / segs) * rs + (t % segs) * 1280);
  uint64_t h[4];
  turboshake<160, 0x23>([&](int i) { return src[i]; }, h);
  store32(reinterpret_cast<uint8_t*>(a.p[1]) + t * 32, h);
}

// Pearl's tickets: per 16 x 16 subtile of C~, the 16-lane fold (lane (rb, cb) folds its 4 x 4 block row-major,
// lane <- rotl32(lane 0x9E3779B1 + bits, 13)) hashed as one keyed BLAKE3 block.  p0 C~, p1 out (subtiles row-major x 32 B),
// p2 key (8 words); i0 rows, i1 cols
extern "C" __global__ void __launch_bounds__(128) tickets(const Args a) {
  const int cols = a.i[1], sn = cols / 16;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] / 16 * sn) return;
  const int si = t / sn, sj = t % sn;
  const uint32_t* C = reinterpret_cast<const uint32_t*>(a.p[0]);
  uint32_t z[16], kw[8], h[8];
#pragma unroll
  for (int l = 0; l < 16; l++) {
    uint32_t lane = 0;
#pragma unroll
    for (int f = 0; f < 16; f++) {
      lane = lane * 0x9E3779B1u + C[(size_t)(16 * si + 4 * (l / 4) + f / 4) * cols + 16 * sj + 4 * (l % 4) + f % 4];
      lane = __funnelshift_l(lane, lane, 13);
    }
    z[l] = lane;
  }
#pragma unroll
  for (int i = 0; i < 8; i++) kw[i] = reinterpret_cast<const uint32_t*>(a.p[2])[i];
  blake3_block(kw, z, h);
  uint4* out = reinterpret_cast<uint4*>(a.p[1] + (size_t)t * 32);
  out[0] = make_uint4(h[0], h[1], h[2], h[3]);
  out[1] = make_uint4(h[4], h[5], h[6], h[7]);
}

// words differing between two buffers: p0, p1 (u32), p2 counter (u64); i0 count (in words, < 2^31 per launch)
extern "C" __global__ void diff_words(const Args a) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const bool d = i < (size_t)a.i[0] && reinterpret_cast<const uint32_t*>(a.p[0])[i] != reinterpret_cast<const uint32_t*>(a.p[1])[i];
  const unsigned n = __syncthreads_count(d);
  if (threadIdx.x == 0 && n) atomicAdd(reinterpret_cast<unsigned long long*>(a.p[2]), (unsigned long long)n);
}
