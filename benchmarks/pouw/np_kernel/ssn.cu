// ssn_1l on one H200 (sm_90a): the served sparse-noise GEMM with every k32 accumulator word exported and hm96-hashed.
//
// Layout (K data entries, S = 1 + ceil((K - 28) / 31) k32 steps; lanes are a step's 32 K bytes):
//   A'  step 1: lanes 0-27 A[i, 0..27], lanes 28-31 a1[i, 0..3] in {0,1}; step t >= 2: lanes 0-30 the next 31 entries of A,
//       lane 31 a_t[i] in {0,1}; data lanes past K are 0. The data lanes are written by A's producer with the noise lanes 0;
//       the forming pass (form_k) ORs the noise bits into the noise registers (one 32-bit register a row a step).
//   B'  step 1: lanes 0-27 B[0..27, j], lanes 28-31 the public code C[0..3, j] (the balanced base-13 digits of j + 1, the
//       base-13 digits of j + 14,281 minus 6); step t >= 2: lanes 0-30 the next 31 rows of B, lane 31 = 1. Formed once at load.
//   The accumulator starts at -beta_j, beta_j = floor((sum_e C[e, j] + S - 1 + 1) / 2) (half up), so word (i, j, t) is
//   D_t(i, j) - beta_j + sum_e a1[i, e] C[e, j] + sum_{2 <= t' <= t} a_t'[i], and the last is ssn-1l-b1-v1's P'.
// TAG=1 is ssn-1l-tag: step 1 holds 24 data lanes, lanes 24-27 the row tag T(i) (the balanced base-13 digits of
//   z_i = (floor(i / 2) + 1)(-1)^i, written by A's producer) against G[0..3, j] in [-3, 3] (drawn per call from the call's
//   coins by g_k into B' step 1's lanes 24-27), and lanes 28-31 a1 against C; step t >= 2's lane 31 is 1 + a_t[i] against 1;
//   S = 1 + ceil((K - 24) / 31); beta_j = floor((sum_e C[e, j] + 3 (S - 1) + 1) / 2); word (i, j, t) is
//   D_t(i, j) - beta_j + sum_e a1[i, e] C[e, j] + sum_e T(i)[e] G[e, j] + sum_{2 <= t' <= t} (1 + a_t'[i]).
// BLC=1 is blc4a-t64-e7 (N-EGRESS): the tag layout's step 1, and each later step lanes 0-27 data, lanes 28-30 A' = 1 (written
//   by A's producer) against the call's lane coins s_t[e, tile, j], odd and uniform in [-127, 127], one set a 64-row tile, and
//   lane 31 1 + a_t[i] against 1; S = 1 + ceil((K - 24) / 28). The certifier streams a word a (tile, step, column), the three
//   coins and lane 31's 1, after A_j; the GEMM copies it into B' lanes 28-31 of its stage (ssn_k's cv). Word (i, j, t) adds
//   sum_{2 <= t' <= t} sum_e s_t'[e, i / 64, j] to the tag layout's. BLC_W=w0,w1,w2 puts A' = (w0, w1, w2) in place of
//   (1, 1, 1), so a step adds sum_e w_e s_t'[e, i / 64, j]. BLC=2 (blc4a-t64-e7-s1) gives step 1 lane coins too: its lanes
//   0-19 data, lanes 20-22 A' = (w0, w1, w2) against s_1[e, i / 64, j], lane 23 A' = 0 against the word's fourth byte, lanes
//   24-31 as the tag layout's; S = 1 + ceil((K - 20) / 28), and the sums run from t' = 1.
// ROT=1 (with BLC) is blc4a-t64-e7-rot: A' in the three lane-coin lanes by row parity, (127, 1, 1) on even rows and
//   (1, 127, 1) on odd ones (ROT_W), in place of BLC_W, so a step adds sum_e ROT_W[i mod 2][e] s_t'[e, i / 64, j].
// XU=1 is ssn_1l_tag_xu with kappa = 0 on a tag layout: the last step's lanes from xa (the first 4-aligned lane past its
//   data) hold a1 against -C[0..3, j], -T(i) (written by A's producer) against G[0..3, j] (g_k writes G there too), and from
//   xa + 8 the bytes u0 against 1, u1 against 127 and u2 in e = min(12, spare) lanes against 127, u0 the low 7 bits of row
//   i's u word and u1, u2 its signed bytes 1 and 2 (written by the forming); a step is added when fewer than 12 lanes are
//   left, and S >= 2. The final word is A_i . B_j - beta_j + rho_i + sigma_{tile, class, j}, rho_i = u0 + 127 u1 + 127 e u2
//   + sum_{t >= 2} (1 + a_t[i]) and sigma the lane coins' sum under row class i mod 2's A' (both classes alike without
//   ROT); the GEMM's epilogue serves bf16(fl(fl(v s_j - 1.5 2^23 s_j) s_i)), v the float whose bits are the word
//   + (0x4B400000 - rho_i) + (beta_j - sigma_{tile, class, j}) (one IADD3 of the forming's row source and the certifier's
//   column source), which is stock's epilogue on A_i . B_j bit for bit while |A_i . B_j| <= 2^22 and s_j has a 22-bit
//   significand. Past 2^22 the IADD3 reads back a wrong integer (2^22 + 1 as 2^22 + 2) and nothing flags it; the served
//   output is wrong once that error passes bf16's rounding.
// FOLD=1 (with BLC, not XU) is the lead served noisy: no de-noise lanes, and the same epilogue with row source 0 and column
//   source 0x4B400000 - sigma_{tile, class, j}, so it serves stock's epilogue on the final word less sigma, A_i . B_j - beta_j
//   + sum_e a1[i, e] C[e, j] + sum_e T(i)[e] G[e, j] + sum_{t >= 2} (1 + a_t[i]), bit for bit while that is within 2^22.
// EPI=1 runs stock's epilogue on the final word instead. The lanes, the sources and the fold are layout.h's, which the CPU
// check (cpu_check.cc) also reads.
// Coins: a row's S + 3 bits are bits 0..S+2 of its H = ceil((S + 3) / 32) coin words (a1[i, e] bit e, a_t[i] bit t + 2);
// row i's words are coin words i H .. i H + H - 1 of the call's stream, whose 16-word block c is SHA-512(seed || be64 c),
// seed = SHA-512 of the call's 224-byte seed message (tag, salt, forward, position, root(A_j), root(B_j)). Under TAG=1,
// column j's G is coin word m H + j, and under XU=1 row i's u word is coin word m H + n + i.
// Operands are tiled: A' tile (64-row block, step) and B' tile (BN-column block, step) are contiguous, in dc.cuh's K-major
// layout, so a step's tiles are one linear copy. Exported words: a block (row tile, column tile, step) of 64 x BN s32 in
// wgmma's fragment order (uint4 j * 128 + thread holds the thread's d[4 j .. 4 j + 3]); the hash pass (Hasher) hashes them
// as the served committer's rows or as served_zk's row-seg rows.
//
//   ssn check                 every exported word, the final outputs and the formed registers against the host reference
//                             on small shapes (random, extreme and zero data; columns past 14,280, where the code's top
//                             digit wraps; S = 1 and a
//                             partial last step; random data with every lane coin 127), in the four layouts, then under
//                             XU=1 on the three tag layouts with each served output against stock's epilogue, then
//                             blc4a-t64-e7-s1 under ROT=1 without XU, with XU=1 and with FOLD=1
//   ssn time REPS M [BUDGET_GB [ROW_BYTES ...]]
//                             Llama-3.2-1B's fused linears at M rows, one JSON line a timing (see time_all), the hash pass
//                             by each ROW_BYTES (0: row-seg; default 0 65536 16384)
// L1=1 replaces root(A_j) and the seed message's coins with L1 + (a)'s seed path (l1.h), checked by `ssn check`; L1=2 runs
// the same path shaped for latency beside the hash streams (l1_root2_k, l1_nf_k; L1_CTA 64 or 128), checked against L1=1's.
// Build: run.sh (ssn.cu, the served committer's native_rows.cu and its plain-leaf copy from plain_rows.py).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../dc/dc.cuh"
#include "../dc/wgmma.cuh"
#include "../served_zk/hm96_rows.cu"
#include "l1.h"
#include "layout.h"

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)
#define CU(x)                                                          \
  do {                                                                 \
    CUresult r_ = (x);                                                 \
    if (r_ != CUDA_SUCCESS) {                                          \
      const char* s_ = "";                                             \
      cuGetErrorString(r_, &s_);                                       \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, s_); \
      exit(2);                                                         \
    }                                                                  \
  } while (0)

typedef uint32_t u32;
typedef uint64_t u64;

constexpr int KB = 32, TA = 64 * KB, STAGES = 4;

struct Clk {
  long long c0, c1;
  u64 t0, t1;
};
// The served epilogue (mode 1 stock's: I2F, FMUL s_j, FMUL s_i, the bf16 pack; mode 2 xu's: IADD3 of the row and column
// sources, FFMA(v, s_j, -1.5 2^23 s_j), FMUL s_i, the pack), its outputs two bf16 a word, n a row
struct Xu {
  const int* r;     // 0x4B400000 - rho_i a row (FOLD=1: 0)
  const float* s;   // s_i a row
  const int* c;     // col_source a (64-row tile, row class i mod 2, column), (tile 2 + class) n + j
  const float2* j;  // (s_j, -1.5 2^23 s_j) a column
  u32* o;
  int mode;
};
__device__ __forceinline__ u64 gtime() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void cp16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(s)), "l"(g) : "memory");
}
__device__ __forceinline__ void cp8(void* s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_u32(s)), "l"(g) : "memory");
}
__device__ __forceinline__ void cp4(void* s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(s)), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template <int BN> __device__ __forceinline__ void mma(int (&d)[BN / 2], u64 da, u64 db);
template <> __device__ __forceinline__ void mma<256>(int (&d)[128], u64 da, u64 db) { wgmma_s8_n256(d, da, db, 1); }
template <> __device__ __forceinline__ void mma<128>(int (&d)[64], u64 da, u64 db) { wgmma_s8_n128(d, da, db, 1); }

// One CTA (a warpgroup) a 64 x BN output tile over S steps, STAGES-deep cp.async ring. EXPORT: wait for each step's wgmma
// and store its 64 x BN words (block (rt, ct, t) of W); otherwise one wgmma group stays in flight (the plain GEMM).
// cv (blc4a-t64-e7): the lane coins, word ((rt (S - 1) + t - 1) n + j) the bytes of B' lanes 28-31 of column j at step
// t >= 1 for row tile rt, copied into the stage by one 4-byte cp.async beside B' lanes 0-27 (no register written). c1
// (BLC=2): step 1 has lane coins too, word ((rt S + t) n + j), at step 1 in B' lanes 20-23.
template <int BN, bool EXPORT>
__global__ void __launch_bounds__(128) ssn_k(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                             const int* __restrict__ beta, int S, int n, uint4* __restrict__ W,
                                             int* __restrict__ out, Clk* clk, const u32* __restrict__ cv, int c1, Xu x) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int TB = BN * KB, ST = TA + TB;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid = threadIdx.x, ct = blockIdx.x, rt = blockIdx.y, nct = gridDim.x;
  const uint8_t* a0 = A + (size_t)rt * S * TA;
  const uint8_t* b0 = B + (size_t)ct * S * TB;
  auto load = [&](int t) {
    if (t < S) {
      unsigned char* st = sm + (t % STAGES) * ST;
      cp16(st + 16 * tid, a0 + (size_t)t * TA + 16 * tid);
      if (cv && t + c1 > 0) {
        const u32* c0 = cv + ((size_t)rt * (S - 1 + c1) + t - 1 + c1) * n + ct * BN;
#pragma unroll
        for (int j = 0; j < TB / 16 / 128; j++) {
          const int c = tid + 128 * j;  // kmaj_off's 16-byte chunk: column (c >> 4) 8 + (c & 7), K bytes 16 (c >> 3 & 1) on
          unsigned char* s = st + TA + 16 * c;
          const uint8_t* g = b0 + (size_t)t * TB + 16 * c;
          if ((c & 8) && t == 0) {
            cp4(s, g);
            cp4(s + 4, c0 + (c >> 4) * 8 + (c & 7));
            cp8(s + 8, g + 8);
          } else if (c & 8) {
            cp8(s, g);
            cp4(s + 8, g + 8);
            cp4(s + 12, c0 + (c >> 4) * 8 + (c & 7));
          } else {
            cp16(s, g);
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < TB / 16 / 128; j++) cp16(st + TA + 16 * (tid + 128 * j), b0 + (size_t)t * TB + 16 * (tid + 128 * j));
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int t = 0; t < STAGES - 1; t++) load(t);
  const int w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  int d[BN / 2];
#pragma unroll
  for (int ni = 0; ni < BN / 8; ni++)
#pragma unroll
    for (int e = 0; e < 4; e++) d[4 * ni + e] = beta ? -beta[ct * BN + 8 * ni + 2 * q + (e & 1)] : 0;
  wgmma_hold(d);
  const long long c0 = clock64();
  const u64 t0 = gtime();
  uint4* wb = W + (size_t)(rt * nct + ct) * S * (BN * 16);
#pragma unroll 1
  for (int t = 0; t < S; t++) {
    cp_wait<STAGES - 2>();
    fence_async_smem();
    __syncthreads();
    unsigned char* st = sm + (t % STAGES) * ST;
    const u64 da = wgmma_desc(smem_u32(st), kmaj_lbo(), kmaj_sbo(KB));
    const u64 db = wgmma_desc(smem_u32(st + TA), kmaj_lbo(), kmaj_sbo(KB));
    wgmma_fence();
    mma<BN>(d, da, db);
    wgmma_commit();
    if constexpr (EXPORT) {
      wgmma_wait<0>();
      wgmma_hold(d);
      uint4* r = wb + (size_t)t * (BN * 16);
#pragma unroll
      for (int j = 0; j < BN / 8; j++) r[j * 128 + tid] = make_uint4(d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
    } else {
      wgmma_wait<1>();
    }
    load(t + STAGES - 1);
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  if (clk && rt == 0 && ct == 0 && tid == 0) *clk = Clk{c0, clock64(), t0, gtime()};
  if (out)
#pragma unroll
    for (int ni = 0; ni < BN / 8; ni++)
#pragma unroll
      for (int e = 0; e < 4; e++)
        out[(size_t)(64 * rt + 16 * w + g + 8 * (e >> 1)) * n + ct * BN + 8 * ni + 2 * q + (e & 1)] = d[4 * ni + e];
  if (x.mode) {
    // the thread's rows r0 and r0 + 8 share their parity, g's, so one row class's column sources serve both
    const int r0 = 64 * rt + 16 * w + g;
    const float s0 = x.s[r0], s1 = x.s[r0 + 8];
    const u32 x0 = x.mode == 2 ? (u32)x.r[r0] : 0u, x1 = x.mode == 2 ? (u32)x.r[r0 + 8] : 0u;
    const int* xc = x.c + ((size_t)rt * 2 + (g & 1)) * n;
#pragma unroll
    for (int ni = 0; ni < BN / 8; ni++) {
      const int col = ct * BN + 8 * ni + 2 * q;
      const float2 j0 = x.j[col], j1 = x.j[col + 1];
      float y[4];
      if (x.mode == 2) {
        const u32 k0 = (u32)xc[col], k1 = (u32)xc[col + 1];
        y[0] = xu_fold((u32)d[4 * ni], x0, k0, j0.x, j0.y, s0);
        y[1] = xu_fold((u32)d[4 * ni + 1], x0, k1, j1.x, j1.y, s0);
        y[2] = xu_fold((u32)d[4 * ni + 2], x1, k0, j0.x, j0.y, s1);
        y[3] = xu_fold((u32)d[4 * ni + 3], x1, k1, j1.x, j1.y, s1);
      } else {
        y[0] = stock_fold(d[4 * ni], j0.x, s0);
        y[1] = stock_fold(d[4 * ni + 1], j1.x, s0);
        y[2] = stock_fold(d[4 * ni + 2], j0.x, s1);
        y[3] = stock_fold(d[4 * ni + 3], j1.x, s1);
      }
      const __nv_bfloat162 h0 = __floats2bfloat162_rn(y[0], y[1]), h1 = __floats2bfloat162_rn(y[2], y[3]);
      x.o[((size_t)r0 * n + col) / 2] = *reinterpret_cast<const u32*>(&h0);
      x.o[((size_t)(r0 + 8) * n + col) / 2] = *reinterpret_cast<const u32*>(&h1);
    }
  }
#endif
}

// The call's coin stream: thread b derives the seed from the 256-byte seed message (two blocks; the second carries the
// padding) and writes block b = SHA-512(seed || be64 b), 16 coin words (word 2 j the high half of state word j).
__global__ void coins_k(const u64* __restrict__ msg, u32* __restrict__ coins, long long blocks) {
  const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  u64 st[8], w[16];
  s512_iv(st);
  for (int j = 0; j < 16; j++) w[j] = msg[j];
  s512_compress(st, w);
  for (int j = 0; j < 16; j++) w[j] = msg[16 + j];
  s512_compress(st, w);
  u64 seed[8];
  for (int j = 0; j < 8; j++) seed[j] = st[j];
  s512_iv(st);
  for (int j = 0; j < 8; j++) w[j] = seed[j];
  w[8] = (u64)b;
  w[9] = 0x8000000000000000ull;
  for (int j = 10; j < 15; j++) w[j] = 0;
  w[15] = 72 * 8;
  s512_compress(st, w);
  uint4* o = reinterpret_cast<uint4*>(coins + 16 * b);
  for (int j = 0; j < 4; j++)
    o[j] = make_uint4((u32)(st[2 * j] >> 32), (u32)st[2 * j], (u32)(st[2 * j + 1] >> 32), (u32)st[2 * j + 1]);
}

// The forming pass: one thread a noise register (row i, step t), rows fastest. Step 1's register is a1 spread over its four
// bytes (one IMAD and one LOP3); step t's ORs bit t + 2 of the row's stream into byte 3 (a funnel shift and one LOP3), or
// under the tag layout sets byte 3 to 1 + that bit (a LOP3 and an IADD3, so a repeated call leaves it unchanged).
// XU=1 (uc, the rows' u words): the last step's thread also writes a1 at lane xa, the bytes u0 (7 bits), u1 and xe u2
// from lane xa + 8, and the row source 0x4B400000 - rho_i, rho_i = u0 + 127 u1 + 127 xe u2 + sum_{t >= 2} (1 + a_t[i]).
__global__ void form_k(uint8_t* __restrict__ A, const u32* __restrict__ coins, int m, int S, bool tag,
                       const u32* __restrict__ uc = nullptr, int xa = 0, int xe = 0, int* __restrict__ xr = nullptr) {
  const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= (long long)m * S) return;
  const int i = (int)(x % m), t = (int)(x / m), H = coin_words(S);
  uint8_t* st = A + ((size_t)(i >> 6) * S + t) * TA;
  u32* reg = reinterpret_cast<u32*>(st + kmaj_off(i & 63, 28, KB));
  const u32* h = coins + (size_t)i * H;
  if (t == 0) {
    *reg = ((h[0] & 0xFu) * 0x00204081u) & 0x01010101u;
  } else {
    const int p = t + 3;
    const u32 b = __funnelshift_r(h[p >> 5], h[p >> 5], (p + 8) & 31) & 0x01000000u;
    *reg = tag ? (*reg & 0x00FFFFFFu) + 0x01000000u + b : *reg | b;
  }
  if (uc && t == S - 1) {
    *reinterpret_cast<u32*>(st + kmaj_off(i & 63, xa, KB)) = ((h[0] & 0xFu) * 0x00204081u) & 0x01010101u;
    const u32 u = uc[i];
    const int u0 = u0_of(u), u1 = u1_of(u), u2 = u2_of(u);
    for (int l = xa + 8; l < xa + 10 + xe; l += 4) {
      u32 v = 0;
      for (int b = 0; b < 4; b++) {
        const int ll = l + b;
        const u32 y = ll == xa + 8 ? (u32)u0 : ll == xa + 9 ? (u >> 8) & 0xFFu : ll < xa + 10 + xe ? (u >> 16) & 0xFFu : 0u;
        v |= y << (8 * b);
      }
      *reinterpret_cast<u32*>(st + kmaj_off(i & 63, l, KB)) = v;
    }
    int walk = S - 1;
    for (int p = 4; p < S + 3; p++) walk += (h[p >> 5] >> (p & 31)) & 1;
    xr[i] = 0x4B400000 - (u0 + 127 * u1 + 127 * xe * u2 + walk);
  }
}

// ssn-1l-tag's G: column j's coin word u gives G[0..3, j], the base-7 digits of floor(u 2401 / 2^32) minus 3 (within
// 2401 / 2^32 of uniform on [-3, 3]^4), one 32-bit store into B' step 1's lanes 24-27
// (XU=1: also into the last step's lanes xg to xg + 3)
__device__ __forceinline__ void g_store(uint8_t* __restrict__ B, u32 u, int j, int S, int BN, int xg = -1) {
  u32 v = (u32)(((u64)u * 2401u) >> 32), r = 0;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    r |= (u32)(uint8_t)(int8_t)((int)(v % 7) - 3) << (8 * e);
    v /= 7;
  }
  uint8_t* b = B + (size_t)(j / BN) * S * BN * KB;
  *reinterpret_cast<u32*>(b + kmaj_off(j % BN, 24, KB)) = r;
  if (xg >= 0) *reinterpret_cast<u32*>(b + (size_t)(S - 1) * BN * KB + kmaj_off(j % BN, xg, KB)) = r;
}
__global__ void g_k(uint8_t* __restrict__ B, const u32* __restrict__ gc, int n, int S, int BN, int xg) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) g_store(B, gc[j], j, S, BN, xg);
}

// L1 + (a)'s seed path (l1.h), a call: l1_root_k's CTAs each hash 256 of A_j's 1 KiB leaves (a thread a leaf) and fold
// them to their subtree's root in shared memory; with one CTA it goes on to root(A_j) and the seed, else l1_top_k (one CTA)
// folds the CTAs' roots and does. l1_coins_k then writes the call's noise blocks from the seed. The call's leaf count is a
// power of two.
__device__ void l1_fold(const l1::Call& k, u64 (*lv)[8], int m, int level, u64 first) {
  for (; m > 1; m >>= 1, level++) {
    const int t = threadIdx.x;
    u64 o[8];
    const bool act = t < m / 2;
    if (act) l1::node_of(k, level, (first >> (level + 1)) + t, lv[2 * t], lv[2 * t + 1], o);
    __syncthreads();
    if (act)
      for (int i = 0; i < 8; i++) lv[t][i] = o[i];
    __syncthreads();
  }
}
__device__ void l1_seed(const l1::Call& k, const u64* x, u64* seed) {
  u64 h[8], ra[8], s[8];
  l1::instance(k, x, h, ra);
  l1::seed_of(k, ra, s);
  for (int i = 0; i < 8; i++) seed[i] = s[i];
  seed[8 + 0] = ra[0];  // root(A_j)'s first word, for the check
}
__global__ void __launch_bounds__(256) l1_root_k(const l1::Call* __restrict__ kc, const uint8_t* __restrict__ A,
                                                 u64* __restrict__ roots, u64* __restrict__ seed) {
  __shared__ u64 lv[256][8];
  const l1::Call& k = *kc;
  const int t = threadIdx.x;
  const u64 first = (u64)blockIdx.x * 256;
  const int w = (int)(k.n - first < 256 ? k.n - first : 256);
  if (t < w) {
    u64 o[8];
    l1::a_leaf(k, first + t, A + (size_t)l1::SEG * (first + t), o);
    for (int i = 0; i < 8; i++) lv[t][i] = o[i];
  }
  __syncthreads();
  l1_fold(k, lv, w, 0, first);
  if (t) return;
  if (gridDim.x == 1) l1_seed(k, lv[0], seed);
  else
    for (int i = 0; i < 8; i++) roots[8 * blockIdx.x + i] = lv[0][i];
}
__global__ void __launch_bounds__(256) l1_top_k(const l1::Call* __restrict__ kc, const u64* __restrict__ roots, int m,
                                                u64* __restrict__ seed) {
  __shared__ u64 lv[256][8];
  const l1::Call& k = *kc;
  const int t = threadIdx.x;
  if (t < m)
    for (int i = 0; i < 8; i++) lv[t][i] = roots[8 * t + i];
  __syncthreads();
  l1_fold(k, lv, m, 8, 0);
  if (!t) l1_seed(k, lv[0], seed);
}
__global__ void l1_coins_k(const l1::Call* __restrict__ kc, const u64* __restrict__ seed, u32* __restrict__ coins,
                           long long blocks) {
  const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  u64 s[8], o[8];
  for (int i = 0; i < 8; i++) s[i] = seed[i];
  l1::noise_of(*kc, s, (u64)b, o);
  uint4* c = reinterpret_cast<uint4*>(coins + 16 * b);
  for (int j = 0; j < 4; j++)
    c[j] = make_uint4((u32)(o[2 * j] >> 32), (u32)o[2 * j], (u32)(o[2 * j + 1] >> 32), (u32)o[2 * j + 1]);
}

// L1=2, the same seed path shaped for latency beside the hash streams. l1_root2_k's CTAs are CTA threads (64 or 128, at
// least the leaves / 512), so a CTA needs a quarter or half of an SM's registers instead of all of them (a cap, -DL1_REGS=R,
// spills below about 210); the last CTA to finish folds the CTAs' roots, so there is no second launch. Its shared memory holds the
// level being folded and the next: 1.5 cap digests, cap = max(CTA, CTAs). l1_nf_k then derives the noise blocks, G and the
// forming in one launch.
typedef u64 Dig[8];
template <int CTA>
__device__ Dig* l1_fold2(const l1::Call& k, Dig* src, Dig* dst, int m, int level, u64 first) {
  for (; m > 1; m >>= 1, level++) {
    for (int p = threadIdx.x; p < m / 2; p += CTA)
      l1::node_of(k, level, (first >> (level + 1)) + p, src[2 * p], src[2 * p + 1], dst[p]);
    __syncthreads();
    Dig* t = src;
    src = dst;
    dst = t;
  }
  return src;
}
#ifndef L1_REGS
#define L1_REGS 0
#endif
template <int CTA>
__global__ void __launch_bounds__(CTA, L1_REGS ? 65536 / (CTA * (L1_REGS ? L1_REGS : 1)) : 1) l1_root2_k(const l1::Call* __restrict__ kc, const uint8_t* __restrict__ A,
                                                             u64* __restrict__ roots, u64* __restrict__ seed,
                                                             unsigned* __restrict__ ticket, int cap) {
  constexpr int LG = CTA == 64 ? 6 : 7;
  extern __shared__ u64 l1s[];
  __shared__ bool last;
  Dig* a = reinterpret_cast<Dig*>(l1s);
  Dig* b = a + cap;
  const l1::Call& k = *kc;
  const int t = threadIdx.x, nct = (int)gridDim.x;
  const u64 first = (u64)blockIdx.x * CTA;
  if (first + t < k.n) l1::a_leaf(k, first + t, A + (size_t)l1::SEG * (first + t), a[t]);
  __syncthreads();
  Dig* r = l1_fold2<CTA>(k, a, b, (int)(k.n - first < CTA ? k.n - first : CTA), 0, first);
  if (nct == 1) {
    if (!t) l1_seed(k, r[0], seed);
    return;
  }
  if (!t) {
    for (int i = 0; i < 8; i++) roots[8 * blockIdx.x + i] = r[0][i];
    __threadfence();
    last = atomicAdd(ticket, 1u) == (unsigned)nct - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int i = t; i < nct; i += CTA)
    for (int j = 0; j < 8; j++) a[i][j] = __ldcg(roots + 8 * i + j);
  if (!t) *ticket = 0;
  __syncthreads();
  r = l1_fold2<CTA>(k, a, b, nct, LG, 0);
  if (!t) l1_seed(k, r[0], seed);
}
// Thread i < m: row i's H coin words (from the one or two noise blocks they fall in) to coins, step 1's noise register,
// and lane 31 of steps 2..S as a byte store (bit t + 2, plus 1 under the tag layout; A's producer leaves the lane 0).
// Thread m + g, under the tag layout: noise block m H / 16 + g (m H is a multiple of 16) to coins, and G for columns
// 16 g .. 16 g + 15.
__global__ void __launch_bounds__(128) l1_nf_k(const l1::Call* __restrict__ kc, const u64* __restrict__ seed,
                                               u32* __restrict__ coins, uint8_t* __restrict__ A, uint8_t* __restrict__ B,
                                               int m, int n, int S, int BN, bool tag) {
  __shared__ u32 hw[128][17];
  const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int H = coin_words(S);
  u64 s[8];
  for (int i = 0; i < 8; i++) s[i] = seed[i];
  if (x < m) {
    const int i = (int)x;
    const long long w0 = (long long)i * H, b0 = w0 / 16, b1 = (w0 + H - 1) / 16;
    u32* h = hw[threadIdx.x];
    for (long long bk = b0; bk <= b1; bk++) {
      u64 o[8];
      l1::noise_of(*kc, s, (u64)bk, o);
#pragma unroll
      for (int q = 0; q < 16; q++) {
        const long long w = 16 * bk + q;
        if (w >= w0 && w < w0 + H) {
          const u32 u = q & 1 ? (u32)o[q / 2] : (u32)(o[q / 2] >> 32);
          h[w - w0] = u;
          coins[w] = u;
        }
      }
    }
    uint8_t* reg = A + (size_t)(i >> 6) * S * TA + kmaj_off(i & 63, 28, KB);
    *reinterpret_cast<u32*>(reg) = ((h[0] & 0xFu) * 0x00204081u) & 0x01010101u;
    for (int t = 1; t < S; t++) {
      const int p = t + 3;
      reg[(size_t)t * TA + 3] = (uint8_t)((tag ? 1u : 0u) + ((h[p >> 5] >> (p & 31)) & 1u));
    }
  } else if (tag && x < m + (n + 15) / 16) {
    const int g = (int)(x - m);
    const long long bk = (long long)m * H / 16 + g;
    u64 o[8];
    l1::noise_of(*kc, s, (u64)bk, o);
#pragma unroll
    for (int q = 0; q < 16; q++) {
      const u32 u = q & 1 ? (u32)o[q / 2] : (u32)(o[q / 2] >> 32);
      coins[16 * bk + q] = u;
      if (16 * g + q < n) g_store(B, u, 16 * g + q, S, BN);
    }
  }
}

__global__ void clock_k(Clk* clk, long long cycles) {
  const long long c0 = clock64();
  const u64 t0 = gtime();
  while (clock64() - c0 < cycles) {
  }
  *clk = Clk{c0, clock64(), t0, gtime()};
}

// ---- host ------------------------------------------------------------------------------------------------------------------

static u64 mix64(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

struct Lin {
  int m, n, K, S, BN;
  std::vector<int8_t> A, B;  // A[i * K + k], B[j * K + k] (B transposed: a column's K entries contiguous)
};
static void fill(Lin& L, int kind, u64 seed) {
  L.A.resize((size_t)L.m * L.K);
  L.B.resize((size_t)L.n * L.K);
  auto v = [&](u64 x) -> int8_t {
    if (kind == 1) return 127;
    if (kind == 2) return -127;
    if (kind == 3) return 0;
    return (int8_t)((int)(mix64(x) % 255) - 127);
  };
  for (size_t i = 0; i < L.A.size(); i++) L.A[i] = kind == 2 ? (int8_t)-127 : v(seed * 1000003 + i);
  for (size_t i = 0; i < L.B.size(); i++) L.B[i] = kind == 2 ? (int8_t)127 : v(~seed * 7919 + i);
}
// A' as its producer writes it (data lanes and, under the tag layout, the tags; noise lanes 0) and B' (G 0 until g_k) in
// the tiled layout
static std::vector<uint8_t> tile_a(const Lin& L) {
  std::vector<uint8_t> o((size_t)L.m * L.S * KB, 0);
  for (int i = 0; i < L.m; i++)
    for (int t = 0; t < L.S; t++)
      for (int l = 0; l < KB; l++)
        o[((size_t)(i >> 6) * L.S + t) * TA + kmaj_off(i & 63, l, KB)] =
            (uint8_t)(int8_t)lane_a(i, t, l, L.K, L.S, &L.A[(size_t)i * L.K], nullptr);
  return o;
}
static std::vector<uint8_t> tile_b(const Lin& L) {
  std::vector<uint8_t> o((size_t)L.n * L.S * KB, 0);
  const size_t TB = (size_t)L.BN * KB;
  for (int j = 0; j < L.n; j++)
    for (int t = 0; t < L.S; t++)
      for (int l = 0; l < KB; l++)
        o[((size_t)(j / L.BN) * L.S + t) * TB + kmaj_off(j % L.BN, l, KB)] =
            (uint8_t)(int8_t)lane_b(j, t, l, L.K, L.S, &L.B[(size_t)j * L.K], nullptr, false, 0);
  return o;
}
// the stock-shaped operands: 32 data lanes a step, no noise (the plain GEMM at the linear's own K)
static std::vector<uint8_t> tile_plain(const std::vector<int8_t>& X, int rows, int K, int R) {
  const int S0 = K / KB;
  std::vector<uint8_t> o((size_t)rows * K, 0);
  for (int i = 0; i < rows; i++)
    for (int t = 0; t < S0; t++)
      for (int l = 0; l < KB; l++)
        o[((size_t)(i / R) * S0 + t) * R * KB + kmaj_off(i % R, l, KB)] = (uint8_t)X[(size_t)i * K + KB * t + l];
  return o;
}
template <class T>
static T* up(const std::vector<T>& v) {
  T* p;
  CK(cudaMalloc(&p, v.size() * sizeof(T)));
  CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
static std::vector<u64> seed_msg(u64 call) {
  std::vector<u64> m(32, 0);
  for (int j = 0; j < 28; j++) m[j] = mix64(call * 64 + j);  // tag, salt, forward, position, root(A_j), root(B_j): 224 bytes
  m[28] = 0x8000000000000000ull;
  m[31] = 224 * 8;
  return m;
}
// a call's coin words: the rows' H each, under the tag layout n columns' G, under XU=1 a u word a row
static long long coin_words_of(int m, int S, int n) {
  return (long long)m * coin_words(S) + (g_tag ? n : 0) + (g_xu ? m : 0);
}
static long long coin_blocks(int m, int S, int n) { return (coin_words_of(m, S, n) + 15) / 16; }

template <int BN, bool EXPORT>
static void gemm_bn(const uint8_t* A, const uint8_t* B, const int* beta, int m, int n, int S, uint4* W, int* out, Clk* clk,
                    cudaStream_t st, const u32* cv, Xu x) {
  constexpr int smem = STAGES * (TA + BN * KB);
  static bool init = false;
  if (!init) {
    CK(cudaFuncSetAttribute(ssn_k<BN, EXPORT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    init = true;
  }
  ssn_k<BN, EXPORT><<<dim3(n / BN, m / 64), 128, smem, st>>>(A, B, beta, S, n, W, out, clk, cv, g_blc == 2, x);
  CK(cudaGetLastError());
}
static void gemm(int BN, bool ex, const uint8_t* A, const uint8_t* B, const int* beta, int m, int n, int S, uint4* W, int* out,
                 Clk* clk, cudaStream_t st, const u32* cv = nullptr, Xu x = {}) {
  if (BN == 256 && ex) gemm_bn<256, true>(A, B, beta, m, n, S, W, out, clk, st, cv, x);
  else if (BN == 256) gemm_bn<256, false>(A, B, beta, m, n, S, W, out, clk, st, cv, x);
  else if (ex) gemm_bn<128, true>(A, B, beta, m, n, S, W, out, clk, st, cv, x);
  else gemm_bn<128, false>(A, B, beta, m, n, S, W, out, clk, st, cv, x);
}
// blc4a-t64-e7's lane coins as the certifier streams them: a word a (64-row tile, step t >= 1, column), its bytes three
// coins odd and uniform in [-127, 127] and lane 31's public weight 1
static u32 lane_word(u64 x) {
  const u64 r = mix64(x);
  u32 w = 0x01000000u;
  for (int e = 0; e < 3; e++) w |= (u32)(uint8_t)(int8_t)(2 * (int)((r >> (8 * e)) & 127) - 127) << (8 * e);
  return w;
}
// a call's coins (parts & 1), forming (parts & 2) and, under the tag layout, G (parts & 4)
enum { COINS = 1, FORM = 2, GDRAW = 4 };
// XU=1: xa = xu_at(K, S), and xr the rows' sources
static void coins_form(const u64* msg, u32* coins, uint8_t* A, uint8_t* B, int m, int n, int S, int BN, cudaStream_t st,
                       int parts = COINS | FORM | GDRAW, int xa = 0, int* xr = nullptr) {
  const long long nb = coin_blocks(m, S, n);
  if (parts & COINS) coins_k<<<(unsigned)((nb + 127) / 128), 128, 0, st>>>(msg, coins, nb);
  if (g_tag && (parts & GDRAW))
    g_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(B, coins + (size_t)m * coin_words(S), n, S, BN, g_xu ? xa + 4 : -1);
  const long long r = (long long)m * S;
  const u32* uc = g_xu ? coins + (size_t)m * coin_words(S) + (g_tag ? n : 0) : nullptr;
  if (parts & FORM) form_k<<<(unsigned)((r + 255) / 256), 256, 0, st>>>(A, coins, m, S, g_tag, uc, xa, xu_e(xa), xr);
  CK(cudaGetLastError());
}

// One call's L1 seed path on the device: its constants, the CTAs' subtree roots and the seed (words 0-7; word 8 root(A_j)'s
// first word)
struct L1Dev {
  l1::Call* k;
  u64 *roots, *seed;
  unsigned* ticket;
  int nct, cta2, nct2;  // l1_root_k's CTAs; l1_root2_k's CTA size and CTAs
};
static int g_l1_cta = 64;  // L1_CTA: l1_root2_k's least CTA size, 64 or 128
static L1Dev l1_dev(const l1::Window& w, u64 call, u64 bytes) {
  const u64 n = bytes / l1::SEG;
  if (bytes % l1::SEG || n < 128 || (n & (n - 1)) || n > 65536) {
    fprintf(stderr, "ERR the L1 seed path takes a power-of-two count of 1 KiB leaves from 128 to 65,536\n");
    exit(2);
  }
  L1Dev d{};
  const l1::Call k = l1::call_of(w, call, n);
  CK(cudaMalloc(&d.k, sizeof k));
  CK(cudaMemcpy(d.k, &k, sizeof k, cudaMemcpyHostToDevice));
  d.nct = (int)((n + 255) / 256);
  d.cta2 = (int)std::max<u64>(g_l1_cta, n / 512);
  d.nct2 = (int)(n / d.cta2);
  CK(cudaMalloc(&d.roots, 64 * (size_t)std::max(d.nct, d.nct2)));
  CK(cudaMalloc(&d.seed, 128));
  CK(cudaMalloc(&d.ticket, sizeof(unsigned)));
  CK(cudaMemset(d.ticket, 0, sizeof(unsigned)));
  return d;
}
static void l1_free(const L1Dev& d) {
  CK(cudaFree(d.k)); CK(cudaFree(d.roots)); CK(cudaFree(d.seed)); CK(cudaFree(d.ticket));
}
// L1=2: root(A_j) and the seed, then (nf) the noise blocks, G and the forming of A' and B' on st
static void l1_path2(const L1Dev& d, const uint8_t* A, u32* coins, uint8_t* Ap, uint8_t* Bp, int m, int n, int S, int BN,
                     cudaStream_t st, bool nf = true) {
  static bool init = false;
  if (!init) {
    CK(cudaFuncSetAttribute(l1_root2_k<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * 96));
    CK(cudaFuncSetAttribute(l1_root2_k<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * 96));
    init = true;
  }
  const int cap = std::max(d.cta2, d.nct2);
  if (d.cta2 == 64) l1_root2_k<64><<<d.nct2, 64, (size_t)cap * 96, st>>>(d.k, A, d.roots, d.seed, d.ticket, cap);
  else l1_root2_k<128><<<d.nct2, 128, (size_t)cap * 96, st>>>(d.k, A, d.roots, d.seed, d.ticket, cap);
  if (nf) {
    if (m % 16) {
      fprintf(stderr, "ERR l1_nf_k takes rows in multiples of 16\n");
      exit(2);
    }
    const long long th = m + (g_tag ? (n + 15) / 16 : 0);
    l1_nf_k<<<(unsigned)((th + 127) / 128), 128, 0, st>>>(d.k, d.seed, coins, Ap, Bp, m, n, S, BN, g_tag);
  }
  CK(cudaGetLastError());
}
// root(A_j) and the seed (parts & COINS also the noise blocks) on st
static void l1_path(const L1Dev& d, const uint8_t* A, u32* coins, long long blocks, cudaStream_t st, int parts = COINS) {
  l1_root_k<<<d.nct, 256, 0, st>>>(d.k, A, d.roots, d.seed);
  if (d.nct > 1) l1_top_k<<<1, 256, 0, st>>>(d.k, d.roots, d.nct, d.seed);
  if (parts & COINS) l1_coins_k<<<(unsigned)((blocks + 127) / 128), 128, 0, st>>>(d.k, d.seed, coins, blocks);
  CK(cudaGetLastError());
}

// the device's seed path against l1::reference: the seed, root(A_j)'s first word and every noise block, over random and
// zero A_j
static int check_l1() {
  struct C { u64 rows, K, call; int zero; };
  const C cs[] = {{64, 2048, 5, 0}, {64, 8192, 9, 0}, {512, 8192, 63, 0}, {8192, 2048, 0, 0}, {8192, 8192, 255, 0}, {64, 2048, 1, 1}};
  int bad = 0;
  u64 sd = 1;
  for (const C& c : cs) {
    const l1::Window w = l1::window_of(sd++, 3, 256);
    std::vector<uint8_t> a(c.rows * c.K, 0);
    if (!c.zero)
      for (size_t i = 0; i < a.size(); i++) a[i] = (uint8_t)mix64(sd * 1000003 + i);
    const l1::Ref r = l1::reference(w, c.call, a.data(), a.size());
    const L1Dev d = l1_dev(w, c.call, a.size());
    uint8_t* dA = up(a);
    const long long blocks = 4096;
    u32* dc;
    CK(cudaMalloc(&dc, blocks * 64));
    CK(cudaMemset(d.seed, 0xA5, 128));
    CK(cudaMemset(dc, 0xA5, blocks * 64));
    l1_path(d, dA, dc, blocks, 0);
    CK(cudaDeviceSynchronize());
    u64 seed[9];
    std::vector<u32> coins(blocks * 16);
    CK(cudaMemcpy(seed, d.seed, sizeof seed, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(coins.data(), dc, blocks * 64, cudaMemcpyDeviceToHost));
    const bool seed_ok = l1::digest_bytes(seed) == r.seed && seed[8] == l1::be64(r.root_a.data());
    size_t bad_blocks = 0;
    for (long long b = 0; b < blocks; b++) {
      const l1::Bytes nb = l1::noise_block(r.seed, (u64)b);
      for (int j = 0; j < 16; j++) {
        const u32 want = (u32)nb[4 * j] << 24 | (u32)nb[4 * j + 1] << 16 | (u32)nb[4 * j + 2] << 8 | nb[4 * j + 3];
        if (coins[16 * b + j] != want) {
          bad_blocks++;
          break;
        }
      }
    }
    printf("{\"check\": \"l1_seed_path\", \"rows\": %llu, \"K\": %llu, \"call\": %llu, \"data\": \"%s\", \"leaves\": %llu, "
           "\"ctas\": %d, \"seed_and_root_a_match\": %s, \"noise_blocks\": %lld, \"noise_block_mismatches\": %zu}\n",
           (unsigned long long)c.rows, (unsigned long long)c.K, (unsigned long long)c.call, c.zero ? "zero" : "random",
           (unsigned long long)(a.size() / l1::SEG), d.nct, seed_ok ? "true" : "false", blocks, bad_blocks);
    fflush(stdout);
    bad += !seed_ok || bad_blocks;
    // L1=2 at both least CTA sizes: its seed against the reference, and l1_nf_k's coin words, A' and B' against l1_coins_k's
    // blocks with coins_form's G and forming, over A' and B' as their producer writes them, in both layouts
    const int keep_cta = g_l1_cta;
    const bool keep_tag = g_tag;
    const int keep_blc = g_blc;
    for (int cta : {64, 128}) {
      g_l1_cta = cta;
      const L1Dev d2 = l1_dev(w, c.call, a.size());
      for (int lay = 0; lay < 4; lay++) {
        const bool tag = lay > 0;
        g_tag = tag;
        g_blc = std::max(0, lay - 1);
        Lin L{(int)c.rows, 256, (int)c.K, steps_of((int)c.K), 256, {}, {}};
        fill(L, 0, sd * 31 + lay);
        const auto ta = tile_a(L), tb = tile_b(L);
        const int m = L.m, n = L.n, S = L.S;
        const long long nb = coin_blocks(m, S, n), nw = (long long)m * coin_words(S) + (tag ? n : 0);
        uint8_t *A1 = up(ta), *B1 = up(tb), *A2 = up(ta), *B2 = up(tb);
        u32 *c1, *c2;
        CK(cudaMalloc(&c1, nb * 64));
        CK(cudaMalloc(&c2, nb * 64));
        CK(cudaMemset(c2, 0xA5, nb * 64));
        CK(cudaMemset(d2.seed, 0xA5, 128));
        l1_path(d, dA, c1, nb, 0);
        coins_form(nullptr, c1, A1, B1, m, n, S, L.BN, 0, FORM | GDRAW);
        l1_path2(d2, dA, c2, A2, B2, m, n, S, L.BN, 0);
        CK(cudaDeviceSynchronize());
        u64 s2[9];
        CK(cudaMemcpy(s2, d2.seed, sizeof s2, cudaMemcpyDeviceToHost));
        const bool ok2 = l1::digest_bytes(s2) == r.seed && s2[8] == l1::be64(r.root_a.data());
        std::vector<uint8_t> fa1(ta.size()), fa2(ta.size()), fb1(tb.size()), fb2(tb.size());
        std::vector<u32> w1(nw), w2(nw);
        CK(cudaMemcpy(fa1.data(), A1, ta.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fa2.data(), A2, ta.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fb1.data(), B1, tb.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fb2.data(), B2, tb.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(w1.data(), c1, nw * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(w2.data(), c2, nw * 4, cudaMemcpyDeviceToHost));
        size_t da = 0, db = 0, dw = 0, formed = 0;
        for (size_t i = 0; i < ta.size(); i++) da += fa1[i] != fa2[i], formed += fa1[i] != ta[i];
        for (size_t i = 0; i < tb.size(); i++) db += fb1[i] != fb2[i];
        for (long long i = 0; i < nw; i++) dw += w1[i] != w2[i];
        printf("{\"check\": \"l1_seed_path2\", \"rows\": %llu, \"K\": %llu, \"call\": %llu, \"data\": \"%s\", \"cta\": %d, "
               "\"ctas\": %d, \"layout\": \"%s\", \"seed_and_root_a_match\": %s, \"coin_words\": %lld, "
               "\"coin_word_mismatches\": %zu, \"a_bytes\": %zu, \"a_bytes_formed\": %zu, \"a_byte_mismatches\": %zu, "
               "\"b_bytes\": %zu, \"b_byte_mismatches\": %zu}\n",
               (unsigned long long)c.rows, (unsigned long long)c.K, (unsigned long long)c.call, c.zero ? "zero" : "random",
               d2.cta2, d2.nct2, layout_name(), ok2 ? "true" : "false", nw, dw, ta.size(), formed, da,
               tb.size(), db);
        fflush(stdout);
        bad += !ok2 || dw || da || db;
        CK(cudaFree(A1)); CK(cudaFree(B1)); CK(cudaFree(A2)); CK(cudaFree(B2)); CK(cudaFree(c1)); CK(cudaFree(c2));
      }
      l1_free(d2);
    }
    g_l1_cta = keep_cta;
    g_tag = keep_tag;
    g_blc = keep_blc;
    CK(cudaFree(dA)); CK(cudaFree(dc));
    l1_free(d);
  }
  return bad;
}

// native_rows.cu (the served committer, linked as its own translation unit): hm96-sha512/row/v2 leaves of fixed-size rows
extern "C" int verity_rows_params_bytes();
extern "C" int verity_rows_build(const uint8_t* domain_id, const uint8_t* ks, const uint8_t* row_prefix, const uint8_t* salt_prefix,
                                 const uint8_t* leaf_prefix, const char* schema, int schema_len, uint32_t lb, void* out);
extern "C" int verity_rows_leaves_sf_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows,
                                            const uint8_t* salts, const uint8_t* tn, uint64_t first, uint64_t m, uint64_t n_total,
                                            uint8_t* out, uint8_t* bc, uint8_t* hs, uint8_t* scratch, uint64_t max_blocks,
                                            cudaStream_t st);
extern "C" int verity_rows_levels_launch(const void* params, const uint8_t* in, uint8_t* out, uint64_t n_in, int level,
                                         uint64_t base, int n_levels, cudaStream_t st);
extern "C" int verity_rows_level_launch(const void* params, const uint8_t* in, uint8_t* out, uint64_t n_out, int level,
                                        uint64_t first_index, uint64_t max_blocks, cudaStream_t st);
extern "C" int verity_rows_frame_launch(const void* params, const uint8_t* hs, uint64_t n_h, uint64_t first, uint64_t m,
                                        uint64_t n_total, uint8_t* out, uint64_t max_blocks, cudaStream_t st);
// run.sh's plain-leaf build of native_rows.cu (plain_rows.py): frame-v3-sha512 row leaves without hm96, L1's words
extern "C" int verity_rows_p_build(const uint8_t* domain_id, const uint8_t* ks, const uint8_t* row_prefix,
                                   const uint8_t* salt_prefix, const uint8_t* leaf_prefix, const char* schema, int schema_len,
                                   uint32_t lb, void* out);
extern "C" int verity_rows_p_leaves_sf_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows,
                                              const uint8_t* salts, const uint8_t* tn, uint64_t first, uint64_t m,
                                              uint64_t n_total, uint8_t* out, uint8_t* bc, uint8_t* hs, uint8_t* scratch,
                                              uint64_t max_blocks, cudaStream_t st);
// SSN_SMALL_ROWS: run.sh's second build of native_rows.cu (SMALL_THREADS), its blocks SSN_SMALL_ROWS threads and its symbols
// verity_rows_s_*; a leaves launch of fewer rows than SMALL_ROWS_MAX (default 2 blocks of 256 an SM) uses it, so a small
// call's rows spread over the SMs instead of sharing a few
#ifdef SSN_SMALL_ROWS
extern "C" int verity_rows_s_leaves_sf_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows,
                                              const uint8_t* salts, const uint8_t* tn, uint64_t first, uint64_t m,
                                              uint64_t n_total, uint8_t* out, uint8_t* bc, uint8_t* hs, uint8_t* scratch,
                                              uint64_t max_blocks, cudaStream_t st);
static u64 small_rows_max() {
  static const u64 v = getenv("SMALL_ROWS_MAX") ? strtoull(getenv("SMALL_ROWS_MAX"), nullptr, 10) : 2 * 132 * 256;
  return v;
}
#endif

// frame-v3-sha512's message heads (verity.ml.commitments.merkle): a leaf's before its 64-byte value (169 to 173 bytes by the
// leaf's index) and a node's before its left child; with what follows them, 2 and 3 compressions
constexpr int LEAF_HEAD = 171, NODE_HEAD = 139;
// n bytes of stand-in constants; slot > 0: each slot's first u32 is len, a message head's length (hm96_rows.cu's SLOT)
__global__ void fill_k(uint8_t* p, u64 n, u64 seed, int slot, int len) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n / 8; i += (u64)gridDim.x * blockDim.x) {
    u64 z = (i + seed) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    if (slot && (i * 8) % slot == 0) z = (z & ~0xffffffffull) | (u32)len;
    ((u64*)p)[i] = z;
  }
}
static uint8_t* dev_fill(u64 bytes, u64 seed, int slot = 0, int len = 0) {
  uint8_t* p;
  bytes = (bytes + 7) / 8 * 8;
  CK(cudaMalloc(&p, bytes));
  fill_k<<<1024, 256>>>(p, bytes, seed, slot, len);
  CK(cudaGetLastError());
  return p;
}

// The hash pass over a run of exported words, and roots. Every row is an hm96-sha512 leaf of a frame-v3-sha512 tree.
//   rb > 0: the served committer's hm96-sha512/row/v2 leaves of rb-byte rows (native_rows.cu: salts derived, the split
//           form), one thread a row; its tree by verity_rows_levels_launch.
//   rb < 0: L1's words: frame-v3-sha512's row leaves of -rb-byte rows over their sha512/row/v2 digests without hm96 (the
//           plain-leaf build: the served kernels less the salt, mask and hm96 leaf; an 18-byte stand-in schema, the
//           template's length); its tree as rb > 0's.
//   rb 0:   served_zk's sha512/row-seg/v1 rows (hm96_rows.cu): sz_segments hashes every 1 KiB segment, then sz_finish, a
//           thread a row, x = SHA-512(Q || the row's segment digests), the hm96 leaf under the row's salt and the frame
//           leaf; its tree by sz_level, a launch a level, or in sz_finish's one CTA for at most 256 rows.
// The constants (prefixes, keys, tables, salts, message heads) are stand-ins of the schemes' lengths: they set no
// compression count.
struct Hasher {
  int rb = 0;
  u64* mids = nullptr;
  uint8_t *dig = nullptr, *tn = nullptr, *leaves = nullptr, *scratch = nullptr, *lv = nullptr;
  uint8_t *hm = nullptr, *tables = nullptr, *salts = nullptr, *lheads = nullptr, *nheads = nullptr, *x = nullptr, *bc = nullptr,
          *tree = nullptr;
  std::vector<uint8_t> params;
  // max_bytes: the most bytes a call hashes; max_rows: the most rows (row-seg's rows are at least 32 KiB, or a root's rows)
  void init(size_t max_bytes, int row_bytes, u64 max_rows = 0) {
    rb = row_bytes;
    if (rb) {
      std::vector<uint8_t> c(1024);
      for (size_t i = 0; i < c.size(); i++) c[i] = (uint8_t)mix64(i + 99);
      params.resize(verity_rows_params_bytes());
      const char* schema = rb > 0 ? "hm96-sha512/row/v2" : "sha512/row/v2/l1-w";
      const auto build = rb > 0 ? verity_rows_build : verity_rows_p_build;
      if (build(&c[0], &c[64], &c[128], &c[256], &c[384], schema, (int)strlen(schema), (uint32_t)std::abs(rb), params.data())) {
        fprintf(stderr, "ERR verity_rows_build\n");
        exit(2);
      }
      std::vector<uint8_t> t(256 * 17 * 16);
      for (size_t i = 0; i < t.size(); i++) t[i] = (uint8_t)mix64(i + 5);
      tn = up(t);
      const size_t rows = max_bytes / std::abs(rb) + 2;
      size_t width = 1;
      while (width < rows) width <<= 1;
      CK(cudaMalloc(&leaves, width * 64));
      CK(cudaMalloc(&lv, width * 64));
      CK(cudaMalloc(&scratch, rows * 192));
      return;
    }
    const u64 rows = std::max<u64>(max_bytes / 32768 + 2, max_rows);
    // 64 segment prefixes' states (a row has at most 64 segments) and Q's; hm96's salt and leaf prefixes' states
    std::vector<uint32_t> pre(67 * 32);
    for (size_t i = 0; i < pre.size(); i++) pre[i] = (u32)mix64(i + 77);
    uint32_t* dpre = up(pre);
    CK(cudaMalloc(&mids, 67 * 8 * sizeof(u64)));
    Args a{};
    a.p[0] = (unsigned long long)dpre;
    a.p[1] = (unsigned long long)mids;
    a.i[0] = 67;
    sz_mid<<<1, 96>>>(a);
    CK(cudaGetLastError());
    hm = (uint8_t*)(mids + 65 * 8);
    tables = dev_fill(192 * 256 * 64, 3);
    salts = dev_fill(rows * 192, 4);
    lheads = dev_fill(rows * SLOT, 5, SLOT, LEAF_HEAD);
    nheads = dev_fill(rows * SLOT, 6, SLOT, NODE_HEAD);
    CK(cudaMalloc(&dig, std::max<u64>(max_bytes, rows * 1024) / 16 + 64));
    CK(cudaMalloc(&x, rows * 64));
    CK(cudaMalloc(&bc, rows * 128));
    CK(cudaMalloc(&tree, (4 * rows + 128) * 64));
    CK(cudaFuncSetAttribute(sz_segments, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 32 * 33 * 4));
  }
  // sz_finish over rows rows of s segments each (their digests in dig): frame leaves into tree's level 0; one CTA (at most
  // 256 rows) with width > 1 also builds the tree above them
  void finish(u64 rows, int s, int width, bool one_cta, cudaStream_t st) {
    Args a{};
    const void* p[10] = {dig, mids + 64 * 8, tables, salts, hm, lheads, nheads, x, bc, tree};
    for (int t = 0; t < 10; t++) a.p[t] = (unsigned long long)p[t];
    a.i[0] = (int)rows;
    a.i[1] = s;
    a.i[3] = 2;
    a.i[6] = width;
    const unsigned block = one_cta ? (unsigned)std::min<u64>(256, 32 * ((rows + 31) / 32)) : 128;
    sz_finish<<<one_cta ? 1 : (unsigned)((rows + 127) / 128), block, 0, st>>>(a);
    CK(cudaGetLastError());
  }
  void seg(const void* data, long long rows, int stride, int s, uint8_t* o, cudaStream_t st) {
    Args a{};
    a.p[0] = (unsigned long long)data;
    a.p[1] = (unsigned long long)mids;
    a.p[2] = (unsigned long long)o;
    a.i[0] = (int)rows;
    a.i[1] = stride;
    a.i[2] = s;
    a.i[3] = 1024;
    const long long total = rows * s;
    sz_segments<<<(unsigned)((total + 255) / 256), 256, 8 * 32 * 33 * 4, st>>>(a);
    CK(cudaGetLastError());
  }
  // The leaves of `rows` rows of row_bytes bytes each (row-seg: whole KiB), and their count. rb > 0 reads the rows' bytes
  // as rb-byte rows.
  u64 leaves_of(const uint8_t* data, u64 rows, u64 row_bytes, cudaStream_t st, u64 cap = 0) {
    if (rb) {
      const u64 b = (u64)std::abs(rb), n = (rows * row_bytes + b - 1) / b;
#ifdef SSN_SMALL_ROWS
      auto launch = n < small_rows_max() ? verity_rows_s_leaves_sf_launch : verity_rows_leaves_sf_launch;
#else
      auto launch = verity_rows_leaves_sf_launch;
#endif
      if (rb < 0) launch = verity_rows_p_leaves_sf_launch;
      const int rc = launch(params.data(), data, b, n, nullptr, tn, 0, n, n, leaves, nullptr, nullptr, scratch, cap, st);
      if (rc) {
        fprintf(stderr, "ERR verity_rows_leaves_sf_launch %d\n", rc);
        exit(2);
      }
      return n;
    }
    const int s = (int)(row_bytes / 1024);
    seg(data, (long long)rows, (int)row_bytes, s, dig, st);
    finish(rows, s, 1, false, st);
    return rows;
  }
  // The frame tree above n leaves (the last leaves_of's), its width the power of two at or above n. rb > 0 as the served
  // committer builds it (native_collect.cpp's rows_levels_above): pad leaves by its frame kernel, a level wider than 2^16
  // nodes a launch of a thread a parent, the narrower ones ten levels a launch. Row-seg as served_zk's rows_bench.py: its pads
  // are constants of the shape (not hashed a call), and a launch of sz_level a level.
  void tree_of(u64 n, cudaStream_t st) {
    u64 w = 1;
    while (w < n) w <<= 1;
    if (!rb) {
      for (u64 off = 0; w > 1; off += w, w /= 2) {
        Args a{};
        a.p[0] = (unsigned long long)tree;
        a.p[1] = (unsigned long long)nheads;
        a.i[0] = (int)off;
        a.i[1] = (int)(w / 2);
        sz_level<<<(unsigned)((w / 2 + 127) / 128), 128, 0, st>>>(a);
        CK(cudaGetLastError());
      }
      return;
    }
    int rc = w > n ? verity_rows_frame_launch(params.data(), nullptr, 0, n, w - n, n, leaves + 64 * n, 0, st) : 0;
    const uint8_t* in = leaves;
    uint8_t* out = lv;
    for (int level = 0; !rc && w > 1;) {
      if (w > (1u << 16)) {
        rc = verity_rows_level_launch(params.data(), in, out, w / 2, level, 0, 0, st);
        in = out;
        out += 64 * (w / 2);
        w /= 2;
        level++;
        continue;
      }
      int k = 0;
      while (k < 10 && (1ull << k) < w) k++;
      rc = verity_rows_levels_launch(params.data(), in, out, w, level, 0, k, st);
      in = out + 64 * (w - 2 * (w >> k));
      out += 64 * (w - (w >> k));
      w >>= k;
      level += k;
    }
    if (rc) {
      fprintf(stderr, "ERR the frame tree's launches: %d\n", rc);
      exit(2);
    }
  }
  // The hash pass over an exported run: a row an exported block (row-seg) or rb bytes; with_tree, the tree above them too
  void run(const uint4* W, long long blocks, int block_bytes, cudaStream_t st, u64 cap = 0, bool with_tree = false) {
    const u64 n = leaves_of((const uint8_t*)W, (u64)blocks, (u64)block_bytes, st, cap);
    if (with_tree) tree_of(n, st);
  }
  // A root over `rows` rows of row_bytes bytes (A_j's rows: row-seg's rows, or rb-byte rows over their bytes): the leaves,
  // then the tree. Row-seg with at most 256 rows: one sz_finish CTA does both.
  void root(const uint8_t* data, u64 rows, u64 row_bytes, cudaStream_t st) {
    if (!rb && rows <= 256) {
      const int s = (int)(row_bytes / 1024);
      seg(data, (long long)rows, (int)row_bytes, s, dig, st);
      int width = 1;
      while ((u64)width < rows) width *= 2;
      finish(rows, s, width, true, st);
      return;
    }
    tree_of(leaves_of(data, rows, row_bytes, st), st);
  }
};

// word (i, j, t) of an exported run: block (i / 64, j / BN, t), thread 32 w + 4 g + q, register 4 ni + e
static size_t word_at(int i, int j, int t, int S, int n, int BN) {
  const int r = i & 63, c = j % BN, w = r >> 4, g = r & 7, eh = (r >> 3) & 1, ni = c >> 3, q = (c & 7) >> 1, el = c & 1;
  const size_t blk = ((size_t)(i >> 6) * (n / BN) + j / BN) * S + t;
  return blk * (64 * (size_t)BN) + ((size_t)ni * 128 + 32 * w + 4 * g + q) * 4 + 2 * eh + el;
}

static int check_one(int m, int n, int K, int BN, int kind, u64 seed) {
  Lin L{m, n, K, steps_of(K), BN, {}, {}};
  fill(L, kind, seed);
  const int S = L.S, H = coin_words(S);
  std::vector<int> beta(n);
  for (int j = 0; j < n; j++) beta[j] = beta_of(j, S);
  const auto ta = tile_a(L), tb = tile_b(L);
  uint8_t* dA = up(ta);
  uint8_t* dB = up(tb);
  int* dbeta = up(beta);
  const std::vector<u64> msg = seed_msg(seed);
  u64* dmsg = up(msg);
  u32* dcoins;
  const long long nb = coin_blocks(m, S, n);
  CK(cudaMalloc(&dcoins, nb * 64));
  const size_t words = (size_t)m * n * S;
  uint4* dW;
  int* dout;
  CK(cudaMalloc(&dW, words * 4));
  CK(cudaMalloc(&dout, (size_t)m * n * 4));
  CK(cudaMemset(dW, 0xA5, words * 4));
  CK(cudaMemset(dout, 0xA5, (size_t)m * n * 4));
  std::vector<u32> lc;
  // kind 4: random data with every lane coin 127, sigma at its most
  if (g_blc)
    for (size_t x = 0; x < (size_t)(m / 64) * lc_steps(S) * n; x++)
      lc.push_back(kind == 4 ? 0x017F7F7Fu : lane_word(seed * 0x9E3779B97F4A7C15ull + x));
  u32* dlc = lc.empty() ? nullptr : up(lc);
  // XU=1 or FOLD=1: scales with 22-bit significands, the column source a (tile, row class, column) from the lane coins, and
  // the served outputs
  const bool served = g_xu || g_fold;
  const int xa = xu_at(K, S), xe = xu_e(xa);
  std::vector<float> si(m), sj(n);
  std::vector<float2> xj(n);
  std::vector<int> xc((size_t)(m / 64) * 2 * n);
  auto scale = [](u64 r, int lo) { return std::ldexp((float)((1u << 21) + (u32)(r & ((1u << 21) - 1))), -21 - lo - (int)((r >> 32) % 4)); };
  for (int i = 0; i < m; i++) si[i] = scale(mix64(seed * 31 + i), 4);
  for (int j = 0; j < n; j++) sj[j] = scale(mix64(~seed * 37 + j), 7), xj[j] = make_float2(sj[j], -12582912.0f * sj[j]);
  const int c1 = g_blc == 2;
  auto lane_sum = [&](int i, int t, int j) {
    if (!g_blc || t + c1 == 0) return 0;
    const u32 w = lc[((size_t)(i / 64) * lc_steps(S) + t - 1 + c1) * n + j];
    return coin_w(i, 0) * (int8_t)w + coin_w(i, 1) * (int8_t)(w >> 8) + coin_w(i, 2) * (int8_t)(w >> 16);
  };
  long long max_sigma = 0;
  for (int rt = 0; rt < m / 64; rt++)
    for (int cls = 0; cls < 2; cls++)
      for (int j = 0; j < n; j++) {
        int s = 0;
        for (int t = 0; t < S; t++) s += lane_sum(64 * rt + cls, t, j);
        xc[((size_t)rt * 2 + cls) * n + j] = col_source(beta[j], s);
        max_sigma = std::max(max_sigma, (long long)std::abs(s));
      }
  int* dxr = nullptr;
  u32* dxo = nullptr;
  Xu xu{};
  if (served) {
    CK(cudaMalloc(&dxr, m * 4));
    CK(cudaMalloc(&dxo, (size_t)m * n * 2));
    CK(cudaMemset(dxr, g_xu ? 0xA5 : 0, m * 4));
    CK(cudaMemset(dxo, 0xA5, (size_t)m * n * 2));
    xu = Xu{dxr, up(si), up(xc), up(xj), dxo, 2};
  }
  coins_form(dmsg, dcoins, dA, dB, m, n, S, BN, 0, COINS | FORM | GDRAW, xa, dxr);
  gemm(BN, true, dA, dB, dbeta, m, n, S, dW, dout, nullptr, 0, dlc, xu);
  CK(cudaDeviceSynchronize());
  std::vector<int> xr(m);
  std::vector<uint16_t> xo((size_t)m * n);
  if (served) {
    CK(cudaMemcpy(xr.data(), dxr, m * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(xo.data(), dxo, (size_t)m * n * 2, cudaMemcpyDeviceToHost));
  }
  std::vector<u32> coins(nb * 16);
  std::vector<int> W(words), out((size_t)m * n);
  std::vector<uint8_t> fa(ta.size()), fb(tb.size());
  CK(cudaMemcpy(coins.data(), dcoins, nb * 64, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(W.data(), dW, words * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(out.data(), dout, (size_t)m * n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(fa.data(), dA, ta.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(fb.data(), dB, tb.size(), cudaMemcpyDeviceToHost));
  auto bit = [&](int i, int p) { return (int)((coins[(size_t)i * H + (p >> 5)] >> (p & 31)) & 1u); };
  // G from the host's reading of column j's coin word, and B' against it (G's lanes) and against tile_b (every other lane)
  std::vector<int> G(4 * (size_t)n, 0);
  size_t bad_g = 0;
  if (g_tag)
    for (int j = 0; j < n; j++) {
      u64 v = ((u64)coins[(size_t)m * H + j] * 2401u) >> 32;
      for (int e = 0; e < 4; e++, v /= 7) G[4 * j + e] = (int)(v % 7) - 3;
    }
  for (int j = 0; j < n; j++)
    for (int t = 0; t < S; t++)
      for (int l = 0; l < KB; l++) {
        const size_t o = ((size_t)(j / BN) * S + t) * BN * KB + kmaj_off(j % BN, l, KB);
        const bool gl = g_tag && t == 0 && l >= 24 && l < 28, xl = g_xu && t == S - 1 && l >= xa + 4 && l < xa + 8;
        bad_g += gl ? (int8_t)fb[o] != G[4 * j + l - 24] : xl ? (int8_t)fb[o] != G[4 * j + l - xa - 4] : fb[o] != tb[o];
      }
  // every coin block against a host SHA-512: seed = SHA-512(seed message), block b = SHA-512(seed || be64 b)
  size_t bad_coin = 0;
  {
    u64 sd[8], st[8], w[16];
    hsha::iv(sd);
    hsha::compress(sd, &msg[0]);
    hsha::compress(sd, &msg[16]);
    for (long long b = 0; b < nb; b++) {
      for (int j = 0; j < 8; j++) w[j] = sd[j];
      w[8] = (u64)b;
      w[9] = 0x8000000000000000ull;
      for (int j = 10; j < 15; j++) w[j] = 0;
      w[15] = 72 * 8;
      hsha::iv(st);
      hsha::compress(st, w);
      for (int j = 0; j < 8; j++)
        bad_coin += (coins[16 * b + 2 * j] != (u32)(st[j] >> 32)) + (coins[16 * b + 2 * j + 1] != (u32)st[j]);
    }
  }
  // the formed A' against the spec (lane_a on the row's coin words and u word): data lanes unchanged, noise lanes the
  // row's bits
  size_t bad_form = 0;
  auto row_coins = [&](int i) {
    return RowCoins{&coins[(size_t)i * H], g_xu ? coins[(size_t)m * H + (g_tag ? n : 0) + i] : 0u};
  };
  for (int i = 0; i < m; i++) {
    const RowCoins rc = row_coins(i);
    for (int t = 0; t < S; t++)
      for (int l = 0; l < KB; l++)
        bad_form += (int8_t)fa[((size_t)(i >> 6) * S + t) * TA + kmaj_off(i & 63, l, KB)] !=
                    lane_a(i, t, l, K, S, &L.A[(size_t)i * K], &rc);
  }
  // every word against word (i, j, t) = D_t(i, j) - beta_j + sum_e a1[i, e] C[e, j] + sum_{2 <= t' <= t} a_t'[i], and
  // under the tag layout + sum_e T(i)[e] G[e, j] and 1 + a_t'[i] for a_t'[i], under blc4a-t64-e7 also + the three lane coins
  // of (i / 64, t', j) at each step t'; under XU=1 the last step's - sum_e a1[i, e] C[e, j] - sum_e T(i)[e] G[e, j] + u0 +
  // 127 u1 + 127 e u2. The served output against stock's epilogue, bf16(fl(fl(float(D) s_j) s_i)), on D = A_i . B_j (XU=1)
  // or on the final word less sigma (FOLD=1), where |D| <= 2^22, and the outputs past it counted
  size_t bad_w = 0, bad_o = 0, ones = 0, bad_xr = 0, bad_served = 0, past = 0;
  long long maxabs = 0;
  std::vector<int> code(4 * (size_t)n);
  for (int j = 0; j < n; j++)
    for (int e = 0; e < 4; e++) code[4 * j + e] = code_of(j, e);
  const int d1 = lanes1();
  for (int i = 0; i < m; i++) {
    const int8_t* a = &L.A[(size_t)i * K];
    int T[4];
    for (int e = 0; e < 4; e++) T[e] = tag_of(i, e);
    const u32 u = row_coins(i).u;
    const int u0 = u0_of(u), u1 = u1_of(u), u2 = u2_of(u);
    int rho = u0 + 127 * u1 + 127 * xe * u2;
    for (int t = 1; t < S; t++) rho += 1 + bit(i, t + 3);
    bad_xr += g_xu ? xr[i] != 0x4B400000 - rho : g_fold && xr[i] != 0;
    for (int j = 0; j < n; j++) {
      const int8_t* b = &L.B[(size_t)j * K];
      long long acc = -beta[j], ab = 0, sig = 0;
      for (int e = 0; e < 4; e++) acc += bit(i, e) * code[4 * j + e] + T[e] * G[4 * j + e];
      for (int t = 0; t < S; t++) {
        const int lo = t == 0 ? 0 : d1 + lanesl() * (t - 1), hi = std::min(K, t == 0 ? d1 : lo + lanesl());
        for (int k = lo; k < hi; k++) ab += (int)a[k] * (int)b[k];
        for (int k = lo; k < hi; k++) acc += (int)a[k] * (int)b[k];
        if (t > 0) acc += (g_tag ? 1 : 0) + bit(i, t + 3);
        acc += lane_sum(i, t, j);
        sig += lane_sum(i, t, j);
        if (g_xu && t == S - 1) {
          for (int e = 0; e < 4; e++) acc -= bit(i, e) * code[4 * j + e] + T[e] * G[4 * j + e];
          acc += u0 + 127 * u1 + 127 * xe * u2;
        }
        maxabs = std::max(maxabs, acc < 0 ? -acc : acc);
        bad_w += W[word_at(i, j, t, S, n, BN)] != acc;
      }
      bad_o += out[(size_t)i * n + j] != (int)acc;
      if (served) {
        const long long pv = g_xu ? ab : acc - sig;
        if (pv < -(1 << 22) || pv > (1 << 22)) {
          past++;
          continue;
        }
        bad_served += xo[(size_t)i * n + j] != bf16_rn(stock_fold((int)pv, sj[j], si[i]));
      }
    }
    for (int t = 1; t < S; t++) ones += bit(i, t + 3);
  }
  printf("{\"check\": \"%s\", \"m\": %d, \"n\": %d, \"K\": %d, \"S\": %d, \"BN\": %d, \"data\": \"%s\", \"words\": %zu, "
         "\"word_mismatches\": %zu, \"final_mismatches\": %zu, \"formed_mismatches\": %zu, \"b_mismatches\": %zu, "
         "\"max_abs_word\": %lld, \"step_bits_set\": %zu, \"step_bits\": %zu, \"coin_blocks\": %lld, "
         "\"coin_word_mismatches\": %zu, \"lane_coin_words\": %zu, \"blc_w\": [%d, %d, %d], \"xu_lane\": %d, \"xu_e\": %d, "
         "\"max_abs_sigma\": %lld, \"row_source_mismatches\": %zu, \"served_outputs\": %zu, \"served_mismatches\": %zu, "
         "\"served_past_2_22\": %zu}\n",
         layout_name(), m, n, K, S, BN,
         kind == 0 ? "random" : kind == 1 ? "max" : kind == 2 ? "min" : kind == 3 ? "zero" : "random_max_coins",
         words, bad_w, bad_o, bad_form, bad_g, maxabs, ones, (size_t)m * (S - 1), nb, bad_coin, lc.size(), g_lw[0], g_lw[1],
         g_lw[2], g_xu ? xa : -1, g_xu ? xe : 0, max_sigma, bad_xr, served ? (size_t)m * n - past : 0, bad_served, past);
  fflush(stdout);
  if (dlc) CK(cudaFree(dlc));
  if (served) {
    CK(cudaFree(dxr)); CK(cudaFree(dxo)); CK(cudaFree((void*)xu.s)); CK(cudaFree((void*)xu.c)); CK(cudaFree((void*)xu.j));
  }
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dbeta)); CK(cudaFree(dmsg)); CK(cudaFree(dcoins));
  CK(cudaFree(dW)); CK(cudaFree(dout));
  return bad_w || bad_o || bad_form || bad_g || bad_coin || bad_xr || bad_served;
}

static int check() {
  {
    u64 h[8], m[16] = {0x6162638000000000ull, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24};
    hsha::iv(h);
    hsha::compress(h, m);
    if (h[0] != 0xddaf35a193617abaull || h[7] != 0x2a9ac94fa54ca49full) {
      fprintf(stderr, "ERR host SHA-512 of \"abc\"\n");
      return 1;
    }
  }
  struct C { int m, n, K, BN, kind; };
  const C cs[] = {{128, 512, 2048, 256, 0}, {128, 512, 2048, 128, 0}, {64, 256, 8192, 256, 0}, {64, 512, 100, 256, 0},
                  {64, 256, 28, 256, 0},    {64, 256, 59, 128, 0},    {64, 16384, 90, 256, 0}, {64, 256, 8192, 256, 1},
                  {64, 256, 8192, 256, 2},  {128, 256, 2048, 256, 3}, {192, 768, 2048, 256, 0}, {64, 256, 24, 256, 0},
                  {64, 256, 25, 128, 0},    {28544, 256, 24, 256, 0}, {64, 256, 8192, 256, 4}, {128, 512, 2048, 128, 4}};
  // T is injective on the rows its 4 digits reach
  {
    std::map<std::vector<int>, int> seen;
    for (int i = 0; i < 28560; i++) {
      std::vector<int> t(4);
      for (int e = 0; e < 4; e++) t[e] = tag_of(i, e);
      if (!seen.emplace(t, i).second) {
        fprintf(stderr, "ERR rows %d and %d share a tag\n", seen[t], i);
        return 1;
      }
    }
  }
  int bad = 0;
  // the four layouts, then XU=1 on the three tag layouts, then blc4a-t64-e7-s1 under ROT=1 without XU, with XU=1 and with
  // FOLD=1
  for (int lay = 0; lay < 10; lay++) {
    g_rot = lay > 6;
    g_xu = (lay > 3 && lay < 7) || lay == 8;
    g_fold = lay == 9;
    g_tag = lay > 0;
    g_blc = g_rot ? 2 : std::max(0, (g_xu ? lay - 3 : lay) - 1);
    u64 seed = 1;
    for (const C& c : cs) bad += check_one(c.m, c.n, c.K, c.BN, c.kind, seed++);
  }
  g_rot = false;
  g_xu = false;
  g_fold = false;
  g_tag = false;
  g_blc = 0;
  bad += check_l1();
  return bad ? 1 : 0;
}

// ---- timing ----------------------------------------------------------------------------------------------------------------
static int g_sms = 132;
struct Timer {
  cudaEvent_t a, b;
  Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
  template <class F> double ms(int reps, F f, cudaStream_t st = 0) {
    f();
    CK(cudaDeviceSynchronize());
    std::vector<double> v;
    for (int r = 0; r < reps; r++) {
      CK(cudaEventRecord(a, st));
      f();
      CK(cudaEventRecord(b, st));
      CK(cudaEventSynchronize(b));
      float t = 0;
      CK(cudaEventElapsedTime(&t, a, b));
      v.push_back(t);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  }
};
static double sm_mhz(Clk* dclk) {
  clock_k<<<1, 1>>>(dclk, 20000000);
  CK(cudaDeviceSynchronize());
  Clk c;
  CK(cudaMemcpy(&c, dclk, sizeof c, cudaMemcpyDeviceToHost));
  return (double)(c.c1 - c.c0) / (double)(c.t1 - c.t0) * 1e3;
}
static double units(double ms, double mhz) { return ms * 1e-3 * mhz * 1e6 * g_sms * 4096.0; }

struct LinDev {
  const char* name;
  int n, K, S, BN, chunk;  // chunk: rows a hash chunk
  uint8_t *A, *B, *A0, *B0;
  int* beta;
  u64* msg;
  u32* coins;
  std::vector<L1Dev> l1;  // L1=1: each layer's call of this linear
  // the served epilogue: xu's first lane, the row sources (XU=1, written by the forming), the column sources (host and
  // device), the row and column scales
  int xa = 0;
  int *xr = nullptr, *xc = nullptr, *hxc = nullptr;
  float* xs = nullptr;
  float2* xj = nullptr;
};

// One JSON line a measurement, at M rows: per linear the plain GEMM at its own K, ssn_1l's GEMM without and with export,
// the hash pass over the exported words, coins, forming and coins + forming as one call's latency; then the forward's
// 16 layers end to end, serial (each chunk's GEMM with export, then its hash, one stream) and beside (chunk c's hash on
// buffer c mod B's stream while the GEMM stream runs on), with coins and forming before each linear's GEMM.
static int time_all(int reps, int M, double budget_gb, const std::vector<int>& rbs) {
  struct Spec { const char* name; int n, K; };
  const Spec specs[] = {{"qkv", 3072, 2048}, {"o", 2048, 2048}, {"gate_up", 16384, 2048}, {"down", 2048, 8192}};
  Clk* dclk;
  CK(cudaMalloc(&dclk, sizeof(Clk)));
  std::vector<LinDev> lins;
  size_t max_chunk_words = 0;
  const size_t budget = (size_t)(budget_gb * 1e9);
  for (const Spec& s : specs) {
    Lin L{M, s.n, s.K, steps_of(s.K), ((M / 64) * (s.n / 256) < g_sms) ? 128 : 256, {}, {}};
    fill(L, 0, 11 + s.n + s.K);
    LinDev d{s.name, s.n, s.K, L.S, L.BN, 0};
    d.A = up(tile_a(L));
    d.B = up(tile_b(L));
    d.A0 = up(tile_plain(L.A, M, s.K, 64));
    d.B0 = up(tile_plain(L.B, s.n, s.K, L.BN));
    std::vector<int> beta(s.n);
    for (int j = 0; j < s.n; j++) beta[j] = beta_of(j, L.S);
    d.beta = up(beta);
    d.msg = up(seed_msg(s.n));
    CK(cudaMalloc(&d.coins, coin_blocks(M, L.S, s.n) * 64));
    const size_t row_bytes = (size_t)s.n * L.S * 4;
    d.chunk = (int)std::min<size_t>(M, std::max<size_t>(64, budget / row_bytes / 64 * 64));
    max_chunk_words = std::max(max_chunk_words, (size_t)d.chunk * s.n * L.S);
    d.xa = xu_at(s.K, L.S);
    std::vector<float> xs(M);
    std::vector<float2> xj(s.n);
    for (int i = 0; i < M; i++) xs[i] = std::ldexp((float)((1u << 21) + (u32)(mix64(i + 3) & ((1u << 21) - 1))), -27);
    for (int j = 0; j < s.n; j++) {
      const float sj = std::ldexp((float)((1u << 21) + (u32)(mix64(j + 5) & ((1u << 21) - 1))), -30);
      xj[j] = make_float2(sj, -12582912.0f * sj);
    }
    d.xs = up(xs);
    d.xj = up(xj);
    CK(cudaMalloc(&d.xr, (size_t)M * 4));
    CK(cudaMemset(d.xr, 0, (size_t)M * 4));
    CK(cudaMallocHost(&d.hxc, (size_t)(M / 64) * 2 * s.n * 4));
    for (size_t x = 0; x < (size_t)(M / 64) * 2 * s.n; x++) d.hxc[x] = col_source(beta[x % s.n], 0);
    CK(cudaMalloc(&d.xc, (size_t)(M / 64) * 2 * s.n * 4));
    lins.push_back(d);
  }
  // BLC=1: the certifier's link on a copy stream. A call sends A_j (M K bytes, device to host) once A_j is ready, and its lane
  // coins come back a chunk at a time (host to device) ahead of the chunk's GEMM, into one of two buffers by call parity.
  // The certifier's own hash of A_j and the network's latency aren't modelled.
  size_t lc_max = 0;
  for (const LinDev& d : lins) lc_max = std::max(lc_max, (size_t)(M / 64) * lc_steps(d.S) * d.n);
  u32 *hlc = nullptr, *dlc[2] = {nullptr, nullptr};
  uint8_t* hA = nullptr;
  cudaStream_t sc = nullptr;
  cudaEvent_t ev_a = nullptr, ev_c = nullptr;
  if (g_blc) {
    CK(cudaMallocHost(&hlc, lc_max * 4));
    for (size_t x = 0; x < lc_max; x++) hlc[x] = lane_word(x);
    for (u32*& p : dlc) {
      CK(cudaMalloc(&p, lc_max * 4));
      CK(cudaMemcpy(p, hlc, lc_max * 4, cudaMemcpyHostToDevice));
    }
    // the column sources (col_source less sigma_{tile, class, j}), which the certifier streams with the lane coins
    for (LinDev& d : lins) {
      const int c1 = g_blc == 2, ls = lc_steps(d.S);
      for (int rt = 0; rt < M / 64; rt++)
        for (int cls = 0; cls < 2; cls++)
          for (int j = 0; j < d.n; j++)
            d.hxc[((size_t)rt * 2 + cls) * d.n + j] -=
                xu_sigma(cls, d.S, [&](int t) { return hlc[((size_t)rt * ls + t - 1 + c1) * d.n + j]; });
    }
  }
  for (LinDev& d : lins) CK(cudaMemcpy(d.xc, d.hxc, (size_t)(M / 64) * 2 * d.n * 4, cudaMemcpyHostToDevice));
  if (g_blc) {
    CK(cudaMallocHost(&hA, (size_t)M * 8192));
    CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ev_a, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&ev_c, cudaEventDisableTiming));
  }
  auto lane_at = [&](const LinDev& d, int par, int r0) -> const u32* {
    return g_blc ? dlc[par] + (size_t)(r0 / 64) * lc_steps(d.S) * d.n : nullptr;
  };
  // L1=1: root(A_j) and the seed by L1 + (a)'s seed path (l1_path; ROOT_A and the call's seed message unused), call 4 l + i
  // the forward's linear i in layer l; L1=2 by l1_path2 (CTAs of at least L1_CTA threads)
  const int l1_mode = getenv("L1") ? atoi(getenv("L1")) : 0;
  const bool use_l1 = l1_mode > 0;
  if (getenv("L1_CTA")) g_l1_cta = atoi(getenv("L1_CTA")) == 128 ? 128 : 64;
  if (use_l1) {
    const l1::Window w = l1::window_of(7, 0, 64);
    for (size_t i = 0; i < lins.size(); i++)
      for (int l = 0; l < 16; l++) lins[i].l1.push_back(l1_dev(w, 4 * l + i, (u64)M * lins[i].K));
  }
  // beside's buffers, each with its own hash stream: as many as fit in 48 GB, 2 to NBUF (default 8)
  const size_t nbuf_max = getenv("NBUF") ? (size_t)std::max(2, atoi(getenv("NBUF"))) : 8;
  const int nbuf = (int)std::max<size_t>(2, std::min<size_t>(nbuf_max, (size_t)48e9 / (max_chunk_words * 4)));
  std::vector<uint4*> Wb(nbuf);
  for (auto& w : Wb) CK(cudaMalloc(&w, max_chunk_words * 4 + 2 * 65536));
  std::vector<Hasher> hsv(nbuf * rbs.size());
  for (size_t h = 0; h < hsv.size(); h++) hsv[h].init(max_chunk_words * 4, rbs[h / nbuf]);
  int hi = 0;  // the hasher in use: hsv[nbuf hi + b] for buffer b
  // beside's cap on a row hash launch's blocks (HASH_CAP; 0 none): the committer's max_blocks, which leaves SM slots to the
  // GEMM stream's export
  const u64 hash_cap = getenv("HASH_CAP") ? strtoull(getenv("HASH_CAP"), nullptr, 10) : 0;
  // root(A_j), which each call's seed reads, over A_j's M·K bytes (Hasher::root: the leaves and the tree above them), on
  // Hashers of its own so it never shares buffers with the words' hash streams. Timed a call by each of ROOT_A_LIST's
  // schemes (row bytes, 0 row-seg; default the words'). ROOT_A=1 puts it in the forward, ahead of the coins, by the words'
  // scheme, or by ROOT_A_ROWS's when that is set.
  const int root_a = getenv("ROOT_A") ? atoi(getenv("ROOT_A")) : 0;
  const int root_a_rows = getenv("ROOT_A_ROWS") ? atoi(getenv("ROOT_A_ROWS")) : -1;
  std::vector<int> ras = rbs;
  if (const char* e = getenv("ROOT_A_LIST")) {
    ras.clear();
    for (char* p = (char*)e; *p;) {
      char* q;
      const long v = strtol(p, &q, 10);
      if (q == p) break;
      ras.push_back((int)v);
      p = q;
    }
  }
  std::map<int, Hasher> hra;
  for (int r : ras) hra[r];
  for (int r : rbs) hra[r];
  if (root_a_rows >= 0) hra[root_a_rows];
  for (auto& [r, h] : hra) h.init((size_t)M * 8192, r, (u64)M);
#ifdef SSN_SMALL_ROWS
  // the small-block build's leaves against the served build's, over each linear's A_j by each committer scheme
  for (auto& [r, h] : hra) {
    if (r <= 0) continue;
    for (const LinDev& d : lins) {
      const u64 n = ((u64)M * d.K + r - 1) / r;
      std::vector<uint8_t> want(64 * n), got(64 * n);
      for (int small = 0; small < 2; small++) {
        const auto launch = small ? verity_rows_s_leaves_sf_launch : verity_rows_leaves_sf_launch;
        if (launch(h.params.data(), d.A0, r, n, nullptr, h.tn, 0, n, n, h.leaves, nullptr, nullptr, h.scratch, 0, 0)) {
          fprintf(stderr, "ERR the %s build's leaves\n", small ? "small-block" : "served");
          return 6;
        }
        CK(cudaMemcpy(small ? got.data() : want.data(), h.leaves, 64 * n, cudaMemcpyDeviceToHost));
      }
      u64 bad = 0;
      for (u64 i = 0; i < n; i++) bad += memcmp(&want[64 * i], &got[64 * i], 64) != 0;
      printf("{\"what\": \"small_leaves_check\", \"hasher\": \"rows%d\", \"linear\": \"%s\", \"M\": %d, \"leaves\": %llu, "
             "\"small_threads\": %d, \"mismatched_leaves\": %llu}\n",
             r, d.name, M, (unsigned long long)n, SSN_SMALL_ROWS, (unsigned long long)bad);
      if (bad) return 6;
    }
  }
#endif
  auto fwd_root_a = [&](const LinDev& d, cudaStream_t st) {
    if (root_a) hra[root_a_rows >= 0 ? root_a_rows : rbs[hi]].root(d.A0, (u64)M, (u64)d.K, st);
  };
  // WORDS_TREE=1: each chunk's hash in the forwards builds the frame tree above its leaves (the levels above the chunks'
  // subtrees, a few nodes a linear, aren't run)
  const bool words_tree = getenv("WORDS_TREE") && atoi(getenv("WORDS_TREE"));
  auto rname = [](int rb) { return rb ? (rb > 0 ? "rows" : "plain") + std::to_string(std::abs(rb)) : std::string("rowseg"); };
  auto hname = [&](int h) { return h < 0 ? std::string("none") : rname(rbs[h]); };
  int* dout;
  CK(cudaMalloc(&dout, (size_t)M * 16384 * 4));
  // the served epilogue in each GEMM, on the chunk at r0: XU=1 and FOLD=1 xu's (FOLD=1 the lead's, folding out the lane
  // coins), EPI=1 stock's on the final words, else none (the final words are only exported)
  const int epi = g_xu || g_fold ? 2 : getenv("EPI") && atoi(getenv("EPI")) ? 1 : 0;
  auto xu_of = [&](const LinDev& d, int r0) {
    return Xu{d.xr + r0, d.xs + r0, d.xc + (size_t)(r0 / 64) * 2 * d.n, d.xj, (u32*)dout + (size_t)r0 * d.n / 2, epi};
  };
  Timer T;
  cudaStream_t s1;
  std::vector<cudaStream_t> s2(nbuf);
  // PRIO=1: the GEMM stream (root(A_j), coins, forming, the GEMMs) at the device's greatest priority, the hash streams at
  // its least, so a block of the forward's data path is scheduled ahead of the words' hashing
  const bool prio = getenv("PRIO") && atoi(getenv("PRIO"));
  int least = 0, greatest = 0;
  if (prio) CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  CK(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, greatest));
  // SPLIT=R: beside's critical stream sb (root(A_j), coins, forming, the link's waits, the GEMMs) in a green context of R SMs
  // (the device rounds R up to its partition size) and the hash streams in one of the other SMs, so neither stream's chains
  // share an SM with the other's. The serial forward isn't timed under SPLIT; the plain GEMM runs on s1, every SM.
  const int split = getenv("SPLIT") ? atoi(getenv("SPLIT")) : 0;
  cudaStream_t sb = s1;
  unsigned split_sms[2] = {0, 0};
  CUcontext cx_a = nullptr, cx_b = nullptr;
  if (split) {
    CUdevice dev;
    CU(cuDeviceGet(&dev, 0));
    CUdevResource all, part, rest;
    CU(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
    unsigned groups = 1;
    CU(cuDevSmResourceSplitByCount(&part, &groups, &all, &rest, 0, (unsigned)split));
    CUdevResourceDesc da, db;
    CU(cuDevResourceGenerateDesc(&da, &part, 1));
    CU(cuDevResourceGenerateDesc(&db, &rest, 1));
    CUgreenCtx ga, gb;
    CU(cuGreenCtxCreate(&ga, da, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CU(cuGreenCtxCreate(&gb, db, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CU(cuCtxFromGreenCtx(&cx_a, ga));
    CU(cuCtxFromGreenCtx(&cx_b, gb));
    split_sms[0] = part.sm.smCount;
    split_sms[1] = rest.sm.smCount;
    CU(cuGreenCtxStreamCreate((CUstream*)&sb, ga, CU_STREAM_NON_BLOCKING, greatest));
    for (auto& s : s2) CU(cuGreenCtxStreamCreate((CUstream*)&s, gb, CU_STREAM_NON_BLOCKING, least));
  } else {
    for (auto& s : s2) CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, least));
  }
  // an event is recorded only on a stream of the context it was created in
  auto event_in = [](CUcontext cx, unsigned flags) {
    cudaEvent_t e;
    if (cx) CU(cuCtxPushCurrent(cx));
    CU(cuEventCreate((CUevent*)&e, flags));
    if (cx) {
      CUcontext o;
      CU(cuCtxPopCurrent(&o));
    }
    return e;
  };
  Timer TB;
  if (split) {
    TB.a = event_in(cx_a, CU_EVENT_DEFAULT);
    TB.b = event_in(cx_a, CU_EVENT_DEFAULT);
    if (g_blc) ev_a = event_in(cx_a, CU_EVENT_DISABLE_TIMING);
  }
  const double f0 = sm_mhz(dclk);
  auto line = [&](const char* what, const LinDev* d, double ms, double macs, double words, double extra, std::string hn = "") {
    const double u = units(ms, f0);
    if (hn.empty()) hn = std::strstr(what, "hash") || std::strstr(what, "fwd_s") || std::strstr(what, "fwd_b") ? hname(hi) : "none";
    printf("{\"what\": \"%s\", \"hasher\": \"%s\", \"linear\": \"%s\", \"M\": %d, \"n\": %d, \"K\": %d, \"S\": %d, \"BN\": %d, \"ms\": %.6f, "
           "\"mhz\": %.1f, \"units\": %.6e, \"stock_macs\": %.6e, \"units_per_stock_mac\": %.6f, \"words\": %.6e, "
           "\"units_per_word\": %.3f, \"units_per_byte\": %.3f, \"extra\": %.6g}\n",
           what, hn.c_str(),
           d ? d->name : "all", M, d ? d->n : 0, d ? d->K : 0, d ? d->S : 0, d ? d->BN : 0, ms, f0, u, macs,
           macs > 0 ? u / macs : 0, words, words > 0 ? u / words : 0, words > 0 ? u / (4 * words) : 0, extra);
    fflush(stdout);
  };
  double fwd_macs = 0;
  for (const LinDev& d : lins) fwd_macs += 16.0 * M * d.n * (double)d.K;
  for (const LinDev& d : lins) {
    const double macs = (double)M * d.n * d.K, words = (double)M * d.n * d.S;
    line("plain_gemm", &d, T.ms(reps, [&] { gemm(d.BN, false, d.A0, d.B0, nullptr, M, d.n, d.K / KB, nullptr, nullptr, nullptr, 0); }),
         macs, 0, 0);
    line("ssn_gemm", &d, T.ms(reps, [&] { gemm(d.BN, false, d.A, d.B, d.beta, M, d.n, d.S, nullptr, nullptr, nullptr, 0, lane_at(d, 0, 0), xu_of(d, 0)); }),
         macs, 0, 0);
    double ex = 0;
    std::vector<double> hsh(rbs.size(), 0.0), tre(rbs.size(), 0.0);
    for (int r0 = 0; r0 < M; r0 += d.chunk) {
      const int mc = std::min(d.chunk, M - r0);
      const uint8_t* A = d.A + (size_t)(r0 / 64) * d.S * TA;
      ex += T.ms(reps, [&] { gemm(d.BN, true, A, d.B, d.beta, mc, d.n, d.S, Wb[0], nullptr, nullptr, 0, lane_at(d, 0, r0), xu_of(d, r0)); });
      const long long blocks = (long long)(mc / 64) * (d.n / d.BN) * d.S;
      for (size_t h = 0; h < rbs.size(); h++) {
        Hasher& hs = hsv[nbuf * h];
        u64 n = 0;
        hsh[h] += T.ms(reps, [&] { n = hs.leaves_of((const uint8_t*)Wb[0], (u64)blocks, 64ull * d.BN * 4, 0); });
        tre[h] += T.ms(reps, [&] { hs.tree_of(n, 0); });
      }
    }
    line("ssn_gemm_export", &d, ex, macs, words, d.chunk);
    if (g_blc) {
      const size_t lb = (size_t)(M / 64) * lc_steps(d.S) * d.n * 4;
      line("lane_link", &d, T.ms(reps, [&] {
             CK(cudaMemcpyAsync(hA, d.A0, (size_t)M * d.K, cudaMemcpyDeviceToHost, 0));
             CK(cudaMemcpyAsync(dlc[1], hlc, lb, cudaMemcpyHostToDevice, 0));
           }), macs, words, (double)lb);
    }
    for (hi = 0; hi < (int)rbs.size(); hi++) line("hash_pass", &d, hsh[hi], macs, words, d.chunk);
    for (hi = 0; hi < (int)rbs.size(); hi++) line("hash_tree", &d, tre[hi], macs, words, d.chunk);
    hi = -1;
    auto cf = [&](int parts) { coins_form(d.msg, d.coins, d.A, d.B, M, d.n, d.S, d.BN, 0, parts, d.xa, d.xr); };
    line("coins", &d, T.ms(std::max(reps, 51), [&] { cf(COINS); }), macs, words, (double)coin_blocks(M, d.S, d.n));
    line("forming", &d, T.ms(std::max(reps, 51), [&] { cf(FORM); }), macs, words, (double)M * d.S);
    if (g_tag) line("g_draw", &d, T.ms(std::max(reps, 51), [&] { cf(GDRAW); }), macs, words, (double)d.n);
    line("coins_forming", &d, T.ms(std::max(reps, 51), [&] { cf(COINS | FORM | GDRAW); }), macs, words, (double)M * d.S);
    const u64 abytes = (u64)M * d.K;
    if (use_l1) {
      const long long nbk = coin_blocks(M, d.S, d.n);
      line("l1_root", &d, T.ms(std::max(reps, 51), [&] { l1_path(d.l1[0], d.A0, d.coins, nbk, 0, 0); }), macs, abytes / 4.0,
           (double)abytes, "l1");
      line("l1_seed_path", &d, T.ms(std::max(reps, 51), [&] {
             l1_path(d.l1[0], d.A0, d.coins, nbk, 0);
             cf(FORM | GDRAW);
           }), macs, abytes / 4.0, (double)abytes, "l1");
      line("l1_root2", &d, T.ms(std::max(reps, 51), [&] { l1_path2(d.l1[0], d.A0, d.coins, d.A, d.B, M, d.n, d.S, d.BN, 0, false); }),
           macs, abytes / 4.0, d.l1[0].cta2, "l1");
      line("l1_seed_path2", &d, T.ms(std::max(reps, 51), [&] { l1_path2(d.l1[0], d.A0, d.coins, d.A, d.B, M, d.n, d.S, d.BN, 0); }),
           macs, abytes / 4.0, d.l1[0].cta2, "l1");
    }
    for (int r : ras) {
      Hasher& h = hra[r];
      line("root_a_hash", &d, T.ms(std::max(reps, 51), [&] { h.leaves_of(d.A0, (u64)M, (u64)d.K, 0); }), macs, abytes / 4.0,
           (double)abytes, rname(r));
      line("root_a_root", &d, T.ms(std::max(reps, 51), [&] { h.root(d.A0, (u64)M, (u64)d.K, 0); }), macs, abytes / 4.0,
           (double)abytes, rname(r));
    }
  }
  double fwd_words = 0;
  for (const LinDev& d : lins) fwd_words += 16.0 * M * d.n * d.S;
  // forward: plain, ssn without export, serial, beside
  auto fwd_plain = [&] {
    for (int l = 0; l < 16; l++)
      for (const LinDev& d : lins) gemm(d.BN, false, d.A0, d.B0, nullptr, M, d.n, d.K / KB, nullptr, nullptr, nullptr, s1);
  };
  line("fwd_plain_gemm", nullptr, T.ms(reps, fwd_plain, s1), fwd_macs, 0, 0);
  // a call's work before its GEMM: root(A_j) (ROOT_A) and coins from the seed message, or L1's seed path; then G and forming
  auto pre = [&](const LinDev& d, int l, cudaStream_t st) {
    if (l1_mode == 2) {
      l1_path2(d.l1[l], d.A0, d.coins, d.A, d.B, M, d.n, d.S, d.BN, st);
    } else if (use_l1) {
      l1_path(d.l1[l], d.A0, d.coins, coin_blocks(M, d.S, d.n), st);
      coins_form(d.msg, d.coins, d.A, d.B, M, d.n, d.S, d.BN, st, FORM | GDRAW, d.xa, d.xr);
    } else {
      fwd_root_a(d, st);
      coins_form(d.msg, d.coins, d.A, d.B, M, d.n, d.S, d.BN, st, COINS | FORM | GDRAW, d.xa, d.xr);
    }
  };
  // BLC=1: A_j to the certifier once A_j is ready (st's work so far), then each chunk's lane coins back ahead of its GEMM
  auto link_a = [&](const LinDev& d, cudaStream_t st) {
    if (!g_blc) return;
    CK(cudaEventRecord(ev_a, st));
    CK(cudaStreamWaitEvent(sc, ev_a));
    CK(cudaMemcpyAsync(hA, d.A0, (size_t)M * d.K, cudaMemcpyDeviceToHost, sc));
  };
  auto link_c = [&](const LinDev& d, int par, int r0, int mc, bool copy, cudaStream_t st) -> const u32* {
    const u32* p = lane_at(d, par, r0);
    if (p && copy) {
      const size_t off = (size_t)(r0 / 64) * lc_steps(d.S) * d.n;
      CK(cudaMemcpyAsync(dlc[par] + off, hlc + off, (size_t)(mc / 64) * lc_steps(d.S) * d.n * 4, cudaMemcpyHostToDevice, sc));
      if (epi == 2) {
        const size_t xo = (size_t)(r0 / 64) * 2 * d.n;
        CK(cudaMemcpyAsync(d.xc + xo, d.hxc + xo, (size_t)(mc / 64) * 2 * d.n * 4, cudaMemcpyHostToDevice, sc));
      }
      CK(cudaEventRecord(ev_c, sc));
      CK(cudaStreamWaitEvent(st, ev_c));
    }
    return p;
  };
  auto fwd_serial = [&] {
    int call = 0;
    for (int l = 0; l < 16; l++)
      for (const LinDev& d : lins) {
        link_a(d, s1);
        pre(d, l, s1);
        for (int r0 = 0; r0 < M; r0 += d.chunk) {
          const int mc = std::min(d.chunk, M - r0);
          const u32* cv = link_c(d, call & 1, r0, mc, true, s1);
          gemm(d.BN, true, d.A + (size_t)(r0 / 64) * d.S * TA, d.B, d.beta, mc, d.n, d.S, Wb[0], nullptr, nullptr, s1, cv, xu_of(d, r0));
          hsv[nbuf * hi].run(Wb[0], (long long)(mc / 64) * (d.n / d.BN) * d.S, 64 * d.BN * 4, s1, 0, words_tree);
        }
        call++;
      }
  };
  std::vector<cudaEvent_t> done(nbuf), full(nbuf);
  for (int b = 0; b < nbuf; b++) {
    done[b] = event_in(cx_b, CU_EVENT_DISABLE_TIMING);
    full[b] = event_in(cx_a, CU_EVENT_DISABLE_TIMING);
  }
  // BARE=1 also times beside with nothing ahead of each call's GEMM (no root(A_j), coins, G, forming or BLC's link; the GEMM
  // still reads lane coins), the schedule's floor, and beside's GEMM stream alone (the work ahead of each GEMM and the GEMMs,
  // no hash)
  const bool bare = getenv("BARE") && atoi(getenv("BARE"));
  auto fwd_beside_of = [&](bool with_pre, bool with_hash = true) {
    int c = 0, call = 0;
    for (int b = 0; b < nbuf; b++) CK(cudaEventRecord(done[b], s2[b]));
    for (int l = 0; l < 16; l++)
      for (const LinDev& d : lins) {
        if (with_pre) {
          link_a(d, sb);
          pre(d, l, sb);
        }
        for (int r0 = 0; r0 < M; r0 += d.chunk, c++) {
          const int mc = std::min(d.chunk, M - r0), b = c % nbuf;
          const u32* cv = link_c(d, call & 1, r0, mc, with_pre, sb);
          CK(cudaStreamWaitEvent(sb, done[b]));
          gemm(d.BN, true, d.A + (size_t)(r0 / 64) * d.S * TA, d.B, d.beta, mc, d.n, d.S, Wb[b], nullptr, nullptr, sb, cv, xu_of(d, r0));
          if (!with_hash) continue;
          CK(cudaEventRecord(full[b], sb));
          CK(cudaStreamWaitEvent(s2[b], full[b]));
          hsv[nbuf * hi + b].run(Wb[b], (long long)(mc / 64) * (d.n / d.BN) * d.S, 64 * d.BN * 4, s2[b], hash_cap, words_tree);
          CK(cudaEventRecord(done[b], s2[b]));
        }
        call++;
      }
    for (int b = 0; b < nbuf; b++) CK(cudaStreamWaitEvent(sb, done[b]));
  };
  Timer& tb = split ? TB : T;
  for (hi = 0; hi < (int)rbs.size(); hi++) {
    if (!split) line("fwd_serial", nullptr, T.ms(reps, fwd_serial, s1), fwd_macs, fwd_words, 0);
    line("fwd_beside", nullptr, tb.ms(reps, [&] { fwd_beside_of(true); }, sb), fwd_macs, fwd_words, 0);
    if (bare) line("fwd_beside_bare", nullptr, tb.ms(reps, [&] { fwd_beside_of(false); }, sb), fwd_macs, fwd_words, 0);
    if (bare && hi == 0)
      line("fwd_beside_crit", nullptr, tb.ms(reps, [&] { fwd_beside_of(true, false); }, sb), fwd_macs, fwd_words, 0);
  }
  printf("{\"what\": \"clock\", \"mhz_start\": %.1f, \"mhz_end\": %.1f, \"beside_buffers\": %d, \"hash_cap\": %llu, "
         "\"root_a_in_forward\": %d, \"root_a_rows\": %d, \"root_a_tree\": 1, \"words_tree\": %d, \"rowseg_leaf\": "
         "\"sz_finish\", \"prio\": %d, \"layout\": \"%s\", \"blc_w\": [%d, %d, %d], \"l1\": %d, \"l1_cta\": %d, "
         "\"split_sms\": [%u, %u], \"epilogue\": \"%s\", \"small_threads\": %d, \"small_rows_max\": %llu}\n",
         f0, sm_mhz(dclk), nbuf, (unsigned long long)hash_cap, root_a, root_a_rows, (int)words_tree, (int)prio,
         layout_name(), g_lw[0], g_lw[1], g_lw[2], l1_mode, g_l1_cta, split_sms[0], split_sms[1],
         epi == 2 ? (g_xu ? "xu" : "fold") : epi == 1 ? "stock" : "none",
#ifdef SSN_SMALL_ROWS
         SSN_SMALL_ROWS, (unsigned long long)small_rows_max()
#else
         0, 0ull
#endif
  );
  return 0;
}

int main(int argc, char** argv) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_sms = p.multiProcessorCount;
  if (p.major != 9) {
    fprintf(stderr, "ERR device sm_%d%d: this binary runs on sm_90a\n", p.major, p.minor);
    return 2;
  }
  if (const char* w = getenv("BLC_W"))
    if (sscanf(w, "%d%*[,:]%d%*[,:]%d", &g_lw[0], &g_lw[1], &g_lw[2]) != 3 || std::abs(g_lw[0]) > 127 ||
        std::abs(g_lw[1]) > 127 || std::abs(g_lw[2]) > 127) {
      fprintf(stderr, "ERR BLC_W=%s: three int8 weights, e.g. 1:127:1 or 1,127,1\n", w);
      return 2;
    }
  if (argc == 2 && !strcmp(argv[1], "check")) return check();
  if (argc >= 4 && !strcmp(argv[1], "time")) {
    g_blc = getenv("BLC") ? std::min(2, std::max(0, atoi(getenv("BLC")))) : 0;
    g_tag = g_blc || (getenv("TAG") && atoi(getenv("TAG")));
    g_xu = getenv("XU") && atoi(getenv("XU"));
    g_rot = getenv("ROT") && atoi(getenv("ROT"));
    g_fold = getenv("FOLD") && atoi(getenv("FOLD"));
    if (g_xu && (!g_tag || (getenv("L1") && atoi(getenv("L1")) == 2))) {
      fprintf(stderr, "ERR XU=1 takes a tag layout (TAG=1 or BLC) and L1 0 or 1\n");
      return 2;
    }
    if (g_fold && (!g_blc || g_xu)) {
      fprintf(stderr, "ERR FOLD=1 folds out the lane coins, so it takes BLC=1 or 2, and not XU=1 (which folds them too)\n");
      return 2;
    }
    if (g_rot && !g_blc) {
      fprintf(stderr, "ERR ROT=1 sets the lane-coin lanes' A', so it takes BLC=1 or 2\n");
      return 2;
    }
    std::vector<int> rbs;
    for (int a = 5; a < argc; a++) rbs.push_back(atoi(argv[a]));
    if (rbs.empty()) rbs = {0, 65536, 16384};
    return time_all(atoi(argv[2]), atoi(argv[3]), argc >= 5 ? atof(argv[4]) : 16.0, rbs);
  }
  fprintf(stderr, "usage: ssn check | time REPS M [BUDGET_GB [ROW_BYTES ...]]   (ROW_BYTES 0: row-seg; -B: L1's plain leaves)\n");
  return 2;
}
