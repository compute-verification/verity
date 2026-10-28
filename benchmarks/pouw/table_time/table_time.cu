// QSTAR: the cheapest CUDA-core programs that produce correct Dither prefix words.
// A word is T_ℓ[i, j] = Σ_{t < 32(ℓ+1)} Ã[i, t] B̃[t, j], exact s32, held in a register and folded by the duo receipt
// (receipt_folds.cuh rf_duo_word, one chain per (i, j) over ℓ, keys from h_falsify/ref.py), so every arm's output is one
// receipt word per (i, j) and the gate checks it bit for bit against the CPU reference. Arms:
//
//   imma_gen   generic int8 Ã, B̃: one m16n8k32 IMMA a segment (the honest word, for the time ratio)
//   imma_zero  the zero instance (Ã = R ∈ {0,1}, B̃[t, j] = U[j, t mod 32]), U resident in the B fragments
//   dp4a_gen   IDP.4A, 8 a word, a 4×8 register tile from double-buffered shared memory (generic data)
//   dp4a_zero  the same kernel on the zero instance's R and expanded U
//   bits       bit planes of U: Σ_b ±2^b popc(R_ℓ & P_b[j]), 8 LOP3 + 8 POPC a word (zero instance)
//   fr8        Four Russians, four 8-bit groups of R_ℓ, tables of U-row sums in shared memory (zero instance)
//   fr16       Four Russians, two 16-bit groups, tables in L2 read with 256-bit loads: one IADD3 a word (zero)
//   co_*       a fused kernel: IMMA warps on the zero instance beside scalar warps of one arm, sharing every SM
//   imma_p*    persistent IMMA on the zero instance, 8-warp blocks two an SM (imma_p) or 4-warp blocks three (imma_p3)
//   co2_*, co3_*  that IMMA on n - co2 columns, and on a second stream fr16 on the other co2 (none: the control)
//   design A (Ã = R, B̃ = Rc, both 0/1 dither bits): imma_bb its honest IMMA (imma_gen's kernel); bb its words, LOP3 AND +
//   POPC + IADD3 a word; bx, bx_hs, bx_mix the depth-k bit product alone (out = O, no word a segment; CO-DITHER's x):
//   AND + POPC, Harley–Seal carry-save, half each; co3_bb, co3_bx, co3_mix each on the second stream beside co3's IMMA
//
// Each timed loop is one segment an iteration (two for IMMA) between two SR_CLOCKLO reads, so count.py pins its
// instructions (expect.json) and turns them into instructions a word. gpu_kit carries poisoning and the timed items.
#include "../gpu_kit/bench.cuh"
#include "../receipts/receipt_folds.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK GK_CK
static const uint32_t SEED = 0x48464C53u;  // h_falsify/ref.py SEED
constexpr int SEG = 32, MAXL = 448;

// ---- the duo fold's public keys (h_falsify/ref.py keys) and a chain's start ---------------------------------------------
static uint32_t key_a(int ell) {
  for (int i = 0;; i++) {
    uint32_t c = rf_mix32(0xA0000000u ^ ((uint32_t)ell << 8) ^ (uint32_t)i) | 1u;
    if (rf_key_ok(c)) return c;
  }
}
static uint32_t key_k(int ell) { return rf_mix32(0xB0000000u + (uint32_t)ell); }
static int key_r(int ell) { return (ell * 7 + 5) % 31 + 1; }
__host__ __device__ inline uint32_t chain_y0(uint32_t idx) { return rf_mix32(SEED ^ rf_mix32(idx)); }
__host__ __device__ inline uint32_t chain_z0(uint32_t y0) { return rf_mix32(y0 + 0x9E3779B9u); }

__constant__ uint4 c_key[MAXL];  // (a, k, r, 0) of segment ℓ: one 128-bit constant load a segment

// A block's SM cycles between its two clock reads, and the SM it ran on.
struct Clk {
  long long cyc;
  int smid, pad;
};
// A body runs TIMED in its own kernel (clock reads around the loop, behind block barriers) or untimed beside other warps.
#define QS_BEGIN           \
  long long qs_t0_ = 0;    \
  if constexpr (TIMED) {   \
    __syncthreads();       \
    qs_t0_ = clock64();    \
  }
#define QS_END(rec)                                                           \
  if constexpr (TIMED) {                                                      \
    __syncthreads();                                                          \
    if (threadIdx.x == 0) {                                                   \
      const long long t1_ = clock64();                                        \
      int s_;                                                                 \
      asm volatile("mov.u32 %0, %%smid;" : "=r"(s_));                         \
      (rec)[blockIdx.y * gridDim.x + blockIdx.x] = Clk{t1_ - qs_t0_, s_, 0};  \
    }                                                                         \
  }

#define KEYS(ell)                    \
  const uint4 key_ = c_key[ell];     \
  const uint32_t ka_ = key_.x, kk_ = key_.y; \
  const int kr_ = (int)key_.z;
#define FOLD(y, z, d) rf_duo_word(y, z, (uint32_t)(d), kk_, ka_, kr_)

// ---- IMMA (m16n8k32 s8): the accumulator is the prefix word ------------------------------------------------------------
__device__ __forceinline__ void imma(int* c, uint4 a, uint2 b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}

// One warp: 16 rows (block bi) by WN 8-column tiles from nt0, every segment. Ap [MB][L][32 lanes] uint4; Bp is the zero
// instance's resident U fragments [NB][32] or the generic, segment-major [L][NB][32]. Each segment's fragments are
// loaded two segments ahead, into the registers its MMAs have just read (the loop is unrolled by two, L even); Ap and
// the generic Bp end in two spare segments, so the prefetch needs no clamp.
constexpr int WN = 4;
template <bool ZERO>
__device__ __forceinline__ void imma_warp(const uint4* __restrict__ Ap, const uint2* __restrict__ Bp,
                                          uint32_t* __restrict__ out, int L, int n, int bi, int nt0, Clk* rec,
                                          bool timed) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  int acc[WN][4];
  uint32_t y[WN][4], z[WN][4];
  uint2 bf[WN];
#pragma unroll
  for (int s = 0; s < WN; s++) {
    if (ZERO) bf[s] = Bp[(nt0 + s) * 32 + lane];
#pragma unroll
    for (int c = 0; c < 4; c++) {
      acc[s][c] = 0;
      const int ii = 16 * bi + g + (c >> 1) * 8, jj = 8 * (nt0 + s) + 2 * t + (c & 1);
      y[s][c] = chain_y0((uint32_t)ii * n + jj);
      z[s][c] = chain_z0(y[s][c]);
    }
  }
  const uint4* ap = Ap + (size_t)bi * L * 32 + lane;
  const size_t bs = (size_t)(n / 8) * 32;  // the generic Bp's segment stride
  const uint2* bp = Bp + (size_t)nt0 * 32 + lane;
  uint4 a0 = ap[0], a1 = ap[32];
  ap += 64;
  uint2 b0[WN], b1[WN];
  if (!ZERO)
#pragma unroll
    for (int s = 0; s < WN; s++) b0[s] = bp[32 * s], b1[s] = bp[bs + 32 * s];
  bp += 2 * bs;
  auto step = [&](int ell, uint4& a, uint2(&b)[WN]) {
    KEYS(ell)
#pragma unroll
    for (int s = 0; s < WN; s++) imma(acc[s], a, ZERO ? bf[s] : b[s]);
    a = *ap;
    ap += 32;
    if (!ZERO)
#pragma unroll
      for (int s = 0; s < WN; s++) b[s] = bp[32 * s];
    bp += bs;
#pragma unroll
    for (int s = 0; s < WN; s++)
#pragma unroll
      for (int w = 0; w < 4; w++) FOLD(y[s][w], z[s][w], acc[s][w]);
  };
  const long long t0 = clock64();
#pragma unroll 1
  for (int ell = 0; ell < L; ell += 2) {
    step(ell, a0, b0);
    step(ell + 1, a1, b1);
  }
  if (timed && lane == 0) {
    const long long t1 = clock64();
    int s_;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s_));
    rec[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + (threadIdx.x >> 5)] = Clk{t1 - t0, s_, 0};
  }
#pragma unroll
  for (int s = 0; s < WN; s++)
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const int ii = 16 * bi + g + (c >> 1) * 8, jj = 8 * (nt0 + s) + 2 * t + (c & 1);
      out[(size_t)ii * n + jj] = rf_duo_final(y[s][c], z[s][c]);
    }
}

// 8 warps a block share one 16-row block; grid (m/16, n/(8·WN·8)).
template <bool ZERO>
__device__ __forceinline__ void imma_body(const uint4* Ap, const uint2* Bp, uint32_t* out, int L, int n, Clk* rec) {
  imma_warp<ZERO>(Ap, Bp, out, L, n, blockIdx.x, (blockIdx.y * 8 + (threadIdx.x >> 5)) * WN, rec, true);
}
extern "C" __global__ void __launch_bounds__(256, 1) k_imma_gen(const uint4* Ap, const uint2* Bp, uint32_t* out, int L,
                                                             int n, Clk* rec) {
  imma_body<false>(Ap, Bp, out, L, n, rec);
}
extern "C" __global__ void __launch_bounds__(256, 1) k_imma_zero(const uint4* Ap, const uint2* Bp, uint32_t* out, int L,
                                                              int n, Clk* rec) {
  imma_body<true>(Ap, Bp, out, L, n, rec);
}
// The zero instance's IMMA on the columns [0, ni), persistent over the units (16 rows by 32·W columns for W warps a
// block, unit u at row block u % MB). The host gives it dynamic shared memory it doesn't use, so that at most B blocks
// sit on an SM and a concurrent scalar kernel (no shared memory) gets the registers and warps left over. The register
// file is split among the SM's four sub-partitions: 8-warp blocks two an SM leave 4096 registers a sub-partition (one
// warp of at most 128 registers), 4-warp blocks three an SM leave 7168 (one fr16_c32 warp, 168).
extern "C" __global__ void __maxnreg__(96) k_imma_p(const uint4* Ap, const uint2* Bp, uint32_t* out, int L,
                                                           int n, int MB, int ni) {
  const int W = blockDim.x >> 5, units = MB * (ni / (32 * W)), warp = threadIdx.x >> 5;
  for (int u = blockIdx.x; u < units; u += gridDim.x)
    imma_warp<true>(Ap, Bp, out, L, n, u % MB, ((u / MB) * W + warp) * WN, nullptr, false);
}

// ---- dp4a: a block tile of 64 rows by 128 columns, a thread 4 rows by 8 columns (columns tx + 16c), one segment's Ã
// and B̃ rows (32 bytes each) double-buffered in shared memory. B's row stride is 48 B so the 16 column lanes' LDS.128
// fall on distinct banks. A [m][k/16] and B [n][k/16] as uint4 (32 int8 a segment = 2 uint4). -------------------------
constexpr int DM = 64, DN = 128, DTM = 4, DTN = 8;
extern "C" __global__ void __launch_bounds__(256) k_dp4a(const uint4* __restrict__ A, const uint4* __restrict__ B,
                                                         uint32_t* __restrict__ out, int L, int n, Clk* rec) {
  constexpr bool TIMED = true;
  __shared__ __align__(16) uint32_t sA[2][DM][8];
  __shared__ __align__(16) uint32_t sB[2][DN][12];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int i0 = blockIdx.x * DM, j0 = blockIdx.y * DN, K16 = 2 * L;
  const int lr = tid >> 1, lh = tid & 1;  // the loader's row and half
  const bool loadA = tid < 2 * DM;
  int acc[DTM][DTN];
  uint32_t y[DTM][DTN], z[DTM][DTN];
#pragma unroll
  for (int r = 0; r < DTM; r++)
#pragma unroll
    for (int c = 0; c < DTN; c++) {
      acc[r][c] = 0;
      y[r][c] = chain_y0((uint32_t)(i0 + ty * DTM + r) * n + (j0 + tx + 16 * c));
      z[r][c] = chain_z0(y[r][c]);
    }
  uint4 ra = loadA ? A[(size_t)(i0 + lr) * K16 + lh] : make_uint4(0, 0, 0, 0);
  uint4 rb = B[(size_t)(j0 + lr) * K16 + lh];
  if (loadA) *(uint4*)&sA[0][lr][4 * lh] = ra;
  *(uint4*)&sB[0][lr][4 * lh] = rb;
  QS_BEGIN
#pragma unroll 1
  for (int ell = 0; ell < L; ell++) {
    const int buf = ell & 1, nx = min(ell + 1, L - 1);
    if (loadA) ra = A[(size_t)(i0 + lr) * K16 + 2 * nx + lh];
    rb = B[(size_t)(j0 + lr) * K16 + 2 * nx + lh];
    KEYS(ell)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint4 a[DTM], b[DTN];
#pragma unroll
      for (int r = 0; r < DTM; r++) a[r] = *(const uint4*)&sA[buf][ty * DTM + r][4 * h];
#pragma unroll
      for (int c = 0; c < DTN; c++) b[c] = *(const uint4*)&sB[buf][tx + 16 * c][4 * h];
#pragma unroll
      for (int r = 0; r < DTM; r++)
#pragma unroll
        for (int c = 0; c < DTN; c++) {
          acc[r][c] = __dp4a((int)a[r].x, (int)b[c].x, acc[r][c]);
          acc[r][c] = __dp4a((int)a[r].y, (int)b[c].y, acc[r][c]);
          acc[r][c] = __dp4a((int)a[r].z, (int)b[c].z, acc[r][c]);
          acc[r][c] = __dp4a((int)a[r].w, (int)b[c].w, acc[r][c]);
        }
    }
#pragma unroll
    for (int r = 0; r < DTM; r++)
#pragma unroll
      for (int c = 0; c < DTN; c++) FOLD(y[r][c], z[r][c], acc[r][c]);
    if (loadA) *(uint4*)&sA[buf ^ 1][lr][4 * lh] = ra;
    *(uint4*)&sB[buf ^ 1][lr][4 * lh] = rb;
    __syncthreads();
  }
  QS_END(rec)
#pragma unroll
  for (int r = 0; r < DTM; r++)
#pragma unroll
    for (int c = 0; c < DTN; c++)
      out[(size_t)(i0 + ty * DTM + r) * n + j0 + tx + 16 * c] = rf_duo_final(y[r][c], z[r][c]);
}

// ---- bit planes: U[j, p] as 8-bit two's complement, P[j][b] bit p = bit b of U[j, p]; a segment's sum is
// Σ_{b<7} 2^b popc(R & P_b) − 2^7 popc(R & P_7). A thread is one row and BT columns; Rw [L][m] is R_ℓ's 32 bits. ------
constexpr int BT = 8;
template <bool TIMED>
__device__ __forceinline__ void bits_thread(const uint32_t* __restrict__ Rw, const uint32_t (&pl)[BT][8], int L, int m,
                                            int n, int i, int j0, uint32_t* __restrict__ out, Clk* rec) {
  int acc[BT];
  uint32_t y[BT], z[BT];
#pragma unroll
  for (int c = 0; c < BT; c++) {
    acc[c] = 0;
    y[c] = chain_y0((uint32_t)i * n + j0 + c);
    z[c] = chain_z0(y[c]);
  }
  QS_BEGIN
#pragma unroll 1
  for (int ell = 0; ell < L; ell++) {
    const uint32_t R = Rw[(size_t)ell * m + i];
    KEYS(ell)
#pragma unroll
    for (int c = 0; c < BT; c++) {
      int s = -(__popc(R & pl[c][7]) << 7);
#pragma unroll
      for (int b = 0; b < 7; b++) s += __popc(R & pl[c][b]) << b;
      acc[c] += s;
      FOLD(y[c], z[c], acc[c]);
    }
  }
  QS_END(rec)
#pragma unroll
  for (int c = 0; c < BT; c++) out[(size_t)i * n + j0 + c] = rf_duo_final(y[c], z[c]);
}
extern "C" __global__ void __launch_bounds__(256) k_bits(const uint32_t* __restrict__ Rw, const uint32_t* __restrict__ P,
                                                         uint32_t* __restrict__ out, int L, int m, int n, Clk* rec) {
  const int i = blockIdx.x * 256 + threadIdx.x, j0 = blockIdx.y * BT;
  uint32_t pl[BT][8];
#pragma unroll
  for (int c = 0; c < BT; c++)
#pragma unroll
    for (int b = 0; b < 8; b++) pl[c][b] = P[(j0 + c) * 8 + b];
  bits_thread<true>(Rw, pl, L, m, n, i, j0, out, rec);
}

// ---- Four Russians, 8-bit groups in shared memory: tab[g][q][p] holds columns 4q..4q+3 of Σ_{b<8} p_b U[j, 8g + b]
// (an int4, 16 B), 64 KB for a block's 16 columns. A thread is one row and the 16 columns; a word is acc + four entries
// (two IADD3), an entry four columns (LDS.128 at an immediate offset from the group's index). -------------------------
constexpr int F8C = 16, F8T = 512;
constexpr size_t F8_SMEM = 4 * 4 * 256 * 16;
__device__ __forceinline__ void fr8_build(int4* tab, const int8_t* __restrict__ U, int j0, int t0, int nt) {
  for (int e = t0; e < 4 * 4 * 256; e += nt) {
    const int p = e & 255, q = (e >> 8) & 3, g = e >> 10;
    int v[4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
      int s = 0;
      for (int b = 0; b < 8; b++)
        if (p & (1 << b)) s += U[(j0 + 4 * q + c) * SEG + 8 * g + b];
      v[c] = s;
    }
    tab[e] = make_int4(v[0], v[1], v[2], v[3]);
  }
}
template <bool TIMED>
__device__ __forceinline__ void fr8_thread(const int4* tab, const uint32_t* __restrict__ Rw, int L, int m, int n, int i,
                                           int j0, uint32_t* __restrict__ out, Clk* rec) {
  const char* tb = (const char*)tab;
  int acc[F8C];
  uint32_t y[F8C], z[F8C];
#pragma unroll
  for (int c = 0; c < F8C; c++) {
    acc[c] = 0;
    y[c] = chain_y0((uint32_t)i * n + j0 + c);
    z[c] = chain_z0(y[c]);
  }
  QS_BEGIN
#pragma unroll 1
  for (int ell = 0; ell < L; ell++) {
    const uint32_t R = Rw[(size_t)ell * m + i];
    KEYS(ell)
    uint32_t off[4];
#pragma unroll
    for (int g = 0; g < 4; g++) off[g] = ((R >> (8 * g)) & 255u) << 4;
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int4 e[4];
#pragma unroll
      for (int g = 0; g < 4; g++) e[g] = *(const int4*)(tb + off[g] + (g * 4 + q) * 4096);
      acc[4 * q + 0] += e[0].x + e[1].x + e[2].x + e[3].x;
      acc[4 * q + 1] += e[0].y + e[1].y + e[2].y + e[3].y;
      acc[4 * q + 2] += e[0].z + e[1].z + e[2].z + e[3].z;
      acc[4 * q + 3] += e[0].w + e[1].w + e[2].w + e[3].w;
    }
#pragma unroll
    for (int c = 0; c < F8C; c++) FOLD(y[c], z[c], acc[c]);
  }
  QS_END(rec)
#pragma unroll
  for (int c = 0; c < F8C; c++) out[(size_t)i * n + j0 + c] = rf_duo_final(y[c], z[c]);
}
// grid (m/512, n/16), 512 threads, 64 KB dynamic shared memory
extern "C" __global__ void __launch_bounds__(F8T, 1) k_fr8(const uint32_t* __restrict__ Rw, const int8_t* __restrict__ U,
                                                           uint32_t* __restrict__ out, int L, int m, int n, Clk* rec) {
  extern __shared__ int4 tab[];
  const int j0 = blockIdx.y * F8C;
  fr8_build(tab, U, j0, threadIdx.x, F8T);
  fr8_thread<true>(tab, Rw, L, m, n, blockIdx.x * F8T + threadIdx.x, j0, out, rec);
}

// ---- Four Russians, 16-bit groups in L2: T[pass][g][p][CPT] int32 = Σ_{b<16} p_b U[j0 + c, 16g + b], 512·CPT KB a
// pass. A thread is one row and CPT columns; a word is acc + e0 + e1 (one IADD3), eight entries one 256-bit load. The
// next segment's loads are issued before this segment's fold, and R two segments ahead. --------------------------------
// Eight entries of each group a 256-bit load (LDG.E.ENL2.256), as the 32-byte vector type so ptxas folds the offsets.
__device__ __forceinline__ void unpack8(const ulonglong4_32a& a, int* v) {
  v[0] = (int)(uint32_t)a.x, v[1] = (int)(a.x >> 32), v[2] = (int)(uint32_t)a.y, v[3] = (int)(a.y >> 32);
  v[4] = (int)(uint32_t)a.z, v[5] = (int)(a.z >> 32), v[6] = (int)(uint32_t)a.w, v[7] = (int)(a.w >> 32);
}
template <int Q>
__device__ __forceinline__ void ld8s(const char* p0, const char* p1, int* e0, int* e1) {
#pragma unroll
  for (int q = 0; q < Q; q++) {
    unpack8(((const ulonglong4_32a*)p0)[q], e0 + 8 * q);
    unpack8(((const ulonglong4_32a*)p1)[q], e1 + 8 * q);
  }
}
template <int CPT, bool TIMED = true>
__device__ __forceinline__ void fr16_thread(const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1,
                                            const uint32_t* __restrict__ Rw, int L, int m, int n, int i, int j0,
                                            uint32_t* __restrict__ out, Clk* rec) {
  int acc[CPT], e0[CPT], e1[CPT];
  uint32_t y[CPT], z[CPT];
#pragma unroll
  for (int c = 0; c < CPT; c++) {
    acc[c] = 0;
    y[c] = chain_y0((uint32_t)i * n + j0 + c);
    z[c] = chain_z0(y[c]);
  }
  const char *b0 = (const char*)T0, *b1 = (const char*)T1;
  const uint32_t* rp = Rw + i;  // Rw has two zero segments after the last, so the prefetch needs no clamp
  uint32_t Rn = rp[m];
  ld8s<CPT / 8>(b0 + (rp[0] & 0xFFFFu) * (4 * CPT), b1 + (rp[0] >> 16) * (4 * CPT), e0, e1);
  rp += 2 * m;
  QS_BEGIN
#pragma unroll 1
  for (int ell = 0; ell < L; ell++) {
#pragma unroll
    for (int c = 0; c < CPT; c++) acc[c] += e0[c] + e1[c];
    ld8s<CPT / 8>(b0 + (Rn & 0xFFFFu) * (4 * CPT), b1 + (Rn >> 16) * (4 * CPT), e0, e1);
    Rn = *rp;
    rp += m;
    KEYS(ell)
#pragma unroll
    for (int c = 0; c < CPT; c++) FOLD(y[c], z[c], acc[c]);
  }
  QS_END(rec)
#pragma unroll
  for (int c = 0; c < CPT; c++) out[(size_t)i * n + j0 + c] = rf_duo_final(y[c], z[c]);
}
// grid (m/F16T, P): block y is a pass, columns jbase + CPT·y; T holds P passes of [2][65536][CPT], P·512·CPT KB that
// must stay in L2, so a launch has only P·m threads, in small blocks to spread them over the SMs.
constexpr int F16T = 64;
#define FR16_KERNEL(CPT)                                                                                         \
  extern "C" __global__ void __launch_bounds__(F16T) k_fr16_c##CPT(const uint32_t* __restrict__ T,               \
                                                                   const uint32_t* __restrict__ Rw,              \
                                                                   uint32_t* __restrict__ out, int L, int m,    \
                                                                   int n, int jbase, Clk* rec) {                \
    const uint32_t* T0 = T + (size_t)blockIdx.y * 2 * 65536 * CPT;                                              \
    fr16_thread<CPT>(T0, T0 + (size_t)65536 * CPT, Rw, L, m, n, blockIdx.x * F16T + threadIdx.x,               \
                     jbase + CPT * blockIdx.y, out, rec);                                                       \
  }
FR16_KERNEL(16)
FR16_KERNEL(32)

// The tables: halves H[pass][g][h][b][c] = Σ_{bit<8} b_bit U[j, 16g + 8h + bit], then T[..][p][c] = H0[p&255] + H1[p>>8].
__global__ void k_fr16_half(const int8_t* __restrict__ U, int* __restrict__ H, int jbase, int P, int cpt) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;  // ((pass*2 + g)*2 + h)*256 + b, times cpt, plus c
  if (e >= P * 2 * 2 * 256 * cpt) return;
  const int c = e % cpt, b = (e / cpt) & 255, h = (e / cpt / 256) & 1, g = (e / cpt / 512) & 1, pass = e / cpt / 1024;
  const int j = jbase + pass * cpt + c;
  int s = 0;
  for (int bit = 0; bit < 8; bit++)
    if (b & (1 << bit)) s += U[j * SEG + 16 * g + 8 * h + bit];
  H[e] = s;
}
__global__ void k_fr16_expand(const int* __restrict__ H, int* __restrict__ T, int P, int cpt) {
  const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // (pg*65536 + p)*(cpt/8) + c8
  const int c8n = cpt / 8;
  if (e >= (size_t)P * 2 * 65536 * c8n) return;
  const int c8 = (int)(e % c8n), p = (int)((e / c8n) & 65535), pg = (int)(e / c8n / 65536);
  const int4* lo = (const int4*)(H + ((size_t)(pg * 2 + 0) * 256 + (p & 255)) * cpt + 8 * c8);
  const int4* hi = (const int4*)(H + ((size_t)(pg * 2 + 1) * 256 + (p >> 8)) * cpt + 8 * c8);
  int4* dst = (int4*)(T + ((size_t)pg * 65536 + p) * cpt + 8 * c8);
#pragma unroll
  for (int v = 0; v < 2; v++) {
    const int4 a = lo[v], b = hi[v];
    dst[v] = make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
}


// ---- beside saturated IMMA: a 512-thread block of 8 IMMA warps (one k_imma_zero unit, 16 rows by 256 columns) and 8
// scalar warps (one unit: fr8 256 rows by 16 columns, bits 256 rows by 8), both on the zero instance. IMMA takes the
// columns [0, ni) and the scalar arm [ni, n); block b runs IMMA unit b if b < Ui and scalar unit b if b < Us. ---------
constexpr int CO_T = 512, CO_ROWS = 256;
template <int ARM>  // 0: no scalar unit (the control), 1: fr8, 2: bits
__global__ void __launch_bounds__(CO_T, 1) k_co(const uint4* __restrict__ Ap, const uint2* __restrict__ Uf,
                                                const uint32_t* __restrict__ Rw, const int8_t* __restrict__ U,
                                                const uint32_t* __restrict__ P, uint32_t* __restrict__ out, int L, int m,
                                                int n, int ni, int Ui, int Us) {
  extern __shared__ int4 co_tab[];
  const int warp = threadIdx.x >> 5, b = blockIdx.x;
  if (warp < 8) {
    if (b < Ui) {
      const int mb = m / 16;
      imma_warp<true>(Ap, Uf, out, L, n, b % mb, ((b / mb) * 8 + warp) * WN, nullptr, false);
    }
  } else if (ARM != 0 && b < Us) {
    const int st = threadIdx.x - 256, tr = m / CO_ROWS, i = (b % tr) * CO_ROWS + st;
    if constexpr (ARM == 1) {
      const int j0 = ni + (b / tr) * F8C;
      fr8_build(co_tab, U, j0, st, 256);
      asm volatile("bar.sync 1, 256;" ::: "memory");
      fr8_thread<false>(co_tab, Rw, L, m, n, i, j0, out, nullptr);
    } else {
      const int j0 = ni + (b / tr) * BT;
      uint32_t pl[BT][8];
#pragma unroll
      for (int c = 0; c < BT; c++)
#pragma unroll
        for (int q = 0; q < 8; q++) pl[c][q] = P[(j0 + c) * 8 + q];
      bits_thread<false>(Rw, pl, L, m, n, i, j0, out, nullptr);
    }
  }
}

// ---- design A: both operands dither bits, Ã = R and B̃ = Rc ∈ {0,1}, so a word's
// segment is popc(R_ℓ[i] & C_ℓ[j]), with R_ℓ[i] and C_ℓ[j] the 32 bits as in Rw. A thread is BR = 4 consecutive rows by
// BCPT = 4 consecutive columns; a block of 64 is BG × BG = 8 × 8 of them, 32 rows by 32 columns, so each 128-bit row
// and column load is shared by 8 threads through L1 and L2 carries a quarter byte a lane-word (4 columns a block
// carried 1 byte, which bound every arm at L2's ≈ 3.3 TB/s). The operands are group-major, [rows/4][L + 4] uint4
// (pack_g4), so a thread's segments are consecutive: one 128-bit load a segment for its rows and one for its columns, at
// immediate offsets from two pointers that move once an iteration. Segments are loaded ahead into the registers their
// ANDs have just read, as in imma_warp.
constexpr int BR = 4, BBT = 64, BCPT = 4, BG = 8;
struct Seg {
  uint4 r, c;
};
__device__ __forceinline__ uint32_t w4(const uint4& v, int e) { return e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w; }
// A carry-save adder's two LOP3s, written as lop3 so ptxas keeps one instruction each
__device__ __forceinline__ uint32_t maj3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
__device__ __forceinline__ void seg_and(const Seg& s, uint32_t (&x)[BR][BCPT]) {
#pragma unroll
  for (int a = 0; a < BR; a++)
#pragma unroll
    for (int c = 0; c < BCPT; c++) x[a][c] = w4(s.r, a) & w4(s.c, c);
}

// Design A's words: T += popc(R & C) (LOP3, POPC, IADD3) and the duo fold, every segment; two segments an iteration.
template <bool TIMED>
__device__ __forceinline__ void bb_thread(const uint4* __restrict__ Rg, const uint4* __restrict__ Cg, int L, int n,
                                          int i0, int j0, uint32_t* __restrict__ out, Clk* rec) {
  int acc[BR][BCPT];
  uint32_t y[BR][BCPT], z[BR][BCPT];
#pragma unroll
  for (int a = 0; a < BR; a++)
#pragma unroll
    for (int c = 0; c < BCPT; c++) {
      acc[a][c] = 0;
      y[a][c] = chain_y0((uint32_t)(i0 + a) * n + j0 + c);
      z[a][c] = chain_z0(y[a][c]);
    }
  const uint4 *rp = Rg + (size_t)(i0 / 4) * (L + 4), *cp = Cg + (size_t)(j0 / 4) * (L + 4);
  Seg s0{rp[0], cp[0]}, s1{rp[1], cp[1]};
  auto step = [&](int ell, Seg& s, int h) {
    uint32_t x[BR][BCPT];
    seg_and(s, x);
    s = Seg{rp[2 + h], cp[2 + h]};
    KEYS(ell)
#pragma unroll
    for (int a = 0; a < BR; a++)
#pragma unroll
      for (int c = 0; c < BCPT; c++) {
        acc[a][c] += __popc(x[a][c]);
        FOLD(y[a][c], z[a][c], acc[a][c]);
      }
  };
  QS_BEGIN
#pragma unroll 1
  for (int ell = 0; ell < L; ell += 2) {
    step(ell, s0, 0);
    step(ell + 1, s1, 1);
    rp += 2, cp += 2;
  }
  QS_END(rec)
#pragma unroll
  for (int a = 0; a < BR; a++)
#pragma unroll
    for (int c = 0; c < BCPT; c++) out[(size_t)(i0 + a) * n + j0 + c] = rf_duo_final(y[a][c], z[a][c]);
}

// x: the depth-k bit product O[i, j] = Σ_ℓ popc(R_ℓ[i] & C_ℓ[j]) with no word a segment (out = O), eight segments an
// iteration. Rows a < NP count with POPC (AND, POPC a lane-word, IADD3 a pair: the XU pipe); the others with Harley–Seal
// carry-save over the eight (8 AND + 7 CSAs of two LOP3 + 1 POPC: the ALU pipe), so NP sets the two pipes' balance.
template <int NP, bool TIMED>
__device__ __forceinline__ void bx_thread(const uint4* __restrict__ Rg, const uint4* __restrict__ Cg, int L, int n,
                                          int i0, int j0, uint32_t* __restrict__ out, Clk* rec) {
  constexpr int CPT = BCPT;
  int acc[BR][CPT];
  uint32_t one[BR][CPT], two[BR][CPT], four[BR][CPT], t2[BR][CPT], t4[BR][CPT];
#pragma unroll
  for (int a = 0; a < BR; a++)
#pragma unroll
    for (int c = 0; c < CPT; c++) acc[a][c] = 0, one[a][c] = two[a][c] = four[a][c] = t2[a][c] = t4[a][c] = 0;
  const uint4 *rp = Rg + (size_t)(i0 / 4) * (L + 4), *cp = Cg + (size_t)(j0 / 4) * (L + 4);
  Seg sb[4];
#pragma unroll
  for (int q = 0; q < 4; q++) sb[q] = Seg{rp[q], cp[q]};
  auto pair = [&](int q) {  // segments 2q, 2q + 1 of the eight, in sb[2(q & 1)], sb[2(q & 1) + 1]
    uint32_t x0[BR][CPT], x1[BR][CPT];
    seg_and(sb[2 * (q & 1)], x0);
    seg_and(sb[2 * (q & 1) + 1], x1);
    sb[2 * (q & 1)] = Seg{rp[2 * q + 4], cp[2 * q + 4]};
    sb[2 * (q & 1) + 1] = Seg{rp[2 * q + 5], cp[2 * q + 5]};
#pragma unroll
    for (int a = 0; a < BR; a++)
#pragma unroll
      for (int c = 0; c < CPT; c++) {
        if (a < NP) {
          acc[a][c] += __popc(x0[a][c]) + __popc(x1[a][c]);
          continue;
        }
        uint32_t &o = one[a][c], &w = two[a][c], &f = four[a][c];
        const uint32_t h = maj3(o, x0[a][c], x1[a][c]);
        o = xor3(o, x0[a][c], x1[a][c]);
        if (!(q & 1)) {
          t2[a][c] = h;
          continue;
        }
        const uint32_t h4 = maj3(w, t2[a][c], h);
        w = xor3(w, t2[a][c], h);
        if (q == 1) {
          t4[a][c] = h4;
          continue;
        }
        const uint32_t e8 = maj3(f, t4[a][c], h4);
        f = xor3(f, t4[a][c], h4);
        acc[a][c] += __popc(e8);
      }
  };
  QS_BEGIN
#pragma unroll 1
  for (int g = 0; g < L; g += 8) {
    pair(0);
    pair(1);
    pair(2);
    pair(3);
    rp += 8, cp += 8;
  }
  QS_END(rec)
#pragma unroll
  for (int a = 0; a < BR; a++)
#pragma unroll
    for (int c = 0; c < CPT; c++)
      out[(size_t)(i0 + a) * n + j0 + c] =
          a < NP ? (uint32_t)acc[a][c]
                 : 8u * acc[a][c] + 4u * __popc(four[a][c]) + 2u * __popc(two[a][c]) + __popc(one[a][c]);
}
// grid (m/(BG·BR), columns/(BG·BCPT)): the columns [jbase, jbase + BG·BCPT·gridDim.y)
#define BIT_I0 ((blockIdx.x * BG + threadIdx.x / BG) * BR)
#define BIT_J0 (jbase + (blockIdx.y * BG + threadIdx.x % BG) * BCPT)
extern "C" __global__ void __launch_bounds__(BBT) k_bb(const uint4* __restrict__ Rg, const uint4* __restrict__ Cg,
                                                       uint32_t* __restrict__ out, int L, int n, int jbase, Clk* rec) {
  bb_thread<true>(Rg, Cg, L, n, BIT_I0, BIT_J0, out, rec);
}
#define BX_KERNEL(NAME, NP)                                                                                       \
  extern "C" __global__ void __launch_bounds__(BBT) NAME(const uint4* __restrict__ Rg, const uint4* __restrict__ Cg, \
                                                         uint32_t* __restrict__ out, int L, int n, int jbase,      \
                                                         Clk* rec) {                                               \
    bx_thread<NP, true>(Rg, Cg, L, n, BIT_I0, BIT_J0, out, rec);                                                   \
  }
BX_KERNEL(k_bx, BR)
BX_KERNEL(k_bx_hs, 0)
BX_KERNEL(k_bx_mix, BR / 2)

// ---- design A, SPEC §4.2a packed shared-memory byte route. Four Russians over the
// four 8-bit chunks of the row word r = R_ℓ[i]: a segment's bit-product count is ≤ 32, so it fits a byte and tab[k][v][c]
// = popc(v & chunk_k(C_ℓ[c])) is one byte (the count of column c's bits that v selects in chunk k). A word is
// acc + Σ_{k<4} tab[k][chunk_k(r)][c]; the four chunks' byte partials are summed packed (no byte carry, ≤ 32) and the
// writer is one IDP.4A with a unit byte selector that extracts column c's byte and adds it to its prefix. The table is in
// shared memory, rebuilt every 32-step (both operands fresh) and shared by the block's ATPB rows; its rebuild is inside
// the timed loop, so the measured time a word includes the per-instance build amortized over those rows. AC = 16 columns
// a block are read 16 a 128-bit LDS. -----------------------------------------------------------------------------------
constexpr int AC = 16, ATPB = 512;
constexpr int A_ENTRIES = 4 * 256 * AC;            // [chunk 4][value 256][column AC]
constexpr int A_BLD = A_ENTRIES / ATPB;            // build entries a thread, fully unrolled (no nested loop in the timed loop)
static_assert(A_ENTRIES % ATPB == 0, "the build must divide evenly so it fully unrolls");
constexpr size_t A_SMEM = (size_t)A_ENTRIES;       // bytes
__device__ __forceinline__ void a_build(signed char* tab, const uint32_t* __restrict__ csm, int t0, int nt) {
  // nt threads build the A_ENTRIES byte entries of this segment's table from the AC column words in shared memory
  // (csm[c] = C_ℓ[j0 + c]); fully unrolled so the timed loop holds no inner loop
#pragma unroll
  for (int it = 0; it < A_BLD; it++) {
    const int e = t0 + it * nt;
    const int c = e & (AC - 1), v = (e >> 4) & 255, k = e >> 12;
    const uint32_t cb = (csm[c] >> (8 * k)) & 0xFFu;
    tab[e] = (signed char)__popc(v & cb);
  }
}
template <bool TIMED>
__device__ __forceinline__ void a_fr8_thread(signed char* tab, const uint32_t* __restrict__ Rw,
                                             const uint32_t* __restrict__ Cw, int L, int m, int n, int i, int j0,
                                             uint32_t* __restrict__ out, Clk* rec, int tid, int nt) {
  __shared__ uint32_t csm[AC];  // this segment's AC column words, read once from global by the first AC threads
  int acc[AC];
  uint32_t y[AC], z[AC];
#pragma unroll
  for (int c = 0; c < AC; c++) {
    acc[c] = 0;
    y[c] = chain_y0((uint32_t)i * n + j0 + c);
    z[c] = chain_z0(y[c]);
  }
  QS_BEGIN
#pragma unroll 1
  for (int ell = 0; ell < L; ell++) {
    if (tid < AC) csm[tid] = Cw[(size_t)ell * n + j0 + tid];
    __syncthreads();
    a_build(tab, csm, tid, nt);
    __syncthreads();
    const uint32_t r = Rw[(size_t)ell * m + i];
    uint32_t ps[AC / 4];  // the four chunks' byte partials summed, 4 columns a word
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint32_t ch = (r >> (8 * k)) & 0xFFu;
      const uint4 e = *(const uint4*)(tab + (size_t)(k * 256 + ch) * AC);  // 16 columns' bytes for chunk k
      const uint32_t* w = (const uint32_t*)&e;
#pragma unroll
      for (int g = 0; g < AC / 4; g++) ps[g] = k ? ps[g] + w[g] : w[g];
    }
    KEYS(ell)
#pragma unroll
    for (int c = 0; c < AC; c++) {
      acc[c] = __dp4a((int)ps[c >> 2], 1 << (8 * (c & 3)), acc[c]);  // extract column c's count byte, add to its prefix
      FOLD(y[c], z[c], acc[c]);
    }
    __syncthreads();  // before the next segment's rebuild
  }
  QS_END(rec)
#pragma unroll
  for (int c = 0; c < AC; c++) out[(size_t)i * n + j0 + c] = rf_duo_final(y[c], z[c]);
}
// grid (m/ATPB, (j1-j0)/AC), A_SMEM dynamic shared memory; columns [jbase, jbase + AC·gridDim.y)
extern "C" __global__ void __launch_bounds__(ATPB, 1) k_a_fr8(const uint32_t* __restrict__ Rw,
                                                              const uint32_t* __restrict__ Cw, uint32_t* __restrict__ out,
                                                              int L, int m, int n, int jbase, Clk* rec) {
  extern __shared__ __align__(16) signed char a_tab[];
  a_fr8_thread<true>(a_tab, Rw, Cw, L, m, n, blockIdx.x * ATPB + threadIdx.x, jbase + blockIdx.y * AC, out, rec,
                     threadIdx.x, ATPB);
}

// ---- host: inputs, packings, the reference receipt -------------------------------------------------------------------
struct Inputs {
  int m, k, n, L;
  std::vector<int8_t> At, Bt;  // generic, m×k and n×k
  std::vector<int8_t> R, U;    // zero instance, m×k (0/1) and n×32 (±1..±64)
  std::vector<int8_t> Rc;      // design A's column bits, n×k (0/1); its Ã is R
};

static Inputs make_inputs(int m, int k, int n, unsigned seed) {
  Inputs in{m, k, n, k / SEG};
  std::mt19937 g(seed);
  in.At.resize((size_t)m * k);
  in.Bt.resize((size_t)n * k);
  in.R.resize((size_t)m * k);
  in.U.resize((size_t)n * SEG);
  std::uniform_int_distribution<int> s8(-128, 127), bit(0, 1), mag(1, 64), sign(0, 1);
  for (auto& v : in.At) v = (int8_t)s8(g);
  for (auto& v : in.Bt) v = (int8_t)s8(g);
  for (auto& v : in.R) v = (int8_t)bit(g);
  for (auto& v : in.U) v = (int8_t)(sign(g) ? mag(g) : -mag(g));
  in.Rc.resize((size_t)n * k);
  for (auto& v : in.Rc) v = (int8_t)bit(g);
  return in;
}

// m16n8k32 A fragments [rows/16][L][32 lanes][4] (h_falsify/ref.py pack_a) and B fragments segment-major, [L][n/8][32][2]
// (pack_b's fragments), each with two zero segments after the last for imma_warp's prefetch.
static std::vector<uint32_t> pack_a(const std::vector<int8_t>& X, int rows, int k) {
  const int L = k / SEG, MB = rows / 16;
  std::vector<uint32_t> out(((size_t)MB * L + 2) * 128, 0);
  for (int mb = 0; mb < MB; mb++)
    for (int ell = 0; ell < L; ell++)
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane / 4, t = lane % 4;
        const int rr[4] = {g, g + 8, g, g + 8}, cc[4] = {4 * t, 4 * t, 16 + 4 * t, 16 + 4 * t};
        for (int reg = 0; reg < 4; reg++) {
          uint32_t w = 0;
          for (int b = 0; b < 4; b++)
            w |= (uint32_t)(uint8_t)X[(size_t)(16 * mb + rr[reg]) * k + ell * SEG + cc[reg] + b] << (8 * b);
          out[(((size_t)mb * L + ell) * 32 + lane) * 4 + reg] = w;
        }
      }
  return out;
}
static std::vector<uint32_t> pack_b(const std::vector<int8_t>& Y, int n, int k) {
  const int L = k / SEG, NB = n / 8;
  std::vector<uint32_t> out(((size_t)L + 2) * NB * 64, 0);
  for (int nb = 0; nb < NB; nb++)
    for (int ell = 0; ell < L; ell++)
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane / 4, t = lane % 4;
        const int cc[2] = {4 * t, 16 + 4 * t};
        for (int reg = 0; reg < 2; reg++) {
          uint32_t w = 0;
          for (int b = 0; b < 4; b++)
            w |= (uint32_t)(uint8_t)Y[(size_t)(8 * nb + g) * k + ell * SEG + cc[reg] + b] << (8 * b);
          out[(((size_t)ell * NB + nb) * 32 + lane) * 2 + reg] = w;
        }
      }
  return out;
}
// R_ℓ's 32 bits a row, [L + 2][m]: bit p of Rw[ℓ][i] is R[i, 32ℓ + p]; the two last segments are zero (fr16's prefetch)
static std::vector<uint32_t> pack_rw(const std::vector<int8_t>& R, int m, int k) {
  const int L = k / SEG;
  std::vector<uint32_t> out((size_t)(L + 2) * m, 0);
  for (int ell = 0; ell < L; ell++)
    for (int i = 0; i < m; i++) {
      uint32_t w = 0;
      for (int p = 0; p < SEG; p++) w |= (uint32_t)(R[(size_t)i * k + ell * SEG + p] & 1) << p;
      out[(size_t)ell * m + i] = w;
    }
  return out;
}
// Group-major bits [rows/4][L + 4][4]: word e of group g, segment ℓ, is row 4g + e's 32 bits (bit p = X[4g + e, 32ℓ + p]);
// the four last segments are zero (the prefetches)
static std::vector<uint32_t> pack_g4(const std::vector<int8_t>& X, int rows, int k) {
  const int L = k / SEG;
  std::vector<uint32_t> out((size_t)rows * (L + 4), 0);
  for (int r = 0; r < rows; r++)
    for (int ell = 0; ell < L; ell++) {
      uint32_t w = 0;
      for (int p = 0; p < SEG; p++) w |= (uint32_t)(X[(size_t)r * k + ell * SEG + p] & 1) << p;
      out[((size_t)(r / 4) * (L + 4) + ell) * 4 + r % 4] = w;
    }
  return out;
}
// U's bit planes [n][8]: bit p of P[j][b] is bit b of U[j, p] as a byte
static std::vector<uint32_t> planes(const std::vector<int8_t>& U, int n) {
  std::vector<uint32_t> out((size_t)n * 8, 0);
  for (int j = 0; j < n; j++)
    for (int p = 0; p < SEG; p++)
      for (int b = 0; b < 8; b++) out[(size_t)j * 8 + b] |= (uint32_t)(((uint8_t)U[(size_t)j * SEG + p] >> b) & 1) << p;
  return out;
}
// B̃ of the zero instance as n×k, S[t, j] = U[j, t mod 32]
static std::vector<int8_t> expand_u(const std::vector<int8_t>& U, int n, int k) {
  std::vector<int8_t> out((size_t)n * k);
  for (int j = 0; j < n; j++)
    for (int t = 0; t < k; t++) out[(size_t)j * k + t] = U[(size_t)j * SEG + (t % SEG)];
  return out;
}
// The receipt word of each (i, j): the plain transcript and the stdlib duo (h_falsify/ref.py receipt_at)
static std::vector<uint32_t> ref_receipt(const int8_t* A, const int8_t* B, int k, int n,
                                         const std::vector<std::pair<int, int>>& pairs) {
  const int L = k / SEG;
  std::vector<uint32_t> out(pairs.size());
  for (size_t q = 0; q < pairs.size(); q++) {
    const int i = pairs[q].first, j = pairs[q].second;
    uint32_t y = chain_y0((uint32_t)i * n + j), z = chain_z0(y);
    int acc = 0;
    for (int ell = 0; ell < L; ell++) {
      for (int t = 0; t < SEG; t++) acc += (int)A[(size_t)i * k + ell * SEG + t] * (int)B[(size_t)j * k + ell * SEG + t];
      rf_duo_word(y, z, (uint32_t)acc, key_k(ell), key_a(ell), key_r(ell));
    }
    out[q] = rf_duo_final(y, z);
  }
  return out;
}
// The plain dot O[i, j] = Σ_t A[i, t] B[j, t] of each pair (the x arms' output)
static std::vector<uint32_t> ref_dot(const int8_t* A, const int8_t* B, int k,
                                     const std::vector<std::pair<int, int>>& pairs) {
  std::vector<uint32_t> out(pairs.size());
  for (size_t q = 0; q < pairs.size(); q++) {
    int acc = 0;
    for (int t = 0; t < k; t++) acc += (int)A[(size_t)pairs[q].first * k + t] * (int)B[(size_t)pairs[q].second * k + t];
    out[q] = (uint32_t)acc;
  }
  return out;
}
template <class T>
static T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  CK(cudaMalloc(&d, h.size() * sizeof(T)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

enum Arm {
  IMMA_GEN, IMMA_ZERO, DP4A_GEN, DP4A_ZERO, BITS, FR8, FR16_C16, FR16_C32, CO_NONE, CO_FR8, CO_BITS,
  IMMA_P, CO2_NONE, CO2_FR16, CO2_C16, IMMA_P3, CO3_NONE, CO3_C32,
  IMMA_BB, BB, BX, BX_HS, BX_MIX, CO3_BB, CO3_BX, CO3_MIX, A_FR8, CO3_AF, NARM
};
static const char* NAMES[NARM] = {"imma_gen", "imma_zero", "dp4a_gen", "dp4a_zero", "bits",     "fr8",
                                  "fr16_c16", "fr16_c32",  "co_none",  "co_fr8",    "co_bits",  "imma_p",
                                  "co2_none", "co2_fr16",  "co2_c16",  "imma_p3",   "co3_none", "co3_c32",
                                  "imma_bb",  "bb",        "bx",       "bx_hs",     "bx_mix",   "co3_bb",
                                  "co3_bx",   "co3_mix",   "a_fr8",    "co3_af"};

struct Dev {
  int m, k, n, L, share, co2, sms, p_smem2, p_smem3;
  cudaStream_t s2;
  cudaEvent_t fork, join;
  const uint4 *Ap_g, *Ap_z, *Aw_g, *Aw_z, *Bw_g, *Bw_z;
  const uint2 *Bp_g, *Uf, *Bp_b;
  const uint32_t *Rw, *P, *Cw;
  const uint4 *Rg, *Cg;
  const int8_t* U;
  int* H;
  uint32_t* T;
  uint32_t* out;
  Clk* rec;
};
constexpr size_t FR16_TABLE_BYTES = (size_t)64 << 20;  // a launch's passes, in L2

// The scalar arm's share of the columns beside IMMA: n/share, a multiple of 256 so IMMA's [0, ni) is whole units.
static int co_ni(const Dev& d) { return d.n - std::max(256, (d.n / d.share) / 256 * 256); }
// The two-stream arms: k_imma_p on [0, n - co2) (co2*: 8-warp blocks two an SM; co3*: 4-warp blocks three an SM) and
// fr16 on [n - co2, n) on a second stream (cpt 32 for co2_fr16 and co3_c32, 16 for co2_c16); imma_p* take all n.
// co3_bb, co3_bx and co3_mix put design A's bb, bx or bx_mix on the second stream instead.
struct Co {
  int warps, cpt, ni;
  Arm bit;  // the design-A arm on the second stream, or NARM
};
static Co co_of(Arm a, const Dev& d) {
  const Arm bit = a == CO3_BB ? BB : a == CO3_BX ? BX : a == CO3_MIX ? BX_MIX : a == CO3_AF ? A_FR8 : NARM;
  const int w = a == IMMA_P3 || a == CO3_NONE || a == CO3_C32 || bit != NARM ? 4 : 8;
  const int cpt = a == CO2_FR16 || a == CO3_C32 ? 32 : a == CO2_C16 ? 16 : 0;
  return {w, cpt, a == IMMA_P || a == IMMA_P3 ? d.n : d.n - d.co2, bit};
}
// The reference a column of an arm's output is checked against: the generic receipt, the zero instance's, design A's,
// or design A's plain dot (the x arms)
enum Ref { REF_G, REF_Z, REF_B, REF_O };
static Ref ref_of(Arm a, int j, const Dev& d) {
  switch (a) {
    case IMMA_GEN:
    case DP4A_GEN: return REF_G;
    case IMMA_BB:
    case A_FR8:
    case BB: return REF_B;
    case BX:
    case BX_HS:
    case BX_MIX: return REF_O;
    case CO3_BB:
    case CO3_AF:
    case CO3_BX:
    case CO3_MIX: return j < co_of(a, d).ni ? REF_Z : (a == CO3_BB || a == CO3_AF) ? REF_B : REF_O;
    default: return REF_Z;
  }
}
// The columns written, for the gate: a control leaves the scalar columns unwritten
static int written(Arm a, const Dev& d) {
  return a == CO_NONE ? co_ni(d) : a == CO2_NONE || a == CO3_NONE ? co_of(a, d).ni : d.n;
}

// Design A's arm a (bb, bx, bx_hs, bx_mix) on the columns [j0, j1)
static void bit_launch(const Dev& d, Arm a, int j0, int j1, cudaStream_t s) {
  if (a == A_FR8) {
    const dim3 grid(d.m / ATPB, (j1 - j0) / AC);
    k_a_fr8<<<grid, ATPB, A_SMEM, s>>>(d.Rw, d.Cw, d.out, d.L, d.m, d.n, j0, d.rec);
    return;
  }
  const dim3 grid(d.m / (BG * BR), (j1 - j0) / (BG * BCPT));
  auto* k = a == BB ? k_bb : a == BX ? k_bx : a == BX_HS ? k_bx_hs : k_bx_mix;
  k<<<grid, BBT, 0, s>>>(d.Rg, d.Cg, d.out, d.L, d.n, j0, d.rec);
}
// fr16 on the columns [j0, j1): per launch, the tables of P passes of cpt columns, then their lookups
static void fr16_launch(const Dev& d, int cpt, int j0, int j1, cudaStream_t s) {
  const int m = d.m, n = d.n, L = d.L, P = (int)(FR16_TABLE_BYTES / ((size_t)2 * 65536 * cpt * 4));
  for (int jb = j0; jb < j1; jb += cpt * P) {
    const int nh = P * 2 * 2 * 256 * cpt;
    k_fr16_half<<<(nh + 255) / 256, 256, 0, s>>>(d.U, d.H, jb, P, cpt);
    const size_t ne = (size_t)P * 2 * 65536 * (cpt / 8);
    k_fr16_expand<<<(unsigned)((ne + 255) / 256), 256, 0, s>>>(d.H, (int*)d.T, P, cpt);
    if (cpt == 16) k_fr16_c16<<<dim3(m / F16T, P), F16T, 0, s>>>(d.T, d.Rw, d.out, L, m, n, jb, d.rec);
    else k_fr16_c32<<<dim3(m / F16T, P), F16T, 0, s>>>(d.T, d.Rw, d.out, L, m, n, jb, d.rec);
  }
}

static void launch(Arm a, const Dev& d, cudaStream_t s) {
  const int m = d.m, n = d.n, L = d.L;
  switch (a) {
    case IMMA_GEN:
    case IMMA_BB:
    case IMMA_ZERO: {
      dim3 grid(m / 16, n / (8 * WN * 8));
      if (a == IMMA_GEN) k_imma_gen<<<grid, 256, 0, s>>>(d.Ap_g, d.Bp_g, d.out, L, n, d.rec);
      else if (a == IMMA_BB) k_imma_gen<<<grid, 256, 0, s>>>(d.Ap_z, d.Bp_b, d.out, L, n, d.rec);
      else k_imma_zero<<<grid, 256, 0, s>>>(d.Ap_z, d.Uf, d.out, L, n, d.rec);
      break;
    }
    case DP4A_GEN:
    case DP4A_ZERO:
      k_dp4a<<<dim3(m / DM, n / DN), 256, 0, s>>>(a == DP4A_GEN ? d.Aw_g : d.Aw_z, a == DP4A_GEN ? d.Bw_g : d.Bw_z,
                                                   d.out, L, n, d.rec);
      break;
    case BITS: k_bits<<<dim3(m / 256, n / BT), 256, 0, s>>>(d.Rw, d.P, d.out, L, m, n, d.rec); break;
    case FR8: k_fr8<<<dim3(m / F8T, n / F8C), F8T, F8_SMEM, s>>>(d.Rw, d.U, d.out, L, m, n, d.rec); break;
    case FR16_C16:
    case FR16_C32: fr16_launch(d, a == FR16_C16 ? 16 : 32, 0, n, s); break;
    case BB:
    case BX:
    case BX_HS:
    case BX_MIX:
    case A_FR8: bit_launch(d, a, 0, n, s); break;
    case IMMA_P:
    case CO2_NONE:
    case CO2_FR16:
    case CO2_C16:
    case IMMA_P3:
    case CO3_NONE:
    case CO3_C32:
    case CO3_BB:
    case CO3_BX:
    case CO3_MIX:
    case CO3_AF: {
      const Co c = co_of(a, d);
      if (c.cpt || c.bit != NARM) {
        CK(cudaEventRecord(d.fork, s));
        CK(cudaStreamWaitEvent(d.s2, d.fork, 0));
        if (c.cpt) fr16_launch(d, c.cpt, c.ni, n, d.s2);
        else bit_launch(d, c.bit, c.ni, n, d.s2);
      }
      if (c.warps == 8) k_imma_p<<<2 * d.sms, 256, d.p_smem2, s>>>(d.Ap_z, d.Uf, d.out, L, n, m / 16, c.ni);
      else k_imma_p<<<3 * d.sms, 128, d.p_smem3, s>>>(d.Ap_z, d.Uf, d.out, L, n, m / 16, c.ni);
      if (c.cpt || c.bit != NARM) {
        CK(cudaEventRecord(d.join, d.s2));
        CK(cudaStreamWaitEvent(s, d.join, 0));
      }
      break;
    }
    case CO_NONE:
    case CO_FR8:
    case CO_BITS: {
      const int ni = co_ni(d), Ui = (m / 16) * (ni / 256);
      const int Us = a == CO_NONE ? 0 : (m / CO_ROWS) * ((n - ni) / (a == CO_FR8 ? F8C : BT));
      const int grid = std::max(Ui, Us);
      if (a == CO_NONE) k_co<0><<<grid, CO_T, F8_SMEM, s>>>(d.Ap_z, d.Uf, d.Rw, d.U, d.P, d.out, L, m, n, ni, Ui, Us);
      else if (a == CO_FR8) k_co<1><<<grid, CO_T, F8_SMEM, s>>>(d.Ap_z, d.Uf, d.Rw, d.U, d.P, d.out, L, m, n, ni, Ui, Us);
      else k_co<2><<<grid, CO_T, F8_SMEM, s>>>(d.Ap_z, d.Uf, d.Rw, d.U, d.P, d.out, L, m, n, ni, Ui, Us);
      break;
    }
    default: break;
  }
  CK(cudaGetLastError());
}

static Arm arm_of(const std::string& t) {
  for (int a = 0; a < NARM; a++)
    if (t == NAMES[a]) return (Arm)a;
  fprintf(stderr, "ERR unknown arm %s\n", t.c_str());
  exit(2);
}

int main(int argc, char** argv) {
  int m = 4096, k = 14336, n = 4096, reps = 10, share = 8, co2 = 256;
  std::string mode = "time", dumpdir = "out", items;
  unsigned seed = 1;
  std::vector<Arm> arms;
  for (int i = 1; i < argc; i++) {
    std::string f = argv[i];
    auto nx = [&] { return std::atoi(argv[++i]); };
    if (f == "--m") m = nx();
    else if (f == "--k") k = nx();
    else if (f == "--n") n = nx();
    else if (f == "--reps") reps = nx();
    else if (f == "--seed") seed = nx();
    else if (f == "--share") share = nx();
    else if (f == "--co2") co2 = nx();
    else if (f == "--mode") mode = argv[++i];
    else if (f == "--dump") dumpdir = argv[++i];
    else if (f == "--items") items = argv[++i];
    else if (f == "--arms")
      while (i + 1 < argc && argv[i + 1][0] != '-') arms.push_back(arm_of(argv[++i]));
  }
  if (arms.empty())
    for (int a = 0; a < NARM; a++) arms.push_back((Arm)a);
  const int L = k / SEG;
  if (L > MAXL || m % 512 || n % 512 || k % (8 * SEG) || co2 % 256 || co2 >= n) {
    fprintf(stderr, "ERR shape: m, n multiples of 512, k of 256, k/32 <= %d, --co2 a multiple of 256 below n\n", MAXL);
    return 2;
  }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  fprintf(stderr, "m=%d k=%d n=%d L=%d mode=%s sms=%d l2=%d\n", m, k, n, L, mode.c_str(), prop.multiProcessorCount,
          prop.l2CacheSize);

  {
    std::vector<uint4> key(L);
    for (int e = 0; e < L; e++) key[e] = make_uint4(key_a(e), key_k(e), (uint32_t)key_r(e), 0);
    CK(cudaMemcpyToSymbol(c_key, key.data(), L * sizeof(uint4)));
  }
  Inputs in = make_inputs(m, k, n, seed);
  std::vector<int8_t> Su = expand_u(in.U, n, k);
  Dev d{};
  d.m = m, d.k = k, d.n = n, d.L = L, d.share = share, d.co2 = co2;
  d.Ap_g = (const uint4*)upload(pack_a(in.At, m, k));
  d.Ap_z = (const uint4*)upload(pack_a(in.R, m, k));
  d.Bp_g = (const uint2*)upload(pack_b(in.Bt, n, k));
  d.Uf = (const uint2*)upload(pack_b(in.U, n, SEG));
  d.Aw_g = (const uint4*)upload(in.At);
  d.Aw_z = (const uint4*)upload(in.R);
  d.Bw_g = (const uint4*)upload(in.Bt);
  d.Bw_z = (const uint4*)upload(Su);
  d.Rw = upload(pack_rw(in.R, m, k));
  d.P = upload(planes(in.U, n));
  d.U = upload(in.U);
  d.Bp_b = (const uint2*)upload(pack_b(in.Rc, n, k));
  d.Rg = (const uint4*)upload(pack_g4(in.R, m, k));
  d.Cg = (const uint4*)upload(pack_g4(in.Rc, n, k));
  d.Cw = upload(pack_rw(in.Rc, n, k));  // design A's column bits [L][n], bit p of Cw[ell][j] = Rc[j, 32ell + p]
  CK(cudaMalloc(&d.H, (size_t)4 << 20));
  CK(cudaMalloc(&d.T, FR16_TABLE_BYTES));
  CK(cudaMalloc(&d.out, (size_t)m * n * 4));
  CK(cudaMalloc(&d.rec, sizeof(Clk) << 20));
  CK(cudaFuncSetAttribute(k_fr8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)F8_SMEM));
  CK(cudaFuncSetAttribute(k_a_fr8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)A_SMEM));
  CK(cudaFuncSetAttribute(k_co<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)F8_SMEM));
  CK(cudaFuncSetAttribute(k_co<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)F8_SMEM));
  CK(cudaFuncSetAttribute(k_co<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)F8_SMEM));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  CK(cudaStreamCreateWithFlags(&d.s2, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&d.fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&d.join, cudaEventDisableTiming));
  d.sms = prop.multiProcessorCount;
  d.p_smem2 = (int)(prop.sharedMemPerMultiprocessor / 3) + 1024;  // two blocks an SM fit, three don't
  d.p_smem3 = (int)(prop.sharedMemPerMultiprocessor / 4) + 1024;  // three fit, four don't
  CK(cudaFuncSetAttribute(k_imma_p, cudaFuncAttributeMaxDynamicSharedMemorySize, d.p_smem2));
  CK(cudaFuncSetAttribute(k_imma_p, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));

  if (mode == "gate") {
    std::vector<std::pair<int, int>> pairs;
    std::mt19937 pg(99);
    for (int t = 0; t < 4096; t++) pairs.push_back({(int)(pg() % m), (int)(pg() % n)});
    const std::vector<uint32_t> ref_g = ref_receipt(in.At.data(), in.Bt.data(), k, n, pairs);
    const std::vector<uint32_t> ref_z = ref_receipt(in.R.data(), Su.data(), k, n, pairs);
    const std::vector<uint32_t> ref_b = ref_receipt(in.R.data(), in.Rc.data(), k, n, pairs);
    const std::vector<uint32_t> ref_o = ref_dot(in.R.data(), in.Rc.data(), k, pairs);
    const std::vector<uint32_t>* refs[4] = {&ref_g, &ref_z, &ref_b, &ref_o};
    bool all = true;
    for (Arm a : arms) {
      const int jmax = written(a, d);
      size_t checked = 0, armchecked = 0;
      gk::PoisonGate g = gk::poison_gate(
          d.out, (size_t)m * n * 4, s, [&](cudaStream_t st) { launch(a, d, st); },
          [&](const void* h, size_t) {
            const uint32_t* r = (const uint32_t*)h;
            checked = 0;
            for (size_t q = 0; q < pairs.size(); q++) {
              if (pairs[q].second >= jmax) continue;
              checked++;
              if (r[(size_t)pairs[q].first * n + pairs[q].second] != (*refs[ref_of(a, pairs[q].second, d)])[q])
                return false;
            }
            armchecked = std::max(armchecked, checked);  // the poisoned control stops at its first mismatch
            return checked > 0;
          });
      printf("GATE arm=%s accepted=%d control_rejected=%d left=%zu checked=%zu\n", NAMES[a], g.accepted,
             g.control_rejected, g.left, armchecked);
      all &= g.ok();
    }
    gk::Dump dump(dumpdir);
    for (Arm a : arms) {
      launch(a, d, s);
      CK(cudaStreamSynchronize(s));
      dump.bytes(std::string("receipt_") + NAMES[a], d.out, (size_t)m * n * 4, s);
    }
    return all ? 0 : 3;
  }

  gk::Nvml nv(0);
  std::vector<gk::Arm> gkarms;
  for (Arm a : arms)
    gkarms.push_back({NAMES[a], [=](cudaStream_t st, int cnt) {
                        for (int i = 0; i < cnt; i++) launch(a, d, st);
                      }});
  const gk::Timing t = gk::timed_items(gkarms, reps, s, nv);
  gk::summary(t, gkarms);
  // words a call: every written (i, j) and segment; the co arms make the whole product (IMMA's share and the scalar's)
  for (size_t a = 0; a < arms.size(); a++) {
    std::vector<double> ms, mhz;
    for (const gk::Item& it : t.items)
      if (it.arm == (int)a) {
        ms.push_back(it.ms / it.calls);
        if (it.sm_mhz()) mhz.push_back(it.sm_mhz());
      }
    const double med = gk::median(ms), f = mhz.empty() ? NAN : gk::median(mhz);
    const double w = (double)m * written(arms[a], d) * L;
    const double smsp_cyc = med * 1e-3 * f * 1e6 * prop.multiProcessorCount * 4;  // a word is 32 products
    printf("QSTAR arm=%s m=%d k=%d n=%d words=%.0f ms=%.6f ps_per_word=%.4f words_per_sm_clk=%.4f sm_mhz=%.0f "
           "smsp_cyc_per_4096=%.3f\n",
           NAMES[arms[a]], m, k, n, w, med, 1e9 * med / w, w / (med * 1e-3 * prop.multiProcessorCount * f * 1e6), f,
           smsp_cyc / (w * SEG / 4096.0));
  }
  if (!items.empty()) {
    FILE* fi = fopen(items.c_str(), "w");
    gk::items_jsonl(t, gkarms, fi);
    fclose(fi);
  }
  if (nv.ok()) fprintf(stderr, "uuid=%s\n", nv.uuid().c_str());
  return 0;
}
