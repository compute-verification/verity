// Credit ceilings: the best sustained FP8-equivalent MAC rate each non-tensor route reaches on one sm_120 SM, as exact
// E4M3 GEMMs checked bit for bit against a CPU int64 reference (CMP-12, worker W10; README.md says what each route is).
//
// Every route computes C = A * B exactly, A (M x K) and B (K x N) E4M3 codes, C in units of 2^-18 as int64 (an E4M3
// value times 2^9 is an integer below 2^18, so a product times 2^18 is an integer below 2^36). Two input distributions:
//   generic  every finite E4M3 code, uniformly (254 codes): 15 binades, the protocol's inputs in general;
//   narrow   exponent field 4..7 (4 binades) and zero: the subset that is exactly int8 at one scale (q = 8..120), the
//            most favorable input a dp4a route can get, and the input of run r20260930-085618-5cfa's r_g = 0.23.
// A route's rate is M*N*K over (SMs x SM cycles); the SM clock of the launch is every block's clock64 over %globaltimer.
// Output: one JSON line per measurement on stdout.
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

// ---------------------------------------------------------------- timing records

struct BlockClk {
  long long cyc, ns;
};

__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

#define CLK_BEGIN       \
  long long _c0 = clock64(); \
  long long _g0 = gtimer();
#define CLK_END                                                                    \
  if (rec && threadIdx.x == 0) {                                                   \
    rec[blockIdx.x + gridDim.x * blockIdx.y].cyc = clock64() - _c0;                \
    rec[blockIdx.x + gridDim.x * blockIdx.y].ns = gtimer() - _g0;                  \
  }

// ---------------------------------------------------------------- E4M3 on the host

static int64_t e4m3_v(uint8_t c) {  // value * 2^9, exact
  int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
  int64_t v = e == 0 ? m : (int64_t)(8 + m) << (e - 1);
  return s ? -v : v;
}
static bool e4m3_nan(uint8_t c) { return (c & 0x7F) == 0x7F; }

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint32_t next() {
    s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
    return (uint32_t)((s * 0x2545F4914F6CDD1Dull) >> 32);
  }
};

static void gen_codes(std::vector<uint8_t>& v, size_t n, bool narrow, uint64_t seed) {
  Rng r(seed);
  v.resize(n);
  for (size_t i = 0; i < n; i++) {
    if (narrow) {
      uint32_t x = r.next() % 65;
      v[i] = x == 64 ? 0 : (uint8_t)(((x & 1) << 7) | ((4 + ((x >> 1) & 3)) << 3) | ((x >> 3) & 7));
    } else {
      uint8_t c;
      do c = (uint8_t)r.next(); while (e4m3_nan(c));
      v[i] = c;
    }
  }
}

// A is M x K row-major codes, B is K x N row-major codes.
static int64_t ref_dot(const std::vector<uint8_t>& A, const std::vector<uint8_t>& B, int N, int K, int i, int j) {
  int64_t s = 0;
  for (int k = 0; k < K; k++) s += e4m3_v(A[(size_t)i * K + k]) * e4m3_v(B[(size_t)k * N + j]);
  return s;
}

// ---------------------------------------------------------------- decoder self-test against the hardware conversion

__global__ void k_decode_all(float* out) {
  int c = threadIdx.x;
  __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)c, __NV_E4M3);
  out[c] = __half2float(__half(h)) * 512.0f;
}

static bool decoder_selftest() {
  float* d;
  CK(cudaMalloc(&d, 256 * sizeof(float)));
  k_decode_all<<<1, 256>>>(d);
  CK(cudaGetLastError());
  std::vector<float> h(256);
  CK(cudaMemcpy(h.data(), d, 256 * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaFree(d));
  int bad = 0;
  for (int c = 0; c < 256; c++)
    if (!e4m3_nan((uint8_t)c) && (double)h[c] != (double)e4m3_v((uint8_t)c)) bad++;
  printf("{\"kind\":\"decoder_selftest\",\"codes\":254,\"mismatches\":%d}\n", bad);
  return bad == 0;
}

// ---------------------------------------------------------------- route kernels
// Thread grid 16 x 16 (tx = threadIdx.x & 15 over columns, ty = threadIdx.x >> 4 over rows); a thread owns TM x TN
// outputs. Operands are k-major, so a warp's B loads are 16 contiguous vectors and its A loads two broadcast addresses.

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Staged register-tiled GEMM over words: a word is 4 packed int8 k's (dp4a) or one FP32 k (FFMA), k-major,
// A[(l*KW + kw)*M + i] and B[(l*KW + kw)*N + j] for limb l. BKW words a stage through a STAGES-deep cp.async
// pipeline; persistent over tiles (block b takes tiles b, b + gridDim.x, ... below ntiles). Thread (tx, ty) owns rows
// m0 + 64v + 4ty + r and columns n0 + 64v + 4tx + c, so a warp's shared-memory reads are two broadcast addresses (A)
// and 256 contiguous bytes (B).
//   EPI 0: narrow inputs, one int8 per value (q = v/8), C = 64 P.
//   EPI 1: generic inputs, 3 signed 6-bit digits per value; limbs are Karatsuba's 6 operands (d0, d1, d2, d0+d1,
//          d1+d2, d0+d2), whose pairwise products give the 5 digit-product coefficients.
//   EPI 2: narrow inputs as FP32 (L = 1): products exact, sums exact below 2^24 (14400 * 1024 < 2^24), so the
//          accumulators flush into C every FLUSHW words.
template <int EPI, int TM, int TN, int L, int BKW, int STAGES, int FLUSHW, int MINB>
__global__ void __launch_bounds__(256, MINB) k_gemm(const void* __restrict__ Av, const void* __restrict__ Bv,
                                              long long* __restrict__ C, int M, int N, int KW, int ntiles,
                                              BlockClk* rec) {
  using T = typename std::conditional<EPI == 2, float, int>::type;
  using V = typename std::conditional<EPI == 2, float4, int4>::type;
  constexpr int BM = 16 * TM, BN = 16 * TN;
  __shared__ __align__(16) T As[STAGES][L][BKW][BM];
  __shared__ __align__(16) T Bs[STAGES][L][BKW][BN];
  const T* A = reinterpret_cast<const T*>(Av);
  const T* B = reinterpret_cast<const T*>(Bv);
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int tiles_n = N / BN, nst = KW / BKW;
  CLK_BEGIN
  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    const int m0 = (tile / tiles_n) * BM, n0 = (tile % tiles_n) * BN;
    T acc[L][TM][TN];
#pragma unroll
    for (int l = 0; l < L; l++)
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[l][i][j] = 0;
    auto load_stage = [&](int s, int kw0) {
      constexpr int CA = L * BKW * (BM / 4), CB = L * BKW * (BN / 4);
      for (int c = threadIdx.x; c < CA; c += 256) {
        const int row = c / (BM / 4), c4 = c % (BM / 4), l = row / BKW, kk = row % BKW;
        cp_async16(&As[s][l][kk][4 * c4], A + ((size_t)l * KW + kw0 + kk) * M + m0 + 4 * c4);
      }
      for (int c = threadIdx.x; c < CB; c += 256) {
        const int row = c / (BN / 4), c4 = c % (BN / 4), l = row / BKW, kk = row % BKW;
        cp_async16(&Bs[s][l][kk][4 * c4], B + ((size_t)l * KW + kw0 + kk) * N + n0 + 4 * c4);
      }
    };
    __syncthreads();
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < nst) load_stage(s, s * BKW);
      cp_commit();
    }
    for (int st = 0; st < nst; st++) {
      cp_wait<STAGES - 2>();
      __syncthreads();
      {
        const int nx = st + STAGES - 1;
        if (nx < nst) load_stage(nx % STAGES, nx * BKW);
        cp_commit();
      }
      const int s = st % STAGES;
#pragma unroll
      for (int kk = 0; kk < BKW; kk++) {
#pragma unroll
        for (int l = 0; l < L; l++) {
          T a[TM], b[TN];
#pragma unroll
          for (int v = 0; v < TM / 4; v++) {
            V t = *reinterpret_cast<const V*>(&As[s][l][kk][64 * v + 4 * ty]);
            a[4 * v] = t.x, a[4 * v + 1] = t.y, a[4 * v + 2] = t.z, a[4 * v + 3] = t.w;
          }
#pragma unroll
          for (int v = 0; v < TN / 4; v++) {
            V t = *reinterpret_cast<const V*>(&Bs[s][l][kk][64 * v + 4 * tx]);
            b[4 * v] = t.x, b[4 * v + 1] = t.y, b[4 * v + 2] = t.z, b[4 * v + 3] = t.w;
          }
#pragma unroll
          for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) {
              if constexpr (EPI == 2) acc[l][i][j] = __fmaf_rn(a[i], b[j], acc[l][i][j]);
              else acc[l][i][j] = __dp4a(a[i], b[j], acc[l][i][j]);
            }
        }
        if constexpr (EPI == 2) {
          const int w = st * BKW + kk + 1;
          if (w % FLUSHW == 0) {
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
              for (int j = 0; j < TN; j++) {
                long long* p = &C[(size_t)(m0 + 64 * (i / 4) + 4 * ty + i % 4) * N + n0 + 64 * (j / 4) + 4 * tx + j % 4];
                const long long v = 64ll * (long long)acc[0][i][j];
                *p = w == FLUSHW ? v : *p + v;
                acc[0][i][j] = 0.f;
              }
          }
        }
      }
    }
    if constexpr (EPI != 2) {
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) {
          long long c;
          if constexpr (EPI == 0) {
            c = 64ll * acc[0][i][j];
          } else {
            const long long p0 = acc[0][i][j], p1 = acc[1][i][j], p2 = acc[2][i][j];
            const long long p01 = acc[3][i][j], p12 = acc[4][i][j], p02 = acc[5][i][j];
            c = p0 + ((p01 - p0 - p1) << 6) + ((p02 - p0 - p2 + p1) << 12) + ((p12 - p1 - p2) << 18) + (p2 << 24);
          }
          C[(size_t)(m0 + 64 * (i / 4) + 4 * ty + i % 4) * N + n0 + 64 * (j / 4) + 4 * tx + j % 4] = c;
        }
    }
  }
  CLK_END
}

// The three configurations timed: tile, limbs, stage depth.
#define GEMM_NARROW k_gemm<0, 8, 8, 1, 8, 3, 0, 2>
#define GEMM_K3 k_gemm<1, 4, 4, 6, 4, 3, 0, 1>
#define GEMM_FFMA k_gemm<2, 8, 8, 1, 8, 3, 1024, 2>

// Integer SWAR on IMAD, narrow inputs: two rows packed in one word, w = q[2p] + 2^16 q[2p+1]; IMAD acc += w * q_b
// gives both products in 16-bit lanes. A lane holds two products (2 * 14400 < 2^15), so lanes are split into int32
// sums every 2 k's. Aw is [k][M/2], Bi is [k][N] (int32).
template <int TP, int TN>
__global__ void __launch_bounds__(256) k_swar2(const int* __restrict__ Aw, const int* __restrict__ Bi,
                                               long long* __restrict__ C, int M, int N, int K, BlockClk* rec) {
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int pr0 = blockIdx.y * (16 * TP) + ty * TP, col0 = blockIdx.x * (16 * TN) + tx * TN;
  const int MP = M / 2;
  int lo[TP][TN], hi[TP][TN];
#pragma unroll
  for (int p = 0; p < TP; p++)
#pragma unroll
    for (int j = 0; j < TN; j++) lo[p][j] = hi[p][j] = 0;
  CLK_BEGIN
#pragma unroll 2
  for (int k = 0; k < K; k += 2) {
    int a0[TP], a1[TP], b0[TN], b1[TN];
#pragma unroll
    for (int v = 0; v < TP / 4; v++) {
      int4 t = __ldg(reinterpret_cast<const int4*>(Aw + (size_t)k * MP + pr0) + v);
      int4 u = __ldg(reinterpret_cast<const int4*>(Aw + (size_t)(k + 1) * MP + pr0) + v);
      a0[4 * v] = t.x, a0[4 * v + 1] = t.y, a0[4 * v + 2] = t.z, a0[4 * v + 3] = t.w;
      a1[4 * v] = u.x, a1[4 * v + 1] = u.y, a1[4 * v + 2] = u.z, a1[4 * v + 3] = u.w;
    }
#pragma unroll
    for (int v = 0; v < TN / 4; v++) {
      int4 t = __ldg(reinterpret_cast<const int4*>(Bi + (size_t)k * N + col0) + v);
      int4 u = __ldg(reinterpret_cast<const int4*>(Bi + (size_t)(k + 1) * N + col0) + v);
      b0[4 * v] = t.x, b0[4 * v + 1] = t.y, b0[4 * v + 2] = t.z, b0[4 * v + 3] = t.w;
      b1[4 * v] = u.x, b1[4 * v + 1] = u.y, b1[4 * v + 2] = u.z, b1[4 * v + 3] = u.w;
    }
#pragma unroll
    for (int p = 0; p < TP; p++)
#pragma unroll
      for (int j = 0; j < TN; j++) {
        int w = (int)((unsigned)a1[p] * (unsigned)b1[j] + (unsigned)a0[p] * (unsigned)b0[j]);
        int l = (int)(short)(w & 0xFFFF);
        lo[p][j] += l;
        hi[p][j] += (w - l) >> 16;
      }
  }
  CLK_END
#pragma unroll
  for (int p = 0; p < TP; p++)
#pragma unroll
    for (int j = 0; j < TN; j++) {
      C[(size_t)(2 * (pr0 + p)) * N + col0 + j] = 64ll * lo[p][j];
      C[(size_t)(2 * (pr0 + p) + 1) * N + col0 + j] = 64ll * hi[p][j];
    }
}

// Shared-memory product table (Four Russians with one code a lookup), generic inputs, exact. For each pair of k's the
// block builds T[k2][q][code] = (hi, lo) of v(code) * v(B[k][col]) for its 4 columns (q = column pair), with
// p = hi * 2^18 + lo, 0 <= lo < 2^18; a thread then looks up its rows' codes and adds hi and lo into int32 sums with
// IADD3 (two k's an add). The layout puts consecutive codes in consecutive 16-byte bank groups.
// A2 holds A's codes for k and k+1 as one ushort, k-major [K/2][M]; Bv is v(B) as int32 [K][N].
// Block: 512 threads x RM rows, 4 columns; the table build is inside the timed region.
template <int RM>
__global__ void __launch_bounds__(512) k_lut(const unsigned short* __restrict__ A2, const int* __restrict__ Bv,
                                             long long* __restrict__ C, int M, int N, int K, BlockClk* rec) {
  __shared__ int4 T[2][2][2][256];  // [buffer][k of the pair][column pair][code]: 32 KB
  const int col0 = blockIdx.x * 4;
  const int t = threadIdx.x;
  const int rowb = blockIdx.y * (512 * RM);
  int hs[RM][4], ls[RM][4];
#pragma unroll
  for (int r = 0; r < RM; r++)
#pragma unroll
    for (int j = 0; j < 4; j++) hs[r][j] = ls[r][j] = 0;
  auto build = [&](int buf, int k) {
    // 2 k's x 2 column pairs x 256 codes = 1024 entries, 2 a thread
    for (int e = t; e < 1024; e += 512) {
      int kk = e >> 9, q = (e >> 8) & 1, code = e & 255;
      int s = code >> 7, ex = (code >> 3) & 15, m = code & 7;
      int va = ex == 0 ? m : (8 + m) << (ex - 1);
      if (s) va = -va;
      if ((code & 0x7F) == 0x7F) va = 0;
      long long p0 = (long long)va * Bv[(size_t)(k + kk) * N + col0 + 2 * q];
      long long p1 = (long long)va * Bv[(size_t)(k + kk) * N + col0 + 2 * q + 1];
      int h0 = (int)(p0 >> 18), h1 = (int)(p1 >> 18);
      T[buf][kk][q][code] = make_int4(h0, (int)(p0 - ((long long)h0 << 18)), h1, (int)(p1 - ((long long)h1 << 18)));
    }
  };
  CLK_BEGIN
  build(0, 0);
  __syncthreads();
  for (int k = 0; k < K; k += 2) {
    const int buf = (k >> 1) & 1;
    if (k + 2 < K) build(buf ^ 1, k + 2);
#pragma unroll
    for (int r = 0; r < RM; r++) {
      const int row = rowb + r * 512 + t;
      const unsigned cc = __ldg(A2 + (size_t)(k >> 1) * M + row);
      const int c0 = cc & 255, c1 = cc >> 8;
#pragma unroll
      for (int q = 0; q < 2; q++) {
        int4 x = T[buf][0][q][c0], y = T[buf][1][q][c1];
        hs[r][2 * q] += x.x + y.x;
        ls[r][2 * q] += x.y + y.y;
        hs[r][2 * q + 1] += x.z + y.z;
        ls[r][2 * q + 1] += x.w + y.w;
      }
    }
    __syncthreads();
  }
  CLK_END
#pragma unroll
  for (int r = 0; r < RM; r++) {
    const int row = rowb + r * 512 + t;
#pragma unroll
    for (int j = 0; j < 4; j++) C[(size_t)row * N + col0 + j] = ((long long)hs[r][j] << 18) + ls[r][j];
  }
}

// The same table route with fewer barriers and more reuse: a stage is 8 k's, T[buf][kk][code] = (hi, lo) for 2
// columns, double-buffered in 64 KB of dynamic shared memory; A8 holds 8 codes of a row as a uint2, k-major
// [K/8][M]; 512 threads x RM rows a block, so the build (2,048 entries a stage) is spread over 4,096 x 2 x 8 MACs.
template <int RM>
__global__ void __launch_bounds__(512, 1) k_lut8(const uint2* __restrict__ A8, const int* __restrict__ Bv,
                                                 long long* __restrict__ C, int M, int N, int K, BlockClk* rec) {
  extern __shared__ int4 T8[];  // [2][8][256]
  const int col0 = blockIdx.x * 2, t = threadIdx.x, rowb = blockIdx.y * (512 * RM);
  int h0[RM], l0[RM], h1[RM], l1[RM];
#pragma unroll
  for (int r = 0; r < RM; r++) h0[r] = l0[r] = h1[r] = l1[r] = 0;
  auto build = [&](int buf, int k0) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int e = t + 512 * q, kk = e >> 8, code = e & 255;
      const int s = code >> 7, ex = (code >> 3) & 15, m = code & 7;
      int va = ex == 0 ? m : (8 + m) << (ex - 1);
      if (s) va = -va;
      const long long p0 = (long long)va * __ldg(Bv + (size_t)(k0 + kk) * N + col0);
      const long long p1 = (long long)va * __ldg(Bv + (size_t)(k0 + kk) * N + col0 + 1);
      const int hh0 = (int)(p0 >> 18), hh1 = (int)(p1 >> 18);
      T8[(buf * 8 + kk) * 256 + code] =
          make_int4(hh0, (int)(p0 - ((long long)hh0 << 18)), hh1, (int)(p1 - ((long long)hh1 << 18)));
    }
  };
  CLK_BEGIN
  build(0, 0);
  __syncthreads();
  const int nst = K / 8;
  for (int st = 0; st < nst; st++) {
    const int buf = st & 1;
    if (st + 1 < nst) build(buf ^ 1, 8 * (st + 1));
    const int4* Tb = T8 + buf * 8 * 256;
#pragma unroll
    for (int r = 0; r < RM; r++) {
      const uint2 cc = __ldg(A8 + (size_t)st * M + rowb + r * 512 + t);
#pragma unroll
      for (int kk = 0; kk < 8; kk += 2) {
        const unsigned w = kk < 4 ? cc.x : cc.y;
        const int sh = 8 * (kk & 3);
        const int4 x = Tb[kk * 256 + ((w >> sh) & 255)], y = Tb[(kk + 1) * 256 + ((w >> (sh + 8)) & 255)];
        h0[r] += x.x + y.x;
        l0[r] += x.y + y.y;
        h1[r] += x.z + y.z;
        l1[r] += x.w + y.w;
      }
    }
    __syncthreads();
  }
  CLK_END
#pragma unroll
  for (int r = 0; r < RM; r++) {
    const size_t row = rowb + r * 512 + t;
    C[row * N + col0] = ((long long)h0[r] << 18) + l0[r];
    C[row * N + col0 + 1] = ((long long)h1[r] << 18) + l1[r];
  }
}

// ---------------------------------------------------------------- pipe-pairing probe
// 32 warps a SM, one block a SM, 8 independent chains per op; per chain step ND dp4a, NI IMAD, NF FFMA, NL LOP3 and
// NF2 packed FFMA2 (fma.rn.f32x2). Immediates keep FFMA off the register-bank limit.
template <int ND, int NI, int NF, int NL, int NF2>
__global__ void __launch_bounds__(1024) k_probe(int* out, int iters, BlockClk* rec) {
  int d[8], m[8], l[8];
  float f[8];
  unsigned long long g[8];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    d[r] = m[r] = l[r] = threadIdx.x + r, f[r] = (float)(threadIdx.x + r);
    g[r] = ((unsigned long long)__float_as_uint(f[r] + 0.5f) << 32) | __float_as_uint(f[r]);
  }
  const int x = 0x01020304 + blockIdx.x, y = 0x05060708 ^ threadIdx.x;
  const unsigned long long gb = 0x3F8001003F800100ull, gc = 0x3F0000003F000000ull;
  __syncthreads();
  CLK_BEGIN
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int r = 0; r < 8; r++) {
#pragma unroll
      for (int q = 0; q < ND; q++) asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(d[r]) : "r"(x), "r"(y));
#pragma unroll
      for (int q = 0; q < NI; q++) asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(m[r]) : "r"(x), "r"(y));
#pragma unroll
      for (int q = 0; q < NF; q++) asm volatile("fma.rn.f32 %0, %0, 0f3F800100, 0f3F000000;" : "+f"(f[r]));
#pragma unroll
      for (int q = 0; q < NL; q++) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(l[r]) : "r"(x), "r"(y));
#pragma unroll
      for (int q = 0; q < NF2; q++) asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(g[r]) : "l"(gb), "l"(gc));
    }
  }
  __syncthreads();
  CLK_END
  int s = 0;
#pragma unroll
  for (int r = 0; r < 8; r++) s += d[r] + m[r] + l[r] + __float_as_int(f[r]) + (int)g[r];
  if (s == 0x7FFFFFFF) out[0] = s;
}


// Split warps: warps with bit 2 of their id clear run ND dp4a a chain step, the others NF FFMA and NF2 FFMA2, so the
// warp schedulers interleave the two routes at run time and no static SASS order decides the mix. Warps go to the SM's
// four sub-partitions by id mod 4, so every sub-partition holds four warps of each role (an even/odd split would put the
// roles on different sub-partitions). Each warp records its own loop cycles.
template <int ND, int NF, int NF2>
__global__ void __launch_bounds__(1024) k_split(int* out, int iters, long long* wcyc, BlockClk* rec) {
  int d[8];
  float f[8];
  unsigned long long g[8];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    d[r] = threadIdx.x + r, f[r] = (float)(threadIdx.x + r);
    g[r] = ((unsigned long long)__float_as_uint(f[r] + 0.5f) << 32) | __float_as_uint(f[r]);
  }
  const int x = 0x01020304 + blockIdx.x, y = 0x05060708 ^ threadIdx.x, warp = threadIdx.x >> 5;
  const unsigned long long gb = 0x3F8001003F800100ull, gc = 0x3F0000003F000000ull;
  __syncthreads();
  CLK_BEGIN
  const long long c0 = clock64();
  if ((warp & 4) == 0) {
    for (int it = 0; it < iters; it++)
#pragma unroll
      for (int r = 0; r < 8; r++)
#pragma unroll
        for (int q = 0; q < ND; q++) asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(d[r]) : "r"(x), "r"(y));
  } else {
    for (int it = 0; it < iters; it++)
#pragma unroll
      for (int r = 0; r < 8; r++) {
#pragma unroll
        for (int q = 0; q < NF; q++) asm volatile("fma.rn.f32 %0, %0, 0f3F800100, 0f3F000000;" : "+f"(f[r]));
#pragma unroll
        for (int q = 0; q < NF2; q++) asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(g[r]) : "l"(gb), "l"(gc));
      }
  }
  const long long c1 = clock64();
  if ((threadIdx.x & 31) == 0) wcyc[blockIdx.x * 32 + warp] = c1 - c0;
  __syncthreads();
  CLK_END
  int s = 0;
#pragma unroll
  for (int r = 0; r < 8; r++) s += d[r] + __float_as_int(f[r]) + (int)g[r];
  if (s == 0x7FFFFFFF) out[0] = s;
}

// ---------------------------------------------------------------- host side

static int g_nsm = 0;

struct Meas {
  double ms, mhz, cyc_med;
};

static Meas timed(const std::function<void(BlockClk*)>& launch, int nblocks, int reps) {
  BlockClk* rec;
  CK(cudaMalloc(&rec, sizeof(BlockClk) * nblocks));
  launch(nullptr);  // warm-up
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<double> ms(reps), mhz(reps), cyc(reps);
  std::vector<BlockClk> h(nblocks);
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(e0));
    launch(rec);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    ms[r] = t;
    CK(cudaMemcpy(h.data(), rec, sizeof(BlockClk) * nblocks, cudaMemcpyDeviceToHost));
    std::vector<double> f, c;
    for (auto& b : h) {
      if (b.ns > 20000) f.push_back((double)b.cyc / (double)b.ns * 1000.0);
      c.push_back((double)b.cyc);
    }
    std::sort(f.begin(), f.end());
    std::sort(c.begin(), c.end());
    mhz[r] = f.empty() ? 0 : f[f.size() / 2];
    cyc[r] = c[c.size() / 2];
  }
  CK(cudaGetLastError());
  CK(cudaFree(rec));
  std::vector<double> s = ms;
  std::sort(s.begin(), s.end());
  int mid = reps / 2;
  int idx = (int)(std::find(ms.begin(), ms.end(), s[mid]) - ms.begin());
  return Meas{s[mid], mhz[idx], cyc[idx]};
}

static void emit(const char* route, const char* dist, int M, int N, int K, const Meas& m, int occ, long long checked,
                 long long bad, const char* extra = "") {
  double macs = (double)M * N * K;
  double per_sm_clk = m.mhz > 0 ? macs / (g_nsm * m.ms * 1e-3 * m.mhz * 1e6) : 0.0;
  printf(
      "{\"kind\":\"route\",\"route\":\"%s\",\"dist\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ms_median\":%.4f,"
      "\"sm_mhz\":%.1f,\"blocks_per_sm\":%d,\"macs_per_sm_clk\":%.2f,\"share_of_1024\":%.4f,\"checked\":%lld,"
      "\"mismatches\":%lld%s}\n",
      route, dist, M, N, K, m.ms, m.mhz, occ, per_sm_clk, per_sm_clk / 1024.0, checked, bad, extra);
  fflush(stdout);
}

// Compare C against the reference: every output when full, else `samples` random ones.
static long long check(const std::vector<long long>& C, const std::vector<uint8_t>& A, const std::vector<uint8_t>& B,
                       int M, int N, int K, int samples, long long* checked) {
  long long bad = 0;
  Rng r(12345);
  int n = samples <= 0 ? M * N : samples;
  for (int s = 0; s < n; s++) {
    int i, j;
    if (samples <= 0) i = s / N, j = s % N;
    else i = r.next() % M, j = r.next() % N;
    if (C[(size_t)i * N + j] != ref_dot(A, B, N, K, i, j)) {
      if (bad < 4)
        fprintf(stderr, "mismatch (%d,%d): got %lld want %lld\n", i, j, (long long)C[(size_t)i * N + j],
                (long long)ref_dot(A, B, N, K, i, j));
      bad++;
    }
  }
  *checked = n;
  return bad;
}

template <class T>
static T* to_dev(const std::vector<T>& h) {
  T* d;
  CK(cudaMalloc(&d, sizeof(T) * h.size()));
  CK(cudaMemcpy(d, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice));
  return d;
}

static int q_narrow(uint8_t c) { return (int)(e4m3_v(c) / 8); }  // exact: narrow values are multiples of 8

// Encodings (host, O(n^2), untimed: a GEMM encodes each operand once and reuses it n times).
static std::vector<int> pack_int8(const std::vector<uint8_t>& X, int rows, int K, bool rowsA, int L, int N_or_M) {
  // rowsA: X is M x K (A); else X is K x N (B). Output [(l*KW + kw)*R + r], R = rows of the k-major layout.
  (void)N_or_M;
  const int KW = K / 4, R = rows;
  std::vector<int> out((size_t)L * KW * R);
  for (int r = 0; r < R; r++)
    for (int k = 0; k < K; k++) {
      uint8_t c = rowsA ? X[(size_t)r * K + k] : X[(size_t)k * R + r];
      int lim[6];
      if (L == 1) {
        lim[0] = q_narrow(c);
      } else {
        int64_t v = e4m3_v(c);
        int s = v < 0 ? -1 : 1;
        int64_t a = v < 0 ? -v : v;
        int d0 = (int)(a & 63), d1 = (int)((a >> 6) & 63), d2 = (int)(a >> 12);
        lim[0] = s * d0, lim[1] = s * d1, lim[2] = s * d2;
        lim[3] = s * (d0 + d1), lim[4] = s * (d1 + d2), lim[5] = s * (d0 + d2);
      }
      for (int l = 0; l < L; l++) {
        uint32_t& w = reinterpret_cast<uint32_t&>(out[((size_t)l * KW + k / 4) * R + r]);
        w |= (uint32_t)(uint8_t)(int8_t)lim[l] << (8 * (k & 3));
      }
    }
  return out;
}

using GemmFn = void (*)(const void*, const void*, long long*, int, int, int, int, BlockClk*);

// Device operands for one route: A is M x K, B is K x N codes; the layout is the route's.
struct Operands {
  void *dA = nullptr, *dB = nullptr;
  int KW = 0;
};

template <int EPI>
static Operands upload(const std::vector<uint8_t>& A, const std::vector<uint8_t>& B, int M, int N, int K) {
  Operands o;
  if constexpr (EPI == 2) {
    std::vector<float> Af((size_t)K * M), Bf((size_t)K * N);
    for (int i = 0; i < M; i++)
      for (int k = 0; k < K; k++) Af[(size_t)k * M + i] = (float)q_narrow(A[(size_t)i * K + k]);
    for (int k = 0; k < K; k++)
      for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = (float)q_narrow(B[(size_t)k * N + j]);
    o.dA = to_dev(Af), o.dB = to_dev(Bf), o.KW = K;
  } else {
    const int L = EPI == 1 ? 6 : 1;
    auto Ap = pack_int8(A, M, K, true, L, 0), Bp = pack_int8(B, N, K, false, L, 0);
    o.dA = to_dev(Ap), o.dB = to_dev(Bp), o.KW = K / 4;
  }
  return o;
}

static const char* gemm_route(int epi) { return epi == 0 ? "dp4a_int8" : epi == 1 ? "dp4a_karatsuba3" : "ffma_fp32"; }

template <int EPI, int TM, int BKW, int STAGES, int MINB>
static void run_gemm(int M, int N, int K, int reps, int samples, int grid_blocks) {
  constexpr int BM = 16 * TM, BN = BM;
  std::vector<uint8_t> A, B;
  gen_codes(A, (size_t)M * K, EPI != 1, 1 + EPI + M + K);
  gen_codes(B, (size_t)K * N, EPI != 1, 2 + EPI + N + K);
  Operands o = upload<EPI>(A, B, M, N, K);
  long long* dC;
  CK(cudaMalloc(&dC, sizeof(long long) * M * N));
  const int ntiles = (M / BM) * (N / BN), grid = grid_blocks > 0 ? grid_blocks : ntiles;
  GemmFn f = k_gemm<EPI, TM, TM, EPI == 1 ? 6 : 1, BKW, STAGES, EPI == 2 ? 1024 : 0, MINB>;
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, 256, 0));
  auto launch = [&](BlockClk* rec) { f<<<grid, 256>>>(o.dA, o.dB, dC, M, N, o.KW, ntiles, rec); };
  Meas m = timed(launch, grid, reps);
  std::vector<long long> C((size_t)M * N);
  CK(cudaMemcpy(C.data(), dC, sizeof(long long) * M * N, cudaMemcpyDeviceToHost));
  long long checked;
  long long bad = check(C, A, B, M, N, K, samples, &checked);
  char extra[160];
  snprintf(extra, sizeof extra, ",\"grid\":%d,\"tiles\":%d,\"config\":\"TM%d BKW%d S%d MINB%d\",\"instr_per_mac\":%s",
           grid, ntiles, TM, BKW, STAGES, MINB, EPI == 0 ? "\"0.25 dp4a\"" : EPI == 1 ? "\"1.5 dp4a\"" : "\"1 FFMA\"");
  emit(gemm_route(EPI), EPI == 1 ? "generic" : "narrow", M, N, K, m, occ, checked, bad, extra);
  CK(cudaFree(o.dA));
  CK(cudaFree(o.dB));
  CK(cudaFree(dC));
}

// dp4a (narrow) and FFMA (narrow) GEMMs at once on two streams, one block of each a SM (188 tiles each). The dp4a
// GEMM's K is calibrated so both kernels end together; the combined rate is both GEMMs' MACs over the later end.
static void run_corun(int reps, int samples) {
  const int nsm = g_nsm, Md = 128 * (nsm / 4), Nd = 128 * 4, Mf = Md, Nf = Nd, Kf = 2048, Kd_max = 32768;
  std::vector<uint8_t> Ad, Bd, Af8, Bf8;
  gen_codes(Ad, (size_t)Md * Kd_max, true, 21);
  gen_codes(Bd, (size_t)Kd_max * Nd, true, 22);
  gen_codes(Af8, (size_t)Mf * Kf, true, 23);
  gen_codes(Bf8, (size_t)Kf * Nf, true, 24);
  Operands od = upload<0>(Ad, Bd, Md, Nd, Kd_max), of = upload<2>(Af8, Bf8, Mf, Nf, Kf);
  long long *dCd, *dCf;
  CK(cudaMalloc(&dCd, sizeof(long long) * Md * Nd));
  CK(cudaMalloc(&dCf, sizeof(long long) * Mf * Nf));
  BlockClk *rd, *rf;
  CK(cudaMalloc(&rd, sizeof(BlockClk) * nsm));
  CK(cudaMalloc(&rf, sizeof(BlockClk) * nsm));
  cudaStream_t s0, s1;
  CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  cudaEvent_t e0, ed, ef;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&ed));
  CK(cudaEventCreate(&ef));
  const int ntd = nsm, ntf = nsm;
  // A's packed layout is [kw][M] for the whole Kd_max, so a shorter K reads the first kw rows: the same matrix
  // truncated in k, which the check below follows.
  auto go = [&](int Kd, bool d_on, bool f_on, float* td, float* tf) {
    CK(cudaEventRecord(e0, s0));
    CK(cudaStreamWaitEvent(s1, e0, 0));
    if (d_on) GEMM_NARROW<<<nsm, 256, 0, s0>>>(od.dA, od.dB, dCd, Md, Nd, Kd / 4, ntd, rd);
    if (f_on) GEMM_FFMA<<<nsm, 256, 0, s1>>>(of.dA, of.dB, dCf, Mf, Nf, Kf, ntf, rf);
    CK(cudaEventRecord(ed, s0));
    CK(cudaEventRecord(ef, s1));
    CK(cudaEventSynchronize(ed));
    CK(cudaEventSynchronize(ef));
    CK(cudaGetLastError());
    CK(cudaEventElapsedTime(td, e0, ed));
    CK(cudaEventElapsedTime(tf, e0, ef));
  };
  float td, tf;
  int Kd = 8192;
  go(Kd, true, true, &td, &tf);  // warm-up
  for (int it = 0; it < 3; it++) {
    go(Kd, true, true, &td, &tf);
    Kd = std::min(Kd_max, std::max(32, (int)std::lround(Kd * (double)tf / td / 32.0) * 32));
  }
  std::vector<float> both, tds, tfs;
  for (int r = 0; r < reps; r++) {
    go(Kd, true, true, &td, &tf);
    both.push_back(std::max(td, tf)), tds.push_back(td), tfs.push_back(tf);
  }
  std::vector<BlockClk> h(nsm);
  CK(cudaMemcpy(h.data(), rd, sizeof(BlockClk) * nsm, cudaMemcpyDeviceToHost));
  std::vector<double> mhz;
  for (auto& b : h) mhz.push_back((double)b.cyc / (double)b.ns * 1000.0);
  std::sort(mhz.begin(), mhz.end());
  const double clk = mhz[nsm / 2];
  float tda, tfa, dummy;
  go(Kd, true, false, &tda, &dummy);
  go(Kd, false, true, &dummy, &tfa);
  auto med = [](std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return (double)v[v.size() / 2];
  };
  const double tb = med(both), macs_d = (double)Md * Nd * Kd, macs_f = (double)Mf * Nf * Kf;
  const double per_ms = nsm * clk * 1e3;  // SM clocks a ms over the device
  std::vector<long long> Cd((size_t)Md * Nd), Cf((size_t)Mf * Nf);
  go(Kd, true, true, &td, &tf);
  CK(cudaMemcpy(Cd.data(), dCd, sizeof(long long) * Md * Nd, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(Cf.data(), dCf, sizeof(long long) * Mf * Nf, cudaMemcpyDeviceToHost));
  std::vector<uint8_t> AdK((size_t)Md * Kd), BdK((size_t)Kd * Nd);
  for (int i = 0; i < Md; i++)
    for (int k = 0; k < Kd; k++) AdK[(size_t)i * Kd + k] = Ad[(size_t)i * Kd_max + k];
  std::copy(Bd.begin(), Bd.begin() + (size_t)Kd * Nd, BdK.begin());
  long long c1, c2;
  long long bad = check(Cd, AdK, BdK, Md, Nd, Kd, samples, &c1) + check(Cf, Af8, Bf8, Mf, Nf, Kf, samples, &c2);
  const double comb = (macs_d + macs_f) / (tb * per_ms);
  printf(
      "{\"kind\":\"route\",\"route\":\"corun_dp4a_ffma\",\"dist\":\"narrow\",\"M\":%d,\"N\":1,\"K\":1,\"Kd\":%d,\"Kf\":%d,"
      "\"ms_median\":%.4f,\"ms_dp4a_in_corun\":%.4f,\"ms_ffma_in_corun\":%.4f,\"ms_dp4a_alone\":%.4f,"
      "\"ms_ffma_alone\":%.4f,\"sm_mhz\":%.1f,\"blocks_per_sm\":2,\"macs_per_sm_clk\":%.2f,\"share_of_1024\":%.4f,"
      "\"dp4a_macs_per_sm_clk\":%.2f,\"ffma_macs_per_sm_clk\":%.2f,\"dp4a_alone_macs_per_sm_clk\":%.2f,"
      "\"ffma_alone_macs_per_sm_clk\":%.2f,\"corun_over_sum_of_alone\":%.3f,\"checked\":%lld,\"mismatches\":%lld}\n",
      (int)((macs_d + macs_f) / 1e3), Kd, Kf, tb, med(tds), med(tfs), (double)tda, (double)tfa, clk, comb,
      comb / 1024.0, macs_d / (tb * per_ms), macs_f / (tb * per_ms), macs_d / (tda * per_ms),
      macs_f / (tfa * per_ms), tb / (tda + tfa), c1 + c2, bad);
  fflush(stdout);
  CK(cudaFree(od.dA));
  CK(cudaFree(od.dB));
  CK(cudaFree(of.dA));
  CK(cudaFree(of.dB));
  CK(cudaFree(dCd));
  CK(cudaFree(dCf));
  CK(cudaFree(rd));
  CK(cudaFree(rf));
}


static void run_swar(int M, int N, int K, int reps, int samples) {
  std::vector<uint8_t> A, B;
  gen_codes(A, (size_t)M * K, true, 5 + M + K);
  gen_codes(B, (size_t)K * N, true, 6 + N + K);
  std::vector<int> Aw((size_t)K * (M / 2)), Bi((size_t)K * N);
  for (int p = 0; p < M / 2; p++)
    for (int k = 0; k < K; k++)
      Aw[(size_t)k * (M / 2) + p] =
          (int)((uint32_t)q_narrow(A[(size_t)(2 * p) * K + k]) + ((uint32_t)q_narrow(A[(size_t)(2 * p + 1) * K + k]) << 16));
  for (int k = 0; k < K; k++)
    for (int j = 0; j < N; j++) Bi[(size_t)k * N + j] = q_narrow(B[(size_t)k * N + j]);
  int *dA = to_dev(Aw), *dB = to_dev(Bi);
  long long* dC;
  CK(cudaMalloc(&dC, sizeof(long long) * M * N));
  constexpr int TP = 4, TN = 8;
  dim3 grid(N / (16 * TN), (M / 2) / (16 * TP));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_swar2<TP, TN>, 256, 0));
  auto launch = [&](BlockClk* rec) { k_swar2<TP, TN><<<grid, 256>>>(dA, dB, dC, M, N, K, rec); };
  Meas m = timed(launch, grid.x * grid.y, reps);
  std::vector<long long> C((size_t)M * N);
  CK(cudaMemcpy(C.data(), dC, sizeof(long long) * M * N, cudaMemcpyDeviceToHost));
  long long checked;
  long long bad = check(C, A, B, M, N, K, samples, &checked);
  emit("imad_swar2", "narrow", M, N, K, m, occ, checked, bad, ",\"imad_per_mac\":0.5");
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
}

static void run_lut(int M, int N, int K, int reps, int samples) {
  std::vector<uint8_t> A, B;
  gen_codes(A, (size_t)M * K, false, 7 + M + K);
  gen_codes(B, (size_t)K * N, false, 8 + N + K);
  std::vector<int> Bv((size_t)K * N);
  for (size_t x = 0; x < Bv.size(); x++) Bv[x] = (int)e4m3_v(B[x]);
  std::vector<unsigned short> A2((size_t)(K / 2) * M);
  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k += 2)
      A2[(size_t)(k / 2) * M + i] = (unsigned short)(A[(size_t)i * K + k] | (A[(size_t)i * K + k + 1] << 8));
  unsigned short* dA = to_dev(A2);
  int* dB = to_dev(Bv);
  long long* dC;
  CK(cudaMalloc(&dC, sizeof(long long) * M * N));
  constexpr int RM = 4;
  dim3 grid(N / 4, M / (512 * RM));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_lut<RM>, 512, 0));
  auto launch = [&](BlockClk* rec) { k_lut<RM><<<grid, 512>>>(dA, dB, dC, M, N, K, rec); };
  Meas m = timed(launch, grid.x * grid.y, reps);
  std::vector<long long> C((size_t)M * N);
  CK(cudaMemcpy(C.data(), dC, sizeof(long long) * M * N, cudaMemcpyDeviceToHost));
  long long checked;
  long long bad = check(C, A, B, M, N, K, samples, &checked);
  emit("smem_lut", "generic", M, N, K, m, occ, checked, bad, ",\"lds128_per_mac\":0.5");
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
}

static void run_lut8(int M, int N, int K, int reps, int samples) {
  std::vector<uint8_t> A, B;
  gen_codes(A, (size_t)M * K, false, 17 + M + K);
  gen_codes(B, (size_t)K * N, false, 18 + N + K);
  std::vector<int> Bv((size_t)K * N);
  for (size_t x = 0; x < Bv.size(); x++) Bv[x] = (int)e4m3_v(B[x]);
  std::vector<uint2> A8((size_t)(K / 8) * M);
  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k += 8) {
      uint32_t w[2] = {0, 0};
      for (int b = 0; b < 8; b++) w[b / 4] |= (uint32_t)A[(size_t)i * K + k + b] << (8 * (b % 4));
      A8[(size_t)(k / 8) * M + i] = make_uint2(w[0], w[1]);
    }
  uint2* dA = to_dev(A8);
  int* dB = to_dev(Bv);
  long long* dC;
  CK(cudaMalloc(&dC, sizeof(long long) * M * N));
  constexpr int RM = 8;
  const int smem = 2 * 8 * 256 * sizeof(int4);
  CK(cudaFuncSetAttribute(k_lut8<RM>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(N / 2, M / (512 * RM));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_lut8<RM>, 512, smem));
  auto launch = [&](BlockClk* rec) { k_lut8<RM><<<grid, 512, smem>>>(dA, dB, dC, M, N, K, rec); };
  Meas m = timed(launch, grid.x * grid.y, reps);
  std::vector<long long> C((size_t)M * N);
  CK(cudaMemcpy(C.data(), dC, sizeof(long long) * M * N, cudaMemcpyDeviceToHost));
  long long checked;
  long long bad = check(C, A, B, M, N, K, samples, &checked);
  emit("smem_lut8", "generic", M, N, K, m, occ, checked, bad, ",\"lds128_per_mac\":0.5");
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
}

template <int ND, int NI, int NF, int NL, int NF2>
static void run_probe(const char* name, int iters) {
  int* out;
  CK(cudaMalloc(&out, sizeof(int)));
  auto launch = [&](BlockClk* rec) { k_probe<ND, NI, NF, NL, NF2><<<g_nsm, 1024>>>(out, iters, rec); };
  Meas m = timed(launch, g_nsm, 5);
  const double steps = 32.0 * 8 * iters / m.cyc_med;  // chain steps a SM a clock (warp-wide)
  const double total = steps * (ND + NI + NF + NL + NF2);
  const double macs = steps * 32 * (4.0 * ND + 2.0 * NI + 1.0 * NF + 2.0 * NF2);
  printf(
      "{\"kind\":\"probe\",\"ops\":\"%s\",\"mix\":[%d,%d,%d,%d,%d],\"warp_instr_per_sm_clk\":%.3f,"
      "\"dp4a\":%.3f,\"imad\":%.3f,\"ffma\":%.3f,\"lop3\":%.3f,\"ffma2\":%.3f,\"sm_mhz\":%.1f,"
      "\"mac_equiv_per_sm_clk\":%.1f}\n",
      name, ND, NI, NF, NL, NF2, total, steps * ND, steps * NI, steps * NF, steps * NL, steps * NF2, m.mhz, macs);
  fflush(stdout);
  CK(cudaFree(out));
}

template <int ND, int NF, int NF2>
static void run_split(const char* name, int iters) {
  int* out;
  long long* wc;
  CK(cudaMalloc(&out, sizeof(int)));
  CK(cudaMalloc(&wc, sizeof(long long) * 32 * g_nsm));
  auto launch = [&](BlockClk* rec) { k_split<ND, NF, NF2><<<g_nsm, 1024>>>(out, iters, wc, rec); };
  Meas m = timed(launch, g_nsm, 5);
  std::vector<long long> h(32 * g_nsm);
  CK(cudaMemcpy(h.data(), wc, sizeof(long long) * h.size(), cudaMemcpyDeviceToHost));
  std::vector<double> td, tf;
  for (int b = 0; b < g_nsm; b++) {
    long long a = 0, c = 0;
    for (int w = 0; w < 32; w++) (w & 4 ? c : a) = std::max(w & 4 ? c : a, h[b * 32 + w]);
    td.push_back((double)a), tf.push_back((double)c);
  }
  std::sort(td.begin(), td.end());
  std::sort(tf.begin(), tf.end());
  const double Td = td[g_nsm / 2], Tf = tf[g_nsm / 2], T = m.cyc_med, steps = 16.0 * 8 * iters;
  const double dp4a = steps * ND / Td, ffma = steps * NF / Tf, ffma2 = steps * NF2 / Tf;
  const double macs = steps * 32 * (4.0 * ND + 1.0 * NF + 2.0 * NF2) / T;
  printf(
      "{\"kind\":\"split\",\"ops\":\"%s\",\"mix\":[%d,%d,%d],\"dp4a\":%.3f,\"ffma\":%.3f,\"ffma2\":%.3f,"
      "\"dp4a_cycles\":%.0f,\"fma_cycles\":%.0f,\"block_cycles\":%.0f,\"sm_mhz\":%.1f,\"mac_equiv_per_sm_clk\":%.1f}\n",
      name, ND, NF, NF2, dp4a, ffma, ffma2, Td, Tf, T, m.mhz, macs);
  fflush(stdout);
  CK(cudaFree(out));
  CK(cudaFree(wc));
}

int main(int argc, char** argv) {
  std::string what = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_nsm = p.multiProcessorCount;
  printf("{\"kind\":\"device\",\"name\":\"%s\",\"sms\":%d,\"cc\":\"%d.%d\"}\n", p.name, g_nsm, p.major, p.minor);
  if (!decoder_selftest()) return 3;
  const bool all = what == "all";
  const int Q = g_nsm / 4;  // 47 on the RTX PRO 6000: grids are multiples of the SM count
  if (all || what == "small") {  // every output checked
    run_gemm<0, 8, 8, 3, 2>(256, 256, 512, 1, 0, 0);
    run_gemm<1, 4, 2, 3, 1>(128, 128, 512, 1, 0, 0);
    run_gemm<2, 8, 8, 3, 2>(256, 256, 2048, 1, 0, 0);
    run_swar(256, 256, 512, 1, 0);
    run_lut(2048, 16, 256, 1, 0);
    run_lut8(4096, 4, 256, 1, 0);
  }
  if (all || what == "probe") {
    run_probe<1, 0, 0, 0, 0>("dp4a", 4096);
    run_probe<0, 1, 0, 0, 0>("imad", 4096);
    run_probe<0, 0, 1, 0, 0>("ffma", 4096);
    run_probe<0, 0, 0, 1, 0>("lop3", 4096);
    run_probe<0, 0, 0, 0, 1>("ffma2", 4096);
    run_probe<1, 1, 0, 0, 0>("dp4a+imad", 4096);
    run_probe<1, 0, 1, 0, 0>("dp4a+ffma", 4096);
    run_probe<2, 0, 1, 0, 0>("2dp4a+ffma", 4096);
    run_probe<1, 0, 2, 0, 0>("dp4a+2ffma", 4096);
    run_probe<1, 0, 0, 0, 1>("dp4a+ffma2", 4096);
    run_probe<2, 0, 0, 0, 1>("2dp4a+ffma2", 4096);
    run_probe<1, 0, 0, 1, 0>("dp4a+lop3", 4096);
    run_probe<2, 0, 1, 1, 0>("2dp4a+ffma+lop3", 4096);
    run_probe<0, 0, 1, 0, 1>("ffma+ffma2", 4096);
    run_probe<0, 1, 1, 0, 0>("imad+ffma", 4096);
    run_split<1, 0, 0>("split dp4a | idle", 4096);
    run_split<1, 1, 0>("split dp4a | ffma", 4096);
    run_split<1, 2, 0>("split dp4a | 2ffma", 4096);
    run_split<2, 1, 0>("split 2dp4a | ffma", 4096);
    run_split<1, 0, 1>("split dp4a | ffma2", 4096);
    run_split<2, 0, 1>("split 2dp4a | ffma2", 4096);
    run_split<4, 0, 1>("split 4dp4a | ffma2", 4096);
  }
  if (all || what == "big") {  // timed; 4096 sampled outputs checked
    run_gemm<0, 8, 8, 3, 2>(128 * Q, 128 * 8, 8192, 9, 4096, 0);
    run_gemm<0, 8, 8, 3, 1>(128 * Q, 128 * 8, 8192, 9, 4096, 0);
    run_gemm<0, 8, 4, 3, 2>(128 * Q, 128 * 8, 8192, 9, 4096, 0);
    run_gemm<0, 8, 4, 4, 2>(128 * Q, 128 * 8, 8192, 9, 4096, 0);
    run_gemm<0, 8, 2, 4, 2>(128 * Q, 128 * 8, 8192, 9, 4096, 0);
    run_gemm<1, 4, 4, 3, 1>(64 * Q * 2, 64 * 8, 4096, 9, 4096, 0);
    run_gemm<1, 4, 2, 3, 1>(64 * Q * 2, 64 * 8, 4096, 9, 4096, 0);
    run_gemm<1, 4, 1, 4, 1>(64 * Q * 2, 64 * 8, 4096, 9, 4096, 0);
    run_gemm<1, 4, 2, 3, 2>(64 * Q * 2, 64 * 8, 4096, 9, 4096, 0);
    run_gemm<2, 8, 8, 3, 2>(128 * Q, 128 * 8, 4096, 9, 4096, 0);
    run_gemm<2, 8, 8, 3, 1>(128 * Q, 128 * 8, 4096, 9, 4096, 0);
    run_gemm<2, 8, 4, 3, 2>(128 * Q, 128 * 8, 4096, 9, 4096, 0);
    run_gemm<2, 8, 2, 4, 2>(128 * Q, 128 * 8, 4096, 9, 4096, 0);
    run_swar(128 * Q, 128 * 8, 4096, 9, 4096);
    run_lut(2048 * 8, 8 * Q * 2, 1024, 5, 4096);
    run_lut8(4096 * 4, 2 * Q * 4, 1024, 5, 4096);
  }
  if (all || what == "corun") run_corun(9, 4096);
  return 0;
}
