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

// dp4a over L packed int8 limbs: A[(l*KW + kw)*M + i] holds 4 consecutive k's of limb l as bytes.
// L = 1: narrow inputs, one int8 per value, C = 64 * P.
// L = 6: generic inputs, 3 signed 6-bit digits per value, Karatsuba's 6 limb pairs (d0, d1, d2, d0+d1, d1+d2, d0+d2).
template <int TM, int TN, int L>
__global__ void __launch_bounds__(256) k_dp4a(const int* __restrict__ A, const int* __restrict__ B,
                                              long long* __restrict__ C, int M, int N, int KW, BlockClk* rec) {
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int row0 = blockIdx.y * (16 * TM) + ty * TM, col0 = blockIdx.x * (16 * TN) + tx * TN;
  int acc[L][TM][TN];
#pragma unroll
  for (int l = 0; l < L; l++)
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++) acc[l][i][j] = 0;
  CLK_BEGIN
#pragma unroll 2
  for (int kw = 0; kw < KW; kw++) {
#pragma unroll
    for (int l = 0; l < L; l++) {
      int a[TM], b[TN];
      const int4* pa = reinterpret_cast<const int4*>(A + ((size_t)l * KW + kw) * M + row0);
      const int4* pb = reinterpret_cast<const int4*>(B + ((size_t)l * KW + kw) * N + col0);
#pragma unroll
      for (int v = 0; v < TM / 4; v++) {
        int4 t = __ldg(pa + v);
        a[4 * v] = t.x, a[4 * v + 1] = t.y, a[4 * v + 2] = t.z, a[4 * v + 3] = t.w;
      }
#pragma unroll
      for (int v = 0; v < TN / 4; v++) {
        int4 t = __ldg(pb + v);
        b[4 * v] = t.x, b[4 * v + 1] = t.y, b[4 * v + 2] = t.z, b[4 * v + 3] = t.w;
      }
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[l][i][j] = __dp4a(a[i], b[j], acc[l][i][j]);
    }
  }
  CLK_END
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) {
      long long c;
      if constexpr (L == 1) {
        c = 64ll * acc[0][i][j];
      } else {
        long long p0 = acc[0][i][j], p1 = acc[1][i][j], p2 = acc[2][i][j];
        long long p01 = acc[3][i][j], p12 = acc[4][i][j], p02 = acc[5][i][j];
        c = p0 + ((p01 - p0 - p1) << 6) + ((p02 - p0 - p2 + p1) << 12) + ((p12 - p1 - p2) << 18) + (p2 << 24);
      }
      C[(size_t)(row0 + i) * N + col0 + j] = c;
    }
}

// FP32 FFMA on narrow inputs (q as float): products exact, sums exact while below 2^24, so accumulators flush to the
// int64 output every FLUSH k's (14400 * 1024 < 2^24). C must be zeroed first.
template <int TM, int TN>
__global__ void __launch_bounds__(256) k_ffma(const float* __restrict__ A, const float* __restrict__ B,
                                              long long* __restrict__ C, int M, int N, int K, BlockClk* rec) {
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int row0 = blockIdx.y * (16 * TM) + ty * TM, col0 = blockIdx.x * (16 * TN) + tx * TN;
  constexpr int FLUSH = 1024;
  float acc[TM][TN];
  long long tot[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) tot[i][j] = 0;
  CLK_BEGIN
  for (int k0 = 0; k0 < K; k0 += FLUSH) {
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
#pragma unroll 4
    for (int k = k0; k < k0 + FLUSH; k++) {
      float a[TM], b[TN];
      const float4* pa = reinterpret_cast<const float4*>(A + (size_t)k * M + row0);
      const float4* pb = reinterpret_cast<const float4*>(B + (size_t)k * N + col0);
#pragma unroll
      for (int v = 0; v < TM / 4; v++) {
        float4 t = __ldg(pa + v);
        a[4 * v] = t.x, a[4 * v + 1] = t.y, a[4 * v + 2] = t.z, a[4 * v + 3] = t.w;
      }
#pragma unroll
      for (int v = 0; v < TN / 4; v++) {
        float4 t = __ldg(pb + v);
        b[4 * v] = t.x, b[4 * v + 1] = t.y, b[4 * v + 2] = t.z, b[4 * v + 3] = t.w;
      }
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
    }
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++) tot[i][j] += (long long)acc[i][j];
  }
  CLK_END
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) C[(size_t)(row0 + i) * N + col0 + j] = 64ll * tot[i][j];
}

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

// dp4a and FFMA together in the same warps, narrow inputs: each thread owns a TMD x TND dp4a tile of Cd and a
// TMF x TNF FFMA tile of Cf over the same K (A8/B8 packed as k_dp4a's L = 1; Af/Bf as k_ffma's). K <= 1024 keeps the
// FP32 sums exact without a flush.
template <int TMD, int TND, int TMF, int TNF>
__global__ void __launch_bounds__(256) k_combo(const int* __restrict__ A8, const int* __restrict__ B8,
                                               const float* __restrict__ Af, const float* __restrict__ Bf,
                                               long long* __restrict__ Cd, long long* __restrict__ Cf, int Md,
                                               int Nd, int Mf, int Nf, int K, BlockClk* rec) {
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int rd0 = blockIdx.y * (16 * TMD) + ty * TMD, cd0 = blockIdx.x * (16 * TND) + tx * TND;
  const int rf0 = blockIdx.y * (16 * TMF) + ty * TMF, cf0 = blockIdx.x * (16 * TNF) + tx * TNF;
  int ad[TMD][TND];
  float af[TMF][TNF];
#pragma unroll
  for (int i = 0; i < TMD; i++)
#pragma unroll
    for (int j = 0; j < TND; j++) ad[i][j] = 0;
#pragma unroll
  for (int i = 0; i < TMF; i++)
#pragma unroll
    for (int j = 0; j < TNF; j++) af[i][j] = 0.f;
  CLK_BEGIN
  const int KW = K / 4;
#pragma unroll 2
  for (int kw = 0; kw < KW; kw++) {
    int a[TMD], b[TND];
#pragma unroll
    for (int v = 0; v < TMD / 4; v++) {
      int4 t = __ldg(reinterpret_cast<const int4*>(A8 + (size_t)kw * Md + rd0) + v);
      a[4 * v] = t.x, a[4 * v + 1] = t.y, a[4 * v + 2] = t.z, a[4 * v + 3] = t.w;
    }
#pragma unroll
    for (int v = 0; v < TND / 4; v++) {
      int4 t = __ldg(reinterpret_cast<const int4*>(B8 + (size_t)kw * Nd + cd0) + v);
      b[4 * v] = t.x, b[4 * v + 1] = t.y, b[4 * v + 2] = t.z, b[4 * v + 3] = t.w;
    }
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      float fa[TMF], fb[TNF];
#pragma unroll
      for (int v = 0; v < TMF / 4; v++) {
        float4 t = __ldg(reinterpret_cast<const float4*>(Af + (size_t)(4 * kw + kk) * Mf + rf0) + v);
        fa[4 * v] = t.x, fa[4 * v + 1] = t.y, fa[4 * v + 2] = t.z, fa[4 * v + 3] = t.w;
      }
#pragma unroll
      for (int v = 0; v < TNF / 4; v++) {
        float4 t = __ldg(reinterpret_cast<const float4*>(Bf + (size_t)(4 * kw + kk) * Nf + cf0) + v);
        fb[4 * v] = t.x, fb[4 * v + 1] = t.y, fb[4 * v + 2] = t.z, fb[4 * v + 3] = t.w;
      }
#pragma unroll
      for (int i = 0; i < TMF; i++)
#pragma unroll
        for (int j = 0; j < TNF; j++) af[i][j] = __fmaf_rn(fa[i], fb[j], af[i][j]);
    }
#pragma unroll
    for (int i = 0; i < TMD; i++)
#pragma unroll
      for (int j = 0; j < TND; j++) ad[i][j] = __dp4a(a[i], b[j], ad[i][j]);
  }
  CLK_END
#pragma unroll
  for (int i = 0; i < TMD; i++)
#pragma unroll
    for (int j = 0; j < TND; j++) Cd[(size_t)(rd0 + i) * Nd + cd0 + j] = 64ll * ad[i][j];
#pragma unroll
  for (int i = 0; i < TMF; i++)
#pragma unroll
    for (int j = 0; j < TNF; j++) Cf[(size_t)(rf0 + i) * Nf + cf0 + j] = 64ll * (long long)af[i][j];
}

// ---------------------------------------------------------------- pipe-pairing probe
// 32 warps a SM, one block a SM, 8 independent chains per op per thread; immediates keep FFMA off the register-bank
// limit. MODE bits: 1 dp4a, 2 IMAD, 4 FFMA, 8 LOP3.
template <int MODE>
__global__ void __launch_bounds__(1024) k_probe(int* out, int iters, BlockClk* rec) {
  int d[8], m[8], l[8];
  float f[8];
#pragma unroll
  for (int r = 0; r < 8; r++) d[r] = m[r] = l[r] = threadIdx.x + r, f[r] = (float)(threadIdx.x + r);
  const int x = 0x01020304 + blockIdx.x, y = 0x05060708 ^ threadIdx.x;
  __syncthreads();
  CLK_BEGIN
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int r = 0; r < 8; r++) {
      if (MODE & 1) asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(d[r]) : "r"(x), "r"(y));
      if (MODE & 2) asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(m[r]) : "r"(x), "r"(y));
      if (MODE & 4) asm volatile("fma.rn.f32 %0, %0, 0f3F800100, 0f3F000000;" : "+f"(f[r]));
      if (MODE & 8) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(l[r]) : "r"(x), "r"(y));
    }
  }
  __syncthreads();
  CLK_END
  int s = 0;
#pragma unroll
  for (int r = 0; r < 8; r++) s += d[r] + m[r] + l[r] + __float_as_int(f[r]);
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

static void run_dp4a(bool generic, int M, int N, int K, int reps, int samples) {
  std::vector<uint8_t> A, B;
  gen_codes(A, (size_t)M * K, !generic, 1 + M + K);
  gen_codes(B, (size_t)K * N, !generic, 2 + N + K);
  const int L = generic ? 6 : 1;
  auto Ap = pack_int8(A, M, K, true, L, 0), Bp = pack_int8(B, N, K, false, L, 0);
  int *dA = to_dev(Ap), *dB = to_dev(Bp);
  long long* dC;
  CK(cudaMalloc(&dC, sizeof(long long) * M * N));
  constexpr int TMn = 8, TNn = 8, TMg = 4, TNg = 4;
  const int TM = generic ? TMg : TMn, TN = generic ? TNg : TNn;
  dim3 grid(N / (16 * TN), M / (16 * TM));
  int occ = 0;
  if (generic) {
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_dp4a<TMg, TNg, 6>, 256, 0));
  } else {
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_dp4a<TMn, TNn, 1>, 256, 0));
  }
  auto launch = [&](BlockClk* rec) {
    if (generic) k_dp4a<TMg, TNg, 6><<<grid, 256>>>(dA, dB, dC, M, N, K / 4, rec);
    else k_dp4a<TMn, TNn, 1><<<grid, 256>>>(dA, dB, dC, M, N, K / 4, rec);
  };
  Meas m = timed(launch, grid.x * grid.y, reps);
  std::vector<long long> C((size_t)M * N);
  CK(cudaMemcpy(C.data(), dC, sizeof(long long) * M * N, cudaMemcpyDeviceToHost));
  long long checked;
  long long bad = check(C, A, B, M, N, K, samples, &checked);
  emit(generic ? "dp4a_karatsuba3" : "dp4a_int8", generic ? "generic" : "narrow", M, N, K, m, occ, checked, bad,
       generic ? ",\"dp4a_per_mac\":1.5" : ",\"dp4a_per_mac\":0.25");
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
}

static void run_ffma(int M, int N, int K, int reps, int samples) {
  std::vector<uint8_t> A, B;
  gen_codes(A, (size_t)M * K, true, 3 + M + K);
  gen_codes(B, (size_t)K * N, true, 4 + N + K);
  std::vector<float> Af((size_t)K * M), Bf((size_t)K * N);
  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++) Af[(size_t)k * M + i] = (float)q_narrow(A[(size_t)i * K + k]);
  for (int k = 0; k < K; k++)
    for (int j = 0; j < N; j++) Bf[(size_t)k * N + j] = (float)q_narrow(B[(size_t)k * N + j]);
  float *dA = to_dev(Af), *dB = to_dev(Bf);
  long long* dC;
  CK(cudaMalloc(&dC, sizeof(long long) * M * N));
  constexpr int TM = 8, TN = 8;
  dim3 grid(N / (16 * TN), M / (16 * TM));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_ffma<TM, TN>, 256, 0));
  auto launch = [&](BlockClk* rec) { k_ffma<TM, TN><<<grid, 256>>>(dA, dB, dC, M, N, K, rec); };
  Meas m = timed(launch, grid.x * grid.y, reps);
  std::vector<long long> C((size_t)M * N);
  CK(cudaMemcpy(C.data(), dC, sizeof(long long) * M * N, cudaMemcpyDeviceToHost));
  long long checked;
  long long bad = check(C, A, B, M, N, K, samples, &checked);
  emit("ffma_fp32", "narrow", M, N, K, m, occ, checked, bad);
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
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

template <int TMD, int TND, int TMF, int TNF>
static void run_combo(int gx, int gy, int K, int reps, int samples) {
  const int Md = gy * 16 * TMD, Nd = gx * 16 * TND, Mf = gy * 16 * TMF, Nf = gx * 16 * TNF;
  std::vector<uint8_t> Ad, Bd, Af8, Bf8;
  gen_codes(Ad, (size_t)Md * K, true, 9 + K);
  gen_codes(Bd, (size_t)K * Nd, true, 10 + K);
  gen_codes(Af8, (size_t)Mf * K, true, 11 + K);
  gen_codes(Bf8, (size_t)K * Nf, true, 12 + K);
  auto Ap = pack_int8(Ad, Md, K, true, 1, 0), Bp = pack_int8(Bd, Nd, K, false, 1, 0);
  std::vector<float> Af((size_t)K * Mf), Bf((size_t)K * Nf);
  for (int i = 0; i < Mf; i++)
    for (int k = 0; k < K; k++) Af[(size_t)k * Mf + i] = (float)q_narrow(Af8[(size_t)i * K + k]);
  for (int k = 0; k < K; k++)
    for (int j = 0; j < Nf; j++) Bf[(size_t)k * Nf + j] = (float)q_narrow(Bf8[(size_t)k * Nf + j]);
  int *dA8 = to_dev(Ap), *dB8 = to_dev(Bp);
  float *dAf = to_dev(Af), *dBf = to_dev(Bf);
  long long *dCd, *dCf;
  CK(cudaMalloc(&dCd, sizeof(long long) * Md * Nd));
  CK(cudaMalloc(&dCf, sizeof(long long) * Mf * Nf));
  dim3 grid(gx, gy);
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_combo<TMD, TND, TMF, TNF>, 256, 0));
  auto launch = [&](BlockClk* rec) {
    k_combo<TMD, TND, TMF, TNF><<<grid, 256>>>(dA8, dB8, dAf, dBf, dCd, dCf, Md, Nd, Mf, Nf, K, rec);
  };
  Meas m = timed(launch, gx * gy, reps);
  std::vector<long long> Cd((size_t)Md * Nd), Cf((size_t)Mf * Nf);
  CK(cudaMemcpy(Cd.data(), dCd, sizeof(long long) * Md * Nd, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(Cf.data(), dCf, sizeof(long long) * Mf * Nf, cudaMemcpyDeviceToHost));
  long long c1, c2;
  long long bad = check(Cd, Ad, Bd, Md, Nd, K, samples, &c1) + check(Cf, Af8, Bf8, Mf, Nf, K, samples, &c2);
  double macs_d = (double)Md * Nd * K, macs_f = (double)Mf * Nf * K;
  double clk = g_nsm * m.ms * 1e-3 * m.mhz * 1e6;
  char extra[256];
  snprintf(extra, sizeof extra, ",\"tiles\":\"dp4a %dx%d + ffma %dx%d\",\"dp4a_macs_per_sm_clk\":%.2f,\"ffma_macs_per_sm_clk\":%.2f",
           TMD, TND, TMF, TNF, macs_d / clk, macs_f / clk);
  // emit() takes M*N*K as the MAC count: pass the sum as one M x 1 x K.
  int Meq = (int)((macs_d + macs_f) / K);
  emit("combo_dp4a_ffma", "narrow", Meq, 1, K, m, occ, c1 + c2, bad, extra);
  CK(cudaFree(dA8));
  CK(cudaFree(dB8));
  CK(cudaFree(dAf));
  CK(cudaFree(dBf));
  CK(cudaFree(dCd));
  CK(cudaFree(dCf));
}

template <int MODE>
static void run_probe(const char* name, int iters) {
  int* out;
  CK(cudaMalloc(&out, sizeof(int)));
  auto launch = [&](BlockClk* rec) { k_probe<MODE><<<g_nsm, 1024>>>(out, iters, rec); };
  Meas m = timed(launch, g_nsm, 5);
  int ops = ((MODE & 1) != 0) + ((MODE & 2) != 0) + ((MODE & 4) != 0) + ((MODE & 8) != 0);
  double warp_instr = 32.0 * 8 * iters * ops;  // per SM: 32 warps x 8 chains x iters x ops
  double per_clk = warp_instr / m.cyc_med;
  double per_op = per_clk / ops;
  double mac_lanes = 32.0 * per_op * (4.0 * ((MODE & 1) != 0) + 2.0 * ((MODE & 2) != 0) + 1.0 * ((MODE & 4) != 0));
  printf(
      "{\"kind\":\"probe\",\"ops\":\"%s\",\"warp_instr_per_sm_clk\":%.3f,\"per_op\":%.3f,\"sm_mhz\":%.1f,"
      "\"mac_equiv_per_sm_clk\":%.1f}\n",
      name, per_clk, per_op, m.mhz, mac_lanes);
  fflush(stdout);
  CK(cudaFree(out));
}

int main(int argc, char** argv) {
  std::string what = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_nsm = p.multiProcessorCount;
  printf("{\"kind\":\"device\",\"name\":\"%s\",\"sms\":%d,\"cc\":\"%d.%d\"}\n", p.name, g_nsm, p.major, p.minor);
  if (!decoder_selftest()) return 3;
  const bool all = what == "all";
  const int S = g_nsm;  // grid sizes are multiples of the SM count
  // small instances: every output checked
  if (all || what == "small") {
    run_dp4a(false, 256, 256, 512, 1, 0);
    run_dp4a(true, 128, 128, 512, 1, 0);
    run_ffma(256, 256, 1024, 1, 0);
    run_swar(256, 256, 512, 1, 0);
    run_lut(2048, 16, 256, 1, 0);
    run_combo<8, 8, 4, 4>(2, 2, 512, 1, 0);
  }
  if (all || what == "probe") {
    run_probe<1>("dp4a", 4096);
    run_probe<2>("imad", 4096);
    run_probe<4>("ffma_imm", 4096);
    run_probe<8>("lop3", 4096);
    run_probe<1 | 2>("dp4a+imad", 4096);
    run_probe<1 | 4>("dp4a+ffma_imm", 4096);
    run_probe<1 | 8>("dp4a+lop3", 4096);
    run_probe<1 | 4 | 8>("dp4a+ffma_imm+lop3", 4096);
    run_probe<2 | 4>("imad+ffma_imm", 4096);
  }
  // timed instances: grids of 2 x SMs blocks or more; 4096 sampled outputs checked
  if (all || what == "big") {
    run_dp4a(false, 128 * (S / 4), 128 * 8, 8192, 9, 4096);    // 47 x 8 = 376 blocks
    run_dp4a(true, 64 * (S / 4), 64 * 8, 4096, 9, 4096);       // 376 blocks
    run_ffma(128 * (S / 4), 128 * 8, 4096, 9, 4096);
    run_swar(128 * (S / 4), 128 * 8, 4096, 9, 4096);           // pairs: 64 per block row
    run_lut(2048 * 8, 8 * (S / 4) * 2, 1024, 5, 4096);         // 94 x 8 = 752 blocks
    run_combo<8, 8, 4, 4>(8, S / 4, 1024, 9, 4096);
    run_combo<8, 8, 2, 4>(8, S / 4, 1024, 9, 4096);
    run_combo<8, 8, 4, 8>(8, S / 4, 1024, 9, 4096);
  }
  return 0;
}
