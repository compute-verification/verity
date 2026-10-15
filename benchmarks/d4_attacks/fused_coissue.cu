// The co-issue hybrid in one kernel, on sm_120: each CTA computes a 128×128 tile of the exact integer product
// D = A·Bᵀ (entries of A and B i.i.d. uniform on a small alphabet S), with warps 0–7 running an int8 tensor-core
// mainloop (mma.sync m16n8k32 s8, fragments loaded straight from pre-swizzled operands, no shared memory) over
// k in [0, k1) and warps 8–15 of the same CTA running, at the same time, CUDA-core work over k in [k1, n):
//   ROLE_FR    Four Russians: per chunk of b consecutive k, a table T[p][c] = sum_t S[p_t]·B[c][base+t] for every
//              pattern p in S^b and the tile's 128 columns, int16 entries biased to be non-negative and packed two
//              columns a word, so one conflict-free LDS.32 a lane gathers two entries; the pattern of each row is
//              computed once into shared memory and read as a warp broadcast.
//   ROLE_DP4A  fixed-address CUDA-core MACs: DP4A on int8 words (the CUDA-core share of B's R* mix).
//   ROLE_MIX   half the k range on DP4A, half on HFMA2 (FP16 products, exact while |partial| <= 2048, flushed to
//              int32 every 16 k), the other half of the mix.
// The two partials are summed through shared memory in the epilogue. ROLE_NONE launches warps 0–7 only (our own
// tensor-only kernel). Calibrates NoLookupGain(S, n, η) (note:20261015T1814Z-draft-rs-machine-model §6.3, item 2).
//
// Gates before timing, in the same process: cuBLASLt int8 against a CPU int64 reference (every entry at n <= 1024,
// 4098 sampled entries otherwise), then every route against it bit for bit, outputs poisoned with 0xA5 first; a
// control (int8 with one entry of A changed) must be rejected. Timing: interleaved rounds of ~25 ms items with the SM
// clock read by NVML before and after each. JSON lines on stdout.
//
//   fused_coissue <n> <alphabet> <seed> <rounds> <alphas64>     alphabet: k2 k4 k6 k8 u6
#ifndef __CUDACC_RTC__
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#else
#include <cuda_fp16.h>
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif

struct Vals {
  int v[17];
};

__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__global__ void k_gen(uint8_t* I, int8_t* V, size_t count, uint64_t seed, int m, Vals s) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t d = uint32_t(mix64(seed ^ (i * 0xD1B54A32D192ED03ull)) % uint64_t(m));
    I[i] = uint8_t(d);
    V[i] = int8_t(s.v[d]);
  }
}

// mma.m16n8k32 s8 fragments in lane order: A_sw[(mt·KT + kt)·32 + lane] holds a0..a3 of the 16×32 tile (mt, kt),
// B_sw[(nt·KT + kt)·32 + lane] holds b0, b1 of the 32×8 tile (kt, nt), with B's columns our B's rows.
__global__ void k_swz_a(const int8_t* __restrict__ A, uint4* __restrict__ Asw, int n) {
  const int KT = n / 32;
  const size_t total = size_t(n / 16) * KT * 32;
  for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < total; e += size_t(gridDim.x) * blockDim.x) {
    const int lane = int(e & 31), kt = int((e >> 5) % KT), mt = int((e >> 5) / KT), g = lane >> 2, t = lane & 3;
    const int8_t* p0 = A + size_t(mt * 16 + g) * n + kt * 32 + t * 4;
    const int8_t* p1 = p0 + size_t(8) * n;
    Asw[e] = make_uint4(*reinterpret_cast<const uint32_t*>(p0), *reinterpret_cast<const uint32_t*>(p1),
                        *reinterpret_cast<const uint32_t*>(p0 + 16), *reinterpret_cast<const uint32_t*>(p1 + 16));
  }
}
__global__ void k_swz_b(const int8_t* __restrict__ B, uint2* __restrict__ Bsw, int n) {
  const int KT = n / 32;
  const size_t total = size_t(n / 8) * KT * 32;
  for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < total; e += size_t(gridDim.x) * blockDim.x) {
    const int lane = int(e & 31), kt = int((e >> 5) % KT), nt = int((e >> 5) / KT), g = lane >> 2, t = lane & 3;
    const int8_t* p = B + size_t(nt * 8 + g) * n + kt * 32 + t * 4;
    Bsw[e] = make_uint2(*reinterpret_cast<const uint32_t*>(p), *reinterpret_cast<const uint32_t*>(p + 16));
  }
}
// Bh[k][c] = B[c][k] as FP16 (transposed so a lane's two columns at one k are one half2); Ah = A as FP16.
__global__ void k_half_t(const int8_t* __restrict__ B, __half* __restrict__ Bh, int n) {
  for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < size_t(n) * n; e += size_t(gridDim.x) * blockDim.x) {
    const size_t k = e / n, c = e % n;
    Bh[e] = __int2half_rn(B[c * n + k]);
  }
}
__global__ void k_half(const int8_t* __restrict__ A, __half* __restrict__ Ah, size_t count) {
  for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < count; e += size_t(gridDim.x) * blockDim.x)
    Ah[e] = __int2half_rn(A[e]);
}

__device__ __forceinline__ void mma_s8(int* c, const uint4& a, const uint2& b) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ void bar_cuda_warps() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }

enum { ROLE_NONE = 0, ROLE_FR = 1, ROLE_DP4A = 2, ROLE_MIX = 3 };
constexpr int RSTRIDE = 132;  // epilogue partials, ints a row

struct FusedArgs {
  const uint4* Asw;
  const uint2* Bsw;
  const uint8_t* AI;
  const uint8_t* BI;
  const int8_t* A;
  const int8_t* B;
  const __half* Ah;
  const __half* Bh;
  int* D;
  int n, k1, b, m, mp, G, bias;
  Vals s;
};

// Four Russians over k in [k1, n) for the tile (r0, c0): lane holds rows rq·32 .. +32 at columns cg·64 + 2·lane, +1.
__device__ __forceinline__ void cuda_fr(const FusedArgs& f, unsigned char* smem, int r0, int c0, int lw, int lane,
                                        int* lo, int* hi) {
  const int n = f.n, b = f.b, m = f.m, mp = f.mp, G = f.G, ltid = lw * 32 + lane;
  const int cg = lw & 1, rq = lw >> 1;
  uint32_t* T = reinterpret_cast<uint32_t*>(smem);
  int* P = reinterpret_cast<int*>(smem + size_t(G) * mp * 64 * 4);
  for (int base0 = f.k1; base0 < n; base0 += G * b) {
    const int Gs = min(G, (n - base0 + b - 1) / b);
    for (int e = ltid; e < Gs * mp * 64; e += 256) {
      const int g = e / (mp * 64), rem = e - g * mp * 64, p = rem >> 6, wc = rem & 63;
      const int base = base0 + g * b, w = min(b, n - base);
      const uint8_t* b0 = f.BI + size_t(c0 + 2 * wc) * n + base;
      const uint8_t* b1 = b0 + n;
      int q = p, x0 = 0, x1 = 0;
      for (int t = 0; t < w; ++t) {
        const int sv = f.s.v[q % m];
        q /= m;
        x0 += sv * f.s.v[b0[t]];
        x1 += sv * f.s.v[b1[t]];
      }
      T[e] = uint32_t(x0 + f.bias) | (uint32_t(x1 + f.bias) << 16);
    }
    for (int e = ltid; e < Gs * 128; e += 256) {
      const int g = e >> 7, r = e & 127, base = base0 + g * b, w = min(b, n - base);
      const uint8_t* ar = f.AI + size_t(r0 + r) * n + base;
      int p = 0, pw = 1;
      for (int t = 0; t < w; ++t) {
        p += ar[t] * pw;
        pw *= m;
      }
      P[e] = (g * mp + p) * 64;
    }
    bar_cuda_warps();
    const uint32_t* Tl = T + cg * 32 + lane;
    const int sub = Gs * f.bias;
#pragma unroll
    for (int rr = 0; rr < 32; ++rr) {
      const int* Pr = P + rq * 32 + rr;
      uint32_t sum = 0;
#pragma unroll 4
      for (int g = 0; g < Gs; ++g) sum += Tl[Pr[g * 128]];
      lo[rr] += int(sum & 0xFFFF) - sub;
      hi[rr] += int(sum >> 16) - sub;
    }
    bar_cuda_warps();
  }
}

// DP4A over k in [ka, kb) (multiples of 16): lane rows rq·32 .. +32, columns cg·64 + 2·lane, +1. The tile's 128 A
// rows of each 16-k step are staged in shared memory, so a row is one broadcast LDS.128 at a constant offset.
__device__ __forceinline__ void cuda_dp4a(const FusedArgs& f, unsigned char* smem, int ka, int kb, int r0, int c0,
                                          int lw, int lane, int* lo, int* hi) {
  const int n = f.n, cg = lw & 1, rq = lw >> 1, c = c0 + cg * 64 + 2 * lane, ltid = lw * 32 + lane;
  uint4* As = reinterpret_cast<uint4*>(smem);
  for (int k = ka; k < kb; k += 16) {
    if (ltid < 128) As[ltid] = *reinterpret_cast<const uint4*>(f.A + size_t(r0 + ltid) * n + k);
    const uint4 bw0 = *reinterpret_cast<const uint4*>(f.B + size_t(c) * n + k);
    const uint4 bw1 = *reinterpret_cast<const uint4*>(f.B + size_t(c + 1) * n + k);
    bar_cuda_warps();
#pragma unroll
    for (int rr = 0; rr < 32; ++rr) {
      const uint4 aw = As[rq * 32 + rr];
      lo[rr] = __dp4a(int(aw.x), int(bw0.x), lo[rr]);
      lo[rr] = __dp4a(int(aw.y), int(bw0.y), lo[rr]);
      lo[rr] = __dp4a(int(aw.z), int(bw0.z), lo[rr]);
      lo[rr] = __dp4a(int(aw.w), int(bw0.w), lo[rr]);
      hi[rr] = __dp4a(int(aw.x), int(bw1.x), hi[rr]);
      hi[rr] = __dp4a(int(aw.y), int(bw1.y), hi[rr]);
      hi[rr] = __dp4a(int(aw.z), int(bw1.z), hi[rr]);
      hi[rr] = __dp4a(int(aw.w), int(bw1.w), hi[rr]);
    }
    bar_cuda_warps();
  }
}

__device__ __forceinline__ __half2 dup_lo(uint32_t u) { return __half2half2(__ushort_as_half((unsigned short)(u & 0xFFFF))); }
__device__ __forceinline__ __half2 dup_hi(uint32_t u) { return __half2half2(__ushort_as_half((unsigned short)(u >> 16))); }

// HFMA2 over k in [ka, kb) (multiples of 16): half2 lanes are the two columns; exact while each 16-k partial is
// at most 16·64 = 1024 in magnitude, then flushed into int32. A's FP16 rows are staged as in cuda_dp4a.
__device__ __forceinline__ void cuda_hfma2(const FusedArgs& f, unsigned char* smem, int ka, int kb, int r0, int c0,
                                           int lw, int lane, int* lo, int* hi) {
  const int n = f.n, cg = lw & 1, rq = lw >> 1, c = c0 + cg * 64 + 2 * lane, ltid = lw * 32 + lane;
  uint4* As = reinterpret_cast<uint4*>(smem);
  for (int k = ka; k < kb; k += 16) {
    As[ltid] = *reinterpret_cast<const uint4*>(f.Ah + size_t(r0 + (ltid >> 1)) * n + k + 8 * (ltid & 1));
    __half2 bv[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) bv[t] = *reinterpret_cast<const __half2*>(f.Bh + size_t(k + t) * n + c);
    bar_cuda_warps();
#pragma unroll
    for (int rr = 0; rr < 32; ++rr) {
      const uint4 a0 = As[2 * (rq * 32 + rr)], a1 = As[2 * (rq * 32 + rr) + 1];
      const uint32_t aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      __half2 acc = __float2half2_rn(0.f);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        acc = __hfma2(dup_lo(aw[t]), bv[2 * t], acc);
        acc = __hfma2(dup_hi(aw[t]), bv[2 * t + 1], acc);
      }
      lo[rr] += __half2int_rn(__low2half(acc));
      hi[rr] += __half2int_rn(__high2half(acc));
    }
    bar_cuda_warps();
  }
}

template <int ROLE>
__global__ void __launch_bounds__(512, 1) k_fused(FusedArgs f) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r0 = blockIdx.y * 128, c0 = blockIdx.x * 128, n = f.n, KT = n / 32;
  int* R = reinterpret_cast<int*>(smem);
  if (warp >= 8) {
    if (ROLE != ROLE_NONE) {
      const int lw = warp - 8, cg = lw & 1, rq = lw >> 1;
      int lo[32], hi[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) lo[i] = hi[i] = 0;
      if (ROLE == ROLE_FR) cuda_fr(f, smem, r0, c0, lw, lane, lo, hi);
      if (ROLE == ROLE_DP4A) cuda_dp4a(f, smem, f.k1, n, r0, c0, lw, lane, lo, hi);
      if (ROLE == ROLE_MIX) {
        const int km = f.k1 + (n - f.k1) / 32 * 16;
        cuda_dp4a(f, smem, f.k1, km, r0, c0, lw, lane, lo, hi);
        cuda_hfma2(f, smem, km, n, r0, c0, lw, lane, lo, hi);
      }
#pragma unroll
      for (int rr = 0; rr < 32; ++rr)
        *reinterpret_cast<int2*>(R + (rq * 32 + rr) * RSTRIDE + cg * 64 + 2 * lane) = make_int2(lo[rr], hi[rr]);
    }
  } else {
    const int wm = warp >> 1, wn = warp & 1, mt0 = (r0 >> 4) + wm * 2, nt0 = (c0 >> 3) + wn * 8;
    int acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0;
    const uint4* pa = f.Asw + size_t(mt0) * KT * 32 + lane;
    const uint2* pb = f.Bsw + size_t(nt0) * KT * 32 + lane;
    const size_t sa = size_t(KT) * 32;
    const int kt1 = f.k1 / 32;
    uint4 a[2];
    uint2 bb[8];
    if (kt1 > 0) {
#pragma unroll
      for (int i = 0; i < 2; ++i) a[i] = pa[i * sa];
#pragma unroll
      for (int j = 0; j < 8; ++j) bb[j] = pb[j * sa];
    }
    for (int kt = 0; kt < kt1; ++kt) {
      const int kn = min(kt + 1, kt1 - 1) * 32;
      uint4 an[2];
      uint2 bn[8];
#pragma unroll
      for (int i = 0; i < 2; ++i) an[i] = pa[i * sa + kn];
#pragma unroll
      for (int j = 0; j < 8; ++j) bn[j] = pb[j * sa + kn];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) mma_s8(acc[i][j], a[i], bb[j]);
#pragma unroll
      for (int i = 0; i < 2; ++i) a[i] = an[i];
#pragma unroll
      for (int j = 0; j < 8; ++j) bb[j] = bn[j];
    }
    __syncthreads();
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int lr = wm * 32 + i * 16 + g + 8 * h, lc = wn * 64 + j * 8 + 2 * t;
          int2 v = make_int2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
          if (ROLE != ROLE_NONE) {
            const int2 w = *reinterpret_cast<const int2*>(R + lr * RSTRIDE + lc);
            v.x += w.x;
            v.y += w.y;
          }
          *reinterpret_cast<int2*>(f.D + size_t(r0 + lr) * n + c0 + lc) = v;
        }
    return;
  }
  __syncthreads();
}

#ifdef __CUDACC_RTC__
template __global__ void k_fused<ROLE_NONE>(FusedArgs);
template __global__ void k_fused<ROLE_FR>(FusedArgs);
template __global__ void k_fused<ROLE_DP4A>(FusedArgs);
template __global__ void k_fused<ROLE_MIX>(FusedArgs);
#endif

#ifndef __CUDACC_RTC__
#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));         \
      exit(3);                                                                                   \
    }                                                                                            \
  } while (0)
#define LTC(x)                                                                                   \
  do {                                                                                           \
    cublasStatus_t s_ = (x);                                                                     \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                           \
      fprintf(stderr, "%s:%d %s: cublasStatus %d\n", __FILE__, __LINE__, #x, int(s_));           \
      exit(4);                                                                                   \
    }                                                                                            \
  } while (0)

struct Alphabet {
  const char* name;
  int m;
  int v[17];
};
static const Alphabet ALPHABETS[] = {
    {"k2", 4, {-2, -1, 1, 2}},
    {"k4", 8, {-4, -3, -2, -1, 1, 2, 3, 4}},
    {"k6", 12, {-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6}},
    {"k8", 16, {-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8}},
    {"u6", 13, {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}},
};

__global__ void k_fp8(const int8_t* M, uint8_t* out, size_t count) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    out[i] = __nv_fp8_e4m3(float(M[i])).__x;
}
__global__ void k_cmp_f32(const float* X, const int32_t* R, size_t count, unsigned long long* bad) {
  unsigned long long b = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    b += (X[i] != float(R[i])) || (float(R[i]) != float(double(R[i])));
  if (b) atomicAdd(bad, b);
}
__global__ void k_cmp_i32(const int32_t* X, const int32_t* R, size_t count, unsigned long long* bad) {
  unsigned long long b = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    b += X[i] != R[i];
  if (b) atomicAdd(bad, b);
}

enum Kind { K_INT8, K_FP8 };
static const char* kind_name[] = {"int8", "fp8_e4m3"};
struct Plan {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulPreference_t pref;
  Kind kind;
  cublasLtMatmulAlgo_t algo;
  size_t ws;
  int ncand = 0, best = -1;
};
static Plan make_plan(cublasLtHandle_t h, Kind kind, int n, size_t ws) {
  Plan p{};
  p.h = h;
  p.kind = kind;
  p.ws = ws;
  bool i8 = kind == K_INT8;
  cudaDataType_t in = i8 ? CUDA_R_8I : CUDA_R_8F_E4M3;
  cudaDataType_t out = i8 ? CUDA_R_32I : CUDA_R_32F;
  LTC(cublasLtMatmulDescCreate(&p.op, i8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F, i8 ? CUDA_R_32I : CUDA_R_32F));
  cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
  LTC(cublasLtMatrixLayoutCreate(&p.a, in, n, n, n));
  LTC(cublasLtMatrixLayoutCreate(&p.b, in, n, n, n));
  LTC(cublasLtMatrixLayoutCreate(&p.c, out, n, n, n));
  LTC(cublasLtMatmulPreferenceCreate(&p.pref));
  LTC(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  return p;
}
static void run(Plan& p, const cublasLtMatmulAlgo_t* algo, const void* A, const void* B, void* D, void* ws) {
  int32_t ia = 1, ib = 0;
  float fa = 1.f, fb = 0.f;
  const void* al = p.kind == K_INT8 ? static_cast<const void*>(&ia) : &fa;
  const void* be = p.kind == K_INT8 ? static_cast<const void*>(&ib) : &fb;
  LTC(cublasLtMatmul(p.h, p.op, al, B, p.a, A, p.b, be, D, p.c, D, p.c, algo, ws, p.ws, 0));
}
static float time_calls(const std::function<void()>& f, int iters) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for (int i = 0; i < iters; ++i) f();
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / iters;
}
static bool autotune(Plan& p, const void* A, const void* B, void* D, void* ws, std::string& log) {
  cublasLtMatmulHeuristicResult_t res[24];
  int got = 0;
  cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(p.h, p.op, p.a, p.b, p.c, p.c, p.pref, 24, res, &got);
  if (hs != CUBLAS_STATUS_SUCCESS || got == 0) return false;
  p.ncand = got;
  float best = 1e30f;
  char buf[96];
  for (int i = 0; i < got; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize > p.ws) continue;
    auto f = [&] { run(p, &res[i].algo, A, B, D, ws); };
    f();
    f();
    CK(cudaDeviceSynchronize());
    float t = 1e30f;
    for (int r = 0; r < 3; ++r) t = std::min(t, time_calls(f, 3));
    snprintf(buf, sizeof buf, "%s%.4f", log.empty() ? "" : ",", t);
    log += buf;
    if (t < best) {
      best = t;
      p.best = i;
      p.algo = res[i].algo;
    }
  }
  return p.best >= 0;
}
static nvmlDevice_t g_nvml;
static bool g_nvml_ok = false;
static unsigned sm_clock() {
  unsigned c = 0;
  if (g_nvml_ok) nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &c);
  return c;
}
static std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p;) {
    v.push_back(atoi(p));
    while (*p && *p != ',') ++p;
    if (*p == ',') ++p;
  }
  return v;
}
constexpr size_t SMEM_MAX = 99 * 1024;
static size_t fused_smem(int role, int mp, int G) {
  size_t epi = size_t(128) * RSTRIDE * 4;
  if (role != ROLE_FR) return role == ROLE_NONE ? 0 : epi;
  return std::max(epi, size_t(G) * mp * 64 * 4 + size_t(G) * 128 * 4);
}
static void launch_fused(int role, const FusedArgs& f) {
  dim3 grid(f.n / 128, f.n / 128);
  size_t sh = fused_smem(role, f.mp, f.G);
  if (role == ROLE_NONE) k_fused<ROLE_NONE><<<grid, 256, sh>>>(f);
  else if (role == ROLE_FR) k_fused<ROLE_FR><<<grid, 512, sh>>>(f);
  else if (role == ROLE_DP4A) k_fused<ROLE_DP4A><<<grid, 512, sh>>>(f);
  else k_fused<ROLE_MIX><<<grid, 512, sh>>>(f);
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: fused_coissue <n> <alphabet> <seed> <rounds> <alphas64>\n");
    return 2;
  }
  const int n = atoi(argv[1]);
  const Alphabet* al = nullptr;
  for (const Alphabet& a : ALPHABETS)
    if (!strcmp(a.name, argv[2])) al = &a;
  const uint64_t seed = strtoull(argv[3], nullptr, 0);
  const int rounds = atoi(argv[4]);
  const std::vector<int> alphas = parse_list(argv[5]);
  if (!al || n % 128) {
    fprintf(stderr, "bad alphabet or n (n must be a multiple of 128)\n");
    return 2;
  }
  const size_t N2 = size_t(n) * n;
  char pci[32];
  CK(cudaDeviceGetPCIBusId(pci, sizeof pci, 0));
  if (nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByPciBusId_v2(pci, &g_nvml) == NVML_SUCCESS) g_nvml_ok = true;
  char uuid[96] = "?";
  if (g_nvml_ok) nvmlDeviceGetUUID(g_nvml, uuid, sizeof uuid);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  CK(cudaFuncSetAttribute(k_fused<ROLE_FR>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM_MAX)));
  CK(cudaFuncSetAttribute(k_fused<ROLE_DP4A>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM_MAX)));
  CK(cudaFuncSetAttribute(k_fused<ROLE_MIX>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM_MAX)));

  Vals sv{};
  int maxs = 0;
  for (int i = 0; i < al->m; ++i) {
    sv.v[i] = al->v[i];
    maxs = std::max(maxs, abs(al->v[i]));
  }
  uint8_t *AI, *BI;
  int8_t *A, *B;
  CK(cudaMalloc(&AI, N2));
  CK(cudaMalloc(&BI, N2));
  CK(cudaMalloc(&A, N2));
  CK(cudaMalloc(&B, N2));
  k_gen<<<1024, 256>>>(AI, A, N2, seed, al->m, sv);
  k_gen<<<1024, 256>>>(BI, B, N2, seed ^ 0xB0B0B0B0B0B0B0B0ull, al->m, sv);
  uint4* Asw;
  uint2* Bsw;
  __half *Ah, *Bh;
  CK(cudaMalloc(&Asw, N2));
  CK(cudaMalloc(&Bsw, N2));
  CK(cudaMalloc(&Ah, N2 * 2));
  CK(cudaMalloc(&Bh, N2 * 2));
  k_swz_a<<<1024, 256>>>(A, Asw, n);
  k_swz_b<<<1024, 256>>>(B, Bsw, n);
  k_half<<<1024, 256>>>(A, Ah, N2);
  k_half_t<<<1024, 256>>>(B, Bh, n);
  CK(cudaGetLastError());

  int32_t *Dref, *Di;
  float* Df;
  CK(cudaMalloc(&Dref, N2 * 4));
  CK(cudaMalloc(&Di, N2 * 4));
  CK(cudaMalloc(&Df, N2 * 4));
  const size_t WS = size_t(64) << 20;
  void* ws;
  CK(cudaMalloc(&ws, WS));
  unsigned long long* dbad;
  CK(cudaMalloc(&dbad, 8));
  cublasLtHandle_t h;
  LTC(cublasLtCreate(&h));
  printf("{\"event\":\"start\",\"prog\":\"fused_coissue\",\"n\":%d,\"alphabet\":\"%s\",\"m\":%d,\"seed\":%llu,"
         "\"gpu\":\"%s\",\"uuid\":\"%s\",\"sms\":%d,\"cublasLt\":%zu,\"sm_clock_mhz\":%u}\n",
         n, al->name, al->m, (unsigned long long)seed, prop.name, uuid, prop.multiProcessorCount,
         cublasLtGetVersion(), sm_clock());

  Plan p8 = make_plan(h, K_INT8, n, WS);
  std::string tl8;
  if (!autotune(p8, A, B, Dref, ws, tl8)) return 5;
  CK(cudaMemset(Dref, 0xA5, N2 * 4));
  run(p8, &p8.algo, A, B, Dref, ws);
  CK(cudaDeviceSynchronize());
  {
    std::vector<int8_t> hA(N2), hB(N2);
    std::vector<int32_t> hD(N2);
    CK(cudaMemcpy(hA.data(), A, N2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hB.data(), B, N2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hD.data(), Dref, N2 * 4, cudaMemcpyDeviceToHost));
    size_t checked = 0, bad = 0;
    auto chk = [&](size_t r, size_t c) {
      int64_t s = 0;
      for (int j = 0; j < n; ++j) s += int64_t(hA[r * n + j]) * int64_t(hB[c * n + j]);
      ++checked;
      bad += s != int64_t(hD[r * n + c]);
    };
    if (n <= 1024) {
      for (size_t r = 0; r < size_t(n); ++r)
        for (size_t c = 0; c < size_t(n); ++c) chk(r, c);
    } else {
      for (uint64_t i = 0; i < 4096; ++i) {
        uint64_t x = mix64(seed + 977 * i);
        chk(x % n, (x >> 32) % n);
      }
      chk(0, 0);
      chk(n - 1, n - 1);
    }
    printf("{\"event\":\"gate\",\"route\":\"int8\",\"against\":\"cpu_int64\",\"checked\":%zu,\"mismatches\":%zu,"
           "\"candidates\":%d,\"best\":%d,\"autotune_ms\":[%s]}\n",
           checked, bad, p8.ncand, p8.best, tl8.c_str());
    if (bad) return 6;
  }
  auto cmp_i = [&](const int32_t* X) {
    CK(cudaMemset(dbad, 0, 8));
    k_cmp_i32<<<1024, 256>>>(X, Dref, N2, dbad);
    unsigned long long b;
    CK(cudaMemcpy(&b, dbad, 8, cudaMemcpyDeviceToHost));
    return b;
  };
  auto cmp_f = [&](const float* X) {
    CK(cudaMemset(dbad, 0, 8));
    k_cmp_f32<<<1024, 256>>>(X, Dref, N2, dbad);
    unsigned long long b;
    CK(cudaMemcpy(&b, dbad, 8, cudaMemcpyDeviceToHost));
    return b;
  };

  struct Route {
    std::string name;
    std::function<void()> f;
    double alpha = 0;
    int b = 0;
    double ms1 = 0;
  };
  std::vector<Route> routes;
  routes.push_back({"int8", [&] { run(p8, &p8.algo, A, B, Di, ws); }});

  uint8_t *A8, *B8;
  CK(cudaMalloc(&A8, N2));
  CK(cudaMalloc(&B8, N2));
  k_fp8<<<1024, 256>>>(A, A8, N2);
  k_fp8<<<1024, 256>>>(B, B8, N2);
  Plan pf8 = make_plan(h, K_FP8, n, WS);
  std::string tlf8;
  if (autotune(pf8, A8, B8, Df, ws, tlf8)) {
    CK(cudaMemset(Df, 0xA5, N2 * 4));
    run(pf8, &pf8.algo, A8, B8, Df, ws);
    CK(cudaDeviceSynchronize());
    unsigned long long b = cmp_f(Df);
    printf("{\"event\":\"gate\",\"route\":\"fp8_e4m3\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu}\n", N2,
           b);
    if (!b) routes.push_back({"fp8_e4m3", [&] { run(pf8, &pf8.algo, A8, B8, Df, ws); }});
  }

  // fused routes; each gated against int8 bit for bit (poisoned output first)
  std::vector<FusedArgs> fargs;
  fargs.reserve(256);
  auto add_fused = [&](const char* name, int role, int k1, int b, double alpha) {
    FusedArgs f{Asw, Bsw, AI, BI, A, B, Ah, Bh, Di, n, k1, b, al->m, 1, 1, 0, sv};
    if (role == ROLE_FR) {
      int mp = 1;
      for (int t = 0; t < b; ++t) mp *= al->m;
      f.mp = mp;
      f.G = int(std::min<size_t>(8, (SMEM_MAX - 1024) / (size_t(mp) * 256 + 512)));
      f.bias = b * maxs * maxs;
      if (f.G < 1) {
        printf("{\"event\":\"skip\",\"route\":\"%s\",\"reason\":\"table over shared memory\"}\n", name);
        return;
      }
    }
    fargs.push_back(f);
    const FusedArgs* fp = &fargs.back();
    CK(cudaMemset(Di, 0xA5, N2 * 4));
    launch_fused(role, *fp);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    unsigned long long bad = cmp_i(Di);
    printf("{\"event\":\"gate\",\"route\":\"%s\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu,"
           "\"k_tensor\":%d,\"k_cuda\":%d,\"b\":%d,\"G\":%d,\"smem\":%zu}\n",
           name, N2, bad, k1, n - k1, b, fp->G, fused_smem(role, fp->mp, fp->G));
    if (bad) return;
    Route r{name, [role, fp] { launch_fused(role, *fp); }};
    r.alpha = double(n - k1) / n;
    r.b = b;
    routes.push_back(r);
  };
  add_fused("own_mma", ROLE_NONE, n, 0, 0);
  add_fused("fused_fr_a0", ROLE_FR, n, 2, 0);
  char name[64];
  const int bmax = al->m <= 4 ? 4 : al->m <= 8 ? 3 : 2;
  for (int j : alphas) {
    const int k1 = (n - int(double(n) * j / 64.0)) / 32 * 32;
    if (k1 <= 0 || k1 >= n || (n - k1) % 16) continue;
    for (int b = 1; b <= bmax; ++b) {
      snprintf(name, sizeof name, "fused_fr_b%d_a%d", b, j);
      add_fused(name, ROLE_FR, k1, b, double(n - k1) / n);
    }
    snprintf(name, sizeof name, "fused_dp4a_a%d", j);
    add_fused(name, ROLE_DP4A, k1, 0, double(n - k1) / n);
    if ((n - k1) % 32 == 0) {
      snprintf(name, sizeof name, "fused_mix_a%d", j);
      add_fused(name, ROLE_MIX, k1, 0, double(n - k1) / n);
    }
  }

  {  // control: the int8 product with one entry of A changed to another alphabet value must be rejected
    int8_t v0;
    CK(cudaMemcpy(&v0, A, 1, cudaMemcpyDeviceToHost));
    int8_t v1 = int8_t(al->v[0] == v0 ? al->v[1] : al->v[0]);
    CK(cudaMemcpy(A, &v1, 1, cudaMemcpyHostToDevice));
    CK(cudaMemset(Di, 0xA5, N2 * 4));
    run(p8, &p8.algo, A, B, Di, ws);
    CK(cudaDeviceSynchronize());
    unsigned long long bc = cmp_i(Di);
    CK(cudaMemcpy(A, &v0, 1, cudaMemcpyHostToDevice));
    printf("{\"event\":\"control\",\"route\":\"int8\",\"changed\":\"A[0][0] %d->%d\",\"mismatches\":%llu,"
           "\"rejected\":%s}\n",
           v0, v1, bc, bc ? "true" : "false");
    if (!bc) return 7;
  }

  // a pre-pass time for every gated route; the interleaved rounds time the baselines, own_mma, fused_fr_a0 and the
  // five fastest fused hybrids of each role
  for (Route& r : routes) {
    r.f();
    CK(cudaDeviceSynchronize());
    r.ms1 = time_calls(r.f, 3);
    printf("{\"event\":\"prepass\",\"route\":\"%s\",\"alpha\":%.4f,\"b\":%d,\"ms\":%.5f}\n", r.name.c_str(), r.alpha,
           r.b, r.ms1);
  }
  std::vector<Route> timed;
  for (const Route& r : routes)
    if (r.name.rfind("fused_", 0) != 0 || r.name == "fused_fr_a0") timed.push_back(r);
  for (const char* pre : {"fused_fr_b", "fused_dp4a_", "fused_mix_"}) {
    std::vector<Route> c;
    for (const Route& r : routes)
      if (r.name.rfind(pre, 0) == 0) c.push_back(r);
    std::sort(c.begin(), c.end(), [](const Route& x, const Route& y) { return x.ms1 < y.ms1; });
    for (size_t i = 0; i < c.size() && i < 5; ++i) timed.push_back(c[i]);
  }
  std::vector<std::vector<double>> per(timed.size());
  std::vector<std::vector<unsigned>> clk(timed.size());
  std::vector<int> iters(timed.size());
  for (size_t i = 0; i < timed.size(); ++i) iters[i] = std::max(1, int(25.0 / timed[i].ms1));
  for (int rd = 0; rd < rounds; ++rd)
    for (size_t i = 0; i < timed.size(); ++i) {
      unsigned c0 = sm_clock();
      double ms = time_calls(timed[i].f, iters[i]);
      unsigned c1 = sm_clock();
      per[i].push_back(ms);
      clk[i].push_back(c0);
      clk[i].push_back(c1);
      printf("{\"event\":\"rep\",\"route\":\"%s\",\"round\":%d,\"iters\":%d,\"ms\":%.5f,\"clk0\":%u,\"clk1\":%u}\n",
             timed[i].name.c_str(), rd, iters[i], ms, c0, c1);
    }
  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size();
    return m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
  };
  const double base = med(per[0]);
  const double macs = double(n) * n * n;
  for (size_t i = 0; i < timed.size(); ++i) {
    double m = med(per[i]), mn = *std::min_element(per[i].begin(), per[i].end());
    unsigned cmin = *std::min_element(clk[i].begin(), clk[i].end()),
             cmax = *std::max_element(clk[i].begin(), clk[i].end());
    printf("{\"event\":\"summary\",\"prog\":\"fused_coissue\",\"n\":%d,\"alphabet\":\"%s\",\"route\":\"%s\","
           "\"alpha\":%.4f,\"b\":%d,\"median_ms\":%.5f,\"min_ms\":%.5f,\"ratio_vs_int8\":%.4f,"
           "\"macs_per_sm_clk_at_2092\":%.1f,\"clk_min\":%u,\"clk_max\":%u}\n",
           n, al->name, timed[i].name.c_str(), timed[i].alpha, timed[i].b, m, mn, m / base,
           macs / (m * 1e-3) / (prop.multiProcessorCount * 2092e6), cmin, cmax);
  }
  return 0;
}
#endif
