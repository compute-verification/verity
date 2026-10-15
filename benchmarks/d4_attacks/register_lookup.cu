// Standalone table lookups for the exact product D = A·Bᵀ (entries i.i.d. uniform on a small alphabet S) on sm_120,
// with the tables where the shared-memory route (lookup_coissue.cu's fr_b<b>) doesn't put them:
//   fr_l1_b<b>   Four Russians as fr_b<b>, its int16 table in global memory (one region a CTA), gathered by LDG from L1.
//   prmt_b1      a register table per (column, k): S[d]·B[c][k] + β for the eight d in one 8-byte register pair, and
//                one PRMT gathers it for four rows at once (the selector's nibbles are the rows' indices of A).
//                Past 8 values (|S| > 8) the split: a second PRMT on a second table and a byte mask choose per row.
//   shfl_b<b>    a warp table per (4 columns, chunk of b k): lane l holds pattern l's four biased byte entries, and
//                one SHFL gathers them for the lane's row. Past 32 patterns the split: two SHFLs and a select.
// Entries are biased to be non-negative bytes and accumulated four to a word, widened into int32 at the overflow
// bound. Gates and timing as lookup_coissue.cu: int8 against a CPU int64 reference, every route against int8 bit for
// bit after 0xA5 poisoning, a control that must be rejected, interleaved ~25 ms items with NVML SM clocks.
//
//   register_lookup <n> <alphabet> <seed> <rounds>      alphabet: k2 k4 k6 k8 u6
#ifndef __CUDACC_RTC__
#include <cublasLt.h>
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
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef short int16_t;
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

// Four Russians with the table in global memory: as lookup_coissue.cu's k_fr, T a CTA's own region of Tg.
constexpr int FR_RT = 4, FR_RB = 256 * FR_RT;
template <int COLS>
__global__ void __launch_bounds__(256) k_fr_l1(const uint8_t* __restrict__ AI, const uint8_t* __restrict__ BI,
                                               int* __restrict__ D, int16_t* Tg, int n, int b, int m, int mpmax,
                                               Vals s) {
  int16_t* T = Tg + size_t(blockIdx.y * gridDim.x + blockIdx.x) * mpmax * COLS;
  const int tid = threadIdx.x, c0 = blockIdx.x * COLS, rbase = blockIdx.y * FR_RB, k = n;
  int acc[FR_RT][COLS];
#pragma unroll
  for (int i = 0; i < FR_RT; ++i)
#pragma unroll
    for (int c = 0; c < COLS; ++c) acc[i][c] = 0;
  for (int base = 0; base < k; base += b) {
    const int w = min(b, k - base);
    int mp = 1;
    for (int t = 0; t < w; ++t) mp *= m;
    for (int e = tid; e < mp * COLS; e += 256) {
      const int c = e % COLS;
      int p = e / COLS, x = 0;
      const uint8_t* bc = BI + size_t(c0 + c) * k + base;
      for (int t = 0; t < w; ++t) {
        x += s.v[p % m] * s.v[bc[t]];
        p /= m;
      }
      T[e] = int16_t(x);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < FR_RT; ++i) {
      const uint8_t* ar = AI + size_t(rbase + i * 256 + tid) * k + base;
      int p = 0, pw = 1;
      for (int t = 0; t < w; ++t) {
        p += ar[t] * pw;
        pw *= m;
      }
      const int16_t* row = T + p * COLS;
#pragma unroll
      for (int c = 0; c < COLS; ++c) acc[i][c] += row[c];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < FR_RT; ++i)
#pragma unroll
    for (int c = 0; c < COLS; ++c) D[size_t(rbase + i * 256 + tid) * n + c0 + c] = acc[i][c];
}

// PRMT selectors: Asel[q·n + k] holds the low 3 bits of A's indices of rows 4q..4q+3 at k as nibbles, Amask the bytes
// 0xFF where an index is >= 8 (the split's second table).
__global__ void k_asel(const uint8_t* __restrict__ AI, uint16_t* Asel, uint32_t* Amask, int n) {
  for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < size_t(n / 4) * n; e += size_t(gridDim.x) * blockDim.x) {
    const size_t q = e / n, k = e % n;
    uint32_t sel = 0, msk = 0;
    for (int j = 0; j < 4; ++j) {
      const uint32_t d = AI[(4 * q + j) * n + k];
      sel |= (d & 7) << (4 * j);
      msk |= (d >= 8 ? 0xFFu : 0u) << (8 * j);
    }
    Asel[e] = uint16_t(sel);
    Amask[e] = msk;
  }
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}

// A CTA: 32 columns (a lane each) × 256 rows (a warp 8 quads = 32 rows). W = byte-accumulation steps before widening.
template <bool SPLIT>
__global__ void __launch_bounds__(256) k_prmt(const uint16_t* __restrict__ Asel, const uint32_t* __restrict__ Amask,
                                              const int8_t* __restrict__ B, int* __restrict__ D, int n, int m, int beta,
                                              int W, Vals s) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, c = blockIdx.x * 32 + lane;
  const int q0 = blockIdx.y * 64 + warp * 8;
  int acc[8][4];
  uint32_t accb[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    accb[q] = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[q][j] = 0;
  }
  int cnt = 0;
  for (int k0 = 0; k0 < n; k0 += 16) {
    const uint4 bq = *reinterpret_cast<const uint4*>(B + size_t(c) * n + k0);
    const uint32_t bw[4] = {bq.x, bq.y, bq.z, bq.w};
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      const int bv = int(int8_t(bw[kk >> 2] >> (8 * (kk & 3))));
      uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        t0 |= uint32_t(s.v[d] * bv + beta) << (8 * d);
        t1 |= uint32_t(s.v[4 + d] * bv + beta) << (8 * d);
        if (SPLIT) {
          t2 |= uint32_t(s.v[8 + d] * bv + beta) << (8 * d);
          t3 |= uint32_t(s.v[12 + d] * bv + beta) << (8 * d);
        }
      }
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const size_t idx = size_t(q0 + q) * n + k0 + kk;
        const uint32_t sel = Asel[idx];
        uint32_t v = prmt(t0, t1, sel);
        if (SPLIT) {
          const uint32_t msk = Amask[idx];
          v = (v & ~msk) | (prmt(t2, t3, sel) & msk);
        }
        accb[q] += v;
      }
      if (++cnt == W) {
        cnt = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[q][j] += int((accb[q] >> (8 * j)) & 0xFF);
          accb[q] = 0;
        }
      }
    }
  }
#pragma unroll
  for (int q = 0; q < 8; ++q)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int v = acc[q][j] + int((accb[q] >> (8 * j)) & 0xFF) - n * beta;
      D[size_t(4 * (q0 + q) + j) * n + c] = v;
    }
}

// SHFL patterns: Pat[r·nch + ch] = sum_t A_idx[r][ch·b + t]·m^t (b divides n here).
__global__ void k_pat(const uint8_t* __restrict__ AI, uint8_t* Pat, int n, int b, int m) {
  const int nch = n / b;
  for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < size_t(n) * nch; e += size_t(gridDim.x) * blockDim.x) {
    const size_t r = e / nch, ch = e % nch;
    int p = 0, pw = 1;
    for (int t = 0; t < b; ++t) {
      p += AI[r * n + ch * b + t] * pw;
      pw *= m;
    }
    Pat[e] = uint8_t(p);
  }
}

// A warp: 4 columns (bytes of one word) × 4 groups of 32 rows; a CTA 8 warps = 32 columns × 128 rows. Per chunk, lane l
// builds pattern l's entries (and l + 32's for the split); each row's lane gathers its pattern's word by SHFL.
template <int B, bool SPLIT>
__global__ void __launch_bounds__(256) k_shfl(const uint8_t* __restrict__ Pat, const int8_t* __restrict__ Bm,
                                              int* __restrict__ D, int n, int m, int beta, int W, Vals s) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, c0 = blockIdx.x * 32 + warp * 4;
  const int rbase = blockIdx.y * 128, nch = n / B;
  int acc[4][4];
  uint32_t accb[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    accb[g] = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[g][j] = 0;
  }
  int dg0[B], dg1[B];  // the digits of patterns lane and lane + 32
  {
    int p0 = lane, p1 = lane + 32;
#pragma unroll
    for (int t = 0; t < B; ++t) {
      dg0[t] = s.v[p0 % m];
      dg1[t] = s.v[p1 % m];
      p0 /= m;
      p1 /= m;
    }
  }
  int cnt = 0;
  for (int ch0 = 0; ch0 < nch; ch0 += 16) {
    uint4 pw[4];
#pragma unroll
    for (int g = 0; g < 4; ++g) pw[g] = *reinterpret_cast<const uint4*>(Pat + size_t(rbase + g * 32 + lane) * nch + ch0);
#pragma unroll 4
    for (int cc = 0; cc < 16; ++cc) {
      const int k = (ch0 + cc) * B;
      uint32_t e0 = 0, e1 = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int x0 = 0, x1 = 0;
#pragma unroll
        for (int t = 0; t < B; ++t) {
          const int bv = Bm[size_t(c0 + j) * n + k + t];
          x0 += dg0[t] * bv;
          x1 += dg1[t] * bv;
        }
        e0 |= uint32_t(x0 + beta) << (8 * j);
        e1 |= uint32_t(x1 + beta) << (8 * j);
      }
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const uint32_t word = cc < 8 ? (cc < 4 ? pw[g].x : pw[g].y) : (cc < 12 ? pw[g].z : pw[g].w);
        const int p = int((word >> (8 * (cc & 3))) & 0xFF);
        uint32_t v = __shfl_sync(0xFFFFFFFFu, e0, p & 31);
        if (SPLIT) {
          const uint32_t v1 = __shfl_sync(0xFFFFFFFFu, e1, p & 31);
          v = p < 32 ? v : v1;
        }
        accb[g] += v;
      }
      if (++cnt == W) {
        cnt = 0;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[g][j] += int((accb[g] >> (8 * j)) & 0xFF);
          accb[g] = 0;
        }
      }
    }
  }
#pragma unroll
  for (int g = 0; g < 4; ++g)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      D[size_t(rbase + g * 32 + lane) * n + c0 + j] = acc[g][j] + int((accb[g] >> (8 * j)) & 0xFF) - nch * beta;
}

#ifdef __CUDACC_RTC__
template __global__ void k_fr_l1<32>(const uint8_t*, const uint8_t*, int*, int16_t*, int, int, int, int, Vals);
template __global__ void k_prmt<false>(const uint16_t*, const uint32_t*, const int8_t*, int*, int, int, int, int, Vals);
template __global__ void k_prmt<true>(const uint16_t*, const uint32_t*, const int8_t*, int*, int, int, int, int, Vals);
template __global__ void k_shfl<1, false>(const uint8_t*, const int8_t*, int*, int, int, int, int, Vals);
template __global__ void k_shfl<2, false>(const uint8_t*, const int8_t*, int*, int, int, int, int, Vals);
template __global__ void k_shfl<2, true>(const uint8_t*, const int8_t*, int*, int, int, int, int, Vals);
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
__global__ void k_cmp_i32(const int32_t* X, const int32_t* R, size_t count, unsigned long long* bad) {
  unsigned long long b = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    b += X[i] != R[i];
  if (b) atomicAdd(bad, b);
}
struct Plan {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulAlgo_t algo;
  size_t ws;
  int ncand = 0, best = -1;
};
static Plan make_plan(cublasLtHandle_t h, int n, size_t ws) {
  Plan p{};
  p.h = h;
  p.ws = ws;
  LTC(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
  LTC(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8I, n, n, n));
  LTC(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8I, n, n, n));
  LTC(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32I, n, n, n));
  LTC(cublasLtMatmulPreferenceCreate(&p.pref));
  LTC(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  return p;
}
static void run(Plan& p, const cublasLtMatmulAlgo_t* algo, const void* A, const void* B, void* D, void* ws) {
  int32_t ia = 1, ib = 0;
  LTC(cublasLtMatmul(p.h, p.op, &ia, B, p.a, A, p.b, &ib, D, p.c, D, p.c, algo, ws, p.ws, 0));
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
static bool autotune(Plan& p, const void* A, const void* B, void* D, void* ws) {
  cublasLtMatmulHeuristicResult_t res[24];
  int got = 0;
  if (cublasLtMatmulAlgoGetHeuristic(p.h, p.op, p.a, p.b, p.c, p.c, p.pref, 24, res, &got) != CUBLAS_STATUS_SUCCESS ||
      !got)
    return false;
  p.ncand = got;
  float best = 1e30f;
  for (int i = 0; i < got; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize > p.ws) continue;
    auto f = [&] { run(p, &res[i].algo, A, B, D, ws); };
    f();
    CK(cudaDeviceSynchronize());
    float t = 1e30f;
    for (int r = 0; r < 3; ++r) t = std::min(t, time_calls(f, 3));
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

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: register_lookup <n> <alphabet> <seed> <rounds>\n");
    return 2;
  }
  const int n = atoi(argv[1]);
  const Alphabet* al = nullptr;
  for (const Alphabet& a : ALPHABETS)
    if (!strcmp(a.name, argv[2])) al = &a;
  const uint64_t seed = strtoull(argv[3], nullptr, 0);
  const int rounds = atoi(argv[4]);
  if (!al || n % 1024) {
    fprintf(stderr, "bad alphabet or n (n must be a multiple of 1024)\n");
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
  int32_t *Dref, *Di;
  CK(cudaMalloc(&Dref, N2 * 4));
  CK(cudaMalloc(&Di, N2 * 4));
  const size_t WS = size_t(64) << 20;
  void* ws;
  CK(cudaMalloc(&ws, WS));
  unsigned long long* dbad;
  CK(cudaMalloc(&dbad, 8));
  cublasLtHandle_t h;
  LTC(cublasLtCreate(&h));
  printf("{\"event\":\"start\",\"prog\":\"register_lookup\",\"n\":%d,\"alphabet\":\"%s\",\"m\":%d,\"gpu\":\"%s\","
         "\"uuid\":\"%s\",\"sms\":%d,\"sm_clock_mhz\":%u}\n",
         n, al->name, al->m, prop.name, uuid, prop.multiProcessorCount, sm_clock());

  Plan p8 = make_plan(h, n, WS);
  if (!autotune(p8, A, B, Dref, ws)) return 5;
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
    printf("{\"event\":\"gate\",\"route\":\"int8\",\"against\":\"cpu_int64\",\"checked\":%zu,\"mismatches\":%zu}\n",
           checked, bad);
    if (bad) return 6;
  }
  auto cmp_i = [&](const int32_t* X) {
    CK(cudaMemset(dbad, 0, 8));
    k_cmp_i32<<<1024, 256>>>(X, Dref, N2, dbad);
    unsigned long long b;
    CK(cudaMemcpy(&b, dbad, 8, cudaMemcpyDeviceToHost));
    return b;
  };
  {
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

  struct Route {
    std::string name;
    std::function<void()> f;
    int b = 0;
    double ms1 = 0;
  };
  std::vector<Route> routes;
  routes.push_back({"int8", [&] { run(p8, &p8.algo, A, B, Di, ws); }});
  auto gate = [&](const std::string& name, int b, std::function<void()> f) {
    CK(cudaMemset(Di, 0xA5, N2 * 4));
    f();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    unsigned long long bad = cmp_i(Di);
    printf("{\"event\":\"gate\",\"route\":\"%s\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu}\n",
           name.c_str(), N2, bad);
    if (!bad) routes.push_back({name, f, b});
  };

  // L1 tables: the widths whose 32-column int16 table is at most 64 KiB (m^b <= 1024)
  int16_t* Tg = nullptr;
  for (int b = 1; b <= 5; ++b) {
    long mp = 1;
    for (int t = 0; t < b; ++t) mp *= al->m;
    if (mp > 1024) break;
    const size_t ctas = size_t(n / 32) * (n / FR_RB);
    if (!Tg) CK(cudaMalloc(&Tg, ctas * 1024 * 32 * 2));
    const int mpi = int(mp);
    gate("fr_l1_b" + std::to_string(b), b, [=] {
      k_fr_l1<32><<<dim3(n / 32, n / FR_RB), 256>>>(AI, BI, Di, Tg, n, b, al->m, mpi, sv);
    });
  }

  // PRMT, b = 1: direct for |S| <= 8, split past it (|S| <= 16)
  uint16_t* Asel;
  uint32_t* Amask;
  CK(cudaMalloc(&Asel, N2 / 4 * 2));
  CK(cudaMalloc(&Amask, N2 / 4 * 4));
  k_asel<<<1024, 256>>>(AI, Asel, Amask, n);
  {
    const int beta = maxs * maxs, W = std::max(1, 255 / (2 * beta));
    const bool split = al->m > 8;
    gate(split ? "prmt_b1_split" : "prmt_b1", 1, [=] {
      if (split) k_prmt<true><<<dim3(n / 32, n / 256), 256>>>(Asel, Amask, B, Di, n, al->m, beta, W, sv);
      else k_prmt<false><<<dim3(n / 32, n / 256), 256>>>(Asel, Amask, B, Di, n, al->m, beta, W, sv);
    });
  }

  // SHFL: b = 1 for every |S| <= 32; b = 2 where m^2 <= 32, split where m^2 <= 64
  uint8_t* Pat;
  CK(cudaMalloc(&Pat, N2));
  for (int b = 1; b <= 2; ++b) {
    const int mp = b == 1 ? al->m : al->m * al->m;
    if (mp > 64) continue;
    const bool split = mp > 32;
    const int beta = b * maxs * maxs, W = std::max(1, 255 / (2 * beta));
    if (2 * beta > 255) continue;
    uint8_t* P = Pat;
    std::string name = "shfl_b" + std::to_string(b) + (split ? "_split" : "");
    gate(name, b, [=] {
      k_pat<<<1024, 256>>>(AI, P, n, b, al->m);
      dim3 grid(n / 32, n / 128);
      if (b == 1) k_shfl<1, false><<<grid, 256>>>(P, B, Di, n, al->m, beta, W, sv);
      else if (!split) k_shfl<2, false><<<grid, 256>>>(P, B, Di, n, al->m, beta, W, sv);
      else k_shfl<2, true><<<grid, 256>>>(P, B, Di, n, al->m, beta, W, sv);
    });
  }

  for (Route& r : routes) {
    r.f();
    CK(cudaDeviceSynchronize());
    r.ms1 = time_calls(r.f, 2);
  }
  std::vector<std::vector<double>> per(routes.size());
  std::vector<std::vector<unsigned>> clk(routes.size());
  std::vector<int> iters(routes.size());
  for (size_t i = 0; i < routes.size(); ++i) iters[i] = std::max(1, int(25.0 / routes[i].ms1));
  for (int rd = 0; rd < rounds; ++rd)
    for (size_t i = 0; i < routes.size(); ++i) {
      unsigned c0 = sm_clock();
      double ms = time_calls(routes[i].f, iters[i]);
      unsigned c1 = sm_clock();
      per[i].push_back(ms);
      clk[i].push_back(c0);
      clk[i].push_back(c1);
      printf("{\"event\":\"rep\",\"route\":\"%s\",\"round\":%d,\"iters\":%d,\"ms\":%.5f,\"clk0\":%u,\"clk1\":%u}\n",
             routes[i].name.c_str(), rd, iters[i], ms, c0, c1);
    }
  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size();
    return m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
  };
  const double base = med(per[0]), macs = double(n) * n * n;
  for (size_t i = 0; i < routes.size(); ++i) {
    double m = med(per[i]), mn = *std::min_element(per[i].begin(), per[i].end());
    unsigned cmin = *std::min_element(clk[i].begin(), clk[i].end()),
             cmax = *std::max_element(clk[i].begin(), clk[i].end());
    printf("{\"event\":\"summary\",\"prog\":\"register_lookup\",\"n\":%d,\"alphabet\":\"%s\",\"route\":\"%s\","
           "\"alpha\":1,\"b\":%d,\"median_ms\":%.5f,\"min_ms\":%.5f,\"ratio_vs_int8\":%.4f,"
           "\"macs_per_sm_clk_at_2092\":%.1f,\"clk_min\":%u,\"clk_max\":%u}\n",
           n, al->name, routes[i].name.c_str(), routes[i].b, m, mn, m / base,
           macs / (m * 1e-3) / (prop.multiProcessorCount * 2092e6), cmin, cmax);
  }
  return 0;
}
#endif
