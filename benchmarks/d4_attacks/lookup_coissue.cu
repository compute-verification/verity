// Table-lookup (Four Russians) arithmetic beside the tensor-core GEMM on sm_120, for the exact integer product
// D = A·Bᵀ of two random n×n matrices with entries i.i.d. uniform on a small alphabet S. It calibrates the named premise
// NoLookupGain(S, n, η) (note:20261015T1814Z-draft-rs-machine-model §6.3).
//
// Routes, all exact, all producing int32 D:
//   int8         cuBLASLt int8 IMMA over the whole k (the tensor-only baseline T_tensor).
//   fp8_e4m3     cuBLASLt FP8 E4M3 over the whole k, FP32 out (an alternative 8-bit baseline).
//   fr_b<b>      a standalone Four Russians GEMM at table width b: for each chunk of b consecutive k and the CTA's COLS
//                columns, T[p][c] = sum_t S[p_t]·B[c][base+t] is built in shared memory (int16 entries) for every
//                pattern p in S^b, then each row adds T[pattern(A[r][chunk])][c]. A b-tail handles b not dividing k.
//   hyb_a<j>     the co-issue hybrid: cuBLASLt int8 over k1 = (1 − α)k on one stream and Four Russians over the last
//                α·k on another, concurrently on the same SMs, then summed. α = j/16.
//
// Gates before timing, in the same process: int8 against a CPU int64 reference (every entry at n <= 1024, 4096 sampled
// entries otherwise), then every other route against int8 bit for bit; a mismatching route is reported and not timed.
// A control (int8 with one entry of A changed to another alphabet value) must mismatch. Outputs are poisoned with 0xA5
// before each gate call. Timing: interleaved rounds of items of about 25 ms of device time, the SM clock read by NVML
// before and after each item. JSON lines on stdout.
//
//   lookup_coissue <n> <alphabet> <seed> <rounds> <bs> <alphas16>
//     alphabet: k2 k4 k6 k8 u6     bs: comma list of table widths     alphas16: comma list of j (α = j/16)
#include <cublasLt.h>
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

struct Vals {
  int v[17];
};

__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Indices in [0, m), uniform; and the value matrix S[idx].
__global__ void k_gen(uint8_t* I, int8_t* V, size_t count, uint64_t seed, int m, Vals s) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t d = uint32_t(mix64(seed ^ (i * 0xD1B54A32D192ED03ull)) % uint64_t(m));
    I[i] = uint8_t(d);
    V[i] = int8_t(s.v[d]);
  }
}

__global__ void k_fp8(const int8_t* M, uint8_t* out, size_t count) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    out[i] = __nv_fp8_e4m3(float(M[i])).__x;
}

// Four Russians over k in [k0, k0 + kk): D[r][c] = sum_{k0<=j<k0+kk} S[A[r][j]]·S[B[c][j]], written with '='.
// A CTA: COLS columns × FR_RB rows, a thread FR_RT rows × COLS columns of int32 accumulators. For each chunk of w <= b
// consecutive k, the CTA builds T[p][c] (int16) for every pattern p in [0, m^w) and its COLS columns, then each of its
// rows adds T[p(r)][c]. mp_max = m^b sizes the dynamic shared table.
constexpr int FR_RT = 4, FR_RB = 256 * FR_RT;
template <int COLS>
__global__ void __launch_bounds__(256) k_fr(const uint8_t* __restrict__ AI, const uint8_t* __restrict__ BI,
                                            int32_t* __restrict__ D, int n, int k, int k0, int kk, int b, int m,
                                            Vals s) {
  extern __shared__ int16_t T[];
  const int tid = threadIdx.x, c0 = blockIdx.x * COLS, rbase = blockIdx.y * FR_RB;
  int acc[FR_RT][COLS];
#pragma unroll
  for (int i = 0; i < FR_RT; ++i)
#pragma unroll
    for (int c = 0; c < COLS; ++c) acc[i][c] = 0;
  for (int base = k0; base < k0 + kk; base += b) {
    const int w = min(b, k0 + kk - base);
    int mp = 1;
    for (int t = 0; t < w; ++t) mp *= m;
    for (int e = tid; e < mp * COLS; e += 256) {
      const int c = e % COLS;
      int p = e / COLS, acc_e = 0;
      const uint8_t* bc = BI + size_t(c0 + c) * k + base;
      for (int t = 0; t < w; ++t) {
        acc_e += s.v[p % m] * s.v[bc[t]];
        p /= m;
      }
      T[e] = int16_t(acc_e);
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

__global__ void k_add(int32_t* __restrict__ X, const int32_t* __restrict__ Y, size_t count) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    X[i] += Y[i];
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

// Our D (n x n, row-major) = A[:, :kc] · B[:, :kc]^T, A and B n x n row-major (leading dimension n).
static Plan make_plan(cublasLtHandle_t h, Kind kind, int n, int kc, size_t ws) {
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
  LTC(cublasLtMatrixLayoutCreate(&p.a, in, kc, n, n));
  LTC(cublasLtMatrixLayoutCreate(&p.b, in, kc, n, n));
  LTC(cublasLtMatrixLayoutCreate(&p.c, out, n, n, n));
  LTC(cublasLtMatmulPreferenceCreate(&p.pref));
  LTC(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  return p;
}

static void run(Plan& p, const cublasLtMatmulAlgo_t* algo, const void* A, const void* B, void* D, void* ws,
                cudaStream_t st = 0) {
  int32_t ia = 1, ib = 0;
  float fa = 1.f, fb = 0.f;
  const void* al = p.kind == K_INT8 ? static_cast<const void*>(&ia) : &fa;
  const void* be = p.kind == K_INT8 ? static_cast<const void*>(&ib) : &fb;
  LTC(cublasLtMatmul(p.h, p.op, al, B, p.a, A, p.b, be, D, p.c, D, p.c, algo, ws, p.ws, st));
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
  if (hs != CUBLAS_STATUS_SUCCESS || got == 0) {
    printf("{\"event\":\"unsupported\",\"route\":\"%s\",\"cublasStatus\":%d,\"candidates\":%d}\n", kind_name[p.kind],
           int(hs), got);
    return false;
  }
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

// Columns a CTA covers at table width b: as many as the int16 table allows in 96 KiB, at most 32.
static int fr_cols(int m, int b) {
  long mp = 1;
  for (int t = 0; t < b; ++t) mp *= m;
  for (int cols : {32, 16, 8})
    if (mp * cols * 2 <= 96 * 1024) return cols;
  return 0;
}

static void launch_fr(int cols, const uint8_t* AI, const uint8_t* BI, int32_t* D, int n, int k0, int kk, int b,
                      int m, Vals s, cudaStream_t st) {
  long mp = 1;
  for (int t = 0; t < b; ++t) mp *= m;
  const size_t sh = size_t(mp) * cols * 2;
  dim3 grid(n / cols, n / FR_RB);
  if (cols == 32) k_fr<32><<<grid, 256, sh, st>>>(AI, BI, D, n, n, k0, kk, b, m, s);
  else if (cols == 16) k_fr<16><<<grid, 256, sh, st>>>(AI, BI, D, n, n, k0, kk, b, m, s);
  else k_fr<8><<<grid, 256, sh, st>>>(AI, BI, D, n, n, k0, kk, b, m, s);
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: lookup_coissue <n> <alphabet> <seed> <rounds> <bs> <alphas16>\n");
    return 2;
  }
  const int n = atoi(argv[1]);
  const Alphabet* al = nullptr;
  for (const Alphabet& a : ALPHABETS)
    if (!strcmp(a.name, argv[2])) al = &a;
  const uint64_t seed = strtoull(argv[3], nullptr, 0);
  const int rounds = atoi(argv[4]);
  const std::vector<int> bs = parse_list(argv[5]), alphas = parse_list(argv[6]);
  if (!al || n % FR_RB) {
    fprintf(stderr, "bad alphabet or n (n must be a multiple of %d)\n", FR_RB);
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
  for (auto f : {k_fr<32>, k_fr<16>, k_fr<8>})
    CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));

  Vals sv{};
  for (int i = 0; i < al->m; ++i) sv.v[i] = al->v[i];
  uint8_t *AI, *BI;
  int8_t *A, *B;
  CK(cudaMalloc(&AI, N2));
  CK(cudaMalloc(&BI, N2));
  CK(cudaMalloc(&A, N2));
  CK(cudaMalloc(&B, N2));
  k_gen<<<1024, 256>>>(AI, A, N2, seed, al->m, sv);
  k_gen<<<1024, 256>>>(BI, B, N2, seed ^ 0xB0B0B0B0B0B0B0B0ull, al->m, sv);
  CK(cudaGetLastError());

  int32_t *Dref, *Di, *Dl;
  float* Df;
  CK(cudaMalloc(&Dref, N2 * 4));
  CK(cudaMalloc(&Di, N2 * 4));
  CK(cudaMalloc(&Dl, N2 * 4));
  CK(cudaMalloc(&Df, N2 * 4));
  const size_t WS = size_t(64) << 20;
  void *ws, *ws2;
  CK(cudaMalloc(&ws, WS));
  CK(cudaMalloc(&ws2, WS));
  unsigned long long* dbad;
  CK(cudaMalloc(&dbad, 8));
  cublasLtHandle_t h;
  LTC(cublasLtCreate(&h));

  printf("{\"event\":\"start\",\"n\":%d,\"alphabet\":\"%s\",\"m\":%d,\"seed\":%llu,\"gpu\":\"%s\",\"uuid\":\"%s\","
         "\"sms\":%d,\"cublasLt\":%zu,\"sm_clock_mhz\":%u}\n",
         n, al->name, al->m, (unsigned long long)seed, prop.name, uuid, prop.multiProcessorCount,
         cublasLtGetVersion(), sm_clock());

  Plan p8 = make_plan(h, K_INT8, n, n, WS);
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
      const int8_t* a = &hA[r * n];
      const int8_t* b = &hB[c * n];
      for (int j = 0; j < n; ++j) s += int64_t(a[j]) * int64_t(b[j]);
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

  {  // control: int8 with one entry of A changed to another alphabet value must fail the comparison
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
    double ms1 = 0;
    double alpha = 0;
    int b = 0;
  };
  std::vector<Route> routes;
  routes.push_back({"int8", [&] { run(p8, &p8.algo, A, B, Di, ws); }});

  uint8_t *A8, *B8;
  CK(cudaMalloc(&A8, N2));
  CK(cudaMalloc(&B8, N2));
  k_fp8<<<1024, 256>>>(A, A8, N2);
  k_fp8<<<1024, 256>>>(B, B8, N2);
  Plan pf8 = make_plan(h, K_FP8, n, n, WS);
  std::string tlf8;
  if (autotune(pf8, A8, B8, Df, ws, tlf8)) {
    CK(cudaMemset(Df, 0xA5, N2 * 4));
    run(pf8, &pf8.algo, A8, B8, Df, ws);
    CK(cudaDeviceSynchronize());
    unsigned long long b = cmp_f(Df);
    printf("{\"event\":\"gate\",\"route\":\"fp8_e4m3\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu,"
           "\"candidates\":%d,\"best\":%d,\"autotune_ms\":[%s]}\n",
           N2, b, pf8.ncand, pf8.best, tlf8.c_str());
    if (!b) routes.push_back({"fp8_e4m3", [&] { run(pf8, &pf8.algo, A8, B8, Df, ws); }});
  }

  // Standalone Four Russians at each table width b; the fastest one (one untimed-order call) is the hybrid's.
  int hb = 0;
  float hb_ms = 1e30f;
  for (int b : bs) {
    const int cols = fr_cols(al->m, b);
    if (!cols) {
      printf("{\"event\":\"skip\",\"route\":\"fr_b%d\",\"reason\":\"table over 96 KiB\"}\n", b);
      continue;
    }
    CK(cudaMemset(Dl, 0xA5, N2 * 4));
    launch_fr(cols, AI, BI, Dl, n, 0, n, b, al->m, sv, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    unsigned long long bad = cmp_i(Dl);
    char name[32];
    snprintf(name, sizeof name, "fr_b%d", b);
    printf("{\"event\":\"gate\",\"route\":\"%s\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu,"
           "\"cols\":%d}\n",
           name, N2, bad, cols);
    if (!bad) {
      Route r{name, [=, &sv] { launch_fr(cols, AI, BI, Dl, n, 0, n, b, al->m, sv, 0); }};
      r.b = b;
      r.alpha = 1;
      float t = time_calls(r.f, 1);
      if (t < hb_ms) {
        hb_ms = t;
        hb = b;
      }
      routes.push_back(r);
    }
  }

  // The co-issue hybrid: int8 over [0, k1) on s1 and Four Russians over [k1, n) on s2, at once, then D = D1 + D2.
  const int hcols = hb ? fr_cols(al->m, hb) : 0;
  printf("{\"event\":\"hybrid_b\",\"b\":%d,\"cols\":%d,\"fr_ms\":%.4f}\n", hb, hcols, hb ? hb_ms : 0.f);
  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  cudaEvent_t ev0, ev1, ev2;
  CK(cudaEventCreateWithFlags(&ev0, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming));
  std::vector<Plan> hplans;
  hplans.reserve(alphas.size());
  for (int j : alphas) {
    if (!hcols) break;
    const int k1 = (n - int(double(n) * j / 16.0)) / 256 * 256, k2 = n - k1;
    if (k1 <= 0 || k2 <= 0) continue;
    hplans.push_back(make_plan(h, K_INT8, n, k1, WS));
    Plan& hp = hplans.back();
    std::string tl;
    if (!autotune(hp, A, B, Di, ws, tl)) continue;
    auto hyb = [&hp, &sv, A, B, AI, BI, Di, Dl, k1, k2, n, hb, hcols, al, ws, s1, s2, ev0, ev1, ev2, N2] {
      CK(cudaEventRecord(ev0, 0));
      CK(cudaStreamWaitEvent(s1, ev0, 0));
      CK(cudaStreamWaitEvent(s2, ev0, 0));
      run(hp, &hp.algo, A, B, Di, ws, s1);
      launch_fr(hcols, AI, BI, Dl, n, k1, k2, hb, al->m, sv, s2);
      CK(cudaEventRecord(ev1, s1));
      CK(cudaEventRecord(ev2, s2));
      CK(cudaStreamWaitEvent(0, ev1, 0));
      CK(cudaStreamWaitEvent(0, ev2, 0));
      k_add<<<1024, 256>>>(Di, Dl, N2);
    };
    // the two halves alone, for the overlap diagnostic
    auto tens = [&hp, A, B, Di, ws] { run(hp, &hp.algo, A, B, Di, ws); };
    auto look = [&sv, AI, BI, Dl, k1, k2, n, hb, hcols, al] { launch_fr(hcols, AI, BI, Dl, n, k1, k2, hb, al->m, sv, 0); };
    CK(cudaMemset(Di, 0xA5, N2 * 4));
    CK(cudaMemset(Dl, 0xA5, N2 * 4));
    hyb();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    unsigned long long bad = cmp_i(Di);
    char name[32];
    snprintf(name, sizeof name, "hyb_a%d", j);
    printf("{\"event\":\"gate\",\"route\":\"%s\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu,"
           "\"k_tensor\":%d,\"k_lookup\":%d,\"b\":%d,\"cols\":%d,\"candidates\":%d,\"best\":%d}\n",
           name, N2, bad, k1, k2, hb, hcols, hp.ncand, hp.best);
    if (!bad) {
      Route r{name, hyb};
      r.alpha = double(k2) / n;
      r.b = hb;
      routes.push_back(r);
      Route rt{std::string(name) + "_tensor_part", tens};
      rt.alpha = double(k2) / n;
      routes.push_back(rt);
      Route rl{std::string(name) + "_lookup_part", look};
      rl.alpha = double(k2) / n;
      rl.b = hb;
      routes.push_back(rl);
    }
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
  const double base = med(per[0]);
  const double macs = double(n) * n * n;
  for (size_t i = 0; i < routes.size(); ++i) {
    double m = med(per[i]), mn = *std::min_element(per[i].begin(), per[i].end());
    unsigned cmin = *std::min_element(clk[i].begin(), clk[i].end()),
             cmax = *std::max_element(clk[i].begin(), clk[i].end());
    printf("{\"event\":\"summary\",\"n\":%d,\"alphabet\":\"%s\",\"route\":\"%s\",\"alpha\":%.4f,\"b\":%d,"
           "\"median_ms\":%.5f,\"min_ms\":%.5f,\"ratio_vs_int8\":%.4f,\"macs_per_sm_clk_at_2092\":%.1f,"
           "\"clk_min\":%u,\"clk_max\":%u}\n",
           n, al->name, routes[i].name.c_str(), routes[i].alpha, routes[i].b, m, mn, m / base,
           macs / (m * 1e-3) / (prop.multiProcessorCount * 2092e6), cmin, cmax);
  }
  return 0;
}
