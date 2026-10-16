// The core-plus-correction (FP4 hybrid) route to the exact integer product D = A·Bᵀ of two random n×n noise matrices whose
// entries are i.i.d. uniform on a small alphabet FP4 can't hold (u5 = {-5..5}, {±1,±2,±4,±5}, u8 = {-8..8}), on one
// sm_120 GPU, against the plain int8 and FP8 cuBLASLt GEMMs on the same matrices.
//
// The method (note:20261016T0016Z-draft-fullfloor-s-r3): split each entry x = C(x) + Δ(x), with a constant per-operand
// shift t chosen to minimise the out-of-grid density δ. C lands on E2M1's integer grid {0,±1,±2,±3,±4,±6} (one NVFP4 MMA
// at 2× int8's rate), Δ is a small sparse correction (±1, ±2) on the δ fraction of entries. Writing P = C + Δ = A + t·J,
//   A·Bᵀ = C_A C_Bᵀ + C_A Δ_Bᵀ + Δ_A C_Bᵀ + Δ_A Δ_Bᵀ − t_B·rowsum(P_A) − t_A·rowsum(P_B) + n·t_A·t_B,
// where the three Δ terms are the corrections and the last three are a rank-1 O(n²) epilogue. Two correction backends:
//   method_densefp4   : the three Δ terms as dense FP4 GEMMs (Δ is E2M1-representable), i.e. four FP4 GEMMs total.
//   method_sparsecuda : the two dense×sparse Δ terms as CSR-sparse × dense products on the CUDA cores (Δ_A Δ_Bᵀ, tiny,
//                       folded into the sparse pass against a dense Δ transpose), i.e. one FP4 MMA plus CUDA-core sparse.
//
// Gates before timing, in the same process: int8's output against a CPU int64 reference (every entry at n ≤ 1024, 4096
// sampled entries otherwise); then each method's output against int8's, bit for bit; a method whose output mismatches is
// reported and not timed. A control (the method with its corrections zeroed) must mismatch. Timing: cuBLASLt autotuned
// over its top heuristic candidates, then rounds of interleaved items of about equal device time, the SM clock read by
// NVML before and after each item; the FP4 core and the corrections are also timed on their own for the split. JSON lines.
//
//   hybrid <n> <alphabet> <seed> <rounds>      alphabet: u5 s1245 u8
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

// A noise alphabet, the shift t that minimises its out-of-E2M1 density, and (filled in main) the per-value core/correction.
struct Alphabet {
  const char* name;
  int n;
  int v[17];
  int shift;
};
static Alphabet ALPHABETS[] = {
    {"u5", 11, {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}, 1},
    {"s1245", 8, {-5, -4, -2, -1, 1, 2, 4, 5}, 2},
    {"u8", 17, {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}, 0},
    {"u4", 9, {-4, -3, -2, -1, 0, 1, 2, 3, 4}, 0},  // control: FP4 holds it, δ = 0 (no correction)
};

struct Table17 {
  int8_t v[17];
};

__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__global__ void k_gen(int8_t* M, size_t count, uint64_t seed, int na, Table17 t) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    M[i] = t.v[mix64(seed ^ (i * 0xD1B54A32D192ED03ull)) % uint64_t(na)];
}

__global__ void k_map(const int8_t* M, int8_t* out, size_t count, Table17 t) {  // out = t[M + 8]
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    out[i] = t.v[M[i] + 8];
}

__global__ void k_fp8(const int8_t* M, uint8_t* out, size_t count) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    out[i] = __nv_fp8_e4m3(float(M[i])).__x;
}

// E2M1 code of an integer at scale 1: magnitudes 0 1 2 3 4 6 are codes 0 2 4 5 6 7; sign bit 8. Anything else flags.
__device__ inline uint32_t e2m1_code(int v, int* bad) {
  int m = v < 0 ? -v : v;
  uint32_t c;
  switch (m) {
    case 0: c = 0; break;
    case 1: c = 2; break;
    case 2: c = 4; break;
    case 3: c = 5; break;
    case 4: c = 6; break;
    case 6: c = 7; break;
    default: atomicAdd(bad, 1); c = 0;
  }
  return c | (v < 0 ? 8u : 0u);
}

__global__ void k_fp4(const int8_t* M, uint8_t* out, size_t pairs, int* bad) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < pairs; i += size_t(gridDim.x) * blockDim.x)
    out[i] = uint8_t(e2m1_code(M[2 * i], bad) | (e2m1_code(M[2 * i + 1], bad) << 4));
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

// The rank-1 shift epilogue: D[i,j] -= t_B·rP_A[i] + t_A·rP_B[j] - n·t_A·t_B, on the FP32 accumulator from the GEMMs.
__global__ void k_shift_epilogue(float* D, const int32_t* rPA, const int32_t* rPB, int n, int tA, int tB) {
  size_t total = size_t(n) * n;
  for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total; idx += size_t(gridDim.x) * blockDim.x) {
    int i = int(idx / n), j = int(idx % n);
    D[idx] -= float(tB) * float(rPA[i]) + float(tA) * float(rPB[j]) - float(n) * float(tA) * float(tB);
  }
}

// CSR sparse × dense on the CUDA cores: out[i,j] += Σ_{(k,v) ∈ row i of S} v · Bt[k,j], Bt row-major (Bt[k,j] contiguous
// in j). One warp per output row; each of 32 lanes keeps COLS_PER_LANE accumulators in registers for a 32·COLS_PER_LANE
// column panel, loops the row's nonzeros (coalesced reads of Bt[k, panel]) and writes the panel once. This is the faithful
// cost of delivering a data-dependent correction operand per entry (the note's κ, measured).
constexpr int COLS_PER_LANE = 8;  // a warp covers 256 columns a pass
__global__ void k_spmm_add(const int* __restrict__ rowptr, const int* __restrict__ colidx,
                           const int8_t* __restrict__ vals, const int8_t* __restrict__ Bt, float* __restrict__ out,
                           int n) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int row = warp % n;
  const int panel = (warp / n) * (32 * COLS_PER_LANE);
  if (panel >= n) return;
  float acc[COLS_PER_LANE];
#pragma unroll
  for (int c = 0; c < COLS_PER_LANE; ++c) acc[c] = 0.f;
  const int lo = rowptr[row], hi = rowptr[row + 1];
  for (int e = lo; e < hi; ++e) {
    const int k = colidx[e];
    const float v = float(vals[e]);
    const int8_t* b = Bt + size_t(k) * n + panel + lane;
#pragma unroll
    for (int c = 0; c < COLS_PER_LANE; ++c) acc[c] += v * float(b[c * 32]);
  }
  float* o = out + size_t(row) * n + panel + lane;
#pragma unroll
  for (int c = 0; c < COLS_PER_LANE; ++c) o[c * 32] += acc[c];
}

// Add a transposed buffer into the accumulator: out[i,j] += T[j,i]. For the C_A Δ_Bᵀ term, whose natural sparse layout is
// indexed [j,i].
__global__ void k_transpose_add(float* out, const float* T, int n) {
  size_t total = size_t(n) * n;
  for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total; idx += size_t(gridDim.x) * blockDim.x) {
    int i = int(idx / n), j = int(idx % n);
    out[idx] += T[size_t(j) * n + i];
  }
}

enum Kind { K_INT8, K_FP8, K_NVFP4 };
static const char* kind_name[] = {"int8", "fp8_e4m3", "nvfp4"};

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

// Our D (n x n, row-major) = A (n x k) . B (n x k)^T, k = n.
static Plan make_plan(cublasLtHandle_t h, Kind kind, int n, size_t ws) {
  Plan p{};
  p.h = h;
  p.kind = kind;
  p.ws = ws;
  bool i8 = kind == K_INT8;
  cudaDataType_t in = kind == K_INT8 ? CUDA_R_8I : kind == K_FP8 ? CUDA_R_8F_E4M3 : CUDA_R_4F_E2M1;
  cudaDataType_t out = i8 ? CUDA_R_32I : CUDA_R_32F;
  LTC(cublasLtMatmulDescCreate(&p.op, i8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F, i8 ? CUDA_R_32I : CUDA_R_32F));
  cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
  if (kind == K_NVFP4) {
    int32_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof sm));
    LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof sm));
  }
  LTC(cublasLtMatrixLayoutCreate(&p.a, in, n, n, n));  // our B (n x k row-major) = k x n column-major, op T
  LTC(cublasLtMatrixLayoutCreate(&p.b, in, n, n, n));  // our A, n x k row-major
  LTC(cublasLtMatrixLayoutCreate(&p.c, out, n, n, n));
  LTC(cublasLtMatmulPreferenceCreate(&p.pref));
  LTC(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  return p;
}

static void set_scales(Plan& p, const void* sa, const void* sb) {
  if (p.kind != K_NVFP4) return;
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sb, sizeof sb));
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa, sizeof sa));
}

static void run(Plan& p, const cublasLtMatmulAlgo_t* algo, const void* A, const void* B, void* D, const void* sa,
                const void* sb, void* ws, float alpha = 1.f, float beta = 0.f, cudaStream_t st = 0) {
  set_scales(p, sa, sb);
  int32_t ia = int32_t(alpha), ib = int32_t(beta);
  const void* al = p.kind == K_INT8 ? static_cast<const void*>(&ia) : &alpha;
  const void* be = p.kind == K_INT8 ? static_cast<const void*>(&ib) : &beta;
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

// The fastest of cuBLASLt's top heuristic candidates for this plan (2 warm-up calls, then the best of 3 x 3 calls).
static bool autotune(Plan& p, const void* A, const void* B, void* D, const void* sa, const void* sb, void* ws,
                     std::string& log) {
  set_scales(p, sa, sb);
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
    auto f = [&] { run(p, &res[i].algo, A, B, D, sa, sb, ws); };
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
  if (p.best < 0) {
    printf("{\"event\":\"unsupported\",\"route\":\"%s\",\"reason\":\"no candidate within workspace\"}\n",
           kind_name[p.kind]);
    return false;
  }
  return true;
}

static nvmlDevice_t g_nvml;
static bool g_nvml_ok = false;
static unsigned sm_clock() {
  unsigned c = 0;
  if (g_nvml_ok) nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &c);
  return c;
}

static Alphabet* find_alphabet(const char* s) {
  for (Alphabet& a : ALPHABETS)
    if (!strcmp(a.name, s)) return &a;
  return nullptr;
}

static bool e2m1_int(int m) {
  m = abs(m);
  return m <= 4 || m == 6;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: hybrid <n> <alphabet> <seed> <rounds>\n");
    return 2;
  }
  const int n = atoi(argv[1]);
  Alphabet* al = find_alphabet(argv[2]);
  const uint64_t seed = strtoull(argv[3], nullptr, 0);
  const int rounds = atoi(argv[4]);
  if (!al || n % 256) {
    fprintf(stderr, "bad alphabet or n (n must be a multiple of 256)\n");
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

  // Per-value core C(x) and correction Δ(x) = (x + shift) − C(x), with C on the E2M1 integer grid and |Δ| minimal.
  const int tA = al->shift, tB = al->shift;
  Table17 coreTab{}, corrTab{};
  int out_count = 0;
  for (int idx = 0; idx < al->n; ++idx) {
    int x = al->v[idx];
    int y = x + al->shift;
    int core, corr;
    if (e2m1_int(y)) {
      core = y;
      corr = 0;
    } else {  // nearest E2M1 integer among {0,±1,±2,±3,±4,±6}, ties to the one giving the smaller |corr|
      int best = 0, bestd = 1 << 30;
      for (int g = -6; g <= 6; ++g)
        if (e2m1_int(g) && abs(y - g) < bestd) {
          bestd = abs(y - g);
          best = g;
        }
      core = best;
      corr = y - core;
      ++out_count;
    }
    coreTab.v[x + 8] = int8_t(core);
    corrTab.v[x + 8] = int8_t(corr);
  }
  const double delta = double(out_count) / al->n;

  int8_t *A, *B;
  CK(cudaMalloc(&A, N2));
  CK(cudaMalloc(&B, N2));
  Table17 tv{};
  for (int i = 0; i < al->n; ++i) tv.v[i] = int8_t(al->v[i]);
  k_gen<<<1024, 256>>>(A, N2, seed, al->n, tv);
  k_gen<<<1024, 256>>>(B, N2, seed ^ 0xB0B0B0B0B0B0B0B0ull, al->n, tv);
  CK(cudaGetLastError());

  int32_t *Dref, *Di;
  float* Df;
  CK(cudaMalloc(&Dref, N2 * 4));
  CK(cudaMalloc(&Df, N2 * 4));
  CK(cudaMalloc(&Di, N2 * 4));
  const size_t WS = size_t(64) << 20;
  void* ws;
  CK(cudaMalloc(&ws, WS));
  unsigned long long* dbad;
  CK(cudaMalloc(&dbad, 8));
  cublasLtHandle_t h;
  LTC(cublasLtCreate(&h));

  printf("{\"event\":\"start\",\"n\":%d,\"alphabet\":\"%s\",\"shift\":%d,\"delta\":%.5f,\"out_values\":%d,\"seed\":%llu,"
         "\"gpu\":\"%s\",\"uuid\":\"%s\",\"sms\":%d,\"cublasLt\":%zu,\"sm_clock_mhz\":%u}\n",
         n, al->name, al->shift, delta, out_count, (unsigned long long)seed, prop.name, uuid,
         prop.multiProcessorCount, cublasLtGetVersion(), sm_clock());

  // int8 baseline and its CPU int64 check.
  Plan p8 = make_plan(h, K_INT8, n, WS);
  std::string tl8;
  if (!autotune(p8, A, B, Dref, nullptr, nullptr, ws, tl8)) return 5;
  run(p8, &p8.algo, A, B, Dref, nullptr, nullptr, ws);
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
      for (int k = 0; k < n; ++k) s += int(a[k]) * int(b[k]);
      ++checked;
      bad += s != hD[r * n + c];
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
    printf("{\"event\":\"gate\",\"route\":\"int8\",\"against\":\"cpu\",\"checked\":%zu,\"mismatches\":%zu,"
           "\"candidates\":%d,\"best\":%d,\"autotune_ms\":[%s]}\n",
           checked, bad, p8.ncand, p8.best, tl8.c_str());
    if (bad) return 6;
  }

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
  };
  std::vector<Route> routes;
  routes.push_back({"int8", [&] { run(p8, &p8.algo, A, B, Di, nullptr, nullptr, ws); }});

  // FP8 E4M3 baseline, FP32 out.
  uint8_t *A8, *B8;
  CK(cudaMalloc(&A8, N2));
  CK(cudaMalloc(&B8, N2));
  k_fp8<<<1024, 256>>>(A, A8, N2);
  k_fp8<<<1024, 256>>>(B, B8, N2);
  Plan pf8 = make_plan(h, K_FP8, n, WS);
  std::string tlf8;
  float* Df8;
  CK(cudaMalloc(&Df8, N2 * 4));
  if (autotune(pf8, A8, B8, Df8, nullptr, nullptr, ws, tlf8)) {
    CK(cudaMemset(Df8, 0xA5, N2 * 4));
    run(pf8, &pf8.algo, A8, B8, Df8, nullptr, nullptr, ws);
    CK(cudaDeviceSynchronize());
    unsigned long long bf8 = cmp_f(Df8);
    printf("{\"event\":\"gate\",\"route\":\"fp8_e4m3\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu,"
           "\"candidates\":%d,\"best\":%d,\"autotune_ms\":[%s]}\n",
           N2, bf8, pf8.ncand, pf8.best, tlf8.c_str());
    if (!bf8) routes.push_back({"fp8_e4m3", [&] { run(pf8, &pf8.algo, A8, B8, Df8, nullptr, nullptr, ws); }});
  }

  // Build the core C and correction Δ for A and B as int8 matrices; pack C and Δ to E2M1; compute rowsums of P = C + Δ.
  int8_t *CAi, *CBi, *DAi, *DBi;
  CK(cudaMalloc(&CAi, N2));
  CK(cudaMalloc(&CBi, N2));
  CK(cudaMalloc(&DAi, N2));
  CK(cudaMalloc(&DBi, N2));
  k_map<<<1024, 256>>>(A, CAi, N2, coreTab);
  k_map<<<1024, 256>>>(B, CBi, N2, coreTab);
  k_map<<<1024, 256>>>(A, DAi, N2, corrTab);
  k_map<<<1024, 256>>>(B, DBi, N2, corrTab);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  int* dflag;
  CK(cudaMalloc(&dflag, 4));
  auto pack4 = [&](const int8_t* M) {
    uint8_t* o;
    CK(cudaMalloc(&o, N2 / 2));
    CK(cudaMemset(dflag, 0, 4));
    k_fp4<<<1024, 256>>>(M, o, N2 / 2, dflag);
    int bad;
    CK(cudaMemcpy(&bad, dflag, 4, cudaMemcpyDeviceToHost));
    if (bad) {
      fprintf(stderr, "e2m1 pack: %d values off the grid\n", bad);
      exit(8);
    }
    return o;
  };
  uint8_t* CA4 = pack4(CAi);
  uint8_t* CB4 = pack4(CBi);
  uint8_t* DA4 = pack4(DAi);
  uint8_t* DB4 = pack4(DBi);

  // rowsum(P) on the host from P = C + Δ (small ints), uploaded.
  std::vector<int8_t> hCA(N2), hCB(N2), hDA(N2), hDB(N2);
  CK(cudaMemcpy(hCA.data(), CAi, N2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hCB.data(), CBi, N2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hDA.data(), DAi, N2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hDB.data(), DBi, N2, cudaMemcpyDeviceToHost));
  std::vector<int32_t> rPA(n), rPB(n);
  for (int i = 0; i < n; ++i) {
    int64_t sa = 0, sb = 0;
    for (int k = 0; k < n; ++k) {
      sa += int(hCA[size_t(i) * n + k]) + int(hDA[size_t(i) * n + k]);
      sb += int(hCB[size_t(i) * n + k]) + int(hDB[size_t(i) * n + k]);
    }
    rPA[i] = int32_t(sa);
    rPB[i] = int32_t(sb);
  }
  int32_t *dRPA, *dRPB;
  CK(cudaMalloc(&dRPA, n * 4));
  CK(cudaMalloc(&dRPB, n * 4));
  CK(cudaMemcpy(dRPA, rPA.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dRPB, rPB.data(), n * 4, cudaMemcpyHostToDevice));

  const uint8_t SCALE = 0x38;  // UE4M3 for 1.0
  uint8_t* s_nv;
  CK(cudaMalloc(&s_nv, N2 / 16));
  CK(cudaMemset(s_nv, SCALE, N2 / 16));
  Plan pnv = make_plan(h, K_NVFP4, n, WS);
  std::string tlnv;
  if (!autotune(pnv, CA4, CB4, Df, s_nv, s_nv, ws, tlnv)) return 5;

  auto epilogue = [&] { k_shift_epilogue<<<2048, 256>>>(Df, dRPA, dRPB, n, tA, tB); };

  // method_densefp4: four FP4 GEMMs (core + three corrections) accumulated into Df, then the rank-1 epilogue.
  auto core_gemm = [&] { run(pnv, &pnv.algo, CA4, CB4, Df, s_nv, s_nv, ws, 1.f, 0.f); };
  auto corr_dense = [&] {
    run(pnv, &pnv.algo, CA4, DB4, Df, s_nv, s_nv, ws, 1.f, 1.f);
    run(pnv, &pnv.algo, DA4, CB4, Df, s_nv, s_nv, ws, 1.f, 1.f);
    run(pnv, &pnv.algo, DA4, DB4, Df, s_nv, s_nv, ws, 1.f, 1.f);
    epilogue();
  };
  auto method_densefp4 = [&] {
    core_gemm();
    corr_dense();
  };

  // CSR of Δ_A and Δ_B (nonzeros ±1/±2), and dense transposes Bt[k,j] = X[j,k] for the sparse correction passes.
  auto build_csr = [&](const std::vector<int8_t>& D, std::vector<int>& rp, std::vector<int>& ci,
                       std::vector<int8_t>& vv) {
    rp.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
      int c = 0;
      for (int k = 0; k < n; ++k) c += D[size_t(i) * n + k] != 0;
      rp[i + 1] = rp[i] + c;
    }
    ci.resize(rp[n]);
    vv.resize(rp[n]);
    for (int i = 0; i < n; ++i) {
      int p = rp[i];
      for (int k = 0; k < n; ++k)
        if (D[size_t(i) * n + k]) {
          ci[p] = k;
          vv[p] = D[size_t(i) * n + k];
          ++p;
        }
    }
  };
  std::vector<int> rpA, ciA, rpB, ciB;
  std::vector<int8_t> vvA, vvB;
  build_csr(hDA, rpA, ciA, vvA);
  build_csr(hDB, rpB, ciB, vvB);
  auto up_i = [](const std::vector<int>& v) {
    int* d;
    CK(cudaMalloc(&d, std::max<size_t>(1, v.size()) * 4));
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return d;
  };
  auto up_b = [](const std::vector<int8_t>& v) {
    int8_t* d;
    CK(cudaMalloc(&d, std::max<size_t>(1, v.size())));
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice));
    return d;
  };
  int *dRpA = up_i(rpA), *dCiA = up_i(ciA), *dRpB = up_i(rpB), *dCiB = up_i(ciB);
  int8_t *dVvA = up_b(vvA), *dVvB = up_b(vvB);
  // Dense transposes CBt[k,j] = C_B[j,k], CAt[k,i] = C_A[i,k], DBt[k,j] = Δ_B[j,k].
  std::vector<int8_t> hCBt(N2), hCAt(N2), hDBt(N2);
  for (int r = 0; r < n; ++r)
    for (int k = 0; k < n; ++k) {
      hCBt[size_t(k) * n + r] = hCB[size_t(r) * n + k];
      hCAt[size_t(k) * n + r] = hCA[size_t(r) * n + k];
      hDBt[size_t(k) * n + r] = hDB[size_t(r) * n + k];
    }
  int8_t *dCBt, *dCAt, *dDBt;
  CK(cudaMalloc(&dCBt, N2));
  CK(cudaMalloc(&dCAt, N2));
  CK(cudaMalloc(&dDBt, N2));
  CK(cudaMemcpy(dCBt, hCBt.data(), N2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dCAt, hCAt.data(), N2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dDBt, hDBt.data(), N2, cudaMemcpyHostToDevice));

  float* Dtmp;
  CK(cudaMalloc(&Dtmp, N2 * 4));
  const int panels = (n + 32 * COLS_PER_LANE - 1) / (32 * COLS_PER_LANE);
  const long warps = long(n) * panels;
  const int SP_BLK = 256;
  const long sp_blocks = (warps * 32 + SP_BLK - 1) / SP_BLK;
  // T1 = Δ_A C_Bᵀ (+Df), T3 = Δ_A Δ_Bᵀ (+Df), T2 = C_A Δ_Bᵀ via spmm(Δ_B, CAt) into Dtmp then transpose-add.
  auto corr_sparse = [&] {
    k_spmm_add<<<sp_blocks, SP_BLK>>>(dRpA, dCiA, dVvA, dCBt, Df, n);
    k_spmm_add<<<sp_blocks, SP_BLK>>>(dRpA, dCiA, dVvA, dDBt, Df, n);
    CK(cudaMemsetAsync(Dtmp, 0, N2 * 4));
    k_spmm_add<<<sp_blocks, SP_BLK>>>(dRpB, dCiB, dVvB, dCAt, Dtmp, n);
    k_transpose_add<<<2048, 256>>>(Df, Dtmp, n);
    epilogue();
  };
  auto method_sparsecuda = [&] {
    core_gemm();
    corr_sparse();
  };

  // Gate each method against int8, bit for bit; a control with corrections zeroed must mismatch.
  auto gate = [&](const char* name, const std::function<void()>& f) {
    CK(cudaMemset(Df, 0xA5, N2 * 4));
    f();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    unsigned long long b = cmp_f(Df);
    printf("{\"event\":\"gate\",\"route\":\"%s\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu}\n", name, N2,
           b);
    if (!b) routes.push_back({name, f});
    return b == 0;
  };
  gate("method_densefp4", method_densefp4);
  gate("method_sparsecuda", method_sparsecuda);
  {  // control: core only (corrections dropped) must mismatch, since δ > 0
    CK(cudaMemset(Df, 0xA5, N2 * 4));
    core_gemm();
    epilogue();
    CK(cudaDeviceSynchronize());
    unsigned long long bc = cmp_f(Df);
    printf("{\"event\":\"control\",\"route\":\"core_only\",\"mismatches\":%llu,\"rejected\":%s}\n", bc,
           bc ? "true" : "false");
    if (!bc && out_count) return 7;  // a δ > 0 alphabet whose core-only product already matches means the gate is blind
  }

  // Split timing: the FP4 core and the correction backends on their own.
  core_gemm();
  CK(cudaDeviceSynchronize());
  double core_ms = time_calls(core_gemm, 5);
  double dense_corr_ms = time_calls(corr_dense, 5);
  double sparse_corr_ms = time_calls(corr_sparse, 3);
  printf("{\"event\":\"split\",\"n\":%d,\"alphabet\":\"%s\",\"core_ms\":%.5f,\"dense_corr_ms\":%.5f,"
         "\"sparse_corr_ms\":%.5f}\n",
         n, al->name, core_ms, dense_corr_ms, sparse_corr_ms);

  // Interleaved timing: items of about 25 ms of device time, one item per route a round, SM clock before and after.
  std::vector<double> ms1(routes.size());
  for (size_t i = 0; i < routes.size(); ++i) {
    routes[i].f();
    CK(cudaDeviceSynchronize());
    ms1[i] = time_calls(routes[i].f, 3);
  }
  std::vector<std::vector<double>> per(routes.size());
  std::vector<int> iters(routes.size());
  for (size_t i = 0; i < routes.size(); ++i) iters[i] = std::max(1, int(25.0 / ms1[i]));
  for (int rd = 0; rd < rounds; ++rd)
    for (size_t i = 0; i < routes.size(); ++i) {
      unsigned c0 = sm_clock();
      double ms = time_calls(routes[i].f, iters[i]);
      unsigned c1 = sm_clock();
      per[i].push_back(ms);
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
    printf("{\"event\":\"summary\",\"n\":%d,\"alphabet\":\"%s\",\"route\":\"%s\",\"median_ms\":%.5f,\"min_ms\":%.5f,"
           "\"ratio_vs_int8\":%.4f,\"int_macs_per_s\":%.4e}\n",
           n, al->name, routes[i].name.c_str(), m, mn, m / base, macs / (m * 1e-3));
  }
  return 0;
}
