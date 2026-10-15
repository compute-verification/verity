// Exact routes to the integer product D = A·Bᵀ of two random n×n noise matrices with entries from a small alphabet, on one
// sm_120 GPU: int8 IMMA (the baseline, int32 out), FP8 E4M3, NVFP4 (UE4M3 block scale) and MXFP4 (UE8M0) through
// cuBLASLt with FP32 out, the three-product Karatsuba FP4 split for alphabets no FP4 scale holds, a CUDA-core XOR+POPC
// binary GEMM for {±1} (bit-slicing) and a Four Russians GEMM for {±1} (shared-memory tables of 8-row partial sums).
//
// Gates before timing, in the same process: int8's output against a CPU reference (every entry at n <= 1024, 4096
// sampled entries otherwise), then every route's output against int8's bit for bit at the same n; a route that
// mismatches is reported and not timed. A control (FP8 with one entry of A changed) must mismatch.
// Timing: cuBLASLt autotuned over its top heuristic candidates; then rounds of interleaved items of about equal device
// time, one item per route a round, the SM clock read by NVML before and after each item. JSON lines on stdout.
//
//   attacks <n> <alphabet> <seed> <rounds>      alphabet: pm1 pm1pm3 pm5pm7 u2 u4 u8
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
  int n;
  int v[17];
};
static const Alphabet ALPHABETS[] = {
    {"pm1", 2, {-1, 1}},
    {"pm1pm3", 4, {-3, -1, 1, 3}},
    {"pm5pm7", 4, {-7, -5, 5, 7}},
    {"u2", 5, {-2, -1, 0, 1, 2}},
    {"u4", 9, {-4, -3, -2, -1, 0, 1, 2, 3, 4}},
    {"u8", 17, {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}},
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

// Sign plane along k: bit b of word w of row r is 1 when M[r][32w + b] < 0 (entries ±1).
__global__ void k_bits(const int8_t* M, uint32_t* out, size_t words) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t w = 0;
    const int8_t* p = M + 32 * i;
#pragma unroll
    for (int b = 0; b < 32; ++b) w |= uint32_t(p[b] < 0) << b;
    out[i] = w;
  }
}

// Bit-slicing on the CUDA cores: D[r][c] = K - 2 popc(a_r XOR b_c), a 128x128 tile a CTA, 8x8 outputs a thread.
constexpr int BT = 128, BKW = 8;
__global__ void __launch_bounds__(256) k_bin(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B,
                                             int32_t* __restrict__ D, int n, int kw, int K) {
  __shared__ uint32_t As[BKW][BT];
  __shared__ uint32_t Bs[BKW][BT];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int r0 = blockIdx.y * BT, c0 = blockIdx.x * BT;
  int acc[8][8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0;
  for (int w0 = 0; w0 < kw; w0 += BKW) {
    for (int i = threadIdx.x; i < BT * BKW; i += 256) {
      int r = i / BKW, w = i % BKW;
      As[w][r] = A[size_t(r0 + r) * kw + w0 + w];
      Bs[w][r] = B[size_t(c0 + r) * kw + w0 + w];
    }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < BKW; ++w) {
      uint32_t a[8], b[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) a[i] = As[w][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < 8; ++j) b[j] = Bs[w][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] += __popc(a[i] ^ b[j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) D[size_t(r0 + ty + 16 * i) * n + c0 + tx + 16 * j] = K - 2 * acc[i][j];
}

// Four Russians for {±1}: k in chunks of 8 (a byte of A's sign plane), 8 chunks a stage. For each chunk and the CTA's 16
// columns, T[mask][c] = 8 + sum_j s_j(mask) b_j(c) (s_j = -1 where bit j is set) is built in shared memory from two 4-bit
// half tables (one POPC per half-table entry, then one add per entry) and shared by the CTA's 1,024 rows. Each row adds
// T[a_byte][c] for its 16 columns as 4 packed-byte adds of one LDS.128; a byte never exceeds 8 chunks x 16 = 128, and
// the bytes widen into int32 once a stage. A tile of 1,024 rows x 16 columns, a thread 4 rows.
constexpr int FR_R = 4, FR_COLS = 16, FR_G = 8, FR_ROWS = 256 * FR_R;
__global__ void __launch_bounds__(256) k_4r(const uint8_t* __restrict__ A8, const uint8_t* __restrict__ B8,
                                            int32_t* __restrict__ D, int n, int kb) {
  __shared__ uint4 HT[FR_G][2][16];
  __shared__ uint4 T[FR_G][256];
  const int tid = threadIdx.x, c0 = blockIdx.x * FR_COLS, rbase = blockIdx.y * FR_ROWS;
  int acc[FR_R][FR_COLS];
#pragma unroll
  for (int rr = 0; rr < FR_R; ++rr)
#pragma unroll
    for (int c = 0; c < FR_COLS; ++c) acc[rr][c] = 0;
  for (int ch0 = 0; ch0 < kb; ch0 += FR_G) {
    {  // half tables: thread = (chunk g, half, nibble); entry 4 + (4 - 2 popc(nib ^ b_nibble)) in [0, 8]
      const int g = tid >> 5, half = (tid >> 4) & 1, nib = tid & 15;
      uint32_t w[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint32_t x = 0;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          uint32_t b = B8[size_t(c0 + 4 * u + v) * kb + ch0 + g];
          b = half ? b >> 4 : b & 15;
          x |= uint32_t(8 - 2 * __popc(nib ^ b)) << (8 * v);
        }
        w[u] = x;
      }
      HT[g][half][nib] = make_uint4(w[0], w[1], w[2], w[3]);
    }
    __syncthreads();
    for (int e = tid; e < FR_G * 256; e += 256) {
      const int g = e >> 8, m = e & 255;
      const uint4 lo = HT[g][0][m & 15], hi = HT[g][1][m >> 4];
      T[g][m] = make_uint4(lo.x + hi.x, lo.y + hi.y, lo.z + hi.z, lo.w + hi.w);
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < FR_R; ++rr) {
      const int r = rbase + rr * 256 + tid;
      const uint2 ab = *reinterpret_cast<const uint2*>(A8 + size_t(r) * kb + ch0);
      uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
      for (int g = 0; g < FR_G; ++g) {
        const uint32_t a = ((g < 4 ? ab.x : ab.y) >> (8 * (g & 3))) & 0xFF;
        const uint4 t = T[g][a];
        s0 += t.x;
        s1 += t.y;
        s2 += t.z;
        s3 += t.w;
      }
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        acc[rr][v] += int((s0 >> (8 * v)) & 0xFF) - 8 * FR_G;
        acc[rr][4 + v] += int((s1 >> (8 * v)) & 0xFF) - 8 * FR_G;
        acc[rr][8 + v] += int((s2 >> (8 * v)) & 0xFF) - 8 * FR_G;
        acc[rr][12 + v] += int((s3 >> (8 * v)) & 0xFF) - 8 * FR_G;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int rr = 0; rr < FR_R; ++rr)
#pragma unroll
    for (int c = 0; c < FR_COLS; ++c) D[size_t(rbase + rr * 256 + tid) * n + c0 + c] = acc[rr][c];
}

__global__ void k_bytes(const uint32_t* bits, uint8_t* out, size_t count) {  // bytes of the sign plane, in k order
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    out[i] = uint8_t(bits[i / 4] >> (8 * (i % 4)));
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

enum Kind { K_INT8, K_FP8, K_NVFP4, K_MXFP4 };
static const char* kind_name[] = {"int8", "fp8_e4m3", "nvfp4", "mxfp4"};

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
  cudaDataType_t in = kind == K_INT8 ? CUDA_R_8I : kind == K_FP8 ? CUDA_R_8F_E4M3 : CUDA_R_4F_E2M1;
  cudaDataType_t out = i8 ? CUDA_R_32I : CUDA_R_32F;
  LTC(cublasLtMatmulDescCreate(&p.op, i8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F, i8 ? CUDA_R_32I : CUDA_R_32F));
  cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
  if (kind == K_NVFP4 || kind == K_MXFP4) {
    int32_t sm = kind == K_NVFP4 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 : CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof sm));
    LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof sm));
  }
  LTC(cublasLtMatrixLayoutCreate(&p.a, in, n, n, n));  // our B (n x k row-major) = k x n column-major, op T
  LTC(cublasLtMatrixLayoutCreate(&p.b, in, n, n, n));  // our A
  LTC(cublasLtMatrixLayoutCreate(&p.c, out, n, n, n));
  LTC(cublasLtMatmulPreferenceCreate(&p.pref));
  LTC(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  return p;
}

static void set_scales(Plan& p, const void* sa, const void* sb) {
  if (p.kind != K_NVFP4 && p.kind != K_MXFP4) return;
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sb, sizeof sb));
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa, sizeof sa));
}

static void run(Plan& p, const cublasLtMatmulAlgo_t* algo, const void* A, const void* B, void* D, const void* sa,
                const void* sb, void* ws, float alpha = 1.f, float beta = 0.f) {
  set_scales(p, sa, sb);
  int32_t ia = int32_t(alpha), ib = int32_t(beta);
  const void* al = p.kind == K_INT8 ? static_cast<const void*>(&ia) : &alpha;
  const void* be = p.kind == K_INT8 ? static_cast<const void*>(&ib) : &beta;
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

static const Alphabet* find_alphabet(const char* s) {
  for (const Alphabet& a : ALPHABETS)
    if (!strcmp(a.name, s)) return &a;
  return nullptr;
}

static bool fp4_holds(const Alphabet& a) {
  for (int i = 0; i < a.n; ++i) {
    int m = abs(a.v[i]);
    if (!(m <= 4 || m == 6)) return false;
  }
  return true;
}
static bool e2m1_int(int m) { m = abs(m); return m <= 4 || m == 6; }

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: attacks <n> <alphabet> <seed> <rounds>\n");
    return 2;
  }
  const int n = atoi(argv[1]);
  const Alphabet* al = find_alphabet(argv[2]);
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

  Table17 tv{};
  for (int i = 0; i < al->n; ++i) tv.v[i] = int8_t(al->v[i]);
  int8_t *A, *B;
  CK(cudaMalloc(&A, N2));
  CK(cudaMalloc(&B, N2));
  k_gen<<<1024, 256>>>(A, N2, seed, al->n, tv);
  k_gen<<<1024, 256>>>(B, N2, seed ^ 0xB0B0B0B0B0B0B0B0ull, al->n, tv);
  CK(cudaGetLastError());

  int32_t* Dref;
  float* Df;
  int32_t* Di;
  CK(cudaMalloc(&Dref, N2 * 4));
  CK(cudaMalloc(&Df, N2 * 4));
  CK(cudaMalloc(&Di, N2 * 4));
  const size_t WS = size_t(64) << 20;
  void* ws;
  CK(cudaMalloc(&ws, WS));
  unsigned long long* dbad;
  int* dflag;
  CK(cudaMalloc(&dbad, 8));
  CK(cudaMalloc(&dflag, 4));
  cublasLtHandle_t h;
  LTC(cublasLtCreate(&h));

  printf("{\"event\":\"start\",\"n\":%d,\"alphabet\":\"%s\",\"seed\":%llu,\"gpu\":\"%s\",\"uuid\":\"%s\",\"sms\":%d,"
         "\"cublasLt\":%zu,\"sm_clock_mhz\":%u}\n",
         n, al->name, (unsigned long long)seed, prop.name, uuid, prop.multiProcessorCount, cublasLtGetVersion(),
         sm_clock());

  // int8 baseline and its CPU check.
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
  auto cmp_i = [&](const int32_t* X) {
    CK(cudaMemset(dbad, 0, 8));
    k_cmp_i32<<<1024, 256>>>(X, Dref, N2, dbad);
    unsigned long long b;
    CK(cudaMemcpy(&b, dbad, 8, cudaMemcpyDeviceToHost));
    return b;
  };

  struct Route {
    std::string name;
    std::function<void()> f;
    double ms1 = 0;
  };
  std::vector<Route> routes;
  routes.push_back({"int8", [&] { run(p8, &p8.algo, A, B, Di, nullptr, nullptr, ws); }});

  // FP8 E4M3, FP32 out.
  uint8_t *A8, *B8;
  CK(cudaMalloc(&A8, N2));
  CK(cudaMalloc(&B8, N2));
  k_fp8<<<1024, 256>>>(A, A8, N2);
  k_fp8<<<1024, 256>>>(B, B8, N2);
  Plan pf8 = make_plan(h, K_FP8, n, WS);
  std::string tlf8;
  if (autotune(pf8, A8, B8, Df, nullptr, nullptr, ws, tlf8)) {
    CK(cudaMemset(Df, 0xA5, N2 * 4));
    run(pf8, &pf8.algo, A8, B8, Df, nullptr, nullptr, ws);
    CK(cudaDeviceSynchronize());
    unsigned long long bf8 = cmp_f(Df);
    printf("{\"event\":\"gate\",\"route\":\"fp8_e4m3\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu,"
           "\"candidates\":%d,\"best\":%d,\"autotune_ms\":[%s]}\n",
           N2, bf8, pf8.ncand, pf8.best, tlf8.c_str());
    if (!bf8) routes.push_back({"fp8_e4m3", [&] { run(pf8, &pf8.algo, A8, B8, Df, nullptr, nullptr, ws); }});
  }
  {  // control: the int8 product with one entry of A changed to another alphabet value must fail the comparison
    int8_t v0;
    CK(cudaMemcpy(&v0, A, 1, cudaMemcpyDeviceToHost));
    int8_t v1 = int8_t(al->v[0] == v0 ? al->v[1] : al->v[0]);
    CK(cudaMemcpy(A, &v1, 1, cudaMemcpyHostToDevice));
    run(p8, &p8.algo, A, B, Di, nullptr, nullptr, ws);
    CK(cudaDeviceSynchronize());
    unsigned long long bc = cmp_i(Di);
    CK(cudaMemcpy(A, &v0, 1, cudaMemcpyHostToDevice));
    printf("{\"event\":\"control\",\"route\":\"int8\",\"changed\":\"A[0][0] %d->%d\",\"mismatches\":%llu,"
           "\"rejected\":%s}\n",
           v0, v1, bc, bc ? "true" : "false");
    if (!bc) return 7;
  }

  // FP4: scales all 1 (UE4M3 0x38, UE8M0 0x7F), so the scale layout is immaterial. n x n/16 bytes covers both modes.
  const size_t SB = N2 / 16;
  uint8_t *s_nv, *s_mx;
  CK(cudaMalloc(&s_nv, SB));
  CK(cudaMalloc(&s_mx, SB));
  CK(cudaMemset(s_nv, 0x38, SB));
  CK(cudaMemset(s_mx, 0x7F, SB));
  Plan pnv = make_plan(h, K_NVFP4, n, WS), pmx = make_plan(h, K_MXFP4, n, WS);
  std::vector<uint8_t*> fp4bufs;
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
    fp4bufs.push_back(o);
    return o;
  };
  if (fp4_holds(*al)) {
    uint8_t* A4 = pack4(A);
    uint8_t* B4 = pack4(B);
    for (Plan* pp : {&pnv, &pmx}) {
      Plan& p = *pp;
      const uint8_t* s = p.kind == K_NVFP4 ? s_nv : s_mx;
      std::string tl;
      if (!autotune(p, A4, B4, Df, s, s, ws, tl)) continue;
      CK(cudaMemset(Df, 0xA5, N2 * 4));
      run(p, &p.algo, A4, B4, Df, s, s, ws);
      CK(cudaDeviceSynchronize());
      unsigned long long b = cmp_f(Df);
      printf("{\"event\":\"gate\",\"route\":\"%s\",\"against\":\"int8\",\"checked\":%zu,\"mismatches\":%llu,"
             "\"candidates\":%d,\"best\":%d,\"autotune_ms\":[%s]}\n",
             kind_name[p.kind], N2, b, p.ncand, p.best, tl.c_str());
      if (!b) routes.push_back({kind_name[p.kind], [&p, A4, B4, s, &Df, &ws] { run(p, &p.algo, A4, B4, Df, s, s, ws); }});
    }
  } else {
    // Karatsuba: a = 2h + l with h, l, h + l on the E2M1 integer grid; ab = 2 hh' + 2 (h+l)(h'+l') - ll'.
    Table17 th{}, tl{}, ts{};
    bool ok = true;
    for (int v = -8; v <= 8; ++v) {
      bool f = false;
      for (int hh = -8; hh <= 8 && !f; ++hh) {
        int ll = v - 2 * hh;
        if (e2m1_int(hh) && e2m1_int(ll) && e2m1_int(hh + ll)) {
          th.v[v + 8] = int8_t(hh);
          tl.v[v + 8] = int8_t(ll);
          ts.v[v + 8] = int8_t(hh + ll);
          f = true;
        }
      }
      ok = ok && f;
    }
    if (!ok) return 9;
    int8_t* tmp;
    CK(cudaMalloc(&tmp, N2));
    uint8_t* P[2][3];
    for (int side = 0; side < 2; ++side) {
      const Table17* tabs[3] = {&th, &ts, &tl};
      for (int j = 0; j < 3; ++j) {
        k_map<<<1024, 256>>>(side ? B : A, tmp, N2, *tabs[j]);
        P[side][j] = pack4(tmp);
      }
    }
    CK(cudaFree(tmp));
    std::string tlog;
    if (autotune(pnv, P[0][0], P[1][0], Df, s_nv, s_nv, ws, tlog)) {
      auto kara = [&pnv, P, s_nv, &Df, &ws] {
        run(pnv, &pnv.algo, P[0][0], P[1][0], Df, s_nv, s_nv, ws, 2.f, 0.f);
        run(pnv, &pnv.algo, P[0][1], P[1][1], Df, s_nv, s_nv, ws, 2.f, 1.f);
        run(pnv, &pnv.algo, P[0][2], P[1][2], Df, s_nv, s_nv, ws, -1.f, 1.f);
      };
      CK(cudaMemset(Df, 0xA5, N2 * 4));
      kara();
      CK(cudaDeviceSynchronize());
      unsigned long long b = cmp_f(Df);
      printf("{\"event\":\"gate\",\"route\":\"nvfp4_karatsuba3\",\"against\":\"int8\",\"checked\":%zu,"
             "\"mismatches\":%llu,\"candidates\":%d,\"best\":%d,\"autotune_ms\":[%s]}\n",
             N2, b, pnv.ncand, pnv.best, tlog.c_str());
      if (!b) routes.push_back({"nvfp4_karatsuba3", kara});
    }
  }

  // Bit-slicing and Four Russians, {±1} only.
  uint32_t *Ab = nullptr, *Bb = nullptr;
  uint8_t *Ay = nullptr, *By = nullptr;
  if (!strcmp(al->name, "pm1")) {
    CK(cudaMalloc(&Ab, N2 / 8));
    CK(cudaMalloc(&Bb, N2 / 8));
    k_bits<<<1024, 256>>>(A, Ab, N2 / 32);
    k_bits<<<1024, 256>>>(B, Bb, N2 / 32);
    auto bin = [&, n] { k_bin<<<dim3(n / BT, n / BT), 256>>>(Ab, Bb, Di, n, n / 32, n); };
    CK(cudaMemset(Di, 0xA5, N2 * 4));
    bin();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    unsigned long long b = cmp_i(Di);
    printf("{\"event\":\"gate\",\"route\":\"bitslice_xor_popc\",\"against\":\"int8\",\"checked\":%zu,"
           "\"mismatches\":%llu}\n",
           N2, b);
    if (!b) routes.push_back({"bitslice_xor_popc", bin});
  }
  if (!strcmp(al->name, "pm1") && n % FR_ROWS == 0) {
    CK(cudaMalloc(&Ay, N2 / 8));
    CK(cudaMalloc(&By, N2 / 8));
    k_bytes<<<1024, 256>>>(Ab, Ay, N2 / 8);
    k_bytes<<<1024, 256>>>(Bb, By, N2 / 8);
    unsigned long long b;
    auto fr = [&, n] { k_4r<<<dim3(n / FR_COLS, n / FR_ROWS), 256>>>(Ay, By, Di, n, n / 8); };
    CK(cudaMemset(Di, 0xA5, N2 * 4));
    fr();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    b = cmp_i(Di);
    printf("{\"event\":\"gate\",\"route\":\"four_russians_t8\",\"against\":\"int8\",\"checked\":%zu,"
           "\"mismatches\":%llu}\n",
           N2, b);
    if (!b) routes.push_back({"four_russians_t8", fr});
  }

  // Interleaved timing: items of about 25 ms of device time, one item per route a round, SM clock before and after.
  for (Route& r : routes) {
    r.f();
    CK(cudaDeviceSynchronize());
    r.ms1 = time_calls(r.f, 3);
  }
  std::vector<std::vector<double>> per(routes.size());
  std::vector<int> iters(routes.size());
  for (size_t i = 0; i < routes.size(); ++i) iters[i] = std::max(1, int(25.0 / routes[i].ms1));
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
    double clk = sm_clock();
    printf("{\"event\":\"summary\",\"n\":%d,\"alphabet\":\"%s\",\"route\":\"%s\",\"median_ms\":%.5f,\"min_ms\":%.5f,"
           "\"ratio_vs_int8\":%.4f,\"int_macs_per_s\":%.4e,\"macs_per_sm_clk_at_2092\":%.1f}\n",
           n, al->name, routes[i].name.c_str(), m, mn, m / base, macs / (m * 1e-3),
           macs / (m * 1e-3) / (prop.multiProcessorCount * 2092e6));
    (void)clk;
  }
  return 0;
}
