// The three-piece 2:4-sparse NVFP4 route to the exact integer product P = X·W (X n×n uniform on {±1..±k}, W n×n ±1), on one
// sm_120 GPU, against the plain int8 and FP8 cuBLASLt GEMMs on the same product (RS-D2F round 2,
// note:20261016T0101Z-draft-rs-d2-wide-peel; the router is benchmarks/steiner_referee/wide_peel.py decompose_group).
//
// Method.  Each entry x of X is written x = x1 + x2 + z + drop with the fixed piece scales (x1,x2 at 2, z at 5/4), each
// piece E2M1-holdable and with at most two nonzeros in every group of four columns (a 2:4 structure), the rest sent to a
// sparse int8 correction (drops).  Then
//     X·W = 2·((x1/2)·W) + 2·((x2/2)·W) + (5/4)·((z/(5/4))·W) + drops·W,
// three NVFP4 MMAs (scalar alpha {2, 2, 5/4}, FP32 accumulate, bit-exact for these small-integer sums) plus a CSR drops
// correction.  The three MMAs are the whole speed case: at the census rates (sparse NVFP4 4094, dense int8 1024 MACs/SM/clk)
// three sparse MMAs are 3·1024/4096 = 0.75× honest int8.  Correctness does not need the sparsity, so the three-piece
// arithmetic is gated bit-exact here with DENSE cuBLASLt NVFP4 GEMMs (which run on sm_120); the sparse speed is then
// measured on its own: a cuSPARSELt 2:4 GEMM (int8 always, FP4 if the library has a kernel for this arch), the sparse-vs-
// dense efficiency a real structured-sparse GEMM reaches here.  Gate: P against a CPU int64 reference (every entry at
// n ≤ 1024, sampled otherwise); a control dropping the corrections must mismatch.  Timing: cuBLASLt autotuned, interleaved
// rounds, SM clock read by NVML; the routing/packing and the MMAs are timed apart.  JSON lines.
//
//   tp <n> <k> <seed> <rounds>      k in {8,32,64}; the exact router table is built only for k ≤ 8
#include <cublasLt.h>
#include <cusparseLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <array>
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

__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// X: uniform on {±1..±k} (2k values).  W: uniform ±1.
__global__ void k_gen_x(int8_t* M, size_t count, uint64_t seed, int k) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
    uint64_t r = mix64(seed ^ (i * 0xD1B54A32D192ED03ull));
    int m = int(r % uint64_t(k)) + 1;       // 1..k
    M[i] = int8_t((r & 0x8000000000000000ull) ? -m : m);
  }
}
__global__ void k_gen_w(int8_t* M, size_t count, uint64_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    M[i] = (mix64(seed ^ (i * 0x9E3779B97F4A7C15ull)) & 1) ? 1 : -1;
}
__global__ void k_transpose(const int8_t* in, int8_t* out, int n) {
  size_t total = size_t(n) * n;
  for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total; idx += size_t(gridDim.x) * blockDim.x) {
    int i = int(idx / n), j = int(idx % n);
    out[size_t(j) * n + i] = in[idx];
  }
}
__global__ void k_fp8(const int8_t* M, uint8_t* out, size_t count) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    out[i] = __nv_fp8_e4m3(float(M[i])).__x;
}

// E2M1 code of a piece value c at its piece's scale.  piece 0,1 at scale 2 hold |c| in {1,2,3,4,6,8,12} (stored |c|/2 is a
// half-integer on the grid); piece 2 at scale 5/4 holds |c| in {5} (stored |c|/1.25 = 4).  Sign bit 8.  Flags off-grid.
__device__ inline uint8_t e2m1_piece(int c, int piece, int* bad) {
  int m = c < 0 ? -c : c, code;
  if (piece < 2) {
    switch (m) {
      case 0: code = 0; break;   case 1: code = 1; break;  case 2: code = 2; break;  case 3: code = 3; break;
      case 4: code = 4; break;   case 6: code = 5; break;  case 8: code = 6; break;  case 12: code = 7; break;
      default: atomicAdd(bad, 1); code = 0;
    }
  } else {
    switch (m) {
      case 0: code = 0; break;  case 5: code = 6; break;  // 5/1.25 = 4.0 -> E2M1 value 4.0 -> code 6
      default: atomicAdd(bad, 1); code = 0;
    }
  }
  return uint8_t(code | (c < 0 ? 8 : 0));
}
// W (±1) to E2M1 at scale 1: +1 -> code 2, -1 -> code 0xA.
__device__ inline uint8_t e2m1_pm1(int v) { return v < 0 ? 0xA : 0x2; }

__global__ void k_pack_piece(const int8_t* piece, uint8_t* out, size_t pairs, int which, int* bad) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < pairs; i += size_t(gridDim.x) * blockDim.x)
    out[i] = uint8_t(e2m1_piece(piece[2 * i], which, bad) | (e2m1_piece(piece[2 * i + 1], which, bad) << 4));
}
__global__ void k_pack_w(const int8_t* W, uint8_t* out, size_t pairs) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < pairs; i += size_t(gridDim.x) * blockDim.x)
    out[i] = uint8_t(e2m1_pm1(W[2 * i]) | (e2m1_pm1(W[2 * i + 1]) << 4));
}

// ---------------------------------------------------------------- the router (fixed scales 2, 2, 5/4), ported from
// wide_peel.decompose_group: an exact per-group decomposition into three pieces (<= 2 nonzeros each in a group of 4) and a
// CUDA remainder.  Built as a table over all groups for k <= 8, then applied on the GPU by lookup.
static bool in_held(int piece, int v) {
  v = v < 0 ? -v : v;
  if (piece < 2) return v == 1 || v == 2 || v == 3 || v == 4 || v == 6 || v == 8 || v == 12;
  return v == 5;
}
static std::vector<int> held_list(int piece) {
  std::vector<int> base = piece < 2 ? std::vector<int>{1, 2, 3, 4, 6, 8, 12} : std::vector<int>{5}, out;
  for (int b : base) { out.push_back(b); out.push_back(-b); }
  std::sort(out.begin(), out.end(), [](int a, int b) { return abs(a) < abs(b); });
  return out;
}
// The 7 nonempty subsets of {0,1,2}, size-sorted (singles, then pairs, then the triple).
static const std::vector<std::vector<int>>& subsets() {
  static const std::vector<std::vector<int>> S = {{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}};
  return S;
}
static bool feasible(int v, const std::vector<int>& sub) {
  if (sub.size() == 1) return in_held(sub[0], v);
  if (sub.size() == 2) {
    for (int a : held_list(sub[0])) if (in_held(sub[1], v - a)) return true;
    return false;
  }
  for (int a : held_list(sub[0])) for (int b : held_list(sub[1])) if (in_held(sub[2], v - a - b)) return true;
  return false;
}
static std::vector<const std::vector<int>*> options(int v) {
  std::vector<const std::vector<int>*> opt;
  for (const auto& s : subsets()) if (feasible(v, s)) opt.push_back(&s);
  return opt;
}
// Decompose a group of 4 values -> pieces[3][4] and cuda[4]; returns drops = #cuda nonzeros.
static int decompose_group(const int vals[4], int8_t pieces[3][4], int8_t cuda[4]) {
  std::vector<const std::vector<int>*> opt[4];
  for (int q = 0; q < 4; ++q) opt[q] = options(vals[q]);
  for (int keep = 4; keep >= 0; --keep) {
    std::vector<int> kept(4);
    for (int i = 0; i < 4; ++i) kept[i] = i;
    // choose which 'keep' of the 4 to keep (combinations), in increasing-index order
    std::vector<int> idx(keep);
    for (int i = 0; i < keep; ++i) idx[i] = i;
    auto try_combo = [&](const std::vector<int>& kp) -> bool {
      // iterate the product of opt over kept positions
      int K = kp.size();
      std::vector<int> ci(K, 0);
      for (bool more = true; more;) {
        int use[3] = {0, 0, 0};
        bool ok = true;
        for (int t = 0; t < K; ++t) {
          if (opt[kp[t]].empty()) { ok = false; break; }
          for (int pc : *opt[kp[t]][ci[t]]) use[pc]++;
        }
        if (ok && use[0] <= 2 && use[1] <= 2 && use[2] <= 2) {
          for (int p = 0; p < 3; ++p) for (int q = 0; q < 4; ++q) pieces[p][q] = 0;
          for (int q = 0; q < 4; ++q) cuda[q] = int8_t(vals[q]);
          for (int t = 0; t < K; ++t) {
            int q = kp[t];
            const std::vector<int>& sub = *opt[q][ci[t]];
            int v = vals[q];
            // find the split: first product of held values summing to v
            bool done = false;
            if (sub.size() == 1) { pieces[sub[0]][q] = int8_t(v); done = true; }
            else if (sub.size() == 2) {
              for (int a : held_list(sub[0])) { for (int b : held_list(sub[1])) if (a + b == v) {
                pieces[sub[0]][q] = int8_t(a); pieces[sub[1]][q] = int8_t(b); done = true; break; } if (done) break; }
            } else {
              for (int a : held_list(sub[0])) { for (int b : held_list(sub[1])) { for (int cc : held_list(sub[2]))
                if (a + b + cc == v) { pieces[sub[0]][q] = int8_t(a); pieces[sub[1]][q] = int8_t(b);
                  pieces[sub[2]][q] = int8_t(cc); done = true; break; } if (done) break; } if (done) break; }
            }
            cuda[q] = 0;
          }
          return true;
        }
        // advance ci (mixed radix over opt sizes)
        more = false;
        for (int t = K - 1; t >= 0; --t) {
          if (++ci[t] < int(opt[kp[t]].size())) { more = true; break; }
          ci[t] = 0;
        }
      }
      return false;
    };
    // iterate combinations of 'keep' indices out of 4
    std::vector<int> comb(keep);
    for (int i = 0; i < keep; ++i) comb[i] = i;
    while (true) {
      if (try_combo(comb)) return 4 - keep;
      int i = keep - 1;
      while (i >= 0 && comb[i] == 4 - keep + i) --i;
      if (i < 0) break;
      ++comb[i];
      for (int j = i + 1; j < keep; ++j) comb[j] = comb[j - 1] + 1;
    }
    if (keep == 0) break;
  }
  for (int p = 0; p < 3; ++p) for (int q = 0; q < 4; ++q) pieces[p][q] = 0;
  for (int q = 0; q < 4; ++q) cuda[q] = int8_t(vals[q]);
  return 4;
}

// GPU router: each thread owns one group of 4 columns of a row, looks up TAB (indexed by the 4 values encoded base 2k) and
// writes x1,x2,z and drops (all n×n int8, row-major).  TAB[g] = 3*4 piece bytes then 4 cuda bytes.
__global__ void k_route(const int8_t* X, const int8_t* TAB, int8_t* x1, int8_t* x2, int8_t* z, int8_t* drops, int n,
                        int k) {
  const int twoK = 2 * k;
  size_t groups = size_t(n) * (n / 4);
  for (size_t gi = blockIdx.x * size_t(blockDim.x) + threadIdx.x; gi < groups; gi += size_t(gridDim.x) * blockDim.x) {
    int row = int(gi / (n / 4)), g = int(gi % (n / 4)) * 4;
    size_t base = size_t(row) * n + g;
    uint32_t sig = 0, mul = 1;
    for (int q = 0; q < 4; ++q) {
      int v = X[base + q];                    // in {-k..k}\{0}
      int idx = v < 0 ? (v + k) : (v + k - 1); // map to 0..2k-1
      sig += idx * mul;
      mul *= twoK;
    }
    const int8_t* t = TAB + size_t(sig) * 16;
    for (int q = 0; q < 4; ++q) {
      x1[base + q] = t[0 * 4 + q];
      x2[base + q] = t[1 * 4 + q];
      z[base + q] = t[2 * 4 + q];
      drops[base + q] = t[12 + q];
    }
  }
}

__global__ void k_cmp_f32(const float* X, const int32_t* R, size_t count, unsigned long long* bad) {
  unsigned long long b = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    b += (X[i] != float(R[i])) || (float(R[i]) != float(double(R[i])));
  if (b) atomicAdd(bad, b);
}

// CSR drops × dense W on the CUDA cores: out[i,j] += Σ_{(l,v)∈row i} v·W[l,j], W row-major.  One warp a row panel.
constexpr int COLS_PER_LANE = 8;
__global__ void k_spmm_add(const int* __restrict__ rowptr, const int* __restrict__ colidx,
                           const int8_t* __restrict__ vals, const int8_t* __restrict__ W, float* __restrict__ out,
                           int n) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  const int row = warp % n, panel = (warp / n) * (32 * COLS_PER_LANE);
  if (panel >= n) return;
  float acc[COLS_PER_LANE];
#pragma unroll
  for (int c = 0; c < COLS_PER_LANE; ++c) acc[c] = 0.f;
  const int lo = rowptr[row], hi = rowptr[row + 1];
  for (int e = lo; e < hi; ++e) {
    const float v = float(vals[e]);
    const int8_t* b = W + size_t(colidx[e]) * n + panel + lane;
#pragma unroll
    for (int c = 0; c < COLS_PER_LANE; ++c) acc[c] += v * float(b[c * 32]);
  }
  float* o = out + size_t(row) * n + panel + lane;
#pragma unroll
  for (int c = 0; c < COLS_PER_LANE; ++c) o[c * 32] += acc[c];
}

// ---------------------------------------------------------------- cuBLASLt P = X·W (via the proven A·Bᵀ config with the
// second operand pre-transposed): kinds int8, fp8 e4m3, dense NVFP4.
enum Kind { K_INT8, K_FP8, K_NVFP4 };
static const char* kind_name[] = {"int8", "fp8_e4m3", "dense_nvfp4"};
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
  p.h = h; p.kind = kind; p.ws = ws;
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
  LTC(cublasLtMatrixLayoutCreate(&p.a, in, n, n, n));
  LTC(cublasLtMatrixLayoutCreate(&p.b, in, n, n, n));
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
// D = alpha·(A·Bᵀ) + beta·D, A=first (n×n), B=second (n×n), both row-major → with the second pre-transposed this is X·W.
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
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for (int i = 0; i < iters; ++i) f();
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / iters;
}
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
    f(); f(); CK(cudaDeviceSynchronize());
    float t = 1e30f;
    for (int r = 0; r < 3; ++r) t = std::min(t, time_calls(f, 3));
    snprintf(buf, sizeof buf, "%s%.4f", log.empty() ? "" : ",", t);
    log += buf;
    if (t < best) { best = t; p.best = i; p.algo = res[i].algo; }
  }
  if (p.best < 0) { printf("{\"event\":\"unsupported\",\"route\":\"%s\",\"reason\":\"no candidate\"}\n", kind_name[p.kind]); return false; }
  return true;
}

static nvmlDevice_t g_nvml;
static bool g_nvml_ok = false;
static unsigned sm_clock() { unsigned c = 0; if (g_nvml_ok) nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &c); return c; }

// ---------------------------------------------------------------- cuSPARSELt 2:4-sparse GEMM timing (int8 always; FP4 if a
// kernel exists for this arch).  Returns the per-call ms, or -1 if the config is unsupported here.
static double sparse_time(cusparseLtHandle_t& H, int n, cudaDataType in, cudaDataType out, cusparseComputeType comp,
                          bool scale, int* status_out) {
  const int M = n, K = n, N = n, AL = 16;
  const bool fp4 = in == CUDA_R_4F_E2M1;
  size_t bA = fp4 ? size_t(M) * K / 2 : size_t(M) * K, bB = fp4 ? size_t(K) * N / 2 : size_t(K) * N;
  uint8_t *dA, *dB; void* dD;
  CK(cudaMalloc(&dA, bA)); CK(cudaMalloc(&dB, bB));
  CK(cudaMalloc(&dD, size_t(M) * N * (out == CUDA_R_16F ? 2 : 4)));
  CK(cudaMemset(dA, 0x11, bA)); CK(cudaMemset(dB, 0x22, bB));  // structured enough for timing; values irrelevant to cost
  cusparseLtMatDescriptor_t mA, mB, mC;
  cusparseStatus_t s;
  s = cusparseLtStructuredDescriptorInit(&H, &mA, M, K, K, AL, in, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT);
  if (s) { *status_out = s; return -1; }
  s = cusparseLtDenseDescriptorInit(&H, &mB, K, N, N, AL, in, CUSPARSE_ORDER_ROW);
  if (s) { *status_out = s; return -1; }
  s = cusparseLtDenseDescriptorInit(&H, &mC, M, N, N, AL, out, CUSPARSE_ORDER_ROW);
  if (s) { *status_out = s; return -1; }
  cusparseLtMatmulDescriptor_t mm;
  s = cusparseLtMatmulDescriptorInit(&H, &mm, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &mA, &mB,
                                     &mC, &mC, comp);
  if (s) { *status_out = s; return -1; }
  uint8_t *dSA = nullptr, *dSB = nullptr;
  if (scale) {
    cusparseLtMatmulMatrixScale_t sc = CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3;
    cusparseLtMatmulDescSetAttribute(&H, &mm, CUSPARSELT_MATMUL_A_SCALE_MODE, &sc, sizeof sc);
    cusparseLtMatmulDescSetAttribute(&H, &mm, CUSPARSELT_MATMUL_B_SCALE_MODE, &sc, sizeof sc);
    CK(cudaMalloc(&dSA, size_t(M) * K)); CK(cudaMalloc(&dSB, size_t(K) * N));
    CK(cudaMemset(dSA, 0x38, size_t(M) * K)); CK(cudaMemset(dSB, 0x38, size_t(K) * N));
    void *pA = dSA, *pB = dSB;
    cusparseLtMatmulDescSetAttribute(&H, &mm, CUSPARSELT_MATMUL_A_SCALE_POINTER, &pA, sizeof pA);
    cusparseLtMatmulDescSetAttribute(&H, &mm, CUSPARSELT_MATMUL_B_SCALE_POINTER, &pB, sizeof pB);
  }
  cusparseLtMatmulAlgSelection_t alg;
  s = cusparseLtMatmulAlgSelectionInit(&H, &alg, &mm, CUSPARSELT_MATMUL_ALG_DEFAULT);
  if (s) { *status_out = s; return -1; }
  cusparseLtMatmulPlan_t plan;
  s = cusparseLtMatmulPlanInit(&H, &plan, &mm, &alg);
  if (s) { *status_out = s; return -1; }
  size_t cSize = 0, cBuf = 0;
  cusparseLtSpMMACompressedSize(&H, &plan, &cSize, &cBuf);
  uint8_t *dAc, *dAcBuf;
  CK(cudaMalloc(&dAc, cSize ? cSize : 1)); CK(cudaMalloc(&dAcBuf, cBuf ? cBuf : 1));
  s = cusparseLtSpMMACompress(&H, &plan, dA, dAc, dAcBuf, nullptr);
  if (s) { *status_out = s; return -1; }
  size_t wsSize = 0;
  cusparseLtMatmulGetWorkspace(&H, &plan, &wsSize);
  void* dWs; CK(cudaMalloc(&dWs, wsSize ? wsSize : 1));
  float alpha = 1.f, beta = 0.f;
  cudaStream_t st = nullptr;
  auto f = [&] { cusparseLtMatmul(&H, &plan, &alpha, dAc, dB, &beta, dD, dD, dWs, &st, 1); };
  s = cusparseLtMatmul(&H, &plan, &alpha, dAc, dB, &beta, dD, dD, dWs, &st, 1);
  if (s) { *status_out = s; return -1; }
  CK(cudaDeviceSynchronize());
  double t = 1e30;
  for (int r = 0; r < 3; ++r) t = std::min(t, double(time_calls(f, 5)));
  *status_out = 0;
  cusparseLtMatmulPlanDestroy(&plan);
  cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(dAc); cudaFree(dAcBuf); cudaFree(dWs);
  if (dSA) cudaFree(dSA);
  if (dSB) cudaFree(dSB);
  return t;
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: tp <n> <k> <seed> <rounds>\n"); return 2; }
  const int n = atoi(argv[1]), k = atoi(argv[2]);
  const uint64_t seed = strtoull(argv[3], nullptr, 0);
  const int rounds = atoi(argv[4]);
  if (n % 256) { fprintf(stderr, "n must be a multiple of 256\n"); return 2; }
  const size_t N2 = size_t(n) * n;
  char pci[32];
  CK(cudaDeviceGetPCIBusId(pci, sizeof pci, 0));
  if (nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByPciBusId_v2(pci, &g_nvml) == NVML_SUCCESS) g_nvml_ok = true;
  char uuid[96] = "?";
  if (g_nvml_ok) nvmlDeviceGetUUID(g_nvml, uuid, sizeof uuid);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));

  int8_t *X, *W, *Wt;
  CK(cudaMalloc(&X, N2)); CK(cudaMalloc(&W, N2)); CK(cudaMalloc(&Wt, N2));
  k_gen_x<<<1024, 256>>>(X, N2, seed, k);
  k_gen_w<<<1024, 256>>>(W, N2, seed ^ 0xB0B0B0B0ull);
  k_transpose<<<2048, 256>>>(W, Wt, n);
  CK(cudaGetLastError());

  const size_t WS = size_t(128) << 20;
  void* ws; CK(cudaMalloc(&ws, WS));
  unsigned long long* dbad; CK(cudaMalloc(&dbad, 8));
  cublasLtHandle_t h; LTC(cublasLtCreate(&h));
  cusparseLtHandle_t H;
  bool sparse_ok = cusparseLtInit(&H) == CUSPARSE_STATUS_SUCCESS;
  int sver = 0; if (sparse_ok) cusparseLtGetVersion(&H, &sver);

  printf("{\"event\":\"start\",\"n\":%d,\"k\":%d,\"seed\":%llu,\"gpu\":\"%s\",\"uuid\":\"%s\",\"sms\":%d,"
         "\"cublasLt\":%zu,\"cusparseLt\":%d,\"sm_clock_mhz\":%u}\n",
         n, k, (unsigned long long)seed, prop.name, uuid, prop.multiProcessorCount, cublasLtGetVersion(), sver,
         sm_clock());

  // Pack W to FP4 (±1) and FP8; scale arrays for dense NVFP4 (VEC16 UE4M3 1.0 = 0x38).
  uint8_t *W4; CK(cudaMalloc(&W4, N2 / 2));
  k_pack_w<<<1024, 256>>>(W, W4, N2 / 2);
  uint8_t *s_nv; CK(cudaMalloc(&s_nv, N2 / 16)); CK(cudaMemset(s_nv, 0x38, N2 / 16));
  uint8_t *Xf8, *Wtf8; CK(cudaMalloc(&Xf8, N2)); CK(cudaMalloc(&Wtf8, N2));
  k_fp8<<<1024, 256>>>(X, Xf8, N2);
  k_fp8<<<1024, 256>>>(Wt, Wtf8, N2);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  int32_t* Dref; float* Df;
  CK(cudaMalloc(&Dref, N2 * 4)); CK(cudaMalloc(&Df, N2 * 4));

  // int8 baseline P = X·W and its CPU int64 check.
  Plan p8 = make_plan(h, K_INT8, n, WS);
  std::string tl8;
  if (!autotune(p8, X, Wt, Dref, nullptr, nullptr, ws, tl8)) return 5;
  run(p8, &p8.algo, X, Wt, Dref, nullptr, nullptr, ws);
  CK(cudaDeviceSynchronize());
  {
    std::vector<int8_t> hX(N2), hW(N2);
    std::vector<int32_t> hD(N2);
    CK(cudaMemcpy(hX.data(), X, N2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hW.data(), W, N2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hD.data(), Dref, N2 * 4, cudaMemcpyDeviceToHost));
    size_t checked = 0, bad = 0;
    auto chk = [&](size_t r, size_t c) {
      int64_t s = 0;
      for (int l = 0; l < n; ++l) s += int(hX[r * n + l]) * int(hW[size_t(l) * n + c]);
      ++checked; bad += s != hD[r * n + c];
    };
    if (n <= 1024) { for (size_t r = 0; r < size_t(n); ++r) for (size_t c = 0; c < size_t(n); ++c) chk(r, c); }
    else { for (uint64_t i = 0; i < 4096; ++i) { uint64_t x = mix64(seed + 977 * i); chk(x % n, (x >> 32) % n); } chk(0, 0); chk(n - 1, n - 1); }
    printf("{\"event\":\"gate\",\"route\":\"int8\",\"against\":\"cpu\",\"checked\":%zu,\"mismatches\":%zu,\"candidates\":%d}\n",
           checked, bad, p8.ncand);
    if (bad) return 6;
  }

  auto cmp_f = [&](const float* Xv) {
    CK(cudaMemset(dbad, 0, 8));
    k_cmp_f32<<<1024, 256>>>(Xv, Dref, N2, dbad);
    unsigned long long b; CK(cudaMemcpy(&b, dbad, 8, cudaMemcpyDeviceToHost)); return b;
  };

  struct Route { std::string name; std::function<void()> f; };
  std::vector<Route> routes;
  routes.push_back({"int8", [&] { run(p8, &p8.algo, X, Wt, Df, nullptr, nullptr, ws); }});

  // FP8 baseline.
  Plan pf8 = make_plan(h, K_FP8, n, WS);
  std::string tlf8; float* Df8; CK(cudaMalloc(&Df8, N2 * 4));
  if (autotune(pf8, Xf8, Wtf8, Df8, nullptr, nullptr, ws, tlf8)) {
    CK(cudaMemset(Df8, 0xA5, N2 * 4));
    run(pf8, &pf8.algo, Xf8, Wtf8, Df8, nullptr, nullptr, ws);
    CK(cudaDeviceSynchronize());
    unsigned long long bf8 = cmp_f(Df8);
    printf("{\"event\":\"gate\",\"route\":\"fp8_e4m3\",\"against\":\"int8\",\"mismatches\":%llu}\n", bf8);
    if (!bf8) routes.push_back({"fp8_e4m3", [&] { run(pf8, &pf8.algo, Xf8, Wtf8, Df8, nullptr, nullptr, ws); }});
  }

  // Router table (k ≤ 8) and the three pieces + drops.
  bool have_method = (2 * k) * (2 * k) * (2 * k) * (2 * k) <= (1 << 22) && 2 * k <= 16;
  int8_t *x1, *x2, *zp, *drp;
  uint8_t *x14, *x24, *z4;
  double route_stats_drops = -1;
  if (have_method) {
    const int twoK = 2 * k;
    size_t ng = size_t(twoK) * twoK * twoK * twoK;
    std::vector<int8_t> tab(ng * 16);
    long long tot_drops = 0;
    for (size_t g = 0; g < ng; ++g) {
      int vals[4]; size_t t = g;
      for (int q = 0; q < 4; ++q) { int idx = int(t % twoK); t /= twoK; vals[q] = idx < k ? idx - k : idx - k + 1; }
      int8_t pieces[3][4], cuda[4];
      tot_drops += decompose_group(vals, pieces, cuda);
      int8_t* e = &tab[g * 16];
      for (int p = 0; p < 3; ++p) for (int q = 0; q < 4; ++q) e[p * 4 + q] = pieces[p][q];
      for (int q = 0; q < 4; ++q) e[12 + q] = cuda[q];
    }
    route_stats_drops = double(tot_drops) / (ng * 4);  // mean drop fraction over the uniform alphabet (per entry)
    int8_t* dTAB; CK(cudaMalloc(&dTAB, tab.size())); CK(cudaMemcpy(dTAB, tab.data(), tab.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&x1, N2)); CK(cudaMalloc(&x2, N2)); CK(cudaMalloc(&zp, N2)); CK(cudaMalloc(&drp, N2));
    CK(cudaMalloc(&x14, N2 / 2)); CK(cudaMalloc(&x24, N2 / 2)); CK(cudaMalloc(&z4, N2 / 2));
    auto do_route = [&] { k_route<<<4096, 256>>>(X, dTAB, x1, x2, zp, drp, n, k); };
    do_route(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int* dflag; CK(cudaMalloc(&dflag, 4)); CK(cudaMemset(dflag, 0, 4));
    k_pack_piece<<<1024, 256>>>(x1, x14, N2 / 2, 0, dflag);
    k_pack_piece<<<1024, 256>>>(x2, x24, N2 / 2, 1, dflag);
    k_pack_piece<<<1024, 256>>>(zp, z4, N2 / 2, 2, dflag);
    int bad; CK(cudaMemcpy(&bad, dflag, 4, cudaMemcpyDeviceToHost));
    if (bad) { fprintf(stderr, "piece pack off-grid: %d\n", bad); return 8; }
    CK(cudaDeviceSynchronize());

    // Drops CSR (host) from drp.
    std::vector<int8_t> hdrp(N2);
    CK(cudaMemcpy(hdrp.data(), drp, N2, cudaMemcpyDeviceToHost));
    std::vector<int> rp(n + 1, 0), ci; std::vector<int8_t> vv;
    for (int i = 0; i < n; ++i) { int c = 0; for (int l = 0; l < n; ++l) c += hdrp[size_t(i) * n + l] != 0; rp[i + 1] = rp[i] + c; }
    ci.resize(rp[n]); vv.resize(rp[n]);
    long long nnz = rp[n];
    for (int i = 0; i < n; ++i) { int p = rp[i]; for (int l = 0; l < n; ++l) if (hdrp[size_t(i) * n + l]) { ci[p] = l; vv[p] = hdrp[size_t(i) * n + l]; ++p; } }
    int *dRp, *dCi; int8_t* dVv;
    CK(cudaMalloc(&dRp, (n + 1) * 4)); CK(cudaMalloc(&dCi, std::max<size_t>(1, ci.size()) * 4)); CK(cudaMalloc(&dVv, std::max<size_t>(1, vv.size())));
    CK(cudaMemcpy(dRp, rp.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
    if (!ci.empty()) { CK(cudaMemcpy(dCi, ci.data(), ci.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dVv, vv.data(), vv.size(), cudaMemcpyHostToDevice)); }
    const int panels = (n + 32 * COLS_PER_LANE - 1) / (32 * COLS_PER_LANE);
    const long sp_blocks = (long(n) * panels * 32 + 255) / 256;

    // Dense NVFP4 plan for the pieces (A·Bᵀ with B=Wt-as-FP4; we pack Wt to FP4 for the pieces).
    uint8_t* Wt4; CK(cudaMalloc(&Wt4, N2 / 2)); k_pack_w<<<1024, 256>>>(Wt, Wt4, N2 / 2); CK(cudaDeviceSynchronize());
    Plan pnv = make_plan(h, K_NVFP4, n, WS);
    std::string tlnv;
    bool nv_ok = autotune(pnv, x14, Wt4, Df, s_nv, s_nv, ws, tlnv);

    auto drops_spmm = [&] { k_spmm_add<<<sp_blocks, 256>>>(dRp, dCi, dVv, W, Df, n); };
    // Three-piece method (dense NVFP4 pieces): 2·(x1/2·W) + 2·(x2/2·W) + 5/4·(z/1.25·W) + drops·W.
    auto method_dense = [&] {
      run(pnv, &pnv.algo, x14, Wt4, Df, s_nv, s_nv, ws, 2.f, 0.f);
      run(pnv, &pnv.algo, x24, Wt4, Df, s_nv, s_nv, ws, 2.f, 1.f);
      run(pnv, &pnv.algo, z4, Wt4, Df, s_nv, s_nv, ws, 1.25f, 1.f);
      drops_spmm();
    };
    if (nv_ok) {
      // Gate the method bit-exact.
      CK(cudaMemset(Df, 0xA5, N2 * 4));
      method_dense();
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      unsigned long long bm = cmp_f(Df);
      printf("{\"event\":\"gate\",\"route\":\"method_dense_pieces\",\"against\":\"int8\",\"mismatches\":%llu,"
             "\"drop_frac\":%.5f,\"drop_nnz\":%lld}\n", bm, route_stats_drops, nnz);
      // Control: drop the correction -> must mismatch.
      CK(cudaMemset(Df, 0xA5, N2 * 4));
      run(pnv, &pnv.algo, x14, Wt4, Df, s_nv, s_nv, ws, 2.f, 0.f);
      run(pnv, &pnv.algo, x24, Wt4, Df, s_nv, s_nv, ws, 2.f, 1.f);
      run(pnv, &pnv.algo, z4, Wt4, Df, s_nv, s_nv, ws, 1.25f, 1.f);
      CK(cudaDeviceSynchronize());
      unsigned long long bc = cmp_f(Df);
      printf("{\"event\":\"control\",\"route\":\"no_drops\",\"mismatches\":%llu,\"rejected\":%s}\n", bc, bc ? "true" : "false");

      // Split timing: routing/packing vs the 3 dense MMAs vs drops.
      auto routing = [&] {
        do_route();
        int* f2; static int* kept = nullptr; if (!kept) CK(cudaMalloc(&kept, 4)); f2 = kept;
        k_pack_piece<<<1024, 256>>>(x1, x14, N2 / 2, 0, f2);
        k_pack_piece<<<1024, 256>>>(x2, x24, N2 / 2, 1, f2);
        k_pack_piece<<<1024, 256>>>(zp, z4, N2 / 2, 2, f2);
      };
      auto three_mma = [&] {
        run(pnv, &pnv.algo, x14, Wt4, Df, s_nv, s_nv, ws, 2.f, 0.f);
        run(pnv, &pnv.algo, x24, Wt4, Df, s_nv, s_nv, ws, 2.f, 1.f);
        run(pnv, &pnv.algo, z4, Wt4, Df, s_nv, s_nv, ws, 1.25f, 1.f);
      };
      routing(); three_mma(); drops_spmm(); CK(cudaDeviceSynchronize());
      double t_route = time_calls(routing, 10), t_mma = time_calls(three_mma, 5), t_drop = time_calls(drops_spmm, 5);
      double t_one_dense = time_calls([&] { run(pnv, &pnv.algo, x14, Wt4, Df, s_nv, s_nv, ws, 2.f, 0.f); }, 5);
      printf("{\"event\":\"split\",\"n\":%d,\"k\":%d,\"routing_pack_ms\":%.5f,\"three_dense_mma_ms\":%.5f,"
             "\"drops_spmm_ms\":%.5f,\"one_dense_nvfp4_ms\":%.5f}\n", n, k, t_route, t_mma, t_drop, t_one_dense);

      // Interleaved method (dense pieces) vs int8, SM clock before/after; the method's captured buffers are block-local so
      // this is timed here rather than in the generic interleave below.
      auto int8_route = [&] { run(p8, &p8.algo, X, Wt, Df, nullptr, nullptr, ws); };
      int8_route(); method_dense(); CK(cudaDeviceSynchronize());
      double mi = time_calls(int8_route, 3), mm2 = time_calls(method_dense, 3);
      int it_i = std::max(1, int(25.0 / mi)), it_m = std::max(1, int(25.0 / mm2));
      std::vector<double> pi, pm;
      for (int rd = 0; rd < rounds; ++rd) {
        unsigned c0 = sm_clock(); double a = time_calls(int8_route, it_i); unsigned c1 = sm_clock();
        pi.push_back(a);
        printf("{\"event\":\"rep\",\"route\":\"int8_m\",\"round\":%d,\"iters\":%d,\"ms\":%.5f,\"clk0\":%u,\"clk1\":%u}\n", rd, it_i, a, c0, c1);
        unsigned d0 = sm_clock(); double b = time_calls(method_dense, it_m); unsigned d1 = sm_clock();
        pm.push_back(b);
        printf("{\"event\":\"rep\",\"route\":\"method_dense_pieces\",\"round\":%d,\"iters\":%d,\"ms\":%.5f,\"clk0\":%u,\"clk1\":%u}\n", rd, it_m, b, d0, d1);
      }
      auto med1 = [](std::vector<double> v) { std::sort(v.begin(), v.end()); size_t m = v.size(); return m ? (m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2])) : 0.0; };
      double bi = med1(pi), bm2 = med1(pm);
      printf("{\"event\":\"summary\",\"n\":%d,\"k\":%d,\"route\":\"int8_m\",\"median_ms\":%.5f,\"ratio_vs_int8\":1.0}\n", n, k, bi);
      printf("{\"event\":\"summary\",\"n\":%d,\"k\":%d,\"route\":\"method_dense_pieces\",\"median_ms\":%.5f,"
             "\"ratio_vs_int8\":%.4f}\n", n, k, bm2, bm2 / bi);
      // Theoretical sparse floor: three MMAs at the census sparse-NVFP4 rate are 1.5x one dense NVFP4 GEMM; add drops.
      printf("{\"event\":\"floor\",\"n\":%d,\"k\":%d,\"three_sparse_nvfp4_ideal_ms\":%.5f,\"plus_drops_ms\":%.5f,"
             "\"ratio_vs_int8\":%.4f,\"note\":\"sparse NVFP4 at 2x dense (census 4094 vs 2048); no cuSPARSELt/CUTLASS kernel on sm_120\"}\n",
             n, k, 1.5 * t_one_dense, 1.5 * t_one_dense + t_drop, (1.5 * t_one_dense + t_drop) / bi);
    }
  }

  // cuSPARSELt real sparse-GEMM efficiency: int8 2:4 (always), FP4 2:4 (if a kernel exists for this arch).
  if (sparse_ok) {
    int st = 0;
    double t_dense_i8_ref = -1;
    { Plan pi = make_plan(h, K_INT8, n, WS); std::string tl; if (autotune(pi, X, Wt, Dref, nullptr, nullptr, ws, tl)) t_dense_i8_ref = time_calls([&]{ run(pi,&pi.algo,X,Wt,Dref,nullptr,nullptr,ws); }, 5); }
    double ti8 = sparse_time(H, n, CUDA_R_8I, CUDA_R_32I, CUSPARSE_COMPUTE_32I, false, &st);
    printf("{\"event\":\"sparse\",\"kind\":\"int8_2:4\",\"n\":%d,\"ms\":%.5f,\"status\":%d,\"dense_int8_ms\":%.5f}\n", n, ti8, st, t_dense_i8_ref);
    int st4 = 0;
    double tf4 = sparse_time(H, n, CUDA_R_4F_E2M1, CUDA_R_32F, CUSPARSE_COMPUTE_32F, true, &st4);
    printf("{\"event\":\"sparse\",\"kind\":\"nvfp4_2:4\",\"n\":%d,\"ms\":%.5f,\"status\":%d,\"name\":\"%s\"}\n", n, tf4, st4,
           cusparseLtGetErrorName((cusparseStatus_t)st4));
  }

  // Interleaved timing of the runnable routes, SM clock before/after.
  std::vector<double> ms1(routes.size());
  for (size_t i = 0; i < routes.size(); ++i) { if (!routes[i].f) continue; routes[i].f(); CK(cudaDeviceSynchronize()); ms1[i] = time_calls(routes[i].f, 3); }
  std::vector<std::vector<double>> per(routes.size());
  std::vector<int> iters(routes.size());
  for (size_t i = 0; i < routes.size(); ++i) iters[i] = std::max(1, int(25.0 / std::max(0.01, ms1[i])));
  for (int rd = 0; rd < rounds; ++rd)
    for (size_t i = 0; i < routes.size(); ++i) {
      if (!routes[i].f || ms1[i] == 0) continue;
      unsigned c0 = sm_clock();
      double ms = time_calls(routes[i].f, iters[i]);
      unsigned c1 = sm_clock();
      per[i].push_back(ms);
      printf("{\"event\":\"rep\",\"route\":\"%s\",\"round\":%d,\"iters\":%d,\"ms\":%.5f,\"clk0\":%u,\"clk1\":%u}\n",
             routes[i].name.c_str(), rd, iters[i], ms, c0, c1);
    }
  auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); size_t m = v.size(); return m ? (m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2])) : 0.0; };
  const double base = med(per[0]);
  const double macs = double(n) * n * n;
  for (size_t i = 0; i < routes.size(); ++i) {
    if (per[i].empty()) continue;
    double m = med(per[i]), mn = *std::min_element(per[i].begin(), per[i].end());
    printf("{\"event\":\"summary\",\"n\":%d,\"k\":%d,\"route\":\"%s\",\"median_ms\":%.5f,\"min_ms\":%.5f,"
           "\"ratio_vs_int8\":%.4f,\"int_macs_per_s\":%.4e}\n", n, k, routes[i].name.c_str(), m, mn, m / base, macs / (m * 1e-3));
  }
  return 0;
}
