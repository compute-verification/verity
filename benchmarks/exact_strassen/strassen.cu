// Exact integer C = A . B^T on the RTX PRO 6000 (sm_120): cuBLASLt's plain int8 and FP8 E4M3 GEMMs against
// Winograd's variant of Strassen (7 products, 15 additions) at one and two levels, sub-products on the same cuBLASLt
// int8 GEMM. A is batch x m x k, B batch x n x k (both row-major, K-major), C batch x m x n in s32.
//
//   strassen M N K BATCH AMAX SEED REPS [full]
//
// Entries are uniform in [-AMAX, AMAX]. Methods:
//   lt_int8      cuBLASLt int8 -> s32 (always)
//   lt_fp8       cuBLASLt E4M3 -> FP32 (AMAX <= 16, where E4M3 holds every entry)
//   win{L}_int8  Winograd at L levels, sub-products in int8: operands reach 4^L * AMAX, so only while that is <= 127
//   win{L}_dig   Winograd at L levels for wider entries: every summed operand v is split into base-128 digits
//                (lo in [-64, 63], hi = (v - lo) / 128), one-sided products into 2 GEMMs and two-sided ones into 3
//                (Karatsuba on the digits), all in int8
// Each Strassen call is three steps: the operand kernels (A, B -> every sub-operand, int8, one pass each), one batched
// cuBLASLt GEMM over all sub-products (s32), and one combine kernel (every sub-product read once, C written once).
// All arithmetic is mod 2^32 (mod 2^8 for packed operand sums whose true values fit int8), which is exact because
// |C| < 2^31.
//
// Gates, before any timing, on outputs and temporaries poisoned with 0xA5: every method against a CPU int64 reference
// (every entry with `full`, else 65536 sampled entries), and every entry against lt_int8 on the device; a control
// (one combine weight negated) must mismatch. Then REPS interleaved reps of equal-time items, each item's SM clock
// read through NVML. Prints JSON lines.
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CK(x)                                                                                       \
  do {                                                                                              \
    cudaError_t e_ = (x);                                                                           \
    if (e_ != cudaSuccess) {                                                                        \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));            \
      exit(3);                                                                                      \
    }                                                                                               \
  } while (0)
#define LT(x)                                                                                       \
  do {                                                                                              \
    cublasStatus_t s_ = (x);                                                                        \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                              \
      fprintf(stderr, "%s:%d %s: cublasStatus %d\n", __FILE__, __LINE__, #x, int(s_));              \
      exit(3);                                                                                      \
    }                                                                                               \
  } while (0)

constexpr int QMAX = 160, NBMAX = 16, SLOTS = 4;
__constant__ int8_t cA[SLOTS][QMAX * NBMAX], cB[SLOTS][QMAX * NBMAX], tA[SLOTS][QMAX], tB[SLOTS][QMAX];
__constant__ int32_t cW[SLOTS][NBMAX * QMAX];

__host__ __device__ inline uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__global__ void gen(int8_t* p, size_t count, uint64_t seed, int amax) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    p[i] = int8_t(int(mix(seed ^ mix(i)) % uint64_t(2 * amax + 1)) - amax);
}

__global__ void to_fp8(const int8_t* x, __nv_fp8_e4m3* y, size_t count) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    y[i] = __nv_fp8_e4m3(float(x[i]));
}

// Sub-operands whose sums fit int8: 16 bytes a thread from each of the 4^L blocks, packed byte-wise sums (mod 2^8).
template <int L>
__global__ void operands_packed(const int8_t* __restrict__ X, int rows, int cols, int batch, int Q, int slot, int side,
                                int8_t* __restrict__ Y) {
  constexpr int NB = 1 << (2 * L), S = 1 << L;
  const int rh = rows >> L, ch = cols >> L, cv = ch / 16;
  const size_t per = size_t(rh) * cv, total = per * batch, blk = size_t(rh) * ch;
  const int8_t* co = side ? cB[slot] : cA[slot];
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < total; t += size_t(gridDim.x) * blockDim.x) {
    const int b = int(t / per), r = int((t % per) / cv), c = int(t % cv);
    const int8_t* xb = X + size_t(b) * rows * cols + size_t(r) * cols + 16 * c;
    uint4 x[NB];
#pragma unroll
    for (int P = 0; P < NB; ++P) x[P] = __ldcs(reinterpret_cast<const uint4*>(xb + size_t((P / S) * rh) * cols + (P % S) * ch));
    int8_t* yb = Y + size_t(b) * Q * blk + size_t(r) * ch + 16 * c;
    for (int q = 0; q < Q; ++q) {
      uint4 a = make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int P = 0; P < NB; ++P) {
        const int k = co[q * NBMAX + P];
        if (k == 1) {
          a.x = __vadd4(a.x, x[P].x); a.y = __vadd4(a.y, x[P].y); a.z = __vadd4(a.z, x[P].z); a.w = __vadd4(a.w, x[P].w);
        } else if (k == -1) {
          a.x = __vsub4(a.x, x[P].x); a.y = __vsub4(a.y, x[P].y); a.z = __vsub4(a.z, x[P].z); a.w = __vsub4(a.w, x[P].w);
        }
      }
      *reinterpret_cast<uint4*>(yb + q * blk) = a;
    }
  }
}

__device__ inline int digit(int v, int t) {
  const int lo = ((v + 64) & 127) - 64, hi = (v - lo) >> 7;
  return t == 0 ? v : t == 1 ? hi : t == 2 ? lo : hi + lo;
}

// Sub-operands with digit splits: 4 bytes a thread from each block, sums in s32, then the digit, range-checked.
template <int L>
__global__ void operands_wide(const int8_t* __restrict__ X, int rows, int cols, int batch, int Q, int slot, int side,
                              int8_t* __restrict__ Y, int* bad) {
  constexpr int NB = 1 << (2 * L), S = 1 << L;
  const int rh = rows >> L, ch = cols >> L, cv = ch / 4;
  const size_t per = size_t(rh) * cv, total = per * batch, blk = size_t(rh) * ch;
  const int8_t* co = side ? cB[slot] : cA[slot];
  const int8_t* tr = side ? tB[slot] : tA[slot];
  int oob = 0;
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < total; t += size_t(gridDim.x) * blockDim.x) {
    const int b = int(t / per), r = int((t % per) / cv), c = int(t % cv);
    const int8_t* xb = X + size_t(b) * rows * cols + size_t(r) * cols + 4 * c;
    uint32_t x[NB];
#pragma unroll
    for (int P = 0; P < NB; ++P) x[P] = __ldcs(reinterpret_cast<const unsigned*>(xb + size_t((P / S) * rh) * cols + (P % S) * ch));
    int8_t* yb = Y + size_t(b) * Q * blk + size_t(r) * ch + 4 * c;
    for (int q = 0; q < Q; ++q) {
      int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
      for (int P = 0; P < NB; ++P) {
        const int k = co[q * NBMAX + P];
        if (k) {
          const int w = int(x[P]);
          s0 += k * ((w << 24) >> 24); s1 += k * ((w << 16) >> 24); s2 += k * ((w << 8) >> 24); s3 += k * (w >> 24);
        }
      }
      const int tt = tr[q];
      s0 = digit(s0, tt); s1 = digit(s1, tt); s2 = digit(s2, tt); s3 = digit(s3, tt);
      oob |= (s0 < -128) | (s0 > 127) | (s1 < -128) | (s1 > 127) | (s2 < -128) | (s2 > 127) | (s3 < -128) | (s3 > 127);
      *reinterpret_cast<uint32_t*>(yb + q * blk) =
          (uint32_t(s0) & 255u) | ((uint32_t(s1) & 255u) << 8) | ((uint32_t(s2) & 255u) << 16) | (uint32_t(s3) << 24);
    }
  }
  if (oob) atomicAdd(bad, 1);
}

// C's 4^L blocks from the Q sub-products: 4 words a thread, every sub-product read once, every C block written once.
template <int L>
__global__ void combine(const int32_t* __restrict__ M, int m, int n, int batch, int Q, int slot, int32_t* __restrict__ C) {
  constexpr int NB = 1 << (2 * L), S = 1 << L;
  const int mh = m >> L, nh = n >> L, nv = nh / 4;
  const size_t per = size_t(mh) * nv, total = per * batch, blk = size_t(mh) * nh;
  const int32_t* w = cW[slot];
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < total; t += size_t(gridDim.x) * blockDim.x) {
    const int b = int(t / per), r = int((t % per) / nv), c = int(t % nv);
    const int32_t* mb = M + size_t(b) * Q * blk + size_t(r) * nh + 4 * c;
    uint4 acc[NB];
#pragma unroll
    for (int O = 0; O < NB; ++O) acc[O] = make_uint4(0, 0, 0, 0);
    for (int q = 0; q < Q; ++q) {
      const uint4 x = __ldcs(reinterpret_cast<const uint4*>(mb + q * blk));
#pragma unroll
      for (int O = 0; O < NB; ++O) {
        const uint32_t k = uint32_t(w[O * QMAX + q]);
        if (k) { acc[O].x += k * x.x; acc[O].y += k * x.y; acc[O].z += k * x.z; acc[O].w += k * x.w; }
      }
    }
    int32_t* cb = C + size_t(b) * m * n + size_t(r) * n + 4 * c;
#pragma unroll
    for (int O = 0; O < NB; ++O) __stcs(reinterpret_cast<uint4*>(cb + size_t((O / S) * mh) * n + (O % S) * nh), acc[O]);
  }
}

__global__ void cmp_i32(const int32_t* a, const int32_t* b, size_t count, unsigned long long* bad) {
  unsigned long long z = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    z += a[i] != b[i];
  if (z) atomicAdd(bad, z);
}
__global__ void cmp_f32(const float* a, const int32_t* b, size_t count, unsigned long long* bad) {
  unsigned long long z = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    z += a[i] != float(b[i]) || fabsf(a[i]) >= 16777216.0f;
  if (z) atomicAdd(bad, z);
}
__global__ void gather(const void* C, int is_f32, const size_t* idx, int count, long long* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
    if (is_f32) {
      const float f = static_cast<const float*>(C)[idx[i]];
      out[i] = (f == floorf(f) && fabsf(f) < 16777216.0f) ? (long long)f : (long long)0x7fffffffffffll;
    } else {
      out[i] = static_cast<const int32_t*>(C)[idx[i]];
    }
  }
}

static int grid_for(size_t threads) {
  return int(std::min<size_t>((threads + 255) / 256, size_t(188) * 16));
}

// ---- cuBLASLt ------------------------------------------------------------------------------------------------------
struct Plan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatmulAlgo_t algo;
  bool fp8;
  int m, n, k, batch, algo_id = -1, tile = -1, stages = -1, splitk = -1;
  float tuned_ms = 0;
};
static cublasLtHandle_t g_lt;
static void* g_ws;
static const size_t WS = 64ull << 20;

static void layout(cublasLtMatrixLayout_t* l, cudaDataType_t t, int rows, int cols, int batch, long long stride) {
  LT(cublasLtMatrixLayoutCreate(l, t, rows, cols, rows));
  if (batch > 1) {
    LT(cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof batch));
    LT(cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof stride));
  }
}

// Our D (m x n row-major) = A (m x k) . B^T (B n x k): cuBLASLt's D^T (n x m column-major) = op(B) . A with op = T.
static Plan make_plan(bool fp8, int m, int n, int k, int batch) {
  Plan p{};
  p.fp8 = fp8; p.m = m; p.n = n; p.k = k; p.batch = batch;
  const cudaDataType_t it = fp8 ? CUDA_R_8F_E4M3 : CUDA_R_8I, ot = fp8 ? CUDA_R_32F : CUDA_R_32I;
  LT(cublasLtMatmulDescCreate(&p.op, fp8 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32I, fp8 ? CUDA_R_32F : CUDA_R_32I));
  cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
  LT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
  LT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
  layout(&p.la, it, k, n, batch, (long long)n * k);
  layout(&p.lb, it, k, m, batch, (long long)m * k);
  layout(&p.lc, ot, n, m, batch, (long long)m * n);
  return p;
}

static void run_plan(const Plan& p, const cublasLtMatmulAlgo_t* algo, const void* A, const void* B, void* D) {
  float one = 1.f, zero = 0.f;
  int32_t ione = 1, izero = 0;
  const void* al = p.fp8 ? static_cast<const void*>(&one) : &ione;
  const void* be = p.fp8 ? static_cast<const void*>(&zero) : &izero;
  LT(cublasLtMatmul(g_lt, p.op, al, B, p.la, A, p.lb, be, D, p.lc, D, p.lc, algo, g_ws, WS, 0));
}

static cudaEvent_t g_e0, g_e1;
static float time_ms(const std::function<void()>& f, int iters) {
  CK(cudaEventRecord(g_e0));
  for (int i = 0; i < iters; ++i) f();
  CK(cudaEventRecord(g_e1));
  CK(cudaEventSynchronize(g_e1));
  float ms;
  CK(cudaEventElapsedTime(&ms, g_e0, g_e1));
  return ms / iters;
}

// The fastest of cuBLASLt's top heuristic candidates on these operands.
static void tune(Plan& p, const void* A, const void* B, void* D) {
  cublasLtMatmulPreference_t pref;
  LT(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = WS;
  LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  LT(cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.la, p.lb, p.lc, p.lc, pref, 16, res, &got));
  cublasLtMatmulPreferenceDestroy(pref);
  if (got == 0) { fprintf(stderr, "no cuBLASLt algorithm\n"); exit(3); }
  float best = 1e30f;
  for (int i = 0; i < got; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    auto f = [&] { run_plan(p, &res[i].algo, A, B, D); };
    f();
    const float est = time_ms(f, 1);
    const int iters = std::max(1, std::min(200, int(8.0f / std::max(est, 1e-3f))));
    const float ms = std::min(time_ms(f, iters), time_ms(f, iters));
    if (ms < best) { best = ms; p.algo = res[i].algo; }
  }
  p.tuned_ms = best;
  size_t w;
  cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_ID, &p.algo_id, sizeof p.algo_id, &w);
  uint32_t u = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &u, sizeof u, &w); p.tile = int(u);
  cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &u, sizeof u, &w); p.stages = int(u);
  cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &p.splitk, sizeof p.splitk, &w);
}

// ---- Winograd tables ----------------------------------------------------------------------------------------------
struct Table {
  int L = 0, nb = 0;
  std::vector<std::vector<int>> a, b;  // per sub-product: coefficients over the 4^L blocks of A (B)
  std::vector<int> ta, tb;             // digit: 0 the whole sum, 1 hi, 2 lo, 3 hi + lo
  std::vector<std::vector<long long>> w;  // per C block: weights over the sub-products
  int Q() const { return int(a.size()); }
};

// C = A . B'; A's blocks 0..3 = A11 A12 A21 A22. B' = B^T, so B'_{lj} is block (j, l) of our B: B'12 is B21.
static Table winograd(int L) {
  const int SA[7][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 1, -1, -1}, {0, 0, 0, 1}, {0, 0, 1, 1}, {-1, 0, 1, 1}, {1, 0, -1, 0}};
  const int SBp[7][4] = {{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {1, -1, -1, 1}, {-1, 1, 0, 0}, {1, -1, 0, 1}, {0, -1, 0, 1}};
  const int W[4][7] = {{1, 1, 0, 0, 0, 0, 0}, {1, 0, 1, 0, 1, 1, 0}, {1, 0, 0, -1, 0, 1, 1}, {1, 0, 0, 0, 1, 1, 1}};
  int SB[7][4];
  for (int q = 0; q < 7; ++q) { SB[q][0] = SBp[q][0]; SB[q][1] = SBp[q][2]; SB[q][2] = SBp[q][1]; SB[q][3] = SBp[q][3]; }
  Table t;
  t.L = L;
  t.nb = 1 << (2 * L);
  const int S = 1 << L;
  auto blockidx = [&](const std::vector<int>& p) {  // per-level quadrant indices, outermost first -> block index
    int R = 0, C = 0;
    for (int x : p) { R = 2 * R + x / 2; C = 2 * C + x % 2; }
    return R * S + C;
  };
  int nq = 1;
  for (int l = 0; l < L; ++l) nq *= 7;
  t.w.assign(t.nb, std::vector<long long>(nq, 0));
  for (int q = 0; q < nq; ++q) {
    std::vector<int> qs(L);
    for (int l = L - 1, x = q; l >= 0; --l, x /= 7) qs[l] = x % 7;
    std::vector<int> va(t.nb, 0), vb(t.nb, 0);
    for (int P = 0; P < t.nb; ++P) {
      std::vector<int> ps(L);
      for (int l = L - 1, x = P; l >= 0; --l, x /= 4) ps[l] = x % 4;
      int ka = 1, kb = 1;
      for (int l = 0; l < L; ++l) { ka *= SA[qs[l]][ps[l]]; kb *= SB[qs[l]][ps[l]]; }
      va[blockidx(ps)] = ka;
      vb[blockidx(ps)] = kb;
      int kw = 1;
      for (int l = 0; l < L; ++l) kw *= W[ps[l]][qs[l]];
      t.w[blockidx(ps)][q] = kw;
    }
    t.a.push_back(va);
    t.b.push_back(vb);
  }
  t.ta.assign(nq, 0);
  t.tb.assign(nq, 0);
  return t;
}

// Splits every summed operand into base-128 digits: S.T = 128 Sh.T + Sl.T, or with both sides split
// S.T = 16256 Sh.Th + 128 (Sh + Sl).(Th + Tl) - 127 Sl.Tl.
static Table digits(const Table& in) {
  Table t;
  t.L = in.L;
  t.nb = in.nb;
  t.w.assign(t.nb, {});
  auto nnz = [](const std::vector<int>& v) { int z = 0; for (int x : v) z += x != 0; return z; };
  for (int q = 0; q < in.Q(); ++q) {
    const bool sa = nnz(in.a[q]) > 1, sb = nnz(in.b[q]) > 1;
    std::vector<std::array<long long, 3>> cs;
    if (!sa && !sb) cs = {{0, 0, 1}};
    else if (sa && !sb) cs = {{1, 0, 128}, {2, 0, 1}};
    else if (!sa && sb) cs = {{0, 1, 128}, {0, 2, 1}};
    else cs = {{1, 1, 16256}, {2, 2, -127}, {3, 3, 128}};
    for (auto& c : cs) {
      t.a.push_back(in.a[q]);
      t.b.push_back(in.b[q]);
      t.ta.push_back(int(c[0]));
      t.tb.push_back(int(c[1]));
      for (int O = 0; O < t.nb; ++O) t.w[O].push_back(in.w[O][q] * c[2]);
    }
  }
  return t;
}

static void upload(const Table& t, int slot) {
  std::vector<int8_t> a(QMAX * NBMAX, 0), b(QMAX * NBMAX, 0), ta(QMAX, 0), tb(QMAX, 0);
  std::vector<int32_t> w(NBMAX * QMAX, 0);
  if (t.Q() > QMAX) { fprintf(stderr, "Q %d > QMAX\n", t.Q()); exit(3); }
  for (int q = 0; q < t.Q(); ++q) {
    for (int P = 0; P < t.nb; ++P) { a[q * NBMAX + P] = int8_t(t.a[q][P]); b[q * NBMAX + P] = int8_t(t.b[q][P]); }
    ta[q] = int8_t(t.ta[q]);
    tb[q] = int8_t(t.tb[q]);
    for (int O = 0; O < t.nb; ++O) w[O * QMAX + q] = int32_t(uint32_t(uint64_t(t.w[O][q])));
  }
  CK(cudaMemcpyToSymbol(cA, a.data(), a.size(), slot * a.size()));
  CK(cudaMemcpyToSymbol(cB, b.data(), b.size(), slot * b.size()));
  CK(cudaMemcpyToSymbol(tA, ta.data(), ta.size(), slot * ta.size()));
  CK(cudaMemcpyToSymbol(tB, tb.data(), tb.size(), slot * tb.size()));
  CK(cudaMemcpyToSymbol(cW, w.data(), w.size() * 4, slot * w.size() * 4));
}

// ---- NVML (dlopen: the driver's library, no link-time dependency) ---------------------------------------------------
struct Nvml {
  void* dev = nullptr;
  int (*clock)(void*, int, unsigned*) = nullptr;
  int (*reasons)(void*, unsigned long long*) = nullptr;
  void init() {
    void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!h) return;
    auto in = reinterpret_cast<int (*)()>(dlsym(h, "nvmlInit_v2"));
    auto by = reinterpret_cast<int (*)(const char*, void**)>(dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2"));
    clock = reinterpret_cast<int (*)(void*, int, unsigned*)>(dlsym(h, "nvmlDeviceGetClockInfo"));
    reasons = reinterpret_cast<int (*)(void*, unsigned long long*)>(dlsym(h, "nvmlDeviceGetCurrentClocksEventReasons"));
    if (!reasons) reasons = reinterpret_cast<int (*)(void*, unsigned long long*)>(dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons"));
    if (!in || !by || !clock || in() != 0) return;
    char bus[64];
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
    if (by(bus, &dev) != 0) dev = nullptr;
  }
  unsigned sm() { unsigned c = 0; if (dev) clock(dev, 1, &c); return c; }
  unsigned long long why() { unsigned long long r = 0; if (dev && reasons) reasons(dev, &r); return r; }
};

// ---- methods ------------------------------------------------------------------------------------------------------
struct Method {
  std::string name, format;
  int level = 0, slot = -1, Q = 0;
  bool wide = false, gate_only = false;
  Plan plan;
  void* out = nullptr;
  bool f32 = false;
  std::function<void()> ops, gemm, comb;
  std::function<void()> call() const {
    return [this] { if (ops) ops(); gemm(); if (comb) comb(); };
  }
};

int main(int argc, char** argv) {
  if (argc < 8) { fprintf(stderr, "usage: strassen M N K BATCH AMAX SEED REPS [full]\n"); return 2; }
  const int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]), batch = atoi(argv[4]), amax = atoi(argv[5]);
  const uint64_t seed = strtoull(argv[6], nullptr, 0);
  const int reps = atoi(argv[7]);
  const bool full = argc > 8 && std::string(argv[8]) == "full";
  CK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  LT(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, WS));
  CK(cudaEventCreate(&g_e0));
  CK(cudaEventCreate(&g_e1));
  Nvml nv;
  nv.init();
  char uuid[64];
  {
    const unsigned char* u = reinterpret_cast<const unsigned char*>(&prop.uuid);
    snprintf(uuid, sizeof uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1],
             u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
  }
  const char* tag = getenv("STR_TAG") ? getenv("STR_TAG") : "";
  printf("{\"kind\":\"config\",\"tag\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"batch\":%d,\"amax\":%d,\"seed\":%llu,"
         "\"reps\":%d,\"full\":%d,\"gpu\":\"%s\",\"uuid\":\"%s\",\"sms\":%d,\"cublaslt\":%zu,\"cudart\":%d}\n",
         tag, m, n, k, batch, amax, (unsigned long long)seed, reps, int(full), prop.name, uuid, prop.multiProcessorCount,
         cublasLtGetVersion(), CUDART_VERSION);
  fflush(stdout);

  const size_t na = size_t(batch) * m * k, nbs = size_t(batch) * n * k, nc = size_t(batch) * m * n;
  int8_t *A, *B;
  CK(cudaMalloc(&A, na));
  CK(cudaMalloc(&B, nbs));
  gen<<<grid_for(na), 256>>>(A, na, seed * 2 + 1, amax);
  gen<<<grid_for(nbs), 256>>>(B, nbs, seed * 2 + 2, amax);
  CK(cudaGetLastError());

  std::vector<Method> ms;
  {
    Method x;
    x.name = "lt_int8"; x.format = "int8";
    ms.push_back(x);
  }
  __nv_fp8_e4m3 *A8 = nullptr, *B8 = nullptr;
  if (amax <= 16) {
    CK(cudaMalloc(&A8, na));
    CK(cudaMalloc(&B8, nbs));
    to_fp8<<<grid_for(na), 256>>>(A, A8, na);
    to_fp8<<<grid_for(nbs), 256>>>(B, B8, nbs);
    Method x;
    x.name = "lt_fp8"; x.format = "e4m3"; x.f32 = true;
    ms.push_back(x);
  }
  std::vector<Table> tables;
  for (int L = 1; L <= 2; ++L) {
    if ((m >> L) % 16 || (n >> L) % 16 || (k >> L) % 16) continue;
    int reach = amax;
    for (int l = 0; l < L; ++l) reach *= 4;
    Table t = winograd(L);
    Method x;
    x.level = L;
    if (reach <= 127) { x.name = "win" + std::to_string(L) + "_int8"; x.format = "int8"; }
    else { t = digits(t); x.name = "win" + std::to_string(L) + "_dig"; x.format = "int8 base-128 digits"; x.wide = true; }
    x.Q = t.Q();
    x.slot = int(tables.size());
    tables.push_back(t);
    ms.push_back(x);
  }
  {  // control: the level-1 method with one combine weight negated (C22's M5 term); its gate must fail
    for (auto& y : ms)
      if (y.level == 1) {
        Table t = tables[y.slot];
        int qn = 0;
        for (int q = 0; q < t.Q(); ++q) if (t.w[3][q] != 0 && t.w[1][q] != 0 && t.w[0][q] == 0) { qn = q; break; }
        t.w[3][qn] = -t.w[3][qn];
        Method x = y;
        x.name = y.name + "_ctl"; x.gate_only = true; x.slot = int(tables.size());
        tables.push_back(t);
        ms.push_back(x);
        break;
      }
  }
  for (size_t i = 0; i < tables.size(); ++i) upload(tables[i], int(i));

  // shared temporaries, sized for the largest Strassen method that fits beside the outputs
  size_t free_b, total_b;
  CK(cudaMemGetInfo(&free_b, &total_b));
  size_t outs = nc * 4 * (amax <= 16 ? 3 : 2);
  size_t maxM = 0, maxS = 0;
  for (auto& x : ms) {
    if (!x.level) continue;
    const size_t blkA = size_t(m >> x.level) * (k >> x.level), blkB = size_t(n >> x.level) * (k >> x.level),
                 blkC = size_t(m >> x.level) * (n >> x.level);
    const size_t needM = size_t(batch) * x.Q * blkC * 4, needS = size_t(batch) * x.Q * (blkA + blkB);
    if (outs + needM + needS + (2ull << 30) > free_b) { x.gate_only = true; x.name += "_SKIPPED_MEMORY"; continue; }
    maxM = std::max(maxM, needM);
    maxS = std::max(maxS, needS);
  }
  int32_t *Cref, *Cs = nullptr, *Mt = nullptr;
  float* Cf = nullptr;
  int8_t* St = nullptr;
  int* dbad;
  unsigned long long* dmis;
  CK(cudaMalloc(&Cref, nc * 4));
  if (amax <= 16) CK(cudaMalloc(&Cf, nc * 4));
  CK(cudaMalloc(&Cs, nc * 4));
  if (maxM) CK(cudaMalloc(&Mt, maxM));
  if (maxS) CK(cudaMalloc(&St, maxS));
  CK(cudaMalloc(&dbad, sizeof(int)));
  CK(cudaMalloc(&dmis, sizeof(unsigned long long)));
  CK(cudaMemset(dbad, 0, sizeof(int)));

  for (auto& x : ms) {
    if (x.name.find("SKIPPED") != std::string::npos) continue;
    if (x.name == "lt_int8") {
      x.plan = make_plan(false, m, n, k, batch); x.out = Cref;
      Method* px = &x;
      x.gemm = [px, A, B] { run_plan(px->plan, &px->plan.algo, A, B, px->out); };
    } else if (x.name == "lt_fp8") {
      x.plan = make_plan(true, m, n, k, batch); x.out = Cf;
      Method* px = &x;
      x.gemm = [px, A8, B8] { run_plan(px->plan, &px->plan.algo, A8, B8, px->out); };
    } else {
      const int L = x.level, mh = m >> L, nh = n >> L, kh = k >> L, Q = x.Q, slot = x.slot;
      x.plan = make_plan(false, mh, nh, kh, batch * Q);
      x.out = Cs;
      int8_t* SAp = St;
      int8_t* SBp = St + size_t(batch) * Q * mh * kh;
      Method* px = &x;
      const bool wide = x.wide;
      x.ops = [=] {
        const size_t ta = size_t(batch) * mh * (kh / (wide ? 4 : 16)), tb = size_t(batch) * nh * (kh / (wide ? 4 : 16));
        if (wide) {
          if (L == 1) {
            operands_wide<1><<<grid_for(ta), 256>>>(A, m, k, batch, Q, slot, 0, SAp, dbad);
            operands_wide<1><<<grid_for(tb), 256>>>(B, n, k, batch, Q, slot, 1, SBp, dbad);
          } else {
            operands_wide<2><<<grid_for(ta), 256>>>(A, m, k, batch, Q, slot, 0, SAp, dbad);
            operands_wide<2><<<grid_for(tb), 256>>>(B, n, k, batch, Q, slot, 1, SBp, dbad);
          }
        } else if (L == 1) {
          operands_packed<1><<<grid_for(ta), 256>>>(A, m, k, batch, Q, slot, 0, SAp);
          operands_packed<1><<<grid_for(tb), 256>>>(B, n, k, batch, Q, slot, 1, SBp);
        } else {
          operands_packed<2><<<grid_for(ta), 256>>>(A, m, k, batch, Q, slot, 0, SAp);
          operands_packed<2><<<grid_for(tb), 256>>>(B, n, k, batch, Q, slot, 1, SBp);
        }
      };
      x.gemm = [px, SAp, SBp, Mt] { run_plan(px->plan, &px->plan.algo, SAp, SBp, Mt); };
      x.comb = [=] {
        const size_t tc = size_t(batch) * mh * (nh / 4);
        if (L == 1) combine<1><<<grid_for(tc), 256>>>(Mt, m, n, batch, Q, slot, Cs);
        else combine<2><<<grid_for(tc), 256>>>(Mt, m, n, batch, Q, slot, Cs);
      };
    }
  }
  // tune each plan on its own operands (the Strassen plans after their operand kernels ran once)
  for (auto& x : ms) {
    if (x.name.find("SKIPPED") != std::string::npos) continue;
    if (x.ops) x.ops();
    if (x.name == "lt_int8") tune(x.plan, A, B, Cref);
    else if (x.name == "lt_fp8") tune(x.plan, A8, B8, Cf);
    else {
      const int L = x.level;
      tune(x.plan, St, St + size_t(batch) * x.Q * (m >> L) * (k >> L), Mt);
    }
    CK(cudaDeviceSynchronize());
    printf("{\"kind\":\"plan\",\"method\":\"%s\",\"format\":\"%s\",\"level\":%d,\"products\":%d,\"gemm_m\":%d,"
           "\"gemm_n\":%d,\"gemm_k\":%d,\"gemm_batch\":%d,\"algo\":%d,\"tile\":%d,\"stages\":%d,\"splitk\":%d,"
           "\"tuned_gemm_ms\":%.6f}\n",
           x.name.c_str(), x.format.c_str(), x.level, x.level ? x.Q : 1, x.plan.m, x.plan.n, x.plan.k, x.plan.batch,
           x.plan.algo_id, x.plan.tile, x.plan.stages, x.plan.splitk, x.plan.tuned_ms);
    fflush(stdout);
  }

  // ---- gates ------------------------------------------------------------------------------------------------------
  std::vector<int8_t> hA(na), hB(nbs);
  CK(cudaMemcpy(hA.data(), A, na, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hB.data(), B, nbs, cudaMemcpyDeviceToHost));
  std::vector<size_t> idx;
  if (full) {
    idx.resize(nc);
    for (size_t i = 0; i < nc; ++i) idx[i] = i;
  } else {
    idx.resize(65536);
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = mix(seed * 7919 + i) % nc;
    idx[0] = 0; idx[1] = nc - 1;
  }
  std::vector<long long> ref(idx.size());
#pragma omp parallel for schedule(static)
  for (long long s = 0; s < (long long)idx.size(); ++s) {
    const size_t e = idx[s], b = e / (size_t(m) * n), i = (e / n) % m, j = e % n;
    const int8_t* a = &hA[(b * m + i) * k];
    const int8_t* bb = &hB[(b * n + j) * k];
    long long z = 0;
    for (int t = 0; t < k; ++t) z += int(a[t]) * int(bb[t]);
    ref[s] = z;
  }
  size_t* didx;
  long long* dgot;
  CK(cudaMalloc(&didx, idx.size() * sizeof(size_t)));
  CK(cudaMalloc(&dgot, idx.size() * sizeof(long long)));
  CK(cudaMemcpy(didx, idx.data(), idx.size() * sizeof(size_t), cudaMemcpyHostToDevice));
  std::vector<long long> got(idx.size());
  bool all_ok = true, ctl_rejected = true;
  for (auto& x : ms) {
    if (x.name.find("SKIPPED") != std::string::npos) {
      printf("{\"kind\":\"gate\",\"method\":\"%s\",\"skipped\":\"memory\"}\n", x.name.c_str());
      continue;
    }
    CK(cudaMemset(x.out, 0xA5, nc * 4));
    if (x.level) {
      CK(cudaMemset(Mt, 0xA5, maxM));
      CK(cudaMemset(St, 0xA5, maxS));
    }
    CK(cudaMemset(dbad, 0, sizeof(int)));
    x.call()();
    CK(cudaDeviceSynchronize());
    int bad = 0;
    CK(cudaMemcpy(&bad, dbad, sizeof bad, cudaMemcpyDeviceToHost));
    gather<<<grid_for(idx.size()), 256>>>(x.out, int(x.f32), didx, int(idx.size()), dgot);
    CK(cudaMemcpy(got.data(), dgot, idx.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    size_t cpu_bad = 0;
    for (size_t s = 0; s < idx.size(); ++s) cpu_bad += got[s] != ref[s];
    unsigned long long dev_bad = 0;
    if (x.name != "lt_int8") {
      CK(cudaMemset(dmis, 0, sizeof *dmis));
      if (x.f32) cmp_f32<<<grid_for(nc), 256>>>(Cf, Cref, nc, dmis);
      else cmp_i32<<<grid_for(nc), 256>>>(static_cast<int32_t*>(x.out), Cref, nc, dmis);
      CK(cudaMemcpy(&dev_bad, dmis, sizeof dev_bad, cudaMemcpyDeviceToHost));
    }
    const bool ok = cpu_bad == 0 && dev_bad == 0 && bad == 0;
    if (x.gate_only) ctl_rejected = ctl_rejected && !ok;
    else all_ok = all_ok && ok;
    printf("{\"kind\":\"gate\",\"method\":\"%s\",\"control\":%d,\"cpu_checked\":%zu,\"cpu_mismatch\":%zu,"
           "\"device_checked\":%zu,\"device_mismatch\":%llu,\"operand_out_of_range\":%d,\"pass\":%d}\n",
           x.name.c_str(), int(x.gate_only), idx.size(), cpu_bad, x.name == "lt_int8" ? size_t(0) : nc, dev_bad, bad,
           int(ok));
    fflush(stdout);
  }
  printf("{\"kind\":\"gates\",\"all_pass\":%d,\"control_rejected\":%d}\n", int(all_ok), int(ctl_rejected));
  fflush(stdout);
  if (!all_ok || !ctl_rejected) return 1;

  // ---- timing: interleaved reps of equal-time items ---------------------------------------------------------------
  std::vector<Method*> timed;
  for (auto& x : ms) if (!x.gate_only) timed.push_back(&x);
  const float item_ms = 30.f;
  std::vector<int> iters(timed.size());
  for (size_t i = 0; i < timed.size(); ++i) {
    auto f = timed[i]->call();
    f();
    const float est = time_ms(f, 2);
    iters[i] = std::max(1, std::min(2000, int(item_ms / std::max(est, 1e-3f))));
  }
  auto median = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  std::vector<std::vector<float>> t(timed.size());
  std::vector<std::vector<unsigned>> clk(timed.size());
  unsigned long long reasons = 0;
  for (int r = 0; r < reps; ++r)
    for (size_t i = 0; i < timed.size(); ++i) {
      t[i].push_back(time_ms(timed[i]->call(), iters[i]));
      clk[i].push_back(nv.sm());
      reasons |= nv.why();
    }
  // the split of each Strassen call, timed the same way
  std::vector<std::array<float, 3>> split(timed.size(), {0, 0, 0});
  for (size_t i = 0; i < timed.size(); ++i) {
    if (!timed[i]->level) continue;
    std::vector<float> a, g, c;
    for (int r = 0; r < std::max(3, reps / 2); ++r) {
      a.push_back(time_ms(timed[i]->ops, iters[i]));
      g.push_back(time_ms(timed[i]->gemm, iters[i]));
      c.push_back(time_ms(timed[i]->comb, iters[i]));
    }
    split[i] = {median(a), median(g), median(c)};
  }
  float base = 1e30f;
  std::string base_name;
  for (size_t i = 0; i < timed.size(); ++i)
    if (!timed[i]->level && median(t[i]) < base) { base = median(t[i]); base_name = timed[i]->name; }
  for (size_t i = 0; i < timed.size(); ++i) {
    Method& x = *timed[i];
    const float med = median(t[i]);
    const double ops = 2.0 * m * n * double(k) * batch;
    const int L = x.level;
    // bytes the additions move: operand kernels read A, B and write every sub-operand; the GEMMs write Q sub-products
    // the plain GEMM's C does not; the combine reads them all and writes C
    double op_bytes = 0, comb_bytes = 0;
    if (L) {
      const double sa = double(batch) * (m >> L) * (k >> L), sb = double(batch) * (n >> L) * (k >> L);
      op_bytes = double(na + nbs) + x.Q * (sa + sb);
      comb_bytes = double(batch) * x.Q * (m >> L) * double(n >> L) * 4 + double(nc) * 4;
    }
    printf("{\"kind\":\"time\",\"method\":\"%s\",\"format\":\"%s\",\"level\":%d,\"products\":%d,\"iters\":%d,"
           "\"median_ms\":%.6f,\"min_ms\":%.6f,\"tops\":%.2f,\"baseline\":\"%s\",\"baseline_ms\":%.6f,"
           "\"one_plus_gain\":%.4f,\"ops_ms\":%.6f,\"gemm_ms\":%.6f,\"combine_ms\":%.6f,\"gemm_only_one_plus_gain\":%.4f,"
           "\"operand_bytes\":%.0f,\"combine_bytes\":%.0f,\"ops_gbps\":%.1f,\"combine_gbps\":%.1f,\"ms\":[",
           x.name.c_str(), x.format.c_str(), L, L ? x.Q : 1, iters[i], med, *std::min_element(t[i].begin(), t[i].end()),
           ops / med / 1e9, base_name.c_str(), base, base / med, split[i][0], split[i][1], split[i][2],
           L ? base / split[i][1] : 0.0, op_bytes, comb_bytes, L ? op_bytes / split[i][0] / 1e6 : 0.0,
           L ? comb_bytes / split[i][2] / 1e6 : 0.0);
    for (size_t r = 0; r < t[i].size(); ++r) printf("%s%.6f", r ? "," : "", t[i][r]);
    printf("],\"sm_mhz\":[");
    for (size_t r = 0; r < clk[i].size(); ++r) printf("%s%u", r ? "," : "", clk[i][r]);
    printf("]}\n");
  }
  printf("{\"kind\":\"clocks\",\"event_reasons_or\":%llu}\n", reasons);
  return 0;
}
