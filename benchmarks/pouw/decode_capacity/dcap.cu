// What share of one RTX PRO 6000's int8 tensor capacity Llama-3.1-8B's matmuls use, decode and prefill (CAP, cmp-13).
//
// Nominal work is the int8 MACs the served matmuls execute: per token, the linears (qkv 4096x6144, o 4096x4096,
// gate_up 4096x28672, down 14336x4096, 32 layers, and lm_head 4096x128256 in decode) and attention (QK^T and P.V,
// 32 query heads over 8 KV heads of 128, against the context). Capacity is the int8 IMMA peak, 1024 MACs/SM/clk, at the
// SM clock NVML reads during each rep. Clocks are read, never set.
//
// Modes (one JSON line each): gate (cuBLASLt int8 and the two attention kernels bit for bit against the host), bw (DRAM
// read bandwidth), lin M (one step's 32 layers of linears at M rows, distinct weights per layer so L2 never holds them,
// plus lm_head when M <= 256), att B L (32 layers of decode attention, distinct int8 KV per layer), step B L (the two
// interleaved layer by layer, as a decode step runs them). Each step is one CUDA graph; reps are timed by events.
// Norms, RoPE, softmax and quantization are left out, so a step here is a lower bound on a served step.
// fill B L FILL_SMS SHAPE: the step beside certified filler on a green-context split. The filler is T2's chained int8 GEMM
// (C_s = X_s W_s^T, the s32 output read as the next s8 X), on FILL_SMS SMs, kept queued by a host thread.
// dgate, dalone, done, dfill B L PLACE FILL_SMS W NC S: the DRAM-free filler (FILL; see "dfill" below): its receipts bit
// for bit against the host, its rate alone, one chunk for ncu, and the step beside it, on a green split (PLACE split) or
// co-resident on every SM at the lowest stream priority with the step at the highest (PLACE co).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a dcap.cu -lcublasLt -lnvidia-ml -ldl
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)
#define CL(x)                                                                         \
  do {                                                                                \
    cublasStatus_t s_ = (x);                                                          \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                \
      fprintf(stderr, "cuBLASLt %s at %s:%d: status %d\n", #x, __FILE__, __LINE__, (int)s_); \
      exit(2);                                                                        \
    }                                                                                 \
  } while (0)

constexpr int LAYERS = 32, HID = 4096, QKV = 6144, FF = 14336, VOCAB = 128256, HQ = 32, HKV = 8, HD = 128;
constexpr size_t KV_TOKEN_BYTES = (size_t)LAYERS * HKV * HD * 2;  // K and V, int8, all layers: 64 KiB a token

// ---------------------------------------------------------------- kernels
__global__ void bw_read(const int4* __restrict__ p, size_t n16, int* out) {
  const size_t st = (size_t)gridDim.x * blockDim.x;
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  int acc = 0;
  for (; i + 3 * st < n16; i += 4 * st) {
    const int4 a = __ldcs(p + i), b = __ldcs(p + i + st), c = __ldcs(p + i + 2 * st), d = __ldcs(p + i + 3 * st);
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  for (; i < n16; i += st) {
    const int4 a = __ldcs(p + i);
    acc ^= a.x ^ a.y ^ a.z ^ a.w;
  }
  if (acc == 0x7f3a5c21) out[0] = acc;
}

// S[b][4g+h][key] = q[b][4g+h][:] . K[b][g][key][:], s8 x s8 -> s32. K row-major [B][HKV][L][HD]. Eight lanes a key
// (16 B each), so a warp instruction reads 512 contiguous bytes; 8 warps x 32 keys a block.
constexpr int QK_W = 8, QK_KW = 32;
__global__ void __launch_bounds__(256) qk_dec(const int8_t* __restrict__ K, const int8_t* __restrict__ q, int* __restrict__ S,
                                              int L) {
  const int b = blockIdx.z, g = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31, sub = lane & 7, kr = lane >> 3;
  const int4* Kp = reinterpret_cast<const int4*>(K + ((size_t)b * HKV + g) * L * HD);
  int4 qv[4];
#pragma unroll
  for (int h = 0; h < 4; h++) qv[h] = *reinterpret_cast<const int4*>(q + ((size_t)b * HQ + g * 4 + h) * HD + sub * 16);
  const int key0 = (blockIdx.x * QK_W + w) * QK_KW;
  int4 kv[8];
#pragma unroll
  for (int it = 0; it < 8; it++) {
    const int key = key0 + it * 4 + kr;
    kv[it] = key < L ? __ldcs(Kp + (size_t)key * 8 + sub) : make_int4(0, 0, 0, 0);
  }
#pragma unroll
  for (int it = 0; it < 8; it++) {
    const int key = key0 + it * 4 + kr;
    int s[4];
#pragma unroll
    for (int h = 0; h < 4; h++)
      s[h] = __dp4a(kv[it].x, qv[h].x, __dp4a(kv[it].y, qv[h].y, __dp4a(kv[it].z, qv[h].z, __dp4a(kv[it].w, qv[h].w, 0))));
#pragma unroll
    for (int off = 1; off < 8; off <<= 1)
#pragma unroll
      for (int h = 0; h < 4; h++) s[h] += __shfl_xor_sync(0xffffffffu, s[h], off);
    const int v = sub == 0 ? s[0] : sub == 1 ? s[1] : sub == 2 ? s[2] : s[3];
    if (sub < 4 && key < L) S[((size_t)b * HQ + g * 4 + sub) * L + key] = v;
  }
}

// O[b][4g+h][d] += sum_key P[b][4g+h][key] V[b][g][key][d], s8 x s8 -> s32. V is stored four keys interleaved,
// [B][HKV][L/4][HD][4], so one s32 word holds a dimension's four keys and one dp4a takes four MACs. A warp instruction
// reads one 512-byte group (four keys, every dimension); 8 warps x 16 groups a block, reduced in shared memory, then one
// atomic a (head, dimension) a block.
constexpr int PV_W = 8, PV_GW = 16;
__global__ void __launch_bounds__(256) pv_dec(const int8_t* __restrict__ V, const int8_t* __restrict__ P, int* __restrict__ O,
                                              int L) {
  __shared__ int red[PV_W][4][HD];
  const int b = blockIdx.z, g = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int G = L / 4;
  const int4* Vp = reinterpret_cast<const int4*>(V + ((size_t)b * HKV + g) * L * HD);
  const int* Pp = reinterpret_cast<const int*>(P + ((size_t)b * HQ + g * 4) * L);  // head h's group gk at Pp[h*G + gk]
  int acc[4][4] = {};
  const int g0 = (blockIdx.x * PV_W + w) * PV_GW;
#pragma unroll
  for (int half = 0; half < 2; half++) {
    int4 vv[8];
#pragma unroll
    for (int it = 0; it < 8; it++) {
      const int gk = g0 + half * 8 + it;
      vv[it] = gk < G ? __ldcs(Vp + (size_t)gk * 32 + lane) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int it = 0; it < 8; it++) {
      const int gk = g0 + half * 8 + it;
      if (gk >= G) break;
#pragma unroll
      for (int h = 0; h < 4; h++) {
        const int pw = __ldg(Pp + (size_t)h * G + gk);
        acc[h][0] = __dp4a(vv[it].x, pw, acc[h][0]);
        acc[h][1] = __dp4a(vv[it].y, pw, acc[h][1]);
        acc[h][2] = __dp4a(vv[it].z, pw, acc[h][2]);
        acc[h][3] = __dp4a(vv[it].w, pw, acc[h][3]);
      }
    }
  }
#pragma unroll
  for (int h = 0; h < 4; h++)
#pragma unroll
    for (int j = 0; j < 4; j++) red[w][h][lane * 4 + j] = acc[h][j];
  __syncthreads();
  for (int e = threadIdx.x; e < 4 * HD; e += blockDim.x) {
    const int h = e / HD, d = e % HD;
    int s = 0;
#pragma unroll
    for (int ww = 0; ww < PV_W; ww++) s += red[ww][h][d];
    atomicAdd(O + ((size_t)b * HQ + g * 4 + h) * HD + d, s);
  }
}

// R[r][j] = A[r·rs] . W[j], s8 x s8 -> s32 by dp4a, for the sampled rows r < mr (k a multiple of 16)
__global__ void ref_rows(const int8_t* __restrict__ A, const int8_t* __restrict__ W, int* __restrict__ R, int mr, int rs,
                         int n, int k) {
  const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= (size_t)mr * n) return;
  const int r = (int)(idx / n), j = (int)(idx % n);
  const int4* a = reinterpret_cast<const int4*>(A + (size_t)r * rs * k);
  const int4* w = reinterpret_cast<const int4*>(W + (size_t)j * k);
  int s = 0;
  for (int t = 0; t < k / 16; t++) {
    const int4 x = a[t], y = __ldg(w + t);
    s = __dp4a(x.x, y.x, __dp4a(x.y, y.y, __dp4a(x.z, y.z, __dp4a(x.w, y.w, s))));
  }
  R[idx] = s;
}
__global__ void count_neq_rows(const int* __restrict__ D, const int* __restrict__ R, int mr, int rs, int n,
                               unsigned long long* c) {
  const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= (size_t)mr * n) return;
  const int r = (int)(idx / n), j = (int)(idx % n);
  if (D[(size_t)r * rs * n + j] != R[idx]) atomicAdd(c, 1ull);
}

__global__ void fill_s8(int8_t* p, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ seed;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
    p[i] = (int8_t)((int)(x & 0xff) - 128);
  }
}

// ---------------------------------------------------------------- cuBLASLt int8: D (m x n, row-major) = A (m x k) W^T (W n x k)
struct Lt {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatmulAlgo_t algo;
  int algo_id = -2;
  int rejected = 0;  // heuristic candidates whose output differed from ref_rows
  float ms = 0;
};
static cublasLtHandle_t g_lt;
static void* g_ws;
static const size_t g_wsz = 64u << 20;
static int* g_ref;                 // ref_rows' output, up to 256 rows x VOCAB
static unsigned long long* g_cnt;  // mismatch counter
static void lt_run(const Lt& L, const void* A, const void* W, void* D, cudaStream_t st) {
  const int32_t one = 1, zero = 0;
  CL(cublasLtMatmul(g_lt, L.op, &one, W, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &L.algo, g_ws, g_wsz, st));
}
// the fastest of the heuristic's top 16 over 5 calls, timed on the given buffers, among the candidates whose output equals
// ref_rows on every row (m <= 256) or on 256 rows spread over m: a configuration that computes something else is never timed.
static Lt lt_make(int m, int n, int k, const void* A, const void* W, void* D, cudaStream_t st) {
  Lt L;
  CL(cublasLtMatmulDescCreate(&L.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CL(cublasLtMatrixLayoutCreate(&L.la, CUDA_R_8I, k, n, k));
  CL(cublasLtMatrixLayoutCreate(&L.lb, CUDA_R_8I, k, m, k));
  CL(cublasLtMatrixLayoutCreate(&L.ld, CUDA_R_32I, n, m, n));
  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_wsz, sizeof g_wsz));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  if (cublasLtMatmulAlgoGetHeuristic(g_lt, L.op, L.la, L.lb, L.ld, L.ld, pref, 16, res, &got) != CUBLAS_STATUS_SUCCESS) got = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float best = 1e30f;
  int bi = -1;
  const int32_t one = 1, zero = 0;
  const int mr = std::min(m, 256), rs = m / mr;
  const size_t nchk = (size_t)mr * n;
  ref_rows<<<(unsigned)((nchk + 255) / 256), 256, 0, st>>>((const int8_t*)A, (const int8_t*)W, g_ref, mr, rs, n, k);
  for (int i = 0; i < got; i++) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    CK(cudaMemsetAsync(D, 0xA5, (size_t)m * n * sizeof(int), st));
    if (cublasLtMatmul(g_lt, L.op, &one, W, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, g_ws, g_wsz, st) !=
        CUBLAS_STATUS_SUCCESS)
      continue;
    CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, st));
    count_neq_rows<<<(unsigned)((nchk + 255) / 256), 256, 0, st>>>((const int*)D, g_ref, mr, rs, n, g_cnt);
    unsigned long long bad = 0;
    CK(cudaMemcpyAsync(&bad, g_cnt, sizeof bad, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    if (bad) {
      L.rejected++;
      continue;
    }
    CK(cudaEventRecord(e0, st));
    for (int r = 0; r < 5; r++) cublasLtMatmul(g_lt, L.op, &one, W, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, g_ws, g_wsz, st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) { best = ms; bi = i; }
  }
  CK(cudaGetLastError());
  if (bi >= 0) {
    L.algo = res[bi].algo;
    L.ms = best / 5;
    size_t wr;
    cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_ID, &L.algo_id, sizeof L.algo_id, &wr);
  } else {
    fprintf(stderr, "cuBLASLt int8: no algorithm ran for m %d n %d k %d\n", m, n, k);
  }
  CL(cublasLtMatmulPreferenceDestroy(pref));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return L;
}

// ---------------------------------------------------------------- device state
struct Dev {
  cudaStream_t st;
  nvmlDevice_t nv;
  int sms;
  int8_t* W[LAYERS][4];  // qkv, o, gate_up, down: W is n x k row-major
  int8_t* Wlm;
  int8_t* X;             // activations, M x FF max
  int* D;                // outputs
  int8_t* kv;            // the KV pool
  size_t kv_bytes;
  int8_t *q, *P;
  int* S;
  int* O;
  int* sink;
};
static const int LN[4] = {QKV, HID, 2 * FF, HID}, LK[4] = {HID, HID, HID, FF};
static const char* LNAME[4] = {"qkv", "o", "gate_up", "down"};

static double nvml_mhz(const Dev& d) {
  unsigned int c = 0;
  return nvmlDeviceGetClockInfo(d.nv, NVML_CLOCK_SM, &c) == NVML_SUCCESS ? (double)c : -1;
}
static double nvml_w(const Dev& d) {
  unsigned int p = 0;
  return nvmlDeviceGetPowerUsage(d.nv, &p) == NVML_SUCCESS ? p / 1000.0 : -1;
}

struct Rep {
  double ms, mhz, watt;
};
static bool g_graph = true;  // DCAP_NOGRAPH=1 enqueues each rep directly
// the work as one CUDA graph, launched `reps` times after 2 warm-ups; each rep's time by events, its clock and power by
// NVML polled while it runs
static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// t0, t1: host times just before the first timed rep is launched and once the last has finished
static std::vector<Rep> time_work(Dev& d, const std::function<void()>& work, int reps, double* t0 = nullptr,
                                  double* t1 = nullptr) {
  cudaGraphExec_t ge = nullptr;
  cudaGraph_t g = nullptr;
  if (g_graph) {
    CK(cudaStreamBeginCapture(d.st, cudaStreamCaptureModeThreadLocal));
    work();
    CK(cudaStreamEndCapture(d.st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
  }
  auto launch = [&] {
    if (g_graph) CK(cudaGraphLaunch(ge, d.st));
    else work();
  };
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int i = 0; i < 2; i++) launch();
  CK(cudaStreamSynchronize(d.st));
  std::vector<Rep> out;
  if (t0) *t0 = now_ms();
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(e0, d.st));
    launch();
    CK(cudaEventRecord(e1, d.st));
    double cs = 0, ps = 0;
    int nc = 0, np = 0;
    while (cudaEventQuery(e1) == cudaErrorNotReady) {
      const double c = nvml_mhz(d), p = nvml_w(d);
      if (c > 0) { cs += c; nc++; }
      if (p > 0) { ps += p; np++; }
    }
    CK(cudaEventSynchronize(e1));
    if (nc == 0) { cs = nvml_mhz(d); nc = 1; }
    if (np == 0) { ps = nvml_w(d); np = 1; }
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    out.push_back({ms, cs / nc, ps / np});
  }
  if (t1) *t1 = now_ms();
  CK(cudaGetLastError());
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  if (g_graph) {
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
  }
  return out;
}
static void emit_reps(const std::vector<Rep>& v, double macs, double bytes, int sms) {
  std::vector<Rep> s = v;
  std::sort(s.begin(), s.end(), [](const Rep& a, const Rep& b) { return a.ms < b.ms; });
  const Rep& m = s[s.size() / 2];
  const double peak = 1024.0 * sms * m.mhz * 1e6;  // int8 MACs/s at the read clock
  printf(", \"ms\": %.4f, \"ms_min\": %.4f, \"ms_max\": %.4f, \"mhz\": %.1f, \"watt\": %.1f, \"macs\": %.6e, \"bytes\": %.6e, "
         "\"mac_per_s\": %.6e, \"share_of_peak\": %.5f, \"share_of_peak_2430\": %.5f, \"gb_per_s\": %.2f, \"reps\": [",
         m.ms, s.front().ms, s.back().ms, m.mhz, m.watt, macs, bytes, macs / (m.ms * 1e-3), macs / (m.ms * 1e-3) / peak,
         macs / (m.ms * 1e-3) / (1024.0 * sms * 2430e6), bytes / (m.ms * 1e-3) / 1e9);
  for (size_t i = 0; i < v.size(); i++) printf("%s[%.4f, %.0f, %.0f]", i ? ", " : "", v[i].ms, v[i].mhz, v[i].watt);
  printf("]}\n");
  fflush(stdout);
}

// the int8 KV of one layer: K then V, each B x HKV x L x HD; layers laid out one after another, wrapping in the pool
static int8_t* kv_layer(const Dev& d, int B, int L, int layer, bool v, bool* wrapped) {
  const size_t per = (size_t)B * HKV * L * HD, lay = 2 * per;
  const size_t nfit = d.kv_bytes / lay;
  if (nfit < (size_t)LAYERS) *wrapped = true;
  const size_t slot = nfit ? (size_t)layer % nfit : 0;
  return d.kv + slot * lay + (v ? per : 0);
}

static std::vector<Lt> lin_plans(Dev& d, int M) {
  std::vector<Lt> p(5);
  for (int j = 0; j < 4; j++) p[j] = lt_make(M, LN[j], LK[j], d.X, d.W[0][j], d.D, d.st);
  if (M <= 256) p[4] = lt_make(M, VOCAB, HID, d.X, d.Wlm, d.D, d.st);
  return p;
}

static void enqueue_att(Dev& d, int B, int L, int layer, bool* wrapped) {
  const int8_t* K = kv_layer(d, B, L, layer, false, wrapped);
  const int8_t* V = kv_layer(d, B, L, layer, true, wrapped);
  qk_dec<<<dim3((L + QK_W * QK_KW - 1) / (QK_W * QK_KW), HKV, B), 256, 0, d.st>>>(K, d.q, d.S, L);
  CK(cudaMemsetAsync(d.O, 0, (size_t)B * HQ * HD * sizeof(int), d.st));
  pv_dec<<<dim3((L / 4 + PV_W * PV_GW - 1) / (PV_W * PV_GW), HKV, B), 256, 0, d.st>>>(V, d.P, d.O, L);
}

// kind: 0 linears only, 1 attention only, 2 both interleaved
static void enqueue_step(Dev& d, const std::vector<Lt>& p, int kind, int M, int L, bool* wrapped) {
  for (int l = 0; l < LAYERS; l++) {
    if (kind != 1 && p[0].algo_id != -2) lt_run(p[0], d.X, d.W[l][0], d.D, d.st);
    if (kind != 0) enqueue_att(d, M, L, l, wrapped);
    for (int j = 1; j < 4; j++)
      if (kind != 1 && p[j].algo_id != -2) lt_run(p[j], d.X, d.W[l][j], d.D, d.st);
  }
  if (kind != 1 && M <= 256 && p[4].algo_id != -2) lt_run(p[4], d.X, d.Wlm, d.D, d.st);
}
static void run_step(Dev& d, int kind, int M, int L, int reps) {
  std::vector<Lt> p;
  if (kind != 1) p = lin_plans(d, M);
  bool wrapped = false;
  const std::vector<Rep> v = time_work(d, [&] { enqueue_step(d, p, kind, M, L, &wrapped); }, reps);
  double lin_macs = 0, lin_bytes = 0;
  bool missing = false;
  if (kind != 1) {
    for (int j = 0; j < 4; j++) {
      lin_macs += (double)LAYERS * M * LN[j] * LK[j];
      lin_bytes += (double)LAYERS * LN[j] * LK[j];
      missing |= p[j].algo_id == -2;
    }
    if (M <= 256) {
      lin_macs += (double)M * VOCAB * HID;
      lin_bytes += (double)VOCAB * HID;
      missing |= p[4].algo_id == -2;
    }
  }
  const double att_macs = kind != 0 ? 2.0 * LAYERS * M * HQ * HD * (double)L : 0;
  const double att_bytes = kind != 0 ? (double)M * L * KV_TOKEN_BYTES : 0;
  static const char* KN[3] = {"lin", "att", "step"};
  printf("{\"mode\": \"%s\", \"M\": %d, \"L\": %d, \"graph\": %s, \"lin_macs\": %.6e, \"att_macs\": %.6e, \"kv_wrapped\": %s, "
         "\"missing_algo\": %s, \"algos\": [",
         KN[kind], M, kind == 0 ? 0 : L, g_graph ? "true" : "false", lin_macs, att_macs, wrapped ? "true" : "false",
         missing ? "true" : "false");
  for (size_t j = 0; j < p.size(); j++)
    printf("%s{\"name\": \"%s\", \"id\": %d, \"rejected\": %d, \"ms_alone\": %.4f}", j ? ", " : "", j < 4 ? LNAME[j] : "lm_head",
           p[j].algo_id, p[j].rejected, p[j].ms);
  printf("]");
  emit_reps(v, lin_macs + att_macs, lin_bytes + att_bytes, d.sms);
}

static void run_bw(Dev& d, size_t bytes, int reps) {
  const size_t n16 = bytes / 16;
  const std::vector<Rep> v = time_work(d, [&] {
    bw_read<<<d.sms * 8, 512, 0, d.st>>>(reinterpret_cast<const int4*>(d.kv), n16, d.sink);
  }, reps);
  printf("{\"mode\": \"bw\", \"bytes_read\": %zu, \"graph\": %s", bytes, g_graph ? "true" : "false");
  emit_reps(v, 0, (double)bytes, d.sms);
}

// ---------------------------------------------------------------- fill: the step beside chained int8 GEMMs on a green split
// Green contexts through the driver API, by dlopen (as fill_compute's gc_split), so nothing links libcuda.
template <class F> static F drv(const char* name) {
  static void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  return h ? reinterpret_cast<F>(dlsym(h, name)) : nullptr;
}
static int green_stream(CUdevResource* r, CUdevice dev, CUstream* s) {
  auto gen = drv<decltype(&cuDevResourceGenerateDesc)>("cuDevResourceGenerateDesc");
  auto mk = drv<decltype(&cuGreenCtxCreate)>("cuGreenCtxCreate");
  auto st = drv<decltype(&cuGreenCtxStreamCreate)>("cuGreenCtxStreamCreate");
  if (!gen || !mk || !st) return 1000;
  CUdevResourceDesc desc;
  CUgreenCtx g;
  int e;
  if ((e = gen(&desc, r, 1))) return 4000 + e;
  if ((e = mk(&g, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM))) return 5000 + e;
  if ((e = st(s, g, CU_STREAM_NON_BLOCKING, 0))) return 6000 + e;
  return 0;
}
// the SMs split into the step's (at least n_step, as the driver rounds) and the rest, a stream on each; 1000 x the failing
// stage + its CUresult on failure
static int gc_split(int n_step, cudaStream_t* s_step, cudaStream_t* s_fill, int got[2]) {
  auto dget = drv<decltype(&cuDeviceGet)>("cuDeviceGet");
  auto res = drv<decltype(&cuDeviceGetDevResource)>("cuDeviceGetDevResource");
  auto split = drv<decltype(&cuDevSmResourceSplitByCount)>("cuDevSmResourceSplitByCount");
  if (!dget || !res || !split) return 1000;
  CUdevice dev;
  CUdevResource all{}, part{}, rest{};
  unsigned nb = 1;
  int e;
  if ((e = dget(&dev, 0))) return 2000 + e;
  if ((e = res(dev, &all, CU_DEV_RESOURCE_TYPE_SM))) return 2000 + e;
  if ((e = split(&part, &nb, &all, &rest, 0, (unsigned)n_step))) return 3000 + e;
  got[0] = (int)part.sm.smCount;
  got[1] = (int)rest.sm.smCount;
  CUstream a = nullptr, b = nullptr;
  if ((e = green_stream(&part, dev, &a))) return e;
  if (rest.sm.smCount && (e = green_stream(&rest, dev, &b))) return e;
  *s_step = (cudaStream_t)a;
  *s_fill = (cudaStream_t)b;
  return 0;
}
__global__ void smid_probe(unsigned* bits) {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  if (threadIdx.x == 0) atomicOr(&bits[(s / 32) & 7], 1u << (s % 32));
  const long long t = clock64();
  while (clock64() - t < 20000) {
  }
}
// the distinct SMs that 4 x sms blocks launched on `st` (in a graph when `graph`) ran on
static int sm_spread(const Dev& d, cudaStream_t st, bool graph) {
  unsigned* bits;
  CK(cudaMalloc(&bits, 32));
  CK(cudaMemsetAsync(bits, 0, 32, st));
  if (graph) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    smid_probe<<<4 * d.sms, 32, 0, st>>>(bits);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
  } else {
    smid_probe<<<4 * d.sms, 32, 0, st>>>(bits);
  }
  unsigned h[8];
  CK(cudaMemcpyAsync(h, bits, 32, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  CK(cudaFree(bits));
  int n = 0;
  for (unsigned w : h) n += __builtin_popcount(w);
  return n;
}

static cublasLtHandle_t g_lt2;  // the filler's handle and workspace: it runs beside the step's
static void* g_ws2;
// T2's chain: step s computes C_s = X_s W_s^T (s8 x s8 -> s32, m x n, n = k/4) into the buffer step s + 1 reads as X
struct Chain {
  Lt L;
  int8_t* X[2];
  int8_t* W[2];
  int m, n, k;
};
static void chain_run(const Chain& c, int s, cudaStream_t st) {
  const int32_t one = 1, zero = 0;
  void* out = c.X[(s + 1) & 1];
  CL(cublasLtMatmul(g_lt2, c.L.op, &one, c.W[s & 1], c.L.la, c.X[s & 1], c.L.lb, &zero, out, c.L.ld, out, c.L.ld,
                    &c.L.algo, g_ws2, g_wsz, st));
}
// A host thread keeps `depth` chunks of `per` chain steps queued on the filler's stream and notes when each chunk is seen
// done. Stopping launches nothing more and waits for what is in flight.
struct Pump {
  std::thread th;
  std::atomic<bool> on{false};
  std::atomic<long long> done{0};
  std::vector<double> t;  // host time each finished step was seen, read only after the join
};
static void pump_start(Pump& p, const Chain& c, cudaStream_t fs, int per, int depth) {
  p.on = true;
  p.done = 0;
  p.t.clear();
  p.th = std::thread([&p, &c, fs, per, depth] {
    std::vector<cudaEvent_t> ev(depth);
    for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync));
    int s = 0;
    for (long long k = 0; p.on.load(); k++) {
      const int slot = (int)(k % depth);
      if (k >= depth) {
        CK(cudaEventSynchronize(ev[slot]));
        const double now = now_ms();
        for (int j = 0; j < per; j++) p.t.push_back(now);
        p.done += per;
      }
      for (int j = 0; j < per; j++) chain_run(c, s++, fs);
      CK(cudaEventRecord(ev[slot], fs));
    }
    CK(cudaStreamSynchronize(fs));
    for (auto& e : ev) CK(cudaEventDestroy(e));
  });
}
static void pump_stop(Pump& p) {
  p.on = false;
  p.th.join();
}
static long long steps_in(const Pump& p, double t0, double t1) {
  long long n = 0;
  for (double x : p.t) n += x > t0 && x <= t1;
  return n;
}
static Rep median(const std::vector<Rep>& v) {
  std::vector<Rep> s = v;
  std::sort(s.begin(), s.end(), [](const Rep& a, const Rep& b) { return a.ms < b.ms; });
  return s[s.size() / 2];
}
static void print_arm(const char* name, const std::vector<Rep>& v, bool comma) {
  const Rep m = median(v);
  printf("\"%s\": {\"ms\": %.4f, \"mhz\": %.1f, \"watt\": %.1f, \"reps_ms\": [", name, m.ms, m.mhz, m.watt);
  for (size_t i = 0; i < v.size(); i++) printf("%s%.4f", i ? ", " : "", v[i].ms);
  printf("]}%s", comma ? ", " : "");
}

// arms: the step on the whole GPU, on its partition alone, on its partition beside the filler, and on the whole GPU again;
// before them the filler alone on its partition
static int run_fill(Dev& d, int B, int L, int fill_sms, bool big, int reps) {
  Chain c;
  c.m = c.k = big ? 8192 : 4096;
  c.n = c.k / 4;
  for (int i = 0; i < 2; i++) {
    CK(cudaMalloc(&c.X[i], (size_t)c.m * c.k));
    CK(cudaMalloc(&c.W[i], (size_t)c.n * c.k));
    fill_s8<<<1024, 256, 0, d.st>>>(c.X[i], (size_t)c.m * c.k, 0x2000 + i);
    fill_s8<<<1024, 256, 0, d.st>>>(c.W[i], (size_t)c.n * c.k, 0x3000 + i);
  }
  CK(cudaStreamSynchronize(d.st));
  c.L = lt_make(c.m, c.n, c.k, c.X[0], c.W[0], c.X[1], d.st);
  if (c.L.algo_id == -2) return 6;
  const double fmacs = (double)c.m * c.n * c.k;
  const double fbytes = (double)c.m * c.k + (double)c.n * c.k + (double)c.m * c.n * 4;  // X and W read, C written
  const std::vector<Lt> p = lin_plans(d, B);
  bool wrapped = false;
  const cudaStream_t full = d.st;
  cudaStream_t ss = nullptr, fs = nullptr;
  int got[2] = {0, 0};
  const int e = gc_split(d.sms - fill_sms, &ss, &fs, got);
  if (e || !fs) {
    fprintf(stderr, "gc_split(%d): %d (filler SMs %d)\n", d.sms - fill_sms, e, got[1]);
    return 7;
  }
  const int sp_full = sm_spread(d, full, false), sp_step = sm_spread(d, ss, false), sp_graph = sm_spread(d, ss, true),
            sp_fill = sm_spread(d, fs, false);
  const int per = big ? 1 : 8;
  Pump pump;
  pump_start(pump, c, fs, per, 3);
  const long long n_alone = big ? 160 : 1280;
  double cs = 0, ps = 0;
  long long np = 0;
  while (pump.done.load() < n_alone) {
    cs += nvml_mhz(d);
    ps += nvml_w(d);
    np++;
  }
  pump_stop(pump);
  const size_t ia = 2 * per - 1, ib = pump.t.size() - 1;
  const double alone = (double)(ib - ia) * fmacs / ((pump.t[ib] - pump.t[ia]) * 1e-3);
  const double alone_mhz = np ? cs / np : nvml_mhz(d);
  auto work = [&] { enqueue_step(d, p, 2, B, L, &wrapped); };
  d.st = full;
  const std::vector<Rep> a_full = time_work(d, work, reps);
  d.st = ss;
  const std::vector<Rep> a_split = time_work(d, work, reps);
  pump_start(pump, c, fs, per, 3);
  while (pump.done.load() < 2 * per) {
  }
  double t0 = 0, t1 = 0;
  const std::vector<Rep> a_fill = time_work(d, work, reps, &t0, &t1);
  pump_stop(pump);
  const long long nwin = steps_in(pump, t0, t1);
  const double rate = (double)nwin * fmacs / ((t1 - t0) * 1e-3);
  d.st = full;
  const std::vector<Rep> a_full2 = time_work(d, work, reps);
  double step_macs = 0;
  for (int j = 0; j < 4; j++) step_macs += (double)LAYERS * B * LN[j] * LK[j];
  step_macs += (double)B * VOCAB * HID + 2.0 * LAYERS * B * HQ * HD * (double)L;
  const Rep mf = median(a_fill), m0 = median(a_full), m2 = median(a_full2);
  const double peak = 1024.0 * d.sms * mf.mhz * 1e6;
  printf("{\"mode\": \"fill\", \"B\": %d, \"L\": %d, \"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"algo\": %d, "
         "\"rejected\": %d, \"fill_sms_req\": %d, \"step_sms\": %d, \"fill_sms\": %d, \"spread\": {\"full\": %d, \"step\": %d, "
         "\"step_graph\": %d, \"fill\": %d}, \"kv_wrapped\": %s, \"step_macs\": %.6e, \"fill_macs_a_step\": %.6e, "
         "\"fill_bytes_a_step\": %.6e, \"alone\": {\"mac_per_s\": %.6e, \"share\": %.5f, \"mhz\": %.1f, \"watt\": %.1f, "
         "\"steps\": %zu}, \"window\": {\"ms\": %.3f, \"fill_steps\": %lld, \"fill_mac_per_s\": %.6e, \"fill_share\": %.5f, "
         "\"fill_gb_per_s\": %.2f}, \"slowdown\": %.4f, \"step_share_full\": %.5f, \"step_share_fill\": %.5f, \"arms\": {",
         B, L, big ? "big" : "l2", c.m, c.n, c.k, c.L.algo_id, c.L.rejected, fill_sms, got[0], got[1], sp_full, sp_step,
         sp_graph, sp_fill, wrapped ? "true" : "false", step_macs, fmacs, fbytes, alone,
         alone / (1024.0 * d.sms * alone_mhz * 1e6), alone_mhz, np ? ps / np : nvml_w(d), ib - ia, t1 - t0, nwin, rate,
         rate / peak, rate / fmacs * fbytes / 1e9, mf.ms / ((m0.ms + m2.ms) / 2),
         step_macs / (m0.ms * 1e-3) / (1024.0 * d.sms * m0.mhz * 1e6), step_macs / (mf.ms * 1e-3) / peak);
  print_arm("full", a_full, true);
  print_arm("split", a_split, true);
  print_arm("split_fill", a_fill, true);
  print_arm("full2", a_full2, false);
  printf("}}\n");
  fflush(stdout);
  return 0;
}

// ---------------------------------------------------------------- dfill: chained int8 MMAs that touch no DRAM (FILL)
// A tile is NC chains, each a 16 x 32 s8 state. The tile's seed comes from the round's coins (c0, c1) and the tile's id,
// and expands on chip into B_0, B_1 (32 x 8 s8, k x n), C (16 x 8 s32, |C| < 2^23) and each chain's A_0. Step s of a chain
// is D = A_s B_{s mod 2} + C (mma.sync m16n8k32 s8 x s8 -> s32; |A B| <= 2^19, so no s32 word overflows), and A_{s+1} is
// D's bytes under the fixed map (g < 8, t < 4, q < 4; byte q of a word is bits 8q..8q+7):
//   A'[g][4t+q] = byte q of D[g][2t]           A'[g+8][4t+q] = byte q of D[g+8][2t]
//   A'[g][16+4t+q] = byte q of D[g+8][2t+1]    A'[g+8][16+4t+q] = byte q of D[g][2t+1]
// Each A' word is an accumulator register the thread already holds, so a step is one MMA and nothing else, and step s + 1
// can't start before step s ends. After S steps the tile's NC final states fold into an 8-byte receipt, the XOR over
// (chain c, row r, word w) of drec(seed, c, r, w, word w of row r), which is all the tile writes. dfill_ref is the same
// tile on the host, written from the matrices above rather than from the fragment layout.
__host__ __device__ inline uint64_t smix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
__host__ __device__ inline uint64_t dseed(uint64_t c0, uint64_t c1, uint64_t tile) { return smix(smix(c0 ^ smix(tile)) ^ c1); }
// word (i, j) of the generated matrix `tag`: 1 + b is B_b (i = column n, j = word of k), 3 is C, 16 + c chain c's A_0
// (i = row, j = word of k); four s8 elements, element 4j + q in byte q
__host__ __device__ inline uint32_t dword(uint64_t seed, uint32_t tag, uint32_t i, uint32_t j) {
  return (uint32_t)smix(seed ^ ((uint64_t)tag << 48) ^ ((uint64_t)i << 24) ^ j);
}
__host__ __device__ inline int32_t dcval(uint64_t seed, int r, int n) { return (int32_t)(dword(seed, 3, r, n) & 0xffffffu) - 0x800000; }
__host__ __device__ inline uint64_t drec(uint64_t seed, int c, int r, int w, uint32_t word) {
  return smix(seed ^ ((uint64_t)(c * 16 + r) << 40) ^ ((uint64_t)w << 32) ^ word);
}

__device__ __forceinline__ void dstep(uint32_t (&a)[4], const uint32_t (&b)[2], const uint32_t (&c)[4]) {
  uint32_t d0, d1, d2, d3;
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
  a[0] = d0;  // A'[g][4t..]       = D[g][2t]
  a[1] = d2;  // A'[g+8][4t..]     = D[g+8][2t]
  a[2] = d3;  // A'[g][16+4t..]    = D[g+8][2t+1]
  a[3] = d1;  // A'[g+8][16+4t..]  = D[g][2t+1]
}
// Warp `wid` runs tiles tile0 + wid·tiles + i, i < tiles, S steps each (S a multiple of 8), and writes tile i's receipt to
// rec[wid·tiles + i]. Lane (g, t) holds A rows g and g + 8, words t and 4 + t; B column g, words t and 4 + t; C rows g
// and g + 8, columns 2t and 2t + 1 (the PTX fragment layout of m16n8k32 .s8).
template <int NC>
__global__ void __launch_bounds__(512) dfill_k(uint64_t c0, uint64_t c1, unsigned long long tile0, int tiles, int steps,
                                              unsigned long long* __restrict__ rec) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const unsigned long long wid = (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  for (int i = 0; i < tiles; i++) {
    const unsigned long long idx = wid * tiles + i;
    const uint64_t seed = dseed(c0, c1, tile0 + idx);
    uint32_t b[2][2], c[4], a[NC][4];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      b[j][0] = dword(seed, 1 + j, g, t);
      b[j][1] = dword(seed, 1 + j, g, 4 + t);
    }
    c[0] = (uint32_t)dcval(seed, g, 2 * t);
    c[1] = (uint32_t)dcval(seed, g, 2 * t + 1);
    c[2] = (uint32_t)dcval(seed, g + 8, 2 * t);
    c[3] = (uint32_t)dcval(seed, g + 8, 2 * t + 1);
#pragma unroll
    for (int ch = 0; ch < NC; ch++) {
      a[ch][0] = dword(seed, 16 + ch, g, t);
      a[ch][1] = dword(seed, 16 + ch, g + 8, t);
      a[ch][2] = dword(seed, 16 + ch, g, 4 + t);
      a[ch][3] = dword(seed, 16 + ch, g + 8, 4 + t);
    }
#pragma unroll 1
    for (int s = 0; s < steps; s += 8) {
#pragma unroll
      for (int u = 0; u < 8; u++)
#pragma unroll
        for (int ch = 0; ch < NC; ch++) dstep(a[ch], b[u & 1], c);
    }
    uint64_t h = 0;
#pragma unroll
    for (int ch = 0; ch < NC; ch++)
      h ^= drec(seed, ch, g, t, a[ch][0]) ^ drec(seed, ch, g + 8, t, a[ch][1]) ^ drec(seed, ch, g, 4 + t, a[ch][2]) ^
           drec(seed, ch, g + 8, 4 + t, a[ch][3]);
#pragma unroll
    for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
    if (lane == 0) rec[idx] = h;
  }
}

static uint64_t dfill_ref(uint64_t c0, uint64_t c1, uint64_t tile, int nc, int steps) {
  const uint64_t seed = dseed(c0, c1, tile);
  int8_t Bt[2][8][32];  // Bt[b][n][k] = B_b[k][n]
  int32_t C[16][8];
  for (int b = 0; b < 2; b++)
    for (int n = 0; n < 8; n++)
      for (int w = 0; w < 8; w++) {
        const uint32_t x = dword(seed, 1 + b, n, w);
        for (int q = 0; q < 4; q++) Bt[b][n][4 * w + q] = (int8_t)(x >> (8 * q));
      }
  for (int r = 0; r < 16; r++)
    for (int n = 0; n < 8; n++) C[r][n] = dcval(seed, r, n);
  uint64_t h = 0;
  for (int ch = 0; ch < nc; ch++) {
    int8_t A[16][32];
    for (int r = 0; r < 16; r++)
      for (int w = 0; w < 8; w++) {
        const uint32_t x = dword(seed, 16 + ch, r, w);
        for (int q = 0; q < 4; q++) A[r][4 * w + q] = (int8_t)(x >> (8 * q));
      }
    for (int s = 0; s < steps; s++) {
      int32_t D[16][8];
      for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++) {
          int32_t acc = C[r][n];
          for (int k = 0; k < 32; k++) acc += (int32_t)A[r][k] * (int32_t)Bt[s & 1][n][k];
          D[r][n] = acc;
        }
      auto byte = [](int32_t v, int q) { return (int8_t)((uint32_t)v >> (8 * q)); };
      for (int g = 0; g < 8; g++)
        for (int t = 0; t < 4; t++)
          for (int q = 0; q < 4; q++) {
            A[g][4 * t + q] = byte(D[g][2 * t], q);
            A[g + 8][4 * t + q] = byte(D[g + 8][2 * t], q);
            A[g][16 + 4 * t + q] = byte(D[g + 8][2 * t + 1], q);
            A[g + 8][16 + 4 * t + q] = byte(D[g][2 * t + 1], q);
          }
    }
    for (int r = 0; r < 16; r++)
      for (int w = 0; w < 8; w++) {
        uint32_t x = 0;
        for (int q = 0; q < 4; q++) x |= (uint32_t)(uint8_t)A[r][4 * w + q] << (8 * q);
        h ^= drec(seed, ch, r, w, x);
      }
  }
  return h;
}

// one chunk: `grid` blocks of W warps, each warp `tiles` tiles of NC chains x S steps; chunk k covers tiles from k·per on
struct DFill {
  int W = 4, NC = 2, S = 2048, tiles = 1, grid = 0;
  uint64_t c0 = 0, c1 = 0;
  unsigned long long* rec = nullptr;
  long long per() const { return (long long)grid * W * tiles; }
  double macs() const { return (double)per() * NC * S * 4096.0; }
};
static void dfill_launch(const DFill& f, long long chunk, cudaStream_t st) {
  const unsigned long long t0 = (unsigned long long)chunk * f.per();
  const dim3 gr(f.grid), bl(32 * f.W);
  switch (f.NC) {
    case 1: dfill_k<1><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 2: dfill_k<2><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 4: dfill_k<4><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    default: fprintf(stderr, "NC must be 1, 2 or 4\n"); exit(1);
  }
}
static void dfill_prep() {
  CK(cudaFuncSetAttribute(dfill_k<1>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<2>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<4>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
}
// tiles a warp so that a chunk is about 2^17 MMAs a scheduler, ~1 ms at the tensor peak
static DFill dfill_make(int grid, int W, int NC, int S, uint64_t c0, uint64_t c1) {
  DFill f;
  f.grid = grid; f.W = W; f.NC = NC; f.S = S; f.c0 = c0; f.c1 = c1;
  f.tiles = std::max(1, (int)((4LL << 17) / ((long long)W * NC * S)));
  CK(cudaMalloc(&f.rec, f.per() * sizeof(unsigned long long)));
  return f;
}
// the host's receipts for `n` tiles of chunk `chunk` spread over it, against the device's; returns the mismatches
static int dfill_check(const DFill& f, long long chunk, int n) {
  std::vector<unsigned long long> h(f.per());
  CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; i++) {
    const long long idx = (long long)i * f.per() / n + (i * 7919) % std::max(1LL, f.per() / n);
    bad += h[idx] != dfill_ref(f.c0, f.c1, (unsigned long long)chunk * f.per() + idx, f.NC, f.S);
  }
  return bad;
}

// A host thread keeps `depth` chunks queued on `fs` and notes when each is seen done.
struct DPump {
  std::thread th;
  std::atomic<bool> on{false};
  std::atomic<long long> done{0}, launched{0};
  std::vector<double> t;
};
static void dpump_start(DPump& p, const DFill& f, cudaStream_t fs, int depth) {
  p.on = true;
  p.done = 0;
  p.t.clear();
  p.th = std::thread([&p, &f, fs, depth] {
    std::vector<cudaEvent_t> ev(depth);
    for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync));
    long long k = 0;
    for (; p.on.load(); k++) {
      const int slot = (int)(k % depth);
      if (k >= depth) {
        CK(cudaEventSynchronize(ev[slot]));
        p.t.push_back(now_ms());
        p.done++;
      }
      dfill_launch(f, k, fs);
      CK(cudaEventRecord(ev[slot], fs));
    }
    p.launched = k;
    CK(cudaStreamSynchronize(fs));
    for (auto& e : ev) CK(cudaEventDestroy(e));
  });
}
static void dpump_stop(DPump& p) {
  p.on = false;
  p.th.join();
}
// the filler alone on `fs` for `n` chunks: MACs/s from the first to the last chunk seen done after the first two
static double dfill_alone(Dev& d, const DFill& f, cudaStream_t fs, int n, double* mhz, double* watt,
                          long long* last = nullptr) {
  DPump p;
  dpump_start(p, f, fs, 3);
  double cs = 0, ps = 0;
  long long np = 0;
  while (p.done.load() < n) {
    cs += nvml_mhz(d);
    ps += nvml_w(d);
    np++;
  }
  dpump_stop(p);
  if (last) *last = p.launched.load() - 1;
  *mhz = np ? cs / np : nvml_mhz(d);
  *watt = np ? ps / np : nvml_w(d);
  const size_t ia = 1, ib = p.t.size() - 1;
  return (double)(ib - ia) * f.macs() / ((p.t[ib] - p.t[ia]) * 1e-3);
}

static uint64_t g_c0 = 0x0123456789abcdefull, g_c1 = 0xfedcba9876543210ull;  // the round's coins (DFILL_COINS)

// the receipts of a few tiles for each NC at S = 8 and the timed S, every receipt checked; then a control (S - 8 on the
// host) that must differ everywhere
static int dgate(Dev& d) {
  int fails = 0;
  dfill_prep();
  for (int nc : {1, 2, 4})
    for (int S : {8, 2048}) {
      DFill f;
      f.grid = 3; f.W = 4; f.NC = nc; f.S = S; f.tiles = 2; f.c0 = g_c0; f.c1 = g_c1;
      CK(cudaMalloc(&f.rec, f.per() * sizeof(unsigned long long)));
      CK(cudaMemset(f.rec, 0xA5, f.per() * sizeof(unsigned long long)));
      dfill_launch(f, 5, d.st);
      CK(cudaStreamSynchronize(d.st));
      CK(cudaGetLastError());
      std::vector<unsigned long long> h(f.per());
      CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
      long bad = 0, ctl_equal = 0;
      for (long long i = 0; i < f.per(); i++) {
        const unsigned long long tile = 5ull * f.per() + i;
        bad += h[i] != dfill_ref(f.c0, f.c1, tile, nc, S);
        ctl_equal += h[i] == dfill_ref(f.c0, f.c1, tile, nc, S - 8);
      }
      printf("{\"mode\": \"dgate\", \"NC\": %d, \"S\": %d, \"tiles\": %lld, \"mismatches\": %ld, \"control_equal\": %ld, "
             "\"ok\": %s}\n",
             nc, S, f.per(), bad, ctl_equal, bad || ctl_equal ? "false" : "true");
      fails += bad != 0 || ctl_equal != 0;
      CK(cudaFree(f.rec));
    }
  fflush(stdout);
  return fails;
}

// the filler alone on every SM, one block of W warps an SM
static void run_dalone(Dev& d, int W, int NC, int S) {
  dfill_prep();
  DFill f = dfill_make(d.sms, W, NC, S, g_c0, g_c1);
  double mhz, watt;
  long long last = 0;
  const double r = dfill_alone(d, f, d.st, 40, &mhz, &watt, &last);
  const int bad = dfill_check(f, last, 8);
  printf("{\"mode\": \"dalone\", \"W\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %d, \"grid\": %d, \"chunk_macs\": %.6e, "
         "\"mac_per_s\": %.6e, \"share\": %.5f, \"mhz\": %.1f, \"watt\": %.1f, \"rec_checked\": 8, \"rec_bad\": %d}\n",
         W, NC, S, f.tiles, f.grid, f.macs(), r, r / (1024.0 * d.sms * mhz * 1e6), mhz, watt, bad);
  fflush(stdout);
  CK(cudaFree(f.rec));
}
// three chunks on every SM, for ncu
static void run_done(Dev& d, int W, int NC, int S) {
  dfill_prep();
  DFill f = dfill_make(d.sms, W, NC, S, g_c0, g_c1);
  for (int k = 0; k < 3; k++) dfill_launch(f, k, d.st);
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  printf("{\"mode\": \"done\", \"W\": %d, \"NC\": %d, \"S\": %d, \"chunk_macs\": %.6e, \"rec_bad\": %d}\n", W, NC, S, f.macs(),
         dfill_check(f, 2, 4));
  CK(cudaFree(f.rec));
}

// The step beside the filler. place 0 (split): a green split as run_fill, the filler on fill_sms SMs, one block of W warps
// an SM; arms full, split, split + filler, full again. place 1 (co): the filler on every SM at the lowest stream priority,
// one block of W warps an SM, the step on the whole GPU at the highest; arms full, co + filler, full again. Before them the
// filler alone; after them 8 of the window's last chunk's receipts checked on the host.
static int run_dfill(Dev& d, int B, int L, int place, int fill_sms, int W, int NC, int S, int reps) {
  dfill_prep();
  const std::vector<Lt> p = lin_plans(d, B);
  bool wrapped = false;
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t full = d.st, ss = nullptr, fs = nullptr;
  int got[2] = {d.sms, 0};
  if (place == 0) {
    const int e = gc_split(d.sms - fill_sms, &ss, &fs, got);
    if (e || !fs) {
      fprintf(stderr, "gc_split(%d): %d (filler SMs %d)\n", d.sms - fill_sms, e, got[1]);
      return 7;
    }
  } else {
    CK(cudaStreamCreateWithPriority(&full, cudaStreamDefault, hi));
    CK(cudaStreamCreateWithPriority(&fs, cudaStreamNonBlocking, lo));
    ss = full;
    got[1] = d.sms;
  }
  DFill f = dfill_make(got[1], W, NC, S, g_c0, g_c1);
  const int sp_step = sm_spread(d, ss, true), sp_fill = sm_spread(d, fs, false);
  double amhz, awatt;
  const double alone = dfill_alone(d, f, fs, 30, &amhz, &awatt);
  auto work = [&] { enqueue_step(d, p, 2, B, L, &wrapped); };
  d.st = full;
  const std::vector<Rep> a_full = time_work(d, work, reps);
  std::vector<Rep> a_split;
  if (place == 0) {
    d.st = ss;
    a_split = time_work(d, work, reps);
  }
  DPump pump;
  dpump_start(pump, f, fs, 3);
  while (pump.done.load() < 2) {
  }
  double t0 = 0, t1 = 0;
  d.st = ss;
  const std::vector<Rep> a_fill = time_work(d, work, reps, &t0, &t1);
  dpump_stop(pump);
  long long nwin = 0;
  for (double x : pump.t) nwin += x > t0 && x <= t1;
  const double rate = (double)nwin * f.macs() / ((t1 - t0) * 1e-3);
  const int bad = dfill_check(f, pump.launched.load() - 1, 8);
  d.st = full;
  const std::vector<Rep> a_full2 = time_work(d, work, reps);
  double step_macs = 0;
  for (int j = 0; j < 4; j++) step_macs += (double)LAYERS * B * LN[j] * LK[j];
  step_macs += (double)B * VOCAB * HID + 2.0 * LAYERS * B * HQ * HD * (double)L;
  const Rep mf = median(a_fill), m0 = median(a_full), m2 = median(a_full2);
  const double peak = 1024.0 * d.sms * mf.mhz * 1e6;
  char shape[64];
  snprintf(shape, sizeof shape, "dfree-%s-w%d-nc%d", place ? "co" : "split", W, NC);
  printf("{\"mode\": \"dfill\", \"place\": \"%s\", \"shape\": \"%s\", \"B\": %d, \"L\": %d, \"W\": %d, \"NC\": %d, \"S\": %d, "
         "\"tiles\": %d, \"grid\": %d, \"coins\": [\"%016llx\", \"%016llx\"], \"prio\": [%d, %d], \"fill_sms_req\": %d, "
         "\"step_sms\": %d, \"fill_sms\": %d, \"spread\": {\"step_graph\": %d, \"fill\": %d}, \"kv_wrapped\": %s, "
         "\"step_macs\": %.6e, \"chunk_macs\": %.6e, \"alone\": {\"mac_per_s\": %.6e, \"share\": %.5f, \"mhz\": %.1f, "
         "\"watt\": %.1f}, \"window\": {\"ms\": %.3f, \"fill_chunks\": %lld, \"fill_mac_per_s\": %.6e, \"fill_share\": %.5f, "
         "\"fill_gb_per_s\": %.6f}, \"rec_checked\": 8, \"rec_bad\": %d, \"slowdown\": %.4f, \"step_share_full\": %.5f, "
         "\"step_share_fill\": %.5f, \"arms\": {",
         place ? "co" : "split", shape, B, L, W, NC, S, f.tiles, f.grid, (unsigned long long)g_c0, (unsigned long long)g_c1,
         lo, hi, place ? d.sms : fill_sms, place ? d.sms : got[0], got[1], sp_step, sp_fill, wrapped ? "true" : "false",
         step_macs, f.macs(), alone, alone / (1024.0 * d.sms * amhz * 1e6), amhz, awatt, t1 - t0, nwin, rate, rate / peak,
         (double)nwin * f.per() * 8 / ((t1 - t0) * 1e-3) / 1e9, bad, mf.ms / ((m0.ms + m2.ms) / 2),
         step_macs / (m0.ms * 1e-3) / (1024.0 * d.sms * m0.mhz * 1e6), step_macs / (mf.ms * 1e-3) / peak);
  print_arm("full", a_full, true);
  if (place == 0) print_arm("split", a_split, true);
  print_arm(place ? "co_fill" : "split_fill", a_fill, true);
  print_arm("full2", a_full2, false);
  printf("}}\n");
  fflush(stdout);
  CK(cudaFree(f.rec));
  return bad ? 8 : 0;
}

// ---------------------------------------------------------------- gate: every kernel and cuBLASLt bit for bit on the host
static int gate(Dev& d) {
  int fails = 0;
  std::mt19937_64 rng(0x5EED);
  auto rnd = [&](std::vector<int8_t>& v) { for (auto& x : v) x = (int8_t)((int)(rng() & 0xff) - 128); };
  // cuBLASLt at decode-sized and prefill-sized m
  for (int m : {1, 3, 8, 64, 300}) {
    const int n = 96, k = 448;
    std::vector<int8_t> A((size_t)m * k), W((size_t)n * k);
    rnd(A); rnd(W);
    CK(cudaMemcpy(d.X, A.data(), A.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.Wlm, W.data(), W.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(d.D, 0xA5, (size_t)m * n * 4));
    Lt p = lt_make(m, n, k, d.X, d.Wlm, d.D, d.st);
    if (p.algo_id == -2) { printf("{\"mode\": \"gate\", \"what\": \"cublaslt\", \"m\": %d, \"ok\": null, \"note\": \"no algorithm\"}\n", m); continue; }
    CK(cudaMemset(d.D, 0xA5, (size_t)m * n * 4));
    lt_run(p, d.X, d.Wlm, d.D, d.st);
    std::vector<int> Dh((size_t)m * n);
    CK(cudaMemcpy(Dh.data(), d.D, Dh.size() * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        int s = 0;
        for (int t = 0; t < k; t++) s += (int)A[(size_t)i * k + t] * (int)W[(size_t)j * k + t];
        bad += s != Dh[(size_t)i * n + j];
      }
    printf("{\"mode\": \"gate\", \"what\": \"cublaslt\", \"m\": %d, \"n\": %d, \"k\": %d, \"algo\": %d, \"rejected\": %d, "
           "\"mismatches\": %ld, \"ok\": %s}\n",
           m, n, k, p.algo_id, p.rejected, bad, bad ? "false" : "true");
    fails += bad != 0;
  }
  // attention at B = 3, L = 1536 (several blocks in x, a partial block), against the host
  {
    const int B = 3, L = 1536;
    const size_t per = (size_t)B * HKV * L * HD;
    std::vector<int8_t> K(per), V(per), q((size_t)B * HQ * HD), P((size_t)B * HQ * L);
    rnd(K); rnd(V); rnd(q); rnd(P);
    std::vector<int8_t> Vi(per);  // [B][HKV][L/4][HD][4]
    for (int b = 0; b < B; b++)
      for (int g = 0; g < HKV; g++)
        for (int key = 0; key < L; key++)
          for (int e = 0; e < HD; e++) {
            const size_t base = ((size_t)b * HKV + g) * L * HD;
            Vi[base + ((size_t)(key / 4) * HD + e) * 4 + key % 4] = V[base + (size_t)key * HD + e];
          }
    bool wr = false;
    int8_t* Kd = kv_layer(d, B, L, 0, false, &wr);
    int8_t* Vd = kv_layer(d, B, L, 0, true, &wr);
    CK(cudaMemcpy(Kd, K.data(), per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Vd, Vi.data(), per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.q, q.data(), q.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.P, P.data(), P.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(d.S, 0xA5, (size_t)B * HQ * L * 4));
    CK(cudaMemset(d.O, 0xA5, (size_t)B * HQ * HD * 4));
    enqueue_att(d, B, L, 0, &wr);
    CK(cudaStreamSynchronize(d.st));
    CK(cudaGetLastError());
    std::vector<int> S((size_t)B * HQ * L), O((size_t)B * HQ * HD);
    CK(cudaMemcpy(S.data(), d.S, S.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O.data(), d.O, O.size() * 4, cudaMemcpyDeviceToHost));
    long bs = 0, bo = 0;
    for (int b = 0; b < B; b++)
      for (int hq = 0; hq < HQ; hq++) {
        const int g = hq / 4;
        const size_t base = ((size_t)b * HKV + g) * L * HD;
        for (int key = 0; key < L; key++) {
          int s = 0;
          for (int e = 0; e < HD; e++) s += (int)q[((size_t)b * HQ + hq) * HD + e] * (int)K[base + (size_t)key * HD + e];
          bs += s != S[((size_t)b * HQ + hq) * L + key];
        }
        for (int e = 0; e < HD; e++) {
          int s = 0;
          for (int key = 0; key < L; key++) s += (int)P[((size_t)b * HQ + hq) * L + key] * (int)V[base + (size_t)key * HD + e];
          bo += s != O[((size_t)b * HQ + hq) * HD + e];
        }
      }
    printf("{\"mode\": \"gate\", \"what\": \"attention\", \"B\": %d, \"L\": %d, \"qk_mismatches\": %ld, \"pv_mismatches\": %ld, \"ok\": %s}\n",
           B, L, bs, bo, bs || bo ? "false" : "true");
    fails += bs != 0 || bo != 0;
  }
  fflush(stdout);
  return fails;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: dcap gate | bw GiB | lin M | att B L | step B L | fill B L FILL_SMS big|l2  [reps]\n"
                    "       dcap dgate | dalone W NC S | done W NC S | dfill B L split|co FILL_SMS W NC S [reps]\n"
                    "       (DFILL_COINS=<32 hex digits>: the round's coins for the d* modes)\n");
    return 1;
  }
  const std::string mode = argv[1];
  g_graph = !(getenv("DCAP_NOGRAPH") && atoi(getenv("DCAP_NOGRAPH")));
  if (const char* cs = getenv("DFILL_COINS")) {
    if (strlen(cs) != 32 || sscanf(cs, "%16llx%16llx", (unsigned long long*)&g_c0, (unsigned long long*)&g_c1) != 2) {
      fprintf(stderr, "DFILL_COINS must be 32 hex digits\n");
      return 1;
    }
  }
  Dev d;
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&d.st));  // blocking: the gate's cudaMemset and cudaMemcpy on the default stream order with it
  CK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, 0));
  if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); return 2; }
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
  if (nvmlDeviceGetHandleByPciBusId_v2(bus, &d.nv) != NVML_SUCCESS) { fprintf(stderr, "NVML handle failed\n"); return 2; }
  char uuid[96] = {0}, name[96] = {0};
  nvmlDeviceGetUUID(d.nv, uuid, sizeof uuid);
  nvmlDeviceGetName(d.nv, name, sizeof name);
  const char* want = getenv("GPU_LEASE_UUID");
  if (want && *want && strcmp(want, uuid) != 0) {
    fprintf(stderr, "device %s is not the leased %s\n", uuid, want);
    return 3;
  }
  fprintf(stderr, "device %s %s, %d SMs\n", name, uuid, d.sms);
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  CK(cudaMalloc(&g_ref, (size_t)256 * VOCAB * sizeof(int)));
  CK(cudaMalloc(&g_cnt, sizeof *g_cnt));
  const bool need_w = mode == "lin" || mode == "step" || mode == "gate" || mode == "fill" || mode == "dfill";
  if (need_w) {
    for (int l = 0; l < LAYERS; l++)
      for (int j = 0; j < 4; j++) {
        const size_t n = (size_t)LN[j] * LK[j];
        CK(cudaMalloc(&d.W[l][j], n));
        fill_s8<<<1024, 256, 0, d.st>>>(d.W[l][j], n, 0x1000 + l * 4 + j);
      }
    CK(cudaMalloc(&d.Wlm, (size_t)VOCAB * HID));
    fill_s8<<<1024, 256, 0, d.st>>>(d.Wlm, (size_t)VOCAB * HID, 0x77);
  }
  CK(cudaMalloc(&d.X, (size_t)8192 * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(d.X, (size_t)8192 * FF, 0x55);
  CK(cudaMalloc(&d.D, (size_t)1 << 30));
  const size_t maxBL = 786432;  // tokens of context over the batch
  d.kv_bytes = (mode == "lin") ? ((size_t)1 << 20) : maxBL * KV_TOKEN_BYTES / LAYERS * 4;  // room for 4 layers at maxBL
  if (mode == "att" || mode == "step" || mode == "bw" || mode == "fill" || mode == "dfill") {
    size_t fr, tot;
    CK(cudaMemGetInfo(&fr, &tot));
    d.kv_bytes = std::min((size_t)48 << 30, fr - ((size_t)3 << 30));
    if ((mode == "fill" || mode == "dfill") && argc > 3)  // every layer's KV distinct, and room for the filler's buffers
      d.kv_bytes = std::min(d.kv_bytes, (size_t)atoi(argv[2]) * atoi(argv[3]) * KV_TOKEN_BYTES + ((size_t)1 << 20));
  }
  CK(cudaMalloc(&d.kv, d.kv_bytes));
  fill_s8<<<4096, 256, 0, d.st>>>(d.kv, d.kv_bytes, 0x99);
  CK(cudaMalloc(&d.q, (size_t)256 * HQ * HD));
  fill_s8<<<64, 256, 0, d.st>>>(d.q, (size_t)256 * HQ * HD, 0x31);
  CK(cudaMalloc(&d.P, maxBL * HQ));
  fill_s8<<<1024, 256, 0, d.st>>>(d.P, maxBL * HQ, 0x32);
  CK(cudaMalloc(&d.S, maxBL * HQ * sizeof(int)));
  CK(cudaMalloc(&d.O, (size_t)256 * HQ * HD * sizeof(int)));
  CK(cudaMalloc(&d.sink, 64));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  const int reps = argc > 4 ? atoi(argv[4]) : (argc > 3 && (mode == "lin" || mode == "bw") ? atoi(argv[3]) : 9);
  int rc = 0;
  if (mode == "gate") rc = gate(d);
  else if (mode == "bw") run_bw(d, (size_t)(atof(argv[2]) * (1u << 30)) & ~(size_t)15, reps);
  else if (mode == "lin") run_step(d, 0, atoi(argv[2]), 0, reps);
  else if (mode == "att" || mode == "step") {
    const int B = atoi(argv[2]), L = atoi(argv[3]);
    if ((size_t)B * L > maxBL || L % 512 || B > 256) { fprintf(stderr, "B*L <= %zu, L %% 512 == 0, B <= 256\n", maxBL); return 1; }
    run_step(d, mode == "att" ? 1 : 2, B, L, reps);
  } else if (mode == "fill") {
    if (argc < 6) { fprintf(stderr, "fill B L FILL_SMS big|l2 [reps]\n"); return 1; }
    const int B = atoi(argv[2]), L = atoi(argv[3]), fsm = atoi(argv[4]);
    if ((size_t)B * L > maxBL || L % 512 || B > 256 || fsm <= 0 || fsm >= d.sms) {
      fprintf(stderr, "B*L <= %zu, L %% 512 == 0, B <= 256, 0 < FILL_SMS < %d\n", maxBL, d.sms);
      return 1;
    }
    CL(cublasLtCreate(&g_lt2));
    CK(cudaMalloc(&g_ws2, g_wsz));
    rc = run_fill(d, B, L, fsm, strcmp(argv[5], "big") == 0, argc > 6 ? atoi(argv[6]) : 9);
  } else if (mode == "dgate") {
    rc = dgate(d);
  } else if (mode == "dalone" || mode == "done") {
    if (argc < 5) { fprintf(stderr, "%s W NC S\n", mode.c_str()); return 1; }
    const int W = atoi(argv[2]), NC = atoi(argv[3]), S = atoi(argv[4]);
    if (W < 1 || W > 16 || S % 8 || S <= 0) { fprintf(stderr, "1 <= W <= 16, S a positive multiple of 8\n"); return 1; }
    if (mode == "dalone") run_dalone(d, W, NC, S);
    else run_done(d, W, NC, S);
  } else if (mode == "dfill") {
    if (argc < 9) { fprintf(stderr, "dfill B L split|co FILL_SMS W NC S [reps]\n"); return 1; }
    const int B = atoi(argv[2]), L = atoi(argv[3]), place = strcmp(argv[4], "co") == 0, fsm = atoi(argv[5]);
    const int W = atoi(argv[6]), NC = atoi(argv[7]), S = atoi(argv[8]);
    if ((size_t)B * L > maxBL || L % 512 || B > 256 || (!place && (fsm <= 0 || fsm >= d.sms)) || W < 1 || W > 16 || S % 8 ||
        S <= 0) {
      fprintf(stderr, "B*L <= %zu, L %% 512 == 0, B <= 256, 0 < FILL_SMS < %d (split), 1 <= W <= 16, 8 | S\n", maxBL, d.sms);
      return 1;
    }
    rc = run_dfill(d, B, L, place, fsm, W, NC, S, argc > 9 ? atoi(argv[9]) : 9);
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }
  return rc;
}
