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
    fprintf(stderr, "usage: dcap gate | bw GiB | lin M | att B L | step B L | fill B L FILL_SMS big|l2  [reps]\n");
    return 1;
  }
  const std::string mode = argv[1];
  g_graph = !(getenv("DCAP_NOGRAPH") && atoi(getenv("DCAP_NOGRAPH")));
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
  const bool need_w = mode == "lin" || mode == "step" || mode == "gate" || mode == "fill";
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
  if (mode == "att" || mode == "step" || mode == "bw" || mode == "fill") {
    size_t fr, tot;
    CK(cudaMemGetInfo(&fr, &tot));
    d.kv_bytes = std::min((size_t)48 << 30, fr - ((size_t)3 << 30));
    if (mode == "fill" && argc > 3)  // every layer's KV distinct, and room for the filler's buffers
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
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }
  return rc;
}
