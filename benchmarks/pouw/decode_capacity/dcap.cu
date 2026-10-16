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
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a dcap.cu -lcublasLt -lnvidia-ml
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
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
static std::vector<Rep> time_work(Dev& d, const std::function<void()>& work, int reps) {
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
static void run_step(Dev& d, int kind, int M, int L, int reps) {
  std::vector<Lt> p;
  if (kind != 1) p = lin_plans(d, M);
  bool wrapped = false;
  const std::vector<Rep> v = time_work(d, [&] {
    for (int l = 0; l < LAYERS; l++) {
      if (kind != 1 && p[0].algo_id != -2) lt_run(p[0], d.X, d.W[l][0], d.D, d.st);
      if (kind != 0) enqueue_att(d, M, L, l, &wrapped);
      for (int j = 1; j < 4; j++)
        if (kind != 1 && p[j].algo_id != -2) lt_run(p[j], d.X, d.W[l][j], d.D, d.st);
    }
    if (kind != 1 && M <= 256 && p[4].algo_id != -2) lt_run(p[4], d.X, d.Wlm, d.D, d.st);
  }, reps);
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
    fprintf(stderr, "usage: dcap gate | bw GiB | lin M | att B L | step B L  [reps]\n");
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
  const bool need_w = mode == "lin" || mode == "step" || mode == "gate";
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
  if (mode == "att" || mode == "step" || mode == "bw") {
    size_t fr, tot;
    CK(cudaMemGetInfo(&fr, &tot));
    d.kv_bytes = std::min((size_t)48 << 30, fr - ((size_t)3 << 30));
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
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }
  return rc;
}
