// overlap: does the in-step hm96-sha512 hashing of a prefill step hide under the step's GEMMs?
//
// Qwen3-8B's prefill GEMMs at M = 2,560 tokens (BF16, cuBLASLt autotuned, or a registered GEMM such as a CUTLASS
// configuration) run on a high-priority stream while a low-priority stream hashes the tensor the previous call committed:
// a 4 KiB row a leaf (hm96-sha512/row/v2 in a frame-v3-sha512 tree, commit_gpu's construction, bit for bit
// verity.ml.commitments'). The step's part of the hash is sharaw's v1 data path (a thread a leaf: x over the row, then h
// and the frame leaf's second block) and commit_gpu's tree schedule. Read ahead of the step and not timed: S = M_key y || c
// (Committer::salts), the frame leaf's first block (F) and the pads, as the served Commit takes its salts (OS bytes read
// ahead).
//
// Every hash variant is gated before anything is timed: its levels, computed while a GEMM runs beside it, must equal
// Committer's full path on the same bytes (device compare), and `dump` writes them for reference.py verify.
//
//   overlap dump PDIR OUT         a dump dir per (tensor, variant): params.bin, operand.bin, levels.bin, dump.json
//   overlap bench PDIR REPS       JSON lines: tuning, gates, interleaved arms per shape, the layer chain, a summary
//   overlap nsys PDIR             every GEMM and hash once, for a trace (kernel resources)
// PDIR holds p-N.bin, reference.py's CGP1 file for an N-leaf domain, for every N below (one Ks for all).
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <nvml.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define COMMIT_GPU_DOMS 4
#include "../../pouw/commit_gpu/commit_gpu.cuh"

#include "gemms.h"

using bf16 = __nv_bfloat16;

#define LT(x)                                                                         \
  do {                                                                                \
    cublasStatus_t s_ = (x);                                                          \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                \
      fprintf(stderr, "cuBLASLt %d at %s:%d: %s\n", (int)s_, __FILE__, __LINE__, #x); \
      exit(4);                                                                        \
    }                                                                                 \
  } while (0)

namespace ov {
constexpr u32 LB = 4096;
constexpr int NBLK = LB / 128;
constexpr int M = 2560, H = 4096, QKV = 6144, I = 12288;

// x = SHA-512(P(32768) || row): sharaw's row4k<false, FULL> (cursor/hash-sha-raw-16d5 @ 57bb0c0f7)
DEV void row4k(const uint8_t* __restrict__ p, u64* x) {
  cpy8(x, C.mid_P);
#pragma unroll 1
  for (int b = 0; b < NBLK; ++b) {
    u64 W[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 v = ldg16(p + 128 * b + 16 * c);
      W[2 * c] = be64(v.x, v.y);
      W[2 * c + 1] = be64(v.z, v.w);
    }
    sha::compress(x, W);
  }
  sha::compress_kw(x, C.kw_xpad);
}

// The data phase, a thread a leaf, into level 0.
template <int TPB, int MINB>
__global__ void __launch_bounds__(TPB, MINB) data(int dom, const uint8_t* __restrict__ rows, u64 n, const u64* __restrict__ S,
                                                  const u64* __restrict__ F, u64* __restrict__ lv0) {
  const u64 pos = (u64)blockIdx.x * TPB + threadIdx.x;
  if (pos >= n) return;
  u64 x[8], o[8];
  row4k(rows + pos * LB, x);
  leaf_from_digest<0>(dom, pos, x, S, o, nullptr, F);
  st8(lv0 + 8 * pos, o);
}

// Read ahead: F (the frame leaf's first block) for pos < n, the pads for n <= pos < width.
__global__ void pre(int dom, u64 n, u64 width, u64* __restrict__ F, u64* __restrict__ lv0) {
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 o[8];
  if (pos < n) {
    frame_first(dom, pos, o);
    st8(F + 8 * pos, o);
  } else if (pos < width) {
    pad(dom, pos, o);
    st8(lv0 + 8 * pos, o);
  }
}

// --- the hash server: one persistent CTA an SM beside the step, launched before it -----------------------------------------
// Work is a queue of 128-leaf chunks over the step's committed tensors (jobs) in order. A CTA takes the next chunk, waits for
// its job's ready flag (set on the step's stream once the tensor exists), hashes the chunk's leaves (a thread a leaf, rows
// read from L2), and folds the chunk's 7 levels in place; the CTA that finishes a job's last chunk folds the levels above.
// Same leaves and nodes as the data kernels and commit_gpu's tree, so the levels equal Committer's.
constexpr int SRV_T = 128, SRV_LG = 7;
constexpr int SRV_MAXD = 24;
struct SrvJob {
  const uint8_t* rows;
  const u64* S;
  const u64* F;
  u64* lv[SRV_MAXD];
  u64 n, width;
  int depth, dom;
  u32 chunk0, chunks;   // the job's first chunk in the queue, and its count (width / 128)
};
struct SrvCtl {
  u32 next;        // the queue's head
  u32 done[64];    // chunks finished, per job
  u32 cta[1024];   // a CTA's current chunk, and whether it folds its job's top (no shared memory: the server must fit in
  u32 last[1024];  // the reserved KiB a CTA leaves beside a GEMM's CTA)
};
DEV void row4k_cg(const uint8_t* __restrict__ p, u64* x) {
  cpy8(x, C.mid_P);
#pragma unroll 1
  for (int b = 0; b < NBLK; ++b) {
    u64 W[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 v = __ldcg(reinterpret_cast<const uint4*>(p + 128 * b + 16 * c));
      W[2 * c] = be64(v.x, v.y);
      W[2 * c + 1] = be64(v.z, v.w);
    }
    sha::compress(x, W);
  }
  sha::compress_kw(x, C.kw_xpad);
}
DEV void srv_node(int dom, int d, u64 i, u64* const* lv) {   // level d+1's node i from level d
  u64 L[8], R[8], o[8];
  ld8cg(lv[d] + 16 * i, L);
  ld8cg(lv[d] + 16 * i + 8, R);
  node(dom, d, i, L, R, o);
  st8(lv[d + 1] + 8 * i, o);
}
template <int MINB>
__global__ void __launch_bounds__(SRV_T, MINB) server(const SrvJob* __restrict__ jobs, int njobs, u32 total, const u32* ready, u32 epoch,
                                                      SrvCtl* ctl) {
  volatile u32* my = ctl->cta + blockIdx.x;
  volatile u32* myl = ctl->last + blockIdx.x;
  for (;;) {
    if (threadIdx.x == 0) {
      const u32 c = atomicAdd(&ctl->next, 1u);
      *my = c;
      if (c < total) {
        int j = 0;
        while (j + 1 < njobs && c >= jobs[j + 1].chunk0) ++j;
        const volatile u32* f = ready + j;
        while (*f < epoch) __nanosleep(200);
        __threadfence();
      }
    }
    __syncthreads();
    const u32 c = *my;
    __syncthreads();
    if (c >= total) return;
    int sj = 0;
    while (sj + 1 < njobs && c >= jobs[sj + 1].chunk0) ++sj;
    const SrvJob& J = jobs[sj];
    const u64 base = (u64)(c - J.chunk0) * SRV_T;
    const u64 pos = base + threadIdx.x;
    if (pos < J.n) {
      u64 x[8], o[8];
      row4k_cg(J.rows + pos * LB, x);
      leaf_from_digest<0>(J.dom, pos, x, J.S, o, nullptr, J.F);
      st8(J.lv[0] + 8 * pos, o);
    }
    __syncthreads();
    for (int d = 0; d < SRV_LG; ++d) {
      if (threadIdx.x < (SRV_T >> (d + 1))) srv_node(J.dom, d, (base >> (d + 1)) + threadIdx.x, J.lv);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      __threadfence();
      *myl = atomicAdd(&ctl->done[sj], 1u) == J.chunks - 1;
      __threadfence();
    }
    __syncthreads();
    const bool last = *myl;
    __syncthreads();
    if (last) {
      for (int d = SRV_LG; d < J.depth; ++d) {
        for (u64 i = threadIdx.x; i < (J.width >> (d + 1)); i += SRV_T) srv_node(J.dom, d, i, J.lv);
        __syncthreads();
      }
    }
  }
}
__global__ void srv_signal(u32* f, u32 v) {
  __threadfence();
  atomicExch(f, v);
}

__global__ void diff(const u64* a, const u64* b, u64 words, unsigned long long* bad) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (u64)gridDim.x * blockDim.x)
    if (a[i] != b[i]) atomicAdd(bad, 1ull);
}
// uniform bf16 in [-1, 1) from a counter hash
__global__ void fill_bf16(bf16* p, size_t n, u32 seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    u32 x = (u32)i * 0x9E3779B1u ^ seed ^ (u32)(i >> 32) * 0x85EBCA77u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = __float2bfloat16((float)(x >> 8) * (2.0f / 16777216.0f) - 1.0f);
  }
}
// vLLM's silu_and_mul: gu is [m, 2 i] (gate, then up), act [m, i]
__global__ void silu_mul(const bf16* __restrict__ gu, bf16* __restrict__ act, int m, int i) {
  const size_t tot = (size_t)m * i;
  for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; k < tot; k += (size_t)gridDim.x * blockDim.x) {
    const size_t r = k / i, c = k % i;
    const float g = __bfloat162float(gu[r * 2 * i + c]), u = __bfloat162float(gu[r * 2 * i + i + c]);
    act[k] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
  }
}
// The non-GEMM stand-ins (memory-bound, as vLLM's fused Triton kernels): fused_add_rms_norm over [m, h] (a block a row, 8 bf16
// a thread a load) and a vectorized silu(g) u.
__global__ void __launch_bounds__(512) rms_add(const bf16* __restrict__ x, bf16* __restrict__ res, const bf16* __restrict__ wt,
                                               bf16* __restrict__ y, int h) {
  const size_t row = blockIdx.x;
  float v[8], ss = 0;
  const int c = threadIdx.x * 8;
  if (c < h) {
    const uint4 a = *reinterpret_cast<const uint4*>(x + row * h + c), r = *reinterpret_cast<const uint4*>(res + row * h + c);
    const bf16* pa = reinterpret_cast<const bf16*>(&a);
    const bf16* pr = reinterpret_cast<const bf16*>(&r);
    uint4 o;
    bf16* po = reinterpret_cast<bf16*>(&o);
    for (int i = 0; i < 8; ++i) {
      v[i] = __bfloat162float(pa[i]) + __bfloat162float(pr[i]);
      po[i] = __float2bfloat16(v[i]);
      ss += v[i] * v[i];
    }
    *reinterpret_cast<uint4*>(res + row * h + c) = o;
  }
  __shared__ float red[32];
  for (int s = 16; s; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0;
    for (int s = 16; s; s >>= 1) t += __shfl_xor_sync(0xffffffffu, t, s);
    if (threadIdx.x == 0) red[0] = rsqrtf(t / h + 1e-6f);
  }
  __syncthreads();
  if (c < h) {
    const uint4 g = *reinterpret_cast<const uint4*>(wt + c);
    const bf16* pg = reinterpret_cast<const bf16*>(&g);
    uint4 o;
    bf16* po = reinterpret_cast<bf16*>(&o);
    for (int i = 0; i < 8; ++i) po[i] = __float2bfloat16(v[i] * red[0] * __bfloat162float(pg[i]));
    *reinterpret_cast<uint4*>(y + row * h + c) = o;
  }
}
__global__ void silu_mul8(const bf16* __restrict__ gu, bf16* __restrict__ act, int m, int i) {
  const size_t tot = (size_t)m * i / 8;
  for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; k < tot; k += (size_t)gridDim.x * blockDim.x) {
    const size_t r = k * 8 / i, c = k * 8 % i;
    const uint4 g = *reinterpret_cast<const uint4*>(gu + r * 2 * i + c), u = *reinterpret_cast<const uint4*>(gu + r * 2 * i + i + c);
    const bf16* pg = reinterpret_cast<const bf16*>(&g);
    const bf16* pu = reinterpret_cast<const bf16*>(&u);
    uint4 o;
    bf16* po = reinterpret_cast<bf16*>(&o);
    for (int j = 0; j < 8; ++j) {
      const float a = __bfloat162float(pg[j]);
      po[j] = __float2bfloat16(a / (1.0f + __expf(-a)) * __bfloat162float(pu[j]));
    }
    *reinterpret_cast<uint4*>(act + k * 8) = o;
  }
}
__global__ void maxrel(const bf16* a, const bf16* b, size_t n, float* out) {
  float m = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
    m = fmaxf(m, fabsf(x - y) / fmaxf(1.0f, fabsf(y)));
  }
  atomicMax(reinterpret_cast<int*>(out), __float_as_int(m));
}

// --- hash variants ----------------------------------------------------------------------------------------------------
struct Variant {
  std::string name;
  void (*launch)(int dom, const uint8_t* rows, u64 n, const u64* S, const u64* F, u64* lv0, cudaStream_t st);
  const void* fn;   // for cudaFuncGetAttributes
};
template <int TPB, int MINB>
void launch_data(int dom, const uint8_t* rows, u64 n, const u64* S, const u64* F, u64* lv0, cudaStream_t st) {
  data<TPB, MINB><<<(unsigned)((n + TPB - 1) / TPB), TPB, 0, st>>>(dom, rows, n, S, F, lv0);
}
#define VAR(T, B) Variant{"d" #T "_" #B, launch_data<T, B>, (const void*)data<T, B>}
static const std::vector<Variant>& variants() {
  static const std::vector<Variant> v = {VAR(128, 4), VAR(128, 5), VAR(128, 6), VAR(64, 8), VAR(256, 2)};
  return v;
}

// One committed tensor's hashing: its domain (slot, params), the read-ahead state (S in cm, F, pads in level 0) and its levels.
struct Hash {
  int dom = 0;
  u64 n = 0;
  Committer cm;   // S, levels and commit_gpu's tree schedule over them
  u64* F = nullptr;
  void init(const Params& p, u64 n_, int dom_) {
    dom = dom_; n = n_;
    cm.init(p, n, LB, false, false, 0, dom);
    CK(cudaMalloc(&F, n * 64));
  }
  void readahead(cudaStream_t st) {
    cm.salts<true>(nullptr, st);
    pre<<<(unsigned)((cm.width + 127) / 128), 128, 0, st>>>(dom, n, cm.width, F, cm.lv[0]);
  }
  void run(const Variant& v, const uint8_t* rows, cudaStream_t st) {
    v.launch(dom, rows, n, cm.S, F, cm.lv[0], st);
    cm.nodes(st);
  }
  u64 words() const { u64 w = 0; for (int d = 0; d <= cm.depth; ++d) w += 8 * (cm.width >> d); return w; }
};

// --- GEMMs ------------------------------------------------------------------------------------------------------------
struct Shape {
  std::string name;
  int n, k;
  bool residual;     // D = X W^T + R (o_proj and down: the residual add inside the unit)
  std::string hashed;  // the committed tensor this call's unit outputs
};
static const std::vector<Shape> SHAPES = {
    {"qkv", QKV, H, false, "qkv"}, {"o", H, H, true, "res"}, {"gate_up", 2 * I, H, false, "act"}, {"down", H, I, true, "res"}};
// committed tensors: name -> (bf16 columns, domain slot); attn is rope_q || kv_k || attention (non-GEMM outputs)
// (res2, the post-MLP residual, is res's domain with its own buffer and levels, so a layer's two residuals hash concurrently)
static const std::map<std::string, std::pair<int, int>> TENSORS = {
    {"qkv", {QKV, 0}}, {"attn", {9216, 1}}, {"res", {H, 2}}, {"act", {I, 3}}, {"res2", {H, 2}}};
constexpr int NSLOT = 4;   // tensor slots: a layer of the chain each (the per-shape arms double-buffer over slots 0 and 1)
constexpr int NB = 20;     // hash streams: one per call of the chain's 4 layers

struct LtGemm {
  cublasLtHandle_t h{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t la{}, lb{}, lc{};
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
  int best = 0;
  void* ws = nullptr;
  size_t wsz = 64ull << 20;
  int m = 0, n = 0, k = 0;
  void init(int m_, int n_, int k_) {
    m = m_; n = n_; k = k_;
    LT(cublasLtCreate(&h));
    LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    // column-major views: W [n, k] row-major is k x n (op T), X [m, k] is k x m, Y [m, n] is n x m
    LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, n, k));
    LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, m, k));
    LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, n, m, n));
    CK(cudaMalloc(&ws, wsz));
    cublasLtMatmulPreference_t pref;
    LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
    algos.resize(16);
    int got = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, lc, lc, pref, 16, algos.data(), &got));
    algos.resize(got);
    cublasLtMatmulPreferenceDestroy(pref);
  }
  cublasStatus_t try_run(int a, const bf16* X, const bf16* W, const bf16* R, bf16* Y, cudaStream_t st) const {
    const float one = 1.0f, beta = R ? 1.0f : 0.0f;
    return cublasLtMatmul(h, op, &one, W, la, X, lb, &beta, R ? R : Y, lc, Y, lc, &algos[a].algo, ws, wsz, st);
  }
  void run(int a, const bf16* X, const bf16* W, const bf16* R, bf16* Y, cudaStream_t st) const { LT(try_run(a, X, W, R, Y, st)); }
  int cfg(int a, cublasLtMatmulAlgoConfigAttributes_t at) const {
    int v = -1;
    size_t w = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&algos[a].algo, at, &v, sizeof v, &w);
    return v;
  }
};

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  unsigned sm() { unsigned c = 0; if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c); return c; }
  unsigned mw() { unsigned p = 0; if (ok) nvmlDeviceGetPowerUsage(d, &p); return p; }
};
static double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

// --- the bench's world: weights, inputs, outputs, hashes ------------------------------------------------------------
struct World {
  std::map<std::string, Params> params;   // by tensor
  cudaStream_t A{}, B{};                  // A: the step (high priority), B: the hash (low priority)
  cudaStream_t P[NB]{};                   // the hash stream pool (low priority; P[0] is B)
  std::map<std::string, bf16*> W, X, R;   // by shape: weight, input, residual
  bf16* out[2] = {nullptr, nullptr};      // GEMM outputs, double-buffered (the largest, gate_up's)
  std::map<std::string, bf16*> T[NSLOT];  // committed tensors by slot
  std::map<std::string, std::unique_ptr<Hash>> hs[NSLOT];   // their hashing state
  std::map<std::string, LtGemm> lt;
  std::map<std::string, std::vector<ExtGemm>> ext;   // registered GEMMs (gemms.h) by shape
  std::vector<cudaEvent_t> ev;
  int nev = 0;
  cudaEvent_t e() { if (nev == (int)ev.size()) { ev.emplace_back(); CK(cudaEventCreateWithFlags(&ev.back(), cudaEventDisableTiming)); } return ev[nev++]; }
  void join() {   // A and B wait on each other
    cudaEvent_t a = e(), b = e();
    CK(cudaEventRecord(a, A)); CK(cudaEventRecord(b, B));
    CK(cudaStreamWaitEvent(B, a, 0)); CK(cudaStreamWaitEvent(A, b, 0));
  }
  void join_all() {   // A and every pool stream
    cudaEvent_t a = e();
    CK(cudaEventRecord(a, A));
    for (auto p : P) {
      cudaEvent_t x = e();
      CK(cudaEventRecord(x, p));
      CK(cudaStreamWaitEvent(A, x, 0));
      CK(cudaStreamWaitEvent(p, a, 0));
    }
  }
  static u64 leaves(const std::string& t) { return (u64)M * TENSORS.at(t).first * 2 / LB; }
  static void sync(const std::string& what) {
    const cudaError_t e_ = cudaDeviceSynchronize();
    if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s after %s\n", cudaGetErrorString(e_), what.c_str()); exit(3); }
  }
  void init(const std::string& pdir) {
    int lo, hi;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CK(cudaStreamCreateWithPriority(&A, cudaStreamNonBlocking, hi));
    CK(cudaStreamCreateWithPriority(&B, cudaStreamNonBlocking, lo));
    P[0] = B;
    for (int i = 1; i < NB; ++i) CK(cudaStreamCreateWithPriority(&P[i], cudaStreamNonBlocking, lo));
    u32 seed = 1;
    for (const auto& s : SHAPES) {
      const size_t wn = (size_t)s.n * s.k, xn = (size_t)M * s.k;
      CK(cudaMalloc(&W[s.name], wn * 2)); fill_bf16<<<4096, 256>>>(W[s.name], wn, seed++);
      CK(cudaMalloc(&X[s.name], xn * 2)); fill_bf16<<<4096, 256>>>(X[s.name], xn, seed++);
      if (s.residual) { CK(cudaMalloc(&R[s.name], (size_t)M * s.n * 2)); fill_bf16<<<4096, 256>>>(R[s.name], (size_t)M * s.n, seed++); }
      lt[s.name].init(M, s.n, s.k);
      ext[s.name] = ext_gemms(M, s.n, s.k);
      sync("operands " + s.name);
    }
    for (int b = 0; b < 2; ++b) {
      CK(cudaMalloc(&out[b], (size_t)M * 2 * I * 2));
      fill_bf16<<<4096, 256>>>(out[b], (size_t)M * 2 * I, 100 + b);
    }
    for (int b = 0; b < NSLOT; ++b) {
      for (const auto& [t, cd] : TENSORS) {
        const size_t elems = (size_t)M * cd.first;
        CK(cudaMalloc(&T[b][t], elems * 2));
        fill_bf16<<<4096, 256>>>(T[b][t], elems, 200 + 10 * b + cd.second);
        const u64 n = leaves(t);
        if (!params.count(t)) params[t] = read_params(pdir + "/p-" + std::to_string(n) + ".bin");
        sync("tensor " + t);
        hs[b][t] = std::make_unique<Hash>();
        hs[b][t]->init(params[t], n, cd.second);
        sync("committer " + t);
        hs[b][t]->readahead(A);
        sync("readahead " + t);
      }
    }
    CK(cudaStreamSynchronize(A));
    CK(cudaDeviceSynchronize());
  }
  // --- the hash server's plan: the tensors of slots 0 .. L-1, a layer at a time in the order their units finish
  static constexpr const char* LAYER_TENSORS[5] = {"qkv", "attn", "res", "act", "res2"};
  std::vector<SrvJob> jobs;
  SrvJob* djobs = nullptr;
  SrvCtl* ctl = nullptr;
  u32* ready = nullptr;
  u32 epoch = 0, total = 0;
  int sms = 0;
  // the server's variants: at most 128 registers a thread (4 CTAs an SM's worth), or 80 (6)
  static constexpr int SRV_MINB[2] = {4, 6};
  int srv_regs[2] = {0, 0};
  void srv_init(int L) {
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    if (sms > 1024) { fprintf(stderr, "server: too many SMs\n"); exit(6); }
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, server<4>));
    srv_regs[0] = fa.numRegs;
    printf("{\"kernel\": \"srv4\", \"regs\": %d, \"smem\": %zu, \"local\": %zu, \"max_threads\": %d}\n", fa.numRegs, fa.sharedSizeBytes,
           fa.localSizeBytes, fa.maxThreadsPerBlock);
    CK(cudaFuncGetAttributes(&fa, server<6>));
    srv_regs[1] = fa.numRegs;
    printf("{\"kernel\": \"srv6\", \"regs\": %d, \"smem\": %zu, \"local\": %zu, \"max_threads\": %d}\n", fa.numRegs, fa.sharedSizeBytes,
           fa.localSizeBytes, fa.maxThreadsPerBlock);
    jobs.clear();
    total = 0;
    for (int l = 0; l < L; ++l)
      for (const char* t : LAYER_TENSORS) {
        const Hash& h = *hs[l][t];
        if (h.cm.depth < SRV_LG || h.cm.depth >= SRV_MAXD) { fprintf(stderr, "server: depth %d out of range\n", h.cm.depth); exit(6); }
        SrvJob j{};
        j.rows = reinterpret_cast<const uint8_t*>(T[l][t]);
        j.S = h.cm.S; j.F = h.F;
        for (int d = 0; d <= h.cm.depth; ++d) j.lv[d] = h.cm.lv[d];
        j.n = h.n; j.width = h.cm.width; j.depth = h.cm.depth; j.dom = h.dom;
        j.chunk0 = total; j.chunks = (u32)(h.cm.width / SRV_T);
        total += j.chunks;
        jobs.push_back(j);
      }
    if (jobs.size() > 64) { fprintf(stderr, "server: too many jobs\n"); exit(6); }
    if (!djobs) {
      CK(cudaMalloc(&djobs, 64 * sizeof(SrvJob)));
      CK(cudaMalloc(&ctl, sizeof(SrvCtl)));
      CK(cudaMalloc(&ready, 64 * 4));
      CK(cudaMemset(ready, 0, 64 * 4));
    }
    CK(cudaMemcpy(djobs, jobs.data(), jobs.size() * sizeof(SrvJob), cudaMemcpyHostToDevice));
  }
  static int job(int l, const std::string& t) {
    for (int i = 0; i < 5; ++i) if (t == LAYER_TENSORS[i]) return 5 * l + i;
    return -1;
  }
  // ep 0: every job is ready at once (the server alone)
  void srv_launch(int var, u32 ep, cudaStream_t st) {
    CK(cudaMemsetAsync(ctl, 0, sizeof(SrvCtl), st));
    if (var == 0) server<4><<<sms, SRV_T, 0, st>>>(djobs, (int)jobs.size(), total, ready, ep, ctl);
    else server<6><<<sms, SRV_T, 0, st>>>(djobs, (int)jobs.size(), total, ready, ep, ctl);
  }
  void sig(int j, u32 ep) { srv_signal<<<1, 1, 0, A>>>(ready + j, ep); }
  // the server fits on every SM beside a CTA of this GEMM (else the step would wait on it, and it on the step)
  bool srv_fits(const ExtGemm& g, int var) const {
    int regs_sm, smem_sm, rsv;
    CK(cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0));
    CK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    CK(cudaDeviceGetAttribute(&rsv, cudaDevAttrReservedSharedMemoryPerBlock, 0));
    auto warp_regs = [](int r) { return ((r * 32 + 255) / 256) * 256; };
    const int need_regs = warp_regs(g.regs) * ((g.threads + 31) / 32) + warp_regs(srv_regs[var]) * (SRV_T / 32);
    const int need_smem = g.smem + rsv + rsv;
    return g.threads > 0 && need_regs <= regs_sm && need_smem <= smem_sm && g.threads + SRV_T <= 2048;
  }
  // the committed tensor a shape's unit outputs, from its GEMM's output (gate_up: silu(g) u; else the output itself)
  void unit_out(const Shape& s, const bf16* y, int b, cudaStream_t st) {
    bf16* t = T[b][s.hashed];
    if (s.name == "gate_up") silu_mul<<<2048, 256, 0, st>>>(y, t, M, I);
    else CK(cudaMemcpyAsync(t, y, (size_t)M * s.n * 2, cudaMemcpyDeviceToDevice, st));
  }
};

// --- the GEMM arms of a shape -----------------------------------------------------------------------------------------
struct GemmArm {
  std::string name;
  std::function<void(const bf16* X, const bf16* W, const bf16* R, bf16* Y, cudaStream_t)> run;
};
static std::vector<GemmArm> gemm_arms(World& w, const Shape& s) {
  std::vector<GemmArm> g;
  LtGemm& L = w.lt[s.name];
  g.push_back({"lt", [&L](const bf16* X, const bf16* W, const bf16* R, bf16* Y, cudaStream_t st) { L.run(L.best, X, W, R, Y, st); }});
  for (auto& e : w.ext[s.name]) {
    ExtGemm* p = &e;
    g.push_back({p->name, [p](const bf16* X, const bf16* W, const bf16* R, bf16* Y, cudaStream_t st) {
                   if (p->run(p->state, X, W, R, Y, st)) { fprintf(stderr, "%s failed\n", p->name.c_str()); exit(5); }
                 }});
  }
  return g;
}

static void tune(World& w, const Shape& s) {
  LtGemm& L = w.lt[s.name];
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  const bf16* R = s.residual ? w.R[s.name] : nullptr;
  double bt = 1e30;
  for (int i = 0; i < (int)L.algos.size(); ++i) {
    if (L.try_run(i, w.X[s.name], w.W[s.name], R, w.out[0], w.A) != CUBLAS_STATUS_SUCCESS) {
      printf("{\"tune\": \"%s\", \"algo\": %d, \"failed\": true}\n", s.name.c_str(), i);
      continue;
    }
    for (int r = 0; r < 3; ++r) L.run(i, w.X[s.name], w.W[s.name], R, w.out[0], w.A);
    CK(cudaEventRecord(a, w.A));
    for (int r = 0; r < 20; ++r) L.run(i, w.X[s.name], w.W[s.name], R, w.out[0], w.A);
    CK(cudaEventRecord(b, w.A));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    ms /= 20;
    printf("{\"tune\": \"%s\", \"algo\": %d, \"id\": %d, \"tile\": %d, \"stages\": %d, \"splitk\": %d, \"swizzle\": %d, \"ms\": %.5f}\n",
           s.name.c_str(), i, L.cfg(i, CUBLASLT_ALGO_CONFIG_ID), L.cfg(i, CUBLASLT_ALGO_CONFIG_TILE_ID),
           L.cfg(i, CUBLASLT_ALGO_CONFIG_STAGES_ID), L.cfg(i, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
           L.cfg(i, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING), ms);
    if (ms < bt) { bt = ms; L.best = i; }
  }
  printf("{\"tuned\": \"%s\", \"best\": %d, \"of\": %zu, \"ms\": %.5f}\n", s.name.c_str(), L.best, L.algos.size(), bt);
  fflush(stdout);
}

// A hash variant's gate: its levels for tensor t (buffer b), computed beside a running GEMM, against Committer's full path.
static bool gate(World& w, const Variant& v, const std::string& t, int b, const GemmArm& g, const Shape& s) {
  Hash& h = *w.hs[b][t];
  static std::map<std::string, std::unique_ptr<Committer>> ref;
  if (!ref.count(t)) { ref[t] = std::make_unique<Committer>(); ref[t]->init(w.params[t], h.n, LB, false, false, 0, h.dom); }
  Committer& rc = *ref[t];
  const auto* rows = reinterpret_cast<const uint8_t*>(w.T[b][t]);
  CK(cudaMemsetAsync(h.cm.lv[1], 0xA5, 64 * (2 * h.cm.width - h.cm.width - 1), w.B));
  // level 0's leaves (not its pads, read ahead) poisoned too
  CK(cudaMemsetAsync(h.cm.lv[0], 0xA5, 64 * h.n, w.B));
  w.nev = 0;
  w.join();
  g.run(w.X[s.name], w.W[s.name], s.residual ? w.R[s.name] : nullptr, w.out[1 - b], w.A);
  h.run(v, rows, w.B);
  w.join();
  rc.salts<true>(nullptr, w.A);
  rc.leaves<RowBytes>(RowBytes{rows, LB, LB}, w.A);
  rc.nodes(w.A);
  unsigned long long* bad;
  CK(cudaMallocManaged(&bad, 8));
  *bad = 0;
  diff<<<1024, 256, 0, w.A>>>(rc.levels, h.cm.levels, h.words(), bad);
  CK(cudaStreamSynchronize(w.A));
  CK(cudaDeviceSynchronize());
  const unsigned long long nb = *bad;
  cudaFree(bad);
  return nb == 0;
}

// The server's gate: poison every job's levels (its leaves and nodes; the pads stay, read ahead), then after the server has
// run, each job's levels against Committer's full path on the same rows.
static void srv_poison(World& w) {
  for (const auto& j : w.jobs) {
    CK(cudaMemsetAsync(j.lv[0], 0xA5, 64 * j.n, w.A));
    CK(cudaMemsetAsync(j.lv[1], 0xA5, 64 * (j.width - 1), w.A));
  }
  CK(cudaStreamSynchronize(w.A));
}
static unsigned long long srv_check(World& w) {
  static std::map<std::string, std::unique_ptr<Committer>> ref;
  unsigned long long* bad;
  CK(cudaMallocManaged(&bad, 8));
  *bad = 0;
  for (size_t i = 0; i < w.jobs.size(); ++i) {
    const int l = (int)i / 5;
    const std::string t = World::LAYER_TENSORS[i % 5];
    const Hash& h = *w.hs[l][t];
    if (!ref.count(t)) { ref[t] = std::make_unique<Committer>(); ref[t]->init(w.params[t], h.n, LB, false, false, 0, h.dom); }
    Committer& rc = *ref[t];
    const auto* rows = reinterpret_cast<const uint8_t*>(w.T[l][t]);
    rc.salts<true>(nullptr, w.A);
    rc.leaves<RowBytes>(RowBytes{rows, LB, LB}, w.A);
    rc.nodes(w.A);
    diff<<<1024, 256, 0, w.A>>>(rc.levels, h.cm.levels, h.words(), bad);
  }
  CK(cudaStreamSynchronize(w.A));
  CK(cudaDeviceSynchronize());
  const unsigned long long nb = *bad;
  cudaFree(bad);
  return nb;
}

// --- dump: every (tensor, variant) beside the shape's cuBLASLt GEMM, for reference.py verify ---------------------------
static int dump(const std::string& pdir, const std::string& outd) {
  World w;
  w.init(pdir);
  int fails = 0;
  for (const auto& [t, cd] : TENSORS) {
    if (t == "res2") continue;   // res's domain, size and kernels
    const Shape& s = *std::find_if(SHAPES.begin(), SHAPES.end(), [&](const Shape& x) { return x.hashed == t || (t == "attn" && x.name == "o"); });
    auto gs = gemm_arms(w, s);
    // a real unit output: the GEMM's own result where the unit is a GEMM's (attn stays filled)
    if (t != "attn") {
      gs[0].run(w.X[s.name], w.W[s.name], s.residual ? w.R[s.name] : nullptr, w.out[0], w.A);
      w.unit_out(s, w.out[0], 0, w.A);
      CK(cudaStreamSynchronize(w.A));
    }
    Hash& h = *w.hs[0][t];
    const size_t bytes = h.n * LB;
    std::vector<uint8_t> op(bytes);
    CK(cudaMemcpy(op.data(), w.T[0][t], bytes, cudaMemcpyDeviceToHost));
    const std::string od = outd + "/" + t;
    if (system(("mkdir -p '" + od + "'").c_str())) return 2;
    writef(od + "/operand.bin", op.data(), op.size());
    for (const auto& v : variants()) {
      const bool ok = gate(w, v, t, 0, gs[0], s);
      fails += !ok;
      const std::string d = od + "-" + v.name;
      if (system(("mkdir -p '" + d + "' && ln -sf ../" + t + "/operand.bin '" + d + "/operand.bin' && cp '" + pdir + "/p-" +
                  std::to_string(h.n) + ".bin' '" + d + "/params.bin'").c_str()))
        return 2;
      std::vector<u64> lv(h.words());
      CK(cudaMemcpy(lv.data(), h.cm.levels, lv.size() * 8, cudaMemcpyDeviceToHost));
      std::vector<uint8_t> lb(lv.size() * 8);
      for (size_t i = 0; i < lv.size(); ++i) sha::put_be64(lb.data() + 8 * i, lv[i]);
      writef(d + "/levels.bin", lb.data(), lb.size());
      const std::string js = "{\"row_bytes\": 4096, \"tensor\": \"" + t + "\", \"variant\": \"" + v.name + "\", \"beside\": \"" + s.name + ":lt\"}";
      writef(d + "/dump.json", js.data(), js.size());
      printf("{\"dump\": \"%s\", \"n\": %llu, \"device_gate\": %s}\n", d.c_str(), (unsigned long long)h.n, ok ? "true" : "false");
    }
  }
  // the hash server over slot 0's tensors, alone (every job ready at once)
  w.srv_init(1);
  srv_poison(w);
  w.nev = 0;
  w.join_all();
  w.srv_launch(0, 0, w.B);
  w.join_all();
  CK(cudaStreamSynchronize(w.A));
  World::sync("server");
  const unsigned long long sb = srv_check(w);
  fails += sb != 0;
  for (const char* t : {"qkv", "attn", "res", "act"}) {
    const Hash& h = *w.hs[0][t];
    const std::string d = outd + "/" + t + "-srv";
    if (system(("mkdir -p '" + d + "' && ln -sf ../" + t + "/operand.bin '" + d + "/operand.bin' && cp '" + pdir + "/p-" +
                std::to_string(h.n) + ".bin' '" + d + "/params.bin'").c_str()))
      return 2;
    std::vector<u64> lv(h.words());
    CK(cudaMemcpy(lv.data(), h.cm.levels, lv.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> lb(lv.size() * 8);
    for (size_t i = 0; i < lv.size(); ++i) sha::put_be64(lb.data() + 8 * i, lv[i]);
    writef(d + "/levels.bin", lb.data(), lb.size());
    const std::string js = std::string("{\"row_bytes\": 4096, \"tensor\": \"") + t + "\", \"variant\": \"srv\", \"beside\": \"none\"}";
    writef(d + "/dump.json", js.data(), js.size());
    printf("{\"dump\": \"%s\", \"n\": %llu, \"device_gate\": %s}\n", d.c_str(), (unsigned long long)h.n, sb == 0 ? "true" : "false");
  }
  fflush(stdout);
  return fails ? 1 : 0;
}

// --- bench --------------------------------------------------------------------------------------------------------------
struct Arm {
  std::string shape, name, gemm, var;   // var empty: no hash; gemm empty: no GEMM
  bool seq = false;                     // the hash after the GEMM on A (the separate pass)
  std::vector<double> ms;
  std::vector<unsigned> mhz;
};

static void resources(const std::vector<Variant>& vs) {
  int dev, sms;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  cudaDeviceProp pr;
  CK(cudaGetDeviceProperties(&pr, dev));
  printf("{\"device\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"regs_per_sm\": %d, \"smem_per_sm\": %zu}\n", pr.name, pr.major,
         pr.minor, sms, pr.regsPerMultiprocessor, pr.sharedMemPerMultiprocessor);
  for (const auto& v : vs) {
    cudaFuncAttributes a;
    CK(cudaFuncGetAttributes(&a, v.fn));
    printf("{\"kernel\": \"%s\", \"regs\": %d, \"smem\": %zu, \"local\": %zu, \"max_threads\": %d}\n", v.name.c_str(), a.numRegs,
           a.sharedSizeBytes, a.localSizeBytes, a.maxThreadsPerBlock);
  }
}

static int bench(const std::string& pdir, int reps, bool nsys) {
  World w;
  w.init(pdir);
  Nv nv;
  const auto& vs = variants();
  resources(vs);
  const int iters = nsys ? 2 : 16;
  std::vector<Arm> all;
  for (const auto& s : SHAPES) {
    tune(w, s);
    auto gs = gemm_arms(w, s);
    const std::string t = s.hashed;
    // GEMM sanity: a registered GEMM's output against cuBLASLt's (accumulation order differs; reported, not gated)
    for (size_t gi = 1; gi < gs.size(); ++gi) {
      gs[0].run(w.X[s.name], w.W[s.name], s.residual ? w.R[s.name] : nullptr, w.out[0], w.A);
      gs[gi].run(w.X[s.name], w.W[s.name], s.residual ? w.R[s.name] : nullptr, w.out[1], w.A);
      float* mr;
      CK(cudaMallocManaged(&mr, 4));
      *mr = 0;
      maxrel<<<1024, 256, 0, w.A>>>(w.out[1], w.out[0], (size_t)M * s.n, mr);
      CK(cudaStreamSynchronize(w.A));
      printf("{\"gemm_check\": \"%s\", \"gemm\": \"%s\", \"max_rel_vs_lt\": %.3g}\n", s.name.c_str(), gs[gi].name.c_str(), *mr);
      cudaFree(mr);
    }
    // real unit outputs in both buffers
    for (int b = 0; b < 2; ++b) {
      gs[0].run(w.X[s.name], w.W[s.name], s.residual ? w.R[s.name] : nullptr, w.out[b], w.A);
      w.unit_out(s, w.out[b], b, w.A);
    }
    CK(cudaStreamSynchronize(w.A));
    std::vector<const Variant*> ok;
    for (const auto& v : vs) {
      bool g = true;
      for (const auto& ga : gs) g &= gate(w, v, t, 0, ga, s);
      printf("{\"gate\": \"%s\", \"tensor\": \"%s\", \"variant\": \"%s\", \"beside\": %zu, \"ok\": %s}\n", s.name.c_str(), t.c_str(),
             v.name.c_str(), gs.size(), g ? "true" : "false");
      if (g) ok.push_back(&v);
    }
    fflush(stdout);
    std::vector<Arm> arms;
    for (const auto& ga : gs) arms.push_back({s.name, ga.name, ga.name, ""});
    for (const auto* v : ok) {
      arms.push_back({s.name, "h:" + v->name, "", v->name});
      for (const auto& ga : gs) {
        arms.push_back({s.name, "seq:" + ga.name + ":" + v->name, ga.name, v->name, true});
        arms.push_back({s.name, "par:" + ga.name + ":" + v->name, ga.name, v->name});
      }
    }
    std::map<std::string, const Variant*> vby;
    for (const auto* v : ok) vby[v->name] = v;
    std::map<std::string, const GemmArm*> gby;
    for (const auto& ga : gs) gby[ga.name] = &ga;
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    auto item = [&](const Arm& a) {
      w.nev = 0;
      w.join();
      CK(cudaEventRecord(t0, w.A));
      for (int i = 0; i < iters; ++i) {
        const int b = i & 1;   // the GEMM writes out[b]; the hash reads the tensor buffer 1 - b (the last call's)
        if (!a.gemm.empty()) gby[a.gemm]->run(w.X[s.name], w.W[s.name], s.residual ? w.R[s.name] : nullptr, w.out[b], w.A);
        if (!a.var.empty()) w.hs[1 - b][t]->run(*vby[a.var], reinterpret_cast<const uint8_t*>(w.T[1 - b][t]), a.seq ? w.A : w.B);
        w.join();
      }
      CK(cudaEventRecord(t1, w.A));
      CK(cudaEventSynchronize(t1));
      float ms;
      CK(cudaEventElapsedTime(&ms, t0, t1));
      return (double)ms / iters;
    };
    for (const auto& a : arms) item(a);   // warm-up
    for (int r = 0; r < reps; ++r)
      for (size_t j = 0; j < arms.size(); ++j) {
        Arm& a = arms[(j + r) % arms.size()];
        a.ms.push_back(item(a));
        a.mhz.push_back(nv.sm());
      }
    for (const auto& a : arms) {
      printf("{\"shape\": \"%s\", \"arm\": \"%s\", \"iters\": %d, \"median_ms\": %.5f, \"ms\": [", a.shape.c_str(), a.name.c_str(), iters,
             median(a.ms));
      for (size_t i = 0; i < a.ms.size(); ++i) printf("%s%.5f", i ? ", " : "", a.ms[i]);
      printf("], \"sm_mhz\": [");
      for (size_t i = 0; i < a.mhz.size(); ++i) printf("%s%u", i ? ", " : "", a.mhz[i]);
      printf("]}\n");
      all.push_back(a);
    }
    // the summary: against cuBLASLt alone plus the separate hash
    const double lt = median(arms[0].ms);
    for (const auto* v : ok) {
      double hm = 0;
      for (const auto& a : arms) if (a.name == "h:" + v->name) hm = median(a.ms);
      for (const auto& ga : gs) {
        double g = 0, par = 0, seq = 0;
        for (const auto& a : arms) {
          if (a.name == ga.name) g = median(a.ms);
          if (a.name == "par:" + ga.name + ":" + v->name) par = median(a.ms);
          if (a.name == "seq:" + ga.name + ":" + v->name) seq = median(a.ms);
        }
        printf("{\"summary\": \"%s\", \"gemm\": \"%s\", \"variant\": \"%s\", \"leaves\": %llu, \"lt_ms\": %.5f, \"gemm_ms\": %.5f, "
               "\"hash_ms\": %.5f, \"seq_ms\": %.5f, \"par_ms\": %.5f, \"hidden_vs_own_gemm\": %.4f, \"hidden_vs_lt\": %.4f, "
               "\"net_vs_lt_plus_hash\": %.4f, \"exposed_hash_ms\": %.5f}\n",
               s.name.c_str(), ga.name.c_str(), v->name.c_str(), (unsigned long long)w.hs[0][t]->n, lt, g, hm, seq, par,
               (g + hm - par) / hm, (lt + hm - par) / hm, par / (lt + hm), par - lt);
      }
    }
    fflush(stdout);
  }
  // --- the layer chain: L layers of the prefill step's GEMMs on A, each layer's committed tensors in their own slot -------
  // A layer's calls: qkv after qkv; attn (rope_q || kv_k || attention) and res (the post-attention residual) after o; act
  // after gate_up (silu(g) u); res2 (the post-MLP residual) after down. Modes: 0 the GEMMs alone; 1 h1, the hashes alone,
  // one after another on one stream (each call latency-bound); 2 hN, the hashes alone, each call on its own stream (all
  // of the chain's calls at once: the separate pass at the throughput the hash reaches over the step's bytes); 3 seq, each
  // hash on A after its GEMM; 4 par1, the hashes on one stream beside the GEMMs, each after its tensor exists; 5 parN, as
  // par1 with each call on its own stream; 6 srv, the hash server beside the GEMMs (each tensor's jobs signalled once it
  // exists); 7 the server alone (every job ready at once).
  const int L = nsys ? 1 : NSLOT;
  for (int l = 0; l < L; ++l)   // every slot holds real unit outputs
    for (const auto& s : SHAPES) {
      w.lt[s.name].run(w.lt[s.name].best, w.X[s.name], w.W[s.name], s.residual ? w.R[s.name] : nullptr, w.out[0], w.A);
      bf16* t = w.T[l][s.name == "down" ? "res2" : s.hashed];
      if (s.name == "gate_up") silu_mul<<<2048, 256, 0, w.A>>>(w.out[0], t, M, I);
      else CK(cudaMemcpyAsync(t, w.out[0], (size_t)M * s.n * 2, cudaMemcpyDeviceToDevice, w.A));
    }
  CK(cudaStreamSynchronize(w.A));
  struct Chain { std::string name; std::string gemm; std::string var; int mode; };
  std::vector<Chain> chains;
  std::vector<std::string> gnames = {"lt"};
  for (const auto& e : w.ext[SHAPES[0].name]) {
    bool every = true;
    for (const auto& s : SHAPES) every &= std::any_of(w.ext[s.name].begin(), w.ext[s.name].end(), [&](const ExtGemm& x) { return x.name == e.name; });
    if (every) gnames.push_back(e.name);
  }
  for (const auto& g : gnames) chains.push_back({"chain:" + g, g, "", 0});
  for (const auto& v : vs) {
    chains.push_back({"chain:h1:" + v.name, "", v.name, 1});
    chains.push_back({"chain:hN:" + v.name, "", v.name, 2});
    for (const auto& g : gnames) {
      chains.push_back({"chain:seq:" + g + ":" + v.name, g, v.name, 3});
      chains.push_back({"chain:par1:" + g + ":" + v.name, g, v.name, 4});
      chains.push_back({"chain:parN:" + g + ":" + v.name, g, v.name, 5});
    }
  }
  w.srv_init(L);
  std::vector<std::pair<std::string, std::string>> sg;   // (server variant, GEMM): the server fits beside the GEMM on every SM
  for (int sv = 0; sv < 2; ++sv) {
    const std::string sn = "srv" + std::to_string(World::SRV_MINB[sv]);
    chains.push_back({"chain:" + sn, "", sn, 7});
    for (const auto& g : gnames) {
      if (g == "lt") continue;
      bool fits = true;
      for (const auto& s : SHAPES)
        for (const auto& e : w.ext[s.name]) if (e.name == g) fits &= w.srv_fits(e, sv);
      printf("{\"server_fits\": \"%s\", \"server\": \"%s\", \"fits\": %s, \"server_regs\": %d}\n", g.c_str(), sn.c_str(),
             fits ? "true" : "false", w.srv_regs[sv]);
      if (fits) {
        sg.push_back({sn, g});
        chains.push_back({"chain:" + sn + ":" + g, g, sn, 6});
      }
    }
  }
  std::map<std::string, std::vector<GemmArm>> gas;
  for (const auto& s : SHAPES) gas[s.name] = gemm_arms(w, s);
  auto garm = [&](const std::string& shape, const std::string& g) -> const GemmArm& {
    for (const auto& a : gas[shape]) if (a.name == g) return a;
    fprintf(stderr, "no gemm %s for %s\n", g.c_str(), shape.c_str());
    exit(5);
  };
  cudaEvent_t c0, c1;
  CK(cudaEventCreate(&c0)); CK(cudaEventCreate(&c1));
  std::map<std::string, std::vector<double>> cms;
  std::map<std::string, std::vector<unsigned>> cmhz;
  auto chain = [&](const Chain& c) {
    w.nev = 0;
    w.join_all();
    CK(cudaEventRecord(c0, w.A));
    const Variant* v = nullptr;
    for (const auto& x : vs) if (x.name == c.var) v = &x;
    u32 ep = 0;
    if (c.mode >= 6) {
      ep = c.mode == 6 ? ++w.epoch : 0;
      w.srv_launch(c.var == "srv4" ? 0 : 1, ep, w.B);
    }
    int k = 0;   // the chain's call index
    for (int l = 0; l < L; ++l) {
      auto hash = [&](const std::string& t) {
        if (c.mode == 6) w.sig(World::job(l, t), ep);
        if (!v) return;
        const cudaStream_t st = c.mode == 3 ? w.A : (c.mode == 1 || c.mode == 4) ? w.B : w.P[k % NB];
        ++k;
        if (c.mode >= 4) { cudaEvent_t x = w.e(); CK(cudaEventRecord(x, w.A)); CK(cudaStreamWaitEvent(st, x, 0)); }
        w.hs[l][t]->run(*v, reinterpret_cast<const uint8_t*>(w.T[l][t]), st);
      };
      for (const auto& s : SHAPES) {
        if (c.mode == 0 || (c.mode >= 3 && c.mode != 7)) {
          garm(s.name, c.gemm).run(w.X[s.name], w.W[s.name], s.residual ? w.R[s.name] : nullptr, w.out[l & 1], w.A);
          if (s.name == "gate_up") silu_mul<<<2048, 256, 0, w.A>>>(w.out[l & 1], w.T[l]["act"], M, I);
        }
        if (s.name == "qkv") hash("qkv");
        if (s.name == "o") { hash("attn"); hash("res"); }
        if (s.name == "gate_up") hash("act");
        if (s.name == "down") hash("res2");
      }
    }
    w.join_all();
    CK(cudaEventRecord(c1, w.A));
    CK(cudaEventSynchronize(c1));
    float ms;
    CK(cudaEventElapsedTime(&ms, c0, c1));
    return (double)ms / L;
  };
  for (const auto& c : chains) chain(c);
  // the server's gates: alone and beside every GEMM it runs beside, its levels against Committer's before any is timed
  {
    std::vector<Chain> keep;
    for (const auto& c : chains) {
      if (c.mode < 6) { keep.push_back(c); continue; }
      srv_poison(w);
      chain(c);
      World::sync("server " + c.name);
      const unsigned long long nb = srv_check(w);
      printf("{\"server_gate\": \"%s\", \"jobs\": %zu, \"bad_words\": %llu, \"ok\": %s}\n", c.name.c_str(), w.jobs.size(), nb,
             nb ? "false" : "true");
      if (!nb) keep.push_back(c);
    }
    chains.swap(keep);
    fflush(stdout);
  }
  for (int r = 0; r < reps; ++r)
    for (size_t j = 0; j < chains.size(); ++j) {
      const Chain& c = chains[(j + r) % chains.size()];
      cms[c.name].push_back(chain(c));
      cmhz[c.name].push_back(nv.sm());
    }
  for (const auto& c : chains) {
    printf("{\"chain\": \"%s\", \"layers\": %d, \"median_ms_per_layer\": %.5f, \"ms\": [", c.name.c_str(), L, median(cms[c.name]));
    for (size_t i = 0; i < cms[c.name].size(); ++i) printf("%s%.5f", i ? ", " : "", cms[c.name][i]);
    printf("], \"sm_mhz\": [");
    for (size_t i = 0; i < cmhz[c.name].size(); ++i) printf("%s%u", i ? ", " : "", cmhz[c.name][i]);
    printf("]}\n");
  }
  // against cuBLASLt alone plus the separate pass at its throughput (hN); h1 is the one-stream pass, for reference
  const double clt = median(cms["chain:lt"]);
  for (const auto& v : vs) {
    const double h1 = median(cms["chain:h1:" + v.name]), hn = median(cms["chain:hN:" + v.name]);
    for (const auto& g : gnames) {
      const double cg = median(cms["chain:" + g]), seq = median(cms["chain:seq:" + g + ":" + v.name]),
                   p1 = median(cms["chain:par1:" + g + ":" + v.name]), pn = median(cms["chain:parN:" + g + ":" + v.name]);
      const double par = std::min(p1, pn);
      printf("{\"chain_summary\": \"%s\", \"variant\": \"%s\", \"lt_ms\": %.5f, \"gemm_ms\": %.5f, \"h1_ms\": %.5f, \"hN_ms\": %.5f, "
             "\"seq_ms\": %.5f, \"par1_ms\": %.5f, \"parN_ms\": %.5f, \"par_ms\": %.5f, \"hidden_vs_own_gemm\": %.4f, \"hidden_vs_lt\": %.4f, "
             "\"net_vs_lt_plus_hash\": %.4f, \"exposed_hash_ms\": %.5f, \"bytes_per_layer\": %llu}\n",
             g.c_str(), v.name.c_str(), clt, cg, h1, hn, seq, p1, pn, par, (cg + hn - par) / hn, (clt + hn - par) / hn, par / (clt + hn),
             par - clt, (unsigned long long)(M * (QKV + 9216 + H + I + H) * 2ull));
    }
  }
  // the server: against cuBLASLt alone plus the better separate pass (hN of the best variant, or the server alone)
  {
    double hb = 1e30;
    std::string hbn;
    for (const char* sn : {"srv4", "srv6"})
      if (cms.count(std::string("chain:") + sn) && median(cms[std::string("chain:") + sn]) < hb) { hb = median(cms[std::string("chain:") + sn]); hbn = sn; }
    for (const auto& v : vs) if (median(cms["chain:hN:" + v.name]) < hb) { hb = median(cms["chain:hN:" + v.name]); hbn = "hN:" + v.name; }
    for (const auto& [sn, g] : sg) {
      if (!cms.count("chain:" + sn + ":" + g) || !cms.count("chain:" + sn)) continue;
      const double cg = median(cms["chain:" + g]), sa = median(cms["chain:" + sn]), sb = median(cms["chain:" + sn + ":" + g]);
      printf("{\"srv_summary\": \"%s\", \"server\": \"%s\", \"lt_ms\": %.5f, \"gemm_ms\": %.5f, \"srv_alone_ms\": %.5f, \"sep_ms\": %.5f, \"sep\": \"%s\", "
             "\"par_ms\": %.5f, \"hidden_vs_own_gemm\": %.4f, \"hidden_vs_lt\": %.4f, \"net_vs_lt_plus_hash\": %.4f, \"exposed_hash_ms\": %.5f, "
             "\"bytes_per_layer\": %llu}\n",
             g.c_str(), sn.c_str(), clt, cg, sa, hb, hbn.c_str(), sb, (cg + sa - sb) / sa, (clt + hb - sb) / hb, sb / (clt + hb), sb - clt,
             (unsigned long long)(M * (QKV + 9216 + H + I + H) * 2ull));
    }
  }
  // --- the non-GEMM stand-in: a layer's memory-bound kernels (3 fused_add_rms_norm, 1 silu(g) u), R layers of them on A,
  // beside the chain's separate pass (hN: every call of the L layers on its own stream); R is set so the stand-in alone
  // lasts about as long as the pass alone. Overlap efficiency (E + hN - par) / min(E, hN): 1 is full overlap, 0 serial.
  if (!nsys) {
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    auto elt = [&](int r_) {
      for (int r = 0; r < r_; ++r) {
        for (int j = 0; j < 3; ++j) rms_add<<<M, H / 8, 0, w.A>>>(w.X["o"], w.R["o"], w.W["o"], w.out[1], H);
        silu_mul8<<<8 * sms, 256, 0, w.A>>>(w.out[0], w.out[1], M, I);
      }
    };
    auto hn = [&](const Variant& v) {
      int k = 0;
      for (int l = 0; l < L; ++l)
        for (const char* t : {"qkv", "attn", "res", "act", "res2"}) w.hs[l][t]->run(v, reinterpret_cast<const uint8_t*>(w.T[l][t]), w.P[k++ % NB]);
    };
    auto timed = [&](const std::function<void()>& f) {
      w.nev = 0;
      w.join_all();
      CK(cudaEventRecord(c0, w.A));
      f();
      w.join_all();
      CK(cudaEventRecord(c1, w.A));
      CK(cudaEventSynchronize(c1));
      float ms;
      CK(cudaEventElapsedTime(&ms, c0, c1));
      return (double)ms;
    };
    for (int i = 0; i < 3; ++i) timed([&] { elt(1); });
    std::vector<double> e1;
    for (int i = 0; i < 9; ++i) e1.push_back(timed([&] { elt(1); }));
    const double e1m = median(e1);
    for (const auto& v : vs) {
      const double hnm = median(cms["chain:hN:" + v.name]) * L;
      const int R = std::max(1, (int)std::lround(hnm / e1m));
      std::vector<double> te, th, tp;
      for (int i = 0; i < 2; ++i) { timed([&] { elt(R); }); timed([&] { hn(v); }); timed([&] { elt(R); hn(v); }); }
      for (int r = 0; r < reps; ++r) {
        const int o = r % 3;
        for (int j = 0; j < 3; ++j) {
          const int a = (j + o) % 3;
          if (a == 0) te.push_back(timed([&] { elt(R); }));
          if (a == 1) th.push_back(timed([&] { hn(v); }));
          if (a == 2) tp.push_back(timed([&] { hn(v); elt(R); }));
        }
      }
      const double E = median(te), Hn = median(th), Pp = median(tp);
      printf("{\"nongemm\": \"%s\", \"layer_ms\": %.5f, \"R\": %d, \"elt_ms\": %.5f, \"hN_ms\": %.5f, \"par_ms\": %.5f, "
             "\"overlap_eff\": %.4f, \"hidden_of_hN\": %.4f, \"elt_ms_all\": [", v.name.c_str(), e1m, R, E, Hn, Pp,
             (E + Hn - Pp) / std::min(E, Hn), (E + Hn - Pp) / Hn);
      for (size_t i = 0; i < te.size(); ++i) printf("%s%.5f", i ? ", " : "", te[i]);
      printf("], \"par_ms_all\": [");
      for (size_t i = 0; i < tp.size(); ++i) printf("%s%.5f", i ? ", " : "", tp[i]);
      printf("]}\n");
      fflush(stdout);
    }
  }
  fflush(stdout);
  return 0;
}
}  // namespace ov

int main(int argc, char** argv) {
  if (argc >= 4 && !strcmp(argv[1], "dump")) return ov::dump(argv[2], argv[3]);
  if (argc >= 4 && !strcmp(argv[1], "bench")) return ov::bench(argv[2], atoi(argv[3]), false);
  if (argc >= 3 && !strcmp(argv[1], "nsys")) return ov::bench(argv[2], 1, true);
  fprintf(stderr, "overlap dump PDIR OUT | bench PDIR REPS | nsys PDIR\n");
  return 2;
}
