// The engine's kernels (rows.cuh), launched on the caller's stream; built by engine.device_lib with nvcc. Every launch
// function returns 0, or the CUDA error of its launch. sxr_graph_* rekey the leaf launches of a graph captured from them.
//
// The row kernel comes in every compression variant the bench sweeps (VARIANTS: sx::compress's V and ROLL, and "pouw",
// sha::compress, commit_gpu's, the one commit_slots runs); the other kernels use SX_V and SX_ROLL.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "rows.cuh"

#ifndef SX_V
#define SX_V 0
#endif
#ifndef SX_ROLL
#define SX_ROLL 1
#endif
#ifndef SXR_THREADS
#define SXR_THREADS 256
#endif
#ifndef SXR_MINB
#define SXR_MINB 2
#endif
#ifndef SXR_LEAF_MINB
#define SXR_LEAF_MINB 1
#endif

using namespace sxr;

namespace {

template <int V, int ROLL>
struct Sx {
  static __host__ __device__ __forceinline__ void run(u64* st, u64* w) { sx::compress<V, ROLL>(st, w); }
  static __host__ __device__ __forceinline__ void kw(u64* st, const u64* kw) { sx::compress_kw<V, ROLL>(st, kw); }
};
struct Pouw {
  static __host__ __device__ __forceinline__ void run(u64* st, u64* w) { sha::compress(st, w); }
  static __host__ __device__ __forceinline__ void kw(u64* st, const u64* kw) {   // the block is K + W's first 16, less K
    u64 w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = kw[i] - sx::kconst()[i];
    sha::compress(st, w);
  }
};
using D = Sx<SX_V, SX_ROLL>;

unsigned blocks(u64 n) { return (unsigned)((n + SXR_THREADS - 1) / SXR_THREADS); }
int launched() { return (int)cudaGetLastError(); }

__global__ void __launch_bounds__(SXR_THREADS) k_row_setup(RowSpec* rows, u32 G) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < G) row_setup<D>(rows[t]);
}

// a thread a (row g, slot s), t = j S + s with g = order[j]: a warp's rows are one row of the table in consecutive slots, and
// CTAs start in `order` (the longest rows first, so the last wave is short rows); x stays at (g, s)
template <class C>
__global__ void __launch_bounds__(SXR_THREADS, SXR_MINB) k_rows(const RowSpec* __restrict__ rows, const u32* __restrict__ order,
                                                                u32 G, u32 S, u64* __restrict__ x) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)G * S) return;
  const u32 j = (u32)(t / S), s = (u32)(t % S), g = order ? __ldg(order + j) : j;
  u64 d[8];
  row_digest<C>(rows[g], s, d);
  u64* o = x + 8 * ((u64)g * S + s);
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = d[i];
}

__global__ void __launch_bounds__(SXR_THREADS) k_leaf_setup(LeafRow* rows, u32 G, const u64* dids, const uint8_t* schema,
                                                             u32 schema_len) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < G) leaf_setup<D>(rows[t], dids, schema, schema_len);
}

// T in shared memory (a CTA's 64 KB, loaded once), then each thread's salt words (48 KB a CTA; word w of thread j at
// w SXR_THREADS + j, so a warp's loads of a word are in 32 banks); the leaves grid-stride over resident CTAs; MINB CTAs
// an SM caps the registers (1: as many as it wants, 2: 128)
template <int MINB>
__global__ void __launch_bounds__(SXR_THREADS, MINB) k_leaves(const __grid_constant__ LeafArgs a) {
  extern __shared__ uint4 smem[];
  const uint4* Tg = reinterpret_cast<const uint4*>(a.T);
  for (int i = threadIdx.x; i < T_BYTES / 16; i += blockDim.x) smem[i] = __ldg(Tg + i);
  __syncthreads();
  const uint8_t* T = reinterpret_cast<const uint8_t*>(smem);
  u32* y = reinterpret_cast<u32*>(smem + T_BYTES / 16) + threadIdx.x;
  const u64 n = (u64)a.G * a.S, step = (u64)gridDim.x * blockDim.x;
#pragma unroll 1
  for (u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += step)
    leaf<D>(a, T, y, SXR_THREADS, threadIdx.x & 7, (u32)(t / a.S), (u32)(t % a.S));
}
constexpr int LEAF_SMEM = T_BYTES + 4 * SALT_U32 * SXR_THREADS;

template <int MINB>
int launch_leaves(cudaStream_t stream, const LeafArgs& a, u64 n) {
  static int per_sm_[64] = {}, sms_[64] = {};   // a device's resident leaf CTAs an SM and its SMs, once its
                                                // shared-memory limit is raised
  int dev = 0, rc;
  if ((rc = cudaGetDevice(&dev))) return rc;
  if (dev < 0 || dev >= 64) return (int)cudaErrorInvalidDevice;
  if (!per_sm_[dev]) {
    int sms = 0, per_sm = 0;
    if ((rc = cudaFuncSetAttribute(k_leaves<MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, LEAF_SMEM)) ||
        (rc = cudaFuncSetAttribute(k_leaves<MINB>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   cudaSharedmemCarveoutMaxShared)) ||
        (rc = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)) ||
        (rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_leaves<MINB>, SXR_THREADS, LEAF_SMEM)))
      return rc;
    sms_[dev] = sms > 0 ? sms : 1;
    per_sm_[dev] = per_sm > 0 ? per_sm : 1;
  }
  const u64 sms = a.sms ? std::min<u64>(a.sms, (u64)sms_[dev]) : (u64)sms_[dev];
  const unsigned grid = (unsigned)std::min<u64>((n + SXR_THREADS - 1) / SXR_THREADS, (u64)per_sm_[dev] * sms);
  k_leaves<MINB><<<grid, SXR_THREADS, LEAF_SMEM, stream>>>(a);
  return (int)cudaGetLastError();
}

__global__ void __launch_bounds__(SXR_THREADS) k_frame_leaves(const u64* dids, u32 D_, const u64* values, u32 T, u32 n,
                                                               u64 rank0, const uint8_t* schema, u32 schema_len, u64* out) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)T * n) return;
  const u64 tree = t / n, j = t % n;
  frame_leaf<D>(dids + 8 * (D_ == 1 ? 0 : tree), rank0 + j, schema, schema_len, values + 8 * t, out + 8 * t);
}

__global__ void __launch_bounds__(SXR_THREADS) k_frame_setup(LeafRow* rows, u32 n, u64 rank0, const u64* did,
                                                              const uint8_t* schema, u32 schema_len) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) frame_setup<D>(rows[t], rank0 + t, did, schema, schema_len);
}

// out[s][units[i]] = the slot-tree leaf of values[i][s] under frames[units[i]], for n units and S slots
__global__ void __launch_bounds__(SXR_THREADS) k_frame_units(const LeafRow* frames, const u64* units, const u64* values, u32 n,
                                                              u32 S, u32 U, u64* out) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)n * S) return;
  const u64 d = units[t / S], s = t % S;
  u64 v[8];
  for (int i = 0; i < 8; ++i) v[i] = values[8 * t + i];
  frame_tail<D>(frames[d], v, v);
  for (int i = 0; i < 8; ++i) out[8 * (s * U + d) + i] = v[i];
}

__global__ void __launch_bounds__(SXR_THREADS) k_node_mids(const u64* dids, u32 D_, u32 level, u64* mids) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < D_) node_mid<D>(dids + 8 * t, level, mids + 8 * t);
}

// tree t's children are pre[t] (if pre) and then in[t][0 ..], its trees in_stride words apart
__global__ void __launch_bounds__(SXR_THREADS) k_level(const u64* mids, u32 D_, const u64* pre, u64 pre_stride, const u64* in,
                                                        u64 in_stride, u32 T, u32 w, u64 first, u64* out) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)T * w) return;
  const u64 tree = t / w, i = t % w, j = 2 * i;
  const u64* kids = in + tree * in_stride;
  const u64* l = !pre ? kids + 8 * j : j ? kids + 8 * (j - 1) : pre + tree * pre_stride;
  const u64* r = kids + 8 * (pre ? j : j + 1);
  node<D>(mids + 8 * (D_ == 1 ? 0 : tree), first + i, l, r, out + 8 * t);
}

__global__ void __launch_bounds__(SXR_THREADS) k_pads(const u64* dids, u32 D_, u32 T, u64 first, u32 n, u64* out) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)T * n) return;
  const u64 tree = t / n, j = t % n;
  pad<D>(dids + 8 * (D_ == 1 ? 0 : tree), first + j, out + 8 * t);
}

using RowKernel = void (*)(const RowSpec*, const u32*, u32, u32, u64*);
struct Variant {
  const char* name;
  RowKernel k;
};
// V's digits are the modes of S0, S1, s0 and s1 (verity/ml/kernels/sha512.cuh); a trailing "s" is ROLL = 0
#define SXR_V(a, b, c, d) ((a) | (b) << 2 | (c) << 4 | (d) << 6)
const Variant VARIANTS[] = {
    {"pouw", k_rows<Pouw>},
    {"0000", k_rows<Sx<SXR_V(0, 0, 0, 0), 1>>},
    {"2000", k_rows<Sx<SXR_V(2, 0, 0, 0), 1>>},
    {"1100", k_rows<Sx<SXR_V(1, 1, 0, 0), 1>>},
    {"2200", k_rows<Sx<SXR_V(2, 2, 0, 0), 1>>},
    {"2210", k_rows<Sx<SXR_V(2, 2, 1, 0), 1>>},
    {"2220", k_rows<Sx<SXR_V(2, 2, 2, 0), 1>>},
    {"2202", k_rows<Sx<SXR_V(2, 2, 0, 2), 1>>},
    {"2211", k_rows<Sx<SXR_V(2, 2, 1, 1), 1>>},
    {"2221", k_rows<Sx<SXR_V(2, 2, 2, 1), 1>>},
    {"2222", k_rows<Sx<SXR_V(2, 2, 2, 2), 1>>},
    {"2122", k_rows<Sx<SXR_V(2, 1, 2, 2), 1>>},
    {"2112", k_rows<Sx<SXR_V(2, 1, 1, 2), 1>>},
    {"1111", k_rows<Sx<SXR_V(1, 1, 1, 1), 1>>},
    {"1122", k_rows<Sx<SXR_V(1, 1, 2, 2), 1>>},
    {"0003", k_rows<Sx<SXR_V(0, 0, 0, 3), 1>>},
    {"0033", k_rows<Sx<SXR_V(0, 0, 3, 3), 1>>},
    {"1033", k_rows<Sx<SXR_V(1, 0, 3, 3), 1>>},
    {"0333", k_rows<Sx<SXR_V(0, 3, 3, 3), 1>>},
    {"0000s", k_rows<Sx<SXR_V(0, 0, 0, 0), 0>>},
    {"2220s", k_rows<Sx<SXR_V(2, 2, 2, 0), 0>>},
    {"2211s", k_rows<Sx<SXR_V(2, 2, 1, 1), 0>>},
};
#undef SXR_V
constexpr int NVAR = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

}  // namespace

extern "C" {

int sxr_variants() { return NVAR; }
const char* sxr_variant_name(int i) { return i >= 0 && i < NVAR ? VARIANTS[i].name : nullptr; }
const char* sxr_default() {
  static char name[8];
  const int v = SX_V;
  snprintf(name, sizeof name, "%d%d%d%d%s", v & 3, (v >> 2) & 3, (v >> 4) & 3, (v >> 6) & 3, SX_ROLL ? "" : "s");
  return name;
}

void sxr_midstate(const uint8_t* block, uint64_t* out) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) w[i] = sha::be64(block + 8 * i);
  sx::iv(out);
  sx::compress<0, 1>(out, w);
}

int sxr_row_setup(void* stream, RowSpec* rows, uint32_t G) {
  if (G) k_row_setup<<<blocks(G), SXR_THREADS, 0, (cudaStream_t)stream>>>(rows, G);
  return launched();
}

int sxr_rows(void* stream, int variant, const RowSpec* rows, const uint32_t* order, uint32_t G, uint32_t S, uint64_t* x) {
  if (variant < 0 || variant >= NVAR) return -1;
  const u64 n = (u64)G * S;
  if (n) VARIANTS[variant].k<<<blocks(n), SXR_THREADS, 0, (cudaStream_t)stream>>>(rows, order, G, S, x);
  return launched();
}

int sxr_leaf_setup(void* stream, LeafRow* rows, uint32_t G, const uint64_t* dids, const uint8_t* schema, uint32_t schema_len) {
  if (G) k_leaf_setup<<<blocks(G), SXR_THREADS, 0, (cudaStream_t)stream>>>(rows, G, dids, schema, schema_len);
  return launched();
}

int sxr_leaves(void* stream, const LeafArgs* a) {
  const u64 n = (u64)a->G * a->S;
  if (!n) return 0;
  const u32 minb = a->minb ? a->minb : SXR_LEAF_MINB;
  if (minb == 1) return launch_leaves<1>((cudaStream_t)stream, *a, n);
  if (minb == 2) return launch_leaves<2>((cudaStream_t)stream, *a, n);
  return (int)cudaErrorInvalidValue;
}

int sxr_frame_leaves(void* stream, const uint64_t* dids, uint32_t D_, const uint64_t* values, uint32_t T, uint32_t n,
                     uint64_t rank0, const uint8_t* schema, uint32_t schema_len, uint64_t* out) {
  const u64 m = (u64)T * n;
  if (m) k_frame_leaves<<<blocks(m), SXR_THREADS, 0, (cudaStream_t)stream>>>(dids, D_, values, T, n, rank0, schema, schema_len, out);
  return launched();
}

int sxr_frame_setup(void* stream, LeafRow* rows, uint32_t n, uint64_t rank0, const uint64_t* did, const uint8_t* schema,
                    uint32_t schema_len) {
  if (n) k_frame_setup<<<blocks(n), SXR_THREADS, 0, (cudaStream_t)stream>>>(rows, n, rank0, did, schema, schema_len);
  return launched();
}

int sxr_frame_units(void* stream, const LeafRow* frames, const uint64_t* units, const uint64_t* values, uint32_t n, uint32_t S,
                    uint32_t U, uint64_t* out) {
  const u64 m = (u64)n * S;
  if (m) k_frame_units<<<blocks(m), SXR_THREADS, 0, (cudaStream_t)stream>>>(frames, units, values, n, S, U, out);
  return launched();
}

int sxr_node_mids(void* stream, const uint64_t* dids, uint32_t D_, uint32_t level, uint64_t* mids) {
  if (D_) k_node_mids<<<blocks(D_), SXR_THREADS, 0, (cudaStream_t)stream>>>(dids, D_, level, mids);
  return launched();
}

int sxr_level(void* stream, const uint64_t* mids, uint32_t D_, const uint64_t* pre, uint64_t pre_stride, const uint64_t* in,
              uint64_t in_stride, uint32_t T, uint32_t w, uint64_t first, uint64_t* out) {
  const u64 m = (u64)T * w;
  if (m) k_level<<<blocks(m), SXR_THREADS, 0, (cudaStream_t)stream>>>(mids, D_, pre, pre_stride, in, in_stride, T, w, first, out);
  return launched();
}

int sxr_pads(void* stream, const uint64_t* dids, uint32_t D_, uint32_t T, uint64_t first, uint32_t n, uint64_t* out) {
  const u64 m = (u64)T * n;
  if (m) k_pads<<<blocks(m), SXR_THREADS, 0, (cudaStream_t)stream>>>(dids, D_, T, first, n, out);
  return launched();
}

// The leaf kernel's nodes in a captured graph (a cudaGraph_t): up to max of them to out. Returns how many there are, or a
// CUDA error negated. A node another runtime launched (torch's) may not give its parameters here; it is not a leaf node.
int sxr_graph_leaf_nodes(void* graph, void** out, uint32_t max) {
  size_t n = 0;
  int rc = cudaGraphGetNodes((cudaGraph_t)graph, nullptr, &n);
  if (rc) return -rc;
  std::vector<cudaGraphNode_t> nodes(n);
  if (n && (rc = cudaGraphGetNodes((cudaGraph_t)graph, nodes.data(), &n))) return -rc;
  int k = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    if ((rc = cudaGraphNodeGetType(node, &type))) return -rc;
    if (type != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams p;
    if (cudaGraphKernelNodeGetParams(node, &p)) {
      cudaGetLastError();
      continue;
    }
    const bool leaf = p.func == (void*)k_leaves<1> || p.func == (void*)k_leaves<2> ||
                      (p.sharedMemBytes == (unsigned)LEAF_SMEM && p.blockDim.x == SXR_THREADS);
    if (!leaf) continue;
    if ((uint32_t)k < max) out[k] = node;
    ++k;
  }
  return k;
}

// Each of n leaf nodes (sxr_graph_leaf_nodes) in the graph's instantiation (a cudaGraphExec_t) to salt under key (eight
// words) from its next launch on; every other argument stays as captured. The current context is the one the graph was
// captured in. A refusal is returned, and not left for the next launch's cudaGetLastError.
int sxr_graph_rekey(void* exec, void* const* nodes, uint32_t n, const uint32_t* key) {
  for (uint32_t i = 0; i < n; ++i) {
    cudaKernelNodeParams p;
    int rc = cudaGraphKernelNodeGetParams((cudaGraphNode_t)nodes[i], &p);
    if (!rc) {
      LeafArgs a = *static_cast<const LeafArgs*>(p.kernelParams[0]);
      std::memcpy(a.key, key, sizeof a.key);
      void* args[] = {&a};
      p.kernelParams = args;
      p.extra = nullptr;
      rc = cudaGraphExecKernelNodeSetParams((cudaGraphExec_t)exec, (cudaGraphNode_t)nodes[i], &p);
    }
    if (rc) {
      cudaGetLastError();
      return rc;
    }
  }
  return 0;
}

}
