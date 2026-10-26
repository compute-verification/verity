// The engine's kernels (rows.cuh), launched on the caller's stream; built by engine.device_lib with nvcc. Every launch
// function returns 0, or the CUDA error of its launch.
//
// The row kernel comes in every compression variant the bench sweeps (VARIANTS: sha512x.cuh's V and ROLL, and "pouw",
// commit_gpu's sha::compress, the one commit_slots runs); the other kernels use SX_V and SX_ROLL.
#include <cuda_runtime.h>

#include <cstdio>

#include "rows.cuh"
#include "../../pouw/commit_gpu/sha512.cuh"

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

using namespace sxr;

namespace {

template <int V, int ROLL>
struct Sx {
  static __host__ __device__ __forceinline__ void run(u64* st, u64* w) { sx::compress<V, ROLL>(st, w); }
};
struct Pouw {
  static __host__ __device__ __forceinline__ void run(u64* st, u64* w) { sha::compress(st, w); }
};
using D = Sx<SX_V, SX_ROLL>;

unsigned blocks(u64 n) { return (unsigned)((n + SXR_THREADS - 1) / SXR_THREADS); }
int launched() { return (int)cudaGetLastError(); }

__global__ void __launch_bounds__(SXR_THREADS) k_row_setup(RowSpec* rows, u32 G) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < G) row_setup<D>(rows[t]);
}

// a thread a (row g, slot s), t = g S + s: a warp's rows are one row of the table in consecutive slots
template <class C>
__global__ void __launch_bounds__(SXR_THREADS, SXR_MINB) k_rows(const RowSpec* __restrict__ rows, u32 G, u32 S,
                                                                u64* __restrict__ x) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)G * S) return;
  const u32 g = (u32)(t / S), s = (u32)(t % S);
  u64 d[8];
  row_digest<C>(rows[g], s, d);
#pragma unroll
  for (int i = 0; i < 8; ++i) x[8 * t + i] = d[i];
}

__global__ void __launch_bounds__(SXR_THREADS) k_leaf_setup(LeafRow* rows, u32 G, const u64* dids, const uint8_t* schema,
                                                             u32 schema_len) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < G) leaf_setup<D>(rows[t], dids, schema, schema_len);
}

__global__ void __launch_bounds__(SXR_THREADS) k_leaves(const __grid_constant__ LeafArgs a) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (u64)a.G * a.S) leaf<D>(a, (u32)(t / a.S), (u32)(t % a.S));
}

__global__ void __launch_bounds__(SXR_THREADS) k_frame_leaves(const u64* dids, u32 D_, const u64* values, u32 T, u32 n,
                                                               u64 rank0, const uint8_t* schema, u32 schema_len, u64* out) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)T * n) return;
  const u64 tree = t / n, j = t % n;
  frame_leaf<D>(dids + 8 * (D_ == 1 ? 0 : tree), rank0 + j, schema, schema_len, values + 8 * t, out + 8 * t);
}

__global__ void __launch_bounds__(SXR_THREADS) k_node_mids(const u64* dids, u32 D_, u32 level, u64* mids) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < D_) node_mid<D>(dids + 8 * t, level, mids + 8 * t);
}

__global__ void __launch_bounds__(SXR_THREADS) k_level(const u64* mids, u32 D_, const u64* in, u32 T, u32 w, u64* out) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)T * w) return;
  const u64 tree = t / w, i = t % w;
  const u64* l = in + 8 * (2 * tree * w + 2 * i);
  node<D>(mids + 8 * (D_ == 1 ? 0 : tree), i, l, l + 8, out + 8 * t);
}

__global__ void __launch_bounds__(SXR_THREADS) k_pads(const u64* dids, u32 D_, u32 T, u64 first, u32 n, u64* out) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (u64)T * n) return;
  const u64 tree = t / n, j = t % n;
  pad<D>(dids + 8 * (D_ == 1 ? 0 : tree), first + j, out + 8 * t);
}

using RowKernel = void (*)(const RowSpec*, u32, u32, u64*);
struct Variant {
  const char* name;
  RowKernel k;
};
// V's digits are the modes of S0, S1, s0 and s1 (sha512x.cuh); a trailing "s" is ROLL = 0
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

int sxr_rows(void* stream, int variant, const RowSpec* rows, uint32_t G, uint32_t S, uint64_t* x) {
  if (variant < 0 || variant >= NVAR) return -1;
  const u64 n = (u64)G * S;
  if (n) VARIANTS[variant].k<<<blocks(n), SXR_THREADS, 0, (cudaStream_t)stream>>>(rows, G, S, x);
  return launched();
}

int sxr_leaf_setup(void* stream, LeafRow* rows, uint32_t G, const uint64_t* dids, const uint8_t* schema, uint32_t schema_len) {
  if (G) k_leaf_setup<<<blocks(G), SXR_THREADS, 0, (cudaStream_t)stream>>>(rows, G, dids, schema, schema_len);
  return launched();
}

int sxr_leaves(void* stream, const LeafArgs* a) {
  const u64 n = (u64)a->G * a->S;
  if (n) k_leaves<<<blocks(n), SXR_THREADS, 0, (cudaStream_t)stream>>>(*a);
  return launched();
}

int sxr_frame_leaves(void* stream, const uint64_t* dids, uint32_t D_, const uint64_t* values, uint32_t T, uint32_t n,
                     uint64_t rank0, const uint8_t* schema, uint32_t schema_len, uint64_t* out) {
  const u64 m = (u64)T * n;
  if (m) k_frame_leaves<<<blocks(m), SXR_THREADS, 0, (cudaStream_t)stream>>>(dids, D_, values, T, n, rank0, schema, schema_len, out);
  return launched();
}

int sxr_node_mids(void* stream, const uint64_t* dids, uint32_t D_, uint32_t level, uint64_t* mids) {
  if (D_) k_node_mids<<<blocks(D_), SXR_THREADS, 0, (cudaStream_t)stream>>>(dids, D_, level, mids);
  return launched();
}

int sxr_level(void* stream, const uint64_t* mids, uint32_t D_, const uint64_t* in, uint32_t T, uint32_t w, uint64_t* out) {
  const u64 m = (u64)T * w;
  if (m) k_level<<<blocks(m), SXR_THREADS, 0, (cudaStream_t)stream>>>(mids, D_, in, T, w, out);
  return launched();
}

int sxr_pads(void* stream, const uint64_t* dids, uint32_t D_, uint32_t T, uint64_t first, uint32_t n, uint64_t* out) {
  const u64 m = (u64)T * n;
  if (m) k_pads<<<blocks(m), SXR_THREADS, 0, (cudaStream_t)stream>>>(dids, D_, T, first, n, out);
  return launched();
}

}
