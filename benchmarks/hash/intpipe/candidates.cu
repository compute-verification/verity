// One integer-pipe candidate (hashes.cuh) timed against full SHA-512 (sha512.cuh's compression) rep by rep on one GPU.
// Build one binary a candidate: -DCAND_KIND=0 (SHA-512, the baseline itself), 1 (SHA-512 at -DCAND_R rounds), 2 (SHA-256),
// 3 (BLAKE3, a thread a leaf), 4 (BLAKE3, a thread a chunk), 5 (no hash: hashes.cuh's Xor control); -DSHA_ROLL=0|1 as sha512.cuh (1: the rolled form the
// baseline's README times); -DFETCH=128|256..., the bytes of each leaf stream the loader reads at a time.
//
//   candidates dump DIR OPERAND N     the timed kernels' outputs for check.py: the raw chains at the timed grid
//                                    (DIR/rate-base.bin, rate-cand.bin) and the 4 KiB-leaf tree of OPERAND's N leaves
//                                    (DIR/levels-base.bin, levels-cand.bin; level 0 the leaves' digests, then each level up)
//   candidates bench REPS N[,N...]   JSON lines: REPS reps of the raw rate and of the 4 KiB-leaf tree over each N leaves,
//                                    each rep the baseline and the candidate back to back (order alternating), with SM clocks
//
// Raw rate: each thread chains CHAIN = 256 compressions, its block's first 8 words XOR the last state (commit_gpu's
// sha_rate), and stores its final state. 4 KiB leaves: each leaf hashed with the candidate's standard hash (SHA-512 and
// SHA-256 Merkle-Damgard with padding, BLAKE3's 4 chunks and 3 parents), a thread a leaf (BLAKE3 also a thread a chunk),
// its blocks loaded a warp at a time through shared memory (hashes.cuh's WarpLoader, the same for every hash); then a
// binary tree of node = H(left || right): a launch a level while a level has more than TOP nodes, then one CTA.
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hashes.cuh"

#ifndef CAND_KIND
#define CAND_KIND 0
#endif
#ifndef CAND_R
#define CAND_R 80
#endif

#if CAND_KIND == 0
using Cand = ip::Sha512<0>;
static const std::string CAND_NAME = "sha512";
static const int CAND_LEAF_COMPS = 33, CAND_NODE_COMPS = 2;
#elif CAND_KIND == 1
using Cand = ip::Sha512<CAND_R>;
static const std::string CAND_NAME = "sha512r" + std::to_string(CAND_R);
static const int CAND_LEAF_COMPS = 33, CAND_NODE_COMPS = 2;
#elif CAND_KIND == 2
using Cand = ip::Sha256;
static const std::string CAND_NAME = "sha256";
static const int CAND_LEAF_COMPS = 65, CAND_NODE_COMPS = 2;
#elif CAND_KIND == 3
using Cand = ip::Blake3;
static const std::string CAND_NAME = "blake3";
static const int CAND_LEAF_COMPS = 67, CAND_NODE_COMPS = 1;
#elif CAND_KIND == 4
using Cand = ip::Blake3Chunks;
static const std::string CAND_NAME = "blake3chunks";
static const int CAND_LEAF_COMPS = 67, CAND_NODE_COMPS = 1;
#elif CAND_KIND == 5
using Cand = ip::Xor;
static const std::string CAND_NAME = "xor";
static const int CAND_LEAF_COMPS = 64, CAND_NODE_COMPS = 1;
#endif
using Base = ip::Sha512<0>;
using u64 = uint64_t;
#ifndef LEAF_WARPS
#define LEAF_WARPS 4
#endif
constexpr int CHAIN = 256, RATE_THREADS = 256, LEAF_THREADS = 32 * LEAF_WARPS, NODE_THREADS = 128, TOP = 256;

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
      exit(3);                                                                                   \
    }                                                                                            \
  } while (0)

template <class H>
__device__ __forceinline__ void rate_body(typename H::W* io, int chain) {
  using W = typename H::W;
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  W st[8], w[16];
  H::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = (W)(t * 16 + i);
  for (int r = 0; r < chain; ++r) H::rate_step(st, w, r);
#pragma unroll
  for (int i = 0; i < 8; ++i) io[t * 8 + i] = st[i];
}
extern "C" __global__ void __launch_bounds__(RATE_THREADS) rate_base(Base::W* io, int chain) { rate_body<Base>(io, chain); }
extern "C" __global__ void __launch_bounds__(RATE_THREADS) rate_cand(Cand::W* io, int chain) { rate_body<Cand>(io, chain); }

template <class H>
using Loader = ip::WarpLoader<H::BLOCK, H::STRIDE>;
// the grid is exact: n TPL threads, a multiple of LEAF_THREADS
template <class H>
__device__ __forceinline__ void leaf_body(const uint8_t* src, typename H::W* out) {
  extern __shared__ uint4 smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const Loader<H> ld{src + (t - lane) * H::STRIDE, smem + warp * (Loader<H>::SMEM / 16), lane};
  ld.start();
  H::leaf(ld, t, out);
}
template <class H>
__device__ __forceinline__ void node_body(const typename H::W* in, typename H::W* out, u64 n_out) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_out) return;
  typename H::W d[8];
  H::node(in + 16 * t, in + 16 * t + 8, d);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[t * 8 + i] = d[i];
}
// levels from + 1 .. depth of an n-leaf tree's level buffer, in one CTA
template <class H>
__device__ __forceinline__ void top_body(typename H::W* lv, u64 n, int from, int depth) {
  for (int l = from + 1; l <= depth; ++l) {
    const typename H::W* in = lv + 8 * (2 * n - (2 * n >> (l - 1)));
    typename H::W* out = lv + 8 * (2 * n - (2 * n >> l));
    for (u64 t = threadIdx.x; t < (n >> l); t += blockDim.x) {
      typename H::W d[8];
      H::node(in + 16 * t, in + 16 * t + 8, d);
#pragma unroll
      for (int i = 0; i < 8; ++i) out[t * 8 + i] = d[i];
    }
    __syncthreads();
  }
}
extern "C" __global__ void __launch_bounds__(LEAF_THREADS) leaf_base(const uint8_t* s, Base::W* o) { leaf_body<Base>(s, o); }
extern "C" __global__ void __launch_bounds__(LEAF_THREADS) leaf_cand(const uint8_t* s, Cand::W* o) { leaf_body<Cand>(s, o); }
extern "C" __global__ void __launch_bounds__(NODE_THREADS) node_base(const Base::W* i, Base::W* o, u64 n) { node_body<Base>(i, o, n); }
extern "C" __global__ void __launch_bounds__(NODE_THREADS) node_cand(const Cand::W* i, Cand::W* o, u64 n) { node_body<Cand>(i, o, n); }
extern "C" __global__ void __launch_bounds__(NODE_THREADS) top_base(Base::W* lv, u64 n, int f, int d) { top_body<Base>(lv, n, f, d); }
extern "C" __global__ void __launch_bounds__(NODE_THREADS) top_cand(Cand::W* lv, u64 n, int f, int d) { top_body<Cand>(lv, n, f, d); }

__global__ void fill(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = (uint8_t)x;
  }
}

// The card's own read ceiling for the leaf mode's bytes: every byte once, a warp's loads contiguous (grid-stride uint4).
__global__ void memread(const uint4* __restrict__ p, size_t n16, uint4* out) {
  uint4 a = make_uint4(0, 0, 0, 0);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = __ldg(p + i);
    a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
  }
  out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = a;
}

template <class H>
struct Tree {   // levels of an n-leaf tree (n a power of two) in one buffer, 8 words a digest
  using W = typename H::W;
  u64 n = 0;
  int depth = 0;
  W* lv = nullptr;
  void init(u64 n_) {
    n = n_;
    while ((1ull << depth) < n) ++depth;
    if ((1ull << depth) != n) { fprintf(stderr, "leaves: a power of two\n"); exit(2); }
    CK(cudaMalloc(&lv, (2 * n) * 8 * sizeof(W)));
  }
  W* level(int l) const { return lv + 8 * (2 * n - (2 * n >> l)); }
};
template <class H, class LeafK>
static void leaves(const Tree<H>& T, const uint8_t* src, LeafK lk, cudaStream_t st) {
  const u64 threads = T.n * H::TPL;
  if (threads % LEAF_THREADS) { fprintf(stderr, "leaves x threads a leaf: a multiple of %d\n", LEAF_THREADS); exit(2); }
  constexpr int smem = (LEAF_THREADS / 32) * Loader<H>::SMEM;
  CK(cudaFuncSetAttribute(lk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  lk<<<(unsigned)(threads / LEAF_THREADS), LEAF_THREADS, smem, st>>>(src, T.lv);
}
template <class H, class NodeK, class TopK>
static void nodes(const Tree<H>& T, NodeK nk, TopK tk, cudaStream_t st) {
  int l = 1;
  for (; l <= T.depth && (T.n >> l) > (u64)TOP; ++l) {
    const u64 m = T.n >> l;
    nk<<<(unsigned)((m + NODE_THREADS - 1) / NODE_THREADS), NODE_THREADS, 0, st>>>(T.level(l - 1), T.level(l), m);
  }
  if (l <= T.depth) tk<<<1, NODE_THREADS, 0, st>>>(T.lv, T.n, l - 1, T.depth);
}

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  unsigned sm() {
    unsigned c = 0;
    if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c);
    return c;
  }
};

static void write_file(const std::string& path, const void* dev, size_t bytes) {
  std::vector<uint8_t> h(bytes);
  CK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(h.data(), 1, bytes, f) != bytes) { fprintf(stderr, "write %s\n", path.c_str()); exit(2); }
  fclose(f);
}

static int rate_blocks() {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  return prop.multiProcessorCount * 8;
}

static void meta(FILE* f, const cudaDeviceProp& prop) {
  fprintf(f, "\"candidate\": \"%s\", \"cand_kind\": %d, \"cand_r\": %d, \"sha_roll\": %d, \"gpu\": \"%s\", \"sms\": %d, "
          "\"cc\": %d, \"block_bytes\": %d, \"word_bytes\": %d, \"leaf_comps\": %d, \"node_comps\": %d, \"base_leaf_comps\": 33, "
          "\"base_node_comps\": 2, \"threads_per_leaf\": %d, \"fetch\": %d, \"loader_async\": %d, \"leaf_warps\": %d, \"chain\": %d, \"rate_threads\": %d, \"rate_blocks\": %d",
          CAND_NAME.c_str(), CAND_KIND, CAND_KIND == 1 ? CAND_R : CAND_KIND == 2 ? 64 : CAND_KIND == 5 ? 0 : CAND_KIND >= 3 ? 7 : 80, SHA_ROLL, prop.name,
          prop.multiProcessorCount, prop.major * 10 + prop.minor, Cand::BLOCK, (int)sizeof(Cand::W), CAND_LEAF_COMPS,
          CAND_NODE_COMPS, Cand::TPL, FETCH, LOADER_ASYNC, LEAF_WARPS, CHAIN, RATE_THREADS, prop.multiProcessorCount * 8);
}

static int dump(const std::string& dir, const std::string& operand, u64 n) {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  const int blocks = rate_blocks();
  const u64 threads = (u64)blocks * RATE_THREADS;
  Base::W* rb; Cand::W* rc;
  CK(cudaMalloc(&rb, threads * 8 * sizeof(Base::W)));
  CK(cudaMalloc(&rc, threads * 8 * sizeof(Cand::W)));
  rate_base<<<blocks, RATE_THREADS>>>(rb, CHAIN);
  rate_cand<<<blocks, RATE_THREADS>>>(rc, CHAIN);
  CK(cudaDeviceSynchronize());
  write_file(dir + "/rate-base.bin", rb, threads * 8 * sizeof(Base::W));
  write_file(dir + "/rate-cand.bin", rc, threads * 8 * sizeof(Cand::W));
  std::vector<uint8_t> h(n * ip::LEAF);
  FILE* f = fopen(operand.c_str(), "rb");
  if (!f || fread(h.data(), 1, h.size(), f) != h.size()) { fprintf(stderr, "read %s\n", operand.c_str()); return 2; }
  fclose(f);
  uint8_t* src;
  CK(cudaMalloc(&src, h.size()));
  CK(cudaMemcpy(src, h.data(), h.size(), cudaMemcpyHostToDevice));
  Tree<Base> tb; Tree<Cand> tc;
  tb.init(n); tc.init(n);
  leaves(tb, src, leaf_base, 0); nodes(tb, node_base, top_base, 0);
  leaves(tc, src, leaf_cand, 0); nodes(tc, node_cand, top_cand, 0);
  CK(cudaDeviceSynchronize());
  write_file(dir + "/levels-base.bin", tb.lv, (2 * n - 1) * 8 * sizeof(Base::W));
  write_file(dir + "/levels-cand.bin", tc.lv, (2 * n - 1) * 8 * sizeof(Cand::W));
  FILE* m = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(m, "{");
  meta(m, prop);
  fprintf(m, ", \"leaves\": %llu}\n", (unsigned long long)n);
  fclose(m);
  return 0;
}

static int bench(int reps, const std::vector<u64>& sizes) {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  const int blocks = rate_blocks();
  const u64 threads = (u64)blocks * RATE_THREADS;
  Base::W* rb; Cand::W* rc;
  CK(cudaMalloc(&rb, threads * 8 * sizeof(Base::W)));
  CK(cudaMalloc(&rc, threads * 8 * sizeof(Cand::W)));
  const u64 nmax = *std::max_element(sizes.begin(), sizes.end());
  uint8_t* src;
  CK(cudaMalloc(&src, nmax * ip::LEAF));
  fill<<<1024, 256>>>(src, nmax * ip::LEAF, 7u);
  cudaEvent_t e[3];
  for (auto& x : e) CK(cudaEventCreate(&x));
  Nv nv;
  auto ms = [&](cudaEvent_t a, cudaEvent_t b) { float t; CK(cudaEventElapsedTime(&t, a, b)); return (double)t; };
  auto raw_base = [&]() { rate_base<<<blocks, RATE_THREADS>>>(rb, CHAIN); };
  auto raw_cand = [&]() { rate_cand<<<blocks, RATE_THREADS>>>(rc, CHAIN); };
  auto timed = [&](auto launch) {
    CK(cudaEventRecord(e[0]));
    launch();
    CK(cudaEventRecord(e[1]));
    CK(cudaEventSynchronize(e[1]));
    return std::make_pair(ms(e[0], e[1]), (double)nv.sm());
  };
  auto tree_timed = [&](auto lf, auto nd) {
    CK(cudaEventRecord(e[0]));
    lf();
    CK(cudaEventRecord(e[1]));
    nd();
    CK(cudaEventRecord(e[2]));
    CK(cudaEventSynchronize(e[2]));
    return std::make_tuple(ms(e[0], e[2]), ms(e[0], e[1]), (double)nv.sm());
  };
  for (int w = 0; w < 2; ++w) { timed(raw_base); timed(raw_cand); }
  CK(cudaGetLastError());
  const double raw_comps = (double)threads * CHAIN;
  for (int r = 0; r < reps; ++r) {
    std::pair<double, double> b, c;
    if (r % 2 == 0) { b = timed(raw_base); c = timed(raw_cand); }
    else { c = timed(raw_cand); b = timed(raw_base); }
    printf("{");
    meta(stdout, prop);
    printf(", \"mode\": \"raw\", \"rep\": %d, \"ms\": %.5f, \"compressions\": %.0f, \"gcomp_per_s\": %.4f, \"gb_per_s\": %.3f, "
           "\"mhz\": %.0f, \"base_ms\": %.5f, \"base_gcomp_per_s\": %.4f, \"base_gb_per_s\": %.3f, \"base_mhz\": %.0f}\n",
           r, c.first, raw_comps, raw_comps / c.first * 1e-6, raw_comps * Cand::BLOCK / c.first * 1e-6, c.second, b.first,
           raw_comps / b.first * 1e-6, raw_comps * 128 / b.first * 1e-6, b.second);
  }
  for (u64 n : sizes) {
    Tree<Base> tb; Tree<Cand> tc;
    tb.init(n); tc.init(n);
    auto lf_base = [&]() { leaves(tb, src, leaf_base, 0); };
    auto nd_base = [&]() { nodes(tb, node_base, top_base, 0); };
    auto lf_cand = [&]() { leaves(tc, src, leaf_cand, 0); };
    auto nd_cand = [&]() { nodes(tc, node_cand, top_cand, 0); };
    for (int w = 0; w < 2; ++w) { tree_timed(lf_base, nd_base); tree_timed(lf_cand, nd_cand); }
    CK(cudaGetLastError());
    const double bytes = (double)n * ip::LEAF;
    const double cc = (double)n * CAND_LEAF_COMPS + (double)(n - 1) * CAND_NODE_COMPS, bc = (double)n * 33 + (double)(n - 1) * 2;
    for (int r = 0; r < reps; ++r) {
      std::tuple<double, double, double> b, c;
      if (r % 2 == 0) { b = tree_timed(lf_base, nd_base); c = tree_timed(lf_cand, nd_cand); }
      else { c = tree_timed(lf_cand, nd_cand); b = tree_timed(lf_base, nd_base); }
      printf("{");
      meta(stdout, prop);
      printf(", \"mode\": \"leaf4k\", \"rep\": %d, \"leaves\": %llu, \"operand_bytes\": %.0f, \"ms\": %.5f, \"leaf_ms\": %.5f, "
             "\"compressions\": %.0f, \"gcomp_per_s\": %.4f, \"gb_per_s\": %.3f, \"leaf_gb_per_s\": %.3f, \"mhz\": %.0f, "
             "\"base_ms\": %.5f, \"base_leaf_ms\": %.5f, \"base_gcomp_per_s\": %.4f, \"base_gb_per_s\": %.3f, "
             "\"base_leaf_gb_per_s\": %.3f, \"base_mhz\": %.0f}\n",
             r, (unsigned long long)n, bytes, std::get<0>(c), std::get<1>(c), cc, cc / std::get<0>(c) * 1e-6,
             bytes / std::get<0>(c) * 1e-6, bytes / std::get<1>(c) * 1e-6, std::get<2>(c), std::get<0>(b), std::get<1>(b),
             bc / std::get<0>(b) * 1e-6, bytes / std::get<0>(b) * 1e-6, bytes / std::get<1>(b) * 1e-6, std::get<2>(b));
    }
    CK(cudaFree(tb.lv)); CK(cudaFree(tc.lv));
    const int mb = prop.multiProcessorCount * 8;
    uint4* mo;
    CK(cudaMalloc(&mo, (size_t)mb * 256 * sizeof(uint4)));
    auto mr = [&]() { memread<<<mb, 256>>>(reinterpret_cast<const uint4*>(src), n * ip::LEAF / 16, mo); };
    for (int w = 0; w < 2; ++w) timed(mr);
    for (int r = 0; r < reps; ++r) {
      const auto m = timed(mr);
      printf("{");
      meta(stdout, prop);
      printf(", \"mode\": \"memread\", \"rep\": %d, \"leaves\": %llu, \"ms\": %.5f, \"gb_per_s\": %.3f, \"mhz\": %.0f}\n", r,
             (unsigned long long)n, m.first, bytes / m.first * 1e-6, m.second);
    }
    CK(cudaFree(mo));
  }
  fflush(stdout);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 5 && !strcmp(argv[1], "dump")) return dump(argv[2], argv[3], strtoull(argv[4], nullptr, 10));
  if (argc >= 4 && !strcmp(argv[1], "bench")) {
    std::vector<u64> sizes;
    for (char* p = argv[3]; *p;) { sizes.push_back(strtoull(p, &p, 10)); if (*p == ',') ++p; }
    return bench(atoi(argv[2]), sizes);
  }
  fprintf(stderr, "usage: candidates dump DIR OPERAND N | bench REPS N[,N...]\n");
  return 2;
}
