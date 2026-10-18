// One integer-pipe candidate (hashes.cuh) timed against full SHA-512 (sha512.cuh's compression) rep by rep on one GPU.
// Build one binary a candidate: -DCAND_KIND=0 (SHA-512, the baseline itself), 1 (SHA-512 at -DCAND_R rounds), 2 (SHA-256),
// 3 (BLAKE3); -DSHA_ROLL=0|1 as sha512.cuh (1: the rolled form the baseline's README times).
//
//   candidates dump DIR OPERAND N     the timed kernels' outputs for check.py: the raw chains at the timed grid
//                                    (DIR/rate-base.bin, rate-cand.bin) and the 4 KiB-leaf tree of OPERAND's N leaves
//                                    (DIR/levels-base.bin, levels-cand.bin; level 0 the leaves' digests, then each level up)
//   candidates bench REPS N          JSON lines: REPS reps of the raw rate and of the 4 KiB-leaf tree over N leaves, each
//                                    rep the baseline and the candidate back to back (order alternating), with SM clocks
//
// Raw rate: each thread chains CHAIN = 256 compressions, its block's first 8 words XOR the last state (commit_gpu's
// sha_rate), and stores its final state. 4 KiB leaves: a thread a leaf hashes it with the candidate's standard hash
// (SHA-512 and SHA-256 Merkle-Damgard with padding, BLAKE3's 4 chunks and 3 parents), then a binary tree of node =
// H(left || right), one launch a level, to the root.
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
#endif
using Base = ip::Sha512<0>;
using u64 = uint64_t;
constexpr int CHAIN = 256, RATE_THREADS = 256, LEAF_THREADS = 128, NODE_THREADS = 128;

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
__device__ __forceinline__ void leaf_body(const uint8_t* src, typename H::W* out, u64 n) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  typename H::W d[8];
  H::leaf(reinterpret_cast<const uint4*>(src + t * ip::LEAF), d);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[t * 8 + i] = d[i];
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
extern "C" __global__ void __launch_bounds__(LEAF_THREADS) leaf_base(const uint8_t* s, Base::W* o, u64 n) { leaf_body<Base>(s, o, n); }
extern "C" __global__ void __launch_bounds__(LEAF_THREADS) leaf_cand(const uint8_t* s, Cand::W* o, u64 n) { leaf_body<Cand>(s, o, n); }
extern "C" __global__ void __launch_bounds__(NODE_THREADS) node_base(const Base::W* i, Base::W* o, u64 n) { node_body<Base>(i, o, n); }
extern "C" __global__ void __launch_bounds__(NODE_THREADS) node_cand(const Cand::W* i, Cand::W* o, u64 n) { node_body<Cand>(i, o, n); }

__global__ void fill(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = (uint8_t)x;
  }
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
  lk<<<(unsigned)((T.n + LEAF_THREADS - 1) / LEAF_THREADS), LEAF_THREADS, 0, st>>>(src, T.lv, T.n);
}
template <class H, class NodeK>
static void nodes(const Tree<H>& T, NodeK nk, cudaStream_t st) {
  for (int l = 1; l <= T.depth; ++l) {
    const u64 m = T.n >> l;
    nk<<<(unsigned)((m + NODE_THREADS - 1) / NODE_THREADS), NODE_THREADS, 0, st>>>(T.level(l - 1), T.level(l), m);
  }
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
          "\"base_node_comps\": 2, \"chain\": %d, \"rate_threads\": %d, \"rate_blocks\": %d",
          CAND_NAME.c_str(), CAND_KIND, CAND_KIND == 1 ? CAND_R : (CAND_KIND == 2 ? 64 : CAND_KIND == 3 ? 7 : 80), SHA_ROLL, prop.name,
          prop.multiProcessorCount, prop.major * 10 + prop.minor, Cand::BLOCK, (int)sizeof(Cand::W), CAND_LEAF_COMPS,
          CAND_NODE_COMPS, CHAIN, RATE_THREADS, prop.multiProcessorCount * 8);
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
  leaves(tb, src, leaf_base, 0); nodes(tb, node_base, 0);
  leaves(tc, src, leaf_cand, 0); nodes(tc, node_cand, 0);
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

static int bench(int reps, u64 n) {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  const int blocks = rate_blocks();
  const u64 threads = (u64)blocks * RATE_THREADS;
  Base::W* rb; Cand::W* rc;
  CK(cudaMalloc(&rb, threads * 8 * sizeof(Base::W)));
  CK(cudaMalloc(&rc, threads * 8 * sizeof(Cand::W)));
  uint8_t* src;
  CK(cudaMalloc(&src, n * ip::LEAF));
  fill<<<1024, 256>>>(src, n * ip::LEAF, 7u);
  Tree<Base> tb; Tree<Cand> tc;
  tb.init(n); tc.init(n);
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
  auto lf_base = [&]() { leaves(tb, src, leaf_base, 0); };
  auto nd_base = [&]() { nodes(tb, node_base, 0); };
  auto lf_cand = [&]() { leaves(tc, src, leaf_cand, 0); };
  auto nd_cand = [&]() { nodes(tc, node_cand, 0); };
  for (int w = 0; w < 2; ++w) { timed(raw_base); timed(raw_cand); tree_timed(lf_base, nd_base); tree_timed(lf_cand, nd_cand); }
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
  const double bytes = (double)n * ip::LEAF;
  const double cc = (double)n * CAND_LEAF_COMPS + (double)(n - 1) * CAND_NODE_COMPS, bc = (double)n * 33 + (double)(n - 1) * 2;
  for (int r = 0; r < reps; ++r) {
    std::tuple<double, double, double> b, c;
    if (r % 2 == 0) { b = tree_timed(lf_base, nd_base); c = tree_timed(lf_cand, nd_cand); }
    else { c = tree_timed(lf_cand, nd_cand); b = tree_timed(lf_base, nd_base); }
    printf("{");
    meta(stdout, prop);
    printf(", \"mode\": \"leaf4k\", \"rep\": %d, \"leaves\": %llu, \"operand_bytes\": %.0f, \"ms\": %.5f, \"leaf_ms\": %.5f, "
           "\"compressions\": %.0f, \"gcomp_per_s\": %.4f, \"gb_per_s\": %.3f, \"mhz\": %.0f, \"base_ms\": %.5f, "
           "\"base_leaf_ms\": %.5f, \"base_gcomp_per_s\": %.4f, \"base_gb_per_s\": %.3f, \"base_mhz\": %.0f}\n",
           r, (unsigned long long)n, bytes, std::get<0>(c), std::get<1>(c), cc, cc / std::get<0>(c) * 1e-6,
           bytes / std::get<0>(c) * 1e-6, std::get<2>(c), std::get<0>(b), std::get<1>(b), bc / std::get<0>(b) * 1e-6,
           bytes / std::get<0>(b) * 1e-6, std::get<2>(b));
  }
  fflush(stdout);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 5 && !strcmp(argv[1], "dump")) return dump(argv[2], argv[3], strtoull(argv[4], nullptr, 10));
  if (argc >= 4 && !strcmp(argv[1], "bench")) return bench(atoi(argv[2]), strtoull(argv[3], nullptr, 10));
  fprintf(stderr, "usage: candidates dump DIR OPERAND N | bench REPS N\n");
  return 2;
}
