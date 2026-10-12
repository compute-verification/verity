// Node 2's GPU I/O paths: sustained throughput and where the bytes land.
//
//   paths copy   --gpu G             copy engines: D2H, H2D, both at once (pinned host memory)
//   paths zc     --gpu G             SM stores to and loads from mapped (zero-copy) host memory, by grid
//   paths peer   --gpu G --peer P    peer-to-peer: copy engine, SM stores into P, SM loads from P, by grid
//   paths landg  --gpu G [--peer P]  where inbound bytes land in a GPU: a pointer chase's latency over a buffer just
//                                    written by H2D, by SM stores, by the peer, and after an eviction
//   paths landh  --gpu G             where the GPU's bytes land in the host: a CPU pointer chase over pinned memory just
//                                    written by D2H or by SM stores, against flushed (DRAM) and CPU-written lines
//   paths loop   --gpu G --what W --mib M --seconds S [--beside-read GIB]   a DMA loop for nsys's DRAM and PCIe
//                                    sampling (W: h2d, d2h, zcw, zcr, p2pw, p2pce, smw, none), optionally beside a
//                                    reader of GIB GiB on the device the ring lands in
//
// Every timed path is gated first: its destination must equal its source byte for byte (exit 3 otherwise).
// Output: one JSON object per line on stdout.
#include <cuda_runtime.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                       \
  do {                                                                                              \
    cudaError_t e_ = (x);                                                                           \
    if (e_ != cudaSuccess) {                                                                        \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
      exit(2);                                                                                      \
    }                                                                                               \
  } while (0)

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static void gate(const char* what, const void* a, const void* b, size_t n) {
  if (memcmp(a, b, n) != 0) {
    printf("{\"gate\": \"%s\", \"ok\": false}\n", what);
    fflush(stdout);
    exit(3);
  }
  printf("{\"gate\": \"%s\", \"ok\": true, \"bytes\": %zu}\n", what, n);
}

__global__ void k_fill(uint4* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed;
    p[i] = make_uint4(x, x ^ 0x9e3779b9u, x * 3u, ~x);
  }
}
// SM stores of src into dst (dst may be mapped host memory or a peer's memory)
__global__ void k_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = __ldcs(src + i);
}
// SM stores of a generated pattern (no source read), for the write-only rate
__global__ void k_store(uint4* __restrict__ dst, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i ^ seed;
    dst[i] = make_uint4(x, x, x, x);
  }
}
// SM loads (system or peer memory), xor-reduced so they can't be dropped
__global__ void k_load(const uint4* __restrict__ src, size_t n, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldcv(src + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) *sink = acc;
}
__global__ void k_xor(const uint4* __restrict__ src, size_t n, uint32_t* out) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldcv(src + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  atomicXor(out, acc);
}
// a competing reader, standing in for the GPU's own HBM traffic: `passes` plain-load passes over src
__global__ void k_read_passes(const uint4* __restrict__ src, size_t n, int passes, uint32_t* sink) {
  uint32_t acc = 0;
  for (int p = 0; p < passes; p++)
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
      uint4 v = src[i];
      acc ^= (v.x ^ v.y ^ v.z ^ v.w) + p;
    }
  if (acc == 0x12345678u) *sink = acc;
}
// one thread chases next[] (indices of 128-byte lines, word 0 of each line) and reports cycles per load
__global__ void k_chase(const uint32_t* __restrict__ next, uint32_t start, int loads, unsigned long long* cycles,
                        uint32_t* end) {
  uint32_t i = start;
  long long t0 = clock64();
  for (int k = 0; k < loads; k++) i = __ldcg(next + (size_t)i * 32);
  long long t1 = clock64();
  *cycles = (unsigned long long)(t1 - t0);
  *end = i;
}

struct Args {
  std::string mode, what = "h2d";
  int gpu = 0, peer = -1, reps = 9;
  double seconds = 2.0;
  size_t mib = 256, read_gib = 0;
};

static Args parse(int argc, char** argv) {
  Args a;
  a.mode = argc > 1 ? argv[1] : "";
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--gpu") a.gpu = atoi(v.c_str());
    else if (k == "--peer") a.peer = atoi(v.c_str());
    else if (k == "--reps") a.reps = atoi(v.c_str());
    else if (k == "--mib") a.mib = strtoull(v.c_str(), 0, 10);
    else if (k == "--seconds") a.seconds = atof(v.c_str());
    else if (k == "--what") a.what = v;
    else if (k == "--beside-read") a.read_gib = strtoull(v.c_str(), 0, 10);
    else { fprintf(stderr, "unknown %s\n", k.c_str()); exit(2); }
  }
  return a;
}

static int sms(int g) {
  int n;
  CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, g));
  return n;
}

// time one async operation on a stream, reps times; returns GB/s per rep (1e9 bytes)
template <class F>
static std::vector<double> time_op(cudaStream_t s, size_t bytes, int reps, F f) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  f();  // warm-up
  CK(cudaStreamSynchronize(s));
  std::vector<double> r;
  for (int i = 0; i < reps; i++) {
    CK(cudaEventRecord(a, s));
    f();
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    r.push_back(bytes / (ms * 1e-3) / 1e9);
  }
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return r;
}

static void emit(const char* path, const char* detail, size_t bytes, const std::vector<double>& r) {
  printf("{\"path\": \"%s\", %s\"bytes\": %zu, \"gbps_p50\": %.2f, \"gbps_min\": %.2f, \"gbps_max\": %.2f}\n", path,
         detail, bytes, median(r), *std::min_element(r.begin(), r.end()), *std::max_element(r.begin(), r.end()));
  fflush(stdout);
}

static void mode_copy(const Args& a) {
  CK(cudaSetDevice(a.gpu));
  size_t B = a.mib << 20, n = B / 16;
  uint4 *d, *d2, *h, *h2;
  CK(cudaMalloc(&d, B));
  CK(cudaMalloc(&d2, B));
  CK(cudaHostAlloc(&h, B, cudaHostAllocDefault));
  CK(cudaHostAlloc(&h2, B, cudaHostAllocDefault));
  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  k_fill<<<1024, 256>>>(d, n, 1);
  CK(cudaDeviceSynchronize());
  // gates: D2H then H2D round trip equals the device source
  CK(cudaMemcpy(h, d, B, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(d2, h, B, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(h2, d2, B, cudaMemcpyDeviceToHost));
  gate("copy d2h/h2d round trip", h, h2, B);
  emit("d2h_copy_engine", "", B, time_op(s1, B, a.reps, [&] { CK(cudaMemcpyAsync(h, d, B, cudaMemcpyDeviceToHost, s1)); }));
  emit("h2d_copy_engine", "", B, time_op(s1, B, a.reps, [&] { CK(cudaMemcpyAsync(d2, h2, B, cudaMemcpyHostToDevice, s1)); }));
  // both directions at once: each direction's rate from the pair's span
  std::vector<double> up, down;
  for (int i = 0; i < a.reps + 1; i++) {
    CK(cudaDeviceSynchronize());
    double t0 = now_s();
    CK(cudaMemcpyAsync(h, d, B, cudaMemcpyDeviceToHost, s1));
    CK(cudaMemcpyAsync(d2, h2, B, cudaMemcpyHostToDevice, s2));
    CK(cudaStreamSynchronize(s1));
    double t1 = now_s();
    CK(cudaStreamSynchronize(s2));
    double t2 = now_s();
    if (i == 0) continue;
    down.push_back(B / (t1 - t0) / 1e9);
    up.push_back(B / (t2 - t0) / 1e9);
  }
  emit("d2h_copy_engine_beside_h2d", "", B, down);
  emit("h2d_copy_engine_beside_d2h", "", B, up);
}

static void mode_zc(const Args& a) {
  CK(cudaSetDevice(a.gpu));
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  size_t B = a.mib << 20, n = B / 16;
  uint4 *d, *h, *hd, *chk;
  uint32_t* sink;
  CK(cudaMalloc(&d, B));
  CK(cudaMalloc(&sink, 4));
  CK(cudaHostAlloc(&h, B, cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer((void**)&hd, h, 0));
  chk = (uint4*)malloc(B);
  k_fill<<<1024, 256>>>(d, n, 7);
  CK(cudaDeviceSynchronize());
  k_copy<<<sms(a.gpu) * 2, 512>>>(d, hd, n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(chk, d, B, cudaMemcpyDeviceToHost));
  gate("zero-copy SM stores equal the device source", h, chk, B);
  // loads: the device's xor over mapped memory equals its xor over the device copy
  uint32_t x1 = 0, x2 = 0, *xd;
  CK(cudaMalloc(&xd, 4));
  CK(cudaMemset(xd, 0, 4));
  k_xor<<<sms(a.gpu) * 2, 512>>>(hd, n, xd);
  CK(cudaMemcpy(&x1, xd, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemset(xd, 0, 4));
  k_xor<<<sms(a.gpu) * 2, 512>>>(d, n, xd);
  CK(cudaMemcpy(&x2, xd, 4, cudaMemcpyDeviceToHost));
  gate("zero-copy SM loads equal the device copy (xor)", &x1, &x2, 4);
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  int S = sms(a.gpu);
  for (int g : {8, 16, 32, 64, S, 2 * S, 4 * S}) {
    char det[64];
    snprintf(det, sizeof det, "\"ctas\": %d, ", g);
    emit("zc_sm_store_to_host", det, B, time_op(s, B, a.reps, [&] { k_store<<<g, 512, 0, s>>>(hd, n, 3); }));
    emit("zc_sm_load_from_host", det, B, time_op(s, B, a.reps, [&] { k_load<<<g, 512, 0, s>>>(hd, n, sink); }));
  }
}

static void mode_peer(const Args& a) {
  int can01 = 0, can10 = 0;
  CK(cudaDeviceCanAccessPeer(&can01, a.gpu, a.peer));
  CK(cudaDeviceCanAccessPeer(&can10, a.peer, a.gpu));
  printf("{\"peer_access\": {\"gpu_to_peer\": %d, \"peer_to_gpu\": %d}}\n", can01, can10);
  int pa = 0;
  cudaDeviceGetP2PAttribute(&pa, cudaDevP2PAttrPerformanceRank, a.gpu, a.peer);
  int na = 0;
  cudaDeviceGetP2PAttribute(&na, cudaDevP2PAttrNativeAtomicSupported, a.gpu, a.peer);
  printf("{\"p2p_attr\": {\"performance_rank\": %d, \"native_atomics\": %d}}\n", pa, na);
  size_t B = a.mib << 20, n = B / 16;
  uint4 *d0, *d1, *d1b;
  uint32_t* sink;
  CK(cudaSetDevice(a.gpu));
  CK(cudaMalloc(&d0, B));
  CK(cudaMalloc(&sink, 4));
  k_fill<<<1024, 256>>>(d0, n, 11);
  CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(a.peer));
  CK(cudaMalloc(&d1, B));
  CK(cudaMalloc(&d1b, B));
  k_fill<<<1024, 256>>>(d1b, n, 13);
  CK(cudaDeviceSynchronize());
  std::vector<char> h0(B), h1(B);
  CK(cudaMemcpy(h0.data(), d0, B, cudaMemcpyDeviceToHost));
  cudaStream_t s0, s1;
  CK(cudaSetDevice(a.gpu));
  CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
  CK(cudaSetDevice(a.peer));
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  // without peer access the driver stages through the host
  CK(cudaSetDevice(a.gpu));
  CK(cudaMemcpyPeer(d1, a.peer, d0, a.gpu, B));
  CK(cudaMemcpy(h1.data(), d1, B, cudaMemcpyDeviceToHost));
  gate("memcpyPeer (no peer access) equals the source", h0.data(), h1.data(), B);
  emit("p2p_memcpy_no_peer_access", "", B,
       time_op(s0, B, a.reps, [&] { CK(cudaMemcpyPeerAsync(d1, a.peer, d0, a.gpu, B, s0)); }));
  if (!(can01 && can10)) return;
  CK(cudaSetDevice(a.gpu));
  CK(cudaDeviceEnablePeerAccess(a.peer, 0));
  CK(cudaSetDevice(a.peer));
  CK(cudaDeviceEnablePeerAccess(a.gpu, 0));
  CK(cudaSetDevice(a.gpu));
  CK(cudaMemset(d1, 0, B));
  CK(cudaMemcpyPeer(d1, a.peer, d0, a.gpu, B));
  CK(cudaMemcpy(h1.data(), d1, B, cudaMemcpyDeviceToHost));
  gate("memcpyPeer (peer access) equals the source", h0.data(), h1.data(), B);
  emit("p2p_copy_engine_write", "", B,
       time_op(s0, B, a.reps, [&] { CK(cudaMemcpyPeerAsync(d1, a.peer, d0, a.gpu, B, s0)); }));
  emit("p2p_copy_engine_read_by_peer", "", B,
       time_op(s1, B, a.reps, [&] { CK(cudaMemcpyPeerAsync(d1, a.peer, d0, a.gpu, B, s1)); }));
  // SM stores from gpu into the peer
  CK(cudaSetDevice(a.peer));
  CK(cudaMemset(d1, 0, B));
  CK(cudaDeviceSynchronize());
  CK(cudaSetDevice(a.gpu));
  k_copy<<<sms(a.gpu) * 2, 512>>>(d0, d1, n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h1.data(), d1, B, cudaMemcpyDeviceToHost));
  gate("SM stores into the peer equal the source", h0.data(), h1.data(), B);
  // SM loads from the peer: gpu's xor over the peer's buffer equals the peer's own
  uint32_t x1 = 0, x2 = 0, *xd0, *xd1;
  CK(cudaMalloc(&xd0, 4));
  CK(cudaMemset(xd0, 0, 4));
  k_xor<<<sms(a.gpu) * 2, 512>>>(d1b, n, xd0);
  CK(cudaMemcpy(&x1, xd0, 4, cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(a.peer));
  CK(cudaMalloc(&xd1, 4));
  CK(cudaMemset(xd1, 0, 4));
  k_xor<<<sms(a.peer) * 2, 512>>>(d1b, n, xd1);
  CK(cudaMemcpy(&x2, xd1, 4, cudaMemcpyDeviceToHost));
  gate("SM loads from the peer equal the peer's own (xor)", &x1, &x2, 4);
  CK(cudaSetDevice(a.gpu));
  int S = sms(a.gpu);
  for (int g : {16, 64, S, 2 * S, 4 * S}) {
    char det[64];
    snprintf(det, sizeof det, "\"ctas\": %d, ", g);
    emit("p2p_sm_store_to_peer", det, B, time_op(s0, B, a.reps, [&] { k_store<<<g, 512, 0, s0>>>(d1, n, 5); }));
    emit("p2p_sm_load_from_peer", det, B, time_op(s0, B, a.reps, [&] { k_load<<<g, 512, 0, s0>>>(d1b, n, sink); }));
  }
  // both directions of copy-engine P2P at once
  std::vector<double> ab, ba;
  uint4* d0b;
  CK(cudaMalloc(&d0b, B));
  for (int i = 0; i < a.reps + 1; i++) {
    CK(cudaSetDevice(a.gpu));
    CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(a.peer));
    CK(cudaDeviceSynchronize());
    double t0 = now_s();
    CK(cudaMemcpyPeerAsync(d1, a.peer, d0, a.gpu, B, s0));
    CK(cudaMemcpyPeerAsync(d0b, a.gpu, d1b, a.peer, B, s1));
    CK(cudaStreamSynchronize(s0));
    double t1 = now_s();
    CK(cudaStreamSynchronize(s1));
    double t2 = now_s();
    if (i == 0) continue;
    ab.push_back(B / (t1 - t0) / 1e9);
    ba.push_back(B / (t2 - t0) / 1e9);
  }
  emit("p2p_copy_engine_write_beside_reverse", "", B, ab);
  emit("p2p_copy_engine_reverse_beside_write", "", B, ba);
}

// a single random cycle over L lines (Sattolo), as next-line indices at word 0 of each 128-byte line
static std::vector<uint32_t> cycle_lines(size_t L, uint32_t seed) {
  std::vector<uint32_t> perm(L);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937_64 rng(seed);
  for (size_t i = L - 1; i > 0; i--) std::swap(perm[i], perm[rng() % i]);
  std::vector<uint32_t> buf(L * 32, 0);
  for (size_t i = 0; i < L; i++) buf[(size_t)perm[i] * 32] = perm[(i + 1) % L];
  return buf;
}

static double gpu_chase(const uint32_t* dev, int loads, unsigned long long* dc, uint32_t* de, int clock_khz) {
  k_chase<<<1, 1>>>(dev, 0, loads, dc, de);
  CK(cudaDeviceSynchronize());
  unsigned long long c;
  CK(cudaMemcpy(&c, dc, 8, cudaMemcpyDeviceToHost));
  (void)clock_khz;
  return (double)c / loads;
}

static void mode_landg(const Args& a) {
  CK(cudaSetDevice(a.gpu));
  int clk;
  CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, a.gpu));
  int l2;
  CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, a.gpu));
  printf("{\"l2_bytes\": %d}\n", l2);
  size_t evB = (size_t)4 << 30;
  uint4* ev;
  CK(cudaMalloc(&ev, evB));
  CK(cudaMemset(ev, 1, evB));
  uint32_t* sinkd;
  CK(cudaMalloc(&sinkd, 4));
  unsigned long long* dc;
  uint32_t* de;
  CK(cudaMalloc(&dc, 8));
  CK(cudaMalloc(&de, 4));
  auto evict = [&] {
    k_load<<<sms(a.gpu) * 4, 512>>>(ev, evB / 16, sinkd);
    CK(cudaDeviceSynchronize());
  };
  for (size_t mib : {8, 32, 64, 96}) {
    size_t B = mib << 20, L = B / 128;
    int loads = (int)std::min<size_t>(L, 1 << 16);
    auto host = cycle_lines(L, 99 + (uint32_t)mib);
    uint32_t *h, *dsrc, *dst;
    CK(cudaHostAlloc(&h, B, cudaHostAllocDefault));
    memcpy(h, host.data(), B);
    CK(cudaMalloc(&dsrc, B));
    CK(cudaMalloc(&dst, B));
    CK(cudaMemcpy(dsrc, h, B, cudaMemcpyHostToDevice));
    std::vector<uint32_t> back(L * 32);
    auto row = [&](const char* how, double cyc) {
      printf("{\"landing\": \"gpu\", \"mib\": %zu, \"written_by\": \"%s\", \"cycles_per_load\": %.1f}\n", mib, how, cyc);
      fflush(stdout);
    };
    // H2D into dst, then chase at once
    evict();
    CK(cudaMemcpy(dst, h, B, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(back.data(), dst, B, cudaMemcpyDeviceToHost));
    gate("landg h2d", back.data(), host.data(), B);
    evict();
    CK(cudaMemcpy(dst, h, B, cudaMemcpyHostToDevice));
    row("h2d_copy_engine", gpu_chase(dst, loads, dc, de, clk));
    // the same lines after an eviction (HBM)
    evict();
    row("h2d_then_evicted", gpu_chase(dst, loads, dc, de, clk));
    // SM stores (a device copy), then chase
    evict();
    k_copy<<<sms(a.gpu) * 2, 512>>>((const uint4*)dsrc, (uint4*)dst, B / 16);
    CK(cudaDeviceSynchronize());
    row("sm_stores_local", gpu_chase(dst, loads, dc, de, clk));
    // device-to-device copy engine
    evict();
    CK(cudaMemcpy(dst, dsrc, B, cudaMemcpyDeviceToDevice));
    row("d2d_copy_engine", gpu_chase(dst, loads, dc, de, clk));
    if (a.peer >= 0) {
      int can = 0;
      CK(cudaDeviceCanAccessPeer(&can, a.peer, a.gpu));
      if (can) {
        CK(cudaSetDevice(a.peer));
        cudaError_t e = cudaDeviceEnablePeerAccess(a.gpu, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
        cudaGetLastError();
        uint32_t* psrc;
        CK(cudaMalloc(&psrc, B));
        CK(cudaMemcpy(psrc, h, B, cudaMemcpyHostToDevice));
        CK(cudaSetDevice(a.gpu));
        evict();
        CK(cudaMemcpyPeer(dst, a.gpu, psrc, a.peer, B));
        row("peer_copy_engine", gpu_chase(dst, loads, dc, de, clk));
        evict();
        CK(cudaSetDevice(a.peer));
        k_copy<<<sms(a.peer) * 2, 512>>>((const uint4*)psrc, (uint4*)dst, B / 16);
        CK(cudaDeviceSynchronize());
        CK(cudaSetDevice(a.gpu));
        CK(cudaMemcpy(back.data(), dst, B, cudaMemcpyDeviceToHost));
        gate("landg peer SM stores", back.data(), host.data(), B);
        evict();
        CK(cudaSetDevice(a.peer));
        k_copy<<<sms(a.peer) * 2, 512>>>((const uint4*)psrc, (uint4*)dst, B / 16);
        CK(cudaDeviceSynchronize());
        CK(cudaSetDevice(a.gpu));
        row("peer_sm_stores", gpu_chase(dst, loads, dc, de, clk));
        CK(cudaSetDevice(a.peer));
        CK(cudaFree(psrc));
        CK(cudaSetDevice(a.gpu));
      }
    }
    CK(cudaFreeHost(h));
    CK(cudaFree(dsrc));
    CK(cudaFree(dst));
  }
}

// CPU chase over L lines starting at line 0: ns per load
static double cpu_chase(const uint32_t* buf, size_t loads) {
  uint32_t i = 0;
  double t0 = now_s();
  for (size_t k = 0; k < loads; k++) i = *(volatile const uint32_t*)(buf + (size_t)i * 32);
  double t1 = now_s();
  if (i == 0xffffffffu) printf("x");
  return (t1 - t0) * 1e9 / loads;
}
static void flush(const void* p, size_t B) {
  for (size_t o = 0; o < B; o += 64) _mm_clflush((char*)p + o);
  _mm_sfence();
}

static void mode_landh(const Args& a) {
  CK(cudaSetDevice(a.gpu));
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  for (size_t mib : {4, 8, 16, 32, 64, 128, 256}) {
    size_t B = mib << 20, L = B / 128;
    size_t loads = std::min<size_t>(L, 1 << 17);
    auto host = cycle_lines(L, 7 + (uint32_t)mib);
    uint32_t *h, *hd, *d;
    CK(cudaHostAlloc(&h, B, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**)&hd, h, 0));
    CK(cudaMalloc(&d, B));
    CK(cudaMemcpy(d, host.data(), B, cudaMemcpyHostToDevice));
    auto row = [&](const char* how, double ns) {
      printf("{\"landing\": \"host\", \"mib\": %zu, \"written_by\": \"%s\", \"ns_per_load\": %.1f}\n", mib, how, ns);
      fflush(stdout);
    };
    std::vector<double> r;
    // DRAM: the lines flushed
    memcpy(h, host.data(), B);
    gate("landh cpu", h, host.data(), B);
    for (int k = 0; k < 3; k++) { flush(h, B); r.push_back(cpu_chase(h, loads)); }
    row("flushed_dram", median(r)); r.clear();
    // CPU-written lines (this core's caches and the LLC)
    for (int k = 0; k < 3; k++) { memcpy(h, host.data(), B); r.push_back(cpu_chase(h, loads)); }
    row("cpu_stores", median(r)); r.clear();
    // D2H by the copy engine into flushed lines
    memset(h, 0, B);
    flush(h, B);
    CK(cudaMemcpy(h, d, B, cudaMemcpyDeviceToHost));
    gate("landh d2h", h, host.data(), B);
    for (int k = 0; k < 3; k++) {
      flush(h, B);
      CK(cudaMemcpy(h, d, B, cudaMemcpyDeviceToHost));
      r.push_back(cpu_chase(h, loads));
    }
    row("d2h_copy_engine", median(r)); r.clear();
    // SM stores into flushed lines
    memset(h, 0, B);
    flush(h, B);
    k_copy<<<sms(a.gpu) * 2, 512>>>((const uint4*)d, (uint4*)hd, B / 16);
    CK(cudaDeviceSynchronize());
    gate("landh SM stores", h, host.data(), B);
    for (int k = 0; k < 3; k++) {
      flush(h, B);
      k_copy<<<sms(a.gpu) * 2, 512>>>((const uint4*)d, (uint4*)hd, B / 16);
      CK(cudaDeviceSynchronize());
      r.push_back(cpu_chase(h, loads));
    }
    row("zc_sm_stores", median(r)); r.clear();
    CK(cudaFreeHost(h));
    CK(cudaFree(d));
  }
}

// a steady DMA loop for nsys's DRAM / PCIe sampling: phases of `seconds`, buffer of `mib`
static void mode_loop(const Args& a) {
  CK(cudaSetDevice(a.gpu));
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  size_t B = a.mib << 20, n = B / 16;
  uint4 *d, *h, *hd, *p = nullptr;
  uint32_t* sink;
  CK(cudaMalloc(&d, B));
  CK(cudaMalloc(&sink, 4));
  CK(cudaHostAlloc(&h, B, cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer((void**)&hd, h, 0));
  memset(h, 1, B);
  CK(cudaMemset(d, 2, B));
  if (a.what == "p2pw" || a.what == "p2pce") {
    CK(cudaSetDevice(a.peer));
    CK(cudaMalloc(&p, B));
    CK(cudaSetDevice(a.gpu));
    cudaError_t e = cudaDeviceEnablePeerAccess(a.peer, 0);
    if (e != cudaSuccess) CK(e);
  }
  // --beside-read G: a reader of G GiB on the device the ring lands in (the peer for p2p), running as long as the ring
  int rdev = p ? a.peer : a.gpu, passes = 0;
  uint4* rb = nullptr;
  uint32_t* rsink = nullptr;
  cudaStream_t rs = nullptr;
  cudaEvent_t r0, r1;
  size_t RB = a.read_gib << 30;
  if (RB) {
    CK(cudaSetDevice(rdev));
    CK(cudaMalloc(&rb, RB));
    CK(cudaMalloc(&rsink, 4));
    CK(cudaMemset(rb, 3, RB));
    CK(cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking));
    CK(cudaEventCreate(&r0));
    CK(cudaEventCreate(&r1));
    CK(cudaDeviceSynchronize());
    passes = std::max(1, (int)(a.seconds * 1.2e12 / RB));
    CK(cudaEventRecord(r0, rs));
    k_read_passes<<<2 * sms(rdev), 512, 0, rs>>>(rb, RB / 16, passes, rsink);
    CK(cudaEventRecord(r1, rs));
    CK(cudaSetDevice(a.gpu));
  }
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  CK(cudaDeviceSynchronize());
  double t0 = now_s(), t;
  size_t moved = 0;
  int S = sms(a.gpu);
  do {
    for (int k = 0; k < 8 && a.what != "none"; k++) {
      if (a.what == "h2d") CK(cudaMemcpyAsync(d, h, B, cudaMemcpyHostToDevice, s));
      else if (a.what == "d2h") CK(cudaMemcpyAsync(h, d, B, cudaMemcpyDeviceToHost, s));
      else if (a.what == "zcw") k_store<<<2 * S, 512, 0, s>>>(hd, n, k);
      else if (a.what == "zcr") k_load<<<2 * S, 512, 0, s>>>(hd, n, sink);
      else if (a.what == "p2pw") k_store<<<2 * S, 512, 0, s>>>(p, n, k);
      else if (a.what == "p2pce") CK(cudaMemcpyPeerAsync(p, a.peer, d, a.gpu, B, s));
      else if (a.what == "smw") k_store<<<2 * S, 512, 0, s>>>(d, n, k);
      else { fprintf(stderr, "unknown --what\n"); exit(2); }
      moved += B;
    }
    CK(cudaStreamSynchronize(s));
    t = now_s();
  } while (RB ? cudaEventQuery(r1) == cudaErrorNotReady : t - t0 < a.seconds);
  printf("{\"loop\": \"%s\", \"mib\": %zu, \"seconds\": %.3f, \"gbps\": %.2f", a.what.c_str(), a.mib, t - t0,
         moved / (t - t0) / 1e9);
  if (RB) {
    float ms;
    CK(cudaEventSynchronize(r1));
    CK(cudaEventElapsedTime(&ms, r0, r1));
    printf(", \"beside_read_gib\": %zu, \"read_device\": %d, \"read_gbps\": %.2f", a.read_gib, rdev,
           (double)passes * RB / (ms * 1e-3) / 1e9);
  }
  printf("}\n");
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  if (a.mode == "copy") mode_copy(a);
  else if (a.mode == "zc") mode_zc(a);
  else if (a.mode == "peer") mode_peer(a);
  else if (a.mode == "landg") mode_landg(a);
  else if (a.mode == "landh") mode_landh(a);
  else if (a.mode == "loop") mode_loop(a);
  else { fprintf(stderr, "usage: paths copy|zc|peer|landg|landh|loop [--gpu G] [--peer P] ...\n"); return 2; }
  return 0;
}
