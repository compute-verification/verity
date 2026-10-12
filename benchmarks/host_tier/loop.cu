// The host tier's honest loop: every visible GPU streams its part of the host tier from pinned host DRAM with zero copy
// and makes a receipt per 4 KiB block bound to its current z; after each chunk it ships a fresh z down to host memory.
// Host CPU threads make receipts over the rest of the host tier (one region a NUMA node, its 64 KiB groups taken from a
// shared counter), each bound to the latest z a GPU shipped. Adversary threads, if given, read DRAM of their own
// meanwhile: what they get is bandwidth the loop left over.
//
//   loop --seconds 5 --gpu-mib 4096 --grid 16 --cpus 52-95 --cpu-mib 512 --ks 16 [--adv-cpus L --adv-mib M]
//        [--gpus N] [--gpu-node local|N] [--host-cpus L] [--sha c|ni] [--pf BYTES] [--go-file F] [--gate DIR] [--label S]
//
// --gpus N streams from the first N visible GPUs only (0: the CPUs alone); GPU d's driver thread runs on the d-th CPU
// of --host-cpus if given. CPU lists take lo-hi:step. --go-file F is created when the timed phase starts.
//
// Receipts (measurement constructions, not certified ones; reference.py recomputes both). Both XOR the block with a
// 4 KiB mask M expanded from all of z (receipt.h's cpu_mask), so that no precomputation smaller than the block answers
// a fresh z (with z itself repeating every 64 bytes, the coefficients of z's bits would be 2-3.5 KiB a block):
//   gpu  r(b) = (sum_i (w_i ^ M_i) k_i) | (sum_i (w_i ^ M_i) k'_i) << 32, over the block's 1,024 u32 words, mod 2^32
//        each, keys from the seed.
//   cpu  s_l(b) = sum_j K_l[j mod 256] (x_j ^ M_j) for l < KS, over the block's 4,096 bytes (u8 x, s8 K: VNNI),
//        and per 16 blocks a leaf SHA-256(seed || group || z sequence || the 16 x KS sketches, int32 LE).
// z here is the GPU's fold of its last chunk's receipts, standing in for the PoUW tile output the design binds.
// --gate DIR: compute the first blocks' receipts with z0 on GPU 0 and CPU thread 0, write DIR/gate.json, and exit.
#include "receipt.h"

#include <cuda_runtime.h>
#include <emmintrin.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                            \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) {                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(6);                                                                           \
    }                                                                                    \
  } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

const uint8_t SEED[32] = {0x68, 0x6f, 0x73, 0x74, 0x2d, 0x74, 0x69, 0x65, 0x72, 0x2f, 0x76, 0x30, 1, 2, 3, 4,
                                 5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17, 18, 19, 20};
static uint64_t seed64() {
  uint64_t s;
  memcpy(&s, SEED, 8);
  return s;
}

// ---- keys, content, z0 (reference.py has the same) ----
static void gpu_keys(uint32_t* k) {  // k[0..1023], then k'[0..1023]
  for (int i = 0; i < 1024; ++i) {
    uint64_t v = splitmix64(seed64() ^ (0x6b00000000ull + i));
    k[i] = (uint32_t)v | 1, k[1024 + i] = (uint32_t)(v >> 32) | 1;
  }
}
static void z0(uint32_t* z) {
  for (int t = 0; t < 16; ++t) z[t] = (uint32_t)splitmix64(seed64() ^ (0x7a00000000ull + t));
}
static void fill(uint64_t* p, size_t words, uint64_t region, int threads) {
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t)
    th.emplace_back([=] {
      for (size_t i = words * t / threads; i < words * (t + 1) / threads; ++i) p[i] = splitmix64(i ^ (region << 48));
    });
  for (auto& x : th) x.join();
}

// ---- GPU ----
struct ZSlot {  // written by a GPU over PCIe, read by host threads: a seqlock, odd while z is being written
  volatile uint64_t seq;
  volatile uint32_t z[16];
  uint8_t pad[56];
};

__device__ __host__ inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

__global__ void stream_receipts(const uint4* __restrict__ host, size_t nblocks, const uint32_t* __restrict__ keys,
                                const uint32_t* __restrict__ z, uint64_t seed, unsigned long long* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  const size_t warp = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = ((size_t)gridDim.x * blockDim.x) >> 5;
  __shared__ uint4 mask[256];  // the 4 KiB mask under z, as u32 words 4u .. 4u + 3
  uint64_t h = seed;
  for (int t = 0; t < 16; ++t) h = mix64(h ^ z[t]);
  for (int i = threadIdx.x; i < 512; i += blockDim.x) reinterpret_cast<uint64_t*>(mask)[i] = mix64(h ^ (0x6d00000000ull + i));
  __syncthreads();
  for (size_t b = warp; b < nblocks; b += nw) {
    const uint4* p = host + b * 256;
    uint4 v[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) v[l] = p[l * 32 + lane];
    uint32_t r0 = 0, r1 = 0;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      const int i = 128 * l + 4 * lane;
      const uint4 m = mask[l * 32 + lane];
      uint32_t x0 = v[l].x ^ m.x, x1 = v[l].y ^ m.y, x2 = v[l].z ^ m.z, x3 = v[l].w ^ m.w;
      r0 += x0 * __ldg(keys + i) + x1 * __ldg(keys + i + 1) + x2 * __ldg(keys + i + 2) + x3 * __ldg(keys + i + 3);
      r1 += x0 * __ldg(keys + 1024 + i) + x1 * __ldg(keys + 1025 + i) + x2 * __ldg(keys + 1026 + i) +
            x3 * __ldg(keys + 1027 + i);
    }
    for (int o = 16; o; o >>= 1) r0 += __shfl_xor_sync(0xffffffffu, r0, o), r1 += __shfl_xor_sync(0xffffffffu, r1, o);
    if (lane == 0) out[b] = (unsigned long long)r0 | ((unsigned long long)r1 << 32);
  }
}

// z for the next chunk: a fold of this chunk's receipts, kept on the device and shipped to the host's slot
__global__ void publish_z(const unsigned long long* __restrict__ r, size_t n, uint32_t* zdev, ZSlot* slot,
                          unsigned long long seq) {
  const int t = threadIdx.x;  // 16 threads
  uint32_t a = 0x811c9dc5u + t;
  for (size_t i = t; i < n; i += 16 * 64) a = (a ^ (uint32_t)r[i] ^ (uint32_t)(r[i] >> 32)) * 0x01000193u;
  zdev[t] = a;
  if (t == 0) slot->seq = 2 * seq - 1;
  __threadfence_system();
  __syncwarp(0xffffu);
  slot->z[t] = a;
  __threadfence_system();
  __syncwarp(0xffffu);
  if (t == 0) slot->seq = 2 * seq;
  __threadfence_system();
}

static bool read_z(const ZSlot* s, uint32_t* z, uint64_t* seq) {
  uint64_t a = s->seq;
  if (a & 1) return false;
  _mm_lfence();
  for (int t = 0; t < 16; ++t) z[t] = s->z[t];
  _mm_lfence();
  return s->seq == a ? (*seq = a / 2, true) : false;
}

static void* host_alloc(size_t n, int node) {
  void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return nullptr;
  madvise(p, n, MADV_HUGEPAGE);
  if (node >= 0) {
    unsigned long mask = 1ul << node;
    if (syscall(SYS_mbind, p, n, 2, &mask, 64, 1) != 0) return nullptr;
  }
  return p;
}

static std::vector<int> parse_cpus(const std::string& s) {
  std::vector<int> out;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    std::string tok = s.substr(p, q - p);
    int lo, hi, step = 1;
    if (sscanf(tok.c_str(), "%d-%d:%d", &lo, &hi, &step) >= 2)
      for (int c = lo; c <= hi; c += step > 0 ? step : 1) out.push_back(c);
    else if (!tok.empty())
      out.push_back(atoi(tok.c_str()));
    p = q + 1;
  }
  return out;
}

static void pin(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

static int cpu_node(int cpu) {
  for (int n = 0; n < 8; ++n) {
    char path[96];
    snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpu%d", n, cpu);
    if (access(path, F_OK) == 0) return n;
  }
  return -1;
}

int main(int argc, char** argv) {
  double seconds = 5;
  size_t gpu_bytes = 4096ull << 20, cpu_bytes = 512ull << 20, adv_bytes = 256ull << 20;
  int grid = 16, ks = 16, use_gpus = -1;
  std::string cpus, adv_cpus, host_cpus, gate_dir, go_file, label, gpu_node = "local";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i], v = argv[i + 1];
    if (a == "--seconds") seconds = atof(v.c_str());
    else if (a == "--gpu-mib") gpu_bytes = (size_t)atol(v.c_str()) << 20;
    else if (a == "--cpu-mib") cpu_bytes = (size_t)atol(v.c_str()) << 20;
    else if (a == "--adv-mib") adv_bytes = (size_t)atol(v.c_str()) << 20;
    else if (a == "--grid") grid = atoi(v.c_str());
    else if (a == "--ks") ks = atoi(v.c_str());
    else if (a == "--cpus") cpus = v;
    else if (a == "--adv-cpus") adv_cpus = v;
    else if (a == "--gate") gate_dir = v;
    else if (a == "--label") label = v;
    else if (a == "--gpu-node") gpu_node = v;
    else if (a == "--gpus") use_gpus = atoi(v.c_str());
    else if (a == "--host-cpus") host_cpus = v;
    else if (a == "--go-file") go_file = v;
    else if (a == "--sha") sha_ni = v == "ni";
    else if (a == "--pf") pf_bytes = atoi(v.c_str());
    else {
      fprintf(stderr, "unknown flag %s\n", a.c_str());
      return 2;
    }
  }
  if (ks != 4 && ks != 16) return 2;
  const bool gating = !gate_dir.empty();
  if (gating) gpu_bytes = 64ull << 20, cpu_bytes = 2ull << 20;
  int ngpu = 0;
  if (cudaGetDeviceCount(&ngpu) != cudaSuccess) ngpu = 0;
  if (use_gpus >= 0 && use_gpus < ngpu) ngpu = use_gpus;
  std::vector<int> cc = parse_cpus(cpus), ac = parse_cpus(adv_cpus), hc = parse_cpus(host_cpus);
  if (gating) cc.resize(cc.empty() ? 0 : 1), ac.clear();
  uint32_t gk[2048], zinit[16];
  gpu_keys(gk);
  z0(zinit);
  alignas(64) int8_t K[16 * 256];
  cpu_keys(K, ks);

  // the z slots live in one page of mapped pinned memory, one per GPU
  ZSlot* slots = nullptr;
  if (ngpu) CK(cudaHostAlloc((void**)&slots, 4096, cudaHostAllocMapped | cudaHostAllocPortable));
  else slots = (ZSlot*)aligned_alloc(4096, 4096);
  for (size_t d = 0; d < 4096 / sizeof(ZSlot); ++d) {
    slots[d].seq = 0;
    for (int t = 0; t < 16; ++t) slots[d].z[t] = zinit[t];
  }

  std::atomic<int> stop{0};
  pthread_barrier_t ready, go;
  const int nthreads = ngpu + (int)cc.size() + (int)ac.size();
  pthread_barrier_init(&ready, nullptr, nthreads + 1);
  pthread_barrier_init(&go, nullptr, nthreads + 1);
  std::vector<double> gpu_rate(ngpu), cpu_bytes_done(cc.size()), adv_bytes_done(ac.size());
  std::vector<uint64_t> z_published(ngpu), z_distinct(cc.size());
  std::vector<std::string> gpu_bus(ngpu);
  std::vector<unsigned long long> gate_gpu(16);
  std::vector<int32_t> gate_sk(16 * ks);
  uint8_t gate_leaf[32];
  int gate_region = 1000;
  std::vector<std::thread> th;

  for (int d = 0; d < ngpu; ++d)
    th.emplace_back([&, d] {
      if (d < (int)hc.size()) pin(hc[d]);
      CK(cudaSetDevice(d));
      char bus[32];
      CK(cudaDeviceGetPCIBusId(bus, sizeof bus, d));
      gpu_bus[d] = bus;
      std::string sys = std::string("/sys/bus/pci/devices/") + bus;
      for (auto& ch : sys) ch = tolower(ch);
      int numa = -1;
      if (FILE* f = fopen((sys + "/numa_node").c_str(), "r")) {
        if (fscanf(f, "%d", &numa) != 1) numa = -1;
        fclose(f);
      }
      int node = gpu_node == "local" ? numa : atoi(gpu_node.c_str());
      uint4* host = (uint4*)host_alloc(gpu_bytes, node);
      if (!host) exit(5);
      fill((uint64_t*)host, gpu_bytes / 8, (uint64_t)d, 8);
      CK(cudaHostRegister(host, gpu_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
      uint4* hdev;
      CK(cudaHostGetDevicePointer((void**)&hdev, host, 0));
      ZSlot* sdev;
      CK(cudaHostGetDevicePointer((void**)&sdev, slots + d, 0));
      uint32_t *keys, *zdev;
      unsigned long long* rec;
      const size_t nb = gpu_bytes / 4096, chunk_blocks = 16384;  // 64 MiB a launch
      CK(cudaMalloc(&keys, sizeof gk));
      CK(cudaMalloc(&zdev, 64));
      CK(cudaMalloc(&rec, nb * 8));
      CK(cudaMemcpy(keys, gk, sizeof gk, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(zdev, zinit, 64, cudaMemcpyHostToDevice));
      cudaStream_t s;
      CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      if (gating) {
        if (d == 0) {
          stream_receipts<<<grid, 512, 0, s>>>(hdev, 16, keys, zdev, seed64(), rec);
          CK(cudaGetLastError());
          CK(cudaStreamSynchronize(s));
          CK(cudaMemcpy(gate_gpu.data(), rec, 16 * 8, cudaMemcpyDeviceToHost));
        }
        pthread_barrier_wait(&ready);
        pthread_barrier_wait(&go);
        return;
      }
      // one untimed chunk, so the module load stays out of the time
      stream_receipts<<<grid, 512, 0, s>>>(hdev, chunk_blocks, keys, zdev, seed64(), rec);
      CK(cudaGetLastError());
      CK(cudaStreamSynchronize(s));
      cudaEvent_t ev[2];
      CK(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming));
      pthread_barrier_wait(&ready);
      pthread_barrier_wait(&go);
      double t0 = now(), bytes = 0;
      uint64_t seq = 0;
      size_t b0 = 0;
      for (int r = 0;; ++r) {
        size_t n = b0 + chunk_blocks <= nb ? chunk_blocks : nb - b0;
        stream_receipts<<<grid, 512, 0, s>>>(hdev + b0 * 256, n, keys, zdev, seed64(), rec + b0);
        publish_z<<<1, 16, 0, s>>>(rec + b0, n, zdev, sdev, ++seq);
        CK(cudaGetLastError());
        CK(cudaEventRecord(ev[r & 1], s));
        b0 = (b0 + n) % nb;
        if (r > 0) {
          CK(cudaEventSynchronize(ev[(r - 1) & 1]));
          bytes += (double)chunk_blocks * 4096;
          if (stop.load(std::memory_order_relaxed)) break;
        }
      }
      CK(cudaStreamSynchronize(s));
      bytes += (double)chunk_blocks * 4096;
      gpu_rate[d] = bytes / (now() - t0) / 1e9;
      z_published[d] = seq;
    });

  // the CPUs' part of the host tier: one region a NUMA node, cpu_bytes for each of its threads, which take its 64 KiB
  // groups four at a time from a shared counter, so a pass is done when the region is, whichever thread is slow
  struct Pool {
    int node = -1;
    size_t members = 0, groups = 0;
    uint8_t* x = nullptr;
    std::vector<uint8_t> leaves;
    alignas(64) std::atomic<uint64_t> next{0};
  };
  std::vector<std::unique_ptr<Pool>> pools;
  std::vector<size_t> pool_of(cc.size()), rank(cc.size());
  for (size_t t = 0; t < cc.size(); ++t) {
    const int n = cpu_node(cc[t]);
    size_t i = 0;
    while (i < pools.size() && pools[i]->node != n) ++i;
    if (i == pools.size()) pools.emplace_back(new Pool), pools.back()->node = n;
    pool_of[t] = i, rank[t] = pools[i]->members++;
  }
  for (auto& q : pools) {
    q->x = (uint8_t*)host_alloc(q->members * cpu_bytes, q->node);
    if (!q->x) exit(5);
    q->groups = q->members * cpu_bytes / 65536;
    q->leaves.resize(q->groups * 32);
  }
  for (size_t t = 0; t < cc.size(); ++t)
    th.emplace_back([&, t] {
      pin(cc[t]);
      Pool& q = *pools[pool_of[t]];
      const size_t w = cpu_bytes / 8;
      uint64_t* xw = (uint64_t*)q.x;
      for (size_t i = rank[t] * w; i < (rank[t] + 1) * w; ++i) xw[i] = splitmix64(i ^ ((uint64_t)(1000 + q.node) << 48));
      std::vector<int32_t> sk(16 * ks);
      const ZSlot* slot = slots + (ngpu ? t % ngpu : 0);
      uint32_t z[16];
      alignas(64) uint8_t M[4096];
      uint64_t zseq = 0, last = UINT64_MAX, distinct = 0;
      if (gating) {
        if (t == 0) cpu_mask(zinit, M), cpu_group(q.x, M, 0, 0, K, ks, gate_leaf, gate_sk.data()), gate_region = 1000 + q.node;
        pthread_barrier_wait(&ready);
        pthread_barrier_wait(&go);
        return;
      }
      pthread_barrier_wait(&ready);
      pthread_barrier_wait(&go);
      double bytes = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        const uint64_t g0 = q.next.fetch_add(4, std::memory_order_relaxed);
        for (uint64_t k = 0; k < 4; ++k) {
          const size_t g = (g0 + k) % q.groups;
          while (!read_z(slot, z, &zseq)) _mm_pause();
          if (zseq != last) ++distinct, last = zseq, cpu_mask(z, M);
          cpu_group(q.x + g * 65536, M, zseq, g, K, ks, &q.leaves[g * 32], sk.data());
        }
        bytes += 4 * 65536;
      }
      cpu_bytes_done[t] = bytes;
      z_distinct[t] = distinct;
    });

  for (size_t t = 0; t < ac.size(); ++t)
    th.emplace_back([&, t] {
      pin(ac[t]);
      uint8_t* x = (uint8_t*)host_alloc(adv_bytes, cpu_node(ac[t]));
      if (!x) exit(5);
      fill((uint64_t*)x, adv_bytes / 8, 2000 + t, 1);
      const size_t chunk = 1u << 20, nc = adv_bytes / chunk;
      pthread_barrier_wait(&ready);
      pthread_barrier_wait(&go);
      double bytes = 0;
      volatile uint64_t sink = 0;
      for (size_t c = 0; !stop.load(std::memory_order_relaxed); c = (c + 1) % nc) {
        sink = sink + adv_chunk(x + c * chunk, chunk / 64);
        bytes += chunk;
      }
      adv_bytes_done[t] = bytes;
    });

  pthread_barrier_wait(&ready);
  double t0 = now();
  pthread_barrier_wait(&go);
  if (!go_file.empty())
    if (FILE* f = fopen(go_file.c_str(), "w")) fclose(f);
  if (!gating) std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  stop = 1;
  for (auto& x : th) x.join();
  double el = now() - t0;

  if (gating) {
    std::string path = gate_dir + "/gate.json";
    FILE* f = fopen(path.c_str(), "w");
    fprintf(f, "{\"seed\": \"");
    for (int i = 0; i < 32; ++i) fprintf(f, "%02x", SEED[i]);
    fprintf(f, "\", \"ks\": %d, \"gpu_region\": 0, \"gpu_receipts\": [", ks);
    for (int b = 0; b < 16; ++b) fprintf(f, "%s\"%016llx\"", b ? ", " : "", gate_gpu[b]);
    fprintf(f, "], \"cpu_region\": %d, \"cpu_sketches\": [", gate_region);
    for (int i = 0; i < 16 * ks; ++i) fprintf(f, "%s%d", i ? ", " : "", gate_sk[i]);
    fprintf(f, "], \"cpu_leaf\": \"");
    for (int i = 0; i < 32; ++i) fprintf(f, "%02x", gate_leaf[i]);
    fprintf(f, "\"}\n");
    fclose(f);
    printf("gate written to %s\n", path.c_str());
    return 0;
  }
  double g = 0, c = 0, a = 0, cmin = 1e30, cmax = 0;
  std::string per = "[";
  for (int d = 0; d < ngpu; ++d) {
    g += gpu_rate[d];
    char b[128];
    snprintf(b, sizeof b, "%s{\"bus\": \"%s\", \"gb_per_s\": %.2f, \"z\": %llu}", d ? ", " : "", gpu_bus[d].c_str(),
             gpu_rate[d], (unsigned long long)z_published[d]);
    per += b;
  }
  per += "]";
  uint64_t zmin = UINT64_MAX;
  for (size_t t = 0; t < cc.size(); ++t) {
    double r = cpu_bytes_done[t] / el / 1e9;
    c += r, cmin = r < cmin ? r : cmin, cmax = r > cmax ? r : cmax;
    zmin = z_distinct[t] < zmin ? z_distinct[t] : zmin;
  }
  for (size_t t = 0; t < ac.size(); ++t) a += adv_bytes_done[t] / el / 1e9;
  printf("{\"label\": \"%s\", \"seconds\": %.2f, \"gpus\": %d, \"grid\": %d, \"gpu_gb_per_s\": %.2f, \"cpu_threads\": %zu, "
         "\"ks\": %d, \"sha_ni\": %d, \"pf\": %d, \"cpu_gb_per_s\": %.2f, \"cpu_thread_min\": %.2f, \"cpu_thread_max\": %.2f, \"cpu_z_distinct_min\": %llu, "
         "\"adv_threads\": %zu, \"adv_gb_per_s\": %.2f, \"honest_gb_per_s\": %.2f, \"per_gpu\": %s}\n",
         label.c_str(), el, ngpu, grid, g, cc.size(), ks, sha_ni, pf_bytes, c, cc.empty() ? 0 : cmin, cmax,
         (unsigned long long)(cc.empty() ? 0 : zmin), ac.size(), a, g + c, per.c_str());
  return 0;
}
