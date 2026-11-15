// Write ceilings on one RTX PRO 6000 (sm_120): the fastest a fill writes into the card's memory by each path a prover
// has, for poub's `timing.GPU_WRITE_BYTES_PER_S`, and the card's PCIe link into host DRAM, for
// `timing.HOST_WRITE_BYTES_PER_S`.
//
//   st<U>      labels derived on the device: grid-stride, each thread U independent 16-byte stores of labels it computes
//              (`label`, a 32-bit mix of the word's index), `cs` with st.global.cs (evict-first streaming stores)
//   bulk<S>    labels derived on the device into a 16 KiB stage of shared memory, stored by one thread with cp.async.bulk
//              (TMA), S stages in flight
//   memset     cudaMemsetAsync of one byte: not labels but a constant, the driver's own fill path
//   h2d<k>     labels streamed in by the copy engines: cudaMemcpyAsync from pinned host memory, PIECE-byte pieces over
//              k streams
//   zcw        labels streamed in by the SMs: a kernel reads mapped pinned host memory and stores it to device memory
//   d2h<k>     the host's streamed-in path from this card: device memory to pinned host memory, as h2d
//
// Every arm passes its gate before it is timed, in the same run: the destination is filled with 0xA5, the arm writes
// once, and 2^16 sampled 16-byte slots, the first and the last must hold what the arm writes; a control that writes
// nothing must fail the same check (`gate`). One JSON line an arm on stdout: the GB/s (1e9 bytes) of each rep over the
// whole destination, its best and median. A buffer is far larger than the 128 MB L2, so every rep reaches DRAM.
//
//   bwwrite --gib 32 --host-gib 8 --reps 7 [--modes memset,dev,d2h,h2d,zcw]
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define OK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(3);                                                                                \
    }                                                                                         \
  } while (0)

constexpr uint32_t SEED = 0x6a09e667u, POISON = 0xA5A5A5A5u, MEMSET_BYTE = 0x3C;
constexpr uint32_t BULK_BYTES = 16384, SAMPLES = 1u << 16;
constexpr uint64_t PIECE = 64ull << 20;

// The label of 32-bit word w: lowbias32 of the word's index, so no two neighbouring words repeat and nothing compresses.
__host__ __device__ __forceinline__ uint32_t label(uint64_t w) {
  uint32_t x = (uint32_t)w ^ SEED ^ (uint32_t)(w >> 32) * 0x9E3779B9u;
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

__host__ __device__ __forceinline__ uint4 slot(uint64_t i) {
  return make_uint4(label(4 * i), label(4 * i + 1), label(4 * i + 2), label(4 * i + 3));
}

template <bool CS>
__device__ __forceinline__ void st16(uint4* p, uint4 v) {
  if (CS)
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
  else
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

template <int U, bool CS>
__global__ void __launch_bounds__(256) st_kernel(uint4* __restrict__ p, uint64_t n16) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n16; i += U * stride) {
#pragma unroll
    for (int u = 0; u < U; ++u) st16<CS>(p + i + u * stride, slot(i + u * stride));
  }
  for (; i < n16; i += stride) st16<CS>(p + i, slot(i));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// CTA b derives pieces b, b + grid, ... into stage k mod S and stores each with one bulk copy; a stage is rewritten only
// once the bulk copy that read it, S groups back, has finished reading it.
template <int S>
__global__ void __launch_bounds__(256) bulk_kernel(uint8_t* __restrict__ p, uint64_t pieces) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint32_t k = 0;
  for (uint64_t q = blockIdx.x; q < pieces; q += gridDim.x, ++k) {
    uint4* v = (uint4*)(sm + (k % S) * BULK_BYTES);
    if (threadIdx.x == 0) asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(S - 1) : "memory");
    __syncthreads();
    const uint64_t base = q * (BULK_BYTES / 16);
    for (uint32_t i = threadIdx.x; i < BULK_BYTES / 16; i += blockDim.x) v[i] = slot(base + i);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(p + q * BULK_BYTES),
                   "r"(smem_u32(v)), "r"(BULK_BYTES)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
  }
  if (threadIdx.x == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// Zero copy in: each thread U 16-byte loads of mapped host memory in flight, each stored to device memory.
template <int U>
__global__ void __launch_bounds__(256) zc_kernel(const uint4* __restrict__ h, uint4* __restrict__ d, uint64_t n16) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n16; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u)
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w)
                   : "l"(h + i + u * stride));
#pragma unroll
    for (int u = 0; u < U; ++u) st16<true>(d + i + u * stride, v[u]);
  }
  for (; i < n16; i += stride) d[i] = h[i];
}

__global__ void gather(const uint4* __restrict__ p, const uint64_t* __restrict__ idx, uint4* out, uint32_t n) {
  const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) out[j] = p[idx[j]];
}

static std::vector<uint64_t> sample_idx(uint64_t n16, uint64_t salt) {
  std::mt19937_64 g(0x5eed ^ salt);
  std::vector<uint64_t> v(SAMPLES + 2);
  v[0] = 0, v[1] = n16 - 1;
  for (uint32_t j = 2; j < v.size(); ++j) v[j] = g() % n16;
  return v;
}

static bool expected(const uint4& got, uint64_t i, bool memset_arm) {
  if (memset_arm) {
    const uint32_t m = 0x01010101u * MEMSET_BYTE;
    return got.x == m && got.y == m && got.z == m && got.w == m;
  }
  const uint4 e = slot(i);
  return got.x == e.x && got.y == e.y && got.z == e.z && got.w == e.w;
}

struct Gate {
  uint64_t checked, wrong;
};

// The sampled slots of device buffer d (n16 slots) against what the arm writes.
static Gate gate_dev(const uint4* d, uint64_t n16, uint64_t salt, bool memset_arm) {
  const std::vector<uint64_t> idx = sample_idx(n16, salt);
  uint64_t* di;
  uint4* dout;
  OK(cudaMalloc(&di, idx.size() * 8));
  OK(cudaMalloc(&dout, idx.size() * 16));
  OK(cudaMemcpy(di, idx.data(), idx.size() * 8, cudaMemcpyHostToDevice));
  gather<<<(uint32_t)((idx.size() + 255) / 256), 256>>>(d, di, dout, (uint32_t)idx.size());
  OK(cudaGetLastError());
  std::vector<uint4> got(idx.size());
  OK(cudaMemcpy(got.data(), dout, idx.size() * 16, cudaMemcpyDeviceToHost));
  OK(cudaFree(di));
  OK(cudaFree(dout));
  Gate g{idx.size(), 0};
  for (size_t j = 0; j < idx.size(); ++j) g.wrong += !expected(got[j], idx[j], memset_arm);
  return g;
}

static Gate gate_host(const uint4* h, uint64_t n16, uint64_t salt) {
  const std::vector<uint64_t> idx = sample_idx(n16, salt);
  Gate g{idx.size(), 0};
  for (uint64_t i : idx) g.wrong += !expected(h[i], i, false);
  return g;
}

static void host_labels(uint4* h, uint64_t n16) {
  const unsigned t = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> ts;
  for (unsigned k = 0; k < t; ++k)
    ts.emplace_back([=] {
      for (uint64_t i = n16 * k / t; i < n16 * (k + 1) / t; ++i) h[i] = slot(i);
    });
  for (auto& x : ts) x.join();
}

static void poison(void* d, uint64_t bytes) {
  OK(cudaMemset(d, 0xA5, bytes));
  OK(cudaDeviceSynchronize());
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

static int REPS = 7;

// Times `fn` (which enqueues one whole write on stream 0, or across `streams`) REPS times after one warm-up, and prints
// the arm's line with its gate.
template <class F>
static void timed(const char* arm, const char* path, std::string config, uint64_t bytes, Gate g, F fn,
                  cudaStream_t s0) {
  if (g.wrong) {
    printf("{\"arm\":\"%s\",\"path\":\"%s\",%s,\"bytes\":%llu,\"gate\":{\"checked\":%llu,\"wrong\":%llu},"
           "\"timed\":false}\n",
           arm, path, config.c_str(), (unsigned long long)bytes, (unsigned long long)g.checked,
           (unsigned long long)g.wrong);
    fflush(stdout);
    return;
  }
  cudaEvent_t a, b;
  OK(cudaEventCreate(&a));
  OK(cudaEventCreate(&b));
  fn();
  OK(cudaDeviceSynchronize());
  std::vector<double> gbs;
  for (int r = 0; r < REPS; ++r) {
    OK(cudaEventRecord(a, s0));
    fn();
    OK(cudaEventRecord(b, s0));
    OK(cudaEventSynchronize(b));
    float ms;
    OK(cudaEventElapsedTime(&ms, a, b));
    gbs.push_back(bytes / (ms * 1e-3) / 1e9);
  }
  OK(cudaEventDestroy(a));
  OK(cudaEventDestroy(b));
  std::string reps;
  for (double x : gbs) reps += (reps.empty() ? "" : ",") + std::to_string(x);
  printf("{\"arm\":\"%s\",\"path\":\"%s\",%s,\"bytes\":%llu,\"gate\":{\"checked\":%llu,\"wrong\":0},\"timed\":true,"
         "\"gb_per_s\":[%s],\"best\":%.3f,\"median\":%.3f}\n",
         arm, path, config.c_str(), (unsigned long long)bytes, (unsigned long long)g.checked, reps.c_str(),
         *std::max_element(gbs.begin(), gbs.end()), median(gbs));
  fflush(stdout);
}

template <int U, bool CS>
static void st_arm(uint4* d, uint64_t n16, int sms, cudaStream_t s0) {
  for (int per : {1, 2, 4, 8}) {
    const int grid = sms * per;
    auto fn = [=] {
      st_kernel<U, CS><<<grid, 256, 0, s0>>>(d, n16);
      OK(cudaGetLastError());
    };
    poison(d, n16 * 16);
    fn();
    OK(cudaDeviceSynchronize());
    const Gate g = gate_dev(d, n16, U * 16 + CS * 8 + per, false);
    char cfg[128];
    snprintf(cfg, sizeof cfg, "\"u\":%d,\"cs\":%s,\"ctas_per_sm\":%d,\"grid\":%d", U, CS ? "true" : "false", per, grid);
    timed(CS ? "st-cs" : "st", "device", cfg, n16 * 16, g, fn, s0);
  }
}

template <int S>
static void bulk_arm(uint4* d, uint64_t n16, int sms, cudaStream_t s0) {
  const size_t smem = (size_t)S * BULK_BYTES;
  OK(cudaFuncSetAttribute(bulk_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int fit = 0;
  OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, bulk_kernel<S>, 256, smem));
  for (int per = 1; per <= std::min(fit, 4); ++per) {
    const int grid = sms * per;
    const uint64_t pieces = n16 * 16 / BULK_BYTES;
    auto fn = [=] {
      bulk_kernel<S><<<grid, 256, smem, s0>>>((uint8_t*)d, pieces);
      OK(cudaGetLastError());
    };
    poison(d, n16 * 16);
    fn();
    OK(cudaDeviceSynchronize());
    const Gate g = gate_dev(d, n16, 1000 + S * 16 + per, false);
    char cfg[128];
    snprintf(cfg, sizeof cfg, "\"stages\":%d,\"ctas_per_sm\":%d,\"grid\":%d", S, per, grid);
    timed("bulk", "device", cfg, n16 * 16, g, fn, s0);
  }
}

// `bytes` from src to dst in PIECE pieces over `k` streams, each stream waiting on stream 0's start, stream 0 on each end.
static void copy_pieces(void* dst, const void* src, uint64_t bytes, cudaMemcpyKind kind, std::vector<cudaStream_t>& ss,
                        int k, std::vector<cudaEvent_t>& ev) {
  OK(cudaEventRecord(ev[0], ss[0]));
  for (int j = 1; j < k; ++j) OK(cudaStreamWaitEvent(ss[j], ev[0], 0));
  uint64_t q = 0;
  for (uint64_t off = 0; off < bytes; off += PIECE, ++q)
    OK(cudaMemcpyAsync((uint8_t*)dst + off, (const uint8_t*)src + off, std::min(PIECE, bytes - off), kind, ss[q % k]));
  for (int j = 1; j < k; ++j) {
    OK(cudaEventRecord(ev[j], ss[j]));
    OK(cudaStreamWaitEvent(ss[0], ev[j], 0));
  }
}

int main(int argc, char** argv) {
  double gib = 32, host_gib = 8;
  std::string modes = "memset,dev,d2h,h2d,zcw";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--gib" && i + 1 < argc) gib = atof(argv[++i]);
    else if (a == "--host-gib" && i + 1 < argc) host_gib = atof(argv[++i]);
    else if (a == "--reps" && i + 1 < argc) REPS = atoi(argv[++i]);
    else if (a == "--modes" && i + 1 < argc) modes = argv[++i];
    else {
      fprintf(stderr, "usage: bwwrite [--gib G] [--host-gib H] [--reps R] [--modes memset,dev,d2h,h2d,zcw]\n");
      return 2;
    }
  }
  auto has = [&](const char* m) { return ("," + modes + ",").find(std::string(",") + m + ",") != std::string::npos; };
  cudaDeviceProp prop;
  OK(cudaGetDeviceProperties(&prop, 0));
  const int sms = prop.multiProcessorCount;
  const uint64_t bytes = (uint64_t)(gib * (1ull << 30)) / PIECE * PIECE, n16 = bytes / 16;
  const uint64_t hbytes = std::min(bytes, (uint64_t)(host_gib * (1ull << 30)) / PIECE * PIECE), hn16 = hbytes / 16;
  printf("{\"device\":\"%s\",\"sms\":%d,\"l2_bytes\":%d,\"bus_bits\":%d,\"bytes\":%llu,\"host_bytes\":%llu,\"reps\":%d,"
         "\"piece\":%llu}\n",
         prop.name, sms, prop.l2CacheSize, prop.memoryBusWidth, (unsigned long long)bytes, (unsigned long long)hbytes,
         REPS, (unsigned long long)PIECE);
  uint4* d;
  OK(cudaMalloc(&d, bytes));
  std::vector<cudaStream_t> ss(4);
  for (auto& s : ss) OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  std::vector<cudaEvent_t> ev(4);
  for (auto& e : ev) OK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  cudaStream_t s0 = ss[0];

  // The control: a destination nothing writes after its poison fails the gate on every sampled slot.
  poison(d, bytes);
  const Gate ctl = gate_dev(d, n16, 7, false);
  printf("{\"control\":\"no write\",\"checked\":%llu,\"wrong\":%llu,\"rejected\":%s}\n", (unsigned long long)ctl.checked,
         (unsigned long long)ctl.wrong, ctl.wrong == ctl.checked ? "true" : "false");
  fflush(stdout);
  if (ctl.wrong != ctl.checked) return 4;

  if (has("memset")) {
    auto fn = [=] { OK(cudaMemsetAsync(d, MEMSET_BYTE, bytes, s0)); };
    poison(d, bytes);
    fn();
    OK(cudaDeviceSynchronize());
    timed("memset", "device", "\"value\":60", bytes, gate_dev(d, n16, 3, true), fn, s0);
  }
  if (has("dev")) {
    st_arm<1, false>(d, n16, sms, s0);
    st_arm<2, false>(d, n16, sms, s0);
    st_arm<4, false>(d, n16, sms, s0);
    st_arm<8, false>(d, n16, sms, s0);
    st_arm<1, true>(d, n16, sms, s0);
    st_arm<2, true>(d, n16, sms, s0);
    st_arm<4, true>(d, n16, sms, s0);
    st_arm<8, true>(d, n16, sms, s0);
    bulk_arm<2>(d, n16, sms, s0);
    bulk_arm<4>(d, n16, sms, s0);
  }
  uint4* h = nullptr;
  if (has("d2h") || has("h2d") || has("zcw")) {
    OK(cudaHostAlloc(&h, hbytes, cudaHostAllocMapped | cudaHostAllocPortable));
    host_labels(h, hn16);
  }
  if (has("d2h")) {
    st_kernel<4, true><<<sms * 4, 256, 0, s0>>>(d, n16);
    OK(cudaDeviceSynchronize());
    for (int k : {1, 2, 4}) {
      auto fn = [&, k] { copy_pieces(h, d, hbytes, cudaMemcpyDeviceToHost, ss, k, ev); };
      memset(h, 0xA5, hbytes);
      fn();
      OK(cudaDeviceSynchronize());
      timed("d2h", "pcie-out", "\"streams\":" + std::to_string(k), hbytes, gate_host(h, hn16, 100 + k), fn, s0);
    }
    host_labels(h, hn16);
  }
  if (has("h2d")) {
    for (int k : {1, 2, 4}) {
      auto fn = [&, k] { copy_pieces(d, h, hbytes, cudaMemcpyHostToDevice, ss, k, ev); };
      poison(d, hbytes);
      fn();
      OK(cudaDeviceSynchronize());
      timed("h2d", "pcie-in", "\"streams\":" + std::to_string(k), hbytes, gate_dev(d, hn16, 200 + k, false), fn, s0);
    }
  }
  if (has("zcw")) {
    uint4* hd;
    OK(cudaHostGetDevicePointer((void**)&hd, h, 0));
    for (int per : {2, 4, 8}) {
      const int grid = sms * per;
      auto fn = [=] {
        zc_kernel<4><<<grid, 256, 0, s0>>>(hd, d, hn16);
        OK(cudaGetLastError());
      };
      poison(d, hbytes);
      fn();
      OK(cudaDeviceSynchronize());
      timed("zcw", "pcie-in", "\"u\":4,\"ctas_per_sm\":" + std::to_string(per), hbytes,
            gate_dev(d, hn16, 300 + per, false), fn, s0);
    }
  }
  if (h) OK(cudaFreeHost(h));
  OK(cudaFree(d));
  return 0;
}
