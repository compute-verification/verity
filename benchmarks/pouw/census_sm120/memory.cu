// Memory census (item 5): the device's memory properties, DRAM read and write bandwidth by data pattern on a plain
// cudaMalloc and on a cuMemCreate(CU_MEM_ALLOCATION_COMP_GENERIC) allocation (is compression per allocation, and which
// patterns compress), device-to-device copy, an L2 footprint sweep (capacity knee and bandwidth; also for compressible
// zeros), shared-memory load/store bandwidth a SM a clock, and random-row gather bandwidth by row size.
#include "common.cuh"

#include <cuda.h>
#include <cstring>

#define CKD(x)                                                                              \
  do {                                                                                      \
    CUresult e_ = (x);                                                                      \
    if (e_ != CUDA_SUCCESS) {                                                               \
      const char* s_ = "?";                                                                 \
      cuGetErrorString(e_, &s_);                                                            \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, s_);                    \
      exit(2);                                                                              \
    }                                                                                       \
  } while (0)

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}

static const char* PATTERNS[] = {"random", "zeros", "const_a5", "repeat_32B", "repeat_128B", "repeat_4KiB", "bits_0x01",
                                 "nibbles_0x0f", "half_zero_sector", "sparse_1in8", "low16", "counter"};
constexpr int NPAT = sizeof(PATTERNS) / sizeof(PATTERNS[0]);

__device__ __forceinline__ uint32_t pattern_word(int pat, uint64_t i) {
  uint32_t lo = (uint32_t)i, hi = (uint32_t)(i >> 32);
  switch (pat) {
    case 0: return hash32(lo * 0x9e3779b9u ^ hash32(hi + 0x1234567u));
    case 1: return 0u;
    case 2: return 0xA5A5A5A5u;
    case 3: return hash32((lo & 7) + 1);
    case 4: return hash32((lo & 31) + 1);
    case 5: return hash32((lo & 1023) + 1);
    case 6: return hash32(lo ^ hi * 0x85ebca6bu) & 0x01010101u;
    case 7: return hash32(lo ^ hi * 0x85ebca6bu) & 0x0F0F0F0Fu;
    case 8: return (lo & 4) ? 0u : hash32(lo ^ hi * 0x85ebca6bu);
    case 9: return (lo & 7) ? 0u : hash32(lo ^ hi * 0x85ebca6bu);
    case 10: return hash32(lo ^ hi * 0x85ebca6bu) & 0xFFFFu;
    case 11: return lo;
  }
  return 0;
}

__global__ void k_fill(uint32_t* p, uint64_t n, int pat) {
  uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, nt = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t i = t; i < n; i += nt) p[i] = pattern_word(pat, i);
}

// Streaming write of a pattern, 16 bytes a lane a store.
__global__ void k_write(uint4* p, uint64_t n16, int pat) {
  uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, nt = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t i = t; i < n16; i += nt) {
    uint64_t w = i * 4;
    p[i] = make_uint4(pattern_word(pat, w), pattern_word(pat, w + 1), pattern_word(pat, w + 2), pattern_word(pat, w + 3));
  }
}

// Reads n16 uint4 words `reps` times (grid-stride, 4 loads in flight a lane), through L2 (ld.global.cg).
__global__ void k_read(const uint4* __restrict__ p, uint64_t n16, int reps, uint32_t* sink) {
  uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, nt = (uint64_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (int r = 0; r < reps; r++) {
    uint64_t i = t;
    for (; i + 3 * nt < n16; i += 4 * nt) {
      uint4 a = __ldcg(p + i), b = __ldcg(p + i + nt), c = __ldcg(p + i + 2 * nt), d = __ldcg(p + i + 3 * nt);
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n16; i += nt) { uint4 a = __ldcg(p + i); acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

// The same streaming read through each load path: 0 ld.global.cg (L2 only), 1 ld.global.nc (__ldg, the
// read-only/texture path through L1), 2 ld.global (default caching), 3 one ld.global.nc in flight a lane (M5's k_read).
template <int PATH>
__device__ __forceinline__ uint4 ld_path(const uint4* p) {
  if (PATH == 0) return __ldcg(p);
  if (PATH == 1 || PATH == 3) return __ldg(p);
  return *p;
}
template <int PATH>
__global__ void k_read_path(const uint4* __restrict__ p, uint64_t n16, int reps, uint32_t* sink) {
  uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, nt = (uint64_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (int r = 0; r < reps; r++) {
    uint64_t i = t;
    if (PATH != 3)
      for (; i + 3 * nt < n16; i += 4 * nt) {
        uint4 a = ld_path<PATH>(p + i), b = ld_path<PATH>(p + i + nt), c = ld_path<PATH>(p + i + 2 * nt),
              d = ld_path<PATH>(p + i + 3 * nt);
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
      }
    for (; i < n16; i += nt) { uint4 a = ld_path<PATH>(p + i); acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

// Random-row gather: groups of `row/16` lanes (or the whole warp, `row/512` times) read one random row of `row` bytes.
__global__ void k_gather(const uint4* __restrict__ p, uint64_t rows, int row, int iters, uint32_t* sink) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lanes_per_row = row >= 512 ? 32 : row / 16;
  const int words_per_row = row / 16;
  uint32_t acc = 0;
  for (int i = 0; i < iters; i++) {
    uint32_t grp = lane / lanes_per_row;
    uint64_t r = hash32(gw * 0x9e3779b9u + (uint32_t)i * 977u + grp * 0x85ebca6bu) % rows;
    const uint4* q = p + r * words_per_row;
    for (int w = lane % lanes_per_row; w < words_per_row; w += lanes_per_row) {
      uint4 v = __ldcg(q + w);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

// Shared memory: conflict-free 128-bit (or 32-bit) loads and 128-bit stores, one block a SM, timed by clock64.
template <int MODE>
__global__ void __launch_bounds__(1024) k_smem(int iters, uint32_t* sink, BlockClk* rec) {
  extern __shared__ __align__(16) uint32_t sm[];
  const int n = 48 * 1024 / 4;  // 48 KiB of words
  for (int i = threadIdx.x; i < n; i += blockDim.x) sm[i] = i * 0x9e3779b9u;
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
  CLK_BEGIN
#pragma unroll 1
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      if (MODE == 0) {
        uint32_t off = ((threadIdx.x + (i * 8 + u) * 64) & (n / 4 - 1)) * 16;
        uint32_t x, y, z, w;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(base + off));
        a0 ^= x; a1 ^= y; a2 ^= z; a3 ^= w;
      } else if (MODE == 1) {
        uint32_t off = ((threadIdx.x + (i * 8 + u) * 64) & (n - 1)) * 4;
        uint32_t x;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base + off));
        a0 ^= x;
      } else {
        uint32_t off = ((threadIdx.x + (i * 8 + u) * 64) & (n / 4 - 1)) * 16;
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(base + off), "r"(a0), "r"(a1), "r"(a2), "r"(i) : "memory");
        a0 += 1;
      }
    }
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + threadIdx.x] = a0 ^ a1 ^ a2 ^ a3 ^ sm[threadIdx.x];
}

static int g_sms;
static uint32_t* g_sink;
static BlockClk* g_rec;

struct Alloc {
  void* p = nullptr;
  size_t bytes = 0;
  bool vmm = false;
  CUmemGenericAllocationHandle h = 0;
  int granted = -1;
};

static Alloc alloc_generic(size_t bytes) {
  Alloc a;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  int dev = 0;
  CK(cudaGetDevice(&dev));
  prop.location.id = dev;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
  size_t gran = 0;
  CKD(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  a.bytes = (bytes + gran - 1) / gran * gran;
  CKD(cuMemCreate(&a.h, a.bytes, &prop, 0));
  CUmemAllocationProp got = {};
  CKD(cuMemGetAllocationPropertiesFromHandle(&got, a.h));
  a.granted = (int)got.allocFlags.compressionType;
  CUdeviceptr ptr;
  CKD(cuMemAddressReserve(&ptr, a.bytes, 0, 0, 0));
  CKD(cuMemMap(ptr, a.bytes, 0, a.h, 0));
  CUmemAccessDesc acc = {};
  acc.location = prop.location;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CKD(cuMemSetAccess(ptr, a.bytes, &acc, 1));
  a.p = (void*)ptr;
  a.vmm = true;
  return a;
}

static Alloc alloc_plain(size_t bytes) {
  Alloc a;
  CK(cudaMalloc(&a.p, bytes));
  a.bytes = bytes;
  return a;
}

static void free_generic(Alloc& a) {
  CKD(cuMemUnmap((CUdeviceptr)a.p, a.bytes));
  CKD(cuMemAddressFree((CUdeviceptr)a.p, a.bytes));
  CKD(cuMemRelease(a.h));
}

static double read_bw(const void* p, size_t bytes, int reps_in_kernel = 1) {
  const int blocks = g_sms * 4, threads = 512;
  double ms = best_ms([&] { k_read<<<blocks, threads>>>((const uint4*)p, bytes / 16, reps_in_kernel, g_sink); }, 4);
  return (double)bytes * reps_in_kernel / (ms * 1e-3);
}

static void props() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));
  CUdevice cd;
  CKD(cuDeviceGet(&cd, dev));
  auto attr = [&](CUdevice_attribute a) { int v = -1; cuDeviceGetAttribute(&v, a, cd); return v; };
  char uuid[64];
  const unsigned char* u = (const unsigned char*)p.uuid.bytes;
  snprintf(uuid, sizeof uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2],
           u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
  size_t fr = 0, tot = 0;
  CK(cudaMemGetInfo(&fr, &tot));
  printf("{\"probe\":\"props\",\"name\":\"%s\",\"uuid\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"l2_bytes\":%d,"
         "\"persisting_l2_max_bytes\":%d,\"smem_per_sm_bytes\":%zu,\"smem_per_block_optin_bytes\":%zu,"
         "\"regs_per_sm\":%d,\"regs_per_block\":%d,\"max_threads_per_sm\":%d,\"max_blocks_per_sm\":%d,\"warp_size\":%d,"
         "\"mem_bus_bits\":%d,\"mem_total_bytes\":%zu,\"sm_clock_khz_attr\":%d,\"mem_clock_khz_attr\":%d,"
         "\"generic_compression\":%d,\"async_engines\":%d,\"pci_bus\":%d,\"pci_device\":%d,\"pci_domain\":%d,"
         "\"l1_global_caching\":%d,\"concurrent_kernels\":%d,\"cooperative_launch\":%d,\"max_cluster\":%d}\n",
         p.name, uuid, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize, p.persistingL2CacheMaxSize,
         p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin, p.regsPerMultiprocessor, p.regsPerBlock,
         p.maxThreadsPerMultiProcessor, p.maxBlocksPerMultiProcessor, p.warpSize, p.memoryBusWidth, tot,
         attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE), attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE),
         attr(CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED), p.asyncEngineCount, p.pciBusID, p.pciDeviceID,
         p.pciDomainID, p.globalL1CacheSupported, p.concurrentKernels, p.cooperativeLaunch,
         attr(CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH));
  fflush(stdout);
}

static const char* PATH_NAME[] = {"ld.global.cg x4", "ld.global.nc x4", "ld.global x4", "ld.global.nc x1 (M5 k_read)"};
template <int PATH>
static double read_path_bw(const void* p, size_t bytes, int reps, int blocks_per_sm) {
  double ms = best_ms([&] { k_read_path<PATH><<<g_sms * blocks_per_sm, 512>>>((const uint4*)p, bytes / 16, reps, g_sink); }, 4);
  return (double)bytes * reps / (ms * 1e-3);
}

// Which load path reads compressed data fastest: every path on plain random, COMP_GENERIC zeros / 0xA5 / random, as a
// 4 GiB (DRAM-resident) and a 32 MiB (L2-resident) footprint.
static void read_paths(Alloc& plain, Alloc& gen, size_t B, const char* only) {
  if (only && !strstr("paths", only)) return;
  struct Case { const char* name; Alloc* a; int pat; };
  Case cases[] = {{"plain_random", &plain, 0}, {"generic_zeros", &gen, 1}, {"generic_const_a5", &gen, 2},
                  {"generic_random", &gen, 0}};
  for (auto& c : cases) {
    k_fill<<<g_sms * 4, 512>>>((uint32_t*)c.a->p, B / 4, c.pat);
    CK(cudaDeviceSynchronize());
    for (int l2 = 0; l2 < 2; l2++) {
      size_t bytes = l2 ? (32ull << 20) : B;
      int reps = l2 ? 128 : 1;
      for (int bps : {4, 8}) {
        double r[4] = {read_path_bw<0>(c.a->p, bytes, reps, bps), read_path_bw<1>(c.a->p, bytes, reps, bps),
                       read_path_bw<2>(c.a->p, bytes, reps, bps), read_path_bw<3>(c.a->p, bytes, reps, bps)};
        for (int k = 0; k < 4; k++)
          printf("{\"probe\":\"read_path\",\"data\":\"%s\",\"footprint_bytes\":%zu,\"path\":\"%s\",\"path_id\":%d,"
                 "\"blocks_per_sm\":%d,\"threads\":512,\"read_bytes_per_s\":%.4e}\n",
                 c.name, bytes, PATH_NAME[k], k, bps, r[k]);
        fflush(stdout);
      }
    }
  }
}

static void dram_patterns(const char* only) {
  const size_t B = 4ull << 30;
  Alloc plain = alloc_plain(B);
  Alloc gen = alloc_generic(B);
  printf("{\"probe\":\"compress_alloc\",\"requested\":\"CU_MEM_ALLOCATION_COMP_GENERIC\",\"granted\":%d,\"bytes\":%zu}\n",
         gen.granted, gen.bytes);
  for (int a = 0; a < 2; a++) {
    Alloc& al = a ? gen : plain;
    for (int pat = 0; pat < NPAT; pat++) {
      char label[64];
      snprintf(label, sizeof label, "dram:%s:%s", a ? "generic" : "plain", PATTERNS[pat]);
      if (only && !strstr(label, only)) continue;
      double wms = best_ms([&] { k_write<<<g_sms * 4, 512>>>((uint4*)al.p, B / 16, pat); }, 3);
      CK(cudaDeviceSynchronize());
      double r = read_bw(al.p, B);
      printf("{\"probe\":\"dram\",\"alloc\":\"%s\",\"pattern\":\"%s\",\"bytes\":%zu,\"read_bytes_per_s\":%.4e,"
             "\"write_bytes_per_s\":%.4e}\n",
             a ? "generic" : "plain", PATTERNS[pat], B, r, B / (wms * 1e-3));
      fflush(stdout);
    }
  }
  // device-to-device copy of random data, plain to plain (bytes moved = read + write)
  if (!only || strstr("dram:copy", only)) {
    k_fill<<<g_sms * 4, 512>>>((uint32_t*)plain.p, B / 4, 0);
    Alloc dst = alloc_plain(B / 2);
    double ms = best_ms([&] { CK(cudaMemcpyAsync(dst.p, plain.p, B / 2, cudaMemcpyDeviceToDevice)); }, 4);
    printf("{\"probe\":\"dram_copy\",\"bytes\":%zu,\"ms\":%.4f,\"copy_bytes_per_s\":%.4e,\"read_plus_write_bytes_per_s\":%.4e}\n",
           B / 2, ms, (B / 2) / (ms * 1e-3), B / (ms * 1e-3));
    fflush(stdout);
    CK(cudaFree(dst.p));
  }
  read_paths(plain, gen, B, only);
  // L2 sweep on compressible zeros: does compressed data take less L2?
  if (!only || strstr("l2gen", only)) {
    k_fill<<<g_sms * 4, 512>>>((uint32_t*)gen.p, B / 4, 1);
    for (int mb : {32, 64, 96, 128, 192, 256, 384, 512, 1024}) {
      size_t bytes = (size_t)mb << 20;
      int reps = std::max(1, (int)(8e9 / bytes));
      double r = read_bw(gen.p, bytes, reps);
      printf("{\"probe\":\"l2_sweep\",\"alloc\":\"generic\",\"pattern\":\"zeros\",\"footprint_mib\":%d,\"reps\":%d,"
             "\"read_bytes_per_s\":%.4e}\n", mb, reps, r);
      fflush(stdout);
    }
  }
  CK(cudaFree(plain.p));
  free_generic(gen);
}

// Does how zeros were written change how fast COMP_GENERIC reads them? A fresh 4 GiB allocation each case, zeros written
// by cudaMemset, 4-byte kernel stores or 16-byte kernel stores, onto fresh memory or over random data; then read six times
// the way M5's compress probe does (r20261015-043430-0a05: one __ldg a lane, 8 blocks of 512 a SM), and once with
// ld.global.cg at 4 blocks a SM (this census's dram rows).
static void fill_methods(const char* only) {
  if (only && !strstr("fill", only)) return;
  const size_t B = 4ull << 30;
  const char* names[] = {"memset", "kfill32", "kwrite128", "random_then_memset", "random_then_kfill32"};
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int trial = 0; trial < 2; trial++)
    for (int m = 0; m < 5; m++) {
      Alloc g = alloc_generic(B);
      uint32_t* p = (uint32_t*)g.p;
      if (m >= 3) k_fill<<<g_sms * 4, 512>>>(p, B / 4, 0);
      if (m == 0 || m == 3) CK(cudaMemset(p, 0, B));
      if (m == 1 || m == 4) k_fill<<<g_sms * 4, 512>>>(p, B / 4, 1);
      if (m == 2) k_write<<<g_sms * 4, 512>>>((uint4*)p, B / 16, 1);
      CK(cudaDeviceSynchronize());
      double r[6], best = 0;
      for (int rep = 0; rep < 6; rep++) {
        CK(cudaEventRecord(e0));
        k_read_path<3><<<g_sms * 8, 512>>>((const uint4*)p, B / 16, 1, g_sink);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        r[rep] = B / (ms * 1e-3);
        if (rep > 0) best = std::max(best, r[rep]);
      }
      double cg = read_path_bw<0>(p, B, 1, 4);
      printf("{\"probe\":\"fill_method\",\"method\":\"%s\",\"trial\":%d,\"bytes\":%zu,\"m5_reads_bytes_per_s\":[%.4e,%.4e,"
             "%.4e,%.4e,%.4e,%.4e],\"m5_best_after_first\":%.4e,\"cg_b4_read_bytes_per_s\":%.4e}\n",
             names[m], trial, B, r[0], r[1], r[2], r[3], r[4], r[5], best, cg);
      fflush(stdout);
      free_generic(g);
    }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
}

static void l2_sweep(const char* only) {
  if (only && !strstr("l2plain", only)) return;
  const size_t B = 1ull << 30;
  Alloc a = alloc_plain(B);
  k_fill<<<g_sms * 4, 512>>>((uint32_t*)a.p, B / 4, 0);
  CK(cudaDeviceSynchronize());
  for (int mb : {1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 256, 512, 1024}) {
    size_t bytes = (size_t)mb << 20;
    int reps = std::max(1, (int)(8e9 / bytes));
    double r = read_bw(a.p, bytes, reps);
    printf("{\"probe\":\"l2_sweep\",\"alloc\":\"plain\",\"pattern\":\"random\",\"footprint_mib\":%d,\"reps\":%d,"
           "\"read_bytes_per_s\":%.4e}\n", mb, reps, r);
    fflush(stdout);
  }
  CK(cudaFree(a.p));
}

static void gather(const char* only) {
  if (only && !strstr("gather", only)) return;
  const size_t B = 4ull << 30;
  Alloc a = alloc_plain(B);
  k_fill<<<g_sms * 4, 512>>>((uint32_t*)a.p, B / 4, 0);
  CK(cudaDeviceSynchronize());
  for (int l2 = 0; l2 < 2; l2++)
    for (int row : {32, 64, 128, 256, 512, 1024, 2048, 4096}) {
      size_t span = l2 ? (32ull << 20) : B;
      uint64_t rows = span / row;
      const int blocks = g_sms * 8, threads = 256;
      int rows_per_warp_iter = row >= 512 ? 1 : 512 / row;
      int iters = std::max(16, (int)(2e9 / ((double)blocks * threads / 32 * rows_per_warp_iter * row)));
      double ms = best_ms([&] { k_gather<<<blocks, threads>>>((const uint4*)a.p, rows, row, iters, g_sink); }, 3);
      double moved = (double)blocks * threads / 32 * iters * rows_per_warp_iter * row;
      printf("{\"probe\":\"gather\",\"target\":\"%s\",\"row_bytes\":%d,\"iters\":%d,\"ms\":%.4f,\"bytes_per_s\":%.4e,"
             "\"rows_per_s\":%.4e}\n", l2 ? "l2" : "dram", row, iters, ms, moved / (ms * 1e-3), moved / row / (ms * 1e-3));
      fflush(stdout);
    }
  CK(cudaFree(a.p));
}

template <int MODE>
static void smem_one(const char* name, int bytes_per_lane_op) {
  auto kern = k_smem<MODE>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  const int iters = 20000, warps = 32;
  Timing t = run_timed([&] { kern<<<g_sms, warps * 32, ONE_BLOCK_SMEM>>>(iters, g_sink, g_rec); }, g_sms, g_rec);
  double bytes = (double)warps * 32 * iters * 8 * bytes_per_lane_op;
  printf("{\"probe\":\"smem\",\"op\":\"%s\",\"warps\":%d,\"iters\":%d,", name, warps, iters);
  print_timing(t);
  printf(",\"bytes_per_sm_per_clk\":%.3f,\"warp_instr_per_sm_per_clk\":%.4f,\"device_bytes_per_s\":%.4e}\n",
         bytes / t.cyc_med, bytes / bytes_per_lane_op / 32 / t.cyc_med, bytes * g_sms / (t.ms * 1e-3));
  fflush(stdout);
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  CK(cudaFree(0));
  g_sms = num_sms();
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  props();
  if (!only || strstr("smem", only)) {
    smem_one<0>("lds128", 16);
    smem_one<1>("lds32", 4);
    smem_one<2>("sts128", 16);
  }
  l2_sweep(only);
  fill_methods(only);
  dram_patterns(only);
  gather(only);
  printf("{\"probe\":\"memory\",\"done\":true}\n");
  return 0;
}
