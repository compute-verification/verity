// Attack A2 and B on one RTX PRO 6000.
//
// A2 (compression, the filler case and the compressor's grain): under a compressible VMM allocation (4 GiB), the read
// rate of
//   zeros_enc    an all-zero payload encoded by the deployed kv.cu (p1024/k3/masked): the store a node holds for filler
//   const5a      every byte 0x5A, not encoded: does the controller compress a constant, or only zero?
//   zrun_N       alternating runs of N zero bytes and N random bytes (half the store zero), N = 32 … 4096: the smallest
//                aligned zero run the controller compresses
// A higher rate than plain means fewer HBM bytes.
//
// B (on-chip state across a pass boundary): a pass is its own kernel launch, so registers and shared memory do not
// survive it, and L2 is what a non-persistent receipt kernel can reuse.  For a working set of X bytes, read it in R
// back-to-back launches and report R·X over the time: where it stays far above the HBM rate the set is served on chip
// across launches.  The knee bounds, from the attacker's side, how much of the store one GPU can re-serve on chip per
// pass without an HBM read (μ_chip's S_chip).  `persist` repeats each size with an L2 access-policy window marking the
// set persisting (hitRatio 1), the best an attacker can ask of L2.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../../word_codec/kv.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)
#define CKD(x) do { CUresult e_ = (x); if (e_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorString(e_, &s_); \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, s_); return 1; } } while (0)

__device__ __forceinline__ unsigned mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return (unsigned)(z ^ (z >> 31));
}

constexpr unsigned ALL_RANDOM = ~0u;

// run = 0: every byte `cval`; run = ALL_RANDOM: every byte random; else byte offset o is zero where (o / run) is even,
// random where odd (run a multiple of 16).
__global__ void fill_kernel(uint4* p, size_t n16, unsigned run, unsigned cval) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v;
    if (!run) {
      v = make_uint4(cval, cval, cval, cval);
    } else if (run == ALL_RANDOM) {
      v = make_uint4(mix(i), mix(i ^ 1), mix(i ^ 2), mix(i ^ 3));
    } else {
      const size_t o = i * 16;
      v = ((o / run) & 1) ? make_uint4(mix(i), mix(i ^ 1), mix(i ^ 2), mix(i ^ 3)) : make_uint4(0, 0, 0, 0);
    }
    p[i] = v;
  }
}

__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, unsigned* sink) {
  unsigned acc = 0;
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nth = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n16; i += nth) { uint4 v = __ldg(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

static int read_rate(const uint4* p, size_t bytes, int sms, unsigned* sink, double* out) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double best = 0;
  for (int rep = 0; rep < 8; rep++) {
    CK(cudaEventRecord(e0));
    read_kernel<<<sms * 8, 512>>>(p, bytes / 16, sink);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    double r = bytes / (ms / 1e3);
    if (rep > 1 && r > best) best = r;
  }
  *out = best;
  return 0;
}

static int set_key(const char* path) {
  uint8_t key[5 * 128];
  FILE* f = fopen(path, "rb");
  if (!f || fread(key, 1, sizeof key, f) != sizeof key) { fprintf(stderr, "key.bin: %s\n", path); return 1; }
  fclose(f);
  const uint32_t* w = reinterpret_cast<const uint32_t*>(key);
  if (kv_set_key(w, 3, w + 3 * 32) || kv_set_mask(w + 4 * 32)) { fprintf(stderr, "kv_set_key/mask\n"); return 1; }
  return 0;
}

// R launches over the first X bytes of p, after one warming launch; R·X / time.
static int resident_rate(const uint4* p, size_t X, int sms, unsigned* sink, cudaStream_t s, double* out) {
  const int R = (int)((16ull << 30) / X) < 4 ? 4 : (int)((16ull << 30) / X);
  const int launches = R > 4096 ? 4096 : R;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const unsigned grid = (unsigned)(X / 16 / 512) < (unsigned)(sms * 8) ? (unsigned)(X / 16 / 512 + 1) : sms * 8;
  read_kernel<<<grid, 512, 0, s>>>(p, X / 16, sink);
  double best = 0;
  for (int trial = 0; trial < 3; trial++) {
    CK(cudaEventRecord(e0, s));
    for (int r = 0; r < launches; r++) read_kernel<<<grid, 512, 0, s>>>(p, X / 16, sink);
    CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    const double r = (double)X * launches / (ms / 1e3);
    if (r > best) best = r;
  }
  *out = best;
  return 0;
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  if (set_key(argc > 1 ? argv[1] : "key.bin")) return 1;
  CUdevice dev; CKD(cuDeviceGet(&dev, 0));
  int sms = 0, l2 = 0, persist_max = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  CK(cudaDeviceGetAttribute(&persist_max, cudaDevAttrMaxPersistingL2CacheSize, 0));
  printf("{\"probe\":\"props\",\"sms\":%d,\"l2_bytes\":%d,\"persisting_l2_max_bytes\":%d}\n", sms, l2, persist_max);
  unsigned* sink; CK(cudaMalloc(&sink, 4));

  // ---- A2: compressible allocation ----
  const size_t B = 4ull << 30;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
  size_t gran = 0;
  CKD(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  const size_t size = (B + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle hnd;
  CKD(cuMemCreate(&hnd, size, &prop, 0));
  CUdeviceptr ptr;
  CKD(cuMemAddressReserve(&ptr, size, 0, 0, 0));
  CKD(cuMemMap(ptr, size, 0, hnd, 0));
  CUmemAccessDesc acc = {};
  acc.location = prop.location;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CKD(cuMemSetAccess(ptr, size, &acc, 1));
  uint4* cp = (uint4*)ptr;
  double r;

  CK(cudaMemset(cp, 0, B)); CK(cudaDeviceSynchronize());
  if (read_rate(cp, B, sms, sink, &r)) return 1;
  printf("{\"probe\":\"A2\",\"alloc\":\"generic\",\"pattern\":\"zeros\",\"read_bytes_per_s\":%.4e}\n", r);
  if (kv_encode(cp, cp, B / 128, 0, 3, 0, 0)) { fprintf(stderr, "kv_encode\n"); return 1; }
  CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  if (read_rate(cp, B, sms, sink, &r)) return 1;
  printf("{\"probe\":\"A2\",\"alloc\":\"generic\",\"pattern\":\"zeros_enc\",\"read_bytes_per_s\":%.4e}\n", r);
  fflush(stdout);

  fill_kernel<<<sms * 4, 512>>>(cp, B / 16, 0, 0x5A5A5A5Au); CK(cudaDeviceSynchronize());
  if (read_rate(cp, B, sms, sink, &r)) return 1;
  printf("{\"probe\":\"A2\",\"alloc\":\"generic\",\"pattern\":\"const5a\",\"read_bytes_per_s\":%.4e}\n", r);
  const unsigned runs[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 65536};
  for (unsigned run : runs) {
    fill_kernel<<<sms * 4, 512>>>(cp, B / 16, run, 0); CK(cudaDeviceSynchronize());
    if (read_rate(cp, B, sms, sink, &r)) return 1;
    printf("{\"probe\":\"A2\",\"alloc\":\"generic\",\"pattern\":\"zrun_%u\",\"zero_fraction\":0.5,"
           "\"read_bytes_per_s\":%.4e}\n", run, r);
    fflush(stdout);
  }
  CKD(cuMemUnmap(ptr, size)); CKD(cuMemRelease(hnd)); CKD(cuMemAddressFree(ptr, size));

  // ---- B: working sets across launches (plain allocation) ----
  const size_t MB = 1ull << 20, BMAX = 1024 * MB;
  uint4* p; CK(cudaMalloc(&p, BMAX));
  fill_kernel<<<sms * 4, 512>>>(p, BMAX / 16, ALL_RANDOM, 0); CK(cudaDeviceSynchronize());
  cudaStream_t s; CK(cudaStreamCreate(&s));
  const size_t sizes[] = {1, 4, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 256, 384, 512, 1024};
  for (int persist = 0; persist < 2; persist++) {
    for (size_t mb : sizes) {
      const size_t X = mb * MB;
      if (persist) {
        CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)persist_max));
        cudaStreamAttrValue a = {};
        a.accessPolicyWindow.base_ptr = p;
        a.accessPolicyWindow.num_bytes = X < (size_t)persist_max ? X : (size_t)persist_max;
        a.accessPolicyWindow.hitRatio = 1.0f;
        a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &a));
      }
      if (resident_rate(p, X, sms, sink, s, &r)) return 1;
      printf("{\"probe\":\"B\",\"persist\":%d,\"set_mib\":%zu,\"read_bytes_per_s\":%.4e}\n", persist, mb, r);
      fflush(stdout);
    }
    if (persist) {
      cudaStreamAttrValue a = {};
      a.accessPolicyWindow.num_bytes = 0;
      CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &a));
      CK(cudaCtxResetPersistingL2Cache());
    }
  }
  return 0;
}
