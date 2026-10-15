// Axiom probe: the device's properties the model's instance reads (SMs, L2, shared memory, registers, persisting L2,
// memory), and whether generic memory compression lets a read of compressible data beat the DRAM byte budget.
// If CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED is set, 4 GiB is allocated with CU_MEM_ALLOCATION_COMP_GENERIC
// and read when all zero and when filled with a hash pattern, beside a plain allocation; else that is reported.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)
#define CKD(x) do { CUresult e_ = (x); if (e_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorString(e_, &s_); \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, s_); return 1; } } while (0)

__global__ void k_read(const uint4* __restrict__ p, size_t n, unsigned* sink) {
  unsigned acc = 0;
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nth = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n; i += nth) { uint4 v = __ldg(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

__global__ void k_fill(uint4* p, size_t n) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nth = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n; i += nth) {
    unsigned h = (unsigned)i * 0x9e3779b9u;
    p[i] = make_uint4(h ^ 0x85ebca6bu, h * 3u, h ^ (h >> 13), ~h);
  }
}

static int read_rate(const uint4* p, size_t bytes, int sms, unsigned* sink, double* out) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double best = 0;
  for (int rep = 0; rep < 6; rep++) {
    CK(cudaEventRecord(e0));
    k_read<<<sms * 8, 512>>>(p, bytes / 16, sink);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    double r = bytes / (ms / 1e3);
    if (rep > 0 && r > best) best = r;
  }
  *out = best;
  return 0;
}

int main() {
  CK(cudaFree(0));
  CUdevice dev;
  CKD(cuDeviceGet(&dev, 0));
  struct { const char* name; CUdevice_attribute a; } attrs[] = {
      {"sms", CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT},
      {"l2_bytes", CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE},
      {"persisting_l2_max_bytes", CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE},
      {"smem_per_sm_bytes", CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR},
      {"smem_per_block_optin_bytes", CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN},
      {"regs_per_sm", CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR},
      {"max_threads_per_sm", CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR},
      {"mem_bus_bits", CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH},
      {"async_engines", CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT},
      {"generic_compression", CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED},
      {"cc_major", CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR},
      {"cc_minor", CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR},
  };
  int sms = 0, comp = 0;
  printf("{\"probe\":\"props\"");
  for (auto& x : attrs) {
    int v = -1;
    cuDeviceGetAttribute(&v, x.a, dev);
    printf(",\"%s\":%d", x.name, v);
    if (x.a == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) sms = v;
    if (x.a == CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED) comp = v;
  }
  size_t free_b = 0, total_b = 0;
  CK(cudaMemGetInfo(&free_b, &total_b));
  printf(",\"mem_total_bytes\":%zu}\n", total_b);

  unsigned* sink;
  CK(cudaMalloc(&sink, 4));
  const size_t B = 4ull << 30;
  uint4* plain;
  CK(cudaMalloc(&plain, B));
  double r_plain_zero, r_plain_hash;
  CK(cudaMemset(plain, 0, B)); CK(cudaDeviceSynchronize());
  if (read_rate(plain, B, sms, sink, &r_plain_zero)) return 1;
  k_fill<<<sms * 4, 512>>>(plain, B / 16); CK(cudaDeviceSynchronize());
  if (read_rate(plain, B, sms, sink, &r_plain_hash)) return 1;
  printf("{\"probe\":\"compress\",\"alloc\":\"plain\",\"zero_read_bytes_per_s\":%.4e,\"hash_read_bytes_per_s\":%.4e}\n",
         r_plain_zero, r_plain_hash);
  if (!comp) {
    printf("{\"probe\":\"compress\",\"alloc\":\"generic\",\"supported\":false}\n");
    return 0;
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
  size_t gran = 0;
  CKD(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  size_t size = (B + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle hnd;
  CKD(cuMemCreate(&hnd, size, &prop, 0));
  CUmemAllocationProp got = {};
  CKD(cuMemGetAllocationPropertiesFromHandle(&got, hnd));
  CUdeviceptr ptr;
  CKD(cuMemAddressReserve(&ptr, size, 0, 0, 0));
  CKD(cuMemMap(ptr, size, 0, hnd, 0));
  CUmemAccessDesc acc = {};
  acc.location = prop.location;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CKD(cuMemSetAccess(ptr, size, &acc, 1));
  uint4* cp = (uint4*)ptr;
  double r_zero, r_hash;
  CK(cudaMemset(cp, 0, B)); CK(cudaDeviceSynchronize());
  if (read_rate(cp, B, sms, sink, &r_zero)) return 1;
  k_fill<<<sms * 4, 512>>>(cp, B / 16); CK(cudaDeviceSynchronize());
  if (read_rate(cp, B, sms, sink, &r_hash)) return 1;
  printf("{\"probe\":\"compress\",\"alloc\":\"generic\",\"supported\":true,\"granted\":%d,"
         "\"zero_read_bytes_per_s\":%.4e,\"hash_read_bytes_per_s\":%.4e}\n",
         (int)got.allocFlags.compressionType, r_zero, r_hash);
  return 0;
}
