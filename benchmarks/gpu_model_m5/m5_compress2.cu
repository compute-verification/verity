// Axiom probe: which contents does generic memory compression turn into reads above the DRAM byte budget? A 4 GiB
// CU_MEM_ALLOCATION_COMP_GENERIC allocation is filled with each pattern and read (128-bit loads, best of 5), beside a
// plain allocation with the same pattern:
//   zero, const (one 32-bit word repeated), line-const (each 128-byte line one random word repeated),
//   bytes16 / bytes2 (each byte uniform over 16 / 2 values), e4m3 (Gaussian weights quantized to E4M3, about 6.5 bits
//   a byte), hash (pseudorandom, the encoded store's stand-in).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)
#define CKD(x) do { CUresult e_ = (x); if (e_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorString(e_, &s_); \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, s_); return 1; } } while (0)

__device__ __forceinline__ unsigned mix(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}

// Round to the nearest E4M3 value's code (saturating at 448), sign-magnitude.
__device__ unsigned e4m3_code(float v) {
  unsigned s = v < 0 ? 0x80u : 0u;
  float a = fabsf(v);
  if (a >= 448.f) return s | 0x7eu;
  unsigned best = 0; float bd = 1e30f;
  for (unsigned c = 0; c < 0x7f; c++) {
    unsigned e = c >> 3, m = c & 7;
    float x = e == 0 ? ldexpf(m / 8.f, -6) : ldexpf(1.f + m / 8.f, (int)e - 7);
    float d = fabsf(x - a);
    if (d < bd) { bd = d; best = c; }
  }
  return s | best;
}

__global__ void k_fill(unsigned* p, size_t n, int pat) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nth = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n; i += nth) {
    unsigned h = mix((unsigned)i ^ mix((unsigned)(i >> 32) + 0x9e3779b9u)), w = 0;
    switch (pat) {
      case 0: w = 0; break;
      case 1: w = 0x3c3c3c3cu; break;
      case 2: w = mix((unsigned)(i / 32) + 12345u); break;
      case 3: w = h & 0x0f0f0f0fu; break;
      case 4: w = h & 0x01010101u; break;
      case 5: {
        for (int b = 0; b < 4; b++) {
          unsigned u1 = mix(h + 2 * b) | 1u, u2 = mix(h + 2 * b + 1);
          float g = sqrtf(-2.f * logf(u1 / 4294967296.f)) * cospif(2.f * (u2 / 4294967296.f));
          w |= e4m3_code(g * 16.f) << (8 * b);
        }
        break;
      }
      default: w = h; break;
    }
    p[i] = w;
  }
}

__global__ void k_read(const uint4* __restrict__ p, size_t n, unsigned* sink) {
  unsigned acc = 0;
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nth = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n; i += nth) { uint4 v = __ldg(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

static int read_rate(const void* p, size_t bytes, int sms, unsigned* sink, double* out) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double best = 0;
  for (int rep = 0; rep < 6; rep++) {
    CK(cudaEventRecord(e0));
    k_read<<<sms * 8, 512>>>((const uint4*)p, bytes / 16, sink);
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
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const size_t B = 4ull << 30;
  unsigned* sink;
  CK(cudaMalloc(&sink, 4));
  void* plain;
  CK(cudaMalloc(&plain, B));
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
  CUdeviceptr ptr;
  CKD(cuMemAddressReserve(&ptr, size, 0, 0, 0));
  CKD(cuMemMap(ptr, size, 0, hnd, 0));
  CUmemAccessDesc acc = {};
  acc.location = prop.location;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CKD(cuMemSetAccess(ptr, size, &acc, 1));
  size_t freeb = 0, totalb = 0;
  CK(cudaMemGetInfo(&freeb, &totalb));
  printf("{\"probe\":\"compress2-footprint\",\"free_after_both_bytes\":%zu,\"total_bytes\":%zu,\"alloc_bytes\":%zu}\n",
         freeb, totalb, size);
  const char* names[7] = {"zero", "const", "line-const", "bytes16", "bytes2", "e4m3", "hash"};
  for (int pat = 0; pat < 7; pat++) {
    double rp, rc;
    k_fill<<<sms * 4, 512>>>((unsigned*)plain, B / 4, pat);
    k_fill<<<sms * 4, 512>>>((unsigned*)ptr, B / 4, pat);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    if (read_rate(plain, B, sms, sink, &rp)) return 1;
    if (read_rate((void*)ptr, B, sms, sink, &rc)) return 1;
    printf("{\"probe\":\"compress2\",\"pattern\":\"%s\",\"plain_read_bytes_per_s\":%.4e,"
           "\"generic_read_bytes_per_s\":%.4e,\"ratio\":%.3f}\n", names[pat], rp, rc, rc / rp);
  }
  return 0;
}
