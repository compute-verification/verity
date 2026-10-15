// Attack A (compression): does NVIDIA generic memory compression (CU_MEM_ALLOCATION_COMP_GENERIC) reduce the HBM
// traffic of reading the real Sloth-encoded store?  If it does, a node earns receipt credit while moving far fewer HBM
// bytes than the ledger's R·Σ assumes, so the audit over-credits useful memory bandwidth.  MEM-22 (r20261015-043430-0a05)
// measured a synthetic hash pattern as a stand-in for "random"; this measures the actual registered codec's store, built
// on the device by the deployed `kv.cu` (p1024/k3/masked) under a key from the repository's `codec`.
//
// For a 4 GiB buffer under a plain cudaMalloc and under a compressible VMM allocation, the best read rate of:
//   zeros        the compression ceiling (one sector compresses to nothing)
//   hash         MEM-22's `k_fill` pattern (a weak PRNG), for comparison with its run
//   rand         a canonical random payload, NOT encoded (random is already incompressible)
//   rand_enc     that payload encoded: a generic store
//   lowent       a low-entropy payload (half the bytes zero), NOT encoded: a compressible pre-image (e.g. weights)
//   lowent_enc   that same compressible payload encoded: the store a prover would hold if it kept the cheap pre-image
// A higher read rate means less HBM traffic, i.e. the controller compressed the data.  The headline is whether `rand_enc`
// and `lowent_enc` under the compressible allocation read at the plain rate (incompressible: the codec defends) or near
// the zeros ceiling (compressible: an attack).  A separate small dump lets `codec_key.py check` confirm byte-for-byte
// that what the device built is the registered codec's store.
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

enum Pattern { PAT_ZERO, PAT_HASH, PAT_RAND, PAT_LOWENT };

__device__ __forceinline__ unsigned mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return (unsigned)(z ^ (z >> 31));
}

// A thread a uint4 (8 per 128-byte codec word).  PAT_RAND/PAT_LOWENT keep every word canonical (< p) by zeroing the top
// byte of the word's last uint4, so `kv_encode` accepts them.
__global__ void fill_kernel(uint4* p, size_t n16, int pat) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    const bool top = (i & 7) == 7;
    uint4 v;
    if (pat == PAT_HASH) {
      unsigned h = (unsigned)i * 0x9e3779b9u;
      v = make_uint4(h ^ 0x85ebca6bu, h * 3u, h ^ (h >> 13), ~h);
    } else if (pat == PAT_LOWENT) {
      unsigned h = mix(i);
      v = make_uint4(h, 0u, mix(i * 3 + 1), 0u);   // half the bytes zero: a compressible pre-image
    } else {  // PAT_RAND
      v = make_uint4(mix(i), mix(i ^ 0x1u), mix(i ^ 0x2u), mix(i ^ 0x3u));
    }
    if (top && pat != PAT_HASH) v.w &= 0x00FFFFFFu;   // word < 2^(1024-8) < p
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

// Fill `p` with `pat` and, if `enc`, encode it in place as the store (base 0, k=3).  `sms*4` CTAs for the fill.
static int build(uint4* p, size_t bytes, int sms, int pat, bool enc) {
  if (pat == PAT_ZERO) { CK(cudaMemset(p, 0, bytes)); }
  else fill_kernel<<<sms * 4, 512>>>(p, bytes / 16, pat);
  CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  if (enc) {
    const uint64_t words = bytes / 128;   // 128-byte p1024 words
    if (kv_encode(p, p, words, 0, 3, 0, 0)) { fprintf(stderr, "kv_encode failed\n"); return 1; }
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  }
  return 0;
}

static int set_key(const char* path) {
  uint8_t key[5 * 128];
  FILE* f = fopen(path, "rb");
  if (!f || fread(key, 1, sizeof key, f) != sizeof key) { fprintf(stderr, "key.bin: %s\n", path); return 1; }
  fclose(f);
  const uint32_t* w = reinterpret_cast<const uint32_t*>(key);   // 32 u32 LE a word
  if (kv_set_key(w, 3, w + 3 * 32) || kv_set_mask(w + 4 * 32)) { fprintf(stderr, "kv_set_key/mask\n"); return 1; }
  return 0;
}

// Byte-exact dump for `codec_key.py check`: blocks 0 … count-1, each (u64le j, payload, encoded), to `path`.
static int dump_blocks(const char* path, int count) {
  const size_t words = (size_t)count * 32, bytes = words * 128;
  uint4 *buf; CK(cudaMalloc(&buf, bytes));
  int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  fill_kernel<<<sms, 512>>>(buf, bytes / 16, PAT_RAND); CK(cudaDeviceSynchronize());
  uint8_t* payload = (uint8_t*)malloc(bytes); CK(cudaMemcpy(payload, buf, bytes, cudaMemcpyDeviceToHost));
  if (kv_encode(buf, buf, words, 0, 3, 0, 0)) return 1;
  CK(cudaDeviceSynchronize());
  uint8_t* enc = (uint8_t*)malloc(bytes); CK(cudaMemcpy(enc, buf, bytes, cudaMemcpyDeviceToHost));
  FILE* f = fopen(path, "wb");
  for (int j = 0; j < count; j++) {
    uint64_t jj = j; fwrite(&jj, 8, 1, f);
    fwrite(payload + (size_t)j * 4096, 1, 4096, f);
    fwrite(enc + (size_t)j * 4096, 1, 4096, f);
  }
  fclose(f); free(payload); free(enc); CK(cudaFree(buf));
  return 0;
}

static const char* NAMES[] = {"zeros", "hash", "rand", "rand_enc", "lowent", "lowent_enc"};
static const int PATS[] = {PAT_ZERO, PAT_HASH, PAT_RAND, PAT_RAND, PAT_LOWENT, PAT_LOWENT};
static const bool ENC[] = {false, false, false, true, false, true};

static int sweep(const char* alloc, uint4* p, size_t B, int sms, unsigned* sink) {
  for (int c = 0; c < 6; c++) {
    if (build(p, B, sms, PATS[c], ENC[c])) return 1;
    double r; if (read_rate(p, B, sms, sink, &r)) return 1;
    printf("{\"probe\":\"attackA\",\"alloc\":\"%s\",\"pattern\":\"%s\",\"read_bytes_per_s\":%.4e}\n", alloc, NAMES[c], r);
    fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  CK(cudaFree(0));
  const char* keyfile = argc > 1 ? argv[1] : "key.bin";
  const char* dumpfile = argc > 2 ? argv[2] : "store_dump.bin";
  if (set_key(keyfile)) return 1;

  CUdevice dev; CKD(cuDeviceGet(&dev, 0));
  int comp = 0; cuDeviceGetAttribute(&comp, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, dev);
  int sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  printf("{\"probe\":\"attackA-props\",\"generic_compression\":%d,\"sms\":%d}\n", comp, sms);

  if (dump_blocks(dumpfile, 8)) return 1;

  unsigned* sink; CK(cudaMalloc(&sink, 4));
  const size_t B = 4ull << 30;

  uint4* plain; CK(cudaMalloc(&plain, B));
  if (sweep("plain", plain, B, sms, sink)) return 1;
  CK(cudaFree(plain));

  if (!comp) { printf("{\"probe\":\"attackA\",\"alloc\":\"generic\",\"supported\":false}\n"); return 0; }
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
  printf("{\"probe\":\"attackA\",\"alloc\":\"generic\",\"granted_compression\":%d}\n", (int)got.allocFlags.compressionType);
  CUdeviceptr ptr;
  CKD(cuMemAddressReserve(&ptr, size, 0, 0, 0));
  CKD(cuMemMap(ptr, size, 0, hnd, 0));
  CUmemAccessDesc acc = {};
  acc.location = prop.location;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CKD(cuMemSetAccess(ptr, size, &acc, 1));
  if (sweep("generic", (uint4*)ptr, B, sms, sink)) return 1;
  return 0;
}
