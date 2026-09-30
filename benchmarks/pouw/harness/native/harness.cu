// The PoUW bench harness's device side (libpouw_harness.so): memory, operand fills, the read-only L2 flush, events and CUDA
// graphs, cuBLASLt's heuristic candidates and their calls, the gates' device helpers, and the card's characterization
// kernels (tensor-core peak rates, DRAM and L2 bandwidth).  A C ABI for ctypes; every call returns 0 or an error code whose
// text `ph_last_error` gives.  Build: build.sh (sm_120a).
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
thread_local char g_err[512];

int fail_cuda(cudaError_t e, const char* what) {
  snprintf(g_err, sizeof g_err, "%s: %s (%d)", what, cudaGetErrorString(e), int(e));
  return 1000 + int(e);
}
int fail_lt(cublasStatus_t s, const char* what) {
  snprintf(g_err, sizeof g_err, "%s: cublasStatus %d", what, int(s));
  return 2000 + int(s);
}
int fail_arg(const char* what) {
  snprintf(g_err, sizeof g_err, "%s", what);
  return 9;
}
#define CK(x)                                    \
  do {                                           \
    cudaError_t e_ = (x);                        \
    if (e_ != cudaSuccess) return fail_cuda(e_, #x); \
  } while (0)
#define LT(x)                                    \
  do {                                           \
    cublasStatus_t s_ = (x);                     \
    if (s_ != CUBLAS_STATUS_SUCCESS) return fail_lt(s_, #x); \
  } while (0)
}  // namespace

extern "C" {

const char* ph_last_error() { return g_err; }

// ---- device and memory -------------------------------------------------------------------------------------------------

struct ph_info {
  char name[256];
  unsigned char uuid[16];
  char pci_bus_id[32];
  int major, minor, sms, l2_bytes, smem_optin, persisting_l2_max, clock_khz_unused, mem_bus_width;
  unsigned long long total_mem;
  int runtime_version, driver_version;
  unsigned long long cublaslt_version;
};

int ph_init(int* device_count) {
  CK(cudaGetDeviceCount(device_count));
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  return 0;
}

int ph_device_info(ph_info* o) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  memset(o, 0, sizeof *o);
  strncpy(o->name, p.name, sizeof o->name - 1);
  memcpy(o->uuid, p.uuid.bytes, 16);
  CK(cudaDeviceGetPCIBusId(o->pci_bus_id, sizeof o->pci_bus_id, 0));
  o->major = p.major;
  o->minor = p.minor;
  o->sms = p.multiProcessorCount;
  o->l2_bytes = p.l2CacheSize;
  o->smem_optin = int(p.sharedMemPerBlockOptin);
  o->persisting_l2_max = p.persistingL2CacheMaxSize;
  o->mem_bus_width = p.memoryBusWidth;
  o->total_mem = p.totalGlobalMem;
  CK(cudaRuntimeGetVersion(&o->runtime_version));
  CK(cudaDriverGetVersion(&o->driver_version));
  o->cublaslt_version = cublasLtGetVersion();
  return 0;
}

int ph_malloc(void** p, size_t n) { CK(cudaMalloc(p, n < 256 ? 256 : n)); return 0; }
int ph_free(void* p) { CK(cudaFree(p)); return 0; }
int ph_mem_info(size_t* free_b, size_t* total_b) { CK(cudaMemGetInfo(free_b, total_b)); return 0; }
int ph_h2d(void* d, const void* h, size_t n) { CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice)); return 0; }
int ph_d2h(void* h, const void* d, size_t n) { CK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost)); return 0; }
int ph_d2h_2d(void* h, size_t hpitch, const void* d, size_t dpitch, size_t width, size_t rows) {
  CK(cudaMemcpy2D(h, hpitch, d, dpitch, width, rows, cudaMemcpyDeviceToHost));
  return 0;
}
int ph_memset(void* d, int v, size_t n) { CK(cudaMemset(d, v, n)); return 0; }
int ph_sync() { CK(cudaDeviceSynchronize()); return 0; }

// ---- streams, events, graphs -------------------------------------------------------------------------------------------

int ph_stream_create(void** s) { CK(cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t*>(s), cudaStreamNonBlocking)); return 0; }
int ph_stream_sync(void* s) { CK(cudaStreamSynchronize(cudaStream_t(s))); return 0; }
int ph_event_create(void** e) { CK(cudaEventCreate(reinterpret_cast<cudaEvent_t*>(e))); return 0; }
int ph_event_destroy(void* e) { CK(cudaEventDestroy(cudaEvent_t(e))); return 0; }
int ph_event_record(void* e, void* s) { CK(cudaEventRecord(cudaEvent_t(e), cudaStream_t(s))); return 0; }
int ph_event_sync(void* e) { CK(cudaEventSynchronize(cudaEvent_t(e))); return 0; }
int ph_event_ms(float* ms, void* a, void* b) { CK(cudaEventElapsedTime(ms, cudaEvent_t(a), cudaEvent_t(b))); return 0; }

// ThreadLocal capture: an arm that allocates, syncs or copies inside its timed call fails the capture instead of being timed.
int ph_capture_begin(void* s) { CK(cudaStreamBeginCapture(cudaStream_t(s), cudaStreamCaptureModeThreadLocal)); return 0; }
int ph_capture_end(void* s, void** exec, size_t* nodes) {
  cudaGraph_t g;
  CK(cudaStreamEndCapture(cudaStream_t(s), &g));
  CK(cudaGraphGetNodes(g, nullptr, nodes));
  cudaGraphExec_t x;
  CK(cudaGraphInstantiate(&x, g, 0));
  CK(cudaGraphDestroy(g));
  *exec = x;
  return 0;
}
int ph_graph_launch(void* exec, void* s) { CK(cudaGraphLaunch(cudaGraphExec_t(exec), cudaStream_t(s))); return 0; }
int ph_graph_destroy(void* exec) { CK(cudaGraphExecDestroy(cudaGraphExec_t(exec))); return 0; }

// ---- operand fills -----------------------------------------------------------------------------------------------------
// fill.py's numpy twin must give the same bytes (checked on the device by the preflight).

// Timing operands are full-entropy finite values of each format; the baselines' gate operands are ternary ({-1, 0, 1}),
// whose products and sums are exact under any accumulator the card could have (an aligned adder of 14 bits or more).
enum {
  FMT_F32 = 0, FMT_BF16 = 1, FMT_E4M3 = 2, FMT_E2M1 = 3, FMT_INT8 = 4,
  FMT_UE4M3_ONE = 5, FMT_UE8M0_ONE = 6, FMT_F32_ONE = 7, FMT_ZERO = 8,
  FMT_F32_TERNARY = 12, FMT_BF16_TERNARY = 13, FMT_E4M3_TERNARY = 14, FMT_E2M1_TERNARY = 15, FMT_INT8_TERNARY = 16,
};

__device__ __host__ inline uint64_t splitmix(uint64_t seed, uint64_t i) {
  uint64_t z = seed + (i + 1) * 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__device__ inline int ternary(uint64_t h, int shift) { return int((h >> shift) % 3u) - 1; }

__global__ void k_fill(int fmt, uint8_t* p, size_t units, uint64_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < units; i += size_t(gridDim.x) * blockDim.x) {
    uint64_t h = splitmix(seed, i);
    switch (fmt) {
      case FMT_F32: {  // sign, exponent 2^-5 .. 2^2, full mantissa
        uint32_t sign = uint32_t(h >> 63), exp = 122 + uint32_t((h >> 40) % 8u), mant = uint32_t(h) & 0x7FFFFF;
        reinterpret_cast<uint32_t*>(p)[i] = (sign << 31) | (exp << 23) | mant;
        break;
      }
      case FMT_BF16: {
        uint32_t sign = uint32_t(h >> 63), exp = 122 + uint32_t((h >> 40) % 8u), mant = uint32_t(h >> 16) & 0x7F;
        reinterpret_cast<uint16_t*>(p)[i] = uint16_t((sign << 15) | (exp << 7) | mant);
        break;
      }
      case FMT_E4M3: {
        uint8_t c = uint8_t(h >> 24);
        p[i] = (c & 0x7F) == 0x7F ? uint8_t(c - 1) : c;  // every finite code; 0x7F and 0xFF are NaN
        break;
      }
      case FMT_E2M1: p[i] = uint8_t(h >> 24); break;  // two E2M1 codes per byte, all 16 finite
      case FMT_INT8: p[i] = uint8_t(h >> 40); break;
      case FMT_UE4M3_ONE: p[i] = 0x38; break;
      case FMT_UE8M0_ONE: p[i] = 0x7F; break;
      case FMT_F32_ONE: reinterpret_cast<float*>(p)[i] = 1.0f; break;
      case FMT_F32_TERNARY: reinterpret_cast<float*>(p)[i] = float(ternary(h, 32)); break;
      case FMT_BF16_TERNARY: {
        int v = ternary(h, 32);
        reinterpret_cast<uint16_t*>(p)[i] = uint16_t(v == 0 ? 0 : v > 0 ? 0x3F80 : 0xBF80);
        break;
      }
      case FMT_E4M3_TERNARY: {
        int v = ternary(h, 32);
        p[i] = uint8_t(v == 0 ? 0 : v > 0 ? 0x38 : 0xB8);
        break;
      }
      case FMT_E2M1_TERNARY: {  // E2M1 1.0 is 0b0010, -1.0 0b1010
        int lo = ternary(h, 32), hi = ternary(h, 40);
        auto code = [](int v) { return uint8_t(v == 0 ? 0 : v > 0 ? 0x2 : 0xA); };
        p[i] = uint8_t(code(lo) | (code(hi) << 4));
        break;
      }
      case FMT_INT8_TERNARY: p[i] = uint8_t(int8_t(ternary(h, 32))); break;
      default: p[i] = 0;
    }
  }
}

int ph_unit_bytes(int fmt) {
  switch (fmt) {
    case FMT_F32: case FMT_F32_ONE: case FMT_F32_TERNARY: return 4;
    case FMT_BF16: case FMT_BF16_TERNARY: return 2;
    default: return 1;
  }
}

// Fills `units` elements (bytes for E2M1: two codes each) of format `fmt` at p from `seed`.
int ph_fill(int fmt, void* p, size_t units, uint64_t seed, void* stream) {
  k_fill<<<2048, 256, 0, cudaStream_t(stream)>>>(fmt, static_cast<uint8_t*>(p), units, seed);
  CK(cudaGetLastError());
  return 0;
}

// ---- the L2 flush (read-only) and the bandwidth kernels ------------------------------------------------------------------

__global__ void k_read(const uint4* __restrict__ p, size_t n16, int passes, unsigned* sink) {
  unsigned acc = 0;
  for (int r = 0; r < passes; ++r)
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n16; i += size_t(gridDim.x) * blockDim.x) {
      uint4 v = __ldcg(p + i);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  if (acc == 0x9E3779B9u) sink[0] = acc;  // never true for a zeroed buffer; keeps the loads live
}

__global__ void k_copy(uint4* __restrict__ d, const uint4* __restrict__ s, size_t n16) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n16; i += size_t(gridDim.x) * blockDim.x) d[i] = s[i];
}

int ph_sms() {
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0);
  return v;
}

// Reads `bytes` of p `passes` times (the flush: a buffer zeroed once, 4x the L2, one pass; so it leaves no dirty lines).
int ph_read(const void* p, size_t bytes, int passes, void* sink, int blocks_per_sm, void* stream) {
  k_read<<<ph_sms() * blocks_per_sm, 512, 0, cudaStream_t(stream)>>>(static_cast<const uint4*>(p), bytes / 16, passes,
                                                                     static_cast<unsigned*>(sink));
  CK(cudaGetLastError());
  return 0;
}

int ph_copy(void* d, const void* s, size_t bytes, int blocks_per_sm, void* stream) {
  k_copy<<<ph_sms() * blocks_per_sm, 512, 0, cudaStream_t(stream)>>>(static_cast<uint4*>(d), static_cast<const uint4*>(s),
                                                                     bytes / 16);
  CK(cudaGetLastError());
  return 0;
}

// ---- gate helpers ------------------------------------------------------------------------------------------------------

__global__ void k_count_eq(const uint8_t* p, size_t words, int wb, uint32_t pattern, unsigned long long* out) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t w = wb == 4 ? reinterpret_cast<const uint32_t*>(p)[i] : wb == 2 ? reinterpret_cast<const uint16_t*>(p)[i] : p[i];
    c += (w == pattern);
  }
  atomicAdd(out, c);
}

// How many `wb`-byte words of p equal `pattern` (the sentinel a gate fills an output with before the gated launch).
int ph_count_eq(const void* p, size_t words, int wb, uint32_t pattern, unsigned long long* count) {
  unsigned long long* d;
  CK(cudaMalloc(&d, 8));
  CK(cudaMemset(d, 0, 8));
  k_count_eq<<<ph_sms() * 4, 256>>>(static_cast<const uint8_t*>(p), words, wb, pattern, d);
  CK(cudaGetLastError());
  CK(cudaMemcpy(count, d, 8, cudaMemcpyDeviceToHost));
  CK(cudaFree(d));
  return 0;
}

__global__ void k_xor_byte(uint8_t* p, size_t off, uint8_t mask) { p[off] ^= mask; }

// The negative control: flips bits `mask` of byte `off` of p, in stream order.
int ph_xor_byte(void* p, size_t off, int mask, void* stream) {
  k_xor_byte<<<1, 1, 0, cudaStream_t(stream)>>>(static_cast<uint8_t*>(p), off, uint8_t(mask));
  CK(cudaGetLastError());
  return 0;
}

// ---- tensor-core peak rates ----------------------------------------------------------------------------------------------
// Each warp issues CHAINS independent MMAs per iteration (no dependency stalls), `iters` times; the result is stored only
// when a runtime flag says so, so nothing is elided.  FLOPs per MMA: 2*M*N*K (m16n8k16 BF16/F16, m16n8k32 E4M3/E2M1/S8,
// m16n8k64 block-scaled E2M1).

}  // extern "C"

enum { MMA_BF16_F32 = 0, MMA_F16_F16 = 1, MMA_E4M3_F32 = 2, MMA_E4M3_F16 = 3, MMA_F8F6F4_E4M3_F32 = 4,
       MMA_F8F6F4_E2M1_F32 = 5, MMA_MXF8_E4M3_F32 = 6, MMA_NVF4_E2M1_F32 = 7, MMA_MXF4_E2M1_F32 = 8, MMA_S8_S32 = 9,
       MMA_KINDS = 10 };

constexpr int CHAINS = 8;

template <int KIND>
__device__ __forceinline__ void mma_step(float (&d)[CHAINS][4], uint32_t (&h)[CHAINS][2], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2], uint32_t sf) {
#pragma unroll
  for (int c = 0; c < CHAINS; ++c) {
    if constexpr (KIND == MMA_BF16_F32) {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (KIND == MMA_F16_F16) {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                   : "+r"(h[c][0]), "+r"(h[c][1])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (KIND == MMA_E4M3_F32) {
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (KIND == MMA_E4M3_F16) {
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                   : "+r"(h[c][0]), "+r"(h[c][1])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (KIND == MMA_F8F6F4_E4M3_F32) {
      asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (KIND == MMA_F8F6F4_E2M1_F32) {
      asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (KIND == MMA_MXF8_E4M3_F32) {
      asm volatile(
          "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%10}, {%11,%12};\n"
          : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf), "h"(uint16_t(0)), "h"(uint16_t(0)));
    } else if constexpr (KIND == MMA_NVF4_E2M1_F32) {
      asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%10}, {%11,%12};\n"
          : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf), "h"(uint16_t(0)), "h"(uint16_t(0)));
    } else if constexpr (KIND == MMA_MXF4_E2M1_F32) {
      asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%10}, {%11,%12};\n"
          : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf), "h"(uint16_t(0)), "h"(uint16_t(0)));
    } else if constexpr (KIND == MMA_S8_S32) {
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+r"(reinterpret_cast<uint32_t&>(d[c][0])), "+r"(reinterpret_cast<uint32_t&>(d[c][1])),
                     "+r"(reinterpret_cast<uint32_t&>(d[c][2])), "+r"(reinterpret_cast<uint32_t&>(d[c][3]))
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
  }
}

template <int KIND>
__global__ void k_mma_peak(int iters, uint32_t seed, int store, float* out) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i) a[i] = uint32_t(splitmix(seed, threadIdx.x * 8 + i)) & 0x3C3C3C3Cu;
  for (int i = 0; i < 2; ++i) b[i] = uint32_t(splitmix(seed, threadIdx.x * 8 + 4 + i)) & 0x3C3C3C3Cu;
  uint32_t sf = KIND == MMA_NVF4_E2M1_F32 ? 0x38383838u : 0x7F7F7F7Fu;
  float d[CHAINS][4] = {};
  uint32_t h[CHAINS][2] = {};
  for (int it = 0; it < iters; ++it) mma_step<KIND>(d, h, a, b, sf);
  if (store) {
    float s = 0;
    for (int c = 0; c < CHAINS; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3] + __uint_as_float(h[c][0] ^ h[c][1]);
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

typedef void (*mma_kernel_t)(int, uint32_t, int, float*);
static const mma_kernel_t MMA_KERNELS[MMA_KINDS] = {
    k_mma_peak<0>, k_mma_peak<1>, k_mma_peak<2>, k_mma_peak<3>, k_mma_peak<4>,
    k_mma_peak<5>, k_mma_peak<6>, k_mma_peak<7>, k_mma_peak<8>, k_mma_peak<9>};
static const int MMA_K[MMA_KINDS] = {16, 16, 32, 32, 32, 32, 32, 64, 64, 32};

extern "C" {

// FLOPs one warp-level MMA of `kind` does (2*16*8*K), or 0 for an unknown kind.
long long ph_mma_flops(int kind) { return kind < 0 || kind >= MMA_KINDS ? 0 : 2LL * 16 * 8 * MMA_K[kind]; }
int ph_mma_chains() { return CHAINS; }

// Launches blocks x threads of the `kind` peak kernel, `iters` iterations of CHAINS MMAs per warp, and times it.
int ph_mma_peak(int kind, int blocks, int threads, int iters, float* out_scratch, float* ms, void* stream) {
  if (kind < 0 || kind >= MMA_KINDS) return fail_arg("ph_mma_peak: unknown kind");
  cudaStream_t s = cudaStream_t(stream);
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a, s));
  MMA_KERNELS[kind]<<<blocks, threads, 0, s>>>(iters, 12345u, 0, out_scratch);
  CK(cudaGetLastError());
  CK(cudaEventRecord(b, s));
  CK(cudaEventSynchronize(b));
  CK(cudaEventElapsedTime(ms, a, b));
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return 0;
}

// ---- cuBLASLt ------------------------------------------------------------------------------------------------------------
// The harness's D (m x n, row-major) = A (m x k, row-major) . B^T, B (n x k, row-major): in cuBLASLt's column-major terms
// D^T = op(B) . A^T with op = T, so cuBLASLt's (m, n, k) is our (n, m, k), its A is our B and its B our A ("TN").

enum { LT_BF16 = 0, LT_FP8 = 1, LT_FP8_ROWWISE = 2, LT_MXFP8 = 3, LT_FP8_BLK128 = 4, LT_NVFP4 = 5, LT_MXFP4 = 6, LT_INT8 = 7,
       LT_KINDS = 8 };

struct ph_lt_plan {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c, d;
  cublasLtMatmulPreference_t pref;
  int kind, m, n, k;
  size_t ws;
};

struct ph_lt_candidate {
  cublasLtMatmulAlgo_t algo;
  size_t workspace;
  float waves;
  int algo_id, tile, stages, splitk, reduction, swizzle, custom, inner_shape, cluster_shape;
};

static cudaDataType_t lt_in(int kind) {
  switch (kind) {
    case LT_BF16: return CUDA_R_16BF;
    case LT_NVFP4: case LT_MXFP4: return CUDA_R_4F_E2M1;
    case LT_INT8: return CUDA_R_8I;
    default: return CUDA_R_8F_E4M3;
  }
}

// Creates the plan for our (m, n, k), rows `ld` elements apart (ld = k but for the known-bad half-k plan); `ws` bytes of
// workspace are allowed.  Scale modes: per kind (see the enum).
int ph_lt_plan_create_ld(int kind, int m, int n, int k, int ld, size_t ws, int swap_blk, ph_lt_plan** out) {
  if (kind < 0 || kind >= LT_KINDS) return fail_arg("ph_lt_plan_create: unknown kind");
  ph_lt_plan* p = new ph_lt_plan();
  p->kind = kind; p->m = m; p->n = n; p->k = k; p->ws = ws;
  LT(cublasLtCreate(&p->h));
  bool i8 = kind == LT_INT8;
  LT(cublasLtMatmulDescCreate(&p->op, i8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F, i8 ? CUDA_R_32I : CUDA_R_32F));
  cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
  LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
  LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
  int32_t sa = -1, sb = -1;
  switch (kind) {
    case LT_FP8_ROWWISE: sa = sb = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F; break;
    case LT_MXFP8: case LT_MXFP4: sa = sb = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0; break;
    case LT_NVFP4: sa = sb = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3; break;
    case LT_FP8_BLK128:  // cuBLASLt's A is our weight B: 128x128 blocks on it and 1x128 on our A, or the reverse
      sa = swap_blk ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F : CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
      sb = swap_blk ? CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F : CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
      break;
  }
  if (sa >= 0) {
    LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sa, sizeof sa));
    LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sb, sizeof sb));
  }
  cudaDataType_t in = lt_in(kind), outt = i8 ? CUDA_R_32I : CUDA_R_16BF;
  LT(cublasLtMatrixLayoutCreate(&p->a, in, k, n, ld));     // our B, n x k row-major = k x n column-major, op T
  LT(cublasLtMatrixLayoutCreate(&p->b, in, k, m, ld));     // our A, m x k row-major = k x m column-major
  LT(cublasLtMatrixLayoutCreate(&p->c, outt, n, m, n));
  LT(cublasLtMatrixLayoutCreate(&p->d, outt, n, m, n));
  LT(cublasLtMatmulPreferenceCreate(&p->pref));
  LT(cublasLtMatmulPreferenceSetAttribute(p->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  *out = p;
  return 0;
}

int ph_lt_plan_create(int kind, int m, int n, int k, size_t ws, int swap_blk, ph_lt_plan** out) {
  return ph_lt_plan_create_ld(kind, m, n, k, k, ws, swap_blk, out);
}

int ph_lt_plan_destroy(ph_lt_plan* p) {
  cublasLtMatmulPreferenceDestroy(p->pref);
  cublasLtMatrixLayoutDestroy(p->a);
  cublasLtMatrixLayoutDestroy(p->b);
  cublasLtMatrixLayoutDestroy(p->c);
  cublasLtMatrixLayoutDestroy(p->d);
  cublasLtMatmulDescDestroy(p->op);
  cublasLtDestroy(p->h);
  delete p;
  return 0;
}

static int set_scales(ph_lt_plan* p, const void* a_scale, const void* b_scale) {
  if (p->kind == LT_BF16 || p->kind == LT_INT8) return 0;
  // cuBLASLt's A is our B: its A scale is our B's
  LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &b_scale, sizeof b_scale));
  LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_scale, sizeof a_scale));
  return 0;
}

// Up to `want` heuristic candidates (cuBLASLt's order), their ids and configs; *got is how many.
int ph_lt_candidates(ph_lt_plan* p, const void* a_scale, const void* b_scale, int want, ph_lt_candidate* out, int* got) {
  if (int r = set_scales(p, a_scale, b_scale)) return r;
  cublasLtMatmulHeuristicResult_t res[64];
  if (want > 64) want = 64;
  *got = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(p->h, p->op, p->a, p->b, p->c, p->d, p->pref, want, res, got);
  if (s == CUBLAS_STATUS_NOT_SUPPORTED) { *got = 0; return 0; }
  if (s != CUBLAS_STATUS_SUCCESS) return fail_lt(s, "cublasLtMatmulAlgoGetHeuristic");
  for (int i = 0; i < *got; ++i) {
    ph_lt_candidate& c = out[i];
    memset(&c, 0, sizeof c);
    c.algo = res[i].algo;
    c.workspace = res[i].workspaceSize;
    c.waves = res[i].wavesCount;
    size_t w;
    cublasLtMatmulAlgoConfigGetAttribute(&c.algo, CUBLASLT_ALGO_CONFIG_ID, &c.algo_id, sizeof c.algo_id, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&c.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &c.tile, sizeof c.tile, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&c.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &c.stages, sizeof c.stages, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&c.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &c.splitk, sizeof c.splitk, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&c.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &c.reduction, sizeof c.reduction, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&c.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &c.swizzle, sizeof c.swizzle, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&c.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &c.custom, sizeof c.custom, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&c.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &c.inner_shape, sizeof c.inner_shape, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&c.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &c.cluster_shape, sizeof c.cluster_shape, &w);
  }
  return 0;
}

// D = A . B^T (alpha 1, beta 0) with candidate `c`, on `stream`.  Scale pointers are per operand set.
int ph_lt_run(ph_lt_plan* p, const ph_lt_candidate* c, const void* A, const void* B, void* D, const void* a_scale,
              const void* b_scale, void* ws, void* stream) {
  if (int r = set_scales(p, a_scale, b_scale)) return r;
  float one = 1.0f, zero = 0.0f;
  int32_t ione = 1, izero = 0;
  const void* alpha = p->kind == LT_INT8 ? static_cast<const void*>(&ione) : &one;
  const void* beta = p->kind == LT_INT8 ? static_cast<const void*>(&izero) : &zero;
  LT(cublasLtMatmul(p->h, p->op, alpha, B, p->a, A, p->b, beta, D, p->c, D, p->d, &c->algo, ws, p->ws,
                    cudaStream_t(stream)));
  return 0;
}

int ph_lt_candidate_size() { return int(sizeof(ph_lt_candidate)); }
int ph_info_size() { return int(sizeof(ph_info)); }

}  // extern "C"
