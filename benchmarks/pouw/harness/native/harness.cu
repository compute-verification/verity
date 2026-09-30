// The PoUW bench harness's device side (libpouw_harness.so): memory, operand fills, the read-only L2 flush, events and CUDA
// graphs, the dependent chain's derive, the gates' device helpers, and the card's characterization kernels (tensor-core peak
// rates, DRAM and L2 bandwidth); cuBLASLt is lt.cu's.  A C ABI for ctypes; every call returns 0 or an error code whose text
// `ph_last_error` gives.  Build: build.sh (sm_120a).
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

thread_local char ph_err[512];  // lt.cu writes it too

namespace {
int fail_cuda(cudaError_t e, const char* what) {
  snprintf(ph_err, sizeof ph_err, "%s: %s (%d)", what, cudaGetErrorString(e), int(e));
  return 1000 + int(e);
}
int fail_arg(const char* what) {
  snprintf(ph_err, sizeof ph_err, "%s", what);
  return 9;
}
#define CK(x)                                    \
  do {                                           \
    cudaError_t e_ = (x);                        \
    if (e_ != cudaSuccess) return fail_cuda(e_, #x); \
  } while (0)
}  // namespace

extern "C" {

const char* ph_last_error() { return ph_err; }

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
// Stream s waits for event e's last record (inside a capture: a fork or join between the captured streams).
int ph_stream_wait_event(void* s, void* e) { CK(cudaStreamWaitEvent(cudaStream_t(s), cudaEvent_t(e), 0)); return 0; }

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

// ---- the dependent chain's derive ----------------------------------------------------------------------------------------
// The next call's A (m x k) from this call's y (m x n, row-major): row r of A reads row r of y, column c reads column c mod n.
// Each row is scaled by 2^(T - E), where the row's amax is in [2^(E-1), 2^E) (no scaling for a zero row), so |x| < 2^T and the
// chain's values stay bounded over any number of calls; then converted to the consumer's format, rounding to nearest even and
// saturating, with real block scales (UE8M0 per 32, UE4M3 per 16) in the 128 x 4 tiled layout.  derive.py's numpy twin must
// give the same bytes (checked on the device by the preflight, and on every chain by its gate).

// It is in the plain GEMM's divisor too, so it is kept to about a launch: a block per row, 8 columns per thread in registers,
// 16-byte loads and stores where y and A allow them, block amaxes by lane shuffles.

enum { DY_BF16 = 0, DY_INT32 = 1, DY_F32 = 2 };
enum { DV_F32 = 0, DV_BF16 = 1, DV_E4M3 = 2, DV_MXFP8 = 3, DV_NVFP4 = 4, DV_MXFP4 = 5, DV_INT8 = 6, DV_KINDS = 7 };
constexpr int DERIVE_MAX_THREADS = 1024;

__device__ __forceinline__ int derive_t(int kind) {
  switch (kind) {
    case DV_E4M3: return 8;
    case DV_NVFP4: return 11;
    case DV_INT8: return 7;
    case DV_MXFP8: case DV_MXFP4: return 0;
    default: return 2;
  }
}

__device__ __forceinline__ float dy_load(int src, const void* y, size_t i) {
  if (src == DY_BF16) return __uint_as_float(uint32_t(static_cast<const uint16_t*>(y)[i]) << 16);
  if (src == DY_INT32) return __int2float_rn(static_cast<const int32_t*>(y)[i]);
  return static_cast<const float*>(y)[i];
}

// y[i .. i + 7]; `vec`: i is a multiple of 8 and y is 16-byte aligned.
__device__ __forceinline__ void dy_load8(int src, const void* y, size_t i, bool vec, float* v) {
  if (!vec) {
    for (int j = 0; j < 8; ++j) v[j] = dy_load(src, y, i + j);
  } else if (src == DY_BF16) {
    const uint4 q = *reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(y) + i);
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
    for (int j = 0; j < 4; ++j) {
      v[2 * j] = __uint_as_float(w[j] << 16);
      v[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
    }
  } else {
    const uint4* p = reinterpret_cast<const uint4*>(static_cast<const uint32_t*>(y) + i);
    const uint4 q0 = p[0], q1 = p[1];
    const uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    for (int j = 0; j < 8; ++j) v[j] = src == DY_INT32 ? __int2float_rn(int32_t(w[j])) : __uint_as_float(w[j]);
  }
}

// 2^s for s in [-149, 127], exactly.
__device__ __forceinline__ float pow2f(int s) { return __int_as_float(s >= -126 ? (s + 127) << 23 : 1 << (s + 149)); }

// x * 2^s rounded once (ldexpf's result) for any s, as x * a * b * c, powers of two picked once per s (musl's scalbnf): the
// first two products are exact, or overflow only where x * 2^s does, or are subnormal only where x * 2^s rounds to zero;
// the last rounds. Not ldexpf, which ptxas builds from MUFU.EX2 (refused by the SASS gate), and all in FP32: a double
// fallback, however rare, is if-converted into every element's path at a fraction of FP32's rate.
struct Scale2 {
  float a, b, c;
};
__device__ __forceinline__ Scale2 scale2_of(int s) {
  float a = 1.f, b = 1.f;
  if (s > 127) {
    a = 0x1p127f, s -= 127;
    if (s > 127) b = 0x1p127f, s = min(s - 127, 127);
  } else if (s < -126) {
    a = 0x1p-102f, s += 102;
    if (s < -126) b = 0x1p-102f, s = max(s + 102, -126);
  }
  return {a, b, pow2f(s)};
}
__device__ __forceinline__ float scale2(float x, Scale2 m) { return __fmul_rn(__fmul_rn(__fmul_rn(x, m.a), m.b), m.c); }

__device__ __forceinline__ void put_bytes(uint8_t* p, const uint8_t* b, int nb, bool vec) {
  if (!vec) {
    for (int i = 0; i < nb; ++i) p[i] = b[i];
  } else if (nb >= 16) {
    for (int i = 0; i < nb; i += 16) *reinterpret_cast<uint4*>(p + i) = *reinterpret_cast<const uint4*>(b + i);
  } else if (nb == 8) {
    *reinterpret_cast<uint2*>(p) = *reinterpret_cast<const uint2*>(b);
  } else {
    *reinterpret_cast<uint32_t*>(p) = *reinterpret_cast<const uint32_t*>(b);
  }
}

__device__ __forceinline__ uint8_t e4m3_rne(float x) {
  return uint8_t(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3_value(uint8_t c) {
  const int e = (c >> 3) & 15, mt = c & 7;
  const float v = e == 0 ? __fmul_rn(float(mt), 0x1p-9f) : __int_as_float(((e + 120) << 23) | (mt << 20));
  return (c & 0x80) ? -v : v;
}

// E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6}: nearest, ties to the even code, saturating at 6.
__device__ __forceinline__ uint8_t e2m1_rne(float x) {
  float a = fabsf(x);
  uint8_t c = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
  return uint8_t((signbit(x) ? 8 : 0) | c);
}

// NVFP4 rounds quotients, and a division would bring div.rn's slow path, which ptxas builds into each kernel specialized to
// its call sites: bodies the SASS gate hasn't pinned. These round them without dividing, from exact products: RN(q) <= t
// exactly when q <= t + ulp(t) / 2 (t even), and for q = x / s that is x <= t s, since t s is exact and no float lies within
// s ulp(t) / 2 < ulp(t s) of it; likewise RN(q) < t exactly when x < t s (t s no power of two). Checked against derive.py's
// twin on every b >= 0 for the scale and around every threshold of every scale for E2M1.

// e2m1_rne(RN(v / s)) for an E4M3 scale s > 0.
__device__ __forceinline__ uint8_t e2m1_rne_quot(float v, float s) {
  const float a = fabsf(v);
  const uint8_t c = a <= __fmul_rn(0.25f, s) ? 0 : a < __fmul_rn(0.75f, s) ? 1 : a <= __fmul_rn(1.25f, s) ? 2
                  : a < __fmul_rn(1.75f, s) ? 3 : a <= __fmul_rn(2.5f, s) ? 4 : a < __fmul_rn(3.5f, s) ? 5
                  : a <= __fmul_rn(5.0f, s) ? 6 : 7;
  return uint8_t((signbit(v) ? 8 : 0) | c);
}

// Whether e4m3_rne(RN(b / 6)) is code i or above: b past 6 times the midpoint below code i, ties to the even code.
__device__ __forceinline__ bool e4m3_sixth_reaches(float b, int i) {
  const float t = __fmul_rn(3.f, __fadd_rn(e4m3_value(uint8_t(i - 1)), e4m3_value(uint8_t(i))));
  return i % 2 == 0 ? b >= t : b > t;
}

// e4m3_rne(RN(b / 6)) for b >= 0: b times RN(1/6) rounds to within one code of it.
__device__ __forceinline__ uint8_t e4m3_rne_sixth(float b) {
  const int c = e4m3_rne(__fmul_rn(b, 0x1.555556p-3f));
  if (c < 126 && e4m3_sixth_reaches(b, c + 1)) return uint8_t(c + 1);
  if (c > 0 && !e4m3_sixth_reaches(b, c)) return uint8_t(c - 1);
  return uint8_t(c);
}

__device__ __forceinline__ size_t sf_offset(int row, int col, int cols) {
  return (size_t(row / 128) * ((cols + 3) / 4) + col / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + (col % 4);
}

__device__ __forceinline__ float block_max(float v, float* red) {
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    float w = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, o));
    if (threadIdx.x == 0) red[0] = w;
  }
  __syncthreads();
  return red[0];
}

// Each thread holds columns c0 .. c0 + 7 of a pass, so a scale block spans 2 (NVFP4) or 4 (MX) adjacent lanes. `yvec`: n % 8 ==
// 0 and y 16-byte aligned; `avec`: A 16-byte aligned (k % 8 == 0 always).
__global__ void k_derive(int src, const void* __restrict__ y, int n, int kind, uint8_t* __restrict__ a,
                         uint8_t* __restrict__ sf, int k, bool yvec, bool avec) {
  __shared__ float red[32];
  const int r = blockIdx.x;
  const size_t yr = size_t(r) * n, ar = size_t(r) * k;
  float amax = 0.f, v[8];
  if (yvec) {
    for (int c = 8 * threadIdx.x; c < n; c += 8 * blockDim.x) {
      dy_load8(src, y, yr + c, true, v);
      for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
    }
  } else {
    for (int c = threadIdx.x; c < n; c += blockDim.x) amax = fmaxf(amax, fabsf(dy_load(src, y, yr + c)));
  }
  amax = block_max(amax, red);
  int e = 0;
  if (amax > 0.f && isfinite(amax)) frexpf(amax, &e);
  const int shift = amax > 0.f && isfinite(amax) ? derive_t(kind) - e : 0;
  const Scale2 row = scale2_of(shift);
  const int bs = kind == DV_NVFP4 ? 16 : 32, lanes = bs / 8;
  const size_t elem2 = kind == DV_F32 ? 8 : kind == DV_BF16 ? 4 : kind == DV_NVFP4 || kind == DV_MXFP4 ? 1 : 2;  // 2 x bytes
  for (int base = 0; base < k; base += 8 * blockDim.x) {
    const int c0 = base + 8 * threadIdx.x;
    const bool on = c0 < k;
    if (on) {
      if (yvec) {
        dy_load8(src, y, yr + c0 % n, true, v);
      } else {
        for (int j = 0; j < 8; ++j) v[j] = dy_load(src, y, yr + (c0 + j) % n);
      }
      for (int j = 0; j < 8; ++j) v[j] = scale2(v[j], row);
    } else {
      for (int j = 0; j < 8; ++j) v[j] = 0.f;
    }
    alignas(16) uint8_t out[32];
    switch (kind) {
      case DV_F32:
        for (int j = 0; j < 8; ++j) reinterpret_cast<float*>(out)[j] = v[j];
        break;
      case DV_BF16:
        for (int j = 0; j < 8; ++j) reinterpret_cast<uint16_t*>(out)[j] = __bfloat16_as_ushort(__float2bfloat16_rn(v[j]));
        break;
      case DV_E4M3:
        for (int j = 0; j < 8; ++j) out[j] = e4m3_rne(v[j]);
        break;
      case DV_INT8:
        for (int j = 0; j < 8; ++j) out[j] = uint8_t(int8_t(fminf(fmaxf(rintf(v[j]), -127.f), 127.f)));
        break;
      default: {  // block-scaled: MXFP8 (E4M3, UE8M0 per 32), NVFP4 (E2M1, UE4M3 per 16), MXFP4 (E2M1, UE8M0 per 32)
        float bmax = 0.f;
        for (int j = 0; j < 8; ++j) bmax = fmaxf(bmax, fabsf(v[j]));
        for (int o = 1; o < lanes; o <<= 1) bmax = fmaxf(bmax, __shfl_xor_sync(0xffffffffu, bmax, o));
        uint8_t code[8], scale;
        if (kind == DV_NVFP4) {
          scale = e4m3_rne_sixth(bmax);
          const float s = e4m3_value(scale);
          for (int j = 0; j < 8; ++j) code[j] = s > 0.f ? e2m1_rne_quot(v[j], s) : 0;
        } else {
          int eb = 0;
          if (bmax > 0.f) frexpf(bmax, &eb);
          int xs = bmax > 0.f ? eb - 1 - (kind == DV_MXFP8 ? 8 : 2) : -127;
          xs = xs < -127 ? -127 : xs > 127 ? 127 : xs;
          scale = uint8_t(xs + 127);
          const float q = pow2f(-xs);
          for (int j = 0; j < 8; ++j) code[j] = kind == DV_MXFP8 ? e4m3_rne(__fmul_rn(v[j], q)) : e2m1_rne(__fmul_rn(v[j], q));
        }
        if (kind == DV_MXFP8) {
          for (int j = 0; j < 8; ++j) out[j] = code[j];
        } else {
          for (int j = 0; j < 8; j += 2) out[j / 2] = uint8_t(code[j] | (code[j + 1] << 4));
        }
        if (on && (threadIdx.x & (lanes - 1)) == 0) sf[sf_offset(r, c0 / bs, k / bs)] = scale;
      }
    }
    if (on) put_bytes(a + (ar + c0) * elem2 / 2, out, int(8 * elem2 / 2), avec);
  }
}

// A (and its block scales `sf`, for the block-scaled kinds) of m x k from y of m x n (format `src`), on `stream`.
int ph_derive(int src, const void* y, int m, int n, int kind, void* a, void* sf, int k, void* stream) {
  if (kind < 0 || kind >= DV_KINDS || src < DY_BF16 || src > DY_F32) return fail_arg("ph_derive: unknown kind or source");
  const bool blocked = kind == DV_MXFP8 || kind == DV_NVFP4 || kind == DV_MXFP4;
  if (m <= 0 || n <= 0 || k <= 0 || k % 8 || (blocked && (k % 32 || !sf)))
    return fail_arg("ph_derive: bad shape (k a multiple of 8, of 32 when block-scaled) or no scale buffer");
  const int groups = (std::max(n, k) + 7) / 8;
  const int threads = std::min(DERIVE_MAX_THREADS, std::max(32, (groups + 31) / 32 * 32));
  const bool yvec = n % 8 == 0 && reinterpret_cast<uintptr_t>(y) % 16 == 0, avec = reinterpret_cast<uintptr_t>(a) % 16 == 0;
  k_derive<<<m, threads, 0, cudaStream_t(stream)>>>(src, y, n, kind, static_cast<uint8_t*>(a), static_cast<uint8_t*>(sf), k,
                                                    yvec, avec);
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

int ph_info_size() { return int(sizeof(ph_info)); }

}  // extern "C"
