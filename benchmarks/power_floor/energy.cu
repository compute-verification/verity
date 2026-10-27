// Energy per operation on a GPU's pipes, alone and in pairs (POWER-MODEL, CMP-172): board energy from NVML's counter
// over a steady window of back-to-back launches, divided among the operations the launches counted.
//
//   energy ROLE_A ROLE_B MODE SECONDS [WARMUP_S] [SM_FRACTION]
//   energy gemm MODE SECONDS [WARMUP_S]        cuBLASLt s8 GEMMs at the demo's shapes (8B's QKV, O, gate+up, down, M 16,384)
//   energy idle SECONDS                        no launches: the board with a CUDA context open and nothing running
//
// Every block has 16 warps; the first 8 run ROLE_A a fixed count of iterations and the last 8 run ROLE_B until every
// ROLE_A warp has finished (ROLE_A == ROLE_B: all 16 run it). Roles: imma (mma.sync m16n8k32 s8, 8 accumulators a
// warp), wgmma (sm_90a: m64n128k32 s8 SS, 4 a commit group, a warpgroup), dp4a (16 chains), ffma (16 chains, 2 steps), int (8
// chains of IADD3, LOP3 and SHF), sha (one SHA-512 compression a thread an iteration), lds (8 LDS.128 a warp an
// iteration, conflict free), ldg (a 4 GiB buffer streamed with LDG.128 through L2), sleep (nanosleep: the SMs resident
// and idle), none. MODE: 0 zero operands, 1 random operands held fixed, 2 random operands alternating with their
// complement from one instruction to the next (the most toggling a fixed program gets). SM_FRACTION < 1 launches that
// share of one block an SM (the rate at a partial load, for linearity).
//
// Prints one JSON line: the window's energy (NVML total-energy counter, mJ), its wall time, the operations each role
// did, NVML's SM clock, power, temperature and clock-event reasons sampled every 50 ms, and the SM clock the blocks ran
// at (clock64 over %globaltimer).
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

typedef unsigned long long u64;
namespace sh {
typedef unsigned long long u64;
#include "../codec_tensor/sha512.cuh"
}  // namespace sh

#ifdef PF_SM90
#include "../pouw/dc/dc.cuh"
#include "../pouw/dc/wgmma.cuh"
#endif

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                             \
    }                                                                                      \
  } while (0)

enum Role { NONE = 0, IMMA, WGMMA, DP4A, FFMA, INT, SHA, LDS, LDG, SLEEP, NROLES };
static const char* ROLE_NAME[NROLES] = {"none", "imma", "wgmma", "dp4a", "ffma", "int", "sha", "lds", "ldg", "sleep"};
// What one warp-iteration of a role does, in the role's unit: s8 MACs (imma, wgmma's share a warp, dp4a), FP32 FMAs,
// integer lane instructions, SHA-512 compressions, shared-memory words, device-memory bytes, nanosleep calls.
static const double UNIT_PER_WARP_ITER[NROLES] = {0, 8.0 * 16 * 8 * 32, 4.0 * 64 * 128 * 32 / 4, 16.0 * 32 * 4,
                                                  32.0 * 32, 48.0 * 32, 32, 8.0 * 4 * 32, 4.0 * 32 * 16, 32};
static const char* UNIT_NAME[NROLES] = {"", "mac", "mac", "mac", "fma", "lane_op", "compression", "word", "byte", "sleep"};

constexpr int TPB = 512, WARPS = TPB / 32, HALF = WARPS / 2;
constexpr size_t LDG_BYTES = 4ull << 30;
constexpr int LDS_BYTES = 16384;

struct Args {
  int iters;
  uint32_t zmask;  // 0 in mode 0, else ~0: operands are (random & zmask), so the compiler never sees a constant
  uint32_t tmask;  // ~0 in mode 2, else 0: the second operand set is the first ^ tmask
  u64* counts;     // [2]: warp-iterations done by role A's warps and role B's
  long long* times;  // a block: clock64 and %globaltimer at the start and the end
  uint32_t* smid;
  const uint4* gbuf;
  uint32_t* sink;
};

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ bool stop(const volatile int* done) { return done != nullptr && *done != 0; }
struct Ret {
  uint32_t s;
  int n;  // iterations done
};

__device__ __forceinline__ void imma(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ Ret run_imma(const Args& a, int iters, const volatile int* done, uint32_t seed) {
  uint32_t a0[4], a1[4], b0[2], b1[2];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    a0[i] = mix32(seed + i) & a.zmask;
    a1[i] = a0[i] ^ a.tmask;
  }
#pragma unroll
  for (int i = 0; i < 2; i++) {
    b0[i] = mix32(seed + 7 + i) & a.zmask;
    b1[i] = b0[i] ^ a.tmask;
  }
  int d[8][4] = {};
  int it = 0;
#pragma unroll 1
  for (; it < iters; it++) {
#pragma unroll
    for (int j = 0; j < 8; j++) imma(d[j], (j & 1) ? a1 : a0, (j & 1) ? b1 : b0);
    if (stop(done)) {
      it++;
      break;
    }
  }
  uint32_t s = it;
#pragma unroll
  for (int j = 0; j < 8; j++) s ^= d[j][0] ^ d[j][1] ^ d[j][2] ^ d[j][3];
  return {s, it};
}

__device__ Ret run_dp4a(const Args& a, int iters, const volatile int* done, uint32_t seed) {
  uint32_t x[2][4], y[2];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    x[0][i] = mix32(seed + i) & a.zmask;
    x[1][i] = x[0][i] ^ a.tmask;
  }
  y[0] = mix32(seed + 9) & a.zmask;
  y[1] = y[0] ^ a.tmask;
  int c[16] = {};
  int it = 0;
#pragma unroll 1
  for (; it < iters; it++) {
#pragma unroll
    for (int j = 0; j < 16; j++)
      asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(c[j]) : "r"(x[j & 1][(j >> 1) & 3]), "r"(y[j & 1]));
    if (stop(done)) {
      it++;
      break;
    }
  }
  uint32_t s = it;
#pragma unroll
  for (int j = 0; j < 16; j++) s ^= c[j];
  return {s, it};
}

__device__ Ret run_ffma(const Args& a, int iters, const volatile int* done, uint32_t seed) {
  // f <- f * m + e with m in [0.5, 1) and e in [1, 2): bounded. 16 chains, two steps an iteration, each step taking
  // the other operand set (mode 2: the mantissas complemented, so the multiplier's inputs toggle every instruction).
  float m[2], e[2], f[16];
  m[0] = __uint_as_float(((mix32(seed) & 0x007FFFFFu) | 0x3F000000u) & a.zmask);
  m[1] = __uint_as_float(__float_as_uint(m[0]) ^ (a.tmask & 0x007FFFFFu));
  e[0] = __uint_as_float(((mix32(seed + 1) & 0x007FFFFFu) | 0x3F800000u) & a.zmask);
  e[1] = __uint_as_float(__float_as_uint(e[0]) ^ (a.tmask & 0x007FFFFFu));
#pragma unroll
  for (int j = 0; j < 16; j++) f[j] = __uint_as_float(((mix32(seed + 2 + j) & 0x007FFFFFu) | 0x3F800000u) & a.zmask);
  int it = 0;
#pragma unroll 1
  for (; it < iters; it++) {
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
      for (int j = 0; j < 16; j++)
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[j]) : "f"(m[(j + r) & 1]), "f"(e[(j + r) & 1]));
    if (stop(done)) {
      it++;
      break;
    }
  }
  uint32_t s = it;
#pragma unroll
  for (int j = 0; j < 16; j++) s ^= __float_as_uint(f[j]);
  return {s, it};
}

__device__ Ret run_int(const Args& a, int iters, const volatile int* done, uint32_t seed) {
  // SHA-512's lane mix on 32-bit halves: IADD3, LOP3 (xor3) and a funnel shift, 8 independent chains, two steps an
  // iteration (48 lane instructions a thread).
  uint32_t x[8], y[8], z[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    x[j] = mix32(seed + 3 * j) & a.zmask;
    y[j] = mix32(seed + 3 * j + 1) & a.zmask;
    z[j] = mix32(seed + 3 * j + 2) & a.zmask;
  }
  int it = 0;
#pragma unroll 1
  for (; it < iters; it++) {
#pragma unroll
    for (int r = 0; r < 2; r++) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        asm volatile("add.u32 %0, %0, %1;\n\tadd.u32 %0, %0, %2;" : "+r"(x[j]) : "r"(y[j]), "r"(z[j]));
        asm volatile("lop3.b32 %0, %1, %0, %2, 0x96;" : "+r"(y[j]) : "r"(x[j]), "r"(z[j]));
        asm volatile("shf.l.wrap.b32 %0, %0, %1, 13;" : "+r"(z[j]) : "r"(x[j]));
      }
    }
    if (stop(done)) {
      it++;
      break;
    }
  }
  uint32_t s = it;
#pragma unroll
  for (int j = 0; j < 8; j++) s ^= x[j] ^ y[j] ^ z[j];
  return {s, it};
}

__device__ Ret run_sha(const Args& a, int iters, const volatile int* done, uint32_t seed) {
  sh::u64 st[8], w[16];
#pragma unroll
  for (int i = 0; i < 8; i++) st[i] = ((sh::u64)mix32(seed + 2 * i) << 32 | mix32(seed + 2 * i + 1)) & a.zmask;
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = ((sh::u64)mix32(seed + 20 + i) << 32 | mix32(seed + 40 + i)) & a.zmask;
  int it = 0;
#pragma unroll 1
  for (; it < iters; it++) {
    sh::sha512_compress(st, w);
    w[0] ^= st[0];
    if (stop(done)) {
      it++;
      break;
    }
  }
  uint32_t s = it;
#pragma unroll
  for (int i = 0; i < 8; i++) s ^= (uint32_t)st[i] ^ (uint32_t)(st[i] >> 32);
  return {s, it};
}

__device__ Ret run_lds(const Args& a, int iters, const volatile int* done, const unsigned char* region) {
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(region);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t s = 0;
  int it = 0;
#pragma unroll 1
  for (; it < iters; it++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t off = (((it * 8 + j + warp) * 512) + lane * 16) & (LDS_BYTES - 1);
      uint32_t v0, v1, v2, v3;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(base + off));
      s ^= v0 ^ v1 ^ v2 ^ v3;
    }
    if (stop(done)) {
      it++;
      break;
    }
  }
  return {s, it};
}

__device__ Ret run_ldg(const Args& a, int iters, const volatile int* done, int gwarp, int nwarps) {
  const int lane = threadIdx.x & 31;
  const size_t n16 = LDG_BYTES / 16;
  uint32_t s = 0;
  int it = 0;
#pragma unroll 1
  for (; it < iters; it++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const size_t idx = ((((size_t)it * 4 + j) * nwarps + gwarp) * 32 + lane) & (n16 - 1);
      uint32_t v0, v1, v2, v3;
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                   : "l"(a.gbuf + idx));
      s ^= v0 ^ v1 ^ v2 ^ v3;
    }
    if (stop(done)) {
      it++;
      break;
    }
  }
  return {s, it};
}

__device__ Ret run_sleep(int iters, const volatile int* done) {
  int it = 0;
#pragma unroll 1
  for (; it < iters; it++) {
    __nanosleep(1000);
    if (stop(done)) {
      it++;
      break;
    }
  }
  return {(uint32_t)it, it};
}

#ifdef PF_SM90
constexpr int WG_KB = 32, WG_AROWS = 64, WG_BROWS = 128, WG_TILE = (WG_AROWS + WG_BROWS) * WG_KB;
__device__ __forceinline__ void wg_s8_n128(int (&d)[64], uint64_t da, uint64_t db, int sd) { wgmma_s8_n128(d, da, db, sd); }
// A warpgroup: per iteration 4 wgmma m64n128k32 s8 alternately from tiles 0 and 1 (1 = 0's complement in mode 2), commit,
// and wait for the group before last. Every one accumulates: a group whose D the next overwrites is dead, and ptxas
// replaces it with an HGMMA.64x8x16 placeholder (build b33cea773e did that to 3 iterations in 4).
__device__ Ret run_wgmma(const Args& a, int iters, const volatile int* done, const unsigned char* tiles) {
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(tiles);
  const uint64_t da0 = wgmma_desc(base, kmaj_lbo(), kmaj_sbo(WG_KB));
  const uint64_t db0 = wgmma_desc(base + WG_AROWS * WG_KB, kmaj_lbo(), kmaj_sbo(WG_KB));
  const uint64_t da1 = wgmma_desc(base + WG_TILE, kmaj_lbo(), kmaj_sbo(WG_KB));
  const uint64_t db1 = wgmma_desc(base + WG_TILE + WG_AROWS * WG_KB, kmaj_lbo(), kmaj_sbo(WG_KB));
  int d[64] = {};
  int it = 0;
#pragma unroll 1
  for (; it < iters; it++) {
    wgmma_fence();
    wg_s8_n128(d, da0, db0, 1);
    wg_s8_n128(d, da1, db1, 1);
    wg_s8_n128(d, da0, db0, 1);
    wg_s8_n128(d, da1, db1, 1);
    wgmma_commit();
    wgmma_wait<1>();
    if (stop(done)) {
      it++;
      break;
    }
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  uint32_t s = it;
#pragma unroll
  for (int i = 0; i < 64; i++) s ^= d[i];
  return {s, it};
}
#endif

template <int R>
__device__ Ret run_role(const Args& a, int iters, const volatile int* done, uint32_t seed, unsigned char* smem,
                             int gwarp, int nwarps) {
  if constexpr (R == IMMA) return run_imma(a, iters, done, seed);
  if constexpr (R == DP4A) return run_dp4a(a, iters, done, seed);
  if constexpr (R == FFMA) return run_ffma(a, iters, done, seed);
  if constexpr (R == INT) return run_int(a, iters, done, seed);
  if constexpr (R == SHA) return run_sha(a, iters, done, seed);
  if constexpr (R == LDS) return run_lds(a, iters, done, smem);
  if constexpr (R == LDG) return run_ldg(a, iters, done, gwarp, nwarps);
  if constexpr (R == SLEEP) return run_sleep(iters, done);
#ifdef PF_SM90
  if constexpr (R == WGMMA) return run_wgmma(a, iters, done, smem + LDS_BYTES);
#endif
  return {0, 0};
}

template <int RA, int RB>
__global__ void __launch_bounds__(TPB, 1) kern(Args a) {
#ifdef PF_SM90
  __shared__ __align__(1024) unsigned char smem[LDS_BYTES + 2 * WG_TILE];
  constexpr int SMEM_WORDS = (LDS_BYTES + 2 * WG_TILE) / 4;
#else
  __shared__ __align__(1024) unsigned char smem[LDS_BYTES];
  constexpr int SMEM_WORDS = LDS_BYTES / 4;
#endif
  __shared__ int done, finished;
  uint32_t* w = reinterpret_cast<uint32_t*>(smem);
  for (int i = threadIdx.x; i < SMEM_WORDS; i += TPB) w[i] = mix32(i * 0x9E3779B9u + 0x1234567u) & a.zmask;
#ifdef PF_SM90
  for (int i = threadIdx.x; i < WG_TILE / 4; i += TPB)
    w[(LDS_BYTES + WG_TILE) / 4 + i] = (mix32((LDS_BYTES / 4 + i) * 0x9E3779B9u + 0x1234567u) & a.zmask) ^ a.tmask;
  fence_async_smem();
#endif
  if (threadIdx.x == 0) {
    done = 0;
    finished = 0;
  }
  __syncthreads();
  long long c0 = clock64();
  unsigned long long g0 = gtime();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool roleA = (RA == RB) || warp < HALF;
  const uint32_t seed = mix32(blockIdx.x * TPB + threadIdx.x);
  const int gwarp = blockIdx.x * WARPS + warp, nwarps = gridDim.x * WARPS;
  Ret r;
  if (roleA) {
    r = run_role<RA>(a, a.iters, nullptr, seed, smem, gwarp, nwarps);
    if (RA != RB) {
      __syncwarp();
      if (lane == 0 && atomicAdd(&finished, 1) == HALF - 1) atomicExch(&done, 1);
    }
  } else {
    r = run_role<RB>(a, 1 << 30, &done, seed, smem, gwarp, nwarps);
    if (lane == 0) atomicAdd(&a.counts[1], (u64)r.n);
  }
  __syncthreads();
  long long c1 = clock64();
  unsigned long long g1 = gtime();
  if (threadIdx.x == 0) {
    a.times[4 * blockIdx.x + 0] = c0;
    a.times[4 * blockIdx.x + 1] = c1;
    a.times[4 * blockIdx.x + 2] = (long long)g0;
    a.times[4 * blockIdx.x + 3] = (long long)g1;
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    a.smid[blockIdx.x] = sm;
  }
  a.sink[blockIdx.x * TPB + threadIdx.x] = r.s;
}

// Mode 1: random 16-byte chunks; mode 2: every odd chunk the complement of the one before (a line's sectors toggle).
__global__ void fill(uint4* g, size_t n, int mode) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t k = (mode == 2) ? (i & ~(size_t)1) : i;
    const uint32_t s = (uint32_t)(k * 4) ^ (uint32_t)(k >> 30) * 0x632BE5ABu;
    uint4 v = make_uint4(mix32(s), mix32(s + 1), mix32(s + 2), mix32(s + 3));
    if (mode == 2 && (i & 1)) v = make_uint4(~v.x, ~v.y, ~v.z, ~v.w);
    g[i] = v;
  }
}

// ---- host: NVML through dlopen -----------------------------------------------------------------------------------
typedef void* nvmlDevice_t;
struct Nvml {
  void* h = nullptr;
  int (*init)() = nullptr;
  int (*byPci)(const char*, nvmlDevice_t*) = nullptr;
  int (*energy)(nvmlDevice_t, unsigned long long*) = nullptr;
  int (*power)(nvmlDevice_t, unsigned*) = nullptr;
  int (*clock)(nvmlDevice_t, int, unsigned*) = nullptr;
  int (*temp)(nvmlDevice_t, int, unsigned*) = nullptr;
  int (*reasons)(nvmlDevice_t, unsigned long long*) = nullptr;
  int (*limit)(nvmlDevice_t, unsigned*) = nullptr;
  nvmlDevice_t dev = nullptr;
  bool open(const char* pci) {
    h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!h) return false;
    init = (int (*)())dlsym(h, "nvmlInit_v2");
    byPci = (int (*)(const char*, nvmlDevice_t*))dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2");
    energy = (int (*)(nvmlDevice_t, unsigned long long*))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
    power = (int (*)(nvmlDevice_t, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage");
    clock = (int (*)(nvmlDevice_t, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo");
    temp = (int (*)(nvmlDevice_t, int, unsigned*))dlsym(h, "nvmlDeviceGetTemperature");
    reasons = (int (*)(nvmlDevice_t, unsigned long long*))dlsym(h, "nvmlDeviceGetCurrentClocksEventReasons");
    if (!reasons) reasons = (int (*)(nvmlDevice_t, unsigned long long*))dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons");
    limit = (int (*)(nvmlDevice_t, unsigned*))dlsym(h, "nvmlDeviceGetEnforcedPowerLimit");
    return init && byPci && energy && init() == 0 && byPci(pci, &dev) == 0;
  }
  unsigned long long mj() {
    unsigned long long v = 0;
    energy(dev, &v);
    return v;
  }
};

struct Sample {
  double t, w;
  unsigned mhz, temp;
  unsigned long long reasons;
};

struct Sampler {
  Nvml& nv;
  std::atomic<bool> run{true};
  std::vector<Sample> s;
  std::thread th;
  std::chrono::steady_clock::time_point t0;
  explicit Sampler(Nvml& n) : nv(n) {
    t0 = std::chrono::steady_clock::now();
    th = std::thread([this] {
      while (run.load()) {
        Sample x{};
        x.t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        unsigned mw = 0;
        nv.power(nv.dev, &mw);
        x.w = mw / 1000.0;
        nv.clock(nv.dev, 1, &x.mhz);
        nv.temp(nv.dev, 0, &x.temp);
        if (nv.reasons) nv.reasons(nv.dev, &x.reasons);
        s.push_back(x);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }
  void finish() {
    run = false;
    th.join();
  }
};

static int role_of(const char* s) {
  for (int i = 0; i < NROLES; i++)
    if (!strcmp(s, ROLE_NAME[i])) return i;
  fprintf(stderr, "unknown role %s\n", s);
  exit(2);
}

typedef void (*KernFn)(Args);
// The kernels built: every role alone, and the pairs a run measures (role B beside role A; B = sleep is A on half the
// warps, the baseline for its pairs).
#define PF_PAIRS(X)                                                                                                   \
  X(NONE, NONE) X(SLEEP, SLEEP) X(IMMA, IMMA) X(WGMMA, WGMMA) X(DP4A, DP4A) X(FFMA, FFMA) X(INT, INT) X(SHA, SHA)      \
  X(LDS, LDS) X(LDG, LDG) X(IMMA, SLEEP) X(IMMA, DP4A) X(IMMA, FFMA) X(IMMA, INT) X(IMMA, SHA) X(IMMA, LDS)           \
  X(IMMA, LDG) X(WGMMA, SLEEP) X(WGMMA, DP4A) X(WGMMA, FFMA) X(WGMMA, INT) X(WGMMA, SHA) X(WGMMA, LDS) X(WGMMA, LDG) \
  X(WGMMA, IMMA) X(DP4A, FFMA) X(DP4A, INT) X(FFMA, INT) X(DP4A, SLEEP) X(FFMA, SLEEP) X(INT, SLEEP)
static KernFn pick(int ra, int rb) {
#define PF_PICK(A, B) \
  if (ra == A && rb == B) return kern<A, B>;
  PF_PAIRS(PF_PICK)
#undef PF_PICK
  return nullptr;
}

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void emit_samples(FILE* f, const std::vector<Sample>& s, double from) {
  std::vector<double> w;
  std::vector<unsigned> m;
  unsigned long long ru = 0;
  unsigned tmax = 0, tmin = 1000;
  for (auto& x : s)
    if (x.t >= from) {
      w.push_back(x.w);
      m.push_back(x.mhz);
      ru |= x.reasons;
      tmax = std::max(tmax, x.temp);
      tmin = std::min(tmin, x.temp);
    }
  if (w.empty()) return;
  std::sort(w.begin(), w.end());
  std::sort(m.begin(), m.end());
  double mean = 0;
  for (double v : w) mean += v;
  mean /= w.size();
  fprintf(f, ", \"nvml_samples\": %zu, \"nvml_w_mean\": %.1f, \"nvml_w_p05\": %.1f, \"nvml_w_p95\": %.1f", w.size(), mean,
          w[w.size() / 20], w[w.size() * 19 / 20]);
  fprintf(f, ", \"nvml_mhz_median\": %u, \"nvml_mhz_min\": %u, \"nvml_mhz_max\": %u", m[m.size() / 2], m[0], m.back());
  fprintf(f, ", \"temp_min\": %u, \"temp_max\": %u, \"reasons_union\": \"0x%llx\"", tmin, tmax, ru);
}

// cuBLASLt s8 x s8 -> s32 GEMM C = X W^T (TN in column-major), for the honest side at the demo's shapes.
struct Gemm {
  cublasLtHandle_t lt;
  void* ws;
  size_t wsz = 64ull << 20;
  struct Shape {
    int n, k;
    const char* name;
  };
  std::vector<Shape> shapes{{6144, 4096, "qkv"}, {4096, 4096, "o"}, {28672, 4096, "gate_up"}, {4096, 14336, "down"}};
  int M = 16384;
  int8_t *x, *w;
  int32_t* c;
  std::vector<cublasLtMatmulDesc_t> desc;
  std::vector<cublasLtMatrixLayout_t> la, lb, lc;
  std::vector<cublasLtMatmulAlgo_t> algo;
  void init(int mode) {
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) exit(3);
    CK(cudaMalloc(&ws, wsz));
    size_t xs = (size_t)M * 14336, wsb = (size_t)28672 * 4096, cs = (size_t)M * 28672;
    CK(cudaMalloc(&x, xs));
    CK(cudaMalloc(&w, wsb));
    CK(cudaMalloc(&c, cs * 4));
    std::vector<int8_t> h(std::max(xs, wsb));
    uint32_t r = 0x12345u;
    for (auto& v : h) {
      r = r * 1664525u + 1013904223u;
      v = mode == 0 ? 0 : (int8_t)(r >> 24);
    }
    CK(cudaMemcpy(x, h.data(), xs, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w, h.data(), wsb, cudaMemcpyHostToDevice));
    for (auto& s : shapes) {
      cublasLtMatmulDesc_t d;
      cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32I, CUDA_R_32I);
      cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
      cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
      cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
      cublasLtMatrixLayout_t A, B, C;
      cublasLtMatrixLayoutCreate(&A, CUDA_R_8I, s.k, s.n, s.k);
      cublasLtMatrixLayoutCreate(&B, CUDA_R_8I, s.k, M, s.k);
      cublasLtMatrixLayoutCreate(&C, CUDA_R_32I, s.n, M, s.n);
      cublasLtMatmulPreference_t p;
      cublasLtMatmulPreferenceCreate(&p);
      cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz);
      cublasLtMatmulHeuristicResult_t res[8];
      int n = 0;
      cublasLtMatmulAlgoGetHeuristic(lt, d, A, B, C, C, p, 8, res, &n);
      if (n == 0) {
        fprintf(stderr, "no cuBLASLt algo for %s\n", s.name);
        exit(3);
      }
      // the fastest of the heuristic's candidates, timed once
      int best = 0;
      float bt = 1e30f;
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      for (int i = 0; i < n; i++) {
        int32_t al = 1, be = 0;
        if (cublasLtMatmul(lt, d, &al, w, A, x, B, &be, c, C, c, C, &res[i].algo, ws, wsz, 0) != CUBLAS_STATUS_SUCCESS)
          continue;
        cudaEventRecord(e0);
        for (int q = 0; q < 3; q++) cublasLtMatmul(lt, d, &al, w, A, x, B, &be, c, C, c, C, &res[i].algo, ws, wsz, 0);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < bt) {
          bt = ms;
          best = i;
        }
      }
      desc.push_back(d);
      la.push_back(A);
      lb.push_back(B);
      lc.push_back(C);
      algo.push_back(res[best].algo);
    }
    CK(cudaDeviceSynchronize());
  }
  double layer() {  // one pass of the four shapes; returns its MACs
    double macs = 0;
    for (size_t i = 0; i < shapes.size(); i++) {
      int32_t al = 1, be = 0;
      cublasLtMatmul(lt, desc[i], &al, w, la[i], x, lb[i], &be, c, lc[i], c, lc[i], &algo[i], ws, wsz, 0);
      macs += (double)M * shapes[i].n * shapes[i].k;
    }
    return macs;
  }
};

int main(int argc, char** argv) {
  if (argc < 3 || (argc < 4 && strcmp(argv[1], "idle"))) {
    fprintf(stderr,
            "usage: energy ROLE_A ROLE_B MODE SECONDS [WARMUP_S] [SM_FRACTION] | energy gemm MODE SECONDS [WARMUP_S] | "
            "energy idle SECONDS\n");
    return 2;
  }
  const bool gemm = !strcmp(argv[1], "gemm"), idle = !strcmp(argv[1], "idle");
  int ra = 0, rb = 0, mode;
  double secs, warm, frac = 1.0;
  if (idle) {
    mode = 0;
    secs = atof(argv[2]);
    warm = 2.0;
  } else if (gemm) {
    mode = atoi(argv[2]);
    secs = atof(argv[3]);
    warm = argc > 4 ? atof(argv[4]) : 5.0;
  } else {
    ra = role_of(argv[1]);
    rb = role_of(argv[2]);
    mode = atoi(argv[3]);
    secs = atof(argv[4]);
    warm = argc > 5 ? atof(argv[5]) : 5.0;
    frac = argc > 6 ? atof(argv[6]) : 1.0;
  }
  CK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  char pci[64];
  CK(cudaDeviceGetPCIBusId(pci, sizeof pci, 0));
  Nvml nv;
  if (!nv.open(pci)) {
    fprintf(stderr, "NVML: no device %s\n", pci);
    return 3;
  }
  unsigned plim = 0;
  if (nv.limit) nv.limit(nv.dev, &plim);
  const int sms = prop.multiProcessorCount;

  FILE* out = stdout;
  if (idle) {
    Sampler smp(nv);
    std::this_thread::sleep_for(std::chrono::duration<double>(warm));
    const double ws = now_s();
    const unsigned long long e0 = nv.mj();
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    const double we = now_s();
    const unsigned long long e1 = nv.mj();
    smp.finish();
    fprintf(out, "{\"kind\": \"idle\", \"device\": \"%s\", \"sms\": %d, \"power_limit_w\": %.1f, \"window_s\": %.4f", prop.name,
            sms, plim / 1000.0, we - ws);
    fprintf(out, ", \"energy_j\": %.3f, \"power_w\": %.2f", (e1 - e0) / 1000.0, (e1 - e0) / 1000.0 / (we - ws));
    emit_samples(out, smp.s, warm);
    fprintf(out, "}\n");
    return 0;
  }
  if (gemm) {
    Gemm g;
    g.init(mode);
    Sampler smp(nv);
    double t0 = now_s();
    while (now_s() - t0 < warm) {
      g.layer();
      CK(cudaDeviceSynchronize());
    }
    double ts = now_s() - t0;
    const double ws = now_s();
    const unsigned long long e0 = nv.mj();
    double macs = 0;
    int layers = 0;
    while (now_s() - ws < secs) {
      for (int q = 0; q < 4; q++) macs += g.layer();
      CK(cudaDeviceSynchronize());
      layers += 4;
    }
    const double we = now_s();
    const unsigned long long e1 = nv.mj();
    smp.finish();
    fprintf(out, "{\"kind\": \"gemm\", \"device\": \"%s\", \"sms\": %d, \"mode\": %d, \"power_limit_w\": %.1f", prop.name,
            sms, mode, plim / 1000.0);
    fprintf(out, ", \"m\": %d, \"layers\": %d, \"window_s\": %.4f, \"energy_j\": %.3f, \"power_w\": %.2f, \"macs\": %.6e", g.M,
            layers, we - ws, (e1 - e0) / 1000.0, (e1 - e0) / 1000.0 / (we - ws), macs);
    fprintf(out, ", \"mac_per_s\": %.6e, \"j_per_mac\": %.6e", macs / (we - ws), (e1 - e0) / 1000.0 / macs);
    emit_samples(out, smp.s, ts);
    fprintf(out, "}\n");
    return 0;
  }

  KernFn fn = pick(ra, rb);
  if (!fn) {
    fprintf(stderr, "no kernel for %s/%s\n", argv[1], argv[2]);
    return 2;
  }
  if (rb == WGMMA && ra != WGMMA) {
    fprintf(stderr, "wgmma runs only as role A (its warpgroups must stop together)\n");
    return 2;
  }
#ifndef PF_SM90
  if (ra == WGMMA || rb == WGMMA) {
    fprintf(stderr, "wgmma needs the sm_90a build\n");
    return 2;
  }
#endif
  const int blocks = std::max(1, (int)(sms * frac + 0.5));
  Args a{};
  a.zmask = mode == 0 ? 0u : ~0u;
  a.tmask = mode == 2 ? ~0u : 0u;
  CK(cudaMalloc(&a.counts, 2 * sizeof(u64)));
  CK(cudaMalloc(&a.times, 4 * sizeof(long long) * blocks));
  CK(cudaMalloc(&a.smid, sizeof(uint32_t) * blocks));
  CK(cudaMalloc(&a.sink, sizeof(uint32_t) * TPB * blocks));
  uint4* gb = nullptr;
  if (ra == LDG || rb == LDG) {
    CK(cudaMalloc(&gb, LDG_BYTES));
    if (mode == 0)
      CK(cudaMemset(gb, 0, LDG_BYTES));
    else
      fill<<<4 * sms, 256>>>(gb, LDG_BYTES / 16, mode);
    CK(cudaDeviceSynchronize());
  }
  a.gbuf = gb;
  // calibrate: one launch at a guess, then scale to ~100 ms a launch
  a.iters = 1000;
  for (int pass = 0; pass < 2; pass++) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    fn<<<blocks, TPB>>>(a);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    a.iters = std::max(1, (int)(a.iters * 100.0 / std::max(ms, 0.01f)));
    a.iters = std::min(a.iters, 1 << 28);
  }
  Sampler smp(nv);
  double t0 = now_s();
  while (now_s() - t0 < warm) {
    fn<<<blocks, TPB>>>(a);
    CK(cudaDeviceSynchronize());
  }
  const double ts = now_s() - t0;
  std::vector<long long> ht(4 * blocks);
  CK(cudaMemset(a.counts, 0, 2 * sizeof(u64)));
  const double ws = now_s();
  const unsigned long long e0 = nv.mj();
  int launches = 0;
  double kern_ns = 0, clk_ratio = 0;
  while (now_s() - ws < secs) {
    fn<<<blocks, TPB>>>(a);
    CK(cudaDeviceSynchronize());
    launches++;
    if (launches % 4 == 1) {
      CK(cudaMemcpy(ht.data(), a.times, ht.size() * sizeof(long long), cudaMemcpyDeviceToHost));
      double r = 0, ns = 0;
      for (int b = 0; b < blocks; b++) {
        const double g = (double)(ht[4 * b + 3] - ht[4 * b + 2]);
        r += g > 0 ? (double)(ht[4 * b + 1] - ht[4 * b]) / g : 0.0;
        ns += g;
      }
      clk_ratio += r / blocks;
      kern_ns += ns / blocks;
    }
  }
  const double we = now_s();
  const unsigned long long e1 = nv.mj();
  smp.finish();
  const int nsampled = (launches + 3) / 4;
  u64 hc[2];
  CK(cudaMemcpy(hc, a.counts, sizeof hc, cudaMemcpyDeviceToHost));
  const double win = we - ws;
  const double ej = (e1 - e0) / 1000.0;
  const double a_warps = (ra == rb ? WARPS : HALF) * (double)blocks;
  const double ops_a = UNIT_PER_WARP_ITER[ra] * a.iters * a_warps * launches;
  fprintf(out, "{\"kind\": \"pipe\", \"device\": \"%s\", \"sms\": %d, \"blocks\": %d, \"role_a\": \"%s\", \"role_b\": \"%s\"",
          prop.name, sms, blocks, ROLE_NAME[ra], ROLE_NAME[rb]);
  fprintf(out, ", \"mode\": %d, \"power_limit_w\": %.1f, \"iters\": %d, \"launches\": %d, \"window_s\": %.4f", mode,
          plim / 1000.0, a.iters, launches, win);
  fprintf(out, ", \"energy_j\": %.3f, \"power_w\": %.2f, \"unit_a\": \"%s\", \"ops_a\": %.6e, \"rate_a\": %.6e", ej, ej / win,
          UNIT_NAME[ra], ops_a, ops_a / win);
  if (ra != rb) {
    const double ops_b = UNIT_PER_WARP_ITER[rb] * (double)hc[1];
    fprintf(out, ", \"unit_b\": \"%s\", \"ops_b\": %.6e, \"rate_b\": %.6e", UNIT_NAME[rb], ops_b, ops_b / win);
  }
  std::vector<uint32_t> hs(blocks);
  CK(cudaMemcpy(hs.data(), a.smid, hs.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  std::sort(hs.begin(), hs.end());
  fprintf(out, ", \"distinct_sms\": %d", (int)(std::unique(hs.begin(), hs.end()) - hs.begin()));
  fprintf(out, ", \"kernel_ms\": %.3f, \"sm_mhz_kernel\": %.1f", kern_ns / nsampled / 1e6, clk_ratio / nsampled * 1000.0);
  emit_samples(out, smp.s, ts);
  fprintf(out, "}\n");
  return 0;
}
