// pouw_hash.cuh's vectors, rates and epilogue cost, as JSON lines.
//   pouw_hash_bench check                 the vectors from the device (pouw_hash_check.cu; the host build prints the same)
//   pouw_hash_bench rates [ms] [filter]   every lane busy, then one thread's latency, then the concurrency sweep
//   pouw_hash_bench epi [ms] [filter]     an FP8 mma.sync GEMM tile loop with Pearl-C's epilogue, hashed and not
// The device build refuses to run (exit 3) unless the visible device's UUID is $POUW_EXPECT_UUID. check.py holds the check
// lines to their references; run.sh runs check first and times nothing unless it passes.
//
// Rates are message bytes per SM per SM clock (per_sm_clk, from each block's clock64 span on its SM) and GB/s (CUDA events
// over the launch). The units per byte that the accounting uses are the device's MACs per SM per clock over per_sm_clk:
// divide by a GEMM rate measured on the same card (GPU 6's baselines); nothing here assumes one.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <strings.h>

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "vectors.cuh"

void device_checks(CheckOut* h);   // pouw_hash_check.cu

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));              \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

struct Rec {
  unsigned long long c0, c1, t0, t1;
  unsigned smid, pad;
};
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned get_smid() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
#define BEGIN_REC                  \
  __syncthreads();                 \
  unsigned long long _c0 = clock64(), _t0 = gtimer();
#define END_REC(rec)                                                                                          \
  __syncthreads();                                                                                            \
  if (threadIdx.x == 0) (rec)[blockIdx.x] = Rec{_c0, (unsigned long long)clock64(), _t0, gtimer(), get_smid(), 0};

// ---- every lane busy, and one thread's latency (the same kernels on one thread)
PH_HD uint64_t seed(uint32_t t, long long it) { return (uint64_t)(t + 1) * PHI ^ (uint64_t)it << 32; }
template <int ROUNDS, bool SHF = PH_KECCAK_SHF>
__global__ void k_perm(long long iters, Rec* rec, uint64_t* out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = seed(t, i);
  BEGIN_REC
#pragma unroll 1
  for (long long it = 0; it < iters; ++it) ph::keccak_p<ROUNDS, SHF>(s);
  END_REC(rec)
  uint64_t x = 0;
#pragma unroll
  for (int i = 0; i < 25; ++i) x ^= s[i];
  out[t] = x;
}
__global__ void k_sha(long long iters, Rec* rec, uint64_t* out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t h[8];
  ph::sha256_iv(h);
  BEGIN_REC
  sha_chain(h, (uint32_t)(seed(t, 0) >> 32), iters);
  END_REC(rec)
  out[t] = (uint64_t)h[0] << 32 | h[7];
}
__global__ void k_b3(long long iters, Rec* rec, uint64_t* out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cv[8];
  for (int i = 0; i < 8; ++i) cv[i] = (uint32_t)seed(t, 0) + (uint32_t)i * PHI32;
  BEGIN_REC
  b3_chain(cv, (uint32_t)(seed(t, 0) >> 32), iters);
  END_REC(rec)
  out[t] = (uint64_t)cv[0] << 32 | cv[7];
}
// Pearl-C's message digest from registers: 256 message bytes, 2 permutations
__global__ void k_msg(long long iters, Rec* rec, uint64_t* out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t acc = 0;
  BEGIN_REC
#pragma unroll 1
  for (long long it = 0; it < iters; ++it) {
    uint64_t d[4];
    msg_stream(seed(t, it), d);
    acc ^= d[0] ^ d[3];
  }
  END_REC(rec)
  out[t] = acc;
}
// the same messages from mma.sync fragments in registers, through the quad transpose (1 KB of shared memory per warp)
__global__ void k_quad(long long iters, Rec* rec, uint64_t* out) {
  extern __shared__ uint64_t smem[];
  uint64_t* scratch = smem + (threadIdx.x >> 5) * 128;
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t acc = 0;
  BEGIN_REC
#pragma unroll 1
  for (long long it = 0; it < iters; ++it) {
    const uint32_t z = (uint32_t)seed(t, it);
    uint64_t d[4];
    ph::quad_msg_digest<2>([z](int a, int j, int i) { return __uint_as_float(z + (uint32_t)(32 * a + 4 * j + i) * PHI32); }, 0x21,
                           scratch, d);
    acc ^= d[0] ^ d[3];
  }
  END_REC(rec)
  out[t] = acc;
}
// Pearl-C's tile leaf from an L2-resident buffer of digests: 4,096 message bytes, 25 permutations
__global__ void k_leaf(long long iters, const uint64_t* dg, int nleaf, Rec* rec, uint64_t* out) {
  const uint32_t T = gridDim.x * blockDim.x, t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t acc = 0;
  BEGIN_REC
#pragma unroll 1
  for (long long it = 0; it < iters; ++it) {
    const uint64_t* p = dg + (size_t)((t + it * T) % nleaf) * ph::pearlc::LEAF_LANES;
    uint64_t d[4];
    ph::pearlc::tile_leaf([p](int i) { return p[i]; }, d);
    acc ^= d[0] ^ d[3];
  }
  END_REC(rec)
  out[t] = acc;
}

// ---- per-instruction throughput: 8 dependent chains in a ring, 16 rounds unrolled, OP_ROUND x 8 = 128 ops per iteration.
// sass.py holds each kernel's SASS to its opcode (ptxas 12.9 emits add.u32 as IADD3 with RZ, even through volatile inline PTX,
// and fuses two adds into one IADD3: "iadd2" is IADD3 with RZ, "iadd3" is IADD3 with three sources).
enum { OP_IADD2, OP_IADD3, OP_LOP3, OP_SHF, OP_PRMT, OP_IMAD, OP_IDP4A, OP_FFMA, OP_LOP3_IMAD, OP_LOP3_SHF, NOPS };
constexpr int OP_ROUNDS = 16, OP_PER_ITER = 8 * OP_ROUNDS;
template <int OP>
__device__ __forceinline__ uint32_t op1(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  if constexpr (OP == OP_IADD2) asm volatile("add.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  if constexpr (OP == OP_IADD3) d = a + b + c;
  if constexpr (OP == OP_LOP3) asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  if constexpr (OP == OP_SHF) asm volatile("shf.l.wrap.b32 %0, %1, %2, 7;" : "=r"(d) : "r"(a), "r"(b));
  if constexpr (OP == OP_PRMT) asm volatile("prmt.b32 %0, %1, %2, 0x6543;" : "=r"(d) : "r"(a), "r"(b));
  if constexpr (OP == OP_IMAD) asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  if constexpr (OP == OP_IDP4A) asm volatile("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(b), "r"(c), "r"(a));
  if constexpr (OP == OP_FFMA) {
    float f;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(f) : "f"(__uint_as_float(a)), "f"(0.5f), "f"(__uint_as_float(b)));
    d = __float_as_uint(f);
  }
  return d;
}
template <int OP>
__global__ void k_op(long long iters, Rec* rec, uint64_t* out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = (uint32_t)seed(t, i) | 0x3F800000u * (OP == OP_FFMA);
  BEGIN_REC
#pragma unroll 1
  for (long long it = 0; it < iters; ++it)
#pragma unroll
    for (int r = 0; r < OP_ROUNDS; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        // the mixes alternate their two opcodes by chain: whether the two share a pipe
        constexpr int A = OP == OP_LOP3_IMAD || OP == OP_LOP3_SHF ? OP_LOP3 : OP;
        constexpr int B = OP == OP_LOP3_IMAD ? OP_IMAD : OP == OP_LOP3_SHF ? OP_SHF : OP;
        x[i] = i & 1 ? op1<B>(x[i], x[(i + 1) & 7], x[(i + 2) & 7]) : op1<A>(x[i], x[(i + 1) & 7], x[(i + 2) & 7]);
      }
  END_REC(rec)
  uint64_t acc = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) acc = acc * 31 + x[i];
  out[t] = acc;
}

// ---- the epilogue: an FP8 mma.sync tile loop with Pearl-C's checked words
// Each warp owns a 32 x 64 output tile (2 regions of m16 x n64: 16 FP8 m16n8k32 per k-step, 64 FP32 accumulators) and runs
// `tiles` tiles of `ksteps` k-steps (k = 32 ksteps), operands from a shared-memory ring (lane-major, conflict-free, the same
// bytes for every warp: a stand-in for the real kernel's ldmatrix stages, without global loads). After the chain: C~ is copied,
// the peel (four BF16 m16n8k16 steps per n8 block, on stand-in factors) turns the accumulators into U, y = bf16(U) is stored,
// and with HASH each region's 32 messages (C~ ‖ U per row and chunk) go through quad_msg_digest into a digest buffer (the
// leaves are a separate pass: k_leaf's rate). Without HASH the same kernel skips the digests: the difference is the hashing's
// cost inside the kernel, at the kernel's own register pressure and warps in flight.
constexpr int EPI_STAGES = 4, EPI_A = 2 * 32 * 4, EPI_B = 32 * 16;   // words per stage: A (2 m16 blocks), B (8 n8 blocks)
__device__ __forceinline__ void mma_fp8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <int N>
struct Int {
  static constexpr int value = N;
};
template <int HASH, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) k_epi(int tiles, int ksteps, Rec* rec, uint64_t* digests, uint32_t* y, uint32_t* cap) {
  extern __shared__ uint32_t sm[];
  uint32_t* ring = sm;                                                                       // EPI_STAGES x (EPI_A + EPI_B)
  uint64_t* scratch = reinterpret_cast<uint64_t*>(sm + EPI_STAGES * (EPI_A + EPI_B)) + (threadIdx.x >> 5) * 128;
  const int lane = threadIdx.x & 31, warp = blockIdx.x * WARPS + (threadIdx.x >> 5);
  for (int i = threadIdx.x; i < EPI_STAGES * (EPI_A + EPI_B); i += blockDim.x) ring[i] = 0x38303830u ^ (mix32(i) & 0x07070707u);
  uint32_t pa[4] = {0x3F803F80u, 0x3F003F00u, 0x3E803E80u, 0x3F403F40u}, pb0 = 0x3F803F80u, pb1 = 0x3C003C00u;
  BEGIN_REC
#pragma unroll 1
  for (int tile = 0; tile < tiles; ++tile) {
    float acc[2][8][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[r][j][i] = 0.f;
#pragma unroll 1
    for (int ks = 0; ks < ksteps; ++ks) {
      const uint32_t* st = ring + (ks % EPI_STAGES) * (EPI_A + EPI_B);
      uint32_t a[2][4];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const uint4 v = reinterpret_cast<const uint4*>(st)[r * 32 + lane];
        a[r][0] = v.x, a[r][1] = v.y, a[r][2] = v.z, a[r][3] = v.w;
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 v = reinterpret_cast<const uint4*>(st + EPI_A)[q * 32 + lane];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          mma_fp8(acc[r][2 * q], a[r], v.x, v.y);
          mma_fp8(acc[r][2 * q + 1], a[r], v.z, v.w);
        }
      }
    }
    float ct[2][8][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) ct[r][j][i] = acc[r][j][i];
#pragma unroll
    for (int step = 0; step < 4; ++step)
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int j = 0; j < 8; ++j) mma_bf16(acc[r][j], pa, pb0 ^ step, pb1);
    uint32_t* yt = y + ((size_t)warp * 2 + (tile & 1)) * 1024;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 v2 = __floats2bfloat162_rn(acc[r][j][0], acc[r][j][1]);
        __nv_bfloat162 w2 = __floats2bfloat162_rn(acc[r][j][2], acc[r][j][3]);
        yt[(r * 8 + j) * 64 + lane] = *reinterpret_cast<uint32_t*>(&v2);
        yt[(r * 8 + j) * 64 + 32 + lane] = *reinterpret_cast<uint32_t*>(&w2);
      }
    if (HASH && cap && warp == 0 && tile == 0) {   // the gate's copy of warp 0's first tile: [C~, U][row][col]
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int row = 16 * r + (lane >> 2) + 8 * (i >> 1), col = 8 * j + 2 * (lane & 3) + (i & 1);
            cap[row * 64 + col] = __float_as_uint(ct[r][j][i]);
            cap[2048 + row * 64 + col] = __float_as_uint(acc[r][j][i]);
          }
    }
    if (HASH) {
      // one call per region with r fixed at compile time: a rolled loop over r indexes acc and ct at run time (local memory)
      auto region = [&](auto rc) {
        constexpr int r = decltype(rc)::value;
        uint64_t d[4];
        ph::quad_msg_digest<2>([&](int arr, int j, int i) { return arr ? acc[r][j][i] : ct[r][j][i]; }, ph::pearlc::D_MESSAGE,
                               scratch, d);
        uint64_t* out = digests + (((size_t)warp * 2 + (tile & 1)) * 64 + r * 32 + lane) * 4;
        out[0] = d[0], out[1] = d[1], out[2] = d[2], out[3] = d[3];
      };
      region(Int<0>{});
      region(Int<1>{});
    }
  }
  END_REC(rec)
}
// ---- the harness
static nvmlDevice_t g_nvml;
static bool g_nvml_ok = false;
static int g_sms;
static double g_target_ms = 100;
static char g_uuid[64];
struct Clk {
  unsigned sm = 0, mem = 0, power_mw = 0, temp = 0;
  unsigned long long thr = 0;
};
static Clk read_clk() {
  Clk c;
  if (!g_nvml_ok) return c;
  nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &c.sm);
  nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_MEM, &c.mem);
  nvmlDeviceGetCurrentClocksEventReasons(g_nvml, &c.thr);
  nvmlDeviceGetPowerUsage(g_nvml, &c.power_mw);
  nvmlDeviceGetTemperature(g_nvml, NVML_TEMPERATURE_GPU, &c.temp);
  return c;
}
static std::string clk_json(const char* k, const Clk& c) {
  char b[256];
  snprintf(b, sizeof b, "\"%s\":{\"sm_mhz\":%u,\"mem_mhz\":%u,\"reasons\":\"0x%llx\",\"power_w\":%.1f,\"temp_c\":%u}", k, c.sm, c.mem,
           c.thr, c.power_mw / 1000.0, c.temp);
  return b;
}
struct RecStats {
  double per_sm_clk, per_sm_clk_min, per_sm_clk_max, f_eff_ghz, cyc_per_block;
  int sms;
};
static RecStats rec_stats(const std::vector<Rec>& r, double per_block) {
  std::map<unsigned, std::pair<unsigned long long, unsigned long long>> span;
  std::map<unsigned, int> cnt;
  double cyc = 0, ns = 0;
  for (auto& x : r) {
    auto it = span.find(x.smid);
    if (it == span.end()) span[x.smid] = {x.c0, x.c1};
    else it->second.first = std::min(it->second.first, x.c0), it->second.second = std::max(it->second.second, x.c1);
    cnt[x.smid]++, cyc += double(x.c1 - x.c0), ns += double(x.t1 - x.t0);
  }
  RecStats s{0, 1e30, 0, cyc / ns, cyc / r.size(), (int)span.size()};
  for (auto& kv : span) {
    const double v = per_block * cnt[kv.first] / double(kv.second.second - kv.second.first);
    s.per_sm_clk += v, s.per_sm_clk_min = std::min(s.per_sm_clk_min, v), s.per_sm_clk_max = std::max(s.per_sm_clk_max, v);
  }
  s.per_sm_clk /= span.size();
  return s;
}
static float time_launch(const std::function<void(long long)>& launch, long long iters) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  launch(iters);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
  float ms;
  CK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a), cudaEventDestroy(b);
  return ms;
}
static long long calibrate(const std::function<void(long long)>& launch, long long iters, double target_ms) {
  for (;;) {
    const float ms = time_launch(launch, iters);
    if (ms >= target_ms / 4) return (long long)(iters * target_ms / ms) + 1;
    iters *= 4;
  }
}

// One timed row: calibrate to target_ms, one warm launch, one timed launch between NVML reads. `bytes` per thread per
// iteration; `single` times one thread (latency: cycles per iteration).
template <class Kern, class Go>
static void row(const char* filter, const char* group, const char* name, Kern kern, double bytes, int threads, int blocks_per_sm,
                size_t smem, Go go, const char* extra = "", double macs = 0) {
  if (filter && !strstr(name, filter) && !strstr(group, filter)) return;
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, kern));
  if (smem > 48 * 1024) CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, smem));
  const int bps = blocks_per_sm ? std::min(blocks_per_sm, occ) : occ;
  if (bps == 0) {   // e.g. the sweep's 32 warps of a kernel over 64 registers: not resident on one SM
    printf("{\"group\":\"%s\",\"name\":\"%s\",\"skipped\":\"no block of %d threads fits on an SM (%d registers)\"%s}\n", group, name,
           threads, fa.numRegs, extra);
    return;
  }
  const int blocks = threads == 1 ? 1 : g_sms * bps;
  Rec* rec;
  CK(cudaMalloc(&rec, blocks * sizeof(Rec)));
  auto launch = [&](long long it) { go(blocks, threads, smem, it, rec); };
  launch(1);
  CK(cudaDeviceSynchronize());
  const long long iters = calibrate(launch, 1, g_target_ms);
  time_launch(launch, std::max(1LL, iters / 4));
  const Clk before = read_clk();
  const float ms = time_launch(launch, iters);
  const Clk after = read_clk();
  std::vector<Rec> h(blocks);
  CK(cudaMemcpy(h.data(), rec, blocks * sizeof(Rec), cudaMemcpyDeviceToHost));
  const double per_block = bytes * threads * iters;
  const RecStats st = rec_stats(h, per_block);
  printf("{\"group\":\"%s\",\"name\":\"%s\",\"bytes_per_iter\":%.1f,\"regs\":%d,\"local_bytes\":%zu,\"smem\":%zu,\"threads\":%d,"
         "\"blocks_per_sm\":%d,\"warps_per_sm\":%d,\"iters\":%lld,\"ms\":%.4f,\"gb_s\":%.3f,\"per_sm_clk\":%.5f,\"per_sm_clk_min\":%.5f,"
         "\"per_sm_clk_max\":%.5f,\"macs_per_sm_clk\":%.2f,\"cyc_per_iter\":%.2f,\"f_eff_ghz\":%.4f,\"sms\":%d,%s,%s%s}\n",
         group, name, bytes, fa.numRegs, fa.localSizeBytes, smem, threads, bps, bps * ((threads + 31) / 32), iters, ms,
         per_block * blocks / (ms * 1e-3) / 1e9, st.per_sm_clk, st.per_sm_clk_min, st.per_sm_clk_max,
         bytes > 0 ? st.per_sm_clk * macs / bytes : rec_stats(h, macs * threads * iters).per_sm_clk, st.cyc_per_block / iters,
         st.f_eff_ghz, st.sms, clk_json("clk_before", before).c_str(), clk_json("clk_after", after).c_str(), extra);
  fflush(stdout);
  cudaFree(rec);
}

static int check_uuid(const cudaDeviceProp& prop) {
  const unsigned char* u = reinterpret_cast<const unsigned char*>(prop.uuid.bytes);
  snprintf(g_uuid, sizeof g_uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2], u[3],
           u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
  const char* want = getenv("POUW_EXPECT_UUID");
  printf("{\"group\":\"device\",\"name\":\"%s\",\"uuid\":\"%s\",\"sms\":%d,\"cc\":\"%d.%d\",\"clock_khz\":%d,\"smem_optin\":%zu,"
         "\"regs_per_sm\":%d,\"expect_uuid\":\"%s\"}\n",
         prop.name, g_uuid, prop.multiProcessorCount, prop.major, prop.minor, prop.clockRate, prop.sharedMemPerBlockOptin,
         prop.regsPerMultiprocessor, want ? want : "");
  if (!want || strcasecmp(want, g_uuid)) {
    fprintf(stderr, "REFUSED: the visible device is %s, not $POUW_EXPECT_UUID (%s)\n", g_uuid, want ? want : "unset");
    return 3;
  }
  if (nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByUUID(g_uuid, &g_nvml) == NVML_SUCCESS) g_nvml_ok = true;
  return 0;
}

int main(int argc, char** argv) {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  g_sms = prop.multiProcessorCount;
  if (int rc = check_uuid(prop)) return rc;
  const char* mode = argc > 1 ? argv[1] : "";
  if (!strcmp(mode, "check")) {
    static CheckOut h;
    device_checks(&h);
    print_checks(h, "device");
    return 0;
  }
  if (argc > 2) g_target_ms = atof(argv[2]);
  const char* filter = argc > 3 ? argv[3] : nullptr;
  const size_t maxT = (size_t)g_sms * 2048;
  uint64_t* out;
  CK(cudaMalloc(&out, maxT * 8));
  if (!strcmp(mode, "rates")) {
    const int nleaf = 4096;   // 16 MB of digests: L2-resident on a 128 MB L2
    uint64_t* dg;
    CK(cudaMalloc(&dg, (size_t)nleaf * ph::pearlc::LEAF_LANES * 8));
    CK(cudaMemset(dg, 0x5a, (size_t)nleaf * ph::pearlc::LEAF_LANES * 8));
    for (int single = 0; single < 2; ++single) {
      const int th = single ? 1 : 128;
      const char* g = single ? "latency" : "rate";
      row(filter, g, "p12: Keccak-p[1600, 12] (TurboSHAKE128), 168 B per permutation", k_perm<12>, 168, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_perm<12><<<b, t>>>(it, r, out); });
      row(filter, g, "p12x: Keccak-p[1600, 12], the other rotations (not PH_KECCAK_SHF's)", k_perm<12, !PH_KECCAK_SHF>, 168, th,
          0, 0, [&](int b, int t, size_t, long long it, Rec* r) { k_perm<12, !PH_KECCAK_SHF><<<b, t>>>(it, r, out); });
      row(filter, g, "f24: Keccak-f[1600] (SHAKE128's rate), 168 B per permutation", k_perm<24>, 168, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_perm<24><<<b, t>>>(it, r, out); });
      row(filter, g, "sha: SHA-256 compressions, chained, 64 B each", k_sha, 64, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_sha<<<b, t>>>(it, r, out); });
      row(filter, g, "b3: BLAKE3 compressions, chained, 64 B each", k_b3, 64, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_b3<<<b, t>>>(it, r, out); });
      row(filter, g, "msg: Pearl-C message digest from registers, 256 B, 2 permutations", k_msg, 256, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_msg<<<b, t>>>(it, r, out); });
      if (!single)
        row(filter, g, "quad: Pearl-C message digest from mma.sync fragments via the quad transpose, 256 B", k_quad, 256, th, 0,
            (size_t)th / 32 * 1024, [&](int b, int t, size_t s, long long it, Rec* r) { k_quad<<<b, t, s>>>(it, r, out); });
      row(filter, g, "leaf: Pearl-C tile leaf from L2, 4,096 B, 25 permutations", k_leaf, 4096, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_leaf<<<b, t>>>(it, dg, nleaf, r, out); });
    }
    // per-instruction throughput (thread-ops per SM per clock: per_sm_clk with bytes_per_iter = ops per iteration)
    const char* opn[NOPS] = {"op iadd2: IADD3 with RZ (add.u32)", "op iadd3: IADD3, three sources", "op lop3", "op shf: SHF.L.W",
                             "op prmt", "op imad", "op idp4a", "op ffma", "op lop3+imad (half each)", "op lop3+shf (half each)"};
    auto op_row = [&](int k, auto kern) {
      row(filter, "op", opn[k], kern, OP_PER_ITER, 128, 0, 0,
          [&, kern](int b, int t, size_t, long long it, Rec* r) { kern<<<b, t>>>(it, r, out); }, ",\"unit\":\"thread-ops\"");
    };
    op_row(OP_IADD2, k_op<OP_IADD2>), op_row(OP_IADD3, k_op<OP_IADD3>), op_row(OP_LOP3, k_op<OP_LOP3>);
    op_row(OP_SHF, k_op<OP_SHF>), op_row(OP_PRMT, k_op<OP_PRMT>), op_row(OP_IMAD, k_op<OP_IMAD>);
    op_row(OP_IDP4A, k_op<OP_IDP4A>), op_row(OP_FFMA, k_op<OP_FFMA>);
    op_row(OP_LOP3_IMAD, k_op<OP_LOP3_IMAD>), op_row(OP_LOP3_SHF, k_op<OP_LOP3_SHF>);
    // chains in flight: one block per SM of 1 … 32 warps (Keccak-p12, SHA-256, the quad digest)
    for (int w : {1, 2, 4, 8, 12, 16, 24, 32}) {
      char ex[64];
      snprintf(ex, sizeof ex, ",\"sweep_warps\":%d", w);
      row(filter, "sweep", "p12", k_perm<12>, 168, 32 * w, 1, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_perm<12><<<b, t>>>(it, r, out); }, ex);
      row(filter, "sweep", "sha", k_sha, 64, 32 * w, 1, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_sha<<<b, t>>>(it, r, out); }, ex);
      row(filter, "sweep", "quad", k_quad, 256, 32 * w, 1, (size_t)w * 1024,
          [&](int b, int t, size_t s, long long it, Rec* r) { k_quad<<<b, t, s>>>(it, r, out); }, ex);
    }
    return 0;
  }
  if (!strcmp(mode, "epi")) {
    // per warp 32 x 64 outputs per tile; bytes hashed per tile per thread: 2 regions x 256 B / 32 lanes x 32 = 512 B per lane
    uint64_t* dg;
    uint32_t* y;
    CK(cudaMalloc(&dg, (size_t)g_sms * 64 * 2 * 64 * 32)); CK(cudaMalloc(&y, (size_t)g_sms * 64 * 2 * 32 * 64 * 2));
    const size_t base = EPI_STAGES * (EPI_A + EPI_B) * 4;
    // the gate: warp 0's first tile, hashed in the kernel, against the header's msg_digest on the host over the same words
    uint32_t* cap;
    CK(cudaMalloc(&cap, 4096 * 4));
    bool ok = true;
    for (int w : {4, 8}) {
      CK(cudaMemset(cap, 0, 4096 * 4)); CK(cudaMemset(dg, 0, 64 * 32 * 2));
      Rec* r;
      CK(cudaMalloc(&r, sizeof(Rec)));
      if (w == 4) {
        CK(cudaFuncSetAttribute(k_epi<1, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(base + 4096)));
        k_epi<1, 4><<<1, 128, base + 4096>>>(1, 8, r, dg, y, cap);
      } else {
        CK(cudaFuncSetAttribute(k_epi<1, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(base + 8192)));
        k_epi<1, 8><<<1, 256, base + 8192>>>(1, 8, r, dg, y, cap);
      }
      CK(cudaDeviceSynchronize());
      std::vector<uint32_t> hc(4096);
      std::vector<uint64_t> hd(64 * 4);
      CK(cudaMemcpy(hc.data(), cap, 4096 * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hd.data(), dg, 64 * 32, cudaMemcpyDeviceToHost));
      int bad = 0, nonzero = 0;
      for (int x : hc) nonzero += x != 0;
      for (int reg = 0; reg < 2; ++reg)
        for (int lane = 0; lane < 32; ++lane) {
          const int row = 16 * reg + ph::quad_row(lane), h = ph::quad_chunk(lane);
          uint64_t d[4];
          ph::pearlc::msg_digest([&](int j) {
            const int a = j >> 4, col = 32 * h + 2 * (j & 15);
            return ph::lane_u32x2(hc[a * 2048 + row * 64 + col], hc[a * 2048 + row * 64 + col + 1]);
          }, d);
          for (int i = 0; i < 4; ++i) bad += d[i] != hd[(reg * 32 + lane) * 4 + i];
        }
      printf("{\"group\":\"epi_gate\",\"name\":\"epi w%d\",\"messages\":64,\"bad_words\":%d,\"nonzero_words\":%d,\"ok\":%s}\n", w,
             bad, nonzero, bad || nonzero < 4000 ? "false" : "true");
      ok = ok && !bad && nonzero >= 4000;
      cudaFree(r);
    }
    fflush(stdout);
    if (!ok) {
      fprintf(stderr, "epi gate failed: nothing timed\n");
      return 4;
    }
    for (int ks : {32, 64, 128, 256}) {
      const double macs = 2048.0 * ks;   // per thread per tile: 32 x 64 x k per warp
      char ex[96], nm[64];
      for (int hash = 0; hash < 2; ++hash) {
        snprintf(ex, sizeof ex, ",\"k\":%d,\"hash\":%d", 32 * ks, hash);
        const double bytes = hash ? 512 : 0;   // 2 regions x 32 messages x 256 B per warp, per lane
        auto launch = [&](auto kern) {
          return [&, kern](int b, int t, size_t s, long long it, Rec* r) { kern<<<b, t, s>>>((int)it, ks, r, dg, y, nullptr); };
        };
        // w4 at every resident block, then at one block per SM: the difference is how much a second block's tile loop hides
        // the first one's hashing
        for (int one : {0, 1}) {
          snprintf(nm, sizeof nm, "epi w4 k%d %s%s", 32 * ks, hash ? "hash" : "nohash", one ? " 1cta" : "");
          if (hash) row(filter, "epi", nm, k_epi<1, 4>, bytes, 128, one, base + 4 * 1024, launch(k_epi<1, 4>), ex, macs);
          else row(filter, "epi", nm, k_epi<0, 4>, bytes, 128, one, base, launch(k_epi<0, 4>), ex, macs);
        }
        snprintf(nm, sizeof nm, "epi w8 k%d %s", 32 * ks, hash ? "hash" : "nohash");
        if (hash) row(filter, "epi", nm, k_epi<1, 8>, bytes, 256, 0, base + 8 * 1024, launch(k_epi<1, 8>), ex, macs);
        else row(filter, "epi", nm, k_epi<0, 8>, bytes, 256, 0, base, launch(k_epi<0, 8>), ex, macs);
      }
    }
    return 0;
  }
  fprintf(stderr, "usage: pouw_hash_bench check | rates [ms] [filter] | epi [ms] [filter]\n");
  return 2;
}
