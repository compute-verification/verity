// What a SHA-512 warp costs the served GEMM's tensor pipe, taken apart: the served RP 5 CTA (wg_proto.cuh's gemm_r5: NM
// consumer warpgroups at Y registers and the rest at HR, the producer warp first among the rest, the MMAs wg_proto.cuh's
// own imma calls) with each of its parts optional, beside hash warps that run SHA-512 on registers alone
// (sx::compress<0, 1> on a chained block, as the co-issue probe's hash CTAs do), until the consumers are done.
//
// A configuration (CFGS) sets:
//   NWG    the CTA's warpgroups (gemm_r5: 4), NM of them consumers; the register file is 512 registers for each thread
//          of one warpgroup, so NM Y + (NWG - NM) HR is at most 512
//   NM     consumer warpgroups: 1 (a 64 x BN tile) or 2 (128 x BN, gemm_r5's)
//   FORM   n128: one m64n128k32 a k32 atom (BN 128); alt: gemm_r5's 128 x 256 tile, two m64n128k32 an atom into
//          accumulators 0..63 and 64..127 in turn (imma_atom<256, true>); chain: the same eight MMAs a k-tile, the four
//          into 0..63 first, then the four into 64..127 (no n256: ptxas sizes a m64n256k32 by the launch's registers, 128
//          at 512 threads, whatever setmaxnreg gives the consumers)
//   flags  res: the operands resident in one stage of shared memory, the consumers free-running (no barriers);
//          bar: the producer warp hands the stages over on gemm_r5's full and empty mbarriers, loading nothing;
//          load: the producer loads each k-tile with TMA from A (M x K) and B (N x K), tiles in gemm_r5's static
//          schedule (every gridDim.x-th, rows then columns);
//          epi: at each tile's end (K / 128 k-tiles), wgmma.wait_group 0 and gemm_r5's direct O stores (s32, int2 a
//          pair) into an M x N buffer
// A launch's arm: hash warps 0 (the GEMM alone) or NHW of the CTA's free warps (the producer's warpgroup's other three,
// then the next warpgroups'), and the hash alone at the configuration's most hash warps (no consumers; each hash warp
// HREPS compressions).
//
// Per launch, from each CTA's consumer warp 0 (clock64 and %globaltimer at its first and last MMA) and its hash warps'
// compression counts: theta, the INT8 MACs a SM-clock over 4096; the hash's compressions a SM-clock over the bar (sx's
// 4.355 G/s at 1830 MHz over 132 SMs, 0.018029); the SM clock. Each line of output is JSON; the arms are medians of REPS
// launches.
//
//   coresident [REPS] [KTILES] [HREPS]   (KTILES: a resident configuration's k-tiles a CTA)
//
// Build (sm_90a): nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -o coresident coresident.cu -lcuda
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../../torch_gpu/sha512_rows/sha512x.cuh"
#include "wg_proto.cuh"

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));         \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

using u64 = unsigned long long;
using u32 = uint32_t;
using wgp::BK;

enum Form { N128 = 0, ALT = 1, CHAIN = 2 };
enum Flag { BAR = 1, LOAD = 2, EPI = 4 };
constexpr int STAGES = 4;
constexpr double BAR_RATE = 0.018029;   // compressions a SM-clock: 4.355e9 / (132 x 1830e6)

struct Stamp {
  u64 t0, t1, c0, c1;   // the consumers' first and last MMA (or the hash's window, with no consumers)
  u64 comps;            // the CTA's compressions
  u64 sink;
};

struct alignas(128) Prm {
  CUtensorMap a, b;
  int* O;
  Stamp* st;
  int M, N, K, groups, nhw, hreps;
};

__device__ __forceinline__ u64 gtimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

template <int NM, int FORM> struct Shape {
  static constexpr int BN = FORM == N128 ? 128 : 256, TMC = 64 * NM, NA = BN / 2;
  static constexpr int A_BYTES = TMC * BK, STAGE = (TMC + BN) * BK;
  static constexpr double MACS = (double)TMC * BN * BK;   // a k-tile's
};
template <int NM, int FORM, int FL> constexpr int smem_bytes() {
  return 1024 + (FL & LOAD ? STAGES : 1) * Shape<NM, FORM>::STAGE + 16 * STAGES + 16;
}

template <int NM, int FORM, int FL, int Y, int HR, int NWG>
__global__ void __launch_bounds__(NWG * 128, 1) cta(const __grid_constant__ Prm p) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using S = Shape<NM, FORM>;
  constexpr int NT = NWG * 128, BASE = 65536 / NT / 8 * 8;   // the launch's registers a thread, as setmaxnreg counts them
  static_assert(NM * Y + (NWG - NM) * HR <= 512, "the warpgroups' registers exceed the register file");
  static_assert(HR <= BASE && Y >= BASE, "the consumers take registers, the rest give them");
  static_assert(FORM != N128 || Y >= 96, "n128's 64 accumulators");
  constexpr bool bar = FL & (BAR | LOAD);
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t base = (wgp::smem_at(smem) + 1023) & ~1023u;
  const uint32_t bars = base + (FL & LOAD ? STAGES : 1) * S::STAGE;   // full 0..3, empty 0..3, then done
  u32* const done = reinterpret_cast<u32*>(smem + (bars - wgp::smem_at(smem)) + 16 * STAGES);
  const int tid = threadIdx.x, wg = tid / 128, warp = tid / 32, lane = tid % 32;
  const int tiles_m = p.M / S::TMC, tiles_n = p.N / S::BN, tiles = tiles_m * tiles_n, TK = p.K / BK;
  const int my_tiles = (int)blockIdx.x < tiles ? (tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
  const int groups = FL & LOAD ? my_tiles * TK : p.groups;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) {
      wgp::mbar_init(bars + 8 * s, 1);
      wgp::mbar_init(bars + 8 * (STAGES + s), NM * 4);
    }
    *done = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (!(FL & LOAD))   // resident operands: any bytes do
    for (int i = tid; i < S::STAGE / 4; i += NT) reinterpret_cast<u32*>(smem + (base - wgp::smem_at(smem)))[i] = i * 2654435761u;
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  Stamp* const out = p.st + blockIdx.x;

  if (wg >= NM) {
    if constexpr (HR < BASE) wgp::regs_dec<HR>();
    const int w = warp - 4 * NM;   // 0: the producer warp; 1 ..: hash warps
    if (w == 0) {
      if (lane == 0 && bar && groups > 0) {
        if (FL & LOAD) { wgp::tma_prefetch(&p.a); wgp::tma_prefetch(&p.b); }
        int s = 0;
        uint32_t ph = 0;
        for (int g = 0; g < groups; g++) {
          wgp::mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          if constexpr (FL & LOAD) {
            const int t = (int)blockIdx.x + (g / TK) * (int)gridDim.x, kt = g % TK, tm = t / tiles_n, tn = t % tiles_n;
            const uint32_t st = base + s * S::STAGE, full = bars + 8 * s;
            wgp::mbar_expect_tx(full, S::STAGE);
            wgp::tma_load(st, &p.a, full, kt * BK, tm * S::TMC);
            wgp::tma_load(st + S::A_BYTES, &p.b, full, kt * BK, tn * S::BN);
          } else {
            wgp::mbar_arrive(bars + 8 * s);
          }
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
      }
      asm volatile("exit;");
    }
    if (w > p.nhw) asm volatile("exit;");
    // a hash warp: SHA-512 over a chained block, in registers, until the consumers are done (or HREPS with none)
    const u64 g = (u64)blockIdx.x * NT + tid;
    u64 st[8], wv[16];
    sx::iv(st);
#pragma unroll
    for (int i = 0; i < 16; ++i) wv[i] = g * 16 + i;
    u64 n = 0, t0 = 0, c0 = 0;
    if (w == 1 && lane == 0) { t0 = gtimer(); c0 = clock64(); }
    const bool alone = groups == 0;
    for (;;) {
      if (alone ? n >= (u64)p.hreps : *reinterpret_cast<volatile u32*>(done) >= (u32)(NM * 4)) break;
#pragma unroll 1
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int i = 0; i < 8; ++i) wv[i] ^= st[i];
        sx::compress<0, 1>(st, wv);
      }
      n += 4;
    }
    if (lane == 0) atomicAdd(&out->comps, 32 * n);
    if (st[0] == 0x0123456789abcdefull) out->sink = st[1];
    if (alone && w == 1 && lane == 0) { out->t0 = t0; out->c0 = c0; out->t1 = gtimer(); out->c1 = clock64(); }
    asm volatile("exit;");
  }

  if constexpr (Y > BASE) wgp::regs_inc<Y>();
  const int x = tid % 128;
  int32_t d[S::NA];
#pragma unroll
  for (int i = 0; i < S::NA; i++) d[i] = 0;
  int s = 0, pend = -1;
  uint32_t ph = 0;
  u64 t0 = 0, c0 = 0;
  if (tid == 0) { t0 = gtimer(); c0 = clock64(); }
  for (int g = 0; g < groups; g++) {
    if (bar) wgp::mbar_wait_warp(bars + 8 * s, ph);
    const uint32_t stg = base + (FL & LOAD ? s : 0) * S::STAGE, a0 = stg + wg * 64 * BK, b0 = stg + S::A_BYTES;
    const uint32_t sd = g % TK != 0;
    WGP_FENCE;
    if constexpr (FORM == N128) {
#pragma unroll
      for (int i = 0; i < 4; i++) wgp::imma<128>(d, wgp::desc_sw128(a0 + 32 * i), wgp::desc_sw128(b0 + 32 * i), i > 0 || sd);
    } else if constexpr (FORM == ALT) {
#pragma unroll
      for (int i = 0; i < 4; i++) wgp::imma_atom<256, true>(d, a0 + 32 * i, b0 + 32 * i, i > 0 || sd);
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) wgp::imma<128>(d, wgp::desc_sw128(a0 + 32 * i), wgp::desc_sw128(b0 + 32 * i), i > 0 || sd);
#pragma unroll
      for (int i = 0; i < 4; i++)
        wgp::imma<128>(d + 64, wgp::desc_sw128(a0 + 32 * i), wgp::desc_sw128(b0 + 128 * BK + 32 * i), i > 0 || sd);
    }
    WGP_COMMIT;
    WGP_WAIT(1);
    if (bar && pend >= 0 && lane == 0) wgp::mbar_arrive(bars + 8 * (STAGES + pend));
    pend = s;
    if (++s == STAGES) { s = 0; ph ^= 1; }
    if constexpr (FL & EPI) {
      if (g % TK == TK - 1) {
        WGP_WAIT(0);
        wgp::acc_fence<S::NA>(d);
        if (bar && lane == 0) wgp::mbar_arrive(bars + 8 * (STAGES + pend));
        pend = -1;
        const int t = ((int)blockIdx.x + (g / TK) * (int)gridDim.x) % tiles, tm = t / tiles_n, tn = t % tiles_n;
        const int rloc = wg * 64 + 16 * (x / 32) + (x % 32) / 4, cq = 2 * (x % 4);
        int* const o0 = p.O + (size_t)(tm * S::TMC + rloc) * p.N + tn * S::BN + cq;
        int* const o1 = o0 + (size_t)8 * p.N;
#pragma unroll
        for (int j = 0; j < S::NA / 4; j++) {
          *reinterpret_cast<int2*>(o0 + 8 * j) = make_int2(d[4 * j], d[4 * j + 1]);
          *reinterpret_cast<int2*>(o1 + 8 * j) = make_int2(d[4 * j + 2], d[4 * j + 3]);
          if (j % 8 == 7) wgp::ld_fence();
        }
      }
    }
  }
  WGP_WAIT(0);
  wgp::acc_fence<S::NA>(d);
  if (bar && pend >= 0 && lane == 0) wgp::mbar_arrive(bars + 8 * (STAGES + pend));
  if (tid == 0 && groups > 0) { out->t0 = t0; out->c0 = c0; out->t1 = gtimer(); out->c1 = clock64(); }
  u32 xr = 0;
#pragma unroll
  for (int i = 0; i < S::NA; i++) xr ^= (u32)d[i];
  if (xr == 0x9e3779b9u) out->sink = xr;
  __syncwarp();
  if (lane == 0) atomicAdd(done, 1u);
#endif
}

// ---------------------------------------------------------------------------------------------------------- host
struct Cfg {
  const char* name;
  int nm, form, fl, y, hr, nwg, smem;
  double macs;   // a k-tile's, a CTA
  void (*k)(Prm);
};
#define CFG(NAME, NM, FORM, FL, Y, HR, NWG) \
  {NAME, NM, FORM, FL, Y, HR, NWG, smem_bytes<NM, FORM, FL>(), Shape<NM, FORM>::MACS, cta<NM, FORM, FL, Y, HR, NWG>}
// The consumers' register budget Y and, for each, the tiles and CTA sizes it fits, the hash warps at the registers left:
// res, the operands resident (the ceiling), and le, TMA loads and the epilogue (the served call's parts). gemm_r5 is
// y160_m2_alt_w4 (HR 96, 7 hash warps beside the producer).
#define BOTH(NAME, NM, FORM, Y, HR, NWG) CFG(NAME "_res", NM, FORM, 0, Y, HR, NWG), CFG(NAME "_le", NM, FORM, LOAD | EPI, Y, HR, NWG)
static const Cfg CFGS[] = {
    BOTH("y232_m1_alt_w4", 1, ALT, 232, 88, 4),  BOTH("y232_m1_alt_w5", 1, ALT, 232, 64, 5),
    BOTH("y200_m1_alt_w5", 1, ALT, 200, 72, 5),  BOTH("y176_m2_alt_w4", 2, ALT, 176, 80, 4),
    BOTH("y176_m1_alt_w5", 1, ALT, 176, 80, 5),  BOTH("y160_m2_alt_w4", 2, ALT, 160, 96, 4),
    BOTH("y160_m2_alt_w5", 2, ALT, 160, 64, 5),  BOTH("y128_m2_n128_w4", 2, N128, 128, 128, 4),
    BOTH("y128_m2_n128_w5", 2, N128, 128, 80, 5), BOTH("y128_m2_n128_w6", 2, N128, 128, 64, 6),
    BOTH("y104_m2_n128_w6", 2, N128, 104, 72, 6),
};

static CUtensorMap tmap(const void* ptr, int depth, int rows, int box_rows) {
  CUtensorMap m;
  const cuuint64_t dims[2] = {(cuuint64_t)depth, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)depth};
  const cuuint32_t box[2] = {128, (cuuint32_t)box_rows}, es[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, es,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled: %d\n", (int)r); exit(3); }
  return m;
}

__global__ void fill_k(uint8_t* p, size_t n, u64 seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    u64 z = seed + (i + 1) * 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    p[i] = (uint8_t)(z >> 56);
  }
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 5, groups = argc > 2 ? atoi(argv[2]) : 4096, hreps = argc > 3 ? atoi(argv[3]) : 2048;
  const int M = 16384, N = 28672, K = 4096;   // Llama 3 8B's gate-up at M 16,384
  cudaDeviceProp dp;
  CK(cudaGetDeviceProperties(&dp, 0));
  const int nsm = dp.multiProcessorCount;
  printf("{\"device\": \"%s\", \"sms\": %d, \"M\": %d, \"N\": %d, \"K\": %d, \"groups\": %d, \"hreps\": %d}\n", dp.name, nsm, M, N, K,
         groups, hreps);
  uint8_t *A, *B;
  int* O;
  Stamp* st;
  CK(cudaMalloc(&A, (size_t)M * K));
  CK(cudaMalloc(&B, (size_t)N * K));
  CK(cudaMalloc(&O, (size_t)M * N * 4));
  CK(cudaMalloc(&st, nsm * sizeof(Stamp)));
  fill_k<<<1024, 256>>>(A, (size_t)M * K, 1);
  fill_k<<<1024, 256>>>(B, (size_t)N * K, 2);
  CK(cudaDeviceSynchronize());
  for (const Cfg& c : CFGS) {
    CK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, c.smem));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, c.k));
    printf("{\"cfg\": \"%s\", \"nm\": %d, \"bn\": %d, \"y\": %d, \"hr\": %d, \"threads\": %d, \"regs\": %d, \"local\": %zu, \"smem\": %d}\n",
           c.name, c.nm, c.form == N128 ? 128 : 256, c.y, c.hr, 128 * c.nwg, fa.numRegs, fa.localSizeBytes, c.smem);
  }
  auto launch = [&](const Cfg& c, int g, int nhw) {
    Prm p{};
    const int bn = c.form == N128 ? 128 : 256;
    p.a = tmap(A, K, M, 64 * c.nm);
    p.b = tmap(B, K, N, bn);
    p.O = O;
    p.st = st;
    p.M = M; p.N = N; p.K = K; p.groups = g; p.nhw = nhw; p.hreps = hreps;
    CK(cudaMemset(st, 0, nsm * sizeof(Stamp)));
    c.k<<<nsm, 128 * c.nwg, c.smem>>>(p);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<Stamp> s(nsm);
    CK(cudaMemcpy(s.data(), st, nsm * sizeof(Stamp), cudaMemcpyDeviceToHost));
    return s;
  };
  // one arm: theta and the hash's share of the bar a SM-clock, each over the CTAs' own windows; the SM clock; the wall
  auto arm = [&](const Cfg& c, int g, int nhw, const char* what) {
    std::vector<double> th, hb, mhz, ms;
    launch(c, g, nhw);
    for (int r = 0; r < reps; r++) {
      auto s = launch(c, g, nhw);
      const int bn = c.form == N128 ? 128 : 256, tiles = (M / (64 * c.nm)) * (N / bn), TK = K / BK;
      double cyc = 0, ns = 0, macs = 0, comps = 0;
      u64 a = ~0ull, b = 0;
      for (int i = 0; i < nsm; i++) {
        cyc += (double)(s[i].c1 - s[i].c0);
        ns += (double)(s[i].t1 - s[i].t0);
        a = std::min(a, s[i].t0);
        b = std::max(b, s[i].t1);
        comps += (double)s[i].comps;
        const int myt = i < tiles ? (tiles - 1 - i) / nsm + 1 : 0;
        macs += g == 0 ? 0 : c.macs * ((c.fl & LOAD) ? (double)myt * TK : (double)g);
      }
      th.push_back(macs / cyc / 4096);
      hb.push_back(comps / cyc / BAR_RATE);
      mhz.push_back(cyc / ns * 1e3);
      ms.push_back((double)(b - a) * 1e-6);
    }
    printf("{\"cfg\": \"%s\", \"arm\": \"%s\", \"nhw\": %d, \"theta\": %.4f, \"hash_bar\": %.4f, \"mhz\": %.0f, \"ms\": %.4f}\n", c.name,
           what, nhw, median(th), median(hb), median(mhz), median(ms));
    fflush(stdout);
  };
  for (const Cfg& c : CFGS) {
    const int maxhw = (c.nwg - c.nm) * 4 - 1;
    arm(c, groups, 0, "gemm");
    if (maxhw > 7) arm(c, groups, 7, "gemm+hash");
    arm(c, groups, maxhw, "gemm+hash");
    if (!(c.fl & LOAD)) arm(c, 0, maxhw, "hash");
  }
  return 0;
}
