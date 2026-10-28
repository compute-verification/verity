// `int8_fused.scaled_mm`'s product on the H100 and H200 (sm_90a), word for word `exactlin_scaled_mm_tma`'s:
// y[i, n] = cvt.rn.bf16.f32((cvt.rn.f32.s32(Y[i, n]) * s_a[i]) * s_w[n]) with Y = X W^T in s32, X (M x K) and W (N x K)
// row-major int8.  The s32 sum is exact in any order (|Y| <= 128 * 128 * K < 2^31 for K < 2^17) and the epilogue is the
// same three roundings in the same order, built with -fmad=false and no FTZ flag, so y is bit-identical to the Triton
// kernel's at every config.
//
// Persistent and warp-specialized, CUTLASS's ping-pong schedule: one CTA an SM (the shared memory), a producer warpgroup
// whose first thread issues every TMA into STAGES stages of 128 bytes of depth (X's WM rows, then W's BN rows, SWIZZLE_128B),
// and two consumer warpgroups that take the CTA's tiles in turn, each a whole WM x BN tile with WM x BN / 128 s32
// accumulators a thread (wgmma m64nBNk32 .s32.s8.s8, WM / 64 of them an atom).  Each consumer's mainloop starts once the
// other's has issued its last stage (the `order` barriers), so the tensor pipe runs one tile's mainloop while the other
// consumer's epilogue converts and stores the tile before it.  The producer holds 40 registers and the consumers 232.
// int8_wgmma_ppm2_*: clusters of two CTAs on M-adjacent tiles of one column of tiles, the pair sharing each W box by TMA
// multicast (each CTA loads half of it into both).  Barrier arrivals stay at CTA scope (a cluster-scope release puts a
// MEMBAR.ALL.GPU on every consumer warp's stage).  M, N and K need not be multiples of the tile: a box past the matrix reads
// zeros, which add nothing, and the epilogue stores rows < M and columns < N alone (N even, K a multiple of 16).
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "int8_wgmma_acc.h"

#define DEV __device__ __forceinline__

struct alignas(128) Params {
  CUtensorMap a, b;  // X: M rows of K bytes, boxes of WM rows; W: N rows, boxes of BN / CL rows
  const float* sa;   // s_a [M]
  const float* sw;   // s_w [N]
  uint16_t* y;       // y [M, N], bf16 words, row-major
  int M, N, K, group;
};

// ---------------------------------------------------------------- mbarriers, TMA, wgmma
DEV uint32_t smem_at(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEV void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEV void mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }
DEV void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
// try_wait's predicate is per thread: the warp reconverges before the .aligned wgmma that follow
DEV void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
DEV void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
DEV void tma_load_mc(uint32_t dst, const void* map, uint32_t bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask)
               : "memory");
}
DEV uint32_t cta_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DEV void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// an arrive on rank's copy of `bar` at the default .release.cta: the stage was read only by this CTA's retired wgmma, and
// the peer's TMA refills it through the async proxy
DEV void mbar_arrive_at(uint32_t bar, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(bar), "r"(rank));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(r) : "memory");
}
template <int R> DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }

#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

// K-major operands with 128-byte swizzling: 8-row groups 1,024 bytes apart, the start advanced 32 bytes per k32 atom
DEV uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}

// d = A B + (sd ? d : 0): one s8 atom (k = 32) of a 64 x N warpgroup tile.  Accumulator i of thread x is row
// 16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
template <int N> DEV void imma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  static_assert(N == 256 || N == 128, "n256 or n128");
  if constexpr (N == 256)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
                 ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 " ACC64_S
                 ", %64, %65, p;\n}" : ACC64_R : "l"(da), "l"(db), "r"(sd));
}
// after WG_WAIT: the accumulators are read only from here on
template <int NA> DEV void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}

// ---------------------------------------------------------------- the GEMM
constexpr int BK = 128, SMEM_MAX = 232448, MAX_STAGES = 8;

// a WM x BN tile, CO = 0: a consumer's own (ping-pong); CO = 1: both consumers', rows 64 wg .. of it (cooperative)
template <int WM, int BN, int CO> struct Tile {
  static constexpr int R = CO ? WM / 2 : WM;  // a consumer's rows
  static_assert(R * BN == 16384 && (R == 64 || R == 128), "a 64 x 256 or 128 x 128 tile a consumer");
  static constexpr int NH = R / 64, NA = BN / 2;  // m64 atoms a k32 step; accumulators a thread an atom
  static constexpr uint32_t A_BYTES = WM * BK, STAGE = (WM + BN) * BK;
  // the stages from a 1,024-byte-aligned base (SWIZZLE_128B's period), then full[s], empty[s] and order[2]
  static constexpr int FIT = (SMEM_MAX - 1024 - 16) / (STAGE + 16);
  static constexpr int STAGES = FIT < MAX_STAGES ? FIT : MAX_STAGES;
  static constexpr int SMEM = 1024 + STAGES * STAGE + 16 * STAGES + 16;
  static_assert(STAGE % 1024 == 0 && A_BYTES % 1024 == 0 && SMEM <= SMEM_MAX, "1,024-byte-aligned boxes in 227 KiB");
};

DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

// two adjacent columns of one row, each exactlin's: cvt.rn.f32.s32, mul.rn by s_a, mul.rn by s_w, cvt.rn.bf16.f32
DEV uint32_t words(int32_t a, int32_t b, float sr, float c0, float c1) {
  const float ya = __fmul_rn(__fmul_rn(__int2float_rn(a), sr), c0), yb = __fmul_rn(__fmul_rn(__int2float_rn(b), sr), c1);
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(ya)) | (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(yb)) << 16;
}

template <int CL> DEV void release(uint32_t bar) {
  if constexpr (CL == 1) {
    mbar_arrive(bar);
  } else {
    mbar_arrive_at(bar, 0);
    mbar_arrive_at(bar, 1);
  }
}

// CTA c (a pair's with CL = 2) takes units c, c + G, ... (G the CTAs, or pairs), the j-th of them by consumer j % 2 (CO = 0)
// or by both (CO = 1).  The producer fills the ring in that order, so unit j's k-block kb is the ring's fill j KB + kb.  A
// unit is one tile (CL = 1) or a pair of M-adjacent tiles, CTA r taking tile row 2 u + r (zeros past M, stored nowhere) and
// loading half r of W's box.
template <int WM, int BN, int CO, int CL> DEV void body(const Params& p) {
  using T = Tile<WM, BN, CO>;
  constexpr int STAGES = T::STAGES, NH = T::NH, NA = T::NA, NC = CO ? 2 : 1;  // NC: the consumers that read a stage
  static_assert(CL == 1 || CL == 2, "a CTA or a pair");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t base = (smem_at(smem) + 1023) & ~1023u, bars = base + STAGES * T::STAGE, order = bars + 16 * STAGES;
  const int M = p.M, N = p.N, KB = (p.K + BK - 1) / BK;
  const int tiles_m = (M + WM - 1) / WM, tiles_n = (N + BN - 1) / BN, rows_u = (tiles_m + CL - 1) / CL, units = rows_u * tiles_n;
  const int tid = threadIdx.x, wg = tid / 128, rank = CL == 1 ? 0 : (int)cta_rank();
  const int first_u = blockIdx.x / CL, step_u = gridDim.x / CL;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * (STAGES + s), 4 * NC * CL);
    }
    mbar_init(order, 4);
    mbar_init(order + 8, 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if constexpr (CL == 1)
    __syncthreads();
  else
    cluster_sync();

  if (wg == 2) {
    // the producer exits rather than returns: a path back would join its register budget with the consumers'
    regs_dec<40>();
    if (tid == 256) {
      tma_prefetch(&p.a);
      tma_prefetch(&p.b);
      constexpr uint32_t B_PART = BN / CL * BK;
      int s = 0;
      uint32_t ph = 0;
      for (int t = first_u; t < units; t += step_u) {
        int tm, tn;
        raster(t, rows_u, tiles_n, p.group, tm, tn);
        tm = tm * CL + rank;
        for (int kb = 0; kb < KB; kb++) {
          const uint32_t st = base + s * T::STAGE, full = bars + 8 * s;
          mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          mbar_expect_tx(full, T::STAGE);
          tma_load(st, &p.a, full, kb * BK, tm * WM);
          const uint32_t bdst = st + T::A_BYTES + rank * B_PART;
          const int brow = tn * BN + rank * (BN / CL);
          if constexpr (CL == 1)
            tma_load(bdst, &p.b, full, kb * BK, brow);
          else
            tma_load_mc(bdst, &p.b, full, kb * BK, brow, 0x3);
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
      }
      // the peer's consumers arrive on this CTA's barriers: it stays until every stage it loaded is released by both
      if constexpr (CL == 2)
        for (int i = 0; i < STAGES; i++) {
          mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
    }
    asm volatile("exit;");
  }
  regs_inc<232>();

  const int x = tid % 128, lane = tid % 32;
  int32_t d[NH][NA];
  int i = 0;
  constexpr int STEP = CO ? 1 : 2;
  for (int t = first_u + (CO ? 0 : wg) * step_u, j = CO ? 0 : wg; t < units; t += STEP * step_u, j += STEP, i++) {
    int tm, tn;
    raster(t, rows_u, tiles_n, p.group, tm, tn);
    tm = tm * CL + rank;
    // ping-pong: the other consumer's mainloop of unit j - 1 has issued its last stage, so every fill before this unit's
    // first has been waited on and each parity wait below is on the fill it names
    if (!CO && j > 0) mbar_wait_warp(order + 8 * (1 - wg), (uint32_t)(wg == 0 ? i - 1 : i) & 1);
    const uint32_t g = (uint32_t)j * KB;
    int s = g % STAGES, sp = 0;
    uint32_t ph = (g / STAGES) & 1;
    for (int kb = 0; kb < KB; kb++) {
      mbar_wait_warp(bars + 8 * s, ph);
      const uint32_t b0 = base + s * T::STAGE + T::A_BYTES, a0 = base + s * T::STAGE + (CO ? wg * T::R * BK : 0);
      WG_FENCE;
#pragma unroll
      for (int k = 0; k < 4; k++)
#pragma unroll
        for (int h = 0; h < NH; h++)
          imma<BN>(d[h], desc_sw128(a0 + h * 64 * BK + 32 * k), desc_sw128(b0 + 32 * k), kb > 0 || k > 0);
      WG_COMMIT;
      WG_WAIT(1);
      if (kb > 0 && lane == 0) release<CL>(bars + 8 * (STAGES + sp));
      sp = s;
      if (++s == STAGES) { s = 0; ph ^= 1; }
    }
    if (!CO && lane == 0) mbar_arrive(order + 8 * wg);
    WG_WAIT(0);
#pragma unroll
    for (int h = 0; h < NH; h++) acc_fence<NA>(d[h]);
    if (lane == 0) release<CL>(bars + 8 * (STAGES + sp));

#pragma unroll
    for (int h = 0; h < NH; h++) {
      const int row = tm * WM + (CO ? wg * T::R : 0) + 64 * h + 16 * (x / 32) + (x % 32) / 4, col0 = tn * BN + 2 * (x % 4);
      const bool live0 = row < M, live1 = row + 8 < M;
      const float r0 = live0 ? p.sa[row] : 0.0f, r1 = live1 ? p.sa[row + 8] : 0.0f;
      uint32_t* y0 = reinterpret_cast<uint32_t*>(p.y + (size_t)row * N);
      uint32_t* y1 = reinterpret_cast<uint32_t*>(p.y + (size_t)(row + 8) * N);
#pragma unroll
      for (int q = 0; q < NA / 4; q++) {
        const int col = col0 + 8 * q;
        if (col < N) {
          const float2 c = *reinterpret_cast<const float2*>(p.sw + col);
          if (live0) y0[col / 2] = words(d[h][4 * q], d[h][4 * q + 1], r0, c.x, c.y);
          if (live1) y1[col / 2] = words(d[h][4 * q + 2], d[h][4 * q + 3], r1, c.x, c.y);
        }
      }
    }
  }
}

#define KERNELS(S, WM, BN, CO)                                                                                           \
  extern "C" __global__ void __launch_bounds__(384, 1) int8_wgmma_##S##_##WM##x##BN(const __grid_constant__ Params p) {  \
    body<WM, BN, CO, 1>(p);                                                                                              \
  }                                                                                                                      \
  extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1)                                         \
      int8_wgmma_##S##m2_##WM##x##BN(const __grid_constant__ Params p) {                                                 \
    body<WM, BN, CO, 2>(p);                                                                                              \
  }
KERNELS(pp, 128, 128, 0)
KERNELS(pp, 64, 256, 0)
KERNELS(co, 128, 256, 1)

// ---------------------------------------------------------------- the host side (int8_wgmma.py's ctypes calls)
// Variants, in this order: ping-pong 128x128, its pairs, 64x256, its pairs, cooperative 128x256, its pairs.
namespace {
constexpr int NV = 6;
constexpr int VWM[NV] = {128, 128, 64, 64, 128, 128}, VBN[NV] = {128, 128, 256, 256, 256, 256}, VCL[NV] = {1, 2, 1, 2, 1, 2};
using T0 = Tile<128, 128, 0>;
using T1 = Tile<64, 256, 0>;
using T2 = Tile<128, 256, 1>;
constexpr int VSMEM[NV] = {T0::SMEM, T0::SMEM, T1::SMEM, T1::SMEM, T2::SMEM, T2::SMEM};
constexpr int VSTAGES[NV] = {T0::STAGES, T0::STAGES, T1::STAGES, T1::STAGES, T2::STAGES, T2::STAGES};
constexpr int MAX_DEV = 64;
const void* const VFN[NV] = {(const void*)int8_wgmma_pp_128x128, (const void*)int8_wgmma_ppm2_128x128,
                             (const void*)int8_wgmma_pp_64x256,  (const void*)int8_wgmma_ppm2_64x256,
                             (const void*)int8_wgmma_co_128x256, (const void*)int8_wgmma_com2_128x256};
// per device: SMs, and per variant the CTAs (or pairs) the card holds at once; 0 until the device's first call
int g_sms[MAX_DEV], g_most[MAX_DEV][NV];

int setup(int dev) {
  if (g_sms[dev]) return 0;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 1;
  for (int v = 0; v < NV; v++) {
    if (cudaFuncSetAttribute(VFN[v], cudaFuncAttributeMaxDynamicSharedMemorySize, VSMEM[v]) != cudaSuccess) return 2;
    if (cudaFuncSetAttribute(VFN[v], cudaFuncAttributePreferredSharedMemoryCarveout, 100) != cudaSuccess) return 3;
    if (VCL[v] == 1) {
      g_most[dev][v] = sms;
    } else {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(2, 1, 1);
      cfg.blockDim = dim3(384, 1, 1);
      cfg.dynamicSmemBytes = VSMEM[v];
      int n = 0;
      if (cudaOccupancyMaxActiveClusters(&n, VFN[v], &cfg) != cudaSuccess || n < 1) return 4;
      g_most[dev][v] = n;
    }
  }
  g_sms[dev] = sms;
  return 0;
}

// a row-major (rows x depth) int8 matrix as the kernel loads it: boxes of box_rows x 128 bytes, SWIZZLE_128B, zeros past it
int tmap(CUtensorMap* m, const void* ptr, int rows, int depth, int box_rows) {
  const cuuint64_t dims[2] = {(cuuint64_t)depth, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)depth};
  const cuuint32_t box[2] = {128, (cuuint32_t)box_rows}, estr[2] = {1, 1};
  return cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS;
}
}  // namespace

// each variant's (WM, BN, cluster, stages, shared bytes), five ints a variant; returns the variant count
extern "C" int int8_wgmma_layout(int* out) {
  for (int v = 0; v < NV; v++) {
    out[5 * v] = VWM[v];
    out[5 * v + 1] = VBN[v];
    out[5 * v + 2] = VCL[v];
    out[5 * v + 3] = VSTAGES[v];
    out[5 * v + 4] = VSMEM[v];
  }
  return NV;
}

// y = exactlin(xq W^T) on `stream` with variant v, at most `ctas` CTAs (0: every SM) and raster groups of `group` tile rows
// (pairs of them with a cluster); returns 0, or a nonzero code naming the failed step, and writes the grid to *grid
extern "C" int int8_wgmma_launch(int v, const void* xq, const void* w, const float* sa, const float* sw, void* y, int M, int N,
                                 int K, int group, int ctas, void* stream, int* grid) {
  if (v < 0 || v >= NV || M < 1 || N < 2 || N % 2 || K < 16 || K % 16 || group < 1) return 10;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev >= MAX_DEV) return 11;
  if (int e = setup(dev)) return 20 + e;
  const int wm = VWM[v], bn = VBN[v], cl = VCL[v];
  Params p;
  if (tmap(&p.a, xq, M, K, wm) || tmap(&p.b, w, N, K, bn / cl)) return 12;
  p.sa = sa;
  p.sw = sw;
  p.y = static_cast<uint16_t*>(y);
  p.M = M;
  p.N = N;
  p.K = K;
  p.group = group;
  const int units = ((M + wm - 1) / wm + cl - 1) / cl * ((N + bn - 1) / bn);
  int cap = cl == 1 ? g_sms[dev] : g_most[dev][v];
  if (ctas > 0) cap = min(cap, cl == 1 ? ctas : max(ctas / 2, 1));
  // the fewest CTAs (pairs) that keep the same rounds of units
  const int rounds = (units + cap - 1) / cap, n = (units + rounds - 1) / rounds;
  *grid = n * cl;
  void* args[] = {&p};
  if (cudaLaunchKernel(VFN[v], dim3(n * cl), dim3(384), args, VSMEM[v], static_cast<cudaStream_t>(stream)) != cudaSuccess)
    return 13;
  return 0;
}
