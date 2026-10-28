// One exact Strassen level of the int8 GEMM on sm_90a, built from CUTLASS 3.x sm90 pieces (orbit.py is its reference):
// O [M, N] s32 = A [M, K] s8 · B [N, K] s8ᵀ, both row-major, through seven products of half size on wrapped int8 forms.
//
//   forms        A's or B's seven wrapped forms in ORDER, side by side along K (wide [R/2, 7 K/2] s8), each byte the form
//                mod 256, and per slot the number of carries (bytes whose exact form left [−128, 127]); a carry plane
//                [7, R/2, K/2] s8 (c in {−1, 0, 1}) when asked
//   batched      CUTLASS's cooperative TMA GEMM over L = 7: the products P [7, M/2, N/2] s32 (the unfused orbit)
//   assemble     O's four quadrants from P (the unfused orbit's last pass)
//   fused        the products one after another through one CUTLASS TMA pipeline, as one GEMM of shape (M/2, N/2, 7 K/2)
//                over the wide forms; the consumers hold three 128 x 128 accumulators, switch them every K/2 and store
//                each quadrant of O straight from registers once its last product is in
//   plain        CUTLASS's cooperative TMA GEMM on A and B, the baseline
//
// The products are on wrapped forms: O is A Bᵀ exactly when no form carried (|a|, |b| ≤ 63 everywhere, say); otherwise
// each carry's 256 c column is still to add (orbit.py's `products`). ext_gemm.h's entry refuses a call whose forms carried.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/pipeline/pipeline.hpp"

namespace orbit_impl {
using namespace cute;

constexpr int kSlots = 7;

// ---------------------------------------------------------------------------------------------------------------------
// Forms. Slot s is Strassen's product ORDER[s] = (2, 4, 1, 6, 7, 3, 5)[s]; a form is X(i0, j0) + sign · X(i1, j1), or one
// block (sign 0), with X(i, j) the block of the i-th half of the rows and the j-th half of K.
struct Form {
  int i0, j0, i1, j1, sign;
};
__host__ __device__ constexpr Form form_of(int which, int s) {
  constexpr Form a[kSlots] = {{1, 0, 1, 1, 1}, {1, 1, 0, 0, 0}, {0, 0, 1, 1, 1}, {1, 0, 0, 0, -1},
                              {0, 1, 1, 1, -1}, {0, 0, 0, 0, 0}, {0, 0, 0, 1, 1}};
  constexpr Form b[kSlots] = {{0, 0, 0, 0, 0}, {0, 1, 0, 0, -1}, {0, 0, 1, 1, 1}, {0, 0, 1, 0, 1},
                              {0, 1, 1, 1, 1}, {1, 0, 1, 1, -1}, {1, 1, 0, 0, 0}};
  return which ? b[s] : a[s];
}

// One thread a 16-byte column chunk of one row of the half grid: the four blocks' chunks in, the seven forms' out.
// x [R, K] s8 row-major; wide [R/2, 7 Kh] s8; counts[7] u64 (added to); carry [7, R/2, Kh] s8 or null.
template <int Which>
__global__ void forms_kernel(const int8_t* __restrict__ x, int8_t* __restrict__ wide, unsigned long long* counts,
                             int8_t* __restrict__ carry, int rh, int kh) {
  int chunks = kh / 16;
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  bool live = idx < (long long)rh * chunks;
  int r = live ? int(idx / chunks) : 0, c = live ? int(idx % chunks) * 16 : 0;
  int64_t k = 2LL * kh;
  uint4 blk[2][2] = {};
  if (live) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        blk[i][j] = *reinterpret_cast<const uint4*>(x + (int64_t)(i * rh + r) * k + j * kh + c);
  }
  unsigned cnt[kSlots];
#pragma unroll
  for (int s = 0; s < kSlots; ++s) {
    const Form f = form_of(Which, s);
    uint4 a = blk[f.i0][f.j0], b = blk[f.i1][f.j1], w = a, ov = make_uint4(0, 0, 0, 0);
    if (f.sign) {
      const unsigned* pa = &a.x;
      const unsigned* pb = &b.x;
      unsigned* pw = &w.x;
      unsigned* po = &ov.x;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        unsigned wr = f.sign > 0 ? __vadd4(pa[q], pb[q]) : __vsub4(pa[q], pb[q]);
        unsigned sat = f.sign > 0 ? __vaddss4(pa[q], pb[q]) : __vsubss4(pa[q], pb[q]);
        pw[q] = wr;
        po[q] = __vcmpne4(wr, sat);
      }
    }
    unsigned n = __popc(ov.x) + __popc(ov.y) + __popc(ov.z) + __popc(ov.w);
    cnt[s] = live ? n / 8 : 0;
    if (live) {
      *reinterpret_cast<uint4*>(wide + (int64_t)r * kSlots * kh + (int64_t)s * kh + c) = w;
      if (carry) {
        // an overflowed byte's exact value has a's sign: c = −1 where a < 0, else +1
        const unsigned* pa = &a.x;
        const unsigned* po = &ov.x;
        uint4 cv;
        unsigned* pc = &cv.x;
#pragma unroll
        for (int q = 0; q < 4; ++q) pc[q] = po[q] & (__vcmplts4(pa[q], 0u) | 0x01010101u);
        *reinterpret_cast<uint4*>(carry + ((int64_t)s * rh + r) * kh + c) = cv;
      }
    }
  }
#pragma unroll
  for (int s = 0; s < kSlots; ++s) {
    unsigned v = cnt[s];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0 && v) atomicAdd(counts + s, (unsigned long long)v);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The unfused orbit's assembly: O's quadrants from P [7, Mh, Nh] s32 (slots in ORDER: M2 M4 M1 M6 M7 M3 M5), int32 mod 2^32.
__device__ __forceinline__ int add32(int a, int b) { return int(unsigned(a) + unsigned(b)); }
__device__ __forceinline__ int sub32(int a, int b) { return int(unsigned(a) - unsigned(b)); }

__global__ void assemble_kernel(const int* __restrict__ p, int* __restrict__ o, int mh, int nh) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int q4 = nh / 4;
  if (idx >= (long long)mh * q4) return;
  int r = int(idx / q4), c = int(idx % q4) * 4;
  int64_t plane = (int64_t)mh * nh, at = (int64_t)r * nh + c;
  int4 m[kSlots];
#pragma unroll
  for (int s = 0; s < kSlots; ++s) m[s] = *reinterpret_cast<const int4*>(p + s * plane + at);
  const int *m2 = &m[0].x, *m4 = &m[1].x, *m1 = &m[2].x, *m6 = &m[3].x, *m7 = &m[4].x, *m3 = &m[5].x, *m5 = &m[6].x;
  int4 q00, q01, q10, q11;
  int *o00 = &q00.x, *o01 = &q01.x, *o10 = &q10.x, *o11 = &q11.x;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    o00[e] = add32(sub32(add32(m1[e], m4[e]), m5[e]), m7[e]);
    o01[e] = add32(m3[e], m5[e]);
    o10[e] = add32(m2[e], m4[e]);
    o11[e] = add32(add32(sub32(m1[e], m2[e]), m3[e]), m6[e]);
  }
  int64_t n = 2LL * nh;
  *reinterpret_cast<int4*>(o + (int64_t)r * n + c) = q00;
  *reinterpret_cast<int4*>(o + (int64_t)r * n + nh + c) = q01;
  *reinterpret_cast<int4*>(o + (int64_t)(mh + r) * n + c) = q10;
  *reinterpret_cast<int4*>(o + (int64_t)(mh + r) * n + nh + c) = q11;
}

// ---------------------------------------------------------------------------------------------------------------------
// SM clock: clock64 against the global timer over a short spin, from one thread.
__global__ void clock_kernel(unsigned long long* out, long long spin) {
  unsigned long long g0, g1;
  long long c0 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  while (clock64() - c0 < spin) {
  }
  long long c1 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  out[0] = (unsigned long long)(c1 - c0);
  out[1] = g1 - g0;
}

// ---------------------------------------------------------------------------------------------------------------------
// CUTLASS GEMMs: s8 A row-major (K-major), s8 B column-major (N x K, K-major), s32 D row-major, no source.
using RowMajor = cutlass::layout::RowMajor;
using ColMajor = cutlass::layout::ColumnMajor;

template <class TileShape, class ClusterShape>
struct CutlassGemm {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, int32_t, int32_t, void, RowMajor, 4, int32_t, RowMajor, 4,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, int8_t, RowMajor, 16, int8_t, ColMajor, 16, int32_t, TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideD = typename Kernel::StrideD;

  Gemm gemm;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;

  // (m, n, k, l) with A at a (lda, batch stride sa), B at b (ldb, sb), D at d (ldd, sd)
  int run(int m, int n, int k, int l, const int8_t* a, int64_t lda, int64_t sa, const int8_t* b, int64_t ldb,
          int64_t sb, int32_t* d, int64_t ldd, int64_t sd, cudaStream_t st) {
    StrideA dA = make_stride(lda, Int<1>{}, sa);
    StrideB dB = make_stride(ldb, Int<1>{}, sb);
    StrideD dD = make_stride(ldd, Int<1>{}, sd);
    typename Gemm::Arguments args{
        l > 1 ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, l},
        {a, dA, b, dB},
        {{1, 0}, nullptr, dD, d, dD}};
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 101;
    size_t ws = Gemm::get_workspace_size(args);
    if (ws > workspace_bytes) {
      if (workspace) cudaFree(workspace);
      if (cudaMalloc(&workspace, ws) != cudaSuccess) return 102;
      workspace_bytes = ws;
    }
    if (gemm.initialize(args, workspace, st) != cutlass::Status::kSuccess) return 103;
    return gemm.run(st) == cutlass::Status::kSuccess ? 0 : 104;
  }
};

using Plain256 = CutlassGemm<Shape<_128, _256, _128>, Shape<_1, _1, _1>>;
using Plain256c2 = CutlassGemm<Shape<_128, _256, _128>, Shape<_2, _1, _1>>;
using Plain128 = CutlassGemm<Shape<_128, _128, _128>, Shape<_1, _1, _1>>;

// ---------------------------------------------------------------------------------------------------------------------
// The fused orbit: CUTLASS's sm90 TMA warp-specialized mainloop (its producer `load` and consumer `mma`), driven over the
// seven products of each 128 x 128 tile of the half grid by a persistent kernel of its own.
using FusedTile = Shape<_128, _128, _128>;
using FusedMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, int8_t, RowMajor, 16, int8_t, ColMajor, 16, int32_t, FusedTile,
    Shape<_1, _1, _1>, cutlass::gemm::collective::StageCountAutoCarveout<1024>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using FusedPipeline = typename FusedMainloop::MainloopPipeline;
using FusedTiledMma = typename FusedMainloop::TiledMma;

struct FusedParams {
  typename FusedMainloop::Params mainloop;
  int32_t* d;
  int64_t ldd;
  int mh, nh, kt;            // half grid rows and columns, k-tiles of 128 a product
  int tiles_m, tiles_n, group;
};

struct FusedSmem {
  typename FusedMainloop::TensorStorage tensors;
  alignas(16) typename FusedMainloop::PipelineStorage pipeline;
};

constexpr int kFusedThreads = 384;

__device__ __forceinline__ void tile_coord(int t, FusedParams const& p, int& tm, int& tn) {
  // groups of `group` tile columns, the rows fastest within a group, so a round of tiles shares B's columns
  int per = p.group * p.tiles_m;
  int g = t / per, r = t % per;
  int gn = min(p.group, p.tiles_n - g * p.group);
  tn = g * p.group + r % gn;
  tm = r / gn;
}

__global__ void __launch_bounds__(kFusedThreads, 1) orbit_fused_kernel(CUTLASS_GRID_CONSTANT FusedParams const p) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(128) char smem_raw[];
  FusedSmem& smem = *reinterpret_cast<FusedSmem*>(smem_raw);
  int thread_idx = threadIdx.x;
  int warp_group = thread_idx / 128;
  int warp_in_group = (thread_idx / 32) % 4;
  int lane = thread_idx % 32;
  if (thread_idx == 0) FusedMainloop::prefetch_tma_descriptors(p.mainloop);

  typename FusedPipeline::Params pp;
  if (warp_group == 0 && warp_in_group == 0) pp.role = FusedPipeline::ThreadCategory::Producer;
  if (warp_group >= 1) pp.role = FusedPipeline::ThreadCategory::Consumer;
  pp.is_leader = (thread_idx % 128) == 0;
  pp.num_consumers = 256;
  pp.num_producers = 1;
  pp.transaction_bytes = p.mainloop.tma_transaction_bytes;
  FusedPipeline pipeline(smem.pipeline, pp, Shape<_1, _1, _1>{});
  __syncthreads();

  FusedMainloop collective;
  int tiles = p.tiles_m * p.tiles_n;
  auto problem = make_shape(p.mh, p.nh, kSlots * p.kt * 128, 1);
  auto load_inputs = collective.load_init(problem, p.mainloop);
  int ktiles = kSlots * p.kt;

  if (warp_group == 0) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    if (warp_in_group == 0) {
      auto state = cutlass::make_producer_start_state<FusedPipeline>();
      auto gA = get<0>(load_inputs);
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        int tm, tn;
        tile_coord(t, p, tm, tn);
        auto blk = make_coord(tm, tn, _, 0);
        auto k_iter = cute::make_coord_iterator(idx2crd(0, shape<3>(gA)), shape<3>(gA));
        collective.load(p.mainloop, pipeline, state, load_inputs, blk, k_iter, ktiles, lane, 0u, smem.tensors);
        state.advance(ktiles);
      }
      collective.load_tail(pipeline, state);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    int mma_thread_idx = thread_idx - 128;
    FusedTiledMma tiled_mma;
    typename FusedMainloop::PipelineState cstate;
    auto r0 = partition_fragment_C(tiled_mma, take<0, 2>(FusedTile{}));
    auto r1 = partition_fragment_C(tiled_mma, take<0, 2>(FusedTile{}));
    auto r2 = partition_fragment_C(tiled_mma, take<0, 2>(FusedTile{}));
    Tensor cD = make_identity_tensor(make_shape(Int<128>{}, Int<128>{}));
    Tensor tCcD = tiled_mma.get_thread_slice(mma_thread_idx).partition_C(cD);
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      int tm, tn;
      tile_coord(t, p, tm, tn);
      auto product = [&](auto& acc) {
        collective.mma(pipeline, cstate, acc, p.kt, mma_thread_idx, smem.tensors, p.mainloop);
        collective.mma_tail(pipeline, cstate, p.kt);
        cstate.advance(p.kt);
      };
      // O's quadrant (qi, qj) of this tile = x + sign · y
      auto store = [&](auto const& x, auto const& y, int sign, int qi, int qj) {
        int row0 = tm * 128, col0 = tn * 128;
        int32_t* base = p.d + (int64_t)(qi * p.mh + row0) * p.ldd + qj * p.nh + col0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(x); i += 2) {
          auto c = tCcD(i);
          int m = get<0>(c), n = get<1>(c);
          int v0 = sign > 0 ? add32(x(i), y(i)) : sub32(x(i), y(i));
          int v1 = sign > 0 ? add32(x(i + 1), y(i + 1)) : sub32(x(i + 1), y(i + 1));
          if (row0 + m < p.mh && col0 + n < p.nh)
            *reinterpret_cast<int2*>(base + (int64_t)m * p.ldd + n) = make_int2(v0, v1);
        }
      };
      product(r0);  // M2
      product(r1);  // M4
      store(r0, r1, 1, 1, 0);  // O10 = M2 + M4
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(r0); ++i) r0(i) = sub32(0, r0(i));
      product(r2);  // M1
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(r0); ++i) {
        r0(i) = add32(r0(i), r2(i));
        r1(i) = add32(r1(i), r2(i));
      }
      product(r2);  // M6
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(r0); ++i) r0(i) = add32(r0(i), r2(i));
      product(r2);  // M7
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(r1); ++i) r1(i) = add32(r1(i), r2(i));
      product(r2);  // M3
      store(r0, r2, 1, 1, 1);  // O11 = M1 − M2 + M6 + M3
      product(r0);  // M5
      store(r2, r0, 1, 0, 1);  // O01 = M3 + M5
      store(r1, r0, -1, 0, 0);  // O00 = M4 + M1 + M7 − M5
    }
  }
#endif
}

int fused_smem_bytes() { return int(sizeof(FusedSmem)); }

// ---------------------------------------------------------------------------------------------------------------------
int sm_count() {
  static int n = 0;
  if (!n) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}

int launch_forms(const int8_t* x, int8_t* wide, unsigned long long* counts, int8_t* carry, int r, int k, int which,
                 cudaStream_t st) {
  if (r % 2 || k % 32) return 201;
  int rh = r / 2, kh = k / 2;
  long long threads = (long long)rh * (kh / 16);
  unsigned blocks = unsigned((threads + 255) / 256);
  if (which)
    forms_kernel<1><<<blocks, 256, 0, st>>>(x, wide, counts, carry, rh, kh);
  else
    forms_kernel<0><<<blocks, 256, 0, st>>>(x, wide, counts, carry, rh, kh);
  return cudaGetLastError() == cudaSuccess ? 0 : 202;
}

int launch_fused(const int8_t* a_wide, const int8_t* b_wide, int32_t* d, int m, int n, int k, int group, int ctas,
                 cudaStream_t st) {
  if (m % 2 || n % 2 || k % 256) return 301;
  int mh = m / 2, nh = n / 2, kh = k / 2;
  using StrideA = typename FusedMainloop::StrideA;
  using StrideB = typename FusedMainloop::StrideB;
  int64_t ld = (int64_t)kSlots * kh;
  typename FusedMainloop::Arguments margs{a_wide, StrideA(make_stride(ld, Int<1>{}, ld * mh)), b_wide,
                                          StrideB(make_stride(ld, Int<1>{}, ld * nh))};
  auto problem = make_shape(mh, nh, kSlots * kh, 1);
  if (!FusedMainloop::can_implement(problem, margs)) return 302;
  FusedParams p;
  p.mainloop = FusedMainloop::to_underlying_arguments(problem, margs, nullptr);
  p.d = d;
  p.ldd = n;
  p.mh = mh;
  p.nh = nh;
  p.kt = kh / 128;
  p.tiles_m = (mh + 127) / 128;
  p.tiles_n = (nh + 127) / 128;
  p.group = group > 0 ? group : 8;
  int tiles = p.tiles_m * p.tiles_n;
  int grid = ctas > 0 ? ctas : sm_count();
  if (grid > tiles) grid = tiles;
  static bool attr = false;
  int smem = fused_smem_bytes();
  if (!attr) {
    if (cudaFuncSetAttribute(orbit_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
      return 303;
    attr = true;
  }
  orbit_fused_kernel<<<grid, kFusedThreads, smem, st>>>(p);
  return cudaGetLastError() == cudaSuccess ? 0 : 304;
}

Plain256& plain256() {
  static Plain256 g;
  return g;
}
Plain256c2& plain256c2() {
  static Plain256c2 g;
  return g;
}
Plain128& plain128() {
  static Plain128 g;
  return g;
}
Plain256& batched256() {
  static Plain256 g;
  return g;
}

}  // namespace orbit_impl

using namespace orbit_impl;

// ---------------------------------------------------------------------------------------------------------------------
// The library's C surface (bench.py's, by ctypes). Every call is issued on `st` and returns 0, or nonzero on refusal.
extern "C" {

int orbit_info(int* out) {
  out[0] = fused_smem_bytes();
  out[1] = FusedMainloop::DispatchPolicy::Stages;
  out[2] = kFusedThreads;
  out[3] = sm_count();
  return 0;
}

// x [r, k] s8 -> wide [r/2, 7 k/2] s8, counts[7] u64 added to, carry [7, r/2, k/2] s8 when not null; which 0: A, 1: B
int orbit_forms(const int8_t* x, int8_t* wide, unsigned long long* counts, int8_t* carry, int r, int k, int which,
                cudaStream_t st) {
  return launch_forms(x, wide, counts, carry, r, k, which, st);
}

// P [7, m/2, n/2] s32 = the seven products of the wide forms, CUTLASS batched (L = 7)
int orbit_batched(const int8_t* a_wide, const int8_t* b_wide, int32_t* prod, int m, int n, int k, cudaStream_t st) {
  int mh = m / 2, nh = n / 2, kh = k / 2;
  int64_t ld = (int64_t)kSlots * kh;
  return batched256().run(mh, nh, kh, kSlots, a_wide, ld, kh, b_wide, ld, kh, prod, nh, (int64_t)mh * nh, st);
}

// O [m, n] s32 from P
int orbit_assemble(const int32_t* prod, int32_t* o, int m, int n, cudaStream_t st) {
  int mh = m / 2, nh = n / 2;
  if (nh % 4) return 401;
  long long threads = (long long)mh * (nh / 4);
  assemble_kernel<<<unsigned((threads + 255) / 256), 256, 0, st>>>(prod, o, mh, nh);
  return cudaGetLastError() == cudaSuccess ? 0 : 402;
}

// O [m, n] s32 from the wide forms, fused; group: tile columns a raster group (0: 8); ctas: persistent CTAs (0: every SM)
int orbit_fused(const int8_t* a_wide, const int8_t* b_wide, int32_t* o, int m, int n, int k, int group, int ctas,
                cudaStream_t st) {
  return launch_fused(a_wide, b_wide, o, m, n, k, group, ctas, st);
}

// D [m, n] s32 = A Bᵀ by CUTLASS: variant 0 128x256, 1 128x256 in clusters of 2 along M, 2 128x128
int orbit_plain(const int8_t* a, const int8_t* b, int32_t* d, int m, int n, int k, int variant, cudaStream_t st) {
  switch (variant) {
    case 0: return plain256().run(m, n, k, 1, a, k, (int64_t)m * k, b, k, (int64_t)n * k, d, n, (int64_t)m * n, st);
    case 1: return plain256c2().run(m, n, k, 1, a, k, (int64_t)m * k, b, k, (int64_t)n * k, d, n, (int64_t)m * n, st);
    case 2: return plain128().run(m, n, k, 1, a, k, (int64_t)m * k, b, k, (int64_t)n * k, d, n, (int64_t)m * n, st);
  }
  return 501;
}

// out[0] SM clocks and out[1] global-timer nanoseconds over a spin of `spin` clocks on one SM
int orbit_clock(unsigned long long* out, long long spin, cudaStream_t st) {
  clock_kernel<<<1, 1, 0, st>>>(out, spin);
  return cudaGetLastError() == cudaSuccess ? 0 : 601;
}

// ---------------------------------------------------------------------------------------------------------------------
// ext_gemm.h's entry (benchmarks/pouw/accel_v31): D = A Bᵀ through the fused orbit. B's forms are made at each load; a
// call whose forms carried is refused (the carry corrections aren't in the kernel yet), so every D it returns is exact.
struct ExtCtx {
  int m, n, k;
  int8_t* a_wide;
  int8_t* b_wide;
  unsigned long long* counts;  // [2][7]: A's, B's
};

void* v31_ext_init(int m, int n, int k) {
  if (m < 2 || m % 2 || n % 256 || k % 256 || k >= (1 << 17)) return nullptr;
  ExtCtx* h = new ExtCtx{m, n, k, nullptr, nullptr, nullptr};
  size_t aw = (size_t)(m / 2) * kSlots * (k / 2), bw = (size_t)(n / 2) * kSlots * (k / 2);
  if (cudaMalloc(&h->a_wide, aw) != cudaSuccess || cudaMalloc(&h->b_wide, bw) != cudaSuccess ||
      cudaMalloc(&h->counts, 2 * kSlots * sizeof(unsigned long long)) != cudaSuccess) {
    cudaFree(h->a_wide);
    cudaFree(h->b_wide);
    delete h;
    return nullptr;
  }
  return h;
}

int v31_ext_load(void* hv, const int8_t* b, cudaStream_t st) {
  ExtCtx* h = static_cast<ExtCtx*>(hv);
  if (cudaMemsetAsync(h->counts + kSlots, 0, kSlots * sizeof(unsigned long long), st) != cudaSuccess) return 701;
  return launch_forms(b, h->b_wide, h->counts + kSlots, nullptr, h->n, h->k, 1, st);
}

int v31_ext_gemm(void* hv, const int8_t* a, int32_t* d, cudaStream_t st) {
  ExtCtx* h = static_cast<ExtCtx*>(hv);
  if (cudaMemsetAsync(h->counts, 0, kSlots * sizeof(unsigned long long), st) != cudaSuccess) return 711;
  int rc = launch_forms(a, h->a_wide, h->counts, nullptr, h->m, h->k, 0, st);
  if (rc) return rc;
  rc = launch_fused(h->a_wide, h->b_wide, d, h->m, h->n, h->k, 8, 0, st);
  if (rc) return rc;
  unsigned long long c[2 * kSlots];
  if (cudaMemcpyAsync(c, h->counts, sizeof c, cudaMemcpyDeviceToHost, st) != cudaSuccess) return 712;
  if (cudaStreamSynchronize(st) != cudaSuccess) return 713;
  for (unsigned long long v : c)
    if (v) return 714;
  return 0;
}

void v31_ext_free(void* hv) {
  ExtCtx* h = static_cast<ExtCtx*>(hv);
  if (!h) return;
  cudaFree(h->a_wide);
  cudaFree(h->b_wide);
  cudaFree(h->counts);
  delete h;
}

}  // extern "C"
