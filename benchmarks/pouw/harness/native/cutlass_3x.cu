// CUTLASS 3.x sm_120 baselines: TMA warp-specialized kernels on sm_120's mma.sync (f8f6f4, mxf8f6f4, mxf4nvf4), in the
// tile shapes and schedules CUTLASS's own sm_120 tests and examples (79a/b, 87b) use, plus swapped variants for skinny m, and
// the FP4 search's larger tiles and stream-K (parts 5 and 6). Compiled in parts (-DPH_PART=0..6), since one translation unit
// of all of them outgrows a 16 GB builder.
#include <cstdio>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/block_striped.h"
#include "cutlass_api.h"

// Stream-K's middle peers add their FP32 partials into the workspace with atomics, and an FP32 atomic add flushes subnormals
// (RED.ADD.F32.FTZ; PTX has no .noftz for f32), which the SASS gate refuses. Under the Deterministic reduction, which every
// stream-K call here sets, each peer adds only after the one before it has arrived, so a plain load, add and store gives the
// same sum without the flush; store, load and load_add stay CUTLASS's.
namespace cutlass {
template <int BlockThreads, typename ArrayT>
struct BlockStripedReduce<BlockThreads, ArrayT, float> : BlockStriped<BlockThreads, ArrayT, float> {
  CUTLASS_DEVICE
  static void reduce(ArrayT* ptr, const ArrayT& data, int thread_idx) {
    float* out = reinterpret_cast<float*>(ptr);
    const float* in = reinterpret_cast<const float*>(&data);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < BlockStripedReduce::kStripes; ++i) out[BlockThreads * i + thread_idx] += in[i];
  }
};
}  // namespace cutlass

using namespace cute;

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#error "build for sm_120a with CUDA 12.8 or later"
#endif

namespace {

enum { SF_NONE = 0, SF_BLOCKSCALED = 1, SF_GROUPWISE = 2 };

using Groupwise = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;

template <class ElementA, class ElementB, class Tile, class Schedule, bool Swap, int SF, int AlignA, class TileScheduler>
struct Kernel {
  static constexpr int kSF = SF;
  static constexpr bool kStreamK = cute::is_same_v<TileScheduler, cutlass::gemm::StreamKScheduler>;
  using OpClass = cute::conditional_t<SF == SF_BLOCKSCALED, cutlass::arch::OpClassBlockScaledTensorOp,
                                      cutlass::arch::OpClassTensorOp>;
  using ElementD = cutlass::bfloat16_t;
  using LayoutD = cute::conditional_t<Swap, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using Cluster = Shape<_1, _1, _1>;
  using LayoutATag = cute::conditional_t<SF == SF_GROUPWISE,
                                         cute::tuple<cutlass::layout::RowMajor, decltype(Groupwise::deduce_layoutSFA())>,
                                         cutlass::layout::RowMajor>;
  using LayoutBTag = cute::conditional_t<SF == SF_GROUPWISE,
                                         cute::tuple<cutlass::layout::ColumnMajor, decltype(Groupwise::deduce_layoutSFB())>,
                                         cutlass::layout::ColumnMajor>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      ElementD, LayoutD, AlignD, ElementD, LayoutD, AlignD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, OpClass, ElementA, LayoutATag, AlignA, ElementB, LayoutBTag, AlignA, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      Schedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class K>
struct Problem {
  using Gemm = typename K::Gemm;
  using GK = typename Gemm::GemmKernel;
  using ML = typename GK::CollectiveMainloop;
  typename Gemm::Arguments args;
  size_t sfa_bytes = 0, sfb_bytes = 0;

  // Our (m, n, k) and pointers; swapped kernels compute D^T = B . A^T.
  Problem(const phc_args& g, bool swap) {
    const int m = g.m, n = g.n, k = g.k;
    const void *A = g.A, *B = g.B, *sfa = g.sfa, *sfb = g.sfb;
    void* D = g.D;
    int M = swap ? n : m, N = swap ? m : n;
    const void *pa = swap ? B : A, *pb = swap ? A : B, *psa = swap ? sfb : sfa, *psb = swap ? sfa : sfb;
    auto sA = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {M, k, 1});
    auto sB = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {N, k, 1});
    auto sC = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {M, N, 1});
    auto sD = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {M, N, 1});
    using EA = typename ML::ElementA;
    using EB = typename ML::ElementB;
    auto a = reinterpret_cast<EA const*>(pa);
    auto b = reinterpret_cast<EB const*>(pb);
    auto d = reinterpret_cast<typename Gemm::ElementD*>(D);
    if constexpr (K::kSF == SF_BLOCKSCALED) {
      using Cfg = typename ML::Sm1xxBlkScaledConfig;
      using ESF = typename ML::ElementSF;
      auto la = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, k, 1));
      auto lb = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, k, 1));
      (swap ? sfb_bytes : sfa_bytes) = size(filter_zeros(la)) * sizeof(ESF);
      (swap ? sfa_bytes : sfb_bytes) = size(filter_zeros(lb)) * sizeof(ESF);
      args = {cutlass::gemm::GemmUniversalMode::kGemm, {M, N, k, 1},
              {a, sA, b, sB, reinterpret_cast<ESF const*>(psa), la, reinterpret_cast<ESF const*>(psb), lb},
              {{1.0f, 0.0f}, d, sC, d, sD}};
    } else if constexpr (K::kSF == SF_GROUPWISE) {
      auto la = Groupwise::tile_atom_to_shape_SFA(make_shape(M, N, k, 1));
      auto lb = Groupwise::tile_atom_to_shape_SFB(make_shape(M, N, k, 1));
      (swap ? sfb_bytes : sfa_bytes) = size(filter_zeros(la)) * sizeof(float);
      (swap ? sfa_bytes : sfb_bytes) = size(filter_zeros(lb)) * sizeof(float);
      args = {cutlass::gemm::GemmUniversalMode::kGemm, {M, N, k, 1},
              {a, sA, b, sB, reinterpret_cast<float const*>(psa), la, reinterpret_cast<float const*>(psb), lb},
              {{}, d, sC, d, sD}};
      args.epilogue.thread.alpha = 1.0f;
      args.epilogue.thread.beta = 0.0f;
    } else {
      args = {cutlass::gemm::GemmUniversalMode::kGemm, {M, N, k, 1}, {a, sA, b, sB}, {{1.0f, 0.0f}, d, sC, d, sD}};
    }
    if constexpr (K::kStreamK) {
      args.scheduler.reduction_mode = decltype(args.scheduler.reduction_mode)::Deterministic;
    }
    using RO = decltype(args.scheduler.raster_order);
    if (g.swizzle) args.scheduler.max_swizzle_size = g.swizzle;
    if (g.raster) args.scheduler.raster_order = (g.raster == 1) != swap ? RO::AlongM : RO::AlongN;
  }
};

// Stream-K's fixup counters only ever grow, so each call starts from a zeroed workspace (CUTLASS's initialize does it once):
// the memset is part of the plain call, and of its time.
template <class K>
struct Prepared final : phc_prepared {
  typename K::Gemm gemm;
  void* ws = nullptr;
  size_t ws_bytes = 0;
  int launch(void* stream) override {
    if (ws_bytes && cudaMemsetAsync(ws, 0, ws_bytes, cudaStream_t(stream)) != cudaSuccess) return 3200;
    auto s = gemm.run(cudaStream_t(stream));
    return s == cutlass::Status::kSuccess ? 0 : 3000 + int(s);
  }
};

template <class K, bool Swap, int SF>
int query(const phc_args* a, size_t* ws, size_t* sfa, size_t* sfb) {
  Problem<K> p(*a, Swap);
  auto s = K::Gemm::can_implement(p.args);
  if (s != cutlass::Status::kSuccess) return 3000 + int(s);
  *ws = K::Gemm::get_workspace_size(p.args);
  *sfa = p.sfa_bytes;
  *sfb = p.sfb_bytes;
  return 0;
}

template <class K, bool Swap, int SF>
int prepare(const phc_args* a, void* ws, size_t ws_bytes, void* stream, phc_prepared** out) {
  Problem<K> p(*a, Swap);
  size_t need = K::Gemm::get_workspace_size(p.args);
  if (need > ws_bytes) return 3100;
  auto* r = new Prepared<K>();
  auto s = r->gemm.initialize(p.args, ws, cudaStream_t(stream));
  if (s != cutlass::Status::kSuccess) {
    delete r;
    return 3000 + int(s);
  }
  if (K::kStreamK) {
    r->ws = ws;
    r->ws_bytes = need;
  }
  *out = r;
  return 0;
}

using E4M3 = cutlass::float_e4m3_t;
using NVF4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using MXF4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using MXF8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
namespace G = cutlass::gemm;

#define KERNEL(KIND, EA, TILE, SCHED, SWAP, SF, TS) Kernel<EA, EA, TILE, SCHED, SWAP, SF, (KIND == 5 || KIND == 6) ? 32 : 16, TS>
// TS: the tile scheduler (void: CUTLASS's default, the persistent CLC scheduler; G::StreamKScheduler)
#define ENTRY_TS(NAME, KIND, EA, TILE, SCHED, SWAP, SF, TS)                                                          \
  {NAME, KIND, SWAP, &query<KERNEL(KIND, EA, TILE, SCHED, SWAP, SF, TS), SWAP, SF>,                                 \
   &prepare<KERNEL(KIND, EA, TILE, SCHED, SWAP, SF, TS), SWAP, SF>}
#define ENTRY(NAME, KIND, EA, TILE, SCHED, SWAP, SF) ENTRY_TS(NAME, KIND, EA, TILE, SCHED, SWAP, SF, void)

using T128_128_128 = Shape<_128, _128, _128>;
using T128_128_64 = Shape<_128, _128, _64>;
using T64_128_128 = Shape<_64, _128, _128>;
using T128_32_128 = Shape<_128, _32, _128>;
using T128_128_256 = Shape<_128, _128, _256>;
using T128_64_256 = Shape<_128, _64, _256>;
using T128_32_256 = Shape<_128, _32, _256>;
using T128_256_128 = Shape<_128, _256, _128>;
using T256_128_128 = Shape<_256, _128, _128>;
using T128_256_64 = Shape<_128, _256, _64>;
using T256_128_64 = Shape<_256, _128, _64>;

const phc_entry TABLE[] = {
#if PH_PART == 0
    // FP8 E4M3, per-tensor scales (alpha), f8f6f4
    ENTRY("cutlass3x_fp8_128x128x128_coop", 1, E4M3, T128_128_128, G::KernelTmaWarpSpecializedCooperative, false, SF_NONE),
    ENTRY("cutlass3x_fp8_128x128x64_coop", 1, E4M3, T128_128_64, G::KernelTmaWarpSpecializedCooperative, false, SF_NONE),
    ENTRY("cutlass3x_fp8_64x128x128_pingpong", 1, E4M3, T64_128_128, G::KernelTmaWarpSpecializedPingpong, false, SF_NONE),
    ENTRY("cutlass3x_fp8_128x32x128_coop_swap", 1, E4M3, T128_32_128, G::KernelTmaWarpSpecializedCooperative, true, SF_NONE),
#elif PH_PART == 1
    // MXFP8 (E4M3, UE8M0 per 32)
    ENTRY("cutlass3x_mxfp8_128x128x128_coop", 3, MXF8, T128_128_128, G::KernelTmaWarpSpecializedMxf8f6f4Sm120, false, SF_BLOCKSCALED),
    ENTRY("cutlass3x_mxfp8_128x128x128_pingpong", 3, MXF8, T128_128_128, G::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120, false, SF_BLOCKSCALED),
    // FP8 groupwise: 1x128 on A, 128x128 on B (DeepSeek-style)
    ENTRY("cutlass3x_fp8_groupwise_128x128x128_coop", 4, E4M3, T128_128_128, G::KernelTmaWarpSpecializedBlockwiseCooperativeSm120, false, SF_GROUPWISE),
    ENTRY("cutlass3x_fp8_groupwise_64x128x128_pingpong", 4, E4M3, T64_128_128, G::KernelTmaWarpSpecializedBlockwisePingpongSm120, false, SF_GROUPWISE),
#elif PH_PART == 2
    // NVFP4 (E2M1, UE4M3 per 16)
    ENTRY("cutlass3x_nvfp4_128x128x128_coop", 5, NVF4, T128_128_128, G::KernelTmaWarpSpecializedNvf4Sm120, false, SF_BLOCKSCALED),
    ENTRY("cutlass3x_nvfp4_128x128x128_pingpong", 5, NVF4, T128_128_128, G::KernelTmaWarpSpecializedPingpongNvf4Sm120, false, SF_BLOCKSCALED),
    ENTRY("cutlass3x_nvfp4_128x128x256_coop", 5, NVF4, T128_128_256, G::KernelTmaWarpSpecializedNvf4Sm120, false, SF_BLOCKSCALED),
#elif PH_PART == 3
    ENTRY("cutlass3x_nvfp4_128x64x256_coop_swap", 5, NVF4, T128_64_256, G::KernelTmaWarpSpecializedNvf4Sm120, true, SF_BLOCKSCALED),
    ENTRY("cutlass3x_nvfp4_128x32x256_coop_swap", 5, NVF4, T128_32_256, G::KernelTmaWarpSpecializedNvf4Sm120, true, SF_BLOCKSCALED),
#elif PH_PART == 4
    // MXFP4 (E2M1, UE8M0 per 32)
    ENTRY("cutlass3x_mxfp4_128x128x128_coop", 6, MXF4, T128_128_128, G::KernelTmaWarpSpecializedMxf4Sm120, false, SF_BLOCKSCALED),
    ENTRY("cutlass3x_mxfp4_128x128x128_pingpong", 6, MXF4, T128_128_128, G::KernelTmaWarpSpecializedPingpongMxf4Sm120, false, SF_BLOCKSCALED),
    ENTRY("cutlass3x_mxfp4_128x64x256_coop_swap", 6, MXF4, T128_64_256, G::KernelTmaWarpSpecializedMxf4Sm120, true, SF_BLOCKSCALED),
#elif PH_PART == 5
    // NVFP4, the FP4 search: CTA tiles past 128x128 (a quarter fewer L2 bytes per FLOP), pingpong at K 256, and stream-K for
    // skinny m (64 swapped tiles on 188 SMs at m 32, n 8192)
    ENTRY("cutlass3x_nvfp4_128x256x128_coop", 5, NVF4, T128_256_128, G::KernelTmaWarpSpecializedNvf4Sm120, false, SF_BLOCKSCALED),
    ENTRY("cutlass3x_nvfp4_256x128x128_coop", 5, NVF4, T256_128_128, G::KernelTmaWarpSpecializedNvf4Sm120, false, SF_BLOCKSCALED),
    ENTRY("cutlass3x_nvfp4_128x128x256_pingpong", 5, NVF4, T128_128_256, G::KernelTmaWarpSpecializedPingpongNvf4Sm120, false, SF_BLOCKSCALED),
    ENTRY_TS("cutlass3x_nvfp4_128x128x128_coop_streamk", 5, NVF4, T128_128_128, G::KernelTmaWarpSpecializedNvf4Sm120, false, SF_BLOCKSCALED, G::StreamKScheduler),
    ENTRY_TS("cutlass3x_nvfp4_128x32x256_coop_swap_streamk", 5, NVF4, T128_32_256, G::KernelTmaWarpSpecializedNvf4Sm120, true, SF_BLOCKSCALED, G::StreamKScheduler),
    ENTRY_TS("cutlass3x_nvfp4_128x64x256_coop_swap_streamk", 5, NVF4, T128_64_256, G::KernelTmaWarpSpecializedNvf4Sm120, true, SF_BLOCKSCALED, G::StreamKScheduler),
#elif PH_PART == 6
    // MXFP4 as part 5, and FP8 E4M3 (per-tensor) in the same larger CTA tiles, so that FP8's peak fraction is searched alike
    ENTRY("cutlass3x_mxfp4_128x256x128_coop", 6, MXF4, T128_256_128, G::KernelTmaWarpSpecializedMxf4Sm120, false, SF_BLOCKSCALED),
    ENTRY("cutlass3x_mxfp4_256x128x128_coop", 6, MXF4, T256_128_128, G::KernelTmaWarpSpecializedMxf4Sm120, false, SF_BLOCKSCALED),
    ENTRY_TS("cutlass3x_mxfp4_128x64x256_coop_swap_streamk", 6, MXF4, T128_64_256, G::KernelTmaWarpSpecializedMxf4Sm120, true, SF_BLOCKSCALED, G::StreamKScheduler),
    ENTRY("cutlass3x_fp8_128x256x64_coop", 1, E4M3, T128_256_64, G::KernelTmaWarpSpecializedCooperative, false, SF_NONE),
    ENTRY("cutlass3x_fp8_256x128x64_coop", 1, E4M3, T256_128_64, G::KernelTmaWarpSpecializedCooperative, false, SF_NONE),
    ENTRY_TS("cutlass3x_fp8_128x32x128_coop_swap_streamk", 1, E4M3, T128_32_128, G::KernelTmaWarpSpecializedCooperative, true, SF_NONE, G::StreamKScheduler),
#else
#error "PH_PART is 0..6"
#endif
};

}  // namespace

#define PH_CAT2(a, b) a##b
#define PH_CAT(a, b) PH_CAT2(a, b)
const phc_entry* PH_CAT(phc_table_3x_, PH_PART)(int* count) {
  *count = int(sizeof TABLE / sizeof TABLE[0]);
  return TABLE;
}
