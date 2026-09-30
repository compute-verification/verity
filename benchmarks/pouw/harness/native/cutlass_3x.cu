// CUTLASS 3.x sm_120 baselines: TMA warp-specialized kernels on sm_120's mma.sync (f8f6f4, mxf8f6f4, mxf4nvf4), in the
// tile shapes and schedules CUTLASS's own sm_120 tests and examples (79a/b, 87b) use, plus swapped variants for skinny m.
// Compiled once per family (-DPH_PART=0..4), since one translation unit of all of them outgrows a 16 GB builder.
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
#include "cutlass_api.h"

using namespace cute;

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#error "build for sm_120a with CUDA 12.8 or later"
#endif

namespace {

enum { SF_NONE = 0, SF_BLOCKSCALED = 1, SF_GROUPWISE = 2 };

using Groupwise = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;

template <class ElementA, class ElementB, class Tile, class Schedule, bool Swap, int SF, int AlignA>
struct Kernel {
  static constexpr int kSF = SF;
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
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
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
  Problem(int m, int n, int k, const void* A, const void* B, const void* sfa, const void* sfb, void* D, bool swap) {
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
  }
};

template <class K>
struct Prepared final : phc_prepared {
  typename K::Gemm gemm;
  int launch(void* stream) override {
    auto s = gemm.run(cudaStream_t(stream));
    return s == cutlass::Status::kSuccess ? 0 : 3000 + int(s);
  }
};

template <class K, bool Swap, int SF>
int query(int m, int n, int k, size_t* ws, size_t* sfa, size_t* sfb) {
  Problem<K> p(m, n, k, nullptr, nullptr, nullptr, nullptr, nullptr, Swap);
  auto s = K::Gemm::can_implement(p.args);
  if (s != cutlass::Status::kSuccess) return 3000 + int(s);
  *ws = K::Gemm::get_workspace_size(p.args);
  *sfa = p.sfa_bytes;
  *sfb = p.sfb_bytes;
  return 0;
}

template <class K, bool Swap, int SF>
int prepare(const phc_args* a, void* ws, size_t ws_bytes, void* stream, phc_prepared** out) {
  Problem<K> p(a->m, a->n, a->k, a->A, a->B, a->sfa, a->sfb, a->D, Swap);
  if (K::Gemm::get_workspace_size(p.args) > ws_bytes) return 3100;
  auto* r = new Prepared<K>();
  auto s = r->gemm.initialize(p.args, ws, cudaStream_t(stream));
  if (s != cutlass::Status::kSuccess) {
    delete r;
    return 3000 + int(s);
  }
  *out = r;
  return 0;
}

using E4M3 = cutlass::float_e4m3_t;
using NVF4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using MXF4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using MXF8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
namespace G = cutlass::gemm;

#define ENTRY(NAME, KIND, EA, TILE, SCHED, SWAP, SF)                                                               \
  {NAME, KIND, SWAP, &query<Kernel<EA, EA, TILE, SCHED, SWAP, SF, (KIND == 5 || KIND == 6) ? 32 : 16>, SWAP, SF>,  \
   &prepare<Kernel<EA, EA, TILE, SCHED, SWAP, SF, (KIND == 5 || KIND == 6) ? 32 : 16>, SWAP, SF>}

using T128_128_128 = Shape<_128, _128, _128>;
using T128_128_64 = Shape<_128, _128, _64>;
using T64_128_128 = Shape<_64, _128, _128>;
using T128_32_128 = Shape<_128, _32, _128>;
using T128_128_256 = Shape<_128, _128, _256>;
using T128_64_256 = Shape<_128, _64, _256>;
using T128_32_256 = Shape<_128, _32, _256>;

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
#else
#error "PH_PART is 0..4"
#endif
};

}  // namespace

#define PH_CAT2(a, b) a##b
#define PH_CAT(a, b) PH_CAT2(a, b)
const phc_entry* PH_CAT(phc_table_3x_, PH_PART)(int* count) {
  *count = int(sizeof TABLE / sizeof TABLE[0]);
  return TABLE;
}
