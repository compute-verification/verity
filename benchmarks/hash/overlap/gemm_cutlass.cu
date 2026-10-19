// One CUTLASS GEMM configuration for the bench (gemms.h), chosen by macros, so each configuration is its own translation
// unit with its own register budget (gpu.sh builds one object per GEMM_CFGS entry):
//   OV_NAME, OV_ID      its arm's name (a string, and the same as an identifier)
//   OV_ARCH             90 (sm_90a: TMA warp-specialized, wgmma) or 100 (sm_100a: tcgen05/UMMA)
//   OV_KIND             sm_90: 0 cooperative, 1 pingpong; sm_100: 0 1-SM, 1 2-SM
//   OV_TM OV_TN OV_TK   the (MMA) tile; OV_CM OV_CN the cluster; OV_STAGES 0 = the carveout's auto count
// With patch_cutlass.sh applied: OV_MAXNREG caps the kernel's launch registers (__maxnreg__), and on sm_90 OV_LOADR_V /
// OV_MMAR_V replace the producer's and consumers' setmaxnreg counts (they must fit: 128 L + 256 C <= 384 OV_MAXNREG).
#include <cstdio>
#include <map>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "gemms.h"

using namespace cute;

// a named namespace per configuration (OV_ID): nvcc's host stub can't name a kernel templated on anonymous-namespace types
namespace OV_ID {
using EA = cutlass::bfloat16_t;
using EC = cutlass::bfloat16_t;
using LA = cutlass::layout::RowMajor;      // X [m, k]
using LW = cutlass::layout::ColumnMajor;   // W [n, k] row-major: B as k x n, K-major
using LC = cutlass::layout::RowMajor;
using Acc = float;
constexpr int AL = 8;
using Tile = Shape<Int<OV_TM>, Int<OV_TN>, Int<OV_TK>>;
using Cluster = Shape<Int<OV_CM>, Int<OV_CN>, _1>;
#if OV_ARCH == 90
using Arch = cutlass::arch::Sm90;
#if OV_KIND == 0
using KS = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using ES = cutlass::epilogue::TmaWarpSpecializedCooperative;
#else
using KS = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ES = cutlass::epilogue::TmaWarpSpecialized;
#endif
#else
using Arch = cutlass::arch::Sm100;
#if OV_KIND == 0
using KS = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
using ES = cutlass::epilogue::TmaWarpSpecialized1Sm;
#else
using KS = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using ES = cutlass::epilogue::TmaWarpSpecialized2Sm;
#endif
#endif
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<Arch, cutlass::arch::OpClassTensorOp, Tile, Cluster,
                                                                      cutlass::epilogue::collective::EpilogueTileAuto, Acc, Acc,
                                                                      EC, LC, AL, EC, LC, AL, ES>::CollectiveOp;
#if OV_STAGES == 0
using Stages = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>;
#else
using Stages = cutlass::gemm::collective::StageCount<OV_STAGES>;
#endif
using Main = typename cutlass::gemm::collective::CollectiveBuilder<Arch, cutlass::arch::OpClassTensorOp, EA, LA, AL, EA, LW, AL, Acc,
                                                                   Tile, Cluster, Stages, KS>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

struct St {
  int m, n, k;
  std::map<std::pair<const void*, const void*>, std::pair<Gemm*, void*>> g;   // by (Y, R): initialized once, its own workspace
};

int run(void* s_, const __nv_bfloat16* X, const __nv_bfloat16* W, const __nv_bfloat16* R, __nv_bfloat16* Y, cudaStream_t st) {
  St& s = *static_cast<St*>(s_);
  const auto key = std::make_pair((const void*)Y, (const void*)R);
  auto it = s.g.find(key);
  if (it == s.g.end()) {
    using SA = typename Kernel::StrideA;
    using SB = typename Kernel::StrideB;
    using SC = typename Kernel::StrideC;
    using SD = typename Kernel::StrideD;
    const auto sa = cutlass::make_cute_packed_stride(SA{}, make_shape(s.m, s.k, 1));
    const auto sb = cutlass::make_cute_packed_stride(SB{}, make_shape(s.n, s.k, 1));
    const auto sc = cutlass::make_cute_packed_stride(SC{}, make_shape(s.m, s.n, 1));
    const auto sd = cutlass::make_cute_packed_stride(SD{}, make_shape(s.m, s.n, 1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {s.m, s.n, s.k, 1},
                                  {reinterpret_cast<const EA*>(X), sa, reinterpret_cast<const EA*>(W), sb},
                                  {{1.0f, R ? 1.0f : 0.0f}, reinterpret_cast<const EC*>(R ? R : Y), sc, reinterpret_cast<EC*>(Y), sd}};
    int dev = 0;
    cudaGetDevice(&dev);
    args.hw_info.device_id = dev;
    args.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    Gemm* g = new Gemm;
    if (g->can_implement(args) != cutlass::Status::kSuccess) return 1;
    void* ws = nullptr;
    const size_t w = Gemm::get_workspace_size(args);
    if (w && cudaMalloc(&ws, w) != cudaSuccess) return 2;
    if (g->initialize(args, ws, st) != cutlass::Status::kSuccess) return 3;
    it = s.g.emplace(key, std::make_pair(g, ws)).first;
  }
  return it->second.first->run(st) == cutlass::Status::kSuccess ? 0 : 4;
}

std::vector<ExtGemm> make(int m, int n, int k) {
  cudaFuncAttributes a{};
  cudaFuncGetAttributes(&a, cutlass::device_kernel<Kernel>);
  printf("{\"ext_gemm\": \"%s\", \"n\": %d, \"k\": %d, \"threads\": %d, \"regs\": %d, \"smem\": %d, \"local\": %zu}\n", OV_NAME, n, k,
         (int)Kernel::MaxThreadsPerBlock, a.numRegs, (int)Kernel::SharedStorageSize, a.localSizeBytes);
  if (Kernel::SharedStorageSize >= (48 << 10))
    cudaFuncSetAttribute(cutlass::device_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)Kernel::SharedStorageSize);
  return {ExtGemm{OV_NAME, new St{m, n, k, {}}, run, (int)Kernel::MaxThreadsPerBlock, a.numRegs, (int)Kernel::SharedStorageSize}};
}
const ExtReg reg(make);
}  // namespace OV_ID
