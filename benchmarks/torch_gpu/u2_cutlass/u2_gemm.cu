// One configuration of the exact s8 GEMM D = A·Bᵀ (A M×K row-major, B N×K row-major, D M×N s32, alpha 1 in s32), CUTLASS
// 3.x's sm90 warp-specialized persistent kernel, launched from an entry that stamps every consumer warpgroup's clock64 and
// %globaltimer so a timed call carries its own SM clock. build.sh compiles this file once per configuration, each with its
// U2_* macros and its own U2_NAME prefix on the exported functions:
//   U2_TM, U2_TN, U2_TK    the CTA tile (U2_TK in bytes, 128 = one 128-byte swizzle atom)
//   U2_CM, U2_CN           the cluster; the operand shared across it is multicast by TMA
//   U2_STAGES              the mainloop's stages (0: CUTLASS's carveout after the epilogue's shared memory)
//   U2_PINGPONG            1: the ping-pong kernel (each consumer warpgroup a whole tile), else cooperative
//   U2_NOSMEM              1: the epilogue stores from registers (NoSmemWarpSpecialized), else through shared memory by TMA
//   U2_MAXNREG             0: CUTLASS's launch bounds (384 threads at 168 registers, the register file); N: launched at
//                          N registers a thread, with U2_LOAD_REGS and U2_MMA_REGS the producer's and consumers' setmaxnreg
//                          (build.sh's patch of the kernel headers reads them), so CTAs of another kernel fit beside it.
//                          ptxas allocates the whole kernel within N, and a consumer's wgmma m64n256 holds 128 accumulator
//                          registers, so N is at least 160
//   U2_HOOK                1 (cooperative only): the producer warpgroup's three warps that a static scheduler and a void C
//                          leave idle chain SHA-512 compressions while the consumers run, at U2_LOAD_REGS registers, until
//                          a consumer warpgroup of their CTA ends its tiles; off until bench.py switches it on (U2_FN(hook))
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef U2_LOAD_REGS
#define U2_LOAD_REGS (!HeavyRegisterPressure ? 40 : 24)
#endif
#ifndef U2_MMA_REGS
#define U2_MMA_REGS (!HeavyRegisterPressure ? 232 : 240)
#endif
#ifndef U2_HOOK
#define U2_HOOK 0
#endif

// build.sh's patch puts U2_PRODUCER_HOOK after the cooperative kernel's producer warp roles and U2_CONSUMER_HOOK after its
// consumers' tile loop
namespace U2_NAME::hook {
__device__ __forceinline__ void idle(int warp);
__device__ __forceinline__ void done();
}  // namespace U2_NAME::hook
#if U2_HOOK
#define U2_PRODUCER_HOOK \
  if (warp_idx_in_warp_group != 0) ::U2_NAME::hook::idle(warp_idx_in_warp_group);
#define U2_CONSUMER_HOOK \
  if (thread_idx % 128 == 0) ::U2_NAME::hook::done();
#else
#define U2_PRODUCER_HOOK
#define U2_CONSUMER_HOOK
#endif

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#error "build for sm_90a"
#endif

#define U2_CAT2(a, b) a##_##b
#define U2_CAT(a, b) U2_CAT2(a, b)
#define U2_FN(f) U2_CAT(U2_NAME, f)

// a namespace of the configuration's own, since every configuration is linked into one library (an anonymous one collides
// with cute's in nvcc's host stub)
namespace U2_NAME {

using namespace cute;

using Tile = Shape<Int<U2_TM>, Int<U2_TN>, Int<U2_TK>>;
using Cluster = Shape<Int<U2_CM>, Int<U2_CN>, _1>;
using MainSched = cute::conditional_t<U2_PINGPONG, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                      cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
using EpiSched = cute::conditional_t<
    U2_NOSMEM, cutlass::epilogue::NoSmemWarpSpecialized,
    cute::conditional_t<U2_PINGPONG, cutlass::epilogue::TmaWarpSpecialized,
                        cutlass::epilogue::TmaWarpSpecializedCooperative>>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // B is N×K row-major: K-major, as wgmma s8 needs both operands
using LayoutD = cutlass::layout::RowMajor;
using Fusion = cutlass::epilogue::fusion::LinearCombination<int32_t, int32_t, void, int32_t>;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, int32_t, void, LayoutD, 4, int32_t, LayoutD, 4, EpiSched, Fusion>::CollectiveOp;
using Stages = cute::conditional_t<
    U2_STAGES == 0,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::StageCount<(U2_STAGES > 0 ? U2_STAGES : 1)>>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, int8_t, LayoutA, 16, int8_t, LayoutB, 16, int32_t, Tile, Cluster,
    Stages, MainSched>::CollectiveOp;
using GK = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue,
                                                cutlass::gemm::PersistentScheduler>;
#if U2_HOOK
static_assert(!U2_PINGPONG && U2_LOAD_REGS * 128 + U2_MMA_REGS * 256 <= 384 * 168,
              "the hook is the cooperative kernel's, within its launch's registers");
#endif

// inside the configuration's namespace: its __constant__ tables would collide across the library's units at global scope
#include "sha512x.cuh"

namespace hook {
struct Args {
  unsigned long long* sink;
  unsigned int* done;  // a counter per CTA, which each consumer warpgroup bumps as it ends its tiles
  unsigned long long* count;
  int on;
};
__device__ Args args;

__device__ __forceinline__ unsigned cta_index() { return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z); }

// hash.cu's chain on producer warp 1..3, polled once a compression
__device__ __forceinline__ void idle(int warp) {
  const Args a = args;
  if (!a.on) return;
  const unsigned cta = cta_index(), lane = threadIdx.x % 32;
  const volatile unsigned* flag = a.done + cta;
  const unsigned v0 = __shfl_sync(~0u, *flag, 0);
  const unsigned long long t = ((unsigned long long)cta * 3 + (warp - 1)) * 32 + lane;
  unsigned long long st[8], w[16];
  sx::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
  unsigned n = 0;
  do {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sx::compress<0, 1>(st, w);
    ++n;
  } while (__shfl_sync(~0u, *flag, 0) == v0);
  if (st[0] == 0x0123456789abcdefull) a.sink[t] = st[1];
  if (lane == 0) atomicAdd(a.count, 32ull * n);
}

__device__ __forceinline__ void done() {
  if (args.on) atomicAdd(args.done + cta_index(), 1u);
}
}  // namespace hook

// One call's stamps: the earliest start (as its complement, so atomicMax keeps the minimum of a zeroed record), the latest
// end, and the consumer warpgroups' summed clock64 and %globaltimer spans.
struct CallRec {
  unsigned long long nt0, t1, cyc, ns;
  unsigned int n, started;
};
static_assert(sizeof(CallRec) == 40, "CallRec is bench.py's 5 u64 words");

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void body(typename GK::Params const& params, CallRec* rec) {
  extern __shared__ char smem[];
  // the first thread of each consumer warpgroup (the producer warpgroup is 0..127)
  const bool stamp = threadIdx.x % 128 == 0 && threadIdx.x >= 128;
  unsigned long long t0 = 0, c0 = 0;
  if (stamp) {
    t0 = gtimer();
    c0 = clock64();
    if (threadIdx.x == 128) atomicAdd(&rec->started, 1u);
  }
  GK op;
  op(params, smem);
  if (stamp) {
    const unsigned long long c1 = clock64(), t1 = gtimer();
    atomicMax(&rec->nt0, ~t0);
    atomicMax(&rec->t1, t1);
    atomicAdd(&rec->cyc, c1 - c0);
    atomicAdd(&rec->ns, t1 - t0);
    atomicAdd(&rec->n, 1u);
  }
}

#if U2_MAXNREG
__global__ void __maxnreg__(U2_MAXNREG) entry(CUTLASS_GRID_CONSTANT typename GK::Params const params, CallRec* rec) {
  body(params, rec);
}
#else
__global__ void __launch_bounds__(GK::MaxThreadsPerBlock, GK::MinBlocksPerMultiprocessor)
    entry(CUTLASS_GRID_CONSTANT typename GK::Params const params, CallRec* rec) {
  body(params, rec);
}
#endif

struct Handle {
  typename GK::Params params;
  dim3 grid, block;
  int smem, clusters;
  void* workspace;
};

extern "C" {

// The configuration as JSON: its macros, threads, shared memory, the kernel's registers and spill from the CUDA runtime.
int U2_FN(info)(char* buf, int n) {
  cudaFuncAttributes a{};
  const cudaError_t e = cudaFuncGetAttributes(&a, entry);
  return snprintf(buf, n,
                  "{\"tile\": [%d, %d, %d], \"cluster\": [%d, %d], \"stages\": %d, \"pingpong\": %d, \"nosmem\": %d, "
                  "\"maxnreg\": %d, \"hook\": %d, \"threads\": %d, \"smem\": %d, \"regs\": %d, \"local_bytes\": %zu, "
                  "\"load_regs\": %d, \"mma_regs\": %d, \"attr_error\": \"%s\"}",
                  U2_TM, U2_TN, U2_TK, U2_CM, U2_CN, (int)Mainloop::DispatchPolicy::Stages, U2_PINGPONG, U2_NOSMEM,
                  U2_MAXNREG, U2_HOOK, (int)GK::MaxThreadsPerBlock, (int)GK::SharedStorageSize, a.numRegs,
                  a.localSizeBytes, (int)GK::LoadRegisterRequirement, (int)GK::MmaRegisterRequirement,
                  cudaGetErrorString(e));
}

// The hook's switch and buffers (U2_HOOK): done holds a counter per CTA of the grid, count gathers the compressions.
int U2_FN(hook)(void* sink, void* done, void* count, int on) {
  const hook::Args a{static_cast<unsigned long long*>(sink), static_cast<unsigned int*>(done),
                     static_cast<unsigned long long*>(count), on};
  return (int)cudaMemcpyToSymbol(hook::args, &a, sizeof a);
}

// A call's handle for D = A·Bᵀ on device pointers. raster: 0 CUTLASS's heuristic, 1 along M, 2 along N. sms: the SMs the
// persistent grid spans (0: the device's). Returns nullptr and prints why on stderr when CUTLASS can't implement it.
void* U2_FN(make)(int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* D, int raster, int swizzle, int sms) {
  int dev = 0;
  cudaGetDevice(&dev);
  if (sms <= 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  auto* h = new Handle{};
  h->smem = GK::SharedStorageSize;
  if (cudaFuncSetAttribute(entry, cudaFuncAttributeMaxDynamicSharedMemorySize, h->smem) != cudaSuccess) {
    fprintf(stderr, "u2 make: shared memory %d refused: %s\n", h->smem, cudaGetErrorString(cudaGetLastError()));
    delete h;
    return nullptr;
  }
  h->block = GK::get_block_shape();
  {
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = U2_CM;
    attr.val.clusterDim.y = U2_CN;
    attr.val.clusterDim.z = 1;
    cfg.gridDim = dim3(U2_CM * U2_CN, 1, 1);
    cfg.blockDim = h->block;
    cfg.dynamicSmemBytes = h->smem;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    h->clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&h->clusters, entry, &cfg) != cudaSuccess) h->clusters = 0;
    cudaGetLastError();
  }
  using StrideA = typename GK::StrideA;
  using StrideB = typename GK::StrideB;
  using StrideD = typename GK::StrideD;
  typename GK::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {M, N, K, 1};
  args.mainloop = {A, cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1)), B,
                   cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1))};
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = cutlass::make_cute_packed_stride(typename GK::StrideC{}, cute::make_shape(M, N, 1));
  args.epilogue.ptr_D = D;
  args.epilogue.dD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
  args.epilogue.thread.alpha = 1;
  args.epilogue.thread.beta = 0;
  args.hw_info.device_id = dev;
  args.hw_info.sm_count = sms;
  args.hw_info.max_active_clusters = h->clusters;
  args.scheduler.max_swizzle_size = swizzle;
  using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
  args.scheduler.raster_order = raster == 1 ? RO::AlongM : raster == 2 ? RO::AlongN : RO::Heuristic;
  if (!GK::can_implement(args)) {
    fprintf(stderr, "u2 make: CUTLASS can't implement M %d N %d K %d\n", M, N, K);
    delete h;
    return nullptr;
  }
  const size_t ws = GK::get_workspace_size(args);
  h->workspace = nullptr;
  if (ws && cudaMalloc(&h->workspace, ws) != cudaSuccess) {
    delete h;
    return nullptr;
  }
  if (GK::initialize_workspace(args, h->workspace, nullptr) != cutlass::Status::kSuccess) {
    fprintf(stderr, "u2 make: workspace initialization failed\n");
    delete h;
    return nullptr;
  }
  h->params = GK::to_underlying_arguments(args, h->workspace);
  h->grid = GK::get_grid_shape(h->params);
  return h;
}

// The persistent grid of a handle: x, y, z, and the clusters the occupancy calculator allows.
void U2_FN(grid)(void* hp, int* out) {
  auto* h = static_cast<Handle*>(hp);
  out[0] = h->grid.x;
  out[1] = h->grid.y;
  out[2] = h->grid.z;
  out[3] = h->clusters;
}

// One call on a stream, its stamps into rec (zeroed by the caller). Returns the launch's cudaError_t.
int U2_FN(run)(void* hp, void* stream, void* rec) {
  auto* h = static_cast<Handle*>(hp);
  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = U2_CM;
  attr.val.clusterDim.y = U2_CN;
  attr.val.clusterDim.z = 1;
  cfg.gridDim = h->grid;
  cfg.blockDim = h->block;
  cfg.dynamicSmemBytes = h->smem;
  cfg.stream = static_cast<cudaStream_t>(stream);
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  return (int)cudaLaunchKernelEx(&cfg, entry, h->params, static_cast<CallRec*>(rec));
}

void U2_FN(free)(void* hp) {
  auto* h = static_cast<Handle*>(hp);
  if (h->workspace) cudaFree(h->workspace);
  delete h;
}

}  // extern "C"

}  // namespace U2_NAME
