// One kernel of cutlass_receipt.cu: the CUTLASS sm_120 FP8 GEMM vLLM serves for Llama-3.1-8B's linears (its
// cutlass_scaled_mm: MainloopSm120TmaWarpSpecialized from CUTLASS's CollectiveBuilder, TMA into smem stages, a producer
// warpgroup, mma.sync m16n8k32 E4M3 into FP32 accumulators that run in place over all of K, and the epilogue
// bf16(sa[m] * (sb[n] * acc))), in one of its two configurations:
//   CR_CFG 0  prefill: KernelTmaWarpSpecializedCooperative, 128x128x128 tiles, 8 MMA warps (4x2)
//   CR_CFG 1  decode:  KernelTmaWarpSpecializedPingpong, 32x64x128 tiles, two 4-warp MMA warpgroups (2x2)
// with the builder's stage count. vLLM's epilogue uses its own Sm90{Col,Row}OrScalarBroadcast; this uses CUTLASS's
// Sm90{Col,Row}Broadcast, the same loads without the scalar option.
//   CR_MODE 0  the builder's mainloop, unmodified
//   CR_MODE 1  the same mainloop with phi.cuh's receipt in every P-th k-tile: the accumulators then hold the prefix sums
//              at k = 128 P, 256 P, ..., K, so L = 128 P. Each thread chains the 4 words of each of its m16n8 accumulator
//              tiles (one chain a tile) and folds its chains into one word a tile at the end. CR_SPREAD 0 runs every
//              chain after the k-tile's last MMA; CR_SPREAD 1 interleaves them, a quarter of the chains after each of the
//              k-tile's four 32-deep k-blocks (each chain's prefix is then at that k-block's end). CR_GRAIN G > 1 (P 1,
//              no spread) runs every chain G times a k-tile, after every (4 / G)-th k-block: L = 128 / G, down to 32.
//   CR_MODE 2  the dump: every k-block's prefix sums to global memory, and each thread's chain-word coordinates, so the
//              host can recompute every receipt kernel's words
// Compiled once per kernel: -DCR_NAME=name -DCR_CFG -DCR_VAR -DCR_ROUNDS -DCR_P -DCR_MODE -DCR_SPREAD -DCR_GRAIN
// (job.sh's cgemm).
#include "cutlass_receipt.cuh"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "phi.cuh"

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#error "build for sm_120a with CUDA 12.8 or later"
#endif

#ifndef CR_SPREAD
#define CR_SPREAD 0
#endif
#ifndef CR_GRAIN
#define CR_GRAIN 1
#endif
#define CR_STR2(x) #x
#define CR_STR(x) CR_STR2(x)

using namespace cute;

namespace {

struct Out {
    uint32_t* R;
    float* dump;
    uint32_t* map;
    int tiles_n, m, n;
};
__device__ Out g_out;

template <class Base, int VAR, int ROUNDS, int P, int MODE, int SPREAD, int GRAIN>
struct ReceiptMma : Base {
    static_assert(GRAIN == 1 || (P == 1 && !SPREAD), "a grain above 1 receipts every k-tile, without spread");
    using MainloopPipeline = typename Base::MainloopPipeline;
    using PipelineState = typename Base::PipelineState;
    using TensorStorage = typename Base::TensorStorage;
    using Params = typename Base::Params;
    using SmemLayoutA = typename Base::SmemLayoutA;
    using SmemLayoutB = typename Base::SmemLayoutB;
    using TiledMma = typename Base::TiledMma;
    using SmemCopyAtomA = typename Base::SmemCopyAtomA;
    using SmemCopyAtomB = typename Base::SmemCopyAtomB;
    using TileShape = typename Base::TileShape;
    using DispatchPolicy = typename Base::DispatchPolicy;

    // Base::mma (CUTLASS v4.8.0, sm120_mma_tma.hpp) with hook(kt) after k-tile kt's last MMA; FP8 has no FP4 shift
    template <class FrgTensorC, class BlockCoord>
    CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read, FrgTensorC& accum, int k_tile_count,
                            int thread_idx, TensorStorage& shared_tensors, Params const& mainloop_params,
                            BlockCoord& blk_crd) {
        static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
        (void)mainloop_params;
        constexpr int MW = decltype(size<0>(accum))::value;
        constexpr int MM = decltype(size<1>(accum))::value;
        constexpr int NN = decltype(size<2>(accum))::value;
        constexpr int CH = MM * NN;
        static_assert(MW == 4, "an m16n8 FP32 accumulator tile is 4 words a thread");

        clear(accum);
        uint32_t y[CH];
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < CH; ++c) y[c] = 0;

        Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});
        TiledMma tiled_mma;
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
        Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));
        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
        Tensor tCsA = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
        Tensor tCsB = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

        auto K_BLOCK_MAX = size<2>(tCrA);
        int read_stage = smem_pipe_read.index();
        auto tCsA_stage = tCsA(_, _, _, read_stage);
        auto tCsB_stage = tCsB(_, _, _, read_stage);
        auto copy_kblock = [&](auto k_block) {
            copy(smem_tiled_copy_A, tCsA_stage(_, _, k_block), tCrA_copy_view(_, _, k_block));
            copy(smem_tiled_copy_B, tCsB_stage(_, _, k_block), tCrB_copy_view(_, _, k_block));
        };
        auto gemm_kblock = [&](auto k_block) { cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum); };

        constexpr int KB = decltype(K_BLOCK_MAX)::value;
        static_assert(NN == decltype(size<1>(TileShape{}))::value / 16, "cutlass_receipt.cu's nn_max assumes 2 MMA warps in n");
        static_assert(KB % GRAIN == 0, "a grain divides the k-tile's k-blocks");
        constexpr int GS = KB / GRAIN;
        int kt = 0;
        // After k-block b of k-tile kt: receipted k-tiles run φ on every chain at b = KB - 1 (GRAIN > 1: at every
        // b = GS - 1 mod GS, with w the k-block's index from 1), or (SPREAD) on the chains with nn * KB / NN == b, a
        // quarter of them a k-block here.
        auto hook = [&](auto k_block, auto receipted) {
            constexpr int b = decltype(k_block)::value;
            if constexpr (MODE == 1 && decltype(receipted)::value && (SPREAD || (b + 1) % GS == 0)) {
                const uint32_t w = GRAIN > 1 ? (uint32_t)(kt * KB + b + 1) : (uint32_t)(kt + 1);
                CUTLASS_PRAGMA_UNROLL
                for (int mm = 0; mm < MM; ++mm)
                    CUTLASS_PRAGMA_UNROLL
                    for (int nn = 0; nn < NN; ++nn)
                        if (!SPREAD || nn * KB / NN == b) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int e = 0; e < MW; ++e)
                                y[mm * NN + nn] = phi<ROUNDS, VAR>(y[mm * NN + nn], __float_as_uint(accum(e, mm, nn)), w,
                                                                   key(mm * NN + nn, e));
                        }
            } else if constexpr (MODE == 2) {
                Tensor tCcC = thread_mma.partition_C(make_identity_tensor(take<0, 2>(TileShape{})));
                const int m0 = int(get<0>(blk_crd)) * decltype(size<0>(TileShape{}))::value;
                const int n0 = int(get<1>(blk_crd)) * decltype(size<1>(TileShape{}))::value;
                const size_t kb = (size_t)kt * KB + b;
                CUTLASS_PRAGMA_UNROLL
                for (int mm = 0; mm < MM; ++mm)
                    CUTLASS_PRAGMA_UNROLL
                    for (int nn = 0; nn < NN; ++nn)
                        CUTLASS_PRAGMA_UNROLL
                        for (int e = 0; e < MW; ++e) {
                            const int r = int(get<0>(tCcC(e, mm, nn))), c = int(get<1>(tCcC(e, mm, nn)));
                            g_out.dump[(kb * g_out.m + m0 + r) * g_out.n + n0 + c] = accum(e, mm, nn);
                            if (kb == 0 && m0 == 0 && n0 == 0)
                                g_out.map[(thread_idx * CH + mm * NN + nn) * MW + e] = (uint32_t(r) << 16) | uint32_t(c);
                        }
            }
            if constexpr (b == KB - 1) ++kt;
        };

        // One k-tile of the main loop. The stride is unrolled at compile time: a runtime `(kt + 1) % P` test is
        // if-converted by nvcc, which then runs φ on every k-tile and selects the result.
        auto ktile = [&](auto receipted) {
            for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
                if (k_block == K_BLOCK_MAX - 1) {
                    cutlass::arch::NamedBarrier::sync(thr_size(tiled_mma),
                                                      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
                    pipeline.consumer_release(smem_pipe_read);
                    ++smem_pipe_read;
                    read_stage = smem_pipe_read.index();
                    tCsA_stage = tCsA(_, _, _, read_stage);
                    tCsB_stage = tCsB(_, _, _, read_stage);
                    pipeline.consumer_wait(smem_pipe_read);
                }
                copy_kblock(k_block_next);
                gemm_kblock(k_block);
                hook(k_block, receipted);
            });
        };

        // k_tile_count is a multiple of P (make() refuses other K).
        pipeline.consumer_wait(smem_pipe_read);
        copy_kblock(_0{});
        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > P; k_tile_count -= P) {
            for_each(make_int_sequence<P - 1>{}, [&](auto) { ktile(cute::false_type{}); });
            ktile(cute::true_type{});
        }
        for_each(make_int_sequence<P - 1>{}, [&](auto) { ktile(cute::false_type{}); });
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
            auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
            if (k_block == K_BLOCK_MAX - 1) {
                cutlass::arch::NamedBarrier::sync(thr_size(tiled_mma),
                                                  cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
                pipeline.consumer_release(smem_pipe_read);
                ++smem_pipe_read;
            }
            if (k_block_next > 0) copy_kblock(k_block_next);
            gemm_kblock(k_block);
            hook(k_block, cute::true_type{});
        });

        if constexpr (MODE == 1) {
            uint32_t f = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < CH; ++c) f = rotl13(f) ^ y[c];
            constexpr int NT = decltype(size(TiledMma{}))::value;
            const int tile = int(get<0>(blk_crd)) * g_out.tiles_n + int(get<1>(blk_crd));
            g_out.R[(size_t)tile * NT + thread_idx] = f;
        }
    }
};

using E4M3 = cutlass::float_e4m3_t;
using BF16 = cutlass::bfloat16_t;
using Cluster = Shape<_1, _1, _1>;
using Tile = cute::conditional_t<CR_CFG == 0, Shape<_128, _128, _128>, Shape<_32, _64, _128>>;
using Schedule = cute::conditional_t<CR_CFG == 0, cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                     cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
// vLLM's served instantiations name these epilogue tiles (64x32 cooperative, 32x32 ping-pong).
using EpiTile = cute::conditional_t<CR_CFG == 0, Shape<_64, _32>, Shape<_32, _32>>;

namespace F = cutlass::epilogue::fusion;
using ScaleA = F::Sm90ColBroadcast<0, Tile, float, float, Stride<_1, _0, _0>>;
using ScaleB = F::Sm90RowBroadcast<0, Tile, float, float, Stride<_0, _1, _0>>;
using Compute0 = F::Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using EVT0 = F::Sm90EVT<Compute0, ScaleB, F::Sm90AccFetch>;
using Compute1 = F::Sm90Compute<cutlass::multiplies, BF16, float, cutlass::FloatRoundStyle::round_to_nearest>;
using EVT = F::Sm90EVT<Compute1, ScaleA, EVT0>;

constexpr int AlignD = 128 / cutlass::sizeof_bits<BF16>::value;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster, EpiTile,
    float, float, void, cutlass::layout::RowMajor, AlignD, BF16, cutlass::layout::RowMajor, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto, EVT>::CollectiveOp;
using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, E4M3, cutlass::layout::RowMajor, 16, E4M3,
    cutlass::layout::ColumnMajor, 16, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    Schedule>::CollectiveOp;
using Mainloop = cute::conditional_t<CR_MODE == 0, BaseMainloop, ReceiptMma<BaseMainloop, CR_VAR, CR_ROUNDS, CR_P, CR_MODE, CR_SPREAD, CR_GRAIN>>;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

struct Impl final : cr_impl {
    Gemm gemm;
    void* ws = nullptr;
    int launch(cudaStream_t s) override { return gemm.run(s) == cutlass::Status::kSuccess ? 0 : 1; }
    ~Impl() override {
        if (ws) cudaFree(ws);
    }
};

cr_impl* make(const cr_args& a, int* err) {
    using GK = GemmKernel;
    auto sA = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {a.m, a.k, 1});
    auto sB = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {a.n, a.k, 1});
    auto sC = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {a.m, a.n, 1});
    auto sD = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {a.m, a.n, 1});
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {a.m, a.n, a.k, 1},
                                  {reinterpret_cast<E4M3 const*>(a.A), sA, reinterpret_cast<E4M3 const*>(a.B), sB},
                                  {{}, nullptr, sC, reinterpret_cast<BF16*>(a.D), sD}};
    typename EVT0::Arguments e0{typename ScaleB::Arguments{a.sb, 0.0f, {}}, {}, {}};
    args.epilogue.thread = typename EVT::Arguments{typename ScaleA::Arguments{a.sa, 0.0f, {}}, e0, {}};
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess || a.k % (128 * CR_P) != 0) {
        *err = 1;
        return nullptr;
    }
    const Out o{a.R, a.dump, a.map, a.n / decltype(size<1>(Tile{}))::value, a.m, a.n};
    if (cudaMemcpyToSymbol(g_out, &o, sizeof o) != cudaSuccess) {
        *err = 2;
        return nullptr;
    }
    auto* r = new Impl();
    const size_t need = Gemm::get_workspace_size(args);
    if (need && cudaMalloc(&r->ws, need) != cudaSuccess) {
        delete r;
        *err = 3;
        return nullptr;
    }
    if (r->gemm.initialize(args, r->ws) != cutlass::Status::kSuccess) {
        delete r;
        *err = 4;
        return nullptr;
    }
    *err = 0;
    return r;
}

constexpr int kTileM = decltype(size<0>(Tile{}))::value, kTileN = decltype(size<1>(Tile{}))::value;
constexpr int kThreads = decltype(size(typename BaseMainloop::TiledMma{}))::value;

const cr_kernel K{CR_STR(CR_NAME),
                  CR_CFG,
                  CR_VAR,
                  CR_ROUNDS,
                  CR_P,
                  CR_MODE,
                  CR_SPREAD,
                  CR_GRAIN,
                  kTileM,
                  kTileN,
                  kThreads,
                  kTileM * kTileN / kThreads / 4,
                  kTileN / 16,
                  128 / 32,
                  BaseMainloop::DispatchPolicy::Stages,
                  int(sizeof(typename GemmKernel::SharedStorage)),
                  &make};
const cr_register reg(&K);

}  // namespace
