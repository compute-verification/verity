// CUTLASS 2.x baselines for BF16 and int8: mma.sync kernels (Sm80's HMMA.16816 and IMMA.16832, which sm_120 runs), in
// tile shapes that fit sm_120's 99 KB of shared memory per block.  CUTLASS 3.x's sm_120 builder takes only f8f6f4 types.
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass_api.h"

namespace {

using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using cutlass::gemm::GemmShape;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages>
using Bf16 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, RM, cutlass::bfloat16_t, CM, cutlass::bfloat16_t, RM, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, GemmShape<TM, TN, TK>, GemmShape<WM, WN, WK>, GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, Stages>;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages>
using Int8 = cutlass::gemm::device::Gemm<
    int8_t, RM, int8_t, CM, int32_t, RM, int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    GemmShape<TM, TN, TK>, GemmShape<WM, WN, WK>, GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, Stages>;

template <class G>
typename G::Arguments arguments(int m, int n, int k, const void* A, const void* B, void* D) {
  using EA = typename G::ElementA;
  using EB = typename G::ElementB;
  using EC = typename G::ElementC;
  auto a = reinterpret_cast<EA const*>(A);
  auto b = reinterpret_cast<EB const*>(B);
  auto d = reinterpret_cast<EC*>(D);
  using Compute = typename G::EpilogueOutputOp::ElementCompute;
  return typename G::Arguments({m, n, k}, {a, k}, {b, k}, {d, n}, {d, n}, {Compute(1), Compute(0)});
}

template <class G>
struct Prepared final : phc_prepared {
  G gemm;
  int launch(void* stream) override {
    auto s = gemm.run(cudaStream_t(stream));
    return s == cutlass::Status::kSuccess ? 0 : 3000 + int(s);
  }
};

template <class G>
int query(const phc_args* a, size_t* ws, size_t* sfa, size_t* sfb) {
  if (a->swizzle || a->raster) return 3000 + int(cutlass::Status::kErrorNotSupported);
  auto args = arguments<G>(a->m, a->n, a->k, nullptr, nullptr, nullptr);
  auto s = G::can_implement(args);
  if (s != cutlass::Status::kSuccess) return 3000 + int(s);
  *ws = G::get_workspace_size(args);
  *sfa = *sfb = 0;
  return 0;
}

template <class G>
int prepare(const phc_args* a, void* ws, size_t ws_bytes, void* stream, phc_prepared** out) {
  if (a->swizzle || a->raster) return 3000 + int(cutlass::Status::kErrorNotSupported);
  auto args = arguments<G>(a->m, a->n, a->k, a->A, a->B, a->D);
  if (G::get_workspace_size(args) > ws_bytes) return 3100;
  auto* r = new Prepared<G>();
  auto s = r->gemm.initialize(args, ws, cudaStream_t(stream));
  if (s != cutlass::Status::kSuccess) {
    delete r;
    return 3000 + int(s);
  }
  *out = r;
  return 0;
}

#define ENTRY(NAME, KIND, G) {NAME, KIND, 0, &query<G>, &prepare<G>}

using Bf16_128x128x32_s4 = Bf16<128, 128, 32, 64, 64, 32, 4>;
using Bf16_128x128x64_s3 = Bf16<128, 128, 64, 64, 64, 64, 3>;
using Bf16_128x256x32_s3 = Bf16<128, 256, 32, 64, 64, 32, 3>;
using Bf16_256x128x32_s3 = Bf16<256, 128, 32, 64, 64, 32, 3>;
using Bf16_64x128x64_s3 = Bf16<64, 128, 64, 32, 64, 64, 3>;
using Int8_128x128x64_s4 = Int8<128, 128, 64, 64, 64, 64, 4>;
using Int8_128x256x64_s3 = Int8<128, 256, 64, 64, 64, 64, 3>;
using Int8_256x128x64_s3 = Int8<256, 128, 64, 64, 64, 64, 3>;
using Int8_64x128x128_s3 = Int8<64, 128, 128, 32, 64, 128, 3>;

const phc_entry TABLE[] = {
    ENTRY("cutlass2x_bf16_128x128x32_s4", 0, Bf16_128x128x32_s4),
    ENTRY("cutlass2x_bf16_128x128x64_s3", 0, Bf16_128x128x64_s3),
    ENTRY("cutlass2x_bf16_128x256x32_s3", 0, Bf16_128x256x32_s3),
    ENTRY("cutlass2x_bf16_256x128x32_s3", 0, Bf16_256x128x32_s3),
    ENTRY("cutlass2x_bf16_64x128x64_s3", 0, Bf16_64x128x64_s3),
    ENTRY("cutlass2x_int8_128x128x64_s4", 7, Int8_128x128x64_s4),
    ENTRY("cutlass2x_int8_128x256x64_s3", 7, Int8_128x256x64_s3),
    ENTRY("cutlass2x_int8_256x128x64_s3", 7, Int8_256x128x64_s3),
    ENTRY("cutlass2x_int8_64x128x128_s3", 7, Int8_64x128x128_s3),
};

}  // namespace

const phc_entry* phc_table_2x(int* count) {
  *count = int(sizeof TABLE / sizeof TABLE[0]);
  return TABLE;
}
