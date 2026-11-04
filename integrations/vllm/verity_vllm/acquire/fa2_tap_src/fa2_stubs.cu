// Verity (v8): link stubs so that the pinned flash_api.cpp (which dispatches over
// fp16/bf16 x hdim{32..256} x causal) links against a build that only contains the bf16 split-KV kernels of the head
// dimensions selected at build time (build_ext.py --hdims -> -DVERITY_HAVE_SPLIT_HDIM<N>_BF16).  Any call into a stub
// is a hard error; it never silently substitutes a different kernel (the head-dim check fails closed at the dispatch).
#include <stdexcept>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include "namespace_config.h"
#include "flash.h"

namespace FLASH_NAMESPACE {

#define VERITY_STUB_FWD(T, H, C) \
  template <> void run_mha_fwd_<T, H, C>(Flash_fwd_params &, cudaStream_t) { \
    throw std::runtime_error("verity fa2 build: run_mha_fwd_<" #T "," #H "," #C "> not compiled in this build"); }
#define VERITY_STUB_SPLIT(T, H, C) \
  template <> void run_mha_fwd_splitkv_dispatch<T, H, C>(Flash_fwd_params &, cudaStream_t) { \
    throw std::runtime_error("verity fa2 build: run_mha_fwd_splitkv_dispatch<" #T "," #H "," #C "> not compiled in this build (head_dim " #H " not in this tap's --hdims)"); }

#define VERITY_STUB_BOTH_C(MACRO, T, H) MACRO(T, H, true) MACRO(T, H, false)

// Non-split forward kernels (used only when block_table is absent): not built.
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::half_t, 32)
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::half_t, 64)
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::half_t, 96)
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::half_t, 128)
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::half_t, 192)
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::half_t, 256)
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::bfloat16_t, 32)
#ifndef VERITY_HAVE_FWD_HDIM64_BF16
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::bfloat16_t, 64)
#endif
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::bfloat16_t, 96)
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::bfloat16_t, 128)
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::bfloat16_t, 192)
VERITY_STUB_BOTH_C(VERITY_STUB_FWD, cutlass::bfloat16_t, 256)

// Split-KV dispatch: fp16 never; bf16 only the head dims NOT compiled from the pinned flash_fwd_split*_hdim<N>_bf16*.cu files.
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::half_t, 32)
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::half_t, 64)
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::half_t, 96)
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::half_t, 128)
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::half_t, 192)
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::half_t, 256)
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::bfloat16_t, 32)
#ifndef VERITY_HAVE_SPLIT_HDIM64_BF16
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::bfloat16_t, 64)
#endif
#ifndef VERITY_HAVE_SPLIT_HDIM96_BF16
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::bfloat16_t, 96)
#endif
#ifndef VERITY_HAVE_SPLIT_HDIM128_BF16
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::bfloat16_t, 128)
#endif
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::bfloat16_t, 192)
#ifndef VERITY_HAVE_SPLIT_HDIM256_BF16
VERITY_STUB_BOTH_C(VERITY_STUB_SPLIT, cutlass::bfloat16_t, 256)
#endif

}  // namespace FLASH_NAMESPACE
