// Verity: link stubs for the FA3 tap build.  flash_api.cpp's dispatch (with the build's FLASHATTENTION_DISABLE_*
// set: sm90 only, bf16 only, hdim 64/128, PackGQA only, no split/softcap/local/appendKV/fp8/fp16) still names the NON-paged
// PackGQA forward of each head dim; vLLM's paged-KV serving launches never reach it (page 16 % kBlockN != 0 -> PagedKVNonTMA), so it
// is not compiled -- any call is a hard error, never a silent substitute.  Head dims not selected at build time are stubbed too.
#include <stdexcept>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include "flash.h"

#define VERITY_STUB(HD, PAGED) \
  template <> void run_mha_fwd_<90, cutlass::bfloat16_t, HD, HD, false, PAGED, false, true>(Flash_fwd_params &, cudaStream_t) { \
    throw std::runtime_error("verity fa3 tap build: run_mha_fwd_<90,bf16," #HD "," #HD ",Split=0,PagedKVNonTMA=" #PAGED ",softcap=0,PackGQA=1> not compiled in this build"); }

VERITY_STUB(64, false)
VERITY_STUB(128, false)
#ifndef VERITY_HAVE_FA3_HDIM64_BF16
VERITY_STUB(64, true)
#endif
#ifndef VERITY_HAVE_FA3_HDIM128_BF16
VERITY_STUB(128, true)
#endif
