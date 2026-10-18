// GEMMs registered beside cuBLASLt: Y [m, n] = X [m, k] W [n, k]^T (+ R [m, n] when R is given), BF16 in and out, FP32
// accumulation, all row-major. Each translation unit that defines ext_gemms (gemms_none.cu, gemms_sm90.cu, gemms_sm100.cu)
// is one build of the bench.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

struct ExtGemm {
  std::string name;
  void* state;
  // 0 on success
  int (*run)(void* state, const __nv_bfloat16* X, const __nv_bfloat16* W, const __nv_bfloat16* R, __nv_bfloat16* Y, cudaStream_t st);
};
std::vector<ExtGemm> ext_gemms(int m, int n, int k);
