// GEMMs registered beside cuBLASLt: Y [m, n] = X [m, k] W [n, k]^T (+ R [m, n] when R is given), BF16 in and out, FP32
// accumulation, all row-major. Each gemm_cutlass.cu object linked into the bench registers one (ExtReg); gemms.cu collects them.
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
using ExtFactory = std::vector<ExtGemm> (*)(int m, int n, int k);
std::vector<ExtFactory>& ext_registry();
struct ExtReg {
  explicit ExtReg(ExtFactory f) { ext_registry().push_back(f); }
};
std::vector<ExtGemm> ext_gemms(int m, int n, int k);
