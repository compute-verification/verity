// The int8 GEMM entry a served call takes at S = 1 under -DV31_EXT_GEMM (v31.cu's Fill): D = A B^T exactly in s32, A m x k
// and B n x k int8 row-major, D m x n s32 row-major (|a' b'| <= 2^14 a product and K < 2^17, so no sum wraps). Under
// V31_EXT_GEMM=1 v31.cu defines the entry itself on cuBLASLt; under V31_EXT_GEMM=2 another source on the nvcc line does.
// Every call is issued on the stream it's given, in that stream's order, and returns 0, or nonzero to refuse (the call
// exits). A context is driven by one host thread.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

extern "C" {
// A context for one shape; null refuses it.
void* v31_ext_init(int m, int n, int k);
// B for the GEMMs after it: issued each time the call forms b' again (under per-call weight coins, every call). Its forms
// and metadata are prepared here; it may read b until the next load or free.
int v31_ext_load(void* h, const int8_t* b, cudaStream_t st);
// D = A B^T, B the last one loaded.
int v31_ext_gemm(void* h, const int8_t* a, int32_t* d, cudaStream_t st);
void v31_ext_free(void* h);
}
