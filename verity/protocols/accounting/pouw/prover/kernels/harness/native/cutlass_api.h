// The registry of compiled baselines (CUTLASS's and nvf4_plain.cu's), shared by the kernel translation units and the C ABI
// (cutlass_registry.cu).
#pragma once
#include <cstddef>

// Kinds are harness.cu's LT_* values: 0 BF16, 1 FP8 E4M3 (per-tensor), 3 MXFP8, 4 FP8 groupwise (1x128 A, 128x128 B),
// 5 NVFP4, 6 MXFP4, 7 INT8.
struct phc_args {
  int m, n, k;
  const void *A, *B, *sfa, *sfb;  // ours: A m x k and B n x k, both K-major; their scale factors, or null
  void* D;                        // m x n row-major, BF16 (INT8: int32)
  // The 3.x tile scheduler's order of output tiles: raster 0 CUTLASS's heuristic, 1 along our m, 2 along our n (a swapped
  // kernel's N and M); swizzle 0 CUTLASS's default (which orders as 1 does), else its max_swizzle_size (1, 2, 4 or 8),
  // which CUTLASS lowers where the grid is too small for it. 2.x kernels fix both at compile time and take only 0, 0.
  int swizzle = 0, raster = 0;
};

// An initialized kernel for one operand set: `launch` enqueues it (params cached, nothing built per call).
struct phc_prepared {
  virtual ~phc_prepared() {}
  virtual int launch(void* stream) = 0;
};

struct phc_entry {
  // 0 if implementable at a's (m, n, k, swizzle, raster; pointers unused), with the workspace and scale-factor bytes it
  // needs; else a nonzero status
  using query_fn = int (*)(const phc_args* a, size_t* ws, size_t* sfa_bytes, size_t* sfb_bytes);
  // the same for a kernel whose tile order is its own (nvf4_plain.cu's): m, n, k alone; it takes only swizzle 0, raster 0
  using query_fixed_fn = int (*)(int m, int n, int k, size_t* ws, size_t* sfa_bytes, size_t* sfb_bytes);
  using prepare_fn = int (*)(const phc_args* a, void* ws, size_t ws_bytes, void* stream, phc_prepared** out);
  const char* name;
  int kind;
  int swap;  // 1: computes D^T = B . A^T (M-tile over n), for skinny m
  query_fn query;              // one of query and query_fixed is set
  query_fixed_fn query_fixed;
  prepare_fn prepare;
  constexpr phc_entry(const char* name, int kind, int swap, query_fn q, prepare_fn p)
      : name(name), kind(kind), swap(swap), query(q), query_fixed(nullptr), prepare(p) {}
  constexpr phc_entry(const char* name, int kind, int swap, query_fixed_fn q, prepare_fn p)
      : name(name), kind(kind), swap(swap), query(nullptr), query_fixed(q), prepare(p) {}
};

// cutlass_3x.cu compiled with -DPH_PART=0..6, and cutlass_2x.cu.
const phc_entry* phc_table_3x_0(int* count);
const phc_entry* phc_table_3x_1(int* count);
const phc_entry* phc_table_3x_2(int* count);
const phc_entry* phc_table_3x_3(int* count);
const phc_entry* phc_table_3x_4(int* count);
const phc_entry* phc_table_3x_5(int* count);
const phc_entry* phc_table_3x_6(int* count);
const phc_entry* phc_table_2x(int* count);
// ../../nvfp4_sm120/nvf4_plain.cu, no CUTLASS: the plain NVFP4 (kind 5) and plain FP8 E4M3 (kind 1) GEMMs,
// 256 x 128 tiles, one persistent CTA an SM; m a multiple of 256, n of 128, k of 128 (FP8: 64). Each keeps its two
// CUtensorMaps in its workspace (256 bytes, 64-byte aligned) from prepare to every launch.
const phc_entry* phc_table_nvf4(int* count);
const phc_entry* phc_table_fp8(int* count);
