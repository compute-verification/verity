// The C ABI over the baselines' tables (cutlass_3x.cu, cutlass_2x.cu, nvf4_plain.cu).  Built with -DPH_NO_CUTLASS it has
// only nvf4_plain.cu's, so the harness still loads where CUTLASS isn't built.
#include <cuda_runtime.h>

#include <cstdio>

#include "cutlass_api.h"

namespace {
const phc_entry* entry(int i, int* total) {
  using table = const phc_entry* (*)(int*);
  const table tables[] = {
#ifndef PH_NO_CUTLASS
      phc_table_3x_0, phc_table_3x_1, phc_table_3x_2, phc_table_3x_3, phc_table_3x_4, phc_table_2x, phc_table_3x_5,
      phc_table_3x_6,
#endif
      phc_table_nvf4, phc_table_fp8};
  const phc_entry* found = nullptr;
  *total = 0;
  for (table t : tables) {
    int n = 0;
    const phc_entry* e = t(&n);
    if (!found && i >= *total && i < *total + n) found = &e[i - *total];
    *total += n;
  }
  return found;
}

bool sched_ok(int swizzle, int raster) {
  return (swizzle == 0 || swizzle == 1 || swizzle == 2 || swizzle == 4 || swizzle == 8) && raster >= 0 && raster <= 2;
}

int query(const phc_entry* e, const phc_args* a, size_t* ws, size_t* sfa_bytes, size_t* sfb_bytes) {
  if (e->query) return e->query(a, ws, sfa_bytes, sfb_bytes);
  if (a->swizzle || a->raster) return -3;  // its tile order is its own
  return e->query_fixed(a->m, a->n, a->k, ws, sfa_bytes, sfb_bytes);
}
}  // namespace

extern "C" {

int ph_cutlass_count() {
  int total = 0;
  entry(0, &total);
  return total;
}

// The kernel's name, kind (harness.cu's LT_* values) and whether it is swapped; -1 for an unknown index.
int ph_cutlass_describe(int i, const char** name, int* kind, int* swap) {
  int total = 0;
  const phc_entry* e = entry(i, &total);
  if (!e) return -1;
  *name = e->name;
  *kind = e->kind;
  *swap = e->swap;
  return 0;
}

// swizzle and raster as phc_args has them; -2 for a value outside those, -3 for an order a kernel of its own order doesn't
// take.
int ph_cutlass_query_sched(int i, int m, int n, int k, int swizzle, int raster, size_t* ws, size_t* sfa_bytes,
                           size_t* sfb_bytes) {
  int total = 0;
  const phc_entry* e = entry(i, &total);
  if (!e) return -1;
  if (!sched_ok(swizzle, raster)) return -2;
  phc_args a{m, n, k, nullptr, nullptr, nullptr, nullptr, nullptr, swizzle, raster};
  return query(e, &a, ws, sfa_bytes, sfb_bytes);
}

int ph_cutlass_query(int i, int m, int n, int k, size_t* ws, size_t* sfa_bytes, size_t* sfb_bytes) {
  return ph_cutlass_query_sched(i, m, n, k, 0, 0, ws, sfa_bytes, sfb_bytes);
}

// Initializes kernel i for one operand set; *out is a handle for ph_cutlass_launch and ph_cutlass_release.
int ph_cutlass_prepare_sched(int i, int m, int n, int k, int swizzle, int raster, const void* A, const void* B,
                             const void* sfa, const void* sfb, void* D, void* ws, size_t ws_bytes, void* stream,
                             void** out) {
  int total = 0;
  const phc_entry* e = entry(i, &total);
  if (!e) return -1;
  if (!sched_ok(swizzle, raster)) return -2;
  if (!e->query && (swizzle || raster)) return -3;
  phc_args a{m, n, k, A, B, sfa, sfb, D, swizzle, raster};
  phc_prepared* p = nullptr;
  int r = e->prepare(&a, ws, ws_bytes, stream, &p);
  *out = p;
  return r;
}

int ph_cutlass_prepare(int i, int m, int n, int k, const void* A, const void* B, const void* sfa, const void* sfb, void* D,
                       void* ws, size_t ws_bytes, void* stream, void** out) {
  return ph_cutlass_prepare_sched(i, m, n, k, 0, 0, A, B, sfa, sfb, D, ws, ws_bytes, stream, out);
}

int ph_cutlass_launch(void* h, void* stream) {
  int r = static_cast<phc_prepared*>(h)->launch(stream);
  if (r) return r;
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 1000 + int(e);
}

int ph_cutlass_release(void* h) {
  delete static_cast<phc_prepared*>(h);
  return 0;
}

}  // extern "C"
