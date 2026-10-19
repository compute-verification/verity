// kbench's device side, one shared library a family (sm_90a or sm_100a): compute accounting's honest Dither GEMM
// (benchmarks/pouw/dc/dgemm.cu, included unchanged: dg90 on sm_90a, dg100 on sm_100a), a row reference of its receipt, and
// cuBLASLt int8 with its top algorithms, as plain launches on a caller's stream (kbench.py captures them in CUDA graphs).
//
//   kb_init()                                    the family (90 or 100) from device 0; dgemm.cu's g_sm
//   kb_dg_tile(&bm, &bn)                         dgemm.cu's tile: 64 x 128 (dg90), 128 x 256 (dg100)
//   kb_dg_launch(fold, write, X, W, m, n, k, seed, C, R, stream)
//                                                dgemm.cu's launch<FOLD, WRITE>; m, n, k must fit its tile (checked here,
//                                                since dgemm.cu's shape_for exits)
//   kb_ref_rows(X, W, n, k, bn, L, seed, rows, nrows, C, R, stream)
//                                                the receipt by its definition (dgemm.cu's header, host_ref's loop) for the
//                                                rows listed: one thread a chain (row i, tile tn, c), its columns' running
//                                                sums from X and W directly, absorbed at each boundary in ascending column.
//                                                R[r][tn][c] and C[r][j] for the r-th listed row. Shared by dgemm (L = 32)
//                                                and pgemm (L = 32 lm, bn = 512 / NB)
//   kb_lt_create(m, n, k, ws, wsb) / kb_lt_algos(h) / kb_lt_run(h, a, X, W, C, stream) / kb_lt_destroy(h)
//                                                C (s32, m x n) = X W^T by cuBLASLt (s8, COMPUTE_32I), its heuristic's top
//                                                algorithms; kbench.py times each and keeps the fastest
#define main dgemm_main
#include "../../pouw/dc/dgemm.cu"
#undef main

#include <cublasLt.h>

extern "C" int kb_init() {
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 0;
  g_sm = p.major == 10 ? 100 : p.major == 9 ? 90 : 0;
  return g_sm;
}

extern "C" void kb_dg_tile(int* bm, int* bn) {
  *bm = g_sm == 100 ? 128 : 64;
  *bn = g_sm == 100 ? 256 : 128;
}

extern "C" int kb_dg_launch(int fold, int write, const void* X, const void* W, int m, int n, int k, unsigned seed, void* C,
                            void* R, void* stream) {
  int bm, bn;
  kb_dg_tile(&bm, &bn);
  if (!g_sm || m % bm || n % bn || k % L) return 2;
  const Shape sh{m, n, k, bm, bn};
  const auto x = (const int8_t*)X;
  const auto w = (const int8_t*)W;
  const auto st = (cudaStream_t)stream;
  if (fold && write) launch<true, true>(sh, x, w, seed, (int32_t*)C, (uint32_t*)R, st);
  else if (fold) launch<true, false>(sh, x, w, seed, (int32_t*)C, (uint32_t*)R, st);
  else if (write) launch<false, true>(sh, x, w, seed, (int32_t*)C, (uint32_t*)R, st);
  else launch<false, false>(sh, x, w, seed, (int32_t*)C, (uint32_t*)R, st);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

__global__ void kb_ref_rows_k(const int8_t* X, const int8_t* W, int n, int k, int bn, int Lc, uint32_t seed, const int* rows,
                              int nrows, int32_t* C, uint32_t* R) {
  const int tiles_n = n / bn, g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= nrows * tiles_n * NCH) return;
  const int c = g % NCH, tn = (g / NCH) % tiles_n, r = g / (NCH * tiles_n), i = rows[r], ncol = bn / NCH;
  DuoSt st = duo_init(seed, (uint32_t)((i * tiles_n + tn) * NCH + c));
  int32_t T[32];
  for (int q = 0; q < ncol; q++) T[q] = 0;
  const int8_t* x = X + (size_t)i * k;
  for (int s = 0; s < k / Lc; s++) {
    for (int q = 0; q < ncol; q++) {
      const int8_t* wr = W + (size_t)(tn * bn + c + NCH * q) * k;
      int32_t acc = 0;
      for (int kk = s * Lc; kk < (s + 1) * Lc; kk++) acc += (int32_t)x[kk] * (int32_t)wr[kk];
      T[q] = (int32_t)((uint32_t)T[q] + (uint32_t)acc);
    }
    for (int q = 0; q < ncol; q++) {
      const int t = q % DUO_NPOS;
      const uint32_t u = st.y ^ (uint32_t)T[q] ^ duo_xor(t);
      st.y = u * duo_mul(t) + st.z;
      st.z = dc_rotl(u, duo_rot(t)) + st.y;
    }
  }
  for (int q = 0; q < ncol; q++) C[(size_t)r * n + tn * bn + c + NCH * q] = T[q];
  R[((size_t)r * tiles_n + tn) * NCH + c] = duo_finish(st);
}

extern "C" int kb_ref_rows(const void* X, const void* W, int n, int k, int bn, int Lc, unsigned seed, const void* rows,
                           int nrows, void* C, void* R, void* stream) {
  if (n % bn || bn % NCH || bn / NCH > 32 || k % Lc) return 2;
  const int total = nrows * (n / bn) * NCH;
  kb_ref_rows_k<<<(total + 127) / 128, 128, 0, (cudaStream_t)stream>>>((const int8_t*)X, (const int8_t*)W, n, k, bn, Lc, seed,
                                                                       (const int*)rows, nrows, (int32_t*)C, (uint32_t*)R);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

struct KbLt {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulHeuristicResult_t res[16];
  int n = 0;
  void* ws;
  size_t wsb;
};

extern "C" void* kb_lt_create(int m, int n, int k, void* ws, size_t wsb) {
  KbLt* t = new KbLt;
  t->ws = ws;
  t->wsb = wsb;
  cublasLtCreate(&t->h);
  cublasLtMatmulDescCreate(&t->op, CUBLAS_COMPUTE_32I, CUDA_R_32I);
  const cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(t->op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T);
  cublasLtMatmulDescSetAttribute(t->op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N);
  // column-major view: C^T (n x m) = W (k x n)^T X^T (k x m)
  cublasLtMatrixLayoutCreate(&t->a, CUDA_R_8I, k, n, k);
  cublasLtMatrixLayoutCreate(&t->b, CUDA_R_8I, k, m, k);
  cublasLtMatrixLayoutCreate(&t->c, CUDA_R_32I, n, m, n);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &t->wsb, sizeof t->wsb);
  if (cublasLtMatmulAlgoGetHeuristic(t->h, t->op, t->a, t->b, t->c, t->c, pref, 16, t->res, &t->n) != CUBLAS_STATUS_SUCCESS)
    t->n = 0;
  cublasLtMatmulPreferenceDestroy(pref);
  return t;
}

extern "C" int kb_lt_algos(void* h) { return ((KbLt*)h)->n; }

extern "C" int kb_lt_run(void* h, int a, const void* X, const void* W, void* C, void* stream) {
  KbLt* t = (KbLt*)h;
  if (a < 0 || a >= t->n) return 2;
  const int32_t one = 1, zero = 0;
  return cublasLtMatmul(t->h, t->op, &one, W, t->a, X, t->b, &zero, C, t->c, C, t->c, &t->res[a].algo, t->ws, t->wsb,
                        (cudaStream_t)stream) == CUBLAS_STATUS_SUCCESS ? 0 : 3;
}

extern "C" void kb_lt_destroy(void* h) {
  KbLt* t = (KbLt*)h;
  cublasLtMatrixLayoutDestroy(t->a);
  cublasLtMatrixLayoutDestroy(t->b);
  cublasLtMatrixLayoutDestroy(t->c);
  cublasLtMatmulDescDestroy(t->op);
  cublasLtDestroy(t->h);
  delete t;
}
