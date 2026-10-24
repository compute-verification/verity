// kbench's pgemm (sm_100a only): compute accounting's pipelined honest GEMM (benchmarks/pouw/dc/pgemm.cu, included
// unchanged) as a plain launch. Its operands are resident: it reads the first 128 bytes of K of each row (ldx apart) and
// computes the GEMM of the K-periodic inputs those bytes repeat to, so hb_kbench.py feeds and gates it K-periodic inputs.
//
//   kb_pg_prepare()                                         dynamic shared memory for every configuration
//   kb_pg_launch(nb, lw, mode, mi, X, W, ldx, m, n, k, lm, seed, C, R, clk, stream)
//                                                           one launch, grid (n / 512, m / 128); clk holds 3 words a CTA
#define main pgemm_main
#include "../../pouw/dc/pgemm.cu"
#undef main

extern "C" int kb_pg_prepare() {
  prepare();
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int kb_pg_launch(int nb, int lw, int mode, int mi, const void* X, const void* W, int ldx, int m, int n, int k,
                            int lm, unsigned seed, void* C, void* R, void* clk, void* stream) {
  const Cfg* cfg = find(nb, lw, mode, mi);
  if (!cfg || m % BM || n % CTA_N || k % (32 * lm) || k % PER) return 2;
  cfg->k<<<dim3(n / CTA_N, m / BM), cfg->threads(), SMEM, (cudaStream_t)stream>>>(
      (const int8_t*)X, (const int8_t*)W, ldx, n, k, lm, seed, (int32_t*)C, (uint32_t*)R, (long long*)clk);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
