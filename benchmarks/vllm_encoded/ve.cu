// The word codec's fused GEMM as a vLLM linear (`kernels.py`): `wc.cu` (benchmarks/word_codec, built with -I there) plus the
// launches a served linear needs, with nothing host-side inside a launch, so a CUDA graph captures it whole.
//
//   ve_gemm      the three forms of `wc.cu`'s GEMM (one task a warp, persistent, warp-specialized) over plain fragments
//                (k = 0) or codec words (k >= 1), each 512-column chunk's sums written to part[kc][b][n]
//   ve_reduce    out[b][n] = bf16((sum over kc in order of part[kc][b][n]) * xs[b] * ws): deterministic, so a plain and an
//                encoded linear over the same weight give the same bits
//   ve_prepare   the warp-specialized form's shared-memory attribute, set once outside any capture
//   ve_occupancy CTAs a form fits on one SM, for the caller to size the persistent grids
#include "wc.cu"

#include <cuda_bf16.h>

namespace ve {

__global__ void __launch_bounds__(256) reduce_kernel(const float* __restrict__ part, int nch, int n, int mn,
                                                     const float* __restrict__ xs, int xs_rows,
                                                     const float* __restrict__ ws, __nv_bfloat16* __restrict__ out) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= mn) return;
  float4 s = *reinterpret_cast<const float4*>(part + i);
  for (int c = 1; c < nch; ++c) {
    const float4 v = *reinterpret_cast<const float4*>(part + (size_t)c * mn + i);
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  const float sc = xs[xs_rows ? i / n : 0] * ws[0];
  __nv_bfloat162 a = __floats2bfloat162_rn(s.x * sc, s.y * sc), b = __floats2bfloat162_rn(s.z * sc, s.w * sc);
  *reinterpret_cast<__nv_bfloat162*>(out + i) = a;
  *reinterpret_cast<__nv_bfloat162*>(out + i + 2) = b;
}

}  // namespace ve

extern "C" {

int ve_version() { return 1; }

int ve_constants(int* out) { return wc_gemm_constants(out); }

#define VE_NB(M) ((M) <= 8 ? 1 : (M) <= 16 ? 2 : 4)

int ve_prepare(int k, int stages) {
  if (k < 0 || k > WC_KMAX || stages < 1 || stages > 32) return 1;
#define P(KR, NB) do { \
    if (cudaFuncSetAttribute(gemm_ws_kernel<KR, NB, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                             ws_smem_bytes<KR, NB>(stages))) return 2; \
    if (cudaFuncSetAttribute(gemm_ws_kernel<KR, NB, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                             ws_smem_bytes<KR, NB>(stages))) return 2; } while (0)
#define PK(KR) do { P(KR, 1); P(KR, 2); P(KR, 4); } while (0)
  switch (k) {
    case 0: PK(0); break;
    WC_CASES(PK)
    default: return 1;
  }
#undef PK
#undef P
  return 0;
}

// form 0 simple, 1 persistent, 2 warp-specialized; CTAs an SM holds at batch M, or a negative error.
int ve_occupancy(int form, int k, int M, int consumers, int stages) {
  if (k < 0 || k > WC_KMAX || M < 1 || M > 32 || (form == 2 && stages != consumers)) return -1;
  const int nb8 = VE_NB(M);
  int per_sm = 0;
  cudaError_t e = cudaSuccess;
#define O(KR, NB) do { \
    if (form == 0) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm_kernel<KR, NB, true>, GEMM_WARPS * 32, 0); \
    else if (form == 1) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm_pipe_kernel<KR, NB, true>, \
                                                                          GEMM_WARPS * 32, 0); \
    else e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm_ws_kernel<KR, NB, true>, (consumers + 1) * 32, \
                                                           ws_smem_bytes<KR, NB>(stages)); } while (0)
#define OK(KR) do { if (nb8 == 1) O(KR, 1); else if (nb8 == 2) O(KR, 2); else O(KR, 4); } while (0)
  switch (k) {
    case 0: OK(0); break;
    WC_CASES(OK)
    default: return -1;
  }
#undef OK
#undef O
  return e ? -2 : per_sm;
}

// partials 1: part is nchunks x M x N, every element written; 0: part is M x N, zeroed by the caller, the chunks added
// atomically (not deterministic: for timing the epilogue only). groups: the persistent forms' CTA groups per chunk.
int ve_gemm(int form, int k, const void* w, const uint8_t* x, float* part, int M, int N, int Kd, uint64_t base,
            int partials, int groups, int consumers, int stages, cudaStream_t s) {
  if (M < 1 || M > 32 || N % 16 || Kd % TASK_COLS || k < 0 || k > WC_KMAX) return 1;
  if (form != 2 && N % (16 * GEMM_WARPS)) return 1;
  // stages != consumers: a stage has two consumers, and one can wait on a fill two phases ahead of the stage's barrier,
  // whose parity an earlier phase already satisfies; the kernel then reads a stale stage and, on the sm_120 card, faults
  if (form == 2 && (consumers < 1 || consumers > WS_MAX_CONSUMERS || stages != consumers)) return 1;
  if (form != 0 && groups < 1) return 1;
  const int nb8 = VE_NB(M), nchunks = Kd / TASK_COLS, nbs = N / 16;
#define L0(KR, NB, PT) gemm_kernel<KR, NB, PT><<<(nbs / GEMM_WARPS) * nchunks, GEMM_WARPS * 32, 0, s>>>( \
      w, x, part, M, N, Kd, base)
#define L1(KR, NB, PT) gemm_pipe_kernel<KR, NB, PT><<<(groups > nbs / GEMM_WARPS ? nbs / GEMM_WARPS : groups) * nchunks, \
                                                      GEMM_WARPS * 32, 0, s>>>(w, x, part, M, N, Kd, base, \
      groups > nbs / GEMM_WARPS ? nbs / GEMM_WARPS : groups)
#define L2(KR, NB, PT) gemm_ws_kernel<KR, NB, PT><<<(groups > nbs ? nbs : groups) * nchunks, (consumers + 1) * 32, \
                                                    ws_smem_bytes<KR, NB>(stages), s>>>(w, x, part, M, N, Kd, base, \
      groups > nbs ? nbs : groups, consumers, stages)
#define G(KR, NB) do { \
    if (form == 0) { if (partials) L0(KR, NB, true); else L0(KR, NB, false); } \
    else if (form == 1) { if (partials) L1(KR, NB, true); else L1(KR, NB, false); } \
    else { if (partials) L2(KR, NB, true); else L2(KR, NB, false); } } while (0)
#define GK(KR) do { if (nb8 == 1) G(KR, 1); else if (nb8 == 2) G(KR, 2); else G(KR, 4); } while (0)
  switch (k) {
    case 0: GK(0); break;
    WC_CASES(GK)
    default: return 1;
  }
#undef GK
#undef G
#undef L2
#undef L1
#undef L0
  return cudaGetLastError() ? 4 : 0;
}

// xs_rows 0: one activation scale (per tensor); 1: one a row (per token, vLLM's dynamic per-token FP8).
int ve_reduce(const float* part, int nch, int M, int N, const float* xs, int xs_rows, const float* ws, void* out,
              cudaStream_t s) {
  if (M < 1 || N % 4 || nch < 1) return 1;
  const int mn = M * N, threads = 256, grid = (mn / 4 + threads - 1) / threads;
  ve::reduce_kernel<<<grid, threads, 0, s>>>(part, nch, N, mn, xs, xs_rows, ws, static_cast<__nv_bfloat16*>(out));
  return cudaGetLastError() ? 4 : 0;
}

}  // extern "C"
