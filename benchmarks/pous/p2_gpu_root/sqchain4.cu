// sqchain3 with one warp per column pair. Same representation, carries, phase 3 and L2 exchange as sqchain3.cu; C >= 2 CTAs,
// one per SM, each owning `per` digits and computing its columns (owned plus 4 below) as pairs (k, k+1).
//   phase 1  warp w takes pair w; its iterations (the low run, then the high run: 315, or 316 for k = 631, 632) are
//            interleaved over the lanes (run_pair_il), with the products the pair's runs leave out in the last
//            iteration's free lanes
//   phase 2  a shuffle reduce-scatter: at offset 16 lanes 0..15 keep column k's two sums and 16..31 column k+1's, then
//            offsets 8..1; lane 0 (column k) and lane 16 (column k+1) each double, add the square, fold T = S_lo + c' S_hi and cut it into chunks
//   phase 3  one thread per 4 owned digits: the carry passes, then a 16-byte relaxed.gpu store of the tagged digits into
//            each of REPL replicas of the exchange vector
//   arrival  CTA g polls replica g % REPL (16-byte relaxed.gpu loads) until each word has this step's tag. With one copy
//            every CTA's polls land on the same 20 L2 lines, and arrival slowed as CTAs were added; with 8 a step at
//            80 CTAs took 1,143 ns against 1,432 (2 or 4 polls in flight per word made it slower still).
// The partial sums never touch shared memory, so a step has two __syncthreads (after phase 2 and after arrival).
//
// Timing: device clocks only (clock64 and %globaltimer on CTA 0, around the N steps).
// Build: nvcc -O3 -arch=sm_120 -std=c++17 [-DLB=512] [-DREPL=8] -o sqchain4 sqchain4.cu
// Run:   sqchain4 PER TPB N reps in.bin out.bin    (C = ceil(633 / PER) >= 2; PER a multiple of 4; TPB = 32 * ceil((PER + 4) / 2))
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#ifndef REPL
#define REPL 8
#endif
#ifndef LB
#define LB 512
#endif
static const int ND = 633;
static const int ND4 = 636;
static const uint32_t MASK = (1u << 26) - 1;
static const uint32_t VMASK = (1u << 27) - 1;
static const uint32_t CP = 21065u << 10;
static const int PADL = 64, PADR = 96;
static const int DLEN = PADL + ND4 + PADR;
static const int HALO = 4;
static const int MAXCOLS = 640;
static const int RSTR = 1056;                       // words between replicas of the exchange vector
static const int ZERO = ND;                          // cur[ZERO] is always 0

__host__ __device__ __forceinline__ int floordiv2(int x) { return x >= 0 ? x / 2 : -((1 - x) / 2); }
__host__ __device__ __forceinline__ int len_lo(int c) { return floordiv2(c - 1) + 1; }
__host__ __device__ __forceinline__ int me_hi(int k) { return floordiv2(k + ND - 1); }

__device__ __forceinline__ uint64_t madw(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r; asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c)); return r;
}
__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t shfl64(uint64_t v, int o) {
  uint32_t lo = __shfl_xor_sync(0xffffffffu, (uint32_t)v, o), hi = __shfl_xor_sync(0xffffffffu, (uint32_t)(v >> 32), o);
  return ((uint64_t)hi << 32) | lo;
}

// Interleaved: lane l takes iterations l, l+32, l+64, ..., so in each iteration the warp reads 32 consecutive digits and
// two descending runs of partners, each in one shared-memory wavefront (contiguous slices put lanes about 9.8 words apart
// and conflict 2 to 4 ways). T is 315 or 316, so iterations 0..8 are full and the 10th leaves 4 or 5 lanes free; those
// take the pair's (at most 3) left-out products, so its operands come from per-lane indices (ii9, ia9, ib9; ZERO reads
// a zero digit). The count is fixed and unrolled: a runtime-count loop measured 2 to 3 times slower (probe3).
__device__ __forceinline__ void run_pair_il(const uint32_t *cur, int k, int L0, int lane, int ii9, int ia9, int ib9, bool lo9,
                                            uint64_t &l0, uint64_t &l1, uint64_t &h0, uint64_t &h1) {
  const int hoff = k + 2 - L0, kh = k + ND;
  #pragma unroll
  for (int t = 0; t < 9; t++) {
    const int j = lane + 32 * t;
    const bool lo = j < L0;
    const int i = lo ? j : j + hoff, b = lo ? k : kh;
    const uint32_t di = cur[i], w0 = cur[b - i], w1 = cur[b + 1 - i];
    if (lo) { l0 = madw(di, w0, l0); l1 = madw(di, w1, l1); } else { h0 = madw(di, w0, h0); h1 = madw(di, w1, h1); }
  }
  const uint32_t di = cur[ii9], w0 = cur[ia9], w1 = cur[ib9];
  if (lo9) { l0 = madw(di, w0, l0); l1 = madw(di, w1, l1); } else { h0 = madw(di, w0, h0); h1 = madw(di, w1, h1); }
}

__global__ void __launch_bounds__(LB, 1) chain_kernel(const uint32_t *__restrict__ xin, uint32_t *__restrict__ xout, uint32_t *G,
                                                        int nsteps, int per, unsigned long long *timing) {
#ifdef PROF
  unsigned long long pacc[4] = {0, 0, 0, 0};
#endif
  extern __shared__ __align__(16) unsigned char smem[];
  const int tpb = blockDim.x, tid = threadIdx.x, g = blockIdx.x, w = tid >> 5, lane = tid & 31;
  uint32_t *D = (uint32_t *)smem;                                   // [2][DLEN]; digit x at PADL + x
  uint4 *T4 = (uint4 *)(smem + 2 * DLEN * 4);                       // [MAXCOLS]

  const int a = min(ND, g * per), nown = min(ND, a + per) - a;
  const int kstart = ((a - HALO) % ND + ND) % ND, ncols = nown + HALO;
  const int NP = (ncols + 1) / 2;
  for (int q = tid; q < DLEN; q += tpb) {
    int x = q - PADL;
    uint32_t v = (x >= 0 && x < ND) ? xin[x] : 0;
    D[q] = v; D[DLEN + q] = v;
  }
  const bool p1 = w < NP;
  const int k = (kstart + 2 * w) % ND;
  const int L0 = len_lo(k), T = L0 + max(0, me_hi(k) - (k + 1));        // 315, except 316 for k = 631, 632
  // lane 0 leads column k (= 2w), lane 16 column k + 1
  const bool upper = lane >= 16;
  const int cx = 2 * w + (upper ? 1 : 0);
  const bool lead = p1 && (lane & 15) == 0 && cx < ncols;
  const int clo = (kstart + cx) % ND, chi = clo + ND;
  const int sqi = (clo & 1) == 0 ? clo >> 1 : chi >> 1;
  // iteration 9: a regular iteration below T; above it, free lane 0 takes column k's left-out high product (i = k + 1),
  // free lane 1 column k+1's left-out low product, free lane 2 its left-out high product (each run covers the rest)
  int ii9 = ZERO, ia9 = ZERO, ib9 = ZERO;
  bool lo9 = false;
  {
    const int j = lane + 288, c1 = (k + 1) % ND;
    if (j < T) {
      lo9 = j < L0;
      const int i = lo9 ? j : j + k + 2 - L0, b = lo9 ? k : k + ND;
      ii9 = i; ia9 = b - i; ib9 = b + 1 - i;
    } else if (j - T == 0) {
      if (k + 1 < me_hi(k) + 1) { ii9 = k + 1; ia9 = k + ND - (k + 1); }
    } else if (j - T == 1) {
      const int x0 = len_lo(k), y0 = len_lo(c1);
      if (x0 < y0) { lo9 = true; ii9 = x0; ib9 = c1 - x0; }
    } else if (j - T == 2) {
      const int x2 = max(me_hi(k) + 1, c1 + 1), y2 = me_hi(c1) + 1;
      if (x2 < y2) { ii9 = x2; ib9 = c1 + ND - x2; }
    }
  }
  const int ngrp4 = (nown + 3) / 4, m0 = a + 4 * tid;
  const bool p3 = tid < ngrp4;
  const int lb = ((m0 - 4 - kstart) % ND + ND) % ND;
  uint32_t wmask = 0;
  for (int x = 0; x < 5; x++)
    for (int j = 0; j < 4; j++)
      if (m0 - 1 + x - j < 0 && !(m0 == 0 && x == 0)) wmask |= 1u << (4 * x + j);
  __syncthreads();

  uint64_t c0 = clock64(), g0 = gtime();
  #pragma unroll 1
  for (int step = 0; step < nsteps; step++) {
    const uint32_t *cur = D + (step & 1) * DLEN + PADL;
    uint32_t *nxt = D + ((step + 1) & 1) * DLEN + PADL;
#ifdef PROF
    unsigned long long q0 = clock64();
#endif
    if (p1) {
      uint64_t l0 = 0, l1 = 0, h0 = 0, h1 = 0;
      const uint32_t hsq = cur[sqi];
      run_pair_il(cur, k, L0, lane, ii9, ia9, ib9, lo9, l0, l1, h0, h1);
#ifdef PROF
      pacc[0] += clock64() - q0;
#endif
      uint64_t A0 = (upper ? l1 : l0) + shfl64(upper ? l0 : l1, 16);
      uint64_t A1 = (upper ? h1 : h0) + shfl64(upper ? h0 : h1, 16);
      #pragma unroll
      for (int o = 8; o > 0; o >>= 1) { A0 += shfl64(A0, o); A1 += shfl64(A1, o); }
      if (lead) {
        A0 *= 2; A1 *= 2;
        if ((clo & 1) == 0) A0 = madw(hsq, hsq, A0); else A1 = madw(hsq, hsq, A1);
        uint64_t m0_ = madw(CP, (uint32_t)A1, 0);
        uint64_t m1_ = madw(CP, (uint32_t)(A1 >> 32), 0);
        uint64_t lo = A0 + m0_;
        uint64_t hi = (lo < m0_) + (m1_ >> 32);
        uint64_t mid = m1_ << 32;
        lo += mid; hi += (lo < mid);
        T4[cx] = make_uint4((uint32_t)lo & MASK, (uint32_t)(lo >> 26) & MASK, (uint32_t)((lo >> 52) | (hi << 12)) & MASK,
                            (uint32_t)(hi >> 14));
      }
    }
    __syncthreads();
#ifdef PROF
    unsigned long long q2 = clock64(); pacc[1] += q2 - q0;
#endif
    const uint32_t tag = (uint32_t)(step % 31 + 1) << 27;
    uint32_t *Gn = G + ((step + 1) & 1) * REPL * RSTR;
    if (p3) {
      const uint32_t *tw = (const uint32_t *)(T4 + lb);
      uint64_t ep = 0;
      uint32_t f[4];
      #pragma unroll
      for (int x = 0; x < 5; x++) {
        uint64_t e = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) e = madw((wmask >> (4 * x + j)) & 1 ? CP : 1u, tw[4 * (x - j + 3) + j], e);
        if (x > 0) f[x - 1] = m0 + x - 1 < a + nown ? (uint32_t)((e & MASK) + (ep >> 26) * (m0 + x - 1 == 0 ? CP : 1)) : 0;
        ep = e;
      }
      #pragma unroll
      for (int r = 0; r < REPL; r++)
        asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(Gn + r * RSTR + m0), "r"(f[0] | tag), "r"(f[1] | tag), "r"(f[2] | tag), "r"(f[3] | tag) : "memory");
    }
#ifdef PROF
    unsigned long long q3 = clock64(); pacc[2] += q3 - q2;
#endif
    const uint32_t *Gr = Gn + (g % REPL) * RSTR;
    #pragma unroll 1
    for (int q = 4 * tid; q < ND4; q += 4 * tpb) {
      uint4 v;
      do {
        asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(Gr + q) : "memory");
      } while (((v.x & ~VMASK) != tag) | ((v.y & ~VMASK) != tag) | ((v.z & ~VMASK) != tag) | ((v.w & ~VMASK) != tag));
      *(uint4 *)&nxt[q] = make_uint4(v.x & VMASK, v.y & VMASK, v.z & VMASK, v.w & VMASK);
    }
    __syncthreads();
#ifdef PROF
    pacc[3] += clock64() - q3;
#endif
  }
  uint64_t c1 = clock64(), g1 = gtime();
  if (g == 0 && tid == 0) { timing[0] = c1 - c0; timing[1] = g1 - g0; }
#ifdef PROF
  if (tid == 0 && g < 18) for (int q = 0; q < 4; q++) timing[2 + 4 * g + q] = pacc[q];
#endif
  const uint32_t *fin = D + (nsteps & 1) * DLEN + PADL;
  if (g == 0) for (int q = tid; q < ND; q += tpb) xout[q] = fin[q];
}

int main(int argc, char **argv) {
  if (argc < 7) { fprintf(stderr, "usage: %s PER TPB N reps in.bin out.bin\n", argv[0]); return 2; }
  int per = atoi(argv[1]), tpb = atoi(argv[2]), N = atoi(argv[3]), reps = atoi(argv[4]);
  if (per % 4 || per < 4 || per >= ND) { fprintf(stderr, "PER must be a multiple of 4 below 633\n"); return 2; }
  const int C = (ND + per - 1) / per;
  const int NP = (per + HALO + 1) / 2;
  if (C < 2 || tpb != 32 * NP || tpb > LB) { fprintf(stderr, "bad configuration: per %d C %d tpb %d (want %d)\n", per, C, tpb, 32 * NP); return 2; }
  std::vector<uint32_t> x(ND);
  FILE *f = fopen(argv[5], "rb"); if (!f || fread(x.data(), 4, ND, f) != (size_t)ND) { fprintf(stderr, "bad in.bin\n"); return 2; } fclose(f);
  uint32_t *dx, *dy, *dG; unsigned long long *dtm;
  CK(cudaMalloc(&dx, ND * 4)); CK(cudaMalloc(&dy, ND * 4)); CK(cudaMalloc(&dG, 2 * REPL * RSTR * 4)); CK(cudaMalloc(&dtm, 8 * 80));
  CK(cudaMemset(dtm, 0, 8 * 80));
  CK(cudaMemcpy(dx, x.data(), ND * 4, cudaMemcpyHostToDevice));
  size_t sm = (size_t)2 * DLEN * 4 + MAXCOLS * 16;
  CK(cudaFuncSetAttribute(chain_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  int nb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, chain_kernel, tpb, sm));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  if (nb < 1 || C > p.multiProcessorCount) { fprintf(stderr, "C = %d CTAs cannot all be resident\n", C); return 2; }

  std::vector<double> ns, cyc; double prof[4] = {0, 0, 0, 0};
  for (int rep = 0; rep < reps + 1; rep++) {
    CK(cudaMemset(dG, 0, 2 * REPL * RSTR * 4));
    void *args[] = {&dx, &dy, &dG, &N, &per, &dtm};
    CK(cudaLaunchCooperativeKernel((void *)chain_kernel, dim3(C), dim3(tpb), args, sm, 0));
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned long long tm[80]; CK(cudaMemcpy(tm, dtm, 8 * 80, cudaMemcpyDeviceToHost));
    if (rep > 0) {
      cyc.push_back((double)tm[0] / N); ns.push_back((double)tm[1] / N);
      for (int q = 0; q < 4; q++) { prof[q] = 0; for (int gg = 0; gg < std::min(C, 18); gg++) prof[q] = std::max(prof[q], (double)tm[2 + 4 * gg + q] / N); }
    }
  }
  std::vector<uint32_t> y(ND);
  CK(cudaMemcpy(y.data(), dy, ND * 4, cudaMemcpyDeviceToHost));
  f = fopen(argv[6], "wb"); fwrite(y.data(), 4, ND, f); fclose(f);
  std::vector<double> sns = ns; std::sort(sns.begin(), sns.end());
  int imin = (int)(std::min_element(ns.begin(), ns.end()) - ns.begin());
#ifdef PROF
  { unsigned long long tm[80]; CK(cudaMemcpy(tm, dtm, 8 * 80, cudaMemcpyDeviceToHost));
    fprintf(stderr, "per-CTA phase cycles (last rep: phase-1 runs, through phase 2, phase 3, arrival):");
    for (int gg = 0; gg < std::min(C, 18); gg++) fprintf(stderr, " [%llu %llu %llu %llu]", tm[2 + 4 * gg] / N, tm[3 + 4 * gg] / N, tm[4 + 4 * gg] / N, tm[5 + 4 * gg] / N);
    fprintf(stderr, "\n"); }
#endif
  printf("{\"kernel\":\"warp-pairs-l2-r26\",\"mode\":\"l2-tagged-v4-poll\",\"replicas\":%d,\"prof_cyc_max_over_ctas\":[%.0f,%.0f,%.0f,%.0f],\"per\":%d,\"C\":%d,\"tpb\":%d,\"slices\":32,\"N\":%d,\"reps\":%d,"
         "\"ns_per_sq_min\":%.2f,\"ns_per_sq_med\":%.2f,\"cyc_per_sq_at_min\":%.1f,\"sm_clock_mhz_at_min\":%.1f,\"smem\":%zu,\"gpu\":\"%s\",\"sms\":%d}\n",
         REPL, prof[0], prof[1], prof[2], prof[3], per, C, tpb, N, reps, sns[0], sns[sns.size() / 2], cyc[imin], cyc[imin] / ns[imin] * 1000.0, sm,
         p.name, p.multiProcessorCount);
  return 0;
}
