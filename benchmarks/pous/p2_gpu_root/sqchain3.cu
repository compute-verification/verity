// sqchain with column pairs and an L2 exchange. Same representation and carries as sqchain.cu (radix 2^26, 633 digits,
// c' = 21065 * 2^10, T_k = S_k + c' S_{k+633} cut into four chunks, two carry passes); C CTAs, one per SM, each owning
// `per` digits.
//   phase 1  the CTA's columns (owned plus 4 below) form pairs (k, k+1). A pair's products with i < j split into a low run
//            (i in [0, len_lo(k)), partners k - i and k+1 - i) and a high run (i in [k+2, me_hi(k)], partners k+633 - i and
//            k+634 - i), 315 iterations whatever k is; thread t takes pair t % NP and the (t / NP)-th of S equal slices of
//            both runs, with the two partners in a 2-register window, and leaves 4 partial sums in shared memory
//   phase 2  P threads per column sum its S partials (then a shuffle), and the first adds the products the pair's runs
//            leave out (at most one per half), doubles, adds the square, folds T = S_lo + c' S_hi and cuts it into chunks
//   phase 3  one thread per 4 owned digits: the carry passes, then one 16-byte relaxed.gpu store of the digits, tagged
//            with the step (top 5 bits; tags 1..31), into a double-buffered global vector
//   arrival  every CTA polls the whole vector (16-byte relaxed.gpu loads, served by L2) until each word has this step's tag,
//            and keeps the stripped digits in shared memory
// For C = 1 phase 3 writes shared memory and a __syncthreads ends the step. All CTAs must be resident (one per SM): the
// launch is cooperative, which refuses otherwise.
//
// Timing: device clocks only (clock64 and %globaltimer on CTA 0, around the N steps).
// Build: nvcc -O3 -arch=sm_120 -std=c++17 -o sqchain3 sqchain3.cu
// Run:   sqchain3 PER TPB N reps in.bin out.bin [P]     (C = ceil(633 / PER); PER a multiple of 4)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#ifndef LB
#define LB 1024
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

__host__ __device__ __forceinline__ int floordiv2(int x) { return x >= 0 ? x / 2 : -((1 - x) / 2); }
__host__ __device__ __forceinline__ int len_lo(int c) { return floordiv2(c - 1) + 1; }     // i in [0, len_lo): i < c - i
__host__ __device__ __forceinline__ int me_hi(int k) { return floordiv2(k + ND - 1); }     // high column k + 633: i in [k+1, me_hi]

__device__ __forceinline__ uint64_t madw(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r; asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c)); return r;
}
__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t shfl64(uint64_t v, int o) {
  uint32_t lo = __shfl_xor_sync(0xffffffffu, (uint32_t)v, o), hi = __shfl_xor_sync(0xffffffffu, (uint32_t)(v >> 32), o);
  return ((uint64_t)hi << 32) | lo;
}

// a0 += sum d_i d_{base-i}, a1 += sum d_i d_{base+1-i}, i in [i0, i1)
__device__ __forceinline__ void run2(const uint32_t *cur, int base, int i0, int i1, uint64_t &a0, uint64_t &a1) {
  uint32_t w1 = cur[base + 1 - i0];
  #pragma unroll 1
  for (int i = i0; i < i1; i++) {
    const uint32_t w0 = cur[base - i], di = cur[i];
    a0 = madw(di, w0, a0);
    a1 = madw(di, w1, a1);
    w1 = w0;
  }
}

// Every loop in the step stays rolled and every index is computed before the step loop: a step runs each of its
// instructions about once, so its cost is close to its code size in instruction-cache lines, and the unrolled version of
// this kernel spent most of its step refetching code.
template <bool MULTI>
__global__ void __launch_bounds__(LB, 1) chain_kernel(const uint32_t *__restrict__ xin, uint32_t *__restrict__ xout, uint32_t *G,
                                                      int nsteps, int per, int P, int redcap, unsigned long long *timing) {
#ifdef PROF
  unsigned long long pacc[4] = {0, 0, 0, 0};
#endif
  extern __shared__ __align__(16) unsigned char smem[];
  const int tpb = blockDim.x, tid = threadIdx.x, g = blockIdx.x;
  uint32_t *D = (uint32_t *)smem;                                   // [2][DLEN]; digit x at PADL + x
  uint4 *T4 = (uint4 *)(smem + 2 * DLEN * 4);                       // [MAXCOLS]
  uint64_t *RED = (uint64_t *)(T4 + MAXCOLS);                       // [S][NP][4]

  const int a = min(ND, g * per), nown = min(ND, a + per) - a;
  const int halo = MULTI ? HALO : 0;
  const int kstart = ((a - halo) % ND + ND) % ND, ncols = nown + halo;
  const int NP = (ncols + 1) / 2, S = min(tpb / NP, redcap / NP);   // a CTA with fewer columns has more slices
  for (int q = tid; q < DLEN; q += tpb) {
    int x = q - PADL;
    uint32_t v = (x >= 0 && x < ND) ? xin[x] : 0;
    D[q] = v; D[DLEN + q] = v;
  }
  // phase 1: this thread's pair and slices
  const int pr = tid % NP, sl = tid / NP;
  const bool p1 = sl < S;
  const int k = (kstart + 2 * pr) % ND;
  const int L0 = len_lo(k), L1 = max(0, me_hi(k) - (k + 1));
  const int lo_s = (int)((long)sl * L0 / S), lo_e = (int)((long)(sl + 1) * L0 / S);
  const int hi_s = k + 2 + (int)((long)sl * L1 / S), hi_e = k + 2 + (int)((long)(sl + 1) * L1 / S);
  uint64_t *const rp1 = RED + ((size_t)sl * NP + pr) * 4;
  // phase 2: column cx = tid / P, part tid % P; the products its pair's runs leave out (at most one per range) and its square
  const int cx = tid / P, part = tid % P;
  const bool p2 = cx < ncols, lead = p2 && part == 0;
  const int ck = (kstart + cx) % ND, cpk = (kstart + (cx & ~1)) % ND;
  const uint64_t *const rp2 = RED + (size_t)(cx >> 1) * 4 + (cx & 1);
  const int nred = p2 ? (S - part + P - 1) / P : 0, rstride = P * NP * 4;
  const int clo = ck, chi = ck + ND;
  const int x0 = len_lo(cpk), y0 = len_lo(clo);                                   // low: i in [x0, y0)
  const int x1 = clo + 1, y1 = min(cpk + 2, me_hi(clo) + 1);                      // high: i in [x1, y1)
  const int x2 = max(me_hi(cpk) + 1, clo + 1), y2 = me_hi(clo) + 1;               // high: i in [x2, y2)
  const int sqi = (clo & 1) == 0 ? clo >> 1 : chi >> 1;
  // phase 3: digits m0..m0+3; column m0 - 4 + q is local lb + q; bit 4x + j of wmask: weight c' for e_{m0-1+x}'s chunk j
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
      run2(cur, k, lo_s, lo_e, l0, l1);
      run2(cur, k + ND, hi_s, hi_e, h0, h1);
      rp1[0] = l0; rp1[1] = l1; rp1[2] = h0; rp1[3] = h1;
    }
    __syncthreads();
#ifdef PROF
    unsigned long long q1 = clock64(); pacc[0] += q1 - q0;
#endif
    {
      uint64_t A0 = 0, A1 = 0;
      const uint64_t *rp = rp2 + (size_t)part * NP * 4;
      #pragma unroll 1
      for (int s = 0; s < nred; s++, rp += rstride) { A0 += rp[0]; A1 += rp[2]; }
      #pragma unroll 1
      for (int o = P / 2; o > 0; o >>= 1) { A0 += shfl64(A0, o); A1 += shfl64(A1, o); }
      if (lead) {
        if (x0 < y0) A0 = madw(cur[x0], cur[clo - x0], A0);
        if (x1 < y1) A1 = madw(cur[x1], cur[chi - x1], A1);
        if (x2 < y2) A1 = madw(cur[x2], cur[chi - x2], A1);
        A0 *= 2; A1 *= 2;
        const uint32_t h = cur[sqi];
        if ((clo & 1) == 0) A0 = madw(h, h, A0); else A1 = madw(h, h, A1);
        uint64_t m0_ = madw(CP, (uint32_t)A1, 0);
        uint64_t m1_ = madw(CP, (uint32_t)(A1 >> 32), 0);
        uint64_t lo = A0 + m0_;
        uint64_t hi = (lo < m0_) + (m1_ >> 32);
        uint64_t mid = m1_ << 32;
        lo += mid; hi += (lo < mid);
        const uint4 t = make_uint4((uint32_t)lo & MASK, (uint32_t)(lo >> 26) & MASK, (uint32_t)((lo >> 52) | (hi << 12)) & MASK,
                                   (uint32_t)(hi >> 14));
        T4[cx] = t;
        if (!MULTI && cx < 4) T4[ND + cx] = t;          // one CTA: columns 0..3 again after 632, so phase 3's 8 are consecutive
      }
    }
    __syncthreads();
#ifdef PROF
    unsigned long long q2 = clock64(); pacc[1] += q2 - q1;
#endif
    const uint32_t tag = (uint32_t)(step % 31 + 1) << 27;
    uint32_t *Gn = G + ((step + 1) & 1) * ND4;
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
      if (!MULTI) *(uint4 *)&nxt[m0] = make_uint4(f[0], f[1], f[2], f[3]);
      else asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1, %2, %3, %4};"
                        :: "l"(Gn + m0), "r"(f[0] | tag), "r"(f[1] | tag), "r"(f[2] | tag), "r"(f[3] | tag) : "memory");
    }
#ifdef PROF
    unsigned long long q3 = clock64(); pacc[2] += q3 - q2;
#endif
    if (MULTI) {
      #pragma unroll 1
      for (int q = 4 * tid; q < ND4; q += 4 * tpb) {
        uint4 v;
        do {
          asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(Gn + q) : "memory");
        } while (((v.x & ~VMASK) != tag) | ((v.y & ~VMASK) != tag) | ((v.z & ~VMASK) != tag) | ((v.w & ~VMASK) != tag));
        *(uint4 *)&nxt[q] = make_uint4(v.x & VMASK, v.y & VMASK, v.z & VMASK, v.w & VMASK);
      }
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
  if (argc < 7) { fprintf(stderr, "usage: %s PER TPB N reps in.bin out.bin [P]\n", argv[0]); return 2; }
  int per = atoi(argv[1]), tpb = atoi(argv[2]), N = atoi(argv[3]), reps = atoi(argv[4]);
  int P = argc > 7 ? atoi(argv[7]) : 4;
  if (per % 4 || per < 4) { fprintf(stderr, "PER must be a positive multiple of 4\n"); return 2; }
  if (per >= ND) per = ND;
  const int C = (ND + per - 1) / per;
  const int ncols_max = std::min(ND, per) + (C > 1 ? HALO : 0);
  if (tpb > LB || tpb % 32 || P < 1 || P > 32 || (P & (P - 1)) || ncols_max * P > tpb || ncols_max > MAXCOLS ||
      tpb < (ncols_max + 1) / 2) {
    fprintf(stderr, "bad configuration: per %d C %d tpb %d P %d\n", per, C, tpb, P); return 2;
  }
  std::vector<uint32_t> x(ND);
  FILE *f = fopen(argv[5], "rb"); if (!f || fread(x.data(), 4, ND, f) != (size_t)ND) { fprintf(stderr, "bad in.bin\n"); return 2; } fclose(f);
  uint32_t *dx, *dy, *dG; unsigned long long *dtm;
  CK(cudaMalloc(&dx, ND * 4)); CK(cudaMalloc(&dy, ND * 4)); CK(cudaMalloc(&dG, 2 * ND4 * 4)); CK(cudaMalloc(&dtm, 8 * 80));
  CK(cudaMemset(dtm, 0, 8 * 80));
  CK(cudaMemcpy(dx, x.data(), ND * 4, cudaMemcpyHostToDevice));
  const int NP = (ncols_max + 1) / 2, S = tpb / NP;
  size_t sm = (size_t)2 * DLEN * 4 + MAXCOLS * 16 + (size_t)S * NP * 4 * 8;
  auto kern = C > 1 ? chain_kernel<true> : chain_kernel<false>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  int nb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, tpb, sm));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  if (nb < 1 || C > p.multiProcessorCount) { fprintf(stderr, "C = %d CTAs cannot all be resident\n", C); return 2; }

  std::vector<double> ns, cyc; double prof[4] = {0, 0, 0, 0};
  for (int rep = 0; rep < reps + 1; rep++) {
    CK(cudaMemset(dG, 0, 2 * ND4 * 4));
    int redcap = S * NP;
    void *args[] = {&dx, &dy, &dG, &N, &per, &P, &redcap, &dtm};
    CK(cudaLaunchCooperativeKernel((void *)kern, dim3(C), dim3(tpb), args, sm, 0));
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
    fprintf(stderr, "per-CTA phase cycles (last rep):");
    for (int gg = 0; gg < std::min(C, 18); gg++) fprintf(stderr, " [%llu %llu %llu %llu]", tm[2 + 4 * gg] / N, tm[3 + 4 * gg] / N, tm[4 + 4 * gg] / N, tm[5 + 4 * gg] / N);
    fprintf(stderr, "\n"); }
#endif
  printf("{\"kernel\":\"pairs-l2-r26\",\"mode\":\"%s\",\"prof_cyc_max_over_ctas\":[%.0f,%.0f,%.0f,%.0f],\"per\":%d,\"C\":%d,\"tpb\":%d,\"P\":%d,\"slices\":%d,\"N\":%d,\"reps\":%d,"
         "\"ns_per_sq_min\":%.2f,\"ns_per_sq_med\":%.2f,\"cyc_per_sq_at_min\":%.1f,\"sm_clock_mhz_at_min\":%.1f,\"smem\":%zu,\"gpu\":\"%s\",\"sms\":%d}\n",
         C > 1 ? "l2-tagged-v4-poll" : "single-cta", prof[0], prof[1], prof[2], prof[3], per, C, tpb, P, S, N, reps, sns[0], sns[sns.size() / 2],
         cyc[imin], cyc[imin] / ns[imin] * 1000.0, sm, p.name, p.multiProcessorCount);
  return 0;
}
