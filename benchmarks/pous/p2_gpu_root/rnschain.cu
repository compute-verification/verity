// The squaring chain in a residue number system: x mod p held as its residues modulo K primes m_j < 2^26, with K large
// enough that J squarings need no reduction. A round is J dependent squarings of each residue (local), one all-to-all,
// and one reduction mod p by the explicit CRT:
//   xi_i = x_i (M/m_i)^-1 mod m_i,  alpha = floor(sum_i xi_i / m_i),
//   y = sum_i xi_i (M/m_i mod p) + alpha (p - M mod p)   (y = x mod p up to a multiple of p; y < K 2^26 p + K p)
//   y_j = sum_i xi_i T[j][i] + alpha e_j  (mod m_j)        T[j][i] = (M/m_i mod p) mod m_j,  e_j = (p - M mod p) mod m_j
// so a round costs one exchange per J squarings and K^2 multiply-adds.
//
// C CTAs, one per SM; CTA g owns rows j = gR .. gR+R-1. Warp 0's lanes r < R reduce mod m_j, square J times, and publish
// xi with a tag into REPL replicas of the exchange vector (L2, relaxed.gpu 16-byte stores). Every CTA polls its replica
// with one or two 16-byte words per thread, and each poller also sums the fixed-point fractions f_i = floor(xi_i 2^20 / m_i) of
// its word (within K 2^-20 of the true sum, which is alpha plus less than 2^-20 by the range slack). TPR threads per row
// (64, or 32 for wide rows) each hold 4 NC4 columns of the row's table in registers and multiply-add them against the gathered xi (shared memory),
// a shuffle reduction per warp and one add per row finish the sum, while one extra warp adds up the pollers' fractions
// into alpha.
//
// Timing: device clocks only (clock64 and %globaltimer on CTA 0, around the N / J rounds).
// Build: nvcc -O3 -arch=sm_120 -std=c++17 -DRR=7 -DNC4=6 -DJJ=1 [-DTPR=64] [-DREPL=8] [-DNSLEEP=0] -o rnschain rnschain.cu
// Run:   rnschain C N reps params.bin in.bin out.bin   (K = C RR; params.bin = m[K] inv[K] e[K] T[K][K], u32 little-endian)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#ifndef RR
#define RR 7
#endif
#ifndef NC4
#define NC4 6
#endif
#ifndef JJ
#define JJ 1
#endif
#ifndef REPL
#define REPL 8
#endif
#ifndef NSLEEP
#define NSLEEP 0
#endif
#ifndef TPR
#define TPR 64
#endif
static const int WPR = TPR / 32;
static const int TPB = RR * TPR + 32;
static const int KP = 4 * TPR * NC4;              // columns held per row, K padded
static const int V = (RR + 3) / 4;                // 16-byte words a CTA publishes
static const uint32_t VM = (1u << 26) - 1;
static_assert(V * REPL <= 32, "warp 0 publishes every replica");

__device__ __forceinline__ uint64_t madw(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r; asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c)); return r;
}
__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t shfl64(uint64_t v, int o) {
  uint32_t lo = __shfl_xor_sync(0xffffffffu, (uint32_t)v, o), hi = __shfl_xor_sync(0xffffffffu, (uint32_t)(v >> 32), o);
  return ((uint64_t)hi << 32) | lo;
}
// s mod m for any 64-bit s, with mu = floor((2^64 - 1) / m): the quotient estimate is at most 2 short
__device__ __forceinline__ uint32_t mod64(uint64_t s, uint32_t m, uint64_t mu) {
  uint64_t r = s - __umul64hi(s, mu) * m;
  if (r >= m) r -= m;
  if (r >= m) r -= m;
  return (uint32_t)r;
}

__global__ void __launch_bounds__(TPB, 1) rns_kernel(const uint32_t *__restrict__ prm, const uint32_t *__restrict__ yin,
                                                     uint32_t *__restrict__ yout, uint32_t *G, int nrounds, int K, int rstr,
                                                     unsigned long long *timing) {
#ifdef PROF
  unsigned long long pacc[3] = {0, 0, 0};
#endif
  __shared__ __align__(16) uint32_t X[KP];
  __shared__ uint32_t F[TPB];
  __shared__ uint64_t part[RR][WPR];
  __shared__ uint32_t alpha_s;
  __shared__ __align__(16) uint32_t stage[4 * V];
  const int tid = threadIdx.x, g = blockIdx.x, C = gridDim.x, w = tid >> 5, lane = tid & 31;
  const uint32_t *pm = prm, *pinv = prm + K, *pe = prm + 2 * K, *pT = prm + 3 * K;
  const bool roww = tid < RR * TPR;
  const int r = tid / TPR, s = tid % TPR;
  uint4 tab[NC4];
  #pragma unroll
  for (int k = 0; k < NC4; k++) {
    uint32_t v[4];
    #pragma unroll
    for (int q = 0; q < 4; q++) {
      const int col = 4 * (s + TPR * k) + q;
      v[q] = (roww && col < K) ? pT[(size_t)(g * RR + r) * K + col] : 0;
    }
    tab[k] = make_uint4(v[0], v[1], v[2], v[3]);
  }
  for (int q = tid; q < KP; q += TPB) X[q] = 0;
  // poller tid reads words tid and tid + TPB of the exchange vector: word e holds CTA e / V's residues from 4 (e % V)
  uint32_t pr20[2][4];
  int pe0[2];
  #pragma unroll
  for (int n = 0; n < 2; n++) {
    const int e = tid + n * TPB, pb = 4 * (e % V);
    pe0[n] = (e / V) * RR + pb;
    #pragma unroll
    for (int q = 0; q < 4; q++) pr20[n][q] = (e < C * V && pb + q < RR) ? (uint32_t)((1ull << 52) / pm[pe0[n] + q]) : 0;
    if (!(e < C * V)) pe0[n] = -1;
  }
  F[tid] = 0;
  // warp 0 lane l < RR carries row j's state
  const bool own = w == 0 && lane < RR;
  const int j = g * RR + lane;
  uint32_t mj = 1, invj = 0, ej = 0, y = 0;
  uint64_t muj = 0;
  if (own) {
    mj = pm[j]; invj = pinv[j]; ej = pe[j]; y = yin[j];
    muj = ~0ull / mj;
  }
  __syncthreads();

  uint64_t c0 = clock64(), g0 = gtime();
  #pragma unroll 1
  for (int u = 0;; u++) {
    const uint32_t tag = (uint32_t)(((u >> 1) + 1) & 63) << 26;
#ifdef PROF
    unsigned long long q0 = clock64();
#endif
    // publish round u: J squarings and xi, tagged
    if (w == 0) {
      if (own) {
        uint32_t x = y;
        #pragma unroll
        for (int t = 0; t < JJ; t++) x = mod64((uint64_t)x * x, mj, muj);
        stage[lane] = mod64((uint64_t)x * invj, mj, muj);
      } else if (lane < 4 * V) {
        stage[lane] = 0;
      }
      __syncwarp();
      if (lane < V * REPL) {
        const int rep = lane / V, v = lane % V;
        const uint4 a = *(const uint4 *)&stage[4 * v];
        uint32_t *dst = G + ((size_t)(u & 1) * REPL + rep) * rstr + 4 * (g * V + v);
        asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(dst), "r"(a.x | tag), "r"(a.y | tag), "r"(a.z | tag), "r"(a.w | tag) : "memory");
      }
      __syncwarp();
    }
#ifdef PROF
    unsigned long long q1 = clock64();
    if (tid == 0) pacc[0] += q1 - q0;
#endif
    // gather round u: one or two 16-byte words per poller, and their share of the fractions
    {
      const uint32_t *src = G + ((size_t)(u & 1) * REPL + g % REPL) * rstr + 4 * tid;
      uint4 v[2];
      bool ok[2] = {pe0[0] < 0, pe0[1] < 0};
      do {
        #pragma unroll
        for (int n = 0; n < 2; n++)
          if (!ok[n]) {
            asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(v[n].x), "=r"(v[n].y), "=r"(v[n].z), "=r"(v[n].w) : "l"(src + 4 * n * TPB) : "memory");
            ok[n] = ((v[n].x & ~VM) == tag) & ((v[n].y & ~VM) == tag) & ((v[n].z & ~VM) == tag) & ((v[n].w & ~VM) == tag);
          }
        if (NSLEEP && !(ok[0] & ok[1])) __nanosleep(NSLEEP);
      } while (!(ok[0] & ok[1]));
      uint32_t fs = 0;
      #pragma unroll
      for (int n = 0; n < 2; n++)
        if (pe0[n] >= 0) {
          const uint32_t vv[4] = {v[n].x & VM, v[n].y & VM, v[n].z & VM, v[n].w & VM};
          #pragma unroll
          for (int q = 0; q < 4; q++) {
            if (pr20[n][q]) X[pe0[n] + q] = vv[q];
            fs += (uint32_t)(((uint64_t)vv[q] * pr20[n][q]) >> 32);
          }
        }
      F[tid] = fs;
    }
#ifdef PROF
    if (tid == 0) pacc[1] += clock64() - q1;
#endif
    __syncthreads();
#ifdef PROF
    unsigned long long q2 = clock64();
#endif
    if (roww) {
      uint64_t acc[4] = {0, 0, 0, 0};
      #pragma unroll
      for (int k = 0; k < NC4; k++) {
        const uint4 xv = *(const uint4 *)&X[4 * (s + TPR * k)];
        acc[0] = madw(xv.x, tab[k].x, acc[0]); acc[1] = madw(xv.y, tab[k].y, acc[1]);
        acc[2] = madw(xv.z, tab[k].z, acc[2]); acc[3] = madw(xv.w, tab[k].w, acc[3]);
      }
      uint64_t t = (acc[0] + acc[1]) + (acc[2] + acc[3]);
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) t += shfl64(t, o);
      if (lane == 0) part[r][s >> 5] = t;
    } else {
      uint32_t fs = 0;
      #pragma unroll
      for (int q = lane; q < TPB; q += 32) fs += F[q];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) fs += __shfl_xor_sync(0xffffffffu, fs, o);
      if (lane == 0) alpha_s = (fs + (1u << 19)) >> 20;
    }
    __syncthreads();
#ifdef PROF
    if (tid == 0) pacc[2] += clock64() - q2;
#endif
    if (own) {
      uint64_t t = (uint64_t)alpha_s * ej;
      #pragma unroll
      for (int h = 0; h < WPR; h++) t += part[lane][h];
      y = mod64(t, mj, muj);
    }
    if (u + 1 == nrounds) break;
  }
  uint64_t c1 = clock64(), g1 = gtime();
  if (g == 0 && tid == 0) { timing[0] = c1 - c0; timing[1] = g1 - g0; }
#ifdef PROF
  if (tid == 0 && g < 16) for (int q = 0; q < 3; q++) timing[2 + 3 * g + q] = pacc[q];
#endif
  if (own) yout[j] = y;
}

int main(int argc, char **argv) {
  if (argc < 7) { fprintf(stderr, "usage: %s C N reps params.bin in.bin out.bin\n", argv[0]); return 2; }
  const int C = atoi(argv[1]), N = atoi(argv[2]), reps = atoi(argv[3]);
  const int K = C * RR;
  if (N % JJ || K > KP || C * V > 2 * TPB) { fprintf(stderr, "N must be a multiple of J = %d, K = %d at most %d, C V = %d at most %d\n", JJ, K, KP, C * V, 2 * TPB); return 2; }
  const int nrounds = N / JJ;
  std::vector<uint32_t> prm((size_t)3 * K + (size_t)K * K), yin(K);
  FILE *f = fopen(argv[4], "rb");
  if (!f || fread(prm.data(), 4, prm.size(), f) != prm.size()) { fprintf(stderr, "bad params.bin\n"); return 2; }
  fclose(f);
  f = fopen(argv[5], "rb");
  if (!f || fread(yin.data(), 4, K, f) != (size_t)K) { fprintf(stderr, "bad in.bin\n"); return 2; }
  fclose(f);
  const int rstr = ((4 * C * V + 31) / 32) * 32 + 32;
  uint32_t *dp, *dx, *dy, *dG; unsigned long long *dtm;
  CK(cudaMalloc(&dp, prm.size() * 4)); CK(cudaMalloc(&dx, K * 4)); CK(cudaMalloc(&dy, K * 4));
  CK(cudaMalloc(&dG, (size_t)2 * REPL * rstr * 4)); CK(cudaMalloc(&dtm, 8 * 64)); CK(cudaMemset(dtm, 0, 8 * 64));
  CK(cudaMemcpy(dp, prm.data(), prm.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dx, yin.data(), K * 4, cudaMemcpyHostToDevice));
  int nb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, rns_kernel, TPB, 0));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  if (nb < 1 || C > p.multiProcessorCount) { fprintf(stderr, "C = %d CTAs cannot all be resident\n", C); return 2; }
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, rns_kernel));
  std::vector<double> ns, cyc;
  for (int rep = 0; rep < reps + 1; rep++) {
    CK(cudaMemset(dG, 0, (size_t)2 * REPL * rstr * 4));
    int nr = nrounds, kk = K, rs = rstr;
    void *args[] = {&dp, &dx, &dy, &dG, &nr, &kk, &rs, &dtm};
    CK(cudaLaunchCooperativeKernel((void *)rns_kernel, dim3(C), dim3(TPB), args, 0, 0));
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned long long tm[2]; CK(cudaMemcpy(tm, dtm, 16, cudaMemcpyDeviceToHost));
    if (rep > 0) { cyc.push_back((double)tm[0] / N); ns.push_back((double)tm[1] / N); }
  }
#ifdef PROF
  { unsigned long long tm[64]; CK(cudaMemcpy(tm, dtm, 8 * 64, cudaMemcpyDeviceToHost));
    fprintf(stderr, "per-round cycles on CTAs 0..15 (publish, gather to S1, matvec to S2):");
    for (int gg = 0; gg < std::min(C, 16); gg++) fprintf(stderr, " [%llu %llu %llu]", tm[2 + 3 * gg] / nrounds, tm[3 + 3 * gg] / nrounds, tm[4 + 3 * gg] / nrounds);
    fprintf(stderr, "\n"); }
#endif
  std::vector<uint32_t> y(K);
  CK(cudaMemcpy(y.data(), dy, K * 4, cudaMemcpyDeviceToHost));
  f = fopen(argv[6], "wb"); fwrite(y.data(), 4, K, f); fclose(f);
  std::vector<double> sns = ns; std::sort(sns.begin(), sns.end());
  const int imin = (int)(std::min_element(ns.begin(), ns.end()) - ns.begin());
  printf("{\"kernel\":\"rns-explicit-crt\",\"mode\":\"l2-tagged-v4-poll\",\"replicas\":%d,\"J\":%d,\"R\":%d,\"K\":%d,\"C\":%d,\"tpb\":%d,"
         "\"regs\":%d,\"local_bytes\":%zu,\"N\":%d,\"reps\":%d,\"ns_per_sq_min\":%.2f,\"ns_per_sq_med\":%.2f,\"cyc_per_sq_at_min\":%.1f,"
         "\"cyc_per_round_at_min\":%.1f,\"sm_clock_mhz_at_min\":%.1f,\"gpu\":\"%s\",\"sms\":%d}\n",
         REPL, JJ, RR, K, C, TPB, fa.numRegs, fa.localSizeBytes, N, reps, sns[0], sns[sns.size() / 2], cyc[imin], cyc[imin] * JJ,
         cyc[imin] / ns[imin] * 1000.0, p.name, p.multiProcessorCount);
  return 0;
}
