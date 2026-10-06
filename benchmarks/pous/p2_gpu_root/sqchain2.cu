// sqchain with warp-level column blocks: the same representation and carries as sqchain.cu (radix 2^26, 633 digits,
// c' = 21065 * 2^10, T_k = S_k + c' S_{k+633} cut into four chunks, two carry passes), organised to cut the per-step
// synchronisation that dominated sqchain.cu:
//   phase 1  each warp group (WPB warps) takes a block of R consecutive columns k0.., both halves (k0+r and k0+633+r); lane
//            l runs a contiguous, odd-length run of i in each half, with the partners d_{s0+r-i} in a register window that
//            shifts one digit per i, then the warp reduce-scatters its 2R sums with shuffles and leaves one copy per warp
//   phase 2  one thread per column: sums its WPB warp partials, adds the tail past the block's diagonal and the square,
//            folds T = S_lo + c' S_hi and cuts it into chunks
//   phase 3  one thread per (4-digit group, destination CTA): the carry passes for its 4 digits and one 16-byte store of
//            them into that CTA's next buffer
//   arrival  digits carry a step tag in their top 5 bits (a digit needs 27; tags 1..31, as a stripped word reads 0); every
//            CTA polls its own next buffer until each word has this step's tag, strips the tags, and starts the next step
//            (no cluster barrier, no mbarrier)
// For C = 1 phase 3 writes the local buffer and a __syncthreads ends the step.
//
// Timing: device clocks only (clock64 and %globaltimer on CTA 0, around the N steps).
// Build: nvcc -O3 -arch=sm_120 -std=c++17 -DR=4 -o sqchain2 sqchain2.cu
// Run:   sqchain2 C TPB N reps in.bin out.bin [WPB]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#ifndef R
#define R 4
#endif
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
static const int MAXBLK = 2 * (MAXCOLS / R + 2);
static const int NMETA = 10;

struct Blk { int k0, me_lo, l_lo, me_hi, l_hi; };     // columns k0..k0+R-1; lo i in [0, me_lo], hi i in [k0+1, me_hi]; run lengths
struct Col { int k, blk, r; };

struct Plan { int a, nown, kstart, ncols, nblk; std::vector<Blk> blk; std::vector<Col> cols; };

static inline int floordiv2(int x) { return x >= 0 ? x / 2 : -((1 - x) / 2); }
static inline int oddize(int x) { return x == 0 ? 0 : (x | 1); }

static Plan plan_cta(int g, int C, int lanes) {
  Plan P;
  int per = ((ND + C - 1) / C + 3) / 4 * 4;
  P.a = std::min(ND, g * per);
  int b = std::min(ND, P.a + per);
  P.nown = b - P.a;
  int halo = C == 1 ? 0 : HALO;
  P.kstart = ((P.a - halo) % ND + ND) % ND;
  P.ncols = P.nown + halo;
  std::vector<std::pair<int, int>> segs;
  if (P.kstart + P.ncols <= ND) segs.push_back({P.kstart, P.ncols});
  else { segs.push_back({P.kstart, ND - P.kstart}); segs.push_back({0, P.ncols - (ND - P.kstart)}); }
  for (auto &sg : segs)
    for (int k0 = sg.first; k0 < sg.first + sg.second; k0 += R) {
      Blk x;
      x.k0 = k0;
      x.me_lo = floordiv2(k0 - 1);
      x.me_hi = floordiv2(k0 + ND - 1);
      int len_lo = std::max(0, x.me_lo + 1), len_hi = std::max(0, x.me_hi - k0);
      x.l_lo = oddize((len_lo + lanes - 1) / lanes);
      x.l_hi = oddize((len_hi + lanes - 1) / lanes);
      int blk = (int)P.blk.size();
      P.blk.push_back(x);
      for (int r = 0; r < R && k0 + r < sg.first + sg.second; r++) P.cols.push_back({k0 + r, blk, r});
    }
  P.nblk = (int)P.blk.size();
  if (P.ncols > MAXCOLS || P.nblk > MAXBLK) { fprintf(stderr, "plan does not fit\n"); exit(2); }
  return P;
}

struct Dev { const Blk *blk; const Col *cols; const int *meta; };   // meta: a nown kstart ncols nblk boff coff

__device__ __forceinline__ uint64_t madw(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r; asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c)); return r;
}
__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
__device__ __forceinline__ uint64_t shfl64(uint64_t v, int o) {
  uint32_t lo = __shfl_xor_sync(0xffffffffu, (uint32_t)v, o), hi = __shfl_xor_sync(0xffffffffu, (uint32_t)(v >> 32), o);
  return ((uint64_t)hi << 32) | lo;
}

// sum of d_i d_{s0+r-i} for i in [i0, i0+n), r < R, into acc
__device__ __forceinline__ void run_half(const uint32_t *cur, int s0, int i0, int n, uint64_t *acc) {
  uint32_t W[R];
  #pragma unroll
  for (int r = 0; r < R; r++) W[r] = cur[s0 + r - i0];
  #pragma unroll 4
  for (int u = 0; u < n; u++) {
    const int i = i0 + u;
    const uint32_t di = cur[i];
    #pragma unroll
    for (int r = 0; r < R; r++) acc[r] = madw(di, W[r], acc[r]);
    #pragma unroll
    for (int r = R - 1; r > 0; r--) W[r] = W[r - 1];
    W[0] = cur[s0 - i - 1];
  }
}

template <int C>
__global__ void __launch_bounds__(LB, 1) chain_kernel(Dev dv, const uint32_t *__restrict__ xin, uint32_t *__restrict__ xout,
                                                      int nsteps, int wpb, int maxblk, unsigned long long *timing) {
#ifdef PROF
  unsigned long long pacc[4] = {0, 0, 0, 0};
#endif
  extern __shared__ __align__(16) unsigned char smem[];
  const int tpb = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = tpb >> 5;
  uint32_t *D = (uint32_t *)smem;                                   // [2][DLEN]; digit x of a buffer at PADL + x
  uint64_t *RED = (uint64_t *)(smem + 2 * DLEN * 4);                // [maxblk][wpb][2R]
  uint4 *T4 = (uint4 *)(RED + (size_t)maxblk * wpb * 2 * R);        // [MAXCOLS]
  Blk *BK = (Blk *)(T4 + MAXCOLS);                                  // [MAXBLK]

  const int g = C == 1 ? 0 : (int)cg::this_cluster().block_rank();
  const int *mt = dv.meta + NMETA * g;
  const int a = mt[0], nown = mt[1], kstart = mt[2], ncols = mt[3], nblk = mt[4];
  const Col *cl = dv.cols + mt[6];
  for (int q = tid; q < nblk; q += tpb) BK[q] = dv.blk[mt[5] + q];
  for (int q = tid; q < DLEN; q += tpb) {
    int x = q - PADL;
    uint32_t v = (x >= 0 && x < ND) ? xin[x] : 0;
    D[q] = v; D[DLEN + q] = v;
  }
  const int ngroups = nwarps / wpb, wsub = warp % wpb, wgrp = warp / wpb, lam = wsub * 32 + lane;
  const int ngrp4 = (nown + 3) / 4;
  if (C > 1) cg::this_cluster().sync(); else __syncthreads();

  uint64_t c0 = clock64(), g0 = gtime();
  for (int step = 0; step < nsteps; step++) {
    const uint32_t *cur = D + (step & 1) * DLEN + PADL;
    uint32_t *nxt = D + ((step + 1) & 1) * DLEN + PADL;
#ifdef PROF
    unsigned long long q0 = clock64();
#endif
    // phase 1
    for (int bi = wgrp; bi < nblk && wgrp < ngroups; bi += ngroups) {
      const Blk bk = BK[bi];
      uint64_t acc[2 * R];
      #pragma unroll
      for (int v = 0; v < 2 * R; v++) acc[v] = 0;
      {
        const int len = bk.me_lo + 1, i0 = lam * bk.l_lo;
        const int n = min(bk.l_lo, len - i0);
        if (n > 0) run_half(cur, bk.k0, i0, n, acc);
      }
      {
        const int lo = bk.k0 + 1, len = bk.me_hi - bk.k0, i0 = lam * bk.l_hi;
        const int n = min(bk.l_hi, len - i0);
        if (n > 0) run_half(cur, bk.k0 + ND, lo + i0, n, acc + R);
      }
      // reduce-scatter 2R values over the warp's 32 lanes; afterwards lane l holds value vidx(l) (in 32 / (2R) lanes)
      int vbase = 0;
      #pragma unroll
      for (int w = R; w >= 1; w >>= 1) {
        const int o = 32 * w / (2 * R);           // lane offset paired with this halving
        const bool up = (lane & o) != 0;
        #pragma unroll
        for (int v = 0; v < w; v++) {
          uint64_t send = up ? acc[v] : acc[v + w];
          uint64_t keep = up ? acc[v + w] : acc[v];
          acc[v] = keep + shfl64(send, o);
        }
        if (up) vbase += w;
      }
      #pragma unroll
      for (int o = 32 / (2 * R) / 2; o > 0; o >>= 1) acc[0] += shfl64(acc[0], o);
      if ((lane & (32 / (2 * R) - 1)) == 0) RED[(bi * wpb + wsub) * 2 * R + vbase] = acc[0];
    }
    __syncthreads();
#ifdef PROF
    unsigned long long q1 = clock64(); pacc[0] += q1 - q0;
#endif
    // phase 2
    for (int l = tid; l < ncols; l += tpb) {
      const Col c = cl[l];
      const Blk bk = BK[c.blk];
      uint64_t A0 = 0, A1 = 0;
      for (int w = 0; w < wpb; w++) {
        A0 += RED[(c.blk * wpb + w) * 2 * R + c.r];
        A1 += RED[(c.blk * wpb + w) * 2 * R + R + c.r];
      }
      const int clo = c.k, chi = c.k + ND;
      for (int i = bk.me_lo + 1; 2 * i < clo; i++) A0 = madw(cur[i], cur[clo - i], A0);
      for (int i = bk.me_hi + 1; 2 * i < chi; i++) A1 = madw(cur[i], cur[chi - i], A1);
      A0 *= 2; A1 *= 2;
      if ((clo & 1) == 0) { uint32_t h = cur[clo >> 1]; A0 = madw(h, h, A0); }
      if ((chi & 1) == 0) { uint32_t h = cur[chi >> 1]; A1 = madw(h, h, A1); }
      uint64_t m0 = madw(CP, (uint32_t)A1, 0);
      uint64_t m1 = madw(CP, (uint32_t)(A1 >> 32), 0);
      uint64_t lo = A0 + m0;
      uint64_t hi = (lo < m0) + (m1 >> 32);
      uint64_t mid = m1 << 32;
      lo += mid; hi += (lo < mid);
      uint4 t;
      t.x = (uint32_t)lo & MASK;
      t.y = (uint32_t)(lo >> 26) & MASK;
      t.z = (uint32_t)((lo >> 52) | (hi << 12)) & MASK;
      t.w = (uint32_t)(hi >> 14);
      T4[l] = t;
    }
    __syncthreads();
#ifdef PROF
    unsigned long long q2 = clock64(); pacc[1] += q2 - q1;
#endif
    // phase 3
    const uint32_t tag = (uint32_t)(step % 31 + 1) << 27;
    for (int q = tid; q < ngrp4 * C; q += tpb) {
      const int grp = q % ngrp4, h = q / ngrp4;
      const int m0 = a + 4 * grp;
      uint64_t e[5];
      #pragma unroll
      for (int x = 0; x < 5; x++) {
        int mm = m0 - 1 + x; if (mm < 0) mm += ND;
        uint64_t sm_ = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int k = mm - j; uint32_t w = 1;
          if (k < 0) { k += ND; w = CP; }
          int l = k - kstart; if (l < 0) l += ND;
          sm_ = madw(w, ((const uint32_t *)&T4[l])[j], sm_);
        }
        e[x] = sm_;
      }
      uint32_t f[4];
      #pragma unroll
      for (int x = 0; x < 4; x++) {
        const int m = m0 + x;
        f[x] = m < a + nown ? (uint32_t)((e[x + 1] & MASK) + (e[x] >> 26) * (m == 0 ? CP : 1)) : 0;
      }
      if (C == 1) {
        *(uint4 *)&nxt[m0] = make_uint4(f[0], f[1], f[2], f[3]);
      } else {
        asm volatile("st.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "r"(mapa(smem_u32(&nxt[m0]), h)), "r"(f[0] | tag), "r"(f[1] | tag), "r"(f[2] | tag), "r"(f[3] | tag) : "memory");
      }
    }
#ifdef PROF
    unsigned long long q3 = clock64(); pacc[2] += q3 - q2;
#endif
    if (C > 1) {
      for (int q = tid; q < ND4; q += tpb) {
        volatile uint32_t *w = &nxt[q];
        uint32_t v;
        do { v = *w; } while ((v & ~VMASK) != tag);
        nxt[q] = v & VMASK;
      }
    }
    __syncthreads();
#ifdef PROF
    pacc[3] += clock64() - q3;
#endif
  }
  uint64_t c1 = clock64(), g1 = gtime();
  if (C > 1) cg::this_cluster().sync();
  if (g == 0 && tid == 0) { timing[0] = c1 - c0; timing[1] = g1 - g0; }
#ifdef PROF
  if (tid == 0) for (int q = 0; q < 4; q++) timing[2 + 4 * g + q] = pacc[q];
#endif
  const uint32_t *fin = D + (nsteps & 1) * DLEN + PADL;
  if (g == 0) for (int q = tid; q < ND; q += tpb) xout[q] = fin[q];
}

template <int C>
static void launch(Dev dv, int tpb, size_t sm, const uint32_t *xin, uint32_t *xout, int n, int wpb, int maxblk, unsigned long long *tm) {
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(C); cfg.blockDim = dim3(tpb); cfg.dynamicSmemBytes = sm; cfg.attrs = at; cfg.numAttrs = C > 1 ? 1 : 0;
  CK(cudaFuncSetAttribute(chain_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  if (C > 8) CK(cudaFuncSetAttribute(chain_kernel<C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  CK(cudaLaunchKernelEx(&cfg, chain_kernel<C>, dv, xin, xout, n, wpb, maxblk, tm));
}

int main(int argc, char **argv) {
  if (argc < 7) { fprintf(stderr, "usage: %s C TPB N reps in.bin out.bin [WPB]\n", argv[0]); return 2; }
  int C = atoi(argv[1]), tpb = atoi(argv[2]), N = atoi(argv[3]), reps = atoi(argv[4]);
  int wpb = argc > 7 ? atoi(argv[7]) : 1;
  if (tpb > LB || tpb % (32 * wpb)) { fprintf(stderr, "tpb %d: over the launch bound %d or not a multiple of 32 * WPB\n", tpb, LB); return 2; }
  std::vector<uint32_t> x(ND);
  FILE *f = fopen(argv[5], "rb"); if (!f || fread(x.data(), 4, ND, f) != (size_t)ND) { fprintf(stderr, "bad in.bin\n"); return 2; } fclose(f);

  std::vector<Blk> blks; std::vector<Col> cols; std::vector<int> meta;
  int maxblk = 0;
  long long prods = 0;
  for (int g = 0; g < C; g++) {
    Plan P = plan_cta(g, C, 32 * wpb);
    meta.insert(meta.end(), {P.a, P.nown, P.kstart, P.ncols, P.nblk, (int)blks.size(), (int)cols.size(), 0, 0, 0});
    for (auto &b : P.blk) prods += (long long)R * 32 * wpb * (b.l_lo + b.l_hi);
    blks.insert(blks.end(), P.blk.begin(), P.blk.end());
    cols.insert(cols.end(), P.cols.begin(), P.cols.end());
    maxblk = std::max(maxblk, P.nblk);
  }
  Dev dv;
  Blk *db; Col *dc; int *dm;
  CK(cudaMalloc(&db, blks.size() * sizeof(Blk))); CK(cudaMemcpy(db, blks.data(), blks.size() * sizeof(Blk), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dc, cols.size() * sizeof(Col))); CK(cudaMemcpy(dc, cols.data(), cols.size() * sizeof(Col), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dm, meta.size() * sizeof(int))); CK(cudaMemcpy(dm, meta.data(), meta.size() * sizeof(int), cudaMemcpyHostToDevice));
  dv.blk = db; dv.cols = dc; dv.meta = dm;
  uint32_t *dx, *dy; unsigned long long *dtm;
  CK(cudaMalloc(&dx, ND * 4)); CK(cudaMalloc(&dy, ND * 4)); CK(cudaMalloc(&dtm, 8 * 80)); CK(cudaMemset(dtm, 0, 8 * 80));
  CK(cudaMemcpy(dx, x.data(), ND * 4, cudaMemcpyHostToDevice));
  size_t sm = (size_t)2 * DLEN * 4 + (size_t)maxblk * wpb * 2 * R * 8 + MAXCOLS * 16 + MAXBLK * sizeof(Blk);

  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  std::vector<double> ns, cyc; double prof[4] = {0, 0, 0, 0};
  for (int rep = 0; rep < reps + 1; rep++) {
    switch (C) {
      case 1: launch<1>(dv, tpb, sm, dx, dy, N, wpb, maxblk, dtm); break;
      case 2: launch<2>(dv, tpb, sm, dx, dy, N, wpb, maxblk, dtm); break;
      case 4: launch<4>(dv, tpb, sm, dx, dy, N, wpb, maxblk, dtm); break;
      case 8: launch<8>(dv, tpb, sm, dx, dy, N, wpb, maxblk, dtm); break;
      case 16: launch<16>(dv, tpb, sm, dx, dy, N, wpb, maxblk, dtm); break;
      default: fprintf(stderr, "C must be 1, 2, 4, 8 or 16\n"); return 2;
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned long long tm[80]; CK(cudaMemcpy(tm, dtm, 8 * 80, cudaMemcpyDeviceToHost));
    if (rep > 0) {
      cyc.push_back((double)tm[0] / N); ns.push_back((double)tm[1] / N);
      for (int q = 0; q < 4; q++) { prof[q] = 0; for (int gg = 0; gg < C; gg++) prof[q] = std::max(prof[q], (double)tm[2 + 4 * gg + q] / N); }
    }
  }
  std::vector<uint32_t> y(ND);
  CK(cudaMemcpy(y.data(), dy, ND * 4, cudaMemcpyDeviceToHost));
  f = fopen(argv[6], "wb"); fwrite(y.data(), 4, ND, f); fclose(f);
  std::vector<double> sns = ns; std::sort(sns.begin(), sns.end());
  int imin = (int)(std::min_element(ns.begin(), ns.end()) - ns.begin());
  printf("{\"kernel\":\"cluster-warpblock-r26\",\"mode\":\"%s\",\"prof_cyc_max_over_ctas\":[%.0f,%.0f,%.0f,%.0f],\"R\":%d,\"WPB\":%d,\"C\":%d,\"tpb\":%d,\"N\":%d,\"reps\":%d,"
         "\"ns_per_sq_min\":%.2f,\"ns_per_sq_med\":%.2f,\"cyc_per_sq_at_min\":%.1f,\"sm_clock_mhz_at_min\":%.1f,\"max_blocks\":%d,\"products\":%lld,\"smem\":%zu,"
         "\"gpu\":\"%s\",\"sms\":%d}\n",
         C > 1 ? "tagged-st.v4-poll" : "single-cta", prof[0], prof[1], prof[2], prof[3], R, wpb, C, tpb, N, reps, sns[0], sns[sns.size() / 2],
         cyc[imin], cyc[imin] / ns[imin] * 1000.0, maxblk, prods, sm, p.name, p.multiProcessorCount);
  return 0;
}
