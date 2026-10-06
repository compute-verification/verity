// The latency of a dependent chain of 16,448-bit squarings mod p = 2^16448 - 21065 on one GPU: N squarings of one value,
// each needing the whole of the last, on C CTAs (one CTA, or a thread-block cluster of C sharing distributed shared memory).
//
// Representation: x = sum_m d_m 2^(26 m), m < 633, in redundant digits (d_m < 2^26 + 3, digits 0..2 < 1.65 * 2^26; the
// bounds are a fixed point, bounds.py). 26 * 633 = 16458, so 2^(26*633) = 2^10 * 2^16448 = c' = 21065 * 2^10 mod p, and
//   x^2 = sum_c S_c 2^(26 c),  S_c = sum_{i+j=c} d_i d_j  (c < 2*633 - 1),   T_k = S_k + c' S_{k+633}  (k < 633, < 2^86)
// Each T_k is cut into four 26-bit chunks t_k,0..3 and the next digits are two carry passes (no carry chain):
//   e_m = sum_j t_{m-j},j (c' for a chunk past the top),  d'_m = (e_m mod 2^26) + (e_{m-1} >> 26) (times c' at m = 0).
// S_c is formed with symmetry: S_c = 2 A_c + [c even] d_{c/2}^2, A_c = sum_{i < c-i} d_i d_{c-i}. Output digit k is the
// pair of columns k and k + 633, together 316 products and a square whatever k is, so the digits split evenly.
//
// Work: digit k's columns are cut into blocks of R consecutive columns ("block-halves", low k0.., high k0+633..); each
// thread runs one lane task, a contiguous run of i for one block-half, with the R partners d_{s0+r-i} in a register window
// that slides one digit per i, and leaves R partial sums in shared memory. Then one thread per column adds the partials,
// the few products past the block's diagonal (the tail) and the square, folds and cuts T_k; then one thread per owned
// digit runs the carry passes. In a cluster, CTA g owns digits [a_g, b_g), computes 4 columns below it again (the carry
// halo), and writes its digits into every CTA's next buffer (st.shared::cluster), then a cluster barrier.
//
// Timing: device clocks only (clock64 and %globaltimer on CTA 0, around the N steps). Output: one JSON line per run, and
// the final digits (for the driver, which checks them against Python's pow).
// Build: nvcc -O3 -arch=sm_120 -std=c++17 -o sqchain sqchain.cu
// Run:   sqchain C TPB N reps in.bin out.bin [MODE]   (in.bin: 633 little-endian u32 digits; out.bin: the digits after N squarings)
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
#define R 8
#endif
static const int ND = 633;
static const uint32_t MASK = (1u << 26) - 1;
static const uint32_t CP = 21065u << 10;
static const int PADL = 64, PADR = 96;
static const int DLEN = PADL + ND + PADR;
static const int HALO = 4;
static const int MAXCOLS = 640;
static const int DLENS = (DLEN + DLEN / 32 + 4) / 4 * 4;   // the digit buffer with one pad word per 32 (swz)
static const int MAXBH = 2 * ((ND + R - 1) / R + 2);
static const int NMETA = 10;
// x -> its word in a digit buffer: one pad word every 32, so lanes walking runs a multiple of 32 apart miss each other's banks
__host__ __device__ __forceinline__ int swz(int x) { x += PADL; return x + (x >> 5); }

struct Task { int s0, ib, ie, ng, slot; };          // a lane task: i in [ib, ie), ng groups of R, partials at slot
struct BHalf { int s0, main_end, first, cnt; };      // a block-half: columns s0..s0+R-1, lane slots [first, first+cnt)
struct Col { int k, bh, r; };                        // a computed column pair (k, k+ND): low block-half bh, high bh+1

struct Plan {
  int a, nown, kstart, ncols, ntasks, nbh;
  std::vector<Task> tasks;
  std::vector<BHalf> bh;
  std::vector<Col> cols;
};

static inline int floordiv2(int x) { return x >= 0 ? x / 2 : -((1 - x) / 2); }

// CTA g of C: owned digits, computed columns (owned plus the halo below, cyclic), block-halves and lane tasks
static Plan plan_cta(int g, int C, int tpb) {
  Plan P;
  int per = ((ND + C - 1) / C + 3) / 4 * 4;
  P.a = std::min(ND, g * per);
  int b = std::min(ND, P.a + per);
  P.nown = b - P.a;
  int halo = C == 1 ? 0 : HALO;
  P.kstart = ((P.a - halo) % ND + ND) % ND;
  P.ncols = P.nown + halo;
  // contiguous segments of computed columns, in local order
  std::vector<std::pair<int, int>> segs;  // (start k, count)
  if (P.kstart + P.ncols <= ND) segs.push_back({P.kstart, P.ncols});
  else { segs.push_back({P.kstart, ND - P.kstart}); segs.push_back({0, P.ncols - (ND - P.kstart)}); }
  std::vector<int> V;
  for (auto &sg : segs) {
    for (int k0 = sg.first; k0 < sg.first + sg.second; k0 += R) {
      int blk = (int)P.bh.size();
      for (int h = 0; h < 2; h++) {
        BHalf x;
        x.s0 = k0 + h * ND;
        int lo = h == 0 ? 0 : k0 + 1;
        x.main_end = floordiv2(x.s0 - 1);
        x.first = 0; x.cnt = 0;
        P.bh.push_back(x);
        V.push_back(std::max(0, x.main_end - lo + 1));
      }
      for (int r = 0; r < R && k0 + r < sg.first + sg.second; r++) P.cols.push_back({k0 + r, blk, r});
    }
  }
  P.nbh = (int)P.bh.size();
  // lanes: groups of R iterations, spread so every lane gets about the same number of groups
  long gtot = 0;
  std::vector<int> G(P.nbh);
  for (int q = 0; q < P.nbh; q++) { G[q] = (V[q] + R - 1) / R; gtot += G[q]; }
  int lanes = std::min<long>(tpb, gtot);
  int gper = (int)((gtot + lanes - 1) / lanes);
  int slot = 0;
  for (int q = 0; q < P.nbh; q++) {
    int lo = (q % 2 == 0) ? 0 : (P.bh[q].s0 - ND) + 1;
    int ie = P.bh[q].main_end + 1;
    P.bh[q].first = slot;
    for (int g0 = 0; g0 < G[q]; g0 += gper) {
      int ng = std::min(gper, G[q] - g0);
      Task t{P.bh[q].s0, lo + g0 * R, ie, ng, slot++};
      P.tasks.push_back(t);
    }
    P.bh[q].cnt = slot - P.bh[q].first;
  }
  P.ntasks = slot;
  if (P.ntasks > tpb || P.ncols > MAXCOLS || P.nown > tpb || (C > 1 && P.nown > 512)) { fprintf(stderr, "plan does not fit: tasks %d cols %d own %d tpb %d\n", P.ntasks, P.ncols, P.nown, tpb); exit(2); }
  return P;
}

struct Dev {
  const Task *tasks; const BHalf *bh; const Col *cols; const int *meta;   // meta per CTA: a nown kstart ncols ntasks nbh toff bhoff coff tpc
};

// one IMAD.WIDE.U32: nvcc otherwise multiplies the widened operands 64 x 64
__device__ __forceinline__ uint64_t madw(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r; asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c)); return r;
}
__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
__device__ __forceinline__ void st_async(uint32_t raddr, uint32_t v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];" :: "r"(raddr), "r"(v), "r"(rbar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n .reg .pred p;\n W%=: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra W%=;\n}" :: "r"(bar), "r"(parity) : "memory");
}

// MODE 0: digits by st.shared::cluster, then a cluster barrier. MODE 1: digits by st.async, each signalling the receiver's
// mbarrier for that buffer with its bytes; a CTA starts the next step when all 4 * 633 bytes of it have landed.
#ifndef LB
#define LB 512
#endif
template <int C, int MODE>
__global__ void __launch_bounds__(LB, 1) chain_kernel(Dev dv, const uint32_t *__restrict__ xin,
                                                                       uint32_t *__restrict__ xout, int nsteps,
                                                                       unsigned long long *timing) {
  __shared__ __align__(8) uint64_t mbar[2];
#ifdef PROF
  unsigned long long pacc[4] = {0, 0, 0, 0};
#endif
  extern __shared__ __align__(16) unsigned char smem[];
  const int tpb = blockDim.x, tid = threadIdx.x;
  uint32_t *D = (uint32_t *)smem;                                   // [2][DLENS], swizzled
  uint64_t *P = (uint64_t *)(smem + 2 * DLENS * 4);                 // [tpb][R + 1]
  uint4 *T4 = (uint4 *)(P + (size_t)(R + 1) * tpb);                 // [MAXCOLS]
  BHalf *BH = (BHalf *)(T4 + MAXCOLS);                              // [MAXBH]
  uint32_t *stage = (uint32_t *)(BH + MAXBH);                       // [512]

  int g = C == 1 ? 0 : (int)cg::this_cluster().block_rank();
  const int *mt = dv.meta + NMETA * g;
  const int a = mt[0], nown = mt[1], kstart = mt[2], ncols = mt[3], ntasks = mt[4], nbh = mt[5], tpc = mt[9];
  const Task *tk = dv.tasks + mt[6];
  const Col *cl = dv.cols + mt[8];
  for (int q = tid; q < nbh; q += tpb) BH[q] = dv.bh[mt[7] + q];
  for (int q = tid; q < DLEN; q += tpb) {
    int x = q - PADL;
    uint32_t v = (x >= 0 && x < ND) ? xin[x] : 0;
    D[swz(x)] = v; D[DLENS + swz(x)] = v;
  }
  Task my = tid < ntasks ? tk[tid] : Task{0, 0, 0, 0, 0};
  const int cidx = tid / tpc, sub = tid % tpc;
  Col mc = cidx < ncols ? cl[cidx] : Col{0, 0, 0};
  __syncthreads();
  // this thread's two columns: partials at pb, cnt of them, tail from ti, column index cc (cnt 0 past the last column)
  int pb0, cnt0, ti0, cc0, pb1, cnt1, ti1, cc1;
  {
    BHalf blo = BH[mc.bh], bhi = BH[mc.bh + 1];
    bool on = cidx < ncols;
    pb0 = blo.first * (R + 1) + mc.r; cnt0 = on ? blo.cnt : 0; ti0 = blo.main_end + 1; cc0 = on ? blo.s0 + mc.r : 0;
    pb1 = bhi.first * (R + 1) + mc.r; cnt1 = on ? bhi.cnt : 0; ti1 = bhi.main_end + 1; cc1 = on ? bhi.s0 + mc.r : 0;
  }
  if (MODE == 1 && tid == 0) {
    for (int b = 0; b < 2; b++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&mbar[b])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (C > 1) cg::this_cluster().sync(); else __syncthreads();
  if (MODE == 1 && tid == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(&mbar[1])), "r"(ND * 4) : "memory");

  uint64_t c0 = clock64(), g0 = gtime();
  for (int step = 0; step < nsteps; step++) {
    const uint32_t *cur = D + (step & 1) * DLENS;
    uint32_t *nxt = D + ((step + 1) & 1) * DLENS;
#ifdef PROF
    unsigned long long q0 = clock64();
#endif
    // phase 1: lane tasks, R partial sums each
    if (tid < ntasks) {
      uint64_t acc[R];
      uint32_t B[R];
      #pragma unroll
      for (int r = 0; r < R; r++) { acc[r] = 0; B[r] = cur[swz(my.s0 - my.ib + r)]; }
      for (int gg = 0; gg < my.ng; gg++) {
        const int ib = my.ib + gg * R;
        #pragma unroll
        for (int u = 0; u < R; u++) {
          const int i = ib + u;
          B[(R - u) % R] = cur[swz(my.s0 - i)];
          uint32_t di = i < my.ie ? cur[swz(i)] : 0;
          #pragma unroll
          for (int r = 0; r < R; r++) acc[r] = madw(di, B[(r - u + R) % R], acc[r]);
        }
      }
      #pragma unroll
      for (int r = 0; r < R; r++) P[my.slot * (R + 1) + r] = acc[r];
    }
    __syncthreads();
#ifdef PROF
    unsigned long long q1 = clock64(); pacc[0] += q1 - q0;
#endif
    // phase 2: tpc threads per computed column pair; partials, tail, square, then a shuffle sum
    {
      uint64_t s[2];
      #pragma unroll
      for (int h = 0; h < 2; h++) {
        const int pb = h ? pb1 : pb0, cnt = h ? cnt1 : cnt0, i0 = h ? ti1 : ti0, c = h ? cc1 : cc0;
        uint64_t A0 = 0, A1 = 0;
        int j = sub;
        for (; j + tpc < cnt; j += 2 * tpc) {
          A0 += P[pb + j * (R + 1)];
          A1 += P[pb + (j + tpc) * (R + 1)];
        }
        if (j < cnt) A0 += P[pb + j * (R + 1)];
        for (int i = i0 + sub; 2 * i < c; i += tpc) A1 = madw(cur[swz(i)], cur[swz(c - i)], A1);
        uint64_t A = 2 * (A0 + A1);
        for (int o = tpc / 2; o > 0; o /= 2) A += __shfl_xor_sync(0xffffffffu, A, o);
        if ((c & 1) == 0) { uint32_t h2 = cur[swz(c >> 1)]; A = madw(h2, h2, A); }
        s[h] = A;
      }
      if (cidx < ncols && sub == 0) {
        // T = s0 + c' s1 < 2^86: low 64 bits and the rest
        uint64_t m0 = madw(CP, (uint32_t)s[1], 0);
        uint64_t m1 = madw(CP, (uint32_t)(s[1] >> 32), 0);
        uint64_t lo = s[0] + m0;
        uint64_t hi = (lo < m0) + (m1 >> 32);
        uint64_t mid = m1 << 32;
        lo += mid; hi += (lo < mid);
        uint4 t;
        t.x = (uint32_t)lo & MASK;
        t.y = (uint32_t)(lo >> 26) & MASK;
        t.z = (uint32_t)((lo >> 52) | (hi << 12)) & MASK;
        t.w = (uint32_t)(hi >> 14);
        T4[cidx] = t;
      }
    }
    __syncthreads();
#ifdef PROF
    unsigned long long q2 = clock64(); pacc[1] += q2 - q1;
#endif
    // phase 3: owned digits, two carry passes
    if (tid < nown) {
      const int m = a + tid;
      uint64_t e[2];
      #pragma unroll
      for (int q = 0; q < 2; q++) {
        int mm = m - q; if (mm < 0) mm += ND;
        uint64_t sm_ = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int k = mm - j; uint32_t w = 1;
          if (k < 0) { k += ND; w = CP; }
          int l = k - kstart; if (l < 0) l += ND;
          const uint32_t *tt = (const uint32_t *)&T4[l];
          sm_ = madw(w, tt[j], sm_);
        }
        e[q] = sm_;
      }
      uint64_t f = (e[0] & MASK) + (e[1] >> 26) * (m == 0 ? CP : 1);
      if (C == 1) nxt[swz(m)] = (uint32_t)f; else stage[tid] = (uint32_t)f;
    }
    if (C > 1 && MODE == 0) {
      cg::cluster_group cluster = cg::this_cluster();
      __syncthreads();
#ifdef PROF
      unsigned long long q3 = clock64(); pacc[2] += q3 - q2;
#endif
      for (int q = tid; q < C * nown; q += tpb) {
        int h = q / nown, j = q - h * nown;
        uint32_t *rem = cluster.map_shared_rank(nxt, h);
        rem[swz(a + j)] = stage[j];
      }
      cluster.sync();
#ifdef PROF
      pacc[3] += clock64() - q3;
#endif
    } else if (C > 1) {
      const int b = (step + 1) & 1;
      __syncthreads();
#ifdef PROF
      unsigned long long q3 = clock64(); pacc[2] += q3 - q2;
#endif
      const uint32_t lnxt = smem_u32(nxt), lbar = smem_u32(&mbar[b]);
      for (int q = tid; q < C * nown; q += tpb) {
        int h = q / nown, j = q - h * nown;
        st_async(mapa(lnxt + 4 * swz(a + j), h), stage[j], mapa(lbar, h));
      }
      mbar_wait(lbar, (step >> 1) & 1);
      if (tid == 0 && step + 1 < nsteps)                       // re-arm the other buffer for the step after
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(&mbar[b ^ 1])), "r"(ND * 4) : "memory");
#ifdef PROF
      pacc[3] += clock64() - q3;
#endif
    } else {
      __syncthreads();
    }
  }
  if (C > 1 && MODE == 1) cg::this_cluster().sync();
  uint64_t c1 = clock64(), g1 = gtime();
  if (g == 0 && tid == 0) { timing[0] = c1 - c0; timing[1] = g1 - g0; }
#ifdef PROF
  if (tid == 0) for (int q = 0; q < 4; q++) timing[2 + 4 * g + q] = pacc[q];
#endif
  const uint32_t *fin = D + (nsteps & 1) * DLENS;
  if (g == 0) for (int q = tid; q < ND; q += tpb) xout[q] = fin[swz(q)];
}

template <int C, int MODE>
static void launch(Dev dv, int tpb, size_t sm, const uint32_t *xin, uint32_t *xout, int n, unsigned long long *tm) {
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(C); cfg.blockDim = dim3(tpb); cfg.dynamicSmemBytes = sm; cfg.attrs = at; cfg.numAttrs = C > 1 ? 1 : 0;
  CK(cudaFuncSetAttribute(chain_kernel<C, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  if (C > 8) CK(cudaFuncSetAttribute(chain_kernel<C, MODE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  CK(cudaLaunchKernelEx(&cfg, chain_kernel<C, MODE>, dv, xin, xout, n, tm));
}

int main(int argc, char **argv) {
  if (argc < 7) { fprintf(stderr, "usage: %s C TPB N reps in.bin out.bin\n", argv[0]); return 2; }
  int C = atoi(argv[1]), tpb = atoi(argv[2]), N = atoi(argv[3]), reps = atoi(argv[4]);
  int mode = argc > 7 ? atoi(argv[7]) : 0;
  if (tpb > LB) { fprintf(stderr, "tpb %d is over this build's launch bound %d\n", tpb, LB); return 2; }
  std::vector<uint32_t> x(ND);
  FILE *f = fopen(argv[5], "rb"); if (!f || fread(x.data(), 4, ND, f) != (size_t)ND) { fprintf(stderr, "bad in.bin\n"); return 2; } fclose(f);

  std::vector<Task> tasks; std::vector<BHalf> bhs; std::vector<Col> cols; std::vector<int> meta;
  int maxtasks = 0, maxcols = 0;
  long long prods = 0;
  for (int g = 0; g < C; g++) {
    Plan P = plan_cta(g, C, tpb);
    int tpc = 1;
    while (tpc < 8 && 2 * tpc * P.ncols <= tpb) tpc *= 2;
    if (P.ncols * tpc > tpb || P.nbh > MAXBH) { fprintf(stderr, "columns do not fit\n"); return 2; }
    meta.insert(meta.end(), {P.a, P.nown, P.kstart, P.ncols, P.ntasks, P.nbh, (int)tasks.size(), (int)bhs.size(), (int)cols.size(), tpc});
    for (auto &t : P.tasks) prods += (long long)t.ng * R * R;
    tasks.insert(tasks.end(), P.tasks.begin(), P.tasks.end());
    bhs.insert(bhs.end(), P.bh.begin(), P.bh.end());
    cols.insert(cols.end(), P.cols.begin(), P.cols.end());
    maxtasks = std::max(maxtasks, P.ntasks); maxcols = std::max(maxcols, P.ncols);
  }
  Dev dv;
  Task *dt; BHalf *db; Col *dc; int *dm;
  CK(cudaMalloc(&dt, tasks.size() * sizeof(Task))); CK(cudaMemcpy(dt, tasks.data(), tasks.size() * sizeof(Task), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&db, bhs.size() * sizeof(BHalf))); CK(cudaMemcpy(db, bhs.data(), bhs.size() * sizeof(BHalf), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dc, cols.size() * sizeof(Col))); CK(cudaMemcpy(dc, cols.data(), cols.size() * sizeof(Col), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dm, meta.size() * sizeof(int))); CK(cudaMemcpy(dm, meta.data(), meta.size() * sizeof(int), cudaMemcpyHostToDevice));
  dv.tasks = dt; dv.bh = db; dv.cols = dc; dv.meta = dm;
  uint32_t *dx, *dy; unsigned long long *dtm;
  CK(cudaMalloc(&dx, ND * 4)); CK(cudaMalloc(&dy, ND * 4)); CK(cudaMalloc(&dtm, 8 * 80)); CK(cudaMemset(dtm, 0, 8 * 80));
  CK(cudaMemcpy(dx, x.data(), ND * 4, cudaMemcpyHostToDevice));
  size_t sm = (size_t)2 * DLENS * 4 + (size_t)(R + 1) * tpb * 8 + MAXCOLS * 16 + MAXBH * sizeof(BHalf) + 512 * 4;

  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  std::vector<double> ns, cyc; double prof[4] = {0, 0, 0, 0};
  for (int rep = 0; rep < reps + 1; rep++) {
    switch (C) {
      case 1: if (mode) launch<1, 1>(dv, tpb, sm, dx, dy, N, dtm); else launch<1, 0>(dv, tpb, sm, dx, dy, N, dtm); break;
      case 2: if (mode) launch<2, 1>(dv, tpb, sm, dx, dy, N, dtm); else launch<2, 0>(dv, tpb, sm, dx, dy, N, dtm); break;
      case 4: if (mode) launch<4, 1>(dv, tpb, sm, dx, dy, N, dtm); else launch<4, 0>(dv, tpb, sm, dx, dy, N, dtm); break;
      case 8: if (mode) launch<8, 1>(dv, tpb, sm, dx, dy, N, dtm); else launch<8, 0>(dv, tpb, sm, dx, dy, N, dtm); break;
      case 16: if (mode) launch<16, 1>(dv, tpb, sm, dx, dy, N, dtm); else launch<16, 0>(dv, tpb, sm, dx, dy, N, dtm); break;
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
  printf("{\"kernel\":\"cluster-schoolbook-r26\",\"mode\":\"%s\",\"prof_cyc_max_over_ctas\":[%.0f,%.0f,%.0f,%.0f],\"R\":%d,\"C\":%d,\"tpb\":%d,\"N\":%d,\"reps\":%d,\"ns_per_sq_min\":%.2f,\"ns_per_sq_med\":%.2f,"
         "\"cyc_per_sq_at_min\":%.1f,\"sm_clock_mhz_at_min\":%.1f,\"max_tasks\":%d,\"max_cols\":%d,\"products\":%lld,\"smem\":%zu,"
         "\"gpu\":\"%s\",\"sms\":%d}\n",
         mode ? "mbarrier-st.async" : "cluster-barrier", prof[0], prof[1], prof[2], prof[3], R, C, tpb, N, reps, sns[0], sns[sns.size() / 2], cyc[imin], cyc[imin] / ns[imin] * 1000.0, maxtasks, maxcols, prods, sm,
         p.name, p.multiProcessorCount);
  return 0;
}
