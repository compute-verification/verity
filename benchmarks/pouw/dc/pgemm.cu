// The honest Dither GEMM on datacenter Blackwell (sm_100a), pipelined: tcgen05.mma into tiles of tensor memory in flight,
// the tcgen05.ld loaders and the duo fold on mbarriers, at commit spacing L = 32 lm.
//
// The receipt is dgemm.cu's (dgemm_ref.py is the twin) with the boundary every L steps: chain (i, tn, c), c in 0..15,
// starts at duo_init(seed, (i tiles_n + tn) 16 + c) and absorbs, at each s = 1..k/L and in ascending tile column j with
// j % 16 == c, the s32 running sum T_s = X[:, :L s] W[:, :L s]^T of row i at key position (j / 16) % 16.
//
// A CTA computes a 128 x 512 block as NB tiles of 128 x BN (BN = 512 / NB: NB = 2 is two 256-column accumulators, NB = 4
// four of 128; either way TMEM's 512 columns). The running sum lives in TMEM, so the buffers are the tiles:
//   the MMA warp (the last; one elected thread), for each group g of lm k-steps and each tile b: waits empty[b] (tile b's
//     loaders have read group g - 1), issues lm tcgen05.mma kind::i8 128xBNx32 accumulating into tile b, and
//     tcgen05.commit's full[b];
//   loader warpgroup w of LW (tiles b with b % LW == w; a thread owns one row): waits full[b], reads the row with
//     tcgen05.ld.32x32b.x32 and tcgen05.wait::ld, arrives on empty[b] after the last load, and folds the words.
// MODE 0: the loaders only hand the barrier back (the pipeline's own rate); 1: they load every word and don't fold (the
// TMEM stage); 2: load and fold (honest); 3: no loaders, the MMA warp waits on its own commit before it reissues a tile
// (the round trip's MMA and commit part alone).
// Operands are resident: the CTA's rows of X and W for k in [0, PER) go to shared memory once and step s reads stage
// s % (PER / 32), so the kernel computes the GEMM of K-periodic inputs (period PER), which is what the gate feeds it. The
// rate is the tensor, TMEM and fold stage's; operand streaming (TMA) is not in it.
//
// Modes:
//   pgemm gate DIR M N K SEED LM NB LW   K-periodic inputs, C and R poisoned with 0xA5, MODE 2, compared with the host
//                                        reference; X, W, C, R and meta.json to DIR for dgemm_ref.py
//   pgemm rate REPS KSTEPS               every (NB, LW, MODE) at lm 1, 2, 4, 8 on one CTA a SM, REPS interleaved
//                                        launches; one JSON line a configuration (cycles from clock64 a CTA)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a pgemm.cu -o pgemm
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dc.cuh"

#define CK(x)                                                                                          \
  do {                                                                                                 \
    cudaError_t e_ = (x);                                                                              \
    if (e_ != cudaSuccess) {                                                                           \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));           \
      exit(2);                                                                                         \
    }                                                                                                  \
  } while (0)

constexpr int PER = 128, STAGES = PER / 32, BM = 128, CTA_N = 512, NCH = 16;
constexpr uint32_t A_STAGE = BM * 32, B_STAGE = CTA_N * 32;
// More than half of the SM's 228 KB, so one CTA a SM (a second would wait in tcgen05.alloc for the 512 columns).
constexpr int SMEM = 120 * 1024;
static_assert(STAGES * (A_STAGE + B_STAGE) <= SMEM, "operand stages");

struct Bars {
  uint64_t full[4], empty[4];
  uint32_t tslot;
};

DC_DEV void hand_back(uint64_t* empty, uint32_t lane) {
  tc_before_sync();
  __syncwarp();
  if (lane == 0) mbar_arrive(empty);
}
// One tile row, honest: BN / 32 loads of 32 words, the barrier handed back after the last, each word folded at its key
// position (compile-time, so unrolled; the fold's dependence keeps ptxas from hoisting the loads).
template <int U, int END>
DC_DEV void fold_pass(DuoSt (&ch)[NCH], uint32_t ta, uint64_t* empty, uint32_t lane) {
  if constexpr (U < END) {
    uint32_t r[32];
    tmem_ld32(ta + 32u * U, r);
    tmem_wait_ld();
    if constexpr (U == END - 1) hand_back(empty, lane);
#pragma unroll
    for (int c = 0; c < 16; c++) duo<(2 * U) % DUO_NPOS>(ch[c], r[c]);
#pragma unroll
    for (int c = 0; c < 16; c++) duo<(2 * U + 1) % DUO_NPOS>(ch[c], r[16 + c]);
    fold_pass<U + 1, END>(ch, ta, empty, lane);
  }
}
// The same loads without the fold, two in flight as in the probes. ptxas turns tcgen05.wait::ld into scoreboard waits, so
// an unrolled pass would issue every load first and spill; the loop's back edge stops it. STORE: C instead of the sink.
template <int BN, bool STORE>
DC_DEV void load_pass(uint32_t ta, int32_t* crow, uint32_t& sink) {
#pragma unroll 1
  for (int u = 0; u < BN; u += 64) {
    uint32_t r0[32], r1[32];
    tmem_ld32(ta + (uint32_t)u, r0);
    tmem_ld32(ta + (uint32_t)u + 32u, r1);
    tmem_wait_ld();
    if constexpr (STORE) {
#pragma unroll
      for (int c = 0; c < 32; c += 4) {
        *reinterpret_cast<int4*>(crow + u + c) = make_int4((int)r0[c], (int)r0[c + 1], (int)r0[c + 2], (int)r0[c + 3]);
        *reinterpret_cast<int4*>(crow + u + 32 + c) =
            make_int4((int)r1[c], (int)r1[c + 1], (int)r1[c + 2], (int)r1[c + 3]);
      }
    } else {
      sink += r0[0] ^ r1[31];
    }
  }
}

// The register cap a block of 4 LW + 1 warps can launch with: an SMSP's 16K registers over the most warps it holds (LW = 4,
// 17 warps, would get 96, and the honest loader spills there).
template <int NB, int LW, int MODE>
__global__ void __maxnreg__(LW == 2 ? 168 : 255)
    pg100(const int8_t* X, const int8_t* W, int ldx, int n, int k, int lm, uint32_t seed, int32_t* C, uint32_t* R,
          long long* clk) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  constexpr int BN = CTA_N / NB, TPW = NB / LW, MMA_WARP = 4 * LW;
  static_assert(NB % LW == 0 && NB <= 4 && LW <= 2, "tiles a loader warpgroup");
  extern __shared__ __align__(1024) unsigned char dsm[];
  unsigned char* sA = dsm;
  unsigned char* sB = dsm + STAGES * A_STAGE;
  __shared__ Bars bars;
  const uint32_t w = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
  const int tm = blockIdx.y, cn = blockIdx.x, tiles_n = n / BN;
  for (int c = threadIdx.x; c < BM * (PER / 16); c += blockDim.x) {
    const int r = c / (PER / 16), kk = 16 * (c % (PER / 16));
    *reinterpret_cast<uint4*>(sA + (kk / 32) * A_STAGE + kmaj_off(r, kk % 32, 32)) =
        *reinterpret_cast<const uint4*>(X + (size_t)(tm * BM + r) * ldx + kk);
  }
  for (int c = threadIdx.x; c < CTA_N * (PER / 16); c += blockDim.x) {
    const int r = c / (PER / 16), kk = 16 * (c % (PER / 16));
    *reinterpret_cast<uint4*>(sB + (kk / 32) * B_STAGE + kmaj_off(r, kk % 32, 32)) =
        *reinterpret_cast<const uint4*>(W + (size_t)(cn * CTA_N + r) * ldx + kk);
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int b = 0; b < NB; b++) {
      mbar_init(&bars.full[b], 1);
      mbar_init(&bars.empty[b], 4);
    }
    fence_mbar_init();
  }
  if (w == MMA_WARP) tmem_alloc(&bars.tslot, 512);
  fence_async_smem();
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = bars.tslot;
  const int groups = k / (32 * lm);
  const long long t0 = clock64();
  if (w == MMA_WARP) {
    constexpr uint32_t idesc = umma_idesc(2, 1, 1, BM, BN);
    const uint32_t a0 = smem_u32(sA), b0 = smem_u32(sB);
    for (int g = 0; g < groups; g++) {
#pragma unroll
      for (int b = 0; b < NB; b++) {
        if (g > 0) mbar_wait(MODE == 3 ? &bars.full[b] : &bars.empty[b], (uint32_t)((g - 1) & 1));
        tc_after_sync();
        if (elect_one()) {
          for (int i = 0; i < lm; i++) {
            const int s = g * lm + i, st = s % STAGES;
            umma_i8(tmem + (uint32_t)(b * BN), umma_desc(a0 + st * A_STAGE, kmaj_lbo(), kmaj_sbo(32)),
                    umma_desc(b0 + st * B_STAGE + b * BN * 32, kmaj_lbo(), kmaj_sbo(32)), idesc, s > 0);
          }
          umma_commit(&bars.full[b]);
        }
        __syncwarp();
      }
    }
    if constexpr (MODE == 3) {
#pragma unroll
      for (int b = 0; b < NB; b++) mbar_wait(&bars.full[b], (uint32_t)((groups - 1) & 1));
    }
  } else if constexpr (MODE != 3) {
    const uint32_t wg = w >> 2, q = w & 3;
    const int row = tm * BM + 32 * (int)q + (int)lane;
    DuoSt ch[TPW][NCH];
    if constexpr (MODE == 2) {
#pragma unroll
      for (int tb = 0; tb < TPW; tb++)
#pragma unroll
        for (int c = 0; c < NCH; c++)
          ch[tb][c] = duo_init(seed, (uint32_t)((row * tiles_n + cn * NB + tb * LW + (int)wg) * NCH + c));
    }
    uint32_t sink = 0;
    auto taddr = [&](int b) { return tmem + ((32u * q) << 16) + (uint32_t)(b * BN); };
    auto crow = [&](int b) { return C + (size_t)row * n + (cn * NB + b) * BN; };
    for (int g = 0; g < groups; g++) {
#pragma unroll
      for (int tb = 0; tb < TPW; tb++) {
        const int b = tb * LW + (int)wg;
        mbar_wait(&bars.full[b], (uint32_t)(g & 1));
        tc_after_sync();
        if constexpr (MODE == 2) {
          fold_pass<0, BN / 32>(ch[tb], taddr(b), &bars.empty[b], lane);
        } else {
          if constexpr (MODE == 1) load_pass<BN, false>(taddr(b), nullptr, sink);
          hand_back(&bars.empty[b], lane);
        }
      }
    }
    // The MMA warp issues nothing after the last group, so the tiles still hold T_{k/L}: C is one more read of each.
    if constexpr (MODE != 0) {
#pragma unroll 1
      for (int tb = 0; tb < TPW; tb++) load_pass<BN, true>(taddr(tb * LW + (int)wg), crow(tb * LW + (int)wg), sink);
    }
    if constexpr (MODE == 2) {
#pragma unroll
      for (int tb = 0; tb < TPW; tb++)
#pragma unroll
        for (int c = 0; c < NCH; c++)
          R[(size_t)(row * tiles_n + cn * NB + tb * LW + (int)wg) * NCH + c] = duo_finish(ch[tb][c]);
    } else if (wg == 0) {
      R[(size_t)row * tiles_n * NCH + cn * NB * NCH] = sink;
    }
  }
  tc_before_sync();
  __syncthreads();
  const long long t1 = clock64();
  if (threadIdx.x == 32 * MMA_WARP) {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const size_t cta = (size_t)blockIdx.y * gridDim.x + blockIdx.x;
    clk[3 * cta] = t0;
    clk[3 * cta + 1] = t1;
    clk[3 * cta + 2] = smid;
  }
  tc_after_sync();
  if (w == MMA_WARP) tmem_dealloc(tmem, 512);
#endif
}

// ---- host ------------------------------------------------------------------------------------------------------------------
using Kern = void (*)(const int8_t*, const int8_t*, int, int, int, int, uint32_t, int32_t*, uint32_t*, long long*);
struct Cfg {
  int nb, lw, mode;
  Kern k;
};
#define CFG(NB, LW)                                                                                   \
  {NB, LW, 0, pg100<NB, LW, 0>}, {NB, LW, 1, pg100<NB, LW, 1>}, {NB, LW, 2, pg100<NB, LW, 2>}, \
      {NB, LW, 3, pg100<NB, LW, 3>}
static const Cfg CFGS[] = {CFG(2, 1), CFG(2, 2), CFG(4, 1), CFG(4, 2)};
#undef CFG
static const Cfg* find(int nb, int lw, int mode) {
  for (const Cfg& c : CFGS)
    if (c.nb == nb && c.lw == lw && c.mode == mode) return &c;
  return nullptr;
}
static void prepare() {
  for (const Cfg& c : CFGS) CK(cudaFuncSetAttribute((const void*)c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
}

static int8_t code(uint32_t seed, uint32_t which, uint32_t i) {
  return (int8_t)((int)(dc_mix32(dc_mix32(seed ^ (which << 28)) + i) % 255u) - 127);
}
// K-periodic inputs: entry (i, kk) is the code of (i, kk mod PER).
static void make_inputs(int m, int n, int k, uint32_t seed, std::vector<int8_t>& X, std::vector<int8_t>& W) {
  X.resize((size_t)m * k);
  W.resize((size_t)n * k);
  for (int i = 0; i < m; i++)
    for (int kk = 0; kk < k; kk++) X[(size_t)i * k + kk] = code(seed, 1, (uint32_t)(i * PER + kk % PER));
  for (int j = 0; j < n; j++)
    for (int kk = 0; kk < k; kk++) W[(size_t)j * k + kk] = code(seed, 2, (uint32_t)(j * PER + kk % PER));
}
// The receipt by its definition (header), one row at a time.
static void host_ref(int m, int n, int k, int bn, int L, uint32_t seed, const std::vector<int8_t>& X,
                     const std::vector<int8_t>& W, std::vector<int32_t>& C, std::vector<uint32_t>& R) {
  const int tiles_n = n / bn;
  C.assign((size_t)m * n, 0);
  R.assign((size_t)m * tiles_n * NCH, 0);
  std::vector<int32_t> T(n);
  std::vector<DuoSt> ch(tiles_n * NCH);
  for (int i = 0; i < m; i++) {
    for (int tn = 0; tn < tiles_n; tn++)
      for (int c = 0; c < NCH; c++) ch[tn * NCH + c] = duo_init(seed, (uint32_t)((i * tiles_n + tn) * NCH + c));
    std::fill(T.begin(), T.end(), 0);
    for (int s = 0; s < k / L; s++) {
      for (int j = 0; j < n; j++) {
        int32_t acc = 0;
        for (int kk = s * L; kk < (s + 1) * L; kk++) acc += (int32_t)X[(size_t)i * k + kk] * W[(size_t)j * k + kk];
        T[j] = (int32_t)((uint32_t)T[j] + (uint32_t)acc);
      }
      for (int tn = 0; tn < tiles_n; tn++)
        for (int jl = 0; jl < bn; jl++) {
          DuoSt& st = ch[tn * NCH + jl % NCH];
          const uint32_t p = (uint32_t)T[tn * bn + jl];
          const int t = (jl / 16) % DUO_NPOS;
          const uint32_t u = st.y ^ p ^ duo_xor(t);
          st.y = u * duo_mul(t) + st.z;
          st.z = dc_rotl(u, duo_rot(t)) + st.y;
        }
    }
    for (int j = 0; j < n; j++) C[(size_t)i * n + j] = T[j];
    for (int q = 0; q < tiles_n * NCH; q++) R[(size_t)i * tiles_n * NCH + q] = duo_finish(ch[q]);
  }
}
template <class T>
static void dump(const std::string& dir, const char* name, const std::vector<T>& v) {
  FILE* f = fopen((dir + "/" + name).c_str(), "wb");
  if (!f) { fprintf(stderr, "ERR open %s/%s\n", dir.c_str(), name); exit(2); }
  fwrite(v.data(), sizeof(T), v.size(), f);
  fclose(f);
}

static int run_gate(const std::string& dir, int m, int n, int k, uint32_t seed, int lm, int nb, int lw) {
  const Cfg* cfg = find(nb, lw, 2);
  const int L = 32 * lm, bn = CTA_N / nb;
  if (!cfg || m % BM || n % CTA_N || k % L || k % PER) {
    fprintf(stderr, "ERR gate: (nb, lw) in (2,1) (2,2) (4,1) (4,2); m %% %d, n %% %d, k %% L and %% %d\n", BM, CTA_N, PER);
    return 2;
  }
  std::vector<int8_t> X, W;
  make_inputs(m, n, k, seed, X, W);
  std::vector<int32_t> Cr, C;
  std::vector<uint32_t> Rr, R;
  host_ref(m, n, k, bn, L, seed, X, W, Cr, Rr);
  int8_t *dX, *dW;
  int32_t* dC;
  uint32_t* dR;
  long long* dclk;
  const dim3 grid(n / CTA_N, m / BM);
  CK(cudaMalloc(&dX, X.size()));
  CK(cudaMalloc(&dW, W.size()));
  CK(cudaMalloc(&dC, Cr.size() * 4));
  CK(cudaMalloc(&dR, Rr.size() * 4));
  CK(cudaMalloc(&dclk, (size_t)grid.x * grid.y * 3 * 8));
  CK(cudaMemcpy(dX, X.data(), X.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0xA5, Cr.size() * 4));
  CK(cudaMemset(dR, 0xA5, Rr.size() * 4));
  cfg->k<<<grid, 32 * (4 * lw + 1), SMEM>>>(dX, dW, k, n, k, lm, seed, dC, dR, dclk);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  C.resize(Cr.size());
  R.resize(Rr.size());
  CK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(R.data(), dR, R.size() * 4, cudaMemcpyDeviceToHost));
  size_t bc = 0, br = 0;
  for (size_t i = 0; i < C.size(); i++) bc += C[i] != Cr[i];
  for (size_t i = 0; i < R.size(); i++) br += R[i] != Rr[i];
  dump(dir, "X.bin", X);
  dump(dir, "W.bin", W);
  dump(dir, "C.bin", C);
  dump(dir, "R.bin", R);
  FILE* f = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(f,
          "{\"m\": %d, \"n\": %d, \"k\": %d, \"bm\": %d, \"bn\": %d, \"seed\": %u, \"L\": %d, \"nch\": %d, \"period\": %d, "
          "\"nb\": %d, \"lw\": %d, \"arch\": \"sm_100a\", \"source\": \"gpu\", \"c_bad\": %zu, \"r_bad\": %zu}\n",
          m, n, k, BM, bn, seed, L, NCH, PER, nb, lw, bc, br);
  fclose(f);
  const bool ok = bc == 0 && br == 0;
  printf("{\"mode\": \"gate\", \"m\": %d, \"n\": %d, \"k\": %d, \"L\": %d, \"nb\": %d, \"lw\": %d, \"exact\": %s, "
         "\"c_bad\": %zu, \"r_bad\": %zu, \"words\": %zu, \"receipts\": %zu}\n",
         m, n, k, L, nb, lw, ok ? "true" : "false", bc, br, C.size(), R.size());
  CK(cudaFree(dX));
  CK(cudaFree(dW));
  CK(cudaFree(dC));
  CK(cudaFree(dR));
  CK(cudaFree(dclk));
  return ok ? 0 : 4;
}

static int run_rate(int reps, int ksteps) {
  int dev = 0, nsm = 0;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  const int lms[] = {1, 2, 4, 8};
  if (ksteps % 8) { fprintf(stderr, "ERR rate: KSTEPS a multiple of 8\n"); return 2; }
  const int m = BM * nsm, n = CTA_N, k = 32 * ksteps;
  std::vector<int8_t> X, W;
  make_inputs(m, n, PER, 7, X, W);
  int8_t *dX, *dW;
  int32_t* dC;
  uint32_t* dR;
  long long* dclk;
  CK(cudaMalloc(&dX, X.size()));
  CK(cudaMalloc(&dW, W.size()));
  CK(cudaMalloc(&dC, (size_t)m * n * 4));
  CK(cudaMalloc(&dR, (size_t)m * (n / 128) * NCH * 4));
  CK(cudaMalloc(&dclk, (size_t)nsm * 3 * 8));
  CK(cudaMemcpy(dX, X.data(), X.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  struct Point {
    const Cfg* c;
    int lm;
    std::vector<double> med, lo, hi, ms;
  };
  std::vector<Point> pts;
  for (const Cfg& c : CFGS)
    for (int lm : lms) pts.push_back({&c, lm, {}, {}, {}, {}});
  std::vector<long long> clk((size_t)nsm * 3);
  for (int r = 0; r <= reps; r++)
    for (Point& p : pts) {
      CK(cudaEventRecord(e0));
      p.c->k<<<dim3(1, nsm), 32 * (4 * p.c->lw + 1), SMEM>>>(dX, dW, PER, n, k, p.lm, 7, dC, dR, dclk);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      CK(cudaGetLastError());
      if (r == 0) continue;  // the first round warms up
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      CK(cudaMemcpy(clk.data(), dclk, clk.size() * 8, cudaMemcpyDeviceToHost));
      std::vector<double> cyc(nsm);
      for (int i = 0; i < nsm; i++) cyc[i] = (double)(clk[3 * i + 1] - clk[3 * i]);
      std::sort(cyc.begin(), cyc.end());
      p.med.push_back(cyc[nsm / 2]);
      p.lo.push_back(cyc[0]);
      p.hi.push_back(cyc[nsm - 1]);
      p.ms.push_back(ms);
    }
  for (Point& p : pts) {
    auto median = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    const double cyc = median(p.med), slots = (double)ksteps * BM * CTA_N, ms = median(p.ms);
    printf("{\"mode\": \"rate\", \"nb\": %d, \"lw\": %d, \"arm\": %d, \"L\": %d, \"ksteps\": %d, \"ctas\": %d, "
           "\"reps\": %d, \"cycles\": %.0f, \"cycles_lo\": %.0f, \"cycles_hi\": %.0f, \"ms\": %.4f, \"mhz\": %.1f, "
           "\"slots_per_clk\": %.3f, \"committed_words_per_clk\": %.3f, \"group_clk\": %.1f}\n",
           p.c->nb, p.c->lw, p.c->mode, 32 * p.lm, ksteps, nsm, reps, cyc, median(p.lo), median(p.hi), ms,
           cyc / (ms * 1e3), slots / cyc, slots / cyc / p.lm, cyc / (ksteps / p.lm));
  }
  return 0;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode != "gate" && mode != "rate") {
    fprintf(stderr, "usage: pgemm gate DIR M N K SEED LM NB LW | rate REPS KSTEPS\n");
    return 2;
  }
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  if (p.major != 10) { fprintf(stderr, "ERR device sm_%d%d: this binary runs on sm_100a\n", p.major, p.minor); return 2; }
  prepare();
  if (mode == "gate" && argc == 10)
    return run_gate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 10),
                    atoi(argv[7]), atoi(argv[8]), atoi(argv[9]));
  if (mode == "rate" && argc == 4) return run_rate(atoi(argv[2]), atoi(argv[3]));
  fprintf(stderr, "usage: pgemm gate DIR M N K SEED LM NB LW | rate REPS KSTEPS\n");
  return 2;
}
