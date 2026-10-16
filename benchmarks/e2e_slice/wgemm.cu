// EGEMM's edited GEMM (benchmarks/edited_gemm/egemm.cu, included unchanged) with checkpointed receipts: one receipt word a
// thread a window of S = 4 SW segments (SW stages of BK = 128), segment.PouwSegCheck_v2{S, duo}'s. Window w of tile t,
// thread c: the chains keyed from ((t NCT + c) NW + w) 8 + j (NW = K / 32 S; segment.init), then T_(w S - 1) absorbed (0 for
// w = 0) and T_(w S) .. T_(w S + S - 1), the running sums of C = X W'^T after each 32-step segment (EGEMM's absorbs, in its
// order), and R_w = xor_j (y_j ^ rotl(z_j, 16)) stored at R[(t NW + w) NCT + c] (the receipts' (tile, window, thread)
// layout: a 2 KiB row is one window of one tile). At a window's end the lagged absorbs are flushed, R_w stored, the chains
// re-keyed and the whole accumulator absorbed; the next stage's first atom then absorbs nothing of the previous atom's.
// Kernels w_l<LAG><tie>_s<S>[_r96] (EGEMM's e_l2q, e_l3qm with windows; _r96 at 96 registers).
// The swapped launch (arms s:, sedited:, seditedf:) gives the same kernel a = W' (n x k), b = X (m x k), pa = P_B and
// pb = P_A, with M = n and N = m: it stores Y^T and its receipts are over C^T = W' X^T, so W' = W + U is the a-side edit
// and X = floor(x / s) + b the b-side dither, as in PouwSegCheckDither_v2 (the peel's tail, U S^T, is the transpose of
// S U^T).
// Modes:
//   wgemm gate DIR M N K SEED       each w_ kernel once on poisoned buffers, unswapped and swapped: its output against
//                                   cuBLASLt's X W^T (the swapped one transposed), and x, W, X, W' and its receipts to DIR
//                                   for wgate.py (derive.win_receipts recomputes them)
//   wgemm il REPS ARM BASE M N K    egemm's il with the w_ kernels (ARM: [s:]w_NAME, [s]edited:w_NAME,
//                                   [s]editedf:w_NAME, or any of egemm's arms)
//   wgemm pipe REPS ARM BASE M N K PARAMS S   ARM's calls with their commitments beside the next call (cgpipe.cu; PARAMS
//                                   COMMIT-GPU's params file for 16,384-bit rows), against BASE. env: WG_RESERVE=k (the
//                                   GEMM's persistent grid on SMs - k), WG_GRID=tiles (a CTA a tile, not persistent;
//                                   or a number of CTAs, each taking every grid-th tile),
//                                   WG_PRIO=gemm (the GEMM's stream at the highest priority and the commitments' at the
//                                   lowest, so their CTAs go only where the GEMM's don't fit; default: the commitments'
//                                   at the highest), WG_GREEN=k (the commitments in a green context of k SMs, the arm's
//                                   GEMMs in one of the rest, the base on the whole device)
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a wgemm.cu cgpipe.cu -lcuda -lcublasLt
//        -lnvidia-ml
#define main egemm_main
#include "../edited_gemm/egemm.cu"
#undef main

template <int LAG, int TIE, int SW> struct WL : EL<LAG, TIE, true> {
  using EB = EL<LAG, TIE, true>;
  using EB::EB;
  static constexpr int MI = EB::MI, NI = EB::NI, NCT = EB::NCT;
  // EL::atom with FRESH: a window's first atom, which absorbs none of the previous atom's tiles (the boundary did)
  template <bool FIRST, bool FRESH, int IDX = 0>
  DEV static void atom(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&fa)[MI][4], const uint32_t (&fb)[NI][2], uint32_t zero) {
    if constexpr (IDX < MI * NI) {
      constexpr int mi = IDX / NI, ni = IDX % NI, P = IDX - LAG;
      if constexpr (FIRST) {
        imma0(acc[mi][ni], fa[mi], fb[ni]);
      } else {
        if constexpr (TIE != 0 && IDX % (TIE > 0 ? TIE : -TIE) == 0) {
          constexpr int TP = (IDX - 1 - LAG + 2 * MI * NI) % (MI * NI);
          tie<(TIE < 0)>(acc[mi][ni][0], ch[4 * (TP % 2) + 3].z, zero);
        }
        imma(acc[mi][ni], fa[mi], fb[ni]);
      }
      if constexpr (P >= 0) EB::template ab<P>(ch, acc);
      else if constexpr (!FIRST && !FRESH) EB::template ab<P + MI * NI>(ch, acc);
      atom<FIRST, FRESH, IDX + 1>(acc, ch, fa, fb, zero);
    }
  }
  template <bool FIRST, bool FRESH, bool LAST>
  DEV void wstage(int (&acc)[MI][NI][4], St (&ch)[8], uint32_t (&fa)[2][MI][4], uint32_t (&fb)[2][NI][2], uint32_t zero) {
    const uint32_t sb = this->stage(this->s), done = this->empty(this->s);
    this->load(fa[1], fb[1], sb, 1);
    atom<FIRST, FRESH>(acc, ch, fa[0], fb[0], zero);
    this->load(fa[0], fb[0], sb, 2);
    atom<false, false>(acc, ch, fa[1], fb[1], zero);
    this->load(fa[1], fb[1], sb, 3);
    atom<false, false>(acc, ch, fa[0], fb[0], zero);
    this->advance();
    if (!LAST) {
      mbar_wait_warp(this->full(this->s), this->ph);
      this->load(fa[0], fb[0], this->stage(this->s), 0);
    }
    atom<false, false>(acc, ch, fa[1], fb[1], zero);
    this->release(done);
  }
  DEV static void key(St (&ch)[8], uint32_t seed, uint32_t b) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      ch[j].y = mix32(seed ^ mix32(b + j));
      ch[j].z = mix32(ch[j].y + 0x9E3779B9u);
    }
  }
  DEV static uint32_t fold(const St (&ch)[8]) {
    uint32_t r = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) r ^= finish<DUO>(ch, j);
    return r;
  }
  // the end of window w: its lagged absorbs, R_w, the chains keyed for w + 1 (kb: window 0's key index), T_((w + 1) S - 1)
  DEV static void boundary(int (&acc)[MI][NI][4], St (&ch)[8], uint32_t* rp, int w, uint32_t seed, uint32_t kb) {
    EB::template flush<MI * NI - LAG>(acc, ch);
    rp[(size_t)w * NCT] = fold(ch);
    key(ch, seed, kb + 8u * (w + 1));
    EB::template flush<0>(acc, ch);
  }
  DEV void tile_w(int K, int (&acc)[MI][NI][4], St (&ch)[8], uint32_t zero, uint32_t* rp, uint32_t seed, uint32_t kb) {
    const int KT = K / EB::BK;
    uint32_t fa[2][MI][4], fb[2][NI][2];
    mbar_wait_warp(this->full(this->s), this->ph);
    this->load(fa[0], fb[0], this->stage(this->s), 0);
    wstage<true, false, false>(acc, ch, fa, fb, zero);
#pragma unroll 1
    for (int kt = 1; kt < KT - 1; kt++) {
      if (kt % SW == 0) {
        boundary(acc, ch, rp, kt / SW - 1, seed, kb);
        wstage<false, true, false>(acc, ch, fa, fb, zero);
      } else {
        wstage<false, false, false>(acc, ch, fa, fb, zero);
      }
    }
    if ((KT - 1) % SW == 0) {
      boundary(acc, ch, rp, (KT - 1) / SW - 1, seed, kb);
      wstage<false, true, true>(acc, ch, fa, fb, zero);
    } else {
      wstage<false, false, true>(acc, ch, fa, fb, zero);
    }
    EB::template flush<MI * NI - LAG>(acc, ch);
    rp[(size_t)(KT / SW - 1) * NCT] = fold(ch);
  }
};

template <int LAG, int TIE, int SW> DEV void w_body(const Args& a) {
  using L = WL<LAG, TIE, SW>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps, a.M, a.N, a.K, a.group, true, 1);
  ml.pa_rows = a.pa_rows;
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int tiles_m = a.M / 128, tiles_n = a.N / 128, nw = a.K / (L::BK * SW);
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  const uint32_t zero = (uint32_t)a.KR;
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn, bi;
    ml.tile_of(t, bi, tm, tn);
    const uint32_t tile_id = tm * tiles_n + tn, kb = (tile_id * L::NCT + ctid) * (uint32_t)nw * 8u;
    St ch[8];
    L::key(ch, a.seed, kb);
    {
      int z0[MI][NI][4] = {};
      L::EB::template flush<0>(z0, ch);
    }
    int acc[MI][NI][4];
    ml.tile_w(a.K, acc, ch, zero, a.R + (size_t)tile_id * nw * L::NCT + ctid, a.seed, kb);
    ml.peel2(acc);
    const int row0 = tm * 128 + wm * 32 + g, col0 = tn * 128 + wn * 32 + 2 * q;
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int* p = a.U + (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni;
        *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
        *reinterpret_cast<int2*>(p + (size_t)8 * a.N) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
      }
  }
}
#define WGEMM(NAME, LAG, TIE, SW) \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ Args a) { w_body<LAG, TIE, SW>(a); }
WGEMM(w_l2q_s4, 2, 4, 1)
WGEMM(w_l2q_s8, 2, 4, 2)
WGEMM(w_l2q_s16, 2, 4, 4)
WGEMM(w_l2q_s32, 2, 4, 8)
WGEMM(w_l3qm_s8, 3, -4, 2)
WGEMM(w_l3qm_s16, 3, -4, 4)
// _r96: registers capped at 96 (bounds of 640 threads; launched with 512), so a 128-thread CTA of the committer's small
// path (at most 102 registers) fits on the SM beside the GEMM's CTA
#define WGEMM96(NAME, LAG, TIE, SW) \
  __global__ void __launch_bounds__(MLW::NT + 128, 1) NAME(const __grid_constant__ Args a) { w_body<LAG, TIE, SW>(a); }
WGEMM96(w_l3qm_s16_r96, 3, -4, 4)
WGEMM96(w_l2q_s16_r96, 2, 4, 4)

struct WK { const char* name; Kern k; int S; };
static const WK WKS[] = {{"w_l2q_s4", w_l2q_s4, 4},           {"w_l2q_s8", w_l2q_s8, 8},
                         {"w_l2q_s16", w_l2q_s16, 16},        {"w_l2q_s32", w_l2q_s32, 32},
                         {"w_l3qm_s8", w_l3qm_s8, 8},         {"w_l3qm_s16", w_l3qm_s16, 16},
                         {"w_l3qm_s16_r96", w_l3qm_s16_r96, 16}, {"w_l2q_s16_r96", w_l2q_s16_r96, 16}};
static uint32_t* g_rw = nullptr;   // the windowed receipts, room for S = 4: (m / 128)(n / 128) 512 (k / 128) words
static size_t rw_words(const EP& P, int S) { return (size_t)(P.m / 128) * (P.n / 128) * 512 * (P.k / (32 * S)); }
static Maps *g_mS = nullptr, *g_mSF = nullptr;   // the swapped launch's maps over W', X, P_B, P_A (and qwe's fused ones)
static void wsetup(EP& P, int m, int n, int k, uint32_t seed, bool gate) {
  esetup(P, m, n, k, seed, gate);
  CK(cudaMalloc(&g_rw, rw_words(P, 4) * 4));
  for (auto& w : WKS) {
    if (k % (32 * w.S)) { fprintf(stderr, "k %d: not whole windows of %d segments\n", k, w.S); exit(2); }
    CK(cudaFuncSetAttribute(w.k, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  }
  g_mS = upload_maps(Maps{tmap(P.Wp, k, n, 128), tmap(P.X, k, m, 128), tmap(P.PB, 128, n, 128), tmap(P.PA, 128, m, 128)});
  g_mSF = upload_maps(Maps{tmap(P.Wf, k, n, 128), tmap(P.Xf, k, m, 128), tmap(P.PBf, 128, n, 128), tmap(P.PAf, 128, m, 128)});
}
static void wfree(EP& P) {
  CK(cudaDeviceSynchronize());
  CK(cudaFree(g_rw)); CK(cudaFree(g_mS)); CK(cudaFree(g_mSF));
  efree(P);
}
// the swapped launch: Y^T (n x m) to Y, receipts over C^T
static void srun(EP& P, Kern kern, const Maps* maps) {
  Args a{maps, P.Y, g_rw, P.n, P.m, P.k, 8, P.seed, 0, 1, 0, 0, {0, 0, 0, 0}, nullptr, 0};
  kern<<<P.grid, MLW::NT, CSMEM, P.st>>>(a);
}
static const WK* wfind(const std::string& n) {
  for (auto& w : WKS) if (n == w.name) return &w;
  return nullptr;
}
// an arm's call: its producers (pre: W's edit, or the fused quantizer and edit; none for egemm's arms) and its GEMM
struct WArm { std::function<void(EP&)> pre, run; };
static WArm warm2(const std::string& a) {
  for (const char* pre : {"", "edited:", "editedf:", "s:", "sedited:", "seditedf:"}) {
    const std::string p = pre;
    if (a.rfind(p, 0) != 0) continue;
    const WK* w = wfind(a.substr(p.size()));
    if (!w) continue;
    Kern k = w->k;
    const bool sw = p[0] == 's', fused = p.find("editedf:") != std::string::npos;
    std::function<void(EP&)> e;
    if (fused) e = [](EP& P) { qf(P, sigma(P.seed, P.inst++)); };
    else if (p.find("edited:") != std::string::npos) e = [](EP& P) { we(P, sigma(P.seed, P.inst++)); };
    if (sw) return {e, [k, fused](EP& P) { srun(P, k, fused ? g_mSF : g_mS); }};
    return {e, [k, fused](EP& P) { egemm_run(P, k, P.Y, g_rw, fused ? P.mF : nullptr); }};
  }
  return {nullptr, earm(a)};
}
static std::function<void(EP&)> warm(const std::string& a) {
  const WArm r = warm2(a);
  if (!r.pre) return r.run;
  return [r](EP& P) { r.pre(P); r.run(P); };
}
__global__ void tcmp(const int* __restrict__ Yt, const int* __restrict__ Y, int m, int n, unsigned long long* bad) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)m * n) return;
  if (Yt[(i % n) * m + i / n] != Y[i]) atomicAdd(bad, 1ull);
}
static unsigned long long tcount(EP& P) {
  CK(cudaMemsetAsync(P.bad, 0, 8, P.st));
  tcmp<<<(unsigned)((P.ny + 255) / 256), 256, 0, P.st>>>(P.Y, P.Yr, P.m, P.n, P.bad);
  unsigned long long b;
  CK(cudaMemcpyAsync(&b, P.bad, 8, cudaMemcpyDeviceToHost, P.st));
  CK(cudaStreamSynchronize(P.st));
  return b;
}

static int wgate(const std::string& dir, int m, int n, int k, uint32_t seed) {
  EP P;
  wsetup(P, m, n, k, seed, true);
  const size_t xb = (size_t)m * k, wb = (size_t)n * k;
  for (auto [p, b] : {std::pair<void*, size_t>{P.X, xb}, {P.Wp, wb}, {P.PA, (size_t)m * 128}, {P.PB, (size_t)n * 128},
                      {P.Y, P.ny * 4}, {P.Yr, P.ny * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, P.st));
  const Sig s = sigma(seed, 0);
  qd(P, s);
  we(P, s);
  ltg_run(P.plain, P.st);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  dump(dir, "X.bin", P.X, xb, true);
  dump(dir, "Wp.bin", P.Wp, wb, true);
  dump(dir, "x.bin", P.x, xb * 2, true);
  dump(dir, "W.bin", P.W, wb, true);
  bool ok = true;
  std::string res;
  for (const bool sw : {false, true})
    for (auto& w : WKS) {
      CK(cudaMemsetAsync(P.Y, 0xA5, P.ny * 4, P.st));
      CK(cudaMemsetAsync(g_rw, 0xA5, rw_words(P, 4) * 4, P.st));
      if (sw) srun(P, w.k, g_mS);
      else egemm_run(P, w.k, P.Y, g_rw);
      CK(cudaStreamSynchronize(P.st));
      CK(cudaGetLastError());
      const unsigned long long y = sw ? tcount(P) : ecmp(P, P.Y, P.Yr, P.ny);
      const std::string name = (sw ? "s_" : "") + std::string(w.name);
      dump(dir, ("R_" + name + ".bin").c_str(), g_rw, rw_words(P, w.S) * 4, true);
      res += (res.empty() ? "" : ", ") + std::string("\"") + name + "\": {\"S\": " + std::to_string(w.S) + ", \"swap\": " +
             (sw ? "true" : "false") + ", \"y_bad\": " + std::to_string(y) + "}";
      ok &= y == 0;
    }
  uint32_t inv_bits;
  memcpy(&inv_bits, &P.inv_s, 4);
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"inv_s_bits\": %u, \"device\": {%s}, \"device_ok\": %s}\n", m, n, k,
          seed, inv_bits, res.c_str(), ok ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"device_ok\": %s, \"device\": {%s}}\n", dir.c_str(), m, n, k,
         ok ? "true" : "false", res.c_str());
  wfree(P);
  return ok ? 0 : 1;
}

// egemm's eil with warm's arms
static int wil(int reps, const std::string& arm, const std::string& base, int m, int n, int k) {
  Nv nv;
  EP P;
  wsetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
  auto fa = warm(arm), fb = warm(base);
  const bool cold = getenv("EG_COLD") && atoi(getenv("EG_COLD"));
  const size_t flb = (size_t)512 << 20;
  void* fl = nullptr;
  if (cold) CK(cudaMalloc(&fl, flb));
  cudaEvent_t e[4];
  for (auto& x : e) CK(cudaEventCreate(&x));
  auto call = [&](const std::function<void(EP&)>& f, cudaEvent_t a, cudaEvent_t b, int r) {
    if (cold) CK(cudaMemsetAsync(fl, r & 0xFF, flb, P.st));
    CK(cudaEventRecord(a, P.st));
    f(P);
    CK(cudaEventRecord(b, P.st));
  };
  std::vector<double> ta, tb, mz;
  for (int r = -5; r < reps; r++) {
    const bool afirst = (r & 1) != 0;
    call(afirst ? fa : fb, e[0], e[1], 2 * r);
    call(afirst ? fb : fa, e[2], e[3], 2 * r + 1);
    CK(cudaEventSynchronize(e[3]));
    float t0, t1;
    CK(cudaEventElapsedTime(&t0, e[0], e[1]));
    CK(cudaEventElapsedTime(&t1, e[2], e[3]));
    if (r < 0) continue;
    ta.push_back(afirst ? t0 : t1);
    tb.push_back(afirst ? t1 : t0);
    mz.push_back(nv.sm());
  }
  CK(cudaGetLastError());
  const double a = median(ta), b = median(tb);
  printf("{\"il\": \"%s\", \"base\": \"%s\", \"cold\": %s, \"m\": %d, \"n\": %d, \"k\": %d, \"reps\": %d, \"ms\": %.6f, "
         "\"base_ms\": %.6f, \"ratio\": %.4f, \"mhz\": %.0f, \"mhz_min\": %.0f, \"mhz_max\": %.0f}\n",
         arm.c_str(), base.c_str(), cold ? "true" : "false", m, n, k, reps, a, b, a / b, median(mz),
         *std::min_element(mz.begin(), mz.end()), *std::max_element(mz.begin(), mz.end()));
  fflush(stdout);
  for (auto& x : e) CK(cudaEventDestroy(x));
  if (fl) CK(cudaFree(fl));
  wfree(P);
  return 0;
}

// cgpipe.cu
void* cgp_new(const char* params, const uint8_t* X, uint64_t m, uint64_t k, const uint8_t* R, uint64_t rows);
void cgp_salts(void* h, cudaStream_t st);
void cgp_data(void* h, cudaStream_t st);
bool cgp_small();

// a call with its commitments, the slice's GPU side: 8 calls of ARM on st, each call's commitments (X's tile tree and its
// receipts' tree, S segments a window: salts, then data) on a stream at the device's highest priority once its GEMM has
// ended, so they run beside the next call's GEMM (pipe); against BASE alone, ARM alone, the commitments alone, and ARM then
// its commitments on one stream (serial). The commitments read X and the receipts while the next call rewrites them: a
// timing, not a commitment of each call's own values.
static void cu(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* s = "?";
  cuGetErrorString(r, &s);
  fprintf(stderr, "%s: %s\n", what, s);
  exit(3);
}
// a context's side of the pipe: its stream and the events its stream records (an event is recorded only by a stream of
// its own context; a wait may cross contexts)
struct Side {
  CUcontext ctx = nullptr;
  cudaStream_t st = nullptr;
  int sms = 0;
  cudaEvent_t e0, e1, ej;
  std::vector<cudaEvent_t> ev;
  void use() const { if (ctx) cu(cuCtxSetCurrent(ctx), "cuCtxSetCurrent"); }
  void events(int calls) {
    use();
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreateWithFlags(&ej, cudaEventDisableTiming));
    ev.resize(calls);
    for (auto& x : ev) CK(cudaEventCreateWithFlags(&x, cudaEventDisableTiming));
  }
};
// the device's SMs split into a green context of k (rounded up as the split requires) for the commitments and one of the
// rest for the GEMMs, each with a stream (the commitments' at priority prio)
static void green_split(int k, Side& gemm, Side& commit, int prio) {
  CUdevice dev;
  cu(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUdevResource all, grp, rest;
  unsigned nb = 1;
  cu(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource");
  cu(cuDevSmResourceSplitByCount(&grp, &nb, &all, &rest, 0, (unsigned)k), "cuDevSmResourceSplitByCount");
  Side* sd[2] = {&commit, &gemm};
  CUdevResource* rs[2] = {&grp, &rest};
  for (int i = 0; i < 2; i++) {
    CUdevResourceDesc d;
    CUgreenCtx g;
    CUstream s;
    cu(cuDevResourceGenerateDesc(&d, rs[i], 1), "cuDevResourceGenerateDesc");
    cu(cuGreenCtxCreate(&g, d, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate");
    cu(cuCtxFromGreenCtx(&sd[i]->ctx, g), "cuCtxFromGreenCtx");
    cu(cuGreenCtxStreamCreate(&s, g, CU_STREAM_NON_BLOCKING, i == 0 ? prio : 0), "cuGreenCtxStreamCreate");
    sd[i]->st = reinterpret_cast<cudaStream_t>(s);
    sd[i]->sms = (int)rs[i]->sm.smCount;
  }
}
static int wpipe(int reps, const std::string& arm, const std::string& base, int m, int n, int k, const char* params, int S) {
  Nv nv;
  EP P;
  wsetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const int tiles = (m / 128) * (n / 128), reserve = getenv("WG_RESERVE") ? atoi(getenv("WG_RESERVE")) : 0;
  const int green = getenv("WG_GREEN") ? atoi(getenv("WG_GREEN")) : 0;
  const char* wg = getenv("WG_GRID");
  const bool per_tile = wg != nullptr;
  const bool gemm_first = getenv("WG_PRIO") && !strcmp(getenv("WG_PRIO"), "gemm");
  if (reserve < 0 || reserve >= nsm) { fprintf(stderr, "WG_RESERVE %d of %d SMs\n", reserve, nsm); exit(2); }
  P.grid = !wg ? std::min(tiles, nsm - reserve) : !strcmp(wg, "tiles") ? tiles : std::max(1, std::min(tiles, atoi(wg)));
  auto fa = warm(arm), fb = warm(base);
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  constexpr int calls = 8;
  // pb: the base's side (the primary context, the whole device); pa the arm's GEMMs'; pc the commitments'
  Side pb, pa, pc;
  cu(cuCtxGetCurrent(&pb.ctx), "cuCtxGetCurrent");
  if (gemm_first) {
    CK(cudaStreamSynchronize(P.st));
    CK(cudaStreamDestroy(P.st));
    CK(cudaStreamCreateWithPriority(&P.st, cudaStreamDefault, hi));
  }
  pb.st = P.st;
  pb.sms = nsm;
  if (green > 0) {
    green_split(green, pa, pc, hi);
    if (!per_tile) P.grid = std::min(tiles, pa.sms);
  } else {
    pa = pb;
    pc.ctx = pb.ctx;
    pc.sms = nsm;
    CK(cudaStreamCreateWithPriority(&pc.st, cudaStreamNonBlocking, gemm_first ? lo : hi));
  }
  pc.use();
  void* h = cgp_new(params, reinterpret_cast<const uint8_t*>(P.X), m, k, reinterpret_cast<const uint8_t*>(g_rw), rw_words(P, S) / 512);
  pb.events(calls);
  pa.events(calls);
  pc.events(calls);
  auto timed = [&](const Side& sd, const std::function<void()>& f) {
    sd.use();
    CK(cudaEventRecord(sd.e0, sd.st));
    f();
    sd.use();
    CK(cudaEventRecord(sd.e1, sd.st));
    CK(cudaEventSynchronize(sd.e1));
    float t;
    CK(cudaEventElapsedTime(&t, sd.e0, sd.e1));
    return (double)t / calls;
  };
  auto on = [&](const Side& sd, const std::function<void(EP&)>& f) {
    sd.use();
    const cudaStream_t s0 = P.st;
    P.st = sd.st;
    f(P);
    P.st = s0;
  };
  auto commit = [&](cudaStream_t st) { pc.use(); cgp_salts(h, st); cgp_data(h, st); };
  // the arm's call i, then its commitments on pc after it (serial: pa then waits for them)
  auto call = [&](int i, bool wait) {
    on(pa, fa);
    CK(cudaEventRecord(pa.ev[i], pa.st));
    CK(cudaStreamWaitEvent(pc.st, pa.ev[i]));
    commit(pc.st);
    if (wait) {
      CK(cudaEventRecord(pc.ev[i], pc.st));
      pa.use();
      CK(cudaStreamWaitEvent(pa.st, pc.ev[i]));
    }
  };
  auto base_only = [&] { for (int i = 0; i < calls; i++) on(pb, fb); };
  auto arm_only = [&] { for (int i = 0; i < calls; i++) on(pa, fa); };
  auto commits = [&] { for (int i = 0; i < calls; i++) commit(pc.st); };
  auto serial = [&] { for (int i = 0; i < calls; i++) call(i, true); };
  // GEMM-first: call i - 1's commitments wait for call i's producers too, so they're released with call i's GEMM, whose
  // CTAs then go first (a producer's small blocks would otherwise let the commitments' CTAs onto every SM)
  const WArm A = warm2(arm);
  auto release = [&](int i) {
    CK(cudaEventRecord(pa.ev[i], pa.st));
    CK(cudaStreamWaitEvent(pc.st, pa.ev[i]));
    commit(pc.st);
  };
  auto piped = [&] {
    if (gemm_first && !green) {
      for (int i = 0; i < calls; i++) {
        if (A.pre) on(pa, A.pre);
        pa.use();
        if (i > 0) release(i - 1);
        on(pa, A.run);
      }
      pa.use();
      release(calls - 1);
    } else {
      for (int i = 0; i < calls; i++) call(i, false);
    }
    pc.use();
    CK(cudaEventRecord(pc.ej, pc.st));
    pa.use();
    CK(cudaStreamWaitEvent(pa.st, pc.ej));
  };
  std::vector<double> tb, ta, tc, ts, tp, mz;
  for (int r = -2; r < reps; r++) {
    const double b = timed(pb, base_only), a = timed(pa, arm_only), c = timed(pc, commits), s = timed(pa, serial),
                 p = timed(pa, piped);
    if (r < 0) continue;
    tb.push_back(b); ta.push_back(a); tc.push_back(c); ts.push_back(s); tp.push_back(p);
    mz.push_back(nv.sm());
  }
  pb.use();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  const double b = median(tb), a = median(ta), c = median(tc), s = median(ts), p = median(tp);
  printf("{\"pipe\": \"%s\", \"base\": \"%s\", \"S\": %d, \"m\": %d, \"n\": %d, \"k\": %d, \"calls\": %d, \"reps\": %d, "
         "\"reserve\": %d, \"green\": %d, \"grid\": %d, \"sms\": %d, \"gemm_sms\": %d, \"commit_sms\": %d, \"prio\": \"%s\", "
         "\"small\": %s, \"receipt_mib\": %.1f, \"base_ms\": %.6f, \"arm_ms\": %.6f, \"commit_ms\": %.6f, \"serial_ms\": %.6f, "
         "\"pipe_ms\": %.6f, \"arm_ratio\": %.4f, \"serial_ratio\": %.4f, \"pipe_ratio\": %.4f, \"mhz\": %.0f, \"mhz_min\": %.0f}\n",
         arm.c_str(), base.c_str(), S, m, n, k, calls, reps, per_tile ? -1 : reserve, green, P.grid, nsm, pa.sms, pc.sms,
         gemm_first ? "gemm" : "commit", cgp_small() ? "true" : "false", rw_words(P, S) * 4.0 / (1 << 20), b, a, c, s, p, a / b,
         s / b, p / b, median(mz), *std::min_element(mz.begin(), mz.end()));
  fflush(stdout);
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return wgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 8 && !strcmp(argv[1], "il")) return wil(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
  if (argc >= 10 && !strcmp(argv[1], "pipe"))
    return wpipe(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), argv[8], atoi(argv[9]));
  fprintf(stderr, "usage: wgemm gate DIR M N K SEED | il REPS ARM BASE M N K | pipe REPS ARM BASE M N K PARAMS S\n");
  return 2;
}
