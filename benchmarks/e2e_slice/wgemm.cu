// EGEMM's edited GEMM (benchmarks/edited_gemm/egemm.cu, included unchanged) with checkpointed receipts: one receipt word a
// thread a window of S = 4 SW segments (SW stages of BK = 128), segment.PouwSegCheck_v2{S, duo}'s. Window w of tile t,
// thread c: the chains keyed from ((t NCT + c) NW + w) 8 + j (NW = K / 32 S; segment.init), then T_(w S - 1) absorbed (0 for
// w = 0) and T_(w S) .. T_(w S + S - 1), the running sums of C = X W'^T after each 32-step segment (EGEMM's absorbs, in its
// order), and R_w = xor_j (y_j ^ rotl(z_j, 16)) stored at R[(t NW + w) NCT + c] (the receipts' (tile, window, thread)
// layout: a 2 KiB row is one window of one tile). At a window's end the lagged absorbs are flushed, R_w stored, the chains
// re-keyed and the whole accumulator absorbed; the next stage's first atom then absorbs nothing of the previous atom's.
// Kernels w_l<LAG><tie>_s<S> (EGEMM's e_l2q, e_l3qm with windows).
// Modes:
//   wgemm gate DIR M N K SEED       each w_ kernel once on poisoned buffers: its output against cuBLASLt's X W^T, and X, W'
//                                   and its receipts to DIR for wgate.py (derive.win_receipts recomputes them)
//   wgemm il REPS ARM BASE M N K    egemm's il with the w_ kernels (ARM: w_NAME, edited:w_NAME, editedf:w_NAME, or any of
//                                   egemm's arms)
//   wgemm pipe REPS ARM BASE M N K PARAMS S   ARM's calls with their commitments beside the next call (cgpipe.cu; PARAMS
//                                   COMMIT-GPU's params file for 16,384-bit rows), against BASE
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

struct WK { const char* name; Kern k; int S; };
static const WK WKS[] = {{"w_l2q_s4", w_l2q_s4, 4}, {"w_l2q_s8", w_l2q_s8, 8}, {"w_l2q_s16", w_l2q_s16, 16},
                         {"w_l2q_s32", w_l2q_s32, 32}, {"w_l3qm_s8", w_l3qm_s8, 8}, {"w_l3qm_s16", w_l3qm_s16, 16}};
static uint32_t* g_rw = nullptr;   // the windowed receipts, room for S = 4: (m / 128)(n / 128) 512 (k / 128) words
static size_t rw_words(const EP& P, int S) { return (size_t)(P.m / 128) * (P.n / 128) * 512 * (P.k / (32 * S)); }
static void wsetup(EP& P, int m, int n, int k, uint32_t seed, bool gate) {
  esetup(P, m, n, k, seed, gate);
  CK(cudaMalloc(&g_rw, rw_words(P, 4) * 4));
  for (auto& w : WKS) {
    if (k % (32 * w.S)) { fprintf(stderr, "k %d: not whole windows of %d segments\n", k, w.S); exit(2); }
    CK(cudaFuncSetAttribute(w.k, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  }
}
static const WK* wfind(const std::string& n) {
  for (auto& w : WKS) if (n == w.name) return &w;
  return nullptr;
}
static std::function<void(EP&)> warm(const std::string& a) {
  for (const char* pre : {"", "edited:", "editedf:"}) {
    const std::string p = pre;
    if (a.rfind(p, 0) != 0) continue;
    const WK* w = wfind(a.substr(p.size()));
    if (!w) continue;
    Kern k = w->k;
    if (p.empty()) return [k](EP& P) { egemm_run(P, k, P.Y, g_rw); };
    if (p == "edited:") return [k](EP& P) { we(P, sigma(P.seed, P.inst++)); egemm_run(P, k, P.Y, g_rw); };
    return [k](EP& P) { qf(P, sigma(P.seed, P.inst++)); egemm_run(P, k, P.Y, g_rw, P.mF); };
  }
  return earm(a);
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
  bool ok = true;
  std::string res;
  for (auto& w : WKS) {
    CK(cudaMemsetAsync(P.Y, 0xA5, P.ny * 4, P.st));
    CK(cudaMemsetAsync(g_rw, 0xA5, rw_words(P, 4) * 4, P.st));
    egemm_run(P, w.k, P.Y, g_rw);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    const unsigned long long y = ecmp(P, P.Y, P.Yr, P.ny);
    dump(dir, (std::string("R_") + w.name + ".bin").c_str(), g_rw, rw_words(P, w.S) * 4, true);
    res += (res.empty() ? "" : ", ") + std::string("\"") + w.name + "\": {\"S\": " + std::to_string(w.S) + ", \"y_bad\": " + std::to_string(y) + "}";
    ok &= y == 0;
  }
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"device\": {%s}, \"device_ok\": %s}\n", m, n, k, seed, res.c_str(),
          ok ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"device_ok\": %s, \"device\": {%s}}\n", dir.c_str(), m, n, k,
         ok ? "true" : "false", res.c_str());
  CK(cudaFree(g_rw));
  efree(P);
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
  CK(cudaFree(g_rw));
  efree(P);
  return 0;
}

// cgpipe.cu
void* cgp_new(const char* params, const uint8_t* X, uint64_t m, uint64_t k, const uint8_t* R, uint64_t rows);
void cgp_salts(void* h, cudaStream_t st);
void cgp_data(void* h, cudaStream_t st);

// a call with its commitments, the slice's GPU side: 8 calls of ARM on st, each call's commitments (X's tile tree and its
// receipts' tree, S segments a window: salts, then data) on a stream at the device's highest priority once its GEMM has
// ended, so they run beside the next call's GEMM (pipe); against BASE alone, ARM alone, the commitments alone, and ARM then
// its commitments on one stream (serial). The commitments read X and the receipts while the next call rewrites them: a
// timing, not a commitment of each call's own values.
static int wpipe(int reps, const std::string& arm, const std::string& base, int m, int n, int k, const char* params, int S) {
  Nv nv;
  EP P;
  wsetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
  auto fa = warm(arm), fb = warm(base);
  void* h = cgp_new(params, reinterpret_cast<const uint8_t*>(P.X), m, k, reinterpret_cast<const uint8_t*>(g_rw), rw_words(P, S) / 512);
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t sp;
  CK(cudaStreamCreateWithPriority(&sp, cudaStreamNonBlocking, hi));
  constexpr int calls = 8;
  std::vector<cudaEvent_t> ev(calls);
  for (auto& x : ev) CK(cudaEventCreateWithFlags(&x, cudaEventDisableTiming));
  cudaEvent_t e0, e1, ej;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreateWithFlags(&ej, cudaEventDisableTiming));
  auto timed = [&](const std::function<void()>& f) {
    CK(cudaEventRecord(e0, P.st));
    f();
    CK(cudaEventRecord(e1, P.st));
    CK(cudaEventSynchronize(e1));
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    return (double)t / calls;
  };
  auto base_only = [&] { for (int i = 0; i < calls; i++) fb(P); };
  auto arm_only = [&] { for (int i = 0; i < calls; i++) fa(P); };
  auto commits = [&] { for (int i = 0; i < calls; i++) { cgp_salts(h, P.st); cgp_data(h, P.st); } };
  auto serial = [&] { for (int i = 0; i < calls; i++) { fa(P); cgp_salts(h, P.st); cgp_data(h, P.st); } };
  auto piped = [&] {
    CK(cudaEventRecord(ej, P.st));
    CK(cudaStreamWaitEvent(sp, ej));
    for (int i = 0; i < calls; i++) {
      fa(P);
      CK(cudaEventRecord(ev[i], P.st));
      CK(cudaStreamWaitEvent(sp, ev[i]));
      cgp_salts(h, sp);
      cgp_data(h, sp);
    }
    CK(cudaEventRecord(ej, sp));
    CK(cudaStreamWaitEvent(P.st, ej));
  };
  std::vector<double> tb, ta, tc, ts, tp, mz;
  for (int r = -2; r < reps; r++) {
    const double b = timed(base_only), a = timed(arm_only), c = timed(commits), s = timed(serial), p = timed(piped);
    if (r < 0) continue;
    tb.push_back(b); ta.push_back(a); tc.push_back(c); ts.push_back(s); tp.push_back(p);
    mz.push_back(nv.sm());
  }
  CK(cudaGetLastError());
  const double b = median(tb), a = median(ta), c = median(tc), s = median(ts), p = median(tp);
  printf("{\"pipe\": \"%s\", \"base\": \"%s\", \"S\": %d, \"m\": %d, \"n\": %d, \"k\": %d, \"calls\": %d, \"reps\": %d, "
         "\"receipt_mib\": %.1f, \"base_ms\": %.6f, \"arm_ms\": %.6f, \"commit_ms\": %.6f, \"serial_ms\": %.6f, \"pipe_ms\": %.6f, "
         "\"arm_ratio\": %.4f, \"serial_ratio\": %.4f, \"pipe_ratio\": %.4f, \"mhz\": %.0f, \"mhz_min\": %.0f}\n",
         arm.c_str(), base.c_str(), S, m, n, k, calls, reps, rw_words(P, S) * 4.0 / (1 << 20), b, a, c, s, p, a / b, s / b, p / b,
         median(mz), *std::min_element(mz.begin(), mz.end()));
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
