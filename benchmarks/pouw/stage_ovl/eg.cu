// ovl.cu's edited arms (eg.h): EGEMM's edited GEMM (benchmarks/pouw/edited_gemm/egemm.cu, cursor/egemm-ea24@63f78c35a) with
// its windowed duo receipts, and the receipts' commitment under commit_gpu's small path.
//
// The GEMM is hashed_gemm's w_d (benchmarks/pouw/hashed_gemm/hgemm.cu's WL, E2E's w_l3qm_s16 ported to egemm.cu's EL), a kernel
// for each window built: sw stages of BK = 128 (4 32-step segments) a window, so S = 4 sw segments
// (segment.PouwSegCheck_v2{S, duo}); thread ctid of tile t keys window w's chains with
// y_j = mix32(seed ^ mix32(((t 512 + ctid) nw + w) 8 + j)), z_j = mix32(y_j + 0x9E3779B9), absorbs T_(wS-1) first and each of
// the window's segments after, and stores R[(t nw + w) 512 + ctid] = xor_j (y_j ^ rotl(z_j, 16)). Two schedules of the same
// receipts: w_d (lag 3, the IMAD tie every fourth MMA, hgemm.cu's) and w_d4q (lag 4, the LOP3 tie every fourth: e_l4q's).
// A call is egemm.cu's qwe<0> (the dithered quantizer, X and P_A from bf16 x, and W' = W + U, P_B, under one key) and the
// GEMM (Y = X W^T exactly, the receipts keyed by the draw's seed). The receipts are committed as rows of 2 KiB, one
// window of one tile a row (hm96-sha512/row/v2, frame-v3-sha512), by commit_gpu's Committer in this translation unit, so
// its constants (one row size, one Ks) are this unit's and ovl.cu's X committers keep theirs.
#define main egemm_main
#include "../edited_gemm/egemm.cu"
#undef main
#define COMMIT_GPU_DOMS 4
#define smem cg_smem
namespace cgr {
#include "../commit_gpu/commit_gpu.cuh"
}
#undef smem
#include "eg.h"

constexpr uint32_t RROW = 2048;   // a receipt row: one window of one tile, 512 words

template <int LAG, int TIE> struct WL : EL<DUO, LAG, TIE> {
  using EB = EL<DUO, LAG, TIE>;
  using EB::EB;
  static constexpr int MI = EB::MI, NI = EB::NI, NCT = EB::NCT, U = MI * NI;
  static_assert(LAG >= 0 && LAG < U, "an absorb runs before its accumulator's next MMA");
  template <int J> DEV static void wab(St (&ch)[8], const int (&acc)[MI][NI][4]) {
    constexpr int mi = J / NI, ni = J % NI;
    EB::template absorb<DUO, mi, ni>(ch, acc[mi][ni]);
  }
  template <int TP> DEV static uint32_t wtz(const St (&ch)[8]) { return ch[4 * (TP % 2) + 3].z; }
  template <int J, int E> DEV static void ab_range(const int (&acc)[MI][NI][4], St (&ch)[8]) {
    if constexpr (J < E) {
      wab<J>(ch, acc);
      ab_range<J + 1, E>(acc, ch);
    }
  }
  // after MMA IDX, the absorb of unit IDX - LAG (the previous atom's when negative; none in a tile's first atom, nor in a
  // window's first, FRESH: the boundary absorbed it)
  template <bool FIRST, bool FRESH, int IDX = 0>
  DEV static void watom(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&fa)[MI][4], const uint32_t (&fb)[NI][2], uint32_t zero) {
    if constexpr (IDX < U) {
      constexpr int mi = IDX / NI, ni = IDX % NI, P = IDX - LAG;
      if constexpr (FIRST) {
        imma0(acc[mi][ni], fa[mi], fb[ni]);
      } else {
        if constexpr (TIE != 0 && IDX % (TIE > 0 ? TIE : -TIE) == 0)
          tie<(TIE < 0)>(acc[mi][ni][0], wtz<(IDX - 1 - LAG + 2 * U) % U>(ch), zero);
        imma(acc[mi][ni], fa[mi], fb[ni]);
      }
      if constexpr (P >= 0) wab<P>(ch, acc);
      else if constexpr (!FIRST && !FRESH) wab<P + U>(ch, acc);
      watom<FIRST, FRESH, IDX + 1>(acc, ch, fa, fb, zero);
    }
  }
  template <bool FIRST, bool FRESH, bool LAST>
  DEV void wstage(int (&acc)[MI][NI][4], St (&ch)[8], uint32_t (&fa)[2][MI][4], uint32_t (&fb)[2][NI][2], uint32_t zero) {
    const uint32_t sb = this->stage(this->s), done = this->empty(this->s);
    this->load(fa[1], fb[1], sb, 1);
    watom<FIRST, FRESH>(acc, ch, fa[0], fb[0], zero);
    this->load(fa[0], fb[0], sb, 2);
    watom<false, false>(acc, ch, fa[1], fb[1], zero);
    this->load(fa[1], fb[1], sb, 3);
    watom<false, false>(acc, ch, fa[0], fb[0], zero);
    this->advance();
    if (!LAST) {
      mbar_wait_warp(this->full(this->s), this->ph);
      this->load(fa[0], fb[0], this->stage(this->s), 0);
    }
    watom<false, false>(acc, ch, fa[1], fb[1], zero);
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
    ab_range<U - LAG, U>(acc, ch);
    rp[(size_t)w * NCT] = fold(ch);
    key(ch, seed, kb + 8u * (w + 1));
    ab_range<0, U>(acc, ch);
  }
  // K / BK stages, a window every SW (hgemm.cu's tile_w; SW a constant, as there: counted at run time it spills more)
  template <int SW>
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
    ab_range<U - LAG, U>(acc, ch);
    rp[(size_t)(KT / SW - 1) * NCT] = fold(ch);
  }
};

template <int LAG, int TIE, bool THREE, int SW> DEV void w_body(const Args& a) {
  using L = WL<LAG, TIE>;
  constexpr int MI = L::MI, NI = L::NI, U = L::U;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps, a.M, a.N, a.K, a.group, true, 1);
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int tiles_m = a.M / 128, tiles_n = a.N / 128, nw = a.K / (L::BK * SW);
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  const uint32_t zero = (uint32_t)a.KR;
  typename L::Place pl{0, 0};
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    ml.place_step(pl, t == (int)blockIdx.x ? t : (int)gridDim.x);
    L::place_tile(pl, tm, tn);
    const uint32_t tile_id = tm * tiles_n + tn, kb = (tile_id * L::NCT + ctid) * (uint32_t)nw * 8u;
    St ch[8];
    L::key(ch, a.seed, kb);
    {
      const int z0[MI][NI][4] = {};
      L::template ab_range<0, U>(z0, ch);
    }
    int acc[MI][NI][4];
    ml.template tile_w<SW>(a.K, acc, ch, zero, a.R + (size_t)tile_id * nw * L::NCT + ctid, a.seed, kb);
    ml.template peel<THREE>(acc);
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
typedef void (*WKern)(const Args);
template <int LAG, int TIE, bool THREE, int SW>
__global__ void __launch_bounds__(MLW::NT, 1) w_kern(const __grid_constant__ Args a) { w_body<LAG, TIE, THREE, SW>(a); }
// the windows built: S = 4 SW segments of 16, 32, 64, 128 and down's 112 (k = 14336: 448 segments, so 4 windows)
#define WSW(LAG, TIE, THREE) {w_kern<LAG, TIE, THREE, 4>, w_kern<LAG, TIE, THREE, 8>, w_kern<LAG, TIE, THREE, 16>, \
                              w_kern<LAG, TIE, THREE, 28>, w_kern<LAG, TIE, THREE, 32>}
static const int WSWS[5] = {4, 8, 16, 28, 32};
// w_d: lag 3, the IMAD tie every fourth MMA (hgemm.cu's); w_d4q: lag 4, the LOP3 tie every fourth (e_l4q's)
struct WK { const char* name; WKern k[5], k3[5]; };
static const WK WKS[2] = {{"w_d", WSW(3, -4, false), WSW(3, -4, true)}, {"w_d4q", WSW(4, 4, false), WSW(4, 4, true)}};
static int sw_slot(int sw) {
  for (int i = 0; i < 5; i++)
    if (WSWS[i] == sw) return i;
  return -1;
}

namespace eg {
struct Lin {
  int m, n, k, S, sw, nw, tiles, grid, kern = 0;
  int8_t* X;
  const uint16_t* x;
  int* Y;
  int8_t *Wp, *PA, *PB, *Xq;
  uint32_t* R;
  Maps* maps;
  cgr::Committer rc;
  std::string rparams;
  float inv_s = 1.f;
};
static bool g_init = false;
static cgr::Consts g_hc;
static bool g_hc_set = false;
static unsigned long long* g_bad;

void init() {
  if (g_init) return;
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  CK(cudaMalloc(&g_bad, 8));
  for (const WK& w : WKS)
    for (int i = 0; i < 5; i++)
      for (WKern f : {w.k[i], w.k3[i]}) CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  g_init = true;
}
void set_carve(int p) {
  const void* g[] = {(const void*)cgr::salt_small<true>, (const void*)cgr::leaf_small<cgr::RowBytes, 0>, (const void*)cgr::level_kernel,
                     (const void*)cgr::top_small};
  for (const void* f : g) CK(cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, p));
}
uint16_t* make_x(int rows, int k, uint32_t seed) {
  uint16_t* x;
  CK(cudaMalloc(&x, (size_t)rows * k * 2));
  xfill<<<4096, 256>>>(x, (size_t)rows * k, mix32(seed ^ 0x3333u), k, 0);
  CK(cudaDeviceSynchronize());
  return x;
}
void fill_w(int8_t* W, size_t count, uint32_t seed) {
  sfill<I8><<<4096, 256>>>(reinterpret_cast<uint8_t*>(W), count, mix32(seed ^ 0x4444u), -63, 127);
}
Lin* lin_new(int m, int n, int k, int S, int8_t* X, const uint16_t* x, int* Y, const std::string& rparams, int dom, int slots) {
  init();
  if (m % 1024 || n % 128 || k % 256 || k > 16128 || S % 4 || (k / 32) % S || sw_slot(S / 4) < 0) {
    fprintf(stderr, "eg: shape %d %d %d S %d: m %% 1024, n %% 128, k %% 256 == 0, k <= 16128, S | k / 32, S one of 16 32 64 112 128\n",
            m, n, k, S);
    exit(2);
  }
  Lin* L = new Lin;
  L->m = m; L->n = n; L->k = k; L->S = S; L->sw = S / 4; L->nw = k / (32 * S);
  L->tiles = (m / 128) * (n / 128);
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const char* gr = getenv("EG_GRID");
  L->grid = gr && !strcmp(gr, "sms") ? std::min(L->tiles, nsm) : L->tiles;   // default a CTA a tile
  L->X = X; L->x = x; L->Y = Y;
  CK(cudaMalloc(&L->Wp, (size_t)n * k));
  CK(cudaMalloc(&L->PA, (size_t)m * 128));
  CK(cudaMalloc(&L->PB, (size_t)n * 128));
  CK(cudaMalloc(&L->Xq, (size_t)m * k));
  CK(cudaMalloc(&L->R, lin_rbytes(L)));
  L->maps = upload_maps(Maps{tmap(X, k, m, 128), tmap(L->Wp, k, n, 128), tmap(L->PA, 128, m, 128), tmap(L->PB, 128, n, 128)});
  L->rparams = rparams;
  const cgr::Params p = cgr::read_params(rparams);
  L->rc.init(p, lin_leaves(L), RROW, false, false, 0, dom, true, slots);
  if (!g_hc_set) {
    g_hc = L->rc.hc;
    g_hc_set = true;
  } else if (memcmp(&g_hc, &L->rc.hc, sizeof g_hc)) {
    fprintf(stderr, "eg: %s's constants differ from the first receipts committer's (one Ks and row size)\n", rparams.c_str());
    exit(4);
  }
  CK(cudaDeviceSynchronize());
  return L;
}
int lin_nw(const Lin* L) { return L->nw; }
size_t lin_leaves(const Lin* L) { return (size_t)L->tiles * L->nw; }
size_t lin_rbytes(const Lin* L) { return lin_leaves(L) * RROW; }
void lin_kernel(Lin* L, int kern) { L->kern = kern; }
int lin_kernel_of(const Lin* L) { return L->kern; }
void lin_flags(Lin* L, bool fuse, bool pf) {
  L->rc.fuse_top = fuse;
  L->rc.prefetch = pf;
}
void lin_qwe(Lin* L, const int8_t* W, uint32_t key, cudaStream_t st) {
  const int qb = (L->m + QROWS - 1) / QROWS;
  qwe<0><<<qb + L->n / 8, 256, 0, st>>>(L->x, L->X, L->PA, L->m, L->k, L->inv_s, qb, reinterpret_cast<const uint8_t*>(W),
                                        reinterpret_cast<uint8_t*>(L->Wp), reinterpret_cast<uint8_t*>(L->PB), sigma(0x5EED0000u, key));
}
static void gemm_with(Lin* L, int kern, uint32_t seed, cudaStream_t st) {
  const Args a{L->maps, L->Y, L->R, L->m, L->n, L->k, 8, seed, 0, 1, 0, 0, {0, 0, 0, 0}, nullptr, 0};
  const WK& w = WKS[kern];
  const int i = sw_slot(L->sw);
  (L->k > PEEL2_MAXK ? w.k3[i] : w.k[i])<<<L->grid, MLW::NT, CSMEM, st>>>(a);
}
void lin_gemm(Lin* L, uint32_t seed, cudaStream_t st) { gemm_with(L, L->kern, seed, st); }
void lin_quant(Lin* L, cudaStream_t st) {
  quant<false><<<(L->m + QROWS - 1) / QROWS, 256, 0, st>>>(L->x, L->Xq, nullptr, L->m, L->k, L->inv_s, sigma(0x5EED0000u, 0));
}
void lin_rsalts(Lin* L, cudaStream_t st) { L->rc.salts<true>(nullptr, st); }
void lin_rcommit(Lin* L, int slot, cudaStream_t st) {
  L->rc.leaves(cgr::RowBytes{reinterpret_cast<const uint8_t*>(L->R), RROW, RROW}, st, slot);
  L->rc.nodes(st);
}
void lin_rpoison(Lin* L) {
  const cgr::Committer& c = L->rc;
  CK(cudaMemset(c.levels, 0xA5, 2 * c.width * 64));
  CK(cudaMemset(c.S, 0xA5, c.n * 128 * c.slots));
  CK(cudaMemset(c.F, 0xA5, c.n * 64));
  CK(cudaDeviceSynchronize());
}
void lin_root(const Lin* L, uint8_t* out64) {
  CK(cudaMemcpy(out64, L->rc.levels + 8 * (2 * L->rc.width - 2), 64, cudaMemcpyDeviceToHost));
}
double lin_time(Lin* L, int kern, int reps, cudaStream_t st) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  std::vector<double> t;
  for (int r = -2; r < reps; r++) {
    CK(cudaEventRecord(a, st));
    gemm_with(L, kern, 0x1234u + r, st);
    CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    if (r >= 0) t.push_back(ms);
  }
  CK(cudaGetLastError());
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}
static unsigned long long count_bad(const void* a, const void* b, size_t words, cudaStream_t st) {
  CK(cudaMemsetAsync(g_bad, 0, 8, st));
  cmp_words<<<1024, 256, 0, st>>>(reinterpret_cast<const uint32_t*>(a), reinterpret_cast<const uint32_t*>(b), words, g_bad);
  unsigned long long h;
  CK(cudaMemcpyAsync(&h, g_bad, 8, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  return h;
}
static void dump_file(const std::string& path, const void* dev, size_t bytes) {
  std::vector<uint8_t> h(bytes);
  CK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
  cgr::writef(path, h.data(), h.size());
}
std::string lin_gate(Lin* L, const int8_t* W, uint32_t key, uint32_t seed, int slot, const std::string& dir, bool& ok) {
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  const size_t ny = (size_t)L->m * L->n, rb = lin_rbytes(L);
  int* Yr;
  uint32_t* R0;
  CK(cudaMalloc(&Yr, ny * 4));
  CK(cudaMalloc(&R0, rb));
  CK(cudaMemsetAsync(Yr, 0xA5, ny * 4, st));
  lin_qwe(L, W, key, st);
  LtG pl = ltg(I8, L->m, L->n, L->k, L->k, L->k, L->n, L->X, W, Yr, 0, st);
  ltg_run(pl, st);
  unsigned long long yb[2], rpois[2], r01 = 0, ctrl = 0;
  for (int kern = 0; kern < 2; kern++) {
    CK(cudaMemsetAsync(L->Y, 0xA5, ny * 4, st));
    CK(cudaMemsetAsync(L->R, 0xA5, rb, st));
    gemm_with(L, kern, seed, st);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    yb[kern] = count_bad(L->Y, Yr, ny, st);
    // the receipts left at the poison word (a word can be 0xA5A5A5A5 by chance: a few at most)
    {
      uint32_t* pw;
      CK(cudaMalloc(&pw, rb));
      CK(cudaMemsetAsync(pw, 0xA5, rb, st));
      rpois[kern] = rb / 4 - count_bad(L->R, pw, rb / 4, st);
      CK(cudaFree(pw));
    }
    if (kern == 0) CK(cudaMemcpyAsync(R0, L->R, rb, cudaMemcpyDeviceToDevice, st));
    else r01 = count_bad(L->R, R0, rb / 4, st);
  }
  // the control: another seed's receipts must differ almost everywhere
  gemm_with(L, L->kern, seed ^ 0x80000000u, st);
  CK(cudaStreamSynchronize(st));
  ctrl = count_bad(L->R, R0, rb / 4, st);
  gemm_with(L, L->kern, seed, st);
  CK(cudaStreamSynchronize(st));
  ok = yb[0] == 0 && yb[1] == 0 && r01 == 0 && rpois[0] < 16 && rpois[1] < 16 && ctrl > rb / 4 / 2;
  char buf[512];
  snprintf(buf, sizeof buf,
           "{\"m\": %d, \"n\": %d, \"k\": %d, \"S\": %d, \"nw\": %d, \"leaves\": %zu, \"grid\": %d, \"y_bad\": [%llu, %llu], "
           "\"r_bad_w_d_vs_w_d4q\": %llu, \"r_poison_left\": [%llu, %llu], \"r_other_seed_differ\": %llu, \"ok\": %s}",
           L->m, L->n, L->k, L->S, L->nw, lin_leaves(L), L->grid, yb[0], yb[1], r01, rpois[0], rpois[1], ctrl, ok ? "true" : "false");
  if (!dir.empty()) {
    if (system(("mkdir -p " + dir).c_str())) exit(4);
    lin_rcommit(L, slot, st);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    dump_file(dir + "/X.bin", L->X, (size_t)L->m * L->k);
    dump_file(dir + "/Wp.bin", L->Wp, (size_t)L->n * L->k);
    dump_file(dir + "/operand.bin", L->R, rb);
    const cgr::Committer& c = L->rc;
    cgr::u64 used = 0;
    for (int l = 0; l <= c.depth; l++) used += c.width >> l;
    std::vector<cgr::u64> lv(8 * used);
    CK(cudaMemcpy(lv.data(), c.levels, lv.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> lb(lv.size() * 8);
    for (size_t i = 0; i < lv.size(); i++) cgr::sha::put_be64(lb.data() + 8 * i, lv[i]);
    cgr::writef(dir + "/levels.bin", lb.data(), lb.size());
    const std::vector<uint8_t> pf = cgr::readf(L->rparams);
    cgr::writef(dir + "/params.bin", pf.data(), pf.size());
  }
  CK(cudaFree(Yr));
  CK(cudaFree(R0));
  CK(cudaStreamDestroy(st));
  return buf;
}
}  // namespace eg
