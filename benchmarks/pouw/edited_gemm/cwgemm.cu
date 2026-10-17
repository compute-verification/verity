// co32c's linear (cgemm.cu's c_l7qs) with the weight edit inside the GEMM: W' = W + c + U is formed in shared memory
// from the cached weight and the per-call edit, not by cwedit's pass writing W' (n k bytes read, n k written) to memory.
//
// The cached weight is kept biased, Wb = W + 64 in [0, 126] (a one-time encoding of the weight cache, as W itself is), so
// with Ub = U + 64 in [0, 128] the bytewise sum Wb + Ub + c is at most 255 and one IADD3 makes four bytes with no carry
// between them; W' = (Wb + Ub + c) ^ 0x80 bytewise. Hashing c inside the GEMM would redo a tile's whole k column of c
// (k / 4 SHA-512 blocks) in every tile row, so a pass makes the per-call edit from the key first:
//   cplane   c (tag 8, cgemm.cu's patch layout) as a 1-bit plane, n k / 8 bytes: word (tn, kt, row, j) at
//            ((tn KT + kt) 128 + row) 4 + j holds row tn 128 + row's columns kt 128 + 32 j .. + 31, entry 4 w + b (byte b
//            of word w) at bit 8 b + w, so word w's four c bits are (P >> w) & 0x01010101; and from beta (tag 10) P_B =
//            [-U | U | U | 0] as cwedit's and UB = U + 64 (n x 32 bytes)
//   cw_*     c_*'s GEMM (e_l7qs's mainloop, co32c's tail and receipts) on X and Wb: each consumer thread edits 32 bytes of
//            every stage's B tile in place (row 32 wn + 8 wm + lane / 4, logical chunks 2 j, 2 j + 1 with j = lane % 4, so
//            the 4 warps of column wn edit the 32 rows they read), the next stage's edit between this stage's atoms 1
//            and 2 and an arrival on the slot's edited mbarrier for column wn, whose 4 warps (all on one SMSP) wait on it
//            before the first ldmatrix of that stage, after atom 2 (a named barrier there runs them in lockstep: 19%
//            slower than c_l7qs at 8192^3, against 17% with the mbarrier); a tile's first stage is edited before its
//            first atom. The release's fence.proxy.async orders the edit's stores before the slot's TMA refill
// Modes:
//   cwgemm gate DIR M N K SEED    cgemm.cu's gate into DIR (c_l7qs and its peers, the dumps cgemm_ref.py reads), then
//                                 under the same key: cplane's P_B against cwedit's, cw's O against cuBLASLt's X W^T and
//                                 its receipts against c_l7qs's, a no-write control; three tiles' O, receipts, plane and UB
//                                 to DIR (cw_meta.json) for cwgemm_ref.py
//   cwgemm bench REPS M N K       hot items interleaved across CW_ARMS (cgemm.cu's and egemm.cu's arms too), one JSON line
//   cwgemm one ARM M N K          three warm-up calls, then one between cudaProfilerStart and Stop (for ncu)
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a cwgemm.cu -lcuda -lcublasLt -lnvidia-ml
#define CGEMM_LIB
#include "cgemm.cu"

// c's 32 bits of a row's 32-column segment (bit 4 w + b, byte b of word w) to the plane's order (bit 8 b + w)
DEV uint32_t ctrans(uint32_t d) {
  uint32_t p = 0;
#pragma unroll
  for (int b = 0; b < 4; b++) {
    uint32_t x = (d >> b) & 0x11111111u;
    x = (x | (x >> 3)) & 0x03030303u;
    x = (x | (x >> 6)) & 0x000F000Fu;
    x = (x | (x >> 12)) & 0xFFu;
    p |= x << (8 * b);
  }
  return p;
}
// grid (n / 512, KT + 1), 128 threads, thread g of a column the row group 4 g .. 4 g + 3: at y < KT its patch (g, y) of c
// as 16 plane words; at y = KT its rows' U (beta blocks 2 g, 2 g + 1) as P_B and UB
__global__ void __launch_bounds__(128) cplane(uint32_t* __restrict__ plane, uint8_t* __restrict__ PB, uint32_t* __restrict__ UB,
                                              int n, int KT, const __grid_constant__ Sig sg) {
  const int g = blockIdx.x * 128 + threadIdx.x, y = blockIdx.y;
  if (g * 4 >= n) return;
  uint64_t st[8];
  if (y < KT) {
    sha_block(sg.w, (uint64_t)C_TAG << 56 | ((uint64_t)g * KT + y), st);
#pragma unroll
    for (int rr = 0; rr < 4; rr++) {
      // digest bytes 16 rr + 4 j .. + 3, little-endian: row 4 g + rr's columns 32 j .. 32 j + 31 of the block
      const uint32_t hi = (uint32_t)(st[2 * rr] >> 32), lo = (uint32_t)st[2 * rr], hi2 = (uint32_t)(st[2 * rr + 1] >> 32),
                     lo2 = (uint32_t)st[2 * rr + 1];
      const int row = 4 * g + rr;
      uint4* dst = reinterpret_cast<uint4*>(plane + (((size_t)(row >> 7) * KT + y) * 128 + (row & 127)) * 4);
      *dst = make_uint4(ctrans(__byte_perm(hi, 0, 0x0123)), ctrans(__byte_perm(lo, 0, 0x0123)), ctrans(__byte_perm(hi2, 0, 0x0123)),
                        ctrans(__byte_perm(lo2, 0, 0x0123)));
    }
    return;
  }
#pragma unroll 1
  for (int h = 0; h < 2; h++) {
    sha_block(sg.w, (uint64_t)BETA_TAG << 56 | (uint64_t)(2 * g + h), st);
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
      const size_t row = 4 * (size_t)g + 2 * h + rr;
      uint32_t u[8];
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
          const int e = 32 * rr + 4 * q + b;
          v |= ((uint32_t)u_beta((uint32_t)(st[e / 8] >> (56 - 8 * (e % 8))) & 0xFFu) & 0xFFu) << (8 * b);
        }
        u[q] = v;
      }
      uint4* pb = reinterpret_cast<uint4*>(PB + row * 128);
      uint4* ub = reinterpret_cast<uint4*>(UB + row * 8);
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const uint32_t a[4] = {u[4 * i], u[4 * i + 1], u[4 * i + 2], u[4 * i + 3]};
        uint32_t nu[4], b64[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          nu[e] = add_noise<false>(~a[e], 0x01010101u);
          b64[e] = ((a[e] & 0x7F7F7F7Fu) + 0x40404040u) ^ (a[e] & 0x80808080u);
        }
        pb[i] = make_uint4(nu[0], nu[1], nu[2], nu[3]);
        pb[2 + i] = make_uint4(a[0], a[1], a[2], a[3]);
        pb[4 + i] = make_uint4(a[0], a[1], a[2], a[3]);
        pb[6 + i] = make_uint4(0, 0, 0, 0);
        ub[i] = make_uint4(b64[0], b64[1], b64[2], b64[3]);
      }
    }
  }
}
// the cached weight's encoding, Wb = W + 64
__global__ void wbias(const int8_t* __restrict__ W, uint8_t* __restrict__ Wb, size_t count) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
    Wb[i] = (uint8_t)(W[i] + 64);
}

struct WArgs {
  Args a;
  const uint32_t* plane;
  const uint4* ub;
};

// e_l7qs's mainloop (EL) with each stage's B tile edited in place before its first ldmatrix
template <int LAG, int TIE, bool SLEEP> struct WL : EL<DUO, LAG, TIE, SLEEP> {
  using B = EL<DUO, LAG, TIE, SLEEP>;
  using B::B;
  static constexpr int MI = B::MI, NI = B::NI, NST = B::NST;
  int erow = 0, ej = 0, KTn = 0;
  // ebar: this column's edited barriers (slot i's at ebar + 8 i, 128 arrivals: the column's 4 warps), eph: their phases
  // (bit i), toggled at each wait, since the tail's stage takes a slot without an edit
  uint32_t eoff = 0, ebar = 0, eph = 0, pw = 0;
  const uint32_t* cpl = nullptr;   // this tile's plane words for this thread: stage q's at cpl[512 q]
  const uint4* cub = nullptr;      // this thread's row of UB
  DEV static uint32_t edited0(uint32_t bars) { return bars + 16 * NST; }
  DEV void edit_init(int warp, int lane) {
    const int wm = warp / B::WARPS_N, wn = warp % B::WARPS_N;
    erow = 32 * wn + 8 * wm + (lane >> 2);
    ej = lane & 3;
    eoff = B::A_BYTES + erow * B::BK + (((2 * ej) ^ (erow & 7)) << 4);
    ebar = edited0(this->bars) + 8 * NST * wn;
  }
  // stage q's edit in slot s (W' = (Wb + Ub + c) ^ 0x80 on logical chunks 2 j, 2 j + 1) and the arrival on its edited
  // barrier, then stage q + 1's plane word (the plane has a stage of padding, so the last stage's prefetch stays in it)
  DEV void edit(int q) {
    const uint32_t a0 = this->stage(this->s) + eoff, a1 = a0 ^ 16u;
    uint32_t v[8];
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(a0) : "memory");
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]) : "r"(a1) : "memory");
    const uint4 u0 = __ldg(cub), u1 = __ldg(cub + 1);
    const uint32_t u[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    const uint32_t p = pw;
#pragma unroll
    for (int w = 0; w < 8; w++) v[w] = (v[w] + u[w] + ((p >> w) & 0x01010101u)) ^ 0x80808080u;
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(a0), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(a1), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]) : "memory");
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(ebar + 8 * this->s) : "memory");
    pw = __ldg(cpl + 512 * (q + 1));
  }
  // slot s's edit complete in the column (the mbarrier's release and acquire order the edit's stores before the ldmatrix)
  DEV void meet() {
    const uint32_t par = (eph >> this->s) & 1u;
    eph ^= 1u << this->s;
    if constexpr (SLEEP) mbar_sleep_warp(ebar + 8 * this->s, par);
    else mbar_wait_warp(ebar + 8 * this->s, par);
  }
  template <bool FIRST, bool LAST>
  DEV void stage_w(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[1], uint32_t (&fa)[2][MI][4], uint32_t (&fb)[2][NI][2],
                   uint32_t zero, int kt) {
    const uint32_t sb = this->stage(this->s), done = this->empty(this->s);
    this->load(fa[1], fb[1], sb, 1);
    B::template atom<FIRST, 0>(acc, ch, kr, fa[0], fb[0], zero);
    this->load(fa[0], fb[0], sb, 2);
    B::template atom<false, 1>(acc, ch, kr, fa[1], fb[1], zero);
    this->advance();
    if (!LAST) {
      this->wait_full();
      edit(kt + 1);
    }
    this->load(fa[1], fb[1], sb, 3);
    B::template atom<false, 2>(acc, ch, kr, fa[0], fb[0], zero);
    if (!LAST) {
      meet();
      this->load(fa[0], fb[0], this->stage(this->s), 0);
    }
    B::template atom<false, 3>(acc, ch, kr, fa[1], fb[1], zero);
    this->release(done);
  }
  DEV void tile_w(int K, int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[1], uint32_t zero) {
    KTn = K / B::BK;
    uint32_t fa[2][MI][4], fb[2][NI][2];
    pw = __ldg(cpl);
    this->wait_full();
    edit(0);
    meet();
    this->load(fa[0], fb[0], this->stage(this->s), 0);
    if (KTn == 1) {
      stage_w<true, true>(acc, ch, kr, fa, fb, zero, 0);
    } else {
      stage_w<true, false>(acc, ch, kr, fa, fb, zero, 0);
#pragma unroll 1
      for (int kt = 1; kt < KTn - 1; kt++) stage_w<false, false>(acc, ch, kr, fa, fb, zero, kt);
      stage_w<false, true>(acc, ch, kr, fa, fb, zero, KTn - 1);
    }
    B::flush(ch, kr, acc);
  }
};

// c_body (SERVE 1: O; 0: O') on WL
template <int LAG, int TIE, bool SLEEP, bool THREE, int SERVE> DEV void cw_body(const WArgs& w) {
  const Args& a = w.a;
  using L = WL<LAG, TIE, SLEEP>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps, a.M, a.N, a.K, a.group, true, 1);
  if (threadIdx.x == 0)
    for (int i = 0; i < L::WARPS_N * L::NST; i++) mbar_init(L::edited0(ml.bars) + 8 * i, 128);
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  ml.edit_init(warp, lane);
  const int tiles_m = a.M / 128, tiles_n = a.N / 128, KT = a.K / 128;
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  const uint32_t zero = (uint32_t)a.KR;
  uint32_t kr[1];
  kr[0] = 0;
  typename L::Place pl{0, 0};
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    ml.place_step(pl, t == (int)blockIdx.x ? t : (int)gridDim.x);
    L::place_tile(pl, tm, tn);
    const uint32_t tile_id = tm * tiles_n + tn;
    St ch[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      ch[e].y = mix32(a.seed ^ mix32((tile_id * L::NCT + ctid) * 8 + e));
      ch[e].z = mix32(ch[e].y + 0x9E3779B9u);
    }
    ml.cpl = w.plane + ((size_t)tn * KT * 128 + ml.erow) * 4 + ml.ej;
    ml.cub = w.ub + (size_t)(tn * 128 + ml.erow) * 2;
    int acc[MI][NI][4];
    ml.tile_w(a.K, acc, ch, kr, zero);
    ctail<L, THREE, SERVE != 0>(ml, acc, ch);
    const int row0 = tm * 128 + wm * 32 + g, col0 = tn * 128 + wn * 32 + 2 * q;
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int* p = a.U + (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni;
        *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
        *reinterpret_cast<int2*>(p + (size_t)8 * a.N) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
      }
    uint4* r = reinterpret_cast<uint4*>(a.R) + ((size_t)tile_id * L::NCT + ctid) * 2;
    r[0] = make_uint4(finish<DUO>(ch, 0), finish<DUO>(ch, 1), finish<DUO>(ch, 2), finish<DUO>(ch, 3));
    r[1] = make_uint4(finish<DUO>(ch, 4), finish<DUO>(ch, 5), finish<DUO>(ch, 6), finish<DUO>(ch, 7));
  }
}
#define CWGEMM(NAME, LAG, TIE, SLEEP)                                                                                                 \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ WArgs w) { cw_body<LAG, TIE, SLEEP, false, 1>(w); }     \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME##_k3(const __grid_constant__ WArgs w) { cw_body<LAG, TIE, SLEEP, true, 1>(w); }
CWGEMM(cw_l7qs, 7, 4, true)

// ---------------------------------------------------------------- host
// CSMEM is ML's stages, its 6 stage barriers and 1,024 bytes of alignment slack; the 12 edited barriers follow the 6
constexpr int WSMEM = CSMEM + 128;
static_assert(WSMEM <= 101376, "sm_120's 99 KB a block");
typedef void (*WKern)(const WArgs);
struct WKn { const char* name; WKern k, k3; };
static const WKn WKERN[] = {{"cw_l7qs", cw_l7qs, cw_l7qs_k3}};
static const WKn* wkern(const std::string& n) {
  for (auto& e : WKERN) if (n == e.name) return &e;
  return nullptr;
}
// cgemm.cu's CP and the in-GEMM edit's own buffers: Wb (the cached weight's encoding), the plane, UB and cplane's P_B
struct WP {
  CP C;
  uint8_t* Wb;
  uint32_t *plane, *UB;
  uint8_t* PB;
  Maps* mW;
  cudaStream_t s2;
  cudaEvent_t fork, join;
};
static void cpl(WP& W, const Sig& s) {
  const EP& P = W.C.E;
  cplane<<<dim3((P.n / 4 + 127) / 128, P.k / 128 + 1), 128, 0, P.st>>>(W.plane, W.PB, W.UB, P.n, P.k / 128, s);
}
static void wrun(WP& W, const WKn* e, int* Y, uint32_t* R) {
  EP& P = W.C.E;
  WArgs w{Args{W.mW, Y, R, P.m, P.n, P.k, 8, P.seed, 0, 1, 0, 0, {0, 0, 0, 0}, nullptr, 0}, W.plane,
          reinterpret_cast<const uint4*>(W.UB)};
  (n_dig(P.k) == 3 ? e->k3 : e->k)<<<P.grid, MLW::NT, WSMEM, P.st>>>(w);
}
// f's launches on the second stream beside g's on the main one; what the main stream launches next waits for both
static void beside(WP& W, const std::function<void()>& f, const std::function<void()>& g) {
  EP& P = W.C.E;
  const cudaStream_t st = P.st;
  CK(cudaEventRecord(W.fork, st));
  CK(cudaStreamWaitEvent(W.s2, W.fork, 0));
  P.st = W.s2;
  f();
  P.st = st;
  g();
  CK(cudaEventRecord(W.join, W.s2));
  CK(cudaStreamWaitEvent(st, W.join, 0));
}
static void wsetup(WP& W, int m, int n, int k, uint32_t seed, bool gate) {
  csetup(W.C, m, n, k, seed, gate);
  if (W.C.half) { fprintf(stderr, "cwgemm: co32c only (k = %d)\n", k); exit(2); }
  EP& P = W.C.E;
  CK(cudaMalloc(&W.Wb, (size_t)n * k)); CK(cudaMalloc(&W.plane, (size_t)n * k / 8 + 2048));
  CK(cudaMalloc(&W.UB, (size_t)n * 32)); CK(cudaMalloc(&W.PB, (size_t)n * 128));
  CK(cudaStreamCreateWithFlags(&W.s2, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&W.fork, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&W.join, cudaEventDisableTiming));
  wbias<<<4096, 256, 0, P.st>>>(W.C.W7, W.Wb, (size_t)n * k);
  cpl(W, sigma(seed, 0xFFFFFFFFu));
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  W.mW = upload_maps(Maps{tmap(W.C.X, k, m, 128), tmap(W.Wb, k, n, 128), tmap(W.C.PA, 128, m, 128), tmap(W.PB, 128, n, 128)});
  for (auto& e : WKERN)
    for (WKern kern : {e.k, e.k3}) CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, WSMEM));
}
static void wfree(WP& W) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)W.Wb, (void*)W.plane, (void*)W.UB, (void*)W.PB, (void*)W.mW}) CK(cudaFree(p));
  CK(cudaEventDestroy(W.fork)); CK(cudaEventDestroy(W.join)); CK(cudaStreamDestroy(W.s2));
  cfree(W.C);
}
// an arm: cplane, a cw_* kernel (on cplane's plane and cquant's X), cwedited:KERN (cquant, cplane, KERN under one key: the
// served linear), cpar:KERN and cwpar:KERN (cedited's and cwedited's with the weight pass on a second stream beside
// cquant), or any of cgemm.cu's arms (cedited:c_l7qs, cwedit, plain, plainq, ...)
static std::function<void(WP&)> warm(const std::string& a) {
  if (a == "cplane") return [](WP& W) { cpl(W, sigma(W.C.E.seed, W.C.inst++)); };
  if (a.rfind("cpar:", 0) == 0) {
    const CKn* k = ckern(a.substr(5));
    if (!k) { fprintf(stderr, "no kernel %s\n", a.c_str()); exit(2); }
    return [k](WP& W) {
      const Sig s = sigma(W.C.E.seed, W.C.inst++);
      beside(W, [&] { cw(W.C, s); }, [&] { cq(W.C, s); });
      cgemm_run(W.C, cpick(W.C, k), W.C.Y, W.C.R);
    };
  }
  if (a.rfind("cwpar:", 0) == 0) {
    const WKn* k = wkern(a.substr(6));
    if (!k) { fprintf(stderr, "no kernel %s\n", a.c_str()); exit(2); }
    return [k](WP& W) {
      const Sig s = sigma(W.C.E.seed, W.C.inst++);
      beside(W, [&] { cpl(W, s); }, [&] { cq(W.C, s); });
      wrun(W, k, W.C.Y, W.C.R);
    };
  }
  if (a.rfind("cwedited:", 0) == 0) {
    const WKn* k = wkern(a.substr(9));
    if (!k) { fprintf(stderr, "no kernel %s\n", a.c_str()); exit(2); }
    return [k](WP& W) {
      const Sig s = sigma(W.C.E.seed, W.C.inst++);
      cq(W.C, s); cpl(W, s); wrun(W, k, W.C.Y, W.C.R);
    };
  }
  if (const WKn* k = wkern(a)) return [k](WP& W) { wrun(W, k, W.C.Y, W.C.R); };
  auto f = carm(a);
  return [f](WP& W) { f(W.C); };
}

static int wgate(const std::string& dir, int m, int n, int k, uint32_t seed) {
  const int rc = cgate(dir, m, n, k, seed);
  WP W;
  wsetup(W, m, n, k, seed, true);
  CP& C = W.C;
  EP& P = C.E;
  const size_t wb = (size_t)n * k;
  for (auto [p, b] : {std::pair<void*, size_t>{C.X, (size_t)m * k}, {C.PA, (size_t)m * 128}, {C.Wp, wb}, {C.PB, (size_t)n * 128},
                      {C.Wh, wb}, {C.Y, P.ny * 4}, {C.Yr, P.ny * 4}, {C.R, P.nr * 4}, {C.R2, P.nr * 4}, {W.plane, wb / 8},
                      {W.UB, (size_t)n * 32}, {W.PB, (size_t)n * 128}})
    CK(cudaMemsetAsync(p, 0xA5, b, P.st));
  const Sig s = sigma(seed, 0);
  cq(C, s);
  cw(C, s);
  phat<false><<<4096, 256, 0, P.st>>>(C.Wp, C.PB, C.Wh, (size_t)n, k);
  ltg_run(C.ref, P.st);
  cgemm_run(C, cpick(C, ckern("c_l7qs")), C.Y, C.R2);
  cpl(W, s);
  CK(cudaMemsetAsync(C.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(C.R, 0xA5, P.nr * 4, P.st));
  wrun(W, wkern("cw_l7qs"), C.Y, C.R);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  const unsigned long long y = ecmp(P, C.Y, C.Yr, P.ny), r = ecmp(P, C.R, C.R2, P.nr), pb = ecmp(P, W.PB, C.PB, (size_t)n * 32);
  // the overlap arms' order under the gate's key, from poisoned operands: cwedit (cplane) beside cquant, then c_l7qs (cw_l7qs)
  unsigned long long par[2][2];
  for (int i = 0; i < 2; i++) {
    for (auto [p, b] : {std::pair<void*, size_t>{C.X, (size_t)m * k}, {C.PA, (size_t)m * 128}, {C.Wp, wb}, {C.PB, (size_t)n * 128},
                        {W.plane, wb / 8}, {W.UB, (size_t)n * 32}, {W.PB, (size_t)n * 128}, {C.Y, P.ny * 4}, {C.R, P.nr * 4}})
      CK(cudaMemsetAsync(p, 0xA5, b, P.st));
    if (i == 0) {
      beside(W, [&] { cw(C, s); }, [&] { cq(C, s); });
      cgemm_run(C, cpick(C, ckern("c_l7qs")), C.Y, C.R);
    } else {
      beside(W, [&] { cpl(W, s); }, [&] { cq(C, s); });
      wrun(W, wkern("cw_l7qs"), C.Y, C.R);
    }
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    par[i][0] = ecmp(P, C.Y, C.Yr, P.ny);
    par[i][1] = ecmp(P, C.R, C.R2, P.nr);
  }
  const bool ok = y == 0 && r == 0 && pb == 0 && par[0][0] == 0 && par[0][1] == 0 && par[1][0] == 0 && par[1][1] == 0;
  const int tiles_m = m / 128, tiles_n = n / 128, KT = k / 128;
  const int tl[3][2] = {{0, 0}, {tiles_m / 2, tiles_n / 2}, {tiles_m - 1, tiles_n - 1}};
  std::string tj;
  for (int i = 0; i < 3; i++) {
    const int tm = tl[i][0], tn = tl[i][1];
    const std::string sfx = "_" + std::to_string(tm) + "_" + std::to_string(tn) + ".bin";
    dump(dir, ("R_cw_l7qs" + sfx).c_str(), C.R + (size_t)(tm * tiles_n + tn) * 512 * 8, 512 * 8 * 4, true);
    std::vector<int> yt(128 * 128);
    for (int rr = 0; rr < 128; rr++)
      CK(cudaMemcpy(yt.data() + rr * 128, C.Y + (size_t)(tm * 128 + rr) * n + tn * 128, 128 * 4, cudaMemcpyDeviceToHost));
    dump(dir, ("Y_cw_l7qs" + sfx).c_str(), yt.data(), yt.size() * 4, false);
    dump(dir, ("plane" + sfx).c_str(), W.plane + (size_t)tn * KT * 512, (size_t)KT * 2048, true);
    dump(dir, ("UB" + sfx).c_str(), W.UB + (size_t)tn * 128 * 8, 128 * 32, true);
    tj += std::string(i ? ", " : "") + "[" + std::to_string(tm) + ", " + std::to_string(tn) + "]";
  }
  CK(cudaMemsetAsync(C.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(C.R, 0xA5, P.nr * 4, P.st));
  const unsigned long long cy = ecmp(P, C.Y, C.Yr, P.ny), cr = ecmp(P, C.R, C.R2, P.nr);
  const bool control_rejected = cy > 0 && cr > 0;
  char res[768];
  snprintf(res, sizeof res,
           "\"cw_l7qs\": {\"y_bad\": %llu, \"r_bad\": %llu}, \"pb_bad\": %llu, \"cpar:c_l7qs\": {\"y_bad\": %llu, \"r_bad\": %llu}, "
           "\"cwpar:cw_l7qs\": {\"y_bad\": %llu, \"r_bad\": %llu}, \"control\": {\"y_bad\": %llu, \"r_bad\": %llu}",
           y, r, pb, par[0][0], par[0][1], par[1][0], par[1][1], cy, cr);
  FILE* fp = fopen((dir + "/cw_meta.json").c_str(), "w");
  fprintf(fp, "{\"kernel\": \"cw_l7qs\", \"reference\": \"c_l7qs\", \"tiles\": [%s], \"device\": {%s}, \"device_ok\": %s, "
              "\"control_rejected\": %s}\n",
          tj.c_str(), res, ok ? "true" : "false", control_rejected ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"kernel\": \"cw_l7qs\", \"m\": %d, \"n\": %d, \"k\": %d, \"device_ok\": %s, \"control_rejected\": %s, "
         "\"device\": {%s}}\n",
         dir.c_str(), m, n, k, ok ? "true" : "false", control_rejected ? "true" : "false", res);
  wfree(W);
  return rc == 0 && ok && control_rejected ? 0 : 1;
}

static int wbench(int reps, int m, int n, int k) {
  Nv nv;
  const std::vector<std::string> names =
      split(getenv("CW_ARMS"), "plain,plainq,c_l7qs,cw_l7qs,cquant,cwedit,cplane,cedited:c_l7qs,cwedited:cw_l7qs,cpar:c_l7qs,cwpar:cw_l7qs");
  WP W;
  wsetup(W, m, n, k, 0x5EED0000u + m + n + k, false);
  EP& P = W.C.E;
  std::vector<std::function<void(WP&)>> arms;
  for (auto& a : names) arms.push_back(warm(a));
  const int NA = (int)arms.size();
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_arm = [&](int j, int calls) {
    CK(cudaEventRecord(e0, P.st));
    for (int c = 0; c < calls; c++) arms[j](W);
    CK(cudaEventRecord(e1, P.st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return (double)ms / calls;
  };
  std::vector<int> calls(NA);
  for (int j = 0; j < NA; j++) {
    time_arm(j, 1);
    const double t = time_arm(j, 2);
    calls[j] = std::max(2, std::min(400, (int)(20.0 / std::max(t, 1e-4))));
  }
  CK(cudaGetLastError());
  std::vector<std::vector<double>> ms(NA), mhz(NA);
  for (int r = 0; r < reps; r++)
    for (int j = 0; j < NA; j++) {
      const int a = (j + r) % NA;
      ms[a].push_back(time_arm(a, calls[a]));
      mhz[a].push_back(nv.sm());
    }
  CK(cudaGetLastError());
  printf("{\"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d, \"lt_algo\": %d, \"arms\": {", m, n, k, P.grid, P.plain.algo_id);
  for (int j = 0; j < NA; j++) {
    printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", names[j].c_str(), median(ms[j]),
           median(mhz[j]), calls[j]);
    for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
    printf("], \"mhz_all\": [");
    for (size_t i = 0; i < mhz[j].size(); i++) printf("%s%.0f", i ? ", " : "", mhz[j][i]);
    printf("]}");
  }
  printf("}}\n");
  fflush(stdout);
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  wfree(W);
  return 0;
}

static int wone(const std::string& arm, int m, int n, int k) {
  WP W;
  wsetup(W, m, n, k, 0x5EED0000u + m + n + k, false);
  auto f = warm(arm);
  for (int i = 0; i < 3; i++) f(W);
  CK(cudaStreamSynchronize(W.C.E.st));
  CK(cudaProfilerStart());
  f(W);
  CK(cudaStreamSynchronize(W.C.E.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"one\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d}\n", arm.c_str(), m, n, k, W.C.E.grid);
  wfree(W);
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return wgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 6 && !strcmp(argv[1], "bench")) return wbench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 6 && !strcmp(argv[1], "one")) return wone(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: cwgemm gate DIR M N K SEED | bench REPS M N K | one ARM M N K\n");
  return 2;
}
