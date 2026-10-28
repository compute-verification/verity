// The adopted P.V (P's w64 peel, V dithered: ppl's `adopted-attention`, `checked_strip.ts_pv_adopted`) on sm_120a, a
// (head, 128-query tile) at a time, on cgemm.cu's GEMM (e_l7qs's mainloop and duo receipts).
//
// Attention's P.V for H query heads sharing kvh = H / rep KV heads, s tokens (s <= 32768), head dimension 128. P_prob (H s
// x s, row h s + q, in [0, 127]: softmax's quantized P, one scale a query row) is the peeled side; key 0 is its own P block
// (p_sink), so the unit's block is keys 1 .. q and M = [1 <= key <= q]. P~ = M (P_prob - 64 + U(q, key mod 32)), U on
// +-1 .. +-64 sampled once a call (s x 32) and shared by the heads. V (kvh 128 x s, channel-major, floor-quantized) is the
// served side: X = V + b (b a fresh bit an entry). A 32-query block j (warp row w of 128-query tile qt, j = 4 qt + w) has
// b_l = j: its keys are 0 .. 32 (j + 1) - 1, Ycum(rho, ch) = sum of X(key, ch) over 1 <= key < 32 j with key = rho mod 32,
// split as Ycum = Y0 + 256 Y1 + 65536 Y2 (Y0, Y1 its low bytes as s8; |Ycum| <= 4 s, so Y2 = 0 at s <= 4096 and |Y2| <= 2
// at s <= 32768), and its diagonal block's noise is W = (U M_d) X_diag. Every cell commits, mod 2^32, in this order:
//   T_t = sum_{keys < 32 (t + 1)} P~ X (t <= b_l), O = T_last - U Y0 - W - 256 U Y1 - 65536 U Y2 = sum M (P_prob - 64) X,
// i.e. checked_strip.pv_words with c = 0, m = 1 (O' = O, no digits) and key 0's row of X zeroed. With S = sum M X the served
// value is sum M P_prob X = O + 64 S, and key 0's block adds z_q 127 X(0) (z_q its scale): f = (float(O + 64 S) s_q +
// z_q float(X(0))) s_ch, each operation rounded on its own.
// A CTA tile is 128 queries (four query blocks, one a warp row) x the 128 channels over keys 0 .. 128 (qt + 1) - 1: qt
// full stages on e_l7qs's interleaved absorbs, then the diagonal stage, warp row w absorbing its atoms t <= 4 qt + w right
// after each. The tail stage's rows P_A = [-U | -U | -U M_d | 64 M_d] (shared by the heads, TMA) meet each warp's own DG
// rows [Y0 | Y1 | X_diag | Y2] for its b_l (global memory): atoms 0 and 2 onto T_last, atom 1 into 256 U Y1 (and, at
// s > 4096, P_A's atom 1 against Y2 into 65536 U Y2), O absorbed;
// then atom 3 against X_diag adds 64 S_diag for f, and the epilogue adds 64 S over the blocks before b_l (Sq).
// Noise from the per-call key (sha_pass.cuh's Sig; SHA-512(key || BE64(tag << 56 | j))): b under tag 7 in NS 1's patch
// layout over V^T (as cpv.cu's vedit); U under tag 14 (row q: block q / 2, digest byte (q % 2) 32 + rho, U = (byte & 63) +
// 1, negated where bit 6 is set: egemm_ref.u_rows' map, LP_UW 64).
// Kernels:
//   uedit            U - 64 (s x 32, for pedit), P_A's rows (s x 128) and apv_fu's V_u = U + 64 in fragment order
//   pedit            P~ over each tile's keys
//   vedit            X = V + b
//   vsum             DG's rows for every (KV head, b_l), Sq (S over keys 1 .. 32 b_l - 1) and X(0)
//   apv_l7qs         the tile GEMM with its words and receipts, f to F; apv_l7qs_i: O to O (for the CPU reference)
//   apv_plain        the same tiles on P_prob and V with no edit: f = (float(P_prob V) s_q + z_q float(V(0))) s_ch, the
//                    baseline (P_prob zero at key 0 and past the query, as the quantizer leaves it)
//   apv_fu           apv_l7qs with P~ formed in registers from P_prob after each ldmatrix (no pedit pass)
//   apv_cf, apv_cc   timing only: ts_pv_adopted's cheater at a one-hot P row (T's words and O's fold, no tail stage), f
//                    not served (cf) or served (cc)
// Modes:
//   apv gate DIR S H REP SEED       poisoned buffers, every tile: apv_l7qs_i's receipts against apv_l7qs's, a no-write
//                                   control, apv_fu's f and receipts against apv_l7qs's; sampled tiles' operands, O, f
//                                   and receipts to DIR for apv_ref.py
//   apv bench REPS S H REP          hot items interleaved across APV_ARMS, one JSON line
//   apv one ARM S H REP             three warm-up calls, then one between cudaProfilerStart and Stop (for ncu)
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a apv.cu -lcuda -lcublasLt -lnvidia-ml
// Run: aprun.sh (build, SASS gate, device gates, apv_ref.py, then bench, ncu and eps_profile.py)
#define CGEMM_LIB
#include "cgemm.cu"

constexpr uint32_t AU_TAG = 14;
constexpr int AP_Y1S = 4096;    // |Ycum| <= 128 s / 32: Y1 alone (Y2 = 0) up to here
constexpr int AP_MAXS = 32768;  // |Y2| <= 2

struct ApArgs {
  const Maps* maps;   // a: P~ (or P_prob) (H s x s), b: X^T (or V^T) (kvh 128 x s), pa: P_A (s x 128)
  const int4* tab;    // a tile: h, qt, kvh
  int ntiles;
  const int8_t* DG;
  const int* Sq;
  const int* X0;
  const uint32_t* Uf;   // FUSE: V_u = U + 64 in fragment order (uedit)
  const float *sp, *sz, *sv;
  float* F;
  int* O;
  uint32_t* R;
  int s, nJ;
  uint32_t seed, KR;
};

// e_l7qs's loop with its tiles from the table: the producer loads a tile's qt + 1 stages of P~ and X^T, then (TAIL) its
// tail stage, P_A's 128 query rows alone. FUSE: the A tiles are P_prob and P~ is formed after each ldmatrix: U has period
// 32 in the key, so a lane's fragments (rows g, g + 8 of each 16-row block, bytes 4 q and 16 + 4 q of the atom) take the
// same 4 words of V_u = U + 64 in [0, 128] at every atom, and with P_prob in [0, 127] no byte carries:
// P~ = (P_prob + V_u) ^ 0x80808080, zeroed past the diagonal and at key 0
template <bool PLAIN, bool TAIL = !PLAIN, bool FUSE = false>
struct APL : EL<PLAIN ? NONE : DUO, PLAIN ? 0 : 7, PLAIN ? 0 : 4, true> {
  using E = EL<PLAIN ? NONE : DUO, PLAIN ? 0 : 7, PLAIN ? 0 : 4, true>;
  using E::s; using E::pt; using E::pkt; using E::ps; using E::pph; using E::maps; using E::issuer;
  using E::stage; using E::full; using E::empty; using E::advance; using E::wait_full; using E::load; using E::load_a;
  using E::load_b;
  static constexpr int MI = E::MI, NI = E::NI, NKR = E::NKR;
  const int4* tab = nullptr;
  int ntiles = 0, s_len = 0;
  int jpt = -1, jA = 0, jB = 0, jKT = 0, jPA = 0;
  const uint4* uf = nullptr;
  int lg = 0, lq = 0;
  DEV APL(uint8_t* smem, const Maps* m) : E(smem, m, 128, 128, 256, 8, TAIL, 1) {}
  // FUSE: the lane's V_u words of 32-row block blk (uedit's fragment order, 8 words a lane), read at every atom: held in
  // registers across the tile they spill at the 128-register cap
  DEV void set_uf(const uint32_t* Uf, int blk, int lane) {
    lg = lane >> 2; lq = lane & 3;
    uf = reinterpret_cast<const uint4*>(Uf) + (size_t)blk * 64 + 2 * lane;
  }
  DEV void ld(uint32_t (&fa)[MI][4], uint32_t (&fb)[NI][2], uint32_t sb, int kk) const {
    load(fa, fb, sb, kk);
    if constexpr (FUSE) {
      static_assert(MI == 2, "two 16-row blocks a warp row");
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        const uint4 u = __ldg(uf + mi);
        fa[mi][0] = (fa[mi][0] + u.x) ^ 0x80808080u; fa[mi][1] = (fa[mi][1] + u.y) ^ 0x80808080u;
        fa[mi][2] = (fa[mi][2] + u.z) ^ 0x80808080u; fa[mi][3] = (fa[mi][3] + u.w) ^ 0x80808080u;
      }
    }
  }
  // FUSE: key 0 (byte 0 of a0 and a1 in the lanes with q = 0 at the tile's first atom) is its own block
  DEV void mask_key0(uint32_t (&fa)[MI][4]) const {
    if (FUSE && lq == 0) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++) { fa[mi][0] &= 0xFFFFFF00u; fa[mi][1] &= 0xFFFFFF00u; }
    }
  }
  DEV void issue() {
    if (pt >= ntiles) return;
    if (pt != jpt) {
      const int4 e = __ldg(tab + pt);
      jA = e.x * s_len + 128 * e.y; jB = e.z * 128; jKT = e.y + 1; jPA = 128 * e.y;
      jpt = pt;
    }
    mbar_wait(empty(ps), pph ^ 1);
    if (pkt < jKT) {
      mbar_expect_tx(full(ps), E::STAGE);
      tma_load(stage(ps), &maps->a, full(ps), pkt * E::BK, jA);
      tma_load(stage(ps) + E::A_BYTES, &maps->b, full(ps), pkt * E::BK, jB);
    } else {
      mbar_expect_tx(full(ps), E::A_BYTES);
      tma_load(stage(ps), &maps->pa, full(ps), 0, jPA);
    }
    if (++pkt == jKT + (TAIL ? 1 : 0)) { pkt = 0; pt += gridDim.x; }
    if (++ps == E::NST) { ps = 0; pph ^= 1; }
  }
  DEV void prologue() {
    if (issuer) {
      tma_acquire(&maps->a); tma_acquire(&maps->b); tma_acquire(&maps->pa);
      for (int i = 0; i < E::NST; i++) issue();
    }
    __syncwarp();
  }
  DEV void release(uint32_t done) {
    mbar_release(done);
    if (issuer) issue();
    __syncwarp();
  }
  // EL's stage_il and tile_il over KT full stages (they call this release, and so this issue)
  template <bool FIRST, bool LAST>
  DEV void stage_a(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[NKR], uint32_t (&fa)[2][MI][4],
                   uint32_t (&fb)[2][NI][2], uint32_t zero) {
    const uint32_t sb = stage(s), done = empty(s);
    ld(fa[1], fb[1], sb, 1);
    E::template atom<FIRST, 0>(acc, ch, kr, fa[0], fb[0], zero);
    ld(fa[0], fb[0], sb, 2);
    E::template atom<false, 1>(acc, ch, kr, fa[1], fb[1], zero);
    ld(fa[1], fb[1], sb, 3);
    E::template atom<false, 2>(acc, ch, kr, fa[0], fb[0], zero);
    advance();
    if (!LAST) {
      wait_full();
      ld(fa[0], fb[0], stage(s), 0);
    }
    E::template atom<false, 3>(acc, ch, kr, fa[1], fb[1], zero);
    release(done);
  }
  DEV void tile_a(int KT, int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[NKR], uint32_t zero) {
    uint32_t fa[2][MI][4], fb[2][NI][2];
    wait_full();
    ld(fa[0], fb[0], stage(s), 0);
    mask_key0(fa[0]);
    if (KT == 1) {
      stage_a<true, true>(acc, ch, kr, fa, fb, zero);
    } else {
      stage_a<true, false>(acc, ch, kr, fa, fb, zero);
#pragma unroll 1
      for (int kt = 1; kt < KT - 1; kt++) stage_a<false, false>(acc, ch, kr, fa, fb, zero);
      stage_a<false, true>(acc, ch, kr, fa, fb, zero);
    }
    if constexpr (E::R) E::flush(ch, kr, acc);
  }
  // the diagonal stage: atom t0 + pos's words absorbed right after its MMAs while t0 + pos <= b_l. FUSE: atom pos against
  // warp row bl - t0 is whole before it, the triangle key <= query on it (byte b of a fragment word at k = kb + b of row
  // rr in the block: kept while kb + b <= rr), zero after it
  template <bool FIRST> DEV void dstage(int t0, int bl, int (&acc)[MI][NI][4], St (&ch)[8]) {
    wait_full();
    const uint32_t sb = stage(s), done = empty(s);
#pragma unroll
    for (int pos = 0; pos < 4; pos++) {
      uint32_t fa[MI][4], fb[NI][2];
      ld(fa, fb, sb, pos);
      if constexpr (FUSE) {
        const int w = bl - t0;
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const int nv = min(max(16 * mi + 8 * (i & 1) + lg - (4 * lq + 16 * (i >> 1)) + 1, 0), 4);
            const uint32_t tri = nv == 4 ? 0xFFFFFFFFu : (1u << (8 * nv)) - 1u;
            fa[mi][i] &= pos < w ? 0xFFFFFFFFu : pos == w ? tri : 0u;
          }
        if (FIRST && pos == 0) mask_key0(fa);
      }
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          if (FIRST && pos == 0) imma0(acc[mi][ni], fa[mi], fb[ni]);
          else imma(acc[mi][ni], fa[mi], fb[ni]);
        }
      if (t0 + pos <= bl) cabs_all<APL>(ch, acc);
    }
    release(done);
    advance();
  }
  template <int KK> DEV static void ldg_b(uint32_t (&fb)[NI][2], const int8_t* dgw, int g, int q) {
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
      const int8_t* p = dgw + (size_t)(8 * ni + g) * 128 + 32 * KK + 4 * q;
      fb[ni][0] = __ldg(reinterpret_cast<const uint32_t*>(p));
      fb[ni][1] = __ldg(reinterpret_cast<const uint32_t*>(p + 16));
    }
  }
  // the tail stage: O = T_last + (-U) Y0 + (-U M_d) X_diag (atoms 0, 2) + 256 (-U) Y1 (atom 1) + (Y2S) 65536 (-U) Y2
  // (atom 1 against DG's fourth atom), absorbed; OUT 0: then + 64 S_diag (atom 3, 64 M_d, against X_diag) for f
  template <int OUT> DEV void atail(int (&acc)[MI][NI][4], St (&ch)[8], const int8_t* dgw, int g, int q, bool y2s) {
    wait_full();
    const uint32_t tb = stage(s);
    uint32_t fa[MI][4], fb[NI][2];
    ldg_b<0>(fb, dgw, g, q);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      load_a(fa[mi], tb, 0, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) imma(acc[mi][ni], fa[mi], fb[ni]);
    }
    ldg_b<2>(fb, dgw, g, q);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      load_a(fa[mi], tb, 2, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) imma(acc[mi][ni], fa[mi], fb[ni]);
    }
    ldg_b<1>(fb, dgw, g, q);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      load_a(fa[mi], tb, 1, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int hi[4];
        imma0(hi, fa[mi], fb[ni]);
#pragma unroll
        for (int e = 0; e < 4; e++) acc[mi][ni][e] = (int)((uint32_t)acc[mi][ni][e] + ((uint32_t)hi[e] << 8));
      }
    }
    if (y2s) {
      ldg_b<3>(fb, dgw, g, q);
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        load_a(fa[mi], tb, 1, mi);
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          int hi[4];
          imma0(hi, fa[mi], fb[ni]);
#pragma unroll
          for (int e = 0; e < 4; e++) acc[mi][ni][e] = (int)((uint32_t)acc[mi][ni][e] + ((uint32_t)hi[e] << 16));
        }
      }
    }
    cabs_all<APL>(ch, acc);
    if constexpr (OUT == 0) {
      ldg_b<2>(fb, dgw, g, q);
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        load_a(fa[mi], tb, 3, mi);
#pragma unroll
        for (int ni = 0; ni < NI; ni++) imma(acc[mi][ni], fa[mi], fb[ni]);
      }
    }
    release(empty(s));
    advance();
  }
};

// OUT 0: f to F; 1: O to O (s32). PLAIN: no edit, no words, f = (float(acc) s_q + z_q float(V(0))) s_ch. OUT 2 and 3 time
// ts_pv_adopted's cheater at a one-hot P row, where O is a copy: T's words, then O folded from T_last with no tail stage
// (the strip, its triangle and the sums skipped); 2 serves nothing (served free), 3 runs f's epilogue (served checked).
// FUSE: maps' a is P_prob and P~ is formed in registers (APL)
template <bool PLAIN, int OUT, bool FUSE> DEV void apv_body(const ApArgs& a) {
  using L = APL<PLAIN, !PLAIN && OUT < 2, FUSE>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps);
  ml.tab = a.tab; ml.ntiles = a.ntiles; ml.s_len = a.s;
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  const uint32_t zero = a.KR;
  uint32_t kr[L::NKR];
  kr[0] = 0;
  for (int t = blockIdx.x; t < a.ntiles; t += gridDim.x) {
    const int4 te = __ldg(a.tab + t);
    const int h = te.x, qt = te.y, kvh = te.z, bl = 4 * qt + wm;
    St ch[8];
    if constexpr (!PLAIN) {
#pragma unroll
      for (int e = 0; e < 8; e++) {
        ch[e].y = mix32(a.seed ^ mix32(((uint32_t)t * L::NCT + ctid) * 8 + e));
        ch[e].z = mix32(ch[e].y + 0x9E3779B9u);
      }
    }
    int acc[MI][NI][4];
    if constexpr (FUSE) ml.set_uf(a.Uf, 4 * qt + wm, lane);
    if constexpr (PLAIN) {
      ml.tile_a(qt + 1, acc, ch, kr, zero);
    } else {
      if (qt > 0) {
        ml.tile_a(qt, acc, ch, kr, zero);
        ml.template dstage<false>(4 * qt, bl, acc, ch);
      } else {
        ml.template dstage<true>(0, bl, acc, ch);
      }
      if constexpr (OUT < 2)
        ml.template atail<OUT>(acc, ch, a.DG + ((size_t)(kvh * a.nJ + bl) * 128 + wn * 32) * 128, g, q, a.s > AP_Y1S);
      else cabs_all<L>(ch, acc);
    }
    const int r0 = wm * 32 + g, c0 = wn * 32 + 2 * q;
    const size_t orow = (size_t)h * a.s + 128 * qt + r0;
    if constexpr (OUT == 1) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          int* p = a.O + (orow + 16 * mi) * 128 + c0 + 8 * ni;
          *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
          *reinterpret_cast<int2*>(p + 8 * 128) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    } else if constexpr (OUT != 2) {
      float rs[MI][2], rz[MI][2];
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          rs[mi][hh] = __ldg(a.sp + orow + 16 * mi + 8 * hh);
          rz[mi][hh] = __ldg(a.sz + orow + 16 * mi + 8 * hh);
        }
      const int* Sr = a.Sq + (size_t)(kvh * a.nJ + bl) * 128;
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        const float2 cs = __ldg(reinterpret_cast<const float2*>(a.sv + kvh * 128 + c0 + 8 * ni));
        const int2 x0 = __ldg(reinterpret_cast<const int2*>(a.X0 + kvh * 128 + c0 + 8 * ni));
        int2 sq = make_int2(0, 0);
        if constexpr (!PLAIN) sq = __ldg(reinterpret_cast<const int2*>(Sr + c0 + 8 * ni));
        const float xf[2] = {__int2float_rn(x0.x), __int2float_rn(x0.y)};
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
          float f[4];
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const int v = (int)((uint32_t)acc[mi][ni][e] + 64u * (uint32_t)(e % 2 ? sq.y : sq.x));
            const float body = __fmul_rn(__int2float_rn(v), rs[mi][e / 2]), sink = __fmul_rn(rz[mi][e / 2], xf[e % 2]);
            f[e] = __fmul_rn(__fadd_rn(body, sink), e % 2 ? cs.y : cs.x);
          }
          float* p = a.F + (orow + 16 * mi) * 128 + c0 + 8 * ni;
          *reinterpret_cast<float2*>(p) = make_float2(f[0], f[1]);
          *reinterpret_cast<float2*>(p + 8 * 128) = make_float2(f[2], f[3]);
        }
      }
    }
    if constexpr (!PLAIN) {
      uint4* r = reinterpret_cast<uint4*>(a.R) + ((size_t)t * L::NCT + ctid) * 2;
      r[0] = make_uint4(finish<DUO>(ch, 0), finish<DUO>(ch, 1), finish<DUO>(ch, 2), finish<DUO>(ch, 3));
      r[1] = make_uint4(finish<DUO>(ch, 4), finish<DUO>(ch, 5), finish<DUO>(ch, 6), finish<DUO>(ch, 7));
    }
  }
}
using ApKern = void (*)(const ApArgs);
#define APK(NAME, PLAIN, OUT, FUSE) \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ ApArgs a) { apv_body<PLAIN, OUT, FUSE>(a); }
APK(apv_l7qs, false, 0, false)
APK(apv_l7qs_i, false, 1, false)
APK(apv_plain, true, 0, false)
APK(apv_cf, false, 2, false)
APK(apv_cc, false, 3, false)
APK(apv_fu, false, 0, true)

// ---------------------------------------------------------------- producers
DEV int u_of(uint32_t byte) { const int u = (int)(byte & 63u) + 1; return byte & 64u ? -u : u; }
// a thread a row pair (one digest): U - 64 (32 bytes a row), P_A's row [-U | -U | -U M_d | 64 M_d] and apv_fu's V_u
__global__ void __launch_bounds__(128) uedit(int8_t* __restrict__ Um, int8_t* __restrict__ PA, uint32_t* __restrict__ Uf, int s,
                                             const __grid_constant__ Sig sg) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (2 * j >= s) return;
  uint64_t st[8];
  sha_block(sg.w, (uint64_t)AU_TAG << 56 | (uint64_t)j, st);
  uint8_t by[64];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int k = 0; k < 8; k++) by[8 * i + k] = (uint8_t)(st[i] >> (56 - 8 * k));
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int qq = 2 * j + h, qd = qq % 32;
    uint32_t um[8], nu[8], nm[8], mk[8];
#pragma unroll
    for (int w4 = 0; w4 < 8; w4++) {
      um[w4] = nu[w4] = nm[w4] = mk[w4] = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int rho = 4 * w4 + i, u = u_of(by[32 * h + rho]);
        const bool md = rho <= qd && !(qq < 32 && rho == 0);
        um[w4] |= (uint32_t)((u - 64) & 0xFF) << (8 * i);
        nu[w4] |= (uint32_t)(-u & 0xFF) << (8 * i);
        nm[w4] |= (uint32_t)((md ? -u : 0) & 0xFF) << (8 * i);
        mk[w4] |= (uint32_t)(md ? 64 : 0) << (8 * i);
      }
    }
    uint4* ur = reinterpret_cast<uint4*>(Um + (size_t)qq * 32);
    ur[0] = make_uint4(um[0], um[1], um[2], um[3]); ur[1] = make_uint4(um[4], um[5], um[6], um[7]);
    // apv_fu's V_u = U + 64 in fragment order: row qq = 32 B + 16 mi + 8 h + g, word cw (q = cw % 4, hh = cw / 4) goes to
    // lane 4 g + q's slot 4 mi + 2 hh + h of block B
    const int rr = qq % 32, mi = rr / 16, hb = (rr / 8) % 2, g = rr % 8;
#pragma unroll
    for (int cw = 0; cw < 8; cw++)
      Uf[(size_t)(qq / 32) * 256 + (4 * g + cw % 4) * 8 + 4 * mi + 2 * (cw / 4) + hb] = um[cw] ^ 0x80808080u;
    uint4* pr = reinterpret_cast<uint4*>(PA + (size_t)qq * 128);
    pr[0] = make_uint4(nu[0], nu[1], nu[2], nu[3]); pr[1] = make_uint4(nu[4], nu[5], nu[6], nu[7]);
    pr[2] = pr[0]; pr[3] = pr[1];
    pr[4] = make_uint4(nm[0], nm[1], nm[2], nm[3]); pr[5] = make_uint4(nm[4], nm[5], nm[6], nm[7]);
    pr[6] = make_uint4(mk[0], mk[1], mk[2], mk[3]); pr[7] = make_uint4(mk[4], mk[5], mk[6], mk[7]);
  }
}
// a block a tile, a warp 32 of its query rows, 16 keys a lane: P~ = M (P_prob + (U - 64)), bytewise (no carries: P_prob
// in [0, 127], U - 64 in [-128, 0])
__global__ void __launch_bounds__(128) pedit(const int8_t* __restrict__ P, int8_t* __restrict__ Pt, const int8_t* __restrict__ Um,
                                             const int4* __restrict__ tab, int s) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int4 te = __ldg(tab + blockIdx.x);
  const int h = te.x, qt = te.y, K = 128 * (qt + 1);
#pragma unroll 1
  for (int rr = 0; rr < 32; rr++) {
    const int qq = 128 * qt + 32 * warp + rr;
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(Um + (size_t)qq * 32) + (lane & 1));
    const size_t row = ((size_t)h * s + qq) * s;
#pragma unroll 1
    for (int j = 16 * lane; j < K; j += 512) {
      const uint4 p = __ldg(reinterpret_cast<const uint4*>(P + row + j));
      uint32_t w[4] = {add_noise<false>(p.x, u.x), add_noise<false>(p.y, u.y), add_noise<false>(p.z, u.z), add_noise<false>(p.w, u.w)};
      if (j == 0) w[0] &= 0xFFFFFF00u;
      const int keep = qq - j + 1;
      if (keep < 16) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const int kb = keep - 4 * i;
          w[i] &= kb >= 4 ? ~0u : kb <= 0 ? 0u : (~0u >> (32 - 8 * kb));
        }
      }
      *reinterpret_cast<uint4*>(Pt + row + j) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
}
// X^T = V^T + b over patches (4 rows x 128 keys, cpv.cu's vedit's digests): grid (s / 128, rows / (4 VE_G)), a block
// VE_G patches of one key chunk, thread g < VE_G hashing patch g's digest, then a warp a patch at a time
constexpr int VE_G = 32;
__global__ void __launch_bounds__(256) vedit(const int8_t* __restrict__ Vt, int8_t* __restrict__ Xt, int rows, int s,
                                             const __grid_constant__ Sig sg) {
  __shared__ uint32_t dgs[VE_G][16];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, chunk = blockIdx.x, g0 = blockIdx.y * VE_G;
  if (threadIdx.x < VE_G && (g0 + (int)threadIdx.x) * 4 < rows) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)DITHER_TAG << 56 | ((uint64_t)(g0 + threadIdx.x) * gridDim.x + chunk), st);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      dgs[threadIdx.x][2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      dgs[threadIdx.x][2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int gi = warp; gi < VE_G && (g0 + gi) * 4 < rows; gi += 8) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dgs[gi]);
#pragma unroll
    for (int rr = 0; rr < 4; rr++) {
      const size_t off = (size_t)(4 * (g0 + gi) + rr) * s + 128 * chunk + 4 * lane;
      const uint32_t bits = (bytes[16 * rr + lane / 2] >> (4 * (lane & 1))) & 0xFu;
      *reinterpret_cast<uint32_t*>(Xt + off) = add_noise<false>(__ldg(reinterpret_cast<const uint32_t*>(Vt + off)), bits01(bits));
    }
  }
}
// DG's rows [Y0 | Y1 | X_diag | Y2] and Sq of every query block of one KV head, a warp a channel and a lane a residue r:
// each lane walks the blocks once, Ycum[r] before block j its running sum of keys 32 j' + r (j' < j, key 0 left out),
// S its warp sum; the block with j = 0 also writes X(0). Grid (128 / 4, KV heads), 4 warps a CTA
__global__ void __launch_bounds__(128) vsum(const int8_t* __restrict__ Xt, int8_t* __restrict__ DG, int* __restrict__ Sq,
                                            int* __restrict__ X0, int s) {
  const int kvh = blockIdx.y, ch = blockIdx.x * 4 + threadIdx.x / 32, r = threadIdx.x % 32, nJ = s / 32;
  const int8_t* xr = Xt + (size_t)(kvh * 128 + ch) * s;
  int Y = 0, S = 0;
#pragma unroll 4
  for (int j = 0; j < nJ; j++) {
    const int xv = __ldg(xr + 32 * j + r);
    int8_t* dst = DG + ((size_t)(kvh * nJ + j) * 128 + ch) * 128;
    const int y0 = (int)(int8_t)(Y & 0xFF), r1 = (Y - y0) >> 8, y1 = (int)(int8_t)(r1 & 0xFF), y2 = (r1 - y1) >> 8;
    dst[r] = (int8_t)y0; dst[32 + r] = (int8_t)y1; dst[64 + r] = (int8_t)xv; dst[96 + r] = (int8_t)y2;
    if (r == 0) Sq[(size_t)(kvh * nJ + j) * 128 + ch] = S;
    if (j == 0 && r == 0) X0[kvh * 128 + ch] = xv;
    const int v = (j == 0 && r == 0) ? 0 : xv;
    Y += v;
    int t = v;
#pragma unroll
    for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xFFFFFFFFu, t, o);
    S += t;
  }
}
// vedit and vsum in one pass. A CTA a patch row group grp (channels 4 grp % 128 .. + 3 of KV head grp / 32): its threads
// hash the row group's s / 128 patch digests (one a thread), then its 8 warps take a 128-key chunk each a round. Lane l
// holds keys 32 jj + 4 t .. + 3 of its chunk (jj = l / 8, t = l % 8: block j = 4 c + jj, residues 4 t + i), forms X = V + b
// and, with SUMS, X's bytes biased to [0, 255] as two 16-bit pairs (bytes 0 and 2, 1 and 3), scanned over jj by
// shuffles, over the round's warps through shared memory and over earlier rounds in registers, so Yb = Ycum + 128 is
// whole a lane a residue: Y0 = byte 0 of Yb ^ 0x80, Y1 = byte 1 of Yb and Y2 = byte 2 of Yb + 32768 (D3, s > AP_Y1S;
// else 0, as Y2 is there). Key 0 enters the sums as 0. SUMS false is X alone: vedit's output, what a one-hot cheater
// still needs
constexpr int VF_NT = 256, VF_SM_FIXED = 4 * (512 + 512 + 64);
template <bool SUMS, bool D3>
__global__ void __launch_bounds__(VF_NT, 2) vfold(const int8_t* __restrict__ Vt, int8_t* __restrict__ Xt, int8_t* __restrict__ DG,
                                                 int* __restrict__ Sq, int* __restrict__ X0, int s, const __grid_constant__ Sig sg) {
  extern __shared__ uint32_t vfs[];   // digests (nC x 16 words), then tot[w][rr][t][2], pre[w][rr][t][2], rtot[rr][t][2]
  const int grp = blockIdx.x, nC = s / 128, nJ = s / 32, kvh = grp / 32, ch0 = 4 * (grp % 32);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, jj = lane >> 3, t = lane & 7;
  uint32_t *tot = vfs + 16 * nC, *pre = tot + 512, *rtot = pre + 512;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(vfs);
  const int8_t* vrow = Vt + (size_t)grp * 4 * s + 4 * lane;
  int8_t* xrow = Xt + (size_t)grp * 4 * s + 4 * lane;
  uint32_t vw[4];
#pragma unroll
  for (int rr = 0; rr < 4; rr++) vw[rr] = warp < nC ? __ldg(reinterpret_cast<const uint32_t*>(vrow + (size_t)rr * s + 128 * warp)) : 0u;
  for (int i = threadIdx.x; i < nC; i += VF_NT) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)DITHER_TAG << 56 | ((uint64_t)grp * nC + i), st);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      vfs[16 * i + 2 * k] = __byte_perm((uint32_t)(st[k] >> 32), 0, 0x0123);
      vfs[16 * i + 2 * k + 1] = __byte_perm((uint32_t)st[k], 0, 0x0123);
    }
  }
  __syncthreads();
  int B[4][4] = {};   // a row's biased sums over the earlier rounds' blocks, a byte i (residue 4 t + i)
#pragma unroll 1
  for (int c0 = 0; c0 < nC; c0 += 8) {
    const int c = c0 + warp;
    const bool live = c < nC;
    uint32_t vn[4], xw[4], e02[4], e13[4];
#pragma unroll
    for (int rr = 0; rr < 4; rr++)
      vn[rr] = c + 8 < nC ? __ldg(reinterpret_cast<const uint32_t*>(vrow + (size_t)rr * s + 128 * (c + 8))) : 0u;
#pragma unroll
    for (int rr = 0; rr < 4; rr++) {
      uint32_t x = 0;
      if (live) {
        const uint32_t bits = (bytes[64 * c + 16 * rr + lane / 2] >> (4 * (lane & 1))) & 0xFu;
        x = add_noise<false>(vw[rr], bits01(bits));
        *reinterpret_cast<uint32_t*>(xrow + (size_t)rr * s + 128 * c) = x;
        if (c == 0 && lane == 0) X0[4 * grp + rr] = (int)(int8_t)(x & 0xFFu);
      }
      xw[rr] = x;
      if constexpr (SUMS) {
        const uint32_t u = live ? ((c == 0 && lane == 0) ? x & 0xFFFFFF00u : x) ^ 0x80808080u : 0u;
        const uint32_t p02 = u & 0x00FF00FFu, p13 = (u >> 8) & 0x00FF00FFu;
        uint32_t i02 = p02, i13 = p13, a02 = __shfl_up_sync(~0u, i02, 8), a13 = __shfl_up_sync(~0u, i13, 8);
        if (lane >= 8) { i02 += a02; i13 += a13; }
        a02 = __shfl_up_sync(~0u, i02, 16); a13 = __shfl_up_sync(~0u, i13, 16);
        if (lane >= 16) { i02 += a02; i13 += a13; }
        e02[rr] = i02 - p02; e13[rr] = i13 - p13;
        if (jj == 3) { tot[warp * 64 + (rr * 8 + t) * 2] = i02; tot[warp * 64 + (rr * 8 + t) * 2 + 1] = i13; }
      }
    }
#pragma unroll
    for (int rr = 0; rr < 4; rr++) vw[rr] = vn[rr];
    if constexpr (SUMS) {
      __syncthreads();
      if (threadIdx.x < 64) {
        uint32_t acc = 0;
#pragma unroll
        for (int w = 0; w < 8; w++) { const uint32_t v = tot[w * 64 + threadIdx.x]; pre[w * 64 + threadIdx.x] = acc; acc += v; }
        rtot[threadIdx.x] = acc;
      }
      __syncthreads();
      if (live) {
        const int j = 4 * c + jj, off = 128 - 128 * j;
        uint32_t* drow = reinterpret_cast<uint32_t*>(DG + ((size_t)(kvh * nJ + j) * 128 + ch0) * 128) + t;
#pragma unroll
        for (int rr = 0; rr < 4; rr++) {
          const int k = warp * 64 + (rr * 8 + t) * 2;
          const uint32_t s02 = pre[k] + e02[rr], s13 = pre[k + 1] + e13[rr];
          const int y0 = B[rr][0] + (int)(s02 & 0xFFFFu) + off, y2 = B[rr][2] + (int)(s02 >> 16) + off;
          const int y1 = B[rr][1] + (int)(s13 & 0xFFFFu) + off, y3 = B[rr][3] + (int)(s13 >> 16) + off;
          const uint32_t t01 = __byte_perm(y0, y1, 0x5140), t23 = __byte_perm(y2, y3, 0x5140);
          uint32_t* d = drow + rr * 32;
          d[0] = __byte_perm(t01, t23, 0x5410) ^ 0x80808080u;
          d[8] = __byte_perm(t01, t23, 0x7632);
          d[16] = xw[rr];
          uint32_t w2 = 0;
          if constexpr (D3) {
            const uint32_t q01 = __byte_perm(y0 + 32768, y1 + 32768, 0x0062), q23 = __byte_perm(y2 + 32768, y3 + 32768, 0x0062);
            w2 = __byte_perm(q01, q23, 0x5410);
          }
          d[24] = w2;
          int sm = y0 + y1 + y2 + y3;
          sm += __shfl_xor_sync(~0u, sm, 1);
          sm += __shfl_xor_sync(~0u, sm, 2);
          sm += __shfl_xor_sync(~0u, sm, 4);
          if (t == 0) Sq[(size_t)(kvh * nJ + j) * 128 + ch0 + rr] = sm - 4096;
        }
      }
#pragma unroll
      for (int rr = 0; rr < 4; rr++) {
        const uint32_t r02 = rtot[(rr * 8 + t) * 2], r13 = rtot[(rr * 8 + t) * 2 + 1];
        B[rr][0] += (int)(r02 & 0xFFFFu); B[rr][2] += (int)(r02 >> 16);
        B[rr][1] += (int)(r13 & 0xFFFFu); B[rr][3] += (int)(r13 >> 16);
      }
    }
  }
}
// the producers' check against Ycum itself: a thread a (KV head, channel, residue) walks the blocks, Ycum in full, and
// counts each DG row whose Y0 + 256 Y1 + 65536 Y2 is not Ycum or whose X_diag is not X, and each Sq that is not the sum
// of Ycum
__global__ void vcheck(const int8_t* __restrict__ Xt, const int8_t* __restrict__ DG, const int* __restrict__ Sq, int rows, int s,
                       unsigned long long* bad) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x, row = g / 32, r = g % 32, nJ = s / 32;
  if (row >= rows) return;
  const int kvh = row / 128, ch = row % 128;
  const int8_t* xr = Xt + (size_t)row * s;
  long long Y = 0;
  unsigned long long c = 0;
  for (int j = 0; j < nJ; j++) {
    const int8_t* dg = DG + ((size_t)(kvh * nJ + j) * 128 + ch) * 128;
    const long long d = (long long)dg[r] + 256LL * dg[32 + r] + 65536LL * dg[96 + r];
    c += d != Y || dg[64 + r] != xr[32 * j + r];
    long long S = Y;
#pragma unroll
    for (int o = 16; o; o >>= 1) S += __shfl_xor_sync(0xFFFFFFFFu, S, o);
    if (r == 0) c += Sq[(size_t)(kvh * nJ + j) * 128 + ch] != S;
    if (!(j == 0 && r == 0)) Y += xr[32 * j + r];
  }
  if (c) atomicAdd(bad, c);
}
// the scales: s_q in [2^-11, 2^-10), z_q in [2^-12, 2^-11), s_ch in [2^-10, 2^-9)
__global__ void apfill(float* sp, float* sz, int nq, float* sv, int nc, uint32_t seed) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 2 * nq + nc; i += gridDim.x * blockDim.x) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)i));
    if (i < nq) sp[i] = __uint_as_float(0x3A000000u | (h & 0x7FFFFFu));
    else if (i < 2 * nq) sz[i - nq] = __uint_as_float(0x39800000u | (h & 0x7FFFFFu));
    else sv[i - 2 * nq] = __uint_as_float(0x3A800000u | (h & 0x7FFFFFu));
  }
}
// P_prob as the quantizer leaves it: 0 at key 0 (its own block) and past the query; V(0) a channel, plain's sink operand.
// A block a (query, head) row, so no index is divided (the SASS gate pins no runtime integer division)
__global__ void apmask(int8_t* P, int s, const int8_t* Vt, int* V0, int vrows) {
  const int qq = blockIdx.x;
  int8_t* row = P + ((size_t)blockIdx.y * s + qq) * s;
  for (int key = threadIdx.x; key < s; key += blockDim.x)
    if (key == 0 || key > qq) row[key] = 0;
  if (qq == 0 && blockIdx.y == 0)
    for (int i = threadIdx.x; i < vrows; i += blockDim.x) V0[i] = Vt[(size_t)i * s];
}
__global__ void apcmp(const uint32_t* a, const uint32_t* b, size_t n, unsigned long long* bad) {
  unsigned long long c = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += a[i] != b[i];
  if (c) atomicAdd(bad, c);
}

// ---------------------------------------------------------------- host
struct AP {
  int s = 0, H = 0, rep = 1, kvh = 0, nJ = 0, ntiles = 0, grid = 0;
  uint32_t seed = 0, inst = 0;
  cudaStream_t st;
  int8_t *P, *Vt, *Pt, *Xt, *Um, *PA, *DG;
  int *Sq, *X0, *V0, *O;
  float *sp, *sz, *sv, *F, *Fr;
  uint32_t *R, *R2, *R3, *Uf;
  int4* tab;
  unsigned long long* bad;
  std::vector<int4> htab;
  Maps *maps, *pmaps, *fmaps;
};
static void apsetup(AP& V, int s, int H, int rep, uint32_t seed, bool gate) {
  if (s % 256 || H % rep || s < 256 || s > AP_MAXS) {
    fprintf(stderr, "shape s %d H %d rep %d: s %% 256 == 0, 256 <= s <= %d, H %% rep == 0\n", s, H, rep, AP_MAXS);
    exit(2);
  }
  V.s = s; V.H = H; V.rep = rep; V.kvh = H / rep; V.nJ = s / 32; V.seed = seed;
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  // tiles longest first, heads innermost (the rep heads of a KV head read its X^T and DG together), dealt to the grid's
  // CTAs in a snake (round r reversed when odd) so each CTA's total keys stay near the mean
  std::vector<int4> lin;
  for (int qt = s / 128 - 1; qt >= 0; qt--)
    for (int h = 0; h < H; h++) lin.push_back(make_int4(h, qt, h / rep, 0));
  V.ntiles = (int)lin.size();
  V.grid = std::min(V.ntiles, nsm);
  for (int r = 0; r * V.grid < V.ntiles; r++)
    for (int b = 0; b < V.grid && r * V.grid + b < V.ntiles; b++) {
      const int i = r % 2 ? std::min(V.ntiles, (r + 1) * V.grid) - 1 - b : r * V.grid + b;
      V.htab.push_back(lin[i]);
    }
  CK(cudaStreamCreateWithFlags(&V.st, cudaStreamNonBlocking));
  const size_t ps = (size_t)H * s * s, vs = (size_t)V.kvh * 128 * s, nt = (size_t)V.ntiles;
  CK(cudaMalloc(&V.P, ps)); CK(cudaMalloc(&V.Pt, ps)); CK(cudaMalloc(&V.Vt, vs)); CK(cudaMalloc(&V.Xt, vs));
  CK(cudaMalloc(&V.Um, (size_t)s * 32)); CK(cudaMalloc(&V.PA, (size_t)s * 128));
  CK(cudaMalloc(&V.DG, (size_t)V.kvh * V.nJ * 128 * 128)); CK(cudaMalloc(&V.Sq, (size_t)V.kvh * V.nJ * 128 * 4));
  CK(cudaMalloc(&V.X0, (size_t)V.kvh * 128 * 4)); CK(cudaMalloc(&V.V0, (size_t)V.kvh * 128 * 4));
  CK(cudaMalloc(&V.sp, (size_t)H * s * 4)); CK(cudaMalloc(&V.sz, (size_t)H * s * 4)); CK(cudaMalloc(&V.sv, (size_t)V.kvh * 128 * 4));
  CK(cudaMalloc(&V.F, (size_t)H * s * 128 * 4)); CK(cudaMalloc(&V.O, (size_t)H * s * 128 * 4));
  CK(cudaMalloc(&V.Fr, (size_t)H * s * 128 * 4));
  CK(cudaMalloc(&V.R, nt * MLW::NCT * 8 * 4)); CK(cudaMalloc(&V.R2, nt * MLW::NCT * 8 * 4));
  CK(cudaMalloc(&V.R3, nt * MLW::NCT * 8 * 4)); CK(cudaMalloc(&V.Uf, (size_t)s * 32));
  CK(cudaMalloc(&V.tab, nt * sizeof(int4))); CK(cudaMalloc(&V.bad, 8));
  CK(cudaMemcpy(V.tab, V.htab.data(), nt * sizeof(int4), cudaMemcpyHostToDevice));
  sfill<I8><<<4096, 256, 0, V.st>>>(reinterpret_cast<uint8_t*>(V.P), ps, mix32(seed ^ 0x1111u), 0, 128);
  sfill<I8><<<4096, 256, 0, V.st>>>(reinterpret_cast<uint8_t*>(V.Vt), vs, mix32(seed ^ 0x2222u), -128, 255);
  if (gate) {   // P's last two rows of every head at 127 and 0 (P~ reaches 127 and -128), V's first two channels at 126 and -128
    for (int h = 0; h < H; h++) {
      CK(cudaMemsetAsync(V.P + ((size_t)h * s + s - 2) * s, 0x7F, s, V.st));
      CK(cudaMemsetAsync(V.P + ((size_t)h * s + s - 1) * s, 0x00, s, V.st));
    }
    CK(cudaMemsetAsync(V.Vt, 0x7E, s, V.st)); CK(cudaMemsetAsync(V.Vt + s, 0x80, s, V.st));
  }
  apmask<<<dim3(s, H), 256, 0, V.st>>>(V.P, s, V.Vt, V.V0, V.kvh * 128);
  apfill<<<256, 256, 0, V.st>>>(V.sp, V.sz, H * s, V.sv, V.kvh * 128, mix32(seed ^ 0x3333u));
  V.maps = upload_maps(Maps{tmap(V.Pt, s, H * s, 128), tmap(V.Xt, s, V.kvh * 128, 128), tmap(V.PA, 128, s, 128), tmap(V.PA, 128, s, 128)});
  V.pmaps = upload_maps(Maps{tmap(V.P, s, H * s, 128), tmap(V.Vt, s, V.kvh * 128, 128), tmap(V.PA, 128, s, 128), tmap(V.PA, 128, s, 128)});
  V.fmaps = upload_maps(Maps{tmap(V.P, s, H * s, 128), tmap(V.Xt, s, V.kvh * 128, 128), tmap(V.PA, 128, s, 128), tmap(V.PA, 128, s, 128)});
  for (ApKern k : {apv_l7qs, apv_l7qs_i, apv_plain, apv_cf, apv_cc, apv_fu})
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
}
static void apfree(AP& V) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)V.P, (void*)V.Pt, (void*)V.Vt, (void*)V.Xt, (void*)V.Um, (void*)V.PA, (void*)V.DG, (void*)V.Sq, (void*)V.X0,
                  (void*)V.V0, (void*)V.sp, (void*)V.sz, (void*)V.sv, (void*)V.F, (void*)V.O, (void*)V.R, (void*)V.R2, (void*)V.tab,
                  (void*)V.bad, (void*)V.maps, (void*)V.pmaps, (void*)V.fmaps, (void*)V.Fr, (void*)V.R3, (void*)V.Uf})
    CK(cudaFree(p));
  CK(cudaStreamDestroy(V.st));
}
static void ued(AP& V, const Sig& s) { uedit<<<(V.s / 2 + 127) / 128, 128, 0, V.st>>>(V.Um, V.PA, V.Uf, V.s, s); }
static void ped(AP& V) { pedit<<<V.ntiles, 128, 0, V.st>>>(V.P, V.Pt, V.Um, V.tab, V.s); }
static void ved(AP& V, const Sig& s) {
  vedit<<<dim3(V.s / 128, (V.kvh * 128 + 4 * VE_G - 1) / (4 * VE_G)), 256, 0, V.st>>>(V.Vt, V.Xt, V.kvh * 128, V.s, s);
}
static void vsm(AP& V) { vsum<<<dim3(32, V.kvh), 128, 0, V.st>>>(V.Xt, V.DG, V.Sq, V.X0, V.s); }
template <bool SUMS>
static void vfl(const int8_t* Vt, int8_t* Xt, int8_t* DG, int* Sq, int* X0, int kvh, int s, const Sig& sg, cudaStream_t st) {
  const size_t sm = (size_t)64 * (s / 128) + VF_SM_FIXED;
  if (SUMS && s > AP_Y1S) vfold<true, true><<<kvh * 32, VF_NT, sm, st>>>(Vt, Xt, DG, Sq, X0, s, sg);
  else vfold<SUMS, false><<<kvh * 32, VF_NT, sm, st>>>(Vt, Xt, DG, Sq, X0, s, sg);
}
static void vfd(AP& V, const Sig& s) { vfl<true>(V.Vt, V.Xt, V.DG, V.Sq, V.X0, V.kvh, V.s, s, V.st); }
static void vfx(AP& V, const Sig& s) { vfl<false>(V.Vt, V.Xt, V.DG, V.Sq, V.X0, V.kvh, V.s, s, V.st); }
static void producers(AP& V, const Sig& s) { ued(V, s); ped(V); vfd(V, s); }
static void producers_r2(AP& V, const Sig& s) { ued(V, s); ped(V); ved(V, s); vsm(V); }
static void aprun(AP& V, ApKern k, uint32_t* R) {
  const bool plain = k == apv_plain;
  ApArgs a{plain ? V.pmaps : k == apv_fu ? V.fmaps : V.maps, V.tab, V.ntiles, V.DG, V.Sq, plain ? V.V0 : V.X0, V.Uf,
           V.sp, V.sz, V.sv, V.F, V.O, R, V.s, V.nJ, V.seed, 0};
  k<<<V.grid, MLW::NT, CSMEM, V.st>>>(a);
}
// an arm: apv (apv_l7qs), apv_i, plain (apv_plain, eps_profile.py's baseline), cf and cc (the one-hot cheater, served free
// and served checked), fu (apv_fu), uedit, pedit, vedit, vsum, vfold, vfold_x, apvedited (uedit, pedit, vfold and apv under
// one key), fuedited (uedit, vfold and apv_fu: no pedit), or apvedited_r2 and fuedited_r2 (the same with vedit and vsum
// for vfold)
static std::function<void(AP&)> aparm(const std::string& a) {
  if (a == "apv") return [](AP& V) { aprun(V, apv_l7qs, V.R); };
  if (a == "apv_i") return [](AP& V) { aprun(V, apv_l7qs_i, V.R); };
  if (a == "fu") return [](AP& V) { aprun(V, apv_fu, V.R); };
  if (a == "fuedited") return [](AP& V) {
    const Sig sg = sigma(V.seed, V.inst++);
    ued(V, sg); vfd(V, sg);
    aprun(V, apv_fu, V.R);
  };
  if (a == "fuedited_r2") return [](AP& V) {
    const Sig sg = sigma(V.seed, V.inst++);
    ued(V, sg); ved(V, sg); vsm(V);
    aprun(V, apv_fu, V.R);
  };
  if (a == "vfold") return [](AP& V) { vfd(V, sigma(V.seed, V.inst++)); };
  if (a == "vfold_x") return [](AP& V) { vfx(V, sigma(V.seed, V.inst++)); };
  if (a == "apvedited_r2") return [](AP& V) {
    producers_r2(V, sigma(V.seed, V.inst++));
    aprun(V, apv_l7qs, V.R);
  };
  if (a == "cf") return [](AP& V) { aprun(V, apv_cf, V.R); };
  if (a == "cc") return [](AP& V) { aprun(V, apv_cc, V.R); };
  if (a == "plain" || a == "apv_plain") return [](AP& V) { aprun(V, apv_plain, V.R); };
  if (a == "uedit") return [](AP& V) { ued(V, sigma(V.seed, V.inst++)); };
  if (a == "pedit") return [](AP& V) { ped(V); };
  if (a == "vedit") return [](AP& V) { ved(V, sigma(V.seed, V.inst++)); };
  if (a == "vsum") return [](AP& V) { vsm(V); };
  if (a == "apvedited") return [](AP& V) {
    producers(V, sigma(V.seed, V.inst++));
    aprun(V, apv_l7qs, V.R);
  };
  fprintf(stderr, "no arm %s\n", a.c_str());
  exit(2);
}
static unsigned long long apbad(AP& V, const void* a, const void* b, size_t words) {
  CK(cudaMemsetAsync(V.bad, 0, 8, V.st));
  apcmp<<<1024, 256, 0, V.st>>>(reinterpret_cast<const uint32_t*>(a), reinterpret_cast<const uint32_t*>(b), words, V.bad);
  unsigned long long h;
  CK(cudaMemcpyAsync(&h, V.bad, 8, cudaMemcpyDeviceToHost, V.st));
  CK(cudaStreamSynchronize(V.st));
  return h;
}
static void dump2d(const std::string& dir, const std::string& name, const void* src, size_t pitch, size_t width, int rows) {
  std::vector<uint8_t> h(width * rows);
  CK(cudaMemcpy2D(h.data(), width, src, pitch, width, rows, cudaMemcpyDeviceToHost));
  dump(dir, name.c_str(), h.data(), h.size(), false);
}

static int apgate(const std::string& dir, int s, int H, int rep, uint32_t seed) {
  AP V;
  apsetup(V, s, H, rep, seed, true);
  const size_t nt = V.ntiles, rw = nt * MLW::NCT * 8;
  for (auto [p, b] : {std::pair<void*, size_t>{V.Pt, (size_t)H * s * s}, {V.Xt, (size_t)V.kvh * 128 * s}, {V.Um, (size_t)s * 32},
                      {V.PA, (size_t)s * 128}, {V.DG, (size_t)V.kvh * V.nJ * 128 * 128}, {V.Sq, (size_t)V.kvh * V.nJ * 128 * 4},
                      {V.X0, (size_t)V.kvh * 128 * 4}, {V.F, (size_t)H * s * 128 * 4}, {V.O, (size_t)H * s * 128 * 4},
                      {V.R, rw * 4}, {V.R2, rw * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, V.st));
  const Sig sg = sigma(seed, 0);
  producers(V, sg);
  aprun(V, apv_l7qs_i, V.R2);
  aprun(V, apv_l7qs, V.R);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
  // every tile: f's receipts (apv_l7qs) against O's (apv_l7qs_i), the same words; the no-write control
  const unsigned long long r_bad = apbad(V, V.R, V.R2, rw);
  uint32_t* pz;
  CK(cudaMalloc(&pz, rw * 4));
  CK(cudaMemsetAsync(pz, 0xA5, rw * 4, V.st));
  const unsigned long long c_bad = apbad(V, pz, V.R2, rw);
  CK(cudaFree(pz));
  // apv_fu (P~ formed in registers from P_prob) against apv_l7qs on every tile: f and receipts bitwise; F keeps apv_l7qs's
  const size_t fw = (size_t)H * s * 128;
  CK(cudaMemcpyAsync(V.Fr, V.F, fw * 4, cudaMemcpyDeviceToDevice, V.st));
  CK(cudaMemsetAsync(V.F, 0xA5, fw * 4, V.st));
  CK(cudaMemsetAsync(V.R3, 0xA5, rw * 4, V.st));
  aprun(V, apv_fu, V.R3);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
  const unsigned long long fu_r_bad = apbad(V, V.R3, V.R, rw), fu_f_bad = apbad(V, V.F, V.Fr, fw);
  CK(cudaMemcpyAsync(V.F, V.Fr, fw * 4, cudaMemcpyDeviceToDevice, V.st));
  // vfold's X, DG, Sq and X(0) (the producers' above) against r2's vedit and vsum from poisoned buffers under the same key,
  // and vfold_x's X; vfold's are put back for the dumps
  const size_t vb = (size_t)V.kvh * 128 * s, dgb = (size_t)V.kvh * V.nJ * 128 * 128, sqb = (size_t)V.kvh * V.nJ * 128 * 4,
               x0b = (size_t)V.kvh * 128 * 4;
  int8_t *cx, *cdg;
  int *csq, *cx0;
  CK(cudaMalloc(&cx, vb)); CK(cudaMalloc(&cdg, dgb)); CK(cudaMalloc(&csq, sqb)); CK(cudaMalloc(&cx0, x0b));
  for (auto [d, p, b] : {std::tuple<void*, void*, size_t>{cx, V.Xt, vb}, {cdg, V.DG, dgb}, {csq, V.Sq, sqb}, {cx0, V.X0, x0b}}) {
    CK(cudaMemcpyAsync(d, p, b, cudaMemcpyDeviceToDevice, V.st));
    CK(cudaMemsetAsync(p, 0xA5, b, V.st));
  }
  ved(V, sg); vsm(V);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
  const unsigned long long vf_x = apbad(V, V.Xt, cx, vb / 4), vf_dg = apbad(V, V.DG, cdg, dgb / 4), vf_sq = apbad(V, V.Sq, csq, sqb / 4),
                           vf_x0 = apbad(V, V.X0, cx0, x0b / 4);
  CK(cudaMemsetAsync(V.Xt, 0xA5, vb, V.st));
  vfx(V, sg);
  const unsigned long long vfx_x = apbad(V, V.Xt, cx, vb / 4);
  for (auto [d, p, b] : {std::tuple<void*, void*, size_t>{V.Xt, cx, vb}, {V.DG, cdg, dgb}, {V.Sq, csq, sqb}, {V.X0, cx0, x0b}})
    CK(cudaMemcpyAsync(d, p, b, cudaMemcpyDeviceToDevice, V.st));
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
  for (void* p : {(void*)cx, (void*)cdg, (void*)csq, (void*)cx0}) CK(cudaFree(p));
  const bool vf_ok = vf_x == 0 && vf_dg == 0 && vf_sq == 0 && vf_x0 == 0 && vfx_x == 0;
  const bool ok = r_bad == 0 && fu_r_bad == 0 && fu_f_bad == 0 && vf_ok, control_rejected = c_bad > 0;
  // the sampled tiles: all at most 24, else 24 spread over the table
  std::vector<int> pick;
  const int ns = std::min(V.ntiles, 24);
  for (int i = 0; i < ns; i++) pick.push_back(ns == V.ntiles ? i : (int)((long long)i * (V.ntiles - 1) / (ns - 1)));
  std::string tj;
  for (int t : pick) {
    const int4 e = V.htab[t];
    const int h = e.x, qt = e.y, kvh = e.z, nk = 128 * (qt + 1);
    const std::string sfx = "_" + std::to_string(t) + ".bin";
    const size_t prow = (size_t)h * s + 128 * qt, vrow = (size_t)kvh * 128;
    dump2d(dir, "P" + sfx, V.P + prow * s, s, nk, 128);
    dump2d(dir, "Pt" + sfx, V.Pt + prow * s, s, nk, 128);
    dump2d(dir, "V" + sfx, V.Vt + vrow * s, s, nk, 128);
    dump2d(dir, "X" + sfx, V.Xt + vrow * s, s, nk, 128);
    dump(dir, ("Um" + sfx).c_str(), V.Um + (size_t)128 * qt * 32, 128 * 32, true);
    dump(dir, ("PA" + sfx).c_str(), V.PA + (size_t)128 * qt * 128, 128 * 128, true);
    dump(dir, ("DG" + sfx).c_str(), V.DG + (size_t)(kvh * V.nJ + 4 * qt) * 128 * 128, 4 * 128 * 128, true);
    dump(dir, ("Sq" + sfx).c_str(), V.Sq + (size_t)(kvh * V.nJ + 4 * qt) * 128, 4 * 128 * 4, true);
    dump(dir, ("X0" + sfx).c_str(), V.X0 + vrow, 128 * 4, true);
    dump(dir, ("sp" + sfx).c_str(), V.sp + prow, 128 * 4, true);
    dump(dir, ("sz" + sfx).c_str(), V.sz + prow, 128 * 4, true);
    dump(dir, ("sv" + sfx).c_str(), V.sv + vrow, 128 * 4, true);
    dump(dir, ("O" + sfx).c_str(), V.O + prow * 128, 128 * 128 * 4, true);
    dump(dir, ("F" + sfx).c_str(), V.F + prow * 128, 128 * 128 * 4, true);
    dump(dir, ("R" + sfx).c_str(), V.R + (size_t)t * MLW::NCT * 8, MLW::NCT * 8 * 4, true);
    tj += std::string(tj.empty() ? "" : ", ") + "[" + std::to_string(t) + ", " + std::to_string(h) + ", " + std::to_string(qt) +
          ", " + std::to_string(kvh) + "]";
  }
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": 128, \"k\": %d, \"s\": %d, \"H\": %d, \"rep\": %d, \"ntiles\": %d, \"seed\": %u, "
              "\"key\": [\"%016llx\", \"%016llx\", \"%016llx\", \"%016llx\"], \"tiles\": [%s], \"kinds\": {\"apv_l7qs\": \"apv\"}, "
              "\"device\": {\"apv_l7qs\": {\"r_bad\": %llu}, \"apv_fu\": {\"r_bad\": %llu, \"f_bad\": %llu}, \"control\": {\"r_bad\": %llu}, "
              "\"vfold\": {\"x_bad\": %llu, \"dg_bad\": %llu, \"sq_bad\": %llu, \"x0_bad\": %llu}, \"vfold_x\": {\"x_bad\": %llu}}, "
              "\"device_ok\": %s, \"control_rejected\": %s}\n",
          H * s, s, s, H, rep, V.ntiles, seed, (unsigned long long)sg.w[0], (unsigned long long)sg.w[1], (unsigned long long)sg.w[2],
          (unsigned long long)sg.w[3], tj.c_str(), r_bad, fu_r_bad, fu_f_bad, c_bad, vf_x, vf_dg, vf_sq, vf_x0, vfx_x,
          ok ? "true" : "false", control_rejected ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"s\": %d, \"H\": %d, \"rep\": %d, \"ntiles\": %d, \"device_ok\": %s, \"control_rejected\": %s, "
         "\"r_bad\": %llu, \"fu_r_bad\": %llu, \"fu_f_bad\": %llu, \"vfold_bad\": [%llu, %llu, %llu, %llu], \"vfold_x_bad\": %llu}\n",
         dir.c_str(), s, H, rep, V.ntiles, ok ? "true" : "false", control_rejected ? "true" : "false", r_bad, fu_r_bad, fu_f_bad,
         vf_x, vf_dg, vf_sq, vf_x0, vfx_x);
  apfree(V);
  return ok && control_rejected ? 0 : 1;
}

static int apbench(int reps, int s, int H, int rep) {
  Nv nv;
  const std::vector<std::string> names = split(getenv("APV_ARMS"), "plain,apv,cf,cc,fu,uedit,pedit,vedit,vsum,apvedited,fuedited");
  AP V;
  apsetup(V, s, H, rep, 0x5EED0000u + s + H, false);
  producers(V, sigma(V.seed, 0xFFFFFFFFu));
  std::vector<std::function<void(AP&)>> arms;
  for (auto& a : names) arms.push_back(aparm(a));
  const int NA = (int)arms.size();
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_arm = [&](int j, int calls) {
    CK(cudaEventRecord(e0, V.st));
    for (int c = 0; c < calls; c++) arms[j](V);
    CK(cudaEventRecord(e1, V.st));
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
  long long stages = 0;
  for (auto& e : V.htab) stages += e.y + 1;
  printf("{\"m\": %d, \"n\": 128, \"k\": %d, \"s\": %d, \"H\": %d, \"rep\": %d, \"ntiles\": %d, \"stages\": %lld, \"grid\": %d, \"arms\": {",
         H * s, s, s, H, rep, V.ntiles, stages, V.grid);
  for (int j = 0; j < NA; j++) {
    printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", names[j].c_str(), median(ms[j]),
           median(mhz[j]), calls[j]);
    for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
    printf("]}");
  }
  printf("}}\n");
  fflush(stdout);
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  apfree(V);
  return 0;
}

static int apone(const std::string& arm, int s, int H, int rep) {
  AP V;
  apsetup(V, s, H, rep, 0x5EED0000u + s + H, false);
  producers(V, sigma(V.seed, 0xFFFFFFFFu));
  auto f = aparm(arm);
  for (int i = 0; i < 3; i++) f(V);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaProfilerStart());
  f(V);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"one\": \"%s\", \"s\": %d, \"H\": %d, \"rep\": %d, \"ntiles\": %d, \"grid\": %d}\n", arm.c_str(), s, H, rep, V.ntiles, V.grid);
  apfree(V);
  return 0;
}

#ifndef APV_LIB
// ---------------------------------------------------------------- the V producers alone
struct VP {
  int s = 0, kvh = 0, rows = 0;
  uint32_t seed = 0, inst = 0;
  size_t vb = 0, dgb = 0, sqb = 0, x0b = 0;
  cudaStream_t st;
  int8_t *Vt, *Xt, *DG, *Xr, *DGr;
  int *Sq, *X0, *Sqr, *X0r;
  unsigned long long* bad;
};
static void vpsetup(VP& P, int s, int kvh, uint32_t seed, bool gate) {
  if (s % 256 || s < 256 || kvh < 1) { fprintf(stderr, "s %d kvh %d: s %% 256 == 0, s >= 256, kvh >= 1\n", s, kvh); exit(2); }
  P.s = s; P.kvh = kvh; P.rows = kvh * 128; P.seed = seed;
  P.vb = (size_t)P.rows * s; P.dgb = (size_t)kvh * (s / 32) * 128 * 128; P.sqb = (size_t)kvh * (s / 32) * 128 * 4; P.x0b = (size_t)P.rows * 4;
  CK(cudaStreamCreateWithFlags(&P.st, cudaStreamNonBlocking));
  CK(cudaMalloc(&P.Vt, P.vb)); CK(cudaMalloc(&P.Xt, P.vb)); CK(cudaMalloc(&P.Xr, P.vb));
  CK(cudaMalloc(&P.DG, P.dgb)); CK(cudaMalloc(&P.DGr, P.dgb)); CK(cudaMalloc(&P.Sq, P.sqb)); CK(cudaMalloc(&P.Sqr, P.sqb));
  CK(cudaMalloc(&P.X0, P.x0b)); CK(cudaMalloc(&P.X0r, P.x0b)); CK(cudaMalloc(&P.bad, 8));
  sfill<I8><<<4096, 256, 0, P.st>>>(reinterpret_cast<uint8_t*>(P.Vt), P.vb, mix32(seed ^ 0x2222u), -128, 255);
  if (gate) { CK(cudaMemsetAsync(P.Vt, 0x7E, s, P.st)); CK(cudaMemsetAsync(P.Vt + s, 0x80, s, P.st)); }
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
}
static void vpfree(VP& P) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)P.Vt, (void*)P.Xt, (void*)P.Xr, (void*)P.DG, (void*)P.DGr, (void*)P.Sq, (void*)P.Sqr, (void*)P.X0, (void*)P.X0r,
                  (void*)P.bad})
    CK(cudaFree(p));
  CK(cudaStreamDestroy(P.st));
}
static void vp_ved(VP& P, const Sig& s) {
  vedit<<<dim3(P.s / 128, (P.rows + 4 * VE_G - 1) / (4 * VE_G)), 256, 0, P.st>>>(P.Vt, P.Xt, P.rows, P.s, s);
}
static void vp_vsm(VP& P) { vsum<<<dim3(32, P.kvh), 128, 0, P.st>>>(P.Xt, P.DG, P.Sq, P.X0, P.s); }
template <bool SUMS> static void vp_vf(VP& P, const Sig& s) { vfl<SUMS>(P.Vt, P.Xt, P.DG, P.Sq, P.X0, P.kvh, P.s, s, P.st); }
// an arm: vedit, vsum (r2's), vpair (vedit then vsum), vfold or vfold_x
static std::function<void(VP&)> vparm(const std::string& a) {
  if (a == "vedit") return [](VP& P) { vp_ved(P, sigma(P.seed, P.inst++)); };
  if (a == "vsum") return [](VP& P) { vp_vsm(P); };
  if (a == "vpair") return [](VP& P) { vp_ved(P, sigma(P.seed, P.inst++)); vp_vsm(P); };
  if (a == "vfold") return [](VP& P) { vp_vf<true>(P, sigma(P.seed, P.inst++)); };
  if (a == "vfold_x") return [](VP& P) { vp_vf<false>(P, sigma(P.seed, P.inst++)); };
  fprintf(stderr, "no producer arm %s\n", a.c_str());
  exit(2);
}
static unsigned long long vpcount(VP& P, const std::function<void()>& f) {
  CK(cudaMemsetAsync(P.bad, 0, 8, P.st));
  f();
  unsigned long long h;
  CK(cudaMemcpyAsync(&h, P.bad, 8, cudaMemcpyDeviceToHost, P.st));
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  return h;
}
static unsigned long long vpbad(VP& P, const void* a, const void* b, size_t words) {
  return vpcount(P, [&] {
    apcmp<<<1024, 256, 0, P.st>>>(reinterpret_cast<const uint32_t*>(a), reinterpret_cast<const uint32_t*>(b), words, P.bad);
  });
}
// poisoned buffers, one key: vfold's DG digits, X_diag and Sq against Ycum recomputed in full (vcheck); its X, X(0), Sq and
// DG against r2's vedit and vsum; vfold_x's X; a no-write control
static int vgate(int s, int kvh, uint32_t seed) {
  VP P;
  vpsetup(P, s, kvh, seed, true);
  const auto poison = [&] {
    for (auto [p, b] : {std::pair<void*, size_t>{P.Xt, P.vb}, {P.DG, P.dgb}, {P.Sq, P.sqb}, {P.X0, P.x0b}}) CK(cudaMemsetAsync(p, 0xA5, b, P.st));
  };
  const Sig sg = sigma(seed, 0);
  poison();
  vp_vf<true>(P, sg);
  for (auto [d, p, b] : {std::tuple<void*, void*, size_t>{P.Xr, P.Xt, P.vb}, {P.DGr, P.DG, P.dgb}, {P.Sqr, P.Sq, P.sqb}, {P.X0r, P.X0, P.x0b}})
    CK(cudaMemcpyAsync(d, p, b, cudaMemcpyDeviceToDevice, P.st));
  const unsigned long long y_bad =
      vpcount(P, [&] { vcheck<<<(P.rows * 32 + 255) / 256, 256, 0, P.st>>>(P.Xr, P.DGr, P.Sqr, P.rows, s, P.bad); });
  poison();
  vp_ved(P, sg);
  vp_vsm(P);
  const unsigned long long x_bad = vpbad(P, P.Xt, P.Xr, P.vb / 4), x0_bad = vpbad(P, P.X0, P.X0r, P.x0b / 4),
                           sq_bad = vpbad(P, P.Sq, P.Sqr, P.sqb / 4), dg_bad = vpbad(P, P.DG, P.DGr, P.dgb / 4);
  CK(cudaMemsetAsync(P.Xt, 0xA5, P.vb, P.st));
  vp_vf<false>(P, sg);
  const unsigned long long fx_bad = vpbad(P, P.Xt, P.Xr, P.vb / 4);
  CK(cudaMemsetAsync(P.Xt, 0xA5, P.vb, P.st));
  const unsigned long long c_bad = vpbad(P, P.Xt, P.Xr, P.vb / 4);
  const bool ok = y_bad == 0 && x_bad == 0 && x0_bad == 0 && sq_bad == 0 && dg_bad == 0 && fx_bad == 0, control_rejected = c_bad > 0;
  printf("{\"vgate\": true, \"s\": %d, \"kvh\": %d, \"digits\": %d, \"vfold\": {\"ycum_bad\": %llu, \"x_bad\": %llu, \"x0_bad\": %llu, "
         "\"sq_bad\": %llu, \"dg_bad\": %llu}, \"vfold_x\": {\"x_bad\": %llu}, \"control\": {\"x_bad\": %llu}, \"device_ok\": %s, "
         "\"control_rejected\": %s}\n",
         s, kvh, s > AP_Y1S ? 3 : 2, y_bad, x_bad, x0_bad, sq_bad, dg_bad, fx_bad, c_bad, ok ? "true" : "false",
         control_rejected ? "true" : "false");
  vpfree(P);
  return ok && control_rejected ? 0 : 1;
}
static int vbench(int reps, int s, int kvh) {
  Nv nv;
  const std::vector<std::string> names = split(getenv("VP_ARMS"), "vedit,vsum,vpair,vfold,vfold_x");
  VP P;
  vpsetup(P, s, kvh, 0x5EED0000u + s + kvh, false);
  vp_vf<true>(P, sigma(P.seed, 0xFFFFFFFFu));
  std::vector<std::function<void(VP&)>> arms;
  for (auto& a : names) arms.push_back(vparm(a));
  const int NA = (int)arms.size();
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_arm = [&](int j, int calls) {
    CK(cudaEventRecord(e0, P.st));
    for (int c = 0; c < calls; c++) arms[j](P);
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
  std::vector<std::vector<double>> ms(NA), mhz(NA);
  for (int r = 0; r < reps; r++)
    for (int j = 0; j < NA; j++) {
      const int a = (j + r) % NA;
      ms[a].push_back(time_arm(a, calls[a]));
      mhz[a].push_back(nv.sm());
    }
  CK(cudaGetLastError());
  printf("{\"vprod\": true, \"s\": %d, \"kvh\": %d, \"entries\": %zu, \"digits\": %d, \"arms\": {", s, kvh, P.vb, s > AP_Y1S ? 3 : 2);
  for (int j = 0; j < NA; j++) {
    printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", names[j].c_str(), median(ms[j]),
           median(mhz[j]), calls[j]);
    for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
    printf("]}");
  }
  printf("}}\n");
  fflush(stdout);
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  vpfree(P);
  return 0;
}
static int vone(const std::string& arm, int s, int kvh) {
  VP P;
  vpsetup(P, s, kvh, 0x5EED0000u + s + kvh, false);
  vp_vf<true>(P, sigma(P.seed, 0xFFFFFFFFu));
  auto f = vparm(arm);
  for (int i = 0; i < 3; i++) f(P);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaProfilerStart());
  f(P);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"vone\": \"%s\", \"s\": %d, \"kvh\": %d}\n", arm.c_str(), s, kvh);
  vpfree(P);
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 5 && !strcmp(argv[1], "vgate")) return vgate(atoi(argv[2]), atoi(argv[3]), (uint32_t)strtoul(argv[4], 0, 0));
  if (argc >= 5 && !strcmp(argv[1], "vbench")) return vbench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  if (argc >= 5 && !strcmp(argv[1], "vone")) return vone(argv[2], atoi(argv[3]), atoi(argv[4]));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return apgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 6 && !strcmp(argv[1], "bench")) return apbench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 6 && !strcmp(argv[1], "one")) return apone(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: apv gate DIR S H REP SEED | bench REPS S H REP | one ARM S H REP | vgate S KVH SEED | vbench REPS S KVH | "
                  "vone ARM S KVH\n");
  return 2;
}
#endif
