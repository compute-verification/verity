// Dither-C's P.V (`pv`, `checked_strip.pv_words` and `pv_served`) on
// sm_120a, unit by unit, on cgemm.cu's GEMM (e_l7qs's mainloop and duo receipts).
//
// Attention's P.V for H query heads sharing kvh = H / rep KV heads, s tokens, head dimension 128. P (P_prob - 64, in
// [-64, 62], H s x s, row h s + q) is the peeled side: P^ = P + c (c a fresh bit an entry) and P~ = M (P^ + U), U(q, rho)
// = -64 + sum_p w_p beta_p over seven fresh planes, fresh for every query and key block, M the causal mask. V (kvh 128 x
// s, channel-major) is the served side: X = V + b (b a fresh bit an entry). With p_block = 256 every (head, 32-query
// block, 256-key block B) is its own unit: its keys are B's up to the query block's diagonal 32-key block b_l (b_l = 7
// in a full unit, whose queries all follow B); Ycum, X's residue sums mod 32 over the unit's complete blocks before b_l,
// is split into m digits (m = 1 at b_l <= 1, else 2) with fresh carry coins z, shared by every query block of the KV
// head that reads them. Every cell commits, mod 2^32, in this order:
//   T_t = sum_{keys < 32 (t + 1)} P~ X (t <= b_l), O' = T_last - U A_0 - (U M) X_diag, D_1 = U A_1 (b_l >= 2),
// and the served O = O' - 128 D_1 = sum M P^ X, exact in s32; with S = sum_{keys <= q} X the served value is
// 2 (O + 64 S) - S = sum M (2 P_prob + 2 c - 1) X, written as f = float(2 O + 127 S) s_q s_ch (s_q carrying the 1/2).
// A CTA tile is 128 queries (four query blocks, one a warp row) x the 128 channels, over one unit's keys: a full unit
// (2 B + 2 <= qt, qt the 128-query tile) runs 256 keys on e_l7qs's interleaved absorbs; the diagonal unit (B = qt / 2)
// runs 128 (qt even) or 256 keys, warp row w absorbing atoms t <= b_l = 4 (qt & 1) + w right after each. The tail
// stage's rows P_A = [-U | U | -U M_diag | 127 M_diag] meet DG's [A_0 | A_1 | X_diag | 0] for the warp row's b_l (P_B,
// b_l = 7, in a full unit; each warp's own rows from global memory in a diagonal one).
// Noise from the per-call key (sha_pass.cuh's Sig; SHA-512(key || BE64(tag << 56 | j))): c under tag 8 in NS 1's patch
// layout over P (block (row / 4) (s / 128) + key / 128, entry (row % 4) 128 + key % 128, bit e % 8 of digest byte e / 8);
// b under tag 7 in the same layout over V; U under tag 12 (P_A's row i = (h nB + B) s + q: block i / 2, digest byte
// (i % 2) 32 + rho, beta_p = bit p, bit 7 unread); z under tag 13 (unit u = kvh nB + B: block (8 u + b_l) 8 + ch / 16, z(rho,
// ch) = bit (ch % 16) 32 + rho).
// Kernels:
//   pedit            P~ over each unit's keys, and its P_A rows
//   vedit            X = V + b
//   vdig             DG's rows for every (KV head, B, b_l) and S's sums Sb (b_l = 8: the whole block)
//   pv_l7qs          the unit GEMM with its words and receipts, f to F; pv_l7qs_i: O to O (for the CPU reference)
//   pv_plain         the same tiles with no edit: f = float(P X) s_q s_ch, the baseline
// Modes:
//   cpv gate DIR S H REP SEED       poisoned buffers, every unit: pv_l7qs_i's receipts against pv_l7qs's, a no-write
//                                   control; sampled tiles' operands, O, f and receipts to DIR for cpv_ref.py
//   cpv bench REPS S H REP          hot items interleaved across PV_ARMS, one JSON line
//   cpv one ARM S H REP             three warm-up calls, then one between cudaProfilerStart and Stop (for ncu)
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a cpv.cu -lcuda -lcublasLt -lnvidia-ml
// Run: cpvrun.sh (build, SASS gate, device gates, cpv_ref.py, then bench, ncu and eps_profile.py)
#define CGEMM_LIB
#include "cgemm.cu"

constexpr uint32_t PU_TAG = 12, PZ_TAG = 13;

// U from four digest bytes, bytewise (u_beta's -64 + (v & 63) + 65 bit6), and its negation
DEV uint32_t u4(uint32_t v) { return add_noise<false>((v & 0x3F3F3F3Fu) + 65u * ((v >> 6) & 0x01010101u), 0xC0C0C0C0u); }
DEV uint32_t neg4(uint32_t v) { return add_noise<false>(~v, 0x01010101u); }

struct PvArgs {
  const Maps* maps;   // a: P~ (H s x s), b: X^T (kvh 128 x s), pa: P_A (H nB s x 128), pb: DG (kvh nB 8 128 x 128)
  const int4* tab;    // a tile: h, qt, B, kvh << 1 | diagonal
  int ntiles;
  const int8_t* DG;
  const int* Sb;
  const float *sp, *sv;
  float* F;
  int* O;
  uint32_t* R;
  int s, nB;
  uint32_t seed, KR;
};

// e_l7qs's loop with its tiles from the table: the producer loads a unit's KT stages of P~ and X^T at its keys, then its
// tail stage (P_A's 128 query rows, DG's 128 channel rows for b_l = 7)
template <bool PLAIN> struct PVL : EL<PLAIN ? NONE : DUO, PLAIN ? 0 : 7, PLAIN ? 0 : 4, true> {
  using E = EL<PLAIN ? NONE : DUO, PLAIN ? 0 : 7, PLAIN ? 0 : 4, true>;
  using E::s; using E::pt; using E::pkt; using E::ps; using E::pph; using E::maps; using E::issuer;
  using E::stage; using E::full; using E::empty; using E::advance; using E::wait_full; using E::load; using E::load_a;
  using E::load_b;
  static constexpr int MI = E::MI, NI = E::NI, NKR = E::NKR;
  const int4* tab = nullptr;
  int ntiles = 0, s_len = 0, nB = 0;
  int jpt = -1, jA = 0, jB = 0, jK = 0, jKT = 0, jPA = 0, jPB = 0;
  DEV PVL(uint8_t* smem, const Maps* m) : E(smem, m, 128, 128, 256, 8, !PLAIN, 1) {}
  DEV void issue() {
    if (pt >= ntiles) return;
    if (pt != jpt) {
      const int4 e = __ldg(tab + pt);
      const int kvh = e.w >> 1;
      jA = e.x * s_len + 128 * e.y; jB = kvh * 128; jK = 256 * e.z; jKT = (e.w & 1) ? (e.y & 1) + 1 : 2;
      jPA = (e.x * nB + e.z) * s_len + 128 * e.y; jPB = ((kvh * nB + e.z) * 8 + 7) * 128;
      jpt = pt;
    }
    mbar_wait(empty(ps), pph ^ 1);
    mbar_expect_tx(full(ps), E::STAGE);
    if (pkt < jKT) {
      tma_load(stage(ps), &maps->a, full(ps), jK + pkt * E::BK, jA);
      tma_load(stage(ps) + E::A_BYTES, &maps->b, full(ps), jK + pkt * E::BK, jB);
    } else {
      tma_load(stage(ps), &maps->pa, full(ps), 0, jPA);
      tma_load(stage(ps) + E::A_BYTES, &maps->pb, full(ps), 0, jPB);
    }
    if (++pkt == jKT + (PLAIN ? 0 : 1)) { pkt = 0; pt += gridDim.x; }
    if (++ps == E::NST) { ps = 0; pph ^= 1; }
  }
  DEV void prologue() {
    if (issuer) {
      tma_acquire(&maps->a); tma_acquire(&maps->b); tma_acquire(&maps->pa); tma_acquire(&maps->pb);
      for (int i = 0; i < E::NST; i++) issue();
    }
    __syncwarp();
  }
  DEV void release(uint32_t done) {
    mbar_release(done);
    if (issuer) issue();
    __syncwarp();
  }
  // EL's stage_il and tile_il at K = 256
  template <bool FIRST, bool LAST>
  DEV void stage_pv(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[NKR], uint32_t (&fa)[2][MI][4],
                    uint32_t (&fb)[2][NI][2], uint32_t zero) {
    const uint32_t sb = stage(s), done = empty(s);
    load(fa[1], fb[1], sb, 1);
    E::template atom<FIRST, 0>(acc, ch, kr, fa[0], fb[0], zero);
    load(fa[0], fb[0], sb, 2);
    E::template atom<false, 1>(acc, ch, kr, fa[1], fb[1], zero);
    load(fa[1], fb[1], sb, 3);
    E::template atom<false, 2>(acc, ch, kr, fa[0], fb[0], zero);
    advance();
    if (!LAST) {
      wait_full();
      load(fa[0], fb[0], stage(s), 0);
    }
    E::template atom<false, 3>(acc, ch, kr, fa[1], fb[1], zero);
    release(done);
  }
  DEV void tile_full(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[NKR], uint32_t zero) {
    uint32_t fa[2][MI][4], fb[2][NI][2];
    wait_full();
    load(fa[0], fb[0], stage(s), 0);
    stage_pv<true, false>(acc, ch, kr, fa, fb, zero);
    stage_pv<false, true>(acc, ch, kr, fa, fb, zero);
    if constexpr (E::R) E::flush(ch, kr, acc);
  }
  // a diagonal unit's stage: atom t0 + pos's words absorbed right after its MMAs while t0 + pos <= b_l
  template <bool FIRST> DEV void dstage(int t0, int bl, int (&acc)[MI][NI][4], St (&ch)[8]) {
    wait_full();
    const uint32_t sb = stage(s), done = empty(s);
#pragma unroll
    for (int pos = 0; pos < 4; pos++) {
      uint32_t fa[MI][4], fb[NI][2];
      load(fa, fb, sb, pos);
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          if (FIRST && pos == 0) imma0(acc[mi][ni], fa[mi], fb[ni]);
          else imma(acc[mi][ni], fa[mi], fb[ni]);
        }
      if constexpr (E::R)
        if (t0 + pos <= bl) cabs_all<PVL>(ch, acc);
    }
    release(done);
    advance();
  }
  template <int KK> DEV void ldb(uint32_t (&fb)[NI][2], uint32_t tb, bool diag, const int8_t* dgw, int g, int q) const {
    if (!diag) {
      load_b(fb, tb, KK);
      return;
    }
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
      const int8_t* p = dgw + (size_t)(8 * ni + g) * 128 + 32 * KK + 4 * q;
      fb[ni][0] = __ldg(reinterpret_cast<const uint32_t*>(p));
      fb[ni][1] = __ldg(reinterpret_cast<const uint32_t*>(p + 16));
    }
  }
  template <int KK> DEV void tmma(int (&acc)[MI][NI][4], uint32_t (&fa)[MI][4], const uint32_t (&fb)[NI][2], uint32_t tb) const {
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      load_a(fa[mi], tb, KK, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) imma(acc[mi][ni], fa[mi], fb[ni]);
    }
  }
  // the tail stage: O' = T_last - U A_0 - (U M) X_diag (atoms 0 and 2), absorbed; D_1 = U A_1 (atom 1) where b_l >= 2,
  // absorbed, O = O' - 128 D_1; OUT 0: then 2 O + 127 S_diag (atom 3 against X_diag, a diagonal unit's) for f
  template <int OUT> DEV void ptail(int (&acc)[MI][NI][4], St (&ch)[8], bool diag, int bl, const int8_t* dgw, int g, int q) {
    wait_full();
    const uint32_t tb = stage(s);
    uint32_t fa[MI][4], fb[NI][2];
    ldb<0>(fb, tb, diag, dgw, g, q);
    tmma<0>(acc, fa, fb, tb);
    ldb<2>(fb, tb, diag, dgw, g, q);
    tmma<2>(acc, fa, fb, tb);
    cabs_all<PVL>(ch, acc);
    ldb<1>(fb, tb, diag, dgw, g, q);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) load_a(fa[mi], tb, 1, mi);
    if (bl >= 2) cdstep<PVL, 1, true>(acc, ch, fa, fb);
    if constexpr (OUT == 0) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
          for (int e = 0; e < 4; e++) acc[mi][ni][e] = (int)((uint32_t)acc[mi][ni][e] << 1);
      if (diag) {
        ldb<2>(fb, tb, diag, dgw, g, q);
        tmma<3>(acc, fa, fb, tb);
      }
    }
    release(empty(s));
    advance();
  }
};

// OUT 0: f to F; 1: O to O (s32). PLAIN: no edit, no words, f = float(acc) s_q s_ch
template <bool PLAIN, int OUT> DEV void pv_body(const PvArgs& a) {
  using L = PVL<PLAIN>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps);
  ml.tab = a.tab; ml.ntiles = a.ntiles; ml.s_len = a.s; ml.nB = a.nB;
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
    const int h = te.x, qt = te.y, B = te.z, kvh = te.w >> 1;
    const bool diag = te.w & 1;
    const int bl = diag ? 4 * (qt & 1) + wm : 7;
    St ch[8];
    if constexpr (!PLAIN) {
#pragma unroll
      for (int e = 0; e < 8; e++) {
        ch[e].y = mix32(a.seed ^ mix32(((uint32_t)t * L::NCT + ctid) * 8 + e));
        ch[e].z = mix32(ch[e].y + 0x9E3779B9u);
      }
    }
    int acc[MI][NI][4];
    if (!diag) {
      ml.tile_full(acc, ch, kr, zero);
    } else {
      ml.template dstage<true>(0, bl, acc, ch);
      if (qt & 1) ml.template dstage<false>(4, bl, acc, ch);
    }
    const int unit = kvh * a.nB + B;
    if constexpr (!PLAIN) ml.template ptail<OUT>(acc, ch, diag, bl, a.DG + ((size_t)(unit * 8 + bl) * 128 + wn * 32) * 128, g, q);
    const int r0 = wm * 32 + g, c0 = wn * 32 + 2 * q;
    if constexpr (OUT == 1) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          int* p = a.O + ((size_t)t * 128 + r0 + 16 * mi) * 128 + c0 + 8 * ni;
          *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
          *reinterpret_cast<int2*>(p + 8 * 128) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    } else {
      float rs[MI][2];
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) rs[mi][hh] = __ldg(a.sp + (size_t)h * a.s + 128 * qt + r0 + 16 * mi + 8 * hh);
      const int* Sr = a.Sb + (unit * 9 + (diag ? bl : 8)) * 128;
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        const float2 cs = __ldg(reinterpret_cast<const float2*>(a.sv + kvh * 128 + c0 + 8 * ni));
        int2 sv = make_int2(0, 0);
        if constexpr (!PLAIN) sv = __ldg(reinterpret_cast<const int2*>(Sr + c0 + 8 * ni));
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
          float f[4];
#pragma unroll
          for (int e = 0; e < 4; e++)
            f[e] = __int2float_rn(acc[mi][ni][e] + 127 * (e % 2 ? sv.y : sv.x)) * (rs[mi][e / 2] * (e % 2 ? cs.y : cs.x));
          float* p = a.F + ((size_t)t * 128 + r0 + 16 * mi) * 128 + c0 + 8 * ni;
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
using PvKern = void (*)(const PvArgs);
#define PVK(NAME, PLAIN, OUT) \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ PvArgs a) { pv_body<PLAIN, OUT>(a); }
PVK(pv_l7qs, false, 0)
PVK(pv_l7qs_i, false, 1)
PVK(pv_plain, true, 0)

// ---------------------------------------------------------------- producers
// a block a unit tile, a warp 32 of its query rows: lane l < 16 hashes c's patch for rows 4 (l / 2) .. + 3 and the unit's
// 128-key chunk l % 2, lane 16 + j U's block j (P_A rows 2j, 2j + 1); then 8 keys a lane a row, and P_A's row 4 bytes a
// lane
__global__ void __launch_bounds__(128) pedit(const int8_t* __restrict__ P, int8_t* __restrict__ Pt, int8_t* __restrict__ PA,
                                             const int4* __restrict__ tab, int s, int nB, const __grid_constant__ Sig sg) {
  __shared__ uint32_t cds[4][16][16], uds[4][16][16];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, t = blockIdx.x;
  const int4 te = __ldg(tab + t);
  const int h = te.x, qt = te.y, B = te.z;
  const bool diag = te.w & 1;
  const int KT = diag ? (qt & 1) + 1 : 2, q0 = 128 * qt + 32 * warp;
  const size_t prow0 = (size_t)h * s + q0, parow0 = ((size_t)h * nB + B) * s + q0;
  if (lane >= 16 || (lane & 1) < KT) {
    const uint64_t ctr = lane < 16 ? ((uint64_t)C_TAG << 56 | ((prow0 / 4 + lane / 2) * (uint64_t)(s / 128) + 2 * B + (lane & 1)))
                                   : ((uint64_t)PU_TAG << 56 | (parow0 / 2 + lane - 16));
    uint64_t st[8];
    sha_block(sg.w, ctr, st);
    uint32_t* d = lane < 16 ? cds[warp][lane] : uds[warp][lane - 16];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      d[2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      d[2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
  }
  __syncwarp();
  const uint8_t* ub = reinterpret_cast<const uint8_t*>(uds[warp]);
#pragma unroll 1
  for (int rr = 0; rr < 32; rr++) {
    const int qrel = q0 + rr - 256 * B;   // the query's place among the unit's keys (>= 256 in a full unit)
    const uint8_t* cb = reinterpret_cast<const uint8_t*>(cds[warp][2 * (rr / 4)]);
    const uint8_t* ur = ub + 32 * rr;
    if (lane < 16 * KT) {
      const int j = 8 * lane;
      const size_t off = (prow0 + rr) * s + 256 * B + j;
      const uint2 pw = __ldg(reinterpret_cast<const uint2*>(P + off));
      const uint32_t cbits = cb[64 * (j >> 7) + 16 * (rr & 3) + ((j & 127) >> 3)];
      const uint2 uw = *reinterpret_cast<const uint2*>(ur + (j & 31));
      uint32_t lo = add_noise<false>(add_noise<false>(pw.x, bits01(cbits & 0xFu)), u4(uw.x));
      uint32_t hi = add_noise<false>(add_noise<false>(pw.y, bits01(cbits >> 4)), u4(uw.y));
      const int keep = qrel - j + 1;
      if (keep < 8) {
        const uint64_t m = keep <= 0 ? 0ull : (~0ull >> (64 - 8 * keep));
        lo &= (uint32_t)m;
        hi &= (uint32_t)(m >> 32);
      }
      *reinterpret_cast<uint2*>(Pt + off) = make_uint2(lo, hi);
    }
    const int at = lane >> 3, rho = 4 * (lane & 7);
    const uint32_t U4 = u4(*reinterpret_cast<const uint32_t*>(ur + rho));
    const int keep = diag ? (qrel & 31) - rho + 1 : 4;
    const uint32_t mk = keep >= 4 ? ~0u : keep <= 0 ? 0u : (~0u >> (32 - 8 * keep));
    const uint32_t v = at == 0 ? neg4(U4) : at == 1 ? U4 : at == 2 ? (neg4(U4) & mk) : (0x7F7F7F7Fu & mk);
    reinterpret_cast<uint32_t*>(PA + (parow0 + rr) * 128)[lane] = v;
  }
}
// X^T = V^T + b, a warp a patch (4 rows x 128 keys): grid (s / 128, rows / 32)
__global__ void __launch_bounds__(256) vedit(const int8_t* __restrict__ Vt, int8_t* __restrict__ Xt, int rows, int s,
                                             const __grid_constant__ Sig sg) {
  __shared__ uint32_t dgs[8][16];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, chunk = blockIdx.x, grp = blockIdx.y * 8 + warp;
  if (grp * 4 >= rows) return;
  if (lane == 0) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)DITHER_TAG << 56 | ((uint64_t)grp * gridDim.x + chunk), st);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      dgs[warp][2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      dgs[warp][2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
  }
  __syncwarp();
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dgs[warp]);
#pragma unroll
  for (int rr = 0; rr < 4; rr++) {
    const size_t off = (size_t)(4 * grp + rr) * s + 128 * chunk + 4 * lane;
    const uint32_t bits = (bytes[16 * rr + lane / 2] >> (4 * (lane & 1))) & 0xFu;
    *reinterpret_cast<uint32_t*>(Xt + off) = add_noise<false>(__ldg(reinterpret_cast<const uint32_t*>(Vt + off)), bits01(bits));
  }
}
// DG's rows [A_0 | A_1 | X_diag | 0] of every b_l < 8 and Sb (S over the blocks before b_l; b_l = 8 the whole block) for
// one (B, KV head) a CTA, a channel a thread: Ycum grows by a block a step; threads 0 .. 63 hash z's 64 blocks first
__global__ void __launch_bounds__(128) vdig(const int8_t* __restrict__ Xt, int8_t* __restrict__ DG, int* __restrict__ Sb, int s,
                                            int nB, const __grid_constant__ Sig sg) {
  __shared__ uint32_t zd[64][16];
  const int B = blockIdx.x, kvh = blockIdx.y, ch = threadIdx.x, unit = kvh * nB + B;
  if (ch < 64) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)PZ_TAG << 56 | ((uint64_t)unit * 64 + ch), st);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      zd[ch][2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
      zd[ch][2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
    }
  }
  __syncthreads();
  const uint4* xr = reinterpret_cast<const uint4*>(Xt + (size_t)(kvh * 128 + ch) * s + 256 * B);
  int Y[32];
#pragma unroll
  for (int r = 0; r < 32; r++) Y[r] = 0;
  int S = 0;
#pragma unroll 1
  for (int bl = 0; bl < 8; bl++) {
    const uint4 x0 = __ldg(xr + 2 * bl), x1 = __ldg(xr + 2 * bl + 1);
    const uint32_t zw = zd[8 * bl + ch / 16][ch % 16];
    uint32_t a0w[8], a1w[8];
#pragma unroll
    for (int w4 = 0; w4 < 8; w4++) {
      a0w[w4] = 0; a1w[w4] = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int rho = 4 * w4 + i, R = Y[rho];
        int a0 = R, a1 = 0;
        if (bl >= 2) {
          const int q0 = R >> 7;
          int z = (int)((zw >> rho) & 1u);
          if (q0 == 127) z = 0;
          a0 = (R & 127) - 128 * z;
          a1 = q0 + z;
        }
        a0w[w4] |= (uint32_t)(a0 & 0xFF) << (8 * i);
        a1w[w4] |= (uint32_t)(a1 & 0xFF) << (8 * i);
      }
    }
    uint4* dst = reinterpret_cast<uint4*>(DG + ((size_t)(unit * 8 + bl) * 128 + ch) * 128);
    dst[0] = make_uint4(a0w[0], a0w[1], a0w[2], a0w[3]); dst[1] = make_uint4(a0w[4], a0w[5], a0w[6], a0w[7]);
    dst[2] = make_uint4(a1w[0], a1w[1], a1w[2], a1w[3]); dst[3] = make_uint4(a1w[4], a1w[5], a1w[6], a1w[7]);
    dst[4] = x0; dst[5] = x1;
    dst[6] = make_uint4(0, 0, 0, 0); dst[7] = make_uint4(0, 0, 0, 0);
    Sb[(unit * 9 + bl) * 128 + ch] = S;
    const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int r = 0; r < 32; r++) {
      const int v = (int)(int8_t)(xw[r / 4] >> (8 * (r % 4)));
      Y[r] += v;
      S += v;
    }
  }
  Sb[(unit * 9 + 8) * 128 + ch] = S;
}
// the scales: s_q (with P.V's 1/2) in [2^-11, 2^-10), s_ch in [2^-10, 2^-9)
__global__ void scfill(float* sp, int nq, float* sv, int nc, uint32_t seed) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nq + nc; i += gridDim.x * blockDim.x) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)i));
    if (i < nq) sp[i] = __uint_as_float(0x3A000000u | (h & 0x7FFFFFu));
    else sv[i - nq] = __uint_as_float(0x3A800000u | (h & 0x7FFFFFu));
  }
}
__global__ void pvcmp(const uint32_t* a, const uint32_t* b, size_t n, unsigned long long* bad) {
  unsigned long long c = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += a[i] != b[i];
  if (c) atomicAdd(bad, c);
}

// ---------------------------------------------------------------- host
struct PV {
  int s = 0, H = 0, rep = 1, kvh = 0, nB = 0, ntiles = 0, grid = 0;
  uint32_t seed = 0, inst = 0;
  cudaStream_t st;
  int8_t *P, *Vt, *Pt, *Xt, *PA, *DG;
  int *Sb, *O;
  float *sp, *sv, *F;
  uint32_t *R, *R2;
  int4* tab;
  unsigned long long* bad;
  std::vector<int4> htab;
  Maps* maps;
};
static void pvsetup(PV& V, int s, int H, int rep, uint32_t seed, bool gate) {
  if (s % 256 || H % rep || s < 256) {
    fprintf(stderr, "shape s %d H %d rep %d: s %% 256 == 0, H %% rep == 0\n", s, H, rep);
    exit(2);
  }
  V.s = s; V.H = H; V.rep = rep; V.kvh = H / rep; V.nB = s / 256; V.seed = seed;
  // units: 128-query tiles from the last, then key blocks, heads innermost (the rep heads of a KV head read its X^T and DG)
  for (int qt = s / 128 - 1; qt >= 0; qt--)
    for (int B = 0; B <= qt / 2; B++)
      for (int h = 0; h < H; h++) V.htab.push_back(make_int4(h, qt, B, (h / rep) << 1 | (B == qt / 2 ? 1 : 0)));
  V.ntiles = (int)V.htab.size();
  CK(cudaStreamCreateWithFlags(&V.st, cudaStreamNonBlocking));
  const size_t ps = (size_t)H * s * s, vs = (size_t)V.kvh * 128 * s, nt = (size_t)V.ntiles;
  CK(cudaMalloc(&V.P, ps)); CK(cudaMalloc(&V.Pt, ps)); CK(cudaMalloc(&V.Vt, vs)); CK(cudaMalloc(&V.Xt, vs));
  CK(cudaMalloc(&V.PA, (size_t)H * V.nB * s * 128)); CK(cudaMalloc(&V.DG, (size_t)V.kvh * V.nB * 8 * 128 * 128));
  CK(cudaMalloc(&V.Sb, (size_t)V.kvh * V.nB * 9 * 128 * 4));
  CK(cudaMalloc(&V.sp, (size_t)H * s * 4)); CK(cudaMalloc(&V.sv, (size_t)V.kvh * 128 * 4));
  CK(cudaMalloc(&V.F, nt * 128 * 128 * 4)); CK(cudaMalloc(&V.O, nt * 128 * 128 * 4));
  CK(cudaMalloc(&V.R, nt * MLW::NCT * 8 * 4)); CK(cudaMalloc(&V.R2, nt * MLW::NCT * 8 * 4));
  CK(cudaMalloc(&V.tab, nt * sizeof(int4))); CK(cudaMalloc(&V.bad, 8));
  CK(cudaMemcpy(V.tab, V.htab.data(), nt * sizeof(int4), cudaMemcpyHostToDevice));
  sfill<I8><<<4096, 256, 0, V.st>>>(reinterpret_cast<uint8_t*>(V.P), ps, mix32(seed ^ 0x1111u), -64, 127);
  sfill<I8><<<4096, 256, 0, V.st>>>(reinterpret_cast<uint8_t*>(V.Vt), vs, mix32(seed ^ 0x2222u), -127, 254);
  if (gate) {   // P's first two rows at 62 and -64, V's first two channels at 126 and -127
    CK(cudaMemsetAsync(V.P, 0x3E, s, V.st)); CK(cudaMemsetAsync(V.P + s, 0xC0, s, V.st));
    CK(cudaMemsetAsync(V.Vt, 0x7E, s, V.st)); CK(cudaMemsetAsync(V.Vt + s, 0x81, s, V.st));
  }
  scfill<<<256, 256, 0, V.st>>>(V.sp, H * s, V.sv, V.kvh * 128, mix32(seed ^ 0x3333u));
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  V.grid = std::min(V.ntiles, nsm);
  V.maps = upload_maps(Maps{tmap(V.Pt, s, H * s, 128), tmap(V.Xt, s, V.kvh * 128, 128), tmap(V.PA, 128, H * V.nB * s, 128),
                            tmap(V.DG, 128, V.kvh * V.nB * 8 * 128, 128)});
  for (PvKern k : {pv_l7qs, pv_l7qs_i, pv_plain}) CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
}
static void pvfree(PV& V) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)V.P, (void*)V.Pt, (void*)V.Vt, (void*)V.Xt, (void*)V.PA, (void*)V.DG, (void*)V.Sb, (void*)V.sp, (void*)V.sv,
                  (void*)V.F, (void*)V.O, (void*)V.R, (void*)V.R2, (void*)V.tab, (void*)V.bad, (void*)V.maps})
    CK(cudaFree(p));
  CK(cudaStreamDestroy(V.st));
}
static void ped(PV& V, const Sig& s) { pedit<<<V.ntiles, 128, 0, V.st>>>(V.P, V.Pt, V.PA, V.tab, V.s, V.nB, s); }
static void ved(PV& V, const Sig& s) { vedit<<<dim3(V.s / 128, V.kvh * 128 / 32), 256, 0, V.st>>>(V.Vt, V.Xt, V.kvh * 128, V.s, s); }
static void vdg(PV& V, const Sig& s) { vdig<<<dim3(V.nB, V.kvh), 128, 0, V.st>>>(V.Xt, V.DG, V.Sb, V.s, V.nB, s); }
static void pvrun(PV& V, PvKern k, uint32_t* R) {
  PvArgs a{V.maps, V.tab, V.ntiles, V.DG, V.Sb, V.sp, V.sv, V.F, V.O, R, V.s, V.nB, V.seed, 0};
  k<<<V.grid, MLW::NT, CSMEM, V.st>>>(a);
}
// an arm: pv (pv_l7qs), pv_i, pv_plain (or plain, eps_profile.py's baseline), pedit, vedit, vdig, or pvedited (the three
// producers and pv under one key)
static std::function<void(PV&)> pvarm(const std::string& a) {
  if (a == "pv") return [](PV& V) { pvrun(V, pv_l7qs, V.R); };
  if (a == "pv_i") return [](PV& V) { pvrun(V, pv_l7qs_i, V.R); };
  if (a == "pv_plain" || a == "plain") return [](PV& V) { pvrun(V, pv_plain, V.R); };
  if (a == "pedit") return [](PV& V) { ped(V, sigma(V.seed, V.inst++)); };
  if (a == "vedit") return [](PV& V) { ved(V, sigma(V.seed, V.inst++)); };
  if (a == "vdig") return [](PV& V) { vdg(V, sigma(V.seed, V.inst++)); };
  if (a == "pvedited") return [](PV& V) {
    const Sig s = sigma(V.seed, V.inst++);
    ped(V, s); ved(V, s); vdg(V, s); pvrun(V, pv_l7qs, V.R);
  };
  fprintf(stderr, "no arm %s\n", a.c_str());
  exit(2);
}
static unsigned long long pvbad(PV& V, const void* a, const void* b, size_t words) {
  CK(cudaMemsetAsync(V.bad, 0, 8, V.st));
  pvcmp<<<1024, 256, 0, V.st>>>(reinterpret_cast<const uint32_t*>(a), reinterpret_cast<const uint32_t*>(b), words, V.bad);
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

static int pvgate(const std::string& dir, int s, int H, int rep, uint32_t seed) {
  PV V;
  pvsetup(V, s, H, rep, seed, true);
  const size_t nt = V.ntiles, rw = nt * MLW::NCT * 8;
  for (auto [p, b] : {std::pair<void*, size_t>{V.Pt, (size_t)H * s * s}, {V.Xt, (size_t)V.kvh * 128 * s},
                      {V.PA, (size_t)H * V.nB * s * 128}, {V.DG, (size_t)V.kvh * V.nB * 8 * 128 * 128},
                      {V.Sb, (size_t)V.kvh * V.nB * 9 * 128 * 4}, {V.F, nt * 128 * 128 * 4}, {V.O, nt * 128 * 128 * 4},
                      {V.R, rw * 4}, {V.R2, rw * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, V.st));
  const Sig sg = sigma(seed, 0);
  ped(V, sg); ved(V, sg); vdg(V, sg);
  pvrun(V, pv_l7qs_i, V.R2);
  pvrun(V, pv_l7qs, V.R);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaGetLastError());
  // every unit: f's receipts (pv_l7qs) against O's (pv_l7qs_i), the same words; the no-write control
  const unsigned long long r_bad = pvbad(V, V.R, V.R2, rw);
  uint32_t* pz;
  CK(cudaMalloc(&pz, rw * 4));
  CK(cudaMemsetAsync(pz, 0xA5, rw * 4, V.st));
  const unsigned long long c_bad = pvbad(V, pz, V.R2, rw);
  CK(cudaFree(pz));
  const bool ok = r_bad == 0, control_rejected = c_bad > 0;
  // the sampled tiles: all at most 32, else 32 spread over the table
  std::vector<int> pick;
  const int ns = std::min(V.ntiles, 32);
  for (int i = 0; i < ns; i++) pick.push_back(ns == V.ntiles ? i : (int)((long long)i * (V.ntiles - 1) / (ns - 1)));
  std::string tj;
  for (int t : pick) {
    const int4 e = V.htab[t];
    const int h = e.x, qt = e.y, B = e.z, kvh = e.w >> 1, diag = e.w & 1, nk = diag ? 128 * ((qt & 1) + 1) : 256;
    const int unit = kvh * V.nB + B;
    const std::string sfx = "_" + std::to_string(t) + ".bin";
    const size_t prow = (size_t)h * s + 128 * qt, vrow = (size_t)kvh * 128;
    dump2d(dir, "P" + sfx, V.P + prow * s + 256 * B, s, nk, 128);
    dump2d(dir, "Pt" + sfx, V.Pt + prow * s + 256 * B, s, nk, 128);
    dump2d(dir, "V" + sfx, V.Vt + vrow * s + 256 * B, s, 256, 128);
    dump2d(dir, "X" + sfx, V.Xt + vrow * s + 256 * B, s, 256, 128);
    dump(dir, ("PA" + sfx).c_str(), V.PA + (((size_t)h * V.nB + B) * s + 128 * qt) * 128, 128 * 128, true);
    dump(dir, ("DG" + sfx).c_str(), V.DG + (size_t)unit * 8 * 128 * 128, 8 * 128 * 128, true);
    dump(dir, ("Sb" + sfx).c_str(), V.Sb + (size_t)unit * 9 * 128, 9 * 128 * 4, true);
    dump(dir, ("sp" + sfx).c_str(), V.sp + prow, 128 * 4, true);
    dump(dir, ("sv" + sfx).c_str(), V.sv + vrow, 128 * 4, true);
    dump(dir, ("O" + sfx).c_str(), V.O + (size_t)t * 128 * 128, 128 * 128 * 4, true);
    dump(dir, ("F" + sfx).c_str(), V.F + (size_t)t * 128 * 128, 128 * 128 * 4, true);
    dump(dir, ("R" + sfx).c_str(), V.R + (size_t)t * MLW::NCT * 8, MLW::NCT * 8 * 4, true);
    tj += std::string(tj.empty() ? "" : ", ") + "[" + std::to_string(t) + ", " + std::to_string(h) + ", " + std::to_string(qt) +
          ", " + std::to_string(B) + ", " + std::to_string(kvh) + ", " + std::to_string(diag) + "]";
  }
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": 128, \"k\": %d, \"s\": %d, \"H\": %d, \"rep\": %d, \"nB\": %d, \"ntiles\": %d, \"seed\": %u, "
              "\"key\": [\"%016llx\", \"%016llx\", \"%016llx\", \"%016llx\"], \"tiles\": [%s], \"kinds\": {\"pv_l7qs\": \"pv\"}, "
              "\"device\": {\"pv_l7qs\": {\"r_bad\": %llu}, \"control\": {\"r_bad\": %llu}}, \"device_ok\": %s, \"control_rejected\": %s}\n",
          H * s, s, s, H, rep, V.nB, V.ntiles, seed, (unsigned long long)sg.w[0], (unsigned long long)sg.w[1], (unsigned long long)sg.w[2],
          (unsigned long long)sg.w[3], tj.c_str(), r_bad, c_bad, ok ? "true" : "false", control_rejected ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"s\": %d, \"H\": %d, \"rep\": %d, \"ntiles\": %d, \"device_ok\": %s, \"control_rejected\": %s, "
         "\"r_bad\": %llu}\n", dir.c_str(), s, H, rep, V.ntiles, ok ? "true" : "false", control_rejected ? "true" : "false", r_bad);
  pvfree(V);
  return ok && control_rejected ? 0 : 1;
}

static int pvbench(int reps, int s, int H, int rep) {
  Nv nv;
  const std::vector<std::string> names = split(getenv("PV_ARMS"), "pv_plain,pv,pedit,vedit,vdig,pvedited");
  PV V;
  pvsetup(V, s, H, rep, 0x5EED0000u + s + H, false);
  {
    const Sig sg = sigma(V.seed, 0xFFFFFFFFu);
    ped(V, sg); ved(V, sg); vdg(V, sg);
  }
  std::vector<std::function<void(PV&)>> arms;
  for (auto& a : names) arms.push_back(pvarm(a));
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
  long long full = 0, dg = 0;
  for (auto& e : V.htab) (e.w & 1 ? dg : full)++;
  printf("{\"m\": %d, \"n\": 128, \"k\": %d, \"s\": %d, \"H\": %d, \"rep\": %d, \"ntiles\": %d, \"full\": %lld, \"diag\": %lld, "
         "\"grid\": %d, \"arms\": {", H * s, s, s, H, rep, V.ntiles, full, dg, V.grid);
  for (int j = 0; j < NA; j++) {
    printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", names[j].c_str(), median(ms[j]),
           median(mhz[j]), calls[j]);
    for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
    printf("]}");
  }
  printf("}}\n");
  fflush(stdout);
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  pvfree(V);
  return 0;
}

static int pvone(const std::string& arm, int s, int H, int rep) {
  PV V;
  pvsetup(V, s, H, rep, 0x5EED0000u + s + H, false);
  {
    const Sig sg = sigma(V.seed, 0xFFFFFFFFu);
    ped(V, sg); ved(V, sg); vdg(V, sg);
  }
  auto f = pvarm(arm);
  for (int i = 0; i < 3; i++) f(V);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaProfilerStart());
  f(V);
  CK(cudaStreamSynchronize(V.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"one\": \"%s\", \"s\": %d, \"H\": %d, \"rep\": %d, \"ntiles\": %d, \"grid\": %d}\n", arm.c_str(), s, H, rep, V.ntiles, V.grid);
  pvfree(V);
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return pvgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 6 && !strcmp(argv[1], "bench")) return pvbench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 6 && !strcmp(argv[1], "one")) return pvone(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: cpv gate DIR S H REP SEED | bench REPS S H REP | one ARM S H REP\n");
  return 2;
}
