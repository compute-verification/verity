// EGEMM's edited GEMM with its commitments hashed inside its own CTAs: the windowed receipts (segment.PouwSegCheck_v2{S
// = 16, FOLD}, E2E's w_ kernels on cursor/e2e-slice-ea24@3341b0f46, ported to egemm.cu's EL) and X's tile leaves (R = 64
// rows x 32 values) committed as hm96-sha512/row/v2 rows in frame-v3-sha512 trees, byte for byte verity.ml.commitments'
// (commit_gpu.cuh's constants and parts; hgref.py checks every level against the reference).
//
// The CTA: EGEMM's 16 MMA warps (lpgemm.cu's 128 x 128 TMA mainloop, the receipts' absorbs interleaved with the MMAs) and
// a hash warpgroup of 4 warps (640 threads, one CTA an SM, persistent). The launch's register pool is 640 x 96, so with
// setmaxnreg the MMA warps run at 104 registers and the hash warps at 64 (HREG 64; HREG 0: every warp at 96). An MMA warp
// stores its 32 receipt words of a window (one 2 KiB row is one window of one tile: R[(tile nw + w) 512 + thread]) and
// counts it in shared memory (release, CTA scope); a hash lane that holds the window's job waits for all 16 counts.
// Every hash lane is a small state machine doing one SHA-512 compression an iteration (one call site, so a warp's lanes
// compress together whatever job each holds):
//   leaf  20 compressions: the row's sha512/row/v2 digest x (16 data blocks and the padding block), h = SHA-512(leaf
//         prefix || (x ^ M_key y) || c) (2), the frame leaf's second block from its precomputed first-block state (1)
//   node  2 compressions after the level's midstate. The tree is built in dataflow: the lane that stores node (d, i)
//         counts its arrival at parent i / 2 (a counter a node, its parity flipping twice a use, so never reset); the
//         second arrival computes the parent, up to the root. Pads and pad-only subtrees are precomputed once.
// Jobs, per CTA in this order: the previous call's deferred windows (the last D windows this CTA's MMA warps stored in
// that call, whose leaves would otherwise be the kernel's tail: a leaf is 20 serial compressions), X's leaves
// [nX b / G, nX (b + 1) / G), then this call's windows in the order the MMA warps store them, but for its own last D.
// Before them, a salt phase: y = PRF(Ks, off + pos) (3 compressions), M_key y and c (2) for every leaf this CTA owns this
// call, off = (2 call + tree) 2^40, so no two leaves of any call or tree share a salt. Two buffer sets alternate (levels,
// node counters, salts, receipts); h_flush ends the last call's deferred windows and its trees.
// Kernels: w_d, w_x (unhashed, 512 threads: E2E's w_l3qm_s16 and XOR-in13 with S = 16, the device reference for the
// receipts), h_d, h_x (hashed, HREG 64), h_d96 (hashed, every warp at 96); _k3: the 3-atom peel (k > 4096). FOLD duo: duo
// l3qm (lag 3, the IMAD tie every fourth MMA); XOR-in13: y = (rotl(y, 13) ^ d0) a0_t + d1, y = (rotl(y, 19) ^ d2) a1_t +
// d3 on chain t, keys y from the windows' mix32 keys, a0/a1 segment.XA0/XA1 (lag 2, the LOP3 tie every eighth MMA).
// Modes:
//   hgemm gate DIR M N K SEED CALLS   CALLS calls (each its own W' = W + U), then the flush, on poisoned buffers: each
//                                     call's Y against cuBLASLt's X W^T, its hashed receipts against the unhashed kernel's
//                                     (every word), and X, W'_c, R_c and every level of both trees of every call to DIR
//                                     for hgref.py (params DIR/px.bin, DIR/pr.bin). env HG_GRID (CTAs), HG_D (deferral)
//   hgemm pipe REPS M N K PX PR ARM...  8-call sequences of each ARM, interleaved, per call (with the flush: total / 8;
//                                     steady: calls 1..7), medians. ARM: plain | edited:w_* | edited:h_* |
//                                     edited:h_*/nohash (the hash warps exit at once) | edited:w_*+commit (then
//                                     commit_gpu's Committer on X and the receipts, serial)
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a hgemm.cu -lcuda -lcublasLt -lnvidia-ml
#define main egemm_main
#include "../edited_gemm/egemm.cu"
#undef main
#define COMMIT_GPU_DOMS 2
#define smem cg_smem
#include "../commit_gpu/commit_gpu.cuh"
#undef smem

// segment.key_ok: a odd, and neither a nor -a mod 2^30 an int8 ratio (lam b = b' for int8 b != 0, b')
__host__ __device__ constexpr bool seg_key_ok(uint32_t a) {
  if (!(a & 1u)) return false;
  for (int s = 0; s < 2; s++) {
    const uint32_t lam = s ? 0u - a : a;
    for (int b = -128; b < 128; b++) {
      if (!b) continue;
      const uint32_t t = (lam * (uint32_t)b + (1u << 29)) & ((1u << 30) - 1u);
      if (t >= (1u << 29) - 128u && t < (1u << 29) + 128u) return false;
    }
  }
  return true;
}
struct XinK { uint32_t a[16]; };
__host__ __device__ constexpr XinK xin_seg_keys() {
  XinK k{};
  int n = 0;
  for (uint32_t i = 0; n < 16; i++) {
    const uint32_t a = mix32(0xC0000000u + i) | 1u;
    if (seg_key_ok(a)) k.a[n++] = a;
  }
  return k;
}
constexpr XinK XK = xin_seg_keys();
template <int T> constexpr uint32_t XA0 = XK.a[2 * T];
template <int T> constexpr uint32_t XA1 = XK.a[2 * T + 1];

constexpr int SW = 4;               // stages (BK = 128, four 32-step segments) a window: S = 16
constexpr int HT = 128;             // the hash warpgroup
constexpr int HNT = MLW::NT + HT;   // 640
constexpr int XR = 64;              // X's tile leaves: XR rows x 32 values
constexpr uint32_t ROW = 2048;      // bytes a row, both trees

struct HS { uint32_t cnt[16]; uint32_t next; };   // each MMA warp's stored windows (x 32 lanes); the job counter

DEV void sig_add(uint32_t addr) { asm volatile("red.release.cta.shared::cta.add.u32 [%0], 1;" ::"r"(addr) : "memory"); }
DEV uint32_t ld_acq(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  return v;
}

// FK: 0 duo, 1 XOR-in13
template <int FK, int LAG, int TIE> struct WL : EL<DUO, LAG, TIE> {
  using EB = EL<DUO, LAG, TIE>;
  using EB::EB;
  static constexpr int MI = EB::MI, NI = EB::NI, NCT = EB::NCT, U = MI * NI;
  static_assert(LAG >= 0 && LAG < U, "an absorb runs before its accumulator's next MMA");
  template <int J> DEV static void wab(St (&ch)[8], const int (&acc)[MI][NI][4]) {
    constexpr int mi = J / NI, ni = J % NI;
    if constexpr (FK == 0) {
      EB::template absorb<DUO, mi, ni>(ch, acc[mi][ni]);
    } else {
      uint32_t y = ch[J].y;
      y = (rotl(y, 13) ^ (uint32_t)acc[mi][ni][0]) * XA0<J> + (uint32_t)acc[mi][ni][1];
      y = (rotl(y, 19) ^ (uint32_t)acc[mi][ni][2]) * XA1<J> + (uint32_t)acc[mi][ni][3];
      ch[J].y = y;
    }
  }
  // the state unit TP's absorb wrote last (the tie's operand)
  template <int TP> DEV static uint32_t wtz(const St (&ch)[8]) {
    if constexpr (FK == 0) return ch[4 * (TP % 2) + 3].z;
    else return ch[TP].y;
  }
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
      if constexpr (FK == 0) ch[j].z = mix32(ch[j].y + 0x9E3779B9u);
    }
  }
  DEV static uint32_t fold(const St (&ch)[8]) {
    uint32_t r = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) r ^= FK == 0 ? finish<DUO>(ch, j) : ch[j].y;
    return r;
  }
  // window w's word to rp[w NCT], counted at sg (SIG)
  template <bool SIG> DEV static void store(uint32_t* rp, int w, uint32_t r, uint32_t sg) {
    rp[(size_t)w * NCT] = r;
    if constexpr (SIG) sig_add(sg);
  }
  // the end of window w: its lagged absorbs, R_w, the chains keyed for w + 1 (kb: window 0's key index), T_((w + 1) S - 1)
  template <bool SIG>
  DEV static void boundary(int (&acc)[MI][NI][4], St (&ch)[8], uint32_t* rp, int w, uint32_t seed, uint32_t kb, uint32_t sg) {
    ab_range<U - LAG, U>(acc, ch);
    store<SIG>(rp, w, fold(ch), sg);
    key(ch, seed, kb + 8u * (w + 1));
    ab_range<0, U>(acc, ch);
  }
  template <bool SIG>
  DEV void tile_w(int K, int (&acc)[MI][NI][4], St (&ch)[8], uint32_t zero, uint32_t* rp, uint32_t seed, uint32_t kb, uint32_t sg) {
    const int KT = K / EB::BK;
    uint32_t fa[2][MI][4], fb[2][NI][2];
    mbar_wait_warp(this->full(this->s), this->ph);
    this->load(fa[0], fb[0], this->stage(this->s), 0);
    wstage<true, false, false>(acc, ch, fa, fb, zero);
#pragma unroll 1
    for (int kt = 1; kt < KT - 1; kt++) {
      if (kt % SW == 0) {
        boundary<SIG>(acc, ch, rp, kt / SW - 1, seed, kb, sg);
        wstage<false, true, false>(acc, ch, fa, fb, zero);
      } else {
        wstage<false, false, false>(acc, ch, fa, fb, zero);
      }
    }
    if ((KT - 1) % SW == 0) {
      boundary<SIG>(acc, ch, rp, (KT - 1) / SW - 1, seed, kb, sg);
      wstage<false, true, true>(acc, ch, fa, fb, zero);
    } else {
      wstage<false, false, true>(acc, ch, fa, fb, zero);
    }
    ab_range<U - LAG, U>(acc, ch);
    store<SIG>(rp, KT / SW - 1, fold(ch), sg);
  }
};

// ---------------------------------------------------------------- the hash lanes
struct HTree {
  u64* levels;     // every level, leaves first (level_at)
  uint32_t* cnt;   // a parity counter an internal node: level d >= 1 at width - (width >> (d - 1))
  u64* S;          // the salt phase's M_key y || c, 16 words a leaf
  const u64* F;    // the frame leaves' first-block states, 8 words a leaf
  u64 prf;         // this commitment's salt PRF offset
  uint32_t n, width;
  int depth, dom;
};
struct HSet { HTree t[2]; const uint32_t* R; };   // t[0] X's tile tree, t[1] the receipts' (rows of R)
struct HA {
  HSet s[2];        // s[0] this call's, s[1] the previous call's
  int cur, prev;    // this call's leaves (its salts, X, its windows but the last D), the previous call's deferred ones
  const uint8_t* X;
  uint32_t K, mr;   // X is (64 mr) x K
  int nw, tiles, tiles_n, D;
  const uint8_t* T; // M_key's table
};

enum { H_IDLE = 0, H_WAIT = 1, H_LEAF = 2, H_NODE = 3, H_DONE = 4 };

DEV uint32_t tile_id_of(int t, int tiles_n) {   // EL::Place's raster (group 8)
  const int blk = t / (8 * tiles_n), r = t % (8 * tiles_n);
  return (uint32_t)((8 * blk + (r & 7)) * tiles_n + (r >> 3));
}
// salt_one's salt with the PRF read at prfpos, to S
DEV void h_salt(u64 prfpos, u64* S, const uint8_t* T) {
  u64 Y[24], c[8], my[8];
  u32 m[16];
  salt_prf(prfpos, Y);
  hm96_mask<false, true>(T, Y, m);
  salt_digest(Y, c);
  rc::mask_words(m, my);
  st8(S, my);
  st8(S + 8, c);
}
// a frame leaf's second block (frame_second_nb's, offsets from byte 128)
template <int NB> DEV void fs_build(int dom, u64 pos, const u64* h, u64* W) {
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = CD[dom].fl_tpl[NB - 1][16 + i];
  or_be<1 + NB, NB>(W, pos);
  or_digest<35 + 2 * NB>(W, h);
}
// block `half` of a node message (node_ni's blocks 2 and 3) over children in[0..7], in[8..15], read from L2
template <int NI> DEV void node_build(int dom, int half, u64 index, const u64* in, u64* W) {
  constexpr int S = 8 * ((10 + NI) % 8);   // the children's offset in a word (10 + NI and 82 + NI bytes)
  if (half == 0) {
#pragma unroll
    for (int i = 0; i < 16; ++i) W[i] = CD[dom].node_tpl[NI - 1][i];
    W[0] |= index << (8 * (6 - NI));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const u64 l = __ldcg(in + i);
      W[1 + i] |= l >> S;
      W[2 + i] |= l << (64 - S);
    }
#pragma unroll
    for (int i = 0; i < 6; ++i) {
      const u64 r = __ldcg(in + 8 + i);
      W[10 + i] |= r >> S;
      if (i < 5) W[11 + i] |= r << (64 - S);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) W[i] = CD[dom].node_tpl[NI - 1][16 + i];
    const u64 r5 = __ldcg(in + 13), r6 = __ldcg(in + 14), r7 = __ldcg(in + 15);
    W[0] |= r6 >> S | r5 << (64 - S);
    W[1] |= r7 >> S | r6 << (64 - S);
    W[2] |= r7 << (64 - S);
  }
}
// a stored node (lv, idx): false if it is the root or the first of its parent's two arrivals, else (lv, idx) the parent's
DEV bool climb(const HTree& tr, int& lv, uint32_t& idx) {
  if (lv == tr.depth) return false;
  const uint32_t p = idx >> 1;
  if (((2ull * p + 1) << lv) < tr.n) {   // the sibling holds leaves: two arrivals
    __threadfence();
    const uint32_t old = atomicAdd(tr.cnt + (tr.width - (tr.width >> lv)) + p, 1u);
    if (!(old & 1u)) return false;
    __threadfence();
  }
  ++lv;
  idx = p;
  return true;
}

// The hash lanes of CTA blockIdx.x (ht: 0 .. HT - 1). CUR: this call's salts and leaves too (h_flush: the deferred ones only)
template <bool CUR> DEV void hash_lanes(const HA& h, HS& hs, int ht) {
  const int b = blockIdx.x, G = gridDim.x;
  const int nt = (h.tiles - b + G - 1) / G, ns = nt * h.nw, nd = min(h.D, ns);
  const u64 nx = (u64)h.mr * (h.K / 32);
  const uint32_t x0 = (uint32_t)(nx * b / G), x1 = (uint32_t)(nx * (b + 1) / G);
  const uint32_t jp = h.prev ? nd : 0, jx = jp + (CUR && h.cur ? x1 - x0 : 0), jw = jx + (CUR && h.cur ? ns - nd : 0);
  auto win_pos = [&](int s) { return tile_id_of(b + G * (s / h.nw), h.tiles_n) * (uint32_t)h.nw + (uint32_t)(s % h.nw); };
  if constexpr (CUR) {
    if (h.cur) {
#pragma unroll 1
      for (uint32_t q = ht; q < (x1 - x0) + (uint32_t)ns; q += HT) {
        const bool isx = q < x1 - x0;
        const HTree& tr = h.s[0].t[isx ? 0 : 1];
        const uint32_t pos = isx ? x0 + q : win_pos((int)(q - (x1 - x0)));
        h_salt(tr.prf + pos, tr.S + 16ull * pos, h.T);
      }
    }
    asm volatile("bar.sync 1, %0;" ::"n"(HT) : "memory");
  }
  const uint32_t cnt0 = smem_u32(&hs.cnt[0]), next = smem_u32(&hs.next);
  int kind = H_IDLE, st = 0, lv = 0, step = 0;
  uint32_t idx = 0, seq = 0;
  u64 cv[8];
#pragma unroll 1
  for (;;) {
    if (kind == H_IDLE) {
      uint32_t q;
      asm volatile("atom.relaxed.cta.shared::cta.add.u32 %0, [%1], 1;" : "=r"(q) : "r"(next) : "memory");
      lv = 0;
      step = 0;
      if (q >= jw) {
        kind = H_DONE;
      } else if (q < jp) {
        st = 3;
        idx = win_pos(ns - nd + (int)q);
        kind = H_LEAF;
      } else if (q < jx) {
        st = 0;
        idx = x0 + (q - jp);
        kind = H_LEAF;
      } else {
        st = 1;
        seq = q - jx;
        idx = win_pos((int)seq);
        kind = H_WAIT;
      }
    }
    if (kind == H_WAIT) {
      bool ok = true;
#pragma unroll 1
      for (int w = 0; w < 16 && ok; w++) ok = ld_acq(cnt0 + 4 * w) >= 32u * (seq + 1);
      if (ok) kind = H_LEAF;
    }
    const bool work = kind == H_LEAF || kind == H_NODE;
    if (!__ballot_sync(~0u, work)) {
      if (__all_sync(~0u, kind == H_DONE)) break;
      __nanosleep(200);
      continue;
    }
    if (work) {
      const HTree& tr = h.s[st >> 1].t[st & 1];
      u64 W[16];
      if (kind == H_LEAF) {
        if (step < 16) {
          if (step == 0) cpy8(cv, C.mid_P);
          uint4 v[8];
          if (st & 1) {
            const uint4* p = reinterpret_cast<const uint4*>(h.s[st >> 1].R + (size_t)idx * (ROW / 4)) + 8 * step;
#pragma unroll
            for (int c = 0; c < 8; ++c) v[c] = p[c];
          } else {
            const uint32_t br = idx % h.mr, bc = idx / h.mr;
            const uint8_t* p = h.X + ((size_t)br * XR + 4 * step) * h.K + 32 * bc;
#pragma unroll
            for (int c = 0; c < 8; ++c) v[c] = __ldg(reinterpret_cast<const uint4*>(p + (size_t)(c >> 1) * h.K) + (c & 1));
          }
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            W[2 * c] = be64(v[c].x, v[c].y);
            W[2 * c + 1] = be64(v[c].z, v[c].w);
          }
        } else if (step == 16 || step == 18) {
#pragma unroll
          for (int i = 1; i < 15; ++i) W[i] = 0;
          W[0] = 0x8000000000000000ull;
          W[15] = step == 16 ? (u64)(128 + ROW) * 8 : 256 * 8;
        } else if (step == 17) {
          const u64* s = tr.S + 16ull * idx;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            W[i] = cv[i] ^ s[i];
            W[8 + i] = s[8 + i];
          }
          cpy8(cv, C.hm.mid_leaf);
        } else {
          switch (nbytes(idx)) {
            case 1: fs_build<1>(tr.dom, idx, cv, W); break;
            case 2: fs_build<2>(tr.dom, idx, cv, W); break;
            case 3: fs_build<3>(tr.dom, idx, cv, W); break;
            default: fs_build<4>(tr.dom, idx, cv, W); break;
          }
          ld8(tr.F + 8ull * idx, cv);
        }
      } else {
        const u64* in = level_at(tr.levels, tr.width, lv - 1) + 16ull * idx;
        switch (nbytes(idx)) {
          case 1: node_build<1>(tr.dom, step, idx, in, W); break;
          case 2: node_build<2>(tr.dom, step, idx, in, W); break;
          case 3: node_build<3>(tr.dom, step, idx, in, W); break;
          default: node_build<4>(tr.dom, step, idx, in, W); break;
        }
        if (step == 0) cpy8(cv, CD[tr.dom].node_mid[lv - 1]);
      }
      sha::compress(cv, W);
      if (++step == (kind == H_LEAF ? 20 : 2)) {
        st8(level_at(tr.levels, tr.width, lv) + 8ull * idx, cv);
        kind = climb(tr, lv, idx) ? H_NODE : H_IDLE;
        step = 0;
      }
    }
  }
}

// ---------------------------------------------------------------- the kernels
// HASH: the hash warpgroup beside the MMA warps (640 threads), else E2E's unhashed windowed kernel (512); HREG: the hash
// warps' registers (setmaxnreg; the MMA warps get the rest of the launch's 640 x 96), 0 for every warp at 96
template <int FK, int LAG, int TIE, bool THREE, bool HASH, int HREG> DEV void h_body(const Args& a, const HA& h) {
  using L = WL<FK, LAG, TIE>;
  constexpr int MI = L::MI, NI = L::NI, U = L::U;
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ HS hs;
  if constexpr (HASH) {
    if (threadIdx.x < 16) hs.cnt[threadIdx.x] = 0;
    if (threadIdx.x == 16) hs.next = 0;
  }
  L ml(smem, a.maps, a.M, a.N, a.K, a.group, true, 1);
  ml.init(threadIdx.x);
  if constexpr (HASH) {
    if (threadIdx.x >= L::NCT) {
      if constexpr (HREG > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(HREG));
      hash_lanes<true>(h, hs, threadIdx.x - L::NCT);
      return;
    }
    if constexpr (HREG > 0) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"((5 * 96 - HREG) / 4));
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int tiles_m = a.M / 128, tiles_n = a.N / 128, nw = a.K / (L::BK * SW);
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  const uint32_t zero = (uint32_t)a.KR, sg = HASH ? smem_u32(&hs.cnt[warp]) : 0u;
  typename L::Place pl{0, 0};
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    ml.place_step(pl, t == (int)blockIdx.x ? t : (int)gridDim.x);
    L::place_tile(pl, tiles_m, tm, tn);
    const uint32_t tile_id = tm * tiles_n + tn, kb = (tile_id * L::NCT + ctid) * (uint32_t)nw * 8u;
    St ch[8];
    L::key(ch, a.seed, kb);
    {
      const int z0[MI][NI][4] = {};
      L::template ab_range<0, U>(z0, ch);
    }
    int acc[MI][NI][4];
    ml.template tile_w<HASH>(a.K, acc, ch, zero, a.R + (size_t)tile_id * nw * L::NCT + ctid, a.seed, kb, sg);
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
typedef void (*HKern)(const Args, const HA);
#define WKERN(NAME, FK, LAG, TIE)                                                                                              \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ Args a, const __grid_constant__ HA h) {          \
    h_body<FK, LAG, TIE, false, false, 0>(a, h);                                                                             \
  }                                                                                                                          \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME##_k3(const __grid_constant__ Args a, const __grid_constant__ HA h) {     \
    h_body<FK, LAG, TIE, true, false, 0>(a, h);                                                                              \
  }
#define HKERN(NAME, FK, LAG, TIE, HREG)                                                                                        \
  __global__ void __launch_bounds__(HNT, 1) NAME(const __grid_constant__ Args a, const __grid_constant__ HA h) {              \
    h_body<FK, LAG, TIE, false, true, HREG>(a, h);                                                                           \
  }                                                                                                                          \
  __global__ void __launch_bounds__(HNT, 1) NAME##_k3(const __grid_constant__ Args a, const __grid_constant__ HA h) {         \
    h_body<FK, LAG, TIE, true, true, HREG>(a, h);                                                                            \
  }
WKERN(w_d, 0, 3, -4)
WKERN(w_x, 1, 2, 8)
HKERN(h_d, 0, 3, -4, 64)
HKERN(h_x, 1, 2, 8, 64)
HKERN(h_d96, 0, 3, -4, 0)

// the previous call's deferred windows and trees, after the last call (the same grid as the GEMM's)
__global__ void __launch_bounds__(HT, 1) h_flush(const __grid_constant__ HA h) {
  __shared__ HS hs;
  if (threadIdx.x == 0) hs.next = 0;
  __syncthreads();
  hash_lanes<false>(h, hs, threadIdx.x);
}
// once a tree: the frame leaves' first blocks
__global__ void h_frame_first(int dom, uint32_t n, u64* F) {
  const uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= n) return;
  u64 f[8];
  frame_first(dom, pos, f);
  st8(F + 8ull * pos, f);
}
// once a tree and buffer set: level d's pad-only nodes (the pads at d = 0)
__global__ void h_pads(int dom, u64* levels, uint32_t n, uint32_t width, int d) {
  const uint32_t first = (uint32_t)((((u64)n) + (1ull << d) - 1) >> d), i = first + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (width >> d)) return;
  u64 o[8];
  if (d == 0) {
    pad(dom, i, o);
  } else {
    u64 Lc[8], Rc[8];
    const u64* in = level_at(levels, width, d - 1);
    ld8(in + 16ull * i, Lc);
    ld8(in + 16ull * i + 8, Rc);
    node(dom, d - 1, i, Lc, Rc, o);
  }
  st8(level_at(levels, width, d) + 8ull * i, o);
}

struct HK { const char* name; HKern k, k3; bool hashed; };
static const HK HKS[] = {{"w_d", w_d, w_d_k3, false}, {"w_x", w_x, w_x_k3, false}, {"h_d", h_d, h_d_k3, true},
                         {"h_x", h_x, h_x_k3, true},  {"h_d96", h_d96, h_d96_k3, true}};
static const HK* hfind(const std::string& n) {
  for (auto& k : HKS) if (n == k.name) return &k;
  fprintf(stderr, "no kernel %s\n", n.c_str());
  exit(2);
}
// the unhashed twin with the same receipts
static const HK* twin(const HK* k) { return hfind(std::string(k->name).find("_x") != std::string::npos ? "w_x" : "w_d"); }

// ---------------------------------------------------------------- host
struct HC {
  Committer cx, cr;   // X's tile tree (domain slot 0) and the receipts' (slot 1): constants, T, and their own levels
                      // for the serial reference arm
  uint32_t n[2], width[2];
  int depth[2], D, nw, tiles, grid;
  u64 *lv[2][2], *S[2][2], *F[2];
  uint32_t *cnt[2][2], *R[2], *Rw;
};
static void hsetup(HC& H, EP& P, const std::string& px, const std::string& pr, int D, bool poison) {
  if (P.k % (128 * SW)) { fprintf(stderr, "k %d: not whole windows of %d segments\n", P.k, 4 * SW); exit(2); }
  const Params p0 = read_params(px), p1 = read_params(pr);
  H.nw = P.k / (128 * SW);
  H.tiles = (P.m / 128) * (P.n / 128);
  H.n[0] = (uint32_t)tile_leaves(P.m, P.k, XR);
  H.n[1] = (uint32_t)(H.tiles * H.nw);
  H.cx.init(p0, H.n[0], ROW, false, false, 0, 0);
  H.cr.init(p1, H.n[1], ROW, false, false, 0, 1);
  if (memcmp(&H.cx.hc, &H.cr.hc, sizeof(Consts))) { fprintf(stderr, "%s and %s differ beyond the domain\n", px.c_str(), pr.c_str()); exit(2); }
  if (!memcmp(&H.cx.hd, &H.cr.hd, sizeof(Dom))) { fprintf(stderr, "%s and %s share a domain\n", px.c_str(), pr.c_str()); exit(2); }
  H.D = D;
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  H.grid = getenv("HG_GRID") ? std::max(1, std::min(H.tiles, atoi(getenv("HG_GRID")))) : std::min(H.tiles, nsm);
  P.grid = H.grid;
  for (int t = 0; t < 2; t++) {
    H.depth[t] = depth_of(H.n[t]);
    H.width[t] = 1u << H.depth[t];
    CK(cudaMalloc(&H.F[t], (size_t)H.n[t] * 64));
    h_frame_first<<<(H.n[t] + 255) / 256, 256, 0, P.st>>>(t, H.n[t], H.F[t]);
    for (int s = 0; s < 2; s++) {
      CK(cudaMalloc(&H.lv[s][t], (size_t)2 * H.width[t] * 64));
      CK(cudaMalloc(&H.S[s][t], (size_t)H.n[t] * 128));
      CK(cudaMalloc(&H.cnt[s][t], (size_t)H.width[t] * 4));
      CK(cudaMemsetAsync(H.cnt[s][t], 0, (size_t)H.width[t] * 4, P.st));
      if (poison) {
        CK(cudaMemsetAsync(H.lv[s][t], 0xA5, (size_t)2 * H.width[t] * 64, P.st));
        CK(cudaMemsetAsync(H.S[s][t], 0xA5, (size_t)H.n[t] * 128, P.st));
      }
      for (int d = 0; d <= H.depth[t]; d++) {
        const uint32_t first = (uint32_t)(((u64)H.n[t] + (1ull << d) - 1) >> d), m = (H.width[t] >> d) - first;
        if (m) h_pads<<<(m + 127) / 128, 128, 0, P.st>>>(t, H.lv[s][t], H.n[t], H.width[t], d);
      }
    }
  }
  for (int s = 0; s < 2; s++) CK(cudaMalloc(&H.R[s], (size_t)H.n[1] * ROW));
  CK(cudaMalloc(&H.Rw, (size_t)H.n[1] * ROW));
  if (poison)
    for (uint32_t* r : {H.R[0], H.R[1], H.Rw}) CK(cudaMemsetAsync(r, 0xA5, (size_t)H.n[1] * ROW, P.st));
  for (auto& k : HKS)
    for (HKern f : {k.k, k.k3}) CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
}
// call c's arguments: this call's set c % 2, the previous call's (c - 1) % 2
static HA hargs(const HC& H, const EP& P, int c, bool cur, bool prev) {
  HA h{};
  for (int i = 0; i < 2; i++) {
    const int cc = std::max(0, c - i), s = cc % 2;
    for (int t = 0; t < 2; t++)
      h.s[i].t[t] = HTree{H.lv[s][t], H.cnt[s][t], H.S[s][t], H.F[t], (u64)(2 * cc + t) << 40, H.n[t], H.width[t], H.depth[t], t};
    h.s[i].R = H.R[s];
  }
  h.cur = cur;
  h.prev = prev && c > 0;
  h.X = reinterpret_cast<const uint8_t*>(P.X);
  h.K = P.k;
  h.mr = P.m / XR;
  h.nw = H.nw;
  h.tiles = H.tiles;
  h.tiles_n = P.n / 128;
  h.D = H.D;
  h.T = reinterpret_cast<const uint8_t*>(H.cx.T);
  return h;
}
static void hrun(EP& P, const HC& H, const HK* k, int c, uint32_t* R, bool hash = true) {
  Args a{P.mE, P.Y, R, P.m, P.n, P.k, 8, P.seed, 0, 1, 0, 0, {0, 0, 0, 0}, nullptr, 0};
  const HA h = hargs(H, P, c, hash, hash);
  (P.k > PEEL2_MAXK ? k->k3 : k->k)<<<H.grid, k->hashed ? HNT : MLW::NT, CSMEM, P.st>>>(a, h);
}
static void hflush(EP& P, const HC& H, int c) { h_flush<<<H.grid, HT, 0, P.st>>>(hargs(H, P, c, false, true)); }
// commit_gpu's Committer on call c's X and receipts (rows of R), serial: E2E's commitments
static void hcommit(EP& P, HC& H, const uint32_t* R) {
  H.cx.salts<true>(nullptr, P.st);
  H.cx.tiles(reinterpret_cast<const uint8_t*>(P.X), P.m, P.k, XR, false, P.st);
  H.cx.nodes(P.st);
  H.cr.salts<true>(nullptr, P.st);
  H.cr.leaves(RowBytes{reinterpret_cast<const uint8_t*>(R), ROW, ROW}, P.st);
  H.cr.nodes(P.st);
}
static void hfree(EP& P, HC& H) {
  CK(cudaDeviceSynchronize());
  for (int t = 0; t < 2; t++) {
    CK(cudaFree(H.F[t]));
    for (int s = 0; s < 2; s++) { CK(cudaFree(H.lv[s][t])); CK(cudaFree(H.S[s][t])); CK(cudaFree(H.cnt[s][t])); }
  }
  for (uint32_t* r : {H.R[0], H.R[1], H.Rw}) CK(cudaFree(r));
  H.cx.release();
  H.cr.release();
  efree(P);
}

static int hgate(const std::string& dir, int m, int n, int k, uint32_t seed, int calls) {
  EP P;
  esetup(P, m, n, k, seed, true);
  HC H;
  const int D = getenv("HG_D") ? atoi(getenv("HG_D")) : 16;
  hsetup(H, P, dir + "/px.bin", dir + "/pr.bin", D, true);
  const size_t xb = (size_t)m * k, wb = (size_t)n * k, rb = (size_t)H.n[1] * ROW;
  for (auto [p, b] : {std::pair<void*, size_t>{P.X, xb}, {P.Wp, wb}, {P.PA, (size_t)m * 128}, {P.PB, (size_t)n * 128},
                      {P.Y, P.ny * 4}, {P.Yr, P.ny * 4}})
    CK(cudaMemsetAsync(p, 0xA5, b, P.st));
  qd(P, sigma(seed, 0));
  ltg_run(P.plain, P.st);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  dump(dir, "X.bin", P.X, xb, true);
  bool ok = true;
  std::string res;
  auto levels = [&](int c) {
    for (int t = 0; t < 2; t++)
      dump(dir, ("lv_" + std::to_string(c) + (t ? "_R" : "_X") + ".bin").c_str(), H.lv[c % 2][t], (size_t)2 * H.width[t] * 64, true);
  };
  for (const char* kn : {"h_d", "h_x"}) {
    const HK* hk = hfind(kn);
    const HK* wk = twin(hk);
    for (int c = 0; c < calls; c++) {
      we(P, sigma(seed, 1 + c));
      CK(cudaMemsetAsync(P.Y, 0xA5, P.ny * 4, P.st));
      CK(cudaMemsetAsync(H.Rw, 0xA5, rb, P.st));
      hrun(P, H, wk, c, H.Rw, false);
      CK(cudaStreamSynchronize(P.st));
      CK(cudaGetLastError());
      const unsigned long long yw = ecmp(P, P.Y, P.Yr, P.ny);
      CK(cudaMemsetAsync(P.Y, 0xA5, P.ny * 4, P.st));
      CK(cudaMemsetAsync(H.R[c % 2], 0xA5, rb, P.st));
      hrun(P, H, hk, c, H.R[c % 2]);
      CK(cudaStreamSynchronize(P.st));
      CK(cudaGetLastError());
      const unsigned long long y = ecmp(P, P.Y, P.Yr, P.ny), r = ecmp(P, H.R[c % 2], H.Rw, rb / 4);
      if (c > 0) levels(c - 1);
      if (hk == HKS + 2 && c == 0) {   // call 0's X tree (PRF offset 0) by commit_gpu's own Committer, and both salt phases
        H.cx.salts<true>(nullptr, P.st);
        H.cx.tiles(reinterpret_cast<const uint8_t*>(P.X), P.m, P.k, XR, false, P.st);
        H.cx.nodes(P.st);
        CK(cudaStreamSynchronize(P.st));
        CK(cudaGetLastError());
        dump(dir, "cg_X.bin", H.cx.levels, (size_t)2 * H.width[0] * 64, true);
        dump(dir, "cgS_X.bin", H.cx.S, (size_t)H.n[0] * 128, true);
        dump(dir, "S_X.bin", H.S[0][0], (size_t)H.n[0] * 128, true);
      }
      const std::string tag = std::string(kn) + "_" + std::to_string(c);
      if (hk == HKS + 2) dump(dir, ("Wp_" + std::to_string(c) + ".bin").c_str(), P.Wp, wb, true);
      dump(dir, ("R_" + tag + ".bin").c_str(), H.R[c % 2], rb, true);
      res += (res.empty() ? "" : ", ") + std::string("\"") + tag + "\": {\"y_bad\": " + std::to_string(y) + ", \"y_bad_unhashed\": " +
             std::to_string(yw) + ", \"r_bad_vs_unhashed\": " + std::to_string(r) + "}";
      ok &= y == 0 && yw == 0 && r == 0;
    }
    hflush(P, H, calls);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    levels(calls - 1);
    for (int c = 0; c < calls; c++)
      for (int t = 0; t < 2; t++) {
        const std::string a = dir + "/lv_" + std::to_string(c) + (t ? "_R" : "_X") + ".bin";
        const std::string b = dir + "/lv_" + std::string(kn) + "_" + std::to_string(c) + (t ? "_R" : "_X") + ".bin";
        if (rename(a.c_str(), b.c_str())) { fprintf(stderr, "rename %s\n", a.c_str()); exit(3); }
      }
    // the next kernel's sequence starts over on the same sets: its first call (no previous) must not read them
    CK(cudaStreamSynchronize(P.st));
  }
  uint32_t inv_bits;
  memcpy(&inv_bits, &P.inv_s, 4);
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"calls\": %d, \"D\": %d, \"grid\": %d, \"nw\": %d, \"xr\": %d, "
              "\"kernels\": {\"h_d\": \"duo\", \"h_x\": \"xin\"}, \"inv_s_bits\": %u, \"device\": {%s}, \"device_ok\": %s}\n",
          m, n, k, seed, calls, D, H.grid, H.nw, XR, inv_bits, res.c_str(), ok ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d, \"D\": %d, \"device_ok\": %s, \"device\": {%s}}\n", dir.c_str(),
         m, n, k, H.grid, D, ok ? "true" : "false", res.c_str());
  hfree(P, H);
  return ok ? 0 : 1;
}

// ARM's 8-call sequence on P.st; the events after each call (ev[0] before the first) and after the flush (ev[9])
struct HArm { std::string name; std::function<void(EP&, int)> call; std::function<void(EP&, int)> flush; };
static HArm harm(HC& H, const std::string& a) {
  if (a == "plain") return {a, [](EP& P, int) { ltg_run(P.plain, P.st); }, nullptr};
  if (a.rfind("edited:", 0) != 0) { fprintf(stderr, "arm %s\n", a.c_str()); exit(2); }
  std::string kn = a.substr(7);
  const bool commit = kn.size() > 7 && kn.compare(kn.size() - 7, 7, "+commit") == 0;
  const bool nohash = kn.size() > 7 && kn.compare(kn.size() - 7, 7, "/nohash") == 0;
  if (commit || nohash) kn = kn.substr(0, kn.size() - 7);
  const HK* k = hfind(kn);
  HC* hp = &H;
  if (!k->hashed || nohash)
    return {a, [k, hp, commit](EP& P, int c) {
              we(P, sigma(P.seed, P.inst++));
              hrun(P, *hp, k, c, hp->R[c % 2], false);
              if (commit) hcommit(P, *hp, hp->R[c % 2]);
            }, nullptr};
  return {a, [k, hp](EP& P, int c) { we(P, sigma(P.seed, P.inst++)); hrun(P, *hp, k, c, hp->R[c % 2]); },
          [hp](EP& P, int c) { hflush(P, *hp, c); }};
}
static int hpipe(int reps, int m, int n, int k, const std::string& px, const std::string& pr, const std::vector<std::string>& arms) {
  Nv nv;
  EP P;
  esetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
  HC H;
  hsetup(H, P, px, pr, getenv("HG_D") ? atoi(getenv("HG_D")) : 16, false);
  constexpr int calls = 8;
  std::vector<HArm> A;
  for (auto& a : arms) A.push_back(harm(H, a));
  cudaEvent_t ev[calls + 2];
  for (auto& e : ev) CK(cudaEventCreate(&e));
  std::vector<std::vector<double>> tot(A.size()), steady(A.size()), last(A.size());
  std::vector<double> mz;
  for (int r = -2; r < reps; r++) {
    for (size_t i = 0; i < A.size(); i++) {
      CK(cudaEventRecord(ev[0], P.st));
      for (int c = 0; c < calls; c++) {
        A[i].call(P, c);
        CK(cudaEventRecord(ev[c + 1], P.st));
      }
      if (A[i].flush) A[i].flush(P, calls);
      CK(cudaEventRecord(ev[calls + 1], P.st));
      CK(cudaEventSynchronize(ev[calls + 1]));
      float t, s, f;
      CK(cudaEventElapsedTime(&t, ev[0], ev[calls + 1]));
      CK(cudaEventElapsedTime(&s, ev[1], ev[calls]));
      CK(cudaEventElapsedTime(&f, ev[calls], ev[calls + 1]));
      if (r < 0) continue;
      tot[i].push_back(t / calls);
      steady[i].push_back(s / (calls - 1));
      last[i].push_back(f);
    }
    if (r >= 0) mz.push_back(nv.sm());
  }
  CK(cudaGetLastError());
  double base = 0;
  for (size_t i = 0; i < A.size(); i++)
    if (A[i].name == "plain") base = median(tot[i]);
  for (size_t i = 0; i < A.size(); i++) {
    const double t = median(tot[i]), s = median(steady[i]);
    printf("{\"pipe\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"calls\": %d, \"reps\": %d, \"grid\": %d, \"D\": %d, "
           "\"ms\": %.6f, \"steady_ms\": %.6f, \"flush_ms\": %.6f, \"base_ms\": %.6f, \"ratio\": %.4f, \"steady_ratio\": %.4f, "
           "\"mhz\": %.0f, \"mhz_min\": %.0f}\n",
           A[i].name.c_str(), m, n, k, calls, reps, H.grid, H.D, t, s, median(last[i]), base, base > 0 ? t / base : 0.0,
           base > 0 ? s / base : 0.0, median(mz), *std::min_element(mz.begin(), mz.end()));
  }
  fflush(stdout);
  for (auto& e : ev) CK(cudaEventDestroy(e));
  hfree(P, H);
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 8 && !strcmp(argv[1], "gate"))
    return hgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0), atoi(argv[7]));
  if (argc >= 9 && !strcmp(argv[1], "pipe"))
    return hpipe(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6], argv[7],
                 std::vector<std::string>(argv + 8, argv + argc));
  fprintf(stderr, "usage: hgemm gate DIR M N K SEED CALLS | pipe REPS M N K PX PR ARM...\n");
  return 2;
}
