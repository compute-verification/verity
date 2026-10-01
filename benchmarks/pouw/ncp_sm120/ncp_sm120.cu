// NCP-FP8's checked chain on sm_120a, and what binding its checked words costs.
//
// The chain is D-3s's core (`fp8-is-h100`, #295) with the RTX PRO 6000's own atom: over the stacked depth K (3k, block-major),
// atom tau is one m16n8k32 E4M3 mma.sync from +0 over k in [32 tau, 32 tau + 32) (I_tau, verity.ml.tc's
// BLACKWELL_SM120_E4M3_M16N8K32), and the running word is R_0 = I_0, R_tau = add.rn.f32(R_(tau-1), I_tau).  D-3s checks
// I_0 ... I_(T-1) and R_1 ... R_(T-1), two words per output per atom.  The operands are formed codes (X~ and Y~, rows of K
// bytes); forming is not in this kernel.
//
// Modes (template MODE):
//   PLAIN  one accumulator over all of K (the plain FP8 GEMM, or the stacked MMA work alone at K = 3k); C = the last word
//   CORE   the D-3s chain; C = R_(T-1)
//   STORE  CORE, and every bound atom's words stored to a ring in global memory (the traffic an unfused binding pays)
//   HASH   CORE, and every bound atom's words absorbed into the lane's leaf from registers (the binding fused into the loop)
//   CHASH  PLAIN over K, and the running accumulator after every bound atom absorbed as HASH does (the plain chained GEMM
//          with checkpoints: the honest schedule when only every S-th word is bound, whose unbound FADDs a prover skips)
// An atom tau is bound when (tau + 1 + phase) % S == 0 (S = i[5]; S = 1 binds every atom; phase = 0, or the warp index mod S
// when i[7] staggers the bound atoms across warps).  At a bound atom the lane emits its R words (one per output it holds),
// then, when WORDS = 1, its I words; with WORDS = 1 the duplicate R_0 = I_0 is not emitted, so S = 1 emits exactly D-3s's
// checked words.  CHASH emits the accumulator's words.  A lane whose warp's rows are all padding (row >= i[6], decode's
// m = 32 in a 64-row tile) emits nothing: its digest is its key.
//
// The lane's leaf (HASH): BLAKE3 compressions chained over its emitted words in order, 16 words a block (an R or I block is
// the words of four consecutive fragments (mi, ni) in mi-major order, each fragment's four words e = 0 ... 3), the block
// counter its index in the lane's stream, length 64, flag KEYED_HASH, and the first chaining value the key
//   IV xor (tile, tid, S | words << 16, i[7], 0, 0, 0, 0),  tile = tm * tiles_n + tn,  tid the consumer thread (0 ... NCT - 1),
//   words = WORDS (HASH) or 2 (CHASH).
// The tile's leaf would be a tree over its lanes' digests, one compression per pair (`-h2`-style segments); this kernel
// writes the lane digests, which is where the work is.
//
// Args: p0 the tensor maps (ml::Maps: A, B; P_A, P_B unused), p1 C (float, M x N), p2 the lane digests (u32, tiles x NCT x
// 8), p3 the ring (float, grid x 2^CTA_RING_LOG words); i0 M (whole tiles), i1 N, i2 K, i3 raster group, i5 S, i6 the rows
// that exist (<= M), i7 stagger (0 or 1).
#include <cstdint>

#include "../pearl_c_sm120/mainloop_sm120.cuh"
#include "../pouw_hash/pouw_hash.cuh"

#define DEV __device__ __forceinline__

struct Args {
  uint64_t p[16];
  int i[16];
};

enum { M_PLAIN = 0, M_CORE = 1, M_STORE = 2, M_HASH = 3, M_CHASH = 4 };
constexpr int CTA_RING_LOG = 22;   // 16 MiB a CTA's region of the ring (4 GiB covers 256 CTAs)

DEV uint32_t f2u(float x) { return __float_as_uint(x); }

template <class L> struct Emit {
  static constexpr int MI = L::MI, NI = L::NI, P = MI * NI, BLOCKS = P / 4;
  static_assert(P % 4 == 0, "whole 16-word blocks");
  uint32_t cv[8];
  uint32_t ctr;
  uint32_t off;   // STORE: the CTA's next word in its region
  bool live;      // the warp holds a row that exists
  int phase;      // the bound atoms' offset

  DEV bool bound(int tau, int S) const { return live && (tau + phase) % S == 0; }
  DEV void init(int tile, int tid, int S, int words, int stagger) {
    const uint32_t IV[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = IV[i];
    cv[0] ^= (uint32_t)tile;
    cv[1] ^= (uint32_t)tid;
    cv[2] ^= (uint32_t)S | (uint32_t)words << 16;
    cv[3] ^= (uint32_t)stagger;
    ctr = 0;
  }
  // one block of 16 words: fragments 4b ... 4b + 3 of w
  DEV void absorb(const float (&w)[MI][NI][4], int b) {
    uint32_t m[16];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int p = 4 * b + j, mi = p / NI, ni = p % NI;
#pragma unroll
      for (int e = 0; e < 4; e++) m[4 * j + e] = f2u(w[mi][ni][e]);
    }
    uint32_t out[8];
    ph::blake3_compress(cv, m, ctr, 64, ph::B3_KEYED_HASH, out);
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = out[i];
    ctr++;
  }
  // the ring's address afresh at each emit (hoisted out of the K loop, the 4P addresses spill)
  template <int NCT> DEV void store(float* ring, const float (&w)[MI][NI][4]) {
    uint32_t tid, cta;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(cta));
    float* const region = ring + ((size_t)cta << CTA_RING_LOG);
    const uint32_t o = off + tid, mask = (1u << CTA_RING_LOG) - 1;
#pragma unroll
    for (int p = 0; p < P; p++)
#pragma unroll
      for (int e = 0; e < 4; e++) region[(o + (uint32_t)(4 * p + e) * NCT) & mask] = w[p / NI][p % NI][e];
    off += 4 * P * NCT;
  }
};

// One atom tau's work, after its fragments are in registers.  CORE family: first R_(tau-1) (a copy at tau = 1), then the
// emit of atom tau - 1's words if bound, then I_tau from +0 over every fragment.  PLAIN: the chained MMA (CHASH: after the
// emit of the accumulator if atom tau - 1 is bound).
template <int MODE, int WORDS, class L, bool T0, bool T1>
DEV void atom(int tau, int S, float (&acc)[L::MI][L::NI][4], float (&total)[L::MI][L::NI][4], const uint32_t (&fa)[L::MI][4],
              const uint32_t (&fb)[L::NI][2], Emit<L>& em, float* ring) {
  constexpr int MI = L::MI, NI = L::NI;
  if constexpr (MODE == M_PLAIN || MODE == M_CHASH) {
    if constexpr (MODE == M_CHASH) {
      if (!T0 && em.bound(tau, S)) {
#pragma unroll
        for (int b = 0; b < Emit<L>::BLOCKS; b++) em.absorb(acc, b);
      }
    }
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        if (T0) ml::qmma0(acc[mi][ni], fa[mi], fb[ni]);
        else ml::qmma(acc[mi][ni], fa[mi], fb[ni], acc[mi][ni]);
      }
    return;
  } else {
    if (!T0) {
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
          for (int e = 0; e < 4; e++) total[mi][ni][e] = T1 ? acc[mi][ni][e] : __fadd_rn(total[mi][ni][e], acc[mi][ni][e]);
      // a block boundary before I_tau's MMAs (never taken: CORE runs with S > T; a loop-invariant test is hoisted): without
      // it ptxas hoists them over the adds, holds both accumulators live and spills in the K loop (the emit's branch does
      // this in STORE and HASH)
      if constexpr (MODE == M_CORE) {
        if (tau == S) __trap();
      }
      if constexpr (MODE == M_STORE || MODE == M_HASH) {
        if (em.bound(tau, S)) {   // atom tau - 1 is bound
          if constexpr (MODE == M_HASH) {
            if constexpr (!(T1 && WORDS)) {
#pragma unroll
              for (int b = 0; b < Emit<L>::BLOCKS; b++) em.absorb(total, b);
            }
            if constexpr (WORDS) {
#pragma unroll
              for (int b = 0; b < Emit<L>::BLOCKS; b++) em.absorb(acc, b);
            }
          } else {
            if constexpr (!(T1 && WORDS)) em.template store<L::NCT>(ring, total);
            if constexpr (WORDS) em.template store<L::NCT>(ring, acc);
          }
        }
      }
    }
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) ml::qmma0(acc[mi][ni], fa[mi], fb[ni]);
  }
}

// One stage (128 bytes of depth, 4 atoms), fragments double-buffered by atom, the next stage's first atom loaded after this
// stage's last ldmatrix (the stage released there: its fragments are in registers).  FIRST: the tile's first stage.
template <int MODE, int WORDS, class L, bool FIRST>
DEV void stage(L& ml, int kt, int KT, int S, float (&acc)[L::MI][L::NI][4], float (&total)[L::MI][L::NI][4],
               uint32_t (&fa)[2][L::MI][4], uint32_t (&fb)[2][L::NI][2], Emit<L>& em, float* ring) {
  const uint32_t sb = ml.stage(ml.s), done = ml.empty(ml.s);
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
    if (kk < 3) {
      ml.load(fa[(kk + 1) & 1], fb[(kk + 1) & 1], sb, kk + 1);
    } else {
      ml::mbar_release(done);
      ml.advance();
      if (kt + 1 < KT) {
        ml::mbar_wait_warp(ml.full(ml.s), ml.ph);
        ml.load(fa[0], fb[0], ml.stage(ml.s), 0);
      }
    }
    const int tau = 4 * kt + kk;
    if (FIRST && kk == 0) atom<MODE, WORDS, L, true, false>(tau, S, acc, total, fa[0], fb[0], em, ring);
    else if (FIRST && kk == 1) atom<MODE, WORDS, L, false, true>(tau, S, acc, total, fa[1], fb[1], em, ring);
    else atom<MODE, WORDS, L, false, false>(tau, S, acc, total, fa[kk & 1], fb[kk & 1], em, ring);
  }
}

template <int MODE, int WORDS, class L>
DEV void body(const Args& a) {
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int M = a.i[0], N = a.i[1], K = a.i[2], group = a.i[3], S = a.i[5], rows = a.i[6], stagger = a.i[7];
  const int tiles_m = M / L::TM_, tiles_n = N / L::BN_, KT = K / 128, T = K / 32;
  L ml(smem);
  ml.roles(reinterpret_cast<const ml::Maps*>(a.p[0]), M, N, K, group);
  float* ring = reinterpret_cast<float*>(a.p[3]);
  Emit<L> em;
  em.off = 0;
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    ml::raster(t, tiles_m, tiles_n, group, tm, tn);
    float acc[MI][NI][4], total[MI][NI][4];
    uint32_t tid;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
    em.init(tm * tiles_n + tn, tid, S, MODE == M_CHASH ? 2 : WORDS, stagger);
    em.live = tm * L::TM_ + (int)(tid / 32 / L::WARPS_N) * L::WM < rows;
    em.phase = stagger ? (int)(tid / 32) % S : 0;
    uint32_t fa[2][MI][4], fb[2][NI][2];
    ml::mbar_wait_warp(ml.full(ml.s), ml.ph);
    ml.load(fa[0], fb[0], ml.stage(ml.s), 0);
    stage<MODE, WORDS, L, true>(ml, 0, KT, S, acc, total, fa, fb, em, ring);
#pragma unroll 1
    for (int kt = 1; kt < KT; kt++) stage<MODE, WORDS, L, false>(ml, kt, KT, S, acc, total, fa, fb, em, ring);
    if constexpr (MODE != M_PLAIN && MODE != M_CHASH) {
      // R_(T-1), and atom T - 1's words
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
          for (int e = 0; e < 4; e++) total[mi][ni][e] = T == 1 ? acc[mi][ni][e] : __fadd_rn(total[mi][ni][e], acc[mi][ni][e]);
      if constexpr (MODE == M_STORE || MODE == M_HASH) {
        if (em.bound(T, S)) {
          if constexpr (MODE == M_HASH) {
#pragma unroll
            for (int b = 0; b < Emit<L>::BLOCKS; b++) em.absorb(total, b);
            if constexpr (WORDS) {
#pragma unroll
              for (int b = 0; b < Emit<L>::BLOCKS; b++) em.absorb(acc, b);
            }
          } else {
            em.template store<L::NCT>(ring, total);
            if constexpr (WORDS) em.template store<L::NCT>(ring, acc);
          }
        }
      }
    } else {
      if constexpr (MODE == M_CHASH) {
        if (em.bound(T, S)) {
#pragma unroll
          for (int b = 0; b < Emit<L>::BLOCKS; b++) em.absorb(acc, b);
        }
      }
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
          for (int e = 0; e < 4; e++) total[mi][ni][e] = acc[mi][ni][e];
    }
    if (K < 0) __trap();   // a block boundary (never taken) between the chain and the epilogue, as in pearl_c_sm120.cu
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));   // afresh (held through the chain, the indices spill)
    const int warp = tid / 32, lane = tid % 32, wm = warp / L::WARPS_N, wn = warp % L::WARPS_N;
    float* C = reinterpret_cast<float*>(a.p[1]);
    const int g = lane >> 2, q = lane & 3;
    const int row0 = tm * L::TM_ + wm * L::WM + g, col0 = tn * L::BN_ + wn * L::WN + 2 * q;
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        const int row = row0 + 16 * mi, col = col0 + 8 * ni;
        if (row < M) *reinterpret_cast<float2*>(C + (size_t)row * N + col) = make_float2(total[mi][ni][0], total[mi][ni][1]);
        if (row + 8 < M)
          *reinterpret_cast<float2*>(C + (size_t)(row + 8) * N + col) = make_float2(total[mi][ni][2], total[mi][ni][3]);
      }
    if constexpr (MODE == M_HASH || MODE == M_CHASH) {
      uint4* dg = reinterpret_cast<uint4*>(a.p[2]) + ((size_t)(tm * tiles_n + tn) * L::NCT + tid) * 2;
      dg[0] = make_uint4(em.cv[0], em.cv[1], em.cv[2], em.cv[3]);
      dg[1] = make_uint4(em.cv[4], em.cv[5], em.cv[6], em.cv[7]);
    }
  }
}

// The mainloop with its tile shape exposed.  P: 128 x 128, 8 consumer warps of 64 x 32 (prefill); D: 64 x 64, 8 warps of 16 x 32
// (decode: m = 32 padded to one 64-row tile, as Pearl-C's decode is)
template <int TM, int BN, int STAGES, int WMR, int RP>
struct Shape : ml::Mainloop<TM, BN, 0, STAGES, false, 1, RP, WMR> {
  static constexpr int TM_ = TM, BN_ = BN;
  using ml::Mainloop<TM, BN, 0, STAGES, false, 1, RP, WMR>::Mainloop;
};
using LP = Shape<128, 128, 3, 64, 24>;
using LD = Shape<64, 64, 6, 16, 24>;

#define NCP_KERNEL(NAME, MODE, WORDS, L)                                                                         \
  extern "C" __global__ void __launch_bounds__(L::NT, 1) NAME(const Args a) { body<MODE, WORDS, L>(a); } \
  extern "C" __global__ void NAME##_cfg(int* out) {                                                        \
    out[0] = L::SMEM;                                                                                       \
    out[1] = L::NT;                                                                                         \
    out[2] = L::NCT;                                                                                        \
    out[3] = L::TM_;                                                                                        \
    out[4] = L::BN_;                                                                                        \
  }

NCP_KERNEL(p_plain, M_PLAIN, 0, LP)
NCP_KERNEL(p_core, M_CORE, 0, LP)
NCP_KERNEL(p_store_r, M_STORE, 0, LP)
NCP_KERNEL(p_store_ir, M_STORE, 1, LP)
NCP_KERNEL(p_hash_r, M_HASH, 0, LP)
NCP_KERNEL(p_hash_ir, M_HASH, 1, LP)
NCP_KERNEL(p_chash, M_CHASH, 0, LP)
NCP_KERNEL(d_plain, M_PLAIN, 0, LD)
NCP_KERNEL(d_core, M_CORE, 0, LD)
NCP_KERNEL(d_store_r, M_STORE, 0, LD)
NCP_KERNEL(d_store_ir, M_STORE, 1, LD)
NCP_KERNEL(d_hash_r, M_HASH, 0, LD)
NCP_KERNEL(d_hash_ir, M_HASH, 1, LD)
NCP_KERNEL(d_chash, M_CHASH, 0, LD)

// The operands: row-major bytes, code(idx) = sign | (4 + (h >> 8 & 7)) << 3 | (h >> 16 & 7), h = mix(idx * 0x9E3779B9 + seed):
// finite E4M3 codes of magnitude 2^-3 ... 2^4 x (1 + f/8), no NaN.  i0 count, i1 seed; p0 out.
DEV uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
extern "C" __global__ void gen_codes(const Args a) {
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[0]);
  const uint64_t n = (uint64_t)(uint32_t)a.i[0] | (uint64_t)(uint32_t)a.i[2] << 32;
  const uint32_t seed = (uint32_t)a.i[1];
  for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (uint64_t)gridDim.x * blockDim.x) {
    const uint32_t h = mix32((uint32_t)idx * 0x9E3779B9u + seed);
    out[idx] = (uint8_t)((h & 0x80) | ((4 + ((h >> 8) & 7)) << 3) | ((h >> 16) & 7));
  }
}
