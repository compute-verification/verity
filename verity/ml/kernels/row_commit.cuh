// The row commitment on the device and the host: frame-v3-sha512's leaf, pad and node (`merkle.CommitmentDomain`'s through
// SHA-512) and the hm96-sha512 leaf (`hm96.SHA512`), the one copy the CUDA committers include. A committer keeps only its
// schedule: which rows, in which order and streams, how it digests a row, where these constants live and where its salts
// come from.
//
// A frame message is fixed by its domain id up to a few bytes, so its blocks are built once (the builders at the end, on
// the host or the device) and each hash ORs in what varies:
//   leaf(pos)       a 64-byte value's leaf at rank = position = pos: two compressions, the first block's rounds 0..14 ahead
//                   (FrameK)
//   pad(rank)       two compressions (FrameK)
//   node(level, index, L, R)
//                   two compressions past the chaining value after its first block, which fixes only the level (NodeK)
// Positions, ranks and indices are below 2^32 (MAX_NB bytes), and uint's length is picked at compile time (`_nb`, a body a
// length) or read at run time (`_rt`, one body for every length).
//
// The hm96-sha512 leaf of a row digest x under the salt y: c = SHA-512(salt_prefix || y), b = x ^ M_key y and the value
// h = SHA-512(leaf_prefix(key) || b || c) the frame leaf binds (Hm96K). M_key y comes from a table of T_v, the XOR of
// (key >> s) over the set bits s of v: the nibble table (T_v = T_(v & 15) ^ T_(v & 0xF0), TN_BYTES) or the byte table (a
// row of 256 bytes a byte value), read through L1 or from shared memory.
//
// Every digest is 8 big-endian words, as SHA-512's state holds it. The compression is a policy C, by default `Sha`
// (sha512.cuh's `sha::` forms; `Compact` is its one-body form), so a kernel passes its own form of the same function.
#pragma once
#include <cstdint>
#include <cstring>

#include "sha512.cuh"

#define RC_HD __host__ __device__ __forceinline__
#ifdef __CUDA_ARCH__
#define RC_UNROLL _Pragma("unroll")
#define RC_UNROLL1 _Pragma("unroll 1")
#else
#define RC_UNROLL
#define RC_UNROLL1
#endif

namespace rc {

using u64 = uint64_t;
using u32 = uint32_t;

constexpr int MAX_NB = 4;         // positions, ranks and node indices below 2^32
constexpr int LEVELS = 33;
constexpr int MAX_SCHEMA = 22;    // the longest schema whose leaf is two blocks at every nb
constexpr int TN_BYTES = 8192;    // the nibble table

#ifdef __CUDACC__
using q4 = uint4;
#else
struct q4 {
  u32 x, y, z, w;
};
#endif

// What a domain id and a schema fix of the frame leaf and the pad.
struct FrameK {
  u64 fl_tpl[MAX_NB][32];    // the leaf message's two blocks per nb = len(uint(pos)) (index nb - 1), pos and the value zero
  u64 fl_m15[8];             // the state after rounds 0..14 of its first block, whose words 0..14 don't depend on nb
  u64 pad_tpl[MAX_NB][16];   // a pad message's first block per nb, rank zero
  int sx;                    // the schema's length - 18
};
// What a domain id fixes of the node.
struct NodeK {
  u64 node_mid[LEVELS][8];   // per level: the chaining value after a node message's first block
  u64 node_tpl[MAX_NB][32];  // its blocks 2 and 3 per ni = len(uint(index)), index and children zero
};
// What hm96-sha512's key fixes.
struct Hm96K {
  u64 mid_salt[8];           // the chaining value after salt_prefix
  u64 mid_leaf[8];           // after leaf_prefix(key)
  u64 kw_leafpad[80];        // the leaf message's all-padding last block (it is 256 bytes), scheduled (pad_schedule)
};

// sha512.cuh's forms. A kernel's own form is a struct with the same three members.
struct Sha {
  static RC_HD void compress(u64* st, u64* w) { sha::compress(st, w); }
  template <int R0>
  static RC_HD void compress_from(u64* cv, const u64* m, u64* w) { sha::compress_from<R0>(cv, m, w); }
  static RC_HD void compress_kw(u64* st, const u64* kw) { sha::compress_kw(st, kw); }
};
// Sha with compress as sha512.cuh's one-body form (compress_compact), for a fold whose CTA has mostly idle warps.
struct Compact : Sha {
  static RC_HD void compress(u64* st, u64* w) { sha::compress_compact(st, w); }
};

RC_HD int nbytes(u64 x) {
  int n = 1;
  while (x >> (8 * n) && n < 8) ++n;
  return n;
}
RC_HD int nb_of(u64 x) {
  const int n = nbytes(x);
  return n < MAX_NB ? n : MAX_NB;
}
RC_HD void cpy8(u64* d, const u64* s) {
  RC_UNROLL
  for (int i = 0; i < 8; ++i) d[i] = s[i];
}
// 16 bytes at p (16-byte aligned on the device): ldg16 through the non-coherent cache, ld16 an ordinary load
RC_HD q4 ldg16(const void* p) {
#ifdef __CUDA_ARCH__
  return __ldg(reinterpret_cast<const uint4*>(p));
#else
  q4 v;
  memcpy(&v, p, sizeof v);
  return v;
#endif
}
RC_HD q4 ld16(const void* p) {
#ifdef __CUDA_ARCH__
  return *reinterpret_cast<const uint4*>(p);
#else
  q4 v;
  memcpy(&v, p, sizeof v);
  return v;
#endif
}
// the low word of (hi:lo) >> s, s < 32
RC_HD u32 fshr(u32 lo, u32 hi, u32 s) {
#ifdef __CUDA_ARCH__
  return __funnelshift_r(lo, hi, s);
#else
  return (u32)(((u64)hi << 32 | lo) >> (s & 31));
#endif
}
// n words (n even) to p in digest byte order (16-byte aligned on the device)
RC_HD void st_be(uint8_t* p, const u64* d, int n) {
#ifdef __CUDA_ARCH__
  uint4* o = reinterpret_cast<uint4*>(p);
  RC_UNROLL
  for (int i = 0; i < n / 2; ++i)
    o[i] = make_uint4(sx::bswap32((u32)(d[2 * i] >> 32)), sx::bswap32((u32)d[2 * i]), sx::bswap32((u32)(d[2 * i + 1] >> 32)),
                      sx::bswap32((u32)d[2 * i + 1]));
#else
  for (int i = 0; i < n; ++i) sha::put_be64(p + 8 * i, d[i]);
#endif
}

// OR the low NB bytes of v, big-endian, into the message words W at byte offset OFF.
template <int OFF, int NB>
RC_HD void or_be(u64* W, u64 v) {
  constexpr int END = OFF + NB, K = (END - 1) / 8, SH = 8 * (7 - (END - 1) % 8);
  W[K] |= v << SH;
  if constexpr (OFF < 8 * K) W[K - 1] |= v >> (64 - SH);
}
// OR a digest (8 words) into W at byte offset OFF.
template <int OFF>
RC_HD void or_digest(u64* W, const u64* d) {
  constexpr int K0 = OFF / 8, S = 8 * (OFF % 8);
  RC_UNROLL
  for (int i = 0; i < 8; ++i) {
    if constexpr (S == 0) {
      W[K0 + i] |= d[i];
    } else {
      W[K0 + i] |= d[i] >> S;
      W[K0 + i + 1] |= d[i] << (64 - S);
    }
  }
}
// or_digest at word K0 and bit shift S (0 < S < 64) read at run time
template <int K0>
RC_HD void or_digest_s(u64* W, const u64* d, int S) {
  RC_UNROLL
  for (int i = 0; i < 8; ++i) {
    W[K0 + i] |= d[i] >> S;
    W[K0 + i + 1] |= d[i] << (64 - S);
  }
}

// --- the frame: leaf (its two blocks), pad, node -------------------------------------------------------------------------
// leaf message: frame (33) || u32be(4) || "leaf" || u64be(64) || did || u64be(nb) || pos || u64be(nb) || pos || u64be(S) ||
// schema || u64be(64) || value: pos at 121 and 129 + nb, the value at 145 + 2 nb + S, S = 18 + SX. The first block holds
// pos but not the value, so its compression can run ahead (frame_first), and frame_second ends the leaf from its state.
template <int NB, class C = Sha>
RC_HD void frame_first_nb(const FrameK& K, u64 pos, u64* out) {
  u64 W[16];
  RC_UNROLL
  for (int i = 0; i < 16; ++i) W[i] = K.fl_tpl[NB - 1][i];
  or_be<121, NB>(W, pos);
  sha::iv(out);
  C::template compress_from<15>(out, K.fl_m15, W);
}
template <class C = Sha>
RC_HD void frame_first(const FrameK& K, u64 pos, u64* out) {
  switch (nbytes(pos)) {
    case 1: frame_first_nb<1, C>(K, pos, out); break;
    case 2: frame_first_nb<2, C>(K, pos, out); break;
    case 3: frame_first_nb<3, C>(K, pos, out); break;
    default: frame_first_nb<4, C>(K, pos, out); break;
  }
}
template <int SX, int NB, class C = Sha>
RC_HD void frame_second_nb(const FrameK& K, u64 pos, const u64* v, u64* st) {
  u64 W[16];
  RC_UNROLL
  for (int i = 0; i < 16; ++i) W[i] = K.fl_tpl[NB - 1][16 + i];
  or_be<1 + NB, NB>(W, pos);
  or_digest<35 + 2 * NB + SX>(W, v);
  C::compress(st, W);
}
template <int SX, class C = Sha>
RC_HD void frame_second(const FrameK& K, u64 pos, const u64* v, u64* st) {
  switch (nbytes(pos)) {
    case 1: frame_second_nb<SX, 1, C>(K, pos, v, st); break;
    case 2: frame_second_nb<SX, 2, C>(K, pos, v, st); break;
    case 3: frame_second_nb<SX, 3, C>(K, pos, v, st); break;
    default: frame_second_nb<SX, 4, C>(K, pos, v, st); break;
  }
}
template <int SX, class C = Sha>
RC_HD void frame_leaf(const FrameK& K, u64 pos, const u64* v, u64* out) {
  frame_first<C>(K, pos, out);
  frame_second<SX, C>(K, pos, v, out);
}
// frame_leaf with nb fixed: both blocks are formed before the first compression, so the second block's template loads
// don't wait behind it
template <int SX, int NB, class C = Sha>
RC_HD void frame_leaf_nb(const FrameK& K, u64 pos, const u64* v, u64* out) {
  u64 W[32];
  RC_UNROLL
  for (int i = 0; i < 32; ++i) W[i] = K.fl_tpl[NB - 1][i];
  or_be<121, NB>(W, pos);
  or_be<129 + NB, NB>(W, pos);
  or_digest<163 + 2 * NB + SX>(W, v);
  sha::iv(out);
  C::template compress_from<15>(out, K.fl_m15, W);
  C::compress(out, W + 16);
}

template <int NB, class C = Sha>
RC_HD void pad_nb(const FrameK& K, u64 rank, u64* out) {
  u64 W[16];
  RC_UNROLL
  for (int i = 0; i < 16; ++i) W[i] = K.pad_tpl[NB - 1][i];
  or_be<120, NB>(W, rank);
  sha::iv(out);
  C::compress(out, W);
  RC_UNROLL
  for (int i = 0; i < 15; ++i) W[i] = 0;
  W[15] = (120 + NB) * 8;
  C::compress(out, W);
}
template <class C = Sha>
RC_HD void pad(const FrameK& K, u64 rank, u64* out) {
  switch (nbytes(rank)) {
    case 1: pad_nb<1, C>(K, rank, out); break;
    case 2: pad_nb<2, C>(K, rank, out); break;
    case 3: pad_nb<3, C>(K, rank, out); break;
    default: pad_nb<4, C>(K, rank, out); break;
  }
}

// The leaf's blocks and the pad with nb read at run time. For nb <= 4 pos lands in word 15 of the leaf's first block and of
// the pad's, across words 0 and 1 of the leaf's second, and the value in words 4.. or 5.. of it at a shift that is never 0
// for the two hm96 schemas (SX 0 and 4).
template <class C = Sha>
RC_HD void frame_first_rt(const FrameK& K, u64 pos, u64* out) {
  const int nb = nb_of(pos);
  u64 W[16];
  RC_UNROLL
  for (int i = 0; i < 16; ++i) W[i] = K.fl_tpl[nb - 1][i];
  W[15] |= pos << (8 * (7 - nb));
  sha::iv(out);
  C::template compress_from<15>(out, K.fl_m15, W);
}
template <int SX, class C = Sha>
RC_HD void frame_second_rt(const FrameK& K, u64 pos, const u64* v, u64* st) {
  static_assert(SX == 0 || SX == 4, "the run-time leaf takes the hm96 schemas");
  const int nb = nb_of(pos), off = 35 + 2 * nb + SX, sh = 8 * (15 - 2 * nb);
  u64 W[16];
  RC_UNROLL
  for (int i = 0; i < 16; ++i) W[i] = K.fl_tpl[nb - 1][16 + i];
  if (sh >= 64) {
    W[0] |= pos << (sh - 64);
  } else {
    W[0] |= pos >> (64 - sh);
    W[1] |= pos << sh;
  }
  if ((off >> 3) == 4) or_digest_s<4>(W, v, 8 * (off & 7));
  else or_digest_s<5>(W, v, 8 * (off & 7));
  C::compress(st, W);
}
template <class C = Sha>
RC_HD void pad_rt(const FrameK& K, u64 rank, u64* out) {
  const int nb = nb_of(rank);
  u64 W[16];
  RC_UNROLL
  for (int i = 0; i < 16; ++i) W[i] = K.pad_tpl[nb - 1][i];
  W[15] |= rank << (8 * (8 - nb));
  sha::iv(out);
  RC_UNROLL1
  for (int k = 0; k < 2; ++k) {
    C::compress(out, W);
    RC_UNROLL
    for (int i = 0; i < 15; ++i) W[i] = 0;
    W[15] = (120 + nb) * 8;
  }
}

// node(level, index, L, R) from tpl (NodeK's node_tpl) and mid (its node_mid[level]), the index of NI bytes.
template <int NI, class C = Sha>
RC_HD void node_nb_from(const u64 (*tpl)[32], const u64* mid, u64 index, const u64* L, const u64* R, u64* out) {
  u64 W[32];
  RC_UNROLL
  for (int i = 0; i < 32; ++i) W[i] = tpl[NI - 1][i];
  or_be<2, NI>(W, index);
  or_digest<10 + NI>(W, L);
  or_digest<82 + NI>(W, R);
  cpy8(out, mid);
  C::compress(out, W);
  C::compress(out, W + 16);
}
template <class C = Sha>
RC_HD void node_from(const u64 (*tpl)[32], const u64* mid, u64 index, const u64* L, const u64* R, u64* out) {
  switch (nbytes(index)) {
    case 1: node_nb_from<1, C>(tpl, mid, index, L, R, out); break;
    case 2: node_nb_from<2, C>(tpl, mid, index, L, R, out); break;
    case 3: node_nb_from<3, C>(tpl, mid, index, L, R, out); break;
    default: node_nb_from<4, C>(tpl, mid, index, L, R, out); break;
  }
}
template <class C = Sha>
RC_HD void node(const NodeK& N, int level, u64 index, const u64* L, const u64* R, u64* out) {
  node_from<C>(N.node_tpl, N.node_mid[level], index, L, R, out);
}
// node_from with ni read at run time: for ni <= 4 the index and both children land in the same words as or_be<2, ni> and
// or_digest<10 + ni>, <82 + ni> put them, and only the shifts depend on ni, so one compression body serves every length and
// both blocks.
template <class C = Sha>
RC_HD void node_rt_from(const u64 (*tpl)[32], const u64* mid, u64 index, const u64* L, const u64* R, u64* out) {
  const int ni = nb_of(index), S = 8 * (2 + ni);
  u64 W[32];
  RC_UNROLL
  for (int i = 0; i < 32; ++i) W[i] = tpl[ni - 1][i];
  W[0] |= index << (8 * (6 - ni));
  RC_UNROLL
  for (int i = 0; i < 8; ++i) {
    W[1 + i] |= L[i] >> S;
    W[2 + i] |= L[i] << (64 - S);
    W[10 + i] |= R[i] >> S;
    W[11 + i] |= R[i] << (64 - S);
  }
  cpy8(out, mid);
  RC_UNROLL1
  for (int k = 0; k < 2; ++k) {
    u64 w[16];
    RC_UNROLL
    for (int i = 0; i < 16; ++i) w[i] = k ? W[16 + i] : W[i];
    C::compress(out, w);
  }
}
template <class C = Sha>
RC_HD void node_rt(const NodeK& N, int level, u64 index, const u64* L, const u64* R, u64* out) {
  node_rt_from<C>(N.node_tpl, N.node_mid[level], index, L, R, out);
}

// --- the hm96-sha512 leaf ------------------------------------------------------------------------------------------------

// c = SHA-512(salt_prefix || y), y the salt as 24 big-endian words
template <class C = Sha>
RC_HD void salt_digest(const Hm96K& H, const u64* Y, u64* c) {
  u64 w[16];
  cpy8(c, H.mid_salt);
  RC_UNROLL
  for (int i = 0; i < 16; ++i) w[i] = Y[i];
  C::compress(c, w);
  RC_UNROLL
  for (int i = 0; i < 8; ++i) w[i] = Y[16 + i];
  w[8] = 0x8000000000000000ull;
  RC_UNROLL
  for (int i = 9; i < 15; ++i) w[i] = 0;
  w[15] = (128 + 192) * 8;
  C::compress(c, w);
}

// h = SHA-512(leaf_prefix(key) || b || c) from B = b || c (16 words, which the schedule overwrites)
template <class C = Sha>
RC_HD void hm96_value(const Hm96K& H, u64* B, u64* h) {
  cpy8(h, H.mid_leaf);
  C::compress(h, B);
  C::compress_kw(h, H.kw_leafpad);
}

// The hm96 leaf's frame leaf, into f, which holds the state after the leaf's first block (frame_first, frame_first_rt
// when RT), from the row digest x, M_key y as words (my) and c; b || c to bc and h to hs, in digest byte order, where given.
template <int SX, bool RT = false, class C = Sha>
RC_HD void finish_leaf(const FrameK& F, const Hm96K& H, u64 pos, const u64* x, const u64* my, const u64* c, u64* f, uint8_t* bc,
                       uint8_t* hs) {
  u64 B[16], h[8];
  RC_UNROLL
  for (int w = 0; w < 8; ++w) {
    B[w] = x[w] ^ my[w];
    B[8 + w] = c[w];
  }
  if (bc) st_be(bc, B, 16);
  hm96_value<C>(H, B, h);
  if (hs) st_be(hs, h, 8);
  if constexpr (RT) frame_second_rt<SX, C>(F, pos, h, f);
  else frame_second<SX, C>(F, pos, h, f);
}

// The nibble table (nib_entry's), through L1 (LDG) or from shared memory: chunk c of row T_u at 256 c + 16 u, of T_16u
// 4096 on.
template <bool LDG>
struct NibTab {
  const uint8_t* T;
  // A ^= the 80 bytes [16 q, 16 q + 80) of T_v
  RC_HD void xor80(u32* A, int q, u32 v) const {
    const uint8_t* lo = T + 256 * q + 16 * (v & 15u);
    const uint8_t* hi = T + 4096 + 256 * q + 16 * (v >> 4);
    RC_UNROLL
    for (int i = 0; i < 5; ++i) {
      const q4 u = LDG ? ldg16(lo + 256 * i) : ld16(lo + 256 * i), w = LDG ? ldg16(hi + 256 * i) : ld16(hi + 256 * i);
      A[4 * i] ^= u.x ^ w.x; A[4 * i + 1] ^= u.y ^ w.y; A[4 * i + 2] ^= u.z ^ w.z; A[4 * i + 3] ^= u.w ^ w.w;
    }
  }
};
// The byte table: row T_v (256 bytes) at v STRIDE, through L1 (LDG) or from shared memory, where a STRIDE of 272 puts a
// warp's rows in different banks (byte_chunk fills it from the nibble table).
template <int STRIDE, bool LDG>
struct ByteTab {
  const uint8_t* T;
  RC_HD void xor80(u32* A, int q, u32 v) const {
    const uint8_t* t = T + v * STRIDE + 16 * q;
    RC_UNROLL
    for (int i = 0; i < 5; ++i) {
      const q4 u = LDG ? ldg16(t + 16 * i) : ld16(t + 16 * i);
      A[4 * i] ^= u.x; A[4 * i + 1] ^= u.y; A[4 * i + 2] ^= u.z; A[4 * i + 3] ^= u.w;
    }
  }
};

// M_key y as 16 little-endian u32 (mask byte 4 w + k is byte k of m[w]) from the table T (NibTab, ByteTab): for salt byte
// p = 16 q + r, the bytes [16 q, 16 q + 80) of T_(y_p) are xored into A_r, and A_r shifted down r bytes into m. ROLL: the
// loop over r rolled, A_r shifted by whole words in place, for a kernel that waits on instruction fetch.
template <bool ROLL, class Tab>
RC_HD void mask(const Tab& T, const u64* Y, u32* m) {
  RC_UNROLL
  for (int w = 0; w < 16; ++w) m[w] = 0;
  if constexpr (ROLL) {
    RC_UNROLL1
    for (int r = 0; r < 16; ++r) {
      u32 A[20];
      RC_UNROLL
      for (int i = 0; i < 20; ++i) A[i] = 0;
      const int sh = 56 - 8 * (r & 7);
      RC_UNROLL
      for (int q = 0; q < 12; ++q) T.xor80(A, q, (u32)((r < 8 ? Y[2 * q] : Y[2 * q + 1]) >> sh) & 0xFFu);
      if (r & 4) {
        RC_UNROLL
        for (int k = 0; k < 19; ++k) A[k] = A[k + 1];
      }
      if (r & 8) {
        RC_UNROLL
        for (int k = 0; k < 18; ++k) A[k] = A[k + 2];
      }
      const u32 s = 8 * (r & 3);
      RC_UNROLL
      for (int w = 0; w < 16; ++w) m[w] ^= fshr(A[w], A[w + 1], s);
    }
  } else {
    RC_UNROLL
    for (int r = 0; r < 16; ++r) {
      u32 A[20];
      RC_UNROLL
      for (int i = 0; i < 20; ++i) A[i] = 0;
      RC_UNROLL
      for (int q = 0; q < 12; ++q) {
        const int p = 16 * q + r;
        T.xor80(A, q, (u32)(Y[p >> 3] >> (56 - 8 * (p & 7))) & 0xFFu);
      }
      const int sw = r >> 2, s = 8 * (r & 3);
      RC_UNROLL
      for (int w = 0; w < 16; ++w) m[w] ^= s ? fshr(A[w + sw], A[w + sw + 1], s) : A[w + sw];
    }
  }
}
// mask's words as M_key y's 8 big-endian words
RC_HD void mask_words(const u32* m, u64* my) {
  RC_UNROLL
  for (int i = 0; i < 8; ++i) my[i] = (u64)sx::bswap32(m[2 * i]) << 32 | sx::bswap32(m[2 * i + 1]);
}

// --- the constants, on the host or the device ----------------------------------------------------------------------------

RC_HD u64 be64_at(const uint8_t* p) {
  u64 v = 0;
  for (int i = 0; i < 8; ++i) v = v << 8 | p[i];
  return v;
}
RC_HD void be_put(uint8_t* p, u64 v, int n) {
  for (int i = n - 1; i >= 0; --i, v >>= 8) p[i] = (uint8_t)v;
}
// the chaining value after one block
RC_HD void midstate(const uint8_t* block, u64* st) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) w[i] = be64_at(block + 8 * i);
  sha::iv(st);
  sha::compress(st, w);
}
// the all-padding last block of a message of `bytes` bytes (a multiple of 128), scheduled for compress_kw
RC_HD void pad_schedule(u64 bytes, u64* kw) {
  u64 blk[16] = {0x8000000000000000ull};
  blk[15] = bytes * 8;
  sha::schedule_kw(blk, kw);
}

// frame-v3's message (merkle._message) into m, zero past it: FRAME || u32be(len tag) || tag || u64be(64) || did, then the
// parts; each returns the length so far.
RC_HD int frame_head(uint8_t* m, const char* tag, const uint8_t* did) {
  const char frame[] = "veritor/protocol/merkle/frame/v3";
  int n = 0;
  for (int i = 0; i < (int)sizeof frame; ++i) m[n++] = (uint8_t)frame[i];   // with its NUL: 33 bytes
  int tl = 0;
  while (tag[tl]) ++tl;
  be_put(m + n, (u64)tl, 4);
  n += 4;
  for (int i = 0; i < tl; ++i) m[n++] = (uint8_t)tag[i];
  be_put(m + n, 64, 8);
  n += 8;
  for (int i = 0; i < 64; ++i) m[n++] = did[i];
  return n;
}
// u64be(len) || b (len zero bytes when b is null)
RC_HD int frame_part(uint8_t* m, int n, const uint8_t* b, int len) {
  be_put(m + n, (u64)len, 8);
  n += 8;
  for (int i = 0; i < len; ++i) m[n++] = b ? b[i] : 0;
  return n;
}
// u64be(len(uint(v))) || uint(v), uint(v) big-endian in the fewest bytes (one for 0)
RC_HD int frame_uint(uint8_t* m, int n, u64 v) {
  const int k = nbytes(v);
  be_put(m + n, (u64)k, 8);
  be_put(m + n + 8, v, k);
  return n + 8 + k;
}
// the n-byte message in m (zero past n, with room for its padding) padded, as words into w; the words
RC_HD int padded_words(uint8_t* m, int n, u64* w) {
  const int total = (n + 17 + 127) / 128 * 128;
  m[n] = 0x80;
  be_put(m + total - 8, (u64)n * 8, 8);
  for (int i = 0; i < total / 8; ++i) w[i] = be64_at(m + 8 * i);
  return total / 8;
}
// SHA-512 of the n-byte message in m (zero past n, with room for its padding), any number of blocks
RC_HD void digest_padded(uint8_t* m, int n, u64* out) {
  const int total = (n + 17 + 127) / 128 * 128;
  m[n] = 0x80;
  be_put(m + total - 8, (u64)n * 8, 8);
  sha::iv(out);
  for (int b = 0; b < total; b += 128) {
    u64 w[16];
    for (int i = 0; i < 16; ++i) w[i] = be64_at(m + b + 8 * i);
    sha::compress(out, w);
  }
}

// FrameK's fl_tpl[nb - 1] for domain id did and a schema of sl bytes (at most MAX_SCHEMA): the leaf's two blocks with pos
// and the value zero. 0, or -1.
RC_HD int leaf_frame_nb(const uint8_t* did, const uint8_t* schema, int sl, int nb, FrameK& K) {
  if (sl < 0 || sl > MAX_SCHEMA || nb < 1 || nb > MAX_NB) return -1;
  uint8_t m[256] = {};
  int n = frame_head(m, "leaf", did);
  n = frame_part(m, n, nullptr, nb);
  n = frame_part(m, n, nullptr, nb);
  n = frame_part(m, n, schema, sl);
  n = frame_part(m, n, nullptr, 64);
  if (n + 17 > 256 || padded_words(m, n, K.fl_tpl[nb - 1]) != 32) return -1;
  K.sx = sl - 18;
  return 0;
}
// fl_m15 from fl_tpl[0]: rounds 0..14 of the leaf's first block, whose words 0..14 are every nb's
RC_HD void leaf_m15(FrameK& K) {
  u64 st[8];
  sha::iv(st);
  sha::rounds_prefix(st, K.fl_tpl[0], 15, K.fl_m15);
}
// fl_tpl for every nb, sx and fl_m15. 0, or -1.
RC_HD int leaf_frame(const uint8_t* did, const uint8_t* schema, int sl, FrameK& K) {
  for (int nb = 1; nb <= MAX_NB; ++nb) {
    if (leaf_frame_nb(did, schema, sl, nb, K)) return -1;
    for (int i = 0; i < 15; ++i)
      if (K.fl_tpl[nb - 1][i] != K.fl_tpl[0][i]) return -1;
  }
  leaf_m15(K);
  return 0;
}
// FrameK's pad_tpl[nb - 1] for domain id did: a pad's first block with rank zero
RC_HD void pad_tpl_nb(const uint8_t* did, int nb, FrameK& K) {
  uint8_t m[256] = {};
  u64 w[32];
  padded_words(m, frame_part(m, frame_head(m, "pad", did), nullptr, nb), w);
  for (int i = 0; i < 16; ++i) K.pad_tpl[nb - 1][i] = w[i];
}
RC_HD void pad_tpls(const uint8_t* did, FrameK& K) {
  for (int nb = 1; nb <= MAX_NB; ++nb) pad_tpl_nb(did, nb, K);
}
// the whole FrameK of domain id did and the schema. 0, or -1.
RC_HD int frame_plan(const uint8_t* did, const uint8_t* schema, int sl, FrameK& K) {
  if (leaf_frame(did, schema, sl, K)) return -1;
  pad_tpls(did, K);
  return 0;
}
// node_tpl: a node message's blocks 2 and 3 per ni = len(uint(index)), index and children zero; no domain or level reaches
// them. 0, or -1.
RC_HD int node_tpls(u64 (*tpl)[32]) {
  const uint8_t zero[64] = {}, level = 0;
  for (int ni = 1; ni <= MAX_NB; ++ni) {
    uint8_t m[384] = {};
    u64 w[48];
    int n = frame_head(m, "node", zero);
    n = frame_part(m, n, &level, 1);
    n = frame_part(m, n, nullptr, ni);
    n = frame_part(m, n, nullptr, 64);
    n = frame_part(m, n, nullptr, 64);
    if (padded_words(m, n, w) != 48) return -1;
    for (int i = 0; i < 32; ++i) tpl[ni - 1][i] = w[16 + i];
  }
  return 0;
}
// node_mid[0 .. levels - 1] for domain did: the chaining value after the first block of a node message at each level
RC_HD void node_mids(const uint8_t* did, u64 (*mid)[8], int levels) {
  for (int l = 0; l < levels; ++l) {
    uint8_t m[384] = {};
    const uint8_t lv = (uint8_t)l;
    frame_part(m, frame_head(m, "node", did), &lv, 1);
    midstate(m, mid[l]);
  }
}
// the whole NodeK of domain id did. 0, or -1.
RC_HD int node_plan(const uint8_t* did, NodeK& N) {
  if (node_tpls(N.node_tpl)) return -1;
  node_mids(did, N.node_mid, LEVELS);
  return 0;
}
// the root of domain did's empty tree
RC_HD void empty_root(const uint8_t* did, u64* out) {
  uint8_t m[256] = {};
  digest_padded(m, frame_head(m, "empty", did), out);
}

// Hm96K of key (256 bytes): the chaining values after salt_prefix and after leaf_prefix(key), each its tag (with its NUL)
// zero-padded to a block, leaf_prefix's after the tag SHA-512(key); and kw_leafpad, which is the key's padding too
RC_HD void hm96_plan(const uint8_t* key, Hm96K& H) {
  const char salt_tag[] = "verity/hm96-sha512/salt/v1", leaf_tag[] = "verity/hm96-sha512/leaf/v1";
  uint8_t blk[128] = {};
  for (int i = 0; i < (int)sizeof salt_tag; ++i) blk[i] = (uint8_t)salt_tag[i];
  midstate(blk, H.mid_salt);
  pad_schedule(256, H.kw_leafpad);
  u64 st[8], w[16];
  sha::iv(st);
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 16; ++i) w[i] = be64_at(key + 128 * b + 8 * i);
    sha::compress(st, w);
  }
  sha::compress_kw(st, H.kw_leafpad);
  for (int i = 0; i < 128; ++i) blk[i] = 0;
  for (int i = 0; i < (int)sizeof leaf_tag; ++i) blk[i] = (uint8_t)leaf_tag[i];
  for (int i = 0; i < 8; ++i) be_put(blk + (int)sizeof leaf_tag + 8 * i, st[i], 8);
  midstate(blk, H.mid_leaf);
}
// entry t < 512 of key's nibble table (NibTab's, TN_BYTES): T_v for v a low nibble u (h = 0) or a high one 16 u (h = 1),
// its chunk c (16 bytes) at 4096 h + 256 c + 16 u of tab
RC_HD void nib_entry(const uint8_t* key, u32 t, uint8_t* tab) {
  const u32 h = t >> 8, c = (t >> 4) & 15, u = t & 15, v = h ? 16 * u : u;
  uint8_t* o = tab + 4096 * h + 256 * c + 16 * u;
  for (int k = 0; k < 16; ++k) {
    const int at = 16 * c + k;
    u32 acc = 0;
    for (int s = 0; s < 8; ++s)
      if (v >> s & 1) {
        const u32 lo = key[at], hi = at + 1 < 256 ? key[at + 1] : 0;
        acc ^= ((lo | hi << 8) >> s) & 0xFF;
      }
    o[k] = (uint8_t)acc;
  }
}
// chunk c of the byte table's row T_b (ByteTab's) from the nibble table Tn
RC_HD q4 byte_chunk(const q4* Tn, u32 b, u32 c) {
  const q4 lo = Tn[16 * c + (b & 15)], hi = Tn[256 + 16 * c + (b >> 4)];
  q4 r;
  r.x = lo.x ^ hi.x; r.y = lo.y ^ hi.y; r.z = lo.z ^ hi.z; r.w = lo.w ^ hi.w;
  return r;
}

}  // namespace rc
