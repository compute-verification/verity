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
// A kernel with no such plan takes the forms at the end: a leaf from its head (any rank, position and schema, the prefix
// built once per row), a node from its chaining value alone and a pad from its domain id, for any value below 2^64.
// Any message can also be built byte by byte into a sink of the kernel's own, whole (put_leaf, put_pad, put_node) or a part
// at a time (put_tag, put_head, put_part, put_part64, put_words, put_uint), then padded (put_padding) or hashed from it
// (digest), or hashed as it is built by a sink that streams.
//
// The hm96-sha512 leaf of a row digest x under the salt y: c = SHA-512(salt_prefix || y), b = x ^ M_key y and the value
// h = SHA-512(leaf_prefix(key) || b || c) the frame leaf binds (Hm96K). M_key y comes from a table of T_v, the XOR of
// (key >> s) over the set bits s of v: the nibble table (T_v = T_(v & 15) ^ T_(v & 0xF0), TN_BYTES) or the byte table (a
// row of 256 bytes a byte value), read through L1 or from shared memory, or from the position table (an entry a salt byte
// and byte value, PosTab).
//
// Every digest is 8 big-endian words, as SHA-512's state holds it. The compression is a policy C, by default `Sha`
// (sha512.cuh's `sha::` forms; `Compact` is its one-body form), so a kernel passes its own form of the same function.
#pragma once
#include <cstdint>
#include <cstring>
#include <utility>

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
  while (n < 8 && x >> (8 * n)) ++n;
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

// The salt's 24 big-endian words as y(i): from words (SaltWords), or from 48 little-endian u32 words ys apart (SaltU32, in
// shared memory, say, so that a thread's registers stay free).
struct SaltWords {
  const u64* Y;
  RC_HD u64 operator()(int i) const { return Y[i]; }
};
struct SaltU32 {
  const u32* y;
  u32 ys;
  RC_HD u64 operator()(int i) const { return sx::be64(y[2 * i * ys], y[(2 * i + 1) * ys]); }
};

// c = SHA-512(salt_prefix || y) from mid_salt (Hm96K's)
template <class C = Sha, class Y>
RC_HD void salt_digest(const u64* mid_salt, const Y& y, u64* c) {
  u64 w[16];
  cpy8(c, mid_salt);
  RC_UNROLL
  for (int i = 0; i < 16; ++i) w[i] = y(i);
  C::compress(c, w);
  RC_UNROLL
  for (int i = 0; i < 8; ++i) w[i] = y(16 + i);
  w[8] = 0x8000000000000000ull;
  RC_UNROLL
  for (int i = 9; i < 15; ++i) w[i] = 0;
  w[15] = (128 + 192) * 8;
  C::compress(c, w);
}
// with y the salt as 24 big-endian words
template <class C = Sha>
RC_HD void salt_digest(const Hm96K& H, const u64* Y, u64* c) {
  salt_digest<C>(H.mid_salt, SaltWords{Y}, c);
}

// h = SHA-512(leaf_prefix(key) || b || c) from mid_leaf, the schedule kw of the message's all-padding last block (Hm96K's)
// and B = b || c (16 words, which the schedule overwrites)
template <class C = Sha>
RC_HD void hm96_value(const u64* mid_leaf, const u64* kw, u64* B, u64* h) {
  cpy8(h, mid_leaf);
  C::compress(h, B);
  C::compress_kw(h, kw);
}
template <class C = Sha>
RC_HD void hm96_value(const Hm96K& H, u64* B, u64* h) {
  hm96_value<C>(H.mid_leaf, H.kw_leafpad, B, h);
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
// The position table: entry (p, v) at 64 (256 p + v) is hm96's byte_tables(key)[p][v] as 64 little-endian bytes, the XOR of
// columns 8 p + s over the set bits s of v (3 MB), through L1 (LDG) or an ordinary load: a lookup a salt byte and no shift.
// WORDS: each entry as its 8 big-endian words instead (64-bit words in memory), so M_key y XORs into a digest's words as
// read. xor64 accumulates into 16 u32 m either way, and words(m, my) gives M_key y's 8 big-endian words.
template <bool LDG, bool WORDS = false>
struct PosTab {
  const uint8_t* R;
  RC_HD const uint8_t* at(int p, u32 v) const { return R + 64 * (256 * (size_t)p + v); }
  RC_HD void xor64(u32* m, int p, u32 v) const {
    const uint8_t* e = at(p, v);
    RC_UNROLL
    for (int k = 0; k < 4; ++k) {
      const q4 u = LDG ? ldg16(e + 16 * k) : ld16(e + 16 * k);
      m[4 * k] ^= u.x; m[4 * k + 1] ^= u.y; m[4 * k + 2] ^= u.z; m[4 * k + 3] ^= u.w;
    }
  }
  RC_HD void words(const u32* m, u64* my) const;
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
// mask from the salt as 48 little-endian u32 words ys apart (SaltU32's), taking the salt's 16-byte column q = col(k) at step
// k, col a permutation of 0 .. 11: a lane's order, which spreads a warp's loads over shared memory's banks, and the XOR is
// the same in any order. Word 4 q + a holds salt bytes 16 q + 4 a .. 16 q + 4 a + 3, so each word loaded serves four r.
template <class Tab, class Col>
RC_HD void mask_lanes(const Tab& T, const u32* y, u32 ys, const Col& col, u32* m) {
  RC_UNROLL
  for (int w = 0; w < 16; ++w) m[w] = 0;
  RC_UNROLL
  for (int a = 0; a < 4; ++a) {
    u32 yw[12];
    RC_UNROLL
    for (int k = 0; k < 12; ++k) yw[k] = y[(4 * col(k) + a) * ys];
    RC_UNROLL
    for (int rr = 0; rr < 4; ++rr) {
      u32 A[20];
      RC_UNROLL
      for (int i = 0; i < 20; ++i) A[i] = 0;
      RC_UNROLL
      for (int k = 0; k < 12; ++k) T.xor80(A, col(k), (yw[k] >> (8 * rr)) & 0xFFu);
      const int sh = 8 * rr;
      RC_UNROLL
      for (int w = 0; w < 16; ++w) m[w] ^= sh ? (A[w + a] >> sh | A[w + a + 1] << (32 - sh)) : A[w + a];
    }
  }
}
// mask from the position table (PosTab), the salt as its 48 little-endian u32 words
template <class Tab>
RC_HD void mask_positions(const Tab& T, const u32* y, u32* m) {
  RC_UNROLL
  for (int w = 0; w < 16; ++w) m[w] = 0;
  RC_UNROLL
  for (int q = 0; q < 48; ++q) {
    RC_UNROLL
    for (int r = 0; r < 4; ++r) T.xor64(m, 4 * q + r, (y[q] >> (8 * r)) & 0xFFu);
  }
}
// mask's words as M_key y's 8 big-endian words
RC_HD void mask_words(const u32* m, u64* my) {
  RC_UNROLL
  for (int i = 0; i < 8; ++i) my[i] = (u64)sx::bswap32(m[2 * i]) << 32 | sx::bswap32(m[2 * i + 1]);
}
template <bool LDG, bool WORDS>
RC_HD void PosTab<LDG, WORDS>::words(const u32* m, u64* my) const {
  if constexpr (WORDS) {
    RC_UNROLL
    for (int i = 0; i < 8; ++i) my[i] = (u64)m[2 * i + 1] << 32 | m[2 * i];
  } else {
    mask_words(m, my);
  }
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

// A message as a sink: s.put(b) appends the byte b and s.n counts them; s.word(i) reads bytes [8 i, 8 i + 8) back as a
// big-endian word (digest). Bytes is a contiguous buffer; a kernel that keeps its messages elsewhere (a strided slice of
// shared memory, say, so that no thread's message is a local array) or hashes them as they are built passes its own, which
// may also take a whole big-endian word at once (s.put8(v)).
struct Bytes {
  uint8_t* m;
  int n;
  RC_HD void put(u32 b) { m[n++] = (uint8_t)b; }
  RC_HD u64 word(int i) const { return be64_at(m + 8 * i); }
};
// v's low k bytes, big-endian
template <class S>
RC_HD void put_be(S& s, u64 v, int k) {
  for (int i = k - 1; i >= 0; --i) s.put((u32)(v >> (8 * i)) & 0xFFu);
}
// A sink streams when it takes a whole word (s.put8): it hashes as it fills, so any of its puts may end a block and
// compress it, and the forms below keep its puts few and in loops rather than unrolled.
template <class S, class = void>
struct Streams {
  static constexpr bool value = false;
};
template <class S>
struct Streams<S, decltype((void)std::declval<S&>().put8(u64{}))> {
  static constexpr bool value = true;
};
// v's 8 bytes, big-endian: through s.put8 where the sink streams
template <class S>
RC_HD void put_word(S& s, u64 v) {
  if constexpr (Streams<S>::value) s.put8(v);
  else put_be(s, v, 8);
}
constexpr int FRAME_LEN = 33;   // merkle._FRAME, with its NUL
// frame-v3's message (merkle._message) up to its parts: FRAME || u32be(len tag) || tag; into a sink that streams, the
// frame's 32 characters a word at a time, then its NUL
template <class S>
RC_HD void put_tag(S& s, const char* tag) {
  const char* frame = "veritor/protocol/merkle/frame/v3";
  if constexpr (Streams<S>::value) {
    RC_UNROLL
    for (int w = 0; w < (FRAME_LEN - 1) / 8; ++w) {
      u64 v = 0;
      RC_UNROLL
      for (int i = 0; i < 8; ++i) v = v << 8 | (uint8_t)frame[8 * w + i];
      s.put8(v);
    }
    s.put(0);
  } else {
    RC_UNROLL
    for (int i = 0; i < FRAME_LEN; ++i) s.put((uint8_t)frame[i]);
  }
  int tl = 0;
  while (tag[tl]) ++tl;
  put_be(s, (u64)tl, 4);
  for (int i = 0; i < tl; ++i) s.put((uint8_t)tag[i]);
}
// a part: u64be(len) || b (len zero bytes when b is null)
template <class S>
RC_HD void put_part(S& s, const uint8_t* b, int len) {
  put_be(s, (u64)len, 8);
  for (int i = 0; i < len; ++i) s.put(b ? b[i] : 0);
}
// a 64-byte part (a domain id, a value, a child): its bytes, or its 8 big-endian words (a digest's) of any 64-bit type
template <class S>
RC_HD void put_part64(S& s, const uint8_t* b) {
  put_part(s, b, 64);
}
template <class S, class W>
RC_HD void put_part64(S& s, const W* d) {
  static_assert(sizeof(W) == 8, "a 64-byte part's words are 64-bit (put_words takes little-endian u32)");
  put_be(s, 64, 8);
  for (int k = 0; k < 8; ++k) put_word(s, (u64)d[k]);
}
// every frame message's head: the tag, then the domain id's part
template <class S, class D>
RC_HD void put_head(S& s, const char* tag, const D* did) {
  put_tag(s, tag);
  put_part64(s, did);
}
// a part of the 4 nw bytes the little-endian u32 words q hold (a part in device memory, read a word at a time)
template <class S>
RC_HD void put_words(S& s, const u32* q, int nw) {
  put_be(s, 4 * (u64)nw, 8);
  for (int i = 0; i < nw; ++i) {
    const u32 x = q[i];
    RC_UNROLL
    for (int j = 0; j < 4; ++j) s.put((x >> (8 * j)) & 0xFFu);
  }
}
// the part uint(v): u64be(len(uint(v))) || uint(v), uint(v) big-endian in the fewest bytes (one for 0)
template <class S>
RC_HD void put_uint(S& s, u64 v) {
  const int k = nbytes(v);
  put_be(s, (u64)k, 8);
  put_be(s, v, k);
}
// The tree's messages whole, the domain id and the 64-byte parts as put_part64 takes them. A kernel that resumes a message
// from a snapshot of its first parts puts the rest of them, in this order.
//   leaf: the head, uint(rank), uint(position), the schema's part, the value's part
template <class S, class D, class V>
RC_HD void put_leaf(S& s, const D* did, u64 rank, u64 position, const uint8_t* schema, int sl, const V* v) {
  put_head(s, "leaf", did);
  put_uint(s, rank);
  put_uint(s, position);
  put_part(s, schema, sl);
  put_part64(s, v);
}
//   pad: the head, uint(rank)
template <class S, class D>
RC_HD void put_pad(S& s, const D* did, u64 rank) {
  put_head(s, "pad", did);
  put_uint(s, rank);
}
//   node: the head, uint(level), uint(index), L's part, R's part
template <class S, class D, class W>
RC_HD void put_node(S& s, const D* did, u64 level, u64 index, const W* L, const W* R) {
  put_head(s, "node", did);
  put_uint(s, level);
  put_uint(s, index);
  put_part64(s, L);
  put_part64(s, R);
}
// SHA-512's padding of the message in s, appended through s: its blocks
template <class S>
RC_HD int put_padding(S& s) {
  const int n = s.n, total = (n + 17 + 127) / 128 * 128;
  s.put(0x80);
  while (s.n < total - 8) s.put(0);
  put_be(s, (u64)n * 8, 8);
  return total / 128;
}
// SHA-512 of the message in s, any number of blocks: its padding appended through s, then its blocks read back
template <class C = Sha, class S>
RC_HD void digest(S& s, u64* out) {
  const int blocks = put_padding(s);
  sha::iv(out);
  for (int b = 0; b < blocks; ++b) {
    u64 w[16];
    RC_UNROLL
    for (int i = 0; i < 16; ++i) w[i] = s.word(16 * b + i);
    C::compress(out, w);
  }
}
// the chaining value after one block
template <class C = Sha>
RC_HD void midstate(const uint8_t* block, u64* st) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) w[i] = be64_at(block + 8 * i);
  sha::iv(st);
  C::compress(st, w);
}
// the all-padding last block of a message of `bytes` bytes (a multiple of 128), scheduled for compress_kw
RC_HD void pad_schedule(u64 bytes, u64* kw) {
  u64 blk[16] = {0x8000000000000000ull};
  blk[15] = bytes * 8;
  sha::schedule_kw(blk, kw);
}

// The same into the buffer m from byte n, zero past it (put_head at 0, put_part, put_uint); each returns the length so far.
RC_HD int frame_head(uint8_t* m, const char* tag, const uint8_t* did) {
  Bytes s{m, 0};
  put_head(s, tag, did);
  return s.n;
}
RC_HD int frame_part(uint8_t* m, int n, const uint8_t* b, int len) {
  Bytes s{m, n};
  put_part(s, b, len);
  return s.n;
}
RC_HD int frame_uint(uint8_t* m, int n, u64 v) {
  Bytes s{m, n};
  put_uint(s, v);
  return s.n;
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
template <class C = Sha>
RC_HD void digest_padded(uint8_t* m, int n, u64* out) {
  const int total = (n + 17 + 127) / 128 * 128;
  m[n] = 0x80;
  be_put(m + total - 8, (u64)n * 8, 8);
  sha::iv(out);
  for (int b = 0; b < total; b += 128) {
    u64 w[16];
    for (int i = 0; i < 16; ++i) w[i] = be64_at(m + b + 8 * i);
    C::compress(out, w);
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
// the chaining value after the first block of domain did's node message at level (below 256): its bytes past the level are
// the first six of u64be(len uint(index)), all zero
template <class C = Sha>
RC_HD void node_mid(const uint8_t* did, int level, u64* mid) {
  uint8_t m[128] = {};
  const uint8_t lv = (uint8_t)level;
  frame_part(m, frame_head(m, "node", did), &lv, 1);
  midstate<C>(m, mid);
}
// node_mid[0 .. levels - 1] for domain did
RC_HD void node_mids(const uint8_t* did, u64 (*mid)[8], int levels) {
  for (int l = 0; l < levels; ++l) node_mid(did, l, mid[l]);
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

// --- with no plan: a leaf from its head, a node from its chaining value alone, a pad from its domain id ------------------

// w[0, NW) |= the NV words v at byte offset off (off >> 3 <= MAXQ, off + 8 NV <= 8 NW). Every index is a constant (the word
// offset is selected, not indexed), so w and v stay in registers.
template <int NW, int NV, int MAXQ>
RC_HD void place(u64* w, const u64* v, u32 off) {
  const u32 qw = off >> 3, sh = 8 * (off & 7);
  u64 u[NV + 1];
  RC_UNROLL
  for (int k = 0; k <= NV; ++k) u[k] = (k < NV ? v[k] >> sh : 0) | ((k > 0 && sh) ? v[k - 1] << (64 - sh) : 0);
  RC_UNROLL
  for (int i = 0; i < NW; ++i) {
    RC_UNROLL
    for (int j = 0; j <= MAXQ; ++j)
      if (i - j >= 0 && i - j <= NV && qw == (u32)j) w[i] |= u[i - j];
  }
}

constexpr int MAX_HEAD_SCHEMA = 95;   // the longest schema whose leaf prefix fits two blocks at any rank and position

// The head of leaf (rank, position, schema) in domain did: its prefix (everything before the value) as the chaining value
// after its first block (mid) and the q bytes after that (tail: 16 big-endian words, zero past q). Returns q, or -1.
template <class C = Sha>
RC_HD int leaf_head(const uint8_t* did, u64 rank, u64 position, const uint8_t* schema, int sl, u64* mid, u64* tail) {
  if (sl < 0 || sl > MAX_HEAD_SCHEMA) return -1;
  uint8_t m[256] = {};
  int n = frame_head(m, "leaf", did);
  n = frame_uint(m, n, rank);
  n = frame_uint(m, n, position);
  n = frame_part(m, n, schema, sl);
  be_put(m + n, 64, 8);
  midstate<C>(m, mid);
  for (int i = 0; i < 16; ++i) tail[i] = be64_at(m + 128 + 8 * i);
  return n + 8 - 128;
}
// The leaf of the 64-byte value v from its head (out may be v): one compression from mid when the tail and v fit one block
// (q <= 47: under an 18-byte schema, a rank and a position below 2^48), in registers; else two.
template <class C = Sha>
RC_HD void leaf_tail(const u64* mid, const u64* tail, u32 q, const u64* v, u64* out) {
  if (q <= 47) {
    u64 w[16], vm[9];
    RC_UNROLL
    for (int i = 0; i < 16; ++i) w[i] = i < 6 ? tail[i] : 0;
    RC_UNROLL
    for (int i = 0; i < 8; ++i) vm[i] = v[i];
    vm[8] = 0x8000000000000000ull;
    place<16, 9, 5>(w, vm, q);
    w[15] = 8 * (128 + 64 + (u64)q);
    RC_UNROLL
    for (int i = 0; i < 8; ++i) out[i] = mid[i];
    C::compress(out, w);
    return;
  }
  uint8_t m[256] = {};
  for (u32 k = 0; k < q; ++k) m[k] = (uint8_t)(tail[k >> 3] >> (56 - 8 * (k & 7)));
  for (int i = 0; i < 8; ++i) be_put(m + q + 8 * i, v[i], 8);
  m[q + 64] = 0x80;
  be_put(m + 248, 8 * (128 + 64 + (u64)q), 8);
  cpy8(out, mid);
  for (int b = 0; b < 2; ++b) {
    u64 w[16];
    for (int i = 0; i < 16; ++i) w[i] = be64_at(m + 128 * b + 8 * i);
    C::compress(out, w);
  }
}

// node(level, index, L, R) from mid (node_mid's at the level) alone: the 146 + n bytes after the first block (u64be(n)'s last
// two, index's n, then u64be(64) || L || u64be(64) || R) and the padding are two blocks, built in registers
template <class C = Sha>
RC_HD void node_at(const u64* mid, u64 index, const u64* L, const u64* R, u64* out) {
  const u32 n = (u32)nbytes(index);
  u64 w[32], v[19];
  RC_UNROLL
  for (int i = 0; i < 32; ++i) w[i] = 0;
  w[0] = (u64)n << 48;
  const u64 ix = n == 8 ? index : index << (64 - 8 * n);
  place<32, 1, 0>(w, &ix, 2);
  v[0] = v[9] = 64;
  RC_UNROLL
  for (int i = 0; i < 8; ++i) {
    v[1 + i] = L[i];
    v[10 + i] = R[i];
  }
  v[18] = 0x8000000000000000ull;
  place<32, 19, 1>(w, v, 2 + n);
  w[31] = 8 * (128 + 146 + (u64)n);
  cpy8(out, mid);
  C::compress(out, w);
  C::compress(out, w + 16);
}

// pad(rank) in domain did, its message built and hashed whole
template <class C = Sha>
RC_HD void pad_of(const uint8_t* did, u64 rank, u64* out) {
  uint8_t m[256] = {};
  Bytes s{m, 0};
  put_pad(s, did, rank);
  digest_padded<C>(m, s.n, out);
}

}  // namespace rc
