// libvc's device functions, header only: SHA-512's compression (commit_gpu's `sha::compress`, `compress_from`, `compress_kw`),
// sha512/row/v2's digest (`hash_tail` over bytes, `hash_words` over big-endian words), the hm96-sha512 leaf's pieces (the salt
// digest, M_key y by the nibble or byte table, the leaf value and its frame leaf: `finish_leaf`), and frame-v3-sha512's leaf,
// pad and node, each from a plan's constants (LeafK, NodeK). The builders at the end make those constants, and the nibble
// table, on the host or the device. vc.cu's kernels are built on these functions and its plans on these builders, so
// conformance.py's cases cover both; a kernel of another library (a fused pass whose constants live in its own __constant__
// memory) includes this header, fills a LeafK and a node template with the builders, and calls the same functions.
//
// Positions and node indices are below 2^32 (MAX_NB bytes). Every digest is 8 big-endian words, as SHA-512's state holds it.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "vc.h"
#include "../../pouw/commit_gpu/sha512.cuh"

namespace vc {

using u64 = uint64_t;
using u32 = uint32_t;
#define VC_DEV __device__ __forceinline__
#define VC_HD __host__ __device__ __forceinline__

constexpr int MAX_NB = 4;            // positions and node indices below 2^32
constexpr int LEVELS = 33;
constexpr int TN_BYTES = 8192;       // the nibble table

struct LeafK {
  u64 mid_P[8];                      // after P(l) (row/v2)
  u64 kw_xpad[80];                   // row/v2 with l a multiple of 1024: its all-padding last block, scheduled
  u64 mid_salt[8];                   // after the salt prefix
  u64 mid_leaf[8];                   // after leaf_prefix(key)
  u64 ks[8];                         // PRF: the salt key
  u64 ks_m8[8];                      // PRF: state after rounds 0..7 of a salt block (its words 0..7 are the key)
  u32 cc_key[8];                     // ChaCha20: the key
  u32 cc_w13, cc_w14, cc_w15, cc_ctr64;
  u64 fl_m15[8];                     // state after rounds 0..14 of the frame leaf's first block (words 0..14 don't depend on pos)
  u64 kw_leafpad[80];                // the hm96 leaf's padding block, scheduled
  u64 fl_tpl[MAX_NB][32];            // the frame leaf's two blocks per nb = len(uint(pos)) (index nb - 1), pos and value zero
  u64 pad_tpl[MAX_NB][16];           // a pad's first block per nb, rank zero
  u32 lb;                            // row bytes (row/v2)
  u32 salt_src;
  u32 sx;                            // schema length - 18
  u32 xpad;                          // kw_xpad applies (lb % 128 == 0)
};
struct NodeK {
  u64 node_mid[LEVELS][8];           // per level: the chaining value after a node message's first block (it fixes only the level)
  u64 node_tpl[MAX_NB][32];          // a node message's blocks 2 and 3 per ni = len(uint(index)), index and children zero
};

VC_DEV u32 bswap32(u32 x) { return __byte_perm(x, 0, 0x0123); }
VC_DEV u64 be64(u32 lo, u32 hi) { return (u64)bswap32(lo) << 32 | bswap32(hi); }   // 8 bytes loaded as two LE u32
VC_HD int nbytes(u64 x) { int n = 1; while (x >> (8 * n) && n < 8) ++n; return n; }
VC_DEV uint4 ldg16(const uint8_t* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

VC_HD void cpy8(u64* d, const u64* s) {
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = s[i];
}
// n words from / to digest byte order
VC_DEV void ld_be(const uint8_t* p, u64* d, int n) {
#pragma unroll
  for (int i = 0; i < n / 2; ++i) {
    const uint4 v = ldg16(p + 16 * i);
    d[2 * i] = be64(v.x, v.y);
    d[2 * i + 1] = be64(v.z, v.w);
  }
}
// ld_be with ordinary loads, for what this launch wrote before a barrier
VC_DEV void ld_be_written(const uint8_t* p, u64* d, int n) {
#pragma unroll
  for (int i = 0; i < n / 2; ++i) {
    const uint4 v = reinterpret_cast<const uint4*>(p)[i];
    d[2 * i] = be64(v.x, v.y);
    d[2 * i + 1] = be64(v.z, v.w);
  }
}
// ld_be through L2 only, for what another CTA of this launch wrote before a fence
VC_DEV void ld_be_cg(const uint8_t* p, u64* d, int n) {
#pragma unroll
  for (int i = 0; i < n / 2; ++i) {
    const uint4 v = __ldcg(reinterpret_cast<const uint4*>(p) + i);
    d[2 * i] = be64(v.x, v.y);
    d[2 * i + 1] = be64(v.z, v.w);
  }
}
VC_DEV void st_be(uint8_t* p, const u64* d, int n) {
  uint4* o = reinterpret_cast<uint4*>(p);
#pragma unroll
  for (int i = 0; i < n / 2; ++i)
    o[i] = make_uint4(bswap32((u32)(d[2 * i] >> 32)), bswap32((u32)d[2 * i]), bswap32((u32)(d[2 * i + 1] >> 32)), bswap32((u32)d[2 * i + 1]));
}
// native words (scratch): 16-byte vector loads and stores
VC_DEV void ld_w(const u64* p, u64* d, int n) {
#pragma unroll
  for (int i = 0; i < n / 2; ++i) {
    const ulonglong2 v = __ldg(reinterpret_cast<const ulonglong2*>(p) + i);
    d[2 * i] = v.x;
    d[2 * i + 1] = v.y;
  }
}
VC_DEV void st_w(u64* p, const u64* d, int n) {
#pragma unroll
  for (int i = 0; i < n / 2; ++i) reinterpret_cast<ulonglong2*>(p)[i] = make_ulonglong2(d[2 * i], d[2 * i + 1]);
}

// OR the low NB bytes of v, big-endian, into the message words W at byte offset OFF.
template <int OFF, int NB>
VC_DEV void or_be(u64* W, u64 v) {
  constexpr int END = OFF + NB, K = (END - 1) / 8, SH = 8 * (7 - (END - 1) % 8);
  W[K] |= v << SH;
  if constexpr (OFF < 8 * K) W[K - 1] |= v >> (64 - SH);
}
// OR a digest (8 words) into W at byte offset OFF.
template <int OFF>
VC_DEV void or_digest(u64* W, const u64* d) {
  constexpr int K0 = OFF / 8, S = 8 * (OFF % 8);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if constexpr (S == 0) {
      W[K0 + i] |= d[i];
    } else {
      W[K0 + i] |= d[i] >> S;
      W[K0 + i + 1] |= d[i] << (64 - S);
    }
  }
}

#define VC_BY_LEN(v, F, ...)          \
  switch (vc::nbytes(v)) {            \
    case 1: F<1>(__VA_ARGS__); break; \
    case 2: F<2>(__VA_ARGS__); break; \
    case 3: F<3>(__VA_ARGS__); break; \
    default: F<4>(__VA_ARGS__); break; \
  }

// --- the frame: leaf (its two blocks), pad, node ---------------------------------------------------------------------
// leaf message: frame (33) || u32be(4) || "leaf" || u64be(64) || did || u64be(nb) || pos || u64be(nb) || pos || u64be(S) ||
// schema || u64be(64) || value: pos at 121 and 129 + nb, the value at 145 + 2 nb + S, S = 18 + SX.
template <int NB>
VC_DEV void frame_first_nb(const LeafK& K, u64 pos, u64* out) {
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = K.fl_tpl[NB - 1][i];
  or_be<121, NB>(W, pos);
  sha::iv(out);
  sha::compress_from<15>(out, K.fl_m15, W);
}
VC_DEV void frame_first(const LeafK& K, u64 pos, u64* out) { VC_BY_LEN(pos, frame_first_nb, K, pos, out) }

template <int SX, int NB>
VC_DEV void frame_second_nb(const LeafK& K, u64 pos, const u64* v, u64* st) {
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = K.fl_tpl[NB - 1][16 + i];
  or_be<1 + NB, NB>(W, pos);
  or_digest<35 + 2 * NB + SX>(W, v);
  sha::compress(st, W);
}
template <int SX>
VC_DEV void frame_second(const LeafK& K, u64 pos, const u64* v, u64* st) {
  switch (nbytes(pos)) {
    case 1: frame_second_nb<SX, 1>(K, pos, v, st); break;
    case 2: frame_second_nb<SX, 2>(K, pos, v, st); break;
    case 3: frame_second_nb<SX, 3>(K, pos, v, st); break;
    default: frame_second_nb<SX, 4>(K, pos, v, st); break;
  }
}
template <int SX>
VC_DEV void frame_leaf(const LeafK& K, u64 pos, const u64* v, u64* out) {
  frame_first(K, pos, out);
  frame_second<SX>(K, pos, v, out);
}

template <int NB>
VC_DEV void pad_nb(const LeafK& K, u64 rank, u64* out) {
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = K.pad_tpl[NB - 1][i];
  or_be<120, NB>(W, rank);
  sha::iv(out);
  sha::compress(out, W);
#pragma unroll
  for (int i = 0; i < 15; ++i) W[i] = 0;
  W[15] = (120 + NB) * 8;
  sha::compress(out, W);
}
VC_DEV void pad(const LeafK& K, u64 rank, u64* out) { VC_BY_LEN(rank, pad_nb, K, rank, out) }

// The frame leaf's blocks and the pad with nb = len(uint(pos)) read at run time, one compression body each instead of one a
// length (k_leaf_spec, whose two kinds of warp run different code at once). For nb <= 4 pos lands in word 15 of the leaf's
// first block (or_be<121, nb>) and of the pad's (or_be<120, nb>), across words 0 and 1 of the leaf's second (or_be<1 + nb,
// nb>), and the value in words 4.. or 5.. of it at a shift that is never 0 (or_digest<35 + 2 nb + SX>).
VC_DEV void frame_first_rt(const LeafK& K, u64 pos, u64* out) {
  const int nb = min(nbytes(pos), MAX_NB);
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = K.fl_tpl[nb - 1][i];
  W[15] |= pos << (8 * (7 - nb));
  sha::iv(out);
  sha::compress_from<15>(out, K.fl_m15, W);
}
template <int K0>
VC_DEV void or_digest_s(u64* W, const u64* d, int S) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    W[K0 + i] |= d[i] >> S;
    W[K0 + i + 1] |= d[i] << (64 - S);
  }
}
template <int SX>
VC_DEV void frame_second_rt(const LeafK& K, u64 pos, const u64* v, u64* st) {
  const int nb = min(nbytes(pos), MAX_NB), off = 35 + 2 * nb + SX, sh = 8 * (15 - 2 * nb);
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = K.fl_tpl[nb - 1][16 + i];
  if (sh >= 64) {
    W[0] |= pos << (sh - 64);
  } else {
    W[0] |= pos >> (64 - sh);
    W[1] |= pos << sh;
  }
  if ((off >> 3) == 4) or_digest_s<4>(W, v, 8 * (off & 7));
  else or_digest_s<5>(W, v, 8 * (off & 7));
  sha::compress(st, W);
}
VC_DEV void pad_rt(const LeafK& K, u64 rank, u64* out) {
  const int nb = min(nbytes(rank), MAX_NB);
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = K.pad_tpl[nb - 1][i];
  W[15] |= rank << (8 * (8 - nb));
  sha::iv(out);
#pragma unroll 1
  for (int k = 0; k < 2; ++k) {
    sha::compress(out, W);
#pragma unroll
    for (int i = 0; i < 15; ++i) W[i] = 0;
    W[15] = (120 + nb) * 8;
  }
}

// node(level, index, L, R) from tpl (node_tpl) and mid (node_mid[level]), the index of NI bytes.
template <int NI>
VC_DEV void node_ni_from(const u64 (*tpl)[32], const u64* mid, u64 index, const u64* L, const u64* R, u64* out) {
  u64 W[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) W[i] = tpl[NI - 1][i];
  or_be<2, NI>(W, index);
  or_digest<10 + NI>(W, L);
  or_digest<82 + NI>(W, R);
  cpy8(out, mid);
  sha::compress(out, W);
  sha::compress(out, W + 16);
}
template <int NI>
VC_DEV void node_ni(const NodeK& N, int level, u64 index, const u64* L, const u64* R, u64* out) {
  node_ni_from<NI>(N.node_tpl, N.node_mid[level], index, L, R, out);
}
VC_DEV void node(const NodeK& N, int level, u64 index, const u64* L, const u64* R, u64* out) {
  VC_BY_LEN(index, node_ni, N, level, index, L, R, out)
}

// sharaw's compress_compact: one 16-round body for all 80 rounds (rounds 0..15 take the message word where later rounds take
// the schedule's), so a fold whose CTA has mostly idle warps waits on instruction fetch less.
VC_DEV void compress_compact(u64* st, const u64* m) {
  u64 w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = m[i];
  u64 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll 1
  for (int j = 0; j < 80; j += 16) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const u64 sch = sha::add3(sha::add(sha::s1(w[(i - 2) & 15]), w[(i - 7) & 15]), sha::s0(w[(i - 15) & 15]), w[i]);
      const u64 wi = j ? sch : w[i];
      w[i] = wi;
      const u64 t1 = sha::add3(h, sha::S1(e), sha::ch(e, f, g));
      const u64 t1b = sha::add3(t1, sha::SHA_K_TABLE[j + i], wi);
      const u64 mj = sha::maj(a, b, c);
      h = g; g = f; f = e; e = sha::add(d, t1b);
      d = c; c = b; b = a; a = sha::add3(t1b, sha::S0(b), mj);
    }
  }
  st[0] = sha::add(st[0], a); st[1] = sha::add(st[1], b); st[2] = sha::add(st[2], c); st[3] = sha::add(st[3], d);
  st[4] = sha::add(st[4], e); st[5] = sha::add(st[5], f); st[6] = sha::add(st[6], g); st[7] = sha::add(st[7], h);
}
// node_ni_from with ni read at run time (sharaw's node_rt): for ni <= 4 the index and both children land in the same words as
// or_be<2, ni> and or_digest<10 + ni>, <82 + ni> put them, and only the shifts depend on ni, so one compression body serves
// every length and both blocks. COMPACT: through compress_compact, else sha::compress.
template <bool COMPACT>
VC_DEV void node_rt_from(const u64 (*tpl)[32], const u64* mid, u64 index, const u64* L, const u64* R, u64* out) {
  const int ni = min(nbytes(index), MAX_NB), S = 8 * (2 + ni);
  u64 W[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) W[i] = tpl[ni - 1][i];
  W[0] |= index << (8 * (6 - ni));
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    W[1 + i] |= L[i] >> S;
    W[2 + i] |= L[i] << (64 - S);
    W[10 + i] |= R[i] >> S;
    W[11 + i] |= R[i] << (64 - S);
  }
  cpy8(out, mid);
#pragma unroll 1
  for (int k = 0; k < 2; ++k) {
    u64 w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = k ? W[16 + i] : W[i];
    if constexpr (COMPACT) compress_compact(out, w);
    else sha::compress(out, w);
  }
}
template <bool COMPACT>
VC_DEV void node_rt(const NodeK& N, int level, u64 index, const u64* L, const u64* R, u64* out) {
  node_rt_from<COMPACT>(N.node_tpl, N.node_mid[level], index, L, R, out);
}

// --- hashing bytes -------------------------------------------------------------------------------------------------

// SHA-512 from chaining value cv after `pre` bytes (whole blocks) over lb bytes at p (16-byte aligned; the loads may run up to
// 15 bytes past p + lb). kw: when lb % 128 == 0, the scheduled all-padding last block for pre + lb bytes, or null.
VC_DEV void hash_tail(const u64* cv, u64 pre, const uint8_t* p, u32 lb, u64* x, const u64* kw) {
  cpy8(x, cv);
  const u32 full = lb / 128;
#pragma unroll 1
  for (u32 b = 0; b < full; ++b) {
    u64 W[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 v = ldg16(p + 128 * b + 16 * c);
      W[2 * c] = be64(v.x, v.y);
      W[2 * c + 1] = be64(v.z, v.w);
    }
    sha::compress(x, W);
  }
  const u32 rem = lb - 128 * full;
  if (rem == 0 && kw) {
    sha::compress_kw(x, kw);
    return;
  }
  u64 W[16];
  const uint8_t* q = p + 128 * full;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const u32 off = 16 * c;
    const uint4 v = off < rem ? ldg16(q + off) : make_uint4(0, 0, 0, 0);
    W[2 * c] = be64(v.x, v.y);
    W[2 * c + 1] = be64(v.z, v.w);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const u32 base = 8 * i;
    if (base + 8 > rem) {
      if (base <= rem) {
        const u32 j = rem - base;
        W[i] = (j ? W[i] & (~0ull << (64 - 8 * j)) : 0) | (0x80ull << (56 - 8 * j));
      } else {
        W[i] = 0;
      }
    }
  }
  const u64 bits = (pre + lb) * 8;
  if (rem + 17 <= 128) {
    W[15] = bits;
    sha::compress(x, W);
  } else {
    sha::compress(x, W);
#pragma unroll
    for (int i = 0; i < 15; ++i) W[i] = 0;
    W[15] = bits;
    sha::compress(x, W);
  }
}

// hash_tail over n words at p (a message of 8 n bytes, its words as SHA-512 reads them; 16-byte aligned, read through the
// non-coherent cache), e.g. a row/v2 digest over 64-byte digests kept as words. kw: when n % 16 == 0, the scheduled
// all-padding last block for pre + 8 n bytes, or null.
VC_DEV void hash_words(const u64* cv, u64 pre, const u64* p, u32 n, u64* x, const u64* kw) {
  cpy8(x, cv);
  const u32 full = n / 16;
#pragma unroll 1
  for (u32 b = 0; b < full; ++b) {
    u64 W[16];
    ld_w(p + 16 * b, W, 16);
    sha::compress(x, W);
  }
  const u32 rem = n - 16 * full;
  if (rem == 0 && kw) {
    sha::compress_kw(x, kw);
    return;
  }
  u64 W[16];
  const u64* q = p + 16 * full;
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = (u32)i < rem ? __ldg(q + i) : (u32)i == rem ? 0x8000000000000000ull : 0;
  const u64 bits = (pre + 8ull * n) * 8;
  if (rem <= 13) {
    W[15] = bits;
    sha::compress(x, W);
  } else {
    sha::compress(x, W);
#pragma unroll
    for (int i = 0; i < 15; ++i) W[i] = 0;
    W[15] = bits;
    sha::compress(x, W);
  }
}

// --- salts ---------------------------------------------------------------------------------------------------------

VC_DEV void salt_prf(const LeafK& K, u64 pos, u64* Y) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    u64 w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = K.ks[i];
    w[8] = pos; w[9] = r; w[10] = 0x8000000000000000ull;
    w[11] = w[12] = w[13] = w[14] = 0;
    w[15] = 80 * 8;
    sha::iv(Y + 8 * r);
    sha::compress_from<8>(Y + 8 * r, K.ks_m8, w);
  }
}

VC_DEV void chacha_qr(u32& a, u32& b, u32& c, u32& d) {
  a += b; d ^= a; d = __funnelshift_l(d, d, 16);
  c += d; b ^= c; b = __funnelshift_l(b, b, 12);
  a += b; d ^= a; d = __funnelshift_l(d, d, 8);
  c += d; b ^= c; b = __funnelshift_l(b, b, 7);
}
// ChaCha20 chunk `ctr` as 8 big-endian u64 (its 64 bytes in order)
VC_DEV void chacha_block(const LeafK& K, u64 ctr, u64* out) {
  u32 s[16] = {0x61707865u, 0x3320646Eu, 0x79622D32u, 0x6B206574u, K.cc_key[0], K.cc_key[1], K.cc_key[2], K.cc_key[3],
               K.cc_key[4], K.cc_key[5], K.cc_key[6], K.cc_key[7], (u32)ctr, K.cc_ctr64 ? (u32)(ctr >> 32) : K.cc_w13,
               K.cc_w14, K.cc_w15};
  u32 x[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) x[i] = s[i];
#pragma unroll 1
  for (int r = 0; r < 10; ++r) {
    chacha_qr(x[0], x[4], x[8], x[12]); chacha_qr(x[1], x[5], x[9], x[13]);
    chacha_qr(x[2], x[6], x[10], x[14]); chacha_qr(x[3], x[7], x[11], x[15]);
    chacha_qr(x[0], x[5], x[10], x[15]); chacha_qr(x[1], x[6], x[11], x[12]);
    chacha_qr(x[2], x[7], x[8], x[13]); chacha_qr(x[3], x[4], x[9], x[14]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = be64(x[2 * i] + s[2 * i], x[2 * i + 1] + s[2 * i + 1]);
}

VC_DEV void salt_of(const LeafK& K, const uint8_t* salts, u64 i, u64 pos, u64* Y) {
  if (K.salt_src == VC_SALT_PRF) {
    salt_prf(K, pos, Y);
  } else if (K.salt_src == VC_SALT_GIVEN) {
    ld_be(salts + 192 * i, Y, 24);
  } else {
#pragma unroll 1
    for (int r = 0; r < 3; ++r) chacha_block(K, 3 * pos + r, Y + 8 * r);
  }
}

// c = SHA-512(salt_prefix || y), y as 24 big-endian words
VC_DEV void salt_digest(const LeafK& K, const u64* Y, u64* c) {
  u64 w[16];
  cpy8(c, K.mid_salt);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = Y[i];
  sha::compress(c, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = Y[16 + i];
  w[8] = 0x8000000000000000ull;
#pragma unroll
  for (int i = 9; i < 15; ++i) w[i] = 0;
  w[15] = (128 + 192) * 8;
  sha::compress(c, w);
}

// M_key y as 16 little-endian u32 (mask byte 4w + k is byte k of m[w]) from the nibble table through L1: T_v = T_(v & 15) ^
// T_(v & 0xF0), chunk c of row u at 256 c + 16 u, the high nibble's rows 4 KB on (native_rows' hm96_mask).
VC_DEV void mask_nib(const uint8_t* T, const u64* Y, u32* m) {
#pragma unroll
  for (int w = 0; w < 16; ++w) m[w] = 0;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    u32 A[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) A[i] = 0;
#pragma unroll
    for (int q = 0; q < 12; ++q) {
      const int p = 16 * q + r;
      const u32 v = (u32)(Y[p >> 3] >> (56 - 8 * (p & 7))) & 0xFFu;
      const uint8_t* lo = T + 256 * q + 16 * (v & 15u);
      const uint8_t* hi = T + 4096 + 256 * q + 16 * (v >> 4);
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const uint4 u = ldg16(lo + 256 * i), w = ldg16(hi + 256 * i);
        A[4 * i] ^= u.x ^ w.x; A[4 * i + 1] ^= u.y ^ w.y; A[4 * i + 2] ^= u.z ^ w.z; A[4 * i + 3] ^= u.w ^ w.w;
      }
    }
    const int sw = r >> 2, s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= s ? __funnelshift_r(A[w + sw], A[w + sw + 1], s) : A[w + sw];
  }
}

// M_key y from the byte table in shared memory: T8[17 b + c] is chunk c of T_b (sharaw's mask8; the stride of 17 spreads the
// lanes' random b over the banks).
VC_DEV void mask8(const uint4* T8, const u64* Y, u32* m) {
#pragma unroll
  for (int w = 0; w < 16; ++w) m[w] = 0;
#pragma unroll 1
  for (int r = 0; r < 16; ++r) {
    u32 A[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) A[i] = 0;
    const int sh = 56 - 8 * (r & 7);
#pragma unroll
    for (int q = 0; q < 12; ++q) {
      const u64 wq = r < 8 ? Y[2 * q] : Y[2 * q + 1];
      const uint4* p = T8 + 17 * ((u32)(wq >> sh) & 0xFFu) + q;
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const uint4 u = p[i];
        A[4 * i] ^= u.x; A[4 * i + 1] ^= u.y; A[4 * i + 2] ^= u.z; A[4 * i + 3] ^= u.w;
      }
    }
    if (r & 4) {
#pragma unroll
      for (int k = 0; k < 19; ++k) A[k] = A[k + 1];
    }
    if (r & 8) {
#pragma unroll
      for (int k = 0; k < 18; ++k) A[k] = A[k + 2];
    }
    const u32 s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= __funnelshift_r(A[w], A[w + 1], s);
  }
}
// mask_nib from the nibble table in shared memory, its loop over r rolled as mask8's (sharaw's mask_roll).
VC_DEV void mask_nib_s(const uint8_t* T, const u64* Y, u32* m) {
#pragma unroll
  for (int w = 0; w < 16; ++w) m[w] = 0;
#pragma unroll 1
  for (int r = 0; r < 16; ++r) {
    u32 A[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) A[i] = 0;
    const int sh = 56 - 8 * (r & 7);
#pragma unroll
    for (int q = 0; q < 12; ++q) {
      const u32 v = (u32)((r < 8 ? Y[2 * q] : Y[2 * q + 1]) >> sh) & 0xFFu;
      const uint4* lo = reinterpret_cast<const uint4*>(T + 256 * q + 16 * (v & 15u));
      const uint4* hi = reinterpret_cast<const uint4*>(T + 4096 + 256 * q + 16 * (v >> 4));
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const uint4 u = lo[16 * i], w = hi[16 * i];
        A[4 * i] ^= u.x ^ w.x; A[4 * i + 1] ^= u.y ^ w.y; A[4 * i + 2] ^= u.z ^ w.z; A[4 * i + 3] ^= u.w ^ w.w;
      }
    }
    if (r & 4) {
#pragma unroll
      for (int k = 0; k < 19; ++k) A[k] = A[k + 1];
    }
    if (r & 8) {
#pragma unroll
      for (int k = 0; k < 18; ++k) A[k] = A[k + 2];
    }
    const u32 s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= __funnelshift_r(A[w], A[w + 1], s);
  }
}
VC_DEV void mask_words(const u32* m, u64* my) {
#pragma unroll
  for (int i = 0; i < 8; ++i) my[i] = (u64)bswap32(m[2 * i]) << 32 | bswap32(m[2 * i + 1]);
}

// h = SHA-512(leaf_prefix || (x ^ M y) || c), then the frame leaf from its first block's state f
template <int SX, bool RT = false>
VC_DEV void finish_leaf(const LeafK& K, u64 pos, const u64* x, const u64* my, const u64* c, u64* f, uint8_t* bc, uint8_t* hs) {
  u64 B[16], h[8];
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    B[w] = x[w] ^ my[w];
    B[8 + w] = c[w];
  }
  if (bc) st_be(bc, B, 16);
  cpy8(h, K.mid_leaf);
  sha::compress(h, B);
  sha::compress_kw(h, K.kw_leafpad);
  if (hs) st_be(hs, h, 8);
  if constexpr (RT) frame_second_rt<SX>(K, pos, h, f);
  else frame_second<SX>(K, pos, h, f);
}

// --- the constants, on the host or the device ---------------------------------------------------------------------------

VC_HD u64 be64_at(const uint8_t* p) {
  u64 v = 0;
  for (int i = 0; i < 8; ++i) v = v << 8 | p[i];
  return v;
}
VC_HD void be_put(uint8_t* p, u64 v, int n) {
  for (int i = n - 1; i >= 0; --i, v >>= 8) p[i] = (uint8_t)v;
}
VC_HD void midstate(const uint8_t* block, u64* st) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) w[i] = be64_at(block + 8 * i);
  sha::iv(st);
  sha::compress(st, w);
}
// kw[i] = K[i] + W[i] over the fixed block's whole schedule (sha::schedule_kw), for compress_kw
VC_HD void schedule_kw(const u64* block, u64* kw) {
  u64 w[80];
  for (int i = 0; i < 16; ++i) w[i] = block[i];
  for (int i = 16; i < 80; ++i) w[i] = sha::s1(w[i - 2]) + w[i - 7] + sha::s0(w[i - 15]) + w[i - 16];
  for (int i = 0; i < 80; ++i) kw[i] = sha::SHA_K_TABLE[i] + w[i];
}
// the all-padding last block of a message of `bytes` bytes (a multiple of 128), scheduled
VC_HD void pad_schedule(u64 bytes, u64* kw) {
  u64 blk[16] = {0x8000000000000000ull};
  blk[15] = bytes * 8;
  schedule_kw(blk, kw);
}
// the state after rounds 0..r-1 of block from chaining value cv (sha::rounds_prefix): what compress_from<r> starts from
VC_HD void rounds_prefix(const u64* cv, const u64* block, int r, u64* m) {
  u64 a = cv[0], b = cv[1], c = cv[2], d = cv[3], e = cv[4], f = cv[5], g = cv[6], h = cv[7];
  for (int i = 0; i < r; ++i) {
    const u64 t1 = h + sha::S1(e) + sha::ch(e, f, g) + sha::SHA_K_TABLE[i] + block[i];
    const u64 t2 = sha::S0(a) + sha::maj(a, b, c);
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  m[0] = a; m[1] = b; m[2] = c; m[3] = d; m[4] = e; m[5] = f; m[6] = g; m[7] = h;
}

// frame-v3's message (merkle._message) into m, zero past it: FRAME || u32be(len tag) || tag || u64be(64) || did, then
// frame_part's parts; each returns the length so far.
VC_HD int frame_head(uint8_t* m, const char* tag, const uint8_t* did) {
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
VC_HD int frame_part(uint8_t* m, int n, const uint8_t* b, int len) {
  be_put(m + n, (u64)len, 8);
  n += 8;
  for (int i = 0; i < len; ++i) m[n++] = b ? b[i] : 0;
  return n;
}
// the n-byte message in m (zero past n) padded, as words into w; the words
VC_HD int padded_words(uint8_t* m, int n, u64* w) {
  const int total = (n + 17 + 127) / 128 * 128;
  m[n] = 0x80;
  be_put(m + total - 8, (u64)n * 8, 8);
  for (int i = 0; i < total / 8; ++i) w[i] = be64_at(m + 8 * i);
  return total / 8;
}

// LeafK's fl_tpl[nb - 1] for len(uint(pos)) = nb, domain id did and schema (18 or 22 bytes): the frame leaf's two blocks
// with pos and the value zero. 0, or -1.
VC_HD int leaf_frame_nb(const uint8_t* did, const uint8_t* schema, int sl, int nb, LeafK& K) {
  if ((sl != 18 && sl != 22) || nb < 1 || nb > MAX_NB) return -1;
  uint8_t m[256] = {};
  int n = frame_head(m, "leaf", did);
  n = frame_part(m, n, nullptr, nb);
  n = frame_part(m, n, nullptr, nb);
  n = frame_part(m, n, schema, sl);
  n = frame_part(m, n, nullptr, 64);
  if (n + 17 > 256 || padded_words(m, n, K.fl_tpl[nb - 1]) != 32) return -1;
  K.sx = (u32)(sl - 18);
  return 0;
}
// fl_m15 from fl_tpl[0]: rounds 0..14 of the frame leaf's first block, whose words 0..14 are every nb's
VC_HD void leaf_m15(LeafK& K) {
  u64 st[8];
  sha::iv(st);
  rounds_prefix(st, K.fl_tpl[0], 15, K.fl_m15);
}
// fl_tpl for every nb, sx and fl_m15. 0, or -1.
VC_HD int leaf_frame(const uint8_t* did, const uint8_t* schema, int sl, LeafK& K) {
  for (int nb = 1; nb <= MAX_NB; ++nb) {
    if (leaf_frame_nb(did, schema, sl, nb, K)) return -1;
    for (int i = 0; i < 15; ++i)
      if (K.fl_tpl[nb - 1][i] != K.fl_tpl[0][i]) return -1;
  }
  leaf_m15(K);
  return 0;
}
// LeafK's pad_tpl[nb - 1] for domain id did: a pad's first block with rank zero
VC_HD void pad_tpl_nb(const uint8_t* did, int nb, LeafK& K) {
  uint8_t m[256] = {};
  u64 w[32];
  padded_words(m, frame_part(m, frame_head(m, "pad", did), nullptr, nb), w);
  for (int i = 0; i < 16; ++i) K.pad_tpl[nb - 1][i] = w[i];
}
VC_HD void pad_tpls(const uint8_t* did, LeafK& K) {
  for (int nb = 1; nb <= MAX_NB; ++nb) pad_tpl_nb(did, nb, K);
}
// node_tpl: a node message's blocks 2 and 3 per ni = len(uint(index)), index and children zero; no domain or level reaches
// them. 0, or -1.
VC_HD int node_tpls(u64 (*tpl)[32]) {
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
VC_HD void node_mids(const uint8_t* did, u64 (*mid)[8], int levels) {
  for (int l = 0; l < levels; ++l) {
    uint8_t m[384] = {};
    const uint8_t lv = (uint8_t)l;
    frame_part(m, frame_head(m, "node", did), &lv, 1);
    midstate(m, mid[l]);
  }
}
// entry t < 512 of key's nibble table (mask_nib's T, TN_BYTES): T_v = xor of (key >> s) over the set bits s of v, for v a low
// nibble u (h = 0) or a high one 16 u (h = 1), its chunk c (16 bytes) at 4096 h + 256 c + 16 u of tab
VC_HD void nib_entry(const uint8_t* key, u32 t, uint8_t* tab) {
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

}  // namespace vc
