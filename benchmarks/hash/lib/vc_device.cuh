// libvc's device functions, header only. frame-v3-sha512's leaf, pad and node, the hm96-sha512 leaf's pieces (the salt
// digest, M_key y by the nibble or byte table, the leaf value and its frame leaf: `finish_leaf`) and the builders of their
// constants are the GPU's one row commitment (`rc`, verity/ml/kernels/row_commit.cuh, over verity/ml/kernels/sha512.cuh's
// SHA-512), named here over libvc's plan: LeafK (rc's FrameK and Hm96K with the row digest's and the salts' constants) and
// rc's NodeK. What stays libvc's is its schedule: sha512/row/v2's digest (`hash_tail` over bytes, `hash_words` over
// big-endian words), the salts, its loads and stores, and which form of each piece a kernel takes. vc.cu's kernels are
// built on these functions and its plans on these builders, so conformance.py's cases cover both; a kernel of another
// library (a fused pass whose constants live in its own __constant__ memory) includes this header, fills a LeafK and a node
// template with the builders, and calls the same functions.
//
// Positions and node indices are below 2^32 (MAX_NB bytes). Every digest is 8 big-endian words, as SHA-512's state holds it.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "vc.h"
#include "../../../verity/ml/kernels/row_commit.cuh"

namespace vc {

using u64 = uint64_t;
using u32 = uint32_t;
#define VC_DEV __device__ __forceinline__
#define VC_HD __host__ __device__ __forceinline__

using rc::LEVELS;
using rc::MAX_NB;
using rc::NodeK;
using rc::TN_BYTES;

// A domain's leaf constants: rc's FrameK (fl_tpl, fl_m15, pad_tpl, sx) and Hm96K (mid_salt, mid_leaf, kw_leafpad), and
// the row digest's and the salts'.
struct LeafK : rc::FrameK, rc::Hm96K {
  u64 mid_P[8];                      // after P(l) (row/v2)
  u64 kw_xpad[80];                   // row/v2 with l a multiple of 1024: its all-padding last block, scheduled
  u64 ks[8];                         // PRF: the salt key
  u64 ks_m8[8];                      // PRF: state after rounds 0..7 of a salt block (its words 0..7 are the key)
  u32 cc_key[8];                     // ChaCha20: the key
  u32 cc_w13, cc_w14, cc_w15, cc_ctr64;
  u32 lb;                            // row bytes (row/v2)
  u32 salt_src;
  u32 xpad;                          // kw_xpad applies (lb % 128 == 0)
};

using rc::cpy8;
using rc::ldg16;
using rc::nbytes;
using rc::st_be;
using sx::be64;
using sx::bswap32;

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

// --- the frame: leaf (its two blocks), pad, node: rc's -----------------------------------------------------------------

using rc::frame_first;
using rc::frame_first_rt;
using rc::frame_leaf;
using rc::frame_second;
using rc::frame_second_rt;
using rc::node;
using rc::pad;
using rc::pad_rt;

// rc's node with the index's length read at run time; COMPACT: through sha::compress_compact (rc::Compact), else
// sha::compress
template <bool COMPACT>
using NodeForm = std::conditional_t<COMPACT, rc::Compact, rc::Sha>;
template <bool COMPACT>
VC_DEV void node_rt_from(const u64 (*tpl)[32], const u64* mid, u64 index, const u64* L, const u64* R, u64* out) {
  rc::node_rt_from<NodeForm<COMPACT>>(tpl, mid, index, L, R, out);
}
template <bool COMPACT>
VC_DEV void node_rt(const NodeK& N, int level, u64 index, const u64* L, const u64* R, u64* out) {
  rc::node_rt<NodeForm<COMPACT>>(N, level, index, L, R, out);
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

// --- the hm96-sha512 leaf: rc's ---------------------------------------------------------------------------------------

using rc::mask_words;
using rc::salt_digest;

// M_key y (rc::mask) from the nibble table through L1, its loop over r unrolled (native_rows' hm96_mask)
VC_DEV void mask_nib(const uint8_t* T, const u64* Y, u32* m) { rc::mask<false>(rc::NibTab<true>{T}, Y, m); }
// from the byte table in shared memory, T8[17 b + c] chunk c of T_b, the loop rolled (sharaw's mask8; the stride of 17
// spreads the lanes' random b over the banks)
VC_DEV void mask8(const uint4* T8, const u64* Y, u32* m) {
  rc::mask<true>(rc::ByteTab<17 * 16, false>{reinterpret_cast<const uint8_t*>(T8)}, Y, m);
}
// from the nibble table in shared memory, the loop rolled as mask8's (sharaw's mask_roll)
VC_DEV void mask_nib_s(const uint8_t* T, const u64* Y, u32* m) { rc::mask<true>(rc::NibTab<false>{T}, Y, m); }

// h = SHA-512(leaf_prefix || (x ^ M y) || c), then the frame leaf from its first block's state f (rc::finish_leaf)
template <int SX, bool RT = false>
VC_DEV void finish_leaf(const LeafK& K, u64 pos, const u64* x, const u64* my, const u64* c, u64* f, uint8_t* bc, uint8_t* hs) {
  rc::finish_leaf<SX, RT>(K, K, pos, x, my, c, f, bc, hs);
}

// --- the constants, on the host or the device: rc's --------------------------------------------------------------------

using rc::byte_chunk;
using rc::leaf_m15;
using rc::midstate;
using rc::nib_entry;
using rc::node_mids;
using rc::node_tpls;
using rc::pad_schedule;
using rc::pad_tpl_nb;
using rc::pad_tpls;

// rc's leaf templates (fl_tpl[nb - 1]; or every nb's, sx and fl_m15) for the two hm96 schemas only, 18 and 22 bytes, which
// vc.cu's kernels take (SX 0 or 4). 0, or -1.
VC_HD int leaf_frame_nb(const uint8_t* did, const uint8_t* schema, int sl, int nb, LeafK& K) {
  return sl == 18 || sl == 22 ? rc::leaf_frame_nb(did, schema, sl, nb, K) : -1;
}
VC_HD int leaf_frame(const uint8_t* did, const uint8_t* schema, int sl, LeafK& K) {
  return sl == 18 || sl == 22 ? rc::leaf_frame(did, schema, sl, K) : -1;
}

}  // namespace vc
