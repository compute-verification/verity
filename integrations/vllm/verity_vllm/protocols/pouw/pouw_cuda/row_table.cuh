// A serving step's row tables (`verity.ml.commitments.row_table`, `commit.row_tables.step_roots`) as functions that compile for
// the host (g++, `row_table_host.cpp`, the twin the CPU tests hold to the Python reference) and the device (nvcc,
// `row_table.cu`). One thread computes one segment digest, one row's leaf or one tree node. The hm96-sha512 leaf and the
// frame-v3-sha512 leaf, pad and node are the GPU's one row commitment (`row_commit.cuh`, `rc`, its forms with no plan, from
// the domain id the tables carry); the row's segment and outer digests and its salt are this file's, the digests through
// `ncp2.cuh`'s SHA-512.
//
// A row is its K s8 codes then its f32 scale's four bytes little-endian (`row_table.s8_row`). Its leaf, rank r of its table:
//   d_j  = SHA-512(S(j, s, m_j) || row[1024 j : 1024 j + m_j / 8])           `rowleaf.sha512_row_seg_digest`
//   x    = SHA-512(Q(n_bits) || d_0 || ... || d_(s-1))
//   y    = ChaCha20(key, nonce = engine step, blocks 3 g .. 3 g + 2)          g the row's index in the step (`step_salts`)
//   b, c = x ^ M_key y, SHA-512(salt_prefix || y)                              `hm96.SHA512.commit_string`
//   leaf = frame-v3-sha512 leaf(domain, r, "hm96-sha512/row-seg/v1", SHA-512(leaf_prefix || b || c))
// Each table's root is the frame-v3-sha512 tree over its leaves, and the step's root the tree over its tables' roots, each a
// "frame-v3-sha512/root/v1" leaf under the step's domain. Domain ids come from the host. Words are SHA-512's big-endian u64s.
#pragma once
#include "ncp2.cuh"
#include "row_commit.cuh"

namespace rowtab {

using ncp2::Sha512;
using ncp2::sha512_compress;

constexpr uint32_t SEG_BYTES = 1024;
// String literals, so device code reads them too: the row-seg tags (25 and 33 bytes with their NULs) and the two leaf
// schemas.
#define ROWTAB_SEG_TAG "verity/sha512-row-seg/v1"
#define ROWTAB_SEG_SEGMENT_TAG "verity/sha512-row-seg/v1/segment"
#define ROWTAB_ROW_SCHEMA "hm96-sha512/row-seg/v1"
#define ROWTAB_ROOTS_SCHEMA "frame-v3-sha512/root/v1"

// One table of the step, as the host lays it out (`row_tables_device._tabs`: 14 u64 words).
struct Tab {
  uint64_t codes;          // the (B, K) s8 codes, row-major
  uint64_t scales;         // the (B,) f32 scales
  uint32_t K, n, row0, s;  // codes per row, rows, its first row's index in the step, segments per row
  uint64_t seg0;           // its first segment's index in the step
  uint32_t W, tree0;       // its leaves padded to a power of two, and its place in the tree scratch
  uint64_t did[8];         // its domain id
};
static_assert(sizeof(Tab) == 112, "Tab is 14 words");

struct Args {
  uint32_t key[8];    // the run's row key, ChaCha20's key words
  uint32_t nonce[3];  // the engine step, 12 bytes little-endian
  uint32_t T;         // tables
  uint64_t did[8];    // the step's domain id
};
static_assert(sizeof(Args) == 112, "Args is 14 words");

HD uint64_t bswap64(uint64_t x) {
#ifdef __CUDA_ARCH__
  return ((uint64_t)__byte_perm((uint32_t)x, 0, 0x0123) << 32) | __byte_perm((uint32_t)(x >> 32), 0, 0x0123);
#else
  return __builtin_bswap64(x);
#endif
}

HD uint32_t bswap32(uint32_t x) { return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24); }

HD void iv(uint64_t h[8]) {
  Sha512 s;
  ncp2::sha512_init(s);
  for (int i = 0; i < 8; ++i) h[i] = s.h[i];
}

// ``nbytes`` big-endian bytes of ``v`` into the block ``w`` at byte ``pos`` (constant arguments fold to register writes).
HD void put_be(uint64_t w[16], int pos, uint64_t v, int nbytes) {
#pragma unroll
  for (int k = 0; k < nbytes; ++k) {
    const int p = pos + k;
    w[p >> 3] |= ((v >> (8 * (nbytes - 1 - k))) & 0xffu) << (56 - 8 * (p & 7));
  }
}

template <int N>
HD void put_str(uint64_t w[16], const char (&s)[N]) {
#pragma unroll
  for (int k = 0; k < N - 1; ++k) w[k >> 3] |= (uint64_t)(uint8_t)s[k] << (56 - 8 * (k & 7));
}

HD void zero(uint64_t w[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
}

// ---- sha512/row-seg/v1 ---------------------------------------------------------------------------------------------------

HD uint8_t row_byte(const int8_t* row, uint32_t K, uint32_t scale, uint32_t x) {
  return x < K ? (uint8_t)row[x] : (uint8_t)(scale >> (8 * (x - K)));
}

HD uint32_t segments(uint32_t K) { return (K + 4 + SEG_BYTES - 1) / SEG_BYTES; }

// d_j of the row (codes ``row``, scale bits ``scale``): S(j, s, m) = "verity/sha512-row-seg/v1/segment\0" || u64be j || u64be s ||
// u64be m, zero padded to a block, then the segment's bytes.
HD void seg_digest(const int8_t* row, uint32_t K, uint32_t scale, uint32_t j, uint32_t s, uint64_t out[8]) {
  const uint32_t start = SEG_BYTES * j, len = (K + 4 - start) < SEG_BYTES ? (K + 4 - start) : SEG_BYTES;
  uint64_t h[8], w[16];
  iv(h);
  zero(w);
  put_str(w, ROWTAB_SEG_SEGMENT_TAG);
  put_be(w, 33, j, 8);
  put_be(w, 41, s, 8);
  put_be(w, 49, 8ull * len, 8);
  sha512_compress(h, w);
  const bool aligned = K % 8 == 0 && (uintptr_t)row % 8 == 0;
  const uint32_t blocks = (len + 17 + 127) / 128;
  for (uint32_t q = 0; q < blocks; ++q) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t p = 128 * q + 8 * i;
      uint64_t v = 0;
      if (aligned && p + 8 <= len && start + p + 8 <= K) {
        v = bswap64(*(const uint64_t*)(row + start + p));
      } else {
        for (int k = 0; k < 8; ++k) {
          const uint32_t x = p + k;
          const uint64_t b = x < len ? row_byte(row, K, scale, start + x) : (x == len ? 0x80u : 0u);
          v |= b << (56 - 8 * k);
        }
      }
      w[i] = v;
    }
    if (q == blocks - 1) w[15] = 8ull * (128 + len);
    sha512_compress(h, w);
  }
  for (int i = 0; i < 8; ++i) out[i] = h[i];
}

// x = SHA-512(Q || d_0 .. d_(s-1)), the digests ``dig`` (8 words each); Q = "verity/sha512-row-seg/v1\0" || u64be n_bits ||
// u64be s || u32be 1024, zero padded.
HD void outer_digest(const uint64_t* dig, uint32_t K, uint32_t s, uint64_t out[8]) {
  uint64_t h[8], w[16];
  iv(h);
  zero(w);
  put_str(w, ROWTAB_SEG_TAG);
  put_be(w, 25, 8ull * (K + 4), 8);
  put_be(w, 33, s, 8);
  put_be(w, 41, SEG_BYTES, 4);
  sha512_compress(h, w);
  const uint64_t bits = 8ull * (128 + 64ull * s);
  for (uint32_t j = 0; j < s; j += 2) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = dig[8 * j + i];
    if (j + 1 < s) {
#pragma unroll
      for (int i = 0; i < 8; ++i) w[8 + i] = dig[8 * (j + 1) + i];
    } else {
      w[8] = 0x80ull << 56;
#pragma unroll
      for (int i = 9; i < 15; ++i) w[i] = 0;
      w[15] = bits;
    }
    sha512_compress(h, w);
  }
  if (s % 2 == 0) {
    zero(w);
    w[0] = 0x80ull << 56;
    w[15] = bits;
    sha512_compress(h, w);
  }
  for (int i = 0; i < 8; ++i) out[i] = h[i];
}

// ---- the salt: ChaCha20 (RFC 8439) ---------------------------------------------------------------------------------------

HD uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

#define ROWTAB_QR(a, b, c, d)                    \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16); \
  x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12); \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8);  \
  x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);

HD void chacha20_block(const uint32_t key[8], uint32_t counter, const uint32_t nonce[3], uint32_t out[16]) {
  uint32_t s[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u, key[0], key[1], key[2], key[3],
                    key[4],      key[5],      key[6],      key[7],      counter, nonce[0], nonce[1], nonce[2]};
  uint32_t x[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) x[i] = s[i];
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    ROWTAB_QR(0, 4, 8, 12) ROWTAB_QR(1, 5, 9, 13) ROWTAB_QR(2, 6, 10, 14) ROWTAB_QR(3, 7, 11, 15)
    ROWTAB_QR(0, 5, 10, 15) ROWTAB_QR(1, 6, 11, 12) ROWTAB_QR(2, 7, 8, 13) ROWTAB_QR(3, 4, 9, 14)
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) out[i] = x[i] + s[i];
}

// Row ``g``'s 192-byte salt, as 48 little-endian u32 words: the step's keystream bytes [192 g, 192 g + 192).
HD void salt(const Args& a, uint32_t g, uint32_t y[48]) {
#pragma unroll
  for (int c = 0; c < 3; ++c) chacha20_block(a.key, 3 * g + c, a.nonce, y + 16 * c);
}

// ---- hm96-sha512 ---------------------------------------------------------------------------------------------------------

// b || c of the inner digest ``x`` under the salt ``y``; ``hm`` the key's byte tables (`hm96.Instance.byte_tables`), 192 x 256
// entries of 8 words (rc's position table as words), and ``ms_salt`` the midstate after salt_prefix.
HD void commit_string(const uint64_t x[8], const uint32_t y[48], const uint64_t* hm, const uint64_t ms_salt[8], uint64_t bc[16]) {
  const rc::PosTab<false, true> T{reinterpret_cast<const uint8_t*>(hm)};
  uint32_t m[16];
  uint64_t my[8];
  rc::mask_positions(T, y, m);
  T.words(m, my);
  for (int i = 0; i < 8; ++i) bc[i] = x[i] ^ my[i];
  rc::salt_digest(ms_salt, rc::SaltU32{y, 1}, bc + 8);
}

// SHA-512(leaf_prefix || b || c), from ``ms_leaf``, the midstate after leaf_prefix.
HD void tree_leaf(const uint64_t bc[16], const uint64_t ms_leaf[8], uint64_t out[8]) {
  uint64_t B[16];
  for (int i = 0; i < 16; ++i) B[i] = bc[i];
  rc::hm96_value(ms_leaf, sx::kw_pad256(), B, out);
}

// ---- frame-v3-sha512 -----------------------------------------------------------------------------------------------------

// rc's forms with no plan, from the domain id as the tables carry it (8 big-endian words).
HD void did_bytes(const uint64_t did[8], uint8_t d[64]) {
  for (int i = 0; i < 8; ++i) rc::be_put(d + 8 * i, did[i], 8);
}

// CommitmentDomain.leaf(rank, rank, schema, value) for a 64-byte value: from its head (rc's leaf_head, leaf_tail).
template <int N>
HD void frame_leaf(const uint64_t did[8], uint64_t rank, const char (&schema)[N], const uint64_t value[8], uint64_t out[8]) {
  static_assert(N - 1 <= rc::MAX_HEAD_SCHEMA, "a schema whose leaf head fits two blocks");
  uint8_t d[64];
  uint64_t mid[8], tail[16];
  did_bytes(did, d);
  const int q = rc::leaf_head(d, rank, rank, reinterpret_cast<const uint8_t*>(schema), N - 1, mid, tail);
  rc::leaf_tail(mid, tail, (uint32_t)q, value, out);
}

// Its node at level (below 256): rc's node_at from the level's chaining value (node_mid).
HD void frame_node(const uint64_t did[8], uint64_t level, uint64_t index, const uint64_t* left, const uint64_t* right,
                   uint64_t out[8]) {
  uint8_t d[64];
  uint64_t mid[8];
  did_bytes(did, d);
  rc::node_mid(d, (int)level, mid);
  rc::node_at(mid, index, left, right, out);
}

HD void frame_pad(const uint64_t did[8], uint64_t rank, uint64_t out[8]) {
  uint8_t d[64];
  did_bytes(did, d);
  rc::pad_of(d, rank, out);
}

// ---- one row, one node ---------------------------------------------------------------------------------------------------

// Row ``g`` of the step (table ``t``, rank ``g - t.row0``): its frame leaf, from its segment digests ``dig`` (the table's
// ``s`` digests of that rank). ``ms`` holds the salt_prefix and leaf_prefix midstates.
HD void row_leaf(const Tab& t, const Args& a, uint32_t g, const uint64_t* dig, const uint64_t* hm, const uint64_t ms[16],
                 uint64_t out[8]) {
  uint64_t x[8], bc[16], tl[8];
  uint32_t y[48];
  outer_digest(dig, t.K, t.s, x);
  salt(a, g, y);
  commit_string(x, y, hm, ms, bc);
  tree_leaf(bc, ms + 8, tl);
  frame_leaf(t.did, g - t.row0, ROWTAB_ROW_SCHEMA, tl, out);
}

// Level ``l`` >= 1 of a tree of width ``W`` lives at ``W - (W >> (l - 1))`` in its scratch, ``W >> l`` nodes.
HD uint32_t level_at(uint32_t W, uint32_t l) { return W - (W >> (l - 1)); }

HD uint32_t width(uint32_t n) {
  uint32_t w = 1;
  while (w < n) w <<= 1;
  return w;
}

HD uint32_t depth(uint32_t W) {
  uint32_t d = 0;
  while ((1u << d) < W) ++d;
  return d;
}

// The table segment ``v`` (``seg``) or row ``v`` belongs to: the last whose first index is at most ``v``.
HD uint32_t find(const Tab* tabs, uint32_t T, uint64_t v, bool seg) {
  uint32_t lo = 0, hi = T;
  while (hi - lo > 1) {
    const uint32_t mid = (lo + hi) / 2;
    if ((seg ? tabs[mid].seg0 : (uint64_t)tabs[mid].row0) <= v) lo = mid; else hi = mid;
  }
  return lo;
}

// Segment ``i`` of the step: digest d_j of its table's rank r, row ``perm[row0 + r]`` of the table's codes.
HD void step_segment(const Tab* tabs, uint32_t T, const uint32_t* perm, uint64_t i, uint64_t out[8]) {
  const Tab& t = tabs[find(tabs, T, i, true)];
  const uint64_t local = i - t.seg0;
  const uint32_t r = (uint32_t)(local / t.s), j = (uint32_t)(local % t.s), b = perm[t.row0 + r];
  seg_digest((const int8_t*)t.codes + (size_t)b * t.K, t.K, ((const uint32_t*)t.scales)[b], j, t.s, out);
}

// Row ``g`` of the step: its frame leaf, from the step's segment digests ``dig``.
HD void step_row(const Tab* tabs, const Args& a, uint32_t g, const uint64_t* dig, const uint64_t* hm, const uint64_t ms[16],
                 uint64_t out[8]) {
  const Tab& t = tabs[find(tabs, a.T, g, false)];
  row_leaf(t, a, g, dig + 8 * (t.seg0 + (uint64_t)(g - t.row0) * t.s), hm, ms, out);
}

// The midstates after salt_prefix and after leaf_prefix (``prefixes``: the two blocks, 32 words).
HD void midstates(const uint64_t prefixes[32], uint64_t ms[16]) {
  for (int b = 0; b < 2; ++b) {
    uint64_t h[8], w[16];
    iv(h);
    for (int i = 0; i < 16; ++i) w[i] = prefixes[16 * b + i];
    sha512_compress(h, w);
    for (int i = 0; i < 8; ++i) ms[8 * b + i] = h[i];
  }
}

}  // namespace rowtab
