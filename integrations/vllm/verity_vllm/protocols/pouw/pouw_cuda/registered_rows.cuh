// H1's registered values (`pouw_h1_registered.commit`, core's `registered.Registered.commit` under ``hm96-sha512/row/v2``) as
// functions that compile for the host (g++, `registered_rows_host.cpp`, the twin the CPU tests hold to the Python reference) and
// the device (nvcc, `registered_rows.cu`). One thread computes one row's leaf, one pad or one tree node; SHA-512, ChaCha20, the
// hm96 commit string and the frame-v3-sha512 leaf, pad and node come through `row_table.cuh` (the last two are the GPU's one
// row commitment, `row_commit.cuh`).
//
// Row j of a value of W u16 words is its 2 W bytes as they sit in memory (the words little-endian). Its leaf:
//   x    = SHA-512(P(16 W) || row)                                   P = "verity/sha512-row/v2\0" || u64be n_bits, one block
//   y    = ChaCha20(key, nonce, blocks 3 j .. 3 j + 2), or the given  `audit.hm96_salt`: a 64-bit counter, a 64-bit nonce
//   b, c = x ^ M_key y, SHA-512(salt_prefix || y)                     `hm96.SHA512.commit_string`
//   leaf = frame-v3-sha512 leaf(domain, j, "hm96-sha512/row/v2", SHA-512(leaf_prefix || b || c))
// and the value's tree is the frame-v3-sha512 tree over its leaves padded to a power of two, every level kept. The domain id
// comes from the host (`registered.domain`).
#pragma once
#include "row_table.cuh"

namespace regrows {

using namespace rowtab;

#define REGROWS_V2_TAG "verity/sha512-row/v2"
#define REGROWS_SCHEMA "hm96-sha512/row/v2"

// One value as the host lays it out (`pouw_h1_registered_device.Table`: 18 u64 words).
struct Table {
  uint64_t rows;       // the rows' bytes, row j at rows + row_bytes * (j - j0) for the rows this launch holds
  uint64_t salts;      // the n given salts (192 bytes each), or 0: ChaCha20 under key at nonce
  uint32_t n, row_bytes;
  uint32_t W, depth;   // leaves padded to a power of two, and the tree's depth
  uint32_t key[8];     // the salt key's ChaCha20 words
  uint32_t nonce[4];   // the counter's high word (0), the nonce's low and high words, unused
  uint64_t did[8];     // the domain id
};
static_assert(sizeof(Table) == 144, "Table is 18 words");

// Eight bytes of the row at ``p`` as one big-endian word, the bytes past ``len`` SHA-512's padding (0x80, then zeros).
HD uint64_t row_word(const uint8_t* row, uint32_t len, uint32_t p) {
  if (p + 8 <= len) {
    const uintptr_t a = (uintptr_t)(row + p);
    if (a % 8 == 0) return bswap64(*(const uint64_t*)(row + p));
    if (a % 4 == 0)
      return ((uint64_t)bswap32(*(const uint32_t*)(row + p)) << 32) | bswap32(*(const uint32_t*)(row + p + 4));
  }
  uint64_t v = 0;
  for (int k = 0; k < 8; ++k) {
    const uint32_t x = p + k;
    const uint64_t b = x < len ? row[x] : (x == len ? 0x80u : 0u);
    v |= b << (56 - 8 * k);
  }
  return v;
}

// x = SHA-512(P(8 len) || row) of the row's ``len`` bytes (`rowleaf.sha512_row_v2_digest`).
HD void row_digest(const uint8_t* row, uint32_t len, uint64_t out[8]) {
  uint64_t w[16];
  iv(out);
  zero(w);
  put_str(w, REGROWS_V2_TAG);
  put_be(w, 21, 8ull * len, 8);
  sha512_compress(out, w);
  const uint32_t blocks = (len + 17 + 127) / 128;
  for (uint32_t q = 0; q < blocks; ++q) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = row_word(row, len, 128 * q + 8 * i);
    if (q == blocks - 1) w[15] = 8ull * (128 + (uint64_t)len);
    sha512_compress(out, w);
  }
}

// Row j's salt as 48 little-endian u32 words: the given one, or ChaCha20 blocks 3 j .. 3 j + 2 (j < 2^32 / 3).
HD void row_salt(const Table& t, uint32_t j, uint32_t y[48]) {
  if (t.salts) {
    const uint32_t* s = (const uint32_t*)t.salts + 48 * (size_t)j;
    for (int i = 0; i < 48; ++i) y[i] = s[i];
    return;
  }
#pragma unroll
  for (int c = 0; c < 3; ++c) chacha20_block(t.key, 3 * j + c, t.nonce, y + 16 * c);
}

// Row j's b || c into ``bc`` and its frame leaf into ``leaf``, from its bytes ``row``; ``ms`` the salt_prefix and leaf_prefix
// midstates, ``hm`` the key's byte tables.
HD void row_leaf(const Table& t, uint32_t j, const uint8_t* row, const uint64_t* hm, const uint64_t ms[16], uint64_t bc[16],
                 uint64_t leaf[8]) {
  uint64_t x[8], tl[8];
  uint32_t y[48];
  row_digest(row, t.row_bytes, x);
  row_salt(t, j, y);
  commit_string(x, y, hm, ms, bc);
  tree_leaf(bc, ms + 8, tl);
  frame_leaf(t.did, j, REGROWS_SCHEMA, tl, leaf);
}

// A public table's salt head as the host lays it out (`pouw_h1_registered_device.Pub`): `core_check.public_row_salt`'s
// domain, u64be context length and context, as big-endian words, the bytes past ``len`` zero.
constexpr int PUB_WORDS = 40;
struct Pub {
  uint64_t w[PUB_WORDS];
  uint32_t len, n;  // the head's bytes (at most 8 PUB_WORDS) and the salts a launch derives
  uint64_t first;   // the first salt's position
};
static_assert(sizeof(Pub) == 8 * PUB_WORDS + 16, "Pub is 42 words");

// Position ``pos``'s public salt, SHA-512(head || u64be pos || u8 i) for i = 0, 1, 2, into ``out`` as its 192 bytes.
HD void public_salt(const Pub& p, uint64_t pos, uint64_t out[24]) {
  for (int c = 0; c < 3; ++c) {
    uint8_t m[8 * PUB_WORDS + 64] = {};  // the head, pos and i, with room for their padding
    for (uint32_t i = 0; i < p.len; ++i) m[i] = (uint8_t)(p.w[i >> 3] >> (56 - 8 * (i & 7)));
    rc::be_put(m + p.len, pos, 8);
    m[p.len + 8] = (uint8_t)c;
    uint64_t h[8];
    rc::digest_padded(m, (int)p.len + 9, h);
    for (int i = 0; i < 8; ++i) out[8 * c + i] = bswap64(h[i]);
  }
}

// Level l of a tree of width W starts at node 2 W - 2 (W >> l) in its levels (level 0 the W leaves and pads, then each
// level above, the root last: 2 W - 1 nodes).
HD uint64_t level_off(uint32_t W, uint32_t l) { return 2ull * W - 2ull * (W >> l); }

// Entry r of level 0: row r's leaf (written by `row_leaf`), else the pad at r.
HD void pad(const Table& t, uint32_t r, uint64_t out[8]) { frame_pad(t.did, r, out); }

// Node j of level l >= 1 from level l - 1 (``levels`` the whole tree).
HD void node(const Table& t, uint32_t l, uint32_t j, uint64_t* levels) {
  const uint64_t* below = levels + 8 * level_off(t.W, l - 1);
  frame_node(t.did, l - 1, j, below + 16 * (size_t)j, below + 16 * (size_t)j + 8, levels + 8 * (level_off(t.W, l) + j));
}

}  // namespace regrows
