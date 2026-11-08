// An interior's word rows as `verity.ml.windows.rows.commit_ports` commits them (``hm96-sha512/row/v1``), as functions that
// compile for the host (g++, `interior_rows_host.cpp`, the twin the CPU tests hold to `commit_ports`) and the device (nvcc,
// `interior_rows.cu`). One thread computes one row's leaf, one pad or one tree node; ChaCha20, the hm96 string, the tree leaf,
// the frame-v3-sha512 leaf, pad and node and a value's tree layout are `registered_rows.cuh`'s (the PoUW committer's, over the
// GPU's one row commitment, `row_commit.cuh`); only the row digest and the leaf schema are this file's.
//
// Row j of a port of W u16 words is its 2 W bytes, the words little-endian. Its leaf:
//   x    = SHA-512(sha512_row_prefix(ROLE_X, 16, W) || row)           one constant block, then the row
//   y    = ChaCha20(key, blocks 3 j .. 3 j + 2), state words 13..15 the port's index (`rows.salts`: 12 bytes LE)
//   b, c = x ^ M_key y, SHA-512(salt_prefix || y)
//   leaf = frame-v3-sha512 leaf(domain, j, "hm96-sha512/row/v1", SHA-512(leaf_prefix || b || c))
// and the port's tree is the frame-v3-sha512 tree over its leaves padded to a power of two, every level kept.
#pragma once
#include "registered_rows.cuh"

namespace introws {

using namespace regrows;

#define INTROWS_TAG "verity/sha512-row/v1"
#define INTROWS_SCHEMA "hm96-sha512/row/v1"

// A launch's piece of one port, as the host lays it out (`interior_gpu.Seg`: 22 u64 words). A leaf launch's piece is rows
// j0 .. j0 + count - 1 (t.rows the device bytes of row j0); a tree launch's is level j0's count nodes (level 0: the pads).
struct Seg {
  Table t;          // the port: n rows of row_bytes bytes, W leaves, its key, nonce words and domain id
  uint64_t bcs;     // the port's b || c, n x 16 words
  uint64_t levels;  // the port's tree, (2 W - 1) x 8 words, level 0 its leaves and pads
  uint64_t first;   // the piece's first thread over the launch
  uint32_t j0, count;
};
static_assert(sizeof(Seg) == 176, "Seg is 22 words");

// x = SHA-512(sha512_row_prefix(ROLE_X, 16, len / 2) || row) of the row's ``len`` bytes (`rowleaf.sha512_row_prefix`).
HD void row_digest_v1(const uint8_t* row, uint32_t len, uint64_t out[8]) {
  uint64_t w[16];
  iv(out);
  zero(w);
  put_str(w, INTROWS_TAG);
  put_be(w, 21, 1, 1);
  put_be(w, 22, 16, 1);
  put_be(w, 23, len / 2, 4);
  sha512_compress(out, w);
  const uint32_t blocks = (len + 17 + 127) / 128;
  for (uint32_t q = 0; q < blocks; ++q) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = row_word(row, len, 128 * q + 8 * i);
    if (q == blocks - 1) w[15] = 8ull * (128 + (uint64_t)len);
    sha512_compress(out, w);
  }
}

// The piece thread ``g`` falls in: the last whose first thread is at most ``g``.
HD uint32_t find_seg(const Seg* s, uint32_t S, uint64_t g) {
  uint32_t lo = 0, hi = S;
  while (hi - lo > 1) {
    const uint32_t mid = (lo + hi) / 2;
    if (s[mid].first <= g) lo = mid; else hi = mid;
  }
  return lo;
}

// Thread ``g`` of a leaf launch: its row's b || c and frame leaf, into its port's ``bcs`` and level 0.
HD void seg_leaf(const Seg* s, uint32_t S, uint64_t g, const uint64_t* hm, const uint64_t ms[16]) {
  const Seg& p = s[find_seg(s, S, g)];
  const uint32_t i = (uint32_t)(g - p.first), j = p.j0 + i;
  uint64_t x[8], bc[16], tl[8], lf[8];
  uint32_t y[48];
  row_digest_v1((const uint8_t*)p.t.rows + (size_t)p.t.row_bytes * i, p.t.row_bytes, x);
  row_salt(p.t, j, y);
  commit_string(x, y, hm, ms, bc);
  tree_leaf(bc, ms + 8, tl);
  frame_leaf(p.t.did, j, INTROWS_SCHEMA, tl, lf);
  uint64_t* bcs = (uint64_t*)p.bcs + 16 * (size_t)j;
  uint64_t* lv = (uint64_t*)p.levels + 8 * (size_t)j;
  for (int k = 0; k < 16; ++k) bcs[k] = bc[k];
  for (int k = 0; k < 8; ++k) lv[k] = lf[k];
}

// Thread ``g`` of a tree launch: a pad (level 0, rank n + its index) or node j of level j0 >= 1.
HD void seg_node(const Seg* s, uint32_t S, uint64_t g) {
  const Seg& p = s[find_seg(s, S, g)];
  const uint32_t j = (uint32_t)(g - p.first);
  uint64_t* levels = (uint64_t*)p.levels;
  if (p.j0 == 0) pad(p.t, p.t.n + j, levels + 8 * (size_t)(p.t.n + j));
  else node(p.t, p.j0, j, levels);
}

}  // namespace introws
