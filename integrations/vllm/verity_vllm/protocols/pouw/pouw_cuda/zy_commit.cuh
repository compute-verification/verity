// A call's z/y records committed for both tables that hold them, from one read of each record: the call's z table
// (`pouw_h1_tables_device`, `row_table.cuh`'s hm96-sha512/row-seg/v1 leaves under the call table's domain) and its registered
// value ``zy@<c>`` (`pouw_h1_registered_device`, `registered_rows.cuh`'s hm96-sha512/row/v2 leaves under the value's domain),
// as functions that compile for the host (g++, `zy_commit_host.cpp`, the twin the CPU tests hold to the two tables' own
// functions) and the device (nvcc, `zy_commit.cu`). One thread commits one record.
//
// A record is R bytes, R a multiple of 1024 (a z/y record is 6144). Its two row digests are different SHA-512 messages over
// the same bytes, and their blocks fall on the same 128 bytes of it, so each block is loaded once and compressed into both:
//   x_v2  = SHA-512(P || rec)                                   P the row/v2 prefix block (`rowleaf.sha512_row_v2_prefix`)
//   d_j   = SHA-512(S_j || rec[1024 j : 1024 (j + 1)])          S_j segment j's prefix block, j < s = R / 1024
//   x_seg = SHA-512(Q || d_0 || ... || d_(s-1))                 Q the row-seg/v1 prefix block
// The prefix blocks come from the host, as `verity.ml.commitments.rowleaf` builds them. Each table's leaf u is then its
// hm96-sha512 leaf of its x under its salt, ChaCha20 blocks 3u .. 3u + 2 of its key at its nonce (`rowtab::chacha20_block`),
// and that leaf's frame leaf at rank u: `row_commit.cuh`'s salt digest, mask and `finish_leaf`. The call table's tree is
// built here a level at a time (`row_commit.cuh`'s node and pad), the registered value's by `registered_rows.cuh`'s.
#pragma once
#include "registered_rows.cuh"
// ncp2.cuh (through registered_rows.cuh) and sha512.cuh (through row_commit.cuh) each define HD, ncp2's static inline on the
// host
#undef HD
#include "row_commit.cuh"

namespace zyc {

using rc::u32;
using rc::u64;

constexpr u32 SEG = 1024;
constexpr int MAX_SEGS = 8;
constexpr int CALL = 0, REG = 1;
// the two tables' schemas: row-seg/v1 is 22 bytes (SX 4), row/v2 18 (SX 0)
#define ZYC_SEG_SCHEMA "hm96-sha512/row-seg/v1"
#define ZYC_ROW_SCHEMA "hm96-sha512/row/v2"

// What a launch's two tables fix, as the host gives it (`pouw_h1_registered_device.ZyArgs`): each table's domain id, salt key
// and nonce, the hm96 key, the prefix blocks, and the records' count and bytes.
struct Args {
  uint8_t did[2][64];             // CALL's then REG's domain id
  uint8_t hm_key[256];            // the hm96 key
  uint8_t pre_v2[128], pre_seg_outer[128], pre_seg[MAX_SEGS][128];
  uint32_t key[2][8];             // each table's ChaCha20 key words
  uint64_t nonce[2];              // each table's u64 ChaCha20 nonce
  uint32_t n, R;                  // records and their bytes
};

// What `plan` builds from Args, once a launch, for every thread.
struct Plan {
  rc::FrameK leaf[2];
  rc::NodeK node;                 // CALL's nodes
  rc::Hm96K hm;
  u64 mid_v2[8], mid_outer[8], mid_seg[MAX_SEGS][8];
  u32 key[2][8];
  u32 nonce[2][3];                // 0, then the nonce's low and high words
  u32 n, R, s;
};

// The plan of ``a``: 0, or -1 for what the functions here can't commit (R not 1024 to 8 KiB in whole segments, no record,
// a salt counter past 32 bits).
RC_HD int plan(const Args& a, Plan& p) {
  if (a.n == 0 || a.R == 0 || a.R % SEG || a.R / SEG > (u32)MAX_SEGS || (u64)a.n > (1ull << 32) / 3) return -1;
  if (rc::frame_plan(a.did[CALL], (const uint8_t*)ZYC_SEG_SCHEMA, 22, p.leaf[CALL]) ||
      rc::frame_plan(a.did[REG], (const uint8_t*)ZYC_ROW_SCHEMA, 18, p.leaf[REG]) || rc::node_plan(a.did[CALL], p.node))
    return -1;
  rc::hm96_plan(a.hm_key, p.hm);
  p.n = a.n, p.R = a.R, p.s = a.R / SEG;
  rc::midstate(a.pre_v2, p.mid_v2);
  rc::midstate(a.pre_seg_outer, p.mid_outer);
  for (u32 j = 0; j < p.s; ++j) rc::midstate(a.pre_seg[j], p.mid_seg[j]);
  for (int t = 0; t < 2; ++t) {
    for (int i = 0; i < 8; ++i) p.key[t][i] = a.key[t][i];
    p.nonce[t][0] = 0, p.nonce[t][1] = (u32)a.nonce[t], p.nonce[t][2] = (u32)(a.nonce[t] >> 32);
  }
  return 0;
}

// The last block of a message of ``bytes`` bytes that ends on a block: 0x80, zeros, its bit length.
RC_HD void end_block(u64* w, u64 bytes) {
  w[0] = 0x8000000000000000ull;
  RC_UNROLL
  for (int i = 1; i < 15; ++i) w[i] = 0;
  w[15] = 8 * bytes;
}

// x_v2 and x_seg of the record at ``rec`` (16-byte aligned on the device).
template <class C = rc::Sha>
RC_HD void digests(const Plan& p, const uint8_t* rec, u64* x_v2, u64* x_seg) {
  u64 w[16], v[16], seg[8], held[8];
  rc::cpy8(x_v2, p.mid_v2);
  rc::cpy8(x_seg, p.mid_outer);
  RC_UNROLL1
  for (u32 j = 0; j < p.s; ++j) {
    rc::cpy8(seg, p.mid_seg[j]);
    RC_UNROLL1
    for (int q = 0; q < 8; ++q) {
      const uint8_t* b = rec + SEG * j + 128 * q;
      RC_UNROLL
      for (int i = 0; i < 8; ++i) {
        const rc::q4 u = rc::ldg16(b + 16 * i);
        w[2 * i] = v[2 * i] = sx::be64(u.x, u.y);
        w[2 * i + 1] = v[2 * i + 1] = sx::be64(u.z, u.w);
      }
      C::compress(x_v2, w);
      C::compress(seg, v);
    }
    end_block(v, 128 + SEG);
    C::compress(seg, v);
    if (j % 2 == 0) {
      rc::cpy8(held, seg);
    } else {
      rc::cpy8(w, held);
      rc::cpy8(w + 8, seg);
      C::compress(x_seg, w);
    }
  }
  if (p.s % 2) {
    rc::cpy8(w, held);
    w[8] = 0x8000000000000000ull;
    RC_UNROLL
    for (int i = 9; i < 15; ++i) w[i] = 0;
    w[15] = 8 * (128 + 64 * (u64)p.s);
  } else {
    end_block(w, 128 + 64 * (u64)p.s);
  }
  C::compress(x_seg, w);
  end_block(w, 128 + (u64)p.R);
  C::compress(x_v2, w);
}

// Table ``t``'s leaf u from its row digest ``x``: b || c into ``bc`` (16 words) and the frame leaf into ``out``. ``T`` the
// hm96 key's mask table (`rc::NibTab`, `rc::ByteTab`).
template <int SX, class Tab, class C = rc::Sha>
RC_HD void leaf(const Plan& p, int t, const Tab& T, u32 u, const u64* x, u64* bc, u64* out) {
  u32 y[48], mk[16];
  u64 Y[24], my[8], c[8];
  RC_UNROLL1
  for (u32 k = 0; k < 3; ++k) rowtab::chacha20_block(p.key[t], 3 * u + k, p.nonce[t], y + 16 * k);
  RC_UNROLL
  for (int i = 0; i < 24; ++i) Y[i] = sx::be64(y[2 * i], y[2 * i + 1]);
  rc::mask<false>(T, Y, mk);
  rc::mask_words(mk, my);
  rc::salt_digest<C>(p.hm, Y, c);
  RC_UNROLL
  for (int i = 0; i < 8; ++i) bc[i] = x[i] ^ my[i], bc[8 + i] = c[i];
  rc::frame_first_rt<C>(p.leaf[t], u, out);
  rc::finish_leaf<SX, true, C>(p.leaf[t], p.hm, u, x, my, c, out, nullptr, nullptr);
}

// Record u at ``rec``: the call table's leaf into ``call_leaf``, the registered value's b || c into ``reg_bc`` and its leaf
// into ``reg_leaf``.
template <class Tab, class C = rc::Sha>
RC_HD void record(const Plan& p, const Tab& T, u32 u, const uint8_t* rec, u64* call_leaf, u64* reg_bc, u64* reg_leaf) {
  u64 x_v2[8], x_seg[8], bc[16];
  digests<C>(p, rec, x_v2, x_seg);
  leaf<4, Tab, C>(p, CALL, T, u, x_seg, bc, call_leaf);
  leaf<0, Tab, C>(p, REG, T, u, x_v2, reg_bc, reg_leaf);
}

// The call table's tree in ``levels`` (level 0 its W leaves and pads, then each level above, the root last: 2 W - 1 nodes,
// `regrows::level_off`): entry r >= n of level 0, a pad, and node j of level l >= 1.
RC_HD void call_pad(const Plan& p, u32 r, u64* out) { rc::pad_rt(p.leaf[CALL], r, out); }

template <class C = rc::Sha>
RC_HD void call_node(const Plan& p, u32 W, u32 l, u32 j, u64* levels) {
  const u64* below = levels + 8 * regrows::level_off(W, l - 1);
  rc::node_rt<C>(p.node, (int)l - 1, j, below + 16 * (size_t)j, below + 16 * (size_t)j + 8,
                 levels + 8 * (regrows::level_off(W, l) + j));
}

}  // namespace zyc
