// The serve-time commit call's device code (slots.py is the reference, README.md the specification), as host+device
// functions: nvcc builds commit_slots.cu (the kernels), g++ builds commit_slots_host.cpp (the host twin, one loop a kernel),
// and both run the same functions below. SHA-512 is commit_gpu's (benchmarks/pouw/commit_gpu/sha512.cuh). Every digest is
// 8 u64 words, each the big-endian value of its 8 bytes.
//
// A step is committed group by group (a group: e.g. a layer's ops). Per group, a thread per (row, slot) computes a row's
// leaf (row_thread), a thread per (unit, slot) folds a unit's leaves to its root and that root's leaf in the slot tree
// (unit_thread), and a thread per slot pushes those leaves onto the slot tree's frontier (fold_thread). finish_thread pads a
// slot's tree to its root and makes that root's leaf in the step tree; the step tree is then built a level at a time.
// Pads of the unit and slot trees don't depend on the slot, so the host computes each tree's pad subtrees once (cs_tail).
#pragma once
#include <cstdint>
#include <cstring>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __forceinline__ inline
#define __constant__
struct uint4 {
  uint32_t x, y, z, w;
};
#endif

#include "../../pouw/commit_gpu/sha512.cuh"

namespace cs {

using u64 = uint64_t;
using u32 = uint32_t;

constexpr int MAX_SRC = 24;     // tensors a group's add takes
constexpr int SLOT_LV = 17;     // a slot tree of at most 2^16 units: its frontier's levels 0..16
constexpr int UNIT_LV = 7;      // a unit of at most 64 rows
constexpr int TSTRIDE = 272;    // bytes a T row in shared memory: 256 + 16, so rows start in different bank groups

// --- layout and arguments, as the host lays them out (slots_gpu.py mirrors them) -----------------------------------

struct RowDesc {      // 128 bytes
  u32 unit;           // its unit, in the group
  u32 rank;           // its rank in the unit
  u32 g;              // its index among the group's rows (the slot's row g is the group's row_base + g)
  u32 nbytes;
  u32 a_src, a_off, a_len;   // its first part: a_len bytes at a_off of a slot's row of tensor a_src
  u32 b_src, b_off, b_len;   // its second part, or b_len = 0
  u32 pad_[6];
  u64 midP[8];        // the chaining value after P(8 nbytes), sha512/row/v2's first block
};
static_assert(sizeof(RowDesc) == 128, "RowDesc is 128 bytes");

struct UnitDesc {     // 16 bytes
  u32 row0;           // its first row among the group's rows
  u32 rows;
  u32 depth;          // its tree's depth
  u32 pad_;
};
static_assert(sizeof(UnitDesc) == 16, "UnitDesc is 16 bytes");

struct Ctx {          // what a run's context and layout fix
  const uint8_t* T;          // hm96's Hankel table: T_v, 256 rows of 256 bytes
  const u64* unit_did;       // every unit's domain id
  const u32* unit_tail;      // every unit's pad tail: (offset, count) into tail_lv / tail_node
  const u32* tail_lv;        // a tail entry's level
  const u64* tail_node;      // a tail entry's node
  u64 slot_did[8];
  u64 ms_salt[8];            // the chaining value after hm96's salt prefix
  u64 ms_leaf[8];            // after its leaf prefix (the key's)
  u32 U;                     // units a slot
  u32 slot_depth;
  u32 slot_tail_off, slot_tail_n;
};

struct Step {         // one step's
  u32 key[8];                // the salt key, ChaCha20's key words
  u64 step_did[8];
  u64* front;                // S x SLOT_LV x 8: the slot trees' frontiers
  u64* leaves;               // a group's row leaves: (row g, slot) x 8
  u64* rl;                   // a group's unit roots' leaves in the slot tree: (unit, slot) x 8
  u64* slot_root;            // S x 8
  u64* lv_a;                 // the step tree's levels, W x 8 each, ping-pong
  u64* lv_b;
  u64* root;                 // 8
  u32 S, W, depth, pad_;
};

struct Grp {          // one group's add
  const RowDesc* rows;       // its rows, longest first (each names its own place)
  const UnitDesc* units;
  const uint8_t* src[MAX_SRC];
  u64 stride[MAX_SRC];       // a tensor's bytes from one slot's row to the next
  u32 nrows, nunits, unit_base, row_base;
};

// --- portability -----------------------------------------------------------------------------------------------------

HD u32 bswap32(u32 x) {
#ifdef __CUDA_ARCH__
  return __byte_perm(x, 0, 0x0123);
#else
  return __builtin_bswap32(x);
#endif
}
HD u64 be64(u32 lo, u32 hi) { return (u64)bswap32(lo) << 32 | bswap32(hi); }   // 8 bytes loaded as two LE u32
HD uint4 ld16(const uint8_t* p) {
#ifdef __CUDA_ARCH__
  return __ldg(reinterpret_cast<const uint4*>(p));
#else
  uint4 v;
  memcpy(&v, p, 16);
  return v;
#endif
}
HD uint4 ld16_t(const uint8_t* p) {   // the T table, in shared memory on the device
#ifdef __CUDA_ARCH__
  return *reinterpret_cast<const uint4*>(p);
#else
  uint4 v;
  memcpy(&v, p, 16);
  return v;
#endif
}
HD u32 fshr(u32 lo, u32 hi, int s) {
#ifdef __CUDA_ARCH__
  return __funnelshift_r(lo, hi, s);
#else
  return (u32)((((u64)hi << 32) | lo) >> s);
#endif
}
HD void cpy8(u64* d, const u64* s) {
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = s[i];
}

// --- the row: x = SHA-512(P(l) || row) ---------------------------------------------------------------------------------

// A row's bytes as 16-byte loads: a_len bytes at a, then b_len at b. A 16-byte chunk inside the first part loads at once
// when a is 16-byte aligned; any other chunk byte by byte.
struct Row2 {
  const uint8_t* a;
  const uint8_t* b;
  u32 la, lb;
  bool a16;
  HD uint4 ld16c(u32 off) const {
    if (a16 && off + 16 <= la) return ld16(a + off);
    u32 v[4] = {0, 0, 0, 0};
    for (u32 k = 0; k < 16; ++k) {
      const u32 o = off + k;
      const u32 byte = o < la ? a[o] : (o < la + lb ? b[o - la] : 0u);
      v[k >> 2] |= byte << (8 * (k & 3));
    }
    uint4 r;
    r.x = v[0]; r.y = v[1]; r.z = v[2]; r.w = v[3];
    return r;
  }
};

// SHA-512 of a 128-byte first block (its chaining value cv) and lb row bytes from src (commit_gpu's hash_after_block).
HD void hash_after_block(const u64* cv, const Row2& src, u32 lb, u64* x) {
  cpy8(x, cv);
  const u32 nblk = (lb + 17 + 127) / 128;
  for (u32 b = 0; b < nblk; ++b) {
    u64 W[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const u32 off = 128 * b + 16 * c;
      uint4 v;
      if (off < lb) {
        v = src.ld16c(off);
      } else {
        v.x = v.y = v.z = v.w = 0;
      }
      W[2 * c] = be64(v.x, v.y);
      W[2 * c + 1] = be64(v.z, v.w);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const u32 base = 128 * b + 8 * i;
      if (base + 8 > lb) {
        if (base <= lb) {
          const u32 j = lb - base;
          W[i] = (j ? W[i] & (~0ull << (64 - 8 * j)) : 0) | (0x80ull << (56 - 8 * j));
        } else {
          W[i] = 0;
        }
      }
    }
    if (b == nblk - 1) W[15] = (u64)(128 + lb) * 8;
    sha::compress(x, W);
  }
}

// --- the salt: ChaCha20, a 64-bit counter and a 64-bit nonce (C-Flock's hm96_salt) -------------------------------------

HD u32 rotl32(u32 x, int n) { return (x << n) | (x >> (32 - n)); }
#define CS_QR(a, b, c, d)                        \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16); \
  x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12); \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8);  \
  x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);

HD void chacha20_block(const u32 key[8], u64 counter, u64 nonce, u32 out[16]) {
  const u32 s[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u, key[0], key[1], key[2], key[3],
                     key[4], key[5], key[6], key[7], (u32)counter, (u32)(counter >> 32), (u32)nonce, (u32)(nonce >> 32)};
  u32 x[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) x[i] = s[i];
#pragma unroll 1
  for (int r = 0; r < 10; ++r) {
    CS_QR(0, 4, 8, 12) CS_QR(1, 5, 9, 13) CS_QR(2, 6, 10, 14) CS_QR(3, 7, 11, 15)
    CS_QR(0, 5, 10, 15) CS_QR(1, 6, 11, 12) CS_QR(2, 7, 8, 13) CS_QR(3, 4, 9, 14)
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) out[i] = x[i] + s[i];
}

// Row g's salt in slot `nonce`: ChaCha20 blocks 3g, 3g+1, 3g+2, as 24 big-endian words.
HD void salt_words(const u32 key[8], u64 nonce, u64 g, u64 Y[24]) {
#pragma unroll 1
  for (int c = 0; c < 3; ++c) {
    u32 o[16];
    chacha20_block(key, 3 * g + c, nonce, o);
#pragma unroll
    for (int i = 0; i < 8; ++i) Y[8 * c + i] = (u64)bswap32(o[2 * i]) << 32 | bswap32(o[2 * i + 1]);
  }
}

// --- hm96-sha512 ---------------------------------------------------------------------------------------------------------

// M_key y as 16 little-endian u32 (mask byte 4w + k is byte k of m[w]), from T (rows `stride` apart): for salt bytes
// p = 16 q + r, bytes [16 q, 16 q + 80) of T_{y_p} are xored into A_r, and A_r shifted down r bytes into m
// (commit_gpu's hm96_mask).
HD void hm96_mask(const uint8_t* T, int stride, const u64* Y, u32* m) {
#pragma unroll
  for (int w = 0; w < 16; ++w) m[w] = 0;
#pragma unroll 1
  for (int r = 0; r < 16; ++r) {
    u32 A[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) A[i] = 0;
#pragma unroll
    for (int q = 0; q < 12; ++q) {
      const int p = 16 * q + r;
      const u32 v = (u32)(Y[p >> 3] >> (56 - 8 * (p & 7))) & 0xFFu;
      const uint8_t* t = T + v * stride + 16 * q;
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const uint4 u = ld16_t(t + 16 * i);
        A[4 * i] ^= u.x; A[4 * i + 1] ^= u.y; A[4 * i + 2] ^= u.z; A[4 * i + 3] ^= u.w;
      }
    }
    const int sw = r >> 2, s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= s ? fshr(A[w + sw], A[w + sw + 1], s) : A[w + sw];
  }
}

// c = SHA-512(salt_prefix || y), from the chaining value after the prefix.
HD void salt_digest(const u64* ms_salt, const u64* Y, u64* c) {
  u64 w[16];
  cpy8(c, ms_salt);
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

// SHA-512(leaf_prefix(key) || b || c), from the chaining value after the prefix.
HD void tree_leaf(const u64* ms_leaf, const u64* bc, u64* out) {
  u64 w[16];
  cpy8(out, ms_leaf);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = bc[i];
  sha::compress(out, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = 0x8000000000000000ull;
  w[15] = 256 * 8;
  sha::compress(out, w);
}

// --- frame-v3-sha512 (row_table.cuh's Frame) ------------------------------------------------------------------------------

#define CS_FRAME "veritor/protocol/merkle/frame/v3"
#define CS_ROW_SCHEMA "hm96-sha512/row/v2"
#define CS_ROOTS_SCHEMA "frame-v3-sha512/root/v1"

constexpr int FRAME_WORDS = 48;   // a node, the longest message, is under 3 blocks with its padding
struct Frame {
  u64 w[FRAME_WORDS];
  u32 p;
};

// nbytes (1 to 8) big-endian bytes of v at the message's end
HD void frame_be(Frame& f, u64 v, int nbytes) {
  const u64 x = nbytes == 8 ? v : v << (64 - 8 * nbytes);
  const u32 o = f.p & 7, i = f.p >> 3;
  f.w[i] |= x >> (8 * o);
  if (o + nbytes > 8) f.w[i + 1] |= x << (64 - 8 * o);
  f.p += nbytes;
}
template <int N>
HD void frame_str(Frame& f, const char (&s)[N], bool nul) {
  for (int k = 0; k < (nul ? N : N - 1); ++k) frame_be(f, (uint8_t)s[k], 1);
}
HD int uint_len(u64 v) {   // merkle._uint: big-endian, at least one byte
  int n = 1;
  while (n < 8 && (v >> (8 * n))) ++n;
  return n;
}
HD void frame_uint(Frame& f, u64 v) {
  const int n = uint_len(v);
  frame_be(f, (u64)n, 8);
  frame_be(f, v, n);
}
HD void frame_digest(Frame& f, const u64* v) {
  frame_be(f, 64, 8);
  for (int i = 0; i < 8; ++i) frame_be(f, v[i], 8);
}
template <int N>
HD void frame_head(Frame& f, const char (&tag)[N], const u64* did) {
  for (int i = 0; i < FRAME_WORDS; ++i) f.w[i] = 0;
  f.p = 0;
  frame_str(f, CS_FRAME, true);
  frame_be(f, N - 1, 4);
  frame_str(f, tag, false);
  frame_digest(f, did);
}
HD void frame_hash(Frame& f, u64* out) {
  const u32 len = f.p, blocks = (len + 17 + 127) / 128;
  frame_be(f, 0x80, 1);
  f.w[16 * blocks - 1] = 8ull * len;
  sha::iv(out);
#pragma unroll 1
  for (u32 b = 0; b < blocks; ++b) sha::compress(out, f.w + 16 * b);
}
// CommitmentDomain.leaf(rank, rank, schema, value) of a 64-byte value
template <int N>
HD void frame_leaf(const u64* did, u64 rank, const char (&schema)[N], const u64* value, u64* out) {
  Frame f;
  frame_head(f, "leaf", did);
  frame_uint(f, rank);
  frame_uint(f, rank);
  frame_be(f, N - 1, 8);
  frame_str(f, schema, false);
  frame_digest(f, value);
  frame_hash(f, out);
}
HD void frame_node(const u64* did, u64 level, u64 index, const u64* left, const u64* right, u64* out) {
  Frame f;
  frame_head(f, "node", did);
  frame_uint(f, level);
  frame_uint(f, index);
  frame_digest(f, left);
  frame_digest(f, right);
  frame_hash(f, out);
}
HD void frame_pad(const u64* did, u64 rank, u64* out) {
  Frame f;
  frame_head(f, "pad", did);
  frame_uint(f, rank);
  frame_hash(f, out);
}

// A tree's frontier: node at `level` (its subtree's leaves [count, count + 2^level), count a multiple of 2^level) onto the
// frontier `front` (levels x 8 words) of a tree whose first `count` leaves are in; tree_levels' nodes, so once the count is
// the tree's width 2^d its root is front[d].
HD void push(const u64* did, u64* front, u64 count, int level, const u64* node) {
  u64 cur[8];
  cpy8(cur, node);
  u64 j = count >> level;
  int k = level;
  while (j & 1) {
    u64 up[8];
    frame_node(did, k, j >> 1, front + 8 * k, cur, up);
    cpy8(cur, up);
    j >>= 1;
    ++k;
  }
  cpy8(front + 8 * k, cur);
}

// --- one thread's work -----------------------------------------------------------------------------------------------------

// Row j of the group (rows[j], longest first) in slot s: its leaf in its unit's tree, to leaves[g, s].
HD void row_thread(const Ctx& c, const Step& st, const Grp& gr, u32 j, u32 s, const uint8_t* T, int stride) {
  const RowDesc& d = gr.rows[j];
  Row2 src;
  src.a = gr.src[d.a_src] + (u64)s * gr.stride[d.a_src] + d.a_off;
  src.la = d.a_len;
  src.b = d.b_len ? gr.src[d.b_src] + (u64)s * gr.stride[d.b_src] + d.b_off : src.a;
  src.lb = d.b_len;
  src.a16 = ((uintptr_t)src.a & 15) == 0;
  u64 x[8], Y[24], bc[16], h[8], leaf[8];
  u32 m[16];
  hash_after_block(d.midP, src, d.nbytes, x);
  salt_words(st.key, s, (u64)gr.row_base + d.g, Y);
  hm96_mask(T, stride, Y, m);
#pragma unroll
  for (int i = 0; i < 8; ++i) bc[i] = x[i] ^ ((u64)bswap32(m[2 * i]) << 32 | bswap32(m[2 * i + 1]));
  salt_digest(c.ms_salt, Y, bc + 8);
  tree_leaf(c.ms_leaf, bc, h);
  frame_leaf(c.unit_did + 8 * (u64)(gr.unit_base + d.unit), d.rank, CS_ROW_SCHEMA, h, leaf);
  cpy8(st.leaves + 8 * ((u64)d.g * st.S + s), leaf);
}

// Unit i of the group in slot s: its root from its rows' leaves and its pad tail, then that root's leaf in the slot tree.
HD void unit_thread(const Ctx& c, const Step& st, const Grp& gr, u32 i, u32 s) {
  const UnitDesc& u = gr.units[i];
  const u32 unit = gr.unit_base + i;
  const u64* did = c.unit_did + 8 * (u64)unit;
  u64 front[8 * (UNIT_LV + 1)];
  u64 n = 0;
  for (u32 r = 0; r < u.rows; ++r, ++n) push(did, front, n, 0, st.leaves + 8 * ((u64)(u.row0 + r) * st.S + s));
  const u32 off = c.unit_tail[2 * unit], cnt = c.unit_tail[2 * unit + 1];
  for (u32 k = 0; k < cnt; ++k) {
    const int lv = (int)c.tail_lv[off + k];
    push(did, front, n, lv, c.tail_node + 8 * (u64)(off + k));
    n += 1ull << lv;
  }
  u64 rl[8];
  frame_leaf(c.slot_did, unit, CS_ROOTS_SCHEMA, front + 8 * u.depth, rl);
  cpy8(st.rl + 8 * ((u64)i * st.S + s), rl);
}

// Slot s: the group's unit-root leaves onto its slot tree's frontier, in unit order.
HD void fold_thread(const Ctx& c, const Step& st, const Grp& gr, u32 s) {
  u64* front = st.front + (u64)s * SLOT_LV * 8;
  for (u32 i = 0; i < gr.nunits; ++i) push(c.slot_did, front, (u64)gr.unit_base + i, 0, st.rl + 8 * ((u64)i * st.S + s));
}

// Slot s: its tree's pad tail, its root (to slot_root[s]) and that root's leaf in the step tree (to lv_a[s]).
HD void finish_thread(const Ctx& c, const Step& st, u32 s) {
  u64* front = st.front + (u64)s * SLOT_LV * 8;
  u64 n = c.U;
  for (u32 k = 0; k < c.slot_tail_n; ++k) {
    const int lv = (int)c.tail_lv[c.slot_tail_off + k];
    push(c.slot_did, front, n, lv, c.tail_node + 8 * (u64)(c.slot_tail_off + k));
    n += 1ull << lv;
  }
  const u64* root = front + 8 * c.slot_depth;
  cpy8(st.slot_root + 8 * (u64)s, root);
  u64 leaf[8];
  frame_leaf(st.step_did, s, CS_ROOTS_SCHEMA, root, leaf);
  cpy8(st.lv_a + 8 * (u64)s, leaf);
}

// Position i >= S of the step tree's leaves: a pad.
HD void step_pad_thread(const Step& st, u32 i) { frame_pad(st.step_did, i, st.lv_a + 8 * (u64)i); }

// Node i at level l + 1 of the step tree.
HD void level_thread(const Step& st, const u64* in, u64* out, int l, u32 i) {
  frame_node(st.step_did, l, i, in + 16 * (u64)i, in + 16 * (u64)i + 8, out + 8 * (u64)i);
}

// --- host helpers: midstates and pad tails ---------------------------------------------------------------------------------

inline void midstate(const uint8_t* block, u64* out) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) w[i] = sha::be64(block + 8 * i);
  sha::iv(out);
  sha::compress(out, w);
}

// The root of the pads at level l, index j (leaves [j 2^l, (j + 1) 2^l)).
inline void pad_root(const u64* did, int l, u64 j, u64* out) {
  if (l == 0) {
    frame_pad(did, j, out);
    return;
  }
  u64 a[8], b[8];
  pad_root(did, l - 1, 2 * j, a);
  pad_root(did, l - 1, 2 * j + 1, b);
  frame_node(did, l - 1, j, a, b, out);
}

// The pushes that take a tree of n >= 1 leaves (of its 2^depth) to its width: (level, pad subtree) each; returns how many.
inline int tail(const u64* did, u64 n, int depth, u32* lv, u64* nodes) {
  int k = 0;
  for (u64 c = n; c < (1ull << depth); ++k) {
    int l = 0;
    while (!((c >> l) & 1)) ++l;
    lv[k] = (u32)l;
    pad_root(did, l, c >> l, nodes + 8 * k);
    c += 1ull << l;
  }
  return k;
}

}  // namespace cs
