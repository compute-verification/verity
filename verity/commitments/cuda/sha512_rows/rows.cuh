// The commitment's hashes as host+device functions, one thread's work each (engine.py is the call, README.md the formats):
// nvcc builds rows.cu (the kernels), g++ builds rows_host.cpp (the host twin, a loop a kernel), and both run what is below.
// The frame-v3-sha512 leaf, node and pad and the hm96 leaf's mask, salt digest and value are row_commit.cuh's; this file
// keeps the schedule: the row digest, the layout engine.py fills, where each leaf's salt is read and each lane's column
// order. A leaf's salt is data: 192 bytes the caller drew for it, read from the caller's buffer and derived from nothing.
// A digest is 8 u64 words, each the big-endian value of its 8 bytes. C is a compression, row_commit.cuh's policy:
// C::compress(st, w) has sx::compress's contract, C::compress_kw(st, kw) sx::compress_kw's.
#pragma once
#include "../row_commit.cuh"

#ifndef __CUDACC__
struct uint4 {
  uint32_t x, y, z, w;
};
#endif

namespace sxr {

using sx::u32;
using sx::u64;

// --- layout, as engine.py lays it out ------------------------------------------------------------------------------------

struct RowSpec {     // 752 bytes: row g's bytes in slot s are a_len at a + s a_stride, then b_len at b + s b_stride
  u64 a, a_stride;
  u64 b, b_stride;
  u32 a_len, b_len;
  u32 nbytes;        // a_len + b_len
  u32 a16;           // a and a_stride are multiples of 16
  u64 mid[8];        // row_setup: the chaining value after P(8 nbytes), sha512/row/v2's first block
  u64 kw[80];        // row_setup, when the last block is padding alone (pad_alone): its K[t] + W[t], for compress_kw
};
static_assert(sizeof(RowSpec) == 752, "RowSpec is 752 bytes");

struct LeafRow {     // 224 bytes: row g's frame leaf, rank and position in the domain dids[did]
  u64 rank, position;
  u32 did, pad0_;
  u32 q, pad_;       // leaf_setup: the leaf prefix's bytes after its first block
  u64 mid[8];        // leaf_setup: the chaining value after that block
  u64 tail[16];      // leaf_setup: those q bytes, big-endian packed
};
static_assert(sizeof(LeafRow) == 224, "LeafRow is 224 bytes");

constexpr int SALT_U32 = 48;     // hm96-sha512's 192-byte salt, as little-endian u32 words
constexpr int T_BYTES = 256 * 256;   // T: 256 rows of 256 bytes, in shared memory as in global memory

// --- loads ---------------------------------------------------------------------------------------------------------------

SX_HD uint4 ld16(const uint8_t* p) {
#ifdef __CUDA_ARCH__
  return __ldg(reinterpret_cast<const uint4*>(p));
#else
  uint4 v;
  memcpy(&v, p, 16);
  return v;
#endif
}
SX_HD u64 ld8(const uint64_t* p) {
#ifdef __CUDA_ARCH__
  return __ldg(p);
#else
  return *p;
#endif
}
SX_HD uint8_t ld1(const uint8_t* p) {
#ifdef __CUDA_ARCH__
  return __ldg(p);
#else
  return *p;
#endif
}

// --- sha512/row/v2 -------------------------------------------------------------------------------------------------------

// whether an n-byte row's last block is its padding alone (0x80 and the length, or the length): n mod 128 is 0 or at least
// 112. That block is the same in every slot, so row_setup schedules it once and row_digest runs its rounds alone.
SX_HD bool pad_alone(u32 n) { return n % 128 == 0 || n % 128 >= 112; }

// row_setup: P(8 nbytes) = ("verity/sha512-row/v2\0" || u64be(8 nbytes)) zero-padded to 128 bytes, compressed from the IV,
// and with pad_alone, the last block's K[t] + W[t]
template <class C>
SX_HD void row_setup(RowSpec& r) {
  const char tag[] = "verity/sha512-row/v2";
  uint8_t b[128] = {};
  for (int i = 0; i < (int)sizeof tag; ++i) b[i] = (uint8_t)tag[i];
  rc::be_put(b + sizeof tag, 8 * (u64)r.nbytes, 8);
  rc::midstate<C>(b, r.mid);
  if (pad_alone(r.nbytes)) {
    u64 w[16] = {};
    w[0] = r.nbytes % 128 == 0 ? 0x8000000000000000ull : 0;
    w[15] = 8 * (128 + (u64)r.nbytes);
    sx::kw_of(w, r.kw);
  }
}

// the 8 message bytes at row offset o (o < nbytes + 1), past part a's whole blocks: the row's bytes, then 0x80, then zeros
SX_HD u64 row_word(const uint8_t* a, u32 la, const uint8_t* b, u32 n, u32 o) {
  if (o + 8 <= la && (((uintptr_t)(a + o)) & 7) == 0) return sx::bswap64(ld8(reinterpret_cast<const uint64_t*>(a + o)));
  if (o > n) return 0;
  u64 v = 0;
  for (u32 k = 0; k < 8; ++k) {
    const u32 p = o + k;
    const u64 byte = p < la ? ld1(a + p) : p < n ? ld1(b + (p - la)) : p == n ? 0x80 : 0;
    v = v << 8 | byte;
  }
  return v;
}

// x = SHA-512(P(8 nbytes) || row g's bytes in slot s)
template <class C>
SX_HD void row_digest(const RowSpec& r, u64 s, u64* x) {
  const uint8_t* a = reinterpret_cast<const uint8_t*>(r.a + s * r.a_stride);
  const uint8_t* b = reinterpret_cast<const uint8_t*>(r.b + s * r.b_stride);
  const u32 n = r.nbytes, blocks = (n + 17 + 127) / 128, fast = r.a16 ? r.a_len / 128 : 0, last = blocks - pad_alone(n);
  for (int i = 0; i < 8; ++i) x[i] = r.mid[i];
  u64 w[16];
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
  for (u32 k = 0; k < fast; ++k) {
    const uint8_t* p = a + 128 * k;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint4 u = ld16(p + 16 * i);
      w[2 * i] = sx::be64(u.x, u.y);
      w[2 * i + 1] = sx::be64(u.z, u.w);
    }
    C::compress(x, w);
  }
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
  for (u32 k = fast; k < last; ++k) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = row_word(a, r.a_len, b, n, 128 * k + 8 * i);
    if (k + 1 == blocks) w[15] = 8 * (128 + (u64)n);
    C::compress(x, w);
  }
  if (last < blocks) C::compress_kw(x, r.kw);
}

// --- hm96-sha512 under hiding leaves, then the frame-v3-sha512 leaf ------------------------------------------------------

struct LeafArgs {
  const LeafRow* rows;
  const u64* x;          // (G, S, 8) inner digests
  u64* out;              // (G, S, 8) leaves; with frames, (rest, S, 8): the leaves of the rows whose unit has several
  const uint8_t* T;      // hm96's Hankel table (engine.hankel_table): T_v, 256 rows of 256 bytes
  const uint8_t* salts;  // the caller's y, 192 bytes a leaf: row g's in slot s at salts + g salt_g + s salt_s, byte n of
                         // the salt at offset n
  u64 salt_g, salt_s;    // the bytes from a row's salts to the next row's and from a slot's to the next slot's (192 S and
                         // 192 for a (G, S, 192) buffer); on the device they and salts are multiples of 16
  u64 ms_salt[8];        // the chaining value after hm96's salt prefix
  u64 ms_leaf[8];        // after its leaf prefix (the key's)
  u32 G, S;
  u32 minb;              // the device kernel's blocks an SM (its register limit): 1 or 2, 0 for the default
  u32 sms;               // the SMs its grid fills (a green context's), 0 for the device's
  const LeafRow* frames; // null, or each unit's slot-tree leaf (frame_setup)
  const u32* dest;       // with frames, row g's unit d < units when that unit is the row alone: its leaf is the unit's
                         // root, whose slot-tree leaf goes to units_out[s][d]; else units + i, its leaf going to out[i][s]
  u64* units_out;        // with frames, (S, units, 8)
  u32 units, pad2_;
};

// a domain id's 64 bytes from its words
SX_HD void did_bytes(const u64* w, uint8_t* b) {
  for (int i = 0; i < 8; ++i) rc::be_put(b + 8 * i, w[i], 8);
}

// leaf_setup: the leaf prefix (merkle.CommitmentDomain.leaf_prefix(rank, position, schema, 64)) as its head (rc::leaf_head)
template <class C>
SX_HD void leaf_setup(LeafRow& r, const u64* dids, const uint8_t* schema, u32 schema_len) {
  uint8_t did[64];
  did_bytes(dids + 8 * (u64)r.did, did);
  r.q = (u32)rc::leaf_head<C>(did, r.rank, r.position, schema, (int)schema_len, r.mid, r.tail);
}

// frame_setup: a unit's slot-tree leaf (rank, rank, schema) in domain did, as leaf_setup sets up a row's
template <class C>
SX_HD void frame_setup(LeafRow& r, u64 rank, const u64* did, const uint8_t* schema, u32 schema_len) {
  r.rank = r.position = rank;
  r.did = r.pad0_ = r.pad_ = 0;
  leaf_setup<C>(r, did, schema, schema_len);
}

// the frame leaf of the 64-byte value v under r's head (out may be v)
template <class C>
SX_HD void frame_tail(const LeafRow& r, const u64* v, u64* out) {
  rc::leaf_tail<C>(r.mid, r.tail, r.q, v, out);
}

// the 16-byte column q a lane l (0-7) takes at step k (0-11) of each r. A 16-byte load from T_v's column q, unit i, is in
// bank group (q + i) mod 8 whatever v (T's rows are 256 bytes apart), and a 16-byte load's lanes go to shared memory 8 at a
// time, so lanes on columns distinct mod 8 don't conflict: in steps 0-3 lanes 0-3 take columns 8-11 and lanes 4-7 columns
// 4-7, in steps 4-7 the other way round, and in steps 8-11 every lane takes columns 0-3 (two lanes a bank group). Each lane
// takes every column once, and the mask is their XOR, so the order changes no byte.
SX_HD u32 mask_column(int k, u32 l) {
  const u32 base = k >= 8 ? 0 : ((k >> 2) ^ (l >> 2)) ? 4 : 8;
  return base + ((k + l) & 3);
}
// lane l's column order, for rc::mask_lanes
struct LaneColumns {
  u32 l;
  SX_HD u32 operator()(int k) const { return mask_column(k, l); }
};

// row g's leaf in slot s under its salt y, the 192 bytes the caller gave for it (A.salts): b = x ^ M y, c =
// SHA-512(salt_prefix || y), h = SHA-512(leaf_prefix(key) || b || c), then the frame leaf SHA-512(leaf prefix || h), and with
// A.frames, for a unit of this row alone, its slot-tree leaf over that (the unit's root). T is A.T's table (in shared memory
// on the device), y the thread's SALT_U32 words of scratch, ys apart (in shared memory on the device, so that y's 48
// registers are free), and l the thread's lane in its group of 8 (mask_column).
template <class C>
SX_HD void leaf(const LeafArgs& A, const uint8_t* T, u32* y, u32 ys, u32 l, u32 g, u32 s) {
  const LeafRow& r = A.rows[g];
  const u64 gs = (u64)g * A.S + s;
  const uint8_t* sp = A.salts + g * A.salt_g + s * A.salt_s;
  for (int i = 0; i < 12; ++i) {
    const uint4 u = ld16(sp + 16 * i);
    y[4 * i * ys] = u.x;
    y[(4 * i + 1) * ys] = u.y;
    y[(4 * i + 2) * ys] = u.z;
    y[(4 * i + 3) * ys] = u.w;
  }
  u64 bc[16];
  {
    u32 m[16];
    u64 my[8];
    rc::mask_lanes(rc::ByteTab<256, false>{T}, y, ys, LaneColumns{l}, m);
    rc::mask_words(m, my);
    for (int i = 0; i < 8; ++i) bc[i] = ld8(A.x + 8 * gs + i) ^ my[i];
  }
  rc::salt_digest<C>(A.ms_salt, rc::SaltU32{y, ys}, bc + 8);
  u64 h[8];
  rc::hm96_value<C>(A.ms_leaf, sx::kw_pad256(), bc, h);
  const LeafRow* unit = nullptr;
  u64* dst = A.out + 8 * gs;
  if (A.frames) {
    const u32 d = A.dest[g];
    if (d < A.units) {
      unit = A.frames + d;
      dst = A.units_out + 8 * ((u64)s * A.units + d);
    } else {
      dst = A.out + 8 * ((u64)(d - A.units) * A.S + s);
    }
  }
  const u32 hops = unit ? 2 : 1;
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
  for (u32 k = 0; k < hops; ++k) frame_tail<C>(k ? *unit : r, h, h);
  for (int i = 0; i < 8; ++i) dst[i] = h[i];
}

// --- frame-v3-sha512 leaves of digests, node midstates and pads ----------------------------------------------------------

// CommitmentDomain.leaf(rank, rank, schema, value) of a 64-byte value
template <class C>
SX_HD void frame_leaf(const u64* did, u64 rank, const uint8_t* schema, u32 schema_len, const u64* value, u64* out) {
  uint8_t d[64];
  u64 mid[8], tail[16];
  did_bytes(did, d);
  const int q = rc::leaf_head<C>(d, rank, rank, schema, (int)schema_len, mid, tail);
  rc::leaf_tail<C>(mid, tail, (u32)q, value, out);
}

// a node's chaining value after its first block at `level` (< 256), from which rc::node_at hashes the node
template <class C>
SX_HD void node_mid(const u64* did, u32 level, u64* mid) {
  uint8_t d[64];
  did_bytes(did, d);
  rc::node_mid<C>(d, (int)level, mid);
}

// CommitmentDomain.pad(rank)
template <class C>
SX_HD void pad(const u64* did, u64 rank, u64* out) {
  uint8_t d[64];
  did_bytes(did, d);
  rc::pad_of<C>(d, rank, out);
}

}  // namespace sxr
