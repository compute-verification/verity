// The commitment's hashes as host+device functions, one thread's work each (engine.py is the call, README.md the formats):
// nvcc builds rows.cu (the kernels), g++ builds rows_host.cpp (the host twin, a loop a kernel), and both run what is below.
// A digest is 8 u64 words, each the big-endian value of its 8 bytes. C is a compression: C::run(st, w) has sx::compress's
// contract, C::kw(st, kw) sx::compress_kw's. SXR_COUNT_CHACHA(), if defined, runs once a ChaCha20 block (the host twin's
// count).
#pragma once
#include "../../../verity/ml/kernels/sha512.cuh"

#ifndef SXR_COUNT_CHACHA
#define SXR_COUNT_CHACHA()
#endif

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

struct LeafRow {     // 224 bytes: row g's frame leaf, rank and position in the domain dids[did]; its salt is row salt_row's
  u64 rank, position;
  u32 did, salt_row;
  u32 q, pad_;       // leaf_setup: the leaf prefix's bytes after its first block
  u64 mid[8];        // leaf_setup: the chaining value after that block
  u64 tail[16];      // leaf_setup: those q bytes, big-endian packed
};
static_assert(sizeof(LeafRow) == 224, "LeafRow is 224 bytes");

constexpr int SALT_U32 = 48;     // hm96-sha512's 192-byte salt, as little-endian u32 words
constexpr int T_BYTES = 256 * 256;   // T: 256 rows of 256 bytes, in shared memory as in global memory

#define SXR_FRAME "veritor/protocol/merkle/frame/v3"

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
SX_HD uint4 ld16_any(const uint8_t* p) {   // 16 aligned bytes from any space (shared memory included)
#ifdef __CUDA_ARCH__
  return *reinterpret_cast<const uint4*>(p);
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

// --- a message built a byte run at a time (frame-v3's framing), big-endian packed --------------------------------------

constexpr int MSG_WORDS = 48;
struct Msg {
  u64 w[MSG_WORDS];
  u32 n;
};
SX_HD void msg_init(Msg& m) {
  for (int i = 0; i < MSG_WORDS; ++i) m.w[i] = 0;
  m.n = 0;
}
// the low nbytes (0 to 8) bytes of v, big-endian
SX_HD void put(Msg& m, u64 v, int nbytes) {
  if (nbytes == 0) return;
  const u64 x = nbytes == 8 ? v : v << (64 - 8 * nbytes);
  const u32 o = m.n & 7, i = m.n >> 3;
  m.w[i] |= x >> (8 * o);
  if (o && (int)o + nbytes > 8) m.w[i + 1] |= x << (64 - 8 * o);
  m.n += nbytes;
}
template <int N>
SX_HD void put_str(Msg& m, const char (&s)[N], bool nul) {
  for (int k = 0; k < (nul ? N : N - 1); ++k) put(m, (uint8_t)s[k], 1);
}
SX_HD void put_words(Msg& m, const u64* v, int nbytes) {   // nbytes of big-endian packed words
  for (int k = 0; k < nbytes / 8; ++k) put(m, v[k], 8);
  if (nbytes & 7) put(m, v[nbytes / 8] >> (64 - 8 * (nbytes & 7)), nbytes & 7);
}
SX_HD int uint_len(u64 v) {   // merkle._uint: big-endian in the fewest bytes, one for 0
  int n = 1;
  while (n < 8 && (v >> (8 * n))) ++n;
  return n;
}
SX_HD void put_uint(Msg& m, u64 v) {   // a frame part: u64be(length) || uint(v)
  const int n = uint_len(v);
  put(m, (u64)n, 8);
  put(m, v, n);
}
SX_HD void put_digest(Msg& m, const u64* d) {   // a frame part: u64be(64) || d
  put(m, 64, 8);
  for (int i = 0; i < 8; ++i) put(m, d[i], 8);
}
// SHA-512's padding for a message whose first `before` bytes are already in cv, then its blocks
template <class C>
SX_HD void msg_hash(Msg& m, u64 before, u64* cv) {
  const u32 len = m.n, blocks = (len + 17 + 127) / 128;
  put(m, 0x80, 1);
  m.w[16 * blocks - 1] = 8 * (before + len);
  for (u32 b = 0; b < blocks; ++b) C::run(cv, m.w + 16 * b);
}
template <int N>
SX_HD void frame_head(Msg& m, const char (&tag)[N], const u64* did) {   // frame || u32be(len tag) || tag || did's part
  msg_init(m);
  put_str(m, SXR_FRAME, true);
  put(m, N - 1, 4);
  put_str(m, tag, false);
  put_digest(m, did);
}

// --- a block built in registers: Msg's offsets are run-time values, so the device keeps a Msg in local memory ------------

// w[0, NW) |= the NV words v at byte offset off (off >> 3 <= MAXQ, off + 8 NV <= 8 NW). Every index is a constant (the word
// offset is selected, not indexed), so w and v stay in registers.
template <int NW, int NV, int MAXQ>
SX_HD void place(u64* w, const u64* v, u32 off) {
  const u32 qw = off >> 3, sh = 8 * (off & 7);
  u64 u[NV + 1];
#pragma unroll
  for (int k = 0; k <= NV; ++k) u[k] = (k < NV ? v[k] >> sh : 0) | ((k > 0 && sh) ? v[k - 1] << (64 - sh) : 0);
#pragma unroll
  for (int i = 0; i < NW; ++i) {
#pragma unroll
    for (int j = 0; j <= MAXQ; ++j)
      if (i - j >= 0 && i - j <= NV && qw == (u32)j) w[i] |= u[i - j];
  }
}

// --- sha512/row/v2 -------------------------------------------------------------------------------------------------------

// whether an n-byte row's last block is its padding alone (0x80 and the length, or the length): n mod 128 is 0 or at least
// 112. That block is the same in every slot, so row_setup schedules it once and row_digest runs its rounds alone.
SX_HD bool pad_alone(u32 n) { return n % 128 == 0 || n % 128 >= 112; }

// row_setup: P(8 nbytes) = ("verity/sha512-row/v2\0" || u64be(8 nbytes)) zero-padded to 128 bytes, compressed from the IV,
// and with pad_alone, the last block's K[t] + W[t]
template <class C>
SX_HD void row_setup(RowSpec& r) {
  Msg m;
  msg_init(m);
  put_str(m, "verity/sha512-row/v2", true);
  put(m, 8 * (u64)r.nbytes, 8);
  sx::iv(r.mid);
  C::run(r.mid, m.w);
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
    C::run(x, w);
  }
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
  for (u32 k = fast; k < last; ++k) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = row_word(a, r.a_len, b, n, 128 * k + 8 * i);
    if (k + 1 == blocks) w[15] = 8 * (128 + (u64)n);
    C::run(x, w);
  }
  if (last < blocks) C::kw(x, r.kw);
}

// --- hm96-sha512 under hiding leaves, then the frame-v3-sha512 leaf ------------------------------------------------------

// ChaCha20 (20 rounds, 64-bit counter in words 12-13, 64-bit nonce in 14-15: verity.ml.crypto.chacha20)
#define SXR_QR(a, b, c, d)                                                                                                 \
  x[a] += x[b]; x[d] ^= x[a]; x[d] = (x[d] << 16) | (x[d] >> 16);                                                          \
  x[c] += x[d]; x[b] ^= x[c]; x[b] = (x[b] << 12) | (x[b] >> 20);                                                          \
  x[a] += x[b]; x[d] ^= x[a]; x[d] = (x[d] << 8) | (x[d] >> 24);                                                           \
  x[c] += x[d]; x[b] ^= x[c]; x[b] = (x[b] << 7) | (x[b] >> 25);
SX_HD void chacha20(const u32* key, u64 counter, u64 nonce, u32* out) {
  SXR_COUNT_CHACHA();
  const u32 st[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u, key[0], key[1], key[2], key[3], key[4], key[5],
                      key[6], key[7], (u32)counter, (u32)(counter >> 32), (u32)nonce, (u32)(nonce >> 32)};
  u32 x[16];
  for (int i = 0; i < 16; ++i) x[i] = st[i];
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
  for (int r = 0; r < 10; ++r) {
    SXR_QR(0, 4, 8, 12) SXR_QR(1, 5, 9, 13) SXR_QR(2, 6, 10, 14) SXR_QR(3, 7, 11, 15)
    SXR_QR(0, 5, 10, 15) SXR_QR(1, 6, 11, 12) SXR_QR(2, 7, 8, 13) SXR_QR(3, 4, 9, 14)
  }
  for (int i = 0; i < 16; ++i) out[i] = x[i] + st[i];
}
#undef SXR_QR

// y as SALT_U32 little-endian words, word w at y[w ys]: ChaCha20 blocks 3 salt_row .. 3 salt_row + 2 under key with nonce
// (C-Flock's hm96_salt)
SX_HD void salt_key_words(const u32* key, u64 nonce, u64 salt_row, u32* y, u32 ys) {
  for (int c = 0; c < 3; ++c) {
    u32 o[16];
    chacha20(key, 3 * salt_row + c, nonce, o);
    for (int i = 0; i < 16; ++i) y[(16 * c + i) * ys] = o[i];
  }
}

struct LeafArgs {
  const LeafRow* rows;
  const u64* x;          // (G, S, 8) inner digests
  u64* out;              // (G, S, 8) leaves; with frames, (rest, S, 8): the leaves of the rows whose unit has several
  const uint8_t* T;      // hm96's Hankel table (engine.hankel_table): T_v, 256 rows of 256 bytes
  const uint8_t* salts;  // (G, S, 192) the caller's y, or null for the salt key's
  u32 key[8];            // the salt key (ChaCha20's key words), when salts is null
  u64 nonce0;            // slot s's nonce is nonce0 + s
  u64 ms_salt[8];        // the chaining value after hm96's salt prefix
  u64 ms_leaf[8];        // after its leaf prefix (the key's)
  u32 G, S;
  u32 minb;              // the device kernel's blocks an SM (its register cap): 1 or 2, 0 for the default
  u32 sms;               // the SMs its grid fills (a green context's), 0 for the device's
  const LeafRow* frames; // null, or each unit's slot-tree leaf (frame_setup)
  const u32* dest;       // with frames, row g's unit d < units when that unit is the row alone: its leaf is the unit's
                         // root, whose slot-tree leaf goes to units_out[s][d]; else units + i, its leaf going to out[i][s]
  u64* units_out;        // with frames, (S, units, 8)
  u32 units, pad2_;
};

// leaf_setup: the leaf prefix (merkle.CommitmentDomain.leaf_prefix(rank, position, schema, 64)), its first block compressed
template <class C>
SX_HD void leaf_setup(LeafRow& r, const u64* dids, const uint8_t* schema, u32 schema_len) {
  Msg m;
  frame_head(m, "leaf", dids + 8 * (u64)r.did);
  put_uint(m, r.rank);
  put_uint(m, r.position);
  put(m, schema_len, 8);
  for (u32 k = 0; k < schema_len; ++k) put(m, schema[k], 1);
  put(m, 64, 8);
  sx::iv(r.mid);
  C::run(r.mid, m.w);
  r.q = m.n - 128;
  for (int i = 0; i < 16; ++i) r.tail[i] = m.w[16 + i];
}

// frame_setup: a unit's slot-tree leaf (rank, rank, schema) in domain did, as leaf_setup sets up a row's
template <class C>
SX_HD void frame_setup(LeafRow& r, u64 rank, const u64* did, const uint8_t* schema, u32 schema_len) {
  r.rank = r.position = rank;
  r.did = r.salt_row = r.pad_ = 0;
  leaf_setup<C>(r, did, schema, schema_len);
}

// the frame leaf of the 64-byte value v under r's prefix (out may be v): one compression from r.mid when the rest of the
// prefix and v fit one block (q <= 47: ranks and positions of at most 6 bytes together, as a row's; a unit's below 2^24), in
// registers; else Msg's two
template <class C>
SX_HD void frame_tail(const LeafRow& r, const u64* v, u64* out) {
  const u32 q = r.q;
  if (q <= 47) {
    u64 w[16], vm[9];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = i < 6 ? r.tail[i] : 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) vm[i] = v[i];
    vm[8] = 0x8000000000000000ull;
    place<16, 9, 5>(w, vm, q);
    w[15] = 8 * (128 + 64 + (u64)q);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = r.mid[i];
    C::run(out, w);
    return;
  }
  Msg m;
  msg_init(m);
  put_words(m, r.tail, (int)q);
  for (int i = 0; i < 8; ++i) put(m, v[i], 8);
  for (int i = 0; i < 8; ++i) out[i] = r.mid[i];
  msg_hash<C>(m, 128, out);
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

// M_key y as 16 little-endian u32 (mask byte 4w + k is byte k of m[w]), from T (rows 256 bytes apart) and y (SALT_U32 words,
// word w at y[w ys]), the column order lane l's. Bytes [p, p + 64) of T_v are byte_tables(key)[p][v], so for salt bytes
// p = 16 q + r the 80 bytes [16 q, 16 q + 80) of T_{y_p} are xored into A_r, and A_r shifted down r bytes into m (commit_gpu's
// hm96_mask, benchmarks/pouw/commit_gpu/commit_gpu.cuh, with its columns in lane l's order).
SX_HD void mask(const uint8_t* T, const u32* y, u32 ys, u32 l, u32* m) {
#pragma unroll
  for (int w = 0; w < 16; ++w) m[w] = 0;
#pragma unroll
  for (int a = 0; a < 4; ++a) {   // r = 4 a .. 4 a + 3: bytes of the words 4 q + a
    u32 yw[12];
#pragma unroll
    for (int k = 0; k < 12; ++k) yw[k] = y[(4 * mask_column(k, l) + a) * ys];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
      const int r = 4 * a + rr;
      u32 A[20];
#pragma unroll
      for (int i = 0; i < 20; ++i) A[i] = 0;
#pragma unroll
      for (int k = 0; k < 12; ++k) {
        const u32 v = (yw[k] >> (8 * rr)) & 0xFFu;
        const uint8_t* t = T + 256 * v + 16 * mask_column(k, l);
#pragma unroll
        for (int i = 0; i < 5; ++i) {
          const uint4 u = ld16_any(t + 16 * i);
          A[4 * i] ^= u.x; A[4 * i + 1] ^= u.y; A[4 * i + 2] ^= u.z; A[4 * i + 3] ^= u.w;
        }
      }
      const int sw = r >> 2, sh = 8 * (r & 3);
#pragma unroll
      for (int w = 0; w < 16; ++w) m[w] ^= sh ? (A[w + sw] >> sh | A[w + sw + 1] << (32 - sh)) : A[w + sw];
    }
  }
}

// row g's leaf in slot s: b = x ^ M y, c = SHA-512(salt_prefix || y), h = SHA-512(leaf_prefix(key) || b || c), then the frame
// leaf SHA-512(leaf prefix || h), and with A.frames, for a unit of this row alone, its slot-tree leaf over that (the unit's
// root). T is A.T's table (in shared memory on the device), y the thread's SALT_U32 words of scratch, ys apart (in shared
// memory on the device, so that y's 48 registers are free), and l the thread's lane in its group of 8 (mask_column).
template <class C>
SX_HD void leaf(const LeafArgs& A, const uint8_t* T, u32* y, u32 ys, u32 l, u32 g, u32 s) {
  const LeafRow& r = A.rows[g];
  const u64 gs = (u64)g * A.S + s;
  if (A.salts) {
    const uint8_t* sp = A.salts + 192 * gs;
    for (int i = 0; i < 12; ++i) {
      const uint4 u = ld16(sp + 16 * i);
      y[4 * i * ys] = u.x;
      y[(4 * i + 1) * ys] = u.y;
      y[(4 * i + 2) * ys] = u.z;
      y[(4 * i + 3) * ys] = u.w;
    }
  } else {
    salt_key_words(A.key, A.nonce0 + s, r.salt_row, y, ys);
  }
  u64 bc[16];
  {
    u32 m[16];
    mask(T, y, ys, l, m);
    for (int i = 0; i < 8; ++i) bc[i] = ld8(A.x + 8 * gs + i) ^ ((u64)sx::bswap32(m[2 * i]) << 32 | sx::bswap32(m[2 * i + 1]));
  }
  u64 w[16];
  u64* c = bc + 8;
  for (int i = 0; i < 8; ++i) c[i] = A.ms_salt[i];
  for (int i = 0; i < 16; ++i) w[i] = sx::be64(y[2 * i * ys], y[(2 * i + 1) * ys]);
  C::run(c, w);
  for (int i = 0; i < 8; ++i) w[i] = sx::be64(y[(32 + 2 * i) * ys], y[(33 + 2 * i) * ys]);
  w[8] = 0x8000000000000000ull;
  for (int i = 9; i < 15; ++i) w[i] = 0;
  w[15] = (128 + 192) * 8;
  C::run(c, w);
  u64 h[8];
  for (int i = 0; i < 8; ++i) h[i] = A.ms_leaf[i];
  for (int i = 0; i < 16; ++i) w[i] = bc[i];
  C::run(h, w);
  C::kw(h, sx::kw_pad256());
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

// --- frame-v3-sha512 leaves of digests, nodes and pads -------------------------------------------------------------------

// CommitmentDomain.leaf(rank, rank, schema, value) of a 64-byte value
template <class C>
SX_HD void frame_leaf(const u64* did, u64 rank, const uint8_t* schema, u32 schema_len, const u64* value, u64* out) {
  Msg m;
  frame_head(m, "leaf", did);
  put_uint(m, rank);
  put_uint(m, rank);
  put(m, schema_len, 8);
  for (u32 k = 0; k < schema_len; ++k) put(m, schema[k], 1);
  put_digest(m, value);
  sx::iv(out);
  msg_hash<C>(m, 0, out);
}

// node_mid: a node's first block at `level` (< 256), whose bytes after the domain id are u64be(1), level and the first
// six bytes of u64be(len uint(index)), all zero
template <class C>
SX_HD void node_mid(const u64* did, u32 level, u64* mid) {
  Msg m;
  frame_head(m, "node", did);
  put(m, 1, 8);
  put(m, level, 1);
  sx::iv(mid);
  C::run(mid, m.w);
}

// CommitmentDomain.node(level, index, left, right), from node_mid's chaining value: the 146 + n bytes after its first block
// (u64be(n)'s last two, index's n, then u64be(64) || left || u64be(64) || right) and the padding are two blocks, built in
// registers
template <class C>
SX_HD void node(const u64* mid, u64 index, const u64* left, const u64* right, u64* out) {
  const u32 n = (u32)uint_len(index);
  u64 w[32], v[19];
#pragma unroll
  for (int i = 0; i < 32; ++i) w[i] = 0;
  w[0] = (u64)n << 48;
  const u64 ix = n == 8 ? index : index << (64 - 8 * n);
  place<32, 1, 0>(w, &ix, 2);
  v[0] = v[9] = 64;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[1 + i] = left[i];
    v[10 + i] = right[i];
  }
  v[18] = 0x8000000000000000ull;
  place<32, 19, 1>(w, v, 2 + n);
  w[31] = 8 * (128 + 146 + (u64)n);
  for (int i = 0; i < 8; ++i) out[i] = mid[i];
  C::run(out, w);
  C::run(out, w + 16);
}

// CommitmentDomain.pad(rank)
template <class C>
SX_HD void pad(const u64* did, u64 rank, u64* out) {
  Msg m;
  frame_head(m, "pad", did);
  put_uint(m, rank);
  sx::iv(out);
  msg_hash<C>(m, 0, out);
}

}  // namespace sxr
