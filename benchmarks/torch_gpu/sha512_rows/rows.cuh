// The commitment's hashes as host+device functions, one thread's work each (engine.py is the call, README.md the formats):
// nvcc builds rows.cu (the kernels), g++ builds rows_host.cpp (the host twin, a loop a kernel), and both run what is below.
// A digest is 8 u64 words, each the big-endian value of its 8 bytes. C is a compression: C::run(st, w) has sx::compress's
// contract.
#pragma once
#include "sha512x.cuh"

#ifndef __CUDACC__
struct uint4 {
  uint32_t x, y, z, w;
};
#endif

namespace sxr {

using sx::u32;
using sx::u64;

// --- layout, as engine.py lays it out ------------------------------------------------------------------------------------

struct RowSpec {     // 112 bytes: row g's bytes in slot s are a_len at a + s a_stride, then b_len at b + s b_stride
  u64 a, a_stride;
  u64 b, b_stride;
  u32 a_len, b_len;
  u32 nbytes;        // a_len + b_len
  u32 a16;           // a and a_stride are multiples of 16
  u64 mid[8];        // row_setup: the chaining value after P(8 nbytes), sha512/row/v2's first block
};
static_assert(sizeof(RowSpec) == 112, "RowSpec is 112 bytes");

struct LeafRow {     // 224 bytes: row g's frame leaf, rank and position in the domain dids[did]; its salt is row salt_row's
  u64 rank, position;
  u32 did, salt_row;
  u32 q, pad_;       // leaf_setup: the leaf prefix's bytes after its first block
  u64 mid[8];        // leaf_setup: the chaining value after that block
  u64 tail[16];      // leaf_setup: those q bytes, big-endian packed
};
static_assert(sizeof(LeafRow) == 224, "LeafRow is 224 bytes");

constexpr int SALT_WORDS = 24;   // hm96-sha512's 192-byte salt
constexpr int T_BYTES = 256 * 256;
constexpr int TSTRIDE = 272;     // bytes a T row in shared memory: 256 + 16, so rows start in different bank groups

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

// --- sha512/row/v2 -------------------------------------------------------------------------------------------------------

// row_setup: P(8 nbytes) = ("verity/sha512-row/v2\0" || u64be(8 nbytes)) zero-padded to 128 bytes, compressed from the IV
template <class C>
SX_HD void row_setup(RowSpec& r) {
  Msg m;
  msg_init(m);
  put_str(m, "verity/sha512-row/v2", true);
  put(m, 8 * (u64)r.nbytes, 8);
  sx::iv(r.mid);
  C::run(r.mid, m.w);
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
  const u32 n = r.nbytes, blocks = (n + 17 + 127) / 128, fast = r.a16 ? r.a_len / 128 : 0;
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
  for (u32 k = fast; k < blocks; ++k) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = row_word(a, r.a_len, b, n, 128 * k + 8 * i);
    if (k + 1 == blocks) w[15] = 8 * (128 + (u64)n);
    C::run(x, w);
  }
}

// --- hm96-sha512 under hiding leaves, then the frame-v3-sha512 leaf ------------------------------------------------------

// ChaCha20 (20 rounds, 64-bit counter in words 12-13, 64-bit nonce in 14-15: verity.ml.crypto.chacha20)
#define SXR_QR(a, b, c, d)                                                                                                 \
  x[a] += x[b]; x[d] ^= x[a]; x[d] = (x[d] << 16) | (x[d] >> 16);                                                          \
  x[c] += x[d]; x[b] ^= x[c]; x[b] = (x[b] << 12) | (x[b] >> 20);                                                          \
  x[a] += x[b]; x[d] ^= x[a]; x[d] = (x[d] << 8) | (x[d] >> 24);                                                           \
  x[c] += x[d]; x[b] ^= x[c]; x[b] = (x[b] << 7) | (x[b] >> 25);
SX_HD void chacha20(const u32* key, u64 counter, u64 nonce, u32* out) {
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

// y as 24 big-endian words: ChaCha20 blocks 3 salt_row .. 3 salt_row + 2 under key with nonce (C-Flock's hm96_salt)
SX_HD void salt_key_words(const u32* key, u64 nonce, u64 salt_row, u64* Y) {
  for (int c = 0; c < 3; ++c) {
    u32 o[16];
    chacha20(key, 3 * salt_row + c, nonce, o);
    for (int i = 0; i < 8; ++i) Y[8 * c + i] = sx::be64(o[2 * i], o[2 * i + 1]);
  }
}

struct LeafArgs {
  const LeafRow* rows;
  const u64* x;          // (G, S, 8) inner digests
  u64* out;              // (G, S, 8) leaves
  const uint8_t* T;      // hm96's Hankel table (engine.hankel_table): T_v, 256 rows of 256 bytes
  const uint8_t* salts;  // (G, S, 192) the caller's y, or null for the salt key's
  u32 key[8];            // the salt key (ChaCha20's key words), when salts is null
  u64 nonce0;            // slot s's nonce is nonce0 + s
  u64 ms_salt[8];        // the chaining value after hm96's salt prefix
  u64 ms_leaf[8];        // after its leaf prefix (the key's)
  u32 G, S;
  u32 minb;              // the device kernel's blocks an SM (its register cap): 1 or 2, 0 for the default
  u32 pad_;
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

// M_key y as 16 little-endian u32 (mask byte 4w + k is byte k of m[w]), from T (rows `stride` apart: TSTRIDE in shared
// memory, 256 on the host). Bytes [p, p + 64) of T_v are byte_tables(key)[p][v], so for salt bytes p = 16 q + r the 80 bytes
// [16 q, 16 q + 80) of T_{y_p} are xored into A_r, and A_r shifted down r bytes into m. Copied from commit_gpu's hm96_mask
// (benchmarks/pouw/commit_gpu/commit_gpu.cuh, its shared-memory path), with the stride an argument as in commit_slots'.
SX_HD void mask(const uint8_t* T, int stride, const u64* Y, u32* m) {
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
      const uint8_t* t = T + v * stride + 16 * q;
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

// row g's leaf in slot s: b = x ^ M y, c = SHA-512(salt_prefix || y), h = SHA-512(leaf_prefix(key) || b || c), then the frame
// leaf SHA-512(leaf prefix || h). T is A.T's table at `stride` (in shared memory on the device).
template <class C>
SX_HD void leaf(const LeafArgs& A, const uint8_t* T, int stride, u32 g, u32 s) {
  const LeafRow& r = A.rows[g];
  const u64 gs = (u64)g * A.S + s;
  u64 Y[SALT_WORDS];
  if (A.salts) {
    const uint8_t* y = A.salts + 192 * gs;
    for (int i = 0; i < 12; ++i) {
      const uint4 u = ld16(y + 16 * i);
      Y[2 * i] = sx::be64(u.x, u.y);
      Y[2 * i + 1] = sx::be64(u.z, u.w);
    }
  } else {
    salt_key_words(A.key, A.nonce0 + s, r.salt_row, Y);
  }
  u64 bc[16];
  {
    u32 m[16];
    mask(T, stride, Y, m);
    for (int i = 0; i < 8; ++i) bc[i] = ld8(A.x + 8 * gs + i) ^ ((u64)sx::bswap32(m[2 * i]) << 32 | sx::bswap32(m[2 * i + 1]));
  }
  u64 w[16];
  u64* c = bc + 8;
  for (int i = 0; i < 8; ++i) c[i] = A.ms_salt[i];
  for (int i = 0; i < 16; ++i) w[i] = Y[i];
  C::run(c, w);
  for (int i = 0; i < 8; ++i) w[i] = Y[16 + i];
  w[8] = 0x8000000000000000ull;
  for (int i = 9; i < 15; ++i) w[i] = 0;
  w[15] = (128 + 192) * 8;
  C::run(c, w);
  u64 h[8];
  for (int i = 0; i < 8; ++i) h[i] = A.ms_leaf[i];
  for (int i = 0; i < 16; ++i) w[i] = bc[i];
  C::run(h, w);
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = 0x8000000000000000ull;
  w[15] = (128 + 128) * 8;
  C::run(h, w);
  Msg m;
  msg_init(m);
  put_words(m, r.tail, (int)r.q);
  for (int i = 0; i < 8; ++i) put(m, h[i], 8);
  u64 out[8];
  for (int i = 0; i < 8; ++i) out[i] = r.mid[i];
  msg_hash<C>(m, 128, out);
  for (int i = 0; i < 8; ++i) A.out[8 * gs + i] = out[i];
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

// CommitmentDomain.node(level, index, left, right), from node_mid's chaining value
template <class C>
SX_HD void node(const u64* mid, u64 index, const u64* left, const u64* right, u64* out) {
  Msg m;
  msg_init(m);
  const int n = uint_len(index);
  put(m, (u64)n, 2);
  put(m, index, n);
  put_digest(m, left);
  put_digest(m, right);
  for (int i = 0; i < 8; ++i) out[i] = mid[i];
  msg_hash<C>(m, 128, out);
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
