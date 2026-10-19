// The served window's tables on the device, bit for bit `served_h1/commit.py`'s: hm96-sha512 hiding leaves
// (hm96-sha512/row-seg/v1) in one frame-v3-sha512 tree a table, every table a served step commits (each call's A table, its
// admit words and its receipts; the a-step, admit-step and round trees over their roots) and the install's B tables.  Plain
// launches on the caller's stream into caller-owned buffers, the step read from device memory, so a step's sequence is the
// same launches every step and a CUDA graph captures it.
//
//   h1c_domains  one thread a table: its domain id (`CommitmentDomain.domain_id`) from its fields' canonical JSON, whose
//                step-dependent values (the step, a receipt table's base) are spliced in on the device (v = a s + b)
//   h1c_segs     one thread a (table, row, segment): d_j = SHA-512(S(j, s, m_j) || segment j)            (sha512/row-seg/v1)
//   h1c_leaves   one thread a (table, rank < width): a row's leaf, x = SHA-512(Q(n_bits) || d_0 || ..), its salt y
//                (`commit.salt`: SHA-512(SALT_TAG || key || u32be len(label) || label || u64be i || u8 c), c < 3), b = x ^
//                M_key y (hm96's byte tables), c = SHA-512(salt_prefix || y), h = SHA-512(leaf_prefix(key) || b || c), the
//                frame leaf (rank, rank, "hm96-sha512/row-seg/v1", h); a pad above the count; a given table's leaves are
//                another table's roots, so only its pads
//   h1c_level    one thread a node of level d of every table (`merkle.tree_levels`); the top one is the table's root
//   h1c_a_*      a served call's A table in one launch, its row-independent parts once a step (below)
//
// A row is `body` bytes at data + i stride, then len - body bytes at tail + i tail_stride (an A or B row's f32 scale).
#include <cstdint>
#include <cuda_runtime.h>

namespace sh {
typedef unsigned long long u64;
#include "../../codec_tensor/sha512.cuh"
}  // namespace sh
typedef unsigned long long u64;
typedef unsigned int u32;

struct H1Const {
  u64 salt_prefix[16];  // hm96's salt_prefix block, big-endian words
  u64 leaf_prefix[16];  // leaf_prefix(KEY)
  uint8_t head[128];    // SALT_TAG || the window's key
  uint8_t frame[64];    // merkle._FRAME
  uint8_t schema[32];   // "hm96-sha512/row-seg/v1"
  uint8_t qtag[48];     // rowleaf.SHA512_ROW_SEG_TAG
  uint8_t stag[48];     // rowleaf.SHA512_ROW_SEG_SEGMENT_TAG
  int head_len, frame_len, schema_len, qtag_len, stag_len;
};
__constant__ H1Const HC;

struct H1Dom {
  const uint8_t* bind;    // _TAGGED_SHA512_PREFIX || u32be len(tag) || tag
  const uint8_t* js[3];   // the canonical JSON's literal pieces around the spliced values
  int bind_len, js_len[3];
  int nval;               // 0, 1 or 2 values spliced
  u64 va[2], vb[2];       // value v = va s + vb
  u64 pos[8];             // RangeIndexedDomain(0, count).identity_sha512()
};

struct H1Tab {
  const uint8_t* data;  // null: a given table (its leaves are other tables' roots)
  const uint8_t* tail;
  long long stride, tail_stride;
  const uint8_t* lab[2];  // the label: lab[0] || dec(s) || lab[1] (labs 0: lab[0] alone)
  int lab_len[2], labs;
  int len, body, count, depth;
  H1Dom dm;
  u64* dom;    // 8 words, h1c_domains'
  u64* level;  // 2 width - 1 digests: level 0, then each level up; the root last
  u64* segs;   // count * nseg digests
  u64* root;   // 8 words
};

// -- SHA-512 over a byte stream ----------------------------------------------------------------------------------------
struct Hs {
  u64 h[8], w[16];
  int n;
  u64 tot;
};
__device__ __forceinline__ void hs_init(Hs& s) {
#pragma unroll
  for (int j = 0; j < 8; ++j) s.h[j] = sh::kSha512Iv[j];
#pragma unroll
  for (int j = 0; j < 16; ++j) s.w[j] = 0;
  s.n = 0;
  s.tot = 0;
}
__device__ __forceinline__ void hs_flush(Hs& s) {
  sh::sha512_compress(s.h, s.w);
#pragma unroll
  for (int j = 0; j < 16; ++j) s.w[j] = 0;
  s.n = 0;
}
__device__ void hs_u8(Hs& s, u32 b) {
  s.w[s.n >> 3] |= (u64)(b & 0xFFu) << (56 - 8 * (s.n & 7));
  ++s.tot;
  if (++s.n == 128) hs_flush(s);
}
__device__ void hs_be(Hs& s, u64 x, int nbytes) {
  for (int i = nbytes - 1; i >= 0; --i) hs_u8(s, (u32)(x >> (8 * i)));
}
__device__ void hs_words(Hs& s, const u64* x, int nw) {
  for (int k = 0; k < nw; ++k) {
    if ((s.n & 7) == 0) {
      s.w[s.n >> 3] = x[k];
      s.n += 8;
      s.tot += 8;
      if (s.n == 128) hs_flush(s);
    } else {
      hs_be(s, x[k], 8);
    }
  }
}
__device__ void hs_bytes(Hs& s, const uint8_t* p, int n) {
  for (int i = 0; i < n; ++i) hs_u8(s, p[i]);
}
__device__ void hs_zeros(Hs& s, int n) {
  for (int i = 0; i < n; ++i) hs_u8(s, 0);
}
// merkle._uint: u64be len || the value's minimal big-endian bytes (one byte for 0)
__device__ void hs_uint(Hs& s, u64 v) {
  int l = 1;
  while (l < 8 && (v >> (8 * l))) ++l;
  hs_be(s, (u64)l, 8);
  hs_be(s, v, l);
}
__device__ void hs_final(Hs& s, u64 out[8]) {
  const u64 bits = 8 * s.tot;
  s.w[s.n >> 3] |= 0x80ull << (56 - 8 * (s.n & 7));
  ++s.n;
  if (s.n > 112) hs_flush(s);
  s.w[15] = bits;
  sh::sha512_compress(s.h, s.w);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[j] = s.h[j];
}
// merkle._message's head: _FRAME || u32be len(tag) || tag
__device__ void hs_frame(Hs& s, const char* tag, int tl) {
  hs_bytes(s, HC.frame, HC.frame_len);
  hs_be(s, (u64)tl, 4);
  for (int i = 0; i < tl; ++i) hs_u8(s, (u32)(uint8_t)tag[i]);
}
__device__ int dec(u64 v, uint8_t d[20]) {
  int n = 0;
  uint8_t r[20];
  do {
    r[n++] = (uint8_t)('0' + v % 10);
    v /= 10;
  } while (v);
  for (int i = 0; i < n; ++i) d[i] = r[n - 1 - i];
  return n;
}

__device__ __forceinline__ u32 bswap32(u32 x) { return __byte_perm(x, 0, 0x0123); }

__device__ __forceinline__ uint8_t row_byte(const H1Tab& T, long long i, int p) {
  return p < T.body ? T.data[i * T.stride + p] : T.tail[i * T.tail_stride + (p - T.body)];
}

// bytes [off, off + m) of row i
__device__ void hs_row(Hs& s, const H1Tab& T, long long i, int off, int m) {
  int p = off;
  const int end = off + m;
  const uint8_t* base = T.data + i * T.stride;
  while (p < end) {
    if (s.n == 0 && end - p >= 128 && p + 128 <= T.body && ((reinterpret_cast<uintptr_t>(base + p) & 15) == 0)) {
      const uint4* q = reinterpret_cast<const uint4*>(base + p);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const uint4 v = q[k];
        s.w[2 * k] = (u64)bswap32(v.x) << 32 | bswap32(v.y);
        s.w[2 * k + 1] = (u64)bswap32(v.z) << 32 | bswap32(v.w);
      }
      s.tot += 128;
      hs_flush(s);
      p += 128;
    } else {
      hs_u8(s, row_byte(T, i, p));
      ++p;
    }
  }
}

__host__ __device__ __forceinline__ int nseg_of(int len) { return (len + 1023) / 1024; }

extern "C" __global__ void h1c_domains(H1Tab* __restrict__ tabs, int ntab, const int* __restrict__ step) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= ntab) return;
  const H1Tab& T = tabs[t];
  const H1Dom& D = T.dm;
  const u64 st = (u64)(long long)*step;
  uint8_t d[2][20];
  int dl[2] = {0, 0};
  long long jl = D.js_len[0] + D.js_len[1] + D.js_len[2];
  for (int v = 0; v < D.nval; ++v) {
    dl[v] = dec(D.va[v] * st + D.vb[v], d[v]);
    jl += dl[v];
  }
  Hs s;
  hs_init(s);
  hs_bytes(s, D.bind, D.bind_len);
  hs_be(s, (u64)jl, 8);
  hs_bytes(s, D.js[0], D.js_len[0]);
  for (int v = 0; v < D.nval; ++v) {
    hs_bytes(s, d[v], dl[v]);
    hs_bytes(s, D.js[v + 1], D.js_len[v + 1]);
  }
  u64 binding[8];
  hs_final(s, binding);
  hs_init(s);
  hs_frame(s, "domain", 6);
  hs_be(s, 64, 8);
  hs_words(s, binding, 8);
  hs_uint(s, 1);  // owner -1, as _uint(owner + 2)
  hs_be(s, 64, 8);
  hs_words(s, D.pos, 8);
  hs_uint(s, (u64)T.count);
  hs_final(s, T.dom);
}

extern "C" __global__ void h1c_segs(const H1Tab* __restrict__ tabs, int ntab) {
  const H1Tab& T = tabs[blockIdx.y];
  if (!T.data) return;
  const int ns = nseg_of(T.len);
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)T.count * ns) return;
  const long long i = t / ns;
  const int j = (int)(t % ns), off = 1024 * j, m = min(1024, T.len - off);
  Hs s;
  hs_init(s);
  hs_bytes(s, HC.stag, HC.stag_len);
  hs_be(s, (u64)j, 8);
  hs_be(s, (u64)ns, 8);
  hs_be(s, 8ull * m, 8);
  hs_zeros(s, 128 - HC.stag_len - 24);
  hs_row(s, T, i, off, m);
  hs_final(s, T.segs + t * 8);
}

extern "C" __global__ void h1c_leaves(const H1Tab* __restrict__ tabs, int ntab, const int* __restrict__ step,
                                      const u64* __restrict__ btab) {
  const H1Tab& T = tabs[blockIdx.y];
  const long long width = 1LL << T.depth;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width) return;
  Hs s;
  u64 out[8];
  if (i >= T.count) {
    hs_init(s);
    hs_frame(s, "pad", 3);
    hs_be(s, 64, 8);
    hs_words(s, T.dom, 8);
    hs_uint(s, (u64)i);
    hs_final(s, out);
  } else {
    if (!T.data) return;
    const int ns = nseg_of(T.len);
    u64 x[8], y[24], mk[8], c[8], h[8];
    hs_init(s);
    hs_bytes(s, HC.qtag, HC.qtag_len);
    hs_be(s, 8ull * T.len, 8);
    hs_be(s, (u64)ns, 8);
    hs_be(s, 1024, 4);
    hs_zeros(s, 128 - HC.qtag_len - 20);
    hs_words(s, T.segs + i * ns * 8, 8 * ns);
    hs_final(s, x);
    uint8_t d[20];
    int dl = 0;
    if (T.labs) dl = dec((u64)(long long)*step, d);
    const int ll = T.lab_len[0] + (T.labs ? dl + T.lab_len[1] : 0);
    for (int cc = 0; cc < 3; ++cc) {
      hs_init(s);
      hs_bytes(s, HC.head, HC.head_len);
      hs_be(s, (u64)ll, 4);
      hs_bytes(s, T.lab[0], T.lab_len[0]);
      if (T.labs) {
        hs_bytes(s, d, dl);
        hs_bytes(s, T.lab[1], T.lab_len[1]);
      }
      hs_be(s, (u64)i, 8);
      hs_u8(s, (u32)cc);
      hs_final(s, y + 8 * cc);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) mk[k] = 0;
    for (int p = 0; p < 192; ++p) {
      const u32 v = (u32)(y[p >> 3] >> (56 - 8 * (p & 7))) & 0xFFu;
      const uint4* e = reinterpret_cast<const uint4*>(btab + ((size_t)p * 256 + v) * 8);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint4 q = __ldg(e + k);
        mk[2 * k] ^= (u64)q.y << 32 | q.x;
        mk[2 * k + 1] ^= (u64)q.w << 32 | q.z;
      }
    }
    hs_init(s);
    hs_words(s, HC.salt_prefix, 16);
    hs_words(s, y, 24);
    hs_final(s, c);
#pragma unroll
    for (int k = 0; k < 8; ++k) mk[k] ^= x[k];
    hs_init(s);
    hs_words(s, HC.leaf_prefix, 16);
    hs_words(s, mk, 8);
    hs_words(s, c, 8);
    hs_final(s, h);
    hs_init(s);
    hs_frame(s, "leaf", 4);
    hs_be(s, 64, 8);
    hs_words(s, T.dom, 8);
    hs_uint(s, (u64)i);
    hs_uint(s, (u64)i);
    hs_be(s, (u64)HC.schema_len, 8);
    hs_bytes(s, HC.schema, HC.schema_len);
    hs_be(s, 64, 8);
    hs_words(s, h, 8);
    hs_final(s, out);
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) T.level[i * 8 + k] = out[k];
  if (width == 1)
#pragma unroll
    for (int k = 0; k < 8; ++k) T.root[k] = out[k];
}

extern "C" __global__ void h1c_level(const H1Tab* __restrict__ tabs, int ntab, int lv) {
  const H1Tab& T = tabs[blockIdx.y];
  if (lv >= T.depth) return;
  const long long width = 1LL << T.depth, nodes = width >> (lv + 1);
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nodes) return;
  long long off = 0;
  for (int e = 0; e < lv; ++e) off += width >> e;
  const u64* in = T.level + 8 * (off + 2 * idx);
  u64* o = T.level + 8 * (off + (width >> lv) + idx);
  Hs s;
  hs_init(s);
  hs_frame(s, "node", 4);
  hs_be(s, 64, 8);
  hs_words(s, T.dom, 8);
  hs_uint(s, (u64)lv);
  hs_uint(s, (u64)idx);
  hs_be(s, 64, 8);
  hs_words(s, in, 8);
  hs_be(s, 64, 8);
  hs_words(s, in + 8, 8);
  u64 out[8];
  hs_final(s, out);
#pragma unroll
  for (int k = 0; k < 8; ++k) o[k] = out[k];
  if (lv == T.depth - 1)
#pragma unroll
    for (int k = 0; k < 8; ++k) T.root[k] = out[k];
}

// -- a served call's A table in one launch -------------------------------------------------------------------------------
// The same table as h1c_build's, split by what it reads.  Once a window (h1c_a_init): each segment's first block and x's
// (they read only the row length), and x of the canonical empty row.  Once a step, before its forward (h1c_a_step: the
// domains, then h1c_a_pre): what reads the step but no row (each row's salts y, M_key y and c, its leaf frame up to h, each
// level's node prefix) and every leaf and node over rows >= m alone (the canonical empty row's).  Per call (h1c_a_call): the
// slot rows, then the live rows' segments, x, h and leaves, and the nodes over a live row by the block that arrives last;
// its last level writes the root.
//
// Hb is SHA-512 with its state in registers and the block being filled in the thread's HB-byte stride of shared memory
// (33 words: threads filling the same offset hit distinct banks); Hs's dynamic index into its words puts its whole state
// in local memory.  A Snap is a stream's state after a prefix (h, the bytes absorbed, the partial block), to resume from.
constexpr int HB = 132;
constexpr int SNAP = 25;  // words: h[8], tot, the partial block's 16 words
constexpr int A_NSEG = 32, A_DEPTH = 16, AW = 4;
// a table's words of `pre`: x's first block, x of the empty row, h's first block (leaf_prefix), c's (salt_prefix), each
// level's node prefix, each segment's first block, the call's arrival counter, then a row's leaf frame prefix, M_key y, c
enum : int {
  A_QS = 0,
  A_XE = A_QS + SNAP,
  A_LPS = A_XE + 8,
  A_SPS = A_LPS + SNAP,
  A_NS = A_SPS + SNAP,
  A_SS = A_NS + SNAP * A_DEPTH,
  A_CNT = A_SS + SNAP * A_NSEG,
  A_ROW = A_CNT + 1,
  A_RW = SNAP + 16
};

struct Hb {
  u64 h[8];
  uint8_t* b;
  int n;
  u64 tot;
};
__device__ __forceinline__ void hb_init(Hb& s, uint8_t* b) {
#pragma unroll
  for (int j = 0; j < 8; ++j) s.h[j] = sh::kSha512Iv[j];
  s.b = b;
  s.n = 0;
  s.tot = 0;
}
// one block, not inlined: every flush site shares one compression's code
__device__ __noinline__ void hb_block(u64* h, const uint8_t* b) {
  u64 w[16], v[8];
  const u32* q = reinterpret_cast<const u32*>(b);
#pragma unroll
  for (int k = 0; k < 16; ++k) w[k] = (u64)bswap32(q[2 * k]) << 32 | bswap32(q[2 * k + 1]);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = h[j];
  sh::sha512_compress(v, w);
#pragma unroll
  for (int j = 0; j < 8; ++j) h[j] = v[j];
}
__device__ __forceinline__ void hb_flush(Hb& s) {
  hb_block(s.h, s.b);
  s.n = 0;
}
__device__ __forceinline__ void hb_u8(Hb& s, u32 v) {
  s.b[s.n] = (uint8_t)v;
  ++s.tot;
  if (++s.n == 128) hb_flush(s);
}
__device__ __forceinline__ void hb_be(Hb& s, u64 x, int nbytes) {
  for (int i = nbytes - 1; i >= 0; --i) hb_u8(s, (u32)(x >> (8 * i)));
}
// one big-endian word at any offset
__device__ __forceinline__ void hb_put8(Hb& s, u64 v) {
  if ((s.n & 7) == 0) {
    u32* q = reinterpret_cast<u32*>(s.b + s.n);
    q[0] = bswap32((u32)(v >> 32));
    q[1] = bswap32((u32)v);
  } else if (s.n <= 120) {
    uint8_t* p = s.b + s.n;
#pragma unroll
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (56 - 8 * i));
  } else {
    hb_be(s, v, 8);
    return;
  }
  s.n += 8;
  s.tot += 8;
  if (s.n == 128) hb_flush(s);
}
__device__ __forceinline__ void hb_words(Hb& s, const u64* x, int nw) {
  for (int k = 0; k < nw; ++k) hb_put8(s, x[k]);
}
__device__ __forceinline__ void hb_bytes(Hb& s, const uint8_t* p, int n) {
  for (int i = 0; i < n; ++i) hb_u8(s, p[i]);
}
__device__ __forceinline__ void hb_zeros(Hb& s, int n) {
  for (int i = 0; i < n; ++i) hb_u8(s, 0);
}
__device__ __forceinline__ void hb_uint(Hb& s, u64 v) {
  int l = 1;
  while (l < 8 && (v >> (8 * l))) ++l;
  hb_be(s, (u64)l, 8);
  hb_be(s, v, l);
}
__device__ __forceinline__ void hb_frame(Hb& s, const char* tag, int tl) {
  hb_bytes(s, HC.frame, HC.frame_len);
  hb_be(s, (u64)tl, 4);
  for (int i = 0; i < tl; ++i) hb_u8(s, (u32)(uint8_t)tag[i]);
}
__device__ __forceinline__ void hb_final(Hb& s, u64* out) {
  const u64 bits = 8 * s.tot;
  s.b[s.n++] = 0x80;
  if (s.n > 112) {
    while (s.n < 128) s.b[s.n++] = 0;
    hb_flush(s);
  }
  while (s.n & 3) s.b[s.n++] = 0;
  for (; s.n < 120; s.n += 4) *reinterpret_cast<u32*>(s.b + s.n) = 0;
  u32* q = reinterpret_cast<u32*>(s.b);
  q[30] = bswap32((u32)(bits >> 32));
  q[31] = bswap32((u32)bits);
  hb_flush(s);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[j] = s.h[j];
}
__device__ __forceinline__ void hb_save(const Hb& s, u64* o) {
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = s.h[j];
  o[8] = s.tot;
  uint8_t* ob = reinterpret_cast<uint8_t*>(o + 9);
  for (int i = 0; i < s.n; ++i) ob[i] = s.b[i];
}
__device__ __forceinline__ void hb_load(Hb& s, const u64* o, uint8_t* b) {
#pragma unroll
  for (int j = 0; j < 8; ++j) s.h[j] = o[j];
  s.tot = o[8];
  s.n = (int)(s.tot & 127);
  s.b = b;
  const u32* ib = reinterpret_cast<const u32*>(o + 9);
  for (int i = 0; 4 * i < s.n; ++i) reinterpret_cast<u32*>(b)[i] = ib[i];
}

// node idx of level lv (its first block from the step's prefix), children read through L2 (another block wrote them)
__device__ __forceinline__ void a_node(Hb& s, uint8_t* b, const u64* P, const H1Tab& T, int lv, long long idx,
                                       long long off) {
  const long long width = 1LL << T.depth;
  const u64* in = T.level + 8 * (off + 2 * idx);
  u64* o = T.level + 8 * (off + (width >> lv) + idx);
  u64 c[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) c[k] = __ldcg(in + k);
  hb_load(s, P + A_NS + SNAP * lv, b);
  int l = 1;
  while (l < 8 && (idx >> (8 * l))) ++l;
  hb_u8(s, (u32)l);  // uint(idx)'s length's low byte: the prefix holds its other seven
  hb_be(s, (u64)idx, l);
  hb_put8(s, 64);
  hb_words(s, c, 8);
  hb_put8(s, 64);
  hb_words(s, c + 8, 8);
  u64 out[8];
  hb_final(s, out);
#pragma unroll
  for (int k = 0; k < 8; ++k) o[k] = out[k];
  if (lv == T.depth - 1)
#pragma unroll
    for (int k = 0; k < 8; ++k) T.root[k] = out[k];
}

// row i's leaf from its x (h = SHA-512(leaf_prefix || x ^ M_key y || c), then the frame leaf from the row's prefix)
__device__ __forceinline__ void a_leaf(Hb& s, uint8_t* b, const u64* P, long long i, const u64 x[8], u64 out[8]) {
  const u64* R = P + A_ROW + (long long)A_RW * i;
  u64 mk[8], h[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) mk[k] = x[k] ^ R[SNAP + k];
  hb_load(s, P + A_LPS, b);
  hb_words(s, mk, 8);
  hb_words(s, R + SNAP + 8, 8);
  hb_final(s, h);
  hb_load(s, R, b);
  hb_words(s, h, 8);
  hb_final(s, out);
}

extern "C" __global__ void __launch_bounds__(32) h1c_a_init_k(const H1Tab* __restrict__ tabs, u64* __restrict__ pre, int wpt) {
  __shared__ __align__(16) uint8_t bf[32 * HB];
  __shared__ u64 dg[A_NSEG][8];
  const H1Tab& T = tabs[blockIdx.x];
  u64* P = pre + (long long)blockIdx.x * wpt;
  const int lane = threadIdx.x, ns = nseg_of(T.len);
  Hb s;
  uint8_t* b = bf + lane * HB;
  if (lane < ns) {
    const int m = min(1024, T.len - 1024 * lane);
    hb_init(s, b);
    hb_bytes(s, HC.stag, HC.stag_len);
    hb_be(s, (u64)lane, 8);
    hb_be(s, (u64)ns, 8);
    hb_be(s, 8ull * m, 8);
    hb_zeros(s, 128 - HC.stag_len - 24);
    hb_save(s, P + A_SS + SNAP * lane);
    hb_zeros(s, m);
    hb_final(s, dg[lane]);
  }
  __syncwarp();
  hb_init(s, b);
  if (lane == 0) {
    hb_bytes(s, HC.qtag, HC.qtag_len);
    hb_be(s, 8ull * T.len, 8);
    hb_be(s, (u64)ns, 8);
    hb_be(s, 1024, 4);
    hb_zeros(s, 128 - HC.qtag_len - 20);
    hb_save(s, P + A_QS);
    hb_words(s, &dg[0][0], 8 * ns);
    hb_final(s, P + A_XE);
  } else if (lane == 1) {
    hb_words(s, HC.leaf_prefix, 16);
    hb_save(s, P + A_LPS);
  } else if (lane == 2) {
    hb_words(s, HC.salt_prefix, 16);
    hb_save(s, P + A_SPS);
  }
}

// S: the step's H1OStep (parity, m, pad, flags)
extern "C" __global__ void __launch_bounds__(64) h1c_a_pre_k(const H1Tab* __restrict__ tabs, u64* __restrict__ pre, int wpt,
                                                             const int* __restrict__ step, const u64* __restrict__ btab,
                                                             const int* __restrict__ S) {
  __shared__ __align__(16) uint8_t bf[64 * HB];
  const H1Tab& T = tabs[blockIdx.x];
  u64* P = pre + (long long)blockIdx.x * wpt;
  const int tid = threadIdx.x, nt = blockDim.x;
  const long long m = S[1], width = 1LL << T.depth;
  uint8_t* b = bf + tid * HB;
  Hb s;
  u64 dom[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) dom[k] = T.dom[k];
  for (int lv = tid; lv < T.depth; lv += nt) {
    hb_init(s, b);
    hb_frame(s, "node", 4);
    hb_be(s, 64, 8);
    hb_words(s, dom, 8);
    hb_uint(s, (u64)lv);
    hb_zeros(s, 7);
    hb_save(s, P + A_NS + SNAP * lv);
  }
  uint8_t d[20];
  const int dl = T.labs ? dec((u64)(long long)*step, d) : 0;
  const int ll = T.lab_len[0] + (T.labs ? dl + T.lab_len[1] : 0);
  for (long long i = tid; i < T.count; i += nt) {
    u64* R = P + A_ROW + (long long)A_RW * i;
    u64 y[24], mk[8];
    for (int cc = 0; cc < 3; ++cc) {
      hb_init(s, b);
      hb_bytes(s, HC.head, HC.head_len);
      hb_be(s, (u64)ll, 4);
      hb_bytes(s, T.lab[0], T.lab_len[0]);
      if (T.labs) {
        hb_bytes(s, d, dl);
        hb_bytes(s, T.lab[1], T.lab_len[1]);
      }
      hb_be(s, (u64)i, 8);
      hb_u8(s, (u32)cc);
      hb_final(s, y + 8 * cc);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) mk[k] = 0;
    for (int p = 0; p < 192; ++p) {
      const u32 v = (u32)(y[p >> 3] >> (56 - 8 * (p & 7))) & 0xFFu;
      const uint4* e = reinterpret_cast<const uint4*>(btab + ((size_t)p * 256 + v) * 8);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint4 q = __ldg(e + k);
        mk[2 * k] ^= (u64)q.y << 32 | q.x;
        mk[2 * k + 1] ^= (u64)q.w << 32 | q.z;
      }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) R[SNAP + k] = mk[k];
    hb_load(s, P + A_SPS, b);
    hb_words(s, y, 24);
    hb_final(s, R + SNAP + 8);
    hb_init(s, b);
    hb_frame(s, "leaf", 4);
    hb_be(s, 64, 8);
    hb_words(s, dom, 8);
    hb_uint(s, (u64)i);
    hb_uint(s, (u64)i);
    hb_be(s, (u64)HC.schema_len, 8);
    hb_bytes(s, HC.schema, HC.schema_len);
    hb_be(s, 64, 8);
    hb_save(s, R);
  }
  __syncthreads();
  for (long long i = m + tid; i < width; i += nt) {
    u64 out[8];
    if (i >= T.count) {
      hb_init(s, b);
      hb_frame(s, "pad", 3);
      hb_be(s, 64, 8);
      hb_words(s, dom, 8);
      hb_uint(s, (u64)i);
      hb_final(s, out);
    } else {
      a_leaf(s, b, P, i, P + A_XE, out);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) T.level[i * 8 + k] = out[k];
  }
  long long off = 0;
  for (int lv = 0; lv < T.depth; ++lv) {
    __threadfence_block();
    __syncthreads();
    const long long nodes = width >> (lv + 1), lo = (m + (2LL << lv) - 1) >> (lv + 1);
    for (long long idx = lo + tid; idx < nodes; idx += nt) a_node(s, b, P, T, lv, idx, off);
    off += width >> lv;
  }
  if (tid == 0) *reinterpret_cast<unsigned*>(P + A_CNT) = 0;
}

// one call: block x's AW warps take slot rows AW x .. AW x + AW - 1 (row < m: xq's bytes and sa's scale, else the
// canonical empty row; neither written under F_NW_TILES, bit 2 of S's flags); a live row's lane j hashes segment j from
// xq and sa (the bytes the slot gets), lane 0 its x, h and leaf; the last block to arrive builds the nodes over live rows
extern "C" __global__ void __launch_bounds__(32 * AW) h1c_a_call_k(const H1Tab* __restrict__ tab, u64* __restrict__ P,
                                                                   const int* __restrict__ S, const int8_t* __restrict__ xq,
                                                                   const float* __restrict__ sa, int8_t* __restrict__ slot,
                                                                   float* __restrict__ scale, int k) {
  __shared__ __align__(16) uint8_t bf[32 * AW * HB];
  __shared__ u64 dg[AW][A_NSEG][8];
  __shared__ int last;
  const H1Tab& T = *tab;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const long long r = (long long)blockIdx.x * AW + wid;
  const long long m = S[1];
  const bool nw = (S[3] & 2) != 0;
  uint8_t* b = bf + threadIdx.x * HB;
  Hb s;
  if (r < T.count) {
    const bool alive = r < m;
    if (!nw) {
      uint4* dst = reinterpret_cast<uint4*>(slot + r * k);
      const uint4* src = reinterpret_cast<const uint4*>(xq + (alive ? r : 0) * k);
      for (int i = lane; i < k / 16; i += 32) dst[i] = alive ? src[i] : make_uint4(0, 0, 0, 0);
      if (lane == 0) scale[r] = alive ? sa[r] : 0.0f;
    }
    if (alive) {
      const int ns = nseg_of(T.len);
      const uint8_t* row = reinterpret_cast<const uint8_t*>(xq + r * k);
      const uint8_t* tl = reinterpret_cast<const uint8_t*>(sa + r);
      if (lane < ns) {
        const int p0 = 1024 * lane, end = min(T.len, p0 + 1024);
        hb_load(s, P + A_SS + SNAP * lane, b);
        int p = p0;
        if (s.n == 0 && p + 128 <= end && p + 128 <= k) {
          uint4 nx[8];
          const uint4* q = reinterpret_cast<const uint4*>(row + p);
#pragma unroll
          for (int j = 0; j < 8; ++j) nx[j] = __ldg(q + j);
          while (true) {
            u64 w[16];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              w[2 * j] = (u64)bswap32(nx[j].x) << 32 | bswap32(nx[j].y);
              w[2 * j + 1] = (u64)bswap32(nx[j].z) << 32 | bswap32(nx[j].w);
            }
            p += 128;
            const bool more = p + 128 <= end && p + 128 <= k;
            if (more) {
              q = reinterpret_cast<const uint4*>(row + p);
#pragma unroll
              for (int j = 0; j < 8; ++j) nx[j] = __ldg(q + j);
            }
            sh::sha512_compress(s.h, w);
            s.tot += 128;
            if (!more) break;
          }
        }
        for (; p < end; ++p) hb_u8(s, p < k ? row[p] : tl[p - k]);
        hb_final(s, dg[wid][lane]);
      }
      __syncwarp();
      if (lane == 0) {
        u64 x[8], out[8];
        hb_load(s, P + A_QS, b);
        hb_words(s, &dg[wid][0][0], 8 * ns);
        hb_final(s, x);
        a_leaf(s, b, P, r, x, out);
#pragma unroll
        for (int j = 0; j < 8; ++j) T.level[8 * r + j] = out[j];
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(reinterpret_cast<unsigned*>(P + A_CNT), 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const long long width = 1LL << T.depth;
  long long off = 0;
  for (int lv = 0; lv < T.depth; ++lv) {
    const long long nodes = width >> (lv + 1), live = min(nodes, (m + (2LL << lv) - 1) >> (lv + 1));
    for (long long idx = threadIdx.x; idx < live; idx += blockDim.x) a_node(s, b, P, T, lv, idx, off);
    off += width >> lv;
    __threadfence_block();
    __syncthreads();
  }
  if (threadIdx.x == 0) *reinterpret_cast<unsigned*>(P + A_CNT) = 0;
}

// -- host ----------------------------------------------------------------------------------------------------------------
extern "C" int h1c_const(const H1Const* c) { return (int)cudaMemcpyToSymbol(HC, c, sizeof(H1Const)); }

extern "C" int h1c_a_words(int t) { return A_ROW + A_RW * t; }

// the window's part, once, on `stream`: ntab A tables (tabs_dev) into pre (wpt = h1c_a_words(T) words a table)
extern "C" int h1c_a_init(const H1Tab* tabs, const H1Tab* tabs_dev, int ntab, u64* pre, int wpt, cudaStream_t stream) {
  for (int t = 0; t < ntab; ++t) {
    if (!tabs[t].data || nseg_of(tabs[t].len) > A_NSEG || tabs[t].depth > A_DEPTH || tabs[t].depth < 1) return 100 + t;
    if (A_ROW + (long long)A_RW * tabs[t].count > wpt) return 300 + t;
  }
  h1c_a_init_k<<<ntab, 32, 0, stream>>>(tabs_dev, pre, wpt);
  return (int)cudaGetLastError();
}

// a step's part, before its forward: the A tables' domains at the step, then h1c_a_pre (S: the step's H1OStep, m read)
extern "C" int h1c_a_step(H1Tab* tabs_dev, int ntab, u64* pre, int wpt, const int* step_dev, const u64* btab, const int* S,
                          cudaStream_t stream) {
  h1c_domains<<<(ntab + 63) / 64, 64, 0, stream>>>(tabs_dev, ntab, step_dev);
  h1c_a_pre_k<<<ntab, 64, 0, stream>>>(tabs_dev, pre, wpt, step_dev, btab, S);
  return (int)cudaGetLastError();
}

// one call's A table (tab: its host copy, tab_dev the device's; pre: its words); xq [M >= m, k] s8 and sa [M] f32 the
// call's operands, slot [T, k] and scale [T] the table's rows
extern "C" int h1c_a_call(const H1Tab* tab, const H1Tab* tab_dev, u64* pre, const int* S, const int8_t* xq, const float* sa,
                          int8_t* slot, float* scale, int k, cudaStream_t stream) {
  if (k % 16 || tab->len != k + 4 || tab->body != k || nseg_of(tab->len) > A_NSEG || tab->depth > A_DEPTH) return 1;
  h1c_a_call_k<<<(unsigned)((tab->count + AW - 1) / AW), 32 * AW, 0, stream>>>(tab_dev, pre, S, xq, sa, slot, scale, k);
  return (int)cudaGetLastError();
}
extern "C" int h1c_sizes(int* out) {
  out[0] = (int)sizeof(H1Const);
  out[1] = (int)sizeof(H1Dom);
  out[2] = (int)sizeof(H1Tab);
  return 0;
}

// a set of tables (on the device, tabs_dev) through domains, segments, leaves and every level, on `stream`; the host copy
// (tabs) gives the grid's extents.  Tables whose leaves another set's roots write go in a later call.
extern "C" int h1c_build(const H1Tab* tabs, H1Tab* tabs_dev, int ntab, const int* step_dev, const u64* btab, int domains,
                         cudaStream_t stream) {
  if (ntab <= 0) return 0;
  long long max_segs = 0, max_width = 0;
  int max_depth = 0;
  for (int t = 0; t < ntab; ++t) {
    if (tabs[t].data) max_segs = max(max_segs, (long long)tabs[t].count * nseg_of(tabs[t].len));
    max_width = max(max_width, 1LL << tabs[t].depth);
    max_depth = max(max_depth, tabs[t].depth);
    if (tabs[t].count > (1LL << tabs[t].depth) || (tabs[t].count << 1) <= (1LL << tabs[t].depth)) {
      if (!(tabs[t].count == 1 && tabs[t].depth == 0)) return 100 + t;
    }
  }
  if (ntab > 65535) return 99;
  if (domains) h1c_domains<<<(ntab + 63) / 64, 64, 0, stream>>>(tabs_dev, ntab, step_dev);
  if (max_segs) h1c_segs<<<dim3((unsigned)((max_segs + 127) / 128), ntab), 128, 0, stream>>>(tabs_dev, ntab);
  h1c_leaves<<<dim3((unsigned)((max_width + 127) / 128), ntab), 128, 0, stream>>>(tabs_dev, ntab, step_dev, btab);
  for (int lv = 0; lv < max_depth; ++lv) {
    const long long nodes = max_width >> (lv + 1);
    h1c_level<<<dim3((unsigned)((nodes + 127) / 128), ntab), 128, 0, stream>>>(tabs_dev, ntab, lv);
  }
  return (int)cudaGetLastError();
}
