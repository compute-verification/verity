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
//   h1c_tail     (H1C_NMID) h1c_build of given tables, then of the given table over their roots: what reads no leaf
//                (domains, pads, each level's Snap) in h1c_tail_pre's launch, every level of both in h1c_tail's one
//   h1c_a_*      a served call's A table in one launch, its row-independent parts once a step (below); h1c_a_form(2)
//                serves it from h1c_a_call2_k, the same table with each compression on its critical path paired, and
//                h1c_a_form(3) from h1c_a_call3_k, paired with every message built in registers
//
// A row is `body` bytes at data + i stride, then len - body bytes at tail + (i / tdiv) tail_stride (an A or B row's f32
// scale; tdiv = n past K = 4096, where each of a K row's n chunk leaves ends in the row's one scale).
// H1C_HB = 1: h1c_build's four kernels hash through Hb (below: the state in registers, the block in shared memory) rather
// than Hs, the same bytes in the same order.  With it, H1C_LVF = L >= 0 builds every level from L up in one launch (a
// block a table, h1c_levels), H1C_DOMSEG = 1 the domains in the segments' launch (its first block row), beside them,
// H1C_MK = 1 a leaf's M_key y from M_key's columns, a block's stage of them in shared memory, not 192 byte-table rows, and
// H1C_MID = 1 (with H1C_MK) a leaf's x, c, h and frame hashes from the block's snapshots of their messages' constant
// heads (the table's x header and frame prefix, salt_prefix, leaf_prefix), each taken once a block, and H1C_LVB = 1
// (with H1C_MK) levels 0 to 6 in the leaves' launch, each block the subtree over its 128 leaves after them, so only
// the levels from 7 up are launches of their own.  H1C_LMINB = B (H1C_HB) asks the leaves kernel's registers to leave
// room for B blocks an SM (with H1C_MK and H1C_MID the leaves hold 200 registers on sm_120a, 160 on sm_90a and 192 on
// sm_100a; B = 4 caps them at 128 on all three, with 64 to 96 bytes more stack).  H1C_LT = N (H1C_LVF) gives h1c_levels'
// block N threads, 256 by default: at 128 registers a 256-CTA fits 2 an SM, so a build of more tables than
// twice the SMs runs its levels' serial chain in two waves.  H1C_APW = 1: h1c_a_pre_k reads a row's M_key byte-table
// rows a y word at a time, the word's 8 rows' loads issued before any is folded and y rotated in registers (as hb_mk),
// rather than one row after another with y indexed at run time.  H1C_SEGMID = 1 (H1C_HB): a segment's first block
// (stag, j, ns, 8 m_j) is the same in every row of its table, so where a table has at most SEGMID_MAX segments a row, a
// segments block compresses each of its segments' first blocks once (one lane each) and its threads start from them.
// H1C_SMINB = B: the segments kernel's registers leave room for B blocks an SM (with H1C_SEGMID it holds 120 registers
// on sm_120a, 4 blocks, against 96 without; 96 on sm_90a and sm_100a).  H1C_NMID = 1 (H1C_HB): a node's first 129 bytes
// (frame, "node", the table's domain, uint(lv) and seven of uint(idx)'s length's eight bytes) are its (table, level)'s,
// so h1c_levels' block takes each level's Snap once (a thread a level, before its first level) and h1c_level's block
// its level's, and every node resumes from it: two compressions a node rather than three, as h1c_a_* already do.
// H1C_MKN = 1 (H1C_MK): hb_mk folds M_key y a nibble at a time, a row of a nibble table each, not a column a bit.
// H1C_TAILW = 1 (H1C_NMID): h1c_tail's nodes hash from their messages built as words in registers (tail_node).
// H1C_SEGJ = 1 (H1C_HB): h1c_segs' thread t takes segment t / count of row t % count, not segment t % ns of row t / ns,
// so a warp's lanes hash segments of one length (a receipt row's 64-byte last segment no longer idles the lanes beside
// its 1024-byte first); each digest lands where it did.
#include <cstdint>
#include <cuda_runtime.h>

#ifndef H1C_HB
#define H1C_HB 0
#endif
#ifndef H1C_LVF
#define H1C_LVF -1
#endif
#ifndef H1C_DOMSEG
#define H1C_DOMSEG 0
#endif
#ifndef H1C_MK
#define H1C_MK 0
#endif
#ifndef H1C_MID
#define H1C_MID 0
#endif
#ifndef H1C_LVB
#define H1C_LVB 0
#endif
#ifndef H1C_LMINB
#define H1C_LMINB 1
#endif
#ifndef H1C_LT
#define H1C_LT 256
#endif
#ifndef H1C_APW
#define H1C_APW 0
#endif
#ifndef H1C_SEGMID
#define H1C_SEGMID 0
#endif
#ifndef H1C_NMID
#define H1C_NMID 0
#endif
#ifndef H1C_MKN
#define H1C_MKN 0
#endif
#ifndef H1C_TAILW
#define H1C_TAILW 0
#endif
#ifndef H1C_SEGJ
#define H1C_SEGJ 0
#endif
// unset, h1c_segs' bounds name no blocks an SM: __launch_bounds__(HT, 1) lets it take 136 registers on sm_90a, not 96
#ifdef H1C_SMINB
#define H1C_SEGS_BOUNDS __launch_bounds__(HT, H1C_SMINB)
#else
#define H1C_SEGS_BOUNDS __launch_bounds__(HT)
#endif
static_assert(H1C_HB || (H1C_LVF < 0 && !H1C_DOMSEG && !H1C_MK && !H1C_SEGMID && !H1C_NMID),
              "H1C_LVF, H1C_DOMSEG, H1C_MK, H1C_SEGMID and H1C_NMID are H1C_HB's");
static_assert(H1C_MK || !H1C_MID, "H1C_MID's snapshots are published at H1C_MK's barriers");
static_assert(H1C_MK || !H1C_MKN, "H1C_MKN is H1C_MK's M_key a nibble at a time");
static_assert(H1C_MK || !H1C_LVB, "H1C_LVB's subtrees are H1C_MK's leaves kernel's");
static_assert(H1C_HB || !H1C_SEGJ, "H1C_SEGJ is H1C_HB's segments kernel's");

#include "../../torch_gpu/sha512_rows/sha512x.cuh"
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
  // xd non-null: row i of a block whose rlive[i / rbw] is 0 holds the bytes the caller's dead row i % rbw does, so its x
  // is xd + 8 (i % rbw) and its segments are not hashed (their digests left unwritten); its salt and leaf as any row's
  const u64* xd;
  const uint8_t* rlive;
  int rbw;
  int tdiv;  // rows a tail: row i's tail is tail i / tdiv (0 or 1: its own)
};
__device__ __forceinline__ bool row_dead(const H1Tab& T, long long i) { return T.xd && !T.rlive[i / T.rbw]; }

// -- SHA-512 over a byte stream ----------------------------------------------------------------------------------------
struct Hs {
  u64 h[8], w[16];
  int n;
  u64 tot;
};
__device__ __forceinline__ void hs_init(Hs& s) {
  sx::iv(s.h);
#pragma unroll
  for (int j = 0; j < 16; ++j) s.w[j] = 0;
  s.n = 0;
  s.tot = 0;
}
__device__ __forceinline__ void hs_flush(Hs& s) {
  sx::compress<0, 2>(s.h, s.w);
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
  sx::compress<0, 2>(s.h, s.w);
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
  return p < T.body ? T.data[i * T.stride + p] : T.tail[(T.tdiv > 1 ? i / T.tdiv : i) * T.tail_stride + (p - T.body)];
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

#if !H1C_HB
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
  if (row_dead(T, i)) return;
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
    if (row_dead(T, i)) {
#pragma unroll
      for (int k = 0; k < 8; ++k) x[k] = T.xd[8 * (i % T.rbw) + k];
    } else {
      hs_init(s);
      hs_bytes(s, HC.qtag, HC.qtag_len);
      hs_be(s, 8ull * T.len, 8);
      hs_be(s, (u64)ns, 8);
      hs_be(s, 1024, 4);
      hs_zeros(s, 128 - HC.qtag_len - 20);
      hs_words(s, T.segs + i * ns * 8, 8 * ns);
      hs_final(s, x);
    }
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
#endif

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
  sx::iv(s.h);
  s.b = b;
  s.n = 0;
  s.tot = 0;
}

// Paired compressions (h1c_a_call2_k), pouw-decode's pd_pcmp (pd.cuh at 4dca175d7a): pair p is consumer warp p and producer
// warp 2 + p, sharing AP_WORDS of dynamic shared memory, x[t][lane] for t < 96 (t < 80 the round inputs K_t + W_t, the
// consumer's for t < 16 and the producer's after; 80 + t the message's W_t, t < 16) and a stop word.  The consumer stores
// its message and runs the rounds in five chunks of 16 while the producer computes the schedule a chunk ahead: barrier
// 1 + 5 p the message, 2 + 5 p .. 5 + 5 p the chunks from 16, 32, 48 and 64 ready, 64 threads each.  Every lane of both
// warps takes part in every compression, each lane its own.
constexpr int AP_WORDS = 96 * 32 + 8;
constexpr int AP_SMEM = 2 * 8 * AP_WORDS;
constexpr int AP_ROWS = 4, AP_SEG = 16, AP_MAXCOUNT = 512, AP_CONSUMERS = 11;
__device__ __forceinline__ void ap_sync(int id, int n) { asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void ap_arrive(int id, int n) {
  asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ u64* ap_base(int p) {
  extern __shared__ __align__(16) u64 ap_smem[];
  return ap_smem + AP_WORDS * p;
}
__device__ __forceinline__ void ap_round(u64& a, u64& b, u64& c, u64& d, u64& e, u64& f, u64& g, u64& h, u64 kw) {
  const u64 t1 = h + (sx::rotr<0, 14>(e) ^ sx::rotr<0, 18>(e) ^ sx::rotr<0, 41>(e)) + ((e & f) ^ (~e & g)) + kw;
  const u64 t2 = (sx::rotr<0, 28>(a) ^ sx::rotr<0, 34>(a) ^ sx::rotr<0, 39>(a)) + ((a & b) ^ (a & c) ^ (b & c));
  h = g;
  g = f;
  f = e;
  e = d + t1;
  d = c;
  c = b;
  b = a;
  a = t1 + t2;
}
// the consumer's half: one block's message words w into h; one copy a kernel
__device__ __noinline__ void ap_cmp(u64* h, const u64* w) {
  const int lane = threadIdx.x & 31, p = threadIdx.x >> 5, bar = 1 + 5 * p;
  u64* base = ap_base(p);
  u64* x = base + lane;
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    x[32 * (80 + t)] = w[t];
    x[32 * t] = sx::K_DEV[t] + w[t];
  }
  if (lane == 0) base[96 * 32] = 0;
  ap_arrive(bar, 64);
  u64 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
#pragma unroll 1
  for (int q = 0; q < 5; ++q) {
    if (q) ap_sync(bar + q, 64);
    u64 kw[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) kw[i] = x[32 * (16 * q + i)];
#pragma unroll
    for (int i = 0; i < 16; ++i) ap_round(a, b, c, d, e, f, g, hh, kw[i]);
  }
  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += hh;
}
// ap_cmp inlined, the message and state in registers (h1c_a_call3_k's one site)
__device__ __forceinline__ void ap_cmp_in(u64 h[8], const u64 w[16]) {
  const int lane = threadIdx.x & 31, p = threadIdx.x >> 5, bar = 1 + 5 * p;
  u64* base = ap_base(p);
  u64* x = base + lane;
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    x[32 * (80 + t)] = w[t];
    x[32 * t] = sx::K_DEV[t] + w[t];
  }
  if (lane == 0) base[96 * 32] = 0;
  ap_arrive(bar, 64);
  u64 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
#pragma unroll 1
  for (int q = 0; q < 5; ++q) {
    if (q) ap_sync(bar + q, 64);
    u64 kw[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) kw[i] = x[32 * (16 * q + i)];
#pragma unroll
    for (int i = 0; i < 16; ++i) ap_round(a, b, c, d, e, f, g, hh, kw[i]);
  }
  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += hh;
}
// the producer warp of pair p: each message's schedule, until its consumer's ap_stop
__device__ void ap_serve(int p) {
  const int lane = threadIdx.x & 31, bar = 1 + 5 * p;
  u64* base = ap_base(p);
  u64* x = base + lane;
  for (;;) {
    ap_sync(bar, 64);
    if (*reinterpret_cast<volatile u64*>(base + 96 * 32)) return;
    u64 w[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) w[t] = x[32 * (80 + t)];
#pragma unroll
    for (int q = 1; q < 5; ++q) {
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        const u64 w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
        w[t] += (sx::rotr<0, 1>(w15) ^ sx::rotr<0, 8>(w15) ^ (w15 >> 7)) + w[(t + 9) & 15] +
                (sx::rotr<0, 19>(w2) ^ sx::rotr<0, 61>(w2) ^ (w2 >> 6));
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) x[32 * (16 * q + i)] = sx::K_DEV[16 * q + i] + w[i];
      ap_arrive(bar + q, 64);
    }
  }
}
__device__ __forceinline__ void ap_stop() {  // a consumer warp: its producer returns
  const int p = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) ap_base(p)[96 * 32] = 1;
  ap_arrive(1 + 5 * p, 64);
}

// one block, not inlined: every flush site shares one compression's code
__device__ __noinline__ void hb_block(u64* h, const uint8_t* b) {
  u64 w[16], v[8];
  const u32* q = reinterpret_cast<const u32*>(b);
#pragma unroll
  for (int k = 0; k < 16; ++k) w[k] = (u64)bswap32(q[2 * k]) << 32 | bswap32(q[2 * k + 1]);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = h[j];
  sx::compress<0, 2>(v, w);
#pragma unroll
  for (int j = 0; j < 8; ++j) h[j] = v[j];
}
// PR: the flush is a paired compression (ap_cmp), so every lane of the consumer warp streams the same lengths
template <int PR = 0>
__device__ __forceinline__ void hb_flush(Hb& s) {
  if constexpr (PR != 0) {
    u64 w[16];
    const u32* q = reinterpret_cast<const u32*>(s.b);
#pragma unroll
    for (int k = 0; k < 16; ++k) w[k] = (u64)bswap32(q[2 * k]) << 32 | bswap32(q[2 * k + 1]);
    ap_cmp(s.h, w);
  } else {
    hb_block(s.h, s.b);
  }
  s.n = 0;
}
template <int PR = 0>
__device__ __forceinline__ void hb_u8(Hb& s, u32 v) {
  s.b[s.n] = (uint8_t)v;
  ++s.tot;
  if (++s.n == 128) hb_flush<PR>(s);
}
template <int PR = 0>
__device__ __forceinline__ void hb_be(Hb& s, u64 x, int nbytes) {
  for (int i = nbytes - 1; i >= 0; --i) hb_u8<PR>(s, (u32)(x >> (8 * i)));
}
// one big-endian word at any offset
template <int PR = 0>
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
    hb_be<PR>(s, v, 8);
    return;
  }
  s.n += 8;
  s.tot += 8;
  if (s.n == 128) hb_flush<PR>(s);
}
template <int PR = 0>
__device__ __forceinline__ void hb_words(Hb& s, const u64* x, int nw) {
  for (int k = 0; k < nw; ++k) hb_put8<PR>(s, x[k]);
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
template <int PR = 0>
__device__ __forceinline__ void hb_final(Hb& s, u64* out) {
  const u64 bits = 8 * s.tot;
  s.b[s.n++] = 0x80;
  if (s.n > 112) {
    while (s.n < 128) s.b[s.n++] = 0;
    hb_flush<PR>(s);
  }
  while (s.n & 3) s.b[s.n++] = 0;
  for (; s.n < 120; s.n += 4) *reinterpret_cast<u32*>(s.b + s.n) = 0;
  u32* q = reinterpret_cast<u32*>(s.b);
  q[30] = bswap32((u32)(bits >> 32));
  q[31] = bswap32((u32)bits);
  hb_flush<PR>(s);
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

// node idx of level lv (its first block from the step's prefix), children read through L2 (another block wrote them);
// a paired lane with no node (keep false) hashes node idx's message and writes nothing
template <int PR = 0>
__device__ __forceinline__ void a_node(Hb& s, uint8_t* b, const u64* P, const H1Tab& T, int lv, long long idx,
                                       long long off, bool keep = true) {
  const long long width = 1LL << T.depth;
  const u64* in = T.level + 8 * (off + 2 * idx);
  u64* o = T.level + 8 * (off + (width >> lv) + idx);
  u64 c[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) c[k] = __ldcg(in + k);
  hb_load(s, P + A_NS + SNAP * lv, b);
  int l = 1;
  while (l < 8 && (idx >> (8 * l))) ++l;
  hb_u8<PR>(s, (u32)l);  // uint(idx)'s length's low byte: the prefix holds its other seven
  hb_be<PR>(s, (u64)idx, l);
  hb_put8<PR>(s, 64);
  hb_words<PR>(s, c, 8);
  hb_put8<PR>(s, 64);
  hb_words<PR>(s, c + 8, 8);
  u64 out[8];
  hb_final<PR>(s, out);
  if (!keep) return;
#pragma unroll
  for (int k = 0; k < 8; ++k) o[k] = out[k];
  if (lv == T.depth - 1)
#pragma unroll
    for (int k = 0; k < 8; ++k) T.root[k] = out[k];
}

// row i's leaf from its x (h = SHA-512(leaf_prefix || x ^ M_key y || c), then the frame leaf from the row's prefix)
template <int PR = 0>
__device__ __forceinline__ void a_leaf(Hb& s, uint8_t* b, const u64* P, long long i, const u64 x[8], u64 out[8]) {
  const u64* R = P + A_ROW + (long long)A_RW * i;
  u64 mk[8], h[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) mk[k] = x[k] ^ R[SNAP + k];
  hb_load(s, P + A_LPS, b);
  hb_words<PR>(s, mk, 8);
  hb_words<PR>(s, R + SNAP + 8, 8);
  hb_final<PR>(s, h);
  hb_load(s, R, b);
  hb_words<PR>(s, h, 8);
  hb_final<PR>(s, out);
}

#if H1C_HB
// h1c_build's kernels on Hb: the same messages as the Hs forms above
constexpr int HT = 128;  // threads a block

// bytes [off, off + m) of row i; a whole aligned block of the body is compressed from registers
__device__ __forceinline__ void hb_row(Hb& s, const H1Tab& T, long long i, int off, int m) {
  int p = off;
  const int end = off + m;
  const uint8_t* base = T.data + i * T.stride;
  while (p < end) {
    if (s.n == 0 && end - p >= 128 && p + 128 <= T.body && ((reinterpret_cast<uintptr_t>(base + p) & 15) == 0)) {
      const uint4* q = reinterpret_cast<const uint4*>(base + p);
      u64 w[16];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const uint4 v = q[k];
        w[2 * k] = (u64)bswap32(v.x) << 32 | bswap32(v.y);
        w[2 * k + 1] = (u64)bswap32(v.z) << 32 | bswap32(v.w);
      }
      sx::compress<0, 2>(s.h, w);
      s.tot += 128;
      p += 128;
    } else {
      hb_u8(s, row_byte(T, i, p));
      ++p;
    }
  }
}

__device__ __forceinline__ void hb_domain(H1Tab& T, u64 st, uint8_t* b) {
  const H1Dom& D = T.dm;
  uint8_t d[2][20];
  int dl[2] = {0, 0};
  long long jl = D.js_len[0] + D.js_len[1] + D.js_len[2];
  for (int v = 0; v < D.nval; ++v) {
    dl[v] = dec(D.va[v] * st + D.vb[v], d[v]);
    jl += dl[v];
  }
  Hb s;
  hb_init(s, b);
  hb_bytes(s, D.bind, D.bind_len);
  hb_be(s, (u64)jl, 8);
  hb_bytes(s, D.js[0], D.js_len[0]);
  for (int v = 0; v < D.nval; ++v) {
    hb_bytes(s, d[v], dl[v]);
    hb_bytes(s, D.js[v + 1], D.js_len[v + 1]);
  }
  u64 binding[8];
  hb_final(s, binding);
  hb_init(s, b);
  hb_frame(s, "domain", 6);
  hb_be(s, 64, 8);
  hb_words(s, binding, 8);
  hb_uint(s, 1);  // owner -1, as _uint(owner + 2)
  hb_be(s, 64, 8);
  hb_words(s, D.pos, 8);
  hb_uint(s, (u64)T.count);
  hb_final(s, T.dom);
}

extern "C" __global__ void __launch_bounds__(64) h1c_domains(H1Tab* __restrict__ tabs, int ntab, const int* __restrict__ step) {
  __shared__ __align__(16) uint8_t bf[64 * HB];
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= ntab) return;
  hb_domain(tabs[t], (u64)(long long)*step, bf + threadIdx.x * HB);
}

// segment j's first block S(j, ns, m_j), exactly 128 bytes: compressed, s.n = 0
__device__ __forceinline__ void seg_head(Hb& s, const H1Tab& T, int j, int ns) {
  hb_bytes(s, HC.stag, HC.stag_len);
  hb_be(s, (u64)j, 8);
  hb_be(s, (u64)ns, 8);
  hb_be(s, 8ull * min(1024, T.len - 1024 * j), 8);
  hb_zeros(s, 128 - HC.stag_len - 24);
}
constexpr int SEGMID_MAX = 16;

// dom = 1 (H1C_DOMSEG): block row 0 is h1c_domains' work, a thread a table, and row y + 1 table y's segments
extern "C" __global__ void H1C_SEGS_BOUNDS h1c_segs(H1Tab* __restrict__ tabs, int ntab, const int* __restrict__ step,
                                                   int dom) {
  __shared__ __align__(16) uint8_t bf[HT * HB];
  if ((int)blockIdx.y < dom) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < ntab) hb_domain(tabs[t], (u64)(long long)*step, bf + threadIdx.x * HB);
    return;
  }
  const H1Tab& T = tabs[blockIdx.y - dom];
  if (!T.data) return;
  const int ns = nseg_of(T.len);
  const long long t0 = (long long)blockIdx.x * blockDim.x, t = t0 + threadIdx.x, total = (long long)T.count * ns;
#if H1C_SEGMID
  // lane q < ns compresses segment (t0 + q) % ns's first block, which thread t0 + r starts from for r % ns = q
  // (H1C_SEGJ: lane q < nj segment j0 + q's, the block's threads taking segments j0 to j0 + nj - 1, nj <= ns)
  __shared__ u64 shd[SEGMID_MAX][8];
  const bool sharedh = ns <= SEGMID_MAX;
  if (t0 >= total) return;
#if H1C_SEGJ
  const int j0 = (int)(t0 / T.count);
  const int nh = (int)((min(t0 + (long long)blockDim.x, total) - 1) / T.count) - j0 + 1;
#else
  const int nh = ns;
#endif
  if (sharedh) {
    if ((int)threadIdx.x < nh) {
      Hb s;
      hb_init(s, bf + threadIdx.x * HB);
#if H1C_SEGJ
      seg_head(s, T, j0 + (int)threadIdx.x, ns);
#else
      seg_head(s, T, (int)((t0 + threadIdx.x) % ns), ns);
#endif
#pragma unroll
      for (int k = 0; k < 8; ++k) shd[threadIdx.x][k] = s.h[k];
    }
    __syncthreads();
  }
#endif
  if (t >= total) return;
#if H1C_SEGJ
  const long long i = t % T.count;
  const int j = (int)(t / T.count);
#else
  const long long i = t / ns;
  const int j = (int)(t % ns);
#endif
  if (row_dead(T, i)) return;
  const int off = 1024 * j, m = min(1024, T.len - off);
  Hb s;
  hb_init(s, bf + threadIdx.x * HB);
#if H1C_SEGMID
  if (sharedh) {
#if H1C_SEGJ
    const int q = j - j0;
#else
    const int q = (int)((t - t0) % ns);
#endif
#pragma unroll
    for (int k = 0; k < 8; ++k) s.h[k] = shd[q][k];
    s.tot = 128;
  } else
#endif
    seg_head(s, T, j, ns);
  hb_row(s, T, i, off, m);
  hb_final(s, T.segs + (i * ns + j) * 8);
}

// H1C_MK: byte table p's row v is the XOR of M_key's columns 8p + s over v's set bits s, column 8p + s its row 1 << s,
// so M_key y is the XOR of the columns of y's set bits.  The block stages MKW words' columns at a time in bf (every
// thread of the block, at its barriers), and its threads read each column at one address: a broadcast, 1536 columns
// from shared memory rather than 192 rows from L2.  y comes back in its order (24 rotations).
#if H1C_MKN
// H1C_MKN: the same XOR a nibble at a time.  Byte table p's rows v << 4 and v (v < 16) are its high and low nibbles'
// tables, so M_key y is the XOR of one row of each of y's 384 nibble tables.  A stage is one y word's 16 tables (16 KB
// of bf), and a row's 16-byte chunk k sits at slot k ^ (v >> 1 & 3): the lanes reading chunk k of rows v < 8 hit eight
// different bank groups.  A row a nibble, where the columns cost a masked row a bit.
static_assert(16 * 16 * 64 <= HT * HB, "a word's nibble tables fit bf");
__device__ __forceinline__ void hb_mk(u64 y[24], u64 mk[8], const u64* __restrict__ btab, uint8_t* bf, bool real) {
  uint4* cs = reinterpret_cast<uint4*>(bf);
#pragma unroll
  for (int k = 0; k < 8; ++k) mk[k] = 0;
#pragma unroll 1
  for (int wd = 0; wd < 24; ++wd) {
    __syncthreads();
    for (int q = threadIdx.x; q < 16 * 16 * 4; q += HT) {
      const int t = q >> 6, r = (q >> 2) & 15, k = q & 3;  // table t = 2e + h (byte e from the top, h 1 its low nibble)
      const u32 v = (t & 1) ? (u32)r : (u32)r << 4;
      cs[(q & ~3) | (k ^ ((r >> 1) & 3))] =
          __ldg(reinterpret_cast<const uint4*>(btab + ((size_t)(8 * wd + (t >> 1)) * 256 + v) * 8) + k);
    }
    __syncthreads();
    const u64 w = y[0];
#pragma unroll
    for (int k = 0; k < 23; ++k) y[k] = y[k + 1];
    y[23] = w;
    if (!real) continue;
    const uint8_t* cb = reinterpret_cast<const uint8_t*>(cs);
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const u32 v = (u32)(w >> (60 - 4 * t)) & 15u;
      const u32 s4 = (v << 3) & 0x30u;
      const uint8_t* row = cb + (t << 10) + (v << 6);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint4 q = *reinterpret_cast<const uint4*>(row + (((u32)k << 4) ^ s4));
        mk[2 * k] ^= (u64)q.y << 32 | q.x;
        mk[2 * k + 1] ^= (u64)q.w << 32 | q.z;
      }
    }
  }
  __syncthreads();
}
#else
constexpr int MKW = 4;  // y words a stage: 8 MKW positions of 8 columns of 64 bytes
static_assert(MKW * 8 * 8 * 64 <= HT * HB && 24 % MKW == 0, "a stage fits bf and the stages cover y");
__device__ __forceinline__ void hb_mk(u64 y[24], u64 mk[8], const u64* __restrict__ btab, uint8_t* bf, bool real) {
  uint4* cs = reinterpret_cast<uint4*>(bf);
#pragma unroll
  for (int k = 0; k < 8; ++k) mk[k] = 0;
#pragma unroll 1
  for (int wd = 0; wd < 24; ++wd) {
    if (wd % MKW == 0) {
      __syncthreads();
      for (int q = threadIdx.x; q < MKW * 8 * 8 * 4; q += HT) {
        const int c = q >> 2, p = 8 * wd + (c >> 3);
        cs[q] = __ldg(reinterpret_cast<const uint4*>(btab + ((size_t)p * 256 + (1u << (c & 7))) * 8) + (q & 3));
      }
      __syncthreads();
    }
    u64 w = y[0];
#pragma unroll
    for (int k = 0; k < 23; ++k) y[k] = y[k + 1];
    y[23] = w;
    if (!real) continue;
    const uint4* cw = cs + (wd % MKW) * 64 * 4;
#pragma unroll 1
    for (int e = 0; e < 8; ++e) {
      const u32 v = (u32)(w >> 56);
      w <<= 8;
#pragma unroll
      for (int sb = 0; sb < 8; ++sb) {
        const u32 m = 0u - ((v >> sb) & 1u);
        const uint4* q4 = cw + (8 * e + sb) * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const uint4 q = q4[k];
          mk[2 * k] ^= (u64)(q.y & m) << 32 | (q.x & m);
          mk[2 * k + 1] ^= (u64)(q.w & m) << 32 | (q.z & m);
        }
      }
    }
  }
  __syncthreads();
}
#endif

// row leaf i's x = SHA-512(Q(n_bits) || d_0 || ..), its salt y, and (mk = M_key y on entry) its frame leaf.  sn (H1C_MID,
// leaf_head's snapshots) null: every message from its first byte
constexpr int SNW = 32;  // a snapshot's words: hb_save's 9 and a partial block
__device__ __forceinline__ void x_head(Hb& s, const H1Tab& T) {
  hb_bytes(s, HC.qtag, HC.qtag_len);
  hb_be(s, 8ull * T.len, 8);
  hb_be(s, (u64)nseg_of(T.len), 8);
  hb_be(s, 1024, 4);
  hb_zeros(s, 128 - HC.qtag_len - 20);
}
__device__ __forceinline__ void frame_head(Hb& s, const H1Tab& T) {
  hb_frame(s, "leaf", 4);
  hb_be(s, 64, 8);
  hb_words(s, T.dom, 8);
}
// snapshot k of table T's constant heads: 0 x's header, 1 salt_prefix, 2 leaf_prefix, 3 the frame leaf's prefix
__device__ __forceinline__ void leaf_head(const H1Tab& T, int k, uint8_t* b, u64* o) {
  Hb s;
  hb_init(s, b);
  if (k == 0) x_head(s, T);
  else if (k == 1) hb_words(s, HC.salt_prefix, 16);
  else if (k == 2) hb_words(s, HC.leaf_prefix, 16);
  else frame_head(s, T);
  hb_save(s, o);
}
__device__ __forceinline__ void leaf_x(const H1Tab& T, long long i, uint8_t* b, u64 x[8], const u64* sn = nullptr) {
  if (row_dead(T, i)) {
#pragma unroll
    for (int k = 0; k < 8; ++k) x[k] = T.xd[8 * (i % T.rbw) + k];
    return;
  }
  const int ns = nseg_of(T.len);
  Hb s;
  if (sn) {
    hb_load(s, sn, b);
  } else {
    hb_init(s, b);
    x_head(s, T);
  }
  hb_words(s, T.segs + i * ns * 8, 8 * ns);
  hb_final(s, x);
}
__device__ __forceinline__ void leaf_salt(const H1Tab& T, long long i, const int* __restrict__ step, uint8_t* b,
                                          u64 y[24]) {
  uint8_t d[20];
  int dl = 0;
  if (T.labs) dl = dec((u64)(long long)*step, d);
  const int ll = T.lab_len[0] + (T.labs ? dl + T.lab_len[1] : 0);
  Hb s;
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
}
__device__ __forceinline__ void leaf_out(const H1Tab& T, long long i, uint8_t* b, const u64 x[8], const u64 y[24],
                                         u64 mk[8], u64 out[8], const u64* sn = nullptr) {
  u64 c[8], h[8];
  Hb s;
  if (sn) {
    hb_load(s, sn + SNW, b);
  } else {
    hb_init(s, b);
    hb_words(s, HC.salt_prefix, 16);
  }
  hb_words(s, y, 24);
  hb_final(s, c);
#pragma unroll
  for (int k = 0; k < 8; ++k) mk[k] ^= x[k];
  if (sn) {
    hb_load(s, sn + 2 * SNW, b);
  } else {
    hb_init(s, b);
    hb_words(s, HC.leaf_prefix, 16);
  }
  hb_words(s, mk, 8);
  hb_words(s, c, 8);
  hb_final(s, h);
  if (sn) {
    hb_load(s, sn + 3 * SNW, b);
  } else {
    hb_init(s, b);
    frame_head(s, T);
  }
  hb_uint(s, (u64)i);
  hb_uint(s, (u64)i);
  hb_be(s, (u64)HC.schema_len, 8);
  hb_bytes(s, HC.schema, HC.schema_len);
  hb_be(s, 64, 8);
  hb_words(s, h, 8);
  hb_final(s, out);
}

__device__ __forceinline__ void hb_node(const H1Tab& T, uint8_t* b, int lv, long long idx);

// H1C_LVB: the levels a leaves block's 128 leaves complete
constexpr int LVB = 7;
static_assert(HT == 1 << LVB, "a leaves block is a subtree of LVB levels");

extern "C" __global__ void __launch_bounds__(HT, H1C_LMINB) h1c_leaves(const H1Tab* __restrict__ tabs, int ntab,
                                                                      const int* __restrict__ step,
                                                                      const u64* __restrict__ btab) {
  __shared__ __align__(16) uint8_t bf[HT * HB];
  const H1Tab& T = tabs[blockIdx.y];
  const long long width = 1LL << T.depth;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t* b = bf + threadIdx.x * HB;
  u64 out[8];
#if H1C_MK
  if ((long long)blockIdx.x * blockDim.x >= width) return;
  if (T.data && (long long)blockIdx.x * blockDim.x < T.count) {
    // a block with a row leaf: every thread runs hb_mk's barriers, the row leaves' threads hash their salts first
    const bool real = i < T.count;
    u64 x[8], y[24], mk[8];
#if H1C_MID
    __shared__ u64 sn[4 * SNW];  // lane 0 of warp k writes snapshot k; hb_mk's first barrier publishes them
    static_assert(HT == 4 * 32, "a warp a snapshot");
    if ((threadIdx.x & 31) == 0) leaf_head(T, threadIdx.x >> 5, b, sn + SNW * (threadIdx.x >> 5));
    const u64* snp = sn;
#else
    const u64* snp = nullptr;
#endif
    if (real) leaf_salt(T, i, step, b, y);
    hb_mk(y, mk, btab, bf, real);
    if (real) {
      leaf_x(T, i, b, x, snp);
      leaf_out(T, i, b, x, y, mk, out, snp);
    }
  }
  const bool mine = i < width && (i >= T.count || T.data);
#if !H1C_LVB
  if (!mine) return;
#endif
  if (mine && i >= T.count) {
#else
  if (i >= width) return;
  constexpr bool mine = true;
  if (i >= T.count) {
#endif
    Hb s;
    hb_init(s, b);
    hb_frame(s, "pad", 3);
    hb_be(s, 64, 8);
    hb_words(s, T.dom, 8);
    hb_uint(s, (u64)i);
    hb_final(s, out);
  } else {
#if !H1C_MK
    if (!T.data) return;
    u64 x[8], y[24], mk[8];
    leaf_x(T, i, b, x);
    leaf_salt(T, i, step, b, y);
#pragma unroll
    for (int k = 0; k < 8; ++k) mk[k] = 0;
#pragma unroll
    for (int wd = 0; wd < 24; ++wd) {
      const u64 yw = y[wd];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const u32 v = (u32)(yw >> (56 - 8 * e)) & 0xFFu;
        const uint4* q4 = reinterpret_cast<const uint4*>(btab + ((size_t)(8 * wd + e) * 256 + v) * 8);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const uint4 q = __ldg(q4 + k);
          mk[2 * k] ^= (u64)q.y << 32 | q.x;
          mk[2 * k + 1] ^= (u64)q.w << 32 | q.z;
        }
      }
    }
    leaf_out(T, i, b, x, y, mk, out);
#endif
  }
  if (mine) {
#pragma unroll
    for (int k = 0; k < 8; ++k) T.level[i * 8 + k] = out[k];
    if (width == 1)
#pragma unroll
      for (int k = 0; k < 8; ++k) T.root[k] = out[k];
  }
#if H1C_LVB
  // a table's given leaves are in level 0 before this launch; the barrier publishes the block's own
  __syncthreads();
  const int top = min(T.depth, LVB);
  for (int lv = 0; lv < top; ++lv) {
    const long long n = HT >> (lv + 1), idx = (long long)blockIdx.x * n + threadIdx.x;
    if (threadIdx.x < n && idx < (width >> (lv + 1))) hb_node(T, b, lv, idx);
    __syncthreads();
  }
#endif
}

__device__ __forceinline__ void hb_node(const H1Tab& T, uint8_t* b, int lv, long long idx) {
  const long long width = 1LL << T.depth;
  long long off = 0;
  for (int e = 0; e < lv; ++e) off += width >> e;
  const u64* in = T.level + 8 * (off + 2 * idx);
  u64* o = T.level + 8 * (off + (width >> lv) + idx);
  Hb s;
  hb_init(s, b);
  hb_frame(s, "node", 4);
  hb_be(s, 64, 8);
  hb_words(s, T.dom, 8);
  hb_uint(s, (u64)lv);
  hb_uint(s, (u64)idx);
  hb_be(s, 64, 8);
  hb_words(s, in, 8);
  hb_be(s, 64, 8);
  hb_words(s, in + 8, 8);
  u64 out[8];
  hb_final(s, out);
#pragma unroll
  for (int k = 0; k < 8; ++k) o[k] = out[k];
  if (lv == T.depth - 1)
#pragma unroll
    for (int k = 0; k < 8; ++k) T.root[k] = out[k];
}

#if H1C_NMID
// a level's node Snap: h, tot and the one byte past the first block (NSNAP words); levels from a launch's first up to
// NMAXD of them have one in h1c_levels, the rest hash through hb_node
constexpr int NSNAP = 10, NMAXD = 32;
__device__ __forceinline__ void node_snap(const H1Tab& T, int lv, uint8_t* b, u64* o) {
  Hb s;
  hb_init(s, b);
  hb_frame(s, "node", 4);
  hb_be(s, 64, 8);
  hb_words(s, T.dom, 8);
  hb_uint(s, (u64)lv);
  hb_zeros(s, 7);
  hb_save(s, o);
}
// hb_node's message from its level's Snap
__device__ __forceinline__ void hb_node_snap(const H1Tab& T, uint8_t* b, int lv, long long idx, const u64* snap) {
  const long long width = 1LL << T.depth;
  long long off = 0;
  for (int e = 0; e < lv; ++e) off += width >> e;
  const u64* in = T.level + 8 * (off + 2 * idx);
  u64* o = T.level + 8 * (off + (width >> lv) + idx);
  Hb s;
  hb_load(s, snap, b);
  int l = 1;
  while (l < 8 && (idx >> (8 * l))) ++l;
  hb_u8(s, (u32)l);  // uint(idx)'s length's low byte: the Snap holds its other seven
  hb_be(s, (u64)idx, l);
  hb_put8(s, 64);
  hb_words(s, in, 8);
  hb_put8(s, 64);
  hb_words(s, in + 8, 8);
  u64 out[8];
  hb_final(s, out);
#pragma unroll
  for (int k = 0; k < 8; ++k) o[k] = out[k];
  if (lv == T.depth - 1)
#pragma unroll
    for (int k = 0; k < 8; ++k) T.root[k] = out[k];
}
#endif

extern "C" __global__ void __launch_bounds__(HT) h1c_level(const H1Tab* __restrict__ tabs, int ntab, int lv) {
  __shared__ __align__(16) uint8_t bf[HT * HB];
  const H1Tab& T = tabs[blockIdx.y];
  if (lv >= T.depth) return;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
#if H1C_NMID
  __shared__ __align__(16) u64 ns[NSNAP];
  const long long nodes = (1LL << T.depth) >> (lv + 1);
  if ((long long)blockIdx.x * blockDim.x >= nodes) return;
  if (threadIdx.x == 0) node_snap(T, lv, bf, ns);
  __syncthreads();
  if (idx >= nodes) return;
  hb_node_snap(T, bf + threadIdx.x * HB, lv, idx, ns);
#else
  if (idx >= (1LL << T.depth) >> (lv + 1)) return;
  hb_node(T, bf + threadIdx.x * HB, lv, idx);
#endif
}

// H1C_LVF = L >= 0: a table's levels from L up in one block (LT threads), a barrier between levels, so the levels whose
// nodes a block covers cost one launch rather than one each
constexpr int LT = H1C_LT;
static_assert(LT >= 32 && LT <= 1024 && LT % 32 == 0, "H1C_LT: whole warps, at most 1024 threads");
extern "C" __global__ void __launch_bounds__(LT) h1c_levels(const H1Tab* __restrict__ tabs, int ntab, int lv0) {
  __shared__ __align__(16) uint8_t bf[LT * HB];
  const H1Tab& T = tabs[blockIdx.x];
#if H1C_NMID
  __shared__ __align__(16) u64 ns[NMAXD][NSNAP];
  for (int q = threadIdx.x; q < NMAXD && lv0 + q < T.depth; q += LT) node_snap(T, lv0 + q, bf + threadIdx.x * HB, ns[q]);
  __syncthreads();
#endif
  for (int lv = lv0; lv < T.depth; ++lv) {
    const long long nodes = (1LL << T.depth) >> (lv + 1);
#if H1C_NMID
    if (lv - lv0 < NMAXD) {
      for (long long idx = threadIdx.x; idx < nodes; idx += LT) hb_node_snap(T, bf + threadIdx.x * HB, lv, idx, ns[lv - lv0]);
      __syncthreads();
      continue;
    }
#endif
    for (long long idx = threadIdx.x; idx < nodes; idx += LT) hb_node(T, bf + threadIdx.x * HB, lv, idx);
    __syncthreads();
  }
}

#if H1C_NMID
// h1c_tail: two sets of given tables, mid's roots another's (rnd's) leaves.  h1c_tail_pre_k is what reads no leaf, a block
// a table (mid's, then rnd's): its domain, its pads and each level's Snap into snaps (NMAXD Snaps a table); h1c_tail_k a
// block a mid table, its levels from those Snaps, and the last block to finish rnd's levels, so the part that waits on
// the leaves is one launch
extern "C" __global__ void __launch_bounds__(LT) h1c_tail_pre_k(H1Tab* __restrict__ mid, int nmid, H1Tab* __restrict__ rnd,
                                                               const int* __restrict__ step, u64* __restrict__ snaps) {
  __shared__ __align__(16) uint8_t bf[LT * HB];
  const int t = blockIdx.x;
  H1Tab& T = t < nmid ? mid[t] : rnd[t - nmid];
  uint8_t* b = bf + threadIdx.x * HB;
  if (threadIdx.x == 0) hb_domain(T, (u64)(long long)*step, b);
  __syncthreads();
  const long long width = 1LL << T.depth;
  for (long long i = T.count + threadIdx.x; i < width; i += LT) {
    Hb s;
    hb_init(s, b);
    hb_frame(s, "pad", 3);
    hb_be(s, 64, 8);
    hb_words(s, T.dom, 8);
    hb_uint(s, (u64)i);
    u64 out[8];
    hb_final(s, out);
#pragma unroll
    for (int k = 0; k < 8; ++k) T.level[i * 8 + k] = out[k];
  }
  for (int q = threadIdx.x; q < T.depth; q += LT) node_snap(T, q, b, snaps + ((size_t)t * NMAXD + q) * NSNAP);
}
// H1C_TAILW = 1: a node whose Snap leaves one byte pending and whose idx is below 256 (its children at byte 3, as
// h1c_a_call3_k's) hashes its two blocks from words built in registers, one inlined compression, not through Hb's
// shared-memory block and hb_block's call; any other node through hb_node_snap
__device__ __forceinline__ void tail_node(const H1Tab& T, uint8_t* b, int lv, long long idx, const u64* snap) {
#if H1C_TAILW
  if (idx < 256 && (snap[8] & 127) == 1) {
    const long long width = 1LL << T.depth;
    long long off = 0;
    for (int e = 0; e < lv; ++e) off += width >> e;
    const u64* in = T.level + 8 * (off + 2 * idx);
    u64 c[16], h[8];
#pragma unroll
    for (int i = 0; i < 16; ++i) c[i] = __ldcg(in + i);
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = snap[i];
    const u64 bits = 8 * (snap[8] + 146);
#pragma unroll 1
    for (int q = 0; q < 2; ++q) {
      u64 w[16];
      if (q == 0) {
        w[0] = (snap[9] & 0xFF) << 56 | 1ull << 48 | (u64)idx << 40;
        w[1] = 64ull << 40 | c[0] >> 24;
#pragma unroll
        for (int i = 2; i < 9; ++i) w[i] = c[i - 2] << 40 | c[i - 1] >> 24;
        w[9] = c[7] << 40;
        w[10] = 64ull << 40 | c[8] >> 24;
#pragma unroll
        for (int i = 11; i < 16; ++i) w[i] = c[i - 3] << 40 | c[i - 2] >> 24;
      } else {
        w[0] = c[13] << 40 | c[14] >> 24;
        w[1] = c[14] << 40 | c[15] >> 24;
        w[2] = c[15] << 40 | 0x80ull << 32;
#pragma unroll
        for (int i = 3; i < 15; ++i) w[i] = 0;
        w[15] = bits;
      }
      sx::compress<0, 2>(h, w);
    }
    u64* o = T.level + 8 * (off + (width >> lv) + idx);
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = h[k];
    if (lv == T.depth - 1)
#pragma unroll
      for (int k = 0; k < 8; ++k) T.root[k] = h[k];
    return;
  }
#endif
  hb_node_snap(T, b, lv, idx, snap);
}
__device__ __forceinline__ void tail_levels(const H1Tab& T, const u64* snaps, uint8_t* b) {
  for (int lv = 0; lv < T.depth; ++lv) {
    const long long nodes = (1LL << T.depth) >> (lv + 1);
    for (long long idx = threadIdx.x; idx < nodes; idx += LT) tail_node(T, b, lv, idx, snaps + lv * NSNAP);
    __syncthreads();
  }
}
extern "C" __global__ void __launch_bounds__(LT) h1c_tail_k(const H1Tab* __restrict__ mid, int nmid,
                                                           const H1Tab* __restrict__ rnd, int nrnd,
                                                           const u64* __restrict__ snaps, unsigned* __restrict__ done,
                                                           u64* __restrict__ host_root, u64 seq) {
  __shared__ __align__(16) uint8_t bf[LT * HB];
  __shared__ bool last;
  uint8_t* b = bf + threadIdx.x * HB;
  tail_levels(mid[blockIdx.x], snaps + (size_t)blockIdx.x * NMAXD * NSNAP, b);
  // cooperative groups' grid barrier, one way: every mid root is out before the last block reads rnd's leaves
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(done, 1u) == (unsigned)nmid - 1;
    __threadfence();
  }
  __syncthreads();
  if (!last) return;
  if (threadIdx.x == 0) *done = 0;
  for (int r = 0; r < nrnd; ++r) tail_levels(rnd[r], snaps + (size_t)(nmid + r) * NMAXD * NSNAP, b);
  if (host_root) {
    for (int k = threadIdx.x; k < 8 * nrnd; k += LT) host_root[k] = rnd[k >> 3].root[k & 7];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *reinterpret_cast<volatile u64*>(host_root + 8 * nrnd) = seq;
  }
}
#endif
#endif

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
#if H1C_APW
    // 24 rotations bring y back in order for the salt hash below.
#pragma unroll 1
    for (int wd = 0; wd < 24; ++wd) {
      const u64 w = y[0];
#pragma unroll
      for (int k = 0; k < 23; ++k) y[k] = y[k + 1];
      y[23] = w;
      uint4 q[8][4];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const uint4* r =
            reinterpret_cast<const uint4*>(btab + ((size_t)(8 * wd + e) * 256 + ((w >> (56 - 8 * e)) & 0xFFu)) * 8);
#pragma unroll
        for (int k = 0; k < 4; ++k) q[e][k] = __ldg(r + k);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          mk[2 * k] ^= (u64)q[e][k].y << 32 | q[e][k].x;
          mk[2 * k + 1] ^= (u64)q[e][k].w << 32 | q[e][k].z;
        }
    }
#else
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
#endif
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
            sx::compress<0, 2>(s.h, w);
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

// segment block q of a row (the bytes after the segment's prefix chunk: L of them, then SHA-512's padding) as message words;
// row's bytes below k, the tail's (sa's) at k and above
__device__ __forceinline__ void a_seg_words(u64 w[16], const uint8_t* row, const uint8_t* tl, int k, int p0, int L, int nb,
                                            int q) {
#pragma unroll 1
  for (int t = 0; t < 16; ++t) {
    u64 v = 0;
    for (int i = 0; i < 8; ++i) {
      const int o = 128 * q + 8 * t + i, p = p0 + o;
      const u32 c = o < L ? (p < k ? row[p] : tl[p - k]) : (o == L ? 0x80u : 0u);
      v = v << 8 | c;
    }
    w[t] = v;
  }
  if (q == nb - 1) w[15] = 8ull * (128 + L);
}

// h1agate's phase stamps (h1c_a_stamps; null on the served path): the last block's index at 0, %globaltimer at 1 + lv after
// each level, and block b's at 32 + 8 b + i: i = 0 its start, 1 its segments, 2 its x, 3 its leaves, 4 its arrival
__device__ u64* ap_stamps = nullptr;
__device__ __forceinline__ void ap_stamp(int i) {
  if (threadIdx.x == 0 && ap_stamps) {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    ap_stamps[i] = t;
  }
}

// one call, paired (h1c_a_form(2)): h1c_a_call_k's table.  Block x is rows AP_ROWS x .. + 3 on two pairs (warps 0 and 2,
// 1 and 3): consumer warp p's half-warp h is row AP_ROWS x + 2 p + h, its lane j segment j (the block steps every lane
// through as many compressions as its longest segment, a lane past its own dropping a dummy's); then each half's x, h and
// frame leaf, all its lanes on the same message and lane 0's kept; the last block to arrive builds the nodes over live
// rows, 64 lanes at a time.  T.count a multiple of AP_ROWS and at most AP_MAXCOUNT (a node index fits a byte), at most
// AP_SEG segments a row.
extern "C" __global__ void __launch_bounds__(128) h1c_a_call2_k(const H1Tab* __restrict__ tab, u64* __restrict__ P,
                                                                const int* __restrict__ S, const int8_t* __restrict__ xq,
                                                                const float* __restrict__ sa, int8_t* __restrict__ slot,
                                                                float* __restrict__ scale, int k) {
  __shared__ __align__(16) uint8_t bf[64 * HB];
  __shared__ u64 dg[AP_ROWS][AP_SEG][8];
  __shared__ int last;
  const H1Tab& T = *tab;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const long long r0 = (long long)blockIdx.x * AP_ROWS;
  const long long m = S[1];
  const bool nw = (S[3] & 2) != 0;
  if (wid >= 2) {
    ap_serve(wid - 2);
    return;
  }
  ap_stamp(32 + 8 * blockIdx.x);
  const int half = lane >> 4, j = lane & 15, rb = 2 * wid + half;
  const long long r = r0 + rb;
  const bool alive = r < m;
  const int ns = nseg_of(T.len);
  const uint8_t* row = reinterpret_cast<const uint8_t*>(xq + (alive ? r : 0) * k);
  const uint8_t* tl = reinterpret_cast<const uint8_t*>(sa + (alive ? r : 0));
  uint8_t* b = bf + tid * HB;
  {
    // the slot row: a live row's from the blocks its segment lanes load, a dead row's zero
    const bool real = alive && j < ns;
    const int p0 = 1024 * j, L = real ? min(T.len, p0 + 1024) - p0 : 0, nb = real ? (L + 17 + 127) / 128 : 0;
    const int nf = real ? max(0, min(L, k - p0)) / 128 : 0;
    uint4* dst = reinterpret_cast<uint4*>(slot + r * k + p0);
    const int nq = __reduce_max_sync(0xffffffffu, (unsigned)nb);
    u64 hs[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) hs[i] = real ? P[A_SS + SNAP * j + i] : 0;
    const auto fast = [&](int q) { return q < nf; };
    uint4 nx[8];
    if (fast(0)) {
      const uint4* g = reinterpret_cast<const uint4*>(row + p0);
#pragma unroll
      for (int i = 0; i < 8; ++i) nx[i] = __ldg(g + i);
    }
#pragma unroll 1
    for (int q = 0; q < nq; ++q) {
      u64 w[16];
      if (fast(q)) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          w[2 * i] = (u64)bswap32(nx[i].x) << 32 | bswap32(nx[i].y);
          w[2 * i + 1] = (u64)bswap32(nx[i].z) << 32 | bswap32(nx[i].w);
        }
        if (!nw)
#pragma unroll
          for (int i = 0; i < 8; ++i) dst[8 * q + i] = nx[i];
      } else if (q < nb) {
        a_seg_words(w, row, tl, k, p0, L, nb, q);
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = 0;
      }
      if (fast(q + 1)) {
        const uint4* g = reinterpret_cast<const uint4*>(row + p0 + 128 * (q + 1));
#pragma unroll
        for (int i = 0; i < 8; ++i) nx[i] = __ldg(g + i);
      }
      u64 t[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) t[i] = hs[i];
      ap_cmp(t, w);
      if (q < nb)
#pragma unroll
        for (int i = 0; i < 8; ++i) hs[i] = t[i];
    }
    if (real)
#pragma unroll
      for (int i = 0; i < 8; ++i) dg[rb][j][i] = hs[i];
    if (!nw) {
      const int body = max(0, min(1024, k - p0)) / 16;  // the segment's body bytes, in uint4s
      if (alive) {
        const uint4* g = reinterpret_cast<const uint4*>(row + p0);
        for (int i = 8 * nf; i < body; ++i) dst[i] = g[i];
      } else {
        for (int i = 0; i < body; ++i) dst[i] = make_uint4(0, 0, 0, 0);
      }
      if (j == 0) scale[r] = alive ? sa[r] : 0.0f;
    }
  }
  ap_stamp(32 + 8 * blockIdx.x + 1);
  __syncwarp();
  if (__any_sync(0xffffffffu, alive)) {
    Hb s;
    u64 x[8], out[8];
    hb_load(s, P + A_QS, b);
    hb_words<1>(s, &dg[rb][0][0], 8 * ns);
    hb_final<1>(s, x);
    ap_stamp(32 + 8 * blockIdx.x + 2);
    a_leaf<1>(s, b, P, r, x, out);
    if (alive && j == 0)
#pragma unroll
      for (int i = 0; i < 8; ++i) T.level[8 * r + i] = out[i];
  }
  ap_stamp(32 + 8 * blockIdx.x + 3);
  __threadfence();
  ap_sync(AP_CONSUMERS, 64);
  if (tid == 0) last = atomicAdd(reinterpret_cast<unsigned*>(P + A_CNT), 1u) == gridDim.x - 1;
  ap_sync(AP_CONSUMERS, 64);
  ap_stamp(32 + 8 * blockIdx.x + 4);
  if (last) {
    if (tid == 0 && ap_stamps) ap_stamps[0] = blockIdx.x;
    __threadfence();
    const long long width = 1LL << T.depth;
    long long off = 0;
    Hb s;
    for (int lv = 0; lv < T.depth; ++lv) {
      const long long nodes = width >> (lv + 1), live = min(nodes, (m + (2LL << lv) - 1) >> (lv + 1));
      for (long long i0 = 0; i0 < live; i0 += 64) {
        const long long idx = i0 + tid;
        a_node<1>(s, b, P, T, lv, idx < live ? idx : 0, off, idx < live);
      }
      off += width >> lv;
      __threadfence_block();
      ap_sync(AP_CONSUMERS, 64);
      ap_stamp(1 + lv);
    }
    if (tid == 0) *reinterpret_cast<unsigned*>(P + A_CNT) = 0;
  }
  ap_stop();
}

// h1c_a_form(3): h1c_a_call2_k's table, every message built as words in registers and every compression the one inlined
// ap_cmp_in site of a loop over the lane's blocks (segments, x, h, the frame leaf, then the last block's nodes).  It reads
// the layout the window's constants give (h1c_a_call refuses form 3 otherwise, a3_layout): the segment, x and h prefixes
// end on a block, a node prefix leaves one byte pending (an idx below 256 then puts its children at byte 3), a row's leaf
// frame prefix 41 to 47 (its children at word 5, byte 1 to 7).  Each consumer warp stages its rows' and the levels'
// prefixes into shared memory first (cs: x's and h's first-block states, its two rows' frame prefixes, M_key y and c,
// each level's node prefix).
constexpr int A3_R = 16, A3_N = 80, A3_W = A3_N + 10 * 9;
__device__ __forceinline__ u64 bswap64(u64 v) { return (u64)bswap32((u32)v) << 32 | bswap32((u32)(v >> 32)); }
extern "C" __global__ void __launch_bounds__(128) h1c_a_call3_k(const H1Tab* __restrict__ tab, u64* __restrict__ P,
                                                                const int* __restrict__ S, const int8_t* __restrict__ xq,
                                                                const float* __restrict__ sa, int8_t* __restrict__ slot,
                                                                float* __restrict__ scale, int k) {
  __shared__ u64 dg[AP_ROWS][AP_SEG][8];
  __shared__ u64 cs[2][A3_W];
  __shared__ int last;
  const H1Tab& T = *tab;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const long long r0 = (long long)blockIdx.x * AP_ROWS;
  const long long m = S[1];
  const bool nw = (S[3] & 2) != 0;
  if (wid >= 2) {
    ap_serve(wid - 2);
    return;
  }
  ap_stamp(32 + 8 * blockIdx.x);
  const int half = lane >> 4, j = lane & 15, rb = 2 * wid + half;
  const long long r = r0 + rb;
  const bool alive = r < m;
  const int ns = nseg_of(T.len), depth = T.depth;
  u64* C = cs[wid];
  {
    const u64* R = P + A_ROW + (long long)A_RW * (r0 + 2 * wid + half);
    if (j < 8) C[j] = P[A_QS + j];
    else C[j] = P[A_LPS + j - 8];
    C[A3_R + 32 * half + j] = j < 15 ? R[j] : 0;
    C[A3_R + 32 * half + 16 + j] = R[SNAP + j];
    for (int i = lane; i < 10 * depth; i += 32) C[A3_N + i] = P[A_NS + SNAP * (i / 10) + i % 10];
  }
  const uint8_t* row = reinterpret_cast<const uint8_t*>(xq + (alive ? r : 0) * k);
  const uint8_t* tl = reinterpret_cast<const uint8_t*>(sa + (alive ? r : 0));
  const bool real = alive && j < ns;
  const int p0 = 1024 * j, L = real ? min(T.len, p0 + 1024) - p0 : 0, nb = real ? (L + 17 + 127) / 128 : 0;
  const int nf = real ? max(0, min(L, k - p0)) / 128 : 0;
  const int nq = __reduce_max_sync(0xffffffffu, (unsigned)nb);
  const int nbx = (64 * ns + 17 + 127) / 128;
  const bool wany = __any_sync(0xffffffffu, alive);
  uint4* dst = reinterpret_cast<uint4*>(slot + r * k + p0);
  u64 hs[8], x[8], c[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) hs[i] = real ? P[A_SS + SNAP * j + i] : 0;
  uint4 nx[8];
  if (nf > 0) {
    const uint4* g = reinterpret_cast<const uint4*>(row + p0);
#pragma unroll
    for (int i = 0; i < 8; ++i) nx[i] = __ldg(g + i);
  }
  __syncwarp();
  {
    const int n0 = (int)(C[A3_R + 8] & 127), n1 = (int)(C[A3_R + 32 + 8] & 127);
    bool bad = (n0 >> 3) != 5 || !(n0 & 7) || (n1 >> 3) != 5 || !(n1 & 7);
    for (int lv = 0; lv < depth; ++lv) bad |= (C[A3_N + 10 * lv + 8] & 127) != 1;
    if (bad) __trap();
  }
  const u64* Rh = C + A3_R + 32 * half;
  enum { PSEG, PX, PH, PF, PN };
  int ph = PSEG, q = 0, lv = 0;
  long long i0 = 0, off = 0, live = 0, idx = 0;
  bool keep = false;
  const long long width = 1LL << depth;
  const auto arrive = [&]() {
    ap_stamp(32 + 8 * blockIdx.x + 3);
    __threadfence();
    ap_sync(AP_CONSUMERS, 64);
    if (tid == 0) last = atomicAdd(reinterpret_cast<unsigned*>(P + A_CNT), 1u) == gridDim.x - 1;
    ap_sync(AP_CONSUMERS, 64);
    ap_stamp(32 + 8 * blockIdx.x + 4);
    if (!last) return false;
    if (tid == 0 && ap_stamps) ap_stamps[0] = blockIdx.x;
    __threadfence();
    ph = PN;
    q = 0;
    lv = 0;
    i0 = 0;
    off = 0;
    live = min(width >> 1, (m + 1) >> 1);
    return true;
  };
  // a warp with no live row has no x, h or leaf to hash
  const auto after_seg = [&]() {
    if (real)
#pragma unroll
      for (int i = 0; i < 8; ++i) dg[rb][j][i] = hs[i];
    if (!nw) {
      const int body = max(0, min(1024, k - p0)) / 16;
      if (alive) {
        const uint4* g = reinterpret_cast<const uint4*>(row + p0);
        for (int i = 8 * nf; i < body; ++i) dst[i] = g[i];
      } else {
        for (int i = 0; i < body; ++i) dst[i] = make_uint4(0, 0, 0, 0);
      }
      if (j == 0) scale[r] = alive ? sa[r] : 0.0f;
    }
    ap_stamp(32 + 8 * blockIdx.x + 1);
    __syncwarp();
    if (wany) {
      ph = PX;
      q = 0;
      return true;
    }
    return arrive();
  };
  if (nq == 0 && !after_seg()) {
    ap_stop();
    return;
  }
#pragma unroll 1
  for (;;) {
    u64 w[16], t[8];
    if (ph == PSEG) {
      if (q < nf) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          w[2 * i] = (u64)bswap32(nx[i].x) << 32 | bswap32(nx[i].y);
          w[2 * i + 1] = (u64)bswap32(nx[i].z) << 32 | bswap32(nx[i].w);
        }
        if (!nw)
#pragma unroll
          for (int i = 0; i < 8; ++i) dst[8 * q + i] = nx[i];
      } else if (q < nb && 128 * q == L && p0 + L <= k) {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = 0;
        w[0] = 0x80ull << 56;
        w[15] = 8ull * (128 + L);
      } else if (q < nb && p0 == k && L == 4) {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = 0;
        w[0] = (u64)bswap32(__float_as_uint(sa[alive ? r : 0])) << 32 | 0x80000000ull;
        w[15] = 8ull * (128 + 4);
      } else if (q < nb) {
        u64 v[16];
        a_seg_words(v, row, tl, k, p0, L, nb, q);
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = v[i];
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = 0;
      }
      if (q + 1 < nf) {
        const uint4* g = reinterpret_cast<const uint4*>(row + p0 + 128 * (q + 1));
#pragma unroll
        for (int i = 0; i < 8; ++i) nx[i] = __ldg(g + i);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) t[i] = hs[i];
    } else if (ph == PX) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int g = 16 * q + i;
        w[i] = g < 8 * ns ? dg[rb][g >> 3][g & 7] : (g == 8 * ns ? 0x80ull << 56 : 0);
      }
      if (q == nbx - 1) w[15] = 8ull * (128 + 64 * ns);
#pragma unroll
      for (int i = 0; i < 8; ++i) t[i] = q ? hs[i] : C[i];
    } else if (ph == PH) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        w[i] = q ? (i ? 0 : 0x80ull << 56) : x[i] ^ Rh[16 + i];
        w[8 + i] = q ? (i == 7 ? 8ull * 256 : 0) : Rh[24 + i];
        t[i] = q ? hs[i] : C[8 + i];
      }
    } else if (ph == PF) {
      // the frame prefix's n pending bytes (n >> 3 == 5), then h at byte n, SHA-512's padding, the length
      const u64 tot = Rh[8];
      const int sb = 8 * (int)(tot & 7);
#pragma unroll
      for (int i = 0; i < 5; ++i) w[i] = bswap64(Rh[9 + i]);
      w[5] = (bswap64(Rh[14]) & (~0ull << (64 - sb))) | (x[0] >> sb);
#pragma unroll
      for (int i = 1; i < 8; ++i) w[5 + i] = (x[i - 1] << (64 - sb)) | (x[i] >> sb);
      w[13] = (x[7] << (64 - sb)) | (0x80ull << (56 - sb));
      w[14] = 0;
      w[15] = 8 * (tot + 64);
#pragma unroll
      for (int i = 0; i < 8; ++i) t[i] = Rh[i];
    } else {
      // node idx of level lv: the prefix's pending byte, uint(idx) = 1 then idx, 64, the left child, 64, the right; its
      // children D at byte 3: D = [64, c 0..7, 64, c 8..15]
      const u64* N = C + A3_N + 10 * lv;
      if (q == 0) {
        idx = i0 + tid;
        keep = idx < live;
        const u64* in = T.level + 8 * (off + 2 * (keep ? idx : 0));
#pragma unroll
        for (int i = 0; i < 16; ++i) c[i] = __ldcg(in + i);
        w[0] = (N[9] & 0xFF) << 56 | 1ull << 48 | (u64)(idx & 0xFF) << 40;
        w[1] = 64ull << 40 | c[0] >> 24;
#pragma unroll
        for (int i = 2; i < 9; ++i) w[i] = c[i - 2] << 40 | c[i - 1] >> 24;
        w[9] = c[7] << 40;
        w[10] = 64ull << 40 | c[8] >> 24;
#pragma unroll
        for (int i = 11; i < 16; ++i) w[i] = c[i - 3] << 40 | c[i - 2] >> 24;
#pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = N[i];
      } else {
        w[0] = c[13] << 40 | c[14] >> 24;
        w[1] = c[14] << 40 | c[15] >> 24;
        w[2] = c[15] << 40 | 0x80ull << 32;
#pragma unroll
        for (int i = 3; i < 15; ++i) w[i] = 0;
        w[15] = 8 * (N[8] + 146);
#pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = hs[i];
      }
    }
    ap_cmp_in(t, w);
    if (ph == PSEG) {
      if (q < nb)
#pragma unroll
        for (int i = 0; i < 8; ++i) hs[i] = t[i];
      if (++q < nq) continue;
      if (!after_seg()) break;
    } else if (ph == PX) {
#pragma unroll
      for (int i = 0; i < 8; ++i) hs[i] = t[i];
      if (++q < nbx) continue;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = hs[i];
      ap_stamp(32 + 8 * blockIdx.x + 2);
      ph = PH;
      q = 0;
    } else if (ph == PH) {
#pragma unroll
      for (int i = 0; i < 8; ++i) hs[i] = t[i];
      if (++q < 2) continue;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = hs[i];
      ph = PF;
      q = 0;
    } else if (ph == PF) {
      if (alive && j == 0)
#pragma unroll
        for (int i = 0; i < 8; ++i) T.level[8 * r + i] = t[i];
      if (!arrive()) break;
    } else {
      if (q == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) hs[i] = t[i];
        q = 1;
        continue;
      }
      if (keep) {
        u64* o = T.level + 8 * (off + (width >> lv) + idx);
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = t[i];
        if (lv == depth - 1)
#pragma unroll
          for (int i = 0; i < 8; ++i) T.root[i] = t[i];
      }
      q = 0;
      i0 += 64;
      if (i0 < live) continue;
      off += width >> lv;
      ++lv;
      i0 = 0;
      __threadfence_block();
      ap_sync(AP_CONSUMERS, 64);
      ap_stamp(lv);
      if (lv == depth) {
        if (tid == 0) *reinterpret_cast<unsigned*>(P + A_CNT) = 0;
        break;
      }
      live = min(width >> (lv + 1), (m + (2LL << lv) - 1) >> (lv + 1));
    }
  }
  ap_stop();
}

// -- host ----------------------------------------------------------------------------------------------------------------
static H1Const hc_host;
static bool hc_set = false;
extern "C" int h1c_const(const H1Const* c) {
  hc_host = *c;
  hc_set = true;
  return (int)cudaMemcpyToSymbol(HC, c, sizeof(H1Const));
}
// h1c_a_call3_k's layout: a node prefix (frame, "node", 64, the domain, uint(lv), seven zeros) one byte past a block, a
// row's leaf frame prefix (frame, "leaf", 64, the domain, uint(i) twice, the schema, 64) 41 to 47 bytes past one for every
// row below AP_MAXCOUNT (uint(i) one or two bytes)
static bool a3_layout() {
  if (!hc_set || (hc_host.frame_len + 96) % 128 != 1) return false;
  for (int l = 1; l <= 2; ++l) {
    const int n = (hc_host.frame_len + hc_host.schema_len + 112 + 2 * l) % 128;
    if (n >> 3 != 5 || !(n & 7)) return false;
  }
  return true;
}

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

// h1c_a_call's kernel from here on: 0 h1c_a_call_k, 2 h1c_a_call2_k (paired compressions), 3 h1c_a_call3_k (paired, the
// messages in registers); set before any capture
static int a_form = 0;
extern "C" int h1c_a_form(int f) {
  if (f != 0 && f != 2 && f != 3) return 1;
  if (f == 2 || f == 3) {
    const cudaError_t e = cudaFuncSetAttribute(f == 2 ? h1c_a_call2_k : h1c_a_call3_k,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, AP_SMEM);
    if (e != cudaSuccess) return (int)e;
  }
  a_form = f;
  return 0;
}
// h1c_a_call2_k's and h1c_a_call3_k's phase stamps into p (32 + 8 T / AP_ROWS words), or none (null)
extern "C" int h1c_a_stamps(u64* p) { return (int)cudaMemcpyToSymbol(ap_stamps, &p, sizeof(p)); }

// one call's A table (tab: its host copy, tab_dev the device's; pre: its words); xq [M >= m, k] s8 and sa [M] f32 the
// call's operands, slot [T, k] and scale [T] the table's rows
extern "C" int h1c_a_call(const H1Tab* tab, const H1Tab* tab_dev, u64* pre, const int* S, const int8_t* xq, const float* sa,
                          int8_t* slot, float* scale, int k, cudaStream_t stream) {
  if (k % 16 || tab->len != k + 4 || tab->body != k || nseg_of(tab->len) > A_NSEG || tab->depth > A_DEPTH) return 1;
  if (a_form == 2 || a_form == 3) {
    if (tab->count % AP_ROWS || tab->count > AP_MAXCOUNT || nseg_of(tab->len) > AP_SEG) return 2;
    if (a_form == 3) {
      if (!a3_layout() || tab->depth > 9) return 3;
      h1c_a_call3_k<<<(unsigned)(tab->count / AP_ROWS), 128, AP_SMEM, stream>>>(tab_dev, pre, S, xq, sa, slot, scale, k);
    } else {
      h1c_a_call2_k<<<(unsigned)(tab->count / AP_ROWS), 128, AP_SMEM, stream>>>(tab_dev, pre, S, xq, sa, slot, scale, k);
    }
  } else {
    h1c_a_call_k<<<(unsigned)((tab->count + AW - 1) / AW), 32 * AW, 0, stream>>>(tab_dev, pre, S, xq, sa, slot, scale, k);
  }
  return (int)cudaGetLastError();
}

// h1c_build(mid); h1c_build(rnd) as two calls: h1c_tail_pre, which reads no leaf, then h1c_tail, every level.  Both sets'
// tables are given (no data), 1 to NMAXD levels deep, and rnd's leaves are written by mid's roots or before h1c_tail;
// snaps holds (nmid + nrnd) NMAXD NSNAP words, done one u32, zero before the first h1c_tail and left zero by each;
// host_root, if not null, 8 nrnd + 1 words of the caller's pinned host memory (cudaHostAlloc's, which a kernel writes
// through its host pointer under unified addressing): rnd's roots, then seq once they are out, so a host that reads
// seq there reads the roots with no copy and no wait on the stream.  h1c_tail_words: snaps' words a table.
extern "C" int h1c_tail_words() {
#if H1C_HB && H1C_NMID
  return NMAXD * NSNAP;
#else
  return 0;
#endif
}
extern "C" int h1c_tail_pre(const H1Tab* mid, H1Tab* mid_dev, int nmid, const H1Tab* rnd, H1Tab* rnd_dev, int nrnd,
                            const int* step_dev, u64* snaps, cudaStream_t stream) {
#if H1C_HB && H1C_NMID
  if (nmid <= 0 || nrnd <= 0 || nmid + nrnd > 65535) return 99;
  for (int t = 0; t < nmid + nrnd; ++t) {
    const H1Tab& T = t < nmid ? mid[t] : rnd[t - nmid];
    if (T.data || T.depth < 1 || T.depth > NMAXD || T.count > (1LL << T.depth) || (T.count << 1) <= (1LL << T.depth))
      return 100 + t;
  }
  h1c_tail_pre_k<<<nmid + nrnd, LT, 0, stream>>>(mid_dev, nmid, rnd_dev, step_dev, snaps);
  return (int)cudaGetLastError();
#else
  return 98;
#endif
}
extern "C" int h1c_tail(const H1Tab* mid_dev, int nmid, const H1Tab* rnd_dev, int nrnd, const u64* snaps, unsigned* done,
                        u64* host_root, u64 seq, cudaStream_t stream) {
#if H1C_HB && H1C_NMID
  if (nmid <= 0 || nrnd <= 0) return 99;
  h1c_tail_k<<<nmid, LT, 0, stream>>>(mid_dev, nmid, rnd_dev, nrnd, snaps, done, host_root, seq);
  return (int)cudaGetLastError();
#else
  return 98;
#endif
}

// n bytes of device memory into pinned host memory (cudaHostAlloc's) by a kernel's stores, not a copy engine's DMA, so
// the copy waits on no other stream's transfers
extern "C" __global__ void h1c_put_k(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
  __threadfence_system();
}
extern "C" int h1c_put(const void* src, void* dst, int n, cudaStream_t stream) {
  if (n <= 0) return 0;
  h1c_put_k<<<1, 256, 0, stream>>>(static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), n);
  return (int)cudaGetLastError();
}
extern "C" int h1c_sizes(int* out) {
  out[0] = (int)sizeof(H1Const);
  out[1] = (int)sizeof(H1Dom);
  out[2] = (int)sizeof(H1Tab);
  return 0;
}

// a set of tables (on the device, tabs_dev) through domains, segments, leaves and every level, on `stream`; the host copy
// (tabs) gives the grid's extents.  Tables whose leaves another set's roots write go in a later call.  lvf = L >= 0:
// the levels from L up in h1c_levels, one block a table (H1C_HB; a plain build has no h1c_levels and launches every
// level), L < 0 every level its own launch.  h1c_build is the build's H1C_LVF; a set of few deep tables, whose
// h1c_levels is a block or two on the whole GPU, takes a later L of its own here.
extern "C" int h1c_build_lv(const H1Tab* tabs, H1Tab* tabs_dev, int ntab, const int* step_dev, const u64* btab, int domains,
                            int lvf_from, cudaStream_t stream) {
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
#if H1C_HB
  const int ds = H1C_DOMSEG && domains && max_segs;
  if (domains && !ds) h1c_domains<<<(ntab + 63) / 64, 64, 0, stream>>>(tabs_dev, ntab, step_dev);
  if (max_segs) {
    const unsigned bx = (unsigned)max((max_segs + 127) / 128, ds ? (long long)(ntab + 127) / 128 : 1LL);
    h1c_segs<<<dim3(bx, ntab + ds), 128, 0, stream>>>(tabs_dev, ntab, step_dev, ds);
  }
#else
  if (domains) h1c_domains<<<(ntab + 63) / 64, 64, 0, stream>>>(tabs_dev, ntab, step_dev);
  if (max_segs) h1c_segs<<<dim3((unsigned)((max_segs + 127) / 128), ntab), 128, 0, stream>>>(tabs_dev, ntab);
#endif
  h1c_leaves<<<dim3((unsigned)((max_width + 127) / 128), ntab), 128, 0, stream>>>(tabs_dev, ntab, step_dev, btab);
#if H1C_LVB
  const int lv0 = min(LVB, max_depth);
#else
  const int lv0 = 0;
#endif
#if H1C_HB
  const int lvf = lvf_from < 0 ? max_depth : max(lv0, min(lvf_from, max_depth));
#else
  const int lvf = max_depth;
#endif
  for (int lv = lv0; lv < lvf; ++lv) {
    const long long nodes = max_width >> (lv + 1);
    h1c_level<<<dim3((unsigned)((nodes + 127) / 128), ntab), 128, 0, stream>>>(tabs_dev, ntab, lv);
  }
#if H1C_HB
  if (lvf < max_depth) h1c_levels<<<ntab, LT, 0, stream>>>(tabs_dev, ntab, lvf);
#endif
  return (int)cudaGetLastError();
}
extern "C" int h1c_build(const H1Tab* tabs, H1Tab* tabs_dev, int ntab, const int* step_dev, const u64* btab, int domains,
                         cudaStream_t stream) {
  return h1c_build_lv(tabs, tabs_dev, ntab, step_dev, btab, domains, H1C_LVF, stream);
}
