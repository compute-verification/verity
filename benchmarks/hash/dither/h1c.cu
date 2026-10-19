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

__device__ __forceinline__ int nseg_of(int len) { return (len + 1023) / 1024; }

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

// -- host ----------------------------------------------------------------------------------------------------------------
extern "C" int h1c_const(const H1Const* c) { return (int)cudaMemcpyToSymbol(HC, c, sizeof(H1Const)); }
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
