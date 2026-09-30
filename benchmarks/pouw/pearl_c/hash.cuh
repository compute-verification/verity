// Pearl-C's hashing kernels, arch-neutral (no wgmma, no bulk copies): BLAKE3's compression, hashing format -h1 (the message
// digests, the tile leaves and the frame-b3 trees of A's rows and of the tile leaves, as `verity_pouw.audit` commits them) and
// Pearl's lottery tickets.  pearl_c.cu includes it for sm_90a; hash.cu builds it alone for sm_120a.
#pragma once
#include <cstdint>

struct Args { uint64_t p[16]; int32_t i[16]; };

#define DEV __device__ __forceinline__

__device__ const uint32_t B3_IV[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
DEV uint32_t rotr(uint32_t x, int n) { return __funnelshift_r(x, x, n); }
DEV void b3g(uint32_t* v, int a, int b, int c, int d, uint32_t x, uint32_t y) {
  v[a] = v[a] + v[b] + x; v[d] = rotr(v[d] ^ v[a], 16); v[c] = v[c] + v[d]; v[b] = rotr(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + y; v[d] = rotr(v[d] ^ v[a], 8); v[c] = v[c] + v[d]; v[b] = rotr(v[b] ^ v[c], 7);
}
enum : uint32_t { B3_START = 1, B3_END = 2, B3_PARENT = 4, B3_ROOT = 8, B3_KEYED = 16 };
// one BLAKE3 compression: the first 8 output words (a chaining value, or under ROOT the hash); out may alias cv
DEV void b3c(const uint32_t* cv, const uint32_t* msg, uint64_t counter, uint32_t len, uint32_t flags, uint32_t* out) {
  constexpr int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t v[16], m[16], t[16];
#pragma unroll
  for (int i = 0; i < 8; i++) { v[i] = cv[i]; v[i + 8] = i < 4 ? B3_IV[i] : 0; }
  v[12] = (uint32_t)counter; v[13] = (uint32_t)(counter >> 32); v[14] = len; v[15] = flags;
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = msg[i];
#pragma unroll
  for (int r = 0; r < 7; r++) {
    b3g(v, 0, 4, 8, 12, m[0], m[1]); b3g(v, 1, 5, 9, 13, m[2], m[3]); b3g(v, 2, 6, 10, 14, m[4], m[5]); b3g(v, 3, 7, 11, 15, m[6], m[7]);
    b3g(v, 0, 5, 10, 15, m[8], m[9]); b3g(v, 1, 6, 11, 12, m[10], m[11]); b3g(v, 2, 7, 8, 13, m[12], m[13]); b3g(v, 3, 4, 9, 14, m[14], m[15]);
#pragma unroll
    for (int i = 0; i < 16; i++) t[i] = m[PERM[i]];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = t[i];
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}
// one keyed BLAKE3 hash of a single 64-byte block (CHUNK_START | CHUNK_END | ROOT | KEYED_HASH), the first 32 bytes
DEV void blake3_block(const uint32_t* key, const uint32_t* msg, uint32_t* out) {
  b3c(key, msg, 0, 64, B3_START | B3_END | B3_ROOT | B3_KEYED, out);
}


// ---------------------------------------------------------------- hashing, format -h1: BLAKE3 throughout
// The trees are `verity.commitments.merkle`'s frame-b3 (hash="blake3"), keyed by each tree's domain id, which the host
// supplies: leaf 0x01 ‖ u64 rank ‖ u64 position ‖ u16 len(schema) ‖ schema ‖ value, node 0x02 ‖ u8 level ‖ u64 index ‖
// left ‖ right, pad 0x03 ‖ u64 rank.  Integers little-endian.  A tree's levels sit one after another in one buffer
// (level 0 at offset 0, padded to a power of two), 32 bytes per node.
__constant__ uint8_t ROW_SCHEMA[18] = {'v', 'e', 'r', 'i', 't', 'y', '/', 'p', 'o', 'u', 'w', '/', 'r', 'o', 'w', '/', 'v', '1'};
__constant__ uint8_t TILE_SCHEMA[19] = {'v', 'e', 'r', 'i', 't', 'y', '/', 'p', 'o', 'u', 'w', '/', 't', 'i', 'l', 'e', '/', 'v', '1'};
constexpr int ROW_PREFIX = 1 + 8 + 8 + 2 + 18, MAX_CHUNKS = 260;   // a row of at most 2^16 FP32 words: 257 chunks

DEV void load_key(uint64_t p, uint32_t* k) {   // 32 bytes at device address p as 8 words
#pragma unroll
  for (int i = 0; i < 8; i++) k[i] = reinterpret_cast<const uint32_t*>(p)[i];
}
DEV void store_cv(uint64_t p, const uint32_t* h) {
  reinterpret_cast<uint4*>(p)[0] = make_uint4(h[0], h[1], h[2], h[3]);
  reinterpret_cast<uint4*>(p)[1] = make_uint4(h[4], h[5], h[6], h[7]);
}
DEV void b3_parent(const uint32_t* key, const uint32_t* l, const uint32_t* r, uint32_t flags, uint32_t* out) {
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 8; i++) { m[i] = l[i]; m[i + 8] = r[i]; }
  b3c(key, m, 0, 64, B3_KEYED | B3_PARENT | flags, out);
}
DEV uint32_t word_le(uint64_t v, int j) { return (uint32_t)(v >> (8 * j)) & 0xFF; }
DEV uint32_t cv_byte(const uint32_t* h, int j) { return (h[j >> 2] >> (8 * (j & 3))) & 0xFF; }
// one keyed BLAKE3 hash of an N-byte message (N <= 128, two blocks at most) whose byte j is get(j), j a constant after unrolling
template <int N, class Get> DEV void b3_short(const uint32_t* key, const Get& get, uint32_t* out) {
  constexpr int BLOCKS = (N + 63) / 64;
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = key[i];
#pragma unroll
  for (int b = 0; b < BLOCKS; b++) {
    uint32_t m[16];
#pragma unroll
    for (int w = 0; w < 16; w++) {
      uint32_t x = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) x |= (64 * b + 4 * w + j < N ? get(64 * b + 4 * w + j) : 0u) << (8 * j);
      m[w] = x;
    }
    const uint32_t f = B3_KEYED | (b == 0 ? B3_START : 0) | (b == BLOCKS - 1 ? B3_END | B3_ROOT : 0);
    b3c(cv, m, 0, b == BLOCKS - 1 ? N - 64 * b : 64, f, cv);
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// message digests: keyed BLAKE3 (key blake3("pearl-c/h1/message")) of le32 C~ of the 32-column chunk ‖ le32 U of it, per
// (row, chunk): one chunk of four blocks.  p0 C~ (rows x cols), p1 U, p2 out (rows x cols/32 x 32 B), p3 key; i0 rows, i1 cols
extern "C" __global__ void __launch_bounds__(128) hash_msg(const Args a) {
  const int cols = a.i[1], per = cols / 32;
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)a.i[0] * per) return;
  const uint32_t* c = reinterpret_cast<const uint32_t*>(a.p[0]) + t * 32;
  const uint32_t* u = reinterpret_cast<const uint32_t*>(a.p[1]) + t * 32;
  uint32_t key[8], cv[8];
  load_key(a.p[3], key);
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = key[i];
#pragma unroll
  for (int b = 0; b < 4; b++) {
    uint32_t m[16];
#pragma unroll
    for (int w = 0; w < 16; w++) m[w] = b < 2 ? c[16 * b + w] : u[16 * (b - 2) + w];
    b3c(cv, m, 0, 64, B3_KEYED | (b == 0 ? B3_START : 0) | (b == 3 ? B3_END | B3_ROOT : 0), cv);
  }
  store_cv(a.p[2] + t * 32, cv);
}

// tile leaves: keyed BLAKE3 (key blake3("pearl-c/h1/leaf")) of a 64 x 64 tile's 128 message digests, rows in order, 2 chunks
// each: 4,096 bytes, four chunks under two parent levels.  p0 digests, p1 out (tiles row-major x 32 B), p2 key; i0 rows, i1 cols
extern "C" __global__ void __launch_bounds__(64) hash_leaf(const Args a) {
  const int cols = a.i[1], tn = cols / 64, per = cols / 32;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] / 64 * tn) return;
  const int ti = t / tn, tj = t % tn;
  const uint32_t* dg = reinterpret_cast<const uint32_t*>(a.p[0]);
  uint32_t key[8], cvs[4][8];
  load_key(a.p[2], key);
#pragma unroll
  for (int ch = 0; ch < 4; ch++) {
#pragma unroll
    for (int i = 0; i < 8; i++) cvs[ch][i] = key[i];
#pragma unroll 1
    for (int b = 0; b < 16; b++) {
      uint32_t m[16];
#pragma unroll
      for (int w = 0; w < 16; w++) {
        const int d = 32 * ch + 2 * b + w / 8;          // the tile's digest: row d / 2, its chunk d % 2
        m[w] = dg[((size_t)(64 * ti + d / 2) * per + 2 * tj + d % 2) * 8 + w % 8];
      }
      b3c(cvs[ch], m, ch, 64, B3_KEYED | (b == 0 ? B3_START : 0) | (b == 15 ? B3_END : 0), cvs[ch]);
    }
  }
  uint32_t l[8], r[8], h[8];
  b3_parent(key, cvs[0], cvs[1], 0, l);
  b3_parent(key, cvs[2], cvs[3], 0, r);
  b3_parent(key, l, r, B3_ROOT, h);
  store_cv(a.p[1] + (size_t)t * 32, h);
}

// A's rows as the verifier commits them: each row's frame-b3 leaf (schema verity/pouw/row/v1, rank = position = the row),
// one CTA per row: its chunks in parallel, then BLAKE3's left-heavy tree, whose perfect blocks reduce level by level and
// whose block roots fold right to left.  p0 rows, p1 out (32 B per row: level 0 of the tree), p2 domain id; i0 rows, i1 row bytes
DEV uint32_t row_msg_byte(const uint8_t* src, uint32_t rb, uint64_t row, uint64_t i) {
  if (i >= ROW_PREFIX) return i - ROW_PREFIX < rb ? src[i - ROW_PREFIX] : 0u;
  if (i == 0) return 1u;
  if (i < 17) return word_le(row, (int)((i - 1) % 8));
  if (i == 17) return 18u;
  if (i == 18) return 0u;
  return ROW_SCHEMA[i - 19];
}
DEV int block_size(int n, int i) {   // the size of the perfect block of BLAKE3's tree over n chunks that holds chunk i
  int off = 0;
  for (int b = 30; b >= 0; b--)
    if (n >> b & 1) {
      if (i < off + (1 << b)) return 1 << b;
      off += 1 << b;
    }
  return 0;
}
extern "C" __global__ void __launch_bounds__(256) hash_rows(const Args a) {
  __shared__ uint32_t cvs[MAX_CHUNKS][8];
  const uint64_t row = blockIdx.x;
  const uint32_t rb = (uint32_t)a.i[1];
  const uint64_t len = ROW_PREFIX + (uint64_t)rb;
  const int n = (int)((len + 1023) / 1024);
  const uint8_t* src = reinterpret_cast<const uint8_t*>(a.p[0]) + row * rb;
  uint32_t key[8];
  load_key(a.p[2], key);
  for (int c = threadIdx.x; c < n; c += blockDim.x) {
    const uint64_t left = len - 1024ull * c;
    const uint32_t clen = (uint32_t)(left < 1024 ? left : 1024);
    const int blocks = (int)((clen + 63) / 64);
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    for (int b = 0; b < blocks; b++) {
      uint32_t m[16];
      const uint64_t base = 1024ull * c + 64ull * b;
#pragma unroll
      for (int w = 0; w < 16; w++) {
        uint32_t x = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint64_t i = base + 4 * w + j;
          x |= (i < len ? row_msg_byte(src, rb, row, i) : 0u) << (8 * j);
        }
        m[w] = x;
      }
      const bool last = b == blocks - 1;
      const uint32_t f = B3_KEYED | (b == 0 ? B3_START : 0) | (last ? B3_END : 0) | (last && n == 1 ? B3_ROOT : 0);
      b3c(cv, m, (uint64_t)c, last ? clen - 64 * b : 64, f, cv);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) cvs[c][i] = cv[i];
  }
  __syncthreads();
  for (int s = 1; s < n; s <<= 1) {
    for (int i = 2 * s * threadIdx.x; i < n; i += 2 * s * blockDim.x)
      if (i + s < n && block_size(n, i) >= 2 * s) b3_parent(key, cvs[i], cvs[i + s], 2 * s == n ? B3_ROOT : 0, cvs[i]);
    __syncthreads();
  }
  if (threadIdx.x != 0) return;
  uint32_t h[8];
  const int top = 31 - __clz(n);
  if (n == 1 << top) {
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = cvs[0][i];
  } else {
    int off = n - (n & -n);             // the last perfect block, then fold its left neighbours in
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = cvs[off][i];
    int rest = n - (n & -n);
    while (rest) {
      const int low = rest & -rest;
      off = rest - low;
      rest -= low;
      b3_parent(key, cvs[off], h, rest ? 0 : B3_ROOT, h);
    }
  }
  store_cv(a.p[1] + row * 32, h);
}

// pads of a frame-b3 tree: ranks i0 … i1 − 1 of level 0.  p0 tree, p1 domain id; i0 first pad rank, i1 the padded width
extern "C" __global__ void __launch_bounds__(128) hash_pad(const Args a) {
  const uint64_t rank = (uint64_t)a.i[0] + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= (uint64_t)a.i[1]) return;
  uint32_t key[8], h[8];
  load_key(a.p[1], key);
  b3_short<9>(key, [&](int j) { return j == 0 ? 3u : word_le(rank, j - 1); }, h);
  store_cv(a.p[0] + rank * 32, h);
}

// one level of a frame-b3 tree: node i of level i0 + 1 from nodes 2i and 2i + 1 of level i0.  p0 the level below, p1 the level
// above, p2 domain id; i0 the level below's index, i1 nodes above
extern "C" __global__ void __launch_bounds__(128) hash_nodes(const Args a) {
  const uint64_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (uint64_t)a.i[1]) return;
  uint32_t key[8], l[8], r[8], h[8];
  load_key(a.p[2], key);
  load_key(a.p[0] + 64 * i, l);
  load_key(a.p[0] + 64 * i + 32, r);
  const uint32_t level = (uint32_t)a.i[0];
  b3_short<74>(key, [&](int j) {
    return j == 0 ? 2u : j == 1 ? level : j < 10 ? word_le(i, j - 2) : j < 42 ? cv_byte(l, j - 10) : cv_byte(r, j - 42);
  }, h);
  store_cv(a.p[1] + i * 32, h);
}

// the tile tree's leaves: tile t's frame-b3 leaf over its 32-byte leaf digest (schema verity/pouw/tile/v1, rank = position = t).
// p0 tile leaves, p1 out (level 0 of the tree), p2 domain id; i0 tiles
extern "C" __global__ void __launch_bounds__(128) hash_tree_leaves(const Args a) {
  const uint64_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (uint64_t)a.i[0]) return;
  uint32_t key[8], d[8], h[8];
  load_key(a.p[2], key);
  load_key(a.p[0] + t * 32, d);
  b3_short<70>(key, [&](int j) {
    return j == 0 ? 1u : j < 17 ? word_le(t, (j - 1) % 8) : j == 17 ? 19u : j == 18 ? 0u : j < 38 ? (uint32_t)TILE_SCHEMA[j - 19]
                                                                                      : cv_byte(d, j - 38);
  }, h);
  store_cv(a.p[1] + t * 32, h);
}

// Pearl's tickets: per 16 x 16 subtile of C~, the 16-lane fold (lane (rb, cb) folds its 4 x 4 block row-major,
// lane <- rotl32(lane 0x9E3779B1 + bits, 13)) hashed as one keyed BLAKE3 block.  p0 C~, p1 out (subtiles row-major x 32 B),
// p2 key (8 words); i0 rows, i1 cols
extern "C" __global__ void __launch_bounds__(128) tickets(const Args a) {
  const int cols = a.i[1], sn = cols / 16;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] / 16 * sn) return;
  const int si = t / sn, sj = t % sn;
  const uint32_t* C = reinterpret_cast<const uint32_t*>(a.p[0]);
  uint32_t z[16], kw[8], h[8];
#pragma unroll
  for (int l = 0; l < 16; l++) {
    uint32_t lane = 0;
#pragma unroll
    for (int f = 0; f < 16; f++) {
      lane = lane * 0x9E3779B1u + C[(size_t)(16 * si + 4 * (l / 4) + f / 4) * cols + 16 * sj + 4 * (l % 4) + f % 4];
      lane = __funnelshift_l(lane, lane, 13);
    }
    z[l] = lane;
  }
#pragma unroll
  for (int i = 0; i < 8; i++) kw[i] = reinterpret_cast<const uint32_t*>(a.p[2])[i];
  blake3_block(kw, z, h);
  uint4* out = reinterpret_cast<uint4*>(a.p[1] + (size_t)t * 32);
  out[0] = make_uint4(h[0], h[1], h[2], h[3]);
  out[1] = make_uint4(h[4], h[5], h[6], h[7]);
}


// ---------------------------------------------------------------- -h1 on sm_120: the same bytes, fewer serial steps
// The kernels below commit exactly what hash_rows, hash_nodes, hash_leaf and hash_msg do; the host picks either form.

// one -h1 message digest from a lane's 32 C~ words and 32 U words (registers or memory), for hash_msg_v and GEMM epilogues
DEV void b3_msg_digest(const uint32_t* c, const uint32_t* u, const uint32_t* key, uint32_t* out) {
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = key[i];
#pragma unroll
  for (int b = 0; b < 4; b++)
    b3c(out, (b < 2 ? c : u) + 16 * (b & 1), 0, 64, B3_KEYED | (b == 0 ? B3_START : 0) | (b == 3 ? B3_END | B3_ROOT : 0), out);
}

// hash_msg with 16-byte loads.  Same arguments.
extern "C" __global__ void __launch_bounds__(128) hash_msg_v(const Args a) {
  const int per = a.i[1] / 32;
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)a.i[0] * per) return;
  const uint4* c4 = reinterpret_cast<const uint4*>(a.p[0]) + t * 8;
  const uint4* u4 = reinterpret_cast<const uint4*>(a.p[1]) + t * 8;
  uint32_t c[32], u[32], key[8], h[8];
#pragma unroll
  for (int q = 0; q < 8; q++) {
    const uint4 x = c4[q], y = u4[q];
    c[4 * q] = x.x; c[4 * q + 1] = x.y; c[4 * q + 2] = x.z; c[4 * q + 3] = x.w;
    u[4 * q] = y.x; u[4 * q + 1] = y.y; u[4 * q + 2] = y.z; u[4 * q + 3] = y.w;
  }
  load_key(a.p[3], key);
  b3_msg_digest(c, u, key, h);
  store_cv(a.p[2] + t * 32, h);
}

// A row's message byte i, i < ROW_PREFIX: the frame-b3 leaf prefix (rank = position = row)
DEV uint32_t row_prefix_byte(uint64_t row, int i) {
  if (i == 0) return 1u;
  if (i < 17) return word_le(row, (i - 1) % 8);
  if (i == 17) return 18u;
  if (i == 18) return 0u;
  return ROW_SCHEMA[i - 19];
}
// shared words hash_rows_w needs per warp for a row of rb bytes: its chunks' chaining values
__host__ __device__ inline int rows_w_words(int rb) { return 8 * ((ROW_PREFIX + rb + 1023) / 1024); }

// A row's frame-b3 leaf by one warp: lane c hashes chunks c, c + 32, … straight from the row (each message word one funnel
// shift of two row words when the row is word-aligned, the next block's words loaded while this block compresses; the
// prefix's and the tail's words byte by byte), the chaining values go to cvs (8 words a chunk), and the chunk tree is built
// as hash_rows builds it.  Lane 0 returns the leaf in h.
DEV void row_leaf_w(const uint8_t* src, uint32_t rb, uint64_t row, const uint32_t* key, uint32_t (*cvs)[8], int lane, uint32_t* h) {
  const uint64_t len = ROW_PREFIX + (uint64_t)rb;
  const int n = (int)((len + 1023) / 1024);
  const uint32_t* src_w = reinterpret_cast<const uint32_t*>(src);
  const bool aligned = rb % 4 == 0 && (reinterpret_cast<uint64_t>(src) & 3) == 0;
  for (int c = lane; c < n; c += 32) {
    const uint64_t left = len - 1024ull * c;
    const uint32_t clen = (uint32_t)(left < 1024 ? left : 1024);
    const int blocks = (int)((clen + 63) / 64);
    auto fast = [&](int b) { const uint64_t j0 = 256ull * c + 16ull * b; return aligned && j0 >= 10 && 4 * (j0 + 6) + 3 < rb; };
    uint32_t cv[8], nxt[17];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    bool nxt_fast = fast(0);
    if (nxt_fast) {
#pragma unroll
      for (int q = 0; q < 17; q++) nxt[q] = src_w[256ull * c - 10 + q];
    }
    for (int b = 0; b < blocks; b++) {
      const uint64_t j0 = 256ull * c + 16ull * b;             // the block's first message word
      uint32_t cur[17], m[16];
      const bool cur_fast = nxt_fast;
#pragma unroll
      for (int q = 0; q < 17; q++) cur[q] = nxt[q];
      if (b + 1 < blocks) {
        nxt_fast = fast(b + 1);
        if (nxt_fast) {
#pragma unroll
          for (int q = 0; q < 17; q++) nxt[q] = src_w[j0 + 16 - 10 + q];
        }
      }
      if (cur_fast) {                                         // message word j is bytes 4j − 37 … of the row
#pragma unroll
        for (int w = 0; w < 16; w++) m[w] = __funnelshift_r(cur[w], cur[w + 1], 24);
      } else {
#pragma unroll
        for (int w = 0; w < 16; w++) {
          uint32_t x = 0;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const uint64_t i = 4 * (j0 + w) + k;
            const uint32_t v = i < ROW_PREFIX ? row_prefix_byte(row, (int)i) : i < len ? (uint32_t)src[i - ROW_PREFIX] : 0u;
            x |= v << (8 * k);
          }
          m[w] = x;
        }
      }
      const bool last = b == blocks - 1;
      const uint32_t f = B3_KEYED | (b == 0 ? B3_START : 0) | (last ? B3_END : 0) | (last && n == 1 ? B3_ROOT : 0);
      b3c(cv, m, (uint64_t)c, last ? clen - 64 * b : 64, f, cv);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) cvs[c][i] = cv[i];
  }
  __syncwarp();
  for (int s = 1; s < n; s <<= 1) {
    for (int i = 2 * s * lane; i < n; i += 64 * s)
      if (i + s < n && block_size(n, i) >= 2 * s) b3_parent(key, cvs[i], cvs[i + s], 2 * s == n ? B3_ROOT : 0, cvs[i]);
    __syncwarp();
  }
  if (lane != 0) return;
  const int top = 31 - __clz(n);
  if (n == 1 << top) {
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = cvs[0][i];
  } else {
    int off = n - (n & -n);
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = cvs[off][i];
    int rest = n - (n & -n);
    while (rest) {
      const int low = rest & -rest;
      off = rest - low;
      rest -= low;
      b3_parent(key, cvs[off], h, rest ? 0 : B3_ROOT, h);
    }
  }
}

// hash_rows, one warp per row and blockDim / 32 rows per CTA (row_leaf_w).  Dynamic shared memory: rows_w_words(rb) words per
// warp.  Same arguments as hash_rows.
extern "C" __global__ void __launch_bounds__(256) hash_rows_w(const Args a) {
  extern __shared__ uint32_t sh[];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const uint64_t row = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp;
  const uint32_t rb = (uint32_t)a.i[1];
  if (row >= (uint64_t)a.i[0]) return;
  uint32_t key[8], h[8];
  load_key(a.p[2], key);
  row_leaf_w(reinterpret_cast<const uint8_t*>(a.p[0]) + row * rb, rb, row, key,
             reinterpret_cast<uint32_t(*)[8]>(sh + (size_t)warp * rows_w_words((int)rb)), lane, h);
  if (lane == 0) store_cv(a.p[1] + row * 32, h);
}

// A frame-b3 tree over m ≤ 32 rows in one CTA of 32m threads: each warp its row's leaf (row_leaf_w), the pads, then every
// level from shared memory; the levels written where hash_rows_w, hash_pad and hash_tree_small leave them.  Dynamic shared
// memory: m · rows_w_words(rb) words, then 2 · width · 8.  Same arguments as hash_rows; the tree buffer at p1.
extern "C" __global__ void __launch_bounds__(1024) hash_rows_tree(const Args a) {
  extern __shared__ uint32_t sh[];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, m = a.i[0];
  const uint32_t rb = (uint32_t)a.i[1];
  const int width = 1 << (32 - __clz(m > 1 ? m - 1 : 1)) >> (m > 1 ? 0 : 1);
  uint32_t(*lv)[8] = reinterpret_cast<uint32_t(*)[8]>(sh + (size_t)m * rows_w_words((int)rb));
  uint32_t key[8], h[8];
  load_key(a.p[2], key);
  if (warp < m) {
    row_leaf_w(reinterpret_cast<const uint8_t*>(a.p[0]) + (uint64_t)warp * rb, rb, (uint64_t)warp, key,
               reinterpret_cast<uint32_t(*)[8]>(sh + (size_t)warp * rows_w_words((int)rb)), lane, h);
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < 8; i++) lv[warp][i] = h[i];
      store_cv(a.p[1] + (uint64_t)warp * 32, h);
    }
  }
  for (int r = m + (int)threadIdx.x; r < width; r += blockDim.x) {
    b3_short<9>(key, [&](int j) { return j == 0 ? 3u : word_le((uint64_t)r, j - 1); }, h);
#pragma unroll
    for (int i = 0; i < 8; i++) lv[r][i] = h[i];
    store_cv(a.p[1] + (uint64_t)r * 32, h);
  }
  __syncthreads();
  uint32_t(*below)[8] = lv;
  uint64_t out = a.p[1] + 32ull * width;
  uint32_t level = 0;
  for (int w = width; w > 1; w >>= 1, level++) {
    uint32_t(*above)[8] = below + w;
    for (int i = threadIdx.x; i < w / 2; i += blockDim.x) {
      b3_short<74>(key, [&](int j) {
        return j == 0 ? 2u : j == 1 ? level : j < 10 ? word_le((uint64_t)i, j - 2) : j < 42 ? cv_byte(below[2 * i], j - 10)
                                                                                             : cv_byte(below[2 * i + 1], j - 42);
      }, h);
#pragma unroll
      for (int q = 0; q < 8; q++) above[i][q] = h[q];
      store_cv(out + 32ull * i, h);
    }
    __syncthreads();
    below = above;
    out += 32ull * (w / 2);
  }
}

// every level of a frame-b3 tree in one CTA, for a level 0 (pads included, width a power of two) of at most 2 blockDim nodes:
// the levels hash_nodes writes one launch at a time, in the same places.  p0 the tree (level 0 at offset 0), p1 domain id;
// i0 the width of level 0
extern "C" __global__ void __launch_bounds__(1024) hash_tree_small(const Args a) {
  uint32_t key[8];
  load_key(a.p[1], key);
  uint64_t below = a.p[0];
  uint32_t level = 0;
  for (uint64_t w = (uint64_t)a.i[0]; w > 1; w >>= 1, level++) {
    const uint64_t above = below + 32 * w;
    for (uint64_t i = threadIdx.x; i < w / 2; i += blockDim.x) {
      uint32_t l[8], r[8], h[8];
      load_key(below + 64 * i, l);
      load_key(below + 64 * i + 32, r);
      b3_short<74>(key, [&](int j) {
        return j == 0 ? 2u : j == 1 ? level : j < 10 ? word_le(i, j - 2) : j < 42 ? cv_byte(l, j - 10) : cv_byte(r, j - 42);
      }, h);
      store_cv(above + 32 * i, h);
    }
    __syncthreads();
    below = above;
  }
}

// hash_leaf with each of a tile's chunks (32 digests) on its own thread, four threads a tile and 32 tiles a CTA of 128, and
// with partial tiles: a tile of r < 64 rows (the decode headline's m = 32) is its 2r digests in ⌈2r / 32⌉ chunks under
// BLAKE3's tree, as the reference's leaf() hashes it.  Launch ⌈tiles / 32⌉ CTAs of 128.  Same arguments as hash_leaf.
extern "C" __global__ void __launch_bounds__(128) hash_leaf_p(const Args a) {
  __shared__ uint32_t cvs[32][4][8];
  const int rows = a.i[0], cols = a.i[1], tn = cols / 64, per = cols / 32;
  const int lt = threadIdx.x / 4, ch = threadIdx.x % 4;
  const int t = blockIdx.x * 32 + lt, tiles = (rows + 63) / 64 * tn;
  const int ti = t / tn, tj = t % tn;
  const int r = t < tiles ? min(64, rows - 64 * ti) : 0, nd = 2 * r, nch = (nd + 31) / 32;
  uint32_t key[8];
  load_key(a.p[2], key);
  if (ch < nch) {
    const uint32_t* dg = reinterpret_cast<const uint32_t*>(a.p[0]);
    const int d0 = 32 * ch, dn = min(32, nd - d0), blocks = (dn + 1) / 2;
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
#pragma unroll 1
    for (int b = 0; b < blocks; b++) {
      uint32_t m[16];
#pragma unroll
      for (int w = 0; w < 16; w++) {
        const int d = d0 + 2 * b + w / 8;
        m[w] = d < nd ? dg[((size_t)(64 * ti + d / 2) * per + 2 * tj + d % 2) * 8 + w % 8] : 0u;
      }
      const bool last = b == blocks - 1;
      const uint32_t f = B3_KEYED | (b == 0 ? B3_START : 0) | (last ? B3_END | (nch == 1 ? B3_ROOT : 0) : 0);
      b3c(cv, m, (uint64_t)ch, last && (dn & 1) ? 32 : 64, f, cv);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) cvs[lt][ch][i] = cv[i];
  }
  __syncthreads();
  if (ch != 0 || t >= tiles) return;
  uint32_t l[8], h[8];
  if (nch == 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = cvs[lt][0][i];
  } else if (nch == 2) {
    b3_parent(key, cvs[lt][0], cvs[lt][1], B3_ROOT, h);
  } else if (nch == 3) {
    b3_parent(key, cvs[lt][0], cvs[lt][1], 0, l);
    b3_parent(key, l, cvs[lt][2], B3_ROOT, h);
  } else {
    uint32_t r2[8];
    b3_parent(key, cvs[lt][0], cvs[lt][1], 0, l);
    b3_parent(key, cvs[lt][2], cvs[lt][3], 0, r2);
    b3_parent(key, l, r2, B3_ROOT, h);
  }
  store_cv(a.p[1] + (size_t)t * 32, h);
}
