// Pearl-C's hashing kernels, arch-neutral (no wgmma, no bulk copies): BLAKE3's compression, hashing format -h1 (the message
// digests, the tile leaves and the frame-b3 trees of A's rows and of the tile leaves, as `verity.core.protocols.pouw.audit` commits them), the
// unit's seed_A over A's root and its subkeys, and Pearl's lottery tickets.  pearl_c.cu includes it for sm_90a; hash.cu builds it
// alone for sm_120a.
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
constexpr uint32_t B3_START = 1, B3_END = 2, B3_PARENT = 4, B3_ROOT = 8, B3_KEYED = 16;
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
// The trees are `verity.ml.commitments.merkle`'s frame-b3 (hash="blake3"), keyed by each tree's domain id, which the host
// supplies: leaf H ‖ value with H = 0x01 ‖ u64 rank ‖ u64 position ‖ u16 len(schema) ‖ schema zero-padded to one 64-byte
// block; pad 0x03 ‖ u64 rank; a level-l node left ‖ right (one block) under the level's key keyed(domain id, 0x02 ‖ u8 l).
// Integers little-endian.  A tree's levels sit one after another in one buffer (level 0 at offset 0, padded to a power of
// two), 32 bytes per node.  Every launch takes a stream (run.py), so the post-GEMM hashing can go on a side stream.
DEV uint32_t word_le(uint64_t v, int j) { return (uint32_t)(v >> (8 * j)) & 0xFF; }
DEV uint32_t cv_byte(const uint32_t* h, int j) { return (h[j >> 2] >> (8 * (j & 3))) & 0xFF; }
__constant__ uint8_t ROW_SCHEMA[18] = {'v', 'e', 'r', 'i', 't', 'y', '/', 'p', 'o', 'u', 'w', '/', 'r', 'o', 'w', '/', 'v', '1'};
__constant__ uint8_t TILE_SCHEMA[19] = {'v', 'e', 'r', 'i', 't', 'y', '/', 'p', 'o', 'u', 'w', '/', 't', 'i', 'l', 'e', '/', 'v', '1'};
constexpr int ROW_PREFIX = 64, MAX_CHUNKS = 260;   // the leaf header's block; a row of at most 2^16 FP32 words: 257 chunks
// byte j < 64 of a frame-b3 leaf header for rank = position = r and a schema of `len` bytes from `schema`
DEV uint32_t header_byte(int j, uint64_t r, const uint8_t* schema, int len) {
  if (j == 0) return 1u;
  if (j < 17) return word_le(r, (j - 1) % 8);
  if (j == 17) return (uint32_t)len;
  if (j == 18) return 0u;
  return j < 19 + len ? (uint32_t)schema[j - 19] : 0u;
}

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

// tile leaves: keyed BLAKE3 (key blake3("pearl-c/h1/leaf")) of a tile's message digests, rows in order, 2 chunks each: a 64 x 64
// tile's 128 (4,096 bytes, four chunks under two parent levels), and a last, partial row block of 16, 32 or 48 rows (the
// decode tile at m = 32: 64 digests, two chunks under one parent), BLAKE3's left-heavy tree over its 1 to 4 chunks.
// p0 digests, p1 out (tiles row-major x 32 B), p2 key; i0 rows (a multiple of 16), i1 cols
extern "C" __global__ void __launch_bounds__(64) hash_leaf(const Args a) {
  const int rows = a.i[0], cols = a.i[1], tn = cols / 64, per = cols / 32;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (rows + 63) / 64 * tn) return;
  const int ti = t / tn, tj = t % tn;
  const int nc = min(64, rows - 64 * ti) / 16;       // chunks: 32 digests, 16 of the tile's rows, each
  const uint32_t* dg = reinterpret_cast<const uint32_t*>(a.p[0]);
  uint32_t key[8], cvs[4][8];
  load_key(a.p[2], key);
#pragma unroll
  for (int ch = 0; ch < 4; ch++) {
    if (ch >= nc) break;
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
      b3c(cvs[ch], m, ch, 64, B3_KEYED | (b == 0 ? B3_START : 0) | (b == 15 ? B3_END | (nc == 1 ? B3_ROOT : 0) : 0), cvs[ch]);
    }
  }
  uint32_t l[8], r[8], h[8];
  if (nc == 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = cvs[0][i];
  } else if (nc == 2) {
    b3_parent(key, cvs[0], cvs[1], B3_ROOT, h);
  } else if (nc == 3) {
    b3_parent(key, cvs[0], cvs[1], 0, l);
    b3_parent(key, l, cvs[2], B3_ROOT, h);
  } else {
    b3_parent(key, cvs[0], cvs[1], 0, l);
    b3_parent(key, cvs[2], cvs[3], 0, r);
    b3_parent(key, l, r, B3_ROOT, h);
  }
  store_cv(a.p[1] + (size_t)t * 32, h);
}

// A's rows as the verifier commits them: each row's frame-b3 leaf (schema verity/pouw/row/v1, rank = position = the row),
// one CTA per row: its chunks in parallel, then BLAKE3's left-heavy tree, whose perfect blocks reduce level by level and
// whose block roots fold right to left.  p0 rows, p1 out (32 B per row: level 0 of the tree), p2 domain id; i0 rows, i1 row bytes
// word w of the leaf's message: the header's words, then the row's (32-bit loads when rows are whole words)
DEV uint32_t row_msg_word(const uint8_t* src, uint32_t rb, uint64_t row, uint64_t w) {
  if (w < ROW_PREFIX / 4) {
    uint32_t x = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) x |= header_byte((int)(4 * w) + j, row, ROW_SCHEMA, 18) << (8 * j);
    return x;
  }
  const uint64_t off = 4 * w - ROW_PREFIX;
  if ((rb & 3) == 0) return off < rb ? reinterpret_cast<const uint32_t*>(src)[off / 4] : 0u;
  uint32_t x = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) x |= (off + j < rb ? (uint32_t)src[off + j] : 0u) << (8 * j);
  return x;
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
      const uint64_t base = (1024ull * c + 64ull * b) / 4;
#pragma unroll
      for (int w = 0; w < 16; w++) m[w] = row_msg_word(src, rb, row, base + w);
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

// a frame-b3 tree's level keys, keyed(domain id, 0x02 ‖ u8 l) for l < i0, computable once the domain id is known (before the
// leaves).  p0 out (32 B per level), p1 domain id; i0 levels
extern "C" __global__ void __launch_bounds__(32) hash_node_keys(const Args a) {
  const int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= a.i[0]) return;
  uint32_t key[8], h[8];
  load_key(a.p[1], key);
  b3_short<2>(key, [&](int j) { return j == 0 ? 2u : (uint32_t)l; }, h);
  store_cv(a.p[0] + (uint64_t)l * 32, h);
}

// one level of a frame-b3 tree: node i of level i0 + 1 from nodes 2i and 2i + 1 of level i0, one compression under level i0's
// key.  p0 the level below, p1 the level above, p2 level i0's key; i0 the level below's index, i1 nodes above
extern "C" __global__ void __launch_bounds__(128) hash_nodes(const Args a) {
  const uint64_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (uint64_t)a.i[1]) return;
  uint32_t key[8], m[16], h[8];
  load_key(a.p[2], key);
  load_key(a.p[0] + 64 * i, m);
  load_key(a.p[0] + 64 * i + 32, m + 8);
  blake3_block(key, m, h);
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
  b3_short<96>(key, [&](int j) { return j < 64 ? header_byte(j, t, TILE_SCHEMA, 19) : cv_byte(d, j - 64); }, h);
  store_cv(a.p[1] + t * 32, h);
}

// Seeds: -h3's row seeds and a unit's seed_A, each the standard keyed BLAKE3 under the seed-A label's key (p2,
// blake3("pearl-c/v0/seed-A")) of salt ‖ root ‖ root_B ‖ u64 LE ι (104 bytes: two blocks).  i2 = 1: row r of i0 rows, root its
// leaf (p0 + 32 r) and ι = i1 · 2^32 + r (`pearl_c.row_seed`, the one-row unit's seed_A); i2 = 0: the unit's, root p0 and
// ι = i1 (`pearl_c.unit_seed_a`).  p0 leaves or the root, p1 out (32 B a seed), p2 the key, p3 salt, p4 root_B, p5 0 or the
// unit index's address (its low 32 bits, little-endian, in place of i1: a captured graph's per-call key slot); i0 rows,
// i1 the unit index, i2 the mode
extern "C" __global__ void __launch_bounds__(128) b3_seeds(const Args a) {
  const uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= (uint64_t)a.i[0]) return;
  uint32_t key[8], m[16], cv[8];
  load_key(a.p[2], key);
  load_key(a.p[3], m);
  load_key(a.p[0] + 32 * r, m + 8);
  b3c(key, m, 0, 64, B3_KEYED | B3_START, cv);
  const uint32_t unit = a.p[5] ? *reinterpret_cast<const uint32_t*>(a.p[5]) : (uint32_t)a.i[1];
  const uint64_t iota = a.i[2] ? ((uint64_t)unit << 32 | r) : (uint64_t)unit;
  uint32_t m2[16] = {0};
  load_key(a.p[4], m2);
  m2[8] = (uint32_t)iota;
  m2[9] = (uint32_t)(iota >> 32);
  b3c(cv, m2, 0, 40, B3_KEYED | B3_END | B3_ROOT, cv);
  store_cv(a.p[1] + 32 * r, cv);
}

// A seed's two subkeys (`pearl_kw.subkey`): BLAKE3 keyed by the seed over "pearl/v4/FP8/noise-line", the key E's lines are
// drawn from, and over "pearl/v4/FP8/jackpot", the tickets' key.  p0 the seed, p1 out (the line key, then the jackpot key)
__constant__ uint8_t NOISE_LINE_LABEL[23] = {'p', 'e', 'a', 'r', 'l', '/', 'v', '4', '/', 'F', 'P', '8', '/', 'n', 'o', 'i', 's', 'e',
                                             '-', 'l', 'i', 'n', 'e'};
__constant__ uint8_t JACKPOT_LABEL[20] = {'p', 'e', 'a', 'r', 'l', '/', 'v', '4', '/', 'F', 'P', '8', '/', 'j', 'a', 'c', 'k', 'p',
                                          'o', 't'};
extern "C" __global__ void __launch_bounds__(32) seed_subkeys(const Args a) {
  const int t = threadIdx.x;
  if (blockIdx.x || t > 1) return;
  uint32_t key[8], h[8];
  load_key(a.p[0], key);
  if (t == 0)
    b3_short<23>(key, [&](int j) { return (uint32_t)NOISE_LINE_LABEL[j]; }, h);
  else
    b3_short<20>(key, [&](int j) { return (uint32_t)JACKPOT_LABEL[j]; }, h);
  store_cv(a.p[1] + 32 * t, h);
}

// Pearl's tickets: per 16 x 16 subtile of C~, the 16-lane fold (lane (rb, cb) folds its 4 x 4 block row-major,
// lane <- rotl32(lane 0x9E3779B1 + bits, 13)) hashed as one keyed BLAKE3 block.  p0 C~, p1 out (subtiles row-major x 32 B),
// p2 key (8 words); i0 rows, i1 cols
DEV void ticket_fold(const uint32_t* C, int cols, int si, int sj, uint32_t* z) {
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
}
extern "C" __global__ void __launch_bounds__(128) tickets(const Args a) {
  const int cols = a.i[1], sn = cols / 16;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] / 16 * sn) return;
  const int si = t / sn, sj = t % sn;
  uint32_t z[16], kw[8], h[8];
  ticket_fold(reinterpret_cast<const uint32_t*>(a.p[0]), cols, si, sj, z);
#pragma unroll
  for (int i = 0; i < 8; i++) kw[i] = reinterpret_cast<const uint32_t*>(a.p[2])[i];
  blake3_block(kw, z, h);
  uint4* out = reinterpret_cast<uint4*>(a.p[1] + (size_t)t * 32);
  out[0] = make_uint4(h[0], h[1], h[2], h[3]);
  out[1] = make_uint4(h[4], h[5], h[6], h[7]);
}

