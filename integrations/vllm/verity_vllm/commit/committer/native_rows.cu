// The rows-sha512 scheme's kernels (verity_vllm.commit.step_rows): COMMIT-GPU's leaf, pad and node (benchmarks/pouw/commit_gpu/
// commit_gpu.cuh: frame_leaf_nb, pad_nb, node_ni, hash_after_block, salt_prf, salt_given, salt_digest, hm96_mask's nibble table and
// leaf_from_digest), bit for bit core's hm96-sha512/row/v2 leaves in frame-v3-sha512 trees.  Two changes from COMMIT-GPU, neither in
// what is hashed:
//   * every per-step constant (the domain's frame templates and midstates, the salt key) is a kernel parameter (`LeafK`, `NodeK`,
//     __grid_constant__) instead of __constant__ memory, so the worker's steps (one domain each) run back to back on its side stream
//     with no symbol copy between them;
//   * digests are stored in byte order (a [n, 64] uint8 tensor is the digests), each 8-byte word swapped on load and store.
// Kernel names contain rows_leaf (leaves and pads) or rows_level (nodes): benchmarks/served_commit/nsys_summary.py files GPU time by
// those substrings, and any other name counts as the model's.
// extern "C" launchers and the host builder for native_collect.cpp's StepHasher, which runs them with the GIL released.
//
// Per position pos of a domain of n leaves (rank = position, the tree padded to width = 2^depth):
//   x    = SHA-512(P(l) || row)                                   row = l / 8 bytes at rows + (pos - first) * stride
//   y    = salt (192 bytes): given, or y_r = SHA-512(Ks || u64be(pos) || u64be(r)), r = 0, 1, 2
//   c    = SHA-512(salt_prefix || y);  b = x ^ M_key y;  h = SHA-512(leaf_prefix(key) || b || c)
//   leaf = SHA-512(frame || "leaf" || domain id || uint(pos) || uint(pos) || "hm96-sha512/row/v2" || h)   (framed parts)
//   pad  = SHA-512(frame || "pad" || domain id || uint(pos)) for n <= pos < width
// then node(level, index, left, right) up the tree, up to ten levels a launch, each block folding its own aligned slice (rows_levels_kernel).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sha512.cuh"

namespace {

using u64 = uint64_t;
using u32 = uint32_t;
#define DEV __device__ __forceinline__

constexpr int MAX_NB = 4;            // positions and node indices below 2^32
constexpr int LEVELS = 33;
constexpr int THREADS = 256;
// The leaf kernel's __launch_bounds__ min blocks per SM: 2 caps it at 128 registers, the fastest form on H200 at 256-byte rows
// (benchmarks/served_commit/rows_floor.py times 1, 2 and 3).
#ifndef VERITY_ROWS_MIN_BLOCKS
#define VERITY_ROWS_MIN_BLOCKS 2
#endif

struct LeafK {
  u64 mid_P[8];                      // after P(l)
  u64 mid_salt[8];                   // after the salt prefix
  u64 mid_leaf[8];                   // after leaf_prefix(key)
  u64 ks[8];                         // the salt key
  u64 ks_m8[8];                      // state after rounds 0..7 of a salt block (its words 0..7 are the key)
  u64 fl_m15[8];                     // state after rounds 0..14 of the frame leaf's first block (words 0..14 don't depend on pos)
  u64 kw_leafpad[80];                // the hm96 leaf's padding block, scheduled
  u64 fl_tpl[MAX_NB][32];            // the frame leaf's two blocks per nb = len(uint(pos)) (index nb - 1), pos and h zero
  u64 pad_tpl[MAX_NB][16];           // a pad's first block per nb, rank zero
  u32 lb;                            // row bytes
};
struct NodeK {
  u64 node_mid[LEVELS][8];           // per level: the chaining value after a node message's first block (it fixes only the level)
  u64 node_tpl[MAX_NB][32];          // a node message's blocks 2 and 3 per ni = len(uint(index)), index and children zero
};
struct Params {
  LeafK leaf;
  NodeK node;
};

DEV u32 bswap32(u32 x) { return __byte_perm(x, 0, 0x0123); }
DEV u64 be64(u32 lo, u32 hi) { return (u64)bswap32(lo) << 32 | bswap32(hi); }   // 8 bytes loaded as two LE u32
DEV int nbytes(u64 x) { int n = 1; while (x >> (8 * n) && n < 8) ++n; return n; }
DEV uint4 ldg16(const uint8_t* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

DEV void cpy8(u64* d, const u64* s) {
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = s[i];
}
// n words from / to digest byte order
DEV void ld_be(const uint8_t* p, u64* d, int n) {
#pragma unroll
  for (int i = 0; i < n / 2; ++i) {
    const uint4 v = ldg16(p + 16 * i);
    d[2 * i] = be64(v.x, v.y);
    d[2 * i + 1] = be64(v.z, v.w);
  }
}
DEV void st_be(uint8_t* p, const u64* d, int n) {
  uint4* o = reinterpret_cast<uint4*>(p);
#pragma unroll
  for (int i = 0; i < n / 2; ++i)
    o[i] = make_uint4(bswap32((u32)(d[2 * i] >> 32)), bswap32((u32)d[2 * i]), bswap32((u32)(d[2 * i + 1] >> 32)), bswap32((u32)d[2 * i + 1]));
}

// OR the low NB bytes of v, big-endian, into the message words W at byte offset OFF.
template <int OFF, int NB>
DEV void or_be(u64* W, u64 v) {
  constexpr int END = OFF + NB, K = (END - 1) / 8, SH = 8 * (7 - (END - 1) % 8);
  W[K] |= v << SH;
  if constexpr (OFF < 8 * K) W[K - 1] |= v >> (64 - SH);
}
// OR a digest (8 words) into W at byte offset OFF.
template <int OFF>
DEV void or_digest(u64* W, const u64* d) {
  constexpr int K0 = OFF / 8, S = 8 * (OFF % 8);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if constexpr (S == 0) {
      W[K0 + i] |= d[i];
    } else {
      W[K0 + i] |= d[i] >> S;
      W[K0 + i + 1] |= d[i] << (64 - S);
    }
  }
}

// --- the frame: leaf, pad, node ----------------------------------------------------------------------------------

template <int NB>
DEV void frame_leaf_nb(const LeafK& K, u64 pos, const u64* h, u64* out) {
  u64 W[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) W[i] = K.fl_tpl[NB - 1][i];
  or_be<121, NB>(W, pos);
  or_be<129 + NB, NB>(W, pos);
  or_digest<163 + 2 * NB>(W, h);
  sha::iv(out);
  sha::compress_from<15>(out, K.fl_m15, W);
  sha::compress(out, W + 16);
}
DEV void frame_leaf(const LeafK& K, u64 pos, const u64* h, u64* out) {
  switch (nbytes(pos)) {
    case 1: frame_leaf_nb<1>(K, pos, h, out); break;
    case 2: frame_leaf_nb<2>(K, pos, h, out); break;
    case 3: frame_leaf_nb<3>(K, pos, h, out); break;
    default: frame_leaf_nb<4>(K, pos, h, out); break;
  }
}

template <int NB>
DEV void pad_nb(const LeafK& K, u64 rank, u64* out) {
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = K.pad_tpl[NB - 1][i];
  or_be<120, NB>(W, rank);
  sha::iv(out);
  sha::compress(out, W);
#pragma unroll
  for (int i = 0; i < 15; ++i) W[i] = 0;
  W[15] = (120 + NB) * 8;
  sha::compress(out, W);
}
DEV void pad(const LeafK& K, u64 rank, u64* out) {
  switch (nbytes(rank)) {
    case 1: pad_nb<1>(K, rank, out); break;
    case 2: pad_nb<2>(K, rank, out); break;
    case 3: pad_nb<3>(K, rank, out); break;
    default: pad_nb<4>(K, rank, out); break;
  }
}

template <int NI>
DEV void node_ni(const NodeK& N, int level, u64 index, const u64* L, const u64* R, u64* out) {
  u64 W[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) W[i] = N.node_tpl[NI - 1][i];
  or_be<2, NI>(W, index);
  or_digest<10 + NI>(W, L);
  or_digest<82 + NI>(W, R);
  cpy8(out, N.node_mid[level]);
  sha::compress(out, W);
  sha::compress(out, W + 16);
}
DEV void node(const NodeK& N, int level, u64 index, const u64* L, const u64* R, u64* out) {
  switch (nbytes(index)) {
    case 1: node_ni<1>(N, level, index, L, R, out); break;
    case 2: node_ni<2>(N, level, index, L, R, out); break;
    case 3: node_ni<3>(N, level, index, L, R, out); break;
    default: node_ni<4>(N, level, index, L, R, out); break;
  }
}

// --- the leaf's parts ----------------------------------------------------------------------------------------------

// SHA-512 of a 128-byte first block (its chaining value cv) and lb row bytes at p (16-byte aligned; the loads may run up to 15
// bytes past the row, inside its stride).
DEV u32 row_blocks(u32 lb) { return (lb + 17 + 127) / 128; }

// hash_after_block's compressions b0 .. b1 - 1 (b1 <= row_blocks(lb)) on the chaining value x.
DEV void hash_row_blocks(u64* x, const uint8_t* p, u32 lb, u32 b0, u32 b1) {
  const u32 nblk = row_blocks(lb);
  for (u32 b = b0; b < b1; ++b) {
    u64 W[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const u32 off = 128 * b + 16 * c;
      const uint4 v = off < lb ? ldg16(p + off) : make_uint4(0, 0, 0, 0);
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

DEV void hash_after_block(const u64* cv, const uint8_t* p, u32 lb, u64* x) {
  cpy8(x, cv);
  hash_row_blocks(x, p, lb, 0, row_blocks(lb));
}

DEV void salt_prf(const LeafK& K, u64 pos, u64* Y) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    u64 w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = K.ks[i];
    w[8] = pos; w[9] = r; w[10] = 0x8000000000000000ull;
    w[11] = w[12] = w[13] = w[14] = 0;
    w[15] = 80 * 8;
    sha::iv(Y + 8 * r);
    sha::compress_from<8>(Y + 8 * r, K.ks_m8, w);
  }
}

// c = SHA-512(salt_prefix || y)
DEV void salt_digest(const LeafK& K, const u64* Y, u64* c) {
  u64 w[16];
  cpy8(c, K.mid_salt);
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

// M_key y as 16 little-endian u32 (mask byte 4w + k is byte k of m[w]): for salt bytes p = 16 q + r, the 80 bytes [16 q, 16 q + 80)
// of T_{y_p} are xored into A_r, and A_r shifted down r bytes into m.  T is the nibble table (step_rows.t_nibbles): T_v =
// T_(v & 15) ^ T_(v & 0xF0), rows stored chunk-major (chunk c of row u at 256 c + 16 u, the high nibble's rows 4 KB on), 8 KB that
// stays in L1.
DEV void hm96_mask(const uint8_t* T, const u64* Y, u32* m) {
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
      const uint8_t* lo = T + 256 * q + 16 * (v & 15u);
      const uint8_t* hi = T + 4096 + 256 * q + 16 * (v >> 4);
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const uint4 u = ldg16(lo + 256 * i), w = ldg16(hi + 256 * i);
        A[4 * i] ^= u.x ^ w.x; A[4 * i + 1] ^= u.y ^ w.y; A[4 * i + 2] ^= u.z ^ w.z; A[4 * i + 3] ^= u.w ^ w.w;
      }
    }
    const int sw = r >> 2, s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= s ? __funnelshift_r(A[w + sw], A[w + sw + 1], s) : A[w + sw];
  }
}

// A thread a position first + i (i < m): the tree leaf when i < n_rows and pos < n_total (row i of `rows`), else the pad; b || c to
// bc[i] and h to hs[i] for a leaf when they are given.  salts: row i's given salt at 192 i, or null for derived salts.
__global__ void __launch_bounds__(THREADS, VERITY_ROWS_MIN_BLOCKS)
    rows_leaf_kernel(const __grid_constant__ LeafK K, const uint8_t* __restrict__ rows, u64 stride, u64 n_rows,
                     const uint8_t* __restrict__ salts, const uint8_t* __restrict__ Tn, u64 first, u64 m, u64 n_total,
                     uint8_t* __restrict__ out, uint8_t* __restrict__ bc, uint8_t* __restrict__ hs) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = first + i;
  u64 o[8];
  if (i < n_rows && pos < n_total) {
    u64 x[8], Y[24], c[8], B[16], h[8];
    u32 mk[16];
    hash_after_block(K.mid_P, rows + i * stride, K.lb, x);
    if (salts) ld_be(salts + 192 * i, Y, 24);
    else salt_prf(K, pos, Y);
    hm96_mask(Tn, Y, mk);
    salt_digest(K, Y, c);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      B[w] = x[w] ^ ((u64)bswap32(mk[2 * w]) << 32 | bswap32(mk[2 * w + 1]));
      B[8 + w] = c[w];
    }
    if (bc) st_be(bc + 128 * i, B, 16);
    cpy8(h, K.mid_leaf);
    sha::compress(h, B);
    sha::compress_kw(h, K.kw_leafpad);
    if (hs) st_be(hs + 64 * i, h, 8);
    frame_leaf(K, pos, h, o);
  } else {
    pad(K, pos, o);
  }
  st_be(out + 64 * i, o, 8);
}

// --- the per-GPU leaf forms (sharaw) -------------------------------------------------------------------------------------------
// rows_leaf_kernel inlines some 25 compression bodies and an unrolled mask, so its warps in different phases fetch different code:
// on 4 KiB rows it reaches 69-91% of the raw compression rate where the row hash alone reaches 100%.  Two forms with less code a
// kernel, the same bytes:
//   fused (sm_120): the same leaf a thread, the salt first, the mask's nibble table in shared memory and its loop over r rolled;
//   split (H200, B200): rows_leaf_salt_kernel (y, M_key y from a byte table in shared memory, c and the frame leaf's first block,
//         to 192 bytes of scratch a position; pads straight to out) and then rows_leaf_data_kernel (x, h and the frame leaf's second
//         block).

constexpr int T8_BYTES = 256 * 17 * 16;

// The nibble table into shared memory: uint4 16 c + u is chunk c of T_u, 256 + 16 c + u chunk c of T_16u.
DEV void load_tn(const uint8_t* Tn, uint4* T) {
  const uint4* s = reinterpret_cast<const uint4*>(Tn);
  for (int k = threadIdx.x; k < 512; k += blockDim.x) T[k] = __ldg(s + k);
  __syncthreads();
}
// The byte table: T8[17 b + c] is chunk c of T_b = T_(b & 15) ^ T_(16 (b >> 4)); the stride of 17 spreads the lanes' bytes over
// the banks.
DEV void load_t8(const uint8_t* Tn, uint4* T8) {
  const uint4* s = reinterpret_cast<const uint4*>(Tn);
  for (int k = threadIdx.x; k < 256 * 16; k += blockDim.x) {
    const int b = k >> 4, c = k & 15;
    const uint4 lo = __ldg(s + 16 * c + (b & 15)), hi = __ldg(s + 256 + 16 * c + (b >> 4));
    T8[17 * b + c] = make_uint4(lo.x ^ hi.x, lo.y ^ hi.y, lo.z ^ hi.z, lo.w ^ hi.w);
  }
  __syncthreads();
}

// hm96_mask's product with its loop over r rolled: A_r's shift down r bytes is a word shift of r >> 2 (two conditional static
// shifts) and a funnel shift of 8 (r & 3).  BYTE: T is the byte table (5 loads a salt byte), else the nibble table (10).
template <bool BYTE>
DEV void mask_roll(const uint4* T, const u64* Y, u32* m) {
#pragma unroll
  for (int w = 0; w < 16; ++w) m[w] = 0;
#pragma unroll 1
  for (int r = 0; r < 16; ++r) {
    u32 A[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) A[i] = 0;
    const int sh = 56 - 8 * (r & 7);
#pragma unroll
    for (int q = 0; q < 12; ++q) {
      const u32 v = (u32)((r < 8 ? Y[2 * q] : Y[2 * q + 1]) >> sh) & 0xFFu;
      if constexpr (BYTE) {
        const uint4* p = T + 17 * v + q;
#pragma unroll
        for (int i = 0; i < 5; ++i) {
          const uint4 u = p[i];
          A[4 * i] ^= u.x; A[4 * i + 1] ^= u.y; A[4 * i + 2] ^= u.z; A[4 * i + 3] ^= u.w;
        }
      } else {
        const uint4* lo = T + 16 * q + (v & 15u);
        const uint4* hi = T + 256 + 16 * q + (v >> 4);
#pragma unroll
        for (int i = 0; i < 5; ++i) {
          const uint4 u = lo[16 * i], w = hi[16 * i];
          A[4 * i] ^= u.x ^ w.x; A[4 * i + 1] ^= u.y ^ w.y; A[4 * i + 2] ^= u.z ^ w.z; A[4 * i + 3] ^= u.w ^ w.w;
        }
      }
    }
    if (r & 4) {
#pragma unroll
      for (int k = 0; k < 19; ++k) A[k] = A[k + 1];
    }
    if (r & 8) {
#pragma unroll
      for (int k = 0; k < 18; ++k) A[k] = A[k + 2];
    }
    const u32 s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= __funnelshift_r(A[w], A[w + 1], s);
  }
}

// salt_prf with its loop over r rolled (one compression body for the three)
DEV void salt_prf_rolled(const LeafK& K, u64 pos, u64* Y) {
#pragma unroll 1
  for (int r = 0; r < 3; ++r) {
    u64 w[16], s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = K.ks[i];
    w[8] = pos; w[9] = r; w[10] = 0x8000000000000000ull;
    w[11] = w[12] = w[13] = w[14] = 0;
    w[15] = 80 * 8;
    sha::iv(s);
    sha::compress_from<8>(s, K.ks_m8, w);
#pragma unroll
    for (int i = 0; i < 8; ++i) Y[8 * r + i] = s[i];
  }
}

DEV u64 mask_word(const u32* mk, int w) { return (u64)bswap32(mk[2 * w]) << 32 | bswap32(mk[2 * w + 1]); }

// The frame leaf in two halves: its first block (only pos varies) and then its second (pos and h).
template <int NB>
DEV void frame_first_nb(const LeafK& K, u64 pos, u64* f) {
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = K.fl_tpl[NB - 1][i];
  or_be<121, NB>(W, pos);
  sha::iv(f);
  sha::compress_from<15>(f, K.fl_m15, W);
}
template <int NB>
DEV void frame_second_nb(const LeafK& K, u64 pos, const u64* h, u64* f) {
  u64 W[32];                                     // words 16..31: or_be and or_digest below touch none under 16
#pragma unroll
  for (int i = 16; i < 32; ++i) W[i] = K.fl_tpl[NB - 1][i];
  or_be<129 + NB, NB>(W, pos);
  or_digest<163 + 2 * NB>(W, h);
  sha::compress(f, W + 16);
}
DEV void frame_first(const LeafK& K, u64 pos, u64* f) {
  switch (nbytes(pos)) {
    case 1: frame_first_nb<1>(K, pos, f); break;
    case 2: frame_first_nb<2>(K, pos, f); break;
    case 3: frame_first_nb<3>(K, pos, f); break;
    default: frame_first_nb<4>(K, pos, f); break;
  }
}
DEV void frame_second(const LeafK& K, u64 pos, const u64* h, u64* f) {
  switch (nbytes(pos)) {
    case 1: frame_second_nb<1>(K, pos, h, f); break;
    case 2: frame_second_nb<2>(K, pos, h, f); break;
    case 3: frame_second_nb<3>(K, pos, h, f); break;
    default: frame_second_nb<4>(K, pos, h, f); break;
  }
}

DEV void st_raw(u64* p, const u64* d, int n) {
#pragma unroll
  for (int i = 0; i < n / 2; ++i) reinterpret_cast<ulonglong2*>(p)[i] = make_ulonglong2(d[2 * i], d[2 * i + 1]);
}
DEV void ld_raw(const u64* p, u64* d, int n) {
#pragma unroll
  for (int i = 0; i < n / 2; ++i) {
    const ulonglong2 v = __ldg(reinterpret_cast<const ulonglong2*>(p) + i);
    d[2 * i] = v.x;
    d[2 * i + 1] = v.y;
  }
}

// The fused form: rows_leaf_kernel's positions and outputs, the salt first from the nibble table in shared memory.  H_ONLY: h alone,
// to hs, for the data positions (no frame leaf, no pad, out unused).
template <bool H_ONLY>
__global__ void __launch_bounds__(THREADS, VERITY_ROWS_MIN_BLOCKS)
    rows_leaf_sf_kernel(const __grid_constant__ LeafK K, const uint8_t* __restrict__ rows, u64 stride, u64 n_rows,
                        const uint8_t* __restrict__ salts, const uint8_t* __restrict__ Tn, u64 first, u64 m, u64 n_total,
                        uint8_t* __restrict__ out, uint8_t* __restrict__ bc, uint8_t* __restrict__ hs) {
  __shared__ uint4 T[512];
  load_tn(Tn, T);
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = first + i;
  u64 o[8];
  if (i < n_rows && pos < n_total) {
    u64 x[8], Y[24], c[8], B[16], h[8];
    u32 mk[16];
    if (salts) ld_be(salts + 192 * i, Y, 24);
    else salt_prf_rolled(K, pos, Y);
    mask_roll<false>(T, Y, mk);
    salt_digest(K, Y, c);
    hash_after_block(K.mid_P, rows + i * stride, K.lb, x);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      B[w] = x[w] ^ mask_word(mk, w);
      B[8 + w] = c[w];
    }
    if (bc) st_be(bc + 128 * i, B, 16);
    cpy8(h, K.mid_leaf);
    sha::compress(h, B);
    sha::compress_kw(h, K.kw_leafpad);
    if (hs) st_be(hs + 64 * i, h, 8);
    if constexpr (H_ONLY) return;
    frame_leaf(K, pos, h, o);
  } else {
    if constexpr (H_ONLY) return;
    pad(K, pos, o);
  }
  st_be(out + 64 * i, o, 8);
}

// The split form's first kernel (T8_BYTES of dynamic shared memory): a data position i < nd gets M_key y || c || the frame leaf's
// first-block state at sf + 192 i, in word order; any other the pad, to out.  H_ONLY: M_key y || c only, for the nd data positions.
template <bool H_ONLY>
__global__ void __launch_bounds__(THREADS, 2)
    rows_leaf_salt_kernel(const __grid_constant__ LeafK K, const uint8_t* __restrict__ salts, const uint8_t* __restrict__ Tn, u64 first, u64 m,
                     u64 nd, uint8_t* __restrict__ out, uint8_t* __restrict__ sf) {
  extern __shared__ uint4 T8[];
  load_t8(Tn, T8);
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = first + i;
  if (i < nd) {
    u64 Y[24], c[8], my[8], f[8];
    u32 mk[16];
    if (salts) ld_be(salts + 192 * i, Y, 24);
    else salt_prf(K, pos, Y);
    mask_roll<true>(T8, Y, mk);
    salt_digest(K, Y, c);
#pragma unroll
    for (int w = 0; w < 8; ++w) my[w] = mask_word(mk, w);
    u64* s = reinterpret_cast<u64*>(sf + 192 * i);
    st_raw(s, my, 8);
    st_raw(s + 8, c, 8);
    if constexpr (H_ONLY) return;
    frame_first(K, pos, f);
    st_raw(s + 16, f, 8);
  } else if constexpr (!H_ONLY) {
    u64 o[8];
    pad(K, pos, o);
    st_be(out + 64 * i, o, 8);
  }
}

// The split form's second kernel: the nd data positions' leaves from their rows and rows_leaf_salt_kernel's scratch.  H_ONLY: h alone, to hs.
// One launch runs the row hash's compressions b0 .. b1 - 1: a launch that ends before the row's last keeps the chaining value in the
// position's own 64 bytes of its output (hs, else out), which the launch that finishes the row overwrites; so each block lasts a
// fraction of a row and frees its SM sooner for a kernel that needs every SM (the serving stream's cooperative GEMMs).
template <bool H_ONLY>
__global__ void __launch_bounds__(THREADS, VERITY_ROWS_MIN_BLOCKS)
    rows_leaf_data_kernel(const __grid_constant__ LeafK K, const uint8_t* __restrict__ rows, u64 stride, const uint8_t* __restrict__ sf, u64 first,
                     u64 nd, uint8_t* __restrict__ out, uint8_t* __restrict__ bc, uint8_t* __restrict__ hs, u32 b0, u32 b1) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nd) return;
  u64 x[8], B[16], h[8], f[8];
  u64* xs = reinterpret_cast<u64*>((H_ONLY ? hs : out) + 64 * i);
  const u32 nblk = row_blocks(K.lb), end = b1 < nblk ? b1 : nblk;
  if (b0 == 0) cpy8(x, K.mid_P);
  else ld_raw(xs, x, 8);
  hash_row_blocks(x, rows + i * stride, K.lb, b0, end);
  if (end < nblk) {
    st_raw(xs, x, 8);
    return;
  }
  const u64* s = reinterpret_cast<const u64*>(sf + 192 * i);
  ld_raw(s, B, 16);
#pragma unroll
  for (int w = 0; w < 8; ++w) B[w] ^= x[w];
  if (bc) st_be(bc + 128 * i, B, 16);
  cpy8(h, K.mid_leaf);
  sha::compress(h, B);
  sha::compress_kw(h, K.kw_leafpad);
  if (hs) st_be(hs + 64 * i, h, 8);
  if constexpr (H_ONLY) return;
  ld_raw(s + 16, f, 8);
  frame_second(K, first + i, h, f);
  st_be(out + 64 * i, f, 8);
}

// The leaves of positions first .. first + m - 1 from their h (hs, one 64-byte digest a data position i < nd), pads after.
__global__ void __launch_bounds__(THREADS) rows_leaf_frame_kernel(const __grid_constant__ LeafK K, const uint8_t* __restrict__ hs, u64 first, u64 m,
                                                             u64 nd, uint8_t* __restrict__ out) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  u64 o[8];
  if (i < nd) {
    u64 h[8];
    ld_be(hs + 64 * i, h, 8);
    frame_leaf(K, first + i, h, o);
  } else {
    pad(K, first + i, o);
  }
  st_be(out + 64 * i, o, 8);
}

// Level `level` + 1 from level `level`, a thread a parent: out[i] = node(level, first_index + i, in[2i], in[2i+1]).
__global__ void __launch_bounds__(THREADS) rows_level_kernel(const __grid_constant__ NodeK N, const uint8_t* __restrict__ in,
                                                             uint8_t* __restrict__ out, u64 n_out, int level, u64 first_index) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  u64 L[8], R[8], o[8];
  ld_be(in + 128 * i, L, 8);
  ld_be(in + 128 * i + 64, R, 8);
  node(N, level, first_index + i, L, R, o);
  st_be(out + 64 * i, o, 8);
}

// ld_be with ordinary loads: a level this block wrote before a barrier is not read through the non-coherent path
DEV void ld_be_written(const uint8_t* p, u64* d, int n) {
#pragma unroll
  for (int i = 0; i < n / 2; ++i) {
    const uint4 v = reinterpret_cast<const uint4*>(p)[i];
    d[2 * i] = be64(v.x, v.y);
    d[2 * i + 1] = be64(v.z, v.w);
  }
}

// n_levels levels above n_in nodes of level `level` (the first at index base), each level's nodes right after the previous level's in
// `out`; block b folds its own slice, nodes [b 2^n_levels, (b+1) 2^n_levels), so no pair crosses blocks.  The same node as
// rows_level_kernel.
__global__ void __launch_bounds__(THREADS) rows_levels_kernel(const __grid_constant__ NodeK N, const uint8_t* in, uint8_t* out, u64 n_in,
                                                              int level, u64 base, int n_levels) {
  const u64 first = (u64)blockIdx.x << n_levels;
  in += 64 * first;
  for (int k = 0; k < n_levels; ++k) {
    const u64 n_out = n_in / 2, len_out = 1ull << (n_levels - k - 1), at = first >> (k + 1);
    uint8_t* dst = out + 64 * at;
    for (u64 i = threadIdx.x; i < len_out; i += blockDim.x) {
      u64 L[8], R[8], o[8];
      ld_be_written(in + 128 * i, L, 8);
      ld_be_written(in + 128 * i + 64, R, 8);
      node(N, level + k, (base >> (k + 1)) + at + i, L, R, o);
      st_be(dst + 64 * i, o, 8);
    }
    __syncthreads();
    in = dst;
    out += 64 * n_out;
    n_in = n_out;
  }
}

// --- host: the step's parameters ---------------------------------------------------------------------------------

const char FRAME[] = "veritor/protocol/merkle/frame/v3";  // + its NUL: 33 bytes
constexpr int SCHEMA_LEN = 18;                             // "hm96-sha512/row/v2": the leaf template's offsets assume its length

// merkle._message: frame || u32be(len tag) || tag || (u64be(len part) || part)...
std::vector<uint8_t> message(const char* tag, const std::vector<std::vector<uint8_t>>& parts) {
  std::vector<uint8_t> m(FRAME, FRAME + sizeof FRAME);
  const size_t tl = strlen(tag);
  for (int i = 3; i >= 0; --i) m.push_back((uint8_t)(tl >> (8 * i)));
  m.insert(m.end(), tag, tag + tl);
  for (const auto& p : parts) {
    for (int i = 7; i >= 0; --i) m.push_back((uint8_t)((u64)p.size() >> (8 * i)));
    m.insert(m.end(), p.begin(), p.end());
  }
  return m;
}
// the padded message's words (len + 17 rounded up to blocks), the length in the last block
std::vector<u64> padded_words(const std::vector<uint8_t>& msg) {
  const size_t total = ((msg.size() + 17 + 127) / 128) * 128;
  std::vector<uint8_t> b(total, 0);
  memcpy(b.data(), msg.data(), msg.size());
  b[msg.size()] = 0x80;
  sha::put_be64(b.data() + total - 8, (u64)msg.size() * 8);
  std::vector<u64> w(total / 8);
  for (size_t i = 0; i < w.size(); ++i) w[i] = sha::be64(b.data() + 8 * i);
  return w;
}
std::vector<uint8_t> zeros(int n) { return std::vector<uint8_t>(n, 0); }
void words_of(const uint8_t* block, u64* w, int n) { for (int i = 0; i < n; ++i) w[i] = sha::be64(block + 8 * i); }
void midstate(const uint8_t* block, u64* st) {
  u64 w[16];
  sha::iv(st);
  words_of(block, w, 16);
  sha::compress(st, w);
}

}  // namespace

extern "C" int verity_rows_params_bytes() { return static_cast<int>(sizeof(Params)); }

// The step's parameters into `out` (verity_rows_params_bytes()): domain id (64 bytes), salt key Ks (64), P(l) (128), the salt prefix
// (128), leaf_prefix(key) (128), the leaf schema (SCHEMA_LEN bytes) and the row's bytes l / 8.  0, or -1 for a schema of another length.
extern "C" int verity_rows_build(const uint8_t* domain_id, const uint8_t* ks, const uint8_t* row_prefix, const uint8_t* salt_prefix,
                                 const uint8_t* leaf_prefix, const char* schema, int schema_len, uint32_t lb, void* out) {
  if (schema_len != SCHEMA_LEN) return -1;
  Params p{};
  LeafK& c = p.leaf;
  NodeK& n = p.node;
  c.lb = lb;
  midstate(row_prefix, c.mid_P);
  midstate(salt_prefix, c.mid_salt);
  midstate(leaf_prefix, c.mid_leaf);
  {
    u64 blk[16] = {0x8000000000000000ull};
    blk[15] = 256 * 8;
    sha::schedule_kw(blk, c.kw_leafpad);
  }
  words_of(ks, c.ks, 8);
  {
    u64 blk[16] = {}, st[8];
    memcpy(blk, c.ks, 64);
    sha::iv(st);
    sha::rounds_prefix(st, blk, 8, c.ks_m8);
  }
  const std::vector<uint8_t> did(domain_id, domain_id + 64);
  const std::vector<uint8_t> sch(schema, schema + schema_len);
  for (int nb = 1; nb <= MAX_NB; ++nb) {
    const auto m = padded_words(message("leaf", {did, zeros(nb), zeros(nb), sch, zeros(64)}));
    if (m.size() != 32) return -1;
    std::copy(m.begin(), m.end(), c.fl_tpl[nb - 1]);
    for (int i = 0; i < 15; ++i)
      if (c.fl_tpl[nb - 1][i] != c.fl_tpl[0][i]) return -1;
    const auto pm = padded_words(message("pad", {did, zeros(nb)}));
    if (pm.size() != 32) return -1;
    std::copy(pm.begin(), pm.begin() + 16, c.pad_tpl[nb - 1]);
    const auto nm = padded_words(message("node", {did, std::vector<uint8_t>{0}, zeros(nb), zeros(64), zeros(64)}));
    if (nm.size() != 48) return -1;
    std::copy(nm.begin() + 16, nm.end(), n.node_tpl[nb - 1]);
  }
  {
    u64 st[8];
    sha::iv(st);
    sha::rounds_prefix(st, c.fl_tpl[0], 15, c.fl_m15);
  }
  for (int level = 0; level < LEVELS; ++level) {
    const auto m = padded_words(message("node", {did, std::vector<uint8_t>{(uint8_t)level}, zeros(1), zeros(64), zeros(64)}));
    u64 blk[16];
    std::copy(m.begin(), m.begin() + 16, blk);
    sha::iv(n.node_mid[level]);
    sha::compress(n.node_mid[level], blk);
  }
  memcpy(out, &p, sizeof p);
  return 0;
}

// The m positions first .. first + m - 1 (rows_leaf_kernel) on `st`.  rows and stride 16-byte aligned, stride >= the row's bytes
// rounded up to 16; first + m <= 2^32.  max_blocks > 0: in launches of at most max_blocks blocks one after another, each the same
// kernel over the next positions, so the hashing holds at most that many blocks' SM slots and leaves the rest to other streams.
extern "C" int verity_rows_leaves_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows, const uint8_t* salts,
                                         const uint8_t* tn, uint64_t first, uint64_t m, uint64_t n_total, uint8_t* out, uint8_t* bc,
                                         uint8_t* hs, uint64_t max_blocks, cudaStream_t st) {
  const Params* p = static_cast<const Params*>(params);
  const u64 chunk = max_blocks ? max_blocks * THREADS : m;
  for (u64 c = 0; c < m; c += chunk) {
    const u64 mc = m - c < chunk ? m - c : chunk;
    rows_leaf_kernel<<<static_cast<unsigned>((mc + THREADS - 1) / THREADS), THREADS, 0, st>>>(
        p->leaf, c < n_rows ? rows + c * stride : rows, stride, c < n_rows ? n_rows - c : 0, salts && c < n_rows ? salts + 192 * c : salts, tn,
        first + c, mc, n_total, out + 64 * c, bc ? bc + 128 * c : bc, hs ? hs + 64 * c : hs);
    if (const cudaError_t rc = cudaGetLastError()) return static_cast<int>(rc);
  }
  return 0;
}

// Once per device: the salt kernels' dynamic shared memory, and the largest shared-memory carveout for every kernel the committer
// launches.  They run beside the model's GEMMs, which take most of an SM's shared memory, and an SM holds one L1/shared split at a
// time, so a hash CTA left to the driver's split can keep a GEMM's CTAs off its SM.  rows_leaf_kernel (rows_floor.py's) keeps its own.
static int rows_kernel_attributes() {
  static bool done[64] = {};
  int dev = 0;
  if (const cudaError_t rc = cudaGetDevice(&dev)) return static_cast<int>(rc);
  if (dev < 0 || dev >= 64) return static_cast<int>(cudaErrorInvalidDevice);
  if (done[dev]) return 0;
  for (const void* f : {reinterpret_cast<const void*>(rows_leaf_salt_kernel<false>), reinterpret_cast<const void*>(rows_leaf_salt_kernel<true>)})
    if (const cudaError_t rc = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, T8_BYTES)) return static_cast<int>(rc);
  for (const void* f : {reinterpret_cast<const void*>(rows_leaf_sf_kernel<false>), reinterpret_cast<const void*>(rows_leaf_sf_kernel<true>),
                        reinterpret_cast<const void*>(rows_leaf_salt_kernel<false>), reinterpret_cast<const void*>(rows_leaf_salt_kernel<true>),
                        reinterpret_cast<const void*>(rows_leaf_data_kernel<false>), reinterpret_cast<const void*>(rows_leaf_data_kernel<true>),
                        reinterpret_cast<const void*>(rows_leaf_frame_kernel), reinterpret_cast<const void*>(rows_level_kernel),
                        reinterpret_cast<const void*>(rows_levels_kernel)})
    if (const cudaError_t rc = cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared))
      return static_cast<int>(rc);
  done[dev] = true;
  return 0;
}

// The leaf form for the current device: 1 fused (sm_120), 2 split (others), or VERITY_ROWS_FORM=fused|split; -1 for another value.
extern "C" int verity_rows_leaves_form() {
  if (const char* e = getenv("VERITY_ROWS_FORM"); e && *e) return !strcmp(e, "fused") ? 1 : !strcmp(e, "split") ? 2 : -1;
  int dev = 0, major = 0;
  if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)) return -1;
  return major == 12 ? 1 : 2;
}

// Compressions a launch of the split form's data kernel runs: a row's nblk in VERITY_ROWS_PHASES launches (default 4; 1 is one launch a
// row, as is a call with no output to keep the chaining value in).
static u32 rows_phase_blocks(u32 nblk, const uint8_t* keep) {
  static const u32 phases = [] {
    const char* e = getenv("VERITY_ROWS_PHASES");
    const long v = e && *e ? strtol(e, nullptr, 10) : 4;
    return static_cast<u32>(v < 1 ? 1 : v > 64 ? 64 : v);
  }();
  if (!keep || phases <= 1 || nblk <= 1) return nblk ? nblk : 1;
  return (nblk + phases - 1) / phases;
}

// The leaves (or, H_ONLY, the h of the data positions alone) in the device's form, in launches of at most max_blocks blocks.
template <bool H_ONLY>
static int rows_forms(const Params* p, const uint8_t* rows, u64 stride, u64 n_rows, const uint8_t* salts, const uint8_t* tn, u64 first, u64 m,
                      u64 n_total, uint8_t* out, uint8_t* bc, uint8_t* hs, uint8_t* scratch, u64 max_blocks, cudaStream_t st) {
  const int form = verity_rows_leaves_form();
  if (form < 0 || (form == 2 && !scratch) || (H_ONLY && !hs)) return static_cast<int>(cudaErrorInvalidValue);
  if (const int rc = rows_kernel_attributes()) return rc;
  u64 nd = n_total > first ? n_total - first : 0;
  nd = nd < n_rows ? nd : n_rows;
  nd = nd < m ? nd : m;
  const u64 chunk = max_blocks ? max_blocks * THREADS : m;
  for (u64 c = 0; c < m; c += chunk) {
    const u64 mc = m - c < chunk ? m - c : chunk, dc = c < nd ? (nd - c < mc ? nd - c : mc) : 0;
    const unsigned blocks = static_cast<unsigned>((mc + THREADS - 1) / THREADS);
    const uint8_t* rows_c = c < n_rows ? rows + c * stride : rows;
    const uint8_t* salts_c = salts && c < n_rows ? salts + 192 * c : salts;
    uint8_t* out_c = out ? out + 64 * c : out;
    uint8_t* bc_c = bc ? bc + 128 * c : bc;
    uint8_t* hs_c = hs ? hs + 64 * c : hs;
    if (form == 1) {
      rows_leaf_sf_kernel<H_ONLY><<<blocks, THREADS, 0, st>>>(p->leaf, rows_c, stride, c < n_rows ? n_rows - c : 0, salts_c, tn, first + c, mc,
                                                              n_total, out_c, bc_c, hs_c);
    } else {
      rows_leaf_salt_kernel<H_ONLY><<<blocks, THREADS, T8_BYTES, st>>>(p->leaf, salts_c, tn, first + c, mc, dc, out_c, scratch + 192 * c);
      if (const cudaError_t rc = cudaGetLastError()) return static_cast<int>(rc);
      const u32 nblk = (p->leaf.lb + 17 + 127) / 128, per = rows_phase_blocks(nblk, H_ONLY ? hs_c : out_c);
      for (u32 b0 = 0; dc && b0 < nblk; b0 += per) {
        rows_leaf_data_kernel<H_ONLY><<<static_cast<unsigned>((dc + THREADS - 1) / THREADS), THREADS, 0, st>>>(p->leaf, rows_c, stride, scratch + 192 * c,
                                                                                                           first + c, dc, out_c, bc_c, hs_c, b0, b0 + per);
        if (const cudaError_t rc = cudaGetLastError()) return static_cast<int>(rc);
      }
    }
    if (const cudaError_t rc = cudaGetLastError()) return static_cast<int>(rc);
  }
  return 0;
}

// verity_rows_leaves_launch's leaves in the device's form (verity_rows_leaves_form); the split form needs scratch of 192 m bytes,
// 16-byte aligned (any other may pass null).
extern "C" int verity_rows_leaves_sf_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows, const uint8_t* salts,
                                            const uint8_t* tn, uint64_t first, uint64_t m, uint64_t n_total, uint8_t* out, uint8_t* bc,
                                            uint8_t* hs, uint8_t* scratch, uint64_t max_blocks, cudaStream_t st) {
  return rows_forms<false>(static_cast<const Params*>(params), rows, stride, n_rows, salts, tn, first, m, n_total, out, bc, hs, scratch, max_blocks,
                           st);
}

// The first half of the leaves of n_rows data positions first .. first + n_rows - 1: h (and b || c when bc is given) alone, which reads
// no frame field, so params may be any domain's.  scratch as verity_rows_leaves_sf_launch's, 192 n_rows bytes.
extern "C" int verity_rows_h_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows, const uint8_t* salts,
                                    const uint8_t* tn, uint64_t first, uint8_t* hs, uint8_t* bc, uint8_t* scratch, uint64_t max_blocks,
                                    cudaStream_t st) {
  return rows_forms<true>(static_cast<const Params*>(params), rows, stride, n_rows, salts, tn, first, n_rows, first + n_rows, nullptr, bc, hs,
                          scratch, max_blocks, st);
}

// The second half: the leaves of positions first .. first + m - 1 of a step of n_total leaves from the h of its first n_h positions
// (hs), pads after (rows_leaf_frame_kernel).
extern "C" int verity_rows_frame_launch(const void* params, const uint8_t* hs, uint64_t n_h, uint64_t first, uint64_t m, uint64_t n_total,
                                        uint8_t* out, uint64_t max_blocks, cudaStream_t st) {
  if (const int rc = rows_kernel_attributes()) return rc;
  const Params* p = static_cast<const Params*>(params);
  u64 nd = n_total > first ? n_total - first : 0;
  nd = nd < n_h ? nd : n_h;
  const u64 chunk = max_blocks ? max_blocks * THREADS : m;
  for (u64 c = 0; c < m; c += chunk) {
    const u64 mc = m - c < chunk ? m - c : chunk;
    rows_leaf_frame_kernel<<<static_cast<unsigned>((mc + THREADS - 1) / THREADS), THREADS, 0, st>>>(p->leaf, c < nd ? hs + 64 * c : hs, first + c, mc,
                                                                                               c < nd ? nd - c : 0, out + 64 * c);
    if (const cudaError_t rc = cudaGetLastError()) return static_cast<int>(rc);
  }
  return 0;
}

// n_out parents of level `level` + 1 from 2 n_out children of level `level`, the first parent at index first_index; max_blocks as
// verity_rows_leaves_launch's.
extern "C" int verity_rows_level_launch(const void* params, const uint8_t* in, uint8_t* out, uint64_t n_out, int level, uint64_t first_index,
                                        uint64_t max_blocks, cudaStream_t st) {
  if (!n_out) return 0;
  if (level < 0 || level + 1 >= LEVELS) return static_cast<int>(cudaErrorInvalidValue);
  const Params* p = static_cast<const Params*>(params);
  if (const int rc = rows_kernel_attributes()) return rc;
  const u64 chunk = max_blocks ? max_blocks * THREADS : n_out;
  for (u64 c = 0; c < n_out; c += chunk) {
    const u64 nc = n_out - c < chunk ? n_out - c : chunk;
    rows_level_kernel<<<static_cast<unsigned>((nc + THREADS - 1) / THREADS), THREADS, 0, st>>>(p->node, in + 128 * c, out + 64 * c, nc, level,
                                                                                               first_index + c);
    if (const cudaError_t rc = cudaGetLastError()) return static_cast<int>(rc);
  }
  return 0;
}

// n_levels (at most 10: a block's slice is 2^n_levels nodes) levels above n_in nodes (a nonzero multiple of 2^n_levels) of level
// `level`, the first at index base, in one launch (rows_levels_kernel); out holds all of them, level after level.
extern "C" int verity_rows_levels_launch(const void* params, const uint8_t* in, uint8_t* out, uint64_t n_in, int level, uint64_t base,
                                         int n_levels, cudaStream_t st) {
  if (n_levels <= 0) return 0;
  if (n_levels > 10 || level < 0 || level + n_levels >= LEVELS || !n_in || n_in % (1ull << n_levels))
    return static_cast<int>(cudaErrorInvalidValue);
  const Params* p = static_cast<const Params*>(params);
  if (const int rc = rows_kernel_attributes()) return rc;
  rows_levels_kernel<<<static_cast<unsigned>(n_in >> n_levels), THREADS, 0, st>>>(p->node, in, out, n_in, level, base, n_levels);
  return static_cast<int>(cudaGetLastError());
}
