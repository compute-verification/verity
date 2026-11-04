// Pearl-C's hashing format -h3, SHA-512 throughout (`verity.core.protocols.pouw.schemes.pearl_c`'s module docstring).  The
// labels' H_l = `pearl_c.s512`, SHA-512 after a 128-byte label block, its first 32 bytes (the labels keep their `pearl-c/h3s/`
// bytes, `pearl_c.S512_LABELS`); A's row digests (`s512_rows`) and the tile leaf, `sha512/row-seg/v1` over each row and over
// the tile's messages; the trees `audit.HM96`: frame-v3-sha512 (`verity.ml.commitments.merkle`, hash="sha512") over
// hm96-sha512 leaves (`audit.Hm96Domain`, `verity.ml.commitments.hm96.SHA512`) salted by ChaCha20 (`audit.hm96_salt`).
// Every SHA-512 word is big-endian; buffers hold bytes, loaded as little-endian u32 (`rs_be64`).  A frame message's integers
// have minimal big-endian lengths, so each thread assembles it in its own slice of shared memory (`SMsg`), never in a local
// array.  SHA-512's compression is sha512.cuh's; the frame messages and the hm96 leaf are verity/ml/kernels/row_commit.cuh's
// over it.  pearl_c_sm120.cu includes it; hash_host.cpp builds it for the host tests.
#pragma once
#include <cstdint>

struct Args { uint64_t p[16]; int32_t i[16]; };

#define DEV __device__ __forceinline__

#include "sha512.cuh"
#include "../../../verity/ml/kernels/row_commit.cuh"

// Pearl-C's compression (rs_compress) as the row commitment's policy
struct PcSha {
  static RC_HD void compress(uint64_t* st, uint64_t* w) { sx::compress<0, 0>(st, w); }
  static RC_HD void compress_kw(uint64_t* st, const uint64_t* kw) { sx::compress_kw<0, 0>(st, kw); }
};

DEV void load_key(uint64_t p, uint32_t* k) {   // 32 bytes at device address p as 8 words
#pragma unroll
  for (int i = 0; i < 8; i++) k[i] = reinterpret_cast<const uint32_t*>(p)[i];
}
DEV void store_cv(uint64_t p, const uint32_t* h) {
  reinterpret_cast<uint4*>(p)[0] = make_uint4(h[0], h[1], h[2], h[3]);
  reinterpret_cast<uint4*>(p)[1] = make_uint4(h[4], h[5], h[6], h[7]);
}
// Pearl's ticket fold: per 16 x 16 subtile of C~, 16 lanes, lane (rb, cb) folding its 4 x 4 block row-major,
// lane <- rotl32(lane 0x9E3779B1 + bits, 13)
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

// ---------------------------------------------------------------- -h3's row digests: SHA-512, sha512/row-seg/v1
// `rowleaf.sha512_row_seg_digest` of each FP32 row (FIPS 180-4 SHA-512, every word big-endian): d_j = SHA-512(S(j, s, 8192) ‖
// segment j) over its s 1 KiB segments, then SHA-512(Q(8 · row bytes) ‖ d_0 ‖ … ‖ d_(s−1)), S and Q the format's 128-byte
// headers.  k is a multiple of 256 here, so every segment is whole.
__constant__ uint8_t RS_TAG[25] = {'v', 'e', 'r', 'i', 't', 'y', '/', 's', 'h', 'a', '5', '1', '2', '-', 'r', 'o', 'w', '-',
                                   's', 'e', 'g', '/', 'v', '1', 0};
__constant__ uint8_t RS_SEG_TAG[33] = {'v', 'e', 'r', 'i', 't', 'y', '/', 's', 'h', 'a', '5', '1', '2', '-', 'r', 'o', 'w', '-',
                                       's', 'e', 'g', '/', 'v', '1', '/', 's', 'e', 'g', 'm', 'e', 'n', 't', 0};
constexpr int RS_SEG = 1024, RS_MAX_SEGS = 128;   // a segment's bytes; s512_rows' most segments a row (k <= 32768)
// the final padding block of a message of `bytes` bytes, a multiple of 128
DEV void rs_pad(uint64_t* st, uint64_t bytes) {
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = 0;
  w[0] = 0x8000000000000000ull;
  w[15] = 8 * bytes;
  rs_compress(st, w);
}
// the state after a format header, its first block: TL bytes of `tag`, u64be f0, u64be f1, f2 as L2 big-endian bytes, zeros
template <int TL, int L2> DEV void rs_header(uint64_t* st, const uint8_t* tag, uint64_t f0, uint64_t f1, uint64_t f2) {
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    uint64_t x = 0;
#pragma unroll
    for (int t = 0; t < 8; t++) {
      const int j = 8 * i + t;
      const uint64_t v = j < TL ? tag[j] : j < TL + 8 ? f0 >> (8 * (TL + 7 - j)) : j < TL + 16 ? f1 >> (8 * (TL + 15 - j))
                       : j < TL + 16 + L2 ? f2 >> (8 * (TL + 15 + L2 - j)) : 0;
      x = x << 8 | (v & 0xFF);
    }
    w[i] = x;
  }
  rs_iv(st);
  rs_compress(st, w);
}
DEV void s512_store(uint64_t p, const uint64_t* st, int words) {   // the first `words` words of a state at device address p
  uint32_t* out = reinterpret_cast<uint32_t*>(p);
#pragma unroll
  for (int t = 0; t < 8; t++)
    if (t < words) {
      out[2 * t] = rs_be32((uint32_t)(st[t] >> 32));
      out[2 * t + 1] = rs_be32((uint32_t)st[t]);
    }
}
// the digest over s segment digests (dig, 8 words each) after the Q header's state qst, into p (64 bytes)
DEV void rs_fold(const uint64_t* qst, const uint64_t* dig, int s, uint64_t p) {
  uint64_t st[8], w[16];
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = qst[t];
  int j = 0;
#pragma unroll 1
  for (; j + 1 < s; j += 2) {
#pragma unroll
    for (int t = 0; t < 16; t++) w[t] = dig[8 * j + t];
    rs_compress(st, w);
  }
  if (j < s) {
#pragma unroll
    for (int t = 0; t < 8; t++) { w[t] = dig[8 * j + t]; w[t + 8] = 0; }
    w[8] = 0x8000000000000000ull;
    w[15] = 8ull * (128 + 64 * s);
    rs_compress(st, w);
  } else {
    rs_pad(st, 128 + 64ull * s);
  }
  s512_store(p, st, 8);
}
// -h3's row digests (`pearl_c.row_digest`): block r hashes row r (p0 + 4 i1 r, i1 = k words), its thread j < s = k / 256
// segment j into shared memory and its last thread Q, then thread 0 the row.  p0 rows, p1 out (64 B a row, the digest's
// bytes); i0 rows, i1 k (a multiple of 256, at most 256 RS_MAX_SEGS)
extern "C" __global__ void __launch_bounds__(128) s512_rows(const Args a) {
  __shared__ uint64_t dig[RS_MAX_SEGS * 8], qst[8];
  const uint64_t r = blockIdx.x;
  if (r >= (uint64_t)a.i[0]) return;
  const int k = a.i[1], s = k / 256;
  const uint64_t row = a.p[0] + 4ull * k * r;
  uint64_t st[8], w[16];
  for (int j = threadIdx.x; j < s; j += blockDim.x) {
    rs_header<33, 8>(st, RS_SEG_TAG, (uint64_t)j, (uint64_t)s, 8ull * RS_SEG);
#pragma unroll 1
    for (int c = 0; c < RS_SEG / 128; c++) {
      const uint4* blk = reinterpret_cast<const uint4*>(row + (uint64_t)RS_SEG * j + 128 * c);
#pragma unroll
      for (int t = 0; t < 8; t++) {
        const uint4 v = blk[t];
        w[2 * t] = rs_be64(v.x, v.y);
        w[2 * t + 1] = rs_be64(v.z, v.w);
      }
      rs_compress(st, w);
    }
    rs_pad(st, 128 + RS_SEG);
#pragma unroll
    for (int t = 0; t < 8; t++) dig[8 * j + t] = st[t];
  }
  if (threadIdx.x == blockDim.x - 1) {
    rs_header<25, 4>(st, RS_TAG, 32ull * k, (uint64_t)s, (uint64_t)RS_SEG);
#pragma unroll
    for (int t = 0; t < 8; t++) qst[t] = st[t];
  }
  __syncthreads();
  if (threadIdx.x == 0) rs_fold(qst, dig, s, a.p[1] + 64 * r);
}

// ---------------------------------------------------------------- s512: the labels' hash
__constant__ char S512_ROLE[5][24] = {"pearl-c/h3s/seed-A", "pearl-c/h3s/seed-B", "pearl-c/h3s/noise-line", "pearl-c/h3s/line",
                                      "pearl-c/h3s/jackpot"};
enum { S512_SEED_A = 0, S512_SEED_B = 1, S512_NOISE_LINE = 2, S512_LINE = 3, S512_JACKPOT = 4 };
// the state after role's label block (`pearl_c.S512_LABELS`)
DEV void s512_label(uint64_t* st, int role) {
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    uint64_t x = 0;
#pragma unroll
    for (int t = 0; t < 8; t++) x = x << 8 | (8 * i + t < 24 ? (uint64_t)(uint8_t)S512_ROLE[role][8 * i + t] : 0);
    w[i] = x;
  }
  rs_iv(st);
  rs_compress(st, w);
}
// the first 32 bytes of a state as 8 little-endian u32 (a seed or key, as load_key reads one)
DEV void s512_out32(const uint64_t* st, uint32_t* h) {
#pragma unroll
  for (int t = 0; t < 4; t++) {
    h[2 * t] = rs_be32((uint32_t)(st[t] >> 32));
    h[2 * t + 1] = rs_be32((uint32_t)st[t]);
  }
}
// H_noise-line(seed): a seed's line key, `pearl_c._s512_line_key`
DEV void s512_line_key(const uint32_t* seed, uint32_t* key) {
  uint64_t st[8], w[16];
  s512_label(st, S512_NOISE_LINE);
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(seed[2 * i], seed[2 * i + 1]) : 0;
  w[4] = 0x8000000000000000ull;
  w[15] = 8 * (128 + 32);
  rs_compress(st, w);
  s512_out32(st, key);
}
// a line's 32 raw bytes, H_line(key ‖ side ‖ factor ‖ u32 LE line) (`pearl_c.sample_line`), as 8 little-endian u32
DEV void s512_line_raw(const uint32_t* key, uint32_t side, uint32_t factor, uint32_t line, uint32_t* h) {
  uint64_t st[8], w[16];
  s512_label(st, S512_LINE);
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(key[2 * i], key[2 * i + 1]) : 0;
  w[4] = (uint64_t)(side & 0xFF) << 56 | (uint64_t)(factor & 0xFF) << 48 | (uint64_t)(line & 0xFF) << 40 |
         (uint64_t)(line >> 8 & 0xFF) << 32 | (uint64_t)(line >> 16 & 0xFF) << 24 | (uint64_t)(line >> 24) << 16 | 0x80ull << 8;
  w[15] = 8 * (128 + 38);
  rs_compress(st, w);
  s512_out32(st, h);
}

// -h3's seeds (`pearl_c.row_seed`, `pearl_c.unit_seed_a`): H_seed-A(salt ‖ x ‖ root_B ‖ u64 LE ι), 168 bytes after the label,
// and each seed's line key H_noise-line(seed).  i2 = 1: row r < i0, x its 64-byte row digest (p0 + 64 r, s512_rows') and
// ι = unit · 2^32 + r; i2 = 0: the unit's seed_A, x root_A (p0) and ι = the unit index.  p0 x, p1 seeds out (32 B each), p2
// line keys out (32 B each) or 0, p3 the unit block: salt (32) ‖ root_B (64) ‖ u64 LE unit index (a graph's per-call slot);
// i0 rows (1 at mode 0), i2 the mode
extern "C" __global__ void __launch_bounds__(128, 1) s512_seeds(const Args a) {
  const uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= (uint64_t)a.i[0]) return;
  const uint32_t* x = reinterpret_cast<const uint32_t*>(a.p[0] + 64 * r);
  const uint32_t* unit = reinterpret_cast<const uint32_t*>(a.p[3]);
  const uint64_t index = (uint64_t)unit[24] | (uint64_t)unit[25] << 32;
  const uint64_t iota = a.i[2] ? (index << 32 | r) : index;
  uint64_t st[8], w[16];
  s512_label(st, S512_SEED_A);
#pragma unroll
  for (int i = 0; i < 16; i++) {   // salt ‖ x ‖ root_B[0:32]
    const uint32_t* src = i < 4 ? unit + 2 * i : i < 12 ? x + 2 * (i - 4) : unit + 8 + 2 * (i - 12);
    w[i] = rs_be64(src[0], src[1]);
  }
  rs_compress(st, w);
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(unit[16 + 2 * i], unit[17 + 2 * i]) : 0;   // root_B[32:64] ‖ ι
  w[4] = rs_be64((uint32_t)iota, (uint32_t)(iota >> 32));
  w[5] = 0x8000000000000000ull;
  w[15] = 8 * (128 + 168);
  rs_compress(st, w);
  uint32_t seed[8];
  s512_out32(st, seed);
  store_cv(a.p[1] + 32 * r, seed);
  if (a.p[2]) {
    uint32_t key[8];
    s512_line_key(seed, key);
    store_cv(a.p[2] + 32 * r, key);
  }
}

// lines' raw bytes for the host tests (pearl_c_sm120.cu's lines_s maps them to codes): line t < i2 under key p0, or for
// t < i4 line 0 under its own key p4 + 32 t; p1 out (32 B a line); i0 side, i1 factor
extern "C" __global__ void __launch_bounds__(128) s512_line_raws(const Args a) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x;
  if (line >= a.i[2]) return;
  const bool own = line < a.i[4];
  uint32_t key[8], h[8];
  load_key(own ? a.p[4] + 32ull * line : a.p[0], key);
  s512_line_raw(key, (uint32_t)a.i[0], (uint32_t)a.i[1], own ? 0u : (uint32_t)line, h);
  store_cv(a.p[1] + 32ull * line, h);
}

// -h3's tickets (`PearlC.tickets`): per 16 x 16 subtile of C~, H_jackpot(seed_A ‖ fold), the fold `ticket_fold`'s.  p0 C~, p1 out
// (subtiles row-major x 32 B), p2 seed_A (32 B); i0 rows, i1 cols
extern "C" __global__ void __launch_bounds__(128) s512_tickets(const Args a) {
  const int cols = a.i[1], sn = cols / 16;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] / 16 * sn) return;
  uint32_t z[16], seed[8], h[8];
  ticket_fold(reinterpret_cast<const uint32_t*>(a.p[0]), cols, t / sn, t % sn, z);
  load_key(a.p[2], seed);
  uint64_t st[8], w[16];
  s512_label(st, S512_JACKPOT);
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(seed[2 * i], seed[2 * i + 1]) : i < 12 ? rs_be64(z[2 * i - 8], z[2 * i - 7]) : 0;
  w[12] = 0x8000000000000000ull;
  w[15] = 8 * (128 + 96);
  rs_compress(st, w);
  s512_out32(st, h);
  store_cv(a.p[1] + 32ull * t, h);
}

// ---------------------------------------------------------------- sha512/row-seg/v1 (`rowleaf.sha512_row_seg_digest`)
// -h3's tile leaves (`pearl_c.leaf`): `sha512/row-seg/v1` over the tile's messages, row-major, each row's two 32-column chunks
// as le32 C~ ‖ le32 U, 512 bytes a row, so segment j is rows 2j and 2j + 1.  One CTA a tile (tiles row-major, a last row block of
// R = 16, 32 or 48 rows partial), thread j < s = R / 2 its segment j, then thread 0 the leaf.  p0 C~ (rows x cols), p1 U, p2 out
// (64 B a tile); i0 rows (a multiple of 16), i1 cols (a multiple of 64)
extern "C" __global__ void __launch_bounds__(32, 1) s512_tile_leaves(const Args a) {
  __shared__ uint64_t dig[32 * 8], qst[8];
  const int cols = a.i[1], tn = cols / 64, t = blockIdx.x;
  const int ti = t / tn, tj = t % tn, R = min(64, a.i[0] - 64 * ti), s = R / 2;
  uint64_t st[8], w[16];
  const int j = threadIdx.x;   // s <= 32: one segment a thread
  if (j < s) {
    rs_header<33, 8>(st, RS_SEG_TAG, (uint64_t)j, (uint64_t)s, 8ull * RS_SEG);
#pragma unroll 1
    for (int c = 0; c < RS_SEG / 128; c++) {
      const size_t row = 64 * (size_t)ti + 2 * j + c / 4;
      const uint4* blk = reinterpret_cast<const uint4*>(a.p[c & 1]) + (row * cols + 64 * tj + 32 * (c >> 1 & 1)) / 4;
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const uint4 v = blk[q];
        w[2 * q] = rs_be64(v.x, v.y);
        w[2 * q + 1] = rs_be64(v.z, v.w);
      }
      rs_compress(st, w);
    }
    rs_pad(st, 128 + RS_SEG);
#pragma unroll
    for (int q = 0; q < 8; q++) dig[8 * j + q] = st[q];
  }
  if (threadIdx.x == blockDim.x - 1) {
    rs_header<25, 4>(st, RS_TAG, 4096ull * R, (uint64_t)s, (uint64_t)RS_SEG);
#pragma unroll
    for (int q = 0; q < 8; q++) qst[q] = st[q];
  }
  __syncthreads();
  if (threadIdx.x == 0) rs_fold(qst, dig, s, a.p[2] + 64ull * t);
}

// ---------------------------------------------------------------- frame-v3-sha512 messages, a thread's own in shared memory
// The row commitment's sink (rc::put_*): byte n of a thread's message is byte n % 4 of word (n / 4) · stride of its slice, so
// a warp's threads writing their n-th bytes hit 32 banks.
constexpr int SMSG_WORDS = 96;   // 384 bytes: a node's message (at most 282 bytes) and its padding
__constant__ uint8_t HM96_SCHEMA[22] = {'h', 'm', '9', '6', '-', 's', 'h', 'a', '5', '1', '2', '/', 'r', 'o', 'w', '-', 's', 'e',
                                        'g', '/', 'v', '1'};
struct SMsg {
  uint32_t* w;
  int stride, n;
  RC_HD void put(uint32_t v) {
    uint32_t& x = w[(n >> 2) * stride];
    const int s = 8 * (n & 3);
    x = (s ? x & ~(0xFFu << s) : 0u) | (v & 0xFF) << s;
    n++;
  }
  RC_HD uint64_t word(int i) const { return sx::be64(w[2 * i * stride], w[(2 * i + 1) * stride]); }
};
DEV void sm_mem64(SMsg& m, uint64_t p) {   // a part: the 64 bytes at device address p
  rc::put_words(m, reinterpret_cast<const uint32_t*>(p), 16);
}
DEV void sm_frame(SMsg& m, const char* tag, uint64_t dom) {   // the tag, then the domain id's part
  rc::put_tag(m, tag);
  sm_mem64(m, dom);
}
// SHA-512 of the message, into p (64 bytes)
DEV void sm_hash(SMsg& m, uint64_t p) {
  uint64_t st[8];
  rc::digest<PcSha>(m, st);
  s512_store(p, st, 8);
}
#define SMSG(T)                                       \
  __shared__ uint32_t smsg_buf[SMSG_WORDS * (T)];     \
  SMsg m{smsg_buf + threadIdx.x, (int)blockDim.x, 0}

// pads of a frame-v3-sha512 tree: ranks i0 … i1 − 1 of level 0, `CommitmentDomain.pad`.  p0 tree, p1 domain id (64 B)
extern "C" __global__ void __launch_bounds__(64) s512_pads(const Args a) {
  SMSG(64);
  const uint64_t rank = (uint64_t)a.i[0] + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= (uint64_t)a.i[1]) return;
  sm_frame(m, "pad", a.p[1]);
  rc::put_uint(m, rank);
  sm_hash(m, a.p[0] + 64 * rank);
}

// node i of level i0 + 1 from nodes 2i and 2i + 1 of level i0, `CommitmentDomain.node`
DEV void s512_node(SMsg& m, uint64_t dom, int level, uint64_t i, uint64_t below, uint64_t above) {
  m.n = 0;
  sm_frame(m, "node", dom);
  rc::put_uint(m, (uint64_t)level);
  rc::put_uint(m, i);
  sm_mem64(m, below + 128 * i);
  sm_mem64(m, below + 128 * i + 64);
  sm_hash(m, above + 64 * i);
}
// one level of a frame-v3-sha512 tree.  p0 the level below, p1 the level above, p2 domain id; i0 the level below's index,
// i1 nodes above
extern "C" __global__ void __launch_bounds__(64) s512_nodes(const Args a) {
  SMSG(64);
  const uint64_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (uint64_t)a.i[1]) return;
  s512_node(m, a.p[2], a.i[0], i, a.p[0], a.p[1]);
}
// every level above a frame-v3-sha512 tree's level i1 in one CTA, the levels s512_nodes writes in the same places.  p0 level
// i1 (width i0, a power of two), each level above it after it; p1 domain id
extern "C" __global__ void __launch_bounds__(64) s512_tree_small(const Args a) {
  SMSG(64);
  uint64_t below = a.p[0];
  int level = a.i[1];
  for (uint64_t w = (uint64_t)a.i[0]; w > 1; w >>= 1, level++) {
    const uint64_t above = below + 64 * w;
    for (uint64_t i = threadIdx.x; i < w / 2; i += blockDim.x) s512_node(m, a.p[1], level, i, below, above);
    __syncthreads();
    below = above;
  }
}

// ---------------------------------------------------------------- hm96-sha512 leaves (`audit.Hm96Domain.leaf`)
// ChaCha20's block, 20 rounds, a 64-bit counter (words 12, 13) and a 64-bit nonce (words 14, 15): `audit.chacha20_block`
#define H3S_QR(a, b, c, d)                                                                         \
  {                                                                                                \
    a += b; d = __funnelshift_l(d ^ a, d ^ a, 16); c += d; b = __funnelshift_l(b ^ c, b ^ c, 12);  \
    a += b; d = __funnelshift_l(d ^ a, d ^ a, 8); c += d; b = __funnelshift_l(b ^ c, b ^ c, 7);    \
  }
DEV void chacha20_block(const uint32_t* key, uint64_t counter, uint64_t nonce, uint32_t* out) {
  uint32_t x[16];
  x[0] = 0x61707865u; x[1] = 0x3320646eu; x[2] = 0x79622d32u; x[3] = 0x6b206574u;
#pragma unroll
  for (int i = 0; i < 8; i++) x[4 + i] = key[i];
  x[12] = (uint32_t)counter; x[13] = (uint32_t)(counter >> 32); x[14] = (uint32_t)nonce; x[15] = (uint32_t)(nonce >> 32);
#pragma unroll
  for (int r = 0; r < 10; r++) {
    H3S_QR(x[0], x[4], x[8], x[12]); H3S_QR(x[1], x[5], x[9], x[13]); H3S_QR(x[2], x[6], x[10], x[14]); H3S_QR(x[3], x[7], x[11], x[15]);
    H3S_QR(x[0], x[5], x[10], x[15]); H3S_QR(x[1], x[6], x[11], x[12]); H3S_QR(x[2], x[7], x[8], x[13]); H3S_QR(x[3], x[4], x[9], x[14]);
  }
  out[0] = x[0] + 0x61707865u; out[1] = x[1] + 0x3320646eu; out[2] = x[2] + 0x79622d32u; out[3] = x[3] + 0x6b206574u;
#pragma unroll
  for (int i = 0; i < 8; i++) out[4 + i] = x[4 + i] + key[i];
  out[12] = x[12] + (uint32_t)counter; out[13] = x[13] + (uint32_t)(counter >> 32);
  out[14] = x[14] + (uint32_t)nonce; out[15] = x[15] + (uint32_t)(nonce >> 32);
}
DEV uint64_t rs_bswap64(uint64_t v) { return rs_be64((uint32_t)v, (uint32_t)(v >> 32)); }

// the inner leaves of an `audit.HM96` tree (C-Flock's ``hm96_finish_leaves``): leaf i < i0 over x_i (p0 + 64 i: a row's
// sha512/row-seg/v1 digest, or a tile leaf) and its salt y_i = `audit.hm96_salt`(key, nonce, i): b = x ⊕ M·y (the byte
// tables), c = SHA-512(salt prefix ‖ y), hl = SHA-512(leaf prefix ‖ b ‖ c); s512_frame_leaves makes hl the frame leaf.  p0 x,
// p1 out (64 B a leaf, hl), p2 the salt key (32 B), p3 `audit.hm96_kernel_constants` (the prefixes' midstates, 8 u64 each,
// then the byte tables, [192][256] entries of 64 B little-endian), p5 0 or the address of the nonce (u64 LE: an activation
// tree's unit index, in the unit block); i0 leaves, i1 and i2 the nonce's low and high 32 bits (when p5 is 0).  The salt
// stays in registers, unrolled, so the table rows' loads are independent.
extern "C" __global__ void __launch_bounds__(64) hm_leaves(const Args a) {
  const uint64_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (uint64_t)a.i[0]) return;
  const uint64_t nonce = a.p[5] ? *reinterpret_cast<const uint64_t*>(a.p[5])
                                : (uint64_t)(uint32_t)a.i[1] | (uint64_t)(uint32_t)a.i[2] << 32;
  const uint64_t* mid = reinterpret_cast<const uint64_t*>(a.p[3]);
  const uint64_t* tables = reinterpret_cast<const uint64_t*>(a.p[3] + 128);
  uint32_t key[8], y[48], mk[16];
  load_key(a.p[2], key);
#pragma unroll
  for (int j = 0; j < 3; j++) chacha20_block(key, 3 * i + j, nonce, y + 16 * j);
  rc::mask_positions(rc::PosTab<false>{reinterpret_cast<const uint8_t*>(tables)}, y, mk);
  const uint32_t* x = reinterpret_cast<const uint32_t*>(a.p[0] + 64 * i);
  uint64_t B[16], h[8];
  rc::mask_words(mk, B);
#pragma unroll
  for (int k = 0; k < 8; k++) B[k] ^= rs_be64(x[2 * k], x[2 * k + 1]);   // b = x ⊕ M·y
  rc::salt_digest<PcSha>(mid, rc::SaltU32{y, 1}, B + 8);
  rc::hm96_value<PcSha>(mid + 8, sx::kw_pad256(), B, h);
  s512_store(a.p[1] + 64 * i, h, 8);
}

// level 0 of an `audit.HM96` tree from hm_leaves' hl, in place: leaf i < i1 the frame leaf over hl_i (schema
// hm96-sha512/row-seg/v1, rank = position = i).  p0 the level (64 B a leaf), p1 domain id (64 B); i1 leaves
extern "C" __global__ void __launch_bounds__(64) s512_frame_leaves(const Args a) {
  SMSG(64);
  const uint64_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (uint64_t)a.i[1]) return;
  sm_frame(m, "leaf", a.p[1]);
  rc::put_uint(m, i);
  rc::put_uint(m, i);
  rc::put_part(m, HM96_SCHEMA, 22);
  sm_mem64(m, a.p[0] + 64 * i);
  sm_hash(m, a.p[0] + 64 * i);
}
