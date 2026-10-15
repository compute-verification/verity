// sha512/row-seg/v1's row digests and tile leaves (hash_h3.cuh's s512_rows and s512_tile_leaves, the same bytes) in two
// launches each, so no lane waits on another: s512_row_segs / s512_tile_segs, a thread a 1 KiB segment over every row or tile,
// starting from its segment header's state (S(j, s, 8192) is a function of (j, s) alone, RS_SEG_HEADS) and writing the
// segment's digest state into scratch; then s512_folds, a thread a row or tile, from the Q header's state (Q(1024 s · 8, s,
// 1024) is a function of s alone: a row of k = 256 s words and a tile of R = 2 s rows both hash s KiB, RS_Q_HEADS) over its s
// digests.  s512_heads writes both tables once per module.  RS_FAST_NT and RS_FAST_MINB are the segment kernels' launch bounds.
#pragma once

#ifndef RS_FAST_NT
#define RS_FAST_NT 128
#endif
#ifndef RS_FAST_MINB
#define RS_FAST_MINB 2
#endif
constexpr int RS_FOLD_NT = 32;
constexpr int RS_HEAD_SEGS = RS_MAX_SEGS * (RS_MAX_SEGS + 1) / 2;   // (j, s) for 1 <= s <= RS_MAX_SEGS, j < s: s (s - 1) / 2 + j

__device__ uint64_t RS_SEG_HEADS[RS_HEAD_SEGS * 8];
__device__ uint64_t RS_Q_HEADS[(RS_MAX_SEGS + 1) * 8];

// the header tables: thread t < RS_HEAD_SEGS the state after S(j, s), the next RS_MAX_SEGS + 1 the state after Q(s)
extern "C" __global__ void __launch_bounds__(128) s512_heads(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t st[8];
  if (t < RS_HEAD_SEGS) {
    int s = 1;
    while (s * (s + 1) / 2 <= t) s++;
    const int j = t - s * (s - 1) / 2;
    rs_header<33, 8>(st, RS_SEG_TAG, (uint64_t)j, (uint64_t)s, 8ull * RS_SEG);
#pragma unroll
    for (int q = 0; q < 8; q++) RS_SEG_HEADS[8 * t + q] = st[q];
  } else if (t < RS_HEAD_SEGS + RS_MAX_SEGS + 1) {
    const int s = t - RS_HEAD_SEGS;
    rs_header<25, 4>(st, RS_TAG, 8192ull * s, (uint64_t)s, (uint64_t)RS_SEG);
#pragma unroll
    for (int q = 0; q < 8; q++) RS_Q_HEADS[8 * s + q] = st[q];
  }
}

DEV void rs_seg_head(uint64_t* st, int j, int s) {
  const uint64_t* h = RS_SEG_HEADS + 8 * (s * (s - 1) / 2 + j);
#pragma unroll
  for (int q = 0; q < 8; q++) st[q] = h[q];
}
DEV void rs_block(uint64_t* st, uint64_t* w, const uint4* blk) {   // one 128-byte block of little-endian u32 words
#pragma unroll
  for (int q = 0; q < 8; q++) {
    const uint4 v = blk[q];
    w[2 * q] = rs_be64(v.x, v.y);
    w[2 * q + 1] = rs_be64(v.z, v.w);
  }
  rs_compress(st, w);
}
DEV void rs_seg_out(uint64_t p, const uint64_t* st) {
  uint4* o = reinterpret_cast<uint4*>(p);
#pragma unroll
  for (int q = 0; q < 4; q++)
    o[q] = make_uint4((uint32_t)st[2 * q], (uint32_t)(st[2 * q] >> 32), (uint32_t)st[2 * q + 1], (uint32_t)(st[2 * q + 1] >> 32));
}

// segment t of the rows (row t / s, segment t % s, s = k / 256: at p0 + 1 KiB · t, the rows being contiguous) into p1 + 64 t.
// p0 rows, p1 segment digests; i0 rows, i1 k (a multiple of 256, at most 256 RS_MAX_SEGS)
extern "C" __global__ void __launch_bounds__(RS_FAST_NT, RS_FAST_MINB) s512_row_segs(const Args a) {
  const int s = a.i[1] / 256;
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (uint64_t)a.i[0] * s) return;
  uint64_t st[8], w[16];
  rs_seg_head(st, (int)(t % s), s);
  const uint4* seg = reinterpret_cast<const uint4*>(a.p[0] + (uint64_t)RS_SEG * t);
#pragma unroll 1
  for (int c = 0; c < RS_SEG / 128; c++) rs_block(st, w, seg + 8 * c);
  rs_pad(st, 128 + RS_SEG);
  rs_seg_out(a.p[1] + 64 * t, st);
}

// the tiles' segments (s512_tile_leaves' messages): segment j of tile (ti, tj) is the tile's rows 2j and 2j + 1, each row's two
// 32-column chunks as le32 C~ ‖ le32 U.  The full row blocks' tiles first, 32 segments each, then the last row block's (R < 64
// rows, s = R / 2), so segment t's digest goes to p2 + 64 t and tile u's s digests are contiguous.  p0 C~ (rows x cols), p1 U,
// p2 segment digests; i0 rows (a multiple of 16), i1 cols (a multiple of 64)
extern "C" __global__ void __launch_bounds__(RS_FAST_NT, RS_FAST_MINB) s512_tile_segs(const Args a) {
  const int rows = a.i[0], cols = a.i[1], tn = cols / 64;
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, n_full = (uint64_t)(rows / 64) * tn * 32;
  int ti, tj, j, s;
  if (t < n_full) {
    const uint64_t tile = t / 32;
    s = 32, j = (int)(t % 32), ti = (int)(tile / tn), tj = (int)(tile % tn);
  } else {
    s = rows % 64 / 2;
    const uint64_t q = t - n_full;
    if (!s || q >= (uint64_t)tn * s) return;
    ti = rows / 64, tj = (int)(q / s), j = (int)(q % s);
  }
  uint64_t st[8], w[16];
  rs_seg_head(st, j, s);
#pragma unroll 1
  for (int c = 0; c < RS_SEG / 128; c++) {
    const size_t row = 64 * (size_t)ti + 2 * j + c / 4;
    rs_block(st, w, reinterpret_cast<const uint4*>(a.p[c & 1]) + (row * cols + 64 * tj + 32 * (c >> 1 & 1)) / 4);
  }
  rs_pad(st, 128 + RS_SEG);
  rs_seg_out(a.p[2] + 64 * t, st);
}

// unit u's digest (a row's or a tile's) over its s segment digests into p1 + 64 u: units u < i2 have i1 segments each, from
// p0 + 64 i1 u, the rest i3 each after them.  p0 segment digests, p1 out (64 B a unit); i0 units, i1 s, i2 units at i1, i3 s
// of the rest
extern "C" __global__ void __launch_bounds__(RS_FOLD_NT) s512_folds(const Args a) {
  const uint64_t u = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= (uint64_t)a.i[0]) return;
  const bool head = u < (uint64_t)a.i[2];
  const int s = head ? a.i[1] : a.i[3];
  const uint64_t off = head ? u * a.i[1] : (uint64_t)a.i[2] * a.i[1] + (u - a.i[2]) * a.i[3];
  rs_fold(RS_Q_HEADS + 8 * s, reinterpret_cast<const uint64_t*>(a.p[0]) + 8 * off, s, a.p[1] + 64 * u);
}
