// sha512/row-seg/v1's digests of rows past RS_MAX_SEGS segments, up to RS_WIDE_MAX_SEGS: the bytes `pearl_c.row_digest`
// gives, by hash_fast.cuh's two launches with the header states out of the module.  s512_rows keeps a row's digests in shared
// memory and hash_fast.cuh's tables hold every (j, s) up to RS_MAX_SEGS, so neither takes a wider row; here one width's
// s + 1 states (after S(j, s, 8192) for each j < s, then after Q(8192 s, s, 1024)) are a caller's buffer, which
// s512_heads_w writes once a width.  s512_row_segs_w, a thread a 1 KiB segment over every row, starts from its segment's state
// there and writes the segment's digest state into scratch; s512_folds_w, a thread a row, folds the row's s digests from Q's.
// Given no table (the pointer 0) each kernel computes its header itself, one compression more a segment: the reference form,
// which reads nothing s512_heads_w wrote.  Every kernel here traps unless k is a positive multiple of 256 of at most
// RS_WIDE_MAX_SEGS segments.  Nothing is in shared memory.  A row of s segments still folds in one SHA-512 message, on one
// thread: s / 2 + 1 compressions for an even s, (s + 1) / 2 for an odd one.  Include after hash_fast.cuh.
#pragma once

constexpr int RS_WIDE_MAX_SEGS = 1024;   // the most segments a row here (k <= 262144)

DEV int rs_wide_segs(int k) {   // a launch's segments a row, k / 256
  if (k <= 0 || k % 256 || k / 256 > RS_WIDE_MAX_SEGS) __trap();
  return k / 256;
}

// one width's header states into p0 (64 B each): thread j < s the state after S(j, s), thread s the state after Q.  i0 k
extern "C" __global__ void __launch_bounds__(128) s512_heads_w(const Args a) {
  const int s = rs_wide_segs(a.i[0]);
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t > s) return;
  uint64_t st[8];
  if (t < s) rs_header<33, 8>(st, RS_SEG_TAG, (uint64_t)t, (uint64_t)s, 8ull * RS_SEG);
  else rs_header<25, 4>(st, RS_TAG, 8192ull * s, (uint64_t)s, (uint64_t)RS_SEG);
  uint64_t* out = reinterpret_cast<uint64_t*>(a.p[0]) + 8 * t;
#pragma unroll
  for (int q = 0; q < 8; q++) out[q] = st[q];
}

// segment t of the rows (row t / s, segment t % s, s = k / 256: at p0 + 1 KiB · t, the rows being contiguous) into p1 + 64 t,
// as s512_row_segs.  p0 rows, p1 segment digests, p2 the width's header states (s512_heads_w's) or 0; i0 rows, i1 k
extern "C" __global__ void __launch_bounds__(RS_FAST_NT, RS_FAST_MINB) s512_row_segs_w(const Args a) {
  const int s = rs_wide_segs(a.i[1]);
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((int64_t)t >= (int64_t)a.i[0] * s) return;
  const int j = (int)(t % s);
  uint64_t st[8], w[16];
  if (a.p[2]) {
    const uint64_t* h = reinterpret_cast<const uint64_t*>(a.p[2]) + 8 * j;
#pragma unroll
    for (int q = 0; q < 8; q++) st[q] = h[q];
  } else {
    rs_header<33, 8>(st, RS_SEG_TAG, (uint64_t)j, (uint64_t)s, 8ull * RS_SEG);
  }
  const uint4* seg = reinterpret_cast<const uint4*>(a.p[0] + (uint64_t)RS_SEG * t);
#pragma unroll 1
  for (int c = 0; c < RS_SEG / 128; c++) rs_block(st, w, seg + 8 * c);
  rs_pad(st, 128 + RS_SEG);
  rs_seg_out(a.p[1] + 64 * t, st);
}

// row u's digest over its s segment digests (at p0 + 64 s u) into p1 + 64 u, as s512_folds at one width.  p0 segment digests,
// p1 out (64 B a row), p2 the width's header states or 0; i0 rows, i1 k
extern "C" __global__ void __launch_bounds__(RS_FOLD_NT) s512_folds_w(const Args a) {
  const int s = rs_wide_segs(a.i[1]);
  const uint64_t u = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((int64_t)u >= (int64_t)a.i[0]) return;
  uint64_t qst[8];
  if (a.p[2]) {
    const uint64_t* h = reinterpret_cast<const uint64_t*>(a.p[2]) + 8 * s;
#pragma unroll
    for (int q = 0; q < 8; q++) qst[q] = h[q];
  } else {
    rs_header<25, 4>(qst, RS_TAG, 8192ull * s, (uint64_t)s, (uint64_t)RS_SEG);
  }
  rs_fold(qst, reinterpret_cast<const uint64_t*>(a.p[0]) + 8ull * s * u, s, a.p[1] + 64 * u);
}
