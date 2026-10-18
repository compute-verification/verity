// sharaw: hm96-sha512/row/v2 leaves of 4 KiB rows in a frame-v3-sha512 tree, bit for bit commit_gpu's (and so
// verity.ml.commitments'), built to run a large step at the raw SHA-512 compression rate. It includes compute-accounting's
// committer (benchmarks/pouw/commit_gpu/commit_gpu.cuh, unedited) for the construction's device functions and constants,
// and times three committers on one die, rep by rep: commit_gpu's own path (salt_kernel, leaf_kernel, a launch a level),
// circuits' rows-sha512 kernels (native_rows.cu, dlopen'd, rows_floor's schedule) and this file's.
//
// v2 (Mine::v2): sr_leaf does the whole leaf a thread, the salt included, so the mask's shared-memory traffic runs beside
// other warps' compressions; then either (FOLD) its warps fold and climb to the root in the same launch, which suits a step
// too small to fill the GPU, or sr_sub launches of three levels a thread over the wide levels and one sr_tree launch that
// folds and climbs the rest.
//
// v1 is two kernels:
//   sr_salt  a thread a position: y (3 compressions from round 8), M_key y from the 8 KB nibble table in shared memory,
//            c (2), and the frame leaf's first block (from round 15), to S[pos] (128 B) and F[pos] (64 B); a pad for
//            n <= pos < width straight into level 0. Nothing here reads the operand.
//   sr_data  a thread a position: x over the 4 KiB row (32 data blocks, then the all-padding block from its precomputed
//            schedule), h (2), the frame leaf's second block; then the CTA folds its own aligned subtree in shared memory,
//            and thread 0 climbs: at each level above, the second of two sibling subtrees to finish (an atomic count a
//            parent) folds the parent and goes on, so the whole tree is this one launch and nothing waits on a level.
//
//   sharaw check PARAMS OUT rows N 32768 4096 ROWS prf     commit_gpu's check interface (levels.bin, bc.bin) for reference.py
//   sharaw bench PARAMS N REPS [NATIVE_SO]                  JSON lines on stdout
//   sharaw nsys PARAMS N ARM                                 one arm, twice, for a profiler
#include <dlfcn.h>
#include <nvml.h>

#include <chrono>
#include <functional>

#define COMMIT_GPU_DOMS 1
#include "../../pouw/commit_gpu/commit_gpu.cuh"

namespace sr {
constexpr u32 LB = 4096;          // row bytes: sha512/row/v2 at 32768 bits
constexpr int NBLK = LB / 128;    // its whole data blocks; the padding block is C.kw_xpad

// Diagnostic forms (timing only, never verified): what a part of the leaf costs, by leaving it out.
enum Diag : int { FULL = 0, NOLOAD = 1, ROWONLY = 2, NOFOLD = 3 };

// x = SHA-512(P(32768) || row). PF: block b+1's loads are issued before block b's compression (a register double buffer).
template <bool PF, int DG>
DEV void row4k(const uint8_t* __restrict__ p, u64* x) {
  cpy8(x, C.mid_P);
  if constexpr (DG == NOLOAD) {
    const u32 s = (u32)(size_t)p;
#pragma unroll 1
    for (int b = 0; b < NBLK; ++b) {
      u64 W[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) W[i] = (u64)(s ^ (u32)(16 * b + i)) * 0x9E3779B97F4A7C15ull;
      sha::compress(x, W);
    }
  } else if constexpr (PF) {
    uint4 nx[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) nx[c] = ldg16(p + 16 * c);
#pragma unroll 1
    for (int b = 0; b < NBLK; ++b) {
      u64 W[16];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        W[2 * c] = be64(nx[c].x, nx[c].y);
        W[2 * c + 1] = be64(nx[c].z, nx[c].w);
      }
      const int nb = b + 1 < NBLK ? b + 1 : b;
#pragma unroll
      for (int c = 0; c < 8; ++c) nx[c] = ldg16(p + 128 * nb + 16 * c);
      sha::compress(x, W);
    }
  } else {
#pragma unroll 1
    for (int b = 0; b < NBLK; ++b) {
      u64 W[16];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const uint4 v = ldg16(p + 128 * b + 16 * c);
        W[2 * c] = be64(v.x, v.y);
        W[2 * c + 1] = be64(v.z, v.w);
      }
      sha::compress(x, W);
    }
  }
  sha::compress_kw(x, C.kw_xpad);
}

// M_key y (hm96_mask<.., NIB>'s product) from the nibble table in shared memory: uint4 16 q + u is chunk q of T_u, and
// 256 + 16 q + u chunk q of T_16u.
DEV void mask_smem(const uint4* T, const u64* Y, u32* m) {
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
      const uint4* lo = T + 16 * q + (v & 15u);
      const uint4* hi = T + 256 + 16 * q + (v >> 4);
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const uint4 u = lo[16 * i], w = hi[16 * i];
        A[4 * i] ^= u.x ^ w.x; A[4 * i + 1] ^= u.y ^ w.y; A[4 * i + 2] ^= u.z ^ w.z; A[4 * i + 3] ^= u.w ^ w.w;
      }
    }
    const int sw = r >> 2, s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= s ? __funnelshift_r(A[w + sw], A[w + sw + 1], s) : A[w + sw];
  }
}

// MASK = false: timing only, M_key y left out.
template <int TPB, bool MASK>
__global__ void __launch_bounds__(TPB) sr_salt(const uint4* __restrict__ Tn, u64 n, u64 width, u64* __restrict__ S,
                                               u64* __restrict__ F, u64* __restrict__ lv0) {
  __shared__ uint4 T[512];
  for (int i = threadIdx.x; i < 512; i += TPB) T[i] = Tn[i];
  __syncthreads();
  const u64 pos = (u64)blockIdx.x * TPB + threadIdx.x;
  if (pos < n) {
    u64 Y[24], c[8], my[8], f[8];
    u32 m[16];
    salt_prf(pos, Y);
    if constexpr (MASK) {
      mask_smem(T, Y, m);
    } else {
#pragma unroll
      for (int w = 0; w < 16; ++w) m[w] = (u32)Y[w];
    }
    salt_digest(Y, c);
#pragma unroll
    for (int i = 0; i < 8; ++i) my[i] = (u64)bswap32(m[2 * i]) << 32 | bswap32(m[2 * i + 1]);
    st8(S + 16 * pos, my);
    st8(S + 16 * pos + 8, c);
    frame_first(0, pos, f);
    st8(F + 8 * pos, f);
  } else if (pos < width) {
    u64 o[8];
    pad(0, pos, o);
    st8(lv0 + 8 * pos, o);
  }
}

// blockDim.x = cnt = min(width, TPB) leaves a CTA, a power of two; gridDim.x = width / cnt.
template <int TPB, int MINB, bool PF, int DG>
__global__ void __launch_bounds__(TPB, MINB)
    sr_data(const uint8_t* __restrict__ rows, u64 n, u64 width, int depth, const u64* __restrict__ S, const u64* __restrict__ F,
            u64* levels, u64* __restrict__ bc, u32* ctr) {
  __shared__ u64 sm[TPB * 8];
  const u32 t = threadIdx.x, cnt = blockDim.x;
  const u64 pos = (u64)blockIdx.x * cnt + t;
  u64 o[8];
  if (pos < n) {
    u64 x[8];
    row4k<PF, DG>(rows + pos * LB, x);
    if constexpr (DG == ROWONLY) {
      cpy8(o, x);
    } else {
      leaf_from_digest<0>(0, pos, x, S, o, bc, F);
    }
    st8(levels + 8 * pos, o);
  } else {
    ld8(levels + 8 * pos, o);
  }
  if constexpr (DG == ROWONLY || DG == NOFOLD) return;
  cpy8(sm + 8 * t, o);
  int L = 0;
  while ((1u << L) < cnt) ++L;
  for (int l = 1; l <= L; ++l) {
    __syncthreads();
    const u32 m = cnt >> l;
    u64 r[8];
    if (t < m) {
      u64 a[8], b[8];
      cpy8(a, sm + 16 * t);
      cpy8(b, sm + 16 * t + 8);
      node(0, l - 1, (u64)blockIdx.x * m + t, a, b, r);
    }
    __syncthreads();
    if (t < m) {
      cpy8(sm + 8 * t, r);
      st8(level_at(levels, width, l) + 8 * ((u64)blockIdx.x * m + t), r);
    }
  }
  if (t != 0 || L >= depth) return;
  // the climb: ctr holds one count a node above level L, level l+1's at (width >> L) - (width >> l)
  u64 cur[8], idx = blockIdx.x;
  cpy8(cur, sm);
  const u64 nL = width >> L;
  for (int l = L; l < depth; ++l) {
    const u64 p = idx >> 1;
    u32* c = ctr + (nL - (width >> l)) + p;
    __threadfence();
    if (atomicAdd(c, 1u) == 0) return;   // the sibling subtree isn't done: its CTA climbs on
    *c = 0;                              // for the next launch
    __threadfence();
    u64 sib[8], r[8];
    ld8cg(level_at(levels, width, l) + 8 * (idx ^ 1), sib);
    if (idx & 1) node(0, l, p, sib, cur, r);
    else node(0, l, p, cur, sib, r);
    st8(level_at(levels, width, l + 1) + 8 * p, r);
    cpy8(cur, r);
    idx = p;
  }
}

// --- v3: v2 with every compression through one of two non-inlined bodies --------------------------------------------------
// commit_gpu's helpers inline a compression at each call, and those that switch on a length (frame leaf, pad, node) one per
// length: a leaf kernel built from them holds some 25 copies, so warps in different phases fetch different code. Here a
// block whose words the host fixed (compress_from's prefix) is hashed whole instead, which costs a few rounds a leaf.
__device__ __noinline__ void cmp(u64* st, u64* w) { sha::compress(st, w); }
__device__ __noinline__ void cmp_kw(u64* st, const u64* kw) { sha::compress_kw(st, kw); }

template <int NB>
DEV void f_first_nb(u64 pos, u64* out) {
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = CD[0].fl_tpl[NB][i];
  or_be<121, NB>(W, pos);
  sha::iv(out);
  cmp(out, W);
}
template <int NB>
DEV void f_second_nb(u64 pos, const u64* h, u64* out) {
  u64 W[32];
#pragma unroll
  for (int i = 16; i < 32; ++i) W[i] = CD[0].fl_tpl[NB][i];
  or_be<129 + NB, NB>(W, pos);
  or_digest<163 + 2 * NB>(W, h);
  cmp(out, W + 16);
}
template <int NB>
DEV void pad_nb3(u64 rank, u64* out) {
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = CD[0].pad_tpl[NB][i];
  or_be<120, NB>(W, rank);
  sha::iv(out);
  cmp(out, W);
#pragma unroll
  for (int i = 0; i < 15; ++i) W[i] = 0;
  W[15] = (120 + NB) * 8;
  cmp(out, W);
}
template <int NI>
DEV void node_ni3(int level, u64 index, const u64* L, const u64* R, u64* out) {
  u64 W[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) W[i] = CD[0].node_tpl[NI][i];
  or_be<2, NI>(W, index);
  or_digest<10 + NI>(W, L);
  or_digest<82 + NI>(W, R);
  cpy8(out, CD[0].node_mid[level]);
  cmp(out, W);
  cmp(out, W + 16);
}
#define SR_BY_LEN(v, F, ...)                     \
  switch (nbytes(v)) {                           \
    case 1: F<1>(__VA_ARGS__); break;            \
    case 2: F<2>(__VA_ARGS__); break;            \
    case 3: F<3>(__VA_ARGS__); break;            \
    default: F<4>(__VA_ARGS__); break;           \
  }
DEV void f_first3(u64 pos, u64* out) { SR_BY_LEN(pos, f_first_nb, pos, out) }
DEV void f_second3(u64 pos, const u64* h, u64* out) { SR_BY_LEN(pos, f_second_nb, pos, h, out) }
DEV void pad3(u64 rank, u64* out) { SR_BY_LEN(rank, pad_nb3, rank, out) }
DEV void node3(int level, u64 index, const u64* L, const u64* R, u64* out) { SR_BY_LEN(index, node_ni3, level, index, L, R, out) }

DEV void salt3(u64 pos, const uint4* T, u64* my, u64* c) {
  u64 Y[24];
  u32 m[16];
#pragma unroll 1
  for (int r = 0; r < 3; ++r) {
    u64 w[16], s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = C.ks[i];
    w[8] = pos; w[9] = r; w[10] = 0x8000000000000000ull;
    w[11] = w[12] = w[13] = w[14] = 0;
    w[15] = 80 * 8;
    sha::iv(s);
    cmp(s, w);
#pragma unroll
    for (int i = 0; i < 8; ++i) Y[8 * r + i] = s[i];
  }
  mask_smem(T, Y, m);
  u64 w[16];
  cpy8(c, C.mid_salt);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = Y[i];
  cmp(c, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = Y[16 + i];
  w[8] = 0x8000000000000000ull;
#pragma unroll
  for (int i = 9; i < 15; ++i) w[i] = 0;
  w[15] = (128 + 192) * 8;
  cmp(c, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) my[i] = (u64)bswap32(m[2 * i]) << 32 | bswap32(m[2 * i + 1]);
}
DEV void row3(const uint8_t* __restrict__ p, u64* x) {
  cpy8(x, C.mid_P);
#pragma unroll 1
  for (int b = 0; b < NBLK; ++b) {
    u64 W[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 v = ldg16(p + 128 * b + 16 * c);
      W[2 * c] = be64(v.x, v.y);
      W[2 * c + 1] = be64(v.z, v.w);
    }
    cmp(x, W);
  }
  cmp_kw(x, C.kw_xpad);
}

template <bool U>
DEV void node_any(int level, u64 index, const u64* L, const u64* R, u64* out) {
  if constexpr (U) node3(level, index, L, R, out);
  else node(0, level, index, L, R, out);
}

// --- v2: the salt fused into the leaf, the tree as wide multi-level launches and one fold-and-climb launch ----------------

// y_r (3 compressions from round 8), M_key y (the nibble table in shared memory) and c = SHA-512(salt_prefix || y)
DEV void salt_regs(u64 pos, const uint4* T, u64* my, u64* c) {
  u64 Y[24];
  u32 m[16];
  salt_prf(pos, Y);
  mask_smem(T, Y, m);
  salt_digest(Y, c);
#pragma unroll
  for (int i = 0; i < 8; ++i) my[i] = (u64)bswap32(m[2 * i]) << 32 | bswap32(m[2 * i + 1]);
}

// Nodes at levels d0 + 1 .. depth above `o`, node i of level d0 (blockDim.x = min(width >> d0, TPB) threads a CTA, one a
// node): the warp folds its own aligned subtree through shuffles, then lane 0 climbs; at each level above, the second of
// two sibling subtrees to finish (an atomic count a parent) folds the parent and goes on.
template <bool U>
DEV void fold_climb(u64* o, u64 i, int d0, u64 width, int depth, u64* levels, u32* ctr) {
  const u32 lane = threadIdx.x & 31, wsz = blockDim.x < 32 ? blockDim.x : 32;
  const unsigned mask = wsz == 32 ? 0xFFFFFFFFu : (1u << wsz) - 1;
  int L = 0;
  while ((1u << L) < wsz) ++L;
  for (int l = 1; l <= L; ++l) {
    u64 p[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) p[k] = __shfl_xor_sync(mask, o[k], 1 << (l - 1));
    if ((lane & ((1u << l) - 1)) == 0) {
      u64 r[8];
      node_any<U>(d0 + l - 1, i >> l, o, p, r);
      cpy8(o, r);
      st8(level_at(levels, width, d0 + l) + 8 * (i >> l), o);
    }
  }
  if (lane != 0 || d0 + L >= depth) return;
  // ctr: one count a node above level d0 + L, level l+1's at (width >> (d0 + L)) - (width >> l)
  const u64 nL = width >> (d0 + L);
  u64 idx = i >> L;
  for (int l = d0 + L; l < depth; ++l) {
    const u64 p = idx >> 1;
    u32* c = ctr + (nL - (width >> l)) + p;
    __threadfence();
    if (atomicAdd(c, 1u) == 0) return;
    *c = 0;
    __threadfence();
    u64 sib[8], r[8];
    ld8cg(level_at(levels, width, l) + 8 * (idx ^ 1), sib);
    if (idx & 1) node_any<U>(l, p, sib, o, r);
    else node_any<U>(l, p, o, sib, r);
    st8(level_at(levels, width, l + 1) + 8 * p, r);
    cpy8(o, r);
    idx = p;
  }
}

// The whole leaf a thread (blockDim.x = min(width, TPB)): salt, x over the 4 KiB row, h, both frame-leaf blocks; a pad
// for n <= pos < width. ORDER: 0 salt then row, 1 row then salt, 2 by warp parity (the mask's shared-memory traffic
// beside the other warps' compressions). FOLD: then fold_climb to the root (one launch for the whole commitment).
template <int TPB, int MINB, int ORDER, bool FOLD>
__global__ void __launch_bounds__(TPB, MINB)
    sr_leaf(const uint8_t* __restrict__ rows, const uint4* __restrict__ Tn, u64 n, u64 width, int depth, u64* levels, u32* ctr,
            u64* __restrict__ bc) {
  __shared__ uint4 T[512];
  for (int i = threadIdx.x; i < 512; i += blockDim.x) T[i] = Tn[i];
  __syncthreads();
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 o[8];
  if (pos < n) {
    const bool salt_first = ORDER == 0 || (ORDER == 2 && !((threadIdx.x >> 5) & 1));
    u64 x[8], my[8], c[8];
    if (salt_first) {
      salt_regs(pos, T, my, c);
      row4k<false, FULL>(rows + pos * LB, x);
    } else {
      row4k<false, FULL>(rows + pos * LB, x);
      salt_regs(pos, T, my, c);
    }
    u64 B[16], h[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { B[k] = x[k] ^ my[k]; B[8 + k] = c[k]; }
    if (bc) {
      st8(bc + 16 * pos, B);
      st8(bc + 16 * pos + 8, B + 8);
    }
    cpy8(h, C.mid_leaf);
    sha::compress(h, B);
    sha::compress_kw(h, C.kw_leafpad);
    frame_first(0, pos, o);
    frame_second<0>(0, pos, h, o);
  } else {
    pad(0, pos, o);
  }
  st8(levels + 8 * pos, o);
  if constexpr (FOLD) fold_climb<false>(o, pos, 0, width, depth, levels, ctr);
}

// node idx of level d + K from level d's nodes, every node between stored
template <int K, bool U>
DEV void sub_rec(u64* levels, u64 width, int d, u64 idx, u64* out) {
  if constexpr (K == 0) {
    ld8(level_at(levels, width, d) + 8 * idx, out);
  } else {
    u64 a[8], b[8];
    sub_rec<K - 1, U>(levels, width, d, 2 * idx, a);
    sub_rec<K - 1, U>(levels, width, d, 2 * idx + 1, b);
    node_any<U>(d + K - 1, idx, a, b, out);
    st8(level_at(levels, width, d + K) + 8 * idx, out);
  }
}
template <int K, bool U>
__global__ void __launch_bounds__(256) sr_sub(u64* levels, u64 width, int d) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width >> (d + K)) return;
  u64 o[8];
  sub_rec<K, U>(levels, width, d, i, o);
}
template <bool U>
__global__ void __launch_bounds__(256) sr_tree(u64* levels, u64 width, int depth, int d0, u32* ctr) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 o[8];
  ld8(level_at(levels, width, d0) + 8 * i, o);
  fold_climb<U>(o, i, d0, width, depth, levels, ctr);
}

template <int TPB, int MINB, int ORDER, bool FOLD>
__global__ void __launch_bounds__(TPB, MINB)
    sr_leaf3(const uint8_t* __restrict__ rows, const uint4* __restrict__ Tn, u64 n, u64 width, int depth, u64* levels, u32* ctr,
             u64* __restrict__ bc) {
  __shared__ uint4 T[512];
  for (int i = threadIdx.x; i < 512; i += blockDim.x) T[i] = Tn[i];
  __syncthreads();
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 o[8];
  if (pos < n) {
    const bool salt_first = ORDER == 0 || (ORDER == 2 && !((threadIdx.x >> 5) & 1));
    u64 x[8], my[8], c[8];
    if (salt_first) {
      salt3(pos, T, my, c);
      row3(rows + pos * LB, x);
    } else {
      row3(rows + pos * LB, x);
      salt3(pos, T, my, c);
    }
    u64 B[16], h[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { B[k] = x[k] ^ my[k]; B[8 + k] = c[k]; }
    if (bc) {
      st8(bc + 16 * pos, B);
      st8(bc + 16 * pos + 8, B + 8);
    }
    cpy8(h, C.mid_leaf);
    cmp(h, B);
    cmp_kw(h, C.kw_leafpad);
    f_first3(pos, o);
    f_second3(pos, h, o);
  } else {
    pad3(pos, o);
  }
  st8(levels + 8 * pos, o);
  if constexpr (FOLD) fold_climb<true>(o, pos, 0, width, depth, levels, ctr);
}

// v3s: v1's split (sr_salt ahead, nothing of it on the data's path) with v3's data kernel: S[pos] and F[pos] from the salt
// kernel, the pads already in level 0
template <bool FOLD>
__global__ void __launch_bounds__(256, 2)
    sr_data3(const uint8_t* __restrict__ rows, u64 n, u64 width, int depth, const u64* __restrict__ S, const u64* __restrict__ F,
             u64* levels, u32* ctr, u64* __restrict__ bc) {
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 o[8];
  if (pos < n) {
    u64 x[8], B[16], h[8];
    row3(rows + pos * LB, x);
    ld8(S + 16 * pos, B);
    ld8(S + 16 * pos + 8, B + 8);
#pragma unroll
    for (int k = 0; k < 8; ++k) B[k] ^= x[k];
    if (bc) {
      st8(bc + 16 * pos, B);
      st8(bc + 16 * pos + 8, B + 8);
    }
    cpy8(h, C.mid_leaf);
    cmp(h, B);
    cmp_kw(h, C.kw_leafpad);
    ld8(F + 8 * pos, o);
    f_second3(pos, h, o);
    st8(levels + 8 * pos, o);
  } else {
    ld8(levels + 8 * pos, o);
  }
  if constexpr (FOLD) fold_climb<true>(o, pos, 0, width, depth, levels, ctr);
}

struct Mine {
  u64 n = 0, width = 0;
  int depth = 0;
  u64 *levels = nullptr, *S = nullptr, *F = nullptr;
  u32* ctr = nullptr;
  uint4* Tn = nullptr;
  // v2 launches: the leaf kernel; K = 3 launches from level d while (width >> (d + 3)) >= minw, at most `subs` of them;
  // then sr_tree from there. ev: an event after each launch.
  template <int ORDER, bool FOLD, bool U = false>
  void v2(const uint8_t* rows, cudaStream_t st, int subs, u64 minw, std::vector<cudaEvent_t>* ev = nullptr,
          std::vector<std::string>* names = nullptr, u64* bc = nullptr) {
    auto mark = [&](std::string nm) {
      if (ev) CK(cudaEventRecord((*ev)[names->size()], st));
      if (names) names->push_back(nm);
    };
    const u32 cnt = (u32)std::min<u64>(width, 256);
    if constexpr (ORDER == 3) {   // v3s: the salt kernel, then the data kernel
      salts(st);
      sr_data3<FOLD><<<(unsigned)(width / cnt), cnt, 0, st>>>(rows, n, width, depth, S, F, levels, ctr, bc);
    } else if constexpr (U) sr_leaf3<256, 2, ORDER, FOLD><<<(unsigned)(width / cnt), cnt, 0, st>>>(rows, Tn, n, width, depth, levels, ctr, bc);
    else sr_leaf<256, 2, ORDER, FOLD><<<(unsigned)(width / cnt), cnt, 0, st>>>(rows, Tn, n, width, depth, levels, ctr, bc);
    mark("leaf");
    if (FOLD) return;
    int d = 0;
    for (int s = 0; s < subs && d + 3 <= depth && (width >> (d + 3)) >= minw; ++s, d += 3) {
      const u64 m = width >> (d + 3);
      sr_sub<3, U><<<(unsigned)((m + 255) / 256), 256, 0, st>>>(levels, width, d);
      mark("sub3_from_" + std::to_string(d));
    }
    if (d < depth) {
      const u64 m = width >> d;
      const u32 c = (u32)std::min<u64>(m, 256);
      sr_tree<U><<<(unsigned)(m / c), c, 0, st>>>(levels, width, depth, d, ctr);
      mark("tree_from_" + std::to_string(d));
    }
  }
  void init(const Params& p, u64 n_) {
    n = n_; depth = depth_of(n); width = 1ull << depth;
    CK(cudaMalloc(&levels, 2 * width * 64));
    CK(cudaMalloc(&S, n * 128));
    CK(cudaMalloc(&F, n * 64));
    CK(cudaMalloc(&ctr, (width + 1) * 4));
    CK(cudaMemset(ctr, 0, (width + 1) * 4));
    std::vector<uint8_t> tn(8192);
    for (int h = 0; h < 2; ++h)
      for (int c = 0; c < 16; ++c)
        for (int u = 0; u < 16; ++u) memcpy(&tn[4096 * h + 256 * c + 16 * u], &p.T[256 * (h ? 16 * u : u) + 16 * c], 16);
    CK(cudaMalloc(&Tn, tn.size()));
    CK(cudaMemcpy(Tn, tn.data(), tn.size(), cudaMemcpyHostToDevice));
  }
  template <bool MASK = true>
  void salts(cudaStream_t st) {
    constexpr int T = 128;
    sr_salt<T, MASK><<<(unsigned)((width + T - 1) / T), T, 0, st>>>(Tn, n, width, S, F, levels);
  }
  template <int TPB, int MINB, bool PF, int DG = FULL>
  void data(const uint8_t* rows, cudaStream_t st, u64* bc = nullptr) {
    const u32 cnt = (u32)std::min<u64>(width, TPB);
    sr_data<TPB, MINB, PF, DG><<<(unsigned)(width / cnt), cnt, 0, st>>>(rows, n, width, depth, S, F, levels, bc, ctr);
  }
  void release() { cudaFree(levels); cudaFree(S); cudaFree(F); cudaFree(ctr); cudaFree(Tn); }
};
}  // namespace sr

// --- circuits' rows-sha512 kernels (native_rows.cu, its extern "C" launchers) ---------------------------------------------
struct Native {
  void* h = nullptr;
  int (*params_bytes)() = nullptr;
  int (*build)(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, const char*, int, uint32_t, void*) = nullptr;
  int (*leaves)(const void*, const uint8_t*, uint64_t, uint64_t, const uint8_t*, const uint8_t*, uint64_t, uint64_t, uint64_t, uint8_t*,
                uint8_t*, uint8_t*, cudaStream_t) = nullptr;
  int (*level)(const void*, const uint8_t*, uint8_t*, uint64_t, int, uint64_t, cudaStream_t) = nullptr;
  std::vector<uint8_t> P;
  uint8_t* lv = nullptr;
  uint4* Tn = nullptr;
  u64 n = 0, width = 0;
  bool load(const char* so) {
    h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen %s: %s\n", so, dlerror()); return false; }
    params_bytes = (int (*)())dlsym(h, "verity_rows_params_bytes");
    build = (decltype(build))dlsym(h, "verity_rows_build");
    leaves = (decltype(leaves))dlsym(h, "verity_rows_leaves_launch");
    level = (decltype(level))dlsym(h, "verity_rows_level_launch");
    return params_bytes && build && leaves && level;
  }
  void init(const Params& p, u64 n_, const uint4* tn) {
    n = n_; width = 1ull << depth_of(n);
    P.resize(params_bytes());
    static const char schema[] = "hm96-sha512/row/v2";
    if (build(p.domain_id, p.ks, p.P, p.salt_prefix, p.leaf_prefix, schema, 18, sr::LB, P.data())) { fprintf(stderr, "native build\n"); exit(2); }
    CK(cudaMalloc(&lv, (2 * width) * 64));
    Tn = const_cast<uint4*>(tn);
  }
  // rows_floor's schedule: the leaf kernel over the padded width, then a launch a level
  void run(const uint8_t* rows, cudaStream_t st, std::vector<cudaEvent_t>* ev = nullptr) {
    if (leaves(P.data(), rows, sr::LB, n, nullptr, (const uint8_t*)Tn, 0, width, n, lv, nullptr, nullptr, st)) { fprintf(stderr, "native leaves\n"); exit(3); }
    if (ev) CK(cudaEventRecord((*ev)[0], st));
    u64 off = 0, m = width;
    for (int l = 0; m > 1; ++l) {
      if (level(P.data(), lv + 64 * off, lv + 64 * (off + m), m / 2, l, 0, st)) { fprintf(stderr, "native level\n"); exit(3); }
      if (ev) CK(cudaEventRecord((*ev)[1 + l], st));
      off += m; m /= 2;
    }
  }
};

// --- device helpers ------------------------------------------------------------------------------------------------
__global__ void fill(uint8_t* p, size_t n, u32 seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    u32 x = (u32)i * 0x9E3779B1u ^ seed ^ (u32)(i >> 32) * 0x85EBCA77u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = (uint8_t)x;
  }
}
// differing 64-byte digests between a (commit_gpu's word layout) and b (BSWAP: digest byte order, native_rows')
template <bool BSWAP>
__global__ void diff_kernel(const u64* a, const u64* b, u64 words, unsigned long long* bad) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (u64)gridDim.x * blockDim.x) {
    u64 y = b[i];
    if (BSWAP) y = (u64)bswap32((u32)y) << 32 | bswap32((u32)(y >> 32));
    if (a[i] != y) atomicAdd(bad, 1ull);
  }
}
// the raw rate: commit_gpu's sha_rate (each thread chains compressions, its block from the last state)
__global__ void sha_rate(u64* io, int reps) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 st[8], w[16];
  sha::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sha::compress(st, w);
  }
  if (st[0] == 0x0123456789abcdefull) io[t] = st[1];
}
__global__ void spin(long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
}

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  unsigned sm() { unsigned c = 0; if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c); return c; }
};
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
static void jlist(const std::vector<double>& v, const char* fmt = "%.4f") {
  printf("[");
  for (size_t i = 0; i < v.size(); ++i) { if (i) printf(", "); printf(fmt, v[i]); }
  printf("]");
}

// --- check: commit_gpu's interface, rows mode at 4 KiB, derived salts ---------------------------------------------------
static int check(int argc, char** argv) {
  if (argc < 10) { fprintf(stderr, "check PARAMS OUT rows N 32768 4096 ROWS prf\n"); return 2; }
  const Params p = read_params(argv[2]);
  const std::string out = argv[3];
  const u64 n = strtoull(argv[5], nullptr, 10), lbits = strtoull(argv[6], nullptr, 10), stride = strtoull(argv[7], nullptr, 10);
  if (strcmp(argv[4], "rows") || lbits != 8 * sr::LB || stride != sr::LB || strcmp(argv[9], "prf")) {
    fprintf(stderr, "sharaw commits contiguous 4 KiB rows with derived salts only\n");
    return 2;
  }
  auto rows = readf(argv[8]);
  if (rows.size() < n * sr::LB) { fprintf(stderr, "rows: short\n"); return 2; }
  uint8_t* d;
  CK(cudaMalloc(&d, n * sr::LB));
  CK(cudaMemcpy(d, rows.data(), n * sr::LB, cudaMemcpyHostToDevice));
  Committer cm;   // only for C and CD[0]
  cm.init(p, 1, sr::LB, false);
  sr::Mine me;
  me.init(p, n);
  u64* bc;
  CK(cudaMalloc(&bc, n * 128));
  const char* form = getenv("SHARAW_FORM");
  const int f = form ? atoi(form) : 0;
  const char* mw = getenv("SHARAW_MINW");
  const u64 minw = mw ? strtoull(mw, nullptr, 10) : 1;   // 1: the K = 3 launches run even on the small cases
  for (int rep = 0; rep < 2; ++rep) {   // twice: the climb's counts must come back to zero
    switch (f) {
      case 1: me.v2<0, false>(d, 0, 2, minw, nullptr, nullptr, bc); break;
      case 2: me.v2<0, false, true>(d, 0, 2, minw, nullptr, nullptr, bc); break;
      case 3: me.v2<2, false, true>(d, 0, 1, minw, nullptr, nullptr, bc); break;
      case 4: me.v2<1, true, true>(d, 0, 0, minw, nullptr, nullptr, bc); break;
      case 5: me.v2<2, true, true>(d, 0, 0, minw, nullptr, nullptr, bc); break;
      case 6: me.v2<0, false, true>(d, 0, 0, minw, nullptr, nullptr, bc); break;
      case 7: me.v2<3, false, true>(d, 0, 2, minw, nullptr, nullptr, bc); break;
      case 8: me.v2<3, true, true>(d, 0, 0, minw, nullptr, nullptr, bc); break;
      default: me.salts(0); me.data<256, 2, false>(d, 0, bc); break;
    }
  }
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<u64> lv(16 * me.width), bcv(16 * n);
  CK(cudaMemcpy(lv.data(), me.levels, lv.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(bcv.data(), bc, bcv.size() * 8, cudaMemcpyDeviceToHost));
  u64 used = 0;
  for (int dd = 0; dd <= me.depth; ++dd) used += me.width >> dd;
  lv.resize(8 * used);
  auto to_bytes = [](const std::vector<u64>& v) {
    std::vector<uint8_t> b(v.size() * 8);
    for (size_t i = 0; i < v.size(); ++i) sha::put_be64(b.data() + 8 * i, v[i]);
    return b;
  };
  const auto lb = to_bytes(lv), bb = to_bytes(bcv);
  writef(out + "/levels.bin", lb.data(), lb.size());
  writef(out + "/bc.bin", bb.data(), bb.size());
  printf("{\"check\": \"rows\", \"n\": %llu, \"depth\": %d, \"form\": %d}\n", (unsigned long long)n, me.depth, f);
  return 0;
}

// --- bench ----------------------------------------------------------------------------------------------------------
struct Arm {
  std::string name;
  bool verified;                         // its levels were compared in this process
  std::function<void(cudaStream_t)> run;
  double comps;                          // SHA-512 compressions a call (the work it does)
};

static int bench(int argc, char** argv, bool nsys_mode) {
  if (argc < 5) { fprintf(stderr, "bench PARAMS N REPS [NATIVE_SO] | nsys PARAMS N ARM [NATIVE_SO]\n"); return 2; }
  const Params p = read_params(argv[2]);
  const u64 n = strtoull(argv[3], nullptr, 10);
  const int reps = nsys_mode ? 2 : atoi(argv[4]);
  const std::string only = nsys_mode ? argv[4] : "";
  const char* so = argc >= 6 ? argv[5] : nullptr;
  const u64 bytes = n * sr::LB;
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  uint8_t* src;
  CK(cudaMalloc(&src, bytes + 64));
  fill<<<4096, 256>>>(src, bytes + 64, 7u);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Committer cm;
  cm.init(p, n, sr::LB, false);
  sr::Mine me;
  me.init(p, n);
  Native nat;
  const bool have_nat = so && nat.load(so);
  if (have_nat) nat.init(p, n, me.Tn);
  const u64 width = cm.width;
  const int depth = cm.depth;
  // compressions a call: x (33), salt (3) and c (2), h (2), the frame leaf (2) a leaf; a pad (2); a node (2)
  const double c_leaf = 33 + 5 + 4, comps = n * c_leaf + 2.0 * (width - n) + 2.0 * (width - 1);
  const double c_salt = n * 5.0 + 2.0 * (width - n) + n * 1.0;   // sr_salt: y, c, F's block; the pads
  auto cg = [&](cudaStream_t s) {
    cm.salts<true>(nullptr, s);
    cm.leaves<RowBytes>(RowBytes{src, sr::LB, sr::LB}, s);
    cm.nodes(s);
  };
  std::vector<Arm> arms;
  arms.push_back({"commit_gpu", true, cg, comps});
  if (have_nat) arms.push_back({"native_rows", true, [&](cudaStream_t s) { nat.run(src, s); }, comps});
  arms.push_back({"sharaw_v1", true, [&](cudaStream_t s) { me.salts(s); me.data<256, 2, false>(src, s); }, comps});
  const u64 MINW = 4096;   // a K = 3 launch only while it has at least this many threads
#define V2(NAME, ORDER, FOLD, SUBS, U) \
  arms.push_back({NAME, true, [&](cudaStream_t s) { me.v2<ORDER, FOLD, U>(src, s, SUBS, MINW); }, comps})
  V2("v2_salt_s2", 0, false, 2, false);
  V2("v3_salt_s2", 0, false, 2, true);
  V2("v3_par_s2", 2, false, 2, true);
  V2("v3_row_s2", 1, false, 2, true);
  V2("v3_salt_s1", 0, false, 1, true);
  V2("v3_salt_fold", 0, true, 0, true);
  V2("v3_par_fold", 2, true, 0, true);
  V2("v3s_s2", 3, false, 2, true);
  V2("v3s_fold", 3, true, 0, true);
#undef V2
  u64* io;
  const int rt = 256, rb = prop.multiProcessorCount * 8, chain = 256;
  CK(cudaMalloc(&io, (size_t)rt * rb * 8));
  arms.push_back({"raw", false, [&](cudaStream_t s) { sha_rate<<<rb, rt, 0, s>>>(io, chain); }, (double)rt * rb * chain});
  // diagnostic forms of this file's committer (timing only)
  std::vector<Arm> diags = {
      {"salt_only", false, [&](cudaStream_t s) { me.salts<true>(s); }, c_salt},
      {"salt_nomask", false, [&](cudaStream_t s) { me.salts<false>(s); }, c_salt},
      {"data_only", false, [&](cudaStream_t s) { me.data<256, 2, false>(src, s); }, comps - c_salt},
      {"data_noload", false, [&](cudaStream_t s) { me.data<256, 2, false, sr::NOLOAD>(src, s); }, comps - c_salt},
      {"data_rowonly", false, [&](cudaStream_t s) { me.data<256, 2, false, sr::ROWONLY>(src, s); }, n * 33.0},
      {"data_nofold", false, [&](cudaStream_t s) { me.data<256, 2, false, sr::NOFOLD>(src, s); }, n * 36.0},
  };

  // 1. the same commitment from every committer, before anything is timed
  unsigned long long* dbad;
  CK(cudaMalloc(&dbad, sizeof *dbad));
  const u64 words = 8 * (2 * width - 1);
  u64 root_cg[8] = {};
  if (!nsys_mode) {
    cg(st);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    CK(cudaMemcpy(root_cg, cm.lv[depth], 64, cudaMemcpyDeviceToHost));
  }
  bool all_ok = true;
  printf("{\"device\": \"%s\", \"sms\": %d, \"n\": %llu, \"width\": %llu, \"depth\": %d, \"bytes\": %llu, \"compressions\": %.0f, "
         "\"root\": \"", prop.name, prop.multiProcessorCount, (unsigned long long)n, (unsigned long long)width, depth,
         (unsigned long long)bytes, comps);
  for (int i = 0; i < 8; ++i) printf("%016llx", (unsigned long long)root_cg[i]);
  printf("\"}\n");
  for (auto& a : arms) {
    if (nsys_mode || !a.verified || a.name == "commit_gpu") continue;
    if (a.name == "native_rows") CK(cudaMemset(nat.lv, 0, 2 * width * 64));
    else CK(cudaMemset(me.levels, 0xA5, 2 * width * 64));
    for (int k = 0; k < 2; ++k) a.run(st);   // twice: the second call must not depend on the first's leftovers
    CK(cudaMemsetAsync(dbad, 0, sizeof *dbad, st));
    if (a.name == "native_rows") diff_kernel<true><<<1024, 256, 0, st>>>(cm.levels, (const u64*)nat.lv, words, dbad);
    else diff_kernel<false><<<1024, 256, 0, st>>>(cm.levels, me.levels, words, dbad);
    unsigned long long bad = 0;
    CK(cudaMemcpyAsync(&bad, dbad, sizeof bad, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    printf("{\"same_levels\": \"%s\", \"against\": \"commit_gpu\", \"words\": %llu, \"differing\": %llu}\n", a.name.c_str(),
           (unsigned long long)words, bad);
    if (bad) { a.verified = false; all_ok = false; }
  }
  fflush(stdout);
  if (!all_ok) { fprintf(stderr, "a committer's levels differ from commit_gpu's: nothing is timed\n"); return 1; }

  Nv nv;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto ms = [](cudaEvent_t a, cudaEvent_t b) { float t; CK(cudaEventElapsedTime(&t, a, b)); return (double)t; };
  const long long SPIN = 1000000;   // ~0.5 ms: what follows is enqueued while it runs
  auto timed = [&](Arm& a, unsigned& mhz) {
    spin<<<1, 1, 0, st>>>(SPIN);
    CK(cudaEventRecord(e0, st));
    a.run(st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    mhz = nv.sm();
    return ms(e0, e1);
  };
  if (nsys_mode) {
    for (auto& a : arms) if (a.name == only) for (int r = 0; r < reps; ++r) { unsigned m; timed(a, m); }
    for (auto& a : diags) if (a.name == only) for (int r = 0; r < reps; ++r) { unsigned m; timed(a, m); }
    CK(cudaDeviceSynchronize());
    return 0;
  }
  // 2. warm-up, then every arm once a rep, in turn
  for (auto& a : arms) { unsigned m; timed(a, m); timed(a, m); }
  std::vector<std::vector<double>> t(arms.size()), mz(arms.size());
  for (int r = 0; r < reps; ++r)
    for (size_t i = 0; i < arms.size(); ++i) {
      unsigned m;
      t[i].push_back(timed(arms[i], m));
      mz[i].push_back(m);
    }
  size_t iraw = 0;
  for (size_t i = 0; i < arms.size(); ++i) if (arms[i].name == "raw") iraw = i;
  std::vector<double> raw_rate;   // per rep, G compressions/s
  for (double x : t[iraw]) raw_rate.push_back(arms[iraw].comps / x * 1e-6);
  for (size_t i = 0; i < arms.size(); ++i) {
    const auto& a = arms[i];
    std::vector<double> pct;
    for (int r = 0; r < reps; ++r) pct.push_back(100.0 * (a.comps / t[i][r] * 1e-6) / raw_rate[r]);
    const double tm = median(t[i]);
    printf("{\"arm\": \"%s\", \"verified\": %s, \"ms\": %.4f, \"gb_per_s\": %.2f, \"gcomp_per_s\": %.4f, \"pct_raw\": %.2f, "
           "\"mhz\": %.0f, \"ms_all\": ", a.name.c_str(), a.verified ? "true" : "false", tm, a.name == "raw" ? 0.0 : bytes / tm * 1e-6,
           a.comps / tm * 1e-6, median(pct), median(mz[i]));
    jlist(t[i]);
    printf(", \"mhz_all\": ");
    jlist(mz[i], "%.0f");
    printf(", \"pct_raw_all\": ");
    jlist(pct, "%.2f");
    printf("}\n");
  }
  fflush(stdout);
  // 3. the time split: commit_gpu's and native_rows' kernels one by one (an event after each launch), this file's diagnostic
  // forms each timed alone, interleaved with the full form
  {
    std::vector<cudaEvent_t> ev(64);
    for (auto& x : ev) CK(cudaEventCreate(&x));
    std::vector<std::vector<double>> seg(64);
    int nseg = 0;
    std::vector<std::string> names;
    for (int r = 0; r < reps; ++r) {
      spin<<<1, 1, 0, st>>>(SPIN);
      int k = 0;
      CK(cudaEventRecord(ev[k++], st));
      cm.salts<true>(nullptr, st);
      CK(cudaEventRecord(ev[k++], st));
      cm.leaves<RowBytes>(RowBytes{src, sr::LB, sr::LB}, st);
      CK(cudaEventRecord(ev[k++], st));
      int d = 0;
      std::vector<std::string> nm = {"salt_kernel", "leaf_kernel"};
      for (; d < depth && (width >> d) > 2 * NODE_THREADS; ++d) {
        const u64 n_out = width >> (d + 1);
        const unsigned th = n_out >= (1u << 16) ? NODE_THREADS : 64;
        level_kernel<<<(unsigned)((n_out + th - 1) / th), th, 0, st>>>(0, cm.lv[d], cm.lv[d + 1], width >> d, d);
        CK(cudaEventRecord(ev[k++], st));
        nm.push_back("level_" + std::to_string(d + 1));
      }
      if (d < depth) {
        LevelPtrs o{};
        for (int l = 0; l < depth - d; ++l) o.p[l] = cm.lv[d + 1 + l];
        node_kernel<<<1, NODE_THREADS, 0, st>>>(0, cm.lv[d], o, width >> d, d, depth - d);
        CK(cudaEventRecord(ev[k++], st));
        nm.push_back("node_kernel_" + std::to_string(d + 1) + "_" + std::to_string(depth));
      }
      CK(cudaEventSynchronize(ev[k - 1]));
      for (int j = 1; j < k; ++j) seg[j - 1].push_back(ms(ev[j - 1], ev[j]));
      nseg = k - 1;
      names = nm;
    }
    printf("{\"split\": \"commit_gpu\", \"parts\": {");
    for (int j = 0; j < nseg; ++j) printf("%s\"%s\": %.4f", j ? ", " : "", names[j].c_str(), median(seg[j]));
    printf("}}\n");
    if (have_nat) {
      for (auto& s : seg) s.clear();
      for (int r = 0; r < reps; ++r) {
        spin<<<1, 1, 0, st>>>(SPIN);
        CK(cudaEventRecord(e0, st));
        std::vector<cudaEvent_t> sub(ev.begin(), ev.begin() + depth + 1);
        nat.run(src, st, &sub);
        CK(cudaEventSynchronize(sub[depth]));
        seg[0].push_back(ms(e0, sub[0]));
        for (int l = 1; l <= depth; ++l) seg[l].push_back(ms(sub[l - 1], sub[l]));
      }
      printf("{\"split\": \"native_rows\", \"parts\": {\"rows_leaf_kernel\": %.4f", median(seg[0]));
      for (int l = 1; l <= depth; ++l) printf(", \"level_%d\": %.4f", l, median(seg[l]));
      printf("}}\n");
    }
    {
      std::vector<std::string> nm;
      for (auto& x : seg) x.clear();
      for (int r = 0; r < reps; ++r) {
        spin<<<1, 1, 0, st>>>(SPIN);
        CK(cudaEventRecord(e0, st));
        nm.clear();
        me.v2<0, false, true>(src, st, 2, MINW, &ev, &nm);
        CK(cudaEventSynchronize(ev[nm.size() - 1]));
        seg[0].push_back(ms(e0, ev[0]));
        for (size_t j = 1; j < nm.size(); ++j) seg[j].push_back(ms(ev[j - 1], ev[j]));
      }
      printf("{\"split\": \"v3_salt_s2\", \"parts\": {");
      for (size_t j = 0; j < nm.size(); ++j) printf("%s\"%s\": %.4f", j ? ", " : "", nm[j].c_str(), median(seg[j]));
      printf("}}\n");
    }
    std::vector<Arm> sp = diags;
    sp.insert(sp.begin(), arms[0]);
    for (auto& a : arms) if (a.name == "sharaw_v1") sp.push_back(a);
    std::vector<std::vector<double>> td(sp.size()), md(sp.size());
    for (auto& a : sp) { unsigned m; timed(a, m); }
    for (int r = 0; r < reps; ++r)
      for (size_t i = 0; i < sp.size(); ++i) { unsigned m; td[i].push_back(timed(sp[i], m)); md[i].push_back(m); }
    for (size_t i = 0; i < sp.size(); ++i)
      printf("{\"diag\": \"%s\", \"ms\": %.4f, \"gcomp_per_s\": %.4f, \"mhz\": %.0f}\n", sp[i].name.c_str(), median(td[i]),
             sp[i].comps / median(td[i]) * 1e-6, median(md[i]));
  }
  fflush(stdout);
  return 0;
}

static int selftest() {
  uint8_t d[64];
  sha::digest((const uint8_t*)"abc", 3, d);
  const bool ok = d[0] == 0xdd && d[1] == 0xaf && d[63] == 0x9f;
  printf("{\"selftest\": %s}\n", ok ? "true" : "false");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "selftest")) return selftest();
  if (argc >= 2 && !strcmp(argv[1], "check")) return check(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "bench")) return bench(argc, argv, false);
  if (argc >= 2 && !strcmp(argv[1], "nsys")) return bench(argc, argv, true);
  fprintf(stderr, "usage: sharaw selftest | check ... | bench PARAMS N REPS [NATIVE_SO] | nsys PARAMS N ARM [NATIVE_SO]\n");
  return 2;
}
