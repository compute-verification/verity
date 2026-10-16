// A GPU committer for hm96-sha512/row/v2 rows in frame-v3-sha512 trees, bit for bit the reference's
// (verity.ml.commitments: hm96.SHA512, rowleaf.sha512_row_v2_*, merkle.CommitmentDomain and tree_levels; reference.py here).
//
// Per position of a domain of n leaves (rank = position, the tree padded to width = 2^depth):
//   x    = SHA-512(P(l) || row)                         the row's sha512/row/v2 digest
//   y    = salt (192 bytes): given, or y_r = SHA-512(Ks || u64be(pos) || u64be(r)), r = 0, 1, 2 (reference.prf_salt)
//   c    = SHA-512(salt_prefix || y);  b = x ^ M_key y;  h = SHA-512(leaf_prefix(key) || b || c)
//   leaf = SHA-512(frame || "leaf" || domain id || uint(pos) || uint(pos) || "hm96-sha512/row/v2" || h)   (framed parts)
//   pad  = SHA-512(frame || "pad" || domain id || uint(pos)) for n <= pos < width
// then node(level, index, left, right) up the tree. Every 64-byte digest is kept as 8 u64 words, each the big-endian
// value of its 8 bytes (the host swaps them back).
//
// M_key y uses M's Hankel form: column j of M is bits j .. j+511 of the key, so the reference's byte table (p, v) is bytes
// [p, p+64) of T_v = xor of (key >> s) over the set bits s of v: one 64 KB table T (reference.t_table), in shared memory.
//
// Row sources: `e2e` (the e2e slice's leaf: 32 int8 of an operand as bf16 words twice, l = 1024, leaf pos at bytes
// [32 pos, 32 pos + 32)) and `rows` (n rows of l bits at a byte stride; an int8 operand's own memory is rows of 8 L bits).
//
//   commit_gpu selftest
//   commit_gpu check PARAMS OUT e2e|rows N LBITS STRIDE ROWS SALTS|prf     (writes OUT/levels.bin, OUT/bc.bin)
//   commit_gpu bench PARAMS e2e|rows N LBITS REPS [GEMM_M GEMM_N GEMM_K]     (JSON lines on stdout)
//   commit_gpu sha REPS                                                      (raw compression rate)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a commit_gpu.cu -lcublasLt -lnvidia-ml [-DSHA_ADD=0|1] [-DSHA_ROT=0..15]
// (-DCOMMIT_GPU_DIAG_NOMASK: a timing-only build that skips M_key y, to measure its share; it refuses `check`)
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sha512.cuh"

using u64 = uint64_t;
using u32 = uint32_t;
#define DEV __device__ __forceinline__

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
      exit(3);                                                                                  \
    }                                                                                           \
  } while (0)
#define CL(x)                                                                         \
  do {                                                                                \
    cublasStatus_t s_ = (x);                                                          \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                \
      fprintf(stderr, "cuBLASLt %d at %s:%d: %s\n", (int)s_, __FILE__, __LINE__, #x); \
      exit(3);                                                                        \
    }                                                                                 \
  } while (0)

#ifndef LEAF_THREADS
#define LEAF_THREADS 256
#endif
constexpr int TSTRIDE = 272;                 // bytes a T row in shared memory: 256 + 16, so rows start in different bank groups
constexpr int TSMEM = 256 * TSTRIDE;          // 69,632 bytes
constexpr int NODE_THREADS = 256;             // a node pass covers 512 children a block, up to 9 levels
constexpr int MAX_NB = 4;                     // positions and node indices below 2^32
constexpr int MAX_NI = 4;

struct Consts {
  u64 mid_P[8];             // after P(l)
  u64 kw_xpad[80];          // the row digest's last block when the row fills whole blocks (all padding)
  u64 mid_salt[8];          // after the salt prefix
  u64 mid_leaf[8];          // after leaf_prefix(key)
  u64 kw_leafpad[80];       // the hm96 leaf's padding block
  u64 ks[8];                // the salt key
  u64 ks_m8[8];             // state after rounds 0..7 of a salt block (its words 0..7 are the key)
  u64 fl_tpl[MAX_NB + 1][32];   // the frame leaf's two blocks per nb = len(uint(pos)), pos and h zero
  u64 fl_m15[8];            // state after rounds 0..14 of its first block (words 0..14 don't depend on pos)
  u64 node_mid[64][8];      // per level: the chaining value after a node message's first block (it fixes only the level)
  u64 node_tpl[MAX_NI + 1][32];  // a node message's blocks 2 and 3 per ni = len(uint(index)), index and children zero
  u64 pad_tpl[MAX_NB + 1][16];   // a pad's first block per nb, rank zero
};
__constant__ Consts C;

DEV u32 bswap32(u32 x) { return __byte_perm(x, 0, 0x0123); }
DEV u64 be64(u32 lo, u32 hi) { return (u64)bswap32(lo) << 32 | bswap32(hi); }   // 8 bytes loaded as two LE u32
HD int nbytes(u64 x) { int n = 1; while (x >> (8 * n) && n < 8) ++n; return n; }

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

DEV void cpy8(u64* d, const u64* s) {
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = s[i];
}

// --- the frame: leaf, pad, node ----------------------------------------------------------------------------------

template <int NB>
DEV void frame_leaf_nb(u64 pos, const u64* h, u64* out) {
  u64 W[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) W[i] = C.fl_tpl[NB][i];
  or_be<121, NB>(W, pos);
  or_be<129 + NB, NB>(W, pos);
  or_digest<163 + 2 * NB>(W, h);
  sha::iv(out);
  sha::compress_from<15>(out, C.fl_m15, W);
  sha::compress(out, W + 16);
}
DEV void frame_leaf(u64 pos, const u64* h, u64* out) {
  switch (nbytes(pos)) {
    case 1: frame_leaf_nb<1>(pos, h, out); break;
    case 2: frame_leaf_nb<2>(pos, h, out); break;
    case 3: frame_leaf_nb<3>(pos, h, out); break;
    default: frame_leaf_nb<4>(pos, h, out); break;
  }
}

template <int NB>
DEV void pad_nb(u64 rank, u64* out) {
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = C.pad_tpl[NB][i];
  or_be<120, NB>(W, rank);
  sha::iv(out);
  sha::compress(out, W);
#pragma unroll
  for (int i = 0; i < 15; ++i) W[i] = 0;
  W[15] = (120 + NB) * 8;
  sha::compress(out, W);
}
DEV void pad(u64 rank, u64* out) {
  switch (nbytes(rank)) {
    case 1: pad_nb<1>(rank, out); break;
    case 2: pad_nb<2>(rank, out); break;
    case 3: pad_nb<3>(rank, out); break;
    default: pad_nb<4>(rank, out); break;
  }
}

template <int NI>
DEV void node_ni(int level, u64 index, const u64* L, const u64* R, u64* out) {
  u64 W[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) W[i] = C.node_tpl[NI][i];
  or_be<2, NI>(W, index);
  or_digest<10 + NI>(W, L);
  or_digest<82 + NI>(W, R);
  cpy8(out, C.node_mid[level]);
  sha::compress(out, W);
  sha::compress(out, W + 16);
}
DEV void node(int level, u64 index, const u64* L, const u64* R, u64* out) {
  switch (nbytes(index)) {
    case 1: node_ni<1>(level, index, L, R, out); break;
    case 2: node_ni<2>(level, index, L, R, out); break;
    case 3: node_ni<3>(level, index, L, R, out); break;
    default: node_ni<4>(level, index, L, R, out); break;
  }
}

// --- the leaf's parts ----------------------------------------------------------------------------------------------

DEV uint4 ldg16(const uint8_t* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// x for the e2e leaf: the 32 int8 at A[32 pos ..] as bf16 words (little-endian), twice: one data block, then P(1024)'s
// fixed padding block.
DEV void row_e2e(const uint8_t* A, u64 pos, u64* x) {
  const uint4 a0 = ldg16(A + 32 * pos), a1 = ldg16(A + 32 * pos + 16);
  const u32 aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  u64 W[16];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    u32 h[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) h[t] = __float_as_uint((float)(int8_t)(aw[q] >> (8 * t))) >> 16;
    W[q] = W[q + 8] = be64(h[0] | h[1] << 16, h[2] | h[3] << 16);
  }
  cpy8(x, C.mid_P);
  sha::compress(x, W);
  sha::compress_kw(x, C.kw_xpad);
}

// x for a row of lb bytes at `row` (16-byte aligned; reads may run up to 15 bytes past the row, inside the buffer).
DEV void row_bytes(const uint8_t* row, u32 lb, u64* x) {
  cpy8(x, C.mid_P);
  const u32 nblk = (lb + 17 + 127) / 128;
  for (u32 b = 0; b < nblk; ++b) {
    u64 W[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const u32 off = 128 * b + 16 * c;
      const uint4 v = off < lb ? ldg16(row + off) : make_uint4(0, 0, 0, 0);
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

DEV void salt_prf(u64 pos, u64* Y) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    u64 w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = C.ks[i];
    w[8] = pos; w[9] = r; w[10] = 0x8000000000000000ull;
    w[11] = w[12] = w[13] = w[14] = 0;
    w[15] = 80 * 8;
    sha::iv(Y + 8 * r);
    sha::compress_from<8>(Y + 8 * r, C.ks_m8, w);
  }
}
DEV void salt_given(const uint8_t* s, u64* Y) {
#pragma unroll
  for (int c = 0; c < 12; ++c) {
    const uint4 v = ldg16(s + 16 * c);
    Y[2 * c] = be64(v.x, v.y);
    Y[2 * c + 1] = be64(v.z, v.w);
  }
}

// c = SHA-512(salt_prefix || y)
DEV void salt_digest(const u64* Y, u64* c) {
  u64 w[16];
  cpy8(c, C.mid_salt);
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

// M_key y as 16 little-endian u32 (mask byte 4w + k is byte k of m[w]): for salt bytes p = 16 q + r, the 80 bytes
// [16 q, 16 q + 80) of T_{y_p} are xored into A_r, and A_r shifted down r bytes into m.
// T in shared memory (TS, rows TSTRIDE apart) or read from global memory through L1 (rows 256 apart).
template <bool TS>
DEV void hm96_mask(const uint8_t* T, const u64* Y, u32* m) {
  constexpr int STRIDE = TS ? TSTRIDE : 256;
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
      const uint8_t* t = T + v * STRIDE + 16 * q;
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const uint4 u = TS ? *reinterpret_cast<const uint4*>(t + 16 * i) : ldg16(t + 16 * i);
        A[4 * i] ^= u.x; A[4 * i + 1] ^= u.y; A[4 * i + 2] ^= u.z; A[4 * i + 3] ^= u.w;
      }
    }
    const int sw = r >> 2, s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= s ? __funnelshift_r(A[w + sw], A[w + sw + 1], s) : A[w + sw];
  }
}

struct RowE2E {
  const uint8_t* A;
  DEV void digest(u64 pos, u64* x) const { row_e2e(A, pos, x); }
};
struct RowBytes {
  const uint8_t* base;
  u64 stride;
  u32 lb;
  DEV void digest(u64 pos, u64* x) const { row_bytes(base + pos * stride, lb, x); }
};

// One thread a position: the tree leaf (or pad) to lv0[pos]; b || c to bc[pos] when bc is given.
template <class Row, bool PRF, bool TS>
__global__ void __launch_bounds__(LEAF_THREADS, 1)
    leaf_kernel(Row row, const uint8_t* __restrict__ salts, const uint4* __restrict__ Tg, u64 n, u64 width,
                u64* __restrict__ lv0, u64* __restrict__ bc) {
  extern __shared__ uint4 smem[];
  if constexpr (TS) {
    for (int i = threadIdx.x; i < 256 * 16; i += blockDim.x) smem[(i >> 4) * (TSTRIDE / 16) + (i & 15)] = Tg[i];
    __syncthreads();
  }
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= width) return;
  u64 out[8];
  if (pos < n) {
    u64 Y[24], c[8], x[8], B[16];
    u32 m[16];
    if constexpr (PRF) salt_prf(pos, Y);
    else salt_given(salts + 192 * pos, Y);
#ifdef COMMIT_GPU_DIAG_NOMASK
    for (int w = 0; w < 16; ++w) m[w] = (u32)Y[w];
#else
    hm96_mask<TS>(TS ? reinterpret_cast<const uint8_t*>(smem) : reinterpret_cast<const uint8_t*>(Tg), Y, m);
#endif
    salt_digest(Y, c);
    row.digest(pos, x);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      B[i] = x[i] ^ ((u64)bswap32(m[2 * i]) << 32 | bswap32(m[2 * i + 1]));
      B[8 + i] = c[i];
    }
    if (bc) {
      uint4* o = reinterpret_cast<uint4*>(bc + 16 * pos);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        o[i] = make_uint4((u32)B[2 * i], (u32)(B[2 * i] >> 32), (u32)B[2 * i + 1], (u32)(B[2 * i + 1] >> 32));
    }
    u64 h[8];
    cpy8(h, C.mid_leaf);
    sha::compress(h, B);
    sha::compress_kw(h, C.kw_leafpad);
    frame_leaf(pos, h, out);
  } else {
    pad(pos, out);
  }
  uint4* o = reinterpret_cast<uint4*>(lv0 + 8 * pos);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    o[i] = make_uint4((u32)out[2 * i], (u32)(out[2 * i] >> 32), (u32)out[2 * i + 1], (u32)(out[2 * i + 1] >> 32));
}

DEV void ld8(const u64* p, u64* d) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = q[i];
    d[2 * i] = (u64)v.y << 32 | v.x;
    d[2 * i + 1] = (u64)v.w << 32 | v.z;
  }
}
DEV void st8(u64* p, const u64* d) {
  uint4* q = reinterpret_cast<uint4*>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) q[i] = make_uint4((u32)d[2 * i], (u32)(d[2 * i] >> 32), (u32)d[2 * i + 1], (u32)(d[2 * i + 1] >> 32));
}

// Level d+1 from level d (n_in nodes), a thread a node.
__global__ void __launch_bounds__(NODE_THREADS) level_kernel(const u64* __restrict__ in, u64* __restrict__ out, u64 n_in, int d) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_in / 2) return;
  u64 L[8], R[8], o[8];
  ld8(in + 16 * i, L);
  ld8(in + 16 * i + 8, R);
  node(d, i, L, R, o);
  st8(out + 8 * i, o);
}

// The top: levels d+1 .. d+nl from level d (n_in <= 512 nodes) in one block, the levels in shared memory.
// out[l] is level d+1+l's array.
struct LevelPtrs { u64* p[10]; };
__global__ void __launch_bounds__(NODE_THREADS) node_kernel(const u64* __restrict__ in, LevelPtrs out, u64 n_in, int d, int nl) {
  __shared__ u64 s[NODE_THREADS * 8];
  const int t = threadIdx.x;
  const u64 n1 = n_in / 2, i1 = (u64)blockIdx.x * NODE_THREADS + t;
  u64 L[8], R[8], o[8];
  if (i1 < n1) {
    ld8(in + 16 * i1, L);
    ld8(in + 16 * i1 + 8, R);
    node(d, i1, L, R, o);
    st8(s + 8 * t, o);
    st8(out.p[0] + 8 * i1, o);
  }
  u64 cnt = n1 < NODE_THREADS ? n1 : NODE_THREADS;
  for (int l = 1; l < nl && cnt > 1; ++l) {
    __syncthreads();
    cnt >>= 1;
    if (t < (int)cnt) {
      cpy8(L, s + 16 * t);
      cpy8(R, s + 16 * t + 8);
      node(d + l, (u64)blockIdx.x * cnt + t, L, R, o);
    }
    __syncthreads();
    if (t < (int)cnt) {
      cpy8(s + 8 * t, o);
      st8(out.p[l] + 8 * ((u64)blockIdx.x * cnt + t), o);
    }
  }
}

// The raw compression rate: each thread chains REPS compressions, its block from the last state.
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

// --- host --------------------------------------------------------------------------------------------------------

static const char FRAME[] = "veritor/protocol/merkle/frame/v3";  // + its NUL: 33 bytes
static const char SCHEMA[] = "hm96-sha512/row/v2";

struct Params {
  uint8_t P[128], salt_prefix[128], leaf_prefix[128], domain_id[64], ks[64];
  std::vector<uint8_t> T;   // 256 x 256
};

static std::vector<uint8_t> readf(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
  std::vector<uint8_t> v;
  uint8_t buf[1 << 16];
  size_t r;
  while ((r = fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + r);
  fclose(f);
  return v;
}
static void writef(const std::string& path, const void* p, size_t n) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
  fclose(f);
}

static Params read_params(const std::string& path) {
  const auto v = readf(path);
  const size_t want = 4 + 128 * 3 + 64 * 2 + 65536;
  if (v.size() != want || memcmp(v.data(), "CGP1", 4)) { fprintf(stderr, "%s: not a CGP1 params file\n", path.c_str()); exit(2); }
  Params p;
  const uint8_t* q = v.data() + 4;
  memcpy(p.P, q, 128); q += 128;
  memcpy(p.salt_prefix, q, 128); q += 128;
  memcpy(p.leaf_prefix, q, 128); q += 128;
  memcpy(p.domain_id, q, 64); q += 64;
  memcpy(p.ks, q, 64); q += 64;
  p.T.assign(q, q + 65536);
  return p;
}

// reference merkle._message: frame || u32be(len tag) || tag || (u64be(len part) || part)...
static std::vector<uint8_t> message(const char* tag, const std::vector<std::vector<uint8_t>>& parts) {
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
static std::vector<u64> padded_words(const std::vector<uint8_t>& msg) {
  const size_t total = ((msg.size() + 17 + 127) / 128) * 128;
  std::vector<uint8_t> b(total, 0);
  memcpy(b.data(), msg.data(), msg.size());
  b[msg.size()] = 0x80;
  sha::put_be64(b.data() + total - 8, (u64)msg.size() * 8);
  std::vector<u64> w(total / 8);
  for (size_t i = 0; i < w.size(); ++i) w[i] = sha::be64(b.data() + 8 * i);
  return w;
}
static std::vector<uint8_t> zeros(int n) { return std::vector<uint8_t>(n, 0); }
static std::vector<uint8_t> bytes_of(const uint8_t* p, int n) { return std::vector<uint8_t>(p, p + n); }
static void words_of(const uint8_t* block, u64* w, int n) { for (int i = 0; i < n; ++i) w[i] = sha::be64(block + 8 * i); }

static Consts build_consts(const Params& p, u32 lb) {
  Consts c{};
  u64 w[16], st[8];
  sha::iv(st); words_of(p.P, w, 16); sha::compress(st, w); memcpy(c.mid_P, st, 64);
  {
    u64 blk[16] = {0x8000000000000000ull};
    blk[15] = (u64)(128 + lb) * 8;
    sha::schedule_kw(blk, c.kw_xpad);
  }
  sha::iv(st); words_of(p.salt_prefix, w, 16); sha::compress(st, w); memcpy(c.mid_salt, st, 64);
  sha::iv(st); words_of(p.leaf_prefix, w, 16); sha::compress(st, w); memcpy(c.mid_leaf, st, 64);
  {
    u64 blk[16] = {0x8000000000000000ull};
    blk[15] = 256 * 8;
    sha::schedule_kw(blk, c.kw_leafpad);
  }
  words_of(p.ks, c.ks, 8);
  {
    u64 blk[16] = {};
    memcpy(blk, c.ks, 64);
    sha::iv(st);
    sha::rounds_prefix(st, blk, 8, c.ks_m8);
  }
  const auto did = bytes_of(p.domain_id, 64);
  const std::vector<uint8_t> schema(SCHEMA, SCHEMA + strlen(SCHEMA));
  for (int nb = 1; nb <= MAX_NB; ++nb) {
    const auto m = padded_words(message("leaf", {did, zeros(nb), zeros(nb), schema, zeros(64)}));
    if (m.size() != 32) { fprintf(stderr, "frame leaf at nb %d is not two blocks\n", nb); exit(2); }
    std::copy(m.begin(), m.end(), c.fl_tpl[nb]);
    for (int i = 0; i < 15; ++i)
      if (c.fl_tpl[nb][i] != c.fl_tpl[1][i]) { fprintf(stderr, "frame leaf word %d depends on nb\n", i); exit(2); }
  }
  sha::iv(st);
  sha::rounds_prefix(st, c.fl_tpl[1], 15, c.fl_m15);
  for (int ni = 1; ni <= MAX_NI; ++ni) {
    const auto m = padded_words(message("node", {did, std::vector<uint8_t>{0}, zeros(ni), zeros(64), zeros(64)}));
    if (m.size() != 48) { fprintf(stderr, "node at ni %d is not three blocks\n", ni); exit(2); }
    std::copy(m.begin() + 16, m.end(), c.node_tpl[ni]);
  }
  for (int level = 0; level < 64; ++level) {
    const auto m = padded_words(message("node", {did, std::vector<uint8_t>{(uint8_t)level}, zeros(1), zeros(64), zeros(64)}));
    u64 blk[16];
    std::copy(m.begin(), m.begin() + 16, blk);
    sha::iv(c.node_mid[level]);
    sha::compress(c.node_mid[level], blk);
  }
  for (int nb = 1; nb <= MAX_NB; ++nb) {
    const auto m = padded_words(message("pad", {did, zeros(nb)}));
    if (m.size() != 32) { fprintf(stderr, "pad at nb %d is not two blocks\n", nb); exit(2); }
    std::copy(m.begin(), m.begin() + 16, c.pad_tpl[nb]);
  }
  return c;
}

static int depth_of(u64 n) { int d = 0; while ((1ull << d) < n) ++d; return d; }

struct Committer {
  u64 n = 0, width = 0;
  int depth = 0;
  u64* levels = nullptr;            // every level, leaves first: level d at offset 8 (2 width - 2 width >> d) words
  u64* bc = nullptr;
  uint4* T = nullptr;
  std::vector<u64*> lv;
  void init(const Params& p, u64 n_, u32 lb, bool keep_bc) {
    n = n_; depth = depth_of(n); width = 1ull << depth;
    if (n == 0 || n > (1ull << 32)) { fprintf(stderr, "1 to 2^32 leaves, not %llu\n", (unsigned long long)n); exit(2); }
    Consts c = build_consts(p, lb);
    CK(cudaMemcpyToSymbol(C, &c, sizeof c));
    CK(cudaMalloc(&levels, (2 * width) * 64));
    if (keep_bc) CK(cudaMalloc(&bc, n * 128));
    CK(cudaMalloc(&T, 65536));
    CK(cudaMemcpy(T, p.T.data(), 65536, cudaMemcpyHostToDevice));
    u64 off = 0;
    for (int d = 0; d <= depth; ++d) { lv.push_back(levels + 8 * off); off += width >> d; }
  }
  // Above 2^16 leaves T sits in shared memory (one block of LEAF_THREADS an SM); at or below, where the rows dominate a
  // leaf's work, blocks of 64 threads read T through L1, which spreads few leaves over the SMs and takes no shared memory.
  template <class Row, bool PRF>
  void leaves(const Row& row, const uint8_t* salts, cudaStream_t st) {
    if (width > (1u << 16)) {
      auto k = leaf_kernel<Row, PRF, true>;
      static bool attr = false;
      if (!attr) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, TSMEM)); attr = true; }
      k<<<(unsigned)((width + LEAF_THREADS - 1) / LEAF_THREADS), LEAF_THREADS, TSMEM, st>>>(row, salts, T, n, width, lv[0], bc);
    } else {
      leaf_kernel<Row, PRF, false><<<(unsigned)((width + 63) / 64), 64, 0, st>>>(row, salts, T, n, width, lv[0], bc);
    }
  }
  // A level a launch while a level is wider than 512 nodes (blocks of 64 threads below 2^16 nodes, so few nodes still spread
  // over the SMs), then the top in one block.
  void nodes(cudaStream_t st) {
    int d = 0;
    for (; d < depth && (width >> d) > 2 * NODE_THREADS; ++d) {
      const u64 n_out = width >> (d + 1);
      const unsigned t = n_out >= (1u << 16) ? NODE_THREADS : 64;
      level_kernel<<<(unsigned)((n_out + t - 1) / t), t, 0, st>>>(lv[d], lv[d + 1], width >> d, d);
    }
    if (d < depth) {
      LevelPtrs o{};
      for (int l = 0; l < depth - d; ++l) o.p[l] = lv[d + 1 + l];
      node_kernel<<<1, NODE_THREADS, 0, st>>>(lv[d], o, width >> d, d, depth - d);
    }
  }
  void release() { cudaFree(levels); cudaFree(bc); cudaFree(T); lv.clear(); }
};

static int selftest() {
  // FIPS 180-4's "abc" and the two-block vector
  const char* in[2] = {"abc", "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"};
  const char* want[2] = {"ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
                         "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"};
  int bad = 0;
  for (int i = 0; i < 2; ++i) {
    uint8_t d[64];
    sha::digest((const uint8_t*)in[i], strlen(in[i]), d);
    char hex[129];
    for (int j = 0; j < 64; ++j) sprintf(hex + 2 * j, "%02x", d[j]);
    bad += strcmp(hex, want[i]) != 0;
  }
  printf("{\"selftest\": %s}\n", bad ? "false" : "true");
  return bad ? 1 : 0;
}

static std::string out_dir;

static int check(int argc, char** argv) {
  // check PARAMS OUT MODE N LBITS STRIDE ROWS SALTS|prf
  if (argc < 10) { fprintf(stderr, "check PARAMS OUT e2e|rows N LBITS STRIDE ROWS SALTS|prf\n"); return 2; }
  const Params p = read_params(argv[2]);
  out_dir = argv[3];
  const std::string mode = argv[4];
  const u64 n = strtoull(argv[5], nullptr, 10), lbits = strtoull(argv[6], nullptr, 10), stride = strtoull(argv[7], nullptr, 10);
  const u32 lb = (u32)((lbits + 7) / 8);
  const bool prf = !strcmp(argv[9], "prf");
  if (mode != "e2e" && (stride % 16 || stride < lb)) { fprintf(stderr, "a row stride is a multiple of 16 bytes, at least the row\n"); return 2; }
  auto rows = readf(argv[8]);
  if (rows.size() < (mode == "e2e" ? 32 * n : (n - 1) * stride + lb)) { fprintf(stderr, "rows: %zu bytes is short\n", rows.size()); return 2; }
  rows.resize(rows.size() + 32, 0);
  uint8_t *drows, *dsalts = nullptr;
  CK(cudaMalloc(&drows, rows.size()));
  CK(cudaMemcpy(drows, rows.data(), rows.size(), cudaMemcpyHostToDevice));
  if (!prf) {
    const auto s = readf(argv[9]);
    if (s.size() != 192 * n) { fprintf(stderr, "salts: %zu bytes for %llu leaves\n", s.size(), (unsigned long long)n); return 2; }
    CK(cudaMalloc(&dsalts, s.size() + 16));
    CK(cudaMemcpy(dsalts, s.data(), s.size(), cudaMemcpyHostToDevice));
  }
  Committer cm;
  cm.init(p, n, mode == "e2e" ? 128 : lb, true);
  if (mode == "e2e") {
    if (lbits != 1024) { fprintf(stderr, "an e2e leaf is 1024 bits\n"); return 2; }
    RowE2E r{drows};
    prf ? cm.leaves<RowE2E, true>(r, dsalts, 0) : cm.leaves<RowE2E, false>(r, dsalts, 0);
  } else {
    RowBytes r{drows, stride, lb};
    prf ? cm.leaves<RowBytes, true>(r, dsalts, 0) : cm.leaves<RowBytes, false>(r, dsalts, 0);
  }
  cm.nodes(0);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<u64> lv(16 * cm.width), bcv(16 * n);
  CK(cudaMemcpy(lv.data(), cm.levels, lv.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(bcv.data(), cm.bc, bcv.size() * 8, cudaMemcpyDeviceToHost));
  auto to_bytes = [](std::vector<u64>& v) {
    std::vector<uint8_t> b(v.size() * 8);
    for (size_t i = 0; i < v.size(); ++i) sha::put_be64(b.data() + 8 * i, v[i]);
    return b;
  };
  u64 used = 0;
  for (int d = 0; d <= cm.depth; ++d) used += cm.width >> d;
  lv.resize(8 * used);
  auto lb_ = to_bytes(lv), bb = to_bytes(bcv);
  writef(out_dir + "/levels.bin", lb_.data(), lb_.size());
  writef(out_dir + "/bc.bin", bb.data(), bb.size());
  printf("{\"check\": \"%s\", \"n\": %llu, \"depth\": %d}\n", mode.c_str(), (unsigned long long)n, cm.depth);
  cm.release();
  return 0;
}

// --- timing ------------------------------------------------------------------------------------------------------

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  unsigned sm() {
    unsigned c = 0;
    if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c);
    return c;
  }
};

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
static void json_list(const char* name, const std::vector<double>& v) {
  printf("\"%s\": [", name);
  for (size_t i = 0; i < v.size(); ++i) printf("%s%.5f", i ? ", " : "", v[i]);
  printf("]");
}

__global__ void fill(uint8_t* p, size_t n, u32 seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    u32 x = (u32)i * 0x9E3779B1u ^ seed;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = (uint8_t)x;
  }
}

struct Gemm {   // y (m x n, s32) = A (m x k, int8, row-major) B (n x k, int8, row-major)^T by cuBLASLt, autotuned
  cublasLtHandle_t lt{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t la{}, lb{}, lc{};
  cublasLtMatmulAlgo_t algo{};
  void* ws = nullptr;
  size_t wsz = 64 << 20;
  int8_t *A = nullptr, *B = nullptr;
  int32_t* Y = nullptr;
  int m, n, k, algo_index = -1;
  float best_ms = 0;
  void init(int m_, int n_, int k_, cudaStream_t st) {
    m = m_; n = n_; k = k_;
    CL(cublasLtCreate(&lt));
    CK(cudaMalloc(&ws, wsz));
    CK(cudaMalloc(&A, (size_t)m * k)); CK(cudaMalloc(&B, (size_t)n * k)); CK(cudaMalloc(&Y, (size_t)m * n * 4));
    fill<<<1024, 256>>>((uint8_t*)A, (size_t)m * k, 11u);
    fill<<<1024, 256>>>((uint8_t*)B, (size_t)n * k, 13u);
    CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    const cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N));
    CL(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, k, n, k));
    CL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, k, m, k));
    CL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, n, m, n));
    cublasLtMatmulPreference_t pref;
    CL(cublasLtMatmulPreferenceCreate(&pref));
    CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
    cublasLtMatmulHeuristicResult_t res[8];
    int got = 0;
    CL(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, res, &got));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < got; ++i) {
      algo = res[i].algo;
      std::vector<double> t;
      bool ok = true;
      for (int r = 0; r < 6 && ok; ++r) {
        CK(cudaEventRecord(e0, st));
        ok = run(st) == CUBLAS_STATUS_SUCCESS;
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (r) t.push_back(ms);
      }
      if (ok && (algo_index < 0 || median(t) < best_ms)) { best_ms = (float)median(t); algo_index = i; }
    }
    if (algo_index < 0) { fprintf(stderr, "no cuBLASLt int8 algorithm\n"); exit(3); }
    algo = res[algo_index].algo;
  }
  cublasStatus_t run(cudaStream_t st) {
    const int32_t one = 1, zero = 0;
    return cublasLtMatmul(lt, op, &one, B, la, A, lb, &zero, Y, lc, Y, lc, &algo, ws, wsz, st);
  }
};

static int bench(int argc, char** argv) {
  // bench PARAMS MODE N LBITS REPS [GM GN GK]
  if (argc < 7) { fprintf(stderr, "bench PARAMS e2e|rows N LBITS REPS [GEMM_M GEMM_N GEMM_K]\n"); return 2; }
  const Params p = read_params(argv[2]);
  const std::string mode = argv[3];
  const u64 n = strtoull(argv[4], nullptr, 10), lbits = strtoull(argv[5], nullptr, 10);
  const int reps = atoi(argv[6]);
  const bool gemm = argc >= 10;
  const u32 lb = (u32)((lbits + 7) / 8);
  const u64 data = mode == "e2e" ? 32 * n : (u64)lb * n;   // operand bytes the commitment covers
  if (mode == "rows" && lb % 16) { fprintf(stderr, "bench rows: a multiple of 128 bits\n"); return 2; }
  uint8_t* src;
  CK(cudaMalloc(&src, data + 64));
  fill<<<1024, 256>>>(src, data + 64, 7u);
  cudaStream_t sc, sg;
  CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&sg, cudaStreamNonBlocking));
  Committer cm;
  cm.init(p, n, mode == "e2e" ? 128 : lb, false);
  auto commit = [&](cudaStream_t st) {
    if (mode == "e2e") cm.leaves<RowE2E, true>(RowE2E{src}, nullptr, st);
    else cm.leaves<RowBytes, true>(RowBytes{src, lb, lb}, nullptr, st);
    cm.nodes(st);
  };
  cudaEvent_t e[6];
  for (auto& x : e) CK(cudaEventCreate(&x));
  auto ms = [](cudaEvent_t a, cudaEvent_t b) { float t; CK(cudaEventElapsedTime(&t, a, b)); return (double)t; };
  Nv nv;
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  for (int w = 0; w < 2; ++w) commit(sc);
  CK(cudaStreamSynchronize(sc));
  CK(cudaGetLastError());
  // alone: the whole commitment, and its leaf pass alone
  std::vector<double> t_all, t_leaf, mhz;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e[0], sc));
    if (mode == "e2e") cm.leaves<RowE2E, true>(RowE2E{src}, nullptr, sc);
    else cm.leaves<RowBytes, true>(RowBytes{src, lb, lb}, nullptr, sc);
    CK(cudaEventRecord(e[1], sc));
    cm.nodes(sc);
    CK(cudaEventRecord(e[2], sc));
    CK(cudaEventSynchronize(e[2]));
    t_leaf.push_back(ms(e[0], e[1]));
    t_all.push_back(ms(e[0], e[2]));
    mhz.push_back(nv.sm());
  }
  const double tc = median(t_all);
  const double comps = (double)n * (mode == "e2e" ? 2 : (lb + 17 + 127) / 128) + 3.0 * n + 2.0 * n + 2.0 * n + 2.0 * n + 2.0 * (cm.width - 1);
  printf("{\"bench\": \"%s\", \"gpu\": \"%s\", \"sms\": %d, \"sha_add\": %d, \"sha_rot\": %d, \"leaf_threads\": %d, \"n\": %llu, \"lbits\": %llu, "
         "\"depth\": %d, \"operand_bytes\": %llu, \"commit_ms\": %.5f, \"leaf_ms\": %.5f, \"nodes_ms\": %.5f, \"mhz\": %.0f, "
         "\"compressions\": %.0f, \"gcomp_per_s\": %.3f, \"sha_gb_per_s\": %.2f, \"operand_gb_per_s\": %.2f, ",
         mode.c_str(), prop.name, prop.multiProcessorCount, SHA_ADD, SHA_ROT, LEAF_THREADS, (unsigned long long)n,
         (unsigned long long)lbits, cm.depth, (unsigned long long)data, tc, median(t_leaf), tc - median(t_leaf), median(mhz),
         comps, comps / tc * 1e-6, comps * 128 / tc * 1e-6, data / tc * 1e-6);
  json_list("commit_ms_all", t_all);
  printf("}\n");
  fflush(stdout);
  if (gemm) {
    Gemm G;
    G.init(atoi(argv[7]), atoi(argv[8]), atoi(argv[9]), sg);
    // GEMM alone, then the same GEMMs with the commitment beside them on another stream, then both on one stream
    const int ng = std::max(1, (int)(tc / G.best_ms + 0.5));
    std::vector<double> g_alone, both_wall, both_commit, both_gemm, serial_wall, mz;
    for (int r = 0; r < reps; ++r) {
      CK(cudaEventRecord(e[0], sg));
      for (int i = 0; i < ng; ++i) CL(G.run(sg));
      CK(cudaEventRecord(e[1], sg));
      CK(cudaEventSynchronize(e[1]));
      g_alone.push_back(ms(e[0], e[1]) / ng);
      // concurrent: both streams wait on one start event
      CK(cudaEventRecord(e[0], sg));
      CK(cudaStreamWaitEvent(sc, e[0]));
      for (int i = 0; i < ng; ++i) CL(G.run(sg));
      CK(cudaEventRecord(e[1], sg));
      commit(sc);
      CK(cudaEventRecord(e[2], sc));
      CK(cudaEventSynchronize(e[1]));
      CK(cudaEventSynchronize(e[2]));
      const double tg = ms(e[0], e[1]), tcm = ms(e[0], e[2]);
      both_gemm.push_back(tg);
      both_commit.push_back(tcm);
      both_wall.push_back(std::max(tg, tcm));
      mz.push_back(nv.sm());
      // serial on one stream
      CK(cudaEventRecord(e[3], sg));
      for (int i = 0; i < ng; ++i) CL(G.run(sg));
      commit(sg);
      CK(cudaEventRecord(e[4], sg));
      CK(cudaEventSynchronize(e[4]));
      serial_wall.push_back(ms(e[3], e[4]));
    }
    const double ga = median(g_alone), bw = median(both_wall), sw = median(serial_wall);
    // the share of the shorter side that concurrency hides, against both on one stream measured in the same loop
    const double hidden = (sw - bw) / std::min(ng * ga, tc);
    printf("{\"beside_gemm\": {\"m\": %d, \"n\": %d, \"k\": %d, \"lt_algo_index\": %d, \"gemm_ms\": %.5f, \"gemms\": %d, "
           "\"commit_alone_ms\": %.5f, \"both_wall_ms\": %.5f, \"both_commit_ms\": %.5f, \"both_gemm_ms\": %.5f, "
           "\"serial_wall_ms\": %.5f, \"hidden_fraction\": %.4f, \"commit_over_gemm\": %.4f, \"commit_gb_per_s_beside\": %.2f, "
           "\"mhz\": %.0f, ",
           G.m, G.n, G.k, G.algo_index, ga, ng, tc, bw, median(both_commit), median(both_gemm), sw, hidden, tc / ga,
           data / median(both_commit) * 1e-6, median(mz));
    json_list("both_wall_all", both_wall);
    printf(", ");
    json_list("serial_wall_all", serial_wall);
    printf("}}\n");
  }
  cm.release();
  return 0;
}

static int sha_bench(int reps) {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  const int threads = 256, blocks = prop.multiProcessorCount * 8, chain = 256;
  u64* io; CK(cudaMalloc(&io, (size_t)threads * blocks * 8));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  Nv nv;
  std::vector<double> t, mz;
  for (int r = 0; r <= reps; ++r) {
    CK(cudaEventRecord(e0));
    sha_rate<<<blocks, threads>>>(io, chain);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    if (r) { t.push_back(ms); mz.push_back(nv.sm()); }
  }
  const double comps = (double)threads * blocks * chain, tm = median(t);
  printf("{\"sha\": {\"sha_add\": %d, \"sha_rot\": %d, \"compressions\": %.0f, \"ms\": %.5f, \"gcomp_per_s\": %.3f, \"gb_per_s\": %.2f, \"mhz\": %.0f}}\n",
         SHA_ADD, SHA_ROT, comps, tm, comps / tm * 1e-6, comps * 128 / tm * 1e-6, median(mz));
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "selftest")) return selftest();
#ifdef COMMIT_GPU_DIAG_NOMASK
  if (argc >= 2 && !strcmp(argv[1], "check")) { fprintf(stderr, "a timing-only build (COMMIT_GPU_DIAG_NOMASK) has no check\n"); return 2; }
#endif
  if (argc >= 2 && !strcmp(argv[1], "check")) return check(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "bench")) return bench(argc, argv);
  if (argc >= 3 && !strcmp(argv[1], "sha")) return sha_bench(atoi(argv[2]));
  fprintf(stderr, "usage: commit_gpu selftest | check ... | bench ... | sha REPS\n");
  return 2;
}
