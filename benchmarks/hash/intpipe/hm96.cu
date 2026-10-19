// hm96 leaf mode: 4 KiB rows committed in Verity's leaf format, base SHA-512 against one candidate primitive in every hash
// position, rep by rep on one GPU. The format (hm96_ref.py, README "hm96 mode"): per position pos of an n-leaf domain
// (rank = pos, the tree padded to width = 2^depth), with H the arm's primitive, NB its digest bytes, SALT = 2 NB + 64:
//   d_j  = H(S(j, 4, 8192) || segment j)  j < 4      x = H(Q(32768) || d_0 || .. || d_3)     (sha512/row-seg/v1)
//   y    = H(Ks || u64be pos || u64be r)  r < SALT / NB      (the salt, derived on the device as commit_gpu derives it)
//   c    = H(salt_prefix || y);  b = x ^ M_key y;  h = H(leaf_prefix(key) || b || c)                   (hm96-sha512/v1)
//   leaf = H(frame "leaf" domain_id uint(pos) uint(pos) "hm96-sha512/row-seg/v1" h);  pad, node   (frame-v3-sha512)
// Base SHA-512 is that format exactly (hm96_ref.py holds it to verity.ml.commitments); a candidate is the hypothetical
// hm96-<P>. -DCAND_KIND=0 (SHA-512 itself, a noise control), 1 (SHA-512 at -DCAND_R rounds), 3 (keyed BLAKE3).
//
//   hm96 dump PARAMS DIR N          both arms' levels and b || c over N leaves of the fill operand (for hm96_ref.py check)
//   hm96 bench PDIR REPS N[,N...]   JSON lines (PDIR/params-N.bin): REPS reps over each N, base and candidate back to back
//                                   (order alternating), each its data phase alone ("pre": salts precomputed) and its
//                                   whole step (salt phase + data phase), each launched kernel by kernel from the host
//                                   and as one CUDA graph ("_graph"), then the data phase split by kernel (segments,
//                                   leaves, level launches, the top CTA) on a pre-filled queue, with SM clocks
//
// Two phases, as commit_gpu splits them: the salt phase (y, M_key y, c, the frame leaf's first blocks F, the pads and every
// node above pads only) reads no data; the data phase (segments, x, b, h, the frame leaf, the data's nodes) is the rest.
// Segments: a thread a segment, their blocks through hashes.cuh's WarpLoader (the candidates driver's loader).
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hashes.cuh"

#ifndef CAND_KIND
#define CAND_KIND 1
#endif
#ifndef CAND_R
#define CAND_R 24
#endif
#ifndef SEG_WARPS
#define SEG_WARPS 4
#endif

using u32 = uint32_t;
using u64 = uint64_t;
#ifndef DEV
#define DEV __device__ __forceinline__
#endif
#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
      exit(3);                                                                                   \
    }                                                                                            \
  } while (0)

#ifndef LEAF_BYTES
#define LEAF_BYTES 4096
#endif
#ifndef SEG_BYTES
#define SEG_BYTES 1024
#endif
constexpr int LEAF = LEAF_BYTES, SEGB = SEG_BYTES, NSEG = LEAF / SEGB, MAXW = 48, LEVELS = 33, THREADS = 256, TOP = 256;
static_assert(LEAF % SEGB == 0 && NSEG <= 4 && SEGB % 128 == 0, "1 to 4 segments of whole SHA-512 blocks");
constexpr int SEG_THREADS = 32 * SEG_WARPS;

// One arm's constants (hm96_ref.Format.params): u64 fields, a BLAKE3 word in the low half. Templates are whole messages
// with the variable fields zero and no padding; midstates are chaining values after a message's first block.
struct HP {
  u64 iv[8];                 // SHA-512's IV; BLAKE3: its key's words
  u64 seg_mid[4][8];         // after S(j, NSEG, 8 SEGB), j < NSEG (the rest zero)
  u64 q_mid[8], salt_mid[8], leaf_mid[8];
  u64 node_mid[LEVELS][8];   // after a node message's first block at each level
  u64 prf_tpl[16];           // Ks || u64be 0 || u64be 0
  u64 fl_tpl[4][MAXW], node_tpl[4][MAXW], pad_tpl[4][MAXW];   // by len(uint(pos or index)), 1 .. 3
};
static_assert(sizeof(HP) == 8 * (8 + 32 + 24 + 8 * LEVELS + 16 + 12 * MAXW), "HP");
__constant__ HP HPC[2];   // 0: base SHA-512, 1: the candidate

// SHA-512 at R rounds (R = 80: sha512.cuh's compression, the baseline): big-endian u64 words, MD padding
template <int R>
struct PSha {
  using W = u64;
  static constexpr int NB = 64, BLOCK = 128, WB = 8;
  static constexpr bool BE = true;
  __host__ __device__ static constexpr int nblk(int len) { return (len + 17 + 127) / 128; }
  template <int LEN>
  static DEV void comp(W* cv, W* m, int) {
    if constexpr (R == 80) sha::compress(cv, m);
    else ip::sha512_rounds<R>(cv, m);
  }
  template <int LEN>
  static DEV void pad(W* m) {
    m[LEN / 8] |= 0x80ull << (56 - 8 * (LEN % 8));
    m[16 * nblk(LEN) - 1] |= 8ull * LEN;
  }
  template <class L>
  static DEV void seg(const L& ld, W* cv, const u64*) {
    W w[16];
#pragma unroll 1
    for (int b = 0; b < SEGB / 128; ++b) {
      uint4 v[8];
      ld.block(b, v);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        w[2 * q] = (u64)ip::bswap32(v[q].x) << 32 | ip::bswap32(v[q].y);
        w[2 * q + 1] = (u64)ip::bswap32(v[q].z) << 32 | ip::bswap32(v[q].w);
      }
      comp<0>(cv, w, 0);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = 0;
    w[0] = 0x8000000000000000ull;
    w[15] = 8ull * (128 + SEGB);
    comp<0>(cv, w, 0);
  }
};

// keyed BLAKE3 (one chunk for every message but a segment's): little-endian u32 words, no padding
struct PB3 {
  using W = u32;
  static constexpr int NB = 32, BLOCK = 64, WB = 4;
  static constexpr bool BE = false;
  __host__ __device__ static constexpr int nblk(int len) { return len <= 64 ? 1 : (len + 63) / 64; }
  template <int LEN>
  static DEV void comp(W* cv, W* m, int b) {
    static_assert(LEN <= 1024, "one chunk");
    constexpr int N = nblk(LEN);
    const u32 f = B3_KEYED | (b == 0 ? B3_START : 0) | (b == N - 1 ? B3_END | B3_ROOT : 0);
    b3c(cv, m, 0, b == N - 1 ? LEN - 64 * (N - 1) : 64, f, cv);
  }
  template <int LEN>
  static DEV void pad(W*) {}
  // S's block, then the 1 KiB: chunk 0 is S and blocks 0 .. 14, chunk 1 block 15, then their parent
  template <class L>
  static DEV void seg(const L& ld, W* cv, const u64* key) {
    u32 m[16], c1[8];
#pragma unroll 1
    for (int b = 0; b < 15; ++b) {
      uint4 v[4];
      ld.block(b, v);
#pragma unroll
      for (int q = 0; q < 4; ++q) { m[4 * q] = v[q].x; m[4 * q + 1] = v[q].y; m[4 * q + 2] = v[q].z; m[4 * q + 3] = v[q].w; }
      b3c(cv, m, 0, 64, B3_KEYED | (b == 14 ? B3_END : 0), cv);
    }
    uint4 v[4];
    ld.block(15, v);
#pragma unroll
    for (int q = 0; q < 4; ++q) { m[4 * q] = v[q].x; m[4 * q + 1] = v[q].y; m[4 * q + 2] = v[q].z; m[4 * q + 3] = v[q].w; }
#pragma unroll
    for (int i = 0; i < 8; ++i) c1[i] = (u32)key[i];
    b3c(c1, m, 1, 64, B3_KEYED | B3_START | B3_END, c1);
#pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = cv[i]; m[8 + i] = c1[i]; cv[i] = (u32)key[i]; }
    b3c(cv, m, 0, 64, B3_KEYED | B3_PARENT | B3_ROOT, cv);
  }
};

using Base = PSha<80>;
#if CAND_KIND == 0
using Cand = PSha<80>;
static const std::string CAND_NAME = "sha512";
#elif CAND_KIND == 1
using Cand = PSha<CAND_R>;
static const std::string CAND_NAME = "sha512r" + std::to_string(CAND_R);
#elif CAND_KIND == 3
using Cand = PB3;
static const std::string CAND_NAME = "blake3k";
#endif

template <class W>
DEV void ldc(W* d, const u64* s, int n) {
#pragma unroll
  for (int i = 0; i < n; ++i) d[i] = (W)s[i];
}
template <class P, int OFF>
DEV void or_byte(typename P::W* m, u32 v) {
  using W = typename P::W;
  constexpr int K = OFF / P::WB, S = P::BE ? 8 * (P::WB - 1 - OFF % P::WB) : 8 * (OFF % P::WB);
  m[K] |= (W)v << S;
}
// the low N bytes of v, big-endian, at byte OFF
template <class P, int OFF, int N>
DEV void or_be(typename P::W* m, u64 v) {
  if constexpr (N > 0) {
    or_byte<P, OFF + N - 1>(m, (u32)v & 0xffu);
    or_be<P, OFF, N - 1>(m, v >> 8);
  }
}
// a digest (8 words) at byte OFF
template <class P, int OFF>
DEV void or_dig(typename P::W* m, const typename P::W* d) {
  constexpr int K0 = OFF / P::WB, S = 8 * (OFF % P::WB), WBITS = 8 * P::WB;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if constexpr (S == 0) {
      m[K0 + i] |= d[i];
    } else if constexpr (P::BE) {
      m[K0 + i] |= d[i] >> S;
      m[K0 + i + 1] |= d[i] << (WBITS - S);
    } else {
      m[K0 + i] |= d[i] << S;
      m[K0 + i + 1] |= d[i] >> (WBITS - S);
    }
  }
}
template <class P>
DEV u32 byte_at(const typename P::W* d, int i) {
  if constexpr (P::BE) return (u32)(d[i / P::WB] >> (8 * (P::WB - 1 - i % P::WB))) & 0xffu;
  else return (u32)(d[i / P::WB] >> (8 * (i % P::WB))) & 0xffu;
}
// pad, then blocks SKIP .. of an LEN-byte message m (whole, from byte 0) from chaining value cv
template <class P, int LEN, int SKIP>
DEV void finish(typename P::W* cv, typename P::W* m) {
  P::template pad<LEN>(m);
#pragma unroll
  for (int b = SKIP; b < P::nblk(LEN); ++b) P::template comp<LEN>(cv, m + 16 * b, b);
}
DEV int nbytes(u64 x) { return x < 256 ? 1 : x < 65536 ? 2 : 3; }

// --- the frame: pad and node -----------------------------------------------------------------------------------------
template <class P, int A, int NBY>
DEV void pad_nb(u64 rank, typename P::W* out) {
  constexpr int LEN = 56 + P::NB + NBY;
  typename P::W m[16 * P::nblk(LEN)];
  ldc(m, HPC[A].pad_tpl[NBY], 16 * P::nblk(LEN));
  or_be<P, 56 + P::NB, NBY>(m, rank);
  ldc(out, HPC[A].iv, 8);
  finish<P, LEN, 0>(out, m);
}
template <class P, int A, int NI>
DEV void node_ni(int level, u64 index, const typename P::W* L, const typename P::W* R, typename P::W* out) {
  constexpr int NB = P::NB, LEN = 82 + 3 * NB + NI;
  static_assert(66 + NB >= P::BLOCK, "the index is past the first block");
  typename P::W m[16 * P::nblk(LEN)];
  ldc(m, HPC[A].node_tpl[NI], 16 * P::nblk(LEN));
  if constexpr (57 + NB >= P::BLOCK) or_byte<P, 57 + NB>(m, (u32)level);
  or_be<P, 66 + NB, NI>(m, index);
  or_dig<P, 74 + NB + NI>(m, L);
  or_dig<P, 82 + 2 * NB + NI>(m, R);
  ldc(out, HPC[A].node_mid[level], 8);
  finish<P, LEN, 1>(out, m);
}
template <class P, int A>
DEV void node(int level, u64 index, const typename P::W* L, const typename P::W* R, typename P::W* out) {
  switch (nbytes(index)) {
    case 1: node_ni<P, A, 1>(level, index, L, R, out); break;
    case 2: node_ni<P, A, 2>(level, index, L, R, out); break;
    default: node_ni<P, A, 3>(level, index, L, R, out); break;
  }
}
template <class W>
DEV void st8(W* p, const W* d) {
#pragma unroll
  for (int i = 0; i < 8; ++i) p[i] = d[i];
}

// --- the salt phase --------------------------------------------------------------------------------------------------
// M_key y as NB/4 little-endian u32: for salt bytes p = 16 q + r, bytes [16 q, 16 q + NB + 16) of T_{y_p} into A_r, then
// A_r shifted down r bytes (commit_gpu's hm96_mask at any digest size). T: 256 rows of KB bytes, TSTR apart.
template <class P>
DEV void hm96_mask(const uint8_t* T, const typename P::W* y, u32* mk) {
  constexpr int NB = P::NB, SALT = 2 * NB + 64, KB = 3 * NB + 64, TSTR = KB + 16, NW = NB / 4, AW = NW + 4;
#pragma unroll
  for (int w = 0; w < NW; ++w) mk[w] = 0;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    u32 Ac[AW];
#pragma unroll
    for (int i = 0; i < AW; ++i) Ac[i] = 0;
#pragma unroll
    for (int q = 0; q < SALT / 16; ++q) {
      const u32 v = byte_at<P>(y, 16 * q + r);
      const uint4* t = reinterpret_cast<const uint4*>(T + v * TSTR + 16 * q);
#pragma unroll
      for (int i = 0; i < AW / 4; ++i) {
        const uint4 u = t[i];
        Ac[4 * i] ^= u.x; Ac[4 * i + 1] ^= u.y; Ac[4 * i + 2] ^= u.z; Ac[4 * i + 3] ^= u.w;
      }
    }
    const int sw = r >> 2, s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < NW; ++w) mk[w] ^= s ? __funnelshift_r(Ac[w + sw], Ac[w + sw + 1], s) : Ac[w + sw];
  }
}
template <class P, int A, int NBY>
DEV void salt_nb(u64 pos, const uint8_t* T, typename P::W* S, typename P::W* F) {
  using W = typename P::W;
  constexpr int NB = P::NB, SALT = 2 * NB + 64, NWY = SALT / P::WB;
  W y[NWY];
#pragma unroll
  for (int r = 0; r < SALT / NB; ++r) {
    W m[16], cv[8];
    ldc(m, HPC[A].prf_tpl, 16);
    or_be<P, NB, 8>(m, pos);
    or_be<P, NB + 8, 8>(m, (u64)r);
    ldc(cv, HPC[A].iv, 8);
    finish<P, NB + 16, 0>(cv, m);
#pragma unroll
    for (int i = 0; i < 8; ++i) y[8 * r + i] = cv[i];
  }
  u32 mk[NB / 4];
  hm96_mask<P>(T, y, mk);
  W my[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if constexpr (P::BE) my[i] = (u64)ip::bswap32(mk[2 * i]) << 32 | ip::bswap32(mk[2 * i + 1]);
    else my[i] = mk[i];
  }
  constexpr int LC = P::BLOCK + SALT;
  W mc[16 * P::nblk(LC)], c[8];
#pragma unroll
  for (int i = 0; i < 16 * P::nblk(LC); ++i) mc[i] = 0;
#pragma unroll
  for (int i = 0; i < NWY; ++i) mc[16 + i] = y[i];
  ldc(c, HPC[A].salt_mid, 8);
  finish<P, LC, 1>(c, mc);
  st8(S + 16 * pos, my);
  st8(S + 16 * pos + 8, c);
  // the frame leaf's blocks before h
  constexpr int LF = 103 + 2 * NB + 2 * NBY, FS = (103 + NB + 2 * NBY) / P::BLOCK;
  W mf[16 * P::nblk(LF)], f[8];
  ldc(mf, HPC[A].fl_tpl[NBY], 16 * P::nblk(LF));
  or_be<P, 57 + NB, NBY>(mf, pos);
  or_be<P, 65 + NB + NBY, NBY>(mf, pos);
  ldc(f, HPC[A].iv, 8);
#pragma unroll
  for (int b = 0; b < FS; ++b) P::template comp<LF>(f, mf + 16 * b, b);
  st8(F + 8 * pos, f);
}
template <class P, int A>
__global__ void __launch_bounds__(THREADS, 1)
    salt_kernel(const uint4* __restrict__ Tg, u64 n, typename P::W* __restrict__ S, typename P::W* __restrict__ F) {
  constexpr int KB = 3 * P::NB + 64, TSTR = KB + 16;
  extern __shared__ uint4 smem[];
  for (int i = threadIdx.x; i < 256 * KB / 16; i += blockDim.x) smem[(i / (KB / 16)) * (TSTR / 16) + i % (KB / 16)] = Tg[i];
  __syncthreads();
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= n) return;
  const uint8_t* T = reinterpret_cast<const uint8_t*>(smem);
  switch (nbytes(pos)) {
    case 1: salt_nb<P, A, 1>(pos, T, S, F); break;
    case 2: salt_nb<P, A, 2>(pos, T, S, F); break;
    default: salt_nb<P, A, 3>(pos, T, S, F); break;
  }
}
template <class P, int A>
__global__ void __launch_bounds__(THREADS) pad_kernel(u64 n, u64 width, typename P::W* __restrict__ lv0) {
  const u64 rank = n + (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= width) return;
  typename P::W o[8];
  switch (nbytes(rank)) {
    case 1: pad_nb<P, A, 1>(rank, o); break;
    case 2: pad_nb<P, A, 2>(rank, o); break;
    default: pad_nb<P, A, 3>(rank, o); break;
  }
  st8(lv0 + 8 * rank, o);
}

// --- the data phase --------------------------------------------------------------------------------------------------
// d_j for segment t = 4 pos + j, its 1 KiB at src + 1024 t; the grid is a multiple of SEG_THREADS (src covers it)
template <class P, int A>
__global__ void __launch_bounds__(SEG_THREADS) seg_kernel(const uint8_t* __restrict__ src, u64 nseg, typename P::W* __restrict__ D) {
  using L = ip::WarpLoader<P::BLOCK, SEGB>;
  extern __shared__ uint4 smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const L ld{src + (t - lane) * SEGB, smem + warp * (L::SMEM / 16), lane};
  ld.start();
  typename P::W cv[8];
  ldc(cv, HPC[A].seg_mid[t % NSEG], 8);
  P::seg(ld, cv, HPC[A].iv);
  if (t < nseg) st8(D + 8 * t, cv);
}
template <class P, int A, int NBY>
DEV void leaf_nb(u64 pos, const typename P::W* D, const typename P::W* S, const typename P::W* F, typename P::W* out,
                 typename P::W* bc) {
  using W = typename P::W;
  constexpr int NB = P::NB, LX = P::BLOCK + NSEG * NB, LH = P::BLOCK + 2 * NB;
  W mx[16 * P::nblk(LX)], x[8];
#pragma unroll
  for (int i = 0; i < 16 * P::nblk(LX); ++i) mx[i] = 0;
#pragma unroll
  for (int i = 0; i < 8 * NSEG; ++i) mx[16 + i] = D[8 * NSEG * pos + i];
  ldc(x, HPC[A].q_mid, 8);
  finish<P, LX, 1>(x, mx);
  W mh[16 * P::nblk(LH)], h[8];
#pragma unroll
  for (int i = 0; i < 16 * P::nblk(LH); ++i) mh[i] = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    mh[16 + i] = x[i] ^ S[16 * pos + i];
    mh[24 + i] = S[16 * pos + 8 + i];
  }
  if (bc) {
#pragma unroll
    for (int i = 0; i < 16; ++i) bc[16 * pos + i] = mh[16 + i];
  }
  ldc(h, HPC[A].leaf_mid, 8);
  finish<P, LH, 1>(h, mh);
  constexpr int LF = 103 + 2 * NB + 2 * NBY, FS = (103 + NB + 2 * NBY) / P::BLOCK;
  W mf[16 * P::nblk(LF)];
  ldc(mf, HPC[A].fl_tpl[NBY], 16 * P::nblk(LF));
  or_be<P, 57 + NB, NBY>(mf, pos);
  or_be<P, 65 + NB + NBY, NBY>(mf, pos);
  or_dig<P, 103 + NB + 2 * NBY>(mf, h);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = F[8 * pos + i];
  finish<P, LF, FS>(out, mf);
}
template <class P, int A>
__global__ void __launch_bounds__(THREADS)
    leaf_kernel(u64 n, const typename P::W* __restrict__ D, const typename P::W* __restrict__ S, const typename P::W* __restrict__ F,
                typename P::W* __restrict__ lv0, typename P::W* __restrict__ bc) {
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= n) return;
  typename P::W o[8];
  switch (nbytes(pos)) {
    case 1: leaf_nb<P, A, 1>(pos, D, S, F, o, bc); break;
    case 2: leaf_nb<P, A, 2>(pos, D, S, F, o, bc); break;
    default: leaf_nb<P, A, 3>(pos, D, S, F, o, bc); break;
  }
  st8(lv0 + 8 * pos, o);
}

// --- the tree: level l from level l - 1, nodes lo .. hi (the data's: below ceil(n / 2^l); the pads': from there) --------
template <class P, int A>
__global__ void __launch_bounds__(THREADS)
    level_kernel(const typename P::W* __restrict__ in, typename P::W* __restrict__ out, u64 lo, u64 hi, int d) {
  const u64 i = lo + (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hi) return;
  typename P::W o[8];
  node<P, A>(d, i, in + 16 * i, in + 16 * i + 8, o);
  st8(out + 8 * i, o);
}
__host__ __device__ inline u64 level_off(u64 width, int l) { return 8 * (2 * width - (2 * width >> l)); }
__host__ __device__ inline u64 cdiv_pow2(u64 n, int l) { return (n + (1ull << l) - 1) >> l; }
template <class P, int A>
__global__ void __launch_bounds__(THREADS) top_kernel(typename P::W* lv, u64 width, u64 n, int from, int depth, int pads) {
  for (int l = from; l <= depth; ++l) {
    const u64 c = cdiv_pow2(n, l), lo = pads ? c : 0, hi = pads ? (width >> l) : c;
    const typename P::W* in = lv + level_off(width, l - 1);
    typename P::W* out = lv + level_off(width, l);
    for (u64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      typename P::W o[8];
      node<P, A>(l - 1, i, in + 16 * i, in + 16 * i + 8, o);
      st8(out + 8 * i, o);
    }
    __syncthreads();
  }
}

__global__ void fill(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = (uint8_t)x;
  }
}

// holds the stream for ns so the host has queued what follows before it runs: the split then times the GPU alone
__global__ void spin(u64 ns) {
  u64 t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); while (t - t0 < ns);
}

// --- host ------------------------------------------------------------------------------------------------------------
static std::vector<uint8_t> readf(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "open %s\n", path.c_str()); exit(2); }
  std::vector<uint8_t> v;
  uint8_t buf[1 << 16];
  size_t k;
  while ((k = fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + k);
  fclose(f);
  return v;
}
static void write_file(const std::string& path, const void* dev, size_t bytes) {
  std::vector<uint8_t> h(bytes);
  CK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(h.data(), 1, bytes, f) != bytes) { fprintf(stderr, "write %s\n", path.c_str()); exit(2); }
  fclose(f);
}

template <class P, int A>
struct Arm {
  using W = typename P::W;
  static constexpr int KB = 3 * P::NB + 64, TSMEM = 256 * (KB + 16);
  static constexpr int SEG_SMEM = SEG_WARPS * ip::WarpLoader<P::BLOCK, SEGB>::SMEM;
  u64 n = 0, width = 0;
  int depth = 0;
  cudaStream_t st = 0;
  uint4* T = nullptr;
  W *S = nullptr, *F = nullptr, *D = nullptr, *lv = nullptr, *bc = nullptr;
  void init(u64 n_, const uint8_t* t, bool with_bc) {
    n = n_;
    depth = 0;
    while ((1ull << depth) < n) ++depth;
    width = 1ull << depth;
    if (width > (1ull << 24)) { fprintf(stderr, "positions below 2^24\n"); exit(2); }
    CK(cudaMalloc(&T, 256 * KB));
    CK(cudaMemcpy(T, t, 256 * KB, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&S, n * 16 * sizeof(W)));
    CK(cudaMalloc(&F, n * 8 * sizeof(W)));
    CK(cudaMalloc(&D, n * NSEG * 8 * sizeof(W)));
    CK(cudaMalloc(&lv, (2 * width) * 8 * sizeof(W)));
    if (with_bc) CK(cudaMalloc(&bc, n * 16 * sizeof(W)));
    CK(cudaFuncSetAttribute(salt_kernel<P, A>, cudaFuncAttributeMaxDynamicSharedMemorySize, TSMEM));
    CK(cudaFuncSetAttribute(seg_kernel<P, A>, cudaFuncAttributeMaxDynamicSharedMemorySize, SEG_SMEM));
  }
  void free_all() {
    for (void* p : {(void*)T, (void*)S, (void*)F, (void*)D, (void*)lv, (void*)bc}) if (p) CK(cudaFree(p));
  }
  // the levels below the top run a launch each (the count is returned), the rest one CTA (top_from() on)
  int top_from() const {
    int l = 1;
    while (l <= depth && (width >> l) > (u64)TOP) ++l;
    return l;
  }
  int tree(bool pads, cudaEvent_t before_top = nullptr) {
    int l = 1, launches = 0;
    for (; l < top_from(); ++l) {
      const u64 c = cdiv_pow2(n, l), lo = pads ? c : 0, hi = pads ? (width >> l) : c;
      if (hi > lo) {
        level_kernel<P, A><<<(unsigned)((hi - lo + THREADS - 1) / THREADS), THREADS, 0, st>>>(lv + level_off(width, l - 1), lv + level_off(width, l), lo, hi, l - 1);
        ++launches;
      }
    }
    if (before_top) CK(cudaEventRecord(before_top, st));
    if (l <= depth) top_kernel<P, A><<<1, THREADS, 0, st>>>(lv, width, n, l, depth, pads ? 1 : 0);
    return launches;
  }
  void salt() {
    salt_kernel<P, A><<<(unsigned)((n + THREADS - 1) / THREADS), THREADS, TSMEM, st>>>(T, n, S, F);
    if (width > n) pad_kernel<P, A><<<(unsigned)((width - n + THREADS - 1) / THREADS), THREADS, 0, st>>>(n, width, lv);
    tree(true);
  }
  // ev (optional): ev[0] after the segments, ev[1] after the leaves, ev[2] before the top CTA; returns the level launches
  int data(const uint8_t* src, cudaEvent_t* ev = nullptr) {
    const u64 segs = n * NSEG, grid = (segs + SEG_THREADS - 1) / SEG_THREADS;
    seg_kernel<P, A><<<(unsigned)grid, SEG_THREADS, SEG_SMEM, st>>>(src, segs, D);
    if (ev) CK(cudaEventRecord(ev[0], st));
    leaf_kernel<P, A><<<(unsigned)((n + THREADS - 1) / THREADS), THREADS, 0, st>>>(n, D, S, F, lv, bc);
    if (ev) CK(cudaEventRecord(ev[1], st));
    return tree(false, ev ? ev[2] : nullptr);
  }
};

struct Params {
  std::vector<uint8_t> raw;
  const uint8_t *t0 = nullptr, *t1 = nullptr;
};
static Params load_params(const std::string& path) {
  Params p;
  p.raw = readf(path);
  const size_t kb0 = 3 * Base::NB + 64, kb1 = 3 * Cand::NB + 64;
  if (p.raw.size() != 4 + 2 * sizeof(HP) + 256 * (kb0 + kb1) || memcmp(p.raw.data(), "HMP1", 4)) {
    fprintf(stderr, "%s: not an HMP1 file for %s\n", path.c_str(), CAND_NAME.c_str());
    exit(2);
  }
  HP hp[2];
  memcpy(hp, p.raw.data() + 4, sizeof hp);
  if (hp[0].iv[0] != 0x6a09e667f3bcc908ull || (Cand::WB == 4 && hp[1].iv[0] >> 32)) {
    fprintf(stderr, "%s: the arms' constants are not where HMP1 puts them\n", path.c_str());
    exit(2);
  }
  CK(cudaMemcpyToSymbol(HPC, hp, sizeof hp));
  p.t0 = p.raw.data() + 4 + 2 * sizeof(HP);
  p.t1 = p.t0 + 256 * kb0;
  return p;
}
static uint8_t* operand(u64 n) {
  const u64 segs = (n * NSEG + SEG_THREADS - 1) / SEG_THREADS * SEG_THREADS;
  uint8_t* src;
  CK(cudaMalloc(&src, segs * SEGB));
  CK(cudaMemset(src, 0, segs * SEGB));
  fill<<<1024, 256>>>(src, n * LEAF, 7u);
  return src;
}

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
static void meta(FILE* f, const cudaDeviceProp& prop) {
  fprintf(f, "\"candidate\": \"%s\", \"cand_kind\": %d, \"cand_r\": %d, \"gpu\": \"%s\", \"sms\": %d, \"cc\": %d, \"fetch\": %d, "
          "\"loader_async\": %d, \"seg_warps\": %d, \"sha_roll\": %d, \"leaf_bytes\": %d, \"seg_bytes\": %d",
          CAND_NAME.c_str(), CAND_KIND, CAND_KIND == 1 ? CAND_R : CAND_KIND == 0 ? 80 : 7, prop.name, prop.multiProcessorCount,
          prop.major * 10 + prop.minor, FETCH, LOADER_ASYNC, SEG_WARPS, SHA_ROLL, LEAF, SEGB);
}

static int dump(const std::string& params, const std::string& dir, u64 n) {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  Params p = load_params(params);
  uint8_t* src = operand(n);
  Arm<Base, 0> b; Arm<Cand, 1> c;
  b.init(n, p.t0, true); c.init(n, p.t1, true);
  b.salt(); b.data(src); c.salt(); c.data(src);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  write_file(dir + "/levels-base.bin", b.lv, (2 * b.width - 1) * 8 * sizeof(Base::W));
  write_file(dir + "/levels-cand.bin", c.lv, (2 * c.width - 1) * 8 * sizeof(Cand::W));
  write_file(dir + "/bc-base.bin", b.bc, n * 16 * sizeof(Base::W));
  write_file(dir + "/bc-cand.bin", c.bc, n * 16 * sizeof(Cand::W));
  FILE* m = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(m, "{");
  meta(m, prop);
  fprintf(m, ", \"leaves\": %llu}\n", (unsigned long long)n);
  fclose(m);
  b.free_all(); c.free_all(); CK(cudaFree(src));
  return 0;
}

static int bench(const std::string& pdir, int reps, const std::vector<u64>& sizes) {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  cudaEvent_t e[6];
  for (auto& x : e) CK(cudaEventCreate(&x));
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  Nv nv;
  auto ms = [&](cudaEvent_t a, cudaEvent_t z) { float t; CK(cudaEventElapsedTime(&t, a, z)); return (double)t; };
  auto capture = [&](auto&& f) {
    cudaGraph_t g;
    cudaGraphExec_t x;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    f();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphDestroy(g));
    return x;
  };
  const char* K[] = {"pre_ms", "step_ms", "salt_ms", "mhz", "pre_graph_ms", "step_graph_ms", "seg_ms", "leaf_ms", "levels_ms",
                     "top_ms"};
  constexpr int NK = sizeof K / sizeof *K;
  for (u64 n : sizes) {
    Params p = load_params(pdir + "/params-" + std::to_string(n) + ".bin");
    uint8_t* src = operand(n);
    Arm<Base, 0> b; Arm<Cand, 1> c;
    b.init(n, p.t0, false); c.init(n, p.t1, false);
    b.st = c.st = s;
    CK(cudaDeviceSynchronize());
    cudaGraphExec_t gb[2] = {capture([&] { b.data(src); }), capture([&] { b.salt(); b.data(src); })};
    cudaGraphExec_t gc[2] = {capture([&] { c.data(src); }), capture([&] { c.salt(); c.data(src); })};
    // eager (each kernel launched from the host as it is reached): the data phase alone (salts precomputed), then a whole
    // step; the same two as CUDA graphs queued behind a spin (GPU time alone, as inside a step's own graph); then the data
    // phase split by kernel, queued the same way
    auto run = [&](auto& arm, cudaGraphExec_t* g) {
      std::vector<double> r(NK);
      CK(cudaEventRecord(e[0], s)); arm.data(src); CK(cudaEventRecord(e[1], s)); CK(cudaEventSynchronize(e[1]));
      r[0] = ms(e[0], e[1]);
      CK(cudaEventRecord(e[0], s)); arm.salt(); CK(cudaEventRecord(e[1], s)); arm.data(src); CK(cudaEventRecord(e[2], s));
      CK(cudaEventSynchronize(e[2]));
      r[1] = ms(e[0], e[2]); r[2] = ms(e[0], e[1]); r[3] = nv.sm();
      for (int m = 0; m < 2; ++m) {
        spin<<<1, 1, 0, s>>>(200000);
        CK(cudaEventRecord(e[0], s)); CK(cudaGraphLaunch(g[m], s)); CK(cudaEventRecord(e[1], s)); CK(cudaEventSynchronize(e[1]));
        r[4 + m] = ms(e[0], e[1]);
      }
      spin<<<1, 1, 0, s>>>(200000);
      CK(cudaEventRecord(e[0], s)); arm.data(src, e + 1); CK(cudaEventRecord(e[4], s)); CK(cudaEventSynchronize(e[4]));
      for (int k = 0; k < 4; ++k) r[6 + k] = ms(e[k], e[k + 1]);
      return r;
    };
    for (int w = 0; w < 3; ++w) { run(b, gb); run(c, gc); }
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    const double bytes = (double)n * LEAF;
    const int launches = 2 + (b.top_from() - 1) + (b.top_from() <= b.depth);
    for (int r = 0; r < reps; ++r) {
      std::vector<double> x, y;
      if (r % 2 == 0) { x = run(b, gb); y = run(c, gc); }
      else { y = run(c, gc); x = run(b, gb); }
      printf("{");
      meta(stdout, prop);
      printf(", \"mode\": \"hm96\", \"rep\": %d, \"leaves\": %llu, \"bytes\": %.0f, \"depth\": %d, \"top_from\": %d, "
             "\"data_launches\": %d",
             r, (unsigned long long)n, bytes, b.depth, b.top_from(), launches);
      for (int k = 0; k < NK; ++k) printf(", \"%s\": %.5f, \"base_%s\": %.5f", K[k], y[k], K[k], x[k]);
      for (const char* m : {"pre", "step", "pre_graph", "step_graph"}) {
        const int k = std::find_if(K, K + NK, [&](const char* q) { return std::string(q) == std::string(m) + "_ms"; }) - K;
        printf(", \"%s_gb_per_s\": %.3f, \"base_%s_gb_per_s\": %.3f", m, bytes / y[k] * 1e-6, m, bytes / x[k] * 1e-6);
      }
      printf("}\n");
      fflush(stdout);
    }
    for (auto g : {gb[0], gb[1], gc[0], gc[1]}) CK(cudaGraphExecDestroy(g));
    b.free_all(); c.free_all(); CK(cudaFree(src));
  }
  return 0;
}

#ifndef HM96_NO_MAIN
int main(int argc, char** argv) {
  if (argc >= 5 && !strcmp(argv[1], "dump")) return dump(argv[2], argv[3], strtoull(argv[4], nullptr, 10));
  if (argc >= 5 && !strcmp(argv[1], "bench")) {
    std::vector<u64> sizes;
    for (char* p = argv[4]; *p;) { sizes.push_back(strtoull(p, &p, 10)); if (*p == ',') ++p; }
    return bench(argv[2], atoi(argv[3]), sizes);
  }
  fprintf(stderr, "usage: hm96 dump PARAMS DIR N | bench PDIR REPS N[,N...]\n");
  return 2;
}
#endif
