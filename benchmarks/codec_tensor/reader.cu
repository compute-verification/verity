// The remote bandwidth driver's receipts and receipt tree on the device: a device reader of the vLLM store
// (integrations/vllm/verity_vllm/store/), `reader.py` its wrapper and gate, and the driver's
// verity/core/protocols/remote_bandwidth/receipts.py the reference. A reader receipts a segment of 4 KiB blocks,
// commits its leaves, builds its subtree and keeps all three for the openings; end_pass takes the subtree's root.
//
//   y_p      SHA512(Z(D(receipt) ‖ frame(seed)) ‖ u64be j ‖ (E_j ⊕ spread(z_j))): a thread a block, 33 compressions
//            from the first block's midstate (the seed's)
//   salt_c   SHAKE256(D(store salt) ‖ key ‖ seed ‖ u64be c), 192 bytes: the store's PRF (`Store.salt`)
//   leaf c   x = SHA512(P(512 m) ‖ y_(16c) ‖ … ‖ y_(16c+m−1)) (sha512/row/v2 of its m receipts), b = x ⊕ M_key·salt (the
//            Hankel matrix of hm96-sha512's key), c' = SHA512(salt prefix ‖ salt), value = SHA512(leaf prefix ‖ b ‖ c'),
//            digest = frame-v3-sha512 leaf(c, c, "hm96-sha512/row/v2", value): a thread a leaf
//   node     frame-v3-sha512 node(level, index, L, R), and pad(rank) past the pass's leaves: a thread a node
//
// A digest is kept as SHA-512's 8 state words: its bytes are their big-endian bytes, and as message words they are
// themselves.
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace rb {

using u32 = uint32_t;
using u64 = uint64_t;

#include "sha512.cuh"

constexpr int BLOCK = 4096, PER_LEAF = 16, SALT = 192, KEY_WORDS = 64;
constexpr int HEAD_LEAF = 0, HEAD_NODE = 1, HEAD_PAD = 2;
constexpr int NODE_LEVELS = 64, NODE_HEAD = 113;

struct Consts {
  u64 receipt_mid[8], row_mid[8], row_mid_last[8], salt_mid[8], leaf_mid[8];
  u64 node_mid[NODE_LEVELS][8];  // a node's first block at each level (`node_digest`)
  u32 key[KEY_WORDS];        // hm96-sha512's key, little-endian: bit t of the key is bit t % 32 of word t / 32
  u64 head[3][16];           // frame-v3 FRAME ‖ u32be |tag| ‖ tag ‖ u64be 64 ‖ domain id, for leaf, node and pad, as
                             // big-endian words, zero past head_len
  u64 schema[4];             // the leaf schema, the same
  u64 salt_block[17];        // SHAKE256's block of salt_in ‖ u64be 0, padded, as little-endian lanes (`salt_of`)
  int head_len[3], schema_len, salt_in_len;
  u64 positions;             // the pass's positions L: ⌈L / 16⌉ leaves, the last of L − 16 (leaves − 1) receipts
};
__constant__ Consts kC;

__host__ __device__ __forceinline__ u32 bswap32(u32 x) {
  return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
}
__host__ __device__ __forceinline__ u64 bswap64(u64 x) {
  return ((u64)bswap32((u32)x) << 32) | bswap32((u32)(x >> 32));
}
// The big-endian message word of 8 bytes held little-endian as (lo, hi).
__device__ __forceinline__ u64 be_word(u32 lo, u32 hi) {
  return ((u64)__byte_perm(lo, 0, 0x0123) << 32) | __byte_perm(hi, 0, 0x0123);
}

// --- receipts ----------------------------------------------------------------------------------------------------

// How a receipt thread reads its block (a thread a block, so a warp's loads fall 4 KiB apart):
//   LOAD_LDG       16 bytes a load through the read-only path
//   LOAD_L2_256    the same, asking L2 to fetch 256 bytes on a miss
//   LOAD_PREFETCH  the same, with an L2 prefetch of the line three compressions ahead
//   LOAD_WARP      the warp loads its 32 blocks' next 128 bytes as whole lines, through shared memory
// and two ceilings that are not receipts: LOAD_HOT reads a 1 MiB window that stays in L2, LOAD_NONE reads nothing.
enum { LOAD_LDG, LOAD_L2_256, LOAD_PREFETCH, LOAD_WARP, LOAD_HOT, LOAD_NONE, LOADS };

__device__ __forceinline__ uint4 ld_l2_256(const uint4 *p) {
  uint4 v;
  asm("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ void receipt_finish(u64 s[8], u64 carry, u64 *__restrict__ out) {
  u64 w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = carry, w[1] = 0x80ULL << 56, w[15] = (u64)(128 + 8 + BLOCK) * 8;
  sha512_compress(s, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = s[i];
}

// One 128-byte chunk d (big-endian words) of the message after u64be j: the compression takes carry ‖ d[0..14].
__device__ __forceinline__ void receipt_step(u64 s[8], u64 &carry, const u64 d[16]) {
  u64 w[16];
  w[0] = carry;
#pragma unroll
  for (int i = 1; i < 16; ++i) w[i] = d[i - 1];
  carry = d[15];
  sha512_compress(s, w);
}

template <bool Z, int L>
__global__ void __launch_bounds__(128) receipts_kernel(const uint4 *__restrict__ data, const uint4 *__restrict__ zs,
                                                       u64 j0, u64 n, u64 *__restrict__ ys) {
  const u64 q = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (q >= n) return;
  const u64 at = L == LOAD_HOT ? (q & 255) : q;
  const uint4 *E = data + at * (BLOCK / 16), *X = Z ? zs + at * (BLOCK / 16) : nullptr;
  u64 s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = kC.receipt_mid[i];
  u64 carry = j0 + q;
#pragma unroll 1
  for (int k = 0; k < BLOCK / 128; ++k) {
    if (L == LOAD_PREFETCH && k + 3 < BLOCK / 128) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(E + 8 * (k + 3)));
      if (Z) asm volatile("prefetch.global.L2 [%0];" ::"l"(X + 8 * (k + 3)));
    }
    u64 d[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint4 v;
      if (L == LOAD_NONE) {
        v = make_uint4((u32)carry ^ (u32)(16 * k + i), (u32)(carry >> 32), (u32)i, (u32)k);
      } else {
        v = L == LOAD_L2_256 ? ld_l2_256(E + 8 * k + i) : __ldg(E + 8 * k + i);
        if (Z) {
          const uint4 x = L == LOAD_L2_256 ? ld_l2_256(X + 8 * k + i) : __ldg(X + 8 * k + i);
          v.x ^= x.x, v.y ^= x.y, v.z ^= x.z, v.w ^= x.w;
        }
      }
      d[2 * i] = be_word(v.x, v.y), d[2 * i + 1] = be_word(v.z, v.w);
    }
    receipt_step(s, carry, d);
  }
  receipt_finish(s, carry, ys + 8 * q);
}

// LOAD_WARP: in each of the 32 steps the warp loads its 32 blocks' next 128-byte chunks, four whole lines a load,
// into shared memory (each block's chunk swizzled by its index, so neither the stores nor the reads conflict), and
// each thread compresses its own block's chunk.
template <bool Z>
__global__ void __launch_bounds__(128) receipts_warp_kernel(const uint4 *__restrict__ data,
                                                            const uint4 *__restrict__ zs, u64 j0, u64 n,
                                                            u64 *__restrict__ ys) {
  __shared__ uint4 buf[4][32 * 8];
  const int lane = threadIdx.x & 31;
  const u64 base = blockIdx.x * (u64)blockDim.x + (threadIdx.x & ~31u), q = base + lane;
  if (base >= n) return;
  uint4 *B = buf[threadIdx.x >> 5];
  u64 s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = kC.receipt_mid[i];
  u64 carry = j0 + q;
#pragma unroll 1
  for (int k = 0; k < BLOCK / 128; ++k) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int b = 4 * i + (lane >> 3), c = lane & 7;
      if (base + b < n) {
        const u64 off = (base + b) * (BLOCK / 16) + 8 * k + c;
        uint4 v = __ldg(data + off);
        if (Z) {
          const uint4 x = __ldg(zs + off);
          v.x ^= x.x, v.y ^= x.y, v.z ^= x.z, v.w ^= x.w;
        }
        B[8 * b + (c ^ (b & 7))] = v;
      }
    }
    __syncwarp();
    u64 d[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint4 v = B[8 * lane + (i ^ (lane & 7))];
      d[2 * i] = be_word(v.x, v.y), d[2 * i + 1] = be_word(v.z, v.w);
    }
    __syncwarp();
    receipt_step(s, carry, d);
  }
  if (q < n) receipt_finish(s, carry, ys + 8 * q);
}

template <bool Z>
void launch_receipts(int mode, const uint4 *data, const uint4 *zs, u64 j0, u64 n, u64 *ys, cudaStream_t s) {
  const unsigned g = (unsigned)((n + 127) / 128);
  switch (mode) {
    case LOAD_LDG: receipts_kernel<Z, LOAD_LDG><<<g, 128, 0, s>>>(data, zs, j0, n, ys); break;
    case LOAD_L2_256: receipts_kernel<Z, LOAD_L2_256><<<g, 128, 0, s>>>(data, zs, j0, n, ys); break;
    case LOAD_PREFETCH: receipts_kernel<Z, LOAD_PREFETCH><<<g, 128, 0, s>>>(data, zs, j0, n, ys); break;
    case LOAD_WARP: receipts_warp_kernel<Z><<<g, 128, 0, s>>>(data, zs, j0, n, ys); break;
    case LOAD_HOT: receipts_kernel<Z, LOAD_HOT><<<g, 128, 0, s>>>(data, zs, j0, n, ys); break;
    default: receipts_kernel<Z, LOAD_NONE><<<g, 128, 0, s>>>(data, zs, j0, n, ys); break;
  }
}

// --- sketch-sha512 receipts (`sketch_receipt.py`) ----------------------------------------------------------------
//
// sketch_kernel: a CTA of 256 threads a block at a time, SK_BATCH blocks a step, thread t holding bytes 16 t … 16 t + 15
// of each and their 16 rows of signs (int8 ±1, 64 registers, loaded once); 64 dp4a a block, then the warp's 16 sums
// reduced by halving (lane l ends with row l / 2) and the 8 warps' added in shared memory. sketch_hash_kernel: a
// thread a block, y = one compression of u64be j ‖ σ_j from the seed block's midstate.

constexpr int SK_THREADS = 256, SK_BATCH = 4, SK_ROWS = 16;
__constant__ u64 kSketchMid[8];

template <bool Z>
__global__ void __launch_bounds__(SK_THREADS, 2) sketch_kernel(const uint4 *__restrict__ data,
                                                               const uint4 *__restrict__ zs, u64 n,
                                                               const uint4 *__restrict__ signs, int *__restrict__ sk) {
  __shared__ int red[SK_BATCH][SK_THREADS / 32][SK_ROWS];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  int sg[SK_ROWS][4];
#pragma unroll
  for (int c = 0; c < SK_ROWS; ++c) {
    const uint4 v = __ldg(signs + c * SK_THREADS + t);
    sg[c][0] = (int)v.x, sg[c][1] = (int)v.y, sg[c][2] = (int)v.z, sg[c][3] = (int)v.w;
  }
  for (u64 b0 = blockIdx.x * (u64)SK_BATCH; b0 < n; b0 += (u64)gridDim.x * SK_BATCH) {
    uint4 d[SK_BATCH];
#pragma unroll
    for (int i = 0; i < SK_BATCH; ++i) {
      d[i] = make_uint4(0, 0, 0, 0);
      if (b0 + i < n) {
        d[i] = __ldcs(data + (b0 + i) * (BLOCK / 16) + t);
        if (Z) {
          const uint4 x = __ldcs(zs + (b0 + i) * (BLOCK / 16) + t);
          d[i].x ^= x.x, d[i].y ^= x.y, d[i].z ^= x.z, d[i].w ^= x.w;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < SK_BATCH; ++i) {
      int a[SK_ROWS];
#pragma unroll
      for (int c = 0; c < SK_ROWS; ++c) {
        a[c] = __dp4a((int)d[i].x, sg[c][0], 0);
        a[c] = __dp4a((int)d[i].y, sg[c][1], a[c]);
        a[c] = __dp4a((int)d[i].z, sg[c][2], a[c]);
        a[c] = __dp4a((int)d[i].w, sg[c][3], a[c]);
      }
#pragma unroll
      for (int w = 8, bit = 16; w >= 1; w /= 2, bit /= 2) {
        const bool hi = lane & bit;
#pragma unroll
        for (int c = 0; c < w; ++c) {
          const int keep = hi ? a[c + w] : a[c], send = hi ? a[c] : a[c + w];
          a[c] = keep + __shfl_xor_sync(~0u, send, bit);
        }
      }
      a[0] += __shfl_xor_sync(~0u, a[0], 1);
      if (!(lane & 1)) red[i][warp][lane >> 1] = a[0];
    }
    __syncthreads();
    if (t < SK_BATCH * SK_ROWS) {
      const int i = t / SK_ROWS, c = t % SK_ROWS;
      int s = 0;
#pragma unroll
      for (int w = 0; w < SK_THREADS / 32; ++w) s += red[i][w][c];
      if (b0 + i < n) sk[(b0 + i) * SK_ROWS + c] = s;
    }
    __syncthreads();
  }
}

__global__ void __launch_bounds__(128) sketch_hash_kernel(const uint4 *__restrict__ sk, u64 j0, u64 n,
                                                          u64 *__restrict__ ys) {
  const u64 q = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (q >= n) return;
  u64 w[16], s[8];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = j0 + q;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = __ldg(sk + 4 * q + i);
    w[1 + 2 * i] = be_word(v.x, v.y), w[2 + 2 * i] = be_word(v.z, v.w);
  }
  w[9] = 0x80ULL << 56, w[15] = (u64)(128 + 8 + 4 * SK_ROWS) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = kSketchMid[i];
  sha512_compress(s, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) ys[8 * q + i] = s[i];
}

// --- framed messages ---------------------------------------------------------------------------------------------

// A message of at most 16 W − 17 bytes as W big-endian words, zero past its n bytes. Its bytes land by compile-time
// word index only (a byte array or a runtime word index would put it on the stack, and every append would be a local
// store that reaches L2).
template <int W>
struct Msg {
  u64 w[W];
  int n;
};
// v's low `bytes` bytes, big-endian, at byte n.
template <int W>
__device__ __forceinline__ void put_be(Msg<W> &m, u64 v, int bytes) {
  const u64 x = bytes == 8 ? v : v << (64 - 8 * bytes);
  const int q = m.n >> 3, r = 8 * (m.n & 7);
  const u64 hi = x >> r, lo = r ? x << (64 - r) : 0;
#pragma unroll
  for (int k = 0; k < W; ++k) m.w[k] |= k == q ? hi : k == q + 1 ? lo : 0;
  m.n += bytes;
}
// A frame-v3 part `_uint(v)`: u64be its length, then v big-endian in the fewest bytes (at least one).
template <int W>
__device__ __forceinline__ void put_uint(Msg<W> &m, u64 v) {
  int len = 1;
  while (len < 8 && v >> (8 * len)) ++len;
  put_be(m, len, 8);
  put_be(m, v, len);
}
template <int W>
__device__ __forceinline__ void put_digest(Msg<W> &m, const u64 *s) {
  put_be(m, 64, 8);
#pragma unroll
  for (int k = 0; k < 8; ++k) put_be(m, s[k], 8);
}
template <int W>
__device__ __forceinline__ void put_head(Msg<W> &m, int t) {
#pragma unroll
  for (int k = 0; k < W; ++k) m.w[k] = k < 16 ? kC.head[t][k] : 0;
  m.n = kC.head_len[t];
}
template <int W>
__device__ __forceinline__ void put_schema(Msg<W> &m) {
  put_be(m, kC.schema_len, 8);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int b = kC.schema_len - 8 * i;
    if (b > 0) put_be(m, b >= 8 ? kC.schema[i] : kC.schema[i] >> (64 - 8 * b), b >= 8 ? 8 : b);
  }
}
// Once m has passed its first block: that block compressed into s, the rest slid down a block (done counts the bytes).
template <int W>
__device__ __forceinline__ void flush_block(Msg<W> &m, u64 (&s)[8], int &done) {
  if (m.n < 128) return;
  sha512_compress(s, m.w);
#pragma unroll
  for (int k = 0; k < W; ++k) m.w[k] = k + 16 < W ? m.w[k + 16] : 0;
  m.n -= 128;
  done += 128;
}
// SHA-512's padding and last blocks of a message whose first done bytes are in s.
template <int W>
__device__ __forceinline__ void sha512_finish(Msg<W> &m, u64 (&s)[8], int done) {
  static_assert(W == 32 || W == 48, "a framed message is two or three blocks");
  const int n = m.n, nb = (n + 17 + 127) / 128;
  put_be(m, 0x80, 1);
  // Unconditional stores at fixed words: a store only where k = 16 nb − 1 folds to one store at a runtime index, and a
  // loop over the blocks runs to the runtime nb with m.w + 16 blk a runtime index.
  const u64 bits = (u64)(done + n) * 8;
  m.w[15] = nb == 1 ? bits : m.w[15];
  m.w[31] = nb == 2 ? bits : m.w[31];
  if constexpr (W == 48) m.w[47] = nb == 3 ? bits : m.w[47];
  sha512_compress(s, m.w);
  if (nb > 1) sha512_compress(s, m.w + 16);
  if constexpr (W == 48)
    if (nb > 2) sha512_compress(s, m.w + 32);
}
template <int W>
__device__ __forceinline__ void sha512_msg(Msg<W> &m, u64 (&s)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = kSha512Iv[i];
  sha512_finish(m, s, 0);
}

// --- leaves ------------------------------------------------------------------------------------------------------

__constant__ u64 kKeccakRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL, 0x000000000000808BULL,
    0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL, 0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
__constant__ int kKeccakRot[25] = {0,  1,  62, 28, 27, 36, 44, 6,  55, 20, 3,  10, 43,
                                   25, 39, 41, 45, 15, 21, 8,  18, 2,  61, 56, 14};

__device__ __forceinline__ u64 rotl64(u64 x, int n) { return n ? (x << n) | (x >> (64 - n)) : x; }

__device__ void keccak_f(u64 (&A)[25]) {
#pragma unroll 1
  for (int r = 0; r < 24; ++r) {
    u64 C[5], B[25];
#pragma unroll
    for (int x = 0; x < 5; ++x) C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
#pragma unroll
    for (int x = 0; x < 5; ++x) {
      const u64 D = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
#pragma unroll
      for (int y = 0; y < 5; ++y) A[x + 5 * y] ^= D;
    }
#pragma unroll
    for (int x = 0; x < 5; ++x)
#pragma unroll
      for (int y = 0; y < 5; ++y) B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(A[x + 5 * y], kKeccakRot[x + 5 * y]);
#pragma unroll
    for (int x = 0; x < 5; ++x)
#pragma unroll
      for (int y = 0; y < 5; ++y) A[x + 5 * y] = B[x + 5 * y] ^ (~B[(x + 1) % 5 + 5 * y] & B[(x + 2) % 5 + 5 * y]);
    A[0] ^= kKeccakRC[r];
  }
}

// salt_c as 24 little-endian lanes: SHAKE256 of kC.salt_in ‖ u64be c (one block at rate 136), squeezed to 192 bytes.
__device__ void salt_of(u64 c, u64 (&S)[24]) {
  const int q = kC.salt_in_len >> 3, r = 8 * (kC.salt_in_len & 7);
  const u64 x = bswap64(c);   // u64be c's bytes as a little-endian lane
  const u64 lo = x << r, hi = r ? x >> (64 - r) : 0;
  u64 A[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) A[i] = i < 17 ? kC.salt_block[i] ^ (i == q ? lo : i == q + 1 ? hi : 0) : 0;
  keccak_f(A);
#pragma unroll
  for (int i = 0; i < 17; ++i) S[i] = A[i];
  keccak_f(A);
#pragma unroll
  for (int i = 17; i < 24; ++i) S[i] = A[i - 17];
}

// Thread tid's digest d to out + 8 tid, tid < n. A thread's own eight 8-byte stores, 64 bytes apart across the warp,
// would fill each 32-byte sector a quarter at a time (4× the digests' bytes in L2 write sectors), so lanes 2a and
// 2a + 1 write their 128 bytes as four whole sectors, each half from one lane. Every lane of the warp calls it.
__device__ __forceinline__ void store_digest(u64 *__restrict__ out, u64 tid, u64 n, const u64 (&d)[8]) {
  const int odd = threadIdx.x & 1;
  u64 mine[4], theirs[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int w0 = 4 * (i >> 1) + (i & 1);
    mine[i] = odd ? d[w0 + 2] : d[w0];
    theirs[i] = __shfl_xor_sync(0xffffffffu, odd ? d[w0] : d[w0 + 2], 1);
  }
  const u64 te = tid & ~(u64)1;
  if (te + 1 < n) {
    ulonglong2 *o = reinterpret_cast<ulonglong2 *>(out + 8 * te);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const bool own = (s < 2) != (odd == 1);
      o[2 * s + odd] = own ? make_ulonglong2(mine[2 * (s & 1)], mine[2 * (s & 1) + 1])
                           : make_ulonglong2(theirs[2 * (s & 1)], theirs[2 * (s & 1) + 1]);
    }
  } else if (tid < n) {
#pragma unroll
    for (int i = 0; i < 8; ++i) out[8 * tid + i] = d[i];
  }
}

// Leaves [c0, c0 + nl) of the pass from their receipts (ys, leaf c0's first), into dig; salts (192 bytes a leaf) if
// asked for.
__global__ void __launch_bounds__(128) leaves_kernel(const u64 *__restrict__ ys, u64 c0, u64 nl, u64 *__restrict__ dig,
                                                     uint8_t *__restrict__ salts) {
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, t = tid < nl ? tid : nl - 1;
  const u64 c = c0 + t, leaves = (kC.positions + PER_LEAF - 1) / PER_LEAF;
  const int m = c + 1 < leaves ? PER_LEAF : (int)(kC.positions - PER_LEAF * c);
  const u64 *Y = ys + t * PER_LEAF * 8;
  u64 x[8], w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = m == PER_LEAF ? kC.row_mid[i] : kC.row_mid_last[i];
  const int words = 8 * m, nb = (64 * m + 17 + 127) / 128;
#pragma unroll 1
  for (int blk = 0; blk < nb; ++blk) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i;
      w[i] = q < words ? Y[q] : q == words ? 0x80ULL << 56 : 0;
    }
    if (blk == nb - 1) w[15] = (u64)(128 + 64 * m) * 8;
    sha512_compress(x, w);
  }

  u64 S[24];
  salt_of(c, S);
  u32 Ys[48];
#pragma unroll
  for (int i = 0; i < 24; ++i) Ys[2 * i] = (u32)S[i], Ys[2 * i + 1] = (u32)(S[i] >> 32);
  // M_key · salt over GF(2): bit i = XOR_j salt_j key_(i + j). For each shift b the key's 63 shifted words serve all
  // 16 output words (output word o, bit b reads shifted words o … o + 47).
  u32 M[16];
#pragma unroll
  for (int o = 0; o < 16; ++o) M[o] = 0;
#pragma unroll 1
  for (int b = 0; b < 32; ++b) {
    u32 kb[63];
#pragma unroll
    for (int u = 0; u < 63; ++u) kb[u] = __funnelshift_r(kC.key[u], kC.key[u + 1], b);
#pragma unroll
    for (int o = 0; o < 16; ++o) {
      u32 acc = 0;
#pragma unroll
      for (int v = 0; v < 48; ++v) acc ^= kb[o + v] & Ys[v];
      M[o] |= (u32)(__popc(acc) & 1) << b;
    }
  }

  u64 cd[8], val[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cd[i] = kC.salt_mid[i];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = bswap64(S[i]);
  sha512_compress(cd, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = i < 8 ? bswap64(S[16 + i]) : 0;
  w[8] = 0x80ULL << 56, w[15] = (u64)(128 + SALT) * 8;
  sha512_compress(cd, w);

#pragma unroll
  for (int i = 0; i < 8; ++i) val[i] = kC.leaf_mid[i];
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = x[i] ^ bswap64(((u64)M[2 * i + 1] << 32) | M[2 * i]), w[8 + i] = cd[i];
  sha512_compress(val, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = 0x80ULL << 56, w[15] = 256 * 8;
  sha512_compress(val, w);

  Msg<32> msg;
  u64 d[8];
  int done = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = kSha512Iv[i];
  put_head(msg, HEAD_LEAF);
  put_uint(msg, c);
  put_uint(msg, c);
  flush_block(msg, d, done);   // past 128 bytes (head 113, two uints of at least 9), at most 129 to come
  put_schema(msg);
  put_digest(msg, val);
  sha512_finish(msg, d, done);
  store_digest(dig, tid, nl, d);
  if (salts && tid < nl)
    for (int i = 0; i < 24; ++i)
      for (int b = 0; b < 8; ++b) salts[t * SALT + 8 * i + b] = (uint8_t)(S[i] >> (8 * b));
}

// --- nodes -------------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(128) pads_kernel(u64 r0, u64 n, u64 *__restrict__ out) {
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, t = tid < n ? tid : n - 1;
  Msg<32> msg;
  put_head(msg, HEAD_PAD);
  put_uint(msg, r0 + t);
  u64 d[8];
  sha512_msg(msg, d);
  store_digest(out, tid, n, d);
}

// Nodes [i0, i0 + n) of the level above level h: node i over in[2 (i − i0)] and in[2 (i − i0) + 1].
// A node's message is head ‖ u64be 1 ‖ h ‖ u64be len(i) ‖ i ‖ u64be 64 ‖ L ‖ u64be 64 ‖ R. For h < NODE_LEVELS its
// first 128 bytes (the 113-byte head, u64be 1, h and the first 6 bytes of u64be len(i)) are fixed by h, so the node
// starts from kC.node_mid[h], and the rest, 146 + len(i) bytes, is built as words: two compressions in all.
__device__ __forceinline__ void node_digest(const u64 *mid, u64 i, const u64 *L, const u64 *R, u64 (&s)[8]) {
  int len = 1;
  while (len < 8 && i >> (8 * len)) ++len;
  const int a = 2 + len, q = a >> 3, r = 8 * (a & 7);   // the tail's bytes before L's frame, at most 8
  u64 U[19];
  U[0] = 64;
#pragma unroll
  for (int k = 0; k < 8; ++k) U[1 + k] = L[k], U[10 + k] = R[k];
  U[9] = 64, U[18] = 0x80ULL << 56;
  const u64 P = ((u64)len << (8 * len)) | i;            // 0x00 ‖ len ‖ i, a bytes big-endian
  u64 w[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) {
    const u64 u0 = k < 19 ? U[k] : 0, u1 = k >= 1 && k <= 19 ? U[k - 1] : 0;
    w[k] = q ? u1 : (u0 >> r) | (u1 << (64 - r));   // q = 1 only at a = 8, r = 0; else 24 ≤ r ≤ 56
  }
  w[0] |= a == 8 ? P : P << (64 - 8 * a);
  w[31] = (u64)(128 + 146 + len) * 8;
#pragma unroll
  for (int k = 0; k < 8; ++k) s[k] = mid[k];
  sha512_compress(s, w);
  sha512_compress(s, w + 16);
}

__global__ void __launch_bounds__(128) level_kernel(const u64 *__restrict__ in, u64 *__restrict__ out, int h, u64 i0,
                                                    u64 n) {
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, t = tid < n ? tid : n - 1;
  u64 d[8];
  if (h < NODE_LEVELS && i0 + t < (1ULL << 48)) {
    node_digest(kC.node_mid[h], i0 + t, in + 16 * t, in + 16 * t + 8, d);
  } else {
    Msg<48> msg;
    put_head(msg, HEAD_NODE);
    put_uint(msg, (u64)h);
    put_uint(msg, i0 + t);
    put_digest(msg, in + 16 * t);
    put_digest(msg, in + 16 * t + 8);
    sha512_msg(msg, d);
  }
  store_digest(out, tid, n, d);
}

__global__ void __launch_bounds__(256) read_kernel(const uint4 *__restrict__ p, u64 n16, unsigned long long *xr) {
  u32 acc = 0;
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n16; i += (u64)gridDim.x * blockDim.x) {
    const uint4 v = __ldcs(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) atomicXor(xr, 1ull);
}

void midstate(const uint8_t *block, u64 *out) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) {
    u64 v = 0;
    for (int b = 0; b < 8; ++b) v = (v << 8) | block[8 * i + b];
    w[i] = v;
  }
  for (int i = 0; i < 8; ++i) out[i] = SHA512_IV_HOST[i];
  sha512_compress(out, w);
}

void be_words(const uint8_t *b, int bytes, u64 *out) {
  for (int i = 0; i < bytes / 8; ++i) {
    u64 v = 0;
    for (int k = 0; k < 8; ++k) v = (v << 8) | b[8 * i + k];
    out[i] = v;
  }
}

int head(uint8_t *out, const char *tag, const uint8_t *domain_id) {
  static const char FRAME[] = "veritor/protocol/merkle/frame/v3";
  int n = 0;
  for (size_t i = 0; i < sizeof FRAME; ++i) out[n++] = (uint8_t)FRAME[i];   // with its zero byte
  const int t = (int)strlen(tag);
  for (int i = 3; i >= 0; --i) out[n++] = (uint8_t)(t >> (8 * i));
  for (int i = 0; i < t; ++i) out[n++] = (uint8_t)tag[i];
  for (int i = 7; i >= 0; --i) out[n++] = (uint8_t)(64 >> (8 * i));
  for (int i = 0; i < 64; ++i) out[n++] = domain_id[i];
  return n;
}

inline unsigned grid(u64 n, int threads) { return (unsigned)((n + threads - 1) / threads); }

}  // namespace rb

using namespace rb;

extern "C" {

// One pass's constants. The 128-byte blocks are the reference's own: prefix(seed), sha512_row_v2_prefix(8192) and
// (8 · 64 m) for the pass's last leaf of m receipts, hm96's salt_prefix and leaf_prefix(key); key is hm96-sha512's
// 256-byte key; salt_in is D(store salt) ‖ the store's key ‖ seed.
int rb_set(const uint8_t *receipt_prefix, const uint8_t *row_prefix, const uint8_t *row_prefix_last,
           const uint8_t *salt_prefix, const uint8_t *leaf_prefix, const uint8_t *key, const uint8_t *domain_id,
           const uint8_t *schema, int schema_len, const uint8_t *salt_in, int salt_in_len, uint64_t positions,
           cudaStream_t s) {
  if (schema_len < 1 || schema_len > 32 || salt_in_len < 1 || salt_in_len + 9 > 136 || positions < 1) return 1;
  static Consts c;
  midstate(receipt_prefix, c.receipt_mid);
  midstate(row_prefix, c.row_mid);
  midstate(row_prefix_last, c.row_mid_last);
  midstate(salt_prefix, c.salt_mid);
  midstate(leaf_prefix, c.leaf_mid);
  for (int i = 0; i < KEY_WORDS; ++i)
    c.key[i] = (u32)key[4 * i] | (u32)key[4 * i + 1] << 8 | (u32)key[4 * i + 2] << 16 | (u32)key[4 * i + 3] << 24;
  static const char *const TAGS[3] = {"leaf", "node", "pad"};
  uint8_t heads[3][128] = {};
  for (int t = 0; t < 3; ++t) {
    c.head_len[t] = head(heads[t], TAGS[t], domain_id);
    be_words(heads[t], 128, c.head[t]);
  }
  if (c.head_len[HEAD_NODE] != NODE_HEAD) return 1;
  for (int h = 0; h < NODE_LEVELS; ++h) {
    uint8_t first[128] = {};
    memcpy(first, heads[HEAD_NODE], NODE_HEAD);
    first[NODE_HEAD + 7] = 1, first[NODE_HEAD + 8] = (uint8_t)h;
    midstate(first, c.node_mid[h]);
  }
  uint8_t sch[32] = {};
  memcpy(sch, schema, schema_len);
  be_words(sch, 32, c.schema);
  c.schema_len = schema_len;
  uint8_t blk[136] = {};
  memcpy(blk, salt_in, salt_in_len);
  blk[salt_in_len + 8] ^= 0x1F;
  blk[135] ^= 0x80;
  for (int i = 0; i < 17; ++i) {
    u64 v = 0;
    for (int b = 7; b >= 0; --b) v = (v << 8) | blk[8 * i + b];
    c.salt_block[i] = v;
  }
  c.salt_in_len = salt_in_len;
  c.positions = positions;
  return cudaMemcpyToSymbolAsync(kC, &c, sizeof c, 0, cudaMemcpyHostToDevice, s) ? 4 : 0;
}

// The receipts of n blocks at data (block q at byte 4096 q, store index j0 + q) into ys (8 words a receipt); zs holds
// each block's spread z (4096 bytes a block) or is null for z's of zero bytes; mode is how a block is read (LOAD_*).
int rb_receipts(const void *data, const void *zs, uint64_t j0, uint64_t n, uint64_t *ys, int mode, cudaStream_t s) {
  if (!data || !ys || n < 1 || mode < 0 || mode >= LOADS) return 1;
  if (zs)
    launch_receipts<true>(mode, (const uint4 *)data, (const uint4 *)zs, j0, n, ys, s);
  else
    launch_receipts<false>(mode, (const uint4 *)data, nullptr, j0, n, ys, s);
  return cudaGetLastError() ? 4 : 0;
}

// sketch-sha512's pass constant: the seed block Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed)).
int rb_sketch_set(const uint8_t *sketch_prefix, cudaStream_t s) {
  static u64 mid[8];
  midstate(sketch_prefix, mid);
  return cudaMemcpyToSymbolAsync(kSketchMid, mid, sizeof mid, 0, cudaMemcpyHostToDevice, s) ? 4 : 0;
}

// The sketches of n blocks at data (each XORed with its spread z in zs, or not when zs is null) into sk (16 int32 a
// block) under signs (16 rows of 4096 int8 ±1 on the device), on ctas CTAs (0: as many as are resident at once).
int rb_sketch(const void *data, const void *zs, uint64_t n, const void *signs, int *sk, int ctas, cudaStream_t s) {
  if (!data || !signs || !sk || n < 1 || ctas < 0) return 1;
  if (!ctas) {
    int dev, sms, per;
    if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)) return 4;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, zs ? sketch_kernel<true> : sketch_kernel<false>,
                                                      SK_THREADS, 0))
      return 4;
    ctas = sms * per;
  }
  const u64 most = (n + SK_BATCH - 1) / SK_BATCH;
  const unsigned g = (unsigned)(most < (u64)ctas ? most : (u64)ctas);
  if (zs)
    sketch_kernel<true><<<g, SK_THREADS, 0, s>>>((const uint4 *)data, (const uint4 *)zs, n, (const uint4 *)signs, sk);
  else
    sketch_kernel<false><<<g, SK_THREADS, 0, s>>>((const uint4 *)data, nullptr, n, (const uint4 *)signs, sk);
  return cudaGetLastError() ? 4 : 0;
}

// The sketch-sha512 receipts of n sketches (store indices j0 …) into ys (8 words a receipt).
int rb_sketch_hash(const int *sk, uint64_t j0, uint64_t n, uint64_t *ys, cudaStream_t s) {
  if (!sk || !ys || n < 1) return 1;
  sketch_hash_kernel<<<grid(n, 128), 128, 0, s>>>((const uint4 *)sk, j0, n, ys);
  return cudaGetLastError() ? 4 : 0;
}

int rb_leaves(const uint64_t *ys, uint64_t c0, uint64_t nl, uint64_t *dig, uint8_t *salts, cudaStream_t s) {
  if (!ys || !dig || nl < 1) return 1;
  leaves_kernel<<<grid(nl, 128), 128, 0, s>>>(ys, c0, nl, dig, salts);
  return cudaGetLastError() ? 4 : 0;
}

int rb_pads(uint64_t r0, uint64_t n, uint64_t *out, cudaStream_t s) {
  if (!out || n < 1) return 1;
  pads_kernel<<<grid(n, 128), 128, 0, s>>>(r0, n, out);
  return cudaGetLastError() ? 4 : 0;
}

int rb_level(const uint64_t *in, uint64_t *out, int h, uint64_t i0, uint64_t n, cudaStream_t s) {
  if (!in || !out || n < 1 || h < 0) return 1;
  level_kernel<<<grid(n, 128), 128, 0, s>>>(in, out, h, i0, n);
  return cudaGetLastError() ? 4 : 0;
}

// Every level above n digests of level 0 (a power of two, the first leaf c0) into out, level after level up to the
// root: n − 1 nodes in one call, so the narrow levels wait on the device and not on the host.
int rb_tree(const uint64_t *in, uint64_t n, uint64_t c0, uint64_t *out, cudaStream_t s) {
  if (!in || !out || n < 2 || n & (n - 1)) return 1;
  for (int h = 0; n > 1; ++h, n /= 2) {
    level_kernel<<<grid(n / 2, 128), 128, 0, s>>>(in, out, h, c0 >> (h + 1), n / 2);
    in = out;
    out += 8 * (n / 2);
  }
  return cudaGetLastError() ? 4 : 0;
}

int rb_read(const void *p, uint64_t bytes, unsigned long long *xr, int blocks, cudaStream_t s) {
  if (!p || !xr || bytes % 16 || blocks < 1) return 1;
  read_kernel<<<blocks, 256, 0, s>>>((const uint4 *)p, bytes / 16, xr);
  return cudaGetLastError() ? 4 : 0;
}

}  // extern "C"
