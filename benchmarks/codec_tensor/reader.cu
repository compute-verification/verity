// The remote bandwidth driver's receipts and receipt tree on the device: a device reader of the vLLM store
// (integrations/vllm/verity_vllm/protocols/remote_bandwidth/), `reader.py` its wrapper and gate, and the driver's
// verity/core/protocols/remote_bandwidth/receipts.py the reference. A reader receipts a segment of 4 KiB blocks,
// commits its leaves, builds its subtree and keeps all three for the openings; end_pass takes the subtree's root.
//
//   y_p      SHA512(Z(D(receipt) ‖ frame(seed)) ‖ u64be j ‖ (E_j ⊕ spread(z_j))): a thread a block, 33 compressions
//            from the first block's midstate (the seed's)
//   salt_c   XOF(D(store salt) ‖ key ‖ seed ‖ u64be c)[0 : 192], XOF(m) = SHA512(m ‖ u32be 0) ‖ SHA512(m ‖ u32be 1)
//            ‖ …: the store's PRF (`NodeMemory.salt`, `store.store_salt`)
//   leaf c   x = SHA512(P(512 m) ‖ y_(16c) ‖ … ‖ y_(16c+m−1)) (sha512/row/v2 of its m receipts), b = x ⊕ M_key·salt (the
//            Hankel matrix of hm96-sha512's key), c' = SHA512(salt prefix ‖ salt), value = SHA512(leaf prefix ‖ b ‖ c'),
//            digest = frame-v3-sha512 leaf(c, c, "hm96-sha512/row/v2", value): a thread a leaf
//   node     frame-v3-sha512 node(level, index, L, R), and pad(rank) past the pass's leaves: a thread a node
//
// The hm96 leaf after its salt, the frame-v3 leaf, the pad and the node are the GPU's one row commitment (`rc`,
// verity/ml/kernels/row_commit.cuh), from the pass's constants kL (rc's leaf and pad templates under the pass's domain
// and hm96's constants), kNodeTpl (a node's last two blocks), gNib (M_key's nibble table) and kC.node_mid; what is the
// reader's own is the receipts, the row digest, the salts and which thread hashes what. Positions are at most 2^36,
// so every leaf, pad and node index is below 2^32.
//
// A digest is kept as SHA-512's 8 state words: its bytes are their big-endian bytes, and as message words they are
// themselves.
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#include "sha512.cuh"
#include "../../verity/ml/kernels/row_commit.cuh"

namespace rb {

using u32 = uint32_t;
using u64 = uint64_t;

constexpr int BLOCK = 4096, PER_LEAF = 16, SALT = 192;
constexpr int NODE_LEVELS = 64;
constexpr u64 MAX_INDEX = 1ULL << 32;   // leaf, pad and node indices: rc's uint(index) is at most 4 bytes

struct Consts {
  u64 receipt_mid[8], row_mid[8], row_mid_last[8];
  u64 node_mid[NODE_LEVELS][8];  // a node's first block at each level (rc::node_mid)
  u64 kw_xpad[80];           // a leaf of PER_LEAF receipts' row/v2 all-padding last block, scheduled (session.cu's leaves)
  u64 salt_block[32];        // salt_in ‖ u64be 0 ‖ u32be 0, padded to SHA-512's one or two blocks, as big-endian
                             // words (`salt_of`)
  int salt_in_len;
  int zero;                  // no padding: two passes' constants compare as bytes (rb_consts)
  u64 positions;             // the pass's positions L: ⌈L / 16⌉ leaves, the last of L − 16 (leaves − 1) receipts
};
__constant__ Consts kC;
// rc's frame-v3 leaf and pad templates under the pass's domain (FrameK) and hm96's constants (Hm96K)
struct Leaf : rc::FrameK, rc::Hm96K {};
__constant__ Leaf kL;
__constant__ u64 kNodeTpl[rc::MAX_NB][32];
__device__ __align__(16) uint8_t gNib[rc::TN_BYTES];

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

// --- sketch-sha512 receipts (`receipts.SketchReceipter`) ---------------------------------------------------------
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

// --- sketch-sha512 at k_s = 64 rows (a block's 2^−64) -------------------------------------------------------------
//
// The sketch as an int8 tensor-core product, σ (blocks × 64) = E (blocks × 4096) · Sᵀ, by mma.m16n8k32: a warp takes
// 16 MT blocks (MT m-tiles) through all 4096 bytes, its 8 n-tiles the 64 rows. K runs over a block's bytes in an order
// of the kernel's own: at p = 0 … 63 lane (g, t) holds 16 bytes of block g and of block g + 8 at 64 p + 16 t, and their
// words (x, y) are A's (a0, a2) and (a1, a3) of k-step 2 p, (z, w) the same of k-step 2 p + 1. B is S's row 8 n + g at
// the same bytes. The signs sit in shared memory as 32 KiB of bits (reader.py's `pack64`): lane (g, t)'s uint4 at p
// holds, in bit 8 i + r of word R / 8, element i of its B register R = 4 n + 2 s + b (n-tile n, k-step 2 p + s, b0 or
// b1) at r = R mod 8, and a register is one IMAD, 0x01010101 + 0xFE · ((word >> r) & 0x01010101): a bit of 1 is −1.
//
// sketch64_kernel writes σ (64 int32 a block); sketch64_hash_kernel is a thread a block, y = three compressions of
// u64be j ‖ σ_j (264 bytes) from the seed block's midstate. sketch64_receipts_kernel is both in one: each warp stages
// its 32 blocks' σ in shared memory and each lane hashes one, so σ never leaves the SM.

constexpr int SK64_ROWS = 64, SK64_P = BLOCK / 64, SK64_WARPS = 4, SK64R_WARPS = 8;
constexpr int SK64_SIGNS = SK64_P * 32;                            // uint4s of packed signs: 32 KiB
constexpr int SK64R_SMEM = SK64_SIGNS * 16 + SK64R_WARPS * 32 * SK64_ROWS * 4;

__device__ __forceinline__ void imma16832(int (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ u32 sign_reg(u32 word, int r) { return 0x01010101u + 0xFEu * ((word >> r) & 0x01010101u); }

// Lane (g, t)'s 16 bytes at p of blocks b0 + 16 mt + g (+ 8 for hi), zero past n, each XORed with its z when Z. A warp
// reads 64 bytes of each of 16 blocks at p, so the loads ask L2 for 256: p + 1 … p + 3 then hit it.
template <bool Z, int MT>
__device__ __forceinline__ void sk64_load(uint4 (&d)[MT][2], const uint4 *__restrict__ data,
                                          const uint4 *__restrict__ zs, u64 b0, u64 n, int p, int g, int t) {
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
      const u64 blk = b0 + 16 * mt + g + 8 * hi;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (blk < n) {
        const u64 at = blk * (BLOCK / 16) + 4 * p + t;
        v = ld_l2_256(data + at);
        if (Z) {
          const uint4 x = ld_l2_256(zs + at);
          v.x ^= x.x, v.y ^= x.y, v.z ^= x.z, v.w ^= x.w;
        }
      }
      d[mt][hi] = v;
    }
  }
}

// σ of the warp's blocks b0 … b0 + 16 MT − 1 into acc: lane (g, t) ends with rows 8 n + 2 t, +1 of block 16 mt + g in
// acc[mt][n][0, 1] and of block 16 mt + g + 8 in acc[mt][n][2, 3].
template <bool Z, int MT>
__device__ __forceinline__ void sk64_tile(int (&acc)[MT][8][4], const uint4 *__restrict__ sg,
                                          const uint4 *__restrict__ data, const uint4 *__restrict__ zs, u64 b0, u64 n,
                                          int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0;
  uint4 cur[MT][2], nxt[MT][2];
  sk64_load<Z, MT>(cur, data, zs, b0, n, 0, g, t);
#pragma unroll 1
  for (int p = 0; p < SK64_P; ++p) {
    if (p + 1 < SK64_P) sk64_load<Z, MT>(nxt, data, zs, b0, n, p + 1, g, t);
    const uint4 w = sg[p * 32 + lane];
    const u32 ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const u32 word = ws[nt >> 1];
      const int r = 4 * (nt & 1);
      const u32 b00 = sign_reg(word, r), b01 = sign_reg(word, r + 1);
      const u32 b10 = sign_reg(word, r + 2), b11 = sign_reg(word, r + 3);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const uint4 &lo = cur[mt][0], &hi = cur[mt][1];
        imma16832(acc[mt][nt], lo.x, hi.x, lo.y, hi.y, b00, b01);
        imma16832(acc[mt][nt], lo.z, hi.z, lo.w, hi.w, b10, b11);
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) cur[mt][0] = nxt[mt][0], cur[mt][1] = nxt[mt][1];
  }
}

template <int MT>
__device__ __forceinline__ void sk64_store(const int (&acc)[MT][8][4], int *__restrict__ sk, u64 b0, u64 n, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const u64 blk = b0 + 16 * mt + g;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      if (blk < n) *(int2 *)(sk + blk * SK64_ROWS + 8 * nt + 2 * t) = make_int2(acc[mt][nt][0], acc[mt][nt][1]);
      if (blk + 8 < n)
        *(int2 *)(sk + (blk + 8) * SK64_ROWS + 8 * nt + 2 * t) = make_int2(acc[mt][nt][2], acc[mt][nt][3]);
    }
  }
}

// y_j from σ_j's 16 uint4s (`at(q)`, q = 0 … 15) into out.
template <class At>
__device__ __forceinline__ void sk64_receipt(At at, u64 j, u64 *__restrict__ out) {
  u64 w[16], s[8], carry;
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = kSketchMid[i];
  w[0] = j;
#pragma unroll
  for (int b = 0; b < 2; ++b) {
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const uint4 v = at(8 * b + q);
      const u64 x = be_word(v.x, v.y), y = be_word(v.z, v.w);
      if (q < 7) w[1 + 2 * q] = x, w[2 + 2 * q] = y;
      else w[15] = x, carry = y;
    }
    sha512_compress(s, w);
    w[0] = carry;
  }
#pragma unroll
  for (int i = 1; i < 15; ++i) w[i] = 0;
  w[1] = 0x80ULL << 56, w[15] = (u64)(128 + 8 + 4 * SK64_ROWS) * 8;
  sha512_compress(s, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = s[i];
}

template <bool Z, int MT>
__global__ void __launch_bounds__(32 * SK64_WARPS, MT > 2 ? 2 : 3) sketch64_kernel(const uint4 *__restrict__ data,
                                                                                const uint4 *__restrict__ zs, u64 n,
                                                                                const uint4 *__restrict__ signs,
                                                                                int *__restrict__ sk) {
  __shared__ uint4 sg[SK64_SIGNS];
  for (int i = threadIdx.x; i < SK64_SIGNS; i += blockDim.x) sg[i] = __ldg(signs + i);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const u64 tiles = (n + 16 * MT - 1) / (16 * MT);
  for (u64 tile = blockIdx.x * (u64)SK64_WARPS + (threadIdx.x >> 5); tile < tiles;
       tile += (u64)gridDim.x * SK64_WARPS) {
    int acc[MT][8][4];
    sk64_tile<Z, MT>(acc, sg, data, zs, tile * 16 * MT, n, lane);
    sk64_store<MT>(acc, sk, tile * 16 * MT, n, lane);
  }
}

__global__ void __launch_bounds__(128) sketch64_hash_kernel(const uint4 *__restrict__ sk, u64 j0, u64 n,
                                                            u64 *__restrict__ ys) {
  const u64 q = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (q >= n) return;
  const uint4 *S = sk + q * (SK64_ROWS / 4);
  sk64_receipt([&](int i) { return __ldg(S + i); }, j0 + q, ys + 8 * q);
}

// A warp's 32 blocks' σ staged at row r ^ 8 (block & 3) (the C fragments' int2 stores then hit 32 banks a phase).
template <bool Z>
__global__ void __launch_bounds__(32 * SK64R_WARPS, 1) sketch64_receipts_kernel(const uint4 *__restrict__ data,
                                                                              const uint4 *__restrict__ zs, u64 j0,
                                                                              u64 n, const uint4 *__restrict__ signs,
                                                                              int *__restrict__ sk,
                                                                              u64 *__restrict__ ys) {
  extern __shared__ uint4 smem[];
  uint4 *sg = smem;
  for (int i = threadIdx.x; i < SK64_SIGNS; i += blockDim.x) sg[i] = __ldg(signs + i);
  __syncthreads();
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  int *st = (int *)(smem + SK64_SIGNS) + (threadIdx.x >> 5) * 32 * SK64_ROWS;
  const u64 tiles = (n + 31) / 32;
  for (u64 tile = blockIdx.x * (u64)SK64R_WARPS + (threadIdx.x >> 5); tile < tiles;
       tile += (u64)gridDim.x * SK64R_WARPS) {
    const u64 b0 = tile * 32;
    int acc[2][8][4];
    sk64_tile<Z, 2>(acc, sg, data, zs, b0, n, lane);
    if (sk) sk64_store<2>(acc, sk, b0, n, lane);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int blk = 16 * mt + g;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int r = (8 * nt + 2 * t) ^ (8 * (g & 3));
        *(int2 *)(st + blk * SK64_ROWS + r) = make_int2(acc[mt][nt][0], acc[mt][nt][1]);
        *(int2 *)(st + (blk + 8) * SK64_ROWS + r) = make_int2(acc[mt][nt][2], acc[mt][nt][3]);
      }
    }
    __syncwarp();
    if (b0 + lane < n) {
      const uint4 *S = (const uint4 *)(st + lane * SK64_ROWS);
      const int x = 2 * (lane & 3);
      sk64_receipt([&](int i) { return S[i ^ x]; }, j0 + b0 + lane, ys + 8 * (b0 + lane));
    }
    __syncwarp();
  }
}

// --- leaves ------------------------------------------------------------------------------------------------------

// salt_c as 24 little-endian lanes: SHA512(salt_in ‖ u64be c ‖ u32be k) for k = 0, 1, 2 over kC.salt_block, k in the
// counter's last byte (its others are zero). When that byte is in the second block, the first is every output's,
// compressed once.
__device__ void salt_of(u64 c, u64 (&S)[24]) {
  const int L = kC.salt_in_len, q = L >> 3, r = 8 * (L & 7);
  const int nb = (L + 12 + 17 + 127) / 128, kw = (L + 11) >> 3, ks = 8 * (7 - ((L + 11) & 7)), kb = kw >> 4;
  u64 m[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) m[i] = kC.salt_block[i] | (i == q ? c >> r : r && i == q + 1 ? c << (64 - r) : 0);
  u64 mid[8], w[16];
  sx::iv(mid);
  if (kb) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = m[i];
    sha512_compress(mid, w);
  }
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    u64 s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = mid[i];
#pragma unroll 1
    for (int b = kb; b < nb; ++b) {
#pragma unroll
      for (int i = 0; i < 16; ++i) w[i] = (b ? m[16 + i] : m[i]) | (16 * b + i == kw ? (u64)k << ks : 0);
      sha512_compress(s, w);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) S[8 * k + i] = bswap64(s[i]);
  }
}

// The hm96 leaf over x (a row's sha512/row/v2 digest) at position c under its salt S (salt_of(c)), and its frame-v3
// leaf, into d.
__device__ __forceinline__ void hm96_leaf(u64 c, const u64 (&x)[8], const u64 (&S)[24], u64 (&d)[8]) {
  u64 Y[24], my[8], cd[8];
#pragma unroll
  for (int i = 0; i < 24; ++i) Y[i] = bswap64(S[i]);
  u32 m[16];
  rc::mask<false>(rc::NibTab<true>{gNib}, Y, m);
  rc::mask_words(m, my);
  rc::salt_digest(kL, Y, cd);
  rc::frame_first_rt(kL, c, d);
  rc::finish_leaf<0, true>(kL, kL, c, x, my, cd, d, nullptr, nullptr);
}
__device__ __forceinline__ void hm96_leaf(u64 c, const u64 (&x)[8], u64 (&d)[8]) {
  u64 S[24];
  salt_of(c, S);
  hm96_leaf(c, x, S, d);
}

// node(h, i, L, R) under the pass's domain, h < NODE_LEVELS and i < 2^32
__device__ __forceinline__ void node_of(int h, u64 i, const u64 *L, const u64 *R, u64 (&d)[8]) {
  rc::node_rt_from(kNodeTpl, kC.node_mid[h], i, L, R, d);
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

  u64 S[24], d[8];
  salt_of(c, S);
  hm96_leaf(c, x, S, d);
  store_digest(dig, tid, nl, d);
  if (salts && tid < nl)
    for (int i = 0; i < 24; ++i)
      for (int b = 0; b < 8; ++b) salts[t * SALT + 8 * i + b] = (uint8_t)(S[i] >> (8 * b));
}

// --- nodes -------------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(128) pads_kernel(u64 r0, u64 n, u64 *__restrict__ out) {
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, t = tid < n ? tid : n - 1;
  u64 d[8];
  rc::pad_rt(kL, r0 + t, d);
  store_digest(out, tid, n, d);
}

// Nodes [i0, i0 + n) of the level above level h: node i over in[2 (i − i0)] and in[2 (i − i0) + 1].
__global__ void __launch_bounds__(128) level_kernel(const u64 *__restrict__ in, u64 *__restrict__ out, int h, u64 i0,
                                                    u64 n) {
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, t = tid < n ? tid : n - 1;
  u64 d[8];
  node_of(h, i0 + t, in + 16 * t, in + 16 * t + 8, d);
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

// --- the chained pass ------------------------------------------------------------------------------------------------

// A pass's seed-dependent constants as rb_chain_set's templates, which chain_kernel completes from the pass's seed on
// the device (wt.cu's g_pass.seed: s_j from root_(j−1)), so the receipt prefix, the tree's binding and domain and the
// salts are s_j's, receipts.domain(s_j, plan) and NodeMemory.salt's: each message's 64 seed-derived bytes are zero in its
// template, at its *_at.
// The domain id it makes fixes the rest: kC's node chaining values and kL's frame leaf and pad.
struct Chain {
  uint8_t receipt[128];   // prefix(seed)
  uint8_t binding[256];   // TREE_DOMAIN ‖ seed ‖ plan digest
  uint8_t domain[256];    // domain_message(binding, …)
  uint8_t schema[32];     // the leaves' schema, schema_len bytes
  u64 salt_block[32];     // Consts::salt_block, the seed's bytes at salt_at zero
  int receipt_at, binding_at, binding_len, domain_at, domain_len, salt_at, schema_len;
};
__device__ Chain g_chain;
__device__ Consts g_stage;
__device__ Leaf g_leaf;   // the chained pass's kL: its hm96 constants rb_chain_set's, its frame chain_kernel's

__device__ __forceinline__ void be_load(const uint8_t *b, u64 *w) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    u64 v = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) v = v << 8 | b[8 * i + k];
    w[i] = v;
  }
}
// SHA-512 of m's first n bytes, n ≤ 239 (m 256 bytes, zero past n), or with n < 0 the midstate of its first block.
__device__ void sha512_bytes(uint8_t *m, int n, u64 (&s)[8], uint8_t *out) {
  const int nb = n < 0 ? 1 : (n + 17 + 127) / 128;
  if (n >= 0) {
    m[n] = 0x80;
    for (int k = 0; k < 8; ++k) m[128 * nb - 1 - k] = (uint8_t)((u64)n * 8 >> (8 * k));
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = sx::IV_DEV[i];
  for (int b = 0; b < nb; ++b) {
    u64 w[16];
    be_load(m + 128 * b, w);
    sha512_compress(s, w);
  }
  if (out)
#pragma unroll
    for (int i = 0; i < 64; ++i) out[i] = (uint8_t)(s[i / 8] >> (56 - 8 * (i % 8)));
}

constexpr int CONST_WORDS = sizeof(Consts) / 8;
static_assert(sizeof(Consts) % 8 == 0, "Consts is whole u64 words");

// The pass's constants are made in shared memory from g_stage's seed-free ones (thread 0 the seed's messages, then each
// thread its level's node chaining value) and stored back whole, a warp's stores whole sectors. Threads t < MAX_NB make
// g_leaf's frame leaf template for len(uint(pos)) = t + 1 (thread 0 then fl_m15 from its own), the next MAX_NB its pad's.
__global__ void __launch_bounds__(NODE_LEVELS) chain_kernel(const uint8_t *__restrict__ seed) {
  static_assert(NODE_LEVELS >= 2 * rc::MAX_NB, "a thread a frame template");
  __shared__ Consts sc;
  __shared__ uint8_t m[256], d[64];
  u64 *sw = reinterpret_cast<u64 *>(&sc), *gw = reinterpret_cast<u64 *>(&g_stage);
  for (int i = threadIdx.x; i < CONST_WORDS; i += NODE_LEVELS) sw[i] = gw[i];
  __syncthreads();
  const Chain &c = g_chain;
  if (threadIdx.x == 0) {
    u64 s[8];
    for (int i = 0; i < 256; ++i) m[i] = i < 128 ? c.receipt[i] : 0;
    for (int i = 0; i < 64; ++i) m[c.receipt_at + i] = seed[i];
    sha512_bytes(m, -1, s, nullptr);
#pragma unroll
    for (int i = 0; i < 8; ++i) sc.receipt_mid[i] = s[i];
    for (int i = 0; i < 256; ++i) m[i] = c.binding[i];
    for (int i = 0; i < 64; ++i) m[c.binding_at + i] = seed[i];
    sha512_bytes(m, c.binding_len, s, d);
    for (int i = 0; i < 256; ++i) m[i] = c.domain[i];
    for (int i = 0; i < 64; ++i) m[c.domain_at + i] = d[i];
    sha512_bytes(m, c.domain_len, s, d);   // the domain id
    for (int i = 0; i < 32; ++i) sc.salt_block[i] = c.salt_block[i];
    for (int i = 0; i < 64; ++i) {
      const int p = c.salt_at + i;
      sc.salt_block[p / 8] |= (u64)seed[i] << 8 * (7 - p % 8);
    }
  }
  __syncthreads();
  const int t = threadIdx.x;
  rc::node_mid(d, t, sc.node_mid[t]);
  if (t < rc::MAX_NB) {
    rc::leaf_frame_nb(d, c.schema, c.schema_len, t + 1, g_leaf);
    if (!t) rc::leaf_m15(g_leaf);
  } else if (t < 2 * rc::MAX_NB) {
    rc::pad_tpl_nb(d, t - rc::MAX_NB + 1, g_leaf);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < CONST_WORDS; i += NODE_LEVELS) gw[i] = sw[i];
}

inline unsigned grid(u64 n, int threads) { return (unsigned)((n + threads - 1) / threads); }

}  // namespace rb

using namespace rb;

extern "C" {

// One pass's constants. The 128-byte blocks are the reference's own: prefix(seed), sha512_row_v2_prefix(8192) and
// (8 · 64 m) for the pass's last leaf of m receipts, hm96's salt_prefix and leaf_prefix(key); salt_in is D(store salt)
// ‖ the store's key ‖ seed, at most 227 bytes so that it, u64be c, u32be k and SHA-512's padding fill at most two
// blocks. With domain_id null the node chaining values and L's frame templates stay zero (chain_kernel makes them). The
// schema is hm96-sha512/row/v2's 18 bytes and positions are at most 2^36, so every leaf, pad and node index is below
// 2^32.
static int build(Consts &c, Leaf &L, const uint8_t *receipt_prefix, const uint8_t *row_prefix,
                 const uint8_t *row_prefix_last, const uint8_t *salt_prefix, const uint8_t *leaf_prefix,
                 const uint8_t *domain_id, const uint8_t *schema, int schema_len, const uint8_t *salt_in,
                 int salt_in_len, uint64_t positions) {
  if (!receipt_prefix || !row_prefix || !row_prefix_last || !salt_prefix || !leaf_prefix || !schema || !salt_in ||
      schema_len != 18 || salt_in_len < 1 || salt_in_len + 12 + 17 > 256 || positions < 1 || positions > (1ULL << 36))
    return 1;
  memset(&c, 0, sizeof c);
  memset(&L, 0, sizeof L);
  rc::midstate(receipt_prefix, c.receipt_mid);
  rc::midstate(row_prefix, c.row_mid);
  rc::midstate(row_prefix_last, c.row_mid_last);
  rc::pad_schedule(128 + 64 * PER_LEAF, c.kw_xpad);
  rc::midstate(salt_prefix, L.mid_salt);
  rc::midstate(leaf_prefix, L.mid_leaf);
  rc::pad_schedule(256, L.kw_leafpad);
  if (domain_id) {
    if (rc::leaf_frame(domain_id, schema, schema_len, L)) return 1;
    rc::pad_tpls(domain_id, L);
    rc::node_mids(domain_id, c.node_mid, NODE_LEVELS);
  }
  uint8_t blk[256] = {};
  memcpy(blk, salt_in, salt_in_len);
  blk[salt_in_len + 12] = 0x80;
  const int nb = (salt_in_len + 12 + 17 + 127) / 128;
  for (int b = 0; b < 8; ++b) blk[128 * nb - 1 - b] = (uint8_t)((u64)(salt_in_len + 12) * 8 >> (8 * b));
  for (int i = 0; i < 32; ++i) c.salt_block[i] = rc::be64_at(blk + 8 * i);
  c.salt_in_len = salt_in_len;
  c.positions = positions;
  return 0;
}

// M_key's nibble table (key: hm96-sha512's 256-byte key) and a node's last two blocks, on stream s.
static int tables(const uint8_t *key, cudaStream_t s) {
  static uint8_t tab[rc::TN_BYTES];
  static u64 tpl[rc::MAX_NB][32];
  if (!key) return 1;
  for (u32 t = 0; t < 512; ++t) rc::nib_entry(key, t, tab);
  if (rc::node_tpls(tpl)) return 1;
  return cudaMemcpyToSymbolAsync(gNib, tab, sizeof tab, 0, cudaMemcpyHostToDevice, s) ||
                 cudaMemcpyToSymbolAsync(kNodeTpl, tpl, sizeof tpl, 0, cudaMemcpyHostToDevice, s)
             ? 4
             : 0;
}

int rb_set(const uint8_t *receipt_prefix, const uint8_t *row_prefix, const uint8_t *row_prefix_last,
           const uint8_t *salt_prefix, const uint8_t *leaf_prefix, const uint8_t *key, const uint8_t *domain_id,
           const uint8_t *schema, int schema_len, const uint8_t *salt_in, int salt_in_len, uint64_t positions,
           cudaStream_t s) {
  static Consts c;
  static Leaf L;
  if (!domain_id) return 1;
  if (int e = build(c, L, receipt_prefix, row_prefix, row_prefix_last, salt_prefix, leaf_prefix, domain_id, schema,
                    schema_len, salt_in, salt_in_len, positions))
    return e;
  if (int e = tables(key, s)) return e;
  return cudaMemcpyToSymbolAsync(kC, &c, sizeof c, 0, cudaMemcpyHostToDevice, s) ||
                 cudaMemcpyToSymbolAsync(kL, &L, sizeof L, 0, cudaMemcpyHostToDevice, s)
             ? 4
             : 0;
}

// A chained pass's constants (rb_chain completes them from its seed on the device): rb_set's, with the seed's 64 bytes
// zero in receipt_prefix (at receipt_at) and salt_in (at salt_at), and in place of the domain id the binding's message
// (binding_len bytes, TREE_DOMAIN ‖ seed ‖ plan digest, the seed at binding_at) and the domain's (domain_len bytes,
// receipts.domain's domain_message, the binding at domain_at), each with those bytes zero.
int rb_chain_set(const uint8_t *receipt_prefix, int receipt_at, const uint8_t *row_prefix,
                 const uint8_t *row_prefix_last, const uint8_t *salt_prefix, const uint8_t *leaf_prefix,
                 const uint8_t *key, const uint8_t *binding, int binding_len, int binding_at, const uint8_t *domain,
                 int domain_len, int domain_at, const uint8_t *schema, int schema_len, const uint8_t *salt_in,
                 int salt_in_len, int salt_at, uint64_t positions, cudaStream_t s) {
  if (!binding || !domain || receipt_at < 0 || receipt_at + 64 > 128 || binding_len > 239 || binding_at < 0 ||
      binding_at + 64 > binding_len || domain_len > 239 || domain_at < 0 || domain_at + 64 > domain_len || salt_at < 0 ||
      salt_at + 64 > salt_in_len)
    return 1;
  static Consts c;
  static Leaf L;
  static Chain ch;
  if (int e = build(c, L, receipt_prefix, row_prefix, row_prefix_last, salt_prefix, leaf_prefix, nullptr, schema,
                    schema_len, salt_in, salt_in_len, positions))
    return e;
  if (int e = tables(key, s)) return e;
  memset(&ch, 0, sizeof ch);
  memcpy(ch.receipt, receipt_prefix, 128);
  memcpy(ch.binding, binding, binding_len);
  memcpy(ch.domain, domain, domain_len);
  memcpy(ch.schema, schema, schema_len);
  memcpy(ch.salt_block, c.salt_block, sizeof ch.salt_block);
  ch.receipt_at = receipt_at, ch.binding_at = binding_at, ch.binding_len = binding_len;
  ch.domain_at = domain_at, ch.domain_len = domain_len, ch.salt_at = salt_at, ch.schema_len = schema_len;
  return cudaMemcpyToSymbolAsync(g_stage, &c, sizeof c, 0, cudaMemcpyHostToDevice, s) ||
                 cudaMemcpyToSymbolAsync(g_chain, &ch, sizeof ch, 0, cudaMemcpyHostToDevice, s) ||
                 cudaMemcpyToSymbolAsync(g_leaf, &L, sizeof L, 0, cudaMemcpyHostToDevice, s)
             ? 4
             : 0;
}

// A chained pass's constants from its 64-byte seed on the device (wt.cu's g_pass.seed), into kC and kL, on stream s.
int rb_chain(const uint8_t *seed, cudaStream_t s) {
  if (!seed) return 1;
  chain_kernel<<<1, NODE_LEVELS, 0, s>>>(seed);
  void *stage = nullptr, *leaf = nullptr;
  if (cudaGetLastError() || cudaGetSymbolAddress(&stage, g_stage) || cudaGetSymbolAddress(&leaf, g_leaf)) return 4;
  return cudaMemcpyToSymbolAsync(kC, stage, sizeof(Consts), 0, cudaMemcpyDeviceToDevice, s) ||
                 cudaMemcpyToSymbolAsync(kL, leaf, sizeof(Leaf), 0, cudaMemcpyDeviceToDevice, s)
             ? 4
             : 0;
}

// The pass's constants as kC and kL hold them (bytes = rb_consts_bytes()), after the device is idle.
int rb_consts(uint8_t *out, uint64_t bytes) {
  if (!out || bytes != sizeof(Consts) + sizeof(Leaf)) return 1;
  return cudaMemcpyFromSymbol(out, kC, sizeof(Consts)) || cudaMemcpyFromSymbol(out + sizeof(Consts), kL, sizeof(Leaf))
             ? 4
             : 0;
}
uint64_t rb_consts_bytes() { return sizeof(Consts) + sizeof(Leaf); }

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
  rc::midstate(sketch_prefix, mid);
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

// σ at 64 rows of n blocks at data (each XORed with its spread z in zs, or not when zs is null) into sk (64 int32 a
// block) under signs (reader.py's `pack64`, 32 KiB on the device), mt m-tiles a warp (1, 2 or 4), on ctas CTAs (0:
// as many as are resident at once).
int rb_sketch64(const void *data, const void *zs, uint64_t n, const void *signs, int *sk, int mt, int ctas,
                cudaStream_t s) {
  if (!data || !signs || !sk || n < 1 || ctas < 0 || (mt != 1 && mt != 2 && mt != 4)) return 1;
  using K = void (*)(const uint4 *, const uint4 *, u64, const uint4 *, int *);
  const K k = zs ? (mt == 1 ? sketch64_kernel<true, 1> : mt == 2 ? sketch64_kernel<true, 2> : sketch64_kernel<true, 4>)
                 : (mt == 1 ? sketch64_kernel<false, 1> : mt == 2 ? sketch64_kernel<false, 2>
                                                                  : sketch64_kernel<false, 4>);
  if (!ctas) {
    int dev, sms, per;
    if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 32 * SK64_WARPS, 0))
      return 4;
    ctas = sms * per;
  }
  const u64 most = (n + 16 * mt * SK64_WARPS - 1) / (16 * mt * SK64_WARPS);
  const unsigned g = (unsigned)(most < (u64)ctas ? most : (u64)ctas);
  k<<<g, 32 * SK64_WARPS, 0, s>>>((const uint4 *)data, (const uint4 *)zs, n, (const uint4 *)signs, sk);
  return cudaGetLastError() ? 4 : 0;
}

// The 64-row receipts of n sketches (store indices j0 …) into ys (8 words a receipt).
int rb_sketch64_hash(const int *sk, uint64_t j0, uint64_t n, uint64_t *ys, cudaStream_t s) {
  if (!sk || !ys || n < 1) return 1;
  sketch64_hash_kernel<<<grid(n, 128), 128, 0, s>>>((const uint4 *)sk, j0, n, ys);
  return cudaGetLastError() ? 4 : 0;
}

// rb_sketch64 and rb_sketch64_hash in one kernel (2 m-tiles a warp): the receipts of n blocks into ys, and their σ
// into sk unless it is null.
int rb_sketch64_receipts(const void *data, const void *zs, uint64_t j0, uint64_t n, const void *signs, int *sk,
                         uint64_t *ys, int ctas, cudaStream_t s) {
  if (!data || !signs || !ys || n < 1 || ctas < 0) return 1;
  const auto k = zs ? sketch64_receipts_kernel<true> : sketch64_receipts_kernel<false>;
  static bool opted[2];
  if (!opted[zs != nullptr]) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SK64R_SMEM)) return 4;
    opted[zs != nullptr] = true;
  }
  if (!ctas) {
    int dev, sms, per;
    if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 32 * SK64R_WARPS, SK64R_SMEM))
      return 4;
    ctas = sms * per;
  }
  const u64 most = (n + 32 * SK64R_WARPS - 1) / (32 * SK64R_WARPS);
  const unsigned g = (unsigned)(most < (u64)ctas ? most : (u64)ctas);
  k<<<g, 32 * SK64R_WARPS, SK64R_SMEM, s>>>((const uint4 *)data, (const uint4 *)zs, j0, n, (const uint4 *)signs, sk,
                                            ys);
  return cudaGetLastError() ? 4 : 0;
}

int rb_leaves(const uint64_t *ys, uint64_t c0, uint64_t nl, uint64_t *dig, uint8_t *salts, cudaStream_t s) {
  if (!ys || !dig || nl < 1 || c0 > MAX_INDEX || nl > MAX_INDEX - c0) return 1;
  leaves_kernel<<<grid(nl, 128), 128, 0, s>>>(ys, c0, nl, dig, salts);
  return cudaGetLastError() ? 4 : 0;
}

int rb_pads(uint64_t r0, uint64_t n, uint64_t *out, cudaStream_t s) {
  if (!out || n < 1 || r0 > MAX_INDEX || n > MAX_INDEX - r0) return 1;
  pads_kernel<<<grid(n, 128), 128, 0, s>>>(r0, n, out);
  return cudaGetLastError() ? 4 : 0;
}

int rb_level(const uint64_t *in, uint64_t *out, int h, uint64_t i0, uint64_t n, cudaStream_t s) {
  if (!in || !out || n < 1 || h < 0 || h >= NODE_LEVELS || i0 > MAX_INDEX || n > MAX_INDEX - i0) return 1;
  level_kernel<<<grid(n, 128), 128, 0, s>>>(in, out, h, i0, n);
  return cudaGetLastError() ? 4 : 0;
}

// Every level above n digests of level 0 (a power of two, the first leaf c0) into out, level after level up to the
// root: n − 1 nodes in one call, so the narrow levels wait on the device and not on the host.
int rb_tree(const uint64_t *in, uint64_t n, uint64_t c0, uint64_t *out, cudaStream_t s) {
  if (!in || !out || n < 2 || n & (n - 1) || c0 >> 1 > MAX_INDEX || n / 2 > MAX_INDEX - (c0 >> 1)) return 1;
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
