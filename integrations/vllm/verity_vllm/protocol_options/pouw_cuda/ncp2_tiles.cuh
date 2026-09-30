// A tile's route-U accumulator hashed as it is computed: the function (`tile_fused`, and the low-byte variant, timed only) that
// `ncp2_host.cpp`'s twin runs on the CPU against `pouw_native`, and on the device `kf_tiles` and its launch (`launch_tiles`),
// which `ncp2_fused.cu` calls. Everything about how a tile's chain is computed and scheduled is here: the running sums' storage,
// the SHA-256 compression the word leaf uses, the block shape and its shared memory. The leaves and their order are fixed by
// `pouw_native`; `pouw_device.first_difference` holds every build of this file to it.
//
// The chain's layout is a build choice; every setting computes the same leaves and Z:
//   NCP2_TILE_REG_ROWS    rows of a tile's running sums held in registers, 8 (default) or 0; the other rows are in shared
//                         memory, so a chain needs 512 bytes of it instead of 1 KB and an SM runs twice the chains
//   NCP2_TILE_REG_UNROLL  register rows per pass of the row loop (1, 2, 4 or 8; default 2): the other register rows rotate
//                         into place between passes, 128 / unroll moves per row for a code body unroll rows long
//   NCP2_TILE_SHA         1 (default): the tile's own SHA-256 compression (`tile_compress`); 0: ncp2.cuh's `sha256_compress`
#pragma once
#include "ncp2_fast.cuh"

#ifndef NCP2_TILE_REG_ROWS
#define NCP2_TILE_REG_ROWS 8
#endif
#ifndef NCP2_TILE_REG_UNROLL
#define NCP2_TILE_REG_UNROLL 2
#endif
#ifndef NCP2_TILE_SHA
#define NCP2_TILE_SHA 1
#endif

namespace ncp2 {

constexpr int TILE_REG_ROWS = NCP2_TILE_REG_ROWS;
constexpr int TILE_REG_UNROLL = NCP2_TILE_REG_UNROLL;
static_assert(TILE_REG_ROWS == 0 || TILE_REG_ROWS == 8, "NCP2_TILE_REG_ROWS is 0 or 8");
static_assert(TILE_REG_UNROLL >= 1 && TILE_REG_UNROLL <= 8 && 8 % TILE_REG_UNROLL == 0, "NCP2_TILE_REG_UNROLL is 1, 2, 4 or 8");

// ---- a tile, hashed as it is computed -------------------------------------------------------------------------------------

// c plus Σ over 4 bytes of u8(a)·s8(b), mod 2^32.
HD int32_t dot4(uint32_t a, uint32_t b, int32_t c) {
#ifdef __CUDA_ARCH__
  int32_t d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#else
  uint32_t d = (uint32_t)c;
  for (int i = 0; i < 4; ++i) d += (uint32_t)((int32_t)((a >> (8 * i)) & 0xFFu) * (int32_t)(int8_t)(b >> (8 * i)));
  return (int32_t)d;
#endif
}

// SHA-256's compression (`sha256_compress`, the same function) for one chain's latency: h + K + W is summed a round before
// it is needed (the next round's h is this round's g), the new e and a share t1, Ch and Maj are one three-input operation
// each, and each message word is scheduled a round before its use. A round is 14 instructions, as before; what it shortens
// is the path from one round's e to the next.
HD uint32_t tile_rotr(uint32_t x, int n) {
#ifdef __CUDA_ARCH__
  return __funnelshift_r(x, x, n);
#else
  return rotr32(x, n);
#endif
}

HD void tile_compress(uint32_t h[8], const uint32_t blk[16]) {
  const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
      0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
      0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t w[16], s[8];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = blk[i];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = h[i];
  // Round i's working variable v (a = 0 … h = 7) is s[(v − i) & 7]: renaming instead of moving.
  uint32_t hkw = s[7] + K[0] + w[0];
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    const uint32_t a = s[(0 - i) & 7], b = s[(1 - i) & 7], c = s[(2 - i) & 7];
    const uint32_t e = s[(4 - i) & 7], f = s[(5 - i) & 7], g = s[(6 - i) & 7];
    if (i + 1 >= 16 && i + 1 < 64) {
      const int j = i + 1;
      const uint32_t w15 = w[(j + 1) & 15], w2 = w[(j + 14) & 15];
      w[j & 15] += (tile_rotr(w15, 7) ^ tile_rotr(w15, 18) ^ (w15 >> 3)) + w[(j + 9) & 15] +
                   (tile_rotr(w2, 17) ^ tile_rotr(w2, 19) ^ (w2 >> 10));
    }
    const uint32_t t1 = (tile_rotr(e, 6) ^ tile_rotr(e, 11) ^ tile_rotr(e, 25)) + (g ^ (e & (f ^ g))) + hkw;
    if (i + 1 < 64) hkw = g + K[i + 1] + w[(i + 1) & 15];
    s[(3 - i) & 7] += t1;
    s[(7 - i) & 7] = t1 + (tile_rotr(a, 2) ^ tile_rotr(a, 13) ^ tile_rotr(a, 22)) + ((a & b) | (c & (a | b)));
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] += s[i];
}

// 16 stream words into the word leaf: `wl_block`, with the tile's compression.
HD void tile_block(WordLeaf& s, const uint32_t w[16]) {
#if NCP2_TILE_SHA
  uint32_t b[16];
#pragma unroll
  for (int i = 0; i < 6; ++i) b[i] = shift3(s.p[i], s.p[i + 1]);
  b[6] = shift3(s.p[6], w[0]);
#pragma unroll
  for (int i = 0; i < 9; ++i) b[7 + i] = shift3(w[i], w[i + 1]);
  tile_compress(s.h, b);
#pragma unroll
  for (int i = 0; i < 7; ++i) s.p[i] = w[9 + i];
  s.n += 16;
#else
  wl_block(s, w);
#endif
}

// One row's 16-deep step: each running sum in ``v`` plus that row of X·Y's step, the sum carried through the dot products
// (mod 2^32, the same value as adding the step's sum to it).
HD void tile_row(const uint32_t xv[4], const uint32_t yv[16][4], uint32_t v[16]) {
#pragma unroll
  for (int c = 0; c < 16; ++c) {
    int32_t acc = dot4(xv[0], yv[c][0], (int32_t)v[c]);
    acc = dot4(xv[1], yv[c][1], acc);
    acc = dot4(xv[2], yv[c][2], acc);
    v[c] = (uint32_t)dot4(xv[3], yv[c][3], acc);
  }
}

// A tile's chain: X strip rows ``xs`` and Y strip columns ``ys`` (16 × k3 bytes each). Rows 0 … REG − 1 of the running sums
// are ``reg``, the rest ``hi(16·(r − REG) + c)``; both hold c₀ at the start and Z at the end. Every 16-deep step adds X·Y's
// step to each running sum and hands row r's 16 words to the leaf as one block, step-major then row-major. With HASH false,
// the same arithmetic unhashed.
template <int REG, int U, bool HASH, class Hi>
HD void tile_chain(const uint8_t* xs, const uint8_t* ys, int k3, int steps, uint32_t (&reg)[REG > 0 ? REG : 1][16], Hi hi,
                   WordLeaf& wl) {
  for (int t = 0; t < steps; ++t) {
    uint32_t yv[16][4];
#pragma unroll
    for (int c = 0; c < 16; ++c) load16(ys + (size_t)c * k3 + 16 * t, yv[c]);
    if constexpr (REG > 0) {
      static_assert(REG % U == 0, "the register rows are passes of U rows");
#pragma unroll 1
      for (int g = 0; g < REG / U; ++g) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
          uint32_t xv[4];
          load16(xs + (size_t)(g * U + u) * k3 + 16 * t, xv);
          tile_row(xv, yv, reg[u]);
          if (HASH) tile_block(wl, reg[u]);
        }
        if constexpr (U < REG) {
          uint32_t done[U][16];
#pragma unroll
          for (int u = 0; u < U; ++u)
#pragma unroll
            for (int c = 0; c < 16; ++c) done[u][c] = reg[u][c];
#pragma unroll
          for (int q = 0; q < REG - U; ++q)
#pragma unroll
            for (int c = 0; c < 16; ++c) reg[q][c] = reg[q + U][c];
#pragma unroll
          for (int u = 0; u < U; ++u)
#pragma unroll
            for (int c = 0; c < 16; ++c) reg[REG - U + u][c] = done[u][c];
        }
      }
    }
#pragma unroll 1
    for (int r = REG; r < D; ++r) {
      uint32_t xv[4], w[16];
      load16(xs + (size_t)r * k3 + 16 * t, xv);
#pragma unroll
      for (int c = 0; c < 16; ++c) w[c] = hi(16 * (r - REG) + c);
      tile_row(xv, yv, w);
#pragma unroll
      for (int c = 0; c < 16; ++c) hi(16 * (r - REG) + c) = w[c];
      if (HASH) tile_block(wl, w);
    }
  }
}

template <class Run>
struct RowsFrom {
  Run run;
  int first;
  HDM uint32_t& operator()(int j) const { return run(first + j); }
};

// One tile with ``run(j)``, j = 16·r + c, holding its running sums: c₀ at the start and Z at the end.
template <bool HASH, class Run>
HD void tile_fused(const uint8_t* xs, const uint8_t* ys, int k3, int steps, Run run, uint8_t leaf[32]) {
  constexpr int REG = TILE_REG_ROWS;
  uint32_t reg[REG > 0 ? REG : 1][16];
#pragma unroll
  for (int r = 0; r < REG; ++r)
#pragma unroll
    for (int c = 0; c < 16; ++c) reg[r][c] = run(16 * r + c);
  WordLeaf wl;
  if (HASH) wl_init(wl);
  tile_chain<REG, TILE_REG_UNROLL, HASH>(xs, ys, k3, steps, reg, RowsFrom<Run>{run, D * REG}, wl);
#pragma unroll
  for (int r = 0; r < REG; ++r)
#pragma unroll
    for (int c = 0; c < 16; ++c) run(16 * r + c) = reg[r][c];
  if (HASH) wl_final(wl, leaf);
}

// The low-byte leaf variant (cheap-binding's NCP-INT row 4; its γ is unproved: timed only, not committed, never a result): TurboSHAKE128 with domain byte 0x0B over
// the low byte of every checkpoint in the word leaf's order, 8 rows (128 bytes, 16 lanes) per permutation, each block's other
// 5 lanes of the rate zero.
constexpr uint8_t LOWBYTE_DS = 0x0B;

template <class Run>
HD void tile_fused_lowbyte(const uint8_t* xs, const uint8_t* ys, int k3, int steps, Run run, uint8_t leaf[32]) {
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
  for (int t = 0; t < steps; ++t) {
    uint32_t yv[16][4];
#pragma unroll
    for (int c = 0; c < 16; ++c) load16(ys + (size_t)c * k3 + 16 * t, yv[c]);
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const int r = 8 * half + rr;
        uint32_t xv[4];
        load16(xs + (size_t)r * k3 + 16 * t, xv);
        uint64_t lo = 0, hi = 0;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          int32_t acc = dot4(xv[0], yv[c][0], 0);
          acc = dot4(xv[1], yv[c][1], acc);
          acc = dot4(xv[2], yv[c][2], acc);
          acc = dot4(xv[3], yv[c][3], acc);
          uint32_t v = run(16 * r + c) + (uint32_t)acc;
          run(16 * r + c) = v;
          if (c < 8) lo |= (uint64_t)(v & 0xFFu) << (8 * c);
          else hi |= (uint64_t)(v & 0xFFu) << (8 * (c - 8));
        }
        s[2 * rr] ^= lo;
        s[2 * rr + 1] ^= hi;
      }
      if (t == steps - 1 && half == 1) {
        s[16] ^= LOWBYTE_DS;
        s[20] ^= 0x80ull << 56;
      }
      keccak_p<12>(s);
    }
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) leaf[i] = (uint8_t)(s[i >> 3] >> (8 * (i & 7)));
}

}  // namespace ncp2

#ifdef __CUDACC__

namespace ncp2 {

template <int S>
struct SmemRun {
  uint32_t* base;
  __device__ __forceinline__ uint32_t& operator()(int j) const { return base[j * S]; }
};

// A block of chains keeps 32 KB of shared memory, so an SM holds three blocks (with the 1 KB each block reserves): 32 lanes
// with every running sum in shared memory, 64 with half of them in registers. The low-byte variant keeps every running sum
// in shared memory.
template <int LEAF>
struct TileShape {
  static constexpr int reg = LEAF == 2 ? 0 : TILE_REG_ROWS;
  static constexpr int threads = reg ? 64 : 32;
  static constexpr size_t smem = (size_t)threads * (D - reg) * D * 4;
};

// LEAF 0: no leaf (the no-hash control); 1: the word leaf (committed); 2: the low-byte variant (timed only; γ unproved).
template <int LEAF>
__global__ void __launch_bounds__(TileShape<LEAF>::threads) kf_tiles(int ns, size_t tiles, int steps, int k3, int n,
                                                                     const uint8_t* x, const uint8_t* y, const uint32_t* c0,
                                                                     uint8_t* tile_leaves, uint32_t* z) {
  using S = TileShape<LEAF>;
  constexpr int REG = S::reg;
  extern __shared__ uint32_t sm[];
  const int lane = threadIdx.x;
  const size_t tile = (size_t)blockIdx.x * S::threads + lane;
  if (tile >= tiles) return;
  const int rs = (int)(tile / ns), cs = (int)(tile % ns);
  const uint8_t* xs = x + (size_t)rs * D * k3;
  const uint8_t* ys = y + (size_t)cs * D * k3;
  SmemRun<S::threads> hi{sm + lane};
  uint32_t c0v[16];
#pragma unroll
  for (int c = 0; c < 16; ++c) c0v[c] = c0[cs * D + c];
  uint8_t leaf[32];
  uint32_t reg[REG > 0 ? REG : 1][16];
  if constexpr (LEAF == 2) {
    for (int j = 0; j < D * D; ++j) hi(j) = c0v[j & 15];
    tile_fused_lowbyte(xs, ys, k3, steps, hi, leaf);
  } else {
#pragma unroll
    for (int r = 0; r < REG; ++r)
#pragma unroll
      for (int c = 0; c < 16; ++c) reg[r][c] = c0v[c];
    for (int j = 0; j < (D - REG) * D; ++j) hi(j) = c0v[j & 15];
    WordLeaf wl;
    if (LEAF == 1) wl_init(wl);
    tile_chain<REG, TILE_REG_UNROLL, LEAF == 1>(xs, ys, k3, steps, reg, hi, wl);
    if (LEAF == 1) wl_final(wl, leaf);
  }
  if (LEAF)
    for (int i = 0; i < 32; ++i) tile_leaves[32 * tile + i] = leaf[i];
  uint32_t* zt = z + (size_t)rs * D * n + (size_t)cs * D;
#pragma unroll
  for (int r = 0; r < REG; ++r)
#pragma unroll
    for (int c = 0; c < 16; ++c) zt[(size_t)r * n + c] = reg[r][c];
  for (int j = 0; j < (D - REG) * D; ++j) zt[(size_t)(REG + (j >> 4)) * n + (j & 15)] = hi(j);
}

template <int LEAF>
static cudaError_t tile_attributes() {
  cudaError_t e = cudaFuncSetAttribute(kf_tiles<LEAF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TileShape<LEAF>::smem);
  if (e == cudaSuccess)
    e = cudaFuncSetAttribute(kf_tiles<LEAF>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);
  return e;
}

// The tile kernel's attributes, once per process: its shared memory, and the carveout that fits three blocks an SM.
static cudaError_t tiles_setup() {
  cudaError_t e = tile_attributes<0>();
  if (e == cudaSuccess) e = tile_attributes<1>();
  if (e == cudaSuccess) e = tile_attributes<2>();
  return e;
}

template <int LEAF>
static void launch_tiles_as(int ns, size_t tiles, int steps, int k3, int n, const uint8_t* x, const uint8_t* y,
                            const uint32_t* c0, uint8_t* tile_leaves, uint32_t* z, cudaStream_t stream) {
  using S = TileShape<LEAF>;
  const unsigned grid = (unsigned)((tiles + S::threads - 1) / S::threads);
  kf_tiles<LEAF><<<grid, S::threads, S::smem, stream>>>(ns, tiles, steps, k3, n, x, y, c0, tile_leaves, z);
}

// Every tile of a call on ``stream``: its leaf into ``tile_leaves`` (``leaf`` 1 the word leaf, 2 the low-byte variant, 0 none)
// and its Z into ``z`` (m × n, row-major).
static void launch_tiles(int leaf, int ns, size_t tiles, int steps, int k3, int n, const uint8_t* x, const uint8_t* y,
                         const uint32_t* c0, uint8_t* tile_leaves, uint32_t* z, cudaStream_t stream) {
  if (leaf == 2) launch_tiles_as<2>(ns, tiles, steps, k3, n, x, y, c0, tile_leaves, z, stream);
  else if (leaf == 1) launch_tiles_as<1>(ns, tiles, steps, k3, n, x, y, c0, tile_leaves, z, stream);
  else launch_tiles_as<0>(ns, tiles, steps, k3, n, x, y, c0, nullptr, z, stream);
}

}  // namespace ncp2

#endif
