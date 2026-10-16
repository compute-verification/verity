// pfill: a DRAM-free filler in Theorem E's own setting (HC-PROVED, cmp-13). dcap.cu includes this file in place of FILL
// v1's chained filler when DFILL_PROVED is defined, under the same ABI (dfill_k<NC>, DFill, dfill_launch, dfill_prep,
// dfill_make, dfill_check, dfill_ref), so dcap's d* modes and filler/dscore.cu run it unchanged.
//
// A tile is one noise GEMM C = A B with A (TM x 32) and B (32 x PF_TN) i.i.d. uniform on {±1} and nothing else in either
// operand (A_M = B_M = 0), so K = L = 32: every m16n8k32 output is a whole k32 prefix sum, folded into the receipt at once
// and never accumulated. The noise is SHA-512 counter mode, one compression a 512-entry block, over the 40-byte message
// (c0, c1, tile, PF_DOM, ctr), each a big-endian u64: entry e of a block is bit e % 8 of digest byte e / 8, value
// 1 - 2 bit. A's fragment F (rows 16F..16F+15) is block ctr = F, entry 32 i + k its (row 16F + i, k); B's block b (columns
// 16b..16b+15) is ctr = 2^56 + b, entry 32 i + k its (k, column 16b + i); the receipt keys are block ctr = 2^57. Tile ids
// come from a public range the verifier fixes before the coins (the anti-replay gate), so no noise is known before the
// round.
//
// A block is PF_W = 8 warps and runs `tiles` tiles, ids tile0 + blockIdx.x * tiles + i. Warp w owns fragments w S .. w S +
// S - 1 (S = steps, a multiple of 32) and sweeps each against all PF_TN / 8 B fragments. B sits in shared memory as s8
// bytes (32 KB), made once a tile by lanes 0..7 of every warp (lane 8 makes the keys). A sits there as bits: lane l of a
// warp computes fragment f0 + l's block, 64 bytes, and each fragment is expanded to s8 registers once and reused over 128
// MMAs, so the SHA costs about 4 bits an MMA. Warps w and w + 4 share an SMSP; the upper four start their SHA phases half
// a pass (16 fragments) late, so the two warps of an SMSP never hash at the same time. The schedule changes when the
// hashing happens, never what a chain absorbs.
//
// The receipt: lane (g, t) of warp w keeps one chain per fragment in flight, c < 4, fragment f's outputs going to chain f
// % 4 in order of f and then of B fragment j; each MMA gives the chain (d0, d1) = C[16F+g][8j+2t, 8j+2t+1] and (d2, d3) =
// the same of row 16F+g+8. NC picks the fold:
//   4  ram  y <- (rotl(y, 7) + d0) a0 + d1;  y <- (rotl(y, 19) + d2) a1 + d3   (two ops a pair)
//   2  duo  per word p, keys (k, a, r): u = y ^ p ^ k;  y <- u a + z;  z <- rotl(u, r) + y   (three ops a word)
//   1  xor  y <- y ^ d0 ^ d1 ^ d2 ^ d3, the cheapest fold that reads every word: a cost reference, not a receipt (it is
//           linear, so it binds nothing)
//   5, 3    ram, xor with A made once a tile and reused (a probe of the SHA's cost; its receipts are not dfill_ref's)
// Keys are the key block's row words in order, the first two that are odd, at least 2^8 + 1 away from 0 mod 2^24 and not
// ±1 mod 32 (a0, a1); duo's k0, k1 are its rows 14 and 15. A tile's receipt is the XOR over its chains of
// smix(smix(K ^ pos) ^ v), K = dseed(c0, c1, tile), pos = (32 w + lane) 4 + c, v the chain's state (duo: z << 32 | y).
#pragma once
#include "../pearl_c/sha512.cuh"

#define PF_W 8
#define PF_TN 1024
#define DFILL_GATE_W PF_W
#define DFILL_GATE_S {32, 512}
constexpr uint64_t PF_DOM = 0x5046494C4C2D7631ull;  // "PFILL-v1"
constexpr uint64_t PF_B = 1ull << 56, PF_KEY = 2ull << 56;
static_assert(PF_TN == 16 * 8 * PF_W, "lanes 0..7 of every warp make B's blocks");

__host__ __device__ inline uint32_t pf_bswap(uint32_t x) {
  return (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
}
__host__ __device__ inline uint32_t pf_rotl(uint32_t x, int r) {
#ifdef __CUDA_ARCH__
  return __funnelshift_l(x, x, r);
#else
  return (x << r) | (x >> (32 - r));
#endif
}
// row word r of a digest: digest bytes 4r..4r+3 as a little-endian u32, so bit k is entry 32 r + k's bit
__host__ __device__ inline uint32_t pf_row(const uint64_t* st, int r) {
  return pf_bswap((r & 1) ? (uint32_t)st[r >> 1] : (uint32_t)(st[r >> 1] >> 32));
}
// four entries (bits of nib, entry q in bit q) as s8 bytes 1 - 2 bit, entry q in byte q
__host__ __device__ inline uint32_t pf_pm1(uint32_t nib) { return 0x01010101u | (((nib * 0x00204081u) & 0x01010101u) * 0xFEu); }
__host__ __device__ inline bool pf_key_ok(uint32_t a) {
  const uint32_t m = a & 0xFFFFFFu, r = a & 31u;
  return (a & 1u) && m > 256u && m < (1u << 24) - 256u && r != 1u && r != 31u;
}
struct PfKeys {
  uint32_t a0, a1, k0, k1;
};
__host__ __device__ inline PfKeys pf_keys(const uint64_t* st) {
  PfKeys k{0x9E3779B9u, 0x7F4A7C15u, pf_row(st, 14), pf_row(st, 15)};
  int n = 0;
#pragma unroll
  for (int r = 0; r < 16; r++) {
    const uint32_t a = pf_row(st, r);
    if (pf_key_ok(a)) {
      if (n == 0) k.a0 = a;
      else if (n == 1) k.a1 = a;
      n++;
    }
  }
  return k;
}
template <int NC>
__host__ __device__ __forceinline__ void pf_absorb(uint32_t& y, uint32_t& z, uint32_t d0, uint32_t d1, uint32_t d2,
                                                   uint32_t d3, const PfKeys& k) {
  if (NC == 4 || NC == 5) {
    y = (pf_rotl(y, 7) + d0) * k.a0 + d1;
    y = (pf_rotl(y, 19) + d2) * k.a1 + d3;
  } else if (NC == 2) {
    uint32_t u;
    u = y ^ d0 ^ k.k0; y = u * k.a0 + z; z = pf_rotl(u, 5) + y;
    u = y ^ d1 ^ k.k1; y = u * k.a1 + z; z = pf_rotl(u, 11) + y;
    u = y ^ d2 ^ k.k0; y = u * k.a0 + z; z = pf_rotl(u, 17) + y;
    u = y ^ d3 ^ k.k1; y = u * k.a1 + z; z = pf_rotl(u, 29) + y;
  } else {
    y ^= d0 ^ d1 ^ d2 ^ d3;
  }
}
__host__ __device__ inline uint32_t pf_y0(uint32_t pos) { return pos; }
__host__ __device__ inline uint32_t pf_z0(uint32_t pos) { return pos * 0x9E3779B9u; }
__host__ __device__ inline uint64_t pf_fold(uint64_t K, uint32_t pos, uint64_t v) { return smix(smix(K ^ pos) ^ v); }

__device__ __forceinline__ void pf_block(uint64_t c0, uint64_t c1, uint64_t tile, uint64_t ctr, uint64_t* st) {
  uint64_t w[16];
  w[0] = c0; w[1] = c1; w[2] = tile; w[3] = PF_DOM; w[4] = ctr; w[5] = 0x8000000000000000ull;
#pragma unroll
  for (int i = 6; i < 15; i++) w[i] = 0;
  w[15] = 320;
  rs_iv(st);
  rs_compress(st, w);
}

template <int NC>
__global__ void __launch_bounds__(32 * PF_W, 1) dfill_k(uint64_t c0, uint64_t c1, unsigned long long tile0, int tiles,
                                                       int steps, unsigned long long* __restrict__ rec) {
  constexpr bool regen = NC == 1 || NC == 2 || NC == 4;
  __shared__ __align__(16) uint8_t Bs[PF_TN * 32];  // column n's 32 bytes at 32 n, words in order 0 4 1 5 2 6 3 7
  __shared__ __align__(16) uint32_t Ab[PF_W][32 * 16];  // a warp's 32 fragments, 16 row words each
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int S = steps;
  for (int i = 0; i < tiles; i++) {
    const unsigned long long idx = (unsigned long long)blockIdx.x * tiles + i, tile = tile0 + idx;
    PfKeys key{0, 0, 0, 0};
    if (lane <= 8) {
      uint64_t st[8];
      pf_block(c0, c1, tile, lane < 8 ? PF_B + (uint64_t)(w * 8 + lane) : PF_KEY, st);
      if (lane < 8) {
#pragma unroll
        for (int nn = 0; nn < 16; nn++) {
          const uint32_t R = pf_row(st, nn);
          uint4* p = (uint4*)&Bs[(16 * (w * 8 + lane) + nn) * 32];
          p[0] = make_uint4(pf_pm1(R & 15), pf_pm1((R >> 16) & 15), pf_pm1((R >> 4) & 15), pf_pm1((R >> 20) & 15));
          p[1] = make_uint4(pf_pm1((R >> 8) & 15), pf_pm1((R >> 24) & 15), pf_pm1((R >> 12) & 15), pf_pm1(R >> 28));
        }
      } else {
        key = pf_keys(st);
      }
    }
    key.a0 = __shfl_sync(~0u, key.a0, 8);
    key.a1 = __shfl_sync(~0u, key.a1, 8);
    key.k0 = __shfl_sync(~0u, key.k0, 8);
    key.k1 = __shfl_sync(~0u, key.k1, 8);
    __syncthreads();
    auto agen = [&](int f0) {
      __syncwarp();
      if (f0 + lane < S) {
        uint64_t st[8];
        pf_block(c0, c1, tile, (uint64_t)w * S + f0 + lane, st);
        uint4* p = (uint4*)&Ab[w][lane * 16];
#pragma unroll
        for (int q = 0; q < 4; q++)
          p[q] = make_uint4(pf_row(st, 4 * q), pf_row(st, 4 * q + 1), pf_row(st, 4 * q + 2), pf_row(st, 4 * q + 3));
      }
      __syncwarp();
    };
    uint32_t y[4], z[4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const uint32_t pos = (uint32_t)((w * 32 + lane) * 4 + c);
      y[c] = pf_y0(pos);
      z[c] = pf_z0(pos);
    }
    int f0 = 0, next = (w >> 2) & 1 ? 16 : 32;
    agen(0);
    const uint8_t* bp = Bs + g * 32 + 8 * t;
#pragma unroll 1
    for (int f = 0; f < S; f += 4) {
      if (f == next) {
        if (regen) {
          f0 = f;
          agen(f0);
        }
        next = f + 32;
      }
      uint32_t a[4][4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const int s = regen ? f - f0 + c : (f + c) & 31;
        const uint32_t R0 = Ab[w][s * 16 + g], R1 = Ab[w][s * 16 + g + 8];
        a[c][0] = pf_pm1((R0 >> (4 * t)) & 15);
        a[c][1] = pf_pm1((R1 >> (4 * t)) & 15);
        a[c][2] = pf_pm1((R0 >> (16 + 4 * t)) & 15);
        a[c][3] = pf_pm1((R1 >> (16 + 4 * t)) & 15);
      }
#pragma unroll 4
      for (int j = 0; j < PF_TN / 8; j++) {
        const uint2 b = *(const uint2*)(bp + j * 256);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          uint32_t d0, d1, d2, d3;
          asm(
              "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
              : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
              : "r"(a[c][0]), "r"(a[c][1]), "r"(a[c][2]), "r"(a[c][3]), "r"(b.x), "r"(b.y), "r"(0));
          pf_absorb<NC>(y[c], z[c], d0, d1, d2, d3, key);
        }
      }
    }
    const uint64_t K = dseed(c0, c1, tile);
    uint64_t h = 0;
#pragma unroll
    for (int c = 0; c < 4; c++)
      h ^= pf_fold(K, (uint32_t)((w * 32 + lane) * 4 + c), NC == 2 ? (uint64_t)z[c] << 32 | y[c] : (uint64_t)y[c]);
#pragma unroll
    for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(~0u, h, o);
    __syncwarp();
    if (lane == 0) {
      Ab[w][0] = (uint32_t)h;
      Ab[w][1] = (uint32_t)(h >> 32);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint64_t x = 0;
#pragma unroll
      for (int v = 0; v < PF_W; v++) x ^= (uint64_t)Ab[v][0] | (uint64_t)Ab[v][1] << 32;
      rec[idx] = x;
    }
  }
}

// ---------------------------------------------------------------- the host's tile, from the matrices
static uint64_t pf_K[80];
static void pf_host_k() {
  static bool ok = false;
  if (!ok) CK(cudaMemcpyFromSymbol(pf_K, RS_K, sizeof pf_K));
  ok = true;
}
static inline uint64_t pf_rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
static void pf_block_h(uint64_t c0, uint64_t c1, uint64_t tile, uint64_t ctr, uint64_t* st) {
  uint64_t w[80] = {c0, c1, tile, PF_DOM, ctr, 0x8000000000000000ull, 0, 0, 0, 0, 0, 0, 0, 0, 0, 320};
  for (int i = 16; i < 80; i++)
    w[i] = (pf_rotr64(w[i - 2], 19) ^ pf_rotr64(w[i - 2], 61) ^ (w[i - 2] >> 6)) + w[i - 7] +
           (pf_rotr64(w[i - 15], 1) ^ pf_rotr64(w[i - 15], 8) ^ (w[i - 15] >> 7)) + w[i - 16];
  static const uint64_t iv[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                                 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  uint64_t v[8];
  for (int i = 0; i < 8; i++) v[i] = iv[i];
  for (int i = 0; i < 80; i++) {
    const uint64_t e = v[4], a = v[0];
    const uint64_t t1 = v[7] + (pf_rotr64(e, 14) ^ pf_rotr64(e, 18) ^ pf_rotr64(e, 41)) + ((e & v[5]) ^ (~e & v[6])) + pf_K[i] + w[i];
    const uint64_t t2 = (pf_rotr64(a, 28) ^ pf_rotr64(a, 34) ^ pf_rotr64(a, 39)) + ((a & v[1]) ^ (a & v[2]) ^ (v[1] & v[2]));
    v[7] = v[6]; v[6] = v[5]; v[5] = v[4]; v[4] = v[3] + t1; v[3] = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = t1 + t2;
  }
  for (int i = 0; i < 8; i++) st[i] = iv[i] + v[i];
}
template <int NC>
static uint64_t pf_ref(uint64_t c0, uint64_t c1, uint64_t tile, int steps) {
  pf_host_k();
  uint64_t st[8];
  std::vector<uint32_t> Bb(PF_TN);  // bit k of Bb[n]: entry (k, n)
  for (int b = 0; b < PF_TN / 16; b++) {
    pf_block_h(c0, c1, tile, PF_B + b, st);
    for (int i = 0; i < 16; i++) Bb[16 * b + i] = pf_row(st, i);
  }
  pf_block_h(c0, c1, tile, PF_KEY, st);
  const PfKeys key = pf_keys(st);
  const uint64_t K = dseed(c0, c1, tile);
  std::vector<uint32_t> Ar(16 * std::max(steps, 0));  // bit k of Ar[16 f + i]: entry (row 16 f + i of the warp's rows, k)
  uint64_t h = 0;
  for (int w = 0; w < PF_W; w++) {
    for (int f = 0; f < steps; f++) {
      pf_block_h(c0, c1, tile, (uint64_t)w * steps + f, st);
      for (int i = 0; i < 16; i++) Ar[16 * f + i] = pf_row(st, i);
    }
    for (int lane = 0; lane < 32; lane++) {
      const int g = lane >> 2, t = lane & 3;
      for (int c = 0; c < 4; c++) {
        const uint32_t pos = (uint32_t)((w * 32 + lane) * 4 + c);
        uint32_t y = pf_y0(pos), z = pf_z0(pos);
        for (int f = c; f < steps; f += 4) {
          const uint32_t R0 = Ar[16 * f + g], R1 = Ar[16 * f + g + 8];
          for (int j = 0; j < PF_TN / 8; j++) {
            const uint32_t b0 = Bb[8 * j + 2 * t], b1 = Bb[8 * j + 2 * t + 1];
            pf_absorb<NC>(y, z, 32 - 2 * __builtin_popcount(R0 ^ b0), 32 - 2 * __builtin_popcount(R0 ^ b1),
                          32 - 2 * __builtin_popcount(R1 ^ b0), 32 - 2 * __builtin_popcount(R1 ^ b1), key);
          }
        }
        h ^= pf_fold(K, pos, NC == 2 ? (uint64_t)z << 32 | y : (uint64_t)y);
      }
    }
  }
  return h;
}
// steps need not be a multiple of 32 here (dgate's control runs S - 8); the probes (3, 5) have no host tile, and
// dfill_check reports -1 for them
static uint64_t dfill_ref(uint64_t c0, uint64_t c1, uint64_t tile, int nc, int steps) {
  switch (nc) {
    case 1: return pf_ref<1>(c0, c1, tile, steps);
    case 2: return pf_ref<2>(c0, c1, tile, steps);
    case 4: return pf_ref<4>(c0, c1, tile, steps);
    default: return 0;
  }
}

// one chunk: `grid` blocks of PF_W warps, each `tiles` tiles of PF_W S fragments x PF_TN columns
struct DFill {
  int W = PF_W, NC = 4, S = 512, tiles = 1, grid = 0;
  uint64_t c0 = 0, c1 = 0;
  unsigned long long* rec = nullptr;
  long long per() const { return (long long)grid * tiles; }
  double macs() const { return (double)per() * W * S * (PF_TN / 8) * 4096.0; }
};
static void dfill_launch(const DFill& f, long long chunk, cudaStream_t st) {
  const unsigned long long t0 = (unsigned long long)chunk * f.per();
  const dim3 gr(f.grid), bl(32 * PF_W);
  switch (f.NC) {
    case 1: dfill_k<1><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 2: dfill_k<2><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 3: dfill_k<3><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 4: dfill_k<4><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 5: dfill_k<5><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    default: fprintf(stderr, "pfill: NC must be 1 (xor), 2 (duo), 4 (ram), or the probes 3, 5\n"); exit(1);
  }
}
static void dfill_prep() {
  CK(cudaFuncSetAttribute(dfill_k<1>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<2>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<3>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<4>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<5>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
}
// tiles a block so that a chunk is about 2^DFILL_CHUNK_LOG2 (default 19) MMAs an SM
static DFill dfill_make(int grid, int W, int NC, int S, uint64_t c0, uint64_t c1) {
  if (W != PF_W || S < 32 || S % 32) {
    fprintf(stderr, "pfill: W must be %d and S a positive multiple of 32 (W %d, S %d)\n", PF_W, W, S);
    exit(1);
  }
  DFill f;
  f.grid = grid; f.W = W; f.NC = NC; f.S = S; f.c0 = c0; f.c1 = c1;
  const int lg = getenv("DFILL_CHUNK_LOG2") ? atoi(getenv("DFILL_CHUNK_LOG2")) : 19;
  f.tiles = std::max(1, (int)((1LL << lg) / ((long long)W * S * (PF_TN / 8))));
  CK(cudaMalloc(&f.rec, f.per() * sizeof(unsigned long long)));
  return f;
}
static int dfill_check(const DFill& f, long long chunk, int n) {
  if (f.NC == 3 || f.NC == 5) return -1;  // a probe: not checked
  std::vector<unsigned long long> h(f.per());
  CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
  int bad = 0;
  n = (int)std::min<long long>(n, f.per());
  for (int i = 0; i < n; i++) {
    const long long idx = (long long)i * f.per() / n + (i * 7919) % std::max(1LL, f.per() / n);
    bad += h[idx] != dfill_ref(f.c0, f.c1, (unsigned long long)chunk * f.per() + idx, f.NC, f.S);
  }
  return bad;
}
