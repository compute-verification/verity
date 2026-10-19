// H1 + fx1's unit (`pouw.circuit.h1tile`, eps1-sm120-h1fx1) on the device: every live 32 x 32 unit of a step's served calls,
// its receipt bit for bit `h1tile.reference`'s (and `served_h1/receipts.py`'s), in two plain launches on the caller's stream
// into caller-owned buffers (no allocation, no host sync), so a CUDA graph captures them.
//
//   h1r_seeds   one thread a (live unit, seed): seed_j and σ_R (`seeds.seed`, `seeds.receipt_seed`), SHA-512 of the 224-byte
//               tagged message, two compressions
//   h1r_units   one warp a unit.  Lane l derives row l's and column l's noise blocks itself (`seeds.noise_block`, one
//               compression each), 16 segments at a time, and lane l's fold keys from σ_R's block l.  Per 32-wide segment s
//               lane l writes row l of X = A + b and column l of W' = B + c + U into shared memory; the warp then runs the
//               tile's 2 x 4 mma.sync m16n8k32 s8 (lane 4 g + q's accumulators are exactly h1tile's lane entries, entry
//               4 (4 mi + ni) + e at acc[mi][ni][e]), so the accumulators are T_s, folded with fx1 + duo at once, and 8 more
//               against the identity, the residue sums Y.  After the last segment: the digits A_d from Y and the row's
//               digit coins, O' = T_last - A_0 U^T and D_d = A_d U^T (U as the B operand), folded in order.  After set s,
//               receipt word 2 s is the XOR over lanes and chains of y, 2 s + 1 of z (CP-W).
//
// A step is a table of calls (`H1Call`): A [T, K] and B [N, K] s8 row-major (16-byte aligned rows), the unit positions
// base (plus the step's, read from the device) + u, the call's roots and its row blocks' live bits (a unit is live iff its
// row block holds an admitted row, `receipts.gated_receipts`); a dead unit's receipt is zero words and nothing of it is
// hashed.  K % 512 == 0 (each stream's
// tail, the digit coins or U, is then one block of its own); the host wrapper refuses any other K.
#include <cstdint>
#include <cuda_runtime.h>

namespace sh {
typedef unsigned long long u64;
#include "../../codec_tensor/sha512.cuh"
}  // namespace sh
typedef unsigned long long u64;
typedef unsigned int u32;

struct H1Call {
  const int8_t* a;      // A [T, K]
  const int8_t* b;      // B [N, K]
  long long lda, ldb;   // row strides in bytes
  const u64* roots;     // root_a: 8 big-endian u64 words
  const u64* root_b;    // root_b: 8 big-endian u64 words
  const uint8_t* live;  // T / 32 row-block live bits
  u64 base;             // unit u's position is base + u
  int k, ncb, units;    // K, N / 32, (T / 32)(N / 32)
  int seed0;            // this call's first unit in the step's seed buffer
  long long out0;       // this call's first receipt word in the step's output
};

__host__ __device__ constexpr u32 mix32(u32 x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
__host__ __device__ constexpr u32 KA(int t) { return mix32(0xA0000000u + 2u * t) | 1u; }
__host__ __device__ constexpr u32 KK(int t) { return mix32(0xB0000000u + 2u * t + 1u); }
__host__ __device__ constexpr int ROT(int t) { return ((t * 7 + 5) % 31) + 1; }
__host__ __device__ constexpr u32 FX1(int q) { return mix32(0xA0000000u + 2u * (16u + 32u * q)) | 1u; }

__host__ __device__ inline int n_digits(int ks) {
  long long p = 128;  // 128^m
  for (int m = 1;; ++m, p *= 128)
    if (p >= 128LL * ks && (127LL * ks) / (p / 128) <= 127) return m;
}

__device__ __forceinline__ u32 rotl(u32 x, int r) { return __funnelshift_l(x, x, r); }

// per-byte a + b mod 2^8
__device__ __forceinline__ u32 badd(u32 a, u32 b) {
  return ((a & 0x7F7F7F7Fu) + (b & 0x7F7F7F7Fu)) ^ ((a ^ b) & 0x80808080u);
}
// bits 0..3 of x, one to each byte's bit 0
__device__ __forceinline__ u32 spread4(u32 x) { return ((x & 0xFu) * 0x00204081u) & 0x01010101u; }

__device__ __forceinline__ void mma(int c[4], const u32 a[4], u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// block t of seed sd's stream as 16 big-endian halves
__device__ __forceinline__ void nblock(const u64* sd, u64 t, u64 nt0, u64 nt1, u64 nt2, u32 h[16]) {
  u64 w[16], s[8];
  w[0] = nt0;
  w[1] = nt1;
  w[2] = nt2;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[3 + j] = sd[j];
  w[11] = t;
  w[12] = 0x8000000000000000ULL;
  w[13] = 0;
  w[14] = 0;
  w[15] = 96 * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j] = sh::kSha512Iv[j];
  sh::sha512_compress(s, w);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    h[2 * j] = (u32)(s[j] >> 32);
    h[2 * j + 1] = (u32)s[j];
  }
}

// one set into the lane's state (fx1 then duo), and its receipt words
__device__ __forceinline__ void absorb(u32 st[16], const int p[2][4][4], u32* o, int s, int lane, int nowrite) {
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    const int h = 4 * (t % 2);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const u32 v = (u32)p[t / 4][t % 4][e] * FX1(4 * t + e);
      const u32 u = st[h + e] ^ v ^ KK(t);
      st[h + e] = u * KA(t) + st[8 + h + e];
      st[8 + h + e] = rotl(u, ROT(t)) + st[h + e];
    }
  }
  u32 wy = st[0] ^ st[1] ^ st[2] ^ st[3] ^ st[4] ^ st[5] ^ st[6] ^ st[7];
  u32 wz = st[8] ^ st[9] ^ st[10] ^ st[11] ^ st[12] ^ st[13] ^ st[14] ^ st[15];
  wy = __reduce_xor_sync(0xFFFFFFFFu, wy);
  wz = __reduce_xor_sync(0xFFFFFFFFu, wz);
  if (lane == 0 && !nowrite) *reinterpret_cast<uint2*>(o + 2 * s) = make_uint2(wy, wz);
}

// the seeds of every live unit: seeds[(seed0 + u) * 2 + kind] (8 words), kind 0 seed_j, 1 σ_R
extern "C" __global__ void h1r_seeds(const H1Call* __restrict__ calls, int ncalls, const u64* __restrict__ salt,
                                     const u64* __restrict__ step_base, u64 st0, u64 st1, u64 st2, u64 rt0, u64 rt1, u64 rt2,
                                     u64* __restrict__ seeds) {
  const int c = blockIdx.y;
  if (c >= ncalls) return;
  const H1Call C = calls[c];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * C.units) return;
  const int u = i >> 1, kind = i & 1;
  if (!C.live[u / C.ncb]) return;
  const u64 p = C.base + (step_base ? *step_base : 0ULL) + u;
  u64 w[16], s[8];
  w[0] = p;
  w[1] = (kind ? rt0 : st0) ^ p;
  w[2] = (kind ? rt1 : st1) ^ p;
  w[3] = (kind ? rt2 : st2) ^ p;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[4 + j] = salt[j] ^ p;
#pragma unroll
  for (int j = 0; j < 4; ++j) w[12 + j] = C.roots[j] ^ p;
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j] = sh::kSha512Iv[j];
  sh::sha512_compress(s, w);
#pragma unroll
  for (int j = 0; j < 4; ++j) w[j] = C.roots[4 + j] ^ p;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[4 + j] = C.root_b[j] ^ p;
  w[12] = 0x8000000000000000ULL;
  w[13] = 0;
  w[14] = 0;
  w[15] = 224 * 8;
  sh::sha512_compress(s, w);
  u64* o = seeds + ((long long)(C.seed0 + u) * 2 + kind) * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = s[j];
}

constexpr int XS = 48;  // shared row stride in bytes: lane (g, q)'s fragment words fall in 32 distinct banks
constexpr int NZ = 17;  // shared noise stride in words: lane l's word s16 in bank (17 l + s16) % 32

__device__ __forceinline__ void sha_round(u64 v[8], u64 k, u64 w) {
  const u64 a = v[0], b = v[1], c = v[2], e = v[4], f = v[5], g = v[6];
  const u64 t1 = v[7] + (sh::rotr64(e, 14) ^ sh::rotr64(e, 18) ^ sh::rotr64(e, 41)) + ((e & f) ^ (~e & g)) + k + w;
  const u64 t2 = (sh::rotr64(a, 28) ^ sh::rotr64(a, 34) ^ sh::rotr64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
  v[7] = g;
  v[6] = f;
  v[5] = e;
  v[4] = v[3] + t1;
  v[3] = c;
  v[2] = b;
  v[1] = a;
  v[0] = t1 + t2;
}

// a noise block's state after rounds 0 .. 10, which read only the tag and the seed (words 0 .. 10): one a unit
__device__ __forceinline__ void noise_mid(const u64* sd, u64 nt0, u64 nt1, u64 nt2, u64 mid[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) mid[j] = sh::kSha512Iv[j];
  sha_round(mid, sh::kSha512K[0], nt0);
  sha_round(mid, sh::kSha512K[1], nt1);
  sha_round(mid, sh::kSha512K[2], nt2);
#pragma unroll
  for (int j = 0; j < 8; ++j) sha_round(mid, sh::kSha512K[3 + j], sd[j]);
}

// `nblock` from the unit's midstate: rounds 11 .. 79 only
__device__ __forceinline__ void nblock_mid(const u64* mid, const u64* sd, u64 t, u64 nt0, u64 nt1, u64 nt2, u32 h[16]) {
  u64 w[16], v[8];
  w[0] = nt0;
  w[1] = nt1;
  w[2] = nt2;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[3 + j] = sd[j];
  w[11] = t;
  w[12] = 0x8000000000000000ULL;
  w[13] = 0;
  w[14] = 0;
  w[15] = 96 * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = mid[j];
#pragma unroll
  for (int j = 11; j < 16; ++j) sha_round(v, sh::kSha512K[j], w[j]);
#pragma unroll 1
  for (int r = 16; r < 80; r += 16) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const u64 w15 = w[(j + 1) & 15], w2 = w[(j + 14) & 15];
      w[j] += (sh::rotr64(w15, 1) ^ sh::rotr64(w15, 8) ^ (w15 >> 7)) + w[(j + 9) & 15] +
              (sh::rotr64(w2, 19) ^ sh::rotr64(w2, 61) ^ (w2 >> 6));
      sha_round(v, sh::kSha512K[r + j], w[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const u64 x = sh::kSha512Iv[j] + v[j];
    h[2 * j] = (u32)(x >> 32);
    h[2 * j + 1] = (u32)x;
  }
}

__device__ __forceinline__ void prefetch_l2(const void* p, u32 bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
}

// V 1: the first kernel.  V 2: a column block's row blocks on adjacent warps (B's tile from L2 when T > 32); noise blocks
// from the unit's midstate; each chunk's noise words in shared memory; the next segment's A and B words loaded a segment
// ahead and the next chunk's rows prefetched into L2; at most 170 registers (3 blocks of 4 warps an SM).
// unit v of call C (v in the version's unit order) on one warp, with the warp's shared buffers: Xs and Ws (32 * XS bytes
// each), sdw (24 words) and, for V 2, nzw (2 * 32 * NZ words)
template <int V>
__device__ __forceinline__ void h1r_unit(const H1Call& C, int v, int lane, uint8_t* Xs, uint8_t* Ws, u64* sdw, u32* nzw,
                                         const u64* __restrict__ seeds, u64 nt0, u64 nt1, u64 nt2, u32* __restrict__ out,
                                         int nowrite) {
  int ri, cj;
  if constexpr (V >= 2) {
    const int nrb = C.units / C.ncb;
    ri = v % nrb;
    cj = v / nrb;
  } else {
    ri = v / C.ncb;
    cj = v % C.ncb;
  }
  const int u = ri * C.ncb + cj;
  const int KS = C.k >> 5, m = n_digits(KS), sets = KS + m;
  u32* o = out + C.out0 + (long long)u * 2 * sets;
  if (!C.live[ri]) {
    if (!nowrite)
      for (int i = lane; i < 2 * sets; i += 32) o[i] = 0;
    return;
  }
  const int g = lane >> 2, q = lane & 3;
  if (lane < 16) sdw[lane] = seeds[(long long)(C.seed0 + u) * 16 + lane];
  __syncwarp();
  const u64* sj = sdw;
  const u64* sr = sdw + 8;
  const u64* mid = sdw + 16;
  if constexpr (V >= 2) {
    u64 mm[8];
    noise_mid(sj, nt0, nt1, nt2, mm);
    if (lane < 8) sdw[16 + lane] = mm[lane];
    __syncwarp();
  }
  const int nb = KS / 16, RB = nb + 1;
  const u64 rowb = (u64)lane * RB, colb = (u64)32 * RB + (u64)lane * RB;
  auto noise = [&](u64 t, u32 h[16]) {
    if constexpr (V >= 2)
      nblock_mid(mid, sj, t, nt0, nt1, nt2, h);
    else
      nblock(sj, t, nt0, nt1, nt2, h);
  };

  const int8_t* arow = C.a + (32LL * ri + lane) * C.lda;
  const int8_t* brow = C.b + (32LL * cj + lane) * C.ldb;
  if constexpr (V >= 2) {
    prefetch_l2(arow, 512);
    prefetch_l2(brow, 512);
  }
  u32 st[16];
  nblock(sr, lane, nt0, nt1, nt2, st);
  u32 zc[3];
  u32 Uw[8];
  {
    u32 h[16];
    noise(rowb + nb, h);
#pragma unroll
    for (int d = 0; d < 3; ++d) zc[d] = h[d];
    noise(colb + nb, h);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      u32 x = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const u32 by = (h[w] >> (8 * i)) & 0xFFu;
        x |= ((by + (by >= 129u ? 63u : 192u)) & 0xFFu) << (8 * i);  // -64 + by mod 129, mod 2^8
      }
      Uw[w] = x;
    }
  }
  // U as the B operand (k = residue, n = column): through Ws
#pragma unroll
  for (int w = 0; w < 8; ++w) *reinterpret_cast<u32*>(Ws + lane * XS + 4 * w) = Uw[w];
  __syncwarp();
  u32 UB[4][2], IB[4][2];
#pragma unroll
  for (int ni = 0; ni < 4; ++ni) {
    UB[ni][0] = *reinterpret_cast<const u32*>(Ws + (8 * ni + g) * XS + 4 * q);
    UB[ni][1] = *reinterpret_cast<const u32*>(Ws + (8 * ni + g) * XS + 16 + 4 * q);
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      u32 x = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) x |= (u32)(16 * hh + 4 * q + i == 8 * ni + g) << (8 * i);
      IB[ni][hh] = x;
    }
  }
  __syncwarp();

  int acc[2][4][4], Y[2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mi][ni][e] = Y[mi][ni][e] = 0;

  uint4 pa0, pa1, pb0, pb1;
  if constexpr (V >= 2) {
    pa0 = *reinterpret_cast<const uint4*>(arow);
    pa1 = *reinterpret_cast<const uint4*>(arow + 16);
    pb0 = *reinterpret_cast<const uint4*>(brow);
    pb1 = *reinterpret_cast<const uint4*>(brow + 16);
  }
  u32* nzr = nzw + lane * NZ;
  u32* nzc = nzw + 32 * NZ + lane * NZ;
  for (int ch = 0; ch < nb; ++ch) {
    u32 hr[16], hc[16];
    if constexpr (V >= 2) {
      if (ch + 1 < nb) {
        prefetch_l2(arow + 512 * (ch + 1), 512);
        prefetch_l2(brow + 512 * (ch + 1), 512);
      }
      noise(rowb + ch, hr);
#pragma unroll
      for (int j = 0; j < 16; ++j) nzr[j] = hr[j];
      noise(colb + ch, hc);
#pragma unroll
      for (int j = 0; j < 16; ++j) nzc[j] = hc[j];
    } else {
      noise(rowb + ch, hr);
      noise(colb + ch, hc);
    }
#pragma unroll
    for (int s16 = 0; s16 < 16; ++s16) {
      const int s = 16 * ch + s16;
      uint4 a0, a1, b0, b1;
      u32 nr, nc;
      if constexpr (V >= 2) {
        a0 = pa0;
        a1 = pa1;
        b0 = pb0;
        b1 = pb1;
        nr = nzr[s16];
        nc = nzc[s16];
      } else {
        a0 = *reinterpret_cast<const uint4*>(arow + 32 * s);
        a1 = *reinterpret_cast<const uint4*>(arow + 32 * s + 16);
        b0 = *reinterpret_cast<const uint4*>(brow + 32 * s);
        b1 = *reinterpret_cast<const uint4*>(brow + 32 * s + 16);
        nr = hr[s16];
        nc = hc[s16];
      }
      const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const u32 bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
      u32 xw[8], ww[8];
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        xw[w] = badd(av[w], spread4(nr >> (4 * w)));
        ww[w] = badd(badd(bv[w], Uw[w]), spread4(nc >> (4 * w)));
      }
      *reinterpret_cast<uint4*>(Xs + lane * XS) = make_uint4(xw[0], xw[1], xw[2], xw[3]);
      *reinterpret_cast<uint4*>(Xs + lane * XS + 16) = make_uint4(xw[4], xw[5], xw[6], xw[7]);
      *reinterpret_cast<uint4*>(Ws + lane * XS) = make_uint4(ww[0], ww[1], ww[2], ww[3]);
      *reinterpret_cast<uint4*>(Ws + lane * XS + 16) = make_uint4(ww[4], ww[5], ww[6], ww[7]);
      if constexpr (V >= 2) {
        if (s + 1 < KS) {
          pa0 = *reinterpret_cast<const uint4*>(arow + 32 * (s + 1));
          pa1 = *reinterpret_cast<const uint4*>(arow + 32 * (s + 1) + 16);
          pb0 = *reinterpret_cast<const uint4*>(brow + 32 * (s + 1));
          pb1 = *reinterpret_cast<const uint4*>(brow + 32 * (s + 1) + 16);
        }
      }
      __syncwarp();
      u32 af[2][4], bf[4][2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        af[mi][0] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 4 * q);
        af[mi][1] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 4 * q);
        af[mi][2] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 16 + 4 * q);
        af[mi][3] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 16 + 4 * q);
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        bf[ni][0] = *reinterpret_cast<const u32*>(Ws + (8 * ni + g) * XS + 4 * q);
        bf[ni][1] = *reinterpret_cast<const u32*>(Ws + (8 * ni + g) * XS + 16 + 4 * q);
      }
      __syncwarp();
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          mma(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
          mma(Y[mi][ni], af[mi], IB[ni][0], IB[ni][1]);
        }
      absorb(st, acc, o, s, lane, nowrite);
    }
  }

  // the digits of Y (entry (mi, ni, e): row 16 mi + g + 8 (e >> 1), residue 8 ni + 2 q + (e & 1)), one at a time
  u32 zr[4][3];
#pragma unroll
  for (int rs = 0; rs < 4; ++rs)
#pragma unroll
    for (int d = 0; d < 3; ++d) zr[rs][d] = __shfl_sync(0xFFFFFFFFu, zc[d], g + 8 * rs);
  for (int d = 0; d < m; ++d) {
    u32 zd4[4];
#pragma unroll
    for (int rs = 0; rs < 4; ++rs) zd4[rs] = d == 0 ? zr[rs][0] : d == 1 ? zr[rs][1] : zr[rs][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int row = 16 * mi + g + 8 * (e >> 1), rho = 8 * ni + 2 * q + (e & 1);
          const int R = Y[mi][ni][e];
          int dig;
          if (d < m - 1) {
            const int qq = R >> 7;
            u32 zd = (zd4[2 * mi + (e >> 1)] >> rho) & 1u;
            if (d == m - 2 && qq == 127) zd = 0;
            dig = (R & 127) | (int)(zd << 7);
            Y[mi][ni][e] = qq + (int)zd;
          } else {
            dig = R & 0xFF;
          }
          Xs[row * XS + rho] = (uint8_t)dig;
        }
    __syncwarp();
    u32 af[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      af[mi][0] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 4 * q);
      af[mi][1] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 4 * q);
      af[mi][2] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 16 + 4 * q);
      af[mi][3] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 16 + 4 * q);
    }
    __syncwarp();
    int P[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
        for (int e = 0; e < 4; ++e) P[mi][ni][e] = 0;
        mma(P[mi][ni], af[mi], UB[ni][0], UB[ni][1]);
      }
    if (d == 0) {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
          for (int e = 0; e < 4; ++e) P[mi][ni][e] = (int)((u32)acc[mi][ni][e] - (u32)P[mi][ni][e]);
    }
    absorb(st, P, o, KS + d, lane, nowrite);
  }
}

template <int NW, int V>
__global__ void __launch_bounds__(32 * NW, V >= 2 ? 3 : 1) h1r_units_k(const H1Call* __restrict__ calls, int ncalls,
                                                                       const u64* __restrict__ seeds, u64 nt0, u64 nt1,
                                                                       u64 nt2, u32* __restrict__ out, int nowrite) {
  __shared__ __align__(16) uint8_t smem[NW][2][32 * XS];
  __shared__ u64 sd[NW][24];
  __shared__ u32 nzs[V >= 2 ? NW : 1][2][32 * NZ];
  const int c = blockIdx.y;
  if (c >= ncalls) return;
  const H1Call C = calls[c];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int v = blockIdx.x * NW + wid;
  if (v >= C.units) return;
  h1r_unit<V>(C, v, lane, smem[wid][0], smem[wid][1], sd[wid], nzs[V >= 2 ? wid : 0][0], seeds, nt0, nt1, nt2, out, nowrite);
}

static u64 be64(const char* s, int at) {
  u64 x = 0;
  for (int i = 0; i < 8; ++i) x = (x << 8) | (u64)(uint8_t)s[at + i];
  return x;
}

// a 24-byte tag (the name, zero-padded) as three big-endian words
static void tag3(const char* name, u64 out[3]) {
  char b[24] = {0};
  for (int i = 0; name[i] && i < 24; ++i) b[i] = name[i];
  for (int j = 0; j < 3; ++j) out[j] = be64(b, 8 * j);
}

constexpr int NW = 4;

// step_base (device, may be null) is added to every call's base, so one call table serves every step of a window
extern "C" int h1r_launch(const H1Call* calls_dev, int ncalls, int max_units, const int* ks, const u64* salt_dev,
                          const u64* step_base, u64* seeds_dev, u32* out_dev, int nowrite, int version, cudaStream_t stream) {
  for (int i = 0; i < ncalls; ++i)
    if (ks[i] % 512 || ks[i] < 512) return 1;
  u64 st[3], rt[3], nt[3];
  tag3("verity/pouw/seed/v2", st);
  tag3("verity/pouw/receipt/v1", rt);
  tag3("verity/pouw/noise/v1", nt);
  h1r_seeds<<<dim3((2 * max_units + 127) / 128, ncalls), 128, 0, stream>>>(calls_dev, ncalls, salt_dev, step_base, st[0],
                                                                          st[1], st[2], rt[0], rt[1], rt[2], seeds_dev);
  const dim3 grid((max_units + NW - 1) / NW, ncalls);
  if (version == 0) return (int)cudaGetLastError();  // the seeds alone (h1f computes the units)
  if (version == 1)
    h1r_units_k<NW, 1><<<grid, 32 * NW, 0, stream>>>(calls_dev, ncalls, seeds_dev, nt[0], nt[1], nt[2], out_dev, nowrite);
  else
    h1r_units_k<NW, 2><<<grid, 32 * NW, 0, stream>>>(calls_dev, ncalls, seeds_dev, nt[0], nt[1], nt[2], out_dev, nowrite);
  return (int)cudaGetLastError();
}

extern "C" int h1r_call_bytes() { return (int)sizeof(H1Call); }
extern "C" int h1r_n_digits(int ks) { return n_digits(ks); }
