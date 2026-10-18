// The integer-pipe candidates beside full SHA-512, one interface: SHA-512 itself (compute-accounting's compression,
// benchmarks/pouw/commit_gpu/sha512.cuh, unedited), SHA-512 at R < 80 rounds (the same round and schedule code with the
// round count a compile-time constant: a speed curve, not a hash anyone serves), SHA-256 (FIPS 180-4) and BLAKE3
// (archive/pouw-blake3/pearl_c/hash.cuh's compression, unedited).
//
// Each candidate H has: BLOCK (bytes a compression takes), state words of type `W`; `iv`, `comp` (one compression, the
// message words' 16-word ring left as the schedule leaves it), `rate_step` (the raw rate's chained step: the block's first
// words XOR the state, then a compression), `leaf` (the standard hash of one 4096-byte leaf, its blocks from a
// WarpLoader; TPL threads a leaf, each a stream of STRIDE bytes) and `node` (the standard hash of left || right).
#pragma once
#include <cstdint>

#include "../../pouw/commit_gpu/sha512.cuh"
#include "../../../archive/pouw-blake3/pearl_c/hash.cuh"

namespace ip {
using u32 = uint32_t;
using u64 = uint64_t;
constexpr int LEAF = 4096;

__device__ __forceinline__ u32 bswap32(u32 x) { return __byte_perm(x, 0, 0x0123); }

// A warp's 32 threads each read a stream of BLOCK-byte blocks, stream s at base + s STRIDE, FETCH bytes of each stream
// at a time (the same for every hash, so the DRAM pattern is the loader's, not the hash's): when b starts a fetch,
// `block(b, q)` loads the 32 streams' next FETCH bytes cooperatively (each load instruction covers whole 128-byte lines)
// into this warp's shared rows (padded by one uint4, so the per-thread reads hit distinct banks); then it gives the
// thread block b of its row. Every lane calls it with the same b, in order.
#ifndef FETCH
#define FETCH 128
#endif
template <int BLOCK, int STRIDE>
struct WarpLoader {
  static_assert(FETCH % BLOCK == 0 && STRIDE % FETCH == 0, "FETCH: a multiple of the block, dividing the stream");
  static constexpr int G = FETCH / BLOCK, BQ = BLOCK / 16, Q = FETCH / 16, ROW = Q + 1;
  static constexpr int SMEM = 32 * ROW * 16;   // bytes a warp
  const uint8_t* base;
  uint4* sm;
  int lane;
  __device__ __forceinline__ void block(int b, uint4* q) const {
    if (b % G == 0) {
      __syncwarp();
#pragma unroll
      for (int k = 0; k < Q; ++k) {
        const int i = k * 32 + lane, s = i / Q, o = i % Q;
        sm[s * ROW + o] = __ldg(reinterpret_cast<const uint4*>(base + (size_t)s * STRIDE + (size_t)(b / G) * FETCH) + o);
      }
      __syncwarp();
    }
    const int off = lane * ROW + (b % G) * BQ;
#pragma unroll
    for (int k = 0; k < BQ; ++k) q[k] = sm[off + k];
  }
};

// --- SHA-512 at R rounds: sha512.cuh's compress_from<0> with the round count R ----------------------------------------
template <int R>
__device__ __forceinline__ void sha512_rounds(u64* st, u64* w) {
  static_assert(R >= 16 && R <= 80, "16..80 rounds");
  using namespace sha;
  u64 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const u64 kw = add(K_dev[i], w[i]);
    const u64 t1 = add3(h, S1(e), ch(e, f, g));
    const u64 t1b = add(t1, kw);
    const u64 mj = maj(a, b, c);
    h = g; g = f; f = e; e = add(d, t1b);
    d = c; c = b; b = a; a = add3(t1b, S0(b), mj);
  }
  constexpr int FULL = (R - 16) / 16, REM = (R - 16) % 16;
  SHA_ROLL_PRAGMA
  for (int j = 16; j < 16 + 16 * FULL; j += 16) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const u64 wi = add3(add(s1(w[(i - 2) & 15]), w[(i - 7) & 15]), s0(w[(i - 15) & 15]), w[i]);
      w[i] = wi;
      const u64 t1 = add3(h, S1(e), ch(e, f, g));
      const u64 t1b = add3(t1, K_dev[j + i], wi);
      const u64 mj = maj(a, b, c);
      h = g; g = f; f = e; e = add(d, t1b);
      d = c; c = b; b = a; a = add3(t1b, S0(b), mj);
    }
  }
#pragma unroll
  for (int i = 0; i < REM; ++i) {
    const u64 wi = add3(add(s1(w[(i - 2) & 15]), w[(i - 7) & 15]), s0(w[(i - 15) & 15]), w[i]);
    w[i] = wi;
    const u64 t1 = add3(h, S1(e), ch(e, f, g));
    const u64 t1b = add3(t1, K_dev[16 + 16 * FULL + i], wi);
    const u64 mj = maj(a, b, c);
    h = g; g = f; f = e; e = add(d, t1b);
    d = c; c = b; b = a; a = add3(t1b, S0(b), mj);
  }
  st[0] = add(st[0], a); st[1] = add(st[1], b); st[2] = add(st[2], c); st[3] = add(st[3], d);
  st[4] = add(st[4], e); st[5] = add(st[5], f); st[6] = add(st[6], g); st[7] = add(st[7], h);
}

// R = 0: sha::compress itself, the baseline; R in 16..80: sha512_rounds<R>
template <int R>
struct Sha512 {
  using W = u64;
  static constexpr int BLOCK = 128, TPL = 1, STRIDE = LEAF;
  static __device__ __forceinline__ void iv(u64* s) { sha::iv(s); }
  static __device__ __forceinline__ void comp(u64* s, u64* w) {
    if constexpr (R == 0) sha::compress(s, w);
    else sha512_rounds<R>(s, w);
  }
  static __device__ __forceinline__ void rate_step(u64* s, u64* w, int) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= s[i];
    comp(s, w);
  }
  template <class L>
  static __device__ __forceinline__ void leaf(const L& ld, u64 t, u64* out) {
    u64 w[16], st[8];
    iv(st);
#pragma unroll 1
    for (int blk = 0; blk < LEAF / 128; ++blk) {
      uint4 v[8];
      ld.block(blk, v);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        w[2 * q] = (u64)bswap32(v[q].x) << 32 | bswap32(v[q].y);
        w[2 * q + 1] = (u64)bswap32(v[q].z) << 32 | bswap32(v[q].w);
      }
      comp(st, w);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = 0;
    w[0] = 0x8000000000000000ull;
    w[15] = 8ull * LEAF;
    comp(st, w);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[t * 8 + i] = st[i];
  }
  static __device__ __forceinline__ void node(const u64* l, const u64* r, u64* out) {
    u64 w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) { w[i] = l[i]; w[i + 8] = r[i]; }
    iv(out);
    comp(out, w);
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = 0;
    w[0] = 0x8000000000000000ull;
    w[15] = 1024;
    comp(out, w);
  }
};

// --- SHA-256 (FIPS 180-4), the same form: 16 rounds straight, the rest as rolled 16-round bodies ----------------------
__constant__ u32 K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ __forceinline__ u32 r32(u32 x, int n) { return __funnelshift_r(x, x, n); }
__device__ __forceinline__ u32 B0(u32 a) { return r32(a, 2) ^ r32(a, 13) ^ r32(a, 22); }
__device__ __forceinline__ u32 B1(u32 e) { return r32(e, 6) ^ r32(e, 11) ^ r32(e, 25); }
__device__ __forceinline__ u32 b0(u32 w) { return r32(w, 7) ^ r32(w, 18) ^ (w >> 3); }
__device__ __forceinline__ u32 b1(u32 w) { return r32(w, 17) ^ r32(w, 19) ^ (w >> 10); }
__device__ __forceinline__ u32 ch32(u32 e, u32 f, u32 g) { return (e & f) ^ (~e & g); }
__device__ __forceinline__ u32 maj32(u32 a, u32 b, u32 c) { return (a & b) | (c & (a | b)); }

__device__ __forceinline__ void sha256_comp(u32* st, u32* w) {
  u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const u32 t1 = h + B1(e) + ch32(e, f, g) + K256[i] + w[i];
    const u32 mj = maj32(a, b, c);
    h = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + B0(b) + mj;
  }
  SHA_ROLL_PRAGMA
  for (int j = 16; j < 64; j += 16) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const u32 wi = b1(w[(i - 2) & 15]) + w[(i - 7) & 15] + b0(w[(i - 15) & 15]) + w[i];
      w[i] = wi;
      const u32 t1 = h + B1(e) + ch32(e, f, g) + K256[j + i] + wi;
      const u32 mj = maj32(a, b, c);
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + B0(b) + mj;
    }
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

struct Sha256 {
  using W = u32;
  static constexpr int BLOCK = 64, TPL = 1, STRIDE = LEAF;
  static __device__ __forceinline__ void iv(u32* s) {
    s[0] = 0x6a09e667; s[1] = 0xbb67ae85; s[2] = 0x3c6ef372; s[3] = 0xa54ff53a;
    s[4] = 0x510e527f; s[5] = 0x9b05688c; s[6] = 0x1f83d9ab; s[7] = 0x5be0cd19;
  }
  static __device__ __forceinline__ void comp(u32* s, u32* w) { sha256_comp(s, w); }
  static __device__ __forceinline__ void rate_step(u32* s, u32* w, int) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= s[i];
    comp(s, w);
  }
  template <class L>
  static __device__ __forceinline__ void leaf(const L& ld, u64 t, u32* out) {
    u32 w[16], st[8];
    iv(st);
#pragma unroll 1
    for (int blk = 0; blk < LEAF / 64; ++blk) {
      uint4 v[4];
      ld.block(blk, v);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        w[4 * q] = bswap32(v[q].x); w[4 * q + 1] = bswap32(v[q].y); w[4 * q + 2] = bswap32(v[q].z); w[4 * q + 3] = bswap32(v[q].w);
      }
      comp(st, w);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = 0;
    w[0] = 0x80000000u;
    w[15] = 8u * LEAF;
    comp(st, w);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[t * 8 + i] = st[i];
  }
  static __device__ __forceinline__ void node(const u32* l, const u32* r, u32* out) {
    u32 w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) { w[i] = l[i]; w[i + 8] = r[i]; }
    iv(out);
    comp(out, w);
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = 0;
    w[0] = 0x80000000u;
    w[15] = 512;
    comp(out, w);
  }
};

// --- BLAKE3: hash.cuh's b3c; a 4096-byte leaf is BLAKE3's 4 chunks of 16 blocks and 3 parents, a node one block ---------
struct Blake3 {
  using W = u32;
  static constexpr int BLOCK = 64, TPL = 1, STRIDE = LEAF;
  static __device__ __forceinline__ void iv(u32* s) {
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = B3_IV[i];
  }
  static __device__ __forceinline__ void comp(u32* s, u32* w) { b3c(s, w, 0, 64, 0, s); }
  static __device__ __forceinline__ void rate_step(u32* s, u32* w, int r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= s[i];
    b3c(s, w, (uint64_t)r, 64, 0, s);
  }
  // hash.cuh's b3_parent is keyed; the unkeyed hash's parent is IV, PARENT
  static __device__ __forceinline__ void parent(const u32* l, const u32* r, u32 flags, u32* out) {
    u32 m[16], key[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = l[i]; m[i + 8] = r[i]; }
    iv(key);
    b3c(key, m, 0, 64, B3_PARENT | flags, out);
  }
  // chunk c (1024 bytes) of a stream whose blocks b0 .. b0 + 15 it is
  template <class L>
  static __device__ __forceinline__ void chunk(const L& ld, int b0, u64 c, u32* cv) {
    u32 m[16];
    iv(cv);
#pragma unroll 1
    for (int blk = 0; blk < 16; ++blk) {
      uint4 v[4];
      ld.block(b0 + blk, v);
#pragma unroll
      for (int q = 0; q < 4; ++q) { m[4 * q] = v[q].x; m[4 * q + 1] = v[q].y; m[4 * q + 2] = v[q].z; m[4 * q + 3] = v[q].w; }
      b3c(cv, m, c, 64, (blk == 0 ? B3_START : 0) | (blk == 15 ? B3_END : 0), cv);
    }
  }
  template <class L>
  static __device__ __forceinline__ void leaf(const L& ld, u64 t, u32* out) {
    u32 a[8], b[8], l[8], r[8], h[8];
    chunk(ld, 0, 0, a);
    chunk(ld, 16, 1, b);
    parent(a, b, 0, l);
    chunk(ld, 32, 2, a);
    chunk(ld, 48, 3, b);
    parent(a, b, 0, r);
    parent(l, r, B3_ROOT, h);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[t * 8 + i] = h[i];
  }
  static __device__ __forceinline__ void node(const u32* l, const u32* r, u32* out) {
    u32 m[16], key[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = l[i]; m[i + 8] = r[i]; }
    iv(key);
    b3c(key, m, 0, 64, B3_START | B3_END | B3_ROOT, out);
  }
};

// The same hash, chunk-parallel as BLAKE3 is meant to run: 4 threads a leaf, thread t chunk t % 4, then the parents across
// lanes (every lane computes both parent levels; lanes t % 4 == 0 hold the root)
struct Blake3Chunks : Blake3 {
  static constexpr int TPL = 4, STRIDE = 1024;
  template <class L>
  static __device__ __forceinline__ void leaf(const L& ld, u64 t, u32* out) {
    u32 cv[8], o[8], p[8], h[8];
    chunk(ld, 0, t & 3, cv);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = __shfl_down_sync(0xffffffffu, cv[i], 1);
    parent(cv, o, 0, p);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = __shfl_down_sync(0xffffffffu, p[i], 2);
    parent(p, o, B3_ROOT, h);
    if ((t & 3) == 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i) out[(t >> 2) * 8 + i] = h[i];
    }
  }
};
}  // namespace ip
