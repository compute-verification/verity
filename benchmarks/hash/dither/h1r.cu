// H1 + fx1's unit (`pouw.circuit.h1tile`, eps1-sm120-h1fx1) on the device: every live 32 x 32 unit of a step's served calls,
// its receipt bit for bit `h1tile.reference`'s (and `served_h1/receipts.py`'s), in two plain launches on the caller's stream
// into caller-owned buffers (no allocation, no host sync), so a CUDA graph captures them.
//
//   h1r_seeds   one thread a (live unit, seed): seed_j and σ_R (`seeds.seed`, `seeds.receipt_seed`), SHA-512 of the 224-byte
//               tagged message, two compressions
//   h1r_units   one warp a unit.  Lane l derives row l's and column l's noise chunks itself (`seeds.noise_block`, one
//               compression each), 16 segments at a time, and lane l's fold keys from σ_R's block l.  Per 32-wide segment s
//               lane l writes row l of X = A + b and column l of W' = B + c + U into shared memory; the warp then runs the
//               tile's 2 x 4 mma.sync m16n8k32 s8 (lane 4 g + q's accumulators are exactly h1tile's lane entries, entry
//               4 (4 mi + ni) + e at acc[mi][ni][e]), so the accumulators are T_s, folded with fx1 + duo at once, and 8 more
//               against the identity, the residue sums Y.  After the last segment: the digits A_d from Y and the row's
//               digit coins, O' = T_last - A_0 U^T and D_d = A_d U^T (U as the B operand), folded in order.  After set s,
//               receipt word 2 s is the XOR over lanes and chains of y, 2 s + 1 of z (CP-W).
//               Version 3 (`h1b_unit`) runs the same unit on a block of five warps: four hash the noise and build the
//               operands, one folds.  Version 4 (`h1s_unit`) splits it six ways: two warps hash a chunk ahead, two
//               build the segments, and the fold is split in two by the fx1 + duo chains it feeds.
//               `h1r_launch_j10r` runs version 1 under J10-R's call-scale noise (`h1r_seeds_jr`, `h1r_units_jr_k`;
//               the verifier's form): the served J10-R units' receipts reference.
//
// A step is a table of calls (`H1Call`): A [T, K] and B [N, K] s8 row-major (16-byte aligned rows), the unit positions
// base (plus the step's, read from the device) + u, the call's roots and its row tiles' live bits (a unit is live iff its
// row tile holds an admitted row, `receipts.gated_receipts`); a dead unit's receipt is zero words and nothing of it is
// hashed.  K % 512 == 0 (each stream's
// tail, the digit coins or U, is then one block of its own); the host wrapper refuses any other K.
#include <cstdint>
#include <cuda_runtime.h>

#include "../../torch_gpu/sha512_rows/sha512x.cuh"
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
// 1 in each byte of x that is 0x80, else 0
__device__ __forceinline__ u32 is80(u32 x) {
  const u32 v = x ^ 0x80808080u;
  return (~(((v & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | v) & 0x80808080u) >> 7;
}

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
  sx::iv(s);
  sx::compress<0, 2>(s, w);
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

// the seed at position p under tag (t0, t1, t2): SHA-512 of the 224-byte message, position XORed into every word
__device__ __forceinline__ void unit_seed(const H1Call& C, u64 p, const u64* __restrict__ salt, u64 t0, u64 t1, u64 t2,
                                          u64 s[8]) {
  u64 w[16];
  w[0] = p;
  w[1] = t0 ^ p;
  w[2] = t1 ^ p;
  w[3] = t2 ^ p;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[4 + j] = salt[j] ^ p;
#pragma unroll
  for (int j = 0; j < 4; ++j) w[12 + j] = C.roots[j] ^ p;
  sx::iv(s);
  sx::compress<0, 2>(s, w);
#pragma unroll
  for (int j = 0; j < 4; ++j) w[j] = C.roots[4 + j] ^ p;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[4 + j] = C.root_b[j] ^ p;
  w[12] = 0x8000000000000000ULL;
  w[13] = 0;
  w[14] = 0;
  w[15] = 224 * 8;
  sx::compress<0, 2>(s, w);
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
  u64 s[8];
  unit_seed(C, C.base + (step_base ? *step_base : 0ULL) + u, salt, kind ? rt0 : st0, kind ? rt1 : st1, kind ? rt2 : st2, s);
  u64* o = seeds + ((long long)(C.seed0 + u) * 2 + kind) * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = s[j];
}

// h1r_seeds under J10-R (h1r_launch_j10r): kind 0 seed_c (`seeds.call_seed`, tag ct at the call's first position
// C.base + *step_base, the same for every unit of the call), kind 1 σ_R at the unit's own position
extern "C" __global__ void h1r_seeds_jr(const H1Call* __restrict__ calls, int ncalls, const u64* __restrict__ salt,
                                        const u64* __restrict__ step_base, u64 ct0, u64 ct1, u64 ct2, u64 rt0, u64 rt1,
                                        u64 rt2, u64* __restrict__ seeds) {
  const int c = blockIdx.y;
  if (c >= ncalls) return;
  const H1Call C = calls[c];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * C.units) return;
  const int u = i >> 1, kind = i & 1;
  if (!C.live[u / C.ncb]) return;
  const u64 call = C.base + (step_base ? *step_base : 0ULL);
  u64 s[8];
  unit_seed(C, kind ? call + u : call, salt, kind ? rt0 : ct0, kind ? rt1 : ct1, kind ? rt2 : ct2, s);
  u64* o = seeds + ((long long)(C.seed0 + u) * 2 + kind) * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = s[j];
}

constexpr int XS = 48;  // shared row stride in bytes: lane (g, q)'s fragment words fall in 32 distinct banks
constexpr int NZ = 17;  // shared noise stride in words: lane l's word s16 in bank (17 l + s16) % 32

__device__ __forceinline__ void sha_round(u64 v[8], u64 k, u64 w) {
  const u64 a = v[0], b = v[1], c = v[2], e = v[4], f = v[5], g = v[6];
  const u64 t1 = v[7] + (sx::rotr<0, 14>(e) ^ sx::rotr<0, 18>(e) ^ sx::rotr<0, 41>(e)) + ((e & f) ^ (~e & g)) + k + w;
  const u64 t2 = (sx::rotr<0, 28>(a) ^ sx::rotr<0, 34>(a) ^ sx::rotr<0, 39>(a)) + ((a & b) ^ (a & c) ^ (b & c));
  v[7] = g;
  v[6] = f;
  v[5] = e;
  v[4] = v[3] + t1;
  v[3] = c;
  v[2] = b;
  v[1] = a;
  v[0] = t1 + t2;
}

// a noise chunk's state after rounds 0 .. 10, which read only the tag and the seed (words 0 .. 10): one a unit.  This
// and nblock_mid read the IV from constant memory (sx::IV_DEV): with sx::iv's immediates version 2 runs 2-4% slower at
// T = 64, where nblock (version 1's) runs 1% faster with them
__device__ __forceinline__ void noise_mid(const u64* sd, u64 nt0, u64 nt1, u64 nt2, u64 mid[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) mid[j] = sx::IV_DEV[j];
  sha_round(mid, sx::K_DEV[0], nt0);
  sha_round(mid, sx::K_DEV[1], nt1);
  sha_round(mid, sx::K_DEV[2], nt2);
#pragma unroll
  for (int j = 0; j < 8; ++j) sha_round(mid, sx::K_DEV[3 + j], sd[j]);
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
  for (int j = 11; j < 16; ++j) sha_round(v, sx::K_DEV[j], w[j]);
#pragma unroll 1
  for (int r = 16; r < 80; r += 16) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const u64 w15 = w[(j + 1) & 15], w2 = w[(j + 14) & 15];
      w[j] += (sx::rotr<0, 1>(w15) ^ sx::rotr<0, 8>(w15) ^ (w15 >> 7)) + w[(j + 9) & 15] +
              (sx::rotr<0, 19>(w2) ^ sx::rotr<0, 61>(w2) ^ (w2 >> 6));
      sha_round(v, sx::K_DEV[r + j], w[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const u64 x = sx::IV_DEV[j] + v[j];
    h[2 * j] = (u32)(x >> 32);
    h[2 * j + 1] = (u32)x;
  }
}

__device__ __forceinline__ void prefetch_l2(const void* p, u32 bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
}

// h1o_band_launch's mode 3 launches the units as the band-rows pass's programmatic dependents, so every thread that reads
// the band workspace waits for the pass first (a no-op when the units were not launched as its dependents)
__device__ __forceinline__ void band_wait() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

// a band workspace's noise chunk (16 words, 64-byte aligned) into registers, and from them to a lane's noise row
__device__ __forceinline__ void band_load(const u32* src, uint4 (&v)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = __ldg(reinterpret_cast<const uint4*>(src) + i);
}
__device__ __forceinline__ void band_store(const uint4 (&v)[4], u32* d) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    d[4 * i] = v[i].x;
    d[4 * i + 1] = v[i].y;
    d[4 * i + 2] = v[i].z;
    d[4 * i + 3] = v[i].w;
  }
}

// The unit's served output (h1o.cu): with E::ON the lane's accumulators become O = O' - sum_d 128^d D_d as the tail
// absorbs O' and each D_d, plus the exact wrap correction 256 X (wb⁺ - wb⁻)^T (H1V3's PouwWrapSeg): each segment's wb
// from the B, c and U bytes its W' is made of, as an s8 {-1, 0, 1} operand in E's shared buffer (E::wbs, 32 * XS bytes),
// and one more MMA a fragment against the same X.  E writes z; it also says which A rows are the canonical empty row (read
// as zero), what a dead unit writes, and may perturb a noised accumulator word or flip a wb⁺ bit (the conformance tests).
// With E::PEEL (H1V3's PouwPeelSeg) z is A Bᵀ taken from the same O: z = O + 256 X (wb⁺ - wb⁻)ᵀ
// - X cᵀ - (b - 256 xa) Bᵀ, xa = [A = 127 and b = 1] (X = -128, where A + b wrapped), from X's own fragment and b.  The
// peel's MMAs are binary-by-int8 beside the wrap MMA (X against -c, -b and xa against B, B's bytes in the slot's fourth
// row); no product of A itself is formed, so z moves with every noised word.  E may flip an xa bit (its conformance test).
// The block and split forms have it.
// NoEpi: the receipts alone.
struct NoEpi {
  static constexpr bool ON = false;
  static constexpr bool PEEL = false;
  static constexpr bool SEEDS = false;  // with SEEDS, lanes 0 and 1 derive the unit's seed_j and σ_R (E::seed), not read them
  // with BAND (J10 banded, h1o.cu's h1o_band_launch; measured only), a unit's rows are its band leader's, read from the
  // band workspace E::band_rows gives (row l's block c at + 16 (l RB + c)), and only its columns are hashed in the body:
  // BAND 1 (block form: groups of 4 chunks, the rows read from the workspace in the segment loop) or 2 (staged: each row
  // block copied into the noise buffer where the per-unit form hashes it); the split form stages under either
  static constexpr int BAND = 0;
  __device__ const u32* band_rows(const H1Call&, int, int, int) const { return nullptr; }
  __device__ bool a_ok(int) const { return true; }
  __device__ void dead(int, int, int) const {}
  __device__ void fault(int, int, int, int (&)[2][4][4]) const {}
  __device__ void fault_word(int, int, int, int&) const {}
  __device__ void out(int, int, int, const int (&)[2][4][4]) const {}
  template <int F>
  __device__ void out_half(int, int, int, const int (&)[2][2][4]) const {}
};

// per byte, r = b + u + c mod 2^8 (c a bit a byte) read as s8 wraps where b + u + c leaves [-128, 127]: b and u share a sign
// that r doesn't.  P has bit 7 of each byte that wrapped up (wb⁺, r negative), N of each that wrapped down (wb⁻)
__device__ __forceinline__ void wrap_masks(u32 b, u32 u, u32 r, u32& P, u32& N) {
  const u32 v = ~(b ^ u) & (b ^ r) & 0x80808080u;
  P = v & r;
  N = v & ~r;
}
// wb⁺ - wb⁻ a byte as s8 (both bits set: 0, as PouwWrapSeg reads them)
__device__ __forceinline__ u32 wrap_s8(u32 P, u32 N) {
  return ((P ^ N) >> 7) | (((N & ~P) >> 7) * 0xFEu);
}

// E::PEEL's segment MMAs: Y += X Iᵀ as unpeeled, and cr += 256 (X wbᵀ + xa Bᵀ) - X cᵀ - b Bᵀ a fragment at a time (the
// 256 terms in a four-word temporary, so cr stays one accumulator), from the fragments of X (af), wb (wf) and B (bf) and
// the segment's noise words of rows g + 8 j (nb4) and of columns 8 j + g (nc4); the binary operands are made where they
// are read (xa and -b a row half at a time, -c a column fragment at a time), to keep the builders under their registers.
// xa B^T is skipped in a row half where no lane has an xa (A = 127 is rare in served rows)
template <class E>
__device__ __forceinline__ void peel_mmas(int (&Y)[2][4][4], int (&cr)[2][4][4], const u32 (&af)[2][4],
                                          const u32 (&wf)[4][2], const u32 (&bf)[4][2], const u32 (&IB)[4][2],
                                          const u32 (&nb4)[4], const u32 (&nc4)[4], int q, int u, int lane, int s,
                                          const E& ep) {
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    u32 xa[4], nb[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const u32 b = spread4(nb4[2 * mi + (e & 1)] >> (16 * (e >> 1) + 4 * q));
      xa[e] = is80(af[mi][e]) & b;
      nb[e] = b * 0xFFu;
    }
    ep.xa_fault(u, lane, s, mi, xa);
    const bool any_xa = __any_sync(0xFFFFFFFFu, (xa[0] | xa[1] | xa[2] | xa[3]) != 0u);
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const u32 nc0 = spread4(nc4[ni] >> (4 * q)) * 0xFFu, nc1 = spread4(nc4[ni] >> (16 + 4 * q)) * 0xFFu;
      int t[4] = {0, 0, 0, 0};
      mma(Y[mi][ni], af[mi], IB[ni][0], IB[ni][1]);
      mma(t, af[mi], wf[ni][0], wf[ni][1]);
      mma(cr[mi][ni], af[mi], nc0, nc1);
      if (any_xa) mma(t, xa, bf[ni][0], bf[ni][1]);
      mma(cr[mi][ni], nb, bf[ni][0], bf[ni][1]);
#pragma unroll
      for (int e = 0; e < 4; ++e) cr[mi][ni][e] = (int)((u32)cr[mi][ni][e] + ((u32)t[e] << 8));
    }
  }
}

// V 1: the first kernel.  V 2: a column tile's row tiles on adjacent warps (B's tile from L2 when T > 32); noise chunks
// from the unit's midstate; each chunk's noise words in shared memory; the next segment's A and B words loaded a segment
// ahead and the next chunk's rows prefetched into L2; at most 170 registers (3 blocks of 4 warps an SM).
// unit v of call C (v in the version's unit order) on one warp, with the warp's shared buffers: Xs and Ws (32 * XS bytes
// each), sdw (24 words) and, for V 2, nzw (2 * 32 * NZ words)
// band > 1 (V 1 only; h1r_launch_band, the banded gates' receipts reference): the rows from the band leader's seed_j,
// unit ri ncb + (cj / band) band's, read from the seeds buffer
// jrm > 0 (V 1 only; h1r_launch_j10r, J10-R's receipts reference): the seeds' first 8 words are the call's seed_c and the
// unit reads its call's stream at call scale, Layout(jrm, N, K): row 32 ri + l's blocks at (32 ri + l) RB, column
// 32 cj + l's at jrm RB + (32 cj + l) RB; a live unit whose row tile lies past jrm gets all-ones words (no such unit)
template <int V, class E = NoEpi>
__device__ __forceinline__ void h1r_unit(const H1Call& C, int v, int lane, uint8_t* Xs, uint8_t* Ws, u64* sdw, u32* nzw,
                                         const u64* __restrict__ seeds, u64 nt0, u64 nt1, u64 nt2, u32* __restrict__ out,
                                         int nowrite, const E& ep = E(), int band = 1, int jrm = 0) {
  static_assert(!E::PEEL, "the warp form serves the dithered product only");
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
    if constexpr (E::ON) ep.dead(ri, cj, lane);
    return;
  }
  const bool jr = V == 1 && jrm > 0;
  if (jr && 32 * (ri + 1) > jrm) {
    if (!nowrite)
      for (int i = lane; i < 2 * sets; i += 32) o[i] = 0xFFFFFFFFu;
    return;
  }
  const int g = lane >> 2, q = lane & 3;
  if constexpr (E::SEEDS) {
    if (lane < 2) ep.seed(C, u, lane, sdw + 8 * lane);
  } else if (lane < 16) {
    sdw[lane] = seeds[(long long)(C.seed0 + u) * 16 + lane];
  }
  const bool banded = V == 1 && band > 1;
  if (banded && lane < 8) sdw[16 + lane] = seeds[(long long)(C.seed0 + ri * C.ncb + (cj / band) * band) * 16 + lane];
  __syncwarp();
  const u64* sj = sdw;
  const u64* sr = sdw + 8;
  const u64* mid = sdw + 16;
  const u64* sjr = banded ? sdw + 16 : sdw;
  if constexpr (V >= 2) {
    u64 mm[8];
    noise_mid(sj, nt0, nt1, nt2, mm);
    if (lane < 8) sdw[16 + lane] = mm[lane];
    __syncwarp();
  }
  const int nb = KS / 16, RB = nb + 1;
  const u64 rowb = jr ? (u64)(32 * ri + lane) * RB : (u64)lane * RB;
  const u64 colb = jr ? (u64)jrm * RB + (u64)(32 * cj + lane) * RB : (u64)32 * RB + (u64)lane * RB;
  auto noise = [&](u64 t, u32 h[16]) {
    if constexpr (V >= 2)
      nblock_mid(mid, sj, t, nt0, nt1, nt2, h);
    else
      nblock(sj, t, nt0, nt1, nt2, h);
  };
  auto rnoise = [&](u64 t, u32 h[16]) {
    if constexpr (V >= 2)
      nblock_mid(mid, sj, t, nt0, nt1, nt2, h);
    else
      nblock(sjr, t, nt0, nt1, nt2, h);
  };

  const int8_t* arow = C.a + (32LL * ri + lane) * C.lda;
  const int8_t* brow = C.b + (32LL * cj + lane) * C.ldb;
  const bool aok = ep.a_ok(32 * ri + lane);
  const uint4 zero4 = make_uint4(0, 0, 0, 0);
  if constexpr (V >= 2) {
    if (aok) prefetch_l2(arow, 512);
    prefetch_l2(brow, 512);
  }
  u32 st[16];
  nblock(sr, lane, nt0, nt1, nt2, st);
  u32 zc[3];
  u32 Uw[8];
  {
    u32 h[16];
    rnoise(rowb + nb, h);
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

  int acc[2][4][4], Y[2][4][4], cr[2][4][4];  // cr: X wbᵀ (E::ON)
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mi][ni][e] = Y[mi][ni][e] = cr[mi][ni][e] = 0;

  uint4 pa0, pa1, pb0, pb1;
  if constexpr (V >= 2) {
    pa0 = aok ? *reinterpret_cast<const uint4*>(arow) : zero4;
    pa1 = aok ? *reinterpret_cast<const uint4*>(arow + 16) : zero4;
    pb0 = *reinterpret_cast<const uint4*>(brow);
    pb1 = *reinterpret_cast<const uint4*>(brow + 16);
  }
  u32* nzr = nzw + lane * NZ;
  u32* nzc = nzw + 32 * NZ + lane * NZ;
  for (int ch = 0; ch < nb; ++ch) {
    u32 hr[16], hc[16];
    if constexpr (V >= 2) {
      if (ch + 1 < nb) {
        if (aok) prefetch_l2(arow + 512 * (ch + 1), 512);
        prefetch_l2(brow + 512 * (ch + 1), 512);
      }
      noise(rowb + ch, hr);
#pragma unroll
      for (int j = 0; j < 16; ++j) nzr[j] = hr[j];
      noise(colb + ch, hc);
#pragma unroll
      for (int j = 0; j < 16; ++j) nzc[j] = hc[j];
    } else {
      rnoise(rowb + ch, hr);
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
        a0 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * s) : zero4;
        a1 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * s + 16) : zero4;
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
      if constexpr (E::ON) {
        u32 P[8], N[8], wq[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) wrap_masks(bv[w], Uw[w], ww[w], P[w], N[w]);
        ep.wb_fault(u, lane, s, P);
#pragma unroll
        for (int w = 0; w < 8; ++w) wq[w] = wrap_s8(P[w], N[w]);
        *reinterpret_cast<uint4*>(ep.wbs + lane * XS) = make_uint4(wq[0], wq[1], wq[2], wq[3]);
        *reinterpret_cast<uint4*>(ep.wbs + lane * XS + 16) = make_uint4(wq[4], wq[5], wq[6], wq[7]);
      }
      if constexpr (V >= 2) {
        if (s + 1 < KS) {
          pa0 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * (s + 1)) : zero4;
          pa1 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * (s + 1) + 16) : zero4;
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
      u32 wf[4][2];
      if constexpr (E::ON) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          wf[ni][0] = *reinterpret_cast<const u32*>(ep.wbs + (8 * ni + g) * XS + 4 * q);
          wf[ni][1] = *reinterpret_cast<const u32*>(ep.wbs + (8 * ni + g) * XS + 16 + 4 * q);
        }
      }
      __syncwarp();
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          mma(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
          mma(Y[mi][ni], af[mi], IB[ni][0], IB[ni][1]);
          if constexpr (E::ON) mma(cr[mi][ni], af[mi], wf[ni][0], wf[ni][1]);
        }
      if constexpr (E::ON) ep.fault(u, lane, s, acc);
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
    if constexpr (E::ON) {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            acc[mi][ni][e] = d == 0 ? P[mi][ni][e] : (int)((u32)acc[mi][ni][e] - ((u32)P[mi][ni][e] << (7 * d)));
    }
  }
  if constexpr (E::ON) {
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mi][ni][e] = (int)((u32)acc[mi][ni][e] + ((u32)cr[mi][ni][e] << 8));
    ep.out(ri, cj, lane, acc);
  }
}

__device__ __forceinline__ void nbar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void nbar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
// the same with the id an immediate: a register id makes ptxas reserve all 16 of a block's barriers, which caps an SM at
// one such block (ncu's Block Limit Barriers)
template <int ID, bool ARRIVE>
__device__ __forceinline__ void nbar_c(int n) {
  if constexpr (ARRIVE)
    asm volatile("bar.arrive %0, %1;\n" ::"n"(ID), "r"(n) : "memory");
  else
    asm volatile("bar.sync %0, %1;\n" ::"n"(ID), "r"(n) : "memory");
}
// barrier BASE + i, i < 4 and warp-uniform; IMM: through immediates (a switch on i), else through a register id
template <bool IMM, int BASE, bool ARRIVE = false>
__device__ __forceinline__ void nbar4(int i, int n) {
  if constexpr (IMM) {
    switch (i) {
      case 0: nbar_c<BASE, ARRIVE>(n); break;
      case 1: nbar_c<BASE + 1, ARRIVE>(n); break;
      case 2: nbar_c<BASE + 2, ARRIVE>(n); break;
      default: nbar_c<BASE + 3, ARRIVE>(n); break;
    }
  } else if constexpr (ARRIVE) {
    nbar_arrive(BASE + i, n);
  } else {
    nbar_sync(BASE + i, n);
  }
}
template <bool IMM, int ID, bool ARRIVE = false>
__device__ __forceinline__ void nbar1(int n) {
  if constexpr (IMM)
    nbar_c<ID, ARRIVE>(n);
  else if constexpr (ARRIVE)
    nbar_arrive(ID, n);
  else
    nbar_sync(ID, n);
}

// The block form, for calls whose units leave most of the GPU idle (decode): one unit a block of BOW + 1 warps, the same
// values as `h1r_unit<2, E>` (so the same receipt, z and y).  The operand warps 0 .. BOW - 1 hash the noise a group of
// chunks at a time, one compression a thread (group 0: chunk 0, with U's and σ_R's blocks; then the chunks in pairs,
// double-buffered), and build segment s's X, W' and wb rows (peeled, and B's) in warp s % BOW's slot; each also runs its
// segments' residue and wrap MMAs (Y and X wbᵀ, or the peel's correction, read only after the last segment), summed over
// the operand warps through shared memory at the end.  The fold warp (BOW) runs T_s's MMAs, E's fault and the absorb in
// segment order, then the tail and E's output.
// A slot is handed over by named barriers: FULL after the operand warp's stores, EMPTY after the fold warp's loads.
// IMM: every barrier id an immediate (`nbar4`), 11 barriers, not 16, the same values.
constexpr int BOW = 4;
static_assert(BOW == 4, "nbar4 dispatches four slots");
constexpr int BTH = 32 * (BOW + 1);
enum : int { NB_FULL = 1, NB_EMPTY = NB_FULL + BOW, NB_OPS = NB_EMPTY + BOW, NB_RED = NB_OPS + 1 };
constexpr int RS = 33;  // the reduction's lane stride in words

struct __align__(16) H1BShared {
  uint8_t op[BOW][4][32 * XS];  // operand warp w's slot: X (row l at l XS), W' (column l), wb (column l), peeled B (column l)
  union {
    u32 nz[2][2][2][32 * NZ];  // [buffer][chunk of the group][row, column][l NZ + s16]
    int red[2][32 * RS];       // after the last segment: Y and X wbᵀ (peeled: the correction) summed over the operand warps
  };
  uint8_t us[32 * XS];  // U: column l's 32 residues at l XS
  uint8_t xs[32 * XS];  // the fold warp's digit rows
  u32 sts[32 * NZ];     // σ_R's block l at l NZ, lane l's fold state
  u64 sd[16];           // seed_j, σ_R
};

template <class E = NoEpi, bool IMM = false>
__device__ __forceinline__ void h1b_unit(const H1Call& C, int v, H1BShared& sh, const u64* __restrict__ seeds, u64 nt0,
                                         u64 nt1, u64 nt2, u32* __restrict__ out, int nowrite, const E& ep = E()) {
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const bool fold = wid == BOW;
  const int nrb = C.units / C.ncb, ri = v % nrb, cj = v / nrb, u = ri * C.ncb + cj;
  const int KS = C.k >> 5, m = n_digits(KS), sets = KS + m;
  u32* o = out + C.out0 + (long long)u * 2 * sets;
  if (!C.live[ri]) {
    if (fold) {
      if (!nowrite)
        for (int i = lane; i < 2 * sets; i += 32) o[i] = 0;
      if constexpr (E::ON) ep.dead(ri, cj, lane);
    }
    return;
  }
  const int g = lane >> 2, q = lane & 3;
  const int nb = KS / 16, RB = nb + 1;
  const u64 rowb = (u64)lane * RB, colb = (u64)32 * RB + (u64)lane * RB;
  const int8_t* arow = C.a + (32LL * ri + lane) * C.lda;
  const int8_t* brow = C.b + (32LL * cj + lane) * C.ldb;
  const bool aok = ep.a_ok(32 * ri + lane);
  if (wid == 0 && aok) prefetch_l2(arow, 512);
  if (wid == 1) prefetch_l2(brow, 512);
  if constexpr (E::SEEDS) {
    if (lane == 0 && wid < 2) ep.seed(C, u, wid, sh.sd + 8 * wid);
  } else if (tid < 16) {
    sh.sd[tid] = seeds[(long long)(C.seed0 + u) * 16 + tid];
  }
  __syncthreads();
  const u64* sj = sh.sd;
  u64 mid[8];
  noise_mid(sj, nt0, nt1, nt2, mid);
  auto noise = [&](u64 t, u32 h[16]) { nblock_mid(mid, sj, t, nt0, nt1, nt2, h); };
  // E::BAND 1 (LDG): the rows read from the band workspace in the segment loop, so a group past group 0 is GW = 4 chunks,
  // each operand warp hashing one column tile of it, chunk c0 + i's column tile in nz[buf][i >> 1][i & 1].  E::BAND 2:
  // the per-unit schedule, each row tile copied from the workspace where it would be hashed
  constexpr bool LDG = E::BAND == 1;
  constexpr int GW = LDG ? 4 : 2;
  const u32* rwp = E::BAND ? ep.band_rows(C, ri, cj, RB) : nullptr;

  // round 0, a compression a thread: chunk 0's row (warp 0) and column (warp 1) blocks, U (warp 2), σ_R's blocks (warp
  // 3), the digit coins (the fold warp)
  u32 zc[3] = {0, 0, 0};
  {
    u32 h[16];
    if (fold) {
      if constexpr (E::BAND) {
        band_wait();
#pragma unroll
        for (int d = 0; d < 3; ++d) zc[d] = __ldg(rwp + 16 * (lane * RB + nb) + d);
      } else {
        noise(rowb + nb, h);
#pragma unroll
        for (int d = 0; d < 3; ++d) zc[d] = h[d];
      }
    } else if (wid < 2) {
      if (E::BAND && wid == 0) {
        if constexpr (!LDG) {
          band_wait();
          uint4 v[4];
          band_load(rwp + 16 * lane * RB, v);
          band_store(v, sh.nz[0][0][0] + lane * NZ);
        }
      } else {
        noise(wid ? colb : rowb, h);
        u32* d = (LDG ? sh.nz[0][0][0] : sh.nz[0][0][wid]) + lane * NZ;
#pragma unroll
        for (int j = 0; j < 16; ++j) d[j] = h[j];
      }
    } else if (wid == 2) {
      noise(colb + nb, h);
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        u32 x = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const u32 by = (h[w] >> (8 * i)) & 0xFFu;
          x |= ((by + (by >= 129u ? 63u : 192u)) & 0xFFu) << (8 * i);  // -64 + by mod 129, mod 2^8
        }
        *reinterpret_cast<u32*>(sh.us + lane * XS + 4 * w) = x;
      }
    } else {
      nblock(sh.sd + 8, lane, nt0, nt1, nt2, h);
#pragma unroll
      for (int j = 0; j < 16; ++j) sh.sts[lane * NZ + j] = h[j];
    }
  }
  if constexpr (E::BAND) band_wait();
  __syncthreads();

  if (!fold) {
    const int w = wid;
    const uint4 zero4 = make_uint4(0, 0, 0, 0);
    u32 Uw[8], IB[4][2];
#pragma unroll
    for (int i = 0; i < 8; ++i) Uw[i] = *reinterpret_cast<const u32*>(sh.us + lane * XS + 4 * i);
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        u32 x = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) x |= (u32)(16 * hh + 4 * q + i == 8 * ni + g) << (8 * i);
        IB[ni][hh] = x;
      }
    int Y[2][4][4], cr[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e) Y[mi][ni][e] = cr[mi][ni][e] = 0;
    uint8_t* Xs = sh.op[w][0];
    uint8_t* Ws = sh.op[w][1];
    uint8_t* Bs = sh.op[w][2];
    uint8_t* Rs = sh.op[w][3];
    // group 0 is chunk 0, group gi >= 1 chunks GW (gi - 1) + 1 .. GW gi (< nb)
    const int ng = LDG ? 1 + (nb + 2) / 4 : 1 + nb / 2;
    for (int gi = 0; gi < ng; ++gi) {
      const int c0 = gi == 0 ? 0 : GW * (gi - 1) + 1, nch = gi == 0 ? 1 : min(GW, nb - c0), buf = gi & 1;
      if (gi > 0) {
        if constexpr (LDG) {
          if (w < nch) {
            u32 h[16];
            noise(colb + c0 + w, h);
            u32* d = sh.nz[buf][w >> 1][w & 1] + lane * NZ;
#pragma unroll
            for (int j = 0; j < 16; ++j) d[j] = h[j];
          }
        } else if ((w >> 1) < nch) {
          u32* d = sh.nz[buf][w >> 1][w & 1] + lane * NZ;
          if (E::BAND && !(w & 1)) {
            uint4 v[4];
            band_load(rwp + 16 * (lane * RB + c0 + (w >> 1)), v);
            band_store(v, d);
          } else {
            u32 h[16];
            noise(((w & 1) ? colb : rowb) + c0 + (w >> 1), h);
#pragma unroll
            for (int j = 0; j < 16; ++j) d[j] = h[j];
          }
        }
        nbar1<IMM, NB_OPS>(32 * BOW);
      }
      if (gi + 1 < ng) {
        if constexpr (LDG) {
          const int cn = GW * gi + 1 + w;
          if (cn < nb) {
            prefetch_l2(brow + 512 * cn, 512);
            if (aok) prefetch_l2(arow + 512 * cn, 512);
          }
        } else {
          const int cn = 2 * gi + 1 + (w >> 1);
          if (cn < nb) {
            if (w & 1)
              prefetch_l2(brow + 512 * cn, 512);
            else if (aok)
              prefetch_l2(arow + 512 * cn, 512);
          }
        }
      }
      const int s1 = 16 * (c0 + nch);
      int s = 16 * c0 + w;
      uint4 pa0 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * s) : zero4;
      uint4 pa1 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * s + 16) : zero4;
      uint4 pb0 = *reinterpret_cast<const uint4*>(brow + 32 * s);
      uint4 pb1 = *reinterpret_cast<const uint4*>(brow + 32 * s + 16);
      for (; s < s1; s += BOW) {
        const uint4 a0 = pa0, a1 = pa1, b0 = pb0, b1 = pb1;
        if (s + BOW < s1) {
          pa0 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * (s + BOW)) : zero4;
          pa1 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * (s + BOW) + 16) : zero4;
          pb0 = *reinterpret_cast<const uint4*>(brow + 32 * (s + BOW));
          pb1 = *reinterpret_cast<const uint4*>(brow + 32 * (s + BOW) + 16);
        }
        const int ci = (s >> 4) - c0, s16 = s & 15;
        const u32* ncol = LDG ? sh.nz[buf][ci >> 1][ci & 1] : sh.nz[buf][ci][1];
        const u32 nr = LDG ? __ldg(rwp + 16 * (lane * RB + (s >> 4)) + s16) : sh.nz[buf][ci][0][lane * NZ + s16];
        const u32 nc = ncol[lane * NZ + s16];
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const u32 bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        u32 xw[8], ww[8], wq[8], nb4[4], nc4[4];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          xw[i] = badd(av[i], spread4(nr >> (4 * i)));
          ww[i] = badd(badd(bv[i], Uw[i]), spread4(nc >> (4 * i)));
        }
        if constexpr (E::ON) {
          u32 P[8], N[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) wrap_masks(bv[i], Uw[i], ww[i], P[i], N[i]);
          ep.wb_fault(u, lane, s, P);
#pragma unroll
          for (int i = 0; i < 8; ++i) wq[i] = wrap_s8(P[i], N[i]);
        }
        if constexpr (E::PEEL) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            nb4[j] = LDG ? __ldg(rwp + 16 * ((g + 8 * j) * RB + (s >> 4)) + s16)
                         : sh.nz[buf][ci][0][(g + 8 * j) * NZ + s16];
            nc4[j] = ncol[(8 * j + g) * NZ + s16];
          }
        }
        if (s >= BOW) nbar4<IMM, NB_EMPTY>(w, 64);
        *reinterpret_cast<uint4*>(Xs + lane * XS) = make_uint4(xw[0], xw[1], xw[2], xw[3]);
        *reinterpret_cast<uint4*>(Xs + lane * XS + 16) = make_uint4(xw[4], xw[5], xw[6], xw[7]);
        *reinterpret_cast<uint4*>(Ws + lane * XS) = make_uint4(ww[0], ww[1], ww[2], ww[3]);
        *reinterpret_cast<uint4*>(Ws + lane * XS + 16) = make_uint4(ww[4], ww[5], ww[6], ww[7]);
        if constexpr (E::ON) {
          *reinterpret_cast<uint4*>(Bs + lane * XS) = make_uint4(wq[0], wq[1], wq[2], wq[3]);
          *reinterpret_cast<uint4*>(Bs + lane * XS + 16) = make_uint4(wq[4], wq[5], wq[6], wq[7]);
        }
        if constexpr (E::PEEL) {
          *reinterpret_cast<uint4*>(Rs + lane * XS) = b0;
          *reinterpret_cast<uint4*>(Rs + lane * XS + 16) = b1;
        }
        nbar4<IMM, NB_FULL, true>(w, 64);
        __syncwarp();
        u32 af[2][4], wf[4][2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          af[mi][0] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 4 * q);
          af[mi][1] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 4 * q);
          af[mi][2] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 16 + 4 * q);
          af[mi][3] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 16 + 4 * q);
        }
        if constexpr (E::ON) {
#pragma unroll
          for (int ni = 0; ni < 4; ++ni) {
            wf[ni][0] = *reinterpret_cast<const u32*>(Bs + (8 * ni + g) * XS + 4 * q);
            wf[ni][1] = *reinterpret_cast<const u32*>(Bs + (8 * ni + g) * XS + 16 + 4 * q);
          }
        }
        if constexpr (E::PEEL) {
          u32 bf[4][2];
#pragma unroll
          for (int ni = 0; ni < 4; ++ni) {
            bf[ni][0] = *reinterpret_cast<const u32*>(Rs + (8 * ni + g) * XS + 4 * q);
            bf[ni][1] = *reinterpret_cast<const u32*>(Rs + (8 * ni + g) * XS + 16 + 4 * q);
          }
          peel_mmas(Y, cr, af, wf, bf, IB, nb4, nc4, q, u, lane, s, ep);
        } else {
#pragma unroll
          for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
              mma(Y[mi][ni], af[mi], IB[ni][0], IB[ni][1]);
              if constexpr (E::ON) mma(cr[mi][ni], af[mi], wf[ni][0], wf[ni][1]);
            }
        }
      }
    }
    // the noise buffers are free once every operand warp is past its last segment
    nbar1<IMM, NB_OPS>(32 * BOW);
    for (int i = tid; i < 2 * 32 * RS; i += 32 * BOW) (&sh.red[0][0])[i] = 0;
    nbar1<IMM, NB_OPS>(32 * BOW);
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          atomicAdd(&sh.red[0][lane * RS + 16 * mi + 4 * ni + e], Y[mi][ni][e]);
          if constexpr (E::ON) atomicAdd(&sh.red[1][lane * RS + 16 * mi + 4 * ni + e], cr[mi][ni][e]);
        }
    nbar1<IMM, NB_RED, true>(BTH);
    return;
  }

  // the fold warp
  u32 st[16], UB[4][2];
#pragma unroll
  for (int j = 0; j < 16; ++j) st[j] = sh.sts[lane * NZ + j];
#pragma unroll
  for (int ni = 0; ni < 4; ++ni) {
    UB[ni][0] = *reinterpret_cast<const u32*>(sh.us + (8 * ni + g) * XS + 4 * q);
    UB[ni][1] = *reinterpret_cast<const u32*>(sh.us + (8 * ni + g) * XS + 16 + 4 * q);
  }
  int acc[2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mi][ni][e] = 0;
  for (int s0 = 0; s0 < KS; s0 += BOW) {
#pragma unroll
    for (int w = 0; w < BOW; ++w) {
      const int s = s0 + w;
      const uint8_t* Xs = sh.op[w][0];
      const uint8_t* Ws = sh.op[w][1];
      nbar4<IMM, NB_FULL>(w, 64);
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
      if (s + BOW < KS) nbar4<IMM, NB_EMPTY, true>(w, 64);
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) mma(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
      if constexpr (E::ON) ep.fault(u, lane, s, acc);
      absorb(st, acc, o, s, lane, nowrite);
    }
  }
  int Y[2][4][4], cr[2][4][4];
  nbar1<IMM, NB_RED>(BTH);
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        Y[mi][ni][e] = sh.red[0][lane * RS + 16 * mi + 4 * ni + e];
        cr[mi][ni][e] = E::ON ? sh.red[1][lane * RS + 16 * mi + 4 * ni + e] : 0;
      }

  // the tail, as h1r_unit's
  uint8_t* Xs = sh.xs;
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
    if constexpr (E::ON) {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            acc[mi][ni][e] = d == 0 ? P[mi][ni][e] : (int)((u32)acc[mi][ni][e] - ((u32)P[mi][ni][e] << (7 * d)));
    }
  }
  if constexpr (E::ON) {
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          acc[mi][ni][e] = (int)((u32)acc[mi][ni][e] + (E::PEEL ? (u32)cr[mi][ni][e] : (u32)cr[mi][ni][e] << 8));
    ep.out(ri, cj, lane, acc);
  }
}

// The split form: one unit a block of six warps, the same values as `h1r_unit<2, E>`.  Warps 2 and 3 hash chunk c's row
// and column noise chunks (one compression a thread) while chunk c - 1 is built, into nz[c & 1]; warps 0 and 1 build the
// segments s = b mod 2 of each chunk into slot s % 4 (X, W' and wb, from A, B and the noise; peeled, and B) and run their
// residue and wrap MMAs (Y and X wbᵀ or the peel's correction, summed through shared memory after the last segment);
// warps 4 and 5 are the fold's
// halves.  Set t's chains (st[4 (t % 2) + e] and st[8 + 4 (t % 2) + e]) read only fragment (t / 4, t % 4), so half F owns the column
// fragments ni = F and F + 2, their accumulators, chains and outputs, and XORs its share of each receipt word into
// shared memory; the tail's digits are exchanged through shared memory, a digit at a time.  Live units come first, a
// column tile's live row tiles on adjacent blocks.  Named barriers: a slot's FULL and EMPTY (a builder and both
// halves), a noise buffer's FULL and EMPTY (hashers and builders), the builders' own, the reduction, the halves'.
constexpr int SNW = 6;
constexpr int STH = 32 * SNW;
constexpr int SSETS = 16384 / 32 + 4;  // the receipt's sets at the largest K the form takes (16384)
enum : int { SN_FULL = 1, SN_EMPTY = 5, SN_NZF = 9, SN_NZE = 11, SN_BLD = 13, SN_RED = 14, SN_FOLD = 15 };

struct __align__(16) H1SShared {
  uint8_t op[4][4][32 * XS];  // slot s % 4: X (row l at l XS), W' (column l), wb (column l), peeled B (column l)
  union {
    u32 nz[2][2][32 * NZ];  // [chunk & 1][row, column][l NZ + s16]
    int red[2][32 * RS];    // after the last segment: Y and X wbᵀ (peeled: the correction) summed over the builders
  };
  uint8_t us[32 * XS];     // U: column l's 32 residues at l XS
  uint8_t xs[2][32 * XS];  // the tail's digit rows, digit d in xs[d & 1]
  u32 sts[32 * NZ];        // σ_R's block l at l NZ, lane l's fold state
  u32 rcp[2 * SSETS];      // the receipt words, each half's share XORed in
  u64 sd[16];              // seed_j, σ_R
};

// unit v's row and column tile: the live row tiles' units first, then the dead ones', a column tile's row tiles on
// adjacent v
__device__ __forceinline__ void live_first(const H1Call& C, int v, int& ri, int& cj) {
  const int nrb = C.units / C.ncb;
  int nl = 0;
  for (int r = 0; r < nrb; ++r) nl += C.live[r] != 0;
  const bool lv = v < nl * C.ncb;
  const int w = lv ? v : v - nl * C.ncb, nk = lv ? nl : nrb - nl;
  int k = w % nk;
  cj = w / nk;
  for (ri = 0; ri < nrb; ++ri)
    if ((C.live[ri] != 0) == lv && k-- == 0) break;
}

// half F's share of one set: its chains, and its share of the set's receipt words XORed into rcp
template <int F>
__device__ __forceinline__ void absorb_half(u32 st[8], const int p[2][2][4], u32* rcp, int s, int lane) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int t = F + 2 * j;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const u32 v = (u32)p[t / 4][(t % 4) >> 1][e] * FX1(4 * t + e);
      const u32 u = st[e] ^ v ^ KK(t);
      st[e] = u * KA(t) + st[4 + e];
      st[4 + e] = rotl(u, ROT(t)) + st[e];
    }
  }
  u32 wy = st[0] ^ st[1] ^ st[2] ^ st[3];
  u32 wz = st[4] ^ st[5] ^ st[6] ^ st[7];
  wy = __reduce_xor_sync(0xFFFFFFFFu, wy);
  wz = __reduce_xor_sync(0xFFFFFFFFu, wz);
  if (lane == 0) {
    atomicXor(rcp + 2 * s, wy);
    atomicXor(rcp + 2 * s + 1, wz);
  }
}

template <int F, class E>
__device__ __forceinline__ void h1s_fold(H1SShared& sh, int u, int ri, int cj, int lane, int KS, int m, int sets,
                                         const u32 zc[3], u32* __restrict__ o, int nowrite, const E& ep) {
  const int g = lane >> 2, q = lane & 3;
  u32 st[8], UB[2][2];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    st[e] = sh.sts[lane * NZ + 4 * F + e];
    st[4 + e] = sh.sts[lane * NZ + 8 + 4 * F + e];
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int ni = F + 2 * k;
    UB[k][0] = *reinterpret_cast<const u32*>(sh.us + (8 * ni + g) * XS + 4 * q);
    UB[k][1] = *reinterpret_cast<const u32*>(sh.us + (8 * ni + g) * XS + 16 + 4 * q);
  }
  int acc[2][2][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mi][k][e] = 0;
  for (int s = 0; s < KS; ++s) {
    const int sl = s & 3;
    const uint8_t* Xs = sh.op[sl][0];
    const uint8_t* Ws = sh.op[sl][1];
    nbar_sync(SN_FULL + sl, 96);
    u32 af[2][4], bf[2][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      af[mi][0] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 4 * q);
      af[mi][1] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 4 * q);
      af[mi][2] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 16 + 4 * q);
      af[mi][3] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 16 + 4 * q);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int ni = F + 2 * k;
      bf[k][0] = *reinterpret_cast<const u32*>(Ws + (8 * ni + g) * XS + 4 * q);
      bf[k][1] = *reinterpret_cast<const u32*>(Ws + (8 * ni + g) * XS + 16 + 4 * q);
    }
    if (s + 4 < KS) nbar_arrive(SN_EMPTY + sl, 96);
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int k = 0; k < 2; ++k) mma(acc[mi][k], af[mi], bf[k][0], bf[k][1]);
    if constexpr (E::ON && F == 0) ep.fault_word(u, lane, s, acc[0][0][0]);
    absorb_half<F>(st, acc, sh.rcp, s, lane);
  }
  int Y[2][2][4], cr[2][2][4];
  nbar_sync(SN_RED, 128);
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = lane * RS + 16 * mi + 4 * (F + 2 * k) + e;
        Y[mi][k][e] = sh.red[0][i];
        cr[mi][k][e] = E::ON ? sh.red[1][i] : 0;
      }
  u32 zr[4][3];
#pragma unroll
  for (int rs = 0; rs < 4; ++rs)
#pragma unroll
    for (int d = 0; d < 3; ++d) zr[rs][d] = __shfl_sync(0xFFFFFFFFu, zc[d], g + 8 * rs);
  for (int d = 0; d < m; ++d) {
    uint8_t* Xs = sh.xs[d & 1];
    u32 zd4[4];
#pragma unroll
    for (int rs = 0; rs < 4; ++rs) zd4[rs] = d == 0 ? zr[rs][0] : d == 1 ? zr[rs][1] : zr[rs][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int row = 16 * mi + g + 8 * (e >> 1), rho = 8 * (F + 2 * k) + 2 * q + (e & 1);
          const int R = Y[mi][k][e];
          int dig;
          if (d < m - 1) {
            const int qq = R >> 7;
            u32 zd = (zd4[2 * mi + (e >> 1)] >> rho) & 1u;
            if (d == m - 2 && qq == 127) zd = 0;
            dig = (R & 127) | (int)(zd << 7);
            Y[mi][k][e] = qq + (int)zd;
          } else {
            dig = R & 0xFF;
          }
          Xs[row * XS + rho] = (uint8_t)dig;
        }
    nbar_sync(SN_FOLD, 64);
    u32 af[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      af[mi][0] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 4 * q);
      af[mi][1] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 4 * q);
      af[mi][2] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 16 + 4 * q);
      af[mi][3] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 16 + 4 * q);
    }
    int P[2][2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int k = 0; k < 2; ++k) {
#pragma unroll
        for (int e = 0; e < 4; ++e) P[mi][k][e] = 0;
        mma(P[mi][k], af[mi], UB[k][0], UB[k][1]);
      }
    if (d == 0) {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
          for (int e = 0; e < 4; ++e) P[mi][k][e] = (int)((u32)acc[mi][k][e] - (u32)P[mi][k][e]);
    }
    absorb_half<F>(st, P, sh.rcp, KS + d, lane);
    if constexpr (E::ON) {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            acc[mi][k][e] = d == 0 ? P[mi][k][e] : (int)((u32)acc[mi][k][e] - ((u32)P[mi][k][e] << (7 * d)));
    }
  }
  if constexpr (E::ON) {
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          acc[mi][k][e] = (int)((u32)acc[mi][k][e] + (E::PEEL ? (u32)cr[mi][k][e] : (u32)cr[mi][k][e] << 8));
    ep.template out_half<F>(ri, cj, lane, acc);
  }
  nbar_sync(SN_FOLD, 64);
  if (!nowrite)
    for (int i = lane + 32 * F; i < 2 * sets; i += 64) o[i] = sh.rcp[i];
}

template <class E = NoEpi>
__device__ __forceinline__ void h1s_unit(const H1Call& C, int v, H1SShared& sh, const u64* __restrict__ seeds, u64 nt0,
                                         u64 nt1, u64 nt2, u32* __restrict__ out, int nowrite, const E& ep = E()) {
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  int ri, cj;
  live_first(C, v, ri, cj);
  const int u = ri * C.ncb + cj;
  const int KS = C.k >> 5, m = n_digits(KS), sets = KS + m;
  u32* o = out + C.out0 + (long long)u * 2 * sets;
  if (!C.live[ri]) {
    if (wid == 4) {
      if (!nowrite)
        for (int i = lane; i < 2 * sets; i += 32) o[i] = 0;
      if constexpr (E::ON) ep.dead(ri, cj, lane);
    }
    return;
  }
  const int g = lane >> 2, q = lane & 3;
  const int nb = KS / 16, RB = nb + 1;
  const u64 rowb = (u64)lane * RB, colb = (u64)32 * RB + (u64)lane * RB;
  const int8_t* arow = C.a + (32LL * ri + lane) * C.lda;
  const int8_t* brow = C.b + (32LL * cj + lane) * C.ldb;
  const bool aok = ep.a_ok(32 * ri + lane);
  if (wid == 0 && aok) prefetch_l2(arow, 512);
  if (wid == 1) prefetch_l2(brow, 512);
  if constexpr (E::SEEDS) {
    if (lane == 0 && wid < 2) ep.seed(C, u, wid, sh.sd + 8 * wid);
  } else if (tid < 16) {
    sh.sd[tid] = seeds[(long long)(C.seed0 + u) * 16 + tid];
  }
  for (int i = tid; i < 2 * sets; i += STH) sh.rcp[i] = 0;
  __syncthreads();
  const u64* sj = sh.sd;
  u64 mid[8];
  noise_mid(sj, nt0, nt1, nt2, mid);
  auto noise = [&](u64 t, u32 h[16]) { nblock_mid(mid, sj, t, nt0, nt1, nt2, h); };
  // E::BAND: the two hashers take alternate chunks (chunk c's by hasher c & 1), each hashing its chunk's column tile and
  // copying its row tile from the band workspace into nz[c & 1], so each noise buffer's barriers are one hasher's and
  // the builders' (96 threads)
  const u32* rwp = E::BAND ? ep.band_rows(C, ri, cj, RB) : nullptr;
  constexpr int NZT = E::BAND ? 96 : 128;

  // round 0, a compression a thread: U (warp 0), σ_R's blocks (warp 1), chunk 0's row and column tiles (warps 2, 3), the
  // digit coins (each half)
  u32 zc[3] = {0, 0, 0};
  {
    u32 h[16];
    if (wid >= 4) {
      if constexpr (E::BAND) {
        band_wait();
#pragma unroll
        for (int d = 0; d < 3; ++d) zc[d] = __ldg(rwp + 16 * (lane * RB + nb) + d);
      } else {
        noise(rowb + nb, h);
#pragma unroll
        for (int d = 0; d < 3; ++d) zc[d] = h[d];
      }
    } else if (wid >= 2) {
      u32* d = sh.nz[0][wid - 2] + lane * NZ;
      if (E::BAND && wid == 2) {
        band_wait();
        uint4 v[4];
        band_load(rwp + 16 * lane * RB, v);
        band_store(v, d);
      } else {
        noise(wid == 3 ? colb : rowb, h);
#pragma unroll
        for (int j = 0; j < 16; ++j) d[j] = h[j];
      }
    } else if (wid == 0) {
      noise(colb + nb, h);
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        u32 x = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const u32 by = (h[w] >> (8 * i)) & 0xFFu;
          x |= ((by + (by >= 129u ? 63u : 192u)) & 0xFFu) << (8 * i);  // -64 + by mod 129, mod 2^8
        }
        *reinterpret_cast<u32*>(sh.us + lane * XS + 4 * w) = x;
      }
    } else {
      nblock(sh.sd + 8, lane, nt0, nt1, nt2, h);
#pragma unroll
      for (int j = 0; j < 16; ++j) sh.sts[lane * NZ + j] = h[j];
    }
  }
  if constexpr (E::BAND) band_wait();
  __syncthreads();

  if (wid == 4) return h1s_fold<0>(sh, u, ri, cj, lane, KS, m, sets, zc, o, nowrite, ep);
  if (wid == 5) return h1s_fold<1>(sh, u, ri, cj, lane, KS, m, sets, zc, o, nowrite, ep);

  if (wid >= 2) {  // the hashers: chunk c's blocks computed, then stored once the builders are past chunk c - 2
    const int side = wid - 2;
    if constexpr (E::BAND) {
      for (int c = side ? 1 : 2; c < nb; c += 2) {
        u32 h[16];
        uint4 v[4];
        band_load(rwp + 16 * (lane * RB + c), v);
        noise(colb + c, h);
        if (c >= 2) nbar_sync(SN_NZE + (c & 1), NZT);
        band_store(v, sh.nz[c & 1][0] + lane * NZ);
        u32* d = sh.nz[c & 1][1] + lane * NZ;
#pragma unroll
        for (int j = 0; j < 16; ++j) d[j] = h[j];
        nbar_arrive(SN_NZF + (c & 1), NZT);
      }
      return;
    }
    for (int c = 1; c < nb; ++c) {
      u32 h[16];
      noise((side ? colb : rowb) + c, h);
      if (c >= 2) nbar_sync(SN_NZE + (c & 1), 128);
      u32* d = sh.nz[c & 1][side] + lane * NZ;
#pragma unroll
      for (int j = 0; j < 16; ++j) d[j] = h[j];
      nbar_arrive(SN_NZF + (c & 1), 128);
    }
    return;
  }

  // the builders
  const int b = wid;
  const uint4 zero4 = make_uint4(0, 0, 0, 0);
  u32 Uw[8], IB[4][2];
#pragma unroll
  for (int i = 0; i < 8; ++i) Uw[i] = *reinterpret_cast<const u32*>(sh.us + lane * XS + 4 * i);
#pragma unroll
  for (int ni = 0; ni < 4; ++ni)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      u32 x = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) x |= (u32)(16 * hh + 4 * q + i == 8 * ni + g) << (8 * i);
      IB[ni][hh] = x;
    }
  int Y[2][4][4], cr[2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) Y[mi][ni][e] = cr[mi][ni][e] = 0;
  uint4 pa0 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * b) : zero4;
  uint4 pa1 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * b + 16) : zero4;
  uint4 pb0 = *reinterpret_cast<const uint4*>(brow + 32 * b);
  uint4 pb1 = *reinterpret_cast<const uint4*>(brow + 32 * b + 16);
  for (int s = b; s < KS; s += 2) {
    const int c = s >> 4, s16 = s & 15, sl = s & 3;
    if (s16 == b) {
      if (c > 0) nbar_sync(SN_NZF + (c & 1), NZT);
      if (c + 1 < nb) {
        if (b)
          prefetch_l2(brow + 512 * (c + 1), 512);
        else if (aok)
          prefetch_l2(arow + 512 * (c + 1), 512);
      }
    }
    const uint4 a0 = pa0, a1 = pa1, b0 = pb0, b1 = pb1;
    if (s + 2 < KS) {
      pa0 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * (s + 2)) : zero4;
      pa1 = aok ? *reinterpret_cast<const uint4*>(arow + 32 * (s + 2) + 16) : zero4;
      pb0 = *reinterpret_cast<const uint4*>(brow + 32 * (s + 2));
      pb1 = *reinterpret_cast<const uint4*>(brow + 32 * (s + 2) + 16);
    }
    const u32 nr = sh.nz[c & 1][0][lane * NZ + s16];
    const u32 nc = sh.nz[c & 1][1][lane * NZ + s16];
    u32 nb4[4], nc4[4];
    if constexpr (E::PEEL) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        nb4[j] = sh.nz[c & 1][0][(g + 8 * j) * NZ + s16];
        nc4[j] = sh.nz[c & 1][1][(8 * j + g) * NZ + s16];
      }
    }
    if (s16 == 14 + b && c + 2 < nb) nbar_arrive(SN_NZE + (c & 1), NZT);
    const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const u32 bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    u32 xw[8], ww[8], wq[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      xw[i] = badd(av[i], spread4(nr >> (4 * i)));
      ww[i] = badd(badd(bv[i], Uw[i]), spread4(nc >> (4 * i)));
    }
    if constexpr (E::ON) {
      u32 P[8], N[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) wrap_masks(bv[i], Uw[i], ww[i], P[i], N[i]);
      ep.wb_fault(u, lane, s, P);
#pragma unroll
      for (int i = 0; i < 8; ++i) wq[i] = wrap_s8(P[i], N[i]);
    }
    uint8_t* Xs = sh.op[sl][0];
    uint8_t* Ws = sh.op[sl][1];
    uint8_t* Bs = sh.op[sl][2];
    uint8_t* Rs = sh.op[sl][3];
    if (s >= 4) nbar_sync(SN_EMPTY + sl, 96);
    *reinterpret_cast<uint4*>(Xs + lane * XS) = make_uint4(xw[0], xw[1], xw[2], xw[3]);
    *reinterpret_cast<uint4*>(Xs + lane * XS + 16) = make_uint4(xw[4], xw[5], xw[6], xw[7]);
    *reinterpret_cast<uint4*>(Ws + lane * XS) = make_uint4(ww[0], ww[1], ww[2], ww[3]);
    *reinterpret_cast<uint4*>(Ws + lane * XS + 16) = make_uint4(ww[4], ww[5], ww[6], ww[7]);
    if constexpr (E::ON) {
      *reinterpret_cast<uint4*>(Bs + lane * XS) = make_uint4(wq[0], wq[1], wq[2], wq[3]);
      *reinterpret_cast<uint4*>(Bs + lane * XS + 16) = make_uint4(wq[4], wq[5], wq[6], wq[7]);
    }
    if constexpr (E::PEEL) {
      *reinterpret_cast<uint4*>(Rs + lane * XS) = b0;
      *reinterpret_cast<uint4*>(Rs + lane * XS + 16) = b1;
    }
    nbar_arrive(SN_FULL + sl, 96);
    __syncwarp();
    u32 af[2][4], wf[4][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      af[mi][0] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 4 * q);
      af[mi][1] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 4 * q);
      af[mi][2] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g) * XS + 16 + 4 * q);
      af[mi][3] = *reinterpret_cast<const u32*>(Xs + (16 * mi + g + 8) * XS + 16 + 4 * q);
    }
    if constexpr (E::ON) {
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        wf[ni][0] = *reinterpret_cast<const u32*>(Bs + (8 * ni + g) * XS + 4 * q);
        wf[ni][1] = *reinterpret_cast<const u32*>(Bs + (8 * ni + g) * XS + 16 + 4 * q);
      }
    }
    if constexpr (E::PEEL) {
      u32 bf[4][2];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        bf[ni][0] = *reinterpret_cast<const u32*>(Rs + (8 * ni + g) * XS + 4 * q);
        bf[ni][1] = *reinterpret_cast<const u32*>(Rs + (8 * ni + g) * XS + 16 + 4 * q);
      }
      peel_mmas(Y, cr, af, wf, bf, IB, nb4, nc4, q, u, lane, s, ep);
    } else {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          mma(Y[mi][ni], af[mi], IB[ni][0], IB[ni][1]);
          if constexpr (E::ON) mma(cr[mi][ni], af[mi], wf[ni][0], wf[ni][1]);
        }
    }
  }
  // the noise buffers become the reduction once both builders are past their last segment
  nbar_sync(SN_BLD, 64);
  for (int i = tid; i < 2 * 32 * RS; i += 64) (&sh.red[0][0])[i] = 0;
  nbar_sync(SN_BLD, 64);
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        atomicAdd(&sh.red[0][lane * RS + 16 * mi + 4 * ni + e], Y[mi][ni][e]);
        if constexpr (E::ON) atomicAdd(&sh.red[1][lane * RS + 16 * mi + 4 * ni + e], cr[mi][ni][e]);
      }
  nbar_arrive(SN_RED, 128);
}

template <int NW, int V>
__global__ void __launch_bounds__(32 * NW, V >= 2 ? 3 : 1) h1r_units_k(const H1Call* __restrict__ calls, int ncalls,
                                                                       const u64* __restrict__ seeds, u64 nt0, u64 nt1,
                                                                       u64 nt2, u32* __restrict__ out, int nowrite,
                                                                       int band) {
  __shared__ __align__(16) uint8_t smem[NW][2][32 * XS];
  __shared__ u64 sd[NW][24];
  __shared__ u32 nzs[V >= 2 ? NW : 1][2][32 * NZ];
  const int c = blockIdx.y;
  if (c >= ncalls) return;
  const H1Call C = calls[c];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int v = blockIdx.x * NW + wid;
  if (v >= C.units) return;
  h1r_unit<V>(C, v, lane, smem[wid][0], smem[wid][1], sd[wid], nzs[V >= 2 ? wid : 0][0], seeds, nt0, nt1, nt2, out, nowrite,
              NoEpi(), band);
}

// h1r_units_k's version 1 under J10-R at call layout jrm (h1r_launch_j10r)
template <int NW>
__global__ void __launch_bounds__(32 * NW, 1) h1r_units_jr_k(const H1Call* __restrict__ calls, int ncalls,
                                                             const u64* __restrict__ seeds, u64 nt0, u64 nt1, u64 nt2,
                                                             u32* __restrict__ out, int nowrite, int jrm) {
  __shared__ __align__(16) uint8_t smem[NW][2][32 * XS];
  __shared__ u64 sd[NW][24];
  __shared__ u32 nzs[1][2][32 * NZ];
  const int c = blockIdx.y;
  if (c >= ncalls) return;
  const H1Call C = calls[c];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int v = blockIdx.x * NW + wid;
  if (v >= C.units) return;
  h1r_unit<1>(C, v, lane, smem[wid][0], smem[wid][1], sd[wid], nzs[0][0], seeds, nt0, nt1, nt2, out, nowrite, NoEpi(), 1,
              jrm);
}

__global__ void __launch_bounds__(BTH, 2) h1r_unitsb_k(const H1Call* __restrict__ calls, int ncalls,
                                                       const u64* __restrict__ seeds, u64 nt0, u64 nt1, u64 nt2,
                                                       u32* __restrict__ out, int nowrite) {
  __shared__ H1BShared sh;
  const int c = blockIdx.y;
  if (c >= ncalls) return;
  const H1Call C = calls[c];
  if ((int)blockIdx.x >= C.units) return;
  h1b_unit(C, blockIdx.x, sh, seeds, nt0, nt1, nt2, out, nowrite);
}

__global__ void __launch_bounds__(STH, 2) h1r_unitss_k(const H1Call* __restrict__ calls, int ncalls,
                                                       const u64* __restrict__ seeds, u64 nt0, u64 nt1, u64 nt2,
                                                       u32* __restrict__ out, int nowrite) {
  __shared__ H1SShared sh;
  const int c = blockIdx.y;
  if (c >= ncalls) return;
  const H1Call C = calls[c];
  if ((int)blockIdx.x >= C.units) return;
  h1s_unit(C, blockIdx.x, sh, seeds, nt0, nt1, nt2, out, nowrite);
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
    h1r_units_k<NW, 1><<<grid, 32 * NW, 0, stream>>>(calls_dev, ncalls, seeds_dev, nt[0], nt[1], nt[2], out_dev, nowrite, 1);
  else if (version == 3)  // the block form
    h1r_unitsb_k<<<dim3(max_units, ncalls), BTH, 0, stream>>>(calls_dev, ncalls, seeds_dev, nt[0], nt[1], nt[2], out_dev,
                                                              nowrite);
  else if (version == 4) {  // the split form
    for (int i = 0; i < ncalls; ++i)
      if (ks[i] > 16384) return 1;
    h1r_unitss_k<<<dim3(max_units, ncalls), STH, 0, stream>>>(calls_dev, ncalls, seeds_dev, nt[0], nt[1], nt[2], out_dev,
                                                              nowrite);
  }
  else
    h1r_units_k<NW, 2><<<grid, 32 * NW, 0, stream>>>(calls_dev, ncalls, seeds_dev, nt[0], nt[1], nt[2], out_dev, nowrite, 1);
  return (int)cudaGetLastError();
}

// h1r_launch's version 1 on J10's banded noise (the banded gates' receipts reference; measured only): unit (ri, cj)'s rows
// from its band leader (ri, (cj / band) band)'s seed_j, its columns and keys its own
extern "C" int h1r_launch_band(const H1Call* calls_dev, int ncalls, int max_units, const int* ks, const u64* salt_dev,
                               const u64* step_base, u64* seeds_dev, u32* out_dev, int nowrite, int band,
                               cudaStream_t stream) {
  if (band < 1) return 1;
  const int rc = h1r_launch(calls_dev, ncalls, max_units, ks, salt_dev, step_base, seeds_dev, out_dev, nowrite, 0, stream);
  if (rc) return rc;
  u64 nt[3];
  tag3("verity/pouw/noise/v1", nt);
  h1r_units_k<NW, 1><<<dim3((max_units + NW - 1) / NW, ncalls), 32 * NW, 0, stream>>>(calls_dev, ncalls, seeds_dev, nt[0],
                                                                                    nt[1], nt[2], out_dev, nowrite, band);
  return (int)cudaGetLastError();
}

// h1r_launch's version 1 under J10-R (J10R's PouwH1Tile_v5 receipt; the served J10-R units' receipts reference, measured
// only): seed_c at each call's first position (h1r_seeds_jr), every unit's noise from its call's stream at call scale
// Layout(m, N, K) (h1r_units_jr_k), its fold keys its own σ_R's; m a multiple of 32, the call's layout rows (its slot T,
// or its live rows rounded up to 32), no call's live row tile past it
extern "C" int h1r_launch_j10r(const H1Call* calls_dev, int ncalls, int max_units, const int* ks, const u64* salt_dev,
                               const u64* step_base, u64* seeds_dev, u32* out_dev, int nowrite, int m, cudaStream_t stream) {
  for (int i = 0; i < ncalls; ++i)
    if (ks[i] % 512 || ks[i] < 512) return 1;
  if (m < 32 || m % 32) return 1;
  u64 ct[3], rt[3], nt[3];
  tag3("verity/pouw/call-seed/v1", ct);
  tag3("verity/pouw/receipt/v1", rt);
  tag3("verity/pouw/noise/v1", nt);
  h1r_seeds_jr<<<dim3((2 * max_units + 127) / 128, ncalls), 128, 0, stream>>>(calls_dev, ncalls, salt_dev, step_base, ct[0],
                                                                             ct[1], ct[2], rt[0], rt[1], rt[2], seeds_dev);
  h1r_units_jr_k<NW><<<dim3((max_units + NW - 1) / NW, ncalls), 32 * NW, 0, stream>>>(calls_dev, ncalls, seeds_dev, nt[0],
                                                                                    nt[1], nt[2], out_dev, nowrite, m);
  return (int)cudaGetLastError();
}

extern "C" int h1r_call_bytes() { return (int)sizeof(H1Call); }
extern "C" int h1r_n_digits(int ks) { return n_digits(ks); }
