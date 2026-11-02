// pfill: a DRAM-free filler in Theorem E's own setting. dcap.cu includes this file in place of FILL
// v1's chained filler unless DFILL_V1 is defined, under the same ABI (dfill_k<NC>, DFill, dfill_launch, dfill_prep,
// dfill_make, dfill_check, dfill_ref), so dcap's d* modes and dscore.cu run it unchanged. The scorer's fill-v1 arm (and
// fused_arm.py's chained arms) pass -DDFILL_V1 through NVCC_APPEND_FLAGS to build v1's chained filler instead.
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
//   4  ram  y <- (rotl(y, 7) + d0) a0 + d1;  y <- (rotl(y, 19) + d2) a1 + d3   (two ops a pair). Not a receipt of
//           both words: a step is rotl(y, 7) a0 + Q, so it reads a pair only through Q = a0 d0 + d1, which one product a
//           pair computes (pfill_ref.q_receipt). Built with -DPFILL_LI
//           or -DPFILL_XIN13, NC 4 is one of the r9 folds instead, on the
//           device and in dfill_ref alike, each committing the whole y:
//      LI     y <- (y ^ d0 ^ k0) a0 + d1;  y <- (y ^ d2 ^ k1) a1 + d3   (LOP3, IMAD: two ops a pair). Sound only while no
//             free operand is shared across a hand-off (d1, d2) or (d3, the next d0): the filler's rows are hashed
//      XIN13  y <- (rotl(y, 13) ^ d0) a0 + d1;  y <- (rotl(y, 19) ^ d2) a1 + d3   (SHF, LOP3, IMAD: three ops a pair)
//           Each has an all-state kernel of fixed word bijections, complement flips that spare no product
//           (tests/test_pfill_fold.py); duo has none
//   2  duo  per word p, keys (k, a, r): u = y ^ p ^ k;  y <- u a + z;  z <- rotl(u, r) + y   (three ops a word)
//   1  xor  y <- y ^ d0 ^ d1 ^ d2 ^ d3, the cheapest fold that reads every word: a cost reference, not a receipt (it is
//           linear, so it binds nothing)
//   5, 3    ram, xor with A made once a tile and reused (a probe of the SHA's cost; its receipts are not dfill_ref's)
//   6, 7, 8 probes of the fold's pipes, A made once: ram with immediate keys, LEA.HI only, IMAD only (none a receipt)
//   9  arx  y <- rotl(y ^ d0, 13) + d1;  y <- rotl(y ^ d2, 19) + d3 (LOP3, LEA.HI a word, no IMAD), A by SHA as a receipt
//           fold's is: the cost of an IMAD-free fold, not a receipt (whether such a fold binds every word is FOLD-OPT's)
// Keys are the key block's row words in order, the first two that are odd, at least 2^8 + 1 away from 0 mod 2^24 and not
// ±1 mod 32 (a0, a1), and in an LI or XIN13 build (every fold's keys there) with no a b within 128 of 0 mod 2^30 for
// b in 1..128 (pf_key_ok_r9); k0, k1 (duo's and LI's) are its rows 14 and 15. A tile's receipt is the XOR over its
// chains of smix(smix(K ^ pos) ^ v), K = dseed(c0, c1, tile), pos = (32 w + lane) 4 + c, v the chain's state (duo:
// z << 32 | y).
//
// Built with -DPFILL_E2M1 (score.py's build takes no flags, so an arm passes it through NVCC_APPEND_FLAGS, which nvcc
// reads), dfill_k and dfill_ref run the e2m1 tile instead: K = L = 64 on OMMA m16n8k64 (kind::mxf4nvf4, every ue4m3
// block scale 1), the same noise as e2m1 codes (+1 = 0x2, -1 = 0xA) in the domain PF_DOM4, each fragment and B block two
// SHA blocks (ctr 2 F + h and 2^56 + 2 b + h for k in 32 h .. 32 h + 31), and each word the bits of the exact f32 sum
// 64 - 2 popcount(a ^ b). The chains, keys and fold are the s8 tile's. An OMMA holds the tensor pipe as long as an IMMA
// and does twice the products, each half an int8 slot, so DFill::macs() counts int8 slots in both kinds (the e2m1 tile's
// products / 2). One build holds one kind: with both in dfill_k, the s8 tile kept 0.96 of its alone rate beside decode
// instead of 0.99.
//
// Built with -DPFILL_BITS instead, the tile is Theorem K's: every A and B
// entry is its bit itself, the s8 byte 0 or 1, in the domain PF_DOM01, on the same s8 IMMA m16n8k32, so each word is
// the s32 sum popcount(a & b) in [0, 32], read unscaled into the same receipt.
//
// Built with -DPFILL_BU instead, the tile is pfill-bU, in Theorem K's
// setting: the same s8 IMMA m16n8k32 with C = 0 and the same ram receipt, in the domain PF_DOM_BU. A's entries are their
// bits, the s8 byte 0 or 1; B's are U in {±1, ..., ±64}, one digest byte t an entry, U = (t & 0x7F) - 64 + ((t >> 6) & 1)
// = -64 + sum_{p<6} 2^p t_p + 65 t_6 (bit 7 unused), so each word is the s32 sum of a U over the row's set bits, in
// [-2048, 2048], read unscaled. B's block b (columns 2b, 2b + 1) is ctr = 2^56 + b, entry 32 i + k its (k, column 2b + i)
// from digest byte 32 i + k: 64 W blocks a W-warp tile, two for each lane, and thread 0 makes the keys after them and
// hands them to the block through shared memory. dcap's dgate also runs pf_bu_uniform, the histograms of U and of A on
// the host generators that dfill_ref (and so dgate's receipts) holds the GPU to.
#pragma once
#include "../pearl_c/sha512.cuh"
#include "../receipts/receipt_folds.cuh"

#define PF_W 8
#ifndef PF_CTAS
#define PF_CTAS 2  // blocks an SM: 2 x 48 KB, 4 warps an SMSP
#endif
#ifndef PF_MINB
#define PF_MINB PF_CTAS  // the pump's launch-bounds blocks an SM, its register cap; above PF_CTAS, room beside it
#endif
#ifndef PF_SHA_ROLL
#define PF_SHA_ROLL 1  // pf_compress (16 rounds a loop iteration); 0: rs_compress (all 80 unrolled); same digests
#endif
#define PF_TN 1024
#ifndef PF_PW
#define PF_PW 8  // the phase filler's warps a block (dfill_p): its tile is PF_PW warps x 128 PF_PW columns
#endif
#ifndef PF_PMIN
#define PF_PMIN (PF_MINB * PF_W / PF_PW)  // dfill_p's launch-bounds blocks an SM: its register cap, 128 by default
#endif
static_assert(PF_PW == 2 || PF_PW == 4 || PF_PW == 8, "PF_PW is 2, 4 or 8");
#define DFILL_GATE_W PF_W
#define DFILL_GATE_S {32, 512}
constexpr uint64_t PF_DOM = 0x5046494C4C2D7631ull;   // "PFILL-v1": the s8 tile
constexpr uint64_t PF_DOM4 = 0x5046494C4C2D6634ull;  // "PFILL-f4": the e2m1 tile
constexpr uint64_t PF_DOM01 = 0x5046494C4C2D6231ull; // "PFILL-b1": the s8 tile on {0,1} bits
constexpr uint64_t PF_DOM_BU = 0x5046494C4C2D6255ull;  // "PFILL-bU": the bU tile
constexpr uint64_t PF_B = 1ull << 56, PF_KEY = 2ull << 56;
static_assert(PF_TN == 16 * 8 * PF_W, "lanes 0..7 (e2m1: 0..15) of every warp make B's blocks");
enum { PF_S8 = 0, PF_E2M1 = 1, PF_BITS = 2, PF_BU = 3 };  // the tile kinds, as pfill_ref.py numbers them
#if defined(PFILL_E2M1) + defined(PFILL_BITS) + defined(PFILL_BU) > 1
#error "pfill: one tile kind a build, -DPFILL_E2M1, -DPFILL_BITS or -DPFILL_BU"
#elif defined(PFILL_E2M1)
constexpr int PF_KIND = PF_E2M1;
#elif defined(PFILL_BITS)
constexpr int PF_KIND = PF_BITS;
#elif defined(PFILL_BU)
constexpr int PF_KIND = PF_BU;
#else
constexpr int PF_KIND = PF_S8;
#endif
#if defined(PFILL_XIN)
#error "pfill: -DPFILL_XIN (XOR-in, rotl 7) is retired: ±128 is an int8 ratio at its step boundary (note:20261016T1117Z-draft-fullfloor-s-r9). Build -DPFILL_XIN13"
#endif
#if defined(PFILL_LI) && defined(PFILL_XIN13)
#error "pfill: one receipt fold a build, -DPFILL_LI or -DPFILL_XIN13"
#endif
#if defined(PFILL_LI) || defined(PFILL_XIN13)
constexpr bool PF_R9 = true;
#else
constexpr bool PF_R9 = false;
#endif
// The fold's cost dial: PF_ROUNDS is the number of fold rounds a committed word in the
// NC 4 (XOR-in / LI / ram) absorb. The default 1 is byte-identical to the fold of record; a build passes -DPF_ROUNDS=R
// through NVCC_APPEND_FLAGS to measure the cost a word a round beside the saturated noise IMMA.
#ifndef PF_ROUNDS
#define PF_ROUNDS 1
#endif
static_assert(PF_ROUNDS >= 1 && PF_ROUNDS <= 64, "PF_ROUNDS is 1..64");
// PF_YIELD: a pump warp yields its SM while a serving CTA marks it (pf_busy[smid] > 0: flin's consumer warps, from
// griddepcontrol.wait to their flush, or with PF_YIELD_EARLY from the CTA's start), read every PF_YIELD_J B fragments
// one read ahead. 1 waits until the mark clears (a poll every PF_YIELD_POLL_NS), 2 sleeps PF_YIELD_NS a read while it
// holds. What a tile computes is unchanged; only when.
#ifndef PF_YIELD
#define PF_YIELD 0
#endif
#ifndef PF_YIELD_EARLY
#define PF_YIELD_EARLY 0
#endif
#ifndef PF_YIELD_J
#define PF_YIELD_J 16
#endif
#ifndef PF_YIELD_NS
#define PF_YIELD_NS 256
#endif
#ifndef PF_YIELD_POLL_NS
#define PF_YIELD_POLL_NS 1000
#endif
static_assert(PF_YIELD >= 0 && PF_YIELD <= 2 && PF_YIELD_J % 4 == 0, "PF_YIELD is 0, 1 or 2; PF_YIELD_J a multiple of 4");
__host__ __device__ constexpr uint64_t pf_dom(int kd) {
  return kd == PF_E2M1 ? PF_DOM4 : kd == PF_BITS ? PF_DOM01 : kd == PF_BU ? PF_DOM_BU : PF_DOM;
}

__host__ __device__ inline uint32_t pf_bswap(uint32_t x) {
  return (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
}
__host__ __device__ inline uint32_t pf_rotl(uint32_t x, int r) { return rf_rotl(x, r); }
// row word r of a digest: digest bytes 4r..4r+3 as a little-endian u32, so bit k is entry 32 r + k's bit
__host__ __device__ inline uint32_t pf_row(const uint64_t* st, int r) {
  return pf_bswap((r & 1) ? (uint32_t)st[r >> 1] : (uint32_t)(st[r >> 1] >> 32));
}
// four entries (bits of nib, entry q in bit q) as s8 bytes 1 - 2 bit, entry q in byte q
__host__ __device__ inline uint32_t pf_pm1(uint32_t nib) { return 0x01010101u | (((nib * 0x00204081u) & 0x01010101u) * 0xFEu); }
// the same four entries as s8 bytes equal to their bits, 0 or 1 (the bits tile, and the bU tile's A)
__host__ __device__ inline uint32_t pf_01(uint32_t nib) { return (nib * 0x00204081u) & 0x01010101u; }
template <int KD>
__host__ __device__ __forceinline__ uint32_t pf_s8x4(uint32_t nib) {
  return KD == PF_BITS || KD == PF_BU ? pf_01(nib) : pf_pm1(nib);
}
// four digest bytes t (byte q of x) as the bU tile's s8 U, byte q: t_6 = 1 gives (t & 0x3F) + 1 in 1..64, t_6 = 0 gives
// (t & 0x3F) - 64 = (t & 0x3F) | 0xC0 in -64..-1, so no byte carries into the next
__host__ __device__ inline uint32_t pf_u4(uint32_t x) {
  const uint32_t t6 = (x >> 6) & 0x01010101u;
  return ((x & 0x3F3F3F3Fu) + t6) | ((t6 ^ 0x01010101u) * 0xC0u);
}
// eight entries (bits of byte, entry q in bit q) as e2m1 codes of 1 - 2 bit, entry q in nibble q
__host__ __device__ inline uint32_t pf_e2m1x8(uint32_t byte) {
  uint32_t x = (byte | (byte << 12)) & 0x000F000Fu;
  x = (x | (x << 6)) & 0x03030303u;
  x = (x | (x << 3)) & 0x11111111u;
  return 0x22222222u | (x << 3);
}
// the receipts reference's key rules (receipt_folds.cuh): key_ok, and key_ok_r9 = key_ok with neither a nor -a an int8
// ratio mod 2^30
__host__ __device__ inline bool pf_key_ok(uint32_t a) { return rf_key_ok(a); }
__host__ __device__ inline bool pf_key_ok_r9(uint32_t a) { return rf_key_ok_r9(a); }
struct PfKeys {
  uint32_t a0, a1, k0, k1;
};
__host__ __device__ inline PfKeys pf_keys(const uint64_t* st) {
  PfKeys k{0x9E3779B9u, 0x7F4A7C15u, pf_row(st, 14), pf_row(st, 15)};
  int n = 0;
#pragma unroll
  for (int r = 0; r < 16; r++) {
    const uint32_t a = pf_row(st, r);
    if (n < 2 && (PF_R9 ? pf_key_ok_r9(a) : pf_key_ok(a))) {
      if (n == 0) k.a0 = a;
      else k.a1 = a;
      n++;
    }
  }
  return k;
}
template <int NC>
__host__ __device__ __forceinline__ void pf_absorb(uint32_t& y, uint32_t& z, uint32_t d0, uint32_t d1, uint32_t d2,
                                                   uint32_t d3, const PfKeys& k) {
  if (NC == 4 || NC == 5) {
#if PF_ROUNDS > 1
    // The R-round weave's cost dial. Each round re-folds the committed words
    // into the running state with the same XOR-in / LI / ram step, a serial u32 chain of depth PF_ROUNDS a word. The host
    // dfill_ref shares this loop, so dgate stays bit-exact; only the throughput and the receipt value change.
#pragma unroll 1
    for (int rnd = 0; rnd < PF_ROUNDS; rnd++) {
#if defined(PFILL_LI)
      y = rf_li_step(y, d0, d1, d2, d3, k.a0, k.a1, k.k0, k.k1);
#elif defined(PFILL_XIN13)
      y = rf_xin13_step(y, d0, d1, d2, d3, k.a0, k.a1);
#else
      y = rf_ram_step(y, d0, d1, d2, d3, k.a0, k.a1);
#endif
    }
#else
#if defined(PFILL_LI)
    y = rf_li_step(y, d0, d1, d2, d3, k.a0, k.a1, k.k0, k.k1);
#elif defined(PFILL_XIN13)
    y = rf_xin13_step(y, d0, d1, d2, d3, k.a0, k.a1);
#else
    y = rf_ram_step(y, d0, d1, d2, d3, k.a0, k.a1);
#endif
#endif
  } else if (NC == 6) {
    y = (pf_rotl(y, 7) + d0) * 0x9E3779B1u + d1;
    y = (pf_rotl(y, 19) + d2) * 0x85EBCA77u + d3;
  } else if (NC == 7) {
    y = pf_rotl(y, 7) + d0; y = pf_rotl(y, 19) + d1; y = pf_rotl(y, 7) + d2; y = pf_rotl(y, 19) + d3;
  } else if (NC == 8) {
    y = y * k.a0 + d0; y = y * k.a1 + d1; y = y * k.a0 + d2; y = y * k.a1 + d3;
  } else if (NC == 9) {
    y = pf_rotl(y ^ d0, 13) + d1;
    y = pf_rotl(y ^ d2, 19) + d3;
  } else if (NC == 2) {
    rf_duo_step(y, z, d0, d1, d2, d3, k.a0, k.a1, k.k0, k.k1);
  } else {
    y ^= d0 ^ d1 ^ d2 ^ d3;
  }
}
__host__ __device__ inline uint32_t pf_y0(uint32_t pos) { return rf_pfill_y0(pos); }
__host__ __device__ inline uint32_t pf_z0(uint32_t pos) { return rf_pfill_z0(pos); }
__host__ __device__ inline uint64_t pf_fold(uint64_t K, uint32_t pos, uint64_t v) { return smix(smix(K ^ pos) ^ v); }

#ifdef __CUDACC__
#if PF_YIELD
// one 128-byte line an SM: 188 marks in 8 lines put every poll on a few L2 slices, which decode's loads need too
__device__ int pf_busy[256 * 32];
__device__ __forceinline__ int* pf_mark() {
  unsigned sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  return &pf_busy[(sm & 255) * 32];
}
#endif
// rs_compress's rounds, 16 a loop iteration: a fifth of its code, so the hashing phases fetch fewer instructions
__device__ __forceinline__ void pf_round(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t& e, uint64_t& f,
                                         uint64_t& g, uint64_t& h, uint64_t k, uint64_t wi) {
  const uint64_t t1 = h + (rs_rotr(e, 14) ^ rs_rotr(e, 18) ^ rs_rotr(e, 41)) + ((e & f) ^ (~e & g)) + k + wi;
  const uint64_t t2 = (rs_rotr(a, 28) ^ rs_rotr(a, 34) ^ rs_rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
  h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
}
__device__ __forceinline__ void pf_compress(uint64_t* st, uint64_t* w) {
  uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
  for (int i = 0; i < 16; ++i) pf_round(a, b, c, d, e, f, g, h, RS_K[i], w[i]);
#pragma unroll 1
  for (int r = 16; r < 80; r += 16) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint64_t w15 = w[(i + 1) & 15], w2 = w[(i + 14) & 15];
      w[i] += (rs_rotr(w2, 19) ^ rs_rotr(w2, 61) ^ (w2 >> 6)) + w[(i + 9) & 15] +
              (rs_rotr(w15, 1) ^ rs_rotr(w15, 8) ^ (w15 >> 7));
      pf_round(a, b, c, d, e, f, g, h, RS_K[r + i], w[i]);
    }
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}
__device__ __forceinline__ void pf_block(uint64_t dom, uint64_t c0, uint64_t c1, uint64_t tile, uint64_t ctr, uint64_t* st) {
  uint64_t w[16];
  w[0] = c0; w[1] = c1; w[2] = tile; w[3] = dom; w[4] = ctr; w[5] = 0x8000000000000000ull;
#pragma unroll
  for (int i = 6; i < 15; i++) w[i] = 0;
  w[15] = 320;
  rs_iv(st);
#if PF_SHA_ROLL
  pf_compress(st, w);
#else
  rs_compress(st, w);
#endif
}

__device__ __forceinline__ void pf_mma(uint32_t (&d)[4], const uint32_t (&a)[4], uint2 b) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "r"(0));
}
// e2m1 x e2m1 over k64, every block scale 1 (ue4m3 0x38), C = 0: exact f32 sums of 64 products of ±1
__device__ __forceinline__ void pf_omma(uint32_t (&d)[4], const uint32_t (&a)[4], uint2 b) {
  float f0, f1, f2, f3;
  asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, {%11}, {0, 0}, {%11}, {0, 0};"
      : "=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "f"(0.f), "r"(0x38383838u));
  d[0] = __float_as_uint(f0); d[1] = __float_as_uint(f1); d[2] = __float_as_uint(f2); d[3] = __float_as_uint(f3);
}

// The tile loop of any kind KD (F4: the e2m1 tile). Bs: column n's 32 bytes at 32 n, s8 words in order 0 4 1 5 2 6 3 7
// (word q holds k 4q..4q+3), e2m1 words in order (q, h) (word 2q + h holds k 32h + 8q .. 32h + 8q + 7), so a lane's B
// fragment is one 8-byte load in both. Ab: a warp's 32 SHA blocks of A, 16 row words each: P = 32 / KH fragments a pass.
// One tile of W warps (B is 32 x 128 W: lanes 0..7 of each warp make its 8 W blocks; bU: every lane two of its 64 W),
// the whole block's: its receipt in thread 0.
template <int NC, int KD, int W = PF_W>
__device__ __forceinline__ uint64_t pf_tile(uint8_t* Bs, uint32_t (*Ab)[32 * 16], uint64_t c0, uint64_t c1,
                                            unsigned long long tile, int steps) {
  constexpr bool regen = NC == 1 || NC == 2 || NC == 4 || NC == 9, F4 = KD == PF_E2M1;
  constexpr int KH = F4 ? 2 : 1, P = 32 / KH, TN = 128 * W;
  constexpr uint64_t dom = pf_dom(KD);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int S = steps;
  {
    PfKeys key{0, 0, 0, 0};
    if (KD == PF_BU) {
      // blocks 32 w + lane and 32 W + 32 w + lane, then thread 0's key block (one warp's third pass, not every warp's),
      // handed over in Ab[0][0..3], which nothing reads from the tile's end until agen; one copy of the compression's code
#pragma unroll 1
      for (int h = 0; h < 2 + (threadIdx.x == 0); h++) {
        const int b = 32 * w + lane + 32 * W * h;
        uint64_t st[8];
        pf_block(dom, c0, c1, tile, h < 2 ? PF_B + (uint64_t)b : PF_KEY, st);
        if (h == 2) {
          key = pf_keys(st);
          Ab[0][0] = key.a0;
          Ab[0][1] = key.a1;
          Ab[0][2] = key.k0;
          Ab[0][3] = key.k1;
          continue;
        }
#pragma unroll
        for (int c = 0; c < 2; c++) {
          uint4* p = (uint4*)&Bs[(2 * b + c) * 32];
          p[0] = make_uint4(pf_u4(pf_row(st, 8 * c)), pf_u4(pf_row(st, 8 * c + 4)), pf_u4(pf_row(st, 8 * c + 1)),
                            pf_u4(pf_row(st, 8 * c + 5)));
          p[1] = make_uint4(pf_u4(pf_row(st, 8 * c + 2)), pf_u4(pf_row(st, 8 * c + 6)), pf_u4(pf_row(st, 8 * c + 3)),
                            pf_u4(pf_row(st, 8 * c + 7)));
        }
      }
      __syncthreads();
      key = PfKeys{Ab[0][0], Ab[0][1], Ab[0][2], Ab[0][3]};
    } else if (lane <= 8 * KH) {
      uint64_t st[8];
      const int bb = w * 8 + (lane & 7), h = lane >> 3;
      pf_block(dom, c0, c1, tile, lane < 8 * KH ? PF_B + (uint64_t)(KH * bb + h) : PF_KEY, st);
      if (lane < 8 * KH) {
#pragma unroll
        for (int nn = 0; nn < 16; nn++) {
          const uint32_t R = pf_row(st, nn);
          if (F4) {
            uint32_t* p = (uint32_t*)&Bs[(16 * bb + nn) * 32];
#pragma unroll
            for (int q = 0; q < 4; q++) p[2 * q + h] = pf_e2m1x8((R >> (8 * q)) & 255);
          } else {
            uint4* p = (uint4*)&Bs[(16 * bb + nn) * 32];
            p[0] = make_uint4(pf_s8x4<KD>(R & 15), pf_s8x4<KD>((R >> 16) & 15), pf_s8x4<KD>((R >> 4) & 15),
                              pf_s8x4<KD>((R >> 20) & 15));
            p[1] = make_uint4(pf_s8x4<KD>((R >> 8) & 15), pf_s8x4<KD>((R >> 24) & 15), pf_s8x4<KD>((R >> 12) & 15),
                              pf_s8x4<KD>(R >> 28));
          }
        }
      } else {
        key = pf_keys(st);
      }
    }
    if (KD != PF_BU) {
      key.a0 = __shfl_sync(~0u, key.a0, 8 * KH);
      key.a1 = __shfl_sync(~0u, key.a1, 8 * KH);
      key.k0 = __shfl_sync(~0u, key.k0, 8 * KH);
      key.k1 = __shfl_sync(~0u, key.k1, 8 * KH);
    }
    __syncthreads();
    auto agen = [&](int f0) {
      __syncwarp();
      if (f0 + lane / KH < S) {
        uint64_t st[8];
        pf_block(dom, c0, c1, tile, (uint64_t)KH * ((uint64_t)w * S + f0 + lane / KH) + lane % KH, st);
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
    int f0 = 0, next = (w >> 2) & 1 ? P / 2 : P;
    agen(0);
    const uint8_t* bp = Bs + g * 32 + 8 * t;
#if PF_YIELD
    const volatile int* busy = pf_mark();
    int mark = 0;
#endif
#pragma unroll 1
    for (int f = 0; f < S; f += 4) {
      if (f == next) {
        if (regen) {
          f0 = f;
          agen(f0);
        }
        next = f + P;
      }
      uint32_t a[4][4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const uint32_t* r = &Ab[w][KH * (regen ? f - f0 + c : (f + c) & (P - 1)) * 16];
        if (F4) {
          a[c][0] = pf_e2m1x8((r[g] >> (8 * t)) & 255);
          a[c][1] = pf_e2m1x8((r[g + 8] >> (8 * t)) & 255);
          a[c][2] = pf_e2m1x8((r[16 + g] >> (8 * t)) & 255);
          a[c][3] = pf_e2m1x8((r[16 + g + 8] >> (8 * t)) & 255);
        } else {
          a[c][0] = pf_s8x4<KD>((r[g] >> (4 * t)) & 15);
          a[c][1] = pf_s8x4<KD>((r[g + 8] >> (4 * t)) & 15);
          a[c][2] = pf_s8x4<KD>((r[g] >> (16 + 4 * t)) & 15);
          a[c][3] = pf_s8x4<KD>((r[g + 8] >> (16 + 4 * t)) & 15);
        }
      }
#pragma unroll 4
      for (int j = 0; j < TN / 8; j++) {
#if PF_YIELD
        if (j % PF_YIELD_J == 0) {
          if (mark > 0) {
            if (PF_YIELD == 1)
              while (*busy > 0) __nanosleep(PF_YIELD_POLL_NS);
            else
              __nanosleep(PF_YIELD_NS);
            __syncwarp();
          }
          mark = *busy;
        }
#endif
        const uint2 b = *(const uint2*)(bp + j * 256);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          uint32_t d[4];
          if (F4) pf_omma(d, a[c], b);
          else pf_mma(d, a[c], b);
          pf_absorb<NC>(y[c], z[c], d[0], d[1], d[2], d[3], key);
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
    uint64_t x = 0;
    if (threadIdx.x == 0) {
#pragma unroll
      for (int v = 0; v < W; v++) x ^= (uint64_t)Ab[v][0] | (uint64_t)Ab[v][1] << 32;
    }
    return x;
  }
}
template <int NC, int KD>
__device__ __forceinline__ void pf_tiles(uint8_t* Bs, uint32_t (*Ab)[32 * 16], uint64_t c0, uint64_t c1,
                                         unsigned long long tile0, int tiles, int steps, unsigned long long* rec) {
  for (int i = 0; i < tiles; i++) {
    const unsigned long long idx = (unsigned long long)blockIdx.x * tiles + i;
    const uint64_t x = pf_tile<NC, KD>(Bs, Ab, c0, c1, tile0 + idx, steps);
    if (threadIdx.x == 0) rec[idx] = x;
  }
}
template <int NC>
__global__ void __launch_bounds__(32 * PF_W, PF_MINB) dfill_k(uint64_t c0, uint64_t c1, unsigned long long tile0, int tiles,
                                                       int steps, unsigned long long* __restrict__ rec) {
  __shared__ __align__(16) uint8_t Bs[PF_TN * 32];
  __shared__ __align__(16) uint32_t Ab[PF_W][32 * 16];
  pf_tiles<NC, PF_KIND>(Bs, Ab, c0, c1, tile0, tiles, steps, rec);
}

// The phase filler (dcap.cu's PFill, psched's placement): the same tile at PF_PW warps under the phase ids PBIT | (base +
// slot), a tile a block at a time, so a slot is a block's tile (slot off + blockIdx.x tiles + i) and `steps` is a warp's
// fragments (any multiple of 4). The stop flag is read once a tile by thread 0 and taken by the whole block, since a tile
// syncs it. Each slot keeps (tile id, receipt); st[1] counts the tiles done. A block of PF_PW warps holds PF_PW / 8 of the
// pump block's registers (the same 128 a thread) and shared memory (6 PF_PW KB), so less of decode attention's occupancy:
// at 8 warps, qk_dec (64 x 256) keeps 2 of its 4 blocks an SM beside it and pv_dec (70 x 256, 16 KB) 1 of 3.
constexpr unsigned long long PBIT = 1ull << 62;
#define DFILL_P_W PF_PW
__host__ __device__ constexpr double pf_tile_macs(int steps) { return (double)PF_PW * steps * (16 * PF_PW) * 4096.0; }
template <int NC>
__global__ void __launch_bounds__(32 * PF_PW, PF_PMIN)
    dfill_p(uint64_t c0, uint64_t c1, unsigned long long* __restrict__ st, unsigned long long off, int tiles, int steps,
            const int* stop, unsigned long long* __restrict__ rec, unsigned* smbits) {
  __shared__ __align__(16) uint8_t Bs[128 * PF_PW * 32];
  __shared__ __align__(16) uint32_t Ab[PF_PW][32 * 16];
  const unsigned long long base = *(volatile unsigned long long*)st;
  if (threadIdx.x == 0 && smbits) {
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    atomicOr(&smbits[(s / 32) & 7], 1u << (s % 32));
  }
  for (int i = 0; i < tiles; i++) {
    if (stop && __syncthreads_or(threadIdx.x == 0 && *(volatile const int*)stop)) break;
    const unsigned long long slot = off + (unsigned long long)blockIdx.x * tiles + i, tile = PBIT | (base + slot);
    const uint64_t x = pf_tile<NC, PF_KIND, PF_PW>(Bs, Ab, c0, c1, tile, steps);
    if (threadIdx.x == 0) {
      rec[2 * slot] = tile;
      rec[2 * slot + 1] = x;
      atomicAdd(st + 1, 1ull);
    }
  }
}

#endif

// ---------------------------------------------------------------- the host's tile, from the matrices (pfill_host.cpp builds
// this part with g++ for tests/test_pfill_ref.py)
static void pf_block_h(uint64_t dom, uint64_t c0, uint64_t c1, uint64_t tile, uint64_t ctr, uint64_t* st) {
  uint64_t w[16] = {c0, c1, tile, dom, ctr, 0x8000000000000000ull, 0, 0, 0, 0, 0, 0, 0, 0, 0, 320};
  sx::iv(st);
  sx::compress<0, 0>(st, w);
}
// B's planes a column: one (its entries' bits) for the bit tiles, seven for bU (plane p: bit p of each entry's byte)
template <int KD>
constexpr int pf_planes() { return KD == PF_BU ? 7 : 1; }
// a word of A row bits a and B column planes b: the s8 tile's s32 sum 32 - 2 popcount(a ^ b), the e2m1 tile's f32 sum
// 64 - 2 popcount(a ^ b) as its bits, the bits tile's s32 sum popcount(a & b), or bU's s32 sum of U over a's set bits,
// -64 popcount(a) + sum_{p<6} 2^p popcount(a & b_p) + 65 popcount(a & b_6)
template <int KD>
static inline uint32_t pf_word(uint64_t a, const uint64_t* b) {
  if (KD == PF_BU) {
    int s = 65 * __builtin_popcountll(a & b[6]) - 64 * __builtin_popcountll(a);
    for (int p = 0; p < 6; p++) s += __builtin_popcountll(a & b[p]) << p;
    return (uint32_t)s;
  }
  if (KD == PF_BITS) return (uint32_t)__builtin_popcountll(a & b[0]);
  const int s = (KD == PF_E2M1 ? 64 : 32) - 2 * __builtin_popcountll(a ^ b[0]);
  if (KD != PF_E2M1) return (uint32_t)s;
  const float v = (float)s;
  uint32_t u;
  memcpy(&u, &v, sizeof u);
  return u;
}
// B's planes, NP a column, of a tile of TN columns: bU's block b holds columns 2b and 2b + 1, entry (k, 2b + i) digest
// byte 32 i + k
template <int KD>
static std::vector<uint64_t> pf_bcols(uint64_t c0, uint64_t c1, uint64_t tile, int TN) {
  constexpr int KH = KD == PF_E2M1 ? 2 : 1, NP = pf_planes<KD>();
  constexpr uint64_t dom = pf_dom(KD);
  std::vector<uint64_t> Bb(TN * NP, 0);
  uint64_t st[8];
  if constexpr (KD == PF_BU) {
    for (int b = 0; b < TN / 2; b++) {
      pf_block_h(dom, c0, c1, tile, PF_B + (uint64_t)b, st);
      for (int i = 0; i < 2; i++)
        for (int k = 0; k < 32; k++) {
          const uint32_t byte = (pf_row(st, 8 * i + k / 4) >> (8 * (k % 4))) & 255u;
          for (int p = 0; p < NP; p++) Bb[NP * (2 * b + i) + p] |= (uint64_t)((byte >> p) & 1u) << k;
        }
    }
  } else {
    for (int b = 0; b < TN / 16; b++)  // bit k of Bb[n]: entry (k, n), k < 32 KH
      for (int h = 0; h < KH; h++) {
        pf_block_h(dom, c0, c1, tile, PF_B + (uint64_t)(KH * b + h), st);
        for (int i = 0; i < 16; i++) Bb[16 * b + i] |= (uint64_t)pf_row(st, i) << (32 * h);
      }
  }
  return Bb;
}
template <int NC, int KD>
static uint64_t pf_ref(uint64_t c0, uint64_t c1, uint64_t tile, int steps, int W) {
  constexpr int KH = KD == PF_E2M1 ? 2 : 1, NP = pf_planes<KD>();
  constexpr uint64_t dom = pf_dom(KD);
  const int TN = 128 * W;
  uint64_t st[8];
  const std::vector<uint64_t> Bb = pf_bcols<KD>(c0, c1, tile, TN);
  pf_block_h(dom, c0, c1, tile, PF_KEY, st);
  const PfKeys key = pf_keys(st);
  const uint64_t K = dseed(c0, c1, tile);
  std::vector<uint64_t> Ar(16 * std::max(steps, 0));  // bit k of Ar[16 f + i]: entry (row 16 f + i of the warp's rows, k)
  uint64_t h = 0;
  for (int w = 0; w < W; w++) {
    for (int f = 0; f < steps; f++)
      for (int hh = 0; hh < KH; hh++) {
        pf_block_h(dom, c0, c1, tile, (uint64_t)KH * ((uint64_t)w * steps + f) + hh, st);
        for (int i = 0; i < 16; i++) Ar[16 * f + i] = (hh ? Ar[16 * f + i] : 0) | (uint64_t)pf_row(st, i) << (32 * hh);
      }
    for (int lane = 0; lane < 32; lane++) {
      const int g = lane >> 2, t = lane & 3;
      for (int c = 0; c < 4; c++) {
        const uint32_t pos = (uint32_t)((w * 32 + lane) * 4 + c);
        uint32_t y = pf_y0(pos), z = pf_z0(pos);
        for (int f = c; f < steps; f += 4) {
          const uint64_t R0 = Ar[16 * f + g], R1 = Ar[16 * f + g + 8];
          for (int j = 0; j < TN / 8; j++) {
            const uint64_t *b0 = &Bb[NP * (8 * j + 2 * t)], *b1 = b0 + NP;
            pf_absorb<NC>(y, z, pf_word<KD>(R0, b0), pf_word<KD>(R0, b1), pf_word<KD>(R1, b0), pf_word<KD>(R1, b1),
                          key);
          }
        }
        h ^= pf_fold(K, pos, NC == 2 ? (uint64_t)z << 32 | y : (uint64_t)y);
      }
    }
  }
  return h;
}
// steps need not be a multiple of 32 here (dgate's control runs S - 8); the probes have no host tile, and
// dfill_check reports -1 for them
template <int KD>
static uint64_t pf_ref_nc(uint64_t c0, uint64_t c1, uint64_t tile, int nc, int steps, int W) {
  switch (nc) {
    case 1: return pf_ref<1, KD>(c0, c1, tile, steps, W);
    case 2: return pf_ref<2, KD>(c0, c1, tile, steps, W);
    case 4: return pf_ref<4, KD>(c0, c1, tile, steps, W);
    default: return 0;
  }
}
static uint64_t dfill_ref_kind(uint64_t c0, uint64_t c1, uint64_t tile, int nc, int steps, int kind, int W) {
  switch (kind) {
    case PF_S8: return pf_ref_nc<PF_S8>(c0, c1, tile, nc, steps, W);
    case PF_E2M1: return pf_ref_nc<PF_E2M1>(c0, c1, tile, nc, steps, W);
    case PF_BITS: return pf_ref_nc<PF_BITS>(c0, c1, tile, nc, steps, W);
    case PF_BU: return pf_ref_nc<PF_BU>(c0, c1, tile, nc, steps, W);
    default: return 0;
  }
}
// a phase id (bit 62, dfill_p's) names a PF_PW-warp tile, any other id a pump tile of PF_W warps
static inline uint64_t dfill_ref(uint64_t c0, uint64_t c1, uint64_t tile, int nc, int steps) {
  return dfill_ref_kind(c0, c1, tile, nc, steps, PF_KIND, (tile >> 62) == 1 ? PF_PW : PF_W);
}

// pfill-bU's uniformity gate: the s8 bytes of U over `tiles` tiles' B
// blocks and of A over fragments 0 .. frags - 1 of each, from the round's coins, on the host generators dfill_ref reads,
// so dgate's receipts hold the GPU's operands to them. hu, ha: 256 counts each, by the byte's u8 value.
static void pf_bu_hist(uint64_t c0, uint64_t c1, int tiles, int frags, uint64_t* hu, uint64_t* ha) {
  uint64_t st[8];
  memset(hu, 0, 256 * sizeof *hu);
  memset(ha, 0, 256 * sizeof *ha);
  for (int tile = 0; tile < tiles; tile++) {
    for (int b = 0; b < PF_TN / 2; b++) {
      pf_block_h(PF_DOM_BU, c0, c1, (uint64_t)tile, PF_B + (uint64_t)b, st);
      for (int r = 0; r < 16; r++) {
        const uint32_t x = pf_u4(pf_row(st, r));
        for (int q = 0; q < 4; q++) hu[(x >> (8 * q)) & 255u]++;
      }
    }
    for (int f = 0; f < frags; f++) {
      pf_block_h(PF_DOM_BU, c0, c1, (uint64_t)tile, (uint64_t)f, st);
      for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++) {
          const uint32_t x = pf_s8x4<PF_BU>((pf_row(st, r) >> (4 * n)) & 15u);
          for (int q = 0; q < 4; q++) ha[(x >> (8 * q)) & 255u]++;
        }
    }
  }
}
// the gate's verdict on those counts: no U byte outside {±1, ..., ±64} (0 absent) and none of A's outside {0, 1}, and
// each histogram's chi-square on its own values under the bound a 6-sigma (p ~ 1e-9) tail puts on it: 248 at U's 127
// degrees of freedom (Wilson-Hilferty at z = 6), 36 at A's one
struct PfUniform {
  uint64_t u_n, u_outside, u_zero, a_n, a_ones, a_outside;
  int u_values;
  double u_chi2, a_chi2;
  bool ok;
};
constexpr double PF_U_CHI2_MAX = 248.0, PF_A_CHI2_MAX = 36.0;
static PfUniform pf_bu_uniform(uint64_t c0, uint64_t c1, int tiles, int frags) {
  uint64_t hu[256], ha[256];
  pf_bu_hist(c0, c1, tiles, frags, hu, ha);
  PfUniform r{};
  for (int v = 0; v < 256; v++) {
    const int s = (int8_t)v;
    r.u_n += hu[v];
    r.a_n += ha[v];
    if (s == 0 || s < -64 || s > 64) r.u_outside += hu[v];
    else r.u_values += hu[v] != 0;
    if (v > 1) r.a_outside += ha[v];
  }
  r.u_zero = hu[0];
  r.a_ones = ha[1];
  const double eu = (double)r.u_n / 128, ea = (double)r.a_n / 2;
  for (int v = 0; v < 256; v++) {
    const int s = (int8_t)v;
    if (s != 0 && s >= -64 && s <= 64) r.u_chi2 += ((double)hu[v] - eu) * ((double)hu[v] - eu) / eu;
  }
  r.a_chi2 = (((double)ha[0] - ea) * ((double)ha[0] - ea) + ((double)ha[1] - ea) * ((double)ha[1] - ea)) / ea;
  r.ok = r.u_n > 0 && r.a_n > 0 && r.u_outside == 0 && r.a_outside == 0 && r.u_values == 128 &&
         r.u_chi2 < PF_U_CHI2_MAX && r.a_chi2 < PF_A_CHI2_MAX;
  return r;
}
#ifdef PFILL_BU
// dgate's extra check for this build: 16 tiles' B (524288 U) and 2048 fragments of A each (16.8M bits)
static int pf_gate_extra(uint64_t c0, uint64_t c1) {
  const int tiles = 16, frags = 2048;
  const PfUniform r = pf_bu_uniform(c0, c1, tiles, frags);
  printf("{\"mode\": \"dgate-uniform\", \"kind\": \"bU\", \"tiles\": %d, \"frags\": %d, \"u_n\": %llu, \"u_values\": %d, "
         "\"u_zero\": %llu, \"u_outside\": %llu, \"u_chi2\": %.2f, \"u_chi2_max\": %.1f, \"a_n\": %llu, \"a_ones\": %llu, "
         "\"a_outside\": %llu, \"a_chi2\": %.3f, \"a_chi2_max\": %.1f, \"ok\": %s}\n",
         tiles, frags, (unsigned long long)r.u_n, r.u_values, (unsigned long long)r.u_zero,
         (unsigned long long)r.u_outside, r.u_chi2, PF_U_CHI2_MAX, (unsigned long long)r.a_n,
         (unsigned long long)r.a_ones, (unsigned long long)r.a_outside, r.a_chi2, PF_A_CHI2_MAX, r.ok ? "true" : "false");
  return r.ok ? 0 : 1;
}
#define DFILL_GATE_EXTRA pf_gate_extra
#endif

#ifdef __CUDACC__
// one chunk: `grid` blocks of PF_W warps, each `tiles` tiles of PF_W S fragments x PF_TN columns
struct DFill {
  int W = PF_W, NC = 4, S = 512, tiles = 1, grid = 0;
  uint64_t c0 = 0, c1 = 0;
  unsigned long long* rec = nullptr;
  long long per() const { return (long long)grid * tiles; }
  double macs() const { return (double)per() * W * S * (PF_TN / 8) * 4096.0; }
};
// DFILL_PAD_KB: unused dynamic shared memory a pump block, so that PF_CTAS blocks an SM is all that fits (lfill, or the
// step's blocks, beside it)
static size_t g_pf_pad = 0;
static void dfill_launch(const DFill& f, long long chunk, cudaStream_t st) {
  const unsigned long long t0 = (unsigned long long)chunk * f.per();
  const dim3 gr(f.grid), bl(32 * PF_W);
  const size_t pad = g_pf_pad;
  switch (f.NC) {
    case 1: dfill_k<1><<<gr, bl, pad, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 2: dfill_k<2><<<gr, bl, pad, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 3: dfill_k<3><<<gr, bl, pad, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 4: dfill_k<4><<<gr, bl, pad, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 5: dfill_k<5><<<gr, bl, pad, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 6: dfill_k<6><<<gr, bl, pad, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 7: dfill_k<7><<<gr, bl, pad, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 8: dfill_k<8><<<gr, bl, pad, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 9: dfill_k<9><<<gr, bl, pad, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    default: fprintf(stderr, "pfill: NC must be 1 (xor), 2 (duo), 4 (ram), or the probes 3, 5, 6, 7, 8, 9\n"); exit(1);
  }
}
template <int NC> static void dfill_pad_attr() {
  CK(cudaFuncSetAttribute(dfill_k<NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g_pf_pad));
}
static void dfill_prep() {
  if (getenv("DFILL_PAD_KB")) {
    g_pf_pad = (size_t)atoi(getenv("DFILL_PAD_KB")) << 10;
    dfill_pad_attr<1>(); dfill_pad_attr<2>(); dfill_pad_attr<3>(); dfill_pad_attr<4>();
    dfill_pad_attr<5>(); dfill_pad_attr<6>(); dfill_pad_attr<7>(); dfill_pad_attr<8>(); dfill_pad_attr<9>();
  }
  CK(cudaFuncSetAttribute(dfill_k<1>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<2>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<3>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<4>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<5>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<6>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<7>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<8>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_k<9>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
}
// PF_CTAS blocks for each of `grid` SMs, and tiles a block so that a chunk is about 2^DFILL_CHUNK_LOG2 (default 19) MMAs
// an SM
static DFill dfill_make(int grid, int W, int NC, int S, uint64_t c0, uint64_t c1) {
  if (W != PF_W || S < 32 || S % 32) {
    fprintf(stderr, "pfill: W must be %d and S a positive multiple of 32 (W %d, S %d)\n", PF_W, W, S);
    exit(1);
  }
  DFill f;
  f.grid = grid * PF_CTAS; f.W = W; f.NC = NC; f.S = S; f.c0 = c0; f.c1 = c1;
  const int lg = getenv("DFILL_CHUNK_LOG2") ? atoi(getenv("DFILL_CHUNK_LOG2")) : 19;
  f.tiles = std::max(1, (int)((1LL << lg) / ((long long)PF_CTAS * W * S * (PF_TN / 8))));
  CK(cudaMalloc(&f.rec, f.per() * sizeof(unsigned long long)));
  return f;
}
static int dfill_check(const DFill& f, long long chunk, int n) {
  if (f.NC != 1 && f.NC != 2 && f.NC != 4) return -1;  // a probe: not checked
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
#endif
