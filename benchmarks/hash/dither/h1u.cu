// H1 + fx1's unit (pouw.circuit.h1tile) fused on sm_90a / sm_100a, bit for bit served_h1/receipts.py's
// `call_receipts`: a served call's A [M, K] and B [N, K] s8, its units the full 32 x 32 tiles in
// row-block-major order, unit u at position base + u.
//
//   h1_seeds   one thread a unit: seed_j and sigma_R (seeds.seed / receipt_seed, SHA-512 over 224 bytes, two
//              compressions each), the digests' big-endian words, 16 a unit
//   h1u        one warp a unit. The lane's fold keys are block `lane` of sigma_R's stream. seed_j's midstate goes to
//              shared memory once; then per 16-segment chunk c each lane hashes its row's block lane RB + c and its
//              column's block 32 RB + lane CB + c into shared memory (the rows' dither bits b and the columns' coin bits c
//              of 16 segments), after one first pass over block K / 512 (the rows' digit coins, the columns' U bytes).
//              A segment: X = A + b and W' = B + c + U (mod 2^8) built in registers as mma.sync m16n8k32 s8 fragments
//              (2 m-tiles x 4 n-tiles: the lane's 32 entries are exactly h1tile.TileShape's at segment.row_slot / col_slot),
//              T += X W'^T, and Y += X P (P the identity under the fragments' K order, so Y is the residue sums); then
//              the set T_s folded: entry q times fx1_key(q), absorbed by duo chain q % 8 at key 2 (q / 8) + (q % 8 >= 4);
//              each set's XOR over the lane's chains to a ring, reduced across lanes once a chunk. After the KS sets: the
//              digits A_d from Y and the coins, O' = T_last + A_0 (-U)^T and D_d = A_d U^T on the tensor cores (A_d
//              packed from the accumulator layout in registers, no shuffle), folded as sets KS .. KS + m - 1.
// The K order inside a segment is a bijection both operands share (lane q's 8 bytes 8q..8q+7 are its A/B fragment's
// k = 4q..4q+3 and 16+4q..16+4q+3), so every product is the segment's; the tail's order is the accumulator columns'.
// A dead row tile (live[ri] == 0, PouwH1Tile_v4's dead tile) writes the zero receipt and derives no noise.
// Requires K % 512 == 0 (KS % 16: each noise chunk is 16 segments' bits) and m = n_digits(KS) in {2, 3}: m is 2 up to
// K = 4096 and 3 from K = 4128 to K = 524288 (KS <= 16384); the entries refuse any other m.
//
// kh_unit_ws is the same receipts in two launches, for calls too small to fill the GPU one warp a unit:
//   h1s_noise  one warp an (unit, chunk) item, J blocks a lane: every block of seed_j's stream the unit reads (the rows'
//              blocks t = lane RB + c, the columns' 32 RB + lane CB + c), its 16 words to the caller's workspace
//   h1u<WS>    h1u reading those blocks instead of hashing them
// Both forms load the A and B words H1U_PD segments ahead while the current segment is multiplied and folded (WS always,
// the fused form with H1U_PF), in a ring of H1U_PD register sets (H1U_RING = 1: H1U_PD + 1 sets, each segment's load
// going to the set the segment before it read, so a load never waits on the copy out of its set); H1U_L2PF also
// prefetches into L2 the A and B lines 16 segments ahead and (WS) the next chunk's workspace blocks. H1U_WARPS is the
// warps a CTA (one a unit).
// H1U_SPLIT builds h1p in h1u's place: a pair of warps a unit, for calls whose units leave most of the GPU idle (decode).
// Warp p of the pair multiplies n-tiles p and p + 2 (T and Y), so it holds exactly the entries chains 4 p .. 4 p + 3 absorb
// (key t = 2 i + p), and folds those; it hashes (or loads) one of the chunk's two blocks a lane, p = 0 the rows', p = 1 the
// columns', into the pair's shared buffers. A set's ring word is the XOR of the two warps' (every chain's XOR), and A_d
// the OR of their halves (byte 2 (nt & 1) + (e & 1) of a register is n-tile nt's), exchanged in shared memory.
// H1U_KSPLIT = P adds h1k, the unit's K split for the one-stream two-launch form: a CTA a unit, warp 0 its fold and
// warps 1..P its producers. The duo chains are a serial recurrence over the sets, so only the work feeding them splits:
// producer p takes segments P r + p - 1 (round r), loads and builds their fragments and runs every MMA, writing the
// segment's own product dT_s = X_s W'_s^T to shared memory and keeping its Y, Cw and Pz partials; warp 0 adds
// T_s = T_{s-1} + dT_s and folds the round's P sets in segment order while the producers build the next round. Before
// the tail the producers' Y, Cw and Pz go to warp 0 as sums, so every set, digit and z is h1u's (integer sums, the same
// order of folds).
//
//   kh_seeds(data, base, U, seeds, stream)   data: SEED_TAG, RECEIPT_TAG (3 words each), salt, root_a, root_b (8 each)
//   kh_unit(A, B, M, N, K, seeds, live, out, stream)   out: U x kh_words(K) u32 (live may be null: every unit live)
//   kh_units_multi(calls, ncalls, units, M, live, stream)   kh_unit of ncalls calls (UCall) in one launch (h1um), for a
//              step's units dispatched together, whose warps fill the GPU where one call's leave it idle
//   kh_units_multi_sm(calls, ncalls, units, M, live, q, den, next, stream)   the same on the SMs whose id % den < q alone
//              (h1um_sm); kh_units_multi_smc(.., q, den, cap, next, stream) with at most cap CTAs an SM besides
//              (next kh_smc_words() u32), so every SM keeps the rest of its slots to the streams beside it
//   kh_unit_ws(A, B, M, N, K, seeds, live, out, ws, J, stream)   ws: kh_ws_bytes(M, N, K) bytes, the caller's
//   kh_one(A, B, rows, N, K, seeds, live, out, ws, J, z, zo, y, ydump, sa, sw, step, M, stream)   the one-stream form
//              (OneOut, below): kh_unit_ws' receipts (kh_unit's if ws is null) and the served z and y from the same MMAs
//   kh_one_peel(the same)   kh_one peeled (PL, below): z = A B^T, so y is today's
//   kh_one_noise(the same), kh_one_units(the same)   kh_one's noise launch and its units launch, apart
//   kh_one_j10(A, B, rows, N, K, seeds, cseed, live, out, ws, J, z, zo, y, ydump, sa, sw, step, M, stream)   kh_one's
//              two-launch form on J10-R's call-scale noise (measured only; at the end of this file)
//   kh_seeds_at(data, idx, pos, U, seeds, stream)   kh_seeds of U units anywhere: unit u's at position pos[u] from
//              data's record idx[u] (kh_seeds' data, 30 words a record)
//   kh_lanes(A, B, L, K, seeds, out, lr, stream)   L drawn leaves on gathered operands: their receipts and lane records
//              (option A's partials, partials.py; at the end of this file)
//   kh_seeds_j10r(data, call, base, U, seeds, stream), kh_units_multi_j10r(calls, ncalls, units, M, live, mode, stream)
//              kh_units_multi's receipts under J10-R's call-scale noise (the verifier's form; at the end of this file), with
//              kh_units_multi_j10r_sm and a served step's seeds from h1r's call table, kh_seeds_j10r_h1r; a call's
//              workspace is kh_j10r_jw_bytes(M, N, K): kh_j10r_rows_bytes(M, K) but with H1U_JW (its column tiles and
//              W' once a call, after JX's rows); kh_j10r_workspace_bytes(M, N, K, rc) is the size a caller allocates
//              for either form (rc: kh_units_multi_j10rc's), from the build's own flags, kh_j10r_flags() (1 H1U_JW,
//              2 H1U_JRC)
//   kh_units_multi_j10rc(calls, ncalls, units, M, live, mode, stream)   (H1U_JRC) kh_units_multi_j10r's receipts with
//              each call's column tiles hashed once a call too, its workspace kh_j10r_ws_bytes(M, N, K), its units
//              H1U_JSWZ row tiles at a time (H1U_JSWZQ: a block's warps a patch of them); built with H1U_JRY = 1 too,
//              each row tile's tail digits computed once a call (h1jr_ady) and read by its units;
//              kh_units_multi_j10rc_sm(calls, ncalls, units, M, live, q, den, next, stream) its blocks, then its units
//              on q / den of the SMs (as kh_units_multi_j10r_sm); with H1U_JAP either form's workspace holds the
//              call's A' too (kh_j10r_rows_bytes and kh_j10r_ws_bytes count it), and kh_j10r_row_fault_word(M, N, K,
//              rc) is the word of it a row fault flips
//   kh_words(K), kh_digits(K), kh_ws_bytes(M, N, K), kh_j10_ws_bytes(rows, N, K), kh_noise_blocks(K), kh_fault_shift()
#include "h1sha.cuh"
#include <cuda_bf16.h>

#ifndef H1U_WARPS
#define H1U_WARPS 4
#endif
constexpr int WARPS = H1U_WARPS;
// CTAs an SM the register allocation must leave room for (h1u_b3.so: 3, at most 168 registers a thread)
#ifndef H1U_MINB
#define H1U_MINB 1
#endif
#ifndef H1U_PF
#define H1U_PF 0
#endif
#ifndef H1U_PD
#define H1U_PD 1
#endif
#ifndef H1U_RING
#define H1U_RING 0
#endif
#ifndef H1U_L2PF
#define H1U_L2PF 0
#endif
#ifndef H1U_SPLIT
#define H1U_SPLIT 0
#endif
// H1U_NB2 = 1: the fused form hashes a chunk's two blocks a lane (its row's and its column's, one stream) as one paired
// compression (noise_block2), their rounds interleaved; the same blocks
#ifndef H1U_NB2
#define H1U_NB2 0
#endif
// H1U_KSPLIT = P (2 or 4): the one-stream two-launch call (kh_one, kh_one_peel) runs h1k, a CTA a unit, for calls whose
// live units number at most H1K_MAXU (decode's qkv, o and down); larger calls keep h1u
#ifndef H1U_KSPLIT
#define H1U_KSPLIT 0
#endif
#ifndef H1K_MAXU
#define H1K_MAXU 320
#endif
// H1K_AUTO = 1 (with H1U_KSPLIT = 4): 4 producers where the live units fit in one wave of h1k<4> (decode's o and down),
// else 2 up to H1K_MAXU (qkv's 192 units exceed an H200's or a B200's SMs, and h1k<4> runs one CTA an SM)
#ifndef H1K_AUTO
#define H1K_AUTO 0
#endif
// H1K_RONLY = 1 (with H1U_KSPLIT): kh_one writes its receipts only, for a caller whose served z and y come from another
// kernel (dither-a): h1k skips the wrap words, Cw and every z, zo and y store, and calls past h1k take h1u's receipts-only
// form (kh_unit_ws'); the receipts are the same words
#ifndef H1K_RONLY
#define H1K_RONLY 0
#endif
// H1K_PD = D >= 1: h1k's producers load their A and B words D rounds ahead in a ring of D + 1 register sets, round r's
// load going to the set round r - 1 read; 0 keeps one set, copied out and reloaded a round ahead (h1k<4> runs one CTA an
// SM and h1k<2> two, whatever their registers). H1K_NXD = 1: a chunk's noise chunks, loaded at its first round, are stored
// at its second round's end rather than its first's, so a round hides their load. H1K_L2PF = S: the producer whose
// segment is S segments before a 128-byte line's first prefetches that line of the unit's 32 B rows into L2 (A's rows,
// every unit's of the row tile, stay in L2)
#ifndef H1K_PD
#define H1K_PD 0
#endif
#ifndef H1K_NXD
#define H1K_NXD 0
#endif
#ifndef H1K_L2PF
#define H1K_L2PF 0
#endif
// H1K_CPA = S >= 2 (in H1K_PD's place): h1k's producers copy their A and B words with cp.async into a ring of S stages in
// dynamic shared memory (S x 2 KiB a producer), round r + S - 1's issued at round r into the stage round r - 1 read
// (behind that round's barrier), so S - 1 rounds of loads are in flight and none holds a register
#ifndef H1K_CPA
#define H1K_CPA 0
#endif
// H1U_CARVE = 1: kh_one's units kernels ask for the maximum shared-memory carveout, once a process each, before their
// first launch (the driver's default on an RTX PRO 6000 holds one 42 KB h1u CTA an SM, where registers allow two)
#ifndef H1U_CARVE
#define H1U_CARVE 0
#endif
static_assert(!H1K_AUTO || H1U_KSPLIT == 4, "H1K_AUTO picks between h1k<4> and h1k<2>: build it with H1U_KSPLIT = 4");
static_assert(!H1K_RONLY || H1U_KSPLIT, "H1K_RONLY is kh_one's K split without z and y: build it with H1U_KSPLIT");
static_assert(16 % (H1U_PD + H1U_RING) == 0, "a chunk's 16 segments are whole rings of H1U_PD (+ 1 with H1U_RING) sets");
static_assert(!H1U_RING || !H1U_SPLIT, "H1U_RING is h1u's ring; h1p keeps H1U_PD sets");
static_assert(H1K_PD >= 0 && H1K_PD <= 3, "h1k's ring: 0 (one set) to 3 rounds ahead");
static_assert(H1K_L2PF % 4 == 0, "h1k's L2 prefetch distance is whole 128-byte lines of segments");
static_assert(H1K_CPA == 0 || (H1K_CPA >= 2 && H1K_CPA <= 8 && !H1K_PD && !H1K_L2PF),
              "h1k's cp.async ring: 2 to 8 stages, in place of H1K_PD's register ring and H1K_L2PF");
static_assert(H1U_KSPLIT == 0 || H1U_KSPLIT == 2 || H1U_KSPLIT == 4, "h1k's producers: 2 or 4");
static_assert(!H1U_SPLIT || (WARPS % 2 == 0 && WARPS <= 4), "a split unit is a pair of warps; 2 pairs fill 48 KiB");

constexpr u32 mix32c(u32 x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
// segment.KA / KK / ROT (duo's multiplier, xor key and rotation at tile position t) and fx1_key(q)
struct KeysT {
  u32 ka[8], kk[8], rot[8], fx1[32];
  constexpr KeysT() : ka(), kk(), rot(), fx1() {
    for (u32 t = 0; t < 8; ++t) {
      ka[t] = mix32c(0xA0000000u + 2 * t) | 1u;
      kk[t] = mix32c(0xB0000000u + 2 * t + 1);
      rot[t] = ((t * 7 + 5) % 31) + 1;
    }
    for (u32 q = 0; q < 32; ++q) fx1[q] = mix32c(0xA0000000u + 2 * (16 + 32 * q)) | 1u;
  }
};
__device__ constexpr KeysT KEYS{};

__host__ __device__ constexpr int n_digits(int ks) {
  // segment.n_digits: the least m with -128^m <= -128 KS and 127 KS // 128^(m - 1) <= 127
  long long p = 128;
  for (int m = 1; m < 8; ++m, p *= 128)
    if (-p <= -128LL * ks && 127LL * ks / (p / 128) <= 127) return m;
  return 0;
}

struct alignas(16) WarpSmem {
  static constexpr bool RX = false;
  NoiseMid nm;          // seed_j's stream
  u32 rowbits[32][17];  // row i's block: 16 halves (one segment's 32 dither bits each), padded against bank conflicts
  u32 colbits[32][17];
  u32 coin[32][2];      // row i's digit coins: half d's bit rho is z_d(i, rho)
  u32 utab[32][9];      // column j's U + 128 bytes, rho = 0..31
  u32 ring[32][33];     // [2 j + (0 y | 1 z)][lane], a chunk's 16 sets
};
// RC's units under H1U_JRX: no seed stream, and no ring (a set's two words are warp reductions, red_put)
struct alignas(16) WarpSmemX {
  static constexpr bool RX = true;
  u32 rowbits[32][17];
  u32 colbits[32][17];
  u32 coin[32][2];
  u32 utab[32][9];
};

__device__ __forceinline__ u32 expand4(u32 n) { return (n * 0x00204081u) & 0x01010101u; }
// bytewise a + bb mod 2^8 for bb's bytes in {0, 1}
__device__ __forceinline__ u32 add01(u32 a, u32 bb) { return ((a & 0x7F7F7F7Fu) + bb) ^ (a & 0x80808080u); }
// bytewise (b + 128) + v mod 2^8, v's bytes in [64, 193] (U + 128 + c): B + c + U
__device__ __forceinline__ u32 addw(u32 b, u32 v) {
  return ((b & 0x7F7F7F7Fu) + (v & 0x7F7F7F7Fu)) ^ (~(b ^ v) & 0x80808080u);
}

// W' = B + c + U's int8 wrap at each byte as s8 WB = wb+ - wb- (b = B, v = U + 128 + c as addw takes them): wb+ where B >= 0
// and c + U >= 0 (v's top bit set) and the 7-bit sum carries into the sign, wb- where both are < 0 and it doesn't
__device__ __forceinline__ u32 wrapw(u32 b, u32 v) {
  const u32 lo = (b & 0x7F7F7F7Fu) + (v & 0x7F7F7F7Fu);
  const u32 p = ~b & v & lo, n = b & ~v & ~lo;
  u32 nn;  // each byte's sign replicated (prmt's sign mode: selector nibbles 8..11)
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(nn) : "r"(n), "r"(0u), "n"(0xBA98));
  return nn | ((p >> 7) & 0x01010101u);
}

__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// H1U_L2HINT = 64 | 128 | 256 (h1u's and h1k's A and B segment loads) and H1U_L2HINT_WS (their workspace loads): the
// load's L2 prefetch size, so a miss brings the line the next segments (blocks) read instead of the one 32-byte sector
#ifndef H1U_L2HINT
#define H1U_L2HINT 0
#endif
#ifndef H1U_L2HINT_WS
#define H1U_L2HINT_WS 0
#endif
template <int H>
__device__ __forceinline__ uint2 ld2(const void* p) {
  if constexpr (H == 0) {
    return __ldg(reinterpret_cast<const uint2*>(p));
  } else {
    static_assert(H == 64 || H == 128 || H == 256, "an L2 prefetch size is 64, 128 or 256 bytes");
    uint2 v;
    if constexpr (H == 64) asm("ld.global.nc.L2::64B.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    else if constexpr (H == 128) asm("ld.global.nc.L2::128B.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    else asm("ld.global.nc.L2::256B.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
  }
}
template <int H>
__device__ __forceinline__ uint4 ld4(const void* p) {
  if constexpr (H == 0) {
    return __ldg(reinterpret_cast<const uint4*>(p));
  } else {
    static_assert(H == 64 || H == 128 || H == 256, "an L2 prefetch size is 64, 128 or 256 bytes");
    uint4 v;
    if constexpr (H == 64)
      asm("ld.global.nc.L2::64B.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else if constexpr (H == 128)
      asm("ld.global.nc.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else
      asm("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
  }
}

// L2 eviction priorities, the bytes unchanged: H1U_EVF = 1 h1u's and h1k's B (weight) and workspace loads evict first
// (each is read once a step), H1S_EVL = 1 h1s_noise's workspace stores evict last (the units read them a few calls later)
#ifndef H1U_EVF
#define H1U_EVF 0
#endif
#ifndef H1S_EVL
#define H1S_EVL 0
#endif
// H1U_EVFU = 0 (with H1U_EVF): h1u's B loads keep the normal policy, since a warp reads a row's 128-byte line over four
// segments and an evict-first line can go before its last; h1k's producers read a line in one round
#ifndef H1U_EVFU
#define H1U_EVFU 1
#endif
template <int H, bool EF>
__device__ __forceinline__ uint2 ld2e(const void* p) {
  if constexpr (!EF) {
    return ld2<H>(p);
  } else {
    static_assert(H == 0 || H == 64 || H == 128 || H == 256, "an L2 prefetch size is 64, 128 or 256 bytes");
    u64 pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    uint2 v;
    if constexpr (H == 0)
      asm("ld.global.nc.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;" : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(pol));
    else if constexpr (H == 64)
      asm("ld.global.nc.L2::cache_hint.L2::64B.v2.u32 {%0, %1}, [%2], %3;" : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(pol));
    else if constexpr (H == 128)
      asm("ld.global.nc.L2::cache_hint.L2::128B.v2.u32 {%0, %1}, [%2], %3;" : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(pol));
    else
      asm("ld.global.nc.L2::cache_hint.L2::256B.v2.u32 {%0, %1}, [%2], %3;" : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(pol));
    return v;
  }
}
template <int H, bool EF>
__device__ __forceinline__ uint4 ld4e(const void* p) {
  if constexpr (!EF) {
    return ld4<H>(p);
  } else {
    static_assert(H == 0 || H == 64 || H == 128 || H == 256, "an L2 prefetch size is 64, 128 or 256 bytes");
    u64 pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    uint4 v;
    if constexpr (H == 0)
      asm("ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
          : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
    else if constexpr (H == 64)
      asm("ld.global.nc.L2::cache_hint.L2::64B.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
          : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
    else if constexpr (H == 128)
      asm("ld.global.nc.L2::cache_hint.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
          : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
    else
      asm("ld.global.nc.L2::cache_hint.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
          : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
    return v;
  }
}
template <bool EL>
__device__ __forceinline__ void st4l(uint4* p, uint4 v) {
  if constexpr (!EL) {
    *p = v;
  } else {
    u64 pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
                 "r"(v.w), "l"(pol)
                 : "memory");
  }
}
// 8 bytes global -> shared through cp.async (H1K_CPA), with ld2e's L2 prefetch size and eviction policy
#define H1U_CP8(Q) asm volatile("cp.async.ca.shared.global" Q " [%0], [%1], 8;" ::"r"(d), "l"(p) : "memory")
#define H1U_CP8P(Q) asm volatile("cp.async.ca.shared.global.L2::cache_hint" Q " [%0], [%1], 8, %2;" ::"r"(d), "l"(p), "l"(pol) : "memory")
template <int H, bool EF>
__device__ __forceinline__ void cp8e(void* s, const void* p) {
  static_assert(H == 0 || H == 64 || H == 128 || H == 256, "an L2 prefetch size is 64, 128 or 256 bytes");
  const u32 d = (u32)__cvta_generic_to_shared(s);
  if constexpr (!EF) {
    if constexpr (H == 0) H1U_CP8("");
    else if constexpr (H == 64) H1U_CP8(".L2::64B");
    else if constexpr (H == 128) H1U_CP8(".L2::128B");
    else H1U_CP8(".L2::256B");
  } else {
    u64 pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    if constexpr (H == 0) H1U_CP8P("");
    else if constexpr (H == 64) H1U_CP8P(".L2::64B");
    else if constexpr (H == 128) H1U_CP8P(".L2::128B");
    else H1U_CP8P(".L2::256B");
  }
}
#undef H1U_CP8
#undef H1U_CP8P
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void mma_s8(int (&d)[4], const u32 (&a)[4], const u32 (&b)[2]) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// H1U_FROT = 1: the fold's rotation on the FMA pipe, rotl(uu, r) + y as the two halves of uu 2^r (one mul.wide.u32,
// 2^r read from constant memory so ptxas can't turn it back into a shift) and y summed (one IADD3), where 0 takes a
// funnel shift and an add on the ALU pipe; the same words
#ifndef H1U_FROT
#define H1U_FROT 0
#endif
#if H1U_FROT
__constant__ u32 FOLD_P2[8] = {1u << KeysT{}.rot[0], 1u << KeysT{}.rot[1], 1u << KeysT{}.rot[2], 1u << KeysT{}.rot[3],
                               1u << KeysT{}.rot[4], 1u << KeysT{}.rot[5], 1u << KeysT{}.rot[6], 1u << KeysT{}.rot[7]};
#endif

// one set: the lane's entry q = 4 t + e is acc[t][e]; chain c absorbs entry 8 i + c at step i, key t = 2 i + (c >= 4)
__device__ __forceinline__ void fold(const int (&acc)[8][4], u32 (&y)[8], u32 (&z)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int t = 2 * i + (c >> 2), e = c & 3;
      const u32 p = (u32)acc[t][e] * KEYS.fx1[8 * i + c];
      const u32 uu = y[c] ^ p ^ KEYS.kk[t];
      y[c] = uu * KEYS.ka[t] + z[c];
#if H1U_FROT
      u64 w;
      asm("mul.wide.u32 %0, %1, %2;" : "=l"(w) : "r"(uu), "r"(FOLD_P2[t]));
      z[c] = (u32)w + (u32)(w >> 32) + y[c];
#else
      z[c] = __funnelshift_l(uu, uu, KEYS.rot[t]) + y[c];
#endif
    }
  }
}

template <class Sm>
__device__ __forceinline__ void ring_put(Sm& S, int slot, u32 lane, const u32 (&y)[8], const u32 (&z)[8]) {
  S.ring[2 * slot][lane] = y[0] ^ y[1] ^ y[2] ^ y[3] ^ y[4] ^ y[5] ^ y[6] ^ y[7];
  S.ring[2 * slot + 1][lane] = z[0] ^ z[1] ^ z[2] ^ z[3] ^ z[4] ^ z[5] ^ z[6] ^ z[7];
}

template <class Sm>
__device__ __forceinline__ u32 ring_xor(const Sm& S, u32 row) {
  u32 x = 0;
#pragma unroll
  for (int l = 0; l < 32; ++l) x ^= S.ring[row][l];
  return x;
}

// ring_put then ring_xor without the ring (WarpSmemX): lane 2 slot keeps the set's y word, lane 2 slot + 1 its z word
__device__ __forceinline__ void red_put(u32& w, u32 slot, u32 lane, const u32 (&y)[8], const u32 (&z)[8]) {
  const u32 ry = __reduce_xor_sync(0xFFFFFFFFu, y[0] ^ y[1] ^ y[2] ^ y[3] ^ y[4] ^ y[5] ^ y[6] ^ y[7]);
  const u32 rz = __reduce_xor_sync(0xFFFFFFFFu, z[0] ^ z[1] ^ z[2] ^ z[3] ^ z[4] ^ z[5] ^ z[6] ^ z[7]);
  w = lane == 2 * slot ? ry : lane == 2 * slot + 1 ? rz : w;
}

// the lane records (DG, OneOut::lr): a chunk's 32 ring rows (receipt words at .. at + 31) as every lane's own, lane l's
// at l W + at of the unit's record, thread r writing word at + r of each lane (row r, column l: no bank conflict)
template <class Sm>
__device__ __forceinline__ void lane_put(const Sm& S, u32* lw, u32 W, u32 at, u32 lane) {
#pragma unroll 8
  for (int l = 0; l < 32; ++l) lw[(u64)l * W + at + lane] = S.ring[lane][l];
}

// the tail's 2 MD ring rows (receipt words at .. at + 2 MD - 1) of every lane
template <int MD, class Sm>
__device__ __forceinline__ void lane_tail(const Sm& S, u32* lw, u32 W, u32 at, u32 lane) {
#pragma unroll
  for (int i = 0; i < 2 * MD; ++i) {
    const u32 k = 32 * i + lane, l = k / (2 * MD), r = k % (2 * MD);
    lw[(u64)l * W + at + r] = S.ring[r][l];
  }
}

__global__ void __launch_bounds__(128) h1_seeds(const u64* __restrict__ data, u64 base, u32 U, u64* __restrict__ out) {
  const u32 u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= U) return;
  const u64 pos = base + u;
#pragma unroll 1
  for (int tag = 0; tag < 2; ++tag) {
    u64 msg[32];
    msg[0] = pos;
#pragma unroll
    for (int w = 0; w < 27; ++w) msg[1 + w] = (w < 3 ? data[3 * tag + w] : data[3 + w]) ^ pos;
    msg[28] = 0x8000000000000000ULL;
    msg[29] = 0;
    msg[30] = 0;
    msg[31] = 224 * 8;
    u64 s[8];
    sha512_words(msg, 2, s);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[16 * (u64)u + 8 * tag + j] = s[j];
  }
}

// h1_seeds of units anywhere (kh_seeds_at): unit u at position pos[u], from data's record idx[u]
__global__ void __launch_bounds__(128) h1_seeds_at(const u64* __restrict__ data, const u32* __restrict__ idx,
                                                   const u64* __restrict__ at, u32 U, u64* __restrict__ out) {
  const u32 u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= U) return;
  const u64 pos = at[u];
  const u64* d = data + 30 * (u64)idx[u];
#pragma unroll 1
  for (int tag = 0; tag < 2; ++tag) {
    u64 msg[32];
    msg[0] = pos;
#pragma unroll
    for (int w = 0; w < 27; ++w) msg[1 + w] = (w < 3 ? d[3 * tag + w] : d[3 + w]) ^ pos;
    msg[28] = 0x8000000000000000ULL;
    msg[29] = 0;
    msg[30] = 0;
    msg[31] = 224 * 8;
    u64 s[8];
    sha512_words(msg, 2, s);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[16 * (u64)u + 8 * tag + j] = s[j];
  }
}

__global__ void __launch_bounds__(256) h1s_noise(const u64* __restrict__ seeds, u32 units, u32 ncb, u32 nb, u32 J,
                                                 const uint8_t* __restrict__ live, u32* __restrict__ ws) {
  const u32 lane = threadIdx.x & 31;
  const u32 per = 32 * J, chunks = (nb + per - 1) / per;
  const u64 item = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (item >= (u64)units * chunks) return;
  const u32 u = (u32)(item / chunks), chunk = (u32)(item % chunks);
  if (live && !live[u / ncb]) return;
  u64 seed[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) seed[j] = __ldg(seeds + 16 * (u64)u + j);
  NoiseMid nm;
  noise_mid(seed, nm);
#pragma unroll 1
  for (u32 j = 0; j < J; ++j) {
    const u32 t = chunk * per + 32 * j + lane;
    if (t >= nb) break;
    u64 s[8];
    noise_block(nm, t, s);
    uint4* o = reinterpret_cast<uint4*>(ws + 16 * ((u64)u * nb + t));
#pragma unroll
    for (int q = 0; q < 4; ++q)
      st4l<H1S_EVL>(o + q, make_uint4((u32)(s[2 * q] >> 32), (u32)s[2 * q], (u32)(s[2 * q + 1] >> 32), (u32)s[2 * q + 1]));
  }
}

// J10 banded (kh_one_band; measured only): h1s_noise's blocks with each band's row tiles hashed by its leader alone, unit
// u = ri ncb + cj a leader where cj % band == 0 (rows_nb = 32 RB: a unit's row tiles come first)
__global__ void __launch_bounds__(256) h1j_noise(const u64* __restrict__ seeds, u32 units, u32 ncb, u32 nb, u32 rows_nb,
                                                 u32 J, u32 band, const uint8_t* __restrict__ live, u32* __restrict__ ws) {
  const u32 lane = threadIdx.x & 31;
  const u32 per = 32 * J, chunks = (nb + per - 1) / per;
  const u64 item = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (item >= (u64)units * chunks) return;
  const u32 u = (u32)(item / chunks), chunk = (u32)(item % chunks);
  if (live && !live[u / ncb]) return;
  const u32 t0 = (u % ncb) % band ? rows_nb : 0;
  if ((chunk + 1) * per <= t0) return;
  u64 seed[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) seed[j] = __ldg(seeds + 16 * (u64)u + j);
  NoiseMid nm;
  noise_mid(seed, nm);
#pragma unroll 1
  for (u32 j = 0; j < J; ++j) {
    const u32 t = chunk * per + 32 * j + lane;
    if (t >= nb) break;
    if (t < t0) continue;
    u64 s[8];
    noise_block(nm, t, s);
    uint4* o = reinterpret_cast<uint4*>(ws + 16 * ((u64)u * nb + t));
#pragma unroll
    for (int q = 0; q < 4; ++q)
      o[q] = make_uint4((u32)(s[2 * q] >> 32), (u32)s[2 * q], (u32)(s[2 * q + 1] >> 32), (u32)s[2 * q + 1]);
  }
}

// The one-stream form (kh_one; PouwH1Tile_v3): the same walk and receipts, plus each cell's served z =
// O' - sum_d 128^d D_d + 256 X WB^T mod 2^32, WB = wb+ - wb- (wrapw) from B and the noise, carried by a third MMA set a
// segment into its own accumulator; then y = RN_bf16(RN32(RN32(z) s_a[row]) s_w[col]) (int8_exact.dequantize), s_a = 0
// for rows >= m. The step (m, flags) is read from the device, as dither_one's H1OStep {parity, m, pad, flags}.
enum { F_DUMP = 1, F_NW_TILES = 2, F_NW_UNITS = 4, F_FAULT = 8, F_FAULT_WB = 16, F_FAULT_XA = 32 };
// F_FAULT: 2^FAULT_SHIFT added to unit 0's accumulator of cell (0, 0) after segment 0's MMAs, before its set is folded;
// F_FAULT_WB: wb+ of unit 0's column 0 at k = 0 flipped (WB moves by +1 where wb+ was clear, else -1); F_FAULT_XA (the
// peel only): xa of unit 0's row 0 at k = 0 flipped (row 0's z moves by +-256 B[col][0]; no receipt word); F_NW_TILES writes
// no z, F_NW_UNITS no receipt (the gate's no-write control); F_DUMP copies y rows < m to ydump
constexpr int FAULT_SHIFT = 24;

struct OneOut {
  int* z;               // [rows, N] s32: the committed z of every unit's cells
  int* zo;              // [M, N] s32 or null: rows < M of z
  __nv_bfloat16* y;     // [M, N] bf16
  __nv_bfloat16* ydump; // [rows, N] or null
  const float* sa;      // s_a, rows < m read
  const float* sw;      // s_w [N]
  const int* step;      // {parity, m, pad, flags}
  u32 M;
  u32 band = 1;         // J10 banded (kh_one_band; measured only): unit (ri, cj)'s rows from its band leader
                        // (ri, (cj / band) band)'s workspace region, its columns from its own
  u32* lr = nullptr;    // DG (kh_lanes): unit u's record of its 32 lanes' words at lr + 32 u W (W = kh_words(K))
};

// PL (kh_one_peel; the peel): z = A B^T, the served model's own product, from the same stream: with xa = [A = 127
// and b = 1] (where the int8 X = A + b wraps, X = A + b - 256 xa), X (B + c)^T - X c^T - b B^T + 256 xa B^T = A B^T, so
// z = O' - sum_d 128^d D_d + 256 (X WB^T + xa B^T) - (X c^T + b B^T): two more MMAs a fragment (X by the coins c, the
// dither bits b by B) into a fourth accumulator, and xa by B into the wrap term's, in a segment where a lane has an xa.
// DG (kh_lanes): the units are the diagonal of gathered A [M, K] and B [M, K], unit u rows 32 u .. of both (N unused),
// each writing its lane record (OneOut::lr) beside its receipt; no other instantiation writes one.
// RW (kh_units_multi_j10r): the fused receipts-only unit under J10-R's noise (h1tile.j10r_noise): seeds' first 8 words are
// the call's seed_c, row 32 ri + lane's blocks come from the call's row workspace ws (h1jr_rows: block t at 16 t words)
// and column 32 cj + lane's blocks, M RB + (32 cj + lane) CB + c, are hashed by the lane; the fold keys stay sigma_R's.
// Its A and B loads run H1U_JPD segments ahead in H1U_JPD + H1U_JRING sets, RW's own H1U_PD and H1U_RING: it hashes one
// block an iteration where h1um hashes two, so its loads have half the time to land.
// h1u_unit is one warp's unit u (u < units) with its shared memory S; h1u runs unit blockIdx.x WARPS + warp of one call,
// h1um (kh_units_multi) unit u of the call its warp's index falls in.
#ifndef H1U_JPD
#define H1U_JPD 3
#endif
#ifndef H1U_JRING
#define H1U_JRING 1
#endif
static_assert(H1U_JPD >= 1 && (H1U_JRING == 0 || H1U_JRING == 1) && 16 % (H1U_JPD + H1U_JRING) == 0,
              "a chunk's 16 segments are whole rings of H1U_JPD (+ 1 with H1U_JRING) sets");
// H1U_JPI = 1 (RW, with H1U_PF, H1U_JRING and a ring of 4 or 8 sets): the ring's loads and L2 prefetches at immediate
// offsets from row pointers advanced once a ring (the segments run in order across the chunks), where 0 computes every
// row's address each segment; the same loads
#ifndef H1U_JPI
#define H1U_JPI 0
#endif
// RC (with RW; kh_units_multi_j10rc, built with H1U_JRC = 1): the column tiles come from the call's workspace too, at
// their stream positions M RB + (32 cj + lane) CB + c (h1jr_cols hashes each once a call), so the unit hashes no block
// and reads no seed_c. At prefill's m a column tile is every row tile's (64 at M = 2048), and RW hashes it in each of
// them; at decode's one live row tile it is hashed once either way
#ifndef H1U_JRC
#define H1U_JRC 0
#endif
// H1U_JRCPF = 1 (with RC): each iteration's L2 prefetch of the next one's row and column tiles, which RC's unit reads
// with no hash to wait behind
#ifndef H1U_JRCPF
#define H1U_JRCPF 0
#endif
// H1U_JPROBE = 1 (RW without RC): a timing probe, never a receipt: the column tiles are the row tile's words, not
// hashed, so a unit's time less its column hash bounds what sharing the column tiles can save (unitsbench --probe).
// With JX, which reads no row tile, they are a multiply of their stream position into the midstate's words. Bits 2
// and 4 (any RW unit) hold B's and A's (X's) segment loads at their first segments, which stay in L1, so the probe's
// time against the unit's bounds what those loads' L2 latency costs
#ifndef H1U_JPROBE
#define H1U_JPROBE 0
#endif
// H1U_JAP = 1 (RW, with or without RC): h1jr_rows also writes each live row's A' = A + its dither bits (bytewise mod
// 2^8, the add01 the unit does a segment) into the call's workspace after its blocks (jap_words), and the units read A'
// for A, so a segment adds no dither and an iteration reads no row tile but the coin block (it 0)
#ifndef H1U_JAP
#define H1U_JAP 0
#endif
static_assert(!(H1U_JAP && H1U_JPROBE), "the probe's column blocks are its row blocks, which H1U_JAP's units don't read");
// H1U_JAY = 1 (with H1U_JAP): Y, A' times P (row i's rho-th sum over the segments, sum_sg A'[i][32 sg + rho]), is a
// row's alone, so h1jr_y sums it once a row into the workspace after A' (jay_words) and the units read it at the tail
// instead of accumulating it a segment: 8 of a segment's 16 MMAs and Y's 32 registers gone
#ifndef H1U_JAY
#define H1U_JAY 0
#endif
static_assert(!H1U_JAY || H1U_JAP, "H1U_JAY sums A', which H1U_JAP writes");
// H1U_JM2 = 1 (with H1U_JRC and H1U_JAY): h1umjc's and h1umjc_sm's warps take a column tile's row tiles two at a time
// (units u and u + N / 32, u's row tile even) at an M of whole pairs, sharing their column side: B's loads, its
// fragments (B + c + U) and the column tiles; each row tile keeps its own receipt state, accumulators, ring and coins
// (h1u_pair, two WarpSmem a warp)
#ifndef H1U_JM2
#define H1U_JM2 0
#endif
static_assert(!H1U_JM2 || (H1U_JRC && H1U_JAY), "H1U_JM2 pairs RC's H1U_JAY units");
// a call's A' (H1U_JAP: M rows of K bytes) in its workspace, in words: after its row tiles (h1umj's) or its row and
// column tiles (rc, h1umjc's); then its Y (H1U_JAY: M rows of 32 int32)
__host__ __device__ __forceinline__ u64 jap_words(u32 M, u32 N, u32 K, bool rc) {
  return 16 * ((u64)M + (rc ? N : 0)) * (K / 512 + 1);
}
__host__ __device__ __forceinline__ u64 jay_words(u32 M, u32 N, u32 K, bool rc) {
  return jap_words(M, N, K, rc) + (u64)M * K / 4;
}
// RY (with RC; kh_units_multi_j10rc built with H1U_JRY = 1 too): Y and the coins are the row tile's alone, so its tail
// digits ad are computed once a call (h1jr_ady, h1u_unit's YO: the row tile's segments with no B, no T and no receipt)
// into the call's workspace after its column tiles, and its units read them: no Y mma in their segments
#ifndef H1U_JRY
#define H1U_JRY 0
#endif
#if H1U_JRY && !H1U_JRC
#error "H1U_JRY is H1U_JRC's units with their row blocks' digits read: build it with H1U_JRC = 1"
#endif
static_assert(!(H1U_JAP && H1U_JRY), "H1U_JAP's A' and H1U_JRY's digits both sit after the column blocks");
// JX (with RW, not RC; kh_units_multi_j10r and _sm built with H1U_JX = 1): a row tile's X = A + b, its Y = X P and its
// digits A_d are the same in every unit of the row tile, so h1jr_x computes them once a call, in the units' launch after
// h1jr_rows, into the call's row workspace after its blocks (X [M, K] s8 at jx_x_off, each live row tile's A_d fragments
// at jx_ad_off: JX_AD_WORDS words [d][4 mt + r][lane]); the unit loads X in A's place, runs no Y MMAs and reads no row
// block, and its tail loads A_d
#ifndef H1U_JX
#define H1U_JX 0
#endif
// H1U_JBW = 1 (RW): W' = B + c + U built as ((B & 7F) + (U & 7F) + c) ^ (B & 80) ^ (~U & 80) bytewise, U's two masks
// kept from the first pass (4 ALU instructions a word where addw over U + c takes 6); the same bytes
#ifndef H1U_JBW
#define H1U_JBW 0
#endif
// H1U_JN2 = 1 (with JX, which leaves rowbits free): two column tiles a pass (noise_block2, two independent SHA-512 chains
// for a scheduler with two warps), the coins' block with block 0 and odd block b with b + 1, b + 1 into rowbits for the
// next pass's segments; the same blocks
#ifndef H1U_JN2
#define H1U_JN2 0
#endif
// H1U_JXS = 1 (with JX): h1jr_x stores a row tile's X in A-fragment order, segment sg's row tile mt at byte
// 1024 sg + 512 mt of the block, lane (g, q)'s 16 bytes {rows g, g + 8} x {k 8 q .., 8 q + 4 ..} in mma_s8's register
// order, so the unit loads each fragment as one 16-byte load into the registers the MMA reads; the same bytes
#ifndef H1U_JXS
#define H1U_JXS 0
#endif
#if H1U_JXS && !H1U_JX
#error "H1U_JXS swizzles JX's X, so it needs H1U_JX = 1"
#endif
// H1U_JCS = G (2 or 4; with JX and H1U_JCTA, not JN2 or H1U_JPROBE & 1): a column tile's blocks are its seed_c
// stream's, the same in every row tile's unit, so h1umj_sm's CTA takes its WARPS units as a tile of r live row tiles
// by WARPS / r column tiles (r = G, else 2, else 1: the most that divides the live row tiles), the live tiles first,
// and a column tile's r warps share its blocks: each hashes 32 / r of its columns for r blocks a pass (sets of r
// blocks, one set ahead, double-buffered in the CTA's shared memory) and they meet once a set. r = 1 is the units in
// their own order, each warp hashing its own blocks; the same blocks and receipts
#ifndef H1U_JCS
#define H1U_JCS 0
#endif
#if H1U_JCS && (!H1U_JX || H1U_JN2 || (H1U_JPROBE & 1) || (H1U_JCS != 2 && H1U_JCS != 4))
#error "H1U_JCS (2 or 4) shares JX's column blocks, which JN2 and the hash-free probe don't hash one at a time"
#endif
// a JCS unit's column-block group: its 2 r buffers (set s's block e at s % 2 * r + e), its rank, r = 1 << lr, and its
// named barrier
struct JShare {
  u32 (*buf)[32][17];
  u32 rank, lr, bar;
};
struct alignas(16) WarpSmemJC {
  static constexpr bool RX = false;
  NoiseMid nm;
  u32 utab[32][9];
  u32 ring[32][33];
};
// immediate barrier ids and counts: a register id has ptxas reserve all 16 of the CTA's barriers. barrier.sync, not
// bar.sync (barrier.sync.aligned, undefined for a warp that reaches it diverged): a warp can come to it straight from
// cs_hash's per-lane store
static_assert(WARPS <= 8, "js_sync names barriers 1 .. WARPS / 2");
__device__ __forceinline__ void js_sync(const JShare& js) {
  if (js.lr == 0) {
    __syncwarp();
  } else if (js.lr == 2) {
    if (js.bar == 1) asm volatile("barrier.sync 1, 128;" ::: "memory");
    else asm volatile("barrier.sync 2, 128;" ::: "memory");
  } else {
    switch (js.bar) {
      case 1: asm volatile("barrier.sync 1, 64;" ::: "memory"); break;
      case 2: asm volatile("barrier.sync 2, 64;" ::: "memory"); break;
      case 3: asm volatile("barrier.sync 3, 64;" ::: "memory"); break;
      default: asm volatile("barrier.sync 4, 64;" ::: "memory"); break;
    }
  }
}
// JX already does JAP and JAY's work: h1jr_x writes X = A + b (JAP's A', at the same offset) and A_d from Y (JAY's
// sums), so a JX unit (JCS's too) has no dither add, Y MMA or Y tail left for them to remove
static_assert(!(H1U_JAP && H1U_JX), "H1U_JX's X is H1U_JAP's A' and its A_d covers H1U_JAY's Y: build one of them");
constexpr u32 JX_AD_WORDS = 3 * 8 * 32;
__host__ __device__ __forceinline__ u64 jx_x_off(u32 M, u32 K) { return 64ull * M * (K / 512 + 1); }
__host__ __device__ __forceinline__ u64 jx_ad_off(u32 M, u32 K) { return jx_x_off(M, K) + (u64)M * K; }
// H1U_JW = 1 (with JX, not JCS, JN2, H1U_JPROBE or H1U_JRC): W' = B + c + U is a column tile's alone, as X is a row
// block's, so h1jr_wcols hashes each call's column tiles once a call into its workspace after JX's (jw_cb_off: column
// j's block b at 64 (j CB + b) bytes) and h1jr_w writes W' [N, K] s8 after them (jw_w_off), in the units' launch after
// h1jr_x; the unit loads W' in B's place, hashes no block, builds no W' and reads its columns' coins blocks for U (its
// tail's), and h1umj's and h1umj_sm's units go H1U_JWSWZ row tiles at a time, column tile by column tile, so a CTA's
// warps share a column tile's W' and a band's X and W' stay in L2; the same receipts. The workspace is
// kh_j10r_jw_bytes(M, N, K), not kh_j10r_rows_bytes (kh_j10r_workspace_bytes gives it from kh_j10r_flags)
#ifndef H1U_JW
#define H1U_JW 0
#endif
#if H1U_JW && (!H1U_JX || H1U_JCS || H1U_JN2 || H1U_JPROBE || H1U_JRC)
#error "H1U_JW's units read JX's X beside its W' (not RC's workspace), and hash no column block for JCS, JN2 or the probe"
#endif
#ifndef H1U_JWSWZ
#define H1U_JWSWZ 32
#endif
// H1U_JWS = 1 (with JW): h1jr_w stores a column tile's W' in B-fragment order, as JXS stores X in A's: segment sg's
// column tiles 2 p and 2 p + 1 at byte 1024 sg + 512 p of the block, lane (g, q)'s 16 bytes the 8 at k = 8 q of
// column 16 p + g, then of column 16 p + 8 + g, so the unit loads two tiles' fragments as one 16-byte load
#ifndef H1U_JWS
#define H1U_JWS 0
#endif
#if H1U_JWS && !H1U_JW
#error "H1U_JWS swizzles JW's W', so it needs H1U_JW = 1"
#endif
// H1U_JPC = 1 (with JXS or JWS): h1jr_x (JXS) and h1jr_w (JWS) go over their output in fragment order, a thread a lane's
// 16 bytes (h1jr_x: a warp a segment's row tile, its rows' residue sums met in shared memory), so a warp stores 512
// contiguous bytes at a time; the same bytes
#ifndef H1U_JPC
#define H1U_JPC 0
#endif
#if H1U_JPC && !H1U_JXS && !H1U_JWS
#error "H1U_JPC orders JXS's and JWS's stores: build it with one of them"
#endif
__host__ __device__ __forceinline__ u64 jw_cb_off(u32 M, u32 K) { return jx_ad_off(M, K) + 4ull * (M / 32) * JX_AD_WORDS; }
__host__ __device__ __forceinline__ u64 jw_w_off(u32 M, u32 N, u32 K) { return jw_cb_off(M, K) + 64ull * N * (K / 512 + 1); }

// lane (g, q)'s digits A_d from its Y (the accumulator layout: Y[4 mt + nt][e] is row 16 mt + g + 8 (e >> 1), rho
// 8 nt + 2 q + (e & 1)) and its rows' coins (coin(row, d): bit rho is z_d(row, rho)), packed as A fragments
template <int MD, class Coin>
__device__ __forceinline__ void pack_digits(const int (&Y)[8][4], u32 g, u32 q, Coin coin, u32 (&ad)[MD][2][4]) {
#pragma unroll
  for (int d = 0; d < MD; ++d)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int r = 0; r < 4; ++r) ad[d][mt][r] = 0;
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    const int mt = t >> 2, nt = t & 3;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const u32 row = 16 * mt + g + 8 * (e >> 1), rho = 8 * nt + 2 * q + (e & 1);
      int yv = Y[t][e];
      const int reg = (e >> 1) + 2 * (nt >> 1), sh8 = 8 * (2 * (nt & 1) + (e & 1));
#pragma unroll
      for (int d = 0; d < MD - 1; ++d) {
        const int qv = yv >> 7;
        u32 zd = (coin(row, d) >> rho) & 1u;
        if (d == MD - 2 && qv == 127) zd = 0;
        ad[d][mt][reg] |= (((u32)yv & 127u) | (zd << 7)) << sh8;
        yv = qv + (int)zd;
      }
      ad[MD - 1][mt][reg] |= ((u32)yv & 0xFFu) << sh8;
    }
  }
}
template <int MD, bool WS, bool OS, bool J10 = false, bool PL = false, bool DG = false, bool RW = false, bool RC = false,
          bool RY = false, bool YO = false, bool JX = false, class Sm = WarpSmem, bool CS = false>
__device__ __forceinline__ void h1u_unit(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 M, u32 N, u32 K,
                                         const u64* __restrict__ seeds, const uint8_t* __restrict__ live,
                                         u32* __restrict__ out, const u32* __restrict__ ws, const OneOut oo, const u32 u,
                                         Sm& S, const JShare js = JShare{}) {
  constexpr bool RX = Sm::RX;
  static_assert(!RX || (RC && !DG), "WarpSmemX is RC's units' (no seed stream, no ring)");
  static_assert(!DG || !(WS || OS || J10 || PL), "a gathered diagonal is the fused receipts-only form");
  static_assert(!RW || !(WS || OS || J10 || PL || DG), "J10-R's units are the fused receipts-only form");
  static_assert(!RC || RW, "RC is J10-R's units with their column blocks read");
  static_assert(!RY || RC, "RY is RC's units with their row blocks' digits read");
  static_assert(!YO || RY, "YO is RY's row block digits");
  static_assert(!JX || (RW && !RC), "JX is J10-R's units reading their row block's X and A_d");
  static_assert(!CS || (JX && !H1U_JN2 && !(H1U_JPROBE & 1)), "CS shares JX's column blocks (H1U_JCS)");
  constexpr bool BW = RW && H1U_JBW;
  constexpr bool N2 = JX && H1U_JN2;
  constexpr bool XS = JX && H1U_JXS;
  constexpr bool JW = JX && H1U_JW;
  constexpr bool WS8 = JW && H1U_JWS;
  static_assert(!JW || !CS, "H1U_JW's units hash no column block to share");
  constexpr bool PF = WS || H1U_PF;
  constexpr bool AP = RW && H1U_JAP;  // A is the call's A' (H1U_JAP)
  constexpr bool AY = RW && H1U_JAY;  // Y comes from the workspace (H1U_JAY)
  constexpr bool RING = RW ? H1U_JRING : H1U_RING;
  constexpr int PD = RW ? H1U_JPD : H1U_PD, NS = PD + RING;
  constexpr bool PI = RW && H1U_JPI;
  static_assert(!PI || (PF && RING && NS % 4 == 0), "H1U_JPI is the prefetch ring's, in whole groups of 4 segments");
  constexpr bool PBK = RW && (H1U_JPROBE & 2), PAK = RW && (H1U_JPROBE & 4);
  const u32 lane = threadIdx.x & 31;
  const u32 g = lane >> 2, q = lane & 3;
  const u32 ncb = DG ? 1 : N / 32;
  const u32 ri = u / ncb, cj = DG ? u : u % ncb;
  const u32 ks = K / 32, chunks = ks / 16, rb = chunks + 1, cb = chunks + 1;
  const u32 nwords = 2 * (ks + MD);
  u32* o = out + (u64)u * nwords;
  int alive = 0, flags = 0;
  if (OS) {
    const int4 st = *reinterpret_cast<const int4*>(oo.step);
    alive = st.y;
    flags = st.w;
  }
  const bool rcpt = !OS || !(flags & F_NW_UNITS);
  if (live && !live[ri]) {
    if constexpr (YO) return;
    if (rcpt)
      for (u32 w = lane; w < nwords; w += 32) o[w] = 0;
    if (OS) {
      const u32 row = 32 * ri + lane;
      if (!(flags & F_NW_TILES)) {
        uint4* zr = reinterpret_cast<uint4*>(oo.z + (u64)row * N + 32 * cj);
#pragma unroll
        for (int i = 0; i < 8; ++i) zr[i] = make_uint4(0, 0, 0, 0);
      }
      if (row < oo.M) {
        uint4* yr = reinterpret_cast<uint4*>(oo.y + (u64)row * N + 32 * cj);
#pragma unroll
        for (int i = 0; i < 4; ++i) yr[i] = make_uint4(0, 0, 0, 0);
        if (oo.zo) {
          uint4* zr = reinterpret_cast<uint4*>(oo.zo + (u64)row * N + 32 * cj);
#pragma unroll
          for (int i = 0; i < 8; ++i) zr[i] = make_uint4(0, 0, 0, 0);
        }
        if ((flags & F_DUMP) && (int)row < alive && oo.ydump) {
          uint4* dr = reinterpret_cast<uint4*>(oo.ydump + (u64)row * N + 32 * cj);
#pragma unroll
          for (int i = 0; i < 4; ++i) dr[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }
    return;
  }
  u32* const lw = DG ? oo.lr + (u64)u * 32 * nwords : nullptr;
  const u64* sd = seeds + 16 * (u64)u;

  u32 y[8], z[8];
  if constexpr (!YO) {
    u64 msg[16] = {TAG0, TAG1, TAG2};
#pragma unroll
    for (int j = 0; j < 8; ++j) msg[3 + j] = __ldg(sd + 8 + j);
    msg[11] = lane;
    msg[12] = 0x8000000000000000ULL;
    msg[13] = 0;
    msg[14] = 0;
    msg[15] = 96 * 8;
    u64 s[8];
    sha512_words(msg, 1, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      y[2 * j] = (u32)(s[j] >> 32);
      y[2 * j + 1] = (u32)s[j];
      z[2 * j] = (u32)(s[4 + j] >> 32);
      z[2 * j + 1] = (u32)s[4 + j];
    }
  }
  if constexpr (!WS && !RC && !JW) {
    u64 seed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) seed[j] = __ldg(sd + j);
    NoiseMid nm;
    noise_mid(seed, nm);
    if (lane == 0) S.nm = nm;
  }
  __syncwarp();
  const u32* wsu = WS ? (J10 ? ws : ws + 16ull * 32 * (rb + cb) * u) : nullptr;
  const u32* wsr = WS && !J10 ? ws + 16ull * 32 * (rb + cb) * (ri * ncb + cj / oo.band * oo.band) : wsu;

  // P: column n of n-tile nt is rho = 8 nt + n; fragment k = 4 q' + i is segment byte 8 q' + i, k = 16 + 4 q' + i is
  // 8 q' + 4 + i
  u32 idr[4][2];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    idr[nt][0] = (q == (u32)nt && g < 4) ? 1u << (8 * g) : 0u;
    idr[nt][1] = (q == (u32)nt && g >= 4) ? 1u << (8 * (g - 4)) : 0u;
  }
  int T[8][4], Y[8][4], Cw[OS ? 8 : 1][4], Pz[PL ? 8 : 1][4];
#pragma unroll
  for (int t = 0; t < 8; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) T[t][e] = Y[t][e] = 0;
  if constexpr (OS) {
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) Cw[t][e] = 0;
  }
  if constexpr (PL) {
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) Pz[t][e] = 0;
  }
  u32 ulo[4], uhi[4];
  u32 ulo7[BW ? 4 : 1], uhi7[BW ? 4 : 1], ulo8[BW ? 4 : 1], uhi8[BW ? 4 : 1];  // BW: U & 7F and ~U & 80

  const int8_t* Ab = JX ? reinterpret_cast<const int8_t*>(ws) + jx_x_off(M, K) + (u64)(32 * ri) * K
                        : A + (u64)(32 * ri) * K;
  const int8_t* Bb = JW ? reinterpret_cast<const int8_t*>(ws) + jw_w_off(M, N, K) + (u64)(32 * cj) * K
                        : B + (u64)(32 * cj) * K;
  // segment sg's A rows 16 mt + g + 8 h and B columns 8 nt + g, 8 bytes each at k = 32 sg + 8 q, in ring slot sg % NS
  uint2 pa[NS][4], pb[NS][4];
  const auto load_seg = [&](u32 sg, uint2(&a)[4], uint2(&b)[4]) {
    const u32 ka = PAK ? 8 * q : 32 * sg + 8 * q, kb = PBK ? 8 * q : 32 * sg + 8 * q;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if constexpr (XS) {
        if (r % 2 == 0) {
          const uint4 v = ld4<H1U_L2HINT>(Ab + 1024ull * (PAK ? 0 : sg) + 256 * r + 16 * lane);
          a[r] = make_uint2(v.x, v.z);
          a[r + 1] = make_uint2(v.y, v.w);
        }
      } else {
        a[r] = ld2<H1U_L2HINT>(Ab + (u64)(16 * (r >> 1) + g + 8 * (r & 1)) * K + ka);
      }
      if constexpr (WS8 && !YO) {
        if (r % 2 == 0) {
          const uint4 v = ld4<H1U_L2HINT>(Bb + 1024ull * (PBK ? 0 : sg) + 256 * r + 16 * lane);
          b[r] = make_uint2(v.x, v.y);
          b[r + 1] = make_uint2(v.z, v.w);
        }
      } else if constexpr (!YO && !WS8) {
        b[r] = ld2e<H1U_L2HINT, H1U_EVF && H1U_EVFU>(Bb + (u64)(8 * r + g) * K + kb);
      }
    }
  };
  // PI: the rows' pointers at segment PD of the ring pass to come (A's one under XS), the L2 prefetches' at segment 16
  const int8_t* par[PI && !XS ? 4 : 1];
  const int8_t* pbr[PI && !WS8 ? 4 : 1];
  const int8_t *pfa = nullptr, *pfb = nullptr;
  constexpr u32 ASEG = XS ? 1024 : 32, BSEG = WS8 ? 1024 : 32;
  if constexpr (PI) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if constexpr (!XS) par[r] = Ab + (u64)(16 * (r >> 1) + g + 8 * (r & 1)) * K + 8 * q + 32 * PD;
      if constexpr (!WS8) pbr[r] = Bb + (u64)(8 * r + g) * K + 8 * q + 32 * PD;
    }
    if constexpr (XS) par[0] = Ab + 1024 * PD + 16 * lane;
    if constexpr (WS8) pbr[0] = Bb + 1024 * PD + 16 * lane;
    pfa = XS ? Ab + 1024 * 16 + 128 * lane : Ab + (u64)lane * K + 32 * 16;
    pfb = WS8 ? Bb + 1024 * 16 + 128 * lane : Bb + (u64)lane * K + 32 * 16;
  }
  // PI: the segment jo past the pointers'
  const auto load_at = [&](int jo, uint2(&a)[4], uint2(&b)[4]) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if constexpr (XS) {
        if (r % 2 == 0) {
          const uint4 v = ld4<H1U_L2HINT>(par[0] + 1024 * jo + 256 * r);
          a[r] = make_uint2(v.x, v.z);
          a[r + 1] = make_uint2(v.y, v.w);
        }
      } else {
        a[r] = ld2<H1U_L2HINT>(par[r] + 32 * jo);
      }
      if constexpr (WS8) {
        if (r % 2 == 0) {
          const uint4 v = ld4<H1U_L2HINT>(pbr[0] + 1024 * jo + 256 * r);
          b[r] = make_uint2(v.x, v.y);
          b[r + 1] = make_uint2(v.z, v.w);
        }
      } else {
        b[r] = ld2e<H1U_L2HINT, H1U_EVF && H1U_EVFU>(pbr[r] + 32 * jo);
      }
    }
  };
  if (PF) {
#pragma unroll
    for (int d = 0; d < PD; ++d) load_seg(d, pa[d], pb[d]);
  }
  // CS: set st's blocks i = r st + e (pass i's, i = 0 the coins' block), lane's e and column (32 / r of them a warp);
  // its body under CS, since the Sm variants (WarpSmemX, JcSmem) have no nm
  const auto cs_hash = [&](u32 st) {
    if constexpr (CS) {
      const u32 cl = 5 - js.lr, e = lane >> cl, col = (js.rank << cl) + (lane & ((1u << cl) - 1)), i = (st << js.lr) + e;
      u64 s[8];
      noise_block(S.nm, (u64)M * rb + (u64)(32 * cj + col) * cb + (i == 0 ? chunks : i - 1), s);
      if (i <= chunks) {
        u32* d = js.buf[((st & 1) << js.lr) + e][col];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          d[2 * j] = (u32)(s[j] >> 32);
          d[2 * j + 1] = (u32)s[j];
        }
      }
    }
  };
#pragma unroll 1
  for (u32 it = 0; it <= chunks; ++it) {
    const u32 blk = it == 0 ? chunks : it - 1;
    const u32(*cw32)[17];
    if constexpr (CS) {
      // set 0 at it = 0 in the same loop as the sets ahead, so the unit inlines one SHA-512
      if ((it & ((1u << js.lr) - 1)) == 0) {
#pragma unroll 1
        for (u32 h = it == 0 ? 0 : 1; h < 2; ++h) {
          if (h) js_sync(js);
          const u32 st = (it >> js.lr) + h;
          if ((st << js.lr) <= chunks) cs_hash(st);
        }
      }
      cw32 = js.buf[(((it >> js.lr) & 1) << js.lr) + (it & ((1u << js.lr) - 1))];
    } else {
      cw32 = S.colbits;
    }
    if constexpr (CS) {
    } else if constexpr (JW) {
      if (it == 0) {
        const uint4* src = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(ws) + jw_cb_off(M, K)) +
                           4 * ((u64)(32 * cj + lane) * cb + chunks);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const uint4 v = ld4<H1U_L2HINT_WS>(src + w);
          S.colbits[lane][4 * w] = v.x;
          S.colbits[lane][4 * w + 1] = v.y;
          S.colbits[lane][4 * w + 2] = v.z;
          S.colbits[lane][4 * w + 3] = v.w;
        }
      }
    } else if constexpr (N2) {
      const u64 t0 = (u64)M * rb + (u64)(32 * cj + lane) * cb;
      if (it == 0 || (blk % 2 == 1 && blk + 1 < chunks)) {
        u64 sa[8], sb[8];
        noise_block2(S.nm, t0 + blk, t0 + (it == 0 ? 0 : blk + 1), sa, sb);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          S.colbits[lane][2 * j] = (u32)(sa[j] >> 32);
          S.colbits[lane][2 * j + 1] = (u32)sa[j];
          S.rowbits[lane][2 * j] = (u32)(sb[j] >> 32);
          S.rowbits[lane][2 * j + 1] = (u32)sb[j];
        }
      } else if (blk % 2 == 1) {
        u64 s[8];
        noise_block(S.nm, t0 + blk, s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          S.colbits[lane][2 * j] = (u32)(s[j] >> 32);
          S.colbits[lane][2 * j + 1] = (u32)s[j];
        }
      }
    } else if constexpr (JX) {
      u64 s[8];
      if constexpr (H1U_JPROBE & 1) {
        const u64 x = ((u64)(32 * cj + lane) * cb + blk) * 0x9E3779B97F4A7C15ull;
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j] = x ^ S.nm.mid[j];
      } else {
        noise_block(S.nm, (u64)M * rb + (u64)(32 * cj + lane) * cb + blk, s);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        S.colbits[lane][2 * j] = (u32)(s[j] >> 32);
        S.colbits[lane][2 * j + 1] = (u32)s[j];
      }
    } else if constexpr (RW) {
      const bool rows = !AP || it == 0;
      const uint4* src = reinterpret_cast<const uint4*>(ws + 16 * ((u64)(32 * ri + lane) * rb + blk));
      uint4 rv[4];
      if (rows) {
#pragma unroll
        for (int w = 0; w < 4; ++w) rv[w] = ld4<H1U_L2HINT_WS>(src + w);
      }
      if constexpr (RC) {
        const uint4* csrc =
            reinterpret_cast<const uint4*>(ws + 16 * ((u64)M * rb + (u64)(32 * cj + lane) * cb + blk));
        uint4 cv[4];
        if constexpr (!YO) {
#pragma unroll
          for (int w = 0; w < 4; ++w) cv[w] = ld4<H1U_L2HINT_WS>(csrc + w);
          if (H1U_JRCPF && it < chunks) {
            if (!AP) prefetch_l2(ws + 16 * ((u64)(32 * ri + lane) * rb + it));
            prefetch_l2(ws + 16 * ((u64)M * rb + (u64)(32 * cj + lane) * cb + it));
          }
        }
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          if (rows) {
            S.rowbits[lane][4 * w] = rv[w].x;
            S.rowbits[lane][4 * w + 1] = rv[w].y;
            S.rowbits[lane][4 * w + 2] = rv[w].z;
            S.rowbits[lane][4 * w + 3] = rv[w].w;
          }
          if constexpr (!YO) {
            S.colbits[lane][4 * w] = cv[w].x;
            S.colbits[lane][4 * w + 1] = cv[w].y;
            S.colbits[lane][4 * w + 2] = cv[w].z;
            S.colbits[lane][4 * w + 3] = cv[w].w;
          }
        }
      } else {
        u64 s[8];
        if constexpr (H1U_JPROBE & 1) {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            s[j] = ((u64)(j & 1 ? rv[j >> 1].z : rv[j >> 1].x) << 32 | (j & 1 ? rv[j >> 1].w : rv[j >> 1].y)) ^ (blk + cj);
        } else {
          noise_block(S.nm, (u64)M * rb + (u64)(32 * cj + lane) * cb + blk, s);
        }
        if (rows) {
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            S.rowbits[lane][4 * w] = rv[w].x;
            S.rowbits[lane][4 * w + 1] = rv[w].y;
            S.rowbits[lane][4 * w + 2] = rv[w].z;
            S.rowbits[lane][4 * w + 3] = rv[w].w;
          }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          S.colbits[lane][2 * j] = (u32)(s[j] >> 32);
          S.colbits[lane][2 * j + 1] = (u32)s[j];
        }
      }
    } else if constexpr (!WS && !J10 && H1U_NB2) {
      u64 sa[8], sb[8];
      noise_block2(S.nm, (u64)lane * rb + blk, 32ull * rb + (u64)lane * cb + blk, sa, sb);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        S.rowbits[lane][2 * j] = (u32)(sa[j] >> 32);
        S.rowbits[lane][2 * j + 1] = (u32)sa[j];
        S.colbits[lane][2 * j] = (u32)(sb[j] >> 32);
        S.colbits[lane][2 * j + 1] = (u32)sb[j];
      }
    } else {
#pragma unroll 1
      for (int jb = 0; jb < 2; ++jb) {
        const u64 t = J10 ? (jb == 0 ? (u64)(32 * ri + lane) * rb + blk : (u64)M * rb + (u64)(32 * cj + lane) * cb + blk)
                          : (jb == 0 ? (u64)lane * rb + blk : 32ull * rb + (u64)lane * cb + blk);
        u32* dst = jb == 0 ? S.rowbits[lane] : S.colbits[lane];
        if (WS) {
          const uint4* src = reinterpret_cast<const uint4*>((jb == 0 ? wsr : wsu) + 16 * t);
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            const uint4 v = ld4e<H1U_L2HINT_WS, H1U_EVF>(src + w);
            dst[4 * w] = v.x;
            dst[4 * w + 1] = v.y;
            dst[4 * w + 2] = v.z;
            dst[4 * w + 3] = v.w;
          }
        } else {
          u64 s[8];
          noise_block(S.nm, t, s);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            dst[2 * j] = (u32)(s[j] >> 32);
            dst[2 * j + 1] = (u32)s[j];
          }
        }
      }
    }
    if (WS && H1U_L2PF && it < chunks) {
      if (J10) {
        prefetch_l2(wsu + 16 * ((u64)(32 * ri + lane) * rb + it));
        prefetch_l2(wsu + 16 * ((u64)M * rb + (u64)(32 * cj + lane) * cb + it));
      } else {
        prefetch_l2(wsr + 16 * ((u64)lane * rb + it));
        prefetch_l2(wsu + 16 * (32ull * rb + (u64)lane * cb + it));
      }
    }
    __syncwarp();
    if (it == 0) {
      if constexpr ((!RY || YO) && !JX) {
#pragma unroll
        for (int d = 0; d < MD - 1; ++d) S.coin[lane][d] = S.rowbits[lane][d];
      }
      if constexpr (!YO) {
#pragma unroll
        for (int h = 0; h < 8; ++h) {
          const u32 hv = cw32[lane][h];
          u32 w = 0;
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const u32 x = (hv >> (8 * b)) & 0xFFu;
            w |= ((x >= 129 ? x - 129 : x) + 64) << (8 * b);
          }
          S.utab[lane][h] = w;
        }
        __syncwarp();
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          ulo[nt] = S.utab[8 * nt + g][2 * q];
          uhi[nt] = S.utab[8 * nt + g][2 * q + 1];
          if constexpr (BW) {
            ulo7[nt] = ulo[nt] & 0x7F7F7F7Fu;
            uhi7[nt] = uhi[nt] & 0x7F7F7F7Fu;
            ulo8[nt] = ~ulo[nt] & 0x80808080u;
            uhi8[nt] = ~uhi[nt] & 0x80808080u;
          }
        }
      }
      continue;
    }
    const uint8_t* rbits = nullptr;
    const uint8_t* cbits = reinterpret_cast<const uint8_t*>(&cw32[0][0]);
    if constexpr (!JX || N2) {
      rbits = reinterpret_cast<const uint8_t*>(&S.rowbits[0][0]);
      if (N2 && blk % 2 == 0) cbits = rbits;
    }
    u32 rw = 0;  // RX: the lane's word of the chunk (red_put)
#pragma unroll 1
    for (u32 j0 = 0; j0 < 16; j0 += NS) {
#pragma unroll
      for (int jj = 0; jj < NS; ++jj) {
        const u32 j = j0 + jj, sg = 16 * blk + j;
        uint2 ca[4], cw[4];
        if (H1U_L2PF && (PI ? jj % 4 == 0 : sg % 4 == 0) && sg + 16 < ks) {
          if constexpr (PI) {
            prefetch_l2(pfa + ASEG * jj);
            prefetch_l2(pfb + BSEG * jj);
          } else {
            prefetch_l2(XS ? Ab + 1024ull * (sg + 16) + 128 * lane : Ab + (u64)lane * K + 32 * (sg + 16));
            prefetch_l2(WS8 ? Bb + 1024ull * (sg + 16) + 128 * lane : Bb + (u64)lane * K + 32 * (sg + 16));
          }
        }
        if (PF && RING) {
          if (sg + PD < ks) {
            if constexpr (PI) load_at(jj, pa[(jj + PD) % NS], pb[(jj + PD) % NS]);
            else load_seg(sg + PD, pa[(jj + PD) % NS], pb[(jj + PD) % NS]);
          }
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            ca[r] = pa[jj][r];
            if constexpr (!YO) cw[r] = pb[jj][r];
          }
        } else if (PF) {
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            ca[r] = pa[jj][r];
            if constexpr (!YO) cw[r] = pb[jj][r];
          }
          if (sg + PD < ks) load_seg(sg + PD, pa[jj], pb[jj]);
        } else {
          load_seg(sg, ca, cw);
        }
        u32 af[2][4], bf[4][2];
        u32 bb[PL ? 2 : 1][4], xa[PL ? 2 : 1][4], xany = 0;  // PL: b and xa as 0/1 A fragments
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const u32 row = 16 * mt + g + 8 * h;
            const uint2 a8 = ca[2 * mt + h];
            if constexpr (AP || JX) {
              af[mt][h] = a8.x;
              af[mt][2 + h] = a8.y;
              continue;
            }
            const u32 nb8 = rbits[row * 68 + 4 * j + q];
            const u32 e0 = expand4(nb8 & 15u), e1 = expand4(nb8 >> 4);
            af[mt][h] = add01(a8.x, e0);
            af[mt][2 + h] = add01(a8.y, e1);
            if constexpr (PL) {
              bb[mt][h] = e0;
              bb[mt][2 + h] = e1;
              xa[mt][h] = __vcmpeq4(a8.x, 0x7F7F7F7Fu) & e0;
              xa[mt][2 + h] = __vcmpeq4(a8.y, 0x7F7F7F7Fu) & e1;
              xany |= xa[mt][h] | xa[mt][2 + h];
            }
          }
        }
        if constexpr (PL) {
          if ((flags & F_FAULT_XA) && u == 0 && lane == 0 && sg == 0) {
            xa[0][0] ^= 1u;
            xany = 1;
          }
        }
        u32 wbf[OS ? 4 : 1][2], cf[PL ? 4 : 1][2], b0[PL ? 4 : 1][2];  // PL: c as 0/1 B fragments, B itself
#pragma unroll
        for (int nt = 0; nt < (YO ? 0 : 4); ++nt) {
          const u32 col = 8 * nt + g;
          const uint2 b8 = cw[nt];
          if constexpr (JW) {
            bf[nt][0] = b8.x;
            bf[nt][1] = b8.y;
            continue;
          }
          const u32 cb8 = cbits[col * 68 + 4 * j + q];
          const u32 c0 = expand4(cb8 & 15u), c1 = expand4(cb8 >> 4);
          const u32 v0 = ulo[nt] + c0, v1 = uhi[nt] + c1;
          if constexpr (BW) {
            bf[nt][0] = (((b8.x & 0x7F7F7F7Fu) + ulo7[nt] + c0) ^ (b8.x & 0x80808080u)) ^ ulo8[nt];
            bf[nt][1] = (((b8.y & 0x7F7F7F7Fu) + uhi7[nt] + c1) ^ (b8.y & 0x80808080u)) ^ uhi8[nt];
          } else {
            bf[nt][0] = addw(b8.x, v0);
            bf[nt][1] = addw(b8.y, v1);
          }
          if constexpr (OS) {
            wbf[nt][0] = wrapw(b8.x, v0);
            wbf[nt][1] = wrapw(b8.y, v1);
          }
          if constexpr (PL) {
            cf[nt][0] = c0;
            cf[nt][1] = c1;
            b0[nt][0] = b8.x;
            b0[nt][1] = b8.y;
          }
        }
        if constexpr (OS) {
          if ((flags & F_FAULT_WB) && u == 0 && lane == 0 && sg == 0)
            wbf[0][0] = (wbf[0][0] & ~0xFFu) | ((wbf[0][0] & 0xFFu) == 0 ? 1u : 0u);
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            if constexpr (!YO) mma_s8(T[4 * mt + nt], af[mt], bf[nt]);
            if constexpr ((!RY || YO) && !AY && !JX) mma_s8(Y[4 * mt + nt], af[mt], idr[nt]);
            if constexpr (OS) mma_s8(Cw[4 * mt + nt], af[mt], wbf[nt]);
            if constexpr (PL) {
              mma_s8(Pz[4 * mt + nt], af[mt], cf[nt]);
              mma_s8(Pz[4 * mt + nt], bb[mt], b0[nt]);
            }
          }
        if constexpr (PL) {
          if (__any_sync(0xFFFFFFFFu, xany != 0)) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
              for (int nt = 0; nt < 4; ++nt) mma_s8(Cw[4 * mt + nt], xa[mt], b0[nt]);
          }
        }
        if constexpr (OS) {
          if ((flags & F_FAULT) && u == 0 && lane == 0 && sg == 0) T[0][0] = (int)((u32)T[0][0] + (1u << FAULT_SHIFT));
        }
        if constexpr (!YO) {
          fold(T, y, z);
          if constexpr (RX) red_put(rw, j, lane, y, z);
          else ring_put(S, j, lane, y, z);
        }
      }
      if constexpr (PI) {
#pragma unroll
        for (int r = 0; r < (XS ? 1 : 4); ++r) par[r] += PAK ? 0 : ASEG * NS;
#pragma unroll
        for (int r = 0; r < (WS8 ? 1 : 4); ++r) pbr[r] += PBK ? 0 : BSEG * NS;
        pfa += ASEG * NS;
        pfb += BSEG * NS;
      }
    }
    __syncwarp();
    if constexpr (RX) {
      if (!YO && rcpt) o[32 * blk + lane] = rw;
    } else {
      if (!YO && rcpt) o[32 * blk + lane] = ring_xor(S, lane);
    }
    if constexpr (DG) lane_put(S, lw, nwords, 32 * blk, lane);
    __syncwarp();
  }

  if constexpr (AY) {
    const int* ys = reinterpret_cast<const int*>(ws + jay_words(M, N, K, RC)) + 32ull * 32 * ri;
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        Y[t][e] = __ldg(ys + 32 * (16 * (t >> 2) + g + 8 * (e >> 1)) + 8 * (t & 3) + 2 * q + (e & 1));
  }
  // the tail: A_d packed as A fragments in the accumulators' column order (fragment k 4 q' .. of rows g / g + 8 is
  // columns 2 q', 2 q' + 1, 8 + 2 q', 9 + 2 q' of n-tiles 0, 1; k 16 + 4 q' .. those of n-tiles 2, 3)
  u32 ad[MD][2][4];
  // RY: row tile ri's digits in its call's workspace after the column tiles, word (8 d + 4 mt + r) 32 + lane of its MD 256
  const u32* const adr = RY ? ws + 16ull * ((u64)M + N) * rb + (u64)ri * MD * 256 + lane : nullptr;
  if constexpr (JX) {
    const u32* src = ws + jx_ad_off(M, K) / 4 + (u64)ri * JX_AD_WORDS;
#pragma unroll
    for (int d = 0; d < MD; ++d)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int r = 0; r < 4; ++r) ad[d][mt][r] = __ldg(src + 32 * (8 * d + 4 * mt + r) + lane);
  } else if constexpr (RY && !YO) {
#pragma unroll
    for (int d = 0; d < MD; ++d)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int r = 0; r < 4; ++r) ad[d][mt][r] = adr[32 * (8 * d + 4 * mt + r)];
  } else {
    pack_digits<MD>(Y, g, q, [&](u32 row, int d) { return S.coin[row][d]; }, ad);
    if constexpr (YO) {
      u32* const adw = const_cast<u32*>(adr);
#pragma unroll
      for (int d = 0; d < MD; ++d)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int r = 0; r < 4; ++r) adw[32 * (8 * d + 4 * mt + r)] = ad[d][mt][r];
      return;
    }
  }
  u32 uf[4][2], nf[4][2];
  {
    const uint8_t* ub = reinterpret_cast<const uint8_t*>(&S.utab[0][0]);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const uint8_t* r = ub + (8 * nt + g) * 36;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const u32 c0 = 16 * hf + 2 * q;
        const u32 by[4] = {r[c0], r[c0 + 1], r[c0 + 8], r[c0 + 9]};
        u32 pu = 0, pn = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          pu |= ((by[b] - 128u) & 0xFFu) << (8 * b);
          pn |= ((128u - by[b]) & 0xFFu) << (8 * b);
        }
        uf[nt][hf] = pu;
        nf[nt][hf] = pn;
      }
    }
  }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mma_s8(T[4 * mt + nt], ad[0][mt], nf[nt]);
  fold(T, y, z);
  u32 rw = 0;
  if constexpr (RX) red_put(rw, 0, lane, y, z);
  else ring_put(S, 0, lane, y, z);
#pragma unroll
  for (int d = 1; d < MD; ++d) {
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) Y[t][e] = 0;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) mma_s8(Y[4 * mt + nt], ad[d][mt], uf[nt]);
    fold(Y, y, z);
    if constexpr (RX) red_put(rw, d, lane, y, z);
    else ring_put(S, d, lane, y, z);
    if constexpr (OS) {
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) T[t][e] = (int)((u32)T[t][e] - ((u32)Y[t][e] << (7 * d)));
    }
  }
  __syncwarp();
  if constexpr (RX) {
    if (rcpt && lane < 2 * MD) o[2 * ks + lane] = rw;
  } else {
    if (rcpt && lane < 2 * MD) o[2 * ks + lane] = ring_xor(S, lane);
  }
  if constexpr (DG) lane_tail<MD>(S, lw, nwords, 2 * ks, lane);
  if constexpr (OS) {
    // T[4 mt + nt][e] is cell (16 mt + g + 8 (e >> 1), 8 nt + 2 q + (e & 1)): z there, then y
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const u32 row = 32 * ri + 16 * mt + g + 8 * eh;
        const float s_a = (int)row < alive ? oo.sa[row] : 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int t = 4 * mt + nt;
          const u32 col = 32 * cj + 8 * nt + 2 * q;
          int z0 = (int)((u32)T[t][2 * eh] + ((u32)Cw[t][2 * eh] << 8));
          int z1 = (int)((u32)T[t][2 * eh + 1] + ((u32)Cw[t][2 * eh + 1] << 8));
          if constexpr (PL) {
            z0 = (int)((u32)z0 - (u32)Pz[t][2 * eh]);
            z1 = (int)((u32)z1 - (u32)Pz[t][2 * eh + 1]);
          }
          if (!(flags & F_NW_TILES)) *reinterpret_cast<int2*>(oo.z + (u64)row * N + col) = make_int2(z0, z1);
          if (row < oo.M) {
            if (oo.zo) *reinterpret_cast<int2*>(oo.zo + (u64)row * N + col) = make_int2(z0, z1);
            __nv_bfloat162 yv;
            yv.x = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z0), s_a), oo.sw[col]));
            yv.y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z1), s_a), oo.sw[col + 1]));
            *reinterpret_cast<__nv_bfloat162*>(oo.y + (u64)row * N + col) = yv;
            if ((flags & F_DUMP) && (int)row < alive && oo.ydump)
              *reinterpret_cast<__nv_bfloat162*>(oo.ydump + (u64)row * N + col) = yv;
          }
        }
      }
  }
}

template <int MD, bool WS, bool OS, bool J10 = false, bool PL = false, bool DG = false>
__global__ void __launch_bounds__(32 * WARPS, H1U_MINB) h1u(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 M, u32 N,
                                                  u32 K, const u64* __restrict__ seeds, const uint8_t* __restrict__ live,
                                                  u32* __restrict__ out, const u32* __restrict__ ws, OneOut oo) {
  __shared__ WarpSmem smem[WARPS];
  const u32 wp = threadIdx.x >> 5;
  const u32 u = blockIdx.x * WARPS + wp;
  if (u >= (M / 32) * (DG ? 1 : N / 32)) return;
  h1u_unit<MD, WS, OS, J10, PL, DG>(A, B, M, N, K, seeds, live, out, ws, oo, u, smem[wp]);
}

// kh_units_multi's calls: call c's units are warps u0 .. u0 + (M / 32)(N / 32) - 1 of the launch, each h1u's fused
// receipts-only unit (WS, OS false: its noise hashed by its own lanes, its midstate once) of A [M, K] and B [N, K]
struct UCall {
  const int8_t* A;
  const int8_t* B;
  const u64* seeds;  // the call's unit 0's 16 words
  u32* out;          // the call's unit 0's receipt
  u32 N, K, u0, md;  // md = n_digits(K / 32), 2 or 3
};

// H1U_SWZ = G (h1um): a call's warps take its units G row tiles at a time, column tile by column tile, so the G row
// blocks of one column tile read its B lines from L2 together, where row-block-major order (0) reads all of B once a row
// block (prefill's M = 2048: 64 times a step, and gate_up's B is about L2's size). Each unit is still written at its own
// position: only which warp computes it moves.
#ifndef H1U_SWZ
#define H1U_SWZ 0
#endif
__device__ __forceinline__ u32 swz_unit(const UCall& c, u32 gu, u32 M) {
  const u32 w = gu - c.u0;
  if (!H1U_SWZ) return w;
  const u32 ncb = c.N / 32, band = (u32)H1U_SWZ * ncb, r0 = w / band * (u32)H1U_SWZ, r = w % band;
  const u32 rn = min((u32)H1U_SWZ, M / 32 - r0);
  return (r0 + r % rn) * ncb + r / rn;
}

// every unit of ncalls calls in one launch (calls[ncalls].u0 the total), one warp a unit as h1u; live is every call's
__global__ void __launch_bounds__(32 * WARPS, H1U_MINB) h1um(const UCall* __restrict__ calls, u32 ncalls, u32 M,
                                                   const uint8_t* __restrict__ live) {
  __shared__ WarpSmem smem[WARPS];
  const u32 wp = threadIdx.x >> 5;
  const u32 gu = blockIdx.x * WARPS + wp;
  if (gu >= calls[ncalls].u0) return;
  u32 lo = 0, hi = ncalls;
  while (hi - lo > 1) {
    const u32 mid = (lo + hi) >> 1;
    if (calls[mid].u0 <= gu) lo = mid;
    else hi = mid;
  }
  const UCall c = calls[lo];
  if (c.md == 3)
    h1u_unit<3, false, false>(c.A, c.B, M, c.N, c.K, c.seeds, live, c.out, nullptr, OneOut{}, swz_unit(c, gu, M), smem[wp]);
  else
    h1u_unit<2, false, false>(c.A, c.B, M, c.N, c.K, c.seeds, live, c.out, nullptr, OneOut{}, swz_unit(c, gu, M), smem[wp]);
}

// h1um's unit gu, for h1um_sm (h1um keeps its own copy: calling this one changes h1um's register allocation)
__device__ __forceinline__ void h1um_one(const UCall* __restrict__ calls, u32 ncalls, u32 M, const uint8_t* __restrict__ live,
                                         u32 gu, WarpSmem& S) {
  u32 lo = 0, hi = ncalls;
  while (hi - lo > 1) {
    const u32 mid = (lo + hi) >> 1;
    if (calls[mid].u0 <= gu) lo = mid;
    else hi = mid;
  }
  const UCall c = calls[lo];
  if (c.md == 3)
    h1u_unit<3, false, false>(c.A, c.B, M, c.N, c.K, c.seeds, live, c.out, nullptr, OneOut{}, gu - c.u0, S);
  else
    h1u_unit<2, false, false>(c.A, c.B, M, c.N, c.K, c.seeds, live, c.out, nullptr, OneOut{}, gu - c.u0, S);
}

// the _smc launches' per-SM CTA counts, after the unit counter in `next` (a %smid past them shares a count)
constexpr u32 SMC_SLOTS = 1024;
// the _sm launches' largest den: past every %smid of these GPUs, so q / SMQ_DEN_MAX keeps the SMs below q, one block
constexpr u32 SMQ_DEN_MAX = 256;

// whether this CTA takes units: on a kept SM (smid % den < q), and under CAP only the first cap CTAs to reach it
// (next[1 + smid]); off them only the grid's last CTA while no unit has been taken. ONE: thread 0 decides for the CTA,
// for one whose warps take units together: the last CTA's warps reading next apart could see a unit taken between their
// reads, and those that stayed would claim units for those that had left
template <bool CAP, bool ONE = false>
__device__ __forceinline__ bool sm_keep(u32 smid, u32 q, u32 den, u32 cap, u32* next) {
  if constexpr (CAP) {
    __shared__ u32 keep;
    if (threadIdx.x == 0)
      keep = (blockIdx.x + 1 == gridDim.x && !*(volatile u32*)next) ||
             (smid % den < q && atomicAdd(next + 1 + smid % SMC_SLOTS, 1u) < cap);
    __syncthreads();
    return keep;
  } else if constexpr (ONE) {
    __shared__ u32 keep;
    if (threadIdx.x == 0) keep = !(smid % den >= q && (blockIdx.x + 1 < gridDim.x || *(volatile u32*)next));
    __syncthreads();
    return keep;
  } else {
    return !(smid % den >= q && (blockIdx.x + 1 < gridDim.x || *(volatile u32*)next));
  }
}

// h1um on the SMs whose %smid % den < q alone: a CTA on any other SM exits at once, and the warps of the rest take units
// from *next in turn (a unit's receipts are its own, whichever warp takes it), so every other SM stays whole to the
// streams beside it (the next forward's kernels, which a freed h1um slot is too small for). The grid's last CTA exits
// only once a unit has been taken, and takes them itself otherwise: if the kept SMs were full until every other CTA had
// drained through the rest, the units still all run. CAP (kh_units_multi_smc): at most cap CTAs an SM, so every SM
// keeps the rest of its slots to the streams beside it
template <bool CAP>
__global__ void __launch_bounds__(32 * WARPS, H1U_MINB) h1um_sm(const UCall* __restrict__ calls, u32 ncalls, u32 M,
                                                      const uint8_t* __restrict__ live, u32 q, u32 den,
                                                      u32* __restrict__ next, u32 cap) {
  u32 smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (!sm_keep<CAP>(smid, q, den, cap, next)) return;
  __shared__ WarpSmem smem[WARPS];
  const u32 wp = threadIdx.x >> 5, lane = threadIdx.x & 31, total = calls[ncalls].u0;
  for (;;) {
    u32 gu = 0;
    if (lane == 0) gu = atomicAdd(next, 1u);
    gu = __shfl_sync(0xffffffffu, gu, 0);
    if (gu >= total) break;
    h1um_one(calls, ncalls, M, live, gu, smem[wp]);
    __syncwarp();
  }
}

#if H1U_KSPLIT
template <int KP>
struct alignas(16) KSmem {
  int4 dt[2][KP][8][32];   // [round parity][producer][fragment t][lane]: a segment's dT; after the walk, Y, Cw and Pz
  u32 bits[2][2][32][17];  // [chunk parity][0 rows | 1 columns][i][half]
  u32 coin[32][2];
  u32 utab[32][9];
  u32 ring[32][33];
};

template <int MD, bool PL, int KP>
__global__ void __launch_bounds__(32 * (KP + 1), 1) h1k(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 M,
                                                        u32 N, u32 K, const u64* __restrict__ seeds,
                                                        const uint8_t* __restrict__ live, u32* __restrict__ out,
                                                        const u32* __restrict__ ws, OneOut oo) {
  constexpr int KNI = 8 / KP;  // a chunk's 64 blocks are 256 uint4, KNI a producer lane
  __shared__ KSmem<KP> S;
  const u32 lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const u32 g = lane >> 2, q = lane & 3;
  const u32 ncb = N / 32, u = blockIdx.x;
  if (u >= (M / 32) * ncb) return;
  const u32 ri = u / ncb, cj = u % ncb;
  const u32 ks = K / 32, chunks = ks / 16, rb = chunks + 1, cb = chunks + 1;
  const u32 nwords = 2 * (ks + MD);
  u32* o = out + (u64)u * nwords;
  const int4 st = *reinterpret_cast<const int4*>(oo.step);
  const int alive = st.y, flags = st.w;
  const bool rcpt = !(flags & F_NW_UNITS);
  constexpr bool RO = H1K_RONLY;
  static_assert(!(RO && PL), "H1K_RONLY has no peel");
  if (live && !live[ri]) {
    if (wp) return;
    if (rcpt)
      for (u32 w = lane; w < nwords; w += 32) o[w] = 0;
    if constexpr (RO) return;
    const u32 row = 32 * ri + lane;
    if (!(flags & F_NW_TILES)) {
      uint4* zr = reinterpret_cast<uint4*>(oo.z + (u64)row * N + 32 * cj);
#pragma unroll
      for (int i = 0; i < 8; ++i) zr[i] = make_uint4(0, 0, 0, 0);
    }
    if (row < oo.M) {
      uint4* yr = reinterpret_cast<uint4*>(oo.y + (u64)row * N + 32 * cj);
#pragma unroll
      for (int i = 0; i < 4; ++i) yr[i] = make_uint4(0, 0, 0, 0);
      if (oo.zo) {
        uint4* zr = reinterpret_cast<uint4*>(oo.zo + (u64)row * N + 32 * cj);
#pragma unroll
        for (int i = 0; i < 8; ++i) zr[i] = make_uint4(0, 0, 0, 0);
      }
      if ((flags & F_DUMP) && (int)row < alive && oo.ydump) {
        uint4* dr = reinterpret_cast<uint4*>(oo.ydump + (u64)row * N + 32 * cj);
#pragma unroll
        for (int i = 0; i < 4; ++i) dr[i] = make_uint4(0, 0, 0, 0);
      }
    }
    return;
  }
  const u32* wsu = ws + 16ull * 32 * (rb + cb) * u;
  const u32* wsr = ws + 16ull * 32 * (rb + cb) * (ri * ncb + cj / oo.band * oo.band);
  const u32 pp = wp - 1, pi = 32 * pp + lane;
  // chunk blk's 64 blocks (rows' t = i RB + blk, columns' 32 RB + i CB + blk), uint4 w of block kb a load
  const auto load_noise = [&](u32 blk, uint4(&v)[KNI]) {
#pragma unroll
    for (int m = 0; m < KNI; ++m) {
      const u32 k = pi + 32 * KP * m, kb = k >> 2, w = k & 3;
      const u64 t = kb < 32 ? (u64)kb * rb + blk : 32ull * rb + (u64)(kb - 32) * cb + blk;
      v[m] = ld4e<H1U_L2HINT_WS, H1U_EVF>(reinterpret_cast<const uint4*>((kb < 32 ? wsr : wsu) + 16 * t) + w);
    }
  };
  const auto store_noise = [&](u32 bf, const uint4(&v)[KNI]) {
#pragma unroll
    for (int m = 0; m < KNI; ++m) {
      const u32 k = pi + 32 * KP * m, kb = k >> 2, w = k & 3;
      u32* d = &S.bits[bf][kb >> 5][kb & 31][4 * w];
      d[0] = v[m].x;
      d[1] = v[m].y;
      d[2] = v[m].z;
      d[3] = v[m].w;
    }
  };

  u32 y[8], z[8];
  if (wp == 0) {
    const u64* sd = seeds + 16 * (u64)u;
    u64 msg[16] = {TAG0, TAG1, TAG2};
#pragma unroll
    for (int j = 0; j < 8; ++j) msg[3 + j] = __ldg(sd + 8 + j);
    msg[11] = lane;
    msg[12] = 0x8000000000000000ULL;
    msg[13] = 0;
    msg[14] = 0;
    msg[15] = 96 * 8;
    u64 s[8];
    sha512_words(msg, 1, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      y[2 * j] = (u32)(s[j] >> 32);
      y[2 * j + 1] = (u32)s[j];
      z[2 * j] = (u32)(s[4 + j] >> 32);
      z[2 * j + 1] = (u32)s[4 + j];
    }
  } else {
    uint4 v[KNI];
    load_noise(chunks, v);
    store_noise(1, v);
    load_noise(0, v);
    store_noise(0, v);
  }
  __syncthreads();
  if (wp == 1) {
#pragma unroll
    for (int d = 0; d < MD - 1; ++d) S.coin[lane][d] = S.bits[1][0][lane][d];
#pragma unroll
    for (int h = 0; h < 8; ++h) {
      const u32 hv = S.bits[1][1][lane][h];
      u32 w = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const u32 x = (hv >> (8 * b)) & 0xFFu;
        w |= ((x >= 129 ? x - 129 : x) + 64) << (8 * b);
      }
      S.utab[lane][h] = w;
    }
  }
  __syncthreads();

  int T[8][4], Y[8][4], Cw[8][4], Pz[PL ? 8 : 1][4];
#pragma unroll
  for (int t = 0; t < 8; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) T[t][e] = Y[t][e] = Cw[t][e] = 0;
  if constexpr (PL) {
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) Pz[t][e] = 0;
  }
  u32 idr[4][2], ulo[4], uhi[4];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    idr[nt][0] = (q == (u32)nt && g < 4) ? 1u << (8 * g) : 0u;
    idr[nt][1] = (q == (u32)nt && g >= 4) ? 1u << (8 * (g - 4)) : 0u;
    ulo[nt] = S.utab[8 * nt + g][2 * q];
    uhi[nt] = S.utab[8 * nt + g][2 * q + 1];
  }
  const int8_t* Ab = A + (u64)(32 * ri) * K;
  const int8_t* Bb = B + (u64)(32 * cj) * K;
  const auto load_seg = [&](u32 sg, uint2(&a)[4], uint2(&b)[4]) {
    const u32 kofs = 32 * sg + 8 * q;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      a[r] = ld2<H1U_L2HINT>(Ab + (u64)(16 * (r >> 1) + g + 8 * (r & 1)) * K + kofs);
      b[r] = ld2e<H1U_L2HINT, H1U_EVF>(Bb + (u64)(8 * r + g) * K + kofs);
    }
  };
  // round r's segment in ring slot r % NS, loaded KD rounds ahead
  constexpr int KD = H1K_PD ? H1K_PD : 1, NS = H1K_PD ? H1K_PD + 1 : 1;
  uint2 pa[NS][4], pb[NS][4];
  const u32 R = ks / KP;
#if H1K_CPA
  // stage st's [0 A | 1 B][producer][r][lane]
  extern __shared__ uint2 h1k_cp[];
  constexpr u32 CS = H1K_CPA;
  const auto cp_seg = [&](u32 sg, u32 st) {
    const u32 kofs = 32 * sg + 8 * q;
    uint2* sa = h1k_cp + (2 * st * KP + pp) * 128 + lane;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      cp8e<H1U_L2HINT, false>(sa + 32 * r, Ab + (u64)(16 * (r >> 1) + g + 8 * (r & 1)) * K + kofs);
      cp8e<H1U_L2HINT, H1U_EVF>(sa + KP * 128 + 32 * r, Bb + (u64)(8 * r + g) * K + kofs);
    }
  };
  if (wp) {
#pragma unroll
    for (u32 d = 0; d + 1 < CS; ++d) {
      if (d < R) cp_seg(KP * d + pp, d);
      cp_commit();
    }
  }
#else
  if (wp) {
#pragma unroll
    for (int d = 0; d < KD; ++d)
      if (KP * d + pp < ks) load_seg(KP * d + pp, pa[d], pb[d]);
  }
#endif
  uint4 nv[KNI];
  const auto run_round = [&](u32 r, uint2(&ra)[4], uint2(&rb4)[4], uint2(&na)[4], uint2(&nb)[4]) {
    if (wp) {
      if (r < R) {
        const u32 sg = KP * r + pp, blk = sg >> 4, j = sg & 15;
        // the next chunk's blocks: loaded at a chunk's first round, stored at its end (H1K_NXD: its second round's; its
        // buffer's last reader, chunk blk - 1, finished before this round's barrier, and its next is chunk blk + 1)
        const bool nx = ((KP * r) & 15) == 0 && blk + 1 < chunks;
        const bool sx = H1K_NXD ? r >= 1 && ((KP * (r - 1)) & 15) == 0 && blk + 1 < chunks : nx;
        if (nx) load_noise(blk + 1, nv);
        uint2 ca[4], cw[4];
#if H1K_CPA
        if (r + CS - 1 < R) cp_seg(sg + KP * (CS - 1), (r + CS - 1) % CS);
        cp_commit();
        cp_wait<CS - 1>();
        const uint2* sa = h1k_cp + (2 * (r % CS) * KP + pp) * 128 + lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ca[i] = sa[32 * i];
          cw[i] = sa[KP * 128 + 32 * i];
        }
#else
        if (H1K_PD && sg + KP * KD < ks) load_seg(sg + KP * KD, na, nb);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ca[i] = ra[i];
          cw[i] = rb4[i];
        }
        if (!H1K_PD && sg + KP < ks) load_seg(sg + KP, ra, rb4);
#endif
        if (H1K_L2PF && ((sg + H1K_L2PF) & 3) == 0 && sg + H1K_L2PF < ks)
          prefetch_l2(Bb + (u64)lane * K + 32 * (sg + H1K_L2PF));
        const uint8_t* rbits = reinterpret_cast<const uint8_t*>(&S.bits[blk & 1][0][0][0]);
        const uint8_t* cbits = reinterpret_cast<const uint8_t*>(&S.bits[blk & 1][1][0][0]);
        u32 af[2][4], bf[4][2];
        u32 bb[PL ? 2 : 1][4], xa[PL ? 2 : 1][4], xany = 0;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const u32 row = 16 * mt + g + 8 * h;
            const uint2 a8 = ca[2 * mt + h];
            const u32 nb8 = rbits[row * 68 + 4 * j + q];
            const u32 e0 = expand4(nb8 & 15u), e1 = expand4(nb8 >> 4);
            af[mt][h] = add01(a8.x, e0);
            af[mt][2 + h] = add01(a8.y, e1);
            if constexpr (PL) {
              bb[mt][h] = e0;
              bb[mt][2 + h] = e1;
              xa[mt][h] = __vcmpeq4(a8.x, 0x7F7F7F7Fu) & e0;
              xa[mt][2 + h] = __vcmpeq4(a8.y, 0x7F7F7F7Fu) & e1;
              xany |= xa[mt][h] | xa[mt][2 + h];
            }
          }
        }
        if constexpr (PL) {
          if ((flags & F_FAULT_XA) && u == 0 && lane == 0 && sg == 0) {
            xa[0][0] ^= 1u;
            xany = 1;
          }
        }
        u32 wbf[4][2], cf[PL ? 4 : 1][2], b0[PL ? 4 : 1][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const u32 col = 8 * nt + g;
          const uint2 b8 = cw[nt];
          const u32 cb8 = cbits[col * 68 + 4 * j + q];
          const u32 c0 = expand4(cb8 & 15u), c1 = expand4(cb8 >> 4);
          const u32 v0 = ulo[nt] + c0, v1 = uhi[nt] + c1;
          bf[nt][0] = addw(b8.x, v0);
          bf[nt][1] = addw(b8.y, v1);
          if constexpr (!RO) {
            wbf[nt][0] = wrapw(b8.x, v0);
            wbf[nt][1] = wrapw(b8.y, v1);
          }
          if constexpr (PL) {
            cf[nt][0] = c0;
            cf[nt][1] = c1;
            b0[nt][0] = b8.x;
            b0[nt][1] = b8.y;
          }
        }
        if (!RO && (flags & F_FAULT_WB) && u == 0 && lane == 0 && sg == 0)
          wbf[0][0] = (wbf[0][0] & ~0xFFu) | ((wbf[0][0] & 0xFFu) == 0 ? 1u : 0u);
        int dT[8][4];
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
          for (int e = 0; e < 4; ++e) dT[t][e] = 0;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            mma_s8(dT[4 * mt + nt], af[mt], bf[nt]);
            mma_s8(Y[4 * mt + nt], af[mt], idr[nt]);
            if constexpr (!RO) mma_s8(Cw[4 * mt + nt], af[mt], wbf[nt]);
            if constexpr (PL) {
              mma_s8(Pz[4 * mt + nt], af[mt], cf[nt]);
              mma_s8(Pz[4 * mt + nt], bb[mt], b0[nt]);
            }
          }
        if constexpr (PL) {
          if (__any_sync(0xFFFFFFFFu, xany != 0)) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
              for (int nt = 0; nt < 4; ++nt) mma_s8(Cw[4 * mt + nt], xa[mt], b0[nt]);
          }
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) S.dt[r & 1][pp][t][lane] = make_int4(dT[t][0], dT[t][1], dT[t][2], dT[t][3]);
        if (sx) store_noise((blk + 1) & 1, nv);
      }
    } else if (r) {
      const u32 r1 = r - 1;
#pragma unroll
      for (int p = 0; p < KP; ++p) {
        const u32 sg = KP * r1 + p, j = sg & 15;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          const int4 d = S.dt[r1 & 1][p][t][lane];
          T[t][0] = (int)((u32)T[t][0] + (u32)d.x);
          T[t][1] = (int)((u32)T[t][1] + (u32)d.y);
          T[t][2] = (int)((u32)T[t][2] + (u32)d.z);
          T[t][3] = (int)((u32)T[t][3] + (u32)d.w);
        }
        if ((flags & F_FAULT) && u == 0 && lane == 0 && sg == 0) T[0][0] = (int)((u32)T[0][0] + (1u << FAULT_SHIFT));
        fold(T, y, z);
        ring_put(S, j, lane, y, z);
        if (j == 15) {
          __syncwarp();
          if (rcpt) o[32 * (sg >> 4) + lane] = ring_xor(S, lane);
          __syncwarp();
        }
      }
    }
    __syncthreads();
  };
#pragma unroll 1
  for (u32 r0 = 0; r0 <= R; r0 += NS) {
#pragma unroll
    for (int d = 0; d < NS; ++d)
      if (r0 + d <= R) run_round(r0 + d, pa[d], pb[d], pa[(d + KD) % NS], pb[(d + KD) % NS]);
  }
  // the producers' Y, Cw and Pz to warp 0 as sums
  if (wp) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      S.dt[0][pp][t][lane] = make_int4(Y[t][0], Y[t][1], Y[t][2], Y[t][3]);
      if constexpr (!RO) S.dt[1][pp][t][lane] = make_int4(Cw[t][0], Cw[t][1], Cw[t][2], Cw[t][3]);
    }
  }
  __syncthreads();
  if (wp == 0) {
#pragma unroll
    for (int p = 0; p < KP; ++p)
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int4 a = S.dt[0][p][t][lane];
        Y[t][0] = (int)((u32)Y[t][0] + (u32)a.x);
        Y[t][1] = (int)((u32)Y[t][1] + (u32)a.y);
        Y[t][2] = (int)((u32)Y[t][2] + (u32)a.z);
        Y[t][3] = (int)((u32)Y[t][3] + (u32)a.w);
        if constexpr (RO) continue;
        const int4 c = S.dt[1][p][t][lane];
        Cw[t][0] = (int)((u32)Cw[t][0] + (u32)c.x);
        Cw[t][1] = (int)((u32)Cw[t][1] + (u32)c.y);
        Cw[t][2] = (int)((u32)Cw[t][2] + (u32)c.z);
        Cw[t][3] = (int)((u32)Cw[t][3] + (u32)c.w);
      }
  }
  if constexpr (PL) {
    __syncthreads();
    if (wp) {
#pragma unroll
      for (int t = 0; t < 8; ++t) S.dt[0][pp][t][lane] = make_int4(Pz[t][0], Pz[t][1], Pz[t][2], Pz[t][3]);
    }
    __syncthreads();
    if (wp == 0) {
#pragma unroll
      for (int p = 0; p < KP; ++p)
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          const int4 a = S.dt[0][p][t][lane];
          Pz[t][0] = (int)((u32)Pz[t][0] + (u32)a.x);
          Pz[t][1] = (int)((u32)Pz[t][1] + (u32)a.y);
          Pz[t][2] = (int)((u32)Pz[t][2] + (u32)a.z);
          Pz[t][3] = (int)((u32)Pz[t][3] + (u32)a.w);
        }
    }
  }
  if (wp) return;

  // the tail, h1u's
  u32 ad[MD][2][4];
#pragma unroll
  for (int d = 0; d < MD; ++d)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int r = 0; r < 4; ++r) ad[d][mt][r] = 0;
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    const int mt = t >> 2, nt = t & 3;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const u32 row = 16 * mt + g + 8 * (e >> 1), rho = 8 * nt + 2 * q + (e & 1);
      int yv = Y[t][e];
      const int reg = (e >> 1) + 2 * (nt >> 1), sh8 = 8 * (2 * (nt & 1) + (e & 1));
#pragma unroll
      for (int d = 0; d < MD - 1; ++d) {
        const int qv = yv >> 7;
        u32 zd = (S.coin[row][d] >> rho) & 1u;
        if (d == MD - 2 && qv == 127) zd = 0;
        ad[d][mt][reg] |= (((u32)yv & 127u) | (zd << 7)) << sh8;
        yv = qv + (int)zd;
      }
      ad[MD - 1][mt][reg] |= ((u32)yv & 0xFFu) << sh8;
    }
  }
  u32 uf[4][2], nf[4][2];
  {
    const uint8_t* ub = reinterpret_cast<const uint8_t*>(&S.utab[0][0]);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const uint8_t* rr = ub + (8 * nt + g) * 36;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const u32 c0 = 16 * hf + 2 * q;
        const u32 by[4] = {rr[c0], rr[c0 + 1], rr[c0 + 8], rr[c0 + 9]};
        u32 pu = 0, pn = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          pu |= ((by[b] - 128u) & 0xFFu) << (8 * b);
          pn |= ((128u - by[b]) & 0xFFu) << (8 * b);
        }
        uf[nt][hf] = pu;
        nf[nt][hf] = pn;
      }
    }
  }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mma_s8(T[4 * mt + nt], ad[0][mt], nf[nt]);
  fold(T, y, z);
  ring_put(S, 0, lane, y, z);
#pragma unroll
  for (int d = 1; d < MD; ++d) {
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) Y[t][e] = 0;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) mma_s8(Y[4 * mt + nt], ad[d][mt], uf[nt]);
    fold(Y, y, z);
    ring_put(S, d, lane, y, z);
    if constexpr (RO) continue;
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) T[t][e] = (int)((u32)T[t][e] - ((u32)Y[t][e] << (7 * d)));
  }
  __syncwarp();
  if (rcpt && lane < 2 * MD) o[2 * ks + lane] = ring_xor(S, lane);
  if constexpr (RO) return;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int eh = 0; eh < 2; ++eh) {
      const u32 row = 32 * ri + 16 * mt + g + 8 * eh;
      const float s_a = (int)row < alive ? oo.sa[row] : 0.0f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int t = 4 * mt + nt;
        const u32 col = 32 * cj + 8 * nt + 2 * q;
        int z0 = (int)((u32)T[t][2 * eh] + ((u32)Cw[t][2 * eh] << 8));
        int z1 = (int)((u32)T[t][2 * eh + 1] + ((u32)Cw[t][2 * eh + 1] << 8));
        if constexpr (PL) {
          z0 = (int)((u32)z0 - (u32)Pz[t][2 * eh]);
          z1 = (int)((u32)z1 - (u32)Pz[t][2 * eh + 1]);
        }
        if (!(flags & F_NW_TILES)) *reinterpret_cast<int2*>(oo.z + (u64)row * N + col) = make_int2(z0, z1);
        if (row < oo.M) {
          if (oo.zo) *reinterpret_cast<int2*>(oo.zo + (u64)row * N + col) = make_int2(z0, z1);
          __nv_bfloat162 yv;
          yv.x = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z0), s_a), oo.sw[col]));
          yv.y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__int2float_rn(z1), s_a), oo.sw[col + 1]));
          *reinterpret_cast<__nv_bfloat162*>(oo.y + (u64)row * N + col) = yv;
          if ((flags & F_DUMP) && (int)row < alive && oo.ydump)
            *reinterpret_cast<__nv_bfloat162*>(oo.ydump + (u64)row * N + col) = yv;
        }
      }
    }
}
#endif

// a pair's shared state: the chunk's blocks by chunk parity ([0 rows | 1 columns][i][half]), reused after the last chunk
// for the warps' A_d halves; the coins and U bytes; warp 1's ring words by chunk parity; each warp's ring
struct alignas(16) PairSmem {
  NoiseMid nm[2];
  union {
    u32 bits[2][2][32][17];
    u32 xad[2][3][2][4][32];  // [warp][d][mt][reg][lane]
  };
  u32 coin[32][2];
  u32 utab[32][9];
  u32 xr[2][32];
  u32 ring[2][32][33];
};

__device__ __forceinline__ void pair_sync(u32 pair) { asm volatile("bar.sync %0, 64;" ::"r"(1 + pair) : "memory"); }

// half P of a set: local tile l = 2 mt + (nt >> 1) of n-tile nt = P + 2 (l & 1) is key t = 2 l + P, chain 4 P + c's entry e = c
template <int P>
__device__ __forceinline__ void fold_half(const int (&acc)[4][4], u32 (&y)[4], u32 (&z)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int t = 2 * i + P;
      const u32 p = (u32)acc[i][c] * KEYS.fx1[8 * i + 4 * P + c];
      const u32 uu = y[c] ^ p ^ KEYS.kk[t];
      y[c] = uu * KEYS.ka[t] + z[c];
      z[c] = __funnelshift_l(uu, uu, KEYS.rot[t]) + y[c];
    }
  }
}

__device__ __forceinline__ void ring_put_half(u32 (&ring)[32][33], int slot, u32 lane, const u32 (&y)[4], const u32 (&z)[4]) {
  ring[2 * slot][lane] = y[0] ^ y[1] ^ y[2] ^ y[3];
  ring[2 * slot + 1][lane] = z[0] ^ z[1] ^ z[2] ^ z[3];
}

__device__ __forceinline__ u32 ring_xor_half(const u32 (&ring)[32][33], u32 row) {
  u32 x = 0;
#pragma unroll
  for (int l = 0; l < 32; ++l) x ^= ring[row][l];
  return x;
}

template <int MD, bool WS, int P>
__device__ __forceinline__ void h1p_half(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 N, u32 K,
                                         const u64* __restrict__ seeds, u32* __restrict__ out,
                                         const u32* __restrict__ ws, PairSmem& S, u32 pair, u32 u) {
  constexpr bool PF = WS || H1U_PF;
  constexpr int PD = H1U_PD;
  const u32 lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const u32 ncb = N / 32, ri = u / ncb, cj = u % ncb;
  const u32 ks = K / 32, chunks = ks / 16, rb = chunks + 1, cb = chunks + 1;
  u32* o = out + (u64)u * (2 * (ks + MD));
  const u64* sd = seeds + 16 * (u64)u;

  u32 y[4], z[4];
  {
    u64 msg[16] = {TAG0, TAG1, TAG2};
#pragma unroll
    for (int j = 0; j < 8; ++j) msg[3 + j] = __ldg(sd + 8 + j);
    msg[11] = lane;
    msg[12] = 0x8000000000000000ULL;
    msg[13] = 0;
    msg[14] = 0;
    msg[15] = 96 * 8;
    u64 s[8];
    sha512_words(msg, 1, s);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int k = 4 * P + c;
      y[c] = (k & 1) ? (u32)s[k >> 1] : (u32)(s[k >> 1] >> 32);
      z[c] = (k & 1) ? (u32)s[4 + (k >> 1)] : (u32)(s[4 + (k >> 1)] >> 32);
    }
  }
  if (!WS) {
    u64 seed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) seed[j] = __ldg(sd + j);
    NoiseMid nm;
    noise_mid(seed, nm);
    if (lane == 0) S.nm[P] = nm;
  }
  __syncwarp();
  const u32* wsu = WS ? ws + 16ull * 32 * (rb + cb) * u : nullptr;

  u32 idr[2][2];
#pragma unroll
  for (int l = 0; l < 2; ++l) {
    const u32 nt = P + 2 * l;
    idr[l][0] = (q == nt && g < 4) ? 1u << (8 * g) : 0u;
    idr[l][1] = (q == nt && g >= 4) ? 1u << (8 * (g - 4)) : 0u;
  }
  int T[4][4], Y[4][4];
#pragma unroll
  for (int t = 0; t < 4; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) T[t][e] = Y[t][e] = 0;
  u32 ulo[2], uhi[2];

  const int8_t* Ab = A + (u64)(32 * ri) * K;
  const int8_t* Bb = B + (u64)(32 * cj) * K;
  uint2 pa[PD][4], pb[PD][2];
  const auto load_seg = [&](u32 sg, uint2(&a)[4], uint2(&b)[2]) {
    const u32 kofs = 32 * sg + 8 * q;
#pragma unroll
    for (int r = 0; r < 4; ++r)
      a[r] = __ldg(reinterpret_cast<const uint2*>(Ab + (u64)(16 * (r >> 1) + g + 8 * (r & 1)) * K + kofs));
#pragma unroll
    for (int l = 0; l < 2; ++l)
      b[l] = __ldg(reinterpret_cast<const uint2*>(Bb + (u64)(8 * (P + 2 * l) + g) * K + kofs));
  };
  if (PF) {
#pragma unroll
    for (int d = 0; d < PD; ++d) load_seg(d, pa[d], pb[d]);
  }
  u32 pend = 0;
#pragma unroll 1
  for (u32 it = 0; it <= chunks; ++it) {
    const u32 blk = it == 0 ? chunks : it - 1, buf = it & 1;
    {
      const u64 t = P == 0 ? (u64)lane * rb + blk : 32ull * rb + (u64)lane * cb + blk;
      u32* dst = S.bits[buf][P][lane];
      if (WS) {
        const uint4* src = reinterpret_cast<const uint4*>(wsu + 16 * t);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const uint4 v = __ldg(src + w);
          dst[4 * w] = v.x;
          dst[4 * w + 1] = v.y;
          dst[4 * w + 2] = v.z;
          dst[4 * w + 3] = v.w;
        }
        if (H1U_L2PF && it < chunks) prefetch_l2(wsu + 16 * (t - blk + it));
      } else {
        u64 s[8];
        noise_block(S.nm[P], t, s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          dst[2 * j] = (u32)(s[j] >> 32);
          dst[2 * j + 1] = (u32)s[j];
        }
      }
    }
    pair_sync(pair);
    if (P == 0 && it >= 2) o[32 * (it - 2) + lane] = pend ^ S.xr[(it - 1) & 1][lane];
    if (it == 0) {
      if (P == 0) {
#pragma unroll
        for (int d = 0; d < MD - 1; ++d) S.coin[lane][d] = S.bits[0][0][lane][d];
      } else {
#pragma unroll
        for (int h = 0; h < 8; ++h) {
          const u32 hv = S.bits[0][1][lane][h];
          u32 w = 0;
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const u32 x = (hv >> (8 * b)) & 0xFFu;
            w |= ((x >= 129 ? x - 129 : x) + 64) << (8 * b);
          }
          S.utab[lane][h] = w;
        }
      }
      pair_sync(pair);
#pragma unroll
      for (int l = 0; l < 2; ++l) {
        ulo[l] = S.utab[8 * (P + 2 * l) + g][2 * q];
        uhi[l] = S.utab[8 * (P + 2 * l) + g][2 * q + 1];
      }
      continue;
    }
    const uint8_t* rbits = reinterpret_cast<const uint8_t*>(&S.bits[buf][0][0][0]);
    const uint8_t* cbits = reinterpret_cast<const uint8_t*>(&S.bits[buf][1][0][0]);
#pragma unroll 1
    for (u32 j0 = 0; j0 < 16; j0 += PD) {
#pragma unroll
      for (int jj = 0; jj < PD; ++jj) {
        const u32 j = j0 + jj, sg = 16 * blk + j;
        uint2 ca[4], cw[2];
        if (H1U_L2PF && sg % 4 == 0 && sg + 16 < ks) {
          prefetch_l2(Ab + (u64)lane * K + 32 * (sg + 16));
          prefetch_l2(Bb + (u64)lane * K + 32 * (sg + 16));
        }
        if (PF) {
#pragma unroll
          for (int r = 0; r < 4; ++r) ca[r] = pa[jj][r];
#pragma unroll
          for (int l = 0; l < 2; ++l) cw[l] = pb[jj][l];
          if (sg + PD < ks) load_seg(sg + PD, pa[jj], pb[jj]);
        } else {
          load_seg(sg, ca, cw);
        }
        u32 af[2][4], bf[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const u32 row = 16 * mt + g + 8 * h;
            const uint2 a8 = ca[2 * mt + h];
            const u32 nb8 = rbits[row * 68 + 4 * j + q];
            af[mt][h] = add01(a8.x, expand4(nb8 & 15u));
            af[mt][2 + h] = add01(a8.y, expand4(nb8 >> 4));
          }
        }
#pragma unroll
        for (int l = 0; l < 2; ++l) {
          const u32 col = 8 * (P + 2 * l) + g;
          const uint2 b8 = cw[l];
          const u32 cb8 = cbits[col * 68 + 4 * j + q];
          bf[l][0] = addw(b8.x, ulo[l] + expand4(cb8 & 15u));
          bf[l][1] = addw(b8.y, uhi[l] + expand4(cb8 >> 4));
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int l = 0; l < 2; ++l) {
            mma_s8(T[2 * mt + l], af[mt], bf[l]);
            mma_s8(Y[2 * mt + l], af[mt], idr[l]);
          }
        fold_half<P>(T, y, z);
        ring_put_half(S.ring[P], j, lane, y, z);
      }
    }
    __syncwarp();
    const u32 part = ring_xor_half(S.ring[P], lane);
    if (P == 1) S.xr[buf][lane] = part;
    else pend = part;
    __syncwarp();
  }
  pair_sync(pair);
  if (P == 0) o[32 * (chunks - 1) + lane] = pend ^ S.xr[chunks & 1][lane];

  u32 ad[MD][2][4];
#pragma unroll
  for (int d = 0; d < MD; ++d)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int r = 0; r < 4; ++r) ad[d][mt][r] = 0;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int mt = l >> 1, nt = P + 2 * (l & 1);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const u32 row = 16 * mt + g + 8 * (e >> 1), rho = 8 * nt + 2 * q + (e & 1);
      int yv = Y[l][e];
      const int reg = (e >> 1) + 2 * (nt >> 1), sh8 = 8 * (2 * (nt & 1) + (e & 1));
#pragma unroll
      for (int d = 0; d < MD - 1; ++d) {
        const int qv = yv >> 7;
        u32 zd = (S.coin[row][d] >> rho) & 1u;
        if (d == MD - 2 && qv == 127) zd = 0;
        ad[d][mt][reg] |= (((u32)yv & 127u) | (zd << 7)) << sh8;
        yv = qv + (int)zd;
      }
      ad[MD - 1][mt][reg] |= ((u32)yv & 0xFFu) << sh8;
    }
  }
#pragma unroll
  for (int d = 0; d < MD; ++d)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int r = 0; r < 4; ++r) S.xad[P][d][mt][r][lane] = ad[d][mt][r];
  pair_sync(pair);
#pragma unroll
  for (int d = 0; d < MD; ++d)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int r = 0; r < 4; ++r) ad[d][mt][r] |= S.xad[1 - P][d][mt][r][lane];
  u32 uf[2][2], nf[2][2];
  {
    const uint8_t* ub = reinterpret_cast<const uint8_t*>(&S.utab[0][0]);
#pragma unroll
    for (int l = 0; l < 2; ++l) {
      const uint8_t* r = ub + (8 * (P + 2 * l) + g) * 36;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const u32 c0 = 16 * hf + 2 * q;
        const u32 by[4] = {r[c0], r[c0 + 1], r[c0 + 8], r[c0 + 9]};
        u32 pu = 0, pn = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          pu |= ((by[b] - 128u) & 0xFFu) << (8 * b);
          pn |= ((128u - by[b]) & 0xFFu) << (8 * b);
        }
        uf[l][hf] = pu;
        nf[l][hf] = pn;
      }
    }
  }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int l = 0; l < 2; ++l) mma_s8(T[2 * mt + l], ad[0][mt], nf[l]);
  fold_half<P>(T, y, z);
  ring_put_half(S.ring[P], 0, lane, y, z);
#pragma unroll
  for (int d = 1; d < MD; ++d) {
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) Y[t][e] = 0;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int l = 0; l < 2; ++l) mma_s8(Y[2 * mt + l], ad[d][mt], uf[l]);
    fold_half<P>(Y, y, z);
    ring_put_half(S.ring[P], d, lane, y, z);
  }
  __syncwarp();
  const u32 part = lane < 2 * MD ? ring_xor_half(S.ring[P], lane) : 0u;
  if (P == 1) S.xr[0][lane] = part;
  pair_sync(pair);
  if (P == 0 && lane < 2 * MD) o[2 * ks + lane] = part ^ S.xr[0][lane];
}

template <int MD, bool WS>
__global__ void __launch_bounds__(32 * WARPS, H1U_MINB) h1p(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 M, u32 N,
                                                  u32 K, const u64* __restrict__ seeds, const uint8_t* __restrict__ live,
                                                  u32* __restrict__ out, const u32* __restrict__ ws) {
  __shared__ PairSmem smem[WARPS / 2];
  const u32 lane = threadIdx.x & 31, wp = threadIdx.x >> 5, pair = wp >> 1;
  const u32 ncb = N / 32, units = (M / 32) * ncb;
  const u32 u = blockIdx.x * (WARPS / 2) + pair;
  if (u >= units) return;
  if (live && !live[u / ncb]) {
    if (wp & 1) return;
    const u32 nwords = 2 * (K / 32 + MD);
    u32* o = out + (u64)u * nwords;
    for (u32 w = lane; w < nwords; w += 32) o[w] = 0;
    return;
  }
  if (wp & 1) h1p_half<MD, WS, 1>(A, B, N, K, seeds, out, ws, smem[pair], pair, u);
  else h1p_half<MD, WS, 0>(A, B, N, K, seeds, out, ws, smem[pair], pair, u);
}

template <int MD, bool WS>
static void launch_unit(unsigned units, cudaStream_t s, const void* A, const void* B, unsigned M, unsigned N,
                        unsigned K, const void* seeds, const void* live, void* out, const void* ws) {
  const auto a = (const int8_t*)A;
  const auto b = (const int8_t*)B;
  const auto sd = (const u64*)seeds;
  const auto lv = (const uint8_t*)live;
  const auto w = (const u32*)ws;
#if H1U_SPLIT
  constexpr unsigned per = WARPS / 2;
  h1p<MD, WS><<<(units + per - 1) / per, 32 * WARPS, 0, s>>>(a, b, M, N, K, sd, lv, (u32*)out, w);
#else
  h1u<MD, WS, false><<<(units + WARPS - 1) / WARPS, 32 * WARPS, 0, s>>>(a, b, M, N, K, sd, lv, (u32*)out, w, OneOut{});
#endif
}

// H1U_CARVE's attribute for kernel F (a failure leaves the driver's default, and clears the error it set)
template <auto F>
static void carve() {
#if H1U_CARVE
  static const bool set = [] {
    if (cudaFuncSetAttribute((const void*)F, cudaFuncAttributePreferredSharedMemoryCarveout,
                             (int)cudaSharedmemCarveoutMaxShared) == cudaSuccess)
      return true;
    (void)cudaGetLastError();
    return false;
  }();
  (void)set;
#endif
}

#if H1U_KSPLIT
// h1k<MD, PL, KP>'s dynamic shared memory (H1K_CPA's stages), its attributes set once a process before the first launch
// or occupancy query (a failure surfaces as the launch's error)
template <int MD, bool PL, int KP>
static unsigned h1k_dyn() {
  constexpr unsigned b = H1K_CPA * 2 * KP * 128 * sizeof(uint2);
  if constexpr (b != 0) {
    static const bool set = [] {
      const void* f = (const void*)h1k<MD, PL, KP>;
      return cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b) == cudaSuccess
             && cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout,
                                     (int)cudaSharedmemCarveoutMaxShared) == cudaSuccess;
    }();
    (void)set;
  }
  return b;
}

template <int MD, bool PL, int KP>
static void launch_k(unsigned units, cudaStream_t s, const void* A, const void* B, unsigned M, unsigned N, unsigned K,
                     const void* seeds, const void* live, void* out, const void* ws, const OneOut& oo) {
  carve<h1k<MD, PL, KP>>();
  h1k<MD, PL, KP><<<units, 32 * (KP + 1), h1k_dyn<MD, PL, KP>(), s>>>((const int8_t*)A, (const int8_t*)B, M, N, K,
                                                                       (const u64*)seeds, (const uint8_t*)live, (u32*)out,
                                                                       (const u32*)ws, oo);
}

// h1k<MD, PL, KP>'s CTAs resident at once (SMs x the occupancy calculator's CTAs an SM; 0 if either query fails), read
// once a process, so on the device of its first call
template <int MD, bool PL, int KP>
static unsigned h1k_wave() {
  static const unsigned w = [] {
    int dev = 0, sms = 0, per = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
        || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, h1k<MD, PL, KP>, 32 * (KP + 1), h1k_dyn<MD, PL, KP>())
               != cudaSuccess)
      return 0u;
    return (unsigned)(sms * per);
  }();
  return w;
}
#endif

template <int MD, bool WS, bool J10 = false, bool PL = false>
static void launch_one(unsigned units, cudaStream_t s, const void* A, const void* B, unsigned M, unsigned N, unsigned K,
                       const void* seeds, const void* live, void* out, const void* ws, const OneOut& oo) {
  constexpr bool RO = H1K_RONLY && !J10 && !PL;
#if H1U_KSPLIT
  if constexpr (WS && !J10 && (RO || !H1K_RONLY)) {
    const unsigned rows = oo.M < M ? oo.M : M, lu = (rows + 31) / 32 * (N / 32);
#if H1K_AUTO
    if (lu <= h1k_wave<MD, PL, 4>()) {
      launch_k<MD, PL, 4>(units, s, A, B, M, N, K, seeds, live, out, ws, oo);
      return;
    }
    if (lu <= H1K_MAXU) {
      launch_k<MD, PL, 2>(units, s, A, B, M, N, K, seeds, live, out, ws, oo);
      return;
    }
#else
    if (lu <= H1K_MAXU) {
      launch_k<MD, PL, H1U_KSPLIT>(units, s, A, B, M, N, K, seeds, live, out, ws, oo);
      return;
    }
#endif
  }
#endif
  carve<h1u<MD, WS, !RO, J10, PL>>();
  h1u<MD, WS, !RO, J10, PL><<<(units + WARPS - 1) / WARPS, 32 * WARPS, 0, s>>>((const int8_t*)A, (const int8_t*)B, M, N,
                                                                              K, (const u64*)seeds, (const uint8_t*)live,
                                                                              (u32*)out, (const u32*)ws, oo);
}

extern "C" int kh_digits(unsigned K) { return n_digits((int)(K / 32)); }
extern "C" unsigned kh_words(unsigned K) { return 2 * (K / 32 + n_digits((int)(K / 32))); }

extern "C" int kh_seeds(const void* data, unsigned long long base, unsigned U, void* seeds, void* stream) {
  if (!U) return 0;
  h1_seeds<<<(U + 127) / 128, 128, 0, (cudaStream_t)stream>>>((const u64*)data, base, U, (u64*)seeds);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int kh_seeds_at(const void* data, const void* idx, const void* pos, unsigned U, void* seeds, void* stream) {
  if (!U) return 0;
  h1_seeds_at<<<(U + 127) / 128, 128, 0, (cudaStream_t)stream>>>((const u64*)data, (const u32*)idx, (const u64*)pos, U,
                                                                 (u64*)seeds);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int kh_unit(const void* A, const void* B, unsigned M, unsigned N, unsigned K, const void* seeds, const void* live,
                       void* out, void* stream) {
  if (K % 512) return 2;
  const int m = n_digits((int)(K / 32));
  const unsigned units = (M / 32) * (N / 32);
  if (!units) return 0;
  cudaStream_t s = (cudaStream_t)stream;
  if (m == 2)
    launch_unit<2, false>(units, s, A, B, M, N, K, seeds, live, out, nullptr);
  else if (m == 3)
    launch_unit<3, false>(units, s, A, B, M, N, K, seeds, live, out, nullptr);
  else
    return 2;
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// kh_unit's receipts of ncalls calls in one launch: calls a device array of ncalls + 1 UCall (kh_ucall_bytes() each, the
// last's u0 the units in all, `units`), every call M rows with the same live row tiles
extern "C" unsigned kh_ucall_bytes() { return sizeof(UCall); }

extern "C" int kh_units_multi(const void* calls, unsigned ncalls, unsigned units, unsigned M, const void* live,
                              void* stream) {
  if (!ncalls || M % 32) return 2;
  if (!units) return 0;
  carve<h1um>();
  h1um<<<(units + WARPS - 1) / WARPS, 32 * WARPS, 0, (cudaStream_t)stream>>>((const UCall*)calls, ncalls, M,
                                                                            (const uint8_t*)live);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// kh_units_multi on q / den of the SMs (h1um_sm, 1 <= q <= den <= SMQ_DEN_MAX): next a device u32 the launch zeroes
// first, and twice the CTAs that fit at once, so the SMs it keeps get theirs whatever else holds the GPU at the launch
extern "C" int kh_units_multi_sm(const void* calls, unsigned ncalls, unsigned units, unsigned M, const void* live, unsigned q,
                                 unsigned den, void* next, void* stream) {
  if (!ncalls || M % 32 || !q || q > den || den > SMQ_DEN_MAX || !next) return 2;
  if (!units) return 0;
  carve<h1um_sm<false>>();
  int dev = 0, sms = 0, per = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, h1um_sm<false>, 32 * WARPS, 0) != cudaSuccess || per < 1)
    return 3;
  cudaStream_t s = (cudaStream_t)stream;
  if (cudaMemsetAsync(next, 0, sizeof(u32), s) != cudaSuccess) return 3;
  h1um_sm<false><<<2u * (unsigned)per * (unsigned)sms, 32 * WARPS, 0, s>>>((const UCall*)calls, ncalls, M,
                                                                          (const uint8_t*)live, q, den, (u32*)next, 0);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// u32 words of an _smc launch's next: the unit counter, then SMC_SLOTS per-SM CTA counts
extern "C" unsigned kh_smc_words() { return 1 + SMC_SLOTS; }

// kh_units_multi_sm with at most cap CTAs an SM (h1um_sm<true>, 1 <= cap <= 64): next kh_smc_words() u32 the launch
// zeroes first
extern "C" int kh_units_multi_smc(const void* calls, unsigned ncalls, unsigned units, unsigned M, const void* live,
                                  unsigned q, unsigned den, unsigned cap, void* next, void* stream) {
  if (!ncalls || M % 32 || !q || q > den || den > SMQ_DEN_MAX || !cap || cap > 64 || !next) return 2;
  if (!units) return 0;
  carve<h1um_sm<true>>();
  int dev = 0, sms = 0, per = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, h1um_sm<true>, 32 * WARPS, 0) != cudaSuccess || per < 1)
    return 3;
  cudaStream_t s = (cudaStream_t)stream;
  if (cudaMemsetAsync(next, 0, sizeof(u32) * (1 + SMC_SLOTS), s) != cudaSuccess) return 3;
  h1um_sm<true><<<2u * (unsigned)per * (unsigned)sms, 32 * WARPS, 0, s>>>((const UCall*)calls, ncalls, M,
                                                                         (const uint8_t*)live, q, den, (u32*)next, cap);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// seed_j's blocks a unit reads: rows' and columns' (K / 512 + 1) each, 32 apiece
extern "C" unsigned kh_noise_blocks(unsigned K) { return 64 * (K / 512 + 1); }
extern "C" unsigned long long kh_ws_bytes(unsigned M, unsigned N, unsigned K) {
  return 64ull * kh_noise_blocks(K) * ((M / 32) * (N / 32));
}

extern "C" int kh_unit_ws(const void* A, const void* B, unsigned M, unsigned N, unsigned K, const void* seeds,
                          const void* live, void* out, void* ws, unsigned J, void* stream) {
  if (K % 512 || !J || !ws) return 2;
  const int m = n_digits((int)(K / 32));
  if (m != 2 && m != 3) return 2;
  const unsigned units = (M / 32) * (N / 32);
  if (!units) return 0;
  cudaStream_t s = (cudaStream_t)stream;
  const unsigned nb = kh_noise_blocks(K);
  const unsigned long long items = (unsigned long long)units * ((nb + 32 * J - 1) / (32 * J));
  h1s_noise<<<(unsigned)((items * 32 + 255) / 256), 256, 0, s>>>((const u64*)seeds, units, N / 32, nb, J,
                                                                  (const uint8_t*)live, (u32*)ws);
  if (m == 2)
    launch_unit<2, true>(units, s, A, B, M, N, K, seeds, live, out, ws);
  else
    launch_unit<3, true>(units, s, A, B, M, N, K, seeds, live, out, ws);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// the one-stream call: kh_unit's (or with ws, kh_unit_ws') receipts of A [rows, K] and B [N, K], and the cells' z and
// y (OneOut) from the same MMAs; step on the device ({parity, m, pad, flags}), M the y rows. ws null: the fused form.
// PART: 0 both launches, 1 the noise only (h1s_noise into ws), 2 the units only (on a ws the noise already wrote)
// band > 1 (kh_one_band; ws only): J10 banded noise (h1j_noise, OneOut::band); mode 1 the units alone (the workspace
// the last mode 0 or 2 call wrote), 2 the noise alone
template <bool PL, int PART = 0>
static int one_call(const void* A, const void* B, unsigned rows, unsigned N, unsigned K, const void* seeds,
                    const void* live, void* out, void* ws, unsigned J, int* z, int* zo, void* y, void* ydump,
                    const float* sa, const float* sw, const int* step, unsigned M, void* stream, unsigned band = 1,
                    int mode = 0) {
#if H1U_SPLIT
  return 2;
#endif
  if (K % 512 || rows % 32 || N % 32 || !z || !y || !step || (ws && !J) || (H1K_RONLY && PL) || (PART && !ws)) return 2;
  if (!band || mode < 0 || mode > 2 || (band > 1 || mode) && !ws) return 2;
  const int m = n_digits((int)(K / 32));
  if (m != 2 && m != 3) return 2;
  const unsigned units = (rows / 32) * (N / 32);
  if (!units) return 0;
  cudaStream_t s = (cudaStream_t)stream;
  const OneOut oo{z, zo, (__nv_bfloat16*)y, (__nv_bfloat16*)ydump, sa, sw, step, M, band};
  if (ws) {
    const bool noise = PART != 2 && mode != 1, run_units = PART != 1 && mode != 2;
    const unsigned nb = kh_noise_blocks(K);
    const unsigned long long items = (unsigned long long)units * ((nb + 32 * J - 1) / (32 * J));
    const unsigned grid = (unsigned)((items * 32 + 255) / 256);
    if (noise && band > 1)
      h1j_noise<<<grid, 256, 0, s>>>((const u64*)seeds, units, N / 32, nb, 32 * (K / 512 + 1), J, band,
                                     (const uint8_t*)live, (u32*)ws);
    else if (noise)
      h1s_noise<<<grid, 256, 0, s>>>((const u64*)seeds, units, N / 32, nb, J, (const uint8_t*)live, (u32*)ws);
    if (run_units && m == 2)
      launch_one<2, true, false, PL>(units, s, A, B, rows, N, K, seeds, live, out, ws, oo);
    else if (run_units)
      launch_one<3, true, false, PL>(units, s, A, B, rows, N, K, seeds, live, out, ws, oo);
  } else if (m == 2) {
    launch_one<2, false, false, PL>(units, s, A, B, rows, N, K, seeds, live, out, nullptr, oo);
  } else {
    launch_one<3, false, false, PL>(units, s, A, B, rows, N, K, seeds, live, out, nullptr, oo);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int kh_one(const void* A, const void* B, unsigned rows, unsigned N, unsigned K, const void* seeds,
                      const void* live, void* out, void* ws, unsigned J, int* z, int* zo, void* y, void* ydump,
                      const float* sa, const float* sw, const int* step, unsigned M, void* stream) {
  return one_call<false>(A, B, rows, N, K, seeds, live, out, ws, J, z, zo, y, ydump, sa, sw, step, M, stream);
}

// kh_one's two launches apart, for a caller that runs the noise on another stream (dither-a -aomn): kh_one_noise is
// its h1s_noise into ws, kh_one_units its units on that ws once written; kh_one's arguments, the same to both
extern "C" int kh_one_noise(const void* A, const void* B, unsigned rows, unsigned N, unsigned K, const void* seeds,
                            const void* live, void* out, void* ws, unsigned J, int* z, int* zo, void* y, void* ydump,
                            const float* sa, const float* sw, const int* step, unsigned M, void* stream) {
  return one_call<false, 1>(A, B, rows, N, K, seeds, live, out, ws, J, z, zo, y, ydump, sa, sw, step, M, stream);
}

extern "C" int kh_one_units(const void* A, const void* B, unsigned rows, unsigned N, unsigned K, const void* seeds,
                            const void* live, void* out, void* ws, unsigned J, int* z, int* zo, void* y, void* ydump,
                            const float* sa, const float* sw, const int* step, unsigned M, void* stream) {
  return one_call<false, 2>(A, B, rows, N, K, seeds, live, out, ws, J, z, zo, y, ydump, sa, sw, step, M, stream);
}

// kh_one with the peel (PL): z = A B^T and y = dequant(z), today's y, the receipts unchanged
extern "C" int kh_one_peel(const void* A, const void* B, unsigned rows, unsigned N, unsigned K, const void* seeds,
                           const void* live, void* out, void* ws, unsigned J, int* z, int* zo, void* y, void* ydump,
                           const float* sa, const float* sw, const int* step, unsigned M, void* stream) {
  return one_call<true>(A, B, rows, N, K, seeds, live, out, ws, J, z, zo, y, ydump, sa, sw, step, M, stream);
}

// J10 banded (measured only; noise=j10-banded, j10b_ref.py's reading of SharedNoise.lean): kh_one and
// kh_one_peel on band-leader rows, band column tiles a band; mode 0 noise then units, 1 the units alone, 2 the noise alone
extern "C" int kh_one_band(const void* A, const void* B, unsigned rows, unsigned N, unsigned K, const void* seeds,
                           const void* live, void* out, void* ws, unsigned J, int* z, int* zo, void* y, void* ydump,
                           const float* sa, const float* sw, const int* step, unsigned M, unsigned band, int mode,
                           void* stream) {
  return one_call<false>(A, B, rows, N, K, seeds, live, out, ws, J, z, zo, y, ydump, sa, sw, step, M, stream, band, mode);
}

extern "C" int kh_one_peel_band(const void* A, const void* B, unsigned rows, unsigned N, unsigned K, const void* seeds,
                                const void* live, void* out, void* ws, unsigned J, int* z, int* zo, void* y, void* ydump,
                                const float* sa, const float* sw, const int* step, unsigned M, unsigned band, int mode,
                                void* stream) {
  return one_call<true>(A, B, rows, N, K, seeds, live, out, ws, J, z, zo, y, ydump, sa, sw, step, M, stream, band, mode);
}

// J10-R (J10R's Definition, `h1tile.j10r_reference`, in the verifier's form, whose credit derives a unit's call and
// blocks at the window's slot M): the call's noise is one stream, h1tile.noise(cseed, Layout(rows, N, K)) at call
// scale, row i's blocks at i RB and column j's at rows RB + j CB, so unit (ri, cj) reads call rows 32 ri .. and
// columns 32 cj .. of it.
// cseed: the call's seed as 8 big-endian words (16 read as h1s_noise reads a unit's). Each unit keeps its own fold keys
// (seeds' sigma_R). ws: kh_j10_ws_bytes(rows, N, K) bytes; kh_one's other arguments.
extern "C" unsigned long long kh_j10_ws_bytes(unsigned rows, unsigned N, unsigned K) {
  return 64ull * (K / 512 + 1) * ((unsigned long long)rows + N);
}

extern "C" int kh_one_j10(const void* A, const void* B, unsigned rows, unsigned N, unsigned K, const void* seeds,
                          const void* cseed, const void* live, void* out, void* ws, unsigned J, int* z, int* zo, void* y,
                          void* ydump, const float* sa, const float* sw, const int* step, unsigned M, void* stream) {
#if H1U_SPLIT
  return 2;
#endif
  if (K % 512 || rows % 32 || N % 32 || !z || !y || !step || !ws || !J || !cseed) return 2;
  const int m = n_digits((int)(K / 32));
  if (m != 2 && m != 3) return 2;
  const unsigned units = (rows / 32) * (N / 32);
  if (!units) return 0;
  const unsigned long long nb = (unsigned long long)(K / 512 + 1) * ((unsigned long long)rows + N);
  if (nb >> 32) return 2;
  cudaStream_t s = (cudaStream_t)stream;
  const OneOut oo{z, zo, (__nv_bfloat16*)y, (__nv_bfloat16*)ydump, sa, sw, step, M};
  const unsigned long long items = (nb + 32ull * J - 1) / (32ull * J);
  h1s_noise<<<(unsigned)((items * 32 + 255) / 256), 256, 0, s>>>((const u64*)cseed, 1, 1, (u32)nb, J, nullptr, (u32*)ws);
  if (m == 2)
    launch_one<2, true, true>(units, s, A, B, rows, N, K, seeds, live, out, ws, oo);
  else
    launch_one<3, true, true>(units, s, A, B, rows, N, K, seeds, live, out, ws, oo);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int kh_fault_shift() { return FAULT_SHIFT; }

// Option A's prover side (partials.py): drawn leaves' lane partials after the draw. L units on gathered operands, unit u
// A rows 32 u .. 32 u + 31 and B rows 32 u .. 32 u + 31 (A and B each [32 L, K] s8: leaf u's 32 A rows and 32 B rows),
// seeds 16 a unit as kh_seeds writes them (seed_j's 8 big-endian words, then sigma_R's). out: L x kh_words(K) u32, the
// leaves' receipts; lr: L x 32 x kh_words(K) u32, each leaf's record: lane l's 2 (KS + m) words at l kh_words(K), lanes
// in order (lane 4 g + q holds rows g + 8 s and columns 2 q + 8 (s >> 1) + (s & 1)), word 2 s the XOR of its 8 chains' y
// after set s and 2 s + 1 of their z, so the XOR over a record's lanes is its receipt. The fused h1u, every unit live.
template <int MD>
static void launch_lanes(unsigned L, cudaStream_t s, const void* A, const void* B, unsigned K, const void* seeds,
                         void* out, const OneOut& oo) {
  carve<h1u<MD, false, false, false, false, true>>();
  h1u<MD, false, false, false, false, true><<<(L + WARPS - 1) / WARPS, 32 * WARPS, 0, s>>>(
      (const int8_t*)A, (const int8_t*)B, 32 * L, 32, K, (const u64*)seeds, nullptr, (u32*)out, nullptr, oo);
}

extern "C" int kh_lanes(const void* A, const void* B, unsigned L, unsigned K, const void* seeds, void* out, void* lr,
                        void* stream) {
  if (K % 512 || !out || !lr || L > (1u << 26)) return 2;
  const int m = n_digits((int)(K / 32));
  if (m != 2 && m != 3) return 2;
  if (!L) return 0;
  OneOut oo{};
  oo.lr = (u32*)lr;
  cudaStream_t s = (cudaStream_t)stream;
  if (m == 2)
    launch_lanes<2>(L, s, A, B, K, seeds, out, oo);
  else
    launch_lanes<3>(L, s, A, B, K, seeds, out, oo);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// J10-R's receipts-only units (h1tile.PouwH1Tile_v5 / v6's receipt, J10R's Definition; measured
// only until the verifier takes the form): a call of M rows (its A slot) has one noise stream, seed_c's at call scale,
// row i's blocks at i RB and column j's at M RB + j CB; unit (ri, cj) at position base + u reads call rows 32 ri .. and
// columns 32 cj .., and folds under its own sigma_R. A call's row tiles are hashed once (h1jr_rows) and its units' column
// blocks by their own lanes (h1umj).
//   kh_seeds_j10r(data, call, base, U, seeds, stream)   unit u's 16 words [seed_c, sigma_R] (seeds.PouwCallSeeds):
//              seeds.call_seed at the call's first position `call`, receipt_seed at base + u; data is kh_seeds' with
//              CALL_SEED_TAG in SEED_TAG's place
//   kh_units_multi_j10r(calls, ncalls, units, M, live, mode, stream)   kh_units_multi's receipts under J10-R: calls
//              ncalls + 1 UCallR (kh_ucallr_bytes() each), each with its workspace rws of
//              kh_j10r_workspace_bytes(M, N, K, 0) bytes (kh_j10r_rows_bytes(M, K) but under H1U_JW); mode 0 both
//              launches, 1 the rows alone, 2 the units alone (on rows a mode 0 or 1 call wrote; with H1U_JX, h1jr_x's
//              row tiles' X and A_d first, from those rows)
//   kh_units_multi_j10r_sm(calls, ncalls, units, M, live, q, den, next, stream)   both launches, the units' on the SMs
//              whose id % den < q alone (h1umj_sm, as kh_units_multi_sm's h1um_sm); kh_units_multi_j10r_smc(.., q, den,
//              cap, next, stream) with at most cap CTAs an SM besides (as kh_units_multi_smc)
//   kh_seeds_j10r_h1r(calls, ncalls, max_units, salt, step_base, seeds, stream)   a served step's seeds from h1r.cu's call
//              table (H1Call, kh_h1rcall_bytes() each; h1r_launch's version 0 under J10-R): every live unit u of call c
//              gets seed_c at the call's first position c.base + *step_base, sigma_R at c.base + *step_base + u
struct UCallR {
  const int8_t* A;
  const int8_t* B;
  const u64* seeds;  // the call's unit 0's 16 words (kh_seeds_j10r)
  u32* out;          // the call's unit 0's receipt
  u32* rws;          // the call's rows' blocks t < M RB, 16 words each (RC: then its columns'; H1U_JAP: then A';
                     // H1U_JX: then X at jx_x_off, A_d at jx_ad_off)
  u32 N, K, u0, md;
};

__global__ void __launch_bounds__(128) h1_seeds_c(const u64* __restrict__ data, u64 call, u64 base, u32 U,
                                                  u64* __restrict__ out) {
  const u32 u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= U) return;
#pragma unroll 1
  for (int tag = 0; tag < 2; ++tag) {
    const u64 pos = tag == 0 ? call : base + u;
    u64 msg[32];
    msg[0] = pos;
#pragma unroll
    for (int w = 0; w < 27; ++w) msg[1 + w] = (w < 3 ? data[3 * tag + w] : data[3 + w]) ^ pos;
    msg[28] = 0x8000000000000000ULL;
    msg[29] = 0;
    msg[30] = 0;
    msg[31] = 224 * 8;
    u64 s[8];
    sha512_words(msg, 2, s);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[16 * (u64)u + 8 * tag + j] = s[j];
  }
}

__device__ __forceinline__ const int8_t* jap_a(const UCallR& c, u32 M, bool rc) {
  return H1U_JAP ? reinterpret_cast<const int8_t*>(c.rws + jap_words(M, c.N, c.K, rc)) : c.A;
}

// call blockIdx.y's row tiles t < M RB of its seed_c stream, the grid's x threads striding over t, the rows of dead row
// blocks skipped.  H1U_JAP: then each warp writes its threads' blocks' A' (every block t but a row's last, the coin
// block) at jap_words(rc): A's 512 bytes from k = 512 (t % RB) of row t / RB plus the block's bits, 16 bytes a lane
__global__ void __launch_bounds__(256) h1jr_rows(const UCallR* __restrict__ calls, u32 M, const uint8_t* __restrict__ live,
                                                 int rc) {
  const UCallR c = calls[blockIdx.y];
  const u32 rb = c.K / 512 + 1, nb = M * rb;
  if (blockIdx.x * blockDim.x >= nb) return;
  u64 seed[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) seed[j] = __ldg(c.seeds + j);
  NoiseMid nm;
  noise_mid(seed, nm);
#if H1U_JAP
  __shared__ u32 bits[256][17];
  const u32 lane = threadIdx.x & 31, w0 = threadIdx.x & ~31u;
  int8_t* const ap = reinterpret_cast<int8_t*>(c.rws + jap_words(M, c.N, c.K, rc));
#endif
#pragma unroll 1
  for (u32 tb = blockIdx.x * blockDim.x; tb < nb; tb += gridDim.x * blockDim.x) {
    const u32 t = tb + threadIdx.x;
    const bool on = t < nb && (!live || live[t / rb / 32]);
    if (on) {
      u64 s[8];
      noise_block(nm, t, s);
      uint4* o = reinterpret_cast<uint4*>(c.rws + 16 * (u64)t);
#pragma unroll
      for (int q = 0; q < 4; ++q)
        st4l<H1S_EVL>(o + q, make_uint4((u32)(s[2 * q] >> 32), (u32)s[2 * q], (u32)(s[2 * q + 1] >> 32), (u32)s[2 * q + 1]));
#if H1U_JAP
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        bits[threadIdx.x][2 * j] = (u32)(s[j] >> 32);
        bits[threadIdx.x][2 * j + 1] = (u32)s[j];
      }
#endif
    }
#if H1U_JAP
    u32 m = __ballot_sync(0xFFFFFFFFu, on && t % rb != rb - 1);
    __syncwarp();
    while (m) {
      const u32 b = __ffs(m) - 1;
      m &= m - 1;
      const u32 tt = tb + w0 + b;
      const u64 at = (u64)(tt / rb) * c.K + 512 * (tt % rb) + 16 * lane;
      // byte i of the lane's 16 is k = 16 lane + i of the block: word lane / 2's bit 16 (lane % 2) + i
      const u32 wv = bits[w0 + b][lane >> 1] >> (16 * (lane & 1));
      const uint4 a = __ldg(reinterpret_cast<const uint4*>(c.A + at));
      st4l<H1S_EVL>(reinterpret_cast<uint4*>(ap + at),
                    make_uint4(add01(a.x, expand4(wv & 15u)), add01(a.y, expand4((wv >> 4) & 15u)),
                               add01(a.z, expand4((wv >> 8) & 15u)), add01(a.w, expand4((wv >> 12) & 15u))));
    }
    __syncwarp();
#endif
  }
}

#if H1U_JAY
// call blockIdx.y's Y (H1U_JAY) at jay_words(rc): a warp a live row i = 8 blockIdx.x + warp, Y[i][rho] =
// sum_sg A'[i][32 sg + rho]; lane l's 16-byte pieces are k = 512 j + 16 l .., so rho 16 (l % 2) + b
__global__ void __launch_bounds__(256) h1jr_y(const UCallR* __restrict__ calls, u32 M, const uint8_t* __restrict__ live,
                                              int rc) {
  const UCallR c = calls[blockIdx.y];
  const u32 w = threadIdx.x >> 5, lane = threadIdx.x & 31, row = 8 * blockIdx.x + w;
  __shared__ int part[8][32][17];
  const bool on = row < M && (!live || live[row / 32]);
  if (on) {
    const int8_t* ap = reinterpret_cast<const int8_t*>(c.rws + jap_words(M, c.N, c.K, rc)) + (u64)row * c.K;
    int s[16];
#pragma unroll
    for (int b = 0; b < 16; ++b) s[b] = 0;
#pragma unroll 1
    for (u32 k = 16 * lane; k < c.K; k += 512) {
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(ap + k));
      const u32 x[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int b = 0; b < 16; ++b) s[b] += (int)(int8_t)(x[b >> 2] >> (8 * (b & 3)));
    }
#pragma unroll
    for (int b = 0; b < 16; ++b) part[w][lane][b] = s[b];
  }
  __syncwarp();
  if (on) {
    int y = 0;
#pragma unroll
    for (u32 l = lane >> 4; l < 32; l += 2) y += part[w][l][lane & 15];
    reinterpret_cast<int*>(c.rws + jay_words(M, c.N, c.K, rc))[32ull * row + lane] = y;
  }
}
#endif

// a units launch's blocks before its units: h1jr_rows (with H1U_JAY, then h1jr_y); rc: the units are h1umjc's
static void jr_rows(const void* calls, unsigned ncalls, unsigned M, const void* live, int rc, cudaStream_t s) {
  // a thread a block up to K = 16384 (33 blocks a row), past it the threads stride
  h1jr_rows<<<dim3((M * 33 + 255) / 256, ncalls), 256, 0, s>>>((const UCallR*)calls, M, (const uint8_t*)live, rc);
#if H1U_JAY
  h1jr_y<<<dim3((M + 7) / 8, ncalls), 256, 0, s>>>((const UCallR*)calls, M, (const uint8_t*)live, rc);
#endif
}

#if H1U_JX
// JX: call blockIdx.y's live row tile blockIdx.x: X = A + b over its rows (16 threads a row, each 16 bytes at a time:
// byte k's dither bit is bit k % 32 of word (k / 32) % 16 of the row's block k / 512, as h1u_unit's rowbits give it), then
// warp 0 packs the row tile's A_d from Y (X's residue sums, rho = k % 32) and the rows' coins (words 0 .. of block K / 512)
template <int MD>
__device__ __forceinline__ void jx_digits(const UCallR& c, u32 M, u32 ri, const int (&Ys)[32][33]) {
  const u32 lane = threadIdx.x, g = lane >> 2, q = lane & 3, rb = c.K / 512 + 1;
  const u32* coins = c.rws + 16ull * (32 * ri) * rb + 16 * (rb - 1);
  int Y[8][4];
#pragma unroll
  for (int t = 0; t < 8; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) Y[t][e] = Ys[16 * (t >> 2) + g + 8 * (e >> 1)][8 * (t & 3) + 2 * q + (e & 1)];
  u32 ad[MD][2][4];
  pack_digits<MD>(Y, g, q, [&](u32 row, int d) { return coins[16ull * rb * row + d]; }, ad);
  u32* o = c.rws + jx_ad_off(M, c.K) / 4 + (u64)ri * JX_AD_WORDS;
#pragma unroll
  for (int d = 0; d < MD; ++d)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int r = 0; r < 4; ++r) o[32 * (8 * d + 4 * mt + r) + lane] = ad[d][mt][r];
}

// H1U_JPC's: warp wi's lane (g, q) writes its slot of row tile mt = wi % 2 in segments sg = wi / 2 (mod 8), rows
// 16 mt + g and 16 mt + 8 + g at k = 32 sg + 8 q .., and adds those rows' sums at rho = 8 q .. into Ys
__device__ __forceinline__ void jx_x_pc(const UCallR& c, u32 M, u32 ri, int (&Ys)[32][33]) {
  const u32 K = c.K, rb = K / 512 + 1, wi = threadIdx.x >> 5, lane = threadIdx.x & 31, q = lane & 3;
  const u32 r0 = 16 * (wi & 1) + (lane >> 2);
  for (u32 i = threadIdx.x; i < 32 * 33; i += blockDim.x) (&Ys[0][0])[i] = 0;
  __syncthreads();
  const u32* const bits0 = c.rws + 16ull * (32 * ri + r0) * rb;
  const u32* const bits1 = bits0 + 16ull * 8 * rb;
  const int8_t* const a0 = c.A + (u64)(32 * ri + r0) * K + 8 * q;
  const int8_t* const a1 = a0 + 8ull * K;
  uint4* const xo = reinterpret_cast<uint4*>(reinterpret_cast<int8_t*>(c.rws) + jx_x_off(M, K) + (u64)(32 * ri) * K +
                                             512 * (wi & 1)) + lane;
  int acc[2][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[0][i] = acc[1][i] = 0;
#pragma unroll 4
  for (u32 sg = wi >> 1; sg < K / 32; sg += 8) {
    const uint2 v0 = __ldg(reinterpret_cast<const uint2*>(a0 + 32 * sg));
    const uint2 v1 = __ldg(reinterpret_cast<const uint2*>(a1 + 32 * sg));
    const u32 w0 = __ldg(bits0 + 16 * (sg / 16) + sg % 16) >> (8 * q);
    const u32 w1 = __ldg(bits1 + 16 * (sg / 16) + sg % 16) >> (8 * q);
    const u32 x[2][2] = {{add01(v0.x, expand4(w0 & 15u)), add01(v0.y, expand4((w0 >> 4) & 15u))},
                         {add01(v1.x, expand4(w1 & 15u)), add01(v1.y, expand4((w1 >> 4) & 15u))}};
    xo[64 * sg] = make_uint4(x[0][0], x[1][0], x[0][1], x[1][1]);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[h][i] += (int)(int8_t)(x[h][i >> 2] >> (8 * (i & 3)));
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    atomicAdd(&Ys[r0][8 * q + i], acc[0][i]);
    atomicAdd(&Ys[r0 + 8][8 * q + i], acc[1][i]);
  }
}

__global__ void __launch_bounds__(512) h1jr_x(const UCallR* __restrict__ calls, u32 M, const uint8_t* __restrict__ live) {
  const UCallR c = calls[blockIdx.y];
  const u32 ri = blockIdx.x;
  if (live && !live[ri]) return;
  __shared__ int Ys[32][33];
  if constexpr (H1U_JXS && H1U_JPC) {
    jx_x_pc(c, M, ri, Ys);
  } else {
    // thread (r, sub) 16 bytes of row r at a time
    const u32 K = c.K, rb = K / 512 + 1, r = threadIdx.x >> 4, sub = threadIdx.x & 15, row = 32 * ri + r;
    const u32* bits = c.rws + 16ull * row * rb;
    const int8_t* a = c.A + (u64)row * K;
    int8_t* x = reinterpret_cast<int8_t*>(c.rws) + jx_x_off(M, K) + (H1U_JXS ? (u64)(32 * ri) * K : (u64)row * K);
    int acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0;
#pragma unroll 4
    for (u32 gi = sub; gi < K / 16; gi += 16) {
      const u32 k0 = 16 * gi;
      const uint4 av = __ldg(reinterpret_cast<const uint4*>(a + k0));
      const u32 w = __ldg(bits + 16 * (k0 / 512) + (k0 / 32) % 16) >> (k0 % 32);
      const u32 xw[4] = {add01(av.x, expand4(w & 15u)), add01(av.y, expand4((w >> 4) & 15u)),
                         add01(av.z, expand4((w >> 8) & 15u)), add01(av.w, expand4((w >> 12) & 15u))};
      if constexpr (H1U_JXS) {
        // bytes 8 q' + 4 hf .. of row r (mt = r / 16, g = r % 8, h = r / 8 % 2) are word h + 2 hf of lane 4 g + q'
        u32* xo = reinterpret_cast<u32*>(x + 512ull * (2 * (gi / 2) + r / 16) + 16 * (4 * (r % 8) + 2 * (gi % 2))) +
                  (r / 8) % 2;
        xo[0] = xw[0];
        xo[2] = xw[1];
        xo[4] = xw[2];
        xo[6] = xw[3];
      } else {
        *reinterpret_cast<uint4*>(x + k0) = make_uint4(xw[0], xw[1], xw[2], xw[3]);
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[i] += (int)(int8_t)(xw[i >> 2] >> (8 * (i & 3)));
    }
    // the 8 threads of a row with one parity of sub hold rho 16 (sub & 1) + i
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 2);
      acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 4);
      acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 8);
    }
    if (sub < 2) {
#pragma unroll
      for (int i = 0; i < 16; ++i) Ys[r][16 * sub + i] = acc[i];
    }
  }
  __syncthreads();
  if (threadIdx.x >= 32) return;
  if (c.md == 3)
    jx_digits<3>(c, M, ri, Ys);
  else
    jx_digits<2>(c, M, ri, Ys);
}
#endif

#if H1U_JW
// JW: call blockIdx.y's column tiles t < N CB (column j's block b at t = j CB + b, stream position M RB + t) into its
// workspace at jw_cb_off, the grid's x threads striding over t; the stream's midstate made once a CTA
__global__ void __launch_bounds__(256) h1jr_wcols(const UCallR* __restrict__ calls, u32 M) {
  const UCallR c = calls[blockIdx.y];
  const u32 cb = c.K / 512 + 1, nb = c.N * cb;
  __shared__ NoiseMid nm;
  if (threadIdx.x == 0) {
    u64 seed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) seed[j] = __ldg(c.seeds + j);
    noise_mid(seed, nm);
  }
  __syncthreads();
  uint4* const o = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(c.rws) + jw_cb_off(M, c.K));
#pragma unroll 1
  for (u32 t = blockIdx.x * blockDim.x + threadIdx.x; t < nb; t += gridDim.x * blockDim.x) {
    u64 s[8];
    noise_block(nm, (u64)M * cb + t, s);
#pragma unroll
    for (int q = 0; q < 4; ++q)
      o[4 * (u64)t + q] = make_uint4((u32)(s[2 * q] >> 32), (u32)s[2 * q], (u32)(s[2 * q + 1] >> 32), (u32)s[2 * q + 1]);
  }
}

// 4 bytes of W' = B + c + U from B's bw, the coins word hv and c's 4 bits c4
__device__ __forceinline__ u32 jw_word(u32 bw, u32 hv, u32 c4) {
  u32 uw = 0;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const u32 x = (hv >> (8 * b)) & 0xFFu;
    uw |= ((x >= 129 ? x - 129 : x) + 64) << (8 * b);
  }
  const u32 cc = expand4(c4);
  return (((bw & 0x7F7F7F7Fu) + (uw & 0x7F7F7F7Fu) + cc) ^ (bw & 0x80808080u)) ^ (~uw & 0x80808080u);
}

// JW: call blockIdx.y's W' = B + c + U [N, K] s8 at jw_w_off, 16 bytes a thread, the grid's x threads striding: bytes
// k0 .. k0 + 15 of column j are segment sg = k0 / 32's rho = k0 % 32 .., c their bits of word sg % 16 of column j's
// block sg / 16 and U + 128 bytes rho .. of its coins block (block K / 512) as h1u_unit's utab, in JBW's form
__global__ void __launch_bounds__(256) h1jr_w(const UCallR* __restrict__ calls, u32 M) {
  const UCallR c = calls[blockIdx.y];
  const u32 K = c.K, cb = K / 512 + 1, kp = K / 16;
  const u64 np = (u64)c.N * kp;
  const u32* const blk = reinterpret_cast<const u32*>(reinterpret_cast<const uint8_t*>(c.rws) + jw_cb_off(M, K));
  int8_t* const w = reinterpret_cast<int8_t*>(c.rws) + jw_w_off(M, c.N, K);
  if constexpr (H1U_JWS && H1U_JPC) {
    // slot p (16 bytes at 16 p): lane p % 32 = 4 g + q of tile pair tp = p / 32 % 2 of segment sg of column tile jb,
    // the 8 bytes at k = 32 sg + 8 q of column 32 jb + 16 tp + g, then of column 32 jb + 16 tp + 8 + g
    const u32 ns = K / 32;
#pragma unroll 1
    for (u64 p = blockIdx.x * (u64)blockDim.x + threadIdx.x; p < np; p += (u64)gridDim.x * blockDim.x) {
      const u32 lane = (u32)p % 32, tp = (u32)(p / 32) % 2, sg = (u32)(p / 64 % ns), jb = (u32)(p / 64 / ns);
      const u32 rho = 8 * (lane % 4);
      u32 ow[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const u32 j = 32 * jb + 16 * tp + 8 * h + lane / 4;
        const u32* const cbk = blk + 16ull * j * cb;
        const u32 cw = __ldg(cbk + 16 * (sg / 16) + sg % 16) >> rho;
        const uint2 bv = __ldg(reinterpret_cast<const uint2*>(c.B + (u64)j * K + 32 * sg + rho));
        ow[2 * h] = jw_word(bv.x, __ldg(cbk + 16 * (cb - 1) + rho / 4), cw & 15u);
        ow[2 * h + 1] = jw_word(bv.y, __ldg(cbk + 16 * (cb - 1) + rho / 4 + 1), (cw >> 4) & 15u);
      }
      reinterpret_cast<uint4*>(w)[p] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
    return;
  }
#pragma unroll 1
  for (u64 p = blockIdx.x * (u64)blockDim.x + threadIdx.x; p < np; p += (u64)gridDim.x * blockDim.x) {
    const u32 j = (u32)(p / kp), k0 = 16 * (u32)(p % kp), sg = k0 / 32, rho = k0 % 32;
    const u32* const cbk = blk + 16ull * j * cb;
    const u32 cw = __ldg(cbk + 16 * (sg / 16) + sg % 16) >> rho;
    const uint4 bv = __ldg(reinterpret_cast<const uint4*>(c.B + (u64)j * K + k0));
    const u32 bw[4] = {bv.x, bv.y, bv.z, bv.w};
    u32 ow[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) ow[i] = jw_word(bw[i], __ldg(cbk + 16 * (cb - 1) + rho / 4 + i), (cw >> (4 * i)) & 15u);
    if (H1U_JWS) {
      // the 8 bytes at k = 8 q' of column j (tile nt = j % 32 / 8, g = j % 8) are lane 4 g + q''s, at byte 8 (nt % 2)
      // of its 16 in tile pair nt / 2
      const u32 col = j % 32, nt = col / 8;
      int8_t* const o = w + (u64)(j / 32) * 32 * K + 1024ull * sg + 512 * (nt / 2) + 8 * (nt % 2);
#pragma unroll
      for (int h = 0; h < 2; ++h)
        *reinterpret_cast<uint2*>(o + 16 * (4 * (col % 8) + rho / 8 + h)) = make_uint2(ow[2 * h], ow[2 * h + 1]);
    } else {
      *reinterpret_cast<uint4*>(w + (u64)j * K + k0) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
  }
}
#endif

// JW: unit gu - u0 of its call in H1U_JWSWZ-row-block bands, column tile by column tile (h1umjc's jswz_unit order)
__device__ __forceinline__ u32 jw_unit(const UCallR& c, u32 w, u32 M) {
  if (!H1U_JW) return w;
  const u32 ncb = c.N / 32, band = (u32)H1U_JWSWZ * ncb, r0 = w / band * (u32)H1U_JWSWZ, r = w % band;
  const u32 rn = min((u32)H1U_JWSWZ, M / 32 - r0);
  return (r0 + r % rn) * ncb + r / rn;
}

__global__ void __launch_bounds__(32 * WARPS, H1U_MINB) h1umj(const UCallR* __restrict__ calls, u32 ncalls, u32 M,
                                                    const uint8_t* __restrict__ live) {
  __shared__ WarpSmem smem[WARPS];
  const u32 wp = threadIdx.x >> 5;
  const u32 gu = blockIdx.x * WARPS + wp;
  if (gu >= calls[ncalls].u0) return;
  u32 lo = 0, hi = ncalls;
  while (hi - lo > 1) {
    const u32 mid = (lo + hi) >> 1;
    if (calls[mid].u0 <= gu) lo = mid;
    else hi = mid;
  }
  const UCallR c = calls[lo];
  if (c.md == 3)
    h1u_unit<3, false, false, false, false, false, true, false, false, false, (bool)H1U_JX>(
        jap_a(c, M, false), c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, jw_unit(c, gu - c.u0, M), smem[wp]);
  else
    h1u_unit<2, false, false, false, false, false, true, false, false, false, (bool)H1U_JX>(
        jap_a(c, M, false), c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, jw_unit(c, gu - c.u0, M), smem[wp]);
}

// h1umj's unit gu, for h1umj_sm (h1umj keeps its own copy, as h1um does of h1um_one)
__device__ __forceinline__ void h1umj_one(const UCallR* __restrict__ calls, u32 ncalls, u32 M,
                                          const uint8_t* __restrict__ live, u32 gu, WarpSmem& S) {
  u32 lo = 0, hi = ncalls;
  while (hi - lo > 1) {
    const u32 mid = (lo + hi) >> 1;
    if (calls[mid].u0 <= gu) lo = mid;
    else hi = mid;
  }
  const UCallR c = calls[lo];
  if (c.md == 3)
    h1u_unit<3, false, false, false, false, false, true, false, false, false, (bool)H1U_JX>(
        jap_a(c, M, false), c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, jw_unit(c, gu - c.u0, M), S);
  else
    h1u_unit<2, false, false, false, false, false, true, false, false, false, (bool)H1U_JX>(
        jap_a(c, M, false), c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, jw_unit(c, gu - c.u0, M), S);
}

// H1U_JCTA = 1 (h1umj_sm): a CTA's warps take WARPS units in turn together, warp w the w-th, as h1umj's grid places them,
// so they read their shared A rows through one SM's L1 rather than each from L2; 0: each warp
// takes its own, and neighbouring units run on any SMs
#ifndef H1U_JCTA
#define H1U_JCTA 0
#endif
#if H1U_JCS && !H1U_JCTA
#error "H1U_JCS tiles the WARPS units a CTA takes together, so it needs H1U_JCTA = 1"
#endif
#if H1U_JCS
// the k-th live (want) or dead row tile of the M / 32, the warp's lanes taking 32 at a time
__device__ __forceinline__ u32 nth_rb(const uint8_t* __restrict__ live, u32 nrb, u32 k, bool want) {
  const u32 lane = threadIdx.x & 31;
  for (u32 b0 = 0; b0 < nrb; b0 += 32) {
    const u32 b = b0 + lane;
    u32 m = __ballot_sync(0xFFFFFFFFu, b < nrb && (!live || live[b] != 0) == want);
    const u32 n = __popc(m);
    if (k < n) {
      for (u32 j = 0; j < k; ++j) m &= m - 1;
      return b0 + __ffs(m) - 1;
    }
    k -= n;
  }
  return 0;
}

// h1umj_one under H1U_JCS: slot gu of its call's tiles (nl live row tiles, r = 1 << lr of them a tile; r = 1 is the
// unit gu itself), its group's buffers in jcb
__device__ __forceinline__ void h1umj_cs(const UCallR* __restrict__ calls, u32 ncalls, u32 M,
                                         const uint8_t* __restrict__ live, u32 gu, u32 nl, u32 lr, WarpSmemJC& S,
                                         u32 (*jcb)[32][17]) {
  u32 lo = 0, hi = ncalls;
  while (hi - lo > 1) {
    const u32 mid = (lo + hi) >> 1;
    if (calls[mid].u0 <= gu) lo = mid;
    else hi = mid;
  }
  const UCallR c = calls[lo];
  const u32 wp = threadIdx.x >> 5, s = gu - c.u0, ncb = c.N / 32, nrb = M / 32, C = WARPS >> lr, k = wp % C;
  u32 u = s;
  if (lr) {
    if (s < nl * ncb) {
      const u32 t = s / WARPS, nct = ncb / C;
      u = nth_rb(live, nrb, ((t / nct) << lr) + wp / C, true) * ncb + C * (t % nct) + k;
    } else {
      const u32 d = s - nl * ncb;
      u = nth_rb(live, nrb, d / ncb, false) * ncb + d % ncb;
    }
  }
  const JShare js{jcb + ((2 * k) << lr), wp / C, lr, 1 + k};
  if (c.md == 3)
    h1u_unit<3, false, false, false, false, false, true, false, false, false, true, WarpSmemJC, true>(
        c.A, c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, u, S, js);
  else
    h1u_unit<2, false, false, false, false, false, true, false, false, false, true, WarpSmemJC, true>(
        c.A, c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, u, S, js);
}
#endif
// h1umj on the SMs whose %smid % den < q alone, its warps taking units from *next in turn (h1um_sm's partition, its last
// CTA's fallback and its CAP too)
template <bool CAP>
__global__ void __launch_bounds__(32 * WARPS, H1U_MINB) h1umj_sm(const UCallR* __restrict__ calls, u32 ncalls, u32 M,
                                                       const uint8_t* __restrict__ live, u32 q, u32 den,
                                                       u32* __restrict__ next, u32 cap) {
#if H1U_JCS
  u32 smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (!sm_keep<CAP, true>(smid, q, den, cap, next)) return;
  __shared__ WarpSmemJC smem[WARPS];
  __shared__ u32 jcb[2 * WARPS][32][17];
  __shared__ u32 claim;
  const u32 wp = threadIdx.x >> 5, lane = threadIdx.x & 31, total = calls[ncalls].u0, nrb = M / 32;
  // r: H1U_JCS live row tiles a tile, else 2, else 1, the most that divides them; more than 1 only where every call's
  // units are its nrb ncb, from a multiple of WARPS, and its column tiles a multiple of WARPS / r
  u32 nl = 0;
  for (u32 b = lane; b < nrb; b += 32) nl += !live || live[b] ? 1u : 0u;
  nl = __reduce_add_sync(0xFFFFFFFFu, nl);
  u32 lr = (H1U_JCS >= 4 && WARPS % 4 == 0 && nl % 4 == 0) ? 2 : (WARPS % 2 == 0 && nl % 2 == 0) ? 1 : 0;
  bool ok = true;
  for (u32 i = lane; i < ncalls; i += 32) {
    const u32 ncb = calls[i].N / 32;
    ok = ok && calls[i].u0 % WARPS == 0 && calls[i + 1].u0 - calls[i].u0 == nrb * ncb && ncb % (WARPS >> lr) == 0;
  }
  if (!__all_sync(0xFFFFFFFFu, ok)) lr = 0;
  for (;;) {
    __syncthreads();
    if (threadIdx.x == 0) claim = atomicAdd(next, (u32)WARPS);
    __syncthreads();
    const u32 g0 = claim, gu = g0 + wp;
    if (g0 >= total) break;
    if (gu >= total) continue;
    h1umj_cs(calls, ncalls, M, live, gu, nl, lr, smem[wp], jcb);
    __syncwarp();
  }
#else
  u32 smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (!sm_keep<CAP, H1U_JCTA != 0>(smid, q, den, cap, next)) return;
  __shared__ WarpSmem smem[WARPS];
  const u32 wp = threadIdx.x >> 5, lane = threadIdx.x & 31, total = calls[ncalls].u0;
  if constexpr (H1U_JCTA) {
    __shared__ u32 claim;
    for (;;) {
      __syncthreads();
      if (threadIdx.x == 0) claim = atomicAdd(next, (u32)WARPS);
      __syncthreads();
      const u32 g0 = claim, gu = g0 + wp;
      if (g0 >= total) break;
      if (gu >= total) continue;
      h1umj_one(calls, ncalls, M, live, gu, smem[wp]);
      __syncwarp();
    }
  } else {
    for (;;) {
      u32 gu = 0;
      if (lane == 0) gu = atomicAdd(next, 1u);
      gu = __shfl_sync(0xffffffffu, gu, 0);
      if (gu >= total) break;
      h1umj_one(calls, ncalls, M, live, gu, smem[wp]);
      __syncwarp();
    }
  }
#endif
}

// h1r.cu's H1Call, field for field (kh_h1rcall_bytes() against its h1r_call_bytes())
struct H1RCall {
  const int8_t* a;
  const int8_t* b;
  long long lda, ldb;
  const u64* roots;
  const u64* root_b;
  const uint8_t* live;
  u64 base;
  int k, ncb, units;
  int seed0;
  long long out0;
};

// one thread a (live unit, seed) of call blockIdx.y: kind 0 seed_c (seeds.call_seed, tag c0..c2, at the call's first
// position), kind 1 sigma_R (receipt_seed, r0..r2, at the unit's), each 224 bytes under SHA-512 with the position XORed
// into every word, into seeds[(seed0 + u) 16 + 8 kind ..]
__global__ void __launch_bounds__(128) h1jr_seeds(const H1RCall* __restrict__ calls, int ncalls, const u64* __restrict__ salt,
                                                  const u64* __restrict__ step_base, u64 c0, u64 c1, u64 c2, u64 r0, u64 r1,
                                                  u64 r2, u64* __restrict__ seeds) {
  if ((int)blockIdx.y >= ncalls) return;
  const H1RCall C = calls[blockIdx.y];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * C.units) return;
  const int u = i >> 1, kind = i & 1;
  if (!C.live[u / C.ncb]) return;
  const u64 call = C.base + (step_base ? *step_base : 0ULL);
  const u64 pos = kind ? call + (u64)u : call;
  u64 msg[32];
  msg[0] = pos;
  msg[1] = (kind ? r0 : c0) ^ pos;
  msg[2] = (kind ? r1 : c1) ^ pos;
  msg[3] = (kind ? r2 : c2) ^ pos;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    msg[4 + j] = salt[j] ^ pos;
    msg[12 + j] = C.roots[j] ^ pos;
    msg[20 + j] = C.root_b[j] ^ pos;
  }
  msg[28] = 0x8000000000000000ULL;
  msg[29] = 0;
  msg[30] = 0;
  msg[31] = 224 * 8;
  u64 s[8];
  sha512_words(msg, 2, s);
  u64* o = seeds + ((u64)(C.seed0 + u) * 2 + kind) * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = s[j];
}

// a 24-byte tag (the name, zero-padded) as three big-endian words
static void tag3w(const char* name, u64 out[3]) {
  unsigned char b[24] = {0};
  for (int i = 0; i < 24 && name[i]; ++i) b[i] = (unsigned char)name[i];
  for (int w = 0; w < 3; ++w) {
    out[w] = 0;
    for (int i = 0; i < 8; ++i) out[w] = (out[w] << 8) | b[8 * w + i];
  }
}

extern "C" unsigned kh_h1rcall_bytes() { return sizeof(H1RCall); }

extern "C" int kh_seeds_j10r_h1r(const void* calls, int ncalls, int max_units, const void* salt, const void* step_base,
                                 void* seeds, void* stream) {
  if (ncalls < 1 || ncalls > 65535 || max_units < 0 || !salt || !seeds) return 2;
  if (!max_units) return 0;
  u64 ct[3], rt[3];
  tag3w("verity/pouw/call-seed/v1", ct);
  tag3w("verity/pouw/receipt/v1", rt);
  h1jr_seeds<<<dim3((2 * max_units + 127) / 128, ncalls), 128, 0, (cudaStream_t)stream>>>(
      (const H1RCall*)calls, ncalls, (const u64*)salt, (const u64*)step_base, ct[0], ct[1], ct[2], rt[0], rt[1], rt[2],
      (u64*)seeds);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int kh_seeds_j10r(const void* data, unsigned long long call, unsigned long long base, unsigned U, void* seeds,
                             void* stream) {
  if (!U) return 0;
  h1_seeds_c<<<(U + 127) / 128, 128, 0, (cudaStream_t)stream>>>((const u64*)data, call, base, U, (u64*)seeds);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" unsigned kh_ucallr_bytes() { return sizeof(UCallR); }
extern "C" unsigned long long kh_j10r_rows_bytes(unsigned M, unsigned K) {
  return H1U_JX ? jx_ad_off(M, K) + 4ull * (M / 32) * JX_AD_WORDS
                : 64ull * M * (K / 512 + 1) + (H1U_JAP ? (unsigned long long)M * K : 0) + (H1U_JAY ? 128ull * M : 0);
}
// the row workspace's int32 word whose bit 0 unitsbench's row fault flips (after the rows, before the units): row 0's
// first block, or with H1U_JAP its A' bytes A'[0][0 .. 3], which the units read instead
extern "C" unsigned long long kh_j10r_row_fault_word(unsigned M, unsigned N, unsigned K, int rc) {
  return H1U_JAP ? jap_words(M, N, K, rc != 0) : 0;
}

// JX: the call's row tiles' X and A_d, from its rows, before its units
static void jx_launch(const void* calls, unsigned ncalls, unsigned M, const void* live, cudaStream_t s) {
#if H1U_JX
  h1jr_x<<<dim3(M / 32, ncalls), 512, 0, s>>>((const UCallR*)calls, M, (const uint8_t*)live);
#endif
#if H1U_JW
  h1jr_wcols<<<dim3(512, ncalls), 256, 0, s>>>((const UCallR*)calls, M);
  h1jr_w<<<dim3(4096, ncalls), 256, 0, s>>>((const UCallR*)calls, M);
#endif
}

// a call's workspace for its rows and, under H1U_JW, its column tiles and W' (kh_j10r_rows_bytes otherwise)
extern "C" unsigned long long kh_j10r_jw_bytes(unsigned M, unsigned N, unsigned K) {
  return H1U_JW ? jw_w_off(M, N, K) + (unsigned long long)N * K : kh_j10r_rows_bytes(M, K);
}
// the build's flags kh_j10r_workspace_bytes dispatches on: 1 H1U_JW, 2 H1U_JRC (a caller refuses a bit it doesn't know)
extern "C" unsigned kh_j10r_flags() { return (H1U_JW ? 1u : 0u) | (H1U_JRC ? 2u : 0u); }

extern "C" int kh_units_multi_j10r(const void* calls, unsigned ncalls, unsigned units, unsigned M, const void* live,
                                   int mode, void* stream) {
  if (!ncalls || ncalls > 65535 || M % 32 || mode < 0 || mode > 2) return 2;
  if (!units) return 0;
  cudaStream_t s = (cudaStream_t)stream;
  if (mode != 2) jr_rows(calls, ncalls, M, live, 0, s);
  if (mode != 1) {
    jx_launch(calls, ncalls, M, live, s);
    carve<h1umj>();
    h1umj<<<(units + WARPS - 1) / WARPS, 32 * WARPS, 0, s>>>((const UCallR*)calls, ncalls, M, (const uint8_t*)live);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// kh_units_multi_j10r's two launches, the units' on q / den of the SMs (h1umj_sm; kh_units_multi_sm's grid and counter)
extern "C" int kh_units_multi_j10r_sm(const void* calls, unsigned ncalls, unsigned units, unsigned M, const void* live,
                                      unsigned q, unsigned den, void* next, void* stream) {
  if (!ncalls || ncalls > 65535 || M % 32 || !q || q > den || den > SMQ_DEN_MAX || !next) return 2;
  if (!units) return 0;
  carve<h1umj_sm<false>>();
  int dev = 0, sms = 0, per = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, h1umj_sm<false>, 32 * WARPS, 0) != cudaSuccess || per < 1)
    return 3;
  cudaStream_t s = (cudaStream_t)stream;
  jr_rows(calls, ncalls, M, live, 0, s);
  jx_launch(calls, ncalls, M, live, s);
  if (cudaMemsetAsync(next, 0, sizeof(u32), s) != cudaSuccess) return 3;
  h1umj_sm<false><<<2u * (unsigned)per * (unsigned)sms, 32 * WARPS, 0, s>>>((const UCallR*)calls, ncalls, M,
                                                                           (const uint8_t*)live, q, den, (u32*)next, 0);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// kh_units_multi_j10r_sm with at most cap CTAs an SM (h1umj_sm<true>, 1 <= cap <= 64): next kh_smc_words() u32 the
// launch zeroes first
extern "C" int kh_units_multi_j10r_smc(const void* calls, unsigned ncalls, unsigned units, unsigned M, const void* live,
                                       unsigned q, unsigned den, unsigned cap, void* next, void* stream) {
  if (!ncalls || ncalls > 65535 || M % 32 || !q || q > den || den > SMQ_DEN_MAX || !cap || cap > 64 || !next) return 2;
  if (!units) return 0;
  carve<h1umj_sm<true>>();
  int dev = 0, sms = 0, per = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, h1umj_sm<true>, 32 * WARPS, 0) != cudaSuccess || per < 1)
    return 3;
  cudaStream_t s = (cudaStream_t)stream;
  jr_rows(calls, ncalls, M, live, 0, s);
  jx_launch(calls, ncalls, M, live, s);
  if (cudaMemsetAsync(next, 0, sizeof(u32) * (1 + SMC_SLOTS), s) != cudaSuccess) return 3;
  h1umj_sm<true><<<2u * (unsigned)per * (unsigned)sms, 32 * WARPS, 0, s>>>((const UCallR*)calls, ncalls, M,
                                                                          (const uint8_t*)live, q, den, (u32*)next, cap);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

#if H1U_JRC
// call blockIdx.y's column tiles M RB + t (t < N CB: column j's block c at t = j CB + c) of its seed_c stream, into its
// workspace at those stream positions, the grid's x threads striding over t; the stream's midstate made once a CTA, in
// shared memory.  H1U_JCOLX is the grid's x a call: a call's blocks are 1.1 (o) to 8 (gate_up) a thread at 128 and
// T = 64, where the calls' CTAs run beside each other, so a larger x evens their work out
#ifndef H1U_JCOLX
#define H1U_JCOLX 128
#endif
__global__ void __launch_bounds__(256) h1jr_cols(const UCallR* __restrict__ calls, u32 M) {
  const UCallR c = calls[blockIdx.y];
  const u32 rb = c.K / 512 + 1, nb = c.N * rb;
  if (blockIdx.x * blockDim.x >= nb) return;
  const u32 t0 = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ NoiseMid nm;
  if (threadIdx.x == 0) {
    u64 seed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) seed[j] = __ldg(c.seeds + j);
    noise_mid(seed, nm);
  }
  __syncthreads();
  if (t0 >= nb) return;
#pragma unroll 1
  for (u32 t = t0; t < nb; t += gridDim.x * blockDim.x) {
    const u64 p = (u64)M * rb + t;
    u64 s[8];
    noise_block(nm, p, s);
    uint4* o = reinterpret_cast<uint4*>(c.rws + 16 * p);
#pragma unroll
    for (int q = 0; q < 4; ++q)
      st4l<H1S_EVL>(o + q, make_uint4((u32)(s[2 * q] >> 32), (u32)s[2 * q], (u32)(s[2 * q + 1] >> 32), (u32)s[2 * q + 1]));
  }
}

// H1U_JSWZ = G (h1umjc; 0 is the launch's order): H1U_SWZ's order for its units, G row tiles at a time column tile by
// column tile, so B's lines and the column tiles are read from DRAM once for G row tiles (unitsbench --j10r at
// T = M = 2048, step_multi_j10rc: G 0 256.8, 8 240.6, 16 236.0, 32 233.1, 64 237.3 ms)
#ifndef H1U_JSWZ
#define H1U_JSWZ 32
#endif
// H1U_JWARPS: h1umjc's and h1jr_ady's warps a block (WARPS unless set; the other kernels' WarpSmem caps WARPS at 8)
#ifndef H1U_JWARPS
#define H1U_JWARPS WARPS
#endif
constexpr int JWARPS = H1U_JWARPS;
// H1U_JSWZQ = PR (with H1U_JSWZ): a block's JWARPS units a patch of PR row tiles by JWARPS / PR column tiles, the
// band's patches row by row down it, then across, so the block's warps share A's lines and B's through L1; a band whose
// row tiles or a call whose column tiles the patch doesn't divide keeps H1U_JSWZ's order
#ifndef H1U_JSWZQ
#define H1U_JSWZQ 0
#endif
static_assert(!H1U_JSWZQ || (H1U_JSWZ && JWARPS % H1U_JSWZQ == 0 && H1U_JSWZ % H1U_JSWZQ == 0),
              "H1U_JSWZQ's patch is PR row blocks of H1U_JSWZ's band by JWARPS / PR column blocks");
__device__ __forceinline__ u32 jswz_unit(const UCallR& c, u32 w, u32 M) {
  if (!H1U_JSWZ) return w;
  const u32 ncb = c.N / 32, band = (u32)H1U_JSWZ * ncb, r0 = w / band * (u32)H1U_JSWZ, r = w % band;
  const u32 rn = min((u32)H1U_JSWZ, M / 32 - r0);
#if H1U_JSWZQ
  constexpr u32 W = JWARPS, PR = H1U_JSWZQ, PC = W / PR;
  if (rn % PR == 0 && ncb % PC == 0) {
    const u32 q = r % W, p = r / W, hp = rn / PR;
    return (r0 + PR * (p % hp) + q % PR) * ncb + PC * (p / hp) + q / PR;
  }
#endif
  return (r0 + r % rn) * ncb + r / rn;
}

// H1U_JRX = 1: h1umjc's and h1jr_ady's warps take WarpSmemX (red_put's sets, no ring): about half WarpSmem's bytes, so
// more blocks fit an SM
#ifndef H1U_JRX
#define H1U_JRX 0
#endif
#if H1U_JRX
using JcSmem = WarpSmemX;
#else
using JcSmem = WarpSmem;
#endif

// unit gu of h1umjc's order, under RC (and H1U_JRY's digits, as h1umjc reads them)
template <class Sm>
__device__ __forceinline__ void h1umjc_one(const UCallR* __restrict__ calls, u32 ncalls, u32 M,
                                           const uint8_t* __restrict__ live, u32 gu, Sm& S) {
  u32 lo = 0, hi = ncalls;
  while (hi - lo > 1) {
    const u32 mid = (lo + hi) >> 1;
    if (calls[mid].u0 <= gu) lo = mid;
    else hi = mid;
  }
  const UCallR c = calls[lo];
  const u32 u = jswz_unit(c, gu - c.u0, M);
  if (c.md == 3)
    h1u_unit<3, false, false, false, false, false, true, true, (bool)H1U_JRY>(
        jap_a(c, M, true), c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, u, S);
  else
    h1u_unit<2, false, false, false, false, false, true, true, (bool)H1U_JRY>(
        jap_a(c, M, true), c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, u, S);
}

#if H1U_JM2
static_assert(!H1U_JCTA, "H1U_JM2's warps take their own pairs");
// h1u_unit's RC units u and u + N / 32 (u's row tile even; A is A', Y the workspace's) in one warp: one load of B's
// segment, one set of its fragments and the column tiles in S0, each row tile's coins and ring in its own S
template <int MD>
__device__ __forceinline__ void h1u_pair(const int8_t* __restrict__ A, const int8_t* __restrict__ B, u32 M, u32 N, u32 K,
                                         const u64* __restrict__ seeds, u32* __restrict__ out, const u32* __restrict__ ws,
                                         const u32 u, WarpSmem& S0, WarpSmem& S1) {
  constexpr int PD = H1U_JPD, NS = PD + H1U_JRING;
  const u32 lane = threadIdx.x & 31;
  const u32 g = lane >> 2, q = lane & 3;
  const u32 ncb = N / 32, ri = u / ncb, cj = u % ncb;
  const u32 ks = K / 32, chunks = ks / 16, rb = chunks + 1, cb = chunks + 1;
  const u32 nwords = 2 * (ks + MD);
  u32* const o0 = out + (u64)u * nwords;
  u32* const o1 = o0 + (u64)ncb * nwords;
  u32 y[2][8], z[2][8];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const u64* sd = seeds + 16 * ((u64)u + p * ncb);
    u64 msg[16] = {TAG0, TAG1, TAG2};
#pragma unroll
    for (int j = 0; j < 8; ++j) msg[3 + j] = __ldg(sd + 8 + j);
    msg[11] = lane;
    msg[12] = 0x8000000000000000ULL;
    msg[13] = 0;
    msg[14] = 0;
    msg[15] = 96 * 8;
    u64 s[8];
    sha512_words(msg, 1, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      y[p][2 * j] = (u32)(s[j] >> 32);
      y[p][2 * j + 1] = (u32)s[j];
      z[p][2 * j] = (u32)(s[4 + j] >> 32);
      z[p][2 * j + 1] = (u32)s[4 + j];
    }
  }
  int T[2][8][4];
#pragma unroll
  for (int p = 0; p < 2; ++p)
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) T[p][t][e] = 0;
  u32 ulo[4], uhi[4];
  const int8_t* Ab = A + (u64)(32 * ri) * K;
  const int8_t* Bb = B + (u64)(32 * cj) * K;
  // segment sg's rows 32 p + 16 mt + g + 8 h and B columns 8 nt + g, 8 bytes each at k = 32 sg + 8 q
  uint2 pa[NS][2][4], pb[NS][4];
  const auto load_seg = [&](u32 sg, uint2(&a)[2][4], uint2(&b)[4]) {
    const u32 kofs = 32 * sg + 8 * q;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      a[0][r] = ld2<H1U_L2HINT>(Ab + (u64)(16 * (r >> 1) + g + 8 * (r & 1)) * K + kofs);
      a[1][r] = ld2<H1U_L2HINT>(Ab + (u64)(32 + 16 * (r >> 1) + g + 8 * (r & 1)) * K + kofs);
      b[r] = ld2e<H1U_L2HINT, H1U_EVF && H1U_EVFU>(Bb + (u64)(8 * r + g) * K + kofs);
    }
  };
#pragma unroll
  for (int d = 0; d < PD; ++d) load_seg(d, pa[d], pb[d]);
#pragma unroll 1
  for (u32 it = 0; it <= chunks; ++it) {
    const u32 blk = it == 0 ? chunks : it - 1;
    {
      const uint4* csrc = reinterpret_cast<const uint4*>(ws + 16 * ((u64)M * rb + (u64)(32 * cj + lane) * cb + blk));
      uint4 cv[4];
#pragma unroll
      for (int w = 0; w < 4; ++w) cv[w] = ld4<H1U_L2HINT_WS>(csrc + w);
      if (it == 0) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const uint4 rv = ld4<H1U_L2HINT_WS>(reinterpret_cast<const uint4*>(ws + 16 * ((u64)(32 * (ri + p) + lane) * rb + blk)));
          WarpSmem& S = p ? S1 : S0;
          S.coin[lane][0] = rv.x;
          if (MD > 2) S.coin[lane][1] = rv.y;
        }
      }
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        S0.colbits[lane][4 * w] = cv[w].x;
        S0.colbits[lane][4 * w + 1] = cv[w].y;
        S0.colbits[lane][4 * w + 2] = cv[w].z;
        S0.colbits[lane][4 * w + 3] = cv[w].w;
      }
    }
    __syncwarp();
    if (it == 0) {
#pragma unroll
      for (int h = 0; h < 8; ++h) {
        const u32 hv = S0.colbits[lane][h];
        u32 w = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const u32 x = (hv >> (8 * b)) & 0xFFu;
          w |= ((x >= 129 ? x - 129 : x) + 64) << (8 * b);
        }
        S0.utab[lane][h] = w;
      }
      __syncwarp();
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        ulo[nt] = S0.utab[8 * nt + g][2 * q];
        uhi[nt] = S0.utab[8 * nt + g][2 * q + 1];
      }
      continue;
    }
    const uint8_t* cbits = reinterpret_cast<const uint8_t*>(&S0.colbits[0][0]);
#pragma unroll 1
    for (u32 j0 = 0; j0 < 16; j0 += NS) {
#pragma unroll
      for (int jj = 0; jj < NS; ++jj) {
        const u32 j = j0 + jj, sg = 16 * blk + j;
        uint2 ca[2][4], cw[4];
        if (H1U_L2PF && sg % 4 == 0 && sg + 16 < ks) {
          prefetch_l2(Ab + (u64)lane * K + 32 * (sg + 16));
          prefetch_l2(Ab + (u64)(32 + lane) * K + 32 * (sg + 16));
          prefetch_l2(Bb + (u64)lane * K + 32 * (sg + 16));
        }
        if (H1U_JRING) {
          if (sg + PD < ks) load_seg(sg + PD, pa[(jj + PD) % NS], pb[(jj + PD) % NS]);
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            ca[0][r] = pa[jj][0][r];
            ca[1][r] = pa[jj][1][r];
            cw[r] = pb[jj][r];
          }
        } else {
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            ca[0][r] = pa[jj][0][r];
            ca[1][r] = pa[jj][1][r];
            cw[r] = pb[jj][r];
          }
          if (sg + PD < ks) load_seg(sg + PD, pa[jj], pb[jj]);
        }
        u32 bf[4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const u32 cb8 = cbits[(8 * nt + g) * 68 + 4 * j + q];
          bf[nt][0] = addw(cw[nt].x, ulo[nt] + expand4(cb8 & 15u));
          bf[nt][1] = addw(cw[nt].y, uhi[nt] + expand4(cb8 >> 4));
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            const u32 af[4] = {ca[p][2 * mt].x, ca[p][2 * mt + 1].x, ca[p][2 * mt].y, ca[p][2 * mt + 1].y};
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) mma_s8(T[p][4 * mt + nt], af, bf[nt]);
          }
        fold(T[0], y[0], z[0]);
        ring_put(S0, j, lane, y[0], z[0]);
        fold(T[1], y[1], z[1]);
        ring_put(S1, j, lane, y[1], z[1]);
      }
    }
    __syncwarp();
    o0[32 * blk + lane] = ring_xor(S0, lane);
    o1[32 * blk + lane] = ring_xor(S1, lane);
    __syncwarp();
  }
  u32 uf[4][2], nf[4][2];
  {
    const uint8_t* ub = reinterpret_cast<const uint8_t*>(&S0.utab[0][0]);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const uint8_t* r = ub + (8 * nt + g) * 36;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const u32 c0 = 16 * hf + 2 * q;
        const u32 by[4] = {r[c0], r[c0 + 1], r[c0 + 8], r[c0 + 9]};
        u32 pu = 0, pn = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          pu |= ((by[b] - 128u) & 0xFFu) << (8 * b);
          pn |= ((128u - by[b]) & 0xFFu) << (8 * b);
        }
        uf[nt][hf] = pu;
        nf[nt][hf] = pn;
      }
    }
  }
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    WarpSmem& S = p ? S1 : S0;
    int Y[8][4];
    const int* ys = reinterpret_cast<const int*>(ws + jay_words(M, N, K, true)) + 32ull * 32 * (ri + p);
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        Y[t][e] = __ldg(ys + 32 * (16 * (t >> 2) + g + 8 * (e >> 1)) + 8 * (t & 3) + 2 * q + (e & 1));
    u32 ad[MD][2][4];
#pragma unroll
    for (int d = 0; d < MD; ++d)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int r = 0; r < 4; ++r) ad[d][mt][r] = 0;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int mt = t >> 2, nt = t & 3;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const u32 row = 16 * mt + g + 8 * (e >> 1), rho = 8 * nt + 2 * q + (e & 1);
        int yv = Y[t][e];
        const int reg = (e >> 1) + 2 * (nt >> 1), sh8 = 8 * (2 * (nt & 1) + (e & 1));
#pragma unroll
        for (int d = 0; d < MD - 1; ++d) {
          const int qv = yv >> 7;
          u32 zd = (S.coin[row][d] >> rho) & 1u;
          if (d == MD - 2 && qv == 127) zd = 0;
          ad[d][mt][reg] |= (((u32)yv & 127u) | (zd << 7)) << sh8;
          yv = qv + (int)zd;
        }
        ad[MD - 1][mt][reg] |= ((u32)yv & 0xFFu) << sh8;
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) mma_s8(T[p][4 * mt + nt], ad[0][mt], nf[nt]);
    fold(T[p], y[p], z[p]);
    ring_put(S, 0, lane, y[p], z[p]);
#pragma unroll
    for (int d = 1; d < MD; ++d) {
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) Y[t][e] = 0;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma_s8(Y[4 * mt + nt], ad[d][mt], uf[nt]);
      fold(Y, y[p], z[p]);
      ring_put(S, d, lane, y[p], z[p]);
    }
  }
  __syncwarp();
  if (lane < 2 * MD) {
    o0[2 * ks + lane] = ring_xor(S0, lane);
    o1[2 * ks + lane] = ring_xor(S1, lane);
  }
}

// h1umjc's pair gp (H1U_JM2, M of whole pairs): units 2 gp, 2 gp + 1 of the calls' order are those of a row-block pair
// and a column tile, h1u_pair's; a pair with a row tile not live runs as its two units
__device__ __forceinline__ void h1umjc_pair_one(const UCallR* __restrict__ calls, u32 ncalls, u32 M,
                                                const uint8_t* __restrict__ live, u32 gp, WarpSmem& S0, WarpSmem& S1) {
  const u32 gu = 2 * gp;
  u32 lo = 0, hi = ncalls;
  while (hi - lo > 1) {
    const u32 mid = (lo + hi) >> 1;
    if (calls[mid].u0 <= gu) lo = mid;
    else hi = mid;
  }
  const UCallR c = calls[lo];
  const u32 ncb = c.N / 32, w = (gu - c.u0) / 2, u = w / ncb * 2 * ncb + w % ncb;
  if (live && !(live[u / ncb] && live[u / ncb + 1])) {
#pragma unroll 1
    for (u32 p = 0; p < 2; ++p) {
      if (c.md == 3)
        h1u_unit<3, false, false, false, false, false, true, true>(jap_a(c, M, true), c.B, M, c.N, c.K, c.seeds, live, c.out,
                                                                   c.rws, OneOut{}, u + p * ncb, S0);
      else
        h1u_unit<2, false, false, false, false, false, true, true>(jap_a(c, M, true), c.B, M, c.N, c.K, c.seeds, live, c.out,
                                                                   c.rws, OneOut{}, u + p * ncb, S0);
      __syncwarp();
    }
    return;
  }
  if (c.md == 3)
    h1u_pair<3>(jap_a(c, M, true), c.B, M, c.N, c.K, c.seeds, c.out, c.rws, u, S0, S1);
  else
    h1u_pair<2>(jap_a(c, M, true), c.B, M, c.N, c.K, c.seeds, c.out, c.rws, u, S0, S1);
}
#endif
// H1U_JM2's work items of a launch of units (pairs at an M of whole pairs) and each warp's WarpSmem
__host__ __device__ __forceinline__ u32 jm2_items(u32 units, u32 M) { return H1U_JM2 && M % 64 == 0 ? units / 2 : units; }
constexpr int JM2_SM = H1U_JM2 ? 2 : 1;
static_assert(!(H1U_JM2 && (H1U_JRX || H1U_JSWZQ)), "H1U_JM2's pairs take WarpSmem and jm2_items' order, not H1U_JRX's WarpSmemX or H1U_JSWZQ's patches");

// h1umjc on the SMs whose %smid % den < q alone, its warps taking units from *next in turn (h1umj_sm's partition, its
// last CTA's fallback, its CAP and its H1U_JCTA: a CTA's warps take neighbouring units of jswz_unit's order, the row
// blocks of one column tile at decode, so they read its column tiles and B's lines through one SM's L1)
template <bool CAP>
__global__ void __launch_bounds__(32 * WARPS, H1U_MINB) h1umjc_sm(const UCallR* __restrict__ calls, u32 ncalls, u32 M,
                                                        const uint8_t* __restrict__ live, u32 q, u32 den,
                                                        u32* __restrict__ next, u32 cap) {
  u32 smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (!sm_keep<CAP, H1U_JCTA != 0>(smid, q, den, cap, next)) return;
  __shared__ JcSmem smem[JM2_SM * WARPS];
  const u32 wp = threadIdx.x >> 5, lane = threadIdx.x & 31, total = jm2_items(calls[ncalls].u0, M);
  if constexpr (H1U_JCTA) {
    __shared__ u32 claim;
    for (;;) {
      __syncthreads();
      if (threadIdx.x == 0) claim = atomicAdd(next, (u32)WARPS);
      __syncthreads();
      const u32 g0 = claim, gu = g0 + wp;
      if (g0 >= total) break;
      if (gu >= total) continue;
      h1umjc_one(calls, ncalls, M, live, gu, smem[wp]);
      __syncwarp();
    }
  } else {
    for (;;) {
      u32 gu = 0;
      if (lane == 0) gu = atomicAdd(next, 1u);
      gu = __shfl_sync(0xffffffffu, gu, 0);
      if (gu >= total) break;
#if H1U_JM2
      if (M % 64 == 0) h1umjc_pair_one(calls, ncalls, M, live, gu, smem[2 * wp], smem[2 * wp + 1]);
      else
#endif
        h1umjc_one(calls, ncalls, M, live, gu, smem[JM2_SM * wp]);
      __syncwarp();
    }
  }
}

// h1umj with its units' column tiles read (RC)
__global__ void __launch_bounds__(32 * JWARPS, H1U_MINB) h1umjc(const UCallR* __restrict__ calls, u32 ncalls, u32 M,
                                                      const uint8_t* __restrict__ live) {
  __shared__ JcSmem smem[JM2_SM * JWARPS];
  const u32 wp = threadIdx.x >> 5;
  const u32 gu = blockIdx.x * JWARPS + wp;
  if (gu >= jm2_items(calls[ncalls].u0, M)) return;
#if H1U_JM2
  if (M % 64 == 0) {
    h1umjc_pair_one(calls, ncalls, M, live, gu, smem[2 * wp], smem[2 * wp + 1]);
    return;
  }
#endif
  u32 lo = 0, hi = ncalls;
  while (hi - lo > 1) {
    const u32 mid = (lo + hi) >> 1;
    if (calls[mid].u0 <= gu) lo = mid;
    else hi = mid;
  }
  const UCallR c = calls[lo];
  const u32 u = jswz_unit(c, gu - c.u0, M);
  if (c.md == 3)
    h1u_unit<3, false, false, false, false, false, true, true, (bool)H1U_JRY>(
        jap_a(c, M, true), c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, u, smem[JM2_SM * wp]);
  else
    h1u_unit<2, false, false, false, false, false, true, true, (bool)H1U_JRY>(
        jap_a(c, M, true), c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, u, smem[JM2_SM * wp]);
}

#if H1U_JRY
// RY's digits: call blockIdx.y's row tile blockIdx.x JWARPS + warp, by its column tile 0's unit under YO
__global__ void __launch_bounds__(32 * JWARPS) h1jr_ady(const UCallR* __restrict__ calls, u32 M,
                                                      const uint8_t* __restrict__ live) {
  __shared__ JcSmem smem[JWARPS];
  const u32 wp = threadIdx.x >> 5;
  const u32 ri = blockIdx.x * JWARPS + wp;
  if (ri >= M / 32) return;
  const UCallR c = calls[blockIdx.y];
  if (c.md == 3)
    h1u_unit<3, false, false, false, false, false, true, true, true, true>(
        c.A, c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, ri * (c.N / 32), smem[wp]);
  else
    h1u_unit<2, false, false, false, false, false, true, true, true, true>(
        c.A, c.B, M, c.N, c.K, c.seeds, live, c.out, c.rws, OneOut{}, ri * (c.N / 32), smem[wp]);
}
#endif

// a call's workspace under kh_units_multi_j10rc: its M RB row tiles, then its N CB column tiles, 16 words each, then
// under H1U_JRY its M / 32 row tiles' digits (h1jr_ady), MD KiB each; and never less than kh_j10r_rows_bytes (H1U_JX's
// X and A_d), since dither_so runs kh_units_multi_j10r on the same workspace below its J10-RC t
extern "C" unsigned long long kh_j10r_ws_bytes(unsigned M, unsigned N, unsigned K) {
  const unsigned long long rc = 64ull * ((unsigned long long)M + N) * (K / 512 + 1) +
                                (H1U_JAP ? (unsigned long long)M * K : 0) + (H1U_JAY ? 128ull * M : 0) +
                                (H1U_JRY ? 96ull * M : 0);
  const unsigned long long rows = kh_j10r_rows_bytes(M, K);
  return rows > rc ? rows : rc;
}

// kh_units_multi_j10r with each call's column tiles hashed once a call (h1jr_cols) beside its rows (h1jr_rows), and the
// units reading both (h1umjc): calls' rws of kh_j10r_ws_bytes(M, N, K) bytes; mode 0 the blocks and the units, 1 the
// blocks alone, 2 the units alone (on blocks a mode 0 or 1 call wrote; under H1U_JRY the row tiles' digits first)
extern "C" int kh_units_multi_j10rc(const void* calls, unsigned ncalls, unsigned units, unsigned M, const void* live,
                                    int mode, void* stream) {
  if (!ncalls || ncalls > 65535 || M % 32 || mode < 0 || mode > 2) return 2;
  if (!units) return 0;
  cudaStream_t s = (cudaStream_t)stream;
  if (mode != 2) {
    jr_rows(calls, ncalls, M, live, 1, s);
    h1jr_cols<<<dim3(H1U_JCOLX, ncalls), 256, 0, s>>>((const UCallR*)calls, M);
  }
  if (mode != 1) {
#if H1U_JRY
    h1jr_ady<<<dim3((M / 32 + JWARPS - 1) / JWARPS, ncalls), 32 * JWARPS, 0, s>>>((const UCallR*)calls, M,
                                                                                (const uint8_t*)live);
#endif
    carve<h1umjc>();
    h1umjc<<<(jm2_items(units, M) + JWARPS - 1) / JWARPS, 32 * JWARPS, 0, s>>>((const UCallR*)calls, ncalls, M,
                                                                     (const uint8_t*)live);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// kh_units_multi_j10rc's blocks (and H1U_JRY's digits), then its units on q / den of the SMs (h1umjc_sm;
// kh_units_multi_j10r_sm's grid and counter): the blocks' launches take any SM, as kh_units_multi_j10r_sm's rows do
extern "C" int kh_units_multi_j10rc_sm(const void* calls, unsigned ncalls, unsigned units, unsigned M, const void* live,
                                       unsigned q, unsigned den, void* next, void* stream) {
  if (!ncalls || ncalls > 65535 || M % 32 || !q || q > den || den > SMQ_DEN_MAX || !next) return 2;
  if (!units) return 0;
  carve<h1umjc_sm<false>>();
  int dev = 0, sms = 0, per = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, h1umjc_sm<false>, 32 * WARPS, 0) != cudaSuccess || per < 1)
    return 3;
  cudaStream_t s = (cudaStream_t)stream;
  jr_rows(calls, ncalls, M, live, 1, s);
  h1jr_cols<<<dim3(H1U_JCOLX, ncalls), 256, 0, s>>>((const UCallR*)calls, M);
#if H1U_JRY
  h1jr_ady<<<dim3((M / 32 + JWARPS - 1) / JWARPS, ncalls), 32 * JWARPS, 0, s>>>((const UCallR*)calls, M,
                                                                              (const uint8_t*)live);
#endif
  if (cudaMemsetAsync(next, 0, sizeof(u32), s) != cudaSuccess) return 3;
  h1umjc_sm<false><<<2u * (unsigned)per * (unsigned)sms, 32 * WARPS, 0, s>>>((const UCallR*)calls, ncalls, M,
                                                                            (const uint8_t*)live, q, den, (u32*)next, 0);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
#endif

// a call's workspace in this build at M rows, the size its caller allocates: with rc, kh_units_multi_j10rc's
// (kh_j10r_ws_bytes, which kh_units_multi_j10r may run on too; 0 without H1U_JRC), else kh_units_multi_j10r's and its
// _sm and _smc forms' (kh_j10r_jw_bytes: H1U_JW's column tiles and W' after the rows). Monotone in M, so a workspace
// of T rows serves any M <= T
extern "C" unsigned long long kh_j10r_workspace_bytes(unsigned M, unsigned N, unsigned K, int rc) {
#if H1U_JRC
  if (rc) return kh_j10r_ws_bytes(M, N, K);
#endif
  return rc ? 0 : kh_j10r_jw_bytes(M, N, K);
}
