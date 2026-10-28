// The adopted design's edited GEMM on sm_120a (RTX PRO 6000), as close to plain int8 as we can get it.
//
// A linear Y = X W^T (X m x k activations, W n x k weights capped at ±63) served as the adopted design's noised product:
// the served operand is the dithered X = clamp(floor(x / s), -127, 126) + b, b a fresh coin bit an entry, and the weights
// carry U in {±1..±64} (n x 32) peeled on the activations' side: W' = W + U Id^T (W'_jk = W_jk + U_{j, k mod 32}),
// C = X W'^T on the tensor cores, Y = C - S U^T with S = X Id (S_ic = sum_t X_{i, 32 t + c}, |S| <= 128 k / 32). The
// prover commits C's stride-32 running sums by duo receipts (fgemm.cu's; benchmarks/e2e_slice/derive.py's tile_receipts).
// Noise from a 256-bit per-call key (sha_pass.cuh's Sig, SHA-512(key || BE64(tag << 56 | j))): U as lpgemm.cu's LP_UW 64
// (tag 6, block j = rows 2j, 2j + 1, entry (row % 2) 32 + l digest byte, u = (v & 63) + 1, negated if v & 64); b in
// sha_noise.cuh's NS 1 patch layout under tag 7 (block p = (row / 4) (k / 128) + col / 128, entry e = (row % 4) 128 +
// col % 128, b = bit e % 8 of digest byte e / 8).
// Kernels:
//   quant<false>, quant<true>  the activation quantizer from bf16: plain (round) and dithered (floor + b, with S's limbs
//                    into the tail rows P_A = [S_1 | S_2 | S_0 | 0], S = 256 (S_1 + S_2) + S_0, S_2 = 0 at k <= 4096), so
//                    the dither is priced against the quantizer plain serving runs anyway (a warp's 4 rows hash a chunk's
//                    32 patches at once, a chunk 4096 columns)
//   wedit            W' = W + U and the tail rows P_B = [U | U | U | 0], U's hashes in each block of 8 rows
//   qwe<ORD>         quant<true> and wedit in one launch (block orders: the quantizer's first, the edit's, interleaved)
//   e_*, d64_*, x_*, x64_*, x13_*, t*_*  the GEMM: lpgemm.cu's 16-warp 128 x 128 TMA mainloop (fgemm.cu's ML) with the receipts'
//                    absorbs interleaved with the MMAs, and the peel as a 2-atom tail (acc -= 256 S_1 U^T + S_0 U^T; with
//                    S_2's atom a 3-atom tail at k > 4096; the rows t_duo32's 4-atom tail reads). Receipts (EL's RK): duo
//                    every 32 steps (e_*: t_duo32's), duo every 64 (d64_*), the r9 XOR-in fold every 32 (x_*) and every
//                    64 (x64_*), r9's XOR-in with rotations (13, 19) (x13_*), and the tensor-byte
//                    receipts TB4, TB4C, TB2W, TB1W (t4_*, t4c_*, t2w_*, t1w_*: fgemm.cu's), and the fold-cost sweep's
//                    timing variants at r = 1, 1.5, 2, 2.5, 4 ops a word (f10_*, ..., f40_*; FR10's). Interleaving: the absorb of
//                    a running sum runs LAG MMAs after its tile's MMA (each value is still in its accumulator then, so the
//                    absorb order and the receipts are the unscheduled ones), and with TIE every TIE-th MMA's accumulator
//                    is xored with (z & zero), z the state the last absorb wrote and zero a kernel argument that is 0, so
//                    ptxas can't hoist a run of MMAs ahead of the absorbs (without it, it issues IMMA NOP IMMA NOP ... and
//                    then the absorbs in a burst). The refill's issuer (thread 0) computes its tile's place once a tile,
//                    not at every refill
// Modes:
//   egemm gate DIR M N K SEED       every arm once on poisoned buffers: each GEMM's output against cuBLASLt's X W^T, each
//                                   duo32 kernel's receipts against t_duo32's and each other kernel's against its kind's
//                                   first, a no-write control; tiles (every kernel's receipts) and key_ok_r9's
//                                   device answers on its boundary words to DIR for egemm_ref.py
//   egemm bench REPS M N K          hot items (about 20 ms each) interleaved across EG_ARMS, one JSON line
//   egemm il REPS ARM BASE M N K    per call, ARM and BASE alternating (the slice's timing; EG_COLD=1: L2 flushed
//                                   before each call), one JSON line
//   egemm one ARM M N K             three warm-up calls, then one between cudaProfilerStart and Stop (for ncu)
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a egemm.cu -lcuda -lcublasLt -lnvidia-ml
#define LP_LIB
#define LP_NO_ROUTES
#define LP_UW 64
#include "../masked_gemm/lpgemm.cu"

#include <functional>

constexpr uint32_t DITHER_TAG = 7;
using MLW = ML<128, 128, 32, 3>;

// The fold's scalar-only dial: the served GEMM's tensor pipe is already busy with the useful IMMA, so the mandatory
// per-word work must ride the idle scalar issue, not filler MMAs. EG_ROUNDS absorbs each committed word's four s32
// limbs R times through the XOR-in(13,19) fold (RK == XIN/XIN64/XIN13), no extra MMAs, so R is the one cost parameter
// and the tensor time is untouched. EG_ROUNDS == 1 (the default) is byte-identical to the fold of record. The loop runs
// #pragma unroll 1 so the SASS op-count is constant across R (one dial); the honest kernel pins a fixed R and lets
// ptxas unroll it. PROFILE sweeps the knee; FOLD measures correctness and cost a round once.
#ifndef EG_ROUNDS
#define EG_ROUNDS 1
#endif
static_assert(EG_ROUNDS >= 1 && EG_ROUNDS <= 64, "EG_ROUNDS in 1..64");

// The receipt kinds beyond fgemm.cu's DUO, TB4, TB4C, TB2W and TB1W (NONE: no receipts). DUO64: duo on every second
// running sum only (the odd steps', so derive.py's tile_receipts on Ts[1::2]). XIN: the r9 XOR-in fold (pfill.cuh's NC
// 4 under -DPFILL_XIN), three ops a pair: y = (rotl(y, 7) ^ d0) a0 + d1, then y = (rotl(y, 19) ^ d2) a1 + d3 on an
// accumulator's four words, tile t into chain t % 4. XIN64: XIN on the odd steps' running sums only (as DUO64). XIN and
// XIN64 are timing references, not receipts: rotl 7 puts a bad merge ratio at ±128, an int8 ratio, at the step
// boundary. XIN13, the served candidate: the r9 XOR-in, y = (rotl(y, 13) ^ d0) a0 + d1, then y = (rotl(y, 19) ^ d2) a1
// + d3 (receipt_folds_r9.xorin_step), keys under key_ok_r9, chain c of thread ctid of tile t starting at pfill's y =
// pos = (32 w + lane) 4 + c with the output's warps numbered in order (w = 16 t + warp, so pos = (t NCT + ctid) 4 + c).
// Negative, so fgemm.cu's is_tb is false for them
constexpr int DUO64 = -1, XIN = -2, XIN64 = -3, XIN13 = -4;
// XIN's multipliers: pfill.cuh's key rule (odd, more than 2^8 from 0 mod 2^24, not ±1 mod 32) on the words mix32(seed ^
// (0x5A5A0000 + i)), i = 0, 1, ..., the first two that pass (the r9 reference draws them per tile from a SHA block;
// either way two registers); XIN13's the first two that pass key_ok_r9
__host__ __device__ inline bool xin_ok(uint32_t a) {
  const uint32_t m = a & 0xFFFFFFu, r = a & 31u;
  return (a & 1u) && m > 256u && m < (1u << 24) - 256u && r != 1u && r != 31u;
}
// receipt_folds_r9.key_ok_r9 (pfill.cuh's pf_key_ok_r9): xin_ok, and neither a nor -a an int8 ratio mod 2^30, i.e.
// a b mod 2^30 outside [-128, 128] for every b in 1..128
__host__ __device__ inline bool xin_ok_r9(uint32_t a) {
  if (!xin_ok(a)) return false;
#ifdef __CUDA_ARCH__
  // a warp-uniform and all 32 lanes active: lane l tries b = 4 l + 1 .. 4 l + 4
  const uint32_t b0 = 4u * (threadIdx.x & 31u);
  bool bad = false;
#pragma unroll
  for (uint32_t i = 1; i <= 4; i++) bad |= (((b0 + i) * a + 128u) & 0x3FFFFFFFu) <= 256u;
  return !__any_sync(0xFFFFFFFFu, bad);
#else
  uint32_t t = 0;
  for (int b = 1; b <= 128; b++) {
    t += a;
    if (((t + 128u) & 0x3FFFFFFFu) <= 256u) return false;
  }
  return true;
#endif
}
template <bool R9>
__host__ __device__ inline uint2 xin_keys(uint32_t seed) {
  uint2 k;
  int n = 0;
  for (uint32_t i = 0; n < 2; i++) {
    const uint32_t a = mix32(seed ^ (0x5A5A0000u + i));
    if (R9 ? xin_ok_r9(a) : xin_ok(a)) {
      if (n == 0) k.x = a;
      else k.y = a;
      n++;
    }
  }
  return k;
}
// out of line: inlined, the vote costs the GEMM's mainloop spill loads
__host__ __device__ __noinline__ uint2 xin_keys_r9(uint32_t seed) { return xin_keys<true>(seed); }

// The fold-cost sweep's kinds: duo's chains (tile t's word e into chain
// 4 (t % 2) + e), keys and finish with r scalar ops a word, timing variants and not receipts. A is y ^ p ^ kx (LOP3), M
// y a + p (IMAD), AM (y ^ p ^ kx) a + z (LOP3, IMAD; duo without its rotate-add), D duo (LOP3, IMAD, LEA.HI). FR10 (r = 1):
// M and A alternating along each chain, so no two XORs meet and fuse into one LOP3; FR15: AM on words 0 and 2, FR10's op
// on 1 and 3; FR20: AM; FR25: D on words 0 and 2, AM on 1 and 3; FR40: D, then one more dependent op on z (z b + y on words
// 0 and 2, z ^ y ^ k2 on 1 and 3)
constexpr int FR10 = -10, FR15 = -15, FR20 = -20, FR25 = -25, FR40 = -40;
__host__ __device__ constexpr bool is_fr(int k) { return k == FR10 || k == FR15 || k == FR20 || k == FR25 || k == FR40; }
template <int RK, int T, int E> DEV void fr_word(St& s, uint32_t p) {
  constexpr uint32_t a = key_mul(0, T), kx = key_xor(0, T), b = key_mul(2, T), k2 = key_xor(2, T);
  constexpr bool even = E % 2 == 0, alt = (T / 2 + E) % 2 == 0;
  if constexpr (RK == FR10 || (RK == FR15 && !even)) {
    if constexpr (alt) s.y = s.y * a + p;
    else s.y = s.y ^ p ^ kx;
  } else if constexpr (RK == FR20 || RK == FR15 || (RK == FR25 && !even)) {
    s.y = (s.y ^ p ^ kx) * a + s.z;
  } else {
    duo<DUO, 0, T>(s, p);
    if constexpr (RK == FR40) {
      if constexpr (even) s.z = s.z * b + s.y;
      else s.z = s.z ^ s.y ^ k2;
    }
  }
}
// Verified filler: duo folds the end of a dependent chain of x lane ops from the word (t = t a_i, IMAD with a_i odd, and
// t = t ^ k_i, LOP3, alternating and ending on an IMAD so no constant XOR meets duo's LOP3), so a word costs x + 3 ops,
// no op can be dropped or merged, and the chain is a bijection, so the fold binds the word as duo does. FX45 (x = 4.5):
// 4 ops on words 0 and 2, 5 on 1 and 3; FX05 (x = 0.5): one op on words 1 and 3; FX15: one on 0 and 2, two on 1 and 3.
// Timing variants, not receipts
constexpr int FX05 = -150, FX1 = -101, FX15 = -115, FX2 = -102, FX3 = -103, FX45 = -145, FX8 = -108, FX12 = -112, FX24 = -124;
__host__ __device__ constexpr bool is_fx(int k) {
  return k == FX05 || k == FX1 || k == FX15 || k == FX2 || k == FX3 || k == FX45 || k == FX8 || k == FX12 || k == FX24;
}
__host__ __device__ constexpr int fx_ops(int k, int e) {
  return k == FX05 ? e % 2 : k == FX1 ? 1 : k == FX15 ? 1 + e % 2 : k == FX2 ? 2 : k == FX3 ? 3 : k == FX45 ? 4 + e % 2
         : k == FX8 ? 8 : k == FX12 ? 12 : 24;
}
template <int RK, int T, int E, int I = 0> DEV uint32_t fx_chain(uint32_t t) {
  constexpr int n = fx_ops(RK, E);
  if constexpr (I == n) {
    return t;
  } else {
    constexpr int c = (4 * T + E) * 32 + I;
    if constexpr ((n - I) % 2 == 1) t = t * key_mul(1, c);
    else t = t ^ key_xor(1, c);
    return fx_chain<RK, T, E, I + 1>(t);
  }
}

// d ^= z & zero in one LOP3 (LUT 0x78 = a ^ (b & c); ptxas otherwise splits it into an AND and an XOR), or (FMA) d += z zero
// in one IMAD, on the FMA pipe
template <bool FMA> DEV void tie(int& d, uint32_t z, uint32_t zero) {
  if constexpr (FMA) asm("mad.lo.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(z), "r"(zero));
  else asm("lop3.b32 %0, %0, %1, %2, 0x78;" : "+r"(d) : "r"(z), "r"(zero));
}

// mbar_wait_warp with try_wait's suspend-time hint (CUTLASS's 0x989680 ns): a warp whose stage hasn't landed sleeps until
// the phase completes instead of retrying, which takes issue slots from the warps still working
DEV void mbar_sleep_warp(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n EG_SLEEP_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n @!p bra EG_SLEEP_%=;\n}\n" ::"r"(bar),
      "r"(parity), "r"(0x989680u)
      : "memory");
  __syncwarp();
}

// RK: the receipt kind. A receipt's unit is the tiles one absorb reads (one tile; TB2W a pair, TB1W a row block), unit j
// named by its last tile; LAG: unit j is absorbed after MMA j + LAG (past the atom's end, MMA j + LAG - 8 of the next), or
// with LAG -1 after MMA min(2 j + 2, j + 7) (DUO64's spread: an absorb every second MMA over a receipting atom and the
// next). TIE: a tie before every |TIE|-th MMA (0: none; negative: the IMAD tie) on the state the last absorb wrote.
// SLEEP: a consumer waits for its stage with mbar_sleep_warp. UF: UF instructions a MMA after each atom on two pairs of
// dependent warp-uniform chains from the seed, folded into the receipt at the tile's end, the uniform filler's stand-in;
// ptxas keeps them on the vector datapath unless USH broadcasts the chains from lane 0 (__shfl_sync, which ptxas reads as
// warp-uniform) before each atom's (1) or each tile's (2). BAT: problems stacked in rows, 2^bsh row tiles of A each against
// their own tiles_n row tiles of B (d2sgemm.cu's Q K^T, a problem a head)
template <int RK, int LAG, int TIE, bool SLEEP = false, int UF = 0, int USH = 0, bool BAT = false> struct EL : MLW {
  static_assert(LAG >= -1 && LAG < MI * NI, "an absorb runs before its accumulator's next MMA");
  static constexpr bool R = RK != NONE, TB = is_tb(RK), XF = RK == XIN || RK == XIN64 || RK == XIN13,
                        HALF = RK == DUO64 || RK == XIN64;
  static constexpr int ROT0 = RK == XIN13 ? 13 : 7;   // the XOR-in fold's first rotation
  static constexpr int B = tb_base(RK);
  static constexpr int NKR = XF ? 2 : TB ? NK<RK> : 1;   // key registers: XIN's a0, a1; a TB kind's B fragments
  using MLW::MLW;
  static constexpr int NST = (SMEM - 1024) / (STAGE + 16);   // ML's STAGES
  static_assert(NST == 3, "MLW's 3 stages");
  // raster's tile place with group 8 (any-M: esetup needs only m % 128 == 0, so tiles_m is any count and the last group is
  // ragged, tiles_m % 8 rows), without its divisions, whose integer-division seed ptxas sometimes schedules so the SASS
  // gate can't match it: a place keeps its group row blk and its rank r in the group, advanced by subtraction. place_step
  // still steps a group as 8 * tiles_n ranks, which the ragged last group's ranks stay below, so it never over-advances;
  // place_tile maps the ragged group's rank by the row count a compile-time divisor in a switch picks, so ptxas uses a
  // magic multiply, not the division seed
  struct Place { int blk, r; };
  DEV void place_step(Place& p, int d) const {
    p.r += d;
    while (p.r >= 8 * tiles_n) { p.r -= 8 * tiles_n; p.blk++; }
  }
  DEV static void place_tile(const Place& p, int tiles_m, int& tm, int& tn) {
    const int full = tiles_m >> 3, rem = tiles_m & 7;
    if (rem == 0 || p.blk < full) { tm = 8 * p.blk + (p.r & 7); tn = p.r >> 3; return; }
    int col, row;   // the ragged last group of rem rows (1 .. 7): col = r / rem, row = r % rem by a constant divisor a case
    switch (rem) {
      case 1: col = p.r; row = 0; break;
      case 2: col = p.r >> 1; row = p.r & 1; break;
      case 3: col = p.r / 3; row = p.r - col * 3; break;
      case 4: col = p.r >> 2; row = p.r & 3; break;
      case 5: col = p.r / 5; row = p.r - col * 5; break;
      case 6: col = p.r / 6; row = p.r - col * 6; break;
      default: col = p.r / 7; row = p.r - col * 7; break;
    }
    tm = 8 * full + row; tn = col;
  }
  // ML::issue with the refill's tile place kept: it recomputes it (raster's divisions) at every refill, in warp 0, which
  // every warp then waits for at the next refill
  int ipt = -1, itm = 0, itn = 0, bsh = 31;
  Place ipl{0, 0};
  DEV void issue() {
    if (pt >= tiles_m * tiles_n) return;
    if (pt != ipt) { place_step(ipl, ipt < 0 ? pt : pt - ipt); place_tile(ipl, tiles_m, itm, itn); ipt = pt; }
    mbar_wait(empty(ps), pph ^ 1);
    mbar_expect_tx(full(ps), STAGE);
    if (pkt < KT) {
      const int btn = BAT ? itn + (itm >> bsh) * tiles_n : itn;
      tma_load(stage(ps), &maps->a, full(ps), pkt * BK, itm * (A_BYTES / BK));
      tma_load(stage(ps) + A_BYTES, &maps->b, full(ps), pkt * BK, btn * ((STAGE - A_BYTES) / BK));
    } else {
      tma_load(stage(ps), &maps->pa, full(ps), 0, itm * (A_BYTES / BK));
      tma_load(stage(ps) + A_BYTES, &maps->pb, full(ps), 0, itn * ((STAGE - A_BYTES) / BK));
    }
    if (++pkt == steps) { pkt = 0; pt += gridDim.x; }
    if (++ps == NST) { ps = 0; pph ^= 1; }
  }
  DEV void prologue() {
    if (issuer) {
      tma_acquire(&maps->a); tma_acquire(&maps->b); tma_acquire(&maps->pa); tma_acquire(&maps->pb);
      for (int i = 0; i < NST; i++) issue();
    }
    __syncwarp();
  }
  DEV void release(uint32_t done) {
    mbar_release(done);
    if (issuer) issue();
    __syncwarp();
  }
  DEV void wait_full() {
    if constexpr (SLEEP) mbar_sleep_warp(full(s), ph);
    else mbar_wait_warp(full(s), ph);
  }
  uint32_t uf[4] = {0, 0, 0, 0};
  DEV void ubcast() {
#pragma unroll
    for (int i = 0; i < 4; i++) uf[i] = __shfl_sync(0xFFFFFFFFu, uf[i], 0);
  }
  DEV void ufill() {
    if constexpr (USH == 1) ubcast();
#pragma unroll
    for (int i = 0; i < UF * MI * NI / 4; i++) {
      uf[0] = uf[0] * 0x9E3779B1u + uf[1];
      uf[1] = uf[1] ^ uf[0] ^ (0x7F4A7C15u + 2u * i);
      uf[2] = uf[2] * 0x85EBCA77u + uf[3];
      uf[3] = uf[3] ^ uf[2] ^ (0x165667B1u + 2u * i);
    }
  }
  // the schedule: atom POS of a stage receipts (DUO64, XIN64: the odd ones, so the odd steps), unit j's first tile and slot
  __host__ __device__ static constexpr bool rc(int pos) { return !HALF || pos % 2 == 1; }
  __host__ __device__ static constexpr bool unit(int j) { return B == TB2W ? j % 2 == 1 : B == TB1W ? j % NI == NI - 1 : true; }
  __host__ __device__ static constexpr int first(int j) { return B == TB2W ? j - 1 : B == TB1W ? j - (NI - 1) : j; }
  __host__ __device__ static constexpr int slot(int j) { return LAG >= 0 ? j + LAG : 2 * j + 2 < j + 7 ? 2 * j + 2 : j + 7; }
  __host__ __device__ static constexpr bool sched_ok() {
    for (int j = 0; j < MI * NI; j++)
      if (unit(j) && (slot(j) < j || slot(j) > first(j) + MI * NI - 1)) return false;
    return true;
  }
  // the unit whose absorb ran last before MMA IDX of atom POS (the tie reads the state it wrote)
  __host__ __device__ static constexpr int last_unit(int pos, int idx) {
    for (int d = 1; d <= 4 * MI * NI; d++) {
      int s = idx - d, p = pos, u = -1;
      while (s < 0) { s += MI * NI; p = (p + 3) % 4; }
      for (int j = 0; j < MI * NI; j++)
        if (unit(j) && rc((p + 3) % 4) && slot(j) == s + MI * NI) u = j;
      for (int j = 0; j < MI * NI; j++)
        if (unit(j) && rc(p) && slot(j) == s) u = j;
      if (u >= 0) return u;
    }
    return 0;
  }
  template <int J> DEV static uint32_t tz(const St (&ch)[8]) {
    if constexpr (XF) return ch[J % 4].y;
    else if constexpr (TB) return ch[1].z;
    else if constexpr (is_fr(RK) && RK != FR40) return ch[4 * (J % 2) + 3].y;
    else return ch[4 * (J % 2) + 3].z;
  }
  // a TB kind's receipt accumulator R is ch[0].y, ch[0].z, ch[1].y, ch[1].z (fgemm.cu's rimma32 and rchain)
  DEV static void rim(St (&ch)[8], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(ch[0].y), "+r"(ch[0].z), "+r"(ch[1].y), "+r"(ch[1].z)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  DEV static void rch(St (&ch)[8], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%0, %1, %2, %3}, {%4, %5}, {%6, %6, %6, %6};"
        : "+r"(ch[0].y), "+r"(ch[0].z), "+r"(ch[1].y), "+r"(ch[1].z)
        : "r"(b0), "r"(b1), "r"(0));
  }
  // unit J's absorb (fgemm.cu's absorb and tb_at; XIN's fold)
  template <int J> DEV static void ab(St (&ch)[8], const uint32_t (&kr)[NKR], const int (&acc)[MI][NI][4]) {
    constexpr int mi = J / NI, ni = J % NI;
    const int(&w)[4] = acc[mi][ni];
    if constexpr (RK == DUO || RK == DUO64) {
      absorb<DUO, mi, ni>(ch, w);
    } else if constexpr (is_fr(RK)) {
      constexpr int H = 4 * (J % 2);
      fr_word<RK, J, 0>(ch[H], (uint32_t)w[0]);
      fr_word<RK, J, 1>(ch[H + 1], (uint32_t)w[1]);
      fr_word<RK, J, 2>(ch[H + 2], (uint32_t)w[2]);
      fr_word<RK, J, 3>(ch[H + 3], (uint32_t)w[3]);
    } else if constexpr (is_fx(RK)) {
      constexpr int H = 4 * (J % 2);
      duo<DUO, 0, J>(ch[H], fx_chain<RK, J, 0>((uint32_t)w[0]));
      duo<DUO, 0, J>(ch[H + 1], fx_chain<RK, J, 1>((uint32_t)w[1]));
      duo<DUO, 0, J>(ch[H + 2], fx_chain<RK, J, 2>((uint32_t)w[2]));
      duo<DUO, 0, J>(ch[H + 3], fx_chain<RK, J, 3>((uint32_t)w[3]));
    } else if constexpr (XF) {
      St& c = ch[J % 4];
#if EG_ROUNDS > 1
#pragma unroll 1
      for (int eg_r = 0; eg_r < EG_ROUNDS; eg_r++) {
        c.y = (rotl(c.y, ROT0) ^ (uint32_t)w[0]) * kr[0] + (uint32_t)w[1];
        c.y = (rotl(c.y, 19) ^ (uint32_t)w[2]) * kr[1] + (uint32_t)w[3];
      }
#else
      c.y = (rotl(c.y, ROT0) ^ (uint32_t)w[0]) * kr[0] + (uint32_t)w[1];
      c.y = (rotl(c.y, 19) ^ (uint32_t)w[2]) * kr[1] + (uint32_t)w[3];
#endif
    } else {
      if constexpr (tb_chained(RK) && J == (B == TB2W ? 1 : B == TB1W ? NI - 1 : 0)) rch(ch, kr[NKP<RK>], kr[NKP<RK> + 1]);
      if constexpr (B == TB4) {
        rim(ch, (uint32_t)w[0], (uint32_t)w[1], (uint32_t)w[2], (uint32_t)w[3], kr[2 * J], kr[2 * J + 1]);
      } else if constexpr (B == TB2W) {
        const int(&x)[4] = acc[mi][ni - 1];
        rim(ch, __byte_perm((uint32_t)x[0], (uint32_t)x[1], 0x5410), __byte_perm((uint32_t)x[2], (uint32_t)x[3], 0x5410),
            __byte_perm((uint32_t)w[0], (uint32_t)w[1], 0x5410), __byte_perm((uint32_t)w[2], (uint32_t)w[3], 0x5410),
            kr[2 * (J / 2)], kr[2 * (J / 2) + 1]);
      } else if constexpr (B == TB1W) {
        const int(&x0)[4] = acc[mi][0];
        const int(&x1)[4] = acc[mi][1];
        const int(&x2)[4] = acc[mi][2];
        auto b1 = [](const int(&p)[4], const int(&q)[4], int h) {
          return __byte_perm(__byte_perm((uint32_t)p[h], (uint32_t)p[h + 1], 0x0051), __byte_perm((uint32_t)q[h], (uint32_t)q[h + 1], 0x5100), 0x7610);
        };
        rim(ch, b1(x0, x1, 0), b1(x0, x1, 2), b1(x2, w, 0), b1(x2, w, 2), kr[2 * mi], kr[2 * mi + 1]);
      }
    }
  }
  // the absorbs at slot S: SELF, this atom's units; PREV, the previous atom's (slot S + 8)
  template <bool SELF, bool PREV, int S, int J = 0>
  DEV static void abs_at(St (&ch)[8], const uint32_t (&kr)[NKR], const int (&acc)[MI][NI][4]) {
    if constexpr (J < MI * NI) {
      if constexpr (unit(J) && ((SELF && slot(J) == S) || (PREV && slot(J) == S + MI * NI))) ab<J>(ch, kr, acc);
      abs_at<SELF, PREV, S, J + 1>(ch, kr, acc);
    }
  }
  template <int S = 0> DEV static void flush(St (&ch)[8], const uint32_t (&kr)[NKR], const int (&acc)[MI][NI][4]) {
    if constexpr (S < MI * NI) {
      abs_at<false, true, S>(ch, kr, acc);
      flush<S + 1>(ch, kr, acc);
    }
  }
  // atom POS of a stage: after MMA IDX, the absorbs the schedule puts at slot IDX (none of a previous atom in a tile's first)
  template <bool FIRST, int POS, int IDX = 0>
  DEV static void atom(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[NKR], const uint32_t (&fa)[MI][4],
                       const uint32_t (&fb)[NI][2], uint32_t zero) {
    if constexpr (IDX < MI * NI) {
      constexpr int mi = IDX / NI, ni = IDX % NI;
      if constexpr (FIRST) {
        imma0(acc[mi][ni], fa[mi], fb[ni]);
      } else {
        if constexpr (R && TIE != 0 && IDX % (TIE > 0 ? TIE : TIE < 0 ? -TIE : 1) == 0)
          tie<(TIE < 0)>(acc[mi][ni][0], tz<last_unit(POS, IDX)>(ch), zero);
        imma(acc[mi][ni], fa[mi], fb[ni]);
      }
      abs_at<false, R && !FIRST && rc((POS + 3) % 4), IDX>(ch, kr, acc);
      abs_at<R && rc(POS), false, IDX>(ch, kr, acc);
      atom<FIRST, POS, IDX + 1>(acc, ch, kr, fa, fb, zero);
    }
  }
  template <bool FIRST, bool LAST>
  DEV void stage_il(int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[NKR], uint32_t (&fa)[2][MI][4],
                    uint32_t (&fb)[2][NI][2], uint32_t zero) {
    const uint32_t sb = stage(s), done = empty(s);
    load(fa[1], fb[1], sb, 1);
    atom<FIRST, 0>(acc, ch, kr, fa[0], fb[0], zero);
    if constexpr (UF > 0) ufill();
    load(fa[0], fb[0], sb, 2);
    atom<false, 1>(acc, ch, kr, fa[1], fb[1], zero);
    if constexpr (UF > 0) ufill();
    load(fa[1], fb[1], sb, 3);
    atom<false, 2>(acc, ch, kr, fa[0], fb[0], zero);
    if constexpr (UF > 0) ufill();
    advance();
    if (!LAST) {
      wait_full();
      load(fa[0], fb[0], stage(s), 0);
    }
    atom<false, 3>(acc, ch, kr, fa[1], fb[1], zero);
    if constexpr (UF > 0) ufill();
    release(done);
  }
  DEV void tile_il(int K, int (&acc)[MI][NI][4], St (&ch)[8], const uint32_t (&kr)[NKR], uint32_t zero) {
    static_assert(sched_ok(), "unit j is absorbed after its last tile's MMA and before its first tile's next");
    const int KT = K / BK;
    uint32_t fa[2][MI][4], fb[2][NI][2];
    wait_full();
    load(fa[0], fb[0], stage(s), 0);
    if (KT == 1) {
      stage_il<true, true>(acc, ch, kr, fa, fb, zero);
    } else {
      stage_il<true, false>(acc, ch, kr, fa, fb, zero);
#pragma unroll 1
      for (int kt = 1; kt < KT - 1; kt++) stage_il<false, false>(acc, ch, kr, fa, fb, zero);
      stage_il<false, true>(acc, ch, kr, fa, fb, zero);
    }
    if constexpr (R) flush(ch, kr, acc);
  }
  // the tail stage's rows P_A = [S_1 | S_2 | S_0 | 0], P_B = [U | U | U | 0] (S = 256 (S_1 + S_2) + S_0, S_0 the low byte
  // as s8, S_1 the rest clamped to ±127, so S_2 = 0 while |S| <= 127 256): acc -= 256 (S_1 + S_2) U^T + S_0 U^T, atoms 0
  // and 2, and atom 1 if THREE (atoms 0 and 1, then 2, in two passes, so it holds two atoms' fragments as the 2-atom
  // tail does)
  template <bool THREE> DEV void peel(int (&acc)[MI][NI][4]) {
    wait_full();
    const uint32_t tb = stage(s);
    uint32_t fb[2][NI][2];
    load_b(fb[0], tb, 0);
    load_b(fb[1], tb, THREE ? 1 : 2);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      uint32_t fa[2][4];
      load_a(fa[0], tb, 0, mi);
      load_a(fa[1], tb, THREE ? 1 : 2, mi);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int t1[4], t2[4];
        imma0(t1, fa[0], fb[0][ni]);
        if constexpr (THREE) imma(t1, fa[1], fb[1][ni]);
        else imma0(t2, fa[1], fb[1][ni]);
#pragma unroll
        for (int e = 0; e < 4; e++)
          acc[mi][ni][e] = (int)((uint32_t)acc[mi][ni][e] - ((uint32_t)t1[e] << 8) - (THREE ? 0u : (uint32_t)t2[e]));
      }
    }
    if constexpr (THREE) {
      load_b(fb[0], tb, 2);
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        uint32_t fa[4];
        load_a(fa, tb, 2, mi);
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          int t2[4];
          imma0(t2, fa, fb[0][ni]);
#pragma unroll
          for (int e = 0; e < 4; e++) acc[mi][ni][e] = (int)((uint32_t)acc[mi][ni][e] - (uint32_t)t2[e]);
        }
      }
    }
    release(empty(s));
    advance();
  }
};
constexpr int PEEL2_MAXK = 4096;   // |S| <= 128 k / 32, so S_2 = 0 at k <= 4096; a longer k takes the 3-atom tail

template <int RK, int LAG, int TIE, bool THREE, bool SLEEP = false, int UF = 0, int USH = 0> DEV void e_body(const Args& a) {
  using L = EL<RK, LAG, TIE, SLEEP, UF, USH>;
  constexpr int MI = L::MI, NI = L::NI;
  extern __shared__ __align__(1024) uint8_t smem[];
  L ml(smem, a.maps, a.M, a.N, a.K, a.group, true, 1);
  ml.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  ml.prologue();
  ml.consumer_init(warp, lane);
  const int tiles_m = a.M / 128, tiles_n = a.N / 128;
  const int wm = warp / L::WARPS_N, wn = warp % L::WARPS_N, g = lane >> 2, q = lane & 3, ctid = threadIdx.x;
  const uint32_t zero = (uint32_t)a.KR;
  uint32_t kr[L::NKR];
  if constexpr (L::XF) {
    uint2 k;
    if constexpr (RK == XIN13) k = xin_keys_r9(a.seed);
    else k = xin_keys<false>(a.seed);
    kr[0] = k.x, kr[1] = k.y;
  } else if constexpr (L::TB) L::template keys<RK>(kr, lane);
  else kr[0] = 0;
  if constexpr (UF > 0) {
    ml.uf[0] = a.seed, ml.uf[1] = a.seed ^ 0x3C6EF372u, ml.uf[2] = (uint32_t)a.KR, ml.uf[3] = a.seed + 0xA54FF53Au;
  }
  typename L::Place pl{0, 0};
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    ml.place_step(pl, t == (int)blockIdx.x ? t : (int)gridDim.x);
    L::place_tile(pl, tiles_m, tm, tn);
    const uint32_t tile_id = tm * tiles_n + tn;
    St ch[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      ch[e].y = RK == XIN13 ? (tile_id * L::NCT + ctid) * 4 + e : L::TB ? 0u : mix32(a.seed ^ mix32((tile_id * L::NCT + ctid) * 8 + e));
      ch[e].z = L::TB ? 0u : mix32(ch[e].y + 0x9E3779B9u);
    }
    int acc[MI][NI][4];
    if constexpr (USH == 2) ml.ubcast();
    ml.tile_il(a.K, acc, ch, kr, zero);
    ml.template peel<THREE>(acc);
    if constexpr (UF > 0) ch[0].y ^= ml.uf[0] ^ ml.uf[1], ch[1].y ^= ml.uf[2] ^ ml.uf[3];
    const int row0 = tm * 128 + wm * 32 + g, col0 = tn * 128 + wn * 32 + 2 * q;
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        int* p = a.U + (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni;
        *reinterpret_cast<int2*>(p) = make_int2(acc[mi][ni][0], acc[mi][ni][1]);
        *reinterpret_cast<int2*>(p + (size_t)8 * a.N) = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
      }
    if constexpr (L::R) {
      uint4* r = reinterpret_cast<uint4*>(a.R) + ((size_t)tile_id * L::NCT + ctid) * 2;
      if constexpr (L::XF) {
        r[0] = make_uint4(ch[0].y, ch[1].y, ch[2].y, ch[3].y);
        r[1] = make_uint4(0, 0, 0, 0);
      } else if constexpr (L::TB) {
        r[0] = make_uint4(ch[0].y, ch[0].z, ch[1].y, ch[1].z);
        r[1] = make_uint4(0, 0, 0, 0);
      } else {
        r[0] = make_uint4(finish<DUO>(ch, 0), finish<DUO>(ch, 1), finish<DUO>(ch, 2), finish<DUO>(ch, 3));
        r[1] = make_uint4(finish<DUO>(ch, 4), finish<DUO>(ch, 5), finish<DUO>(ch, 6), finish<DUO>(ch, 7));
      }
    }
  }
}
// <kind>_l<LAG><tie>: e duo, d64 DUO64, x XIN, x64 XIN64, x13 XIN13, t4/t4c/t2w/t1w the TB kinds; l s the spread schedule; tie h every second
// MMA, q every fourth, o every eighth, none if absent; m: the IMAD tie; s: SLEEP. NAME_k3 is NAME with the 3-atom tail (k > 4096;
// one kernel with both tails behind a branch on K spills 50 bytes more in the 2-atom one)
#define EGEMM(NAME, RK, LAG, TIE)                                                                                         \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ Args a) { e_body<RK, LAG, TIE, false>(a); } \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME##_k3(const __grid_constant__ Args a) { e_body<RK, LAG, TIE, true>(a); }
// NAME with SLEEP (s): its consumers sleep on a stage's mbarrier instead of retrying
#define EGEMMS(NAME, RK, LAG, TIE)                                                                                              \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ Args a) { e_body<RK, LAG, TIE, false, true>(a); } \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME##_k3(const __grid_constant__ Args a) { e_body<RK, LAG, TIE, true, true>(a); }
EGEMM(e_plain, NONE, 0, 0)
EGEMMS(e_plains, NONE, 0, 0)
EGEMM(e_l2h, DUO, 2, 2)
EGEMM(e_l2q, DUO, 2, 4)
EGEMM(e_l3qm, DUO, 3, -4)
EGEMM(e_l2o, DUO, 2, 8)
EGEMM(e_l4q, DUO, 4, 4)
EGEMM(e_l4o, DUO, 4, 8)
EGEMM(e_l6q, DUO, 6, 4)
EGEMM(e_l7q, DUO, 7, 4)
EGEMM(e_l7o, DUO, 7, 8)
EGEMMS(e_l7qs, DUO, 7, 4)
EGEMMS(e_l2qs, DUO, 2, 4)
EGEMM(d64_l2q, DUO64, 2, 4)
EGEMM(d64_l7q, DUO64, 7, 4)
EGEMM(d64_lsq, DUO64, -1, 4)
EGEMM(d64_lsqm, DUO64, -1, -4)
EGEMM(x_l2, XIN, 2, 0)
EGEMM(x_l2q, XIN, 2, 4)
EGEMM(x_l3qm, XIN, 3, -4)
EGEMM(x_l2o, XIN, 2, 8)
EGEMM(x64_l2q, XIN64, 2, 4)
EGEMM(x64_l2o, XIN64, 2, 8)
EGEMM(x13_l2o, XIN13, 2, 8)
EGEMM(t4_l2, TB4, 2, 0)
EGEMM(t4c_l2, TB4C, 2, 0)
EGEMM(t2w_l0, TB2W, 0, 0)
EGEMM(t2w_l2, TB2W, 2, 0)
EGEMM(t1w_l0, TB1W, 0, 0)
EGEMM(t1w_l2, TB1W, 2, 0)
EGEMM(f10_l2q, FR10, 2, 4)
EGEMM(f15_l2q, FR15, 2, 4)
EGEMM(f20_l2q, FR20, 2, 4)
EGEMM(f25_l2q, FR25, 2, 4)
EGEMM(f40_l2q, FR40, 2, 4)
// the model's predictions against H's schedule (e_l7qs): r = 2 (f20), duo on x-op filler chains (fx<x>), duo with UF
// warp-uniform instructions a MMA (u<UF>; s: broadcast before each atom's, t: before each tile's)
#define EGEMMU(NAME, RK, LAG, TIE, UF, USH)                                                                                             \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME(const __grid_constant__ Args a) { e_body<RK, LAG, TIE, false, true, UF, USH>(a); } \
  __global__ void __launch_bounds__(MLW::NT, 1) NAME##_k3(const __grid_constant__ Args a) { e_body<RK, LAG, TIE, true, true, UF, USH>(a); }
EGEMMS(f20_l7qs, FR20, 7, 4)
EGEMMS(fx05_l7qs, FX05, 7, 4)
EGEMMS(fx1_l7qs, FX1, 7, 4)
EGEMMS(fx15_l7qs, FX15, 7, 4)
EGEMMS(fx2_l7qs, FX2, 7, 4)
EGEMMS(fx3_l7qs, FX3, 7, 4)
EGEMMS(fx45_l7qs, FX45, 7, 4)
EGEMMS(fx8_l7qs, FX8, 7, 4)
EGEMMS(fx12_l7qs, FX12, 7, 4)
EGEMMS(fx24_l7qs, FX24, 7, 4)
EGEMMU(u8_l7qs, DUO, 7, 4, 8, 0)
EGEMMU(u16_l7qs, DUO, 7, 4, 16, 0)
EGEMMU(u8s_l7qs, DUO, 7, 4, 8, 1)
EGEMMU(u16s_l7qs, DUO, 7, 4, 16, 1)
EGEMMU(u16t_l7qs, DUO, 7, 4, 16, 2)

// ---------------------------------------------------------------- the operands
// x (rows x k bf16) to X (rows x k int8): a warp a group of 4 rows, its lanes 8 columns each (512 contiguous bytes of x a
// step), 8 warps a block (blk), in chunks of 4096 columns (32 patches, one a lane to hash); DITHER: X = clamp(floor(x
// inv_s), -127, 126) + b and P_A's rows [S_1 | S_2 | S_0 | 0] (EL::peel's; lane c' of the warp ends with S at
// c = 8 (c' % 4) + c' / 4 after a reduce-scatter over the 8 lanes holding that column's partial sums), else
// X = clamp(rint(x inv_s), -127, 127). dgs: 8 x 512 words of shared memory, sps: 8 x 128 (DITHER)
template <bool DITHER>
DEV void quant_blk(int blk, uint32_t* dgs, int* sps, const uint16_t* __restrict__ x, int8_t* __restrict__ X, int8_t* __restrict__ PA,
                   int rows, int k, float inv_s, const Sig& sg) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, grp = blk * 8 + warp;
  if (grp * 4 >= rows) return;
  const int kb = k / 128;
  uint32_t* dg = dgs + (DITHER ? 512 * warp : 0);   // the warp's 32 patch digests, digest byte order
  int* sp = sps + (DITHER ? 128 * warp : 0);         // S over the chunks so far (k > 4096): lane c' at 32 rr + c
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dg);
#pragma unroll 1
  for (int p0 = 0; p0 < kb; p0 += 32) {
    const int np = min(32, kb - p0), steps = np / 2;
    if constexpr (DITHER) {
      if (p0) __syncwarp();
      if (lane < np) {
        uint64_t st[8];
        sha_block(sg.w, (uint64_t)DITHER_TAG << 56 | ((uint64_t)grp * kb + p0 + lane), st);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          dg[16 * lane + 2 * i] = __byte_perm((uint32_t)(st[i] >> 32), 0, 0x0123);
          dg[16 * lane + 2 * i + 1] = __byte_perm((uint32_t)st[i], 0, 0x0123);
        }
      }
      __syncwarp();
    }
#pragma unroll 1
    for (int rr = 0; rr < 4; rr++) {
      const size_t r = (size_t)grp * 4 + rr;
      const uint4* src = reinterpret_cast<const uint4*>(x + r * k) + 16 * p0;
      uint2* dst = reinterpret_cast<uint2*>(X + r * k) + 16 * p0;
      int v[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // S's partial sums at c = 8 (lane % 4) + i
#pragma unroll 4
      for (int it = 0; it < steps; it++) {
        const int ch = it * 32 + lane;   // the chunk's columns 8 ch .. 8 ch + 7
        const uint4 w = __ldg(src + ch);
        uint32_t bits = 0;
        if constexpr (DITHER) bits = bytes[64 * (ch / 16) + 16 * rr + ch % 16];
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        int qv[8];
#pragma unroll
        for (int h = 0; h < 4; h++) {
          const float f0 = __uint_as_float(ww[h] << 16) * inv_s, f1 = __uint_as_float(ww[h] & 0xFFFF0000u) * inv_s;
          if constexpr (DITHER) {
            qv[2 * h] = min(max(__float2int_rd(f0), -127), 126) + (int)((bits >> (2 * h)) & 1u);
            qv[2 * h + 1] = min(max(__float2int_rd(f1), -127), 126) + (int)((bits >> (2 * h + 1)) & 1u);
          } else {
            qv[2 * h] = min(max(__float2int_rn(f0), -127), 127);
            qv[2 * h + 1] = min(max(__float2int_rn(f1), -127), 127);
          }
        }
        if constexpr (DITHER) {
#pragma unroll
          for (int i = 0; i < 8; i++) v[i] += qv[i];
        }
        const uint32_t lo = __byte_perm(__byte_perm(qv[0], qv[1], 0x0040), __byte_perm(qv[2], qv[3], 0x0040), 0x5410);
        const uint32_t hi = __byte_perm(__byte_perm(qv[4], qv[5], 0x0040), __byte_perm(qv[6], qv[7], 0x0040), 0x5410);
        dst[ch] = make_uint2(lo, hi);
      }
      if constexpr (DITHER) {
        const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const int send = b4 ? v[i] : v[i + 4], keep = b4 ? v[i + 4] : v[i];
          v[i] = keep + __shfl_xor_sync(~0u, send, 16);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
          const int send = b3 ? v[i] : v[i + 2], keep = b3 ? v[i + 2] : v[i];
          v[i] = keep + __shfl_xor_sync(~0u, send, 8);
        }
        {
          const int send = b2 ? v[0] : v[1], keep = b2 ? v[1] : v[0];
          v[0] = keep + __shfl_xor_sync(~0u, send, 4);
        }
        const int c = 8 * (lane & 3) + (lane >> 2);
        int S = v[0];
        if (kb > 32) {
          if (p0) S += sp[32 * rr + c];
          if (p0 + 32 < kb) { sp[32 * rr + c] = S; continue; }
        }
        const int s0 = (int)(int8_t)(S & 0xFF), sh = (S - s0) / 256, s1 = min(max(sh, -127), 127);
        int8_t* pr = PA + r * 128;
        pr[c] = (int8_t)s1; pr[32 + c] = (int8_t)(sh - s1); pr[64 + c] = (int8_t)s0; pr[96 + c] = 0;
      }
    }
  }
}
constexpr int QROWS = 32;   // rows a quant block
template <bool DITHER>
__global__ void __launch_bounds__(256) quant(const uint16_t* __restrict__ x, int8_t* __restrict__ X, int8_t* __restrict__ PA,
                                             int rows, int k, float inv_s, const __grid_constant__ Sig sg) {
  __shared__ uint32_t dg[DITHER ? 8 * 512 : 1];
  __shared__ int sp[DITHER ? 8 * 128 : 1];
  quant_blk<DITHER>(blockIdx.x, dg, sp, x, X, PA, rows, k, inv_s, sg);
}

// W' = W + U (W'_jk = W_jk + U_{j, k mod 32}) and P_B's rows [U | U | U | 0]: a block (blk) 8 rows, its threads loading
// W's first 16-byte chunks (8 a thread, all of them at k <= 4096) while threads 0..3 hash U's blocks (rows 2j, 2j + 1).
// us: 64 words of shared memory
DEV void wedit_blk(int blk, uint32_t* us, const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp, uint8_t* __restrict__ PB, int k,
                   const Sig& sg) {
  const int r0 = blk * 8, t = threadIdx.x, cpr = k / 16, total = 8 * cpr;
  constexpr int MAXC = 8;
  uint4 w[MAXC];
  const uint4* src = reinterpret_cast<const uint4*>(W + (size_t)r0 * k);
#pragma unroll
  for (int i = 0; i < MAXC; i++)
    if (t + 256 * i < total) w[i] = __ldg(src + t + 256 * i);
  if (t < 4) {
    uint64_t st[8];
    sha_block(sg.w, (uint64_t)U_TAG << 56 | (uint64_t)(r0 / 2 + t), st);
#pragma unroll
    for (int rr = 0; rr < 2; rr++)
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
          const int e = 32 * rr + 4 * q + b;
          v |= u_byte((uint32_t)(st[e / 8] >> (56 - 8 * (e % 8))) & 0xFFu) << (8 * b);
        }
        us[8 * (2 * t + rr) + q] = v;
      }
  }
  __syncthreads();
  {
    const int rr = t / 32, wq = t % 32;
    reinterpret_cast<uint32_t*>(PB + (size_t)(r0 + rr) * 128)[wq] = wq / 8 == 3 ? 0u : us[8 * rr + wq % 8];
  }
  uint4* dst = reinterpret_cast<uint4*>(Wp + (size_t)r0 * k);
#pragma unroll 1
  for (int c0 = 0; c0 < total; c0 += 256 * MAXC) {
    if (c0) {
#pragma unroll
      for (int i = 0; i < MAXC; i++)
        if (c0 + t + 256 * i < total) w[i] = __ldg(src + c0 + t + 256 * i);
    }
#pragma unroll
    for (int i = 0; i < MAXC; i++) {
      const int c = c0 + t + 256 * i;
      if (c < total) {
        const uint32_t* u = us + 8 * (c / cpr) + (4 * (c % cpr)) % 8;
        const uint4 x = w[i];
        dst[c] = make_uint4(add_noise<false>(x.x, u[0]), add_noise<false>(x.y, u[1]), add_noise<false>(x.z, u[2]), add_noise<false>(x.w, u[3]));
      }
    }
  }
}
__global__ void __launch_bounds__(256) wedit(const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp, uint8_t* __restrict__ PB, int k,
                                             const __grid_constant__ Sig sg) {
  __shared__ uint32_t us[64];
  wedit_blk(blockIdx.x, us, W, Wp, PB, k, sg);
}
// quant<true> and wedit in one launch under one key, qb quantizer blocks and the edit's (8 rows of W each) filling the
// SMs the quantizer's 8-warp blocks leave idle; ORD orders them: 0 the quantizer's first, 1 the edit's first, 2 every
// third block a quantizer's while they last. 3 blocks an SM: wedit's 80 registers, not the 122 ptxas takes unbounded
template <int ORD>
__global__ void __launch_bounds__(256, 3) qwe(const uint16_t* __restrict__ x, int8_t* __restrict__ X, int8_t* __restrict__ PA, int rows,
                                           int k, float inv_s, int qb, const uint8_t* __restrict__ W, uint8_t* __restrict__ Wp,
                                           uint8_t* __restrict__ PB, const __grid_constant__ Sig sg) {
  __shared__ uint32_t sm[8 * 512];
  __shared__ int sp[8 * 128];
  const int b = blockIdx.x, nw = (int)gridDim.x - qb;
  int q, e;   // the quantizer's block, else the edit's
  if constexpr (ORD == 0) { q = b < qb ? b : -1; e = b - qb; }
  else if constexpr (ORD == 1) { q = b >= nw ? b - nw : -1; e = b; }
  else {
    // every third block quantizes while quantizer blocks remain; when the grid runs out first (qb > ceil(grid/3)),
    // the blocks past the last edit take the rest
    const int Q = min(qb, ((int)gridDim.x + 2) / 3);
    q = b % 3 == 0 && b / 3 < qb ? b / 3 : -1;
    e = b - min((b + 2) / 3, qb);
    if (q < 0 && e >= nw) q = Q + (e - nw);
  }
  if (q >= 0) quant_blk<true>(q, sm, sp, x, X, PA, rows, k, inv_s, sg);
  else wedit_blk(e, sm, W, Wp, PB, k, sg);
}

// x in bf16, (h / 2^24 * 2 - 1) 127.5 rounded to bf16; the gate's rows 0 and 1 at +200 and -200 (clamped)
__global__ void __launch_bounds__(256) xfill(uint16_t* x, size_t count, uint32_t seed, size_t k, int ends) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)i) ^ mix32((uint32_t)(i >> 32) + 0x77u));
    float f = ((float)(h & 0xFFFFFF) / 16777216.f * 2.f - 1.f) * 127.5f;
    if (ends && i < k) f = 200.f;
    else if (ends && i < 2 * k) f = -200.f;
    const uint32_t b = __float_as_uint(f);
    x[i] = (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
  }
}
__global__ void cmp_words(const uint32_t* a, const uint32_t* b, size_t n, unsigned long long* bad) {
  unsigned long long c = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += a[i] != b[i];
  if (c) atomicAdd(bad, c);
}

// ---------------------------------------------------------------- host
struct EP {
  int m, n, k, grid;
  uint16_t* x;
  int8_t *X, *Xq, *W, *Wp, *PA, *PB;
  int *Y, *Yr, *Yr2;
  uint32_t *R, *R2;
  size_t ny, nr;
  LtG plain, plainq;
  Maps* mE;
  uint32_t seed, inst = 0;
  float inv_s = 1.f;
  cudaStream_t st;
  unsigned long long* bad;
  // overlap arms: W'_i + 1 built on st2 into the other buffer while call i's GEMM runs on st
  int8_t *Wo[2], *PBo[2];
  Maps* mO[2];
  cudaStream_t st2;
  cudaEvent_t ef, ej;
  uint32_t ov = 0;
  // the fused producer's (qwe's) own X, P_A, W', P_B, so the gate compares them with quant<true>'s and wedit's
  int8_t *Xf, *PAf, *Wf, *PBf;
  Maps* mF;
};

// the GEMM kernels (k3: the 3-atom tail's twin) and their receipt kinds (egemm_ref.py's names), each kind's together
struct EKn { const char* name; Kern k, k3; const char* kind; };
#define EKE(NAME, KIND) {#NAME, NAME, NAME##_k3, KIND}
static const EKn EK[] = {
    EKE(e_plain, ""), {"t_duo32", t_duo32, t_duo32, "duo32"}, EKE(e_l2h, "duo32"), EKE(e_l2q, "duo32"),
    EKE(e_l3qm, "duo32"), EKE(e_l2o, "duo32"), EKE(e_l4q, "duo32"), EKE(e_l4o, "duo32"), EKE(e_l6q, "duo32"),
    EKE(e_l7q, "duo32"), EKE(e_l7o, "duo32"), EKE(e_l7qs, "duo32"), EKE(e_l2qs, "duo32"), EKE(e_plains, ""),
    EKE(d64_l2q, "duo64"), EKE(d64_l7q, "duo64"), EKE(d64_lsq, "duo64"), EKE(d64_lsqm, "duo64"),
    EKE(x_l2, "xin"), EKE(x_l2q, "xin"), EKE(x_l3qm, "xin"), EKE(x_l2o, "xin"), EKE(x64_l2q, "xin64"), EKE(x64_l2o, "xin64"),
    EKE(x13_l2o, "xin13"),
    EKE(t4_l2, "tb4"), EKE(t4c_l2, "tb4c"), EKE(t2w_l0, "tb2w"), EKE(t2w_l2, "tb2w"),
    EKE(t1w_l0, "tb1w"), EKE(t1w_l2, "tb1w"),
    EKE(f10_l2q, ""), EKE(f15_l2q, ""), EKE(f20_l2q, ""), EKE(f25_l2q, ""), EKE(f40_l2q, ""), EKE(f20_l7qs, ""),
    EKE(fx05_l7qs, ""), EKE(fx1_l7qs, ""), EKE(fx15_l7qs, ""), EKE(fx2_l7qs, ""), EKE(fx3_l7qs, ""),
    EKE(fx45_l7qs, ""), EKE(fx8_l7qs, ""), EKE(fx12_l7qs, ""), EKE(fx24_l7qs, ""), EKE(u8_l7qs, ""), EKE(u16_l7qs, ""),
    EKE(u8s_l7qs, ""), EKE(u16s_l7qs, ""), EKE(u16t_l7qs, "")};
static const EKn* ekern(const std::string& n) {
  for (auto& e : EK) if (n == e.name) return &e;
  fprintf(stderr, "no kernel %s\n", n.c_str());
  exit(2);
}

static Sig osig(const EP& P, uint32_t i) { return sigma(P.seed, 0x40000000u + i); }
static void wedit_to(EP& P, cudaStream_t st, int8_t* Wp, int8_t* PB, const Sig& s) {
  wedit<<<P.n / 8, 256, 0, st>>>(reinterpret_cast<const uint8_t*>(P.W), reinterpret_cast<uint8_t*>(Wp), reinterpret_cast<uint8_t*>(PB), P.k, s);
}
static void qd(EP& P, const Sig& s) { quant<true><<<(P.m + QROWS - 1) / QROWS, 256, 0, P.st>>>(P.x, P.X, P.PA, P.m, P.k, P.inv_s, s); }
static void qp(EP& P, const Sig& s) { quant<false><<<(P.m + QROWS - 1) / QROWS, 256, 0, P.st>>>(P.x, P.Xq, nullptr, P.m, P.k, P.inv_s, s); }
static void we(EP& P, const Sig& s) { wedit_to(P, P.st, P.Wp, P.PB, s); }
using QweK = void (*)(const uint16_t*, int8_t*, int8_t*, int, int, float, int, const uint8_t*, uint8_t*, uint8_t*, Sig);
static const std::pair<const char*, QweK> QWE[] = {{"", qwe<0>}, {"_w", qwe<1>}, {"_i", qwe<2>}};
static void qf(EP& P, const Sig& s, QweK f = qwe<0>) {
  const int qb = (P.m + QROWS - 1) / QROWS;
  f<<<qb + P.n / 8, 256, 0, P.st>>>(P.x, P.Xf, P.PAf, P.m, P.k, P.inv_s, qb, reinterpret_cast<const uint8_t*>(P.W),
                                     reinterpret_cast<uint8_t*>(P.Wf), reinterpret_cast<uint8_t*>(P.PBf), s);
}
static void esetup(EP& P, int m, int n, int k, uint32_t seed, bool gate) {
  P.m = m; P.n = n; P.k = k; P.seed = seed;
  // k = 128: cgemm.cu's cohalf, on its own operands (quant's 256-column steps leave this file's X unwritten there)
  if (m % 128 || n % 128 || (k % 256 && k != 128) || k > 16128) {
    fprintf(stderr, "shape %d %d %d: m %% 128 == 0 (any-M: EL's raster has a ragged last group), n %% 128 == 0, k %% 256 == 0 or 128, k <= 16128 (S_1, S_2 in s8)\n", m, n, k);
    exit(2);
  }
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  P.grid = std::min((m / 128) * (n / 128), nsm);
  CK(cudaStreamCreate(&P.st));
  P.ny = (size_t)m * n;
  P.nr = (size_t)(m / 128) * (n / 128) * 512 * 8;
  CK(cudaMalloc(&P.x, (size_t)m * k * 2));
  CK(cudaMalloc(&P.X, (size_t)m * k)); CK(cudaMalloc(&P.Xq, (size_t)m * k));
  CK(cudaMalloc(&P.W, (size_t)n * k)); CK(cudaMalloc(&P.Wp, (size_t)n * k));
  CK(cudaMalloc(&P.PA, (size_t)m * 128)); CK(cudaMalloc(&P.PB, (size_t)n * 128));
  CK(cudaMalloc(&P.Y, P.ny * 4)); CK(cudaMalloc(&P.Yr, P.ny * 4)); CK(cudaMalloc(&P.Yr2, P.ny * 4));
  CK(cudaMalloc(&P.R, P.nr * 4)); CK(cudaMalloc(&P.R2, P.nr * 4));
  CK(cudaMalloc(&P.bad, 8));
  for (int b = 0; b < 2; b++) { CK(cudaMalloc(&P.Wo[b], (size_t)n * k)); CK(cudaMalloc(&P.PBo[b], (size_t)n * 128)); }
  CK(cudaMalloc(&P.Xf, (size_t)m * k)); CK(cudaMalloc(&P.PAf, (size_t)m * 128));
  CK(cudaMalloc(&P.Wf, (size_t)n * k)); CK(cudaMalloc(&P.PBf, (size_t)n * 128));
  CK(cudaStreamCreate(&P.st2));
  CK(cudaEventCreateWithFlags(&P.ef, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&P.ej, cudaEventDisableTiming));
  xfill<<<4096, 256, 0, P.st>>>(P.x, (size_t)m * k, mix32(seed ^ 0x3333u), k, gate);
  sfill<I8><<<4096, 256, 0, P.st>>>(reinterpret_cast<uint8_t*>(P.W), (size_t)n * k, mix32(seed ^ 0x4444u), -63, 127);
  if (gate) { CK(cudaMemsetAsync(P.W, 0x3F, k, P.st)); CK(cudaMemsetAsync(P.W + k, 0xC1, k, P.st)); }   // rows at +63, -63
  qd(P, sigma(seed, 0xFFFFFFFFu));
  qp(P, sigma(seed, 0xFFFFFFFFu));
  we(P, sigma(seed, 0xFFFFFFFFu));
  wedit_to(P, P.st, P.Wo[0], P.PBo[0], osig(P, 0));
  qf(P, sigma(seed, 0xFFFFFFFFu));
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  P.plain = ltg(I8, m, n, k, k, k, n, P.X, P.W, P.Yr, 0, P.st);
  P.plainq = ltg(I8, m, n, k, k, k, n, P.Xq, P.W, P.Yr, 0, P.st);
  P.mE = upload_maps(Maps{tmap(P.X, k, m, 128), tmap(P.Wp, k, n, 128), tmap(P.PA, 128, m, 128), tmap(P.PB, 128, n, 128)});
  for (int b = 0; b < 2; b++)
    P.mO[b] = upload_maps(Maps{tmap(P.X, k, m, 128), tmap(P.Wo[b], k, n, 128), tmap(P.PA, 128, m, 128), tmap(P.PBo[b], 128, n, 128)});
  P.mF = upload_maps(Maps{tmap(P.Xf, k, m, 128), tmap(P.Wf, k, n, 128), tmap(P.PAf, 128, m, 128), tmap(P.PBf, 128, n, 128)});
  for (auto& e : EK)
    for (Kern kern : {e.k, e.k3}) CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
}
static void efree(EP& P) {
  CK(cudaDeviceSynchronize());
  for (void* p : {(void*)P.x, (void*)P.X, (void*)P.Xq, (void*)P.W, (void*)P.Wp, (void*)P.PA, (void*)P.PB, (void*)P.Y, (void*)P.Yr, (void*)P.Yr2,
                  (void*)P.R, (void*)P.R2, (void*)P.bad, (void*)P.mE, (void*)P.Wo[0], (void*)P.Wo[1], (void*)P.PBo[0],
                  (void*)P.PBo[1], (void*)P.mO[0], (void*)P.mO[1], (void*)P.Xf, (void*)P.PAf, (void*)P.Wf, (void*)P.PBf, (void*)P.mF})
    CK(cudaFree(p));
  CK(cudaEventDestroy(P.ef)); CK(cudaEventDestroy(P.ej));
  CK(cudaStreamDestroy(P.st2));
  CK(cudaStreamDestroy(P.st));
}
static Kern pick(const EP& P, const EKn* e) { return P.k > PEEL2_MAXK ? e->k3 : e->k; }
static void egemm_run(EP& P, Kern kern, int* Y, uint32_t* R, const Maps* maps = nullptr) {
  Args a{maps ? maps : P.mE, Y, R, P.m, P.n, P.k, 8, P.seed, 0, 1, 0, 0, {0, 0, 0, 0}, nullptr, 0};
  kern<<<P.grid, MLW::NT, CSMEM, P.st>>>(a);
}
// call i of the overlap arms: fork st2 off st, which builds W'_(i+1) under key i + 1 into buffer (i + 1) % 2 (call i - 1's,
// whose GEMM st has finished) while st runs call i (the dither under key i if DITHER, then the GEMM on buffer i % 2); st
// then waits for st2, so a call's time is the longer of the two and nothing spills into the next call
static void ov_call(EP& P, Kern k, bool dither) {
  const int b = P.ov & 1;
  CK(cudaEventRecord(P.ef, P.st));
  CK(cudaStreamWaitEvent(P.st2, P.ef));
  wedit_to(P, P.st2, P.Wo[b ^ 1], P.PBo[b ^ 1], osig(P, P.ov + 1));
  if (dither) qd(P, osig(P, P.ov));
  egemm_run(P, k, P.Y, P.R, P.mO[b]);
  CK(cudaEventRecord(P.ej, P.st2));
  CK(cudaStreamWaitEvent(P.st, P.ej));
  P.ov++;
}

// an arm: plain (cuBLASLt on X, W), plainq (the plain quantizer, then cuBLASLt on its X), qplain, qdither, wedit, qwe
// (the fused producer), a GEMM kernel by name (EK), edited:KERN (wedit, then KERN), editedq:KERN (qdither, wedit, KERN:
// one key for both), editedf:KERN (qwe, then KERN on its operands; qwe_w, qwe_i and editedf_w:, editedf_i: the other
// block orders), overlap:KERN and overlapq:KERN (ov_call without and with the dither)
static std::function<void(EP&)> earm(const std::string& a) {
  if (a == "plain") return [](EP& P) { ltg_run(P.plain, P.st); };
  if (a == "plainq") return [](EP& P) { qp(P, sigma(P.seed, P.inst++)); ltg_run(P.plainq, P.st); };
  if (a == "qplain") return [](EP& P) { qp(P, sigma(P.seed, P.inst++)); };
  if (a == "qdither") return [](EP& P) { qd(P, sigma(P.seed, P.inst++)); };
  if (a == "wedit") return [](EP& P) { we(P, sigma(P.seed, P.inst++)); };
  for (auto& [sfx, f] : QWE) {
    const std::string qn = std::string("qwe") + sfx, en = std::string("editedf") + sfx + ":";
    if (a == qn) return [f](EP& P) { qf(P, sigma(P.seed, P.inst++), f); };
    if (a.rfind(en, 0) == 0) {
      const EKn* k = ekern(a.substr(en.size()));
      return [k, f](EP& P) { qf(P, sigma(P.seed, P.inst++), f); egemm_run(P, pick(P, k), P.Y, P.R, P.mF); };
    }
  }
  if (a.rfind("edited:", 0) == 0) {
    const EKn* k = ekern(a.substr(7));
    return [k](EP& P) { we(P, sigma(P.seed, P.inst++)); egemm_run(P, pick(P, k), P.Y, P.R); };
  }
  if (a.rfind("editedq:", 0) == 0) {
    const EKn* k = ekern(a.substr(8));
    return [k](EP& P) {
      const Sig s = sigma(P.seed, P.inst++);
      qd(P, s); we(P, s); egemm_run(P, pick(P, k), P.Y, P.R);
    };
  }
  if (a.rfind("overlap:", 0) == 0) {
    const EKn* k = ekern(a.substr(8));
    return [k](EP& P) { ov_call(P, pick(P, k), false); };
  }
  if (a.rfind("overlapq:", 0) == 0) {
    const EKn* k = ekern(a.substr(9));
    return [k](EP& P) { ov_call(P, pick(P, k), true); };
  }
  const EKn* k = ekern(a);
  return [k](EP& P) { egemm_run(P, pick(P, k), P.Y, P.R); };
}

static unsigned long long ecmp(EP& P, const void* a, const void* b, size_t words) {
  CK(cudaMemsetAsync(P.bad, 0, 8, P.st));
  cmp_words<<<1024, 256, 0, P.st>>>((const uint32_t*)a, (const uint32_t*)b, words, P.bad);
  unsigned long long h;
  CK(cudaMemcpyAsync(&h, P.bad, 8, cudaMemcpyDeviceToHost, P.st));
  CK(cudaStreamSynchronize(P.st));
  return h;
}

// key_ok_r9 on the device, on words at its boundary (a b = t mod 2^30 for each b in 1..128 and t just inside and just
// outside [-128, 128], three lifts each) and 4096 others; egemm_ref.py holds every answer to receipt_folds_r9.key_ok_r9
__global__ void r9_probe(const uint32_t* a, uint8_t* ok, int n) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (w < n) {
    const bool r = xin_ok_r9(a[w]);
    if ((threadIdx.x & 31) == 0) ok[w] = r;
  }
}
static std::vector<uint32_t> r9_words() {
  std::vector<uint32_t> v;
  for (int b = 1; b <= 128; b++) {
    const int e = __builtin_ctz(b), s = 1 << e;
    const uint32_t o = (uint32_t)b >> e, mod = 1u << (30 - e);
    uint32_t inv = o;   // o^-1 mod 2^32 by Newton's iteration
    for (int i = 0; i < 5; i++) inv *= 2u - o * inv;
    for (int t : {127, 128, 129, 128 + s, -127, -128, -129, -128 - s}) {
      if (t % s) continue;
      const uint32_t a0 = ((uint32_t)(t / s) * inv) & (mod - 1);
      for (uint32_t j : {0u, 1u, 3u}) v.push_back(a0 + j * mod);
    }
  }
  for (uint32_t i = 0; i < 4096; i++) v.push_back(mix32(0xC0FFEE00u + i));
  return v;
}

static int egate(const std::string& dir, int m, int n, int k, uint32_t seed) {
  EP P;
  esetup(P, m, n, k, seed, true);
  const size_t xb = (size_t)m * k, wb = (size_t)n * k;
  auto poison = [&] {
    for (auto [p, b] : {std::pair<void*, size_t>{P.X, xb}, {P.Xq, xb}, {P.Wp, wb}, {P.PA, (size_t)m * 128}, {P.PB, (size_t)n * 128},
                        {P.Y, P.ny * 4}, {P.Yr, P.ny * 4}, {P.R, P.nr * 4}, {P.R2, P.nr * 4}})
      CK(cudaMemsetAsync(p, 0xA5, b, P.st));
  };
  poison();
  const Sig s = sigma(seed, 0);
  qd(P, s);
  we(P, s);
  qp(P, s);
  ltg_run(P.plain, P.st);
  egemm_run(P, t_duo32, P.Y, P.R2);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaGetLastError());
  bool ok = true;
  std::string res;
  auto rec = [&](const std::string& name, unsigned long long y, long long r) {
    res += (res.empty() ? "" : ", ") + std::string("\"") + name + "\": {\"y_bad\": " + std::to_string(y) + ", \"r_bad\": " + std::to_string(r) + "}";
  };
  const unsigned long long ty = ecmp(P, P.Y, P.Yr, P.ny);
  rec("t_duo32", ty, -1);
  ok &= ty == 0;
  // every kernel's output against cuBLASLt's; a duo32 kernel's receipts against t_duo32's, any other kind's against its
  // kind's first kernel's (R3); every kernel's receipts on three tiles to DIR, for egemm_ref.py
  const int tiles_m = m / 128, tiles_n = n / 128;
  const int tl[3][2] = {{0, 0}, {tiles_m / 2, tiles_n / 2}, {tiles_m - 1, tiles_n - 1}};
  auto dump_r = [&](const std::string& name, const uint32_t* R) {
    for (int i = 0; i < 3; i++) {
      const int tm = tl[i][0], tn = tl[i][1];
      dump(dir, ("R_" + name + "_" + std::to_string(tm) + "_" + std::to_string(tn) + ".bin").c_str(),
           R + (size_t)(tm * tiles_n + tn) * 512 * 8, 512 * 8 * 4, true);
    }
  };
  uint32_t* R3;
  CK(cudaMalloc(&R3, P.nr * 4));
  CK(cudaMemsetAsync(R3, 0xA5, P.nr * 4, P.st));
  std::string kinds, kind0;
  const Kern l2h = pick(P, ekern("e_l2h"));
  for (auto& e : EK) {
    const char *name = e.name, *kind = e.kind;
    const Kern kern = pick(P, &e);
    if (*kind) kinds += std::string(kinds.empty() ? "" : ", ") + "\"" + name + "\": \"" + kind + "\"";
    if (kern == t_duo32) { dump_r(name, P.R2); continue; }
    CK(cudaMemsetAsync(P.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(P.R, 0xA5, P.nr * 4, P.st));
    egemm_run(P, kern, P.Y, P.R);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    const bool duo = !strcmp(kind, "duo32"), first = *kind && !duo && kind0 != kind;
    const unsigned long long y = ecmp(P, P.Y, P.Yr, P.ny), r = *kind && !first ? ecmp(P, P.R, duo ? P.R2 : R3, P.nr) : 0;
    if (first) { CK(cudaMemcpyAsync(R3, P.R, P.nr * 4, cudaMemcpyDeviceToDevice, P.st)); kind0 = kind; }
    if (*kind) dump_r(name, P.R);
    rec(name, y, *kind && !first ? (long long)r : -1);
    ok &= y == 0 && r == 0;
  }
  CK(cudaStreamSynchronize(P.st));
  CK(cudaFree(R3));
  {
    const std::vector<uint32_t> pw = r9_words();
    uint32_t* dw;
    uint8_t* dok;
    CK(cudaMalloc(&dw, pw.size() * 4));
    CK(cudaMalloc(&dok, pw.size()));
    CK(cudaMemcpy(dw, pw.data(), pw.size() * 4, cudaMemcpyHostToDevice));
    r9_probe<<<(unsigned)(pw.size() * 32 + 255) / 256, 256, 0, P.st>>>(dw, dok, (int)pw.size());
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    dump(dir, "r9_words.bin", dw, pw.size() * 4, true);
    dump(dir, "r9_ok.bin", dok, pw.size(), true);
    CK(cudaFree(dw));
    CK(cudaFree(dok));
  }
  // each fused producer under the same key: its X, P_A, W', P_B bitwise those of quant<true> and wedit, and e_l2h on them
  for (auto& [sfx, f] : QWE) {
    for (auto [p, b] : {std::pair<void*, size_t>{P.Xf, xb}, {P.PAf, (size_t)m * 128}, {P.Wf, wb}, {P.PBf, (size_t)n * 128},
                        {P.Y, P.ny * 4}, {P.R, P.nr * 4}})
      CK(cudaMemsetAsync(p, 0xA5, b, P.st));
    qf(P, s, f);
    egemm_run(P, l2h, P.Y, P.R, P.mF);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    const unsigned long long fx = ecmp(P, P.Xf, P.X, xb / 4) + ecmp(P, P.PAf, P.PA, (size_t)m * 32) + ecmp(P, P.Wf, P.Wp, wb / 4) +
                                  ecmp(P, P.PBf, P.PB, (size_t)n * 32);
    const unsigned long long y = ecmp(P, P.Y, P.Yr, P.ny), r = ecmp(P, P.R, P.R2, P.nr);
    rec(std::string("qwe") + sfx + "_operands", fx, -1);
    rec(std::string("qwe") + sfx + "_e_l2h", y, (long long)r);
    ok &= fx == 0 && y == 0 && r == 0;
  }
  // the no-write control: poisoned, nothing run; both compares must fail
  CK(cudaMemsetAsync(P.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(P.R, 0xA5, P.nr * 4, P.st));
  const unsigned long long cy = ecmp(P, P.Y, P.Yr, P.ny), cr = ecmp(P, P.R, P.R2, P.nr);
  const bool control_rejected = cy > 0 && cr > 0;
  rec("control", cy, (long long)cr);
  // the tiles egemm_ref.py recomputes from x, W and the key: the operands and e_l2h's output (every kernel's equals
  // cuBLASLt's on device)
  egemm_run(P, l2h, P.Y, P.R);
  CK(cudaStreamSynchronize(P.st));
  std::string tj;
  for (int i = 0; i < 3; i++) {
    const int tm = tl[i][0], tn = tl[i][1];
    const std::string sfx = "_" + std::to_string(tm) + "_" + std::to_string(tn) + ".bin";
    dump(dir, ("x" + sfx).c_str(), P.x + (size_t)tm * 128 * k, 128 * (size_t)k * 2, true);
    dump(dir, ("X" + sfx).c_str(), P.X + (size_t)tm * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("Xq" + sfx).c_str(), P.Xq + (size_t)tm * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("PA" + sfx).c_str(), P.PA + (size_t)tm * 128 * 128, 128 * 128, true);
    dump(dir, ("W" + sfx).c_str(), P.W + (size_t)tn * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("Wp" + sfx).c_str(), P.Wp + (size_t)tn * 128 * k, 128 * (size_t)k, true);
    dump(dir, ("PB" + sfx).c_str(), P.PB + (size_t)tn * 128 * 128, 128 * 128, true);
    std::vector<int> yt(128 * 128);
    for (int r = 0; r < 128; r++)
      CK(cudaMemcpy(yt.data() + r * 128, P.Y + (size_t)(tm * 128 + r) * n + tn * 128, 128 * 4, cudaMemcpyDeviceToHost));
    dump(dir, ("Y" + sfx).c_str(), yt.data(), yt.size() * 4, false);
    tj += std::string(i ? ", " : "") + "[" + std::to_string(tm) + ", " + std::to_string(tn) + "]";
  }
  // the overlap arm: three calls (both buffers, keys 1-3), each output against cuBLASLt's and its receipts against t_duo32's
  // on the same buffer
  for (int c = 0; c < 3; c++) {
    const int b = P.ov & 1;
    CK(cudaMemsetAsync(P.Y, 0xA5, P.ny * 4, P.st)); CK(cudaMemsetAsync(P.R, 0xA5, P.nr * 4, P.st));
    CK(cudaMemsetAsync(P.R2, 0xA5, P.nr * 4, P.st));
    ov_call(P, l2h, false);
    egemm_run(P, t_duo32, P.Yr2, P.R2, P.mO[b]);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    const unsigned long long y = ecmp(P, P.Y, P.Yr, P.ny), y2 = ecmp(P, P.Yr2, P.Yr, P.ny), r = ecmp(P, P.R, P.R2, P.nr);
    rec("overlap_e_l2h_" + std::to_string(c), y + y2, (long long)r);
    ok &= y == 0 && y2 == 0 && r == 0;
  }
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"inv_s\": %.9g, \"grid\": %d, \"key\": [\"%016llx\", \"%016llx\", "
              "\"%016llx\", \"%016llx\"], \"tiles\": [%s], \"kinds\": {%s}, \"device\": {%s}, \"device_ok\": %s, \"control_rejected\": %s}\n",
          m, n, k, seed, P.inv_s, P.grid, (unsigned long long)s.w[0], (unsigned long long)s.w[1], (unsigned long long)s.w[2],
          (unsigned long long)s.w[3], tj.c_str(), kinds.c_str(), res.c_str(), ok ? "true" : "false", control_rejected ? "true" : "false");
  fclose(fp);
  printf("{\"gate\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"device_ok\": %s, \"control_rejected\": %s, \"device\": {%s}}\n",
         dir.c_str(), m, n, k, ok ? "true" : "false", control_rejected ? "true" : "false", res.c_str());
  efree(P);
  return ok && control_rejected ? 0 : 1;
}

static std::vector<std::string> split(const char* s, const char* dflt) {
  std::vector<std::string> v;
  std::string a = s ? s : dflt, cur;
  for (char c : a + ",") {
    if (c == ',') { if (!cur.empty()) v.push_back(cur); cur.clear(); }
    else cur += c;
  }
  return v;
}

static int ebench(int reps, int m, int n, int k) {
  Nv nv;
  const std::vector<std::string> names = split(getenv("EG_ARMS"), "plain,plainq,qplain,qdither,wedit,e_plain,t_duo32,e_l2h");
  EP P;
  esetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
  std::vector<std::function<void(EP&)>> arms;
  for (auto& a : names) arms.push_back(earm(a));
  const int NA = (int)arms.size();
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_arm = [&](int j, int calls) {
    CK(cudaEventRecord(e0, P.st));
    for (int c = 0; c < calls; c++) arms[j](P);
    CK(cudaEventRecord(e1, P.st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return (double)ms / calls;
  };
  std::vector<int> calls(NA);
  for (int j = 0; j < NA; j++) {
    time_arm(j, 1);
    const double t = time_arm(j, 2);
    calls[j] = std::max(2, std::min(400, (int)(20.0 / std::max(t, 1e-4))));
  }
  CK(cudaGetLastError());
  std::vector<std::vector<double>> ms(NA), mhz(NA);
  for (int r = 0; r < reps; r++)
    for (int j = 0; j < NA; j++) {
      const int a = (j + r) % NA;
      ms[a].push_back(time_arm(a, calls[a]));
      mhz[a].push_back(nv.sm());
    }
  CK(cudaGetLastError());
  printf("{\"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d, \"lt_algo\": %d, \"arms\": {", m, n, k, P.grid, P.plain.algo_id);
  for (int j = 0; j < NA; j++) {
    printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", names[j].c_str(), median(ms[j]),
           median(mhz[j]), calls[j]);
    for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
    printf("], \"mhz_all\": [");
    for (size_t i = 0; i < mhz[j].size(); i++) printf("%s%.0f", i ? ", " : "", mhz[j][i]);
    printf("]}");
  }
  printf("}}\n");
  fflush(stdout);
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  efree(P);
  return 0;
}

// per call, the slice's way: ARM and BASE alternate which goes first, each call timed by its own events; EG_COLD=1
// writes a 512 MiB scratch buffer before each call, outside its events, so every call starts with x, W and the rest out
// of L2 (as a serving layer's weights are)
static int eil(int reps, const std::string& arm, const std::string& base, int m, int n, int k) {
  Nv nv;
  EP P;
  esetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
  auto fa = earm(arm), fb = earm(base);
  const bool cold = getenv("EG_COLD") && atoi(getenv("EG_COLD"));
  const size_t flb = (size_t)512 << 20;
  void* fl = nullptr;
  if (cold) CK(cudaMalloc(&fl, flb));
  cudaEvent_t e[4];
  for (auto& x : e) CK(cudaEventCreate(&x));
  auto call = [&](const std::function<void(EP&)>& f, cudaEvent_t a, cudaEvent_t b, int r) {
    if (cold) CK(cudaMemsetAsync(fl, r & 0xFF, flb, P.st));
    CK(cudaEventRecord(a, P.st));
    f(P);
    CK(cudaEventRecord(b, P.st));
  };
  std::vector<double> ta, tb, mz;
  for (int r = -5; r < reps; r++) {
    const bool afirst = (r & 1) != 0;
    call(afirst ? fa : fb, e[0], e[1], 2 * r);
    call(afirst ? fb : fa, e[2], e[3], 2 * r + 1);
    CK(cudaEventSynchronize(e[3]));
    float t0, t1;
    CK(cudaEventElapsedTime(&t0, e[0], e[1]));
    CK(cudaEventElapsedTime(&t1, e[2], e[3]));
    if (r < 0) continue;
    ta.push_back(afirst ? t0 : t1);
    tb.push_back(afirst ? t1 : t0);
    mz.push_back(nv.sm());
  }
  CK(cudaGetLastError());
  const double a = median(ta), b = median(tb);
  printf("{\"il\": \"%s\", \"base\": \"%s\", \"cold\": %s, \"m\": %d, \"n\": %d, \"k\": %d, \"reps\": %d, \"ms\": %.6f, "
         "\"base_ms\": %.6f, \"ratio\": %.4f, \"mhz\": %.0f, \"mhz_min\": %.0f, \"mhz_max\": %.0f}\n",
         arm.c_str(), base.c_str(), cold ? "true" : "false", m, n, k, reps, a, b, a / b, median(mz),
         *std::min_element(mz.begin(), mz.end()), *std::max_element(mz.begin(), mz.end()));
  fflush(stdout);
  for (auto& x : e) CK(cudaEventDestroy(x));
  if (fl) CK(cudaFree(fl));
  efree(P);
  return 0;
}

// three warm-up calls of ARM, then one between cudaProfilerStart and Stop, for ncu --profile-from-start off
static int eone(const std::string& arm, int m, int n, int k) {
  EP P;
  esetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
  auto f = earm(arm);
  for (int i = 0; i < 3; i++) f(P);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaProfilerStart());
  f(P);
  CK(cudaStreamSynchronize(P.st));
  CK(cudaProfilerStop());
  CK(cudaGetLastError());
  printf("{\"one\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d}\n", arm.c_str(), m, n, k, P.grid);
  efree(P);
  return 0;
}

#ifndef EGEMM_LIB
int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return egate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 6 && !strcmp(argv[1], "bench")) return ebench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (argc >= 8 && !strcmp(argv[1], "il")) return eil(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
  if (argc >= 6 && !strcmp(argv[1], "one")) return eone(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: egemm gate DIR M N K SEED | bench REPS M N K | il REPS ARM BASE M N K | one ARM M N K\n");
  return 2;
}
#endif
