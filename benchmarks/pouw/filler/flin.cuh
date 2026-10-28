// ---------------------------------------------------------------- flin: the decode linears as a warp-specialized GEMM with filler warps
// (FILL, round 2; included by dcap.cu after the dfill section, whose dseed/dword/dcval/drec/dstep/dchain_ref it reuses,
// and pfill.cuh's pf_* on a pfill build)
//
// D (M x N, row-major s32) = X (M x K s8) W^T (W N x K s8), computed as D^T = W X^T: the weights are the MMA's A operand
// (16 output features a fragment), the tokens its B operand (8 a fragment). Weights and activations are repacked once into
// tiles that are contiguous in memory and already in mma.sync m16n8k32's fragment order:
//   W tile (nt, kb): BN features x BK of k, as [i < BN/16][c < BK/32][lane][a0 a1 a2 a3] (512 B a fragment);
//   X tile kb: M tokens x BK of k, as [j < M/8][p < BK/64][lane][b0(2p) b1(2p) b0(2p+1) b1(2p+1)];
// so one bulk async copy (cp.async.bulk, the TMA engine) moves a stage's W or X tile, and a lane's fragment is one 16- or
// 8-byte shared load with no bank conflict. One CTA a SM, three roles:
//   producer warp: lane 0 streams the CTA's units (W tile + X tile) into an ST-stage ring guarded by full/empty mbarriers;
//   consumer warps (CWN x CWM): the served MMAs, a warp BN/CWN features x M/CWM tokens, partial sums to D by red.add.s32;
//   filler warps (FW): round 1's chain (dfill_k's tile, from the coins and the warp's slot) until every consumer is done,
//     then the tile's receipt and its step count; with thr set, a filler warp steps only while every consumer waits on a
//     stage (it sleeps otherwise).
// Work is split stream-K: the (N/BN) x (K/BK) units in tile-major order, a contiguous run of them a CTA, so every SM streams
// the same bytes. Partial tiles meet in D by integer red.add, exact in any order; D must start at zero, so each CTA also
// zeroes its share of the next linear's output (Z, zn ints, a multiple of 4).
constexpr int FLIN_BN = 128;
// FLIN_FIT: tiles whose GEMM-only CTA (FW = 0) fits beside one pump block on its SM, so a co-scheduled pump never holds a
// linear off (flin_cfg): ≤ 41 KB of shared memory, and registers by SMSP, since each SMSP's file is 16 K and the pump's
// two warps there hold 8 K. A 5-warp CTA puts two warps on one SMSP, so ≤ 128 a thread (launch bounds of 4 blocks: 102).
// M 128 at BN 128 would need 9 warps, three on one SMSP at ≤ 85, which its accumulators (64 a thread) don't leave,
// so it runs BN 64 tiles on 2 x 2 consumer warps, each warp the M 64 tile's 32 x 64 (the same
// tiled W: a 64-row tile is half of a 128-row one). FLIN_FIT=2 takes M 32 in four half stages. M > 128 keeps its tile.
#ifdef FLIN_FIT
#define FLIN_MINB(CW) ((CW) <= 4 ? 4 : 2)
#else
#define FLIN_MINB(CW) 1
#endif

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, unsigned n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(sa(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_expect(uint64_t* b, unsigned bytes) {
  asm volatile("{ .reg .b64 s; mbarrier.arrive.expect_tx.shared.b64 s, [%0], %1; }" ::"r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
  asm volatile("{ .reg .b64 s; mbarrier.arrive.shared.b64 s, [%0]; }" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ bool mb_try(uint64_t* b, unsigned par) {
  unsigned ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok)
               : "r"(sa(b)), "r"(par)
               : "memory");
  return ok;
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void mma_acc(int (&d)[4], const uint4& a, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// The filler warps' round, on the device so that one captured graph serves every step of a round: its coins, the tile id
// of the step's first filler slot (the round's first id, tag included, plus step · per) and the step's index, which
// fadv_k advances once a step.
struct FState {
  uint64_t c0, c1;
  unsigned long long base, step;
};
// A launch's filler slots: slot s runs tile base + off + s and leaves its receipt and step count at (step % ring) · per +
// off + s of a ring of ring · per, so a block of up to `ring` steps keeps every step's.
struct FSlots {
  const FState* st;
  unsigned long long* rec;
  int* cnt;
  unsigned long long off, per;
  unsigned ring;
};
__global__ void fadv_k(FState* s, unsigned long long per) {
  s->base += per;
  s->step++;
}

#ifndef DFILL_V1
// ---------------------------------------------------------------- the filler warps' tile on a pfill build (round 4)
// pfill-k's arithmetic (pfill.cuh, -DPFILL_BITS: {0,1} SHA-512 noise, s8 IMMA m16n8k32 with C = 0, every s32 word into
// the ram receipt) in one warp's registers, as a step a serving kernel's filler warp calls: dfill_k_warp(c0, c1, tile,
// step) reads nothing but its arguments (the round's coins, the slot's public tile id, the step's index) and returns the
// lane's share of the step's receipt; every filler MMA is inside it. Lane l hashes one SHA-512 block, (c0, c1, tile,
// PW_DOM, 32 step + l) as pfill's 40-byte message, its row words R[0..15] (pf_row: bit k of R[r] is entry 32 r + k).
// The step's matrices A_i (16 x 32, i < 16) and B_j (32 x 8, j < 28) are those bits in m16n8k32's fragment layout, each
// entry held by the one lane that loads it, so no bit crosses lanes: lane (g, t)'s word q of A_i is the four entries
// A_i[g + 8 (q & 1)][16 (q >> 1) + 4 t + e], e < 4, bits 16 (i & 1) + 4 q + e of R[i >> 1]; its B_j words are
// B_j[16 h + 4 t + e][g], bits 8 (j & 3) + 4 h + e of R[8 + j / 4]. The keys (a0, a1) are the first two pf_key_ok words
// (pf_key_ok_r9 on a -DPFILL_LI or -DPFILL_XIN13 build) among R[15] of lanes 0..31 (pfill's defaults if fewer), bits no
// operand reads. Lane (g, t) keeps chains c < 4 with y = 4 lane + c; chain c absorbs A_{4 a + c} B_j for a < 4, then
// j < 28 (pf_absorb<4>: pfill's ram, or the build's r9 fold), and the lane returns
// the XOR over c of pf_fold(smix(dseed(c0, c1, tile) ^ step), 4 lane + c, y). pw_ref is the same step on the host, from
// the matrices. A step is 448 MMAs from one SHA block a lane (pfill: 4 bits an MMA; here 36, so the hashing is a third
// of a step's issue slots).
constexpr uint64_t PW_DOM = 0x5046494C4C2D7731ull;  // "PFILL-w1"
constexpr int PW_NA = 16, PW_NB = 28, PW_MMAS = PW_NA * PW_NB;

__device__ __noinline__ uint64_t dfill_k_warp(uint64_t c0, uint64_t c1, unsigned long long tile, unsigned long long step) {
  const int lane = threadIdx.x & 31;
  const uint64_t K = smix(dseed(c0, c1, tile) ^ step);  // first, so that only K is live through the hashing
  uint32_t R[16];
  {
    uint64_t st[8];
    pf_block(PW_DOM, c0, c1, tile, step << 5 | lane, st);
#pragma unroll
    for (int r = 0; r < 16; r++) R[r] = pf_row(st, r);
  }
  PfKeys key{0x9E3779B9u, 0x7F4A7C15u, 0, 0};
  const unsigned ok = __ballot_sync(~0u, PF_R9 ? pf_key_ok_r9(R[15]) : pf_key_ok(R[15])), ok1 = ok & (ok - 1);
  const uint32_t k0 = __shfl_sync(~0u, R[15], ok ? __ffs(ok) - 1 : 0), k1 = __shfl_sync(~0u, R[15], ok1 ? __ffs(ok1) - 1 : 0);
  if (ok) key.a0 = k0;
  if (ok1) key.a1 = k1;
  uint32_t y[4];
#pragma unroll
  for (int c = 0; c < 4; c++) y[c] = 4 * lane + c;
#pragma unroll 1
  for (int ag = 0; ag < PW_NA / 4; ag++) {
    uint32_t a[4][4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const uint32_t w = (c < 2 ? R[0] : R[1]) >> (16 * (c & 1));
#pragma unroll
      for (int q = 0; q < 4; q++) a[c][q] = pf_01((w >> (4 * q)) & 15);
    }
#pragma unroll
    for (int j = 0; j < PW_NB; j++) {
      const uint32_t by = (R[8 + (j >> 2)] >> (8 * (j & 3))) & 255;
      const uint2 b = make_uint2(pf_01(by & 15), pf_01(by >> 4));
#pragma unroll
      for (int c = 0; c < 4; c++) {
        uint32_t d[4], z = 0;
        pf_mma(d, a[c], b);
        pf_absorb<4>(y[c], z, d[0], d[1], d[2], d[3], key);
      }
    }
    R[0] = R[2]; R[1] = R[3]; R[2] = R[4]; R[3] = R[5]; R[4] = R[6]; R[5] = R[7];  // group ag + 1's A words
  }
  uint64_t h = 0;
#pragma unroll
  for (int c = 0; c < 4; c++) h ^= pf_fold(K, 4 * lane + c, y[c]);
  return h;
}
// A filler warp's receipt and step count into its slot of the ring: its only global stores, in a dfill_k function of
// their own, since the scorer's path check (filler-scorer/v4) refuses a global store on a filler warp's path outside one.
__device__ __noinline__ void dfill_k_put(unsigned long long* rec, int* cnt, uint64_t h, int steps) {
  *rec = h;
  *cnt = steps;
}
static uint64_t pw_step_ref(uint64_t c0, uint64_t c1, uint64_t tile, uint64_t step) {
  uint32_t R[32][16];
  uint64_t st[8];
  for (int l = 0; l < 32; l++) {
    pf_block_h(PW_DOM, c0, c1, tile, step << 5 | l, st);
    for (int r = 0; r < 16; r++) R[l][r] = pf_row(st, r);
  }
  PfKeys key{0x9E3779B9u, 0x7F4A7C15u, 0, 0};
  for (int l = 0, n = 0; l < 32 && n < 2; l++)
    if (PF_R9 ? pf_key_ok_r9(R[l][15]) : pf_key_ok(R[l][15])) (n++ ? key.a1 : key.a0) = R[l][15];
  uint32_t Ar[PW_NA][16] = {}, Bc[PW_NB][8] = {};  // bit k of Ar[i][row] is A_i[row][k], of Bc[j][n] B_j[k][n]
  for (int l = 0; l < 32; l++) {
    const int g = l >> 2, t = l & 3;
    for (int i = 0; i < PW_NA; i++)
      for (int q = 0; q < 4; q++)
        for (int e = 0; e < 4; e++)
          Ar[i][g + 8 * (q & 1)] |= (R[l][i >> 1] >> (16 * (i & 1) + 4 * q + e) & 1u) << (16 * (q >> 1) + 4 * t + e);
    for (int j = 0; j < PW_NB; j++)
      for (int h = 0; h < 2; h++)
        for (int e = 0; e < 4; e++) Bc[j][g] |= (R[l][8 + j / 4] >> (8 * (j & 3) + 4 * h + e) & 1u) << (16 * h + 4 * t + e);
  }
  const uint64_t K = smix(dseed(c0, c1, tile) ^ step);
  uint64_t h = 0;
  for (int l = 0; l < 32; l++) {
    const int g = l >> 2, t = l & 3;
    for (int c = 0; c < 4; c++) {
      uint32_t y = 4 * l + c, z = 0;
      for (int ag = 0; ag < PW_NA / 4; ag++) {
        const int i = 4 * ag + c;
        for (int j = 0; j < PW_NB; j++)
          pf_absorb<4>(y, z, __builtin_popcount(Ar[i][g] & Bc[j][2 * t]), __builtin_popcount(Ar[i][g] & Bc[j][2 * t + 1]),
                       __builtin_popcount(Ar[i][g + 8] & Bc[j][2 * t]), __builtin_popcount(Ar[i][g + 8] & Bc[j][2 * t + 1]),
                       key);
      }
      h ^= pf_fold(K, 4 * l + c, y);
    }
  }
  return h;
}
// a filler slot's receipt after `steps` steps, the step count fill_tally reads
static uint64_t fwarp_ref(uint64_t c0, uint64_t c1, uint64_t tile, int, int steps) {
  uint64_t h = 0;
  for (int s = 0; s < steps; s++) h ^= pw_step_ref(c0, c1, tile, s);
  return h;
}
static double fwarp_macs(int, long long steps) { return (double)steps * PW_MMAS * 4096.0; }
constexpr int FW_CTL = 1;  // the receipts gate's control: the same slot one step short

// One filler warp: dfill_k_warp's steps for its slot while *done < ndone (and, with thr, only while *wait == ndone), and
// while *stop.prog < stop.at, then its receipt and its step count to the ring. Every lane reads the flags through lane 0.
// LAZY > 0 tests a read of done issued a step earlier. NC is the chain build's and unused here.
struct FStop {
  const int* prog;
  int at;
};
template <int NC, int LAZY = 0>
__device__ __forceinline__ void filler_warp(const FSlots& fs, size_t slot, const int* done, const int* wait, int ndone,
                                            int thr, FStop stop = {nullptr, 0}) {
  const int lane = threadIdx.x & 31;
  const FState rs = *fs.st;
  const size_t at = (size_t)(rs.step % fs.ring) * fs.per + fs.off + slot;
  const unsigned long long tile = rs.base + fs.off + slot;
  const volatile int* vd = done;
  const volatile int* vw = wait;
  const volatile int* vp = stop.prog;
  uint64_t h = 0;
  int steps = 0, raw = LAZY && lane == 0 ? *vd : 0, d = 0;
  for (;;) {
    if constexpr (LAZY == 0) d = __shfl_sync(0xffffffffu, lane == 0 ? *vd : 0, 0);
    if (d >= ndone) break;
    if constexpr (LAZY > 0) {
      d = __shfl_sync(0xffffffffu, raw, 0);
      raw = lane == 0 ? *vd : 0;
    }
    if (vp && __shfl_sync(0xffffffffu, lane == 0 ? *vp : 0, 0) >= stop.at) break;
    if (thr) {
      const int w = __shfl_sync(0xffffffffu, lane == 0 ? *vw : 0, 0);
      if (w < ndone) {
        __nanosleep(64);
        continue;
      }
    }
    h ^= dfill_k_warp(rs.c0, rs.c1, tile, (unsigned long long)steps);
    steps++;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
  if (lane == 0) dfill_k_put(fs.rec + at, fs.cnt + at, h, steps);
}
#else
static uint64_t fwarp_ref(uint64_t c0, uint64_t c1, uint64_t tile, int nc, int steps) {
  return dchain_ref(c0, c1, tile, nc, steps);
}
static double fwarp_macs(int nc, long long steps) { return (double)steps * nc * 4096.0; }
constexpr int FW_CTL = 8;  // the receipts gate's control: the same slot 8 steps short
struct FStop {
  const int* prog;
  int at;
};
// One filler warp: dfill_k's tile for its slot (NC chains), stepping 8 at a time while *done < ndone (and, with thr, only
// while *wait == ndone), then its receipt and its step count to the ring. Every lane reads the flags through lane 0.
// LAZY > 0 steps 8·LAZY at a time and tests a read of done issued a round earlier: the served warps' memory traffic can
// hold a shared load in the MIO queue for hundreds of cycles, and a warp waiting on it stalls its chains (pv's filler
// warps lost a third of their loop there). Such a warp runs up to two rounds past the last served warp. stop is the
// pfill build's (FLIN_STOP) and unused here.
template <int NC, int LAZY = 0>
__device__ __forceinline__ void filler_warp(const FSlots& fs, size_t slot, const int* done, const int* wait, int ndone,
                                            int thr, FStop = {nullptr, 0}) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const FState rs = *fs.st;
  const size_t at = (size_t)(rs.step % fs.ring) * fs.per + fs.off + slot;
  const uint64_t seed = dseed(rs.c0, rs.c1, rs.base + fs.off + slot);
  uint32_t b[2][2], c[4], st[NC][4];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    b[j][0] = dword(seed, 1 + j, g, t);
    b[j][1] = dword(seed, 1 + j, g, 4 + t);
  }
  c[0] = (uint32_t)dcval(seed, g, 2 * t);
  c[1] = (uint32_t)dcval(seed, g, 2 * t + 1);
  c[2] = (uint32_t)dcval(seed, g + 8, 2 * t);
  c[3] = (uint32_t)dcval(seed, g + 8, 2 * t + 1);
#pragma unroll
  for (int ch = 0; ch < NC; ch++) {
    st[ch][0] = dword(seed, 16 + ch, g, t);
    st[ch][1] = dword(seed, 16 + ch, g + 8, t);
    st[ch][2] = dword(seed, 16 + ch, g, 4 + t);
    st[ch][3] = dword(seed, 16 + ch, g + 8, 4 + t);
  }
  const volatile int* vd = done;
  const volatile int* vw = wait;
  int steps = 0, raw = LAZY && lane == 0 ? *vd : 0, d = 0;
  for (;;) {
    if constexpr (LAZY == 0) d = __shfl_sync(0xffffffffu, lane == 0 ? *vd : 0, 0);
    if (d >= ndone) break;
    if constexpr (LAZY > 0) {
      d = __shfl_sync(0xffffffffu, raw, 0);
      raw = lane == 0 ? *vd : 0;
    }
    if (thr) {
      const int w = __shfl_sync(0xffffffffu, lane == 0 ? *vw : 0, 0);
      if (w < ndone) {
        __nanosleep(64);
        continue;
      }
    }
#pragma unroll
    for (int u = 0; u < 8 * (LAZY ? LAZY : 1); u++)
#pragma unroll
      for (int ch = 0; ch < NC; ch++) dstep(st[ch], b[u & 1], c);
    steps += 8 * (LAZY ? LAZY : 1);
  }
  uint64_t h = 0;
#pragma unroll
  for (int ch = 0; ch < NC; ch++)
    h ^= drec(seed, ch, g, t, st[ch][0]) ^ drec(seed, ch, g + 8, t, st[ch][1]) ^ drec(seed, ch, g, 4 + t, st[ch][2]) ^
         drec(seed, ch, g + 8, 4 + t, st[ch][3]);
#pragma unroll
  for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
  if (lane == 0) {
    fs.rec[at] = h;
    fs.cnt[at] = steps;
  }
}
#endif

struct FlinArgs {
  const int8_t* W;  // tiled, [N/BN][K/BK][BN·BK]
  const int8_t* X;  // tiled, [K/BK][M·BK]
  int* D;           // M x N, zero on entry
  int* Z;           // the next linear's output, zeroed here
  long long zn;
  int N, K;
  FSlots fs;  // filler warp `slot` = blockIdx.x · FW + fw
  int thr;
  int stop;  // > 0: the filler warps start no step once consumer warp 0 has `stop` or fewer of the CTA's units left
};

template <int M, int BK, int ST, int CWN, int CWM, int FW, int NC, int BN = FLIN_BN>
__global__ void __launch_bounds__(32 * (CWN * CWM + 1 + FW), FW == 0 ? FLIN_MINB(CWN * CWM) : 1) flin_k(const FlinArgs a) {
  constexpr int CW = CWN * CWM, FM = BN / CWN / 16, TN = M / CWM / 8, KC = BK / 32;
  constexpr int WB = BN * BK, XB = M * BK, SB = WB + XB;
  static_assert(FM >= 1 && TN >= 1 && KC % 2 == 0 && FLIN_BN % BN == 0 && BN % 16 == 0, "flin tile shape");
  extern __shared__ __align__(128) uint8_t sm[];
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + ST * SB);
  uint64_t* empty = full + ST;
  int* flag = reinterpret_cast<int*>(empty + ST);  // [0] consumer warps done, [1] consumer warps waiting on a stage,
                                                    // [2] the units consumer warp 0 has finished
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int KB = a.K / BK, U = a.N / BN * KB;
  const int u0 = (int)((long long)blockIdx.x * U / gridDim.x), u1 = (int)((long long)(blockIdx.x + 1) * U / gridDim.x);
  // W is tiled at FLIN_BN (repack_w), whose 16-row groups are contiguous, so a BN-row tile is a run of one of its tiles
  auto wtile = [&](int nt, int kb) {
    return a.W + ((size_t)(nt * BN / FLIN_BN) * KB + kb) * (FLIN_BN * BK) + (size_t)(nt * BN % FLIN_BN) * BK;
  };
  if (threadIdx.x == 0) {
    for (int s = 0; s < ST; s++) {
      mb_init(&full[s], 1);
      mb_init(&empty[s], CW);
    }
    flag[0] = flag[1] = flag[2] = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#if PF_YIELD && PF_YIELD_EARLY
    atomicAdd(pf_mark(), CW);  // each consumer warp clears its own after its flush
#endif
  }
  __syncthreads();
  // Launched with programmatic stream serialization, the next linear's CTA takes this SM as soon as this one leaves it:
  // its filler warps start at once and its producer prefetches weights, while X, Z and D wait for this grid (griddepcontrol.wait).
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (warp == CW) {  // producer
    if (lane == 0) {
      uint64_t pw, px;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
      const int pre = min(ST, u1 - u0);
      for (int i = 0; i < pre; i++) {
        const int u = u0 + i;
        mb_expect(&full[i], SB);
        bulk_g2s(sm + i * SB, wtile(u / KB, u % KB), WB, &full[i], pw);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int i = 0; i < pre; i++) bulk_g2s(sm + i * SB + WB, a.X + (size_t)((u0 + i) % KB) * XB, XB, &full[i], px);
      for (int u = u0 + pre, i = pre; u < u1; u++, i++) {
        const int s = i % ST;
        while (!mb_try(&empty[s], ((i / ST) - 1) & 1)) {
        }
        mb_expect(&full[s], SB);
        const int nt = u / KB, kb = u % KB;
        bulk_g2s(sm + s * SB, wtile(nt, kb), WB, &full[s], pw);
        bulk_g2s(sm + s * SB + WB, a.X + (size_t)kb * XB, XB, &full[s], px);
      }
    }
    return;
  }
  if (warp > CW) {
    if constexpr (FW > 0) {
      filler_warp<NC>(a.fs, blockIdx.x * FW + (warp - CW - 1), &flag[0], &flag[1], CW, a.thr,
                      FStop{a.stop > 0 ? &flag[2] : nullptr, u1 - u0 - a.stop});
    }
    return;
  }
  // consumer
  const int wn = warp % CWN, wm = warp / CWN, g = lane >> 2, t = lane & 3;
  asm volatile("griddepcontrol.wait;" ::: "memory");
#if PF_YIELD
  int* busy = nullptr;
  if (lane == 0) {
    busy = pf_mark();
    if (!PF_YIELD_EARLY) atomicAdd(busy, 1);
  }
#endif
  {
    const long long n4 = a.zn / 4, z0 = (long long)blockIdx.x * n4 / gridDim.x, z1 = (long long)(blockIdx.x + 1) * n4 / gridDim.x;
    int4* z = reinterpret_cast<int4*>(a.Z);
    for (long long e = z0 + threadIdx.x; e < z1; e += 32 * CW) z[e] = make_int4(0, 0, 0, 0);
  }
  int acc[FM][TN][4];
  auto clear = [&] {
#pragma unroll
    for (int f = 0; f < FM; f++)
#pragma unroll
      for (int j = 0; j < TN; j++)
#pragma unroll
        for (int r = 0; r < 4; r++) acc[f][j][r] = 0;
  };
  auto flush = [&](int tile) {
#pragma unroll
    for (int f = 0; f < FM; f++)
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const int n = tile * BN + (wn * FM + f) * 16 + g, m = (wm * TN + j) * 8 + 2 * t;
        int* d = a.D + (size_t)m * a.N + n;
        atomicAdd(d, acc[f][j][0]);
        atomicAdd(d + a.N, acc[f][j][1]);
        atomicAdd(d + 8, acc[f][j][2]);
        atomicAdd(d + a.N + 8, acc[f][j][3]);
      }
  };
  clear();
  int nt = u0 / KB;
  for (int u = u0, i = 0; u < u1; u++, i++) {
    if (u / KB != nt) {
      flush(nt);
      clear();
      nt = u / KB;
    }
    const int s = i % ST;
    const unsigned par = (i / ST) & 1;
    if (!mb_try(&full[s], par)) {
      if (lane == 0) atomicAdd(&flag[1], 1);
      while (!mb_try(&full[s], par)) {
      }
      if (lane == 0) atomicSub(&flag[1], 1);
    }
    __syncwarp();  // lanes leave the wait at different polls; mma.sync.aligned needs the warp converged
    const uint8_t* ws = sm + s * SB + wn * FM * KC * 512 + lane * 16;
    const uint8_t* xs = sm + s * SB + WB + wm * TN * (KC / 2) * 512 + lane * 16;
    // one k pair's fragments live at a time: the whole stage's would not fit beside the accumulators and the filler warps
#pragma unroll 1
    for (int p = 0; p < KC / 2; p++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint4 aw[FM];
#pragma unroll
        for (int f = 0; f < FM; f++) aw[f] = *reinterpret_cast<const uint4*>(ws + (f * KC + 2 * p + h) * 512);
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const uint2 bx = *reinterpret_cast<const uint2*>(xs + (j * (KC / 2) + p) * 512 + h * 8);
#pragma unroll
          for (int f = 0; f < FM; f++) mma_acc(acc[f][j], aw[f], bx.x, bx.y);
        }
      }
    __syncwarp();
    if (lane == 0) mb_arrive(&empty[s]);
    if (warp == 0 && lane == 0) *(volatile int*)&flag[2] = i + 1;
  }
  flush(nt);
  __syncwarp();
  if (lane == 0) atomicAdd(&flag[0], 1);
#if PF_YIELD
  if (lane == 0) atomicSub(busy, 1);
#endif
}

// ---------------------------------------------------------------- attention blocks with filler warps
// qk_dec's and pv_dec's blocks, unchanged in what their 8 warps do, each with FW filler warps more that run the chain
// until the block's attention warps are done (the hardware still schedules the blocks, so the load stays balanced).
// Filler warp f of block blk (x fastest) is slot blk·FW + f (FSlots). Launched with programmatic stream serialization
// (FATT_PDL 1 or 2), a block's attention warps wait for the previous grid, and so do its filler warps unless `early`
// (FATT_PDL 2), which starts them as soon as the block is resident. The qk blocks also zero pv's output O (on ints),
// which dcap's step does with a memset between the two kernels. Persistent (items > 0, a 1-D grid that fits on the SMs at
// once): a block's attention warps take items blockIdx.x, blockIdx.x + gridDim.x, ... of the 3-D grid's blocks (bx
// fastest, nx of them, then g, then b) and its filler warps run for the whole launch.
struct FattArgs {
  FSlots fs;
  int early;
  long long items;
  int nx;
  int lazy;  // the filler warps' LAZY (0, 2, 4)
};
template <int NC>
__device__ __forceinline__ void att_filler(const FattArgs& fa, size_t slot, const int* done, int ndone) {
  switch (fa.lazy) {
    case 0: filler_warp<NC, 0>(fa.fs, slot, done, done, ndone, 0); break;
    case 2: filler_warp<NC, 2>(fa.fs, slot, done, done, ndone, 0); break;
    default: filler_warp<NC, 4>(fa.fs, slot, done, done, ndone, 0);
  }
}
template <int FW, int NC>
__global__ void __launch_bounds__(QK_W * 32 + 32 * FW, 2) fqk_k(const int8_t* __restrict__ K, const int8_t* __restrict__ q,
                                                                 int* __restrict__ S, int L, int* __restrict__ O, int on,
                                                                 const FattArgs fa) {
  __shared__ int done;
  if (threadIdx.x == 0) done = 0;
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t blk = ((size_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  if (w >= QK_W) {
    if (!fa.early) asm volatile("griddepcontrol.wait;" ::: "memory");
    att_filler<NC>(fa, blk * FW + (w - QK_W), &done, QK_W);
    return;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  auto item = [&](size_t it, size_t nb, int bx, int g, int b) {
    const size_t o0 = it * on / nb, o1 = (it + 1) * on / nb;
    for (size_t e = o0 + threadIdx.x; e < o1; e += QK_W * 32) O[e] = 0;
    qk_body(K, q, S, L, bx, g, b, w, lane);
  };
  if (fa.items == 0) item(blk, (size_t)gridDim.x * gridDim.y * gridDim.z, blockIdx.x, blockIdx.y, blockIdx.z);
  for (long long it = blockIdx.x; it < fa.items; it += gridDim.x) {
    const long long r = it / fa.nx;
    item(it, fa.items, (int)(it % fa.nx), (int)(r % HKV), (int)(r / HKV));
  }
  __syncwarp();
  if (lane == 0) atomicAdd(&done, 1);
}
template <int FW, int NC>
__global__ void __launch_bounds__(PV_W * 32 + 32 * FW, 2) fpv_k(const int8_t* __restrict__ V, const int8_t* __restrict__ P,
                                                                 int* __restrict__ O, int L, const FattArgs fa) {
  __shared__ int red[PV_W][4][HD];
  __shared__ int done;
  if (threadIdx.x == 0) done = 0;
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (w >= PV_W) {
    if (!fa.early) asm volatile("griddepcontrol.wait;" ::: "memory");
    att_filler<NC>(fa, (((size_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * FW + (w - PV_W), &done,
                   PV_W);
    return;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (fa.items == 0) pv_body<true>(V, P, O, L, blockIdx.x, blockIdx.y, blockIdx.z, w, lane, red);
  for (long long it = blockIdx.x; it < fa.items; it += gridDim.x) {
    const long long r = it / fa.nx;
    pv_body<true>(V, P, O, L, (int)(it % fa.nx), (int)(r % HKV), (int)(r / HKV), w, lane, red);
    asm volatile("bar.sync 1, %0;" ::"n"(PV_W * 32) : "memory");  // red is read by every warp before the next item writes it
  }
  __syncwarp();
  if (lane == 0) atomicAdd(&done, 1);
}

// Wt[nt][kb][i][c][lane][4 words] from W (N x K row-major): word r of lane (g, t) is W[row][col..col+3] with
// row = nt·BN + 16i + g + 8·(r & 1), col = kb·BK + 32c + 16·(r >> 1) + 4t (a0 a1 a2 a3 of m16n8k32's A fragment)
__global__ void repack_w(const int8_t* __restrict__ W, int8_t* __restrict__ Wt, int N, int K, int BK) {
  const size_t words = (size_t)N * K / 4;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < words; e += (size_t)gridDim.x * blockDim.x) {
    const int r = e & 3, lane = (e >> 2) & 31, g = lane >> 2, t = lane & 3;
    size_t rest = e >> 7;
    const int KC = BK / 32, IB = FLIN_BN / 16, KB = K / BK;
    const int c = rest % KC; rest /= KC;
    const int i = rest % IB; rest /= IB;
    const int kb = rest % KB;
    const int nt = (int)(rest / KB);
    const int row = nt * FLIN_BN + 16 * i + g + 8 * (r & 1), col = kb * BK + 32 * c + 16 * (r >> 1) + 4 * t;
    reinterpret_cast<int*>(Wt)[e] = *reinterpret_cast<const int*>(W + (size_t)row * K + col);
  }
}
// Xt[kb][j][p][lane][4 words] from X (M x K row-major): words b0(2p) b1(2p) b0(2p+1) b1(2p+1) of lane (g, t), where
// b0(c) = X[8j + g][kb·BK + 32c + 4t ..] and b1(c) the same 16 further along k (m16n8k32's B fragment, col layout)
__global__ void repack_x(const int8_t* __restrict__ X, int8_t* __restrict__ Xt, int M, int K, int BK) {
  const size_t words = (size_t)M * K / 4;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < words; e += (size_t)gridDim.x * blockDim.x) {
    const int r = e & 3, lane = (e >> 2) & 31, g = lane >> 2, t = lane & 3;
    size_t rest = e >> 7;
    const int KP = BK / 64, JB = M / 8;
    const int p = rest % KP; rest /= KP;
    const int j = rest % JB;
    const int kb = (int)(rest / JB);
    const int c = 2 * p + (r >> 1);
    const int row = 8 * j + g, col = kb * BK + 32 * c + 16 * (r & 1) + 4 * t;
    reinterpret_cast<int*>(Xt)[e] = *reinterpret_cast<const int*>(X + (size_t)row * K + col);
  }
}

__global__ void count_nz(const int* __restrict__ p, size_t n, unsigned long long* c) {
  unsigned long long k = 0;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) k += p[e] != 0;
  if (k) atomicAdd(c, k);
}

// ---------------------------------------------------------------- host side
struct FlinCfg {
  int BK, ST, CW, BN = FLIN_BN;
};
static bool flin_m_ok(int M) { return M == 32 || M == 64 || M == 128 || M == 160 || M == 192 || M == 256; }
// the flin tile height a batch of B rows runs at (its rows past B are computed and ignored)
static int flin_m_for(int B) { return B <= 32 ? 32 : B <= 64 ? 64 : B <= 128 ? 128 : B <= 160 ? 160 : B <= 192 ? 192 : 256; }
static FlinCfg flin_cfg(int M) {
#ifdef FLIN_FIT
  if (M == 32) return FLIN_FIT == 2 ? FlinCfg{64, 4, 4} : FlinCfg{128, 2, 4};
  if (M == 64) return {64, 3, 4};
  if (M == 128) return {64, 3, 4, 64};
#endif
  if (M == 32 || M == 64) return {128, 4, 4};
  if (M == 128) return {128, 3, 8};
  return {64, 4, 8};
}
static int flin_smem(int M) {
  const FlinCfg c = flin_cfg(M);
  return c.ST * (c.BN + M) * c.BK + 2 * c.ST * 8 + 16;
}
static bool g_pdl = true;  // FLIN_PDL=0 launches each linear only after the previous grid has finished
static int g_flin_stop = 0;  // FLIN_STOP: Flin.stop (a pfill build's filler warps start no step with that few units left)
static int g_flin_lm = 0;    // FLIN_LM=1: fstep's lm_head through flin (Flin.lm) when any linear is
template <int M, int BK, int ST, int CWN, int CWM, int BN, int FW, int NC>
static void flin_go(const FlinArgs& a, int grid, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    CK(cudaFuncSetAttribute(flin_k<M, BK, ST, CWN, CWM, FW, NC, BN>, cudaFuncAttributeMaxDynamicSharedMemorySize, flin_smem(M)));
#ifdef FLIN_FIT
    CK(cudaFuncSetAttribute(flin_k<M, BK, ST, CWN, CWM, FW, NC, BN>, cudaFuncAttributePreferredSharedMemoryCarveout,
                            cudaSharedmemCarveoutMaxShared));
#endif
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(32 * (CWN * CWM + 1 + FW));
  cfg.dynamicSmemBytes = flin_smem(M);
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = g_pdl ? 1 : 0;
  CK(cudaLaunchKernelEx(&cfg, flin_k<M, BK, ST, CWN, CWM, FW, NC, BN>, a));
}
// the (FW, NC) pairs built for each M: FW = 0 is the GEMM alone. FLIN_PUMP_ONLY (fscore.cu's FSCORE_PUMP_ONLY) builds
// FW = 0 alone and no attention filler warps (fatt_with), so a pump-only arm's fscore holds no clone of dfill_k_warp
// for the scorer to read as its filler.
template <int M, int BK, int ST, int CWN, int CWM, int BN = FLIN_BN>
static bool flin_fw(const FlinArgs& a, int grid, int FW, int NC, cudaStream_t st) {
  const int k = FW == 0 ? 0 : FW * 10 + NC;
  if (k == 0) { flin_go<M, BK, ST, CWN, CWM, BN, 0, 2>(a, grid, st); return true; }
#ifndef FLIN_PUMP_ONLY
  if constexpr (M == 32) {
    if (k == 114) { flin_go<M, BK, ST, CWN, CWM, BN, 11, 4>(a, grid, st); return true; }
  }
  if constexpr (M <= 128) {
    switch (k) {
      case 14: flin_go<M, BK, ST, CWN, CWM, BN, 1, 4>(a, grid, st); return true;
      case 24: flin_go<M, BK, ST, CWN, CWM, BN, 2, 4>(a, grid, st); return true;
    }
  }
  if constexpr (M <= 64) {
    switch (k) {
      case 42: flin_go<M, BK, ST, CWN, CWM, BN, 4, 2>(a, grid, st); return true;
      case 44: flin_go<M, BK, ST, CWN, CWM, BN, 4, 4>(a, grid, st); return true;
      case 82: flin_go<M, BK, ST, CWN, CWM, BN, 8, 2>(a, grid, st); return true;
      case 84: flin_go<M, BK, ST, CWN, CWM, BN, 8, 4>(a, grid, st); return true;
    }
  } else if constexpr (M == 128) {
    switch (k) {
      case 42: flin_go<M, BK, ST, CWN, CWM, BN, 4, 2>(a, grid, st); return true;
      case 44: flin_go<M, BK, ST, CWN, CWM, BN, 4, 4>(a, grid, st); return true;
    }
  } else if constexpr (M < 256) {
    switch (k) {
      case 14: flin_go<M, BK, ST, CWN, CWM, BN, 1, 4>(a, grid, st); return true;
      case 22: flin_go<M, BK, ST, CWN, CWM, BN, 2, 2>(a, grid, st); return true;
      case 24: flin_go<M, BK, ST, CWN, CWM, BN, 2, 4>(a, grid, st); return true;
      case 44: flin_go<M, BK, ST, CWN, CWM, BN, 4, 4>(a, grid, st); return true;
    }
  } else {
    switch (k) {
      case 14: flin_go<M, BK, ST, CWN, CWM, BN, 1, 4>(a, grid, st); return true;
      case 22: flin_go<M, BK, ST, CWN, CWM, BN, 2, 2>(a, grid, st); return true;
      case 24: flin_go<M, BK, ST, CWN, CWM, BN, 2, 4>(a, grid, st); return true;
    }
  }
#endif
  return false;
}
static bool flin_launch(int M, const FlinArgs& a, int grid, int FW, int NC, cudaStream_t st) {
  switch (M) {
#if FLIN_FIT == 2
    case 32: return flin_fw<32, 64, 4, 4, 1>(a, grid, FW, NC, st);
    case 64: return flin_fw<64, 64, 3, 4, 1>(a, grid, FW, NC, st);
    case 128: return flin_fw<128, 64, 3, 2, 2, 64>(a, grid, FW, NC, st);
#elif defined(FLIN_FIT)
    case 32: return flin_fw<32, 128, 2, 4, 1>(a, grid, FW, NC, st);
    case 64: return flin_fw<64, 64, 3, 4, 1>(a, grid, FW, NC, st);
    case 128: return flin_fw<128, 64, 3, 2, 2, 64>(a, grid, FW, NC, st);
#else
    case 32: return flin_fw<32, 128, 4, 4, 1>(a, grid, FW, NC, st);
    case 64: return flin_fw<64, 128, 4, 4, 1>(a, grid, FW, NC, st);
    case 128: return flin_fw<128, 128, 3, 4, 2>(a, grid, FW, NC, st);
#endif
    case 160: return flin_fw<160, 64, 4, 4, 2>(a, grid, FW, NC, st);
    case 192: return flin_fw<192, 64, 4, 4, 2>(a, grid, FW, NC, st);
    case 256: return flin_fw<256, 64, 4, 4, 2>(a, grid, FW, NC, st);
  }
  return false;
}
#ifndef FLIN_PUMP_ONLY
static const char* flin_built(int M) {
  return M == 32    ? "0, 1:4, 2:4, 4:2, 4:4, 8:2, 8:4, 11:4"
         : M == 64  ? "0, 1:4, 2:4, 4:2, 4:4, 8:2, 8:4"
         : M == 128 ? "0, 1:4, 2:4, 4:2, 4:4"
         : M < 256  ? "0, 1:4, 2:2, 2:4, 4:4"
                    : "0, 1:4, 2:2, 2:4";
}
#else
static const char* flin_built(int) { return "0"; }
#endif

// The filler warps' ring on the host (FSlots): the round's coins and first tile id (its tag included), the slots a step
// and the ring's depth in steps.
struct FRing {
  FState* st = nullptr;
  unsigned long long* rec = nullptr;
  int* cnt = nullptr;
  unsigned long long per = 0, lo = 0;
  unsigned ring = 1;
  uint64_t c0 = 0, c1 = 0;
  size_t cap = 0;
};
// opens a round on the ring: `ring` steps of `per` slots, the first step's slots at tiles lo, lo + 1, ..., coins (c0, c1),
// every receipt poisoned and every count 0
static void fring_open(FRing& r, unsigned long long per, unsigned ring, uint64_t c0, uint64_t c1, unsigned long long lo,
                       cudaStream_t st) {
  const size_t n = (size_t)std::max(per, 1ull) * ring;
  if (n > r.cap) {
    if (r.rec) {
      CK(cudaFree(r.rec));
      CK(cudaFree(r.cnt));
    }
    CK(cudaMalloc(&r.rec, n * sizeof(unsigned long long)));
    CK(cudaMalloc(&r.cnt, n * sizeof(int)));
    r.cap = n;
  }
  if (!r.st) CK(cudaMalloc(&r.st, sizeof(FState)));
  r.per = per;
  r.ring = ring;
  r.c0 = c0;
  r.c1 = c1;
  r.lo = lo;
  const FState s{c0, c1, lo, 0};
  CK(cudaMemcpy(r.st, &s, sizeof s, cudaMemcpyHostToDevice));
  CK(cudaMemsetAsync(r.rec, 0xA5, n * sizeof(unsigned long long), st));
  CK(cudaMemsetAsync(r.cnt, 0, n * sizeof(int), st));
  CK(cudaStreamSynchronize(st));
}
static FState fring_state(const FRing& r) {
  FState s;
  CK(cudaMemcpy(&s, r.st, sizeof s, cudaMemcpyDeviceToHost));
  return s;
}
static FSlots fslots(const FRing& r, unsigned long long off) { return {r.st, r.rec, r.cnt, off, r.per, r.ring}; }

// The step's linears through flin: every layer's weights tiled for M, X tiled for K = HID and K = FF, two output buffers
// (a sequence's k-th flin launch writes Db[k & 1] and zeroes Db[(k + 1) & 1] for the next; a sequence has an even count).
// Launch i's filler slots are [i · stride, i · stride + grid · FW) of the step's layout (stride: grid · the configured FW).
// With lm (flin_prepare_lm), lm_head is launch LAUNCHES: weights Wlm tiled, output Dlm, which the sequence's last launch
// zeroes, and it zeroes Db[0] for the next step's first.
struct Flin {
  int M = 0, BK = 0, FW = 0, NC = 2, thr = 0, grid = 0, stop = 0;
  bool lm = false;
  int8_t* Wt[LAYERS][4] = {};
  int8_t *XtH = nullptr, *XtF = nullptr, *Wlm = nullptr;
  int* Db[2] = {};
  int* Dlm = nullptr;
  const FRing* ring = nullptr;
  unsigned long long stride = 0;
  static constexpr int LAUNCHES = LAYERS * 4;
};
static void flin_tile_x(const Dev& d, Flin& f) {
  const size_t h = (size_t)f.M * HID, ff = (size_t)f.M * FF;
  if (!f.XtH) CK(cudaMalloc(&f.XtH, h));
  if (!f.XtF) CK(cudaMalloc(&f.XtF, ff));
  repack_x<<<1024, 256, 0, d.st>>>(d.X, f.XtH, f.M, HID, f.BK);
  repack_x<<<1024, 256, 0, d.st>>>(d.X, f.XtF, f.M, FF, f.BK);
}
static void flin_prepare(Dev& d, Flin& f, int M, bool all_layers) {
  f.M = M;
  f.BK = flin_cfg(M).BK;
  f.grid = d.sms;
  for (int l = 0; l < (all_layers ? LAYERS : 1); l++)
    for (int j = 0; j < 4; j++) {
      const size_t n = (size_t)LN[j] * LK[j];
      if (!f.Wt[l][j]) CK(cudaMalloc(&f.Wt[l][j], n));
      repack_w<<<4096, 256, 0, d.st>>>(d.W[l][j], f.Wt[l][j], LN[j], LK[j], f.BK);
    }
  flin_tile_x(d, f);
  f.Db[0] = reinterpret_cast<int*>(reinterpret_cast<char*>(d.D) + ((size_t)256 << 20));
  f.Db[1] = reinterpret_cast<int*>(reinterpret_cast<char*>(d.D) + ((size_t)512 << 20));
  CK(cudaMemsetAsync(f.Db[0], 0, (size_t)M * 2 * FF * sizeof(int), d.st));
  CK(cudaMemsetAsync(f.Db[1], 0, (size_t)M * 2 * FF * sizeof(int), d.st));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
}
// lm_head's weights tiled for f.M and its output Dlm (d.D past cuBLASLt's outputs and Db), zeroed for the first step
static void flin_prepare_lm(Dev& d, Flin& f) {
  f.lm = true;
  if (!f.Wlm) CK(cudaMalloc(&f.Wlm, (size_t)VOCAB * HID));
  repack_w<<<4096, 256, 0, d.st>>>(d.Wlm, f.Wlm, VOCAB, HID, f.BK);
  f.Dlm = reinterpret_cast<int*>(reinterpret_cast<char*>(d.D) + ((size_t)768 << 20));
  CK(cudaMemsetAsync(f.Dlm, 0, (size_t)f.M * VOCAB * sizeof(int), d.st));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
}
// launch i (0 ≤ i < 128: layer i / 4, linear i % 4) as its sequence's k-th flin launch, the next one linear next_j
// (4: lm_head, whose output Dlm it zeroes)
static bool flin_one(Dev& d, const Flin& f, int i, int k, int next_j) {
  const int l = i / 4, j = i % 4;
  FlinArgs a;
  a.W = f.Wt[l][j];
  a.X = j == 3 ? f.XtF : f.XtH;
  a.D = f.Db[k & 1];
  a.Z = next_j == 4 ? f.Dlm : f.Db[(k + 1) & 1];
  a.zn = (long long)f.M * (next_j == 4 ? VOCAB : LN[next_j]);
  a.N = LN[j];
  a.K = LK[j];
  a.fs = fslots(*f.ring, (unsigned long long)i * f.stride);
  a.thr = f.thr;
  a.stop = f.stop;
  return flin_launch(f.M, a, f.grid, f.FW, f.NC, d.st);
}
// lm_head as launch LAUNCHES into Dlm, zeroing Db[0] for the next step's first launch (linear next_j)
static bool flin_lm(Dev& d, const Flin& f, int next_j) {
  FlinArgs a;
  a.W = f.Wlm;
  a.X = f.XtH;
  a.D = f.Dlm;
  a.Z = f.Db[0];
  a.zn = (long long)f.M * LN[next_j];
  a.N = VOCAB;
  a.K = HID;
  a.fs = fslots(*f.ring, (unsigned long long)Flin::LAUNCHES * f.stride);
  a.thr = f.thr;
  a.stop = f.stop;
  return flin_launch(f.M, a, f.grid, f.FW, f.NC, d.st);
}
// the flin launches of a step whose linears `seq` run through flin: seq, then lm_head's when f.lm
static std::vector<int> flin_lin(const Flin& f, const std::vector<int>& seq) {
  std::vector<int> v = seq;
  if (f.lm && !seq.empty()) v.push_back(Flin::LAUNCHES);
  return v;
}
// A launch's filler slots in the ring: n of them from ring index `at`, slot s ran tile tile0 + s with NC chains.
struct FillLaunch {
  size_t at, n;
  unsigned long long tile0;
  int NC;
};
// the filler's MACs over `ls`, and up to n receipts drawn from the slots that stepped, checked on the host against
// fwarp_ref at the ring's coins, each with the control FW_CTL steps short that must differ; `ids` collects the checked tiles less
// `tag`
struct FlinTally {
  double macs = 0;
  long long steps = 0, slots = 0, idle = 0;
  int checked = 0, bad = 0, ctl_equal = 0;
};
static FlinTally fill_tally(const FRing& r, const std::vector<FillLaunch>& ls, int n, uint64_t salt,
                            std::vector<unsigned long long>* ids = nullptr, unsigned long long tag = 0) {
  FlinTally t;
  if (ls.empty()) return t;
  size_t lo = SIZE_MAX, hi = 0;
  for (const FillLaunch& l : ls) {
    lo = std::min(lo, l.at);
    hi = std::max(hi, l.at + l.n);
  }
  std::vector<unsigned long long> rec(hi - lo);
  std::vector<int> cnt(hi - lo);
  CK(cudaMemcpy(rec.data(), r.rec + lo, rec.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cnt.data(), r.cnt + lo, cnt.size() * 4, cudaMemcpyDeviceToHost));
  std::vector<std::pair<int, size_t>> cand;
  for (size_t q = 0; q < ls.size(); q++)
    for (size_t s = 0; s < ls[q].n; s++) {
      const int c = cnt[ls[q].at - lo + s];
      t.steps += c;
      t.macs += fwarp_macs(ls[q].NC, c);
      t.slots++;
      t.idle += c == 0;
      if (c > 0) cand.push_back({(int)q, s});
    }
  std::mt19937_64 rng(salt);
  for (int k = 0; k < n && !cand.empty(); k++) {
    const auto [q, s] = cand[rng() % cand.size()];
    const unsigned long long tile = ls[q].tile0 + s;
    const size_t i = ls[q].at - lo + s;
    t.bad += rec[i] != fwarp_ref(r.c0, r.c1, tile, ls[q].NC, cnt[i]);
    if (cnt[i] >= FW_CTL) t.ctl_equal += rec[i] == fwarp_ref(r.c0, r.c1, tile, ls[q].NC, cnt[i] - FW_CTL);
    t.checked++;
    if (ids) ids->push_back(tile & ~tag);
  }
  return t;
}

// The attention blocks' filler: FW warps a block (0: dcap's own kernels), NC chains each. Launch k = 2·layer is qk,
// 2·layer + 1 is pv; their filler slots are sl[k & 1] each (blocks · FW), at fatt_off(k) of the step's layout. Persistent
// on `sms` SMs (sms > 0): qk's and pv's grids pg are as many blocks as fit on those SMs at once (at most their items).
// lazy: the filler warps' LAZY (filler_warp).
struct Fatt {
  int FW = 0, NC = 4, sms = 0, lazy = 0;
  size_t pg[2] = {0, 0}, sl[2] = {0, 0};
  unsigned long long off0 = 0;
  const FRing* ring = nullptr;
};
static unsigned long long fatt_off(const Fatt& a, int k) {
  return a.off0 + (unsigned long long)(k / 2) * (a.sl[0] + a.sl[1]) + (k & 1) * a.sl[0];
}
static dim3 qk_grid(int B, int L) { return dim3((L + QK_W * QK_KW - 1) / (QK_W * QK_KW), HKV, B); }
static dim3 pv_grid(int B, int L) { return dim3((L / 4 + PV_W * PV_GW - 1) / (PV_W * PV_GW), HKV, B); }
static size_t blocks(dim3 g) { return (size_t)g.x * g.y * g.z; }
template <int FW_, int NC_>
struct FwNc {
  static constexpr int FW = FW_, NC = NC_;
};
// f(FwNc<FW, NC>{}) for a built (FW, NC); false if it isn't
template <class F>
static bool fatt_with(int FW, int NC, F&& f) {
#ifndef FLIN_PUMP_ONLY
  switch (FW * 10 + NC) {
    case 24: f(FwNc<2, 4>{}); return true;
    case 42: f(FwNc<4, 2>{}); return true;
    case 44: f(FwNc<4, 4>{}); return true;
    case 62: f(FwNc<6, 2>{}); return true;
    case 64: f(FwNc<6, 4>{}); return true;
    case 82: f(FwNc<8, 2>{}); return true;
    case 84: f(FwNc<8, 4>{}); return true;
  }
#else
  (void)FW, (void)NC, (void)f;
#endif
  return false;
}
#ifndef FLIN_PUMP_ONLY
static const char* fatt_built() { return "0, 2:4, 4:2, 4:4, 6:2, 6:4, 8:2, 8:4"; }
#else
static const char* fatt_built() { return "0"; }
#endif
static void fatt_prepare(Fatt& a, int FW, int NC, int B, int L, int sms = 0) {
  a.FW = FW;
  a.NC = NC;
  a.sms = sms;
  a.sl[0] = a.sl[1] = 0;
  if (FW == 0) return;
  if (sms > 0)
    fatt_with(FW, NC, [&](auto t) {
      using T = decltype(t);
      int occ[2];
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[0], fqk_k<T::FW, T::NC>, QK_W * 32 + 32 * T::FW, 0));
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[1], fpv_k<T::FW, T::NC>, PV_W * 32 + 32 * T::FW, 0));
      a.pg[0] = std::min(blocks(qk_grid(B, L)), (size_t)occ[0] * sms);
      a.pg[1] = std::min(blocks(pv_grid(B, L)), (size_t)occ[1] * sms);
    });
  a.sl[0] = (sms > 0 ? a.pg[0] : blocks(qk_grid(B, L))) * FW;
  a.sl[1] = (sms > 0 ? a.pg[1] : blocks(pv_grid(B, L))) * FW;
}
// the step's filler layout: flin's launches 0 … LAUNCHES (the last reserved for lm_head), grid · fw slots each, then every
// layer's attention launches; the slots a step
static unsigned long long fused_layout(Flin& f, int fw, Fatt& a) {
  f.stride = (unsigned long long)f.grid * fw;
  a.off0 = (Flin::LAUNCHES + 1) * f.stride;
  return a.off0 + (unsigned long long)LAYERS * (a.sl[0] + a.sl[1]);
}
// step s's filler launches in the ring: flin's launches `lin` (with f and its FW > 0) and the first `layers` layers'
// attention launches (with a and its FW > 0)
static std::vector<FillLaunch> fused_launches(const FRing& r, const Flin* f, const std::vector<int>& lin, const Fatt* a,
                                              int layers, unsigned long long s) {
  const size_t at0 = (size_t)(s % r.ring) * r.per;
  const unsigned long long t0 = r.lo + s * r.per;
  std::vector<FillLaunch> v;
  if (f && f->FW)
    for (int i : lin) v.push_back({at0 + i * f->stride, (size_t)f->grid * f->FW, t0 + i * f->stride, f->NC});
  if (a && a->FW)
    for (int k = 0; k < 2 * layers; k++) v.push_back({at0 + fatt_off(*a, k), a->sl[k & 1], t0 + fatt_off(*a, k), a->NC});
  return v;
}
static int g_fatt_pdl = 0;  // FATT_PDL: the attention filler blocks without PDL (0), with it (1), with early filler warps (2)
static int g_fatt_nc = 4;   // FATT_NC: the attention filler warps' chains (fatt_built)
static int g_fatt_per = 0;  // FATT_PERSIST: the attention blocks with filler warps persistent on the step's SMs
static int g_fatt_lazy = 0;  // FATT_LAZY: the attention filler warps' LAZY (filler_warp)
template <class... P, class... A>
static void pdl_go(void (*k)(P...), dim3 grid, int threads, cudaStream_t st, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(threads);
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = g_fatt_pdl ? 1 : 0;
  CK(cudaLaunchKernelEx(&cfg, k, args...));
}
template <int FW, int NC>
static void fatt_go(Dev& d, const Fatt& a, int B, int L, const int8_t* K, const int8_t* V, int layer) {
  const int kq = 2 * layer, kp = kq + 1;
  const int early = g_fatt_pdl == 2;
  const dim3 gq = qk_grid(B, L), gp = pv_grid(B, L);
  const bool per = a.sms > 0;
  pdl_go(fqk_k<FW, NC>, per ? dim3((unsigned)a.pg[0]) : gq, QK_W * 32 + 32 * FW, d.st, K, (const int8_t*)d.q, d.S, L, d.O,
         B * HQ * HD,
         FattArgs{fslots(*a.ring, fatt_off(a, kq)), early, per ? (long long)blocks(gq) : 0, (int)gq.x, a.lazy});
  pdl_go(fpv_k<FW, NC>, per ? dim3((unsigned)a.pg[1]) : gp, PV_W * 32 + 32 * FW, d.st, V, (const int8_t*)d.P, d.O, L,
         FattArgs{fslots(*a.ring, fatt_off(a, kp)), early, per ? (long long)blocks(gp) : 0, (int)gp.x, a.lazy});
}
// enqueue_att's work for `layer`, with the attention filler when a.FW > 0; false if (FW, NC) isn't built
static bool enqueue_fatt(Dev& d, const Fatt& a, int B, int L, int layer, bool* wrapped) {
  if (a.FW == 0) {
    enqueue_att(d, B, L, layer, wrapped);
    return true;
  }
  const int8_t* K = kv_layer(d, B, L, layer, false, wrapped);
  const int8_t* V = kv_layer(d, B, L, layer, true, wrapped);
  return fatt_with(a.FW, a.NC, [&](auto t) { fatt_go<decltype(t)::FW, decltype(t)::NC>(d, a, B, L, K, V, layer); });
}

// fgate: flin bit for bit against ref_rows (every row) for each linear's shape at M = 32, 64, 128, 160, 192, 256, alone and
// with filler warps (whose receipts are checked against fwarp_ref, with the control FW_CTL steps short), from poisoned outputs; then
// the attention blocks with filler warps against the host (att_gate), with their receipts checked the same way.
static int flin_gate(Dev& d) {
  int fails = 0;
  std::mt19937_64 rng(0xFA77);
  for (const auto [fw, nc, sms, lazy] :
       {std::tuple{2, 4, 0, 0}, {4, 2, 0, 0}, {4, 4, 0, 0}, {6, 2, 0, 0}, {6, 4, 0, 0}, {8, 2, 0, 0}, {8, 4, 0, 0},
        {2, 4, 5, 0}, {4, 2, 5, 0}, {4, 4, 5, 0}, {6, 2, 5, 0}, {6, 4, 5, 0}, {8, 2, 5, 0}, {8, 4, 5, 0},
        {4, 2, 0, 2}, {8, 2, 0, 4}, {2, 4, 5, 4}, {4, 4, 5, 2}, {6, 2, 5, 2}, {6, 2, 5, 4}, {8, 2, 5, 2}, {8, 2, 5, 4}}) {
    Fatt a;
    FRing r;
    fatt_prepare(a, fw, nc, 3, 1536, sms);  // persistent on 5 SMs: every block takes several items
    a.lazy = lazy;
    a.ring = &r;
    // three steps on a ring of four: the gated one, then two more past fadv_k, every step's receipts checked at its tiles
    const unsigned long long lo = 0x7ull << 56, per = a.sl[0] + a.sl[1];
    fring_open(r, per, 4, g_c0, g_c1, lo, d.st);
    char what[80];
    snprintf(what, sizeof what, "attention FW %d NC %d%s lazy %d", fw, nc, sms ? " persistent" : "", lazy);
    bool built = true;
    fails += att_gate(d, rng, "fgate", what, [&](int B, int L) {
      bool wr = false;
      built = enqueue_fatt(d, a, B, L, 0, &wr);
      for (int s = 1; s < 3; s++) {
        fadv_k<<<1, 1, 0, d.st>>>(r.st, r.per);
        enqueue_fatt(d, a, B, L, 0, &wr);
      }
    });
    if (!built) { fprintf(stderr, "fatt: FW %d NC %d not built (%s)\n", fw, nc, fatt_built()); return fails + 1; }
    const FState st = fring_state(r);
    std::vector<FillLaunch> ls;
    for (int s = 0; s < 3; s++) {
      const auto v = fused_launches(r, nullptr, {}, &a, 1, s);
      ls.insert(ls.end(), v.begin(), v.end());
    }
    const FlinTally t = fill_tally(r, ls, 24, 0xA77 + fw * 8 + nc);
    const bool adv = st.step == 2 && st.base == lo + 2 * per;
    const bool ok = t.bad == 0 && t.ctl_equal == 0 && t.checked > 0 && adv;
    printf("{\"mode\": \"fgate\", \"what\": \"%s receipts\", \"steps\": 3, \"advanced\": %s, \"slots\": %lld, "
           "\"filler_steps\": %lld, \"idle_slots\": %lld, \"rec_checked\": %d, \"rec_bad\": %d, \"ctl_equal\": %d, "
           "\"ok\": %s}\n",
           what, adv ? "true" : "false", t.slots, t.steps, t.idle, t.checked, t.bad, t.ctl_equal, ok ? "true" : "false");
    fflush(stdout);
    fails += !ok;
  }
  for (int M : {32, 64, 128, 160, 192, 256}) {
    Flin f;
    FRing r;
    flin_prepare(d, f, M, false);
    f.ring = &r;
    const int fw0 = M == 256 ? 2 : 4, nc0 = M == 160 || M == 192 ? 4 : 2;
    std::vector<std::pair<int, int>> cfgs = {{0, nc0}, {fw0, nc0}};
    if (M >= 160) cfgs.insert(cfgs.end(), {{1, 4}, {2, 2}});
    for (int j = 0; j < 4; j++)
      for (const auto [fw, nc] : cfgs) {
        f.FW = fw;
        f.NC = nc;
        f.stride = (unsigned long long)f.grid * fw;
        fring_open(r, (Flin::LAUNCHES + 1) * f.stride, 1, g_c0, g_c1, 0x5ull << 56, d.st);
        const int i = j;  // layer 0's linear j
        CK(cudaMemsetAsync(f.Db[0], 0, (size_t)M * LN[j] * 4, d.st));
        if (!flin_one(d, f, i, 0, (j + 1) % 4)) { fprintf(stderr, "flin: M %d FW %d not built\n", M, fw); return 1; }
        ref_rows<<<(unsigned)(((size_t)M * LN[j] + 255) / 256), 256, 0, d.st>>>(d.X, d.W[0][j], g_ref, M, 1, LN[j], LK[j]);
        CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, d.st));
        count_neq_rows<<<(unsigned)(((size_t)M * LN[j] + 255) / 256), 256, 0, d.st>>>(f.Db[0], g_ref, M, 1, LN[j], g_cnt);
        unsigned long long bad = 0;
        CK(cudaMemcpyAsync(&bad, g_cnt, sizeof bad, cudaMemcpyDeviceToHost, d.st));
        CK(cudaStreamSynchronize(d.st));
        CK(cudaGetLastError());
        const FlinTally t = fill_tally(r, fused_launches(r, &f, {i}, nullptr, 0, 0), 8, 0x6A7E + M * 16 + j);
        const bool ok = bad == 0 && t.bad == 0 && t.ctl_equal == 0 && (fw == 0 || t.checked > 0);
        printf("{\"mode\": \"fgate\", \"M\": %d, \"linear\": \"%s\", \"N\": %d, \"K\": %d, \"FW\": %d, \"NC\": %d, "
               "\"mismatches\": %llu, \"rec_checked\": %d, \"rec_bad\": %d, \"ctl_equal\": %d, \"filler_steps\": %lld, "
               "\"idle_slots\": %lld, \"ok\": %s}\n",
               M, LNAME[j], LN[j], LK[j], fw, f.NC, bad, t.checked, t.bad, t.ctl_equal, t.steps, t.idle,
               ok ? "true" : "false");
        fflush(stdout);
        fails += !ok;
      }
    // lm_head after layer 0's down (as a step's last two launches): down zeroes a poisoned Dlm, lm_head fills it and
    // zeroes a poisoned Db[0]
    flin_prepare_lm(d, f);
    ref_rows<<<(unsigned)(((size_t)M * VOCAB + 255) / 256), 256, 0, d.st>>>(d.X, d.Wlm, g_ref, M, 1, VOCAB, HID);
    for (const auto [fw, nc] : {std::pair{0, nc0}, {fw0, nc0}}) {
      f.FW = fw;
      f.NC = nc;
      f.stride = (unsigned long long)f.grid * fw;
      fring_open(r, (Flin::LAUNCHES + 1) * f.stride, 1, g_c0, g_c1, 0x5ull << 56, d.st);
      CK(cudaMemsetAsync(f.Db[1], 0, (size_t)M * LN[3] * 4, d.st));
      CK(cudaMemsetAsync(f.Dlm, 0xA5, (size_t)M * VOCAB * 4, d.st));
      CK(cudaMemsetAsync(f.Db[0], 0x5A, (size_t)M * LN[0] * 4, d.st));
      if (!flin_one(d, f, 3, 1, 4) || !flin_lm(d, f, 0)) { fprintf(stderr, "flin: M %d FW %d not built\n", M, fw); return 1; }
      CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, d.st));
      count_neq_rows<<<(unsigned)(((size_t)M * VOCAB + 255) / 256), 256, 0, d.st>>>(f.Dlm, g_ref, M, 1, VOCAB, g_cnt);
      unsigned long long bad = 0, nz = 0;
      CK(cudaMemcpyAsync(&bad, g_cnt, sizeof bad, cudaMemcpyDeviceToHost, d.st));
      CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, d.st));
      count_nz<<<1024, 256, 0, d.st>>>(f.Db[0], (size_t)M * LN[0], g_cnt);
      CK(cudaMemcpyAsync(&nz, g_cnt, sizeof nz, cudaMemcpyDeviceToHost, d.st));
      CK(cudaStreamSynchronize(d.st));
      CK(cudaGetLastError());
      const FlinTally t = fill_tally(r, fused_launches(r, &f, {3, Flin::LAUNCHES}, nullptr, 0, 0), 8, 0x6A7F + M * 16);
      const bool ok = bad == 0 && nz == 0 && t.bad == 0 && t.ctl_equal == 0 && (fw == 0 || t.checked > 0);
      printf("{\"mode\": \"fgate\", \"M\": %d, \"linear\": \"lm_head\", \"N\": %d, \"K\": %d, \"FW\": %d, \"NC\": %d, "
             "\"mismatches\": %llu, \"next_nonzero\": %llu, \"rec_checked\": %d, \"rec_bad\": %d, "
             "\"ctl_equal\": %d, \"filler_steps\": %lld, \"idle_slots\": %lld, \"ok\": %s}\n",
             M, VOCAB, HID, fw, nc, bad, nz, t.checked, t.bad, t.ctl_equal, t.steps, t.idle, ok ? "true" : "false");
      fflush(stdout);
      fails += !ok;
    }
  }
  return fails;
}

#if !defined(DFILL_V1) && !defined(main)
// walone S [reps] (dcap's own build only, so no scored binary holds it): dfill_k_warp with no serving beside it, blocks
// of 8 warps under flin's register cap (128 a thread), as many as fit an SM, each warp S steps of its own slot (tile id
// tile0 + its index) and then its receipt through dfill_k_put: the in-kernel filler's rate and, under ncu, its issue
// slots an MMA beside the pump's (dalone, done). 8 warps' receipts are held to fwarp_ref and to the control one step short.
__global__ void __launch_bounds__(256, 2) kw_alone(uint64_t c0, uint64_t c1, unsigned long long tile0, int steps,
                                                   unsigned long long* rec, int* cnt) {
  const unsigned long long w = (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  uint64_t h = 0;
  for (int s = 0; s < steps; s++) h ^= dfill_k_warp(c0, c1, tile0 + w, (unsigned long long)s);
#pragma unroll
  for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
  if ((threadIdx.x & 31) == 0) dfill_k_put(rec + w, cnt + w, h, steps);
}
static int run_walone(Dev& d, int S, int reps) {
  int per = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kw_alone, 256, 0));
  const int grid = d.sms * per, warps = grid * 8;
  unsigned long long* rec;
  int* cnt;
  CK(cudaMalloc(&rec, warps * sizeof *rec));
  CK(cudaMalloc(&cnt, warps * sizeof *cnt));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  kw_alone<<<grid, 256, 0, d.st>>>(g_c0, g_c1, 0, S, rec, cnt);
  CK(cudaStreamSynchronize(d.st));
  std::vector<double> ms;
  double cs = 0, ps = 0;
  long np = 0;
  for (int r = 1; r <= reps; r++) {
    CK(cudaEventRecord(e0, d.st));
    kw_alone<<<grid, 256, 0, d.st>>>(g_c0, g_c1, (unsigned long long)r * warps, S, rec, cnt);
    CK(cudaEventRecord(e1, d.st));
    while (cudaEventQuery(e1) == cudaErrorNotReady) {
      cs += nvml_mhz(d);
      ps += nvml_w(d);
      np++;
    }
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    ms.push_back(t);
  }
  CK(cudaGetLastError());
  std::vector<unsigned long long> h(warps);
  std::vector<int> n(warps);
  CK(cudaMemcpy(h.data(), rec, warps * sizeof *rec, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(n.data(), cnt, warps * sizeof *cnt, cudaMemcpyDeviceToHost));
  const unsigned long long tile0 = (unsigned long long)reps * warps;
  int bad = 0, ctl = 0;
  for (int i = 0; i < 8; i++) {
    const int w = (int)((long long)i * warps / 8);
    const uint64_t ref = fwarp_ref(g_c0, g_c1, tile0 + w, 0, S);
    bad += h[w] != ref || n[w] != S;
    ctl += h[w] == (ref ^ pw_step_ref(g_c0, g_c1, tile0 + w, S - 1));
  }
  std::sort(ms.begin(), ms.end());
  const double m = ms[ms.size() / 2], mmas = (double)warps * S * PW_MMAS, mhz = np ? cs / np : nvml_mhz(d);
  printf("{\"mode\": \"walone\", \"W\": 8, \"S\": %d, \"blocks_per_sm\": %d, \"grid\": %d, \"warps\": %d, "
         "\"launch_mmas\": %.6e, \"ms\": %.4f, \"mac_per_s\": %.6e, \"share\": %.5f, \"mhz\": %.1f, \"watt\": %.1f, "
         "\"rec_checked\": 8, \"rec_bad\": %d, \"ctl_equal\": %d}\n",
         S, per, grid, warps, mmas, m, mmas * 4096.0 / (m * 1e-3),
         mmas * 4096.0 / (m * 1e-3) / (1024.0 * d.sms * mhz * 1e6), mhz, np ? ps / np : nvml_w(d), bad, ctl);
  fflush(stdout);
  CK(cudaFree(rec));
  CK(cudaFree(cnt));
  return bad || ctl ? 8 : 0;
}
#endif

// flin M FW NC THR: each linear's 32 layers (distinct weights) as one graph: cuBLASLt, flin alone, flin with filler warps
static int run_flin(Dev& d, int M, int FW, int NC, int thr, int reps) {
  Flin f;
  FRing r;
  flin_prepare(d, f, M, true);
  f.thr = thr;
  f.stop = g_flin_stop;
  f.ring = &r;
  f.stride = (unsigned long long)f.grid * FW;
  fring_open(r, (Flin::LAUNCHES + 1) * f.stride, 1, g_c0, g_c1, 0x5ull << 56, d.st);
  std::vector<Lt> p(4);
  for (int j = 0; j < 4; j++) p[j] = lt_make(M, LN[j], LK[j], d.X, d.W[0][j], d.D, d.st);
  for (int j = 0; j < 4; j++) {
    std::vector<int> ls;
    for (int l = 0; l < LAYERS; l++) ls.push_back(l * 4 + j);
    auto fl = [&](int fw) {
      f.FW = fw;
      f.NC = NC;
      for (int k = 0; k < LAYERS; k++)
        if (!flin_one(d, f, ls[k], k, j)) { fprintf(stderr, "flin: M %d FW %d NC %d not built (%s)\n", M, fw, NC, flin_built(M)); exit(1); }
    };
    fl(FW);  // sets the kernels' attributes outside capture
    fl(0);
    CK(cudaStreamSynchronize(d.st));
    const auto vb = time_work(d, [&] { for (int l = 0; l < LAYERS; l++) lt_run(p[j], d.X, d.W[l][j], d.D, d.st); }, reps);
    const auto v0 = time_work(d, [&] { fl(0); }, reps);
    const auto vf = time_work(d, [&] { fl(FW); }, reps);
    const auto vb2 = time_work(d, [&] { for (int l = 0; l < LAYERS; l++) lt_run(p[j], d.X, d.W[l][j], d.D, d.st); }, reps);
    f.FW = FW;
    const FlinTally t = fill_tally(r, fused_launches(r, &f, ls, nullptr, 0, 0), 8, 0xF11 + M * 4 + j);
    const Rep mb = median(vb), m0 = median(v0), mf = median(vf), mb2 = median(vb2);
    const double base = (mb.ms + mb2.ms) / 2, cap = 1024.0 * d.sms * mf.mhz * 1e6;
    const double macs = (double)LAYERS * M * LN[j] * LK[j], bytes = (double)LAYERS * LN[j] * LK[j];
    printf("{\"mode\": \"flin\", \"pdl\": %d, \"M\": %d, \"linear\": \"%s\", \"N\": %d, \"K\": %d, \"FW\": %d, \"NC\": %d, \"thr\": %d, "
           "\"cublas_ms\": %.4f, \"cublas2_ms\": %.4f, \"flin_ms\": %.4f, \"fill_ms\": %.4f, \"flin_vs_cublas\": %.4f, "
           "\"slowdown\": %.4f, \"cublas_gb_s\": %.1f, \"flin_gb_s\": %.1f, \"useful_share\": %.4f, \"fill_share\": %.4f, "
           "\"mhz\": %.0f, \"watt\": %.1f, \"filler_steps\": %lld, \"idle_slots\": %lld, \"rec_checked\": %d, \"rec_bad\": %d, "
           "\"ctl_equal\": %d}\n",
           (int)g_pdl, M, LNAME[j], LN[j], LK[j], FW, NC, thr, mb.ms, mb2.ms, m0.ms, mf.ms, m0.ms / base, mf.ms / base,
           bytes / (base * 1e-3) / 1e9, bytes / (m0.ms * 1e-3) / 1e9, macs / (mf.ms * 1e-3) / cap,
           t.macs / (mf.ms * 1e-3) / cap, mf.mhz, mf.watt, t.steps, t.idle, t.checked, t.bad, t.ctl_equal);
    fflush(stdout);
    if (t.bad || t.ctl_equal) return 1;
  }
  return 0;
}

// fstep B L FW NC THR AFW [FILL_SMS]: the decode step (enqueue_step's order) with its linears through flin, but those whose
// bit is set in FLIN_LT (1 qkv, 2 o, 4 gate_up, 8 down), which stay on cuBLASLt, and its attention blocks with AFW filler warps
// each (0: dcap's kernels); lm_head stays on cuBLASLt unless FLIN_LM=1 runs it through flin too. Arms: the cuBLASLt step on the whole GPU, flin alone with dcap's
// attention, both with filler warps, the cuBLASLt step again. With FILL_SMS > 0 the two flin arms run on a green split's
// other SMs and the last beside round 1's pump (FSTEP_PUMP=W:NC:S, default 8:4:2048) on FILL_SMS SMs, whose window share and
// receipts are reported with the rest. FILL_SMS -1 (co): the pump on every SM at the lowest stream priority and the flin
// arms on the whole GPU at the highest, so with FW = AFW = 0 the last arm is the step beside a co-resident pump (a
// FLIN_FIT build with a 1-block pump, PF_CTAS=1). FSTEP_KIND=0|1 runs only the linears and lm_head, or only attention, in
// every arm. FLIN_GRID > 0: flin's CTAs (default one an SM; a split's step SMs); in a PF_YIELD build the pump yields the
// SMs they occupy. A batch B ≤ 256 runs flin at the next tile height (flin_m_for), attention and the cuBLASLt
// step at B. The filler warps' MACs are their step counts over the last rep's launches (each step's tiles are the next
// `per` ids, fused_layout); the last linear's output is checked on every row when down runs through flin, and so is
// lm_head's when it does.
static int g_lt_mask = 0;
static int g_fstep_kind = 2;  // FSTEP_KIND: fstep's linears and lm_head only (0), its attention only (1), or both (2)
static std::vector<int> fstep_seq() {
  std::vector<int> s;
  for (int i = 0; i < Flin::LAUNCHES; i++)
    if (!(g_lt_mask >> (i % 4) & 1)) s.push_back(i);
  return s;
}
static void enqueue_fstep(Dev& d, const Flin& f, const Fatt& a, const std::vector<Lt>& p, int B, int L, bool* wrapped) {
  static const std::vector<int> seq = fstep_seq();
  const bool lm = f.lm && !seq.empty();
  int k = 0;
  auto lin = [&](int i) {
    const int j = i % 4;
    if (g_fstep_kind == 1) return;
    if (g_lt_mask >> j & 1) {
      lt_run(p[j], d.X, d.W[i / 4][j], d.D, d.st);
      return;
    }
    flin_one(d, f, i, k, lm && k + 1 == (int)seq.size() ? 4 : seq[(k + 1) % seq.size()] % 4);
    k++;
  };
  for (int l = 0; l < LAYERS; l++) {
    lin(4 * l);
    if (g_fstep_kind != 0) enqueue_fatt(d, a, B, L, l, wrapped);
    for (int j = 1; j < 4; j++) lin(4 * l + j);
  }
  if (g_fstep_kind == 1)
    ;
  else if (lm)
    flin_lm(d, f, seq[0] % 4);
  else if (B <= 256 && p[4].algo_id != -2)
    lt_run(p[4], d.X, d.Wlm, d.D, d.st);
  fadv_k<<<1, 1, 0, d.st>>>(f.ring->st, f.ring->per);
}
static int run_fstep(Dev& d, int B, int L, int FW, int NC, int thr, int AFW, int fill_sms, int reps) {
  if (g_lt_mask & ~15) { fprintf(stderr, "FLIN_LT: bits 1 qkv, 2 o, 4 gate_up, 8 down\n"); exit(1); }
  if (const char* e = getenv("FSTEP_KIND")) g_fstep_kind = atoi(e);
  if (g_fstep_kind < 0 || g_fstep_kind > 2) { fprintf(stderr, "FSTEP_KIND: 0 linears, 1 attention, 2 both\n"); exit(1); }
  const std::vector<int> seq = fstep_seq();
  const std::vector<Lt> p = lin_plans(d, B);
  Flin f;
  flin_prepare(d, f, flin_m_for(B), true);
  if (const char* e = getenv("FLIN_GRID"); e && atoi(e) > 0) f.grid = atoi(e);
  if (g_flin_lm && !seq.empty()) flin_prepare_lm(d, f);
  f.thr = thr;
  f.stop = g_flin_stop;
  f.NC = NC;
  Fatt a0, a;
  cudaStream_t full = d.st, ss = d.st, fs = nullptr;
  int got[2] = {d.sms, 0}, lo = 0, hi = 0, pw = 8, pnc = 4, ps = 2048, smem_sm = 0;
  if (const char* e = getenv("FSTEP_PUMP")) sscanf(e, "%d:%d:%d", &pw, &pnc, &ps);
  CK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
  const bool co = fill_sms < 0;
  if (co) {
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CK(cudaStreamCreateWithPriority(&ss, cudaStreamNonBlocking, hi));
    CK(cudaStreamCreateWithPriority(&fs, cudaStreamNonBlocking, lo));
    got[1] = d.sms;
  } else if (fill_sms > 0) {
    const int e = gc_split(d.sms - fill_sms, &ss, &fs, got);
    if (e || !fs) {
      fprintf(stderr, "gc_split(%d): %d (filler SMs %d)\n", d.sms - fill_sms, e, got[1]);
      return 7;
    }
    f.grid = got[0];
  }
  fatt_prepare(a, AFW, g_fatt_nc, B, L, g_fatt_per ? got[0] : 0);
  a.lazy = g_fatt_lazy;
  FRing r;
  f.ring = a.ring = &r;
  fring_open(r, fused_layout(f, FW, a), 1, g_c0, g_c1, 0x5ull << 56, d.st);
  DFill pf;
  if (fs) {
    dfill_prep();
    pf = dfill_make(got[1], pw, pnc, ps, g_c0, g_c1);
  }
  bool wrapped = false;
  d.st = ss;
  if (!enqueue_fatt(d, a, B, L, 0, &wrapped)) { fprintf(stderr, "fatt: FW %d NC %d not built (%s)\n", AFW, g_fatt_nc, fatt_built()); exit(1); }
  for (int fw : {FW, 0}) {  // the kernels' attributes, outside capture
    f.FW = fw;
    if (!seq.empty() && !flin_one(d, f, seq[0], 0, seq[1] % 4)) {
      fprintf(stderr, "flin: M %d FW %d NC %d not built (%s)\n", B, fw, NC, flin_built(B));
      exit(1);
    }
  }
  CK(cudaStreamSynchronize(ss));
  auto base = [&] { enqueue_step(d, p, g_fstep_kind, B, L, &wrapped); };
  auto fused = [&](int fw, const Fatt& at) { f.FW = fw; enqueue_fstep(d, f, at, p, B, L, &wrapped); };
  double amhz = 0, awatt = 0, alone = 0;
  if (fs) alone = dfill_alone(d, pf, fs, 20, &amhz, &awatt);
  d.st = full;
  const auto vb = time_work(d, base, reps);
  d.st = ss;
  const auto v0 = time_work(d, [&] { fused(0, a0); }, reps);
  DPump pump;
  if (fs) {
    dpump_start(pump, pf, fs, 3);
    while (pump.done.load() < 2) {
    }
  }
  double w0 = 0, w1 = 0;
  const auto vf = time_work(d, [&] { fused(FW, a); }, reps, &w0, &w1);
  long long nwin = 0;
  int pbad = 0;
  if (fs) {
    dpump_stop(pump);
    for (double x : pump.t) nwin += x > w0 && x <= w1;
    pbad = dfill_check(pf, pump.launched.load() - 1, 8);
  }
  d.st = full;
  const auto vb2 = time_work(d, base, reps);
  f.FW = FW;
  const unsigned long long last = fring_state(r).step - 1;  // vf's last step
  const FlinTally t = fill_tally(r, fused_launches(r, &f, flin_lin(f, seq), nullptr, 0, last), 16, 0xF57E + B * 8 + L),
                  ta = fill_tally(r, fused_launches(r, nullptr, {}, &a, LAYERS, last), 16, 0xA57E + B * 8 + L);
  // the last flin launch is down's of the last layer unless FLIN_LT keeps down on cuBLASLt, whose output lm_head overwrites;
  // lm_head's through flin is checked too
  const int out_rows = g_lt_mask & 8 || g_fstep_kind == 1 ? 0 : B, N = LN[3], K = LK[3];
  unsigned long long out_bad = 0;
  if (out_rows) {
    ref_rows<<<(unsigned)(((size_t)B * N + 255) / 256), 256, 0, d.st>>>(d.X, d.W[LAYERS - 1][3], g_ref, B, 1, N, K);
    CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, d.st));
    count_neq_rows<<<(unsigned)(((size_t)B * N + 255) / 256), 256, 0, d.st>>>(f.Db[(seq.size() - 1) & 1], g_ref, B, 1, N, g_cnt);
    CK(cudaMemcpyAsync(&out_bad, g_cnt, sizeof out_bad, cudaMemcpyDeviceToHost, d.st));
  }
  unsigned long long lm_bad = 0;
  if (f.lm && g_fstep_kind != 1) {
    ref_rows<<<(unsigned)(((size_t)B * VOCAB + 255) / 256), 256, 0, d.st>>>(d.X, d.Wlm, g_ref, B, 1, VOCAB, HID);
    CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, d.st));
    count_neq_rows<<<(unsigned)(((size_t)B * VOCAB + 255) / 256), 256, 0, d.st>>>(f.Dlm, g_ref, B, 1, VOCAB, g_cnt);
    CK(cudaMemcpyAsync(&lm_bad, g_cnt, sizeof lm_bad, cudaMemcpyDeviceToHost, d.st));
  }
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  double lin_macs = 0;
  for (int j = 0; j < 4; j++) lin_macs += (double)LAYERS * B * LN[j] * LK[j];
  const double step_macs = (g_fstep_kind != 1 ? lin_macs + (double)B * VOCAB * HID : 0.0) +
                           (g_fstep_kind != 0 ? 2.0 * LAYERS * B * HQ * HD * (double)L : 0.0);
  const Rep mb = median(vb), m0 = median(v0), mf = median(vf), mb2 = median(vb2);
  const double bms = (mb.ms + mb2.ms) / 2, cap = 1024.0 * d.sms * mf.mhz * 1e6;
  const double lin_share = t.macs / (mf.ms * 1e-3) / cap, att_share = ta.macs / (mf.ms * 1e-3) / cap;
  const double pump_share = fs ? (double)nwin * pf.macs() / ((w1 - w0) * 1e-3) / cap : 0.0;
  printf("{\"mode\": \"fstep\", \"place\": \"%s\", \"prio\": [%d, %d], \"smem_sm\": %d, \"flin_smem\": %d, \"pdl\": %d, "
         "\"fatt_pdl\": %d, \"lt_mask\": %d, \"fill_sms_req\": %d, \"step_sms\": %d, "
         "\"fill_sms\": %d, \"pump\": {\"W\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %d, \"alone_share\": %.5f, \"share\": %.5f, "
         "\"chunks\": %lld, \"chunk_macs\": %.6e, \"rec_checked\": %d, \"rec_bad\": %d}, \"fill_share_pump\": %.5f, "
         "\"fill_share_total\": %.5f, ",
         co ? "co" : "split", lo, hi, smem_sm, flin_smem(f.M), (int)g_pdl, g_fatt_pdl, g_lt_mask, fill_sms, got[0], got[1],
         pw, pnc, ps, fs ? pf.tiles : 0, fs ? alone / (1024.0 * d.sms * amhz * 1e6) : 0.0, pump_share, nwin,
         fs ? pf.macs() : 0.0, fs ? 8 : 0, pbad, pump_share, lin_share + att_share + pump_share);
  printf("\"M\": %d, \"kind\": %d, \"flin_grid\": %d, \"yield\": [%d, %d, %d, %d, %d], ", f.M, g_fstep_kind, f.grid,
         PF_YIELD, PF_YIELD_J, PF_YIELD_NS, PF_YIELD_EARLY, PF_YIELD_POLL_NS);
  printf("\"APS\": %d, \"ALZ\": %d, \"LM\": %d, \"lm_mismatches\": %llu, \"att_grid\": [%zu, %zu], ", a.sms > 0, a.lazy,
         (int)f.lm, lm_bad, a.pg[0], a.pg[1]);
  printf("\"B\": %d, \"L\": %d, \"FW\": %d, \"NC\": %d, \"thr\": %d, \"AFW\": %d, \"ANC\": %d, "
         "\"coins\": [\"%016llx\", \"%016llx\"], \"kv_wrapped\": %s, \"step_macs\": %.6e, \"lin_macs\": %.6e, \"base_ms\": %.4f, "
         "\"base2_ms\": %.4f, \"fused0_ms\": %.4f, \"fused_ms\": %.4f, \"fused0_vs_base\": %.4f, \"slowdown\": %.4f, "
         "\"step_share_base\": %.5f, \"step_share_fused\": %.5f, \"fill_macs\": %.6e, \"fill_share\": %.5f, "
         "\"fill_share_lin\": %.5f, \"fill_share_att\": %.5f, \"mhz\": %.0f, \"watt\": %.1f, "
         "\"lin\": {\"filler_steps\": %lld, \"slots\": %lld, \"idle_slots\": %lld, \"rec_checked\": %d, \"rec_bad\": %d, \"ctl_equal\": %d}, "
         "\"att\": {\"filler_steps\": %lld, \"slots\": %lld, \"idle_slots\": %lld, \"rec_checked\": %d, \"rec_bad\": %d, \"ctl_equal\": %d}, "
         "\"out_rows\": %d, \"out_mismatches\": %llu, \"arms\": {",
         B, L, FW, NC, thr, AFW, a.NC, (unsigned long long)g_c0, (unsigned long long)g_c1, wrapped ? "true" : "false",
         step_macs, lin_macs, mb.ms, mb2.ms, m0.ms, mf.ms, m0.ms / bms, mf.ms / bms,
         step_macs / (bms * 1e-3) / (1024.0 * d.sms * mb.mhz * 1e6), step_macs / (mf.ms * 1e-3) / cap, t.macs + ta.macs,
         lin_share + att_share, lin_share, att_share, mf.mhz, mf.watt, t.steps, t.slots, t.idle, t.checked, t.bad, t.ctl_equal,
         ta.steps, ta.slots, ta.idle, ta.checked, ta.bad, ta.ctl_equal, out_rows, out_bad);
  print_arm("base", vb, true);
  print_arm("fused0", v0, true);
  print_arm("fused", vf, true);
  print_arm("base2", vb2, false);
  printf("}}\n");
  fflush(stdout);
  return t.bad || t.ctl_equal || ta.bad || ta.ctl_equal || out_bad || lm_bad || pbad ? 8 : 0;
}
