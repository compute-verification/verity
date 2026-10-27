// The form escape on the H200 (sm_90a): does a wgmma form other than the honest's run faster beside the LI fold? The
// census prices IGMMA_S8_N256_RS (A from registers) and IGMMA_U8S8_N256 on DC-90's joint rows JI90_0/JI90_1 at a lane
// instruction's default, 64x under the honest SS form's measured lines, so the model lets a cheater run either at the
// issue bound. These loops are ../dc/probes/sm90_wgmma.cu's, templated on the form: each form alone (wait_group 1), the
// honest LI-CP-W loop (wgli_redux: 4 wgmma, LI's 256 LOP3 + 256 IMAD, 4 checkpoints of 4 LOP3 and one redux.sync.xor,
// then wait), and the interleaved IMAD and LOP3 points. The RS form's A fragment is loop-invariant, the cheater's best
// case (no delivery). wgli_true is the honest loop folding the wgmma's own accumulators, after each wait, wgli_pp the
// same loop with its warpgroups ordered (ping-pong: the next warpgroup's barrier is arrived after the wait), wgli_pp2
// with the barrier arrived right after the commit, and wgli_db one warpgroup double-buffering two N128 accumulators
// (fold one while the other's wgmma runs). The b1 form (BGMMA m64n256k256 and.popc), which the census also prices on
// JI90 at the lane default, runs alone, beside the LI fold and at the IMAD and LOP3 points, so its joint coefficients
// can be fitted as RS's were. Built and gated by ../opcode_probe/probe.py; rs.sh builds it on a CPU node and times it
// on the H200, rs.py reads it.
#include "../opcode_probe/probe.cuh"
#include "../dc/dc.cuh"
#include "../dc/probes/lanes.cuh"
#include "../dc/wgmma.cuh"

constexpr int KB = 32, AROWS = 64, BROWS = 256;

struct Tiles {
  uint64_t da, db;
};
DC_DEV Tiles fill_tiles(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x)
    w[i] = mode == 1 ? 0u : dc_mix32(i * 0x9E3779B9u + 0x1234567u);
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB))};
}

struct F_ss {
  DC_DEV static void mma(int (&d)[128], const Tiles& t, const uint32_t (&)[4], int sd) { wgmma_s8_n256(d, t.da, t.db, sd); }
};
struct F_rs {
  DC_DEV static void mma(int (&d)[128], const Tiles& t, const uint32_t (&a)[4], int sd) { wgmma_s8_n256_rs(d, a, t.db, sd); }
};
struct F_u8 {
  DC_DEV static void mma(int (&d)[128], const Tiles& t, const uint32_t (&)[4], int sd) { wgmma_u8s8_n256(d, t.da, t.db, sd); }
};
// m64n256k256 b1 and.popc on the same 32-byte k-major tiles (k256 bits a row is s8's k32 bytes)
struct F_b1 {
  DC_DEV static void mma(int (&d)[128], const Tiles& t, const uint32_t (&)[4], int sd) { wgmma_b1_n256(d, t.da, t.db, sd); }
};

template <class F>
DC_DEV void group4(int (&d)[128], const Tiles& t, const uint32_t (&a)[4]) {
  wgmma_fence();
  F::mma(d, t, a, 0);
  F::mma(d, t, a, 1);
  F::mma(d, t, a, 1);
  F::mma(d, t, a, 1);
  wgmma_commit();
}

template <int N>
DC_DEV uint32_t fold_acc(const int (&d)[N]) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < N; i++) x ^= (uint32_t)d[i];
  return x;
}

#define SETUP                                                                                                   \
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];                                            \
  const Tiles t = fill_tiles(sm, a.mode);                                                                       \
  const uint32_t af[4] = {(uint32_t)t.da ^ threadIdx.x, (uint32_t)(t.da >> 32), (uint32_t)t.db, 0x01020304u}; \
  int d[128] = {};

template <class F>
DC_DEV void wg_alone(const OpArgs& a) {
  SETUP
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t, af);
    wgmma_wait<1>();
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ threadIdx.x)
}

template <class F>
DC_DEV void wg_li(const OpArgs& a) {
  SETUP
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t, af);
#pragma unroll
    for (int c = 0; c < 256; c++) {
      l.op<L_LOP3>(c);
      l.op<L_IMAD>(c);
    }
#pragma unroll
    for (int s = 0; s < 4; s++) {
      l.op<L_LOP3>(0);
      l.op<L_LOP3>(2);
      l.op<L_LOP3>(4);
      l.op<L_LOP3>(6);
      l.op<L_REDUX>(0);
    }
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}

template <class F, int X, int R>
DC_DEV void wg_mix(const OpArgs& a) {
  SETUP
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t, af);
    lane_ops<X, R>(l);
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}

// The receipt's LI fold: the 8 chains and their multipliers, N/2 pairs of a step's accumulators folded in
// (y = (y ^ d0 ^ k) * a + d1: one LOP3 and one IMAD a pair), and a segment's checkpoint (the 8 chains XORed by 4 LOP3
// and reduced by one redux.sync.xor).
struct Li {
  uint32_t y[8], m[8], k, cp = 0;
  DC_DEV Li(int mode) : k(dc_mix32(blockIdx.x * 977u + threadIdx.x)) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      y[c] = dc_mix32(threadIdx.x * 8u + c);
      m[c] = dc_mix32(0xA0000000u + c + (uint32_t)mode) | 1u;
    }
  }
  template <int N>
  DC_DEV void fold(const int (&d)[N]) {
#pragma unroll
    for (int p = 0; p < N / 2; p++) {
      uint32_t x;
      asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(x) : "r"(y[p & 7]), "r"((uint32_t)d[2 * p]), "r"(k));
      asm volatile("mad.lo.u32 %0,%1,%2,%3;" : "=r"(y[p & 7]) : "r"(x), "r"(m[p & 7]), "r"((uint32_t)d[2 * p + 1]));
    }
  }
  DC_DEV void checkpoint() {
    uint32_t x0, x1, x2, x3;
    asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(x0) : "r"(y[0]), "r"(y[1]), "r"(y[2]));
    asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(x1) : "r"(x0), "r"(y[3]), "r"(y[4]));
    asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(x2) : "r"(x1), "r"(y[5]), "r"(y[6]));
    asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(x3) : "r"(x2), "r"(y[7]), "r"(cp));
    asm volatile("redux.sync.xor.b32 %0,%1,0xffffffff;" : "=r"(cp) : "r"(x3));
  }
  DC_DEV uint32_t sink() const { return cp ^ y[0] ^ y[7]; }
};

// wgli_true: the honest LI-CP-W loop with the receipt's data dependence. A committed word at L = 32 is a
// k32 prefix, which is the wgmma's accumulator and readable only after its wait, so each step issues one wgmma, waits,
// LI-folds that step's 128 words and checkpoints. A warpgroup's fold can't overlap its own wgmma, only another
// warpgroup's: _w2 has 2 warpgroups a CTA (wgli_redux's), _w3 has 3. Same instructions a MMA as wgli_redux.
template <class F>
DC_DEV void wg_li_true(const OpArgs& a) {
  SETUP
  Li li(a.mode);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int s = 0; s < 4; s++) {
      wgmma_fence();
      F::mma(d, t, af, 1);
      wgmma_commit();
      wgmma_wait<0>();
      li.fold(d);
      li.checkpoint();
    }
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ li.sink() ^ threadIdx.x)
}

// wgli_pp: wgli_true's loop with its W warpgroups taking turns at the tensor pipe, as CUTLASS's ping-pong kernel orders
// its consumers: warpgroup g waits on named barrier 1 + g, issues its wgmma, arrives on the next warpgroup's barrier
// (EARLY: as soon as it has committed, so the next wgmma queues behind it; else once its own has finished) and folds
// while the others' wgmma run. Each barrier counts 256 threads: one warpgroup's arrive and the next's sync.
template <class F, int W, bool EARLY>
DC_DEV void wg_li_pp(const OpArgs& a) {
  SETUP
  Li li(a.mode);
  const uint32_t g = threadIdx.x / 128, mine = 1 + g, next = 1 + (g + 1) % W;
  if (g == W - 1) asm volatile("bar.arrive 1, 256;" ::: "memory");
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int s = 0; s < 4; s++) {
      asm volatile("bar.sync %0, 256;" ::"r"(mine) : "memory");
      wgmma_fence();
      F::mma(d, t, af, 1);
      wgmma_commit();
      if constexpr (EARLY) asm volatile("bar.arrive %0, 256;" ::"r"(next) : "memory");
      wgmma_wait<0>();
      if constexpr (!EARLY) asm volatile("bar.arrive %0, 256;" ::"r"(next) : "memory");
      li.fold(d);
      li.checkpoint();
    }
  }
  wgmma_hold(d);
  OP_END
  if (g == 0) asm volatile("bar.sync 1, 256;" ::: "memory");
  OP_SINK(fold_acc(d) ^ li.sink() ^ threadIdx.x)
}

// wgli_db: the receipt's loop on two N128 tiles a warpgroup, double-buffered, so each step's fold overlaps the other
// tile's wgmma: issue B's step, wait<1> (A's is done), fold A's 64 words; issue A's next, wait<1>, fold B's. Two N128
// wgmma write one N256's words at its MACs, the same LI and checkpoints a N256-equivalent as wgli_true. _w1 has one
// warpgroup a CTA, _w2 two.
DC_DEV void wg_li_db(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles(sm, a.mode);
  int dA[64] = {}, dB[64] = {};
  Li li(a.mode);
  wgmma_fence();
  wgmma_s8_n128(dA, t.da, t.db, 1);
  wgmma_commit();
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int s = 0; s < 4; s++) {
      wgmma_fence();
      wgmma_s8_n128(dB, t.da, t.db, 1);
      wgmma_commit();
      wgmma_wait<1>();
      wgmma_hold(dA);
      li.fold(dA);
      wgmma_fence();
      wgmma_s8_n128(dA, t.da, t.db, 1);
      wgmma_commit();
      wgmma_wait<1>();
      wgmma_hold(dB);
      li.fold(dB);
      li.checkpoint();
    }
  }
  wgmma_wait<0>();
  wgmma_hold(dA);
  wgmma_hold(dB);
  OP_END
  OP_SINK(fold_acc(dA) ^ fold_acc(dB) ^ li.sink() ^ threadIdx.x)
}

OP_KERNEL(wg_s8_n256_a1, 256) { wg_alone<F_ss>(a); }
OP_KERNEL(wg_s8_n256_rs_a1, 256) { wg_alone<F_rs>(a); }
OP_KERNEL(wg_u8s8_n256_a1, 256) { wg_alone<F_u8>(a); }
OP_KERNEL(wgli_redux, 256) { wg_li<F_ss>(a); }
OP_KERNEL(wgli_redux_rs, 256) { wg_li<F_rs>(a); }
OP_KERNEL(wgli_redux_u8, 256) { wg_li<F_u8>(a); }
OP_KERNEL(wgmix_imad_r256, 256) { wg_mix<F_ss, L_IMAD, 256>(a); }
OP_KERNEL(wgmix_imad_r256_rs, 256) { wg_mix<F_rs, L_IMAD, 256>(a); }
OP_KERNEL(wgmix_lop3_r256, 256) { wg_mix<F_ss, L_LOP3, 256>(a); }
OP_KERNEL(wgmix_lop3_r256_rs, 256) { wg_mix<F_rs, L_LOP3, 256>(a); }
OP_KERNEL(wgli_true_w2, 256) { wg_li_true<F_ss>(a); }
OP_KERNEL(wgli_true_w3, 384) { wg_li_true<F_ss>(a); }
OP_KERNEL(wgli_pp_w2, 256) { wg_li_pp<F_ss, 2, false>(a); }
OP_KERNEL(wgli_pp_w3, 384) { wg_li_pp<F_ss, 3, false>(a); }
OP_KERNEL(wgli_pp2_w2, 256) { wg_li_pp<F_ss, 2, true>(a); }
OP_KERNEL(wgli_pp2_w3, 384) { wg_li_pp<F_ss, 3, true>(a); }
OP_KERNEL(wgli_db_w1, 128) { wg_li_db(a); }
OP_KERNEL(wgli_db_w2, 256) { wg_li_db(a); }
OP_KERNEL(wg_b1_n256_a1, 256) { wg_alone<F_b1>(a); }
OP_KERNEL(wgli_redux_b1, 256) { wg_li<F_b1>(a); }
OP_KERNEL(wgmix_imad_r256_b1, 256) { wg_mix<F_b1, L_IMAD, 256>(a); }
OP_KERNEL(wgmix_lop3_r256_b1, 256) { wg_mix<F_b1, L_LOP3, 256>(a); }

#define S8 "IGMMA.64x256x32.S8.S8*=4"
#define U8 "IGMMA.64x256x32.U8.S8*=4"
#define B1 "BGMMA.64x256x256.AND.POPC*=4"
#define S8H "IGMMA.64x128x32.S8.S8*=8"
#define LI ",LOP3.LUT=272,IMAD=256,REDUX*=4"
#define P(NAME, PAT) {.name = #NAME, .kernel = (const void*)NAME, .tpb = 256, .expect = PAT, .iters = 4096}
#ifndef RS_NO_MAIN
static const OpProbe PROBES[] = {
    P(wg_s8_n256_a1, S8),
    P(wg_s8_n256_rs_a1, S8),
    P(wg_u8s8_n256_a1, U8),
    P(wgli_redux, S8 LI),
    P(wgli_redux_rs, S8 LI),
    P(wgli_redux_u8, U8 LI),
    P(wgmix_imad_r256, S8 ",IMAD=256"),
    P(wgmix_imad_r256_rs, S8 ",IMAD=256"),
    P(wgmix_lop3_r256, S8 ",LOP3.LUT=256"),
    P(wgmix_lop3_r256_rs, S8 ",LOP3.LUT=256"),
    P(wgli_true_w2, S8 LI),
    {.name = "wgli_true_w3", .kernel = (const void*)wgli_true_w3, .tpb = 384, .expect = S8 LI, .iters = 4096},
    P(wgli_pp_w2, S8 LI),
    {.name = "wgli_pp_w3", .kernel = (const void*)wgli_pp_w3, .tpb = 384, .expect = S8 LI, .iters = 4096},
    P(wgli_pp2_w2, S8 LI),
    {.name = "wgli_pp2_w3", .kernel = (const void*)wgli_pp2_w3, .tpb = 384, .expect = S8 LI, .iters = 4096},
    {.name = "wgli_db_w1", .kernel = (const void*)wgli_db_w1, .tpb = 128, .expect = S8H LI, .iters = 4096},
    P(wgli_db_w2, S8H LI),
    P(wg_b1_n256_a1, B1),
    P(wgli_redux_b1, B1 LI),
    P(wgmix_imad_r256_b1, B1 ",IMAD=256"),
    P(wgmix_lop3_r256_b1, B1 ",LOP3.LUT=256"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
#endif
