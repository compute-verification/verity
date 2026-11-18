// Combined-route attacks on sm_90a (H200), the cases the single-stream probes leave open. Built by
// ../opcode_probe/probe.py build --arch sm_90a; every probe passes the SASS gate, the poisoned sink and its no-write
// control before it is timed (probe.cuh). Accumulators stay in registers; A and B come from one shared tile through
// no-swizzle K-major descriptors (dc.cuh).
//
// The one facet the census never measured (isa_escape.py): a
// uniform-datapath instruction's issue cost beside a SATURATED asynchronous wgmma. sm90_async.cu's uwgh put
// the uniform chain beside a saturated wgmma and found it nearly free -- but its vector issue sat idle, so "free" there
// is only the scheduler filling empty issue slots, not a second instruction a cycle. A2/A2u say an SMSP issues at most
// one warp instruction a cycle, uniform included: the sum over EVERY class of warp instructions a SMSP-cycle is <= 1.
// The decisive test is the uniform chain beside a saturated wgmma WHILE the vector pipes already carry issue toward 1 a
// cycle: if a uniform instruction is still free there, two instructions issue in one cycle and A2/A2u break; if it costs
// an issue slot (the loop lengthens one cycle a uniform op), A2u holds where it was never measured.
//
//   emp_pack_<S>_r<R>_k<K>  2 warpgroups, each iteration: 4 wgmma s8 m64n256k32 (a commit group, saturating the tensor
//                           pipe as wg_s8_n256), the honest step (8 mma.sync m16n8k32 + its duo fold, the one structure
//                           where ptxas keeps the chain on the uniform datapath, as uwgh), R FFMA on independent vector
//                           accumulators (the issue filler: FADD/FFMA is the vector pipe, distinct from the fold's
//                           IMAD/LOP3/LEA), and a K-instruction warp-uniform chain (uchain.cuh), then waits. The gate
//                           pins UIMAD/ULOP3 (the chain's uniform half), FFMA = R (the filler), IMMA = 8 and IGMMA = 4.
//                           counted is the chain op (K > 0) or the IGMMA (K = 0); analyze sums every class's rate a
//                           SMSP-cycle (total issue) and reads the K-slope at each R. R <= 256 leaves issue under 0.6
//                           (the trip waits on its wgmma); R = 1024 is issue-bound, the regime A2u is about.
//
// A3 says a 2:4 sparse form does at most twice the dense peak on its logical k: the same instruction rate as the dense
// form of half its logical k. wgmma.mma_async.sp was never measured in the tree.
//   wgsp_<form>   2 warpgroups, 4 sparse wgmma m64n256 a commit group, wait_group 1 (as wg_<form>_a1): s8 and e4m3
//                 k64, f16 k32 (logical), A compressed (64 rows x 32 B), B 256 rows x 64 B, metadata 0x44444444
//                 (indices 0 and 1 of every group of 4: a valid 2:4 pattern; the values don't move a peak).
//   wgd_<form>    its dense twin in the same kernel shape and binary (k32 s8/e4m3, k16 f16): the ratio wgsp/wgd is the
//                 sparse speedup at equal instruction count, A3's bound is 1 (twice the MACs on a logical k twice as deep).
//   wgsprs_s8, wgdrs_s8  the s8 pair with A from registers: B alone (16 KB a sparse instruction) is then the
//                 shared-memory traffic, 128 B a SM-clock at the dense instruction rate.
//
// A5 says shared memory serves 128 B an SM clock, loads and stores together. A saturated SS-mode wgmma reads its A and
// B tiles from shared memory (6 KB a wgmma s8 m64n128k32, 96 B a SM-clock at the tensor peak) through no
// shared-memory class instruction, so the sheet's smem facet never sees it.
//   emp_smem      warpgroups 0-1 wgmma s8 m64n128k32 flat out (OP_LOOP), warpgroups 2-3 8 conflict-free LDS.128 a trip
//                 (lm.cu's ring) only while they run (OP_UNTIL): LDS bytes a SM-clock beside the wgmma's own reads.
//   emp_lds_solo  the same LDS warpgroups alone (OP_LOOP), the LDS ceiling at this warp count.
//   emp_reg       wg_until's split with setmaxnreg (the producer/consumer register handoff): the wgmma warpgroups raise
//                 to 232 registers, the lane warpgroup drops to 40 and runs 64 IMAD a trip (emp_reg_d64, _d96: 208 / 64
//                 and 184 / 96); registers are no facet of the sheet, so its rates should be wgU_imad's.
//   emp_pack3_i_r1024_k<K>  emp_pack at 3 warpgroups on the n128 form, so the filler issues nearer 1 a cycle.
#include "../../opcode_probe/probe.cuh"
#include "../../opcode_probe/probes/honest.cuh"
#include "../dc.cuh"
#include "../wgmma.cuh"
#include "lanes.cuh"
#include "uchain.cuh"

constexpr int ATPB = 256, AITERS = 4096, KB = 32, AROWS = 64, BROWS = 256;

DC_DEV uint32_t emp_word(uint32_t i, int mode) {
  return mode == 1 ? 0u : dc_mix32(i * 0x9E3779B9u + 0x1234567u);
}
struct Tiles {
  uint64_t da, db;
};
DC_DEV Tiles emp_tiles(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x) w[i] = emp_word(i, mode);
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB))};
}
// Four wgmma into one accumulator: the first overwrites (scale_d 0), the rest accumulate; saturates the tensor pipe.
DC_DEV void emp_group4(int (&d)[128], const Tiles& t) {
  wgmma_fence();
  wgmma_s8_n256(d, t.da, t.db, 0);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_commit();
}
// The n128 form (4095.4 MAC/SM-clk in G1, as n256) for kernels over 256 threads: n256's 128 accumulators need 154
// registers, over 512 threads' 128, and serialize at 384 threads' 168.
DC_DEV void emp_group4n128(int (&d)[64], const Tiles& t) {
  wgmma_fence();
  wgmma_s8_n128(d, t.da, t.db, 0);
  wgmma_s8_n128(d, t.da, t.db, 1);
  wgmma_s8_n128(d, t.da, t.db, 1);
  wgmma_s8_n128(d, t.da, t.db, 1);
  wgmma_commit();
}
template <int N>
DC_DEV uint32_t emp_fold(const int (&d)[N]) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < N; i++) x ^= (uint32_t)d[i];
  return x;
}

// NW warpgroups: past 2 (more warps a SMSP, which p_ffma_lop3 needs to issue near 1 a cycle) on the n128 form, whose 64
// accumulators fit 384 threads' 168 registers.
template <int S, int R, int K, int NW = 2>
DC_DEV void emp_pack(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = emp_tiles(sm, a.mode);
  int d[NW == 2 ? 128 : 64] = {};
  Honest h(a.mode);  // 8 mma.sync + duo fold: forces ptxas to keep the uniform chain on the uniform datapath
  Lane lv((uint32_t)threadIdx.x);
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    if constexpr (NW > 2)
      emp_group4n128(d, t);
    else
      emp_group4(d, t);
    h.step<8, true>();
    lane_ops<L_FFMA, R>(lv);
    u.step();
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(emp_fold(d) ^ h.fold() ^ lv.fold() ^ u.fold() ^ threadIdx.x)
}

#define PACK(S, SN, R, K) \
  OP_KERNEL(emp_pack_##SN##_r##R##_k##K, ATPB) { emp_pack<S, R, K>(a); }
PACK(S_I, i, 0, 0)
PACK(S_I, i, 0, 64)
PACK(S_I, i, 0, 128)
PACK(S_I, i, 128, 0)
PACK(S_I, i, 128, 64)
PACK(S_I, i, 128, 128)
PACK(S_I, i, 256, 0)
PACK(S_I, i, 256, 64)
PACK(S_I, i, 256, 128)
PACK(S_I, i, 1024, 0)
PACK(S_I, i, 1024, 64)
PACK(S_I, i, 1024, 128)
#undef PACK
#define PACK3(S, SN, R, K) \
  OP_KERNEL(emp_pack3_##SN##_r##R##_k##K, 384) { emp_pack<S, R, K, 3>(a); }
PACK3(S_I, i, 1024, 0)
PACK3(S_I, i, 1024, 64)
PACK3(S_I, i, 1024, 128)
#undef PACK3

// ---- sparse wgmma beside its dense twin ---------------------------------------------------------------------------
constexpr int SKB = 64;  // B's K bytes a row: k64 s8/e4m3, k32 f16 (A compressed to 32 B a row, as the dense k32 tile)

// Operand bytes: s8 every code (TY 0), e4m3 no NaN or inf code (1), f16 in ±[0.5, 1) (4); zeros in mode 1.
template <int TY>
DC_DEV uint32_t sp_word(uint32_t i, int mode) {
  if (mode == 1) return 0u;
  const uint32_t x = dc_mix32(i * 0x9E3779B9u + 0x1234567u);
  if constexpr (TY == 1) return x & 0x77777777u;
  if constexpr (TY == 4) return (x & 0x83FF83FFu) | 0x38003800u;
  return x;
}
// A (64 rows x 32 B) then B (256 rows x BK bytes), K-major: the sparse form's B is twice the dense form's.
template <int TY, int BK>
DC_DEV Tiles sp_tiles(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS * KB + BROWS * BK) / 4; i += blockDim.x) w[i] = sp_word<TY>(i, mode);
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(BK))};
}
// A's fragment from registers (sm90_wgmma.cu's F_s8_n256_rs): 4 words a thread, the dense k32 and the sparse k64 alike.
DC_DEV void rs_frag(uint32_t (&a)[4], const Tiles& t) {
  a[0] = (uint32_t)t.da ^ threadIdx.x;
  a[1] = (uint32_t)(t.da >> 32);
  a[2] = (uint32_t)t.db;
  a[3] = 0x01020304u;
}
#define SPF(NAME, ACC, TY, BK, CALL)                                                       \
  struct G_##NAME {                                                                       \
    using Acc = ACC;                                                                      \
    static constexpr int TYPE = TY, B_K = BK;                                             \
    DC_DEV static void mma(Acc (&d)[128], const Tiles& t, uint32_t e, int sd) { CALL; }   \
  };
SPF(s8_sp, int, 0, SKB, wgmma_s8_n256_sp(d, t.da, t.db, e, sd))
SPF(e4m3_sp, float, 1, SKB, wgmma_e4m3_n256_sp(d, t.da, t.db, e, sd))
SPF(f16_sp, float, 4, SKB, wgmma_f16_n256_sp(d, t.da, t.db, e, sd))
SPF(s8_d, int, 0, KB, wgmma_s8_n256(d, t.da, t.db, sd))
SPF(e4m3_d, float, 1, KB, wgmma_e4m3_n256(d, t.da, t.db, sd))
SPF(f16_d, float, 4, KB, wgmma_f16_n256(d, t.da, t.db, sd))
SPF(s8_rs_sp, int, 0, SKB, uint32_t ra[4]; rs_frag(ra, t); wgmma_s8_n256_rs_sp(d, ra, t.db, e, sd))
SPF(s8_rs_d, int, 0, KB, uint32_t ra[4]; rs_frag(ra, t); wgmma_s8_n256_rs(d, ra, t.db, sd))
#undef SPF
template <class G>
DC_DEV void wg_sp(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[AROWS * KB + BROWS * SKB];
  const Tiles t = sp_tiles<G::TYPE, G::B_K>(sm, a.mode);
  typename G::Acc d[128] = {};
  const uint32_t e = 0x44444444u;
  OP_BEGIN
  OP_LOOP(it) {
    wgmma_fence();
    G::mma(d, t, e, 0);
    G::mma(d, t, e, 1);
    G::mma(d, t, e, 1);
    G::mma(d, t, e, 1);
    wgmma_commit();
    wgmma_wait<1>();
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  OP_END
  uint32_t x = threadIdx.x;
#pragma unroll
  for (int i = 0; i < 128; i++) x ^= *reinterpret_cast<const uint32_t*>(&d[i]);
  OP_SINK(x)
}
OP_KERNEL(wgsp_s8, ATPB) { wg_sp<G_s8_sp>(a); }
OP_KERNEL(wgsp_e4m3, ATPB) { wg_sp<G_e4m3_sp>(a); }
OP_KERNEL(wgsp_f16, ATPB) { wg_sp<G_f16_sp>(a); }
OP_KERNEL(wgd_s8, ATPB) { wg_sp<G_s8_d>(a); }
OP_KERNEL(wgd_e4m3, ATPB) { wg_sp<G_e4m3_d>(a); }
OP_KERNEL(wgd_f16, ATPB) { wg_sp<G_f16_d>(a); }
OP_KERNEL(wgsprs_s8, ATPB) { wg_sp<G_s8_rs_sp>(a); }
OP_KERNEL(wgdrs_s8, ATPB) { wg_sp<G_s8_rs_d>(a); }

// ---- shared memory beside the wgmma's own shared-memory reads, and setmaxnreg ---------------------------------------
constexpr int LU = 8, RING = 2048;
__host__ __device__ inline unsigned ring_off(int slot, int lane, int warp, int it) {
  return ((((unsigned)lane + 8u * LU * (unsigned)(warp + it)) & (RING - 1)) + 8u * (unsigned)slot) * 16u;
}
DC_DEV uint32_t lds8(unsigned rbase, int lane, int w, int it) {
  uint32_t acc = 0;
#pragma unroll
  for (int u = 0; u < LU; u++) {
    unsigned r0, r1, r2, r3;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(rbase + ring_off(u, lane, w, it)));
    acc ^= r0 ^ r1 ^ r2 ^ r3;
  }
  return acc;
}
constexpr int RING_BYTES = (RING + 8 * LU) * 16;
template <bool WG>
DC_DEV void emp_smem_k(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB + RING_BYTES];
  const Tiles t = emp_tiles(sm, a.mode);
  unsigned char* ring = sm + (AROWS + BROWS) * KB;
  for (int i = threadIdx.x; i < RING_BYTES / 4; i += blockDim.x)
    reinterpret_cast<uint32_t*>(ring)[i] = (uint32_t)i * 2654435761u + blockIdx.x + 1;
  __syncthreads();
  const unsigned rbase = smem_u32(ring);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const bool mma_role = WG && threadIdx.x < 256;
  int d[64] = {};
  uint32_t acc = threadIdx.x + 1;
  if constexpr (WG) {
    OP_BEGIN_STOP
    if (mma_role) {
      OP_LOOP(it) {
        emp_group4n128(d, t);
        wgmma_wait<1>();
      }
      wgmma_wait<0>();
      OP_DONE
      wgmma_hold(d);
    } else {
      OP_UNTIL(n) { acc ^= lds8(rbase, lane, w, n); }
      OP_TRIPS(n)
    }
    OP_END_STOP
  } else {
    OP_BEGIN
    OP_LOOP(it) { acc ^= lds8(rbase, lane, w, it); }
    OP_END
  }
  OP_SINK((mma_role ? emp_fold(d) : acc) ^ threadIdx.x)
}
OP_KERNEL(emp_smem, 512) { emp_smem_k<true>(a); }
OP_KERNEL(emp_lds_solo, 256) { emp_smem_k<false>(a); }

template <int INC, int DEC>
DC_DEV void emp_reg_k(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = emp_tiles(sm, a.mode);
  const bool mma_role = threadIdx.x < 256;
  uint32_t out;
  // setmaxnreg.inc is a retry loop (USETMAXREG.TRY_ALLOC) until the lane role's registers are released, so both run
  // before the clock starts. Each role folds its own state inside its branch: the lane role at 40 registers must not
  // hold the accumulators live.
  if (mma_role)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(INC) : "memory");
  else
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(DEC) : "memory");
  OP_BEGIN_STOP
  if (mma_role) {
    int d[64] = {};
    OP_LOOP(it) {
      emp_group4n128(d, t);
      wgmma_wait<1>();
    }
    wgmma_wait<0>();
    OP_DONE
    wgmma_hold(d);
    out = emp_fold(d);
  } else {
    Lane l((uint32_t)threadIdx.x);
    OP_UNTIL(n) { lane_ops<L_IMAD, 64>(l); }
    OP_TRIPS(n)
    out = l.fold();
  }
  OP_END_STOP
  OP_SINK(out ^ threadIdx.x)
}
// ptxas drops a setmaxnreg it can't honour (C7507) and the SASS then has no USETMAXREG, so three releases: 2 x 128 x INC
// + 128 x DEC <= 65536.
OP_KERNEL(emp_reg, 384) { emp_reg_k<232, 40>(a); }
OP_KERNEL(emp_reg_d64, 384) { emp_reg_k<208, 64>(a); }
OP_KERNEL(emp_reg_d96, 384) { emp_reg_k<184, 96>(a); }

#define WGI "IGMMA.64x256x32.S8.S8=4,IMMA.16832.S8.S8=8,LEA.HI=32"
#define HON WGI ",IMAD=32,LOP3.LUT=32"
// ptxas 13.0 keeps half of emp_pack's chain uniform (uwgh keeps all of it): K/4 UIMAD and ULOP3, and K/4 each on the
// lanes beside the honest step's 32 IMAD and 32 LOP3.
#define K64 WGI ",IMAD=48,LOP3.LUT=48,UIMAD=16,ULOP3.LUT=16"
#define K128 WGI ",IMAD=64,LOP3.LUT=64,UIMAD=32,ULOP3.LUT=32"
// counted: the chain's UIMAD (K > 0) or the IGMMA (K = 0). FFMA = R is the filler.
#define P(SN, R, K, EX, CNT) \
  {.name = "emp_pack_" #SN "_r" #R "_k" #K, .kernel = (const void*)emp_pack_##SN##_r##R##_k##K, .tpb = ATPB, \
   .expect = EX, .counted = CNT, .iters = AITERS}
static const OpProbe PROBES[] = {
    P(i, 0, 0, HON, "IGMMA.64x256x32.S8.S8"),
    P(i, 0, 64, K64, "UIMAD"),
    P(i, 0, 128, K128, "UIMAD"),
    P(i, 128, 0, HON ",FFMA=128", "IGMMA.64x256x32.S8.S8"),
    P(i, 128, 64, K64 ",FFMA=128", "UIMAD"),
    P(i, 128, 128, K128 ",FFMA=128", "UIMAD"),
    P(i, 256, 0, HON ",FFMA=256", "IGMMA.64x256x32.S8.S8"),
    P(i, 256, 64, K64 ",FFMA=256", "UIMAD"),
    P(i, 256, 128, K128 ",FFMA=256", "UIMAD"),
    P(i, 1024, 0, HON ",FFMA=1024", "IGMMA.64x256x32.S8.S8"),
    P(i, 1024, 64, K64 ",FFMA=1024", "UIMAD"),
    P(i, 1024, 128, K128 ",FFMA=1024", "UIMAD"),
// How ptxas splits the chain between the uniform datapath and the lanes moves with occupancy, so these pin only the
// wgmma, the honest MMAs and the filler; emp.py prices every opcode of the loop from its SASS.
#define P3(R, K) \
  {.name = "emp_pack3_i_r" #R "_k" #K, .kernel = (const void*)emp_pack3_i_r##R##_k##K, .tpb = 384, \
   .expect = "IGMMA.64x128x32.S8.S8=4,IMMA.16832.S8.S8=8,FFMA=" #R, .counted = "IGMMA.64x128x32.S8.S8", .iters = AITERS}
    P3(1024, 0), P3(1024, 64), P3(1024, 128),
#undef P3
#define P_SP(NM) {.name = #NM, .kernel = (const void*)NM, .tpb = ATPB, .expect = "*GMMA*=4", .iters = AITERS}
    P_SP(wgsp_s8), P_SP(wgsp_e4m3), P_SP(wgsp_f16), P_SP(wgd_s8), P_SP(wgd_e4m3), P_SP(wgd_f16), P_SP(wgsprs_s8),
    P_SP(wgdrs_s8),
#undef P_SP
    // No bank check here (the OP_UNTIL loop's stop poll is one more LDS): its ring is emp_lds_solo's, which has one.
    {.name = "emp_smem", .kernel = (const void*)emp_smem, .tpb = 512, .expect = "IGMMA.64x128x32.S8.S8=4;LDS.128=8",
     .warps = "8;8", .iters = AITERS, .until = 2},
    {.name = "emp_lds_solo", .kernel = (const void*)emp_lds_solo, .tpb = 256, .expect = "LDS.128=8",
     .smem_ops = "LDS.128*8", .smem = ring_off, .iters = AITERS},
    {.name = "emp_reg", .kernel = (const void*)emp_reg, .tpb = 384, .expect = "IGMMA.64x128x32.S8.S8=4;IMAD=64",
     .warps = "8;4", .iters = AITERS, .until = 2},
    {.name = "emp_reg_d64", .kernel = (const void*)emp_reg_d64, .tpb = 384, .expect = "IGMMA.64x128x32.S8.S8=4;IMAD=64",
     .warps = "8;4", .iters = AITERS, .until = 2},
    {.name = "emp_reg_d96", .kernel = (const void*)emp_reg_d96, .tpb = 384, .expect = "IGMMA.64x128x32.S8.S8=4;IMAD=64",
     .warps = "8;4", .iters = AITERS, .until = 2},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
