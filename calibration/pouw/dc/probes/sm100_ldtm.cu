// The issue price of tcgen05.ld.32x32b.x128 (LDTM.x128, 4096 words a warp instruction, the
// widest load) beside uniform.cuh's issue-saturated streams, as sm100_sat.cu prices a uniform instruction. One block a
// SM; each of 8 stream warps (2 a SMSP: a load's 128 destination registers leave no room for more) issues K loads of
// its lane quarter (columns 256-511) an iteration, spread over the stream's IADD3 pairs (LVec). Each load adds its LDTM,
// the wait's NOP and the next address's R2UR and LOP3; ldtm.py charges the slowdown to them.
//   l<F>_<K>   uniform.cuh's stream F (s, v, w or x: the honest step and UVec), 256 threads; K <= 2 (at 4 ptxas spills)
//   n<F>_<K>   UVec's stream F alone (v, w, x) on warps 0-7 of 288 threads, warp 8 idle: m's kernel without its MMA
//   m<F>_<K>   n<F>_<K> beside a saturated tcgen05.mma: warp 8 issues UTCIMMA 128x256x32 into columns 0-255 while the
//              stream runs (OP_UNTIL). No mma.sync in the stream: it shares the tensor pipe
// The table pins each stream loop's counts (ldtm.py tables writes it from the gate's JSON); ldtm.py prices the load.
#include "uniform.cuh"
#include "../dc.cuh"

constexpr int LTPB = 256, LMTPB = 288, LKB = 32, LAROWS = 128, LBROWS = 256;

struct LSmem {
  unsigned char tile[(LAROWS + LBROWS) * LKB];
  uint64_t bar;
  uint32_t tmem;
};

DC_DEV uint32_t lwarp() { return __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0); }

DC_DEV uint32_t lsetup(LSmem& s, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(s.tile);
  for (uint32_t i = threadIdx.x; i < (LAROWS + LBROWS) * LKB / 4; i += blockDim.x)
    w[i] = dc_mix32(i * 0x9E3779B9u + 0x1234567u + (uint32_t)mode);
  if (threadIdx.x == 0) {
    mbar_init(&s.bar, 1);
    fence_mbar_init();
  }
  if (threadIdx.x < 32) tmem_alloc(&s.tmem, 512);
  fence_async_smem();
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  return s.tmem;
}

DC_DEV void lteardown(uint32_t tmem) {
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (threadIdx.x < 32) tmem_dealloc(tmem, 512);
}

// UVec's stream with K loads spread over its IADD3 pairs, S = NI / 2 / K apart: load j issues at pair j S and its
// word j is read S - 1 pairs later, in place of that pair's IADD3 constant (ptxas deletes a load whose registers are
// never read). The next load's address ORs in that word under a mask that is 0 at run time (`zm`, from the mode), one
// LOP3 a load, so one load is in flight a warp and its 128 registers are all a load needs: two would spill.
template <int NF, int NI, int R, int K>
struct LVec : UVec<NF, NI, R> {
  using UVec<NF, NI, R>::UVec;
  __device__ __forceinline__ void step(uint32_t& addr, uint32_t zm, uint32_t (&z)[128]) {
    auto& f = this->f;
    auto& v = this->v;
#pragma unroll
    for (int q = 0; q < NF; q++)
      f[q & 15] = R == 2   ? __fadd_rn(f[q & 15], 9.765625e-04f)
                  : R == 1 ? __fmaf_rn(f[q & 15], -6.103515625e-05f, f[q & 15])
                           : __fmaf_rn(f[q & 15], 0.99993896484375f, this->g);
    constexpr int S = K ? NI / 2 / K : 1;
    static_assert(K == 0 || S >= 1, "more loads than IADD3s");
#pragma unroll
    for (int q = 0; q < NI / 2; q++) {
      const int p = (q & 7) * 2;
      const uint32_t k = 0x85EBCA6Bu + (uint32_t)q * 0xC2B2AE35u;
      uint32_t c = k;
      if (q / S < K && q % S == 0) tmem_ld128(addr, z);
      if (q / S < K && q % S == S - 1) {
        tmem_wait_ld();
        c = z[(q / S) & 127];
        addr |= c & zm;
      }
      v[p] = v[p] + v[p + 1] + c;
      v[p + 1] = v[p + 1] ^ v[p] ^ (k >> 5);
    }
  }
};

template <int K, int NF, int NI, int R, bool H>
DC_DEV void ldtm_sat(const OpArgs& a) {
  __shared__ __align__(1024) LSmem s;
  const uint32_t tmem = lsetup(s, a.mode);
  const uint32_t w = lwarp(), base = tmem + ((32u * (w & 3u)) << 16) + 256u + (w >> 2) * 128u;
  const uint32_t zm = (uint32_t)a.mode & 0x80000000u;
  uint32_t addr = base;
  Honest h(a.mode);
  LVec<NF, NI, R, K> v(a.mode);
  uint32_t z[128];
  OP_BEGIN
  OP_LOOP(it) {
    if constexpr (H) h.step();
    v.step(addr, zm, z);
  }
  OP_END
  const uint32_t x = (H ? h.fold() : 0u) ^ v.fold();
  lteardown(tmem);
  OP_SINK(x)
}

template <int K, int NF, int NI, int R, bool MMA>
DC_DEV void ldtm_mma(const OpArgs& a) {
  __shared__ __align__(1024) LSmem s;
  const uint32_t w = lwarp();
  uint32_t x = 0;
  {
    const uint32_t tmem = lsetup(s, a.mode);
    OP_BEGIN_STOP
    if (w < 8) {
      const uint32_t zm = (uint32_t)a.mode & 0x80000000u;
      uint32_t addr = tmem + ((32u * (w & 3u)) << 16) + 256u + ((w >> 2) & 1u) * 128u;
      LVec<NF, NI, R, K> v(a.mode);
      uint32_t z[128];
      OP_LOOP(it) { v.step(addr, zm, z); }
      OP_DONE
      x = v.fold();
    } else if (MMA) {
      const uint64_t da = umma_desc(smem_u32(s.tile), kmaj_lbo(), kmaj_sbo(LKB));
      const uint64_t db = umma_desc(smem_u32(s.tile) + LAROWS * LKB, kmaj_lbo(), kmaj_sbo(LKB));
      constexpr uint32_t idesc = umma_idesc(2, 1, 1, 128, 256);
      OP_UNTIL(n) {
        if (elect_one()) {
          umma_i8(tmem, da, db, idesc, 0);
          umma_i8(tmem, da, db, idesc, 1);
          umma_i8(tmem, da, db, idesc, 1);
          umma_i8(tmem, da, db, idesc, 1);
        }
        __syncwarp();
      }
      OP_TRIPS(n)
    }
    OP_END_STOP
  }
  if (MMA && w == 8) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
    mbar_wait(&s.bar, 0);
  }
  lteardown(*(volatile uint32_t*)&s.tmem);
  OP_SINK(x ^ threadIdx.x)
}

#define LK(nm, K, NF, NI, R, H) \
  OP_KERNEL(nm, LTPB) { ldtm_sat<K, NF, NI, R, H>(a); }
#define MK(nm, K, NF, NI, R, M) \
  OP_KERNEL(nm, LMTPB) { ldtm_mma<K, NF, NI, R, M>(a); }
#define LSTREAM(F, NF, NI, R) LK(l##F##_0, 0, NF, NI, R, true) LK(l##F##_1, 1, NF, NI, R, true) LK(l##F##_2, 2, NF, NI, R, true)
#define MSTREAM4(P, F, NF, NI, R, M) \
  MK(P##F##_0, 0, NF, NI, R, M) MK(P##F##_1, 1, NF, NI, R, M) MK(P##F##_2, 2, NF, NI, R, M) MK(P##F##_4, 4, NF, NI, R, M)
#define MSTREAM8(P, F, NF, NI, R, M) MSTREAM4(P, F, NF, NI, R, M) MK(P##F##_8, 8, NF, NI, R, M)
#define MSTREAM(P, F, NF, NI, R, M) MSTREAM8(P, F, NF, NI, R, M) MK(P##F##_16, 16, NF, NI, R, M)
LSTREAM(s, 512, 128, 0)
LSTREAM(v, 512, 128, 1)
LSTREAM(w, 1024, 256, 1)
LSTREAM(x, 1024, 64, 2)
MSTREAM(n, v, 512, 128, 1, false)
MSTREAM(n, w, 1024, 256, 1, false)
MSTREAM8(n, x, 1024, 64, 2, false)  // nx_16, mv_16, mx_8, mx_16 spill at 168 registers
MSTREAM8(m, v, 512, 128, 1, true)
MSTREAM(m, w, 1024, 256, 1, true)
MSTREAM4(m, x, 1024, 64, 2, true)

#define P_L(nm, EX, CNT) {.name = #nm, .kernel = (const void*)nm, .tpb = LTPB, .expect = EX, .counted = CNT, .iters = UITERS}
#define P_N(nm, EX, CNT) {.name = #nm, .kernel = (const void*)nm, .tpb = LMTPB, .expect = EX, .counted = CNT, .warps = "8", .iters = UITERS}
#define P_M(nm, EX, CNT) \
  {.name = #nm, .kernel = (const void*)nm, .tpb = LMTPB, .expect = EX ";UTCIMMA*=4", .counted = CNT, .warps = "8;1", .iters = UITERS, .until = 2}

static const OpProbe PROBES[] = {
    P_L(ls_0, "FFMA=512,LOP3.LUT=96,IADD3=64,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,UISETP.NE.AND=1", "IMMA.16832.S8.S8"),
    P_L(ls_1, "FFMA=512,LOP3.LUT=98,IADD3=64,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,R2UR=1,LDTM.x128=1", "IMMA.16832.S8.S8"),
    P_L(ls_2, "FFMA=512,LOP3.LUT=100,IADD3=64,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,R2UR=2,LDTM.x128=2", "IMMA.16832.S8.S8"),
    P_L(lv_0, "FFMA=512,LOP3.LUT=96,IADD3=64,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,UISETP.NE.AND=1", "IMMA.16832.S8.S8"),
    P_L(lv_1, "FFMA=512,LOP3.LUT=98,IADD3=64,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,R2UR=1,LDTM.x128=1", "IMMA.16832.S8.S8"),
    P_L(lv_2, "FFMA=512,LOP3.LUT=100,IADD3=64,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,R2UR=2,LDTM.x128=2", "IMMA.16832.S8.S8"),
    P_L(lw_0, "FFMA=1024,LOP3.LUT=160,IADD3=128,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,UISETP.NE.AND=1", "IMMA.16832.S8.S8"),
    P_L(lw_1, "FFMA=1024,LOP3.LUT=162,IADD3=128,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,R2UR=1,LDTM.x128=1", "IMMA.16832.S8.S8"),
    P_L(lw_2, "FFMA=1024,LOP3.LUT=164,IADD3=128,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,R2UR=2,LDTM.x128=2", "IMMA.16832.S8.S8"),
    P_L(lx_0, "FADD=1024,LOP3.LUT=64,IMAD=32,LEA.HI=32,IADD3=32,IMMA.16832.S8.S8=8,UIADD3=1,UISETP.NE.AND=1", "IMMA.16832.S8.S8"),
    P_L(lx_1, "FADD=1024,LOP3.LUT=66,IMAD=32,LEA.HI=32,IADD3=32,IMMA.16832.S8.S8=8,R2UR=1,LDTM.x128=1", "IMMA.16832.S8.S8"),
    P_L(lx_2, "FADD=1024,LOP3.LUT=68,IADD3=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,R2UR=2,LDTM.x128=2", "IMMA.16832.S8.S8"),
    P_N(nv_0, "FFMA=512,IADD3=64,LOP3.LUT=64,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nv_1, "FFMA=512,LOP3.LUT=66,IADD3=64,R2UR=1,LDTM.x128=1,LDC=1,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nv_2, "FFMA=512,LOP3.LUT=66,IADD3=64,R2UR=2,LDTM.x128=2,NOP=2,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nv_4, "FFMA=512,LOP3.LUT=68,IADD3=64,R2UR=4,LDTM.x128=4,NOP=4,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nv_8, "FFMA=512,LOP3.LUT=72,IADD3=64,R2UR=8,LDTM.x128=8,NOP=8,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nv_16, "FFMA=512,LOP3.LUT=80,IADD3=64,R2UR=16,LDTM.x128=16,NOP=16,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nw_0, "FFMA=1024,IADD3=128,LOP3.LUT=128,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nw_1, "FFMA=1024,LOP3.LUT=130,IADD3=128,R2UR=1,LDTM.x128=1,LDC=1,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nw_2, "FFMA=1024,LOP3.LUT=130,IADD3=128,R2UR=2,LDTM.x128=2,NOP=2,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nw_4, "FFMA=1024,LOP3.LUT=132,IADD3=128,R2UR=4,LDTM.x128=4,NOP=4,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nw_8, "FFMA=1024,LOP3.LUT=136,IADD3=128,R2UR=8,LDTM.x128=8,NOP=8,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nw_16, "FFMA=1024,LOP3.LUT=144,IADD3=128,R2UR=16,LDTM.x128=16,NOP=16,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_N(nx_0, "FADD=1024,IADD3=32,LOP3.LUT=32,UIADD3=1,UISETP.NE.AND=1", "FADD"),
    P_N(nx_1, "FADD=1024,LOP3.LUT=34,IADD3=32,R2UR=1,LDTM.x128=1,LDC=1,UIADD3=1,UISETP.NE.AND=1", "FADD"),
    P_N(nx_2, "FADD=1024,LOP3.LUT=34,IADD3=32,R2UR=2,LDTM.x128=2,NOP=2,UIADD3=1,UISETP.NE.AND=1", "FADD"),
    P_N(nx_4, "FADD=1024,LOP3.LUT=36,IADD3=32,R2UR=4,LDTM.x128=4,NOP=4,UIADD3=1,UISETP.NE.AND=1", "FADD"),
    P_N(nx_8, "FADD=1024,LOP3.LUT=40,IADD3=32,R2UR=8,LDTM.x128=8,NOP=8,UIADD3=1,UISETP.NE.AND=1", "FADD"),
    P_M(mv_0, "FFMA=512,IADD3=64,LOP3.LUT=64,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mv_1, "FFMA=512,LOP3.LUT=66,IADD3=64,R2UR=1,LDTM.x128=1,LDC=1,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mv_2, "FFMA=512,LOP3.LUT=66,IADD3=64,R2UR=2,LDTM.x128=2,NOP=2,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mv_4, "FFMA=512,LOP3.LUT=68,IADD3=64,R2UR=4,LDTM.x128=4,NOP=4,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mv_8, "FFMA=512,LOP3.LUT=72,IADD3=64,R2UR=8,LDTM.x128=8,NOP=8,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mw_0, "FFMA=1024,IADD3=128,LOP3.LUT=128,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mw_1, "FFMA=1024,LOP3.LUT=130,IADD3=128,R2UR=1,LDTM.x128=1,LDC=1,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mw_2, "FFMA=1024,LOP3.LUT=130,IADD3=128,R2UR=2,LDTM.x128=2,NOP=2,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mw_4, "FFMA=1024,LOP3.LUT=132,IADD3=128,R2UR=4,LDTM.x128=4,NOP=4,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mw_8, "FFMA=1024,LOP3.LUT=136,IADD3=128,R2UR=8,LDTM.x128=8,NOP=8,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mw_16, "FFMA=1024,LOP3.LUT=144,IADD3=128,R2UR=16,LDTM.x128=16,NOP=16,UIADD3=1,UISETP.NE.AND=1", "FFMA"),
    P_M(mx_0, "FADD=1024,IADD3=32,LOP3.LUT=32,UIADD3=1,UISETP.NE.AND=1", "FADD"),
    P_M(mx_1, "FADD=1024,LOP3.LUT=34,IADD3=32,R2UR=1,LDTM.x128=1,LDC=1,UIADD3=1,UISETP.NE.AND=1", "FADD"),
    P_M(mx_2, "FADD=1024,LOP3.LUT=34,IADD3=32,R2UR=2,LDTM.x128=2,NOP=2,UIADD3=1,UISETP.NE.AND=1", "FADD"),
    P_M(mx_4, "FADD=1024,LOP3.LUT=36,IADD3=32,R2UR=4,LDTM.x128=4,NOP=4,UIADD3=1,UISETP.NE.AND=1", "FADD"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
