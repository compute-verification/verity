// N-W1's lane (sm90_w1.cu): the register writers CORE-H32's run of record left unprobed, on h32.cuh's pattern. 8
// independent chains of each kind a thread; an op writes its chain from itself and a neighbour, so ptxas can neither
// fold nor hoist it. A form whose output is a predicate chains it into the next op's predicate input, so every
// predicate written is read.
#pragma once
#include <utility>
#include "../dc.cuh"

enum : int {
  // int32 lane forms outside CORE-H32's table
  W_BREV, W_BMSK, W_SGXT, W_VABSDIFF4, W_VIADDMAX, W_VIADDMIN, W_VIMAX3, W_IABS, W_I2IP, W_IMADW8, W_IMADWS, W_MULWIDE,
  // predicate writers: LOP3 with a predicate output, ISETP with two, PLOP3, predicates to and from a register
  W_LOP3P, W_ISETP2, W_PLOP3, W_R2P, W_P2R,
  // special registers
  W_CS2R, W_CLK32, W_LANEID,
  // fp64 conversions
  W_I2F64, W_F2I64,
  // other: half conversions, warp-wide
  W_HADD2F32, W_F2FP_H, W_MOVM, W_REDUX, W_MATCH,
  // mix partners on the r and d chains (independent of rr, which the IMAD.WIDE forms use)
  W_LOP3, W_IADD3, W_IMAD, W_FADD, W_DADD, W_HADD2,
  // reading a predicate: IMAD guarded by pm != 0 (loop-invariant), SEL on it; a branch on bit i of the trip's coin
  // word cw (warp-uniform), and on bit i of a lane's own chain word (divergent), each arm 4 ops on 4 chains (16 ops
  // for the *16 forms)
  W_IMADP, W_SEL, W_UBRA, W_DBRA, W_UBRA16, W_DBRA16,
  // the census's unprobed writers: shift-and-add (LEA), a 64-bit add (IADD3 and its carry-in IADD3.X), shuffles at a
  // lane index and up; an indexed branch on 2 bits of the trip's coin word to one of 4 arms of 16 ops (BRX)
  W_LEA, W_IADD64, W_SHFLI, W_SHFLU, W_BRX4,
  // the same on 3 and 4 bits to one of 8 and 16 arms of 16 ops, each arm's LOP3 table and IMAD multiplier its own
  W_BRX8, W_BRX16,
  // FRND.F64 after a DADD on its chain (rounding an integral value again would fold)
  W_FRND64,
};

struct WLane {
  uint32_t r[8];
  double d[8];
  uint64_t q[8];
  uint32_t rr[16];
  uint32_t pm;  // the predicate chain's carrier: its bits are the predicates a form's next op reads
  uint32_t cw = 0;  // the trip's coin word (b_run), the same in every lane
  DC_DEV explicit WLane(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
      d[i] = 1.0 + (double)(r[i] & 0xffffu) * 0x1p-20;
      q[i] = ((uint64_t)r[i] << 32) ^ (r[i] * 0x27d4eb2fu);
      rr[2 * i] = r[i] ^ 0x5bd1e995u;
      rr[2 * i + 1] = r[i] * 3u;
    }
    pm = tid;
  }
  template <int X>
  DC_DEV void op(int i) {
    const int s = i & 7, t = (i + 1) & 7, u = (i + 2) & 7;
    const bool odd = (i >> 3) & 1;
    if constexpr (X == W_BREV) asm volatile("{\n.reg .b32 q;\nbrev.b32 q,%0;\nxor.b32 %0,q,%1;\n}" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == W_BMSK) asm volatile("bmsk.clamp.b32 %0,%0,%1;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == W_SGXT) asm volatile("szext.clamp.s32 %0,%0,%1;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == W_VABSDIFF4)
      asm volatile("vabsdiff4.u32.u32.u32.add %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(r[u]));
    if constexpr (X == W_VIADDMAX) r[s] = (uint32_t)__viaddmax_s32((int)r[s], (int)r[t], (int)r[u]);
    if constexpr (X == W_VIADDMIN) r[s] = (uint32_t)__viaddmin_s32((int)r[s], (int)r[t], (int)r[u]);
    if constexpr (X == W_VIMAX3) r[s] = (uint32_t)__vimax3_s32((int)r[s], (int)r[t], (int)r[u]);
    if constexpr (X == W_IABS) {
      if (odd) asm volatile("abs.s32 %0,%0;" : "+r"(r[s]));
      else asm volatile("sub.s32 %0,%1,%0;" : "+r"(r[s]) : "r"(r[t]));
    }
    if constexpr (X == W_I2IP) asm volatile("cvt.pack.sat.s8.s32.b32 %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(r[u]));
    // mul.wide on 8 independent pairs of 16 registers (IMAD.WIDE with an RZ addend: two sources); W_MULWIDE has 4.
    // mad.wide with a 64-bit addend lowers to IMAD.WIDE + IADD3 + IADD3.X, as sm90_h32.cu's x_imadw does.
    if constexpr (X == W_IMADW8) {
      const int p = 2 * s;
      asm volatile("{\n.reg .b64 w;\nmul.wide.u32 w,%0,%1;\nmov.b64 {%0,%1},w;\n}" : "+r"(rr[p]), "+r"(rr[p + 1]));
    }
    if constexpr (X == W_IMADWS) {
      const int p = 2 * s;
      asm volatile("{\n.reg .b64 w;\nmul.wide.s32 w,%0,%1;\nmov.b64 {%0,%1},w;\n}" : "+r"(rr[p]), "+r"(rr[p + 1]));
    }
    if constexpr (X == W_LOP3) asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(r[s]) : "r"(r[t]), "r"(0x85EBCA6Bu));
    if constexpr (X == W_IADD3)
      asm volatile("{\n.reg .u32 q;\nadd.u32 q,%1,%2;\nadd.u32 %0,%0,q;\n}" : "+r"(r[s]) : "r"(r[t]), "r"(r[u]));
    if constexpr (X == W_IMAD) asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(0x9E3779B9u));
    if constexpr (X == W_FADD) asm volatile("add.rn.f32 %0,%0,0f3F800000;" : "+r"(r[s]));
    if constexpr (X == W_DADD) asm volatile("add.rn.f64 %0,%0,0d3FF0000000000000;" : "+d"(d[s]));
    if constexpr (X == W_HADD2) asm volatile("add.rn.f16x2 %0,%0,%1;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == W_MULWIDE) {
      const int p = 2 * (i & 3);
      asm volatile("{\n.reg .b64 w;\nmul.wide.u32 w,%0,%1;\nmov.b64 {%0,%1},w;\n}" : "+r"(r[p]), "+r"(r[p + 1]));
    }
    // LOP3 writing a register and a predicate (PTX 8.2's d|p form): p = (d != 0) OR q, q the last op's predicate
    if constexpr (X == W_LOP3P)
      asm volatile("{\n.reg .pred p,q;\nsetp.ne.u32 q,%1,0;\nlop3.or.b32 %0|p,%0,%2,%3,0x96,q;\nselp.u32 %1,1,0,p;\n}"
                   : "+r"(r[s]), "+r"(pm)
                   : "r"(r[t]), "r"(r[u]));
    // ISETP writing two predicates (p|q), the second chained into the next compare
    if constexpr (X == W_ISETP2)
      asm volatile("{\n.reg .pred p,q,c;\nsetp.ne.u32 c,%1,0;\nsetp.lt.and.u32 p|q,%0,%2,c;\nselp.u32 %0,%0,%2,p;\n"
                   "selp.u32 %1,1,0,q;\n}"
                   : "+r"(r[s]), "+r"(pm)
                   : "r"(r[t]));
    if constexpr (X == W_PLOP3)
      asm volatile("{\n.reg .pred a,b,c,e;\nsetp.ne.u32 a,%1,0;\nsetp.lt.u32 b,%0,%2;\nsetp.lt.u32 c,%2,%3;\n"
                   "xor.pred e,a,b;\nand.pred e,e,c;\nselp.u32 %1,1,0,e;\n}"
                   : "+r"(r[s]), "+r"(pm)
                   : "r"(r[t]), "r"(r[u]));
    if constexpr (X == W_R2P || X == W_P2R) {
      // 7 predicates from a register's bits, combined, and packed back: R2P and P2R if ptxas picks them
      asm volatile("{\n.reg .pred p<7>;\n.reg .b32 b<7>;\n"
                   "and.b32 b0,%0,1;\nsetp.ne.u32 p0,b0,0;\nand.b32 b1,%0,2;\nsetp.ne.u32 p1,b1,0;\n"
                   "and.b32 b2,%0,4;\nsetp.ne.u32 p2,b2,0;\nand.b32 b3,%0,8;\nsetp.ne.u32 p3,b3,0;\n"
                   "and.b32 b4,%0,16;\nsetp.ne.u32 p4,b4,0;\nand.b32 b5,%0,32;\nsetp.ne.u32 p5,b5,0;\n"
                   "and.b32 b6,%0,64;\nsetp.ne.u32 p6,b6,0;\n"
                   "selp.u32 b0,1,0,p6;\nselp.u32 b1,2,0,p0;\nselp.u32 b2,4,0,p1;\nselp.u32 b3,8,0,p2;\n"
                   "selp.u32 b4,16,0,p3;\nselp.u32 b5,32,0,p4;\nselp.u32 b6,64,0,p5;\n"
                   "or.b32 b0,b0,b1;\nor.b32 b2,b2,b3;\nor.b32 b4,b4,b5;\nor.b32 b0,b0,b2;\nor.b32 b4,b4,b6;\n"
                   "or.b32 b0,b0,b4;\nadd.u32 %0,b0,%1;\n}"
                   : "+r"(r[s])
                   : "r"(r[t]));
    }
    if constexpr (X == W_CS2R) {
      uint64_t c;
      asm volatile("mov.u64 %0,%%clock64;" : "=l"(c));
      asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(r[s]) : "r"((uint32_t)c), "r"((uint32_t)(c >> 32)));
    }
    if constexpr (X == W_CLK32) {
      uint32_t c;
      asm volatile("mov.u32 %0,%%clock;" : "=r"(c));
      r[s] ^= c;
    }
    if constexpr (X == W_LANEID) {
      uint32_t c;
      asm volatile("mov.u32 %0,%%globaltimer_lo;" : "=r"(c));
      r[s] += c;
    }
    if constexpr (X == W_I2F64) {
      if (odd) asm volatile("cvt.rn.f64.s32 %0,%1;" : "=d"(d[s]) : "r"(r[s]));
      else asm volatile("cvt.rzi.s32.f64 %0,%1;" : "=r"(r[s]) : "d"(d[s]));
    }
    if constexpr (X == W_F2I64) {
      if (odd) asm volatile("{\n.reg .s64 q;\ncvt.rzi.s64.f64 q,%1;\nmov.b64 {%0,%2},q;\n}" : "=r"(r[s]), "+d"(d[s]), "=r"(r[t]));
      else asm volatile("{\n.reg .s64 q;\nmov.b64 q,{%1,%2};\ncvt.rn.f64.s64 %0,q;\n}" : "=d"(d[s]) : "r"(r[s]), "r"(r[t]));
    }
    if constexpr (X == W_HADD2F32) {
      if (odd) asm volatile("{\n.reg .f32 f;\n.reg .b16 h;\nmov.b32 {h,_},%0;\ncvt.f32.f16 f,h;\nmov.b32 %0,f;\n}" : "+r"(r[s]));
      else asm volatile("{\n.reg .b16 h;\n.reg .f32 f;\nmov.b32 f,%0;\ncvt.rn.f16.f32 h,f;\nmov.b32 %0,{h,h};\n}" : "+r"(r[s]));
    }
    if constexpr (X == W_F2FP_H) {
      if (odd) asm volatile("{\n.reg .f32 a,b;\nmov.b32 a,%0;\nmov.b32 b,%1;\ncvt.rn.f16x2.f32 %0,a,b;\n}" : "+r"(r[s]) : "r"(r[t]));
      else asm volatile("{\n.reg .f32 f;\n.reg .b16 h,l;\nmov.b32 {l,h},%0;\ncvt.f32.f16 f,l;\nmov.b32 %0,f;\n}" : "+r"(r[s]));
    }
    if constexpr (X == W_MOVM) asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0,%0;" : "+r"(r[s]));
    if constexpr (X == W_REDUX) {
      uint32_t c;
      asm volatile("redux.sync.add.u32 %0,%1,0xffffffff;" : "=r"(c) : "r"(r[s]));
      r[s] = c ^ r[t];
    }
    if constexpr (X == W_MATCH) asm volatile("match.any.sync.b32 %0,%0,0xffffffff;" : "+r"(r[s]));
    if constexpr (X == W_IMADP)
      asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p,%1,0;\n@p mad.lo.u32 %0,%0,%2,%3;\n}"
                   : "+r"(r[s])
                   : "r"(pm), "r"(r[t]), "r"(0x9E3779B9u));
    if constexpr (X == W_SEL)
      asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p,%1,0;\nselp.b32 %0,%2,%0,p;\n}" : "+r"(r[s]) : "r"(pm), "r"(r[t]));
    if constexpr (X == W_UBRA || X == W_DBRA || X == W_UBRA16 || X == W_DBRA16 || X == W_BRX4 || X == W_BRX8 ||
                  X == W_BRX16) {
#define W1_M4 "mad.lo.u32 %0,%0,%6,%7;\nmad.lo.u32 %1,%1,%6,%7;\nmad.lo.u32 %2,%2,%6,%7;\nmad.lo.u32 %3,%3,%6,%7;\n"
#define W1_L4 \
  "lop3.b32 %0,%0,%6,%7,0x96;\nlop3.b32 %1,%1,%6,%7,0x96;\nlop3.b32 %2,%2,%6,%7,0x96;\nlop3.b32 %3,%3,%6,%7,0x96;\n"
#define W1_ARMS(BRA, M, L)                                                                           \
  "{\n.reg .pred p;\n.reg .b32 b;\nand.b32 b,%4,%5;\nsetp.ne.u32 p,b,0;\n@p " BRA " W1T;\n" M BRA \
  " W1E;\nW1T:\n" L "W1E:\n}"
#define W1_OUT : "+r"(r[s]), "+r"(r[(i + 2) & 7]), "+r"(r[(i + 4) & 7]), "+r"(r[(i + 6) & 7])
      const uint32_t m = 1u << (i & 31);
      if constexpr (X == W_UBRA) asm volatile(W1_ARMS("bra.uni", W1_M4, W1_L4) W1_OUT : "r"(cw), "r"(m), "r"(r[t]), "r"(0x9E3779B9u));
      if constexpr (X == W_DBRA) asm volatile(W1_ARMS("bra", W1_M4, W1_L4) W1_OUT : "r"(r[t]), "r"(m), "r"(r[t]), "r"(0x9E3779B9u));
      if constexpr (X == W_UBRA16)
        asm volatile(W1_ARMS("bra.uni", W1_M4 W1_M4 W1_M4 W1_M4, W1_L4 W1_L4 W1_L4 W1_L4) W1_OUT
                     : "r"(cw), "r"(m), "r"(r[t]), "r"(0x9E3779B9u));
      if constexpr (X == W_DBRA16)
        asm volatile(W1_ARMS("bra", W1_M4 W1_M4 W1_M4 W1_M4, W1_L4 W1_L4 W1_L4 W1_L4) W1_OUT
                     : "r"(r[t]), "r"(m), "r"(r[t]), "r"(0x9E3779B9u));
      if constexpr (X == W_BRX4)
        asm volatile("{\n.reg .u32 k;\nshr.b32 k,%4,%5;\nand.b32 k,k,3;\nW1X: .branchtargets W1A,W1B,W1C,W1D;\nbrx.idx.uni k,W1X;\n"
                     "W1A:\n" W1_M4 W1_M4 W1_M4 W1_M4 "bra.uni W1Z;\n"
                     "W1B:\n" W1_L4 W1_L4 W1_L4 W1_L4 "bra.uni W1Z;\n"
                     "W1C:\nmad.lo.u32 %0,%0,%7,%6;\nmad.lo.u32 %1,%1,%7,%6;\nmad.lo.u32 %2,%2,%7,%6;\n"
                     "mad.lo.u32 %3,%3,%7,%6;\n" W1_M4 W1_M4 W1_M4 "bra.uni W1Z;\n"
                     "W1D:\nlop3.b32 %0,%0,%7,%6,0x69;\nlop3.b32 %1,%1,%7,%6,0x69;\nlop3.b32 %2,%2,%7,%6,0x69;\n"
                     "lop3.b32 %3,%3,%7,%6,0x69;\n" W1_L4 W1_L4 W1_L4 "W1Z:\n}" W1_OUT
                     : "r"(cw), "r"(2u * (uint32_t)(i & 15)), "r"(r[t]), "r"(0x9E3779B9u));
#define W1_LX(L) \
  "lop3.b32 %0,%0,%6,%7," L ";\nlop3.b32 %1,%1,%6,%7," L ";\nlop3.b32 %2,%2,%6,%7," L ";\nlop3.b32 %3,%3,%6,%7," L ";\n"
#define W1_MX(M) "mad.lo.u32 %0,%0," M ",%6;\nmad.lo.u32 %1,%1," M ",%6;\nmad.lo.u32 %2,%2," M ",%6;\nmad.lo.u32 %3,%3," M ",%6;\n"
#define W1_ARM(N, L, M) "W1" N ":\n" W1_LX(L) W1_MX(M) W1_LX(L) W1_MX(M)
#define W1_A8                                                                                                         \
  W1_ARM("0", "0x96", "0x9E3779B1") "bra.uni W1Z;\n" W1_ARM("1", "0x69", "0x85EBCA6B") "bra.uni W1Z;\n"               \
  W1_ARM("2", "0xe8", "0xC2B2AE35") "bra.uni W1Z;\n" W1_ARM("3", "0x17", "0x27D4EB2F") "bra.uni W1Z;\n"               \
  W1_ARM("4", "0xca", "0x165667B1") "bra.uni W1Z;\n" W1_ARM("5", "0x35", "0xD3A2646D") "bra.uni W1Z;\n"               \
  W1_ARM("6", "0x1e", "0xFD7046C5") "bra.uni W1Z;\n" W1_ARM("7", "0xe1", "0xB55A4F09")
      if constexpr (X == W_BRX8)
        asm volatile("{\n.reg .u32 k;\nshr.b32 k,%4,%5;\nand.b32 k,k,7;\n"
                     "W1X: .branchtargets W10,W11,W12,W13,W14,W15,W16,W17;\nbrx.idx.uni k,W1X;\n" W1_A8 "W1Z:\n}" W1_OUT
                     : "r"(cw), "r"(3u * (uint32_t)(i & 7)), "r"(r[t]), "r"(0x9E3779B9u));
      if constexpr (X == W_BRX16)
        asm volatile("{\n.reg .u32 k;\nshr.b32 k,%4,%5;\nand.b32 k,k,15;\n"
                     "W1X: .branchtargets W10,W11,W12,W13,W14,W15,W16,W17,W18,W19,W1a,W1b,W1c,W1d,W1e,W1f;\n"
                     "brx.idx.uni k,W1X;\n" W1_A8 "bra.uni W1Z;\n"
                     W1_ARM("8", "0x2d", "0x7FEB352D") "bra.uni W1Z;\n" W1_ARM("9", "0xd2", "0x846CA68B") "bra.uni W1Z;\n"
                     W1_ARM("a", "0x4b", "0x9E485565") "bra.uni W1Z;\n" W1_ARM("b", "0xb4", "0xEF1D6B47") "bra.uni W1Z;\n"
                     W1_ARM("c", "0x78", "0x2C1B3C6D") "bra.uni W1Z;\n" W1_ARM("d", "0x87", "0x297A2D39") "bra.uni W1Z;\n"
                     W1_ARM("e", "0x5a", "0xA136AAAD") "bra.uni W1Z;\n" W1_ARM("f", "0xa5", "0x9E3779BB") "W1Z:\n}" W1_OUT
                     : "r"(cw), "r"(4u * (uint32_t)(i & 3)), "r"(r[t]), "r"(0x9E3779B9u));
#undef W1_A8
#undef W1_ARM
#undef W1_MX
#undef W1_LX
#undef W1_OUT
#undef W1_ARMS
#undef W1_L4
#undef W1_M4
    }
    if constexpr (X == W_LEA) asm volatile("{\n.reg .b32 q;\nshl.b32 q,%0,3;\nadd.u32 %0,q,%1;\n}" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == W_IADD64) asm volatile("add.u64 %0,%0,%1;" : "+l"(q[s]) : "l"(q[t]));
    if constexpr (X == W_SHFLI) asm volatile("shfl.sync.idx.b32 %0,%0,%1,0x1f,0xffffffff;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == W_SHFLU) asm volatile("shfl.sync.up.b32 %0,%0,1,0,0xffffffff;" : "+r"(r[s]));
    if constexpr (X == W_FRND64)
      asm volatile("{\nadd.rn.f64 %0,%0,0d3FE8000000000000;\ncvt.rni.f64.f64 %0,%0;\n}" : "+d"(d[s]));
  }
  DC_DEV uint32_t fold() const {
    uint32_t x = pm;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      x += r[i];
      const unsigned long long w = (unsigned long long)__double_as_longlong(d[i]);
      x ^= (uint32_t)w ^ (uint32_t)(w >> 32);
      x += (uint32_t)q[i] ^ (uint32_t)(q[i] >> 32) ^ rr[2 * i] ^ rr[2 * i + 1];
    }
    return x;
  }
};

// R ops a trip: the forms X... round-robin, each on its own run of slots (h32.cuh's h32_mix)
template <int... X, size_t... P>
DC_DEV void w1_round_(WLane& l, int b, std::index_sequence<P...>) {
  constexpr int n = sizeof...(X);
  (l.template op<X>(b * n + (int)P), ...);
}
template <int R, int... X>
DC_DEV void w1_mix(WLane& l) {
  static_assert(R % sizeof...(X) == 0, "a trip holds whole rounds of the mix");
#pragma unroll
  for (int b = 0; b < R / (int)sizeof...(X); b++) w1_round_<X...>(l, b, std::make_index_sequence<sizeof...(X)>{});
}
