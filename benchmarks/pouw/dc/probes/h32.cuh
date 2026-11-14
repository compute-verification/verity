// The H32 probes' lane (sm90_h32*.cu): 8 independent chains of each register kind a thread (u32, f32, f64 and f16x2),
// and one op a form of MH100's scalar classes. An op writes its chain from itself and its neighbours, so ptxas can
// neither fold nor hoist it. An integer form with an immediate alternates on its chain with a form it can't compose
// with (add then xor, multiply then xor): alone, ptxas folds a chain's adds of constants into one. A conversion runs
// with its inverse on one chain, so each one's result is read.
#pragma once
#include <utility>
#include "../dc.cuh"

enum : int {
  // ffma: FP32 add, multiply and fused multiply-add; _I an immediate in place of a register source
  O_FADD_I, O_FADD_R, O_FMUL_I, O_FMUL_R, O_FFMA_RI, O_FFMA_RR, O_FFMA_II,
  // int32
  O_IADD_R, O_IADD3_RR, O_ADDXOR_I, O_MULXOR_I, O_LOP3_R, O_LOP3_RR, O_SHF_R, O_LEA, O_IMAD_R, O_IMAD_RR, O_IMADW,
  O_IMADHI, O_IDP4A, O_IDP2A, O_IMNMX, O_POPC, O_FLO,
  // fp64
  O_DADD_I, O_DADD_R, O_DMUL_I, O_DMUL_R, O_DFMA_RI, O_DFMA_RR, O_DMNMX, O_F2F64,
  // other: half2 and bf16x2, packs and conversions, select, min/max, MUFU, byte permute, warp exchange
  O_HADD2_I, O_HADD2_R, O_HMUL2_I, O_HMUL2_R, O_HFMA2_RI, O_HFMA2_RR, O_BADD2_R, O_BFMA2_RI, O_HMNMX2,
  O_F2FP_B, O_E4M3, O_FMNMX, O_FSEL, O_SEL, O_PRMT, O_EX2, O_RSQ, O_F2I, O_FRND, O_SHFL, O_VOTE,
};

// The chains a form touches (fold reads only those): 1 u32, 2 f32, 4 f64, 8 f16x2.
DC_HD constexpr unsigned h32_uses(int x) {
  if (x >= O_FADD_I && x <= O_FFMA_II) return 2;
  if (x >= O_IADD_R && x <= O_FLO) return 1;
  if (x == O_F2F64) return 4 | 2;
  if (x >= O_DADD_I && x <= O_DMNMX) return 4;
  if (x >= O_HADD2_I && x <= O_HMNMX2) return 8;
  if (x == O_F2FP_B) return 8 | 2;
  if (x == O_E4M3) return 8;
  if (x == O_FMNMX || x == O_FSEL || x == O_EX2 || x == O_RSQ || x == O_FRND) return 2;
  if (x == O_F2I) return 1 | 2;
  return 1;
}

struct HLane {
  uint32_t r[8];
  float f[8];
  double d[8];
  uint32_t h[8];
  uint32_t sel;
  DC_DEV explicit HLane(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
      f[i] = 1.0f + (float)(r[i] & 0xffu) * 0x1p-12f;
      d[i] = 1.0 + (double)(r[i] & 0xffffu) * 0x1p-20;
      h[i] = 0x3C003C00u | (r[i] & 0x00FF00FFu);
    }
    sel = tid & 1u;
  }
  // i: the op's slot in the trip, a constant once the trip is unrolled; its chain is i & 7, its use of it i >> 3.
  template <int X>
  DC_DEV void op(int i) {
    const int s = i & 7, t = (i + 1) & 7, u = (i + 2) & 7;
    const bool odd = (i >> 3) & 1;
    if constexpr (X == O_FADD_I) asm volatile("add.rn.f32 %0,%0,0f3F800000;" : "+f"(f[s]));
    if constexpr (X == O_FADD_R) asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == O_FMUL_I) asm volatile("mul.rn.f32 %0,%0,0f3F7FFF58;" : "+f"(f[s]));
    if constexpr (X == O_FMUL_R) asm volatile("mul.rn.f32 %0,%0,%1;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == O_FFMA_RI) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == O_FFMA_RR) asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(f[s]) : "f"(f[t]), "f"(f[u]));
    if constexpr (X == O_FFMA_II) asm volatile("fma.rn.f32 %0,%0,0f3F7FFF58,0f3F800000;" : "+f"(f[s]));
    if constexpr (X == O_IADD_R) asm volatile("add.u32 %0,%0,%1;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == O_IADD3_RR)
      asm volatile("{\n.reg .u32 q;\nadd.u32 q,%1,%2;\nadd.u32 %0,%0,q;\n}" : "+r"(r[s]) : "r"(r[t]), "r"(r[u]));
    if constexpr (X == O_ADDXOR_I) {
      if (odd) asm volatile("xor.b32 %0,%0,0x85EBCA6B;" : "+r"(r[s]));
      else asm volatile("add.u32 %0,%0,0x9E3779B9;" : "+r"(r[s]));
    }
    if constexpr (X == O_MULXOR_I) {
      if (odd) asm volatile("xor.b32 %0,%0,0x85EBCA6B;" : "+r"(r[s]));
      else asm volatile("mul.lo.u32 %0,%0,0x9E3779B9;" : "+r"(r[s]));
    }
    if constexpr (X == O_LOP3_R) asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(r[s]) : "r"(r[t]), "r"(0x85EBCA6Bu));
    if constexpr (X == O_LOP3_RR) asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(r[s]) : "r"(r[t]), "r"(r[u]));
    if constexpr (X == O_SHF_R) asm volatile("shf.l.wrap.b32 %0,%0,%1,13;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == O_LEA)
      asm volatile("{\n.reg .b32 q;\nshf.l.wrap.b32 q,%1,%1,13;\nadd.u32 %0,%0,q;\n}" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == O_IMAD_R) asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(0x9E3779B9u));
    if constexpr (X == O_IMAD_RR) asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(r[u]));
    if constexpr (X == O_IMADW) {
      const int p = 2 * (i & 3);
      asm volatile("{\n.reg .b64 w;\nmov.b64 w,{%0,%1};\nmad.wide.u32 w,%0,%2,w;\nmov.b64 {%0,%1},w;\n}"
                   : "+r"(r[p]), "+r"(r[p + 1])
                   : "r"(0x9E3779B9u));
    }
    if constexpr (X == O_IMADHI) asm volatile("mul.hi.u32 %0,%0,%1;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == O_IDP4A) asm volatile("dp4a.u32.u32 %0,%0,%1,%0;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == O_IDP2A) asm volatile("dp2a.lo.u32.u32 %0,%0,%1,%0;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == O_IMNMX) asm volatile("min.u32 %0,%0,%1;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == O_POPC) asm volatile("popc.b32 %0,%0;" : "+r"(r[s]));
    if constexpr (X == O_FLO) asm volatile("bfind.u32 %0,%0;" : "+r"(r[s]));
    if constexpr (X == O_DADD_I) asm volatile("add.rn.f64 %0,%0,0d3FF0000000000000;" : "+d"(d[s]));
    if constexpr (X == O_DADD_R) asm volatile("add.rn.f64 %0,%0,%1;" : "+d"(d[s]) : "d"(d[t]));
    if constexpr (X == O_DMUL_I) asm volatile("mul.rn.f64 %0,%0,0d3FEFFFFF00000000;" : "+d"(d[s]));
    if constexpr (X == O_DMUL_R) asm volatile("mul.rn.f64 %0,%0,%1;" : "+d"(d[s]) : "d"(d[t]));
    if constexpr (X == O_DFMA_RI) asm volatile("fma.rn.f64 %0,%0,%1,0d3FF0000000000000;" : "+d"(d[s]) : "d"(d[t]));
    if constexpr (X == O_DFMA_RR) asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(d[s]) : "d"(d[t]), "d"(d[u]));
    if constexpr (X == O_DMNMX) asm volatile("min.f64 %0,%0,%1;" : "+d"(d[s]) : "d"(d[t]));
    if constexpr (X == O_F2F64) {
      if (odd) asm volatile("cvt.f64.f32 %0,%1;" : "=d"(d[s]) : "f"(f[s]));
      else asm volatile("cvt.rn.f32.f64 %0,%1;" : "=f"(f[s]) : "d"(d[s]));
    }
    if constexpr (X == O_HADD2_I) asm volatile("add.rn.f16x2 %0,%0,%1;" : "+r"(h[s]) : "r"(0x3C003C00u));
    if constexpr (X == O_HADD2_R) asm volatile("add.rn.f16x2 %0,%0,%1;" : "+r"(h[s]) : "r"(h[t]));
    if constexpr (X == O_HMUL2_I) asm volatile("mul.rn.f16x2 %0,%0,%1;" : "+r"(h[s]) : "r"(0x3BFF3BFFu));
    if constexpr (X == O_HMUL2_R) asm volatile("mul.rn.f16x2 %0,%0,%1;" : "+r"(h[s]) : "r"(h[t]));
    if constexpr (X == O_HFMA2_RI) asm volatile("fma.rn.f16x2 %0,%0,%1,%2;" : "+r"(h[s]) : "r"(h[t]), "r"(0x3C003C00u));
    if constexpr (X == O_HFMA2_RR) asm volatile("fma.rn.f16x2 %0,%0,%1,%2;" : "+r"(h[s]) : "r"(h[t]), "r"(h[u]));
    if constexpr (X == O_BADD2_R) asm volatile("add.rn.bf16x2 %0,%0,%1;" : "+r"(h[s]) : "r"(h[t]));
    if constexpr (X == O_BFMA2_RI)
      asm volatile("fma.rn.bf16x2 %0,%0,%1,%2;" : "+r"(h[s]) : "r"(h[t]), "r"(0x3F803F80u));
    if constexpr (X == O_HMNMX2) asm volatile("min.f16x2 %0,%0,%1;" : "+r"(h[s]) : "r"(h[t]));
    // f32 pair -> bf16x2 (F2FP pack), then the low half back to f32 (a shift: bf16 is f32's high half)
    if constexpr (X == O_F2FP_B) {
      if (odd) asm volatile("shl.b32 %0,%1,16;" : "=f"(f[s]) : "r"(h[s]));
      else asm volatile("cvt.rn.bf16x2.f32 %0,%1,%2;" : "=r"(h[s]) : "f"(f[t]), "f"(f[s]));
    }
    // f16x2 -> e4m3x2 -> f16x2: two F2FP an op
    if constexpr (X == O_E4M3)
      asm volatile("{\n.reg .b16 q;\ncvt.rn.satfinite.e4m3x2.f16x2 q,%0;\ncvt.rn.f16x2.e4m3x2 %0,q;\n}" : "+r"(h[s]));
    if constexpr (X == O_FMNMX) asm volatile("min.f32 %0,%0,%1;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == O_FSEL)
      asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p,%2,0;\nselp.f32 %0,%0,%1,p;\n}" : "+f"(f[s]) : "f"(f[t]), "r"(sel));
    if constexpr (X == O_SEL)
      asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p,%2,0;\nselp.b32 %0,%0,%1,p;\n}" : "+r"(r[s]) : "r"(r[t]), "r"(sel));
    if constexpr (X == O_PRMT) asm volatile("prmt.b32 %0,%0,%1,0x6420;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == O_EX2) asm volatile("ex2.approx.ftz.f32 %0,%0;" : "+f"(f[s]));
    if constexpr (X == O_RSQ) asm volatile("rsqrt.approx.ftz.f32 %0,%0;" : "+f"(f[s]));
    if constexpr (X == O_F2I) {
      if (odd) asm volatile("cvt.rn.f32.s32 %0,%1;" : "=f"(f[s]) : "r"(r[s]));
      else asm volatile("cvt.rzi.s32.f32 %0,%1;" : "=r"(r[s]) : "f"(f[s]));
    }
    if constexpr (X == O_FRND) {
      if (odd) asm volatile("cvt.rni.f32.f32 %0,%0;" : "+f"(f[s]));
      else asm volatile("add.rn.f32 %0,%0,0f3F000000;" : "+f"(f[s]));
    }
    if constexpr (X == O_SHFL) asm volatile("shfl.sync.bfly.b32 %0,%0,1,0x1f,0xffffffff;" : "+r"(r[s]));
    if constexpr (X == O_VOTE)
      asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p,%1,0;\nvote.sync.ballot.b32 %0,p,0xffffffff;\n}"
                   : "=r"(r[s])
                   : "r"(r[t] & 0x10u));
  }
  template <unsigned M>
  DC_DEV uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr ((M & 1) != 0) x += r[i];
      if constexpr ((M & 2) != 0) x ^= __float_as_uint(f[i]);
      if constexpr ((M & 4) != 0) {
        const unsigned long long w = (unsigned long long)__double_as_longlong(d[i]);
        x += (uint32_t)w ^ (uint32_t)(w >> 32);
      }
      if constexpr ((M & 8) != 0) x ^= h[i];
    }
    return x;
  }
};

// Round b of the forms X..., each on its own run of slots, so a form named twice uses two chains.
template <int... X, size_t... P>
DC_DEV void h32_round_(HLane& l, int b, std::index_sequence<P...>) {
  constexpr int n = sizeof...(X);
  (l.template op<X>(b * n + (int)P), ...);
}
template <int... X>
DC_DEV void h32_round(HLane& l, int b) {
  h32_round_<X...>(l, b, std::make_index_sequence<sizeof...(X)>{});
}
// R ops a trip: the forms X... round-robin.
template <int R, int... X>
DC_DEV void h32_mix(HLane& l) {
  static_assert(R % sizeof...(X) == 0, "a trip holds whole rounds of the mix");
#pragma unroll
  for (int b = 0; b < R / (int)sizeof...(X); b++) h32_round<X...>(l, b);
}
template <int... X>
DC_DEV uint32_t h32_fold(const HLane& l) {
  return l.fold<(h32_uses(X) | ...)>();
}
