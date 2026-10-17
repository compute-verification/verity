// The scalar lane ops the DC probes time alone (scalar.cu) and beside the tensor pipe (sm90_wgmma.cu, sm100_tcgen05.cu).
// Each op reads and writes the thread's own 8 registers (8 independent chains), so ptxas can neither fold nor hoist it.
#pragma once
#include "../dc.cuh"

// mix: add.u32 and FFMA alternating (ptxas splits the adds between IADD3 and IMAD.IADD); duo: the fold, 3 ops a word;
// li: LOP3 and IMAD alternating, LI's op mix; lif: LI with FFMA filler.
enum { L_MIX, L_IMAD, L_LOP3, L_SHFL, L_DUO, L_IADD, L_FFMA, L_SHF, L_LEA, L_LI, L_LIF };
struct Lane {
  uint32_t r[8];
  float f[8];
  DC_DEV explicit Lane(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
      f[i] = 1.0f + (float)(r[i] & 0xffu) * 0x1p-12f;
    }
  }
  template <int X>
  DC_DEV void op(int i) {
    const int s = i & 7, t = (i + 1) & 7;
    if constexpr (X == L_MIX) {
      if (i & 1) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(f[(i >> 1) & 7]) : "f"(f[((i >> 1) + 1) & 7]));
      else asm volatile("add.u32 %0,%0,%1;" : "+r"(r[(i >> 1) & 7]) : "r"(r[((i >> 1) + 1) & 7]));
    }
    if constexpr (X == L_IMAD) asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(0x9E3779B9u));
    if constexpr (X == L_LOP3) asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(r[s]) : "r"(r[t]), "r"(0x85EBCA6Bu));
    if constexpr (X == L_SHFL) asm volatile("shfl.sync.bfly.b32 %0,%0,1,0x1f,0xffffffff;" : "+r"(r[s]));
    if constexpr (X == L_IADD) asm volatile("add.u32 %0,%0,%1;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == L_FFMA) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == L_SHF) asm volatile("shf.l.wrap.b32 %0,%0,%1,13;" : "+r"(r[s]) : "r"(r[t]));
    // rotl(x, 13) + y: ptxas fuses it into one LEA.HI (duo's third op)
    if constexpr (X == L_LEA)
      asm volatile("{\n.reg .b32 t;\nshf.l.wrap.b32 t,%1,%1,13;\nadd.u32 %0,%0,t;\n}" : "+r"(r[s]) : "r"(r[t]));
  }
  // p is a word of the chain's own state, so ptxas can't fold it to a constant.
  template <int T>
  DC_DEV void duo_op(int i) {
    DuoSt st{r[i & 7], r[(i + 4) & 7]};
    duo<T>(st, r[(i + 1) & 7]);
    r[i & 7] = st.y;
    r[(i + 4) & 7] = st.z;
  }
  DC_DEV uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x += r[i] ^ __float_as_uint(f[i]);
    return x;
  }
};
template <int Q, int N>
DC_DEV void duo_ops(Lane& l) {
  if constexpr (Q < N) {
    l.duo_op<Q % DUO_NPOS>(Q);
    duo_ops<Q + 1, N>(l);
  }
}
// R ops alternating X and Y: when the pair's rate exceeds both alone rates, X and Y issue to different pipes.
template <int X, int Y, int R>
DC_DEV void lane_pair(Lane& l) {
#pragma unroll
  for (int q = 0; q < R; q += 2) {
    l.op<X>(q >> 1);
    l.op<Y>((q >> 1) + 4);
  }
}
// R lane ops (the duo lane: R/3 words of three ops each; the LI lane: LI's LOP3 + IMAD pair, R/2 of each; lif: LI with
// one FFMA filler op in eight, 7 LOP3 + 7 IMAD + 2 FFMA a 16)
template <int X, int R>
DC_DEV void lane_ops(Lane& l) {
  if constexpr (X == L_DUO) {
    duo_ops<0, R / 3>(l);
  } else if constexpr (X == L_LI) {
    lane_pair<L_LOP3, L_IMAD, R>(l);
  } else if constexpr (X == L_LIF) {
#pragma unroll
    for (int q = 0; q < R; q++) {
      if (q % 8 == 7) l.op<L_FFMA>(q / 8);
      else if ((q % 8 + q / 8) % 2 == 0) l.op<L_LOP3>(q);
      else l.op<L_IMAD>(q);
    }
  } else {
#pragma unroll
    for (int q = 0; q < R; q++) l.op<X>(q);
  }
}
