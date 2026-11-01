// The IMAD forms of duo's ALU work, alone and beside the fold (SPEC §2.8: an idle unit that does the
// binding row's mandatory work more cheaply). duo is LOP3 + IMAD + LEA.HI a word, two of its three on the ALU pipe;
// LEA.HI's rotl(u, r) + y has an exact IMAD form, hi(u·2^r) + y on IMAD.HI.U32 and then u·2^r + that on IMAD (the
// multiplier comes from a kernel argument, so ptxas can't turn the IMAD back into a shift or LEA). Built as scalar.cu;
// sm90_imad.cu and sm100_imad.cu pin each loop's counts:
//   s_imadhi            1024 threads (8 warps a SMSP), 64 IMAD.HI.U32 an iteration on 8 chains
//   d_<x>               32 duo words and 32 ops of x (IMAD.HI.U32, IMAD, LOP3), on independent chains: x beside the fold
//   f_duo, f_duoi, f_duom  24 duo words: plain, every word's LEA.HI in its IMAD form, one word in three in it
#pragma once
#include "../../opcode_probe/probe.cuh"
#include "lanes.cuh"

template <int T>
DC_DEV void duo_imad(DuoSt& s, uint32_t p, uint32_t sh) {
  constexpr uint32_t a = duo_mul(T % DUO_NPOS), kx = duo_xor(T % DUO_NPOS);
  constexpr int r = duo_rot(T % DUO_NPOS);
  const uint32_t m = 1u << (r + sh), u = s.y ^ p ^ kx;
  s.y = u * a + s.z;
  uint32_t t;
  asm volatile("mad.hi.u32 %0,%1,%2,%3;" : "=r"(t) : "r"(u), "r"(m), "r"(s.y));
  asm volatile("mad.lo.u32 %0,%1,%2,%3;" : "=r"(s.z) : "r"(u), "r"(m), "r"(t));
}

enum { X_IMADHI, X_IMAD, X_LOP3 };
struct Side {
  uint32_t r[8];
  DC_DEV explicit Side(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) r[i] = (tid * 0x2545F491u + i * 0x9E3779B9u) | 1u;
  }
  template <int X>
  DC_DEV void op(int i) {
    const int s = i & 7, t = (i + 3) & 7;
    if constexpr (X == X_IMADHI) asm volatile("mad.hi.u32 %0,%1,%2,%0;" : "+r"(r[s]) : "r"(r[t]), "r"(0x9E3779B9u));
    if constexpr (X == X_IMAD) asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(0x9E3779B9u));
    if constexpr (X == X_LOP3) asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(r[s]) : "r"(r[t]), "r"(0x85EBCA6Bu));
  }
  DC_DEV uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x += r[i];
    return x;
  }
};

// NW duo words; word q is in its IMAD form when q % EVERY == EVERY - 1 (EVERY 0: none)
template <int Q, int NW, int EVERY>
DC_DEV void duo_words(Lane& l, uint32_t sh) {
  if constexpr (Q < NW) {
    DuoSt st{l.r[Q & 7], l.r[(Q + 4) & 7]};
    if constexpr (EVERY && Q % EVERY == EVERY - 1) duo_imad<Q % DUO_NPOS>(st, l.r[(Q + 1) & 7], sh);
    else duo<Q % DUO_NPOS>(st, l.r[(Q + 1) & 7]);
    l.r[Q & 7] = st.y;
    l.r[(Q + 4) & 7] = st.z;
    duo_words<Q + 1, NW, EVERY>(l, sh);
  }
}

DC_DEV void s_imadhi_k(const OpArgs& a) {
  Side x(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < 64; q++) x.op<X_IMADHI>(q);
  }
  OP_END
  OP_SINK(x.fold() ^ threadIdx.x)
}
template <int X>
DC_DEV void d_k(const OpArgs& a) {
  Lane l(threadIdx.x ^ (uint32_t)a.mode);
  Side x(threadIdx.x * 3u ^ (uint32_t)a.mode);
  const uint32_t sh = (uint32_t)a.mode >> 30;
  OP_BEGIN
  OP_LOOP(it) {
    duo_words<0, 32, 0>(l, sh);
#pragma unroll
    for (int q = 0; q < 32; q++) x.op<X>(q);
  }
  OP_END
  OP_SINK(l.fold() ^ x.fold() ^ threadIdx.x)
}
template <int EVERY>
DC_DEV void f_k(const OpArgs& a) {
  Lane l(threadIdx.x ^ (uint32_t)a.mode);
  const uint32_t sh = (uint32_t)a.mode >> 30;
  OP_BEGIN
  OP_LOOP(it) { duo_words<0, 24, EVERY>(l, sh); }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}

OP_KERNEL(s_imadhi, 1024) { s_imadhi_k(a); }
OP_KERNEL(d_imadhi, 1024) { d_k<X_IMADHI>(a); }
OP_KERNEL(d_imad, 1024) { d_k<X_IMAD>(a); }
OP_KERNEL(d_lop3, 1024) { d_k<X_LOP3>(a); }
OP_KERNEL(f_duo, 1024) { f_k<0>(a); }
OP_KERNEL(f_duoi, 1024) { f_k<1>(a); }
OP_KERNEL(f_duom, 1024) { f_k<3>(a); }

#define P_I(NAME, PAT, CNT) {.name = #NAME, .kernel = (const void*)NAME, .tpb = 1024, .expect = PAT, .counted = CNT, .iters = 4096}
