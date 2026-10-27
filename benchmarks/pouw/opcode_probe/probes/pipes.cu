// The pipes of sm_120's lane opcodes that the coverage table places on a pipe no run had placed them on
// (machine_model/data/cover/rows.toml, CMP-166). Joint points: rows.jsonl only, no census line; pipes_verdict.py reads
// them. Each warp runs 8 independent chains of the opcode (4 for predicates: a thread has 7), alone and beside a partner
// on the pipe the table puts it on, on chains of their own:
//   lop3, mufu                       the partners alone: LOP3 on the ALU pipe (2 warp instructions a SM-clock), MUFU.EX2
//                                    on the XU (0.5)
//   plop3, sgxt, bmsk                the table's alu rows: each alone and beside 8 LOP3 (mix_X_lop3); on alu, each alone
//                                    and the pair together hold at most 2
//   iadd, iadd64, iabs               the add rows the table puts on fma (issue-bound), alone and beside 8 LOP3
//   i2f_f2i, i2f16_f2i16, f2f64, frnd
//                                    the table's sfu conversions (a chain alternates a conversion and its inverse; FRND's
//                                    with an FADD): each alone and beside 8 MUFU.EX2 (mix_X_mufu); on the XU, each alone
//                                    and the pair together hold at most 0.5
//   bar_or, mix_bar_or_lop3          BAR.RED.OR alone and beside 16 LOP3: on the collective at 16 alu lanes, LOP3's rate
//                                    plus half the barrier's holds at most 2
#include "../probe.cuh"

constexpr int TPB = 1024;

// P: 0 none, 1 LOP3, 2 MUFU.EX2, each Q/8 a chain step, on 8 chains of their own.
template <int P, int Q>
struct Partner {
  uint32_t w[8];
  float f[8];
  __device__ __forceinline__ explicit Partner(const OpArgs& a) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      w[u] = threadIdx.x * 8u + (unsigned)u + 1u;
      f[u] = -0.001f * (float)(threadIdx.x + (unsigned)u + (unsigned)a.mode);
    }
  }
  __device__ __forceinline__ void step(int u, uint32_t k1, uint32_t k2) {
#pragma unroll
    for (int q = u * Q / 8; q < (u + 1) * Q / 8; q++) {
      if constexpr (P == 1) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(w[q & 7]) : "r"(k1), "r"(k2));
      if constexpr (P == 2) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[q & 7]));
    }
  }
  __device__ __forceinline__ uint32_t out() const {
    uint32_t s = 0;
#pragma unroll
    for (int u = 0; u < 8; u++) s ^= (P == 1 ? w[u] : 0u) ^ (P == 2 ? __float_as_uint(f[u]) : 0u);
    return s;
  }
};

struct Sgxt {
  __device__ __forceinline__ uint32_t operator()(uint32_t x, uint32_t k) const {
    asm volatile("szext.clamp.s32 %0, %0, %1;" : "+r"(x) : "r"(k));
    return x;
  }
};
struct Bmsk {
  __device__ __forceinline__ uint32_t operator()(uint32_t x, uint32_t k) const {
    asm volatile("bmsk.clamp.b32 %0, %0, %1;" : "+r"(x) : "r"(k));
    return x;
  }
};
struct Iadd {
  __device__ __forceinline__ uint32_t operator()(uint32_t x, uint32_t k) const {
    asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(k));
    return x;
  }
};
struct Iabs {
  __device__ __forceinline__ uint32_t operator()(uint32_t x, uint32_t) const {
    asm volatile("abs.s32 %0, %0;" : "+r"(x));
    return x;
  }
};
struct I2fF2i {
  __device__ __forceinline__ uint32_t operator()(uint32_t x, uint32_t) const {
    float f;
    asm volatile("cvt.rp.f32.u32 %0, %1;" : "=f"(f) : "r"(x));
    asm volatile("cvt.rzi.ftz.u32.f32 %0, %1;" : "=r"(x) : "f"(f));
    return x;
  }
};
struct F2f64 {
  __device__ __forceinline__ uint32_t operator()(uint32_t x, uint32_t) const {
    float f = __uint_as_float(x);
    double d;
    asm volatile("cvt.f64.f32 %0, %1;" : "=d"(d) : "f"(f));
    asm volatile("cvt.rn.f32.f64 %0, %1;" : "=f"(f) : "d"(d));
    return __float_as_uint(f);
  }
};
struct Frnd {
  __device__ __forceinline__ uint32_t operator()(uint32_t x, uint32_t k) const {
    float f = __uint_as_float(x);
    asm volatile("cvt.rmi.f32.f32 %0, %0;" : "+f"(f));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(f) : "f"(__uint_as_float(k)));
    return __float_as_uint(f);
  }
};

template <typename Op, int P, int Q>
__device__ __forceinline__ void chains(const OpArgs& a, uint32_t start_xor, uint32_t k) {
  uint32_t v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = ((unsigned)a.iters * 977u + threadIdx.x * 8u + (unsigned)u) ^ start_xor;
  Partner<P, Q> pt(a);
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  const Op op;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      v[u] = op(v[u], k);
      pt.step(u, k1, k2);
    }
  }
  OP_END
  uint32_t s = pt.out();
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= v[u];
  OP_SINK(s)
}

template <int P, int Q>
__device__ __forceinline__ void partner_only(const OpArgs& a) {
  Partner<P, Q> pt(a);
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) pt.step(u, k1, k2);
  }
  OP_END
  OP_SINK(pt.out())
}

template <int P, int Q>
__device__ __forceinline__ void iadd64(const OpArgs& a) {
  unsigned long long v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = ((unsigned long long)a.iters << 33) + threadIdx.x * 8u + (unsigned)u;
  const unsigned long long k = 0x100000001ull * (unsigned long long)(a.mode + 3);
  Partner<P, Q> pt(a);
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      asm volatile("add.u64 %0, %0, %1;" : "+l"(v[u]) : "l"(k));
      pt.step(u, k1, k2);
    }
  }
  OP_END
  uint32_t s = pt.out();
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= (uint32_t)v[u] ^ (uint32_t)(v[u] >> 32);
  OP_SINK(s)
}

template <int P, int Q>
__device__ __forceinline__ void i2f16(const OpArgs& a) {
  unsigned short h[8];
#pragma unroll
  for (int u = 0; u < 8; u++) h[u] = (unsigned short)(threadIdx.x * 8u + (unsigned)u + (unsigned)a.mode);
  Partner<P, Q> pt(a);
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      float f;
      asm volatile("cvt.rn.f32.s16 %0, %1;" : "=f"(f) : "h"(h[u]));
      asm volatile("cvt.rzi.s16.f32 %0, %1;" : "=h"(h[u]) : "f"(f));
      pt.step(u, k1, k2);
    }
  }
  OP_END
  uint32_t s = pt.out();
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= h[u];
  OP_SINK(s)
}

// Predicate chains p ^= q, 4 of them: with q and the loop's own predicate that is 6 of a thread's 7.
template <int P, int Q>
__device__ __forceinline__ void plop3(const OpArgs& a) {
  bool p[4];
#pragma unroll
  for (int u = 0; u < 4; u++) p[u] = ((threadIdx.x >> u) & 1u) != 0u;
  const bool q = (a.mode & 1) == 0;
  Partner<P, Q> pt(a);
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      if (u < 4) p[u] = p[u] != q;
      pt.step(u, k1, k2);
    }
  }
  OP_END
  uint32_t s = pt.out();
#pragma unroll
  for (int u = 0; u < 4; u++) s ^= (uint32_t)p[u] << u;
  OP_SINK(s)
}

template <int P, int Q>
__device__ __forceinline__ void bar_or(const OpArgs& a) {
  int v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = (int)((threadIdx.x >> u) & 1u) ^ (a.mode & 1);
  Partner<P, Q> pt(a);
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      v[u] = __syncthreads_or(v[u]);
      pt.step(u, k1, k2);
    }
  }
  OP_END
  uint32_t s = pt.out();
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= (uint32_t)v[u] << u;
  OP_SINK(s)
}

#define KBITS (13u + ((uint32_t)a.mode & 7u))
#define KHALF (__float_as_uint(0.75f + 1e-3f * (float)a.mode))
OP_KERNEL(lop3, TPB) { partner_only<1, 8>(a); }
OP_KERNEL(mufu, TPB) { partner_only<2, 8>(a); }
OP_KERNEL(sgxt, TPB) { chains<Sgxt, 0, 0>(a, 0u, KBITS); }
OP_KERNEL(mix_sgxt_lop3, TPB) { chains<Sgxt, 1, 8>(a, 0u, KBITS); }
OP_KERNEL(bmsk, TPB) { chains<Bmsk, 0, 0>(a, 0u, KBITS); }
OP_KERNEL(mix_bmsk_lop3, TPB) { chains<Bmsk, 1, 8>(a, 0u, KBITS); }
OP_KERNEL(plop3_k, TPB) { plop3<0, 0>(a); }
OP_KERNEL(mix_plop3_lop3, TPB) { plop3<1, 8>(a); }
OP_KERNEL(iadd, TPB) { chains<Iadd, 0, 0>(a, 0u, 0x9E3779B9u ^ (uint32_t)a.mode); }
OP_KERNEL(mix_iadd_lop3, TPB) { chains<Iadd, 1, 8>(a, 0u, 0x9E3779B9u ^ (uint32_t)a.mode); }
OP_KERNEL(iadd64_k, TPB) { iadd64<0, 0>(a); }
OP_KERNEL(mix_iadd64_lop3, TPB) { iadd64<1, 8>(a); }
OP_KERNEL(iabs, TPB) { chains<Iabs, 0, 0>(a, 0x80000000u, 0u); }
OP_KERNEL(mix_iabs_lop3, TPB) { chains<Iabs, 1, 8>(a, 0x80000000u, 0u); }
OP_KERNEL(i2f_f2i, TPB) { chains<I2fF2i, 0, 0>(a, 0u, 0u); }
OP_KERNEL(mix_i2f_f2i_mufu, TPB) { chains<I2fF2i, 2, 8>(a, 0u, 0u); }
OP_KERNEL(i2f16_f2i16, TPB) { i2f16<0, 0>(a); }
OP_KERNEL(mix_i2f16_f2i16_mufu, TPB) { i2f16<2, 8>(a); }
OP_KERNEL(f2f64, TPB) { chains<F2f64, 0, 0>(a, 0x3F800000u, 0u); }
OP_KERNEL(mix_f2f64_mufu, TPB) { chains<F2f64, 2, 8>(a, 0x3F800000u, 0u); }
OP_KERNEL(frnd, TPB) { chains<Frnd, 0, 0>(a, 0x3F800000u, KHALF); }
OP_KERNEL(mix_frnd_mufu, TPB) { chains<Frnd, 2, 8>(a, 0x3F800000u, KHALF); }
OP_KERNEL(bar_or_k, TPB) { bar_or<0, 0>(a); }
OP_KERNEL(mix_bar_or_lop3, TPB) { bar_or<1, 16>(a); }

#define PR(N, K, E) {.name = N, .kernel = (const void*)K, .tpb = TPB, .expect = E}
static const OpProbe PROBES[] = {
    PR("lop3", lop3, "LOP3.LUT=8"),
    PR("mufu", mufu, "MUFU.EX2=8"),
    PR("sgxt", sgxt, "SGXT=8"),
    PR("mix_sgxt_lop3", mix_sgxt_lop3, "SGXT=8,LOP3.LUT=8"),
    PR("bmsk", bmsk, "BMSK=8"),
    PR("mix_bmsk_lop3", mix_bmsk_lop3, "BMSK=8,LOP3.LUT=8"),
    PR("plop3", plop3_k, "PLOP3.LUT=4"),
    PR("mix_plop3_lop3", mix_plop3_lop3, "PLOP3.LUT=4,LOP3.LUT=8"),
    PR("iadd", iadd, "IADD=8"),
    PR("mix_iadd_lop3", mix_iadd_lop3, "IADD=8,LOP3.LUT=8"),
    PR("iadd64", iadd64_k, "IADD.64=8"),
    PR("mix_iadd64_lop3", mix_iadd64_lop3, "IADD.64=8,LOP3.LUT=8"),
    PR("iabs", iabs, "IABS=8"),
    PR("mix_iabs_lop3", mix_iabs_lop3, "IABS=8,LOP3.LUT=8"),
    PR("i2f_f2i", i2f_f2i, "I2F.U32.RP=8,F2I.FTZ.U32.TRUNC.NTZ=8"),
    PR("mix_i2f_f2i_mufu", mix_i2f_f2i_mufu, "I2F.U32.RP=8,F2I.FTZ.U32.TRUNC.NTZ=8,MUFU.EX2=8"),
    PR("i2f16_f2i16", i2f16_f2i16, "I2F.S16=8,F2I.S16.TRUNC.NTZ=8"),
    PR("mix_i2f16_f2i16_mufu", mix_i2f16_f2i16_mufu, "I2F.S16=8,F2I.S16.TRUNC.NTZ=8,MUFU.EX2=8"),
    PR("f2f64", f2f64, "F2F.F64.F32=8,F2F.F32.F64=8"),
    PR("mix_f2f64_mufu", mix_f2f64_mufu, "F2F.F64.F32=8,F2F.F32.F64=8,MUFU.EX2=8"),
    PR("frnd", frnd, "FRND.FLOOR=8,FADD=8"),
    PR("mix_frnd_mufu", mix_frnd_mufu, "FRND.FLOOR=8,FADD=8,MUFU.EX2=8"),
    PR("bar_or", bar_or_k, "BAR.RED.OR*=8"),
    PR("mix_bar_or_lop3", mix_bar_or_lop3, "BAR.RED.OR*=8,LOP3.LUT=16"),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
