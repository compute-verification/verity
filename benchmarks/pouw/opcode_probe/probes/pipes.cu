// The pipes of sm_120's lane opcodes that the coverage table places on a pipe no run had placed them on
// (machine_model/data/cover/rows.toml). Joint points: rows.jsonl only, no census line; pipes_verdict.py reads
// them. Each warp runs 8 independent chains of the opcode, alone and beside a partner on the pipe the table puts it on,
// on chains of their own:
//   lop3, mufu                       the partners alone: LOP3 on the ALU pipe (2 warp instructions a SM-clock), MUFU.EX2
//                                    on the XU (0.5)
//   plop3, sgxt, bmsk                the table's alu rows: each alone and beside 8 LOP3 (mix_X_lop3); on alu, each alone
//                                    and the pair together hold at most 2. PLOP3's 6 predicates live in one PTX loop:
//                                    ptxas keeps a C++ bool loop's state in registers (LOP3, ISETP and SEL, no PLOP3)
//   iadd64, iabs                     the add rows the table puts on fma (issue-bound), alone and beside 8 LOP3; ptxas 13.0
//                                    lowers a 32-bit add.u32 or sub.u32 to IADD3 or IMAD, never IADD
//   i2f_f2i, i2f16_f2i16, frnd,      the table's 32-bit conversions on sfu (a chain alternates a conversion and its inverse,
//   cvt_s32, cvt_u16, cvt_s8,        FRND.FLOOR's with an FADD, I2I and I2IP's a saturating narrowing): each alone and
//   cvt_frnd, cvt_i2i, cvt_i2ip      beside 8 MUFU.EX2 (mix_X_mufu); on the XU, each alone and the pair together hold at
//                                    most 0.5
//   bar_or, mix_bar_or_lop3          BAR.RED.OR alone and beside 16 LOP3: on the collective at 16 alu lanes, LOP3's rate
//                                    plus half the barrier's holds at most 2 (this mix is barrier-bound)
//   bar16, mix_bar16_lop3            BAR.RED.OR on named barrier 1 in 16 warps, alone and beside 32 LOP3 a trip in the
//                                    other 16 (an `until` role, so the LOP3 stream never waits on the barrier)
//   dfma, f2f64, i2f64, s64,         64-bit conversions (f32<->f64, s32<->f64, f32<->s64, f32<->u64, FRND.F64, u32<->f64,
//   cvt_u64, cvt_frnd64,             s64<->f64): each alone, one pair beside 24 MUFU.EX2 (mix_X_mufu24: on a unit of
//   cvt_f64_u32, cvt_f64_s64         their own, pair and MUFU together would pass 0.5), and two pairs beside 2 DFMA
//                                    (mix_X_dfma: sharing f64's pipe, they hold 0.0625 together)
#include "../probe.cuh"

constexpr int TPB = 1024;

// P: 0 none, 1 LOP3, 2 MUFU.EX2, 3 DFMA, each Q/8 a chain step, on 8 chains of their own.
template <int P, int Q>
struct Partner {
  uint32_t w[8];
  float f[8];
  double d[8];
  __device__ __forceinline__ explicit Partner(const OpArgs& a) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      w[u] = threadIdx.x * 8u + (unsigned)u + 1u;
      f[u] = -0.001f * (float)(threadIdx.x + (unsigned)u + (unsigned)a.mode);
      d[u] = 1.0 + 1e-6 * (double)(threadIdx.x + (unsigned)u + (unsigned)a.mode);
    }
  }
  __device__ __forceinline__ void step(int u, uint32_t k1, uint32_t k2) {
#pragma unroll
    for (int q = u * Q / 8; q < (u + 1) * Q / 8; q++) {
      if constexpr (P == 1) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(w[q & 7]) : "r"(k1), "r"(k2));
      if constexpr (P == 2) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[q & 7]));
      if constexpr (P == 3)
        asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(d[q & 7]) : "d"(0.999), "d"(1e-9));
    }
  }
  __device__ __forceinline__ uint32_t out() const {
    uint32_t s = 0;
#pragma unroll
    for (int u = 0; u < 8; u++) {
      s ^= (P == 1 ? w[u] : 0u) ^ (P == 2 ? __float_as_uint(f[u]) : 0u);
      if constexpr (P == 3) s ^= (uint32_t)__double_as_longlong(d[u]) ^ (uint32_t)(__double_as_longlong(d[u]) >> 32);
    }
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

struct I2fF64 {
  __device__ __forceinline__ uint32_t operator()(uint32_t x, uint32_t) const {
    double d;
    asm volatile("cvt.rn.f64.s32 %0, %1;" : "=d"(d) : "r"(x));
    asm volatile("cvt.rzi.s32.f64 %0, %1;" : "=r"(x) : "d"(d));
    return x;
  }
};

// N of the 8 chains run the opcode (the rest only the partner's steps).
template <typename Op, int P, int Q, int N = 8>
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
      if (u < N) v[u] = op(v[u], k);
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
__device__ __forceinline__ void iadd64_chains(const OpArgs& a) {
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

// 6 predicates, p_i ^= p_(i+1) twice round a trip, and PARTNER on 8 registers of their own, in one nounroll PTX loop of
// a.iters trips (a thread has 7 predicates; the trip count is the 7th).
#define PLOP3_LOOP(PARTNER)                                                                                  \
  asm volatile(                                                                                              \
      "{\n .reg .pred p0, p1, p2, p3, p4, p5, q;\n .reg .u32 n, t, w0, w1, w2, w3, w4, w5, w6, w7;\n"       \
      " and.b32 t, %1, 1; setp.ne.u32 p0, t, 0;\n and.b32 t, %1, 2; setp.ne.u32 p1, t, 0;\n"                \
      " and.b32 t, %1, 4; setp.ne.u32 p2, t, 0;\n and.b32 t, %1, 8; setp.ne.u32 p3, t, 0;\n"                \
      " and.b32 t, %1, 16; setp.ne.u32 p4, t, 0;\n and.b32 t, %1, 32; setp.ne.u32 p5, t, 0;\n"              \
      " add.u32 w0, %1, 1; add.u32 w1, %1, 2; add.u32 w2, %1, 3; add.u32 w3, %1, 4;\n"                      \
      " add.u32 w4, %1, 5; add.u32 w5, %1, 6; add.u32 w6, %1, 7; add.u32 w7, %1, 8;\n"                      \
      " mov.u32 n, %2;\n setp.lt.s32 q, n, 1;\n @q bra PL_END;\n"                                           \
      "PL_TOP:\n .pragma \"nounroll\";\n"                                                                    \
      " xor.pred p0, p0, p1; xor.pred p1, p1, p2; xor.pred p2, p2, p3;\n"                                   \
      " xor.pred p3, p3, p4; xor.pred p4, p4, p5; xor.pred p5, p5, p0;\n"                                   \
      " xor.pred p0, p0, p1; xor.pred p1, p1, p2; xor.pred p2, p2, p3;\n"                                   \
      " xor.pred p3, p3, p4; xor.pred p4, p4, p5; xor.pred p5, p5, p0;\n" PARTNER                           \
      " sub.s32 n, n, 1;\n setp.gt.s32 q, n, 0;\n @q bra PL_TOP;\n"                                         \
      "PL_END:\n"                                                                                            \
      " selp.u32 t, 1, 0, p0; xor.b32 %0, w0, t;\n selp.u32 t, 2, 0, p1; xor.b32 %0, %0, t;\n"              \
      " selp.u32 t, 4, 0, p2; xor.b32 %0, %0, t;\n selp.u32 t, 8, 0, p3; xor.b32 %0, %0, t;\n"              \
      " selp.u32 t, 16, 0, p4; xor.b32 %0, %0, t;\n selp.u32 t, 32, 0, p5; xor.b32 %0, %0, t;\n"            \
      " xor.b32 %0, %0, w1; xor.b32 %0, %0, w2; xor.b32 %0, %0, w3; xor.b32 %0, %0, w4;\n"                  \
      " xor.b32 %0, %0, w5; xor.b32 %0, %0, w6; xor.b32 %0, %0, w7;\n}\n"                                   \
      : "=r"(s)                                                                                              \
      : "r"(threadIdx.x ^ (uint32_t)a.mode), "r"(a.iters), "r"(0x3C013C01u ^ (uint32_t)a.mode),             \
        "r"(0x00010001u ^ (uint32_t)(a.mode << 3)))
#define PLOP3_LOP3                                                                                           \
  " lop3.b32 w0, w0, %3, %4, 0x96; lop3.b32 w1, w1, %3, %4, 0x96; lop3.b32 w2, w2, %3, %4, 0x96;\n"         \
  " lop3.b32 w3, w3, %3, %4, 0x96; lop3.b32 w4, w4, %3, %4, 0x96; lop3.b32 w5, w5, %3, %4, 0x96;\n"         \
  " lop3.b32 w6, w6, %3, %4, 0x96; lop3.b32 w7, w7, %3, %4, 0x96;\n"

template <int P, int Q>
__device__ __forceinline__ void bar_or_chains(const OpArgs& a) {
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

// A conversion chain on N of 8 chains, its state of the functor's type T: a conversion and its inverse (or one
// conversion) a step.
__device__ __forceinline__ uint32_t fold_of(uint32_t x) { return x; }
__device__ __forceinline__ uint32_t fold_of(unsigned long long x) { return (uint32_t)x ^ (uint32_t)(x >> 32); }
__device__ __forceinline__ uint32_t fold_of(double x) { return fold_of((unsigned long long)__double_as_longlong(x)); }
template <typename T>
__device__ __forceinline__ T seed_of(uint32_t s) {
  if constexpr (sizeof(T) == 8 && T(0.5) != T(0)) return (double)(s & 0xffffu) + 0.25;
  else if constexpr (sizeof(T) == 8) return (unsigned long long)s << 7;
  else return s;
}
#define CVT(NAME, T, ...)                                                          \
  struct NAME {                                                                  \
    using type = T;                                                              \
    __device__ __forceinline__ T operator()(T x) const { __VA_ARGS__ return x; } \
  };
CVT(CvtS32, uint32_t, float f; asm volatile("cvt.rp.f32.s32 %0, %1;" : "=f"(f) : "r"(x));
    asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(x) : "f"(f));)
CVT(CvtU16, uint32_t, float f; asm volatile("cvt.rn.f32.u16 %0, %1;" : "=f"(f) : "r"(x));
    asm volatile("cvt.rzi.u16.f32 %0, %1;" : "=r"(x) : "f"(f));)
CVT(CvtS8, uint32_t, float f; asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(f) : "r"(x));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x) : "f"(f));)
CVT(CvtFrnd, uint32_t, float f = __uint_as_float(x); asm volatile("cvt.rzi.f32.f32 %0, %0;" : "+f"(f));
    asm volatile("cvt.rni.f32.f32 %0, %0;" : "+f"(f)); x = __float_as_uint(f);)
CVT(CvtI2i, uint32_t, asm volatile("cvt.sat.s8.s32 %0, %0;" : "+r"(x));)
CVT(CvtI2ip, uint32_t, asm volatile("cvt.pack.sat.s8.s32.b32 %0, %0, %0, %0;" : "+r"(x));)
CVT(CvtU64, unsigned long long, float f; asm volatile("cvt.rn.f32.u64 %0, %1;" : "=f"(f) : "l"(x));
    asm volatile("cvt.rzi.u64.f32 %0, %1;" : "=l"(x) : "f"(f));)
CVT(CvtFrnd64, double, asm volatile("cvt.rmi.f64.f64 %0, %0;" : "+d"(x)); asm volatile("cvt.rzi.f64.f64 %0, %0;" : "+d"(x));)
CVT(CvtF64U32, uint32_t, double d; asm volatile("cvt.rn.f64.u32 %0, %1;" : "=d"(d) : "r"(x));
    asm volatile("cvt.rzi.u32.f64 %0, %1;" : "=r"(x) : "d"(d));)
CVT(CvtF64S64, unsigned long long, double d; asm volatile("cvt.rn.f64.s64 %0, %1;" : "=d"(d) : "l"(x));
    asm volatile("cvt.rzi.s64.f64 %0, %1;" : "=l"(x) : "d"(d));)

template <typename Op, int P, int Q, int N>
__device__ __forceinline__ void cvt_chains(const OpArgs& a) {
  using T = typename Op::type;
  T v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = seed_of<T>((unsigned)a.iters * 977u + threadIdx.x * 8u + (unsigned)u);
  Partner<P, Q> pt(a);
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  const Op op;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      if (u < N) v[u] = op(v[u]);
      pt.step(u, k1, k2);
    }
  }
  OP_END
  uint32_t s = pt.out();
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= fold_of(v[u]);
  OP_SINK(s)
}

// f32 <-> s64 conversions on N of 8 chains.
template <int P, int Q, int N>
__device__ __forceinline__ void s64_chains(const OpArgs& a) {
  long long v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = ((long long)a.iters << 20) + threadIdx.x * 8 + u - 4096;
  Partner<P, Q> pt(a);
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      if (u < N) {
        float f;
        asm volatile("cvt.rn.f32.s64 %0, %1;" : "=f"(f) : "l"(v[u]));
        asm volatile("cvt.rzi.s64.f32 %0, %1;" : "=l"(v[u]) : "f"(f));
      }
      pt.step(u, k1, k2);
    }
  }
  OP_END
  uint32_t s = pt.out();
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= (uint32_t)v[u] ^ (uint32_t)(v[u] >> 32);
  OP_SINK(s)
}

// BAR.RED.OR on named barrier 1 in warps 0-15, 8 a trip in one nounroll PTX loop (the predicate stays a predicate), and
// with Q > 0 Q LOP3 a trip in warps 16-31 for as long as they run (OP_UNTIL): the barrier beside a full ALU stream.
template <int Q>
__device__ __forceinline__ void bar_split(const OpArgs& a) {
  uint32_t s = 0;
  Partner<1, Q> pt(a);
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  OP_BEGIN_STOP
  if ((threadIdx.x >> 5) < 16) {
    asm volatile(
        "{\n .reg .pred p, q;\n .reg .u32 n;\n and.b32 n, %1, 1;\n setp.ne.u32 p, n, 0;\n"
        " mov.u32 n, %2;\n setp.lt.s32 q, n, 1;\n @q bra BS_END;\n"
        "BS_TOP:\n .pragma \"nounroll\";\n"
        " bar.red.or.pred p, 1, 512, p;\n bar.red.or.pred p, 1, 512, !p;\n bar.red.or.pred p, 1, 512, p;\n"
        " bar.red.or.pred p, 1, 512, !p;\n bar.red.or.pred p, 1, 512, p;\n bar.red.or.pred p, 1, 512, !p;\n"
        " bar.red.or.pred p, 1, 512, p;\n bar.red.or.pred p, 1, 512, !p;\n"
        " sub.s32 n, n, 1;\n setp.gt.s32 q, n, 0;\n @q bra BS_TOP;\n"
        "BS_END:\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(s)
        : "r"(threadIdx.x ^ (uint32_t)a.mode), "r"(a.iters));
    OP_DONE
  } else if constexpr (Q > 0) {
    OP_UNTIL(n) {
#pragma unroll
      for (int u = 0; u < 8; u++) pt.step(u, k1, k2);
    }
    OP_TRIPS(n)
  }
  OP_END_STOP
  OP_SINK(s ^ pt.out())
}

#define KBITS (13u + ((uint32_t)a.mode & 7u))
#define KHALF (__float_as_uint(0.75f + 1e-3f * (float)a.mode))
OP_KERNEL(lop3, TPB) { partner_only<1, 8>(a); }
OP_KERNEL(mufu, TPB) { partner_only<2, 8>(a); }
OP_KERNEL(sgxt, TPB) { chains<Sgxt, 0, 0>(a, 0u, KBITS); }
OP_KERNEL(mix_sgxt_lop3, TPB) { chains<Sgxt, 1, 8>(a, 0u, KBITS); }
OP_KERNEL(bmsk, TPB) { chains<Bmsk, 0, 0>(a, 0u, KBITS); }
OP_KERNEL(mix_bmsk_lop3, TPB) { chains<Bmsk, 1, 8>(a, 0u, KBITS); }
OP_KERNEL(plop3, TPB) {
  uint32_t s;
  OP_BEGIN
  PLOP3_LOOP("");
  OP_END
  OP_SINK(s)
}
OP_KERNEL(mix_plop3_lop3, TPB) {
  uint32_t s;
  OP_BEGIN
  PLOP3_LOOP(PLOP3_LOP3);
  OP_END
  OP_SINK(s)
}
OP_KERNEL(iadd64, TPB) { iadd64_chains<0, 0>(a); }
OP_KERNEL(mix_iadd64_lop3, TPB) { iadd64_chains<1, 8>(a); }
OP_KERNEL(iabs, TPB) { chains<Iabs, 0, 0>(a, 0x80000000u, 0u); }
OP_KERNEL(mix_iabs_lop3, TPB) { chains<Iabs, 1, 8>(a, 0x80000000u, 0u); }
OP_KERNEL(i2f_f2i, TPB) { chains<I2fF2i, 0, 0>(a, 0u, 0u); }
OP_KERNEL(mix_i2f_f2i_mufu, TPB) { chains<I2fF2i, 2, 8>(a, 0u, 0u); }
OP_KERNEL(i2f16_f2i16, TPB) { i2f16<0, 0>(a); }
OP_KERNEL(mix_i2f16_f2i16_mufu, TPB) { i2f16<2, 8>(a); }
OP_KERNEL(f2f64, TPB) { chains<F2f64, 0, 0>(a, 0x3F800000u, 0u); }
OP_KERNEL(frnd, TPB) { chains<Frnd, 0, 0>(a, 0x3F800000u, KHALF); }
OP_KERNEL(mix_frnd_mufu, TPB) { chains<Frnd, 2, 8>(a, 0x3F800000u, KHALF); }
OP_KERNEL(bar_or, TPB) { bar_or_chains<0, 0>(a); }
OP_KERNEL(mix_bar_or_lop3, TPB) { bar_or_chains<1, 16>(a); }
OP_KERNEL(dfma, TPB) { partner_only<3, 8>(a); }
OP_KERNEL(mix_f2f64_mufu24, TPB) { chains<F2f64, 2, 24, 1>(a, 0x3F800000u, 0u); }
OP_KERNEL(mix_f2f64_dfma, TPB) { chains<F2f64, 3, 2, 2>(a, 0x3F800000u, 0u); }
OP_KERNEL(i2f64, TPB) { chains<I2fF64, 0, 0>(a, 0u, 0u); }
OP_KERNEL(mix_i2f64_mufu24, TPB) { chains<I2fF64, 2, 24, 1>(a, 0u, 0u); }
OP_KERNEL(mix_i2f64_dfma, TPB) { chains<I2fF64, 3, 2, 2>(a, 0u, 0u); }
OP_KERNEL(s64, TPB) { s64_chains<0, 0, 8>(a); }
OP_KERNEL(mix_s64_mufu24, TPB) { s64_chains<2, 24, 1>(a); }
OP_KERNEL(mix_s64_dfma, TPB) { s64_chains<3, 2, 2>(a); }
#define CVT32(n, OP)                                                     \
  OP_KERNEL(n, TPB) { cvt_chains<OP, 0, 0, 8>(a); }                     \
  OP_KERNEL(mix_##n##_mufu, TPB) { cvt_chains<OP, 2, 8, 8>(a); }
#define CVT64(n, OP)                                                     \
  OP_KERNEL(n, TPB) { cvt_chains<OP, 0, 0, 8>(a); }                     \
  OP_KERNEL(mix_##n##_mufu24, TPB) { cvt_chains<OP, 2, 24, 1>(a); }     \
  OP_KERNEL(mix_##n##_dfma, TPB) { cvt_chains<OP, 3, 2, 2>(a); }
CVT32(cvt_s32, CvtS32)
CVT32(cvt_u16, CvtU16)
CVT32(cvt_s8, CvtS8)
CVT32(cvt_frnd, CvtFrnd)
CVT32(cvt_i2i, CvtI2i)
CVT32(cvt_i2ip, CvtI2ip)
CVT64(cvt_u64, CvtU64)
CVT64(cvt_frnd64, CvtFrnd64)
CVT64(cvt_f64_u32, CvtF64U32)
CVT64(cvt_f64_s64, CvtF64S64)
OP_KERNEL(bar16, TPB) { bar_split<0>(a); }
OP_KERNEL(mix_bar16_lop3, TPB) { bar_split<32>(a); }

#define PR(N, K, E) {.name = N, .kernel = (const void*)K, .tpb = TPB, .expect = E}
static const OpProbe PROBES[] = {
    PR("lop3", lop3, "LOP3.LUT=8"),
    PR("mufu", mufu, "MUFU.EX2=8"),
    PR("sgxt", sgxt, "SGXT=8"),
    PR("mix_sgxt_lop3", mix_sgxt_lop3, "SGXT=8,LOP3.LUT=8"),
    PR("bmsk", bmsk, "BMSK=8"),
    PR("mix_bmsk_lop3", mix_bmsk_lop3, "BMSK=8,LOP3.LUT=8"),
    PR("plop3", plop3, "PLOP3.LUT=7"),
    PR("mix_plop3_lop3", mix_plop3_lop3, "PLOP3.LUT=7,LOP3.LUT=8"),
    PR("iadd64", iadd64, "IADD.64=8"),
    PR("mix_iadd64_lop3", mix_iadd64_lop3, "IADD.64=8,LOP3.LUT=8"),
    PR("iabs", iabs, "IABS=8"),
    PR("mix_iabs_lop3", mix_iabs_lop3, "IABS=8,LOP3.LUT=8"),
    PR("i2f_f2i", i2f_f2i, "I2F.U32.RP=8,F2I.FTZ.U32.TRUNC.NTZ=8"),
    PR("mix_i2f_f2i_mufu", mix_i2f_f2i_mufu, "I2F.U32.RP=8,F2I.FTZ.U32.TRUNC.NTZ=8,MUFU.EX2=8"),
    PR("i2f16_f2i16", i2f16_f2i16, "I2F.S16=8,F2I.S16.TRUNC.NTZ=8"),
    PR("mix_i2f16_f2i16_mufu", mix_i2f16_f2i16_mufu, "I2F.S16=8,F2I.S16.TRUNC.NTZ=8,MUFU.EX2=8"),
    PR("f2f64", f2f64, "F2F.F64.F32=8,F2F.F32.F64=8"),
    PR("frnd", frnd, "FRND.FLOOR=8,FADD=8"),
    PR("mix_frnd_mufu", mix_frnd_mufu, "FRND.FLOOR=8,FADD=8,MUFU.EX2=8"),
    PR("bar_or", bar_or, "BAR.RED.OR*=8"),
    PR("mix_bar_or_lop3", mix_bar_or_lop3, "BAR.RED.OR*=8,LOP3.LUT=16"),
    PR("dfma", dfma, "DFMA=8"),
    PR("mix_f2f64_mufu24", mix_f2f64_mufu24, "F2F.F64.F32=1,F2F.F32.F64=1,MUFU.EX2=24"),
    PR("mix_f2f64_dfma", mix_f2f64_dfma, "F2F.F64.F32=2,F2F.F32.F64=2,DFMA=2"),
    PR("i2f64", i2f64, "I2F.F64=8,F2I.F64.TRUNC=8"),
    PR("mix_i2f64_mufu24", mix_i2f64_mufu24, "I2F.F64=1,F2I.F64.TRUNC=1,MUFU.EX2=24"),
    PR("mix_i2f64_dfma", mix_i2f64_dfma, "I2F.F64=2,F2I.F64.TRUNC=2,DFMA=2"),
    PR("s64", s64, "I2F.S64=8,F2I.S64.TRUNC=8"),
    PR("mix_s64_mufu24", mix_s64_mufu24, "I2F.S64=1,F2I.S64.TRUNC=1,MUFU.EX2=24"),
    PR("mix_s64_dfma", mix_s64_dfma, "I2F.S64=2,F2I.S64.TRUNC=2,DFMA=2"),
#define PR32(n, A, B) PR(#n, n, A "=8" B), PR("mix_" #n "_mufu", mix_##n##_mufu, A "=8" B ",MUFU.EX2=8")
#define PR64(n, A, B)                                                              \
  PR(#n, n, A "=8," B "=8"), PR("mix_" #n "_mufu24", mix_##n##_mufu24, A "=1," B "=1,MUFU.EX2=24"), \
      PR("mix_" #n "_dfma", mix_##n##_dfma, A "=2," B "=2,DFMA=2")
    PR32(cvt_s32, "I2F.RP", ",F2I.TRUNC.NTZ=8"),
    PR32(cvt_u16, "I2F.U16", ",F2I.U16.TRUNC.NTZ=8"),
    PR32(cvt_s8, "I2F.S8", ",F2I.S8.NTZ=8"),
    PR32(cvt_frnd, "FRND.TRUNC", ",FRND=8"),
    PR32(cvt_i2i, "I2I.S8.S32.SAT", ""),
    PR32(cvt_i2ip, "I2IP.S8.S32.SAT", ""),
    PR64(cvt_u64, "I2F.U64", "F2I.U64.TRUNC"),
    PR64(cvt_frnd64, "FRND.F64.FLOOR", "FRND.F64.TRUNC"),
    PR64(cvt_f64_u32, "I2F.F64.U32", "F2I.U32.F64.TRUNC"),
    PR64(cvt_f64_s64, "I2F.F64.S64", "F2I.S64.F64.TRUNC"),
    {.name = "bar16", .kernel = (const void*)bar16, .tpb = TPB, .expect = "BAR.RED.OR*=8", .warps = "16"},
    {.name = "mix_bar16_lop3", .kernel = (const void*)mix_bar16_lop3, .tpb = TPB,
     .expect = "BAR.RED.OR*=8;LOP3.LUT=32", .warps = "16;16", .until = 2},
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
