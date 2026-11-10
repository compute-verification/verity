// The lane and mma.sync chains unpriced.cu (sm_100a) and unpriced90.cu (sm_90a) time: each op on 8 independent chains a
// thread, alone (s_alone) or alternating with another op (s_pair). f32x2 (FFMA2, FMUL2) is sm_100+; instantiate it there.
#pragma once
#include "../opcode_probe/probe.cuh"

#define UP_DEV __device__ __forceinline__

enum { U_FFMA, U_FFMA2, U_FFMA2C, U_FMUL, U_FMUL2, U_DMMA, U_IMMA, U_LOP3, U_IMAD };

struct Chains {
  uint32_t l[8];             // LOP3, IMAD: the integer lanes the pairs interleave
  float f[8];
  unsigned long long p[8];   // f32x2: two FP32 values a register pair
  double d[8][2];            // DMMA m8n8k4: a thread's two f64 accumulators a chain
  uint32_t r[8][4];          // IMMA m16n8k32: a thread's four s32 accumulators a chain
  unsigned long long one2;   // {1.0f, 1.0f}: FFMA2C's addend, FFMA's 1.0 as a pair
  double da, db;
  uint32_t ia[4], ib[2];
  UP_DEV explicit Chains(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t h = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
      l[i] = h;
      f[i] = 1.0f + (float)(h & 0xffu) * 0x1p-12f;
      const float g = 1.0f + (float)((h >> 8) & 0xffu) * 0x1p-12f;
      asm("mov.b64 %0, {%1, %2};" : "=l"(p[i]) : "f"(f[i]), "f"(g));
      d[i][0] = (double)(h & 0xfffu);
      d[i][1] = (double)(h >> 20);
#pragma unroll
      for (int j = 0; j < 4; j++) r[i][j] = h * (j + 1);
    }
    asm("mov.b64 %0, {%1, %1};" : "=l"(one2) : "f"(1.0f));
    da = (double)(int8_t)(tid * 37u);
    db = (double)(int8_t)(tid * 91u);
#pragma unroll
    for (int j = 0; j < 4; j++) ia[j] = tid * 0x01010101u * (2 * j + 3);
    ib[0] = tid * 0x02040801u;
    ib[1] = tid * 0x80402011u;
  }
  // Chain s reads chain t's previous value (as dc/probes/lanes.cuh), so ptxas can neither fold nor hoist it.
  template <int X>
  UP_DEV void op(int i) {
    const int s = i & 7, t = (i + 1) & 7;
    if constexpr (X == U_FFMA) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == U_FFMA2) asm volatile("fma.rn.f32x2 %0,%0,%1,%2;" : "+l"(p[s]) : "l"(p[t]), "l"(p[(i + 2) & 7]));
    if constexpr (X == U_FFMA2C) asm volatile("fma.rn.f32x2 %0,%0,%1,%2;" : "+l"(p[s]) : "l"(p[t]), "l"(one2));
    if constexpr (X == U_FMUL) asm volatile("mul.rn.f32 %0,%0,%1;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == U_FMUL2) asm volatile("mul.rn.f32x2 %0,%0,%1;" : "+l"(p[s]) : "l"(p[t]));
    if constexpr (X == U_DMMA)
      asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                   : "+d"(d[s][0]), "+d"(d[s][1]) : "d"(da), "d"(db));
    if constexpr (X == U_IMMA)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+r"(r[s][0]), "+r"(r[s][1]), "+r"(r[s][2]), "+r"(r[s][3])
                   : "r"(ia[0]), "r"(ia[1]), "r"(ia[2]), "r"(ia[3]), "r"(ib[0]), "r"(ib[1]));
    if constexpr (X == U_LOP3) asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(l[s]) : "r"(l[t]), "r"(0x85EBCA6Bu));
    if constexpr (X == U_IMAD) asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(l[s]) : "r"(l[t]), "r"(0x9E3779B9u));
  }
  // Only the timed op's chains, so the others are dead and the kernel fits 64 registers a thread.
  template <int X>
  UP_DEV uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (X == U_FFMA || X == U_FMUL) x += __float_as_uint(f[i]);
      if constexpr (X == U_FFMA2 || X == U_FFMA2C || X == U_FMUL2) x += (uint32_t)p[i] ^ (uint32_t)(p[i] >> 32);
      if constexpr (X == U_DMMA) x += (uint32_t)__double_as_longlong(d[i][0]) ^ (uint32_t)__double_as_longlong(d[i][1]);
      if constexpr (X == U_IMMA) x += r[i][0] ^ r[i][1] ^ r[i][2] ^ r[i][3];
      if constexpr (X == U_LOP3 || X == U_IMAD) x += l[i];
    }
    return x;
  }
};

template <int X, int N>
UP_DEV void s_alone(const OpArgs& a) {
  Chains c(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < N; q++) c.op<X>(q);
  }
  OP_END
  OP_SINK(c.template fold<X>() ^ threadIdx.x)
}
// N ops alternating X and Y on independent chains (lanes.cuh's lane_pair): with FFMA2 beside LOP3 or IMAD, a sum of the
// two issue rates near 4 a SM-clock means FFMA2 takes one issue slot (and its rate alone is its FP32 pipe's), near 8/3
// that it takes two.
template <int X, int Y, int N>
UP_DEV void s_pair(const OpArgs& a) {
  Chains c(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < N; q += 2) {
      c.op<X>(q >> 1);
      c.op<Y>((q >> 1) + 4);
    }
  }
  OP_END
  OP_SINK((c.template fold<X>() + c.template fold<Y>()) ^ threadIdx.x)
}

UP_DEV uint32_t warp_u() { return __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0); }
