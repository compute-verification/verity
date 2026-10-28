// sm_89 (L40S) uniform-datapath and IMAD-form probes: what a program
// gains by moving the fold's work onto a unit the honest step leaves idle. Built as facet_break's uniform_up.cu is
// (16 warps, 4 a SMSP, one block a SM; the chains loop-carried from a parameter into the sink), with the honest step at
// L = 16 (8 IMMA.16816 and duo on each accumulator). ptxas 13.0.88 keeps a warp-uniform chain on the uniform datapath
// only in a loop with MMAs, so "alone" keeps the MMAs with their results unread (ptxas deletes them after placing the
// chain). The gate pins every loop opcode, vector and uniform alike, so a chain that lands on lanes refuses.
//   u<F>_<K><S>   F: z alone (the MMAs unread); t 8 IMMA.16816 a step, accumulators carried to the sink (the tensor
//                 pipe saturated); h honest16 (the MMAs and duo on each accumulator: the ALU pipe binds); g h and 64
//                 FFMA on the accumulators' bits (facet_break's issue-nearest setting)
//                 S, the uniform step on each of four chain pairs: x ULOP3; a UIADD3 + ULOP3; s USHF + ULOP3;
//                 p UPRMT + ULOP3; m UIMAD + ULOP3
//   i<F>_<op>[_N] the IMAD forms of the fold's ALU work: op imad (mad.lo, a register multiplier), hi (mad.hi with its
//                 addend: IMAD.HI.U32 and the IMAD.MOV.U32 that place its addend), shl (mul.lo by 2^5: ptxas folds the
//                 chain's two into one shift by 10, IMAD.SHL.U32 or SHF.L.U32); F a (alone, 16 on 8 chains) or h (N
//                 beside honest16, reading the accumulators)
//   duo<W>_w<V>   honest16 with W of a lane's 32 words folded by duo's IMAD form (z = rotl(u, r) + y as
//                 umulhi(u, 2^r) + y then u * 2^r + that: ALU 1 + IMAD 3 a word for duo's ALU 2 + IMAD 1), V warps a
//                 SMSP: the same receipts at a different class mix
#include "../probe.cuh"
#include "../probes/honest.cuh"

constexpr int TPB = 512, ITERS = 2048;

enum { S_X, S_A, S_S, S_P, S_M };

template <int K, int S>
struct UChains {
  uint32_t u[8];
  __device__ explicit UChains(uint32_t seed) {
#pragma unroll
    for (int c = 0; c < 8; c++) u[c] = seed * (2u * c + 3u) + (uint32_t)c;
  }
  __device__ __forceinline__ void step() {
#pragma unroll
    for (int i = 0; i < K / 8; i++)
#pragma unroll
      for (int p = 0; p < 8; p += 2) {
        const uint32_t k = 0x7F4A7C15u + (uint32_t)(i * 8 + p) * 0x9E3779B9u;
        if constexpr (S == S_X) {
          u[p] = (u[p] & u[p + 1]) ^ k;
          u[p + 1] = (u[p + 1] | u[p]) ^ (k >> 3);
        }
        if constexpr (S == S_A) {
          u[p] = u[p] + u[p + 1] + k;
          u[p + 1] = u[p + 1] ^ u[p] ^ (k >> 3);
        }
        if constexpr (S == S_S) {
          u[p] = (u[p] >> 3) ^ u[p + 1];
          u[p + 1] = u[p + 1] ^ (u[p] << 5) ^ k;
        }
        if constexpr (S == S_P) {
          u[p] = __byte_perm(u[p], u[p + 1], 0x7531) ^ k;
          u[p + 1] = __byte_perm(u[p + 1], u[p], 0x6420);
        }
        if constexpr (S == S_M) {
          u[p] = u[p] * 0x9E3779B1u + u[p + 1];
          u[p + 1] = u[p + 1] ^ u[p] ^ k;
        }
      }
  }
  __device__ uint32_t fold() const { return u[0] ^ u[1] ^ u[2] ^ u[3] ^ u[4] ^ u[5] ^ u[6] ^ u[7]; }
};

enum { F_Z, F_T, F_H, F_G };

// facet_break's issue-nearest setting: 64 FFMA a warp-iteration on the honest step's accumulator bits (on sm_89
// FFMA beside the fold's ALU work, not issue, holds it: 252 SMSP-cycles a warp-iteration for 171 instructions)
struct Feed {
  float f[8], g;
  __device__ explicit Feed(int mode) {
    g = __uint_as_float(0x3F800001u ^ (uint32_t)mode);
#pragma unroll
    for (int c = 0; c < 8; c++) f[c] = 0.f;
  }
  __device__ __forceinline__ void step(const Honest& h) {
#pragma unroll
    for (int q = 0; q < 64; q++) f[q & 7] = __fmaf_rn(__int_as_float(h.d[q & 7][(q >> 3) & 3]), g, f[q & 7]);
  }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 8; c++) s ^= __float_as_uint(f[c]);
    return s;
  }
};

__device__ __forceinline__ uint32_t acc_fold(const Honest& h) {
  uint32_t s = 0;
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)(h.d[c][0] ^ h.d[c][1] ^ h.d[c][2] ^ h.d[c][3]);
  return s;
}

template <int F, int S, int K>
__device__ __forceinline__ void uni(const OpArgs& a) {
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  Honest h(a.mode);
  Feed e(a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    if constexpr (F == F_H || F == F_G) h.step<8, true, true>();
    else h.step<8, false, true>();
    if constexpr (F == F_G) e.step(h);
    u.step();
  }
  OP_END
  OP_SINK(u.fold() ^ h.fold() ^ (F == F_T ? acc_fold(h) : 0u) ^ (F == F_G ? e.fold() : 0u))
}

enum { O_IMAD, O_HI, O_SHL };

// a.mode is 0 at run time: through it every multiplier stays a register, so ptxas strength-reduces none of them
struct IOps {
  uint32_t v[8], m, c;
  __device__ explicit IOps(const OpArgs& a) {
    m = 0x20u ^ (uint32_t)a.mode;
    c = 0x3C013C01u ^ (uint32_t)(a.mode << 3);
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = threadIdx.x * 8u + i + 1u;
  }
  template <int O>
  __device__ __forceinline__ void op(int s, uint32_t x) {
    if constexpr (O == O_IMAD) asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(v[s]) : "r"(x), "r"(m));
    if constexpr (O == O_HI) asm volatile("mad.hi.u32 %0, %1, %2, %0;" : "+r"(v[s]) : "r"(x), "r"(m));
    if constexpr (O == O_SHL) asm volatile("mul.lo.u32 %0, %0, 32;" : "+r"(v[s]));
  }
  __device__ uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x ^= v[i];
    return x;
  }
};

template <int O>
__device__ __forceinline__ void imad_alone(const OpArgs& a) {
  IOps o(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < 16; q++) o.op<O>(q & 7, o.v[(q + 1) & 7] ^ o.c);
  }
  OP_END
  OP_SINK(o.fold())
}

template <int O, int N>
__device__ __forceinline__ void imad_honest(const OpArgs& a) {
  IOps o(a);
  Honest h(a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    h.step<8, true, true>();
#pragma unroll
    for (int q = 0; q < N; q++) o.op<O>(q & 7, (uint32_t)h.d[q & 7][(q >> 3) & 3]);
  }
  OP_END
  OP_SINK(o.fold() ^ h.fold())
}

// duo's word with z = rotl(u, r) + y on the IMAD pipe: umulhi(u, 2^r) + y = (u >> (32 - r)) + y, then u * 2^r + that
__device__ __forceinline__ void duo_word_imad(uint32_t& y, uint32_t& z, uint32_t d, uint32_t k, uint32_t a,
                                              uint32_t pr) {
  const uint32_t u = y ^ d ^ k;
  y = u * a + z;
  uint32_t t;
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(t) : "r"(u), "r"(pr), "r"(y));
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(z) : "r"(u), "r"(pr), "r"(t));
}

// honest16 with words (word w of accumulator c, w * 8 + c < W) folded in duo's IMAD form: the same (y, z) as
// rf_duo_step
template <int W>
__device__ __forceinline__ void duo_mixed(const OpArgs& a) {
  Honest h(a.mode);
  uint32_t pr[4];
  const int rot[4] = {5, 11, 17, 29};
#pragma unroll
  for (int w = 0; w < 4; w++) pr[w] = (1u << rot[w]) ^ (uint32_t)a.mode;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) mma_imma16(h.d[c], h.x, h.y);
#pragma unroll
    for (int c = 0; c < 8; c++)
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const uint32_t k = (w & 1) ? h.k1 : h.k0, m = (w & 1) ? h.a1 : h.a0;
        if (w * 8 + c < W) duo_word_imad(h.ys[c], h.zs[c], (uint32_t)h.d[c][w], k, m, pr[w]);
        else rf_duo_word(h.ys[c], h.zs[c], (uint32_t)h.d[c][w], k, m, rot[w]);
      }
  }
  OP_END
  OP_SINK(h.fold())
}

#define UK(nm, F, S, K) \
  OP_KERNEL(nm, TPB) { uni<F, S, K>(a); }
UK(uz_64x, F_Z, S_X, 64)
UK(uz_64a, F_Z, S_A, 64)
UK(uz_64s, F_Z, S_S, 64)
UK(uz_64p, F_Z, S_P, 64)
UK(uz_64m, F_Z, S_M, 64)
UK(ut_0, F_T, S_X, 0)
UK(ut_32x, F_T, S_X, 32)
UK(ut_64x, F_T, S_X, 64)
UK(ut_128x, F_T, S_X, 128)
UK(ut_64a, F_T, S_A, 64)
UK(ut_64s, F_T, S_S, 64)
UK(ut_64p, F_T, S_P, 64)
UK(ut_64m, F_T, S_M, 64)
UK(ut_128m, F_T, S_M, 128)
UK(uh_0, F_H, S_X, 0)
UK(uh_32x, F_H, S_X, 32)
UK(uh_64x, F_H, S_X, 64)
UK(uh_64a, F_H, S_A, 64)
UK(uh_64s, F_H, S_S, 64)
UK(uh_64p, F_H, S_P, 64)
UK(uh_64m, F_H, S_M, 64)
UK(uh_128m, F_H, S_M, 128)
UK(ug_0, F_G, S_X, 0)
UK(ug_32x, F_G, S_X, 32)
UK(ug_64a, F_G, S_A, 64)
UK(ug_64m, F_G, S_M, 64)

#define IA(nm, O) \
  OP_KERNEL(nm, TPB) { imad_alone<O>(a); }
#define IH(nm, O, N) \
  OP_KERNEL(nm, TPB) { imad_honest<O, N>(a); }
IA(ia_imad, O_IMAD)
IA(ia_hi, O_HI)
IA(ia_shl, O_SHL)
IH(ih_imad_32, O_IMAD, 32)
IH(ih_imad_64, O_IMAD, 64)
IH(ih_hi_32, O_HI, 32)
IH(ih_hi_64, O_HI, 64)

#define DUO(W, V) \
  OP_KERNEL(duo##W##_w##V, V * 128) { duo_mixed<W>(a); }
DUO(0, 4)
DUO(8, 4)
DUO(12, 4)
DUO(16, 4)
DUO(32, 4)
DUO(0, 8)
DUO(8, 8)
DUO(12, 8)
DUO(16, 8)
DUO(32, 8)

#define P_(nm, threads, EX, CNT) \
  {.name = #nm, .kernel = (const void*)nm, .tpb = threads, .expect = EX, .counted = CNT, .iters = ITERS}
#define P_U(nm, EX, CNT) P_(nm, TPB, EX, CNT)

static const OpProbe PROBES[] = {
    P_U(uz_64x, "ULOP3.LUT=96,IADD3=1,ISETP.GE.AND=1", "ULOP3.LUT"),
    P_U(uz_64a, "UIADD3=32,ULOP3.LUT=32,IADD3=1,ISETP.GE.AND=1", "ULOP3.LUT"),
    P_U(uz_64s, "ULOP3.LUT=64,USHF.L.U32=32,USHF.R.U32.HI=32,IADD3=1,ISETP.GE.AND=1", "ULOP3.LUT"),
    P_U(uz_64p, "UPRMT=64,ULOP3.LUT=32,IADD3=1,ISETP.GE.AND=1", "ULOP3.LUT"),
    P_U(uz_64m, "UIMAD=32,ULOP3.LUT=32,IADD3=1,ISETP.GE.AND=1", "ULOP3.LUT"),
    P_U(ut_0, "IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ut_32x, "ULOP3.LUT=48,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ut_64x, "ULOP3.LUT=96,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ut_128x, "ULOP3.LUT=192,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ut_64a, "UIADD3=32,ULOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ut_64s, "ULOP3.LUT=64,USHF.L.U32=32,USHF.R.U32.HI=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ut_64p, "UPRMT=64,ULOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ut_64m, "UIMAD=32,ULOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ut_128m, "UIMAD=64,ULOP3.LUT=64,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(uh_0, "IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(uh_32x, "ULOP3.LUT=48,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(uh_64x, "ULOP3.LUT=96,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(uh_64a, "IMAD=32,LEA.HI=32,LOP3.LUT=32,UIADD3=32,ULOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(uh_64s, "ULOP3.LUT=64,IMAD=32,LEA.HI=32,LOP3.LUT=32,USHF.L.U32=32,USHF.R.U32.HI=32,IMMA.16816.S8.S8=8,IADD3=1", "IMMA.16816.S8.S8"),
    P_U(uh_64p, "UPRMT=64,IMAD=32,LEA.HI=32,LOP3.LUT=32,ULOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(uh_64m, "IMAD=32,LEA.HI=32,LOP3.LUT=32,UIMAD=32,ULOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(uh_128m, "UIMAD=64,ULOP3.LUT=64,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ug_0, "FFMA=64,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ug_32x, "FFMA=64,ULOP3.LUT=48,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ug_64a, "FFMA=64,IMAD=32,LEA.HI=32,LOP3.LUT=32,UIADD3=32,ULOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1", "IMMA.16816.S8.S8"),
    P_U(ug_64m, "FFMA=64,IMAD=32,LEA.HI=32,LOP3.LUT=32,UIMAD=32,ULOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1", "IMMA.16816.S8.S8"),
    P_U(ia_imad, "IMAD=16,LOP3.LUT=16,IADD3=1,ISETP.GE.AND=1", "IMAD"),
    P_U(ia_hi, "IMAD.HI.U32=16,LOP3.LUT=16,IMAD.MOV.U32=10,MOV=3,IADD3=1,ISETP.GE.AND=1", "IMAD.HI.U32"),
    P_U(ia_shl, "IMAD.SHL.U32=6,SHF.L.U32=2,IADD3=1,ISETP.GE.AND=1", "IMAD.SHL.U32"),
    P_U(ih_imad_32, "IMAD=64,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ih_imad_64, "IMAD=96,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ih_hi_32, "IMAD=32,IMAD.HI.U32=32,LEA.HI=32,LOP3.LUT=32,IMAD.MOV.U32=16,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_U(ih_hi_64, "IMAD.HI.U32=64,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMAD.MOV.U32=16,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo0_w4, 512, "IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo8_w4, 512, "IMAD=40,LOP3.LUT=32,LEA.HI=24,IMAD.HI.U32=8,IMAD.MOV.U32=8,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo12_w4, 512, "IMAD=44,LOP3.LUT=32,LEA.HI=20,IMAD.HI.U32=12,IMAD.MOV.U32=11,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo16_w4, 512, "IMAD=48,LOP3.LUT=32,IMAD.HI.U32=16,LEA.HI=16,IMAD.MOV.U32=14,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo32_w4, 512, "IMAD=64,IMAD.HI.U32=32,LOP3.LUT=32,IMAD.MOV.U32=26,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo0_w8, 1024, "IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo8_w8, 1024, "IMAD=40,LOP3.LUT=32,LEA.HI=24,IMAD.HI.U32=8,IMAD.MOV.U32=8,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo12_w8, 1024, "IMAD=44,LOP3.LUT=32,LEA.HI=20,IMAD.HI.U32=12,IMAD.MOV.U32=12,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo16_w8, 1024, "IMAD=48,LOP3.LUT=32,IMAD.HI.U32=16,LEA.HI=16,IMAD.MOV.U32=11,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
    P_(duo32_w8, 1024, "IMAD=64,IMAD.HI.U32=32,LOP3.LUT=32,IMAD.MOV.U32=24,IMMA.16816.S8.S8=8,IADD3=1,ISETP.GE.AND=1", "IMMA.16816.S8.S8"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
