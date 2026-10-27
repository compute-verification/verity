// A2u at saturated issue: does a uniform-datapath instruction take an SMSP issue slot when the lanes already issue at
// nearly one warp instruction a cycle? uniform.cu's fixed streams never got there (their tensor and ALU pipes bind at
// 0.6-0.9 issue), so its K = 128 points can't tell an issue slot
// from interference. Here the vector stream spreads over four pipes so that issue binds, each a loop:
//   a: 2 IMMA (16 SMSP-cycles each on the tensor pipe: 32), 64 FFMA reading the accumulators (FMA pipe: 64), 16 LOP3 and
//      16 IMAD reading them (ALU and IMAD pipes: 32 each), about 100 warp instructions;
//   b: 2 IMMA, 16 FFMA, 40 LOP3, 40 IMAD (ALU and IMAD pipes: 80 each), about 100 warp instructions.
// uniform.cu's chains run beside it (UIMAD + ULOP3), NP chain pairs, K instructions a loop, seeded from a parameter and
// carried around the loop. ptxas places each chain pair on the uniform datapath or on lanes on its own; each probe's
// expect pins what it placed, so the loop's V and U are exact. At 1024 threads (64 registers) with 3 or 4 pairs it moves
// one pair to the uniform datapath and keeps the rest on lanes; with 1 or 2 pairs, or at 512 threads, it keeps every
// pair on lanes. So each probe with a uniform pair has a 2-pair control with the same chain instructions on lanes and
// none uniform:
//   <s>8_<K>     4 pairs: K/4 uniform instructions and 3K/4 chain instructions on lanes;
//   <s>8_<K>p3   3 pairs: K/3 uniform and 2K/3 on lanes;
//   <s>8_<J>p2   2 pairs: J on lanes, no uniform instruction (J = 3K/4 or 2K/3 of a probe above).
// The stream issues at about 0.8, so up to a fifth of the slots are idle and 16-32 uniform
// instructions fit there either way (they added 4-14 SMSP-cycles). The 3-pair probes at K = 192-384 put 64-128 uniform
// instructions beside 128-256 chain instructions on lanes, more than the idle slots: with A2u a probe's time exceeds its
// control's by most of its uniform count; without it by about 0, the uniform datapath's own cap (0.68-0.77 a SMSP-cycle,
// uniform.cu's uz_*) still below half the loop's time.
// Joint points: rows.jsonl only, no census line.
#include "../probe.cuh"

constexpr int ITERS = 2048;

template <int K, int NP>
struct UChains {
  uint32_t u[2 * NP];
  __device__ explicit UChains(uint32_t seed) {
#pragma unroll
    for (int c = 0; c < 2 * NP; c++) u[c] = seed * (2u * c + 3u) + (uint32_t)c;
  }
  __device__ __forceinline__ void step() {
#pragma unroll
    for (int i = 0; i < K / (2 * NP); i++)
#pragma unroll
      for (int p = 0; p < 2 * NP; p += 2) {
        const uint32_t k = 0x7F4A7C15u + (uint32_t)(i * 8 + p) * 0x9E3779B9u;
        u[p] = u[p] * 0x9E3779B1u + u[p + 1];
        u[p + 1] = u[p + 1] ^ u[p] ^ k;
      }
  }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 2 * NP; c++) s ^= u[c];
    return s;
  }
};

__device__ __forceinline__ void imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int K, int NP, int NF, int NL, int NI>
__device__ __forceinline__ void ai(const OpArgs& a) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x[4], y[2], v[4], w[4];
  int32_t d[2][4] = {};
  float f[8];
  const float g = __uint_as_float(0x3F800001u ^ (uint32_t)a.mode);
  const uint32_t k = 0x3C013C01u ^ (uint32_t)a.mode;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    x[i] = (t * 0x9E3779B1u) ^ (0x01010101u * (uint32_t)(i + 1));
    v[i] = t + (uint32_t)i;
    w[i] = t ^ (uint32_t)(i * 7);
  }
  y[0] = t * 0x85EBCA6Bu;
  y[1] = t * 0xC2B2AE35u;
#pragma unroll
  for (int c = 0; c < 8; c++) f[c] = (float)c;
  UChains<K, NP> u(0x5eed0000u ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    imma(d[0], x, y);
    imma(d[1], x, y);
#pragma unroll
    for (int q = 0; q < NF; q++) f[q & 7] = __fmaf_rn(__int_as_float(d[q & 1][(q >> 1) & 3]), g, f[q & 7]);
#pragma unroll
    for (int q = 0; q < NL; q++) v[q & 3] = v[q & 3] ^ (uint32_t)d[q & 1][(q >> 2) & 3] ^ (k + (uint32_t)q);
#pragma unroll
    for (int q = 0; q < NI; q++) w[q & 3] = w[q & 3] * (0x9E3779B1u + 2u * (uint32_t)q) + (uint32_t)d[(q >> 2) & 1][q & 3];
    u.step();
  }
  OP_END
  uint32_t s = u.fold();
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= __float_as_uint(f[c]);
#pragma unroll
  for (int i = 0; i < 4; i++) s ^= v[i] ^ w[i];
  OP_SINK(s)
}

#define SA 64, 16, 16
#define SB 16, 40, 40
#define AI(nm, TPB, K, NP, ...) \
  OP_KERNEL(nm, TPB) { ai<K, NP, __VA_ARGS__>(a); }
#define AIS(s, S)                    \
  AI(s##8_0, 1024, 0, 4, S)          \
  AI(s##8_64, 1024, 64, 4, S)        \
  AI(s##8_128, 1024, 128, 4, S)      \
  AI(s##8_48p3, 1024, 48, 3, S)      \
  AI(s##8_96p3, 1024, 96, 3, S)      \
  AI(s##8_32p2, 1024, 32, 2, S)      \
  AI(s##8_64p2, 1024, 64, 2, S)      \
  AI(s##8_96p2, 1024, 96, 2, S)      \
  AI(s##8_192p3, 1024, 192, 3, S)    \
  AI(s##8_288p3, 1024, 288, 3, S)    \
  AI(s##8_384p3, 1024, 384, 3, S)    \
  AI(s##8_128p2, 1024, 128, 2, S)    \
  AI(s##8_192p2, 1024, 192, 2, S)    \
  AI(s##8_256p2, 1024, 256, 2, S)
AIS(ai, SA)
AIS(bi, SB)

#define P_AI(nm, EX)                                                                                                   \
  {.name = #nm, .kernel = (const void*)nm, .tpb = 1024, .expect = "IMMA.16832.S8.S8=2," EX,                            \
   .counted = "IMMA.16832.S8.S8", .iters = ITERS, .note = "an issue-bound vector stream beside uniform chains"}
#define U(n) ",UIMAD*=" #n ",ULOP3*=" #n

#define P_AIS(s, F)                                                                                                   \
  P_AI(s##8_0, F U(0)), P_AI(s##8_64, F U(8)), P_AI(s##8_128, F U(16)), P_AI(s##8_48p3, F U(8)),                      \
      P_AI(s##8_96p3, F U(16)), P_AI(s##8_32p2, F U(0)), P_AI(s##8_64p2, F U(0)), P_AI(s##8_96p2, F U(0)),          \
      P_AI(s##8_192p3, F U(32)), P_AI(s##8_288p3, F U(48)), P_AI(s##8_384p3, F U(64)), P_AI(s##8_128p2, F U(0)),     \
      P_AI(s##8_192p2, F U(0)), P_AI(s##8_256p2, F U(0))

static const OpProbe PROBES[] = {P_AIS(ai, "FFMA*=64"), P_AIS(bi, "FFMA*=16")};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
