// A2u at saturated issue (note:20261019T0456Z-draft-rt-machine, RT-MACHINE-4): does a uniform-datapath instruction take
// an SMSP issue slot when the lanes already issue at nearly one warp instruction a cycle? uniform.cu's fixed streams
// never got there (their tensor and ALU pipes bind at 0.6-0.9 issue), so its K = 128 points can't tell an issue slot
// from interference. Here the stream spreads over four pipes, none above about two thirds of issue a warp-iteration:
//   2 IMMA (16 SMSP-cycles each on the tensor pipe: 32), 64 FFMA reading the accumulators (FMA pipe: 64), 16 LOP3 and
//   16 IMAD reading them (ALU and IMAD pipes: 32 each), about 100 warp instructions in all, so issue binds.
// uniform.cu's chains run beside it (UIMAD + ULOP3, K a loop), seeded from a parameter and carried around the loop;
// ptxas puts them on the uniform datapath only beside MMAs whose results the vector work reads, as here.
//   ai<W>_<K>  W warps a SMSP (4: 512 threads, 8: 1024), K uniform instructions a loop.
// With A2u each uniform instruction adds an SMSP-cycle a warp-iteration (T(64) about 1.6 T(0)); without it T stays at
// T(0) until the uniform datapath's own cap (0.68-0.77 a SMSP-cycle, uniform.cu's uz_*) binds, near K = 70.
// Joint points: rows.jsonl only, no census line.
#include "../probe.cuh"

constexpr int ITERS = 2048;

template <int K>
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
        u[p] = u[p] * 0x9E3779B1u + u[p + 1];
        u[p + 1] = u[p + 1] ^ u[p] ^ k;
      }
  }
  __device__ uint32_t fold() const { return u[0] ^ u[1] ^ u[2] ^ u[3] ^ u[4] ^ u[5] ^ u[6] ^ u[7]; }
};

__device__ __forceinline__ void imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int K>
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
  UChains<K> u(0x5eed0000u ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    imma(d[0], x, y);
    imma(d[1], x, y);
#pragma unroll
    for (int q = 0; q < 64; q++) f[q & 7] = __fmaf_rn(__int_as_float(d[q & 1][(q >> 1) & 3]), g, f[q & 7]);
#pragma unroll
    for (int q = 0; q < 16; q++) v[q & 3] = v[q & 3] ^ (uint32_t)d[q & 1][(q >> 2) & 3] ^ (k + (uint32_t)q);
#pragma unroll
    for (int q = 0; q < 16; q++) w[q & 3] = w[q & 3] * (0x9E3779B1u + 2u * (uint32_t)q) + (uint32_t)d[(q >> 2) & 1][q & 3];
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

#define AI(nm, TPB, K) \
  OP_KERNEL(nm, TPB) { ai<K>(a); }
AI(ai4_0, 512, 0)
AI(ai4_32, 512, 32)
AI(ai4_64, 512, 64)
AI(ai4_128, 512, 128)
AI(ai8_0, 1024, 0)
AI(ai8_32, 1024, 32)
AI(ai8_64, 1024, 64)
AI(ai8_128, 1024, 128)

#define EX "IMMA.16832.S8.S8=2,FFMA*=64"
#define P_AI(nm, TPB, U)                                                                                               \
  {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .expect = EX U, .counted = "IMMA.16832.S8.S8", .iters = ITERS, \
   .note = "2 IMMA, 64 FFMA, 16 LOP3 and 16 IMAD a loop beside uniform chains"}

static const OpProbe PROBES[] = {
    P_AI(ai4_0, 512, ",UIMAD*=0,ULOP3*=0"),     P_AI(ai4_32, 512, ",UIMAD*=16,ULOP3*=16"),
    P_AI(ai4_64, 512, ",UIMAD*=32,ULOP3*=32"),  P_AI(ai4_128, 512, ",UIMAD*=64,ULOP3*=64"),
    P_AI(ai8_0, 1024, ",UIMAD*=0,ULOP3*=0"),    P_AI(ai8_32, 1024, ",UIMAD*=16,ULOP3*=16"),
    P_AI(ai8_64, 1024, ",UIMAD*=32,ULOP3*=32"), P_AI(ai8_128, 1024, ",UIMAD*=64,ULOP3*=64"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
