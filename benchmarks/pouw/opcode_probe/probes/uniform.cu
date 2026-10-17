// EPS1 P10 (note:20261017T0324Z-draft-eps1-spec): the vector issue slots a uniform-datapath instruction takes. Each kernel
// is 16 warps (4 a SMSP, one block a SM); its loop runs a fixed stream F and K uniform instructions on eight warp-uniform
// chains. ptxas keeps a chain on the uniform datapath only when a SHFL from lane 0 makes its head warp-uniform (PREDICT's
// u16s form, egemm.cu's ubcast; seeded from blockIdx or a parameter, or carried around the loop, it stays on lanes:
// r20261017-043717-430e), so each iteration broadcasts one head, derives the eight chains from it, steps them K/8 times
// and folds them back into the head: a fixed cost an iteration, K uniform instructions on top.
//   u<F><S>_<K>  F, the fixed stream: n none; f 32 FFMA in 8 chains (issue-bound, the census's 0.97 a SMSP-cycle);
//                l 16 LOP3 + 16 FFMA (the ALU and FMA pipes, issue-bound); a 32 LOP3 (the ALU pipe alone, half the
//                issue slots free); h the honest kernel (8 IMMA, then duo on each accumulator: PREDICT's shape)
//                S, the uniform step: m UIMAD + ULOP3 (PREDICT's); s USHF + UIADD3 + ULOP3; c a constant-bank word at a
//                uniform index (LDCU) into ULOP3 + UIMAD
// The gate pins the fixed stream and the one SHFL; eps1.py's `uniform` reads each loop's uniform instructions from its
// SASS and fits SMSP-cycles a warp-iteration against them.
#include "../probe.cuh"
#include "honest.cuh"

constexpr int TPB = 512, ITERS = 2048;

__constant__ uint32_t UC[64];

enum { S_M, S_S, S_C };

template <int K, int S>
struct UChains {
  uint32_t h;
  __device__ explicit UChains(uint32_t seed) : h(seed) {}
  __device__ __forceinline__ void step(int it) {
    const uint32_t b = __shfl_sync(0xFFFFFFFFu, h, 0);
    uint32_t u[8];
#pragma unroll
    for (int c = 0; c < 8; c++) u[c] = b * (2u * c + 3u) + (uint32_t)c;
#pragma unroll
    for (int i = 0; i < K / 8; i++)
#pragma unroll
      for (int p = 0; p < 8; p += 2) {
        const uint32_t k = 0x7F4A7C15u + (uint32_t)(i * 8 + p) * 0x9E3779B9u;
        if constexpr (S == S_M) {
          u[p] = u[p] * 0x9E3779B1u + u[p + 1];
          u[p + 1] = u[p + 1] ^ u[p] ^ k;
        }
        if constexpr (S == S_S) {
          u[p] = (u[p] >> 3) + u[p + 1];
          u[p + 1] = u[p + 1] ^ (u[p] << 5) ^ k;
        }
        if constexpr (S == S_C) {
          u[p] = u[p] * 0x9E3779B1u + u[p + 1];
          u[p + 1] = u[p + 1] ^ u[p] ^ UC[((uint32_t)it * 8u + (uint32_t)(i * 8 + p)) & 63u];
        }
      }
    h = u[0] ^ u[1] ^ u[2] ^ u[3] ^ u[4] ^ u[5] ^ u[6] ^ u[7];
  }
};

enum { F_N, F_F, F_L, F_A, F_H };

// frontier.cu's operand form: a.mode is 0 at run time, so the operands stay registers and ptxas folds no immediate
struct Fixed {
  uint32_t v[8], k1, k2;
  float f[8], g1, g2;
  __device__ explicit Fixed(const OpArgs& a) {
    k1 = 0x3C013C01u ^ (uint32_t)a.mode;
    k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
    g1 = __uint_as_float(0x3F800001u ^ (uint32_t)a.mode);
    g2 = __uint_as_float(0x33D6BF95u ^ (uint32_t)(a.mode << 3));
#pragma unroll
    for (int i = 0; i < 8; i++) {
      v[i] = threadIdx.x * 8u + i + 1u;
      f[i] = __uint_as_float(0x3F800000u + threadIdx.x * 8u + i);
    }
  }
  __device__ __forceinline__ void ffma(int s) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[s]) : "f"(g1), "f"(g2));
  }
  __device__ __forceinline__ void lop3(int s) {
    asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v[s]) : "r"(k1), "r"(k2));
  }
  template <int F>
  __device__ __forceinline__ void step() {
#pragma unroll
    for (int q = 0; q < 32; q++) {
      if constexpr (F == F_F) ffma(q & 7);
      if constexpr (F == F_A) lop3(q & 7);
      if constexpr (F == F_L) {
        if (q & 1) ffma(q & 7);
        else lop3(q & 7);
      }
    }
  }
  __device__ uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x ^= v[i] ^ __float_as_uint(f[i]);
    return x;
  }
};

template <int F, int S, int K>
__device__ __forceinline__ void uni(const OpArgs& a) {
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  if constexpr (F == F_H) {
    Honest h(a.mode);
    OP_BEGIN
    OP_LOOP(it) {
      h.step();
      u.step(it);
    }
    OP_END
    OP_SINK(h.fold() ^ u.h)
  } else {
    Fixed x(a);
    OP_BEGIN
    OP_LOOP(it) {
      x.step<F>();
      u.step(it);
    }
    OP_END
    OP_SINK(x.fold() ^ u.h)
  }
}

#define UK(nm, F, S, K) \
  OP_KERNEL(nm, TPB) { uni<F, S, K>(a); }
UK(unm_0, F_N, S_M, 0)
UK(unm_16, F_N, S_M, 16)
UK(unm_32, F_N, S_M, 32)
UK(unm_64, F_N, S_M, 64)
UK(unm_128, F_N, S_M, 128)
UK(ufm_0, F_F, S_M, 0)
UK(ufm_8, F_F, S_M, 8)
UK(ufm_16, F_F, S_M, 16)
UK(ufm_32, F_F, S_M, 32)
UK(ufm_64, F_F, S_M, 64)
UK(ufs_16, F_F, S_S, 16)
UK(ufs_64, F_F, S_S, 64)
UK(ufc_32, F_F, S_C, 32)
UK(ufc_64, F_F, S_C, 64)
UK(ulm_0, F_L, S_M, 0)
UK(ulm_16, F_L, S_M, 16)
UK(ulm_32, F_L, S_M, 32)
UK(ulm_64, F_L, S_M, 64)
UK(uam_0, F_A, S_M, 0)
UK(uam_16, F_A, S_M, 16)
UK(uam_32, F_A, S_M, 32)
UK(uam_64, F_A, S_M, 64)
UK(uhm_0, F_H, S_M, 0)
UK(uhm_8, F_H, S_M, 8)
UK(uhm_16, F_H, S_M, 16)
UK(uhm_32, F_H, S_M, 32)
UK(uhm_64, F_H, S_M, 64)
UK(uhs_32, F_H, S_S, 32)

#define SH "SHFL*=1"
#define P_U(nm, EX, CNT) {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .expect = EX, .counted = CNT, .iters = ITERS}
#define P_N(nm) P_U(nm, SH, "SHFL*")
#define P_F(nm) P_U(nm, "FFMA*=32," SH, "FFMA*")
#define P_L(nm) P_U(nm, "FFMA*=16,LOP3*=16," SH, "FFMA*")
#define P_A(nm) P_U(nm, "LOP3*=32," SH, "LOP3*")
#define P_H(nm) P_U(nm, "IMMA.16832.S8.S8=8," SH, "IMMA.16832.S8.S8")

static const OpProbe PROBES[] = {
    P_N(unm_0),  P_N(unm_16), P_N(unm_32), P_N(unm_64), P_N(unm_128), P_F(ufm_0),  P_F(ufm_8),
    P_F(ufm_16), P_F(ufm_32), P_F(ufm_64), P_F(ufs_16), P_F(ufs_64),  P_F(ufc_32), P_F(ufc_64),
    P_L(ulm_0),  P_L(ulm_16), P_L(ulm_32), P_L(ulm_64), P_A(uam_0),   P_A(uam_16), P_A(uam_32),
    P_A(uam_64), P_H(uhm_0),  P_H(uhm_8),  P_H(uhm_16), P_H(uhm_32),  P_H(uhm_64), P_H(uhs_32),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
