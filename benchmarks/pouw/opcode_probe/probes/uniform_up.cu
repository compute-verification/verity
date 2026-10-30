// facet_break's UP search: uniform-datapath forms uniform.cu didn't time,
// against the rows that price them, UP above all (UNI at 0.766 a SMSP-cycle alone, UNI64 at 0.66, neither measured
// beside any form but UIMAD/ULOP3, USHF/ULEA/ULOP3 and LDCU). Built as uniform.cu is, under its measurement standard:
// 16 warps (4 a SMSP, one block a SM), each loop's uniform count
// pinned exactly by the gate, the chains loop-carried from a parameter into the sink, and every vector setting inside
// the honest step with its extra work reading the accumulators, the one place ptxas 13.0 keeps a chain uniform.
//   v<F>_<K><S>  F: h the honest step (8 IMMA, duo on each accumulator); g h + 64 FFMA reading the accumulators
//                (issue nearest saturated)
//                S, the uniform step on each of four chain pairs: x ULOP3 only (AND, OR into a constant); a UIADD3 +
//                ULOP3; w 64-bit add and xor on UR pairs (UIADD3.64, ULOP3, UMOV.64); v 64-bit adds (UIADD3.64, with
//                ULOP3 and UMOV.64 for the constants)
// Every count the gate pins is the loop's own, vector and uniform alike, so a program's load counts every instruction
// but the branch. Alone (the honest MMAs unread) ptxas 13.0.88 puts all these forms, and uniform.cu's own forms at 16
// or 32 chains, on lanes; which forms stay uniform moves with the kernel's shape (dropping an unused chain set moves
// most of them to lanes), so only the gate's counts say where a chain ran.
#include "../probe.cuh"
#include "honest.cuh"

constexpr int TPB = 512, ITERS = 2048;

enum { S_X, S_A, S_W, S_V };

// both chain sets live in every kernel: the shape under which ptxas 13.0 keeps these chains uniform in the honest step
template <int K, int S, int NC>
struct UChains {
  uint32_t u[NC];
  uint64_t w[NC];
  __device__ explicit UChains(uint32_t seed) {
#pragma unroll
    for (int c = 0; c < NC; c++) {
      u[c] = seed * (2u * c + 3u) + (uint32_t)c;
      w[c] = ((uint64_t)u[c] << 32) ^ (uint64_t)(seed + 7u * c);
    }
  }
  __device__ __forceinline__ void step(int) {
#pragma unroll
    for (int i = 0; i < K / NC; i++)
#pragma unroll
      for (int p = 0; p < NC; p += 2) {
        const uint32_t k = 0x7F4A7C15u + (uint32_t)(i * NC + p) * 0x9E3779B9u;
        const uint64_t k64 = ((uint64_t)k << 29) ^ 0x632BE59BD9B4E019ull;
        if constexpr (S == S_X) {
          u[p] = (u[p] & u[p + 1]) ^ k;
          u[p + 1] = (u[p + 1] | u[p]) ^ (k >> 3);
        }
        if constexpr (S == S_A) {
          u[p] = u[p] + u[p + 1] + k;
          u[p + 1] = u[p + 1] ^ u[p] ^ (k >> 3);
        }
        if constexpr (S == S_W) {
          w[p] = w[p] + w[p + 1] + k64;
          w[p + 1] = w[p + 1] ^ w[p] ^ (k64 >> 7);
        }
        if constexpr (S == S_V) {
          w[p] = w[p] + w[p + 1] + k64;
          w[p + 1] = w[p + 1] + (w[p] ^ k64);
        }
      }
  }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < NC; c++) s ^= u[c] ^ (uint32_t)w[c] ^ (uint32_t)(w[c] >> 32);
    return s;
  }
};

enum { F_H, F_G };

template <int NF>
struct Feed {
  float f[8], g;
  __device__ explicit Feed(int mode) {
    g = __uint_as_float(0x3F800001u ^ (uint32_t)mode);
#pragma unroll
    for (int c = 0; c < 8; c++) f[c] = 0.f;
  }
  __device__ __forceinline__ void step(const Honest& h) {
#pragma unroll
    for (int q = 0; q < NF; q++) f[q & 7] = __fmaf_rn(__int_as_float(h.d[q & 7][(q >> 3) & 3]), g, f[q & 7]);
  }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 8; c++) s ^= __float_as_uint(f[c]);
    return s;
  }
};

template <int F, int S, int K, int NC>
__device__ __forceinline__ void uni(const OpArgs& a) {
  UChains<K, S, NC> u(0x5eed0000u ^ (uint32_t)a.mode);
  Honest h(a.mode);
  Feed<F == F_G ? 64 : 0> e(a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    h.step();
    e.step(h);
    u.step(it);
  }
  OP_END
  OP_SINK(u.fold() ^ h.fold() ^ e.fold())
}

#define UK(nm, F, S, K, NC) \
  OP_KERNEL(nm, TPB) { uni<F, S, K, NC>(a); }
UK(vh_0x, F_H, S_X, 0, 8)
UK(vh_128x, F_H, S_X, 128, 8)
UK(vh_128a, F_H, S_A, 128, 8)
UK(vh_64w, F_H, S_W, 64, 8)
UK(vh_128w, F_H, S_W, 128, 8)
UK(vh_128v, F_H, S_V, 128, 8)
UK(vh_256v, F_H, S_V, 256, 8)
UK(vg_0x, F_G, S_X, 0, 8)
UK(vg_128x, F_G, S_X, 128, 8)
UK(vg_128a, F_G, S_A, 128, 8)
UK(vg_64w, F_G, S_W, 64, 8)
UK(vg_128v, F_G, S_V, 128, 8)

#define P_U(nm, EX, CNT) {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .expect = EX, .counted = CNT, .iters = ITERS}

static const OpProbe PROBES[] = {
    P_U(vh_0x, "LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,UISETP.NE.AND=1", "IMMA.16832.S8.S8"),
    P_U(vh_128x, "ULOP3.LUT=175,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,UISETP.NE.AND=1,UMOV=1", "IMMA.16832.S8.S8"),
    P_U(vh_128a, "UIADD3=65,ULOP3.LUT=64,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UISETP.NE.AND=1", "IMMA.16832.S8.S8"),
    P_U(vh_64w, "UIADD3.64=64,ULOP3.LUT=64,UMOV.64=32,LOP3.LUT=32,IMAD=32,LEA.HI=32,UMOV=8,IMMA.16832.S8.S8=8", "IMMA.16832.S8.S8"),
    P_U(vh_128w, "UIADD3.64=128,ULOP3.LUT=128,UMOV.64=64,LOP3.LUT=32,IMAD=32,LEA.HI=32,UMOV=8,IMMA.16832.S8.S8=8", "IMMA.16832.S8.S8"),
    P_U(vh_128v, "UIADD3.64=192,ULOP3.LUT=128,UMOV.64=64,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1", "IMMA.16832.S8.S8"),
    P_U(vh_256v, "UIADD3.64=384,ULOP3.LUT=256,UMOV.64=128,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1", "IMMA.16832.S8.S8"),
    P_U(vg_0x, "FFMA=64,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,UISETP.NE.AND=1", "IMMA.16832.S8.S8"),
    P_U(vg_128x, "ULOP3.LUT=175,FFMA=64,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,UISETP.NE.AND=1", "IMMA.16832.S8.S8"),
    P_U(vg_128a, "UIADD3=65,ULOP3.LUT=64,FFMA=64,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UISETP.NE.AND=1", "IMMA.16832.S8.S8"),
    P_U(vg_64w, "UIADD3.64=64,ULOP3.LUT=64,FFMA=64,UMOV.64=32,LOP3.LUT=32,IMAD=32,LEA.HI=32,UMOV=8,IMMA.16832.S8.S8=8", "IMMA.16832.S8.S8"),
    P_U(vg_128v, "UIADD3.64=192,ULOP3.LUT=128,UMOV.64=64,FFMA=64,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8", "IMMA.16832.S8.S8"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
