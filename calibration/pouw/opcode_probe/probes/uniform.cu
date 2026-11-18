// P10: the vector issue slots a uniform-datapath instruction takes. Each kernel
// is 16 warps (4 a SMSP, one block a SM); its loop runs a fixed stream F and K uniform instructions: eight warp-uniform
// chains seeded from a parameter and carried around the loop, K/8 steps of each chain pair an iteration.
//
// ptxas decides which datapath runs a warp-uniform chain, and it only puts one on the uniform datapath in a loop with
// MMAs whose other vector work it feeds (ptxas 13.0.88, local cubins): beside FFMA or LOP3 streams that don't read an
// MMA's accumulators, in plain C or inline asm, it keeps the chain on lanes, PREDICT's SHFL broadcast (egemm.cu's
// ubcast) included. So every fixed stream here sits in the honest step, and its extra vector
// work reads the accumulators; "alone" keeps the honest step's MMAs with their results unread, which ptxas deletes
// after it has placed the chain, leaving a loop of uniform instructions and its branch. The gate pins each loop's
// uniform count exactly, so a chain that lands on lanes refuses.
//   u<F>_<K><S>  F, the fixed stream: z none (alone); h the honest step (8 IMMA, then duo on each accumulator:
//                PREDICT's shape, the ALU pipe binds with issue spare); f h + 32 FFMA reading the accumulators (issue
//                binds); g h + 64 such FFMA; l h + 32 LOP3 reading the accumulators (the ALU pipe binds harder)
//                S, the uniform step: none UIMAD + ULOP3 (PREDICT's); s USHF/ULEA + ULOP3; c a constant-bank word at a
//                uniform index (LDCU) into UIMAD + ULOP3
// eps1.py's `uniform` reads each loop's uniform instructions from its SASS and fits SMSP-cycles a warp-iteration
// against them.
#include "../probe.cuh"
#include "honest.cuh"

constexpr int TPB = 512, ITERS = 2048;

__constant__ uint32_t UC[64];

enum { S_M, S_S, S_C };

template <int K, int S>
struct UChains {
  uint32_t u[8];
  __device__ explicit UChains(uint32_t seed) {
#pragma unroll
    for (int c = 0; c < 8; c++) u[c] = seed * (2u * c + 3u) + (uint32_t)c;
  }
  __device__ __forceinline__ void step(int it) {
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
  }
  __device__ uint32_t fold() const { return u[0] ^ u[1] ^ u[2] ^ u[3] ^ u[4] ^ u[5] ^ u[6] ^ u[7]; }
};

enum { F_Z, F_H, F_F, F_G, F_L };

// vector work that reads the honest step's accumulators: NF FFMA (on their bits) or NL LOP3
template <int NF, int NL>
struct Feed {
  float f[8], g;
  uint32_t v[8], k;
  __device__ explicit Feed(int mode) {
    g = __uint_as_float(0x3F800001u ^ (uint32_t)mode);
    k = 0x3C013C01u ^ (uint32_t)mode;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      f[c] = 0.f;
      v[c] = (uint32_t)c;
    }
  }
  __device__ __forceinline__ void step(const Honest& h) {
#pragma unroll
    for (int q = 0; q < NF; q++) f[q & 7] = __fmaf_rn(__int_as_float(h.d[q & 7][(q >> 3) & 3]), g, f[q & 7]);
#pragma unroll
    for (int q = 0; q < NL; q++) v[q & 7] = v[q & 7] ^ (uint32_t)h.d[q & 7][(q >> 3) & 3] ^ (k + (uint32_t)q);
  }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 8; c++) s ^= __float_as_uint(f[c]) ^ v[c];
    return s;
  }
};

template <int F, int S, int K>
__device__ __forceinline__ void uni(const OpArgs& a) {
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  Honest h(a.mode);
  Feed<F == F_F ? 32 : F == F_G ? 64 : 0, F == F_L ? 32 : 0> e(a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    if constexpr (F == F_Z) h.step<8, false>();
    else h.step();
    e.step(h);
    u.step(it);
  }
  OP_END
  OP_SINK(u.fold() ^ h.fold() ^ e.fold())
}

#define UK(nm, F, S, K) \
  OP_KERNEL(nm, TPB) { uni<F, S, K>(a); }
UK(uz_16, F_Z, S_M, 16)
UK(uz_32, F_Z, S_M, 32)
UK(uz_64, F_Z, S_M, 64)
UK(uz_128, F_Z, S_M, 128)
UK(uz_64s, F_Z, S_S, 64)
UK(uz_64c, F_Z, S_C, 64)
UK(uh_0, F_H, S_M, 0)
UK(uh_16, F_H, S_M, 16)
UK(uh_32, F_H, S_M, 32)
UK(uh_64, F_H, S_M, 64)
UK(uh_128, F_H, S_M, 128)
UK(uh_64s, F_H, S_S, 64)
UK(uh_64c, F_H, S_C, 64)
UK(uf_0, F_F, S_M, 0)
UK(uf_16, F_F, S_M, 16)
UK(uf_32, F_F, S_M, 32)
UK(uf_64, F_F, S_M, 64)
UK(uf_128, F_F, S_M, 128)
UK(ug_0, F_G, S_M, 0)
UK(ug_32, F_G, S_M, 32)
UK(ug_64, F_G, S_M, 64)
UK(ug_128, F_G, S_M, 128)
UK(ul_0, F_L, S_M, 0)
UK(ul_32, F_L, S_M, 32)
UK(ul_64, F_L, S_M, 64)
UK(ul_128, F_L, S_M, 128)

#define IM "IMMA.16832.S8.S8=8"
#define P_U(nm, EX, CNT) {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .expect = EX, .counted = CNT, .iters = ITERS}
#define P_Z(nm, U) P_U(nm, U, "ULOP3*")
#define P_H(nm, U) P_U(nm, IM U, "IMMA.16832.S8.S8")
#define P_F(nm, U) P_U(nm, IM ",FFMA*=32" U, "IMMA.16832.S8.S8")
#define P_G(nm, U) P_U(nm, IM ",FFMA*=64" U, "IMMA.16832.S8.S8")
#define P_L(nm, U) P_U(nm, IM U, "IMMA.16832.S8.S8")
#define SS ",USHF*=32,ULEA*=32,ULOP3*=32"
#define SC ",LDCU*=32,UIMAD*=32,ULOP3*=64"

static const OpProbe PROBES[] = {
    P_Z(uz_16, "IMMA*=0,UIMAD*=8,ULOP3*=8"),
    P_Z(uz_32, "IMMA*=0,UIMAD*=16,ULOP3*=16"),
    P_Z(uz_64, "IMMA*=0,UIMAD*=32,ULOP3*=32"),
    P_Z(uz_128, "IMMA*=0,UIMAD*=64,ULOP3*=64"),
    P_Z(uz_64s, "IMMA*=0" SS),
    P_Z(uz_64c, "IMMA*=0" SC),
    P_H(uh_0, ",UIMAD*=0,ULOP3*=0"),
    P_H(uh_16, ",UIMAD*=8,ULOP3*=8"),
    P_H(uh_32, ",UIMAD*=16,ULOP3*=16"),
    P_H(uh_64, ",UIMAD*=32,ULOP3*=32"),
    P_H(uh_128, ",UIMAD*=64,ULOP3*=64"),
    P_H(uh_64s, SS),
    P_H(uh_64c, SC),
    P_F(uf_0, ",UIMAD*=0,ULOP3*=0"),
    P_F(uf_16, ",UIMAD*=8,ULOP3*=8"),
    P_F(uf_32, ",UIMAD*=16,ULOP3*=16"),
    P_F(uf_64, ",UIMAD*=32,ULOP3*=32"),
    P_F(uf_128, ",UIMAD*=64,ULOP3*=64"),
    P_G(ug_0, ",UIMAD*=0,ULOP3*=0"),
    P_G(ug_32, ",UIMAD*=16,ULOP3*=16"),
    P_G(ug_64, ",UIMAD*=32,ULOP3*=32"),
    P_G(ug_128, ",UIMAD*=64,ULOP3*=64"),
    P_L(ul_0, ",UIMAD*=0,ULOP3*=0"),
    P_L(ul_32, ",UIMAD*=16,ULOP3*=16"),
    P_L(ul_64, ",UIMAD*=32,ULOP3*=32"),
    P_L(ul_128, ",UIMAD*=64,ULOP3*=64"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
