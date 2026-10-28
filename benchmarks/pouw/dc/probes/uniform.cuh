// The uniform datapath on sm_90a and sm_100a, built as opcode_probe's uniform.cu and uniform_up.cu are: 16 warps (4 a
// SMSP, one block a SM), K uniform instructions an iteration on eight warp-uniform chains seeded from a parameter and
// carried around the loop, and every vector setting inside the honest step (8 IMMA m16n8k32, duo on each accumulator),
// the one place ptxas 13.0 keeps a chain on the uniform datapath. sm90_uniform.cu and sm100_uniform.cu pin each loop's
// counts exactly, so a chain that lands on lanes refuses.
//   u<F>_<K><S>  F, the fixed stream: z none (alone: the honest MMAs unread, which ptxas deletes after placing the chain);
//                h the honest step (beside duo's fold); g h + 64 FFMA reading the accumulators (issue binds); l h + 32
//                LOP3 reading them (the ALU pipe binds harder); m 32 IMMA and one fold (mma.sync binds the tensor pipe
//                that tcgen05 and wgmma share; without a fold reading the accumulators ptxas puts the chains on lanes);
//                b h + 256 FFMA reading the accumulators (issue-loaded, 0.72 of issue: each warp waits on its MMAs);
//                s h + 512 FFMA on 16 accumulators and 64 IADD3 + 64 LOP3 on 8 pairs, none reading the MMAs (UVec:
//                issue binds alone, the ALU at about half, so a uniform instruction's slowdown is its issue price);
//                v as s with each FFMA reading only its accumulator; w v with twice the vector work; x 1024 FADD by an
//                immediate and 32 IADD3 + 32 LOP3 (one register read an FADD: sm_90 loses issue to operand reads)
//                S, the uniform step on each chain pair: i UIMAD + ULOP3; x ULOP3; a UIADD3 + ULOP3; f USHF + ULOP3;
//                p UPRMT + ULOP3; s USHF/ULEA + ULOP3
#pragma once
#include "../../opcode_probe/probe.cuh"
#include "../../opcode_probe/probes/honest.cuh"
#include "uchain.cuh"

constexpr int UTPB = 512, UITERS = 2048;

enum { F_Z, F_H, F_G, F_L, F_M, F_B };

template <int NF, int NL>
struct UFeed {
  float f[8], g;
  uint32_t v[8], k;
  __device__ explicit UFeed(int mode) {
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

// R: 0 FFMA f·c + g; 1 FFMA f·c + f (one register read); 2 FADD f + c (one register read, FADD for FFMA)
template <int NF, int NI, int R = 0>
struct UVec {
  float f[16], g;
  uint32_t v[16];
  __device__ explicit UVec(int mode) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    g = __uint_as_float(0x3A800000u + (t & 0xFFFFu) + (uint32_t)mode);
#pragma unroll
    for (int c = 0; c < 16; c++) {
      f[c] = __uint_as_float(0x3F800000u + t * 16u + (uint32_t)c);
      v[c] = t * 0x9E3779B9u + (uint32_t)c;
    }
  }
  __device__ __forceinline__ void step() {
#pragma unroll
    for (int q = 0; q < NF; q++)
      f[q & 15] = R == 2   ? __fadd_rn(f[q & 15], 9.765625e-04f)
                  : R == 1 ? __fmaf_rn(f[q & 15], -6.103515625e-05f, f[q & 15])
                           : __fmaf_rn(f[q & 15], 0.99993896484375f, g);
#pragma unroll
    for (int q = 0; q < NI / 2; q++) {
      const int p = (q & 7) * 2;
      const uint32_t k = 0x85EBCA6Bu + (uint32_t)q * 0xC2B2AE35u;
      v[p] = v[p] + v[p + 1] + k;
      v[p + 1] = v[p + 1] ^ v[p] ^ (k >> 5);
    }
  }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 16; c++) s ^= __float_as_uint(f[c]) ^ v[c];
    return s;
  }
};

template <int S, int K, int NF = 512, int NI = 128, int R = 0>
__device__ __forceinline__ void uni_sat(const OpArgs& a) {
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  Honest h(a.mode);
  UVec<NF, NI, R> w(a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    h.step();
    w.step();
    u.step();
  }
  OP_END
  OP_SINK(u.fold() ^ h.fold() ^ w.fold())
}

template <int F, int S, int K>
__device__ __forceinline__ void uni(const OpArgs& a) {
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  Honest h(a.mode);
  UFeed<F == F_G ? 64 : F == F_B ? 256 : 0, F == F_L ? 32 : 0> e(a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    if constexpr (F == F_Z) h.step<8, false>();
    if constexpr (F == F_M) {
      h.step<8, false>();
      h.step<8, false>();
      h.step<8, false>();
    }
    if constexpr (F != F_Z) h.step();
    e.step(h);
    u.step();
  }
  OP_END
  OP_SINK(u.fold() ^ h.fold() ^ e.fold())
}

#define UK(nm, F, S, K) \
  OP_KERNEL(nm, UTPB) { uni<F, S, K>(a); }
#define UFORMS(F, N, K)        \
  UK(u##N##_##K##i, F, S_I, K) \
  UK(u##N##_##K##x, F, S_X, K) \
  UK(u##N##_##K##a, F, S_A, K) \
  UK(u##N##_##K##f, F, S_F, K) \
  UK(u##N##_##K##p, F, S_P, K)
UFORMS(F_Z, z, 64)
UFORMS(F_Z, z, 128)
UK(uz_64s, F_Z, S_S, 64)
UK(uh_0, F_H, S_I, 0)
UFORMS(F_H, h, 32)
UFORMS(F_H, h, 64)
UFORMS(F_H, h, 128)
UK(uh_64s, F_H, S_S, 64)
UK(ug_0, F_G, S_I, 0)
UFORMS(F_G, g, 64)
UFORMS(F_G, g, 128)
UK(ul_0, F_L, S_I, 0)
UFORMS(F_L, l, 64)
UFORMS(F_L, l, 128)
UK(um_0, F_M, S_I, 0)
UFORMS(F_M, m, 32)
UFORMS(F_M, m, 64)
UK(ub_0, F_B, S_I, 0)
UFORMS(F_B, b, 64)
UK(ub_128i, F_B, S_I, 128)
UK(ub_128x, F_B, S_X, 128)

#define USK(nm, S, K) \
  OP_KERNEL(nm, UTPB) { uni_sat<S, K>(a); }
#define USAT(K)              \
  USK(us_##K##x, S_X, K)     \
  USK(us_##K##a, S_A, K)     \
  USK(us_##K##i, S_I, K)
USK(us_0, S_I, 0)
USAT(16)
USAT(32)
USAT(64)
USAT(128)
USK(us_64f, S_F, 64)
USK(us_64p, S_P, 64)

#define UVK(nm, S, K, NF, NI, R) \
  OP_KERNEL(nm, UTPB) { uni_sat<S, K, NF, NI, R>(a); }
#define UVSAT(N, NF, NI, R)                 \
  UVK(u##N##_0, S_I, 0, NF, NI, R)          \
  UVK(u##N##_32x, S_X, 32, NF, NI, R)       \
  UVK(u##N##_64x, S_X, 64, NF, NI, R)       \
  UVK(u##N##_64a, S_A, 64, NF, NI, R)       \
  UVK(u##N##_64i, S_I, 64, NF, NI, R)       \
  UVK(u##N##_128x, S_X, 128, NF, NI, R)     \
  UVK(u##N##_128a, S_A, 128, NF, NI, R)     \
  UVK(u##N##_128i, S_I, 128, NF, NI, R)
UVSAT(v, 512, 128, 1)
UVSAT(w, 1024, 256, 1)
UVSAT(x, 1024, 64, 2)

#define P_U(nm, EX, CNT) {.name = #nm, .kernel = (const void*)nm, .tpb = UTPB, .expect = EX, .counted = CNT, .iters = UITERS}
