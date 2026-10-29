// facet_break's memory programs: the loads a table route reads its table with (shared, global through L1, texture),
// alone, two kinds in one loop, and beside a MMA form. `generate.py` instantiates them and writes each probe-table
// entry.
//
//   FB_LD(name, D1, D2, R, P, Q, tpb)          every warp: Q loads an iteration; load q is D2 when q % R == R - 1
//   FB_LDMMA(name, Form, D1, D2, R, P, Q, tpb) every warp: 8 MMAs and Q such loads an iteration, spread after the MMAs
//
// Two shared kinds aren't table reads: LDSM.16.M88.4 (ldmatrix x4, lane l gives row l's address) and STS.128 (a store of a
// loop-invariant quad), the operand-staging traffic of a GEMM beside its MMAs.
//
// A block's table is a ring of 64 lines of 512 B (32 KB): its shared table, or its own region of `a.g` for LDG and TLD, so
// a global table stays L1-resident. Load q of an iteration reads line (QL (warp + it) + q) mod 64, QL the power of two
// at or above Q, so ptxas folds q into the instruction's immediate, can't merge two loads, and the loop is not invariant.
// The lane's offset in its line is the pattern's: LINE, lane l at l x the load's width (one 128-byte phase each, no bank
// conflict); BCAST, every lane the line's first word; QUAD, lanes 4j..4j+3 the line's j-th word.
//
//   FB_L2(name, D, P, Q, tpb)                  every warp: Q L2-resident loads an iteration (the L2 facet of table.toml)
//
// An L2 program's table is the block's own 256 KB region of `a.g` (188 blocks: 47 MB, inside the 128 MB L2 and twice an
// SM's L1), swept before a step repeats: load q of warp w's iteration reads step (QL (it W' + w) + q) mod ring, W' the
// odd number at W + 1 or W + 2.
// LINE: lane l at l x width of a 32 x width line; SCAT: lane l in its own 8 KB of the region, one 32-byte sector a load
// (a gather's traffic: 32 sectors an instruction). The .NA (L1::no_allocate) and .STRONG.GPU (.cg) forms skip L1.
#pragma once
#include "mix.cuh"

namespace fb {

enum Ld { D_NONE, D_LDS32, D_LDS64, D_LDS128, D_LDSU8, D_LDG32, D_LDGNC32, D_LDG64, D_LDG128, D_TLD1, D_TLD2, D_TLD4,
          D_TLD4N, D_LDG256, D_L2G128, D_L2G128NA, D_L2G128CG, D_L2G256, D_L2G256CG, D_LDSM4, D_STS128 };
enum Pat { P_LINE, P_BCAST, P_QUAD, P_SCAT };

constexpr unsigned FB_LINE = 512, FB_RING = 64, FB_RINGB = FB_LINE * FB_RING, FB_L2B = 256u << 10;

__host__ __device__ constexpr bool shared_ld(int d) {
  return (d >= D_LDS32 && d <= D_LDSU8) || d == D_LDSM4 || d == D_STS128;
}
// bytes between two lanes' words: the load's width, or its texel's (tex1's texel is 4 B whatever the fetch's width);
// LDSM's lane gives one 16-byte matrix row
__host__ __device__ constexpr unsigned lane_stride(int d) {
  return d == D_LDS64 || d == D_LDG64 ? 8
         : d == D_LDG256 || d == D_L2G256 || d == D_L2G256CG ? 32
         : d == D_LDS128 || d == D_LDG128 || d == D_TLD4 || d >= D_L2G128 ? 16 : 4;
}
__host__ __device__ constexpr unsigned qlines(int q) { return q <= 1 ? 1 : 2 * qlines((q + 1) / 2); }
__host__ __device__ constexpr int kind_at(int d1, int d2, int r, int q) { return d2 != D_NONE && q % r == r - 1 ? d2 : d1; }
__host__ __device__ constexpr unsigned lane_off(int d, int p, int lane) {
  return p == P_LINE ? lane * lane_stride(d) : p == P_QUAD ? (lane >> 2) * lane_stride(d) : 0u;
}
template <int Q>
__host__ __device__ inline unsigned line_off(int warp, int it, int q) {
  return ((qlines(Q) * (unsigned)(warp + it) + (unsigned)q) & (FB_RING - 1)) * FB_LINE;
}

// The gate's bank check: the byte offsets of the slot-th shared access of an iteration, from the kernel's own formula.
template <int D1, int D2, int R, int P, int Q>
__host__ __device__ unsigned soff(int slot, int lane, int warp, int it) {
  int q = 0;
  for (int seen = -1; q < Q; q++)
    if (shared_ld(kind_at(D1, D2, R, q)) && ++seen == slot) break;
  return line_off<Q>(warp, it, q) + lane_off(kind_at(D1, D2, R, q), P, lane);
}

template <int D>
__device__ __forceinline__ uint32_t ld(const OpArgs& a, uint32_t sbase, const unsigned char* gb, unsigned gel, unsigned o) {
  uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
  if constexpr (D == D_LDS32) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r0) : "r"(sbase + o));
  if constexpr (D == D_LDS64) asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(sbase + o));
  if constexpr (D == D_LDS128)
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sbase + o));
  if constexpr (D == D_LDSU8) asm volatile("ld.shared.u8 %0, [%1];" : "=r"(r0) : "r"(sbase + o));
  if constexpr (D == D_LDG32) asm volatile("ld.global.u32 %0, [%1];" : "=r"(r0) : "l"(gb + o));
  if constexpr (D == D_LDGNC32) asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r0) : "l"(gb + o));
  if constexpr (D == D_LDG64) asm volatile("ld.global.v2.u32 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "l"(gb + o));
  if constexpr (D == D_LDG128 || D == D_L2G128)
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(gb + o));
  if constexpr (D == D_L2G128NA)
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(gb + o));
  if constexpr (D == D_L2G128CG)
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(gb + o));
  if constexpr (D == D_LDG256 || D == D_L2G256)
    asm volatile("ld.global.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7) : "l"(gb + o));
  if constexpr (D == D_L2G256CG)
    asm volatile("ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7) : "l"(gb + o));
  if constexpr (D == D_TLD1) r0 = tex1Dfetch<unsigned>((cudaTextureObject_t)a.tex1, (int)(gel + o / 4));
  if constexpr (D == D_TLD2) {
    const uint2 v = tex1Dfetch<uint2>((cudaTextureObject_t)a.tex1, (int)(gel + o / 4));
    r0 = v.x; r1 = v.y;
  }
  if constexpr (D == D_TLD4 || D == D_TLD4N) {
    const uint4 v = D == D_TLD4 ? tex1Dfetch<uint4>((cudaTextureObject_t)a.tex4, (int)(gel / 4 + o / 16))
                                : tex1Dfetch<uint4>((cudaTextureObject_t)a.tex1, (int)(gel + o / 4));
    r0 = v.x; r1 = v.y; r2 = v.z; r3 = v.w;
  }
  if constexpr (D == D_LDSM4)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sbase + o));
  // a store's quad is loop-invariant, so ptxas keeps it in one aligned quad and the loop holds the STS alone
  if constexpr (D == D_STS128)
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(sbase + o), "r"((uint32_t)a.mode), "r"(gel),
                 "r"((uint32_t)a.mode ^ 0x9E3779B9u), "r"(gel ^ 0x7F4A7C15u));
  return r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
}

// The block's table: shared (filled, nonzero) when a kind reads shared memory; gel is its region of g in 4-byte texels.
#define FB_TABLE(D1, D2)                                                                            \
  constexpr bool fb_s = shared_ld(D1) || shared_ld(D2);                                             \
  __shared__ __align__(128) unsigned char fb_tbl[fb_s ? FB_RINGB : 16];                             \
  if constexpr (fb_s)                                                                               \
    for (int i = threadIdx.x; i < (int)(FB_RINGB / 4); i += blockDim.x)                             \
      reinterpret_cast<uint32_t*>(fb_tbl)[i] = i * 2654435761u + blockIdx.x + 1;                    \
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(fb_tbl);                                \
  const unsigned char* gb = a.g + (size_t)blockIdx.x * FB_RINGB;                                     \
  const unsigned gel = blockIdx.x * (FB_RINGB / 4);                                                  \
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;

template <int D1, int D2, int R, int P, int Q>
__device__ __forceinline__ uint32_t loads(const OpArgs& a, uint32_t sbase, const unsigned char* gb, unsigned gel, int lane,
                                          int w, int it, int q0, int q1) {
  uint32_t acc = 0;
#pragma unroll
  for (int q = q0; q < q1; q++) {
    const unsigned o = line_off<Q>(w, it, q) + lane_off(kind_at(D1, D2, R, q), P, lane);
    if (kind_at(D1, D2, R, q) == D1) acc ^= ld<D1>(a, sbase, gb, gel, o);
    else if constexpr (D2 != D_NONE) acc ^= ld<D2>(a, sbase, gb, gel, o);
  }
  return acc;
}

template <int D1, int D2, int R, int P, int Q>
__device__ __forceinline__ void ld_alone(const OpArgs& a) {
  FB_TABLE(D1, D2)
  uint32_t acc = threadIdx.x + 1;
  OP_BEGIN
  OP_LOOP(it) { acc ^= loads<D1, D2, R, P, Q>(a, sbase, gb, gel, lane, w, it, 0, Q); }
  OP_END
  OP_SINK(acc)
}

template <class F, int D1, int D2, int R, int P, int Q>
__device__ __forceinline__ void ld_mma(const OpArgs& a) {
  FB_TABLE(D1, D2)
  Mma<F> m(lane);
  uint32_t acc = threadIdx.x + 1;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      m.go(c);
      acc ^= loads<D1, D2, R, P, Q>(a, sbase, gb, gel, lane, w, it, c * Q / 8, (c + 1) * Q / 8);
    }
  }
  OP_END
  OP_SINK((acc ^ m.fold()))
}

template <int D, int P, int Q, int W>
__device__ __forceinline__ unsigned l2_off(int w, int it, int q, int lane) {
  constexpr unsigned step = P == P_SCAT ? 32 : 32 * lane_stride(D), ring = (P == P_SCAT ? FB_L2B / 32 : FB_L2B) / step;
  // it's odd multiplier: with QL W a multiple of the ring the step would not depend on it, and ptxas hoists the loads
  return ((qlines(Q) * (unsigned)(it * ((W + 1) | 1) + w) + (unsigned)q) & (ring - 1)) * step +
         (P == P_SCAT ? lane * (FB_L2B / 32) : lane * lane_stride(D));
}

template <int D, int P, int Q, int TPB>
__device__ __forceinline__ void ld_l2(const OpArgs& a) {
  const unsigned char* gb = a.g + (size_t)(blockIdx.x % (unsigned)(a.gbytes / FB_L2B)) * FB_L2B;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  uint32_t acc = threadIdx.x + 1;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < Q; q++) acc ^= ld<D>(a, 0, gb, 0, l2_off<D, P, Q, TPB / 32>(w, it, q, lane));
  }
  OP_END
  OP_SINK(acc)
}

}  // namespace fb

#define FB_LD(name, D1, D2, R, P, Q, TPB) OP_KERNEL(name, TPB) { fb::ld_alone<fb::D1, fb::D2, R, fb::P, Q>(a); }
#define FB_LDMMA(name, F, D1, D2, R, P, Q, TPB) \
  OP_KERNEL(name, TPB) { fb::ld_mma<fb::F, fb::D1, fb::D2, R, fb::P, Q>(a); }
#define FB_L2(name, D, P, Q, TPB) OP_KERNEL(name, TPB) { fb::ld_l2<fb::D, fb::P, Q, TPB>(a); }
