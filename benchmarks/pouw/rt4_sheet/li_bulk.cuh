// li-bulk's device pieces (li_bulk.cu, li_bulk_wg.cu): li's constants and reference fold, the half-step on the
// bulk engine, and the static li_k slabs.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <vector>

#define NCH 8

__host__ __device__ constexpr uint32_t li_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
__host__ __device__ constexpr uint32_t li_a(int c, int h) { return li_mix32(0xA0000000u + 2u * (32 + 2 * c + h)) | 1u; }
__host__ __device__ constexpr uint32_t li_k(int c, int h) { return li_mix32(0xB0000000u + 2u * (32 + 2 * c + h) + 1u); }
__host__ __device__ constexpr int li_entry(int c, int i) { return 4 * (c / 4 + 2 * i) + c % 4; }

// li's fold of one 32-word set into the 8 chains (red_generic.cu's li_set, the reference)
__host__ __device__ inline void li_set(uint32_t (&y)[8], const uint32_t* p) {
  for (int c = 0; c < 8; c++)
    for (int h = 0; h < 2; h++)
      y[c] = (y[c] ^ p[li_entry(c, 2 * h)] ^ li_k(c, h)) * li_a(c, h) + p[li_entry(c, 2 * h + 1)];
}

__device__ __forceinline__ void st128(uint32_t* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void sts128(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void ldcg128(const uint32_t* p, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
}
// one relaxed bulk XOR reduce of `bytes` of shared [s] into global [g] (UBLKRED on sm_90a); issued, not waited
__device__ __forceinline__ void bulk_xor_issue(uint32_t* g, uint32_t s, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%1], %2;" ::"l"(g), "r"(s), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_commit_wait() {
  asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group 0;\n\tfence.proxy.async;" ::: "memory");
}
// li's CP_RED8 checkpoint: each chain reduced by a generic relaxed XOR into the lane's word (REDG.E.XOR), zm 0 at run
// time so ptxas can't prove the address warp-uniform and fold it to one REDUX
__device__ __forceinline__ void cp_red8(const uint32_t (&y)[8], uint32_t* gw) {
#pragma unroll
  for (int c = 0; c < 8; c++) asm volatile("red.relaxed.gpu.xor.b32 [%0], %1;" ::"l"(gw), "r"(y[c]) : "memory");
}

// The fold's barrier: the whole block (LEAD 0), or named barrier 1 over the 128 threads from LEAD (lbw's lane
// warpgroup, beside the wgmma warpgroups).
template <int LEAD>
__device__ __forceinline__ void fold_sync() {
  if constexpr (LEAD == 0)
    __syncthreads();
  else
    asm volatile("bar.sync 1, 128;" ::: "memory");
}

// one half-step over the block: write y, write the even entries to shared, bulk-XOR the shared slot and the h-slab into
// the global slot, read u back, finish with li's IMAD. `pe`/`po` are this lane's even/odd entries for the 8 chains.
template <int BT, int H, int LEAD = 0>
__device__ __forceinline__ void li_half(uint32_t (&y)[8], const uint32_t (&pe)[8], const uint32_t (&po)[8], uint32_t* gslot,
                                     uint32_t sp, uint32_t* gblk, uint32_t spblk, uint32_t skslab) {
  constexpr int SLOT = BT * NCH;
  st128(gslot, y[0], y[1], y[2], y[3]);
  st128(gslot + 4, y[4], y[5], y[6], y[7]);
  sts128(sp, pe[0], pe[1], pe[2], pe[3]);
  sts128(sp + 16, pe[4], pe[5], pe[6], pe[7]);
  asm volatile("fence.proxy.async;" ::: "memory");
  fold_sync<LEAD>();
  if (threadIdx.x == LEAD) {
    bulk_xor_issue(gblk, spblk, SLOT * 4u);
    bulk_xor_issue(gblk, skslab + (uint32_t)(H * SLOT * 4u), SLOT * 4u);
    bulk_commit_wait();
  }
  fold_sync<LEAD>();
  uint32_t u[8];
  ldcg128(gslot, u[0], u[1], u[2], u[3]);
  ldcg128(gslot + 4, u[4], u[5], u[6], u[7]);
#pragma unroll
  for (int c = 0; c < 8; c++) y[c] = u[c] * li_a(c, H) + po[c];
}

// the shared layout: [0, SLOT) the even-entry slot, [SLOT, 3 SLOT) the two static li_k slabs (periodic in NCH). Built
// once a launch; `smem` is SLOT + 2 SLOT words.
template <int BT, int LEAD = 0>
__device__ __forceinline__ void fill_kslab(uint32_t* smem) {
  constexpr int SLOT = BT * NCH;
  for (int i = threadIdx.x - LEAD; i < 2 * SLOT; i += BT) {
    const int h = i / SLOT, c = i % NCH;
    smem[SLOT + i] = li_k(c, h);
  }
}
