// Shared-memory operand layout and matrix descriptors for sm_90a wgmma and sm_100a tcgen05.mma, and the tcgen05 and
// mbarrier wrappers the probes and the Dither GEMM share. One layout everywhere: K-major, no swizzle, the canonical
// ((8, m), (T, 2k)) : ((1T, SBO), (1, LBO)) of PTX ISA 9.7.17.5 / 9.7.18.4: 8-row x 16-byte core matrices, the core matrix
// of row group rg and 16-byte K chunk kc at (rg * KC + kc) * 128 bytes for KC = K-tile bytes / 16. So LBO, the step between
// core matrices along K, is 128 bytes, and SBO, the step between 8-row groups, is KC * 128.
#pragma once
#include <cstdint>

#define DC_DEV __device__ __forceinline__
#define DC_HD __host__ __device__ inline

// The byte offset of element (row, kbyte) of a K-major operand tile whose rows hold kb bytes of K.
DC_HD constexpr uint32_t kmaj_off(uint32_t row, uint32_t kbyte, uint32_t kb) {
  return ((row >> 3) * (kb >> 4) + (kbyte >> 4)) * 128u + (row & 7u) * 16u + (kbyte & 15u);
}
DC_HD constexpr uint32_t kmaj_lbo() { return 128u; }
DC_HD constexpr uint32_t kmaj_sbo(uint32_t kb) { return (kb >> 4) * 128u; }

DC_DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// sm_90 wgmma descriptor: start >> 4 [0,14), LBO >> 4 [16,30), SBO >> 4 [32,46), base offset 0, layout 0 (no swizzle)
// [62,64). dgemm's gate reruns a mismatch with LBO and SBO exchanged, so a misread of the two shows in one run.
DC_HD uint64_t wgmma_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) |
         ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
// sm_100 tcgen05 descriptor: as wgmma's, plus the fixed version 0b001 at [46,49), LBO mode 0, swizzle 0 at [61,64).
DC_HD uint64_t umma_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return wgmma_desc(saddr, lbo, sbo) | (1ull << 46);
}
// tcgen05.mma instruction descriptor (PTX ISA 9.7.18.4.2, kind::f16 / tf32 / f8f6f4 / i8): dense, no saturate,
// D format [4,6) (f32 1, s32 2), A format [7,10), B format [10,13), no negate, both K-major, N >> 3 [17,23), M >> 4 [24,29).
// A and B formats: kind::i8 u8 0, s8 1; kind::f8f6f4 e4m3 0, e5m2 1; kind::f16 f16 0, bf16 1.
DC_HD constexpr uint32_t umma_idesc(uint32_t dfmt, uint32_t afmt, uint32_t bfmt, uint32_t m, uint32_t n) {
  return (dfmt << 4) | (afmt << 7) | (bfmt << 10) | ((n >> 3) << 17) | ((m >> 4) << 24);
}

DC_DEV void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
// Waits for the phase of parity `ph` to complete (try_wait suspends in hardware; the loop is one SASS loop).
DC_DEV void mbar_wait(uint64_t* bar, uint32_t ph) {
  asm volatile(
      "{\n.reg .pred P1;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@!P1 bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)), "r"(ph) : "memory");
}
DC_DEV void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DC_DEV void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

// ---- tcgen05 (sm_100a): tensor memory, the MMA, its commit and the loads ----------------------------------------------
// TMEM address: lane in [31:16], column in [15:0]. A warp reaches only lanes 32 (warp % 4) .. + 31.
DC_DEV void tmem_alloc(uint32_t* slot, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(slot)), "r"(cols)
               : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
DC_DEV void tmem_dealloc(uint32_t taddr, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols) : "memory");
}
DC_DEV void tc_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DC_DEV void tc_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
// D[tmem] (+)= A B^T, one thread issues it; acc 0 overwrites D. KIND: "i8", "f8f6f4", "f16".
#define DC_UMMA(KIND)                                                                                           \
  DC_DEV void umma_##KIND(uint32_t d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc) {                 \
    asm volatile(                                                                                               \
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"                                                             \
        "tcgen05.mma.cta_group::1.kind::" #KIND " [%0], %1, %2, %3, p;\n}\n" ::"r"(d), "l"(da), "l"(db), "r"(idesc), \
        "r"(acc)                                                                                                \
        : "memory");                                                                                            \
  }
DC_UMMA(i8)
DC_UMMA(f8f6f4)
DC_UMMA(f16)
#undef DC_UMMA
// The mbarrier arrives once every tcgen05.mma this thread issued before it has completed.
DC_DEV void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
               : "memory");
}
// One lane of the warp (all 32 active): ptxas then issues a tcgen05.mma once, without a per-thread ELECT loop.
DC_DEV uint32_t elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n.reg .b32 rx;\n.reg .pred px;\nelect.sync rx|px, 0xffffffff;\n@px mov.s32 %0, 1;\n}\n"
      : "+r"(pred));
  return pred;
}
DC_DEV void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// A tcgen05.ld of 16 registers a thread. 32x32b.x16: lane l of the warp gets row (32 (warp % 4) + l), 16 consecutive
// columns; 16x64b.x16, 16x128b.x8 and 16x256b.x4 are the other geometries of the same 512 words a warp.
#define DC_TMEM_LD16(NAME, SHAPE)                                                                                  \
  DC_DEV void NAME(uint32_t taddr, uint32_t (&r)[16]) {                                                            \
    asm volatile("tcgen05.ld.sync.aligned." SHAPE                                                                  \
                 ".b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"                            \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), \
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]),        \
                   "=r"(r[15])                                                                                     \
                 : "r"(taddr)                                                                                      \
                 : "memory");                                                                                      \
  }
DC_TMEM_LD16(tmem_ld16, "32x32b.x16")
DC_TMEM_LD16(tmem_ld16x64, "16x64b.x16")
DC_TMEM_LD16(tmem_ld16x128, "16x128b.x8")
#undef DC_TMEM_LD16
// tcgen05.st.32x32b.x16 (the store twin of tmem_ld16); tcgen05.wait::st before another thread reads what it wrote.
DC_DEV void tmem_st16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               ::"r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
               "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
               : "memory");
}
DC_DEV void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
// 32x32b.x4 and .x1: the narrow loads (4 and 1 words a thread), for the load's fixed cost a instruction.
DC_DEV void tmem_ld4(uint32_t taddr, uint32_t (&r)[4]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(taddr)
               : "memory");
}
DC_DEV void tmem_ld1(uint32_t taddr, uint32_t& r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r) : "r"(taddr) : "memory");
}
// x32 as two x16 halves would be two instructions; this is one.
DC_DEV void tmem_ld32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr)
      : "memory");
}
// 16x256b.x4: 16 lanes x 256 bits x 4 = 16 words a thread, the other load geometry (half the lanes, wider rows).
DC_DEV void tmem_ld16x256(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
}

// ---- the duo fold (floored_gemm/fgemm.cu's DUO; ref.py's twin) --------------------------------------------------------
// A word p at key position t: u = y ^ p ^ kx_t; y = u a_t + z; z = rotl(u, r_t) + y (LOP3, IMAD, LEA.HI or SHF + IADD3).
// Receipt word of a chain: y ^ rotl(z, 16). Keys are compile-time constants (constant-bank operands, no registers).
DC_HD constexpr uint32_t dc_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
constexpr int DUO_NPOS = 16;
DC_HD constexpr uint32_t duo_mul(int t) { return dc_mix32(0xA0000000u + 2u * (uint32_t)t) | 1u; }
DC_HD constexpr uint32_t duo_xor(int t) { return dc_mix32(0xB0000000u + 2u * (uint32_t)t + 1u); }
DC_HD constexpr int duo_rot(int t) { return ((t * 7 + 5) % 31) + 1; }
struct DuoSt { uint32_t y, z; };
DC_HD uint32_t dc_rotl(uint32_t x, int r) {
#if defined(__CUDA_ARCH__)
  return __funnelshift_l(x, x, r);
#else
  return (x << r) | (x >> (32 - r));
#endif
}
template <int T> DC_HD void duo(DuoSt& s, uint32_t p) {
  constexpr uint32_t a = duo_mul(T % DUO_NPOS), kx = duo_xor(T % DUO_NPOS);
  constexpr int r = duo_rot(T % DUO_NPOS);
  const uint32_t u = s.y ^ p ^ kx;
  s.y = u * a + s.z;
  s.z = dc_rotl(u, r) + s.y;
}
DC_HD uint32_t duo_finish(const DuoSt& s) { return s.y ^ dc_rotl(s.z, 16); }
DC_HD DuoSt duo_init(uint32_t seed, uint32_t idx) {
  DuoSt s;
  s.y = dc_mix32(seed ^ dc_mix32(idx));
  s.z = dc_mix32(s.y + 0x9E3779B9u);
  return s;
}
