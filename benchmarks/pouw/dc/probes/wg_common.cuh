// The sm_90a probes' shared pieces: one shared-memory operand tile, the wgmma forms as types, and the commit group.
#pragma once
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"
#include "../wgmma.cuh"

constexpr int KB = 32;  // every form's K tile is 32 bytes of a row: s8/e4m3 k32, bf16 k16, tf32 k8
constexpr int AROWS = 64, BROWS = 256;

// Operand bytes: random codes valid for the type (mode 0), or zeros (mode 1).
template <int TY>
DC_DEV uint32_t operand_word(uint32_t i, int mode) {
  if (mode == 1) return 0u;
  const uint32_t x = dc_mix32(i * 0x9E3779B9u + 0x1234567u);
  if constexpr (TY == 0) return x;                                   // s8/u8: every code
  if constexpr (TY == 1) return x & 0x77777777u;                     // e4m3/e5m2: no NaN or inf code
  if constexpr (TY == 2) return (x & 0x807F807Fu) | 0x3F003F00u;     // bf16 in ±[0.5, 1)
  if constexpr (TY == 4) return (x & 0x83FF83FFu) | 0x38003800u;     // f16 in ±[0.5, 1)
  return (x & 0x007FFFFFu) | 0x3F800000u;                            // tf32 in [1, 2)
}

struct Tiles {
  uint64_t da, db;
};
// Fills A (64 rows) and B (256 rows) in the K-major layout; returns the descriptors. All threads call it.
template <int TY>
DC_DEV Tiles fill_tiles(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x) w[i] = operand_word<TY>(i, mode);
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB))};
}

#define FORM(NAME, ACC, NACC, TY)                                                                  \
  struct F_##NAME {                                                                               \
    using Acc = ACC;                                                                              \
    static constexpr int NA = NACC, TYPE = TY;                                                    \
    DC_DEV static void mma(Acc (&d)[NACC], const Tiles& t, int sd) { wgmma_##NAME(d, t.da, t.db, sd); } \
  };
FORM(s8_n8, int, 4, 0)
FORM(s8_n16, int, 8, 0)
FORM(s8_n32, int, 16, 0)
FORM(s8_n64, int, 32, 0)
FORM(s8_n128, int, 64, 0)
FORM(s8_n192, int, 96, 0)
FORM(s8_n256, int, 128, 0)
FORM(u8s8_n256, int, 128, 0)
FORM(e4m3_n128, float, 64, 1)
FORM(e4m3_n256, float, 128, 1)
FORM(e5m2_n256, float, 128, 1)
FORM(f16_n256, float, 128, 4)
FORM(bf16_n256, float, 128, 2)
FORM(bf16_n128, float, 64, 2)
FORM(tf32_n256, float, 128, 3)
FORM(b1_n256, int, 128, 0)
FORM(e4m3_n8, float, 4, 1)
FORM(e4m3_n16, float, 8, 1)
FORM(e4m3_n32, float, 16, 1)
FORM(e4m3_n64, float, 32, 1)
FORM(f16_n8, float, 4, 4)
FORM(f16_n64, float, 32, 4)
FORM(f16_n128, float, 64, 4)
FORM(bf16_n8, float, 4, 2)
FORM(bf16_n64, float, 32, 2)
FORM(tf32_n8, float, 4, 3)
FORM(tf32_n64, float, 32, 3)
FORM(tf32_n128, float, 64, 3)
FORM(b1_n8, int, 4, 0)
FORM(b1_n64, int, 32, 0)
FORM(b1_n128, int, 64, 0)
// FP16 accumulate: an f16x2 register holds two outputs
FORM(f16f16_n8, uint32_t, 2, 4)
FORM(f16f16_n256, uint32_t, 64, 4)
FORM(e4m3f16_n8, uint32_t, 2, 1)
FORM(e4m3f16_n256, uint32_t, 64, 1)
FORM(e5m2f16_n256, uint32_t, 64, 1)
#undef FORM
struct F_s8_n256_rs {  // A from registers: each thread's 4 words of the 64 x 32 A fragment
  using Acc = int;
  static constexpr int NA = 128, TYPE = 0;
  DC_DEV static void mma(int (&d)[128], const Tiles& t, int sd) {
    const uint32_t a[4] = {(uint32_t)t.da ^ threadIdx.x, (uint32_t)(t.da >> 32), (uint32_t)t.db, 0x01020304u};
    wgmma_s8_n256_rs(d, a, t.db, sd);
  }
};

template <class Acc, int N>
DC_DEV uint32_t fold_acc(const Acc (&d)[N]) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < N; i++) x ^= *reinterpret_cast<const uint32_t*>(&d[i]);
  return x;
}

// Four wgmma into one accumulator: the first overwrites, the rest accumulate.
template <class F>
DC_DEV void group4(typename F::Acc (&d)[F::NA], const Tiles& t) {
  wgmma_fence();
  F::mma(d, t, 0);
  F::mma(d, t, 1);
  F::mma(d, t, 1);
  F::mma(d, t, 1);
  wgmma_commit();
}

// A form's rate alone: 4 wgmma a commit group, then wait_group W.
template <class F, int W>
DC_DEV void wg_alone(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<F::TYPE>(sm, a.mode);
  typename F::Acc d[F::NA] = {};
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t);
    wgmma_wait<W>();
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ threadIdx.x)
}
