// The B200's fp4 tensor kinds alone, against kind::i8 in the same block:
// tcgen05 kind::mxf4 (UTCOMMA, E2M1 with UE8M0 block scales a 32-vector) and kind::mxf4nvf4 (UE4M3 a 16-vector), which
// sm_100's DS_tensor row prices nowhere, and kind::f8f6f4 at E2M1 beside E4M3: does FP4 in an 8-bit container
// run at FP8's rate or FP4's?
// Built and gated by ../opcode_probe/probe.py build --arch sm_100a; rates.sh with SRC=unpriced_fp4.cu is the job.
//   k_<kind>   512 threads; thread 0 of warp 0 issues 4 tcgen05.mma 128x256xK an iteration into one TMEM accumulator
//              (K 32 for i8 and f8f6f4, 64 for the packed fp4 kinds), the scales at 1.0 in TMEM beside it
// A rate is warp instructions a SM a clock; MACs a SM-clock are 128 x 256 x K x it.
#include "../opcode_probe/probe.cuh"
#include "../dc/dc.cuh"
#include "unpriced.cuh"

enum { K_I8, K_F8E4M3, K_F8E2M1, K_MXF4, K_NVF4 };
constexpr int KB = 32, AROWS = 128, BROWS = 256, TMEM_COLS = 512, SFA_COL = 256, SFB_COL = 384;

// Block-scaled instruction descriptor (kind::mxf4, mxf4nvf4): A and B E2M1 (1) at [7,10) and [10,13), N >> 3 at
// [17,23), scale type at 23 (UE4M3 0, UE8M0 1), M >> 4 at [24,29), both scale-factor ids 0; D is FP32 by the kind.
DC_HD constexpr uint32_t idesc_bs(uint32_t m, uint32_t n, uint32_t ue8m0) {
  return (1u << 7) | (1u << 10) | ((n >> 3) << 17) | (ue8m0 << 23) | ((m >> 4) << 24);
}
// kind::f8f6f4 formats at [7,10) and [10,13): E4M3 0, E2M1 5.
template <int K>
UP_DEV void kmma(uint32_t d, uint64_t da, uint64_t db, uint32_t acc) {
  if constexpr (K == K_I8) umma_i8(d, da, db, umma_idesc(2, 1, 1, AROWS, BROWS), acc);
  if constexpr (K == K_F8E4M3) umma_f8f6f4(d, da, db, umma_idesc(1, 0, 0, AROWS, BROWS), acc);
  if constexpr (K == K_F8E2M1) umma_f8f6f4(d, da, db, umma_idesc(1, 5, 5, AROWS, BROWS), acc);
  if constexpr (K == K_MXF4)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;\n}\n" ::"r"(d),
        "l"(da), "l"(db), "r"(idesc_bs(AROWS, BROWS, 1)), "r"(d + SFA_COL), "r"(d + SFB_COL), "r"(acc)
        : "memory");
  if constexpr (K == K_NVF4)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n}\n" ::"r"(d),
        "l"(da), "l"(db), "r"(idesc_bs(AROWS, BROWS, 0)), "r"(d + SFA_COL), "r"(d + SFB_COL), "r"(acc)
        : "memory");
}

struct Tc {
  unsigned char tile[(AROWS + BROWS) * KB];
  uint64_t bar;
  uint32_t tmem;
};
template <int K>
UP_DEV void k_alone(const OpArgs& a) {
  __shared__ __align__(1024) Tc s;
  uint32_t* wd = reinterpret_cast<uint32_t*>(s.tile);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x)
    wd[i] = a.mode == 1 ? 0u : dc_mix32(i * 0x9E3779B9u + 0x1234567u);
  if (threadIdx.x == 0) {
    mbar_init(&s.bar, 1);
    fence_mbar_init();
  }
  if (threadIdx.x < 32) tmem_alloc(&s.tmem, TMEM_COLS);
  fence_async_smem();
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = s.tmem, w = warp_u();
  // every scale 1.0 (UE8M0 0x7F, UE4M3 0x38), 32 columns each from SFA_COL and SFB_COL, all 128 lanes (warps 0-3)
  if (w < 4) {
    uint32_t r[16];
#pragma unroll
    for (int i = 0; i < 16; i++) r[i] = K == K_NVF4 ? 0x38383838u : 0x7F7F7F7Fu;
    const uint32_t row = tmem + ((32u * w) << 16);
#pragma unroll
    for (int u = 0; u < 2; u++) {
      tmem_st16(row + SFA_COL + 16 * u, r);
      tmem_st16(row + SFB_COL + 16 * u, r);
    }
    tmem_wait_st();
  }
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint64_t da = umma_desc(smem_u32(s.tile), kmaj_lbo(), kmaj_sbo(KB));
  const uint64_t db = umma_desc(smem_u32(s.tile) + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB));
  OP_BEGIN
  if (w == 0) {
    OP_LOOP(it) {
      if (elect_one()) {
        kmma<K>(tmem, da, db, 0);
        kmma<K>(tmem, da, db, 1);
        kmma<K>(tmem, da, db, 1);
        kmma<K>(tmem, da, db, 1);
      }
      __syncwarp();
    }
  }
  OP_END
  if (w == 0) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  uint32_t r[16], x = 0;
  tmem_ld16(tmem + ((32u * (w & 3u)) << 16), r);
  tmem_wait_ld();
#pragma unroll
  for (int i = 0; i < 16; i++) x ^= r[i];
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (threadIdx.x < 32) tmem_dealloc(tmem, TMEM_COLS);
  OP_SINK(x ^ threadIdx.x)
}

OP_KERNEL(k_i8, 512) { k_alone<K_I8>(a); }
OP_KERNEL(k_f8e4m3, 512) { k_alone<K_F8E4M3>(a); }
OP_KERNEL(k_f8e2m1, 512) { k_alone<K_F8E2M1>(a); }
OP_KERNEL(k_mxf4, 512) { k_alone<K_MXF4>(a); }
OP_KERNEL(k_nvf4, 512) { k_alone<K_NVF4>(a); }

#define P(NAME, PAT, NOTE) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = 512, .expect = PAT, .warps = "1", .iters = 2048, .note = NOTE}
static const OpProbe PROBES[] = {
    P(k_i8, "UTCIMMA*=4", "kind::i8 128x256x32 into one accumulator: the unit (unpriced.cu's tc_i8 uses two)"),
    P(k_f8e4m3, "UTCQMMA*=4", "kind::f8f6f4 E4M3 128x256x32"),
    P(k_f8e2m1, "UTCQMMA*=4", "kind::f8f6f4 E2M1 128x256x32 (one value a byte): FP8's rate or FP4's?"),
    P(k_mxf4, "UTCOMMA*=4", "kind::mxf4 E2M1 UE8M0 128x256x64 (packed)"),
    P(k_nvf4, "UTCOMMA*=4", "kind::mxf4nvf4 E2M1 UE4M3 128x256x64 (packed)"),
};
#undef P

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
