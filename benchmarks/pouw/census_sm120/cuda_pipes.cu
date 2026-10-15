// CUDA-pipe census (item 2): warp instructions a SM a clock for each non-tensor op an adversary or the honest prover
// can use (FFMA, FADD, FMUL, HFMA2, BF16 FMA, IMAD, IADD3, LOP3, SHF, PRMT, POPC, DP4A, the FP8/FP4/BF16/FP16 packing
// converts, I2F, MUFU, min/max, BREV, FLO), alone and in 1:1 pairs (which pipes issue side by side). The `_c` variants
// take immediates for some operands: an all-register FFMA is register-bank bound, an FFMA with immediates is not.
//
// Each op updates eight registers in rotation, r_i = op(r_i, r_{i+1}, r_{i+2}), so no two consecutive ops have three or
// fewer distinct inputs and ptxas can't merge them (an IADD3 pair is two PTX adds on purpose). One block a SM of 32 warps;
// 16 rounds of 8 ops a loop iteration. `census.py` counts the SASS opcodes in each kernel's loop, so a rate is per SASS
// instruction ptxas actually emitted, with the loop's own overhead reported beside it.
#include "common.cuh"

#include <cstring>

#define F(x) __uint_as_float(x)
#define DEF_OP(NAME, CODE)                                                              \
  struct NAME {                                                                         \
    static constexpr const char* name = #NAME + 3;                                          \
    __device__ __forceinline__ static void op(uint32_t& x, uint32_t y, uint32_t z) { CODE; } \
  };

DEF_OP(op_ffma, float t = F(x); asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(t) : "f"(F(y)), "f"(F(z))); x = __float_as_uint(t))
DEF_OP(op_fadd, float t = F(x); asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(t) : "f"(F(y))); x = __float_as_uint(t))
DEF_OP(op_fmul, float t = F(x); asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(t) : "f"(F(y))); x = __float_as_uint(t))
DEF_OP(op_fmnmx, float t; asm volatile("max.f32 %0, %1, %2;" : "=f"(t) : "f"(F(y)), "f"(F(z))); x ^= __float_as_uint(t))
DEF_OP(op_hfma2, asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z)))
DEF_OP(op_bf16fma2, asm volatile("fma.rn.bf16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z)))
DEF_OP(op_imad, asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z)))
DEF_OP(op_iadd3, asm volatile("add.u32 %0, %0, %1;\n\tadd.u32 %0, %0, %2;" : "+r"(x) : "r"(y), "r"(z)))
DEF_OP(op_lop3, asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"(y), "r"(z)))
DEF_OP(op_shf, asm volatile("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z)))
DEF_OP(op_prmt, asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z)))
DEF_OP(op_popc, asm volatile("popc.b32 %0, %1;" : "=r"(x) : "r"(y ^ 0u)))
DEF_OP(op_dp4a, asm volatile("dp4a.u32.u32 %0, %1, %2, %0;" : "+r"(x) : "r"(y), "r"(z)))
DEF_OP(op_imnmx, asm volatile("max.u32 %0, %1, %2;" : "=r"(x) : "r"(y), "r"(z)))
DEF_OP(op_brev, asm volatile("brev.b32 %0, %1;" : "=r"(x) : "r"(y)))
DEF_OP(op_flo, asm volatile("clz.b32 %0, %1;" : "=r"(x) : "r"(y)))
DEF_OP(op_f2fp_e4m3x2,
       asm volatile("{.reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %1, %2; cvt.u32.u16 %0, t;}" : "=r"(x) : "f"(F(y)), "f"(F(z))))
DEF_OP(op_f2fp_e2m1x2,
       asm volatile("{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t;}" : "=r"(x) : "f"(F(y)), "f"(F(z))))
DEF_OP(op_f2fp_bf16x2, asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(x) : "f"(F(y)), "f"(F(z))))
DEF_OP(op_f2fp_f16x2, asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(x) : "f"(F(y)), "f"(F(z))))
DEF_OP(op_i2f, float t; asm volatile("cvt.rn.f32.u32 %0, %1;" : "=f"(t) : "r"(y)); x = __float_as_uint(t))
DEF_OP(op_mufu_ex2, float t; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(t) : "f"(F(y))); x = __float_as_uint(t))
// Constant-operand variants: the same pipes with fewer register-file reads (immediates instead of registers).
DEF_OP(op_ffma_c1, float t = F(x); asm volatile("fma.rn.f32 %0, %0, %1, 0f3F000000;" : "+f"(t) : "f"(F(y))); x = __float_as_uint(t))
DEF_OP(op_ffma_c2, float t = F(x); asm volatile("fma.rn.f32 %0, %0, 0f3F7FFFFE, 0f3F000000;" : "+f"(t)); x = __float_as_uint(t))
DEF_OP(op_imad_c, asm volatile("mad.lo.u32 %0, %0, %1, 0x9e3779b9;" : "+r"(x) : "r"(y)))
DEF_OP(op_iadd3_c, asm volatile("add.u32 %0, %0, %1;\n\tadd.u32 %0, %0, 0x01234567;" : "+r"(x) : "r"(y)))
DEF_OP(op_lop3_c, asm volatile("lop3.b32 %0, %0, %1, 0x5bd1e995, 0x96;" : "+r"(x) : "r"(y)))

constexpr int ROUNDS = 16;

template <class O1, class O2>
__global__ void __launch_bounds__(1024) k_pipe(const uint32_t* __restrict__ seed, int iters, uint32_t* __restrict__ sink,
                                                BlockClk* rec) {
  extern __shared__ uint32_t pad[];
  uint32_t r[8];
#pragma unroll
  for (int i = 0; i < 8; i++) r[i] = seed[(threadIdx.x * 8 + i) & 1023] ^ (threadIdx.x * 0x9e3779b9u);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int q = 0; q < ROUNDS; q++) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        if (i & 1) O2::op(r[i], r[(i + 1) & 7], r[(i + 2) & 7]);
        else O1::op(r[i], r[(i + 1) & 7], r[(i + 2) & 7]);
      }
    }
  }
  CLK_END(rec)
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) x += r[i];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

static uint32_t *g_seed, *g_sink;
static BlockClk* g_rec;
static int g_sms;

template <class O1, class O2>
static void measure(const char* only, int warps = 32) {
  char label[96];
  snprintf(label, sizeof label, "%s+%s", O1::name, O2::name);
  if (only && !strstr(label, only)) return;
  auto kern = k_pipe<O1, O2>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto go = [&](int iters) {
    return run_timed([&] { kern<<<g_sms, warps * 32, ONE_BLOCK_SMEM>>>(g_seed, iters, g_sink, g_rec); }, g_sms, g_rec);
  };
  Timing t0 = go(16);
  int iters = std::max(16, (int)(2e7 / (t0.cyc_med / 16.0)));
  Timing t = go(iters);
  double ops = (double)warps * iters * ROUNDS * 8;  // PTX-level ops a SM (census.py rescales by the SASS loop body)
  printf("{\"probe\":\"cuda_pipe\",\"op1\":\"%s\",\"op2\":\"%s\",\"warps\":%d,\"iters\":%d,\"rounds\":%d,\"ptx_ops_per_iter\":%d,",
         O1::name, O2::name, warps, iters, ROUNDS, ROUNDS * 8);
  print_timing(t);
  printf(",\"ptx_warp_ops_per_sm_per_clk\":%.4f,\"loop_cycles_per_iter_per_warp\":%.4f}\n", ops / t.cyc_med,
         t.cyc_med / iters);
  fflush(stdout);
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  g_sms = num_sms();
  CK(cudaMalloc(&g_seed, 1024 * 4));
  std::vector<uint32_t> h(1024);
  uint64_t s = 7;
  for (auto& w : h) w = (uint32_t)splitmix64(s) & 0x3FFFFFFFu;  // positive finite floats of every scale, as ints too
  CK(cudaMemcpy(g_seed, h.data(), 4096, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  measure<op_ffma, op_ffma>(only);
  measure<op_ffma, op_ffma>(only, 16);
  measure<op_fadd, op_fadd>(only);
  measure<op_fmul, op_fmul>(only);
  measure<op_fmnmx, op_fmnmx>(only);
  measure<op_hfma2, op_hfma2>(only);
  measure<op_bf16fma2, op_bf16fma2>(only);
  measure<op_imad, op_imad>(only);
  measure<op_iadd3, op_iadd3>(only);
  measure<op_lop3, op_lop3>(only);
  measure<op_shf, op_shf>(only);
  measure<op_prmt, op_prmt>(only);
  measure<op_popc, op_popc>(only);
  measure<op_dp4a, op_dp4a>(only);
  measure<op_imnmx, op_imnmx>(only);
  measure<op_brev, op_brev>(only);
  measure<op_flo, op_flo>(only);
  measure<op_f2fp_e4m3x2, op_f2fp_e4m3x2>(only);
  measure<op_f2fp_e2m1x2, op_f2fp_e2m1x2>(only);
  measure<op_f2fp_bf16x2, op_f2fp_bf16x2>(only);
  measure<op_f2fp_f16x2, op_f2fp_f16x2>(only);
  measure<op_i2f, op_i2f>(only);
  measure<op_mufu_ex2, op_mufu_ex2>(only);
  measure<op_ffma_c1, op_ffma_c1>(only);
  measure<op_ffma_c2, op_ffma_c2>(only);
  measure<op_imad_c, op_imad_c>(only);
  measure<op_iadd3_c, op_iadd3_c>(only);
  measure<op_lop3_c, op_lop3_c>(only);
  // 1:1 pairs: does the second op issue beside the first?
  measure<op_ffma, op_lop3>(only);
  measure<op_ffma, op_imad>(only);
  measure<op_ffma, op_iadd3>(only);
  measure<op_ffma, op_f2fp_e4m3x2>(only);
  measure<op_ffma, op_mufu_ex2>(only);
  measure<op_ffma, op_popc>(only);
  measure<op_ffma, op_dp4a>(only);
  measure<op_lop3, op_imad>(only);
  measure<op_lop3, op_iadd3>(only);
  measure<op_lop3, op_f2fp_e4m3x2>(only);
  measure<op_lop3, op_prmt>(only);
  measure<op_lop3, op_shf>(only);
  measure<op_imad, op_f2fp_e4m3x2>(only);
  measure<op_hfma2, op_ffma>(only);
  measure<op_ffma_c2, op_lop3_c>(only);
  measure<op_ffma_c2, op_imad_c>(only);
  measure<op_ffma_c2, op_iadd3_c>(only);
  measure<op_ffma_c2, op_f2fp_e4m3x2>(only);
  measure<op_ffma_c2, op_mufu_ex2>(only);
  measure<op_ffma_c1, op_lop3_c>(only);
  measure<op_imad_c, op_lop3_c>(only);
  measure<op_fadd, op_lop3_c>(only);
  printf("{\"probe\":\"cuda_pipe\",\"done\":true}\n");
  return 0;
}
