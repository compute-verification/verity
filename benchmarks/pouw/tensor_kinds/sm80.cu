// On sm_80 (A100): every native mma.sync form alone and beside a scalar lane, the scalar pipes' rates
// alone, in pairs and beside saturated IMMA, the duo fold beside IMMA, and a layout-free check of which forms leave the
// exact s32 word.
//
// The timed kernels are tensor_kinds.cu's (k_alone, k_spec, k_mixu, k_opalone) with sm_80's forms and lanes; k_duo is
// honest's inner loop (an MMA, then duo on each of its accumulator words). Every case prints one JSON line naming its
// kernel's template (`kern`, `struct`, `targs`), so sm80.py finds its SASS loops; gpu_kit's SASS timing gate runs on the
// binary before the lease, and the analysis reports no number for a refused kernel. Each case is first run once on a
// 0xA5-poisoned sink, which every thread must overwrite, and the no-write control must leave it poisoned (gk::poison_gate);
// NVML's SM clock is read before and after every timed rep.
//
// Build one binary per GROUP (0..11; 8..11 are M1c's extension and the shared-memory loads): nvcc -O3 -std=c++17
// -arch=sm_80 -DGROUP=g sm80.cu -ldl.
#include "../gpu_kit/bench.cuh"
#include "../receipts/receipt_folds.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef GROUP
#define GROUP 0
#endif

#define CK GK_CK

struct BlockClk {
  long long cyc;
  int smid;
  int pad;
};

#define CLK_BEGIN \
  __syncthreads(); \
  long long _t0 = clock64();
#define CLK_END(rec)                                   \
  __syncthreads();                                     \
  if (threadIdx.x == 0) {                              \
    rec[blockIdx.x].cyc = clock64() - _t0;             \
    int s_;                                            \
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s_));    \
    rec[blockIdx.x].smid = s_;                         \
  }

// Dynamic shared memory that keeps a second block off an SM (sm_80: 164 KB a SM, 163 KB a block opt-in).
constexpr int ONE_BLOCK_SMEM = 100 * 1024;

// ---- the forms ----------------------------------------------------------------------------------------------------
// Each form: its accumulator, the 32-bit accumulator registers a thread (NC), useful MACs, 32-bit destination slots and
// outputs a warp instruction, its SASS family (the gate's pattern), and for the exactness check its element code and
// useful products an output (KU). EX: 0 an FP accumulator, 1 s32 that wraps mod 2^32, 2 s32 with .satfinite.
enum { EL_NONE, EL_S8S8, EL_U8S8, EL_U8U8, EL_S4S4, EL_U4U4, EL_B1AND, EL_B1XOR };
#define A4 "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
#define A2 "r"(a[0]), "r"(a[1])
#define A1 "r"(a[0])
#define B4 "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3])
#define B2 "r"(b[0]), "r"(b[1])
#define B1 "r"(b[0])
#define D4I "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
#define D2I "+r"(d[0]), "+r"(d[1])
#define D4F "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])

#define FORM(S, NAME_, ACC, NC_, MACS_, SLOTS_, WORDS_, FAM_, W_, EX_, EL_, KU_) \
  struct S {                                                                   \
    static constexpr const char* SNAME = #S;                                   \
    static constexpr const char* NAME = NAME_;                                 \
    using Acc = ACC;                                                           \
    static constexpr int NC = NC_;                                             \
    static constexpr double MACS = MACS_, SLOTS = SLOTS_, WORDS = WORDS_;      \
    static constexpr const char* FAM = FAM_;                                   \
    static constexpr uint32_t W = W_;                                          \
    static constexpr int EX = EX_, EL = EL_, KU = KU_;                         \
    __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e)

// integer: s8/u8
FORM(ImmaS8k32, "imma_s8_m16n8k32", int32_t, 4, 16 * 8 * 32, 128, 128, "IMMA", 0x01020304u, 1, EL_S8S8, 32) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : D4I : A4, B2);
}};
FORM(ImmaU8S8k32, "imma_u8s8_m16n8k32", int32_t, 4, 16 * 8 * 32, 128, 128, "IMMA", 0x01020304u, 1, EL_U8S8, 32) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : D4I : A4, B2);
}};
FORM(ImmaU8U8k32, "imma_u8u8_m16n8k32", int32_t, 4, 16 * 8 * 32, 128, 128, "IMMA", 0x01020304u, 1, EL_U8U8, 32) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : D4I : A4, B2);
}};
FORM(ImmaS8k32Sat, "imma_s8_m16n8k32_sat", int32_t, 4, 16 * 8 * 32, 128, 128, "IMMA", 0x01020304u, 2, EL_S8S8, 32) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : D4I : A4, B2);
}};
FORM(ImmaS8k16, "imma_s8_m16n8k16", int32_t, 4, 16 * 8 * 16, 128, 128, "IMMA", 0x01020304u, 1, EL_S8S8, 16) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : D4I : A2, B1);
}};
FORM(ImmaS8m8k16, "imma_s8_m8n8k16", int32_t, 2, 8 * 8 * 16, 64, 64, "IMMA", 0x01020304u, 1, EL_S8S8, 16) {
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" : D2I : A1, B1);
}};
// integer: s4/u4
FORM(ImmaS4k64, "imma_s4_m16n8k64", int32_t, 4, 16 * 8 * 64, 128, 128, "IMMA", 0x12345671u, 1, EL_S4S4, 64) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : D4I : A4, B2);
}};
FORM(ImmaU4k64, "imma_u4_m16n8k64", int32_t, 4, 16 * 8 * 64, 128, 128, "IMMA", 0x12345671u, 1, EL_U4U4, 64) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : D4I : A4, B2);
}};
FORM(ImmaS4k32, "imma_s4_m16n8k32", int32_t, 4, 16 * 8 * 32, 128, 128, "IMMA", 0x12345671u, 1, EL_S4S4, 32) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : D4I : A2, B1);
}};
FORM(ImmaS4m8k32, "imma_s4_m8n8k32", int32_t, 2, 8 * 8 * 32, 64, 64, "IMMA", 0x12345671u, 1, EL_S4S4, 32) {
  asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" : D2I : A1, B1);
}};
// integer: b1 (AND and XOR with popc)
FORM(BmmaAndK256, "bmma_and_m16n8k256", int32_t, 4, 16 * 8 * 256, 128, 128, "BMMA", 0x5a5a5a5au, 1, EL_B1AND, 256) {
  asm volatile(
      "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : D4I : A4, B2);
}};
FORM(BmmaXorK256, "bmma_xor_m16n8k256", int32_t, 4, 16 * 8 * 256, 128, 128, "BMMA", 0x5a5a5a5au, 1, EL_B1XOR, 256) {
  asm volatile(
      "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : D4I : A4, B2);
}};
FORM(BmmaAndK128, "bmma_and_m16n8k128", int32_t, 4, 16 * 8 * 128, 128, 128, "BMMA", 0x5a5a5a5au, 1, EL_B1AND, 128) {
  asm volatile("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : D4I : A2, B1);
}};
FORM(BmmaAndM8K128, "bmma_and_m8n8k128", int32_t, 2, 8 * 8 * 128, 64, 64, "BMMA", 0x5a5a5a5au, 1, EL_B1AND, 128) {
  asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc {%0,%1}, {%2}, {%3}, {%0,%1};" : D2I : A1, B1);
}};
// integer: 2:4 sparse (MACS and KU count the useful, nonzero products)
FORM(ImmaSpS8k64, "imma_sp_s8_m16n8k64", int32_t, 4, 16 * 8 * 32, 128, 128, "IMMA", 0x01020304u, 1, EL_S8S8, 32) {
  asm volatile(
      "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%0,%1,%2,%3}, %12, 0x0;"
      : D4I : A4, B4, "r"(e));
}};
FORM(ImmaSpS8k32, "imma_sp_s8_m16n8k32", int32_t, 4, 16 * 8 * 16, 128, 128, "IMMA", 0x01020304u, 1, EL_S8S8, 16) {
  asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
               : D4I : A2, B2, "r"(e));
}};
FORM(ImmaSpS4k128, "imma_sp_s4_m16n8k128", int32_t, 4, 16 * 8 * 64, 128, 128, "IMMA", 0x12345671u, 1, EL_S4S4, 64) {
  asm volatile(
      "mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%0,%1,%2,%3}, %12, 0x0;"
      : D4I : A4, B4, "r"(e));
}};
FORM(ImmaSpS4k64, "imma_sp_s4_m16n8k64", int32_t, 4, 16 * 8 * 32, 128, 128, "IMMA", 0x12345671u, 1, EL_S4S4, 32) {
  asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
               : D4I : A2, B2, "r"(e));
}};
// FP16 / BF16 / TF32 (FP32 or f16 accumulators: no s32 word)
FORM(HmmaF16k16F32, "hmma_f16_m16n8k16_f32", float, 4, 16 * 8 * 16, 128, 128, "HMMA", 0x14001400u, 0, EL_NONE, 16) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : D4F : A4, B2);
}};
FORM(HmmaBf16k16F32, "hmma_bf16_m16n8k16_f32", float, 4, 16 * 8 * 16, 128, 128, "HMMA", 0x3e003e00u, 0, EL_NONE, 16) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : D4F : A4, B2);
}};
FORM(HmmaF16k16F16, "hmma_f16_m16n8k16_f16", uint32_t, 2, 16 * 8 * 16, 64, 128, "HMMA", 0x14001400u, 0, EL_NONE, 16) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" : D2I : A4, B2);
}};
FORM(HmmaF16k8F32, "hmma_f16_m16n8k8_f32", float, 4, 16 * 8 * 8, 128, 128, "HMMA", 0x14001400u, 0, EL_NONE, 8) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : D4F : A2, B1);
}};
FORM(HmmaBf16k8F32, "hmma_bf16_m16n8k8_f32", float, 4, 16 * 8 * 8, 128, 128, "HMMA", 0x3e003e00u, 0, EL_NONE, 8) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : D4F : A2, B1);
}};
FORM(HmmaF16k8F16, "hmma_f16_m16n8k8_f16", uint32_t, 2, 16 * 8 * 8, 64, 128, "HMMA", 0x14001400u, 0, EL_NONE, 8) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};" : D2I : A2, B1);
}};
FORM(HmmaTf32k8, "hmma_tf32_m16n8k8", float, 4, 16 * 8 * 8, 128, 128, "HMMA", 0x3a800000u, 0, EL_NONE, 8) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : D4F : A4, B2);
}};
FORM(HmmaTf32k4, "hmma_tf32_m16n8k4", float, 4, 16 * 8 * 4, 128, 128, "HMMA", 0x3a800000u, 0, EL_NONE, 4) {
  asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : D4F : A2, B1);
}};
FORM(HmmaSpF16k32, "hmma_sp_f16_m16n8k32_f32", float, 4, 16 * 8 * 16, 128, 128, "HMMA", 0x14001400u, 0, EL_NONE, 16) {
  asm volatile(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%0,%1,%2,%3}, %12, 0x0;"
      : D4F : A4, B4, "r"(e));
}};
FORM(HmmaSpF16k16, "hmma_sp_f16_m16n8k16_f32", float, 4, 16 * 8 * 8, 128, 128, "HMMA", 0x14001400u, 0, EL_NONE, 8) {
  asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
               : D4F : A2, B2, "r"(e));
}};
FORM(HmmaSpTf32k16, "hmma_sp_tf32_m16n8k16", float, 4, 16 * 8 * 8, 128, 128, "HMMA", 0x3a800000u, 0, EL_NONE, 8) {
  asm volatile(
      "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%0,%1,%2,%3}, %12, 0x0;"
      : D4F : A4, B4, "r"(e));
}};
// FP64: 64 outputs, each a double in two 32-bit slots
FORM(DmmaF64, "dmma_f64_m8n8k4", double, 2, 8 * 8 * 4, 128, 64, "DMMA", 0x3ff00000u, 0, EL_NONE, 4) {
  const double x = __hiloint2double(a[0], a[1]), y = __hiloint2double(b[0], b[1]);
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
               : "+d"(d[0]), "+d"(d[1]) : "d"(x), "d"(y));
}};

// Small operands of each form's type, so no accumulator chain of millions of MMAs reaches inf or NaN; the integer
// forms wrap. The sparse metadata 0x4444 4444 selects positions (0, 1) of every group of four.
template <class F>
__device__ __forceinline__ void init_ops(uint32_t lane, uint32_t* a, uint32_t* b, uint32_t& e) {
  const uint32_t w = F::W;
#pragma unroll
  for (int i = 0; i < 4; i++) { a[i] = w ^ (lane & 1u); b[i] = w; }
  if (sizeof(typename F::Acc) == 8) { a[0] = 0x3f50624du; a[1] = 0x0u ^ (lane & 1u); b[0] = 0x3ff00000u; b[1] = 0x0u; }
  e = 0x44444444u;
}

// ---- lane ops -------------------------------------------------------------------------------------------------------
// A lane program X is one op, or two ops alternating (X = P + 16 Q, Q = OP_NONE for one op). Each op runs on its own bank
// of 8 registers (a second op of the same file uses the second bank), so the 8 chains a bank give ILP 8.
enum { OP_IADD3 = 0, OP_FFMA = 1, OP_LOP3 = 2, OP_IMAD = 3, OP_DP4A = 4, OP_SHFL = 5, OP_POPC = 6, OP_LDS = 7, OP_IADD3R = 8,
       OP_LOP3I = 9, OP_LDSM4 = 10, OP_LDSM2 = 11, OP_LDSM1 = 12, OP_LDS128 = 13, OP_NONE = 15 };
constexpr int lane1(int p) { return p + 16 * OP_NONE; }
constexpr int lane2(int p, int q) { return p + 16 * q; }
constexpr int LANE_IADD3 = lane1(OP_IADD3), LANE_MIX = lane2(OP_IADD3, OP_FFMA);
constexpr bool is_ldsm(int op) { return op == OP_LDSM4 || op == OP_LDSM2 || op == OP_LDSM1; }

struct Lane {
  uint32_t r[8], q[8], l[8], m[8], v[8], k;
  float f[8], g[8];
};
// shared-memory regions (bytes) of the ldmatrix and LDS.128 chains; LDS.32's is at 0
constexpr uint32_t LDSM_OFF = 16384, V_OFF = 8192;

template <int OP>
__device__ __forceinline__ void op1(Lane& s, int bank, int i) {
  const int j = i & 7, n = (i + 1) & 7;
  uint32_t* r = bank ? s.q : s.r;
  float* f = bank ? s.g : s.f;
  // three inputs: sm_80's ptxas moves a two-input add.u32 to the FMA pipe as IMAD.IADD, but r + n + c is one IADD3
  // (OP_IADD3 with an immediate c, so two register reads as TENSOR's lane; OP_IADD3R with a third register)
  if (OP == OP_IADD3) asm volatile("add.u32 %0,%0,%1;\n\tadd.u32 %0,%0,0x9E3779B9;" : "+r"(r[j]) : "r"(r[n]));
  if (OP == OP_IADD3R) asm volatile("add.u32 %0,%0,%1;\n\tadd.u32 %0,%0,%2;" : "+r"(r[j]) : "r"(r[n]), "r"(s.k));
  // two register reads: majority with an immediate (ptxas merges chains of xor.b32, not of this)
  if (OP == OP_LOP3I) asm volatile("lop3.b32 %0,%0,%1,0x5A5A5A5A,0xE8;" : "+r"(r[j]) : "r"(r[n]));
  if (OP == OP_FFMA) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(f[j]) : "f"(f[n]));
  if (OP == OP_LOP3) asm volatile("lop3.b32 %0,%0,%1,%2,0xE8;" : "+r"(r[j]) : "r"(r[n]), "r"(s.k));
  if (OP == OP_IMAD) asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[j]) : "r"(s.k), "r"(r[n]));
  if (OP == OP_DP4A) asm volatile("dp4a.s32.s32 %0,%1,%2,%0;" : "+r"(r[j]) : "r"(r[n]), "r"(s.k));
  if (OP == OP_SHFL) asm volatile("shfl.sync.bfly.b32 %0,%1,1,0x1f,0xffffffff;" : "=r"(r[j]) : "r"(r[n]));
  if (OP == OP_POPC) asm volatile("popc.b32 %0,%1;" : "=r"(r[j]) : "r"(r[n]));
  if (OP == OP_LDS) asm volatile("ld.shared.u32 %0,[%0];" : "+r"(s.l[j]));
  // ldmatrix: lane L's register 0 is matrix 0's row L / 4, word L % 4, which holds L's own row address (init_r); the
  // other matrices' words are loaded and dropped. LDS.128: each lane's 16 bytes start with their own address, then 0s.
  uint32_t d1, d2, d3;
  if (OP == OP_LDSM4)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(s.m[j]), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(s.m[j]));
  if (OP == OP_LDSM2)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];" : "=r"(s.m[j]), "=r"(d1) : "r"(s.m[j]));
  if (OP == OP_LDSM1) asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0},[%1];" : "=r"(s.m[j]) : "r"(s.m[j]));
  // words 1-3 hold 0 and are folded into the address (ptxas narrows a v4 load whose other words are dead to LDS.32)
  if (OP == OP_LDS128)
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%0];\n\txor.b32 %0,%0,%1;\n\txor.b32 %0,%0,%2;\n\txor.b32 %0,%0,%3;"
                 : "+r"(s.v[j]), "=&r"(d1), "=&r"(d2), "=&r"(d3));
}
constexpr int q_bank(int p, int q) { return p == q || (p != OP_FFMA) == (q != OP_FFMA) ? 1 : 0; }
template <int X>
__device__ __forceinline__ void lane_op(Lane& s, int i) {
  constexpr int P = X & 15, Q = X >> 4;
  if (Q == OP_NONE) op1<P>(s, 0, i);
  else if (i & 1) op1<Q>(s, q_bank(P, Q), i >> 1);
  else op1<P>(s, 0, i >> 1);
}
// The banks program X writes (1 r, 2 q, 4 f, 8 g, 16 l, 32 m, 64 v), so the sink reads only those and no unused bank
// stays live.
enum { B_R = 1, B_Q = 2, B_F = 4, B_G = 8, B_L = 16, B_M = 32, B_V = 64 };
constexpr int bank_of(int op, int bank) {
  return op == OP_NONE ? 0 : op == OP_LDS ? B_L : is_ldsm(op) ? B_M : op == OP_LDS128 ? B_V
       : op == OP_FFMA ? (bank ? B_G : B_F) : (bank ? B_Q : B_R);
}
constexpr int banks(int x) { return bank_of(x & 15, 0) | ((x >> 4) == OP_NONE ? 0 : bank_of(x >> 4, q_bank(x & 15, x >> 4))); }
constexpr bool smem_op(int op) { return op == OP_LDS || is_ldsm(op) || op == OP_LDS128; }
template <int X>
constexpr bool uses_lds() { return smem_op(X & 15) || smem_op(X >> 4); }

extern __shared__ uint32_t g_smem[];
// LDS chains: chain j of a lane holds the shared address of word lane + 32 j, and that word holds its own address, so
// each load returns its address: one LDS an op, every lane of a warp on its own bank.
template <int X>
__device__ __forceinline__ void init_r(Lane& s, uint32_t tid) {
  const uint32_t lane = tid & 31;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(g_smem);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    s.r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
    s.q[i] = (tid * 0x85ebca6bu + i * 0xc2b2ae35u) | 1u;
    s.f[i] = 1.0f + (float)(s.r[i] & 0xffu) * 0x1p-12f;
    s.g[i] = 1.0f + (float)(s.q[i] & 0xffu) * 0x1p-12f;
    s.l[i] = base + 4 * (lane + 32 * i);
    s.m[i] = base + LDSM_OFF + 512 * i + 16 * lane;
    s.v[i] = base + V_OFF + 512 * i + 16 * lane;
  }
  s.k = (tid * 0x27d4eb2fu) | 1u;
  if (uses_lds<X>()) {
    if (tid < 256) {
      const uint32_t j = tid >> 5, l = tid & 31;
      g_smem[tid] = base + 4 * tid;
      g_smem[(LDSM_OFF + 512 * j + 4 * l) / 4] = base + LDSM_OFF + 512 * j + 16 * l;
      const uint32_t v = (V_OFF + 512 * j + 16 * l) / 4;
      g_smem[v] = base + V_OFF + 512 * j + 16 * l;
      g_smem[v + 1] = g_smem[v + 2] = g_smem[v + 3] = 0;
    }
    __syncthreads();
  }
}
template <int X>
__device__ __forceinline__ uint32_t consume(const Lane& s) {
  constexpr int B = banks(X);
  uint32_t x = s.k;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    if (B & B_R) x += s.r[i];
    if (B & B_Q) x ^= s.q[i];
    if (B & B_L) x += s.l[i];
    if (B & B_M) x += s.m[i];
    if (B & B_V) x ^= s.v[i];
    if (B & B_F) x ^= __float_as_uint(s.f[i]);
    if (B & B_G) x += __float_as_uint(s.g[i]);
  }
  return x;
}

template <class F>
__device__ __forceinline__ uint32_t fold_acc(typename F::Acc (*d)[4]) {
  uint32_t x = 0;
#pragma unroll
  for (int c = 0; c < 8; c++)
#pragma unroll
    for (int j = 0; j < F::NC; j++) {
      typename F::Acc v = d[c][j];
      uint32_t w[sizeof(v) / 4];
      memcpy(w, &v, sizeof v);
#pragma unroll
      for (int k = 0; k < (int)(sizeof(v) / 4); k++) x ^= w[k];
    }
  return x;
}

// ---- kernels ----------------------------------------------------------------------------------------------------
// A lane program alone at 8 warps a SMSP (ISSUE's saturation geometry), 64 ops an iteration.
template <int X>
__global__ void __launch_bounds__(1024) k_opalone(int iters, uint32_t* sink, BlockClk* rec) {
  Lane r;
  init_r<X>(r, threadIdx.x);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int q = 0; q < 64; q++) lane_op<X>(r, q);
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + threadIdx.x] = threadIdx.x + consume<X>(r);
}

// MMA alone: NW warps a SMSP, 8 independent accumulators a warp, 8U MMAs a loop iteration (U > 1 thins the loop's
// 3 instructions a 8 MMAs, so the pipe's own rate shows).
template <class F, int NW, int U>
__global__ void __launch_bounds__(128 * NW) k_alone(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, a, b, e);
  typename F::Acc d[8][4] = {};
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int c = 0; c < 8; c++) F::mma(d[c], a, b, e);
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + threadIdx.x] = fold_acc<F>(d) ^ lane;
}

// Warp-specialized: 16 warps a SM; SMSP r hosts warps {r, r+4} (MMA) and {r+8, r+12} (lane); R/8 ops a MMA.
template <class F, int R, int X>
__global__ void __launch_bounds__(512, 1) k_spec(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool mma_role = warp < (blockDim.x >> 6);
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, a, b, e);
  typename F::Acc d[8][4] = {};
  Lane r;
  init_r<X>(r, threadIdx.x);
  CLK_BEGIN
  if (mma_role) {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int c = 0; c < 8; c++) F::mma(d[c], a, b, e);
    }
  } else {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int q = 0; q < R; q++) lane_op<X>(r, q);
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
  if (mma_role) x ^= fold_acc<F>(d);
  else x += consume<X>(r);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// Interleaved: every warp issues a MMA then R/8 lane ops, 8 accumulators, U groups of 8 MMAs an iteration (mixu_u(R)
// unless the sweep sets it). A loop of more than ~200 instructions makes ptxas exit it through CALL.REL.NOINC, which
// hides its data flow from the gate.
constexpr int mixu_u(int r) { return 192 / (8 + r) > 8 ? 8 : 192 / (8 + r) < 1 ? 1 : 192 / (8 + r); }
template <class F, int R, int X, int NW, int U>
__global__ void __launch_bounds__(128 * NW, 1) k_mixu(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, a, b, e);
  typename F::Acc d[8][4] = {};
  Lane r;
  init_r<X>(r, threadIdx.x);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < U; u++) {
#pragma unroll
      for (int c = 0; c < 8; c++) {
        F::mma(d[c], a, b, e);
#pragma unroll
        for (int q = 0; q < R / 8; q++) lane_op<X>(r, c * (R / 8) + q);
      }
    }
  }
  CLK_END(rec)
  uint32_t x = (lane ^ fold_acc<F>(d)) + consume<X>(r);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// Honest's inner loop at L = the form's depth: each MMA's accumulator words, the running s32 prefix, are folded by duo
// (receipt_folds.cuh) as soon as it lands, entry j of MMA c into chain 4 (c % 2) + j as served-duo does; NW warps a SMSP.
// k_duofold folds loop-invariant words on the same chain schedule with no MMA, so it times duo alone.
__device__ __forceinline__ constexpr int duo_rot(int j) { return j == 0 ? 5 : j == 1 ? 11 : j == 2 ? 17 : 29; }
template <class F, int NW>
__global__ void __launch_bounds__(128 * NW) k_duo(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, a, b, e);
  typename F::Acc d[8][4] = {};
  uint32_t y[8], z[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    y[i] = rf_mix32(threadIdx.x * 8 + i);
    z[i] = rf_mix32(y[i] + 0x9E3779B9u);
  }
  const uint32_t ka0 = 0x2545F491u, ka1 = 0x9E3779B1u, kx0 = rf_mix32(lane + 1), kx1 = rf_mix32(lane + 77);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      F::mma(d[c], a, b, e);
#pragma unroll
      for (int j = 0; j < F::NC; j++) {
        const int ch = 4 * (c & 1) + j;
        rf_duo_word(y[ch], z[ch], (uint32_t)d[c][j], j & 1 ? kx1 : kx0, j & 1 ? ka1 : ka0, duo_rot(j));
      }
    }
  }
  CLK_END(rec)
  uint32_t x = fold_acc<F>(d) ^ lane;
#pragma unroll
  for (int i = 0; i < 8; i++) x += rf_duo_final(y[i], z[i]);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
template <int NW>
__global__ void __launch_bounds__(128 * NW) k_duofold(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t w[8], y[8], z[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    w[i] = rf_mix32(threadIdx.x * 31 + i);
    y[i] = rf_mix32(threadIdx.x * 8 + i);
    z[i] = rf_mix32(y[i] + 0x9E3779B9u);
  }
  const uint32_t ka0 = 0x2545F491u, ka1 = 0x9E3779B1u, kx0 = rf_mix32(lane + 1), kx1 = rf_mix32(lane + 77);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int ch = 4 * (c & 1) + j;
        rf_duo_word(y[ch], z[ch], w[(c * 4 + j) & 7], j & 1 ? kx1 : kx0, j & 1 ? ka1 : ka0, duo_rot(j));
      }
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
#pragma unroll
  for (int i = 0; i < 8; i++) x += rf_duo_final(y[i], z[i]);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// One MMA on constant operands (every A element a, every B element b) and per-element random C: D = C + KU a b in
// every element whatever the fragment layout, so the check needs no layout. One warp.
template <class F>
__global__ void k_exact(uint32_t aw, uint32_t bw, const uint32_t* c, uint32_t* out) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4] = {aw, aw, aw, aw}, b[4] = {bw, bw, bw, bw};
  typename F::Acc d[4];
#pragma unroll
  for (int j = 0; j < F::NC; j++) d[j] = (typename F::Acc)c[lane * F::NC + j];
  F::mma(d, a, b, 0x44444444u);
#pragma unroll
  for (int j = 0; j < F::NC; j++) out[lane * F::NC + j] = (uint32_t)d[j];
}

// ---- host ---------------------------------------------------------------------------------------------------------
static uint32_t* g_sink;
static BlockClk* g_rec;
static int g_sms;
static gk::Nvml* g_nv;
static cudaStream_t g_s;
static const int WSM = 16;  // warps a SM in k_spec and k_mixu, 4 a SMSP

struct Timing {
  double ms = 0, cyc_med = 0, cyc_min = 0, cyc_max = 0, mhz = 0;
  int blocks = 0, distinct_sms = 0;
  unsigned nvml_before = 0, nvml_after = 0;
  unsigned long long throttle = 0;
};

template <class F>
Timing run_timed(F launch, int nblocks, int reps = 5) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  launch();
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  Timing best;
  best.cyc_med = 1e300;
  std::vector<BlockClk> h(nblocks);
  for (int r = 0; r < reps; r++) {
    const gk::Sample s0 = g_nv->sample();
    CK(cudaEventRecord(e0, g_s));
    launch();
    CK(cudaEventRecord(e1, g_s));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    const gk::Sample s1 = g_nv->sample();
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaMemcpy(h.data(), g_rec, nblocks * sizeof(BlockClk), cudaMemcpyDeviceToHost));
    std::vector<double> c;
    std::vector<int> s;
    for (auto& b : h) { c.push_back((double)b.cyc); s.push_back(b.smid); }
    std::sort(c.begin(), c.end());
    std::sort(s.begin(), s.end());
    Timing t;
    t.ms = ms;
    t.blocks = nblocks;
    t.distinct_sms = (int)(std::unique(s.begin(), s.end()) - s.begin());
    t.cyc_med = c[c.size() / 2];
    t.cyc_min = c.front();
    t.cyc_max = c.back();
    t.mhz = ms > 0 ? t.cyc_max / (ms * 1e3) : 0;
    t.nvml_before = s0.sm_mhz;
    t.nvml_after = s1.sm_mhz;
    t.throttle = s0.throttle | s1.throttle;
    if (t.cyc_med < best.cyc_med) best = t;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return best;
}

static int size_iters(double cyc_warm, int warm) { return std::max(64, (int)(2e7 / (cyc_warm / warm))); }

// gk::poison_gate on the sink: one iteration's launch must overwrite every thread's word, and the control (no launch)
// must leave the 0xA5 poison, which the judge rejects.
template <class K>
static bool sink_gate(K kern, int threads) {
  const size_t words = (size_t)g_sms * threads, bytes = words * 4;
  auto launch = [&](cudaStream_t st) { kern<<<g_sms, threads, ONE_BLOCK_SMEM, st>>>(1, g_sink, g_rec); };
  auto accept = [&](const void* h, size_t n) { return gk::poisoned_words(h, n) == 0; };
  gk::PoisonGate g = gk::poison_gate(g_sink, bytes, g_s, launch, accept);
  CK(cudaGetLastError());
  return g.ok();
}

template <class K>
static Timing timed(K kern, int threads, int* iters_out, bool* sink_ok) {
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  *sink_ok = sink_gate(kern, threads);
  auto launch = [&](int it) {
    return run_timed([&] { kern<<<g_sms, threads, ONE_BLOCK_SMEM, g_s>>>(it, g_sink, g_rec); }, g_sms);
  };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  *iters_out = iters;
  return launch(iters);
}

// mma_w/lane_w: warps a SMSP in each role; mma_ptx/lane_ops: PTX MMAs and lane ops a warp a loop iteration
static void emit(const char* mode, const char* kern, const char* sname, const char* form, std::vector<int> targs,
                 double macs, double slots, double words, int mma_w, int lane_w, int mma_ptx, int lane_ops, int X,
                 int iters, bool sink_ok, const Timing& t) {
  printf("{\"probe\":\"tensor_kinds_sm80\",\"mode\":\"%s\",\"kern\":\"%s\",\"struct\":\"%s\",\"form\":\"%s\",\"targs\":[",
         mode, kern, sname, form);
  for (size_t i = 0; i < targs.size(); i++) printf("%s%d", i ? "," : "", targs[i]);
  printf("],\"macs\":%.0f,\"slots\":%.0f,\"words\":%.0f,\"mma_warps_smsp\":%d,\"lane_warps_smsp\":%d,\"mma_ptx_iter\":%d,"
         "\"lane_ops_iter\":%d,\"lane\":%d,\"iters\":%d,\"sink_gate\":%s,\"cyc_med\":%.0f,\"cyc_min\":%.0f,\"cyc_max\":%.0f,"
         "\"distinct_sms\":%d,\"ms\":%.4f,\"mhz\":%.1f,\"nvml_mhz_before\":%u,\"nvml_mhz_after\":%u,\"throttle\":\"0x%llx\"}\n",
         macs, slots, words, mma_w, lane_w, mma_ptx, lane_ops, X, iters, sink_ok ? "true" : "false", t.cyc_med, t.cyc_min,
         t.cyc_max, t.distinct_sms, t.ms, t.mhz, t.nvml_before, t.nvml_after, t.throttle);
  fflush(stdout);
}

template <int X>
static void opalone() {
  int iters;
  bool ok;
  Timing t = timed(k_opalone<X>, 1024, &iters, &ok);
  emit("opalone", "k_opalone", "", "", {X}, 0, 0, 0, 0, 8, 0, 64, X, iters, ok, t);
}
template <class F, int NW, int U = 1>
static void alone() {
  int iters;
  bool ok;
  Timing t = timed(k_alone<F, NW, U>, 4 * NW * 32, &iters, &ok);
  char mode[16];
  snprintf(mode, sizeof mode, NW == 2 && U == 1 ? "alone" : U == 1 ? "alone%d" : "alone%du%d", NW, U);
  emit(mode, "k_alone", F::SNAME, F::NAME, {NW, U}, F::MACS, F::SLOTS, F::WORDS, NW, 0, 8 * U, 0, -1, iters, ok, t);
}
template <class F, int R, int X>
static void spec() {
  int iters;
  bool ok;
  Timing t = timed(k_spec<F, R, X>, WSM * 32, &iters, &ok);
  emit("spec", "k_spec", F::SNAME, F::NAME, {R, X}, F::MACS, F::SLOTS, F::WORDS, 2, 2, 8, R, X, iters, ok, t);
}
template <class F, int R, int X, int NW = 4, int U = mixu_u(R)>
static void mixu() {
  int iters;
  bool ok;
  Timing t = timed(k_mixu<F, R, X, NW, U>, 4 * NW * 32, &iters, &ok);
  emit("mixu", "k_mixu", F::SNAME, F::NAME, {R, X, NW, U}, F::MACS, F::SLOTS, F::WORDS, NW, 0, 8 * U, U * R, X, iters, ok,
       t);
}
template <class F, int NW>
static void duo() {
  int iters;
  bool ok;
  Timing t = timed(k_duo<F, NW>, 4 * NW * 32, &iters, &ok);
  emit("duo", "k_duo", F::SNAME, F::NAME, {NW}, F::MACS, F::SLOTS, F::WORDS, NW, 0, 8, 0, -1, iters, ok, t);
}
template <int NW>
static void duofold() {
  int iters;
  bool ok;
  Timing t = timed(k_duofold<NW>, 4 * NW * 32, &iters, &ok);
  emit("duofold", "k_duofold", "", "", {NW}, 0, 0, 128, 0, NW, 0, 32, -1, iters, ok, t);
}

static uint32_t pack(int el, int v, bool a_side) {
  switch (el) {
    case EL_S8S8: return (uint32_t)(v & 0xff) * 0x01010101u;
    case EL_U8S8: return (uint32_t)(v & 0xff) * 0x01010101u;
    case EL_U8U8: return (uint32_t)(v & 0xff) * 0x01010101u;
    case EL_S4S4: case EL_U4U4: return (uint32_t)(v & 0xf) * 0x11111111u;
    case EL_B1AND: case EL_B1XOR: return v ? 0xffffffffu : 0u;
  }
  (void)a_side;
  return 0;
}
// (a, b) pairs at each element type's extremes; b1 values are single bits
static std::vector<std::pair<int, int>> exact_pairs(int el) {
  switch (el) {
    case EL_S8S8: return {{-128, -128}, {127, -128}, {127, 127}, {-1, 1}, {3, -5}};
    case EL_U8S8: return {{255, -128}, {255, 127}, {1, -1}};
    case EL_U8U8: return {{255, 255}, {200, 3}};
    case EL_S4S4: return {{-8, -8}, {7, -8}, {7, 7}, {-1, 3}};
    case EL_U4U4: return {{15, 15}, {9, 2}};
    case EL_B1AND: case EL_B1XOR: return {{1, 1}, {1, 0}, {0, 0}};
  }
  return {};
}

template <class F>
static void exact_int() {
  const int n = 32 * F::NC;
  uint32_t *dc, *dd;
  CK(cudaMalloc(&dc, n * 4));
  CK(cudaMalloc(&dd, n * 4));
  std::vector<uint32_t> hc(n), hd(n);
  uint64_t st = 0x9E3779B97F4A7C15ull ^ (uint64_t)F::KU;
  for (auto [a, b] : exact_pairs(F::EL)) {
    for (int i = 0; i < n; i++) {
      st = st * 6364136223846793005ull + 1442695040888963407ull;
      uint32_t v = (uint32_t)(st >> 32);
      if (i % 4 == 1) v = 0x7fffffffu - (v & 0xffffu);  // within 2^16 of INT_MAX: the products wrap or saturate
      if (i % 4 == 2) v = 0x80000000u + (v & 0xffffu);  // within 2^16 of INT_MIN
      hc[i] = v;
    }
    CK(cudaMemcpy(dc, hc.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dd, gk::POISON, n * 4));
    k_exact<F><<<1, 32>>>(pack(F::EL, a, true), pack(F::EL, b, false), dc, dd);
    CK(cudaGetLastError());
    CK(cudaMemcpy(hd.data(), dd, n * 4, cudaMemcpyDeviceToHost));
    const long long prod = F::EL == EL_B1AND ? (a & b) : F::EL == EL_B1XOR ? (a ^ b) : (long long)a * b;
    const long long inc = (long long)F::KU * prod;
    int wrong = 0, wrapped = 0, saturated = 0;
    for (int i = 0; i < n; i++) {
      const long long exact64 = (long long)(int32_t)hc[i] + inc;
      const uint32_t wrap = (uint32_t)(hc[i] + (uint32_t)inc);
      const long long sat = std::min<long long>(INT32_MAX, std::max<long long>(INT32_MIN, exact64));
      const uint32_t want = F::EX == 2 ? (uint32_t)(int32_t)sat : wrap;
      wrong += hd[i] != want;
      wrapped += exact64 != (long long)(int32_t)wrap;
      saturated += F::EX == 2 && sat != exact64;
    }
    printf("{\"probe\":\"tensor_kinds_sm80\",\"mode\":\"exact\",\"struct\":\"%s\",\"form\":\"%s\",\"a\":%d,\"b\":%d,"
           "\"ku\":%d,\"increment\":%lld,\"elements\":%d,\"wrong\":%d,\"wrapped\":%d,\"saturated\":%d,\"semantics\":\"%s\"}\n",
           F::SNAME, F::NAME, a, b, F::KU, inc, n, wrong, wrapped, saturated, F::EX == 2 ? "satfinite" : "wrap");
  }
  CK(cudaFree(dc));
  CK(cudaFree(dd));
}
template <class F>
static void exact() {
  if constexpr (F::EX != 0) exact_int<F>();
}

// The least bound moves with the loop's length (IMMA.16832 at 8 warps: 5.16 at R = 32, 4.54 at 48, 5.19 at 64), so R
// and U each vary, every loop under ~200 instructions.
template <class F, int NW>
static void sweep() {
  mixu<F, 24, LANE_MIX, NW, 2>(); mixu<F, 24, LANE_MIX, NW, 3>(); mixu<F, 24, LANE_MIX, NW, 4>();
  mixu<F, 32, LANE_MIX, NW, 2>(); mixu<F, 32, LANE_MIX, NW, 3>(); mixu<F, 40, LANE_MIX, NW, 2>();
  mixu<F, 40, LANE_MIX, NW, 3>(); mixu<F, 48, LANE_MIX, NW, 2>(); mixu<F, 56, LANE_MIX, NW, 2>();
  mixu<F, 56, LANE_MIX, NW, 3>(); mixu<F, 64, LANE_MIX, NW, 1>(); mixu<F, 72, LANE_MIX, NW, 1>();
  mixu<F, 72, LANE_MIX, NW, 2>(); mixu<F, 80, LANE_MIX, NW, 1>(); mixu<F, 80, LANE_MIX, NW, 2>();
  mixu<F, 96, LANE_MIX, NW, 1>();
}

// The points sweep() and form() leave out (moved IMMA.SP.16864 3.9% at 7 warps a SMSP)
template <class F, int NW>
static void sweep2() {
  mixu<F, 16, LANE_MIX, NW, 4>(); mixu<F, 16, LANE_MIX, NW, 6>(); mixu<F, 24, LANE_MIX, NW, 5>();
  mixu<F, 88, LANE_MIX, NW, 1>(); mixu<F, 88, LANE_MIX, NW, 2>();
  if constexpr (NW == 5 || NW == 7) {
    mixu<F, 16, LANE_MIX, NW>(); mixu<F, 32, LANE_MIX, NW>(); mixu<F, 40, LANE_MIX, NW>();
  }
}

template <class F>
static void form() {
  exact<F>();
  alone<F, 2>();
  alone<F, 4>();
  alone<F, 8>();
  alone<F, 2, 4>();
  alone<F, 4, 4>();
  alone<F, 8, 4>();
  alone<F, 2, 8>();
  alone<F, 4, 8>();
  alone<F, 8, 8>();
  spec<F, 8, LANE_MIX>(); spec<F, 16, LANE_MIX>(); spec<F, 32, LANE_MIX>(); spec<F, 64, LANE_MIX>(); spec<F, 128, LANE_MIX>();
  mixu<F, 8, LANE_MIX>(); mixu<F, 16, LANE_MIX>(); mixu<F, 32, LANE_MIX>(); mixu<F, 64, LANE_MIX>(); mixu<F, 128, LANE_MIX>();
  // IADD3 alone binds on its pipe; these programs still bound any joint row, which must hold on every program
  spec<F, 8, LANE_IADD3>(); spec<F, 16, LANE_IADD3>(); spec<F, 32, LANE_IADD3>(); spec<F, 64, LANE_IADD3>();
  spec<F, 128, LANE_IADD3>();
  mixu<F, 8, LANE_IADD3>(); mixu<F, 16, LANE_IADD3>(); mixu<F, 32, LANE_IADD3>(); mixu<F, 64, LANE_IADD3>();
  mixu<F, 128, LANE_IADD3>();
  // the interleaved schedule at 2, 6 and 8 warps a SMSP, where M1's least bounds sit at 4
  mixu<F, 32, LANE_MIX, 2>(); mixu<F, 64, LANE_MIX, 2>(); mixu<F, 32, LANE_MIX, 8>(); mixu<F, 64, LANE_MIX, 8>();
  mixu<F, 16, LANE_IADD3, 2>(); mixu<F, 32, LANE_IADD3, 2>(); mixu<F, 16, LANE_IADD3, 8>(); mixu<F, 32, LANE_IADD3, 8>();
  mixu<F, 8, LANE_MIX, 8>(); mixu<F, 16, LANE_MIX, 8>(); mixu<F, 24, LANE_MIX, 8>(); mixu<F, 40, LANE_MIX, 8>();
  mixu<F, 48, LANE_MIX, 8>(); mixu<F, 16, LANE_MIX, 6>(); mixu<F, 32, LANE_MIX, 6>(); mixu<F, 8, LANE_IADD3, 8>();
  sweep<F, 6>();
  sweep<F, 8>();
  // the search's extension (its margin is the spread once two successive extensions agree within 2%)
  sweep<F, 5>();
  sweep<F, 7>();
  sweep2<F, 5>();
  sweep2<F, 6>();
  sweep2<F, 7>();
  sweep2<F, 8>();
}

// A scalar class beside a saturated MMA (2 MMA + 2 lane warps a SMSP, 8 ops a MMA): its coupling with the tensor pipe
template <class F>
static void beside() {
  spec<F, 64, lane1(OP_LOP3)>(); spec<F, 64, lane1(OP_IMAD)>(); spec<F, 64, lane1(OP_DP4A)>();
  spec<F, 64, lane1(OP_FFMA)>(); spec<F, 64, lane1(OP_SHFL)>(); spec<F, 64, lane1(OP_POPC)>();
  spec<F, 64, lane1(OP_LDS)>(); spec<F, 64, lane2(OP_LOP3, OP_IMAD)>(); spec<F, 64, lane1(OP_LOP3I)>();
  spec<F, 64, lane2(OP_LOP3I, OP_FFMA)>(); spec<F, 64, lane2(OP_IADD3, OP_IMAD)>();
}

// M1c on sm_80 (the search's second extension, after facet_break beat sm_120's joint rows with an f16 MMA): each
// form that sets a joint coefficient, beside the lane mixes M1 left out (other pipes, register reads, the operand loads),
// interleaved at three densities and 4, 6 and 8 warps a SMSP, and warp-specialized; M1's IADD3+FFMA least as control.
template <class F, int X>
static void search_lane() {
  mixu<F, 16, X, 4>(); mixu<F, 32, X, 4>(); mixu<F, 64, X, 4>();
  mixu<F, 16, X, 8>(); mixu<F, 32, X, 8>(); mixu<F, 64, X, 8>();
  mixu<F, 24, X, 6>(); mixu<F, 48, X, 6>();
  spec<F, 32, X>(); spec<F, 64, X>();
}
template <class F>
static void search() {
  alone<F, 4>(); alone<F, 8, 4>();
  mixu<F, 48, LANE_MIX, 8, 2>(); mixu<F, 32, LANE_MIX, 8, 3>(); mixu<F, 40, LANE_MIX, 6, 2>();
  search_lane<F, lane1(OP_LOP3)>();
  search_lane<F, lane1(OP_LOP3I)>();
  search_lane<F, lane2(OP_LOP3, OP_IMAD)>();
  search_lane<F, lane2(OP_LOP3, OP_FFMA)>();
  search_lane<F, lane2(OP_LOP3I, OP_FFMA)>();
  search_lane<F, lane2(OP_IADD3, OP_IMAD)>();
  search_lane<F, lane2(OP_IMAD, OP_FFMA)>();
  search_lane<F, lane2(OP_DP4A, OP_LOP3)>();
  search_lane<F, lane2(OP_LDSM4, OP_FFMA)>();
  search_lane<F, lane2(OP_LDS128, OP_IADD3)>();
}

int main() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  CK(cudaStreamCreate(&g_s));
  gk::Nvml nv(dev);
  g_nv = &nv;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));
  printf("{\"probe\":\"tensor_kinds_sm80\",\"group\":%d,\"device\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"gpu_uuid\":\"%s\","
         "\"nvml\":%s}\n",
         GROUP, p.name, p.major, p.minor, g_sms, nv.uuid().c_str(), nv.ok() ? "true" : "false");
#if GROUP == 0
  // the scalar pipes alone, in pairs (do two classes share a pipe?), duo alone and beside the honest MMAs
  opalone<lane1(OP_IADD3)>(); opalone<lane1(OP_FFMA)>(); opalone<lane1(OP_LOP3)>(); opalone<lane1(OP_IMAD)>();
  opalone<lane1(OP_DP4A)>(); opalone<lane1(OP_SHFL)>(); opalone<lane1(OP_POPC)>(); opalone<lane1(OP_LDS)>();
  opalone<LANE_MIX>(); opalone<lane2(OP_IADD3, OP_LOP3)>(); opalone<lane2(OP_IADD3, OP_IMAD)>();
  opalone<lane2(OP_IMAD, OP_FFMA)>(); opalone<lane2(OP_LOP3, OP_FFMA)>(); opalone<lane2(OP_DP4A, OP_FFMA)>();
  opalone<lane2(OP_DP4A, OP_IADD3)>(); opalone<lane2(OP_POPC, OP_FFMA)>(); opalone<lane2(OP_POPC, OP_IADD3)>();
  opalone<lane2(OP_SHFL, OP_FFMA)>(); opalone<lane2(OP_LDS, OP_FFMA)>(); opalone<lane2(OP_LDS, OP_IADD3)>();
  opalone<lane2(OP_LOP3, OP_IMAD)>();
  // register reads: IADD3 with a third register, and LOP3 with two (an immediate third), alone and beside FFMA
  opalone<lane1(OP_IADD3R)>(); opalone<lane1(OP_LOP3I)>(); opalone<lane2(OP_IADD3R, OP_FFMA)>();
  opalone<lane2(OP_LOP3I, OP_FFMA)>(); opalone<lane2(OP_LOP3I, OP_IMAD)>(); opalone<lane2(OP_IADD3, OP_LOP3I)>();
  duofold<2>(); duofold<4>(); duofold<8>();
  duo<ImmaS8k32, 2>(); duo<ImmaS8k32, 4>(); duo<ImmaS8k32, 8>();
  duo<ImmaS8k16, 2>(); duo<ImmaS8k16, 4>(); duo<ImmaS8k16, 8>();
  duo<ImmaS4k64, 2>(); duo<ImmaS4k64, 4>();
  beside<ImmaS8k32>();
  beside<ImmaS8k16>();
  beside<HmmaF16k16F32>();
  form<ImmaS8k32>();
#elif GROUP == 1
  form<ImmaU8S8k32>(); form<ImmaU8U8k32>(); form<ImmaS8k32Sat>(); form<ImmaS8k16>();
#elif GROUP == 2
  form<ImmaS8m8k16>(); form<ImmaS4k64>(); form<ImmaU4k64>(); form<ImmaS4k32>(); form<ImmaS4m8k32>();
#elif GROUP == 3
  form<BmmaAndK256>(); form<BmmaXorK256>(); form<BmmaAndK128>(); form<BmmaAndM8K128>();
#elif GROUP == 4
  form<ImmaSpS8k64>(); form<ImmaSpS8k32>(); form<ImmaSpS4k128>(); form<ImmaSpS4k64>();
#elif GROUP == 5
  form<HmmaF16k16F32>(); form<HmmaBf16k16F32>(); form<HmmaF16k16F16>(); form<HmmaF16k8F32>();
#elif GROUP == 6
  form<HmmaBf16k8F32>(); form<HmmaF16k8F16>(); form<HmmaTf32k8>(); form<HmmaTf32k4>();
#elif GROUP == 7
  form<HmmaSpF16k32>(); form<HmmaSpF16k16>(); form<HmmaSpTf32k16>(); form<DmmaF64>();
  form<ImmaS8k32>();  // the reference again, last, so drift across the run shows
#elif GROUP == 8
  // the shared-memory loads alone, beside FFMA and IADD3 (one pipe with issue?) and beside saturated IMMA; M1's lane
  // gate (IADD3+FFMA alone) again, so this run's joint coefficients count
  opalone<LANE_MIX>(); opalone<lane1(OP_IADD3)>(); opalone<lane1(OP_FFMA)>(); opalone<lane1(OP_DP4A)>();
  opalone<lane1(OP_LDSM4)>(); opalone<lane1(OP_LDSM2)>(); opalone<lane1(OP_LDSM1)>(); opalone<lane1(OP_LDS128)>();
  opalone<lane2(OP_LDSM4, OP_FFMA)>(); opalone<lane2(OP_LDS128, OP_FFMA)>(); opalone<lane2(OP_LDSM4, OP_IADD3)>();
  opalone<lane2(OP_LDS128, OP_IADD3)>(); opalone<lane2(OP_DP4A, OP_LOP3)>(); opalone<lane2(OP_LOP3I, OP_FFMA)>();
  spec<ImmaS8k32, 64, lane1(OP_LDSM4)>(); spec<ImmaS8k32, 64, lane1(OP_LDS128)>();
  spec<ImmaS8k16, 64, lane1(OP_LDSM4)>(); spec<ImmaS8k16, 64, lane1(OP_LDS128)>();
  search<ImmaS8k32>(); search<ImmaS8k16>();
#elif GROUP == 9
  search<ImmaS8m8k16>(); search<HmmaF16k16F32>();
#elif GROUP == 10
  search<HmmaF16k16F16>(); search<HmmaF16k8F32>();
#elif GROUP == 11
  search<HmmaTf32k8>(); search<HmmaF16k8F16>();
  alone<ImmaS8k32, 4>(); mixu<ImmaS8k32, 48, LANE_MIX, 8, 2>();  // the reference again, last, so drift shows
#endif
  printf("{\"probe\":\"tensor_kinds_sm80\",\"group\":%d,\"done\":true,\"num_sms\":%d}\n", GROUP, g_sms);
  return 0;
}
