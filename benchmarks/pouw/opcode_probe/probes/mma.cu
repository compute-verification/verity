// The exact MMA configurations of ../../fp_exact/ beside int8 IMMA: 16 warps, 8 independent accumulators a warp, one
// MMA each an iteration, so the rate is the tensor pipe's and not one chain's latency. Operands are on fp_exact's exact
// grids (E4M3 integers, E2M1 x2 values) with unit block scales. Joint points: rows.jsonl only (machine_model has one
// MMA kind, the honest GEMM's). MACs an instruction: m16n8k32 4096, m16n8k64 8192, 2:4 sparse m16n8k128 16384.
#include "../probe.cuh"

constexpr int TPB = 512;

__device__ __forceinline__ void imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void qmma(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <bool NV>
__device__ __forceinline__ void omma(float* d, const uint32_t* a, const uint32_t* b, uint32_t s) {
  if constexpr (NV)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%10}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(s));
  else
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%10}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(s));
}
template <bool NV>
__device__ __forceinline__ void omma_sp(float* d, const uint32_t* a, const uint32_t* b, uint32_t e, uint32_t s) {
  if constexpr (NV)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1."
        "e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, {%13}, {0, 0}, {%13}, "
        "{0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e), "r"(s));
  else
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1."
        "e2m1.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, {%13}, {0, 0}, {%13}, "
        "{0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e), "r"(s));
}

// Operand words: E4M3 bytes of small integers (0x38 = 1, 0x40 = 2, 0x48 = 4, 0x50 = 8), E2M1 nibbles of the x2 grid
// (2 = 1.0, 3 = 1.5, 5 = 3.0, 7 = 6.0), varied by lane so no two lanes' operands coincide.
__device__ __forceinline__ void init8(uint32_t lane, uint32_t* a, uint32_t* b) {
  a[0] = 0x38404850u; a[1] = 0x50484038u ^ (lane & 7); a[2] = 0x40384850u; a[3] = 0x48503840u;
  b[0] = 0x38384040u ^ ((lane >> 3) & 7); b[1] = 0x48485050u;
}
__device__ __forceinline__ void init4(uint32_t lane, uint32_t* a, uint32_t* b) {
  a[0] = 0x23572357u ^ (lane & 7); a[1] = 0x75327532u; a[2] = 0x35723572u; a[3] = 0x57235723u;
  b[0] = 0x22335577u ^ ((lane >> 3) & 7); b[1] = 0x77553322u; b[2] = 0x27532753u; b[3] = 0x35723572u;
}

__device__ __forceinline__ uint32_t fold(const float (*d)[4], uint32_t lane) {
  uint32_t s = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= __float_as_uint(d[c][0]) ^ __float_as_uint(d[c][3]);
  return s;
}

OP_KERNEL(imma_s8, TPB) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
  init8(lane, x, y);
  x[0] = 0x01020304u; x[1] = 0x05060708u ^ lane;
  int32_t d[8][4] = {};
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) imma(d[c], x, y);
  }
  OP_END
  uint32_t s = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)d[c][0] ^ (uint32_t)d[c][3];
  OP_SINK(s)
}

OP_KERNEL(qmma_e4m3, TPB) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
  init8(lane, x, y);
  float d[8][4] = {};
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) qmma(d[c], x, y);
  }
  OP_END
  OP_SINK(fold(d, lane))
}

template <bool NV>
__device__ __forceinline__ void omma_body(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[4];
  init4(lane, x, y);
  const uint32_t sc = NV ? 0x38383838u : 0x7F7F7F7Fu;
  float d[8][4] = {};
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) omma<NV>(d[c], x, y, sc);
  }
  OP_END
  OP_SINK(fold(d, lane))
}
OP_KERNEL(omma_mxf4, TPB) { omma_body<false>(a); }
OP_KERNEL(omma_nvf4, TPB) { omma_body<true>(a); }

template <bool NV>
__device__ __forceinline__ void omma_sp_body(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[4];
  init4(lane, x, y);
  const uint32_t sc = NV ? 0x38383838u : 0x7F7F7F7Fu;
  const uint32_t e = 0x4E4E4E4Eu;  // chunks keep pairs {2,3} and {0,1} in turn (nibbles 0xE = 3<<2|2, 0x4 = 1<<2|0)
  float d[8][4] = {};
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) omma_sp<NV>(d[c], x, y, e, sc);
  }
  OP_END
  OP_SINK(fold(d, lane))
}
OP_KERNEL(omma_sp_mxf4, TPB) { omma_sp_body<false>(a); }
OP_KERNEL(omma_sp_nvf4, TPB) { omma_sp_body<true>(a); }

#define OMMA "OMMA.SF.16864.F32.E2M1.E2M1"
#define OMMA_SP "OMMA.SF.SP.168128.F32.E2M1.E2M1"
static const OpProbe PROBES[] = {
    {.name = "imma_s8", .kernel = (const void*)imma_s8, .tpb = TPB, .expect = "IMMA.16832.S8.S8=8", .iters = 4096,
     .note = "int8 m16n8k32, s32 accumulator: 4096 MACs an instruction"},
    {.name = "qmma_e4m3", .kernel = (const void*)qmma_e4m3, .tpb = TPB, .expect = "QMMA.16832.F32.E4M3.E4M3=8",
     .iters = 4096, .note = "E4M3 m16n8k32, FP32 accumulator: 4096 MACs an instruction"},
    {.name = "omma_mxf4", .kernel = (const void*)omma_mxf4, .tpb = TPB, .expect = OMMA ".E8=8", .iters = 4096,
     .note = "MXFP4 m16n8k64, UE8M0 unit scales: 8192 MACs an instruction"},
    {.name = "omma_nvf4", .kernel = (const void*)omma_nvf4, .tpb = TPB, .expect = OMMA ".UE4M3.4X=8", .iters = 4096,
     .note = "NVFP4 m16n8k64, UE4M3 unit scales: 8192 MACs an instruction"},
    {.name = "omma_sp_mxf4", .kernel = (const void*)omma_sp_mxf4, .tpb = TPB, .expect = OMMA_SP ".E8=8", .iters = 4096,
     .note = "2:4 sparse MXFP4 m16n8k128, UE8M0 unit scales: 16384 MACs an instruction"},
    {.name = "omma_sp_nvf4", .kernel = (const void*)omma_sp_nvf4, .tpb = TPB, .expect = OMMA_SP ".UE4M3.4X=8",
     .iters = 4096, .note = "2:4 sparse NVFP4 m16n8k128, UE4M3 unit scales: 16384 MACs an instruction"},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
