// The tensor forms A7's menu prices as one class
// or not at all, each timed beside the int8 and E4M3 references in one interleaved run.
//   qmma_<t>  kind::f8f6f4 m16n8k32 at t = e4m3, e5m2, e3m2, e2m3, e2m1: on sm_120a each is one QMMA.16832.F32.<T>.<T>
//             (isa_accept.sh), which loads.py prices as the single class QMMA16832 whatever t is.
//             Equal rates keep that one class; a faster FP6/FP4 rate means the class under-prices those types.
//   dmma      mma.sync m8n8k4 f64: one DMMA.8x8x4 on sm_120a, exact for int8 products, with no census class (loads.py
//             gives None) and no sheet FP64 tensor peak, so the data-sheet machine prices it at nothing.
//   imma_s8   the int8 reference (4096 MACs an instruction).
// 16 warps a block, one block an SM, 8 independent accumulators a warp, one MMA each an iteration, so each rate is the
// pipe's and not one chain's latency. Joint points: rows.jsonl only, no census line.
#include "../probe.cuh"

constexpr int TPB = 512;

// One byte set for every type: each byte is a finite, nonzero value as E4M3, E5M2 and every FP6/FP4 container
// placement, so no type sees a NaN, an infinity or a zero (a data-dependent rate would show against qmma_e4m3).
__device__ __forceinline__ void init_f8(uint32_t lane, uint32_t* a, uint32_t* b) {
  a[0] = 0x14283448u ^ (lane & 3u); a[1] = 0x0C182430u ^ ((lane >> 2) & 3u);
  a[2] = 0x34481428u; a[3] = 0x24300C18u;
  b[0] = 0x48341428u ^ ((lane >> 4) & 1u); b[1] = 0x180C3024u;
}

#define QMMA_FN(t)                                                                                                     \
  __device__ __forceinline__ void qmma_##t##_op(float* d, const uint32_t* a, const uint32_t* b) {                      \
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32." #t "." #t                                       \
                 ".f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"                                          \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                                      \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));                                  \
  }                                                                                                                    \
  OP_KERNEL(qmma_##t, TPB) {                                                                                           \
    const uint32_t lane = threadIdx.x & 31;                                                                            \
    uint32_t x[4], y[2];                                                                                               \
    init_f8(lane, x, y);                                                                                               \
    float d[8][4] = {};                                                                                                \
    OP_BEGIN                                                                                                           \
    OP_LOOP(it) {                                                                                                      \
      _Pragma("unroll") for (int c = 0; c < 8; c++) qmma_##t##_op(d[c], x, y);                                         \
    }                                                                                                                  \
    OP_END                                                                                                             \
    uint32_t s = lane;                                                                                                 \
    _Pragma("unroll") for (int c = 0; c < 8; c++) s ^= __float_as_uint(d[c][0]) ^ __float_as_uint(d[c][3]);           \
    OP_SINK(s)                                                                                                         \
  }
QMMA_FN(e4m3)
QMMA_FN(e5m2)
QMMA_FN(e3m2)
QMMA_FN(e2m3)
QMMA_FN(e2m1)

__device__ __forceinline__ void imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

OP_KERNEL(imma_s8, TPB) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
  init_f8(lane, x, y);
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

__device__ __forceinline__ void dmma(double* d, double a, double b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
               : "+d"(d[0]), "+d"(d[1])
               : "d"(a), "d"(b));
}

OP_KERNEL(dmma_f64, TPB) {
  const uint32_t lane = threadIdx.x & 31;
  const double p = (double)(lane + 1u), q = 1.5 + (double)(lane & 7u);
  double d[8][2] = {};
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) dmma(d[c], p, q);
  }
  OP_END
  uint32_t s = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) {
    const long long v = __double_as_longlong(d[c][0]) ^ __double_as_longlong(d[c][1]);
    s ^= (uint32_t)v ^ (uint32_t)(v >> 32);
  }
  OP_SINK(s)
}

#define P_QMMA(t, T)                                                                                                   \
  {.name = "qmma_" #t, .kernel = (const void*)qmma_##t, .tpb = TPB, .expect = "QMMA.16832.F32." T "." T "=8",          \
   .iters = 4096, .note = "kind::f8f6f4 m16n8k32 at " #t ": 4096 MACs an instruction"}
static const OpProbe PROBES[] = {
    {.name = "imma_s8", .kernel = (const void*)imma_s8, .tpb = TPB, .expect = "IMMA.16832.S8.S8=8", .iters = 4096,
     .note = "int8 m16n8k32, the reference: 4096 MACs an instruction"},
    P_QMMA(e4m3, "E4M3"),
    P_QMMA(e5m2, "E5M2"),
    P_QMMA(e3m2, "E3M2"),
    P_QMMA(e2m3, "E2M3"),
    P_QMMA(e2m1, "E2M1"),
    {.name = "dmma_f64", .kernel = (const void*)dmma_f64, .tpb = TPB, .expect = "DMMA.8x8x4=8", .iters = 512,
     .note = "f64 m8n8k4: 256 MACs an instruction, exact for int8 products"},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
