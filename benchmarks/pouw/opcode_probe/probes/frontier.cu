// The MMA + X joint frontiers that rows.csv's R2 (IMAD), R3 (HFMA2), R4 (ALU) and SHFL rows assumed, re-measured as mix
// points. Each probe but the alone ones is a joint point: rows.jsonl
// holds both classes' rates, and eps1.py turns the points into facets with the 2% noise margin.
//   imma_ref        16 warps, 8 independent IMMA an iteration
//   alone_<x>       16 warps, 16 ops of x an iteration on 8 chains (x's own rate)
//   mix_<x>_<Q>     16 warps, every warp 8 IMMA and Q ops of x an iteration, the ops spread evenly after the IMMAs
//   spec_<x>_w<W>   8 IMMA warps (OP_LOOP) and W x warps that run only while they do (OP_UNTIL), 16 ops a trip
// x: imad (mad.lo.u32, register operands), hfma2 (fma.rn.f16x2), lop3 (the ALU group), shfl (shfl.sync.bfly, lane 1).
#include "../probe.cuh"

constexpr int TPB = 512;

__device__ __forceinline__ void mma_imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void init_mma(uint32_t lane, uint32_t* a, uint32_t* b) {
  a[0] = 0x08080808u ^ lane; a[1] = 0x01010101u; a[2] = 0x08010801u; a[3] = 0x01080108u;
  b[0] = 0x08080808u; b[1] = 0x01010101u ^ lane;
}

enum { X_IMAD, X_HFMA2, X_LOP3, X_SHFL };

struct Ops {
  uint32_t v[8], k1, k2;
  __device__ explicit Ops(const OpArgs& a) {
    // a.mode is 0 at run time; through it the operands stay registers, so ptxas folds no immediate form
    k1 = 0x3C013C01u ^ (uint32_t)a.mode;
    k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = threadIdx.x * 8u + i + 1u;
  }
  template <int X>
  __device__ __forceinline__ void op(int s) {
    if constexpr (X == X_IMAD) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_HFMA2) asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_LOP3) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_SHFL) asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(v[s]));
  }
  __device__ uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x ^= v[i];
    return x;
  }
};

OP_KERNEL(imma_ref, TPB) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
  init_mma(lane, x, y);
  int32_t d[8][4] = {};
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) mma_imma(d[c], x, y);
  }
  OP_END
  uint32_t s = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)d[c][0];
  OP_SINK(s)
}

template <int X>
__device__ __forceinline__ void alone(const OpArgs& a) {
  Ops o(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < 16; q++) o.op<X>(q & 7);
  }
  OP_END
  OP_SINK(o.fold())
}

template <int X, int Q>
__device__ __forceinline__ void mix(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
  init_mma(lane, x, y);
  int32_t d[8][4] = {};
  Ops o(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      mma_imma(d[c], x, y);
#pragma unroll
      for (int q = c * Q / 8; q < (c + 1) * Q / 8; q++) o.op<X>(q & 7);
    }
  }
  OP_END
  uint32_t s = lane ^ o.fold();
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)d[c][0];
  OP_SINK(s)
}

template <int X>
__device__ __forceinline__ void spec(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
  init_mma(lane, x, y);
  int32_t d[8][4] = {};
  Ops o(a);
  OP_BEGIN_STOP
  if ((threadIdx.x >> 5) < 8) {
    OP_LOOP(it) {
#pragma unroll
      for (int c = 0; c < 8; c++) mma_imma(d[c], x, y);
    }
    OP_DONE
  } else {
    OP_UNTIL(n) {
#pragma unroll
      for (int q = 0; q < 16; q++) o.op<X>(q & 7);
    }
    OP_TRIPS(n)
  }
  OP_END_STOP
  uint32_t s = lane ^ o.fold();
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)d[c][0];
  OP_SINK(s)
}

#define ALONE(x, X) OP_KERNEL(alone_##x, TPB) { alone<X>(a); }
#define MIX(x, X, Q) OP_KERNEL(mix_##x##_##Q, TPB) { mix<X, Q>(a); }
#define SPEC(x, X, W) OP_KERNEL(spec_##x##_w##W, (8 + W) * 32) { spec<X>(a); }
#define MIXES(x, X)                                                                                                  \
  ALONE(x, X) MIX(x, X, 2) MIX(x, X, 4) MIX(x, X, 8) MIX(x, X, 16) MIX(x, X, 24) MIX(x, X, 32) MIX(x, X, 48)        \
  MIX(x, X, 64) MIX(x, X, 128)
#define KNEE(x, X) MIX(x, X, 40) MIX(x, X, 56) MIX(x, X, 80) MIX(x, X, 96)
#define KERNELS(x, X) MIXES(x, X) KNEE(x, X) SPEC(x, X, 4) SPEC(x, X, 8)
KERNELS(imad, X_IMAD)
KERNELS(hfma2, X_HFMA2)
KERNELS(lop3, X_LOP3)
MIXES(shfl, X_SHFL)
MIX(shfl, X_SHFL, 18) MIX(shfl, X_SHFL, 20) MIX(shfl, X_SHFL, 22) MIX(shfl, X_SHFL, 28)

#define IMMA "IMMA.16832.S8.S8"
#define P_ALONE(x, OP, KIND, ROW)                                                                                    \
  {.name = "alone_" #x, .kernel = (const void*)alone_##x, .tpb = TPB, .kind = KIND, .row = ROW, .expect = OP "=16", \
   .words = 32, .iters = 4096}
#define P_MIX(x, OP, Q)                                                                                              \
  {.name = "mix_" #x "_" #Q, .kernel = (const void*)mix_##x##_##Q, .tpb = TPB, .expect = IMMA "=8," OP "=" #Q,       \
   .iters = 2048}
#define P_SPEC(x, OP, W)                                                                                             \
  {.name = "spec_" #x "_w" #W, .kernel = (const void*)spec_##x##_w##W, .tpb = (8 + W) * 32,                          \
   .expect = IMMA "=8;" OP "=16", .warps = "8;" #W, .iters = 4096, .until = 2}
#define P_MIXES(x, OP, KIND, ROW)                                                                                    \
  P_ALONE(x, OP, KIND, ROW), P_MIX(x, OP, 2), P_MIX(x, OP, 4), P_MIX(x, OP, 8), P_MIX(x, OP, 16), P_MIX(x, OP, 24),  \
  P_MIX(x, OP, 32), P_MIX(x, OP, 48), P_MIX(x, OP, 64), P_MIX(x, OP, 128)
#define P_KNEE(x, OP) P_MIX(x, OP, 40), P_MIX(x, OP, 56), P_MIX(x, OP, 80), P_MIX(x, OP, 96)
#define PROBES_OF(x, OP, KIND, ROW) P_MIXES(x, OP, KIND, ROW), P_KNEE(x, OP), P_SPEC(x, OP, 4), P_SPEC(x, OP, 8)

// shfl has no spec probe: in a warp-divergent role ptxas gives SHFL a BRA.DIV fallback placed after the closing clock read
// that branches back into the loop, so the loop straddles the clock.
static const OpProbe PROBES[] = {
    {.name = "imma_ref", .kernel = (const void*)imma_ref, .tpb = TPB, .expect = IMMA "=8", .iters = 4096},
    PROBES_OF(imad, "IMAD", "IMAD", "I"),
    PROBES_OF(hfma2, "HFMA2*", "", ""),
    PROBES_OF(lop3, "LOP3.LUT", "ALU", "A"),
    P_MIXES(shfl, "SHFL.BFLY", "", ""),
    P_MIX(shfl, "SHFL.BFLY", 18), P_MIX(shfl, "SHFL.BFLY", 20), P_MIX(shfl, "SHFL.BFLY", 22),
    P_MIX(shfl, "SHFL.BFLY", 28),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
