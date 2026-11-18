// Beside-IMMA joint points: the issue ceiling's round-6 probes (issue_ceiling/issue_wide.cu) on probe.cuh. Each is a
// joint point, not one class's own rate: its rows.jsonl line has every expected class's rate, and it writes no census
// line.
//   imma_alone    k_specmma: 16 warps, 8 independent IMMA an iteration (the reference)
//   spec_<x>_r<R> k_spec: the first 8 warps IMMA, the other 8 R ops of x an iteration (warp-specialized)
//   mixu_<x>_<M>  k_mixu: every warp interleaves M ops of x after each of its 8 IMMA (one instruction stream)
// x: dadd, dfma (each op a self-recurrence on one of 8 accumulators), red64 (red.global.add.u64 streaming over a pool).
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

enum { X_DADD, X_DFMA, X_RED64 };

struct Lane {
  double f[8];
  uint64_t* pool;
  uint32_t gtid, total, mask;
  uint64_t c64;
  double cd;
  __device__ explicit Lane(const OpArgs& a) {
    gtid = blockIdx.x * blockDim.x + threadIdx.x;
    total = gridDim.x * blockDim.x;
    pool = reinterpret_cast<uint64_t*>(a.g);
    mask = (uint32_t)(a.gbytes / 8) - 1;
    c64 = ((uint64_t)(threadIdx.x | 1u) << 32) | (threadIdx.x | 1u);
    cd = 1.0000001192092896;
#pragma unroll
    for (int i = 0; i < 8; i++) f[i] = (double)((threadIdx.x & 0xff) + i + 1);
  }
  template <int X>
  __device__ __forceinline__ void op(int slot, uint32_t seq) {
    if constexpr (X == X_DADD) asm volatile("add.f64 %0,%0,%1;" : "+d"(f[slot]) : "d"(cd));
    if constexpr (X == X_DFMA) asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(f[slot]) : "d"(cd), "d"(cd));
    if constexpr (X == X_RED64) {
      uint64_t* p = pool + (uint32_t)(((uint64_t)gtid + (uint64_t)seq * total) & mask);
      asm volatile("red.global.add.u64 [%0], %1;" ::"l"(p), "l"(c64) : "memory");
    }
  }
  __device__ uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const long long b = __double_as_longlong(f[i]);
      x ^= (uint32_t)b ^ (uint32_t)(b >> 32);
    }
    return x;
  }
};

OP_KERNEL(imma_alone, TPB) {
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

template <int X, int R>
__device__ __forceinline__ void spec(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  const bool mma_role = (threadIdx.x >> 5) < (blockDim.x >> 6);
  uint32_t x[4], y[2];
  init_mma(lane, x, y);
  int32_t d[8][4] = {};
  Lane l(a);
  OP_BEGIN
  if (mma_role) {
    OP_LOOP(it) {
#pragma unroll
      for (int c = 0; c < 8; c++) mma_imma(d[c], x, y);
    }
  } else {
    OP_LOOP(it) {
#pragma unroll
      for (int q = 0; q < R; q++) l.op<X>(q & 7, (uint32_t)it * R + q);
    }
  }
  OP_END
  uint32_t s = lane ^ l.fold();
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)d[c][0];
  OP_SINK(s)
}

template <int X, int M>
__device__ __forceinline__ void mixu(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
  init_mma(lane, x, y);
  int32_t d[8][4] = {};
  Lane l(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      mma_imma(d[c], x, y);
#pragma unroll
      for (int q = 0; q < M; q++) l.op<X>((c + q) & 7, ((uint32_t)it * 8u + c) * M + q);
    }
  }
  OP_END
  uint32_t s = lane ^ l.fold();
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)d[c][0];
  OP_SINK(s)
}

OP_KERNEL(spec_dadd_r2, TPB) { spec<X_DADD, 2>(a); }
OP_KERNEL(spec_dfma_r2, TPB) { spec<X_DFMA, 2>(a); }
OP_KERNEL(spec_red64_r2, TPB) { spec<X_RED64, 2>(a); }
OP_KERNEL(mixu_dadd_1, TPB) { mixu<X_DADD, 1>(a); }
OP_KERNEL(mixu_dfma_1, TPB) { mixu<X_DFMA, 1>(a); }
OP_KERNEL(mixu_red64_1, TPB) { mixu<X_RED64, 1>(a); }

#define IMMA "IMMA.16832.S8.S8"
#define RED64 "REDG.E.ADD.64.STRONG.GPU"
static const OpProbe PROBES[] = {
    {.name = "imma_alone", .kernel = (const void*)imma_alone, .tpb = TPB, .expect = IMMA "=8", .iters = 4096},
    {.name = "spec_dadd_r2", .kernel = (const void*)spec_dadd_r2, .tpb = TPB, .expect = IMMA "=8;DADD=2",
     .warps = "8;8", .iters = 4096},
    {.name = "spec_dfma_r2", .kernel = (const void*)spec_dfma_r2, .tpb = TPB, .expect = IMMA "=8;DFMA=2",
     .warps = "8;8", .iters = 4096},
    {.name = "spec_red64_r2", .kernel = (const void*)spec_red64_r2, .tpb = TPB, .expect = IMMA "=8;" RED64 "=2",
     .warps = "8;8", .iters = 4096},
    {.name = "mixu_dadd_1", .kernel = (const void*)mixu_dadd_1, .tpb = TPB, .expect = IMMA "=8,DADD=8", .iters = 2048},
    {.name = "mixu_dfma_1", .kernel = (const void*)mixu_dfma_1, .tpb = TPB, .expect = IMMA "=8,DFMA=8", .iters = 2048},
    {.name = "mixu_red64_1", .kernel = (const void*)mixu_red64_1, .tpb = TPB, .expect = IMMA "=8," RED64 "=8",
     .iters = 2048},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
