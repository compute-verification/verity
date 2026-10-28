// ldmatrix / stmatrix (LDSM, STSM) by register count, beside LDS.128 / STS.128: RS-W′'s lm.cu (round 7-8) and lmmix.cu
// on probe.cuh. One 1024-thread block an SM, 8 instructions
// an iteration. Each instruction of each iteration has its own 16-B rows (lane l: row base + l + 8 slot, the base moving
// 64 rows a warp and an iteration round a ring of 2048), so ptxas can't merge two, and each 8-row phase is 128
// consecutive bytes: no bank conflict.
//
// Two known-invalid probes stay as the gate's regression tests (refuse = the check that must refuse them):
//   bad_ldsm_merged    round 7's loop: one address set every iteration, no #pragma unroll 1; ptxas unrolled it 16 times
//                      and kept 9 of every 128 loads.
//   bad_stsm_conflict  round 7's stores: a lane stride of 8 slots x 16 B = 128 B, so a phase's 8 rows share 4 banks
//                      (8-way). Given its own OP_LOOP, so the bank check is what refuses it.
#include "../probe.cuh"

constexpr int TPB = 1024, U = 8, RING = 2048;
constexpr int SMEM_BYTES = (RING + 256) * 16;

__host__ __device__ inline unsigned lm_off(int slot, int lane, int warp, int it) {
  return ((((unsigned)lane + 8u * U * (unsigned)(warp + it)) & (RING - 1)) + 8u * (unsigned)slot) * 16u;
}

template <int R, bool T>
__device__ __forceinline__ unsigned ldsm(unsigned addr) {
  unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  if constexpr (R == 1 && !T) asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r0) : "r"(addr));
  if constexpr (R == 1 && T) asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(r0) : "r"(addr));
  if constexpr (R == 2 && !T)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
  if constexpr (R == 2 && T)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
  if constexpr (R == 4 && !T)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
  if constexpr (R == 4 && T)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
  return r0 ^ r1 ^ r2 ^ r3;
}

template <int R>
__device__ __forceinline__ void stsm(unsigned addr, unsigned v) {
  if constexpr (R == 1) asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};" ::"r"(addr), "r"(v));
  if constexpr (R == 2)
    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1,%2};" ::"r"(addr), "r"(v), "r"(v + 1));
  if constexpr (R == 4)
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(v), "r"(v + 1),
                 "r"(v + 2), "r"(v + 3));
}

#define LM_SETUP                                                                          \
  __shared__ __align__(128) unsigned char smem[SMEM_BYTES];                               \
  for (int i = threadIdx.x; i < SMEM_BYTES / 4; i += blockDim.x)                          \
    reinterpret_cast<unsigned*>(smem)[i] = i * 2654435761u + blockIdx.x + 1;              \
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);                        \
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;                                \
  unsigned acc = threadIdx.x + 1;

#define LDSM_PROBE(name, R, T)                                                            \
  OP_KERNEL(name, TPB) {                                                                  \
    LM_SETUP                                                                              \
    OP_BEGIN                                                                              \
    OP_LOOP(it) {                                                                         \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc ^= ldsm<R, T>(sbase + lm_off(u, lane, w, it)); \
    }                                                                                     \
    OP_END                                                                                \
    OP_SINK(acc)                                                                          \
  }
LDSM_PROBE(ldsm_x1, 1, false)
LDSM_PROBE(ldsm_x2, 2, false)
LDSM_PROBE(ldsm_x4, 4, false)
LDSM_PROBE(ldsm_x1t, 1, true)
LDSM_PROBE(ldsm_x2t, 2, true)
LDSM_PROBE(ldsm_x4t, 4, true)

#define STSM_PROBE(name, R)                                                               \
  OP_KERNEL(name, TPB) {                                                                  \
    LM_SETUP                                                                              \
    OP_BEGIN                                                                              \
    OP_LOOP(it) {                                                                         \
      _Pragma("unroll") for (int u = 0; u < U; u++) stsm<R>(sbase + lm_off(u, lane, w, it), acc + u); \
      acc += 0x9E3779B9u;                                                                 \
    }                                                                                     \
    OP_END                                                                                \
    OP_SINK(acc ^ reinterpret_cast<unsigned*>(smem)[threadIdx.x])                         \
  }
STSM_PROBE(stsm_x1, 1)
STSM_PROBE(stsm_x2, 2)
STSM_PROBE(stsm_x4, 4)

OP_KERNEL(lds128, TPB) {
  LM_SETUP
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      unsigned r0, r1, r2, r3;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sbase + lm_off(u, lane, w, it)));
      acc ^= r0 ^ r1 ^ r2 ^ r3;
    }
  }
  OP_END
  OP_SINK(acc)
}

OP_KERNEL(sts128, TPB) {
  LM_SETUP
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++)
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(sbase + lm_off(u, lane, w, it)), "r"(acc), "r"(acc + 1),
                   "r"(acc + 2), "r"(acc + u));
    acc += 0x9E3779B9u;
  }
  OP_END
  OP_SINK(acc ^ reinterpret_cast<unsigned*>(smem)[threadIdx.x])
}

// ---- round 7's forms, which the gate must refuse --------------------------------------------------------------------

constexpr int R7_INTS = 4096;
__host__ __device__ inline unsigned r7_off(int slot, int lane, int warp, int) {
  return ((((unsigned)(warp * 32 + lane) * U + (unsigned)slot) * 4u) & (R7_INTS - 1)) * 4u;
}

OP_KERNEL(bad_ldsm_merged, TPB) {
  __shared__ __align__(128) int smem[R7_INTS];
  for (int i = threadIdx.x; i < R7_INTS; i += blockDim.x) smem[i] = i * 2654435761u + blockIdx.x + 1;
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  unsigned acc = threadIdx.x + 1;
  OP_BEGIN
  for (int it = 0; it < a.iters; it++) {
#pragma unroll
    for (int u = 0; u < U; u++) acc ^= ldsm<1, false>(sbase + r7_off(u, lane, w, it));
  }
  OP_END
  OP_SINK(acc)
}

OP_KERNEL(bad_stsm_conflict, TPB) {
  __shared__ __align__(128) int smem[R7_INTS];
  for (int i = threadIdx.x; i < R7_INTS; i += blockDim.x) smem[i] = i * 2654435761u + blockIdx.x + 1;
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  unsigned acc = threadIdx.x + 1;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) stsm<1>(sbase + r7_off(u, lane, w, it), acc + u);
    acc += 0x9E3779B9u;
  }
  OP_END
  OP_SINK(acc ^ smem[threadIdx.x])
}

#define LDSM_ROW(nm, K, R, OPS, W)                                                                                    \
  {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .kind = K, .row = "ROW_LDSMR" #R, .expect = OPS "=8",       \
   .words = W, .smem_ops = OPS "*8", .smem = lm_off}
#define STSM_ROW(nm, K, R, OPS, W)                                                                                    \
  {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .kind = K, .row = "ROW_STSMR" #R, .expect = OPS "=8",       \
   .words = W, .smem_ops = OPS "*8", .smem = lm_off}

static const OpProbe PROBES[] = {
    LDSM_ROW(ldsm_x1, "LDSM_X1", 1, "LDSM.16.M88", 32),
    LDSM_ROW(ldsm_x2, "LDSM_X2", 2, "LDSM.16.M88.2", 64),
    LDSM_ROW(ldsm_x4, "LDSM_X4", 4, "LDSM.16.M88.4", 128),
    LDSM_ROW(ldsm_x1t, "LDSM_X1T", 1, "LDSM.16.MT88", 32),
    LDSM_ROW(ldsm_x2t, "LDSM_X2T", 2, "LDSM.16.MT88.2", 64),
    LDSM_ROW(ldsm_x4t, "LDSM_X4T", 4, "LDSM.16.MT88.4", 128),
    STSM_ROW(stsm_x1, "STSM_X1", 1, "STSM.16.M88", 32),
    STSM_ROW(stsm_x2, "STSM_X2", 2, "STSM.16.M88.2", 64),
    STSM_ROW(stsm_x4, "STSM_X4", 4, "STSM.16.M88.4", 128),
    {.name = "lds128", .kernel = (const void*)lds128, .tpb = TPB, .kind = "LDS128", .row = "R6r",
     .expect = "LDS.128=8", .words = 128, .smem_ops = "LDS.128*8", .smem = lm_off},
    {.name = "sts128", .kernel = (const void*)sts128, .tpb = TPB, .kind = "STS128", .row = "R6w",
     .expect = "STS.128=8", .words = 128, .smem_ops = "STS.128*8", .smem = lm_off},
    {.name = "bad_ldsm_merged", .kernel = (const void*)bad_ldsm_merged, .tpb = TPB, .kind = "LDSM_X1",
     .row = "ROW_LDSMR1", .expect = "LDSM.16.M88=8", .words = 32, .smem_ops = "LDSM.16.M88*8", .smem = r7_off,
     .refuse = "loops", .note = "round 7's lm.cu LDSM loop (r20261016-112703-0297): ptxas merged its loads"},
    {.name = "bad_stsm_conflict", .kernel = (const void*)bad_stsm_conflict, .tpb = TPB, .kind = "STSM_X1",
     .row = "ROW_STSMR1", .expect = "STSM.16.M88=8", .words = 32, .smem_ops = "STSM.16.M88*8", .smem = r7_off,
     .refuse = "smem", .note = "round 7's lm.cu STSM addresses: a 128-B lane stride, an 8-way bank conflict"},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
