// LDS.128 beside a bulk unit that reads shared memory: R6r
// prices LDS.128, the bulk copy out of shared memory (UBLKCP4K) and the bulk reduction (TMAred512) as one read pool.
// Joint points, rows.jsonl only.
//   ls_<p><W>         W warps: 16 LDS.128 an iteration, alone
//   lb_<u>_<p><W>     the same W warps (OP_LOOP), and one bulk warp whose lane 0 issues unit u only while they do
//   ba_<u>            the bulk warp alone
// p: b (BCAST, every lane the line's first 16 bytes: one wavefront an instruction), l (LINE, lane l at 16 l: four).
// u: cp4k (cp.async.bulk.global.shared::cta, 4 KB, UBLKCP), red512 (cp.reduce.async.bulk .add.u32, 512 B, UBLKRED).
// Load q of an iteration reads line (16 (warp + it) + q) mod 64 of a 32 KB ring, so no two loads merge and none is
// loop-invariant; a bulk op's source is its slot's 4 KB of the same ring (it only reads it).
#include "../probe.cuh"

constexpr int Q = 16, LINE = 512, RING = 64, TBL = LINE * RING, SMEM = 64 * 1024, BOPS = 8;
enum { U_NONE, U_CP4K, U_RED512 };
enum { P_LINE, P_BCAST };

template <int P>
__host__ __device__ unsigned ls_off(int slot, int lane, int warp, int it) {
  return ((Q * (unsigned)(warp + it) + (unsigned)slot) & (RING - 1)) * LINE + (P == P_LINE ? lane * 16u : 0u);
}
// a joint point's accesses: the 16 loads, then the bulk role's stop poll (one word every lane reads)
template <int P>
__host__ __device__ unsigned lb_off(int slot, int lane, int warp, int it) {
  return slot < Q ? ls_off<P>(slot, lane, warp, it) : 0u;
}

template <int P>
__device__ __forceinline__ uint32_t lds16(unsigned sbase, int lane, int w, int it) {
  uint32_t acc = 0;
#pragma unroll
  for (int q = 0; q < Q; q++) {
    uint32_t r0, r1, r2, r3;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sbase + ls_off<P>(q, lane, w, it)));
    acc ^= r0 ^ r1 ^ r2 ^ r3;
  }
  return acc;
}

// One bulk trip from lane 0: BOPS ops, each to its own slot of the block's region of g, then a group bound so the
// unit's queue stays full without unbounded reads of the source in flight (side.cu's bulk_trip).
template <int U>
__device__ __forceinline__ void bulk_trip(unsigned char* g, unsigned sbase) {
  constexpr unsigned B = U == U_CP4K ? 4096u : 512u;
  unsigned char* region = g + (size_t)blockIdx.x * (BOPS * B);
#pragma unroll
  for (int q = 0; q < BOPS; q++) {
    unsigned char* p = region + q * B;
    const unsigned s = sbase + q * B % TBL;
    if constexpr (U == U_CP4K)
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(p), "r"(s), "n"(B) : "memory");
    else
      asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" ::"l"(p), "r"(s),
                   "n"(B) : "memory");
  }
  asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 2;" ::: "memory");
}

#define SMEM_TABLE                                                                                   \
  extern __shared__ __align__(128) unsigned char sb[];                                               \
  for (int i = threadIdx.x; i < TBL / 4; i += blockDim.x)                                            \
    reinterpret_cast<uint32_t*>(sb)[i] = i * 2654435761u + blockIdx.x + 1;                           \
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(sb);                                     \
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;

template <int U, int P, int W>
__device__ __forceinline__ void lds_bulk(const OpArgs& a) {
  SMEM_TABLE
  uint32_t acc = threadIdx.x + 1;
  OP_BEGIN_STOP
  if (w < W) {
    OP_LOOP(it) { acc ^= lds16<P>(sbase, lane, w, it); }
    OP_DONE
  } else if (U != U_NONE) {
    OP_UNTIL(n) {
      if (lane == 0) bulk_trip<U>(a.g, sbase);
    }
    OP_TRIPS(n)
    if (lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  OP_END_STOP
  OP_SINK(acc)
}

template <int U>
__device__ __forceinline__ void bulk_alone(const OpArgs& a) {
  SMEM_TABLE
  OP_BEGIN
  OP_LOOP(it) {
    if (lane == 0) bulk_trip<U>(a.g, sbase);
  }
  if (lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  OP_END
  OP_SINK(w + lane)
}

#define KLS(nm, P, W) \
  extern "C" __global__ void __launch_bounds__(W * 32, 1) nm(OpArgs a) { lds_bulk<U_NONE, P, W>(a); }
#define KLB(nm, U, P, W) \
  extern "C" __global__ void __launch_bounds__((W + 1) * 32, 1) nm(OpArgs a) { lds_bulk<U, P, W>(a); }
#define KBA(nm, U) extern "C" __global__ void __launch_bounds__(32, 1) nm(OpArgs a) { bulk_alone<U>(a); }
KLS(ls_b12, P_BCAST, 12)
KLS(ls_b16, P_BCAST, 16)
KLS(ls_l12, P_LINE, 12)
KLB(lb_cp4k_b12, U_CP4K, P_BCAST, 12)
KLB(lb_cp4k_b16, U_CP4K, P_BCAST, 16)
KLB(lb_cp4k_l12, U_CP4K, P_LINE, 12)
KLB(lb_red512_b12, U_RED512, P_BCAST, 12)
KLB(lb_red512_b16, U_RED512, P_BCAST, 16)
KBA(ba_cp4k, U_CP4K)
KBA(ba_red512, U_RED512)

#define LDS16 "LDS.128=16"
#define P_LS(nm, P, W)                                                                                         \
  {.name = #nm, .kernel = (const void*)nm, .tpb = W * 32, .expect = LDS16, .counted = "LDS.128",              \
   .smem_ops = "LDS.128*16", .smem = ls_off<P>, .dyn_smem = SMEM, .iters = 2048}
#define P_LB(nm, OP, P, W)                                                                                     \
  {.name = #nm, .kernel = (const void*)nm, .tpb = (W + 1) * 32, .expect = LDS16 ";" OP "=8", .counted = "LDS.128", \
   .warps = #W ";1", .smem_ops = "LDS.128*16,LDS*1", .smem = lb_off<P>, .dyn_smem = SMEM, .iters = 2048, .until = 2}
#define P_BA(nm, OP, IT) \
  {.name = #nm, .kernel = (const void*)nm, .tpb = 32, .expect = OP "=8", .dyn_smem = SMEM, .iters = IT}

static const OpProbe PROBES[] = {
    P_LS(ls_b12, P_BCAST, 12),
    P_LS(ls_b16, P_BCAST, 16),
    P_LS(ls_l12, P_LINE, 12),
    P_LB(lb_cp4k_b12, "UBLKCP*", P_BCAST, 12),
    P_LB(lb_cp4k_b16, "UBLKCP*", P_BCAST, 16),
    P_LB(lb_cp4k_l12, "UBLKCP*", P_LINE, 12),
    P_LB(lb_red512_b12, "UBLKRED*", P_BCAST, 12),
    P_LB(lb_red512_b16, "UBLKRED*", P_BCAST, 16),
    P_BA(ba_cp4k, "UBLKCP*", 128),
    P_BA(ba_red512, "UBLKRED*", 512),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
