// H32 on the H200 (sm_90a), the other class's memory writers: lookups whose address is the last lookup's result (a
// table that computes x -> T[x]), and atomics. Notes as sm90_h32.cu's; ../h32.py prices them. An atomic that returns
// writes 2 words a lane (its register and the memory word), a reduction 1. One 1024-thread block an SM, 8 an iteration
// on 8 independent chains a thread.
//   d_lds, d_lds128   shared: lane l's cells in bank l (4 B) or its own 16 B of a 512-B row, a warp's 8 chains on 8
//                     rows, so each access reads one row, conflict-free; a cell holds the next row's address (d_lds128
//                     folds its other 3 words, 2 LOP3 a load, so ptxas keeps all 4)
//   d_ldg64           global through L1: each block's table in its own 16 KB of a.g, 8-B cells holding the next's address
//                     (ld.global.ca: LDG.E.64.STRONG.SM)
//   a_atoms, a_atomg  atom.add with its return, each lane its own word (shared: conflict-free), uncontended
//   a_redg            red.global.add, each lane its own word, on every SM (the L2's atomic path bounds it)
#include "../../opcode_probe/probe.cuh"

constexpr int TPB = 1024, U = 8, ROWS = 64, WARPS = TPB / 32;

// A chain's row at a slot, warp and iteration: the kernels' own addresses, which the gate's bank check reads.
__host__ __device__ inline int mem_row(int slot, int warp, int it) { return (slot + warp * U + it) % ROWS; }
__host__ __device__ inline unsigned lds_off(int slot, int lane, int warp, int it) {
  return (unsigned)(mem_row(slot, warp, it) * 32 + lane) * 4;
}
__host__ __device__ inline unsigned lds128_off(int slot, int lane, int warp, int it) {
  return (unsigned)(mem_row(slot, warp, it) * 32 + lane) * 16;
}
__host__ __device__ inline unsigned atoms_off(int slot, int lane, int warp, int) {
  return (unsigned)((slot * WARPS + warp) * 32 + lane) * 4;
}

OP_KERNEL(d_lds, TPB) {
  extern __shared__ __align__(128) unsigned sb[];
  const unsigned base = (unsigned)__cvta_generic_to_shared(sb);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < ROWS * 32; i += TPB) sb[i] = base + lds_off(1, i & 31, 0, i >> 5);
  unsigned x[U];
#pragma unroll
  for (int u = 0; u < U; u++) x[u] = base + lds_off(u, lane, w, 0);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) asm volatile("ld.shared.u32 %0, [%0];" : "+r"(x[u]));
  }
  OP_END
  unsigned s = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) s ^= x[u];
  OP_SINK(s)
}

OP_KERNEL(d_lds128, TPB) {
  extern __shared__ __align__(128) uint4 sq[];
  const unsigned base = (unsigned)__cvta_generic_to_shared(sq);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < ROWS * 32; i += TPB)
    sq[i] = make_uint4(base + lds128_off(1, i & 31, 0, i >> 5), 0x9E3779B9u * i, 0x85EBCA6Bu ^ i, (unsigned)a.mode + i);
  unsigned x[U], y = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) x[u] = base + lds128_off(u, lane, w, 0);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      unsigned p, q, r;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%0];" : "+r"(x[u]), "=r"(p), "=r"(q), "=r"(r));
      y ^= p ^ q ^ r;
    }
  }
  OP_END
#pragma unroll
  for (int u = 0; u < U; u++) y ^= x[u];
  OP_SINK(y)
}

OP_KERNEL(d_ldg64, TPB) {
  unsigned long long* t = reinterpret_cast<unsigned long long*>(a.g) + (size_t)blockIdx.x * ROWS * 32;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < ROWS * 32; i += TPB) t[i] = (unsigned long long)(t + mem_row(1, 0, i >> 5) * 32 + (i & 31));
  unsigned long long x[U];
#pragma unroll
  for (int u = 0; u < U; u++) x[u] = (unsigned long long)(t + mem_row(u, w, 0) * 32 + lane);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) asm volatile("ld.global.ca.u64 %0, [%0];" : "+l"(x[u]));
  }
  OP_END
  unsigned long long s = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) s ^= x[u];
  OP_SINK(s ^ (s >> 32))
}

OP_KERNEL(a_atoms, TPB) {
  extern __shared__ __align__(128) unsigned sb[];
  const unsigned base = (unsigned)__cvta_generic_to_shared(sb);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < U * TPB; i += TPB) sb[i] = (unsigned)i * 0x9E3779B9u;
  unsigned x[U];
#pragma unroll
  for (int u = 0; u < U; u++) x[u] = threadIdx.x * 2654435761u + u;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++)
      asm volatile("atom.shared.add.u32 %0, [%1], %0;" : "+r"(x[u]) : "r"(base + atoms_off(u, lane, w, it)) : "memory");
  }
  OP_END
  unsigned s = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) s ^= x[u];
  OP_SINK(s)
}

template <bool RET>
__device__ __forceinline__ void atomg_body(const OpArgs& a) {
  unsigned* g = reinterpret_cast<unsigned*>(a.g);
  const unsigned gtid = blockIdx.x * TPB + threadIdx.x, n = gridDim.x * TPB;
  unsigned x[U];
#pragma unroll
  for (int u = 0; u < U; u++) x[u] = gtid * 2654435761u + u;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      unsigned* p = g + (size_t)u * n + gtid;
      if constexpr (RET) asm volatile("atom.global.add.u32 %0, [%1], %0;" : "+r"(x[u]) : "l"(p) : "memory");
      else asm volatile("red.global.add.u32 [%0], %1;" ::"l"(p), "r"(x[u] + it) : "memory");
    }
  }
  OP_END
  unsigned s = gtid;
#pragma unroll
  for (int u = 0; u < U; u++) s ^= x[u];
  OP_SINK(s)
}
OP_KERNEL(a_atomg, TPB) { atomg_body<true>(a); }
OP_KERNEL(a_redg, TPB) { atomg_body<false>(a); }

static const OpProbe PROBES[] = {
    {.name = "d_lds", .kernel = (const void*)d_lds, .tpb = TPB, .expect = "LDS=8", .smem_ops = "LDS*8",
     .smem = lds_off, .dyn_smem = ROWS * 128, .iters = 512, .note = "c=other w=LDS:1"},
    {.name = "d_lds128", .kernel = (const void*)d_lds128, .tpb = TPB, .expect = "LDS.128=8", .smem_ops = "LDS.128*8",
     .smem = lds128_off, .dyn_smem = ROWS * 512, .iters = 512, .note = "c=other w=LDS.128:4"},
    {.name = "d_ldg64", .kernel = (const void*)d_ldg64, .tpb = TPB, .expect = "LDG.E.64.STRONG.SM=8", .iters = 512,
     .note = "c=other w=LDG.E.64.STRONG.SM:2"},
    {.name = "a_atoms", .kernel = (const void*)a_atoms, .tpb = TPB, .expect = "ATOMS.ADD=8", .smem_ops = "ATOMS.ADD*8",
     .smem = atoms_off, .dyn_smem = U * TPB * 4, .iters = 256, .note = "c=other w=ATOMS.ADD:2"},
    {.name = "a_atomg", .kernel = (const void*)a_atomg, .tpb = TPB, .expect = "ATOMG.E.ADD.STRONG.GPU=8", .iters = 128,
     .note = "c=other w=ATOMG.E.ADD.STRONG.GPU:2"},
    {.name = "a_redg", .kernel = (const void*)a_redg, .tpb = TPB, .expect = "REDG.E.ADD.STRONG.GPU=8", .iters = 128,
     .note = "c=other w=REDG.E.ADD.STRONG.GPU:1"},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
