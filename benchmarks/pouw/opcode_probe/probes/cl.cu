// Distributed shared memory atomics and reductions in a 2-CTA cluster: RS-W′'s cl.cu (round 7, pi_32) DA and DR forms on
// probe.cuh, mode 0 (each lane its own word), into the peer CTA's shared memory (peer) or the CTA's own through its
// cluster address (self). One 1024-thread block an SM, 8 an iteration. The loop and a cluster barrier after it are
// timed, so the reductions are timed to completion; a cluster barrier before it waits for the peer's shared memory.
// cl.cu's RA (red.async + mbarrier), SU (sured) and BR (bulk reductions) forms aren't ported yet.
#include "../probe.cuh"

constexpr int TPB = 1024, U = 8, SMEM = 64 * 1024;
enum { D_ATOM_ADD, D_ATOM_CAS, D_RED_ADD };

__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned rank) {
  unsigned r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__host__ __device__ inline unsigned cl_off(int slot, int lane, int warp, int) {
  return ((unsigned)(slot * TPB + warp * 32 + lane) * 4u) & (SMEM - 1);
}

template <int K, bool PEER>
__device__ __forceinline__ void cl_body(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned char sb[];
  for (int i = threadIdx.x; i < SMEM / 4; i += TPB) reinterpret_cast<unsigned*>(sb)[i] = i * 2654435761u;
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const unsigned base = mapa((unsigned)__cvta_generic_to_shared(sb), PEER ? rank ^ 1u : rank);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const unsigned gtid = blockIdx.x * TPB + threadIdx.x;
  unsigned acc = gtid, prev[U];
#pragma unroll
  for (int u = 0; u < U; u++) prev[u] = cl_off(u, lane, w, 0) / 4 * 2654435761u;
  cluster_sync();
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const unsigned addr = base + cl_off(u, lane, w, it), v = gtid * 2654435761u ^ (unsigned)(it * U + u + 1);
      unsigned r = 0;
      if constexpr (K == D_ATOM_ADD)
        asm volatile("atom.shared::cluster.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(addr), "r"(v) : "memory");
      if constexpr (K == D_ATOM_CAS) {
        asm volatile("atom.shared::cluster.cas.b32 %0, [%1], %2, %3;" : "=r"(r) : "r"(addr), "r"(prev[u]), "r"(v)
                     : "memory");
        prev[u] = r == prev[u] ? v : r;
      }
      if constexpr (K == D_RED_ADD) asm volatile("red.shared::cluster.add.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
      acc ^= r;
    }
  }
  cluster_sync();
  OP_END
  OP_SINK(acc)
}

#define CL_KERNEL(name, K, PEER) \
  extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(TPB, 1) name(OpArgs a) { cl_body<K, PEER>(a); }
CL_KERNEL(dsm_atom_add_peer, D_ATOM_ADD, true)
CL_KERNEL(dsm_atom_add_self, D_ATOM_ADD, false)
CL_KERNEL(dsm_atom_cas_peer, D_ATOM_CAS, true)
CL_KERNEL(dsm_atom_cas_self, D_ATOM_CAS, false)
CL_KERNEL(dsm_red_add_peer, D_RED_ADD, true)
CL_KERNEL(dsm_red_add_self, D_RED_ADD, false)

#define CL_ROW(nm, K, SASS, IT)                                                                                     \
  {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .kind = K, .row = "ROW_" K, .expect = SASS "=8", .words = 32, \
   .dyn_smem = SMEM, .cluster = 2, .iters = IT}

static const OpProbe PROBES[] = {
    CL_ROW(dsm_atom_add_peer, "DSMATOM32P", "ATOM.E.ADD.STRONG.GPU", 16),
    CL_ROW(dsm_atom_add_self, "DSMATOM32", "ATOM.E.ADD.STRONG.GPU", 256),
    CL_ROW(dsm_atom_cas_peer, "DSMATOM32P", "ATOM.E.CAS.STRONG.GPU", 16),
    CL_ROW(dsm_atom_cas_self, "DSMATOM32", "ATOM.E.CAS.STRONG.GPU", 256),
    // red.shared::cluster lowers to the same ATOM.E as the returning form, its result unused
    CL_ROW(dsm_red_add_peer, "DSMRED32P", "ATOM.E.ADD.STRONG.GPU", 16),
    CL_ROW(dsm_red_add_self, "DSMRED32", "ATOM.E.ADD.STRONG.GPU", 256),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
