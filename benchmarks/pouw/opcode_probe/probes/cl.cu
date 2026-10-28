// Distributed shared memory atomics and reductions in a 2-CTA cluster: RS-W′'s cl.cu (round 7, pi_32) DA and DR forms on
// probe.cuh, mode 0 (each lane its own word), into the peer CTA's shared memory (peer) or the CTA's own through its
// cluster address (self). One 1024-thread block an SM, 8 an iteration. The loop and a cluster barrier after it are
// timed, so the reductions are timed to completion; a cluster barrier before it waits for the peer's shared memory.
// cl.cu's RA (red.async + mbarrier), SU (sured) and BR (bulk reductions) forms aren't ported yet.
//
// The 64-bit forms (cl_body64) are the own CTA's: exch and cas lower to one ATOM.E.EXCH.64 / CAS.64 each. Every other
// 64-bit op (add, min, max, and, or, xor; atom or red, at any .sem and .scope) lowers to ATOM.E.<op>.64 followed by a
// fallback the hardware's predicate takes for the own CTA's window, QSPC.E.S and an ATOMS.CAST.SPIN.64 loop, so its
// timed loop holds 8 nested loops and the gate refuses it.
#include "../probe.cuh"

constexpr int TPB = 1024, U = 8, SMEM = 64 * 1024;
enum { D_ATOM_ADD, D_ATOM_CAS, D_RED_ADD, D_ATOM_EXCH };
typedef unsigned long long u64;

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

__host__ __device__ inline unsigned cl_off64(int slot, int lane, int warp, int) {
  return ((unsigned)(slot * TPB + warp * 32 + lane) * 8u) & (SMEM - 1);
}

template <int K>
__device__ __forceinline__ void cl_body64(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned char sb[];
  for (int i = threadIdx.x; i < SMEM / 8; i += TPB) reinterpret_cast<u64*>(sb)[i] = i * 0x9E3779B97F4A7C15ull;
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const unsigned base = mapa((unsigned)__cvta_generic_to_shared(sb), rank);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const unsigned gtid = blockIdx.x * TPB + threadIdx.x;
  u64 acc = gtid, prev[U];
#pragma unroll
  for (int u = 0; u < U; u++) prev[u] = cl_off64(u, lane, w, 0) / 8 * 0x9E3779B97F4A7C15ull;
  cluster_sync();
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const unsigned addr = base + cl_off64(u, lane, w, it);
      const u64 v = (u64)(gtid * 2654435761u ^ (unsigned)(it * U + u + 1)) | ((u64)(it + 1) << 40);
      u64 r = 0;
      if constexpr (K == D_ATOM_EXCH)
        asm volatile("atom.shared::cluster.exch.b64 %0, [%1], %2;" : "=l"(r) : "r"(addr), "l"(v) : "memory");
      if constexpr (K == D_ATOM_CAS) {
        asm volatile("atom.shared::cluster.cas.b64 %0, [%1], %2, %3;" : "=l"(r) : "r"(addr), "l"(prev[u]), "l"(v)
                     : "memory");
        prev[u] = r == prev[u] ? v : r;
      }
      if constexpr (K == D_RED_ADD) asm volatile("red.shared::cluster.add.u64 [%0], %1;" ::"r"(addr), "l"(v) : "memory");
      acc ^= r;
    }
  }
  cluster_sync();
  OP_END
  OP_SINK(acc ^ (acc >> 32))
}

#define CL_KERNEL(name, K, PEER) \
  extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(TPB, 1) name(OpArgs a) { cl_body<K, PEER>(a); }
CL_KERNEL(dsm_atom_add_peer, D_ATOM_ADD, true)
CL_KERNEL(dsm_atom_add_self, D_ATOM_ADD, false)
CL_KERNEL(dsm_atom_cas_peer, D_ATOM_CAS, true)
CL_KERNEL(dsm_atom_cas_self, D_ATOM_CAS, false)
CL_KERNEL(dsm_red_add_peer, D_RED_ADD, true)
CL_KERNEL(dsm_red_add_self, D_RED_ADD, false)
#define CL_KERNEL64(name, K) \
  extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(TPB, 1) name(OpArgs a) { cl_body64<K>(a); }
CL_KERNEL64(dsm_atom_exch64_self, D_ATOM_EXCH)
CL_KERNEL64(dsm_atom_cas64_self, D_ATOM_CAS)
CL_KERNEL64(dsm_red_add64_self, D_RED_ADD)

// The own CTA through its cluster address is ATOM.E on ROW_ATOME32 ("generic or DSMEM address"). RS-W′'s rows have none
// for the peer CTA's shared memory, so the peer forms are joint points (rows.jsonl only) until one is added.
#define CL_SELF(nm, SASS)                                                                                           \
  {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .kind = "ATOME32", .row = "ROW_ATOME32", .expect = SASS "=8", \
   .words = 32, .dyn_smem = SMEM, .cluster = 2, .iters = 256}
#define CL_PEER(nm, SASS)                                                                                           \
  {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .expect = SASS "=8", .words = 32, .dyn_smem = SMEM,         \
   .cluster = 2, .iters = 16, .note = "the peer CTA's shared memory: no row in RS-W′'s rows.csv yet"}
// The own CTA's 64-bit forms are ATOM.E.<op>.64 on ROW_ATOME64, the class RS-W′'s round 7 priced from red.add.u64 here.
#define CL_SELF64(nm, SASS, ...)                                                                                     \
  {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .kind = "ATOME64", .row = "ROW_ATOME64", .expect = SASS "=8", \
   .words = 64, .dyn_smem = SMEM, .cluster = 2, .iters = 256 __VA_OPT__(,) __VA_ARGS__}

static const OpProbe PROBES[] = {
    CL_PEER(dsm_atom_add_peer, "ATOM.E.ADD.STRONG.GPU"),
    CL_SELF(dsm_atom_add_self, "ATOM.E.ADD.STRONG.GPU"),
    CL_PEER(dsm_atom_cas_peer, "ATOM.E.CAS.STRONG.GPU"),
    CL_SELF(dsm_atom_cas_self, "ATOM.E.CAS.STRONG.GPU"),
    // red.shared::cluster lowers to the same ATOM.E as the returning form, its result unused
    CL_PEER(dsm_red_add_peer, "ATOM.E.ADD.STRONG.GPU"),
    CL_SELF(dsm_red_add_self, "ATOM.E.ADD.STRONG.GPU"),
    CL_SELF64(dsm_atom_exch64_self, "ATOM.E.EXCH.64.STRONG.GPU"),
    CL_SELF64(dsm_atom_cas64_self, "ATOM.E.CAS.64.STRONG.GPU"),
    // round 7's ROW_ATOME64 (0.29523): ATOM.E.ADD.64 and its QSPC + CAST.SPIN fallback loop
    CL_SELF64(dsm_red_add64_self, "ATOM.E.ADD.64.STRONG.GPU", .refuse = "loops",
              .note = "red.add.u64 at the own rank: a fallback CAST.SPIN loop nested in the timed loop, untimeable"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
