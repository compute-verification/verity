// Atomics and reductions, uncontended (each lane its own word), pi_32, on
// probe.cuh: the mode 0 forms that lower to one instruction. One
// 1024-thread block an SM (a global form: on 16 SMs for its row, on every SM for L2red), 8 an iteration on 8 words a
// lane (shared: lane-consecutive words, no bank conflict). CAS is made to succeed: each lane compares against what it
// last saw. A returning form XORs its result into the sink; a reduction's side effect is its sink.
//
// Spaces: S shared (ATOMS), G global (ATOMG/REDG), GS generic to shared and GG generic to global (ATOM.E). The 64-bit
// shared forms other than CAS and EXCH lower to a CAS loop (ATOMS.CAST.SPIN.64) inside the timed loop, and the 64-bit
// generic ones other than CAS and EXCH to ATOM.E.<op>.64 with that loop as its fallback for a shared address: the gate
// refuses both (a nested loop), so they're not ported. Generic CAS and EXCH .64 are one ATOM.E.CAS.64 / EXCH.64 each.
#include "../probe.cuh"

constexpr int TPB = 1024, U = 8, SMEM = 64 * 1024;
enum { SP_S, SP_G, SP_GS, SP_GG };

typedef unsigned long long u64;

#define AT_S32(fn, ptx)                                                                                           \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64) {                                                          \
    unsigned r;                                                                                                   \
    asm volatile(ptx " %0, [%1], %2;" : "=r"(r) : "r"((unsigned)a), "r"((unsigned)v) : "memory");               \
    return r;                                                                                                     \
  }
#define AT_SCAS32(fn, ptx)                                                                                        \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64 c) {                                                        \
    unsigned r;                                                                                                   \
    asm volatile(ptx " %0, [%1], %2, %3;" : "=r"(r) : "r"((unsigned)a), "r"((unsigned)c), "r"((unsigned)v)       \
                 : "memory");                                                                                     \
    return r;                                                                                                     \
  }
#define AT_S64(fn, ptx)                                                                                           \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64) {                                                          \
    u64 r;                                                                                                        \
    asm volatile(ptx " %0, [%1], %2;" : "=l"(r) : "r"((unsigned)a), "l"(v) : "memory");                          \
    return r;                                                                                                     \
  }
#define AT_SCAS64(fn, ptx)                                                                                        \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64 c) {                                                        \
    u64 r;                                                                                                        \
    asm volatile(ptx " %0, [%1], %2, %3;" : "=l"(r) : "r"((unsigned)a), "l"(c), "l"(v) : "memory");              \
    return r;                                                                                                     \
  }
#define AT_G32(fn, ptx)                                                                                           \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64) {                                                          \
    unsigned r;                                                                                                   \
    asm volatile(ptx " %0, [%1], %2;" : "=r"(r) : "l"(a), "r"((unsigned)v) : "memory");                          \
    return r;                                                                                                     \
  }
#define AT_GCAS32(fn, ptx)                                                                                        \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64 c) {                                                        \
    unsigned r;                                                                                                   \
    asm volatile(ptx " %0, [%1], %2, %3;" : "=r"(r) : "l"(a), "r"((unsigned)c), "r"((unsigned)v) : "memory");    \
    return r;                                                                                                     \
  }
#define AT_G64(fn, ptx)                                                                                           \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64) {                                                          \
    u64 r;                                                                                                        \
    asm volatile(ptx " %0, [%1], %2;" : "=l"(r) : "l"(a), "l"(v) : "memory");                                    \
    return r;                                                                                                     \
  }
#define AT_GCAS64(fn, ptx)                                                                                        \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64 c) {                                                        \
    u64 r;                                                                                                        \
    asm volatile(ptx " %0, [%1], %2, %3;" : "=l"(r) : "l"(a), "l"(c), "l"(v) : "memory");                        \
    return r;                                                                                                     \
  }
#define AT_RED32(fn, ptx)                                                                                         \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64) {                                                          \
    asm volatile(ptx " [%0], %1;" ::"l"(a), "r"((unsigned)v) : "memory");                                         \
    return 0;                                                                                                     \
  }
#define AT_RED64(fn, ptx)                                                                                         \
  __device__ __forceinline__ u64 fn(u64 a, u64 v, u64) {                                                          \
    asm volatile(ptx " [%0], %1;" ::"l"(a), "l"(v) : "memory");                                                   \
    return 0;                                                                                                     \
  }

AT_S32(s_add32, "atom.shared.add.u32") AT_S32(s_min32, "atom.shared.min.s32") AT_S32(s_max32, "atom.shared.max.s32")
AT_S32(s_and32, "atom.shared.and.b32") AT_S32(s_or32, "atom.shared.or.b32") AT_S32(s_xor32, "atom.shared.xor.b32")
AT_S32(s_exch32, "atom.shared.exch.b32") AT_SCAS32(s_cas32, "atom.shared.cas.b32")
AT_S64(s_exch64, "atom.shared.exch.b64") AT_SCAS64(s_cas64, "atom.shared.cas.b64")
AT_G32(g_add32, "atom.global.add.u32") AT_G32(g_min32, "atom.global.min.s32") AT_G32(g_max32, "atom.global.max.s32")
AT_G32(g_and32, "atom.global.and.b32") AT_G32(g_or32, "atom.global.or.b32") AT_G32(g_xor32, "atom.global.xor.b32")
AT_G32(g_exch32, "atom.global.exch.b32") AT_GCAS32(g_cas32, "atom.global.cas.b32")
AT_G64(g_add64, "atom.global.add.u64") AT_G64(g_exch64, "atom.global.exch.b64") AT_GCAS64(g_cas64, "atom.global.cas.b64")
AT_RED32(r_add32, "red.global.add.u32") AT_RED32(r_min32, "red.global.min.s32") AT_RED64(r_add64, "red.global.add.u64")
AT_RED64(r_xor64, "red.global.xor.b64")
AT_G32(gs_add32, "atom.add.u32") AT_GCAS32(gs_cas32, "atom.cas.b32") AT_G32(gg_add32, "atom.add.u32")
AT_G64(gs_exch64, "atom.exch.b64") AT_GCAS64(gs_cas64, "atom.cas.b64")

__host__ __device__ inline unsigned at_soff(int slot, int lane, int warp, int, int es) {
  return ((unsigned)(slot * TPB + warp * 32 + lane) * es) & (SMEM - 1);
}
__host__ __device__ inline unsigned at_soff32(int slot, int lane, int warp, int it) { return at_soff(slot, lane, warp, it, 4); }
__host__ __device__ inline unsigned at_soff64(int slot, int lane, int warp, int it) { return at_soff(slot, lane, warp, it, 8); }

template <u64 (*RUN)(u64, u64, u64), int SP, int W, bool CAS, int N = U>
__device__ __forceinline__ void at_body(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned char sb[];
  for (int i = threadIdx.x; i < SMEM / 8; i += TPB) reinterpret_cast<u64*>(sb)[i] = i * 0x9E3779B97F4A7C15ull;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const unsigned gtid = blockIdx.x * TPB + tid, nthreads = gridDim.x * TPB;
  constexpr int ES = W / 8;
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(sb);
  u64 acc = gtid, prev[N];
#pragma unroll
  for (int u = 0; u < N; u++) prev[u] = 0;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < N; u++) {
      u64 addr;
      if constexpr (SP == SP_S) addr = sbase + at_soff(u, lane, w, it, ES);
      else if constexpr (SP == SP_GS) addr = (u64)(sb + at_soff(u, lane, w, it, ES));
      else addr = (u64)(a.g + ((u64)u * nthreads + gtid) * ES);
      const u64 v = (u64)((unsigned)gtid * 2654435761u ^ (unsigned)(it * N + u + 1)) | ((u64)(it + 1) << 40);
      const u64 r = RUN(addr, v, prev[u]);
      if constexpr (CAS) {
        const u64 seen = W == 32 ? (u64)(unsigned)r : r, want = W == 32 ? (u64)(unsigned)prev[u] : prev[u];
        prev[u] = seen == want ? v : seen;
      }
      acc ^= r;
    }
  }
  OP_END
  OP_SINK(acc ^ (acc >> 32))
}

#define AT_KERNEL(name, fn, SP, W, CAS, ...) \
  OP_KERNEL(name, TPB) { at_body<fn, SP, W, CAS __VA_OPT__(,) __VA_ARGS__>(a); }
AT_KERNEL(atoms_add32, s_add32, SP_S, 32, false)
AT_KERNEL(atoms_min32, s_min32, SP_S, 32, false)
AT_KERNEL(atoms_max32, s_max32, SP_S, 32, false)
AT_KERNEL(atoms_and32, s_and32, SP_S, 32, false)
AT_KERNEL(atoms_or32, s_or32, SP_S, 32, false)
AT_KERNEL(atoms_xor32, s_xor32, SP_S, 32, false)
AT_KERNEL(atoms_exch32, s_exch32, SP_S, 32, false)
AT_KERNEL(atoms_cas32, s_cas32, SP_S, 32, true)
AT_KERNEL(atoms_exch64, s_exch64, SP_S, 64, false)
AT_KERNEL(atoms_cas64, s_cas64, SP_S, 64, true)
AT_KERNEL(atomg_add32, g_add32, SP_G, 32, false)
AT_KERNEL(atomg_min32, g_min32, SP_G, 32, false)
AT_KERNEL(atomg_max32, g_max32, SP_G, 32, false)
AT_KERNEL(atomg_and32, g_and32, SP_G, 32, false)
AT_KERNEL(atomg_or32, g_or32, SP_G, 32, false)
AT_KERNEL(atomg_xor32, g_xor32, SP_G, 32, false)
AT_KERNEL(atomg_exch32, g_exch32, SP_G, 32, false)
AT_KERNEL(atomg_cas32, g_cas32, SP_G, 32, true)
AT_KERNEL(atomg_add64, g_add64, SP_G, 64, false)
AT_KERNEL(atomg_exch64, g_exch64, SP_G, 64, false)
AT_KERNEL(atomg_cas64, g_cas64, SP_G, 64, true, 4)  // 8 an iteration spills at 64 registers
AT_KERNEL(redg_add32, r_add32, SP_G, 32, false)
AT_KERNEL(redg_min32, r_min32, SP_G, 32, false)
AT_KERNEL(redg_add64, r_add64, SP_G, 64, false)
AT_KERNEL(redg_xor64, r_xor64, SP_G, 64, false)
AT_KERNEL(atome_gs_add32, gs_add32, SP_GS, 32, false)
AT_KERNEL(atome_gs_cas32, gs_cas32, SP_GS, 32, true)
AT_KERNEL(atome_gg_add32, gg_add32, SP_GG, 32, false)
AT_KERNEL(atome_gs_exch64, gs_exch64, SP_GS, 64, false)
AT_KERNEL(atome_gs_cas64, gs_cas64, SP_GS, 64, true)

#define AT_S(nm, K, SASS, W, OFF)                                                                                  \
  {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .kind = K, .row = "ROW_" K, .expect = SASS "=8",         \
   .words = W, .smem_ops = SASS "*8", .smem = OFF, .dyn_smem = SMEM, .iters = 256}
#define AT_GX(nm, k, K, ROW, SASS, N, W, IT, BLOCKS)                                                               \
  {.name = nm, .kernel = (const void*)k, .tpb = TPB, .kind = K, .row = ROW, .expect = SASS "=" #N, .words = W,    \
   .dyn_smem = SMEM, .iters = IT, .fn = #k, .blocks = BLOCKS}
// A global form twice, one kernel: on 16 SMs its own row (the SM's path, as the machine model's rows price it), and on every SM
// L2red, the L2's atomic throughput every SM shares (16 SMs don't reach it; 188 are bound by it).
#define AT_GN(nm, K, SASS, N, W, IT) \
  AT_GX(#nm, nm, K, "ROW_" K, SASS, N, W, IT, 16), AT_GX(#nm "_l2", nm, K, "L2red", SASS, N, W, IT, 0)
#define AT_G(nm, K, SASS, W, IT) AT_GN(nm, K, SASS, 8, W, IT)
#define AT_EK(nm, K, SASS, W, IT)                                                                                  \
  {.name = #nm, .kernel = (const void*)nm, .tpb = TPB, .kind = K, .row = "ROW_" K, .expect = SASS "=8", .words = W, \
   .dyn_smem = SMEM, .iters = IT}
#define AT_E(nm, SASS, IT) AT_EK(nm, "ATOME32", SASS, 32, IT)

static const OpProbe PROBES[] = {
    AT_S(atoms_add32, "ATOMS32", "ATOMS.ADD", 32, at_soff32),
    AT_S(atoms_min32, "ATOMS32", "ATOMS.MIN.S32", 32, at_soff32),
    AT_S(atoms_max32, "ATOMS32", "ATOMS.MAX.S32", 32, at_soff32),
    AT_S(atoms_and32, "ATOMS32", "ATOMS.AND", 32, at_soff32),
    AT_S(atoms_or32, "ATOMS32", "ATOMS.OR", 32, at_soff32),
    AT_S(atoms_xor32, "ATOMS32", "ATOMS.XOR", 32, at_soff32),
    AT_S(atoms_exch32, "ATOMS32", "ATOMS.EXCH", 32, at_soff32),
    AT_S(atoms_cas32, "ATOMS32", "ATOMS.CAS", 32, at_soff32),
    AT_S(atoms_exch64, "ATOMS64", "ATOMS.EXCH.64", 64, at_soff64),
    AT_S(atoms_cas64, "ATOMS64", "ATOMS.CAS.64", 64, at_soff64),
    AT_G(atomg_add32, "ATOMG32", "ATOMG.E.ADD.STRONG.GPU", 32, 128),
    AT_G(atomg_min32, "ATOMG32", "ATOMG.E.MIN.S32.STRONG.GPU", 32, 128),
    AT_G(atomg_max32, "ATOMG32", "ATOMG.E.MAX.S32.STRONG.GPU", 32, 128),
    AT_G(atomg_and32, "ATOMG32", "ATOMG.E.AND.STRONG.GPU", 32, 128),
    AT_G(atomg_or32, "ATOMG32", "ATOMG.E.OR.STRONG.GPU", 32, 128),
    AT_G(atomg_xor32, "ATOMG32", "ATOMG.E.XOR.STRONG.GPU", 32, 128),
    AT_G(atomg_exch32, "ATOMG32", "ATOMG.E.EXCH.STRONG.GPU", 32, 128),
    AT_G(atomg_cas32, "ATOMG32", "ATOMG.E.CAS.STRONG.GPU", 32, 128),
    AT_G(atomg_add64, "ATOMG64", "ATOMG.E.ADD.64.STRONG.GPU", 64, 128),
    AT_G(atomg_exch64, "ATOMG64", "ATOMG.E.EXCH.64.STRONG.GPU", 64, 128),
    AT_GN(atomg_cas64, "ATOMG64", "ATOMG.E.CAS.64.STRONG.GPU", 4, 64, 256),
    AT_G(redg_add32, "REDG32", "REDG.E.ADD.STRONG.GPU", 32, 128),
    AT_G(redg_min32, "REDG32", "REDG.E.MIN.S32.STRONG.GPU", 32, 128),
    AT_G(redg_add64, "REDG64", "REDG.E.ADD.64.STRONG.GPU", 64, 128),
    AT_G(redg_xor64, "REDG64", "REDG.E.XOR.64.STRONG.GPU", 64, 128),
    AT_E(atome_gs_add32, "ATOM.E.ADD.STRONG.GPU", 256),
    AT_E(atome_gs_cas32, "ATOM.E.CAS.STRONG.GPU", 256),
    {.name = "atome_gg_add32", .kernel = (const void*)atome_gg_add32, .tpb = TPB, .expect = "ATOM.E.ADD.STRONG.GPU=8",
     .words = 32, .dyn_smem = SMEM, .iters = 128,
     .note = "ATOM.E to a global address takes the L2's path; ROW_ATOME32 is the shared window's, so no census line"},
    AT_EK(atome_gs_exch64, "ATOME64", "ATOM.E.EXCH.64.STRONG.GPU", 64, 256),
    AT_EK(atome_gs_cas64, "ATOME64", "ATOM.E.CAS.64.STRONG.GPU", 64, 256),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
