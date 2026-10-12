// bwchip: how much an RTX PRO 6000's on-chip stores hold, measured, for S_chip (the bits a device holds on chip when σ
// arrives).  Each sweep finds where a store stops holding a working set: the knee is its capacity, and the first size
// that runs at the next level's speed bounds it from above (a store that held that much would serve it).
//
//   icache    straight-line code of N steps (each 4 independent add-then-xor pairs, a distinct 32-bit immediate in every
//             xor: 8 instructions, 128 bytes), looped; cycles per instruction against the code's bytes, for one warp
//             and for four (one a scheduler), on one SM.  Knees: the L0 instruction cache (per scheduler), the SM's L1
//             instruction cache, any level between it and L2.  The kernels are PTX that bwchip writes and ptxas
//             assembles (run_chip.sh, which checks each cubin's SASS holds the 8·N steps).
//   icache8   icache with 8 chains (16 instructions a step), and icachef with 8 chains of FFMA (8 a step, issuable every
//             cycle): code a warp can issue faster than integer code, for an L0 that 4 chains leave hidden.
//   unified   whether code evicts constants: the constant bank chased, N steps of code, the bank chased again.
//   clevel    where the evicted constants come back from: unified's second chase with and without the other SMs
//             thrashing L2 in between.
//   --ur      whether uniform register URn holds a value: urt rewritten (chip_ur.py) into chosen URs.
//   const     a pointer chase through __constant__ memory (64 KiB, the bank's size), one thread: cycles per load against
//             the footprint, at strides of 64 and 256 bytes (the line size from the ratio of the knees).
//   tlb       a pointer chase over N pages of a 16 GiB allocation, one load a page (each in its own L2 set, L1
//             bypassed, all held by L2), at 64 KiB and 2 MiB strides: cycles per load against N; the knees are the
//             TLB levels' entries.
//   l1        a pointer chase through F bytes on one SM, cached in L1 (no shared memory carved out): the L1's capacity.
//   l2        a random pointer chase over F bytes by 4,096 chasers in 128-byte lines: the mean latency against F, from
//             all hits to all misses, and the bytes L2 retained h(F)·F with h from the latencies at 16 MiB and 2 GiB.
//
// Every kernel's result is checked against the host's (the chain's final state, the chase's final index) before its
// time is reported; a mismatch exits 3.  Writes --out/chip.json.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define OK(x)                                                                                           \
  do {                                                                                                  \
    cudaError_t e_ = (x);                                                                               \
    if (e_ != cudaSuccess) {                                                                            \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(4);                                                                                          \
    }                                                                                                   \
  } while (0)

namespace chip {

// ---------------------------------------------------------------------------------------------------------------------
// The instruction cache: code of N steps.

__host__ __device__ constexpr uint32_t imm(uint32_t i) {
  i *= 0x9E3779B1u;
  i ^= i >> 15;
  i *= 0x85EBCA77u;
  i ^= i >> 13;
  return i & 0x7fffffffu;
}

// nvcc takes minutes over straight-line code this long, so bwchip writes each kernel as PTX (--emit DIR), run_chip.sh
// assembles it with ptxas (seconds) and records its SASS, and the sweeps load the cubins.
static const int ICACHE_STEPS[] = {16,  32,  48,  64,  96,  128, 192,  256,  320,  384,  448,  512,  640,  768,
                                   832, 896, 960, 1024, 1088, 1152, 1280, 1536, 2048, 3072, 4096, 6144, 8192};
// With 8 chains a step is 16 instructions, 256 bytes, and a warp can issue every cycle: what 4 chains (one issue every
// other cycle) leave hidden behind the L1 instruction cache, an L0 that can't keep up with it.
static const int ICACHE8_STEPS[] = {4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048};

static const char* const CHAIN[8] = {"%x0", "%x1", "%x2", "%x3", "%x4", "%x5", "%x6", "%x7"};

static std::string ptx_head() { return ".version 8.7\n.target sm_120\n.address_size 64\n\n"; }

// n steps of `chains` chains, each step x_k = (x_k + y) ^ imm(chains·step + k).
static void ptx_steps(std::string& s, int n, int chains) {
  s.reserve(s.size() + (size_t)n * chains * 48);
  char line[96];
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < chains; ++k) {
      snprintf(line, sizeof line, " add.u32 %s, %s, %%y;\n xor.b32 %s, %s, %u;\n", CHAIN[k], CHAIN[k], CHAIN[k],
               CHAIN[k], imm(chains * i + k));
      s += line;
    }
}

// Thread t's chains to out[chains·t ..], from x0 .. x_{chains-1}.
static void ptx_store_chains(std::string& s, int chains) {
  s += " mul.wide.u32 %w, %i, " + std::to_string(4 * chains) + ";\n add.u64 %w, %w, %o;\n";
  for (int k = 0; k < chains; k += 4)
    s += " st.global.v4.u32 [%w+" + std::to_string(4 * k) + "], {%x" + std::to_string(k) + ", %x" +
         std::to_string(k + 1) + ", %x" + std::to_string(k + 2) + ", %x" + std::to_string(k + 3) + "};\n";
}

// icache(out, cycles, reps, y): reps passes over the n steps (pass 0 cold and untimed); thread t writes its chains to
// out[chains·t ..] and the timed passes' cycles to cycles[t].
static std::string icache_ptx(int n, int chains) {
  std::string s = ptx_head() +
                  ".visible .entry icache(.param .u64 out, .param .u64 cyc, .param .u32 reps, .param .u32 y)\n{\n"
                  " .reg .b32 %x<8>, %y, %r, %n, %i;\n .reg .b64 %t0, %t1, %o, %q, %w;\n .reg .pred %p, %z;\n"
                  " ld.param.u64 %o, [out];\n ld.param.u64 %q, [cyc];\n ld.param.u32 %n, [reps];\n"
                  " ld.param.u32 %y, [y];\n cvta.to.global.u64 %o, %o;\n cvta.to.global.u64 %q, %q;\n"
                  " mov.u32 %i, %tid.x;\n";
  for (int k = 0; k < chains; ++k) s += " add.u32 %x" + std::to_string(k) + ", %i, " + std::to_string(k) + ";\n";
  s += " mov.u32 %r, 0;\n mov.u64 %t0, 0;\nLOOP:\n setp.eq.u32 %z, %r, 1;\n @%z mov.u64 %t0, %clock64;\n";
  ptx_steps(s, n, chains);
  s += " add.u32 %r, %r, 1;\n setp.lt.u32 %p, %r, %n;\n @%p bra LOOP;\n mov.u64 %t1, %clock64;\n"
       " sub.u64 %t1, %t1, %t0;\n";
  ptx_store_chains(s, chains);
  s += " mul.wide.u32 %w, %i, 8;\n add.u64 %w, %w, %q;\n st.global.u64 [%w], %t1;\n ret;\n}\n";
  return s;
}

static void icache_host(uint32_t tid, int n, int chains, int reps, uint32_t y, uint32_t* x) {
  for (int k = 0; k < chains; ++k) x[k] = tid + k;
  for (int r = 0; r < reps; ++r)
    for (int s = 0; s < n; ++s)
      for (int k = 0; k < chains; ++k) x[k] = (x[k] + y) ^ imm(chains * s + k);
}

// Whether constants and code share a cache: unified(out, cyc, warm, n, y, mask) chases its .const bank `warm` times,
// runs n steps of code once (seeded by the chase through `mask`, which is 0), then chases the bank again, n loads
// timed (seeded by the code).  If the code evicts the constants, the second chase runs at the next level's latency.
static const int UNIFIED_STEPS[] = {0, 128, 256, 384, 512, 640, 768, 1024, 1536, 2048, 4096};

static std::string unified_ptx(int n) {
  std::string s = ptx_head() + ".const .align 4 .b32 cbank[16384];\n\n" +
                  ".visible .entry unified(.param .u64 out, .param .u64 cyc, .param .u32 warm, .param .u32 n, "
                  ".param .u32 y, .param .u32 mask)\n{\n"
                  " .reg .b32 %x<4>, %y, %i, %j, %k, %wm, %n, %m, %t;\n .reg .b64 %base, %p, %o, %q, %w, %t0, %t1;\n"
                  " .reg .pred %z;\n"
                  " ld.param.u64 %o, [out];\n ld.param.u64 %q, [cyc];\n ld.param.u32 %wm, [warm];\n"
                  " ld.param.u32 %n, [n];\n ld.param.u32 %y, [y];\n ld.param.u32 %m, [mask];\n"
                  " cvta.to.global.u64 %o, %o;\n cvta.to.global.u64 %q, %q;\n mov.u64 %base, cbank;\n"
                  " mov.u32 %i, %tid.x;\n and.b32 %j, %i, %m;\n mov.u32 %k, 0;\n"  // a per-thread j: both chases LDC

                  "WARM:\n mul.wide.u32 %p, %j, 4;\n add.u64 %p, %p, %base;\n ld.const.u32 %j, [%p];\n"
                  " add.u32 %k, %k, 1;\n setp.lt.u32 %z, %k, %wm;\n @%z bra WARM;\n"
                  " and.b32 %t, %j, %m;\n";
  for (int k = 0; k < 4; ++k)
    s += " add.u32 %x" + std::to_string(k) + ", %i, " + std::to_string(k) + ";\n xor.b32 %x" + std::to_string(k) +
         ", %x" + std::to_string(k) + ", %t;\n";
  ptx_steps(s, n, 4);
  s += " and.b32 %t, %x0, %m;\n xor.b32 %j, %j, %t;\n mov.u32 %k, 0;\n mov.u64 %t0, 0;\n"
       "TIMED:\n setp.eq.u32 %z, %k, 1;\n @%z mov.u64 %t0, %clock64;\n mul.wide.u32 %p, %j, 4;\n"
       " add.u64 %p, %p, %base;\n ld.const.u32 %j, [%p];\n add.u32 %k, %k, 1;\n setp.lt.u32 %z, %k, %n;\n"
       " @%z bra TIMED;\n mov.u64 %t1, %clock64;\n sub.u64 %t1, %t1, %t0;\n";
  ptx_store_chains(s, 4);
  s += " st.global.u32 [%o+16], %j;\n st.global.u64 [%q], %t1;\n ret;\n}\n";
  return s;
}

// Where constants go when code evicts them: clevel(out, cyc, warm, n, y, mask, scrub, words, flags, thrash) is unified
// with STEPS steps in block 0's first warp (its j at out[128], its cycles at cyc[lane]), and when thrash is 1 every
// other block (one an SM) reads its `words` of `scrub` through
// L2 (four times L2's size in all) after block 0's code and before its timed chase.  If the constants then come back
// at the latency they had without the thrash, the level that holds them is not L2.
constexpr int CLEVEL_STEPS = 2048;

static std::string clevel_ptx() {
  std::string s = ptx_head() + ".const .align 4 .b32 cbank[16384];\n\n" +
                  ".visible .entry clevel(.param .u64 out, .param .u64 cyc, .param .u32 warm, .param .u32 n, "
                  ".param .u32 y, .param .u32 mask, .param .u64 scrub, .param .u64 words, .param .u64 flags, "
                  ".param .u32 thrash)\n{\n"
                  " .reg .b32 %x<4>, %y, %i, %j, %k, %wm, %n, %m, %t, %th, %c, %b, %nb, %acc, %v;\n"
                  " .reg .b64 %base, %p, %o, %q, %w, %t0, %t1, %sc, %wd, %fl, %e, %end, %step;\n .reg .pred %z;\n"
                  " ld.param.u64 %o, [out];\n ld.param.u64 %q, [cyc];\n ld.param.u32 %wm, [warm];\n"
                  " ld.param.u32 %n, [n];\n ld.param.u32 %y, [y];\n ld.param.u32 %m, [mask];\n"
                  " ld.param.u64 %sc, [scrub];\n ld.param.u64 %wd, [words];\n ld.param.u64 %fl, [flags];\n"
                  " ld.param.u32 %th, [thrash];\n cvta.to.global.u64 %o, %o;\n cvta.to.global.u64 %q, %q;\n"
                  " cvta.to.global.u64 %sc, %sc;\n cvta.to.global.u64 %fl, %fl;\n mov.u32 %b, %ctaid.x;\n"
                  " mov.u32 %nb, %nctaid.x;\n mov.u32 %i, %tid.x;\n setp.ne.u32 %z, %b, 0;\n @%z bra THRASH;\n"
                  " setp.ge.u32 %z, %i, 32;\n @%z bra DONE;\n mov.u64 %base, cbank;\n"
                  " and.b32 %j, %i, %m;\n mov.u32 %k, 0;\n"  // warp 0, a per-thread j: both chases LDC
                  "WARM:\n mul.wide.u32 %p, %j, 4;\n add.u64 %p, %p, %base;\n ld.const.u32 %j, [%p];\n"
                  " add.u32 %k, %k, 1;\n setp.lt.u32 %z, %k, %wm;\n @%z bra WARM;\n and.b32 %t, %j, %m;\n";
  for (int k = 0; k < 4; ++k)
    s += " add.u32 %x" + std::to_string(k) + ", %i, " + std::to_string(k) + ";\n xor.b32 %x" + std::to_string(k) +
         ", %x" + std::to_string(k) + ", %t;\n";
  ptx_steps(s, CLEVEL_STEPS, 4);
  s += " and.b32 %t, %x0, %m;\n xor.b32 %j, %j, %t;\n setp.eq.u32 %z, %th, 0;\n @%z bra CHASE;\n"
       " st.volatile.global.u32 [%fl], 1;\n sub.u32 %nb, %nb, 1;\n"
       "WAIT0:\n ld.volatile.global.u32 %c, [%fl+4];\n setp.lt.u32 %z, %c, %nb;\n @%z bra WAIT0;\n"
       "CHASE:\n mov.u32 %k, 0;\n mov.u64 %t0, 0;\n"
       "TIMED:\n setp.eq.u32 %z, %k, 1;\n @%z mov.u64 %t0, %clock64;\n mul.wide.u32 %p, %j, 4;\n"
       " add.u64 %p, %p, %base;\n ld.const.u32 %j, [%p];\n add.u32 %k, %k, 1;\n setp.lt.u32 %z, %k, %n;\n"
       " @%z bra TIMED;\n mov.u64 %t1, %clock64;\n sub.u64 %t1, %t1, %t0;\n";
  ptx_store_chains(s, 4);
  s += " st.global.u32 [%o+512], %j;\n mul.wide.u32 %w, %i, 8;\n add.u64 %w, %w, %q;\n st.global.u64 [%w], %t1;\n"
       " bra DONE;\n"
       "THRASH:\n setp.eq.u32 %z, %th, 0;\n @%z bra DONE;\n setp.ne.u32 %z, %i, 0;\n @%z bra GO;\n"
       "WAIT1:\n ld.volatile.global.u32 %c, [%fl];\n setp.eq.u32 %z, %c, 0;\n @%z bra WAIT1;\n"
       "GO:\n barrier.sync 0;\n sub.u32 %c, %b, 1;\n mul.wide.u32 %e, %c, 4;\n mul.lo.u64 %e, %e, %wd;\n"
       " add.u64 %e, %e, %sc;\n mul.wide.u32 %p, %i, 4;\n add.u64 %p, %p, %e;\n mul.lo.u64 %end, %wd, 4;\n"
       " add.u64 %end, %end, %e;\n mov.u32 %v, %ntid.x;\n mul.wide.u32 %step, %v, 4;\n mov.u32 %acc, 0;\n"
       "READ:\n setp.ge.u64 %z, %p, %end;\n @%z bra READ_DONE;\n ld.global.cg.u32 %v, [%p];\n"
       " xor.b32 %acc, %acc, %v;\n add.u64 %p, %p, %step;\n bra READ;\n"
       "READ_DONE:\n mul.wide.u32 %p, %b, 4;\n add.u64 %p, %p, %fl;\n add.u64 %p, %p, 64;\n"
       " atom.global.xor.b32 %v, [%p], %acc;\n barrier.sync 0;\n setp.ne.u32 %z, %i, 0;\n @%z bra DONE;\n"
       " atom.global.add.u32 %v, [%fl+4], 1;\n"
       "DONE:\n ret;\n}\n";
  return s;
}

// Code that can issue every cycle: icachef(out, cycles, reps, h) is icache with 8 chains of FFMA, x_k = x_k·h + c, one
// instruction a chain a step (128 bytes a step); h = 0.5 and c in [1, 2) keep every x_k in [0, 4).
static const int ICACHEF_STEPS[] = {8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 1024, 1280, 2048};

static uint32_t ffma_c(int i) { return 0x3f800000u | (imm(i) & 0x7fffffu); }

static std::string icachef_ptx(int n) {
  std::string s = ptx_head() +
                  ".visible .entry icache(.param .u64 out, .param .u64 cyc, .param .u32 reps, .param .f32 h)\n{\n"
                  " .reg .f32 %x<8>, %h;\n .reg .b32 %r, %n, %i, %u;\n .reg .b64 %t0, %t1, %o, %q, %w;\n"
                  " .reg .pred %p, %z;\n ld.param.u64 %o, [out];\n ld.param.u64 %q, [cyc];\n"
                  " ld.param.u32 %n, [reps];\n ld.param.f32 %h, [h];\n cvta.to.global.u64 %o, %o;\n"
                  " cvta.to.global.u64 %q, %q;\n mov.u32 %i, %tid.x;\n";
  for (int k = 0; k < 8; ++k)
    s += " add.u32 %u, %i, " + std::to_string(k) + ";\n cvt.rn.f32.u32 %x" + std::to_string(k) + ", %u;\n";
  s += " mov.u32 %r, 0;\n mov.u64 %t0, 0;\nLOOP:\n setp.eq.u32 %z, %r, 1;\n @%z mov.u64 %t0, %clock64;\n";
  s.reserve(s.size() + (size_t)n * 8 * 40);
  char line[80];
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 8; ++k) {
      snprintf(line, sizeof line, " fma.rn.f32 %%x%d, %%x%d, %%h, 0f%08X;\n", k, k, ffma_c(8 * i + k));
      s += line;
    }
  s += " add.u32 %r, %r, 1;\n setp.lt.u32 %p, %r, %n;\n @%p bra LOOP;\n mov.u64 %t1, %clock64;\n"
       " sub.u64 %t1, %t1, %t0;\n mul.wide.u32 %w, %i, 32;\n add.u64 %w, %w, %o;\n"
       " st.global.v4.f32 [%w], {%x0, %x1, %x2, %x3};\n st.global.v4.f32 [%w+16], {%x4, %x5, %x6, %x7};\n"
       " mul.wide.u32 %w, %i, 8;\n add.u64 %w, %w, %q;\n st.global.u64 [%w], %t1;\n ret;\n}\n";
  return s;
}

static void icachef_host(uint32_t tid, int n, int reps, uint32_t* out) {
  float x[8];
  for (int k = 0; k < 8; ++k) x[k] = (float)(tid + k);
  for (int r = 0; r < reps; ++r)
    for (int s = 0; s < n; ++s)
      for (int k = 0; k < 8; ++k) {
        const uint32_t cb = ffma_c(8 * s + k);
        float c;
        memcpy(&c, &cb, 4);
        x[k] = fmaf(x[k], 0.5f, c);
      }
  memcpy(out, x, 32);
}

// urt(out, v1, pad, v2): thread t writes t·v1 and t ^ v2 to out[2t], out[2t+1]; chip_ur.py moves v1 and v2 into
// chosen uniform registers.
static std::string urt_ptx() {
  return ptx_head() +
         ".visible .entry urt(.param .u64 out, .param .u32 v1, .param .u64 pad, .param .u32 v2)\n{\n"
         " .reg .b32 %v1, %v2, %t, %x1, %x2;\n .reg .b64 %o, %w;\n ld.param.u64 %o, [out];\n"
         " ld.param.u32 %v1, [v1];\n ld.param.u32 %v2, [v2];\n cvta.to.global.u64 %o, %o;\n mov.u32 %t, %tid.x;\n"
         " mul.lo.u32 %x1, %t, %v1;\n xor.b32 %x2, %t, %v2;\n mul.wide.u32 %w, %t, 8;\n add.u64 %w, %w, %o;\n"
         " st.global.v2.u32 [%w], {%x1, %x2};\n ret;\n}\n";
}

// ---------------------------------------------------------------------------------------------------------------------
// Pointer chases.

constexpr int CONST_WORDS = 16384;
__constant__ uint32_t cbank[CONST_WORDS];

__global__ void const_chase(uint32_t start, int warm, int n, uint32_t* out, unsigned long long* cycles) {
  uint32_t i = start;
#pragma unroll 1
  for (int k = 0; k < warm; ++k) i = cbank[i];
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int k = 0; k < n; ++k) i = cbank[i];
  const unsigned long long t1 = clock64();
  out[0] = i;
  cycles[0] = t1 - t0;
}

// One thread chasing 32-bit indices (in 4-byte words) through `p`, through L2 only (cg) or L1 too (ca).
template <bool L1>
__global__ void global_chase(const uint32_t* __restrict__ p, uint64_t start, int warm, int n, uint64_t* out,
                             unsigned long long* cycles) {
  uint64_t i = start;
#pragma unroll 1
  for (int k = 0; k < warm; ++k) i = L1 ? __ldca(p + i) : __ldcg(p + i);
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int k = 0; k < n; ++k) i = L1 ? __ldca(p + i) : __ldcg(p + i);
  const unsigned long long t1 = clock64();
  out[0] = i;
  cycles[0] = t1 - t0;
}

// Many chasers, one a warp, over one random cycle of line indices (each line's first word holds the next line).
__global__ void l2_chase(const uint32_t* __restrict__ p, const uint32_t* __restrict__ starts, int n, uint32_t* out,
                         unsigned long long* cycles) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (threadIdx.x % 32) return;
  uint32_t i = starts[w];
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int k = 0; k < n; ++k) i = __ldcg(p + (uint64_t)i * 32);
  const unsigned long long t1 = clock64();
  out[w] = i;
  cycles[w] = t1 - t0;
}

// ---------------------------------------------------------------------------------------------------------------------
// Host.

static std::string num(double x) {
  char b[64];
  snprintf(b, sizeof b, "%.6g", x);
  return b;
}

static int fail(const std::string& what) {
  fprintf(stderr, "GATE FAILED: %s\n", what.c_str());
  printf("GATE FAILED: %s\n", what.c_str());
  return 3;
}

struct Dev {
  int sms = 0, clock_khz = 0;
  size_t l2 = 0;
};

static int write_file(const std::string& path, const std::string& s) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return fail("cannot write " + path);
  fwrite(s.data(), 1, s.size(), f);
  fclose(f);
  return 0;
}

// icache_N.ptx (4 chains), icache8_N.ptx (8 chains), unified_N.ptx and urt.ptx.
static int emit_ptx(const std::string& dir) {
  int rc = 0;
  for (int n : ICACHE_STEPS) rc |= write_file(dir + "/icache_" + std::to_string(n) + ".ptx", icache_ptx(n, 4));
  for (int n : ICACHE8_STEPS) rc |= write_file(dir + "/icache8_" + std::to_string(n) + ".ptx", icache_ptx(n, 8));
  for (int n : UNIFIED_STEPS) rc |= write_file(dir + "/unified_" + std::to_string(n) + ".ptx", unified_ptx(n));
  for (int n : ICACHEF_STEPS) rc |= write_file(dir + "/icachef_" + std::to_string(n) + ".ptx", icachef_ptx(n));
  rc |= write_file(dir + "/clevel_" + std::to_string(CLEVEL_STEPS) + ".ptx", clevel_ptx());
  rc |= write_file(dir + "/urt.ptx", urt_ptx());
  return rc;
}

static cudaKernel_t load_kernel(const std::string& path, const char* name, cudaLibrary_t* lib) {
  cudaKernel_t fn;
  OK(cudaLibraryLoadFromFile(lib, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  OK(cudaLibraryGetKernel(&fn, *lib, name));
  return fn;
}

static int run_icachef(std::string& j, const std::string& dir) {
  uint32_t* out;
  unsigned long long* cyc;
  OK(cudaMalloc(&out, 128 * 32));
  OK(cudaMalloc(&cyc, 128 * 8));
  j += "\"icachef\":{\"bytes_per_step\":128,\"instructions_per_step\":8,\"runs\":[";
  bool first = true;
  for (int n : ICACHEF_STEPS) {
    cudaLibrary_t lib;
    cudaKernel_t fn = load_kernel(dir + "/icachef_" + std::to_string(n) + ".cubin", "icache", &lib);
    const int reps = std::max(4, (1 << 21) / (8 * n));
    for (int warps : {1, 4, 8}) {
      std::vector<double> cpi;
      for (int t = 0; t < 3; ++t) {
        int r = reps;
        float h = 0.5f;
        void* args[] = {&out, &cyc, &r, &h};
        OK(cudaLaunchKernel((const void*)fn, dim3(1), dim3(32 * warps), args, 0, 0));
        OK(cudaGetLastError());
        OK(cudaDeviceSynchronize());
        std::vector<uint32_t> got(8 * 32 * warps);
        std::vector<unsigned long long> cy(32 * warps);
        OK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(cy.data(), cyc, cy.size() * 8, cudaMemcpyDeviceToHost));
        for (uint32_t tid = 0; tid < 32u * warps; tid += 7) {
          uint32_t x[8];
          icachef_host(tid, n, reps, x);
          if (memcmp(x, &got[8 * tid], 32)) return fail("icachef result, n=" + std::to_string(n));
        }
        double worst = 0;
        for (auto x : cy) worst = std::max(worst, (double)x);
        cpi.push_back(worst / ((double)(reps - 1) * 8 * n));
      }
      std::sort(cpi.begin(), cpi.end());
      j += std::string(first ? "" : ",") + "{\"steps\":" + std::to_string(n) + ",\"code_bytes\":" +
           std::to_string(128 * n) + ",\"warps\":" + std::to_string(warps) + ",\"reps\":" + std::to_string(reps) +
           ",\"cycles_per_instruction\":" + num(cpi[1]) + ",\"min\":" + num(cpi[0]) + ",\"max\":" + num(cpi[2]) + "}";
      first = false;
    }
    OK(cudaLibraryUnload(lib));
  }
  j += "]},";
  cudaFree(out);
  cudaFree(cyc);
  return 0;
}

// The 64 KiB constant bank after 256 KiB of code, without and with the other SMs' L2 thrash between them.
static int run_clevel(std::string& j, const std::string& dir, const Dev& d) {
  cudaLibrary_t lib;
  cudaKernel_t fn = load_kernel(dir + "/clevel_" + std::to_string(CLEVEL_STEPS) + ".cubin", "clevel", &lib);
  void* bank;
  size_t bank_bytes;
  OK(cudaLibraryGetGlobal(&bank, &bank_bytes, lib, "cbank"));
  const int count = 65536 / 64, sw = 16;
  std::vector<uint32_t> img(bank_bytes / 4, 0);
  for (int k = 0; k < count; ++k) img[k * sw] = ((k + 1) % count) * sw;
  OK(cudaMemcpy(bank, img.data(), bank_bytes, cudaMemcpyHostToDevice));
  const int blocks = d.sms;
  const size_t scrub_bytes = 4 * d.l2;
  uint64_t words = scrub_bytes / 4 / (blocks - 1);
  uint32_t *out, *flags, *scrub;
  unsigned long long* cyc;
  OK(cudaMalloc(&out, 1024));
  OK(cudaMalloc(&cyc, 32 * 8));
  OK(cudaMalloc(&flags, 4096));
  OK(cudaMalloc(&scrub, words * 4 * (blocks - 1)));
  OK(cudaMemset(scrub, 0x5A, words * 4 * (blocks - 1)));
  const uint32_t y = 0x2545F491u;
  j += "\"clevel\":{\"code_bytes\":" + std::to_string(128 * CLEVEL_STEPS) +
       ",\"const_footprint\":65536,\"stride\":64,\"thrash_bytes\":" + std::to_string(words * 4 * (blocks - 1)) +
       ",\"runs\":[";
  bool first = true;
  for (uint32_t thrash : {0u, 1u, 0u, 1u}) {
    int warm = 2 * count, loads = 2 * count;
    uint32_t yy = y, mask = 0, th = thrash;
    OK(cudaMemset(flags, 0, 4096));
    void* args[] = {&out, &cyc, &warm, &loads, &yy, &mask, &scrub, &words, &flags, &th};
    OK(cudaLaunchKernel((const void*)fn, dim3(blocks), dim3(256), args, 0, 0));
    OK(cudaGetLastError());
    OK(cudaDeviceSynchronize());
    uint32_t got[129], arrived;
    unsigned long long cy[32];
    OK(cudaMemcpy(got, out, sizeof got, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(cy, cyc, sizeof cy, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(&arrived, flags + 1, 4, cudaMemcpyDeviceToHost));
    uint32_t x0[8], x31[8];
    icache_host(0, CLEVEL_STEPS, 4, 1, y, x0);
    icache_host(31, CLEVEL_STEPS, 4, 1, y, x31);
    if (memcmp(x0, got, 16) || memcmp(x31, got + 124, 16) || got[128] != (uint32_t)(((warm + loads) % count) * sw) ||
        arrived != (thrash ? (uint32_t)(blocks - 1) : 0u))
      return fail("clevel result, thrash=" + std::to_string(thrash));
    const double c = (double)*std::max_element(cy, cy + 32);
    j += std::string(first ? "" : ",") + "{\"thrash\":" + std::to_string(thrash) + ",\"cycles_per_load\":" +
         num(c / (loads - 1)) + "}";
    first = false;
  }
  j += "]},";
  OK(cudaLibraryUnload(lib));
  cudaFree(out);
  cudaFree(cyc);
  cudaFree(flags);
  cudaFree(scrub);
  return 0;
}

static int run_icache(std::string& j, const std::string& dir, int chains) {
  uint32_t* out;
  unsigned long long* cyc;
  OK(cudaMalloc(&out, 128 * 4 * chains));
  OK(cudaMalloc(&cyc, 128 * 8));
  const uint32_t y = 0x5bd1e995u;
  const int per_step = 2 * chains;
  const std::string name = chains == 4 ? "icache" : "icache" + std::to_string(chains);
  j += "\"" + name + "\":{\"bytes_per_step\":" + std::to_string(16 * per_step) +
       ",\"instructions_per_step\":" + std::to_string(per_step) + ",\"runs\":[";
  bool first = true;
  const int* steps = chains == 4 ? ICACHE_STEPS : ICACHE8_STEPS;
  const size_t count = chains == 4 ? sizeof ICACHE_STEPS / sizeof(int) : sizeof ICACHE8_STEPS / sizeof(int);
  for (size_t c = 0; c < count; ++c) {
    const int n = steps[c];
    cudaLibrary_t lib;
    cudaKernel_t fn = load_kernel(dir + "/" + name + "_" + std::to_string(n) + ".cubin", "icache", &lib);
    const int reps = std::max(4, (1 << 21) / (per_step * n));
    for (int warps : {1, 4}) {
      std::vector<double> cpi;
      for (int t = 0; t < 3; ++t) {
        int r = reps;
        uint32_t yy = y;
        void* args[] = {&out, &cyc, &r, &yy};
        OK(cudaLaunchKernel((const void*)fn, dim3(1), dim3(32 * warps), args, 0, 0));
        OK(cudaGetLastError());
        OK(cudaDeviceSynchronize());
        std::vector<uint32_t> got(chains * 32 * warps);
        std::vector<unsigned long long> cy(32 * warps);
        OK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(cy.data(), cyc, cy.size() * 8, cudaMemcpyDeviceToHost));
        for (uint32_t tid = 0; tid < 32u * warps; tid += 7) {
          uint32_t x[8];
          icache_host(tid, n, chains, reps, y, x);
          if (memcmp(x, &got[chains * tid], 4 * chains)) return fail(name + " result, n=" + std::to_string(n));
        }
        double worst = 0;
        for (auto x : cy) worst = std::max(worst, (double)x);
        cpi.push_back(worst / ((double)(reps - 1) * per_step * n));
      }
      std::sort(cpi.begin(), cpi.end());
      j += std::string(first ? "" : ",") + "{\"steps\":" + std::to_string(n) + ",\"code_bytes\":" +
           std::to_string(16 * per_step * n) + ",\"warps\":" + std::to_string(warps) + ",\"reps\":" +
           std::to_string(reps) + ",\"cycles_per_instruction\":" + num(cpi[1]) + ",\"min\":" + num(cpi[0]) +
           ",\"max\":" + num(cpi[2]) + "}";
      first = false;
    }
    OK(cudaLibraryUnload(lib));
  }
  j += "]},";
  cudaFree(out);
  cudaFree(cyc);
  return 0;
}

// For each code size, constant footprints of 16, 32 and 64 KiB at a 64-byte stride: cycles a load of the second chase.
static int run_unified(std::string& j, const std::string& dir) {
  uint32_t* out;
  unsigned long long* cyc;
  OK(cudaMalloc(&out, 64));
  OK(cudaMalloc(&cyc, 8));
  const uint32_t y = 0x2545F491u;
  j += "\"unified\":{\"bytes_per_step\":128,\"stride\":64,\"runs\":[";
  bool first = true;
  for (int n : UNIFIED_STEPS) {
    cudaLibrary_t lib;
    cudaKernel_t fn = load_kernel(dir + "/unified_" + std::to_string(n) + ".cubin", "unified", &lib);
    void* bank;
    size_t bank_bytes;
    OK(cudaLibraryGetGlobal(&bank, &bank_bytes, lib, "cbank"));
    for (int fp : {16384, 32768, 65536}) {
      const int count = fp / 64, sw = 16;
      std::vector<uint32_t> img(bank_bytes / 4, 0);
      for (int k = 0; k < count; ++k) img[k * sw] = ((k + 1) % count) * sw;
      OK(cudaMemcpy(bank, img.data(), bank_bytes, cudaMemcpyHostToDevice));
      int warm = 2 * count, loads = 2 * count;
      uint32_t yy = y, mask = 0;
      std::vector<double> lat;
      for (int t = 0; t < 3; ++t) {
        void* args[] = {&out, &cyc, &warm, &loads, &yy, &mask};
        OK(cudaLaunchKernel((const void*)fn, dim3(1), dim3(1), args, 0, 0));
        OK(cudaGetLastError());
        OK(cudaDeviceSynchronize());
        uint32_t got[5];
        unsigned long long c;
        OK(cudaMemcpy(got, out, 20, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
        uint32_t x[8];
        icache_host(0, n, 4, 1, y, x);
        if (memcmp(x, got, 16) || got[4] != (uint32_t)(((warm + loads) % count) * sw))
          return fail("unified result, n=" + std::to_string(n));
        lat.push_back((double)c / (loads - 1));
      }
      std::sort(lat.begin(), lat.end());
      j += std::string(first ? "" : ",") + "{\"steps\":" + std::to_string(n) + ",\"code_bytes\":" +
           std::to_string(128 * n) + ",\"const_footprint\":" + std::to_string(fp) + ",\"cycles_per_load\":" +
           num(lat[1]) + "}";
      first = false;
    }
    OK(cudaLibraryUnload(lib));
  }
  j += "]},";
  cudaFree(out);
  cudaFree(cyc);
  return 0;
}

// One process a pair: urt rewritten (chip_ur.py) to hold v1 in URa and v2 in URb.  Prints whether both came back.
static int run_ur(const std::string& cubin, int a, int b) {
  OK(cudaSetDevice(0));
  cudaLibrary_t lib;
  cudaKernel_t fn = load_kernel(cubin, "urt", &lib);
  uint32_t* out;
  OK(cudaMalloc(&out, 32 * 8));
  OK(cudaMemset(out, 0, 32 * 8));
  uint32_t v1 = 0x9E3779B1u, v2 = 0x7F4A7C15u;
  uint64_t pad = 0;
  void* args[] = {&out, &v1, &pad, &v2};
  OK(cudaLaunchKernel((const void*)fn, dim3(1), dim3(32), args, 0, 0));
  OK(cudaGetLastError());
  OK(cudaDeviceSynchronize());
  uint32_t got[64];
  OK(cudaMemcpy(got, out, sizeof got, cudaMemcpyDeviceToHost));
  int bad1 = 0, bad2 = 0;
  for (uint32_t t = 0; t < 32; ++t) bad1 += got[2 * t] != t * v1, bad2 += got[2 * t + 1] != (t ^ v2);
  printf("{\"a\":%d,\"b\":%d,\"a_held\":%s,\"b_held\":%s,\"a_word1\":%u,\"b_word1\":%u}\n", a, b,
         bad1 ? "false" : "true", bad2 ? "false" : "true", got[2], got[3]);
  return bad1 || bad2 ? 3 : 0;
}

static int run_const(std::string& j) {
  uint32_t* out;
  unsigned long long* cyc;
  OK(cudaMalloc(&out, 4));
  OK(cudaMalloc(&cyc, 8));
  j += "\"const\":{\"bank_bytes\":" + std::to_string(CONST_WORDS * 4) + ",\"runs\":[";
  bool first = true;
  for (int stride : {64, 256})
    for (int fp : {256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 49152,
                   57344, 65536}) {
      const int count = fp / stride, sw = stride / 4;
      if (count < 2) continue;
      std::vector<uint32_t> bank(CONST_WORDS, 0);
      for (int k = 0; k < count; ++k) bank[k * sw] = ((k + 1) % count) * sw;
      OK(cudaMemcpyToSymbol(cbank, bank.data(), bank.size() * 4));
      const int warm = 2 * count, n = std::max(4096, 8 * count);
      const_chase<<<1, 1>>>(0, warm, n, out, cyc);
      OK(cudaGetLastError());
      OK(cudaDeviceSynchronize());
      std::vector<double> lat;
      for (int t = 0; t < 3; ++t) {
        const_chase<<<1, 1>>>(0, warm, n, out, cyc);
        OK(cudaDeviceSynchronize());
        uint32_t got;
        unsigned long long c;
        OK(cudaMemcpy(&got, out, 4, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
        if (got != (uint32_t)(((warm + n) % count) * sw)) return fail("const chase, footprint " + std::to_string(fp));
        lat.push_back((double)c / n);
      }
      std::sort(lat.begin(), lat.end());
      j += std::string(first ? "" : ",") + "{\"stride\":" + std::to_string(stride) + ",\"footprint\":" +
           std::to_string(fp) + ",\"cycles_per_load\":" + num(lat[1]) + "}";
      first = false;
    }
  j += "]},";
  cudaFree(out);
  cudaFree(cyc);
  return 0;
}

// A single-thread chase over `count` elements at word offsets offs[k]: the median cycles a load over three timed runs.
template <bool L1>
static int chase(const uint32_t* dev, uint32_t* host_view_unused, const std::vector<uint64_t>& offs, double* out_lat,
                 uint64_t* scratch, unsigned long long* cyc, const std::string& what) {
  (void)host_view_unused;
  const int count = (int)offs.size();
  const int warm = 2 * count, n = std::max(4096, 4 * count);
  std::vector<double> lat;
  for (int t = 0; t < 4; ++t) {
    global_chase<L1><<<1, 1>>>(dev, offs[0], warm, n, scratch, cyc);
    OK(cudaGetLastError());
    OK(cudaDeviceSynchronize());
    uint64_t got;
    unsigned long long c;
    OK(cudaMemcpy(&got, scratch, 8, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
    if (got != offs[(warm + n) % count]) return fail(what);
    if (t) lat.push_back((double)c / n);
  }
  std::sort(lat.begin(), lat.end());
  *out_lat = lat[1];
  return 0;
}

// Writes the chain offs[k] -> offs[k+1] into device memory word by word.
static void link_chain(uint32_t* dev, const std::vector<uint64_t>& offs) {
  for (size_t k = 0; k < offs.size(); ++k) {
    const uint32_t next = (uint32_t)offs[(k + 1) % offs.size()];
    OK(cudaMemcpy(dev + offs[k], &next, 4, cudaMemcpyHostToDevice));
  }
}

static int run_tlb(std::string& j) {
  const size_t span = 16ull << 30;
  uint32_t* buf;
  OK(cudaMalloc(&buf, span));
  uint64_t* scratch;
  unsigned long long* cyc;
  OK(cudaMalloc(&scratch, 8));
  OK(cudaMalloc(&cyc, 8));
  j += "\"tlb\":{\"span\":" + std::to_string(span) + ",\"runs\":[";
  bool first = true;
  for (size_t page : {64ull << 10, 2ull << 20})
    for (int count : {4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
                      6144, 8192}) {
      if ((uint64_t)count * page > span) break;
      // Element k at page k, line (k mod 512) of it: one load a page, each in its own L2 set, 4 KiB to 1 MiB in all.
      std::vector<uint64_t> offs(count);
      for (int k = 0; k < count; ++k) offs[k] = ((uint64_t)k * page + (uint64_t)(k % 512) * 128 % page) / 4;
      if (offs.back() > UINT32_MAX) break;
      link_chain(buf, offs);
      double lat;
      if (int rc = chase<false>(buf, nullptr, offs, &lat, scratch, cyc, "tlb chase")) return rc;
      j += std::string(first ? "" : ",") + "{\"page\":" + std::to_string(page) + ",\"pages\":" + std::to_string(count) +
           ",\"cycles_per_load\":" + num(lat) + "}";
      first = false;
    }
  j += "]},";
  cudaFree(buf);
  cudaFree(scratch);
  cudaFree(cyc);
  return 0;
}

static int run_l1(std::string& j) {
  OK(cudaFuncSetAttribute(global_chase<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  const size_t span = 1 << 20;
  uint32_t* buf;
  OK(cudaMalloc(&buf, span));
  uint64_t* scratch;
  unsigned long long* cyc;
  OK(cudaMalloc(&scratch, 8));
  OK(cudaMalloc(&cyc, 8));
  j += "\"l1\":{\"line\":128,\"runs\":[";
  bool first = true;
  for (int kib : {8, 16, 32, 48, 64, 80, 96, 104, 112, 120, 128, 136, 144, 160, 192, 256}) {
    const int count = kib * 1024 / 128;
    std::vector<uint64_t> offs(count);
    for (int k = 0; k < count; ++k) offs[k] = (uint64_t)k * 32;
    link_chain(buf, offs);
    double lat;
    if (int rc = chase<true>(buf, nullptr, offs, &lat, scratch, cyc, "l1 chase")) return rc;
    j += std::string(first ? "" : ",") + "{\"footprint\":" + std::to_string(kib * 1024) + ",\"cycles_per_load\":" +
         num(lat) + "}";
    first = false;
  }
  j += "]},";
  cudaFree(buf);
  cudaFree(scratch);
  cudaFree(cyc);
  return 0;
}

static int run_l2(std::string& j, const Dev& d) {
  const size_t max_bytes = 2ull << 30;
  uint32_t* buf;
  OK(cudaMalloc(&buf, max_bytes));
  const int chasers = 4096, threads = 256;
  uint32_t *starts, *out;
  unsigned long long* cyc;
  OK(cudaMalloc(&starts, chasers * 4));
  OK(cudaMalloc(&out, chasers * 4));
  OK(cudaMalloc(&cyc, chasers * 8));
  std::mt19937_64 rng(12345);
  j += "\"l2\":{\"line\":128,\"chasers\":" + std::to_string(chasers) + ",\"runs\":[";
  std::vector<std::pair<size_t, double>> lat;
  for (size_t mib : {16, 32, 64, 96, 112, 120, 128, 136, 144, 160, 192, 256, 512, 2048}) {
    const size_t bytes = mib << 20, lines = bytes / 128;
    std::vector<uint32_t> order(lines);
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<uint32_t> img(bytes / 4, 0);
    for (size_t k = 0; k < lines; ++k) img[(size_t)order[k] * 32] = order[(k + 1) % lines];
    OK(cudaMemcpy(buf, img.data(), bytes, cudaMemcpyHostToDevice));
    std::vector<uint32_t> st(chasers);
    const size_t per = lines / chasers;
    for (int w = 0; w < chasers; ++w) st[w] = order[(size_t)w * per];
    OK(cudaMemcpy(starts, st.data(), chasers * 4, cudaMemcpyHostToDevice));
    std::vector<double> runs;
    for (int t = 0; t < 3; ++t) {  // pass 0 warms (each chaser covers its share of the cycle once a pass)
      l2_chase<<<chasers * 32 / threads, threads>>>(buf, starts, (int)per, out, cyc);
      OK(cudaGetLastError());
      OK(cudaDeviceSynchronize());
      std::vector<uint32_t> got(chasers);
      std::vector<unsigned long long> c(chasers);
      OK(cudaMemcpy(got.data(), out, chasers * 4, cudaMemcpyDeviceToHost));
      OK(cudaMemcpy(c.data(), cyc, chasers * 8, cudaMemcpyDeviceToHost));
      for (int w = 0; w < chasers; w += 97)
        if (got[w] != order[((size_t)w * per + per) % lines]) return fail("l2 chase, " + std::to_string(mib) + " MiB");
      double sum = 0;
      for (auto x : c) sum += (double)x;
      if (t) runs.push_back(sum / chasers / per);
    }
    lat.push_back({bytes, 0.5 * (runs[0] + runs[1])});
  }
  const double hit = lat.front().second, miss = lat.back().second;
  bool first = true;
  for (auto& [bytes, l] : lat) {
    const double h = std::min(1.0, std::max(0.0, (miss - l) / (miss - hit)));
    j += std::string(first ? "" : ",") + "{\"footprint\":" + std::to_string(bytes) + ",\"cycles_per_load\":" + num(l) +
         ",\"hit_fraction\":" + num(h) + ",\"retained_bytes\":" + num(h * bytes) + "}";
    first = false;
  }
  j += "],\"l2_bytes_attribute\":" + std::to_string(d.l2) + "},";
  cudaFree(buf);
  cudaFree(starts);
  cudaFree(out);
  cudaFree(cyc);
  return 0;
}

}  // namespace chip

int main(int argc, char** argv) {
  std::string out = ".", only, ptx_dir, emit, ur;
  int ur_a = -1, ur_b = -1;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--out" && i + 1 < argc) out = argv[++i];
    else if (k == "--only" && i + 1 < argc) only = argv[++i];
    else if (k == "--ptx" && i + 1 < argc) ptx_dir = argv[++i];
    else if (k == "--emit" && i + 1 < argc) emit = argv[++i];
    else if (k == "--ur" && i + 3 < argc) ur = argv[++i], ur_a = atoi(argv[++i]), ur_b = atoi(argv[++i]);
    else {
      fprintf(stderr,
              "usage: bwchip --emit DIR\n"
              "       bwchip --ur URT_CUBIN A B\n"
              "       bwchip [--out DIR] [--ptx DIR_OF_CUBINS] [--only icache|icache8|unified|const|tlb|l1|l2]\n");
      return 2;
    }
  }
  if (!emit.empty()) return chip::emit_ptx(emit);
  if (!ur.empty()) return chip::run_ur(ur, ur_a, ur_b);
  OK(cudaSetDevice(0));
  cudaDeviceProp p;
  OK(cudaGetDeviceProperties(&p, 0));
  chip::Dev d;
  d.sms = p.multiProcessorCount;
  d.l2 = p.l2CacheSize;
  int clk = 0;
  OK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
  std::string j = "{\"name\":\"" + std::string(p.name) + "\",\"sms\":" + std::to_string(d.sms) +
                  ",\"l2_bytes\":" + std::to_string(p.l2CacheSize) + ",\"regs_per_sm\":" +
                  std::to_string(p.regsPerMultiprocessor) + ",\"smem_per_sm\":" +
                  std::to_string(p.sharedMemPerMultiprocessor) + ",\"max_threads_per_sm\":" +
                  std::to_string(p.maxThreadsPerMultiProcessor) + ",\"max_blocks_per_sm\":" +
                  std::to_string(p.maxBlocksPerMultiProcessor) + ",\"clock_khz_attribute\":" + std::to_string(clk) + ",";
  // A sweep whose gate fails contributes its name to "failed" and nothing else; the rest still run.
  std::string failed;
  auto sweep = [&](const char* name, auto run) {
    if (!only.empty() && only != name) return;
    std::string s;
    if (run(s)) failed += std::string(failed.empty() ? "\"" : ",\"") + name + "\"";
    else j += s;
  };
  if (!ptx_dir.empty()) {
    sweep("icache", [&](std::string& s) { return chip::run_icache(s, ptx_dir, 4); });
    sweep("icache8", [&](std::string& s) { return chip::run_icache(s, ptx_dir, 8); });
    sweep("unified", [&](std::string& s) { return chip::run_unified(s, ptx_dir); });
    sweep("icachef", [&](std::string& s) { return chip::run_icachef(s, ptx_dir); });
    sweep("clevel", [&](std::string& s) { return chip::run_clevel(s, ptx_dir, d); });
  }
  sweep("const", [&](std::string& s) { return chip::run_const(s); });
  sweep("tlb", [&](std::string& s) { return chip::run_tlb(s); });
  sweep("l1", [&](std::string& s) { return chip::run_l1(s); });
  sweep("l2", [&](std::string& s) { return chip::run_l2(s, d); });
  j += "\"failed\":[" + failed + "],\"passed\":" + (failed.empty() ? "true" : "false") + "}";
  FILE* f = fopen((out + "/chip.json").c_str(), "w");
  fprintf(f, "%s\n", j.c_str());
  fclose(f);
  printf("%s\n", j.c_str());
  return failed.empty() ? 0 : 3;
}
