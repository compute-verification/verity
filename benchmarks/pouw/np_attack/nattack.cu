// N-ATTACK: the cheapest adversary for SSN's committed words, and the routes that might beat H32's 32 a word, on one
// H200 (sm_90a). A rate probe in MH100 units (one dense int8 MAC = 1; the tensor does 4,096 a SM a clock), measured
// clock-accurately from clock64 so it never reads wall time or depends on DVFS.
//
// SSN commits every k32 int32 accumulator word  w(i,j,t) = D_t(i,j) - beta_j + col_noise(i,j) + row_cumnoise(i,t),
// where D_t is the data's partial dot (fixed before the noise, so precomputable advice), and the two noise terms are
// small integers from the call's live coins. Honest's wgmma writes every such word at 32 priced units (H32, tc 32).
// The adversary is given A, B and advice computed before the noise; it must output every committed word. The question
// T1 asks: can it produce a committed word for under 32 priced units?
//
// Each mode is one route. The word's data part sits in a register (a free move in MH100w: precomputed advice); the
// per-call cost is the route that adds the noise and materialises the distinct 32-bit word.
//   advadd   word = base + noise by one IADD3 (the ALU pipe alone): the naive adversary
//   advmix   the same words split across the IADD3, IMAD and FADD pipes (co-issue): the adversary's best wall-time rate
//   widemul  IMAD.WIDE: two registers an instruction (the lo and hi of one 64-bit product, not two committed words)
//   carry    add.cc + addc: a 64-bit add, the carry out a second register
//   redg     red.global.add.s32: the word summed in the memory path (an L2 atomic); nothing written on the SM
//   move     LDG.E.128: the real price of moving a precomputed word (free in MH100w; this is the hardware number)
//   ldsm     ldmatrix.x4: four registers an instruction, from shared memory (a multi-register load)
//
//   nattack check             every compute mode's sink (or red counter) against a CPU reference; non-zero on a mismatch
//   nattack run REPS ITERS    one JSON line a (mode, ctas-an-SM): clock64 span, words produced, words a SM a clock, and
//                             the MH100 occupancy price a word (4,096 / words-a-SM-a-clock), plus the SM clock from the
//                             event timer. The priced-work (H32) price a word is reported by the analysis, not here.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a nattack.cu -o nattack
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

enum Mode { ADVADD, ADVMIX, WIDEMUL, CARRY, REDG, MOVE, LDSM, N_MODES };
static const char* NAMES[N_MODES] = {"advadd", "advmix", "widemul", "carry", "redg", "move", "ldsm"};

constexpr int WPT = 32;   // words (or loads) a thread a loop trip
constexpr int TPB = 128;  // threads a block (4 warps), as the honest kernel
constexpr u32 MASK = 0x3ffff;

__device__ __forceinline__ long long clk() {
  long long c;
  asm volatile("mov.s64 %0, %%clock64;" : "=l"(c));
  return c;
}
__host__ __device__ static u32 mix32(u32 x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
// each thread's per-trip "coins": the fresh noise the call draws. Deterministic so the CPU reference matches.
__host__ __device__ static u32 noise0(int gid) { return 0x01010101u ^ ((u32)gid * 2654435761u); }

template <int MODE>
__global__ void __launch_bounds__(TPB) probe_k(int iters, const u32* __restrict__ gbase, u32* __restrict__ gsink,
                                               i32* __restrict__ gred, const u32* __restrict__ gmove, long long* span) {
  const int gid = blockIdx.x * TPB + threadIdx.x;
  u32 base[WPT];
#pragma unroll
  for (int k = 0; k < WPT; k++) base[k] = gbase[(gid * WPT + k) & MASK];  // once, outside the loop: free-move advice
  __shared__ u32 sh[TPB * WPT];
  if constexpr (MODE == LDSM) {
    for (int k = threadIdx.x; k < TPB * WPT; k += TPB) sh[k] = gbase[(blockIdx.x * TPB * WPT + k) & MASK];
    __syncthreads();
  }
  u32 sink = 0x9e3779b9u ^ gid;
  u32 noise = noise0(gid);
  const u32 mulv = (u32)iters | 1u;  // a runtime value ptxas can't fold
  const long long c0 = clk();
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
    noise += 0x00010001u;  // the call's fresh coins advance: every trip's words are distinct and live (one op a word)
    u32 acc = sink;
    if constexpr (MODE == ADVADD) {
#pragma unroll
      for (int k = 0; k < WPT; k++) {
        u32 w;
        asm volatile("add.s32 %0, %1, %2;" : "=r"(w) : "r"(base[k]), "r"(noise));
        acc ^= w;
      }
    } else if constexpr (MODE == ADVMIX) {
#pragma unroll
      for (int k = 0; k < WPT; k++) {
        u32 w;
        if (k % 3 == 0) {
          asm volatile("add.s32 %0, %1, %2;" : "=r"(w) : "r"(base[k]), "r"(noise));
        } else if (k % 3 == 1) {
          // IMAD pipe: a runtime multiplier, so ptxas can't strength-reduce it to an add or LEA on the ALU pipe
          asm volatile("mad.lo.s32 %0, %1, %2, %3;" : "=r"(w) : "r"(base[k]), "r"(mulv), "r"(noise));
        } else {
          // FADD pipe. The operands are masked to finite normals in [1,2) so the sum is bit-reproducible on host and
          // device; a real adversary's words above 2^24 are not exact in f32 (a reported caveat), but the rate is the point.
          u32 bb = (base[k] & 0x007fffffu) | 0x3f800000u, nn = (noise & 0x007fffffu) | 0x3f800000u;
          float f;
          asm volatile("add.f32 %0, %1, %2;" : "=f"(f) : "f"(__int_as_float(bb)), "f"(__int_as_float(nn)));
          w = __float_as_int(f);
        }
        acc ^= w;
      }
    } else if constexpr (MODE == WIDEMUL) {
#pragma unroll
      for (int k = 0; k < WPT; k++) {
        u64 d;
        asm volatile("mad.wide.u32 %0, %1, %2, 0;" : "=l"(d) : "r"(base[k]), "r"(noise));
        acc ^= (u32)d ^ (u32)(d >> 32);
      }
    } else if constexpr (MODE == CARRY) {
#pragma unroll
      for (int k = 0; k < WPT; k++) {
        u32 lo, hi;
        asm volatile("add.cc.u32 %0, %2, %3;\n\taddc.u32 %1, %4, 0;" : "=r"(lo), "=r"(hi)
                     : "r"(base[k]), "r"(noise), "r"(base[(k + 1) & (WPT - 1)]));
        acc ^= lo ^ hi;
      }
    } else if constexpr (MODE == REDG) {
#pragma unroll
      for (int k = 0; k < WPT; k++) {
        i32* a = gred + ((gid * WPT + k) & MASK);
        i32 v = (i32)(base[k] + noise);  // the word as the red's summand; the sum lands in L2/HBM
        asm volatile("red.global.add.s32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
      }
      continue;  // no SM result register; nothing to fold
    } else if constexpr (MODE == MOVE) {
#pragma unroll
      for (int k = 0; k < WPT; k += 4) {
        uint4 v = reinterpret_cast<const uint4*>(gmove)[((gid * WPT + k + it) & MASK) >> 2];  // LDG.E.128
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
    } else if constexpr (MODE == LDSM) {
#pragma unroll
      for (int k = 0; k < WPT; k += 4) {
        u32 r0, r1, r2, r3;
        // one 16-byte row a lane (ldmatrix needs each row address 16-byte aligned); 4 u32 a row
        u32* ptr = sh + ((((threadIdx.x & 31) + (k / 4) * 32 + it * 8) * 4) % (TPB * WPT));
        u32 addr = (u32)__cvta_generic_to_shared(ptr);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
        acc ^= r0 ^ r1 ^ r2 ^ r3;
      }
    }
    sink = acc;
  }
  const long long c1 = clk();
  if constexpr (MODE != REDG) gsink[gid] = sink;
  if (gid == 0) *span = c1 - c0;
}

typedef void (*Kern)(int, const u32*, u32*, i32*, const u32*, long long*);
static const Kern KERNS[N_MODES] = {probe_k<ADVADD>, probe_k<ADVMIX>, probe_k<WIDEMUL>, probe_k<CARRY>,
                                    probe_k<REDG>,   probe_k<MOVE>,    probe_k<LDSM>};

// ---- host -----------------------------------------------------------------------------------------------------------
static int sms() {
  int d = 0; CK(cudaGetDevice(&d));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, d));
  return p.multiProcessorCount;
}

// CPU reference for one compute mode's thread-t sink after `iters` trips (block 0, gid == t).
static u32 ref_sink(int m, int t, int iters, const std::vector<u32>& hbase) {
  u32 base[WPT];
  for (int k = 0; k < WPT; k++) base[k] = hbase[((t * WPT + k) & MASK)];
  u32 sink = 0x9e3779b9u ^ (u32)t;
  u32 noise = noise0(t);
  for (int it = 0; it < iters; it++) {
    noise += 0x00010001u;
    u32 acc = sink;
    for (int k = 0; k < WPT; k++) {
      u32 w;
      if (m == ADVADD) {
        w = base[k] + noise;
      } else if (m == ADVMIX) {
        if (k % 3 == 0) w = base[k] + noise;
        else if (k % 3 == 1) w = base[k] * ((u32)iters | 1u) + noise;
        else {
          u32 bb = (base[k] & 0x007fffffu) | 0x3f800000u, nn = (noise & 0x007fffffu) | 0x3f800000u;
          float fb, fn; std::memcpy(&fb, &bb, 4); std::memcpy(&fn, &nn, 4);
          float f = fb + fn; std::memcpy(&w, &f, 4);
        }
      } else if (m == WIDEMUL) {
        u64 d = (u64)base[k] * (u64)(noise); w = (u32)d ^ (u32)(d >> 32);
      } else {  // CARRY
        u64 s = (u64)base[k] + (u64)(noise);
        u32 lo = (u32)s, hi = base[(k + 1) & (WPT - 1)] + (u32)(s >> 32); w = lo ^ hi;
      }
      acc ^= w;
    }
    sink = acc;
  }
  return sink;
}

int main(int argc, char** argv) {
  const size_t N = 1 << 18;  // base / move / red footprint in words (1 MiB): resident in L2 where that matters
  std::vector<u32> hbase(N);
  for (size_t i = 0; i < N; i++) hbase[i] = mix32((u32)i + 1) | 1u;
  u32 *gbase, *gsink, *gmove; i32* gred; long long* gspan;
  const int S = sms();
  CK(cudaMalloc(&gbase, N * 4)); CK(cudaMemcpy(gbase, hbase.data(), N * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&gmove, N * 4)); CK(cudaMemcpy(gmove, hbase.data(), N * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&gred, N * 4));
  CK(cudaMalloc(&gsink, (size_t)S * 3 * TPB * 4));
  CK(cudaMalloc(&gspan, 8));

  if (argc > 1 && std::string(argv[1]) == "check") {
    const int iters = 3;
    int bad = 0;
    for (int m = 0; m < N_MODES; m++) {
      if (m == MOVE || m == LDSM) continue;  // loads: no committed-word semantics to check
      CK(cudaMemset(gred, 0, N * 4));
      CK(cudaMemset(gsink, 0, (size_t)S * 3 * TPB * 4));
      KERNS[m]<<<1, TPB>>>(iters, gbase, gsink, gred, gmove, gspan);
      CK(cudaDeviceSynchronize());
      bool ok = true;
      if (m == REDG) {
        std::vector<i32> hred(N, 0);
        for (int t = 0; t < TPB; t++) {
          u32 noise = noise0(t);
          for (int it = 0; it < iters; it++) {
            noise += 0x00010001u;
            for (int k = 0; k < WPT; k++) hred[((t * WPT + k) & MASK)] += (i32)(hbase[((t * WPT + k) & MASK)] + noise);
          }
        }
        std::vector<i32> dred(N); CK(cudaMemcpy(dred.data(), gred, N * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < N && ok; i++) ok = (dred[i] == hred[i]);
      } else {
        std::vector<u32> dsink(TPB); CK(cudaMemcpy(dsink.data(), gsink, TPB * 4, cudaMemcpyDeviceToHost));
        for (int t = 0; t < TPB && ok; t++) ok = (dsink[t] == ref_sink(m, t, iters, hbase));
      }
      printf("{\"mode\":\"%s\",\"check\":%s}\n", NAMES[m], ok ? "true" : "false");
      if (!ok) bad = 1;
    }
    return bad;
  }

  const int reps = argc > 2 ? atoi(argv[2]) : 7;
  const int iters = argc > 3 ? atoi(argv[3]) : 20000;
  CK(cudaMemset(gred, 0, N * 4));
  for (int m = 0; m < N_MODES; m++) {
    for (int cps = 1; cps <= 3; cps++) {
      const int blocks = S * cps;
      KERNS[m]<<<blocks, TPB>>>(iters, gbase, gsink, gred, gmove, gspan);  // warm up
      CK(cudaDeviceSynchronize());
      long long best = 1LL << 62;
      float best_ms = 0;
      for (int r = 0; r < reps; r++) {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        KERNS[m]<<<blocks, TPB>>>(iters, gbase, gsink, gred, gmove, gspan);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        long long span = 0; CK(cudaMemcpy(&span, gspan, 8, cudaMemcpyDeviceToHost));
        if (span < best) { best = span; best_ms = ms; }
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
      }
      const double words_cta = (double)TPB * WPT * iters;      // one CTA's words over the run
      const double wspc = words_cta * cps / (double)best;      // words a SM a clock64 tick
      const double price = wspc > 0 ? 4096.0 / wspc : 0;        // MH100 occupancy units a word
      const double mhz = best_ms > 0 ? best / (best_ms * 1e3) : 0;
      printf("{\"mode\":\"%s\",\"ctas_per_sm\":%d,\"iters\":%d,\"span_clk\":%lld,\"ms\":%.4f,\"sm_mhz\":%.1f,"
             "\"words_per_sm_clk\":%.4f,\"occ_price_per_word\":%.3f}\n",
             NAMES[m], cps, iters, best, best_ms, mhz, wspc, price);
      fflush(stdout);
    }
  }
  return 0;
}
