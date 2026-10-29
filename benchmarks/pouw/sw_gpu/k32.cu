// Every-k32 commit on Hopper, a prototype: wgmma m64n256k32 s8 with every k32 accumulator word leaving the accumulator,
// against plain wgmma, and the noised operand formed beside it. One H200 (sm_90a).
//
// A rate probe, not a GEMM: each CTA is one warpgroup running STEPS wgmma steps (K = 32 each) into its 64 x 256 s32
// accumulator, on tiles resident in shared memory (dc.cuh's K-major layout), persistent at 1 CTA an SM and at the most
// that fit. Its time is the tensor core's and the export's or the forming's, nothing else (no operand traffic from HBM).
//   plain    wgmma back to back, one group in flight behind the one issued (wait 1)
//   wait0    wait 0 after every step: the serialization an export needs, exporting nothing
//   xor      wait 0, then every word folded into one register (LOP3: the least a commit can touch a word)
//   fold     wait 0, then every word absorbed by duo into 8 chains a thread (dc.cuh, dgemm.cu's dg90 receipts: 3 ops a word)
//   sts      wait 0, then every word stored to shared memory (64 KiB a step a CTA, STS.128)
//   stg      wait 0, then every word stored to global memory (64 KiB a step a CTA, STG.128, a ring of 8 steps a CTA:
//            larger than L2 at every grid run here, so it reaches HBM)
//   forma    plain, with the A tile (the served side, 64 x 32) formed each step in shared memory: X = A + D + 33 a byte from
//            A and D tiles in shared memory (LDS, IADD3, STS), into the buffer wgmma reads next (double-buffered)
//   formb    the same for the B tile (256 x 32), the weight side
//   formab   both
//   formars  A formed in registers and fed to wgmma from them (wgmma .rs: no STS and no wgmma read of A from shared memory)
//
//   k32 check               every mode's accumulators (and sts's and stg's exported words) against the host, 6 steps
//   k32 run REPS STEPS      one JSON line a (mode, CTAs an SM): ms (median of REPS), its SM clock, MACs, MACs an SM-clock,
//                           and the MAC-times it occupies (ms x f x SMs x 4,096: MH100's unit, one dense int8 MAC = 1)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a k32.cu -o k32
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include "../dc/dc.cuh"
#include "../dc/wgmma.cuh"

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

typedef uint32_t u32;
typedef uint64_t u64;

enum Mode { PLAIN, WAIT0, XOR, FOLD, STS, STG, FORMA, FORMB, FORMAB, FORMARS, N_MODES };
static const char* NAMES[N_MODES] = {"plain", "wait0", "xor", "fold", "sts", "stg", "forma", "formb", "formab", "formars"};
__host__ __device__ constexpr bool exports(int m) { return m == WAIT0 || m == XOR || m == FOLD || m == STS || m == STG; }
__host__ __device__ constexpr bool forms_a(int m) { return m == FORMA || m == FORMAB; }
__host__ __device__ constexpr bool forms_b(int m) { return m == FORMB || m == FORMAB; }

constexpr int KB = 32;                     // K bytes a step
constexpr int TA = 64 * KB, TB = 256 * KB;  // tile bytes
constexpr int SA0 = 0, SA1 = TA, SB0 = 2 * TA, SB1 = 2 * TA + TB, RAWA = 2 * TA + 2 * TB, DA = RAWA + TA, RAWB = DA + TA,
              DB = RAWB + TB, STAGE = DB + TB;
constexpr int SMEM = STAGE, SMEM_STS = STAGE + 65536;
constexpr int RING = 8;                    // stg's steps a CTA before its ring wraps
constexpr u32 K33 = 0x21212121u;

struct Ops {
  const uint4 *A, *DA, *Af, *DAf, *B, *DB;  // K-major tiles; Af and DAf in A-fragment order (formars)
};
struct Clk {
  long long c0, c1;
  u64 t0, t1;
};

__device__ __forceinline__ u64 gtime() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint4 add33(uint4 a, uint4 d) {
  return make_uint4(a.x + d.x + K33, a.y + d.y + K33, a.z + d.z + K33, a.w + d.w + K33);
}

// dgemm.cu's absorb90: thread (warp w, lane = 4 g + q) holds D[16 w + g + 8 (e >> 1)][8 ni + 2 q + (e & 1)] in d[4 ni + e]
template <int NI, int END>
__device__ __forceinline__ void absorb(DuoSt (&ch)[8], const int* d) {
  if constexpr (NI < END) {
#pragma unroll
    for (int e = 0; e < 4; e++) duo<(NI / 2) % 16>(ch[4 * (e >> 1) + 2 * (NI % 2) + (e & 1)], (u32)d[4 * NI + e]);
    absorb<NI + 1, END>(ch, d);
  }
}

template <int MODE>
__global__ void __launch_bounds__(128) k32_k(const __grid_constant__ Ops ops, int steps, int* C, uint4* ring,
                                             uint4* stage_out, u32* sink, Clk* clk) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid = threadIdx.x;
  auto put = [&](int off, const uint4* src, int bytes) {
    for (int i = tid; i < bytes / 16; i += 128) reinterpret_cast<uint4*>(sm + off)[i] = src[i];
  };
  put(SA0, ops.A, TA);
  put(SA1, ops.A, TA);
  put(SB0, ops.B, TB);
  put(SB1, ops.B, TB);
  put(RAWA, MODE == FORMARS ? ops.Af : ops.A, TA);
  put(DA, MODE == FORMARS ? ops.DAf : ops.DA, TA);
  put(RAWB, ops.B, TB);
  put(DB, ops.DB, TB);
  fence_async_smem();
  __syncthreads();
  const u64 da[2] = {wgmma_desc(smem_u32(sm + SA0), kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(smem_u32(sm + SA1), kmaj_lbo(), kmaj_sbo(KB))};
  const u64 db[2] = {wgmma_desc(smem_u32(sm + SB0), kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(smem_u32(sm + SB1), kmaj_lbo(), kmaj_sbo(KB))};
  int d[128];
#pragma unroll
  for (int i = 0; i < 128; i++) d[i] = 0;
  u32 x = 0, a0[4], a1[4];
  DuoSt ch[8];
#pragma unroll
  for (int h = 0; h < 8; h++) ch[h] = duo_init(blockIdx.x, tid * 8 + h);
  const uint4* s4 = reinterpret_cast<const uint4*>(sm);

  auto step = [&](auto BUF, int s) {
    constexpr int buf = decltype(BUF)::value;
    if constexpr (forms_a(MODE)) reinterpret_cast<uint4*>(sm + (buf ? SA1 : SA0))[tid] = add33(s4[RAWA / 16 + tid], s4[DA / 16 + tid]);
    if constexpr (forms_b(MODE)) {
#pragma unroll
      for (int j = 0; j < 4; j++)
        reinterpret_cast<uint4*>(sm + (buf ? SB1 : SB0))[tid + 128 * j] = add33(s4[RAWB / 16 + tid + 128 * j], s4[DB / 16 + tid + 128 * j]);
    }
    if constexpr (forms_a(MODE) || forms_b(MODE)) {
      fence_async_smem();
      __syncthreads();
    }
    u32(&a)[4] = buf ? a1 : a0;
    if constexpr (MODE == FORMARS) {
      const uint4 f = add33(s4[RAWA / 16 + tid], s4[DA / 16 + tid]);
      a[0] = f.x; a[1] = f.y; a[2] = f.z; a[3] = f.w;
    }
    wgmma_fence();
    if constexpr (MODE == FORMARS) wgmma_s8_n256_rs(d, a, db[0], 1);
    else wgmma_s8_n256(d, da[forms_a(MODE) ? buf : 0], db[forms_b(MODE) ? buf : 0], 1);
    wgmma_commit();
    if constexpr (!exports(MODE)) {
      wgmma_wait<1>();
    } else {
      wgmma_wait<0>();
      wgmma_hold(d);
      // the fold's results pinned before the next step, so ptxas can't defer them past the next wgmma (which would need a
      // second copy of the accumulator)
      if constexpr (MODE == XOR) {
#pragma unroll
        for (int i = 0; i < 128; i++) x ^= (u32)d[i];
        asm volatile("" : "+r"(x)::"memory");
      }
      if constexpr (MODE == FOLD) {
        absorb<0, 32>(ch, d);
#pragma unroll
        for (int h = 0; h < 8; h++) asm volatile("" : "+r"(ch[h].y), "+r"(ch[h].z)::"memory");
      }
      if constexpr (MODE == STS) {
#pragma unroll
        for (int j = 0; j < 32; j++)
          reinterpret_cast<uint4*>(sm + STAGE)[j * 128 + tid] = make_uint4(d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
      }
      if constexpr (MODE == STG) {
        uint4* r = ring + ((size_t)blockIdx.x * RING + (s & (RING - 1))) * 4096;
#pragma unroll
        for (int j = 0; j < 32; j++) r[j * 128 + tid] = make_uint4(d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
      }
    }
  };

  const long long c0 = clock64();
  const u64 t0 = gtime();
#pragma unroll 1
  for (int s = 0; s < steps; s += 2) {
    step(std::integral_constant<int, 0>{}, s);
    step(std::integral_constant<int, 1>{}, s + 1);
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  const long long c1 = clock64();
  const u64 t1 = gtime();
  if (blockIdx.x == 0 && tid == 0) *clk = Clk{c0, c1, t0, t1};
  const int w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
#pragma unroll
  for (int ni = 0; ni < 32; ni++)
#pragma unroll
    for (int e = 0; e < 4; e++)
      C[((size_t)blockIdx.x * 64 + 16 * w + g + 8 * (e >> 1)) * 256 + 8 * ni + 2 * q + (e & 1)] = d[4 * ni + e];
  u32 f = x;
#pragma unroll
  for (int h = 0; h < 8; h++) f ^= MODE == FOLD ? duo_finish(ch[h]) : 0u;
  sink[blockIdx.x * 128 + tid] = f;
  if constexpr (MODE == STS) {
    __syncthreads();
    if (blockIdx.x == 0)
      for (int i = tid; i < 4096; i += 128) stage_out[i] = reinterpret_cast<const uint4*>(sm + STAGE)[i];
  }
#endif
}

typedef void (*Kern)(Ops, int, int*, uint4*, uint4*, u32*, Clk*);
static const Kern KERNS[N_MODES] = {k32_k<PLAIN>, k32_k<WAIT0>, k32_k<XOR>,   k32_k<FOLD>,   k32_k<STS>,
                                    k32_k<STG>,   k32_k<FORMA>, k32_k<FORMB>, k32_k<FORMAB>, k32_k<FORMARS>};
static int smem_of(int m) { return m == STS ? SMEM_STS : SMEM; }

// ---- host ------------------------------------------------------------------------------------------------------------------
static u32 mix32(u32 x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
static int8_t form(int8_t a, int8_t d) { return (int8_t)(uint8_t)((uint8_t)a + (uint8_t)d + 33); }

struct Host {
  std::vector<int8_t> A, DA, B, DB;  // logical: A[r * 32 + k], B[n * 32 + k]
  std::vector<uint8_t> kA, kDA, fA, fDA, kB, kDB;
};
static Host make_host() {
  Host h;
  h.A.resize(TA); h.DA.resize(TA); h.B.resize(TB); h.DB.resize(TB);
  // operands offset-binary in 1..127 and noise bytes in 0..63, so the 4-entry word add can't carry between bytes
  for (int i = 0; i < TA; i++) { h.A[i] = (int8_t)(1 + mix32(i + 1) % 127); h.DA[i] = (int8_t)(mix32(i + 0x10000) & 63); }
  for (int i = 0; i < TB; i++) { h.B[i] = (int8_t)(1 + mix32(i + 0x20000) % 127); h.DB[i] = (int8_t)(mix32(i + 0x30000) & 63); }
  h.kA.resize(TA); h.kDA.resize(TA); h.fA.resize(TA); h.fDA.resize(TA); h.kB.resize(TB); h.kDB.resize(TB);
  for (int r = 0; r < 64; r++)
    for (int k = 0; k < KB; k++) {
      h.kA[kmaj_off(r, k, KB)] = h.A[r * KB + k];
      h.kDA[kmaj_off(r, k, KB)] = h.DA[r * KB + k];
    }
  for (int n = 0; n < 256; n++)
    for (int k = 0; k < KB; k++) {
      h.kB[kmaj_off(n, k, KB)] = h.B[n * KB + k];
      h.kDB[kmaj_off(n, k, KB)] = h.DB[n * KB + k];
    }
  // wgmma's A fragment for s8 m64k32: thread (warp w, lane 4 g + q), register j holds row 16 w + g + 8 (j & 1), K bytes
  // 4 q + 16 (j >> 1) .. + 3
  for (int t = 0; t < 128; t++)
    for (int j = 0; j < 4; j++)
      for (int b = 0; b < 4; b++) {
        const int w = t >> 5, g = (t & 31) >> 2, q = t & 3, r = 16 * w + g + 8 * (j & 1), k = 4 * q + 16 * (j >> 1) + b;
        h.fA[16 * t + 4 * j + b] = h.A[r * KB + k];
        h.fDA[16 * t + 4 * j + b] = h.DA[r * KB + k];
      }
  return h;
}
struct Dev {
  Ops ops;
  int* C = nullptr;
  uint4 *ring = nullptr, *stage = nullptr;
  u32* sink = nullptr;
  Clk* clk = nullptr;
  int max_ctas = 0;
};
static uint4* up(const std::vector<uint8_t>& v) {
  uint4* p;
  CK(cudaMalloc(&p, v.size()));
  CK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice));
  return p;
}
static Dev make_dev(const Host& h, int max_ctas) {
  Dev d;
  d.ops = Ops{up(h.kA), up(h.kDA), up(h.fA), up(h.fDA), up(h.kB), up(h.kDB)};
  d.max_ctas = max_ctas;
  CK(cudaMalloc(&d.C, (size_t)max_ctas * 64 * 256 * 4));
  CK(cudaMalloc(&d.ring, (size_t)max_ctas * RING * 65536));
  CK(cudaMalloc(&d.stage, 65536));
  CK(cudaMalloc(&d.sink, (size_t)max_ctas * 128 * 4));
  CK(cudaMalloc(&d.clk, sizeof(Clk)));
  return d;
}
static void launch(int m, const Dev& d, int ctas, int steps, cudaStream_t st = 0) {
  KERNS[m]<<<ctas, 128, smem_of(m), st>>>(d.ops, steps, d.C, d.ring, d.stage, d.sink, d.clk);
  CK(cudaGetLastError());
}
static int g_sms = 132;
static int occupancy(int m) {
  CK(cudaFuncSetAttribute(KERNS[m], cudaFuncAttributeMaxDynamicSharedMemorySize, smem_of(m)));
  int n = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, KERNS[m], 128, smem_of(m)));
  return n;
}

static int check() {
  const int S = 6, ctas = 2 * g_sms;
  const Host h = make_host();
  const Dev d = make_dev(h, ctas);
  std::vector<int> C((size_t)ctas * 64 * 256);
  std::vector<uint4> buf(4096);
  int bad_all = 0;
  for (int m = 0; m < N_MODES; m++) {
    occupancy(m);
    std::vector<int> T(64 * 256);
    for (int r = 0; r < 64; r++)
      for (int n = 0; n < 256; n++) {
        int acc = 0;
        for (int k = 0; k < KB; k++) {
          const int8_t a = forms_a(m) || m == FORMARS ? form(h.A[r * KB + k], h.DA[r * KB + k]) : h.A[r * KB + k];
          const int8_t b = forms_b(m) ? form(h.B[n * KB + k], h.DB[n * KB + k]) : h.B[n * KB + k];
          acc += a * b;
        }
        T[r * 256 + n] = S * acc;
      }
    CK(cudaMemset(d.C, 0xA5, C.size() * 4));
    CK(cudaMemset(d.ring, 0xA5, (size_t)ctas * RING * 65536));
    CK(cudaMemset(d.stage, 0xA5, 65536));
    launch(m, d, ctas, S);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(C.data(), d.C, C.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0, bad_x = 0;
    for (size_t i = 0; i < C.size(); i++) bad += C[i] != T[i % (64 * 256)];
    // an exported step's words, in the order each thread stores them: uint4 j * 128 + t holds d[4 j .. 4 j + 3]
    auto words_bad = [&](const uint4* p) {
      size_t b = 0;
      for (int t = 0; t < 128; t++)
        for (int j = 0; j < 32; j++) {
          const int w = t >> 5, g = (t & 31) >> 2, q = t & 3;
          const u32 v[4] = {p[j * 128 + t].x, p[j * 128 + t].y, p[j * 128 + t].z, p[j * 128 + t].w};
          for (int e = 0; e < 4; e++) b += (int)v[e] != T[(16 * w + g + 8 * (e >> 1)) * 256 + 8 * j + 2 * q + (e & 1)];
        }
      return b;
    };
    if (m == STS) {
      CK(cudaMemcpy(buf.data(), d.stage, 65536, cudaMemcpyDeviceToHost));
      bad_x += words_bad(buf.data());
    }
    if (m == STG)
      for (int c = 0; c < ctas; c += 37) {
        CK(cudaMemcpy(buf.data(), d.ring + ((size_t)c * RING + (S - 1) % RING) * 4096, 65536, cudaMemcpyDeviceToHost));
        bad_x += words_bad(buf.data());
      }
    printf("{\"check\": \"%s\", \"ctas\": %d, \"steps\": %d, \"accumulator_mismatches\": %zu, \"exported_mismatches\": %zu}\n",
           NAMES[m], ctas, S, bad, bad_x);
    bad_all += bad || bad_x;
  }
  return bad_all ? 1 : 0;
}

static int run(int reps, int steps) {
  const Host h = make_host();
  int occ_max = 0;
  for (int m = 0; m < N_MODES; m++) occ_max = std::max(occ_max, occupancy(m));
  const Dev d = make_dev(h, occ_max * g_sms);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int m = 0; m < N_MODES; m++) {
    const int occ = occupancy(m);
    for (int o = 1; o <= occ; o = o == occ ? occ + 1 : occ) {
      const int ctas = o * g_sms;
      launch(m, d, ctas, steps);
      CK(cudaDeviceSynchronize());
      std::vector<double> ms, mhz;
      for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(e0));
        launch(m, d, ctas, steps);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float t = 0;
        CK(cudaEventElapsedTime(&t, e0, e1));
        Clk c;
        CK(cudaMemcpy(&c, d.clk, sizeof c, cudaMemcpyDeviceToHost));
        ms.push_back(t);
        mhz.push_back(c.t1 > c.t0 ? (double)(c.c1 - c.c0) / (double)(c.t1 - c.t0) * 1e3 : 0);
      }
      std::sort(ms.begin(), ms.end());
      std::sort(mhz.begin(), mhz.end());
      const double t = ms[reps / 2], f = mhz[reps / 2];
      const double macs = (double)ctas * steps * 64 * 256 * KB, words = (double)ctas * steps * 64 * 256;
      const double units = t * 1e-3 * f * 1e6 * g_sms * 4096.0;
      printf("{\"mode\": \"%s\", \"occ\": %d, \"occ_max\": %d, \"ctas\": %d, \"steps\": %d, \"ms\": %.6f, \"ms_min\": %.6f, "
             "\"ms_max\": %.6f, \"mhz\": %.1f, \"macs\": %.6e, \"mac_per_sm_clk\": %.2f, \"units\": %.6e, \"units_per_mac\": %.4f, "
             "\"words\": %.6e, \"units_per_word\": %.3f, \"smem\": %d}\n",
             NAMES[m], o, occ, ctas, steps, t, ms[0], ms[reps - 1], f, macs, macs / (t * 1e-3 * f * 1e6 * g_sms), units,
             units / macs, words, units / words, smem_of(m));
      fflush(stdout);
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_sms = p.multiProcessorCount;
  if (p.major != 9) {
    fprintf(stderr, "ERR device sm_%d%d: this binary runs on sm_90a\n", p.major, p.minor);
    return 2;
  }
  if (argc == 2 && !strcmp(argv[1], "check")) return check();
  if (argc == 4 && !strcmp(argv[1], "run")) {
    const int steps = atoi(argv[3]);
    if (steps < 2 || steps % 2) {
      fprintf(stderr, "STEPS even\n");
      return 2;
    }
    return run(atoi(argv[2]), steps);
  }
  fprintf(stderr, "usage: k32 check | run REPS STEPS\n");
  return 2;
}
