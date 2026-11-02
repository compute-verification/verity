// The joint issue cost of every native sm_120 MMA form.
//
// JI3e's MMA coefficient (8.115 SMSP-cycles a MMA beside other vector-pipe instructions) was fitted beside IMMA s8
// m16n8k32 only. The issue form of the tensor theorem is tight only if no other form writes its output slots for less
// issue. Each form here is timed alone and beside IADD3+FFMA at 4, 8, 12 and 16 ops a MMA, in the two schedules of
// ISSUE's round 2: warp-specialized (k_spec: 2 MMA warps + 2 lane warps a SMSP)
// and interleaved (k_mixu: one warp issues both, body unrolled x8). tensor_kinds.py reads the loops' SASS and fits the
// coefficient a in a·#MMA + #other <= cycles for each form.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// TK_SM: the target, 120 (RTX PRO 6000, the default) or 89 (L40S, RTX 4090: its own form list and an alone4 point a
// form). TK_GROUP: build one group of the target's forms (-1: all), so run.sh compiles the groups in parallel.
#ifndef TK_SM
#define TK_SM 120
#endif
#ifndef TK_GROUP
#define TK_GROUP -1
#endif
#define TK_IN(g) (TK_GROUP < 0 || TK_GROUP == (g))

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

struct BlockClk {
  long long cyc;
  int smid;
  int pad;
  long long role[2];  // k_spec: the MMA warps' and the lane warps' own cycles (each role's slowest warp); else 0
};

#define CLK_BEGIN \
  __syncthreads(); \
  long long _t0 = clock64();
#define CLK_END_ROLES(rec, r0, r1)                     \
  __syncthreads();                                     \
  if (threadIdx.x == 0) {                              \
    rec[blockIdx.x].cyc = clock64() - _t0;             \
    int s_;                                            \
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s_));    \
    rec[blockIdx.x].smid = s_;                         \
    rec[blockIdx.x].role[0] = (r0);                    \
    rec[blockIdx.x].role[1] = (r1);                    \
  }
#define CLK_END(rec) CLK_END_ROLES(rec, 0, 0)

struct Timing {
  double ms = 0, cyc_med = 0, cyc_min = 0, cyc_max = 0, mhz = 0, role_med[2] = {0, 0};
  int blocks = 0, distinct_sms = 0;
};

// One warm-up launch, then `reps` launches; the rep with the smallest median block cycles.
template <class F>
Timing run_timed(F launch, int nblocks, BlockClk* d_rec, int reps = 5) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  launch();
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  Timing best;
  best.cyc_med = 1e300;
  std::vector<BlockClk> h(nblocks);
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(e0));
    launch();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaMemcpy(h.data(), d_rec, nblocks * sizeof(BlockClk), cudaMemcpyDeviceToHost));
    std::vector<double> c;
    std::vector<int> s;
    for (auto& b : h) { c.push_back((double)b.cyc); s.push_back(b.smid); }
    std::sort(c.begin(), c.end());
    std::sort(s.begin(), s.end());
    Timing t;
    t.ms = ms;
    t.blocks = nblocks;
    t.distinct_sms = (int)(std::unique(s.begin(), s.end()) - s.begin());
    t.cyc_med = c[c.size() / 2];
    t.cyc_min = c.front();
    t.cyc_max = c.back();
    for (int k = 0; k < 2; k++) {
      std::vector<double> rc;
      for (auto& b : h) rc.push_back((double)b.role[k]);
      std::sort(rc.begin(), rc.end());
      t.role_med[k] = rc[rc.size() / 2];
    }
    t.mhz = ms > 0 ? t.cyc_max / (ms * 1e3) : 0;
    if (t.cyc_med < best.cyc_med) best = t;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return best;
}

// Dynamic shared memory that keeps a second block off an SM (sm_120 and sm_89: 100 KB a SM; sm_100: 228 KB).
#if TK_SM == 100
constexpr int ONE_BLOCK_SMEM = 120 * 1024;
#else
constexpr int ONE_BLOCK_SMEM = 60 * 1024;
#endif

// ---- the forms ----------------------------------------------------------------------------------------------------
// Each form: its accumulator type and register count, its MACs and output slots a warp instruction, and the asm.
// Operand registers a[0..3], b[0..3], e (sparse metadata) are loop-invariant and set by init_ops.
#define A4 "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
#define A2 "r"(a[0]), "r"(a[1])
#define B2 "r"(b[0]), "r"(b[1])
#define B4 "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3])

struct ImmaS8k32 {  // IMMA.16832.S8.S8: the honest MMA and JI3e's reference
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "imma_s8_m16n8k32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B2);
  }
};
struct ImmaU8S8k32 {
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "imma_u8s8_m16n8k32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B2);
  }
};
struct ImmaS8k16 {
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 128;
  static constexpr const char* NAME = "imma_s8_m16n8k16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A2, "r"(b[0]));
  }
};
struct ImmaS8m8k16 {
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 2;
  static constexpr double MACS = 8 * 8 * 16, SLOTS = 64;
  static constexpr const char* NAME = "imma_s8_m8n8k16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(b[0]));
  }
};
struct ImmaSpS8k64 {  // 2:4 sparse, 32 useful MACs an output: a second exact depth-32 form
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;  // useful (nonzero) MACs
  static constexpr const char* NAME = "imma_sp_s8_m16n8k64";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B4, "r"(e));
  }
};
struct HmmaF16k16F32 {
  static constexpr uint32_t W = 0x14001400u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 128;
  static constexpr const char* NAME = "hmma_f16_m16n8k16_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2);
  }
};
struct HmmaF16k16F16 {  // f16 accumulate: 128 outputs in 64 slots
  static constexpr uint32_t W = 0x14001400u;
  using Acc = uint32_t;
  static constexpr int NC = 2;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 64;
  static constexpr const char* NAME = "hmma_f16_m16n8k16_f16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : A4, B2);
  }
};
struct HmmaTf32k8 {
  static constexpr uint32_t W = 0x3a800000u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 8, SLOTS = 128;
  static constexpr const char* NAME = "hmma_tf32_m16n8k8";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2);
  }
};
struct QmmaE4m3k32 {
  static constexpr uint32_t W = 0x20202020u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "qmma_e4m3_m16n8k32_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2);
  }
};
struct OmmaNvf4k64 {
  static constexpr uint32_t W = 0x22222222u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 64, SLOTS = 128;
  static constexpr const char* NAME = "omma_nvf4_m16n8k64";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2, "r"(0x38383838u), "r"(0x38383838u));
  }
};
struct ImmaS8k32Sat {
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "imma_s8_m16n8k32_sat";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B2);
  }
};
struct ImmaSpS8k32 {  // 2:4 sparse, 16 useful MACs an output
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 128;
  static constexpr const char* NAME = "imma_sp_s8_m16n8k32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, "
        "{%0,%1,%2,%3}, %8, 0x0;"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A2, B2, "r"(e));
  }
};
struct HmmaBf16k16F32 {
  static constexpr uint32_t W = 0x3e003e00u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 128;
  static constexpr const char* NAME = "hmma_bf16_m16n8k16_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2);
  }
};
struct HmmaSpF16k32 {  // 2:4 sparse, 16 useful MACs an output
  static constexpr uint32_t W = 0x14001400u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 128;
  static constexpr const char* NAME = "hmma_sp_f16_m16n8k32_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B4, "r"(e));
  }
};
struct HmmaTf32k4 {
  static constexpr uint32_t W = 0x3a800000u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 4, SLOTS = 128;
  static constexpr const char* NAME = "hmma_tf32_m16n8k4";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A2, "r"(b[0]));
  }
};
struct QmmaE5m2k32 {
  static constexpr uint32_t W = 0x30303030u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "qmma_e5m2_m16n8k32_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2);
  }
};
struct QmmaE2m1k32 {  // FP4 through kind::f8f6f4 (one FP4 value an 8-bit container)
  static constexpr uint32_t W = 0x04040404u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "qmma_e2m1_m16n8k32_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2);
  }
};
struct QmmaSfE4m3k32 {  // block-scaled MXF8 (kind::mxf8f6f4, ue8m0 scale 1.0)
  static constexpr uint32_t W = 0x20202020u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "qmma_sf_e4m3_m16n8k32_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2, "r"(0x7f7f7f7fu), "r"(0x7f7f7f7fu));
  }
};
struct QmmaSpE4m3k64 {  // 2:4 sparse FP8, 32 useful MACs an output
  static constexpr uint32_t W = 0x20202020u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "qmma_sp_e4m3_m16n8k64_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B4, "r"(e));
  }
};

// ---- sm_89 (Ada) forms: every native mma.sync shape sm_89 has beyond the ones above (PTX ISA 9.7.16) -----------------
// INT4 and binary are native on sm_80-sm_89 (absent on sm_120); FP8 takes the sm_89 syntax (no kind::), with f16
// accumulate at m16n8k16 and m16n8k32 too; 2:4 sparse covers s4, f16, tf32 and e4m3.
struct ImmaS4k64 {
  static constexpr uint32_t W = 0x12345671u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 64, SLOTS = 128;
  static constexpr const char* NAME = "imma_s4_m16n8k64";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B2);
  }
};
struct ImmaS4k32 {
  static constexpr uint32_t W = 0x12345671u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "imma_s4_m16n8k32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A2, "r"(b[0]));
  }
};
struct ImmaS4m8k32 {
  static constexpr uint32_t W = 0x12345671u;
  using Acc = int32_t;
  static constexpr int NC = 2;
  static constexpr double MACS = 8 * 8 * 32, SLOTS = 64;
  static constexpr const char* NAME = "imma_s4_m8n8k32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(b[0]));
  }
};
struct BmmaAndK256 {  // b1 AND+POPC: 256 one-bit products an output
  static constexpr uint32_t W = 0x5a3c96e1u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 256, SLOTS = 128;
  static constexpr const char* NAME = "bmma_and_m16n8k256";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B2);
  }
};
struct BmmaXorK256 {
  static constexpr uint32_t W = 0x5a3c96e1u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 256, SLOTS = 128;
  static constexpr const char* NAME = "bmma_xor_m16n8k256";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B2);
  }
};
struct BmmaAndK128 {
  static constexpr uint32_t W = 0x5a3c96e1u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 128, SLOTS = 128;
  static constexpr const char* NAME = "bmma_and_m16n8k128";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A2, "r"(b[0]));
  }
};
struct BmmaAndM8k128 {
  static constexpr uint32_t W = 0x5a3c96e1u;
  using Acc = int32_t;
  static constexpr int NC = 2;
  static constexpr double MACS = 8 * 8 * 128, SLOTS = 64;
  static constexpr const char* NAME = "bmma_and_m8n8k128";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(b[0]));
  }
};
struct HmmaF16k8F32 {
  static constexpr uint32_t W = 0x14001400u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 8, SLOTS = 128;
  static constexpr const char* NAME = "hmma_f16_m16n8k8_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A2, "r"(b[0]));
  }
};
struct HmmaF16k8F16 {
  static constexpr uint32_t W = 0x14001400u;
  using Acc = uint32_t;
  static constexpr int NC = 2;
  static constexpr double MACS = 16 * 8 * 8, SLOTS = 64;
  static constexpr const char* NAME = "hmma_f16_m16n8k8_f16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : A2, "r"(b[0]));
  }
};
struct HmmaBf16k8F32 {
  static constexpr uint32_t W = 0x3e003e00u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 8, SLOTS = 128;
  static constexpr const char* NAME = "hmma_bf16_m16n8k8_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A2, "r"(b[0]));
  }
};
struct QmmaE4m3k16F32 {
  static constexpr uint32_t W = 0x20202020u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 128;
  static constexpr const char* NAME = "qmma_e4m3_m16n8k16_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A2, "r"(b[0]));
  }
};
struct QmmaE4m3k16F16 {  // f16 accumulate: 128 outputs in 64 slots
  static constexpr uint32_t W = 0x08080808u;
  using Acc = uint32_t;
  static constexpr int NC = 2;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 64;
  static constexpr const char* NAME = "qmma_e4m3_m16n8k16_f16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : A2, "r"(b[0]));
  }
};
struct QmmaE4m3k32F16 {
  static constexpr uint32_t W = 0x08080808u;
  using Acc = uint32_t;
  static constexpr int NC = 2;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 64;
  static constexpr const char* NAME = "qmma_e4m3_m16n8k32_f16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : A4, B2);
  }
};
struct ImmaSpS4k64 {  // 2:4 sparse s4, 32 useful MACs an output
  static constexpr uint32_t W = 0x12345671u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "imma_sp_s4_m16n8k64";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, "
        "{%0,%1,%2,%3}, %8, 0x0;"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A2, B2, "r"(e));
  }
};
struct ImmaSpS4k128 {  // 2:4 sparse s4, 64 useful MACs an output
  static constexpr uint32_t W = 0x12345671u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 64, SLOTS = 128;
  static constexpr const char* NAME = "imma_sp_s4_m16n8k128";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B4, "r"(e));
  }
};
struct HmmaSpF16k16 {  // 2:4 sparse f16, 8 useful MACs an output
  static constexpr uint32_t W = 0x14001400u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 8, SLOTS = 128;
  static constexpr const char* NAME = "hmma_sp_f16_m16n8k16_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, "
        "{%0,%1,%2,%3}, %8, 0x0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A2, B2, "r"(e));
  }
};
struct HmmaSpTf32k16 {  // 2:4 (1:2) sparse tf32, 8 useful MACs an output
  static constexpr uint32_t W = 0x3a800000u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 8, SLOTS = 128;
  static constexpr const char* NAME = "hmma_sp_tf32_m16n8k16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B4, "r"(e));
  }
};
struct QmmaSpE4m3k64Ada {  // 2:4 sparse FP8 (sm_89 syntax), 32 useful MACs an output
  static constexpr uint32_t W = 0x20202020u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "qmma_sp_e4m3_m16n8k64_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B4, "r"(e));
  }
};

// Small operands of each form's type, so no accumulator chain of millions of MMAs reaches inf or NaN; the integer
// forms wrap. The sparse metadata 0x4444 4444 selects positions (0, 1) of every group of four (ordered, valid). z is 0
// at run time (iters >= 0) but not to ptxas: an operand it can prove constant, or equal to another, it rematerializes
// with IMAD.MOV inside the timed loop, and the first MMA of each trip waits on those (1/64 of the m16n8 forms' rate on
// sm_89).
template <class F>
__device__ __forceinline__ void init_ops(uint32_t lane, int iters, uint32_t* a, uint32_t* b, uint32_t& e) {
  const uint32_t w = F::W, z = (uint32_t)iters >> 31;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    a[i] = (w ^ (lane & 1u)) + z * (i + 1);
    b[i] = w + z * (i + 5);
  }
  e = 0x44444444u + z;
}

// ---- lane ops -------------------------------------------------------------------------------------------------------
// One pipe alone issues at ~0.5 a SMSP-cycle on sm_120 (IADD3, and the issue ceiling's probes find the same for LOP3,
// IMAD and HFMA2), so a one-pipe lane stream binds on its pipe beside an MMA and the fit overstates a. The mixed points
// use LANE_MIX, IADD3 and FFMA alternating (integer ALU and FMA pipes), which can reach one issue a SMSP-cycle.
enum { LANE_IADD3, LANE_FFMA, LANE_MIX };
struct Lane {
  uint32_t r[8];
  float f[8];
};
__device__ __forceinline__ void iadd(Lane& s, int i) {
  asm volatile("add.u32 %0,%0,%1;" : "+r"(s.r[i & 7]) : "r"(s.r[(i + 1) & 7]));
}
__device__ __forceinline__ void ffma(Lane& s, int i) {
  asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(s.f[i & 7]) : "f"(s.f[(i + 1) & 7]));
}
template <int X>
__device__ __forceinline__ void lane_op(Lane& s, int i) {
  if (X == LANE_IADD3) iadd(s, i);
  else if (X == LANE_FFMA) ffma(s, i);
  else if (i & 1) ffma(s, i >> 1);
  else iadd(s, i >> 1);
}
__device__ __forceinline__ void init_r(Lane& s, uint32_t tid) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    s.r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
    s.f[i] = 1.0f + (float)(s.r[i] & 0xffu) * 0x1p-12f;
  }
}
__device__ __forceinline__ uint32_t consume(const Lane& s) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) x += s.r[i] ^ __float_as_uint(s.f[i]);
  return x;
}

template <class F>
__device__ __forceinline__ uint32_t fold_acc(typename F::Acc (*d)[4]) {
  uint32_t x = 0;
#pragma unroll
  for (int c = 0; c < 8; c++)
#pragma unroll
    for (int j = 0; j < F::NC; j++) {
      typename F::Acc v = d[c][j];
      x ^= *reinterpret_cast<uint32_t*>(&v);
    }
  return x;
}

// A lane op alone at 8 warps a SMSP (ISSUE's saturation geometry): tensor_kinds.py gates on LANE_MIX issuing at ~1
// a SMSP-cycle, so the mixed points sit on the issue facet rather than on a pipe's own rate.
template <int X>
__global__ void __launch_bounds__(1024) k_opalone(int iters, uint32_t* sink, BlockClk* rec) {
  Lane r;
  init_r(r, threadIdx.x);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int q = 0; q < 64; q++) lane_op<X>(r, q);
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + threadIdx.x] = threadIdx.x + consume(r);
}

// MMA alone: 2 MMA warps a SMSP, 8 independent accumulators, the geometry k_spec's MMA warps use.
template <class F>
__global__ void __launch_bounds__(512) k_alone(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, iters, a, b, e);
  typename F::Acc d[8][4] = {};
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) F::mma(d[c], a, b, e);
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + threadIdx.x] = fold_acc<F>(d) ^ lane;
}

// MMA alone at 4 MMA warps a SMSP (32 accumulators in flight a SMSP): a form whose pipe k_alone's 2 warps don't fill
// shows here; its rate is the faster of the two.
template <class F>
__global__ void __launch_bounds__(512) k_alone4(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, iters, a, b, e);
  typename F::Acc d[8][4] = {};
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) F::mma(d[c], a, b, e);
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + threadIdx.x] = fold_acc<F>(d) ^ lane;
}

// Warp-specialized: with 16 warps a SM, SMSP r hosts warps {r, r+4} (MMA) and {r+8, r+12} (lane); R/8 ops a MMA.
// Each warp also records its own cycles, so each role's rate is its work over its own time, not over the block's
// (the faster role's rate over the block time is only a lower bound).
template <class F, int R, int X>
__global__ void __launch_bounds__(512) k_spec(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool mma_role = warp < (blockDim.x >> 6);
  __shared__ long long warp_cyc[16];
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, iters, a, b, e);
  typename F::Acc d[8][4] = {};
  Lane r;
  init_r(r, threadIdx.x);
  CLK_BEGIN
  if (mma_role) {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int c = 0; c < 8; c++) F::mma(d[c], a, b, e);
    }
  } else {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int q = 0; q < R; q++) lane_op<X>(r, q);
    }
  }
  const long long mine = clock64() - _t0;
  if (lane == 0) warp_cyc[warp] = mine;
  __syncthreads();
  long long rm = 0, ro = 0;
  if (threadIdx.x == 0) {
    const int nw = blockDim.x >> 5;
#pragma unroll
    for (int w = 0; w < 16; w++) {
      if (w < nw / 2) rm = warp_cyc[w] > rm ? warp_cyc[w] : rm;
      else if (w < nw) ro = warp_cyc[w] > ro ? warp_cyc[w] : ro;
    }
  }
  CLK_END_ROLES(rec, rm, ro)
  uint32_t x = lane;
  if (mma_role) x ^= fold_acc<F>(d);
  else x += consume(r);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// Interleaved: every warp issues a MMA then R/8 lane ops, 8 accumulators, body unrolled x8 (64 MMAs an iteration).
template <class F, int R, int X>
__global__ void __launch_bounds__(512) k_mixu(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, iters, a, b, e);
  typename F::Acc d[8][4] = {};
  Lane r;
  init_r(r, threadIdx.x);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
#pragma unroll
      for (int c = 0; c < 8; c++) {
        F::mma(d[c], a, b, e);
#pragma unroll
        for (int q = 0; q < R / 8; q++) lane_op<X>(r, c * (R / 8) + q);
      }
    }
  }
  CLK_END(rec)
  uint32_t x = (lane ^ fold_acc<F>(d)) + consume(r);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- host ---------------------------------------------------------------------------------------------------------
static uint32_t* g_sink;
static BlockClk* g_rec;
static int g_sms;
static const int WSM = 16;  // warps a SM, 4 a SMSP

static int size_iters(double cyc_warm, int warm) { return std::max(64, (int)(2e7 / (cyc_warm / warm))); }

template <class K>
static Timing timed(K kern, int warps, int* iters_out) {
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) {
    return run_timed([&] { kern<<<g_sms, warps * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec);
  };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  *iters_out = iters;
  return launch(iters);
}

static void emit(const char* mode, const char* form, double macs, double slots, int R, int mma_warps, int op_warps,
                 int iters, double mmas_sm, double ops_sm, const Timing& t) {
  printf("{\"probe\":\"tensor_kinds\",\"mode\":\"%s\",\"form\":\"%s\",\"macs\":%.0f,\"slots\":%.0f,\"ops_per_mma\":%.3f,"
         "\"mma_warps\":%d,\"op_warps\":%d,\"iters\":%d,\"cyc_med\":%.0f,\"cyc_min\":%.0f,\"cyc_max\":%.0f,"
         "\"distinct_sms\":%d,\"mhz\":%.1f,\"mmas_per_sm\":%.0f,\"ops_per_sm\":%.0f,"
         "\"smsp_cycles_per_mma\":%.4f,\"mma_cyc_med\":%.0f,\"op_cyc_med\":%.0f}\n",
         mode, form, macs, slots, mmas_sm > 0 ? ops_sm / mmas_sm : 0.0, mma_warps, op_warps, iters, t.cyc_med, t.cyc_min,
         t.cyc_max, t.distinct_sms, t.mhz, mmas_sm, ops_sm, mmas_sm > 0 ? 4.0 * t.cyc_med / mmas_sm : 0.0, t.role_med[0],
         t.role_med[1]);
  fflush(stdout);
}

template <int X>
static void opalone(const char* name) {
  int iters;
  Timing t = timed(k_opalone<X>, 32, &iters);
  emit("opalone", name, 0, 0, 64, 0, 32, iters, 0, 32.0 * iters * 64, t);
}
template <class F>
static void alone() {
  int iters;
  const int warps = WSM / 2;
  Timing t = timed(k_alone<F>, warps, &iters);
  emit("alone", F::NAME, F::MACS, F::SLOTS, 0, warps, 0, iters, (double)warps * iters * 8, 0, t);
}
template <class F>
static void alone4() {
  int iters;
  Timing t = timed(k_alone4<F>, WSM, &iters);
  emit("alone4", F::NAME, F::MACS, F::SLOTS, 0, WSM, 0, iters, (double)WSM * iters * 8, 0, t);
}
template <class F, int R, int X>
static void spec() {
  int iters;
  Timing t = timed(k_spec<F, R, X>, WSM, &iters);
  emit(X == LANE_MIX ? "spec" : "spec_iadd3", F::NAME, F::MACS, F::SLOTS, R, WSM / 2, WSM / 2, iters, (double)(WSM / 2) * iters * 8,
       (double)(WSM / 2) * iters * R, t);
}
template <class F, int R, int X>
static void mixu() {
  int iters;
  Timing t = timed(k_mixu<F, R, X>, WSM, &iters);
  emit(X == LANE_MIX ? "mixu" : "mixu_iadd3", F::NAME, F::MACS, F::SLOTS, R, WSM, 0, iters, (double)WSM * iters * 64, (double)WSM * iters * 8 * R, t);
}
template <class F>
static void form() {
  alone<F>();
#if TK_SM != 120
  alone4<F>();
#endif
  spec<F, 32, LANE_MIX>(); spec<F, 64, LANE_MIX>(); spec<F, 96, LANE_MIX>(); spec<F, 128, LANE_MIX>();
  mixu<F, 32, LANE_MIX>(); mixu<F, 64, LANE_MIX>(); mixu<F, 96, LANE_MIX>(); mixu<F, 128, LANE_MIX>();
  // IADD3 alone binds on its pipe; these points enter the fit and the JI3e check, since a row must hold on every program
  spec<F, 32, LANE_IADD3>(); spec<F, 64, LANE_IADD3>(); spec<F, 96, LANE_IADD3>(); spec<F, 128, LANE_IADD3>();
  mixu<F, 32, LANE_IADD3>(); mixu<F, 64, LANE_IADD3>(); mixu<F, 96, LANE_IADD3>(); mixu<F, 128, LANE_IADD3>();
}

static void setup() {
  int d = 0;
  CK(cudaGetDevice(&d));
  CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, d));
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
}

#ifndef TENSOR_KINDS_NO_MAIN
int main() {
  setup();
#if TK_SM == 89
  // Reference first (group 0) and last (group 3), so drift across the run shows as a difference between its passes.
#if TK_IN(0)
  opalone<LANE_IADD3>("iadd3");
  opalone<LANE_FFMA>("ffma");
  opalone<LANE_MIX>("iadd3_ffma");
  form<ImmaS8k32>();
  form<ImmaU8S8k32>();
  form<ImmaS8k32Sat>();
  form<ImmaS8k16>();
  form<ImmaS8m8k16>();
  form<ImmaS4k64>();
  form<ImmaS4k32>();
  form<ImmaS4m8k32>();
#endif
#if TK_IN(1)
  form<BmmaAndK256>();
  form<BmmaXorK256>();
  form<BmmaAndK128>();
  form<BmmaAndM8k128>();
  form<HmmaF16k16F32>();
  form<HmmaF16k16F16>();
  form<HmmaBf16k16F32>();
  form<HmmaF16k8F32>();
#endif
#if TK_IN(2)
  form<HmmaF16k8F16>();
  form<HmmaBf16k8F32>();
  form<HmmaTf32k8>();
  form<HmmaTf32k4>();
  form<QmmaE4m3k32>();
  form<QmmaE5m2k32>();
  form<QmmaE4m3k16F32>();
  form<QmmaE4m3k16F16>();
  form<QmmaE4m3k32F16>();
#endif
#if TK_IN(3)
  form<ImmaSpS8k32>();
  form<ImmaSpS8k64>();
  form<ImmaSpS4k64>();
  form<ImmaSpS4k128>();
  form<HmmaSpF16k16>();
  form<HmmaSpF16k32>();
  form<HmmaSpTf32k16>();
  form<QmmaSpE4m3k64Ada>();
  form<ImmaS8k32>();
#endif
#elif TK_SM == 100
  // B200: the legacy mma.sync forms, premise (M′) of the slot lemma: a shallow shape
  // holds no less of the tensor pipe a slot than its kind's full-depth step. Group 0 is int8 and its 2:4 shapes, the
  // reference first and last; group 1 the other shallow shapes.
#if TK_IN(0)
  opalone<LANE_IADD3>("iadd3");
  opalone<LANE_FFMA>("ffma");
  opalone<LANE_MIX>("iadd3_ffma");
  form<ImmaS8k32>();
  form<ImmaSpS8k32>();
  form<ImmaSpS8k64>();
  form<ImmaS8k16>();
  form<ImmaS8m8k16>();
  form<ImmaS8k32>();
#endif
#if TK_IN(1)
  form<HmmaF16k16F32>();
  form<HmmaF16k8F32>();
  form<HmmaSpF16k16>();
  form<HmmaSpF16k32>();
  form<HmmaTf32k8>();
  form<HmmaTf32k4>();
  form<HmmaSpTf32k16>();
  form<QmmaE4m3k32>();
  form<QmmaE4m3k16F32>();
  form<QmmaSpE4m3k64Ada>();
  form<ImmaS8k32>();
#endif
#else
  opalone<LANE_IADD3>("iadd3");
  opalone<LANE_FFMA>("ffma");
  opalone<LANE_MIX>("iadd3_ffma");
  // Reference first and last, so drift across the run shows as a difference between its two passes.
  form<ImmaS8k32>();
  form<ImmaU8S8k32>();
  form<ImmaS8k16>();
  form<ImmaS8m8k16>();
  form<ImmaSpS8k64>();
  form<HmmaF16k16F32>();
  form<HmmaF16k16F16>();
  form<HmmaTf32k8>();
  form<QmmaE4m3k32>();
  form<OmmaNvf4k64>();
  form<ImmaS8k32Sat>();
  form<ImmaSpS8k32>();
  form<HmmaBf16k16F32>();
  form<HmmaSpF16k32>();
  form<HmmaTf32k4>();
  form<QmmaE5m2k32>();
  form<QmmaE2m1k32>();
  form<QmmaSfE4m3k32>();
  form<QmmaSpE4m3k64>();
  form<ImmaS8k32>();
#endif
  printf("{\"probe\":\"tensor_kinds\",\"done\":true,\"num_sms\":%d,\"sm\":%d,\"group\":%d}\n", g_sms, TK_SM, TK_GROUP);
  return 0;
}
#endif
