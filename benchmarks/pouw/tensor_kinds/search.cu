// A schedule search over the programs that set JI3n_0..2.
//
// M1 timed one program (IMMA m8n8k16 beside IADD3, interleaved, R = 32) 6.91% apart in two builds, and VF3n-16 divides
// the joint rows by 1.07 for it. Here one form (FORM) at one lane density (RR ops a MMA warp-instruction group of 8 MMAs,
// R/8 a MMA) is timed over schedules: the lane op (IADD3, LOP3, IMAD, IADD3 and FFMA alternating), its register layout
// (CH independent chains), the warps a SMSP (2 to 6) and their split between MMA and lane roles, and where the lane ops
// sit beside the MMAs. M1's own two facet programs (k_spec and k_mixu beside IADD3) run first and last as the control.
// tensor_kinds.py search reads every loop's SASS, so a program counts the instructions ptxas emitted.
#define TENSOR_KINDS_NO_MAIN
#include "tensor_kinds.cu"

#ifndef FORM
#define FORM ImmaS8k16
#endif
#ifndef RR
#define RR 64
#endif

enum { L_LOP3 = 3, L_IMAD = 4 };  // beside tensor_kinds.cu's LANE_IADD3 0, LANE_FFMA 1, LANE_MIX 2

template <int CH>
struct LaneN {
  uint32_t r[CH];
  float f[CH];
};
template <int X, int CH>
__device__ __forceinline__ void lop(LaneN<CH>& s, int i) {
  if (X == LANE_MIX) {
    const int j = i >> 1;
    if (i & 1) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(s.f[j % CH]) : "f"(s.f[(j + 1) % CH]));
    else asm volatile("add.u32 %0,%0,%1;" : "+r"(s.r[j % CH]) : "r"(s.r[(j + 1) % CH]));
  } else if (X == L_LOP3) {
    asm volatile("xor.b32 %0,%0,%1;" : "+r"(s.r[i % CH]) : "r"(s.r[(i + 1) % CH]));
  } else if (X == L_IMAD) {
    asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(s.r[i % CH]) : "r"(s.r[(i + 1) % CH]), "r"(s.r[(i + 2) % CH]));
  } else {
    asm volatile("add.u32 %0,%0,%1;" : "+r"(s.r[i % CH]) : "r"(s.r[(i + 1) % CH]));
  }
}
template <int X, int CH>
__device__ __forceinline__ void init_n(LaneN<CH>& s, uint32_t tid) {
#pragma unroll
  for (int i = 0; i < CH; i++) {
    s.r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
    if (X == LANE_MIX) s.f[i] = 1.0f + (float)(s.r[i] & 0xffu) * 0x1p-12f;
  }
}
template <int X, int CH>
__device__ __forceinline__ uint32_t consume_n(const LaneN<CH>& s) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < CH; i++) x += s.r[i] ^ (X == LANE_MIX ? __float_as_uint(s.f[i]) : 0u);
  return x;
}

// Warp-specialized: MW MMA warps and LW lane warps a SMSP (warp w on SMSP w % 4); R lane ops a lane-loop iteration.
template <class F, int R, int X, int CH, int MW, int LW>
__global__ void __launch_bounds__(768) k_spec2(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool mma_role = warp < 4 * MW;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, iters, a, b, e);
  typename F::Acc d[8][4] = {};
  LaneN<CH> r;
  init_n<X, CH>(r, threadIdx.x);
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
      for (int q = 0; q < R; q++) lop<X, CH>(r, q);
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
  if (mma_role) x ^= fold_acc<F>(d);
  else x += consume_n<X, CH>(r);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// Interleaved: W warps a SMSP, each issuing 64 MMAs and 8R lane ops an iteration. POS 0: each MMA then its R/8 lane
// ops (M1's k_mixu); 1: half before and half after each MMA; 2: eight MMAs back to back, then R lane ops.
template <class F, int R, int X, int CH, int W, int POS>
__global__ void __launch_bounds__(768) k_mix2(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, iters, a, b, e);
  typename F::Acc d[8][4] = {};
  LaneN<CH> r;
  init_n<X, CH>(r, threadIdx.x);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      if (POS == 2) {
#pragma unroll
        for (int c = 0; c < 8; c++) F::mma(d[c], a, b, e);
#pragma unroll
        for (int q = 0; q < R; q++) lop<X, CH>(r, q);
      } else {
#pragma unroll
        for (int c = 0; c < 8; c++) {
          constexpr int before = POS == 1 ? R / 16 : 0;
#pragma unroll
          for (int q = 0; q < before; q++) lop<X, CH>(r, c * (R / 8) + q);
          F::mma(d[c], a, b, e);
#pragma unroll
          for (int q = before; q < R / 8; q++) lop<X, CH>(r, c * (R / 8) + q);
        }
      }
    }
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + threadIdx.x] = (lane ^ fold_acc<F>(d)) + consume_n<X, CH>(r);
}

static void emit2(const char* kernel, const char* form, std::initializer_list<int> tpl, int mw, int lw, int w,
                  int iters, const Timing& t) {
  printf("{\"probe\":\"tensor_kinds_search\",\"kernel\":\"%s\",\"form\":\"%s\",\"tpl\":[", kernel, form);
  int n = 0;
  for (int v : tpl) printf(n++ ? ",%d" : "%d", v);
  printf("],\"mma_warps\":%d,\"lane_warps\":%d,\"warps\":%d,\"iters\":%d,\"cyc_med\":%.0f,\"cyc_min\":%.0f,"
         "\"cyc_max\":%.0f,\"distinct_sms\":%d,\"mhz\":%.1f}\n",
         mw, lw, w, iters, t.cyc_med, t.cyc_min, t.cyc_max, t.distinct_sms, t.mhz);
  fflush(stdout);
}

template <class F, int R, int X, int CH, int MW, int LW>
static void spec2() {
  int iters;
  Timing t = timed(k_spec2<F, R, X, CH, MW, LW>, 4 * (MW + LW), &iters);
  emit2("k_spec2", F::NAME, {R, X, CH, MW, LW}, MW, LW, 0, iters, t);
}
template <class F, int R, int X, int CH, int W, int POS>
static void mix2() {
  int iters;
  Timing t = timed(k_mix2<F, R, X, CH, W, POS>, 4 * W, &iters);
  emit2("k_mix2", F::NAME, {R, X, CH, W, POS}, 0, 0, W, iters, t);
}
template <class F, int R, int X, int CH>
static void schedules() {
  spec2<F, R, X, CH, 1, 1>(); spec2<F, R, X, CH, 1, 2>(); spec2<F, R, X, CH, 2, 1>(); spec2<F, R, X, CH, 2, 2>();
  spec2<F, R, X, CH, 2, 4>(); spec2<F, R, X, CH, 4, 2>(); spec2<F, R, X, CH, 3, 3>();
  mix2<F, R, X, CH, 2, 0>(); mix2<F, R, X, CH, 4, 0>(); mix2<F, R, X, CH, 6, 0>();
  mix2<F, R, X, CH, 2, 1>(); mix2<F, R, X, CH, 4, 1>(); mix2<F, R, X, CH, 6, 1>();
  mix2<F, R, X, CH, 2, 2>(); mix2<F, R, X, CH, 4, 2>(); mix2<F, R, X, CH, 6, 2>();
}
template <class F, int R>
static void control() {
  int iters;
  Timing t = timed(k_spec<F, R, LANE_IADD3>, WSM, &iters);
  emit2("k_spec", F::NAME, {R, LANE_IADD3}, WSM / 8, WSM / 8, 0, iters, t);
  t = timed(k_mixu<F, R, LANE_IADD3>, WSM, &iters);
  emit2("k_mixu", F::NAME, {R, LANE_IADD3}, 0, 0, WSM / 4, iters, t);
}

int main() {
  setup();
  control<FORM, RR>();
  schedules<FORM, RR, LANE_IADD3, 8>();
  schedules<FORM, RR, LANE_IADD3, 4>();
  schedules<FORM, RR, LANE_IADD3, 16>();
  schedules<FORM, RR, L_LOP3, 8>();
  schedules<FORM, RR, L_IMAD, 8>();
  schedules<FORM, RR, LANE_MIX, 8>();
  control<FORM, RR>();
  printf("{\"probe\":\"tensor_kinds_search\",\"done\":true,\"num_sms\":%d}\n", g_sms);
  return 0;
}
