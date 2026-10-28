// facet_break's mix kernels: a MMA form and up to two lane ops in one warp's loop (mix), in separate warps (spec), the
// lanes alone, or the MMA alone. `generate.py` instantiates them from a seeded grid and writes each one's probe-table
// entry (the exact SASS counts the gate pins); opcode_probe's harness (probe.cuh) gates, poisons and times them.
//
//   FB_MIX(name, Form, L1, L2, Q, P, tpb)   every warp: 8 MMAs and Q lane ops an iteration (L1 and L2 alternating),
//                                           P = 0 spread evenly after the MMAs, 1 all after the 8 MMAs
//   FB_SPEC(name, Form, L1, L2, WM, tpb)    WM warps: 8 MMAs a trip (OP_LOOP); the rest: 16 lane ops a trip (OP_UNTIL)
//   FB_ALONE(name, L1, L2, tpb)             every warp: 16 lane ops an iteration
//   FB_MMA(name, Form, tpb)                 every warp: 8 MMAs an iteration on 8 accumulators
#pragma once
#include "../opcode_probe/probe.cuh"

namespace fb {

#define FB_A4 "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
#define FB_B2 "r"(b[0]), "r"(b[1])

// Each form: its accumulator, the operand words (loop-invariant, nonzero), and the asm (tensor_kinds.cu's forms).
struct I16832 {  // IMMA.16832.S8.S8: the form the generic MMA census lines were measured with
  using Acc = int32_t;
  static constexpr uint32_t W = 0x01020304u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : FB_A4, FB_B2);
  }
};
struct I16816 {  // IMMA.16816.S8.S8
  using Acc = int32_t;
  static constexpr uint32_t W = 0x01020304u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
  }
};
struct H16816 {  // HMMA.16816.F32
  using Acc = float;
  static constexpr uint32_t W = 0x14001400u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : FB_A4, FB_B2);
  }
};
struct H16816F16 {  // HMMA.16816.F16: two accumulator registers
  using Acc = uint32_t;
  static constexpr uint32_t W = 0x14001400u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : FB_A4, FB_B2);
  }
};
struct H1684T {  // HMMA.1684.F32.TF32: two A registers, one B
  using Acc = float;
  static constexpr uint32_t W = 0x3a800000u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
  }
};
struct H1688T {  // HMMA.1688.F32.TF32
  using Acc = float;
  static constexpr uint32_t W = 0x3a800000u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : FB_A4, FB_B2);
  }
};
// The sparse forms take four B registers (b0, b1, a1, a3: loop-invariant like the rest) and metadata 0x44444444, which
// selects positions (0, 1) of every group of four (tensor_kinds.cu's init_ops).
#define FB_B4 "r"(b[0]), "r"(b[1]), "r"(a[1]), "r"(a[3])
struct ISP16832 {  // IMMA.SP.16832: 2:4 sparse s8, two A registers
  using Acc = int32_t;
  static constexpr uint32_t W = 0x01020304u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, "
                 "{%0,%1,%2,%3}, %8, 0x0;"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), FB_B2, "r"(0x44444444u));
  }
};
struct ISP16864 {  // IMMA.SP.16864: 2:4 sparse s8, a dense k32 product
  using Acc = int32_t;
  static constexpr uint32_t W = 0x01020304u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                 "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : FB_A4, FB_B4, "r"(0x44444444u));
  }
};
struct HSP16832 {  // HMMA.SP.16832.F32: 2:4 sparse f16
  using Acc = float;
  static constexpr uint32_t W = 0x14001400u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                 "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : FB_A4, FB_B4, "r"(0x44444444u));
  }
};
struct QSP16864 {  // QMMA.SP.16864.F32.E4M3.E4M3: 2:4 sparse FP8
  using Acc = float;
  static constexpr uint32_t W = 0x20202020u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : FB_A4, FB_B4, "r"(0x44444444u));
  }
};
struct Q16832 {  // QMMA.16832.F32.E4M3.E4M3
  using Acc = float;
  static constexpr uint32_t W = 0x20202020u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : FB_A4, FB_B2);
  }
};
struct QSF16832 {  // QMMA.SF.16832: block-scaled MXF8 (ue8m0 scale 1.0)
  using Acc = float;
  static constexpr uint32_t W = 0x20202020u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : FB_A4, FB_B2, "r"(0x7f7f7f7fu), "r"(0x7f7f7f7fu));
  }
};
struct OSF16864 {  // OMMA.SF.16864: block-scaled NVFP4 (ue4m3 scale 1.0)
  using Acc = float;
  static constexpr uint32_t W = 0x22222222u;
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : FB_A4, FB_B2, "r"(0x38383838u), "r"(0x38383838u));
  }
};

template <class F>
struct Mma {
  uint32_t a[4], b[2];
  typename F::Acc d[8][4];
  __device__ explicit Mma(uint32_t lane) {
    a[0] = F::W ^ (lane << 8); a[1] = F::W; a[2] = F::W ^ 0x00010000u; a[3] = F::W;
    b[0] = F::W; b[1] = F::W ^ (lane << 16);
#pragma unroll
    for (int c = 0; c < 8; c++)
#pragma unroll
      for (int j = 0; j < 4; j++) d[c][j] = 0;
  }
  __device__ __forceinline__ void go(int c) { F::mma(d[c], a, b); }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      typename F::Acc v = d[c][0];
      s ^= *reinterpret_cast<const uint32_t*>(&v);
    }
    return s;
  }
};

// The lane ops. Each acts on chain s of its own register file; the operands derive from a.mode (0 at run time), so ptxas
// folds no immediate form except where the op names one (the *I lanes).
// The register-light lanes (ji.cuh's search) read one register: FMULI, FADDI, SHFI (a rotate), PRMTI (a byte rotate),
// IMADWI and DP4AI by an immediate; AXI adds and xors an immediate in turn on each chain (one alone would fold into one
// op); UAXI adds the next chain and xors an immediate in turn on chains derived from a.mode only, which ptxas 13.0.88
// puts on the uniform datapath beside a MMA. SHFL is a butterfly by 1 (the MIO crossbar's lane class).
enum Lane {
  L_NONE, L_IADD3, L_LOP3, L_SHF, L_PRMT, L_IMAD, L_IMADI, L_IMADW, L_IADD64, L_SEL, L_FFMA, L_FFMAI,
  L_HFMA2, L_MNMX, L_DMUL, L_DFMA, L_MUFU, L_POPC, L_FMUL, L_FADD,
  L_FMULI, L_FADDI, L_AXI, L_SHFI, L_PRMTI, L_IMADWI, L_DP4A, L_DP4AI, L_UAXI, L_SHFL
};

struct Lanes {
  uint32_t v[8], u[8], k1, k2;
  float f[8], kf1, kf2;
  uint64_t w[8], kw;
  double x[8], kd1, kd2;
  bool p;
  __device__ explicit Lanes(const OpArgs& a) {
    k1 = 0x3C013C01u ^ (uint32_t)a.mode;
    k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
    kf1 = 1.0000001f + (float)a.mode;
    kf2 = 1e-7f * (float)(a.mode + 1);
    kw = 0x0000000100000003ull ^ (uint64_t)a.mode;
    kd1 = 1.0000000001 + (double)a.mode;
    kd2 = 1e-12 * (double)(a.mode + 1);
    p = (a.mode & 1) != 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      v[i] = threadIdx.x * 8u + i + 1u;
      u[i] = (uint32_t)a.mode * 0x9E3779B9u + 2u * i + 1u;
      f[i] = 1.0f + 1e-3f * (float)(threadIdx.x * 8u + i);
      w[i] = ((uint64_t)threadIdx.x << 32) | (8u * threadIdx.x + i + 1u);
      x[i] = 1.0 + 1e-6 * (double)(threadIdx.x * 8u + i);
    }
  }
  template <int L>
  __device__ __forceinline__ void op(int q) {
    const int s = q & 7;
    // a chain's own operand twice in a row would merge into one 3-input IADD3: each add reads the next chain
    if constexpr (L == L_IADD3) asm volatile("add.u32 %0, %0, %1;" : "+r"(v[s]) : "r"(v[(s + 1) & 7]));
    if constexpr (L == L_LOP3) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (L == L_SHF) asm volatile("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(v[(s + 1) & 7]), "r"(k2));
    if constexpr (L == L_PRMT) asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (L == L_IMAD) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (L == L_IMADI) asm volatile("mad.lo.u32 %0, %0, 0x10035, %1;" : "+r"(v[s]) : "r"(k2));
    if constexpr (L == L_IMADW) {
      uint32_t lo = (uint32_t)w[s];
      asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(w[s]) : "r"(lo), "r"(k1));
    }
    if constexpr (L == L_IADD64) asm volatile("add.u64 %0, %0, %1;" : "+l"(w[s]) : "l"(kw));
    if constexpr (L == L_SEL) {
      uint32_t o = v[(s + 1) & 7];
      asm volatile("{ .reg .pred q; setp.ne.u32 q, %2, 0; selp.b32 %0, %0, %1, q; }" : "+r"(v[s]) : "r"(o), "r"((uint32_t)p));
    }
    if constexpr (L == L_FFMA) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[s]) : "f"(kf1), "f"(kf2));
    if constexpr (L == L_FFMAI) asm volatile("fma.rn.f32 %0, %0, 0f3F800010, %1;" : "+f"(f[s]) : "f"(kf2));
    if constexpr (L == L_FMUL) asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(f[s]) : "f"(kf1));
    if constexpr (L == L_FADD) asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(f[s]) : "f"(kf2));
    if constexpr (L == L_HFMA2) asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (L == L_MNMX) asm volatile("min.u32 %0, %0, %1;" : "+r"(v[s]) : "r"(v[(s + 3) & 7]));
    if constexpr (L == L_DMUL) asm volatile("mul.rn.f64 %0, %0, %1;" : "+d"(x[s]) : "d"(kd1));
    if constexpr (L == L_DFMA) asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(x[s]) : "d"(kd1), "d"(kd2));
    if constexpr (L == L_MUFU) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[s]));
    if constexpr (L == L_POPC) asm volatile("popc.b32 %0, %0;" : "+r"(v[s]));
    if constexpr (L == L_FMULI) asm volatile("mul.rn.f32 %0, %0, 0f3F800010;" : "+f"(f[s]));
    if constexpr (L == L_FADDI) asm volatile("add.rn.f32 %0, %0, 0f33800000;" : "+f"(f[s]));
    if constexpr (L == L_AXI) {
      if ((q >> 3) & 1) asm volatile("xor.b32 %0, %0, 0x5A5A5A5A;" : "+r"(v[s]));
      else asm volatile("add.u32 %0, %0, 0x9E3779B9;" : "+r"(v[s]));
    }
    if constexpr (L == L_SHFI) asm volatile("shf.l.wrap.b32 %0, %0, %0, 13;" : "+r"(v[s]));
    if constexpr (L == L_PRMTI) asm volatile("prmt.b32 %0, %0, %0, 0x0321;" : "+r"(v[s]));
    if constexpr (L == L_IMADWI) {
      uint32_t lo = (uint32_t)w[s];
      asm volatile("mad.wide.u32 %0, %1, 0x10035, %0;" : "+l"(w[s]) : "r"(lo));
    }
    if constexpr (L == L_DP4A) asm volatile("dp4a.u32.u32 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(v[(s + 1) & 7]));
    if constexpr (L == L_DP4AI) asm volatile("dp4a.u32.u32 %0, %0, 0x01020304, %0;" : "+r"(v[s]));
    if constexpr (L == L_UAXI) {
      if ((q >> 3) & 1) asm volatile("xor.b32 %0, %0, 0x5A5A5A5A;" : "+r"(u[s]));
      else asm volatile("add.u32 %0, %0, %1;" : "+r"(u[s]) : "r"(u[(s + 1) & 7]));
    }
    if constexpr (L == L_SHFL) asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 31, 0xffffffff;" : "+r"(v[s]));
  }
  // q-th op of a stream that alternates L1 and L2 (L2 == L_NONE: L1 only); each lane takes its own chains in turn
  template <int L1, int L2>
  __device__ __forceinline__ void op2(int q) {
    if constexpr (L2 == L_NONE) op<L1>(q);
    else if (q & 1) op<L2>(q >> 1);
    else op<L1>(q >> 1);
  }
  template <int L>
  __host__ __device__ static constexpr int file() {  // 0: v, 1: f, 2: w, 3: x, 4: u
    if constexpr (L == L_FFMA || L == L_FFMAI || L == L_MUFU || L == L_FMUL || L == L_FADD || L == L_FMULI ||
                  L == L_FADDI) return 1;
    if constexpr (L == L_IMADW || L == L_IADD64 || L == L_IMADWI) return 2;
    if constexpr (L == L_DMUL || L == L_DFMA) return 3;
    if constexpr (L == L_UAXI) return 4;
    return 0;
  }
  // q-th op of a stream that takes L1, L2, L3 in turn (the L_NONE ones skipped); each lane takes its own chains in turn
  template <int L1, int L2, int L3>
  __device__ __forceinline__ void op3(int q) {
    constexpr int n = 1 + (L2 != L_NONE) + (L3 != L_NONE);
    const int j = q % n, k = q / n;
    if (j == 0) op<L1>(k);
    else if (j == 1) op<L2 != L_NONE ? L2 : L3>(k);
    else op<L3>(k);
  }
  template <int L1, int L2, int L3>
  __device__ uint32_t fold3() const {
    constexpr int f1 = file<L1>(), f2 = L2 != L_NONE ? file<L2>() : f1, f3 = L3 != L_NONE ? file<L3>() : f1;
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (f1 == 0 || f2 == 0 || f3 == 0) s ^= v[i];
      if constexpr (f1 == 1 || f2 == 1 || f3 == 1) s ^= __float_as_uint(f[i]);
      if constexpr (f1 == 2 || f2 == 2 || f3 == 2) s ^= (uint32_t)w[i] ^ (uint32_t)(w[i] >> 32);
      if constexpr (f1 == 3 || f2 == 3 || f3 == 3) {
        const unsigned long long t = __double_as_longlong(x[i]);
        s ^= (uint32_t)t ^ (uint32_t)(t >> 32);
      }
      if constexpr (f1 == 4 || f2 == 4 || f3 == 4) s ^= u[i];
    }
    return s;
  }
  template <int L1, int L2>
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (file<L1>() == 0 || (L2 != L_NONE && file<L2>() == 0)) s ^= v[i];
      if constexpr (file<L1>() == 1 || (L2 != L_NONE && file<L2>() == 1)) s ^= __float_as_uint(f[i]);
      if constexpr (file<L1>() == 2 || (L2 != L_NONE && file<L2>() == 2)) s ^= (uint32_t)w[i] ^ (uint32_t)(w[i] >> 32);
      if constexpr (file<L1>() == 3 || (L2 != L_NONE && file<L2>() == 3)) {
        const unsigned long long u = __double_as_longlong(x[i]);
        s ^= (uint32_t)u ^ (uint32_t)(u >> 32);
      }
    }
    return s;
  }
};

template <class F, int L1, int L2, int Q, int P>
__device__ __forceinline__ void mix(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  Mma<F> m(lane);
  Lanes o(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      m.go(c);
      if constexpr (P == 0) {
#pragma unroll
        for (int q = c * Q / 8; q < (c + 1) * Q / 8; q++) o.template op2<L1, L2>(q);
      }
    }
    if constexpr (P == 1) {
#pragma unroll
      for (int q = 0; q < Q; q++) o.template op2<L1, L2>(q);
    }
  }
  OP_END
  OP_SINK((lane ^ m.fold() ^ o.template fold<L1, L2>()))
}

template <class F, int L1, int L2, int WM>
__device__ __forceinline__ void spec(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  Mma<F> m(lane);
  Lanes o(a);
  OP_BEGIN_STOP
  if ((int)(threadIdx.x >> 5) < WM) {
    OP_LOOP(it) {
#pragma unroll
      for (int c = 0; c < 8; c++) m.go(c);
    }
    OP_DONE
  } else {
    OP_UNTIL(n) {
#pragma unroll
      for (int q = 0; q < 16; q++) o.template op2<L1, L2>(q);
    }
    OP_TRIPS(n)
  }
  OP_END_STOP
  OP_SINK((lane ^ m.fold() ^ o.template fold<L1, L2>()))
}

template <int L1, int L2>
__device__ __forceinline__ void alone(const OpArgs& a) {
  Lanes o(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < 16; q++) o.template op2<L1, L2>(q);
  }
  OP_END
  OP_SINK((o.template fold<L1, L2>()))
}

template <class F>
__device__ __forceinline__ void mma_alone(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  Mma<F> m(lane);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) m.go(c);
  }
  OP_END
  OP_SINK(lane ^ m.fold())
}

}  // namespace fb

#define FB_MIX(name, F, L1, L2, Q, P, TPB) OP_KERNEL(name, TPB) { fb::mix<fb::F, fb::L1, fb::L2, Q, P>(a); }
#define FB_SPEC(name, F, L1, L2, WM, TPB) OP_KERNEL(name, TPB) { fb::spec<fb::F, fb::L1, fb::L2, WM>(a); }
#define FB_ALONE(name, L1, L2, TPB) OP_KERNEL(name, TPB) { fb::alone<fb::L1, fb::L2>(a); }
#define FB_MMA(name, F, TPB) OP_KERNEL(name, TPB) { fb::mma_alone<fb::F>(a); }
