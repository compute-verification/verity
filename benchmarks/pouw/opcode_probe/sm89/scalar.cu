// sm_89 (L40S) census probes on probe.cuh: every scalar pipe's rate
// alone, the shared-memory loads, IMMA beside every scalar class, and the honest step (8 IMMA then the duo fold of
// their words) at m16n8k32 and m16n8k16. stmatrix (lm.cu's STSM) is sm_90+, so the shared-memory probes are LDS, STS
// and LDSM only. frontier.cu's mixes don't unroll on sm_89, so this file carries its own. The uniform datapath is
// uniform.cu's: ptxas 13.0 keeps a warp-uniform chain there only in a loop with MMAs, and on lanes in these loops,
// whatever its consumer (a shared address, a branch, a SHFL lane). Built for sm_89 by run.sh.
//   alone_<x>      16 warps, 16 ops of x an iteration on 8 chains: x's own rate (a census line)
//   mix_<x>_<Q>    16 warps, every warp 8 IMMA m16n8k32 and Q ops of x an iteration (a joint point)
//   spec_<x>_w<W>  8 IMMA warps and W x warps that run only while they do (a joint point)
//   ldsm_*, lds32, lds128, sts128: one 1024-thread block an SM, 8 accesses an iteration, conflict-free
//   honest32/16    8 IMMA (m16n8k32 / m16n8k16) and the duo fold of their 32 words a lane; imma16_ref the MMAs alone;
//                  _w2/_w6/_w8 the same at 2, 6 and 8 warps a SMSP (honest32/16 run 4)
#include "../probe.cuh"
#include "../probes/honest.cuh"

constexpr int TPB = 512;

__device__ __forceinline__ void mma_k16(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}

// every operand through mode (0 at run time): ptxas rematerializes a constant operand with IMAD.MOV inside the timed
// loop, and the first MMA of each trip waits on it
__device__ __forceinline__ void init_mma(uint32_t lane, int mode, uint32_t* a, uint32_t* b) {
  const uint32_t m = (uint32_t)mode;
  a[0] = 0x08080808u ^ lane ^ m; a[1] = 0x01010101u ^ m; a[2] = 0x08010801u ^ m; a[3] = 0x01080108u ^ m;
  b[0] = 0x08080808u ^ m; b[1] = 0x01010101u ^ lane ^ m;
}

enum { X_IADD3, X_LOP3, X_SHF, X_PRMT, X_IMAD, X_IDP4A, X_FFMA, X_HFMA2, X_IMNMX, X_SHFL, X_POPC, X_FLO };

struct Ops {
  uint32_t v[8], k1, k2;
  float c1, c2;
  __device__ explicit Ops(const OpArgs& a) {
    // a.mode is 0 at run time; through it the operands stay registers, so ptxas folds no immediate form
    k1 = 0x3C013C01u ^ (uint32_t)a.mode;
    k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
    c1 = 0.99999994f + (float)a.mode;
    c2 = 1e-7f * (float)(a.mode + 1);
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = threadIdx.x * 8u + i + 1u;
  }
  template <int X>
  __device__ __forceinline__ void op(int s) {
    // two adds into one register: ptxas writes a three-input IADD3, which has no IMAD.IADD form
    if constexpr (X == X_IADD3) asm volatile("add.u32 %0, %0, %1;\n\tadd.u32 %0, %0, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_LOP3) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_SHF) asm volatile("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_PRMT) asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_IMAD) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_IDP4A) asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_FFMA)
      asm volatile("{\n\t.reg .f32 t;\n\tmov.b32 t, %0;\n\tfma.rn.f32 t, t, %1, %2;\n\tmov.b32 %0, t;\n\t}"
                   : "+r"(v[s]) : "f"(c1), "f"(c2));
    if constexpr (X == X_HFMA2) asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (X == X_IMNMX) asm volatile("max.s32 %0, %0, %1;" : "+r"(v[s]) : "r"(k1));
    if constexpr (X == X_SHFL) asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(v[s]));
    if constexpr (X == X_POPC) asm volatile("popc.b32 %0, %0;" : "+r"(v[s]));
    if constexpr (X == X_FLO) asm volatile("bfind.u32 %0, %0;" : "+r"(v[s]));
  }
  __device__ uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x ^= v[i];
    return x;
  }
};

template <int X>
__device__ __forceinline__ void alone(const OpArgs& a) {
  Ops o(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < 16; q++) o.op<X>(q & 7);
  }
  OP_END
  OP_SINK(o.fold())
}

template <int X, int Q>
__device__ __forceinline__ void mix(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
  init_mma(lane, a.mode, x, y);
  int32_t d[8][4] = {};
  Ops o(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      mma_imma(d[c], x, y);
      if constexpr (Q >= 8) {
#pragma unroll
        for (int j = 0; j < Q / 8; j++) o.op<X>((c * (Q / 8) + j) & 7);
      } else if (c % (8 / Q) == 8 / Q - 1) {
        o.op<X>(c / (8 / Q));
      }
    }
  }
  OP_END
  uint32_t s = lane ^ o.fold();
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)d[c][0];
  OP_SINK(s)
}

template <int X>
__device__ __forceinline__ void spec(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
  init_mma(lane, a.mode, x, y);
  int32_t d[8][4] = {};
  Ops o(a);
  OP_BEGIN_STOP
  if ((threadIdx.x >> 5) < 8) {
    OP_LOOP(it) {
#pragma unroll
      for (int c = 0; c < 8; c++) mma_imma(d[c], x, y);
    }
    OP_DONE
  } else {
    OP_UNTIL(n) {
#pragma unroll
      for (int q = 0; q < 16; q++) o.op<X>(q & 7);
    }
    OP_TRIPS(n)
  }
  OP_END_STOP
  uint32_t s = lane ^ o.fold();
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)d[c][0];
  OP_SINK(s)
}

#define ALONE(x, X) OP_KERNEL(alone_##x, TPB) { alone<X>(a); }
#define MIX(x, X, Q) OP_KERNEL(mix_##x##_##Q, TPB) { mix<X, Q>(a); }
#define SPEC(x, X, W) OP_KERNEL(spec_##x##_w##W, (8 + W) * 32) { spec<X>(a); }
#define MIXES(x, X) MIX(x, X, 4) MIX(x, X, 8) MIX(x, X, 16) MIX(x, X, 32) MIX(x, X, 64) MIX(x, X, 128)
ALONE(iadd3, X_IADD3) ALONE(lop3, X_LOP3) ALONE(shf, X_SHF) ALONE(prmt, X_PRMT) ALONE(imad, X_IMAD)
ALONE(idp4a, X_IDP4A) ALONE(ffma, X_FFMA) ALONE(hfma2, X_HFMA2) ALONE(imnmx, X_IMNMX) ALONE(shfl, X_SHFL)
ALONE(popc, X_POPC) ALONE(flo, X_FLO)
MIXES(iadd3, X_IADD3) MIXES(ffma, X_FFMA) MIXES(idp4a, X_IDP4A) MIXES(popc, X_POPC) MIXES(imad, X_IMAD)
MIXES(hfma2, X_HFMA2) MIXES(lop3, X_LOP3) MIXES(shfl, X_SHFL)
SPEC(iadd3, X_IADD3, 4) SPEC(iadd3, X_IADD3, 8) SPEC(ffma, X_FFMA, 4) SPEC(ffma, X_FFMA, 8) SPEC(imad, X_IMAD, 4)
SPEC(imad, X_IMAD, 8) SPEC(lop3, X_LOP3, 4) SPEC(lop3, X_LOP3, 8)

// ---- shared memory (lm.cu's addressing: each access its own 16-B rows, every 8-row phase 128 consecutive bytes) ----
constexpr int STPB = 1024, U = 8, RING = 2048;
constexpr int SMEM_BYTES = (RING + 256) * 16;

__host__ __device__ inline unsigned lm_off(int slot, int lane, int warp, int it) {
  return ((((unsigned)lane + 8u * U * (unsigned)(warp + it)) & (RING - 1)) + 8u * (unsigned)slot) * 16u;
}
// LDS.32: lane l reads word l of its slot's 128-B line, so a warp's 32 words are one 128-B phase
__host__ __device__ inline unsigned w_off(int slot, int lane, int warp, int it) {
  return ((((unsigned)(warp + it) * U + (unsigned)slot) * 32u + (unsigned)lane) * 4u) & (RING * 16u - 1u);
}

template <int R, bool T>
__device__ __forceinline__ unsigned ldsm(unsigned addr) {
  unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  if constexpr (R == 1 && !T) asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r0) : "r"(addr));
  if constexpr (R == 1 && T) asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(r0) : "r"(addr));
  if constexpr (R == 2 && !T)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
  if constexpr (R == 2 && T)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
  if constexpr (R == 4 && !T)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
  if constexpr (R == 4 && T)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
  return r0 ^ r1 ^ r2 ^ r3;
}

#define LM_SETUP                                                                          \
  __shared__ __align__(128) unsigned char smem[SMEM_BYTES];                               \
  for (int i = threadIdx.x; i < SMEM_BYTES / 4; i += blockDim.x)                          \
    reinterpret_cast<unsigned*>(smem)[i] = i * 2654435761u + blockIdx.x + 1;              \
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);                        \
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;                                \
  unsigned acc = threadIdx.x + 1;

#define LDSM_PROBE(name, R, T)                                                            \
  OP_KERNEL(name, STPB) {                                                                 \
    LM_SETUP                                                                              \
    OP_BEGIN                                                                              \
    OP_LOOP(it) {                                                                         \
      _Pragma("unroll") for (int u = 0; u < U; u++) acc ^= ldsm<R, T>(sbase + lm_off(u, lane, w, it)); \
    }                                                                                     \
    OP_END                                                                                \
    OP_SINK(acc)                                                                          \
  }
LDSM_PROBE(ldsm_x1, 1, false)
LDSM_PROBE(ldsm_x2, 2, false)
LDSM_PROBE(ldsm_x4, 4, false)
LDSM_PROBE(ldsm_x1t, 1, true)
LDSM_PROBE(ldsm_x2t, 2, true)
LDSM_PROBE(ldsm_x4t, 4, true)

OP_KERNEL(lds32, STPB) {
  LM_SETUP
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      unsigned r0;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r0) : "r"(sbase + w_off(u, lane, w, it)));
      acc ^= r0;
    }
  }
  OP_END
  OP_SINK(acc)
}

OP_KERNEL(lds128, STPB) {
  LM_SETUP
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      unsigned r0, r1, r2, r3;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sbase + lm_off(u, lane, w, it)));
      acc ^= r0 ^ r1 ^ r2 ^ r3;
    }
  }
  OP_END
  OP_SINK(acc)
}

OP_KERNEL(sts128, STPB) {
  LM_SETUP
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++)
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(sbase + lm_off(u, lane, w, it)), "r"(acc), "r"(acc + 1),
                   "r"(acc + 2), "r"(acc + u));
    acc += 0x9E3779B9u;
  }
  OP_END
  OP_SINK(acc ^ reinterpret_cast<unsigned*>(smem)[threadIdx.x])
}

// ---- the honest step: 8 IMMA, then duo folds every word they wrote (honest.cuh); m16n8k16 is L = 16's honest ----------
template <bool K16, bool FOLD>
__device__ __forceinline__ void honest_body(const OpArgs& a) {
  Honest h(a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    if constexpr (K16) {
#pragma unroll
      for (int c = 0; c < 8; c++) mma_k16(h.d[c], h.x, h.y);
      if constexpr (FOLD) h.step<0, true>();
    } else {
      h.step<8, FOLD>();
    }
  }
  OP_END
  uint32_t s = h.fold();
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)h.d[c][0];
  OP_SINK(s)
}
OP_KERNEL(honest32, TPB) { honest_body<false, true>(a); }
OP_KERNEL(honest16, TPB) { honest_body<true, true>(a); }
#define HONEST_W(w, threads)                                              \
  OP_KERNEL(honest32_w##w, threads) { honest_body<false, true>(a); }      \
  OP_KERNEL(honest16_w##w, threads) { honest_body<true, true>(a); }
HONEST_W(2, 256)
HONEST_W(6, 768)
HONEST_W(8, 1024)
OP_KERNEL(imma32_ref, TPB) { honest_body<false, false>(a); }
OP_KERNEL(imma16_ref, TPB) { honest_body<true, false>(a); }
OP_KERNEL(fold_only, TPB) {
  Honest h(a.mode);
  OP_BEGIN
  OP_LOOP(it) { h.step<0, true>(); }
  OP_END
  OP_SINK(h.fold())
}

#define IMMA "IMMA.16832.S8.S8"
#define P_ALONE(x, OP, KIND, ROW)                                                                                    \
  {.name = "alone_" #x, .kernel = (const void*)alone_##x, .tpb = TPB, .kind = KIND, .row = ROW, .expect = OP,       \
   .words = 32, .iters = 4096}
#define P_MIX(x, OP, Q, N)                                                                                           \
  {.name = "mix_" #x "_" #Q, .kernel = (const void*)mix_##x##_##Q, .tpb = TPB, .expect = IMMA "=8," OP "=" #N,       \
   .iters = 2048}
#define P_SPEC(x, OP, W)                                                                                             \
  {.name = "spec_" #x "_w" #W, .kernel = (const void*)spec_##x##_w##W, .tpb = (8 + W) * 32,                          \
   .expect = IMMA "=8;" OP, .warps = "8;" #W, .iters = 4096, .until = 2}
#define P_MIXES(x, OP)                                                                                               \
  P_MIX(x, OP, 4, 4), P_MIX(x, OP, 8, 8), P_MIX(x, OP, 16, 16), P_MIX(x, OP, 32, 32), P_MIX(x, OP, 64, 64),          \
      P_MIX(x, OP, 128, 128)
#define SMEM_ROW(nm, K, ROW, OPS, W, FN)                                                                             \
  {.name = #nm, .kernel = (const void*)nm, .tpb = STPB, .kind = K, .row = ROW, .expect = OPS "=8", .words = W,      \
   .smem_ops = OPS "*8", .smem = FN}

static const OpProbe PROBES[] = {
    {.name = "imma_ref", .kernel = (const void*)imma32_ref, .tpb = TPB, .expect = IMMA "=8", .iters = 4096,
     .fn = "imma32_ref"},
    P_ALONE(iadd3, "IADD3=17", "ALU", "A"),
    P_ALONE(lop3, "LOP3.LUT=16", "ALU", "A"),
    P_ALONE(shf, "SHF.L.W*=16", "ALU", "A"),
    P_ALONE(prmt, "PRMT=16", "ALU", "A"),
    P_ALONE(imad, "IMAD=16", "IMAD", "I"),
    P_ALONE(idp4a, "IDP.4A*=16", "IDP4A", "I4A"),
    P_ALONE(ffma, "FFMA=16", "FFMA", "F"),
    P_ALONE(hfma2, "HFMA2*=16", "HFMA2", "H"),
    P_ALONE(imnmx, "IMNMX*=16", "VIMNMX", "VMNX"),
    P_ALONE(shfl, "SHFL.BFLY=16", "SHFL", "SHFL"),
    P_ALONE(popc, "POPC=16", "XU", "XU"),
    P_ALONE(flo, "FLO*=16", "XU", "XU"),
    // the loop counter is an IADD3 too
    P_MIX(iadd3, "IADD3", 4, 5), P_MIX(iadd3, "IADD3", 8, 9), P_MIX(iadd3, "IADD3", 16, 17), P_MIX(iadd3, "IADD3", 32, 33),
    P_MIX(iadd3, "IADD3", 64, 65), P_MIX(iadd3, "IADD3", 128, 129),
    P_MIXES(ffma, "FFMA"),
    P_MIXES(idp4a, "IDP.4A*"),
    P_MIXES(popc, "POPC"),
    P_MIXES(imad, "IMAD"),
    P_MIXES(hfma2, "HFMA2*"),
    P_MIXES(lop3, "LOP3.LUT"),
    P_MIXES(shfl, "SHFL.BFLY"),
    P_SPEC(iadd3, "IADD3=17", 4),
    P_SPEC(iadd3, "IADD3=17", 8),
    P_SPEC(ffma, "FFMA=16", 4),
    P_SPEC(ffma, "FFMA=16", 8),
    P_SPEC(imad, "IMAD=16", 4),
    P_SPEC(imad, "IMAD=16", 8),
    P_SPEC(lop3, "LOP3.LUT=16", 4),
    P_SPEC(lop3, "LOP3.LUT=16", 8),
    SMEM_ROW(ldsm_x1, "LDSM_X1", "ROW_LDSMR1", "LDSM.16.M88", 32, lm_off),
    SMEM_ROW(ldsm_x2, "LDSM_X2", "ROW_LDSMR2", "LDSM.16.M88.2", 64, lm_off),
    SMEM_ROW(ldsm_x4, "LDSM_X4", "ROW_LDSMR4", "LDSM.16.M88.4", 128, lm_off),
    SMEM_ROW(ldsm_x1t, "LDSM_X1T", "ROW_LDSMR1", "LDSM.16.MT88", 32, lm_off),
    SMEM_ROW(ldsm_x2t, "LDSM_X2T", "ROW_LDSMR2", "LDSM.16.MT88.2", 64, lm_off),
    SMEM_ROW(ldsm_x4t, "LDSM_X4T", "ROW_LDSMR4", "LDSM.16.MT88.4", 128, lm_off),
    SMEM_ROW(lds32, "LDS32", "LDSi", "LDS", 32, w_off),
    SMEM_ROW(lds128, "LDS128", "R6r", "LDS.128", 128, lm_off),
    SMEM_ROW(sts128, "STS128", "R6w", "STS.128", 128, lm_off),
    {.name = "honest32", .kernel = (const void*)honest32, .tpb = TPB, .expect = IMMA "=8,LOP3.LUT=32,IMAD=32,LEA.HI=32", .iters = 2048},
    {.name = "honest16", .kernel = (const void*)honest16, .tpb = TPB,
     .expect = "IMMA.16816.S8.S8=8,LOP3.LUT=32,IMAD=32,LEA.HI=32", .iters = 2048},
#define P_HONEST_W(w, threads)                                                                                        \
  {.name = "honest32_w" #w, .kernel = (const void*)honest32_w##w, .tpb = threads,                                   \
   .expect = IMMA "=8,LOP3.LUT=32,IMAD=32,LEA.HI=32", .iters = 2048},                                                \
  {.name = "honest16_w" #w, .kernel = (const void*)honest16_w##w, .tpb = threads,                                   \
   .expect = "IMMA.16816.S8.S8=8,LOP3.LUT=32,IMAD=32,LEA.HI=32", .iters = 2048}
    P_HONEST_W(2, 256),
    P_HONEST_W(6, 768),
    P_HONEST_W(8, 1024),
    {.name = "imma16_ref", .kernel = (const void*)imma16_ref, .tpb = TPB, .expect = "IMMA.16816.S8.S8=8", .iters = 4096},
    {.name = "fold_only", .kernel = (const void*)fold_only, .tpb = TPB, .expect = "LOP3.LUT=32,IMAD=32,LEA.HI=32",
     .iters = 2048},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
