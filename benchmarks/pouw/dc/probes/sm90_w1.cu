// W1's census probes on the H200 (sm_90a): the register writers CORE-H32's run of record left unprobed, priced per 32-bit
// register written, multi-output forms counted per register. Built, gated and timed as sm90_h32.cu's (emp.sh); notes
// as h32.cuh's (c=CLASS w=PATTERN:REGS), plus p=PATTERN:PREDS for the predicates a counted pattern writes a lane and
// u=PATTERN:WORDS for the uniform registers one writes a warp, which ../h32.py counts by their bits.
//   y_<form>      1024 threads (8 warps a SMSP), 256 of the form an iteration, 8 chains a thread (w1.cuh)
//   l_ldsm<x>     ldmatrix .x1/.x2/.x4 (and .x4.trans) from shared memory, 8 fixed conflict-free row addresses a
//                 thread, each lane's rows in their own 16 B of a 128-B row; the loaded words folded by LOP3
//   l_lds64       ld.shared.v2 (LDS.64), each lane its own 8 B of a 256-B row
//   l_ldc_u, l_ldc64_u  ld.const at a warp-uniform index ptxas can't prove uniform (LDC, LDC.64: a broadcast)
//   l_ldc_d       ld.const at a lane-divergent index (LDC replays per distinct address)
//   l_tld, l_tld4 tex1Dfetch<u32>, <uint4> inside a window the texture cache holds (TLD, 1 and 4 words)
//   l_tex         tex2D<float4> bilinear (TEX, 4 words), inside a 2D window the cache holds
//   l_atoms64     atom.shared.exch.b64 with its return (ATOMS.EXCH.64: 2 registers and 2 memory words a lane)
//   l_lds128b, l_lds64b, l_ldg128b, l_ldgnc128b  ld.shared.v4 and .v2, ld.global.v4 and ld.global.nc.v4 at an address
//                 the warp shares but ptxas can't prove uniform (a broadcast), 8 fixed 16-B cells a thread inside a
//                 window L1 holds; each load folded into its own chain by LOP3, which the note counts too, so the row
//                 prices every register the loop writes
//   m_lds128b_g<G>  CORE-H32's scalar co-issue minimum (DADD, IMAD, DADD, FADD) with a broadcast LDS.128 every G ops,
//                 its words that round's DADD operands; m_dadd2_imad_fadd, the same mix without loads, in this harness
//   y_imadp_on/off  IMAD under a predicate the timed pass holds true / false; y_sel, SEL on it
//   b_ubra, b_dbra  a branch on a coin bit, warp-uniform (the trip's coin word) and lane-divergent (a chain's own bit),
//                 arms of 4 IMAD or 4 LOP3: ptxas if-converts them, so both arms issue under the bit's predicate
//   b_ubra16, b_dbra16  the same with 16-op arms, which ptxas keeps as branches (BRA; BSSY/BSYNC when divergent).
//                 The loop holds both arms and a decision runs one (divergent: both), so h32.py's price is not a
//                 decision's; a decision's time is the loop's over its 16 (64) decisions
//   b_brx4        brx.idx.uni on 2 bits of the trip's coin word to one of 4 arms of 16 ops (BRX, its target table
//                 from LDC): a decision reads 2 coin bits; priced as b_ubra16, from the loop's time over its 16 decisions
//   y_lea, y_iadd64, y_shfli, y_shflu  shl+add (ptxas: LEA and IMAD), add.u64 (IADD3 and IMAD.X), shfl.sync .idx at
//                 a lane index from a register and .up by 1
#include "../../opcode_probe/probe.cuh"
#include "w1.cuh"

constexpr int TPB = 1024, U = 8;

template <int R, int... X>
DC_DEV void w_run(const OpArgs& a) {
  WLane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) { w1_mix<R, X...>(l); }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}

#define Y(NAME, ...) OP_KERNEL(y_##NAME, TPB) { w_run<256, __VA_ARGS__>(a); }
#define Y64(NAME, ...) OP_KERNEL(y_##NAME, TPB) { w_run<64, __VA_ARGS__>(a); }
Y(brev, W_BREV)
Y(bmsk, W_BMSK)
Y(sgxt, W_SGXT)
Y(vabsdiff4, W_VABSDIFF4)
Y(viaddmax, W_VIADDMAX)
Y(viaddmin, W_VIADDMIN)
Y(vimax3, W_VIMAX3)
Y(iabs, W_IABS)
Y(i2ip, W_I2IP)
Y(imadw8, W_IMADW8)
Y(imadws, W_IMADWS)
Y(mulwide, W_MULWIDE)
Y(lop3p, W_LOP3P)
Y(isetp2, W_ISETP2)
Y(plop3, W_PLOP3)
Y64(r2p, W_R2P)
Y(cs2r, W_CS2R)
Y(clk32, W_CLK32)
Y(gtimer, W_LANEID)
Y(i2f64, W_I2F64)
Y(f2i64, W_F2I64)
Y(hadd2f32, W_HADD2F32)
Y(f2fp_h, W_F2FP_H)
Y(movm, W_MOVM)
Y(redux, W_REDUX)
Y(match, W_MATCH)
// mixes: the wide multiply beside the int32 class's other pipe, and beside other classes
Y(mulw_lop3, W_IMADW8, W_LOP3)
Y(mulw_lop3x3, W_IMADW8, W_LOP3, W_LOP3, W_LOP3)
Y(mulw3_lop3, W_IMADW8, W_IMADW8, W_IMADW8, W_LOP3)
Y(mulw_iadd3, W_IMADW8, W_IADD3)
Y(mulw_imad, W_IMADW8, W_IMAD)
Y(mulw_fadd, W_IMADW8, W_FADD)
Y(mulw_dadd, W_IMADW8, W_DADD)
Y(mulw_hadd2, W_IMADW8, W_HADD2)
Y(lop3p_imad, W_LOP3P, W_IMAD)
Y(cs2r_imad, W_CS2R, W_IMAD)
Y(i2ip_imad, W_I2IP, W_IMAD)
Y(viaddmax_imad, W_VIADDMAX, W_IMAD)
Y(vimax3_imad, W_VIMAX3, W_IMAD)
Y(lop3_r, W_LOP3)
Y(imad_r, W_IMAD)
Y(imad_lop3, W_IMAD, W_LOP3)
Y(lea, W_LEA)
Y(iadd64, W_IADD64)
Y(shfli, W_SHFLI)
Y(shflu, W_SHFLU)
#undef Y
#undef Y64

// a predicate the timed pass holds true (ON) or false, which ptxas can't prove
template <int R, int X, bool ON>
DC_DEV void p_run(const OpArgs& a) {
  WLane l(threadIdx.x ^ (uint32_t)a.mode);
  l.pm = ON ? (a.mode == 0) : (a.mode != 0);
  OP_BEGIN
  OP_LOOP(it) { w1_mix<R, X>(l); }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}
template <int R, int X>
DC_DEV void b_run(const OpArgs& a) {
  WLane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    l.cw = (uint32_t)it * 0x9E3779B9u ^ (uint32_t)a.mode;
    w1_mix<R, X>(l);
  }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}
OP_KERNEL(y_imadp_on, TPB) { p_run<256, W_IMADP, true>(a); }
OP_KERNEL(y_imadp_off, TPB) { p_run<256, W_IMADP, false>(a); }
OP_KERNEL(y_sel, TPB) { p_run<256, W_SEL, true>(a); }
OP_KERNEL(b_ubra, TPB) { b_run<64, W_UBRA>(a); }
OP_KERNEL(b_dbra, TPB) { b_run<64, W_DBRA>(a); }
OP_KERNEL(b_ubra16, TPB) { b_run<16, W_UBRA16>(a); }
OP_KERNEL(b_dbra16, TPB) { b_run<16, W_DBRA16>(a); }
OP_KERNEL(b_brx4, TPB) { b_run<16, W_BRX4>(a); }

// ---- loads ----------------------------------------------------------------------------------------------------------
constexpr int ROWS = 64;

// chain u's 16-B row cell for lane l: row (u + 8w) mod ROWS, cell l mod 8, so a phase's 8 rows tile all 32 banks
__host__ __device__ inline unsigned ldsm_off(int u, int lane, int warp, int) {
  return (unsigned)(((u + warp * U) % ROWS) * 128 + (lane & 7) * 16);
}
__host__ __device__ inline unsigned lds64_off(int u, int lane, int warp, int it) {
  return (unsigned)(((u + warp * U) % ROWS) * 256 + lane * 8) ^ (((unsigned)it & 1u) << 8);
}

template <int X, bool T>
DC_DEV void ldsm_body(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned sb[];
  const unsigned base = (unsigned)__cvta_generic_to_shared(sb);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < ROWS * 32; i += TPB) sb[i] = (unsigned)i * 0x9E3779B9u ^ (unsigned)a.mode;
  __syncthreads();
  unsigned ad[U], acc = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) ad[u] = base + ldsm_off(u, lane, w, 0);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      if constexpr (X == 1) {
        unsigned p;
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(p) : "r"(ad[u]) : "memory");
        acc ^= p;
      }
      if constexpr (X == 2) {
        unsigned p, q;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(p), "=r"(q) : "r"(ad[u]) : "memory");
        acc ^= p ^ q;
      }
      if constexpr (X == 4 && !T) {
        unsigned p, q, r, s;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(p), "=r"(q), "=r"(r), "=r"(s) : "r"(ad[u]) : "memory");
        acc ^= p ^ q ^ r ^ s;
      }
      if constexpr (X == 4 && T) {
        unsigned p, q, r, s;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(p), "=r"(q), "=r"(r), "=r"(s) : "r"(ad[u]) : "memory");
        acc ^= p ^ q ^ r ^ s;
      }
    }
  }
  OP_END
  OP_SINK(acc)
}
OP_KERNEL(l_ldsm1, TPB) { ldsm_body<1, false>(a); }
OP_KERNEL(l_ldsm2, TPB) { ldsm_body<2, false>(a); }
OP_KERNEL(l_ldsm4, TPB) { ldsm_body<4, false>(a); }
OP_KERNEL(l_ldsm4t, TPB) { ldsm_body<4, true>(a); }

OP_KERNEL(l_lds64, TPB) {
  extern __shared__ __align__(128) unsigned sb[];
  const unsigned base = (unsigned)__cvta_generic_to_shared(sb);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < ROWS * 64; i += TPB) sb[i] = (unsigned)i * 0x9E3779B9u ^ (unsigned)a.mode;
  __syncthreads();
  unsigned ad[U], acc = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) ad[u] = base + lds64_off(u, lane, w, 0);  // + the trip parity's row flip below
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      unsigned p, q;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(p), "=r"(q) : "r"(ad[u] ^ (((unsigned)it & 1u) << 8)) : "memory");
      acc ^= p ^ q;
    }
  }
  OP_END
  OP_SINK(acc)
}

__constant__ unsigned long long w1_ctab[2048];

// an index the warp shares but ptxas can't prove uniform (a.mode is 0 when timed), or one a lane
template <bool WIDE, bool DIV>
DC_DEV void ldc_body(const OpArgs& a) {
  const int lane = threadIdx.x & 31;
  const unsigned m = (unsigned)a.mode;
  unsigned acc = threadIdx.x, ix[U];
#pragma unroll
  for (int u = 0; u < U; u++) ix[u] = DIV ? (unsigned)(lane * 32 + u * 2) : (((unsigned)u * 64u) + (threadIdx.x & m));
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const unsigned k = (ix[u] ^ ((unsigned)it & 1u)) & 2047u;
      if constexpr (WIDE) {
        unsigned long long p;
        asm volatile("ld.const.u64 %0, [%1];" : "=l"(p) : "l"(w1_ctab + k) : "memory");
        acc ^= (unsigned)p ^ (unsigned)(p >> 32);
      } else {
        unsigned p;
        asm volatile("ld.const.u32 %0, [%1];" : "=r"(p) : "l"((const unsigned*)w1_ctab + k) : "memory");
        acc ^= p;
      }
    }
  }
  OP_END
  OP_SINK(acc)
}
OP_KERNEL(l_ldc_u, TPB) { ldc_body<false, false>(a); }
OP_KERNEL(l_ldc64_u, TPB) { ldc_body<true, false>(a); }
OP_KERNEL(l_ldc_d, TPB) { ldc_body<false, true>(a); }

// side.cu's cache-resident windows: 8 KB of u32 texels, 32 KB of uint4, a 64 x 8 float4 2D patch
template <int K>
DC_DEV void tex_body(const OpArgs& a) {
  const cudaTextureObject_t t1 = a.tex1, t2 = a.tex2, t4 = a.tex4;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  unsigned acc = threadIdx.x;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int i = (int)(((unsigned)(w * U + u) * 32u + (unsigned)lane + ((unsigned)it & 7u) * 64u) & 2047u);
      if constexpr (K == 1) acc ^= tex1Dfetch<unsigned>(t1, i);
      if constexpr (K == 4) {
        const uint4 v = tex1Dfetch<uint4>(t4, i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
      if constexpr (K == 2) {
        const float x = (float)(lane + ((w & 1) << 5)) + 0.25f, y = (float)(u + ((it & 1) << 3) + ((w >> 1) & 3)) + 0.75f;
        const float4 f = tex2D<float4>(t2, x, y);
        acc ^= __float_as_uint(f.x) ^ __float_as_uint(f.y) ^ __float_as_uint(f.z) ^ __float_as_uint(f.w);
      }
    }
  }
  OP_END
  OP_SINK(acc)
}
OP_KERNEL(l_tld, TPB) { tex_body<1>(a); }
OP_KERNEL(l_tld4, TPB) { tex_body<4>(a); }
OP_KERNEL(l_tex, TPB) { tex_body<2>(a); }

__host__ __device__ inline unsigned atoms64_off(int u, int lane, int warp, int) {
  return (unsigned)((u * (TPB / 32) + warp) * 32 + lane) * 8;
}
OP_KERNEL(l_atoms64, TPB) {
  extern __shared__ __align__(128) unsigned long long sq[];
  const unsigned base = (unsigned)__cvta_generic_to_shared(sq);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < U * TPB; i += TPB) sq[i] = (unsigned long long)i * 0x9E3779B97F4A7C15ull;
  __syncthreads();
  unsigned long long x[U];
#pragma unroll
  for (int u = 0; u < U; u++) x[u] = threadIdx.x * 2654435761ull + u;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++)
      asm volatile("atom.shared.exch.b64 %0, [%1], %0;" : "+l"(x[u]) : "r"(base + atoms64_off(u, lane, w, it)) : "memory");
  }
  OP_END
  unsigned long long s = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) s ^= x[u];
  OP_SINK(s ^ (s >> 32))
}

// chain u's 16-B cell, one every warp reads: row u, its two cells alternating by trip (so no load is loop-invariant);
// a.mode (0 when timed) splits a warp's lanes over 8 cells only when it isn't
__host__ __device__ inline unsigned bcast_off(int u, int, int, int it) {
  return (unsigned)(u * 128) ^ (((unsigned)it & 1u) << 4);
}

template <int V, int SPACE>  // SPACE: 0 shared, 1 global, 2 global .nc
DC_DEV void bcast_body(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned sb[];
  const unsigned m = (unsigned)a.mode, split = (threadIdx.x & m & 7u) << 5;
  const unsigned base = (unsigned)__cvta_generic_to_shared(sb) + split;
  const unsigned long long gbase = (unsigned long long)a.g + split;
  if constexpr (SPACE == 0) {
    for (int i = threadIdx.x; i < ROWS * 32 + 32; i += TPB) sb[i] = (unsigned)i * 0x9E3779B9u ^ m;
  }
  __syncthreads();
  unsigned acc[U];
#pragma unroll
  for (int u = 0; u < U; u++) acc[u] = threadIdx.x * 0x9E3779B9u + u;
  OP_BEGIN
  OP_LOOP(it) {
    const unsigned d = ((unsigned)it & 1u) << 4;
    const unsigned s0 = base ^ d;
    const unsigned long long g0 = gbase ^ d;
#pragma unroll
    for (int u = 0; u < U; u++) {
      unsigned p, q, r = 0, s = 0;
      if constexpr (V == 4 && SPACE == 0)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(p), "=r"(q), "=r"(r), "=r"(s) : "r"(s0 + u * 128) : "memory");
      if constexpr (V == 2 && SPACE == 0)
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(p), "=r"(q) : "r"(s0 + u * 128) : "memory");
      if constexpr (V == 4 && SPACE == 1)
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(p), "=r"(q), "=r"(r), "=r"(s) : "l"(g0 + u * 128) : "memory");
      if constexpr (V == 4 && SPACE == 2)
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(p), "=r"(q), "=r"(r), "=r"(s) : "l"(g0 + u * 128) : "memory");
      acc[u] ^= V == 4 ? p ^ q ^ r ^ s : p ^ q;
    }
  }
  OP_END
  unsigned t = 0;
#pragma unroll
  for (int u = 0; u < U; u++) t ^= acc[u];
  OP_SINK(t)
}
// CORE-H32's scalar co-issue minimum (DADD, IMAD, DADD, FADD round-robin) with one broadcast LDS.128 every G of its
// ops: that round's two DADDs add the load's two aligned register pairs, so the load adds no instruction and ptxas
// can't narrow it. The cells hold doubles in [1, 2) (an odd word's high bits 0x3FF), so no chain overflows.
__host__ __device__ inline unsigned bmix_off(int k, int, int, int it) {
  return (unsigned)(k * 32) ^ (((unsigned)it & 1u) << 4);
}
constexpr int BMIX_WORDS = 512 + 32;
template <int G>
DC_DEV void bmix_body(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned sb[];
  const unsigned base = (unsigned)__cvta_generic_to_shared(sb);
  for (int i = threadIdx.x; i < BMIX_WORDS; i += TPB) {
    const unsigned h = (unsigned)i * 0x9E3779B9u ^ (unsigned)a.mode;
    sb[i] = (i & 1) ? 0x3FF00000u | (h & 0xFFFFFu) : h;
  }
  __syncthreads();
  WLane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    const unsigned s0 = base ^ (((unsigned)it & 1u) << 4);
#pragma unroll
    for (int b = 0; b < 128; b++) {
      if (b % (G / 4) == 0) {
        unsigned p, q, r, s;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(p), "=r"(q), "=r"(r), "=r"(s)
                     : "r"(s0 + (b / (G / 4)) * 32)
                     : "memory");
        asm volatile("{\n.reg .b64 x;\nmov.b64 x,{%1,%2};\nadd.rn.f64 %0,%0,x;\n}" : "+d"(l.d[(4 * b) & 7]) : "r"(p), "r"(q));
        l.op<W_IMAD>(4 * b + 1);
        asm volatile("{\n.reg .b64 x;\nmov.b64 x,{%1,%2};\nadd.rn.f64 %0,%0,x;\n}" : "+d"(l.d[(4 * b + 2) & 7]) : "r"(r), "r"(s));
        l.op<W_FADD>(4 * b + 3);
      } else {
        w1_round_<W_DADD, W_IMAD, W_DADD, W_FADD>(l, b, std::make_index_sequence<4>{});
      }
    }
  }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}
OP_KERNEL(m_lds128b_g8, TPB) { bmix_body<8>(a); }
OP_KERNEL(m_lds128b_g16, TPB) { bmix_body<16>(a); }
OP_KERNEL(m_lds128b_g32, TPB) { bmix_body<32>(a); }
OP_KERNEL(m_dadd2_imad_fadd, TPB) { w_run<512, W_DADD, W_IMAD, W_DADD, W_FADD>(a); }

OP_KERNEL(l_lds128b, TPB) { bcast_body<4, 0>(a); }
OP_KERNEL(l_lds64b, TPB) { bcast_body<2, 0>(a); }
OP_KERNEL(l_ldg128b, TPB) { bcast_body<4, 1>(a); }
OP_KERNEL(l_ldgnc128b, TPB) { bcast_body<4, 2>(a); }

#define P(NAME, PAT, NOTE) {.name = #NAME, .kernel = (const void*)NAME, .tpb = TPB, .expect = PAT, .iters = 512, .note = NOTE}
#define PS(NAME, PAT, OPS, FN, DYN, NOTE)                                                                               \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = TPB, .expect = PAT, .smem_ops = OPS, .smem = FN, .dyn_smem = DYN, \
   .iters = 512, .note = NOTE}
static const OpProbe PROBES[] = {
    P(y_brev, "BREV=256,LOP3.LUT=256", "c=int32 w=BREV:1,LOP3.LUT:1"),
    P(y_bmsk, "BMSK=256", "c=int32 w=BMSK:1"),
    P(y_sgxt, "SGXT=256", "c=int32 w=SGXT:1"),
    P(y_vabsdiff4, "VABSDIFF4*=256", "c=int32 w=VABSDIFF4*:1"),
    P(y_viaddmax, "VIADDMNMX=256", "c=int32 w=VIADDMNMX:1"),
    P(y_viaddmin, "VIADDMNMX=256", "c=int32 w=VIADDMNMX:1"),
    P(y_vimax3, "VIMNMX3=256", "c=int32 w=VIMNMX3:1"),
    P(y_iabs, "IABS=128,IMAD.IADD=104,IADD3=24", "c=int32 w=IABS:1,IMAD.IADD:1,IADD3:1"),
    P(y_i2ip, "I2IP*=256", "c=int32 w=I2IP*:1"),
    P(y_imadw8, "IMAD.WIDE.U32=256", "c=int32 w=IMAD.WIDE.U32:2"),
    P(y_imadws, "IMAD.WIDE=256", "c=int32 w=IMAD.WIDE:2"),
    P(y_mulwide, "IMAD.WIDE.U32=256", "c=int32 w=IMAD.WIDE.U32:2"),
    P(y_lop3p, "LOP3.LUT=256", "c=int32 w=LOP3.LUT:1 p=LOP3.LUT:1"),
    P(y_isetp2, "ISETP.LT.U32.AND=256,PLOP3.LUT=255,SEL=257", "c=int32 w=SEL:1 p=ISETP.LT.U32.AND:1,PLOP3.LUT:1"),
    P(y_plop3, "ISETP.LT.U32.AND=256,ISETP.LT.U32.XOR=255", "c=int32 w=ISETP.LT.U32.AND:0 p=ISETP.LT.U32.AND:1,ISETP.LT.U32.XOR:1"),
    {.name = "y_r2p", .kernel = (const void*)y_r2p, .tpb = TPB, .expect = "R2P=64,SEL=448,IADD3=192,IMAD.IADD=64",
     .iters = 2048, .note = "c=mix w=SEL:1,IADD3:1,IMAD.IADD:1,R2P:0 p=R2P:7"},
    P(y_cs2r, "CS2R=256,LOP3.LUT=256", "c=mix w=CS2R:2,LOP3.LUT:1"),
    P(y_clk32, "S2UR=128,CS2R.32=128,LOP3.LUT=128", "c=mix w=CS2R.32:1,LOP3.LUT:1 u=S2UR:1"),
    P(y_gtimer, "S2UR=128,CS2R.32=128,IADD3=128", "c=mix w=CS2R.32:1,IADD3:1 u=S2UR:1"),
    P(y_i2f64, "I2F.F64=128,F2I.F64.TRUNC=128", "c=fp64 w=I2F.F64:2,F2I.F64.TRUNC:1"),
    P(y_f2i64, "I2F.F64.S64=112,F2I.S64.F64.TRUNC=112", "c=fp64 w=I2F.F64.S64:2,F2I.S64.F64.TRUNC:2"),
    P(y_hadd2f32, "HADD2.F32=128,F2FP.F16.F32.PACK_AB=64", "c=other w=HADD2.F32:1,F2FP.F16.F32.PACK_AB:1"),
    P(y_f2fp_h, "HADD2.F32=128,F2FP.F16.F32.PACK_AB=76", "c=other w=HADD2.F32:1,F2FP.F16.F32.PACK_AB:1"),
    P(y_movm, "MOVM*=256", "c=other w=MOVM*:1"),
    P(y_redux, "REDUX.SUM=256,IMAD.U32=256,LOP3.LUT=256", "c=mix w=IMAD.U32:1,LOP3.LUT:1 u=REDUX.SUM:1"),
    P(y_match, "MATCH.ANY=256", "c=other w=MATCH.ANY:1"),
    P(y_mulw_lop3, "IMAD.WIDE.U32=128,LOP3.LUT=128", "c=int32 w=IMAD.WIDE.U32:2,LOP3.LUT:1"),
    P(y_mulw_lop3x3, "IMAD.WIDE.U32=64,LOP3.LUT=192", "c=int32 w=IMAD.WIDE.U32:2,LOP3.LUT:1"),
    P(y_mulw3_lop3, "IMAD.WIDE.U32=192,LOP3.LUT=64", "c=int32 w=IMAD.WIDE.U32:2,LOP3.LUT:1"),
    P(y_mulw_iadd3, "IMAD.WIDE.U32=128,IADD3=128", "c=int32 w=IMAD.WIDE.U32:2,IADD3:1"),
    P(y_mulw_imad, "IMAD.WIDE.U32=128,IMAD=128", "c=int32 w=IMAD.WIDE.U32:2,IMAD:1"),
    P(y_mulw_fadd, "IMAD.WIDE.U32=128,FADD=128", "c=mix w=IMAD.WIDE.U32:2,FADD:1"),
    P(y_mulw_dadd, "IMAD.WIDE.U32=128,DADD=128", "c=mix w=IMAD.WIDE.U32:2,DADD:2"),
    P(y_mulw_hadd2, "IMAD.WIDE.U32=128,HFMA2.MMA=64,HADD2=64", "c=mix w=IMAD.WIDE.U32:2,HFMA2.MMA:1,HADD2:1"),
    P(y_lop3p_imad, "LOP3.LUT=128,IMAD=128", "c=int32 w=LOP3.LUT:1,IMAD:1 p=LOP3.LUT:1"),
    P(y_cs2r_imad, "CS2R=128,LOP3.LUT=128,IMAD=128", "c=mix w=CS2R:2,LOP3.LUT:1,IMAD:1"),
    P(y_i2ip_imad, "I2IP*=128,IMAD=128", "c=int32 w=I2IP*:1,IMAD:1"),
    P(y_viaddmax_imad, "VIADDMNMX=128,IMAD=128", "c=int32 w=VIADDMNMX:1,IMAD:1"),
    P(y_vimax3_imad, "VIMNMX3=128,IMAD=128", "c=int32 w=VIMNMX3:1,IMAD:1"),
    P(y_lop3_r, "LOP3.LUT=256", "c=int32 w=LOP3.LUT:1"),
    P(y_imad_r, "IMAD=256", "c=int32 w=IMAD:1"),
    P(y_imad_lop3, "IMAD=128,LOP3.LUT=128", "c=int32 w=IMAD:1,LOP3.LUT:1"),
    P(y_imadp_on, "IMAD=256", "c=int32 w=IMAD:1"),
    P(y_imadp_off, "IMAD=256", "c=int32 w=IMAD:1"),
    P(y_sel, "SEL=256", "c=int32 w=SEL:1"),
    P(b_ubra, "IMAD=257,LOP3.LUT=314,P2R=2,ISETP.NE.AND=2", "c=mix w=IMAD:1,LOP3.LUT:1"),
    P(b_dbra, "IMAD=256,LOP3.LUT=320", "c=mix w=IMAD:1,LOP3.LUT:1"),
    P(b_ubra16, "IMAD=257,LOP3.LUT=273,BRA=33", "c=mix w=IMAD:1,LOP3.LUT:1"),
    P(b_dbra16, "IMAD=256,LOP3.LUT=272,BSSY=16,BSYNC=16,BRA=33", "c=mix w=IMAD:1,LOP3.LUT:1"),
    P(b_brx4, "IMAD=513,LOP3.LUT=528,BRX=16,BRA=49,LDC=16,IMAD.SHL.U32=16,SHF.R.S32.HI=16", "c=mix w=IMAD:1,LOP3.LUT:1"),
    P(y_lea, "LEA=112,IMAD=144", "c=int32 w=LEA:1,IMAD:1"),
    P(y_iadd64, "IADD3=256,IMAD.X=256", "c=int32 w=IADD3:1,IMAD.X:1"),
    P(y_shfli, "SHFL.IDX=256", "c=other w=SHFL.IDX:1"),
    P(y_shflu, "SHFL.UP=256", "c=other w=SHFL.UP:1"),
    PS(l_ldsm1, "LDSM.16.M88=8", "LDSM.16.M88*8", ldsm_off, ROWS * 128, "c=other w=LDSM.16.M88:1"),
    PS(l_ldsm2, "LDSM.16.M88.2=8", "LDSM.16.M88.2*8", ldsm_off, ROWS * 128, "c=other w=LDSM.16.M88.2:2"),
    PS(l_ldsm4, "LDSM.16.M88.4=8", "LDSM.16.M88.4*8", ldsm_off, ROWS * 128, "c=other w=LDSM.16.M88.4:4"),
    PS(l_ldsm4t, "LDSM.16.MT88.4=8", "LDSM.16.MT88.4*8", ldsm_off, ROWS * 128, "c=other w=LDSM.16.MT88.4:4"),
    PS(l_lds64, "LDS.64=8", "LDS.64*8", lds64_off, ROWS * 256, "c=other w=LDS.64:2"),
    P(l_ldc_u, "LDC=8", "c=other w=LDC:1"),
    P(l_ldc64_u, "LDC.64=8", "c=other w=LDC.64:2"),
    {.name = "l_ldc_d", .kernel = (const void*)l_ldc_d, .tpb = TPB, .expect = "LDC=8", .iters = 64, .note = "c=other w=LDC:1"},
    P(l_tld, "TLD.LZ=8", "c=other w=TLD.LZ:1"),
    P(l_tld4, "TLD.LZ=8", "c=other w=TLD.LZ:4"),
    P(l_tex, "TEX.LL=8", "c=other w=TEX.LL:4"),
    PS(l_atoms64, "ATOMS.EXCH.64=8", "ATOMS.EXCH.64*8", atoms64_off, U * TPB * 8, "c=other w=ATOMS.EXCH.64:2"),
    PS(m_lds128b_g8, "DADD=256,IMAD=128,FADD=128,LDS.128=64", "LDS.128*64", bmix_off, BMIX_WORDS * 4,
       "c=mix w=DADD:2,IMAD:1,FADD:1,LDS.128:4"),
    PS(m_lds128b_g16, "DADD=256,IMAD=128,FADD=128,LDS.128=32", "LDS.128*32", bmix_off, BMIX_WORDS * 4,
       "c=mix w=DADD:2,IMAD:1,FADD:1,LDS.128:4"),
    PS(m_lds128b_g32, "DADD=256,IMAD=128,FADD=128,LDS.128=16", "LDS.128*16", bmix_off, BMIX_WORDS * 4,
       "c=mix w=DADD:2,IMAD:1,FADD:1,LDS.128:4"),
    P(m_dadd2_imad_fadd, "DADD=256,IMAD=128,FADD=128", "c=mix w=DADD:2,IMAD:1,FADD:1"),
    PS(l_lds128b, "LDS.128=8,LOP3.LUT=17,IMAD.SHL.U32=1,VIADD=1", "LDS.128*8", bcast_off, ROWS * 128 + 128,
       "c=other w=LDS.128:4,LOP3.LUT:1,IMAD.SHL.U32:1,VIADD:1 p=ISETP.GE.AND:1"),
    PS(l_lds64b, "LDS.64=8,LOP3.LUT=9,IMAD.SHL.U32=1,VIADD=1", "LDS.64*8", bcast_off, ROWS * 128 + 128,
       "c=other w=LDS.64:2,LOP3.LUT:1,IMAD.SHL.U32:1,VIADD:1 p=ISETP.GE.AND:1"),
    P(l_ldg128b, "LDG.E.128=8,LOP3.LUT=17,IMAD.SHL.U32=1,VIADD=1",
      "c=other w=LDG.E.128:4,LOP3.LUT:1,IMAD.SHL.U32:1,VIADD:1 p=ISETP.GE.AND:1"),
    P(l_ldgnc128b, "LDG.E.128.CONSTANT=8,LOP3.LUT=17,IMAD.SHL.U32=1,VIADD=1",
      "c=other w=LDG.E.128.CONSTANT:4,LOP3.LUT:1,IMAD.SHL.U32:1,VIADD:1 p=ISETP.GE.AND:1"),
};
#undef P
#undef PS

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
