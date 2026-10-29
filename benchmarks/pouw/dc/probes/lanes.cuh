// The scalar lane ops the DC probes time alone (scalar.cu) and beside the tensor pipe (sm90_wgmma.cu, sm100_tcgen05.cu).
// Each op reads and writes the thread's own 8 registers (8 independent chains), so ptxas can neither fold nor hoist it.
#pragma once
#include "../dc.cuh"

// mix: add.u32 and FFMA alternating (ptxas splits the adds between IADD3 and IMAD.IADD); duo: the fold, 3 ops a word.
// The multiply lanes (idp4a, dfma, hfma2, ...) are Lane2's, below.
// redux: one redux.sync.xor.b32 (the whole warp's XOR in one instruction, sm_80+); reduce: the same by 5 butterfly
// shuffles and 5 XORs (the fallback). LI-CP-W's per-segment checkpoint is one of these a warp a segment.
// li: LOP3 and IMAD alternating, LI's op mix; lif: LI with FFMA filler. imadw: mad.wide.u32 (IMAD.WIDE.U32), the 64-bit
// product and its addend a register pair, so 4 chains.
enum { L_MIX, L_IMAD, L_LOP3, L_SHFL, L_DUO, L_IADD, L_FFMA, L_SHF, L_LEA, L_REDUX, L_REDUCE, L_LI, L_LIF, L_IMADW };
struct Lane {
  uint32_t r[8];
  float f[8];
  DC_DEV explicit Lane(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
      f[i] = 1.0f + (float)(r[i] & 0xffu) * 0x1p-12f;
    }
  }
  template <int X>
  DC_DEV void op(int i) {
    const int s = i & 7, t = (i + 1) & 7;
    if constexpr (X == L_MIX) {
      if (i & 1) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(f[(i >> 1) & 7]) : "f"(f[((i >> 1) + 1) & 7]));
      else asm volatile("add.u32 %0,%0,%1;" : "+r"(r[(i >> 1) & 7]) : "r"(r[((i >> 1) + 1) & 7]));
    }
    if constexpr (X == L_IMAD) asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(0x9E3779B9u));
    if constexpr (X == L_IMADW) {
      const int p = 2 * (i & 3);
      asm volatile("{\n.reg .b64 w;\nmov.b64 w,{%0,%1};\nmad.wide.u32 w,%0,%2,w;\nmov.b64 {%0,%1},w;\n}"
                   : "+r"(r[p]), "+r"(r[p + 1])
                   : "r"(0x9E3779B9u));
    }
    if constexpr (X == L_LOP3) asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(r[s]) : "r"(r[t]), "r"(0x85EBCA6Bu));
    if constexpr (X == L_SHFL) asm volatile("shfl.sync.bfly.b32 %0,%0,1,0x1f,0xffffffff;" : "+r"(r[s]));
    if constexpr (X == L_IADD) asm volatile("add.u32 %0,%0,%1;" : "+r"(r[s]) : "r"(r[t]));
    if constexpr (X == L_FFMA) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == L_SHF) asm volatile("shf.l.wrap.b32 %0,%0,%1,13;" : "+r"(r[s]) : "r"(r[t]));
    // rotl(x, 13) + y: ptxas fuses it into one LEA.HI (duo's third op)
    if constexpr (X == L_LEA)
      asm volatile("{\n.reg .b32 t;\nshf.l.wrap.b32 t,%1,%1,13;\nadd.u32 %0,%0,t;\n}" : "+r"(r[s]) : "r"(r[t]));
    // XOR the whole warp into r[s] in one instruction (REDUX in SASS).
    if constexpr (X == L_REDUX) asm volatile("redux.sync.xor.b32 %0,%0,0xffffffff;" : "+r"(r[s]));
    // The same reduction by 5 butterfly shuffles and 5 XORs (xor.b32 lowers to LOP3.LUT): the pre-sm_80 fallback.
    if constexpr (X == L_REDUCE)
      asm volatile(
          "{\n.reg .b32 t;\n"
          "shfl.sync.bfly.b32 t,%0,16,0x1f,0xffffffff;\nxor.b32 %0,%0,t;\n"
          "shfl.sync.bfly.b32 t,%0, 8,0x1f,0xffffffff;\nxor.b32 %0,%0,t;\n"
          "shfl.sync.bfly.b32 t,%0, 4,0x1f,0xffffffff;\nxor.b32 %0,%0,t;\n"
          "shfl.sync.bfly.b32 t,%0, 2,0x1f,0xffffffff;\nxor.b32 %0,%0,t;\n"
          "shfl.sync.bfly.b32 t,%0, 1,0x1f,0xffffffff;\nxor.b32 %0,%0,t;\n}"
          : "+r"(r[s]));
  }
  // p is a word of the chain's own state, so ptxas can't fold it to a constant.
  template <int T>
  DC_DEV void duo_op(int i) {
    DuoSt st{r[i & 7], r[(i + 4) & 7]};
    duo<T>(st, r[(i + 1) & 7]);
    r[i & 7] = st.y;
    r[(i + 4) & 7] = st.z;
  }
  DC_DEV uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x += r[i] ^ __float_as_uint(f[i]);
    return x;
  }
};
// The multiply lanes a side pipe could make checkpoint words on, in Lane2 so the lanes above keep their SASS. idp4a:
// dp4a.s32.s32 (IDP.4A.S8.S8, four int8 products into s32); idp2a: dp2a.lo (IDP.2A.LO); dfma: fma.rn.f64 on eight FP64
// chains of its own; hfma2: fma.rn.f16x2 (two FP16 FMAs); popc and prmt. The p_ kinds alternate two of them (lane_pair).
enum { L_IDP4A = 32, L_IDP2A, L_DFMA, L_HFMA2, L_POPC, L_PRMT, L_P_IDP4A_FFMA, L_P_IDP4A_DFMA, L_P_DFMA_FFMA,
       L_P_IDP4A_LOP3 };
struct Lane2 : Lane {
  double d[8];
  DC_DEV explicit Lane2(uint32_t tid) : Lane(tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) d[i] = 1.0 + (double)(r[i] & 0xffffu) * 0x1p-20;
  }
  template <int X>
  DC_DEV void op(int i) {
    const int s = i & 7, t = (i + 1) & 7, u = (i + 2) & 7;
    if constexpr (X < L_IDP4A) Lane::op<X>(i);
    if constexpr (X == L_IDP4A) asm volatile("dp4a.s32.s32 %0,%1,%2,%0;" : "+r"(r[s]) : "r"(r[t]), "r"(r[u]));
    if constexpr (X == L_IDP2A) asm volatile("dp2a.lo.s32.s32 %0,%1,%2,%0;" : "+r"(r[s]) : "r"(r[t]), "r"(r[u]));
    if constexpr (X == L_DFMA) asm volatile("fma.rn.f64 %0,%0,%1,0d3FF0000000000000;" : "+d"(d[s]) : "d"(d[t]));
    if constexpr (X == L_HFMA2) asm volatile("fma.rn.f16x2 %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(0x3C003C00u));
    if constexpr (X == L_POPC) asm volatile("popc.b32 %0,%1;" : "=r"(r[s]) : "r"(r[t]));
    if constexpr (X == L_PRMT) asm volatile("prmt.b32 %0,%0,%1,0x6420;" : "+r"(r[s]) : "r"(r[t]));
  }
  DC_DEV uint32_t fold() const {
    uint32_t x = Lane::fold();
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const unsigned long long b = (unsigned long long)__double_as_longlong(d[i]);
      x += (uint32_t)b ^ (uint32_t)(b >> 32);
    }
    return x;
  }
};
template <int Q, int N>
DC_DEV void duo_ops(Lane& l) {
  if constexpr (Q < N) {
    l.duo_op<Q % DUO_NPOS>(Q);
    duo_ops<Q + 1, N>(l);
  }
}
// R ops alternating X and Y: when the pair's rate exceeds both alone rates, X and Y issue to different pipes.
template <int X, int Y, int R, class L>
DC_DEV void lane_pair(L& l) {
#pragma unroll
  for (int q = 0; q < R; q += 2) {
    l.template op<X>(q >> 1);
    l.template op<Y>((q >> 1) + 4);
  }
}
// R lane ops (the duo lane: R/3 words of three ops each; the LI lane: LI's LOP3 + IMAD pair, R/2 of each; lif: LI with
// one FFMA filler op in eight, 7 LOP3 + 7 IMAD + 2 FFMA a 16)
template <int X, int R, class L>
DC_DEV void lane_ops(L& l) {
  if constexpr (X == L_DUO) {
    duo_ops<0, R / 3>(l);
  } else if constexpr (X == L_LI) {
    lane_pair<L_LOP3, L_IMAD, R>(l);
  } else if constexpr (X == L_LIF) {
#pragma unroll
    for (int q = 0; q < R; q++) {
      if (q % 8 == 7) l.template op<L_FFMA>(q / 8);
      else if ((q % 8 + q / 8) % 2 == 0) l.template op<L_LOP3>(q);
      else l.template op<L_IMAD>(q);
    }
  } else if constexpr (X == L_P_IDP4A_FFMA) {
    lane_pair<L_IDP4A, L_FFMA, R>(l);
  } else if constexpr (X == L_P_IDP4A_DFMA) {
    lane_pair<L_IDP4A, L_DFMA, R>(l);
  } else if constexpr (X == L_P_DFMA_FFMA) {
    lane_pair<L_DFMA, L_FFMA, R>(l);
  } else if constexpr (X == L_P_IDP4A_LOP3) {
    lane_pair<L_IDP4A, L_LOP3, R>(l);
  } else {
#pragma unroll
    for (int q = 0; q < R; q++) l.template op<X>(q);
  }
}

// The memory lanes: R accesses a trip, each one warp instruction whose 32 lanes touch 32 consecutive words (LDS.128: 4
// words a lane). lde: ld.u32 through a generic address (LD.E, which no census class names); atome: red.relaxed.gpu.xor
// through a generic address (ATOM.E, ATOME32); redg: the same to a global address (REDG.E, REDG32); atoms:
// red.shared.xor (ATOMS, ATOMS32); lds128: ld.shared.v4.u32 (LDS.128, LDS128). A global access goes to its warp's own
// 8 KB of `g`, a 128-byte line an access and the trip's parity picking one of two sets of lines, so the lane warps' loads
// stay in L1. Shared accesses go to the block's SX bytes of scratch: an atomic to its warp's own MEM_SX of it, an
// LDS.128 to a 512-byte line of its own among the trip's R and the parity's two sets (ptxas merges two loads of one
// address and hoists a load whose address no trip changes), so SX holds a warp's MEM_SX each and 2R lines. A loaded word
// is XORed into a chain; an atomic XORs a chain into memory.
enum { M_LDE, M_ATOME, M_REDG, M_ATOMS, M_LDS128 };
constexpr int MEM_WARP_BYTES = 8192, MEM_BLOCK_BYTES = 32 * MEM_WARP_BYTES, MEM_SX = 1024;
DC_DEV unsigned char* mem_global(unsigned char* g) {
  return g + (size_t)blockIdx.x * MEM_BLOCK_BYTES + (threadIdx.x >> 5) * MEM_WARP_BYTES + (threadIdx.x & 31) * 4;
}
// `sx`'s words each set to its own offset (read after OP_BEGIN's barrier); its shared address.
template <int SX>
DC_DEV uint32_t mem_shared(unsigned char* sx) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sx);
  for (int i = threadIdx.x; i < SX / 4; i += blockDim.x) w[i] = i;
  return smem_u32(sx);
}
template <int M, int R, int SX>
DC_DEV void mem_ops(Lane& l, unsigned char* gp, uint32_t sb, int n) {
  static_assert(M != M_LDS128 || SX / 512 >= 2 * R, "an LDS.128 lane needs two lines of scratch an access");
  const uint32_t lane = threadIdx.x & 31, sp = sb + (threadIdx.x >> 5) * MEM_SX;
#pragma unroll
  for (int q = 0; q < R; q++) {
    unsigned char* p = gp + (((n & 1) * R + q) % (MEM_WARP_BYTES / 128)) * 128;
    uint32_t& x = l.r[q & 7];
    if constexpr (M == M_LDE) {
      uint32_t v;
      asm volatile("ld.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory");
      x ^= v;
    }
    if constexpr (M == M_ATOME) asm volatile("red.relaxed.gpu.xor.b32 [%0],%1;" ::"l"(p), "r"(x) : "memory");
    if constexpr (M == M_REDG)
      asm volatile("{\n.reg .u64 a;\ncvta.to.global.u64 a,%0;\nred.relaxed.gpu.global.xor.b32 [a],%1;\n}" ::"l"(p),
                   "r"(x)
                   : "memory");
    if constexpr (M == M_ATOMS)
      asm volatile("red.shared.xor.b32 [%0],%1;" ::"r"(sp + (q & 7) * 128 + lane * 4), "r"(x) : "memory");
    if constexpr (M == M_LDS128) {
      uint32_t a, b, c, d;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];"
                   : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                   : "r"(sb + (((n & 1) * R + q) % (SX / 512)) * 512 + lane * 16)
                   : "memory");
      x ^= a ^ b ^ c ^ d;
    }
  }
}
