// Receipt φ candidates on sm_120: what each costs in SASS and in time, alone and fused into an FP8 mma.sync mainloop.
//
// A receipt chains y_t = φ_t(y_{t-1}, p_t) over committed accumulator words p_t ("Three lemmas toward ε < 1% PoUW",
// Receipt lemma). Each candidate below is one φ, keyed by position t from the constant bank (KA/KB odd multipliers,
// KK addends, at most one per instruction), so a key costs no instruction when t is static. Modes:
//
//   phi gate SEED KSEED          every candidate's chain for a few words, for gate.py to recompute
//   phi pure ITERS REPS          each candidate alone: PU chains of PW register words per thread, one wave of blocks
//   phi probe ITERS REPS         one SASS opcode class (or a fixed mix) per kernel, for the pipe model
//   phi fused ITERS REPS [OCC]   FP8 m16n8k32 MMAs into FT tiles per warp, φ over every accumulator word every C MMAs
//                                (commit stride L = 32 C), against the same loop with no φ; OCC pins blocks per SM
//   phi one MODE INDEX ITERS [FM [OCC]]   one launch of one kernel (for ncu)
//
// Results are JSON lines on stdout. Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 phi.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#define CK(x)                                                                                            \
  do {                                                                                                   \
    cudaError_t e_ = (x);                                                                                \
    if (e_ != cudaSuccess) {                                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));               \
      exit(3);                                                                                           \
    }                                                                                                    \
  } while (0)

constexpr int NKEY = 32;
__constant__ uint32_t KA[NKEY], KB[NKEY], KK[NKEY];

__host__ __device__ __forceinline__ uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__host__ __device__ __forceinline__ uint32_t gen(uint32_t seed, uint32_t a, uint32_t b, uint32_t c) {
  return lowbias32(seed ^ lowbias32((a * 0x9E3779B1U) ^ lowbias32((b * 0x85EBCA77U) ^ lowbias32(c + 0x632BE5ABU))));
}

__device__ __forceinline__ uint32_t rotl(uint32_t x, int r) { return __funnelshift_l(x, x, r); }
__host__ __device__ constexpr int rot_of(int t) { return ((t * 7 + 5) % 31) + 1; }

enum Kind { MAD = 0, RA, RAM, XMA, AR, MUM, XSM, XSM4, MXM, CHI, XSM2, DUO, NK };
static const char* KIND_NAME[NK] = {"mad", "ra",  "ram", "xma", "ar",   "mum",
                                    "xsm", "xsm4", "mxm", "chi", "xsm2", "duo"};
__host__ __device__ constexpr bool wide(int k) { return k == MUM || k == DUO; }
__host__ __device__ constexpr bool paired(int k) { return k == RAM || k == XMA; }

struct St {
  uint32_t y, z;
};

// Absorb N words p[0..N-1] at key positions T0..T0+N-1 (paired kinds take them two at a time, keyed by the first).
// MUM keeps a 64-bit state S = (y = hi, z = lo) and DUO two words (y, z); every other candidate keeps one word y.
template <int K, int N, int T0>
__device__ __forceinline__ void absorb(St& s, const uint32_t* p) {
  if constexpr (paired(K)) static_assert(N % 2 == 0, "pairs");
#pragma unroll
  for (int i = 0; i < N; i += (paired(K) ? 2 : 1)) {
    const int t = T0 + i;
    if constexpr (K == MAD) {
      s.y = s.y * KA[t] + p[i];
    } else if constexpr (K == RA) {
      s.y = rotl(s.y, rot_of(t)) + p[i];
    } else if constexpr (K == RAM) {
      s.y = rotl(s.y, rot_of(t)) + p[i];
      s.y = s.y * KA[t] + p[i + 1];
    } else if constexpr (K == XMA) {
      s.y = ((s.y ^ p[i] ^ KK[t]) * KA[t]) + p[i + 1];
    } else if constexpr (K == AR) {
      s.y = rotl(s.y + p[i] + KK[t], rot_of(t));
    } else if constexpr (K == MUM) {
      const uint32_t u = s.y ^ s.z ^ p[i];
      const uint64_t w = (uint64_t)u * KA[t] + (((uint64_t)s.y << 32) | s.z);
      s.y = (uint32_t)(w >> 32);
      s.z = (uint32_t)w;
    } else if constexpr (K == XSM) {
      s.y = (s.y ^ (s.y >> 16) ^ p[i]) * KA[t];
    } else if constexpr (K == XSM4) {
      const uint32_t v = s.y ^ p[i];
      s.y = (v ^ (v >> 16) ^ KK[t]) * KA[t];
    } else if constexpr (K == MXM) {
      const uint32_t u = s.y * KA[t] + p[i];
      s.y = (u ^ (u >> 16) ^ KK[t]) * KB[t];
    } else if constexpr (K == CHI) {
      const uint32_t u = s.y ^ p[i] ^ KK[t];
      s.y = u ^ (~rotl(u, 1) & rotl(u, 2));
    } else if constexpr (K == XSM2) {
      uint32_t u = (s.y ^ (s.y >> 16) ^ p[i]) * KA[t];
      s.y = (u ^ (u >> 13) ^ KK[t]) * KB[t];
    } else if constexpr (K == DUO) {
      const uint32_t u = s.y ^ p[i] ^ KK[t];
      s.y = u * KA[t] + s.z;
      s.z = rotl(u, rot_of(t)) + s.y;
    }
  }
}

// The receipt, once per block. DUO's z is y + rotl(u, r), so y ^ z would see an additive shift of both halves (2^31
// above all) only through carries: its fold rotates z instead.
template <int K>
__device__ __forceinline__ uint32_t finish(const St& s) {
  return K == MUM ? (s.y ^ s.z) : K == DUO ? (s.y ^ rotl(s.z, 16)) : s.y;
}

// ---------------------------------------------------------------------------------------------------------------- pure
constexpr int PU = 4, PW = 8, TPB = 256;

// The words a chain absorbs are the other chains' states (chain u takes chain u+1's, and u+2's for the second word of a
// pair), all read before any is written, so every word changes every step and ptxas can't hoist or reassociate
// around a loop-invariant word; each step is the candidate's instruction sequence on fresh registers.
template <int K>
__global__ void __launch_bounds__(TPB) pure_k(uint32_t* out, int iters, uint32_t seed, long long* clk) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  St s[PU];
#pragma unroll
  for (int u = 0; u < PU; ++u) {
    s[u].y = gen(seed, tid, u, 100);
    s[u].z = gen(seed, tid, u, 101);
  }
  __syncthreads();
  const long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int r = 0; r < PW; r += (paired(K) ? 2 : 1)) {
      uint32_t w[PU][2];
#pragma unroll
      for (int u = 0; u < PU; ++u) {
        w[u][0] = s[(u + 1) % PU].y;
        w[u][1] = s[(u + 2) % PU].y;
      }
#pragma unroll
      for (int u = 0; u < PU; ++u) {
        if (r == 0) absorb<K, paired(K) ? 2 : 1, 0>(s[u], w[u]);
        if (r == 1) absorb<K, paired(K) ? 2 : 1, 1>(s[u], w[u]);
        if (r == 2) absorb<K, paired(K) ? 2 : 1, 2>(s[u], w[u]);
        if (r == 3) absorb<K, paired(K) ? 2 : 1, 3>(s[u], w[u]);
        if (r == 4) absorb<K, paired(K) ? 2 : 1, 4>(s[u], w[u]);
        if (r == 5) absorb<K, paired(K) ? 2 : 1, 5>(s[u], w[u]);
        if (r == 6) absorb<K, paired(K) ? 2 : 1, 6>(s[u], w[u]);
        if (r == 7) absorb<K, paired(K) ? 2 : 1, 7>(s[u], w[u]);
      }
    }
  }
  const long long t1 = clock64();
#pragma unroll
  for (int u = 0; u < PU; ++u) out[tid * PU + u] = finish<K>(s[u]);
  if (threadIdx.x == 0) {
    clk[2 * blockIdx.x] = t0;
    clk[2 * blockIdx.x + 1] = t1;
  }
}

// --------------------------------------------------------------------------------------------------------------- probe
// Each step reads two evolving chains, so no chain of steps folds into fewer instructions.
enum Probe {
  P_LOP3 = 0, P_IADD3, P_SHF, P_PRMT, P_IMAD, P_IMADW, P_IMADHI, P_FFMA, P_FADD,
  P_LOP3_IMAD, P_LOP3_FFMA, P_IADD3_SHF, P_LOP3_IMADW, P_SHF_IMAD, P_2LOP3_IMAD, P_LOP3_IMAD_FFMA,
  P_LEAHI, P_LEAHI_IMAD, NP
};
static const char* PROBE_NAME[NP] = {"lop3", "iadd3", "shf", "prmt", "imad", "imad.wide", "imad.hi", "ffma", "fadd",
                                     "lop3+imad", "lop3+ffma", "iadd3+shf", "lop3+imad.wide", "shf+imad",
                                     "2lop3+imad", "lop3+imad+ffma", "lea.hi", "lea.hi+imad"};

__device__ __forceinline__ uint32_t maj3(uint32_t a, uint32_t b, uint32_t c) { return (a & b) | (a & c) | (b & c); }
__device__ __forceinline__ uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

constexpr int QW = 4;
template <int P>
__global__ void __launch_bounds__(TPB) probe_k(uint32_t* out, int iters, uint32_t seed, long long* clk) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y[PU], q[PU][QW];
  uint64_t w[PU];
  float f[PU];
#pragma unroll
  for (int u = 0; u < PU; ++u) {
    y[u] = gen(seed, tid, u, 100);
    w[u] = ((uint64_t)gen(seed, tid, u, 101) << 32) | gen(seed, tid, u, 102);
    f[u] = __uint_as_float(0x3f000000u | (gen(seed, tid, u, 103) & 0x7fffffu));
#pragma unroll
    for (int r = 0; r < QW; ++r) q[u][r] = gen(seed, tid, u, r);
  }
  __syncthreads();
  const long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
#pragma unroll
      for (int u = 0; u < PU; ++u) {
        const int v = u ^ 1;
        const uint32_t c = q[u][r & (QW - 1)];
        const float cf = __uint_as_float(0x3e800000u | (c & 0x7fffffu));
        if constexpr (P == P_LOP3) y[u] = maj3(y[u], y[v], c);
        if constexpr (P == P_IADD3) y[u] = y[u] + y[v] + c;
        if constexpr (P == P_SHF) y[u] = __funnelshift_l(y[u], y[v], 13);
        if constexpr (P == P_PRMT) y[u] = __byte_perm(y[u], y[v], 0x5140);
        if constexpr (P == P_IMAD) y[u] = y[u] * KA[r] + y[v];
        if constexpr (P == P_IMADW) w[u] = (uint64_t)(uint32_t)w[u] * KA[r] + w[v];
        if constexpr (P == P_IMADHI) y[u] = madhi(y[u], KA[r], y[v]);
        if constexpr (P == P_FFMA) f[u] = fmaf(f[u], cf, f[v]);
        if constexpr (P == P_FADD) f[u] = f[u] + f[v];
        if constexpr (P == P_LOP3_IMAD) {
          y[u] = maj3(y[u], y[v], c);
          y[u] = y[u] * KA[r] + y[v];
        }
        if constexpr (P == P_LOP3_FFMA) {
          y[u] = maj3(y[u], y[v], c);
          f[u] = fmaf(f[u], cf, f[v]);
        }
        if constexpr (P == P_IADD3_SHF) {
          y[u] = y[u] + y[v] + c;
          y[u] = __funnelshift_l(y[u], y[v], 13);
        }
        if constexpr (P == P_LOP3_IMADW) {
          const uint32_t t = maj3((uint32_t)w[u], (uint32_t)(w[u] >> 32), c);
          w[u] = (uint64_t)t * KA[r] + w[v];
        }
        if constexpr (P == P_SHF_IMAD) {
          y[u] = __funnelshift_l(y[u], y[v], 13);
          y[u] = y[u] * KA[r] + y[v];
        }
        if constexpr (P == P_2LOP3_IMAD) {
          y[u] = maj3(y[u], y[v], c);
          y[u] = y[u] ^ (y[(u + 2) % PU] & ~c);
          y[u] = y[u] * KA[r] + y[v];
        }
        if constexpr (P == P_LOP3_IMAD_FFMA) {
          y[u] = maj3(y[u], y[v], c);
          y[u] = y[u] * KA[r] + y[v];
          f[u] = fmaf(f[u], cf, f[v]);
        }
        if constexpr (P == P_LEAHI) y[u] = rotl(y[u], 13) + y[v];
        if constexpr (P == P_LEAHI_IMAD) {
          y[u] = rotl(y[u], 13) + y[v];
          y[u] = y[u] * KA[r] + y[v];
        }
      }
    }
  }
  const long long t1 = clock64();
  uint32_t acc = 0;
#pragma unroll
  for (int u = 0; u < PU; ++u) acc ^= y[u] ^ (uint32_t)w[u] ^ (uint32_t)(w[u] >> 32) ^ __float_as_uint(f[u]);
  out[tid] = acc;
  if (threadIdx.x == 0) {
    clk[2 * blockIdx.x] = t0;
    clk[2 * blockIdx.x + 1] = t1;
  }
}

// --------------------------------------------------------------------------------------------------------------- fused
// Each warp runs FT m16n8k32 tiles; every C MMAs (commit stride L = 32 C) each lane absorbs its 4 accumulator words
// of tile j into its own chain j at positions 4j..4j+3. MODE bit 0 (ping-pong): consecutive strides write alternate
// accumulator sets (the stride's first MMA reads C from one and writes D to the other, as mma.sync allows), so the
// next stride's MMAs never wait for the absorb to read the words; bit 1: INT8 IMMA (s8 x s8 into INT32), PoUW's
// arithmetic, instead of FP8 E4M3 into FP32; bit 2 (with bit 0): the absorb interleaved with the next stride's MMAs
// (`il_stride`) instead of issued as one burst; bit 3: each warp starts rank·64C clocks late, rank its SM warp slot
// / 4 (its place among its SMSP's warps), which spreads the warps' absorbs over a stride instead of starting them in
// lockstep (64C clocks is one warp's INT8 MMAs of a stride). K = -1 is the MMAs alone; K >= NK are controls with the
// absorb's instruction count: FI_IMAD / FI_LOP3 / FI_FADD (4 per tile, chains that never read an accumulator) and
// ACC_LOP3 (one LOP3 per accumulator word into its own register, no chain).
constexpr int FT = 4;
enum Control { FI_IMAD = NK, FI_LOP3, FI_FADD, ACC_LOP3, NF };
static const char* CONTROL_NAME[NF - NK] = {"fill-imad", "fill-lop3", "fill-fadd", "acc-lop3"};
constexpr int NM = 8;  // FUSED's mode index, over MODE = 0, 1, 2, 3, 5, 7, 10, 11
static const char* MODE_NAME[NM] = {"fp8", "fp8pp", "s8", "s8pp", "fp8il", "s8il", "s8st", "s8ppst"};

__device__ __forceinline__ void mma_ip(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_dc(float (&d)[4], const float (&c)[4], const uint32_t (&a)[4],
                                       const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void mma_ip(uint32_t (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_dc(uint32_t (&d)[4], const uint32_t (&c)[4], const uint32_t (&a)[4],
                                       const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
__device__ __forceinline__ uint32_t bits(float x) { return __float_as_uint(x); }
__device__ __forceinline__ uint32_t bits(uint32_t x) { return x; }

template <int K, typename T>
__device__ __forceinline__ void work(St (&s)[FT], uint32_t (&xw)[FT][4], float (&ff)[FT], const T (&acc)[FT][4]) {
  if constexpr (K >= 0 && K < NK) {
#pragma unroll
    for (int j = 0; j < FT; ++j) {
      const uint32_t wd[4] = {bits(acc[j][0]), bits(acc[j][1]), bits(acc[j][2]), bits(acc[j][3])};
      if (j == 0) absorb<K, 4, 0>(s[j], wd);
      if (j == 1) absorb<K, 4, 4>(s[j], wd);
      if (j == 2) absorb<K, 4, 8>(s[j], wd);
      if (j == 3) absorb<K, 4, 12>(s[j], wd);
    }
  } else if constexpr (K == FI_IMAD || K == FI_LOP3 || K == FI_FADD) {
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int j = 0; j < FT; ++j) {
        if constexpr (K == FI_IMAD) s[j].y = s[j].y * KA[4 * j + r] + s[j ^ 1].y;
        if constexpr (K == FI_LOP3) s[j].y = maj3(s[j].y, s[j ^ 1].y, KK[4 * j + r]);
        if constexpr (K == FI_FADD) ff[j] = ff[j] + ff[j ^ 1];
      }
  } else if constexpr (K == ACC_LOP3) {
#pragma unroll
    for (int j = 0; j < FT; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) xw[j][r] = maj3(xw[j][r], bits(acc[j][r]), KK[4 * j + r]);
  }
}

// MODE bit 2 (with ping-pong): one stride's MMAs into d, the first reading C from c0, with the previous stride's words
// (c0's) absorbed a slice at a time between the MMAs instead of all at once after the last. Word W is tile (W/2) % FT's
// word 2 (W/2/FT) + W%2, so a pair stays in one tile and the chains' words alternate. ptxas regroups independent
// instructions into bursts, so each MMA's B operand reads the previous slice's last result (`dep`) through an opaque
// zero (b ^ (dep & z0), one LOP3, which the bare loop's MMAs chain through too): the MMA can't issue before the slice.
template <int K, int W, typename T>
__device__ __forceinline__ uint32_t il_word(St (&s)[FT], uint32_t (&xw)[FT][4], float (&ff)[FT],
                                            const T (&acc)[FT][4], uint32_t dep) {
  constexpr int j = (W >> 1) % FT, r = 2 * ((W >> 1) / FT) + (W & 1);
  if constexpr (K >= 0 && K < NK) {
    if constexpr (!paired(K)) {
      const uint32_t wd[1] = {bits(acc[j][r])};
      absorb<K, 1, 4 * j + r>(s[j], wd);
      return K == DUO ? s[j].z : s[j].y;
    } else if constexpr ((W & 1) != 0) {
      const uint32_t wd[2] = {bits(acc[j][r - 1]), bits(acc[j][r])};
      absorb<K, 2, 4 * j + r - 1>(s[j], wd);
      return s[j].y;
    }
  } else if constexpr (K == FI_IMAD) {
    return s[j].y = s[j].y * KA[4 * j + r] + s[j ^ 1].y;
  } else if constexpr (K == FI_LOP3) {
    return s[j].y = maj3(s[j].y, s[j ^ 1].y, KK[4 * j + r]);
  } else if constexpr (K == FI_FADD) {
    ff[j] = ff[j] + ff[j ^ 1];
    return __float_as_uint(ff[j]);
  } else if constexpr (K == ACC_LOP3) {
    return xw[j][r] = maj3(xw[j][r], bits(acc[j][r]), KK[4 * j + r]);
  }
  return dep;
}
template <int K, int LO, int HI, typename T>
__device__ __forceinline__ uint32_t il_words(St (&s)[FT], uint32_t (&xw)[FT][4], float (&ff)[FT],
                                             const T (&acc)[FT][4], uint32_t dep) {
  if constexpr (LO < HI) return il_words<K, LO + 1, HI>(s, xw, ff, acc, il_word<K, LO>(s, xw, ff, acc, dep));
  return dep;
}
template <int K, int C, int I, typename T>
__device__ __forceinline__ void il_stride(T (&d)[FT][4], const T (&c0)[FT][4], const uint32_t (&a)[4],
                                          const uint32_t (&b)[2][2], St (&s)[FT], uint32_t (&xw)[FT][4],
                                          float (&ff)[FT], uint32_t& dep, uint32_t z0) {
  if constexpr (I < C * FT) {
    constexpr int j = I % FT, M = C * FT, NW = 4 * FT;
    const uint32_t bb[2] = {b[j & 1][0] ^ (dep & z0), b[j & 1][1]};
    if constexpr (I < FT) mma_dc(d[j], c0[j], a, bb);
    else mma_ip(d[j], a, bb);
    dep = il_words<K, I * NW / M, (I + 1) * NW / M>(s, xw, ff, c0, bb[0]);
    il_stride<K, C, I + 1>(d, c0, a, b, s, xw, ff, dep, z0);
  }
}

template <int K, int C, int MODE>
__global__ void __launch_bounds__(TPB) fused_k(uint32_t* out, int iters, uint32_t seed, long long* clk) {
  using T = std::conditional_t<(MODE & 2) != 0, uint32_t, float>;
  constexpr bool PP = (MODE & 1) != 0, IL = (MODE & 4) != 0;
  static_assert(!IL || PP, "interleaving needs the ping-pong accumulators");
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t ma = (MODE & 2) ? 0x3F3F3F3Fu : 0x07070707u, oa = (MODE & 2) ? 0u : 0x38383838u,
                 ob = (MODE & 2) ? 0u : 0x30303030u;
  uint32_t a[4], b[2][2];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = oa | (gen(seed, tid, 7, i) & ma);
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int k = 0; k < 2; ++k) b[i][k] = ob | (gen(seed, tid, 8 + i, k) & ma);
  T acc[2][FT][4];
  St s[FT];
  uint32_t xw[FT][4];
  float ff[FT];
#pragma unroll
  for (int j = 0; j < FT; ++j) {
    s[j].y = gen(seed, tid, j, 100);
    s[j].z = gen(seed, tid, j, 101);
    ff[j] = __uint_as_float(0x3f800000u | (gen(seed, tid, j, 102) & 0x7fffffu));
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      xw[j][r] = gen(seed, tid, j, 110 + r);
      acc[0][j][r] = T(0);
      acc[1][j][r] = T(0);
    }
  }
  // Opaque to ptxas, so it keeps the operands in registers instead of recomputing gen() inside the timed loop.
#pragma unroll
  for (int i = 0; i < 4; ++i) asm volatile("" : "+r"(a[i]));
#pragma unroll
  for (int i = 0; i < 2; ++i) asm volatile("" : "+r"(b[i][0]), "+r"(b[i][1]));
#pragma unroll
  for (int j = 0; j < FT; ++j) {
    asm volatile("" : "+r"(s[j].y), "+r"(s[j].z), "+f"(ff[j]));
    asm volatile("" : "+r"(xw[j][0]), "+r"(xw[j][1]), "+r"(xw[j][2]), "+r"(xw[j][3]));
  }
  __syncthreads();
  if constexpr ((MODE & 8) != 0) {
    uint32_t slot;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(slot));
    const long long d = (long long)(slot >> 2) * 64 * C, s0 = clock64();
    while (clock64() - s0 < d) {
    }
  }
  const long long t0 = clock64();
  if constexpr (!PP) {
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int c = 0; c < C; ++c)
#pragma unroll
        for (int j = 0; j < FT; ++j) mma_ip(acc[0][j], a, b[j & 1]);
      work<K>(s, xw, ff, acc[0]);
    }
  } else if constexpr (IL) {
    uint32_t z0 = (uint32_t)iters >> 31, dep = 0;
    asm volatile("" : "+r"(z0));
#pragma unroll 1
    for (int it = 0; it < iters; it += 2) {
      il_stride<K, C, 0>(acc[1], acc[0], a, b, s, xw, ff, dep, z0);
      il_stride<K, C, 0>(acc[0], acc[1], a, b, s, xw, ff, dep, z0);
    }
  } else {
#pragma unroll 1
    for (int it = 0; it < iters; it += 2) {
#pragma unroll
      for (int c = 0; c < C; ++c)
#pragma unroll
        for (int j = 0; j < FT; ++j) {
          if (c == 0) mma_dc(acc[1][j], acc[0][j], a, b[j & 1]);
          else mma_ip(acc[1][j], a, b[j & 1]);
        }
      work<K>(s, xw, ff, acc[1]);
#pragma unroll
      for (int c = 0; c < C; ++c)
#pragma unroll
        for (int j = 0; j < FT; ++j) {
          if (c == 0) mma_dc(acc[0][j], acc[1][j], a, b[j & 1]);
          else mma_ip(acc[0][j], a, b[j & 1]);
        }
      work<K>(s, xw, ff, acc[0]);
    }
  }
  const long long t1 = clock64();
  uint32_t x = 0;
#pragma unroll
  for (int j = 0; j < FT; ++j) {
    x ^= (K >= 0 && K < NK ? finish<(K >= 0 && K < NK ? K : 0)>(s[j]) : s[j].y) ^ __float_as_uint(ff[j]);
#pragma unroll
    for (int r = 0; r < 4; ++r) x ^= bits(acc[0][j][r]) ^ bits(acc[1][j][r]) ^ xw[j][r];
  }
  out[tid] = x;
  if (threadIdx.x == 0) {
    clk[2 * blockIdx.x] = t0;
    clk[2 * blockIdx.x + 1] = t1;
  }
}

// ---------------------------------------------------------------------------------------------------------------- host
using KFn = void (*)(uint32_t*, int, uint32_t, long long*);

static KFn PURE[NK] = {pure_k<0>, pure_k<1>, pure_k<2>, pure_k<3>, pure_k<4>, pure_k<5>,
                       pure_k<6>, pure_k<7>, pure_k<8>, pure_k<9>, pure_k<10>, pure_k<11>};
static KFn PROBE[NP] = {probe_k<0>,  probe_k<1>,  probe_k<2>,  probe_k<3>,  probe_k<4>,  probe_k<5>,
                        probe_k<6>,  probe_k<7>,  probe_k<8>,  probe_k<9>,  probe_k<10>, probe_k<11>,
                        probe_k<12>, probe_k<13>, probe_k<14>, probe_k<15>, probe_k<16>, probe_k<17>};
constexpr int NC = 6;  // phi.sh's ncu loop indexes FUSED rows by it
static const int CS[NC] = {1, 2, 4, 8, 16, 32};
#define FROW(K, M) \
  {fused_k<K, 1, M>, fused_k<K, 2, M>, fused_k<K, 4, M>, fused_k<K, 8, M>, fused_k<K, 16, M>, fused_k<K, 32, M>}
#define FMODE(M)                                                                                               \
  {FROW(-1, M), FROW(0, M), FROW(1, M), FROW(2, M),  FROW(3, M),  FROW(4, M),  FROW(5, M),  FROW(6, M), \
   FROW(7, M),  FROW(8, M), FROW(9, M), FROW(10, M), FROW(11, M), FROW(12, M), FROW(13, M), FROW(14, M),  \
   FROW(15, M)}
static_assert(NF == 16, "FMODE lists every fused kind");
static KFn FUSED[NM][NF + 1][NC] = {FMODE(0), FMODE(1), FMODE(2), FMODE(3), FMODE(5), FMODE(7), FMODE(10), FMODE(11)};

static const char* fused_name(int k) { return k < 0 ? "none" : k < NK ? KIND_NAME[k] : CONTROL_NAME[k - NK]; }

static void set_keys(uint32_t kseed) {
  uint32_t ka[NKEY], kb[NKEY], kk[NKEY];
  for (int i = 0; i < NKEY; ++i) {
    ka[i] = gen(kseed, 1, i, 0) | 1u;
    kb[i] = gen(kseed, 2, i, 0) | 1u;
    kk[i] = gen(kseed, 3, i, 0);
  }
  CK(cudaMemcpyToSymbol(KA, ka, sizeof ka));
  CK(cudaMemcpyToSymbol(KB, kb, sizeof kb));
  CK(cudaMemcpyToSymbol(KK, kk, sizeof kk));
}

struct Timing {
  double ms, cycles;
  int blocks, per_sm, sms;
};

// occ > 0 pins occ blocks per SM: each block reserves 1/occ of the SM's shared memory, so kernels whose register counts
// would give them different occupancies run with the same number of warps per SMSP.
static Timing time_kernel(KFn fn, int iters, int reps, uint32_t seed, int occ = 0) {
  int dev = 0, sms = 0, per_sm = 0, smem = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  if (occ > 0) {
    int per_smp = 0, reserved = 0, optin = 0;
    CK(cudaDeviceGetAttribute(&per_smp, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    CK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev));
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    smem = std::min(per_smp / occ - reserved, optin);
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  }
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, TPB, smem));
  if (per_sm < 1 || (occ > 0 && per_sm != occ)) {
    fprintf(stderr, "kernel fits %d blocks per SM, not %d\n", per_sm, occ);
    exit(5);
  }
  const int blocks = sms * per_sm;
  uint32_t* out;
  long long* clk;
  CK(cudaMalloc(&out, (size_t)blocks * TPB * PU * sizeof(uint32_t)));
  CK(cudaMalloc(&clk, (size_t)blocks * 2 * sizeof(long long)));
  std::vector<long long> h(2 * blocks);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  fn<<<blocks, TPB, smem>>>(out, iters / 4 + 1, seed, clk);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<double> ms, cyc;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0));
    fn<<<blocks, TPB, smem>>>(out, iters, seed + r, clk);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float t = 0;
    CK(cudaEventElapsedTime(&t, e0, e1));
    CK(cudaMemcpy(h.data(), clk, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    long long span = 0;
    for (int b = 0; b < blocks; ++b) span = std::max(span, h[2 * b + 1] - h[2 * b]);
    ms.push_back(t);
    cyc.push_back((double)span);
  }
  std::sort(ms.begin(), ms.end());
  std::sort(cyc.begin(), cyc.end());
  CK(cudaFree(out));
  CK(cudaFree(clk));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return {ms[ms.size() / 2], cyc[cyc.size() / 2], blocks, per_sm, sms};
}

static void emit(const char* mode, const char* kind, int c, int iters, const Timing& t, double units, const char* unit) {
  const double threads = (double)t.blocks * TPB;
  const double per_sm_clk = units * threads * iters / ((double)t.sms * t.cycles);
  printf("{\"mode\":\"%s\",\"kind\":\"%s\",\"C\":%d,\"iters\":%d,\"blocks\":%d,\"blocks_per_sm\":%d,\"sms\":%d,"
         "\"threads\":%.0f,\"ms\":%.5f,\"cycles\":%.0f,\"mhz\":%.2f,\"%s_per_thread_iter\":%.0f,\"%s_per_sm_clk\":%.6f}\n",
         mode, kind, c, iters, t.blocks, t.per_sm, t.sms, threads, t.ms, t.cycles, t.cycles / (t.ms * 1e3), unit, units,
         unit, per_sm_clk);
  fflush(stdout);
}

static void gate(uint32_t seed, uint32_t kseed) {
  set_keys(kseed);
  const int threads = 64;
  uint32_t* out;
  long long* clk;
  CK(cudaMalloc(&out, threads * PU * sizeof(uint32_t)));
  CK(cudaMalloc(&clk, 2 * sizeof(long long)));
  std::vector<uint32_t> h(threads * PU);
  for (int k = 0; k < NK; ++k)
    for (int iters : {1, 3}) {
      CK(cudaMemset(out, 0xA5, threads * PU * sizeof(uint32_t)));
      PURE[k]<<<1, threads>>>(out, iters, seed, clk);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(), out, h.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      for (int t = 0; t < threads; ++t)
        for (int u = 0; u < PU; ++u)
          printf("%s %d %u %u %d %d %08x\n", KIND_NAME[k], iters, seed, kseed, t, u, h[t * PU + u]);
    }
  CK(cudaFree(out));
  CK(cudaFree(clk));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: phi gate|pure|probe|fused|one ...\n");
    return 2;
  }
  const char* mode = argv[1];
  const uint32_t seed = 0x5EED1234u, kseed = 0xC0FFEE11u;
  if (!strcmp(mode, "gate")) {
    gate(argc > 2 ? (uint32_t)strtoul(argv[2], 0, 0) : seed, argc > 3 ? (uint32_t)strtoul(argv[3], 0, 0) : kseed);
    return 0;
  }
  set_keys(kseed);
  if (!strcmp(mode, "one")) {
    const char* m = argv[2];
    const int idx = atoi(argv[3]), iters = atoi(argv[4]);
    const int fm = argc > 5 ? atoi(argv[5]) : 0, occ = argc > 6 ? atoi(argv[6]) : 0;
    KFn fn = !strcmp(m, "pure") ? PURE[idx] : !strcmp(m, "probe") ? PROBE[idx] : FUSED[fm][idx / NC][idx % NC];
    time_kernel(fn, iters, 1, seed, occ);
    return 0;
  }
  const int iters = argc > 2 ? atoi(argv[2]) : 20000, reps = argc > 3 ? atoi(argv[3]) : 5,
            occ = argc > 4 ? atoi(argv[4]) : 0;
  if (!strcmp(mode, "pure")) {
    for (int k = 0; k < NK; ++k) emit("pure", KIND_NAME[k], 0, iters, time_kernel(PURE[k], iters, reps, seed), PU * PW, "words");
  } else if (!strcmp(mode, "probe")) {
    for (int p = 0; p < NP; ++p) emit("probe", PROBE_NAME[p], 0, iters, time_kernel(PROBE[p], iters, reps, seed), 8 * PU, "steps");
  } else if (!strcmp(mode, "fused")) {
    for (int m = 0; m < NM; ++m)
      for (int k = -1; k < NF; ++k)
        for (int ci = 0; ci < NC; ++ci) {
          const int c = CS[ci];
          const int it = (iters / c) & ~1;
          emit(MODE_NAME[m], fused_name(k), c, it, time_kernel(FUSED[m][k + 1][ci], it, reps, seed, occ), c * FT,
               "mmas");
        }
  } else {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }
  return 0;
}
