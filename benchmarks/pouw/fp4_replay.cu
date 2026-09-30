// Pearl-C4's word_replay (verity_pouw.schemes.pearl_c4.word_replay) on NVFP4 words: per word (A′ row i, B̃ row j), the
// chain from +0 of `blackwell_sm120_nvf4_m16n8k64` (verity.ml.tc.BlockScaledAlignAdd: frac_bits 27, acc_window 36,
// grid_floor -174, +0 on underflow), its identity atoms (a lone skip that leaves the final word unchanged, or leaves
// the model's range) and its sub-grid blocks (2304·m_a·m_b·2^(e_a+e_b−2) below its step's grid).  Exact integer
// arithmetic only.  Built by nvcc (the GPU: a chain kernel per word, then an identity kernel with one block per word
// and one thread per pair of skipped atoms) or by g++ -x c++ (the same functions on the CPU, OpenMP over words).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD static inline
#endif

typedef long long i64;
typedef unsigned long long u64;

enum { ATOM = 64, NS = 4, FRAC_BITS = 27, ACC_ANCHOR = 35, GRID_FLOOR = -174, NOPART = -128, GROUP_MAX = 2304 };
enum { OK = 0, OVERFLOW = 1, NONFINITE = 2, NAN_SCALE = 3 };

HD int bitlen(u64 x) {
#ifdef __CUDA_ARCH__
  return 64 - __clzll((i64)x);
#else
  return x ? 64 - __builtin_clzll(x) : 0;
#endif
}

HD int numerator(uint8_t code) {
  const int n = (0x0C08060403020100ULL >> (8 * (code & 7))) & 0xFF;  // {0, 1, 2, 3, 4, 6, 8, 12} halves
  return (code & 8) ? -n : n;
}

// A UE4M3 scale byte from its low 7 bits: (8 + m, e − 10), or (m, −9) at exponent field 0.  0x7F is NaN.
HD int scale_dyadic(uint8_t byte, int *m, int *e) {
  const int c = byte & 0x7F, ef = c >> 3, mm = c & 7;
  *m = ef ? 8 + mm : mm;
  *e = ef ? ef - 10 : -9;
  return c == 0x7F ? NAN_SCALE : OK;
}

// One step's four group terms: mant[g] = gd·m_a·m_b at unit weight 2^unit[g]; unit[g] = NOPART when the group does
// not participate (no nonzero product, or a zero scale mantissa).  A nonzero mant always participates.
HD int group_terms(const uint8_t *a, const uint8_t *b, const uint8_t *sa, const uint8_t *sb, int *mant,
                   signed char *unit) {
  int st = OK;
  for (int g = 0; g < NS; g++) {
    int gd = 0, any = 0;
    for (int i = g * 16; i < g * 16 + 16; i++) {
      const int na = numerator(a[i]), nb = numerator(b[i]);
      gd += na * nb;
      any |= (na != 0) & (nb != 0);
    }
    int ma, ea, mb, eb;
    st |= scale_dyadic(sa[g], &ma, &ea) | scale_dyadic(sb[g], &mb, &eb);
    mant[g] = gd * ma * mb;
    unit[g] = (any && ma && mb) ? (signed char)(-2 + ea + eb) : (signed char)NOPART;
  }
  return st;
}

// The accumulator word as (signed mantissa, exponent, exponent field).
HD int fp32_dyadic(uint32_t w, i64 *m, int *e, int *ef) {
  *ef = (w >> 23) & 0xFF;
  const i64 frac = w & 0x7FFFFF;
  *m = *ef ? (frac | 0x800000) : frac;
  *e = *ef ? *ef - 150 : -149;
  if (w >> 31) *m = -*m;
  return *ef == 0xFF ? NONFINITE : OK;
}

// BlockScaledAlignAdd.grid: the common grid exponent, or *none when nothing participates and the accumulator is zero.
HD int grid_lsb(i64 am, int ef, const signed char *unit, int *none) {
  int have = 0, lsb = -(1 << 20);
  for (int g = 0; g < NS; g++)
    if (unit[g] != NOPART) {
      const int v = unit[g] - FRAC_BITS;
      lsb = v > lsb ? v : lsb;
      have = 1;
    }
  if (am) {
    const int v = (ef ? ef : 1) - 127 - ACC_ANCHOR;
    lsb = v > lsb ? v : lsb;
    have = 1;
  }
  *none = !have;
  return lsb > GRID_FLOOR ? lsb : GRID_FLOOR;
}

HD i64 trunc_rz(i64 m, int e, int lsb) {
  const int sh = e - lsb;
  if (sh >= 0) return m * ((i64)1 << sh);         // at most 27 (a term) or 12 (the accumulator) on the grid's rule
  if (sh < -62) return 0;
  const i64 q = (m < 0 ? -m : m) >> -sh;
  return m < 0 ? -q : q;
}

// dyadic.rs::to_f32(Rz) of the nonzero n·2^lsb; +0 below the subnormal grid; OVERFLOW past binary32.
HD int to_f32_rz(i64 n, int lsb, uint32_t *out) {
  const int neg = n < 0;
  const u64 mag = neg ? (u64)(-n) : (u64)n;
  const int e_val = bitlen(mag) - 1 + lsb;
  const int grid = e_val >= -126 ? e_val - 23 : -149;
  const int shift = grid - lsb;
  const u64 q = shift > 0 ? (shift > 63 ? 0 : mag >> shift) : mag << -shift;
  if (!q) {
    *out = 0;
    return OK;
  }
  const int e = bitlen(q) - 1 + grid;
  if (e > 127) return OVERFLOW;
  const uint32_t bits = e >= -126 ? (uint32_t)(((u64)(e + 127) << 23) | (q & 0x7FFFFF)) : (uint32_t)q;
  *out = bits | (neg ? 0x80000000u : 0u);
  return OK;
}

// One firing on its group terms (BlockScaledAlignAdd.align_add).
HD int align_add(uint32_t c, const int *mant, const signed char *unit, uint32_t *out) {
  i64 am;
  int ae, ef, none;
  const int st = fp32_dyadic(c, &am, &ae, &ef);
  if (st) return st;
  const int lsb = grid_lsb(am, ef, unit, &none);
  if (none) {
    *out = 0;
    return OK;
  }
  i64 total = 0;
  for (int g = 0; g < NS; g++)
    if (mant[g]) total += trunc_rz(mant[g], unit[g], lsb);
  if (am) total += trunc_rz(am, ae, lsb);
  if (!total) {
    *out = 0;
    return OK;
  }
  return to_f32_rz(total, lsb, out);
}

// Sub-grid blocks of one step, from the accumulator before it.
HD int subgrid_blocks(uint32_t c, const signed char *unit, const uint8_t *sa, const uint8_t *sb) {
  i64 am;
  int ae, ef, none;
  fp32_dyadic(c, &am, &ae, &ef);
  const int lsb = grid_lsb(am, ef, unit, &none);
  if (none) return 0;
  int n = 0;
  for (int g = 0; g < NS; g++) {
    int ma, ea, mb, eb;
    scale_dyadic(sa[g], &ma, &ea);
    scale_dyadic(sb[g], &mb, &eb);
    const i64 top = (i64)GROUP_MAX * ma * mb;
    const int e = lsb - ea - eb + 2;
    n += top == 0 || (e > 0 && (e >= 40 || top < ((i64)1 << e)));
  }
  return n;
}

// Re-chain from start t with its term skipped: 1 when it re-converges on the chain (or leaves the range), else 0.
HD int identity_at(const uint32_t *w, const int *mant, const signed char *unit, int steps, int t) {
  uint32_t c = w[t];
  int s = t + 1;
  while (s < steps && c != w[s]) {
    if (align_add(c, mant + NS * s, unit + NS * s, &c)) return 1;
    s++;
  }
  return c == w[s];
}

#ifdef __CUDACC__

__global__ void k_chain(const uint8_t *ac, const uint8_t *as, const uint8_t *bc, const uint8_t *bs, int R, int C, int K,
                        uint32_t *w, int *sub, int *err) {
  const int word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word >= R * C) return;
  const int i = word / C, j = word % C, steps = K / ATOM;
  const uint8_t *a = ac + (size_t)i * K, *b = bc + (size_t)j * K, *sa = as + (size_t)i * (K / 16),
                *sb = bs + (size_t)j * (K / 16);
  uint32_t *ww = w + (size_t)word * (steps + 1);
  uint32_t c = 0;
  int n = 0, st = OK;
  ww[0] = 0;
  for (int s = 0; s < steps; s++) {
    int mant[NS];
    signed char unit[NS];
    st |= group_terms(a + s * ATOM, b + s * ATOM, sa + s * NS, sb + s * NS, mant, unit);
    n += subgrid_blocks(c, unit, sa + s * NS, sb + s * NS);
    st |= align_add(c, mant, unit, &c);
    ww[s + 1] = c;
  }
  sub[word] = n;
  err[word] = st;
}

__global__ void k_identity(const uint8_t *ac, const uint8_t *as, const uint8_t *bc, const uint8_t *bs, int C, int K,
                           const uint32_t *w, int *ident) {
  extern __shared__ uint32_t smem[];
  const int word = blockIdx.x, i = word / C, j = word % C, steps = K / ATOM;
  uint32_t *sw = smem;                                       // steps + 1
  int *mant = (int *)(sw + steps + 1);                       // NS·steps
  signed char *unit = (signed char *)(mant + NS * steps);    // NS·steps
  __shared__ int count;
  const uint8_t *a = ac + (size_t)i * K, *b = bc + (size_t)j * K, *sa = as + (size_t)i * (K / 16),
                *sb = bs + (size_t)j * (K / 16);
  if (threadIdx.x == 0) count = 0;
  for (int s = threadIdx.x; s <= steps; s += blockDim.x) sw[s] = w[(size_t)word * (steps + 1) + s];
  for (int s = threadIdx.x; s < steps; s += blockDim.x)
    group_terms(a + s * ATOM, b + s * ATOM, sa + s * NS, sb + s * NS, mant + NS * s, unit + NS * s);
  __syncthreads();
  int n = 0;
  for (int p = threadIdx.x; p < (steps + 1) / 2; p += blockDim.x) {   // starts p and steps − 1 − p: steps − 1 firings
    n += identity_at(sw, mant, unit, steps, p);
    if (steps - 1 - p != p) n += identity_at(sw, mant, unit, steps, steps - 1 - p);
  }
  atomicAdd(&count, n);
  __syncthreads();
  if (threadIdx.x == 0) ident[word] = count;
}

extern "C" int fp4_replay(const uint8_t *ac, const uint8_t *as, int R, const uint8_t *bc, const uint8_t *bs, int C, int K,
                          int *ident, int *sub, int *err) {
  const int steps = K / ATOM, words = R * C;
  const size_t na = (size_t)R * K, nb = (size_t)C * K, nsa = na / 16, nsb = nb / 16;
  uint8_t *d_ac, *d_as, *d_bc, *d_bs;
  uint32_t *d_w;
  int *d_ident, *d_sub, *d_err;
  cudaError_t e = cudaSuccess;
#define CK(x) do { e = (x); if (e != cudaSuccess) return (int)e; } while (0)
  CK(cudaMalloc(&d_ac, na));
  CK(cudaMalloc(&d_as, nsa));
  CK(cudaMalloc(&d_bc, nb));
  CK(cudaMalloc(&d_bs, nsb));
  CK(cudaMalloc(&d_w, (size_t)words * (steps + 1) * 4));
  CK(cudaMalloc(&d_ident, (size_t)words * 4));
  CK(cudaMalloc(&d_sub, (size_t)words * 4));
  CK(cudaMalloc(&d_err, (size_t)words * 4));
  CK(cudaMemcpy(d_ac, ac, na, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_as, as, nsa, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bc, bc, nb, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bs, bs, nsb, cudaMemcpyHostToDevice));
  k_chain<<<(words + 127) / 128, 128>>>(d_ac, d_as, d_bc, d_bs, R, C, K, d_w, d_sub, d_err);
  CK(cudaGetLastError());
  int threads = (steps + 1) / 2;
  threads = threads < 32 ? 32 : threads > 256 ? 256 : threads;
  const size_t shm = (size_t)(steps + 1) * 4 + (size_t)NS * steps * 5;
  if (shm > 48 * 1024) CK(cudaFuncSetAttribute(k_identity, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  k_identity<<<words, threads, shm>>>(d_ac, d_as, d_bc, d_bs, C, K, d_w, d_ident);
  CK(cudaGetLastError());
  CK(cudaMemcpy(ident, d_ident, (size_t)words * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sub, d_sub, (size_t)words * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(err, d_err, (size_t)words * 4, cudaMemcpyDeviceToHost));
  cudaFree(d_ac); cudaFree(d_as); cudaFree(d_bc); cudaFree(d_bs);
  cudaFree(d_w); cudaFree(d_ident); cudaFree(d_sub); cudaFree(d_err);
#undef CK
  return 0;
}

#else

extern "C" int fp4_replay(const uint8_t *ac, const uint8_t *as, int R, const uint8_t *bc, const uint8_t *bs, int C, int K,
                          int *ident, int *sub, int *err) {
  const int steps = K / ATOM;
#pragma omp parallel
  {
    int *mant = (int *)malloc(sizeof(int) * NS * steps);
    signed char *unit = (signed char *)malloc(NS * steps);
    uint32_t *w = (uint32_t *)malloc(4 * (size_t)(steps + 1));
#pragma omp for schedule(dynamic, 16)
    for (int word = 0; word < R * C; word++) {
      const int i = word / C, j = word % C;
      const uint8_t *a = ac + (size_t)i * K, *b = bc + (size_t)j * K, *sa = as + (size_t)i * (K / 16),
                    *sb = bs + (size_t)j * (K / 16);
      int st = OK, n = 0, idn = 0;
      w[0] = 0;
      for (int s = 0; s < steps; s++) {
        st |= group_terms(a + s * ATOM, b + s * ATOM, sa + s * NS, sb + s * NS, mant + NS * s, unit + NS * s);
        n += subgrid_blocks(w[s], unit + NS * s, sa + s * NS, sb + s * NS);
        st |= align_add(w[s], mant + NS * s, unit + NS * s, &w[s + 1]);
      }
      if (!st)
        for (int t = 0; t < steps; t++) idn += identity_at(w, mant, unit, steps, t);
      ident[word] = idn;
      sub[word] = n;
      err[word] = st;
    }
    free(mant);
    free(unit);
    free(w);
  }
  return 0;
}

#endif
