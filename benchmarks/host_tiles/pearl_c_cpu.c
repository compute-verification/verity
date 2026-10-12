// Pearl-C's C̃ chain on the host CPU, exact under sm_120 v1 (`pearl_c_u.chain` at `pearl_c_device.SM120`): per window of
// 128 products, local = +0, then four BLACKWELL_SM120_E4M3_M16N8K32 atoms (acc + 32 E4M3 products, one group, width 26,
// floor -133, truncating alignment), then total = RN_f32(total + local).  The k-slice accumulators are every atom's local
// (k_s = 32) and every promotion's total (k_s = 128).
//
//   gate:  pearl_c_cpu gate DIR [scalar]       DIR/{shape,a,b}.bin -> DIR/{atoms,totals}.bin, DIR/stats.json
//   bench: pearl_c_cpu bench M K N W KS REPS DATA   DATA = gauss | uniform | file:PATH (codes to tile); OpenMP threads
//          env PEARL_PREFETCH: bytes ahead to prefetch B̃ (run.sh: 4096); PEARL_FUSE=0: m <= 2 decodes into the block too
//
// The atom in FP64.  Let M be the atom's top exponent (the largest of the accumulator's and the nonzero products'), so the
// alignment grid is g = 2^(M-25).  A product sa*sb*2^(ea'+eb'-20) with sa*sb < 2^8 lies on g whenever its exponent
// e = ea'+eb'-14 is at least M - 19, and the accumulator's alignment is a truncation of its word to g (clear its low
// M - e_acc - 2 bits).  When every product lies on g the atom is RZ_f32(acc_g + sum of products), and that sum is exact in
// FP64 (every partial sum is a multiple of g below 2^34 g): FMAs, then one round-toward-zero conversion.  M is exact from
// int8 lanes (max over l of ea'_il + eb'_jl, a zero code at -100), and so is "every product lies on g": the min over l of
// ea'_il + eb'_jl (a zero code at +100) plus 5 is at least M.  A lane that fails, where some product does lose bits to the
// alignment, takes `atom_exact`, the integer statement of `pearl_c_u.grouped`.  MXCSR keeps its defaults (no FTZ, no DAZ): RN for the promotion, subnormals kept.
//
// B̃ is laid out per 64-column block, [jb][k][64] codes (a per-epoch layout choice of the prover).  Loop order: column block,
// atom (its B̃ block decoded once to FP64 and exponents, 18 KiB, in L1), row pairs.
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ZERO_EXP (-100)
#define NONE (-1000)

// ---- the exact scalar atom and chain ---------------------------------------------------------------------------------

static int bad_input;

static uint32_t atom_exact(uint32_t acc, const uint8_t *a, const uint8_t *b, int bstride) {
  int neg[33], e[33], nt = 0;
  int64_t m[33];
  uint32_t field = (acc >> 23) & 0xFF, man = acc & 0x7FFFFF;
  if (field == 0xFF) bad_input = 1;
  int64_t am = field ? (man | 1u << 23) : man;
  if (am) { neg[nt] = acc >> 31; e[nt] = field ? (int)field - 127 : -126; m[nt] = am; nt++; }
  for (int l = 0; l < 32; l++) {
    uint8_t x = a[l], y = b[(size_t)l * bstride];
    if ((x & 0x7F) == 0x7F || (y & 0x7F) == 0x7F) bad_input = 1;
    int ex = (x >> 3) & 15, ey = (y >> 3) & 15, sx = (x & 7) | (ex ? 8 : 0), sy = (y & 7) | (ey ? 8 : 0);
    if (!sx || !sy) continue;
    neg[nt] = ((x ^ y) >> 7) & 1;
    e[nt] = (ex > 1 ? ex : 1) + (ey > 1 ? ey : 1) - 14;
    m[nt] = (int64_t)(sx * sy) << 17;
    nt++;
  }
  int top = -133;
  for (int t = 0; t < nt; t++) top = e[t] > top ? e[t] : top;
  int64_t total = 0;
  for (int t = 0; t < nt; t++) {
    int sh = top - e[t];
    int64_t s = sh >= 62 ? 0 : (m[t] << 2) >> sh;
    total += neg[t] ? -s : s;
  }
  if (!total) return 0;
  uint32_t ng = total < 0;
  uint64_t mag = ng ? (uint64_t)-total : (uint64_t)total;
  int bits = 64 - __builtin_clzll(mag), ex = top + bits - 26;
  mag = bits > 26 ? mag >> (bits - 26) : mag << (26 - bits);
  if (ex < -126) { int s = -126 - ex; mag = s >= 64 ? 0 : mag >> s; ex = -126; }
  mag >>= 2;
  if (!mag) return 0;
  if (ex > 127) { bad_input = 1; return ng << 31 | 0x7F800000; }
  uint32_t fld = (uint32_t)(ex + 127 - ((mag >> 23) & 1 ? 0 : 1));
  return ng << 31 | fld << 23 | (uint32_t)(mag & 0x7FFFFF);
}

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static uint32_t promote(uint32_t total, uint32_t local) {
  uint32_t t = f2u(u2f(total) + u2f(local));
  if (((t >> 23) & 0xFF) == 0xFF) bad_input = 1;
  return t == 0x80000000u ? 0 : t;
}

// One output's chain: a and b are k codes (b with stride bstride); atoms[t] and totals[w] get the trace.
static void chain_exact(const uint8_t *a, const uint8_t *b, int bstride, int k, uint32_t *atoms, uint32_t *totals) {
  uint32_t total = 0;
  for (int w = 0; w < k / 128; w++) {
    uint32_t local = 0;
    for (int s = 0; s < 4; s++) {
      int t = 4 * w + s;
      local = atom_exact(local, a + 32 * t, b + (size_t)32 * t * bstride, bstride);
      if (((local >> 23) & 0xFF) == 0xFF) bad_input = 1;
      atoms[t] = local;
    }
    total = promote(total, local);
    totals[w] = total;
  }
}

// ---- the vector kernel -----------------------------------------------------------------------------------------------

static double e4m3_value(uint8_t c) {
  int ex = (c >> 3) & 15, mx = c & 7;
  double v = ex ? ldexp(8 + mx, ex - 10) : ldexp(mx, -9);
  return c & 0x80 ? -v : v;
}

static int8_t e4m3_exp(uint8_t c, int8_t zero) {
  if (!(c & 0x7F)) return zero;
  int ex = (c >> 3) & 15;
  return (int8_t)(ex > 1 ? ex : 1);
}

typedef struct {  // af, ae, an are atom-major, [k / 32][m][32], so each atom's rows stream in order
  int m, k;
  double *af;      // value * 256 (B̃ decodes to value / 256 through FP16)
  uint32_t *ae;    // ea' (ZERO_EXP for a zero code) in all four bytes
  uint32_t *an;    // ea' (+100 for a zero code) in all four bytes
  const uint8_t *codes;  // [m][k]
} Arows;

static void prep_a(Arows *A, const uint8_t *codes, int m, int k) {
  A->m = m; A->k = k; A->codes = codes;
  A->af = aligned_alloc(64, (size_t)m * k * 8);
  A->ae = aligned_alloc(64, (size_t)m * k * 4);
  A->an = aligned_alloc(64, (size_t)m * k * 4);
  for (int i = 0; i < m; i++)
    for (int l = 0; l < k; l++) {
      uint8_t c = codes[(size_t)i * k + l];
      size_t x = ((size_t)(l / 32) * m + i) * 32 + l % 32;
      A->af[x] = e4m3_value(c) * 256.0;
      A->ae[x] = (uint8_t)e4m3_exp(c, ZERO_EXP) * 0x01010101u;
      A->an[x] = (uint8_t)e4m3_exp(c, 100) * 0x01010101u;
    }
}

static void free_a(Arows *A) { free(A->af); free(A->ae); free(A->an); }

typedef struct {
  double bf[32 * 64] __attribute__((aligned(64)));
  int8_t be[32 * 64] __attribute__((aligned(64)));
  int8_t bn[32 * 64] __attribute__((aligned(64)));
} Bblock;

static __m512i T_EXP0, T_EXP1, T_MIN0, T_MIN1;

static void init_tables(void) {
  int8_t e[128], n[128];
  for (int c = 0; c < 128; c++) { e[c] = e4m3_exp((uint8_t)c, ZERO_EXP); n[c] = e4m3_exp((uint8_t)c, 100); }
  T_EXP0 = _mm512_loadu_si512(e); T_EXP1 = _mm512_loadu_si512(e + 64);
  T_MIN0 = _mm512_loadu_si512(n); T_MIN1 = _mm512_loadu_si512(n + 64);
}

static long prefetch_dist;  // PEARL_PREFETCH: bytes ahead of the block being decoded (0: none)
static int fuse = 1;        // PEARL_FUSE=0: m <= 2 decodes into the block as larger m does

// One atom's B̃ block: 32 rows l of 64 codes -> FP64 (value / 256 via FP16) and the two int8 exponent forms.  The FP16
// bits of code c are (sign-extended c << 7) with bit 14 cleared: sign at 15, the code's exponent and mantissa at 13..7.
static void decode_b(const uint8_t *blk, Bblock *B) {
  __m512i keep = _mm512_set1_epi16((short)0xBFFF);
  for (int l = 0; l < 32; l++) {
    if (prefetch_dist) _mm_prefetch((const char *)blk + 64 * l + prefetch_dist, _MM_HINT_T0);
    __m512i c = _mm512_loadu_si512(blk + 64 * l);
    _mm512_store_si512(B->be + 64 * l, _mm512_permutex2var_epi8(T_EXP0, c, T_EXP1));
    _mm512_store_si512(B->bn + 64 * l, _mm512_permutex2var_epi8(T_MIN0, c, T_MIN1));
    for (int h = 0; h < 2; h++) {
      __m512i w = _mm512_cvtepi8_epi16(h ? _mm512_extracti64x4_epi64(c, 1) : _mm512_castsi512_si256(c));
      __m512i f = _mm512_and_si512(_mm512_slli_epi16(w, 7), keep);
      double *d = B->bf + 64 * l + 32 * h;
      _mm512_store_pd(d + 0, _mm512_cvtph_pd(_mm_castsi128_ph(_mm512_extracti32x4_epi32(f, 0))));
      _mm512_store_pd(d + 8, _mm512_cvtph_pd(_mm_castsi128_ph(_mm512_extracti32x4_epi32(f, 1))));
      _mm512_store_pd(d + 16, _mm512_cvtph_pd(_mm_castsi128_ph(_mm512_extracti32x4_epi32(f, 2))));
      _mm512_store_pd(d + 24, _mm512_cvtph_pd(_mm_castsi128_ph(_mm512_extracti32x4_epi32(f, 3))));
    }
  }
}

// R rows (i0 ..) of one atom t against one decoded block: local[i][64] in, out.  Returns the number of slow lanes.  Fused
// (m <= 2): B is decoded from the codes bblk into registers as it is used, and B is unused.
static inline __attribute__((always_inline)) int rows_atom(const int R, const int fused, const Arows *A, int i0, int t,
                                                           const Bblock *B, const uint8_t *bblk, uint32_t *local) {
  const int k = A->k;
  __m512d acc[2][8];
  __m512i mx[2], mn[2];
  for (int r = 0; r < R; r++) {
    for (int v = 0; v < 8; v++) acc[r][v] = _mm512_setzero_pd();
    mx[r] = _mm512_set1_epi8(-128);
    mn[r] = _mm512_set1_epi8(127);
  }
  const size_t at = ((size_t)t * A->m + i0) * 32;
  const double *a0 = A->af + at, *a1 = a0 + 32;
  const uint32_t *e0 = A->ae + at, *e1 = e0 + 32;
  const uint32_t *n0 = A->an + at, *n1 = n0 + 32;
  const __m512i keep = _mm512_set1_epi16((short)0xBFFF);
  for (int l = 0; l < 32; l++) {
    __m512d av0 = _mm512_set1_pd(a0[l]), av1 = R > 1 ? _mm512_set1_pd(a1[l]) : av0;
    __m512d bv[8];
    __m512i bx, bn;
    if (fused) {
      if (prefetch_dist) _mm_prefetch((const char *)bblk + 64 * l + prefetch_dist, _MM_HINT_T0);
      __m512i c = _mm512_loadu_si512(bblk + 64 * l);
      bx = _mm512_permutex2var_epi8(T_EXP0, c, T_EXP1);
      bn = _mm512_permutex2var_epi8(T_MIN0, c, T_MIN1);
      for (int h = 0; h < 2; h++) {
        __m512i w = _mm512_cvtepi8_epi16(h ? _mm512_extracti64x4_epi64(c, 1) : _mm512_castsi512_si256(c));
        __m512i f = _mm512_and_si512(_mm512_slli_epi16(w, 7), keep);
        bv[4 * h + 0] = _mm512_cvtph_pd(_mm_castsi128_ph(_mm512_extracti32x4_epi32(f, 0)));
        bv[4 * h + 1] = _mm512_cvtph_pd(_mm_castsi128_ph(_mm512_extracti32x4_epi32(f, 1)));
        bv[4 * h + 2] = _mm512_cvtph_pd(_mm_castsi128_ph(_mm512_extracti32x4_epi32(f, 2)));
        bv[4 * h + 3] = _mm512_cvtph_pd(_mm_castsi128_ph(_mm512_extracti32x4_epi32(f, 3)));
      }
    } else {
      for (int v = 0; v < 8; v++) bv[v] = _mm512_load_pd(B->bf + 64 * l + 8 * v);
      bx = _mm512_load_si512(B->be + 64 * l);
      bn = _mm512_load_si512(B->bn + 64 * l);
    }
    for (int v = 0; v < 8; v++) {
      acc[0][v] = _mm512_fmadd_pd(av0, bv[v], acc[0][v]);
      if (R > 1) acc[1][v] = _mm512_fmadd_pd(av1, bv[v], acc[1][v]);
    }
    mx[0] = _mm512_max_epi8(mx[0], _mm512_adds_epi8(_mm512_set1_epi32((int)e0[l]), bx));
    if (R > 1) mx[1] = _mm512_max_epi8(mx[1], _mm512_adds_epi8(_mm512_set1_epi32((int)e1[l]), bx));
    mn[0] = _mm512_min_epi8(mn[0], _mm512_adds_epi8(_mm512_set1_epi32((int)n0[l]), bn));
    if (R > 1) mn[1] = _mm512_min_epi8(mn[1], _mm512_adds_epi8(_mm512_set1_epi32((int)n1[l]), bn));
  }
  int slow = 0;
  const __m512i none = _mm512_set1_epi32(NONE), ff = _mm512_set1_epi32(0xFF), mabs = _mm512_set1_epi32(0x7FFFFFFF),
                one = _mm512_set1_epi32(1), c127 = _mm512_set1_epi32(127), ones = _mm512_set1_epi32(-1),
                c24 = _mm512_set1_epi32(24), c32 = _mm512_set1_epi32(32), negz = _mm512_set1_epi32((int)0x80000000u);
  for (int r = 0; r < R; r++) {
    int i = i0 + r;
    uint32_t *lw = local + (size_t)i * 64;
    uint32_t res[64] __attribute__((aligned(64)));
    uint64_t fail = 0;
    for (int q = 0; q < 4; q++) {
      __m512i w = _mm512_load_si512(lw + 16 * q);
      __m512i fld = _mm512_and_si512(_mm512_srli_epi32(w, 23), ff);
      __m512i eacc = _mm512_sub_epi32(_mm512_max_epi32(fld, one), c127);
      eacc = _mm512_mask_mov_epi32(none, _mm512_test_epi32_mask(w, mabs), eacc);
      __m512i s = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(mx[r], q));
      __mmask16 hasp = _mm512_cmpge_epi32_mask(s, _mm512_set1_epi32(2));
      __m512i M = _mm512_max_epi32(_mm512_mask_mov_epi32(none, hasp, _mm512_sub_epi32(s, _mm512_set1_epi32(14))), eacc);
      __m512i smin = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(mn[r], q));
      __mmask16 f = _mm512_mask_cmplt_epi32_mask(hasp, _mm512_add_epi32(smin, _mm512_set1_epi32(5)), M);
      __m512i c = _mm512_sub_epi32(_mm512_sub_epi32(M, eacc), _mm512_set1_epi32(2));
      __m512i cc = _mm512_max_epi32(c, _mm512_setzero_si512());
      cc = _mm512_mask_mov_epi32(cc, _mm512_cmpge_epi32_mask(c, c24), c32);
      __m512i at = _mm512_and_si512(w, _mm512_sllv_epi32(ones, cc));
      __m512 atf = _mm512_castsi512_ps(at);
      __m512d s0 = _mm512_add_pd(acc[r][2 * q], _mm512_cvtps_pd(_mm512_castps512_ps256(atf)));
      __m512d s1 = _mm512_add_pd(acc[r][2 * q + 1],
                                 _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(atf), 1))));
      __m256 r0 = _mm512_cvt_roundpd_ps(s0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      __m256 r1 = _mm512_cvt_roundpd_ps(s1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      __m512i rv = _mm512_castps_si512(_mm512_insertf32x8(_mm512_castps256_ps512(r0), r1, 1));
      rv = _mm512_mask_mov_epi32(rv, _mm512_cmpeq_epi32_mask(rv, negz), _mm512_setzero_si512());
      _mm512_store_si512(res + 16 * q, rv);
      fail |= (uint64_t)f << (16 * q);
    }
    while (fail) {
      int j = __builtin_ctzll(fail);
      fail &= fail - 1;
      res[j] = atom_exact(lw[j], A->codes + (size_t)i * k + 32 * t, bblk + j, 64);
      slow++;
    }
    memcpy(lw, res, sizeof res);
  }
  return slow;
}

// One 64-column block of one weight (codes [k][64]) against all of A.  local, total: [m][64] scratch.  When atoms_out /
// totals_out are given (gate) the trace lands there, [t][m][ld] at column offset col; else each snapshot is written to
// snap (the k-slice accumulators a receipt binds: KS = 32 every atom, 128 every total, 0 none).
static long column_block(const Arows *A, const uint8_t *blk, Bblock *B, uint32_t *local, uint32_t *total, int ks,
                         uint32_t *atoms_out, uint32_t *totals_out, int ld, int col, uint32_t *snap) {
  int m = A->m, k = A->k;
  long slow = 0;
  memset(total, 0, (size_t)m * 64 * 4);
  for (int t = 0; t < k / 32; t++) {
    if (t % 4 == 0) memset(local, 0, (size_t)m * 64 * 4);
    const uint8_t *bt = blk + (size_t)32 * t * 64;
    if (m == 1 && fuse)
      slow += rows_atom(1, 1, A, 0, t, B, bt, local);
    else if (m == 2 && fuse)
      slow += rows_atom(2, 1, A, 0, t, B, bt, local);
    else {
      decode_b(bt, B);
      int i = 0;
      for (; i + 2 <= m; i += 2) slow += rows_atom(2, 0, A, i, t, B, bt, local);
      if (i < m) slow += rows_atom(1, 0, A, i, t, B, bt, local);
    }
    if (atoms_out)
      for (int r = 0; r < m; r++) memcpy(atoms_out + ((size_t)t * m + r) * ld + col, local + r * 64, 256);
    else if (ks == 32)
      for (int r = 0; r < m; r++) memcpy(snap + (r & 1) * 64, local + r * 64, 256);
    if (t % 4 == 3) {
      for (int r = 0; r < m; r++)
        for (int q = 0; q < 4; q++) {
          __m512 tv = _mm512_add_ps(_mm512_load_ps((float *)total + r * 64 + 16 * q),
                                    _mm512_load_ps((float *)local + r * 64 + 16 * q));
          __m512i tw = _mm512_castps_si512(tv);
          tw = _mm512_mask_mov_epi32(tw, _mm512_cmpeq_epi32_mask(tw, _mm512_set1_epi32((int)0x80000000u)),
                                     _mm512_setzero_si512());
          _mm512_store_si512(total + r * 64 + 16 * q, tw);
        }
      if (totals_out)
        for (int r = 0; r < m; r++) memcpy(totals_out + ((size_t)(t / 4) * m + r) * ld + col, total + r * 64, 256);
      else if (ks == 128)
        for (int r = 0; r < m; r++) memcpy(snap + (r & 1) * 64, total + r * 64, 256);
    }
  }
  return slow;
}

// B̃ given as n rows of k codes -> [jb][k][64]
static uint8_t *block_b(const uint8_t *b, int n, int k) {
  uint8_t *o = aligned_alloc(64, (size_t)n * k);
  for (int jb = 0; jb < n / 64; jb++)
    for (int l = 0; l < k; l++)
      for (int j = 0; j < 64; j++) o[((size_t)jb * k + l) * 64 + j] = b[(size_t)(jb * 64 + j) * k + l];
  return o;
}

// ---- gate ------------------------------------------------------------------------------------------------------------

static void *rd(const char *dir, const char *name, size_t *len) {
  char p[4096];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE *f = fopen(p, "rb");
  if (!f) { perror(p); exit(2); }
  fseek(f, 0, SEEK_END);
  *len = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  void *b = malloc(*len);
  if (fread(b, 1, *len, f) != *len) exit(2);
  fclose(f);
  return b;
}

static void wr(const char *dir, const char *name, const void *p, size_t len) {
  char path[4096];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  fwrite(p, 1, len, f);
  fclose(f);
}

static int gate(const char *dir, int scalar) {
  size_t len;
  int32_t *shape = rd(dir, "shape.bin", &len);
  int m = shape[0], k = shape[1], n = shape[2];
  uint8_t *a = rd(dir, "a.bin", &len), *b = rd(dir, "b.bin", &len);
  if (k % 128 || n % 64) { fprintf(stderr, "shape\n"); return 2; }
  uint32_t *atoms = calloc((size_t)(k / 32) * m * n, 4), *totals = calloc((size_t)(k / 128) * m * n, 4);
  long slow = 0;
  if (scalar) {
    uint32_t *at = malloc((k / 32) * 4), *tt = malloc((k / 128) * 4);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        chain_exact(a + (size_t)i * k, b + (size_t)j * k, 1, k, at, tt);
        for (int t = 0; t < k / 32; t++) atoms[((size_t)t * m + i) * n + j] = at[t];
        for (int w = 0; w < k / 128; w++) totals[((size_t)w * m + i) * n + j] = tt[w];
      }
  } else {
    Arows A;
    prep_a(&A, a, m, k);
    uint8_t *bb = block_b(b, n, k);
    Bblock *B = aligned_alloc(64, sizeof(Bblock));
    uint32_t *local = aligned_alloc(64, (size_t)m * 256), *total = aligned_alloc(64, (size_t)m * 256);
    for (int jb = 0; jb < n / 64; jb++)
      slow += column_block(&A, bb + (size_t)jb * k * 64, B, local, total, 0, atoms, totals, n, jb * 64, NULL);
  }
  wr(dir, "atoms.bin", atoms, (size_t)(k / 32) * m * n * 4);
  wr(dir, "totals.bin", totals, (size_t)(k / 128) * m * n * 4);
  printf("{\"gate\": \"wrote\", \"mode\": \"%s\", \"m\": %d, \"k\": %d, \"n\": %d, \"slow_lanes\": %ld, \"atom_outputs\": %ld, "
         "\"bad_input\": %d}\n", scalar ? "scalar" : "vector", m, k, n, slow, (long)(k / 32) * m * n, bad_input);
  return bad_input ? 6 : 0;
}

// ---- bench -----------------------------------------------------------------------------------------------------------

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng(void) {
  rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
  return rng_state;
}

static uint8_t e4m3_nearest(double v) {  // RN onto E4M3, saturating at 448
  double best = 1e300;
  uint8_t bc = 0;
  for (int c = 0; c < 127; c++) {
    double d = fabs(e4m3_value((uint8_t)c) - fabs(v));
    if (d < best) { best = d; bc = (uint8_t)c; }
  }
  return v < 0 && bc ? bc | 0x80 : bc;
}

static void fill(uint8_t *p, size_t count, const char *data, const uint8_t *pool, size_t pool_len) {
  if (!strcmp(data, "uniform")) {
    for (size_t i = 0; i < count; i++) { uint8_t c; do c = (uint8_t)rng(); while ((c & 0x7F) == 0x7F); p[i] = c; }
  } else if (pool) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++) p[i] = pool[i % pool_len];
  } else {  // gauss: N(0, 1) * 64 (|x| above 448, the format's max, is about 7 sigma), RN to E4M3; 4096 distinct codes reused
    uint8_t tab[4096];
    for (int i = 0; i < 4096; i++) {
      double u1 = ((rng() >> 11) + 1) * 0x1.0p-53, u2 = (rng() >> 11) * 0x1.0p-53;
      tab[i] = e4m3_nearest(sqrt(-2 * log(u1)) * cos(2 * M_PI * u2) * 64);
    }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++) p[i] = tab[(i * 2654435761u + (i >> 12)) & 4095];
  }
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int bench(int m, int k, int n, int W, int ks, int reps, const char *data) {
  uint8_t *pool = NULL;
  size_t pool_len = 0;
  if (!strncmp(data, "file:", 5)) {
    FILE *f = fopen(data + 5, "rb");
    if (!f) { perror(data + 5); return 2; }
    fseek(f, 0, SEEK_END); pool_len = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    pool = malloc(pool_len);
    if (fread(pool, 1, pool_len, f) != pool_len) return 2;
    fclose(f);
  }
  uint8_t *a = malloc((size_t)m * k), *bw = aligned_alloc(64, (size_t)W * k * n);
  fill(a, (size_t)m * k, data, pool, pool_len);
  fill(bw, (size_t)W * k * n, data, pool ? pool + pool_len / 2 : NULL, pool_len / 2);
  Arows A;
  prep_a(&A, a, m, k);
  int nthreads = 0, blocks = n / 64;
  double best = 1e30, sum = 0;
  long slow = 0;
  uint64_t fold = 0;
  for (int rep = 0; rep < reps + 1; rep++) {
    long sl = 0;
    double t0 = now();
#pragma omp parallel reduction(+ : sl) reduction(^ : fold)
    {
      Bblock *B = aligned_alloc(64, sizeof(Bblock));
      uint32_t *local = aligned_alloc(64, (size_t)m * 256), *total = aligned_alloc(64, (size_t)m * 256);
      uint32_t snap[128] __attribute__((aligned(64)));
#pragma omp single
      nthreads = omp_get_num_threads();
#pragma omp for schedule(static)
      for (int wb = 0; wb < W * blocks; wb++) {
        sl += column_block(&A, bw + (size_t)wb * k * 64, B, local, total, ks, NULL, NULL, 0, 0, snap);
        fold ^= total[wb & 63];
      }
      free(B); free(local); free(total);
    }
    double dt = now() - t0;
    if (rep > 0) { sum += dt; if (dt < best) best = dt; slow = sl; }
  }
  double products = (double)m * k * n * W, wbytes = (double)k * n * W;
  printf("{\"kernel\": \"pearl_c_cpu\", \"data\": \"%s\", \"m\": %d, \"k\": %d, \"n\": %d, \"weights\": %d, \"ks\": %d, "
         "\"prefetch\": %ld, \"fused\": %d, \"threads\": %d, \"reps\": %d, \"best_s\": %.6f, \"mean_s\": %.6f, \"products_per_s\": %.4e, "
         "\"weight_bytes_per_s\": %.4e, \"weight_bytes_per_s_mean\": %.4e, \"slow_lanes\": %ld, \"slow_fraction\": %.3e, "
         "\"fold\": %llu}\n",
         data, m, k, n, W, ks, prefetch_dist, fuse && m <= 2, nthreads, reps, best, sum / reps, products / best, wbytes / best, wbytes / (sum / reps), slow,
         slow / (products / 32), (unsigned long long)fold);
  free_a(&A);
  return 0;
}

int main(int argc, char **argv) {
  init_tables();
  if (getenv("PEARL_PREFETCH")) prefetch_dist = atol(getenv("PEARL_PREFETCH"));
  if (getenv("PEARL_FUSE")) fuse = atoi(getenv("PEARL_FUSE"));
  if (argc >= 3 && !strcmp(argv[1], "gate")) return gate(argv[2], argc >= 4 && !strcmp(argv[3], "scalar"));
  if (argc >= 9 && !strcmp(argv[1], "bench"))
    return bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), argv[8]);
  fprintf(stderr, "usage: pearl_c_cpu gate DIR [scalar] | bench M K N W KS REPS gauss|uniform|file:PATH\n");
  return 2;
}
