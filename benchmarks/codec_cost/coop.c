/* Cooperative dependent squaring chain mod p = 2^W - C across T pinned threads (one per core).
 * The P2 store's internal/efficient-crypto/candidates/p2-coop-cores/coop.c, with threads pinned through $COOP_CPUS.
 *
 * Kernel: the campaign's ifma_ps2 (AVX-512 IFMA, radix 2^52, product scanning, squaring symmetry; see
 * ../alt-primitives/p2-latency/bench.c), with its 40 output-block pairs split among T threads by work (the squaring
 * triangle). Per squaring:
 *   1. thread t computes the column sums S of its output blocks from its private copy of the input (no remote reads);
 *   2. one all-to-all flag barrier (each thread publishes its step number on its own cache line, then spins on the
 *      other T-1 lines; no pause instruction);
 *   3. every thread folds and carries the whole S (reading the other threads' blocks from their caches) into its own
 *      private input copy for the next step (redundant fold: one barrier per squaring, S double-buffered).
 * Modes:
 *   gmp                : one thread, mpn_sqr (GMP's Toom plan at this size) + mpn fold by C (W = 64 * limbs).
 *   ifma T             : cooperative chain on cores 0..T-1 (T = 1 is the plain single-core ifma_ps2 path).
 *   split T            : compute-only (step 1 with no barrier, no fold): per-thread time = ideal compute share.
 *   fold               : the fold + carry alone on one thread.
 *   pingpong           : one-cache-line round trip between every pair of cores.
 * Every chain result is compared with mpz_powm(y, 2^N, p).
 *
 * Build: gcc -O3 -march=native -pthread -DW=16448 -DC=21065 -o coop coop.c -lgmp
 */
#define _GNU_SOURCE
#include <gmp.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef W
#define W 16448
#endif
#ifndef C
#define C 21065
#endif
#define NL ((W + 51) / 52)
#define Z (52 * NL - W)
#define MASK52 ((1ULL << 52) - 1)
#define KFOLD (((uint64_t)(C)) << Z)
#define NB2 ((2 * NL + 7) / 8)
#define PADA ((NL + 23) & ~7)
#define TOT (PADA + NL + PADA)
#define SLEN (8 * NB2 + 16)
#define MAXT 8

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
/* thread t runs on the t-th CPU of $COOP_CPUS (a comma list: one CPU a physical core, one L3), else on CPU t */
static void pin(int core) {
  const char *list = getenv("COOP_CPUS");
  int cpu = core;
  if (list) {
    const char *q = list;
    for (int i = 0; i < core && q; i++) {
      q = strchr(q, ',');
      if (q) q++;
    }
    if (q) cpu = atoi(q);
  }
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

#define LD(x) _mm512_loadu_si512(xa + (x))

static inline void sqr_tail(const uint64_t *xa, uint64_t *S, int k, __m512i l0, __m512i l1, __m512i l2, __m512i l3,
                            __m512i h0, __m512i h1, __m512i h2, __m512i h3) {
  const int bo = 8 * k, i = 4 * k;
  const __m512i M0 = LD(bo - i), M1 = LD(bo - i - 1), M2 = LD(bo - i - 2), M3 = LD(bo - i - 3);
  const __m512i b0 = _mm512_set1_epi64((long long)xa[i]), b1 = _mm512_set1_epi64((long long)xa[i + 1]);
  const __m512i b2 = _mm512_set1_epi64((long long)xa[i + 2]), b3 = _mm512_set1_epi64((long long)xa[i + 3]);
  l0 = _mm512_mask_madd52lo_epu64(l0, 0xFE, M0, b0); h0 = _mm512_mask_madd52hi_epu64(h0, 0xFC, M1, b0);
  l1 = _mm512_mask_madd52lo_epu64(l1, 0xF8, M1, b1); h1 = _mm512_mask_madd52hi_epu64(h1, 0xF0, M2, b1);
  l2 = _mm512_mask_madd52lo_epu64(l2, 0xE0, M2, b2); h2 = _mm512_mask_madd52hi_epu64(h2, 0xC0, M3, b2);
  l3 = _mm512_mask_madd52lo_epu64(l3, 0x80, M3, b3);
  __m512i s = _mm512_add_epi64(_mm512_add_epi64(_mm512_add_epi64(l0, l1), _mm512_add_epi64(l2, l3)),
                               _mm512_add_epi64(_mm512_add_epi64(h0, h1), _mm512_add_epi64(h2, h3)));
  s = _mm512_slli_epi64(s, 1);
  const __m512i v = LD(4 * k);
  const __m512i dl = _mm512_madd52lo_epu64(_mm512_setzero_si512(), v, v);
  const __m512i dh = _mm512_madd52hi_epu64(_mm512_setzero_si512(), v, v);
  s = _mm512_add_epi64(s, _mm512_permutex2var_epi64(dl, _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0), dh));
  _mm512_store_si512(S + 8 * k, s);
}

#define ROWS4(i, l0, l1, l2, l3, h0, h1, h2, h3)                                                               \
  do {                                                                                                        \
    const __m512i b0_ = _mm512_set1_epi64((long long)xa[(i)]), b1_ = _mm512_set1_epi64((long long)xa[(i) + 1]); \
    const __m512i b2_ = _mm512_set1_epi64((long long)xa[(i) + 2]), b3_ = _mm512_set1_epi64((long long)xa[(i) + 3]); \
    l0 = _mm512_madd52lo_epu64(l0, L, b0_);  h0 = _mm512_madd52hi_epu64(h0, L1, b0_);                         \
    l1 = _mm512_madd52lo_epu64(l1, L1, b1_); h1 = _mm512_madd52hi_epu64(h1, L2, b1_);                         \
    l2 = _mm512_madd52lo_epu64(l2, L2, b2_); h2 = _mm512_madd52hi_epu64(h2, L3, b2_);                         \
    l3 = _mm512_madd52lo_epu64(l3, L3, b3_); h3 = _mm512_madd52hi_epu64(h3, L4, b3_);                         \
  } while (0)

static inline int pair_lo_n(int k, int n) {
  int i0a = 8 * k - n; if (i0a < 0) i0a = 0;
  int i0b = 8 * k + 8 - n; if (i0b < 0) i0b = 0;
  const int isa = 4 * k - ((4 * k - i0a + 3) & ~3);
  const int isb = 4 * k + 4 - ((4 * k + 4 - i0b + 3) & ~3);
  return isa < isb - 8 ? isa : isb - 8;
}

#define pair_lo(k) pair_lo_n(k, NL)
/* column sums of output-block pairs (k, k+1) for even k in [kb, ke), operand of n limbs (zero-padded) */
static inline __attribute__((always_inline)) void sqr_pairs_n(const uint64_t *xa, uint64_t *S, int kb, int ke, int n) {
  for (int k = kb; k < ke; k += 2) {
    const int lo = pair_lo_n(k, n), bo = 8 * k;
    __m512i l0 = _mm512_setzero_si512(), l1 = l0, l2 = l0, l3 = l0, h0 = l0, h1 = l0, h2 = l0, h3 = l0;
    __m512i m0 = l0, m1 = l0, m2 = l0, m3 = l0, g0 = l0, g1 = l0, g2 = l0, g3 = l0;
    __m512i L = LD(bo - lo);
    int i = lo;
    for (; i < 4 * k - 4; i += 4) {
      const __m512i L1 = LD(bo - i - 1), L2 = LD(bo - i - 2), L3 = LD(bo - i - 3), L4 = LD(bo - i - 4);
      ROWS4(i, l0, l1, l2, l3, h0, h1, h2, h3);
      ROWS4(i + 8, m0, m1, m2, m3, g0, g1, g2, g3);
      L = L4;
    }
    {
      const __m512i L1 = LD(bo - i - 1), L2 = LD(bo - i - 2), L3 = LD(bo - i - 3), L4 = LD(bo - i - 4);
      ROWS4(i, l0, l1, l2, l3, h0, h1, h2, h3);
    }
    sqr_tail(xa, S, k, l0, l1, l2, l3, h0, h1, h2, h3);
    sqr_tail(xa, S, k + 1, m0, m1, m2, m3, g0, g1, g2, g3);
  }
}
static void sqr_pairs(const uint64_t *xa, uint64_t *S, int kb, int ke) { sqr_pairs_n(xa, S, kb, ke, NL); }

/* out (NL limbs < 2^52) = S_lo + K * S_hi, carried; value < 2^(52 NL), congruent mod p */
static inline void fold_ps(uint64_t *out, const uint64_t *S) {
  uint64_t U[8 * ((NL + 7) / 8)] __attribute__((aligned(64)));
  uint64_t top[8] __attribute__((aligned(64)));
  const __m512i Kv = _mm512_set1_epi64((long long)KFOLD), M = _mm512_set1_epi64((long long)MASK52);
  const __m512i z = _mm512_setzero_si512();
  uint64_t top2[8] __attribute__((aligned(64)));
  /* X = xl + xh 2^52 with xh K up to 2^61 when K is large: lo52(xh K) goes to limb m+1, hi52(xh K) to m+2 */
  __m512i hp = z, hhp = z;
  for (int q = 0; q < (NL + 7) / 8; q++) {
    const __m512i X = _mm512_loadu_si512(S + NL + 8 * q);
    const __m512i xl = _mm512_and_si512(X, M), xh = _mm512_srli_epi64(X, 52);
    const __m512i lo = _mm512_madd52lo_epu64(_mm512_load_si512(S + 8 * q), xl, Kv);
    const __m512i hi = _mm512_madd52hi_epu64(_mm512_madd52lo_epu64(z, xh, Kv), xl, Kv);
    const __m512i hh = _mm512_madd52hi_epu64(z, xh, Kv);
    _mm512_store_si512(U + 8 * q, _mm512_add_epi64(_mm512_add_epi64(lo, _mm512_alignr_epi64(hi, hp, 7)),
                                                   _mm512_alignr_epi64(hh, hhp, 6)));
    hp = hi;
    hhp = hh;
  }
  _mm512_store_si512(top, hp);
  _mm512_store_si512(top2, hhp);
  uint64_t carry = 0;
  for (int m = 0; m < NL; m++) {
    const uint64_t acc = U[m] + carry;
    out[m] = acc & MASK52;
    carry = acc >> 52;
  }
  /* limb NL: carry + hi(NL-1) + hh(NL-2); limb NL+1: hh(NL-1). NL - 1 is not lane 0 of its vector. */
  unsigned __int128 ex = (unsigned __int128)carry + top[(NL - 1) & 7] + top2[(NL - 2) & 7] +
                         ((unsigned __int128)top2[(NL - 1) & 7] << 52);
  unsigned __int128 t = ex * KFOLD;
  while (t) {
    int m = 0;
    for (; m < NL && t; m++) {
      t += out[m];
      out[m] = (uint64_t)t & MASK52;
      t >>= 52;
    }
    if (t) t *= KFOLD;
  }
}

/* ---- conversions ---- */
static mpz_t g_p;
static void to_big(uint64_t *r, const mpz_t z) {
  mpz_t t;
  mpz_init(t);
  mpz_mod(t, z, g_p);
  for (int i = 0; i < NL; i++) {
    r[i] = mpz_getlimbn(t, 0) & MASK52;
    mpz_tdiv_q_2exp(t, t, 52);
  }
  mpz_clear(t);
}
static void big_to_mpz(mpz_t z, const uint64_t *a) {
  mpz_set_ui(z, 0);
  for (int i = NL - 1; i >= 0; i--) {
    mpz_mul_2exp(z, z, 52);
    mpz_add_ui(z, z, a[i]);
  }
  mpz_mod(z, z, g_p);
}

/* ---- work split ---- */
static int g_T;
static int g_kb[MAXT + 1];
static void make_split(int T) {
  double wk[NB2], tot = 0;
  for (int k = 0; k < NB2; k += 2) {
    wk[k] = (double)((4 * k - pair_lo(k)) / 4 + 1) * 16 + 30; /* madds per pair + tails */
    tot += wk[k];
  }
  g_kb[0] = 0;
  int t = 1;
  double acc = 0;
  for (int k = 0; k < NB2 && t < T; k += 2) {
    acc += wk[k];
    if (acc >= tot * t / T) g_kb[t++] = k + 2;
  }
  while (t <= T) g_kb[t++] = NB2;
  g_kb[T] = NB2;
}

/* ---- cooperative chain ---- */
typedef struct { _Alignas(64) atomic_long v; uint64_t sp[7]; } flag_t;
static flag_t g_flag[MAXT];
static flag_t g_go;
static uint64_t g_S[2][SLEN] __attribute__((aligned(64)));
static uint64_t g_xa[MAXT][TOT + 16] __attribute__((aligned(64)));
static uint64_t g_y0[NL];
static long g_nsq;
static int g_mode; /* 0 = chain, 1 = compute-only */
static double g_tthr[MAXT];
static pthread_barrier_t g_pb;

static void *worker(void *arg) {
  const int t = (int)(long)arg, T = g_T;
  pin(t);
  uint64_t *xa = g_xa[t] + PADA;
  const int kb = g_kb[t], ke = g_kb[t + 1];
  pthread_barrier_wait(&g_pb);
  memset(g_xa[t], 0, sizeof g_xa[t]);
  memcpy(xa, g_y0, sizeof g_y0);
  atomic_store(&g_flag[t].v, 0);
  pthread_barrier_wait(&g_pb);
  /* spin start so all threads leave together */
  while (atomic_load_explicit(&g_go.v, memory_order_acquire) == 0) {}
  const double t0 = now_ns();
  if (g_mode == 1) {
    for (long s = 1; s <= g_nsq; s++) sqr_pairs(xa, g_S[t & 1], kb, ke);
  } else if (T == 1) {
    for (long s = 1; s <= g_nsq; s++) {
      sqr_pairs(xa, g_S[0], 0, NB2);
      fold_ps(xa, g_S[0]);
    }
  } else {
    for (long s = 1; s <= g_nsq; s++) {
      uint64_t *S = g_S[s & 1];
      sqr_pairs(xa, S, kb, ke);
      atomic_store_explicit(&g_flag[t].v, s, memory_order_release);
      for (int u = 0; u < T; u++)
        if (u != t)
          while (atomic_load_explicit(&g_flag[u].v, memory_order_acquire) < s) {}
      fold_ps(xa, S);
    }
  }
  g_tthr[t] = now_ns() - t0;
  return NULL;
}

/* returns ns per squaring (thread 0 wall; for compute-only, the slowest thread); verifies chain result */
static double run_coop(int T, int mode, long nsq, const mpz_t y, int *ok) {
  g_T = T;
  g_mode = mode;
  g_nsq = nsq;
  make_split(T);
  to_big(g_y0, y);
  atomic_store(&g_go.v, 0);
  pthread_barrier_init(&g_pb, NULL, T + 1);
  pthread_t th[MAXT];
  for (int t = 0; t < T; t++) pthread_create(&th[t], NULL, worker, (void *)(long)t);
  pthread_barrier_wait(&g_pb);
  pthread_barrier_wait(&g_pb);
  atomic_store_explicit(&g_go.v, 1, memory_order_release);
  for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
  pthread_barrier_destroy(&g_pb);
  double r = g_tthr[0];
  if (mode == 1)
    for (int t = 1; t < T; t++) if (g_tthr[t] > r) r = g_tthr[t];
  if (ok && mode == 0) {
    mpz_t got, ref, e;
    mpz_inits(got, ref, e, NULL);
    big_to_mpz(got, g_xa[0] + PADA);
    mpz_ui_pow_ui(e, 2, nsq);
    mpz_powm(ref, y, e, g_p);
    *ok = mpz_cmp(got, ref) == 0;
    mpz_clears(got, ref, e, NULL);
  }
  return r / nsq;
}

/* ---- owner-computes chain ----
 * Operand of N8 = 8 ceil(NL/8) limbs (redundant: value < 2^(52 N8)); 2^(52 N8) = K 2^(52 D8) mod p with K = C << Z.
 * Thread t owns output limbs [8b, 8e) and computes the column sums of blocks [b, e) and [N8/8 + b, N8/8 + e): for
 * squaring these pairs cost the same for every block, so equal ranges balance. It folds its high columns into its own
 * limbs (offset D8, D8+1, D8+2), carries locally, publishes its limbs and a spill (the carry-out and the fold terms
 * that land above 8e), and after one barrier every thread gathers all limbs and adds every spill (the last one wraps
 * through K) on its private copy. Only the N8-limb result crosses cores. */
#define N8 (((NL + 7) / 8) * 8)
#define D8 (N8 - NL)
#define NBL (N8 / 8)
#define PAD2 ((N8 + 23) & ~7)
static uint64_t g_oxa[MAXT][PAD2 + N8 + PAD2 + 16] __attribute__((aligned(64)));
static uint64_t g_oS[MAXT][2 * N8 + 64] __attribute__((aligned(64)));
static uint64_t g_pub[2][N8] __attribute__((aligned(64)));
static uint64_t g_spill[2][MAXT][8] __attribute__((aligned(64)));
static flag_t g_oflag[2][MAXT];
static int g_demote, g_pfw;
static int g_ob[MAXT + 1];
static double g_ph[MAXT][6]; /* TSC ticks per phase: compute, fold+carry, publish, wait, gather, fixup */
static int g_prefetch, g_ospill;
#ifdef PHASES
#define TS0() (c0 = __rdtsc())
#define TS(i) (c1 = __rdtsc(), ph[i] += c1 - c0, c0 = c1)
#else
#define TS0() ((void)0)
#define TS(i) ((void)0)
#endif

static inline void add_small(uint64_t *xa, int pos, unsigned __int128 v) {
  while (v) {
    if (pos >= N8) { v *= KFOLD; pos = pos - N8 + D8; }
    v += xa[pos];
    xa[pos] = (uint64_t)v & MASK52;
    v >>= 52;
    pos++;
  }
}

static void *oworker(void *arg) {
  const int t = (int)(long)arg, T = g_T;
  pin(t);
  uint64_t *xa = g_oxa[t] + PAD2, *S = g_oS[t];
  const int b = g_ob[t], e = g_ob[t + 1];
  uint64_t Vb[N8 + 16] __attribute__((aligned(64)));
  const __m512i Kv = _mm512_set1_epi64((long long)KFOLD), M = _mm512_set1_epi64((long long)MASK52);
  const __m512i z = _mm512_setzero_si512();
  pthread_barrier_wait(&g_pb);
  memset(g_oxa[t], 0, sizeof g_oxa[t]);
  memcpy(xa, g_y0, sizeof g_y0);
  memset(S, 0, sizeof g_oS[t]);
  atomic_store(&g_oflag[0][t].v, -1);
  atomic_store(&g_oflag[1][t].v, 0);
  pthread_barrier_wait(&g_pb);
  while (atomic_load_explicit(&g_go.v, memory_order_acquire) == 0) {}
  const double t0 = now_ns();
  uint64_t ph[6] = {0};
#ifdef PHASES
  uint64_t c0, c1;
#endif
  for (long s = 1; s <= g_nsq; s++) {
    const int par = s & 1;
    TS0();
    if (g_pfw && T > 1)
      for (int l = b; l < e; l++) _m_prefetchw(g_pub[par] + 8 * l);
    sqr_pairs_n(xa, S, b, e, N8);
    sqr_pairs_n(xa, S, NBL + b, NBL + e, N8);
    TS(0);
    __m512i p0 = z, p1 = z, p2 = z;
    for (int q = b; q <= e; q++) {
      __m512i P0 = z, P1 = z, P2 = z, V = z;
      if (q < e) {
        const __m512i X = _mm512_load_si512(S + N8 + 8 * q);
        const __m512i xl = _mm512_and_si512(X, M), xh = _mm512_srli_epi64(X, 52);
        P0 = _mm512_madd52lo_epu64(z, xl, Kv);
        P1 = _mm512_madd52hi_epu64(_mm512_madd52lo_epu64(z, xh, Kv), xl, Kv);
        P2 = _mm512_madd52hi_epu64(z, xh, Kv);
        V = _mm512_load_si512(S + 8 * q);
      }
      V = _mm512_add_epi64(V, _mm512_alignr_epi64(P0, p0, 8 - D8));
      V = _mm512_add_epi64(V, _mm512_add_epi64(_mm512_alignr_epi64(P1, p1, 7 - D8), _mm512_alignr_epi64(P2, p2, 6 - D8)));
      _mm512_store_si512(Vb + 8 * (q - b), V);
      p0 = P0; p1 = P1; p2 = P2;
    }
    uint64_t carry = 0;
    uint64_t *pub = g_pub[par];
    for (int j = 0; j < 8 * (e - b); j++) {
      const uint64_t acc = Vb[j] + carry;
      xa[8 * b + j] = acc & MASK52;
      carry = acc >> 52;
    }
    TS(1);
    if (T > 1) {
      memcpy(pub + 8 * b, xa + 8 * b, 64 * (e - b));
      uint64_t *sp = g_ospill ? g_spill[par][t] : g_oflag[par][t].sp;
      for (int i = 0; i < D8 + 2; i++) sp[i] = Vb[8 * (e - b) + i];
      sp[0] += carry;
      atomic_store_explicit(&g_oflag[par][t].v, s, memory_order_release);
      if (g_demote) {
        for (int l = b; l < e; l++) _cldemote(pub + 8 * l);
        _cldemote(&g_oflag[par][t]);
      }
      TS(2);
      for (int u = 0; u < T; u++)
        if (u != t)
          while (atomic_load_explicit(&g_oflag[par][u].v, memory_order_acquire) < s) {}
      TS(3);
      if (g_prefetch) {
        for (int u = 0; u < T; u++)
          if (u != t)
            for (int l = g_ob[u]; l < g_ob[u + 1]; l++) _mm_prefetch((const char *)(pub + 8 * l), _MM_HINT_T0);
        if (g_ospill) for (int u = 0; u < T; u++) _mm_prefetch((const char *)g_spill[par][u], _MM_HINT_T0);
      }
      for (int u = 0; u < T; u++)
        if (u != t) memcpy(xa + 8 * g_ob[u], pub + 8 * g_ob[u], 64 * (g_ob[u + 1] - g_ob[u]));
      TS(4);
      for (int u = 0; u < T; u++)
        for (int i = 0; i < D8 + 2; i++)
          add_small(xa, 8 * g_ob[u + 1] + i, g_ospill ? g_spill[par][u][i] : g_oflag[par][u].sp[i]);
      TS(5);
    } else {
      Vb[8 * (e - b)] += carry;
      for (int i = 0; i < 8; i++) add_small(xa, N8 + i, Vb[8 * (e - b) + i]);
    }
  }
  g_tthr[t] = now_ns() - t0;
  for (int i = 0; i < 6; i++) g_ph[t][i] = (double)ph[i];
  return NULL;
}

static double run_owner(int T, long nsq, const mpz_t y, int *ok) {
  if (D8 < 1 || D8 > 5) { fprintf(stderr, "owner mode needs 1 <= D8 <= 5\n"); exit(2); }
  g_T = T;
  g_nsq = nsq;
  for (int t = 0; t <= T; t++) g_ob[t] = 2 * ((NBL / 2) * t / T);
  memset(g_y0, 0, sizeof g_y0);
  to_big(g_y0, y);
  atomic_store(&g_go.v, 0);
  pthread_barrier_init(&g_pb, NULL, T + 1);
  pthread_t th[MAXT];
  for (int t = 0; t < T; t++) pthread_create(&th[t], NULL, oworker, (void *)(long)t);
  pthread_barrier_wait(&g_pb);
  pthread_barrier_wait(&g_pb);
  atomic_store_explicit(&g_go.v, 1, memory_order_release);
  for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
  pthread_barrier_destroy(&g_pb);
  mpz_t got, ref, e;
  mpz_inits(got, ref, e, NULL);
  int same = 1;
  for (int t = 1; t < T; t++) same &= !memcmp(g_oxa[t] + PAD2, g_oxa[0] + PAD2, 8 * N8);
  mpz_set_ui(got, 0);
  for (int i = N8 - 1; i >= 0; i--) { mpz_mul_2exp(got, got, 52); mpz_add_ui(got, got, g_oxa[0][PAD2 + i]); }
  mpz_mod(got, got, g_p);
  mpz_ui_pow_ui(e, 2, nsq);
  mpz_powm(ref, y, e, g_p);
  *ok = same && mpz_cmp(got, ref) == 0;
  mpz_clears(got, ref, e, NULL);
  for (int t = 0; t <= T; t++) g_kb[t] = g_ob[t];
  if (getenv("PHASES") && g_ph[0][0] > 0) {
    double tot = 0;
    for (int i = 0; i < 6; i++) tot += g_ph[0][i];
    fprintf(stderr, "phases T=%d thread0 ns/sqr:", T);
    for (int i = 0; i < 6; i++) fprintf(stderr, " %.0f", g_ph[0][i] / tot * g_tthr[0] / nsq);
    fprintf(stderr, "  (compute fold publish wait gather fixup)\n");
  }
  return g_tthr[0] / nsq;
}

/* ---- GMP: mpn_sqr + fold (W = 64 * NLIMB exactly for W = 16448) ---- */
#define NLIMB ((W + 63) / 64)
static double run_gmp(long nsq, const mpz_t y, int *ok) {
  static mp_limb_t a[NLIMB], pr[2 * NLIMB];
  if (W != 64 * NLIMB) { fprintf(stderr, "gmp mode needs W multiple of 64\n"); exit(2); }
  memset(a, 0, sizeof a);
  size_t n = mpz_size(y);
  for (size_t i = 0; i < n; i++) a[i] = mpz_getlimbn(y, i);
  double t0 = now_ns();
  for (long s = 0; s < nsq; s++) {
    mpn_sqr(pr, a, NLIMB);
    mp_limb_t c = mpn_addmul_1(pr, pr + NLIMB, NLIMB, C);
    while (c) c = mpn_add_1(pr, pr, NLIMB, c * (mp_limb_t)C);
    memcpy(a, pr, sizeof a);
  }
  double dt = now_ns() - t0;
  mpz_t got, ref, e;
  mpz_inits(got, ref, e, NULL);
  mpz_import(got, NLIMB, -1, sizeof(mp_limb_t), 0, 0, a);
  mpz_mod(got, got, g_p);
  mpz_ui_pow_ui(e, 2, nsq);
  mpz_powm(ref, y, e, g_p);
  *ok = mpz_cmp(got, ref) == 0;
  mpz_clears(got, ref, e, NULL);
  return dt / nsq;
}

static double run_fold(long n) {
  static uint64_t out[NL + 8];
  for (int i = 0; i < SLEN; i++) g_S[0][i] = (i < 2 * NL) ? ((uint64_t)i * 0x9e3779b97f4a7c15ULL) >> 5 : 0;
  double t0 = now_ns();
  for (long s = 0; s < n; s++) {
    fold_ps(out, g_S[0]);
    g_S[0][0] ^= out[5] & 1;
  }
  return (now_ns() - t0) / n;
}

/* ---- ping-pong ---- */
static flag_t g_pp;
static long g_ppn;
static int g_ppcore[2];
static double g_pprtt;
static void *pp_worker(void *arg) {
  const int me = (int)(long)arg;
  pin(g_ppcore[me]);
  pthread_barrier_wait(&g_pb);
  double t0 = now_ns();
  for (long i = 0; i < g_ppn; i++) {
    const long want = 2 * i + me;
    while (atomic_load_explicit(&g_pp.v, memory_order_acquire) != want) {}
    atomic_store_explicit(&g_pp.v, want + 1, memory_order_release);
  }
  if (me == 0) g_pprtt = (now_ns() - t0) / g_ppn;
  return NULL;
}
static double pingpong(int a, int b, long n) {
  g_ppcore[0] = a; g_ppcore[1] = b; g_ppn = n;
  atomic_store(&g_pp.v, 0);
  pthread_barrier_init(&g_pb, NULL, 2);
  pthread_t th[2];
  for (int i = 0; i < 2; i++) pthread_create(&th[i], NULL, pp_worker, (void *)(long)i);
  for (int i = 0; i < 2; i++) pthread_join(th[i], NULL);
  pthread_barrier_destroy(&g_pb);
  return g_pprtt;
}

static void stats(double *v, int n, double *mn, double *md) {
  qsort(v, n, sizeof(double), cmpd);
  *mn = v[0];
  *md = v[n / 2];
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "ifma";
  g_prefetch = getenv("PREFETCH") != NULL;
  g_ospill = getenv("OWNSPILL") != NULL;
  g_demote = getenv("DEMOTE") != NULL;
  g_pfw = getenv("PREFETCHW") != NULL;
  int T = argc > 2 ? atoi(argv[2]) : 1;
  int reps = argc > 3 ? atoi(argv[3]) : 15;
  long nsq = argc > 4 ? atol(argv[4]) : 20000;
  mpz_init(g_p);
  mpz_ui_pow_ui(g_p, 2, W);
  mpz_sub_ui(g_p, g_p, C);
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 20260928u + T);
  mpz_t y;
  mpz_init(y);
  double v[256], mn, md;
  int allok = 1;
  if (!strcmp(mode, "pingpong")) {
    int nc = T;
    for (int a = 0; a < nc; a++)
      for (int b = a + 1; b < nc; b++) {
        for (int r = 0; r < reps; r++) v[r] = pingpong(a, b, 200000);
        stats(v, reps, &mn, &md);
        printf("{\"mode\":\"pingpong\",\"cores\":[%d,%d],\"rtt_min_ns\":%.1f,\"rtt_med_ns\":%.1f}\n", a, b, mn, md);
      }
    return 0;
  }
  if (!strcmp(mode, "debug")) {
    mpz_t a, sq, s, f;
    mpz_inits(a, sq, s, f, NULL);
    mpz_urandomm(y, rs, g_p);
    memset(g_xa[0], 0, sizeof g_xa[0]);
    uint64_t *xa = g_xa[0] + PADA;
    to_big(xa, y);
    big_to_mpz(a, xa);
    mpz_mul(sq, a, a);
    sqr_pairs(xa, g_S[0], 0, NB2);
    mpz_set_ui(s, 0);
    for (int i = 8 * NB2 - 1; i >= 0; i--) { mpz_mul_2exp(s, s, 52); mpz_add_ui(s, s, g_S[0][i]); }
    printf("columns == a^2: %d\n", mpz_cmp(s, sq) == 0);
    fold_ps(xa, g_S[0]);
    big_to_mpz(f, xa);
    mpz_mod(sq, sq, g_p);
    printf("fold == a^2 mod p: %d\n", mpz_cmp(f, sq) == 0);
    return 0;
  }
  if (!strcmp(mode, "fold")) {
    pin(0);
    for (int r = 0; r < reps; r++) v[r] = run_fold(nsq);
    stats(v, reps, &mn, &md);
    printf("{\"mode\":\"fold\",\"W\":%d,\"ns_min\":%.1f,\"ns_med\":%.1f}\n", W, mn, md);
    return 0;
  }
  if (!strcmp(mode, "gmp")) pin(0);
  for (int r = 0; r < reps; r++) {
    mpz_urandomm(y, rs, g_p);
    int ok = 1;
    if (!strcmp(mode, "gmp")) v[r] = run_gmp(nsq, y, &ok);
    else if (!strcmp(mode, "split")) v[r] = run_coop(T, 1, nsq, y, NULL);
    else if (!strcmp(mode, "owner")) v[r] = run_owner(T, nsq, y, &ok);
    else v[r] = run_coop(T, 0, nsq, y, &ok);
    allok &= ok;
  }
  stats(v, reps, &mn, &md);
  printf("{\"mode\":\"%s\",\"W\":%d,\"C\":%d,\"T\":%d,\"nsq\":%ld,\"reps\":%d,\"sqr_min_ns\":%.1f,\"sqr_med_ns\":%.1f,"
         "\"verified\":%s,\"split\":[", mode, W, C, T, nsq, reps, mn, md,
         strcmp(mode, "split") ? (allok ? "true" : "false") : "null");
  for (int t = 0; t <= T; t++) printf("%s%d", t ? "," : "", g_kb[t]);
  printf("]}\n");
  return allok ? 0 : 1;
}
