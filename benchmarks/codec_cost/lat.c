/* One word's dependent squaring chain mod p = 2^W - C on one pinned core, scalar (GMP's mpn_sqr at this size, then
 * the fold x = lo + C hi by mpn_addmul_1), any W: the latency floor a Sloth root sits on when the adversary has one
 * word to encode. A root costs about W squarings, so a root's latency is about W times the squaring's.
 *
 *   lat chain REPS NSQ   JSON: ns a squaring (min and median over REPS chains of NSQ), each chain checked against
 *                        mpz_powm(y, 2^NSQ, p)
 *   lat u128 REPS NSQ    the same with the squaring unrolled on unsigned __int128 (W a multiple of 64)
 *   lat clock            JSON: the core's clock from a dependent chain of 1-cycle adds
 *
 * Build: gcc -O3 -march=native -DW=512 -DC=569 -o lat lat.c -lgmp
 */
#define _GNU_SOURCE
#include <gmp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef W
#define W 512
#endif
#ifndef C
#define C 569
#endif
#define NLIMB ((W + 63) / 64)
#define TOPB (W % 64)

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

/* a <- a^2 mod p, not canonical: below 2^W when 64 divides W, else below 2^(W+1) */
static inline void sqr_fold(mp_limb_t *a, mp_limb_t *pr, mp_limb_t *hi) {
  mpn_sqr(pr, a, NLIMB);
#if TOPB == 0
  mp_limb_t c = mpn_addmul_1(pr, pr + NLIMB, NLIMB, C);
  while (c) c = mpn_add_1(pr, pr, NLIMB, c * (mp_limb_t)C);
  memcpy(a, pr, NLIMB * sizeof(mp_limb_t));
#else
  /* lo + C hi with hi = pr >> W (below 2^(W+2)), then the few bits at and above W folded once more */
  const int nh = 2 * NLIMB - W / 64;
  const mp_limb_t mask = (((mp_limb_t)1) << TOPB) - 1;
  mpn_rshift(hi, pr + W / 64, nh, TOPB);
  pr[W / 64] &= mask;
  for (int i = W / 64 + 1; i < 2 * NLIMB + 2; i++) pr[i] = 0;
  pr[nh] += mpn_addmul_1(pr, hi, nh, C);
  const mp_limb_t top = (pr[W / 64] >> TOPB) | (pr[W / 64 + 1] << (64 - TOPB));
  pr[W / 64] &= mask;
  pr[W / 64 + 1] = 0;
  mpn_add_1(pr, pr, NLIMB, top * (mp_limb_t)C);
  memcpy(a, pr, NLIMB * sizeof(mp_limb_t));
#endif
}

#if TOPB == 0
/* The same squaring unrolled at compile time on unsigned __int128 (mulx/adc): product scanning with a three-word
 * accumulator, each off-diagonal product added twice, then the fold by C. a stays below 2^W. */
static inline void acc3(uint64_t *c0, uint64_t *c1, uint64_t *c2, unsigned __int128 p) {
  unsigned __int128 s = (unsigned __int128)*c0 + (uint64_t)p;
  *c0 = (uint64_t)s;
  s = (unsigned __int128)*c1 + (uint64_t)(p >> 64) + (uint64_t)(s >> 64);
  *c1 = (uint64_t)s;
  *c2 += (uint64_t)(s >> 64);
}
static inline __attribute__((always_inline)) void sqr_u128(uint64_t *a) {
  uint64_t t[2 * NLIMB], c0 = 0, c1 = 0, c2 = 0;
#pragma GCC unroll 128
  for (int k = 0; k < 2 * NLIMB - 1; k++) {
#pragma GCC unroll 64
    for (int i = (k < NLIMB ? 0 : k - NLIMB + 1); i < k - i; i++) {
      const unsigned __int128 p = (unsigned __int128)a[i] * a[k - i];
      acc3(&c0, &c1, &c2, p);
      acc3(&c0, &c1, &c2, p);
    }
    if (!(k & 1)) acc3(&c0, &c1, &c2, (unsigned __int128)a[k / 2] * a[k / 2]);
    t[k] = c0;
    c0 = c1;
    c1 = c2;
    c2 = 0;
  }
  t[2 * NLIMB - 1] = c0;
  unsigned __int128 c = 0;
#pragma GCC unroll 64
  for (int i = 0; i < NLIMB; i++) {
    c += (unsigned __int128)t[NLIMB + i] * C + t[i];
    a[i] = (uint64_t)c;
    c >>= 64;
  }
  while (c) {
    c *= C;
    for (int i = 0; i < NLIMB && c; i++) {
      c += a[i];
      a[i] = (uint64_t)c;
      c >>= 64;
    }
  }
}

static double chain_u128(long nsq, const mpz_t y, const mpz_t p, int *ok) {
  uint64_t a[NLIMB] = {0};
  size_t n = mpz_size(y);
  for (size_t i = 0; i < n; i++) a[i] = mpz_getlimbn(y, i);
  const double t0 = now_ns();
  for (long s = 0; s < nsq; s++) sqr_u128(a);
  const double dt = now_ns() - t0;
  mpz_t got, ref, e;
  mpz_inits(got, ref, e, NULL);
  mpz_import(got, NLIMB, -1, sizeof(uint64_t), 0, 0, a);
  mpz_mod(got, got, p);
  mpz_ui_pow_ui(e, 2, nsq);
  mpz_powm(ref, y, e, p);
  *ok = mpz_cmp(got, ref) == 0;
  mpz_clears(got, ref, e, NULL);
  return dt / nsq;
}
#endif

static double chain(long nsq, const mpz_t y, const mpz_t p, int *ok) {
  static mp_limb_t a[NLIMB], pr[2 * NLIMB + 2], hi[2 * NLIMB + 2];
  memset(a, 0, sizeof a);
  memset(pr, 0, sizeof pr);
  size_t n = mpz_size(y);
  for (size_t i = 0; i < n; i++) a[i] = mpz_getlimbn(y, i);
  const double t0 = now_ns();
  for (long s = 0; s < nsq; s++) sqr_fold(a, pr, hi);
  const double dt = now_ns() - t0;
  mpz_t got, ref, e;
  mpz_inits(got, ref, e, NULL);
  mpz_import(got, NLIMB, -1, sizeof(mp_limb_t), 0, 0, a);
  mpz_mod(got, got, p);
  mpz_ui_pow_ui(e, 2, nsq);
  mpz_powm(ref, y, e, p);
  *ok = mpz_cmp(got, ref) == 0;
  mpz_clears(got, ref, e, NULL);
  return dt / nsq;
}

/* imul r64, r64 has a latency of 3 cycles on every recent x86 core; adds of an immediate are folded at rename on some */
static double clock_ghz(void) {
  const long n = 100000000;
  uint64_t x = 3, m = 0x9E3779B97F4A7C15ull;
  const double t0 = now_ns();
  for (long i = 0; i < n; i++)
    __asm__ volatile("imul %1, %0\n\timul %1, %0\n\timul %1, %0\n\timul %1, %0" : "+r"(x) : "r"(m));
  const double dt = now_ns() - t0;
  return 12.0 * n / dt;
}

int main(int argc, char **argv) {
  cpu_set_t s;
  sched_getaffinity(0, sizeof s, &s);
  int first = 0;
  while (first < CPU_SETSIZE && !CPU_ISSET(first, &s)) first++;
  CPU_ZERO(&s);
  CPU_SET(first, &s);
  sched_setaffinity(0, sizeof s, &s);
  if (argc == 2 && !strcmp(argv[1], "clock")) {
    double v[7];
    for (int r = 0; r < 7; r++) v[r] = clock_ghz();
    qsort(v, 7, sizeof(double), cmpd);
    printf("{\"mode\":\"clock\",\"cpu\":%d,\"ghz_max\":%.3f,\"ghz_med\":%.3f}\n", first, v[6], v[3]);
    return 0;
  }
  if (argc != 4 || (strcmp(argv[1], "chain") && strcmp(argv[1], "u128"))) return 2;
  const int u128 = !strcmp(argv[1], "u128");
#if TOPB != 0
  if (u128) return 2;
#endif
  const int reps = atoi(argv[2]);
  const long nsq = atol(argv[3]);
  mpz_t p, y;
  mpz_inits(p, y, NULL);
  mpz_ui_pow_ui(p, 2, W);
  mpz_sub_ui(p, p, C);
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 20261017u);
  double v[256];
  int allok = 1;
  for (int r = 0; r < reps && r < 256; r++) {
    mpz_urandomm(y, rs, p);
    int ok;
#if TOPB == 0
    v[r] = u128 ? chain_u128(nsq, y, p, &ok) : chain(nsq, y, p, &ok);
#else
    v[r] = chain(nsq, y, p, &ok);
#endif
    allok &= ok;
  }
  qsort(v, reps, sizeof(double), cmpd);
  printf("{\"mode\":\"%s\",\"W\":%d,\"C\":%d,\"cpu\":%d,\"nsq\":%ld,\"reps\":%d,\"sqr_min_ns\":%.2f,\"sqr_med_ns\":%.2f,"
         "\"verified\":%s,\"gmp\":\"%s\"}\n",
         u128 ? "u128" : "mpn", W, C, first, nsq, reps, v[0], v[reps / 2], allok ? "true" : "false", gmp_version);
  return allok ? 0 : 1;
}
