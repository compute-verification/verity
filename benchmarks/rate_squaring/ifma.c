// Squarings mod p = 2^512 - 569 on a host core with AVX-512 IFMA (vpmadd52luq / vpmadd52huq), eight words a vector and
// G vectors interleaved a thread: the host CPUs' share h of the adversary's squaring rate.
//
// A word is ten 52-bit limbs (lazily below 2^520). The square takes each unordered pair's low and high halves once
// (90 IFMA), doubles the 19 columns and adds the 10 diagonal squares (20 IFMA). The high half is normalized alone
// (it is below 2^520, as the low columns' carries are left out of it), then folds into the low columns by
// 2^520 = 569 * 2^8 (mod p): 20 IFMA. The low columns are then normalized, and their carry out of 2^520 folds into
// limb 0 the same way. Limb 0's carry into limbs 1 and 2 ends the square; a carry past limb 2 (probability below
// 2^-100) takes a full pass.
//
//   ifma gate K N          N words (a multiple of 8 G) from a fixed seed, K squarings: "tag K 512 569 x0 xK", canonical
//   ifma time K THREADS    THREADS threads, one a CPU of the affinity mask, each squaring 8 G words K times: JSON
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef G
#define G 4
#endif
#define NL 10
typedef __m512i V;
typedef uint64_t u64;

static inline V lo52(V acc, V a, V b) { return _mm512_madd52lo_epu64(acc, a, b); }
static inline V hi52(V acc, V a, V b) { return _mm512_madd52hi_epu64(acc, a, b); }

static inline __attribute__((always_inline)) void sqr(V (*x)[NL]) {
  const V M = _mm512_set1_epi64((1ull << 52) - 1), K = _mm512_set1_epi64(569 << 8), Z = _mm512_setzero_si512();
  V t[G][2 * NL];
#pragma GCC unroll 64
  for (int g = 0; g < G; ++g)
#pragma GCC unroll 64
    for (int k = 0; k < 2 * NL; ++k) t[g][k] = Z;
#pragma GCC unroll 64
  for (int i = 0; i < NL; ++i)
#pragma GCC unroll 64
    for (int j = i + 1; j < NL; ++j)
#pragma GCC unroll 64
      for (int g = 0; g < G; ++g) {
        t[g][i + j] = lo52(t[g][i + j], x[g][i], x[g][j]);
        t[g][i + j + 1] = hi52(t[g][i + j + 1], x[g][i], x[g][j]);
      }
#pragma GCC unroll 64
  for (int g = 0; g < G; ++g) {
#pragma GCC unroll 64
    for (int k = 1; k < 2 * NL; ++k) t[g][k] = _mm512_add_epi64(t[g][k], t[g][k]);
#pragma GCC unroll 64
    for (int i = 0; i < NL; ++i) {
      t[g][2 * i] = lo52(t[g][2 * i], x[g][i], x[g][i]);
      t[g][2 * i + 1] = hi52(t[g][2 * i + 1], x[g][i], x[g][i]);
    }
    V c = Z;
#pragma GCC unroll 64
    for (int k = NL; k < 2 * NL; ++k) {
      const V v = _mm512_add_epi64(t[g][k], c);
      c = _mm512_srli_epi64(v, 52);
      t[g][k] = _mm512_and_si512(v, M);
    }
    V e = Z;
#pragma GCC unroll 64
    for (int j = 0; j < NL; ++j) {
      t[g][j] = lo52(t[g][j], t[g][NL + j], K);
      if (j + 1 < NL) t[g][j + 1] = hi52(t[g][j + 1], t[g][NL + j], K);
      else e = hi52(e, t[g][NL + j], K);
    }
    c = Z;
#pragma GCC unroll 64
    for (int k = 0; k < NL; ++k) {
      const V v = _mm512_add_epi64(t[g][k], c);
      c = _mm512_srli_epi64(v, 52);
      x[g][k] = _mm512_and_si512(v, M);
    }
    x[g][0] = lo52(x[g][0], _mm512_add_epi64(c, e), K);
    c = _mm512_srli_epi64(x[g][0], 52);
    x[g][0] = _mm512_and_si512(x[g][0], M);
    x[g][1] = _mm512_add_epi64(x[g][1], c);
    c = _mm512_srli_epi64(x[g][1], 52);
    x[g][1] = _mm512_and_si512(x[g][1], M);
    x[g][2] = _mm512_add_epi64(x[g][2], c);
    if (__builtin_expect(_mm512_cmpgt_epu64_mask(x[g][2], M) != 0, 0)) {
      do {
        c = Z;
        for (int k = 0; k < NL; ++k) {
          const V v = _mm512_add_epi64(x[g][k], c);
          c = _mm512_srli_epi64(v, 52);
          x[g][k] = _mm512_and_si512(v, M);
        }
        x[g][0] = lo52(x[g][0], c, K);
      } while (_mm512_cmpgt_epu64_mask(x[g][0], M) != 0);
    }
  }
}

static void chain(V (*x)[NL], long K) {
  for (long r = 0; r < K; ++r) sqr(x);
}

static u64 splitmix(u64 *s) {
  u64 z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Word w of a group's lane l: eight u64 words (below 2^512) into ten 52-bit limbs, and back mod p.
static void to_limbs(const u64 w[8], u64 l[NL]) {
  for (int k = 0; k < NL; ++k) {
    const int b = 52 * k, q = b / 64, r = b % 64;
    unsigned __int128 v = w[q] >> r;
    if (r > 12 && q + 1 < 8) v |= (unsigned __int128)w[q + 1] << (64 - r);
    l[k] = (u64)v & ((1ull << 52) - 1);
  }
}

static void canon_hex(const u64 l[NL], char *out) {
  u64 w[9] = {0};
  for (int k = 0; k < NL; ++k) {
    const int b = 52 * k, q = b / 64, r = b % 64;
    w[q] |= l[k] << r;
    if (r > 12) w[q + 1] |= l[k] >> (64 - r);
  }
  u64 add = 569 * w[8];
  for (int pass = 0; pass < 3 && add; ++pass) {
    unsigned __int128 c = add;
    for (int q = 0; q < 8; ++q) {
      c += w[q];
      w[q] = (u64)c;
      c >>= 64;
    }
    add = c ? 569 : 0;
  }
  u64 y[8];
  unsigned __int128 c = 569;
  for (int q = 0; q < 8; ++q) {
    c += w[q];
    y[q] = (u64)c;
    c >>= 64;
  }
  const u64 *z = c ? y : w;
  for (int q = 7; q >= 0; --q) out += sprintf(out, "%016llx", (unsigned long long)z[q]);
}

static void fill(V (*x)[NL], u64 *seed) {
  u64 lanes[8][NL];
  for (int g = 0; g < G; ++g) {
    for (int l = 0; l < 8; ++l) {
      u64 w[8];
      for (int q = 0; q < 8; ++q) w[q] = splitmix(seed);
      to_limbs(w, lanes[l]);
    }
    for (int k = 0; k < NL; ++k) {
      u64 v[8];
      for (int l = 0; l < 8; ++l) v[l] = lanes[l][k];
      x[g][k] = _mm512_loadu_si512(v);
    }
  }
}

static void lane(V (*x)[NL], int g, int l, u64 out[NL]) {
  for (int k = 0; k < NL; ++k) {
    u64 v[8];
    _mm512_storeu_si512(v, x[g][k]);
    out[k] = v[l];
  }
}

typedef struct {
  int cpu;
  long K;
  double seconds;
} Job;

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void *run(void *arg) {
  Job *j = arg;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(j->cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  V x[G][NL];
  u64 seed = 0x5eed512ull + j->cpu;
  fill(x, &seed);
  chain(x, j->K / 10 + 1);
  const double t0 = now();
  chain(x, j->K);
  j->seconds = now() - t0;
  u64 l[NL];
  lane(x, 0, 0, l);
  volatile u64 sink = l[0];
  (void)sink;
  return NULL;
}

int main(int argc, char **argv) {
  if (argc == 4 && !strcmp(argv[1], "gate")) {
    const long K = atol(argv[2]);
    const int n = atoi(argv[3]);
    u64 seed = 0x5eed512ull;
    for (int base = 0; base < n; base += 8 * G) {
      V x[G][NL], x0[G][NL];
      fill(x, &seed);
      memcpy(x0, x, sizeof x);
      chain(x, K);
      for (int g = 0; g < G; ++g)
        for (int l = 0; l < 8; ++l) {
          u64 a[NL], b[NL];
          char ha[200], hb[200];
          lane(x0, g, l, a);
          lane(x, g, l, b);
          canon_hex(a, ha);
          canon_hex(b, hb);
          printf("ifma%d %ld 512 569 %s %s\n", G, K, ha, hb);
        }
    }
    return 0;
  }
  if (argc == 4 && !strcmp(argv[1], "time")) {
    const long K = atol(argv[2]);
    int T = atoi(argv[3]);
    cpu_set_t mask;
    sched_getaffinity(0, sizeof mask, &mask);
    int cpus[CPU_SETSIZE], nc = 0;
    for (int c = 0; c < CPU_SETSIZE; ++c)
      if (CPU_ISSET(c, &mask)) cpus[nc++] = c;
    if (T > nc) T = nc;
    pthread_t th[CPU_SETSIZE];
    Job jobs[CPU_SETSIZE];
    for (int i = 0; i < T; ++i) {
      jobs[i] = (Job){cpus[i], K, 0};
      pthread_create(&th[i], NULL, run, &jobs[i]);
    }
    double worst = 0, sum_rate = 0;
    for (int i = 0; i < T; ++i) {
      pthread_join(th[i], NULL);
      if (jobs[i].seconds > worst) worst = jobs[i].seconds;
      sum_rate += 8.0 * G * K / jobs[i].seconds;
    }
    printf("{\"g\": %d, \"K\": %ld, \"threads\": %d, \"first_cpu\": %d, \"sq_per_s_sum\": %.4e, \"sq_per_s_per_thread\": %.4e, "
           "\"sq_per_s_at_slowest\": %.4e}\n",
           G, K, T, cpus[0], sum_rate, sum_rate / T, 8.0 * G * K * T / worst);
    return 0;
  }
  return 2;
}
