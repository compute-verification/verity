// A lane filler's op mix on the host's AVX-512: chains of LOP3 (y ^ k1 ^ k2) and IMAD (y·a + c) alternating, in Z/2^32,
// the ¾ filler's mix on the B200 (`[filler] mix = { ALU = 1, IMAD = 1 }`, its inputs public functions of the coins and
// the tile).  Each zmm holds 16 lanes' chains and each thread runs V of them, enough to cover vpmulld's latency.  On the
// host one IMAD is vpmulld + vpaddd and one LOP3 is vpternlogd; the rate counts each LOP3 and each IMAD as one GPU lane
// instruction, so it compares with the GPU's lane issue (A2: 4 warp instructions a SM a clock, 32 lanes each).
//
//   gate:  lane_chain gate                AVX-512 chains equal the scalar chain lane for lane (exit 1 on a mismatch)
//   bench: lane_chain bench STEPS REPS    OpenMP threads (OMP_NUM_THREADS, OMP_PLACES); one JSON line
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V 16
#define NK 64
#define LANES (16 * V)

static uint32_t K1[NK], K2[NK], A[NK], C[NK];

static uint32_t lcg(uint64_t *s) {
  *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
  return (uint32_t)(*s >> 32);
}

static void keys(uint64_t seed) {
  uint64_t s = seed;
  for (int i = 0; i < NK; i++) { K1[i] = lcg(&s); K2[i] = lcg(&s); A[i] = lcg(&s) | 1; C[i] = lcg(&s); }
}

static uint32_t scalar_chain(uint32_t y, long steps) {
  for (long t = 0; t < steps; t++) {
    int i = (int)(t & (NK - 1));
    y = y ^ K1[i] ^ K2[i];
    y = y * A[i] + C[i];
  }
  return y;
}

static void avx_chains(uint32_t *y, long steps) {
  __m512i v[V];
  for (int j = 0; j < V; j++) v[j] = _mm512_loadu_si512(y + 16 * j);
  for (long t = 0; t < steps; t++) {
    int i = (int)(t & (NK - 1));
    __m512i k1 = _mm512_set1_epi32((int)K1[i]), k2 = _mm512_set1_epi32((int)K2[i]);
    __m512i a = _mm512_set1_epi32((int)A[i]), c = _mm512_set1_epi32((int)C[i]);
    for (int j = 0; j < V; j++) {
      v[j] = _mm512_ternarylogic_epi32(v[j], k1, k2, 0x96);
      v[j] = _mm512_add_epi32(_mm512_mullo_epi32(v[j], a), c);
    }
  }
  for (int j = 0; j < V; j++) _mm512_storeu_si512(y + 16 * j, v[j]);
}

static int gate(void) {
  static const long steps[] = {1, 2, 63, 64, 65, 1000, 4097};
  int cases = 0;
  for (uint64_t seed = 1; seed <= 3; seed++) {
    keys(seed);
    for (size_t n = 0; n < sizeof steps / sizeof steps[0]; n++) {
      uint32_t y[LANES], want[LANES];
      uint64_t s = seed * 0x9E3779B97F4A7C15ULL + n;
      for (int l = 0; l < LANES; l++) y[l] = lcg(&s);
      y[0] = 0; y[1] = 0xFFFFFFFFu; y[2] = 0x80000000u;
      for (int l = 0; l < LANES; l++) want[l] = scalar_chain(y[l], steps[n]);
      avx_chains(y, steps[n]);
      for (int l = 0; l < LANES; l++) {
        if (y[l] != want[l]) {
          printf("{\"gate\": \"lane chain avx512 equals scalar\", \"ok\": false, \"seed\": %llu, \"steps\": %ld, "
                 "\"lane\": %d, \"got\": %u, \"want\": %u}\n", (unsigned long long)seed, steps[n], l, y[l], want[l]);
          return 1;
        }
      }
      cases++;
    }
  }
  printf("{\"gate\": \"lane chain avx512 equals scalar\", \"ok\": true, \"cases\": %d, \"lanes\": %d}\n", cases, LANES);
  return 0;
}

static int bench(long steps, int reps) {
  keys(7);
  int threads = 0;
  uint32_t check = 0;
  double t0 = 0, t1 = 0;
#pragma omp parallel reduction(^ : check)
  {
    uint32_t y[LANES];
    uint64_t s = 0x5EED0000ULL + (uint64_t)omp_get_thread_num();
    for (int l = 0; l < LANES; l++) y[l] = lcg(&s);
    avx_chains(y, steps);
#pragma omp barrier
#pragma omp master
    { threads = omp_get_num_threads(); t0 = omp_get_wtime(); }
#pragma omp barrier
    for (int r = 0; r < reps; r++) avx_chains(y, steps);
#pragma omp barrier
#pragma omp master
    t1 = omp_get_wtime();
    for (int l = 0; l < LANES; l++) check ^= y[l];
  }
  double sec = t1 - t0, instr = 2.0 * (double)steps * reps * LANES * threads;
  const char *places = getenv("OMP_PLACES");
  printf("{\"kernel\": \"lane_chain\", \"mix\": \"LOP3+IMAD\", \"threads\": %d, \"places\": \"%s\", \"V\": %d, "
         "\"steps\": %ld, \"reps\": %d, \"seconds\": %.6f, \"lane_instr_per_s\": %.6e, \"per_thread\": %.6e, "
         "\"checksum\": %u}\n", threads, places ? places : "", V, steps, reps, sec, instr / sec, instr / sec / threads,
         check);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "gate")) return gate();
  if (argc >= 4 && !strcmp(argv[1], "bench")) return bench(atol(argv[2]), atoi(argv[3]));
  fprintf(stderr, "usage: lane_chain gate | lane_chain bench STEPS REPS\n");
  return 2;
}
