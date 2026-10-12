// The CPUs' read bandwidth from DRAM: OpenMP threads sum a buffer of GIB GiB of 64-bit words, best of REPS passes.  The
// ceiling a tile kernel that streams its weights meets.   usage: stream_read GIB REPS
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: stream_read GIB REPS\n"); return 2; }
  size_t n = (size_t)(atof(argv[1]) * (1 << 30)) / 256 * 32;
  int reps = atoi(argv[2]), threads = 0;
  int64_t *buf = aligned_alloc(64, n * 8);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) buf[i] = (int64_t)i;
  double best = 1e30;
  int64_t fold = 0;
  for (int r = 0; r < reps + 1; r++) {
    double t0 = now();
#pragma omp parallel reduction(+ : fold)
    {
      __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
#pragma omp single
      threads = omp_get_num_threads();
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; i += 32) {
        a0 = _mm512_add_epi64(a0, _mm512_load_si512(buf + i));
        a1 = _mm512_add_epi64(a1, _mm512_load_si512(buf + i + 8));
        a2 = _mm512_add_epi64(a2, _mm512_load_si512(buf + i + 16));
        a3 = _mm512_add_epi64(a3, _mm512_load_si512(buf + i + 24));
      }
      fold += _mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3)));
    }
    double dt = now() - t0;
    if (r > 0 && dt < best) best = dt;
  }
  printf("{\"kernel\": \"stream_read\", \"bytes\": %zu, \"threads\": %d, \"best_s\": %.6f, \"bytes_per_s\": %.4e, \"fold\": %lld}\n",
         n * 8, threads, best, n * 8 / best, (long long)fold);
  return 0;
}
