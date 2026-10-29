// The host CPU's rates for the routes a prover could move off the card: AMX int8 tile dot products (TDPBSSD, the k64
// and k32 forms: a k32 one writes one 32-bit word a k32 partial dot, ssn_1l's committed word), AVX-512 VNNI
// (VPDPBUSD) and AVX-512 32-bit adds (VPADDD: a committed word from a precomputed partial dot plus its noise, one add
// a word) on registers and streamed from memory. host.sh runs it on 1 thread and on every thread, and times SHA-512
// (the hm96 rows' hash) with `openssl speed`.
//   gcc -O3 -march=sapphirerapids -fopenmp host.c -o host
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

struct tilecfg {
  uint8_t palette_id, start_row, reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __attribute__((packed));

// TDPBSSD C[16x16 s32] += A[16 x kb s8] . B[kb/4 x 64 B]: kb = 64 (16,384 MACs, 256 words) or 32 (8,192 MACs, 256
// words). Four independent accumulators so the dependent-chain latency is hidden.
static int amx_cfg(int kb) {
  struct tilecfg c;
  memset(&c, 0, sizeof c);
  c.palette_id = 1;
  for (int t = 0; t < 4; t++) c.rows[t] = 16, c.colsb[t] = 64;            // C tiles: 16 rows x 16 s32
  for (int t = 4; t < 6; t++) c.rows[t] = 16, c.colsb[t] = kb;            // A tiles: 16 rows x kb bytes
  for (int t = 6; t < 8; t++) c.rows[t] = kb / 4, c.colsb[t] = 64;        // B tiles: kb/4 rows x 64 bytes
  _tile_loadconfig(&c);
  return 0;
}

static double amx_run(int kb, long iters, uint32_t* sink) {
  static int8_t A[2][16 * 64] __attribute__((aligned(64))), B[2][16 * 64] __attribute__((aligned(64)));
  static int32_t C[16 * 16] __attribute__((aligned(64)));
  for (int i = 0; i < 16 * 64; i++) A[0][i] = (int8_t)(i * 7), A[1][i] = (int8_t)(i * 13), B[0][i] = (int8_t)(i * 3),
                                    B[1][i] = (int8_t)(i * 5);
  amx_cfg(kb);
  _tile_zero(0);
  _tile_zero(1);
  _tile_zero(2);
  _tile_zero(3);
  _tile_loadd(4, A[0], 64);
  _tile_loadd(5, A[1], 64);
  _tile_loadd(6, B[0], 64);
  _tile_loadd(7, B[1], 64);
  double t0 = now();
  for (long i = 0; i < iters; i++) {
    _tile_dpbssd(0, 4, 6);
    _tile_dpbssd(1, 4, 7);
    _tile_dpbssd(2, 5, 6);
    _tile_dpbssd(3, 5, 7);
  }
  double t = now() - t0;
  _tile_stored(0, C, 64);
  *sink ^= (uint32_t)C[3];
  _tile_stored(3, C, 64);
  *sink ^= (uint32_t)C[17];
  _tile_release();
  return t;
}

static double vnni_run(long iters, uint32_t* sink) {
  __m512i acc[12], a = _mm512_set1_epi32(0x01020304), b = _mm512_set1_epi32(0x05060708);
  for (int u = 0; u < 12; u++) acc[u] = _mm512_set1_epi32(u);
  double t0 = now();
  for (long i = 0; i < iters; i++) {
#pragma GCC unroll 12
    for (int u = 0; u < 12; u++) acc[u] = _mm512_dpbusd_epi32(acc[u], a, b);
    __asm__ volatile("" : "+v"(a));
  }
  double t = now() - t0;
  for (int u = 1; u < 12; u++) acc[0] = _mm512_xor_si512(acc[0], acc[u]);
  *sink ^= (uint32_t)_mm512_reduce_add_epi32(acc[0]);
  return t;
}

static double vpadd_run(long iters, uint32_t* sink) {
  __m512i acc[12], x = _mm512_set1_epi32(3);
  for (int u = 0; u < 12; u++) acc[u] = _mm512_set1_epi32(u);
  double t0 = now();
  for (long i = 0; i < iters; i++) {
#pragma GCC unroll 12
    for (int u = 0; u < 12; u++) acc[u] = _mm512_add_epi32(acc[u], x);
    __asm__ volatile("" : "+v"(x));
  }
  double t = now() - t0;
  for (int u = 1; u < 12; u++) acc[0] = _mm512_xor_si512(acc[0], acc[u]);
  *sink ^= (uint32_t)_mm512_reduce_add_epi32(acc[0]);
  return t;
}

// words[i] = D[i] + X[i mod 256]: a committed word from a precomputed partial dot streamed from memory, written back.
static double vpadd_stream(size_t n, int reps, uint32_t* sink) {
  int32_t* D = aligned_alloc(64, n * 4);
  int32_t X[256] __attribute__((aligned(64)));
  for (size_t i = 0; i < n; i++) D[i] = (int32_t)(i * 2654435761u);
  for (int i = 0; i < 256; i++) X[i] = i;
  double t0 = now();
  for (int r = 0; r < reps; r++)
    for (size_t i = 0; i < n; i += 16) {
      __m512i d = _mm512_load_si512(D + i), x = _mm512_load_si512(X + (i & 255));
      _mm512_store_si512(D + i, _mm512_add_epi32(d, x));
    }
  double t = now() - t0;
  *sink ^= (uint32_t)D[n / 3];
  free(D);
  return t;
}

int main(int argc, char** argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 1;
  double secs = argc > 2 ? atof(argv[2]) : 1.0;
  uint32_t sinks[256] = {0};
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    fprintf(stderr, "host: no AMX permission\n");
    return 2;
  }
  omp_set_num_threads(threads);
  printf("{\"threads\": %d", threads);
  // calibrate each test's iterations on one thread to about `secs` seconds, then run all threads at once
  struct {
    const char* name;
    double per_iter_work;  // the unit counted per iteration
    const char* unit;
    int kind;
  } T[] = {
      {"amx_k64", 4 * 16384.0, "mac", 0}, {"amx_k64_words", 4 * 256.0, "word", 0},
      {"amx_k32", 4 * 8192.0, "mac", 1},  {"amx_k32_words", 4 * 256.0, "word", 1},
      {"vnni", 12 * 64.0, "mac", 2}, {"vpaddd_reg", 12 * 16.0, "word", 3},
  };
  for (int k = 0; k < 6; k += 1) {
    int kind = T[k].kind;
    if (k == 1 || k == 3) continue;  // the words rows share the mac row's run
    long it = 1000;
    for (;;) {
      double t = kind == 0 ? amx_run(64, it, sinks) : kind == 1 ? amx_run(32, it, sinks)
                 : kind == 2 ? vnni_run(it, sinks) : vpadd_run(it, sinks);
      if (t > 0.1) {
        it = (long)(it * secs / t);
        break;
      }
      it *= 4;
    }
    double t0 = now(), tmax = 0;
#pragma omp parallel reduction(max : tmax)
    {
      if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) abort();
      int id = omp_get_thread_num();
      double t = kind == 0 ? amx_run(64, it, sinks + id) : kind == 1 ? amx_run(32, it, sinks + id)
                 : kind == 2 ? vnni_run(it, sinks + id) : vpadd_run(it, sinks + id);
      tmax = t;
    }
    double wall = now() - t0;
    double work = T[k].per_iter_work * it * threads;
    printf(", \"%s_per_s\": %.6g, \"%s_wall_s\": %.4f", T[k].name, work / wall, T[k].name, wall);
    if (kind <= 1) printf(", \"%s_per_s\": %.6g", T[k + 1].name, T[k + 1].per_iter_work * it * threads / wall);
  }
  {  // streamed adds: 64 MiB a thread, out of cache
    size_t n = 16u << 20;
    int reps = 8;
    double t0 = now();
#pragma omp parallel
    { vpadd_stream(n, reps, sinks + omp_get_thread_num()); }
    double wall = now() - t0;
    printf(", \"vpaddd_stream_words_per_s\": %.6g", (double)n * reps * threads / wall);
  }
  uint32_t s = 0;
  for (int i = 0; i < 256; i++) s ^= sinks[i];
  printf(", \"sink\": %u}\n", s);
  return 0;
}
