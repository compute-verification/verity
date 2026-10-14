// The relay's compute route on the host: E4M3 dot products of sm_120's group (32 products) on AMX-BF16 tiles and
// AVX-512 BF16 lanes, gated against the exact sum.
//
//   cpu_e4m3 [--threads T] [--seconds S] [--spread D] [--gate-groups N]
//
// An E4M3 value is exact in BF16 and a product of two exact in FP32.  While a group's products span at most D <= 11
// binades, their sum has at most 8 + 11 + 5 bits and is exact in FP32, whatever the order or rounding mode of the
// adder.  On such a group sm_120's 26-bit truncating group sum (silicon.models.group_sum, width 26) drops no product
// bits unless the accumulator sits more than 18 - D binades above the products, so the device's step is a function of
// the exact sum and the accumulator: the fast path of an exact emulation.  This program measures that fast path only;
// the per-group accumulator step (one per 32 MACs) and the fallback for wider groups are not in it.
//
// Gate: every AMX and AVX-512 result over random groups of spread D equals the exact sum (computed in double, exact),
// bit for bit; exit 3 otherwise.  Then each thread runs, for `seconds`:
//   amx_acc        TDPBF16PS accumulating in tiles (the tile unit's peak: 16 x 16 outputs x 32 products)
//   amx_group      TDPBF16PS on zeroed tiles, every group's 256 sums stored (what the step needs)
//   avx512_bf16    VDPBF16PS, 16 outputs a vector, a broadcast activation pair (batch 1 without padding)
// Output: JSON lines, MAC/s summed over the threads.
#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (uint32_t)(rng >> 16);
}

static float e4m3(uint8_t b) {
  int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  float v = e ? ldexpf((float)(8 + m), e - 10) : ldexpf((float)m, -9);
  return s ? -v : v;
}
static uint16_t bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);  // exact: an E4M3 value has at most 4 significant bits
}
// a random normal E4M3 byte with exponent field in [lo, lo + span], or zero one time in 16
static uint8_t draw(int lo, int span) {
  if ((rnd() & 15) == 0) return 0;
  int e = lo + (int)(rnd() % (unsigned)(span + 1));
  return (uint8_t)(((rnd() & 1) << 7) | (e << 3) | (rnd() & 7));
}

typedef struct {
  uint8_t palette, start_row, reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} tilecfg;

static void tiles_on(void) {
  tilecfg c;
  memset(&c, 0, sizeof c);
  c.palette = 1;
  for (int t = 0; t < 8; t++) c.rows[t] = 16, c.colsb[t] = 64;
  _tile_loadconfig(&c);
}

// A: 16 x 32 bf16 (row m: k = 0..31).  B packed for TDPBF16PS: 16 rows x 32 bf16, row r = (B[2r][n], B[2r+1][n]) for
// n = 0..15.  C: 16 x 16 fp32.
static int gate_amx(int da, int db, int groups) {
  uint16_t A[16 * 32] __attribute__((aligned(64))), Bp[16 * 32] __attribute__((aligned(64)));
  float C[256] __attribute__((aligned(64)));
  float a[16][32], b[32][16];
  for (int g = 0; g < groups; g++) {
    for (int m = 0; m < 16; m++)
      for (int k = 0; k < 32; k++) a[m][k] = e4m3(draw(1, da)), A[m * 32 + k] = bf16(a[m][k]);
    for (int k = 0; k < 32; k++)
      for (int n = 0; n < 16; n++) b[k][n] = e4m3(draw(15 - 1 - db, db));
    for (int r = 0; r < 16; r++)
      for (int n = 0; n < 16; n++) Bp[r * 32 + 2 * n] = bf16(b[2 * r][n]), Bp[r * 32 + 2 * n + 1] = bf16(b[2 * r + 1][n]);
    _tile_zero(0);
    _tile_loadd(4, A, 64);
    _tile_loadd(6, Bp, 64);
    _tile_dpbf16ps(0, 4, 6);
    _tile_stored(0, C, 64);
    for (int m = 0; m < 16; m++)
      for (int n = 0; n < 16; n++) {
        double s = 0;
        for (int k = 0; k < 32; k++) s += (double)a[m][k] * b[k][n];
        if ((float)s != (double)s || C[m * 16 + n] != (float)s) {
          printf("{\"gate\": \"amx bf16 exact group sum\", \"ok\": false, \"m\": %d, \"n\": %d, \"got\": %.9g, \"want\": %.17g}\n",
                 m, n, C[m * 16 + n], s);
          return 0;
        }
      }
  }
  return 1;
}

// 16 outputs a vector: lane n accumulates sum_k a[k] * b[k][n] from pairs (a[2j], a[2j+1]) x (b[2j][n], b[2j+1][n])
static int gate_avx(int da, int db, int groups) {
  float a[32], b[32][16], out[16];
  uint16_t W[16][32] __attribute__((aligned(64)));
  for (int g = 0; g < groups; g++) {
    for (int k = 0; k < 32; k++) a[k] = e4m3(draw(1, da));
    for (int k = 0; k < 32; k++)
      for (int n = 0; n < 16; n++) b[k][n] = e4m3(draw(15 - 1 - db, db));
    for (int j = 0; j < 16; j++)
      for (int n = 0; n < 16; n++) W[j][2 * n] = bf16(b[2 * j][n]), W[j][2 * n + 1] = bf16(b[2 * j + 1][n]);
    __m512 acc = _mm512_setzero_ps();
    for (int j = 0; j < 16; j++) {
      uint32_t pair = (uint32_t)bf16(a[2 * j]) | ((uint32_t)bf16(a[2 * j + 1]) << 16);
      __m512bh x = (__m512bh)_mm512_set1_epi32((int)pair);
      __m512bh w = (__m512bh)_mm512_load_si512((const void*)W[j]);
      acc = _mm512_dpbf16_ps(acc, x, w);
    }
    _mm512_storeu_ps(out, acc);
    for (int n = 0; n < 16; n++) {
      double s = 0;
      for (int k = 0; k < 32; k++) s += (double)a[k] * b[k][n];
      if ((float)s != (double)s || out[n] != (float)s) {
        printf("{\"gate\": \"avx512 bf16 exact group sum\", \"ok\": false, \"n\": %d, \"got\": %.9g, \"want\": %.17g}\n", n,
               out[n], s);
        return 0;
      }
    }
  }
  return 1;
}

typedef struct {
  int kind;
  double seconds, macs;
} job;

static uint16_t TA[2][16 * 32] __attribute__((aligned(64))), TB[2][16 * 32] __attribute__((aligned(64)));

static void* worker(void* p) {
  job* j = (job*)p;
  double t0 = now_s(), t;
  uint64_t iters = 0;
  if (j->kind < 2) {
    tiles_on();
    float C[4][256] __attribute__((aligned(64)));
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);
    do {
      for (int r = 0; r < 64; r++) {
        if (j->kind == 1) {
          _tile_zero(0);
          _tile_zero(1);
          _tile_zero(2);
          _tile_zero(3);
        }
        _tile_loadd(4, TA[0], 64);
        _tile_loadd(5, TA[1], 64);
        _tile_loadd(6, TB[0], 64);
        _tile_loadd(7, TB[1], 64);
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
        if (j->kind == 1) {
          _tile_stored(0, C[0], 64);
          _tile_stored(1, C[1], 64);
          _tile_stored(2, C[2], 64);
          _tile_stored(3, C[3], 64);
        }
      }
      iters += 64;
      t = now_s();
    } while (t - t0 < j->seconds);
    _tile_stored(0, C[0], 64);
    _tile_release();
    j->macs = iters * 4.0 * 16 * 16 * 32 / (t - t0);
  } else {
    __m512 acc[12];
    for (int i = 0; i < 12; i++) acc[i] = _mm512_setzero_ps();
    __m512bh x = (__m512bh)_mm512_set1_epi32(0x3f803f80);
    do {
      for (int r = 0; r < 256; r++)
        for (int i = 0; i < 12; i++)
          acc[i] = _mm512_dpbf16_ps(acc[i], x, (__m512bh)_mm512_load_si512((const void*)&TB[0][(i & 7) * 32]));
      iters += 256;
      t = now_s();
    } while (t - t0 < j->seconds);
    __m512 s = acc[0];
    for (int i = 1; i < 12; i++) s = _mm512_add_ps(s, acc[i]);
    if (_mm512_reduce_add_ps(s) == 1234.5f) printf("#\n");
    j->macs = iters * 12.0 * 32 / (t - t0);
  }
  return NULL;
}

int main(int argc, char** argv) {
  int threads = (int)sysconf(_SC_NPROCESSORS_ONLN), spread = 11, groups = 4000;
  double seconds = 3.0;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--threads")) threads = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--seconds")) seconds = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--spread")) spread = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--gate-groups")) groups = atoi(argv[i + 1]);
    else { fprintf(stderr, "unknown %s\n", argv[i]); return 2; }
  }
  if (spread > 11) { fprintf(stderr, "--spread above 11 isn't exact in FP32\n"); return 2; }
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) { perror("AMX permission"); return 2; }
  tiles_on();
  int da = spread / 2, db = spread - da;
  int ok = gate_amx(da, db, groups) && gate_avx(da, db, groups);
  _tile_release();
  if (!ok) return 3;
  printf("{\"gate\": \"amx and avx512 bf16 group sums equal the exact sum\", \"ok\": true, \"groups\": %d, \"spread\": %d}\n",
         groups, spread);
  fflush(stdout);
  for (int i = 0; i < 16 * 32; i++) TA[0][i] = TA[1][i] = TB[0][i] = TB[1][i] = bf16(e4m3(draw(1, 6)));
  const char* names[3] = {"amx_acc", "amx_group", "avx512_bf16"};
  for (int kind = 0; kind < 3; kind++) {
    pthread_t* th = calloc(threads, sizeof *th);
    job* js = calloc(threads, sizeof *js);
    for (int i = 0; i < threads; i++) js[i].kind = kind, js[i].seconds = seconds, pthread_create(&th[i], 0, worker, &js[i]);
    double total = 0;
    for (int i = 0; i < threads; i++) pthread_join(th[i], 0), total += js[i].macs;
    printf("{\"path\": \"%s\", \"threads\": %d, \"seconds\": %.1f, \"mac_per_s\": %.4g, \"mac_per_s_per_thread\": %.4g}\n",
           names[kind], threads, seconds, total, total / threads);
    fflush(stdout);
    free(th), free(js);
  }
  return 0;
}
