// M5's host-compute probe: one core's AMX throughput (TDPBSSD int8 and TDPBF16PS bf16), four independent
// accumulator tiles so latency is hidden. The model's host MAC cap is cores x the best one-core rate (a one-core
// turbo is the fastest a core runs, so the product bounds the all-core rate from above).
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

typedef struct {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __attribute__((packed)) tilecfg;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int8_t a8[16 * 64] __attribute__((aligned(64))), b8[16 * 64] __attribute__((aligned(64)));
static uint16_t a16[16 * 32] __attribute__((aligned(64))), b16[16 * 32] __attribute__((aligned(64)));
static int32_t out[16 * 16] __attribute__((aligned(64)));

int main(void) {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    printf("{\"probe\":\"amx\",\"error\":\"arch_prctl\"}\n");
    return 1;
  }
  tilecfg c = {0};
  c.palette_id = 1;
  for (int i = 0; i < 8; i++) { c.rows[i] = 16; c.colsb[i] = 64; }
  _tile_loadconfig(&c);
  for (int i = 0; i < 16 * 64; i++) { a8[i] = (int8_t)(i % 7 - 3); b8[i] = (int8_t)(i % 5 - 2); }
  for (int i = 0; i < 16 * 32; i++) { a16[i] = 0x3f80; b16[i] = 0x3f80; }
  const long iters = 20000000;
  for (int pass = 0; pass < 2; pass++) {
    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
    double t0, t1;
    long macs;
    if (pass == 0) {
      _tile_loadd(4, a8, 64); _tile_loadd(5, b8, 64); _tile_loadd(6, b8, 64); _tile_loadd(7, a8, 64);
      t0 = now();
      for (long i = 0; i < iters; i++) {
        _tile_dpbssd(0, 4, 5); _tile_dpbssd(1, 4, 7); _tile_dpbssd(2, 6, 5); _tile_dpbssd(3, 6, 7);
      }
      t1 = now();
      macs = iters * 4L * 16 * 16 * 64;
    } else {
      _tile_loadd(4, a16, 64); _tile_loadd(5, b16, 64); _tile_loadd(6, b16, 64); _tile_loadd(7, a16, 64);
      t0 = now();
      for (long i = 0; i < iters; i++) {
        _tile_dpbf16ps(0, 4, 5); _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(2, 6, 5); _tile_dpbf16ps(3, 6, 7);
      }
      t1 = now();
      macs = iters * 4L * 16 * 16 * 32;
    }
    _tile_stored(0, out, 64);
    printf("{\"probe\":\"amx\",\"op\":\"%s\",\"macs\":%ld,\"seconds\":%.4f,\"macs_per_s_one_core\":%.4e,"
           "\"check\":%d,\"cores_online\":%ld}\n",
           pass == 0 ? "tdpbssd-int8" : "tdpbf16ps-bf16", macs, t1 - t0, macs / (t1 - t0), out[0],
           sysconf(_SC_NPROCESSORS_ONLN));
  }
  _tile_release();
  return 0;
}
