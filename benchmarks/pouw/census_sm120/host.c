// Host census (item 6): what the node's CPUs and disks can do beside the GPUs. Modes:
//   host stream THREADS [TAG]  STREAM-like copy/scale/add/triad over three 512 MiB double arrays (OpenMP), best of 10
//   host amx THREADS SECONDS   AMX tile throughput: TDPBSSD (int8 -> int32, 16x16x64 a tile op) and TDPBF16PS
//                              (bf16 -> fp32, 16x16x32) on four accumulators with operands held in tiles, per thread
//   host disk PATH MIB         write PATH (MIB MiB, fsync), then read it back with O_DIRECT, 4 MiB requests, 1 and 4 readers
// One JSON line a constant (constant, value, unit, method). host.sh runs it under nice 19 with at most 16 threads.
#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void emit(const char* c, double v, const char* unit, const char* method) {
  printf("{\"constant\":\"%s\",\"value\":%.4g,\"unit\":\"%s\",\"method\":\"%s\"}\n", c, v, unit, method);
  fflush(stdout);
}

static int stream(int threads, const char* tag) {
  size_t n = (512ull << 20) / sizeof(double);
  double *a = aligned_alloc(64, n * 8), *b = aligned_alloc(64, n * 8), *c = aligned_alloc(64, n * 8);
  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) a[i] = 1.0, b[i] = 2.0, c[i] = 0.0;
  const char* names[4] = {"copy", "scale", "add", "triad"};
  double bytes[4] = {16.0 * n, 16.0 * n, 24.0 * n, 24.0 * n}, best[4] = {1e30, 1e30, 1e30, 1e30};
  for (int r = 0; r < 10; r++) {
    double t = now();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) c[i] = a[i];
    double t1 = now();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) b[i] = 3.0 * c[i];
    double t2 = now();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
    double t3 = now();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
    double t4 = now(), d[4] = {t1 - t, t2 - t1, t3 - t2, t4 - t3};
    for (int k = 0; k < 4; k++)
      if (d[k] < best[k]) best[k] = d[k];
  }
  char key[96], meth[256];
  for (int k = 0; k < 4; k++) {
    snprintf(key, sizeof key, "host.stream_%s.t%d%s.bytes_per_s", names[k], threads, tag);
    snprintf(meth, sizeof meth, "host.c STREAM-like %s, 3 x 512 MiB double arrays, %d OpenMP threads (host.sh pins them), best of 10, "
             "STREAM byte counting (no write-allocate)%s", names[k], threads, *tag ? "; threads split over both sockets" : "");
    emit(key, bytes[k] / best[k], "B/s", meth);
  }
  int bad = a[1] == 0.0;
  free(a), free(b), free(c);
  return bad;
}

struct __attribute__((packed)) tilecfg {
  uint8_t palette, start_row, rsv[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

static void amx_config(void) {
  struct tilecfg cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.palette = 1;
  for (int t = 0; t < 8; t++) cfg.rows[t] = 16, cfg.colsb[t] = 64;
  _tile_loadconfig(&cfg);
}

static double amx_run(int bf16, double seconds, long* ops_out) {
  static __thread int8_t buf[16 * 64] __attribute__((aligned(64)));
  for (int i = 0; i < 16 * 64; i++) buf[i] = (int8_t)(i * 37 + 11) & 0x3f;
  amx_config();
  for (int t = 0; t < 8; t++) {
    switch (t) {
      case 0: _tile_loadd(0, buf, 64); break;
      case 1: _tile_loadd(1, buf, 64); break;
      case 2: _tile_loadd(2, buf, 64); break;
      case 3: _tile_loadd(3, buf, 64); break;
      case 4: _tile_loadd(4, buf, 64); break;
      case 5: _tile_loadd(5, buf, 64); break;
      case 6: _tile_loadd(6, buf, 64); break;
      case 7: _tile_loadd(7, buf, 64); break;
    }
  }
  long ops = 0;
  double t0 = now(), t = t0;
  while (t - t0 < seconds) {
    for (int k = 0; k < 1024; k++) {
      if (bf16) {
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
      } else {
        _tile_dpbssd(0, 4, 6);
        _tile_dpbssd(1, 4, 7);
        _tile_dpbssd(2, 5, 6);
        _tile_dpbssd(3, 5, 7);
      }
    }
    ops += 4096;
    t = now();
  }
  _tile_stored(0, buf, 64);
  _tile_release();
  *ops_out = ops;
  return t - t0;
}

static int amx(int threads, double seconds) {
  if (syscall(SYS_arch_prctl, 0x1023 /* ARCH_REQ_XCOMP_PERM */, 18 /* XFEATURE_XTILEDATA */)) {
    emit("host.amx.available", 0, "bool", "arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA) refused");
    return 0;
  }
  for (int bf16 = 0; bf16 <= 1; bf16++) {
    int counts[2] = {1, threads};
    for (int ci = 0; ci < (threads > 1 ? 2 : 1); ci++) {
      int nt = counts[ci];
      long total = 0;
      double secs = 0;
      omp_set_num_threads(nt);
#pragma omp parallel reduction(+ : total) reduction(max : secs)
      {
        long ops;
        double s = amx_run(bf16, seconds, &ops);
        total += ops;
        secs = s;
      }
      double macs_per_op = bf16 ? 16 * 16 * 32 : 16 * 16 * 64;
      char key[96], meth[256];
      snprintf(key, sizeof key, "host.amx_%s.t%d.macs_per_s", bf16 ? "bf16" : "int8", nt);
      snprintf(meth, sizeof meth, "host.c %s on 4 accumulator tiles, operands resident in tiles (compute peak, no loads), %d "
               "threads, %.1f s; %.0f MACs a tile op", bf16 ? "TDPBF16PS" : "TDPBSSD", nt, seconds, macs_per_op);
      emit(key, total * macs_per_op / secs, "MAC/s", meth);
    }
  }
  return 0;
}

struct rd {
  int fd;
  off_t off, len;
  char* buf;
};

static void* reader(void* p) {
  struct rd* r = p;
  for (off_t o = r->off; o < r->off + r->len; o += 4 << 20)
    if (pread(r->fd, r->buf, 4 << 20, o) != 4 << 20) return (void*)1;
  return 0;
}

static int disk(const char* path, long mib) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) return perror("open"), 1;
  char* buf;
  if (posix_memalign((void**)&buf, 4096, 4 << 20)) return 1;
  uint64_t s = 88172645463325252ull;
  double t0 = now();
  for (long m = 0; m < mib; m += 4) {
    for (size_t i = 0; i < (4u << 20) / 8; i++) {
      s ^= s << 13, s ^= s >> 7, s ^= s << 17;
      ((uint64_t*)buf)[i] = s;
    }
    if (write(fd, buf, 4 << 20) != 4 << 20) return perror("write"), 1;
  }
  fsync(fd);
  double tw = now() - t0;
  close(fd);
  char key[128], meth[512];
  snprintf(meth, sizeof meth, "host.c: %ld MiB of xorshift data written with write(2) 4 MiB at a time, then fsync, to %s", mib, path);
  emit("host.disk.write_fsync.bytes_per_s", mib * 1048576.0 / tw, "B/s", meth);
  for (int nr = 1; nr <= 4; nr *= 4) {
    fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) return perror("open O_DIRECT"), 1;
    pthread_t th[4];
    struct rd r[4];
    off_t len = (off_t)mib * 1048576 / nr;
    t0 = now();
    for (int i = 0; i < nr; i++) {
      r[i] = (struct rd){fd, i * len, len, 0};
      if (posix_memalign((void**)&r[i].buf, 4096, 4 << 20)) return 1;
      pthread_create(&th[i], 0, reader, &r[i]);
    }
    int bad = 0;
    for (int i = 0; i < nr; i++) {
      void* rv;
      pthread_join(th[i], &rv);
      bad |= rv != 0;
      free(r[i].buf);
    }
    double tr = now() - t0;
    close(fd);
    snprintf(key, sizeof key, "host.disk.read_odirect.r%d.bytes_per_s", nr);
    snprintf(meth, sizeof meth, "host.c: O_DIRECT pread 4 MiB requests, %d reader thread(s) over disjoint ranges, of the %ld MiB file "
             "just written to %s%s", nr, mib, path, bad ? " (short read)" : "");
    emit(key, mib * 1048576.0 / tr, "B/s", meth);
  }
  free(buf);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 3 && !strcmp(argv[1], "stream")) return stream(atoi(argv[2]), argc > 3 ? argv[3] : "");
  if (argc >= 4 && !strcmp(argv[1], "amx")) return amx(atoi(argv[2]), atof(argv[3]));
  if (argc >= 4 && !strcmp(argv[1], "disk")) return disk(argv[2], atol(argv[3]));
  fprintf(stderr, "usage: host stream THREADS | amx THREADS SECONDS | disk PATH MIB\n");
  return 2;
}
