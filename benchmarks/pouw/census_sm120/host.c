// Host census (item 6): what the node's CPUs and disks can do beside the GPUs. Modes:
//   host stream THREADS [TAG]  STREAM-like copy/scale/add/triad over three 512 MiB double arrays (OpenMP), best of 10
//   host amx THREADS SECONDS [TAG [VARIANTS]]
//                              AMX tile throughput: TDPBSSD (int8 -> int32, 16x16x64 a tile op) and TDPBF16PS
//                              (bf16 -> fp32, 16x16x32) on four accumulators, per thread; VARIANTS (default resident)
//                              is any of resident, l1, l2 (where the A and B tiles come from; amx_run)
//   host freq THREADS SECONDS [TAG]  each thread's core clock from a dependent add chain
//   host disk PATH MIB [TAG]   write PATH (MIB MiB, fsync), then read it back with O_DIRECT, 4 MiB requests, 1 and 4 readers
// One JSON line a constant (constant, value, unit, method). host.sh runs it under nice 19 with at most 16 threads;
// host_amx.sh runs amx and freq on every core the run may use.
#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
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
             "STREAM byte counting (no write-allocate)%s", names[k], threads, tag);
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

static double wall(void) {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

// Physical core of a CPU, as package * 65536 + core_id from sysfs (-1 if unreadable).
static long core_of(int cpu) {
  char p[96];
  long v[2] = {-1, -1};
  const char* f[2] = {"physical_package_id", "core_id"};
  for (int i = 0; i < 2; i++) {
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, f[i]);
    FILE* fp = fopen(p, "r");
    if (!fp || fscanf(fp, "%ld", &v[i]) != 1) v[i] = -1;
    if (fp) fclose(fp);
  }
  return v[0] < 0 || v[1] < 0 ? -1 : v[0] * 65536 + v[1];
}

static int cmp_d(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

#define WIN_S 0.5
#define NWIN 64

// Variants: resident keeps every operand in tiles (the compute peak); l1 reloads A0, A1, B0, B1 every step from one
// 4 KiB block (L1 hits); l2 reloads them from a 1 MiB panel walked in order (L2 hits), as a blocked GEMM's inner loop
// does with 8 tiles (2 x 2 accumulators, 4 tile loads a 4 tile ops).
enum { V_RESIDENT, V_L1, V_L2, NVAR };
static const char* VNAME[NVAR] = {"resident", "l1", "l2"};

static void amx_run(int bf16, int var, double t_start, double seconds, long* ops_out, long* win) {
  int8_t* buf = aligned_alloc(4096, 1 << 20);
  for (int i = 0; i < (1 << 20); i++) buf[i] = (int8_t)(i * 37 + 11) & 0x3f;
  amx_config();
  _tile_loadd(0, buf, 64);
  _tile_loadd(1, buf, 64);
  _tile_loadd(2, buf, 64);
  _tile_loadd(3, buf, 64);
  _tile_loadd(4, buf, 64);
  _tile_loadd(5, buf + 1024, 64);
  _tile_loadd(6, buf + 2048, 64);
  _tile_loadd(7, buf + 3072, 64);
  long ops = 0;
  size_t off = 0, span = var == V_L2 ? (1 << 20) : 4096;
  double t = now();
  while (t < t_start) t = now();
  while (t - t_start < seconds) {
    for (int k = 0; k < 1024; k++) {
      if (var != V_RESIDENT) {
        const int8_t* p = buf + off;
        _tile_loadd(4, p, 64);
        _tile_loadd(5, p + 1024, 64);
        _tile_loadd(6, p + 2048, 64);
        _tile_loadd(7, p + 3072, 64);
        off = (off + 4096) & (span - 1);
      }
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
    int w = (int)((t - t_start) / WIN_S);
    if (w < NWIN) win[w] += 4096;
  }
  _tile_stored(0, buf, 64);
  _tile_release();
  free(buf);
  *ops_out = ops;
}

// One JSON line a (type, variant): the aggregate, the per-core rate (aggregate over distinct physical cores), the
// per-thread spread, the clock it implies at 1024 (int8) or 512 (bf16) MACs a core a cycle (one tile op a 16 cycles),
// and the aggregate in WIN_S windows from the common start, with CLOCK_REALTIME bounds to align with other logs.
static int amx(int threads, double seconds, const char* tag, const char* variants) {
  if (syscall(SYS_arch_prctl, 0x1023 /* ARCH_REQ_XCOMP_PERM */, 18 /* XFEATURE_XTILEDATA */)) {
    emit("host.amx.available", 0, "bool", "arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA) refused");
    return 0;
  }
  int nt = threads;
  long* ops = calloc(nt, sizeof *ops);
  long* win = calloc((size_t)nt * NWIN, sizeof *win);
  int* cpu = calloc(nt, sizeof *cpu);
  double* rate = calloc(nt, sizeof *rate);
  omp_set_num_threads(nt);
  for (int var = 0; var < NVAR; var++) {
    if (!strstr(variants, VNAME[var])) continue;
    for (int bf16 = 0; bf16 <= 1; bf16++) {
      memset(win, 0, (size_t)nt * NWIN * sizeof *win);
      double t_start = 0, w0 = 0;
#pragma omp parallel
      {
        int i = omp_get_thread_num();
        cpu[i] = sched_getcpu();
#pragma omp barrier
#pragma omp single
        t_start = now() + 0.05, w0 = wall() + 0.05;
        amx_run(bf16, var, t_start, seconds, &ops[i], &win[(size_t)i * NWIN]);
      }
      double w1 = wall(), mpo = bf16 ? 16 * 16 * 32 : 16 * 16 * 64, total = 0;
      for (int i = 0; i < nt; i++) total += ops[i], rate[i] = ops[i] * mpo / seconds;
      int ncores = 0;
      long* seen = calloc(nt, sizeof *seen);
      for (int i = 0; i < nt; i++) {
        long c = core_of(cpu[i]);
        int dup = 0;
        for (int j = 0; j < ncores; j++) dup |= seen[j] == c;
        if (!dup) seen[ncores++] = c;
      }
      free(seen);
      qsort(rate, nt, sizeof *rate, cmp_d);
      double agg = total * mpo / seconds, med = nt % 2 ? rate[nt / 2] : 0.5 * (rate[nt / 2 - 1] + rate[nt / 2]);
      char key[128], meth[384];
      snprintf(key, sizeof key, "host.amx_%s.%s.t%d%s.macs_per_s", bf16 ? "bf16" : "int8", VNAME[var], nt, tag);
      snprintf(meth, sizeof meth, "host.c %s on 4 accumulator tiles, %s, %d threads on %d physical cores, %.1f s from a "
               "common start; %.0f MACs a tile op", bf16 ? "TDPBF16PS" : "TDPBSSD",
               var == V_RESIDENT ? "operands resident in tiles (compute peak, no loads)"
               : var == V_L1 ? "A and B tiles reloaded each step from one 4 KiB block (L1)"
                             : "A and B tiles reloaded each step from a 1 MiB panel (L2)",
               nt, ncores, seconds, mpo);
      printf("{\"constant\":\"%s\",\"value\":%.4g,\"unit\":\"MAC/s\",\"method\":\"%s\",\"threads\":%d,\"cores\":%d,"
             "\"per_core_macs_per_s\":%.4g,\"per_thread_macs_per_s\":{\"min\":%.4g,\"median\":%.4g,\"max\":%.4g},"
             "\"implied_ghz_per_thread_median\":%.4g,\"t0_unix\":%.3f,\"t1_unix\":%.3f,\"window_s\":%.2f,"
             "\"window_macs_per_s\":[",
             key, agg, meth, nt, ncores, agg / ncores, rate[0], med, rate[nt - 1], med / (bf16 ? 512 : 1024) / 1e9, w0, w1,
             WIN_S);
      int nw = (int)(seconds / WIN_S);
      for (int w = 0; w < nw && w < NWIN; w++) {
        double s = 0;
        for (int i = 0; i < nt; i++) s += win[(size_t)i * NWIN + w];
        printf("%s%.4g", w ? "," : "", s * mpo / WIN_S);
      }
      printf("],\"cpus\":[");
      for (int i = 0; i < nt; i++) printf("%s%d", i ? "," : "", cpu[i]);
      printf("],\"per_thread_macs_per_s_by_cpu\":[");
      for (int i = 0; i < nt; i++) printf("%s%.4g", i ? "," : "", ops[i] * mpo / seconds);
      printf("]}\n");
      fflush(stdout);
    }
  }
  free(ops), free(win), free(cpu), free(rate);
  return 0;
}

// The core clock, read as the rate of a dependent chain of register-register 64-bit adds (one a cycle; a register
// operand, so the renamer cannot fold it as it may an immediate), per thread, in WIN_S windows.
static int freq(int threads, double seconds, const char* tag) {
  int nt = threads;
  long* win = calloc((size_t)nt * NWIN, sizeof *win);
  int* cpu = calloc(nt, sizeof *cpu);
  double t_start = 0, w0 = 0;
  omp_set_num_threads(nt);
#pragma omp parallel
  {
    int i = omp_get_thread_num();
    cpu[i] = sched_getcpu();
#pragma omp barrier
#pragma omp single
    t_start = now() + 0.05, w0 = wall() + 0.05;
    uint64_t x = 0, one = 1;
    double t = now();
    while (t < t_start) t = now();
    while (t - t_start < seconds) {
      for (int r = 0; r < 100; r++) __asm__ volatile(".rept 1000\n\taddq %1, %0\n\t.endr" : "+r"(x) : "r"(one));
      t = now();
      int w = (int)((t - t_start) / WIN_S);
      if (w < NWIN) win[(size_t)i * NWIN + w] += 100000;
    }
  }
  int nw = (int)(seconds / WIN_S);
  for (int i = 0; i < nt; i++) {
    double tot = 0;
    for (int w = 0; w < nw && w < NWIN; w++) tot += win[(size_t)i * NWIN + w];
    printf("{\"constant\":\"host.core_ghz.cpu%d.t%d%s\",\"value\":%.4g,\"unit\":\"GHz\",\"method\":\"host.c dependent "
           "addq chain (1 a cycle), %d threads, %.1f s\",\"t0_unix\":%.3f,\"window_s\":%.2f,\"window_ghz\":[",
           cpu[i], nt, tag, tot / (nw * WIN_S) / 1e9, nt, seconds, w0, WIN_S);
    for (int w = 0; w < nw && w < NWIN; w++) printf("%s%.4g", w ? "," : "", win[(size_t)i * NWIN + w] / WIN_S / 1e9);
    printf("]}\n");
  }
  fflush(stdout);
  free(win), free(cpu);
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

static int disk(const char* path, long mib, const char* tag) {
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
  snprintf(key, sizeof key, "host.disk%s.write_fsync.bytes_per_s", tag);
  emit(key, mib * 1048576.0 / tw, "B/s", meth);
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
    snprintf(key, sizeof key, "host.disk%s.read_odirect.r%d.bytes_per_s", tag, nr);
    snprintf(meth, sizeof meth, "host.c: O_DIRECT pread 4 MiB requests, %d reader thread(s) over disjoint ranges, of the %ld MiB file "
             "just written to %s%s", nr, mib, path, bad ? " (short read)" : "");
    emit(key, mib * 1048576.0 / tr, "B/s", meth);
  }
  free(buf);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 3 && !strcmp(argv[1], "stream")) return stream(atoi(argv[2]), argc > 3 ? argv[3] : "");
  if (argc >= 4 && !strcmp(argv[1], "amx"))
    return amx(atoi(argv[2]), atof(argv[3]), argc > 4 ? argv[4] : "", argc > 5 ? argv[5] : "resident");
  if (argc >= 4 && !strcmp(argv[1], "freq")) return freq(atoi(argv[2]), atof(argv[3]), argc > 4 ? argv[4] : "");
  if (argc >= 4 && !strcmp(argv[1], "disk")) return disk(argv[2], atol(argv[3]), argc > 4 ? argv[4] : "");
  fprintf(stderr, "usage: host stream THREADS [TAG] | amx THREADS SECONDS [TAG [VARIANTS]] | freq THREADS SECONDS [TAG] | "
          "disk PATH MIB [TAG]\n");
  return 2;
}
