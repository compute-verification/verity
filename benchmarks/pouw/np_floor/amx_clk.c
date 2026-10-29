// AMX's int8 rate a core with the core's clock read beside it, for HostsCounted's AMX row. On each core under test one
// hardware thread runs TDPBSSD (host.c's k64 loop: four accumulators, 16,384 MACs each) and its sibling a chain of
// dependent 64-bit IMUL (latency 3 cycles; Golden Cove folds a chain of immediate adds at rename, so an add chain is no
// clock), so 3 x the sibling's IMULs a second is the core's clock while the tile unit runs, at one uop every 3 cycles. The tile unit's peak is 1,024 int8 MACs a cycle (a 16x16x64 TDPBSSD every 16 cycles). Phases, each
// SECONDS long, one JSON line each:
//   one     one core: AMX on one thread, the chain on its sibling
//   idle    the same core: the chain alone
//   all     every core: AMX on one thread, the chain on its sibling
//   allamx  every core: AMX on one thread, the sibling idle
//   both    every core: AMX on both threads (MACs a second only)
//   gcc -O3 -march=sapphirerapids -pthread amx_clk.c -o amx_clk; ./amx_clk [SECONDS]
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
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

static volatile int go, stop;

enum { AMX, CHAIN };
struct job {
  int cpu, kind;
  double t;
  uint64_t n;
  uint32_t sink;
  pthread_t th;
};

static void* run(void* p) {
  struct job* j = p;
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(j->cpu, &s);
  if (pthread_setaffinity_np(pthread_self(), sizeof s, &s)) {
    fprintf(stderr, "amx_clk: can't pin to cpu %d\n", j->cpu);
    exit(2);
  }
  if (j->kind == AMX) {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
      fprintf(stderr, "amx_clk: no AMX permission\n");
      exit(2);
    }
    int8_t A[2][1024] __attribute__((aligned(64))), B[2][1024] __attribute__((aligned(64)));
    int32_t C[256] __attribute__((aligned(64)));
    for (int i = 0; i < 1024; i++)
      A[0][i] = (int8_t)(i * 7), A[1][i] = (int8_t)(i * 13), B[0][i] = (int8_t)(i * 3), B[1][i] = (int8_t)(i * 5);
    struct tilecfg c;
    memset(&c, 0, sizeof c);
    c.palette_id = 1;
    for (int t = 0; t < 8; t++) c.rows[t] = 16, c.colsb[t] = 64;
    _tile_loadconfig(&c);
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);
    _tile_loadd(4, A[0], 64);
    _tile_loadd(5, A[1], 64);
    _tile_loadd(6, B[0], 64);
    _tile_loadd(7, B[1], 64);
    while (!go) {
    }
    const double t0 = now();
    uint64_t n = 0;
    do {
      for (int i = 0; i < 256; i++) {
        _tile_dpbssd(0, 4, 6);
        _tile_dpbssd(1, 4, 7);
        _tile_dpbssd(2, 5, 6);
        _tile_dpbssd(3, 5, 7);
      }
      n += 1024;
    } while (!stop);
    j->t = now() - t0;
    j->n = n;
    _tile_stored(0, C, 64);
    j->sink = (uint32_t)C[3];
    _tile_stored(3, C, 64);
    j->sink ^= (uint32_t)C[17];
    _tile_release();
  } else {
    uint64_t x = (uint64_t)j->cpu, n = 0;
    while (!go) {
    }
    const double t0 = now();
    do {
      __asm__ volatile(".rept 1000\n\timul %0, %0\n\t.endr" : "+r"(x));
      n += 1000;
    } while (!stop);
    j->t = now() - t0;
    j->n = 3 * n;
    j->sink = (uint32_t)x;
  }
  return NULL;
}

static int ncores;
static int first[512], second[512];  // each core's hardware threads (second -1: none)

static void topology(void) {
  const long n = sysconf(_SC_NPROCESSORS_ONLN);
  int seen[1024] = {0};
  for (int cpu = 0; cpu < n; cpu++) {
    if (seen[cpu]) continue;
    char path[128], buf[64];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    FILE* f = fopen(path, "r");
    int a = cpu, b = -1;
    if (f && fgets(buf, sizeof buf, f)) {
      char* e;
      a = (int)strtol(buf, &e, 10);
      if (*e == ',' || *e == '-') b = (int)strtol(e + 1, NULL, 10);
    }
    if (f) fclose(f);
    seen[a] = 1;
    if (b >= 0) seen[b] = 1;
    first[ncores] = a;
    second[ncores] = b;
    ncores++;
  }
}

// AMX on `amx` cpus and the chain on `chn` cpus for `secs`; prints the phase's line
static void phase(const char* name, const int* amx, int na, const int* chn, int nc, double secs) {
  struct job j[1024];
  int n = 0;
  for (int i = 0; i < na; i++) j[n++] = (struct job){.cpu = amx[i], .kind = AMX};
  for (int i = 0; i < nc; i++) j[n++] = (struct job){.cpu = chn[i], .kind = CHAIN};
  go = stop = 0;
  for (int i = 0; i < n; i++) pthread_create(&j[i].th, NULL, run, &j[i]);
  usleep(100000);
  go = 1;
  usleep((useconds_t)(secs * 1e6));
  stop = 1;
  for (int i = 0; i < n; i++) pthread_join(j[i].th, NULL);
  double macs = 0, hz = 0;
  uint32_t sink = 0;
  printf("{\"phase\": \"%s\", \"secs\": %.3f", name, secs);
  printf(", \"amx\": [");
  for (int i = 0; i < na; i++) {
    const double m = (double)j[i].n * 16384.0 / j[i].t;
    macs += m;
    sink ^= j[i].sink;
    printf("%s{\"cpu\": %d, \"macs_per_s\": %.6g}", i ? ", " : "", j[i].cpu, m);
  }
  printf("], \"chain\": [");
  for (int i = na; i < n; i++) {
    const double h = (double)j[i].n / j[i].t;
    hz += h;
    sink ^= j[i].sink;
    printf("%s{\"cpu\": %d, \"hz\": %.6g}", i > na ? ", " : "", j[i].cpu, h);
  }
  printf("], \"macs_per_s\": %.6g", macs);
  if (nc) printf(", \"hz_mean\": %.6g", hz / nc);
  if (na && nc == na) printf(", \"macs_per_cycle_a_core\": %.6g", macs / hz);
  printf(", \"sink\": %u}\n", sink);
  fflush(stdout);
}

int main(int argc, char** argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 3.0;
  topology();
  int a[512], b[512], ab[1024], nb = 0;
  for (int i = 0; i < ncores; i++) {
    a[i] = first[i];
    b[i] = second[i];
    if (second[i] < 0) nb = -1;
  }
  printf("{\"cores\": %d, \"threads_a_core\": %d, \"pairs\": [", ncores, nb < 0 ? 1 : 2);
  for (int i = 0; i < ncores; i++) printf("%s[%d, %d]", i ? ", " : "", first[i], second[i]);
  printf("]}\n");
  if (nb < 0) {
    fprintf(stderr, "amx_clk: a core without a sibling thread; no clock read beside AMX\n");
    phase("allamx", a, ncores, NULL, 0, secs);
    return 0;
  }
  const int k = ncores > 1 ? 1 : 0;  // one core: not cpu 0's
  phase("one", &a[k], 1, &b[k], 1, secs);
  phase("idle", NULL, 0, &b[k], 1, secs);
  phase("all", a, ncores, b, ncores, secs);
  phase("allamx", a, ncores, NULL, 0, secs);
  int n = 0;
  for (int i = 0; i < ncores; i++) ab[n++] = a[i], ab[n++] = b[i];
  phase("both", ab, n, NULL, 0, secs);
  phase("one", &a[k], 1, &b[k], 1, secs);
  return 0;
}
