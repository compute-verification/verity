// Host DRAM read bandwidth on a list of CPUs: one thread a CPU, each summing its own buffer (first-touched on its own
// node) for a number of passes, then one JSON line with the aggregate GB/s. Under a node-level guarantee the host's
// DRAM bandwidth bounds how many words a host process can sketch from DRAM in a step.
//
//   dram_read CPULIST MIB_PER_THREAD PASSES     e.g. dram_read 48-95 512 8
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static size_t words_per_thread;
static int passes;
static pthread_barrier_t start, warm;

typedef struct {
  int cpu;
  uint64_t sum;
  double seconds;
} Worker;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void *run(void *arg) {
  Worker *w = arg;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(w->cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  uint64_t *buf = aligned_alloc(4096, words_per_thread * 8);
  for (size_t i = 0; i < words_per_thread; ++i) buf[i] = i * 0x9e3779b97f4a7c15ull;
  pthread_barrier_wait(&warm);
  uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  pthread_barrier_wait(&start);
  double t0 = now();
  for (int p = 0; p < passes; ++p)
    for (size_t i = 0; i < words_per_thread; i += 4) {
      s0 += buf[i];
      s1 += buf[i + 1];
      s2 += buf[i + 2];
      s3 += buf[i + 3];
    }
  w->seconds = now() - t0;
  w->sum = s0 ^ s1 ^ s2 ^ s3;
  free(buf);
  return NULL;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: dram_read CPULIST MIB_PER_THREAD PASSES\n");
    return 2;
  }
  int cpus[1024], n = 0;
  for (char *tok = strtok(argv[1], ","); tok; tok = strtok(NULL, ",")) {
    int lo, hi;
    if (sscanf(tok, "%d-%d", &lo, &hi) == 2)
      for (int c = lo; c <= hi && n < 1024; ++c) cpus[n++] = c;
    else
      cpus[n++] = atoi(tok);
  }
  words_per_thread = (size_t)atol(argv[2]) * 1024 * 1024 / 8;
  passes = atoi(argv[3]);
  pthread_barrier_init(&warm, NULL, n);
  pthread_barrier_init(&start, NULL, n);
  Worker *w = calloc(n, sizeof *w);
  pthread_t *t = calloc(n, sizeof *t);
  for (int i = 0; i < n; ++i) {
    w[i].cpu = cpus[i];
    pthread_create(&t[i], NULL, run, &w[i]);
  }
  double slowest = 0;
  uint64_t x = 0;
  for (int i = 0; i < n; ++i) {
    pthread_join(t[i], NULL);
    if (w[i].seconds > slowest) slowest = w[i].seconds;
    x ^= w[i].sum;
  }
  double bytes = (double)n * words_per_thread * 8 * passes;
  printf("{\"threads\": %d, \"mib_per_thread\": %s, \"passes\": %d, \"gb_per_s\": %.1f, \"check\": %llu}\n", n, argv[2],
         passes, bytes / slowest / 1e9, (unsigned long long)(x & 0xffff));
  return 0;
}
