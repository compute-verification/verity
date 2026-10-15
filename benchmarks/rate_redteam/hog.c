// A host memory-bandwidth tenant: one thread on each CPU of a list, each copying between two private buffers until
// SIGTERM or SIGINT or a time limit, then one JSON line with the bandwidth it took. The prover's own tenant on the
// challenger's host, on physical cores the challenger doesn't hold.
//
//   hog CPULIST MIB MAXSEC     e.g. hog 78-95 256 120
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static atomic_int stop;
static size_t buf_bytes;

typedef struct {
  int cpu;
  double bytes, seconds;
} Worker;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void* run(void* p) {
  Worker* w = p;
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(w->cpu, &s);
  pthread_setaffinity_np(pthread_self(), sizeof s, &s);
  char *a = aligned_alloc(4096, buf_bytes), *b = aligned_alloc(4096, buf_bytes);
  if (!a || !b) return NULL;
  memset(a, 1, buf_bytes);
  memset(b, 2, buf_bytes);
  double t0 = now();
  while (!atomic_load(&stop)) {
    memcpy(b, a, buf_bytes);
    w->bytes += 2.0 * buf_bytes;
    char* c = a;
    a = b;
    b = c;
  }
  w->seconds = now() - t0;
  free(a);
  free(b);
  return NULL;
}

static void on_signal(int sig) {
  (void)sig;
  atomic_store(&stop, 1);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "hog CPULIST MIB MAXSEC\n");
    return 2;
  }
  int cpus[1024], n = 0;
  for (char* tok = strtok(argv[1], ","); tok && n < 1024; tok = strtok(NULL, ",")) {
    int lo, hi;
    if (sscanf(tok, "%d-%d", &lo, &hi) != 2) hi = lo = atoi(tok);
    for (int c = lo; c <= hi && n < 1024; ++c) cpus[n++] = c;
  }
  buf_bytes = (size_t)atoi(argv[2]) << 20;
  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal);
  signal(SIGALRM, on_signal);
  alarm((unsigned)atoi(argv[3]));
  Worker* w = calloc(n, sizeof *w);
  pthread_t* t = calloc(n, sizeof *t);
  for (int i = 0; i < n; ++i) {
    w[i].cpu = cpus[i];
    pthread_create(&t[i], NULL, run, &w[i]);
  }
  double bytes = 0, seconds = 0;
  for (int i = 0; i < n; ++i) {
    pthread_join(t[i], NULL);
    bytes += w[i].bytes;
    if (w[i].seconds > seconds) seconds = w[i].seconds;
  }
  printf("{\"threads\":%d,\"buf_mib\":%zu,\"seconds\":%.3f,\"copy_gbs\":%.2f}\n", n, buf_bytes >> 20, seconds,
         seconds > 0 ? bytes / seconds / 1e9 : 0.0);
  return 0;
}
