// STREAM's copy (c = a), triad (a = b + s c) and a read (sum of a) on the host: one thread per listed CPU, each with its
// own three arrays of --mib MiB, placed by first touch on the thread's own NUMA node (--node local) or bound to node N,
// for --seconds, then one JSON line with the aggregate rate and each node's share. Bytes are STREAM's count (copy 16,
// triad 24, read 8 a double); `dram_gb_per_s` adds the write-allocate read of a cached store (copy 24, triad 32), which
// the _nt kernels' non-temporal stores don't do.
//
//   stream --cpus 0-95,96-119 --node local --kernel triad|copy|read|triad_nt|copy_nt --mib 64 --seconds 2 [--start-at T]
//
// Each thread reads its CPU and node with getcpu once pinned, and the node of its arrays' first and last pages with
// move_pages; `misplaced` counts threads whose arrays aren't all on the node asked for. --start-at: a CLOCK_REALTIME second
// at which to start timing, so that another process's timed phase (benchmarks/host_tier/pcie.cu) coincides.
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

enum { K_COPY, K_TRIAD, K_READ, K_COPY_NT, K_TRIAD_NT };
static const char *KNAMES[] = {"copy", "triad", "read", "copy_nt", "triad_nt"};
static const double STREAM_B[] = {16, 24, 8, 16, 24}, DRAM_B[] = {24, 32, 8, 16, 24};
#define MAXNODE 8

static size_t elems;
static int kernel;
static const char *node_arg = "local";
static atomic_int stop;
static pthread_barrier_t ready, go;

typedef struct {
  int cpu, node, misplaced;
  uint64_t elems_done;
  double fold;
} Worker;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static double *alloc(size_t bytes) {
  const size_t align = 2u << 20;
  uint8_t *p = mmap(NULL, bytes + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return NULL;
  uint8_t *a = (uint8_t *)(((uintptr_t)p + align - 1) & ~(uintptr_t)(align - 1));
  madvise(a, bytes, MADV_HUGEPAGE);
  if (strcmp(node_arg, "local") != 0) {
    unsigned long mask = 1ul << atoi(node_arg);
    if (syscall(SYS_mbind, a, bytes, 2 /* MPOL_BIND */, &mask, 64, 1 /* MPOL_MF_STRICT */) != 0) {
      perror("mbind");
      exit(5);
    }
  }
  return (double *)a;
}

static int page_node(void *p) {
  void *pages[1] = {(void *)((uintptr_t)p & ~(uintptr_t)4095)};
  int status[1] = {-1};
  if (syscall(SYS_move_pages, 0, 1, pages, NULL, status, 0) != 0) return -1;
  return status[0];
}

static void *run(void *arg) {
  Worker *w = arg;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(w->cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  unsigned cpu = 0, node = 0;
  syscall(SYS_getcpu, &cpu, &node, NULL);
  w->node = (int)node;
  const size_t bytes = elems * 8;
  const int narr = kernel == K_READ ? 1 : 3;  // the read touches a alone
  double *a = alloc(bytes), *b = narr > 1 ? alloc(bytes) : a, *c = narr > 1 ? alloc(bytes) : a;
  if (!a || !b || !c) {
    fprintf(stderr, "alloc failed on cpu %d\n", w->cpu);
    exit(5);
  }
  for (size_t i = 0; i < elems; ++i) a[i] = 1.0, b[i] = 2.0, c[i] = 0.5 + (double)(i & 7);
  const int want = strcmp(node_arg, "local") == 0 ? (int)node : atoi(node_arg);
  double *arrs[3] = {a, b, c};
  for (int k = 0; k < narr; ++k)
    if (page_node(arrs[k]) != want || page_node(arrs[k] + elems - 1) != want) w->misplaced = 1;
  const size_t chunk = (1u << 20) / 8, nchunks = elems / chunk;
  const __m512d s = _mm512_set1_pd(3.0);
  __m512d r0 = _mm512_setzero_pd(), r1 = r0, r2 = r0, r3 = r0;
  uint64_t done = 0;
  pthread_barrier_wait(&ready);
  pthread_barrier_wait(&go);
  for (size_t q = 0; !atomic_load_explicit(&stop, memory_order_relaxed); q = (q + 1 == nchunks) ? 0 : q + 1) {
    double *A = a + q * chunk, *B = b + q * chunk, *C = c + q * chunk;
    switch (kernel) {
      case K_COPY:
        for (size_t i = 0; i < chunk; i += 8) _mm512_store_pd(C + i, _mm512_load_pd(A + i));
        break;
      case K_COPY_NT:
        for (size_t i = 0; i < chunk; i += 8) _mm512_stream_pd(C + i, _mm512_load_pd(A + i));
        break;
      case K_TRIAD:
        for (size_t i = 0; i < chunk; i += 8) _mm512_store_pd(A + i, _mm512_fmadd_pd(s, _mm512_load_pd(C + i), _mm512_load_pd(B + i)));
        break;
      case K_TRIAD_NT:
        for (size_t i = 0; i < chunk; i += 8) _mm512_stream_pd(A + i, _mm512_fmadd_pd(s, _mm512_load_pd(C + i), _mm512_load_pd(B + i)));
        break;
      case K_READ:
        for (size_t i = 0; i < chunk; i += 32) {
          r0 = _mm512_add_pd(r0, _mm512_load_pd(A + i));
          r1 = _mm512_add_pd(r1, _mm512_load_pd(A + i + 8));
          r2 = _mm512_add_pd(r2, _mm512_load_pd(A + i + 16));
          r3 = _mm512_add_pd(r3, _mm512_load_pd(A + i + 24));
        }
        break;
    }
    done += chunk;
  }
  _mm_sfence();
  w->elems_done = done;
  w->fold = _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(r0, r1), _mm512_add_pd(r2, r3))) + a[7] + c[elems - 1];
  return NULL;
}

static int parse_cpus(char *s, int *cpus) {
  int n = 0;
  for (char *tok = strtok(s, ","); tok; tok = strtok(NULL, ",")) {
    int lo, hi;
    if (sscanf(tok, "%d-%d", &lo, &hi) == 2)
      for (int c = lo; c <= hi && n < 1024; ++c) cpus[n++] = c;
    else
      cpus[n++] = atoi(tok);
  }
  return n;
}

int main(int argc, char **argv) {
  char cpulist[4096] = "0";
  double seconds = 2, mib = 64, start_at = 0;
  const char *label = "";
  kernel = -1;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--cpus")) snprintf(cpulist, sizeof cpulist, "%s", argv[i + 1]);
    else if (!strcmp(argv[i], "--node")) node_arg = argv[i + 1];
    else if (!strcmp(argv[i], "--mib")) mib = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--seconds")) seconds = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--label")) label = argv[i + 1];
    else if (!strcmp(argv[i], "--start-at")) start_at = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--kernel")) {
      for (int k = 0; k < (int)(sizeof KNAMES / sizeof *KNAMES); ++k)
        if (!strcmp(argv[i + 1], KNAMES[k])) kernel = k;
    } else {
      fprintf(stderr, "unknown flag %s\n", argv[i]);
      return 2;
    }
  }
  if (kernel < 0) {
    fprintf(stderr, "--kernel copy|triad|read|copy_nt|triad_nt\n");
    return 2;
  }
  char list_copy[4096];
  snprintf(list_copy, sizeof list_copy, "%s", cpulist);
  int cpus[1024];
  const int n = parse_cpus(list_copy, cpus);
  elems = ((size_t)(mib * 1024 * 1024) & ~(size_t)((1u << 20) - 1)) / 8;
  pthread_barrier_init(&ready, NULL, n + 1);
  pthread_barrier_init(&go, NULL, n + 1);
  Worker *w = calloc(n, sizeof *w);
  pthread_t *t = calloc(n, sizeof *t);
  for (int i = 0; i < n; ++i) {
    w[i].cpu = cpus[i];
    pthread_create(&t[i], NULL, run, &w[i]);
  }
  pthread_barrier_wait(&ready);
  if (start_at > 0) {
    struct timespec r;
    clock_gettime(CLOCK_REALTIME, &r);
    const double wait = start_at - (r.tv_sec + 1e-9 * r.tv_nsec);
    if (wait < 0) fprintf(stderr, "start-at passed %.3f s before the arrays were ready\n", -wait);
    else {
      struct timespec d = {(time_t)wait, (long)((wait - (time_t)wait) * 1e9)};
      nanosleep(&d, NULL);
    }
  }
  const double t0 = now();
  pthread_barrier_wait(&go);
  struct timespec d = {(time_t)seconds, (long)((seconds - (time_t)seconds) * 1e9)};
  nanosleep(&d, NULL);
  atomic_store(&stop, 1);
  for (int i = 0; i < n; ++i) pthread_join(t[i], NULL);
  const double el = now() - t0;
  double node_elems[MAXNODE] = {0}, fold = 0, total = 0;
  int node_threads[MAXNODE] = {0}, misplaced = 0;
  for (int i = 0; i < n; ++i) {
    total += (double)w[i].elems_done;
    fold += w[i].fold;
    misplaced += w[i].misplaced;
    if (w[i].node >= 0 && w[i].node < MAXNODE) node_elems[w[i].node] += (double)w[i].elems_done, node_threads[w[i].node]++;
  }
  char per[512] = "";
  for (int k = 0, o = 0; k < MAXNODE; ++k)
    if (node_threads[k])
      o += snprintf(per + o, sizeof per - o, "%s\"%d\": {\"threads\": %d, \"gb_per_s\": %.2f}", o ? ", " : "", k,
                    node_threads[k], node_elems[k] * STREAM_B[kernel] / el / 1e9);
  printf("{\"label\": \"%s\", \"cpus\": \"%s\", \"threads\": %d, \"node\": \"%s\", \"kernel\": \"%s\", \"mib_per_array\": %.0f, "
         "\"seconds\": %.3f, \"gb_per_s\": %.2f, \"dram_gb_per_s\": %.2f, \"by_cpu_node\": {%s}, \"misplaced\": %d, "
         "\"fold\": %.6e}\n",
         label, cpulist, n, node_arg, KNAMES[kernel], mib, el, total * STREAM_B[kernel] / el / 1e9,
         total * DRAM_B[kernel] / el / 1e9, per, misplaced, fold);
  return misplaced ? 6 : 0;
}
