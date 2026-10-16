// The adversary's best host DRAM read rate: one thread per listed CPU, each reading its own buffer (placed on a NUMA node
// or interleaved, 4 KiB pages or transparent huge pages) for a fixed time with one of several read kernels, then one JSON
// line with the aggregate rate. The rate the adversary can reach is the calibrated ceiling the host tier is exhausted
// against (the remote bandwidth design's mu_host).
//
//   dram --cpus 48-95 --node 0 --mib 512 --kernel avx512 --thp 1 --seconds 3 [--label S]
//
// --start-at: a CLOCK_REALTIME second at which to start timing, so that two processes' timed phases coincide.
// --node: N (mbind MPOL_BIND), "i" (interleave over every node), "local" (first touch on the thread's own node).
// --kernel: scalar (4 u64 adds), avx512 (aligned 512-bit loads), nt (vmovntdqa), pf (avx512 with prefetcht0 1 KiB ahead),
//           pfnta (prefetchnta), multi (avx512 over 4 streams of the buffer at once).
// Each thread checks a stop flag every 1 MiB; the rate is the bytes every thread read between the start barrier and the
// last join, over that time. The output's fold of the data read keeps the loads from being dropped.
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

#define MPOL_BIND 2
#define MPOL_INTERLEAVE 3
#define MPOL_MF_STRICT 1

static long mbind(void *addr, unsigned long len, int mode, const unsigned long *mask, unsigned long maxnode,
                  unsigned flags) {
  return syscall(SYS_mbind, addr, len, mode, mask, maxnode, flags);
}

enum { K_SCALAR, K_AVX512, K_NT, K_PF, K_PFNTA, K_MULTI };
static const char *KNAMES[] = {"scalar", "avx512", "nt", "pf", "pfnta", "multi"};

static size_t bytes_per_thread;
static int kernel, thp;
static const char *node_arg = "local";
static atomic_int stop;
static pthread_barrier_t ready, go;

typedef struct {
  int cpu;
  uint64_t bytes, fold;
} Worker;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void *alloc(size_t n) {
  size_t align = 2u << 20;
  uint8_t *p = mmap(NULL, n + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return NULL;
  uint8_t *a = (uint8_t *)(((uintptr_t)p + align - 1) & ~(uintptr_t)(align - 1));
  madvise(a, n, thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  if (strcmp(node_arg, "local") != 0) {
    unsigned long mask = 0;
    int mode;
    if (strcmp(node_arg, "i") == 0) {
      mask = 3;
      mode = MPOL_INTERLEAVE;
    } else {
      mask = 1ul << atoi(node_arg);
      mode = MPOL_BIND;
    }
    if (mbind(a, n, mode, &mask, 64, MPOL_MF_STRICT) != 0) {
      perror("mbind");
      exit(5);
    }
  }
  return a;
}

static inline __m512i chunk_avx512(const __m512i *p, size_t n, __m512i acc) {
  __m512i a0 = acc, a1 = _mm512_setzero_si512(), a2 = a1, a3 = a1;
  for (size_t i = 0; i < n; i += 8) {
    a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i));
    a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + i + 1));
    a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + i + 2));
    a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + i + 3));
    a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i + 4));
    a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + i + 5));
    a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + i + 6));
    a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + i + 7));
  }
  return _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3));
}

static inline __m512i chunk_nt(const __m512i *p, size_t n, __m512i acc) {
  __m512i a0 = acc, a1 = _mm512_setzero_si512(), a2 = a1, a3 = a1;
  for (size_t i = 0; i < n; i += 8) {
    a0 = _mm512_add_epi64(a0, _mm512_stream_load_si512((void *)(p + i)));
    a1 = _mm512_add_epi64(a1, _mm512_stream_load_si512((void *)(p + i + 1)));
    a2 = _mm512_add_epi64(a2, _mm512_stream_load_si512((void *)(p + i + 2)));
    a3 = _mm512_add_epi64(a3, _mm512_stream_load_si512((void *)(p + i + 3)));
    a0 = _mm512_add_epi64(a0, _mm512_stream_load_si512((void *)(p + i + 4)));
    a1 = _mm512_add_epi64(a1, _mm512_stream_load_si512((void *)(p + i + 5)));
    a2 = _mm512_add_epi64(a2, _mm512_stream_load_si512((void *)(p + i + 6)));
    a3 = _mm512_add_epi64(a3, _mm512_stream_load_si512((void *)(p + i + 7)));
  }
  return _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3));
}

#define PF_AHEAD 16  // 512-bit lines: 1 KiB ahead
static inline __m512i chunk_pf(const __m512i *p, size_t n, __m512i acc, int nta) {
  __m512i a0 = acc, a1 = _mm512_setzero_si512(), a2 = a1, a3 = a1;
  for (size_t i = 0; i < n; i += 8) {
    if (nta) {
      _mm_prefetch((const char *)(p + i + PF_AHEAD), _MM_HINT_NTA);
      _mm_prefetch((const char *)(p + i + PF_AHEAD + 4), _MM_HINT_NTA);
    } else {
      _mm_prefetch((const char *)(p + i + PF_AHEAD), _MM_HINT_T0);
      _mm_prefetch((const char *)(p + i + PF_AHEAD + 4), _MM_HINT_T0);
    }
    a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i));
    a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + i + 1));
    a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + i + 2));
    a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + i + 3));
    a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i + 4));
    a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + i + 5));
    a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + i + 6));
    a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + i + 7));
  }
  return _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3));
}

// four sequential streams a quarter of the chunk apart, one line of each per step
static inline __m512i chunk_multi(const __m512i *p, size_t n, __m512i acc) {
  size_t q = n / 4;
  __m512i a0 = acc, a1 = _mm512_setzero_si512(), a2 = a1, a3 = a1;
  for (size_t i = 0; i < q; i += 2) {
    a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i));
    a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + q + i));
    a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + 2 * q + i));
    a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + 3 * q + i));
    a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i + 1));
    a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + q + i + 1));
    a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + 2 * q + i + 1));
    a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + 3 * q + i + 1));
  }
  return _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3));
}

static uint64_t chunk_scalar(const uint64_t *p, size_t n) {
  uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (size_t i = 0; i < n; i += 4) {
    s0 += p[i];
    s1 += p[i + 1];
    s2 += p[i + 2];
    s3 += p[i + 3];
  }
  return s0 ^ s1 ^ s2 ^ s3;
}

static void *run(void *arg) {
  Worker *w = arg;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(w->cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  uint64_t *buf = alloc(bytes_per_thread);
  if (!buf) {
    fprintf(stderr, "alloc failed on cpu %d\n", w->cpu);
    exit(5);
  }
  for (size_t i = 0; i < bytes_per_thread / 8; ++i) buf[i] = i * 0x9e3779b97f4a7c15ull ^ (uint64_t)w->cpu;
  const size_t chunk = 1u << 20;
  const size_t nchunks = bytes_per_thread / chunk;
  __m512i acc = _mm512_setzero_si512();
  uint64_t sacc = 0, bytes = 0;
  pthread_barrier_wait(&ready);
  pthread_barrier_wait(&go);
  for (size_t c = 0; !atomic_load_explicit(&stop, memory_order_relaxed); c = (c + 1 == nchunks) ? 0 : c + 1) {
    const uint8_t *p = (const uint8_t *)buf + c * chunk;
    switch (kernel) {
      case K_SCALAR: sacc ^= chunk_scalar((const uint64_t *)p, chunk / 8); break;
      case K_AVX512: acc = chunk_avx512((const __m512i *)p, chunk / 64, acc); break;
      case K_NT: acc = chunk_nt((const __m512i *)p, chunk / 64, acc); break;
      case K_PF: acc = chunk_pf((const __m512i *)p, chunk / 64, acc, 0); break;
      case K_PFNTA: acc = chunk_pf((const __m512i *)p, chunk / 64, acc, 1); break;
      case K_MULTI: acc = chunk_multi((const __m512i *)p, chunk / 64, acc); break;
    }
    bytes += chunk;
  }
  w->bytes = bytes;
  w->fold = sacc ^ (uint64_t)_mm512_reduce_add_epi64(acc);
  return NULL;
}

static int parse_cpus(char *s, int *cpus) {
  int n = 0;
  for (char *tok = strtok(s, ","); tok; tok = strtok(NULL, ",")) {
    int lo, hi, step = 1;
    if (sscanf(tok, "%d-%d:%d", &lo, &hi, &step) >= 2)
      for (int c = lo; c <= hi && n < 1024; c += step) cpus[n++] = c;
    else
      cpus[n++] = atoi(tok);
  }
  return n;
}

static long anon_huge_kb(void) {
  FILE *f = fopen("/proc/self/smaps_rollup", "r");
  char line[256];
  long kb = -1;
  while (f && fgets(line, sizeof line, f))
    if (sscanf(line, "AnonHugePages: %ld kB", &kb) == 1) break;
  if (f) fclose(f);
  return kb;
}

int main(int argc, char **argv) {
  char cpulist[4096] = "48";
  double seconds = 3, mib = 512, start_at = 0;
  const char *label = "";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--cpus")) snprintf(cpulist, sizeof cpulist, "%s", argv[i + 1]);
    else if (!strcmp(argv[i], "--node")) node_arg = argv[i + 1];
    else if (!strcmp(argv[i], "--mib")) mib = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--seconds")) seconds = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--thp")) thp = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--label")) label = argv[i + 1];
    else if (!strcmp(argv[i], "--start-at")) start_at = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--kernel")) {
      kernel = -1;
      for (int k = 0; k < (int)(sizeof KNAMES / sizeof *KNAMES); ++k)
        if (!strcmp(argv[i + 1], KNAMES[k])) kernel = k;
      if (kernel < 0) {
        fprintf(stderr, "unknown kernel %s\n", argv[i + 1]);
        return 2;
      }
    } else {
      fprintf(stderr, "unknown flag %s\n", argv[i]);
      return 2;
    }
  }
  char list_copy[4096];
  snprintf(list_copy, sizeof list_copy, "%s", cpulist);
  int cpus[1024];
  int n = parse_cpus(list_copy, cpus);
  bytes_per_thread = (size_t)(mib * 1024 * 1024) & ~(size_t)((1u << 20) - 1);
  pthread_barrier_init(&ready, NULL, n + 1);
  pthread_barrier_init(&go, NULL, n + 1);
  Worker *w = calloc(n, sizeof *w);
  pthread_t *t = calloc(n, sizeof *t);
  for (int i = 0; i < n; ++i) {
    w[i].cpu = cpus[i];
    pthread_create(&t[i], NULL, run, &w[i]);
  }
  pthread_barrier_wait(&ready);
  long huge = anon_huge_kb();
  if (start_at > 0) {
    struct timespec r;
    clock_gettime(CLOCK_REALTIME, &r);
    double wait = start_at - (r.tv_sec + 1e-9 * r.tv_nsec);
    if (wait < 0) fprintf(stderr, "start-at passed %.3f s before the buffers were ready\n", -wait);
    else {
      struct timespec d = {(time_t)wait, (long)((wait - (time_t)wait) * 1e9)};
      nanosleep(&d, NULL);
    }
  }
  double t0 = now();
  pthread_barrier_wait(&go);
  struct timespec d = {(time_t)seconds, (long)((seconds - (time_t)seconds) * 1e9)};
  nanosleep(&d, NULL);
  atomic_store(&stop, 1);
  uint64_t total = 0, fold = 0, lo = UINT64_MAX, hi = 0;
  for (int i = 0; i < n; ++i) pthread_join(t[i], NULL);
  double t1 = now();
  for (int i = 0; i < n; ++i) {
    total += w[i].bytes;
    fold ^= w[i].fold;
    if (w[i].bytes < lo) lo = w[i].bytes;
    if (w[i].bytes > hi) hi = w[i].bytes;
  }
  double el = t1 - t0;
  printf("{\"label\": \"%s\", \"cpus\": \"%s\", \"threads\": %d, \"node\": \"%s\", \"kernel\": \"%s\", \"thp\": %d, "
         "\"mib_per_thread\": %.0f, \"anon_huge_kb\": %ld, \"seconds\": %.3f, \"gb_per_s\": %.2f, "
         "\"thread_gb_per_s_min\": %.2f, \"thread_gb_per_s_max\": %.2f, \"fold\": \"%016llx\"}\n",
         label, cpulist, n, node_arg, KNAMES[kernel], thp, mib, huge, el, total / el / 1e9, lo / el / 1e9, hi / el / 1e9,
         (unsigned long long)fold);
  return 0;
}
