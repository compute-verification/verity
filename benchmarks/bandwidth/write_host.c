// Host DRAM write ceilings on vy-nebius (two Xeon 6776P sockets, 16 channels of DDR5-6400): the fastest the host's
// cores write labels into its DRAM, for poub's `timing.HOST_WRITE_BYTES_PER_S`.
//
//   nt     labels derived on the core (write.cu's `label`, 16 words at a time), 64-byte non-temporal stores, which
//          skip the read for ownership
//   st     the same labels with cached 64-byte stores, each line read for ownership before it is written
//   stosb  rep stosb of one byte: not labels but the cores' fast-string fill, which skips the read for ownership too
//
// One thread per listed CPU, each with its own --mib MiB array placed by first touch on the thread's NUMA node, for
// --seconds; then one JSON line: GB/s written (1e9 bytes), each node's share, and `misplaced`, the threads whose array
// isn't on their node. A sampled check after the timed loop must find what the kernel writes in every array (`wrong`).
//
//   bwwrite_host --cpus 48,50,52 --kernel nt|st|stosb --mib 512 --seconds 2
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

enum { K_NT, K_ST, K_STOSB };
static const char *KNAMES[] = {"nt", "st", "stosb"};
#define SEED 0x6a09e667u
#define STOSB_BYTE 0x3C
#define CHUNK (1u << 20)
#define MAXNODE 8

static size_t bytes;
static int kernel;
static atomic_int stop;
static pthread_barrier_t ready, go;

typedef struct {
  int cpu, node, misplaced, index;
  uint64_t bytes_done, wrong;
  double end;
} Worker;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint32_t label(uint64_t w) {
  uint32_t x = (uint32_t)w ^ SEED ^ (uint32_t)(w >> 32) * 0x9E3779B9u;
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

// Words w0 .. w0 + 15, w0 a multiple of 16 (so the low halves don't carry): `label` of each.
static inline __m512i label16(uint64_t w0, __m512i lane) {
  __m512i x = _mm512_xor_si512(_mm512_add_epi32(_mm512_set1_epi32((int)(uint32_t)w0), lane),
                               _mm512_set1_epi32((int)(SEED ^ (uint32_t)(w0 >> 32) * 0x9E3779B9u)));
  x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
  x = _mm512_mullo_epi32(x, _mm512_set1_epi32(0x7FEB352D));
  x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 15));
  x = _mm512_mullo_epi32(x, _mm512_set1_epi32((int)0x846CA68Bu));
  return _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
}

static void stosb(void *p, size_t n) {
  int c = STOSB_BYTE;
  asm volatile("rep stosb" : "+D"(p), "+c"(n) : "a"(c) : "memory");
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
  const size_t align = 2u << 20;
  uint8_t *raw = mmap(NULL, bytes + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) {
    fprintf(stderr, "alloc failed on cpu %d\n", w->cpu);
    exit(5);
  }
  uint8_t *a = (uint8_t *)(((uintptr_t)raw + align - 1) & ~(uintptr_t)(align - 1));
  madvise(a, bytes, MADV_HUGEPAGE);
  memset(a, 0, bytes);
  w->misplaced = page_node(a) != (int)node || page_node(a + bytes - 1) != (int)node;
  const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const uint64_t base = (uint64_t)w->index * (bytes / 4);  // each thread's words have their own indices
  const size_t chunks = bytes / CHUNK;
  uint64_t done = 0;
  pthread_barrier_wait(&ready);
  pthread_barrier_wait(&go);
  for (size_t q = 0; !atomic_load_explicit(&stop, memory_order_relaxed); q = (q + 1 == chunks) ? 0 : q + 1) {
    uint8_t *c = a + q * CHUNK;
    const uint64_t w0 = base + (uint64_t)q * (CHUNK / 4);
    switch (kernel) {
      case K_NT:
        for (size_t i = 0; i < CHUNK; i += 64) _mm512_stream_si512((void *)(c + i), label16(w0 + i / 4, lane));
        break;
      case K_ST:
        for (size_t i = 0; i < CHUNK; i += 64) _mm512_store_si512((void *)(c + i), label16(w0 + i / 4, lane));
        break;
      case K_STOSB:
        stosb(c, CHUNK);
        break;
    }
    done += CHUNK;
  }
  _mm_sfence();
  w->end = now();
  w->bytes_done = done;
  uint64_t s = 0x9e3779b97f4a7c15ull ^ (uint64_t)w->cpu;
  for (int k = 0; k < 4096; ++k) {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    const size_t word = (size_t)(s % (bytes / 4));
    const uint32_t got = ((const uint32_t *)a)[word];
    const uint32_t want = kernel == K_STOSB ? 0x01010101u * STOSB_BYTE : label(base + word);
    w->wrong += got != want;
  }
  munmap(raw, bytes + align);
  return NULL;
}

static int parse_cpus(char *s, int *cpus) {
  int n = 0;
  for (char *tok = strtok(s, ","); tok; tok = strtok(NULL, ",")) {
    int lo, hi;
    if (sscanf(tok, "%d-%d", &lo, &hi) == 2)
      for (int c = lo; c <= hi && n < 1024; ++c) cpus[n++] = c;
    else if (n < 1024)
      cpus[n++] = atoi(tok);
  }
  return n;
}

int main(int argc, char **argv) {
  static int cpus[1024];
  int n = 0;
  double seconds = 2, mib = 512;
  const char *label_arg = "";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--cpus") && i + 1 < argc) n = parse_cpus(argv[++i], cpus);
    else if (!strcmp(argv[i], "--kernel") && i + 1 < argc) {
      const char *k = argv[++i];
      kernel = -1;
      for (int j = 0; j < 3; ++j)
        if (!strcmp(k, KNAMES[j])) kernel = j;
      if (kernel < 0) return 2;
    } else if (!strcmp(argv[i], "--mib") && i + 1 < argc) mib = atof(argv[++i]);
    else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
    else if (!strcmp(argv[i], "--label") && i + 1 < argc) label_arg = argv[++i];
    else {
      fprintf(stderr, "usage: bwwrite_host --cpus LIST --kernel nt|st|stosb [--mib M] [--seconds S] [--label L]\n");
      return 2;
    }
  }
  if (n == 0) return 2;
  bytes = (size_t)(mib * (1 << 20)) / CHUNK * CHUNK;
  Worker *ws = calloc(n, sizeof *ws);
  pthread_t *ts = calloc(n, sizeof *ts);
  pthread_barrier_init(&ready, NULL, n + 1);
  pthread_barrier_init(&go, NULL, n + 1);
  for (int i = 0; i < n; ++i) {
    ws[i].cpu = cpus[i], ws[i].index = i;
    pthread_create(&ts[i], NULL, run, &ws[i]);
  }
  pthread_barrier_wait(&ready);
  const double t0 = now();
  pthread_barrier_wait(&go);
  struct timespec d = {(time_t)seconds, (long)((seconds - (time_t)seconds) * 1e9)};
  nanosleep(&d, NULL);
  atomic_store(&stop, 1);
  for (int i = 0; i < n; ++i) pthread_join(ts[i], NULL);
  double t1 = t0;
  for (int i = 0; i < n; ++i) t1 = ws[i].end > t1 ? ws[i].end : t1;
  double total = 0, node_bytes[MAXNODE] = {0};
  int node_threads[MAXNODE] = {0}, misplaced = 0;
  uint64_t wrong = 0;
  for (int i = 0; i < n; ++i) {
    total += ws[i].bytes_done;
    const int nd = ws[i].node >= 0 && ws[i].node < MAXNODE ? ws[i].node : 0;
    node_bytes[nd] += ws[i].bytes_done, node_threads[nd]++;
    misplaced += ws[i].misplaced, wrong += ws[i].wrong;
  }
  const double el = t1 - t0;
  printf("{\"label\": \"%s\", \"kernel\": \"%s\", \"threads\": %d, \"mib\": %.0f, \"seconds\": %.4f, \"gb_per_s\": %.3f, "
         "\"misplaced\": %d, \"checked\": %d, \"wrong\": %llu, \"nodes\": {",
         label_arg, KNAMES[kernel], n, mib, el, total / el / 1e9, misplaced, 4096 * n, (unsigned long long)wrong);
  int first = 1;
  for (int k = 0; k < MAXNODE; ++k)
    if (node_threads[k]) {
      printf("%s\"%d\": {\"threads\": %d, \"gb_per_s\": %.3f}", first ? "" : ", ", k, node_threads[k],
             node_bytes[k] / el / 1e9);
      first = 0;
    }
  printf("}, \"cpus\": \"");
  for (int i = 0; i < n; ++i) printf("%s%d", i ? "," : "", cpus[i]);
  printf("\"}\n");
  return wrong ? 6 : 0;
}
