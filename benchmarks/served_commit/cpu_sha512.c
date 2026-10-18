// SHA-512 throughput per core on the host CPU: isa-l_crypto's multi-buffer managers (8 lanes in AVX-512, 4 in AVX2) against
// OpenSSL's one-buffer SHA512(), over messages of a fixed length, in THREADS threads. Each thread hashes its own MIB of messages
// (one context each), REPS times; one JSON line per (impl, len, threads). Feasibility only (cpu_sha512.sh builds and runs it).
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sha512_mb.h"

// libcrypto's, declared here so the build needs no OpenSSL headers (cpu_sha512.sh links libcrypto.so.3).
unsigned char *SHA512(const unsigned char *d, size_t n, unsigned char *md);

ISAL_SHA512_HASH_CTX *_sha512_ctx_mgr_submit_avx512(ISAL_SHA512_HASH_CTX_MGR *, ISAL_SHA512_HASH_CTX *, const void *, uint32_t,
                                                    ISAL_HASH_CTX_FLAG);
ISAL_SHA512_HASH_CTX *_sha512_ctx_mgr_flush_avx512(ISAL_SHA512_HASH_CTX_MGR *);
void _sha512_ctx_mgr_init_avx512(ISAL_SHA512_HASH_CTX_MGR *);
ISAL_SHA512_HASH_CTX *_sha512_ctx_mgr_submit_avx2(ISAL_SHA512_HASH_CTX_MGR *, ISAL_SHA512_HASH_CTX *, const void *, uint32_t,
                                                  ISAL_HASH_CTX_FLAG);
ISAL_SHA512_HASH_CTX *_sha512_ctx_mgr_flush_avx2(ISAL_SHA512_HASH_CTX_MGR *);
void _sha512_ctx_mgr_init_avx2(ISAL_SHA512_HASH_CTX_MGR *);

typedef struct {
  const char *name;
  void (*init)(ISAL_SHA512_HASH_CTX_MGR *);
  ISAL_SHA512_HASH_CTX *(*submit)(ISAL_SHA512_HASH_CTX_MGR *, ISAL_SHA512_HASH_CTX *, const void *, uint32_t, ISAL_HASH_CTX_FLAG);
  ISAL_SHA512_HASH_CTX *(*flush)(ISAL_SHA512_HASH_CTX_MGR *);
} Impl;

static const Impl IMPLS[] = {
    {"isal-avx512", _sha512_ctx_mgr_init_avx512, _sha512_ctx_mgr_submit_avx512, _sha512_ctx_mgr_flush_avx512},
    {"isal-avx2", _sha512_ctx_mgr_init_avx2, _sha512_ctx_mgr_submit_avx2, _sha512_ctx_mgr_flush_avx2},
    {"openssl", NULL, NULL, NULL},
};

typedef struct {
  const Impl *impl;
  size_t len, n, reps;
  uint8_t *buf;
  ISAL_SHA512_HASH_CTX *ctx;
  uint8_t out[64];
  pthread_barrier_t *bar;
  double secs;
  int ok;
} Job;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void run_once(Job *j) {
  if (!j->impl->init) {
    for (size_t i = 0; i < j->n; i++) SHA512(j->buf + i * j->len, j->len, j->out);
    return;
  }
  ISAL_SHA512_HASH_CTX_MGR *mgr = aligned_alloc(64, (sizeof *mgr + 63) / 64 * 64);
  j->impl->init(mgr);
  for (size_t i = 0; i < j->n; i++) {
    isal_hash_ctx_init(&j->ctx[i]);
    j->impl->submit(mgr, &j->ctx[i], j->buf + i * j->len, (uint32_t)j->len, ISAL_HASH_ENTIRE);
  }
  while (j->impl->flush(mgr)) {
  }
  free(mgr);
}

static void *worker(void *arg) {
  Job *j = arg;
  run_once(j);
  if (j->impl->init) {
    uint8_t ref[64], got[64];
    SHA512(j->buf, j->len, ref);
    for (int w = 0; w < 8; w++)
      for (int b = 0; b < 8; b++) got[8 * w + b] = (uint8_t)(j->ctx[0].job.result_digest[w] >> (56 - 8 * b));
    j->ok = j->ctx[0].status == ISAL_HASH_CTX_STS_COMPLETE && memcmp(ref, got, 64) == 0;
  } else {
    j->ok = 1;
  }
  pthread_barrier_wait(j->bar);
  double t0 = now();
  for (size_t r = 0; r < j->reps; r++) run_once(j);
  j->secs = now() - t0;
  return NULL;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s IMPLS LENS THREADS MIB REPS (comma lists)\n", argv[0]);
    return 2;
  }
  size_t mib = strtoul(argv[4], NULL, 10), reps = strtoul(argv[5], NULL, 10);
  char *impls = strdup(argv[1]), *sp1;
  for (char *iname = strtok_r(impls, ",", &sp1); iname; iname = strtok_r(NULL, ",", &sp1)) {
    const Impl *impl = NULL;
    for (size_t k = 0; k < sizeof IMPLS / sizeof *IMPLS; k++)
      if (!strcmp(IMPLS[k].name, iname)) impl = &IMPLS[k];
    if (!impl) {
      fprintf(stderr, "unknown impl %s\n", iname);
      return 2;
    }
    char *lens = strdup(argv[2]), *sp2;
    for (char *ls = strtok_r(lens, ",", &sp2); ls; ls = strtok_r(NULL, ",", &sp2)) {
      size_t len = strtoul(ls, NULL, 10), n = (mib << 20) / len;
      char *threads = strdup(argv[3]), *sp3;
      for (char *ts = strtok_r(threads, ",", &sp3); ts; ts = strtok_r(NULL, ",", &sp3)) {
        int nt = atoi(ts);
        Job *jobs = calloc(nt, sizeof *jobs);
        pthread_t *th = calloc(nt, sizeof *th);
        pthread_barrier_t bar;
        pthread_barrier_init(&bar, NULL, nt);
        for (int t = 0; t < nt; t++) {
          size_t ctx_bytes = (n * sizeof(ISAL_SHA512_HASH_CTX) + 63) / 64 * 64;
          jobs[t] = (Job){impl, len, n, reps, aligned_alloc(64, (n * len + 63) / 64 * 64),
                          impl->init ? aligned_alloc(64, ctx_bytes) : NULL, {0}, &bar, 0, 0};
          for (size_t i = 0; i < n * len; i++) jobs[t].buf[i] = (uint8_t)(i * 2654435761u >> 13 ^ t);
        }
        for (int t = 0; t < nt; t++) pthread_create(&th[t], NULL, worker, &jobs[t]);
        double secs = 0;
        int ok = 1;
        for (int t = 0; t < nt; t++) {
          pthread_join(th[t], NULL);
          secs = jobs[t].secs > secs ? jobs[t].secs : secs;
          ok &= jobs[t].ok;
        }
        double msgs = (double)n * reps * nt, blocks = msgs * ((len + 17 + 127) / 128);
        printf("{\"impl\": \"%s\", \"len\": %zu, \"threads\": %d, \"ok\": %s, \"secs\": %.4f, \"msgs_per_s\": %.0f, "
               "\"gb_per_s\": %.3f, \"mcompress_per_s\": %.2f, \"mcompress_per_s_per_thread\": %.2f}\n",
               impl->name, len, nt, ok ? "true" : "false", secs, msgs / secs, msgs * len / secs / 1e9, blocks / secs / 1e6,
               blocks / secs / 1e6 / nt);
        fflush(stdout);
        for (int t = 0; t < nt; t++) {
          free(jobs[t].buf);
          free(jobs[t].ctx);
        }
        pthread_barrier_destroy(&bar);
        free(jobs);
        free(th);
      }
      free(threads);
    }
    free(lens);
  }
  free(impls);
  return 0;
}
