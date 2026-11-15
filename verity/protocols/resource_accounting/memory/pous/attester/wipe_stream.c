// The streamed wipe's fill on the memory challenger's CPUs (streamed_wipe.py, wipe_party.py):
//
//   L_i = SHA-512(LABEL_TAG ‖ K ‖ u64be i)      75 bytes, one SHA-512 compression a 64-byte label
//
// The key K is read from stdin as 64 hex digits, never from the command line. Each mode prints one JSON line:
//
//   check  --start S --count C                          then the C labels from S, a hex line each (the gates)
//   rate   --labels N [--threads T] [--chunk C]         N labels from 0 derived and dropped: the derivation alone
//   stream --labels N --fd FD [--threads T] [--chunk C] N labels from 0 written to FD in order as wipe_party's fill
//                                                       frames, F ‖ u64 first ‖ u32 count ‖ labels, C labels a frame
//
// T threads (by default one a CPU this process may run on) derive frames into a ring of 2T buffers (frame c in buffer
// c mod 2T, derived by thread c mod T) and the main thread writes them in order. Exit 0; 2 on a usage error; 4 when FD's reader went away (the prover left).
//
// Build: cc -O3 -pthread wipe_stream.c -o wipe_stream
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const char LABEL_TAG[] = "verity-pous/streamed-wipe/label/v1";  // + its NUL: 35 bytes
enum { TAG_BYTES = 35, KEY_BYTES = 32, MSG_BYTES = TAG_BYTES + KEY_BYTES + 8, HEAD_BYTES = 13, LABEL_BYTES = 64 };

static const uint64_t K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL,
    0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
    0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
    0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, 0x983e5152ee66dfabULL,
    0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
    0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL,
    0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
    0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, 0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
    0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL,
    0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL, 0xca273eceea26619cULL,
    0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
    0x113f9804bef90daeULL, 0x1b710b35131c471bULL, 0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
    0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL};
static const uint64_t IV[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
                               0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                               0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

static inline uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

// The padded block at index 0 as big-endian words, and the state after rounds 0 to 7, which read only words 0 to 7:
// the tag and the key, the same for every label.
typedef struct {
  uint64_t w[16];
  uint64_t s8[8];
} Keyed;

#define ROUND(a, b, c, d, e, f, g, h, k, x)                                                                \
  do {                                                                                                     \
    const uint64_t t1 = h + (rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41)) + ((e & f) ^ (~e & g)) + (k) + (x); \
    const uint64_t t2 = (rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));       \
    h = g;                                                                                                 \
    g = f;                                                                                                 \
    f = e;                                                                                                 \
    e = d + t1;                                                                                            \
    d = c;                                                                                                 \
    c = b;                                                                                                 \
    b = a;                                                                                                 \
    a = t1 + t2;                                                                                           \
  } while (0)

static void keyed(const uint8_t key[KEY_BYTES], Keyed* k) {
  uint8_t m[128] = {0};
  memcpy(m, LABEL_TAG, TAG_BYTES);
  memcpy(m + TAG_BYTES, key, KEY_BYTES);
  m[MSG_BYTES] = 0x80;
  const uint64_t bits = (uint64_t)MSG_BYTES * 8;
  for (int b = 0; b < 8; ++b) m[120 + b] = (uint8_t)(bits >> (8 * (7 - b)));
  for (int t = 0; t < 16; ++t) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b) v = (v << 8) | m[8 * t + b];
    k->w[t] = v;
  }
  uint64_t a = IV[0], b = IV[1], c = IV[2], d = IV[3], e = IV[4], f = IV[5], g = IV[6], h = IV[7];
  for (int t = 0; t < 8; ++t) ROUND(a, b, c, d, e, f, g, h, K[t], k->w[t]);
  k->s8[0] = a, k->s8[1] = b, k->s8[2] = c, k->s8[3] = d, k->s8[4] = e, k->s8[5] = f, k->s8[6] = g, k->s8[7] = h;
}

// L_i into out, big-endian as hashlib's digest. The index sits at bytes 67 to 74: word 8's low 40 bits and word 9's top 24.
static inline void label(const Keyed* k, uint64_t i, uint8_t* out) {
  uint64_t w[16];
  memcpy(w, k->w, sizeof w);
  w[8] |= i >> 24;
  w[9] |= (i & 0xffffffULL) << 40;
  uint64_t a = k->s8[0], b = k->s8[1], c = k->s8[2], d = k->s8[3], e = k->s8[4], f = k->s8[5], g = k->s8[6],
           h = k->s8[7];
  for (int t = 8; t < 16; ++t) ROUND(a, b, c, d, e, f, g, h, K[t], w[t]);
  for (int t = 16; t < 80; ++t) {
    const uint64_t w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
    w[t & 15] += (rotr(w15, 1) ^ rotr(w15, 8) ^ (w15 >> 7)) + w[(t + 9) & 15] + (rotr(w2, 19) ^ rotr(w2, 61) ^ (w2 >> 6));
    ROUND(a, b, c, d, e, f, g, h, K[t], w[t & 15]);
  }
  const uint64_t s[8] = {IV[0] + a, IV[1] + b, IV[2] + c, IV[3] + d, IV[4] + e, IV[5] + f, IV[6] + g, IV[7] + h};
  for (int q = 0; q < 8; ++q)
    for (int y = 0; y < 8; ++y) out[8 * q + y] = (uint8_t)(s[q] >> (56 - 8 * y));
}

static void put_be(uint8_t* p, uint64_t v, int n) {
  for (int y = 0; y < n; ++y) p[y] = (uint8_t)(v >> (8 * (n - 1 - y)));
}

typedef struct {
  uint8_t* buf;
  size_t len;
  sem_t full, empty;
} Slot;

typedef struct {
  const Keyed* key;
  Slot* slots;
  uint64_t labels, chunk, frames;
  int threads, ring, id;
} Work;

static void* derive(void* arg) {
  const Work* w = (const Work*)arg;
  for (uint64_t c = (uint64_t)w->id; c < w->frames; c += (uint64_t)w->threads) {
    Slot* s = &w->slots[c % (uint64_t)w->ring];
    sem_wait(&s->empty);
    const uint64_t first = c * w->chunk, count = w->labels - first < w->chunk ? w->labels - first : w->chunk;
    s->buf[0] = 'F';
    put_be(s->buf + 1, first, 8);
    put_be(s->buf + 9, count, 4);
    for (uint64_t t = 0; t < count; ++t) label(w->key, first + t, s->buf + HEAD_BYTES + LABEL_BYTES * t);
    s->len = HEAD_BYTES + LABEL_BYTES * count;
    sem_post(&s->full);
  }
  return NULL;
}

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int write_all(int fd, const uint8_t* p, size_t n) {
  while (n) {
    const ssize_t r = write(fd, p, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return -1;
    p += r;
    n -= (size_t)r;
  }
  return 0;
}

// The frames of labels [0, labels) to fd in order, or derived and dropped when fd < 0.
static int pump(const Keyed* key, uint64_t labels, uint64_t chunk, int threads, int fd) {
  const uint64_t frames = (labels + chunk - 1) / chunk;
  const int ring = 2 * threads;
  Slot* slots = calloc((size_t)ring, sizeof *slots);
  Work* work = calloc((size_t)threads, sizeof *work);
  pthread_t* tid = calloc((size_t)threads, sizeof *tid);
  if (!slots || !work || !tid) return 2;
  for (int r = 0; r < ring; ++r) {
    slots[r].buf = malloc(HEAD_BYTES + LABEL_BYTES * chunk);
    if (!slots[r].buf) return 2;
    sem_init(&slots[r].full, 0, 0);
    sem_init(&slots[r].empty, 0, 1);
  }
  const double t0 = now_s();
  for (int t = 0; t < threads; ++t) {
    work[t] = (Work){key, slots, labels, chunk, frames, threads, ring, t};
    pthread_create(&tid[t], NULL, derive, &work[t]);
  }
  uint64_t bytes = 0;
  for (uint64_t c = 0; c < frames; ++c) {
    Slot* s = &slots[c % (uint64_t)ring];
    sem_wait(&s->full);
    if (fd >= 0 && write_all(fd, s->buf, s->len) != 0) {
      printf("{\"error\": \"the reader went away after %llu of %llu frames\", \"errno\": %d}\n", (unsigned long long)c,
             (unsigned long long)frames, errno);
      fflush(stdout);
      _exit(4);
    }
    bytes += s->len - HEAD_BYTES;
    sem_post(&s->empty);
  }
  for (int t = 0; t < threads; ++t) pthread_join(tid[t], NULL);
  const double dt = now_s() - t0;
  printf("{\"mode\": \"%s\", \"labels\": %llu, \"bytes\": %llu, \"chunk\": %llu, \"threads\": %d, \"s\": %.6f, "
         "\"GBps\": %.4f, \"labels_per_s\": %.6e}\n",
         fd >= 0 ? "stream" : "rate", (unsigned long long)labels, (unsigned long long)bytes, (unsigned long long)chunk,
         threads, dt, dt > 0 ? (double)bytes / dt / 1e9 : 0.0, dt > 0 ? (double)labels / dt : 0.0);
  return 0;
}

static int read_key(uint8_t key[KEY_BYTES]) {
  char line[256];
  if (!fgets(line, sizeof line, stdin)) return -1;
  size_t n = strcspn(line, "\r\n");
  if (n != 2 * KEY_BYTES) return -1;
  for (int i = 0; i < KEY_BYTES; ++i) {
    unsigned v;
    if (sscanf(line + 2 * i, "%2x", &v) != 1) return -1;
    key[i] = (uint8_t)v;
  }
  return 0;
}

static int usage(void) {
  fprintf(stderr,
          "usage: wipe_stream check --start S --count C | rate --labels N [--threads T] [--chunk C] | "
          "stream --labels N --fd FD [--threads T] [--chunk C]   (the key: 64 hex digits on stdin)\n");
  return 2;
}

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const char* mode = argv[1];
  uint64_t start = 0, count = 16, labels = 0, chunk = 1ULL << 14;
  cpu_set_t cpus;
  long threads = sched_getaffinity(0, sizeof cpus, &cpus) == 0 ? CPU_COUNT(&cpus) : sysconf(_SC_NPROCESSORS_ONLN);
  int fd = -1;
  for (int i = 2; i + 1 < argc; i += 2) {
    const char *k = argv[i], *v = argv[i + 1];
    if (!strcmp(k, "--start")) start = strtoull(v, NULL, 10);
    else if (!strcmp(k, "--count")) count = strtoull(v, NULL, 10);
    else if (!strcmp(k, "--labels")) labels = strtoull(v, NULL, 10);
    else if (!strcmp(k, "--chunk")) chunk = strtoull(v, NULL, 10);
    else if (!strcmp(k, "--threads")) threads = strtol(v, NULL, 10);
    else if (!strcmp(k, "--fd")) fd = (int)strtol(v, NULL, 10);
    else return usage();
  }
  if (argc % 2) return usage();
  uint8_t raw[KEY_BYTES];
  if (read_key(raw) != 0) {
    fprintf(stderr, "wipe_stream: the key is 64 hex digits on stdin\n");
    return 2;
  }
  Keyed key;
  keyed(raw, &key);
  memset(raw, 0, sizeof raw);
  if (!strcmp(mode, "check")) {
    printf("{\"mode\": \"check\", \"start\": %llu, \"count\": %llu}\n", (unsigned long long)start,
           (unsigned long long)count);
    uint8_t out[LABEL_BYTES];
    for (uint64_t t = 0; t < count; ++t) {
      label(&key, start + t, out);
      for (int b = 0; b < LABEL_BYTES; ++b) printf("%02x", out[b]);
      printf("\n");
    }
    return 0;
  }
  if (threads < 1 || chunk < 1 || chunk > (1ULL << 20) || labels < 1) return usage();
  if (!strcmp(mode, "rate")) return pump(&key, labels, chunk, (int)threads, -1);
  if (!strcmp(mode, "stream")) {
    if (fd < 0) return usage();
    signal(SIGPIPE, SIG_IGN);
    return pump(&key, labels, chunk, (int)threads, fd);
  }
  return usage();
}
