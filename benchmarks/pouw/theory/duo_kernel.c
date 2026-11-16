// duo's all-state kernel at small width n, exhaustively: the word runs that leave a committed value unchanged on every
// start state (y, z) in (Z/2^n)^2.
//
//   duo, per word e (the XOR key folded in, e = d ^ k) at position t with keys (a_t odd, r_t):
//     u = y ^ e;  y <- u a_t + z;  z <- rotl(u, r_t) + y
//
// Modes:
//   full   the final state (y, z): runs e != e' of L words with F_e = F_e' on every state. Since a step with word e is
//          the zero-word step after y ^= e, F_e = F_e' iff T_c H = H' for the (L - 1)-word prefixes H, H' and c the
//          last words' XOR (T_c: y ^= c), and F_e T_x = F_(e ^ x e_1), so the first words are taken as 0 and c_1 != 0.
//          A run with equal first words is a shorter run's.
//   proj   the served receipt y ^ rotl(z, n / 2) of the final state, over runs whose first words differ (taken as 0
//          and c_1 != 0; equal first words are a shorter run's). With -K, a change counts when the two receipts
//          differ by one constant on every state (8 chains XOR their receipts, so two chains' constant differences
//          cancel); without, only when they are equal.
//
// Keys: position t (1-based) uses multiplier A[idx[t]] and rotation rot[t], from the cyclic patterns given with
// -i (key indices) and -r (rotations, reduced mod n), starting at phase -p. -a gives the multipliers; -s sweeps every
// odd assignment of them (OpenMP over assignments). Prints one line a key assignment with a kernel element, and a
// summary line: "n L mode phase configs N hit_configs C hits H nontop T", T the kernel elements with a word difference
// other than 0 and 2^(n-1).
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXL 8
#define MAXK 8
#define MAXP 12

static int n, L, mode_full = 1, kappa = 0, phase = 0, nidx = 0, nrot = 0, nkeys = 0, sweep = 0, verbose = 0;
static int idxpat[16], rotpat[16];
static uint32_t M;

static inline uint32_t rotl(uint32_t x, int r) { r %= n; return r ? (((x << r) | (x >> (n - r))) & M) : x; }

typedef struct { uint32_t a[MAXL + 1]; int r[MAXL + 1]; } Keys;

static inline void step(const Keys *k, int t, uint32_t e, uint32_t *y, uint32_t *z) {
  uint32_t u = *y ^ e;
  *y = (u * k->a[t] + *z) & M;
  *z = (rotl(u, k->r[t]) + *y) & M;
}

typedef struct { uint64_t fp; uint64_t id; } Ent;

static int cmp_ent(const void *a, const void *b) {
  uint64_t x = ((const Ent *)a)->fp, y = ((const Ent *)b)->fp;
  return x < y ? -1 : x > y;
}

static int P;
static uint32_t py[MAXP], pz[MAXP];

// fingerprint of the probes' values after `len` words: full mode packs z of every probe and y ^ y(probe 0) of the rest
// (invariant under T_c); proj mode packs each probe's receipt (xor probe 0's under -K)
static inline uint64_t fingerprint(const uint32_t *y, const uint32_t *z) {
  uint64_t f = 0;
  if (mode_full) {
    for (int i = 0; i < P; i++) f = (f << n) | z[i];
    for (int i = 1; i < P; i++) f = (f << n) | (y[i] ^ y[0]);
  } else {
    uint32_t r0 = y[0] ^ rotl(z[0], n / 2);
    for (int i = kappa ? 1 : 0; i < P; i++) f = (f << n) | ((y[i] ^ rotl(z[i], n / 2)) ^ (kappa ? r0 : 0));
  }
  return f;
}

static void decode(uint64_t id, int len, uint32_t *e) {
  for (int t = len; t >= 1; t--) { e[t] = id & M; id >>= n; }
}

static void run(const Keys *k, const uint32_t *e, int len, uint32_t y, uint32_t z, uint32_t *oy, uint32_t *oz) {
  for (int t = 1; t <= len; t++) step(k, t, e[t], &y, &z);
  *oy = y;
  *oz = z;
}

// exact check on every state of the candidate pair (ids of `len`-word runs)
static int verify(const Keys *k, uint64_t ida, uint64_t idb, int len) {
  uint32_t ea[MAXL + 1], eb[MAXL + 1];
  decode(ida, len, ea);
  decode(idb, len, eb);
  int64_t want = -1;
  for (uint32_t y = 0; y <= M; y++)
    for (uint32_t z = 0; z <= M; z++) {
      uint32_t ya, za, yb, zb;
      run(k, ea, len, y, z, &ya, &za);
      run(k, eb, len, y, z, &yb, &zb);
      int64_t d;
      if (mode_full) {
        if (za != zb) return 0;
        d = ya ^ yb;
      } else {
        d = (ya ^ rotl(za, n / 2)) ^ (yb ^ rotl(zb, n / 2));
        if (!kappa && d) return 0;
      }
      if (want < 0) want = d;
      else if (d != want) return 0;
    }
  return 1;
}

typedef struct { Ent *tab; uint64_t ntab, hits, nontop; const Keys *k; int len; } Ctx;

// whether every word difference of a verified pair (and, in full mode, the last word's, read off one state) is 0 or
// the top bit 2^(n-1)
static int all_top(const Keys *k, uint64_t ida, uint64_t idb, int len) {
  uint32_t ea[MAXL + 1], eb[MAXL + 1], t = 1u << (n - 1);
  decode(ida, len, ea);
  decode(idb, len, eb);
  for (int s = 1; s <= len; s++) if ((ea[s] ^ eb[s]) & (t - 1)) return 0;
  if (mode_full) {
    uint32_t ya, za, yb, zb;
    run(k, ea, len, 0, 0, &ya, &za);
    run(k, eb, len, 0, 0, &yb, &zb);
    if ((ya ^ yb) & (t - 1)) return 0;
  }
  return 1;
}

// depth-first over words 2..len from the probes' states after word 1, calling `leaf` with each run's fingerprint
static void dfs(Ctx *c, int t, uint64_t id, uint32_t *y, uint32_t *z, int build, uint64_t *fill) {
  if (t > c->len) {
    uint64_t f = fingerprint(y, z);
    if (build) { c->tab[*fill].fp = f; c->tab[*fill].id = id; (*fill)++; return; }
    uint64_t lo = 0, hi = c->ntab;
    while (lo < hi) { uint64_t mid = (lo + hi) / 2; if (c->tab[mid].fp < f) lo = mid + 1; else hi = mid; }
    for (uint64_t j = lo; j < c->ntab && c->tab[j].fp == f; j++)
      if (verify(c->k, c->tab[j].id, id, c->len)) {
        c->hits++;
        c->nontop += !all_top(c->k, c->tab[j].id, id, c->len);
        if (verbose && c->hits <= 4) {
          uint32_t ea[MAXL + 1], eb[MAXL + 1];
          decode(c->tab[j].id, c->len, ea);
          decode(id, c->len, eb);
          printf("  kernel:");
          for (int s = 1; s <= c->len; s++) printf(" %u", ea[s]);
          printf(" |");
          for (int s = 1; s <= c->len; s++) printf(" %u", eb[s]);
          printf("\n");
        }
      }
    return;
  }
  uint32_t ny[MAXP], nz[MAXP];
  for (uint32_t e = 0; e <= M; e++) {
    for (int i = 0; i < P; i++) { ny[i] = y[i]; nz[i] = z[i]; step(c->k, t, e, &ny[i], &nz[i]); }
    dfs(c, t + 1, (id << n) | e, ny, nz, build, fill);
  }
}

static uint64_t kernel(const Keys *k, uint64_t *nontop) {
  Ctx c = {0};
  c.k = k;
  c.len = mode_full ? L - 1 : L;
  if (c.len == 0) return 0;  // full, L = 1: T_c = id forces c = 0
  c.ntab = 1ull << (n * (c.len - 1));
  c.tab = malloc(c.ntab * sizeof(Ent));
  uint32_t y[MAXP], z[MAXP];
  uint64_t fill = 0;
  for (int i = 0; i < P; i++) { y[i] = py[i]; z[i] = pz[i]; step(k, 1, 0, &y[i], &z[i]); }
  dfs(&c, 2, 0, y, z, 1, &fill);
  qsort(c.tab, c.ntab, sizeof(Ent), cmp_ent);
  for (uint32_t e1 = 1; e1 <= M; e1++) {
    for (int i = 0; i < P; i++) { y[i] = py[i]; z[i] = pz[i]; step(k, 1, e1, &y[i], &z[i]); }
    dfs(&c, 2, e1, y, z, 0, &fill);
  }
  free(c.tab);
  *nontop = c.nontop;
  return c.hits;
}

static void keys_of(Keys *k, const uint32_t *a) {
  for (int t = 1; t <= L; t++) {
    int s = phase + t - 1;
    k->a[t] = a[idxpat[s % nidx]] & M;
    k->r[t] = rotpat[s % nrot] % n;
  }
}

static int parse_list(const char *s, int *out) {
  int c = 0;
  char *b = strdup(s), *tok = strtok(b, ",");
  while (tok) { out[c++] = (int)strtoul(tok, 0, 0); tok = strtok(0, ","); }
  free(b);
  return c;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: duo_kernel n L full|proj [-K] [-i idx,..] [-r rot,..] [-p phase] [-a a,..|-s] [-v]\n");
    return 2;
  }
  n = atoi(argv[1]);
  L = atoi(argv[2]);
  mode_full = !strcmp(argv[3], "full");
  if (n < 2 || n > 10 || L < 1 || L > MAXL) { fprintf(stderr, "n in 2..10, L in 1..%d\n", MAXL); return 2; }
  M = (1u << n) - 1;
  idxpat[0] = 0; nidx = 1; rotpat[0] = 1; nrot = 1;
  int ai[MAXK] = {0}, na = 0;
  for (int i = 4; i < argc; i++) {
    if (!strcmp(argv[i], "-K")) kappa = 1;
    else if (!strcmp(argv[i], "-s")) sweep = 1;
    else if (!strcmp(argv[i], "-v")) verbose = 1;
    else if (!strcmp(argv[i], "-i")) nidx = parse_list(argv[++i], idxpat);
    else if (!strcmp(argv[i], "-r")) nrot = parse_list(argv[++i], rotpat);
    else if (!strcmp(argv[i], "-p")) phase = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-a")) na = parse_list(argv[++i], ai);
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
  }
  for (int i = 0; i < nidx; i++) if (idxpat[i] + 1 > nkeys) nkeys = idxpat[i] + 1;
  for (int i = 0; i < nrot; i++)
    if (rotpat[i] % n == 0) fprintf(stderr, "note: rotation %d is 0 mod %d (the top flips of two words cancel)\n", rotpat[i], n);
  P = mode_full ? (64 / (2 * n) < MAXP ? 64 / (2 * n) : MAXP) : (64 / n < MAXP ? 64 / n : MAXP);
  if (mode_full && P < 2) P = 2;
  uint64_t seed = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < P; i++) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    py[i] = (seed >> 33) & M;
    pz[i] = (seed >> 13) & M;
  }
  // the keys that matter: positions 1..L-1 in full mode, 1..L in proj mode
  int used[MAXK] = {0}, nused = 0, last = mode_full ? L - 1 : L;
  for (int t = 1; t <= last; t++) {
    int j = idxpat[(phase + t - 1) % nidx];
    if (!used[j]) { used[j] = 1; nused++; }
  }
  uint64_t configs = 0, hit_configs = 0, hits = 0, nontop = 0;
  if (!sweep) {
    if (na < nkeys) { fprintf(stderr, "need %d multipliers (-a) or -s\n", nkeys); return 2; }
    uint32_t a[MAXK];
    for (int i = 0; i < nkeys; i++) a[i] = (uint32_t)ai[i];
    Keys k;
    keys_of(&k, a);
    hits = kernel(&k, &nontop);
    configs = 1;
    hit_configs = hits > 0;
    if (hits) printf("hit keys %s: %llu\n", argv[3], (unsigned long long)hits);
  } else {
    int pos[MAXK], np = 0;
    for (int j = 0; j < nkeys; j++) if (used[j]) pos[np++] = j;
    uint64_t total = 1ull << ((n - 1) * np);
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : configs, hit_configs, hits, nontop)
    for (uint64_t c = 0; c < total; c++) {
      uint32_t a[MAXK];
      for (int j = 0; j < nkeys; j++) a[j] = 1;
      uint64_t x = c;
      for (int q = 0; q < np; q++) { a[pos[q]] = (uint32_t)(((x & ((1u << (n - 1)) - 1)) << 1) | 1); x >>= n - 1; }
      Keys k;
      keys_of(&k, a);
      uint64_t nt = 0, h = kernel(&k, &nt);
      configs++;
      nontop += nt;
      if (h) {
        hit_configs++;
        hits += h;
#pragma omp critical
        {
          printf("hit a=");
          for (int j = 0; j < nkeys; j++) printf("%s%u", j ? "," : "", a[j]);
          printf(" r=");
          for (int t = 1; t <= L; t++) printf("%s%d", t > 1 ? "," : "", k.r[t]);
          printf(": %llu\n", (unsigned long long)h);
          fflush(stdout);
        }
      }
    }
  }
  printf("n %d L %d %s%s phase %d configs %llu hit_configs %llu hits %llu nontop %llu\n", n, L,
         mode_full ? "full" : "proj", kappa ? "-K" : "", phase, (unsigned long long)configs,
         (unsigned long long)hit_configs, (unsigned long long)hits, (unsigned long long)nontop);
  return 0;
}
