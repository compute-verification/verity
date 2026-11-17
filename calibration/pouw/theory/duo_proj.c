// (D-proj) at small width n: word runs whose served receipts R = y ^ rotl(z, h) differ by one constant on every start
// state, exhaustively.
//
//   duo, per word e (the XOR key folded in, e = d ^ k) at position t with keys (a_t, r_t):
//     u = y ^ e;  y <- u a_t + z;  z <- rotl(u, r_t) + y
//
// Two runs e != e' of L words are a kernel element when, over every start state (y, z) in (Z/2^n)^2,
//   exact   R_e = R_e'
//   xor     R_e ^ R_e' is one constant kappa (the served receipt XORs 8 chains, so equal constants in two chains cancel)
//   add     R_e - R_e' is one constant kappa
// With -F the compared value is the full state, R = (y << n) | z, and a kernel element under xor is a pair of runs
// whose states differ by one constant (kappa_y, kappa_z) on every start state: what two chains need to cancel in a
// checkpoint that XORs every chain's y and every chain's z (start states independent per chain); kappa prints as
// (kappa_y << n) | kappa_z.
// Since F_e T_x = F_(e ^ x e_1) for T_x: y ^= x, and T_x permutes the start states, the first run's first word is taken
// as 0 and the second's as e'_1 != 0; a pair with equal first words is a shorter run's (one step permutes the states).
//
// Search: a table of the fingerprints of every run with e_1 = 0 (2^(n(L-1)) runs; a fingerprint packs P probe
// states' receipts, relative to probe 0's under xor/add), probed by every run with e_1 != 0; each fingerprint match
// is checked on every state. With -C, the start states are instead the curve {(y, pi(y))} for pi a fixed odd-multiply
// xorshift bijection (a scaled stand-in for the served z = mix32(y + golden)); the first-word reduction then fails, so
// every pair of runs is compared by a sort of all 2^(nL) fingerprints.
//
// Keys: -a a_1,..,a_L and -r r_1,..,r_L (one multiplier and one rotation a position, at width n); -s sweeps every odd
// multiplier at every position (OpenMP over assignments). -h sets the receipt rotation (default n / 2). Prints each
// key assignment with a kernel element and up to -v kernel elements (words, word differences, kappa; one per
// assignment under -s), and a summary line
//   "n L mode h configs N hit_configs C hits H toponly T kappas K"
// T the elements whose word differences are all 0 or 2^(n-1), K the distinct kappas seen (up to 16 listed).
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXL 10
#define MAXP 13

static int n, L, mode = 1, h = -1, sweep = 0, verbose = 0, curve = 0, full = 0;  // mode 0 exact, 1 xor, 2 add
static uint32_t M, T;
static int P, Q, W;  // probes, packed values, value width
static uint32_t py[MAXP], pz[MAXP];

typedef struct { uint32_t a[MAXL + 1]; int r[MAXL + 1]; } Keys;

static inline uint32_t rotl(uint32_t x, int r) { r %= n; return r ? (((x << r) | (x >> (n - r))) & M) : x; }
static inline void step(const Keys *k, int t, uint32_t e, uint32_t *y, uint32_t *z) {
  uint32_t u = *y ^ e;
  *y = (u * k->a[t] + *z) & M;
  *z = (rotl(u, k->r[t]) + *y) & M;
}
static inline uint32_t receipt(uint32_t y, uint32_t z) { return full ? (y << n) | z : y ^ rotl(z, h); }
static inline uint32_t diff(uint32_t r, uint32_t s) { return mode == 2 ? (r - s) & M : r ^ s; }
static inline uint32_t curve_z(uint32_t y) {
  uint32_t x = y;
  x ^= x >> ((n + 1) / 2);
  x = (x * 0x2C1B3C6Du) & M;
  x ^= x >> (n / 2);
  x = (x * 0x297A2D39u) & M;
  return x ^ (x >> ((n + 1) / 2));
}

static inline uint64_t fingerprint(const uint32_t *y, const uint32_t *z) {
  uint64_t f = 0;
  if (mode == 0) {
    for (int i = 0; i < Q; i++) f = (f << W) | receipt(y[i], z[i]);
  } else {
    uint32_t r0 = receipt(y[0], z[0]);
    for (int i = 1; i <= Q; i++) f = (f << W) | diff(receipt(y[i], z[i]), r0);
  }
  return f;
}

static void decode(uint64_t id, int len, uint32_t *e) {
  for (int t = len; t >= 1; t--) { e[t] = id & M; id >>= n; }
}

static inline uint32_t run_receipt(const Keys *k, const uint32_t *e, uint32_t y, uint32_t z) {
  for (int t = 1; t <= L; t++) step(k, t, e[t], &y, &z);
  return receipt(y, z);
}

// every start state (or every curve state): the receipts' difference is one constant (exact: zero); *kap gets it
static int verify(const Keys *k, const uint32_t *ea, const uint32_t *eb, uint32_t *kap) {
  int64_t want = mode == 0 ? 0 : -1;
  uint32_t zmax = curve ? 0 : M;
  for (uint32_t y = 0; y <= M; y++)
    for (uint32_t zz = 0; zz <= zmax; zz++) {
      uint32_t z = curve ? curve_z(y) : zz;
      uint32_t d = diff(run_receipt(k, ea, y, z), run_receipt(k, eb, y, z));
      if (want < 0) want = d;
      else if (d != (uint32_t)want) return 0;
    }
  *kap = (uint32_t)want;
  return 1;
}

typedef struct { uint64_t fp, id; } Ent;
typedef struct {
  const Keys *k;
  Ent *tab;          // open addressing, id == ~0 empty
  uint64_t mask;
  uint64_t hits, toponly;
  uint32_t kappas[16];
  int nkappa;
  int printed, limit;
} Ctx;

static inline uint64_t hslot(uint64_t fp, uint64_t mask) { return (fp * 0x9E3779B97F4A7C15ull >> 17) & mask; }

static void record(Ctx *c, const uint32_t *ea, const uint32_t *eb, uint32_t kap) {
  int top = 1;
  for (int t = 1; t <= L; t++) if ((ea[t] ^ eb[t]) & (T - 1)) top = 0;
#pragma omp critical
  {
    c->hits++;
    c->toponly += top;
    int seen = 0;
    for (int i = 0; i < c->nkappa; i++) seen |= c->kappas[i] == kap;
    if (!seen && c->nkappa < 16) c->kappas[c->nkappa++] = kap;
    if (c->printed < c->limit) {
      c->printed++;
      printf("  kernel:");
      for (int t = 1; t <= L; t++) printf(" %u", ea[t]);
      printf(" |");
      for (int t = 1; t <= L; t++) printf(" %u", eb[t]);
      printf(" | diff");
      for (int t = 1; t <= L; t++) printf(" %u", ea[t] ^ eb[t]);
      printf(" | kappa %u\n", kap);
      fflush(stdout);
    }
  }
}

// depth-first over words t..L from the probes' states, inserting (build) or looking up each run's fingerprint
static void dfs(Ctx *c, int t, uint64_t id, const uint32_t *y, const uint32_t *z, int build) {
  if (t > L) {
    uint64_t f = fingerprint(y, z), s = hslot(f, c->mask);
    if (build) {
      while (c->tab[s].id != ~0ull) s = (s + 1) & c->mask;
      c->tab[s].fp = f;
      c->tab[s].id = id;
      return;
    }
    for (; c->tab[s].id != ~0ull; s = (s + 1) & c->mask) {
      if (c->tab[s].fp != f) continue;
      uint32_t ea[MAXL + 1], eb[MAXL + 1], kap;
      decode(c->tab[s].id, L, ea);
      decode(id, L, eb);
      if (verify(c->k, ea, eb, &kap)) record(c, ea, eb, kap);
    }
    return;
  }
  uint32_t ny[MAXP], nz[MAXP];
  for (uint32_t e = 0; e <= M; e++) {
    for (int i = 0; i < P; i++) { ny[i] = y[i]; nz[i] = z[i]; step(c->k, t, e, &ny[i], &nz[i]); }
    dfs(c, t + 1, (id << n) | e, ny, nz, build);
  }
}

static void kernel_states(Ctx *c, int inner_parallel) {
  uint64_t ntab = 1ull << (n * (L - 1)), slots = 1;
  while (slots < 2 * ntab) slots <<= 1;
  c->mask = slots - 1;
  c->tab = malloc(slots * sizeof(Ent));
  if (!c->tab) { fprintf(stderr, "out of memory (%llu slots)\n", (unsigned long long)slots); exit(3); }
  memset(c->tab, 0xFF, slots * sizeof(Ent));
  uint32_t y[MAXP], z[MAXP];
  for (int i = 0; i < P; i++) { y[i] = py[i]; z[i] = pz[i]; step(c->k, 1, 0, &y[i], &z[i]); }
  dfs(c, 2, 0, y, z, 1);
#pragma omp parallel for schedule(dynamic, 1) if (inner_parallel)
  for (uint32_t e1 = 1; e1 <= M; e1++) {
    uint32_t yy[MAXP], zz[MAXP];
    for (int i = 0; i < P; i++) { yy[i] = py[i]; zz[i] = pz[i]; step(c->k, 1, e1, &yy[i], &zz[i]); }
    dfs(c, 2, e1, yy, zz, 0);
  }
  free(c->tab);
}

static int cmp_ent(const void *a, const void *b) {
  uint64_t x = ((const Ent *)a)->fp, y = ((const Ent *)b)->fp;
  return x < y ? -1 : x > y;
}

// curve start states: every pair of runs, by sorting all fingerprints
static void kernel_curve(Ctx *c) {
  uint64_t N = 1ull << (n * L);
  Ent *all = malloc(N * sizeof(Ent));
  if (!all) { fprintf(stderr, "out of memory\n"); exit(3); }
#pragma omp parallel for schedule(static)
  for (uint64_t id = 0; id < N; id++) {
    uint32_t e[MAXL + 1], y[MAXP], z[MAXP];
    decode(id, L, e);
    for (int i = 0; i < P; i++) {
      y[i] = py[i]; z[i] = pz[i];
      for (int t = 1; t <= L; t++) step(c->k, t, e[t], &y[i], &z[i]);
    }
    all[id].fp = fingerprint(y, z);
    all[id].id = id;
  }
  qsort(all, N, sizeof(Ent), cmp_ent);
  for (uint64_t i = 0; i < N; i++)
    for (uint64_t j = i + 1; j < N && all[j].fp == all[i].fp; j++) {
      uint32_t ea[MAXL + 1], eb[MAXL + 1], kap;
      decode(all[i].id, L, ea);
      decode(all[j].id, L, eb);
      if (verify(c->k, ea, eb, &kap)) record(c, ea, eb, kap);
    }
  free(all);
}

static int parse_list(const char *s, uint32_t *out) {
  int c = 0;
  char *b = strdup(s), *tok = strtok(b, ",");
  while (tok && c < MAXL) { out[c++] = (uint32_t)strtoul(tok, 0, 0); tok = strtok(0, ","); }
  free(b);
  return c;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: duo_proj n L exact|xor|add [-h H] [-a a,..] [-r r,..] [-s] [-C] [-F] [-v N]\n");
    return 2;
  }
  n = atoi(argv[1]);
  L = atoi(argv[2]);
  mode = !strcmp(argv[3], "exact") ? 0 : !strcmp(argv[3], "xor") ? 1 : !strcmp(argv[3], "add") ? 2 : -1;
  if (mode < 0 || n < 2 || n > 16 || L < 1 || L > MAXL) { fprintf(stderr, "n in 2..16, L in 1..%d\n", MAXL); return 2; }
  M = (1u << n) - 1;
  T = 1u << (n - 1);
  uint32_t a[MAXL] = {0}, r[MAXL] = {0};
  int na = 0, nr = 0;
  for (int i = 4; i < argc; i++) {
    if (!strcmp(argv[i], "-s")) sweep = 1;
    else if (!strcmp(argv[i], "-C")) curve = 1;
    else if (!strcmp(argv[i], "-F")) full = 1;
    else if (!strcmp(argv[i], "-v")) verbose = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-h")) h = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-a")) na = parse_list(argv[++i], a);
    else if (!strcmp(argv[i], "-r")) nr = parse_list(argv[++i], r);
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
  }
  if (h < 0) h = n / 2;
  if (nr != L || (!sweep && na != L)) { fprintf(stderr, "need L rotations (-r) and L multipliers (-a) or -s\n"); return 2; }
  if ((uint64_t)n * (L - (curve ? 0 : 1)) > 40) { fprintf(stderr, "n (L - 1) > 40: too large\n"); return 2; }
  if (full && mode == 2) { fprintf(stderr, "-F compares states under exact or xor\n"); return 2; }
  W = full ? 2 * n : n;
  Q = 64 / W < MAXP - 1 ? 64 / W : MAXP - 1;
  P = mode == 0 ? Q : Q + 1;
  uint64_t seed = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < P; i++) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    py[i] = (seed >> 33) & M;
    pz[i] = curve ? curve_z(py[i]) : (seed >> 13) & M;
  }
  uint64_t configs = 0, hit_configs = 0, hits = 0, toponly = 0;
  uint32_t kappas[16];
  int nkappa = 0;
  uint64_t total = sweep ? 1ull << ((n - 1) * L) : 1;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : configs, hit_configs, hits, toponly) if (sweep)
  for (uint64_t cfg = 0; cfg < total; cfg++) {
    Keys k;
    uint64_t x = cfg;
    for (int t = 1; t <= L; t++) {
      k.a[t] = sweep ? (uint32_t)(((x & ((1u << (n - 1)) - 1)) << 1) | 1) : a[t - 1] & M;
      if (sweep) x >>= n - 1;
      k.r[t] = (int)(r[t - 1] % n);
    }
    Ctx c = {0};
    c.k = &k;
    c.limit = sweep && verbose ? 1 : verbose;
    if (curve) kernel_curve(&c);
    else kernel_states(&c, !sweep);
    configs++;
    hits += c.hits;
    toponly += c.toponly;
    if (c.hits) {
      hit_configs++;
#pragma omp critical
      {
        printf("hit a=");
        for (int t = 1; t <= L; t++) printf("%s%u", t > 1 ? "," : "", k.a[t]);
        printf(" r=");
        for (int t = 1; t <= L; t++) printf("%s%d", t > 1 ? "," : "", k.r[t]);
        printf(": %llu toponly %llu kappa", (unsigned long long)c.hits, (unsigned long long)c.toponly);
        for (int i = 0; i < c.nkappa; i++) printf(" %u", c.kappas[i]);
        printf("\n");
        for (int i = 0; i < c.nkappa; i++) {
          int seen = 0;
          for (int j = 0; j < nkappa; j++) seen |= kappas[j] == c.kappas[i];
          if (!seen && nkappa < 16) kappas[nkappa++] = c.kappas[i];
        }
        fflush(stdout);
      }
    }
  }
  printf("n %d L %d %s%s%s h %d configs %llu hit_configs %llu hits %llu toponly %llu kappas %d", n, L, argv[3],
         curve ? "-curve" : "", full ? "-full" : "", h, (unsigned long long)configs, (unsigned long long)hit_configs,
         (unsigned long long)hits, (unsigned long long)toponly, nkappa);
  for (int i = 0; i < nkappa; i++) printf(" %u", kappas[i]);
  printf("\n");
  return 0;
}
