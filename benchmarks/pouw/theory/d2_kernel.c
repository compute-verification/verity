// D2's all-state kernel at small width n, exhaustively: the pairs of distinct runs of L steps that leave the final state
// (y, z) unchanged on every start state in (Z/2^n)^2.
//
//   D2, one step (two words e = d0 ^ k, f = d1) at position t with keys (a_t odd, r_t):
//     u = y ^ e;  y <- u a_t + z;  z <- rotl(u, r_t) + f
//   (-x gives a per-step absorption pattern: 'x' absorbs e by XOR as above, 'a' by addition, u = y + e.)
//
// A step is a bijection of the state (u = rotr(z' - f), y = u ^ e, z = y' - u a), so two runs whose first steps agree
// are a shorter run's kernel, and only pairs whose first steps differ are new. The first step's absorption is a group
// action on y that commutes into the start state (F_(e1 * x) = F_(e1) g_x), so a pair's first e words are taken as
// (e1, 0): runs with e1 = 0 go in the table (hashed fingerprints of P probe states' finals), runs with e1 != 0 query it,
// and pairs inside the table differ in f1 alone. Every fingerprint match is checked on all 2^(2n) start states.
//
// Keys: step t uses multiplier A[idx[t]] and rotation rot[t], from cyclic patterns (-i key indices, -r rotations,
// reduced mod n). -a gives the multipliers; -s sweeps every odd assignment of them (OpenMP over assignments); -R sweeps
// every rotation in 1..n-1 at every step as well. Prints one line a key assignment with a kernel element, and a summary
// line: "n L absorb configs N hit_configs C hits H nontop T", T the kernel elements with a word difference other than 0
// and 2^(n-1).
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXL 8
#define MAXK 8
#define MAXP 16

static int n, L, nidx = 0, nrot = 0, nkeys = 0, sweep = 0, rsweep = 0, verbose = 0, nabs = 1;
static int idxpat[16], rotpat[16];
static char abspat[16] = "x";
static char rotin = 'u';
static uint32_t M;
static int P;
static uint32_t py[MAXP], pz[MAXP];

static inline uint32_t rotl(uint32_t x, int r) { r %= n; return r ? (((x << r) | (x >> (n - r))) & M) : x; }

typedef struct { uint32_t a[MAXL + 1]; int r[MAXL + 1]; char x[MAXL + 1]; } Keys;

static inline void step(const Keys *k, int t, uint32_t e, uint32_t f, uint32_t *y, uint32_t *z) {
  uint32_t y0 = *y, u = k->x[t] == 'a' ? (y0 + e) & M : y0 ^ e;
  *y = (u * k->a[t] + *z) & M;
  *z = (rotl(rotin == 'y' ? y0 : u, k->r[t]) + f) & M;
}

static inline uint64_t mix(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
  return x;
}

typedef struct { uint64_t h; uint64_t id; } Ent;

static int cmp_ent(const void *a, const void *b) {
  uint64_t x = ((const Ent *)a)->h, y = ((const Ent *)b)->h;
  return x < y ? -1 : x > y;
}

// the probes' final states packed (2n bits a probe), mixed bijectively so the table's top bits index it
static inline uint64_t fingerprint(const uint32_t *y, const uint32_t *z) {
  uint64_t f = 0;
  for (int i = 0; i < P; i++) f = (f << (2 * n)) | ((uint64_t)y[i] << n) | z[i];
  return mix(f);
}

static void decode(uint64_t id, uint32_t *e, uint32_t *f) {
  for (int t = L; t >= 1; t--) { f[t] = id & M; id >>= n; e[t] = id & M; id >>= n; }
}

static void run(const Keys *k, const uint32_t *e, const uint32_t *f, uint32_t y, uint32_t z, uint32_t *oy,
                uint32_t *oz) {
  for (int t = 1; t <= L; t++) step(k, t, e[t], f[t], &y, &z);
  *oy = y;
  *oz = z;
}

static int verify(const Keys *k, uint64_t ida, uint64_t idb) {
  uint32_t ea[MAXL + 1], fa[MAXL + 1], eb[MAXL + 1], fb[MAXL + 1];
  decode(ida, ea, fa);
  decode(idb, eb, fb);
  for (uint32_t y = 0; y <= M; y++)
    for (uint32_t z = 0; z <= M; z++) {
      uint32_t ya, za, yb, zb;
      run(k, ea, fa, y, z, &ya, &za);
      run(k, eb, fb, y, z, &yb, &zb);
      if (ya != yb || za != zb) return 0;
    }
  return 1;
}

static int all_top(uint64_t ida, uint64_t idb) {
  uint32_t ea[MAXL + 1], fa[MAXL + 1], eb[MAXL + 1], fb[MAXL + 1], low = (1u << (n - 1)) - 1;
  decode(ida, ea, fa);
  decode(idb, eb, fb);
  for (int t = 1; t <= L; t++) if (((ea[t] ^ eb[t]) & low) || (((fa[t] - fb[t]) & M) & low)) return 0;
  return 1;
}

typedef struct {
  Ent *tab;
  uint64_t ntab, hits, nontop;
  uint32_t *dir;
  int dbits;
  const Keys *k;
} Ctx;

static void report(Ctx *c, uint64_t ida, uint64_t idb) {
  c->hits++;
  c->nontop += !all_top(ida, idb);
  if (verbose && c->hits <= 4) {
    uint32_t ea[MAXL + 1], fa[MAXL + 1], eb[MAXL + 1], fb[MAXL + 1];
    decode(ida, ea, fa);
    decode(idb, eb, fb);
    printf("  kernel:");
    for (int s = 1; s <= L; s++) printf(" (%u,%u)", ea[s], fa[s]);
    printf(" |");
    for (int s = 1; s <= L; s++) printf(" (%u,%u)", eb[s], fb[s]);
    printf("\n");
  }
}

// depth-first over steps t..L from the probes' states, adding each run to the table or looking it up
static void dfs(Ctx *c, int t, uint64_t id, const uint32_t *y, const uint32_t *z, int build, uint64_t *fill) {
  if (t > L) {
    uint64_t h = fingerprint(y, z);
    if (build) { c->tab[*fill].h = h; c->tab[*fill].id = id; (*fill)++; return; }
    uint64_t b = h >> (64 - c->dbits);
    for (uint64_t j = c->dir[b]; j < c->dir[b + 1]; j++)
      if (c->tab[j].h == h && verify(c->k, c->tab[j].id, id)) report(c, c->tab[j].id, id);
    return;
  }
  uint32_t ny[MAXP], nz[MAXP];
  for (uint32_t e = 0; e <= M; e++)
    for (uint32_t f = 0; f <= M; f++) {
      for (int i = 0; i < P; i++) { ny[i] = y[i]; nz[i] = z[i]; step(c->k, t, e, f, &ny[i], &nz[i]); }
      dfs(c, t + 1, (id << (2 * n)) | ((uint64_t)e << n) | f, ny, nz, build, fill);
    }
}

static uint64_t kernel(const Keys *k, uint64_t *nontop) {
  Ctx c = {0};
  c.k = k;
  c.ntab = 1ull << (n * (2 * L - 1));
  c.tab = malloc(c.ntab * sizeof(Ent));
  c.dbits = 1;
  while ((1ull << c.dbits) < c.ntab) c.dbits++;
  c.dir = calloc((1ull << c.dbits) + 1, sizeof(uint32_t));
  uint32_t y[MAXP], z[MAXP];
  uint64_t fill = 0;
  // step 1 with e1 = 0: every f1, then steps 2..L
  for (uint32_t f1 = 0; f1 <= M; f1++) {
    for (int i = 0; i < P; i++) { y[i] = py[i]; z[i] = pz[i]; step(k, 1, 0, f1, &y[i], &z[i]); }
    dfs(&c, 2, f1, y, z, 1, &fill);
  }
  qsort(c.tab, c.ntab, sizeof(Ent), cmp_ent);
  for (uint64_t j = 0, b = 0; b <= (1ull << c.dbits); b++) {
    while (j < c.ntab && (c.tab[j].h >> (64 - c.dbits)) < b) j++;
    c.dir[b] = (uint32_t)j;
  }
  // pairs inside the table: e1 = 0 on both sides, f1 different
  for (uint64_t i = 0; i < c.ntab;) {
    uint64_t j = i;
    while (j < c.ntab && c.tab[j].h == c.tab[i].h) j++;
    for (uint64_t p = i; p < j; p++)
      for (uint64_t q = p + 1; q < j; q++) {
        uint64_t sh = (uint64_t)2 * n * (L - 1);
        if ((c.tab[p].id >> sh) != (c.tab[q].id >> sh) && verify(k, c.tab[p].id, c.tab[q].id))
          report(&c, c.tab[p].id, c.tab[q].id);
      }
    i = j;
  }
  for (uint32_t e1 = 1; e1 <= M; e1++)
    for (uint32_t f1 = 0; f1 <= M; f1++) {
      for (int i = 0; i < P; i++) { y[i] = py[i]; z[i] = pz[i]; step(k, 1, e1, f1, &y[i], &z[i]); }
      dfs(&c, 2, ((uint64_t)e1 << n) | f1, y, z, 0, &fill);
    }
  free(c.tab);
  free(c.dir);
  *nontop = c.nontop;
  return c.hits;
}

static void keys_of(Keys *k, const uint32_t *a, const int *rot) {
  for (int t = 1; t <= L; t++) {
    k->a[t] = a[idxpat[(t - 1) % nidx]] & M;
    k->r[t] = rot ? rot[t] : rotpat[(t - 1) % nrot] % n;
    k->x[t] = abspat[(t - 1) % nabs];
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
  if (argc < 3) {
    fprintf(stderr, "usage: d2_kernel n L [-z u|y] [-x xa..] [-i idx,..] [-r rot,..] [-a a,..|-s] [-R] [-v]\n");
    return 2;
  }
  n = atoi(argv[1]);
  L = atoi(argv[2]);
  if (n < 2 || n > 10 || L < 1 || L > MAXL || 2 * n * L > 62) {
    fprintf(stderr, "n in 2..10, L in 1..%d, 2 n L <= 62\n", MAXL);
    return 2;
  }
  M = (1u << n) - 1;
  idxpat[0] = 0; nidx = 1; rotpat[0] = 1; nrot = 1;
  int ai[MAXK] = {0}, na = 0;
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "-s")) sweep = 1;
    else if (!strcmp(argv[i], "-R")) rsweep = 1;
    else if (!strcmp(argv[i], "-v")) verbose = 1;
    else if (!strcmp(argv[i], "-i")) nidx = parse_list(argv[++i], idxpat);
    else if (!strcmp(argv[i], "-r")) nrot = parse_list(argv[++i], rotpat);
    else if (!strcmp(argv[i], "-a")) na = parse_list(argv[++i], ai);
    else if (!strcmp(argv[i], "-z")) rotin = argv[++i][0];
    else if (!strcmp(argv[i], "-x")) { strncpy(abspat, argv[++i], 15); nabs = (int)strlen(abspat); }
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
  }
  for (int i = 0; i < nidx; i++) if (idxpat[i] + 1 > nkeys) nkeys = idxpat[i] + 1;
  P = 64 / (2 * n) < MAXP ? 64 / (2 * n) : MAXP;
  uint64_t seed = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < P; i++) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    py[i] = (seed >> 33) & M;
    pz[i] = (seed >> 13) & M;
  }
  int used[MAXK] = {0}, pos[MAXK], np = 0;
  for (int t = 1; t <= L; t++) used[idxpat[(t - 1) % nidx]] = 1;
  for (int j = 0; j < nkeys; j++) if (used[j]) pos[np++] = j;
  if (!sweep && na < nkeys) { fprintf(stderr, "need %d multipliers (-a) or -s\n", nkeys); return 2; }
  uint64_t na_cfg = sweep ? 1ull << ((n - 1) * np) : 1, nr_cfg = 1;
  if (rsweep) for (int t = 1; t <= L; t++) nr_cfg *= (uint64_t)(n - 1);
  uint64_t configs = 0, hit_configs = 0, hits = 0, nontop = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : configs, hit_configs, hits, nontop)
  for (uint64_t c = 0; c < na_cfg * nr_cfg; c++) {
    uint32_t a[MAXK];
    int rot[MAXL + 1];
    for (int j = 0; j < nkeys; j++) a[j] = sweep ? 1 : (uint32_t)ai[j];
    uint64_t x = c % na_cfg, w = c / na_cfg;
    if (sweep)
      for (int q = 0; q < np; q++) { a[pos[q]] = (uint32_t)(((x & ((1u << (n - 1)) - 1)) << 1) | 1); x >>= n - 1; }
    for (int t = 1; t <= L; t++) { rot[t] = 1 + (int)(w % (uint64_t)(n - 1)); w /= (uint64_t)(n - 1); }
    Keys k;
    keys_of(&k, a, rsweep ? rot : NULL);
    uint64_t nt = 0, h = kernel(&k, &nt);
    configs++;
    nontop += nt;
    if (h) {
      hit_configs++;
      hits += h;
#pragma omp critical
      {
        printf("hit a=");
        for (int t = 1; t <= L; t++) printf("%s%u", t > 1 ? "," : "", k.a[t]);
        printf(" r=");
        for (int t = 1; t <= L; t++) printf("%s%d", t > 1 ? "," : "", k.r[t]);
        printf(": %llu\n", (unsigned long long)h);
        fflush(stdout);
      }
    }
  }
  printf("n %d L %d rot %c absorb %s configs %llu hit_configs %llu hits %llu nontop %llu\n", n, L, rotin, abspat,
         (unsigned long long)configs, (unsigned long long)hit_configs, (unsigned long long)hits,
         (unsigned long long)nontop);
  return 0;
}
