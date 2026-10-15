/* dw_f2.c: exact Steiner trees in the rank-metric graph of R x C matrices over F2 (Conjecture A searches).
 *
 * A state is an R x C matrix over F2 (a bilinear form: rows are X-variables, columns Y-variables) lying in a subspace S,
 * given by a basis (S is the whole space when the spec's k is 0). Two states are adjacent when their difference has
 * rank 1 and lies in S. For the root 0 and terminals T_1..T_n in S, Dreyfus-Wagner computes dp[A][v], the length of a
 * shortest tree inside S spanning v and the terminals in A, for every subset A and every state v; dp[A][0] is the
 * Steiner length of {0} + A. On the whole space that is the exact minimum over all trees. On a proper subspace it is
 * the minimum over trees whose nodes all lie in S: an upper bound on the whole space's minimum, and a tree it finds is
 * a valid tree.
 *
 * One relaxation step (v -> v + g, g rank 1 in S) uses that the rank-1 matrices of S, with 0, are the union over
 * nonzero u in F2^R of the subspaces S n (u (x) F2^C) (or, transposed, over w in F2^C of S n (F2^R (x) w)). A min over
 * the cosets of each such subspace is a pairwise min along each of its basis vectors, so a step costs about the sum of
 * those subspaces' dimensions in passes over the states, not one pass per rank-1 matrix. Steps repeat until nothing
 * changes. Values are capped at INF = 120 (a value of 120 means ">= 120").
 *
 * Spec (text; a file may hold many problems, each solved independently):
 *   P <name> <R> <C> <k> <n>
 *   B <hex>            k lines: the basis of S as matrices, bit i*C+j is entry (i,j); k = 0 means the whole space
 *   T <hex> <label>    n lines: the terminals (nonzero, distinct, in S); the root 0 is implicit
 * Output: one JSON line per problem on stdout: dp[A][0] for every subset A (root_dp, indexed by the bit mask over the
 * terminals in spec order), the answer dp[all][0], and one optimal tree for the whole set as edges between matrices.
 *
 * Build: gcc -O3 -march=native -fopenmp -o dw_f2 dw_f2.c
 * Run:   OMP_NUM_THREADS=64 ./dw_f2 SPEC          (parallel inside each problem)
 *        OMP_NUM_THREADS=64 ./dw_f2 SPEC --batch  (problems in parallel, each serial: many small subspaces)
 */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF 120
#define MAXN 16
#define BLK 4096

typedef uint64_t u64;
typedef uint8_t u8;

typedef struct {
    int d;
    u64 v[64];
} span_t;

typedef struct {
    char name[256];
    int R, C, k, n, full;
    u64 basis[64];
    u64 ech[64], echc[64];
    int pivbit[64], nech;
    u64 term[MAXN], tc[MAXN];
    char label[MAXN][64];
} problem_t;

typedef struct {
    int ng, side;
    span_t *g;
    int nr1;
    u64 *r1;
    long cost;
} gens_t;

static int hibit(u64 x) { return 63 - __builtin_clzll(x); }

static void span_insert(span_t *s, u64 x) {
    for (int i = 0; i < s->d; i++) {
        int h = hibit(s->v[i]);
        if ((x >> h) & 1) x ^= s->v[i];
    }
    if (!x) return;
    /* each row was reduced by the earlier ones, so reducing in insertion order clears every pivot */
    s->v[s->d++] = x;
}

static void pcopy(u8 *dst, const u8 *src, u64 N, int par) {
    if (!par || N < (1 << 20)) {
        memcpy(dst, src, N);
        return;
    }
#pragma omp parallel for schedule(static)
    for (u64 B = 0; B < N; B += 1 << 16) memcpy(dst + B, src + B, N - B < (1 << 16) ? N - B : (1 << 16));
}

/* membership in S; on success *coord holds x's coordinates in the basis */
static int to_coord(const problem_t *p, u64 x, u64 *coord) {
    if (p->full) {
        *coord = x;
        return 1;
    }
    u64 c = 0;
    for (int r = 0; r < p->nech; r++)
        if ((x >> p->pivbit[r]) & 1) {
            x ^= p->ech[r];
            c ^= p->echc[r];
        }
    *coord = c;
    return x == 0;
}

static u64 to_mat(const problem_t *p, u64 coord) {
    if (p->full) return coord;
    u64 m = 0;
    for (int i = 0; i < p->k; i++)
        if ((coord >> i) & 1) m ^= p->basis[i];
    return m;
}

static u64 outer(int R, int C, u64 u, u64 w) {
    u64 m = 0;
    for (int i = 0; i < R; i++)
        if ((u >> i) & 1) m |= w << (i * C);
    return m;
}

static long gcost(int d) { return d == 0 ? 0 : d == 1 ? 1 : d + 2; }

static void build_gens(const problem_t *p, gens_t *G) {
    int R = p->R, C = p->C;
    long nu = (1L << R) - 1, nw = (1L << C) - 1;
    span_t *gu = calloc(nu, sizeof(span_t)), *gw = calloc(nw, sizeof(span_t));
    G->r1 = malloc(sizeof(u64) * nu * nw);
    G->nr1 = 0;
    for (long u = 1; u <= nu; u++)
        for (long w = 1; w <= nw; w++) {
            u64 c;
            if (to_coord(p, outer(R, C, u, w), &c)) {
                G->r1[G->nr1++] = c;
                span_insert(&gu[u - 1], c);
                span_insert(&gw[w - 1], c);
            }
        }
    long cu = 0, cw = 0;
    for (long u = 0; u < nu; u++) cu += gcost(gu[u].d);
    for (long w = 0; w < nw; w++) cw += gcost(gw[w].d);
    span_t *src = cu <= cw ? gu : gw;
    long ns = cu <= cw ? nu : nw;
    G->side = cu <= cw ? 0 : 1;
    G->cost = cu <= cw ? cu : cw;
    G->g = malloc(sizeof(span_t) * (ns ? ns : 1));
    G->ng = 0;
    for (long i = 0; i < ns; i++)
        if (src[i].d) G->g[G->ng++] = src[i];
    free(gu);
    free(gw);
}

static inline u8 mn8(u8 a, u8 b) { return a < b ? a : b; }

/* tmp[M] = tmp[M ^ m] = min of the two, over every pair */
static void pairmin(u8 *tmp, u64 N, u64 m, int par) {
    int h = hibit(m);
    u64 half = N >> 1, low = (1ULL << h) - 1;
    if (h >= 6 && (m & 63) == 0) {
#pragma omp parallel for schedule(static) if (par)
        for (u64 b = 0; b < half; b += 64) {
            u64 M = ((b >> h) << (h + 1)) | (b & low);
            u8 *x = tmp + M, *y = tmp + (M ^ m);
#pragma GCC ivdep
            for (int t = 0; t < 64; t++) {
                u8 c = mn8(x[t], y[t]);
                x[t] = c;
                y[t] = c;
            }
        }
    } else {
#pragma omp parallel for schedule(static) if (par)
        for (u64 b = 0; b < half; b++) {
            u64 M = ((b >> h) << (h + 1)) | (b & low), P = M ^ m;
            u8 c = mn8(tmp[M], tmp[P]);
            tmp[M] = c;
            tmp[P] = c;
        }
    }
}

/* acc[M] = min(acc[M], src[M ^ m] + 1) */
static void shiftmin(u8 *acc, const u8 *src, u64 N, u64 m, int par) {
    if (N >= 64 && (m & 63) == 0) {
#pragma omp parallel for schedule(static) if (par)
        for (u64 b = 0; b < N; b += 64) {
            u8 *x = acc + b;
            const u8 *y = src + (b ^ m);
            for (int t = 0; t < 64; t++) x[t] = mn8(x[t], (u8)(y[t] + 1));
        }
    } else {
#pragma omp parallel for schedule(static) if (par)
        for (u64 M = 0; M < N; M++) acc[M] = mn8(acc[M], (u8)(src[M ^ m] + 1));
    }
}

/* f <- min over u of f[u] + dist_S(u, .); returns the number of passes taken */
static int relax(u8 *f, u8 *acc, u8 *tmp, u64 N, const gens_t *G, int par) {
    int iters = 0;
    for (;;) {
        iters++;
        pcopy(acc, f, N, par);
        for (int gi = 0; gi < G->ng; gi++) {
            const span_t *g = &G->g[gi];
            if (g->d == 1) {
                shiftmin(acc, f, N, g->v[0], par);
                continue;
            }
            pcopy(tmp, f, N, par);
            for (int j = 0; j < g->d; j++) pairmin(tmp, N, g->v[j], par);
#pragma omp parallel for schedule(static) if (par)
            for (u64 M = 0; M < N; M++) acc[M] = mn8(acc[M], (u8)(tmp[M] + 1));
        }
        int changed = 0;
#pragma omp parallel for schedule(static) reduction(| : changed) if (par)
        for (u64 M = 0; M < N; M++)
            if (acc[M] != f[M]) {
                changed = 1;
                f[M] = acc[M];
            }
        if (!changed) return iters;
    }
}

static void merge(u8 **dp, int S, u8 *out, u64 N, int par) {
    int low = S & -S, ns = 0;
    int *sp = malloc(sizeof(int) << __builtin_popcount(S));
    for (int s1 = (S - 1) & S; s1; s1 = (s1 - 1) & S)
        if (s1 & low) sp[ns++] = s1;
#pragma omp parallel for schedule(static) if (par)
    for (u64 B = 0; B < N; B += BLK) {
        u64 E = B + BLK < N ? B + BLK : N;
        for (u64 M = B; M < E; M++) out[M] = INF;
        for (int i = 0; i < ns; i++) {
            const u8 *a = dp[sp[i]], *b = dp[S ^ sp[i]];
            for (u64 M = B; M < E; M++) out[M] = mn8(out[M], (u8)(a[M] + b[M]));
        }
        for (u64 M = B; M < E; M++) out[M] = mn8(out[M], INF);
    }
    free(sp);
}

typedef struct {
    u64 a[512], b[512];
    int ne, err;
} tree_t;

static void extract(u8 **dp, int S, u64 v, const gens_t *G, tree_t *T) {
    while (!T->err) {
        u8 val = dp[S][v];
        if (__builtin_popcount(S) == 1 && val == 0) return;
        if (__builtin_popcount(S) >= 2) {
            int low = S & -S, done = 0;
            for (int s1 = (S - 1) & S; s1 && !done; s1 = (s1 - 1) & S)
                if ((s1 & low) && dp[s1][v] + dp[S ^ s1][v] == val) {
                    extract(dp, s1, v, G, T);
                    extract(dp, S ^ s1, v, G, T);
                    done = 1;
                }
            if (done) return;
        }
        int found = 0;
        for (int i = 0; i < G->nr1; i++) {
            u64 w = v ^ G->r1[i];
            if (dp[S][w] + 1 == val) {
                if (T->ne >= 512) {
                    T->err = 2;
                    return;
                }
                T->a[T->ne] = v;
                T->b[T->ne] = w;
                T->ne++;
                v = w;
                found = 1;
                break;
            }
        }
        if (!found) {
            T->err = 1;
            return;
        }
    }
}

static void *xalloc(u64 n) {
    u64 sz = (n + 63) & ~63ULL;
    void *p = aligned_alloc(64, sz < 64 ? 64 : sz);
    if (!p) {
        fprintf(stderr, "out of memory (%llu bytes)\n", (unsigned long long)sz);
        exit(2);
    }
    return p;
}

static void solve(problem_t *p, int par, FILE *out) {
    double t0 = omp_get_wtime();
    gens_t G;
    build_gens(p, &G);
    u64 N = 1ULL << p->k;
    int n = p->n, full = (1 << n) - 1, maxit = 0;
    u8 **dp = calloc(1 << n, sizeof(u8 *));
    u8 *acc = xalloc(N), *tmp = xalloc(N);
    for (int S = 1; S <= full; S++) {
        dp[S] = xalloc(N);
        if (__builtin_popcount(S) == 1) {
            memset(dp[S], INF, N);
            dp[S][p->tc[__builtin_ctz(S)]] = 0;
        } else {
            merge(dp, S, dp[S], N, par);
        }
        int it = relax(dp[S], acc, tmp, N, &G, par);
        if (it > maxit) maxit = it;
    }
    double t1 = omp_get_wtime();
    tree_t *T = calloc(1, sizeof(tree_t));
    extract(dp, full, 0, &G, T);
    long dims = 0;
    for (int i = 0; i < G.ng; i++) dims += G.g[i].d;

    char *buf = malloc(64 * 1024 + 8 * (1 << n) + 64 * 1024);
    int L = 0;
    L += sprintf(buf + L,
                 "{\"name\":\"%s\",\"field\":2,\"R\":%d,\"C\":%d,\"k\":%d,\"n\":%d,\"states\":%llu,\"rank1_in_S\":%d,"
                 "\"groups\":%d,\"group_side\":\"%s\",\"group_dims\":%ld,\"passes_per_step\":%ld,\"max_relax_iters\":%d,"
                 "\"threads\":%d,\"batch\":%d,\"dp_s\":%.3f,\"wall_s\":%.3f,\"answer\":%d,\"root_dp\":[",
                 p->name, p->R, p->C, p->k, n, (unsigned long long)N, G.nr1, G.ng, G.side ? "w" : "u", dims, G.cost,
                 maxit, par ? omp_get_max_threads() : 1, !par, t1 - t0, omp_get_wtime() - t0, dp[full][0]);
    for (int S = 0; S <= full; S++) L += sprintf(buf + L, "%s%d", S ? "," : "", S ? dp[S][0] : 0);
    L += sprintf(buf + L, "],\"terminals\":[");
    for (int i = 0; i < n; i++)
        L += sprintf(buf + L, "%s[\"%llx\",\"%s\"]", i ? "," : "", (unsigned long long)p->term[i], p->label[i]);
    L += sprintf(buf + L, "],\"tree_err\":%d,\"tree\":[", T->err);
    for (int i = 0; i < T->ne; i++)
        L += sprintf(buf + L, "%s[\"%llx\",\"%llx\"]", i ? "," : "", (unsigned long long)to_mat(p, T->a[i]),
                     (unsigned long long)to_mat(p, T->b[i]));
    L += sprintf(buf + L, "]}\n");
#pragma omp critical(out)
    {
        fputs(buf, out);
        fflush(out);
    }
    free(buf);
    free(T);
    for (int S = 1; S <= full; S++) free(dp[S]);
    free(dp);
    free(acc);
    free(tmp);
    free(G.g);
    free(G.r1);
}

static int read_problem(FILE *f, problem_t *p) {
    char tag[8];
    memset(p, 0, sizeof *p);
    if (fscanf(f, "%7s", tag) != 1) return 0;
    if (strcmp(tag, "P")) {
        fprintf(stderr, "spec: expected P, got %s\n", tag);
        exit(2);
    }
    if (fscanf(f, "%255s %d %d %d %d", p->name, &p->R, &p->C, &p->k, &p->n) != 5) exit(2);
    if (p->R * p->C > 64 || p->n > MAXN || p->n < 1) {
        fprintf(stderr, "spec: R*C must be <= 64 and 1 <= n <= %d\n", MAXN);
        exit(2);
    }
    if (p->k == 0) {
        p->full = 1;
        p->k = p->R * p->C;
    } else {
        for (int i = 0; i < p->k; i++) {
            unsigned long long x;
            if (fscanf(f, "%7s %llx", tag, &x) != 2 || strcmp(tag, "B")) exit(2);
            p->basis[i] = x;
            u64 c = 1ULL << i;
            for (int r = 0; r < p->nech; r++)
                if ((x >> p->pivbit[r]) & 1) {
                    x ^= p->ech[r];
                    c ^= p->echc[r];
                }
            if (!x) {
                fprintf(stderr, "%s: basis is dependent at row %d\n", p->name, i);
                exit(2);
            }
            p->pivbit[p->nech] = hibit(x);
            p->ech[p->nech] = x;
            p->echc[p->nech] = c;
            p->nech++;
        }
    }
    if (p->k > 40) {
        fprintf(stderr, "%s: k = %d is too large\n", p->name, p->k);
        exit(2);
    }
    for (int i = 0; i < p->n; i++) {
        unsigned long long x;
        if (fscanf(f, "%7s %llx %63s", tag, &x, p->label[i]) != 3 || strcmp(tag, "T")) exit(2);
        p->term[i] = x;
        if (!to_coord(p, x, &p->tc[i]) || x == 0) {
            fprintf(stderr, "%s: terminal %d is zero or not in S\n", p->name, i);
            exit(2);
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s SPEC [--batch]\n", argv[0]);
        return 2;
    }
    int batch = argc > 2 && !strcmp(argv[2], "--batch");
    FILE *f = fopen(argv[1], "r");
    if (!f) return 2;
    int cap = 16, np = 0;
    problem_t *ps = malloc(sizeof(problem_t) * cap);
    while (1) {
        if (np == cap) ps = realloc(ps, sizeof(problem_t) * (cap *= 2));
        if (!read_problem(f, &ps[np])) break;
        np++;
    }
    fclose(f);
    if (batch) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < np; i++) solve(&ps[i], 0, stdout);
    } else {
        for (int i = 0; i < np; i++) solve(&ps[i], 1, stdout);
    }
    free(ps);
    return 0;
}
