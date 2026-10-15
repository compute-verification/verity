/* dw_f3.c: exact Steiner trees in the rank-metric graph of R x C matrices over F3 (Conjecture A searches).
 *
 * The F3 twin of dw_f2.c on the whole space: states are all 3^(R*C) matrices, adjacent when their difference has
 * rank 1, and Dreyfus-Wagner gives dp[A][v] for every subset A of the terminals and every state v, rooted at 0.
 * A matrix is stored by its dense base-3 index sum_p t_p 3^p (entry (i, j) is trit p = i*C + j). The index splits into
 * a low and a high half; adding a fixed matrix m is two table lookups, one per half, so a min over each coset
 * {M, M+m, M+2m} costs a few loads. A relaxation step is a min over the cosets of u (x) F3^C for each projective u
 * (or of F3^R (x) w for each projective w, whichever side is cheaper), each by three-way mins along its C basis
 * directions. Values are capped at INF = 120.
 *
 * Spec:
 *   P <name> <R> <C> <n>
 *   T <trits> <label>   n lines; <trits> is R*C digits 0-2, entry (i,j) at position i*C+j
 * Output: one JSON line per problem, as dw_f2's, with matrices written as trit strings.
 *
 * Build: gcc -O3 -march=native -fopenmp -o dw_f3 dw_f3.c
 * Run:   OMP_NUM_THREADS=64 ./dw_f3 SPEC
 */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF 120
#define MAXN 12
#define MAXT 20
#define BLK 4096

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

static int R, C, NT, NLO, NHI;
static u64 LO, HI, N, P3[MAXT + 1];

static u64 enc(const u8 *t) {
    u64 x = 0;
    for (int p = NT - 1; p >= 0; p--) x = x * 3 + t[p];
    return x;
}

static void dec(u64 x, u8 *t) {
    for (int p = 0; p < NT; p++) {
        t[p] = x % 3;
        x /= 3;
    }
}

typedef struct {
    u32 *lo, *hi;
} dir_t;

static dir_t make_dir(const u8 *m) {
    dir_t d;
    d.lo = malloc(sizeof(u32) * LO);
    d.hi = malloc(sizeof(u32) * HI);
    for (u64 x = 0; x < LO; x++) {
        u64 y = 0, r = x;
        for (int p = 0; p < NLO; p++) {
            y += ((r % 3 + m[p]) % 3) * P3[p];
            r /= 3;
        }
        d.lo[x] = y;
    }
    for (u64 x = 0; x < HI; x++) {
        u64 y = 0, r = x;
        for (int p = 0; p < NHI; p++) {
            y += ((r % 3 + m[NLO + p]) % 3) * P3[p];
            r /= 3;
        }
        d.hi[x] = y;
    }
    return d;
}

static inline u8 mn8(u8 a, u8 b) { return a < b ? a : b; }

static void tripmin(u8 *tmp, const dir_t *d) {
#pragma omp parallel for schedule(static)
    for (u64 h = 0; h < HI; h++) {
        u64 h1 = d->hi[h], h2 = d->hi[h1];
        for (u64 l = 0; l < LO; l++) {
            u64 M = h * LO + l, l1 = d->lo[l], M1 = h1 * LO + l1, M2 = h2 * LO + d->lo[l1];
            if (M < M1 && M < M2) {
                u8 c = mn8(tmp[M], mn8(tmp[M1], tmp[M2]));
                tmp[M] = c;
                tmp[M1] = c;
                tmp[M2] = c;
            }
        }
    }
}

typedef struct {
    int ng, d;
    dir_t *dirs; /* ng groups of d directions */
} gens_t;

static void pcopy(u8 *dst, const u8 *src) {
#pragma omp parallel for schedule(static)
    for (u64 B = 0; B < N; B += 1 << 16) memcpy(dst + B, src + B, N - B < (1 << 16) ? N - B : (1 << 16));
}

static int relax(u8 *f, u8 *acc, u8 *tmp, const gens_t *G) {
    int iters = 0;
    for (;;) {
        iters++;
        pcopy(acc, f);
        for (int g = 0; g < G->ng; g++) {
            pcopy(tmp, f);
            for (int j = 0; j < G->d; j++) tripmin(tmp, &G->dirs[g * G->d + j]);
#pragma omp parallel for schedule(static)
            for (u64 M = 0; M < N; M++) acc[M] = mn8(acc[M], (u8)(tmp[M] + 1));
        }
        int changed = 0;
#pragma omp parallel for schedule(static) reduction(| : changed)
        for (u64 M = 0; M < N; M++)
            if (acc[M] != f[M]) {
                changed = 1;
                f[M] = acc[M];
            }
        if (!changed) return iters;
    }
}

static void merge(u8 **dp, int S, u8 *out) {
    int low = S & -S, ns = 0;
    int *sp = malloc(sizeof(int) << __builtin_popcount(S));
    for (int s1 = (S - 1) & S; s1; s1 = (s1 - 1) & S)
        if (s1 & low) sp[ns++] = s1;
#pragma omp parallel for schedule(static)
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

/* projective points of F3^n: first nonzero trit is 1 */
static int projective(int n, u8 (*out)[MAXT]) {
    int cnt = 0;
    u64 tot = 1;
    for (int i = 0; i < n; i++) tot *= 3;
    for (u64 x = 1; x < tot; x++) {
        u8 t[MAXT];
        u64 r = x;
        for (int i = 0; i < n; i++) {
            t[i] = r % 3;
            r /= 3;
        }
        int f = 0;
        while (!t[f]) f++;
        if (t[f] != 1) continue;
        memcpy(out[cnt++], t, MAXT);
    }
    return cnt;
}

typedef struct {
    u64 a[256], b[256];
    int ne, err;
} tree_t;

static u8 (*R1)[MAXT];
static int NR1;

static u64 addmat(u64 v, const u8 *g) {
    u8 t[MAXT];
    dec(v, t);
    for (int p = 0; p < NT; p++) t[p] = (t[p] + g[p]) % 3;
    return enc(t);
}

static void extract(u8 **dp, int S, u64 v, tree_t *T) {
    while (!T->err) {
        u8 val = dp[S][v];
        if (__builtin_popcount(S) == 1 && val == 0) return;
        if (__builtin_popcount(S) >= 2) {
            int low = S & -S, done = 0;
            for (int s1 = (S - 1) & S; s1 && !done; s1 = (s1 - 1) & S)
                if ((s1 & low) && dp[s1][v] + dp[S ^ s1][v] == val) {
                    extract(dp, s1, v, T);
                    extract(dp, S ^ s1, v, T);
                    done = 1;
                }
            if (done) return;
        }
        int found = 0;
        for (int i = 0; i < NR1 && !found; i++) {
            u64 w = addmat(v, R1[i]);
            if (dp[S][w] + 1 == val) {
                if (T->ne >= 256) {
                    T->err = 2;
                    return;
                }
                T->a[T->ne] = v;
                T->b[T->ne] = w;
                T->ne++;
                v = w;
                found = 1;
            }
        }
        if (!found) T->err = 1;
    }
}

static char *tstr(u64 x, char *buf) {
    u8 t[MAXT];
    dec(x, t);
    for (int p = 0; p < NT; p++) buf[p] = '0' + t[p];
    buf[NT] = 0;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s SPEC\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) return 2;
    char tag[8], name[256], lab[MAXN][64], ts[64];
    int n;
    while (fscanf(f, "%7s %255s %d %d %d", tag, name, &R, &C, &n) == 5) {
        double t0 = omp_get_wtime();
        NT = R * C;
        if (NT > MAXT || n > MAXN) return 2;
        NLO = NT / 2;
        NHI = NT - NLO;
        P3[0] = 1;
        for (int i = 1; i <= MAXT; i++) P3[i] = P3[i - 1] * 3;
        LO = P3[NLO];
        HI = P3[NHI];
        N = P3[NT];
        u64 term[MAXN];
        for (int i = 0; i < n; i++) {
            if (fscanf(f, "%7s %63s %63s", tag, ts, lab[i]) != 3 || (int)strlen(ts) != NT) return 2;
            u8 t[MAXT];
            for (int p = 0; p < NT; p++) t[p] = ts[p] - '0';
            term[i] = enc(t);
        }
        /* groups: projective u with the C directions u (x) e_j, or projective w with the R directions e_i (x) w */
        static u8 pu[4000][MAXT], pw[4000][MAXT];
        int nu = projective(R, pu), nw = projective(C, pw);
        int side = (long)nu * (C + 2) <= (long)nw * (R + 2) ? 0 : 1;
        gens_t G;
        G.ng = side ? nw : nu;
        G.d = side ? R : C;
        G.dirs = malloc(sizeof(dir_t) * G.ng * G.d);
        for (int g = 0; g < G.ng; g++)
            for (int j = 0; j < G.d; j++) {
                u8 m[MAXT] = {0};
                for (int i = 0; i < R; i++)
                    for (int c = 0; c < C; c++) {
                        int coef = side ? (i == j) * pw[g][c] : pu[g][i] * (c == j);
                        m[i * C + c] = coef % 3;
                    }
                G.dirs[g * G.d + j] = make_dir(m);
            }
        /* rank-1 matrices: projective u times nonzero w */
        NR1 = 0;
        u64 nwall = P3[C] - 1;
        R1 = malloc(sizeof(*R1) * nu * nwall);
        for (int a = 0; a < nu; a++)
            for (u64 w = 1; w <= nwall; w++) {
                u8 wt[MAXT];
                u64 r = w;
                for (int c = 0; c < C; c++) {
                    wt[c] = r % 3;
                    r /= 3;
                }
                for (int i = 0; i < R; i++)
                    for (int c = 0; c < C; c++) R1[NR1][i * C + c] = (pu[a][i] * wt[c]) % 3;
                NR1++;
            }
        int full = (1 << n) - 1, maxit = 0;
        u8 **dp = calloc(1 << n, sizeof(u8 *));
        u8 *acc = aligned_alloc(64, (N + 63) & ~63ULL), *tmp = aligned_alloc(64, (N + 63) & ~63ULL);
        for (int S = 1; S <= full; S++) {
            dp[S] = aligned_alloc(64, (N + 63) & ~63ULL);
            if (!dp[S] || !acc || !tmp) {
                fprintf(stderr, "out of memory\n");
                return 2;
            }
            if (__builtin_popcount(S) == 1) {
                memset(dp[S], INF, N);
                dp[S][term[__builtin_ctz(S)]] = 0;
            } else {
                merge(dp, S, dp[S]);
            }
            int it = relax(dp[S], acc, tmp, &G);
            if (it > maxit) maxit = it;
            fprintf(stderr, "%s: mask %d/%d done at %.1fs (%d passes)\n", name, S, full, omp_get_wtime() - t0, it);
        }
        double t1 = omp_get_wtime();
        tree_t *T = calloc(1, sizeof(tree_t));
        extract(dp, full, 0, T);
        printf("{\"name\":\"%s\",\"field\":3,\"R\":%d,\"C\":%d,\"k\":%d,\"n\":%d,\"states\":%llu,\"rank1_in_S\":%d,"
               "\"groups\":%d,\"group_side\":\"%s\",\"group_dims\":%d,\"max_relax_iters\":%d,\"threads\":%d,"
               "\"dp_s\":%.3f,\"wall_s\":%.3f,\"answer\":%d,\"root_dp\":[",
               name, R, C, NT, n, (unsigned long long)N, NR1, G.ng, side ? "w" : "u", G.ng * G.d, maxit,
               omp_get_max_threads(), t1 - t0, omp_get_wtime() - t0, dp[full][0]);
        for (int S = 0; S <= full; S++) printf("%s%d", S ? "," : "", S ? dp[S][0] : 0);
        char b1[64], b2[64];
        printf("],\"terminals\":[");
        for (int i = 0; i < n; i++) printf("%s[\"%s\",\"%s\"]", i ? "," : "", tstr(term[i], b1), lab[i]);
        printf("],\"tree_err\":%d,\"tree\":[", T->err);
        for (int i = 0; i < T->ne; i++) printf("%s[\"%s\",\"%s\"]", i ? "," : "", tstr(T->a[i], b1), tstr(T->b[i], b2));
        printf("]}\n");
        fflush(stdout);
        for (int S = 1; S <= full; S++) free(dp[S]);
        free(dp);
        free(acc);
        free(tmp);
        for (int i = 0; i < G.ng * G.d; i++) {
            free(G.dirs[i].lo);
            free(G.dirs[i].hi);
        }
        free(G.dirs);
        free(R1);
        free(T);
    }
    return 0;
}
