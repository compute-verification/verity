/* sa.c: simulated annealing over Steiner points in the rank metric over F2 (heuristic search for Conjecture A).
 *
 * In the graph of R x C matrices over F2 with rank-1 steps the distance is d(A, B) = rank(A - B), so a shortest
 * Steiner tree for the terminals {0, T_1..T_n} is a minimum spanning tree, under d, of the terminals together with
 * some Steiner points P (at most n - 1 of them). This searches P by annealing: the cost of P is the MST weight of
 * {0, T} + P, and a move perturbs one point by a random rank-1 matrix, replaces it by a random point on a geodesic
 * between two current points (a random sub-sum of a random rank decomposition of their difference), adds such a point,
 * or deletes one. Each thread runs its own restarts. Any cost under the naive length is a counterexample: it prints
 * the points, and steiner.py expands each MST edge into a geodesic of rank-1 steps and checks the whole tree.
 *
 * Input (stdin): R C n naive iters restarts seed, then n lines of R hex rows (row i: bit j is entry (i, j)).
 * Output: one JSON line: the best cost found, the cost histogram over restarts, and the best point set.
 * Build: gcc -O3 -march=native -fopenmp -o sa sa.c      Rows: R <= 16, C <= 64.
 */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXR 16
#define MAXV 72

typedef uint64_t u64;
typedef struct {
    u64 r[MAXR];
} mat;

static int R, C, NT; /* NT terminals including 0 */
static mat TERM[MAXV];

static inline u64 rng_next(u64 *s) {
    u64 x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

static int rank_diff(const mat *a, const mat *b) {
    u64 rows[MAXR];
    int n = 0;
    for (int i = 0; i < R; i++) {
        u64 x = a->r[i] ^ b->r[i];
        for (int j = 0; j < n; j++) {
            u64 lo = rows[j] & -rows[j];
            if (x & lo) x ^= rows[j];
        }
        if (x) rows[n++] = x;
    }
    return n;
}

/* a random point on a geodesic from a to b: a + a random sub-sum of a random rank decomposition of b - a */
static mat geodesic_point(const mat *a, const mat *b, u64 *s) {
    u64 D[MAXR], basis[MAXR], piv[MAXR];
    int r = 0;
    for (int i = 0; i < R; i++) D[i] = a->r[i] ^ b->r[i];
    /* echelon basis of the row space */
    for (int i = 0; i < R; i++) {
        u64 x = D[i];
        for (int j = 0; j < r; j++)
            if (x & piv[j]) x ^= basis[j];
        if (x) {
            piv[r] = x & -x;
            /* keep earlier rows reduced at this pivot */
            for (int j = 0; j < r; j++)
                if (basis[j] & piv[r]) basis[j] ^= x;
            basis[r++] = x;
        }
    }
    mat out = *a;
    if (!r) return out;
    /* random basis v = G * basis with G invertible over F2 */
    u64 v[MAXR];
    for (;;) {
        u64 g[MAXR], tmp[MAXR];
        int ok = 1, m = 0;
        for (int i = 0; i < r; i++) g[i] = rng_next(s) & ((1ULL << r) - 1);
        for (int i = 0; i < r && ok; i++) {
            u64 x = g[i];
            for (int j = 0; j < m; j++)
                if (x & (tmp[j] & -tmp[j])) x ^= tmp[j];
            if (!x) ok = 0;
            else tmp[m++] = x;
        }
        if (!ok) continue;
        for (int i = 0; i < r; i++) {
            v[i] = 0;
            for (int j = 0; j < r; j++)
                if ((g[i] >> j) & 1) v[i] ^= basis[j];
        }
        break;
    }
    /* coordinates of each row of D in v: reduce against an echelon copy of v tracking combinations */
    u64 ev[MAXR], ec[MAXR], ep[MAXR];
    int ne = 0;
    for (int i = 0; i < r; i++) {
        u64 x = v[i], c = 1ULL << i;
        for (int j = 0; j < ne; j++)
            if (x & ep[j]) {
                x ^= ev[j];
                c ^= ec[j];
            }
        ep[ne] = x & -x;
        for (int j = 0; j < ne; j++)
            if (ev[j] & ep[ne]) {
                ev[j] ^= x;
                ec[j] ^= c;
            }
        ev[ne] = x;
        ec[ne] = c;
        ne++;
    }
    u64 sub = rng_next(s) & ((1ULL << r) - 1);
    for (int i = 0; i < R; i++) {
        u64 x = D[i], c = 0;
        for (int j = 0; j < ne; j++)
            if (x & ep[j]) {
                x ^= ev[j];
                c ^= ec[j];
            }
        u64 e = 0, cs = c & sub;
        for (int j = 0; j < r; j++)
            if ((cs >> j) & 1) e ^= v[j];
        out.r[i] ^= e;
    }
    return out;
}

static mat rank1(u64 *s) {
    mat m;
    u64 u, w;
    do u = rng_next(s) & ((1ULL << R) - 1);
    while (!u);
    do w = rng_next(s) & (C == 64 ? ~0ULL : ((1ULL << C) - 1));
    while (!w);
    for (int i = 0; i < R; i++) m.r[i] = ((u >> i) & 1) ? w : 0;
    return m;
}

static int mst(int V, uint8_t D[MAXV][MAXV]) {
    int in[MAXV] = {0}, best[MAXV], tot = 0;
    in[0] = 1;
    for (int i = 1; i < V; i++) best[i] = D[0][i];
    for (int k = 1; k < V; k++) {
        int p = -1;
        for (int i = 1; i < V; i++)
            if (!in[i] && (p < 0 || best[i] < best[p])) p = i;
        in[p] = 1;
        tot += best[p];
        for (int i = 1; i < V; i++)
            if (!in[i] && D[p][i] < best[i]) best[i] = D[p][i];
    }
    return tot;
}

typedef struct {
    int np;
    mat p[MAXV];
} config;

static int cost_of(const config *c, uint8_t D[MAXV][MAXV]) {
    int V = NT + c->np;
    for (int i = 0; i < c->np; i++) {
        const mat *x = &c->p[i];
        for (int j = 0; j < NT; j++) D[NT + i][j] = D[j][NT + i] = rank_diff(x, &TERM[j]);
        for (int j = 0; j < i; j++) D[NT + i][NT + j] = D[NT + j][NT + i] = rank_diff(x, &c->p[j]);
        D[NT + i][NT + i] = 0;
    }
    return mst(V, D);
}

int main(int argc, char **argv) {
    int n, naive;
    long iters, restarts;
    unsigned long long seed;
    if (scanf("%d %d %d %d %ld %ld %llu", &R, &C, &n, &naive, &iters, &restarts, &seed) != 7) return 2;
    NT = n + 1;
    memset(&TERM[0], 0, sizeof(mat));
    for (int t = 1; t <= n; t++)
        for (int i = 0; i < R; i++) {
            unsigned long long x;
            if (scanf("%llx", &x) != 1) return 2;
            TERM[t].r[i] = x;
        }
    int maxp = n - 1 < MAXV - NT ? n - 1 : MAXV - NT;
    static uint8_t D0[MAXV][MAXV];
    for (int i = 0; i < NT; i++)
        for (int j = 0; j < NT; j++) D0[i][j] = rank_diff(&TERM[i], &TERM[j]);
    long hist[256] = {0};
    int gbest = 1 << 30;
    config gconf;
    double t0 = omp_get_wtime();
#pragma omp parallel
    {
        u64 s = seed * 0x9E3779B97F4A7C15ULL + 7919ULL * (omp_get_thread_num() + 1);
        for (int k = 0; k < 20; k++) rng_next(&s);
        uint8_t D[MAXV][MAXV];
        memcpy(D, D0, sizeof D);
#pragma omp for schedule(dynamic, 1)
        for (long rs = 0; rs < restarts; rs++) {
            config cur = {0}, best;
            int np0 = rng_next(&s) % (maxp + 1);
            for (int i = 0; i < np0; i++) {
                int a = rng_next(&s) % NT, b = rng_next(&s) % NT;
                cur.p[cur.np++] = geodesic_point(&TERM[a], &TERM[b], &s);
            }
            int cc = cost_of(&cur, D), bc = cc;
            best = cur;
            double T0 = 1.2, T1 = 0.15;
            for (long it = 0; it < iters; it++) {
                double T = T0 * pow(T1 / T0, (double)it / iters);
                config nx = cur;
                int mv = rng_next(&s) % 8;
                int V = NT + nx.np;
                if (mv <= 2 && nx.np) {
                    int i = rng_next(&s) % nx.np;
                    mat g = rank1(&s);
                    for (int q = 0; q < R; q++) nx.p[i].r[q] ^= g.r[q];
                } else if (mv <= 5 && nx.np) {
                    int i = rng_next(&s) % nx.np, a = rng_next(&s) % V, b = rng_next(&s) % V;
                    const mat *A = a < NT ? &TERM[a] : &cur.p[a - NT], *B = b < NT ? &TERM[b] : &cur.p[b - NT];
                    nx.p[i] = geodesic_point(A, B, &s);
                } else if (mv == 6 && nx.np < maxp) {
                    int a = rng_next(&s) % V, b = rng_next(&s) % V;
                    const mat *A = a < NT ? &TERM[a] : &cur.p[a - NT], *B = b < NT ? &TERM[b] : &cur.p[b - NT];
                    nx.p[nx.np++] = geodesic_point(A, B, &s);
                } else if (nx.np) {
                    int i = rng_next(&s) % nx.np;
                    nx.p[i] = nx.p[--nx.np];
                } else {
                    int a = rng_next(&s) % NT, b = rng_next(&s) % NT;
                    nx.p[nx.np++] = geodesic_point(&TERM[a], &TERM[b], &s);
                }
                int nc = cost_of(&nx, D);
                if (nc <= cc || (double)(rng_next(&s) % 1000000) / 1e6 < exp((cc - nc) / T)) {
                    cur = nx;
                    cc = nc;
                    if (cc < bc) {
                        bc = cc;
                        best = cur;
                    }
                }
            }
#pragma omp critical
            {
                hist[bc < 255 ? bc : 255]++;
                if (bc < gbest) {
                    gbest = bc;
                    gconf = best;
                }
            }
        }
    }
    printf("{\"R\":%d,\"C\":%d,\"n\":%d,\"naive\":%d,\"iters\":%ld,\"restarts\":%ld,\"threads\":%d,\"wall_s\":%.2f,"
           "\"best\":%d,\"hist\":{",
           R, C, n, naive, iters, restarts, omp_get_max_threads(), omp_get_wtime() - t0, gbest);
    int first = 1;
    for (int i = 0; i < 256; i++)
        if (hist[i]) {
            printf("%s\"%d\":%ld", first ? "" : ",", i, hist[i]);
            first = 0;
        }
    printf("},\"points\":[");
    for (int i = 0; i < gconf.np; i++) {
        printf("%s[", i ? "," : "");
        for (int q = 0; q < R; q++) printf("%s\"%llx\"", q ? "," : "", (unsigned long long)gconf.p[i].r[q]);
        printf("]");
    }
    printf("]}\n");
    return 0;
}
