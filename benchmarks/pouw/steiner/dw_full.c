/* dw_full.c: exact Steiner tree for <a,b,c> over F2 with L = b (the root 0 and the a*c full products
 * t_rj = sum_k x_rk y_kj) on the whole space of (ab) x (bc) matrices, storing one Dreyfus-Wagner array per orbit of
 * terminal subsets: <2,2,4> (2^32 states) in 19 arrays instead of 246, <2,3,2> (2^36) in 3.
 *
 * A state is a matrix M over F2 with rows i = r*b + k (x_rk) and columns j' = k*c + j (y_kj), bit i*C + j'. The group
 * S_a x S_c (permute the rows r of X, the columns j of Y; with the transpose (r,k) <-> (k,j) when a = c) acts by
 * rank-preserving bit permutations that permute the terminals, and dp[gS](v) = dp[S](g^-1 v). So dp[S] is stored only for
 * the least mask of each orbit, every other subset is read through the group element that maps it there, and the full set
 * is evaluated only at its merge points: answer = min_u merge(u) + rank(u).
 *
 * The relaxation f <- min_u f(u) + rank(u + .) runs in place: the rank-1 matrices with 0 are the union
 * over nonzero u in F2^R of u (x) F2^C; the row operations row_i ^= row_p (p = lowest row of u) map that group to
 * e_p (x) F2^C, so a Gray code over the rest of u's support visits all 2^R - 1 groups with one transvection pass and one
 * contiguous coset-min pass each. One sweep is exact (a minimal rank decomposition has independent u's, so its steps fall
 * in distinct groups, in any order); a second confirms.
 *
 * Build: gcc -O3 -march=native -fopenmp -o dw_full dw_full.c
 * Run:   OMP_NUM_THREADS=64 ./dw_full A B C            one JSON line: dp[S][0] for every subset, the answer, a tree
 *        ./dw_full A B C --check                      also every subset's array computed directly, compared with its
 *                                                     orbit image at every state (small shapes)
 *        ./dw_full --terms R C H1 .. Hn               any n <= 10 terminals (hex), no symmetry: against dw_f2
 *        ./dw_full --test-relax SEED                  the in-place relaxation against brute force on small shapes
 */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define INF 120
#define MAXN 10
#define MAXG 160
typedef uint64_t u64;
typedef uint8_t u8;

static int R, C;
static u64 N;

static inline u8 mn8(u8 a, u8 b) { return a < b ? a : b; }

static inline int rank_m(u64 m) {
    u64 basis[8];
    int r = 0;
    u64 mask = (1ULL << C) - 1;
    for (int i = 0; i < R; i++) {
        u64 x = (m >> (i * C)) & mask;
        for (int j = 0; j < r; j++) {
            u64 y = x ^ basis[j];
            if (y < x) x = y;
        }
        if (x) basis[r++] = x;
    }
    return r;
}

/* a bit permutation of the R*C index bits, applied row by row through lookup tables */
typedef struct {
    int pos[64];
    u64 lut[8][256];
} bperm_t;

static void bperm_build(bperm_t *P) {
    for (int r = 0; r < R; r++)
        for (int x = 0; x < (1 << C); x++) {
            u64 o = 0;
            for (int j = 0; j < C; j++)
                if ((x >> j) & 1) o |= 1ULL << P->pos[r * C + j];
            P->lut[r][x] = o;
        }
}

static inline u64 bapply(const bperm_t *P, u64 v) {
    u64 o = 0, mask = (1ULL << C) - 1;
    for (int r = 0; r < R; r++) o |= P->lut[r][(v >> (r * C)) & mask];
    return o;
}

static void *big_alloc(u64 n) {
    void *p = NULL;
    u64 al = n >= (1ULL << 24) ? 1ULL << 21 : 64, sz = (n + al - 1) & ~(al - 1);
    if (posix_memalign(&p, al, sz)) {
        fprintf(stderr, "out of memory (%llu bytes)\n", (unsigned long long)sz);
        exit(2);
    }
    madvise(p, sz, MADV_HUGEPAGE);
    return p;
}

/* f[M] = min(f[M], min_w f[M ^ (e_p (x) w)] + 1); returns whether anything changed */
static int coset_min_row(u8 *f, int p) {
    u64 Sp = 1ULL << (p * C), B = 1ULL << C, span = Sp * B;
    int changed = 0;
    if (p == 0) {
#pragma omp parallel for schedule(static) reduction(| : changed)
        for (u64 base = 0; base < N; base += B) {
            u8 *x = f + base, m = 255;
            for (u64 w = 0; w < B; w++) m = mn8(m, x[w]);
            u8 m1 = m >= INF ? INF : m + 1;
            for (u64 w = 0; w < B; w++)
                if (x[w] > m1) {
                    x[w] = m1;
                    changed = 1;
                }
        }
        return changed;
    }
    u64 CH = Sp < 64 ? Sp : 64, nhi = N / span, nl = Sp / CH;
#pragma omp parallel for schedule(static) reduction(| : changed)
    for (u64 it = 0; it < nhi * nl; it++) {
        u64 base = (it / nl) * span + (it % nl) * CH;
        u8 m[64];
        memset(m, 255, sizeof m);
        for (u64 w = 0; w < B; w++) {
            const u8 *x = f + base + w * Sp;
            for (u64 t = 0; t < CH; t++) m[t] = mn8(m[t], x[t]);
        }
        for (u64 t = 0; t < CH; t++) m[t] = m[t] >= INF ? INF : m[t] + 1;
        int ch = 0;
        for (u64 w = 0; w < B; w++) {
            u8 *x = f + base + w * Sp;
            for (u64 t = 0; t < CH; t++) {
                u8 c = mn8(x[t], m[t]);
                ch |= c != x[t];
                x[t] = c;
            }
        }
        changed |= ch;
    }
    return changed;
}

/* the array permuted by row_i ^= row_p: g[M] = f[M with row_i ^ row_p] */
static void transvect(u8 *f, int i, int p) {
    int lo = i < p ? i : p, hi = i < p ? p : i;
    u64 B = 1ULL << C, Slo = 1ULL << (lo * C), Shi = 1ULL << (hi * C);
    u64 CH = Slo < 64 ? Slo : 64;
    while (CH > 1 && B * B * CH > (1 << 20)) CH >>= 1;
    u64 na = Slo / CH, nb = Shi / (Slo * B), nc = N / (Shi * B);
    u64 sx = i < p ? Slo : Shi, sw = i < p ? Shi : Slo;
#pragma omp parallel
    {
        u8 *buf = malloc(B * B * CH);
#pragma omp for schedule(static)
        for (u64 it = 0; it < na * nb * nc; it++) {
            u64 a = (it % na) * CH, b = (it / na) % nb, c = it / (na * nb);
            u8 *base = f + a + b * Slo * B + c * Shi * B;
            if (CH == 1) {
                for (u64 x = 0; x < B; x++)
                    for (u64 w = 0; w < B; w++) buf[x * B + w] = base[x * sx + w * sw];
                for (u64 x = 0; x < B; x++)
                    for (u64 w = 0; w < B; w++) base[x * sx + w * sw] = buf[(x ^ w) * B + w];
            } else {
                for (u64 x = 0; x < B; x++)
                    for (u64 w = 0; w < B; w++) memcpy(buf + (x * B + w) * CH, base + x * sx + w * sw, CH);
                for (u64 x = 0; x < B; x++)
                    for (u64 w = 0; w < B; w++) memcpy(base + x * sx + w * sw, buf + ((x ^ w) * B + w) * CH, CH);
            }
        }
        free(buf);
    }
}

/* f <- min_u f(u) + rank(u + .), in place; returns the number of sweeps */
static int relax(u8 *f) {
    int sweeps = 0;
    for (;;) {
        sweeps++;
        int changed = 0;
        for (int p = 0; p < R; p++) {
            changed |= coset_min_row(f, p);
            int m = R - 1 - p;
            for (u64 g = 1; g < (1ULL << m); g++) {
                transvect(f, p + 1 + __builtin_ctzll(g), p);
                changed |= coset_min_row(f, p);
            }
            if (m) transvect(f, p + m, p);
        }
        if (!changed) return sweeps;
    }
}

/* ---- terminals, the group, and Dreyfus-Wagner over orbit representatives ---- */

static int n, A_, B_, C_;
static u64 term[MAXN];
static char label[MAXN][24];
static int ng;
static bperm_t *grp;              /* group elements as bit permutations */
static int gterm[MAXG][MAXN];     /* and as permutations of the terminals */

typedef struct {
    int rep;                      /* the orbit's least mask; val(S, v) = arr[rep][map(v)] */
    int map;                      /* group element taking S to rep (-1: identity) */
} sub_t;

typedef struct {
    sub_t sub[1 << MAXN];
    u8 *arr[1 << MAXN];
} dp_t;

static inline int val(const dp_t *D, int S, u64 v) {
    if (__builtin_popcount(S) == 1) return rank_m(term[__builtin_ctz(S)] ^ v);
    const sub_t *s = &D->sub[S];
    return D->arr[s->rep][s->map < 0 ? v : bapply(&grp[s->map], v)];
}

static inline int merge_at(const dp_t *D, int S, u64 v, int *split) {
    int low = S & -S, best = 255;
    for (int s1 = (S - 1) & S; s1; s1 = (s1 - 1) & S)
        if (s1 & low) {
            int x = val(D, s1, v) + val(D, S ^ s1, v);
            if (x < best) {
                best = x;
                if (split) *split = s1;
            }
        }
    return best;
}

static void build_group(int sym) {
    int perm_a[720][6], na = 0, perm_c[720][6], nc = 0;
    /* all permutations of 0..a-1 and 0..c-1 (a, c <= 5) */
    for (int which = 0; which < 2; which++) {
        int m = which ? C_ : A_, p[6], cnt = 0;
        for (int i = 0; i < m; i++) p[i] = i;
        for (;;) {
            memcpy(which ? perm_c[cnt] : perm_a[cnt], p, sizeof p);
            cnt++;
            if (!sym) break;
            int i = m - 2;
            while (i >= 0 && p[i] > p[i + 1]) i--;
            if (i < 0) break;
            int j = m - 1;
            while (p[j] < p[i]) j--;
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
            for (int x = i + 1, y = m - 1; x < y; x++, y--) {
                t = p[x];
                p[x] = p[y];
                p[y] = t;
            }
        }
        if (which) nc = cnt;
        else na = cnt;
    }
    int ntr = sym && A_ == C_ ? 2 : 1;
    ng = na * nc * ntr;
    if (ng > MAXG) {
        fprintf(stderr, "group of order %d is too large\n", ng);
        exit(2);
    }
    grp = calloc(ng, sizeof(bperm_t));
    int g = 0;
    for (int ia = 0; ia < na; ia++)
        for (int ic = 0; ic < nc; ic++)
            for (int e = 0; e < ntr; e++, g++) {
                int *s = perm_a[ia], *p = perm_c[ic];
                for (int r = 0; r < A_; r++)
                    for (int k = 0; k < B_; k++)
                        for (int k2 = 0; k2 < B_; k2++)
                            for (int j = 0; j < C_; j++) {
                                int i0 = (r * B_ + k) * C + k2 * C_ + j;
                                int r1 = s[r], j1 = p[j];
                                int i1 = e ? (j1 * B_ + k2) * C + k * C_ + r1 : (r1 * B_ + k) * C + k2 * C_ + j1;
                                grp[g].pos[i0] = i1;
                            }
                for (int r = 0; r < A_; r++)
                    for (int j = 0; j < C_; j++)
                        gterm[g][r * C_ + j] = e ? p[j] * C_ + s[r] : s[r] * C_ + p[j];
                bperm_build(&grp[g]);
                for (int q = 0; q < n; q++)
                    if (bapply(&grp[g], term[q]) != term[gterm[g][q]]) {
                        fprintf(stderr, "group element %d does not map terminal %d to terminal %d\n", g, q, gterm[g][q]);
                        exit(3);
                    }
            }
}

static int img(int g, int S) {
    int o = 0;
    for (int q = 0; q < n; q++)
        if ((S >> q) & 1) o |= 1 << gterm[g][q];
    return o;
}

static void orbit_table(dp_t *D, int sym) {
    int full = (1 << n) - 1;
    for (int S = 1; S < full; S++) {
        D->sub[S] = (sub_t){S, -1};
        if (!sym) continue;
        for (int g = 0; g < ng; g++) {
            int I = img(g, S);
            if (I < D->sub[S].rep) D->sub[S] = (sub_t){I, g};
        }
    }
}

static int fill_all(dp_t *D, int *nrep) {
    int full = (1 << n) - 1, maxsw = 0;
    *nrep = 0;
    for (int k = 2; k < n; k++)
        for (int S = 1; S < full; S++) {
            if (__builtin_popcount(S) != k || D->sub[S].rep != S) continue;
            double s0 = omp_get_wtime();
            u8 *f = D->arr[S] = big_alloc(N);
#pragma omp parallel for schedule(static)
            for (u64 v = 0; v < N; v++) {
                int x = merge_at(D, S, v, NULL);
                f[v] = x > INF ? INF : x;
            }
            int sw = relax(f);
            if (sw > maxsw) maxsw = sw;
            (*nrep)++;
            if (N >= (1ULL << 26))
                fprintf(stderr, "dw_full: subset %d (size %d) done, %d sweeps, %.1f s, dp[S][0] = %d\n", S, k, sw,
                        omp_get_wtime() - s0, f[0]);
        }
    return maxsw;
}

typedef struct {
    u64 a[512], b[512];
    int ne, err;
} tree_t;

static u64 r1[65536];
static int nr1;

static void add_edge(tree_t *T, u64 a, u64 b) {
    if (T->ne >= 512) {
        T->err = 2;
        return;
    }
    T->a[T->ne] = a;
    T->b[T->ne] = b;
    T->ne++;
}

static void geodesic(tree_t *T, u64 v, u64 t) {
    while (v != t && !T->err) {
        int d = rank_m(v ^ t), done = 0;
        for (int i = 0; i < nr1 && !done; i++)
            if (rank_m((v ^ r1[i]) ^ t) == d - 1) {
                add_edge(T, v, v ^ r1[i]);
                v ^= r1[i];
                done = 1;
            }
        if (!done) T->err = 1;
    }
}

static void extract(const dp_t *D, tree_t *T, int S, u64 v) {
    while (!T->err) {
        if (__builtin_popcount(S) == 1) {
            geodesic(T, v, term[__builtin_ctz(S)]);
            return;
        }
        int x = val(D, S, v), s1 = 0;
        if (merge_at(D, S, v, &s1) == x) {
            extract(D, T, s1, v);
            extract(D, T, S ^ s1, v);
            return;
        }
        int found = 0;
        for (int i = 0; i < nr1 && !found; i++)
            if (val(D, S, v ^ r1[i]) + 1 == x) {
                add_edge(T, v, v ^ r1[i]);
                v ^= r1[i];
                found = 1;
            }
        if (!found) T->err = 1;
    }
}

static int run(const char *name, int sym, int check) {
    double t0 = omp_get_wtime();
    N = 1ULL << (R * C);
    nr1 = 0;
    for (u64 u = 1; u < (1ULL << R); u++)
        for (u64 w = 1; w < (1ULL << C); w++) {
            u64 m = 0;
            for (int i = 0; i < R; i++)
                if ((u >> i) & 1) m |= w << (i * C);
            r1[nr1++] = m;
        }
    int full = (1 << n) - 1, nrep = 0;
    dp_t *D = calloc(1, sizeof(dp_t));
    orbit_table(D, sym);
    int sweeps = fill_all(D, &nrep);
    double tdp = omp_get_wtime() - t0;
    long mism = 0;
    if (check) {
        dp_t *E = calloc(1, sizeof(dp_t));
        int nr2;
        orbit_table(E, 0);
        fill_all(E, &nr2);
        for (int S = 1; S < full; S++) {
            if (__builtin_popcount(S) < 2) continue;
            long bad = 0;
#pragma omp parallel for reduction(+ : bad)
            for (u64 v = 0; v < N; v++) bad += val(D, S, v) != E->arr[S][v];
            if (bad) fprintf(stderr, "check: subset %d differs from its orbit image at %ld states\n", S, bad);
            mism += bad != 0;
        }
    }
    double s0 = omp_get_wtime();
    int best = 255;
    u64 bestu = 0;
#pragma omp parallel
    {
        int lb = 255;
        u64 lu = 0;
#pragma omp for schedule(static)
        for (u64 u = 0; u < N; u++) {
            int x = merge_at(D, full, u, NULL) + rank_m(u);
            if (x < lb) {
                lb = x;
                lu = u;
            }
        }
#pragma omp critical
        if (lb < best || (lb == best && lu < bestu)) {
            best = lb;
            bestu = lu;
        }
    }
    double tfin = omp_get_wtime() - s0;
    tree_t *T = calloc(1, sizeof(tree_t));
    geodesic(T, 0, bestu);
    int s1 = 0;
    merge_at(D, full, bestu, &s1);
    extract(D, T, s1, bestu);
    extract(D, T, full ^ s1, bestu);
    FILE *o = stdout;
    fprintf(o, "{\"name\":\"%s\",\"field\":2,\"R\":%d,\"C\":%d,\"k\":%d,\"n\":%d,\"states\":%llu,\"rank1\":%d,"
               "\"groups\":%d,\"threads\":%d,\"symmetry_order\":%d,\"arrays\":%d,\"max_sweeps\":%d,\"check\":%d,"
               "\"check_mismatch\":%ld,\"dp_s\":%.1f,\"final_s\":%.1f,\"wall_s\":%.1f,\"answer\":%d,\"root_dp\":[0",
            name, R, C, R * C, n, (unsigned long long)N, nr1, (1 << R) - 1, omp_get_max_threads(), sym ? ng : 1, nrep,
            sweeps, check, mism, tdp, tfin, omp_get_wtime() - t0, best);
    for (int S = 1; S <= full; S++) fprintf(o, ",%d", S == full ? best : val(D, S, 0));
    fprintf(o, "],\"merge_point\":\"%llx\",\"terminals\":[", (unsigned long long)bestu);
    for (int q = 0; q < n; q++) fprintf(o, "%s[\"%llx\",\"%s\"]", q ? "," : "", (unsigned long long)term[q], label[q]);
    fprintf(o, "],\"tree_err\":%d,\"tree\":[", T->err);
    for (int i = 0; i < T->ne; i++)
        fprintf(o, "%s[\"%llx\",\"%llx\"]", i ? "," : "", (unsigned long long)T->a[i], (unsigned long long)T->b[i]);
    fprintf(o, "]}\n");
    fflush(o);
    return mism ? 4 : T->err ? 5 : 0;
}

/* the in-place relaxation against brute force: R x C shapes up to 16 bits, rows up to 8 bits */
static int test_relax(unsigned seed) {
    int shapes[][2] = {{2, 2}, {2, 3}, {3, 2}, {3, 3}, {2, 5}, {4, 3}, {3, 4}, {4, 4},
                       {5, 3}, {2, 6}, {6, 2}, {2, 7}, {2, 8}, {1, 8}, {8, 2}};
    int fails = 0;
    srand(seed);
    for (unsigned s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
        for (int rep = 0; rep < 3; rep++) {
            R = shapes[s][0];
            C = shapes[s][1];
            N = 1ULL << (R * C);
            u8 *f = malloc(N), *g = malloc(N);
            int dense = rep == 2 && R * C <= 12;
            for (u64 v = 0; v < N; v++) f[v] = dense ? rand() % 30 : (rand() % 97 == 0 ? rand() % 12 : INF);
            f[rand() % N] = 0;
            for (u64 v = 0; v < N; v++) {
                int best = INF;
                for (u64 u = 0; u < N; u++)
                    if (f[u] < INF) {
                        int x = f[u] + rank_m(u ^ v);
                        if (x < best) best = x;
                    }
                g[v] = best;
            }
            int sw = relax(f);
            u64 bad = 0;
            for (u64 v = 0; v < N; v++) bad += f[v] != g[v];
            printf("{\"test\":\"relax\",\"R\":%d,\"C\":%d,\"dense\":%d,\"sweeps\":%d,\"mismatches\":%llu}\n", R, C, dense,
                   sw, (unsigned long long)bad);
            fails += bad != 0;
            free(f);
            free(g);
        }
    printf("{\"test\":\"relax\",\"fails\":%d,\"pass\":%s}\n", fails, fails ? "false" : "true");
    return fails ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc >= 3 && !strcmp(argv[1], "--test-relax")) return test_relax((unsigned)atoi(argv[2]));
    if (argc >= 5 && !strcmp(argv[1], "--terms")) {
        R = atoi(argv[2]);
        C = atoi(argv[3]);
        n = argc - 4;
        if (R < 1 || C < 1 || R > 8 || C > 8 || R * C > 36 || n < 2 || n > MAXN) {
            fprintf(stderr, "--terms: R, C <= 8, R*C <= 36, 2 <= n <= %d\n", MAXN);
            return 2;
        }
        for (int q = 0; q < n; q++) {
            term[q] = strtoull(argv[4 + q], NULL, 16);
            snprintf(label[q], sizeof label[q], "t%d", q);
        }
        return run("terms", 0, 0);
    }
    if (argc < 4) {
        fprintf(stderr, "usage: %s A B C [--check] | --terms R C H1..Hn | --test-relax SEED\n", argv[0]);
        return 2;
    }
    A_ = atoi(argv[1]);
    B_ = atoi(argv[2]);
    C_ = atoi(argv[3]);
    R = A_ * B_;
    C = B_ * C_;
    n = A_ * C_;
    if (A_ < 1 || B_ < 1 || C_ < 1 || A_ > 5 || C_ > 5 || R > 8 || C > 8 || R * C > 36 || n < 2 || n > MAXN) {
        fprintf(stderr, "shape: a, c <= 5, ab <= 8, bc <= 8, ab*bc <= 36, 2 <= ac <= %d\n", MAXN);
        return 2;
    }
    for (int r = 0; r < A_; r++)
        for (int j = 0; j < C_; j++) {
            u64 t = 0;
            for (int k = 0; k < B_; k++) t |= 1ULL << ((r * B_ + k) * C + k * C_ + j);
            term[r * C_ + j] = t;
            snprintf(label[r * C_ + j], sizeof label[0], "r%dc%d", r, j);
        }
    build_group(1);
    char name[64];
    snprintf(name, sizeof name, "mm%d%d%d_L%d_F2_orbit", A_, B_, C_, B_);
    return run(name, 1, argc > 4 && !strcmp(argv[4], "--check"));
}
