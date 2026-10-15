/* dw_2b2.c: exact Steiner tree for <2,b,2> over F2 with L = b (the root 0 and the four products t_rc = sum_k x_rk y_kc),
 * on the whole space of 2b x 2b matrices: 2^(4b^2) states, 2^36 for b = 3.
 *
 * A state is a matrix M over F2 with rows i = r*b + k (x_rk) and columns j = 2k + c (y_kc), bit i*C + j. Dreyfus-Wagner
 * over the terminals {t00, t01, t10, t11} with root 0 needs dp[S][.] for every proper nonempty S. The terminal set's
 * symmetries (rho: swap r, gamma: swap c, tau: transpose with (r,k) <-> (k,c)) are rank-preserving bit permutations,
 * and dp[gS](v) = dp[S](g^-1 v), so three arrays carry every proper subset: A = dp[{t00,t01}], D = dp[{t00,t11}] and
 * T = dp[{t00,t01,t10}]; the full set is only evaluated at its merge points, answer = min_u merge(u) + rank(u).
 *
 * A relaxation f <- min_u f(u) + rank(u + .) runs in place (Gauss-Seidel, until a sweep changes nothing). The rank-1
 * matrices with 0 are the union over nonzero u in F2^R of u (x) F2^C. For the rows p = lowest row of u and Q the rest,
 * the row operations psi_Q (row_i ^= row_p for i in Q) map u (x) F2^C to e_p (x) F2^C, so the array is permuted by one
 * transvection at a time along a Gray code over Q and each group's coset min is a min over row p's C bits. A sweep is
 * 2^R - 1 coset mins and as many transvections, each one pass over the array, and needs no second array.
 *
 * Build: gcc -O3 -march=native -fopenmp -o dw_2b2 dw_2b2.c
 * Run:   OMP_NUM_THREADS=64 ./dw_2b2 B             one JSON line: the answer, dp[S][0] for every S, an optimal tree
 *        ./dw_2b2 --test-relax SEED              the in-place relaxation against brute force on small shapes
 *        ./dw_2b2 2 --nosym                       b = 2 with every subset's array computed directly, checked against
 *                                                 the symmetric images at every state
 *        ./dw_2b2 B --terms H1 H2 H3 H4           any four terminals (hex), every array computed directly: the
 *                                                 merges, root and tree extraction against dw_f2 on the same instance
 */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define INF 120
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
    u64 lut[8][64];
    int pos[64];
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

/* P∘Q: v -> P(Q(v)) */
static void bperm_compose(bperm_t *out, const bperm_t *P, const bperm_t *Q) {
    for (int i = 0; i < R * C; i++) out->pos[i] = P->pos[Q->pos[i]];
    bperm_build(out);
}

static void *big_alloc(u64 n) {
    void *p = NULL;
    u64 sz = (n + (1 << 21) - 1) & ~((u64)(1 << 21) - 1);
    if (posix_memalign(&p, 1 << 21, sz)) {
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
    u64 CH = Slo < 64 ? Slo : 64, na = Slo / CH, nb = Shi / (Slo * B), nc = N / (Shi * B);
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

/* ---- the <2,b,2> instance ---- */

static int b_;
static u64 term[4];                 /* t00, t01, t10, t11: bit q of a subset mask */
static bperm_t ID, RHO, GAM, TAU, TAUGAM, GAMRHO;
static u8 *arr[16];                 /* the arrays held: by subset mask (A = 3, D = 9, T = 7), or all with --nosym */
typedef struct {
    int kind;                       /* 0 singleton, 1 array */
    int q;                          /* singleton: terminal index */
    int src;                        /* array: the subset whose array holds it */
    const bperm_t *map;             /* val(S, v) = arr[src][map(v)] */
} sub_t;
static sub_t sub[16];

static void build_instance(int b) {
    b_ = b;
    R = C = 2 * b;
    N = 1ULL << (R * C);
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++) {
            u64 t = 0;
            for (int k = 0; k < b; k++) t |= 1ULL << ((r * b + k) * C + 2 * k + c);
            term[r * 2 + c] = t;
        }
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            int r = i / b, k = i % b, kk = j / 2, c = j % 2;
            ID.pos[i * C + j] = i * C + j;
            RHO.pos[i * C + j] = ((1 - r) * b + k) * C + j;
            GAM.pos[i * C + j] = i * C + 2 * kk + 1 - c;
            /* tau: entry (row r*b+k, col 2kk+c) -> (row c*b+kk, col 2k+r) */
            TAU.pos[i * C + j] = (c * b + kk) * C + 2 * k + r;
        }
    bperm_build(&ID);
    bperm_build(&RHO);
    bperm_build(&GAM);
    bperm_build(&TAU);
    bperm_compose(&TAUGAM, &TAU, &GAM);
    bperm_compose(&GAMRHO, &GAM, &RHO);
}

static void sym_table(int nosym) {
    for (int S = 1; S < 15; S++) {
        if (__builtin_popcount(S) == 1) {
            sub[S] = (sub_t){0, __builtin_ctz(S), 0, NULL};
            continue;
        }
        sub[S] = (sub_t){1, 0, S, &ID};
    }
    if (nosym) return;
    /* bits: 0 t00, 1 t01, 2 t10, 3 t11 */
    sub[12] = (sub_t){1, 0, 3, &RHO};    /* {t10,t11} = rho{t00,t01} */
    sub[5] = (sub_t){1, 0, 3, &TAU};     /* {t00,t10} = tau{t00,t01} */
    sub[10] = (sub_t){1, 0, 3, &TAUGAM}; /* {t01,t11} = gamma tau{t00,t01} */
    sub[6] = (sub_t){1, 0, 9, &RHO};     /* {t01,t10} = rho{t00,t11} */
    sub[11] = (sub_t){1, 0, 7, &GAM};    /* {t00,t01,t11} = gamma T */
    sub[13] = (sub_t){1, 0, 7, &RHO};    /* {t00,t10,t11} = rho T */
    sub[14] = (sub_t){1, 0, 7, &GAMRHO}; /* {t01,t10,t11} = rho gamma T */
    /* each image: map(S) must be the source subset, as sets of terminals */
    for (int S = 1; S < 15; S++) {
        if (sub[S].kind != 1 || sub[S].src == S) continue;
        int img = 0;
        for (int q = 0; q < 4; q++)
            if ((S >> q) & 1) {
                u64 t = bapply(sub[S].map, term[q]);
                int hit = -1;
                for (int q2 = 0; q2 < 4; q2++)
                    if (term[q2] == t) hit = q2;
                if (hit < 0) {
                    fprintf(stderr, "symmetry table: subset %d maps a terminal outside the set\n", S);
                    exit(3);
                }
                img |= 1 << hit;
            }
        if (img != sub[S].src) {
            fprintf(stderr, "symmetry table: subset %d maps to %d, not %d\n", S, img, sub[S].src);
            exit(3);
        }
    }
}

static inline int val(int S, u64 v) {
    const sub_t *s = &sub[S];
    if (s->kind == 0) return rank_m(term[s->q] ^ v);
    return arr[s->src][bapply(s->map, v)];
}

static inline int merge_at(int S, u64 v, int *split) {
    int low = S & -S, best = 255;
    for (int s1 = (S - 1) & S; s1; s1 = (s1 - 1) & S)
        if (s1 & low) {
            int x = val(s1, v) + val(S ^ s1, v);
            if (x < best) {
                best = x;
                if (split) *split = s1;
            }
        }
    return best;
}

static void fill_merge(int S) {
    u8 *f = arr[S];
#pragma omp parallel for schedule(static)
    for (u64 v = 0; v < N; v++) {
        int x = merge_at(S, v, NULL);
        f[v] = x > INF ? INF : x;
    }
}

typedef struct {
    u64 a[256], b[256];
    int ne, err;
} tree_t;

static u64 r1[4096];
static int nr1;

static void add_edge(tree_t *T, u64 a, u64 b) {
    if (T->ne >= 256) {
        T->err = 2;
        return;
    }
    T->a[T->ne] = a;
    T->b[T->ne] = b;
    T->ne++;
}

/* a geodesic from v to t: rank(t ^ v) rank-1 steps */
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

static void extract(tree_t *T, int S, u64 v) {
    while (!T->err) {
        if (__builtin_popcount(S) == 1) {
            geodesic(T, v, term[__builtin_ctz(S)]);
            return;
        }
        int x = val(S, v), s1 = 0;
        if (merge_at(S, v, &s1) == x) {
            extract(T, s1, v);
            extract(T, S ^ s1, v);
            return;
        }
        int found = 0;
        for (int i = 0; i < nr1 && !found; i++)
            if (val(S, v ^ r1[i]) + 1 == x) {
                add_edge(T, v, v ^ r1[i]);
                v ^= r1[i];
                found = 1;
            }
        if (!found) T->err = 1;
    }
}

static void print_mat(FILE *o, u64 m) { fprintf(o, "\"%llx\"", (unsigned long long)m); }

static int run(int b, int nosym, const u64 *terms) {
    double t0 = omp_get_wtime();
    build_instance(b);
    if (terms) memcpy(term, terms, sizeof term);
    sym_table(nosym);
    nr1 = 0;
    for (u64 u = 1; u < (1ULL << R); u++)
        for (u64 w = 1; w < (1ULL << C); w++) {
            u64 m = 0;
            for (int i = 0; i < R; i++)
                if ((u >> i) & 1) m |= w << (i * C);
            r1[nr1++] = m;
        }
    int order[16], no = 0, sweeps[16] = {0};
    double secs[16] = {0};
    for (int k = 2; k <= 3; k++)
        for (int S = 1; S < 15; S++)
            if (__builtin_popcount(S) == k && sub[S].kind == 1 && sub[S].src == S) order[no++] = S;
    for (int i = 0; i < no; i++) {
        int S = order[i];
        double s0 = omp_get_wtime();
        arr[S] = big_alloc(N);
        fill_merge(S);
        sweeps[S] = relax(arr[S]);
        secs[S] = omp_get_wtime() - s0;
        fprintf(stderr, "dw_2b2 b=%d: subset %d done (%d sweeps, %.1f s, dp[S][0] = %d)\n", b, S, sweeps[S], secs[S],
                arr[S][0]);
    }
    int mism = 0;
    if (nosym && !terms) {
        /* every image the symmetric table would use, against the directly computed array */
        build_instance(b);
        sub_t direct[16];
        memcpy(direct, sub, sizeof sub);
        sym_table(0);
        for (int S = 1; S < 15; S++) {
            if (sub[S].kind != 1 || sub[S].src == S) continue;
            u64 bad = 0;
#pragma omp parallel for reduction(+ : bad)
            for (u64 v = 0; v < N; v++) bad += arr[S][v] != arr[sub[S].src][bapply(sub[S].map, v)];
            if (bad) fprintf(stderr, "nosym: subset %d differs from its image at %llu states\n", S, (unsigned long long)bad);
            mism += bad != 0;
        }
        memcpy(sub, direct, sizeof sub);
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
            int x = merge_at(15, u, NULL);
            if (x + rank_m(u) < lb) {
                lb = x + rank_m(u);
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
    merge_at(15, bestu, &s1);
    extract(T, s1, bestu);
    extract(T, 15 ^ s1, bestu);
    FILE *o = stdout;
    fprintf(o, "{\"name\":\"mm2%d2_L%d_F2_sym\",\"field\":2,\"R\":%d,\"C\":%d,\"k\":%d,\"n\":4,\"states\":%llu,"
               "\"rank1\":%d,\"groups\":%d,\"threads\":%d,\"nosym\":%d,\"nosym_mismatch\":%d,\"answer\":%d,\"root_dp\":{",
            b, b, R, C, R * C, (unsigned long long)N, nr1, (1 << R) - 1, omp_get_max_threads(), nosym, mism, best);
    for (int S = 1; S < 15; S++) fprintf(o, "%s\"%d\":%d", S > 1 ? "," : "", S, val(S, 0));
    fprintf(o, ",\"15\":%d},\"arrays\":[", best);
    for (int i = 0; i < no; i++)
        fprintf(o, "%s{\"subset\":%d,\"sweeps\":%d,\"s\":%.1f}", i ? "," : "", order[i], sweeps[order[i]],
                secs[order[i]]);
    fprintf(o, "],\"final_s\":%.1f,\"wall_s\":%.1f,\"merge_point\":", tfin, omp_get_wtime() - t0);
    print_mat(o, bestu);
    fprintf(o, ",\"terminals\":[");
    const char *lab[4] = {"r0c0", "r0c1", "r1c0", "r1c1"};
    for (int q = 0; q < 4; q++) {
        fprintf(o, "%s[", q ? "," : "");
        print_mat(o, term[q]);
        fprintf(o, ",\"%s\"]", lab[q]);
    }
    fprintf(o, "],\"tree_err\":%d,\"tree\":[", T->err);
    for (int i = 0; i < T->ne; i++) {
        fprintf(o, "%s[", i ? "," : "");
        print_mat(o, T->a[i]);
        fprintf(o, ",");
        print_mat(o, T->b[i]);
        fprintf(o, "]");
    }
    fprintf(o, "]}\n");
    fflush(o);
    return mism ? 4 : 0;
}

/* the in-place relaxation against brute force: R x C shapes up to 16 bits, random sparse sources */
static int test_relax(unsigned seed) {
    int shapes[][2] = {{2, 2}, {2, 3}, {3, 2}, {3, 3}, {2, 5}, {4, 3}, {3, 4}, {4, 4}, {5, 3}, {2, 6}, {6, 2}};
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
    if (argc < 2) {
        fprintf(stderr, "usage: %s B [--nosym] | --test-relax SEED\n", argv[0]);
        return 2;
    }
    int b = atoi(argv[1]);
    if (b < 1 || b > 3) {
        fprintf(stderr, "b must be 1, 2 or 3 (2b x 2b matrices, at most 36 bits)\n");
        return 2;
    }
    if (argc == 7 && !strcmp(argv[2], "--terms")) {
        u64 t[4];
        for (int q = 0; q < 4; q++) t[q] = strtoull(argv[3 + q], NULL, 16);
        return run(b, 1, t);
    }
    return run(b, argc > 2 && !strcmp(argv[2], "--nosym"), NULL);
}
