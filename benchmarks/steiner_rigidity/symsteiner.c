/* Exact Steiner numbers in the rank metric on symmetric n x n matrices over GF(q) (Dreyfus-Wagner).
 *
 * In characteristic != 2 the rank of a symmetric matrix is the least number of rank-1 symmetric summands c g g^T, so
 * rank(A - B) is the distance in the Cayley graph whose generators are the rank-1 symmetric matrices, and the
 * min-plus convolution f -> min_u f(u) + rank(v - u) is a unit-weight relaxation over those generators.
 *
 * A matrix is its upper triangle, m = n(n+1)/2 digits base q in row-major order (i <= j), index sum_k d_k q^k.
 * Input on stdin: q n k root  then k terminal indices (root = 1: a terminal 0 is added and the answer is
 * St({0} U T); root = 0: St(T)). Output: one line "St <value>", then "S <mask> <value at 0>" for every mask. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int q, n, m, m1, m2;
static int64_t N, Q1, Q2;
static uint32_t *add_lo, *add_hi;
static int ngen;
static int64_t *gen_lo, *gen_hi;

static int64_t ipow(int64_t b, int e) { int64_t r = 1; while (e--) r *= b; return r; }

static void tables(void) {
    m = n * (n + 1) / 2;
    m1 = m / 2 + (m % 2);
    m2 = m - m1;
    Q1 = ipow(q, m1);
    Q2 = ipow(q, m2);
    N = Q1 * Q2;
    add_lo = malloc(sizeof(uint32_t) * Q1 * Q1);
    add_hi = malloc(sizeof(uint32_t) * Q2 * Q2);
    int da[32], db[32];
    for (int64_t a = 0; a < Q1; a++) {
        int64_t t = a; for (int i = 0; i < m1; i++) { da[i] = t % q; t /= q; }
        for (int64_t b = 0; b < Q1; b++) {
            int64_t s = b, r = 0, pw = 1;
            for (int i = 0; i < m1; i++) { db[i] = s % q; s /= q; r += ((da[i] + db[i]) % q) * pw; pw *= q; }
            add_lo[a * Q1 + b] = (uint32_t)r;
        }
    }
    for (int64_t a = 0; a < Q2; a++) {
        int64_t t = a; for (int i = 0; i < m2; i++) { da[i] = t % q; t /= q; }
        for (int64_t b = 0; b < Q2; b++) {
            int64_t s = b, r = 0, pw = 1;
            for (int i = 0; i < m2; i++) { db[i] = s % q; s /= q; r += ((da[i] + db[i]) % q) * pw; pw *= q; }
            add_hi[a * Q2 + b] = (uint32_t)r;
        }
    }
}

static int64_t encode(int *M) {
    int64_t r = 0, pw = 1;
    for (int i = 0; i < n; i++) for (int j = i; j < n; j++) { r += M[i * n + j] * pw; pw *= q; }
    return r;
}

static void generators(void) {
    int64_t cap = ipow(q, n) * q;
    int64_t *all = malloc(sizeof(int64_t) * cap);
    int64_t cnt = 0;
    int g[16], M[256];
    for (int64_t gi = 1; gi < ipow(q, n); gi++) {
        int64_t t = gi; for (int i = 0; i < n; i++) { g[i] = t % q; t /= q; }
        for (int c = 1; c < q; c++) {
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) M[i * n + j] = (c * g[i] * g[j]) % q;
            int64_t e = encode(M);
            int dup = 0;
            for (int64_t s = 0; s < cnt; s++) if (all[s] == e) { dup = 1; break; }
            if (!dup) all[cnt++] = e;
        }
    }
    ngen = (int)cnt;
    gen_lo = malloc(sizeof(int64_t) * cnt);
    gen_hi = malloc(sizeof(int64_t) * cnt);
    for (int64_t s = 0; s < cnt; s++) { gen_lo[s] = all[s] % Q1; gen_hi[s] = all[s] / Q1; }
    free(all);
}

static inline int64_t addg(int64_t v, int s) {
    int64_t lo = v % Q1, hi = v / Q1;
    return (int64_t)add_lo[lo * Q1 + gen_lo[s]] + Q1 * (int64_t)add_hi[hi * Q2 + gen_hi[s]];
}

/* f <- min_u f(u) + rank(v - u), in place, by unit relaxations until stable. */
static void relax(uint8_t *f, uint8_t *tmp) {
    for (;;) {
        int64_t changed = 0;
#pragma omp parallel for schedule(static) reduction(+ : changed)
        for (int64_t v = 0; v < N; v++) {
            uint8_t best = f[v];
            for (int s = 0; s < ngen; s++) {
                uint8_t c = f[addg(v, s)];
                if ((uint8_t)(c + 1) < best && c != 255) best = c + 1;
            }
            tmp[v] = best;
            if (best != f[v]) changed++;
        }
        memcpy(f, tmp, N);
        if (!changed) break;
    }
}

int main(void) {
    int k, root;
    if (scanf("%d %d %d %d", &q, &n, &k, &root) != 4) return 2;
    int64_t term[32];
    for (int i = 0; i < k; i++) if (scanf("%ld", &term[i]) != 1) return 2;
    tables();
    generators();
    fprintf(stderr, "q=%d n=%d N=%ld generators=%d terminals=%d root=%d\n", q, n, (long)N, ngen, k, root);
    int kk = root ? k : k - 1;
    int64_t pivot = root ? 0 : term[k - 1];
    int nm = 1 << kk;
    uint8_t **S = calloc(nm, sizeof(uint8_t *));
    uint8_t *tmp = malloc(N);
    for (int i = 0; i < kk; i++) {
        uint8_t *f = malloc(N);
        memset(f, 255, N);
        f[term[i]] = 0;
        relax(f, tmp);
        S[1 << i] = f;
    }
    for (int mask = 1; mask < nm; mask++) {
        if (__builtin_popcount(mask) < 2) continue;
        uint8_t *f = malloc(N);
        memset(f, 255, N);
        int low = mask & (-mask);
        for (int sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask) {
            if (!(sub & low)) continue;
            uint8_t *a = S[sub], *b = S[mask ^ sub];
#pragma omp parallel for schedule(static)
            for (int64_t v = 0; v < N; v++) {
                int c = (int)a[v] + (int)b[v];
                if (c < f[v]) f[v] = (uint8_t)c;
            }
        }
        relax(f, tmp);
        S[mask] = f;
    }
    printf("St %d\n", S[nm - 1][pivot]);
    for (int mask = 1; mask < nm; mask++) printf("S %d %d\n", mask, S[mask][pivot]);
    return 0;
}
