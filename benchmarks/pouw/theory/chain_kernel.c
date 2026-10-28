/* Runs of two across adjacent committed words when the words are absorbed different numbers of times (the
   fractional round: word j once, word j+1 twice), at width n <= 6, by meeting in the middle over every word pair.

   Round(w) is receipt_folds_r9.xorin_step at width n with keys (a0, a1) and rotations (r1, r2). The chain is
   Round(w1)^c1 ∘ Round(w0)^c0. A kernel element is (w0', w1') != (w0, w1) with the same chain output on every state;
   it is trivial when Round(w0')^c0 == Round(w0)^c0 (each word's own kernel), and a run of two otherwise.
   For each key pair from stdin and each (c0, c1) in {1,2}^2 this prints
     S a0 a1 c0 c1 total nontrivial
   and up to 16 nontrivial elements as  N a0 a1 c0 c1 w0'[4] w1'[4].

   usage: chain_kernel n r1 r2 w0_0 w0_1 w0_2 w0_3 w1_0 w1_1 w1_2 w1_3 < keys
   build: gcc -O2 -fopenmp -o chain_kernel chain_kernel.c */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int n, r1, r2;
static uint32_t mask, ns;

static inline uint32_t rotl(uint32_t x, int r) { return ((x << r) | (x >> (n - r))) & mask; }

static inline uint32_t rnd(uint32_t y, const uint32_t *w, uint32_t a0, uint32_t a1) {
    y = (((rotl(y, r1) ^ w[0]) * a0) + w[1]) & mask;
    return (((rotl(y, r2) ^ w[2]) * a1) + w[3]) & mask;
}

static inline void unpack(uint32_t idx, uint32_t *w) {
    for (int i = 0; i < 4; i++) w[i] = (idx >> (i * n)) & mask;
}

static inline uint64_t hsig(const uint8_t *s) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t y = 0; y < ns; y++) { h ^= s[y]; h *= 1099511628211ull; }
    return h;
}

typedef struct { uint64_t h; uint32_t i; } ent;

static int cmp(const void *a, const void *b) {
    uint64_t x = ((const ent *)a)->h, y = ((const ent *)b)->h;
    return x < y ? -1 : x > y;
}

static void power(uint8_t *out, const uint32_t *w, int c, uint32_t a0, uint32_t a1) {
    for (uint32_t y = 0; y < ns; y++) {
        uint32_t t = y;
        for (int q = 0; q < c; q++) t = rnd(t, w, a0, a1);
        out[y] = (uint8_t)t;
    }
}

int main(int argc, char **argv) {
    if (argc != 12) { fprintf(stderr, "usage: chain_kernel n r1 r2 w0[4] w1[4] < keys\n"); return 2; }
    n = atoi(argv[1]); r1 = atoi(argv[2]); r2 = atoi(argv[3]);
    if (n < 2 || n > 6) { fprintf(stderr, "need 2 <= n <= 6\n"); return 2; }
    mask = (1u << n) - 1; ns = 1u << n;
    uint32_t W0[4], W1[4];
    for (int i = 0; i < 4; i++) { W0[i] = strtoul(argv[4 + i], 0, 0) & mask; W1[i] = strtoul(argv[8 + i], 0, 0) & mask; }
    uint32_t N = 1u << (4 * n);
    ent *L = malloc(sizeof(ent) * (size_t)N);
    unsigned long a0l, a1l;
    while (scanf("%lu %lu", &a0l, &a1l) == 2) {
        uint32_t a0 = a0l & mask, a1 = a1l & mask;
        for (int c0 = 1; c0 <= 2; c0++) {
            uint8_t f0[64];
            power(f0, W0, c0, a0, a1);
#pragma omp parallel for schedule(static)
            for (uint32_t i = 0; i < N; i++) {
                uint32_t w[4]; uint8_t s[64];
                unpack(i, w);
                power(s, w, c0, a0, a1);
                L[i].h = hsig(s); L[i].i = i;
            }
            qsort(L, N, sizeof(ent), cmp);
            for (int c1 = 1; c1 <= 2; c1++) {
                uint8_t f1[64], tot[64];
                power(f1, W1, c1, a0, a1);
                for (uint32_t y = 0; y < ns; y++) tot[y] = f1[f0[y]];
                unsigned long total = 0, nontriv = 0;
                int printed = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+:total, nontriv)
                for (uint32_t j = 0; j < N; j++) {
                    uint32_t w[4]; uint8_t g[64], inv[64], s[64];
                    unpack(j, w);
                    power(g, w, c1, a0, a1);
                    for (uint32_t y = 0; y < ns; y++) inv[g[y]] = (uint8_t)y;
                    for (uint32_t y = 0; y < ns; y++) s[y] = inv[tot[y]];
                    uint64_t h = hsig(s);
                    size_t lo = 0, hi = N;
                    while (lo < hi) { size_t mid = (lo + hi) / 2; if (L[mid].h < h) lo = mid + 1; else hi = mid; }
                    for (size_t k = lo; k < N && L[k].h == h; k++) {
                        uint32_t v[4]; uint8_t t[64];
                        unpack(L[k].i, v);
                        power(t, v, c0, a0, a1);
                        if (memcmp(t, s, ns)) continue;
                        int same = 1;
                        for (int q = 0; q < 4; q++) same &= (v[q] == W0[q]) && (w[q] == W1[q]);
                        if (same) continue;
                        total++;
                        if (memcmp(t, f0, ns)) {
                            nontriv++;
#pragma omp critical
                            if (printed < 16) {
                                printed++;
                                printf("N %u %u %d %d %u %u %u %u %u %u %u %u\n", a0, a1, c0, c1, v[0], v[1], v[2], v[3],
                                       w[0], w[1], w[2], w[3]);
                            }
                        }
                    }
                }
                printf("S %u %u %d %d %lu %lu\n", a0, a1, c0, c1, total, nontriv);
                fflush(stdout);
            }
        }
    }
    free(L);
    return 0;
}
