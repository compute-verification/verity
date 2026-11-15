/* Every all-state kernel element of the same-word R-round XOR-in fold at width n <= 8.

   One round absorbs the word (d0, d1, d2, d3) into the n-bit state y with keys (a0, a1), as
   receipt_folds_r9.xorin_step at reduced width:
     y = (rotl(y, r1) ^ d0) * a0 + d1
     y = (rotl(y, r2) ^ d2) * a1 + d3
   The R-fold applies that round R times with the same word and the same keys (egemm.cu's EG_ROUNDS).
   For each key pair read from stdin ("a0 a1" a line) this prints every word w' != w, over all 2^(4n) of them, whose
   R-fold equals w's on every one of the 2^n states, for each R in 1..rmax:
     K a0 a1 R d0' d1' d2' d3'
   and one summary line per key: "S a0 a1 count_R1 ... count_Rrmax".

   usage: fold_kernel n r1 r2 rmax w0 w1 w2 w3 < keys
   build: cc -O2 -fopenmp -o fold_kernel fold_kernel.c */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RMAX 8

static int n, r1, r2, rmax;
static uint32_t mask, ns;

static inline uint32_t rotl(uint32_t x, int r) { return ((x << r) | (x >> (n - r))) & mask; }

int main(int argc, char **argv) {
    if (argc != 9) { fprintf(stderr, "usage: fold_kernel n r1 r2 rmax w0 w1 w2 w3 < keys\n"); return 2; }
    n = atoi(argv[1]); r1 = atoi(argv[2]); r2 = atoi(argv[3]); rmax = atoi(argv[4]);
    uint32_t w[4];
    for (int i = 0; i < 4; i++) w[i] = (uint32_t)strtoul(argv[5 + i], 0, 0);
    if (n < 2 || n > 8 || rmax < 1 || rmax > RMAX) { fprintf(stderr, "need 2 <= n <= 8, 1 <= rmax <= 8\n"); return 2; }
    mask = (1u << n) - 1; ns = 1u << n;
    uint32_t pairs = 1u << (2 * n);
    uint8_t *G = malloc((size_t)pairs * ns), *H = malloc((size_t)pairs * ns);
    uint8_t (*base)[256] = malloc(sizeof(uint8_t[256]) * (RMAX + 1));
    unsigned long a0, a1;
    while (scanf("%lu %lu", &a0, &a1) == 2) {
        a0 &= mask; a1 &= mask;
        for (uint32_t p = 0; p < pairs; p++) {
            uint32_t x = p & mask, c = p >> n;
            for (uint32_t y = 0; y < ns; y++) {
                G[(size_t)p * ns + y] = (uint8_t)((((rotl(y, r1) ^ x) * a0) + c) & mask);
                H[(size_t)p * ns + y] = (uint8_t)((((rotl(y, r2) ^ x) * a1) + c) & mask);
            }
        }
        uint32_t pw01 = w[0] | (w[1] << n), pw23 = w[2] | (w[3] << n);
        for (uint32_t y = 0; y < ns; y++) {
            uint32_t s = y;
            for (int r = 1; r <= rmax; r++) {
                s = H[(size_t)pw23 * ns + G[(size_t)pw01 * ns + s]];
                base[r][y] = (uint8_t)s;
            }
        }
        unsigned long counts[RMAX + 1];
        memset(counts, 0, sizeof counts);
#pragma omp parallel for schedule(dynamic, 16)
        for (uint32_t p01 = 0; p01 < pairs; p01++) {
            const uint8_t *g = G + (size_t)p01 * ns;
            unsigned long local[RMAX + 1];
            memset(local, 0, sizeof local);
            for (uint32_t p23 = 0; p23 < pairs; p23++) {
                if (p01 == pw01 && p23 == pw23) continue;
                const uint8_t *h = H + (size_t)p23 * ns;
                uint32_t s = 0, hit = 0;
                for (int r = 1; r <= rmax; r++) {
                    s = h[g[s]];
                    if (s == base[r][0]) hit |= 1u << r;
                }
                if (!hit) continue;
                for (int r = 1; r <= rmax; r++) {
                    if (!(hit >> r & 1)) continue;
                    int ok = 1;
                    for (uint32_t y = 1; y < ns && ok; y++) {
                        uint32_t t = y;
                        for (int q = 0; q < r; q++) t = h[g[t]];
                        ok = (t == base[r][y]);
                    }
                    if (ok) {
                        local[r]++;
#pragma omp critical
                        printf("K %lu %lu %d %u %u %u %u\n", a0, a1, r, p01 & mask, p01 >> n, p23 & mask, p23 >> n);
                    }
                }
            }
#pragma omp critical
            for (int r = 1; r <= rmax; r++) counts[r] += local[r];
        }
        printf("S %lu %lu", a0, a1);
        for (int r = 1; r <= rmax; r++) printf(" %lu", counts[r]);
        printf("\n");
        fflush(stdout);
    }
    free(G); free(H); free(base);
    return 0;
}
