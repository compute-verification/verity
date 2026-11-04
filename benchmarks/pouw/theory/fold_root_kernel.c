/* The whole all-state kernel K_R of the same-word R-round XOR-in fold at width n <= 16, in 2^(2n) work a side.

   One round absorbs the word (d0, d1, d2, d3) into the n-bit state y with keys (a0, a1), as fold_kernel.c:
     G_{d0,d1}(y) = (rotl(y, r1) ^ d0) * a0 + d1,   H_{d2,d3}(y) = (rotl(y, r2) ^ d2) * a1 + d3,   F_w = H ∘ G.
   K_R(w) is every word w' with F_{w'}^R = F_w^R on all 2^n states. fold_kernel.c enumerates all 2^(4n) words; this
   uses one necessary condition instead. If Ψ^R = σ with σ = Φ^R (Φ = F_w, Ψ = F_{w'} = H'∘G'), then Ψ commutes with σ,
   which is G' σ G'^-1 = H'^-1 σ H'. The left side reads only (d0', d1') and the right only (d2', d3'), so a meet in the
   middle over the 2^(2n) half-round maps of each side, matched on m probe states, finds every candidate; each one is
   then checked on every state. G_{x,c} = G_{x^t,c+t} with t = 2^(n-1) (P0), and H likewise (P1), so each side runs over
   x < t and every match stands for four words.

   For each job line "a0 a1 w0 w1 w2 w3" on stdin this prints
     S a0 a1 w0 w1 w2 w3 |K_1| .. |K_rmax| pxn g3bad degenerate
   (|K_R| counts w itself; pxn = 1 iff K_1 is exactly w's orbit under the P/X/N group; g3bad counts words in the
   union of the K_R that break (G3) against w; degenerate = 1 if some join passed `cap` candidates and was cut short),
   and one line "E a0 a1 w0 w1 w2 w3 R d0 d1 d2 d3" for each word of K_R outside K_1 (R >= 2) or of K_1 outside the
   orbit (R = 1). With -a it also prints "K a0 a1 R d0 d1 d2 d3" for every w' != w in K_R, as fold_kernel.c does.
   -p sets the number of probe states (default 3, at most 64 / n) and -c the cap as log2 (default 16): more probes
   and a higher cap settle a cut join at a structured Φ^R.

   usage: fold_root_kernel [-a] [-p probes] [-c log2cap] n r1 r2 rmax < jobs
   build: gcc -O2 -fopenmp -o fold_root_kernel fold_root_kernel.c */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RMAX 8
#define MAXPROBES 21
#define MAXPROBE 256

typedef struct { uint32_t a0, a1, w[4]; } job_t;
typedef struct { uint32_t d[4]; } word_t;
typedef struct { uint64_t fp; uint32_t idx; } fpidx_t;

static int fpidx_cmp(const void *p, const void *q) {
    const fpidx_t *a = p, *b = q;
    return a->fp < b->fp ? -1 : a->fp > b->fp ? 1 : (a->idx > b->idx) - (a->idx < b->idx);
}

static int n, r1, r2, rmax, all_out, nprobe = 3;
static uint32_t mask, top, ns;

static inline uint32_t rotl(uint32_t x, int r) { return ((x << r) | (x >> (n - r))) & mask; }
static inline uint32_t rotr(uint32_t x, int r) { return ((x >> r) | (x << (n - r))) & mask; }
static inline uint32_t half(uint32_t y, int r, uint32_t x, uint32_t c, uint32_t a) { return ((rotl(y, r) ^ x) * a + c) & mask; }
static inline uint32_t half_inv(uint32_t z, int r, uint32_t x, uint32_t c, uint32_t ainv) {
    return rotr((((z - c) * ainv) & mask) ^ x, r);
}
static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; return x ^ (x >> 33);
}
static uint32_t inv2(uint32_t a) {  /* a^-1 mod 2^32 for odd a, by Newton */
    uint32_t x = a;
    for (int i = 0; i < 5; i++) x *= 2 - a * x;
    return x;
}
static int word_cmp(const void *p, const void *q) { return memcmp(p, q, sizeof(word_t)); }
static int word_eq(const word_t *p, const word_t *q) { return memcmp(p, q, sizeof(word_t)) == 0; }

/* The P/X/N group at width n: P0, P1, X and N generate it; its orbit of w (16 words when they are distinct). */
static int pxn_orbit(const job_t *j, word_t *out) {
    int cnt = 1;
    out[0] = (word_t){{j->w[0], j->w[1], j->w[2], j->w[3]}};
    for (int g = 0; g < 4; g++) {
        int base = cnt;
        for (int i = 0; i < base; i++) {
            word_t v = out[i];
            if (g == 0) { v.d[0] ^= top; v.d[1] = (v.d[1] + top) & mask; }
            if (g == 1) { v.d[2] ^= top; v.d[3] = (v.d[3] + top) & mask; }
            if (g == 2) { v.d[1] = (v.d[1] + top) & mask; v.d[2] ^= rotl(top, r2); }
            if (g == 3) { v.d[0] ^= mask; v.d[1] = (j->a0 - 1 - v.d[1]) & mask; v.d[2] ^= mask; }
            int seen = 0;
            for (int k = 0; k < cnt; k++) seen |= word_eq(&out[k], &v);
            if (!seen) out[cnt++] = v;
        }
    }
    qsort(out, cnt, sizeof(word_t), word_cmp);
    return cnt;
}

static int g3_ok(const job_t *j, const word_t *v) {
    uint32_t x0 = (j->w[0] ^ v->d[0]) & 7, x2 = (j->w[2] ^ v->d[2]) & 7;
    return (x0 == 0 || x0 == 7) && (x2 == 0 || x2 == 7) && !((j->w[1] ^ v->d[1]) & 1) && !((j->w[3] ^ v->d[3]) & 1);
}

typedef struct {
    uint32_t *sig, *phi, *slot;
    uint64_t *fpa;
    word_t *found;
    size_t nfound, capfound;
} scratch_t;

static void push(scratch_t *s, word_t v) {
    if (s->nfound == s->capfound) {
        s->capfound = s->capfound ? 2 * s->capfound : 64;
        s->found = realloc(s->found, s->capfound * sizeof(word_t));
    }
    s->found[s->nfound++] = v;
}

/* K_R for one job and one R, into s->found (sorted, w included); returns 1 if the join was cut at `cap`. */
static int kernel_r(const job_t *j, int R, scratch_t *s, size_t cap) {
    uint32_t a0 = j->a0, a1 = j->a1, a0i = inv2(a0) & mask, a1i = inv2(a1) & mask;
    uint32_t *sig = s->sig;
    for (uint32_t y = 0; y < ns; y++) {
        uint32_t t = y;
        for (int q = 0; q < R; q++) t = s->phi[t];
        sig[y] = t;
    }
    uint32_t probe[MAXPROBES];
    for (int k = 0; k < nprobe; k++) probe[k] = (uint32_t)(mix64(0x9E3779B97F4A7C15ULL * (k + 1)) & mask);
    size_t half_n = (size_t)(ns / 2) * ns, hsize = (size_t)ns * ns, hmask = hsize - 1;
    memset(s->slot, 0, hsize * sizeof(uint32_t));
    int hashed = 1;
    for (size_t idx = 0; idx < half_n; idx++) {
        uint32_t x = (uint32_t)(idx >> n), c = (uint32_t)(idx & mask);
        uint64_t fp = 0;
        for (int k = 0; k < nprobe; k++)
            fp |= (uint64_t)half(sig[half_inv(probe[k], r1, x, c, a0i)], r1, x, c, a0) << (k * n);
        s->fpa[idx] = fp;
        if (!hashed) continue;
        size_t h = mix64(fp) & hmask, len = 0;
        while (s->slot[h] && ++len < MAXPROBE) h = (h + 1) & hmask;
        if (len >= MAXPROBE) hashed = 0;
        else s->slot[h] = (uint32_t)idx + 1;
    }
    /* A very structured σ (the identity, at a key where the round is a rotation) gives many equal fingerprints, and
       linear probing goes quadratic; the sorted join has no such case. */
    fpidx_t *sorted = NULL;
    if (!hashed) {
        sorted = malloc(half_n * sizeof(fpidx_t));
        for (size_t idx = 0; idx < half_n; idx++) sorted[idx] = (fpidx_t){s->fpa[idx], (uint32_t)idx};
        qsort(sorted, half_n, sizeof(fpidx_t), fpidx_cmp);
    }
    s->nfound = 0;
    size_t cand = 0;
    int cut = 0;
    for (size_t idx = 0; idx < half_n && !cut; idx++) {
        uint32_t x = (uint32_t)(idx >> n), c = (uint32_t)(idx & mask);
        uint64_t fp = 0;
        for (int k = 0; k < nprobe; k++)
            fp |= (uint64_t)half_inv(sig[half(probe[k], r2, x, c, a1)], r2, x, c, a1i) << (k * n);
        size_t h = 0, lo = 0;
        if (hashed) {
            h = mix64(fp) & hmask;
        } else {
            size_t hi = half_n;
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if (sorted[mid].fp < fp) lo = mid + 1; else hi = mid;
            }
        }
        for (;;) {
            uint32_t ia;
            if (hashed) {
                if (!s->slot[h]) break;
                ia = s->slot[h] - 1;
                h = (h + 1) & hmask;
                if (s->fpa[ia] != fp) continue;
            } else {
                if (lo >= half_n || sorted[lo].fp != fp) break;
                ia = sorted[lo++].idx;
            }
            if (++cand > cap) { cut = 1; break; }
            uint32_t xa = ia >> n, ca = ia & mask;
            int ok = 1;
            for (uint32_t y = 0; y < ns && ok; y++) {
                uint32_t t = y;
                for (int q = 0; q < R; q++) t = half(half(t, r1, xa, ca, a0), r2, x, c, a1);
                ok = (t == sig[y]);
            }
            if (!ok) continue;
            for (int p = 0; p < 4; p++) {
                word_t v = {{xa, ca, x, c}};
                if (p & 1) { v.d[0] ^= top; v.d[1] = (v.d[1] + top) & mask; }
                if (p & 2) { v.d[2] ^= top; v.d[3] = (v.d[3] + top) & mask; }
                push(s, v);
            }
        }
    }
    free(sorted);
    qsort(s->found, s->nfound, sizeof(word_t), word_cmp);
    return cut;
}

static int contains(const word_t *set, size_t cnt, const word_t *v) {
    return bsearch(v, set, cnt, sizeof(word_t), word_cmp) != NULL;
}

int main(int argc, char **argv) {
    int ai = 1, log2cap = 16;
    for (; ai < argc && argv[ai][0] == '-'; ai++) {
        if (!strcmp(argv[ai], "-a")) all_out = 1;
        else if (!strcmp(argv[ai], "-p") && ai + 1 < argc) nprobe = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "-c") && ai + 1 < argc) log2cap = atoi(argv[++ai]);
        else break;
    }
    if (argc - ai != 4) {
        fprintf(stderr, "usage: fold_root_kernel [-a] [-p probes] [-c log2cap] n r1 r2 rmax < jobs\n");
        return 2;
    }
    n = atoi(argv[ai]); r1 = atoi(argv[ai + 1]); r2 = atoi(argv[ai + 2]); rmax = atoi(argv[ai + 3]);
    if (n < 3 || n > 16 || r1 < 1 || r1 >= n || r2 < 1 || r2 >= n || rmax < 1 || rmax > RMAX) {
        fprintf(stderr, "need 3 <= n <= 16, 0 < r1, r2 < n, 1 <= rmax <= %d\n", RMAX);
        return 2;
    }
    if (nprobe < 1 || nprobe > MAXPROBES || nprobe * n > 64 || log2cap < 0 || log2cap > 40) {
        fprintf(stderr, "need 1 <= probes <= 64 / n and 0 <= log2cap <= 40\n");
        return 2;
    }
    mask = (1u << n) - 1; top = 1u << (n - 1); ns = 1u << n;
    size_t cap = (size_t)1 << log2cap;
    size_t njobs = 0, capjobs = 1024;
    job_t *jobs = malloc(capjobs * sizeof(job_t));
    unsigned long v[6];
    while (scanf("%lu %lu %lu %lu %lu %lu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) {
        if (njobs == capjobs) jobs = realloc(jobs, (capjobs *= 2) * sizeof(job_t));
        jobs[njobs++] = (job_t){(uint32_t)v[0] & mask, (uint32_t)v[1] & mask,
                                {(uint32_t)v[2] & mask, (uint32_t)v[3] & mask, (uint32_t)v[4] & mask, (uint32_t)v[5] & mask}};
    }
    for (size_t i = 0; i < njobs; i++)
        if (!(jobs[i].a0 & 1) || !(jobs[i].a1 & 1)) { fprintf(stderr, "keys must be odd\n"); return 2; }
#pragma omp parallel
    {
        scratch_t s = {0};
        s.sig = malloc(ns * sizeof(uint32_t));
        s.phi = malloc(ns * sizeof(uint32_t));
        s.slot = malloc((size_t)ns * ns * sizeof(uint32_t));
        s.fpa = malloc((size_t)(ns / 2) * ns * sizeof(uint64_t));
        word_t *k1 = NULL, *uni = NULL;
        size_t k1n = 0, unin = 0, unicap = 0;
#pragma omp for schedule(dynamic, 1)
        for (size_t ji = 0; ji < njobs; ji++) {
            const job_t *j = &jobs[ji];
            for (uint32_t y = 0; y < ns; y++)
                s.phi[y] = half(half(y, r1, j->w[0], j->w[1], j->a0), r2, j->w[2], j->w[3], j->a1);
            word_t orbit[16];
            int on = pxn_orbit(j, orbit);
            word_t self = {{j->w[0], j->w[1], j->w[2], j->w[3]}};
            size_t counts[RMAX + 1];
            int degenerate = 0, pxn = 0;
            unin = 0;
            char *buf = NULL;
            size_t blen = 0;
            FILE *mem = open_memstream(&buf, &blen);
            for (int R = 1; R <= rmax; R++) {
                degenerate |= kernel_r(j, R, &s, cap);
                counts[R] = s.nfound;
                if (R == 1) {
                    k1n = s.nfound;
                    k1 = realloc(k1, (k1n ? k1n : 1) * sizeof(word_t));
                    memcpy(k1, s.found, k1n * sizeof(word_t));
                    pxn = (k1n == (size_t)on) && !memcmp(k1, orbit, on * sizeof(word_t));
                }
                for (size_t i = 0; i < s.nfound; i++) {
                    const word_t *e = &s.found[i];
                    int outside = R == 1 ? !contains(orbit, on, e) : !contains(k1, k1n, e);
                    if (outside)
                        fprintf(mem, "E %u %u %u %u %u %u %d %u %u %u %u\n", j->a0, j->a1, j->w[0], j->w[1], j->w[2],
                                j->w[3], R, e->d[0], e->d[1], e->d[2], e->d[3]);
                    if (all_out && !word_eq(e, &self))
                        fprintf(mem, "K %u %u %d %u %u %u %u\n", j->a0, j->a1, R, e->d[0], e->d[1], e->d[2], e->d[3]);
                    if (unin == unicap) uni = realloc(uni, (unicap = unicap ? 2 * unicap : 64) * sizeof(word_t));
                    uni[unin++] = *e;
                }
            }
            qsort(uni, unin, sizeof(word_t), word_cmp);
            size_t g3bad = 0;
            for (size_t i = 0; i < unin; i++)
                if ((i == 0 || !word_eq(&uni[i], &uni[i - 1])) && !g3_ok(j, &uni[i])) g3bad++;
            fprintf(mem, "S %u %u %u %u %u %u", j->a0, j->a1, j->w[0], j->w[1], j->w[2], j->w[3]);
            for (int R = 1; R <= rmax; R++) fprintf(mem, " %zu", counts[R]);
            fprintf(mem, " %d %zu %d\n", pxn, g3bad, degenerate);
            fclose(mem);
#pragma omp critical
            {
                fwrite(buf, 1, blen, stdout);
                fflush(stdout);
            }
            free(buf);
        }
        free(s.sig); free(s.phi); free(s.slot); free(s.fpa); free(s.found); free(k1); free(uni);
    }
    free(jobs);
    return 0;
}
