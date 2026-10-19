/* FOLD-OPT's two-word residual (fold_opt_fx.py, RT-MACHINE-8): does any 3-instruction straight-line program whose last
   instruction is an IMAD.WIDE or IMAD.WIDE.U32 write a target pair (t1, t2) as that instruction's two words, either way
   round? Exhaustive at width W over the ISA fold_opt_fx.py models: the one-word forms of `_single_values` (every LOP3
   LUT, IADD3 at every sign, IMAD, IMAD.HI, LEA and LEA.HI at every shift) and IMAD.WIDE's two words, signed and not, over
   a two-word addend (c_hi:c_lo).

   The first instruction ranges over the inputs, the second over the inputs and the first's words, each deduplicated by
   the words it adds (a program whose instruction adds no new word is a shorter program's). The last ranges over every
   p, q of the inputs and both instructions' words, its low word fixing c_lo and its high word c_hi; each must be one of
   those words. Pairs p, q of inputs and first-instruction words are solved once a first instruction.

   A value is the vector of a word at NP points, W bits a point, packed W*NP <= 64 bits. stdin:
     W NP NI
     name v_1 .. v_NP              (NI lines: the inputs)
     NJ
     label t1_1 .. t1_NP t2_1 .. t2_NP      (NJ lines: the target pairs)
   stdout, a line a job: "J label found N firsts F prefixes P", then up to 4 witnesses "F label I1 ; I2 ; I3".
   build: gcc -O2 -fopenmp -o fold_wide3 fold_wide3.c */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 24
#define MAXJ 8
#define MAXW 4

static int W, NP, NI, NJ;
static uint64_t MASK, ALL;
static uint64_t in_v[MAXN];
static char in_name[MAXN][16];
static uint64_t t1[MAXJ], t2[MAXJ];
static char jlabel[MAXJ][32];

static inline uint64_t at(uint64_t v, int i) { return (v >> (W * i)) & MASK; }

/* An instruction: its form, operands (indices into the pool it read) and the words it writes. */
enum { LOP3, IADD3, IMAD, IMADHI, LEA, LEAHI, WIDE };
typedef struct {
    uint8_t form, a, b, c, d, imm, nw;
    uint64_t w[2];
} Ins;

static void describe(const Ins *x, char names[][16], char *out, size_t n) {
    static const char *f[] = {"LOP3", "IADD3", "IMAD", "IMAD.HI", "LEA", "LEA.HI", "IMAD.WIDE"};
    if (x->form == WIDE)
        snprintf(out, n, "%s%s(%s,%s,%s:%s)", f[x->form], x->imm ? "" : ".U32", names[x->a], names[x->b], names[x->d],
                 names[x->c]);
    else if (x->form == LEA || x->form == LEAHI)
        snprintf(out, n, "%s.%d(%s,%s)", f[x->form], x->imm, names[x->a], names[x->b]);
    else
        snprintf(out, n, "%s.%#x(%s,%s,%s)", f[x->form], x->imm, names[x->a], names[x->b], names[x->c]);
}

/* (lo, hi) of p*q + (ch:cl) at one point, signed or not */
static inline void wide1(uint64_t p, uint64_t q, uint64_t cl, uint64_t ch, int sgn, uint64_t *lo, uint64_t *hi) {
    int64_t ps = (int64_t)p, qs = (int64_t)q;
    if (sgn) {
        if (p >> (W - 1)) ps -= (int64_t)1 << W;
        if (q >> (W - 1)) qs -= (int64_t)1 << W;
    }
    uint64_t v = ((uint64_t)(ps * qs) + (ch << W) + cl) & ((1ULL << (2 * W)) - 1);
    *lo = v & MASK;
    *hi = v >> W;
}

/* Every instruction over pool[0..n): calls emit(ins) for each. */
typedef void (*Emit)(const Ins *, void *);
static void each_instruction(const uint64_t *pool, int n, Emit emit, void *ctx) {
    Ins x;
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++)
            for (int c = 0; c < n; c++) {
                uint64_t A = pool[a], B = pool[b], C = pool[c], m[8];
                for (int k = 0; k < 8; k++)
                    m[k] = ((k & 1) ? A : ~A) & ((k & 2) ? B : ~B) & ((k & 4) ? C : ~C) & ALL;
                uint64_t v[256];
                v[0] = 0;
                x.a = a; x.b = b; x.c = c; x.d = 0; x.nw = 1;
                for (int lut = 0; lut < 256; lut++) {
                    if (lut) v[lut] = v[lut & (lut - 1)] | m[__builtin_ctz(lut)];
                    x.form = LOP3; x.imm = lut; x.w[0] = v[lut];
                    emit(&x, ctx);
                }
                for (int s = 0; s < 8; s++) {
                    uint64_t r = 0;
                    for (int i = 0; i < NP; i++) {
                        int64_t t = ((s & 1) ? -1 : 1) * (int64_t)at(A, i) + ((s & 2) ? -1 : 1) * (int64_t)at(B, i) +
                                    ((s & 4) ? -1 : 1) * (int64_t)at(C, i);
                        r |= ((uint64_t)t & MASK) << (W * i);
                    }
                    x.form = IADD3; x.imm = s; x.w[0] = r;
                    emit(&x, ctx);
                }
                uint64_t r1 = 0, r2 = 0;
                for (int i = 0; i < NP; i++) {
                    uint64_t pr = at(A, i) * at(B, i);
                    r1 |= ((pr + at(C, i)) & MASK) << (W * i);
                    r2 |= (((pr >> W) + at(C, i)) & MASK) << (W * i);
                }
                x.form = IMAD; x.imm = 0; x.w[0] = r1;
                emit(&x, ctx);
                x.form = IMADHI; x.w[0] = r2;
                emit(&x, ctx);
            }
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++)
            for (int s = 0; s < W; s++) {
                uint64_t r1 = 0, r2 = 0;
                for (int i = 0; i < NP; i++) {
                    uint64_t A = at(pool[a], i), B = at(pool[b], i);
                    uint64_t rot = s ? (((A << s) | (A >> (W - s))) & MASK) : A;
                    r1 |= (((A << s) + B) & MASK) << (W * i);
                    r2 |= ((rot + B) & MASK) << (W * i);
                }
                x.a = a; x.b = b; x.c = 0; x.d = 0; x.nw = 1; x.imm = s;
                x.form = LEA; x.w[0] = r1;
                emit(&x, ctx);
                x.form = LEAHI; x.w[0] = r2;
                emit(&x, ctx);
            }
    for (int p = 0; p < n; p++)
        for (int q = 0; q < n; q++)
            for (int cl = 0; cl < n; cl++)
                for (int ch = 0; ch < n; ch++)
                    for (int sg = 0; sg < 2; sg++) {
                        uint64_t lo = 0, hi = 0, l, h;
                        for (int i = 0; i < NP; i++) {
                            wide1(at(pool[p], i), at(pool[q], i), at(pool[cl], i), at(pool[ch], i), sg, &l, &h);
                            lo |= l << (W * i);
                            hi |= h << (W * i);
                        }
                        x.form = WIDE; x.a = p; x.b = q; x.c = cl; x.d = ch; x.imm = sg; x.nw = 2;
                        x.w[0] = lo; x.w[1] = hi;
                        emit(&x, ctx);
                    }
}

/* An open-addressing set of the words an instruction adds, cleared by generation. */
typedef struct {
    uint64_t *k0, *k1;
    uint32_t *gen;
    uint32_t g, cap;
} Set;

static Set set_new(uint32_t cap) {
    Set s = {calloc(cap, 8), calloc(cap, 8), calloc(cap, 4), 1, cap};
    return s;
}

static int set_add(Set *s, uint64_t a, uint64_t b) {
    uint64_t h = (a * 0x9E3779B97F4A7C15ULL) ^ (b * 0xC2B2AE3D27D4EB4FULL);
    uint32_t i = (uint32_t)(h >> 40) & (s->cap - 1);
    while (s->gen[i] == s->g) {
        if (s->k0[i] == a && s->k1[i] == b) return 0;
        i = (i + 1) & (s->cap - 1);
    }
    s->gen[i] = s->g; s->k0[i] = a; s->k1[i] = b;
    return 1;
}

static int in_pool(const uint64_t *pool, int n, uint64_t v) {
    for (int i = 0; i < n; i++)
        if (pool[i] == v) return 1;
    return 0;
}

/* A first instruction's new words (one or two) as a canonical key, or 0 when it adds none. */
static int new_words(const Ins *x, const uint64_t *pool, int n, uint64_t *k0, uint64_t *k1) {
    int fresh0 = !in_pool(pool, n, x->w[0]);
    int fresh1 = x->nw == 2 && !in_pool(pool, n, x->w[1]) && x->w[1] != x->w[0];
    if (!fresh0 && !fresh1) return 0;
    if (fresh0 && fresh1) {
        *k0 = x->w[0] < x->w[1] ? x->w[0] : x->w[1];
        *k1 = x->w[0] < x->w[1] ? x->w[1] : x->w[0];
        return 2;
    }
    *k0 = *k1 = fresh0 ? x->w[0] : x->w[1];
    return 1;
}

typedef struct {
    Ins *v;
    size_t n, cap;
    Set set;
    const uint64_t *pool;
    int np;
} List;

static void collect(const Ins *x, void *ctx) {
    List *L = ctx;
    uint64_t k0, k1;
    int nw = new_words(x, L->pool, L->np, &k0, &k1);
    if (!nw || !set_add(&L->set, k0, k1)) return;
    if (L->n == L->cap) {
        L->cap = L->cap ? 2 * L->cap : 4096;
        L->v = realloc(L->v, L->cap * sizeof(Ins));
    }
    Ins y = *x;
    y.nw = nw;
    y.w[0] = k0; y.w[1] = k1;
    L->v[L->n++] = y;
}

/* The last instruction's operands p, q fixed: does some c_lo, c_hi of pool[0..n) make its words (L, H)? */
static int solve(uint64_t P, uint64_t Q, int sg, uint64_t L, uint64_t H, const uint64_t *pool, int n, int *cl, int *ch) {
    uint64_t need = 0, plo[16], phi[16];
    for (int i = 0; i < NP; i++) {
        wide1(at(P, i), at(Q, i), 0, 0, sg, &plo[i], &phi[i]);
        need |= ((at(L, i) - plo[i]) & MASK) << (W * i);
    }
    int c = -1;
    for (int k = 0; k < n; k++)
        if (pool[k] == need) { c = k; break; }
    if (c < 0) return 0;
    uint64_t needh = 0;
    for (int i = 0; i < NP; i++)
        needh |= ((at(H, i) - phi[i] - ((plo[i] + at(pool[c], i)) >> W)) & MASK) << (W * i);
    for (int k = 0; k < n; k++)
        if (pool[k] == needh) { *cl = c; *ch = k; return 1; }
    return 0;
}

static void report(int j, const Ins *x1, const Ins *x2, int n1, int p, int q, int sg, int o, int cl, int ch) {
    char names[MAXN][16], d1[96], d2[96] = "none";
    for (int k = 0; k < NI; k++) strcpy(names[k], in_name[k]);
    strcpy(names[NI], "v1");
    if (x1->nw == 2) strcpy(names[NI + 1], "v1b");
    describe(x1, names, d1, sizeof d1);
    if (x2) {
        strcpy(names[n1], "v2");
        if (x2->nw == 2) strcpy(names[n1 + 1], "v2b");
        describe(x2, names, d2, sizeof d2);
    }
#pragma omp critical
    printf("F %s %s ; %s ; IMAD.WIDE%s(%s,%s,%s:%s) -> (%s)\n", jlabel[j], d1, d2, sg ? "" : ".U32", names[p],
           names[q], names[ch], names[cl], o ? "t2,t1" : "t1,t2");
}

int main(void) {
    if (scanf("%d %d %d", &W, &NP, &NI) != 3 || W < 2 || W * NP > 64 || NP > 16 || NI + 2 * MAXW > MAXN) {
        fprintf(stderr, "bad header\n");
        return 2;
    }
    MASK = (1ULL << W) - 1;
    ALL = W * NP == 64 ? ~0ULL : (1ULL << (W * NP)) - 1;
    for (int k = 0; k < NI; k++) {
        if (scanf("%15s", in_name[k]) != 1) return 2;
        in_v[k] = 0;
        for (int i = 0; i < NP; i++) {
            unsigned long long x;
            if (scanf("%llu", &x) != 1) return 2;
            in_v[k] |= ((uint64_t)x & MASK) << (W * i);
        }
    }
    if (scanf("%d", &NJ) != 1 || NJ < 1 || NJ > MAXJ) return 2;
    for (int j = 0; j < NJ; j++) {
        if (scanf("%31s", jlabel[j]) != 1) return 2;
        t1[j] = t2[j] = 0;
        for (int i = 0; i < NP; i++) {
            unsigned long long x;
            if (scanf("%llu", &x) != 1) return 2;
            t1[j] |= ((uint64_t)x & MASK) << (W * i);
        }
        for (int i = 0; i < NP; i++) {
            unsigned long long x;
            if (scanf("%llu", &x) != 1) return 2;
            t2[j] |= ((uint64_t)x & MASK) << (W * i);
        }
    }

    List firsts = {0};
    firsts.set = set_new(1u << 20);
    firsts.pool = in_v;
    firsts.np = NI;
    each_instruction(in_v, NI, collect, &firsts);

    long found[MAXJ] = {0};
    long prefixes = 0;
    int shown[MAXJ] = {0};
#pragma omp parallel reduction(+ : prefixes)
    {
        List seconds = {0};
        seconds.set = set_new(1u << 21);
        int npre = 0, cap = 4 * MAXJ * MAXN * MAXN;
        uint64_t *pre_lo = malloc(cap * 8), *pre_hi = malloc(cap * 8);
        uint8_t *pre_in = malloc(cap), (*pre_m)[5] = malloc(cap * 5);
#pragma omp for schedule(dynamic, 4)
        for (size_t f = 0; f < firsts.n; f++) {
            const Ins *x1 = &firsts.v[f];
            uint64_t pool[MAXN];
            memcpy(pool, in_v, NI * 8);
            int n1 = NI;
            pool[n1++] = x1->w[0];
            if (x1->nw == 2) pool[n1++] = x1->w[1];
            /* p, q of the inputs and the first's words: the c_lo and c_hi the last instruction needs, once */
            npre = 0;
            for (int p = 0; p < n1; p++)
                for (int q = p; q < n1; q++)
                    for (int sg = 0; sg < 2; sg++)
                        for (int j = 0; j < NJ; j++)
                            for (int o = 0; o < 2; o++) {
                                uint64_t L = o ? t2[j] : t1[j], H = o ? t1[j] : t2[j], need = 0, needh = 0, l, h;
                                for (int i = 0; i < NP; i++) {
                                    wide1(at(pool[p], i), at(pool[q], i), 0, 0, sg, &l, &h);
                                    uint64_t c = (at(L, i) - l) & MASK;
                                    need |= c << (W * i);
                                    needh |= ((at(H, i) - h - ((l + c) >> W)) & MASK) << (W * i);
                                }
                                pre_lo[npre] = need; pre_hi[npre] = needh;
                                pre_in[npre] = in_pool(pool, n1, need) | in_pool(pool, n1, needh) << 1;
                                pre_m[npre][0] = p; pre_m[npre][1] = q; pre_m[npre][2] = sg; pre_m[npre][3] = j;
                                pre_m[npre][4] = o;
                                npre++;
                            }
            seconds.n = 0;
            seconds.set.g++;
            seconds.pool = pool;
            seconds.np = n1;
            each_instruction(pool, n1, collect, &seconds);
            /* a pair already in the first's pool is a program of at most two instructions: reported with I2 none */
            for (size_t s = 0; s <= seconds.n; s++) {
                const Ins *x2 = s < seconds.n ? &seconds.v[s] : NULL;
                int n2 = n1;
                if (x2) {
                    pool[n2++] = x2->w[0];
                    if (x2->nw == 2) pool[n2++] = x2->w[1];
                    prefixes++;
                }
                int hits[4 * MAXJ * MAXN * MAXN], nh = 0;
                for (int e = 0; e < npre; e++) {
                    int ok_lo = (pre_in[e] & 1) || (x2 && (pre_lo[e] == pool[n1] || pre_lo[e] == pool[n2 - 1]));
                    int ok_hi = (pre_in[e] & 2) || (x2 && (pre_hi[e] == pool[n1] || pre_hi[e] == pool[n2 - 1]));
                    if (ok_lo && ok_hi && (x2 ? (pre_in[e] & 3) != 3 : 1)) hits[nh++] = e;
                }
                for (int h = 0; h < nh; h++) {
                    int e = hits[h], j = pre_m[e][3], cl = -1, ch = -1;
                    for (int k = 0; k < n2; k++) {
                        if (cl < 0 && pool[k] == pre_lo[e]) cl = k;
                        if (ch < 0 && pool[k] == pre_hi[e]) ch = k;
                    }
#pragma omp atomic
                    found[j]++;
                    int show;
#pragma omp atomic capture
                    show = shown[j]++;
                    if (show < 4) report(j, x1, x2, n1, pre_m[e][0], pre_m[e][1], pre_m[e][2], pre_m[e][4], cl, ch);
                }
                if (!x2) continue;
                /* p or q a word of the second instruction */
                for (int q = n1; q < n2; q++)
                    for (int p = 0; p <= q; p++)
                        for (int sg = 0; sg < 2; sg++)
                            for (int j = 0; j < NJ; j++)
                                for (int o = 0; o < 2; o++) {
                                    int cl, ch;
                                    if (!solve(pool[p], pool[q], sg, o ? t2[j] : t1[j], o ? t1[j] : t2[j], pool, n2,
                                               &cl, &ch))
                                        continue;
#pragma omp atomic
                                    found[j]++;
                                    int show;
#pragma omp atomic capture
                                    show = shown[j]++;
                                    if (show < 4) report(j, x1, x2, n1, p, q, sg, o, cl, ch);
                                }
            }
        }
    }
    for (int j = 0; j < NJ; j++)
        printf("J %s found %ld firsts %zu prefixes %ld\n", jlabel[j], found[j], firsts.n, prefixes);
    return 0;
}
