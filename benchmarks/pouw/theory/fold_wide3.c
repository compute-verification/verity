/* FOLD-OPT's two-word residual (fold_opt_fx.py). For a target pair (t1, t2) at width W,
   exhaustive over the ISA fold_opt_fx.py models together with every sm_120 lane form that writes two words:
     found  does a straight-line program of at most 3 instructions write the pair, either way round, as its last
            instruction's two words?
     sel    does one write it as a pointwise choice between two pairs of words it has (a last SEL.64 under any
            predicate, wherever the predicate came from; IMNMX.64 is the case counted in found)?
     le2    is t1 or t2 an input or a word of a program of at most 2 instructions?
   The one-word forms are `_single_values`'s (every LOP3 LUT, IADD3 at every sign, IMAD, IMAD.HI, LEA and LEA.HI at every
   shift). The two-word forms read two-word operands: IMAD.WIDE p*q + (c_hi:c_lo), signed and not; IADD.64
   ±(a_hi:a_lo) ± (b_hi:b_lo); IMNMX.64, the min or max of (a_hi:a_lo) and (b_hi:b_lo), signed and not. A predicate
   read anywhere but a last SEL.64, a carry chain (IADD3's carry out, IADD3.X) and a guarded instruction are outside it.

   The first instruction ranges over the inputs, the second over the inputs and the first's words, each deduplicated by
   the words it adds (a program whose instruction adds no new word is a shorter program's). The last instruction:
     IMAD.WIDE  p, q of the inputs and the first's words fix the c_lo and c_hi it needs, solved once a first instruction
                and looked up for each second; p or q a second-instruction word is solved for each second.
     IADD.64    its low word ±a_lo ± b_lo reads only the low operands, so b_lo is solved from a_lo, then b_hi from a_hi.
     IMNMX.64   each pair of words is reduced to the points where it equals the target; two pairs whose points cover
                them all are a choice (sel), and an IMNMX.64 when one of its four forms picks it (found).

   A value is the vector of a word at NP points, W bits a point, packed W*NP <= 64 bits, so W = 32 is the full word at two
   points (fold_wide32.py): a program that writes the pair everywhere writes it at the points, so none found is none at
   any NP. stdin:
     W NP NI
     name v_1 .. v_NP              (NI lines: the inputs)
     NJ
     label t1_1 .. t1_NP t2_1 .. t2_NP      (NJ lines: the target pairs)
   stdout, a line a job: "J label found N sel S le2 K firsts F prefixes P", after up to 4 witnesses of each count:
   "F label I1 ; I2 ; I3 -> (t1,t2)", "S label I1 ; I2 ; SEL.64(..) -> (t1,t2)" and "L label I1 ; I2".
   build: gcc -O2 -fopenmp -o fold_wide3 fold_wide3.c */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 24
#define MAXJ 16
#define MAXW 32
/* -DDROP=k (fold_opt_fx.py's mutations) removes one part of the search: 1 a last IADD.64; 2 a last IMNMX.64 or SEL.64
   over a second-instruction word; 3 a last IMAD.WIDE with p or q a second-instruction word; 4 a last IMAD.WIDE with p
   and q before it; 5 a second instruction's words in le2; 6 IADD.64 and IMNMX.64 as a first or second instruction. */
#ifndef DROP
#define DROP 0
#endif

static int W, NP, NI, NJ;
static uint64_t MASK, ALL, M2;
static uint32_t FULLP;
static uint64_t in_v[MAXN];
static char in_name[MAXN][16];
static uint64_t t1[MAXJ], t2[MAXJ];
static char jlabel[MAXJ][32];

static inline uint64_t at(uint64_t v, int i) { return (v >> (W * i)) & MASK; }
/* the two-word operand (hi:lo) at one point */
static inline uint64_t pair_at(uint64_t lo, uint64_t hi, int i) { return at(lo, i) | at(hi, i) << W; }

/* An instruction: its form, operands (indices into the pool it read) and the words it writes. A two-word operand form
   (IADD.64, IMNMX.64, SEL.64) reads (b:a) and (d:c); IMAD.WIDE reads p = a, q = b and (d:c). */
enum { LOP3, IADD3, IMAD, IMADHI, LEA, LEAHI, WIDE, ADD64, MNMX64, SEL64 };
typedef struct {
    uint8_t form, a, b, c, d, imm, nw;
    uint64_t w[2];
} Ins;

static void describe(const Ins *x, char names[][16], char *out, size_t n) {
    static const char *f[] = {"LOP3", "IADD3", "IMAD", "IMAD.HI", "LEA", "LEA.HI"};
    const char *a = names[x->a], *b = names[x->b], *c = names[x->c], *d = names[x->d];
    switch (x->form) {
    case WIDE:
        snprintf(out, n, "IMAD.WIDE%s(%s,%s,%s:%s)", x->imm ? "" : ".U32", a, b, d, c);
        break;
    case ADD64:
        snprintf(out, n, "IADD.64(%s%s:%s,%s%s:%s)", x->imm & 1 ? "-" : "", b, a, x->imm & 2 ? "-" : "", d, c);
        break;
    case MNMX64:
        snprintf(out, n, "IMNMX.%s64.%s(%s:%s,%s:%s)", x->imm & 2 ? "S" : "U", x->imm & 1 ? "MAX" : "MIN", b, a, d, c);
        break;
    case SEL64:
        snprintf(out, n, "SEL.64(%s:%s,%s:%s)", b, a, d, c);
        break;
    case LEA:
    case LEAHI:
        snprintf(out, n, "%s.%d(%s,%s)", f[x->form], x->imm, a, b);
        break;
    default:
        snprintf(out, n, "%s.%#x(%s,%s,%s)", f[x->form], x->imm, a, b, c);
    }
}

/* (lo, hi) of p*q + (ch:cl) at one point, signed or not */
static inline void wide1(uint64_t p, uint64_t q, uint64_t cl, uint64_t ch, int sgn, uint64_t *lo, uint64_t *hi) {
    int64_t ps = (int64_t)p, qs = (int64_t)q;
    if (sgn) {
        if (p >> (W - 1)) ps -= (int64_t)1 << W;
        if (q >> (W - 1)) qs -= (int64_t)1 << W;
    }
    uint64_t v = ((uint64_t)ps * (uint64_t)qs + (ch << W) + cl) & M2;
    *lo = v & MASK;
    *hi = v >> W;
}

/* IADD.64 at one point: bit 0 of s negates A, bit 1 negates B */
static inline uint64_t add64(uint64_t A, uint64_t B, int s) { return (((s & 1) ? 0 - A : A) + ((s & 2) ? 0 - B : B)) & M2; }

/* IMNMX.64 at one point: bit 0 of v takes the max, bit 1 compares signed */
static inline uint64_t mnmx64(uint64_t A, uint64_t B, int v) {
    uint64_t bias = (v & 2) ? 1ULL << (2 * W - 1) : 0;
    int pick_a = (v & 1) ? (A ^ bias) >= (B ^ bias) : (A ^ bias) <= (B ^ bias);
    return pick_a ? A : B;
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
    /* the two-word forms: IMAD.WIDE (2), IADD.64 (4 signs), IMNMX.64 (4 forms) over a, b, c, d */
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++)
            for (int c = 0; c < n; c++)
                for (int d = 0; d < n; d++) {
                    uint64_t lo[10] = {0}, hi[10] = {0};
                    for (int i = 0; i < NP; i++) {
                        uint64_t PA = pair_at(pool[a], pool[b], i), PB = pair_at(pool[c], pool[d], i), r[10], l, h;
                        for (int sg = 0; sg < 2; sg++) {
                            wide1(at(pool[a], i), at(pool[b], i), at(pool[c], i), at(pool[d], i), sg, &l, &h);
                            r[sg] = l | h << W;
                        }
                        for (int s = 0; s < 4; s++) r[2 + s] = add64(PA, PB, s);
                        for (int v = 0; v < 4; v++) r[6 + v] = mnmx64(PA, PB, v);
                        for (int k = 0; k < 10; k++) {
                            lo[k] |= (r[k] & MASK) << (W * i);
                            hi[k] |= (r[k] >> W) << (W * i);
                        }
                    }
                    x.a = a; x.b = b; x.c = c; x.d = d; x.nw = 2;
                    for (int k = 0; k < (DROP == 6 ? 2 : 10); k++) {
                        x.form = k < 2 ? WIDE : k < 6 ? ADD64 : MNMX64;
                        x.imm = k < 2 ? k : k < 6 ? k - 2 : k - 6;
                        x.w[0] = lo[k]; x.w[1] = hi[k];
                        emit(&x, ctx);
                    }
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

static int find(const uint64_t *pool, int n, uint64_t v) {
    for (int i = 0; i < n; i++)
        if (pool[i] == v) return i;
    return -1;
}

/* A pool's words by their values at points 0 and 1 (2W <= 8 bits, else hashed to 8), to reject a needed word before
   scanning for it. */
typedef struct {
    uint64_t b[4];
} Sig;

static inline unsigned sig_of(uint64_t v) {
    uint64_t s = at(v, 0) | (NP > 1 ? at(v, 1) << W : 0);
    return W <= 4 ? (unsigned)s : (unsigned)((s * 0x9E3779B97F4A7C15ULL) >> 56);
}
static inline void sig_add(Sig *s, uint64_t v) { s->b[sig_of(v) >> 6] |= 1ULL << (sig_of(v) & 63); }
static inline int sig_has(const Sig *s, unsigned g) { return (s->b[g >> 6] >> (g & 63)) & 1; }

/* An instruction's new words (one or two) as a canonical key, or 0 when it adds none. */
static int new_words(const Ins *x, const uint64_t *pool, int n, uint64_t *k0, uint64_t *k1) {
    int fresh0 = find(pool, n, x->w[0]) < 0;
    int fresh1 = x->nw == 2 && find(pool, n, x->w[1]) < 0 && x->w[1] != x->w[0];
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
    if (L->n >= L->set.cap / 2) {
        fprintf(stderr, "set over half full (%zu of %u): raise set_new's capacity\n", L->n, L->set.cap);
        exit(4);
    }
    if (L->n == L->cap) {
        L->cap = L->cap ? 2 * L->cap : 4096;
        L->v = realloc(L->v, L->cap * sizeof(Ins));
    }
    Ins y = *x;
    y.nw = nw;
    y.w[0] = k0; y.w[1] = k1;
    L->v[L->n++] = y;
}

/* A program so far: its first and second instruction (NULL: none yet) and the pool they leave. */
typedef struct {
    const Ins *x1, *x2;
    const uint64_t *pool;
    int n1, n2;
    Sig sig;
} Prefix;

static long found[MAXJ], sel[MAXJ], le2[MAXJ];
static int shown[3][MAXJ];

/* a hit of kind 0 (found), 1 (sel) or 2 (le2) for job j */
static void hit(int kind, int j, const Prefix *P, const Ins *last, int o) {
    long *cnt = kind == 0 ? found : kind == 1 ? sel : le2;
#pragma omp atomic
    cnt[j]++;
    int show;
#pragma omp atomic capture
    show = shown[kind][j]++;
    if (show >= 4) return;
    char names[MAXN][16], d1[128] = "none", d2[128] = "none", d3[128] = "";
    for (int k = 0; k < NI; k++) strcpy(names[k], in_name[k]);
    if (P->x1) {
        strcpy(names[NI], "v1");
        if (P->x1->nw == 2) strcpy(names[NI + 1], "v1b");
        describe(P->x1, names, d1, sizeof d1);
    }
    if (P->x2) {
        strcpy(names[P->n1], "v2");
        if (P->x2->nw == 2) strcpy(names[P->n1 + 1], "v2b");
        describe(P->x2, names, d2, sizeof d2);
    }
    if (last) describe(last, names, d3, sizeof d3);
#pragma omp critical
    {
        if (last)
            printf("%c %s %s ; %s ; %s -> (%s)\n", "FS"[kind], jlabel[j], d1, d2, d3, o ? "t2,t1" : "t1,t2");
        else
            printf("L %s %s ; %s\n", jlabel[j], d1, d2);
    }
}

/* The last instruction's operands p, q fixed: does some c_lo, c_hi of pool[0..n) make its words (L, H)? */
static int solve(uint64_t P, uint64_t Q, int sg, uint64_t L, uint64_t H, const Prefix *X, int *cl, int *ch) {
    const uint64_t *pool = X->pool;
    int n = X->n2;
    uint64_t need = 0, plo[16], phi[16];
    for (int i = 0; i < NP; i++) {
        wide1(at(P, i), at(Q, i), 0, 0, sg, &plo[i], &phi[i]);
        need |= ((at(L, i) - plo[i]) & MASK) << (W * i);
        if (i == 1 && !sig_has(&X->sig, sig_of(need))) return 0;
    }
    int c = find(pool, n, need);
    if (c < 0) return 0;
    uint64_t needh = 0;
    for (int i = 0; i < NP; i++)
        needh |= ((at(H, i) - phi[i] - ((plo[i] + at(pool[c], i)) >> W)) & MASK) << (W * i);
    int k = find(pool, n, needh);
    if (k < 0) return 0;
    *cl = c; *ch = k;
    return 1;
}

/* b_lo with ±a_lo ± b_lo = L */
static inline uint64_t add64_blo(uint64_t alo, uint64_t L, int s) {
    uint64_t r = 0;
    for (int i = 0; i < NP; i++) {
        uint64_t b = (at(L, i) - ((s & 1) ? 0 - at(alo, i) : at(alo, i))) & MASK;
        r |= (((s & 2) ? 0 - b : b) & MASK) << (W * i);
    }
    return r;
}

/* A last IADD.64 whose low operands pool[ia], pool[ib] make L: every a_hi, and the b_hi it needs for H. Counts only
   programs reading a second-instruction word once there is a second instruction. */
static void add64_last(const Prefix *P, int j, int o, int ia, int ib, int s) {
    uint64_t L = o ? t2[j] : t1[j], H = o ? t1[j] : t2[j];
    for (int ih = 0; ih < P->n2; ih++) {
        uint64_t need = 0;
        int i = 0;
        for (; i < NP; i++) {
            uint64_t T = at(L, i) | at(H, i) << W, A = pair_at(P->pool[ia], P->pool[ih], i);
            uint64_t B = (T - ((s & 1) ? 0 - A : A)) & M2;
            if (s & 2) B = (0 - B) & M2;
            need |= (B >> W) << (W * i);
            if (i == 1 && !sig_has(&P->sig, sig_of(need))) break;
        }
        if (i < NP) continue;
        int k = find(P->pool, P->n2, need);
        if (k < 0) continue;
        if (P->x2 && ia < P->n1 && ib < P->n1 && ih < P->n1 && k < P->n1) continue;
        Ins last = {ADD64, (uint8_t)ia, (uint8_t)ih, (uint8_t)ib, (uint8_t)k, (uint8_t)s, 2, {0, 0}};
        hit(0, j, P, &last, o);
    }
}

static inline uint32_t eqmask(uint64_t v, uint64_t t) {
    uint32_t m = 0;
    for (int i = 0; i < NP; i++)
        if (at(v, i) == at(t, i)) m |= 1u << i;
    return m;
}

typedef struct {
    uint8_t lo, hi;
    uint32_t m;
} Part;

/* Pairs (pool[A.lo], pool[A.hi]) and (pool[B.lo], pool[B.hi]) that between them equal (L, H) at every point. */
static void choice(const Prefix *P, int j, int o, const Part *A, const Part *B) {
    uint64_t L = o ? t2[j] : t1[j], H = o ? t1[j] : t2[j];
    Ins last = {MNMX64, A->lo, A->hi, B->lo, B->hi, 0, 2, {0, 0}};
    for (int v = 0; v < 4; v++) {
        int ok = 1;
        for (int i = 0; i < NP && ok; i++)
            ok = mnmx64(pair_at(P->pool[A->lo], P->pool[A->hi], i), pair_at(P->pool[B->lo], P->pool[B->hi], i), v) ==
                 (at(L, i) | at(H, i) << W);
        if (ok) {
            last.imm = v;
            hit(0, j, P, &last, o);
            return;
        }
    }
    last.form = SEL64;
    hit(1, j, P, &last, o);
}

int main(void) {
    if (scanf("%d %d %d", &W, &NP, &NI) != 3 || W < 2 || W > MAXW || W * NP > 64 || NP > 16 ||
        NI + 2 * 2 > MAXN) {
        fprintf(stderr, "bad header\n");
        return 2;
    }
    MASK = (1ULL << W) - 1;
    M2 = 2 * W == 64 ? ~0ULL : (1ULL << (2 * W)) - 1;
    ALL = W * NP == 64 ? ~0ULL : (1ULL << (W * NP)) - 1;
    FULLP = (1u << NP) - 1;
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
    Prefix none = {NULL, NULL, in_v, NI, NI, {{0}}};
    for (int j = 0; j < NJ; j++)
        if (find(in_v, NI, t1[j]) >= 0 || find(in_v, NI, t2[j]) >= 0) hit(2, j, &none, NULL, 0);

    List firsts = {0};
    firsts.set = set_new(1u << 21);
    firsts.pool = in_v;
    firsts.np = NI;
    each_instruction(in_v, NI, collect, &firsts);

    long prefixes = 0;
#pragma omp parallel reduction(+ : prefixes)
    {
        List seconds = {0};
        seconds.set = set_new(1u << 21);
        int npre = 0, cap = 4 * MAXJ * MAXN * MAXN;
        uint64_t *pre_lo = malloc(cap * 8), *pre_hi = malloc(cap * 8);
        uint8_t *pre_in = malloc(cap), (*pre_m)[5] = malloc(cap * 5);
        int *hits = malloc(cap * sizeof(int));
        /* IADD.64: b_lo for each a_lo of the first's pool, and where it is in that pool (-1: not) */
        uint64_t *blo = malloc(MAXN * 4 * MAXJ * 2 * 8);
        int *blo_at = malloc(MAXN * 4 * MAXJ * 2 * sizeof(int));
        /* IMNMX.64 / SEL.64: the points where each word equals L, H; the first's pool's pairs that equal (L, H) somewhere */
        uint32_t el[MAXJ][2][MAXN], eh[MAXJ][2][MAXN];
        Part *p1 = malloc(MAXJ * 2 * MAXN * MAXN * sizeof(Part)), *p2 = malloc(MAXN * MAXN * sizeof(Part));
        int np1[MAXJ][2];
#pragma omp for schedule(dynamic, 4)
        for (size_t f = 0; f < firsts.n; f++) {
            const Ins *x1 = &firsts.v[f];
            uint64_t pool[MAXN];
            memcpy(pool, in_v, NI * 8);
            int n1 = NI;
            pool[n1++] = x1->w[0];
            if (x1->nw == 2) pool[n1++] = x1->w[1];
            Prefix P1 = {x1, NULL, pool, n1, n1, {{0}}};
            for (int k = 0; k < n1; k++) sig_add(&P1.sig, pool[k]);
            for (int j = 0; j < NJ; j++)
                if (find(pool + NI, n1 - NI, t1[j]) >= 0 || find(pool + NI, n1 - NI, t2[j]) >= 0) hit(2, j, &P1, NULL, 0);
            /* IMAD.WIDE: p, q of the inputs and the first's words, the c_lo and c_hi the last instruction needs, once */
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
                                pre_in[npre] = (find(pool, n1, need) >= 0) | (find(pool, n1, needh) >= 0) << 1;
                                pre_m[npre][0] = p; pre_m[npre][1] = q; pre_m[npre][2] = sg; pre_m[npre][3] = j;
                                pre_m[npre][4] = o;
                                npre++;
                            }
            for (int ia = 0; ia < n1; ia++)
                for (int s = 0; s < 4; s++)
                    for (int j = 0; j < NJ; j++)
                        for (int o = 0; o < 2; o++) {
                            int e = ((ia * 4 + s) * NJ + j) * 2 + o;
                            blo[e] = add64_blo(pool[ia], o ? t2[j] : t1[j], s);
                            blo_at[e] = find(pool, n1, blo[e]);
                        }
            for (int j = 0; j < NJ; j++)
                for (int o = 0; o < 2; o++) {
                    uint64_t L = o ? t2[j] : t1[j], H = o ? t1[j] : t2[j];
                    for (int k = 0; k < n1; k++) {
                        el[j][o][k] = eqmask(pool[k], L);
                        eh[j][o][k] = eqmask(pool[k], H);
                    }
                    Part *q = p1 + (j * 2 + o) * MAXN * MAXN;
                    np1[j][o] = 0;
                    for (int a = 0; a < n1; a++)
                        for (int b = 0; b < n1; b++)
                            if (el[j][o][a] & eh[j][o][b]) q[np1[j][o]++] = (Part){a, b, el[j][o][a] & eh[j][o][b]};
                }
            seconds.n = 0;
            seconds.set.g++;
            seconds.pool = pool;
            seconds.np = n1;
            each_instruction(pool, n1, collect, &seconds);
            /* s == seconds.n: no second instruction, the programs of at most two instructions */
            for (size_t s = 0; s <= seconds.n; s++) {
                const Ins *x2 = s < seconds.n ? &seconds.v[s] : NULL;
                int n2 = n1;
                if (x2) {
                    pool[n2++] = x2->w[0];
                    if (x2->nw == 2) pool[n2++] = x2->w[1];
                    prefixes++;
                }
                Prefix P = {x1, x2, pool, n1, n2, P1.sig};
                for (int k = n1; k < n2; k++) sig_add(&P.sig, pool[k]);
                if (x2 && DROP != 5)
                    for (int j = 0; j < NJ; j++)
                        if (find(pool + n1, n2 - n1, t1[j]) >= 0 || find(pool + n1, n2 - n1, t2[j]) >= 0)
                            hit(2, j, &P, NULL, 0);
                /* IMAD.WIDE, p and q before the second instruction */
                int nh = 0;
                for (int e = 0; e < npre; e++) {
                    int ok_lo = (pre_in[e] & 1) || (x2 && (pre_lo[e] == pool[n1] || pre_lo[e] == pool[n2 - 1]));
                    int ok_hi = (pre_in[e] & 2) || (x2 && (pre_hi[e] == pool[n1] || pre_hi[e] == pool[n2 - 1]));
                    if (ok_lo && ok_hi && (x2 ? (pre_in[e] & 3) != 3 : 1)) hits[nh++] = e;
                }
                for (int h = 0; h < nh && DROP != 4; h++) {
                    int e = hits[h];
                    Ins last = {WIDE, pre_m[e][0], pre_m[e][1], (uint8_t)find(pool, n2, pre_lo[e]),
                                (uint8_t)find(pool, n2, pre_hi[e]), pre_m[e][2], 2, {0, 0}};
                    hit(0, pre_m[e][3], &P, &last, pre_m[e][4]);
                }
                /* IMAD.WIDE, p or q a word of the second instruction */
                if (x2 && DROP != 3)
                    for (int q = n1; q < n2; q++)
                        for (int p = 0; p <= q; p++)
                            for (int sg = 0; sg < 2; sg++)
                                for (int j = 0; j < NJ; j++)
                                    for (int o = 0; o < 2; o++) {
                                        int cl, ch;
                                        if (!solve(pool[p], pool[q], sg, o ? t2[j] : t1[j], o ? t1[j] : t2[j], &P, &cl,
                                                   &ch))
                                            continue;
                                        Ins last = {WIDE, (uint8_t)p, (uint8_t)q, (uint8_t)cl, (uint8_t)ch, (uint8_t)sg,
                                                    2, {0, 0}};
                                        hit(0, j, &P, &last, o);
                                    }
                /* IADD.64: a_lo of the first's pool (b_lo solved once) or a second-instruction word */
                for (int j = 0; j < NJ && DROP != 1; j++)
                    for (int o = 0; o < 2; o++)
                        for (int s4 = 0; s4 < 4; s4++) {
                            for (int ia = 0; ia < n1; ia++) {
                                int e = ((ia * 4 + s4) * NJ + j) * 2 + o, ib = blo_at[e];
                                if (ib < 0 && x2) ib = blo[e] == pool[n1] ? n1 : blo[e] == pool[n2 - 1] ? n2 - 1 : -1;
                                if (ib >= 0) add64_last(&P, j, o, ia, ib, s4);
                            }
                            for (int ia = n1; ia < n2; ia++) {
                                uint64_t b = add64_blo(pool[ia], o ? t2[j] : t1[j], s4);
                                int ib = sig_has(&P.sig, sig_of(b)) ? find(pool, n2, b) : -1;
                                if (ib >= 0) add64_last(&P, j, o, ia, ib, s4);
                            }
                        }
                /* IMNMX.64 and SEL.64: two pairs covering every point, one of them reading a second-instruction word */
                for (int j = 0; j < NJ; j++)
                    for (int o = 0; o < 2; o++) {
                        const Part *q1 = p1 + (j * 2 + o) * MAXN * MAXN;
                        if (!x2) {
                            for (int a = 0; a < np1[j][o]; a++)
                                for (int b = a; b < np1[j][o]; b++)
                                    if ((q1[a].m | q1[b].m) == FULLP) choice(&P, j, o, &q1[a], &q1[b]);
                            continue;
                        }
                        if (DROP == 2) continue;
                        uint64_t L = o ? t2[j] : t1[j], H = o ? t1[j] : t2[j];
                        for (int k = n1; k < n2; k++) {
                            el[j][o][k] = eqmask(pool[k], L);
                            eh[j][o][k] = eqmask(pool[k], H);
                        }
                        int n2p = 0;
                        for (int a = 0; a < n2; a++)
                            for (int b = a < n1 ? n1 : 0; b < n2; b++)
                                if (el[j][o][a] & eh[j][o][b]) p2[n2p++] = (Part){a, b, el[j][o][a] & eh[j][o][b]};
                        for (int a = 0; a < n2p; a++) {
                            for (int b = a; b < n2p; b++)
                                if ((p2[a].m | p2[b].m) == FULLP) choice(&P, j, o, &p2[a], &p2[b]);
                            for (int b = 0; b < np1[j][o]; b++)
                                if ((p2[a].m | q1[b].m) == FULLP) choice(&P, j, o, &p2[a], &q1[b]);
                        }
                    }
            }
        }
    }
    for (int j = 0; j < NJ; j++)
        printf("J %s found %ld sel %ld le2 %ld firsts %zu prefixes %ld\n", jlabel[j], found[j], sel[j], le2[j], firsts.n,
               prefixes);
    return 0;
}
