/* FOLD-DYN: an average-case superoptimizer over the members of dynamic fold families, at reduced width W <= 8.

   A dynamic fold draws its step's structure, not only its constants, from live coins: theta = (ops, rotations,
   multipliers, keys). For a member theta and S steps, the target is the fold state after S words,
     s_S = step(theta_S, ... step(theta_1, s_0, w_1) ..., w_S),
   a function of the inputs (s_0, w_1, .., w_S); with -z the words are a zero-data prefix chain, w_t = w_{t-1} + d_t,
   and the inputs are (s_0, w_1, d_2, .., d_S). This asks, for each member: does a straight-line program of at most
   D instructions compute the target on every input, with every W-bit constant free in every operand slot (MH100w's
   free set F holds every constant)? D is 1 (direct) or 2 (the first instruction enumerated over the inputs and every
   constant, deduplicated by its values at the test points, the last one inverted against the target).

   The modelled ISA, at width W: IADD3 at every sign, LOP3 at every LUT, IMAD, IMAD with -c, IMAD.HI, IMAD.WIDE's high
   word, LEA and LEA.HI at every shift, the funnel shifts SHF.L.W and SHF.R.W at every immediate shift and at a register
   shift (wrapping mod W). Not in it: PRMT, IDP.4A, IMNMX, SEL and predicates, carries (IADD3.X), the two-word forms
   and memory (tables), and a register-shift funnel of two constants.

   A value is the vector of a word at NP test points; a found program is then checked on the whole input domain (every
   input tuple when it has at most 2^24, else 2^22 random ones), and printed only if it holds there.

   usage: fold_dyn W NP family S D members seed [-z] [-a]
     family: rotadd rotxoradd rotmuladd mulrotadd xormul xorrot rand
     -a: every member (S = 1 only), else `members` random ones
   stdout: "N1 <distinct first instructions>" and, a member a line,
     "M <theta> honest <h> found <d|-> [<program>]"   with found the least depth <= D that computes the target
   build: gcc -O2 -fopenmp -o fold_dyn fold_dyn.c */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPMAX 64
#define MAXIN 8
#define MAXPOOL 10
#define CONST 255

static int W, NP, NIN;
static uint32_t MK;

typedef struct { uint8_t p[NPMAX]; } Val;
typedef struct { uint8_t f, q, s[3]; uint8_t c[3]; } Ins;

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static inline uint32_t rotl(uint32_t x, int r) { r %= W; return r ? (((x << r) | (x >> (W - r))) & MK) : x; }
static inline uint32_t fl(uint32_t lo, uint32_t hi, int n) { n %= W; return n ? (((hi << n) | (lo >> (W - n))) & MK) : hi; }
static inline uint32_t fr(uint32_t lo, uint32_t hi, int n) { n %= W; return n ? (((lo >> n) | (hi << (W - n))) & MK) : lo; }

enum { IADD3, LOP3, IMAD, IMADN, IMADHI, IMADWH, LEA, LEAHI, SHFL, SHFR, SHFLV, SHFRV, NFORM };
static const int AR[NFORM] = {3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 3, 3};
static const char *FN[NFORM] = {"IADD3", "LOP3", "IMAD", "IMAD-c", "IMAD.HI", "IMAD.WIDE.hi", "LEA", "LEA.HI",
                                "SHF.L", "SHF.R", "SHF.L.v", "SHF.R.v"};
/* the slot a constant enters additively (solved, not searched), or -1 */
static const int ADDSLOT[NFORM] = {9, -1, 2, 2, 2, -1, 1, 2, -1, -1, -1, -1};

static int nparam(int f) {
    switch (f) {
    case IADD3: return 8;
    case LOP3: return 256;
    case LEA: case LEAHI: case SHFL: case SHFR: return W;
    default: return 1;
    }
}

static inline uint32_t lop3(int q, uint32_t x, uint32_t y, uint32_t z) {
    uint32_t o = 0;
    for (int i = 0; i < 8; i++)
        if ((q >> i) & 1) o |= ((i & 1) ? x : ~x) & ((i & 2) ? y : ~y) & ((i & 4) ? z : ~z);
    return o & MK;
}

static inline uint32_t ev(int f, int q, uint32_t x, uint32_t y, uint32_t z) {
    switch (f) {
    case IADD3: return (((q & 1) ? -x : x) + ((q & 2) ? -y : y) + ((q & 4) ? -z : z)) & MK;
    case LOP3: return lop3(q, x, y, z);
    case IMAD: return (x * y + z) & MK;
    case IMADN: return (x * y - z) & MK;
    case IMADHI: return (((x * y) >> W) + z) & MK;
    case IMADWH: return ((x * y + z) >> W) & MK;
    case LEA: return ((x << q) + y) & MK;
    case LEAHI: return (fl(x, y, q) + z) & MK;
    case SHFL: return fl(x, y, q);
    case SHFR: return fr(x, y, q);
    case SHFLV: return fl(x, y, z);
    case SHFRV: return fr(x, y, z);
    }
    return 0;
}

/* ---- families: one step, s' = step(theta, s, w) ------------------------------------------------------------------ */

typedef struct { int kind, op1, op2, ab; uint32_t k1, k2; int r1, r2; } Theta;
enum { ROTADD, ROTXORADD, ROTMULADD, MULROTADD, XORMUL, XORROT, RAND, NFAM };
static const char *FAM[NFAM] = {"rotadd", "rotxoradd", "rotmuladd", "mulrotadd", "xormul", "xorrot", "rand"};
/* rand: s' = g2(g1(s) (+|^) w), g1 in {rot, mul, xor, add} by a key, g2 in {rot, mul} */
enum { G_ROT, G_MUL, G_XOR, G_ADD };

static inline uint32_t g(int op, uint32_t x, uint32_t k, int r) {
    switch (op) {
    case G_ROT: return rotl(x, r);
    case G_MUL: return (x * k) & MK;
    case G_XOR: return x ^ k;
    default: return (x + k) & MK;
    }
}

static inline uint32_t step(const Theta *t, uint32_t s, uint32_t w) {
    switch (t->kind) {
    case ROTADD: return (rotl(s, t->r1) + w) & MK;
    case ROTXORADD: return (rotl(s ^ t->k1, t->r1) + w) & MK;
    case ROTMULADD: return (rotl(s, t->r1) * t->k1 + w) & MK;
    case MULROTADD: return (rotl((s * t->k1) & MK, t->r1) + w) & MK;
    case XORMUL: return ((s ^ w) * t->k1) & MK;
    case XORROT: return rotl(s ^ w, t->r1);
    default: {
        uint32_t u = g(t->op1, s, t->k1, t->r1);
        u = t->ab ? (u ^ w) : ((u + w) & MK);
        return g(t->op2, u, t->k2, t->r2);
    }
    }
}

/* honest lane instructions a step: rotadd is one LEA.HI; rand fuses g1 with the absorb where one instruction does both
   (xor-key-xor is LOP3, add-key-add IADD3, mul-add IMAD, rot-add LEA.HI) */
static int honest(const Theta *t) {
    switch (t->kind) {
    case ROTADD: return 1;
    case RAND: {
        int fused = (t->op1 == G_XOR && t->ab) || (t->op1 == G_ADD && !t->ab) || (t->op1 == G_MUL && !t->ab) ||
                    (t->op1 == G_ROT && !t->ab);
        return 3 - fused;
    }
    default: return 2;
    }
}

static uint32_t odd(void) { return ((uint32_t)rnd() | 1u) & MK; }
static int rot1(void) { return 1 + (int)(rnd() % (uint64_t)(W - 1)); }

static Theta draw(int kind) {
    Theta t = {kind, 0, 0, 0, 0, 0, 0, 0};
    t.r1 = rot1(); t.r2 = rot1();
    t.k1 = odd(); t.k2 = odd();
    if (kind == ROTXORADD) t.k1 = (uint32_t)rnd() & MK;
    if (kind == RAND) {
        t.op1 = (int)(rnd() % 4); t.op2 = (rnd() & 1) ? G_ROT : G_MUL; t.ab = (int)(rnd() & 1);
        if (t.op1 == G_XOR || t.op1 == G_ADD) t.k1 = (uint32_t)rnd() & MK;
    }
    return t;
}

static void show(const Theta *t, char *b) {
    static const char *GN[4] = {"rot", "mul", "xor", "add"};
    switch (t->kind) {
    case ROTADD: case XORROT: sprintf(b, "r%d", t->r1); break;
    case XORMUL: sprintf(b, "k%u", t->k1); break;
    case RAND: sprintf(b, "%s:%u:%d/%s/%s:%u:%d", GN[t->op1], t->k1, t->r1, t->ab ? "xor" : "add", GN[t->op2], t->k2,
                       t->r2); break;
    default: sprintf(b, "k%u.r%d", t->k1, t->r1);
    }
}

/* ---- inputs and targets --------------------------------------------------------------------------------------- */

static int S, ZD;
static uint32_t pts[NPMAX][MAXIN];

static uint32_t target_at(const Theta *th, const uint32_t *in) {
    uint32_t s = in[0], w = in[1];
    for (int t = 0; t < S; t++) {
        if (t > 0) w = ZD ? ((w + in[1 + t]) & MK) : in[1 + t];
        s = step(&th[t], s, w);
    }
    return s;
}

/* ---- the search ------------------------------------------------------------------------------------------------ */

static Val pool[MAXPOOL];

static inline uint32_t src(const Val *P, int s, uint32_t c, int i) { return s == CONST ? c : P[s].p[i]; }

/* does `t` equal one instruction over P[0..np) and free constants, reading P[must] if must >= 0? */
static int find1(const Val *t, const Val *P, int np, int must, Ins *out) {
    int ns = np + 1;
    for (int f = 0; f < NFORM; f++) {
        int ar = AR[f];
        int npat = ar == 3 ? ns * ns * ns : ns * ns;
        for (int pat = 0; pat < npat; pat++) {
            int s[3] = {pat % ns, (pat / ns) % ns, ar == 3 ? pat / (ns * ns) : 0};
            int nc = 0, hasm = must < 0, nv = 0;
            for (int j = 0; j < ar; j++) {
                if (s[j] == np) { s[j] = CONST; nc++; } else { nv++; if (s[j] == must) hasm = 1; }
            }
            if (!nv || !hasm) continue;
            if (ar == 2) s[2] = CONST;
            if (f == IADD3 && nc > 1) continue;            /* two constants sum to one */
            if (f == LOP3) {
                /* every LUT at once: per bit, the observed map (x_b, y_b, z_b) -> t_b; constant slots split the bit
                   positions by the constant's bit, so a constant slot is a per-bit choice between LUT halves. */
                int cs[3], ncs = 0, vs[3], nvs = 0;
                for (int j = 0; j < 3; j++) { if (s[j] == CONST) cs[ncs++] = j; else vs[nvs++] = j; }
                if (ncs == 3) continue;
                /* tab[b][key over the variable slots] in {-1, 0, 1} */
                int tab[8][8], ok = 1;
                for (int b = 0; b < W; b++) for (int k = 0; k < 8; k++) tab[b][k] = -1;
                for (int i = 0; i < NP && ok; i++)
                    for (int b = 0; b < W && ok; b++) {
                        int key = 0;
                        for (int j = 0; j < nvs; j++) key |= ((P[s[vs[j]]].p[i] >> b) & 1) << j;
                        int o = (t->p[i] >> b) & 1;
                        if (tab[b][key] < 0) tab[b][key] = o; else if (tab[b][key] != o) ok = 0;
                    }
                if (!ok) continue;
                int nk = 1 << nvs, nh = 1 << ncs;  /* the LUT has nh halves of nk entries, one a constant-bit pattern */
                /* find nh functions (on nk keys) covering every bit's partial table: greedy over the 2^nk choices */
                int nfun = 1 << nk, chosen[4], nchosen = 0, cover[8] = {0}, done = 0;
                for (int b = 0; b < W; b++) cover[b] = 0;
                /* exhaustive over sets of at most nh functions, nk <= 4 so nfun <= 16 */
                for (int a0 = 0; a0 < nfun && !done; a0++)
                    for (int a1 = (nh > 1 ? a0 : nfun - 1); a1 < nfun && !done; a1++)
                        for (int a2 = (nh > 2 ? a1 : nfun - 1); a2 < nfun && !done; a2++)
                            for (int a3 = (nh > 3 ? a2 : nfun - 1); a3 < nfun && !done; a3++) {
                                int fs[4] = {a0, a1, a2, a3}, all = 1;
                                for (int b = 0; b < W && all; b++) {
                                    int any = 0;
                                    for (int h = 0; h < nh && !any; h++) {
                                        int m = 1;
                                        for (int k = 0; k < nk && m; k++)
                                            if (tab[b][k] >= 0 && tab[b][k] != ((fs[h] >> k) & 1)) m = 0;
                                        if (m) { any = 1; cover[b] = h; }
                                    }
                                    if (!any) all = 0;
                                }
                                if (all) { done = 1; for (int h = 0; h < nh; h++) chosen[h] = fs[h]; nchosen = nh; }
                            }
                if (!done) continue;
                /* assemble the LUT and the constants */
                int lut = 0; uint32_t cv[3] = {0, 0, 0};
                for (int idx = 0; idx < 8; idx++) {
                    int key = 0, h = 0;
                    for (int j = 0; j < nvs; j++) key |= ((idx >> vs[j]) & 1) << j;
                    for (int j = 0; j < ncs; j++) h |= ((idx >> cs[j]) & 1) << j;
                    if ((chosen[h < nchosen ? h : 0] >> key) & 1) lut |= 1 << idx;
                }
                for (int b = 0; b < W; b++)
                    for (int j = 0; j < ncs; j++) cv[cs[j]] |= (uint32_t)((cover[b] >> j) & 1) << b;
                out->f = LOP3; out->q = (uint8_t)lut;
                for (int j = 0; j < 3; j++) { out->s[j] = (uint8_t)s[j]; out->c[j] = (uint8_t)cv[j]; }
                /* recheck (the cover's halves must be the LUT's) */
                int good = 1;
                for (int i = 0; i < NP && good; i++)
                    if (lop3(lut, src(P, s[0], cv[0], i), src(P, s[1], cv[1], i), src(P, s[2], cv[2], i)) != t->p[i])
                        good = 0;
                if (good) return 1;
                continue;
            }
            int add = ADDSLOT[f] == 9 ? -1 : ADDSLOT[f];
            if (f == IADD3) for (int j = 0; j < 3; j++) if (s[j] == CONST) add = j;
            int free_[3], nf = 0;
            for (int j = 0; j < ar; j++) if (s[j] == CONST && j != add) free_[nf++] = j;
            int addc = add >= 0 && s[add] == CONST;
            if (nf > 1) {
                if (f != IMADWH) continue;   /* two searched constants: only IMAD.WIDE.hi(x, c, c'), below */
            }
            for (int q = 0; q < nparam(f); q++) {
                if (f == IMADWH && nf == 2) {
                    /* (x * c1 + c2) >> W = t: c2 lies in an interval at each point */
                    int xs = s[0] != CONST ? 0 : (s[1] != CONST ? 1 : 2);
                    if (xs == 2) continue;
                    for (uint32_t c1 = 0; c1 <= MK; c1++) {
                        int64_t lo = 0, hi = (int64_t)MK;
                        for (int i = 0; i < NP && lo <= hi; i++) {
                            int64_t pr = (int64_t)P[s[xs]].p[i] * c1, tv = t->p[i];
                            int64_t a = tv * (MK + 1) - pr, b = (tv + 1) * (MK + 1) - 1 - pr;
                            if (a > lo) lo = a;
                            if (b < hi) hi = b;
                        }
                        if (lo <= hi) {
                            out->f = (uint8_t)f; out->q = 0;
                            for (int j = 0; j < 3; j++) out->s[j] = (uint8_t)s[j];
                            out->c[xs ? 0 : 1] = (uint8_t)c1; out->c[2] = (uint8_t)lo;
                            return 1;
                        }
                    }
                    continue;
                }
                uint32_t cmax = nf ? MK : 0;
                for (uint32_t c = 0; c <= cmax; c++) {
                    uint32_t cv[3] = {0, 0, 0};
                    if (nf) cv[free_[0]] = c;
                    if (addc) {
                        /* solve the additive constant at point 0 */
                        cv[add] = 0;
                        uint32_t v0 = ev(f, q, src(P, s[0], cv[0], 0), src(P, s[1], cv[1], 0), src(P, s[2], cv[2], 0));
                        uint32_t sg = (f == IADD3 && ((q >> add) & 1)) || f == IMADN ? MK : 1;
                        cv[add] = ((t->p[0] - v0) * sg) & MK;
                    }
                    int i = 0;
                    for (; i < NP; i++)
                        if (ev(f, q, src(P, s[0], cv[0], i), src(P, s[1], cv[1], i), src(P, s[2], cv[2], i)) != t->p[i])
                            break;
                    if (i == NP) {
                        out->f = (uint8_t)f; out->q = (uint8_t)q;
                        for (int j = 0; j < 3; j++) { out->s[j] = (uint8_t)s[j]; out->c[j] = (uint8_t)cv[j]; }
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

/* ---- the first instructions, deduplicated ---------------------------------------------------------------------- */

static Val *L1; static Ins *L1i; static size_t nL1, capL1;
static uint64_t *hset; static size_t hcap;

static uint64_t vhash(const Val *v) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < NP; i++) { h ^= v->p[i]; h *= 1099511628211ull; }
    return h | 1;
}

static int seen_or_add(const Val *v, const Ins *ins) {
    uint64_t h = vhash(v);
    size_t m = hcap - 1, i = h & m;
    while (hset[2 * i]) {
        if (hset[2 * i] == h && !memcmp(L1[hset[2 * i + 1]].p, v->p, (size_t)NP)) return 1;
        i = (i + 1) & m;
    }
    if (2 * nL1 >= hcap) { fprintf(stderr, "fold_dyn: more than %zu first instructions\n", hcap / 2); exit(3); }
    if (nL1 == capL1) {
        capL1 *= 2;
        L1 = realloc(L1, capL1 * sizeof(Val));
        L1i = realloc(L1i, capL1 * sizeof(Ins));
    }
    L1[nL1] = *v; L1i[nL1] = *ins;
    hset[2 * i] = h; hset[2 * i + 1] = nL1++;
    return 0;
}

static int is_input(const Val *v) {
    for (int k = 0; k < NIN; k++) if (!memcmp(v->p, pool[k].p, (size_t)NP)) return 1;
    return 0;
}

static int is_const(const Val *v) {
    for (int i = 1; i < NP; i++) if (v->p[i] != v->p[0]) return 0;
    return 1;
}

static void gen_level1(void) {
    hcap = 1u << 24;
    hset = calloc(2 * hcap, sizeof(uint64_t));
    capL1 = 1u << 20; nL1 = 0;
    L1 = malloc(capL1 * sizeof(Val)); L1i = malloc(capL1 * sizeof(Ins));
    int ns = NIN + 1;
    for (int f = 0; f < NFORM; f++) {
        int ar = AR[f], npat = ar == 3 ? ns * ns * ns : ns * ns;
        for (int pat = 0; pat < npat; pat++) {
            int s[3] = {pat % ns, (pat / ns) % ns, ar == 3 ? pat / (ns * ns) : 0}, nc = 0, nv = 0, cs[3];
            for (int j = 0; j < ar; j++) { if (s[j] == NIN) { s[j] = CONST; cs[nc++] = j; } else nv++; }
            if (!nv) continue;
            if (ar == 2) s[2] = CONST;
            if (f == IADD3 && nc > 1) continue;
            if ((f == SHFLV || f == SHFRV) && nc == 2) continue;
            if (f == LOP3 && nc == 2) {
                /* LOP3(x, c, c') is a per-bit choice of 0, 1, x or ~x: LUT 0x6C with (c, c') = (0,0) 0, (1,0) 1,
                   (0,1) x, (1,1) ~x, the variable in slot 0 (the other slot orders are the same values) */
                if (s[0] == CONST) continue;
                for (uint32_t code = 0; code < (1u << (2 * W)); code++) {
                    uint32_t c1 = 0, c2 = 0;
                    for (int b = 0; b < W; b++) {
                        int u = (code >> (2 * b)) & 3;
                        c1 |= (uint32_t)(u == 1 || u == 3) << b;
                        c2 |= (uint32_t)(u == 2 || u == 3) << b;
                    }
                    Val v;
                    for (int i = 0; i < NP; i++) v.p[i] = (uint8_t)lop3(0x6C, pool[s[0]].p[i], c1, c2);
                    if (is_input(&v) || is_const(&v)) continue;
                    Ins ins = {LOP3, 0x6C, {(uint8_t)s[0], CONST, CONST}, {0, (uint8_t)c1, (uint8_t)c2}};
                    seen_or_add(&v, &ins);
                }
                continue;
            }
            uint64_t ncomb = 1;
            for (int j = 0; j < nc; j++) ncomb *= (uint64_t)MK + 1;
            for (int q = 0; q < nparam(f); q++)
                for (uint64_t cc = 0; cc < ncomb; cc++) {
                    uint32_t cv[3] = {0, 0, 0};
                    uint64_t r = cc;
                    for (int j = 0; j < nc; j++) { cv[cs[j]] = (uint32_t)(r % ((uint64_t)MK + 1)); r /= (uint64_t)MK + 1; }
                    Val v;
                    for (int i = 0; i < NP; i++)
                        v.p[i] = (uint8_t)ev(f, q, src(pool, s[0], cv[0], i), src(pool, s[1], cv[1], i),
                                             src(pool, s[2], cv[2], i));
                    if (is_input(&v) || is_const(&v)) continue;
                    Ins ins = {(uint8_t)f, (uint8_t)q, {(uint8_t)s[0], (uint8_t)s[1], (uint8_t)s[2]},
                               {(uint8_t)cv[0], (uint8_t)cv[1], (uint8_t)cv[2]}};
                    seen_or_add(&v, &ins);
                }
        }
    }
}

/* ---- checking a program on the whole domain -------------------------------------------------------------------- */

static uint32_t run_ins(const Ins *a, const uint32_t *regs) {
    uint32_t x[3];
    for (int j = 0; j < 3; j++) x[j] = a->s[j] == CONST ? a->c[j] : regs[a->s[j]];
    return ev(a->f, a->q, x[0], x[1], x[2]);
}

static int holds(const Theta *th, int depth, const Ins *p) {
    uint64_t dom = 1;
    for (int k = 0; k < NIN; k++) dom *= (uint64_t)MK + 1;
    int full = dom <= (1ull << 24);
    uint64_t n = full ? dom : (1ull << 22);
    uint64_t lr = 0x243F6A8885A308D3ull;
    for (uint64_t c = 0; c < n; c++) {
        uint32_t in[MAXIN], regs[MAXPOOL];
        uint64_t r = c;
        for (int k = 0; k < NIN; k++) {
            if (full) { in[k] = (uint32_t)(r % ((uint64_t)MK + 1)); r /= (uint64_t)MK + 1; }
            else { lr ^= lr << 13; lr ^= lr >> 7; lr ^= lr << 17; in[k] = (uint32_t)lr & MK; }
            regs[k] = in[k];
        }
        for (int d = 0; d < depth; d++) regs[NIN + d] = run_ins(&p[d], regs);
        if (regs[NIN + depth - 1] != target_at(th, in)) return 0;
    }
    return 1;
}

static void fmt_ins(const Ins *a, int base, char *b) {
    static const char *IN[] = {"s", "w1", "w2", "w3", "w4", "w5", "w6", "w7"};
    static const char *ZIN[] = {"s", "w1", "d2", "d3", "d4", "d5", "d6", "d7"};
    char ops[3][16];
    for (int j = 0; j < AR[a->f]; j++) {
        if (a->s[j] == CONST) sprintf(ops[j], "#%u", a->c[j]);
        else if (a->s[j] < NIN) sprintf(ops[j], "%s", ZD ? ZIN[a->s[j]] : IN[a->s[j]]);
        else sprintf(ops[j], "v%d", a->s[j] - NIN + 1);
    }
    (void)base;
    if (a->f == IADD3 || a->f == LOP3 || a->f == LEA || a->f == LEAHI || a->f == SHFL || a->f == SHFR)
        sprintf(b, "%s.%d(%s", FN[a->f], a->q, ops[0]);
    else sprintf(b, "%s(%s", FN[a->f], ops[0]);
    for (int j = 1; j < AR[a->f]; j++) { strcat(b, ","); strcat(b, ops[j]); }
    strcat(b, ")");
}

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: fold_dyn W NP family S D members seed [-z] [-a]\n");
        return 2;
    }
    W = atoi(argv[1]); NP = atoi(argv[2]); S = atoi(argv[4]);
    int D = atoi(argv[5]), members = atoi(argv[6]), all = 0;
    rs ^= (uint64_t)strtoull(argv[7], 0, 0) * 0x2545F4914F6CDD1Dull;
    for (int a = 8; a < argc; a++) { if (!strcmp(argv[a], "-z")) ZD = 1; if (!strcmp(argv[a], "-a")) all = 1; }
    int fam = -1;
    for (int k = 0; k < NFAM; k++) if (!strcmp(argv[3], FAM[k])) fam = k;
    if (W < 3 || W > 8 || NP < 8 || NP > NPMAX || S < 1 || S > MAXIN - 1 || fam < 0 || D < 1 || D > 2) {
        fprintf(stderr, "fold_dyn: bad arguments\n");
        return 2;
    }
    MK = (1u << W) - 1;
    NIN = S + 1;
    for (int i = 0; i < NP; i++)
        for (int k = 0; k < NIN; k++) { pts[i][k] = (uint32_t)rnd() & MK; pool[k].p[i] = (uint8_t)pts[i][k]; }

    /* the members */
    int nm = 0, cap = 1 << 16;
    Theta *th = malloc((size_t)cap * MAXIN * sizeof(Theta));
    if (all) {
        if (S != 1) { fprintf(stderr, "fold_dyn: -a needs S = 1\n"); return 2; }
        for (uint32_t k = 0; k <= MK; k++)
            for (int r = 1; r < W; r++) {
                Theta t = {fam, 0, 0, 0, k, 0, r, 0};
                int needk = fam == ROTXORADD || fam == ROTMULADD || fam == MULROTADD || fam == XORMUL;
                int needr = fam != XORMUL;
                if (!needk && k) continue;
                if (!needr && r != 1) continue;
                if ((fam == ROTMULADD || fam == MULROTADD || fam == XORMUL) && !(k & 1)) continue;
                if (fam == RAND) continue;
                th[nm++ * MAXIN] = t;
            }
        if (fam == RAND) {
            for (int o1 = 0; o1 < 4; o1++) for (int ab = 0; ab < 2; ab++) for (int o2 = 0; o2 < 2; o2++)
                for (int x = 0; x < 3; x++) {
                    Theta t = draw(RAND);
                    t.op1 = o1; t.ab = ab; t.op2 = o2 ? G_ROT : G_MUL;
                    if (nm < cap) th[nm++ * MAXIN] = t;
                }
        }
    } else {
        for (nm = 0; nm < members; nm++)
            for (int t = 0; t < S; t++) th[nm * MAXIN + t] = draw(fam);
    }

    if (D >= 2) gen_level1();
    printf("N1 %zu\n", D >= 2 ? nL1 : (size_t)0);
    printf("members %d family %s W %d S %d D %d zd %d\n", nm, FAM[fam], W, S, D, ZD);
    fflush(stdout);

    for (int m = 0; m < nm; m++) {
        const Theta *t = &th[m * MAXIN];
        Val tv;
        for (int i = 0; i < NP; i++) tv.p[i] = (uint8_t)target_at(t, pts[i]);
        int h = 0;
        for (int k = 0; k < S; k++) h += honest(&t[k]);
        if (ZD) h += S - 1;
        char tb[256] = "", b1[128];
        for (int k = 0; k < S; k++) { show(&t[k], b1); if (k) strcat(tb, ","); strcat(tb, b1); }
        Ins p[2];
        int found = 0;
        /* depth 0: the target is an input */
        if (is_input(&tv)) found = -1;
        if (!found && find1(&tv, pool, NIN, -1, &p[0]) && holds(t, 1, p)) found = 1;
        if (!found && D >= 2) {
            volatile int hit = 0;
            Ins best[2];
#pragma omp parallel for schedule(dynamic, 256)
            for (size_t j = 0; j < nL1; j++) {
                if (hit) continue;
                Val P2[MAXPOOL];
                memcpy(P2, pool, sizeof(Val) * NIN);
                P2[NIN] = L1[j];
                Ins last;
                if (find1(&tv, P2, NIN + 1, NIN, &last)) {
                    Ins prog[2] = {L1i[j], last};
                    if (holds(t, 2, prog)) {
#pragma omp critical
                        if (!hit) { hit = 1; best[0] = prog[0]; best[1] = prog[1]; }
                    }
                }
            }
            if (hit) { found = 2; p[0] = best[0]; p[1] = best[1]; }
        }
        char pb[256] = "";
        if (found > 0) {
            for (int d = 0; d < found; d++) {
                char ib[96];
                fmt_ins(&p[d], NIN, ib);
                if (d) strcat(pb, " ; ");
                strcat(pb, ib);
            }
        }
        if (found < 0) printf("M %s honest %d found 0 input\n", tb, h);
        else if (found) printf("M %s honest %d found %d %s\n", tb, h, found, pb);
        else printf("M %s honest %d found -\n", tb, h);
        fflush(stdout);
    }
    return 0;
}
