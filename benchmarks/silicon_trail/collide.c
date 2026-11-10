/* Count collisions of the chain directly.
 *
 *     cc -O3 -march=native -o collide collide.c
 *     collide SPEC BATCHES SEED
 *
 * Runs the chain from random start states on two inputs that differ by a fixed sequence d_1 .. d_k and counts how
 * often the two final states are equal.  512 start states are run at once, one per bit of a 512-bit word (the state is
 * held as 25w such words, one per state bit), so a batch is 512 trials.  The round constant is 0: it changes no
 * collision (selfcheck.py).  collide.py writes SPEC and adds up the counts of several processes.
 *
 * SPEC, one item a line (states are 25w-bit hex numbers, bit w(5y+x)+z the state's (x, y, z)):
 *   w W
 *   k K
 *   xmode 0|1|2        the common input after step 1: all zero, fresh random bits each trial, or the fixed x lines
 *   outer HEX          the absorbed positions
 *   d HEX              K lines: the input difference of each step, placed
 *   x HEX              K - 1 lines (xmode 2): the common input of steps 2 .. K, placed
 *   c HEX              K - 1 lines: the trail's difference after each round, to count trials that follow it
 *   force N            N = 0, or the number of row lines that follow: the start state is then sampled so that round 1
 *                      follows the trail (uniform among the states that do), and the counts are conditional on that
 *   b1 HEX             (force) the difference before chi in round 1
 *   row J0 J1 J2 J3 J4 BASE NB V...   (force) the row's five state-bit indices, one value of the row that gives the
 *                      trail's output difference, and a basis of the directions that keep it
 *
 * Output: trials, collisions, follow[j] (trials whose difference after each of rounds 1 .. j is the trail's),
 * and collisions among the trials that follow the whole trail.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t v512 __attribute__((vector_size(64)));
#define MAXBITS 1600
#define MAXK 12

static int w, k, nbits, xmode, nforce;
static unsigned char outer[MAXBITS], dmask[MAXK][MAXBITS], xmask[MAXK][MAXBITS], cmask[MAXK][MAXBITS], b1[MAXBITS];
static int src0[MAXBITS], src1[MAXBITS], src2[MAXBITS];
static struct { int j[5]; int base; int nb; int v[5]; } rows[MAXBITS / 5];

static const int RHO[5][5] = {  /* [x][y] */
    {0, 36, 3, 41, 18}, {1, 44, 10, 45, 2}, {62, 6, 43, 15, 61}, {28, 55, 25, 21, 56}, {27, 20, 39, 8, 14}};

static int bit(int x, int y, int z) { return w * (5 * y + x) + z; }

static v512 rng[4];
static uint64_t splitmix(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static inline v512 next(void) {  /* xoshiro256++, eight independent streams side by side */
    v512 r = rng[0] + rng[3];
    r = ((r << 23) | (r >> 41)) + rng[0];
    v512 t = rng[1] << 17;
    rng[2] ^= rng[0]; rng[3] ^= rng[1]; rng[1] ^= rng[2]; rng[0] ^= rng[3];
    rng[2] ^= t;
    rng[3] = (rng[3] << 45) | (rng[3] >> 19);
    return r;
}

static void hex(const char *s, unsigned char *out) {
    int n = (int)strlen(s);
    memset(out, 0, MAXBITS);
    for (int i = 0; i < n; i++) {
        char ch = s[n - 1 - i];
        int v = ch <= '9' ? ch - '0' : (ch | 32) - 'a' + 10;
        for (int j = 0; j < 4; j++)
            if (v >> j & 1) {
                if (4 * i + j >= nbits) { fprintf(stderr, "state wider than 25w\n"); exit(2); }
                out[4 * i + j] = 1;
            }
    }
}

static void chi(const v512 *in, v512 *out) {   /* in: before chi, in place of b; out: after chi */
    for (int y = 0; y < 5; y++)
        for (int z = 0; z < w; z++)
            for (int x = 0; x < 5; x++)
                out[bit(x, y, z)] = in[bit(x, y, z)] ^ (~in[bit((x + 1) % 5, y, z)] & in[bit((x + 2) % 5, y, z)]);
}

static void round_(v512 *s, v512 *tmp) {
    v512 par[5 * 64], eff;
    for (int x = 0; x < 5; x++)
        for (int z = 0; z < w; z++) {
            v512 p = s[bit(x, 0, z)];
            for (int y = 1; y < 5; y++) p ^= s[bit(x, y, z)];
            par[x * w + z] = p;
        }
    for (int x = 0; x < 5; x++)
        for (int z = 0; z < w; z++) {
            eff = par[((x + 4) % 5) * w + z] ^ par[((x + 1) % 5) * w + (z + w - 1) % w];
            for (int y = 0; y < 5; y++) s[bit(x, y, z)] ^= eff;
        }
    for (int j = 0; j < nbits; j++) tmp[j] = s[src0[j]] ^ (~s[src1[j]] & s[src2[j]]);
    memcpy(s, tmp, sizeof(v512) * nbits);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: collide SPEC BATCHES SEED\n"); return 2; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 2; }
    uint64_t batches = strtoull(argv[2], 0, 10), seed = strtoull(argv[3], 0, 10);
    static char line[8192], key[32], val[4096];
    int nd = 0, nx = 0, nc = 0, nrow = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%31s %4095s", key, val) != 2) continue;
        if (!strcmp(key, "w")) { w = atoi(val); nbits = 25 * w; }
        else if (!strcmp(key, "k")) k = atoi(val);
        else if (!strcmp(key, "xmode")) xmode = atoi(val);
        else if (!strcmp(key, "outer")) hex(val, outer);
        else if (!strcmp(key, "d")) hex(val, dmask[nd++]);
        else if (!strcmp(key, "x")) hex(val, xmask[nx++]);
        else if (!strcmp(key, "c")) hex(val, cmask[nc++]);
        else if (!strcmp(key, "force")) nforce = atoi(val);
        else if (!strcmp(key, "b1")) hex(val, b1);
        else if (!strcmp(key, "row")) {
            char *p = line + 3;
            for (int i = 0; i < 5; i++) rows[nrow].j[i] = (int)strtol(p, &p, 10);
            rows[nrow].base = (int)strtol(p, &p, 10);
            rows[nrow].nb = (int)strtol(p, &p, 10);
            for (int i = 0; i < rows[nrow].nb; i++) rows[nrow].v[i] = (int)strtol(p, &p, 10);
            nrow++;
        }
    }
    fclose(f);
    if (w < 1 || w > 64 || k < 2 || k > MAXK || nd != k || nc != k - 1 || nrow != nforce || (xmode == 2 && nx != k - 1)) {
        fprintf(stderr, "bad spec\n");
        return 2;
    }
    for (int x = 0; x < 5; x++)
        for (int y = 0; y < 5; y++)
            for (int z = 0; z < w; z++) {
                int dst_x = y, dst_y = (2 * x + 3 * y) % 5, dst_z = (z + RHO[x][y]) % w;
                /* the bit that lands at (dst_x, dst_y, dst_z) comes from (x, y, z); chi then reads its row */
                for (int i = 0; i < 5; i++) {
                    int out = bit((dst_x + 5 - i) % 5, dst_y, dst_z);
                    if (i == 0) src0[out] = bit(x, y, z);
                    if (i == 1) src1[out] = bit(x, y, z);
                    if (i == 2) src2[out] = bit(x, y, z);
                }
            }
    uint64_t sm = seed * 0x2545f4914f6cdd1dULL + 0x1234567;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) rng[i][j] = splitmix(&sm);

    static v512 A[MAXBITS], B[MAXBITS], T[MAXBITS];
    const v512 ones = ~(v512){0};
    uint64_t collisions = 0, follow[MAXK] = {0}, both = 0;
    for (uint64_t it = 0; it < batches; it++) {
        v512 ok = ones;
        for (int j = 0; j < nbits; j++) A[j] = next();
        if (nforce) {
            for (int r = 0; r < nrow; r++) {
                v512 rnd[5];
                for (int i = 0; i < rows[r].nb; i++) rnd[i] = next();
                for (int i = 0; i < 5; i++) {
                    v512 v = (rows[r].base >> i & 1) ? ones : (v512){0};
                    for (int q = 0; q < rows[r].nb; q++)
                        if (rows[r].v[q] >> i & 1) v ^= rnd[q];
                    A[rows[r].j[i]] = v;
                }
            }
            for (int j = 0; j < nbits; j++) B[j] = b1[j] ? ~A[j] : A[j];
            chi(A, T); memcpy(A, T, sizeof(v512) * nbits);
            chi(B, T); memcpy(B, T, sizeof(v512) * nbits);
        } else {
            for (int j = 0; j < nbits; j++) B[j] = dmask[0][j] ? ~A[j] : A[j];
            round_(A, T);
            round_(B, T);
        }
        for (int i = 1; i < k; i++) {      /* after round i: compare with c_i, then absorb step i + 1 */
            v512 bad = {0};
            for (int j = 0; j < nbits; j++) bad |= cmask[i - 1][j] ? ~(A[j] ^ B[j]) : (A[j] ^ B[j]);
            ok &= ~bad;
            for (int q = 0; q < 8; q++) follow[i] += (uint64_t)__builtin_popcountll(ok[q]);
            for (int j = 0; j < nbits; j++) {
                if (xmode == 1 && outer[j]) { v512 r = next(); A[j] ^= r; B[j] ^= r; }
                if (xmode == 2 && xmask[i - 1][j]) { A[j] = ~A[j]; B[j] = ~B[j]; }
                if (dmask[i][j]) B[j] = ~B[j];
            }
            if (i < k - 1) { round_(A, T); round_(B, T); }
        }
        v512 differ = {0};
        for (int j = 0; j < nbits; j++) differ |= A[j] ^ B[j];
        for (int q = 0; q < 8; q++) {
            collisions += (uint64_t)__builtin_popcountll(~differ[q]);
            both += (uint64_t)__builtin_popcountll(~differ[q] & ok[q]);
        }
    }
    printf("{\"trials\": %llu, \"collisions\": %llu, \"follow\": [", (unsigned long long)(batches * 512),
           (unsigned long long)collisions);
    for (int i = 1; i < k; i++) printf("%s%llu", i > 1 ? ", " : "", (unsigned long long)follow[i]);
    printf("], \"collisions_on_trail\": %llu}\n", (unsigned long long)both);
    return 0;
}
