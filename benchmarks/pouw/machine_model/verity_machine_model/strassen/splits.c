/* The split recursion of sched.py, in place on one table pair: T (any schedule) and H (one with a scheme).
   Row-major G x G int64. rest[i*G+e] is the rest's index when g[i] is split at digit index e (e < i, g[e] >= 2). */
#include <stdint.h>

static inline int64_t mn(int64_t a, int64_t b) { return a < b ? a : b; }

void splits(int G, int64_t *T, int64_t *H, const int64_t *rest, const int64_t *g, int64_t big) {
    int two2 = -1;
    for (int e = 0; e < G; e++)
        if (g[e] == 2) two2 = e;
    for (int i = 0; i < G; i++) {
        for (int j = 0; j < G; j++) {
            int64_t t = T[(int64_t)i * G + j], h = H[(int64_t)i * G + j];
            const int64_t *ri = rest + (int64_t)i * G;
            for (int e = 0; e < i; e++) {
                if (g[e] < 2) continue;
                int64_t r = ri[e];
                int64_t te = T[(int64_t)e * G + j], tr = T[r * G + j];
                t = mn(t, te + tr);
                h = mn(h, mn(H[(int64_t)e * G + j] + tr, te + H[r * G + j]));
            }
            const int64_t *rj = rest + (int64_t)j * G;
            const int64_t *Ti = T + (int64_t)i * G, *Hi = H + (int64_t)i * G;
            for (int f = 0; f < j; f++) {
                if (g[f] < 2) continue;
                int64_t r = rj[f];
                t = mn(t, Ti[f] + Ti[r]);
                h = mn(h, mn(Hi[f] + Ti[r], Ti[f] + Hi[r]));
            }
            /* a split whose digit alone covers the range, its other part holding the scheme: at least the whole
               product's cost plus a scheme on a 2-valued part (two2 is the grid index of 2) */
            if (two2 >= 0 && i > two2) h = mn(h, t + H[(int64_t)two2 * G + j]);
            if (two2 >= 0 && j > two2) h = mn(h, t + Hi[two2]);
            T[(int64_t)i * G + j] = mn(t, big);
            H[(int64_t)i * G + j] = mn(h, big);
        }
    }
}
