// trails.py's differential trail search, compiled: duo (one lane, trails.py's σ model) and duo2 (two duo lanes that
// absorb one word difference). The schedule comes on stdin, one line "a1 r1 a2 r2" per word position (trails.py
// schedule writes it).
//
//   trails known|secret LANES BOUND T0 T1 [k1=4] [k2=4] [xr=3] [max=6000000] [fold=1] < schedule.txt
//   trails selftest
//
// The model is trails.py's: a lane's state is the additive difference (P, Q) of (y, z); a word costs the signed-digit
// weight of the XOR pattern y's difference shows (bit 31 free) plus that of u's difference σ (every digit), plus, with
// the key secret, m = 31 - lowbit(σ) for the multiple of 2^lowbit that a_t·v(σ) is. One lane takes any σ, and the
// word difference is X_y ⊕ |σ|. Two lanes take one word difference Δp: lane 1's σ1 comes from the lists below, its X_y
// from every form of P1 within the bound, and lane 2's u-difference is then X_y2 ⊕ Δp for every form of P2, with every
// sign pattern; those forms are the ones of weight ≤ c(P) + XR (a cheapest form is all one lane needs, since any σ is
// open to it). With the key known, lane 1's σ comes from two lists (trails.py's): every σ of weight ≤ K1, and every σ
// that reaches a P' with c(P') ≤ K2. A trail with a lane-1 word outside both costs at least K1 + K2 + 2 from lane 1
// alone, so a search that finds nothing below BOUND has no trail below min(BOUND, K1 + K2 + 2 + lane 2's least cost).
// With the key secret every σ is enumerated.
//
// One JSON object on stdout: the cheapest trail from a start in T0..T1 costing less than BOUND (or null), whether the
// search ran to its end (no state cap hit), and its counts.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint32_t u32;
typedef uint64_t u64;

#define NMAX 1024
static int N, LANES, SECRET, BOUND, K1 = 4, K2 = 4, XR = 3, FOLD = 1;
static long MAXST = 6000000;
static u32 A[2][NMAX], AI[2][NMAX];
static int RT[2][NMAX];

static inline u32 rotl(u32 x, int r) {
  r &= 31;
  return r ? (x << r) | (x >> (32 - r)) : x;
}
static inline int pc(u32 x) { return __builtin_popcount(x); }
// x's non-adjacent form, positive and negative digits (x = p - n).
static inline void naf(u64 x, u64* p, u64* n) {
  u64 h = x >> 1, t = x + h, c = h ^ t;
  *p = t & c;
  *n = h & c;
}
// c(x): the least number of nonzero digits of a signed-digit form of x mod 2^32, bit 31 not counted.
static inline int c31(u32 x) {
  u64 p, n;
  naf(x & 0x7FFFFFFFu, &p, &n);
  return pc((u32)((p | n) & 0x7FFFFFFFu));
}
// The same with every digit counted (σ's weight).
static inline int c32(u32 x) {
  u64 p, n;
  naf(x, &p, &n);
  return pc((u32)(p | n));
}
// The least weight of the bits i.. of a form, given the carry into bit i (a lower bound for reps' search).
static inline int crest(u32 x, int i, int carry, int free31) {
  int k = free31 ? 31 - i : 32 - i;
  if (k <= 0) return 0;
  u64 r = ((u64)(x >> i) + carry) & ((1ull << k) - 1), p, n;
  naf(r, &p, &n);
  return __builtin_popcountll((p | n) & ((1ull << k) - 1));
}
static inline u32 sdval(u32 p, u32 n) { return p - n; }
static inline u32 ctz(u32 x) { return __builtin_ctz(x); }

static void* xalloc(size_t n) {
  void* p = calloc(1, n ? n : 1);
  if (!p) {
    fprintf(stderr, "out of memory (%zu)\n", n);
    exit(3);
  }
  return p;
}

// ------------------------------------------------------------------------------------------------------- forms
typedef struct {
  u32 p, n;
  int w;
} Rep;
typedef struct {
  Rep* a;
  int n, cap;
} Reps;
static void reps_push(Reps* r, Rep x) {
  if (r->n == r->cap) {
    r->cap = r->cap ? 2 * r->cap : 64;
    r->a = realloc(r->a, r->cap * sizeof(Rep));
  }
  r->a[r->n++] = x;
}
static void reps_go(Reps* out, u32 x, int i, int carry, u32 p, u32 n, int w, int kmax, int free31) {
  if (w + crest(x, i, carry, free31) > kmax) return;
  if (i == 32) {
    reps_push(out, (Rep){p, n, w});
    return;
  }
  int s = ((x >> i) & 1) + carry;
  if (s != 1) {
    reps_go(out, x, i + 1, s >> 1, p, n, w, kmax, free31);
    return;
  }
  int dw = (i == 31 && free31) ? 0 : 1;
  reps_go(out, x, i + 1, 0, p | (1u << i), n, w + dw, kmax, free31);
  reps_go(out, x, i + 1, 1, p, n | (1u << i), w + dw, kmax, free31);
}
// Every signed-digit form of x mod 2^32 of weight ≤ kmax (into out, cleared first).
static void reps(Reps* out, u32 x, int kmax, int free31) {
  out->n = 0;
  if (kmax >= 0) reps_go(out, x, 0, 0, 0, 0, 0, kmax, free31);
}
// Every XOR pattern of a form of x (bit 31 free) of weight ≤ min(kmax, c(x) + XR), each at its least weight: Rep.p =
// the pattern.
static void patterns(Reps* out, Reps* tmp, u32 x, int kmax) {
  const int cx = c31(x) + XR;
  reps(tmp, x, kmax < cx ? kmax : cx, 1);
  out->n = 0;
  for (int i = 0; i < tmp->n; ++i) {
    u32 X = tmp->a[i].p | tmp->a[i].n;
    int j = 0;
    for (; j < out->n; ++j)
      if (out->a[j].p == X) break;
    if (j < out->n) {
      if (tmp->a[i].w < out->a[j].w) out->a[j].w = tmp->a[i].w;
    } else {
      reps_push(out, (Rep){X, 0, tmp->a[i].w});
    }
  }
}

// ------------------------------------------------------------------------------------------------------- lists
static u32 *L1P, *L1N;
static uint8_t* L1W;
static long NL1, L1END[40];  // L1END[w]: entries of weight ≤ w
static u32* T2V;
static uint8_t* T2C;
static long NT2, T2END[40];

static void l1_go(int start, int left, u32 p, u32 n, int w, long* k, int fill) {
  if (left == 0) {
    if (fill) {
      L1P[*k] = p;
      L1N[*k] = n;
      L1W[*k] = (uint8_t)w;
    }
    ++*k;
    return;
  }
  for (int i = start; i <= 32 - left; ++i) {
    l1_go(i + 1, left - 1, p | (1u << i), n, w, k, fill);
    l1_go(i + 1, left - 1, p, n | (1u << i), w, k, fill);
  }
}
static void build_l1(void) {
  long k = 0;
  for (int w = 1; w <= K1; ++w) l1_go(0, w, 0, 0, w, &k, 0);
  NL1 = k;
  L1P = xalloc(k * 4);
  L1N = xalloc(k * 4);
  L1W = xalloc(k);
  k = 0;
  L1END[0] = 0;
  for (int w = 1; w <= K1; ++w) {
    l1_go(0, w, 0, 0, w, &k, 1);
    L1END[w] = k;
  }
  for (int w = K1 + 1; w < 40; ++w) L1END[w] = k;
}
static int cmp_u32(const void* a, const void* b) {
  u32 x = *(const u32*)a, y = *(const u32*)b;
  return x < y ? -1 : x > y;
}
static u32* t2buf;
static long t2n;
static void t2_go(int start, int left, u32 p, u32 n) {
  t2buf[t2n++] = sdval(p, n);
  t2buf[t2n++] = sdval(p, n) ^ 0x80000000u;
  if (left == 0) return;
  for (int i = start; i < 31; ++i) {
    t2_go(i + 1, left - 1, p | (1u << i), n);
    t2_go(i + 1, left - 1, p, n | (1u << i));
  }
}
static void build_t2(void) {
  long cnt = 1, c = 1;
  for (int j = 1; j <= K2; ++j) {
    c = c * (31 - j + 1) / j;
    cnt += c << j;
  }
  t2buf = xalloc(2 * cnt * 4);
  t2n = 0;
  t2_go(0, K2, 0, 0);
  qsort(t2buf, t2n, 4, cmp_u32);
  long u = 0;
  for (long i = 0; i < t2n; ++i)
    if (!u || t2buf[i] != t2buf[u - 1]) t2buf[u++] = t2buf[i];
  T2V = xalloc(u * 4);
  T2C = xalloc(u);
  long k = 0;
  for (int w = 0; w <= K2; ++w) {
    for (long i = 0; i < u; ++i)
      if (c31(t2buf[i]) == w) {
        T2V[k] = t2buf[i];
        T2C[k++] = (uint8_t)w;
      }
    T2END[w] = k;
  }
  for (int w = K2 + 1; w < 40; ++w) T2END[w] = k;
  NT2 = k;
  free(t2buf);
}

// ------------------------------------------------------------------------------------------------------- edges
typedef struct {
  u32 sp, sn, P, Q;
  uint8_t w, c;
} Edge;
typedef struct {
  Edge* a;
  long n, cap;
} Edges;
static void edge_push(Edges* e, Edge x) {
  if (e->n == e->cap) {
    e->cap = e->cap ? 2 * e->cap : 1024;
    e->a = realloc(e->a, e->cap * sizeof(Edge));
  }
  e->a[e->n++] = x;
}
static inline u32 qnext(u32 sp, u32 sn, int r, u32 P2) { return sdval(rotl(sp, r), rotl(sn, r)) + P2; }

// Lane L's σ at position t from state Q, each with its cost w (digits, plus m with the key secret) and the next state,
// for w + c(P') ≤ lim; σ = 0 is not among them.
static Reps g_tmp;
static void edges_known(Edges* out, int L, int t, u32 Q, int lim) {
  const u32 a = A[L][t], ai = AI[L][t];
  const int r = RT[L][t];
  out->n = 0;
  if (lim < 1) return;
  long end = L1END[lim < K1 ? lim : K1];
  for (long i = 0; i < end; ++i) {
    u32 P2 = a * sdval(L1P[i], L1N[i]) + Q;
    int c = c31(P2);
    if (L1W[i] + c <= lim) edge_push(out, (Edge){L1P[i], L1N[i], P2, qnext(L1P[i], L1N[i], r, P2), L1W[i], (uint8_t)c});
  }
  int cmax = lim - K1 - 1;
  if (cmax < 0) return;
  long tend = T2END[cmax < K2 ? cmax : K2];
  for (long j = 0; j < tend; ++j) {
    u32 v = ai * (T2V[j] - Q);
    int room = lim - T2C[j];
    if (c32(v) > room) continue;
    reps(&g_tmp, v, room, 0);
    for (int k = 0; k < g_tmp.n; ++k) {
      Rep s = g_tmp.a[k];
      if (s.w <= K1) continue;
      edge_push(out, (Edge){s.p, s.n, T2V[j], qnext(s.p, s.n, r, T2V[j]), (uint8_t)s.w, T2C[j]});
    }
  }
}
static void sec_go(Edges* out, int r, u32 Q, int lo, int m, int pos, int left, u32 p, u32 n, int w, int lim) {
  // σ fixed: every α = odd multiple of 2^lo
  for (u64 j = 0; j < (1ull << m); ++j) {
    u32 P2 = (u32)(((2 * j + 1) << lo)) + Q;
    int c = c31(P2);
    if (w + m + c <= lim) edge_push(out, (Edge){p, n, P2, qnext(p, n, r, P2), (uint8_t)(w + m), (uint8_t)c});
  }
  if (left == 0) return;
  for (int i = pos; i < 32; ++i) {
    sec_go(out, r, Q, lo, m, i + 1, left - 1, p | (1u << i), n, w + 1, lim);
    sec_go(out, r, Q, lo, m, i + 1, left - 1, p, n | (1u << i), w + 1, lim);
  }
}
static void edges_secret(Edges* out, int L, int t, u32 Q, int lim) {
  const int r = RT[L][t];
  out->n = 0;
  for (int lo = 31; lo >= 0; --lo) {
    int m = 31 - lo;
    if (m + 1 > lim) break;
    int left = lim - m - 1;
    sec_go(out, r, Q, lo, m, lo + 1, left, 1u << lo, 0, 1, lim);
    sec_go(out, r, Q, lo, m, lo + 1, left, 0, 1u << lo, 1, lim);
  }
}
static void edges(Edges* out, int L, int t, u32 Q, int lim) {
  if (SECRET) edges_secret(out, L, t, Q, lim);
  else edges_known(out, L, t, Q, lim);
}

// ------------------------------------------------------------------------------------------------------- search
typedef struct {
  u32 P[2], Q[2];
  u32 dp, s1p, s1n, s2p, s2n;
  int32_t parent;
  int16_t t;
  uint8_t fresh, g, w, done;
} Node;
static Node* NODES;
static long NN;
static int32_t* HT;
static u64 HMASK;
static int CAPPED;
typedef struct {
  int32_t* a;
  long n, cap;
} Bucket;
static Bucket BK[64];
static long EXPANDED;
static int BEST = 1 << 30;
static long BEST_IDX = -1;
static int BEST_FC = 0;

static inline u64 hkey(const u32* P, const u32* Q, int t, int fresh) {
  u64 h = ((u64)P[0] << 32 | Q[0]) * 0x9E3779B97F4A7C15ull;
  h ^= ((u64)P[1] << 32 | Q[1]) * 0xC2B2AE3D27D4EB4Full;
  h ^= (u64)(t * 2 + fresh) * 0x165667B19E3779F9ull;
  h ^= h >> 29;
  return h * 0xBF58476D1CE4E5B9ull;
}
static void bpush(int g, int32_t idx) {
  Bucket* b = &BK[g];
  if (b->n == b->cap) {
    b->cap = b->cap ? 2 * b->cap : 1024;
    b->a = realloc(b->a, b->cap * 4);
  }
  b->a[b->n++] = idx;
}
static inline int limit(void) { return BEST < BOUND ? BEST : BOUND; }

static void push(int32_t parent, const u32* P, const u32* Q, int t, int fresh, int g, u32 dp, u32 s1p, u32 s1n,
                 u32 s2p, u32 s2n, int w) {
  if (g >= limit()) return;
  u64 h = hkey(P, Q, t, fresh) & HMASK;
  for (;;) {
    int32_t i = HT[h];
    if (i < 0) break;
    Node* x = &NODES[i];
    if (x->t == t && x->fresh == fresh && x->P[0] == P[0] && x->Q[0] == Q[0] && x->P[1] == P[1] && x->Q[1] == Q[1]) {
      if (g < x->g && !x->done) {
        x->g = (uint8_t)g;
        x->parent = parent;
        x->dp = dp;
        x->s1p = s1p;
        x->s1n = s1n;
        x->s2p = s2p;
        x->s2n = s2n;
        x->w = (uint8_t)w;
        bpush(g, i);
      }
      return;
    }
    h = (h + 1) & HMASK;
  }
  if (NN >= MAXST) {
    CAPPED = 1;
    return;
  }
  Node* x = &NODES[NN];
  *x = (Node){{P[0], P[1]}, {Q[0], Q[1]}, dp, s1p, s1n, s2p, s2n, parent, (int16_t)t, (uint8_t)fresh, (uint8_t)g,
              (uint8_t)w, 0};
  HT[h] = (int32_t)NN;
  bpush(g, (int32_t)NN);
  ++NN;
}

// The cheapest block-end fold of a state: duo's receipt y ⊕ rotl(z, 16) unchanged (one lane), duo2's f1 ⊕ rotl(f2, 8)
// with f = y ⊕ rotl(z, 16) (two lanes); -1 if none within lim.
typedef struct {
  u32 f;
  int w;
} FW;
static int cmp_fw(const void* a, const void* b) {
  const FW *x = a, *y = b;
  return x->f != y->f ? (x->f < y->f ? -1 : 1) : x->w - y->w;
}
static int fold(const Node* s, int lim) {
  static Reps a, b, tmp;
  static FW *f1, *f2;
  static long c1, c2;
  long n1 = 0, n2 = 0;
  for (int L = 0; L < LANES; ++L) {
    patterns(&a, &tmp, s->P[L], lim);
    patterns(&b, &tmp, s->Q[L], lim);
    long need = (long)a.n * b.n;
    FW** f = L ? &f2 : &f1;
    long* cap = L ? &c2 : &c1;
    if (need > *cap) {
      *cap = need;
      *f = realloc(*f, need * sizeof(FW));
    }
    long k = 0;
    for (int i = 0; i < a.n; ++i)
      for (int j = 0; j < b.n; ++j) {
        if (a.a[i].w + b.a[j].w > lim) continue;
        u32 x = a.a[i].p ^ rotl(b.a[j].p, 16);
        (*f)[k++] = (FW){L ? rotl(x, 8) : x, a.a[i].w + b.a[j].w};
      }
    if (L) n2 = k;
    else n1 = k;
  }
  int best = -1;
  if (LANES == 1) {
    for (long i = 0; i < n1; ++i)
      if (f1[i].f == 0 && (best < 0 || f1[i].w < best)) best = f1[i].w;
    return best;
  }
  qsort(f2, n2, sizeof(FW), cmp_fw);
  for (long i = 0; i < n1; ++i) {
    long lo = 0, hi = n2;
    while (lo < hi) {
      long mid = (lo + hi) / 2;
      if (f2[mid].f < f1[i].f) lo = mid + 1;
      else hi = mid;
    }
    if (lo < n2 && f2[lo].f == f1[i].f) {
      int w = f1[i].w + f2[lo].w;
      if (w <= lim && (best < 0 || w < best)) best = w;
    }
  }
  return best;
}

static Edges E1;
static Reps R1, R2, RT1;

static void expand1(int32_t idx) {
  Node s = NODES[idx];
  const int g = s.g, t = s.t;
  const int base = s.fresh ? 0 : c31(s.P[0]);
  const int lim = limit() - 1 - g - base;
  if (lim < 0) return;
  u32 X = 0;
  if (!s.fresh) {
    reps(&RT1, s.P[0], base, 1);
    X = RT1.a[0].p | RT1.a[0].n;
    if (c31(s.Q[0]) <= lim) {
      u32 P[2] = {s.Q[0], 0}, Q[2] = {s.Q[0], 0};
      push(idx, P, Q, t + 1, 0, g + base, X, 0, 0, 0, 0, base);
    }
  }
  edges(&E1, 0, t, s.Q[0], lim);
  for (long i = 0; i < E1.n; ++i) {
    Edge e = E1.a[i];
    u32 P[2] = {e.P, 0}, Q[2] = {e.Q, 0};
    push(idx, P, Q, t + 1, 0, g + base + e.w, X ^ (e.sp | e.sn), e.sp, e.sn, 0, 0, base + e.w);
  }
}

static void expand2(int32_t idx) {
  Node s = NODES[idx];
  const int g = s.g, t = s.t;
  const int lim = limit() - 1 - g;
  const int c1 = s.fresh ? 0 : c31(s.P[0]), c2 = s.fresh ? 0 : c31(s.P[1]);
  if (c1 + c2 > lim) return;
  if (s.fresh) {
    R1.n = R2.n = 0;
    reps_push(&R1, (Rep){0, 0, 0});
    reps_push(&R2, (Rep){0, 0, 0});
  } else {
    patterns(&R1, &RT1, s.P[0], lim - c2);
    patterns(&R2, &RT1, s.P[1], lim - c1);
  }
  edges(&E1, 0, t, s.Q[0], lim - c1 - c2);
  if (!s.fresh) edge_push(&E1, (Edge){0, 0, s.Q[0], s.Q[0], 0, (uint8_t)c31(s.Q[0])});
  const u32 a2 = A[1][t];
  const int r2 = RT[1][t];
  const u32 Q2 = s.Q[1];
  for (int i1 = 0; i1 < R1.n; ++i1)
    for (int i2 = 0; i2 < R2.n; ++i2) {
      const int w12 = R1.a[i1].w + R2.a[i2].w;
      if (w12 > lim) continue;
      const u32 X1 = R1.a[i1].p, X2 = R2.a[i2].p;
      for (long k = 0; k < E1.n; ++k) {
        const Edge e = E1.a[k];
        const int cost = w12 + e.w;
        if (cost + e.c > lim) continue;
        const u32 dp = X1 ^ (e.sp | e.sn), S2 = X2 ^ dp;
        const int wt2 = pc(S2), m2 = (SECRET && S2) ? 31 - (int)ctz(S2) : 0;
        const int cost2 = cost + wt2 + m2;
        if (cost2 + e.c > lim) continue;
        if (!S2) {
          const u32 P[2] = {e.P, Q2}, Q[2] = {e.Q, Q2};
          if (cost2 + e.c + c31(Q2) <= lim) push(idx, P, Q, t + 1, 0, g + cost2, dp, e.sp, e.sn, 0, 0, cost2);
          continue;
        }
        int bitpos[32], nb = 0;
        for (u32 x = S2; x; x &= x - 1) bitpos[nb++] = ctz(x);
        for (u32 mask = 0; mask < (1u << nb); ++mask) {
          u32 p2 = 0, n2 = 0;
          for (int b = 0; b < nb; ++b) {
            if ((mask >> b) & 1) n2 |= 1u << bitpos[b];
            else p2 |= 1u << bitpos[b];
          }
          const u32 rq = sdval(rotl(p2, r2), rotl(n2, r2));
          if (!SECRET) {
            const u32 P2 = a2 * sdval(p2, n2) + Q2;
            if (cost2 + e.c + c31(P2) > lim) continue;
            const u32 P[2] = {e.P, P2}, Q[2] = {e.Q, rq + P2};
            push(idx, P, Q, t + 1, 0, g + cost2, dp, e.sp, e.sn, p2, n2, cost2);
          } else {
            const int lo = ctz(S2);
            for (u64 j = 0; j < (1ull << m2); ++j) {
              const u32 P2 = (u32)((2 * j + 1) << lo) + Q2;
              if (cost2 + e.c + c31(P2) > lim) continue;
              const u32 P[2] = {e.P, P2}, Q[2] = {e.Q, rq + P2};
              push(idx, P, Q, t + 1, 0, g + cost2, dp, e.sp, e.sn, p2, n2, cost2);
            }
          }
        }
      }
    }
}

static int selftest(void) {
  // c31 and c32 against the carry DP of trails.py's cmin (and its every-digit variant)
  u64 st = 0x1234567;
  int bad = 0;
  for (int it = 0; it < 2000000; ++it) {
    st = st * 6364136223846793005ull + 1442695040888963407ull;
    u32 x = (u32)(st >> 32);
    if (it % 3 == 1) x &= (u32)(st >> 7) & (u32)(st >> 13);
    if (it % 3 == 2) x |= (u32)(st >> 7) | (u32)(st >> 13);
    for (int full = 0; full < 2; ++full) {
      int c0 = 0, c1 = 99, nb = full ? 32 : 31;
      for (int i = 0; i < nb; ++i) {
        int b = (x >> i) & 1, n0, n1;
        if (b) {
          n0 = c0 + 1;
          n1 = c0 + 1 < c1 ? c0 + 1 : c1;
        } else {
          n0 = c0 < c1 + 1 ? c0 : c1 + 1;
          n1 = c1 + 1;
        }
        c0 = n0;
        c1 = n1;
      }
      int want = c0 < c1 ? c0 : c1, got = full ? c32(x) : c31(x);
      if (want != got) {
        if (bad < 5) fprintf(stderr, "x=%08x full=%d dp=%d naf=%d\n", x, full, want, got);
        ++bad;
      }
    }
    if (it < 20000) {
      Reps r = {0};
      int k = (it % 4) + c32(x);
      reps(&r, x, k, 0);
      for (int i = 0; i < r.n; ++i)
        if (sdval(r.a[i].p, r.a[i].n) != x || r.a[i].w != pc(r.a[i].p | r.a[i].n) || r.a[i].w > k) ++bad;
      if (!r.n) ++bad;
      free(r.a);
    }
  }
  printf("{\"selftest\":%s,\"bad\":%d}\n", bad ? "false" : "true", bad);
  return bad != 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "selftest")) return selftest();
  if (argc < 6) {
    fprintf(stderr, "usage: trails known|secret LANES BOUND T0 T1 [k1=] [k2=] [xr=] [max=] [fold=] < schedule\n");
    return 2;
  }
  SECRET = !strcmp(argv[1], "secret");
  LANES = atoi(argv[2]);
  BOUND = atoi(argv[3]);
  const int t0 = atoi(argv[4]), t1 = atoi(argv[5]);
  for (int i = 6; i < argc; ++i) {
    const char* v = strchr(argv[i], '=');
    if (!v) return 2;
    ++v;
    if (!strncmp(argv[i], "k1=", 3)) K1 = atoi(v);
    else if (!strncmp(argv[i], "k2=", 3)) K2 = atoi(v);
    else if (!strncmp(argv[i], "xr=", 3)) XR = atoi(v);
    else if (!strncmp(argv[i], "max=", 4)) MAXST = atol(v);
    else if (!strncmp(argv[i], "fold=", 5)) FOLD = atoi(v);
    else return 2;
  }
  if (LANES < 1 || LANES > 2 || BOUND < 1 || BOUND > 60 || K1 < 1 || K1 > 6 || K2 < 0 || K2 > 6) return 2;
  unsigned long a1, r1, a2, r2;
  while (N < NMAX && scanf("%lu %lu %lu %lu", &a1, &r1, &a2, &r2) == 4) {
    A[0][N] = (u32)a1 | 1;
    RT[0][N] = (int)r1;
    A[1][N] = (u32)a2 | 1;
    RT[1][N] = (int)r2;
    for (int L = 0; L < 2; ++L) {
      u32 x = A[L][N];
      for (int i = 0; i < 5; ++i) x *= 2 - A[L][N] * x;
      AI[L][N] = x;
    }
    ++N;
  }
  if (!SECRET) {
    build_l1();
    build_t2();
  }
  u64 hs = 1;
  while (hs < (u64)MAXST * 2) hs <<= 1;
  HMASK = hs - 1;
  HT = xalloc(hs * 4);
  memset(HT, 0xFF, hs * 4);
  NODES = xalloc((size_t)MAXST * sizeof(Node));
  const u32 Z[2] = {0, 0};
  for (int t = t0; t <= t1 && t < N; ++t) push(-1, Z, Z, t, 1, 0, 0, 0, 0, 0, 0, 0);
  for (int g = 0; g < limit(); ++g) {
    for (long k = 0; k < BK[g].n && g < limit(); ++k) {
      const int32_t idx = BK[g].a[k];
      Node* s = &NODES[idx];
      if (s->g != g || s->done) continue;
      s->done = 1;
      if (!s->fresh && !s->P[0] && !s->Q[0] && !s->P[1] && !s->Q[1]) {
        BEST = g;
        BEST_IDX = idx;
        BEST_FC = -1;
        break;
      }
      if (s->t >= N) {
        if (FOLD && !s->fresh) {
          const int fc = fold(s, limit() - 1 - g);
          if (fc >= 0 && g + fc < BEST) {
            BEST = g + fc;
            BEST_IDX = idx;
            BEST_FC = fc;
          }
        }
        continue;
      }
      ++EXPANDED;
      if (LANES == 1) expand1(idx);
      else expand2(idx);
    }
  }
  printf("{\"key\":\"%s\",\"lanes\":%d,\"bound\":%d,\"t0\":%d,\"t1\":%d,\"k1\":%d,\"k2\":%d,\"xr\":%d,\"fold\":%d,"
         "\"complete\":%s,\"expanded\":%ld,\"states\":%ld,\"max_states\":%ld,\"trail\":",
         SECRET ? "secret" : "known", LANES, BOUND, t0, t1, K1, K2, XR, FOLD, CAPPED ? "false" : "true", EXPANDED, NN,
         MAXST);
  if (BEST_IDX < 0) {
    printf("null}\n");
    return 0;
  }
  int path[NMAX + 2], np = 0;
  for (long i = BEST_IDX; i >= 0 && np < NMAX + 2; i = NODES[i].parent) path[np++] = (int)i;
  printf("{\"cost\":%d,\"end\":\"%s\",\"fold_cost\":%d,\"end_t\":%d,\"steps\":[", BEST, BEST_FC < 0 ? "collision" : "fold",
         BEST_FC < 0 ? 0 : BEST_FC, NODES[BEST_IDX].t);
  for (int j = np - 2; j >= 0; --j) {
    const Node* x = &NODES[path[j]];
    const Node* pr = &NODES[path[j + 1]];
    printf("%s{\"t\":%d,\"dp\":%u,\"s1p\":%u,\"s1n\":%u,\"s2p\":%u,\"s2n\":%u,\"w\":%d,\"P\":[%u,%u],\"Q\":[%u,%u]}",
           j == np - 2 ? "" : ",", pr->t, x->dp, x->s1p, x->s1n, x->s2p, x->s2n, x->w, x->P[0], x->P[1], x->Q[0],
           x->Q[1]);
  }
  printf("]}}\n");
  return 0;
}
