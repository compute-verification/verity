// The k = 2 partial-completion reducer, CPU, tuned: leading-word (53-bit) GSO drives a 4-dimensional LLL whose basis
// updates are exact (GMP), then a Cramer/Babai step over the reduced basis recovers the word's low x bits. The same
// algorithm is ported to CUDA in k2lll.cuh / k2_gpu.cu; this build is the fast CPU baseline and the bit-exact oracle.
//
//   k2_cpu reduce VECFILE [REPS]   recover every word in VECFILE (gen_vectors.py), check against the true word, and
//                                  time the reduction+Babai against one GMP modular square root; one JSON line.
//
// The completion is the red team's 4D linearization (`partial_root_k2.py`): the stored word is y = ρ(ρ(a) + c′), the
// adversary keeps y's top b - x bits A and its parity s2 and knows (a, c′); with y = A + u, u < 2^x, a monic quartic
// f(u) ≡ 0 (mod p) linearizes to a 4D lattice whose reduction plus Babai gives u, hence y.
#include <gmp.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

namespace {

constexpr int D = 4;  // the lattice dimension

// value of an mpz as a long double m * 2^e, 53-bit mantissa (the leading word the decisions run on).
inline long double approx(const mpz_t z) {
  long e = 0;
  double m = mpz_get_d_2exp(&e, z);
  return ldexpl((long double)m, (int)e);
}

struct Lattice {
  mpz_t b[D][D];  // row i is basis vector i
  long double G[D][D], mu[D][D], B[D];

  Lattice() {
    for (int i = 0; i < D; ++i)
      for (int j = 0; j < D; ++j) mpz_init(b[i][j]);
  }
  ~Lattice() {
    for (int i = 0; i < D; ++i)
      for (int j = 0; j < D; ++j) mpz_clear(b[i][j]);
  }

  void gram() {  // exact Gram matrix once (mpz dot products), then kept incrementally in long double
    mpz_t s, t;
    mpz_init(s);
    mpz_init(t);
    for (int i = 0; i < D; ++i)
      for (int j = 0; j <= i; ++j) {
        mpz_set_ui(s, 0);
        for (int u = 0; u < D; ++u) mpz_addmul(s, b[i][u], b[j][u]);
        G[i][j] = G[j][i] = approx(s);
      }
    mpz_clear(s);
    mpz_clear(t);
  }

  void gso() {  // Gram-Schmidt from the Gram matrix (long double)
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j < i; ++j) {
        long double s = G[i][j];
        for (int t = 0; t < j; ++t) s -= mu[j][t] * mu[i][t] * B[t];
        mu[i][j] = s / B[j];
      }
      long double s = G[i][i];
      for (int t = 0; t < i; ++t) s -= mu[i][t] * mu[i][t] * B[t];
      B[i] = s;
    }
  }
};

// b[k] -= q * b[j]  (exact); the caller refreshes the Gram afterwards.
inline void submul(Lattice &L, int k, int j, const mpz_t q) {
  for (int t = 0; t < D; ++t) mpz_submul(L.b[k][t], q, L.b[j][t]);
}

// LLL with delta = 0.99, decisions from the leading-word GSO, updates exact.  Returns the step counts.
void lll(Lattice &L, long long &swaps, long long &sizes) {
  const long double delta = 0.99L;
  mpz_t q;
  mpz_init(q);
  L.gram();
  L.gso();
  int k = 1;
  while (k < D) {
    for (int j = k - 1; j >= 0; --j) {
      long double m = L.mu[k][j];
      long long qi = llroundl(m);
      if (qi != 0) {
        ++sizes;
        mpz_set_si(q, (long)qi);
        submul(L, k, j, q);
        L.gram();
        L.gso();
      }
    }
    if (L.B[k] >= (delta - L.mu[k][k - 1] * L.mu[k][k - 1]) * L.B[k - 1]) {
      ++k;
    } else {
      for (int t = 0; t < D; ++t) mpz_swap(L.b[k][t], L.b[k - 1][t]);
      ++swaps;
      L.gram();
      L.gso();
      k = k > 1 ? k - 1 : 1;
    }
  }
  mpz_clear(q);
}

// 3x3 determinant of the entries b[rows[.]][cols[.]] into r.
void det3(const mpz_t b[D][D], const int rows[3], const int cols[3], mpz_t r) {
  mpz_t t, u;
  mpz_init(t);
  mpz_init(u);
  mpz_set_ui(r, 0);
  // r = b00(b11 b22 - b12 b21) - b01(b10 b22 - b12 b20) + b02(b10 b21 - b11 b20)
  int s = 1;
  for (int c = 0; c < 3; ++c) {
    int c1 = (c + 1) % 3, c2 = (c + 2) % 3;
    mpz_mul(t, b[rows[1]][cols[c1]], b[rows[2]][cols[c2]]);
    mpz_submul(t, b[rows[1]][cols[c2]], b[rows[2]][cols[c1]]);
    mpz_mul(u, b[rows[0]][cols[c]], t);
    if (s > 0)
      mpz_add(r, r, u);
    else
      mpz_sub(r, r, u);
    s = -s;
  }
  mpz_clear(t);
  mpz_clear(u);
}

// round(N / Dn) for signed N and positive Dn.
void round_div(const mpz_t N, const mpz_t Dn, mpz_t out) {
  mpz_t two_n, two_d;
  mpz_init(two_n);
  mpz_init(two_d);
  mpz_mul_2exp(two_n, N, 1);
  mpz_mul_2exp(two_d, Dn, 1);
  if (mpz_sgn(two_n) >= 0)
    mpz_add(two_n, two_n, Dn);
  else
    mpz_sub(two_n, two_n, Dn);
  mpz_fdiv_q(out, two_n, two_d);
  mpz_clear(two_n);
  mpz_clear(two_d);
}

struct Word {
  mpz_t A, cp, a, y;  // A = top bits << x, c', a, true word
  int s2;
  Word() {
    mpz_init(A);
    mpz_init(cp);
    mpz_init(a);
    mpz_init(y);
  }
  ~Word() {
    mpz_clear(A);
    mpz_clear(cp);
    mpz_clear(a);
    mpz_clear(y);
  }
};

// build the k = 2 lattice for stored sign s2; returns the quartic coefficients f1,f2,f3 (last column) in f[].
void build(Lattice &L, const Word &w, int x, const mpz_t p, int s2, mpz_t f1, mpz_t f2, mpz_t f3) {
  mpz_t A2, A3, t;
  mpz_init(A2);
  mpz_init(A3);
  mpz_init(t);
  mpz_mul(A2, w.A, w.A);
  mpz_mod(A2, A2, p);
  mpz_mul(A3, A2, w.A);
  mpz_mod(A3, A3, p);
  // f3 = 4A ; f2 = 6A^2 - 2 s2 c' ; f1 = 4A^3 - 4 s2 c' A
  mpz_mul_ui(f3, w.A, 4);
  mpz_mod(f3, f3, p);
  mpz_mul_ui(f2, A2, 6);
  mpz_set(t, w.cp);
  mpz_mul_ui(t, t, 2);
  if (s2 > 0)
    mpz_sub(f2, f2, t);
  else
    mpz_add(f2, f2, t);
  mpz_mod(f2, f2, p);
  mpz_mul_ui(f1, A3, 4);
  mpz_mul(t, w.cp, w.A);
  mpz_mul_ui(t, t, 4);
  if (s2 > 0)
    mpz_sub(f1, f1, t);
  else
    mpz_add(f1, f1, t);
  mpz_mod(f1, f1, p);
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < D; ++j) mpz_set_ui(L.b[i][j], 0);
  mpz_ui_pow_ui(L.b[0][0], 2, 3 * x);
  mpz_ui_pow_ui(L.b[1][1], 2, 2 * x);
  mpz_ui_pow_ui(L.b[2][2], 2, x);
  mpz_sub(L.b[0][3], p, f1);  // store -f_i mod p as p - f_i
  mpz_sub(L.b[1][3], p, f2);
  mpz_sub(L.b[2][3], p, f3);
  mpz_set(L.b[3][3], p);
  mpz_clear(A2);
  mpz_clear(A3);
  mpz_clear(t);
}

// from a reduced basis, Babai-round the target (0,0,0,f0) and read u from coord 0; return true and set y on success.
bool babai(Lattice &L, const Word &w, int x, const mpz_t p, int s2, mpz_t y_out) {
  mpz_t det, cof, coeff[D], coord0, u, f0, A2, A4, t, w0, half, y, y1, lhs, lim;
  mpz_inits(det, cof, coord0, u, f0, A2, A4, t, w0, half, y, y1, lhs, lim, nullptr);
  mpz_ui_pow_ui(lim, 2, x);
  for (int i = 0; i < D; ++i) mpz_init(coeff[i]);
  int all[D] = {0, 1, 2, 3};
  // det of the 4x4 reduced basis, cofactor expansion along column 3
  mpz_set_ui(det, 0);
  for (int i = 0; i < D; ++i) {
    int rows[3], ri = 0;
    for (int r = 0; r < D; ++r)
      if (r != i) rows[ri++] = r;
    int cols[3] = {0, 1, 2};
    det3(L.b, rows, cols, cof);
    mpz_mul(t, L.b[i][3], cof);
    if (((i + 3) & 1) == 0)
      mpz_add(det, det, t);
    else
      mpz_sub(det, det, t);
  }
  int sign = mpz_sgn(det);
  if (sign == 0) {
    for (int i = 0; i < D; ++i) mpz_clear(coeff[i]);
    mpz_clears(det, cof, coord0, u, f0, A2, A4, t, w0, half, y, y1, lhs, lim, nullptr);
    return false;
  }
  mpz_abs(det, det);
  mpz_ui_pow_ui(w0, 2, 3 * x);
  mpz_fdiv_q_2exp(half, w0, 1);
  mpz_mul(A2, w.A, w.A);
  mpz_mod(A2, A2, p);
  mpz_mul(A4, A2, A2);
  mpz_mod(A4, A4, p);
  (void)all;
  bool ok = false;
  for (int si = 0; si < 2 && !ok; ++si) {
    int s1 = si == 0 ? 1 : -1;
    // f0 = A^4 - 2 s2 c' A^2 + c'^2 - s1 a   (mod p)
    mpz_set(f0, A4);
    mpz_mul(t, w.cp, A2);
    mpz_mul_ui(t, t, 2);
    if (s2 > 0)
      mpz_sub(f0, f0, t);
    else
      mpz_add(f0, f0, t);
    mpz_mul(t, w.cp, w.cp);
    mpz_add(f0, f0, t);
    if (s1 > 0)
      mpz_sub(f0, f0, w.a);
    else
      mpz_add(f0, f0, w.a);
    mpz_mod(f0, f0, p);
    // coeff[i] = round( f0 * C_{i,3} / det ), C_{i,3} = (-1)^{i+3} minor(i,3)
    for (int i = 0; i < D; ++i) {
      int rows[3], ri = 0;
      for (int r = 0; r < D; ++r)
        if (r != i) rows[ri++] = r;
      int cols[3] = {0, 1, 2};
      det3(L.b, rows, cols, cof);  // minor(i,3)
      if (((i + 3) & 1) != 0) mpz_neg(cof, cof);
      mpz_mul(t, f0, cof);
      if (sign < 0) mpz_neg(t, t);  // we took |det|; fold det's sign into the numerator
      round_div(t, det, coeff[i]);
    }
    // coord0 = sum coeff[i] * b[i][0]
    mpz_set_ui(coord0, 0);
    for (int i = 0; i < D; ++i) mpz_addmul(coord0, coeff[i], L.b[i][0]);
    // u = floor((coord0 + w0/2)/w0)
    mpz_add(u, coord0, half);
    mpz_fdiv_q(u, u, w0);
    for (int du = 0; du < 3 && !ok; ++du) {
      int d = du == 0 ? 0 : (du == 1 ? -1 : 1);
      mpz_t uu;
      mpz_init_set(uu, u);
      if (d < 0)
        mpz_sub_ui(uu, uu, 1);
      else if (d > 0)
        mpz_add_ui(uu, uu, 1);
      if (mpz_sgn(uu) >= 0 && mpz_cmp(uu, lim) < 0) {
        mpz_add(y, w.A, uu);  // y = A + u
        if (mpz_cmp(y, p) < 0) {
          // y1 = (s2 y^2 - c') mod p ; check s1 y1^2 == a and parity rules
          mpz_mul(y1, y, y);
          mpz_mod(y1, y1, p);
          if (s2 < 0) mpz_neg(y1, y1);
          mpz_sub(y1, y1, w.cp);
          mpz_mod(y1, y1, p);
          mpz_mul(lhs, y1, y1);
          mpz_mod(lhs, lhs, p);
          if (s1 < 0) mpz_neg(lhs, lhs);
          mpz_mod(lhs, lhs, p);
          bool par_y = (mpz_even_p(y) != 0) == (s2 > 0);
          bool par_y1 = (mpz_even_p(y1) != 0) == (s1 > 0);
          if (par_y && par_y1 && mpz_cmp(lhs, w.a) == 0) {
            mpz_set(y_out, y);
            ok = true;
          }
        }
      }
      mpz_clear(uu);
    }
  }
  for (int i = 0; i < D; ++i) mpz_clear(coeff[i]);
  mpz_clears(det, cof, coord0, u, f0, A2, A4, t, w0, half, y, y1, lhs, lim, nullptr);
  return ok;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3 || strcmp(argv[1], "reduce")) {
    fprintf(stderr, "usage: k2_cpu reduce VECFILE [REPS]\n");
    return 2;
  }
  FILE *fp = fopen(argv[2], "r");
  if (!fp) {
    perror("open");
    return 2;
  }
  int b, cc, x, nwords, k;
  if (fscanf(fp, "%d %d %d %d %d", &b, &cc, &x, &nwords, &k) != 5) return 2;
  int reps = argc > 3 ? atoi(argv[3]) : 1;
  mpz_t p, e;
  mpz_init(p);
  mpz_init(e);
  mpz_ui_pow_ui(p, 2, b);
  mpz_sub_ui(p, p, cc);
  mpz_add_ui(e, p, 1);
  mpz_fdiv_q_2exp(e, e, 2);  // (p+1)/4
  std::vector<Word *> words;
  char buf[4096];
  for (int i = 0; i < nwords; ++i) {
    Word *w = new Word();
    char sA[2048], scp[2048], sa[2048], sy[2048];
    int s2;
    if (fscanf(fp, "%s %s %s %s %d", sA, scp, sa, sy, &s2) != 5) {
      fprintf(stderr, "short read at %d\n", i);
      return 2;
    }
    mpz_set_str(w->A, sA, 16);
    mpz_set_str(w->cp, scp, 16);
    mpz_set_str(w->a, sa, 16);
    mpz_set_str(w->y, sy, 16);
    w->s2 = s2;
    // the generator may append extra round constants (k >= 3); consume the rest of the line
    if (fgets(buf, sizeof buf, fp)) {
    }
    words.push_back(w);
  }
  fclose(fp);

  // one modular square root's time (GMP powmod), averaged
  mpz_t r, base;
  mpz_init(r);
  mpz_init(base);
  int root_reps = 200;
  struct timespec t0, t1;
  mpz_set_ui(base, 123456789);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < root_reps; ++i) {
    mpz_add_ui(base, base, 2);
    mpz_powm(r, base, e, p);
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double root_us = ((t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) * 1e-3) / root_reps;

  // reduce + Babai, timed, bit-exact against the true word
  long long swaps = 0, sizes = 0;
  int completed = 0;
  mpz_t y_out, f1, f2, f3;
  mpz_inits(y_out, f1, f2, f3, nullptr);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int rep = 0; rep < reps; ++rep) {
    completed = 0;
    for (auto *w : words) {
      Lattice L;
      build(L, *w, x, p, w->s2, f1, f2, f3);
      lll(L, swaps, sizes);
      if (babai(L, *w, x, p, w->s2, y_out) && mpz_cmp(y_out, w->y) == 0) ++completed;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double total_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) * 1e-3;
  double complete_us = total_us / (reps * (double)nwords);
  printf(
      "{\"impl\": \"cpu-gmp-leadword\", \"bits\": %d, \"c\": %d, \"x\": %d, \"k\": %d, \"words\": %d, \"completed\": %d, "
      "\"root_us\": %.3f, \"complete_us\": %.3f, \"complete_over_root\": %.4f, \"claim_over_root\": %.4f, "
      "\"margin\": %.2f, \"swaps_per_word\": %.1f, \"size_steps_per_word\": %.1f}\n",
      b, cc, x, k, nwords, completed, root_us, complete_us, complete_us / root_us, 2.0 * x / b,
      (complete_us / root_us) / (2.0 * x / b), (double)swaps / (reps * nwords), (double)sizes / (reps * nwords));
  mpz_clears(y_out, f1, f2, f3, r, base, p, e, nullptr);
  for (auto *w : words) delete w;
  return 0;
}
