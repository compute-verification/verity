// The CPU check of layout.h, the lanes, sources and fold ssn.cu's kernel serves, against N-EXACT's Python on its own test
// vectors (fold_vectors.py records them). Two levels:
//   fold    each recorded magic_epilogue call (the final word D, rho_i, the column term c, s_j, s_i): bf16(xu_fold(D,
//           0x4B400000 - rho_i, -c, s_j, -1.5 2^23 s_j, s_i)) equals its outputs bit for bit, out-of-range words included;
//           where A.B^T is recorded the IADD3 reads it back and the fold serves stock's outputs (expect 1), or, for
//           fold_extreme's mutation (an output of 2^22 + 1, expect -1), doesn't read it back.
//   layout  each recorded operands call (A, B, a1, a_t, u, G and the lane coins, odd in [-127, 127] a (step, tile,
//           column)) on blc4a-t64-e7-s1 under the call's lane-coin weights (ROT=1, or BLC_W=1:127:1), twice:
//           XU=1   ssn's final word from lane_a and lane_b, plus its row source and its column source for (tile, row
//                  class i mod 2, column), is 0x4B400000 + A_i . B_j mod 2^32 (the IADD3), and the fold serves stock's bf16
//                  outputs bit for bit where |A_i . B_j| <= 2^22.
//           FOLD=1 (the lead, served noisy) the final word plus its column source (row source 0) is 0x4B400000 + A_i . B_j
//                  + n_ij, n_ij = -beta_j + sum_e a1[i, e] C[e, j] + sum_e T(i)[e] G[e, j] + sum_{t >= 2} (1 + a_t[i])
//                  from the recorded values, and the fold serves stock's epilogue on A_i . B_j + n_ij where it is within
//                  2^22 (FOLD's S is the Python's S, or one less at K 8,192, where xu adds a step; the first S steps of
//                  the recorded coins are used).
//           The recorded a1, a_t, u and G enter as the certifier feeds them (layout.h's row_words, u word and g_word) and
//           are read back through the feed's format (rc_bit, u0_of..u2_of, g_of).
//           ssn's lanes differ from the Python's (their positions; kappa 0; e by its own rule), which changes the
//           words and not what is served, so the check is on what is served. As a control, the other row
//           class's column source is also tried, and under ROT=1 it must serve wrong outputs.
//   draw    G's rejection draw (g_word_of) over all 2^32 words: each of the 2,401 G in [-3, 3]^4 from exactly
//           floor(2^32 / 2401) words, and the other 872 rejected.
// Build: g++ -O2 -std=c++17 -ffp-contract=off cpu_check.cc -o cpu_check. Usage: cpu_check DIR (fold_vectors.py's --out).
// One JSON line a vector and a summary line; exit 1 on any mismatch.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "layout.h"

struct Arr {
  char t = 0;
  std::vector<uint64_t> dims;
  std::vector<uint8_t> raw;
  size_t size() const {
    size_t s = 1;
    for (uint64_t d : dims) s *= d;
    return s;
  }
  int64_t i(size_t k) const {
    switch (t) {
      case 'b': return (int8_t)raw[k];
      case 'H': return ((const uint16_t*)raw.data())[k];
      case 'i': return ((const int32_t*)raw.data())[k];
      case 'q': return ((const int64_t*)raw.data())[k];
    }
    fprintf(stderr, "ERR integer read of dtype %c\n", t);
    exit(2);
  }
  float f(size_t k) const {
    if (t != 'f') {
      fprintf(stderr, "ERR float read of dtype %c\n", t);
      exit(2);
    }
    return ((const float*)raw.data())[k];
  }
};
typedef std::map<std::string, Arr> Vec;

static Vec load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "ERR open %s\n", path.c_str());
    exit(2);
  }
  Vec v;
  uint32_t len;
  while (f.read((char*)&len, 4)) {
    std::string name(len, '\0');
    Arr a;
    uint32_t nd;
    f.read(&name[0], len);
    f.read(&a.t, 1);
    f.read((char*)&nd, 4);
    a.dims.resize(nd);
    f.read((char*)a.dims.data(), 8 * nd);
    const size_t w = a.t == 'b' ? 1 : a.t == 'H' ? 2 : a.t == 'i' || a.t == 'f' ? 4 : 8;
    a.raw.resize(a.size() * w);
    f.read((char*)a.raw.data(), a.raw.size());
    if (!f) {
      fprintf(stderr, "ERR short read of %s in %s\n", name.c_str(), path.c_str());
      exit(2);
    }
    v[name] = std::move(a);
  }
  return v;
}

static uint16_t serve(uint32_t d, uint32_t r, uint32_t c, float sj, float si) {
  return bf16_rn(xu_fold(d, r, c, sj, -12582912.0f * sj, si));
}

// fold level. With A.B^T recorded, the IADD3's word read as fp32 less 1.5 2^23 is the integer the fold serves: expect 1
// (every output in the window) it is A.B^T everywhere and the outputs are stock's; expect -1 (the mutation) it isn't
// somewhere, and the served outputs' differences from stock's are reported (an integer off by one at 2^22 is below bf16's
// precision, so they can be stock's)
static bool fold_check(const std::string& label, const Vec& v, int expect) {
  const Arr &acc = v.at("acc"), &rho = v.at("rho"), &col = v.at("col"), &sc = v.at("s_col"), &sr = v.at("s_row"),
            &mg = v.at("magic");
  const Arr* st = v.count("stock") ? &v.at("stock") : nullptr;
  const Arr* p = v.count("p") ? &v.at("p") : nullptr;
  const size_t m = acc.dims[0], n = acc.dims[1];
  size_t bad = 0, vs_stock = 0, past_int32 = 0, misread = 0;
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      const int64_t d = acc.i(i * n + j);
      past_int32 += d < INT32_MIN || d > INT32_MAX;
      const uint32_t r = XU_MAGIC_BITS - (uint32_t)rho.i(i), c = (uint32_t)-col.i(i * n + j), b = (uint32_t)d + r + c;
      const uint16_t y = serve((uint32_t)d, r, c, sc.f(j), sr.f(i));
      bad += y != (uint16_t)mg.i(i * n + j);
      if (st) vs_stock += y != (uint16_t)st->i(i * n + j);
      if (p) {
        float f;
        memcpy(&f, &b, 4);
        misread += (double)f - 12582912.0 != (double)p->i(i * n + j);
      }
    }
  const bool ok = !bad && !past_int32 && (expect != 1 || (p && st && !misread && !vs_stock)) && (expect != -1 || (p && misread));
  printf("{\"level\": \"fold\", \"vector\": \"%s\", \"outputs\": %zu, \"mismatches_vs_python_magic\": %zu, "
         "\"words_past_int32\": %zu, \"ab_recorded\": %s, \"expect\": %d, \"read_back_not_ab\": %zu, "
         "\"served_differ_from_stock\": %zu, \"pass\": %s}\n",
         label.c_str(), m * n, bad, past_int32, p ? "true" : "false", expect, misread, vs_stock, ok ? "true" : "false");
  return ok;
}

static bool layout_check(const std::string& label, const Vec& v, const std::string& wname, bool xu) {
  const Arr &a = v.at("a"), &b = v.at("b"), &a1 = v.at("a1"), &at = v.at("at"), &u0 = v.at("u0"), &u1 = v.at("u1"),
            &u2 = v.at("u2"), &g = v.at("g"), &coins = v.at("coins"), &wts = v.at("weights"), &meta = v.at("meta"),
            &sc = v.at("s_col"), &sr = v.at("s_row"), &p = v.at("p"), &st = v.at("stock");
  const int m = (int)a.dims[0], K = (int)a.dims[1], n = (int)b.dims[1];
  g_tag = true, g_blc = 2, g_xu = xu, g_fold = !xu, g_rot = wname == "rot";
  bool wok = true;
  for (int c = 0; c < 2; c++)
    for (int e = 0; e < 3; e++) {
      if (g_rot) wok &= wts.i(3 * c + e) == ROT_W[c][e];
      else g_lw[e] = (int)wts.i(e), wok &= wts.i(3 * c + e) == wts.i(e);
    }
  const int S = steps_of(K), py_s = (int)meta.i(1), tile = (int)meta.i(3), tiles = (int)coins.dims[2];
  const bool shape_ok = wok && (xu ? S == py_s : S <= py_s && S >= py_s - 1) && tile == 64 &&
                        coins.dims[0] == (uint64_t)py_s && coins.dims[1] == 3 && tiles == (m + 63) / 64 &&
                        coins.dims[3] == (uint64_t)n && at.dims[1] == (uint64_t)py_s;
  if (!shape_ok) {
    printf("{\"level\": \"layout\", \"vector\": \"%s\", \"layout\": \"%s\", \"weights_match\": %s, \"S\": %d, "
           "\"python_S\": %d, \"tile\": %d, \"pass\": false}\n",
           label.c_str(), layout_name(), wok ? "true" : "false", S, py_s, tile);
    return false;
  }
  const int H = coin_words(S), xa = xu_at(K, S), xe = xu_e(xa);
  // the private values as the certifier feeds them and the forming reads them: a1 bits 0-3 and a_t - 1 bit t + 3 of the
  // row's feed words, the u word
  std::vector<uint32_t> h((size_t)m * H, 0);
  std::vector<RowCoins> rc(m);
  size_t bad_private = 0;
  std::vector<int> bt(S, 0);
  for (int i = 0; i < m; i++) {
    int y1[4];
    for (int e = 0; e < 4; e++) y1[e] = (int)a1.i(4 * i + e), bad_private += y1[e] != 0 && y1[e] != 1;
    bad_private += at.i((size_t)i * py_s) != 0;
    for (int t = 1; t < S; t++) {
      const int64_t x = at.i((size_t)i * py_s + t);
      bad_private += x != 1 && x != 2;
      bt[t] = (int)x - 1;
    }
    row_words(y1, bt.data(), S, &h[(size_t)i * H]);
    const int64_t x0 = u0.i(i), x1 = u1.i(i), x2 = u2.i(i);
    bad_private += x0 < 0 || x0 > 127 || x1 < -128 || x1 > 127 || x2 < -128 || x2 > 127;
    rc[i] = RowCoins{&h[(size_t)i * H], (uint32_t)x0 | ((uint32_t)x1 & 0xFFu) << 8 | ((uint32_t)x2 & 0xFFu) << 16};
  }
  std::vector<int8_t> A((size_t)m * K), Bt((size_t)n * K);
  for (size_t x = 0; x < A.size(); x++) A[x] = (int8_t)a.i(x);
  for (int k = 0; k < K; k++)
    for (int j = 0; j < n; j++) Bt[(size_t)j * K + k] = (int8_t)b.i((size_t)k * n + j);
  // G through its feed word
  std::vector<int> G(4 * (size_t)n);
  for (int j = 0; j < n; j++) {
    int x[4];
    for (int e = 0; e < 4; e++) x[e] = (int)g.i((size_t)e * n + j), bad_private += x[e] < -3 || x[e] > 3;
    const uint32_t w = g_word(x);
    for (int e = 0; e < 4; e++) G[4 * j + e] = g_of(w, e), bad_private += G[4 * j + e] != x[e];
  }
  // the certifier's lane-coin word of (step t, tile q, column j): the three coins and coin_word_top
  auto lw = [&](int t, int q, int j) {
    uint32_t w = coin_word_top(t, j);
    for (int e = 0; e < 3; e++) w |= (uint32_t)(uint8_t)(int8_t)coins.i((((size_t)t * 3 + e) * tiles + q) * n + j) << (8 * e);
    return w;
  };
  std::vector<int> ap((size_t)m * S * 32), bp((size_t)tiles * n * S * 32);
  for (int i = 0; i < m; i++)
    for (int t = 0; t < S; t++)
      for (int l = 0; l < 32; l++) ap[((size_t)i * S + t) * 32 + l] = lane_a(i, t, l, K, S, &A[(size_t)i * K], &rc[i]);
  for (int q = 0; q < tiles; q++)
    for (int j = 0; j < n; j++)
      for (int t = 0; t < S; t++)
        for (int l = 0; l < 32; l++)
          bp[(((size_t)q * n + j) * S + t) * 32 + l] = lane_b(j, t, l, K, S, &Bt[(size_t)j * K], &G[4 * j], true, lw(t, q, j));
  size_t bad_p = 0, bad_exact = 0, bad_served = 0, past = 0, bad_stock_model = 0, wrong_class = 0, ab_max = 0;
  long long max_word = 0, max_col = 0;
  for (int i = 0; i < m; i++) {
    const int q = i / 64;
    const uint32_t r = xu ? xu_row_source(rc[i], K, S) : 0u;
    long long walk = 0;
    for (int t = 1; t < S; t++) walk += at.i((size_t)i * py_s + t);
    for (int j = 0; j < n; j++) {
      long long acc = -beta_of(j, S), ab = 0, nz = walk - beta_of(j, S);
      for (int e = 0; e < 4; e++) nz += a1.i(4 * i + e) * code_of(j, e) + tag_of(i, e) * G[4 * j + e];
      for (int k = 0; k < K; k++) ab += (int)A[(size_t)i * K + k] * (int)Bt[(size_t)j * K + k];
      for (int t = 0; t < S; t++) {
        const int* x = &ap[((size_t)i * S + t) * 32];
        const int* y = &bp[(((size_t)q * n + j) * S + t) * 32];
        for (int l = 0; l < 32; l++) acc += (long long)x[l] * y[l];
        max_word = std::max(max_word, acc < 0 ? -acc : acc);
      }
      const int64_t pij = p.i((size_t)i * n + j);
      bad_p += ab != pij;
      int c[2];
      long long sig[2];
      for (int cls = 0; cls < 2; cls++) {
        sig[cls] = xu_sigma(cls, S, [&](int t) { return lw(t, q, j); });
        c[cls] = col_source(beta_of(j, S), (int)sig[cls]);
        max_col = std::max(max_col, std::abs(xu ? (long long)c[cls] : -sig[cls]));
      }
      const uint32_t d = (uint32_t)acc, ci = (uint32_t)c[i & 1];
      const long long tgt = xu ? pij : pij + nz;
      bad_exact += (uint32_t)(d + r + ci) != (uint32_t)(XU_MAGIC_BITS + (uint32_t)tgt);
      ab_max = std::max(ab_max, (size_t)(tgt < 0 ? -tgt : tgt));
      const float sj = sc.f(j), si = sr.f(i);
      const uint16_t py_stock = (uint16_t)st.i((size_t)i * n + j);
      bad_stock_model += bf16_rn(stock_fold((int)pij, sj, si)) != py_stock;
      const uint16_t want = xu ? py_stock : bf16_rn(stock_fold((int)tgt, sj, si));
      if (tgt < -(1 << 22) || tgt > (1 << 22)) {
        past++;
        continue;
      }
      bad_served += serve(d, r, ci, sj, si) != want;
      wrong_class += serve(d, r, (uint32_t)c[(i & 1) ^ 1], sj, si) != want;
    }
  }
  const bool ok = !bad_private && !bad_p && !bad_exact && !bad_served && !bad_stock_model && max_word < (1ll << 31) &&
                  (!g_rot || wrong_class);
  printf("{\"level\": \"layout\", \"vector\": \"%s\", \"layout\": \"%s\", \"m\": %d, \"n\": %d, \"K\": %d, \"S\": %d, "
         "\"python_S\": %d, \"python_e\": %d, \"xu_lane\": %d, \"xu_e\": %d, \"weights\": [[%d, %d, %d], [%d, %d, %d]], "
         "\"private_value_errors\": %zu, \"ab_mismatches_vs_python\": %zu, \"served_against\": \"%s\", "
         "\"max_abs_folded\": %zu, \"max_abs_word\": %lld, \"max_abs_col_term\": %lld, \"inexact_after_iadd3\": %zu, "
         "\"served_outputs\": %zu, \"served_mismatches\": %zu, \"stock_model_mismatches_vs_python\": %zu, "
         "\"past_2_22\": %zu, \"other_class_source_mismatches\": %zu, \"pass\": %s}\n",
         label.c_str(), layout_name(), m, n, K, S, py_s, (int)meta.i(2), xu ? xa : -1, xu ? xe : 0, coin_w(0, 0),
         coin_w(0, 1), coin_w(0, 2), coin_w(1, 0), coin_w(1, 1), coin_w(1, 2), bad_private, bad_p,
         xu ? "python stock on A.B" : "stock_fold on A.B + noise", ab_max, max_word, max_col, bad_exact,
         (size_t)m * n - past, bad_served, bad_stock_model, past, wrong_class, ok ? "true" : "false");
  return ok;
}

// every 32-bit word through g_word_of: G's index sum_e (G[e] + 3) 7^e counted over the kept words
static bool draw_check() {
  std::vector<uint64_t> hits(2401, 0);
  uint64_t rejected = 0, bad = 0;
  for (uint64_t u = 0; u < (1ull << 32); u++) {
    uint32_t w;
    if (!g_word_of((uint32_t)u, &w)) {
      rejected++;
      continue;
    }
    int v = 0;
    for (int e = 3; e >= 0; e--) {
      const int d = g_of(w, e);
      bad += d < -3 || d > 3;
      v = 7 * v + d + 3;
    }
    hits[v]++;
  }
  const uint64_t each = (1ull << 32) / 2401;
  uint64_t lo = hits[0], hi = hits[0];
  for (uint64_t x : hits) lo = std::min(lo, x), hi = std::max(hi, x);
  const bool ok = !bad && lo == each && hi == each && rejected == (1ull << 32) - 2401 * each && G_KEEP == 2401 * each;
  printf("{\"level\": \"draw\", \"words\": %llu, \"kept_per_g_min\": %llu, \"kept_per_g_max\": %llu, "
         "\"kept_per_g_want\": %llu, \"rejected\": %llu, \"digits_out_of_range\": %llu, \"pass\": %s}\n",
         1ull << 32, (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)each, (unsigned long long)rejected,
         (unsigned long long)bad, ok ? "true" : "false");
  fflush(stdout);
  return ok;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: cpu_check DIR (fold_vectors.py's --out)\n");
    return 2;
  }
  const std::string dir = argv[1];
  std::ifstream idx(dir + "/index.txt");
  if (!idx) {
    fprintf(stderr, "ERR open %s/index.txt\n", dir.c_str());
    return 2;
  }
  std::string line;
  int fold_n = 0, fold_ok = 0, lay_n = 0, lay_ok = 0;
  while (std::getline(idx, line)) {
    std::istringstream s(line);
    std::string kind, label, file, w;
    int expect = 0;
    if (!(s >> kind >> label >> file >> w >> expect)) continue;
    const Vec v = load(dir + "/" + file);
    if (kind == "fold") fold_n++, fold_ok += fold_check(label, v, expect);
    else if (kind == "layout")
      for (const bool xu : {true, false}) lay_n++, lay_ok += layout_check(label, v, w, xu);
    fflush(stdout);
  }
  const bool draw_ok = draw_check();
  const bool ok = fold_n && lay_n && fold_ok == fold_n && lay_ok == lay_n && draw_ok;
  printf("{\"summary\": true, \"fold_vectors\": %d, \"fold_pass\": %d, \"layout_checks\": %d, \"layout_pass\": %d, "
         "\"draw_pass\": %s, \"pass\": %s}\n", fold_n, fold_ok, lay_n, lay_ok, draw_ok ? "true" : "false", ok ? "true" : "false");
  return ok ? 0 : 1;
}
