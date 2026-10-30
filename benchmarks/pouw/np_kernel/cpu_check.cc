// The CPU check of layout.h, the lanes, sources, fold words and fold ssn.cu's kernel serves. Every served output is held
// to verity_int8's epilogue on A_i . B_j (layout.h's int8_epilogue, int8_exact.dequantize's order) under scales with
// 24-bit significands. Three levels need nothing but layout.h:
//   ragged  the lead (blc4a-t64-e7-s1 under ROT=1 and FOLD=1) on calls off the tiles (decode's one row; rows and columns
//           off them), served as ssn.cu serves them: computed over whole 64-row and BN-column tiles with zero data past
//           the call, the certifier's feed over pad_rows(m) rows and pad_cols(n) columns. Every committed row past the
//           call (a padding row) carries its tag, its lane-coin A' and its own coins, read from its feed words as the
//           forming reads them; every word is the statement's, -beta_j + a1 . C + T . G + sum_t (1 + b_t) + the lane
//           coins under the row's A' (plus A_i . B_j, 0 on a padding row); a computed row past the committed ones reads
//           zero coins; and the served output, xu_fold on the word with the row's words (fold_row on the words the
//           forming read), the column source beta_j - sigma and the G and C bytes, is int8_epilogue on A_i . B_j within
//           the bound. A control forms the padding rows from zero words (padding without coins), which the level must
//           reject.
//   window  the fold on p = A_i . B_j at and around +-bound and across the window, each word carrying every coin term
//           (-beta_j, a1 . C, T . G, the walk and sigma over twice the lane coins' range) of a random row, column, S and
//           coins: fold_word reads back p and the served bf16 is int8_epilogue on p; at bound + 1 and -(bound + 1) it
//           reads back another integer, so the bound is the fold's.
//   draw    G's rejection draw (g_word_of) over all 2^32 words: each of the 2,401 G in [-3, 3]^4 from exactly
//           floor(2^32 / 2401) words, and the other 872 rejected.
// served:FILE holds the lead to verity_int8's own y on one call: FILE holds the call as verity_int8 serves it (x, the
//           activations quantized a row, and s_a; w, the weight quantized a row, and s_w; their int32 product z and the
//           served y, from int8_exact; the suite's test writes them at Llama-3.2-1B's linears) and a coin seed. A' and B'
//           are formed from the certifier's coins (drawn from the seed), every final word comes from the lanes, the fold
//           must read back z, and each served bf16 must be y's bit for bit. Controls, which must serve outputs other
//           than y: the same words through 83d82109a8's fold (row source 0, column source 0x4B400000 - sigma, so n_ij =
//           A_i . B_j's word less sigma less A_i . B_j stays in; FFMA(v, s_j, fl(-1.5 2^23 s_j)), then FMUL s_i), and
//           each of its departures alone: n_ij left in, the FFMA, the weight's scale first.
// DIR (fold_vectors.py's --out) reads the lead's Python layout's own test vectors:
//   layout  each recorded operands call (A, B, a1, a_t, u, G and the lane coins, odd in [-127, 127] a (step, tile,
//           column)) on blc4a-t64-e7-s1 under the call's lane-coin weights (ROT=1, or BLC_W=1:127:1), twice, XU=1 and
//           FOLD=1: ssn's final word from lane_a and lane_b, with its row's words (XU=1 0x4B400000 - rho_i alone, FOLD=1
//           fold_row) and its column source for (tile, row class i mod 2, column), folds to 0x4B400000 + A_i . B_j mod
//           2^32 (fold_word), and the fold serves int8_epilogue on the recorded A . B^T under the recorded scales where
//           |A_i . B_j| <= 2^22 (FOLD's S is the Python's S, or one less at K 8,192, where xu adds a step; the first S
//           steps of the recorded coins are used). The recorded a1, a_t, u and G enter as the certifier feeds them
//           (layout.h's row_words, u word and g_word) and are read back through the feed's format (rc_bit, u0_of..u2_of,
//           g_of). ssn's lanes differ from the Python's (their positions; kappa 0; e by its own rule), which changes the
//           words and not what is served, so the check is on what is served. As a control, the other row class's
//           column source is also tried, and under ROT=1 it must serve wrong outputs.
// Build: g++ -O2 -std=c++17 -ffp-contract=off cpu_check.cc -o cpu_check.
// Usage: cpu_check [--bound B] LEVEL...: ragged, window, draw, served:FILE, or DIR (fold_vectors.py's --out); B is the
// fold's bound, 2^22 unless given (serving.LEAD_FOLD_BOUND, which the suite's test passes). One JSON line a vector or
// call and a summary line; exit 1 on any mismatch.
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

static uint64_t mix(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// a scale with a 24-bit significand, as quantize_rows' absmax / 127 gives
static float scale24(uint64_t r, int lo) {
  return std::ldexp((float)((1u << 23) + (uint32_t)(r & ((1u << 23) - 1))), -23 - lo - (int)((r >> 32) % 4));
}
static double read_back(uint32_t q) {
  float f;
  memcpy(&f, &q, 4);
  return (double)f - 12582912.0;
}
// G[0..3, j] as B' holds them: step 1's lanes 24-27 of the column's lanes b, where the epilogue reads them
static uint32_t g_bytes(const int8_t* b) {
  uint32_t w = 0;
  for (int e = 0; e < 4; e++) w |= (uint32_t)(uint8_t)b[24 + e] << (8 * e);
  return w;
}
// a final word: -beta_j plus the lanes' dot product over the S steps (int32 a step, as the wgmma's k32 adds)
static long long word_of(const int8_t* a, const int8_t* b, int S, int beta) {
  long long w = -beta;
  for (int t = 0; t < S; t++) {
    int s = 0;
    for (int l = 0; l < 32; l++) s += (int)a[32 * t + l] * (int)b[32 * t + l];
    w += s;
  }
  return w;
}
// the certifier's row words over R rows (a1 four raw bits, each later step's bit raw) and G words over C columns by
// rejection
static std::vector<uint32_t> draw_feed(int R, int S, int C, uint64_t& x, size_t* redraws) {
  const int H = coin_words(S);
  std::vector<uint32_t> feed((size_t)R * H);
  std::vector<int> bits(S);
  for (int i = 0; i < R; i++) {
    int a1[4];
    const uint64_t r = mix(x);
    for (int e = 0; e < 4; e++) a1[e] = (int)(r >> e) & 1;
    for (int t = 1; t < S; t++) bits[t] = (int)(mix(x) & 1);
    row_words(a1, bits.data(), S, &feed[(size_t)i * H]);
  }
  for (int j = 0; j < C; j++) {
    uint32_t w;
    while (!g_word_of((uint32_t)mix(x), &w)) ++*redraws;
    feed.push_back(w);
  }
  return feed;
}
// the lane-coin words of `tiles` row tiles, S steps and n columns: three odd coins in [-127, 127] and coin_word_top
static std::vector<uint32_t> draw_lane_coins(int tiles, int S, int n, uint64_t& x) {
  std::vector<uint32_t> lc((size_t)tiles * S * n);
  for (int q = 0; q < tiles; q++)
    for (int t = 0; t < S; t++)
      for (int j = 0; j < n; j++) {
        const uint64_t r = mix(x);
        uint32_t w = coin_word_top(t, j);
        for (int e = 0; e < 3; e++) w |= (uint32_t)(uint8_t)(int8_t)(2 * (int)((r >> (8 * e)) & 127) - 127) << (8 * e);
        lc[((size_t)q * S + t) * n + j] = w;
      }
  return lc;
}

static bool layout_check(const std::string& label, const Vec& v, const std::string& wname, bool xu) {
  const Arr &a = v.at("a"), &b = v.at("b"), &a1 = v.at("a1"), &at = v.at("at"), &u0 = v.at("u0"), &u1 = v.at("u1"),
            &u2 = v.at("u2"), &g = v.at("g"), &coins = v.at("coins"), &wts = v.at("weights"), &meta = v.at("meta"),
            &sc = v.at("s_col"), &sr = v.at("s_row"), &p = v.at("p");
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
  std::vector<int8_t> ap((size_t)m * S * 32), bp((size_t)tiles * n * S * 32);
  for (int i = 0; i < m; i++)
    for (int t = 0; t < S; t++)
      for (int l = 0; l < 32; l++)
        ap[((size_t)i * S + t) * 32 + l] = (int8_t)lane_a(i, t, l, K, S, &A[(size_t)i * K], &rc[i]);
  for (int q = 0; q < tiles; q++)
    for (int j = 0; j < n; j++)
      for (int t = 0; t < S; t++)
        for (int l = 0; l < 32; l++)
          bp[(((size_t)q * n + j) * S + t) * 32 + l] =
              (int8_t)lane_b(j, t, l, K, S, &Bt[(size_t)j * K], &G[4 * j], true, lw(t, q, j));
  size_t bad_p = 0, bad_exact = 0, bad_served = 0, past = 0, wrong_class = 0, ab_max = 0;
  long long max_word = 0, max_col = 0;
  for (int i = 0; i < m; i++) {
    const int q = i / 64;
    const FoldRow fr = xu ? FoldRow{xu_row_source(rc[i], K, S), 0u, 0u, 0u} : fold_row(rc[i].h, i, S);
    for (int j = 0; j < n; j++) {
      const int8_t* y = &bp[((size_t)q * n + j) * S * 32];
      const long long acc = word_of(&ap[(size_t)i * S * 32], y, S, beta_of(j, S));
      long long ab = 0;
      for (int k = 0; k < K; k++) ab += (int)A[(size_t)i * K + k] * (int)Bt[(size_t)j * K + k];
      max_word = std::max(max_word, acc < 0 ? -acc : acc);
      const int64_t pij = p.i((size_t)i * n + j);
      bad_p += ab != pij;
      int c[2];
      for (int cls = 0; cls < 2; cls++) {
        c[cls] = col_source(beta_of(j, S), xu_sigma(cls, S, [&](int t) { return lw(t, q, j); }));
        max_col = std::max(max_col, std::abs((long long)c[cls]));
      }
      const uint32_t d = (uint32_t)acc, g4 = g_bytes(y), cj = code_word(j);
      bad_exact += fold_word(d, fr.r, (uint32_t)c[i & 1], fr.nt, g4, fr.na, cj) != XU_MAGIC_BITS + (uint32_t)pij;
      ab_max = std::max(ab_max, (size_t)(pij < 0 ? -pij : pij));
      const float sj = sc.f(j), si = sr.f(i);
      if (pij < -(1 << 22) || pij > (1 << 22)) {
        past++;
        continue;
      }
      const uint16_t want = bf16_rn(int8_epilogue((int)pij, si, sj));
      auto serve = [&](int cc) { return bf16_rn(xu_fold(d, fr.r, (uint32_t)cc, fr.nt, g4, fr.na, cj, si, sj)); };
      bad_served += serve(c[i & 1]) != want;
      wrong_class += serve(c[(i & 1) ^ 1]) != want;
    }
  }
  const bool ok = !bad_private && !bad_p && !bad_exact && !bad_served && max_word < (1ll << 31) && (!g_rot || wrong_class);
  printf("{\"level\": \"layout\", \"vector\": \"%s\", \"layout\": \"%s\", \"m\": %d, \"n\": %d, \"K\": %d, \"S\": %d, "
         "\"python_S\": %d, \"python_e\": %d, \"xu_lane\": %d, \"xu_e\": %d, \"weights\": [[%d, %d, %d], [%d, %d, %d]], "
         "\"private_value_errors\": %zu, \"ab_mismatches_vs_python\": %zu, \"served_against\": \"int8_epilogue(A.B)\", "
         "\"max_abs_ab\": %zu, \"max_abs_word\": %lld, \"max_abs_col_source\": %lld, \"inexact_after_fold\": %zu, "
         "\"served_outputs\": %zu, \"served_mismatches\": %zu, \"past_2_22\": %zu, \"other_class_source_mismatches\": %zu, "
         "\"pass\": %s}\n",
         label.c_str(), layout_name(), m, n, K, S, py_s, (int)meta.i(2), xu ? xa : -1, xu ? xe : 0, coin_w(0, 0),
         coin_w(0, 1), coin_w(0, 2), coin_w(1, 0), coin_w(1, 1), coin_w(1, 2), bad_private, bad_p, ab_max, max_word,
         max_col, bad_exact, (size_t)m * n - past, bad_served, past, wrong_class, ok ? "true" : "false");
  return ok;
}

static void lead_layout() { g_tag = true, g_blc = 2, g_rot = true, g_fold = true, g_xu = false; }

// ragged level: one call of cm rows and cn columns on the lead, served over m x n computed (BN-column tiles); with control
// the forming reads row i's coins only below cm, so the padding rows get zero words
static bool ragged_check(int cm, int cn, int K, int BN, long long bound, uint64_t seed, bool control) {
  lead_layout();
  const int m = (cm + 63) / 64 * 64, n = (cn + BN - 1) / BN * BN, S = steps_of(K), H = coin_words(S), tiles = m / 64;
  const int R = pad_rows(cm), C = pad_cols(cn), formed = control ? cm : R;
  uint64_t x = seed;
  // the certifier's feed, every coin drawn raw: row words over the R committed rows, then G words over the C committed
  // columns by rejection
  size_t redraws = 0;
  const std::vector<uint32_t> feed = draw_feed(R, S, C, x, &redraws);
  const bool feed_ok = (long long)feed.size() == feed_words(R, S, C);
  const std::vector<uint32_t> zero(H, 0u);
  // a1[i, e] and b_t[i] as the certifier drew them (0 past the committed rows), G[e, j] (0 past the committed columns)
  auto fbit = [&](int i, int p) { return i < R ? (int)((feed[(size_t)i * H + (p >> 5)] >> (p & 31)) & 1u) : 0; };
  auto gval = [&](int j, int e) { return j < C ? g_of(feed[(size_t)R * H + j], e) : 0; };
  const std::vector<uint32_t> lc = draw_lane_coins(tiles, S, n, x);
  auto lw = [&](int q, int t, int j) { return lc[((size_t)q * S + t) * n + j]; };
  // data: random within the call, zero past it
  std::vector<int8_t> A((size_t)m * K, 0), Bt((size_t)n * K, 0);
  for (int i = 0; i < cm; i++)
    for (int k = 0; k < K; k++) A[(size_t)i * K + k] = (int8_t)mix(x);
  for (int j = 0; j < cn; j++)
    for (int k = 0; k < K; k++) Bt[(size_t)j * K + k] = (int8_t)mix(x);
  // A' as the forming writes it (row i's words from the feed below `formed`, else zero words), with the row's words the
  // forming writes from the same feed words, and B' per row tile
  std::vector<int8_t> ap((size_t)m * S * 32), bp((size_t)tiles * n * S * 32);
  std::vector<FoldRow> fr(m);
  for (int i = 0; i < m; i++) {
    const uint32_t* h = i < formed ? &feed[(size_t)i * H] : zero.data();
    const RowCoins rc{h, 0u};
    fr[i] = fold_row(h, i, S);
    for (int t = 0; t < S; t++)
      for (int l = 0; l < 32; l++)
        ap[((size_t)i * S + t) * 32 + l] = (int8_t)lane_a(i, t, l, K, S, &A[(size_t)i * K], &rc);
  }
  for (int q = 0; q < tiles; q++)
    for (int j = 0; j < n; j++) {
      int G[4];
      for (int e = 0; e < 4; e++) G[e] = gval(j, e);
      for (int t = 0; t < S; t++)
        for (int l = 0; l < 32; l++)
          bp[(((size_t)q * n + j) * S + t) * 32 + l] = (int8_t)lane_b(j, t, l, K, S, &Bt[(size_t)j * K], G, true, lw(q, t, j));
    }
  // every committed row's non-data lanes against its role (padding rows: tag, lane-coin A', its own a1 and step bits); a
  // computed row past R: zero coins
  size_t pad_wrong = 0, past_wrong = 0, tagless = 0;
  for (int i = cm; i < m; i++) {
    bool wrong = false, tag = false;
    for (int t = 0; t < S; t++)
      for (int l = 0; l < 32; l++) {
        const int v = ap[((size_t)i * S + t) * 32 + l];
        int want = 0;
        if (kidx(t, l, K) >= 0) want = 0;
        else if (l >= 28 && l < 31) want = ROT_W[i & 1][l - 28];
        else if (t == 0 && l >= 24 && l < 28) want = tag_of(i, l - 24), tag |= want != 0;
        else if (t == 0 && ((l >= 21 && l < 24) || l == 31)) want = fbit(i, l == 31 ? 3 : l - 21);
        else if (t > 0 && l == 31) want = 1 + fbit(i, t + 3);
        wrong |= v != want;
      }
    if (i < R) pad_wrong += wrong, tagless += !tag;
    else past_wrong += wrong;
  }
  // every computed word against the statement's, and the served outputs of the committed ones
  size_t bad_word = 0, served = 0, bad_served = 0, past = 0;
  for (int i = 0; i < m; i++) {
    const int q = i / 64;
    const float si = scale24(mix(x), 4);
    for (int j = 0; j < n; j++) {
      const int8_t* b = &bp[((size_t)q * n + j) * S * 32];
      const long long word = word_of(&ap[(size_t)i * S * 32], b, S, beta_of(j, S));
      long long want = -beta_of(j, S), ab = 0;
      for (int t = 1; t < S; t++) want += 1 + fbit(i, t + 3);
      for (int k = 0; k < K; k++) ab += (long long)A[(size_t)i * K + k] * Bt[(size_t)j * K + k];
      for (int e = 0; e < 4; e++) want += fbit(i, e) * code_of(j, e) + tag_of(i, e) * gval(j, e);
      const int sigma = xu_sigma(i & 1, S, [&](int t) { return lw(q, t, j); });
      want += ab + sigma;
      bad_word += word != want;
      if (i >= R || j >= C) continue;
      if (ab < -bound || ab > bound) {
        past++;
        continue;
      }
      served++;
      const float sj = scale24(mix(x), 7);
      const uint32_t c = (uint32_t)col_source(beta_of(j, S), sigma);
      bad_served += bf16_rn(xu_fold((uint32_t)word, fr[i].r, c, fr[i].nt, g_bytes(b), fr[i].na, code_word(j), si, sj)) !=
                    bf16_rn(int8_epilogue((int)ab, si, sj));
    }
  }
  const bool honest = feed_ok && !pad_wrong && !tagless && !past_wrong && !bad_word && !bad_served;
  const bool ok = control ? R > cm && pad_wrong && bad_word : honest;
  printf("{\"level\": \"ragged\", \"layout\": \"%s\", \"control\": %s, \"call_m\": %d, \"call_n\": %d, \"K\": %d, \"S\": %d, "
         "\"BN\": %d, \"committed_m\": %d, \"committed_n\": %d, \"computed_m\": %d, \"computed_n\": %d, \"feed_words\": %zu, "
         "\"g_redraws\": %zu, \"padding_rows\": %d, \"padding_rows_wrong\": %zu, \"padding_rows_tagless\": %zu, "
         "\"uncommitted_rows_wrong\": %zu, \"word_mismatches\": %zu, \"served_against\": \"int8_epilogue(A.B)\", "
         "\"served_outputs\": %zu, \"served_mismatches\": %zu, \"past_bound\": %zu, \"bound\": %lld, \"pass\": %s}\n",
         layout_name(), control ? "true" : "false", cm, cn, K, S, BN, R, C, m, n, feed.size(), redraws, R - cm, pad_wrong,
         tagless, past_wrong, bad_word, served, bad_served, past, bound, ok ? "true" : "false");
  fflush(stdout);
  return ok;
}

// window level: for each p, a random row i, column j, S, row words, G word and sigma; the word d = p + n_ij + sigma with
// n_ij = -beta_j + a1 . C + T . G + the walk, as the lead's GEMM leaves it, and the fold's words from the same coins
static bool window_check(long long bound, uint64_t seed) {
  lead_layout();
  uint64_t x = seed;
  std::vector<long long> ps = {0, 1, -1};
  for (long long k = 0; k < 64; k++) ps.insert(ps.end(), {bound - k, -bound + k});
  for (int r = 0; r < 4096; r++) ps.push_back((long long)(mix(x) % (uint64_t)(2 * bound + 1)) - bound);
  long long max_n = 0;
  auto fold_of = [&](long long p, float si, float sj, uint32_t* q) {
    const int i = (int)(mix(x) % 28560), j = (int)(mix(x) % 28560), S = 1 + (int)(mix(x) % 300), H = coin_words(S);
    std::vector<uint32_t> h(H);
    for (uint32_t& w : h) w = (uint32_t)mix(x);
    uint32_t g;
    while (!g_word_of((uint32_t)mix(x), &g)) {
    }
    const long long sigma = (long long)(mix(x) % (1ull << 24)) - (1ll << 23);
    long long n = -beta_of(j, S) + walk_of(h.data(), S);
    for (int e = 0; e < 4; e++) n += (int)((h[0] >> e) & 1u) * code_of(j, e) + tag_of(i, e) * g_of(g, e);
    max_n = std::max(max_n, n < 0 ? -n : n);
    const FoldRow fr = fold_row(h.data(), i, S);
    const uint32_t d = (uint32_t)(p + n + sigma), c = (uint32_t)col_source(beta_of(j, S), (int)sigma), cj = code_word(j);
    *q = fold_word(d, fr.r, c, fr.nt, g, fr.na, cj);
    return bf16_rn(xu_fold(d, fr.r, c, fr.nt, g, fr.na, cj, si, sj));
  };
  size_t misread = 0, bad = 0;
  for (const long long p : ps) {
    const float sj = scale24(mix(x), 7), si = scale24(mix(x), 4);
    uint32_t q;
    bad += fold_of(p, si, sj, &q) != bf16_rn(int8_epilogue((int)p, si, sj));
    misread += read_back(q) != (double)p;
  }
  uint32_t qa, qb;
  fold_of(bound + 1, 1.0f, 1.0f, &qa);
  fold_of(-(bound + 1), 1.0f, 1.0f, &qb);
  const double above = read_back(qa), below = read_back(qb);
  const bool ok = !misread && !bad && above != (double)(bound + 1) && below != (double)-(bound + 1);
  printf("{\"level\": \"window\", \"layout\": \"%s\", \"bound\": %lld, \"outputs\": %zu, \"max_abs_n_ij\": %lld, "
         "\"read_back_not_p\": %zu, \"served_against\": \"int8_epilogue(A.B)\", \"served_differ_from_int8\": %zu, "
         "\"read_back_at_bound_plus_1\": %.1f, \"read_back_at_minus_bound_minus_1\": %.1f, \"pass\": %s}\n",
         layout_name(), bound, ps.size(), max_n, misread, bad, above, below, ok ? "true" : "false");
  fflush(stdout);
  return ok;
}

// 83d82109a8's FOLD=1 epilogue on the IADD3's word b: FFMA(v, s_j, fl(-1.5 2^23 s_j)), then FMUL s_i
static uint16_t fold_83d(uint32_t b, float si, float sj) {
  float v;
  memcpy(&v, &b, 4);
  return bf16_rn(std::fma(v, sj, -12582912.0f * sj) * si);
}
// the fold's FADD with the weight's scale first: FADD(v, -1.5 2^23), FMUL s_j, then FMUL s_i
static uint16_t fold_sj_first(uint32_t b, float si, float sj) {
  float v;
  memcpy(&v, &b, 4);
  const float f = (v - 12582912.0f) * sj;
  return bf16_rn(f * si);
}

// served level: verity_int8's call in `path` served by the lead
static bool served_check(const std::string& path, long long bound) {
  lead_layout();
  const Vec v = load(path);
  const Arr &xa = v.at("x"), &sa = v.at("s_a"), &wa = v.at("w"), &sw = v.at("s_w"), &za = v.at("z"), &ya = v.at("y");
  const int m = (int)xa.dims[0], K = (int)xa.dims[1], n = (int)wa.dims[0];
  const bool shape_ok = xa.t == 'b' && wa.t == 'b' && za.t == 'i' && ya.t == 'H' && wa.dims[1] == (uint64_t)K &&
                        sa.size() == (size_t)m && sw.size() == (size_t)n && za.size() == (size_t)m * n &&
                        ya.size() == (size_t)m * n;
  if (!shape_ok) {
    printf("{\"level\": \"served\", \"file\": \"%s\", \"shapes_match\": false, \"pass\": false}\n", path.c_str());
    return false;
  }
  const int S = steps_of(K), H = coin_words(S), tiles = (m + 63) / 64, R = pad_rows(m), C = pad_cols(n);
  uint64_t x = (uint64_t)v.at("seed").i(0);
  size_t redraws = 0;
  const std::vector<uint32_t> feed = draw_feed(R, S, C, x, &redraws);
  const std::vector<uint32_t> lc = draw_lane_coins(tiles, S, n, x);
  const int8_t *A = (const int8_t*)xa.raw.data(), *W = (const int8_t*)wa.raw.data();
  std::vector<int8_t> ap((size_t)m * S * 32), bp((size_t)n * S * 32);
  std::vector<FoldRow> fr(m);
  for (int i = 0; i < m; i++) {
    const RowCoins rc{&feed[(size_t)i * H], 0u};
    fr[i] = fold_row(rc.h, i, S);
    for (int t = 0; t < S; t++)
      for (int l = 0; l < 32; l++) ap[((size_t)i * S + t) * 32 + l] = (int8_t)lane_a(i, t, l, K, S, &A[(size_t)i * K], &rc);
  }
  std::vector<uint32_t> cj(n);
  std::vector<int> beta(n);
  for (int j = 0; j < n; j++) cj[j] = code_word(j), beta[j] = beta_of(j, S);
  size_t bad_read = 0, bad_served = 0, past = 0, c_83d = 0, c_left = 0, c_ffma = 0, c_sj = 0;
  long long max_word = 0, max_ab = 0, max_n = 0;
  for (int q = 0; q < tiles; q++) {
    auto lw = [&](int t, int j) { return lc[((size_t)q * S + t) * n + j]; };
    for (int j = 0; j < n; j++) {
      int G[4];
      for (int e = 0; e < 4; e++) G[e] = g_of(feed[(size_t)R * H + j], e);
      for (int t = 0; t < S; t++)
        for (int l = 0; l < 32; l++) bp[((size_t)j * S + t) * 32 + l] = (int8_t)lane_b(j, t, l, K, S, &W[(size_t)j * K], G, true, lw(t, j));
    }
    std::vector<int> sigma(2 * (size_t)n);
    for (int cls = 0; cls < 2; cls++)
      for (int j = 0; j < n; j++) sigma[(size_t)cls * n + j] = xu_sigma(cls, S, [&](int t) { return lw(t, j); });
    for (int i = 64 * q; i < std::min(m, 64 * q + 64); i++) {
      const float si = sa.f(i);
      for (int j = 0; j < n; j++) {
        const int8_t* b = &bp[(size_t)j * S * 32];
        const long long word = word_of(&ap[(size_t)i * S * 32], b, S, beta[j]);
        const int s = sigma[(size_t)(i & 1) * n + j];
        const long long z = za.i((size_t)i * n + j), nij = word - s - z;
        max_word = std::max(max_word, word < 0 ? -word : word);
        max_ab = std::max(max_ab, z < 0 ? -z : z);
        max_n = std::max(max_n, nij < 0 ? -nij : nij);
        const float sj = sw.f(j);
        const uint32_t d = (uint32_t)word, qv = fold_word(d, fr[i].r, (uint32_t)col_source(beta[j], s), fr[i].nt,
                                                          g_bytes(b), fr[i].na, cj[j]);
        bad_read += (int32_t)(qv - XU_MAGIC_BITS) != z;
        if (z < -bound || z > bound) {
          past++;
          continue;
        }
        const uint16_t y = (uint16_t)ya.i((size_t)i * n + j);
        const uint32_t left = d + (XU_MAGIC_BITS - (uint32_t)s);
        bad_served += bf16_rn(xu_fold(d, fr[i].r, (uint32_t)col_source(beta[j], s), fr[i].nt, g_bytes(b), fr[i].na, cj[j],
                                      si, sj)) != y;
        c_83d += fold_83d(left, si, sj) != y;
        c_left += bf16_rn(xu_fold(left, 0u, 0u, 0u, 0u, 0u, 0u, si, sj)) != y;
        c_ffma += fold_83d(qv, si, sj) != y;
        c_sj += fold_sj_first(qv, si, sj) != y;
      }
    }
  }
  const size_t outputs = (size_t)m * n;
  const bool ok = !bad_read && !bad_served && !past && max_word < (1ll << 31) && c_83d > 0;
  printf("{\"level\": \"served\", \"file\": \"%s\", \"layout\": \"%s\", \"m\": %d, \"n\": %d, \"K\": %d, \"S\": %d, "
         "\"committed_m\": %d, \"g_redraws\": %zu, \"outputs\": %zu, \"served_against\": \"int8_exact y\", "
         "\"max_abs_ab\": %lld, \"max_abs_n_ij\": %lld, \"max_abs_word\": %lld, \"past_bound\": %zu, \"bound\": %lld, "
         "\"read_back_not_z\": %zu, \"served_mismatches\": %zu, \"control_83d82109a8_mismatches\": %zu, "
         "\"control_n_ij_left_mismatches\": %zu, \"control_ffma_mismatches\": %zu, "
         "\"control_weight_scale_first_mismatches\": %zu, \"pass\": %s}\n",
         path.c_str(), layout_name(), m, n, K, S, R, redraws, outputs, max_ab, max_n, max_word, past, bound, bad_read,
         bad_served, c_83d, c_left, c_ffma, c_sj, ok ? "true" : "false");
  fflush(stdout);
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

static int vectors(const std::string& dir, int* lay_n, int* lay_ok) {
  std::ifstream idx(dir + "/index.txt");
  if (!idx) {
    fprintf(stderr, "ERR open %s/index.txt\n", dir.c_str());
    return 2;
  }
  std::string line;
  while (std::getline(idx, line)) {
    std::istringstream s(line);
    std::string kind, label, file, w;
    int expect = 0;
    if (!(s >> kind >> label >> file >> w >> expect) || kind != "layout") continue;
    const Vec v = load(dir + "/" + file);
    for (const bool xu : {true, false}) ++*lay_n, *lay_ok += layout_check(label, v, w, xu);
    fflush(stdout);
  }
  return *lay_n ? 0 : 2;
}

int main(int argc, char** argv) {
  long long bound = 1ll << 22;
  std::vector<std::string> levels;
  for (int a = 1; a < argc; a++) {
    if (!strcmp(argv[a], "--bound") && a + 1 < argc) bound = strtoll(argv[++a], nullptr, 10);
    else levels.push_back(argv[a]);
  }
  if (levels.empty() || bound <= 0) {
    fprintf(stderr, "usage: cpu_check [--bound B] LEVEL... (ragged, window, draw, served:FILE, or fold_vectors.py's --out "
            "DIR)\n");
    return 2;
  }
  // (call rows, call columns, K, BN): decode's one row, rows and columns off the tiles, a call of whole 16-row groups
  // (no padding rows, its computed rows past them), and rows off 64 with columns off 256
  struct Call { int m, n, K, BN; };
  const Call calls[] = {{1, 256, 2048, 128}, {17, 13, 100, 128}, {16, 256, 100, 128}, {100, 1000, 2048, 256}};
  int n = 0, passed = 0, lay_n = 0, lay_ok = 0;
  for (const std::string& level : levels) {
    if (level == "ragged") {
      uint64_t seed = 1;
      for (const Call& c : calls) n++, passed += ragged_check(c.m, c.n, c.K, c.BN, bound, seed++, false);
      for (const Call& c : {calls[0], calls[1]}) n++, passed += ragged_check(c.m, c.n, c.K, c.BN, bound, seed++, true);
    } else if (level == "window") {
      n++, passed += window_check(bound, 7);
    } else if (level == "draw") {
      n++, passed += draw_check();
    } else if (level.rfind("served:", 0) == 0) {
      n++, passed += served_check(level.substr(7), bound);
    } else if (vectors(level, &lay_n, &lay_ok)) {
      return 2;
    }
  }
  const bool ok = passed == n && lay_ok == lay_n;
  printf("{\"summary\": true, \"levels\": %zu, \"checks\": %d, \"checks_pass\": %d, \"layout_checks\": %d, "
         "\"layout_pass\": %d, \"pass\": %s}\n", levels.size(), n, passed, lay_n, lay_ok, ok ? "true" : "false");
  return ok ? 0 : 1;
}
