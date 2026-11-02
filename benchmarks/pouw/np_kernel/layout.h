// ssn_1l's lane layout (ssn.cu's header states it), the fold's row and column sources and words, the fold and
// verity_int8's epilogue it must equal: the one definition that ssn.cu's operands, forming, forming check and epilogue
// (xu_fold) and the CPU check (cpu_check.cc) read. Plain C++; under nvcc the fold and the words it reads are also device
// functions.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define SSN_HD __host__ __device__
#else
#define SSN_HD
#endif

static bool g_tag = false;  // TAG=1: the ssn-1l-tag layout
static int g_blc = 0;  // BLC=1: blc4a-t64-e7 (with the tag layout's step 1 and row lane); BLC=2: its lane coins on step 1 too
static int g_lw[3] = {1, 1, 1};  // BLC_W: A' in the three lane-coin lanes
static bool g_rot = false;  // ROT=1: blc4a-t64-e7-rot, A' in the lane-coin lanes by row parity (ROT_W), in place of BLC_W
constexpr int ROT_W[2][3] = {{127, 1, 1}, {1, 127, 1}};
static bool g_xu = false;  // XU=1: ssn_1l_tag_xu's last step and epilogue, on a tag layout
static bool g_fold = false;  // FOLD=1 (with BLC, not XU): served noisy, xu's epilogue folding every coin term out
// A' in lane-coin lane e (0..2) for row i
static inline int coin_w(int i, int e) { return g_rot ? ROT_W[i & 1][e] : g_lw[e]; }
static inline int lanes1() { return g_blc == 2 ? 21 : g_tag ? 24 : 28; }  // step 1's data lanes
static inline int lanesl() { return g_blc ? 28 : 31; }  // a later step's data lanes
// XU=1: the last step's lanes from xu_at (the first 4-aligned lane past its data) hold a1 against -C, -T(i) against G,
// u0 against 1, u1 against 127 and u2 in xu_e lanes against 127; S >= 2, and a step is added when fewer than 12 are left
static inline int xu_at(int k, int S) { return (std::max(0, k - lanes1() - lanesl() * (S - 2)) + 3) & ~3; }
static inline int xu_e(int xa) { return std::min(12, 4 * ((lanesl() - xa - 8) / 4) - 2); }
static inline int steps_of(int k) {
  int S = k <= lanes1() ? 1 : 1 + (k - lanes1() + lanesl() - 1) / lanesl();
  if (g_xu && S < 2) S = 2;
  if (g_xu && lanesl() - xu_at(k, S) < 12) S++;
  return S;
}
static inline int lc_steps(int S) { return g_blc == 2 ? S : S - 1; }  // the steps with lane coins, the last lc_steps(S) of S
static inline const char* layout_name() {
  if (g_rot && g_blc == 2)
    return g_xu ? "blc4a_t64_e7_s1_rot_xu" : g_fold ? "blc4a_t64_e7_s1_rot_fold" : "blc4a_t64_e7_s1_rot";
  if (g_rot) return g_xu ? "blc4a_t64_e7_rot_xu" : g_fold ? "blc4a_t64_e7_rot_fold" : "blc4a_t64_e7_rot";
  if (g_xu) return g_blc == 2 ? "blc4a_t64_e7_s1_xu" : g_blc ? "blc4a_t64_e7_xu" : "ssn_1l_tag_xu";
  if (g_fold) return g_blc == 2 ? "blc4a_t64_e7_s1_fold" : "blc4a_t64_e7_fold";
  return g_blc == 2 ? "blc4a_t64_e7_s1" : g_blc ? "blc4a_t64_e7" : g_tag ? "ssn_1l_tag" : "ssn_1l";
}
SSN_HD inline int coin_words(int s) { return (s + 3 + 31) / 32; }
// A call of m rows and n columns commits its words over pad_rows(m) rows and pad_cols(n) columns, whole MMA tiles (Lean's
// padRows, padCols): a padding row has zero data and its tag, lane-coin A' and coins like any row, a padding column zero
// data and its coins
static inline int pad_rows(int m) { return 16 * ((m + 15) / 16); }
static inline int pad_cols(int n) { return 8 * ((n + 7) / 8); }

// C[e, j]: digit e of j + 1 in balanced base 13 (Lean's colCode, bal13 (j + 1)), the base-13 digits of j + 14,281 minus 6;
// distinct and nonzero on columns below 28,560
SSN_HD inline int code_of(int j, int e) {
  long long c = (long long)j + 14281;
  for (int r = 0; r < e; r++) c /= 13;
  return (int)(c % 13) - 6;
}
// beta_j = floor((sum_e C[e, j] + w (S - 1) + 1) / 2), w 3 under the tag layout and 1 without (Lean's bias)
static inline int beta_of(int j, int S) {
  int s = (g_tag ? 3 : 1) * (S - 1) + 1;
  for (int e = 0; e < 4; e++) s += code_of(j, e);
  return (int)std::floor(s / 2.0);
}
// byte 3 of the certifier's lane-coin word for step t (0-based) and column j, bytes 0-2 its three coins: 1 (against lane
// 31's 1 + a_t[i]) at a later step, the public C[3, j] (against a1[i, 3]) at step 1 under BLC=2
static inline uint32_t coin_word_top(int t, int j) {
  return t == 0 ? (uint32_t)(uint8_t)(int8_t)code_of(j, 3) << 24 : 0x01000000u;
}
// T(i)[e]: digit e of z_i's balanced base-13 expansion, each in [-6, 6]
SSN_HD inline int tag_of(int i, int e) {
  int z = (i / 2 + 1) * (i & 1 ? -1 : 1), d = 0;
  for (int r = 0; r <= e; r++) {
    d = ((z % 13) + 19) % 13 - 6;
    z = (z - d) / 13;
  }
  return d;
}
// data entry of step t (0-based), lane l: its K index, or -1 for a noise, tag or padding lane
static inline int kidx(int t, int l, int K) {
  const int d = lanes1(), e = lanesl();
  const int k = t == 0 ? (l < d ? l : -1) : (l < e ? d + e * (t - 1) + l : -1);
  return k >= 0 && k < K ? k : -1;
}

// The certifier's coin feed for a call of R committed rows (pad_rows) and C committed columns (pad_cols): u32 words,
// each coin drawn raw by the certifier once it holds root(A_j), in this order:
//   row words  R coin_words(S) words, row i's from word i coin_words(S): bit e is a1[i, e] (e < 4), bit t + 3 the step bit
//              b_t[i] (0-based t = 1 .. S - 1), every other bit 0 (b at step 0 enters no word)
//   G words    under the tag layout C words, column j's byte e the int8 G[e, j] in [-3, 3], drawn by g_word_of
//   u words    under XU=1 R words, row i's u word
// The lane-coin words (one a (64-row tile, step, column)) follow in the same release.
static inline long long feed_words(int R, int S, int C) {
  return (long long)R * coin_words(S) + (g_tag ? C : 0) + (g_xu ? R : 0);
}
static inline void row_words(const int* a1, const int* b, int S, uint32_t* h) {
  for (int q = 0; q < coin_words(S); q++) h[q] = 0;
  for (int e = 0; e < 4; e++) h[0] |= (uint32_t)(a1[e] & 1) << e;
  for (int t = 1; t < S; t++) h[(t + 3) >> 5] |= (uint32_t)(b[t] & 1) << ((t + 3) & 31);
}
static inline uint32_t g_word(const int* G) {
  uint32_t w = 0;
  for (int e = 0; e < 4; e++) w |= (uint32_t)(uint8_t)(int8_t)G[e] << (8 * e);
  return w;
}
static inline int g_of(uint32_t w, int e) { return (int8_t)(uint8_t)(w >> (8 * e)); }
// G by rejection: a 32-bit draw u is kept when u < 2401 floor(2^32 / 2401) = 4,294,966,424 (872 of the 2^32 are
// rejected and redrawn), and G[e, j] is digit e of u mod 2401 in base 7, minus 3, so G is uniform on [-3, 3]^4 exactly
constexpr uint32_t G_KEEP = 2401u * (uint32_t)(4294967296ull / 2401u);
static inline bool g_word_of(uint32_t u, uint32_t* w) {
  if (u >= G_KEEP) return false;
  int G[4];
  uint32_t v = u % 2401u;
  for (int e = 0; e < 4; e++, v /= 7) G[e] = (int)(v % 7) - 3;
  *w = g_word(G);
  return true;
}

// A row's private values as the forming reads them: a1[e] is bit e and a_t bit t + 3 (step t >= 1, 0-based) of the row's
// feed words h, and u its u word: u0 its low 7 bits, u1 and u2 its signed bytes 1 and 2
struct RowCoins {
  const uint32_t* h;
  uint32_t u;
};
static inline int rc_bit(const RowCoins& c, int p) { return (int)((c.h[p >> 5] >> (p & 31)) & 1u); }
SSN_HD inline int u0_of(uint32_t u) { return (int)(u & 0x7Fu); }
SSN_HD inline int u1_of(uint32_t u) { return (int8_t)(uint8_t)(u >> 8); }
SSN_HD inline int u2_of(uint32_t u) { return (int8_t)(uint8_t)(u >> 16); }

// A' step t lane l for row i, x its K data entries: with c null as A's producer writes it (the private lanes 0), else
// formed from c
static inline int lane_a(int i, int t, int l, int K, int S, const int8_t* x, const RowCoins* c) {
  const int k = kidx(t, l, K);
  int v = k >= 0 ? x[k] : 0;
  if (g_tag && t == 0 && l >= 24 && l < 28) v = tag_of(i, l - 24);
  if (g_blc && (t > 0 || g_blc == 2) && l >= 28 && l < 31) v = coin_w(i, l - 28);
  if (c && t == 0 && g_blc == 2 && ((l >= 21 && l < 24) || l == 31)) v = rc_bit(*c, l == 31 ? 3 : l - 21);
  if (c && t == 0 && g_blc != 2 && l >= 28) v = rc_bit(*c, l - 28);
  if (c && t > 0 && l == 31) v = (g_tag ? 1 : 0) + rc_bit(*c, t + 3);
  if (g_xu && t == S - 1) {
    const int xa = xu_at(K, S), xe = xu_e(xa);
    if (l >= xa + 4 && l < xa + 8) v = -tag_of(i, l - xa - 4);
    if (c && l >= xa && l < xa + 4) v = rc_bit(*c, l - xa);
    if (c && l == xa + 8) v = u0_of(c->u);
    if (c && l == xa + 9) v = u1_of(c->u);
    if (c && l >= xa + 10 && l < xa + 10 + xe) v = u2_of(c->u);
  }
  return v;
}
// B' step t lane l for column j, y its K data entries: G its four digits (null: 0, as before G's draw); with lw, w is the
// column's lane-coin word at step t (bytes 0-2 the coins, byte 3 coin_word_top's), copied over B' lanes 28-31 by the GEMM
// (ssn_k's cv)
static inline int lane_b(int j, int t, int l, int K, int S, const int8_t* y, const int* G, bool lw, uint32_t w) {
  const int k = kidx(t, l, K);
  int v = k >= 0 ? y[k] : 0;
  if (t == 0 && g_blc == 2 && ((l >= 21 && l < 24) || l == 31)) v = code_of(j, l == 31 ? 3 : l - 21);
  if (t == 0 && g_blc != 2 && l >= 28) v = code_of(j, l - 28);
  if (t > 0 && l == 31) v = 1;
  if (G && g_tag && t == 0 && l >= 24 && l < 28) v = G[l - 24];
  if (g_xu && t == S - 1) {
    const int xa = xu_at(K, S), xe = xu_e(xa);
    if (l >= xa && l < xa + 4) v = -code_of(j, l - xa);
    if (G && l >= xa + 4 && l < xa + 8) v = G[l - xa - 4];
    if (l == xa + 8) v = 1;
    if (l > xa + 8 && l < xa + 10 + xe) v = 127;
  }
  if (lw && g_blc && (t > 0 || g_blc == 2) && l >= 28) v = (int8_t)(uint8_t)(w >> (8 * (l - 28)));
  return v;
}

// xu's row offset rho_i = u0 + 127 u1 + 127 e u2 + sum_{t >= 1} (1 + a_t) (a_t alone without the tag), and its row
// source 0x4B400000 - rho_i
static inline int xu_rho(const RowCoins& c, int K, int S) {
  const int xe = xu_e(xu_at(K, S));
  int r = u0_of(c.u) + 127 * u1_of(c.u) + 127 * xe * u2_of(c.u);
  for (int t = 1; t < S; t++) r += (g_tag ? 1 : 0) + rc_bit(c, t + 3);
  return r;
}
constexpr uint32_t XU_MAGIC_BITS = 0x4B400000u;  // 1.5 2^23
static inline uint32_t xu_row_source(const RowCoins& c, int K, int S) { return XU_MAGIC_BITS - (uint32_t)xu_rho(c, K, S); }
// sigma for a (64-row tile, row class, column): the lane coins' sum weighted by the class's A' (class i mod 2 under ROT=1;
// both classes alike otherwise), lw(t) the column's lane-coin word at step t
template <class LW>
static inline int xu_sigma(int cls, int S, LW lw) {
  int s = 0;
  for (int t = g_blc == 2 ? 0 : 1; g_blc && t < S; t++) {
    const uint32_t w = lw(t);
    for (int e = 0; e < 3; e++) s += coin_w(cls, e) * (int8_t)(uint8_t)(w >> (8 * e));
  }
  return s;
}
// the column source for a (tile, row class, column) from beta_j and its sigma, beta_j - sigma, under XU=1 and FOLD=1 alike
static inline int col_source(int beta, int sigma) { return beta - sigma; }

// The walk w_i = sum_{t >= 2} (1 + a_t[i]), lane 31 of each later step under the tag layout: S - 1 plus the popcount of
// bits 4 to S + 2 of the row's feed words h
SSN_HD inline int walk_of(const uint32_t* h, int S) {
  int w = S - 1;
  for (int p = 4; p < S + 3; p++) w += (int)((h[p >> 5] >> (p & 31)) & 1u);
  return w;
}
// FOLD=1's cross terms, which the epilogue takes out of the word by two IDP4A, as four int8 bytes each: -a1[i, 0..3] (0
// or -1, from bits 0-3 of the row's first feed word h0) against C[0..3, j], and -T(i)[0..3] against G[0..3, j] (the G
// word as fed)
SSN_HD inline uint32_t neg_a1_word(uint32_t h0) { return (((h0 & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu; }
SSN_HD inline uint32_t neg_tag_word(int i) {
  uint32_t w = 0;
  for (int e = 0; e < 4; e++) w |= (uint32_t)(uint8_t)(int8_t)-tag_of(i, e) << (8 * e);
  return w;
}
SSN_HD inline uint32_t code_word(int j) {
  uint32_t w = 0;
  for (int e = 0; e < 4; e++) w |= (uint32_t)(uint8_t)(int8_t)code_of(j, e) << (8 * e);
  return w;
}
// The epilogue's words a row, which the forming writes: the row source r (XU=1 0x4B400000 - rho_i, FOLD=1 0x4B400000 -
// w_i) and FOLD=1's cross bytes na (neg_a1_word) and nt (neg_tag_word), 0 under XU=1, whose last step cancels a1 . C and
// T . G in its lanes
struct alignas(16) FoldRow {
  uint32_t r, na, nt, unused;
};
// and a column, formed at load: s_j and the bytes C[0..3, j] (code_word)
struct alignas(8) FoldCol {
  float s;
  uint32_t c;
};
// FOLD=1's row words for row i from its feed words h (the H zero words for a computed row past the committed ones)
SSN_HD inline FoldRow fold_row(const uint32_t* h, int i, int S) {
  return FoldRow{XU_MAGIC_BITS - (uint32_t)walk_of(h, S), neg_a1_word(h[0]), neg_tag_word(i), 0u};
}

// IDP4A: c plus the dot product of a's and b's four signed bytes, mod 2^32
SSN_HD inline uint32_t dp4a(uint32_t a, uint32_t b, uint32_t c) {
#if defined(__CUDA_ARCH__)
  return (uint32_t)__dp4a((int)a, (int)b, (int)c);
#else
  for (int e = 0; e < 4; e++) c += (uint32_t)((int8_t)(uint8_t)(a >> (8 * e)) * (int8_t)(uint8_t)(b >> (8 * e)));
  return c;
#endif
}
// xu's integer fold on a final word d: q = d + r + c (one IADD3 of the row and column sources), then q = IDP4A(na, cj,
// IDP4A(nt, g, q)), which takes out FOLD=1's a1 . C and T . G (nothing under XU=1), so q = 0x4B400000 + A_i . B_j mod 2^32
SSN_HD inline uint32_t fold_word(uint32_t d, uint32_t r, uint32_t c, uint32_t nt, uint32_t g, uint32_t na, uint32_t cj) {
  return dp4a(na, cj, dp4a(nt, g, d + r + c));
}
// xu's epilogue on a final word d: fold_word's q, then FADD(v, -1.5 2^23), v the float whose bits are q, and the scales in
// verity_int8's order, FMUL s_i, then FMUL s_j. While |A_i . B_j| <= 2^22 the FADD is exact (v and 1.5 2^23 lie in
// [2^23, 2^24]) and gives A_i . B_j, so the fold is int8_epilogue on A_i . B_j bit for bit, whatever the scales; past it v
// is another float (2^22 + 1 reads back as 2^22 + 2) and nothing flags it.
SSN_HD inline float xu_fold(uint32_t d, uint32_t r, uint32_t c, uint32_t nt, uint32_t g, uint32_t na, uint32_t cj, float si,
                            float sj) {
  const uint32_t q = fold_word(d, r, c, nt, g, na, cj);
#if defined(__CUDA_ARCH__)
  return __fmul_rn(__fmul_rn(__fadd_rn(__uint_as_float(q), -12582912.0f), si), sj);
#else
  float v;
  memcpy(&v, &q, 4);
  const float f = (v - 12582912.0f) * si;
  return f * sj;
#endif
}
// verity_int8's epilogue (int8_exact.dequantize, int8_fused._epilogue) on its int32 product y, before the bf16 pack:
// fl(fl(f32(y) s_a[i]) s_w[j]), the activation's scale first
SSN_HD inline float int8_epilogue(int y, float si, float sj) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(__fmul_rn(__int2float_rn(y), si), sj);
#else
  const float f = (float)y * si;
  return f * sj;
#endif
}
// the bf16 pack, round to nearest even (finite values)
static inline uint16_t bf16_rn(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
