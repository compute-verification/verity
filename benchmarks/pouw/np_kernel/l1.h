// L1 + (a)'s seed path on the host (no CUDA): the byte-level reference ssn.cu's kernels are checked against, and the
// per-call constants (templates and midstates) those kernels read. l1_ref.py computes the same values from the tree's
// Python (verity.ml.commitments' frame-v3-sha512 and hm96-sha512, pouw's call seed and noise blocks); l1_check.cpp
// prints this header's.
//
// A call's seed path (N-HIDE's L1 + (a)), everything SHA-512, call j of forward f in a window with salt and ks:
//   A_j's leaves   leaf(D_j, r, r, "pouw/a-seg/v1", A_j[1024 r, 1024 r + 1024)), D_j the range-u64 domain over the call's
//                  leaves under binding SHA-512(salt || u64be(f) || u64be(j)) (the tree's domains take a 64-byte binding)
//   x              the frame-v3-sha512 root over them (pads to a power of two)
//   y              (a)'s PRF: y_r = SHA-512(k_w || u64be(f) || u64be(j) || u64be(r)), r = 0, 1, 2, k_w = SHA-512(ks || salt);
//                  (f, j) is the instance's position in the window (a y used twice would leak x ^ x')
//   b || c         hm96-sha512: b = x ^ M_key y, c = SHA-512(salt_prefix || y); h = SHA-512(leaf_prefix(key) || b || c)
//   root(A_j)      h, the instance's hm96 leaf
//   seed           pouw's call seed at the call's window position c = f · calls + j:
//                  SHA-512(u64(c) || (CALL_SEED_TAG || salt || root(A_j) || root(B_j)) ^ u64(c)*)
//   noise block t  SHA-512(NOISE_TAG || seed || u64be(t))
// A's schema is this benchmark's stand-in of the format's length; nothing else is.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hsha {
static const uint64_t K[80] = {
    0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull,
    0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull, 0x12835b0145706fbeull,
    0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull, 0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull,
    0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
    0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull, 0x983e5152ee66dfabull,
    0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,
    0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull,
    0x53380d139d95b3dfull, 0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
    0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull,
    0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull, 0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull,
    0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull,
    0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
    0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull, 0xca273eceea26619cull,
    0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull,
    0x113f9804bef90daeull, 0x1b710b35131c471bull, 0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull,
    0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull};
static uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
static void iv(uint64_t h[8]) {
  const uint64_t v[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                         0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  for (int i = 0; i < 8; i++) h[i] = v[i];
}
static void compress(uint64_t h[8], const uint64_t m[16]) {
  uint64_t w[80], a[8];
  for (int i = 0; i < 16; i++) w[i] = m[i];
  for (int i = 16; i < 80; i++) {
    const uint64_t s0 = rotr(w[i - 15], 1) ^ rotr(w[i - 15], 8) ^ (w[i - 15] >> 7);
    const uint64_t s1 = rotr(w[i - 2], 19) ^ rotr(w[i - 2], 61) ^ (w[i - 2] >> 6);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  for (int i = 0; i < 8; i++) a[i] = h[i];
  for (int i = 0; i < 80; i++) {
    const uint64_t t1 = a[7] + (rotr(a[4], 14) ^ rotr(a[4], 18) ^ rotr(a[4], 41)) + ((a[4] & a[5]) ^ (~a[4] & a[6])) + K[i] + w[i];
    const uint64_t t2 = (rotr(a[0], 28) ^ rotr(a[0], 34) ^ rotr(a[0], 39)) + ((a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]));
    for (int j = 7; j > 0; j--) a[j] = a[j - 1];
    a[4] += t1;
    a[0] = t1 + t2;
  }
  for (int i = 0; i < 8; i++) h[i] += a[i];
}
}  // namespace hsha

namespace l1 {
using u64 = uint64_t;
using Bytes = std::vector<uint8_t>;

constexpr int SEG = 1024, LEVELS = 24;
static const char* const A_SCHEMA = "pouw/a-seg/v1";

inline u64 be64(const uint8_t* p) {
  u64 v = 0;
  for (int i = 0; i < 8; i++) v = v << 8 | p[i];
  return v;
}
inline Bytes cat(std::initializer_list<Bytes> ps) {
  Bytes o;
  for (const Bytes& p : ps) o.insert(o.end(), p.begin(), p.end());
  return o;
}
inline Bytes str(const char* s, size_t n) { return Bytes(s, s + n); }
inline Bytes str(const char* s) { return str(s, strlen(s)); }
inline Bytes u64be(u64 v) {
  Bytes b(8);
  for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (56 - 8 * i));
  return b;
}
inline int nbytes(u64 v) {
  int n = 1;
  while (n < 8 && (v >> (8 * n))) n++;
  return n;
}
inline Bytes uintb(u64 v) {
  const int n = nbytes(v);
  Bytes b(n);
  for (int i = 0; i < n; i++) b[i] = (uint8_t)(v >> (8 * (n - 1 - i)));
  return b;
}
// the message's words, padded (0x80, zeros, the bit length in the last 16 bytes) to whole 128-byte blocks
inline std::vector<u64> padded(const Bytes& m) {
  const size_t total = (m.size() + 17 + 127) / 128 * 128;
  Bytes b(total, 0);
  memcpy(b.data(), m.data(), m.size());
  b[m.size()] = 0x80;
  const u64 bits = (u64)m.size() * 8;
  for (int i = 0; i < 8; i++) b[total - 1 - i] = (uint8_t)(bits >> (8 * i));
  std::vector<u64> w(total / 8);
  for (size_t i = 0; i < w.size(); i++) w[i] = be64(&b[8 * i]);
  return w;
}
inline Bytes digest_bytes(const u64 h[8]) {
  Bytes o(64);
  for (int j = 0; j < 8; j++)
    for (int i = 0; i < 8; i++) o[8 * j + i] = (uint8_t)(h[j] >> (56 - 8 * i));
  return o;
}
inline Bytes sha512(const Bytes& m) {
  const auto w = padded(m);
  u64 h[8];
  hsha::iv(h);
  for (size_t b = 0; b < w.size() / 16; b++) hsha::compress(h, &w[16 * b]);
  return digest_bytes(h);
}
inline void words(const Bytes& d, u64* w, int n) {
  for (int i = 0; i < n; i++) w[i] = be64(&d[8 * i]);
}
inline void midstate(const u64* block, u64* st) {
  hsha::iv(st);
  hsha::compress(st, block);
}

// frame-v3-sha512 (verity.ml.commitments.merkle)
inline Bytes frame_message(const char* tag, std::initializer_list<Bytes> parts) {
  Bytes m = str("veritor/protocol/merkle/frame/v3", 33);
  const size_t tl = strlen(tag);
  for (int i = 3; i >= 0; i--) m.push_back((uint8_t)(tl >> (8 * i)));
  m.insert(m.end(), tag, tag + tl);
  for (const Bytes& p : parts) {
    const Bytes n = u64be(p.size());
    m.insert(m.end(), n.begin(), n.end());
    m.insert(m.end(), p.begin(), p.end());
  }
  return m;
}
inline Bytes range_u64_domain(const Bytes& binding, long long owner, u64 start, u64 count) {
  return sha512(frame_message("domain/range-u64", {binding, u64be((u64)(owner + 2)), u64be(start), u64be(count)}));
}
inline Bytes leaf(const Bytes& dom, u64 rank, u64 pos, const char* schema, const Bytes& value) {
  return sha512(frame_message("leaf", {dom, uintb(rank), uintb(pos), str(schema), value}));
}
inline Bytes node(const Bytes& dom, int level, u64 index, const Bytes& l, const Bytes& r) {
  return sha512(frame_message("node", {dom, uintb((u64)level), uintb(index), l, r}));
}
inline Bytes pad(const Bytes& dom, u64 rank) { return sha512(frame_message("pad", {dom, uintb(rank)})); }
inline Bytes root(const Bytes& dom, std::vector<Bytes> level) {
  size_t w = 1;
  while (w < level.size()) w <<= 1;
  for (size_t r = level.size(); r < w; r++) level.push_back(pad(dom, r));
  for (int d = 0; level.size() > 1; d++) {
    std::vector<Bytes> up;
    for (size_t i = 0; i < level.size(); i += 2) up.push_back(node(dom, d, i / 2, level[i], level[i + 1]));
    level = up;
  }
  return level[0];
}

// hm96-sha512/v1 (verity.ml.commitments.hm96)
inline Bytes hm96_key() {
  const Bytes label = str("verity/hm96-sha512/key/v1", 26);
  Bytes k;
  for (uint32_t i = 0; i < 4; i++) {
    Bytes m = label;
    for (int s = 24; s >= 0; s -= 8) m.push_back((uint8_t)(i >> s));
    const Bytes d = sha512(m);
    k.insert(k.end(), d.begin(), d.end());
  }
  k[255] &= 0x7f;
  return k;
}
inline Bytes mask(const Bytes& key, const Bytes& y) {  // (M_key y)_i = xor_j k_{i+j} y_j, bit t of a string bit t % 8 of byte t / 8
  Bytes o(64, 0);
  auto bit = [](const Bytes& s, int t) { return (s[t >> 3] >> (t & 7)) & 1; };
  for (int i = 0; i < 512; i++) {
    int v = 0;
    for (int j = 0; j < 1536; j++) v ^= bit(key, i + j) & bit(y, j);
    o[i >> 3] |= (uint8_t)(v << (i & 7));
  }
  return o;
}
inline Bytes block_of(const char* tag, size_t n) {
  Bytes b = str(tag, n);
  b.resize(128, 0);
  return b;
}
inline Bytes salt_prefix() { return block_of("verity/hm96-sha512/salt/v1", 27); }
inline Bytes leaf_prefix(const Bytes& key) {
  Bytes b = cat({str("verity/hm96-sha512/leaf/v1", 27), sha512(key)});
  b.resize(128, 0);
  return b;
}

// pouw (verity.core.protocols.pouw.circuit.seeds)
inline Bytes call_seed_tag() { return str("verity/pouw/call-seed/v1", 24); }
inline Bytes noise_tag() {
  Bytes t = str("verity/pouw/noise/v1");
  t.resize(24, 0);
  return t;
}
inline Bytes call_seed(const Bytes& salt, u64 call, const Bytes& ra, const Bytes& rb) {
  const Bytes p = u64be(call), data = cat({call_seed_tag(), salt, ra, rb});
  Bytes m = p;
  for (size_t i = 0; i < data.size(); i++) m.push_back(data[i] ^ p[i % 8]);
  return sha512(m);
}
inline Bytes noise_block(const Bytes& seed, u64 t) { return sha512(cat({noise_tag(), seed, u64be(t)})); }

// (a)'s PRF and A_j's domain binding
inline Bytes prf_y(const Bytes& kw, u64 forward, u64 call) {
  Bytes y;
  for (u64 r = 0; r < 3; r++) {
    const Bytes d = sha512(cat({kw, u64be(forward), u64be(call), u64be(r)}));
    y.insert(y.end(), d.begin(), d.end());
  }
  return y;
}
inline Bytes a_binding(const Bytes& salt, u64 forward, u64 call) { return sha512(cat({salt, u64be(forward), u64be(call)})); }

struct Window {
  Bytes salt, ks, key, root_b;
  u64 forward;
  u64 calls;
  Bytes kw() const { return sha512(cat({ks, salt})); }
  u64 position(u64 call) const { return forward * calls + call; }
};
inline Window window_of(u64 seed, u64 forward, u64 calls) {
  return Window{sha512(cat({str("np-kernel/salt"), u64be(seed)})), sha512(cat({str("np-kernel/ks"), u64be(seed)})), hm96_key(),
                sha512(cat({str("np-kernel/root-b"), u64be(seed)})), forward, calls};
}

// The byte-level reference of one call's seed path over A_j's bytes (a whole number of 1 KiB segments).
struct Ref {
  Bytes dom, x, y, bc, h, root_a, seed;
  std::vector<Bytes> leaves;
};
inline Ref reference(const Window& w, u64 call, const uint8_t* a, size_t bytes) {
  Ref r;
  const u64 n = bytes / SEG;
  r.dom = range_u64_domain(a_binding(w.salt, w.forward, call), -1, 0, n);
  for (u64 i = 0; i < n; i++) r.leaves.push_back(leaf(r.dom, i, i, A_SCHEMA, Bytes(a + SEG * i, a + SEG * (i + 1))));
  r.x = root(r.dom, r.leaves);
  r.y = prf_y(w.kw(), w.forward, call);
  const Bytes m = mask(w.key, r.y), c = sha512(cat({salt_prefix(), r.y}));
  Bytes b(64);
  for (int i = 0; i < 64; i++) b[i] = r.x[i] ^ m[i];
  r.bc = cat({b, c});
  r.h = sha512(cat({leaf_prefix(w.key), r.bc}));
  r.root_a = r.h;
  r.seed = call_seed(w.salt, w.position(call), r.root_a, w.root_b);
  return r;
}

// One call's constants for the kernels, every digest and block in SHA-512's big-endian words. Nothing in them reads A_j,
// so a window computes them all once its salt is drawn.
struct Call {
  u64 leaf_tpl[3][24];      // the A leaf's prefix (P = 158 + 2 nb bytes) with rank and position zero, nb = 1, 2, 3
  u64 node_mid[LEVELS][8];  // a node message's first-block state by level (only the level varies in it)
  u64 node_tpl[3][32];      // a node message's blocks 2 and 3 with index, left and right zero, by ni = 1, 2, 3
  u64 my[8], c[8];          // M_key y and c
  u64 hmid[8];              // leaf_prefix(key)'s state
  u64 sb[32];               // the seed message's blocks with root(A_j)'s words zero
  u64 ntag[3];              // NOISE_TAG's words
  u64 call, n;              // the call's window position and its leaves
};
inline Call call_of(const Window& w, u64 call, u64 n) {
  Call k{};
  k.call = w.position(call);
  k.n = n;
  const Bytes dom = range_u64_domain(a_binding(w.salt, w.forward, call), -1, 0, n);
  for (int nb = 1; nb <= 3; nb++) {
    const Bytes m = frame_message("leaf", {dom, Bytes(nb, 0), Bytes(nb, 0), str(A_SCHEMA), Bytes(SEG, 0)});
    const size_t P = m.size() - SEG;
    Bytes pre(m.begin(), m.begin() + P);
    pre.resize(24 * 8, 0);
    words(pre, k.leaf_tpl[nb - 1], 24);
  }
  for (int lv = 0; lv < LEVELS; lv++) {
    const auto pw = padded(frame_message("node", {dom, uintb((u64)lv), Bytes(1, 0), Bytes(64, 0), Bytes(64, 0)}));
    midstate(&pw[0], k.node_mid[lv]);
  }
  for (int ni = 1; ni <= 3; ni++) {
    const auto pw = padded(frame_message("node", {dom, Bytes(1, 0), Bytes(ni, 0), Bytes(64, 0), Bytes(64, 0)}));
    for (int i = 0; i < 32; i++) k.node_tpl[ni - 1][i] = pw[16 + i];
  }
  const Bytes y = prf_y(w.kw(), w.forward, call);
  words(mask(w.key, y), k.my, 8);
  words(sha512(cat({salt_prefix(), y})), k.c, 8);
  {
    u64 blk[16];
    words(leaf_prefix(w.key), blk, 16);
    midstate(blk, k.hmid);
  }
  {
    const Bytes p = u64be(k.call), data = cat({call_seed_tag(), w.salt, Bytes(64, 0), w.root_b});
    Bytes m = p;
    for (size_t i = 0; i < data.size(); i++) m.push_back(data[i] ^ p[i % 8]);
    const auto pw = padded(m);
    for (int i = 0; i < 32; i++) k.sb[i] = pw[i];
    for (int i = 12; i < 20; i++) k.sb[i] = 0;
  }
  words(noise_tag(), k.ntag, 3);
  return k;
}

// ---- the kernels' path (host and device): the same values from Call's constants, a compression at a time --------------
#ifdef __CUDACC__
#define L1_HD __host__ __device__ __forceinline__
#else
#define L1_HD inline
#endif
L1_HD void compress(u64 st[8], const u64 w[16]) {
#ifdef __CUDA_ARCH__
  u64 c[16];
#pragma unroll
  for (int i = 0; i < 16; i++) c[i] = w[i];
  s512_compress(st, c);
#else
  hsha::compress(st, w);
#endif
}
L1_HD void iv8(u64 st[8]) {
#ifdef __CUDA_ARCH__
  s512_iv(st);
#else
  hsha::iv(st);
#endif
}
L1_HD void cpy8(u64* d, const u64* s) {
#pragma unroll
  for (int i = 0; i < 8; i++) d[i] = s[i];
}
// big-endian word j of an 8-byte-aligned byte string, 0 outside [0, n)
L1_HD u64 ldbe(const uint8_t* p, int j, int n) {
  if (j < 0 || j >= n) return 0;
#ifdef __CUDA_ARCH__
  return s512_bswap(__ldg(reinterpret_cast<const unsigned long long*>(p) + j));
#else
  return be64(p + 8 * j);
#endif
}
// the bits of the nb-byte big-endian integer v at message bytes [off, off + nb) that fall in word q
L1_HD u64 ins(int q, int off, u64 v, int nb) {
  const int shift = 8 * (8 * q + 8 - off - nb);
  if (shift >= 64 || shift <= -8 * nb) return 0;
  return shift >= 0 ? v << shift : v >> -shift;
}
// a 64-byte digest D at byte offset OFF: its bits in word q
template <int OFF>
L1_HD u64 dig(const u64* D, int q) {
  constexpr int q0 = OFF >> 3, s = OFF & 7;
  const int k = q - q0;
  u64 r = k >= 0 && k < 8 ? D[k] >> (8 * s) : 0;
  if (s && k >= 1 && k <= 8) r |= D[k - 1] << (64 - 8 * s);
  return r;
}

// A_j's leaf r over its segment val: P = 158 + 2 NB prefix bytes (word 20 holds the value's first 8 - s bytes), the
// 1,024 value bytes, ten blocks
template <int NB>
L1_HD void a_leaf_nb(const Call& k, u64 r, const uint8_t* val, u64 out[8]) {
  constexpr int P = 158 + 2 * NB, s = P & 7, L = P + SEG, NBLK = (L + 17 + 127) / 128;
  const u64* tpl = k.leaf_tpl[NB - 1];
  u64 st[8];
  iv8(st);
  u64 prev = 0;
#pragma unroll 1
  for (int b = 0; b < NBLK; b++) {
    u64 W[16];
#pragma unroll
    for (int w = 0; w < 16; w++) {
      const int q = 16 * b + w, j = q - (P >> 3);
      u64 x = q < 21 ? tpl[q] : 0;
      x |= ins(q, 121, r, NB) | ins(q, 129 + NB, r, NB);
      const u64 cur = ldbe(val, j, SEG / 8);
      x |= s ? (prev << (8 * (8 - s))) | (cur >> (8 * s)) : cur;
      prev = cur;
      if (q == (L >> 3)) x |= 0x80ull << (8 * (7 - (L & 7)));
      if (q == 16 * NBLK - 1) x = (u64)L * 8;
      W[w] = x;
    }
    compress(st, W);
  }
  cpy8(out, st);
}
L1_HD void a_leaf(const Call& k, u64 r, const uint8_t* val, u64 out[8]) {
  if (r < 256) a_leaf_nb<1>(k, r, val, out);
  else if (r < 65536) a_leaf_nb<2>(k, r, val, out);
  else a_leaf_nb<3>(k, r, val, out);
}

// node(level, index, L, R): the level's first-block state, then blocks 2 and 3 (index at byte 2, L at 10 + NI, R at 82 + NI)
template <int NI>
L1_HD void node_ni(const Call& k, int level, u64 index, const u64* Lc, const u64* Rc, u64 out[8]) {
  const u64* tpl = k.node_tpl[NI - 1];
  cpy8(out, k.node_mid[level]);
#pragma unroll
  for (int b = 0; b < 2; b++) {
    u64 W[16];
#pragma unroll
    for (int w = 0; w < 16; w++) {
      const int q = 16 * b + w;
      W[w] = tpl[q] | ins(q, 2, index, NI) | dig<10 + NI>(Lc, q) | dig<82 + NI>(Rc, q);
    }
    compress(out, W);
  }
}
L1_HD void node_of(const Call& k, int level, u64 index, const u64* Lc, const u64* Rc, u64 out[8]) {
  if (index < 256) node_ni<1>(k, level, index, Lc, Rc, out);
  else if (index < 65536) node_ni<2>(k, level, index, Lc, Rc, out);
  else node_ni<3>(k, level, index, Lc, Rc, out);
}

// x to root(A_j): b = x ^ M_key y, and root(A_j) = h = SHA-512(leaf_prefix(key) || b || c)
L1_HD void instance(const Call& k, const u64* x, u64 h[8], u64 ra[8]) {
  u64 W[16];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    W[i] = x[i] ^ k.my[i];
    W[8 + i] = k.c[i];
  }
  cpy8(h, k.hmid);
  compress(h, W);
#pragma unroll
  for (int i = 0; i < 16; i++) W[i] = 0;
  W[0] = 0x80ull << 56;
  W[15] = 256 * 8;
  compress(h, W);
  cpy8(ra, h);
}
L1_HD void seed_of(const Call& k, const u64* ra, u64 seed[8]) {
  u64 W[16];
  iv8(seed);
#pragma unroll
  for (int b = 0; b < 2; b++) {
#pragma unroll
    for (int w = 0; w < 16; w++) {
      const int q = 16 * b + w;
      W[w] = q >= 12 && q < 20 ? ra[q - 12] ^ k.call : k.sb[q];
    }
    compress(seed, W);
  }
}
L1_HD void noise_of(const Call& k, const u64* seed, u64 t, u64 out[8]) {
  u64 W[16] = {k.ntag[0], k.ntag[1], k.ntag[2], seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], seed[6], seed[7],
               t, 0x80ull << 56, 0, 0, 96 * 8};
  iv8(out);
  compress(out, W);
}
}  // namespace l1
