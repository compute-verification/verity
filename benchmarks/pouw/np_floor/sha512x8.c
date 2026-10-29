// SHA-512's compression eight messages at a time in AVX-512's 64-bit lanes (VPRORQ for every rotate, VPTERNLOGQ for
// the three-input XORs, Ch and Maj): the host CPU's best SHA-512 for independent hm96 rows. Each thread runs chains
// of compressions whose 16 message words are the previous state's 8 words and their complements, as sha.cu does on the
// card; lanes 0, 3 and 7 are checked against a scalar reference, and the scalar reference against FIPS 180-4's
// "abc" vector. A second form runs two sets of 8 chains a thread interleaved (compress8x2, its lanes checked too). The
// core's clock under 512-bit integer work comes from a dependent VPADDQ chain, so cycles a compression follow even in a
// guest. Prints bytes compressed a second on 1 thread and on N, for both forms, and the clocks.
//   gcc -O3 -march=sapphirerapids -fopenmp sha512x8.c -o sha512x8; ./sha512x8 [THREADS] [SECONDS]
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
static const uint64_t IV[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
                               0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                               0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static inline uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

static void compress1(uint64_t s[8], const uint64_t m[16]) {
  uint64_t w[80];
  for (int i = 0; i < 16; i++) w[i] = m[i];
  for (int i = 16; i < 80; i++)
    w[i] = (rotr(w[i - 2], 19) ^ rotr(w[i - 2], 61) ^ (w[i - 2] >> 6)) + w[i - 7] +
           (rotr(w[i - 15], 1) ^ rotr(w[i - 15], 8) ^ (w[i - 15] >> 7)) + w[i - 16];
  uint64_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
  for (int i = 0; i < 80; i++) {
    uint64_t t1 = h + (rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
    uint64_t t2 = (rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
  }
  s[0] += a, s[1] += b, s[2] += c, s[3] += d, s[4] += e, s[5] += f, s[6] += g, s[7] += h;
}

#define XOR3(x, y, z) _mm512_ternarylogic_epi64(x, y, z, 0x96)
#define CH(e, f, g) _mm512_ternarylogic_epi64(e, f, g, 0xCA)
#define MAJ(a, b, c) _mm512_ternarylogic_epi64(a, b, c, 0xE8)
#define ROR(x, n) _mm512_ror_epi64(x, n)

static inline void compress8(__m512i s[8], const __m512i m[16]) {
  __m512i w[16];
  for (int i = 0; i < 16; i++) w[i] = m[i];
  __m512i a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma GCC unroll 80
  for (int i = 0; i < 80; i++) {
    if (i >= 16) {
      const __m512i w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
      const __m512i s0 = XOR3(ROR(w15, 1), ROR(w15, 8), _mm512_srli_epi64(w15, 7));
      const __m512i s1 = XOR3(ROR(w2, 19), ROR(w2, 61), _mm512_srli_epi64(w2, 6));
      w[i & 15] = _mm512_add_epi64(_mm512_add_epi64(w[i & 15], s0), _mm512_add_epi64(w[(i - 7) & 15], s1));
    }
    const __m512i kw = _mm512_add_epi64(_mm512_set1_epi64((long long)K[i]), w[i & 15]);
    const __m512i t1 = _mm512_add_epi64(_mm512_add_epi64(h, XOR3(ROR(e, 14), ROR(e, 18), ROR(e, 41))),
                                        _mm512_add_epi64(CH(e, f, g), kw));
    const __m512i t2 = _mm512_add_epi64(XOR3(ROR(a, 28), ROR(a, 34), ROR(a, 39)), MAJ(a, b, c));
    h = g, g = f, f = e, e = _mm512_add_epi64(d, t1), d = c, c = b, b = a, a = _mm512_add_epi64(t1, t2);
  }
  s[0] = _mm512_add_epi64(s[0], a), s[1] = _mm512_add_epi64(s[1], b), s[2] = _mm512_add_epi64(s[2], c);
  s[3] = _mm512_add_epi64(s[3], d), s[4] = _mm512_add_epi64(s[4], e), s[5] = _mm512_add_epi64(s[5], f);
  s[6] = _mm512_add_epi64(s[6], g), s[7] = _mm512_add_epi64(s[7], h);
}

// compress8 on two independent states at once, so the core can overlap one's round chain with the other's.
static inline void compress8x2(__m512i s[2][8], const __m512i m[2][16]) {
  __m512i w[2][16], a[2], b[2], c[2], d[2], e[2], f[2], g[2], h[2];
  for (int j = 0; j < 2; j++) {
    for (int i = 0; i < 16; i++) w[j][i] = m[j][i];
    a[j] = s[j][0], b[j] = s[j][1], c[j] = s[j][2], d[j] = s[j][3];
    e[j] = s[j][4], f[j] = s[j][5], g[j] = s[j][6], h[j] = s[j][7];
  }
#pragma GCC unroll 80
  for (int i = 0; i < 80; i++) {
#pragma GCC unroll 2
    for (int j = 0; j < 2; j++) {
      if (i >= 16) {
        const __m512i w15 = w[j][(i - 15) & 15], w2 = w[j][(i - 2) & 15];
        const __m512i s0 = XOR3(ROR(w15, 1), ROR(w15, 8), _mm512_srli_epi64(w15, 7));
        const __m512i s1 = XOR3(ROR(w2, 19), ROR(w2, 61), _mm512_srli_epi64(w2, 6));
        w[j][i & 15] = _mm512_add_epi64(_mm512_add_epi64(w[j][i & 15], s0), _mm512_add_epi64(w[j][(i - 7) & 15], s1));
      }
      const __m512i kw = _mm512_add_epi64(_mm512_set1_epi64((long long)K[i]), w[j][i & 15]);
      const __m512i t1 = _mm512_add_epi64(_mm512_add_epi64(h[j], XOR3(ROR(e[j], 14), ROR(e[j], 18), ROR(e[j], 41))),
                                          _mm512_add_epi64(CH(e[j], f[j], g[j]), kw));
      const __m512i t2 = _mm512_add_epi64(XOR3(ROR(a[j], 28), ROR(a[j], 34), ROR(a[j], 39)), MAJ(a[j], b[j], c[j]));
      h[j] = g[j], g[j] = f[j], f[j] = e[j], e[j] = _mm512_add_epi64(d[j], t1);
      d[j] = c[j], c[j] = b[j], b[j] = a[j], a[j] = _mm512_add_epi64(t1, t2);
    }
  }
  for (int j = 0; j < 2; j++) {
    s[j][0] = _mm512_add_epi64(s[j][0], a[j]), s[j][1] = _mm512_add_epi64(s[j][1], b[j]);
    s[j][2] = _mm512_add_epi64(s[j][2], c[j]), s[j][3] = _mm512_add_epi64(s[j][3], d[j]);
    s[j][4] = _mm512_add_epi64(s[j][4], e[j]), s[j][5] = _mm512_add_epi64(s[j][5], f[j]);
    s[j][6] = _mm512_add_epi64(s[j][6], g[j]), s[j][7] = _mm512_add_epi64(s[j][7], h[j]);
  }
}

// One thread's 8 chains of `blocks` compressions; lane l's seed is seed + l. Returns lane 0's and lane 7's state[0].
static void chain8(uint64_t seed, long blocks, uint64_t out[8]) {
  __m512i s[8], m[16];
  const __m512i lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  for (int i = 0; i < 8; i++) s[i] = _mm512_xor_si512(_mm512_set1_epi64((long long)IV[i]),
                                                     _mm512_add_epi64(_mm512_set1_epi64((long long)seed), lanes));
  for (long r = 0; r < blocks; r++) {
    for (int i = 0; i < 8; i++) m[i] = s[i], m[8 + i] = _mm512_xor_si512(s[i], _mm512_set1_epi64(-1));
    compress8(s, m);
  }
  _mm512_storeu_si512(out, s[0]);
}

// chain8's chains, 16 a thread: two sets of 8 (seeds seed + l and seed + 8 + l) through compress8x2. out[0..7] is the
// first set's state[0], out[8..15] the second's.
static void chain8x2(uint64_t seed, long blocks, uint64_t out[16]) {
  __m512i s[2][8], m[2][16];
  const __m512i lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  for (int j = 0; j < 2; j++)
    for (int i = 0; i < 8; i++)
      s[j][i] = _mm512_xor_si512(_mm512_set1_epi64((long long)IV[i]),
                                 _mm512_add_epi64(_mm512_set1_epi64((long long)(seed + 8 * j)), lanes));
  for (long r = 0; r < blocks; r++) {
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 8; i++) m[j][i] = s[j][i], m[j][8 + i] = _mm512_xor_si512(s[j][i], _mm512_set1_epi64(-1));
    compress8x2(s, m);
  }
  _mm512_storeu_si512(out, s[0][0]);
  _mm512_storeu_si512(out + 8, s[1][0]);
}

// The core's clock while it runs 512-bit integer work: a chain of dependent VPADDQ (latency 1 cycle on Golden Cove),
// and the same with scalar ADD. Cycles a second.
static double clock_hz(int wide) {
  const long n = 400000000;
  double t0 = now();
  if (wide) {
    __m512i x = _mm512_set1_epi64(1), y = _mm512_set1_epi64(3);
    for (long i = 0; i < n; i += 8)
      __asm__ volatile(
          "vpaddq %1, %0, %0\n\tvpaddq %1, %0, %0\n\tvpaddq %1, %0, %0\n\tvpaddq %1, %0, %0\n\t"
          "vpaddq %1, %0, %0\n\tvpaddq %1, %0, %0\n\tvpaddq %1, %0, %0\n\tvpaddq %1, %0, %0"
          : "+v"(x)
          : "v"(y));
    uint64_t o[8];
    _mm512_storeu_si512(o, x);
    if (o[0] == 42) printf(" ");
  } else {
    uint64_t x = 1;
    for (long i = 0; i < n; i += 8)
      __asm__ volatile("add %0, %0\n\tadd %0, %0\n\tadd %0, %0\n\tadd %0, %0\n\t"
                       "add %0, %0\n\tadd %0, %0\n\tadd %0, %0\n\tadd %0, %0"
                       : "+r"(x));
    if (x == 42) printf(" ");
  }
  return n / (now() - t0);
}

static uint64_t chain1(uint64_t seed, long blocks) {
  uint64_t s[8], m[16];
  for (int i = 0; i < 8; i++) s[i] = IV[i] ^ seed;
  for (long r = 0; r < blocks; r++) {
    for (int i = 0; i < 8; i++) m[i] = s[i], m[8 + i] = ~s[i];
    compress1(s, m);
  }
  return s[0];
}

int main(int argc, char** argv) {
  const int threads = argc > 1 ? atoi(argv[1]) : omp_get_max_threads();
  const double secs = argc > 2 ? atof(argv[2]) : 1.0;
  uint64_t s[8], m[16] = {0x6162638000000000ull, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24};
  for (int i = 0; i < 8; i++) s[i] = IV[i];
  compress1(s, m);
  const int abc_ok = s[0] == 0xddaf35a193617abaull;
  uint64_t o[8];
  chain8(1000, 64, o);
  const int lanes_ok = o[0] == chain1(1000, 64) && o[7] == chain1(1007, 64) && o[3] == chain1(1003, 64);
  uint64_t o2[16];
  chain8x2(1000, 64, o2);
  const int x2_ok = o2[0] == chain1(1000, 64) && o2[7] == chain1(1007, 64) && o2[8] == chain1(1008, 64) &&
                    o2[15] == chain1(1015, 64);
  long blocks = 1024;
  for (;;) {
    double t0 = now();
    chain8(1, blocks, o);
    double t = now() - t0;
    if (t > 0.1) {
      blocks = (long)(blocks * secs / t);
      break;
    }
    blocks *= 4;
  }
  printf("{\"abc_ok\": %d, \"lanes_ok\": %d, \"x2_ok\": %d", abc_ok, lanes_ok, x2_ok);
  printf(", \"t1_hz_scalar\": %.6g, \"t1_hz_avx512\": %.6g", clock_hz(0), clock_hz(1));
  for (int pass = 0; pass < 2; pass++) {
    const int nt = pass ? threads : 1;
    for (int x2 = 0; x2 < 2; x2++) {
      uint64_t sink = 0;
      double t0 = now();
#pragma omp parallel num_threads(nt) reduction(^ : sink)
      {
        uint64_t q[16];
        if (x2)
          chain8x2(64 * omp_get_thread_num(), blocks / 2, q);
        else
          chain8(64 * omp_get_thread_num(), blocks, q);
        sink ^= q[0] ^ q[5];
      }
      double wall = now() - t0;
      printf(", \"%st%d_bytes_per_s\": %.6g, \"%st%d_sink\": %llu", x2 ? "x2_" : "", nt,
             8.0 * 128 * (blocks / (x2 ? 2 : 1)) * (x2 ? 2 : 1) * nt / wall, x2 ? "x2_" : "", nt,
             (unsigned long long)(sink & 0xffff));
    }
    double hz = 0;
#pragma omp parallel num_threads(nt) reduction(+ : hz)
    hz += clock_hz(1);
    printf(", \"t%d_hz_avx512_mean\": %.6g", nt, hz / nt);
  }
  printf("}\n");
  return !(abc_ok && lanes_ok && x2_ok);
}
