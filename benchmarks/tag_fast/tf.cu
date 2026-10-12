// The bandwidth audit's certified tag, ro-blake3/v1 (verity/core/protocols/bandwidth/tags.py), fused into the FP8 decode
// GEMM on the RTX PRO 6000 (sm_120), and what it costs beside the native GEMM.
//
// The store is the word codec's plain fragment layout (benchmarks/word_codec): for each 16-row block nb of W (N x K, e4m3),
// its K / 32 tiles in order, a tile 32 lanes x 16 bytes, lane l = 4g + tq holding its m16n8k32 A fragment {a0, a1, a2, a3}.
// A block of the tag is one lane's 16 bytes over SPAN consecutive tiles of one row block (16 SPAN bytes), so a warp tags
// exactly the bytes it multiplies, in registers:
//
//   unit u = nb * (K / 32 / SPAN) + s     the span s of row block nb: tiles s SPAN .. s SPAN + SPAN - 1
//   block j = jbase + 32 u + lane         E_j = lane's uint4 of each tile of the unit, in tile order
//   y_j = BLAKE3-keyed(K, u64be j || E_j) the message's 64-byte block c is (carry, the 4 uint4 of tiles 4c .. 4c + 3 minus
//                                         the last 8 bytes, which carry into block c + 1); carry 0 is u64be j
//
// SPAN 32 is 512-byte blocks (one chunk of 520 bytes: nine compressions), SPAN 64 1,024 (a full chunk, an 8-byte chunk and
// the root parent: 18). The leaves and tree are #1805's (b3_leaves and the RFC 6962 SHA-512 tree, from
// benchmarks/bandwidth/tagger.cu): 16 consecutive tags a leaf, the two half-warps of a unit.
//
// BLAKE3's G has 4 adds, 4 XORs and 4 rotations; on sm_120 the integer ALU runs at half rate and IMAD issues to the FMA
// pipe beside it. The variants move adds (x * 1 + y, the 1 from constant memory so ptxas cannot fold it) and rotations
// (x * 2^(32 - n) low | high: two IMADs) to the FMA pipe:
//
//   V0  a += b + x: IADD3; c += d: IADD3; rotations 16, 8: PRMT; 12, 7: SHF       12 ALU
//   V1  every add on IMAD (a: two, c: one)                                       8 ALU, 6 FMA
//   V2  the a-adds on IMAD                                                       10 ALU, 4 FMA
//   V3  V1 and the 12- and 7-bit rotations as IMAD pairs                         6 ALU, 10 FMA
//   V4  the c-adds on IMAD                                                       10 ALU, 2 FMA
//
// Kernels (extern "C" launchers at the end): the fused GEMM (modes: GEMM only, GEMM + tags to memory, GEMM + tags XORed
// into one word, tags only), the word codec's plain FP8 GEMM, b3_leaves and the tree, compression chains (BLAKE3 variants
// at 7 rounds and, uncertified, fewer; Keccak-p[1600, 12], TurboSHAKE128's permutation; SHA-512), one-shot self-tests,
// a streaming read, spin and empty.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace tf {

constexpr unsigned FULL = 0xffffffffu;
constexpr int LANES = 32;
constexpr int FW = 4;  // warps a CTA of the fused GEMM, one 16-row block each

// ---------------------------------------------------------------------------------------------------------------------
// SHA-512 (benchmarks/bandwidth/tagger.cu).

#define SHA512_K                                                                                                 \
  {0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, \
   0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, \
   0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, \
   0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, \
   0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, 0x983e5152ee66dfabULL, \
   0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, \
   0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, \
   0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL, \
   0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL, \
   0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, 0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, \
   0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, \
   0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL, \
   0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL, 0xca273eceea26619cULL, \
   0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, \
   0x113f9804bef90daeULL, 0x1b710b35131c471bULL, 0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, \
   0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL}
#define SHA512_IV                                                                                               \
  {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, \
   0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL}

__constant__ uint64_t K_DEV[80] = SHA512_K;
__constant__ uint64_t IV_DEV[8] = SHA512_IV;
static const uint64_t K_HOST[80] = SHA512_K;
static const uint64_t IV_HOST[8] = SHA512_IV;

struct Mid {
  uint64_t h[8];
};

__host__ __device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

__host__ __device__ __forceinline__ void sha512_compress(uint64_t s[8], uint64_t w[16]) {
#ifdef __CUDA_ARCH__
  const uint64_t* K = K_DEV;
#else
  const uint64_t* K = K_HOST;
#endif
  uint64_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll 1
  for (int r = 0; r < 80; r += 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      if (r) {
        const uint64_t w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
        w[t] += (rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7)) + w[(t + 9) & 15] +
                (rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6));
      }
      const uint64_t t1 = h + (rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41)) + ((e & f) ^ (~e & g)) + K[r + t] + w[t];
      const uint64_t t2 = (rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
  }
  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;
}

__device__ __forceinline__ uint32_t bswap32(uint32_t x) { return __byte_perm(x, 0, 0x0123); }

static void sha512_host(const uint8_t* msg, size_t len, uint64_t out[8]) {
  uint64_t s[8];
  memcpy(s, IV_HOST, sizeof s);
  std::vector<uint8_t> m(msg, msg + len);
  m.push_back(0x80);
  while (m.size() % 128 != 112) m.push_back(0);
  for (int i = 0; i < 8; ++i) m.push_back(0);
  const uint64_t bits = (uint64_t)len * 8;
  for (int i = 7; i >= 0; --i) m.push_back((uint8_t)(bits >> (8 * i)));
  for (size_t off = 0; off < m.size(); off += 128) {
    uint64_t w[16];
    for (int i = 0; i < 16; ++i) {
      uint64_t v = 0;
      for (int k = 0; k < 8; ++k) v = (v << 8) | m[off + 8 * i + k];
      w[i] = v;
    }
    sha512_compress(s, w);
  }
  memcpy(out, s, sizeof s);
}

static const char TAG_NAME[] = "ro-blake3/v1";
static const char TAG_DOMAIN[] = "verity/bandwidth/tag/v1";        // + its NUL: 24 bytes
static const char LEAF_DOMAIN[] = "verity/bandwidth/tag-leaf/v1";  // + its NUL: 29 bytes

static void append_frame(std::vector<uint8_t>& out, char tag, const uint8_t* body, size_t n) {
  out.push_back((uint8_t)tag);
  for (int i = 7; i >= 0; --i) out.push_back((uint8_t)((uint64_t)n >> (8 * i)));
  out.insert(out.end(), body, body + n);
}

// Z(domain || NUL || frame(name) || frame(seed)), one 128-byte block.
static std::vector<uint8_t> prefix_block(const char* domain, const uint8_t seed[32]) {
  std::vector<uint8_t> p(domain, domain + strlen(domain) + 1);
  append_frame(p, 's', (const uint8_t*)TAG_NAME, strlen(TAG_NAME));
  append_frame(p, 'b', seed, 32);
  p.resize(128, 0);
  return p;
}

static Mid prefix_midstate(const char* domain, const uint8_t seed[32]) {
  std::vector<uint8_t> p = prefix_block(domain, seed);
  Mid m;
  memcpy(m.h, IV_HOST, sizeof m.h);
  uint64_t w[16];
  for (int i = 0; i < 16; ++i) {
    uint64_t v = 0;
    for (int k = 0; k < 8; ++k) v = (v << 8) | p[8 * i + k];
    w[i] = v;
  }
  sha512_compress(m.h, w);
  return m;
}

// ---------------------------------------------------------------------------------------------------------------------
// BLAKE3's compression in the five variants.

constexpr uint32_t IV0 = 0x6A09E667u, IV1 = 0xBB67AE85u, IV2 = 0x3C6EF372u, IV3 = 0xA54FF53Au;
constexpr uint32_t START = 1, END = 2, PARENT = 4, ROOT = 8, KEYED = 16;

__constant__ uint32_t kKey[8];
__constant__ uint32_t kMul[5];  // 1, 2^20, 2^25, 2^16, 2^24: tf_init sets them, so ptxas cannot fold an IMAD into an add

struct Mul {
  uint32_t one, p20, p25, p16, p24;
};

__device__ __forceinline__ Mul muls() { return Mul{kMul[0], kMul[1], kMul[2], kMul[3], kMul[4]}; }

__device__ __forceinline__ uint32_t imad(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ uint32_t imadhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ uint32_t imul(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.lo.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// rotr(x, n) with p = 2^(32 - n): x p's low word is x << (32 - n) and its high word x >> n, disjoint, so their sum is
// their OR.
__device__ __forceinline__ uint32_t rotr_fma(uint32_t x, uint32_t p) { return imadhi(x, p, imul(x, p)); }

template <int V>
struct Var;
template <>
struct Var<0> {
  static constexpr int A3 = 0, A2 = 0, R = 0;
};
template <>
struct Var<1> {
  static constexpr int A3 = 1, A2 = 1, R = 0;
};
template <>
struct Var<2> {
  static constexpr int A3 = 1, A2 = 0, R = 0;
};
template <>
struct Var<3> {
  static constexpr int A3 = 1, A2 = 1, R = 1;
};
template <>
struct Var<4> {
  static constexpr int A3 = 0, A2 = 1, R = 0;
};

template <int V>
__device__ __forceinline__ void g(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t x, uint32_t y,
                                  const Mul& k) {
  using P = Var<V>;
  a = P::A3 ? imad(b, k.one, imad(x, k.one, a)) : a + b + x;
  d ^= a;
  d = __byte_perm(d, 0, 0x1032);
  c = P::A2 ? imad(d, k.one, c) : c + d;
  b ^= c;
  b = P::R ? rotr_fma(b, k.p20) : __funnelshift_r(b, b, 12);
  a = P::A3 ? imad(b, k.one, imad(y, k.one, a)) : a + b + y;
  d ^= a;
  d = __byte_perm(d, 0, 0x0321);
  c = P::A2 ? imad(d, k.one, c) : c + d;
  b ^= c;
  b = P::R ? rotr_fma(b, k.p25) : __funnelshift_r(b, b, 7);
}

// One compression; `out` gets the first 8 words of the output (the next chaining value, or under ROOT the hash). It may
// alias `cv`. ROUNDS < 7 is not BLAKE3: a reduced-round cost probe, uncertified. ROLL: the rounds in a loop (the message
// permutation as moves), a seventh of the code, for paths whose instruction fetch costs more than their issue.
template <int V, int ROUNDS = 7, bool ROLL = false>
__device__ __forceinline__ void b3(const uint32_t cv[8], const uint32_t min[16], uint32_t counter, uint32_t len,
                                   uint32_t flags, uint32_t out[8], const Mul& k) {
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = min[i];
  uint32_t v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
  uint32_t v8 = IV0, v9 = IV1, v10 = IV2, v11 = IV3, v12 = counter, v13 = 0, v14 = len, v15 = flags;
  auto round = [&]() {
    g<V>(v0, v4, v8, v12, m[0], m[1], k);
    g<V>(v1, v5, v9, v13, m[2], m[3], k);
    g<V>(v2, v6, v10, v14, m[4], m[5], k);
    g<V>(v3, v7, v11, v15, m[6], m[7], k);
    g<V>(v0, v5, v10, v15, m[8], m[9], k);
    g<V>(v1, v6, v11, v12, m[10], m[11], k);
    g<V>(v2, v7, v8, v13, m[12], m[13], k);
    g<V>(v3, v4, v9, v14, m[14], m[15], k);
  };
  auto permute = [&]() {
    const uint32_t t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13],
                            m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = t[i];
  };
  if constexpr (ROLL) {
#pragma unroll 1
    for (int r = 0; r < ROUNDS - 1; ++r) {
      round();
      permute();
    }
    round();
  } else {
#pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
      round();
      if (r < ROUNDS - 1) permute();
    }
  }
  out[0] = v0 ^ v8;
  out[1] = v1 ^ v9;
  out[2] = v2 ^ v10;
  out[3] = v3 ^ v11;
  out[4] = v4 ^ v12;
  out[5] = v5 ^ v13;
  out[6] = v6 ^ v14;
  out[7] = v7 ^ v15;
}

// ---------------------------------------------------------------------------------------------------------------------
// The lane stream: one block's tag over the tiles a lane loads, a 64-byte compression per four tiles.

template <int V, int SPAN>
struct LaneTag {
  static_assert(SPAN == 32 || SPAN == 64 || SPAN == 128, "a block is 512, 1,024 or 2,048 bytes");
  uint32_t cv[8], c0, c1;
  uint32_t left[8];  // SPAN 128: chunk 0's chaining value

  __device__ __forceinline__ void begin(uint64_t j) {
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = kKey[i];
    c0 = bswap32((uint32_t)(j >> 32));
    c1 = bswap32((uint32_t)j);
  }

  // Quad qd: tiles 4 qd .. 4 qd + 3 of the span.
  __device__ __forceinline__ void absorb(const uint4 (&t)[4], int qd, const Mul& k) {
    const uint32_t m[16] = {c0,     c1,     t[0].x, t[0].y, t[0].z, t[0].w, t[1].x, t[1].y,
                            t[1].z, t[1].w, t[2].x, t[2].y, t[2].z, t[2].w, t[3].x, t[3].y};
    c0 = t[3].z;
    c1 = t[3].w;
    const int ci = qd & 15;
    const uint32_t fl = KEYED | (ci == 0 ? START : 0) | (SPAN > 32 && ci == 15 ? END : 0);
    b3<V>(cv, m, (uint32_t)(qd >> 4), 64, fl, cv, k);
    if constexpr (SPAN == 128) {
      if (qd == 15) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          left[i] = cv[i];
          cv[i] = kKey[i];
        }
      }
    }
  }

  __device__ __forceinline__ void finish(uint32_t y[8], const Mul& k) {
    uint32_t key[8], m[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) key[i] = kKey[i];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = 0;
    m[0] = c0;
    m[1] = c1;
    if constexpr (SPAN == 32) {
      b3<V>(cv, m, 0, 8, KEYED | END | ROOT, y, k);
    } else if constexpr (SPAN == 64) {
      uint32_t tail[8];
      b3<V>(key, m, 1, 8, KEYED | START | END, tail, k);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        m[i] = cv[i];
        m[8 + i] = tail[i];
      }
      b3<V>(key, m, 0, 64, KEYED | PARENT | ROOT, y, k);
    } else {
      uint32_t tail[8], p[8];
      b3<V>(key, m, 2, 8, KEYED | START | END, tail, k);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        m[i] = left[i];
        m[8 + i] = cv[i];
      }
      b3<V>(key, m, 0, 64, KEYED | PARENT, p, k);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        m[i] = p[i];
        m[8 + i] = tail[i];
      }
      b3<V>(key, m, 0, 64, KEYED | PARENT | ROOT, y, k);
    }
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// The FP8 GEMM, plain and fused.

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A streaming load whose L2 miss fetches the whole 256-byte block (benchmarks/bandwidth/tagger.cu's ld_l2_256).
__device__ __forceinline__ uint4 ldw(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

constexpr int MODE_GEMM = 0, MODE_TAGS = 1, MODE_SINK = 2, MODE_TAGONLY = 3;

// A CTA of FW warps takes span s of row blocks nb = FW * (blockIdx.x / nspans) + warp, x's columns of the span in shared
// memory (stride 16 SPAN + 16 bytes: conflict-free B loads). Each warp walks its unit a quad (four tiles) at a time with
// PD quads' loads in flight, the quad's MMAs, then its compression. MODE_TAGS writes y_j as 8 little-endian words at
// tags[8 (j - jbase)]; MODE_SINK XORs every tag word into *xr. PARTIALS: each span's sums to y[(s M + b) N + n] (the gate)
// instead of atomically adding them to y[b N + n].
template <int V, int SPAN, int NB8, int MODE, int PD, bool PARTIALS>
__global__ void __launch_bounds__(FW * 32) fused_kernel(const uint4* __restrict__ wstore, const uint8_t* __restrict__ x,
                                                        float* __restrict__ y, uint32_t* __restrict__ tags, int M, int N,
                                                        int Kd, uint64_t jbase, unsigned int* __restrict__ xr) {
  constexpr bool GEMM = MODE != MODE_TAGONLY, TAG = MODE != MODE_GEMM;
  constexpr int COLS = 32 * SPAN, XS = COLS + 16, NQ = SPAN / 4, DEPTH = PD + 1;
  extern __shared__ __align__(16) uint8_t xs[];
  const int nspans = Kd / COLS;
  const int s = blockIdx.x % nspans;
  const int nb = (blockIdx.x / nspans) * FW + threadIdx.x / 32;
  const int lane = threadIdx.x & 31, g8 = lane >> 2, tq = lane & 3;
  if constexpr (GEMM) {
    for (int e = threadIdx.x; e < NB8 * 8 * COLS / 16; e += FW * 32) {
      const int b = e / (COLS / 16), c = e % (COLS / 16);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (b < M) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)s * COLS) + c);
      *reinterpret_cast<uint4*>(xs + b * XS + 16 * c) = v;
    }
    __syncthreads();
  }
  if (nb * 16 >= N) return;
  const uint64_t unit = (uint64_t)nb * nspans + s;
  const uint4* wp = wstore + ((size_t)nb * (Kd / 32) + (size_t)s * SPAN) * LANES + lane;
  const Mul k = muls();

  float acc[NB8][4];
#pragma unroll
  for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  LaneTag<V, SPAN> lt;
  if constexpr (TAG) lt.begin(jbase + unit * LANES + lane);

  uint4 buf[DEPTH][4];
  auto load = [&](uint4 (&q)[4], int qd) {
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = ldw(wp + (4 * qd + i) * LANES);
  };
  auto step = [&](const uint4 (&q)[4], int qd) {
    if constexpr (GEMM) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t a[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
        const int col = 32 * (4 * qd + i) + 4 * tq;
#pragma unroll
        for (int j = 0; j < NB8; ++j) {
          const uint8_t* row = xs + (8 * j + g8) * XS;
          mma_e4m3(acc[j], a, *reinterpret_cast<const uint32_t*>(row + col),
                   *reinterpret_cast<const uint32_t*>(row + col + 16));
        }
      }
    }
    if constexpr (TAG) lt.absorb(q, qd, k);
  };

#pragma unroll
  for (int p = 0; p < PD; ++p)
    if (p < NQ) load(buf[p], p);
#pragma unroll 1
  for (int qb = 0; qb < NQ; qb += DEPTH) {
#pragma unroll
    for (int u = 0; u < DEPTH; ++u) {
      const int qd = qb + u;
      if (NQ % DEPTH == 0 || qd < NQ) {
        if (qd + PD < NQ) load(buf[(u + PD) % DEPTH], qd + PD);
        step(buf[u], qd);
      }
    }
  }

  if constexpr (TAG) {
    uint32_t t[8];
    lt.finish(t, k);
    if constexpr (MODE == MODE_SINK) {
      uint32_t a = t[0] ^ t[1] ^ t[2] ^ t[3] ^ t[4] ^ t[5] ^ t[6] ^ t[7];
#pragma unroll
      for (int d = 16; d; d >>= 1) a ^= __shfl_xor_sync(FULL, a, d);
      if (lane == 0) atomicXor(xr, a);
    } else {
      uint4* o = reinterpret_cast<uint4*>(tags + (unit * LANES + lane) * 8);
      o[0] = make_uint4(t[0], t[1], t[2], t[3]);
      o[1] = make_uint4(t[4], t[5], t[6], t[7]);
    }
  }
  if constexpr (GEMM) {
#pragma unroll
    for (int j = 0; j < NB8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int n = nb * 16 + g8 + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
        if (b < M) {
          if constexpr (PARTIALS)
            y[((size_t)s * M + b) * N + n] = acc[j][e];
          else
            atomicAdd(y + (size_t)b * N + n, acc[j][e]);
        }
      }
    }
  }
}

// The word codec's plain FP8 GEMM (benchmarks/word_codec/wc.cu, gemm_kernel at KR = 0): a warp per 16 rows x 512
// columns, all 16 tiles' loads in flight at once, the CTA's chunk of x in shared memory.
constexpr int TASK_TILES = 16, TASK_COLS = 32 * TASK_TILES, PXS = TASK_COLS + 16;

template <int NB8, bool PARTIALS>
__global__ void __launch_bounds__(FW * 32) plain_kernel(const uint4* __restrict__ wstore, const uint8_t* __restrict__ x,
                                                        float* __restrict__ y, int M, int N, int Kd) {
  __shared__ __align__(16) uint8_t xs[NB8 * 8][PXS];
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks;
  const int nb = (blockIdx.x / nchunks) * FW + threadIdx.x / 32;
  const int lane = threadIdx.x & 31, g8 = lane >> 2, tq = lane & 3;
  const uint64_t q = (uint64_t)nb * nchunks + kc;
  for (int e = threadIdx.x; e < NB8 * 8 * TASK_COLS / 16; e += FW * 32) {
    const int b = e / (TASK_COLS / 16), c = e % (TASK_COLS / 16);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < M) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)kc * TASK_COLS) + c);
    *reinterpret_cast<uint4*>(&xs[b][16 * c]) = v;
  }
  __syncthreads();
  if (nb * 16 >= N) return;
  float acc[NB8][4];
#pragma unroll
  for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  const uint4* wp = wstore + q * TASK_TILES * LANES + lane;
  uint4 w[TASK_TILES];
#pragma unroll
  for (int tt = 0; tt < TASK_TILES; ++tt) w[tt] = __ldcs(wp + tt * LANES);
#pragma unroll
  for (int tt = 0; tt < TASK_TILES; ++tt) {
    const uint32_t a[4] = {w[tt].x, w[tt].y, w[tt].z, w[tt].w};
#pragma unroll
    for (int j = 0; j < NB8; ++j) {
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g8][32 * tt + 4 * tq]);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g8][32 * tt + 16 + 4 * tq]);
      mma_e4m3(acc[j], a, b0, b1);
    }
  }
#pragma unroll
  for (int j = 0; j < NB8; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int n = nb * 16 + g8 + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
      if (b < M) {
        if constexpr (PARTIALS)
          y[((size_t)kc * M + b) * N + n] = acc[j][e];
        else
          atomicAdd(y + (size_t)b * N + n, acc[j][e]);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Chunk blocks (layout.py's chunk_tag): a lane's 16 bytes of 64 consecutive tiles, D (1,024 bytes), is one BLAKE3 chunk,
// and L consecutive lanes' chunks are one block of L KiB. The block's bytes are each lane's D rotated left by 8 bytes,
// E_j = D_0[8:] D_0[:8] ... D_(L-1)[8:] D_(L-1)[:8], so that in u64be j || E_j chunk i is P_i || D_i[8:] with P_0 = u64be j
// and P_i = D_(i-1)[:8], whole in lane i once lane i - 1's first tile has landed; the 8-byte tail chunk is
// D_(L-1)[:8]. A lane hashes its chunk (16 compressions), then the warp runs log2 L + 1 more in lock step (2 at L = 1):
// the tail and the first parents, the further parents, and the root over the L chunks' subtree and the tail.
//
//   unit u = s * (N / 16) + nb       tiles 64 s .. 64 s + 63 of row block nb (span-major: a CTA's warps are consecutive)
//   block j = jbase + (32 / L) u + lane / L
//
// The weights stream through a ring of ST quads (4 tiles, 2 KiB a warp) in shared memory, each lane copying its own
// 16 bytes with cp.async, so ST - 1 quads are in flight without holding registers.

constexpr int CHUNK_TILES = 64, CCOLS = 32 * CHUNK_TILES, CXS = CCOLS + 16, QUAD_BYTES = 4 * 32 * 16;

__device__ __forceinline__ void cp16(void* smem, const void* g) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

template <int V, int L, int R = 7>
struct ChunkTag {
  static_assert(L == 1 || L == 2 || L == 4 || L == 8, "a block is 1, 2, 4 or 8 lanes' chunks");
  uint32_t cv[8], d0, d1;  // d0 d1: the lane's first 8 bytes (the next lane's P, or the tail chunk)

  __device__ __forceinline__ void begin() {
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = kKey[i];
  }

  // Quad qd: tiles 4 qd .. 4 qd + 3 of the unit; pos = lane % L, j the block's index.
  __device__ __forceinline__ void absorb(const uint4 (&t)[4], int qd, int pos, uint64_t j, const Mul& k) {
    uint32_t m[16] = {t[0].x, t[0].y, t[0].z, t[0].w, t[1].x, t[1].y, t[1].z, t[1].w,
                      t[2].x, t[2].y, t[2].z, t[2].w, t[3].x, t[3].y, t[3].z, t[3].w};
    if (qd == 0) {
      d0 = t[0].x;
      d1 = t[0].y;
      const uint32_t p0 = __shfl_up_sync(FULL, d0, 1), p1 = __shfl_up_sync(FULL, d1, 1);
      m[0] = pos ? p0 : bswap32((uint32_t)(j >> 32));
      m[1] = pos ? p1 : bswap32((uint32_t)j);
    }
    const uint32_t fl = KEYED | (qd == 0 ? START : 0) | (qd == 15 ? END : 0);
    b3<V, R>(cv, m, (uint32_t)pos, 64, fl, cv, k);
  }

  // The tail chunk (the lane's first 8 bytes), once d0 d1 are known; the block's is lane L - 1's.
  __device__ __forceinline__ void tail_chunk(uint32_t (&tl)[8], const Mul& k) const {
    uint32_t key[8], m[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) key[i] = kKey[i];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = 0;
    m[0] = d0;
    m[1] = d1;
    b3<V, R>(key, m, (uint32_t)L, 8u, KEYED | START | END, tl, k);
  }

  // The tag, in lanes with pos 0. Step 0: the tail chunk (every lane at L = 1, lane L - 1 otherwise) and the parents of
  // lane pairs; steps 1 .. log2 L - 1: the parents at distance 2^step; the last: the root over the subtree and the tail.
  __device__ __forceinline__ void finish(uint32_t y[8], int pos, const Mul& k) {
    constexpr int STEPS = L == 1 ? 2 : L == 2 ? 2 : L == 4 ? 3 : 4;
    uint32_t key[8], h[8], tail[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      key[i] = kKey[i];
      h[i] = cv[i];
      tail[i] = 0;
    }
#pragma unroll 1
    for (int st = 0; st < STEPS; ++st) {
      const bool last = st == STEPS - 1, tl = st == 0 && (L == 1 || pos == L - 1);
      const int d = last ? L - 1 : 1 << st;
      uint32_t m[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t o = __shfl_down_sync(FULL, last ? tail[i] : h[i], d);
        m[i] = tl ? (i == 0 ? d0 : i == 1 ? d1 : 0u) : h[i];
        m[8 + i] = tl ? 0u : o;
      }
      uint32_t out[8];
      b3<V, R>(key, m, tl ? (uint32_t)L : 0u, tl ? 8u : 64u, KEYED | (tl ? START | END : PARENT) | (last ? ROOT : 0), out,
               k);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (st == 0) tail[i] = out[i];
        if (!(L == 1 && st == 0)) h[i] = out[i];
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = h[i];
  }
};

// finish for a whole CTA: its W 32 chunk CVs and W 32 / L tail CVs into sh, then each level of parents on the threads
// that have one (W 16, W 8, ...), then the roots; block b of the CTA is tag c0 + b. The tail of a single-wave kernel is
// its finish, so it runs on as few warps as the parents need instead of on every warp's lanes at 1 / L. Each level starts
// at warp (CTA + level) mod W, so CTAs sharing an SM put their parents on different SMSPs.
template <int V, int L, int W, int R>
__device__ __forceinline__ void cta_finish(uint32_t* sh, const uint32_t (&cv)[8], const uint32_t (&tl)[8], int pos,
                                           uint32_t* tags, uint64_t c0, const Mul& k) {
  constexpr int NBK = W * 32 / L;
  uint32_t* tls = sh + W * 32 * 8;
  const int t = threadIdx.x;
  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) key[i] = kKey[i];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) sh[t * 8 + i] = cv[i];
  if (pos == L - 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) tls[(t / L) * 8 + i] = tl[i];
  }
  __syncthreads();
  int lv = blockIdx.x;
#pragma unroll
  for (int n = W * 16; n >= NBK; n /= 2, ++lv) {
    const int r = (t + 32 * (lv % W)) % (W * 32);
    uint32_t o[8];
    if (r < n) {
      uint32_t m[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = sh[r * 16 + i];
      b3<V, R>(key, m, 0u, 64u, KEYED | PARENT, o, k);
    }
    __syncthreads();
    if (r < n) {
#pragma unroll
      for (int i = 0; i < 8; ++i) sh[r * 8 + i] = o[i];
    }
    __syncthreads();
  }
  const int r = (t + 32 * (lv % W)) % (W * 32);
  if (r < NBK) {
    uint32_t m[16], o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      m[i] = sh[r * 8 + i];
      m[8 + i] = tls[r * 8 + i];
    }
    b3<V, R>(key, m, 0u, 64u, KEYED | PARENT | ROOT, o, k);
    uint4* out = reinterpret_cast<uint4*>(tags + (c0 + r) * 8);
    out[0] = make_uint4(o[0], o[1], o[2], o[3]);
    out[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

// x in shared memory: at M <= 8 its M rows of the span's 2,048 columns, loaded once; above, eighths of them (256 columns,
// two quads) double-buffered with cp.async, so the slice stays small enough for two CTAs an SM at M = 32.
template <int NB8>
struct XSlice {
  static constexpr int SL = NB8 == 1 ? 1 : 8, COLS = CCOLS / SL, STRIDE = COLS + 16, BUFS = SL == 1 ? 1 : 2,
                       QPS = 16 / SL;
};

// W warps a CTA on consecutive row blocks of one span s, each warp one unit. Modes and PARTIALS as fused_kernel's; tags at
// tags[8 (j - jbase)] from the block's first lane. R < 7 rounds is not BLAKE3 (uncertified, a cost probe).
template <int V, int L, int NB8, int MODE, int W, int ST, bool PARTIALS, int R = 7, bool PK = false>
__global__ void __launch_bounds__(W * 32) fused2_kernel(const uint4* __restrict__ wstore, const uint8_t* __restrict__ x,
                                                       float* __restrict__ y, uint32_t* __restrict__ tags, int M, int N,
                                                       int Kd, uint64_t jbase, unsigned int* __restrict__ xr) {
  constexpr bool GEMM = MODE != MODE_TAGONLY, TAG = MODE != MODE_GEMM;
  extern __shared__ __align__(16) uint8_t smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, g8 = lane >> 2, tq = lane & 3;
  using X = XSlice<NB8>;
  uint8_t* xs = smem;
  uint8_t* ring = smem + (GEMM ? X::BUFS * M * X::STRIDE : 0) + warp * ST * QUAD_BYTES + lane * 16;
  const int rts = N / 16, per_span = rts / W;
  const int s = blockIdx.x / per_span, nb = (blockIdx.x % per_span) * W + warp;
  const uint4* wp = wstore + ((size_t)nb * (Kd / 32) + (size_t)s * CHUNK_TILES) * LANES + lane;
  auto issue = [&](int qd) {
    if (qd < 16) {
      uint8_t* dst = ring + (qd % ST) * QUAD_BYTES;
#pragma unroll
      for (int i = 0; i < 4; ++i) cp16(dst + i * 512, wp + (4 * qd + i) * LANES);
    }
    cp_commit();
  };
  // slice q of x into buffer q % BUFS, in the next commit group
  auto issue_x = [&](int q) {
    uint8_t* dst = xs + (q % X::BUFS) * M * X::STRIDE;
    for (int e = threadIdx.x; e < M * X::COLS / 16; e += W * 32) {
      const int b = e / (X::COLS / 16), c = e % (X::COLS / 16);
      cp16(dst + b * X::STRIDE + 16 * c, x + (size_t)b * Kd + (size_t)s * CCOLS + q * X::COLS + 16 * c);
    }
  };
  if constexpr (GEMM && X::SL > 1) issue_x(0);
#pragma unroll
  for (int p = 0; p < ST - 1; ++p) issue(p);
  if constexpr (GEMM && X::SL == 1) {
    for (int e = threadIdx.x; e < M * CCOLS / 16; e += W * 32) {
      const int b = e / (CCOLS / 16), c = e % (CCOLS / 16);
      *reinterpret_cast<uint4*>(xs + b * X::STRIDE + 16 * c) =
          __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)s * CCOLS) + c);
    }
    __syncthreads();
  }
  const Mul k = muls();
  const uint64_t u = (uint64_t)s * rts + nb;
  const int pos = lane % L;
  const uint64_t jl = u * (32 / L) + lane / L;
  float acc[NB8][4];
#pragma unroll
  for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  auto gemm_out = [&]() {
    if constexpr (GEMM) {
#pragma unroll
      for (int j = 0; j < NB8; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int n = nb * 16 + g8 + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
          if (b < M) {
            if constexpr (PARTIALS)
              y[((size_t)s * M + b) * N + n] = acc[j][e];
            else
              atomicAdd(y + (size_t)b * N + n, acc[j][e]);
          }
        }
      }
    }
  };
  ChunkTag<V, L, R> ct;
  uint32_t tl[8];
  if constexpr (TAG) ct.begin();

#pragma unroll 1
  for (int qd = 0; qd < 16; ++qd) {
    issue(qd + ST - 1);
    // slice q + 1 was issued after the barrier at quad QPS q, so it is in group QPS q + ST: at a slice's first quad
    // wait for all but the QPS - 1 newest groups
    constexpr int XW = X::QPS - 1 < ST - 1 ? X::QPS - 1 : ST - 1;
    if (GEMM && X::SL > 1 && qd % X::QPS == 0)
      cp_wait<XW>();
    else
      cp_wait<ST - 1>();
    const uint8_t* xb = xs;
    if constexpr (GEMM && X::SL > 1) {
      if (qd % X::QPS == 0) {
        __syncthreads();
        if (qd / X::QPS + 1 < X::SL) issue_x(qd / X::QPS + 1);
      }
      xb = xs + (qd / X::QPS % X::BUFS) * M * X::STRIDE;
    }
    const uint8_t* src = ring + (qd % ST) * QUAD_BYTES;
    uint4 q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = *reinterpret_cast<const uint4*>(src + i * 512);
    if constexpr (GEMM) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t a[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
        const int col = 32 * (4 * (qd % X::QPS) + i) + 4 * tq;
#pragma unroll
        for (int j = 0; j < NB8; ++j) {
          const bool live = 8 * j + g8 < M;
          const uint8_t* row = xb + (live ? 8 * j + g8 : 0) * X::STRIDE;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + col),
                         b1 = *reinterpret_cast<const uint32_t*>(row + col + 16);
          mma_e4m3(acc[j], a, live ? b0 : 0u, live ? b1 : 0u);
        }
      }
    }
    if constexpr (TAG) ct.absorb(q, qd, pos, jbase + jl, k);
    if constexpr (TAG && PK)
      if (qd == 0) ct.tail_chunk(tl, k);
  }
  cp_wait<0>();

  if constexpr (PK) {
    static_assert(MODE == MODE_TAGS || MODE == MODE_TAGONLY, "packed finish: tags only");
    gemm_out();
    cta_finish<V, L, W, R>(reinterpret_cast<uint32_t*>(smem + (GEMM ? X::BUFS * M * X::STRIDE : 0)), ct.cv, tl, pos,
                           tags, ((uint64_t)s * rts + (blockIdx.x % per_span) * W) * (32 / L), k);
    return;
  }
  if constexpr (TAG) {
    uint32_t t[8];
    ct.finish(t, pos, k);
    if constexpr (MODE == MODE_SINK) {
      uint32_t a = pos ? 0u : t[0] ^ t[1] ^ t[2] ^ t[3] ^ t[4] ^ t[5] ^ t[6] ^ t[7];
#pragma unroll
      for (int dd = 16; dd; dd >>= 1) a ^= __shfl_xor_sync(FULL, a, dd);
      if (lane == 0) atomicXor(xr, a);
    } else if (pos == 0) {
      uint4* o = reinterpret_cast<uint4*>(tags + jl * 8);
      o[0] = make_uint4(t[0], t[1], t[2], t[3]);
      o[1] = make_uint4(t[4], t[5], t[6], t[7]);
    }
  }
  gemm_out();
}

// ChunkTag::finish for a whole fused3 CTA (FW warps, NB = 32 FW / L blocks, tags c0 .. c0 + NB - 1): the chunk CVs and
// each block's tail bytes into shared memory, then level 0 (the NB L / 2 parents of chunk pairs and the NB tail chunks),
// the further parent levels down to NB, and the NB roots, each level on only the threads it needs (rotated by bid + level
// so the CTAs of an SM load different SMSPs). 5 warp-compressions a CTA at L = 4 instead of 12.
constexpr int F3_FINISH_BYTES = FW * 32 * 8 * 4 + FW * 32 * 8 * 4 + FW * 32 * 2 * 4;

template <int V, int L>
__device__ __forceinline__ void f3_finish(uint32_t* sh, const ChunkTag<V, L>& ct, int pos, int bid, uint32_t* tags,
                                          uint64_t c0, const Mul& k) {
  static_assert(L >= 2, "at L = 1 every lane's finish (its tail and root) is already whole");
  constexpr int T = FW * 32, NB = T / L;
  uint32_t* tls = sh + T * 8;
  uint32_t* ds = tls + T * 8;
  const int t = threadIdx.x;
  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) key[i] = kKey[i];
#pragma unroll
  for (int i = 0; i < 8; ++i) sh[t * 8 + i] = ct.cv[i];
  if (pos == L - 1) {
    ds[(t / L) * 2] = ct.d0;
    ds[(t / L) * 2 + 1] = ct.d1;
  }
  __syncthreads();
  int lv = bid;
  {
    const int r = (t + 32 * (lv % FW)) % T, np = NB * L / 2;
    uint32_t o[8];
    if (r < np + NB) {
      uint32_t m[16];
      const bool tl = r >= np;
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = tl ? (i < 2 ? ds[(r - np) * 2 + i] : 0u) : sh[r * 16 + i];
      b3<V, 7, true>(key, m, tl ? (uint32_t)L : 0u, tl ? 8u : 64u, KEYED | (tl ? START | END : PARENT), o, k);
    }
    __syncthreads();
    if (r < np + NB) {
#pragma unroll
      for (int i = 0; i < 8; ++i) (r < np ? sh[r * 8 + i] : tls[(r - np) * 8 + i]) = o[i];
    }
    __syncthreads();
    ++lv;
  }
#pragma unroll
  for (int n = NB * L / 4; n >= NB; n /= 2, ++lv) {
    const int r = (t + 32 * (lv % FW)) % T;
    uint32_t o[8];
    if (r < n) {
      uint32_t m[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = sh[r * 16 + i];
      b3<V, 7, true>(key, m, 0u, 64u, KEYED | PARENT, o, k);
    }
    __syncthreads();
    if (r < n) {
#pragma unroll
      for (int i = 0; i < 8; ++i) sh[r * 8 + i] = o[i];
    }
    __syncthreads();
  }
  const int r = (t + 32 * (lv % FW)) % T;
  if (r < NB) {
    uint32_t m[16], o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      m[i] = sh[r * 8 + i];
      m[8 + i] = tls[r * 8 + i];
    }
    b3<V, 7, true>(key, m, 0u, 64u, KEYED | PARENT | ROOT, o, k);
    uint4* out = reinterpret_cast<uint4*>(tags + (c0 + r) * 8);
    out[0] = make_uint4(o[0], o[1], o[2], o[3]);
    out[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

// A fused3 launch's deferred chunk ends: it hashed each lane's first 15 quads and left, per warp of its grid (gw = CTA
// FW + warp), the lanes' chaining values (cv[2 gw + h][lane], words 4 h .. 4 h + 3) and each block's first 8 bytes
// (d[gw 32 / L + lane / L]); the launch after it hashes the last quads (from its weights, w) and finishes the tags.
struct Tail {
  const uint4* w = nullptr;
  const uint4* cv = nullptr;
  const uint2* d = nullptr;
  uint32_t* tags = nullptr;
  int nspans = 0, rts = 0, Kd = 0, warps = 0;
};

// One deferred warp's end on one warp: phase 0 the last quad of each lane's chunk, phases 1 .. STEPS ChunkTag::finish's
// steps, with the block's tail chunk kept in lane L - 1's h (no lane reads it before the root).
template <int V, int L>
struct TailJob {
  static constexpr int STEPS = L == 1 ? 2 : L == 2 ? 2 : L == 4 ? 3 : 4, PHASES = 1 + STEPS;
  uint32_t h[8], d0, d1;

  __device__ __forceinline__ void phase(const Tail& tl, int pw, int ph, int lane, const Mul& k) {
    static_assert(L >= 2, "deferred ends are for blocks of 2 or more chunks");
    const int pos = lane % L;
    if (ph == 0) {
      const int pb = pw / FW, ps = pb % tl.nspans, pnb = (pb / tl.nspans) * FW + pw % FW;
      const uint4* pp = tl.w + ((size_t)pnb * (tl.Kd / 32) + (size_t)ps * CHUNK_TILES + (CHUNK_TILES - 4)) * LANES + lane;
      uint4 q[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) q[i] = ldw(pp + i * LANES);
      const uint4 c0 = tl.cv[(2 * pw) * 32 + lane], c1 = tl.cv[(2 * pw + 1) * 32 + lane];
      const uint2 dd = tl.d[pw * (32 / L) + lane / L];
      d0 = dd.x;
      d1 = dd.y;
      uint32_t cv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
      uint32_t m[16] = {q[0].x, q[0].y, q[0].z, q[0].w, q[1].x, q[1].y, q[1].z, q[1].w,
                        q[2].x, q[2].y, q[2].z, q[2].w, q[3].x, q[3].y, q[3].z, q[3].w};
      b3<V>(cv, m, (uint32_t)pos, 64, KEYED | END, h, k);
      return;
    }
    const int st = ph - 1;
    const bool last = st == STEPS - 1, tl0 = st == 0 && pos == L - 1;
    const int dist = last ? L - 1 : 1 << st;
    uint32_t key[8], m[16], out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      key[i] = kKey[i];
      const uint32_t o = __shfl_down_sync(FULL, h[i], dist);
      m[i] = tl0 ? (i == 0 ? d0 : i == 1 ? d1 : 0u) : h[i];
      m[8 + i] = tl0 ? 0u : o;
    }
    b3<V>(key, m, tl0 ? (uint32_t)L : 0u, tl0 ? 8u : 64u, KEYED | (tl0 ? START | END : PARENT) | (last ? ROOT : 0), out,
          k);
    if (st == 0 || pos != L - 1) {
#pragma unroll
      for (int i = 0; i < 8; ++i) h[i] = out[i];
    }
    if (last && pos == 0) {
      const int pb = pw / FW, ps = pb % tl.nspans, pnb = (pb / tl.nspans) * FW + pw % FW;
      const uint64_t jl = ((uint64_t)ps * tl.rts + pnb) * (32 / L) + lane / L;
      uint4* o = reinterpret_cast<uint4*>(tl.tags + jl * 8);
      o[0] = make_uint4(h[0], h[1], h[2], h[3]);
      o[1] = make_uint4(h[4], h[5], h[6], h[7]);
    }
  }
};

// fused2's chunk-block map in fused_kernel's shape: FW warps a CTA on one span s of consecutive row blocks (spans fastest
// in the grid), each lane's chunk loaded into registers PD quads ahead (no ring, no cp.async), x's M rows of the span in
// shared memory. Tags and modes as fused2_kernel's; bid is the CTA's index in the GEMM's grid. PK: f3_finish (MODE_TAGS,
// L >= 2) in the shared memory of x's rows, which the CTA no longer reads. DF (MODE_TAGS, one quad ahead): warp gw of
// the grid finishes tl's deferred warps gw, gw + the grid's warps, ..., one phase an iteration from the second quad on
// (the rest after the loop), and with DOWN defers its own ends to cvo and dout instead of finishing them.
template <int V, int L, int NB8, int MODE, int PD, bool PK = false, bool DF = false, bool DOWN = false, int R = 7,
          bool NE = false>
__device__ __forceinline__ void fused3_body(int bid, const uint4* __restrict__ wstore, const uint8_t* __restrict__ x,
                                            float* __restrict__ y, uint32_t* __restrict__ tags, int M, int N, int Kd,
                                            uint64_t jbase, unsigned int* __restrict__ xr, const Tail tl = Tail{},
                                            uint4* __restrict__ cvo = nullptr, uint2* __restrict__ dout = nullptr) {
  static_assert(!DF || (MODE == MODE_TAGS && PD == 1 && !PK), "deferred ends: tags, one quad ahead");
  static_assert(DF || !DOWN, "own ends deferred only in a deferring launch");
  static_assert(!NE || (MODE == MODE_SINK && !DF && !PK), "no chunk ends: a sink diagnostic");
  constexpr bool GEMM = MODE != MODE_TAGONLY, TAG = MODE != MODE_GEMM;
  constexpr int NQ = CHUNK_TILES / 4, DEPTH = PD + 1;
  extern __shared__ __align__(16) uint8_t xs[];
  const int nspans = Kd / CCOLS, rts = N / 16;
  const int s = bid % nspans;
  const int nb = (bid / nspans) * FW + threadIdx.x / 32;
  const int lane = threadIdx.x & 31, g8 = lane >> 2, tq = lane & 3;
  const uint4* wp = wstore + ((size_t)nb * (Kd / 32) + (size_t)s * CHUNK_TILES) * LANES + lane;
  uint4 buf[DEPTH][4];
  auto load = [&](uint4 (&q)[4], int qd) {
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = ldw(wp + (4 * qd + i) * LANES);
  };
#pragma unroll
  for (int p = 0; p < PD; ++p) load(buf[p], p);
  if constexpr (GEMM) {
    for (int e = threadIdx.x; e < M * CCOLS / 16; e += FW * 32) {
      const int b = e / (CCOLS / 16), c = e % (CCOLS / 16);
      *reinterpret_cast<uint4*>(xs + b * CXS + 16 * c) =
          __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)s * CCOLS) + c);
    }
    __syncthreads();
  }
  const Mul k = muls();
  const uint64_t u = (uint64_t)s * rts + nb;
  const int pos = lane % L;
  const uint64_t jl = u * (32 / L) + lane / L;
  float acc[NB8][4];
#pragma unroll
  for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  ChunkTag<V, L, R> ct;
  if constexpr (TAG) ct.begin();
  auto step = [&](const uint4 (&q)[4], int qd) {
    if constexpr (GEMM) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t a[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
        const int col = 32 * (4 * qd + i) + 4 * tq;
#pragma unroll
        for (int j = 0; j < NB8; ++j) {
          const bool live = 8 * j + g8 < M;
          const uint8_t* row = xs + (live ? 8 * j + g8 : 0) * CXS;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + col),
                         b1 = *reinterpret_cast<const uint32_t*>(row + col + 16);
          mma_e4m3(acc[j], a, live ? b0 : 0u, live ? b1 : 0u);
        }
      }
    }
    if constexpr (TAG) {
      if (!(DOWN || NE) || qd < NQ - 1) ct.absorb(q, qd, pos, jbase + jl, k);
    }
  };
  [[maybe_unused]] TailJob<V, L> tj;
  [[maybe_unused]] const int gw = bid * FW + threadIdx.x / 32, gws = gridDim.x * FW;
  constexpr int INLOOP = (NQ - 1) / TailJob<V, L>::PHASES;
#pragma unroll 1
  for (int qb = 0; qb < NQ; qb += DEPTH) {
#pragma unroll
    for (int v = 0; v < DEPTH; ++v) {
      const int qd = qb + v;
      if (NQ % DEPTH == 0 || qd < NQ) {
        if (qd + PD < NQ) load(buf[(v + PD) % DEPTH], qd + PD);
        step(buf[v], qd);
        if constexpr (DF) {
          const int p = qd - 1, ji = p / TailJob<V, L>::PHASES, pw = gw + ji * gws;
          if (p >= 0 && ji < INLOOP && pw < tl.warps) tj.phase(tl, pw, p % TailJob<V, L>::PHASES, lane, k);
        }
      }
    }
  }
  if constexpr (DF) {
    for (int pw = gw + INLOOP * gws; pw < tl.warps; pw += gws) {
#pragma unroll 1
      for (int ph = 0; ph < TailJob<V, L>::PHASES; ++ph) tj.phase(tl, pw, ph, lane, k);
    }
  }
  if constexpr (GEMM) {
#pragma unroll
    for (int j = 0; j < NB8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int n = nb * 16 + g8 + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
        if (b < M) atomicAdd(y + (size_t)b * N + n, acc[j][e]);
      }
    }
  }
  if constexpr (DOWN) {
    cvo[(2 * gw) * 32 + lane] = make_uint4(ct.cv[0], ct.cv[1], ct.cv[2], ct.cv[3]);
    cvo[(2 * gw + 1) * 32 + lane] = make_uint4(ct.cv[4], ct.cv[5], ct.cv[6], ct.cv[7]);
    if (pos == L - 1) dout[gw * (32 / L) + lane / L] = make_uint2(ct.d0, ct.d1);
  } else if constexpr (NE) {
    uint32_t a = ct.cv[0] ^ ct.cv[1] ^ ct.cv[2] ^ ct.cv[3] ^ ct.cv[4] ^ ct.cv[5] ^ ct.cv[6] ^ ct.cv[7] ^ ct.d0 ^ ct.d1;
#pragma unroll
    for (int dd = 16; dd; dd >>= 1) a ^= __shfl_xor_sync(FULL, a, dd);
    if (lane == 0) atomicXor(xr, a);
  } else if constexpr (PK) {
    static_assert(MODE == MODE_TAGS, "the packed finish writes tags");
    __syncthreads();
    f3_finish<V, L>(reinterpret_cast<uint32_t*>(xs), ct, pos, bid, tags,
                    ((uint64_t)s * rts + (uint64_t)(bid / nspans) * FW) * (32 / L), k);
  } else if constexpr (TAG) {
    uint32_t t[8];
    ct.finish(t, pos, k);
    if constexpr (MODE == MODE_SINK) {
      uint32_t a = pos ? 0u : t[0] ^ t[1] ^ t[2] ^ t[3] ^ t[4] ^ t[5] ^ t[6] ^ t[7];
#pragma unroll
      for (int dd = 16; dd; dd >>= 1) a ^= __shfl_xor_sync(FULL, a, dd);
      if (lane == 0) atomicXor(xr, a);
    } else if (pos == 0) {
      uint4* o = reinterpret_cast<uint4*>(tags + jl * 8);
      o[0] = make_uint4(t[0], t[1], t[2], t[3]);
      o[1] = make_uint4(t[4], t[5], t[6], t[7]);
    }
  }
}

template <int V, int L, int NB8, int MODE, int PD, bool PK = false, int R = 7, bool NE = false>
__global__ void __launch_bounds__(FW * 32) fused3_kernel(const uint4* __restrict__ wstore, const uint8_t* __restrict__ x,
                                                         float* __restrict__ y, uint32_t* __restrict__ tags, int M,
                                                         int N, int Kd, uint64_t jbase, unsigned int* __restrict__ xr) {
  fused3_body<V, L, NB8, MODE, PD, PK, false, false, R, NE>(blockIdx.x, wstore, x, y, tags, M, N, Kd, jbase, xr);
}

template <int V, int L, int NB8, bool DOWN>
__global__ void __launch_bounds__(FW * 32, 5) fused3d_kernel(const uint4* __restrict__ wstore, const uint8_t* __restrict__ x,
                                                          float* __restrict__ y, uint32_t* __restrict__ tags, int M,
                                                          int N, int Kd, uint64_t jbase, const Tail tl, uint4* cvo,
                                                          uint2* dout) {
  fused3_body<V, L, NB8, MODE_TAGS, 1, false, true, DOWN>(blockIdx.x, wstore, x, y, tags, M, N, Kd, jbase, nullptr, tl,
                                                           cvo, dout);
}

// ---------------------------------------------------------------------------------------------------------------------
// The commitment: #1805's leaves (16 tags, u64be leaf || 512 bytes after the leaf prefix's midstate: five compressions)
// and its RFC 6962 tree (benchmarks/bandwidth/tagger.cu). leaf0 is the first leaf's index in the store.

__device__ __forceinline__ void leaf_hash(const uint32_t* __restrict__ tags, uint64_t count, uint64_t leaf0,
                                          const Mid& leafmid, uint64_t leaf, uint64_t* __restrict__ leaves) {
  const uint64_t first = leaf * 16;
  if (first >= count) return;
  const int n = (int)(count - first < 16 ? count - first : 16);
  const int data = 1 + 4 * n, nblk = (8 * data + 17 + 127) / 128;
  const uint32_t* T = tags + first * 8;
  uint64_t s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = leafmid.h[i];
#pragma unroll 1
  for (int blk = 0; blk < nblk; ++blk) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i;
      uint64_t v = 0;
      if (q == 0)
        v = leaf0 + leaf;
      else if (q < data)
        v = ((uint64_t)bswap32(T[2 * (q - 1)]) << 32) | bswap32(T[2 * (q - 1) + 1]);
      else if (q == data)
        v = 0x80ULL << 56;
      if (q == 16 * nblk - 1) v = (uint64_t)(128 + 8 + 32 * n) * 8;
      w[i] = v;
    }
    sha512_compress(s, w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) leaves[leaf * 8 + i] = s[i];
}

__global__ void b3_leaves(const uint32_t* __restrict__ tags, uint64_t count, uint64_t leaf0, Mid leafmid,
                          uint64_t* __restrict__ leaves) {
  leaf_hash(tags, count, leaf0, leafmid, blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, leaves);
}

__device__ __forceinline__ void node_hash(const uint64_t* L, const uint64_t* R, uint64_t* out) {
  uint64_t w[16], s[8], prev = 1;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const uint64_t cur = k < 8 ? L[k] : R[k - 8];
    w[k] = (prev << 56) | (cur >> 8);
    prev = cur;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = IV_DEV[i];
  sha512_compress(s, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = (prev << 56) | (0x80ULL << 48);
  w[15] = 129 * 8;
  sha512_compress(s, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = s[i];
}

__device__ __forceinline__ void level_step(const uint64_t* in, uint64_t n, uint64_t* out, uint64_t i) {
  if (2 * i + 1 < n)
    node_hash(in + 16 * i, in + 16 * i + 8, out + 8 * i);
  else
#pragma unroll
    for (int k = 0; k < 8; ++k) out[8 * i + k] = in[16 * i + k];
}

constexpr uint64_t WIDE = 1 << 15;

__global__ void __launch_bounds__(128, 4) node_level(const uint64_t* __restrict__ in, uint64_t n,
                                                     uint64_t* __restrict__ out) {
  const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i < (n + 1) / 2) level_step(in, n, out, i);
}

constexpr int SPANL = 9, SPAN_THREADS = 1 << (SPANL - 1);

struct SpanLv {
  uint64_t off[SPANL + 1], n[SPANL + 1];
  int levels;
};

__global__ void __launch_bounds__(SPAN_THREADS) node_span(uint64_t* nodes, SpanLv s) {
  for (int t = 0; t < s.levels; ++t) {
    const uint64_t i = blockIdx.x * (uint64_t)(SPAN_THREADS >> t) + threadIdx.x;
    if (threadIdx.x < (SPAN_THREADS >> t) && i < s.n[t + 1])
      level_step(nodes + s.off[t] * 8, s.n[t], nodes + s.off[t + 1] * 8, i);
    __syncthreads();
  }
}

// Levels 1 .. s.levels of the complete subtrees of 2^s.levels leaves from leaf `first` on, one a block of
// 2^(s.levels - 1) threads (s from level 0).
__global__ void __launch_bounds__(SPAN_THREADS) node_sub(uint64_t* nodes, SpanLv s, uint64_t first) {
  const int threads = blockDim.x;
  for (int t = 0; t < s.levels; ++t) {
    const uint64_t i = (first >> (t + 1)) + blockIdx.x * (uint64_t)(threads >> t) + threadIdx.x;
    if (threadIdx.x < (threads >> t)) level_step(nodes + s.off[t] * 8, s.n[t], nodes + s.off[t + 1] * 8, i);
    __syncthreads();
  }
}

// b3_leaves, then levels 1 .. s.levels (2^s.levels = the block's threads) over the block's own leaves: one launch
// where b3_leaves and node_level / node_span took a launch a level, each level after the leaves latency-bound.
constexpr int SUB_THREADS = 128, SUB_LEVELS = 7;

__global__ void __launch_bounds__(SUB_THREADS) leaves_sub(const uint32_t* __restrict__ tags, uint64_t count,
                                                          uint64_t leaf0, Mid leafmid, uint64_t* nodes, SpanLv s) {
  leaf_hash(tags, count, leaf0, leafmid, blockIdx.x * (uint64_t)SUB_THREADS + threadIdx.x, nodes);
  __syncthreads();
  for (int t = 0; t < s.levels; ++t) {
    const int live = SUB_THREADS >> (t + 1);
    const uint64_t i = blockIdx.x * (uint64_t)live + threadIdx.x;
    if (threadIdx.x < live && i < s.n[t + 1]) level_step(nodes + s.off[t] * 8, s.n[t], nodes + s.off[t + 1] * 8, i);
    __syncthreads();
  }
}

struct Levels {
  std::vector<uint64_t> off, n;
  uint64_t total = 0;
};

static Levels levels_for(uint64_t leaves) {
  Levels L;
  uint64_t off = 0, n = leaves;
  for (;;) {
    L.off.push_back(off);
    L.n.push_back(n);
    off += n;
    if (n == 1) break;
    n = (n + 1) / 2;
  }
  L.total = off;
  return L;
}

static void launch_tree(uint64_t* nodes, const Levels& lv, cudaStream_t st, size_t k = 0) {
  const size_t top = lv.n.size() - 1;
  for (; k < top && lv.n[k] > WIDE; ++k) {
    const uint64_t nout = lv.n[k + 1];
    node_level<<<(unsigned)((nout + 127) / 128), 128, 0, st>>>(nodes + lv.off[k] * 8, lv.n[k], nodes + lv.off[k + 1] * 8);
  }
  while (k < top) {
    SpanLv s;
    s.levels = (int)(top - k < (size_t)SPANL ? top - k : SPANL);
    for (int t = 0; t <= s.levels; ++t) {
      s.off[t] = lv.off[k + t];
      s.n[t] = lv.n[k + t];
    }
    node_span<<<(unsigned)((lv.n[k] + (2 * SPAN_THREADS) - 1) / (2 * SPAN_THREADS)), SPAN_THREADS, 0, st>>>(nodes, s);
    k += s.levels;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Compression chains: each thread a serial chain of `steps` compressions, its result XORed into *xr.

template <int V, int ROUNDS>
__global__ void __launch_bounds__(256) chain_b3(uint64_t steps, unsigned int* xr) {
  const Mul k = muls();
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cv[8], m[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = kKey[i] ^ (id * 0x9E3779B9u + i);
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = id * 0x85EBCA6Bu + 0x27D4EB2Fu * i;
#pragma unroll 1
  for (uint64_t t = 0; t < steps; ++t) {
    b3<V, ROUNDS>(cv, m, (uint32_t)t, 64, KEYED, cv, k);
    m[0] ^= cv[7];
  }
  atomicXor(xr, cv[0] ^ cv[1] ^ cv[2] ^ cv[3] ^ cv[4] ^ cv[5] ^ cv[6] ^ cv[7]);
}

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) { return n ? (x << n) | (x >> (64 - n)) : x; }

// Keccak-p[1600, ROUNDS]: rounds 24 - ROUNDS .. 23 (TurboSHAKE128's at 12), lanes indexed x + 5y.
template <int ROUNDS>
__device__ __forceinline__ void keccak_p(uint64_t a[25]) {
  constexpr uint64_t KECCAK_RC[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL, 0x000000000000808BULL,
      0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL, 0x0000000000000088ULL,
      0x0000000080008009ULL, 0x000000008000000AULL, 0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
      0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
      0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
  constexpr int KECCAK_ROT[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
#pragma unroll
  for (int r = 24 - ROUNDS; r < 24; ++r) {
    uint64_t c[5], d[5], b[25];
#pragma unroll
    for (int x = 0; x < 5; ++x) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
#pragma unroll
    for (int x = 0; x < 5; ++x) d[x] = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
#pragma unroll
    for (int x = 0; x < 5; ++x)
#pragma unroll
      for (int y = 0; y < 5; ++y) b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(a[x + 5 * y] ^ d[x], KECCAK_ROT[x + 5 * y]);
#pragma unroll
    for (int y = 0; y < 25; y += 5)
#pragma unroll
      for (int x = 0; x < 5; ++x) a[y + x] = b[y + x] ^ (~b[y + (x + 1) % 5] & b[y + (x + 2) % 5]);
    a[0] ^= KECCAK_RC[r];
  }
}

// A sponge's absorb at rate 168: 21 lanes XORed in, then the permutation.
__global__ void __launch_bounds__(256) chain_keccak(uint64_t steps, unsigned int* xr) {
  const uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t a[25], m[21];
#pragma unroll
  for (int i = 0; i < 25; ++i) a[i] = id * 0x9E3779B97F4A7C15ULL + i;
#pragma unroll
  for (int i = 0; i < 21; ++i) m[i] = id * 0xC2B2AE3D27D4EB4FULL ^ (0x165667B19E3779F9ULL * i);
#pragma unroll 1
  for (uint64_t t = 0; t < steps; ++t) {
#pragma unroll
    for (int i = 0; i < 21; ++i) a[i] ^= m[i];
    keccak_p<12>(a);
    m[0] ^= t;
  }
  uint64_t acc = 0;
#pragma unroll
  for (int i = 0; i < 25; ++i) acc ^= a[i];
  atomicXor(xr, (uint32_t)acc ^ (uint32_t)(acc >> 32));
}

__global__ void __launch_bounds__(256) chain_sha512(uint64_t steps, unsigned int* xr) {
  const uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t s[8], w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = IV_DEV[i] ^ id;
#pragma unroll 1
  for (uint64_t t = 0; t < steps; ++t) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = s[i & 7] + i;
    sha512_compress(s, w);
  }
  atomicXor(xr, (uint32_t)(s[0] ^ s[3]) ^ (uint32_t)(s[5] >> 32));
}

// ---------------------------------------------------------------------------------------------------------------------
// Self-tests, a streaming read, spin and empty.

template <int V>
__global__ void b3_one(const uint32_t* in, uint32_t* out) {  // in: cv[8], m[16], counter, len, flags
  const Mul k = muls();
  uint32_t cv[8], m[16], o[8];
  for (int i = 0; i < 8; ++i) cv[i] = in[i];
  for (int i = 0; i < 16; ++i) m[i] = in[8 + i];
  b3<V>(cv, m, in[24], in[25], in[26], o, k);
  for (int i = 0; i < 8; ++i) out[i] = o[i];
}

__global__ void keccak_one(uint64_t* a) {
  uint64_t s[25];
  for (int i = 0; i < 25; ++i) s[i] = a[i];
  keccak_p<12>(s);
  for (int i = 0; i < 25; ++i) a[i] = s[i];
}

__global__ void __launch_bounds__(256) read_kernel(const uint4* __restrict__ p, uint64_t n16, unsigned int* xr) {
  uint32_t acc = 0;
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n16; i += (uint64_t)gridDim.x * blockDim.x) {
    const uint4 v = ldw(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) acc ^= __shfl_xor_sync(FULL, acc, d);
  if ((threadIdx.x & 31) == 0) atomicXor(xr, acc);
}

// A host launch through ctypes costs more than a small GEMM: a timed arm is queued behind `spin` so the device runs it
// back to back, and `empty` (the GEMM's grid, no work) prices what back to back still costs a launch.
__global__ void spin_kernel(long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

__global__ void __launch_bounds__(FW * 32) empty_kernel() {}

static Mid g_leafmid;

}  // namespace tf

using namespace tf;

// ---------------------------------------------------------------------------------------------------------------------
// The C ABI. Every function returns 0 or an error code: 1 bad arguments, 2 a CUDA call failed, 3 not built.

#define FUSED_V(V, SPAN, NB8, MODE, PD, P) \
  if (v == V && span == SPAN && nb8 == NB8 && mode == MODE && pd == PD && partials == P) kern = fused_kernel<V, SPAN, NB8, MODE, PD, P>;
#define FUSED_NB(V, SPAN, MODE, PD, P) FUSED_V(V, SPAN, 1, MODE, PD, P) FUSED_V(V, SPAN, 2, MODE, PD, P) FUSED_V(V, SPAN, 4, MODE, PD, P)
#define FUSED_SP(V, MODE, PD, P) FUSED_NB(V, 32, MODE, PD, P) FUSED_NB(V, 64, MODE, PD, P)

using Kern2 = void (*)(const uint4*, const uint8_t*, float*, uint32_t*, int, int, int, uint64_t, unsigned int*);

template <int L, int MODE, int W, int ST, bool P, int R = 7, bool PK = false>
Kern2 pick2(int nb8) {
  if constexpr (MODE == MODE_TAGONLY)
    return fused2_kernel<1, L, 1, MODE, W, ST, P, R, PK>;
  else
    return nb8 == 1 ? fused2_kernel<1, L, 1, MODE, W, ST, P, R, PK>
           : nb8 == 2 ? fused2_kernel<1, L, 2, MODE, W, ST, P, R, PK>
                      : fused2_kernel<1, L, 4, MODE, W, ST, P, R, PK>;
}

extern "C" {

int tf_version() { return 1; }

// The window's key K and the leaves' midstate from the 32-byte seed; key_out (host, 32 bytes) gets K.
int tf_init(const uint8_t* seed, uint8_t* key_out) {
  const std::vector<uint8_t> p = prefix_block(TAG_DOMAIN, seed);
  uint64_t d[8];
  sha512_host(p.data(), p.size(), d);
  uint8_t kb[32];
  for (int i = 0; i < 32; ++i) kb[i] = (uint8_t)(d[i / 8] >> (56 - 8 * (i % 8)));
  uint32_t kw[8];
  for (int i = 0; i < 8; ++i) kw[i] = kb[4 * i] | kb[4 * i + 1] << 8 | kb[4 * i + 2] << 16 | (uint32_t)kb[4 * i + 3] << 24;
  const uint32_t mul[5] = {1u, 1u << 20, 1u << 25, 1u << 16, 1u << 24};
  if (cudaMemcpyToSymbol(kKey, kw, sizeof kw) || cudaMemcpyToSymbol(kMul, mul, sizeof mul)) return 2;
  g_leafmid = prefix_midstate(LEAF_DOMAIN, seed);
  if (key_out) memcpy(key_out, kb, 32);
  return 0;
}

// The fused GEMM. v: the BLAKE3 variant; span: 32 or 64 tiles a block; mode: 0 GEMM, 1 + tags, 2 + sink, 3 tags only;
// pd: quads of loads in flight (1 or 2); partials: 0 atomic, 1 per-span sums. M <= 32, N a multiple of 64, K of 32 span.
int tf_fused(int v, int span, int mode, int pd, int partials, const void* w, const uint8_t* x, float* y, uint32_t* tags,
             int M, int N, int Kd, uint64_t jbase, unsigned int* xr, cudaStream_t st) {
  if (M < 1 || M > 32 || N % (16 * FW) || (span != 32 && span != 64) || Kd % (32 * span) || mode < 0 || mode > 3)
    return 1;
  const int nb8 = mode == MODE_TAGONLY ? 1 : M <= 8 ? 1 : M <= 16 ? 2 : 4;
  void (*kern)(const uint4*, const uint8_t*, float*, uint32_t*, int, int, int, uint64_t, unsigned int*) = nullptr;
  // GEMM only (V is unused): both depths, and per-span sums at depth 1 for the gate
  FUSED_SP(0, 0, 1, false) FUSED_SP(0, 0, 2, false) FUSED_SP(0, 0, 1, true)
  // + tags: V0, V1, V3 at depth 1; V1 at depth 2; V1 per-span sums for the gate
  FUSED_SP(0, 1, 1, false) FUSED_SP(1, 1, 1, false) FUSED_SP(3, 1, 1, false) FUSED_SP(1, 1, 2, false)
  FUSED_SP(1, 1, 1, true)
  // + sink: V1 at depth 1
  FUSED_SP(1, 2, 1, false)
  // tags only: V0, V1, V3 at depth 1, V1 at depth 2
  FUSED_V(0, 32, 1, 3, 1, false) FUSED_V(0, 64, 1, 3, 1, false) FUSED_V(1, 32, 1, 3, 1, false)
  FUSED_V(1, 64, 1, 3, 1, false) FUSED_V(3, 32, 1, 3, 1, false) FUSED_V(3, 64, 1, 3, 1, false)
  FUSED_V(1, 32, 1, 3, 2, false) FUSED_V(1, 64, 1, 3, 2, false)
  if (!kern) return 3;
  const int smem = mode == MODE_TAGONLY ? 0 : nb8 * 8 * (32 * span + 16);
  if (smem > 48 * 1024 && cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) return 2;
  const int grid = (N / 16 / FW) * (Kd / (32 * span));
  kern<<<grid, FW * 32, smem, st>>>((const uint4*)w, x, y, tags, M, N, Kd, jbase, xr);
  return cudaGetLastError() ? 2 : 0;
}

#define F2(L, MODE, W, ST, P, R) \
  if (lanes == L && mode == MODE && wc == W && stages == ST && partials == P && rounds == R && !packed) \
    kern = pick2<L, MODE, W, ST, P, R>(nb8);
#define F2P(L, MODE, W, ST) \
  if (lanes == L && mode == MODE && wc == W && stages == ST && !partials && rounds == 7 && packed) \
    kern = pick2<L, MODE, W, ST, false, 7, true>(nb8);

// The chunk-block GEMM (fused2_kernel, BLAKE3 variant 1). lanes: L, a block's chunks; mode as tf_fused's; wc: warps a
// CTA; stages: the ring's quads; rounds 7, or 4 (not BLAKE3: uncertified); packed: the CTA's finish (cta_finish), tags
// modes only. M <= 32, N a multiple of 16 wc, K of 2,048. 3: form not built; 4: its x slice and ring exceed a CTA's
// shared memory.
int tf_fused2(int lanes, int mode, int wc, int stages, int rounds, int packed, int partials, const void* w,
              const uint8_t* x, float* y, uint32_t* tags, int M, int N, int Kd, uint64_t jbase, unsigned int* xr,
              cudaStream_t st) {
  if (M < 1 || M > 32 || wc < 1 || N % (16 * wc) || Kd % CCOLS || mode < 0 || mode > 3) return 1;
  const int nb8 = mode == MODE_TAGONLY ? 1 : M <= 8 ? 1 : M <= 16 ? 2 : 4;
  Kern2 kern = nullptr;
  if (mode == MODE_GEMM) lanes = 1;
  // GEMM only, at each CTA shape; per-span sums at 4 warps and 4 stages
  F2(1, 0, 1, 3, false, 7) F2(1, 0, 2, 3, false, 7) F2(1, 0, 4, 2, false, 7) F2(1, 0, 4, 3, false, 7)
  F2(1, 0, 8, 2, false, 7) F2(1, 0, 4, 4, false, 7) F2(1, 0, 4, 4, true, 7)
  // + tags: L = 4 at each CTA shape and with per-span sums, L = 8 and L = 1 at fewer
  F2(4, 1, 1, 3, false, 7) F2(4, 1, 1, 4, false, 7) F2(4, 1, 2, 2, false, 7) F2(4, 1, 2, 3, false, 7)
  F2(4, 1, 4, 2, false, 7) F2(4, 1, 4, 3, false, 7) F2(4, 1, 8, 2, false, 7) F2(4, 1, 4, 4, true, 7)
  F2(8, 1, 1, 3, false, 7) F2(8, 1, 2, 3, false, 7) F2(8, 1, 4, 2, false, 7) F2(8, 1, 4, 3, false, 7)
  F2(1, 1, 1, 3, false, 7) F2(1, 1, 2, 3, false, 7) F2(1, 1, 4, 3, false, 7)
  // + sink; tags only; the 4-round probes
  F2(4, 2, 4, 3, false, 7) F2(1, 3, 4, 3, false, 7) F2(4, 3, 4, 3, false, 7)
  F2(4, 1, 4, 3, false, 4) F2(4, 3, 4, 3, false, 4)
  // packed finish: L = 4 and 8 at 2 and 4 warps, L = 1 at 4; tags only at L = 4
  F2P(4, 1, 2, 3) F2P(4, 1, 4, 2) F2P(4, 1, 4, 3) F2P(8, 1, 2, 3) F2P(8, 1, 4, 2) F2P(8, 1, 4, 3) F2P(1, 1, 4, 3)
  F2P(4, 3, 4, 3)
  if (!kern) return 3;
  const int xbytes = nb8 == 1 ? M * XSlice<1>::STRIDE : XSlice<2>::BUFS * M * XSlice<2>::STRIDE;
  const int smem = (mode == MODE_TAGONLY ? 0 : xbytes) + wc * stages * QUAD_BYTES;
  if (smem > 99 * 1024) return 4;
  static std::unordered_set<const void*> configured;
  if (!configured.count((const void*)kern)) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) ||
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100))
      return 2;
    configured.insert((const void*)kern);
  }
  const int grid = (N / 16 / wc) * (Kd / CCOLS);
  kern<<<grid, wc * 32, smem, st>>>((const uint4*)w, x, y, tags, M, N, Kd, jbase, xr);
  return cudaGetLastError() ? 2 : 0;
}
#undef F2

#define F3(K, L, MODE, PD)                                                                                             \
  if (lanes == L && mode == MODE && pd == PD)                                                                          \
    kern = nb8 == 1 ? K<1, L, 1, MODE, PD> : nb8 == 2 ? K<1, L, 2, MODE, PD> : K<1, L, 4, MODE, PD>;

static int configure3(const void* kern, int M) {
  if (M * CXS > 99 * 1024) return 4;
  static std::unordered_set<const void*> configured;
  if (!configured.count(kern)) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) ||
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100))
      return 2;
    configured.insert(kern);
  }
  return 0;
}

// The chunk-block GEMM in registers (fused3_kernel): lanes L; mode GEMM or tags; pd quads ahead (1 or 2); one quad
// ahead, pk 1 the packed CTA finish (tags, L 4 or 8), 2 four-round BLAKE3 (tags, L 4: uncertified), 3 no chunk ends
// (sink, L 4: the chunks' last quads and finishes left out, a diagnostic of their cost). M <= 32, N a multiple of
// 16 FW, K of 2,048. 3: form not built; 4: x's M rows exceed a CTA's shared memory.
int tf_fused3(int lanes, int mode, int pd, int pk, const void* w, const uint8_t* x, float* y, uint32_t* tags, int M,
              int N, int Kd, uint64_t jbase, unsigned int* xr, cudaStream_t st) {
  if (M < 1 || M > 32 || N % (16 * FW) || Kd % CCOLS || mode < 0 || mode > 3 || pk < 0 || pk > 3 ||
      ((pk == 1 || pk == 2) && mode != MODE_TAGS) || (pk == 3 && mode != MODE_SINK))
    return 1;
  const int nb8 = M <= 8 ? 1 : M <= 16 ? 2 : 4;
  if (mode == MODE_GEMM) lanes = 1;
  Kern2 kern = nullptr;
  if (!pk) {
    F3(fused3_kernel, 1, 0, 1) F3(fused3_kernel, 1, 0, 2) F3(fused3_kernel, 1, 1, 1) F3(fused3_kernel, 1, 1, 2)
    F3(fused3_kernel, 4, 1, 1) F3(fused3_kernel, 4, 1, 2) F3(fused3_kernel, 8, 1, 1) F3(fused3_kernel, 8, 1, 2)
  } else if (pd == 1) {
#define F3K(L, P, MD, PK, R, NE)                                                                                        \
  if (lanes == L && pk == P)                                                                                           \
    kern = nb8 == 1   ? fused3_kernel<1, L, 1, MD, 1, PK, R, NE>                                                       \
           : nb8 == 2 ? fused3_kernel<1, L, 2, MD, 1, PK, R, NE>                                                       \
                      : fused3_kernel<1, L, 4, MD, 1, PK, R, NE>;
    F3K(4, 1, MODE_TAGS, true, 7, false) F3K(8, 1, MODE_TAGS, true, 7, false) F3K(4, 2, MODE_TAGS, false, 4, false)
    F3K(4, 3, MODE_SINK, false, 7, true)
#undef F3K
  }
  if (!kern) return 3;
  if (const int rc = configure3((const void*)kern, M)) return rc;
  const int smem = pk == 1 ? std::max(M * CXS, F3_FINISH_BYTES) : M * CXS;
  kern<<<(N / 16 / FW) * (Kd / CCOLS), FW * 32, smem, st>>>((const uint4*)w, x, y, tags, M, N, Kd, jbase, xr);
  return cudaGetLastError() ? 2 : 0;
}

#undef F3

// tf_fused3's t{lanes}kr1 (lanes 4 or 8) with deferred chunk ends: the previous launch's (its weights pw, chaining
// values pcv, tail bytes pd and tags ptags, shape pN x pKd; pw null for none) finished inside this one's loop, and this
// launch's own deferred to cvo and dout (2 FW 32 uint4 and FW 32 / lanes uint2 a CTA) if own_defer, else finished.
int tf_fused3d(int lanes, int own_defer, const void* w, const uint8_t* x, float* y, uint32_t* tags, int M, int N,
               int Kd, uint64_t jbase, void* cvo, void* dout, const void* pw, const void* pcv, const void* pd,
               uint32_t* ptags, int pN, int pKd, cudaStream_t st) {
  if (M < 1 || M > 32 || N % (16 * FW) || Kd % CCOLS || (lanes != 4 && lanes != 8) || (own_defer && !(cvo && dout)) ||
      (pw && (!pcv || !pd || !ptags || pN % (16 * FW) || pKd % CCOLS)))
    return 1;
  const int nb8 = M <= 8 ? 1 : M <= 16 ? 2 : 4;
  Tail tl;
  if (pw) {
    tl.w = (const uint4*)pw;
    tl.cv = (const uint4*)pcv;
    tl.d = (const uint2*)pd;
    tl.tags = ptags;
    tl.nspans = pKd / CCOLS;
    tl.rts = pN / 16;
    tl.Kd = pKd;
    tl.warps = (pN / 16 / FW) * tl.nspans * FW;
  }
  using KernD = void (*)(const uint4*, const uint8_t*, float*, uint32_t*, int, int, int, uint64_t, const Tail, uint4*,
                         uint2*);
  KernD kern = nullptr;
#define F3D(L, D)                                                                                                      \
  if (lanes == L && !!own_defer == D)                                                                                  \
    kern = nb8 == 1 ? fused3d_kernel<1, L, 1, D> : nb8 == 2 ? fused3d_kernel<1, L, 2, D> : fused3d_kernel<1, L, 4, D>;
  F3D(4, false) F3D(4, true) F3D(8, false) F3D(8, true)
#undef F3D
  if (!kern) return 3;
  if (const int rc = configure3((const void*)kern, M)) return rc;
  kern<<<(N / 16 / FW) * (Kd / CCOLS), FW * 32, M * CXS, st>>>((const uint4*)w, x, y, tags, M, N, Kd, jbase, tl,
                                                                (uint4*)cvo, (uint2*)dout);
  return cudaGetLastError() ? 2 : 0;
}

int tf_plain(const void* w, const uint8_t* x, float* y, int M, int N, int Kd, int partials, cudaStream_t st) {
  if (M < 1 || M > 32 || N % (16 * FW) || Kd % TASK_COLS) return 1;
  const int nb8 = M <= 8 ? 1 : M <= 16 ? 2 : 4, grid = (N / 16 / FW) * (Kd / TASK_COLS);
#define PL(NB, P) plain_kernel<NB, P><<<grid, FW * 32, 0, st>>>((const uint4*)w, x, y, M, N, Kd)
  if (partials) {
    if (nb8 == 1) PL(1, true); else if (nb8 == 2) PL(2, true); else PL(4, true);
  } else {
    if (nb8 == 1) PL(1, false); else if (nb8 == 2) PL(2, false); else PL(4, false);
  }
#undef PL
  return cudaGetLastError() ? 2 : 0;
}

// Leaves over `count` tags (8 words each) into nodes[0 ..), the first leaf's index in the store leaf0.
int tf_leaves(const uint32_t* tags, uint64_t count, uint64_t leaf0, uint64_t* nodes, cudaStream_t st) {
  const uint64_t leaves = (count + 15) / 16;
  b3_leaves<<<(unsigned)((leaves + 127) / 128), 128, 0, st>>>(tags, count, leaf0, g_leafmid, nodes);
  return cudaGetLastError() ? 2 : 0;
}

// The tree's node count over `leaves` leaves (the buffer tf_tree needs, 64 bytes a node), and the tree itself with the
// leaves at nodes[0 ..); the root is the last node.
uint64_t tf_tree_nodes(uint64_t leaves) { return levels_for(leaves).total; }

int tf_tree(uint64_t* nodes, uint64_t leaves, cudaStream_t st) {
  if (!leaves) return 1;
  launch_tree(nodes, levels_for(leaves), st);
  return cudaGetLastError() ? 2 : 0;
}

// tf_leaves then tf_tree in fewer launches: leaves_sub's blocks take the leaves and the levels over each block's 128,
// launch_tree the rest. The same nodes.
int tf_commit(const uint32_t* tags, uint64_t count, uint64_t leaf0, uint64_t* nodes, cudaStream_t st) {
  const uint64_t leaves = (count + 15) / 16;
  if (!leaves) return 1;
  const Levels lv = levels_for(leaves);
  const size_t top = lv.n.size() - 1;
  SpanLv s;
  s.levels = (int)(top < (size_t)SUB_LEVELS ? top : SUB_LEVELS);
  for (int t = 0; t <= s.levels; ++t) {
    s.off[t] = lv.off[t];
    s.n[t] = lv.n[t];
  }
  leaves_sub<<<(unsigned)((leaves + SUB_THREADS - 1) / SUB_THREADS), SUB_THREADS, 0, st>>>(tags, count, leaf0,
                                                                                           g_leafmid, nodes, s);
  launch_tree(nodes, lv, st, (size_t)s.levels);
  return cudaGetLastError() ? 2 : 0;
}

// Part of the tree before all its leaves are in: levels 1 .. levels over leaves [first, first + count), both multiples
// of 2^levels (levels <= 9); then tf_tree_from(nodes, leaves, levels) the rest.
int tf_subtree(uint64_t* nodes, uint64_t leaves, uint64_t first, uint64_t count, int levels, cudaStream_t st) {
  const uint64_t span = 1ULL << levels;
  if (levels < 1 || levels > SPANL || first % span || count % span || first + count > leaves) return 1;
  const Levels lv = levels_for(leaves);
  SpanLv s;
  s.levels = levels;
  for (int t = 0; t <= levels; ++t) {
    s.off[t] = lv.off[t];
    s.n[t] = lv.n[t];
  }
  node_sub<<<(unsigned)(count / span), (unsigned)(span / 2), 0, st>>>(nodes, s, first);
  return cudaGetLastError() ? 2 : 0;
}

int tf_tree_from(uint64_t* nodes, uint64_t leaves, int from, cudaStream_t st) {
  const Levels lv = levels_for(leaves);
  if (!leaves || from < 0 || (size_t)from >= lv.n.size()) return 1;
  launch_tree(nodes, lv, st, (size_t)from);
  return cudaGetLastError() ? 2 : 0;
}

// what: 0 BLAKE3 (variant v, rounds 7 or, uncertified, 4 or 5), 1 Keccak-p[1600, 12] at rate 168, 2 SHA-512.
int tf_chain(int what, int v, int rounds, uint64_t steps, unsigned int* xr, int grid, int block, cudaStream_t st) {
  if (block < 32 || block > 256) return 1;
  if (what == 1) {
    chain_keccak<<<grid, block, 0, st>>>(steps, xr);
  } else if (what == 2) {
    chain_sha512<<<grid, block, 0, st>>>(steps, xr);
  } else if (what == 0) {
#define CH(V, R) if (v == V && rounds == R) { chain_b3<V, R><<<grid, block, 0, st>>>(steps, xr); return cudaGetLastError() ? 2 : 0; }
    CH(0, 7) CH(1, 7) CH(2, 7) CH(3, 7) CH(4, 7) CH(0, 4) CH(1, 4) CH(1, 5)
#undef CH
    return 3;
  } else {
    return 1;
  }
  return cudaGetLastError() ? 2 : 0;
}

// One compression on the device: in (device) = cv[8], m[16], counter, len, flags; out (device) the 8 output words.
int tf_b3_one(int v, const uint32_t* in, uint32_t* out, cudaStream_t st) {
  switch (v) {
    case 0: b3_one<0><<<1, 1, 0, st>>>(in, out); break;
    case 1: b3_one<1><<<1, 1, 0, st>>>(in, out); break;
    case 2: b3_one<2><<<1, 1, 0, st>>>(in, out); break;
    case 3: b3_one<3><<<1, 1, 0, st>>>(in, out); break;
    case 4: b3_one<4><<<1, 1, 0, st>>>(in, out); break;
    default: return 1;
  }
  return cudaGetLastError() ? 2 : 0;
}

int tf_keccak_one(uint64_t* state, cudaStream_t st) {
  keccak_one<<<1, 1, 0, st>>>(state);
  return cudaGetLastError() ? 2 : 0;
}

int tf_read(const void* p, uint64_t bytes, unsigned int* xr, int grid, cudaStream_t st) {
  read_kernel<<<grid, 256, 0, st>>>((const uint4*)p, bytes / 16, xr);
  return cudaGetLastError() ? 2 : 0;
}

int tf_spin(long long cycles, cudaStream_t st) {
  spin_kernel<<<1, 1, 0, st>>>(cycles);
  return cudaGetLastError() ? 2 : 0;
}

int tf_empty(int grid, cudaStream_t st) {
  empty_kernel<<<grid, FW * 32, 0, st>>>();
  return cudaGetLastError() ? 2 : 0;
}

}  // extern "C"
