// The word codec on the RTX PRO 6000 (sm_120): `codec.py` is the reference, `twin.py` this file's arithmetic step for step.
//
// Field p = 2^61 - 1. Lazy forms: `sqr` takes a < 2^61 + 8 and returns a value < 2^61 + 4 congruent to a^2, `fold` takes
// any 64-bit z to < 2^61 + 8, `canon` takes r < 2^61 + 8 to [0, p). A decode round is one `sqr`, the sign rule and the
// subtraction of c_j + t(i) folded into one 64-bit add (`kA`, `kB`, ~t), then `canon(fold(.))`; an encode round is the
// 59 squarings of the root, the square test and the parity choice.
//
// Kernels (extern "C" launchers at the end):
//   decode / encode      a flat store, word i at index base + i: mode 0 reads and XORs the payloads, mode 2 also writes them
//   chain_*              compute only: U independent chains a thread, each step's input the previous step's output, the
//                        index (and so the tweak) advancing by the grid's stride: decode, encode, the bare squaring, the
//                        GF(2^31 - 1) Horner step of the bandwidth audit's tag
//   read, tag            the same flat store read and XORed, or Horner-hashed (two 32-bit steps a word)
//   gemm                 y[b][n] = sum_k x[b][k] W[n][k], FP8 E4M3 mma.m16n8k32 with W the A operand in fragment order
//                        (`codec.fragment_payloads` / `codec.plain_fragments`), split over 512-column chunks; a word
//                        carries WC_PB payload bits (56 or 60); three forms: one task a warp (`gemm`), persistent
//                        with the next half task in registers (`gemm_pipe`), warp-specialized with the next tasks in a
//                        shared-memory ring (`gemm_ws`)
//   gemm_ws's tag        the bandwidth audit's ro-blake3/v1 over the encoded words as they arrive, before the decode: block
//                        j = jbase + 32 q + l is lane l's words of task q, little-endian in order (`codec.lane_block`),
//                        its tag BLAKE3-keyed(K, u64be j ‖ E_j) in the lane's registers, written to tags[8 (32 q + l)]
//   gemm_ws's commit     committer warps hash the leaves (16 tags a leaf) of the previous call's tags; the tree above
//                        the leaves is benchmarks/bandwidth/tagger.cu's (`wc_commit_tree`)
#include <cstdint>
#include <cuda_runtime.h>

#include "../bandwidth/tagger.cu"

#ifndef WC_PB
#define WC_PB 56
#endif

namespace wc {

constexpr uint64_t P = (1ull << 61) - 1;
constexpr int PB = WC_PB;
static_assert(PB == 56 || PB == 60, "codec.PACKINGS");
constexpr int MAXK = 8;
constexpr int LANES = 32;
constexpr int TASK_TILES = 16;
constexpr int LANE_BYTES = 16 * TASK_TILES;
constexpr int LANE_WORDS = (8 * LANE_BYTES + PB - 1) / PB;
constexpr int TASK_WORDS = LANE_WORDS * LANES;
constexpr int TASK_COLS = 32 * TASK_TILES;
constexpr int GEMM_WARPS = 4;
constexpr int XS_STRIDE = TASK_COLS + 16;
constexpr uint32_t M31 = 0x7FFFFFFFu;

__constant__ uint64_t kC[MAXK];   // c_j
__constant__ uint64_t kA[MAXK];   // 2p - c_j + 1: an even y's z = v + kA[j] + ~t = v + 2p - c_j - t (mod 2^64)
__constant__ uint64_t kB[MAXK];   // 6p - c_j + 1: an odd y's  z = kB[j] - v + ~t = 6p - c_j - t - v (mod 2^64)
__constant__ uint64_t kTau;

__device__ __forceinline__ uint64_t lohi(uint32_t lo, uint32_t hi) { return (uint64_t)hi << 32 | lo; }

__device__ __forceinline__ uint64_t fold(uint64_t z) {
  uint32_t zl = (uint32_t)z, zh = (uint32_t)(z >> 32);
  return lohi(zl, zh & 0x1FFFFFFFu) + (zh >> 29);      // < 2^61 + 8
}

#ifndef WC_SQR
#define WC_SQR 0
#endif

// WC_SQR 0: a split at bit 31, three 32 x 32 -> 64 products and one fold. 1: the 128-bit square (a * a, __umul64hi), the
// high part shifted into place, and one fold.
__device__ __forceinline__ uint64_t sqr(uint64_t a) {
#if WC_SQR == 1
  uint64_t lo = a * a, hi = __umul64hi(a, a);
  return fold((lo & P) + ((hi << 3) | (lo >> 61)));    // the sum < 2^62, the result < 2^61 + 2
#else
  uint32_t lo = (uint32_t)a, hi = (uint32_t)(a >> 32);
  uint32_t a0 = lo & M31;
  uint32_t a1 = __funnelshift_r(lo, hi, 31);           // a = a1 2^31 + a0, a1 <= 2^30
  uint64_t s = (uint64_t)(a1 << 1) * a1;               // 2^62 = 2 (mod p): 2 a1^2 <= 2^61
  s += (uint64_t)a0 * a0;                              // < 2^62
  uint64_t m = (uint64_t)a0 * a1;                      // a0 a1 2^32 = (m >> 29) + (m mod 2^29) 2^32 (mod p), m < 2^61
  uint32_t mlo = (uint32_t)m, mhi = (uint32_t)(m >> 32);
  s += lohi(__funnelshift_r(mlo, mhi, 29), mlo & 0x1FFFFFFFu);   // s < 2^63 + 2^32
  uint32_t sl = (uint32_t)s, sh = (uint32_t)(s >> 32);
  return lohi(sl, sh & 0x1FFFFFFFu) + (sh >> 29);      // < 2^61 + 4
#endif
}

__device__ __forceinline__ uint64_t canon(uint64_t r) {
  uint64_t r1 = r + 1;                                 // r >= p iff r + 1 >= 2^61, and then r - p = (r + 1) - 2^61
  return (r1 >> 61) ? (r1 & P) : r;
}

__device__ __forceinline__ uint64_t dec_round(uint64_t y, uint64_t nt, int j) {
  uint64_t v = sqr(y);
  uint64_t z = (y & 1) ? kB[j] - v + nt : v + kA[j] + nt;
  return canon(fold(z));
}

template <int K>
__device__ __forceinline__ uint64_t decode_word(uint64_t y, uint64_t nt) {
#pragma unroll
  for (int j = K - 1; j >= 0; --j) y = dec_round(y, nt, j);
  return y;
}

__device__ __forceinline__ uint64_t root(uint64_t a) {   // rho(a), a canonical
  uint64_t s = a;
#pragma unroll 59
  for (int n = 0; n < 59; ++n) s = sqr(s);
  s = canon(s);
  bool square = canon(sqr(s)) == a;
  bool odd = s & 1;
  uint64_t ns = s ? P - s : 0;
  return (odd != square) ? s : ns;                     // a square takes the even root, a non-square the odd one
}

template <int K>
__device__ __forceinline__ uint64_t encode_word(uint64_t x, uint64_t t) {
#pragma unroll
  for (int j = 0; j < K; ++j) x = root(canon(fold(x + kC[j] + t)));
  return x;
}

__device__ __forceinline__ uint64_t addmod(uint64_t a, uint64_t b) { return canon(fold(a + b)); }

__device__ __forceinline__ uint64_t mulmod(uint64_t a, uint64_t b) {   // a, b canonical; once per thread, not in a loop
  unsigned __int128 q = (unsigned __int128)a * b;
  uint64_t lo = (uint64_t)q & P, hi = (uint64_t)(q >> 61);
  return canon(fold(lo + hi));
}

// --- flat store: decode, encode, read, tag ---------------------------------------------------------------------------

template <int K, int MODE>
__global__ void __launch_bounds__(256) decode_kernel(const uint64_t* __restrict__ store, uint64_t* __restrict__ out,
                                                     uint64_t n, uint64_t base, unsigned long long* xr) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x * 2;
  uint64_t i = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 2;
  uint64_t t0 = mulmod(kTau, (base + i) % P), tstep = mulmod(kTau, stride % P);
  uint64_t acc = 0;
  for (; i + 1 < n; i += stride) {
    ulonglong2 w = __ldcs(reinterpret_cast<const ulonglong2*>(store + i));
    uint64_t t1 = addmod(t0, kTau);
    uint64_t m0 = decode_word<K>(w.x, ~t0), m1 = decode_word<K>(w.y, ~t1);
    if (MODE == 2) *reinterpret_cast<ulonglong2*>(out + i) = make_ulonglong2(m0, m1);
    acc ^= m0 ^ (m1 << 1);
    t0 = addmod(t0, tstep);
  }
  if (i < n) {
    uint64_t m0 = decode_word<K>(store[i], ~t0);
    if (MODE == 2) out[i] = m0;
    acc ^= m0;
  }
  atomicXor(xr, (unsigned long long)acc);
}

template <int K>
__global__ void __launch_bounds__(256) encode_kernel(const uint64_t* __restrict__ in, uint64_t* __restrict__ out, uint64_t n,
                                                     uint64_t base) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t t = mulmod(kTau, (base + i) % P), tstep = mulmod(kTau, stride % P);
  for (; i < n; i += stride) {
    out[i] = encode_word<K>(in[i], t);
    t = addmod(t, tstep);
  }
}

__global__ void __launch_bounds__(256) read_kernel(const uint64_t* __restrict__ store, uint64_t n, unsigned long long* xr) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x * 2;
  uint64_t acc = 0;
  for (uint64_t i = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 2; i + 1 < n; i += stride) {
    ulonglong2 w = __ldcs(reinterpret_cast<const ulonglong2*>(store + i));
    acc ^= w.x ^ (w.y << 1);
  }
  atomicXor(xr, (unsigned long long)acc);
}

__device__ __forceinline__ uint32_t horner31(uint32_t acc, uint32_t g, uint32_t w) {   // acc, w < 2^32, g < 2^31
  uint64_t s = (uint64_t)acc * g + w;
  uint64_t f = (s & M31) + (s >> 31);                  // < 2^31 + 2^32
  return (uint32_t)((f & M31) + (f >> 31));            // < 2^31 + 3
}

__global__ void __launch_bounds__(256) tag_kernel(const uint64_t* __restrict__ store, uint64_t n, uint32_t g,
                                                  unsigned long long* xr) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x * 2;
  uint32_t a = 0, b = 0;
  for (uint64_t i = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 2; i + 1 < n; i += stride) {
    ulonglong2 w = __ldcs(reinterpret_cast<const ulonglong2*>(store + i));
    a = horner31(horner31(a, g, (uint32_t)w.x), g, (uint32_t)(w.x >> 32));
    b = horner31(horner31(b, g, (uint32_t)w.y), g, (uint32_t)(w.y >> 32));
  }
  atomicXor(xr, (unsigned long long)a << 32 | b);
}

// --- compute only ----------------------------------------------------------------------------------------------------

constexpr int U = 4;

template <int K>
__global__ void __launch_bounds__(256) chain_decode(uint64_t steps, uint64_t tstep, unsigned long long* xr) {
  uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t y[U], t[U];
#pragma unroll
  for (int u = 0; u < U; ++u) { y[u] = canon((id * U + u) * 0x9E3779B97F4A7C15ull & P); t[u] = mulmod(kTau, id * U + u); }
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < U; ++u) { y[u] = decode_word<K>(y[u], ~t[u]); t[u] = addmod(t[u], tstep); }
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < U; ++u) acc ^= y[u];
  atomicXor(xr, (unsigned long long)acc);
}

template <int K>
__global__ void __launch_bounds__(256) chain_encode(uint64_t steps, uint64_t tstep, unsigned long long* xr) {
  uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t y[U], t[U];
#pragma unroll
  for (int u = 0; u < U; ++u) { y[u] = canon((id * U + u) * 0x9E3779B97F4A7C15ull & P); t[u] = mulmod(kTau, id * U + u); }
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < U; ++u) { y[u] = encode_word<K>(y[u], t[u]); t[u] = addmod(t[u], tstep); }
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < U; ++u) acc ^= y[u];
  atomicXor(xr, (unsigned long long)acc);
}

template <int UU>
__global__ void __launch_bounds__(256) chain_square(uint64_t steps, unsigned long long* xr) {
  uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t y[UU];
#pragma unroll
  for (int u = 0; u < UU; ++u) y[u] = canon((id * UU + u) * 0x9E3779B97F4A7C15ull & P);
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll 59
    for (int n = 0; n < 59; ++n) {
#pragma unroll
      for (int u = 0; u < UU; ++u) y[u] = sqr(y[u]);
    }
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < UU; ++u) acc ^= y[u];
  atomicXor(xr, (unsigned long long)acc);
}

// One 64-bit word a step feeds all 2U accumulators: the tag's two Horner steps a word, plus one 64-bit add a step for
// 2U words (under 2% of the tag's own work).
__global__ void __launch_bounds__(256) chain_tag(uint64_t steps, uint32_t g, unsigned long long* xr) {
  uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a[2 * U];
#pragma unroll
  for (int u = 0; u < 2 * U; ++u) a[u] = (uint32_t)(id * 2 * U + u) * 0x9E3779B9u;
  uint64_t w = id * 0x9E3779B97F4A7C15ull;
  for (uint64_t s = 0; s < steps; ++s) {
    w += 0x9E3779B97F4A7C15ull;
#pragma unroll
    for (int u = 0; u < 2 * U; ++u) a[u] = horner31(horner31(a[u], g, (uint32_t)w), g, (uint32_t)(w >> 32));
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < 2 * U; ++u) acc ^= (uint64_t)a[u] << (u & 1) * 32;
  atomicXor(xr, (unsigned long long)acc);
}

// ro-blake3/v1's compression: UU chains a thread, one keyed compression of a full 64-byte block a step, the output the
// next message's first half.
template <int UU>
__global__ void __launch_bounds__(256) chain_b3(uint64_t steps, bw::Key3 key, unsigned long long* xr) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t h[UU][8];
#pragma unroll
  for (int u = 0; u < UU; ++u)
#pragma unroll
    for (int i = 0; i < 8; ++i) h[u][i] = (id * UU + u) * 0x9E3779B9u + i;
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < UU; ++u) {
      uint32_t m[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = i < 8 ? h[u][i] : (uint32_t)s + i;
      bw::b3_compress(key.k, m, 0, 64, bw::B3_KEYED | bw::B3_CHUNK_START, h[u]);
    }
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < UU; ++u) acc ^= (uint64_t)h[u][0] << 32 | h[u][1];
  atomicXor(xr, (unsigned long long)acc);
}

// The commit's SHA-512 compression: UU chains a thread, one compression of a 128-byte block a step, the state the next
// block's first half.
template <int UU>
__global__ void __launch_bounds__(256) chain_sha512(uint64_t steps, unsigned long long* xr) {
  const uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t h[UU][8];
#pragma unroll
  for (int u = 0; u < UU; ++u)
#pragma unroll
    for (int i = 0; i < 8; ++i) h[u][i] = (id * UU + u) * 0x9E3779B97F4A7C15ull + i;
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < UU; ++u) {
      uint64_t w[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) w[i] = i < 8 ? h[u][i] : s + i;
      bw::compress(h[u], w);
    }
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < UU; ++u) acc ^= h[u][0];
  atomicXor(xr, (unsigned long long)acc);
}

// --- the FP8 GEMM, plain and fused ----------------------------------------------------------------------------------

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t take4(const uint64_t* x, int bit) {   // stream bits bit .. bit + 31 of the payloads
  int w = bit / PB, sh = bit % PB;
  uint32_t v = (uint32_t)(x[w] >> sh);
  if (sh + 32 > PB) v |= (uint32_t)(x[w + 1] << (PB - sh));
  return v;
}

// KR = 0: plain FP8 fragments; KR >= 1: codec words with KR rounds. NB8: batch columns in groups of 8.
// PARTIALS: write each chunk's sums to y[(kc M + b) N + n] (deterministic, the gate) instead of atomically adding them.
template <int KR, int NB8, bool PARTIALS>
__global__ void __launch_bounds__(GEMM_WARPS * 32) gemm_kernel(const void* __restrict__ wstore, const uint8_t* __restrict__ x,
                                                               float* __restrict__ y, int M, int N, int Kd, uint64_t base) {
  __shared__ __align__(16) uint8_t xs[NB8 * 8][XS_STRIDE];
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks;
  const int nb = (blockIdx.x / nchunks) * GEMM_WARPS + threadIdx.x / 32;
  const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const uint64_t q = (uint64_t)nb * nchunks + kc;
  for (int e = threadIdx.x; e < NB8 * 8 * TASK_COLS / 16; e += GEMM_WARPS * 32) {
    int b = e / (TASK_COLS / 16), c = e % (TASK_COLS / 16);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < M) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)kc * TASK_COLS) + c);
    *reinterpret_cast<uint4*>(&xs[b][16 * c]) = v;
  }
  __syncthreads();
  if (nb * 16 >= N) return;
  float acc[NB8][4];
#pragma unroll
  for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  auto tile = [&](int tt, const uint32_t (&a)[4]) {
#pragma unroll
    for (int j = 0; j < NB8; ++j) {
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 4 * tq]);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 16 + 4 * tq]);
      mma_e4m3(acc[j], a, b0, b1);
    }
  };

  if constexpr (KR == 0) {
    const uint4* wp = reinterpret_cast<const uint4*>(wstore) + q * TASK_TILES * LANES + lane;
    uint4 w[TASK_TILES];
#pragma unroll
    for (int tt = 0; tt < TASK_TILES; ++tt) w[tt] = __ldcs(wp + tt * LANES);
#pragma unroll
    for (int tt = 0; tt < TASK_TILES; ++tt) {
      const uint32_t a[4] = {w[tt].x, w[tt].y, w[tt].z, w[tt].w};
      tile(tt, a);
    }
  } else {
    const uint64_t* wp = reinterpret_cast<const uint64_t*>(wstore) + q * TASK_WORDS + lane;
    uint64_t xw[LANE_WORDS];
#pragma unroll
    for (int w = 0; w < LANE_WORDS; ++w) xw[w] = __ldcs(reinterpret_cast<const unsigned long long*>(wp + w * LANES));
    uint64_t t = mulmod(kTau, (base + q * TASK_WORDS + lane) % P);
    const uint64_t t32 = mulmod(kTau, LANES);
#pragma unroll
    for (int w = 0; w < LANE_WORDS; ++w) {
      xw[w] = decode_word<KR>(xw[w], ~t);
      t = addmod(t, t32);
    }
#pragma unroll
    for (int tt = 0; tt < TASK_TILES; ++tt) {
      const uint32_t a[4] = {take4(xw, 128 * tt), take4(xw, 128 * tt + 32), take4(xw, 128 * tt + 64),
                             take4(xw, 128 * tt + 96)};
      tile(tt, a);
    }
  }

#pragma unroll
  for (int j = 0; j < NB8; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int n = nb * 16 + g + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
      if (b < M) {
        if constexpr (PARTIALS) y[((size_t)kc * M + b) * N + n] = acc[j][e];
        else atomicAdd(y + (size_t)b * N + n, acc[j][e]);
      }
    }
  }
}

// The persistent form: a CTA keeps one 512-column chunk of x in shared memory and its warps walk the 16-row blocks
// nb = g0 * GEMM_WARPS + warp + i * groups * GEMM_WARPS. A task is taken in two halves, tiles 0-7 from words 0 .. H0 - 1
// and tiles 8-15 from word H0 - 1 and words H0 .. LANE_WORDS - 1, and each half's loads are in flight while the other
// half is decoded and multiplied: the task's second half while its first is, the next task's first half while its
// second is. KR = 0 reads plain fragments, eight tiles (16 words) a half.
constexpr int HALF_BITS = 8 * LANE_BYTES / 2;
constexpr int H0 = (HALF_BITS + PB - 1) / PB;
constexpr int H1 = LANE_WORDS - H0;
static_assert(H0 >= 16 && H1 >= 16 && PB * (H0 - 1) <= HALF_BITS, "a half's words");

template <int KR>
__device__ __forceinline__ void load_half(uint64_t* xw, const void* __restrict__ wstore, uint64_t q, int lane, int half) {
  if constexpr (KR == 0) {
    const uint4* wp = reinterpret_cast<const uint4*>(wstore) + (q * TASK_TILES + half * TASK_TILES / 2) * LANES + lane;
#pragma unroll
    for (int tt = 0; tt < TASK_TILES / 2; ++tt) {
      uint4 v = __ldcs(wp + tt * LANES);
      xw[2 * tt] = lohi(v.x, v.y);
      xw[2 * tt + 1] = lohi(v.z, v.w);
    }
  } else {
    const unsigned long long* wp =
        reinterpret_cast<const unsigned long long*>(wstore) + q * TASK_WORDS + (half ? H0 : 0) * LANES + lane;
    if (half) {
#pragma unroll
      for (int w = 0; w < H1; ++w) xw[w] = __ldcs(wp + w * LANES);
    } else {
#pragma unroll
      for (int w = 0; w < H0; ++w) xw[w] = __ldcs(wp + w * LANES);
    }
  }
}

// Tiles t0 .. t0 + 7; xw[0] starts at stream bit `off` (KR > 0) or holds tile t0's first 8 bytes (KR = 0).
template <int KR, int NB8>
__device__ __forceinline__ void half_tiles(float (&acc)[NB8][4], const uint64_t* xw, int t0, int off,
                                           const uint8_t (*xs)[XS_STRIDE], int g, int tq) {
#pragma unroll
  for (int i = 0; i < TASK_TILES / 2; ++i) {
    const int tt = t0 + i;
    uint32_t a[4];
    if constexpr (KR == 0) {
      a[0] = (uint32_t)xw[2 * i]; a[1] = (uint32_t)(xw[2 * i] >> 32);
      a[2] = (uint32_t)xw[2 * i + 1]; a[3] = (uint32_t)(xw[2 * i + 1] >> 32);
    } else {
      const int o = 128 * tt - off;
      a[0] = take4(xw, o); a[1] = take4(xw, o + 32); a[2] = take4(xw, o + 64); a[3] = take4(xw, o + 96);
    }
#pragma unroll
    for (int j = 0; j < NB8; ++j) {
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 4 * tq]);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 16 + 4 * tq]);
      mma_e4m3(acc[j], a, b0, b1);
    }
  }
}

template <int KR, int NB8, bool PARTIALS>
__global__ void __launch_bounds__(GEMM_WARPS * 32) gemm_pipe_kernel(const void* __restrict__ wstore,
                                                                    const uint8_t* __restrict__ x, float* __restrict__ y,
                                                                    int M, int N, int Kd, uint64_t base, int groups) {
  __shared__ __align__(16) uint8_t xs[NB8 * 8][XS_STRIDE];
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks, g0 = blockIdx.x / nchunks;
  const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3, nbs = N / 16, stride = groups * GEMM_WARPS;
  for (int e = threadIdx.x; e < NB8 * 8 * TASK_COLS / 16; e += GEMM_WARPS * 32) {
    int b = e / (TASK_COLS / 16), c = e % (TASK_COLS / 16);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < M) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)kc * TASK_COLS) + c);
    *reinterpret_cast<uint4*>(&xs[b][16 * c]) = v;
  }
  __syncthreads();
  int nb = g0 * GEMM_WARPS + threadIdx.x / 32;
  if (nb >= nbs) return;
  const uint64_t t32 = KR > 0 ? mulmod(kTau, LANES) : 0;
  uint64_t A[H0], B[H1];
  load_half<KR>(A, wstore, (uint64_t)nb * nchunks + kc, lane, 0);
  for (; nb < nbs; nb += stride) {
    const uint64_t q = (uint64_t)nb * nchunks + kc;
    load_half<KR>(B, wstore, q, lane, 1);
    float acc[NB8][4];
#pragma unroll
    for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    uint64_t t = 0;
    if constexpr (KR > 0) {
      t = mulmod(kTau, (base + q * TASK_WORDS + lane) % P);
#pragma unroll
      for (int w = 0; w < H0; ++w) {
        A[w] = decode_word<KR>(A[w], ~t);
        t = addmod(t, t32);
      }
    }
    half_tiles<KR, NB8>(acc, A, 0, 0, xs, g, tq);
    uint64_t h[H1 + 1];
    if constexpr (KR > 0) h[0] = A[H0 - 1];
    if (nb + stride < nbs) load_half<KR>(A, wstore, (uint64_t)(nb + stride) * nchunks + kc, lane, 0);
    if constexpr (KR > 0) {
#pragma unroll
      for (int w = 0; w < H1; ++w) {
        h[w + 1] = decode_word<KR>(B[w], ~t);
        t = addmod(t, t32);
      }
      half_tiles<KR, NB8>(acc, h, TASK_TILES / 2, PB * (H0 - 1), xs, g, tq);
    } else {
      half_tiles<KR, NB8>(acc, B, TASK_TILES / 2, 0, xs, g, tq);
    }
#pragma unroll
    for (int j = 0; j < NB8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int n = nb * 16 + g + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
        if (b < M) {
          if constexpr (PARTIALS) y[((size_t)kc * M + b) * N + n] = acc[j][e];
          else atomicAdd(y + (size_t)b * N + n, acc[j][e]);
        }
      }
    }
  }
}

// The warp-specialized form: a CTA keeps one 512-column chunk of x in shared memory and takes the 16-row blocks
// nb = g0 + i * groups. Consumer warp c of C takes tasks i = c + u C, u = 0, 1, ..., from a ring of S stages of its own
// (u mod S). The last warp's lane 0 is the producer: in task order it copies task i's words (or plain fragments) into
// its stage with one bulk asynchronous copy, completing on the stage's `full` barrier, once the stage's `empty` barrier
// says the consumer read task i - S C. A consumer reads its stage into registers, releases it, decodes, and runs the 16
// tiles in order, so its sums are the plain kernel's bit for bit. Loads are in flight in shared memory, not in
// registers, so S C, not occupancy, sets how much of the read is hidden.
//
// A stage is waited on only by its consumer, which has waited on its previous use: copies complete out of order, and a
// parity wait one use ahead of an incomplete phase would pass at once.
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}\n" ::"r"(
          smem_u32(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}

#ifndef WC_G
#define WC_G 4
#endif

// A lane's words decoded WC_G at a time, round by round across the group, so a warp has WC_G independent chains in
// flight; word w's tweak is t0 + 32 w tau.
template <int KR>
__device__ __forceinline__ void decode_lane(uint64_t (&xw)[LANE_WORDS], uint64_t t, uint64_t t32) {
#pragma unroll
  for (int w0 = 0; w0 < LANE_WORDS; w0 += WC_G) {
    uint64_t nt[WC_G];
#pragma unroll
    for (int i = 0; i < WC_G; ++i) {
      nt[i] = ~t;
      t = addmod(t, t32);
    }
#pragma unroll
    for (int j = KR - 1; j >= 0; --j) {
#pragma unroll
      for (int i = 0; i < WC_G; ++i)
        if (w0 + i < LANE_WORDS) xw[w0 + i] = dec_round(xw[w0 + i], nt[i], j);
    }
  }
}

constexpr int WS_MAX_CONSUMERS = 8;
constexpr int WS_MAX_WARPS = 10;   // consumers, the producer and the committers

// The tag and the commit (both off when the pointers are null): `tags` gets block j = jbase + 32 q + l's ro-blake3/v1 tag
// as 8 little-endian words at 8 (32 q + l); `committers` warps hash the leaves of `ctags` (`cblocks` tags, the previous
// call's) into `leaves`, 8 big-endian words each.
struct WsTag {
  bw::Key3 key;
  bw::Mid leafmid;
  uint64_t jbase;
  uint32_t* tags;
  const uint32_t* ctags;
  uint64_t cblocks;
  uint64_t* leaves;
  int committers;
};

// gemm_ws_kernel's MODE, fixed at compile time so no branch splits the consumer's code: the tag, the decode, and both
// interleaved (tag_decode_lane). Without the decode the partial sums are the encoded words' (timing arms only).
constexpr int WS_TAG = 1, WS_DECODE = 2, WS_INTERLEAVE = 4;

// ro-blake3/v1 over one lane's block: the message u64be j ‖ the LANE_WORDS words little-endian, one chunk of 8 + 8
// LANE_WORDS bytes (five compressions at either packing).
__device__ __forceinline__ void b3_lane_tag(const bw::Key3& key, uint64_t j, const uint64_t (&xw)[LANE_WORDS],
                                            uint32_t (&cv)[8]) {
  constexpr int MSG = 2 + 2 * LANE_WORDS, BLOCKS = (MSG + 15) / 16;
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key.k[i];
#pragma unroll
  for (int b = 0; b < BLOCKS; ++b) {
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int u = 16 * b + i;
      m[i] = u == 0   ? bw::bswap32((uint32_t)(j >> 32))
             : u == 1 ? bw::bswap32((uint32_t)j)
             : u < MSG ? (uint32_t)(xw[(u - 2) / 2] >> (32 * (u & 1)))
                       : 0u;
    }
    const uint32_t len = b + 1 < BLOCKS ? 64 : 4 * (MSG - 16 * (BLOCKS - 1));
    const uint32_t flags =
        bw::B3_KEYED | (b == 0 ? bw::B3_CHUNK_START : 0) | (b + 1 == BLOCKS ? bw::B3_CHUNK_END | bw::B3_ROOT : 0);
    bw::b3_compress(cv, m, 0, len, flags, cv);
  }
}

// b3_lane_tag and decode_lane in one block of code: compression b over its words, still encoded, beside the decode of
// compression b - 1's, which no later compression reads, so the scheduler can issue the tag's logic and the decode's
// multiplies together. Compression b's message holds words max(0, 8 b - 1) .. 8 b + 6, each whole.
template <int KR>
__device__ __forceinline__ void tag_decode_lane(const bw::Key3& key, uint64_t j, uint64_t (&xw)[LANE_WORDS], uint64_t t,
                                                uint64_t t32, uint32_t (&cv)[8]) {
  constexpr int MSG = 2 + 2 * LANE_WORDS, BLOCKS = (MSG + 15) / 16;
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key.k[i];
#pragma unroll
  for (int b = 0; b <= BLOCKS; ++b) {
    if (b < BLOCKS) {
      uint32_t m[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int u = 16 * b + i;
        m[i] = u == 0   ? bw::bswap32((uint32_t)(j >> 32))
               : u == 1 ? bw::bswap32((uint32_t)j)
               : u < MSG ? (uint32_t)(xw[(u - 2) / 2] >> (32 * (u & 1)))
                         : 0u;
      }
      const uint32_t len = b + 1 < BLOCKS ? 64 : 4 * (MSG - 16 * (BLOCKS - 1));
      const uint32_t flags =
          bw::B3_KEYED | (b == 0 ? bw::B3_CHUNK_START : 0) | (b + 1 == BLOCKS ? bw::B3_CHUNK_END | bw::B3_ROOT : 0);
      bw::b3_compress(cv, m, 0, len, flags, cv);
    }
    if (b > 0) {
      const int lo = b == 1 ? 0 : 8 * (b - 1) - 1, hi = 8 * (b - 1) + 6 < LANE_WORDS ? 8 * (b - 1) + 6 : LANE_WORDS - 1;
#pragma unroll
      for (int w0 = lo; w0 <= hi; w0 += WC_G) {
        uint64_t nt[WC_G];
#pragma unroll
        for (int i = 0; i < WC_G; ++i) {
          nt[i] = ~t;
          if (w0 + i <= hi) t = addmod(t, t32);
        }
#pragma unroll
        for (int r = KR - 1; r >= 0; --r) {
#pragma unroll
          for (int i = 0; i < WC_G; ++i)
            if (w0 + i <= hi) xw[w0 + i] = dec_round(xw[w0 + i], nt[i], r);
        }
      }
    }
  }
}

// One leaf of bw::b3_leaves: tags 16 c .. 16 c + 15 of B (u64be c ‖ their 512 bytes, five compressions after the leaf
// prefix's midstate).
__device__ __forceinline__ void b3_leaf(const uint32_t* __restrict__ tags, uint64_t B, uint64_t leaf, const bw::Mid& mid,
                                        uint64_t* __restrict__ leaves) {
  const uint64_t first = leaf * 16;
  const int n = (int)(B - first < 16 ? B - first : 16);
  const int data = 1 + 4 * n, nb = (8 * data + 17 + 127) / 128;
  const uint32_t* T = tags + first * 8;
  uint64_t s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = mid.h[i];
#pragma unroll 1
  for (int blk = 0; blk < nb; ++blk) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i;
      uint64_t v = 0;
      if (q == 0)
        v = leaf;
      else if (q < data)
        v = ((uint64_t)bw::bswap32(T[2 * (q - 1)]) << 32) | bw::bswap32(T[2 * (q - 1) + 1]);
      else if (q == data)
        v = 0x80ULL << 56;
      if (q == 16 * nb - 1) v = (uint64_t)(128 + 8 + 32 * n) * 8;
      w[i] = v;
    }
    bw::compress(s, w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) leaves[leaf * 8 + i] = s[i];
}

template <int KR>
__host__ __device__ constexpr int ws_stage_bytes() { return KR > 0 ? TASK_WORDS * 8 : TASK_TILES * LANES * 16; }

template <int KR, int NB8>
__host__ __device__ constexpr int ws_smem_bytes(int stages) {
  return stages * ws_stage_bytes<KR>() + NB8 * 8 * XS_STRIDE + 2 * stages * 8;
}

template <int KR, int NB8, bool PARTIALS, int MODE>
__global__ void __launch_bounds__(WS_MAX_WARPS * 32) gemm_ws_kernel(const void* __restrict__ wstore,
                                                                   const uint8_t* __restrict__ x, float* __restrict__ y,
                                                                   int M, int N, int Kd, uint64_t base, int groups,
                                                                   int consumers, int per_consumer, WsTag tg) {
  constexpr int SB = ws_stage_bytes<KR>();
  const int stages = consumers * per_consumer;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t (*xs)[XS_STRIDE] = reinterpret_cast<uint8_t (*)[XS_STRIDE]>(smem + stages * SB);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + stages * SB + NB8 * 8 * XS_STRIDE);
  uint64_t* empty = full + stages;
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks, g0 = blockIdx.x / nchunks, nbs = N / 16;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  for (int e = threadIdx.x; e < NB8 * 8 * TASK_COLS / 16; e += blockDim.x) {
    int b = e / (TASK_COLS / 16), c = e % (TASK_COLS / 16);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < M) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)kc * TASK_COLS) + c);
    *reinterpret_cast<uint4*>(&xs[b][16 * c]) = v;
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int ntasks = nbs > g0 ? (nbs - g0 + groups - 1) / groups : 0;

  if (warp == consumers) {
    if (lane == 0) {
      for (int i = 0; i < ntasks; ++i) {
        const int u = i / consumers, s = (i % consumers) * per_consumer + u % per_consumer;
        if (u >= per_consumer) mbar_wait(empty + s, (u / per_consumer - 1) & 1);
        const uint64_t q = (uint64_t)(g0 + i * groups) * nchunks + kc;
        mbar_expect_tx(full + s, SB);
        bulk_load(smem + s * SB, static_cast<const uint8_t*>(wstore) + q * SB, SB, full + s);
      }
    }
    return;
  }
  if (warp > consumers) {
    const uint64_t nleaves = (tg.cblocks + 15) / 16, step = (uint64_t)gridDim.x * tg.committers * 32;
    for (uint64_t leaf = ((uint64_t)blockIdx.x * tg.committers + (warp - consumers - 1)) * 32 + lane; leaf < nleaves;
         leaf += step)
      b3_leaf(tg.ctags, tg.cblocks, leaf, tg.leafmid, tg.leaves);
    return;
  }

  const uint64_t t32 = KR > 0 ? mulmod(kTau, LANES) : 0;
  for (int u = 0, i = warp; i < ntasks; ++u, i += consumers) {
    const int s = warp * per_consumer + u % per_consumer, nb = g0 + i * groups;
    const uint64_t q = (uint64_t)nb * nchunks + kc;
    mbar_wait(full + s, (u / per_consumer) & 1);
    float acc[NB8][4];
#pragma unroll
    for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    auto tile = [&](int tt, const uint32_t (&a)[4]) {
#pragma unroll
      for (int j = 0; j < NB8; ++j) {
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 4 * tq]);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 16 + 4 * tq]);
        mma_e4m3(acc[j], a, b0, b1);
      }
    };
    if constexpr (KR == 0) {
      const uint4* st = reinterpret_cast<const uint4*>(smem + s * SB) + lane;
      uint4 w[TASK_TILES];
#pragma unroll
      for (int tt = 0; tt < TASK_TILES; ++tt) w[tt] = st[tt * LANES];
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + s);
#pragma unroll
      for (int tt = 0; tt < TASK_TILES; ++tt) {
        const uint32_t a[4] = {w[tt].x, w[tt].y, w[tt].z, w[tt].w};
        tile(tt, a);
      }
    } else {
      const unsigned long long* st = reinterpret_cast<const unsigned long long*>(smem + s * SB) + lane;
      uint64_t xw[LANE_WORDS];
#pragma unroll
      for (int w = 0; w < LANE_WORDS; ++w) xw[w] = st[w * LANES];
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + s);
      if constexpr ((MODE & WS_TAG) != 0) {
        uint32_t cv[8];
        if constexpr ((MODE & WS_INTERLEAVE) != 0)
          tag_decode_lane<KR>(tg.key, tg.jbase + q * LANES + lane, xw,
                              mulmod(kTau, (base + q * TASK_WORDS + lane) % P), t32, cv);
        else b3_lane_tag(tg.key, tg.jbase + q * LANES + lane, xw, cv);
        uint4* o = reinterpret_cast<uint4*>(tg.tags + 8 * (q * LANES + lane));
        o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
        o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
      }
      if constexpr ((MODE & WS_DECODE) != 0 && (MODE & WS_INTERLEAVE) == 0)
        decode_lane<KR>(xw, mulmod(kTau, (base + q * TASK_WORDS + lane) % P), t32);
#pragma unroll
      for (int tt = 0; tt < TASK_TILES; ++tt) {
        const uint32_t a[4] = {take4(xw, 128 * tt), take4(xw, 128 * tt + 32), take4(xw, 128 * tt + 64),
                               take4(xw, 128 * tt + 96)};
        tile(tt, a);
      }
    }
#pragma unroll
    for (int j = 0; j < NB8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int n = nb * 16 + g + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
        if (b < M) {
          if constexpr (PARTIALS) y[((size_t)kc * M + b) * N + n] = acc[j][e];
          else atomicAdd(y + (size_t)b * N + n, acc[j][e]);
        }
      }
    }
  }
}

// Lane block j's tag straight from the store (32 q + l's words are q TASK_WORDS + 32 w + l, a warp's loads coalesced),
// its words loaded a compression at a time; jj = jbase + j, its index in the tag.
__device__ __forceinline__ void audit_lane_tag(const uint64_t* __restrict__ store, const bw::Key3& key, uint64_t j,
                                               uint64_t jj, uint32_t* tags) {
  constexpr int MSG = 2 + 2 * LANE_WORDS, BLOCKS = (MSG + 15) / 16;
  const unsigned long long* src =
      reinterpret_cast<const unsigned long long*>(store) + (j / LANES) * TASK_WORDS + j % LANES;
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key.k[i];
#pragma unroll
  for (int b = 0; b < BLOCKS; ++b) {
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      const int u = 16 * b + i, w = (u - 2) / 2;
      const uint64_t v = u == 0 ? 0 : u < MSG ? __ldcs(src + (size_t)w * LANES) : 0;
      m[i] = u == 0 ? bw::bswap32((uint32_t)(jj >> 32)) : (uint32_t)v;
      m[i + 1] = u == 0 ? bw::bswap32((uint32_t)jj) : (uint32_t)(v >> 32);
    }
    const uint32_t len = b + 1 < BLOCKS ? 64 : 4 * (MSG - 16 * (BLOCKS - 1));
    const uint32_t flags =
        bw::B3_KEYED | (b == 0 ? bw::B3_CHUNK_START : 0) | (b + 1 == BLOCKS ? bw::B3_CHUNK_END | bw::B3_ROOT : 0);
    bw::b3_compress(cv, m, 0, len, flags, cv);
  }
  uint4* o = reinterpret_cast<uint4*>(tags + 8 * j);
  o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
  o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
}

// The audit's own pass over the store, no decode and no product: a thread a lane block. Its tags are gemm_ws_kernel's.
__global__ void __launch_bounds__(256) audit_tag_kernel(const uint64_t* __restrict__ store, uint64_t blocks,
                                                        bw::Key3 key, uint64_t jbase, uint32_t* __restrict__ tags) {
  for (uint64_t j = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; j < blocks; j += (uint64_t)gridDim.x * blockDim.x)
    audit_lane_tag(store, key, j, jbase + j, tags);
}

// b3_leaf over 16 tags this warp wrote before its __syncwarp, read through L1 (ld.ca), never the read-only path a
// const __restrict__ pointer allows.
__device__ __forceinline__ void audit_leaf(const uint32_t* T, uint64_t leaf, const bw::Mid& mid, uint64_t* out) {
  uint64_t s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = mid.h[i];
#pragma unroll 1
  for (int blk = 0; blk < 5; ++blk) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i;
      uint64_t v = 0;
      if (q == 0) {
        v = leaf;
      } else if (q <= 64) {
        const uint2 t = __ldca(reinterpret_cast<const uint2*>(T) + (q - 1));
        v = ((uint64_t)bw::bswap32(t.x) << 32) | bw::bswap32(t.y);
      } else if (q == 65) {
        v = 0x80ULL << 56;
      }
      if (q == 79) v = (uint64_t)(128 + 8 + 512) * 8;
      w[i] = v;
    }
    bw::compress(s, w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = s[i];
}

// The audit pass with its leaves: a warp tags 16 tasks' lane blocks (512, as audit_tag_kernel), then each lane hashes
// one of their 32 leaves from the tags just written, still in L1 or L2, so the tags make no second trip to memory.
// blocks % 512 == 0; the leaves are wc_commit_leaves'.
__global__ void __launch_bounds__(256) audit_commit_kernel(const uint64_t* __restrict__ store, uint64_t blocks,
                                                           bw::Key3 key, bw::Mid mid, uint64_t jbase, uint32_t* tags,
                                                           uint64_t* __restrict__ leaves) {
  const int lane = threadIdx.x % 32;
  const uint64_t warps = (uint64_t)gridDim.x * (blockDim.x / 32);
  for (uint64_t g = (blockIdx.x * (uint64_t)blockDim.x + threadIdx.x) / 32; g < blocks / 512; g += warps) {
#pragma unroll 1
    for (int u = 0; u < 16; ++u) {
      const uint64_t j = g * 512 + 32 * u + lane;
      audit_lane_tag(store, key, j, jbase + j, tags);
    }
    __syncwarp();
    const uint64_t c = g * 32 + lane;
    audit_leaf(tags + c * 16 * 8, c, mid, leaves + c * 8);
    __syncwarp();
  }
}

// A host launch through ctypes costs more than a small GEMM: a timed arm is queued behind `spin` so the device runs it
// back to back, and `empty` (the GEMM's grid, no work) prices what back to back still costs a launch.
__global__ void spin_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

__global__ void __launch_bounds__(GEMM_WARPS * 32) empty_kernel() {}

}  // namespace wc

using namespace wc;

// WC_KMAX < 8 builds only k <= WC_KMAX (a quicker build for the GEMM alone); a larger k returns 1.
#ifndef WC_KMAX
#define WC_KMAX 8
#endif
static_assert(WC_KMAX >= 1 && WC_KMAX <= MAXK, "WC_KMAX");
#define WC_CASE(n, F) case n: F(n); break;
#define WC_CASES_1(F) WC_CASE(1, F)
#if WC_KMAX >= 2
#define WC_CASES_2(F) WC_CASES_1(F) WC_CASE(2, F)
#else
#define WC_CASES_2(F) WC_CASES_1(F)
#endif
#if WC_KMAX >= 3
#define WC_CASES_3(F) WC_CASES_2(F) WC_CASE(3, F)
#else
#define WC_CASES_3(F) WC_CASES_2(F)
#endif
#if WC_KMAX >= 4
#define WC_CASES_4(F) WC_CASES_3(F) WC_CASE(4, F)
#else
#define WC_CASES_4(F) WC_CASES_3(F)
#endif
#if WC_KMAX >= 5
#define WC_CASES(F) WC_CASES_4(F) WC_CASE(5, F) WC_CASE(6, F) WC_CASE(7, F) WC_CASE(8, F)
#else
#define WC_CASES(F) WC_CASES_4(F)
#endif
#define WC_K_SWITCH(k, F) switch (k) { WC_CASES(F) default: return 1; }

extern "C" {

int wc_version() { return 1; }

int wc_set_key(const uint64_t* c, int k, uint64_t tau) {
  if (k < 1 || k > MAXK) return 1;
  uint64_t a[MAXK] = {}, b[MAXK] = {}, cc[MAXK] = {};
  for (int j = 0; j < k; ++j) { cc[j] = c[j]; a[j] = 2 * P - c[j] + 1; b[j] = 6 * P - c[j] + 1; }
  if (cudaMemcpyToSymbol(kC, cc, sizeof cc) || cudaMemcpyToSymbol(kA, a, sizeof a) || cudaMemcpyToSymbol(kB, b, sizeof b) ||
      cudaMemcpyToSymbol(kTau, &tau, sizeof tau))
    return 2;
  return cudaDeviceSynchronize() ? 3 : 0;
}

int wc_decode(const uint64_t* store, uint64_t* out, uint64_t n, uint64_t base, int k, unsigned long long* xr, int grid,
              cudaStream_t s) {
#define F(K) do { if (out) decode_kernel<K, 2><<<grid, 256, 0, s>>>(store, out, n, base, xr); \
                  else decode_kernel<K, 0><<<grid, 256, 0, s>>>(store, out, n, base, xr); } while (0)
  WC_K_SWITCH(k, F)
#undef F
  return cudaGetLastError() ? 4 : 0;
}

int wc_encode(const uint64_t* in, uint64_t* out, uint64_t n, uint64_t base, int k, int grid, cudaStream_t s) {
#define F(K) encode_kernel<K><<<grid, 256, 0, s>>>(in, out, n, base)
  WC_K_SWITCH(k, F)
#undef F
  return cudaGetLastError() ? 4 : 0;
}

int wc_read(const uint64_t* store, uint64_t n, unsigned long long* xr, int grid, cudaStream_t s) {
  read_kernel<<<grid, 256, 0, s>>>(store, n, xr);
  return cudaGetLastError() ? 4 : 0;
}

int wc_tag(const uint64_t* store, uint64_t n, uint32_t g, unsigned long long* xr, int grid, cudaStream_t s) {
  tag_kernel<<<grid, 256, 0, s>>>(store, n, g, xr);
  return cudaGetLastError() ? 4 : 0;
}

// what: 0 decode, 1 encode (k rounds a step), 2 the bare squaring (59 a step, `ilp` chains a thread), 3 the tag (two
// Horner steps a word, 2U words a thread). Steps: chain steps a thread.
int wc_chain(int what, int k, int ilp, uint64_t steps, uint64_t tstep, unsigned long long* xr, int grid, cudaStream_t s) {
  if (what == 0) {
#define F(K) chain_decode<K><<<grid, 256, 0, s>>>(steps, tstep, xr)
    WC_K_SWITCH(k, F)
#undef F
  } else if (what == 1) {
#define F(K) chain_encode<K><<<grid, 256, 0, s>>>(steps, tstep, xr)
    WC_K_SWITCH(k, F)
#undef F
  } else if (what == 2) {
    switch (ilp) {
      case 1: chain_square<1><<<grid, 256, 0, s>>>(steps, xr); break;
      case 2: chain_square<2><<<grid, 256, 0, s>>>(steps, xr); break;
      case 4: chain_square<4><<<grid, 256, 0, s>>>(steps, xr); break;
      case 8: chain_square<8><<<grid, 256, 0, s>>>(steps, xr); break;
      default: return 1;
    }
  } else if (what == 3) {
    chain_tag<<<grid, 256, 0, s>>>(steps, 0x2545F491u & M31, xr);
  } else if (what == 4) {
    bw::Key3 key;
    for (int i = 0; i < 8; ++i) key.k[i] = 0x01000193u * (i + 1);
    switch (ilp) {
      case 1: chain_b3<1><<<grid, 256, 0, s>>>(steps, key, xr); break;
      case 2: chain_b3<2><<<grid, 256, 0, s>>>(steps, key, xr); break;
      case 4: chain_b3<4><<<grid, 256, 0, s>>>(steps, key, xr); break;
      default: return 1;
    }
  } else if (what == 5) {
    switch (ilp) {
      case 1: chain_sha512<1><<<grid, 256, 0, s>>>(steps, xr); break;
      case 2: chain_sha512<2><<<grid, 256, 0, s>>>(steps, xr); break;
      default: return 1;
    }
  } else {
    return 1;
  }
  return cudaGetLastError() ? 4 : 0;
}

int wc_spin(long long cycles, cudaStream_t s) {
  spin_kernel<<<1, 1, 0, s>>>(cycles);
  return cudaGetLastError() ? 4 : 0;
}

int wc_empty(int grid, cudaStream_t s) {
  empty_kernel<<<grid, GEMM_WARPS * 32, 0, s>>>();
  return cudaGetLastError() ? 4 : 0;
}

int wc_gemm_constants(int* out) {
  out[0] = LANES; out[1] = TASK_TILES; out[2] = LANE_WORDS; out[3] = TASK_WORDS; out[4] = TASK_COLS; out[5] = GEMM_WARPS;
  out[6] = PB; out[7] = WC_KMAX;
  return 0;
}

int wc_gemm(int k, const void* w, const uint8_t* x, float* y, int M, int N, int Kd, uint64_t base, int partials,
            cudaStream_t s) {
  if (M < 1 || M > 32 || N % (16 * GEMM_WARPS) || Kd % TASK_COLS || k < 0 || k > MAXK) return 1;
  int nb8 = M <= 8 ? 1 : M <= 16 ? 2 : 4;
  int grid = (N / 16 / GEMM_WARPS) * (Kd / TASK_COLS);
#define G(KR, NB) do { if (partials) gemm_kernel<KR, NB, true><<<grid, GEMM_WARPS * 32, 0, s>>>(w, x, y, M, N, Kd, base); \
                       else gemm_kernel<KR, NB, false><<<grid, GEMM_WARPS * 32, 0, s>>>(w, x, y, M, N, Kd, base); } while (0)
#define GK(KR) do { if (nb8 == 1) G(KR, 1); else if (nb8 == 2) G(KR, 2); else G(KR, 4); } while (0)
  switch (k) {
    case 0: GK(0); break;
    WC_CASES(GK)
    default: return 1;
  }
#undef GK
#undef G
  return cudaGetLastError() ? 4 : 0;
}

// The persistent form. groups <= 0: as many CTAs as fit on the device at once, split evenly over the chunks; the grid
// actually launched (groups x chunks) is returned in *launched when it isn't null.
int wc_gemm_pipe(int k, const void* w, const uint8_t* x, float* y, int M, int N, int Kd, uint64_t base, int partials,
                 int groups, int* launched, cudaStream_t s) {
  if (M < 1 || M > 32 || N % (16 * GEMM_WARPS) || Kd % TASK_COLS || k < 0 || k > MAXK) return 1;
  const int nb8 = M <= 8 ? 1 : M <= 16 ? 2 : 4, nchunks = Kd / TASK_COLS, nbg = N / 16 / GEMM_WARPS;
  int sms = 0, per_sm = 0, dev = 0;
  if (groups <= 0 && (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev))) return 2;
#define G(KR, NB) do { \
    auto kern = partials ? gemm_pipe_kernel<KR, NB, true> : gemm_pipe_kernel<KR, NB, false>; \
    if (groups <= 0) { \
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, GEMM_WARPS * 32, 0)) return 2; \
      groups = sms * per_sm / nchunks > 0 ? sms * per_sm / nchunks : 1; \
    } \
    if (groups > nbg) groups = nbg; \
    kern<<<groups * nchunks, GEMM_WARPS * 32, 0, s>>>(w, x, y, M, N, Kd, base, groups); } while (0)
#define GK(KR) do { if (nb8 == 1) G(KR, 1); else if (nb8 == 2) G(KR, 2); else G(KR, 4); } while (0)
  switch (k) {
    case 0: GK(0); break;
    WC_CASES(GK)
    default: return 1;
  }
#undef GK
#undef G
  if (launched) *launched = groups * nchunks;
  return cudaGetLastError() ? 4 : 0;
}

// The warp-specialized form: `consumers` consumer warps and one producer warp a CTA, `per_consumer` stages each consumer's
// ring. groups <= 0: as many CTAs as fit on the device at once (times -groups when groups < 0), split evenly over the
// chunks.
// The tag's key and the leaves' midstate, derived on the host from the 32-byte seed and kept for the last seed.
static const WsTag& ws_tag_keys(const uint8_t* seed) {
  static uint8_t last[32];
  static bool have = false;
  static WsTag t{};
  if (!have || memcmp(last, seed, 32)) {
    memcpy(last, seed, 32);
    have = true;
    t.key = bw::b3_key(seed);
    t.leafmid = bw::prefix_midstate(bw::LEAF_DOMAIN, bw::TAG_NAMES[bw::TAG_B3], seed);
  }
  return t;
}

// The warp-specialized form with the tag and the commit: seed (32 bytes) or null for neither; tags (null: no tag) as in
// WsTag, jbase the first block's index; committers > 0 warps hash the leaves of ctags (cblocks tags) into leaves;
// decode = 0 skips the decode (the partial sums are then the encoded words', a timing arm only), and with a tag decode = 2
// interleaves it with the decode (tag_decode_lane).
int wc_gemm_ws_tag(int k, const void* w, const uint8_t* x, float* y, int M, int N, int Kd, uint64_t base, int partials,
                   int consumers, int per_consumer, int groups, int* launched, int decode, const uint8_t* seed,
                   uint64_t jbase, uint32_t* tags, int committers, const uint32_t* ctags, uint64_t cblocks,
                   uint64_t* leaves, cudaStream_t s) {
  if (M < 1 || M > 32 || N % 16 || Kd % TASK_COLS || k < 0 || k > MAXK || consumers < 1 ||
      consumers > WS_MAX_CONSUMERS || per_consumer < 1 || consumers * per_consumer > 32 || committers < 0 ||
      consumers + 1 + committers > WS_MAX_WARPS || ((tags || committers) && (!seed || k == 0)) ||
      (committers && (!ctags || !leaves || !cblocks)))
    return 1;
  WsTag tg{};
  if (seed) tg = ws_tag_keys(seed);
  tg.jbase = jbase;
  tg.tags = tags;
  tg.ctags = ctags;
  tg.cblocks = cblocks;
  tg.leaves = leaves;
  tg.committers = committers;
  const int mode = (tags ? WS_TAG : 0) | (decode ? WS_DECODE : 0) | (tags && decode == 2 ? WS_INTERLEAVE : 0);
  const int stages = consumers * per_consumer;
  const int nb8 = M <= 8 ? 1 : M <= 16 ? 2 : 4, nchunks = Kd / TASK_COLS, nbs = N / 16;
  const int threads = (consumers + 1 + committers) * 32;
  int sms = 0, per_sm = 0, dev = 0;
  if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)) return 2;
#define G(KR, NB, T) do { \
    auto kern = partials ? gemm_ws_kernel<KR, NB, true, T> : gemm_ws_kernel<KR, NB, false, T>; \
    const int smem = ws_smem_bytes<KR, NB>(stages); \
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) return 2; \
    if (groups <= 0) { \
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, threads, smem) || per_sm < 1) return 2; \
      const int mult = groups < 0 ? -groups : 1; \
      groups = mult * sms * per_sm / nchunks > 0 ? mult * sms * per_sm / nchunks : 1; \
    } \
    if (groups > nbs) groups = nbs; \
    kern<<<groups * nchunks, threads, smem, s>>>(w, x, y, M, N, Kd, base, groups, consumers, per_consumer, tg); } while (0)
#define GN(KR, T) do { if (nb8 == 1) G(KR, 1, T); else if (nb8 == 2) G(KR, 2, T); else G(KR, 4, T); } while (0)
#define GK(KR) do { \
    switch (mode) { \
      case 0: GN(KR, 0); break; \
      case WS_TAG: GN(KR, WS_TAG); break; \
      case WS_DECODE: GN(KR, WS_DECODE); break; \
      case WS_TAG | WS_DECODE: GN(KR, WS_TAG | WS_DECODE); break; \
      default: GN(KR, WS_TAG | WS_DECODE | WS_INTERLEAVE); \
    } } while (0)
  switch (k) {
    case 0: GN(0, WS_DECODE); break;
    WC_CASES(GK)
    default: return 1;
  }
#undef GK
#undef GN
#undef G
  if (launched) *launched = groups * nchunks;
  return cudaGetLastError() ? 4 : 0;
}

int wc_gemm_ws(int k, const void* w, const uint8_t* x, float* y, int M, int N, int Kd, uint64_t base, int partials,
               int consumers, int per_consumer, int groups, int* launched, cudaStream_t s) {
  return wc_gemm_ws_tag(k, w, x, y, M, N, Kd, base, partials, consumers, per_consumer, groups, launched, 1, nullptr, 0,
                        nullptr, 0, nullptr, 0, nullptr, s);
}

// The audit's own tag pass: `blocks` lane blocks of the encoded store, their tags into `tags` as gemm_ws_tag writes them.
int wc_audit_tag(const void* store, uint64_t blocks, const uint8_t* seed, uint64_t jbase, uint32_t* tags, int grid,
                 cudaStream_t s) {
  if (!store || !seed || !tags || !blocks || grid < 1 || blocks % LANES) return 1;
  audit_tag_kernel<<<grid, 256, 0, s>>>(static_cast<const uint64_t*>(store), blocks, ws_tag_keys(seed).key, jbase, tags);
  return cudaGetLastError() ? 4 : 0;
}

// wc_audit_tag with the commit's leaves (a workspace's wc_commit_leaves_at) in the same pass: blocks % 512 == 0.
int wc_audit_commit(const void* store, uint64_t blocks, const uint8_t* seed, uint64_t jbase, uint32_t* tags,
                    uint64_t* leaves, int grid, cudaStream_t s) {
  if (!store || !seed || !tags || !leaves || !blocks || grid < 1 || blocks % 512) return 1;
  const WsTag& tg = ws_tag_keys(seed);
  audit_commit_kernel<<<grid, 256, 0, s>>>(static_cast<const uint64_t*>(store), blocks, tg.key, tg.leafmid, jbase, tags,
                                           leaves);
  return cudaGetLastError() ? 4 : 0;
}

// The commit's workspace for `blocks` tags (tagger.cu's layout: the tags, then every tree level, the leaves first).
size_t wc_commit_bytes(uint64_t blocks) { return bw::workspace_bytes(bw::TAG_B3, blocks); }
uint32_t* wc_commit_tags(void* workspace, uint64_t blocks) { return bw::make_pass(bw::TAG_B3, blocks, workspace).tags; }
uint64_t* wc_commit_leaves_at(void* workspace, uint64_t blocks) {
  return bw::make_pass(bw::TAG_B3, blocks, workspace).nodes;
}

// The leaves of the workspace's tags as their own pass (tagger.cu's b3_leaves), against the committer warps.
int wc_commit_leaves(void* workspace, uint64_t blocks, const uint8_t* seed, cudaStream_t s) {
  bw::Pass p = bw::make_pass(bw::TAG_B3, blocks, workspace);
  const uint64_t leaves = (blocks + 15) / 16;
  bw::b3_leaves<<<(unsigned)((leaves + 127) / 128), 128, 0, s>>>(p.tags, blocks, blocks, ws_tag_keys(seed).leafmid,
                                                                  p.nodes);
  return cudaGetLastError() ? 4 : 0;
}

// The tree above the workspace's leaves, and its root (64 bytes) when root isn't null.
int wc_commit_tree(void* workspace, uint64_t blocks, uint8_t* root, cudaStream_t s) {
  bw::Pass p = bw::make_pass(bw::TAG_B3, blocks, workspace);
  bw::launch_tree(p, s);
  if (cudaGetLastError()) return 4;
  return root ? bw::read_root(p, root, s) : 0;
}

}  // extern "C"
