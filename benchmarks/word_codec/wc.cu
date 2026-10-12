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
//                        (`codec.fragment_payloads` / `codec.plain_fragments`), split over 512-column chunks
#include <cstdint>
#include <cuda_runtime.h>

namespace wc {

constexpr uint64_t P = (1ull << 61) - 1;
constexpr int MAXK = 8;
constexpr int LANES = 32;
constexpr int TASK_TILES = 16;
constexpr int LANE_BYTES = 16 * TASK_TILES;
constexpr int LANE_WORDS = (LANE_BYTES + 6) / 7;
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

__global__ void __launch_bounds__(256) chain_tag(uint64_t steps, uint32_t g, unsigned long long* xr) {
  uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a[2 * U];
#pragma unroll
  for (int u = 0; u < 2 * U; ++u) a[u] = (uint32_t)(id * 2 * U + u) * 0x9E3779B9u;
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < 2 * U; ++u) a[u] = horner31(horner31(a[u], g, a[u] ^ 0x5bd1e995u), g, a[u] + 0x27d4eb2du);
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < 2 * U; ++u) acc ^= (uint64_t)a[u] << (u & 1) * 32;
  atomicXor(xr, (unsigned long long)acc);
}

// --- the FP8 GEMM, plain and fused ----------------------------------------------------------------------------------

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t take4(const uint64_t* x, int o) {   // stream bytes o .. o + 3 of the decoded payloads
  int w = o / 7, sh = o % 7;
  uint32_t v = (uint32_t)(x[w] >> (8 * sh));
  if (sh > 3) v |= (uint32_t)(x[w + 1] << (8 * (7 - sh)));
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
      const uint32_t a[4] = {take4(xw, 16 * tt), take4(xw, 16 * tt + 4), take4(xw, 16 * tt + 8), take4(xw, 16 * tt + 12)};
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

#define WC_K_SWITCH(k, F) \
  switch (k) { case 1: F(1); break; case 2: F(2); break; case 3: F(3); break; case 4: F(4); break; \
               case 5: F(5); break; case 6: F(6); break; case 7: F(7); break; case 8: F(8); break; default: return 1; }

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
    case 0: GK(0); break; case 1: GK(1); break; case 2: GK(2); break; case 3: GK(3); break; case 4: GK(4); break;
    case 5: GK(5); break; case 6: GK(6); break; case 7: GK(7); break; case 8: GK(8); break;
  }
#undef GK
#undef G
  return cudaGetLastError() ? 4 : 0;
}

}  // extern "C"
