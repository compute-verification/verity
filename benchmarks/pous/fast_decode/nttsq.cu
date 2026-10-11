// P2's decode with both squarings by a number-theoretic transform (fast-decode prototype, P2 v4's width only):
// m = d(sigma_K(d(c))) - t mod p, each square by a 2048-point cyclic NTT over Goldilocks (P = 2^64 - 2^32 + 1) on the
// operand's 1028 16-bit limbs, on the CUDA cores. `algo.ntt_square` is its CPU twin.
//
// A CTA of 256 threads decodes one block at a time: warp 0 writes the operand's 16-bit limbs (little-endian) to shared
// memory, every thread places them in bit-reversed order, the forward transform runs decimation in time (bit-reversed in,
// natural out), the pointwise square carries N^-1, the inverse runs decimation in frequency (natural in, bit-reversed out),
// radix 2 with a barrier per stage. The cyclic convolution wraps coefficients 2048 .. 2054 onto 0 .. 6 (every coefficient
// is below 1028 2^32 < P, so the wrap is a plain sum): those seven are recomputed directly and taken out. The limb sums
// V_x = c_{2x} + c_{2x+1} 2^16 go to `fd::fold` (fdcore.cuh), which warp 0 runs with the rest of the decode.
//
// Build: nvcc -O3 -arch=sm_120 -std=c++17 -DFD_PER=16 -Xcompiler -fPIC -shared -o libfdntt.so nttsq.cu
#include <algorithm>

#include "fdcore.cuh"

namespace fd {

static_assert(PER == 16, "one 2048-point transform covers P2 v4's width");
constexpr int N = 2048, LOGN = 11, H16 = 2 * L, NWRAP = 2 * H16 - 1 - N;   // 1028 limbs, coefficients 2048 .. 2054
constexpr int TPB = 256;
constexpr uint64_t P = 0xFFFFFFFF00000001ull, EPS = 0xFFFFFFFFull;
static_assert(NWRAP > 0 && NWRAP <= 32 && H16 <= N, "the wrap is a few coefficients");
// shared memory: X, V, the wrap, the operand's limbs (16-byte aligned for warp 0's stores)
constexpr int VOFF = 8 * N, WOFF = (VOFF + 8 * VWORDS + 15) / 16 * 16, SOFF = WOFF + 64, SMEM = SOFF + 4 * L;

__device__ __forceinline__ uint64_t addm(uint64_t a, uint64_t b) {
  const uint64_t s = a + b;
  return (s < a || s >= P) ? s - P : s;
}
__device__ __forceinline__ uint64_t subm(uint64_t a, uint64_t b) { return a >= b ? a - b : a - b + P; }
// canonical a * b mod P: lo + 2^64 hl + 2^96 hh = lo - hh + hl (2^32 - 1)
__device__ __forceinline__ uint64_t mulm(uint64_t a, uint64_t b) {
  const uint64_t lo = a * b, hi = __umul64hi(a, b);
  const uint64_t hh = hi >> 32, hl = hi & EPS;
  uint64_t t0 = lo - hh;
  if (lo < hh) t0 -= EPS;
  const uint64_t t1 = hl * EPS;
  uint64_t t2 = t0 + t1;
  if (t2 < t1) t2 += EPS;
  return t2 >= P ? t2 - P : t2;
}

__device__ __forceinline__ int brev(int i) { return (int)(__brev((uint32_t)i) >> (32 - LOGN)); }

// tw[j] = w^j, itw[j] = w^-j (j < N / 2), w = 7^((P - 1) / N); ninv = N^-1
struct Twiddles {
  const uint64_t *tw, *itw;
  uint64_t ninv;
};

__device__ __forceinline__ void forward(uint64_t *X, const uint64_t *__restrict__ tw, int tid) {
#pragma unroll 1
  for (int s = 0; s < LOGN; s++) {
    const int span = 1 << s;
#pragma unroll
    for (int r = 0; r < N / 2 / TPB; r++) {
      const int b = tid + TPB * r, j = b & (span - 1), i0 = ((b >> s) << (s + 1)) + j, i1 = i0 + span;
      const uint64_t u = X[i0], v = s == 0 ? X[i1] : mulm(X[i1], __ldg(tw + (j << (LOGN - 1 - s))));
      X[i0] = addm(u, v);
      X[i1] = subm(u, v);
    }
    __syncthreads();
  }
}
__device__ __forceinline__ void inverse(uint64_t *X, const uint64_t *__restrict__ itw, int tid) {
#pragma unroll 1
  for (int s = LOGN - 1; s >= 0; s--) {
    const int span = 1 << s;
#pragma unroll
    for (int r = 0; r < N / 2 / TPB; r++) {
      const int b = tid + TPB * r, j = b & (span - 1), i0 = ((b >> s) << (s + 1)) + j, i1 = i0 + span;
      const uint64_t u = X[i0], v = X[i1];
      X[i0] = addm(u, v);
      X[i1] = s == 0 ? subm(u, v) : mulm(subm(u, v), __ldg(itw + (j << (LOGN - 1 - s))));
    }
    __syncthreads();
  }
}

// V = the limb sums of z^2, z's little-endian 16-bit limbs in S (every thread; ends after a barrier)
__device__ __forceinline__ void ntt_square(uint64_t *X, int64_t *V, uint64_t *wrap, const uint16_t *S, const Twiddles &tw,
                                           int tid) {
#pragma unroll
  for (int r = 0; r < N / TPB; r++) {
    const int i = tid + TPB * r;
    X[brev(i)] = i < H16 ? S[i] : 0;
  }
  if (tid < NWRAP) {
    const int k = N + tid;
    uint64_t acc = 0;
    for (int i = k - H16 + 1; i < H16; i++) acc += (uint64_t)S[i] * S[k - i];
    wrap[tid] = acc;
  }
  __syncthreads();
  forward(X, tw.tw, tid);
#pragma unroll
  for (int r = 0; r < N / TPB; r++) {
    const int i = tid + TPB * r;
    const uint64_t x = X[i];
    X[i] = mulm(mulm(x, x), tw.ninv);
  }
  __syncthreads();
  inverse(X, tw.itw, tid);
  auto coeff = [&](int k) -> uint64_t {
    if (k >= N) return k - N < NWRAP ? wrap[k - N] : 0;
    const uint64_t c = X[brev(k)];
    return k < NWRAP ? c - wrap[k] : c;
  };
  for (int x = tid; x < VLIMBS; x += TPB) V[vpos(x)] = (int64_t)(coeff(2 * x) + (coeff(2 * x + 1) << 16));
  __syncthreads();
}

// z as little-endian 32-bit limbs into S (warp 0)
__device__ __forceinline__ void stage(uint32_t *S, const Num &z, int lane) {
#pragma unroll
  for (int j = 0; j < PER; j += 4)
    *reinterpret_cast<uint4 *>(S + PER * lane + j) = make_uint4(z.l[j], z.l[j + 1], z.l[j + 2], z.l[j + 3]);
  if (lane == 31) *reinterpret_cast<uint2 *>(S + LM) = make_uint2(z.e0, z.e1);
}

}  // namespace fd

using namespace fd;

// mode 0: m = D(c); mode 1: m = d(c)
__global__ void __launch_bounds__(TPB) decode_kernel(const uint32_t *__restrict__ cin, uint32_t *__restrict__ mout,
                                                     const uint32_t *__restrict__ kin, const uint32_t *__restrict__ tin,
                                                     uint32_t n, uint32_t c0, uint32_t *__restrict__ flags, int mode,
                                                     Twiddles tw) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint64_t *X = reinterpret_cast<uint64_t *>(smem);
  int64_t *V = reinterpret_cast<int64_t *>(smem + VOFF);
  uint64_t *wrap = reinterpret_cast<uint64_t *>(smem + WOFF);
  uint32_t *S = reinterpret_cast<uint32_t *>(smem + SOFF);
  const uint16_t *S16 = reinterpret_cast<const uint16_t *>(S);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  Num z, out;
  bool odd = false;
  for (uint32_t j = blockIdx.x; j < n; j += gridDim.x) {
    if (warp == 0) {
      load_row(z, cin + (size_t)j * L, lane);
      odd = __shfl_sync(FULL, z.l[0], 0) & 1;
      stage(S, z, lane);
    }
    __syncthreads();
    ntt_square(X, V, wrap, S16, tw, tid);
    if (warp == 0) {
      fold<false>(V, out, z, odd, c0, lane);   // out = d(c)
      if (mode == 0) {
        sigma(z, out, kin + (size_t)j * L, c0, lane);
        odd = __shfl_sync(FULL, z.l[0], 0) & 1;
        stage(S, z, lane);
      }
    }
    __syncthreads();
    if (mode == 0) {
      ntt_square(X, V, wrap, S16, tw, tid);
      if (warp == 0) {
        fold<false>(V, out, z, odd, c0, lane);   // a = d(sigma_K(d(c)))
        sub_t(out, tin + (size_t)j * L, c0, flags, lane);
      }
    }
    if (warp == 0) store_row(mout + (size_t)j * L, out, lane);
    __syncthreads();
  }
}

static uint64_t host_mul(uint64_t a, uint64_t b) { return (uint64_t)((unsigned __int128)a * b % P); }
static uint64_t host_pow(uint64_t a, uint64_t e) {
  uint64_t r = 1;
  for (; e; e >>= 1, a = host_mul(a, a))
    if (e & 1) r = host_mul(r, a);
  return r;
}

static Twiddles twiddles() {
  static uint64_t *dev = nullptr;
  static Twiddles t{};
  if (!dev) {
    uint64_t host[N];
    const uint64_t w = host_pow(7, (P - 1) / N), wi = host_pow(w, P - 2);
    uint64_t a = 1, b = 1;
    for (int j = 0; j < N / 2; j++, a = host_mul(a, w), b = host_mul(b, wi)) host[j] = a, host[N / 2 + j] = b;
    cudaMalloc(&dev, sizeof host);
    cudaMemcpy(dev, host, sizeof host, cudaMemcpyHostToDevice);
    t = Twiddles{dev, dev + N / 2, host_pow(N, P - 2)};
  }
  return t;
}

static int occupancy() {
  int occ = 0;
  cudaFuncSetAttribute(decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, decode_kernel, TPB, SMEM);
  return occ;
}

// [PER, DEPTH = -1 (the NTT), limbs, bytes, N, 0, 0, 0, smem bytes, 0, CTAs per SM, registers]
extern "C" int fd_config(int *out) {
  cudaFuncAttributes fa{};
  cudaFuncGetAttributes(&fa, decode_kernel);
  const int v[12] = {PER, -1, L, NB, N, 0, 0, 0, SMEM, 0, occupancy(), fa.numRegs};
  for (int i = 0; i < 12; i++) out[i] = v[i];
  return (int)cudaGetLastError();
}

extern "C" int fd_decode(const void *c, void *m, const void *k, const void *t, uint32_t n, uint32_t c0, void *flags,
                         int mode, void *stream) {
  if (n == 0) return 0;
  const Twiddles tw = twiddles();
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int grid = (int)std::max<uint64_t>(1, std::min<uint64_t>(n, (uint64_t)std::max(occupancy(), 1) * std::max(sms, 1)));
  decode_kernel<<<grid, TPB, SMEM, (cudaStream_t)stream>>>((const uint32_t *)c, (uint32_t *)m, (const uint32_t *)k,
                                                           (const uint32_t *)t, n, c0, (uint32_t *)flags, mode, tw);
  return (int)cudaGetLastError();
}
