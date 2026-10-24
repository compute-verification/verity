// Decode at use inside the GEMM's operand load: y = x W^T + b for a linear whose bf16 weight W [N][K] the engine's arena
// holds encoded in place in `../word_codec/kv.cu`'s codec (this file includes it, so its build is kv.so's plus these
// launchers, at the same KV_FIELD, KV_MASK and KV_FOLD6), each word at its codec index: the weight's first word index
// plus the word's offset in W. `fused_bench.py` gates and times it; `engine.py`'s Hooks take it with --fused.
//
// A CTA owns NT rows of W and a range of K's steps. In each step its 128 threads load a word each (KT = 128 / NT words of
// a row), decode it with kv_decode's arithmetic (kv::decode_word_t, its tweak stepped as decode_kernel steps it, by
// tau times the words a step) and store it into a shared tile, while x's rows for the step come in by cp.async; then its
// four warps fold the tile into FP32 accumulators with bf16 mma.m16n8k16, W's rows the MMA's m and x's rows its n. Shared
// rows are XOR-swizzled by 16-byte chunk (chunk c of row r at c ^ (r & 7)), so ldmatrix and the stores don't conflict.
// Each CTA leaves its sums in part[split][M][N]; kv_gemm_reduce adds them in split order, adds the bias and rounds to
// bf16 once, so a call's bits don't depend on how its CTAs ran. No decoded byte goes to global memory, except through
// kv_gemm_dump, which writes the tiles out as the MMA reads them (the gate).
#include "../word_codec/kv.cu"

#include <cuda_bf16.h>

namespace kvg {

using kv::u32;
using kv::u64;
constexpr int NL = kv::NL;
constexpr int TH = 128;
constexpr int WBYTES = NL * 4;
constexpr int WE = WBYTES / 2;
constexpr int CH = WBYTES / 16;
constexpr int MAX_M = 64;
#ifndef KVG_AD
#define KVG_AD 1
#endif
// sqr<true>: FOLD 6's square in the register-lean order (kv.cu's sqr_fold6_ad); the same value, so the same bytes.
constexpr bool AD = KVG_AD;

template <int NT, int MT>
struct Shape {
  static constexpr int WPR = TH / NT;   // words of a row a step
  static constexpr int KT = WPR * WE;   // a step's k
  static constexpr int ROWB = KT * 2;   // a tile row's bytes
  static constexpr int RCH = ROWB / 16;
  static constexpr int WM = NT / 16;    // warps along W's rows
  static constexpr int WN = 4 / WM;     // and along x's
  static constexpr int NJ = MT / 8 / WN;
  static constexpr int SMEM = (NT + MT) * ROWB;
  static_assert(TH % NT == 0 && WM * WN == 4 && NJ >= 1 && MT % (8 * WN) == 0 && RCH >= 8, "a CTA's tiles");
};

__device__ __forceinline__ int swz(int r, int c) { return (c ^ (r & 7)) * 16; }

__device__ __forceinline__ void ldsm2(unsigned a, u32 &r0, u32 &r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
}

__device__ __forceinline__ void prefetch_l2(const void *p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

template <int NT, int MT, bool DUMP>
__global__ void __launch_bounds__(TH, 2) gemm_kernel(const u32 *__restrict__ w, u64 base, const uint16_t *__restrict__ x,
                                                     float *__restrict__ part, uint16_t *__restrict__ dump, int M,
                                                     int N, int K, int k, int per_split, int k_steps) {
  using S = Shape<NT, MT>;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ u32 td[NL];
  uint8_t *ws = smem, *xs = smem + NT * S::ROWB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n0 = blockIdx.x * NT, split = blockIdx.y;
  const int s0 = split * per_split, s1 = min(k_steps, s0 + per_split);
  const int r = tid / S::WPR, kw = tid % S::WPR;
  if (tid == 0) {
    u32 s[NL];
    kv::tweak((u64)S::WPR, s);
#pragma unroll
    for (int l = 0; l < NL; ++l) td[l] = s[l];
  }
  for (int c = M * S::RCH + tid; c < MT * S::RCH; c += TH)
    *reinterpret_cast<uint4 *>(xs + c / S::RCH * S::ROWB + c % S::RCH * 16) = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const u64 at = (u64)(n0 + r) * (u64)(K / WE) + (u64)s0 * S::WPR + kw;
  const u32 *wp = w + at * NL;
  u32 t[NL];
  kv::tweak(base + at, t);
  const int wm = warp % S::WM, wn = warp / S::WM;
  float acc[S::NJ][4];
#pragma unroll
  for (int j = 0; j < S::NJ; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  for (int st = s0; st < s1; ++st) {
    const int k0 = st * S::KT;
    if (!DUMP)
      for (int c = tid; c < M * S::RCH; c += TH) {
        const int m = c / S::RCH, ch = c % S::RCH;
        kv::copy16(xs + m * S::ROWB + swz(m, ch), x + (size_t)m * K + k0 + ch * 8);
      }
    asm volatile("cp.async.commit_group;");
    u32 v[NL];
    kv::load(wp, v);
    if (st + 1 < s1) prefetch_l2(wp + S::WPR * NL);
    if (st != s0) kv::tweak_step(t, td);
    kv::decode_word_t<AD>(v, t, k);
    uint8_t *row = ws + r * S::ROWB;
#pragma unroll
    for (int j = 0; j < CH; ++j)
      *reinterpret_cast<uint4 *>(row + swz(r, kw * CH + j)) = make_uint4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if constexpr (DUMP) {
      for (int c = tid; c < NT * S::RCH; c += TH) {
        const int rr = c / S::RCH, ch = c % S::RCH;
        *reinterpret_cast<uint4 *>(dump + (size_t)(n0 + rr) * K + k0 + ch * 8) =
            *reinterpret_cast<const uint4 *>(ws + rr * S::ROWB + swz(rr, ch));
      }
    } else {
      const unsigned wsa = kv::smem_u32(ws), xsa = kv::smem_u32(xs);
      const int ar = wm * 16 + (lane & 15);
#pragma unroll
      for (int ks = 0; ks < S::KT / 16; ++ks) {
        u32 a0, a1, a2, a3;
        kv::ldsm4(wsa + ar * S::ROWB + swz(ar, 2 * ks + (lane >> 4)), a0, a1, a2, a3);
#pragma unroll
        for (int j = 0; j + 1 < S::NJ; j += 2) {
          const int br = (wn * S::NJ + j) * 8 + (lane >> 4) * 8 + (lane & 7);
          u32 b0, b1, b2, b3;
          kv::ldsm4(xsa + br * S::ROWB + swz(br, 2 * ks + ((lane >> 3) & 1)), b0, b1, b2, b3);
          kv::mma16816(acc[j], a0, a1, a2, a3, b0, b1);
          kv::mma16816(acc[j + 1], a0, a1, a2, a3, b2, b3);
        }
        if constexpr (S::NJ % 2) {
          const int br = (wn * S::NJ + S::NJ - 1) * 8 + (lane & 7);
          u32 b0, b1;
          ldsm2(xsa + br * S::ROWB + swz(br, 2 * ks + ((lane >> 3) & 1)), b0, b1);
          kv::mma16816(acc[S::NJ - 1], a0, a1, a2, a3, b0, b1);
        }
      }
    }
    __syncthreads();
    wp += S::WPR * NL;
  }
  if constexpr (!DUMP) {
    const int g = lane >> 2, t4 = lane & 3, n = n0 + wm * 16 + g;
    float *p = part + (size_t)split * M * N;
#pragma unroll
    for (int j = 0; j < S::NJ; ++j) {
      const int m = (wn * S::NJ + j) * 8 + 2 * t4;
      if (m < M) {
        p[(size_t)m * N + n] = acc[j][0];
        p[(size_t)m * N + n + 8] = acc[j][2];
      }
      if (m + 1 < M) {
        p[(size_t)(m + 1) * N + n] = acc[j][1];
        p[(size_t)(m + 1) * N + n + 8] = acc[j][3];
      }
    }
  }
}

__global__ void reduce_kernel(const float *__restrict__ part, const uint16_t *__restrict__ bias, uint16_t *__restrict__ y,
                              int M, int N, int splits) {
  const size_t mn = (size_t)M * N;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < mn; i += (size_t)gridDim.x * blockDim.x) {
    float s = part[i];
    for (int q = 1; q < splits; ++q) s += part[q * mn + i];
    if (bias) s += __uint_as_float((u32)bias[i % N] << 16);
    y[i] = __bfloat16_as_ushort(__float2bfloat16_rn(s));
  }
}

template <int NT, int MT, bool DUMP>
int occupancy() {
  int dev, sms, per = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  constexpr int smem = Shape<NT, MT>::SMEM;
  cudaFuncSetAttribute(gemm_kernel<NT, MT, DUMP>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, gemm_kernel<NT, MT, DUMP>, TH, smem);
  return per * sms;
}

// The fewest splits of K's steps within 5% of the shortest schedule in waves of `slots` resident CTAs (each CTA its
// split's steps); more splits than that only add partial sums.
inline int pick_splits(int tiles, int steps, int slots, int most) {
  auto span = [&](int s) { return (long)((tiles * s + slots - 1) / slots) * ((steps + s - 1) / s); };
  long best = span(1);
  for (int s = 2; s <= most && s <= steps; ++s) best = std::min(best, span(s));
  for (int s = 1; s <= most && s <= steps; ++s)
    if (span(s) * 100 <= best * 105) return s;
  return 1;
}

template <int NT, int MT, bool DUMP>
int launch(const void *w, uint64_t base, const void *x, void *part, void *dump, int M, int N, int K, int k, int splits,
           cudaStream_t s) {
  using S = Shape<NT, MT>;
  if (N % NT || K % S::KT || M > MT || (!DUMP && M < 1)) return -2;
  const int steps = K / S::KT, tiles = N / NT;
  if (splits <= 0) splits = pick_splits(tiles, steps, occupancy<NT, MT, DUMP>(), 16);
  const int per = (steps + splits - 1) / splits;
  splits = (steps + per - 1) / per;
  gemm_kernel<NT, MT, DUMP><<<dim3(tiles, splits), TH, S::SMEM, s>>>(
      (const u32 *)w, base, (const uint16_t *)x, (float *)part, (uint16_t *)dump, M, N, K, k, per, steps);
  const int rc = (int)cudaPeekAtLastError();
  return rc ? -1000 - rc : splits;
}

// NT 64 (KT 128) for every batch tile; NT 32 (KT 256, half the accumulators a thread) from MT 16.
template <bool DUMP>
int dispatch(int nt, const void *w, uint64_t base, const void *x, void *part, void *dump, int M, int N, int K, int k,
             int splits, cudaStream_t s) {
  const int mt = M <= 8 ? 8 : M <= 16 ? 16 : M <= 32 ? 32 : 64;
  if (nt == 0) nt = 64;
  if (nt == 64) {
    switch (mt) {
      case 8: return launch<64, 8, DUMP>(w, base, x, part, dump, M, N, K, k, splits, s);
      case 16: return launch<64, 16, DUMP>(w, base, x, part, dump, M, N, K, k, splits, s);
      case 32: return launch<64, 32, DUMP>(w, base, x, part, dump, M, N, K, k, splits, s);
      default: return launch<64, 64, DUMP>(w, base, x, part, dump, M, N, K, k, splits, s);
    }
  }
  if (nt == 32) {
    switch (mt) {
      case 8:
      case 16: return launch<32, 16, DUMP>(w, base, x, part, dump, M, N, K, k, splits, s);
      case 32: return launch<32, 32, DUMP>(w, base, x, part, dump, M, N, K, k, splits, s);
      default: return launch<32, 64, DUMP>(w, base, x, part, dump, M, N, K, k, splits, s);
    }
  }
  return -3;
}

}  // namespace kvg

extern "C" {

int kv_gemm_max_m() { return kvg::MAX_M; }

int kv_gemm_fold6() { return kv::FOLD6; }

// The splits kv_gemm picks for this shape (splits 0), without launching.
int kv_gemm_splits(int M, int N, int K, int nt) {
  if (M < 1 || M > kvg::MAX_M) return -1;
  const int mt = M <= 8 ? 8 : M <= 16 ? 16 : M <= 32 ? 32 : 64;
  if (nt == 0) nt = 64;
  auto pick = [&](int kt, int slots) {
    if (N % nt || K % kt) return -2;
    return kvg::pick_splits(N / nt, K / kt, slots, 16);
  };
  if (nt == 64) {
    switch (mt) {
      case 8: return pick(128, kvg::occupancy<64, 8, false>());
      case 16: return pick(128, kvg::occupancy<64, 16, false>());
      case 32: return pick(128, kvg::occupancy<64, 32, false>());
      default: return pick(128, kvg::occupancy<64, 64, false>());
    }
  }
  if (nt == 32) {
    switch (mt) {
      case 8:
      case 16: return pick(256, kvg::occupancy<32, 16, false>());
      case 32: return pick(256, kvg::occupancy<32, 32, false>());
      default: return pick(256, kvg::occupancy<32, 64, false>());
    }
  }
  return -3;
}

// The resident CTAs an SM of kv_gemm's kernel at this batch and nt.
int kv_gemm_occupancy(int M, int nt) {
  int dev, sms;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int mt = M <= 8 ? 8 : M <= 16 ? 16 : M <= 32 ? 32 : 64;
  if (nt == 32) {
    if (mt <= 16) return kvg::occupancy<32, 16, false>() / sms;
    return mt == 32 ? kvg::occupancy<32, 32, false>() / sms : kvg::occupancy<32, 64, false>() / sms;
  }
  switch (mt) {
    case 8: return kvg::occupancy<64, 8, false>() / sms;
    case 16: return kvg::occupancy<64, 16, false>() / sms;
    case 32: return kvg::occupancy<64, 32, false>() / sms;
    default: return kvg::occupancy<64, 64, false>() / sms;
  }
}

// y [M][N] bf16 = x [M][K] bf16 (contiguous, 16-byte aligned) times W^T, W's N K bf16 encoded at w from codec index
// base, plus bias [N] (or none). part holds splits M N floats; part_bytes is its size. nt 0, 32 or 64 (0: 64); splits 0
// picks them (kv_gemm_splits). Returns the splits used, or a negative error (-1000 - a CUDA error).
int kv_gemm(const void *w, uint64_t base, const void *x, void *y, const void *bias, void *part, uint64_t part_bytes, int M,
            int N, int K, int k, int nt, int splits, cudaStream_t s) {
  if (k < 1 || k > kv::KMAX || M < 1 || M > kvg::MAX_M || ((uintptr_t)x & 15) || ((uintptr_t)w & 15)) return -1;
  if (splits <= 0) splits = kv_gemm_splits(M, N, K, nt);
  if (splits <= 0) return splits ? splits : -2;
  if ((uint64_t)splits * M * N * 4 > part_bytes) return -4;
  const int got = kvg::dispatch<false>(nt, w, base, x, part, nullptr, M, N, K, k, splits, s);
  if (got <= 0) return got;
  kvg::reduce_kernel<<<std::min<size_t>(((size_t)M * N + 255) / 256, 4096), 256, 0, s>>>(
      (const float *)part, (const uint16_t *)bias, (uint16_t *)y, M, N, got);
  const int rc = (int)cudaPeekAtLastError();
  return rc ? -1000 - rc : got;
}

// W decoded through kv_gemm's tiles at the batch tile of M rows (the shared rows its MMAs read) into out [N][K] bf16.
int kv_gemm_dump(const void *w, uint64_t base, void *out, int M, int N, int K, int k, int nt, int splits, cudaStream_t s) {
  if (k < 1 || k > kv::KMAX || M < 1 || M > kvg::MAX_M || ((uintptr_t)w & 15)) return -1;
  return kvg::dispatch<true>(nt, w, base, nullptr, nullptr, out, M, N, K, k, splits, s);
}

}  // extern "C"
