// P2's decode with both squarings on the int8 tensor cores (fast-decode prototype; P2 v4's maps, any width
// w = 32 (32 PER + 2)): m = d(sigma_K(d(c))) - t mod p, d(z) = z^2 mod p negated when z is odd. `algo.py` is its CPU twin
// (the tiling, the fragments, the Karatsuba plan, the fold) and the tests pin it there.
//
// A CTA of two warps decodes one block at a time. The square of the operand z (little-endian bytes in a zero-gapped arena)
// is its byte-convolution coefficients c_k = sum z_i z_{k-i}: warp pi accumulates the tiles of output columns
// u = pi + 16 i + 2 j (coefficient 16 u + v) with mma.m16n8k32.u8.u8.s32, A = the Toeplitz block G_m[v][c] = z[16m + v - c]
// (funnel shifts and byte swaps of the arena, loaded once per m and reused across every tile) and B = 8 consecutive
// 32-byte chunks (one 8-byte load a lane; chunks outside the operand read as zero). FD_DEPTH > 0 squares the main part
// (all but the top 8 bytes) by that many Karatsuba levels split at whole tiles, D = |x0 - x1| computed exactly on the
// CUDA cores beforehand, every base square added into the output tiles in registers at its offsets with its sign; the
// top 8 bytes' terms are added in the fold (fdcore.cuh).
//
// The tiles become int64 limb sums (two shuffles), and the block's control warp (alternating between the two) folds them
// mod p and runs the rest of the decode.
//
// Build: nvcc -O3 -arch=sm_120 -std=c++17 -DFD_PER=16 -DFD_DEPTH=0 -Xcompiler -fPIC -shared -o libfdtc.so tcsq.cu
#include <algorithm>

#include "fdcore.cuh"

#ifndef FD_DEPTH
#define FD_DEPTH 0
#endif
#ifndef FD_MINB
#define FD_MINB 8
#endif
#define P2_BITS (32 * (32 * FD_PER + 2))
#include "../p2_v1/p2keys4.cuh"

namespace fd {

constexpr int DEPTH = FD_DEPTH;
constexpr int Q = DEPTH == 0 ? 4 * PER + 1 : 4 * PER;   // chunks the tiles square
constexpr int NT = (Q + 3) / 4;                          // output tiles a warp holds
constexpr int QB = Q >> DEPTH;                           // chunks of a base square
constexpr int NTB = (QB + 3) / 4;
__host__ __device__ constexpr int ipow3(int d) { return d == 0 ? 1 : 3 * ipow3(d - 1); }
constexpr int NOPS = ipow3(DEPTH);
constexpr int GAP = 64;
constexpr int SLOT = 32 * QB + GAP;
constexpr int ARENA = GAP + SLOT * NOPS;
constexpr int SMEM = ARENA + 8 * VWORDS;
static_assert(DEPTH == 0 || QB % 8 == 0, "Karatsuba halves are whole tiles");
static_assert(DEPTH <= 2, "the operand preparation covers depths 0 to 2");
static_assert(ARENA % 16 == 0, "the limb-sum buffer is 16-byte aligned");

// base operand b's digit at level l (1 = the outermost split): 0 low half, 1 high half, 2 = |low - high|
__host__ __device__ constexpr int digit(int b, int l) {
  for (int i = DEPTH; i > l; i--) b /= 3;
  return b % 3;
}
__host__ __device__ constexpr int noffs(int b) {
  int n = 1;
  for (int l = 1; l <= DEPTH; l++) n *= digit(b, l) == 2 ? 1 : 2;
  return n;
}
__host__ __device__ constexpr int sign(int b) {
  int s = 1;
  for (int l = 1; l <= DEPTH; l++) s *= digit(b, l) == 2 ? -1 : 1;
  return s;
}
// the k-th tile offset of base square b (algo.karatsuba_plan's order: the innermost level's choice is k's low bit)
__host__ __device__ constexpr int offset(int b, int k) {
  int o = 0;
  for (int l = DEPTH; l >= 1; l--) {
    const int t = (Q >> l) / 8;
    const int dg = digit(b, l);
    if (dg == 2) {
      o += t;
    } else {
      o += (k & 1) ? t : (dg == 0 ? 0 : 2 * t);
      k >>= 1;
    }
  }
  return o;
}

__host__ __device__ constexpr bool tile_valid(int q, int mh, int i) { return 8 * i - mh < q && 8 * i - mh + 8 > 0; }
inline int mma_count(int q) {
  int n = 0;
  for (int m = 0; m < 2 * q + 2; m++)
    for (int i = 0; i < (q + 3) / 4; i++) n += tile_valid(q, m / 2, i);
  return n;
}

__device__ __forceinline__ void mma(uint32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The tiles of the qq-chunk operand at arena byte `base`, parity pi: acc[i] += G_m B(s0 = 8 i - (m - pi) / 2), m = pi mod 2.
template <int NTILES, int QQ>
__device__ __forceinline__ void square_tiles(const uint8_t *arena, int base, uint32_t (&acc)[NTILES][4], int pi, int lane) {
  const int g = lane >> 2, kq = lane & 3;
  const uint32_t *a32 = reinterpret_cast<const uint32_t *>(arena);
#pragma unroll 1
  for (int mh = 0; mh <= QQ; mh++) {
    const int f = base + 16 * (2 * mh + pi) + g - 8 * kq - 7;
    const int wi = f >> 2, sh = (f & 3) * 8;
    const uint32_t w0 = a32[wi], w1 = a32[wi + 1], w2 = a32[wi + 2], w3 = a32[wi + 3], w4 = a32[wi + 4];
    const uint32_t a2 = bswap(__funnelshift_r(w0, w1, sh)), a0 = bswap(__funnelshift_r(w1, w2, sh)),
                   a3 = bswap(__funnelshift_r(w2, w3, sh)), a1 = bswap(__funnelshift_r(w3, w4, sh));
#pragma unroll
    for (int i = 0; i < NTILES; i++) {
      if (tile_valid(QQ, mh, i)) {
        const int s = 8 * i - mh + g;
        uint2 b = make_uint2(0, 0);
        if ((unsigned)s < (unsigned)QQ) b = *reinterpret_cast<const uint2 *>(arena + base + 32 * s + 8 * kq);
        mma(acc[i], a0, a1, a2, a3, b.x, b.y);
      }
    }
  }
}

template <int B>
__device__ __forceinline__ void kara_steps(const uint8_t *arena, uint32_t (&out)[NT][4], int pi, int lane) {
  if constexpr (B < NOPS) {
    uint32_t t[NTB][4];
#pragma unroll
    for (int i = 0; i < NTB; i++) t[i][0] = t[i][1] = t[i][2] = t[i][3] = 0;
    square_tiles<NTB, QB>(arena, GAP + SLOT * B, t, pi, lane);
    constexpr int NO = noffs(B), SG = sign(B);
#pragma unroll
    for (int k = 0; k < NO; k++) {
      const int o = offset(B, k);
#pragma unroll
      for (int i = 0; i < NTB; i++)
#pragma unroll
        for (int r = 0; r < 4; r++) out[o + i][r] += SG > 0 ? t[i][r] : 0u - t[i][r];
    }
    kara_steps<B + 1>(arena, out, pi, lane);
  }
}

// The square's coefficients as int64 limb sums in V (warp pi's columns).
__device__ __forceinline__ void square_to_v(const uint8_t *arena, int64_t *V, int pi, int lane) {
  uint32_t out[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++) out[i][0] = out[i][1] = out[i][2] = out[i][3] = 0;
  if constexpr (DEPTH == 0) {
    square_tiles<NT, Q>(arena, GAP, out, pi, lane);
  } else {
    kara_steps<0>(arena, out, pi, lane);
  }
  const int g = lane >> 2, kq = lane & 3;
  constexpr int OFF[4] = {0, 32, 8, 40};
#pragma unroll
  for (int i = 0; i < NT; i++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      long long v = (long long)(int32_t)out[i][r] << (8 * (g & 3));
      v += __shfl_xor_sync(FULL, v, 4);
      v += __shfl_xor_sync(FULL, v, 8);
      const int x = (16 * pi + 256 * i + 64 * kq + g + OFF[r]) >> 2;
      if ((g & 3) == 0 && x < VLIMBS) V[vpos(x)] = v;
    }
}

// PER limbs into the arena at byte offset `at` (16-byte aligned when PER is a multiple of 4)
__device__ __forceinline__ void put(uint8_t *arena, int at, const uint32_t (&v)[PER]) {
  if constexpr (PER % 4 == 0) {
#pragma unroll
    for (int k = 0; k < PER / 4; k++)
      reinterpret_cast<uint4 *>(arena + at)[k] = make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
  } else {
#pragma unroll
    for (int k = 0; k < PER / 2; k++) reinterpret_cast<uint2 *>(arena + at)[k] = make_uint2(v[2 * k], v[2 * k + 1]);
  }
}
// operand of `lanes` lanes starting at lane lo, into slot `slot`
__device__ __forceinline__ void put_op(uint8_t *arena, int slot, int lo, int lanes, const uint32_t (&v)[PER], int lane) {
  if (lane >= lo && lane < lo + lanes) put(arena, GAP + SLOT * slot + 4 * PER * (lane - lo), v);
}

// D = |X0 - X1| for every pair of H-lane halves (X0 the lower lanes of each 2H-lane segment): into d on the lower lanes
template <int H>
__device__ __forceinline__ void absdiff(const uint32_t (&x)[PER], uint32_t (&d)[PER], int lane) {
  uint64_t b = 0;
#pragma unroll
  for (int j = 0; j < PER; j++) {
    const uint32_t y = __shfl_down_sync(FULL, x[j], H);
    const uint64_t v = (uint64_t)x[j] - y - b;
    d[j] = (uint32_t)v;
    b = v >> 63;
  }
  uint32_t z = 0;
#pragma unroll
  for (int j = 0; j < PER; j++) z |= d[j];
  const int base = lane & ~(2 * H - 1), rel = lane - base;
  constexpr uint32_t MASK = (1u << H) - 1;
  const uint32_t Bs = (__ballot_sync(FULL, (uint32_t)b) >> base) & MASK;
  const uint32_t As = Bs | ((__ballot_sync(FULL, z == 0) >> base) & MASK);
  const uint32_t S = As + Bs;
  uint64_t bin = ((S ^ As ^ Bs) >> rel) & 1;
#pragma unroll
  for (int j = 0; j < PER; j++) { const uint64_t v = (uint64_t)d[j] - bin; d[j] = (uint32_t)v; bin = v >> 63; }
  // X0 < X1: D = -(X0 - X1) = ~diff + 1 (the ballot outside any branch: segments differ)
  const uint32_t neg = 0u - ((S >> H) & 1);
  uint32_t o = FULL;
#pragma unroll
  for (int j = 0; j < PER; j++) { d[j] ^= neg; o &= d[j]; }
  const uint32_t Os = (__ballot_sync(FULL, o == FULL) >> base) & MASK;
  const uint32_t below = (1u << rel) - 1;
  uint64_t c = neg & ((Os & below) == below);
#pragma unroll
  for (int j = 0; j < PER; j++) { c += d[j]; d[j] = (uint32_t)c; c >>= 32; }
}

// Write the squaring's operands (z, or its Karatsuba base operands) into the arena: the control warp.
__device__ __forceinline__ void prepare(uint8_t *arena, const Num &z, int lane) {
  if constexpr (DEPTH == 0) {
    put(arena, GAP + 4 * PER * lane, z.l);
    if (lane == 31) *reinterpret_cast<uint2 *>(arena + GAP + 4 * LM) = make_uint2(z.e0, z.e1);
  } else if constexpr (DEPTH == 1) {
    uint32_t d[PER];
    absdiff<16>(z.l, d, lane);
    put_op(arena, 0, 0, 16, z.l, lane);
    put_op(arena, 1, 16, 16, z.l, lane);
    put_op(arena, 2, 0, 16, d, lane);
  } else {
    uint32_t d1[PER], d2[PER], dd[PER];
    absdiff<16>(z.l, d1, lane);   // D1 = |z0 - z1| on lanes 0-15
    absdiff<8>(z.l, d2, lane);    // |z00 - z01| on lanes 0-7, |z10 - z11| on lanes 16-23
    absdiff<8>(d1, dd, lane);     // |D10 - D11| on lanes 0-7
    put_op(arena, 0, 0, 8, z.l, lane);
    put_op(arena, 1, 8, 8, z.l, lane);
    put_op(arena, 2, 0, 8, d2, lane);
    put_op(arena, 3, 16, 8, z.l, lane);
    put_op(arena, 4, 24, 8, z.l, lane);
    put_op(arena, 5, 16, 8, d2, lane);
    put_op(arena, 6, 0, 8, d1, lane);
    put_op(arena, 7, 8, 8, d1, lane);
    put_op(arena, 8, 0, 8, dd, lane);
  }
}

}  // namespace fd

using namespace fd;

// mode 0: m = D(c); mode 1: m = d(c) (one squaring, for the gate and the squaring's own rate)
__global__ void __launch_bounds__(64, FD_MINB) decode_kernel(const uint32_t *__restrict__ cin, uint32_t *__restrict__ mout,
                                                             const uint32_t *__restrict__ kin, const uint32_t *__restrict__ tin,
                                                             uint32_t n, uint32_t c0, uint32_t *__restrict__ flags, int mode) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t *arena = smem;
  int64_t *V = reinterpret_cast<int64_t *>(smem + ARENA);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < SMEM / 16; i += blockDim.x) reinterpret_cast<uint4 *>(smem)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  Num z, out;
  bool odd = false;
  uint32_t it = 0;
  for (uint32_t j = blockIdx.x; j < n; j += gridDim.x, it++) {
    const int cw = it & 1;   // the control warp of this block
    if (warp == cw) {
      load_row(z, cin + (size_t)j * L, lane);
      odd = __shfl_sync(FULL, z.l[0], 0) & 1;
      prepare(arena, z, lane);
    }
    __syncthreads();
    square_to_v(arena, V, warp, lane);
    __syncthreads();
    if (warp == cw) {
      fold<DEPTH != 0>(V, out, z, odd, c0, lane);   // out = d(c)
      if (mode == 0) {
        sigma(z, out, kin + (size_t)j * L, c0, lane);
        odd = __shfl_sync(FULL, z.l[0], 0) & 1;
        prepare(arena, z, lane);
      }
    }
    if (mode == 0) {
      __syncthreads();
      square_to_v(arena, V, warp, lane);
      __syncthreads();
      if (warp == cw) {
        fold<DEPTH != 0>(V, out, z, odd, c0, lane);   // a = d(sigma_K(d(c)))
        sub_t(out, tin + (size_t)j * L, c0, flags, lane);
      }
    }
    if (warp == cw) store_row(mout + (size_t)j * L, out, lane);
    __syncthreads();
  }
}

static int occupancy() {
  int occ = 0;
  cudaFuncSetAttribute(decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, decode_kernel, 64, SMEM);
  return occ;
}

static int grid_for(uint32_t n) {
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  return (int)std::max<uint64_t>(1, std::min<uint64_t>(n, (uint64_t)std::max(occupancy(), 1) * std::max(sms, 1)));
}

// [PER, DEPTH, limbs, bytes, Q, NT, QB, NOPS, smem bytes, MMAs per squaring, CTAs per SM, registers]
extern "C" int fd_config(int *out) {
  cudaFuncAttributes fa{};
  cudaFuncGetAttributes(&fa, decode_kernel);
  const int v[12] = {PER, DEPTH, L, NB, Q, NT, QB, NOPS, SMEM, NOPS * mma_count(QB), occupancy(), fa.numRegs};
  for (int i = 0; i < 12; i++) out[i] = v[i];
  return (int)cudaGetLastError();
}

extern "C" int fd_decode(const void *c, void *m, const void *k, const void *t, uint32_t n, uint32_t c0, void *flags,
                         int mode, void *stream) {
  if (n == 0) return 0;
  decode_kernel<<<grid_for(n), 64, SMEM, (cudaStream_t)stream>>>((const uint32_t *)c, (uint32_t *)m, (const uint32_t *)k,
                                                                 (const uint32_t *)t, n, c0, (uint32_t *)flags, mode);
  return (int)cudaGetLastError();
}

extern "C" int fd_keys_version() { return 4; }

extern "C" int fd_keys(const uint8_t *prefix, int plen, uint64_t first, uint32_t count, uint64_t per_segment, void *k, void *t,
                       void *stream) {
  p2keys4::Mid mid;
  if (!p2keys4::midstate(prefix, plen, mid)) return -1;
  if (count == 0) return 0;
  const uint64_t per = p2keys4::threads_per_purpose(count);
  p2keys4::keys_kernel<<<(uint32_t)((2 * per + P2K4_TPB - 1) / P2K4_TPB), P2K4_TPB, 0, (cudaStream_t)stream>>>(
      mid, first, count, per_segment, per, (uint64_t *)k, (uint64_t *)t);
  return (int)cudaGetLastError();
}
