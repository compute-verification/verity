// Dither v3.1's credited static INT8 matmul on the GPU (v3.2 is verity/core/protocols/pouw/dither/PROTOCOL.md), the
// prover's side of one call, bit for bit v31.py's reference (its docstring has the semantics, the coin streams and the
// layout; this file follows it):
//
//   coins    d_A, d_B (bits), U, V (uniform on -80..80) and the noise tiles' permutations, from the challenger's five
//            512-bit seeds in SHA-512 counter mode (a stub: v31.py's StubChallenger issues the seeds)
//   form     X = wrap8(A + d_A + V) and W' = wrap8(B + d_B + U pi_tau) with their wrap lists (k << 1 | negative, a row
//            each), X^T and B~^T = (B + d_B + U pi)^T for the wrap correction, and the peel's tail rows
//            P_A = [Z_0 | -V | Z_1 .. Z_{dz-1} | V x (dy - 1)], P_B = [-U | Y_0 | U x (dz - 1) | Y_1 .. Y_{dy-1}] (Z_X and
//            Y_W in balanced base-128 digits, the last digit a full int8)
//   wrap     dO = w_X B~^T + X w_W^T, sparse, on the CUDA cores (a 128 x 128 tile a CTA); or (V31_WRAP 5) X w_W^T dense
//            on the tensor cores (the plain GEMM on X and w_W as int8), then w_X B~^T, sparse
//   gemm     X W'^T on the tensor cores (mma.sync s8, a 3-stage TMA ring, 128 x 128 tiles of 8 warps); after each segment
//            of L = t L' inner indices every thread stores its accumulators' low 16 bits (T_s) to the prefix-sum buffer; then
//            P_A P_B^T's first two terms (-Z_0 U^T - V Y_0^T) as more main stages, the tail stages subtracting the rest
//            stage by stage at their digit's weight (128^d), and the epilogue adds 256 dO:
//            O = T_S + 256 dO - Z_X U^T - V Y_W^T = (A + d_A)(B + d_B)^T mod 2^32
//   commit   commit_gpu's small path (CTAs that fit beside the GEMM's) over the call's rows: hm96-sha512/row/v2 rows of ROW
//            bytes in one frame-v3-sha512 tree, salts prf_salt(Ks, pos); its salt and leaf phases in at most V31_CCTA
//            CTAs an SM (salt_persist, leaf_persist), so a GEMM CTA finds room beside them
//
// Modes (DIR from `v31.py gpu-prep`, PDIR from `v31.py params`):
//   v31 gate DIR M N K LP T ROW CHEAT     one call on DIR's operands and seeds into poisoned buffers, dumped for
//                                         `v31.py gpu-check` (words, O, levels, X, Wp), then the no-write control's
//                                         dump (DIR/control); CHEAT: none | skip:TAU | forge:S0:FRAC | forgeO:FRAC
//   v31 bench REPS OUT PDIR LP T ROW name:M:N:K...
//                                         per shape: synthetic operands, one checked call with two sampled tiles dumped
//                                         to OUT/samples/NAME (`v31.py gpu-sample`), then gpu_kit's interleaved timing of
//                                         the arms (lt: cuBLASLt's int8 GEMM, the baseline; plain: this file's mainloop
//                                         alone; coins, form, wrap, gemm, commit; call: one call serial; ovl: calls with
//                                         each commit beside the next call's compute; salt, leaf, leaf_nomem: the
//                                         commit's parts; cutlass: CUTLASS's fastest int8 GEMM, the second baseline,
//                                         under -DV31_CUTLASS), one JSON line a shape
//   v31 once PDIR LP T ROW name:M:N:K     one shape's call twice, nothing timed or checked (under a profiler)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1
//        [-DV31_CUTLASS cutlass_2x.o] v31.cu -lcuda -lcublasLt (run.sh builds the harness's cutlass_2x.cu)
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "../commit_gpu/commit_gpu.cuh"
#include "../gpu_kit/bench.cuh"
#ifdef V31_CUTLASS
#include "../harness/native/cutlass_api.h"
#else
struct phc_prepared {   // no CUTLASS: Cut never finds a kernel
  int launch(cudaStream_t) { return 1; }
};
#endif
#ifdef V31_WG
#include "wg_proto.cuh"
#endif

#define CL(x)                                                                                  \
  do {                                                                                         \
    cublasStatus_t s_ = (x);                                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                         \
      fprintf(stderr, "%s:%d %s: cublas %d\n", __FILE__, __LINE__, #x, int(s_));              \
      exit(3);                                                                                 \
    }                                                                                          \
  } while (0)

constexpr int TILE = 128, BK = 128, NOISE = 80;

__host__ __device__ constexpr uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}

// ---------------------------------------------------------------- the coins
// prg(seed, tag, idx) = SHA-512(seed || tag (8 bytes, zero padded) || be64(idx)): one chunk of 80 bytes
enum { S_DA = 0, S_DB = 1, S_U = 2, S_V = 3, S_P = 4, S_PAD = 5 };
struct Seeds { u64 w[6][8]; };   // dA, dB, U, V, P, pad (the filler's): each seed's 64 bytes as big-endian words
HD u64 tag2(char a, char b) { return (u64)(uint8_t)a << 56 | (u64)(uint8_t)b << 48; }
HD u64 tag_of(int s) {
  return s == S_DA ? tag2('d', 'A') : s == S_DB ? tag2('d', 'B') : s == S_U ? tag2('U', 0) : s == S_V ? tag2('V', 0)
       : s == S_P ? tag2('P', 0) : tag2('p', 'a') | (u64)'d' << 40;
}

DEV void prg(const u64* seed, u64 tag, u64 idx, u64* st) {
  u64 w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = seed[i];
  w[8] = tag; w[9] = idx; w[10] = 0x8000000000000000ull;
  w[11] = w[12] = w[13] = w[14] = 0;
  w[15] = 80 * 8;
  sha::iv(st);
  sha::compress(st, w);
}
// little-endian u32 word i (0..15) of a digest held as big-endian u64 words
DEV u32 dword(const u64* st, int i) { return bswap32(i & 1 ? (u32)st[i >> 1] : (u32)(st[i >> 1] >> 32)); }

// d_A or d_B: chunk b = i ceil(K / 512) + k / 512 of row i, its 64 bytes in digest order (bit k % 512 LSB first)
__global__ void bits_kernel(Seeds sd, int which, u64 blocks, uint8_t* __restrict__ out) {
  const u64 b = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  u64 st[8];
  prg(sd.w[which], tag_of(which), b, st);
  uint4* o = reinterpret_cast<uint4*>(out + 64 * b);
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = make_uint4(dword(st, 4 * i), dword(st, 4 * i + 1), dword(st, 4 * i + 2), dword(st, 4 * i + 3));
}
// U or V: word w of the stream ((r 161) >> 32) - 80
__global__ void noise_kernel(Seeds sd, int which, u64 count, int8_t* __restrict__ out) {
  const u64 b = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (16 * b >= count) return;
  u64 st[8];
  prg(sd.w[which], tag_of(which), b, st);
#pragma unroll
  for (int i = 0; i < 16; ++i)
    if (16 * b + i < count) out[16 * b + i] = (int8_t)((int)(((u64)dword(st, i) * (2 * NOISE + 1)) >> 32) - NOISE);
}
// pi_tau: Fisher-Yates over L' with word tau L' + c of the P stream (c from L' - 1 down to 1), a CTA of PERM_THREADS(L')
// threads a tile and 6 L' bytes of shared memory (at most 512 threads: its SHA-512 takes 90 registers, and a 1024-thread CTA
// of those doesn't launch)
__host__ __device__ constexpr int PERM_THREADS(int lp) { return lp < 512 ? lp : 512; }
__global__ void __launch_bounds__(512) perm_kernel(Seeds sd, int lp, uint16_t* __restrict__ perm) {
  extern __shared__ u32 pw[];
  uint16_t* p = reinterpret_cast<uint16_t*>(pw + lp);
  const int tau = blockIdx.x;
  for (int t = threadIdx.x; t < lp / 16; t += blockDim.x) {
    u64 st[8];
    prg(sd.w[S_P], tag_of(S_P), (u64)tau * (lp / 16) + t, st);
#pragma unroll
    for (int i = 0; i < 16; ++i) pw[16 * t + i] = dword(st, i);
  }
  for (int t = threadIdx.x; t < lp; t += blockDim.x) p[t] = (uint16_t)t;
  __syncthreads();
  if (threadIdx.x == 0)
    for (int c = lp - 1; c > 0; --c) {
      const int x = (int)(((u64)pw[c] * (u64)(c + 1)) >> 32);
      const uint16_t s = p[c]; p[c] = p[x]; p[x] = s;
    }
  __syncthreads();
  for (int t = threadIdx.x; t < lp; t += blockDim.x) perm[(size_t)tau * lp + t] = p[t];
}

// ---------------------------------------------------------------- forming
// balanced base-128 digits of z, the last a full int8 (flag 2 when it isn't)
DEV int digit(int& z, bool last, int* flag) {
  if (last) {
    if (z < -128 || z > 127) atomicOr(flag, 2);
    return z;
  }
  const int d = ((z + 64) & 127) - 64;
  z = (z - d) >> 7;
  return d;
}
struct FormArgs {
  const int8_t* Q;        // A (m x K) or B (N x K)
  const uint8_t* bits;    // d_A or d_B
  const int8_t* noise;    // V (m x L') or U (N x L')
  const uint16_t* perm;   // pi (K / L' x L')
  int8_t* X;              // X or W' (rows x K)
  void* T;                // X^T (K x m int8) or B~^T (K x N int16)
  int* list;              // the wrap lists, cap a row
  int* cnt;
  int8_t* P;              // the tail rows (rows x TL)
  int* flag;              // 1: a wrap list overflowed; 2: a top digit left int8
  int rows, K, lp, nt, kb64, dz, dy, cap, TL;
  int* off;               // B side: row j's wraps of noise tile tau are its list's [off[j (nt + 1) + tau], ... + tau + 1)
};
// A row's entry k of A + d_A (or B + d_B), through the read-only path: FormArgs' pointers may alias, so a noise tile's 16
// rows are loaded this way before any of them is stored, or each load would wait for the stores before it
DEV int dithered(const FormArgs& f, size_t i, int k) {
  return (int)__ldg(f.Q + i * f.K + k) + ((__ldg(f.bits + i * f.kb64 + (k >> 3)) >> (k & 7)) & 1);
}
// A side: a CTA of L' threads takes 16 rows; thread c the column c of every noise tile. Z_X[i, pi_tau(c)] += A~[i, tau L' + c]
// through shared atomics; X^T in 16-byte columns of the 16 rows. 80 registers: three CTAs an SM at L' 256 (the rows' 16
// loads in flight take it to 96 uncapped).
__global__ void __maxnreg__(80) form_a_kernel(FormArgs f) {
  extern __shared__ int zs[];   // [16][L']
  __shared__ int scnt[16];
  const int c = threadIdx.x, lp = f.lp;
  const size_t i0 = (size_t)blockIdx.x * 16;
  int v[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) { zs[r * lp + c] = 0; v[r] = f.noise[(i0 + r) * lp + c]; }
  if (c < 16) scnt[c] = 0;
  __syncthreads();
  for (int tau = 0; tau < f.nt; ++tau) {
    const int k = tau * lp + c, pc = __ldg(f.perm + (size_t)tau * lp + c);
    int ad[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) ad[r] = dithered(f, i0 + r, k);
    u32 xp[4] = {0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const size_t i = i0 + r;
      const int at = ad[r] + v[r];
      const int x = ((at + 128) & 255) - 128, w = (at - x) >> 8;
      f.X[i * f.K + k] = (int8_t)x;
      xp[r >> 2] |= (u32)(x & 255) << (8 * (r & 3));
      atomicAdd(&zs[r * lp + pc], at);
      if (w) {
        const int q = atomicAdd(&scnt[r], 1);
        if (q < f.cap) f.list[i * f.cap + q] = k << 1 | (w < 0);
        else atomicOr(f.flag, 1);
      }
    }
    *reinterpret_cast<uint4*>(static_cast<int8_t*>(f.T) + (size_t)k * f.rows + i0) = make_uint4(xp[0], xp[1], xp[2], xp[3]);
  }
  __syncthreads();
  if (c < 16) f.cnt[i0 + c] = min(scnt[c], f.cap);
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    int8_t* p = f.P + (i0 + r) * f.TL;
    int z = zs[r * lp + c];
    for (int d = 0; d < f.dz; ++d) p[(d ? d + 1 : 0) * lp + c] = (int8_t)digit(z, d + 1 == f.dz, f.flag);
    p[lp + c] = (int8_t)-v[r];
    for (int d = 1; d < f.dy; ++d) p[(f.dz + d) * lp + c] = (int8_t)v[r];
  }
}
// B side: thread c holds Y_W[j, c] for its 16 rows in registers; U's row gathered through pi from shared memory
// (WW 1: w_W to ww, N x K of -1, 0 or 1, wrap5's B operand; WW 2: -2 w_W, wrap 6's (wg_proto.cuh's KQ k-tiles); a
// parameter of its own, since a FormArgs field more costs form_a_kernel 72 registers)
template <int WW>
__global__ void form_b_kernel(FormArgs f, int8_t* __restrict__ ww) {
  extern __shared__ int8_t us[];   // [16][L']
  __shared__ int scnt[16];
  const int c = threadIdx.x, lp = f.lp;
  const size_t j0 = (size_t)blockIdx.x * 16;
  int y[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) { us[r * lp + c] = f.noise[(j0 + r) * lp + c]; y[r] = 0; }
  if (c < 16) scnt[c] = 0;
  __syncthreads();
  for (int tau = 0; tau < f.nt; ++tau) {
    if (f.off) {
      __syncthreads();
      if (c < 16) f.off[(j0 + c) * (f.nt + 1) + tau] = min(scnt[c], f.cap);
      __syncthreads();
    }
    const int k = tau * lp + c, pc = __ldg(f.perm + (size_t)tau * lp + c);
    int bds[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) bds[r] = dithered(f, j0 + r, k);
    u32 bp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const size_t j = j0 + r;
      const int bd = bds[r], bt = bd + us[r * lp + pc];
      const int x = ((bt + 128) & 255) - 128, w = (bt - x) >> 8;
      f.X[j * f.K + k] = (int8_t)x;
      if (WW) ww[j * f.K + k] = (int8_t)(WW == 2 ? -2 * w : w);
      y[r] += bd;
      bp[r >> 1] |= (u32)(bt & 0xFFFF) << (16 * (r & 1));
      if (w) {
        const int q = atomicAdd(&scnt[r], 1);
        if (q < f.cap) f.list[j * f.cap + q] = k << 1 | (w < 0);
        else atomicOr(f.flag, 1);
      }
    }
    uint4* o = reinterpret_cast<uint4*>(static_cast<int16_t*>(f.T) + (size_t)k * f.rows + j0);
    o[0] = make_uint4(bp[0], bp[1], bp[2], bp[3]);
    o[1] = make_uint4(bp[4], bp[5], bp[6], bp[7]);
  }
  __syncthreads();
  if (c < 16) f.cnt[j0 + c] = min(scnt[c], f.cap);
  if (f.off && c < 16) f.off[(j0 + c) * (f.nt + 1) + f.nt] = min(scnt[c], f.cap);
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    int8_t* p = f.P + (j0 + r) * f.TL;
    const int8_t u = us[r * lp + c];
    p[c] = (int8_t)-u;
    for (int d = 1; d < f.dz; ++d) p[(d + 1) * lp + c] = u;
    int z = y[r];
    for (int d = 0; d < f.dy; ++d) p[(d ? f.dz + d : 1) * lp + c] = (int8_t)digit(z, d + 1 == f.dy, f.flag);
  }
}

// The no-fold filler's operands (Fill): X = wrap8(A + d_A + V[k mod L']) and W' = wrap8(B + d_B + U[pi_tau(k mod L')]), with
// no wrap lists, transposes or tail rows; a thread 16 entries of a row (k0 a multiple of 16, so its 16 dither bits are
// one aligned u16)
DEV int sx8(u32 w, int e) { return (int)(int8_t)(w >> (8 * e)); }
__global__ void __launch_bounds__(256) fill_a_kernel(const int8_t* __restrict__ A, const uint8_t* __restrict__ bits,
                                                     const int8_t* __restrict__ V, int8_t* __restrict__ X, u64 chunks, int K,
                                                     int lp, int kb64) {
  const u64 q = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= chunks) return;
  const u64 per = (u64)K / 16, i = q / per;
  const int k0 = (int)(q - i * per) * 16;
  const uint4 a4 = __ldg(reinterpret_cast<const uint4*>(A + i * K + k0));
  const uint4 v4 = __ldg(reinterpret_cast<const uint4*>(V + i * lp + k0 % lp));
  const u32 d = __ldg(reinterpret_cast<const uint16_t*>(bits + i * kb64 + (k0 >> 3)));
  const u32 a[4] = {a4.x, a4.y, a4.z, a4.w}, v[4] = {v4.x, v4.y, v4.z, v4.w};
  u32 o[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    u32 r = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) r |= (u32)((sx8(a[w], e) + (int)((d >> (4 * w + e)) & 1) + sx8(v[w], e)) & 255) << (8 * e);
    o[w] = r;
  }
  *reinterpret_cast<uint4*>(X + i * K + k0) = make_uint4(o[0], o[1], o[2], o[3]);
}
__global__ void __launch_bounds__(256) fill_b_kernel(const int8_t* __restrict__ B, const uint8_t* __restrict__ bits,
                                                     const int8_t* __restrict__ U, const uint16_t* __restrict__ perm,
                                                     int8_t* __restrict__ W, u64 chunks, int K, int lp, int kb64) {
  const u64 q = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= chunks) return;
  const u64 per = (u64)K / 16, j = q / per;
  const int k0 = (int)(q - j * per) * 16;
  const uint4 b4 = __ldg(reinterpret_cast<const uint4*>(B + j * K + k0));
  const uint4* pp = reinterpret_cast<const uint4*>(perm + (size_t)(k0 / lp) * lp + k0 % lp);
  const uint4 p0 = __ldg(pp), p1 = __ldg(pp + 1);
  const u32 d = __ldg(reinterpret_cast<const uint16_t*>(bits + j * kb64 + (k0 >> 3)));
  const u32 b[4] = {b4.x, b4.y, b4.z, b4.w}, p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
  const int8_t* u = U + j * lp;
  u32 o[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    u32 r = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int x = 4 * w + e, pc = (int)(p[x >> 1] >> (16 * (x & 1))) & 0xFFFF;
      r |= (u32)((sx8(b[w], e) + (int)((d >> x) & 1) + (int)__ldg(u + pc)) & 255) << (8 * e);
    }
    o[w] = r;
  }
  *reinterpret_cast<uint4*>(W + j * K + k0) = make_uint4(o[0], o[1], o[2], o[3]);
}

// A served call's O operands (Fill::o): row i of A + d_A (in -128 .. 128, not int8) as a' = A + d_A - sigma_i in int8, with
// sigma_i = 0 when the row has no 128 and 1 when it has no -128, and rsum[i] = sum_k a'_ik + addk K sigma_i; a row with both
// sets *bad (refused at the root). Then O = (a' + sigma)(b' + tau)^T = a' b'^T + tau_j ra_i + sigma_i (rb_j + K tau_j), the
// rank-1 terms added as the O rows are hashed (ServedSrc). A CTA a row, 16 entries a thread (the row's dither bits at
// bits + row kb64, LSB first, an aligned u16 a thread).
DEV u32 dith4(u32 a, u32 d, int x, int& lo, int& hi) {   // bytes x .. x + 3 of a with their bits d >> x added, as ints
  u32 r = 0;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int v = sx8(a, e) + (int)((d >> (x + e)) & 1);
    lo = min(lo, v);
    hi = max(hi, v);
    r |= (u32)(v & 255) << (8 * e);
  }
  return r;
}
__global__ void __launch_bounds__(256) serve_rows_kernel(const int8_t* __restrict__ src, const uint8_t* __restrict__ bits, int K,
                                                         int kb64, int addk, int8_t* __restrict__ out, int* __restrict__ rsum,
                                                         int* __restrict__ shift, int* __restrict__ bad) {
  __shared__ int red[3][8];
  const u64 row = blockIdx.x;
  const int8_t* s = src + row * K;
  const uint8_t* bt = bits + row * kb64;
  int lo = 0, hi = 0;
  for (int k0 = 16 * threadIdx.x; k0 < K; k0 += 16 * blockDim.x) {
    const uint4 a4 = __ldg(reinterpret_cast<const uint4*>(s + k0));
    const u32 d = __ldg(reinterpret_cast<const uint16_t*>(bt + (k0 >> 3)));
    dith4(a4.x, d, 0, lo, hi); dith4(a4.y, d, 4, lo, hi); dith4(a4.z, d, 8, lo, hi); dith4(a4.w, d, 12, lo, hi);
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o; o >>= 1) {
    lo = min(lo, __shfl_xor_sync(0xFFFFFFFFu, lo, o));
    hi = max(hi, __shfl_xor_sync(0xFFFFFFFFu, hi, o));
  }
  if (!lane) { red[0][warp] = lo; red[1][warp] = hi; }
  __syncthreads();
  lo = red[0][0]; hi = red[1][0];
  for (int w = 1; w < (int)(blockDim.x >> 5); ++w) { lo = min(lo, red[0][w]); hi = max(hi, red[1][w]); }
  const int sg = hi > 127 ? 1 : 0;
  int sum = 0;
  for (int k0 = 16 * threadIdx.x; k0 < K; k0 += 16 * blockDim.x) {
    const uint4 a4 = __ldg(reinterpret_cast<const uint4*>(s + k0));
    const u32 d = __ldg(reinterpret_cast<const uint16_t*>(bt + (k0 >> 3)));
    const u32 a[4] = {a4.x, a4.y, a4.z, a4.w};
    u32 o[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      u32 r = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int v = sx8(a[w], e) + (int)((d >> (4 * w + e)) & 1) - sg;
        sum += v;
        r |= (u32)(v & 255) << (8 * e);
      }
      o[w] = r;
    }
    *reinterpret_cast<uint4*>(out + row * K + k0) = make_uint4(o[0], o[1], o[2], o[3]);
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
  __syncthreads();
  if (!lane) red[2][warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) sum += red[2][w];
    rsum[row] = (int)((u32)sum + (u32)(addk * K * sg));
    shift[row] = sg;
    if (hi > 127 && lo < -127) atomicOr(bad, 1);
  }
}

// ---------------------------------------------------------------- the wrap correction
struct WrapArgs {
  const int8_t* XT;      // K x m
  const int16_t* BtT;    // K x N
  const int *listA, *cntA, *listB, *cntB;
  int* dO;               // m x N
  int m, n, capA, capB;
  const int* offB;       // FormArgs::off of the B side
  int lp, nt;
};
template <class F> DEV void bytes16(const uint4& v, F&& f) {
  const u32 w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) f(i, (int)(int8_t)(w[i >> 2] >> (8 * (i & 3))));
}
template <class F> DEV void halves8(const uint4& v, F&& f) {
  const u32 w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) f(i, (int)(int16_t)(w[i >> 1] >> (16 * (i & 1))));
}
// A 128 x 128 tile of dO a CTA: X w_W^T by columns (thread: a column, 64 rows; a wrap (k, s) of B's row j adds s X[i, k]
// from X^T's row k), then w_X B~^T by rows (thread: a row, 64 columns; from B~^T's row k), summed in shared memory.
__global__ void __launch_bounds__(256) wrap_kernel(WrapArgs a) {
  extern __shared__ int acc[];   // [128][129]
  const int t = threadIdx.x, h = t >> 7, l = t & 127;
  const size_t i0 = (size_t)blockIdx.y * TILE, j0 = (size_t)blockIdx.x * TILE;
  {
    int r[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) r[i] = 0;
    const int nb = a.cntB[j0 + l];
    const int* L = a.listB + (j0 + l) * a.capB;
    for (int q = 0; q < nb; ++q) {
      const int e = __ldg(L + q);
      const uint4* p = reinterpret_cast<const uint4*>(a.XT + (size_t)(e >> 1) * a.m + i0 + 64 * h);
      const int s = e & 1 ? -1 : 1;
#pragma unroll
      for (int u = 0; u < 4; ++u) bytes16(__ldg(p + u), [&](int i, int x) { r[16 * u + i] += s * x; });
    }
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[(64 * h + i) * 129 + l] = r[i];
  }
  __syncthreads();
  {
    int r[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) r[i] = 0;
    const int na = a.cntA[i0 + l];
    const int* L = a.listA + (i0 + l) * a.capA;
    for (int q = 0; q < na; ++q) {
      const int e = __ldg(L + q);
      const uint4* p = reinterpret_cast<const uint4*>(a.BtT + (size_t)(e >> 1) * a.n + j0 + 64 * h);
      const int s = e & 1 ? -1 : 1;
#pragma unroll
      for (int u = 0; u < 8; ++u) halves8(__ldg(p + u), [&](int i, int x) { r[8 * u + i] += s * x; });
    }
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[l * 129 + 64 * h + i] += r[i];
  }
  __syncthreads();
  for (int e = t; e < TILE * TILE; e += 256) a.dO[(i0 + (e >> 7)) * a.n + j0 + (e & 127)] = acc[(e >> 7) * 129 + (e & 127)];
}

DEV void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
DEV void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// w_X B~^T added into the 128 x CW tile of dO at (i0, j0), after its X w_W^T is stored (rare: A's wraps): thread (row r,
// group g) over the tile's columns 32 g .. 32 g + 31, then 32 (g + G) .., ... (G = blockDim / 128), from B~^T's row k of
// each wrap (k, s);
// ST: stored rather than added (V31_WRAP 6, whose GEMM takes X w_W^T itself), zeros where a row has no wrap
template <int CW, bool ST = false>
DEV void wrap_a_side(const WrapArgs& a, size_t i0, size_t j0) {
  const int t = threadIdx.x, r = t & 127, g = t >> 7, G = blockDim.x >> 7;
  const int na = a.cntA[i0 + r];
  if (na == 0 && !ST) return;
  const int* LA = a.listA + (i0 + r) * a.capA;
#pragma unroll 1
  for (int cb = 32 * g; cb < CW; cb += 32 * G) {
    const size_t c0 = j0 + cb;
    int v[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = 0;
    for (int q = 0; q < na; ++q) {
      const int e = __ldg(LA + q);
      const uint4* p = reinterpret_cast<const uint4*>(a.BtT + (size_t)(e >> 1) * a.n + c0);
      const int s = e & 1 ? -1 : 1;
#pragma unroll
      for (int u = 0; u < 4; ++u) halves8(__ldg(p + u), [&](int i, int x) { v[8 * u + i] += s * x; });
    }
    int4* d = reinterpret_cast<int4*>(a.dO + (i0 + r) * a.n + c0);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      int4 x = ST ? make_int4(0, 0, 0, 0) : d[u];
      x.x += v[4 * u]; x.y += v[4 * u + 1]; x.z += v[4 * u + 2]; x.w += v[4 * u + 3];
      d[u] = x;
    }
  }
}

// The same dO over 128 rows x CW columns a CTA (2 CW threads). X w_W^T by columns: noise tile by noise tile, the tile's
// L' rows of X^T (128 bytes each, 16-byte chunks swizzled by k & 7) go to shared memory, and thread (column l, half h)
// adds s X[i, k] for its column's wraps in the tile (FormArgs::off) to its 64 rows. It adds v = x ^ 0x80 (s = 1) or
// x ^ 0x7F (s = -1), both 0..255, in 16-bit lanes (two rows a register), and the constant -128 or -127 once a wrap;
// at most 256 wraps between flushes, so no lane carries. Then w_X B~^T by rows (rare) from global memory into dO.
template <int CW>
__global__ void __launch_bounds__(2 * CW, CW == 128 ? 2 : 1) wrap2_kernel(WrapArgs a) {
  extern __shared__ __align__(16) uint8_t xs[];
  const int t = threadIdx.x, l = t % CW, h = t / CW, lp = a.lp, nt = a.nt;
  const size_t i0 = (size_t)blockIdx.y * TILE, j0 = (size_t)blockIdx.x * CW;
  const int* L = a.listB + (j0 + l) * a.capB;
  const int* off = a.offB + (j0 + l) * (nt + 1);
  int* dcol = a.dO + (i0 + 64 * h) * a.n + j0 + l;
  u32 acc[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) acc[i] = 0;
  int cst = 0, since = 0;
  bool first = true;
  auto flush = [&]() {
    int* p = dcol;
    const size_t n = a.n;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
      const int v[4] = {(int)(acc[2 * w] & 0xFFFF), (int)(acc[2 * w + 1] & 0xFFFF), (int)(acc[2 * w] >> 16), (int)(acc[2 * w + 1] >> 16)};
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        *p = (first ? 0 : *p) + v[b] + cst;
        p += n;
      }
      acc[2 * w] = acc[2 * w + 1] = 0;
    }
    cst = 0; since = 0; first = false;
  };
  const uint32_t xb = (uint32_t)__cvta_generic_to_shared(xs);
  int o0 = off[0];
  for (int tau = 0; tau < nt; ++tau) {
    __syncthreads();
    for (int e = t; e < lp * 8; e += 2 * CW) {
      const int kk = e >> 3, c = e & 7;
      cp_async16(xb + kk * 128 + ((c ^ (kk & 7)) << 4), a.XT + (size_t)(tau * lp + kk) * a.m + i0 + 16 * c);
    }
    cp_async_wait_all();
    __syncthreads();
    const int o1 = off[tau + 1];
    for (int q = o0; q < o1;) {
      if (since == 256) flush();
      const int qe = min(o1, q + 256 - since);
      since += qe - q;
      for (; q < qe; ++q) {
        const int e = __ldg(L + q), kk = (e >> 1) - tau * lp;
        const u32 mask = e & 1 ? 0x7F7F7F7Fu : 0x80808080u;
        cst += e & 1 ? -127 : -128;
        const uint8_t* row = xs + kk * 128;
        const int sw = kk & 7;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const uint4 v = *reinterpret_cast<const uint4*>(row + (((4 * h + u) ^ sw) << 4));
          const u32 w[4] = {v.x ^ mask, v.y ^ mask, v.z ^ mask, v.w ^ mask};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            acc[2 * (4 * u + i)] += __byte_perm(w[i], 0, 0x4240);
            acc[2 * (4 * u + i) + 1] += __byte_perm(w[i], 0, 0x4341);
          }
        }
      }
    }
    o0 = o1;
  }
  flush();
  __syncthreads();
  wrap_a_side<CW>(a, i0, j0);
}

// wrap3: wrap2's dO with a warp's lanes on one wrap at a time. A warp owns CW / 8 columns and takes them four at a time
// (a quad): lane (g, i) = (lane / 8, lane % 8) adds s X[16 i .. 16 i + 15, k] for the wraps (k, s) of the quad's column g,
// reading 16 bytes of the staged X^T row k (a quad's four reads are four whole 128-byte rows: no bank conflicts). A round
// costs the most wraps of the quad's four columns, not of a warp's 32; a column out of wraps reads the row of 0x80 bytes
// past the tile, which adds 0. A lane keeps its 16 rows in 16-bit lanes (wrap2's sums) and its column's wraps and
// negative wraps since its last flush (n | neg << 16), and flushes the quad into dO before a column passes 256.
template <int CW>
__global__ void __launch_bounds__(256, CW == 128 ? 3 : 2) wrap3_kernel(WrapArgs a) {
  constexpr int CPW = CW / 8, NQ = CPW / 4;
  constexpr u32 ALL = 0xFFFFFFFFu;
  extern __shared__ __align__(16) uint8_t xs[];
  const int t = threadIdx.x, lane = t & 31, g = lane >> 3, i = lane & 7, gl = lane & 24, lp = a.lp, nt = a.nt;
  const size_t i0 = (size_t)blockIdx.y * TILE, jb = (size_t)blockIdx.x * CW, j0 = jb + (t >> 5) * CPW;
  const int* offc = a.offB + (j0 + (lane < CPW ? lane : 0)) * (nt + 1);   // lane c < CPW: column j0 + c's offsets
  u32 acc[NQ][8], cnt[NQ], flushed = 0;
#pragma unroll
  for (int q = 0; q < NQ; ++q) {
    cnt[q] = 0;
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[q][r] = 0;
  }
  if (t < 8) reinterpret_cast<uint4*>(xs + lp * 128)[t] = make_uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u);
  // acc[q][2 k + (b & 1)]'s low (b < 2) or high half: row 16 i + 4 k + b of column 4 q + g
  auto flush = [&](int q) {
    int* d = a.dO + (i0 + 16 * i) * a.n + j0 + 4 * q + g;
    const int cst = (int)(cnt[q] >> 16) - 128 * (int)(cnt[q] & 0xFFFF);
    const bool first = !(flushed >> q & 1);
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        int* p = d + (size_t)(4 * k + b) * a.n;
        const u32 w = acc[q][2 * k + (b & 1)];
        *p = (first ? 0 : *p) + (int)(b < 2 ? w & 0xFFFF : w >> 16) + cst;
      }
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[q][r] = 0;
    cnt[q] = 0;
    flushed |= 1u << q;
  };
  const uint32_t xb = (uint32_t)__cvta_generic_to_shared(xs);
  int o0 = lane < CPW ? __ldg(offc) : 0;
  for (int tau = 0; tau < nt; ++tau) {
    const int o1 = lane < CPW ? __ldg(offc + tau + 1) : 0;
    __syncthreads();
    for (int e = t; e < lp * 8; e += 256) {
      const int kk = e >> 3, c = e & 7;
      cp_async16(xb + kk * 128 + (c << 4), a.XT + (size_t)(tau * lp + kk) * a.m + i0 + 16 * c);
    }
    cp_async_wait_all();
    __syncthreads();
    const int nl = o1 - o0, kb = tau * lp;
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      const int src = 4 * q + g, nc = __shfl_sync(ALL, nl, src), bc = __shfl_sync(ALL, o0, src);
      if (__any_sync(ALL, (int)(cnt[q] & 0xFFFF) + nc > 256)) flush(q);
      const int R = (int)__reduce_max_sync(ALL, (u32)nc);
      const int* Lc = a.listB + (j0 + src) * a.capB + bc;
      u32 inc = 0;
      for (int ch = 0; ch < R; ch += 8) {
        if (ch && (ch & 255) == 0) {   // L' > 256: a column's 257th wrap in one noise tile
          cnt[q] += inc;
          inc = 0;
          flush(q);
        }
        u32 pl = (u32)lp << 7;   // p: the staged row's offset | negative
        if (ch + i < nc) {
          const int e = __ldg(Lc + ch + i);
          pl = (u32)((e >> 1) - kb) << 7 | (u32)(e & 1);
        }
        inc += (u32)min(max(nc - ch, 0), 8) + ((u32)__popc(__ballot_sync(ALL, pl & 1) >> gl & 0xFF) << 16);
        const int nr = min(8, R - ch);
        for (int r = 0; r < nr; ++r) {
          const u32 p = __shfl_sync(ALL, pl, gl | r);
          const uint4 v = *reinterpret_cast<const uint4*>(xs + (p & ~1u) + 16 * i);
          const u32 m = 0x80808080u ^ (0u - (p & 1u)), w[4] = {v.x ^ m, v.y ^ m, v.z ^ m, v.w ^ m};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            acc[q][2 * k] += __byte_perm(w[k], 0, 0x4240);
            acc[q][2 * k + 1] += __byte_perm(w[k], 0, 0x4341);
          }
        }
      }
      cnt[q] += inc;
    }
    o0 = o1;
  }
#pragma unroll
  for (int q = 0; q < NQ; ++q) flush(q);
  __syncthreads();
  wrap_a_side<CW>(a, i0, jb);
}

// wrap4: wrap3 with its global loads off the rounds' path. A noise tile's offsets are loaded a tile ahead, and the first
// eight list entries of each of the warp's quads before the staging wait (lane (g, i): entry i of column 4 q + g; a
// quad's later entries, rare, as wrap3 loads them). Rounds go in pairs, both rows read before either is added; the odd
// round's partner reads a column's lane past its wraps, so the row of 0x80 bytes.
template <int CW>
__global__ void __launch_bounds__(256, CW == 128 ? 3 : 2) wrap4_kernel(WrapArgs a) {
  constexpr int CPW = CW / 8, NQ = CPW / 4;
  constexpr u32 ALL = 0xFFFFFFFFu;
  extern __shared__ __align__(16) uint8_t xs[];
  const int t = threadIdx.x, lane = t & 31, g = lane >> 3, i = lane & 7, gl = lane & 24, lp = a.lp, nt = a.nt;
  const size_t i0 = (size_t)blockIdx.y * TILE, jb = (size_t)blockIdx.x * CW, j0 = jb + (t >> 5) * CPW;
  const int* offc = a.offB + (j0 + (lane < CPW ? lane : 0)) * (nt + 1);   // lane c < CPW: column j0 + c's offsets
  u32 acc[NQ][8], cnt[NQ], flushed = 0;
#pragma unroll
  for (int q = 0; q < NQ; ++q) {
    cnt[q] = 0;
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[q][r] = 0;
  }
  if (t < 8) reinterpret_cast<uint4*>(xs + lp * 128)[t] = make_uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u);
  auto flush = [&](int q) {
    int* d = a.dO + (i0 + 16 * i) * a.n + j0 + 4 * q + g;
    const int cst = (int)(cnt[q] >> 16) - 128 * (int)(cnt[q] & 0xFFFF);
    const bool first = !(flushed >> q & 1);
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        int* p = d + (size_t)(4 * k + b) * a.n;
        const u32 w = acc[q][2 * k + (b & 1)];
        *p = (first ? 0 : *p) + (int)(b < 2 ? w & 0xFFFF : w >> 16) + cst;
      }
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[q][r] = 0;
    cnt[q] = 0;
    flushed |= 1u << q;
  };
  const uint32_t xb = (uint32_t)__cvta_generic_to_shared(xs);
  const u32 zrow = (u32)lp << 7;
  const int last = a.capB - 1;
  int o0 = lane < CPW ? __ldg(offc) : 0, o1 = lane < CPW ? __ldg(offc + 1) : 0;
  for (int tau = 0; tau < nt; ++tau) {
    const int o2 = lane < CPW && tau + 2 <= nt ? __ldg(offc + tau + 2) : 0;
    int pre[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      const int src = 4 * q + g, bc = __shfl_sync(ALL, o0, src);
      pre[q] = __ldg(a.listB + (j0 + src) * a.capB + min(bc + i, last));
    }
    __syncthreads();
    for (int e = t; e < lp * 8; e += 256) {
      const int kk = e >> 3, c = e & 7;
      cp_async16(xb + kk * 128 + (c << 4), a.XT + (size_t)(tau * lp + kk) * a.m + i0 + 16 * c);
    }
    cp_async_wait_all();
    __syncthreads();
    const int nl = o1 - o0, kb = tau * lp;
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      const int src = 4 * q + g, nc = __shfl_sync(ALL, nl, src);
      if (__any_sync(ALL, (int)(cnt[q] & 0xFFFF) + nc > 256)) flush(q);
      const int R = (int)__reduce_max_sync(ALL, (u32)nc);
      u32 inc = 0;
      int e = pre[q];
      for (int ch = 0; ch < R; ch += 8) {
        if (ch) {
          if ((ch & 255) == 0) {   // L' > 256: a column's 257th wrap in one noise tile
            cnt[q] += inc;
            inc = 0;
            flush(q);
          }
          const int bc = __shfl_sync(ALL, o0, src);
          e = ch + i < nc ? __ldg(a.listB + (j0 + src) * a.capB + bc + ch + i) : 0;
        }
        const u32 pl = ch + i < nc ? (u32)((e >> 1) - kb) << 7 | (u32)(e & 1) : zrow;   // the staged row's offset | negative
        inc += (u32)min(max(nc - ch, 0), 8) + ((u32)__popc(__ballot_sync(ALL, pl & 1) >> gl & 0xFF) << 16);
        const int nr = min(8, R - ch);
        for (int r = 0; r < nr; r += 2) {
          const u32 p0 = __shfl_sync(ALL, pl, gl | r), p1 = __shfl_sync(ALL, pl, gl | (r + 1));
          const uint4 v0 = *reinterpret_cast<const uint4*>(xs + (p0 & ~1u) + 16 * i);
          const uint4 v1 = *reinterpret_cast<const uint4*>(xs + (p1 & ~1u) + 16 * i);
          const u32 m0 = 0x80808080u ^ (0u - (p0 & 1u)), m1 = 0x80808080u ^ (0u - (p1 & 1u));
          const u32 w0[4] = {v0.x ^ m0, v0.y ^ m0, v0.z ^ m0, v0.w ^ m0}, w1[4] = {v1.x ^ m1, v1.y ^ m1, v1.z ^ m1, v1.w ^ m1};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            acc[q][2 * k] += __byte_perm(w0[k], 0, 0x4240) + __byte_perm(w1[k], 0, 0x4240);
            acc[q][2 * k + 1] += __byte_perm(w0[k], 0, 0x4341) + __byte_perm(w1[k], 0, 0x4341);
          }
        }
      }
      cnt[q] += inc;
    }
    o0 = o1;
    o1 = o2;
  }
#pragma unroll
  for (int q = 0; q < NQ; ++q) flush(q);
  __syncthreads();
  wrap_a_side<CW>(a, i0, jb);
}

// wrap5's A side: w_X B~^T added into dO's 128 x 128 tile a CTA, after the plain GEMM (X, w_W) has stored X w_W^T there;
// ST (wrap 6): stored as all of dO
template <bool ST>
__global__ void __launch_bounds__(256) wrap_a_kernel(WrapArgs a) {
  wrap_a_side<TILE, ST>(a, (size_t)blockIdx.y * TILE, (size_t)blockIdx.x * TILE);
}

// ---------------------------------------------------------------- TMA, mbarrier, ldmatrix, mma.sync (floored_gemm's)
DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEV void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEV void mbar_release(uint32_t bar) {
  asm volatile("fence.proxy.async.shared::cta;\n mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
DEV void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n V31_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra V31_WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
DEV void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
DEV void tma_acquire(const void* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"(map) : "memory");
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}
DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
DEV void ldsm4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// d += a b, one s8 atom (k = 32): D rows g, g + 8 x columns 2q, 2q + 1 (g = lane / 4, q = lane % 4)
DEV void imma(int* d, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEV void imma0(int* d, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0));
}
DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

// ---------------------------------------------------------------- the GEMM
struct Maps { CUtensorMap a, b, pa, pb; };
struct GArgs {
  const Maps* maps;
  uint16_t* words;   // [tile][s][128 x 128] T_s, little-endian u16
  const int* dO;     // null: none (the no-fold filler)
  int* O;            // m x N, row-major (u32 mod 2^32); null: none (the filler commits only its prefix sums)
  int M, N, K, L, S, TT, group;
  int lp, dz;        // tail stage ts is term b = ts 128 / L' of P_A P_B^T: b 0 and 1 (-Z_0 U^T, -V Y_0^T) add as main
                     // stages do; then Z_{b - 1} U^T (b <= dz) or V Y_{b - dz}^T, at 128^digit, subtract
  u32* ctr;          // the tile counter (null: CTA c takes tiles c, c + grid, ...); a launch advances it by its tiles
  u32 base;          // ctr's value before this launch
  int diag;          // timing only (the bench's gemm_no* arms, wrong values): 1 no segment stores, 2 no tail (TT 0), 4 no dO
};
// 128 x 128 tiles, 8 warps of 64 x 32 (MI = 4 row tiles of 16, NI = 4 column tiles of 8), a 3-stage ring of 128 bytes of
// depth; thread 0 issues every TMA as the CTA releases its slot (floored_gemm's ML, without its receipts and edits). With a
// counter, a CTA's first tile is its index and each next one the counter's (a CTA that starts late, behind another
// kernel's CTAs, takes fewer tiles); the issuer writes a tile's index beside its first stage, -1 when none is left.
struct Ring {
  static constexpr int STAGES = 3, TM = 128, BN = 128, WM = 64, WN = 32, WARPS_N = 4, MI = 4, NI = 4, NCT = 256;
  static constexpr uint32_t A_BYTES = TM * BK, STAGE = (TM + BN) * BK;
  static constexpr int SMEM = STAGES * STAGE + 16 * STAGES + 16 + 1024;
  uint32_t base, bars;
  int s = 0;
  uint32_t ph = 0;
  uint32_t a_off[4], b_off[4];
  const Maps* maps;
  int tiles_m, tiles_n, KT, steps, group, pt, pkt = 0, ps = 0;
  uint32_t pph = 0;
  bool issuer, fin = false;
  volatile int* tid;   // [STAGES]: the tile whose first stage the slot holds
  u32* ctr;
  u32 cbase;
  DEV Ring(uint8_t* smem, const Maps* m, int M, int N, int K, int TT, int grp, u32* c, u32 cb)
      : maps(m), tiles_m(M / TM), tiles_n(N / BN), KT(K / BK), steps(K / BK + TT), group(grp), pt(blockIdx.x),
        issuer(threadIdx.x == 0), ctr(c), cbase(cb) {
    base = (smem_u32(smem) + 1023) & ~1023u;
    bars = base + STAGES * STAGE;
    tid = reinterpret_cast<volatile int*>(smem + (base - smem_u32(smem)) + STAGES * STAGE + 16 * STAGES);
  }
  DEV uint32_t full(int i) const { return bars + 8 * i; }
  DEV uint32_t empty(int i) const { return bars + 8 * (STAGES + i); }
  DEV uint32_t stage(int i) const { return base + i * STAGE; }
  DEV void advance() {
    if (++s == STAGES) { s = 0; ph ^= 1; }
  }
  DEV void init(int tid) {
    if (tid == 0) {
      for (int i = 0; i < STAGES; i++) { mbar_init(full(i), 1); mbar_init(empty(i), NCT); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
  }
  DEV void tile_of(int t, int& tm, int& tn) const { raster(t, tiles_m, tiles_n, group, tm, tn); }
  DEV void issue() {
    if (fin) return;
    mbar_wait(empty(ps), pph ^ 1);
    if (pt >= tiles_m * tiles_n) {
      tid[ps] = -1;
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(full(ps)) : "memory");
      fin = true;
      return;
    }
    if (pkt == 0) tid[ps] = pt;
    int tm, tn;
    tile_of(pt, tm, tn);
    mbar_expect_tx(full(ps), STAGE);
    if (pkt < KT) {
      tma_load(stage(ps), &maps->a, full(ps), pkt * BK, tm * TM);
      tma_load(stage(ps) + A_BYTES, &maps->b, full(ps), pkt * BK, tn * BN);
    } else {
      tma_load(stage(ps), &maps->pa, full(ps), (pkt - KT) * BK, tm * TM);
      tma_load(stage(ps) + A_BYTES, &maps->pb, full(ps), (pkt - KT) * BK, tn * BN);
    }
    if (++pkt == steps) {
      pkt = 0;
      const u32 q = ctr ? gridDim.x + (atomicAdd(ctr, 1u) - cbase) : (u32)pt + gridDim.x;
      pt = q < (u32)(tiles_m * tiles_n) ? (int)q : tiles_m * tiles_n;
    }
    if (++ps == STAGES) { ps = 0; pph ^= 1; }
  }
  DEV void prologue() {
    if (issuer) {
      tma_acquire(&maps->a); tma_acquire(&maps->b);
      if (steps > KT) { tma_acquire(&maps->pa); tma_acquire(&maps->pb); }
      for (int i = 0; i < STAGES; i++) issue();
    }
    __syncwarp();
  }
  DEV void release(uint32_t done) {
    mbar_release(done);
    if (issuer) issue();
    __syncwarp();
  }
  DEV void consumer_init(int warp, int lane) {
    const int wm = warp / WARPS_N, wn = warp % WARPS_N, x = lane & 7;
    const int a_row = wm * WM + (lane & 15), a_half = lane >> 4;
    const int b_row = wn * WN + (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      a_off[kk] = a_row * BK + (((2 * kk + a_half) ^ x) << 4);
      b_off[kk] = A_BYTES + b_row * BK + (((2 * kk + b_half) ^ x) << 4);
    }
  }
  DEV void load_a(uint32_t (&fa)[4], uint32_t sb, int kk, int mi) const { ldsm4(fa, sb + a_off[kk] + mi * 16 * BK); }
  DEV void load_b(uint32_t (&fb)[NI][2], uint32_t sb, int kk) const {
#pragma unroll
    for (int nj = 0; nj < NI / 2; nj++) {
      uint32_t r[4];
      ldsm4(r, sb + b_off[kk] + nj * 16 * BK);
      fb[2 * nj][0] = r[0]; fb[2 * nj][1] = r[1]; fb[2 * nj + 1][0] = r[2]; fb[2 * nj + 1][1] = r[3];
    }
  }
  DEV void load(uint32_t (&fa)[MI][4], uint32_t (&fb)[NI][2], uint32_t sb, int kk) const {
#pragma unroll
    for (int mi = 0; mi < MI; mi++) load_a(fa[mi], sb, kk, mi);
    load_b(fb, sb, kk);
  }
  template <bool FROM0> DEV static void mma(int (&acc)[MI][NI][4], const uint32_t (&fa)[MI][4], const uint32_t (&fb)[NI][2]) {
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        if (FROM0) imma0(acc[mi][ni], fa[mi], fb[ni]);
        else imma(acc[mi][ni], fa[mi], fb[ni]);
      }
  }
  // one stage's four atoms, the next stage's first fragments loaded before the last atom (LAST: no next main stage)
  template <bool FIRST, bool LAST>
  DEV void body(int (&acc)[MI][NI][4], uint32_t (&fa)[2][MI][4], uint32_t (&fb)[2][NI][2]) {
    const uint32_t sb = stage(s), done = empty(s);
    load(fa[1], fb[1], sb, 1);
    mma<FIRST>(acc, fa[0], fb[0]);
    load(fa[0], fb[0], sb, 2);
    mma<false>(acc, fa[1], fb[1]);
    load(fa[1], fb[1], sb, 3);
    mma<false>(acc, fa[0], fb[0]);
    advance();
    if (!LAST) {
      mbar_wait_warp(full(s), ph);
      load(fa[0], fb[0], stage(s), 0);
    }
    mma<false>(acc, fa[1], fb[1]);
    release(done);
  }
  // a tail stage: acc -= (P_A P_B^T over its 128 columns) << shift. B's fragments are loaded a column pair at a time (again
  // for each row tile), so the tail holds 16 more registers than the mainloop, not 48: the GEMM stays small enough for
  // commit_gpu's small-path CTAs to sit beside it.
  DEV void tail(int (&acc)[MI][NI][4], int shift) {
    mbar_wait_warp(full(s), ph);
    const uint32_t tb = stage(s);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      uint32_t fa[4][4];
#pragma unroll
      for (int kk = 0; kk < 4; kk++) load_a(fa[kk], tb, kk, mi);
#pragma unroll
      for (int nj = 0; nj < NI / 2; nj++) {
        int t0[4], t1[4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          uint32_t r[4];
          ldsm4(r, tb + b_off[kk] + nj * 16 * BK);
          const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
          if (kk == 0) { imma0(t0, fa[0], b0); imma0(t1, fa[0], b1); }
          else { imma(t0, fa[kk], b0); imma(t1, fa[kk], b1); }
        }
#pragma unroll
        for (int e = 0; e < 4; e++) {
          acc[mi][2 * nj][e] = (int)((uint32_t)acc[mi][2 * nj][e] - ((uint32_t)t0[e] << shift));
          acc[mi][2 * nj + 1][e] = (int)((uint32_t)acc[mi][2 * nj + 1][e] - ((uint32_t)t1[e] << shift));
        }
      }
    }
    release(empty(s));
    advance();
  }
};

// T_s's low 16 bits. Lane q of a quad holds a row's columns 8 ni + 2 q, + 1 (ni < 4) of the warp's 32; two butterfly rounds
// transpose the quad's 4 x 4 prefix sums, so lane q stores columns 8 q .. 8 q + 7 as one 16-byte store and a warp's store
// writes whole sectors (eight rows of 64 bytes, not of four 16-byte pieces)
DEV void store_words(const int (&acc)[4][4][4], uint16_t* words, size_t tile, int S, int s, int wm, int wn, int g, int q) {
  constexpr u32 ALL = 0xFFFFFFFFu;
  uint4* w = reinterpret_cast<uint4*>(words + (tile * S + (size_t)(s - 1)) * (TILE * TILE));
  const bool o1 = q & 1, o2 = q & 2;
#pragma unroll
  for (int mi = 0; mi < 4; mi++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      u32 a[4];
#pragma unroll
      for (int ni = 0; ni < 4; ni++) a[ni] = __byte_perm((u32)acc[mi][ni][2 * h], (u32)acc[mi][ni][2 * h + 1], 0x5410);
      u32 x0 = __shfl_xor_sync(ALL, o1 ? a[0] : a[1], 1), x1 = __shfl_xor_sync(ALL, o1 ? a[2] : a[3], 1);
      if (o1) { a[0] = x0; a[2] = x1; } else { a[1] = x0; a[3] = x1; }
      x0 = __shfl_xor_sync(ALL, o2 ? a[0] : a[2], 2);
      x1 = __shfl_xor_sync(ALL, o2 ? a[1] : a[3], 2);
      if (o2) { a[0] = x0; a[1] = x1; } else { a[2] = x0; a[3] = x1; }
      w[((wm * 64 + g + 16 * mi + 8 * h) * TILE + wn * 32 + 8 * q) >> 3] = make_uint4(a[0], a[1], a[2], a[3]);
    }
}

// PROTO: the protocol's GEMM (segment stores, tail, 256 dO); else the plain product X W'^T to O. V31_MAXNREG caps its
// registers so a small-path commit CTA (128 threads of 96) fits beside its 256 threads in the SM's 65,536.
#ifndef V31_MAXNREG
#define V31_MAXNREG 208
#endif
#if !V31_WG
constexpr int G_NCT = Ring::NCT, G_SMEM = Ring::SMEM, G_WG_NREG = 0, G_WG_STAGES = 0, G_BN = TILE, G_NBW = 1;
template <bool PROTO, int NB>
__global__ void __maxnreg__(V31_MAXNREG) gemm_kernel(const GArgs a) {
  static_assert(NB == 1, "one tile a CTA");
  extern __shared__ __align__(1024) uint8_t gsm[];
  Ring R(gsm, a.maps, a.M, a.N, a.K, PROTO ? a.TT : 0, a.group, a.ctr, a.base);
  R.init(threadIdx.x);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  R.prologue();
  R.consumer_init(warp, lane);
  const int wm = warp / Ring::WARPS_N, wn = warp % Ring::WARPS_N, g = lane >> 2, q = lane & 3;
  const int KT = a.K / BK, KS = KT + (PROTO && a.TT ? 2 * a.lp / BK : 0);   // the main stages, then the tail's shift-0 blocks
  auto seg = [&](const int (&acc)[4][4][4], size_t tile, int kt) {
    if constexpr (PROTO)
      if (!(a.diag & 1) && (((kt + 1) * BK) % a.L == 0 || kt + 1 == KT))
        store_words(acc, a.words, tile, a.S, ((kt + 1) * BK + a.L - 1) / a.L, wm, wn, g, q);
  };
  for (;;) {
    mbar_wait_warp(R.full(R.s), R.ph);
    const int t = R.tid[R.s];
    if (t < 0) break;
    int tm, tn;
    R.tile_of(t, tm, tn);
    const size_t tile = (size_t)tm * R.tiles_n + tn;
    int acc[4][4][4];
    uint32_t fa[2][4][4], fb[2][4][2];
    R.load(fa[0], fb[0], R.stage(R.s), 0);
    if (KS == 1) {
      R.body<true, true>(acc, fa, fb);
      seg(acc, tile, 0);
    } else {
      R.body<true, false>(acc, fa, fb);
      seg(acc, tile, 0);
#pragma unroll 1
      for (int kt = 1; kt < KS - 1; kt++) {
        R.body<false, false>(acc, fa, fb);
        if (kt < KT) seg(acc, tile, kt);
      }
      R.body<false, true>(acc, fa, fb);
      if (KS - 1 < KT) seg(acc, tile, KS - 1);
    }
    const int row0 = tm * TILE + wm * 64 + g, col0 = tn * TILE + wn * 32 + 2 * q;
    if constexpr (PROTO) {
#pragma unroll 1
      for (int ts = KS - KT; ts < a.TT; ts++) {
        const int blk = ts * BK / a.lp;
        R.tail(acc, 7 * (blk <= a.dz ? blk - 1 : blk - a.dz));
      }
    }
    if (!a.O) continue;
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
      for (int ni = 0; ni < 4; ni++) {
        const size_t p = (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni, p8 = p + (size_t)8 * a.N;
        int2 lo = make_int2(acc[mi][ni][0], acc[mi][ni][1]), hi = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
        if (PROTO && a.dO && !(a.diag & 4)) {
          const int2 d0 = __ldg(reinterpret_cast<const int2*>(a.dO + p)), d1 = __ldg(reinterpret_cast<const int2*>(a.dO + p8));
          lo.x = (int)((u32)lo.x + ((u32)d0.x << 8)); lo.y = (int)((u32)lo.y + ((u32)d0.y << 8));
          hi.x = (int)((u32)hi.x + ((u32)d1.x << 8)); hi.y = (int)((u32)hi.y + ((u32)d1.y << 8));
        }
        *reinterpret_cast<int2*>(a.O + p) = lo;
        *reinterpret_cast<int2*>(a.O + p8) = hi;
      }
  }
}
#else
// V31_WG (sm_90a): the same GEMM on warpgroup MMAs, s8_wgmma.cu's mainloop (benchmarks/torch_gpu/s8_linear). The producer
// warpgroup's first thread issues every TMA in Ring::issue's order (a tile's main stages, then its tail stages; the tile's
// index beside its first stage, -1 when none is left; the counter for the next); consumer warpgroup w takes rows 64 w ..
// 64 w + 63 of the 128 x 128 NB tile as m64n(128 NB)k32 atoms on the stage's SWIZZLE_128B boxes, its accumulator i of
// thread x at row 16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2. A segment's store
// waits for all of the warpgroup's atoms; a tail stage's product is 2 NB m64n64 pieces in registers of their own,
// subtracted at its weight. V31_WG_NREG: a thread's registers at launch (the producer keeps 40 and the consumers take the rest), below 168
// so that a commit CTA fits beside the GEMM's 384 threads; V31_WG_STAGES: the ring's stages; V31_WG_BN 256: a CTA takes
// NB = 2 column-adjacent tiles as one 128 x 256 tile (m64n256 atoms, B's two boxes one after the other in the stage)
// wherever N / 128 is even, a third less traffic from L2 a product than 128 x 128.
#ifndef V31_WG_NREG
#define V31_WG_NREG 168
#endif
#ifndef V31_WG_STAGES
#define V31_WG_STAGES 4
#endif
#ifndef V31_WG_BN
#define V31_WG_BN 128
#endif
#include "../../torch_gpu/s8_linear/s8_acc.h"
struct WRing {
  static constexpr int STAGES = V31_WG_STAGES, NCT = 384, CONS = V31_WG_NREG + (V31_WG_NREG - 40) / 2;
  static constexpr uint32_t A_BYTES = TILE * BK;
  static constexpr int SMEM = STAGES * (1 + V31_WG_BN / TILE) * TILE * BK + 32 * STAGES + 1024;
  static_assert(CONS % 8 == 0 && CONS <= 256, "V31_WG_NREG: setmaxnreg's counts are multiples of 8, at most 256");
  static_assert(V31_WG_BN == 128 || V31_WG_BN == 256, "V31_WG_BN: 128 or 256");
};
constexpr int G_NCT = WRing::NCT, G_SMEM = WRing::SMEM, G_WG_NREG = V31_WG_NREG, G_WG_STAGES = V31_WG_STAGES,
              G_BN = V31_WG_BN, G_NBW = V31_WG_BN / TILE;
#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")
// K-major with 128-byte swizzling: 8-row groups 1,024 bytes apart, the start advanced 32 bytes a k32 atom
DEV uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
// d = A B + (sd ? d : 0) over one k32 atom of a 64 x N warpgroup tile
template <int N> DEV void wg_imma(int* d, uint64_t da, uint64_t db, uint32_t sd) {
  static_assert(N == 256 || N == 128 || N == 64, "n256, n128 or n64");
  if constexpr (N == 256)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
                 ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
  else if constexpr (N == 128)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 " ACC64_S
                 ", %64, %65, p;\n}" : ACC64_R : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 " ACC32_S
                 ", %32, %33, p;\n}" : ACC32_R : "l"(da), "l"(db), "r"(sd));
}
// after WG_WAIT: the accumulators are read only from here on
template <int NA> DEV void acc_fence(int* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}
template <int R> DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
DEV void mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }
DEV void wg_store_words(const int* d, uint16_t* words, size_t tile, int S, int s, int r0, int q) {
  u32* w = reinterpret_cast<u32*>(words + (tile * S + (size_t)(s - 1)) * (TILE * TILE));
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int p = r0 * TILE + 8 * j + 2 * q;
    w[p >> 1] = __byte_perm((u32)d[4 * j], (u32)d[4 * j + 1], 0x5410);
    w[(p + 8 * TILE) >> 1] = __byte_perm((u32)d[4 * j + 2], (u32)d[4 * j + 3], 0x5410);
  }
}
template <bool PROTO, int NB>
__global__ void __maxnreg__(V31_WG_NREG) gemm_kernel(const GArgs a) {
  using W = WRing;
  constexpr int ST = W::STAGES;
  constexpr uint32_t STAGE = (1 + NB) * TILE * BK;
  static_assert(NB == 1 || NB == 2, "one or two tiles a CTA");
  extern __shared__ __align__(1024) uint8_t gsm[];
  const uint32_t base = (smem_u32(gsm) + 1023) & ~1023u, bars = base + ST * STAGE;
  volatile int* tids = reinterpret_cast<volatile int*>(gsm + (base - smem_u32(gsm)) + ST * STAGE + 16 * ST);
  // a work item is NB column-adjacent tiles; tiles keep their 128 x 128 indexing in words and O
  const int tiles_m = a.M / TILE, tiles_n = a.N / TILE, items_n = tiles_n / NB, items = tiles_m * items_n, KT = a.K / BK;
  const int TT = PROTO ? a.TT : 0, KS = KT + (PROTO && TT ? 2 * a.lp / BK : 0);
  const int wg = threadIdx.x / 128;
  auto full = [&](int i) { return bars + 8 * i; };
  auto empty = [&](int i) { return bars + 8 * (ST + i); };
  auto stage = [&](int i) { return base + i * STAGE; };
  if (threadIdx.x == 0) {
    for (int i = 0; i < ST; i++) { mbar_init(full(i), 1); mbar_init(empty(i), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (wg == 2) {
    // the producer exits rather than returns: a path back would join its register budget with the consumers'
    regs_dec<40>();
    if (threadIdx.x == 256) {
      tma_acquire(&a.maps->a); tma_acquire(&a.maps->b);
      if (TT) { tma_acquire(&a.maps->pa); tma_acquire(&a.maps->pb); }
      int pt = blockIdx.x, pkt = 0, ps = 0;
      uint32_t pph = 0;
      for (;;) {
        mbar_wait(empty(ps), pph ^ 1);
        if (pt >= items) {
          tids[ps] = -1;
          mbar_arrive(full(ps));
          break;
        }
        if (pkt == 0) tids[ps] = pt;
        int tm, tn;
        raster(pt, tiles_m, items_n, a.group, tm, tn);
        mbar_expect_tx(full(ps), STAGE);
        const bool body = pkt < KT;
        const CUtensorMap *ma = body ? &a.maps->a : &a.maps->pa, *mb = body ? &a.maps->b : &a.maps->pb;
        const int kc = (body ? pkt : pkt - KT) * BK;
        tma_load(stage(ps), ma, full(ps), kc, tm * TILE);
#pragma unroll
        for (int j = 0; j < NB; j++) tma_load(stage(ps) + W::A_BYTES + j * TILE * BK, mb, full(ps), kc, (NB * tn + j) * TILE);
        if (++pkt == KT + TT) {
          pkt = 0;
          const u32 q = a.ctr ? gridDim.x + (atomicAdd(a.ctr, 1u) - a.base) : (u32)pt + gridDim.x;
          pt = q < (u32)items ? (int)q : items;
        }
        if (++ps == ST) { ps = 0; pph ^= 1; }
      }
    }
    asm volatile("exit;");
  }
  regs_inc<W::CONS>();

  const int x = threadIdx.x % 128, lane = x % 32, q = lane & 3, r0 = wg * 64 + 16 * (x / 32) + (lane >> 2);
  int s = 0;
  uint32_t ph = 0;
  for (;;) {
    mbar_wait_warp(full(s), ph);
    const int t = tids[s];
    if (t < 0) break;
    int tm, tn;
    raster(t, tiles_m, items_n, a.group, tm, tn);
    tn *= NB;
    const size_t tile = (size_t)tm * tiles_n + tn;
    constexpr int NA = 64 * NB;
    int d[NA];
    int pend = -1;   // the stage whose atoms may be in flight, released once they're done
#pragma unroll 1
    for (int kt = 0; kt < KS; kt++) {
      if (kt) mbar_wait_warp(full(s), ph);
      const uint32_t a0 = stage(s) + wg * 64 * BK, b0 = stage(s) + W::A_BYTES;
      WG_FENCE;
      wg_imma<128 * NB>(d, desc_sw128(a0), desc_sw128(b0), kt > 0);
      wg_imma<128 * NB>(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
      wg_imma<128 * NB>(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
      wg_imma<128 * NB>(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
      WG_COMMIT;
      if (PROTO && !(a.diag & 1) && kt < KT && (((kt + 1) * BK) % a.L == 0 || kt + 1 == KT)) {
        WG_WAIT(0);
        acc_fence<NA>(d);
        if (lane == 0) {
          if (pend >= 0) mbar_arrive(empty(pend));
          mbar_arrive(empty(s));
        }
        pend = -1;
        // accumulators 64 j .. 64 j + 63 are tile tn + j's, in the n128 layout
#pragma unroll
        for (int j = 0; j < NB; j++) wg_store_words(d + 64 * j, a.words, tile + j, a.S, ((kt + 1) * BK + a.L - 1) / a.L, r0, q);
      } else {
        WG_WAIT(1);
        if (lane == 0 && pend >= 0) mbar_arrive(empty(pend));
        pend = s;
      }
      if (++s == ST) { s = 0; ph ^= 1; }
    }
    WG_WAIT(0);
    acc_fence<NA>(d);
    if (lane == 0 && pend >= 0) mbar_arrive(empty(pend));
    if constexpr (PROTO) {
#pragma unroll 1
      for (int ts = KS - KT; ts < TT; ts++) {
        const int blk = ts * BK / a.lp, shift = 7 * (blk <= a.dz ? blk - 1 : blk - a.dz);
        mbar_wait_warp(full(s), ph);
        const uint32_t a0 = stage(s) + wg * 64 * BK, b0 = stage(s) + W::A_BYTES;
#pragma unroll
        for (int h = 0; h < 2 * NB; h++) {
          int tt[32];
          const uint32_t bh = b0 + h * 64 * BK;
          WG_FENCE;
          wg_imma<64>(tt, desc_sw128(a0), desc_sw128(bh), 0);
          wg_imma<64>(tt, desc_sw128(a0 + 32), desc_sw128(bh + 32), 1);
          wg_imma<64>(tt, desc_sw128(a0 + 64), desc_sw128(bh + 64), 1);
          wg_imma<64>(tt, desc_sw128(a0 + 96), desc_sw128(bh + 96), 1);
          WG_COMMIT;
          WG_WAIT(0);
          acc_fence<32>(tt);
#pragma unroll
          for (int i = 0; i < 32; i++) d[32 * h + i] = (int)((u32)d[32 * h + i] - ((u32)tt[i] << shift));
        }
        if (lane == 0) mbar_arrive(empty(s));
        if (++s == ST) { s = 0; ph ^= 1; }
      }
    }
    if (!a.O) continue;
    const size_t row = (size_t)tm * TILE + r0;
#pragma unroll
    for (int j = 0; j < 16 * NB; j++) {
      const size_t p = row * a.N + tn * TILE + 8 * j + 2 * q, p8 = p + (size_t)8 * a.N;
      int2 lo = make_int2(d[4 * j], d[4 * j + 1]), hi = make_int2(d[4 * j + 2], d[4 * j + 3]);
      if (PROTO && a.dO && !(a.diag & 4)) {
        const int2 d0 = __ldg(reinterpret_cast<const int2*>(a.dO + p)), d1 = __ldg(reinterpret_cast<const int2*>(a.dO + p8));
        lo.x = (int)((u32)lo.x + ((u32)d0.x << 8)); lo.y = (int)((u32)lo.y + ((u32)d0.y << 8));
        hi.x = (int)((u32)hi.x + ((u32)d1.x << 8)); hi.y = (int)((u32)hi.y + ((u32)d1.y << 8));
      }
      *reinterpret_cast<int2*>(a.O + p) = lo;
      *reinterpret_cast<int2*>(a.O + p8) = hi;
    }
  }
}
#endif

// two column-adjacent tiles a CTA (V31_WG_BN 256) wherever the tiles across N pair up
static bool gemm_pair(int tiles_n) { return G_BN == 2 * TILE && tiles_n % 2 == 0; }
template <bool PROTO> static void gemm_launch(bool pair, int grid, cudaStream_t st, const GArgs& g) {
  if (pair) gemm_kernel<PROTO, G_NBW><<<grid, G_NCT, G_SMEM, st>>>(g);
  else gemm_kernel<PROTO, 1><<<grid, G_NCT, G_SMEM, st>>>(g);
}
static void gemm_attrs() {
  CK(cudaFuncSetAttribute(gemm_kernel<true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, G_SMEM));
  CK(cudaFuncSetAttribute(gemm_kernel<false, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, G_SMEM));
  CK(cudaFuncSetAttribute(gemm_kernel<true, G_NBW>, cudaFuncAttributeMaxDynamicSharedMemorySize, G_SMEM));
  CK(cudaFuncSetAttribute(gemm_kernel<false, G_NBW>, cudaFuncAttributeMaxDynamicSharedMemorySize, G_SMEM));
}

// ---------------------------------------------------------------- cheats (a prover the verifier must refuse)
// forge: the T_s0 prefix sums (s0 >= 1) or (s0 = 0) the O values of every stride-th tile, overwritten with hash noise
__global__ void forge_kernel(uint16_t* words, int* O, int S, int s0, int stride, int tiles_n, int N, u32 seed) {
  const size_t tile = (size_t)blockIdx.x * stride;
  const int tm = (int)(tile / tiles_n), tn = (int)(tile % tiles_n);
  for (int c = threadIdx.x; c < TILE * TILE; c += blockDim.x) {
    const u32 h = mix32(seed ^ mix32((u32)(tile * TILE * TILE + c)));
    if (s0 > 0) words[(tile * S + s0 - 1) * TILE * TILE + c] = (uint16_t)h;
    else O[(size_t)(tm * TILE + c / TILE) * N + tn * TILE + c % TILE] = (int)h;
  }
}

// ---------------------------------------------------------------- the committed rows
// pf (hash_after_block's prefetch of the next 128-byte chunk, V31_PF): 1 into L1, 2 into L2
DEV void pf_to(const uint8_t* p, u32 pf) {
  if (pf == 2) asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
  else pf_l1(p);
}
// a row's 4-byte cell c (from c0) at base + (c >> 7) stride + (c & 127) 4: O's row h of a tile (the tile's first cell,
// stride N 4, c0 = h ROW / 4), or a prefix-sum row (stride 512 and c0 = 0: contiguous)
struct CellRow {
  const uint8_t* base;
  u64 stride;
  u32 c0, pfm;
  DEV const uint8_t* at(u32 off) const {
    const u32 c = c0 + off / 4;
    return base + (c >> 7) * stride + (c & 127) * 4;
  }
  DEV uint4 ld16(u32 off) const { return ldg16(at(off)); }
  DEV void pf(u32 off) const { pf_to(at(off), pfm); }
};

// Full-block rows (ROW a multiple of 128): no data block holds padding, and the padding block is a constant whose schedule
// the host computed (C.kw_xpad, the Committer's lb = ROW), so a row is ROW / 128 data blocks and that block after P.
// hash_full takes x from C.mid_P through them, each data block's 16 words from src.block(b, W) at one address a block,
// the next block's line prefetched (src.pf) as its compression starts.
template <class Src>
DEV void hash_full(u64* x, const Src& src, u32 nblk, bool pf) {
  cpy8(x, C.mid_P);
  for (u32 b = 0; b < nblk; ++b) {
    if (pf && b + 1 < nblk) src.pf(b + 1);
    u64 W[16];
    src.block(b, W);
    sha::compress(x, W);
  }
  sha::compress_kw(x, C.kw_xpad);
}
DEV void be_words(const uint4 v, u64* w) {
  w[0] = be64(v.x, v.y);
  w[1] = be64(v.z, v.w);
}
// Block b of a full-block row laid out in 128-cell rows of 4-byte cells: quarter q = q0 + b (32 cells, 128 bytes) at
// p + (q >> 2) rs + (q & 3) 128 (CellRow's and ServedSrc's addresses, since c0 is a multiple of 32)
struct QuarterSrc {
  const uint8_t* p;
  u32 rs, q0, pfm;
  DEV const uint8_t* at(u32 b) const {
    const u32 q = q0 + b;
    return p + (u64)(q >> 2) * rs + (q & 3) * 128;
  }
  DEV void block(u32 b, u64* W) const {
    const uint4* d = reinterpret_cast<const uint4*>(at(b));
#pragma unroll
    for (int c = 0; c < 8; ++c) be_words(__ldg(d + c), W + 2 * c);
  }
  DEV void pf(u32 b) const { pf_to(at(b), pfm); }
};
// ServedSrc's block b of a full-block row: QuarterSrc's quarter, each cell plus sb_j ra_i + sa_i rb_j for the tile's row
// i0 + (q >> 2) and columns j0 + (q & 3) 32 ... A prefix-sum row reads its tile's sums too, masked to 0 by z, so a warp whose
// leaves straddle the two kinds doesn't diverge. ra .. sb are the call's (the kernel's parameters), so they hold no
// per-thread registers across the compression.
struct ServedBlk {
  QuarterSrc d;
  const int *ra, *sa, *rb, *sb;
  u32 i0, j0, z;
  DEV void block(u32 b, u64* W) const {
    const u32 q = d.q0 + b, i = i0 + (q >> 2), j = j0 + (q & 3) * 32;
    const uint4 *v = reinterpret_cast<const uint4*>(d.at(b)), *bp = reinterpret_cast<const uint4*>(rb + j),
                *tp = reinterpret_cast<const uint4*>(sb + j);
    const u32 a = (u32)__ldg(ra + i) & z, s = (u32)__ldg(sa + i) & z;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 o = __ldg(v + c), x = __ldg(bp + c), t = __ldg(tp + c);
      be_words(make_uint4(o.x + t.x * a + s * x.x, o.y + t.y * a + s * x.y, o.z + t.z * a + s * x.z, o.w + t.w * a + s * x.w),
               W + 2 * c);
    }
  }
  DEV void pf(u32 b) const { d.pf(b); }
};
// leaf pos: tile pos / LPT; j = pos % LPT < S ROW_W a prefix-sum row (segment j / ROW_W), else O's row j - S ROW_W. Both kinds
// go through one CellRow, so a warp whose leaves straddle the two (most warps: LPT isn't a multiple of 32) hashes once,
// not once a kind.
// A row source's leaf phase (leaf_persist) takes a row's digest from leaf_digest: full (set by the host for ROW a multiple
// of 128), hash_full's path, else digest's (hash_after_block).
struct V31Row {
  const uint8_t* words;
  const uint8_t* O;
  u64 lpt, srw, N, tiles_n;
  u32 row, S, pf, full;
  DEV void digest(u64 pos, u64* x) const {
    const u64 t = pos / lpt, j = pos % lpt;
    const bool w = j < srw;
    const uint8_t* base = w ? words + t * S * 32768ull + j * row : O + ((t / tiles_n) * TILE * N + (t % tiles_n) * TILE) * 4;
    hash_after_block(C.mid_P, CellRow{base, w ? 512ull : N * 4, w ? 0u : (u32)((j - srw) * row / 4), pf}, row, x, pf != 0);
  }
  DEV void leaf_digest(u64 pos, u64* x) const {
    if (!full) { digest(pos, x); return; }
    const u64 t = pos / lpt, j = pos % lpt;
    const bool w = j < srw;
    const uint8_t* base = w ? words + t * S * 32768ull + j * row : O + ((t / tiles_n) * TILE * N + (t % tiles_n) * TILE) * 4;
    hash_full(x, QuarterSrc{base, w ? 512u : (u32)(N * 4), w ? 0u : (u32)((j - srw) * row / 128), pf}, row / 128, pf != 0);
  }
};
// timing only (the bench's leaf_nomem arm): a row's bytes made in registers, V31Row's compressions without its loads
struct NoMemSrc {
  u32 seed;
  DEV uint4 ld16(u32 off) const { return make_uint4(seed ^ off, seed + off, seed * 3u ^ off, off); }
  DEV void pf(u32) const {}
};
struct NoMemBlk {
  u32 seed;
  DEV void block(u32 b, u64* W) const {
#pragma unroll
    for (int c = 0; c < 8; ++c) be_words(NoMemSrc{seed}.ld16(128 * b + 16 * c), W + 2 * c);
  }
  DEV void pf(u32) const {}
};
struct NoMemRow {
  u32 row, full;
  DEV void digest(u64 pos, u64* x) const { hash_after_block(C.mid_P, NoMemSrc{(u32)pos}, row, x); }
  DEV void leaf_digest(u64 pos, u64* x) const {
    if (full) hash_full(x, NoMemBlk{(u32)pos}, row / 128, false);
    else digest(pos, x);
  }
};
// A no-fold call's prefix-sum rows (S = 1) read from cuBLASLt's int32 D = X W'^T (m x N row-major), its low halves: the row's
// 16-byte piece at off is cells c0 + off / 2 .. + 7 of the tile at base (one tile row's 8 adjacent cells, 32 bytes of D),
// the bytes V31Row reads from the tile-major prefix-sum buffer
struct LtSrc {
  const int* base;
  u64 N;
  u32 c0, pfm;
  DEV const int* at(u32 off) const {
    const u32 c = c0 + off / 2;
    return base + (u64)(c >> 7) * N + (c & 127);
  }
  DEV uint4 ld16(u32 off) const {
    const int* p = at(off);
    const uint4 a = __ldg(reinterpret_cast<const uint4*>(p)), b = __ldg(reinterpret_cast<const uint4*>(p + 4));
    return make_uint4(__byte_perm(a.x, a.y, 0x5410), __byte_perm(a.z, a.w, 0x5410), __byte_perm(b.x, b.y, 0x5410),
                      __byte_perm(b.z, b.w, 0x5410));
  }
  DEV void pf(u32 off) const {   // the next 128 row bytes: 64 cells, 256 bytes of D
    pf_to(reinterpret_cast<const uint8_t*>(at(off)), pfm);
    pf_to(reinterpret_cast<const uint8_t*>(at(off + 64)), pfm);
  }
};
// LtSrc's block b of a full-block row: half-row h = h0 + b (64 cells, 256 bytes of D) at p + (h >> 1) N + (h & 1) 64
struct LtBlk {
  const int* p;
  u64 N;
  u32 h0, pfm;
  DEV const int* at(u32 b) const {
    const u32 h = h0 + b;
    return p + (u64)(h >> 1) * N + (h & 1) * 64;
  }
  DEV void block(u32 b, u64* W) const {
    const uint4* d = reinterpret_cast<const uint4*>(at(b));
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 a = __ldg(d + 2 * c), e = __ldg(d + 2 * c + 1);
      be_words(make_uint4(__byte_perm(a.x, a.y, 0x5410), __byte_perm(a.z, a.w, 0x5410), __byte_perm(e.x, e.y, 0x5410),
                          __byte_perm(e.z, e.w, 0x5410)),
               W + 2 * c);
    }
  }
  DEV void pf(u32 b) const {
    const uint8_t* q = reinterpret_cast<const uint8_t*>(at(b));
    pf_to(q, pfm);
    pf_to(q + 128, pfm);
  }
};
struct LtRow {
  const int* D;
  u64 lpt, N, tiles_n;
  u32 row, pf, full;
  DEV LtSrc src(u64 pos) const {
    const u64 t = pos / lpt, j = pos % lpt;
    return LtSrc{D + (t / tiles_n) * TILE * N + (t % tiles_n) * TILE, N, (u32)(j * row / 2), pf};
  }
  DEV void digest(u64 pos, u64* x) const { hash_after_block(C.mid_P, src(pos), row, x, pf != 0); }
  DEV void leaf_digest(u64 pos, u64* x) const {
    if (!full) { digest(pos, x); return; }
    const u64 t = pos / lpt, j = pos % lpt;
    hash_full(x, LtBlk{D + (t / tiles_n) * TILE * N + (t % tiles_n) * TILE, N, (u32)(j * row / 128), pf}, row / 128, pf != 0);
  }
};
// A served call's rows (dither.Shape(o=True)): per tile its W prefix-sum rows (the prefix-sum buffer, W tile-segments a
// tile), then its O rows: cells c0 + off / 4 .. + 3 of the tile, O = D + tau_j ra_i + sigma_i rb_j mod 2^32 (serve_rows_kernel: rb
// carries K tau_j; D = a' b'^T from cuBLASLt). One source for both kinds, so a warp whose leaves straddle them diverges only
// in its loads; the pointers are the tile's first row and column. LIVE: D or the prefix-sum buffer is being written by the
// kernel that reads it (RP 4's and 5's hash warps), so its loads are coherent (ld.ca), never the read-only path.
template <bool LIVE = false> struct ServedSrcT {
  const uint8_t* w;   // a prefix-sum row, or null: an O row
  const int *D, *ra, *sa, *rb, *sb;
  u64 N;
  u32 c0, pfm;
  DEV const int* at(u32 off) const {
    const u32 c = c0 + off / 4;
    return D + (u64)(c >> 7) * N + (c & 127);
  }
  DEV uint4 ld16(u32 off) const {
    if (w) return LIVE ? __ldca(reinterpret_cast<const uint4*>(w + off)) : ldg16(w + off);
    const u32 c = c0 + off / 4, r = c >> 7, col = c & 127;
    const uint4* dp = reinterpret_cast<const uint4*>(D + (u64)r * N + col);
    const uint4 d = LIVE ? __ldca(dp) : __ldg(dp);
    const uint4 b = __ldg(reinterpret_cast<const uint4*>(rb + col)), t = __ldg(reinterpret_cast<const uint4*>(sb + col));
    const u32 a = (u32)__ldg(ra + r), s = (u32)__ldg(sa + r);
    return make_uint4(d.x + t.x * a + s * b.x, d.y + t.y * a + s * b.y, d.z + t.z * a + s * b.z, d.w + t.w * a + s * b.w);
  }
  DEV void pf(u32 off) const { pf_to(w ? w + off : reinterpret_cast<const uint8_t*>(at(off)), pfm); }
};
using ServedSrc = ServedSrcT<>;
struct ServedRow {
  const uint8_t* words;
  const int *D, *ra, *sa, *rb, *sb;
  u64 lpt, wrows, N, tiles_n;
  u32 row, W, pf, full;
  template <bool LIVE = false> DEV ServedSrcT<LIVE> src(u64 pos) const {
    const u64 t = pos / lpt, j = pos % lpt, tm = t / tiles_n, tn = t % tiles_n;
    const bool w = j < wrows;
    return ServedSrcT<LIVE>{w ? words + t * W * 32768ull + j * row : nullptr, D + tm * TILE * N + tn * TILE, ra + tm * TILE,
                            sa + tm * TILE, rb + tn * TILE, sb + tn * TILE, N, w ? 0u : (u32)((j - wrows) * row / 4), pf};
  }
  DEV void digest(u64 pos, u64* x) const { hash_after_block(C.mid_P, src(pos), row, x, pf != 0); }
  DEV void leaf_digest(u64 pos, u64* x) const {
    if (!full) { digest(pos, x); return; }
    const u64 t = pos / lpt, j = pos % lpt, tm = t / tiles_n, tn = t % tiles_n;
    const bool w = j < wrows;
    const QuarterSrc d{w ? words + t * W * 32768ull + j * row : reinterpret_cast<const uint8_t*>(D + tm * TILE * N + tn * TILE),
                       w ? 512u : (u32)(N * 4), w ? 0u : (u32)((j - wrows) * row / 128), pf};
    hash_full(x, ServedBlk{d, ra, sa, rb, sb, (u32)(tm * TILE), (u32)(tn * TILE), w ? 0u : ~0u}, row / 128, pf != 0);
  }
};

// hash_after_block's SHA-512 chunks b0 .. of the row (x: the chaining value after the first b0), at most budget of them,
// until stop() holds before a chunk past the first: the chunks done, nblk when x is the digest
struct NoStop {
  DEV bool operator()() const { return false; }
};
DEV u32 row_blocks(u32 lb) { return (lb + 17 + 127) / 128; }
template <class Src, class Stop>
DEV u32 hash_blocks(u64* x, const Src& src, u32 lb, u32 b0, bool pf, const Stop& stop, u32 budget = ~0u) {
  const u32 nblk = row_blocks(lb), end = budget >= nblk - b0 ? nblk : b0 + budget;
  for (u32 b = b0; b < end; ++b) {
    if (b > b0 && stop()) return b;
    if (pf && 128 * (b + 1) < lb) src.pf(128 * (b + 1));
    u64 W[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const u32 off = 128 * b + 16 * c;
      const uint4 v = off < lb ? src.ld16(off) : make_uint4(0, 0, 0, 0);
      W[2 * c] = be64(v.x, v.y);
      W[2 * c + 1] = be64(v.z, v.w);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const u32 base = 128 * b + 8 * i;
      if (base + 8 > lb) {
        if (base <= lb) {
          const u32 j = lb - base;
          W[i] = (j ? W[i] & (~0ull << (64 - 8 * j)) : 0) | (0x80ull << (56 - 8 * j));
        } else {
          W[i] = 0;
        }
      }
    }
    if (b == nblk - 1) W[15] = (u64)(128 + lb) * 8;
    sha::compress(x, W);
  }
  return end;
}
// RP 4's and 5's hasher (wg_proto.cuh's H) on a served call's rows off .. off + lpo - 1 of each commit tile: row j of commit
// tile ct is leaf ct lpt + off + j (off wrows: its O rows, in the GEMM of D; 0: every row, in the GEMM of the prefix sums,
// which runs after it). Its chaining value goes to xd[pos] and flag[pos] = epoch << 10 | the SHA-512 chunks hashed (all of
// them, or as many as the hash warps did before the consumers were done); FusedRow takes the leaf phase on from there, and
// hashes from the first chunk whatever carries another epoch (rows no hash warp reached, a forged copy). A visit hashes at
// most budget chunks on from the row's own flag (chunk, V31_HCHUNK: the hash warps' visits a row, 0 a row whole in one)
constexpr u32 FUSED_EPOCHS = 1u << 22;
struct FusedLeaves {
  ServedRow row;
  u64* xd;
  u32* flag;
  u32 epoch, lpo, chunk, off;
  DEV unsigned lpt() const { return lpo; }
  template <class Stop> DEV bool operator()(unsigned long long ct, unsigned j, const Stop& stop, unsigned budget) const {
    const u64 pos = ct * row.lpt + off + j;
    const u32 nblk = row_blocks(row.row), f = __ldcg(flag + pos);
    u32 b0 = 0;
    u64 x[8];
    if (f >> 10 == epoch) {
      b0 = f & 1023;
      if (b0 == nblk) return true;
      ld8cg(xd + 8 * pos, x);
    } else {
      cpy8(x, C.mid_P);
    }
    const u32 b = hash_blocks(x, row.template src<true>(pos), row.row, b0, row.pf != 0, stop, budget);
    if (b == b0) return false;
    st8(xd + 8 * pos, x);
    flag[pos] = epoch << 10 | b;
    return b == nblk;
  }
};
struct FusedRow {
  ServedRow row;
  const u64* xd;
  const u32* flag;
  u32 epoch;
  DEV void digest(u64 pos, u64* x) const {
    const u32 f = __ldg(flag + pos);
    u32 b = 0;
    if (f >> 10 == epoch) {
      ld8(xd + 8 * pos, x);
      b = f & 1023;
    } else {
      cpy8(x, C.mid_P);
    }
    hash_blocks(x, row.src(pos), row.row, b, row.pf != 0, NoStop{});
  }
  DEV void leaf_digest(u64 pos, u64* x) const { digest(pos, x); }
};

// ---------------------------------------------------------------- the commit in at most cap CTAs an SM
// commit_gpu's small path launches a CTA for every 128 leaves, and up to 5 of them sit on an SM, so a GEMM CTA launched
// beside the commit waits for an SM to drain while the commit's next CTAs take each freed slot. salt_persist and
// leaf_persist do the same work (salt_small's and leaf_small's, group by group) in at most cap CTAs an SM: a CTA claims
// one of its SM's slots (claim[smid] = epoch << 8 | count, epoch new each launch) or leaves at once, and the claimants take
// 128-leaf groups from ctl[0] until none are left. The last CTA out zeroes the counters for the next launch; launches that
// share ctl run in one stream's order.
struct Persist {
  u32* ctl;     // [0] the next group, [1] the CTAs out, [2 + b] CTA b's current group
  u32* claim;   // a word an SM (smid & 1023)
  u32 epoch, cap, nvb;
};
DEV u32 smid() {
  u32 s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
DEV bool persist_claim(const Persist& p) {
  bool got = false;
  if (threadIdx.x == 0) {
    u32* w = p.claim + (smid() & 1023);
    u32 old = *reinterpret_cast<volatile u32*>(w);
    for (;;) {
      const u32 c = old >> 8 == p.epoch ? old & 255 : 0;
      if (c >= p.cap) break;
      const u32 seen = atomicCAS(w, old, p.epoch << 8 | (c + 1));
      if (seen == old) { got = true; break; }
      old = seen;
    }
  }
  return __syncthreads_or(got);
}
// the CTA's next group (nvb or more: none left), the same for every thread
DEV u32 persist_next(const Persist& p) {
  volatile u32* slot = p.ctl + 2 + blockIdx.x;
  __syncthreads();
  if (threadIdx.x == 0) *slot = atomicAdd(p.ctl, 1u);
  __syncthreads();
  return *slot;
}
DEV void persist_done(const Persist& p) {
  if (threadIdx.x == 0 && atomicAdd(p.ctl + 1, 1u) == gridDim.x - 1) { p.ctl[0] = 0; p.ctl[1] = 0; }
}
// fold_cta for group vb
DEV void fold_vb(int dom, u64* levels, u64 width, u32 cnt, int L, u64 vb) {
  const u32 t = threadIdx.x;
  for (int l = 1; l <= L; ++l) {
    __syncthreads();
    const u32 m = cnt >> l;
    if (t < m) {
      const u64 i = vb * m + t;
      const u64* in = level_at(levels, width, l - 1);
      u64 Lc[8], Rc[8], o[8];
      ld8(in + 16 * i, Lc);
      ld8(in + 16 * i + 8, Rc);
      node(dom, l - 1, i, Lc, Rc, o);
      st8(level_at(levels, width, l) + 8 * i, o);
    }
  }
}
// Ks as a launch's argument (C.ks and C.ks_m8's values), for a salt phase on its own stream: no copy into C races it
struct KsArg {
  u64 ks[8], ks_m8[8];
};
// salt_prf's block r (salt words 8 r .. 8 r + 7) into Y, Ks from k when given, else from C
DEV void salt_prf_block(const KsArg* k, u64 pos, int r, u64* Y) {
  u64 w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = k ? k->ks[i] : C.ks[i];
  w[8] = pos; w[9] = r; w[10] = 0x8000000000000000ull;
  w[11] = w[12] = w[13] = w[14] = 0;
  w[15] = 80 * 8;
  sha::iv(Y);
  sha::compress_from<8>(Y, k ? k->ks_m8 : C.ks_m8, w);
}
// hm96_mask<false, true>'s terms from salt bytes 64 B .. 64 B + 63 (Yb, salt_prf's block B: q = 4 B .. 4 B + 3) xored into m
template <int B>
DEV void hm96_mask_block(const uint8_t* T, const u64* Yb, u32* m) {
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    u32 A[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) A[i] = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int p = 16 * j + r;
      const u32 v = (u32)(Yb[p >> 3] >> (56 - 8 * (p & 7))) & 0xFFu;
      const uint8_t* lo = T + 256 * (4 * B + j) + 16 * (v & 15u);
      const uint8_t* hi = T + 4096 + 256 * (4 * B + j) + 16 * (v >> 4);
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const uint4 u = ldg16(lo + 256 * i), w = ldg16(hi + 256 * i);
        A[4 * i] ^= u.x ^ w.x; A[4 * i + 1] ^= u.y ^ w.y; A[4 * i + 2] ^= u.z ^ w.z; A[4 * i + 3] ^= u.w ^ w.w;
      }
    }
    const int sw = r >> 2, s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= s ? __funnelshift_r(A[w + sw], A[w + sw + 1], s) : A[w + sw];
  }
}
// hm96_mask_block from T8, the nibble table as 8-byte halves of its 16-byte entries in shared memory: T8[512 h + 32 c + 16 d + u]
// is half d of chunk c's entry for nibble u (h 1: high nibbles), so a half-warp's lookups in one chunk hit 16 distinct
// bank pairs. The four bytes r = 4 sw + rr (rr fixed) share the shift 8 rr, so their rows go into one accumulator D,
// row sw at word offset -sw, and D is shifted once: m[w] ^= (D[w + 1] : D[w]) >> 8 rr, with D[x] = xor over sw of row sw's
// word x + sw (x 0 .. 16; the words outside never reach m)
template <int B>
DEV void hm96_mask_block_h(const u64* T8, const u64* Yb, u32* m) {
#pragma unroll 1
  for (int rr = 0; rr < 4; ++rr) {
    u32 D[17];
#pragma unroll
    for (int x = 0; x < 17; ++x) D[x] = 0;
    const int sh = 24 - 8 * rr;
#pragma unroll
    for (int sw = 0; sw < 4; ++sw) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const u64 y = Yb[2 * j + (sw >> 1)];
        const u32 t = ((sw & 1) ? (u32)y : (u32)(y >> 32)) >> sh;   // salt byte 16 j + 4 sw + rr of the block in bits 0 .. 7
        const u64* lo = T8 + 32 * (4 * B + j) + (t & 15u);
        const u64* hi = T8 + 512 + 32 * (4 * B + j) + ((t >> 4) & 15u);
#pragma unroll
        for (int i = 0; i < 5; ++i)
#pragma unroll
          for (int d = 0; d < 2; ++d) {
            const int x = 4 * i + 2 * d - sw;
            if (x + 1 < 0 || x > 16) continue;
            const u64 a = lo[32 * i + 16 * d], b = hi[32 * i + 16 * d];
            if (x >= 0) D[x] ^= (u32)a ^ (u32)b;
            if (x + 1 <= 16) D[x + 1] ^= (u32)(a >> 32) ^ (u32)(b >> 32);
          }
      }
    }
#pragma unroll
    for (int w = 0; w < 16; ++w) m[w] ^= __funnelshift_r(D[w], D[w + 1], 8 * rr);
  }
}
// salt_one<true, false, true> at the call's salt offset: S[pos] from prf_salt(Ks, off + pos) (dither.salts(Ks, off)); Ks
// from k when given, else from C. Each of the PRF's three blocks goes into the mask and the digest as it is made, so the
// salt's 24 words are never live at once (hm96_mask over all of them spilled). MK: the mask from T as
// hm96_mask_block_h's halves in shared memory (0), from the nibble table T through L1 (2), or no mask at all (1: timing
// only, S's mask words 0).
template <int MK = 0>
DEV void salt_one_at(u64 pos, u64 off, const uint8_t* T, u64* __restrict__ S, const KsArg* k = nullptr) {
  u64 Y[16], c[8];
  u32 m[16];
#pragma unroll
  for (int w = 0; w < 16; ++w) m[w] = 0;
  const auto mask = [&](auto b, const u64* Yb) {
    if constexpr (MK == 0) hm96_mask_block_h<decltype(b)::value>(reinterpret_cast<const u64*>(T), Yb, m);
    else if constexpr (MK == 2) hm96_mask_block<decltype(b)::value>(T, Yb, m);
  };
  salt_prf_block(k, off + pos, 0, Y);
  mask(std::integral_constant<int, 0>{}, Y);
  salt_prf_block(k, off + pos, 1, Y + 8);
  mask(std::integral_constant<int, 1>{}, Y + 8);
  cpy8(c, C.mid_salt);
  sha::compress(c, Y);
  salt_prf_block(k, off + pos, 2, Y);
  mask(std::integral_constant<int, 2>{}, Y);
  Y[8] = 0x8000000000000000ull;
#pragma unroll
  for (int i = 9; i < 15; ++i) Y[i] = 0;
  Y[15] = (128 + 192) * 8;
  sha::compress(c, Y);
  u64 my[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) my[i] = (u64)bswap32(m[2 * i]) << 32 | bswap32(m[2 * i + 1]);
  st8(S + 16 * pos, my);
  st8(S + 16 * pos + 8, c);
}
// salt_small<true> (slot 0) a group at a time: S and F for positions below n (salted at off + pos), the pads up to width,
// and fl levels of the subtrees of groups that hold only pads (nvb = ceil(width / 128); fl 0 when the leaf phase's level
// kernels fold every group, Lanes::lfold); KP: Ks from ka, else from C; MINB its CTAs an SM by registers (4: 128 a thread,
// as many CTAs as a cap of 4 lets in anyway; 5: 96, spilling); MK, salt_one_at's mask (0: the CTA copies the nibble table
// into shared memory as hm96_mask_block_h's halves)
template <bool KP = false, int MINB = 4, int MK = 0>
__global__ void __launch_bounds__(SMALL_THREADS, MINB)
    salt_persist(Persist p, int dom, const uint4* __restrict__ Tg, u64 n, u64 width, u64* __restrict__ S, u64* __restrict__ F,
                 u64* levels, u64 off, int fl, const __grid_constant__ KsArg ka) {
  __shared__ u64 T8[MK == 0 ? 1024 : 1];
  if constexpr (MK == 0) {
    const u64* Tg8 = reinterpret_cast<const u64*>(Tg);
    for (u32 i = threadIdx.x; i < 1024; i += SMALL_THREADS)   // T8[512 h + 32 c + 16 d + u] = Tn's u64 512 h + 32 c + 2 u + d
      T8[i] = Tg8[(i & ~511u) + 32 * ((i >> 5) & 15) + 2 * (i & 15) + ((i >> 4) & 1)];
    __syncthreads();
  }
  const uint8_t* T = MK == 0 ? reinterpret_cast<const uint8_t*>(T8) : reinterpret_cast<const uint8_t*>(Tg);
  const u32 cnt = (u32)(width < SMALL_THREADS ? width : SMALL_THREADS);
  if (persist_claim(p))
    for (u32 vb; (vb = persist_next(p)) < p.nvb;) {
      const u64 pos = (u64)vb * SMALL_THREADS + threadIdx.x;
      if (pos < n) {
        salt_one_at<MK>(pos, off, T, S, KP ? &ka : nullptr);
        u64 f[8];
        frame_first(dom, pos, f);
        st8(F + 8 * pos, f);
      } else if (pos < width) {
        u64 o[8];
        pad(dom, pos, o);
        st8(levels + 8 * pos, o);
      }
      if ((u64)vb * cnt >= n) fold_vb(dom, levels, width, cnt, fl, vb);
    }
  persist_done(p);
}
// leaf_small (g = 1, no bc, no fused top) a group at a time (nvb = ceil(n / 128)); MINB its CTAs an SM by registers (4:
// 128 a thread, for a leaf phase that has the SMs to itself); fl, the group's levels the CTA folds itself (log2 of its
// leaves, or 0 when level kernels fold them after, Lanes::lfold)
template <class Row, int MINB = SMALL_MINB>
__global__ void __launch_bounds__(SMALL_THREADS, MINB)
    leaf_persist(Persist p, int dom, Row row, const u64* __restrict__ S, const u64* __restrict__ F, u64 n, u64 width,
                 u64* levels, int fl) {
  const u32 cnt = (u32)(width < SMALL_THREADS ? width : SMALL_THREADS);
  if (persist_claim(p))
    for (u32 vb; (vb = persist_next(p)) < p.nvb;) {
      const u64 pos = (u64)vb * SMALL_THREADS + threadIdx.x;
      if (pos < n) {
        u64 x[8], out[8];
        row.leaf_digest(pos, x);
        leaf_from_digest<0>(dom, pos, x, S, out, nullptr, F);
        st8(levels + 8 * pos, out);
      }
      fold_vb(dom, levels, width, cnt, fl, vb);
    }
  persist_done(p);
}

// ---------------------------------------------------------------- synthetic operands
// v31.py's `llm` operands on the device: rows of Student-t(3) activations or Gaussian weights quantized to +-127 by their
// absolute maximum (a CTA a row; the values regenerated from their hash for the second pass)
DEV float unif(u32 seed, u64 idx) { return ((float)mix32(seed ^ mix32((u32)idx ^ mix32((u32)(idx >> 32)))) + 0.5f) * 2.3283064e-10f; }
DEV float gauss(u32 seed, u64 idx) {
  const float u1 = unif(seed, 2 * idx), u2 = unif(seed ^ 0x5bd1e995u, 2 * idx + 1);
  return sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
}
DEV float gen_value(u32 seed, u64 idx, bool t3) {
  const float z = gauss(seed, 4 * idx);
  if (!t3) return z;
  const float a = gauss(seed, 4 * idx + 1), b = gauss(seed, 4 * idx + 2), c = gauss(seed, 4 * idx + 3);
  return z / sqrtf((a * a + b * b + c * c) / 3.f);
}
__global__ void gen_kernel(int8_t* out, int K, u32 seed, bool t3) {
  __shared__ float red[32];
  const u64 row = blockIdx.x;
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(gen_value(seed, row * K + k, t3)));
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    mx = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
    if (threadIdx.x == 0) red[0] = mx;
  }
  __syncthreads();
  mx = fmaxf(red[0], 1e-30f);
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    out[row * K + k] = (int8_t)fminf(127.f, fmaxf(-127.f, rintf(127.f * gen_value(seed, row * K + k, t3) / mx)));
}

// ---------------------------------------------------------------- host
static CUtensorMap tmap(const void* p, int depth, int rows, int box_rows) {
  CUtensorMap m;
  const cuuint64_t dims[2] = {(cuuint64_t)depth, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)depth};
  const cuuint32_t box[2] = {128, (cuuint32_t)box_rows}, es[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(p), dims, strides, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled: %d\n", int(r)); exit(3); }
  return m;
}
// wg_proto.cuh's TST store maps: rows of `cols` cells of es bytes, boxes of 128 bytes by 64 rows, SWIZZLE_128B
static CUtensorMap tmap_st(const void* p, CUtensorMapDataType dt, int es, u64 cols, u64 rows) {
  CUtensorMap m;
  const cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)(cols * es)};
  const cuuint32_t box[2] = {(cuuint32_t)(128 / es), 64}, ess[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, dt, 2, const_cast<void*>(p), dims, strides, box, ess, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled (store): %d\n", int(r)); exit(3); }
  return m;
}

struct Lt {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatmulAlgo_t algo;
  void* ws;
  size_t wsz = 64u << 20;
  int algo_id = -1, tile = -1, stages = -1, splitk = -1, ncand = 0;
};
// D (m x n row-major) = A (m x k) B^T: cuBLASLt's column-major (n, m, k), its A our B (op T), its B our A; autotuned over the
// heuristic's candidates (5 calls each), or V31_LT_TUNE=0 the first that runs (a profiler's pass: none of the candidates')
static void lt_setup(Lt& L, int m, int n, int k, const int8_t* A, const int8_t* B, int* D, cudaStream_t st) {
  CL(cublasLtCreate(&L.h));
  CL(cublasLtMatmulDescCreate(&L.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CL(cublasLtMatrixLayoutCreate(&L.la, CUDA_R_8I, k, n, k));
  CL(cublasLtMatrixLayoutCreate(&L.lb, CUDA_R_8I, k, m, k));
  CL(cublasLtMatrixLayoutCreate(&L.ld, CUDA_R_32I, n, m, n));
  CK(cudaMalloc(&L.ws, L.wsz));
  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &L.wsz, sizeof L.wsz));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  CL(cublasLtMatmulAlgoGetHeuristic(L.h, L.op, L.la, L.lb, L.ld, L.ld, pref, 16, res, &got));
  if (got == 0) { fprintf(stderr, "cuBLASLt: no algorithm\n"); exit(3); }
  L.ncand = got;
  const int32_t one = 1, zero = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float best = 1e30f;
  int bi = -1;
  const char* tv = getenv("V31_LT_TUNE");
  const bool tune = !tv || atoi(tv) != 0;
  for (int i = 0; i < got && (tune || bi < 0); i++) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    bool ok = true;
    for (int w = 0; w < (tune ? 2 : 1) && ok; w++)
      ok = cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, L.ws, L.wsz, st) == CUBLAS_STATUS_SUCCESS;
    if (!ok) continue;
    if (!tune) { bi = i; break; }
    CK(cudaEventRecord(e0, st));
    for (int r = 0; r < 5; r++) cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, L.ws, L.wsz, st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) { best = ms; bi = i; }
  }
  if (bi < 0) { fprintf(stderr, "cuBLASLt: no algorithm ran\n"); exit(3); }
  L.algo = res[bi].algo;
  size_t w;
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_ID, &L.algo_id, sizeof L.algo_id, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &L.tile, sizeof L.tile, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &L.stages, sizeof L.stages, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &L.splitk, sizeof L.splitk, &w);
  CL(cublasLtMatmulPreferenceDestroy(pref));
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
}
static void lt_run(Lt& L, const int8_t* A, const int8_t* B, int* D, cudaStream_t st) {
  const int32_t one = 1, zero = 0;
  CL(cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &L.algo, L.ws, L.wsz, st));
}
static void lt_free(Lt& L) {
  cublasLtMatrixLayoutDestroy(L.la);
  cublasLtMatrixLayoutDestroy(L.lb);
  cublasLtMatrixLayoutDestroy(L.ld);
  cublasLtMatmulDescDestroy(L.op);
  cublasLtDestroy(L.h);
  cudaFree(L.ws);
}

// The second baseline: CUTLASS's int8 GEMMs that sm_120 runs (the harness's cutlass_2x.cu, kind 7: mma.sync IMMA.16832
// at four tile shapes; linked in under -DV31_CUTLASS), the fastest of those that implement the shape (5 calls each), D =
// A B^T as lt computes it
struct Cut {
  phc_prepared* best = nullptr;
  void* ws = nullptr;
  std::string name = "none";
  int ncand = 0;
};
static void cut_setup(Cut& C, int m, int n, int k, const int8_t* A, const int8_t* B, int* D, cudaStream_t st) {
#ifdef V31_CUTLASS
  int count = 0;
  const phc_entry* tab = phc_table_2x(&count);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float best = 1e30f;
  for (int i = 0; i < count; i++) {
    if (tab[i].kind != 7 || !tab[i].query) continue;
    const phc_args a{m, n, k, A, B, nullptr, nullptr, D};
    size_t ws = 0, sfa = 0, sfb = 0;
    if (tab[i].query(&a, &ws, &sfa, &sfb)) continue;
    void* w = nullptr;
    if (ws) CK(cudaMalloc(&w, ws));
    phc_prepared* p = nullptr;
    if (tab[i].prepare(&a, w, ws, st, &p)) { if (w) CK(cudaFree(w)); continue; }
    C.ncand++;
    bool ok = p->launch(st) == 0 && p->launch(st) == 0;
    float ms = 1e30f;
    if (ok) {
      CK(cudaEventRecord(e0, st));
      for (int r = 0; r < 5; r++) ok = p->launch(st) == 0 && ok;
      CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1));
      CK(cudaEventElapsedTime(&ms, e0, e1));
    }
    if (ok && ms < best) {
      if (C.best) { delete C.best; if (C.ws) CK(cudaFree(C.ws)); }
      C.best = p; C.ws = w; C.name = tab[i].name; best = ms;
    } else {
      delete p;
      if (w) CK(cudaFree(w));
    }
  }
  CK(cudaGetLastError());
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
#else
  (void)C; (void)m; (void)n; (void)k; (void)A; (void)B; (void)D; (void)st;
#endif
}
static void cut_run(Cut& C, cudaStream_t st) {
  if (C.best->launch(st)) { fprintf(stderr, "CUTLASS's %s didn't launch\n", C.name.c_str()); exit(3); }
}
__global__ void ne_kernel(const int* a, const int* b, size_t n, unsigned long long* out) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += a[i] != b[i];
  if (c) atomicAdd(out, c);
}
// FusedLeaves' flags of epoch e: out[0] the rows hashed whole (nblk chunks), out[1] the chunks hashed
__global__ void fused_count_kernel(const u32* a, size_t n, u32 e, u32 nblk, unsigned long long* out) {
  unsigned long long c = 0, nb = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    if (a[i] >> 10 == e) { c += (a[i] & 1023) == nblk; nb += a[i] & 1023; }
  if (c) atomicAdd(out, c);
  if (nb) atomicAdd(out + 1, nb);
}

template <class T> static T* dalloc(size_t n) {
  T* p = nullptr;
  CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

struct Cheat {
  int kind = 0;   // 0 none, 1 skip, 2 forge (T_s0; Fill: s0 0 O), 3 forgeO, 4 nodither (Fill: d_A and d_B zero)
  int tau = 0, s0 = 0;
  double frac = 1.0;
};
static Cheat parse_cheat(const std::string& s) {
  Cheat c;
  if (s == "none" || s.empty()) return c;
  if (s == "nodither") { c.kind = 4; return c; }
  if (!s.compare(0, 5, "skip:")) { c.kind = 1; c.tau = atoi(s.c_str() + 5); return c; }
  if (!s.compare(0, 6, "forge:")) {
    c.kind = 2;
    c.s0 = atoi(s.c_str() + 6);
    const char* f = strchr(s.c_str() + 6, ':');
    if (f) c.frac = atof(f + 1);
    return c;
  }
  if (!s.compare(0, 6, "forgeO")) {
    c.kind = 3;
    const char* f = strchr(s.c_str(), ':');
    if (f) c.frac = atof(f + 1);
    return c;
  }
  fprintf(stderr, "cheat %s: none | skip:TAU | forge:S0[:FRAC] | forgeO[:FRAC] | nodither\n", s.c_str());
  exit(2);
}

static int env_int(const char* name, int dflt) {
  const char* v = getenv(name);
  return v && *v ? atoi(v) : dflt;
}
static std::string env_str(const char* name, const char* dflt) {
  const char* v = getenv(name);
  return v && *v ? v : dflt;
}
// The knobs a bench line records. V31_CARVE: every kernel's preferred shared-memory carveout (percent; < 0 leaves the
// driver's choice), so commit_gpu's small-path CTAs (no shared memory) can sit beside a GEMM CTA; V31_DYN: the GEMMs (V31_GEMM
// 1's too, wgl included) take their tiles from a counter; V31_CPRIO: the commit stream's priority against the compute stream's (hi, lo or eq); V31_WRAP:
// the wrap correction's kernel (1: wrap_kernel, 2: wrap2_kernel, 3: wrap3_kernel, 4: wrap4_kernel; 5: X w_W^T on the
// tensor cores, gemm_kernel<false> on X and w_W (form_b writes w_W), then wrap_a_kernel; 6, V31_GEMM 1 only: X w_W^T in
// the protocol GEMM's own accumulator, as X (-2 w_W)^T k-tiles before its tail's x(-128), and wrap_a_kernel stores the
// rest, w_X B~^T, as dO; 7: as 6, and w_X B~^T in the GEMM's epilogue too, a wrap of X at a time from form_a's lists,
// so no wrap kernel runs and no dO is read), V31_WCW its columns a CTA (256 where N allows, or 128; 1 to 4);
// V31_CCTA: the salt and leaf phases' CTAs an SM (salt_persist, leaf_persist; 0: commit_gpu's salt_small and leaf_small);
// V31_PF: each row's next 128-byte chunk prefetched as its compression starts (0 none, 1 into L1, 2 into L2); V31_DIAG 1:
// the bench's timing-only gemm_no* arms; V31_SCHED: the overlapped calls' order (Ctx::call_ovl, Fill::pend's 1 and 2).
// Under -DV31_WG (sm_90a):
// V31_GEMM 1, the protocol's and the plain GEMM on warpgroup MMAs (wg_proto.cuh, the bench's wgl arm wgmma.linear itself),
// V31_WGBN its tiles' columns (256 where N allows, or 128), V31_CM where it keeps the tail's low halves (0 registers, 1 an
// L2 scratch slot a CTA), V31_WGG its raster's tile rows a group, V31_RP 3 the register plan that leaves room for a
// commit CTA beside each GEMM CTA (wg_proto.cuh's gemm_r3; CM 1; the wgl arm keeps RP 0). A served Fill's V31_GEMM 1: its
// D = a' b'^T on the plain wg GEMM (static schedule), V31_RP 4 hashing the O rows in the GEMM's producer warpgroup
// (gemm_r4, FusedLeaves), V31_RP 5 in a hash warpgroup beside it too (gemm_r5; V31_R5H, one of wg_proto.cuh's R5_SPLITS:
// the producer and hash warpgroups' registers, the consumers' 256 less; V31_R5W 0: not in the GEMM of the prefix sums,
// whose RP 5 hashes the prefix-sum rows and whatever O rows the GEMM of D left; V31_TST 1: RP 5's O and prefix sums stored
// through shared memory with TMA, a hash warp issuing the stores); V31_HALL 1: every row there, past the
// consumers' end, timing the hash warps alone, -1: none, timing the GEMM alone in that CTA; V31_HCHUNK
// C > 0: the hash warps' passes over the posted rows, C chunks of each a visit, in place of each row whole in turn;
// V31_WGW 0: the prefix-sum segments on gemm_kernel<true> rather than the wg protocol GEMM).
// A Fill's commit: V31_LMINB 4, 5 (the default), 6 or 8, its leaf phase's CTAs an SM by registers (leaf_persist's MINB: 128,
// 96, 80 or 64 a thread); V31_SAHEAD 1, a call's salt phase on a stream of its own (Ks a launch argument), issued with the
// call, so it runs beside the previous call's GEMM or leaves rather than between its own GEMM and leaves; V31_FULL 0, rows of
// a multiple of 128 bytes hashed as any row (hash_after_block's padding checks) rather than as full blocks;
// V31_SMINB 4 (the default) or 5, salt_persist's MINB; V31_LFOLD 1, a group's levels (1 .. 7) folded by the group's CTA
// (the leaf phase's for a group of leaves, the salt phase's for a group of pads alone), whose other warps wait while a few
// fold, rather than (0, the default) by a level_kernel launch a level, over the whole width, after the leaf phase.
// V31_MASK: the salt phase's hm96 mask (salt_persist's MK): 0 its nibble table in shared memory as hm96_mask_block_h's
// halves, 2 the nibble table through L1, 1 none (timing only; wrong roots); 1 and 2 at V31_SMINB 4 only.
// V31_GSM N > 0: a Fill's calls on a Split, the compute on N SMs and the commit on the rest, each call's commit beside the
// next calls' compute (V31_SCHED 0's order whatever V31_SCHED says); V31_GCOINS 1 there, a call's coins on the commit's
// SMs too, after the previous call's form. V31_GSM_BY "NxK:G_NxK:G ...": a Fill of that N and K on G (0: no split) in
// place of V31_GSM.
struct Knobs {
  int carve = env_int("V31_CARVE", 100), dyn = env_int("V31_DYN", 1), wrap = env_int("V31_WRAP", 2), wcw = env_int("V31_WCW", 256),
      ccta = env_int("V31_CCTA", 1), pf = env_int("V31_PF", 0), diag = env_int("V31_DIAG", 0), sched = env_int("V31_SCHED", 0),
      gemm = env_int("V31_GEMM", 0), wgbn = env_int("V31_WGBN", 256), cm = env_int("V31_CM", 0), wgg = env_int("V31_WGG", 16),
      rp = env_int("V31_RP", 0), lminb = env_int("V31_LMINB", SMALL_MINB), sahead = env_int("V31_SAHEAD", 0),
      gsm = env_int("V31_GSM", 0), gcoins = env_int("V31_GCOINS", 0),
      ltw = env_int("V31_LTW", 0),   // Fill at S = 1: cuBLASLt's int32 X W'^T in place of the protocol GEMM, rows its low halves
      full = env_int("V31_FULL", 1), sminb = env_int("V31_SMINB", 4), lfold = env_int("V31_LFOLD", 0),
      mask = env_int("V31_MASK", 0);
  std::string cprio = env_str("V31_CPRIO", "hi");
  int gsm_of(int n, int k) const {
    std::string s = env_str("V31_GSM_BY", "");
    for (char& c : s)
      if (c == '_' || c == ',') c = ' ';
    std::istringstream is(s);
    for (std::string e; is >> e;) {
      int en, ek, g;
      if (sscanf(e.c_str(), "%dx%d:%d", &en, &ek, &g) != 3) { fprintf(stderr, "V31_GSM_BY: bad entry %s\n", e.c_str()); exit(2); }
      if (en == n && ek == k) return g;
    }
    return gsm;
  }
};
static void set_carveout(int pct) {
  if (pct < 0) return;
  const void* ks[] = {(const void*)gemm_kernel<true, 1>,  (const void*)gemm_kernel<false, 1>,
                      (const void*)gemm_kernel<true, G_NBW>,  (const void*)gemm_kernel<false, G_NBW>,  (const void*)wrap_kernel,
                      (const void*)wrap2_kernel<128>,  (const void*)wrap2_kernel<256>,   (const void*)wrap3_kernel<128>,
                      (const void*)wrap3_kernel<256>,  (const void*)wrap4_kernel<128>,  (const void*)wrap4_kernel<256>,
                      (const void*)wrap_a_kernel<false>, (const void*)wrap_a_kernel<true>, (const void*)form_a_kernel,
                      (const void*)form_b_kernel<0>,   (const void*)form_b_kernel<1>,    (const void*)form_b_kernel<2>,
                      (const void*)bits_kernel,        (const void*)noise_kernel,
                      (const void*)perm_kernel,        (const void*)salt_small<true>,    (const void*)leaf_small<V31Row, 0>,
                      (const void*)level_kernel,       (const void*)top_small,           (const void*)salt_persist<false>,
                      (const void*)salt_persist<true>, (const void*)leaf_persist<FusedRow, 4>, (const void*)leaf_persist<ServedRow, 4>,
                      (const void*)leaf_persist<V31Row>, (const void*)leaf_small<NoMemRow, 0>, (const void*)leaf_persist<NoMemRow>,
                      (const void*)leaf_small<LtRow, 0>, (const void*)leaf_persist<LtRow>,  (const void*)fill_a_kernel,
                      (const void*)fill_b_kernel,      (const void*)leaf_small<ServedRow, 0>, (const void*)leaf_persist<ServedRow>,
                      (const void*)serve_rows_kernel,  (const void*)leaf_persist<FusedRow>,
                      (const void*)leaf_persist<V31Row, 4>, (const void*)leaf_persist<V31Row, 6>, (const void*)leaf_persist<V31Row, 8>,
                      (const void*)leaf_persist<LtRow, 4>, (const void*)leaf_persist<LtRow, 6>, (const void*)leaf_persist<LtRow, 8>,
                      (const void*)leaf_persist<ServedRow, 6>, (const void*)leaf_persist<ServedRow, 8>,
                      (const void*)leaf_persist<NoMemRow, 4>, (const void*)leaf_persist<NoMemRow, 6>,
                      (const void*)leaf_persist<NoMemRow, 8>, (const void*)salt_persist<false, SMALL_MINB>,
                      (const void*)salt_persist<true, SMALL_MINB>, (const void*)salt_persist<false, 4, 1>,
                      (const void*)salt_persist<true, 4, 1>, (const void*)salt_persist<false, 4, 2>, (const void*)salt_persist<true, 4, 2>};
  for (const void* k : ks) CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
}

// Two green contexts over the device's SMs: n for the compute (its coins, forming and GEMMs) and the rest for the commit (its
// salts, leaves and levels), n rounded up to the split's granularity (8 SMs on CC 9.0). A kernel on one of their streams runs
// on its context's SMs alone. An event records only in its stream's context; any stream waits on any event. Each context
// holds its own copy of the module's __constant__ data and may hold its own kernel attributes, so a user sets both there.
struct Split {
  int n = 0, sms[2] = {0, 0};
  CUgreenCtx g[2] = {};
  CUcontext cx[2] = {};
  cudaStream_t s[3] = {};          // the compute's, the commit's, the commit's salt phases (V31_SAHEAD) and coins (V31_GCOINS)
  cudaEvent_t fork = nullptr;      // the device context's
  cudaEvent_t join[3] = {};        // each in its stream's context
};
static std::vector<Split*> g_splits;
static std::vector<std::pair<cudaStream_t, int>> g_stream_sms;
static void cuk(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  fprintf(stderr, "%s: %s\n", what, s ? s : "unknown error");
  exit(3);
}
// cx current while in scope (null: the current one stays)
struct InCtx {
  CUcontext cx;
  explicit InCtx(CUcontext c) : cx(c) { if (cx) cuk(cuCtxPushCurrent(cx), "cuCtxPushCurrent"); }
  ~InCtx() {
    CUcontext p;
    if (cx) cuk(cuCtxPopCurrent(&p), "cuCtxPopCurrent");
  }
  InCtx(const InCtx&) = delete;
  InCtx& operator=(const InCtx&) = delete;
};
static int device_sms() {
  static int n = 0;
  if (!n) CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0));
  return n;
}
// the SMs a stream's kernels run on: its split side's, else the device's
static int sms_of(cudaStream_t st) {
  for (const auto& e : g_stream_sms)
    if (e.first == st) return e.second;
  return device_sms();
}
// the split with n compute SMs (made once a process, until splits_free; n 0: none)
static Split* split_of(int n) {
  if (n <= 0) return nullptr;
  for (Split* p : g_splits)
    if (p->n == n) return p;
  CK(cudaFree(nullptr));
  CUdevice dev;
  cuk(cuCtxGetDevice(&dev), "cuCtxGetDevice");
  CUdevResource all{}, part[2]{};
  unsigned nb = 1;
  cuk(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource");
  cuk(cuDevSmResourceSplitByCount(&part[0], &nb, &all, &part[1], 0, (unsigned)n), "cuDevSmResourceSplitByCount");
  if (nb != 1 || !part[1].sm.smCount) {
    fprintf(stderr, "V31_GSM %d: the device's %u SMs leave no commit side\n", n, all.sm.smCount);
    exit(2);
  }
  Split* p = new Split;
  p->n = n;
  CK(cudaEventCreateWithFlags(&p->fork, cudaEventDisableTiming));
  for (int i = 0; i < 2; i++) {
    p->sms[i] = (int)part[i].sm.smCount;
    CUdevResourceDesc desc;
    cuk(cuDevResourceGenerateDesc(&desc, &part[i], 1), "cuDevResourceGenerateDesc");
    cuk(cuGreenCtxCreate(&p->g[i], desc, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate");
    cuk(cuCtxFromGreenCtx(&p->cx[i], p->g[i]), "cuCtxFromGreenCtx");
  }
  for (int j = 0; j < 3; j++) {
    const int i = j ? 1 : 0;
    CUstream s;
    cuk(cuGreenCtxStreamCreate(&s, p->g[i], CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate");
    p->s[j] = (cudaStream_t)s;
    g_stream_sms.push_back({p->s[j], p->sms[i]});
    InCtx in(p->cx[i]);
    CK(cudaEventCreateWithFlags(&p->join[j], cudaEventDisableTiming));
  }
  g_splits.push_back(p);
  return p;
}
// every split gone (before a device reset; their Fills dropped them first)
static void splits_free() {
  for (Split* p : g_splits) {
    for (int j = 0; j < 3; j++) {
      cuk(cuStreamSynchronize((CUstream)p->s[j]), "cuStreamSynchronize");
      {
        InCtx in(p->cx[j ? 1 : 0]);
        CK(cudaEventDestroy(p->join[j]));
      }
      cuk(cuStreamDestroy((CUstream)p->s[j]), "cuStreamDestroy");
    }
    CK(cudaEventDestroy(p->fork));
    for (int i = 0; i < 2; i++) cuk(cuGreenCtxDestroy(p->g[i]), "cuGreenCtxDestroy");
    delete p;
  }
  g_splits.clear();
  g_stream_sms.clear();
}
// every stream's work so far done, the splits' too (cudaDeviceSynchronize waits on the current context's alone)
static void device_sync() {
  CK(cudaDeviceSynchronize());
  for (const Split* p : g_splits)
    for (int j = 0; j < 3; j++) cuk(cuStreamSynchronize((CUstream)p->s[j]), "cuStreamSynchronize");
}
// s's work so far before the next on each of p's streams, and theirs so far before s's next (s a device-context stream)
static void split_fork(const Split* p, cudaStream_t s) {
  CK(cudaEventRecord(p->fork, s));
  for (int j = 0; j < 3; j++) CK(cudaStreamWaitEvent(p->s[j], p->fork));
}
static void split_join(const Split* p, cudaStream_t s) {
  for (int j = 0; j < 3; j++) {
    CK(cudaEventRecord(p->join[j], p->s[j]));
    CK(cudaStreamWaitEvent(s, p->join[j]));
  }
}

// A committer's salt and leaf phases in at most ccta CTAs an SM (salt_persist, leaf_persist; ccta 0: commit_gpu's small
// path), then its levels above the leaves' CTAs. Launches that share ctl run in one stream's order.
struct Lanes {
  u32 *ctl = nullptr, *claim = nullptr, epoch = 0;
  int nsm = 1, lminb = SMALL_MINB, sminb = 4, lfold = 0, mask = 0;
  void init(int nsm_) {
    nsm = nsm_;
    ctl = dalloc<u32>(2 + 2 * 8 * (size_t)nsm);
    claim = dalloc<u32>(1024);
    CK(cudaMemset(ctl, 0, 4 * (2 + 2 * 8 * (size_t)nsm)));
    CK(cudaMemset(claim, 0, 4096));
  }
  // a capped launch over nvb groups: twice the CTAs the caps allow, so SMs whose first CTAs land late still get theirs
  Persist persist(u64 nvb, int ccta) {
    epoch = (epoch + 1) & 0xFFFFFF;
    return Persist{ctl, claim, epoch, (u32)ccta, (u32)nvb};
  }
  static unsigned grid(u64 nvb, int ccta, cudaStream_t st) { return (unsigned)std::min<u64>(nvb, 2ull * ccta * sms_of(st)); }
  // the levels a group of a width-leaf tree holds (7, or fewer in a smaller tree)
  static int group_levels(u64 width) {
    int L = 0;
    while ((1ull << L) < std::min<u64>(width, SMALL_THREADS)) ++L;
    return L;
  }
  // whether leaves_nodes folds every group's levels on level kernels (lfold 0), so neither phase's CTAs fold their own
  bool folds_after(int ccta) const { return ccta && !lfold; }
  // leaf pos salted prf_salt(Ks, off + pos), Ks from ka when given, else from C; commit_gpu's small path (ccta 0) salts at
  // offset 0 from C only
  void salts(Committer& cm, int ccta, cudaStream_t st, u64 off = 0, const KsArg* ka = nullptr) {
    if (!ccta && !off && !ka) { cm.salts<true>(nullptr, st); return; }
    const int fl = folds_after(ccta) ? 0 : group_levels(cm.width);
    if (!ccta) ccta = 8;
    const u64 nvb = (cm.width + SMALL_THREADS - 1) / SMALL_THREADS;
    const unsigned g = grid(nvb, ccta, st);
    const Persist p = persist(nvb, ccta);
    const KsArg k = ka ? *ka : KsArg{};
    const auto launch = [&](auto kern) {
      kern<<<g, SMALL_THREADS, 0, st>>>(p, cm.dom, cm.Tn, cm.n, cm.width, cm.S, cm.F, cm.levels, off, fl, k);
    };
    if (sminb == SMALL_MINB) launch(ka ? salt_persist<true, SMALL_MINB> : salt_persist<false, SMALL_MINB>);
    else if (mask == 1) launch(ka ? salt_persist<true, 4, 1> : salt_persist<false, 4, 1>);
    else if (mask == 2) launch(ka ? salt_persist<true, 4, 2> : salt_persist<false, 4, 2>);
    else launch(ka ? salt_persist<true> : salt_persist<false>);
  }
  template <class Row>
  void leaves_nodes(Committer& cm, const Row& rows, int ccta, cudaStream_t st) {
    if (!ccta) {
      cm.leaves(rows, st);
    } else {
      const u64 ng = (cm.n + SMALL_THREADS - 1) / SMALL_THREADS;
      const Persist p = persist(ng, ccta);
      const unsigned g = grid(ng, ccta, st);
      const int L = group_levels(cm.width), fl = folds_after(ccta) ? 0 : L;
      if (lminb == 4) {
        leaf_persist<Row, 4><<<g, SMALL_THREADS, 0, st>>>(p, cm.dom, rows, cm.S, cm.F, cm.n, cm.width, cm.levels, fl);
      } else if (lminb == 6 || lminb == 8) {
        if constexpr (std::is_same_v<Row, FusedRow>) {
          fprintf(stderr, "V31_LMINB %d: the fused leaf phase takes 4 or %d\n", lminb, SMALL_MINB);
          exit(2);
        } else if (lminb == 6) {
          leaf_persist<Row, 6><<<g, SMALL_THREADS, 0, st>>>(p, cm.dom, rows, cm.S, cm.F, cm.n, cm.width, cm.levels, fl);
        } else {
          leaf_persist<Row, 8><<<g, SMALL_THREADS, 0, st>>>(p, cm.dom, rows, cm.S, cm.F, cm.n, cm.width, cm.levels, fl);
        }
      } else {
        leaf_persist<Row><<<g, SMALL_THREADS, 0, st>>>(p, cm.dom, rows, cm.S, cm.F, cm.n, cm.width, cm.levels, fl);
      }
      for (int d = fl; d < L; ++d) {   // the whole width: salts left the groups of pads alone to these too
        const u64 n_out = cm.width >> (d + 1);
        level_kernel<<<(unsigned)((n_out + SMALL_THREADS - 1) / SMALL_THREADS), SMALL_THREADS, 0, st>>>(
            cm.dom, cm.lv[d], cm.lv[d + 1], cm.width >> d, d);
      }
    }
    cm.nodes(st);
  }
};

// One shape's prover: every buffer of a call, NB copies of the committed values (words, O) for the overlapped arm.
struct Ctx {
  int m, n, k, lp, t, row, L, S, NT, kb64, dz, dy, TL, TT, tiles, tiles_n, lpt, grid, nsm, capA, capB, nb = 1, items, pair;
  u64 leaves;
  Seeds seeds;
  int8_t *A, *B, *U, *V, *X, *XT, *W, *PA, *PB, *WW = nullptr;
  int16_t* BtT;
  uint8_t *dA, *dB;
  uint16_t* perm;
  int *listA, *cntA, *listB, *cntB, *offB, *dO, *flag;
  int wcw = 128;
  uint16_t* words[2] = {nullptr, nullptr};
  int* O[2] = {nullptr, nullptr};
  Maps *maps_proto, *maps_plain, *maps_dw = nullptr;
  Committer cm;
  Params params;
  cudaStream_t s0, s1;
  cudaEvent_t ev_comp[2], ev_done[2], ev_formed;
  int ovl = 0, pend = -1;   // pend: the buffer whose commit V31_SCHED 1 hasn't issued yet
  Knobs kn;
  u32* gctr = nullptr;
  u32 gbase = 0;
  Lanes pl;
#ifdef V31_WG
  wgp::Args wa_proto{}, wa_plain{}, wa_wgl{};
  int wbn = 256, wgrid = 0;
#endif

  void setup(int m_, int n_, int k_, int lp_, int t_, int row_, const std::string& params_path, int cap, int want_nb) {
    m = m_; n = n_; k = k_; lp = lp_; t = t_; row = row_;
    // form_a_kernel's CTA: L' threads, and 64 L' bytes of dynamic shared memory beside its static, under 48 KiB (no opt-in)
    if (m % TILE || n % TILE || k % BK || lp % BK || k % lp || lp > 640 || 32768 % row || row % 16) {
      fprintf(stderr, "shape m %d n %d k %d L' %d ROW %d: m, N, K, L' multiples of 128, L' | K, L' <= 640, ROW | 32768\n", m, n,
              k, lp, row);
      exit(2);
    }
    L = t * lp; S = (k + L - 1) / L; NT = k / lp; kb64 = (k + 511) / 512 * 64;
    dz = 208LL * NT < 16384 ? 2 : 3;
    dy = 128LL * NT < 16384 ? 2 : 3;
    TL = (dz + dy) * lp; TT = TL / BK;
    tiles_n = n / TILE; tiles = (m / TILE) * tiles_n;
    lpt = S * (32768 / row) + 65536 / row;
    leaves = (u64)tiles * lpt;
    capA = capB = cap;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    pair = gemm_pair(tiles_n);
    items = tiles >> pair;
    grid = std::min(items, nsm);
    int lo, hi;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    if (kn.cprio != "hi" && kn.cprio != "lo" && kn.cprio != "eq") { fprintf(stderr, "V31_CPRIO: hi, lo or eq\n"); exit(2); }
    CK(cudaStreamCreateWithPriority(&s0, cudaStreamNonBlocking, kn.cprio == "lo" ? hi : lo));
    CK(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, kn.cprio == "hi" ? hi : lo));
    set_carveout(kn.carve);
    gctr = dalloc<u32>(1);
    CK(cudaMemset(gctr, 0, 4));
    if (kn.ccta < 0 || kn.ccta > 8) { fprintf(stderr, "V31_CCTA: 0 to 8\n"); exit(2); }
    pl.init(nsm);
    const size_t mk = (size_t)m * k, nk = (size_t)n * k, mn = (size_t)m * n;
    A = dalloc<int8_t>(mk); B = dalloc<int8_t>(nk); X = dalloc<int8_t>(mk); XT = dalloc<int8_t>(mk); W = dalloc<int8_t>(nk);
    BtT = dalloc<int16_t>(nk);
    U = dalloc<int8_t>((size_t)n * lp); V = dalloc<int8_t>((size_t)m * lp);
    dA = dalloc<uint8_t>((size_t)m * kb64); dB = dalloc<uint8_t>((size_t)n * kb64);
    perm = dalloc<uint16_t>((size_t)NT * lp);
    listA = dalloc<int>((size_t)m * capA); cntA = dalloc<int>(m); listB = dalloc<int>((size_t)n * capB); cntB = dalloc<int>(n);
    offB = dalloc<int>((size_t)n * (NT + 1));
    wcw = kn.wcw == 256 && n % 256 == 0 ? 256 : 128;
    PA = dalloc<int8_t>((size_t)m * TL); PB = dalloc<int8_t>((size_t)n * TL);
    dO = dalloc<int>(mn); flag = dalloc<int>(1);
    CK(cudaMemset(flag, 0, 4));
    params = read_params(params_path);
    cm.init(params, leaves, (u32)row, false, false, 0, 0, true, 1);
    for (int b = 0; b < want_nb; b++) {
      if (b > 0) {
        size_t fr, tot;
        CK(cudaMemGetInfo(&fr, &tot));
        if (fr < (size_t)tiles * S * 32768 + mn * 4 + (2ull << 30)) break;
      }
      words[b] = dalloc<uint16_t>((size_t)tiles * S * TILE * TILE);
      O[b] = dalloc<int>(mn);
      nb = b + 1;
    }
    Maps mp{tmap(X, k, m, TILE), tmap(W, k, n, TILE), tmap(PA, TL, m, TILE), tmap(PB, TL, n, TILE)};
    Maps mq{tmap(X, k, m, TILE), tmap(W, k, n, TILE), tmap(PA, TL, m, TILE), tmap(PB, TL, n, TILE)};
    maps_proto = dalloc<Maps>(1); maps_plain = dalloc<Maps>(1);
    CK(cudaMemcpy(maps_proto, &mp, sizeof mp, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(maps_plain, &mq, sizeof mq, cudaMemcpyHostToDevice));
    if (kn.wrap >= 6 && !kn.gemm) { fprintf(stderr, "V31_WRAP 6 and 7 need V31_GEMM 1\n"); exit(2); }
    if (kn.wrap >= 6) WW = dalloc<int8_t>(nk);
    if (kn.wrap == 5) {
      WW = dalloc<int8_t>(nk);
      const Maps mw{tmap(X, k, m, TILE), tmap(WW, k, n, TILE), tmap(PA, TL, m, TILE), tmap(PB, TL, n, TILE)};
      maps_dw = dalloc<Maps>(1);
      CK(cudaMemcpy(maps_dw, &mw, sizeof mw, cudaMemcpyHostToDevice));
    }
    if (kn.gemm) wg_setup();
    gemm_attrs();
    CK(cudaFuncSetAttribute(wrap_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TILE * 129 * 4));
    CK(cudaFuncSetAttribute(wrap2_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, lp * 128));
    CK(cudaFuncSetAttribute(wrap2_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, lp * 128));
    CK(cudaFuncSetAttribute(wrap3_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (lp + 1) * 128));
    CK(cudaFuncSetAttribute(wrap3_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (lp + 1) * 128));
    CK(cudaFuncSetAttribute(wrap4_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (lp + 1) * 128));
    CK(cudaFuncSetAttribute(wrap4_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (lp + 1) * 128));
    for (int b = 0; b < 2; b++) {
      CK(cudaEventCreateWithFlags(&ev_comp[b], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&ev_done[b], cudaEventDisableTiming));
      CK(cudaEventRecord(ev_done[b], s1));
    }
    CK(cudaEventCreateWithFlags(&ev_formed, cudaEventDisableTiming));
  }
  void set_seeds(const uint8_t* b) {
    for (int s = 0; s < 5; s++)
      for (int i = 0; i < 8; i++) seeds.w[s][i] = sha::be64(b + 64 * s + 8 * i);
  }
  FormArgs fa() const {
    return FormArgs{A, dA, V, perm, X, XT, listA, cntA, PA, flag, m, k, lp, NT, kb64, dz, dy, capA, TL, nullptr};
  }
  FormArgs fb() const {
    return FormArgs{B, dB, U, perm, W, BtT, listB, cntB, PB, flag, n, k, lp, NT, kb64, dz, dy, capB, TL, offB};
  }
  GArgs gargs(int b, bool proto, int diag = 0) const {
    return GArgs{proto ? maps_proto : maps_plain, words[b], dO, O[b], m, n, k, L, S, proto && !(diag & 2) ? TT : 0, 8, lp,
                 dz, kn.dyn ? gctr : nullptr, gbase, diag};
  }
  V31Row rowsrc(int b) const {
    return V31Row{reinterpret_cast<const uint8_t*>(words[b]), reinterpret_cast<const uint8_t*>(O[b]), (u64)lpt,
                  (u64)S * (32768 / row), (u64)n, (u64)tiles_n, (u32)row, (u32)S, (u32)kn.pf};
  }
  void coins(cudaStream_t st) {
    const u64 ba = (u64)m * (kb64 / 64), bb = (u64)n * (kb64 / 64);
    bits_kernel<<<(unsigned)((ba + 127) / 128), 128, 0, st>>>(seeds, S_DA, ba, dA);
    bits_kernel<<<(unsigned)((bb + 127) / 128), 128, 0, st>>>(seeds, S_DB, bb, dB);
    const u64 cu = (u64)n * lp, cv = (u64)m * lp;
    noise_kernel<<<(unsigned)((cu / 16 + 127) / 128), 128, 0, st>>>(seeds, S_U, cu, U);
    noise_kernel<<<(unsigned)((cv / 16 + 127) / 128), 128, 0, st>>>(seeds, S_V, cv, V);
    perm_kernel<<<NT, PERM_THREADS(lp), lp * 6, st>>>(seeds, lp, perm);
  }
  void form(cudaStream_t st) {
    form_a_kernel<<<m / 16, lp, 16 * lp * 4, st>>>(fa());
    if (kn.wrap >= 6) form_b_kernel<2><<<n / 16, lp, 16 * lp, st>>>(fb(), WW);
    else if (WW) form_b_kernel<1><<<n / 16, lp, 16 * lp, st>>>(fb(), WW);
    else form_b_kernel<0><<<n / 16, lp, 16 * lp, st>>>(fb(), nullptr);
  }
  void wrap(cudaStream_t st) {
    const WrapArgs w{XT, BtT, listA, cntA, listB, cntB, dO, m, n, capA, capB, offB, lp, NT};
    if (kn.wrap == 7) {
    } else if (kn.wrap == 6) {
      wrap_a_kernel<true><<<dim3(n / TILE, m / TILE), 256, 0, st>>>(w);
    } else if (kn.wrap == 5) {
      launch<false>(st, GArgs{maps_dw, nullptr, nullptr, dO, m, n, k, L, S, 0, 8, lp, dz, kn.dyn ? gctr : nullptr, gbase, 0});
      wrap_a_kernel<false><<<dim3(n / TILE, m / TILE), 256, 0, st>>>(w);
    } else if (kn.wrap == 1) wrap_kernel<<<dim3(n / TILE, m / TILE), 256, TILE * 129 * 4, st>>>(w);
    else if (kn.wrap == 3 && wcw == 256) wrap3_kernel<256><<<dim3(n / 256, m / TILE), 256, (lp + 1) * 128, st>>>(w);
    else if (kn.wrap == 3) wrap3_kernel<128><<<dim3(n / 128, m / TILE), 256, (lp + 1) * 128, st>>>(w);
    else if (kn.wrap == 4 && wcw == 256) wrap4_kernel<256><<<dim3(n / 256, m / TILE), 256, (lp + 1) * 128, st>>>(w);
    else if (kn.wrap == 4) wrap4_kernel<128><<<dim3(n / 128, m / TILE), 256, (lp + 1) * 128, st>>>(w);
    else if (wcw == 256) wrap2_kernel<256><<<dim3(n / 256, m / TILE), 512, lp * 128, st>>>(w);
    else wrap2_kernel<128><<<dim3(n / 128, m / TILE), 256, lp * 128, st>>>(w);
  }
  // V31_GEMM 1: wg_proto.cuh's GEMMs, 128 x wbn tiles, one CTA an SM (s_a and s_w of wgl, gamma's baseline, at 1/127)
  void wg_setup() {
#ifdef V31_WG
    wbn = kn.wgbn == 256 && n % 256 == 0 ? 256 : 128;
    wgrid = std::min((m / wgp::TM) * (n / wbn), nsm);
    if (kn.rp != 0 && kn.rp != 3) { fprintf(stderr, "V31_RP is 0 or 3 (wg_proto.cuh's register plans)\n"); exit(2); }
    if (kn.rp && !kn.cm) { fprintf(stderr, "V31_RP 3 needs V31_CM 1\n"); exit(2); }
    const int KQ = kn.wrap >= 6 ? k / wgp::BK : 0;
    if (TT + KQ > wgp::MAX_TT) {
      fprintf(stderr, "the tail's %d k-tiles exceed wg_proto.cuh's MAX_TT %d\n", TT + KQ, wgp::MAX_TT);
      exit(2);
    }
    float* sa = dalloc<float>(m);
    float* sw = dalloc<float>(n);
    const std::vector<float> ha(m, 1.f / 127), hw(n, 1.f / 127);
    CK(cudaMemcpy(sa, ha.data(), 4 * (size_t)m, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sw, hw.data(), 4 * (size_t)n, cudaMemcpyHostToDevice));
    uint32_t* cbuf = kn.cm ? dalloc<uint32_t>((size_t)wgrid * wgp::TM * wbn / 2) : nullptr;
    auto mk = [&](const int8_t* a, const int8_t* b, int mode) {
      wgp::Args g{};
      g.a = tmap(a, k, m, wgp::TM); g.b = tmap(b, k, n, wbn); g.pa = tmap(PA, TL, m, wgp::TM); g.pb = tmap(PB, TL, n, wbn);
      if (KQ) g.q = tmap(WW, k, n, wbn);
      g.dO = dO; g.cbuf = cbuf; g.sa = sa; g.sw = sw;
      g.M = m; g.N = n; g.K = k; g.L = L; g.S = S; g.lp = lp; g.dz = dz; g.dy = dy; g.TT = TT; g.group = kn.wgg; g.mode = mode;
      g.KQ = mode == 1 ? KQ : 0;
      if (mode == 1 && kn.wrap == 7) { g.lA = listA; g.cA = cntA; g.bt = BtT; g.capA = capA; }
      return g;
    };
    wa_proto = mk(X, W, 1); wa_plain = mk(X, W, 0); wa_wgl = mk(A, B, 2);
    const void* ks[] = {(const void*)wgp::gemm<256, 0>, (const void*)wgp::gemm<256, 1>, (const void*)wgp::gemm<128, 0>,
                        (const void*)wgp::gemm<128, 1>, (const void*)wgp::gemm_r3<256>, (const void*)wgp::gemm_r3<128>};
    const int sm[] = {wgp::Tile<256>::SMEM, wgp::Tile<256>::SMEM, wgp::Tile<128>::SMEM,
                      wgp::Tile<128>::SMEM, wgp::Tile<256>::SMEM, wgp::Tile<128>::SMEM};
    for (int i = 0; i < 6; i++) {
      CK(cudaFuncSetAttribute(ks[i], cudaFuncAttributeMaxDynamicSharedMemorySize, sm[i]));
      if (kn.carve >= 0) CK(cudaFuncSetAttribute(ks[i], cudaFuncAttributePreferredSharedMemoryCarveout, kn.carve));
    }
#else
    fprintf(stderr, "V31_GEMM 1 needs a -DV31_WG build (sm_90a)\n");
    exit(2);
#endif
  }
#ifdef V31_WG
  // V31_DYN 1: an SM the commit's CTAs share takes fewer tiles (gctr, as gemm_kernel's)
  void wg_launch(cudaStream_t st, wgp::Args g, int cm, int rp) {
    constexpr int T0 = wgp::threads<0>(), T3 = wgp::threads<3>();
    constexpr int S256 = wgp::Tile<256>::SMEM, S128 = wgp::Tile<128>::SMEM;
    g.tctr = kn.dyn ? gctr : nullptr;
    g.tbase = gbase;
    gbase += (u32)((m / wgp::TM) * (n / wbn));
    if (wbn == 256 && rp == 3) wgp::gemm_r3<256><<<wgrid, T3, S256, st>>>(g);
    else if (rp == 3) wgp::gemm_r3<128><<<wgrid, T3, S128, st>>>(g);
    else if (wbn == 256 && cm) wgp::gemm<256, 1><<<wgrid, T0, S256, st>>>(g);
    else if (wbn == 256) wgp::gemm<256, 0><<<wgrid, T0, S256, st>>>(g);
    else if (cm) wgp::gemm<128, 1><<<wgrid, T0, S128, st>>>(g);
    else wgp::gemm<128, 0><<<wgrid, T0, S128, st>>>(g);
  }
  // wgmma.linear on A and B (bf16 values into O[b]): the plain GEMM a served call runs, gamma's denominator, always on
  // RP 0's registers (wgmma.linear's own) and CM 1's kernel, which holds no tail registers (mode 2 has no tail) and spills none
  void wgl(cudaStream_t st, int b) {
    wgp::Args g = wa_wgl;
    g.O = O[b];
    wg_launch(st, g, 1, 0);
  }
#endif
  // launches that share gctr run in one stream's order, so each one's base is the work items launched before it
  template <bool PROTO> void launch(cudaStream_t st, const GArgs& g) {
    gemm_launch<PROTO>(pair, grid, st, g);
    gbase += (u32)items;
  }
  void gemm(cudaStream_t st, int b, bool proto, int diag = 0) {
#ifdef V31_WG
    if (kn.gemm) {
      wgp::Args g = proto ? wa_proto : wa_plain;
      g.words = words[b]; g.O = O[b]; g.diag = diag;
      if (diag & 2) { g.TT = 0; g.KQ = 0; }
      wg_launch(st, g, kn.cm, kn.rp);
      return;
    }
#endif
    if (proto) launch<true>(st, gargs(b, true, diag));
    else launch<false>(st, gargs(b, false));
  }
  void compute(cudaStream_t st, int b, const Cheat& ch = Cheat{}) {
    coins(st);
    form(st);
    if (ch.kind == 1) CK(cudaMemset2DAsync(X + (size_t)ch.tau * lp, k, 0, lp, m, st));
    wrap(st);
    gemm(st, b, true);
    if (ch.kind >= 2) {
      const int stride = std::max(1, (int)std::lround(1.0 / ch.frac));
      forge_kernel<<<(tiles + stride - 1) / stride, 256, 0, st>>>(words[b], O[b], S, ch.kind == 2 ? ch.s0 : 0, stride, tiles_n, n, 0xF0F0u);
    }
  }
  void salts(cudaStream_t st) { pl.salts(cm, kn.ccta, st); }
  template <class Row>
  void leaves_nodes_of(cudaStream_t st, const Row& rows) { pl.leaves_nodes(cm, rows, kn.ccta, st); }
  void leaves_nodes(cudaStream_t st, int b) { leaves_nodes_of(st, rowsrc(b)); }
  void commit(cudaStream_t st, int b) {
    salts(st);
    leaves_nodes(st, b);
  }
  // the overlapped call: compute on s0 once the commit that last read buffer b is done; its commit on s1, the salts
  // ahead of the compute they don't read. V31_SCHED 1 (two copies only): the previous call's commit, salts and all, waits
  // for this call's form, so it runs beside this call's GEMMs rather than its coins and form
  void commit_ovl(int b) {
    salts(s1);
    CK(cudaStreamWaitEvent(s1, ev_comp[b]));
    leaves_nodes(s1, b);
    CK(cudaEventRecord(ev_done[b], s1));
  }
  void call_ovl(cudaStream_t st) {
    const int b = nb == 2 ? (ovl++ & 1) : 0;
    CK(cudaStreamWaitEvent(st, ev_done[b]));
    if (kn.sched == 1 && nb == 2) {
      coins(st);
      form(st);
      CK(cudaEventRecord(ev_formed, st));
      if (pend >= 0) {
        CK(cudaStreamWaitEvent(s1, ev_formed));
        commit_ovl(pend);
      }
      wrap(st);
      gemm(st, b, true);
      CK(cudaEventRecord(ev_comp[b], st));
      pend = b;
      return;
    }
    compute(st, b);
    CK(cudaEventRecord(ev_comp[b], st));
    commit_ovl(b);
  }
  void ovl_drain(cudaStream_t st) {
    if (pend >= 0) commit_ovl(pend);
    pend = -1;
    for (int b = 0; b < nb; b++) CK(cudaStreamWaitEvent(st, ev_done[b]));
  }
  int flags() const {
    int f;
    CK(cudaMemcpy(&f, flag, 4, cudaMemcpyDeviceToHost));
    return f;
  }
  size_t levels_count() const { return 2 * cm.width - 1; }
  std::vector<u64> root() const {
    std::vector<u64> r(8);
    CK(cudaMemcpy(r.data(), cm.levels + 8 * (levels_count() - 1), 64, cudaMemcpyDeviceToHost));
    return r;
  }
};

// ---------------------------------------------------------------- Dither's no-fold call (circuit/dither.py)
// X and W' as fill_a_kernel and fill_b_kernel form them; T_s = (X W'^T over the first min(s L, K) inner indices) mod 2^16,
// s = 1 .. S = ceil(K / L), with no wrap correction, tail or binding. A filler call (A the "pad" stream: A[i, k] byte k of
// chunks i K / 64 ..) commits T_1 .. T_S; a served call (o; A the caller's activations) commits T_1 .. T_W, W = S - 1, from
// the protocol GEMM over k < W L, and O = (A + d_A)(B + d_B)^T mod 2^32 from cuBLASLt on serve_rows_kernel's a' and b' (the
// noise never enters O, so at S = 1 a served call forms no X or W'). Per tile its W (filler: S) prefix-sum rows, then (served) its
// O rows (dither.values_buffer at ROW), in a tree under the params file's domain (dither.Shape.domain). Two copies (values
// and committer): call c's commit runs on s1 beside call c + 1's compute on s0, and call c's values stay until call c + 2
// is issued. Call c salts leaf pos prf_salt(Ks, off_c + pos) (dither.salts), Ks the params file's, set into C.ks on s1
// ahead of its salt phase, so every Fill of a process commits on one s1. W' (and b') is formed again only when the
// weight-side seeds (d_B, U, P) or B change: v3.2 draws them once a period, the run of record once a call.
static void ks_consts(const uint8_t* ks, u64* out) {   // C.ks, then C.ks_m8 (Consts' adjacent fields)
  words_of(ks, out, 8);
  u64 blk[16] = {}, st[8];
  memcpy(blk, out, 64);
  sha::iv(st);
  sha::rounds_prefix(st, blk, 8, out + 8);
}
struct Fill {
  int m = 0, n = 0, k = 0, lp = 0, L = 0, row = 0, S = 0, NT = 0, kb64 = 0, tiles = 0, tiles_n = 0, lpt = 0, grid = 0, nsm = 0,
      items = 0;
  bool pair = false;
  int Wn = 0;                         // committed prefix-sum segments: S - 1 served, S filler
  u64 leaves = 0;
  Seeds seeds{};
  int8_t *A = nullptr, *B = nullptr, *U = nullptr, *V = nullptr, *X = nullptr, *W = nullptr;
  uint8_t *dA = nullptr, *dB = nullptr;
  uint16_t* perm = nullptr;
  uint16_t* words[2] = {nullptr, nullptr};
  int* D[2] = {nullptr, nullptr};     // served: cuBLASLt's a' b'^T; V31_LTW at S = 1: its X W'^T in place of the prefix sums
  uint8_t* vbuf = nullptr;            // testing: a copy's values buffer
  Lt lt{};
  bool ltw = false;
  bool o = false;                     // a served call (set before setup)
  int8_t *Ap = nullptr, *Bp = nullptr;          // served: a' and b' (read by the compute alone)
  int *ra[2] = {}, *sa[2] = {}, *rb[2] = {}, *sb[2] = {};   // their row sums and shifts, a copy each (its O rows read them)
  int* bad = nullptr;                 // bad[copy]: a row of A + d_A spans -128 .. 128; bad[2 + copy]: one of B + d_B
  u64 off[2] = {0, 0};                // copy b's call's salt offset
  bool nomem = false;                 // timing only: the leaves hashed from NoMemRow's register bytes (wrong roots)
  Committer cm[2];
  Maps* maps = nullptr;
  cudaStream_t s0 = nullptr, s1 = nullptr, so = nullptr;   // compute, commit, openings
  bool own = false;                                         // s0 and s1 are this Fill's
  cudaEvent_t ev_comp[2], ev_done[2], ev_formed;
  u32* gctr = nullptr;
  u32 gbase = 0;
  Lanes pl;
  cudaStream_t s2 = nullptr;        // V31_SAHEAD: the salt phases, pls's launches, ev_salt[copy] at each one's end
  cudaEvent_t ev_salt[2] = {};
  Lanes pls;
  // the split the calls run on (null: s0, s1 and s2 on every SM): the compute on its s[0], the commit on s[1] and the salt
  // phases and V31_GCOINS's coins on s[2]
  const Split* sp = nullptr;
  cudaEvent_t ev_coins = nullptr;   // V31_GCOINS: a call's coins done
  struct Evs {
    const Split* sp;
    cudaEvent_t comp[2], done[2], formed, salt[2], coins;
  };
  std::vector<Evs> evs;             // the event sets not in use, by split (an event records only in its stream's context)
  Knobs kn;
  u64 calls = 0;         // calls issued; call c's values are in copy c & 1
  int pend = -1;         // V31_SCHED 1: the copy whose commit waits for the next call's form; 2: a call's GEMM waits for
                         // the previous call's commit (its coins and form don't), so the two never share the SMs
  u64 wseed[3][8] = {};  // the d_B, U and P seeds W' was formed from
  bool wformed = false;
  u64* ksw[2] = {nullptr, nullptr};   // pinned: Ks as C.ks, C.ks_m8 (a copy each, for the async copy)
  u64* opos = nullptr;                // openings: positions, then rows and paths
  uint8_t* obuf = nullptr;
  int ocap = 0;
  // V31_GEMM 1 (served): D = a' b'^T on wg_proto.cuh's plain GEMM (mode 0, static schedule) in place of cuBLASLt; V31_RP 4
  // hashes its O rows in the GEMM's producer warpgroup (5: and a hash warpgroup, its registers r5h), digests to xd[copy]
  // flagged with the launch's epoch xep[copy]; wgw (V31_WGW, default 1): the prefix-sum segments on the wg protocol GEMM
  // too, in place of gemm_kernel<true>, with RP 5's hash warps unless r5w (V31_R5W) is 0; tst (V31_TST): RP 5's stores
  // through shared memory with TMA, on the store maps tso[copy] (D) and tsw[copy] (words)
  bool wg = false, wgw = false, r5w = false, tst = false;
  int wbn = 0, wgrid = 0, hchunk = 0, r5h = 0;
#ifdef V31_WG
  CUtensorMap tso[2]{}, tsw[2]{};
#endif
  u64* xd[2] = {nullptr, nullptr};
  u32* xf[2] = {nullptr, nullptr};
  u32 xep[2] = {0, 0}, xctr = 0;
  float *wsa = nullptr, *wsw = nullptr;
#ifdef V31_WG
  wgp::Args wa_plain{}, wa_words{};
#endif

  // "": fine; else why the shape can't be committed
  static std::string bad_shape(int m, int n, int k, int lp, int L, int row) {
    if (m <= 0 || n <= 0 || k <= 0 || m % TILE || n % TILE || k % BK || lp <= 0 || lp % BK || k % lp || lp > 4096 || L <= 0 ||
        L % lp || row <= 0 || row % 16 || 32768 % row)
      return "shape m " + std::to_string(m) + " n " + std::to_string(n) + " k " + std::to_string(k) + " L' " + std::to_string(lp) +
             " L " + std::to_string(L) + " ROW " + std::to_string(row) +
             ": m, n, K, L' multiples of 128, L' | K, L' <= 4096, L a multiple of L', ROW | 32768 and a multiple of 16";
    if (k >= (1 << 17)) return "K " + std::to_string(k) + ": below 2^17 (dither.K_LIMIT)";
    const u64 lv = (u64)(m / TILE) * (n / TILE) * (((k + L - 1) / L) * (32768 / row) + 65536 / row);
    if (lv > (1ull << 32)) return "more than 2^32 leaves";
    return "";
  }
  void setup(int m_, int n_, int k_, int lp_, int L_, int row_, const Params& params, int dom, cudaStream_t c0 = nullptr,
             cudaStream_t c1 = nullptr) {
    m = m_; n = n_; k = k_; lp = lp_; L = L_; row = row_;
    const std::string why = bad_shape(m, n, k, lp, L, row);
    if (!why.empty()) { fprintf(stderr, "%s\n", why.c_str()); exit(2); }
    S = (k + L - 1) / L; NT = k / lp; kb64 = (k + 511) / 512 * 64;
    Wn = o ? S - 1 : S;
    tiles_n = n / TILE; tiles = (m / TILE) * tiles_n;
    lpt = Wn * (32768 / row) + (o ? 65536 / row : 0);
    leaves = (u64)tiles * lpt;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    pair = gemm_pair(tiles_n);
    items = tiles >> pair;
    grid = std::min(items, nsm);
    ltw = kn.ltw && S == 1 && !o;
    if (kn.ccta < 0 || kn.ccta > 8) { fprintf(stderr, "V31_CCTA: 0 to 8\n"); exit(2); }
    if (kn.lminb != 4 && kn.lminb != SMALL_MINB && kn.lminb != 6 && kn.lminb != 8) {
      fprintf(stderr, "V31_LMINB: 4, %d, 6 or 8\n", SMALL_MINB);
      exit(2);
    }
    if (kn.sminb != 4 && kn.sminb != SMALL_MINB) { fprintf(stderr, "V31_SMINB: 4 or %d\n", SMALL_MINB); exit(2); }
    if (kn.lfold != 0 && kn.lfold != 1) { fprintf(stderr, "V31_LFOLD: 0 or 1\n"); exit(2); }
    if (kn.mask < 0 || kn.mask > 2 || (kn.mask && kn.sminb != 4)) {
      fprintf(stderr, "V31_MASK: 0, 1 or 2 (1 and 2 at V31_SMINB 4)\n");
      exit(2);
    }
    own = !c0;
    if (own) {
      int lo, hi;
      CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
      CK(cudaStreamCreateWithPriority(&c0, cudaStreamNonBlocking, kn.cprio == "lo" ? hi : lo));
      CK(cudaStreamCreateWithPriority(&c1, cudaStreamNonBlocking, kn.cprio == "hi" ? hi : lo));
    }
    s0 = c0; s1 = c1;
    CK(cudaStreamCreateWithFlags(&so, cudaStreamNonBlocking));
    set_carveout(kn.carve);
    gctr = dalloc<u32>(1);
    CK(cudaMemset(gctr, 0, 4));
    pl.init(nsm);
    pl.lminb = kn.lminb;
    pl.sminb = kn.sminb;
    pl.lfold = kn.lfold;
    pl.mask = kn.mask;
    if (kn.sahead) {
      int pr;
      CK(cudaStreamGetPriority(s1, &pr));
      CK(cudaStreamCreateWithPriority(&s2, cudaStreamNonBlocking, pr));
      pls.init(nsm);
      pls.sminb = kn.sminb;
      pls.mask = kn.mask;
      pls.lfold = kn.lfold;   // so its salts leave the groups of pads alone to pl's level kernels exactly when those fold them
      for (int b = 0; b < 2; b++) CK(cudaEventCreateWithFlags(&ev_salt[b], cudaEventDisableTiming));
    }
    const size_t mk = (size_t)m * k, nk = (size_t)n * k;
    A = dalloc<int8_t>(mk); B = dalloc<int8_t>(nk); X = dalloc<int8_t>(mk); W = dalloc<int8_t>(nk);
    U = dalloc<int8_t>((size_t)n * lp); V = dalloc<int8_t>((size_t)m * lp);
    dA = dalloc<uint8_t>((size_t)m * kb64); dB = dalloc<uint8_t>((size_t)n * kb64);
    perm = dalloc<uint16_t>((size_t)NT * lp);
    if (o) {
      Ap = dalloc<int8_t>(mk); Bp = dalloc<int8_t>(nk);
      for (int b = 0; b < 2; b++) { ra[b] = dalloc<int>(m); sa[b] = dalloc<int>(m); rb[b] = dalloc<int>(n); sb[b] = dalloc<int>(n); }
      bad = dalloc<int>(4);
      CK(cudaMemset(bad, 0, 16));
    }
    for (int b = 0; b < 2; b++) {
      if (ltw || o) D[b] = dalloc<int>((size_t)m * n);
      if (!ltw && Wn > 0) words[b] = dalloc<uint16_t>((size_t)tiles * Wn * TILE * TILE);
      cm[b].init(params, leaves, (u32)row, false, false, 0, dom, true, 1);
      CK(cudaMallocHost(&ksw[b], 128));
      ks_consts(params.ks, ksw[b]);
    }
    const Maps mp{tmap(X, k, m, TILE), tmap(W, k, n, TILE), tmap(X, k, m, TILE), tmap(W, k, n, TILE)};
    maps = dalloc<Maps>(1);
    CK(cudaMemcpy(maps, &mp, sizeof mp, cudaMemcpyHostToDevice));
    gemm_attrs();
    if (ltw) lt_setup(lt, m, n, k, X, W, D[0], s0);
    if (o) lt_setup(lt, m, n, k, Ap, Bp, D[0], s0);
    if (o) wg_setup();
    for (int b = 0; b < 2; b++) {
      CK(cudaEventCreateWithFlags(&ev_comp[b], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&ev_done[b], cudaEventDisableTiming));
      CK(cudaEventRecord(ev_done[b], s1));
    }
    CK(cudaEventCreateWithFlags(&ev_formed, cudaEventDisableTiming));
    if (const int g = kn.gsm_of(n, k); g > 0) use(split_of(g));
  }
  cudaStream_t cs() const { return sp ? sp->s[0] : s0; }
  cudaStream_t ms() const { return sp ? sp->s[1] : s1; }
  cudaStream_t ss() const { return sp ? sp->s[2] : s2; }
  CUcontext mcx() const { return sp ? sp->cx[1] : nullptr; }
  // every kernel attribute a call's launches need, in the current context
  void attrs() const {
    set_carveout(kn.carve);
    gemm_attrs();
#ifdef V31_WG
    if (o) wg_attrs(kn.carve);
#endif
  }
  // the split's events, attributes and __constant__ data (C, this Fill's CD slot), once
  void prep(const Split* p) {
    if (!p || p == sp) return;
    for (const Evs& e : evs)
      if (e.sp == p) return;
    Evs e{p, {}, {}, nullptr, {}, nullptr};
    for (int i = 0; i < 2; i++) {
      InCtx in(p->cx[i]);
      for (int b = 0; b < 2; b++) {
        CK(cudaEventCreateWithFlags(i ? &e.done[b] : &e.comp[b], cudaEventDisableTiming));
        if (i) CK(cudaEventCreateWithFlags(&e.salt[b], cudaEventDisableTiming));
      }
      CK(cudaEventCreateWithFlags(i ? &e.coins : &e.formed, cudaEventDisableTiming));
      attrs();
      CK(cudaMemcpyToSymbol(C, &cm[0].hc, sizeof cm[0].hc));
      CK(cudaMemcpyToSymbol(CD, &cm[0].hd, sizeof cm[0].hd, sizeof(Dom) * cm[0].dom));
    }
    evs.push_back(e);
  }
  // the calls from here on p's (null: the device's own streams), after every call issued so far is done
  void use(const Split* p) {
    if (p == sp) return;
    flush();
    device_sync();
    if (p) prep(p);
    Evs cur{sp, {ev_comp[0], ev_comp[1]}, {ev_done[0], ev_done[1]}, ev_formed, {ev_salt[0], ev_salt[1]}, ev_coins};
    Evs nx{};
    for (size_t i = 0; i < evs.size(); i++)
      if (evs[i].sp == p) { nx = evs[i]; evs[i] = cur; }
    for (int b = 0; b < 2; b++) {
      ev_comp[b] = nx.comp[b];
      ev_done[b] = nx.done[b];
      ev_salt[b] = nx.salt[b];
    }
    ev_formed = nx.formed;
    ev_coins = nx.coins;
    sp = p;
  }
  // back on the device's streams, every split's events gone (before splits_free)
  void drop_splits() {
    use(nullptr);
    for (const Evs& e : evs) {
      if (!e.sp) continue;
      for (int b = 0; b < 2; b++) {
        CK(cudaEventDestroy(e.comp[b]));
        CK(cudaEventDestroy(e.done[b]));
        CK(cudaEventDestroy(e.salt[b]));
      }
      CK(cudaEventDestroy(e.formed));
      CK(cudaEventDestroy(e.coins));
    }
    evs.clear();
  }
  void release() {
    drop_splits();
    CK(cudaDeviceSynchronize());
    for (void* p : {(void*)A, (void*)B, (void*)X, (void*)W, (void*)U, (void*)V, (void*)dA, (void*)dB, (void*)perm, (void*)words[0],
                    (void*)words[1], (void*)D[0], (void*)D[1], (void*)vbuf, (void*)maps, (void*)gctr, (void*)pl.ctl,
                    (void*)pl.claim, (void*)opos, (void*)obuf, (void*)Ap, (void*)Bp, (void*)ra[0], (void*)sa[0], (void*)rb[0],
                    (void*)sb[0], (void*)ra[1], (void*)sa[1], (void*)rb[1], (void*)sb[1], (void*)bad, (void*)xd[0],
                    (void*)xd[1], (void*)xf[0], (void*)xf[1], (void*)wsa, (void*)wsw, (void*)pls.ctl, (void*)pls.claim})
      cudaFree(p);
    if (ltw || o) lt_free(lt);
    for (int b = 0; b < 2; b++) {
      cm[b].release();
      cudaFreeHost(ksw[b]);
      cudaEventDestroy(ev_comp[b]);
      cudaEventDestroy(ev_done[b]);
      if (ev_salt[b]) cudaEventDestroy(ev_salt[b]);
    }
    cudaEventDestroy(ev_formed);
    cudaStreamDestroy(so);
    if (s2) cudaStreamDestroy(s2);
    if (own) { cudaStreamDestroy(s0); cudaStreamDestroy(s1); }
  }
  // seeds in dither.STREAMS' order: pad, dA, dB, U, V, P (64 bytes each)
  void set_seeds(const uint8_t* b) {
    static const int order[6] = {S_PAD, S_DA, S_DB, S_U, S_V, S_P};
    for (int s = 0; s < 6; s++)
      for (int i = 0; i < 8; i++) seeds.w[order[s]][i] = sha::be64(b + 64 * s + 8 * i);
  }
  bool weights_stale() const {
    return !wformed || memcmp(wseed[0], seeds.w[S_DB], 64) || memcmp(wseed[1], seeds.w[S_U], 64) || memcmp(wseed[2], seeds.w[S_P], 64);
  }
  void coins(cudaStream_t st, bool pad, bool wside) {
    if (pad) {
      const u64 bp = (u64)m * k / 64;
      bits_kernel<<<(unsigned)((bp + 127) / 128), 128, 0, st>>>(seeds, S_PAD, bp, reinterpret_cast<uint8_t*>(A));
    }
    const u64 ba = (u64)m * (kb64 / 64), cv = (u64)m * lp;
    bits_kernel<<<(unsigned)((ba + 127) / 128), 128, 0, st>>>(seeds, S_DA, ba, dA);
    if (Wn) noise_kernel<<<(unsigned)((cv / 16 + 127) / 128), 128, 0, st>>>(seeds, S_V, cv, V);
    if (wside) {
      const u64 bb = (u64)n * (kb64 / 64), cu = (u64)n * lp;
      bits_kernel<<<(unsigned)((bb + 127) / 128), 128, 0, st>>>(seeds, S_DB, bb, dB);
      if (Wn) {
        noise_kernel<<<(unsigned)((cu / 16 + 127) / 128), 128, 0, st>>>(seeds, S_U, cu, U);
        perm_kernel<<<NT, PERM_THREADS(lp), lp * 6, st>>>(seeds, lp, perm);
      }
    }
  }
  // copy b's operands. Served: a' (and, when wside, b') with their sums and shifts, else copy b takes the other copy's B
  // side (the weights and their coins unchanged since); then X (and W') when a prefix-sum segment is committed
  void form(cudaStream_t st, const int8_t* a, bool wside, int b) {
    if (o) {
      CK(cudaMemsetAsync(bad + b, 0, 4, st));
      serve_rows_kernel<<<m, 256, 0, st>>>(a, dA, k, kb64, 0, Ap, ra[b], sa[b], bad + b);
      if (wside) {
        CK(cudaMemsetAsync(bad + 2 + b, 0, 4, st));
        serve_rows_kernel<<<n, 256, 0, st>>>(B, dB, k, kb64, 1, Bp, rb[b], sb[b], bad + 2 + b);
      } else {
        CK(cudaMemcpyAsync(rb[b], rb[b ^ 1], 4 * (size_t)n, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(sb[b], sb[b ^ 1], 4 * (size_t)n, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(bad + 2 + b, bad + 3 - b, 4, cudaMemcpyDeviceToDevice, st));
      }
    }
    if (!Wn) return;
    const u64 ca = (u64)m * k / 16, cb = (u64)n * k / 16;
    fill_a_kernel<<<(unsigned)((ca + 255) / 256), 256, 0, st>>>(a, dA, V, X, ca, k, lp, kb64);
    if (wside) fill_b_kernel<<<(unsigned)((cb + 255) / 256), 256, 0, st>>>(B, dB, U, perm, W, cb, k, lp, kb64);
  }
  // launches that share gctr run in one stream's order, so each one's base is the work items launched before it. Served:
  // D = a' b'^T on cuBLASLt, then T_1 .. T_W from the protocol GEMM over k < W L (its K only counts its k-tiles)
  // served: the wg GEMMs' arguments (wgmma.linear, the bench's baseline, in any -DV31_WG build), and V31_GEMM 1's D
  void wg_setup() {
#ifdef V31_WG
    if (kn.gemm && kn.rp != 0 && kn.rp != 4 && kn.rp != 5) {
      fprintf(stderr, "a served call's V31_RP is 0, 4 or 5\n");
      exit(2);
    }
    r5h = env_int("V31_R5H", wgp::R5_SPLITS[0]);
    tst = kn.rp == 5 && env_int("V31_TST", 0) != 0;
    if (!r5_kernel(256)) { fprintf(stderr, "V31_R5H is one of wg_proto.cuh's R5_SPLITS\n"); exit(2); }
    wg = kn.gemm != 0;
    wbn = kn.wgbn == 256 && n % 256 == 0 ? 256 : 128;
    wgrid = std::min((m / wgp::TM) * (n / wbn), nsm);
    wsa = dalloc<float>(m);
    wsw = dalloc<float>(n);
    const std::vector<float> ha(m, 1.f / 127), hw(n, 1.f / 127);
    CK(cudaMemcpy(wsa, ha.data(), 4 * (size_t)m, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(wsw, hw.data(), 4 * (size_t)n, cudaMemcpyHostToDevice));
    wa_plain = wg_args(Ap, Bp, 0);
    wa_plain.hall = env_int("V31_HALL", 0);
    wgw = wg && Wn > 0 && env_int("V31_WGW", 1) != 0;
    if (wgw) {   // T_1 .. T_W: mode 1 with no tail and no O over k < W L, gemm_kernel<true>'s prefix sums bit for bit
      wa_words = wg_args(X, W, 1);
      wa_words.K = Wn * L;
      wa_words.L = L;
      wa_words.S = Wn;
      wa_words.hall = wa_plain.hall;
      r5w = kn.rp == 5 && env_int("V31_R5W", 1) != 0;
    }
    if (tst)
      for (int b = 0; b < 2; b++) {
        if (D[b]) tso[b] = tmap_st(D[b], CU_TENSOR_MAP_DATA_TYPE_INT32, 4, (u64)n, (u64)m);
        if (r5w && words[b]) tsw[b] = tmap_st(words[b], CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, wgp::TW, (u64)tiles * Wn * wgp::TW);
      }
    hchunk = std::max(0, env_int("V31_HCHUNK", 0));
    if (wg && (kn.rp == 4 || kn.rp == 5))
      for (int b = 0; b < 2; b++) {
        xd[b] = dalloc<u64>(8 * (size_t)leaves);
        xf[b] = dalloc<u32>((size_t)leaves);
        CK(cudaMemset(xf[b], 0, 4 * (size_t)leaves));
      }
    wg_attrs(kn.carve);
#else
    if (kn.gemm) { fprintf(stderr, "V31_GEMM 1 needs a -DV31_WG build (sm_90a)\n"); exit(2); }
#endif
  }
#ifdef V31_WG
  void wg_attrs(int carve) const {
    const void* ks[] = {(const void*)wgp::gemm<256, 1>, (const void*)wgp::gemm<128, 1>,
                        (const void*)wgp::gemm_r4<256, FusedLeaves>, (const void*)wgp::gemm_r4<128, FusedLeaves>,
                        r5_kernel(256), r5_kernel(128)};
    for (int i = 0; i < 6; i++) {
      CK(cudaFuncSetAttribute(ks[i], cudaFuncAttributeMaxDynamicSharedMemorySize, i >= 4 ? r5_smem(i % 2 ? 128 : 256)
                                                                                  : i % 2 ? wgp::Tile<128>::SMEM
                                                                                          : wgp::Tile<256>::SMEM));
      if (carve >= 0) CK(cudaFuncSetAttribute(ks[i], cudaFuncAttributePreferredSharedMemoryCarveout, carve));
    }
  }
  // RP 5's kernel at split hr (null: not one of R5_SPLITS), and its shared memory
  template <int BN, bool TST, int I = 0> static const void* r5_kernel_at(int hr) {
    if constexpr (I == sizeof wgp::R5_SPLITS / sizeof wgp::R5_SPLITS[0]) return nullptr;
    else if (hr == wgp::R5_SPLITS[I]) return (const void*)wgp::gemm_r5<BN, FusedLeaves, wgp::R5_SPLITS[I], TST>;
    else return r5_kernel_at<BN, TST, I + 1>(hr);
  }
  const void* r5_kernel(int bn) const {
    if (tst) return bn == 256 ? r5_kernel_at<256, true>(r5h) : r5_kernel_at<128, true>(r5h);
    return bn == 256 ? r5_kernel_at<256, false>(r5h) : r5_kernel_at<128, false>(r5h);
  }
  int r5_smem(int bn) const {
    if (tst) return bn == 256 ? wgp::Tile<256>::SMEM_TST : wgp::Tile<128>::SMEM_TST;
    return bn == 256 ? wgp::Tile<256>::SMEM : wgp::Tile<128>::SMEM;
  }
  // mode 0: a b^T (s32) into O; mode 2: wgmma.linear's bf16 values (s_a = s_w = 1/127), gamma's baseline
  wgp::Args wg_args(const int8_t* a, const int8_t* b, int mode) const {
    wgp::Args g{};
    g.a = tmap(a, k, m, wgp::TM);
    g.b = tmap(b, k, n, wbn);
    g.sa = wsa; g.sw = wsw;
    g.M = m; g.N = n; g.K = k; g.L = k; g.S = 1; g.lp = lp; g.group = kn.wgg; g.mode = mode;
    return g;
  }
  // RP 0 (wgmma.linear's registers, CM 1's kernel: modes 0 and 2 hold no tail), or RP 4 or 5 with h (mode 0; RP 5's
  // mode 1 too)
  void wg_run(cudaStream_t st, const wgp::Args& g, const FusedLeaves* h = nullptr) {
    constexpr int T = wgp::threads<0>();
    const int sm = wbn == 256 ? wgp::Tile<256>::SMEM : wgp::Tile<128>::SMEM;
    const int gr = std::min(wgrid, sms_of(st));
    if (h && kn.rp == 5) {
      void* args[] = {(void*)&g, (void*)h};
      CK(cudaLaunchKernel(r5_kernel(wbn), dim3(gr), dim3(wgp::threads<5>()), args, r5_smem(wbn), st));
    } else if (h && wbn == 256) wgp::gemm_r4<256, FusedLeaves><<<gr, T, sm, st>>>(g, *h);
    else if (h) wgp::gemm_r4<128, FusedLeaves><<<gr, T, sm, st>>>(g, *h);
    else if (wbn == 256) wgp::gemm<256, 1><<<gr, T, sm, st>>>(g);
    else wgp::gemm<128, 1><<<gr, T, sm, st>>>(g);
  }
  // its O rows (the GEMM of D's hash warps), or every row (the GEMM of the prefix sums')
  FusedLeaves fused_leaves(int b, bool every = false) const {
    const ServedRow r = servedrow(b);
    return FusedLeaves{r, xd[b], xf[b], xep[b], (u32)(every ? r.lpt : r.lpt - r.wrows), (u32)hchunk, (u32)(every ? 0 : r.wrows)};
  }
#endif
  // a call's FusedLeaves epoch: it wraps below FUSED_EPOCHS, and clears both copies' flags then (a cleared flag only
  // makes the leaf phase hash that row from its first chunk)
  u32 epoch(cudaStream_t st) {
    if (++xctr == FUSED_EPOCHS) {
      xctr = 1;
      for (int c = 0; c < 2; c++)
        if (xf[c]) CK(cudaMemsetAsync(xf[c], 0, 4 * (size_t)leaves, st));
    }
    return xctr;
  }
  void gemm(cudaStream_t st, int b) {
    if (ltw) { lt_run(lt, X, W, D[b], st); return; }
#ifdef V31_WG
    if (o && wg) {
      wgp::Args g = wa_plain;
      g.O = D[b];
      if (tst) g.tso = tso[b];
      if (kn.rp == 4 || kn.rp == 5) {
        xep[b] = epoch(st);
        const FusedLeaves h = fused_leaves(b);
        wg_run(st, g, &h);
      } else {
        wg_run(st, g);
      }
    } else
#endif
    if (o) lt_run(lt, Ap, Bp, D[b], st);
    if (!Wn) return;
#ifdef V31_WG
    if (wgw) {
      wgp::Args g = wa_words;
      g.words = words[b];
      if (o && r5w) {
        if (tst) g.tsw = tsw[b];
        const FusedLeaves h = fused_leaves(b, true);
        wg_run(st, g, &h);
      } else {
        wg_run(st, g);
      }
      return;
    }
#endif
    gemm_launch<true>(pair, std::min(grid, sms_of(st)), st,
                      GArgs{maps, words[b], nullptr, nullptr, m, n, o ? Wn * L : k, L, Wn, 0, 8, lp, 0, kn.dyn ? gctr : nullptr,
                            gbase, 0});
    gbase += (u32)items;
  }
  V31Row rowsrc(int b) const {
    const uint8_t* w = reinterpret_cast<const uint8_t*>(words[b]);
    return V31Row{w, w, (u64)lpt, (u64)lpt, (u64)n, (u64)tiles_n, (u32)row, (u32)Wn, (u32)kn.pf, full_rows()};
  }
  LtRow ltrow(int b) const { return LtRow{D[b], (u64)lpt, (u64)n, (u64)tiles_n, (u32)row, (u32)kn.pf, full_rows()}; }
  ServedRow servedrow(int b) const {
    return ServedRow{reinterpret_cast<const uint8_t*>(words[b]), D[b], ra[b], sa[b], rb[b], sb[b], (u64)lpt,
                     (u64)Wn * (32768 / row), (u64)n, (u64)tiles_n, (u32)row, (u32)Wn, (u32)kn.pf, full_rows()};
  }
  // rows of whole SHA-512 blocks (V31_FULL): the leaf phase's hash_full path
  u32 full_rows() const { return kn.full && row % 128 == 0 ? 1u : 0u; }
  void leaves_nodes(int b, cudaStream_t st) {
    if (nomem) pl.leaves_nodes(cm[b], NoMemRow{(u32)row, full_rows()}, kn.ccta, st);
    else if (ltw) pl.leaves_nodes(cm[b], ltrow(b), kn.ccta, st);
    else if (o && xd[b]) pl.leaves_nodes(cm[b], FusedRow{servedrow(b), xd[b], xf[b], xep[b]}, kn.ccta, st);
    else if (o) pl.leaves_nodes(cm[b], servedrow(b), kn.ccta, st);
    else pl.leaves_nodes(cm[b], rowsrc(b), kn.ccta, st);
  }
  // copy b's values poisoned (testing)
  void poison(int b, cudaStream_t st) {
    if (D[b]) gk::poison(D[b], (size_t)m * n * 4, st);
    if (words[b]) gk::poison(words[b], (size_t)tiles * Wn * 32768, st);
  }
  // copy b's committed values (testing): leaves x ROW bytes, leaf pos's row at pos ROW (dither.values_buffer); words[b]
  // itself, or the rows gathered into vbuf
  const uint8_t* values_of(int b);
  // C.ks on st (cx: st's split side, else null), in the current context's copy and in cx's: a kernel on a split's stream
  // reads one of the two, the context the runtime resolves it in
  void set_ks(int b, cudaStream_t st, CUcontext cx = nullptr) {
    CK(cudaMemcpyToSymbolAsync(C, ksw[b], 128, offsetof(Consts, ks), cudaMemcpyHostToDevice, st));
    if (!cx) return;
    InCtx in(cx);
    CK(cudaMemcpyToSymbolAsync(C, ksw[b], 128, offsetof(Consts, ks), cudaMemcpyHostToDevice, st));
  }
  // copy b's commit on ms(): Ks and its call's salts (V31_SAHEAD: issue's, on ss()), then (after its compute) its leaves
  // and levels
  void commit_ovl(int b) {
    const cudaStream_t st = ms();
    if (kn.sahead) {
      CK(cudaStreamWaitEvent(st, ev_salt[b]));
    } else {
      set_ks(b, st, mcx());
      pl.salts(cm[b], kn.ccta, st, off[b]);
    }
    CK(cudaStreamWaitEvent(st, ev_comp[b]));
    leaves_nodes(b, st);
    CK(cudaEventRecord(ev_done[b], st));
  }
  void flush() {
    if (pend >= 0) commit_ovl(pend);
    pend = -1;
  }
  // One call on the seeds set, its leaves salted from off (dither.salts(Ks, off): off_c = c leaves for the c-th call of a
  // tree's sequence), A at a (a device pointer to m x K int8, ready on the caller's side; null: the pad, a filler's only).
  // Waits on the host for call c - 2's commit, whose copy this call takes. wside: form W' (b') again even if its seeds are
  // unchanged (the per-call weight coins). Returns the copy.
  int issue(u64 off_, const int8_t* a, const Cheat& ch = Cheat{}, bool wside = false) {
    if (o && !a) { fprintf(stderr, "a served call takes the caller's A\n"); exit(2); }
    if ((ch.kind == 2 || ch.kind == 3) && (ch.kind == 3 ? 0 : ch.s0) > Wn) { fprintf(stderr, "forge: S0 above W\n"); exit(2); }
    if ((ch.kind == 3 || (ch.kind == 2 && !ch.s0)) && !D[0]) { fprintf(stderr, "forge of O: a served call's\n"); exit(2); }
    const int b = (int)(calls & 1);
    const cudaStream_t c0 = cs(), c2 = ss();
    const int sched = sp ? 0 : kn.sched;
    const bool gc = sp && kn.gcoins;
    if (pend == b) flush();
    CK(cudaEventSynchronize(ev_done[b]));
    off[b] = off_;
    auto salts_ahead = [&] {
      KsArg ka;
      static_assert(sizeof ka == 128, "KsArg: C.ks then C.ks_m8, as ksw holds them");
      memcpy(&ka, ksw[b], sizeof ka);
      pls.salts(cm[b], kn.ccta, c2, off[b], &ka);
      CK(cudaEventRecord(ev_salt[b], c2));
    };
    if (kn.sahead && !gc) salts_ahead();
    const bool ws = wside || weights_stale();
    if (gc) {   // after the previous call's form, which reads the buffers these coins overwrite
      CK(cudaStreamWaitEvent(c2, ev_formed));
      coins(c2, !a, ws);
      CK(cudaEventRecord(ev_coins, c2));
      CK(cudaStreamWaitEvent(c0, ev_coins));
      if (kn.sahead) salts_ahead();
    } else {
      coins(c0, !a, ws);
    }
    if (ch.kind == 4) {   // nodither: d_A and d_B zero
      CK(cudaMemsetAsync(dA, 0, (size_t)m * kb64, c0));
      if (ws) CK(cudaMemsetAsync(dB, 0, (size_t)n * kb64, c0));
    }
    form(c0, a ? a : A, ws, b);
    if (ws) {
      memcpy(wseed[0], seeds.w[S_DB], 64); memcpy(wseed[1], seeds.w[S_U], 64); memcpy(wseed[2], seeds.w[S_P], 64);
      wformed = ch.kind != 4;
    }
    if (ch.kind == 1 && Wn) CK(cudaMemset2DAsync(X + (size_t)ch.tau * lp, k, 0, lp, m, c0));
    if (sched == 1 || gc) CK(cudaEventRecord(ev_formed, c0));
    if (sched == 1 && pend >= 0) {
      CK(cudaStreamWaitEvent(ms(), ev_formed));
      commit_ovl(pend);
      pend = -1;
    }
    if (sched == 2 && calls) CK(cudaStreamWaitEvent(c0, ev_done[b ^ 1]));
    gemm(c0, b);
    if (ch.kind == 2 || ch.kind == 3) {
      const int stride = std::max(1, (int)std::lround(1.0 / ch.frac));
      const int s0f = ch.kind == 3 ? 0 : ch.s0;
      if (ltw || (o && s0f == 0))   // forge_kernel's O path into D: V31_LTW's T_1 (its low halves), or O
        forge_kernel<<<(tiles + stride - 1) / stride, 256, 0, c0>>>(nullptr, D[b], Wn, 0, stride, tiles_n, n, 0xF0F0u);
      else
        forge_kernel<<<(tiles + stride - 1) / stride, 256, 0, c0>>>(words[b], nullptr, Wn, s0f, stride, tiles_n, n, 0xF0F0u);
      xep[b] = epoch(c0);   // RP 4's digests are of the rows before the forge: the leaf phase hashes them again
    }
    CK(cudaEventRecord(ev_comp[b], c0));
    if (sched == 1) pend = b;
    else commit_ovl(b);
    calls++;
    return b;
  }
  // call c's copy once its commit is done, or -1 when c isn't one of the last two calls
  int ready(u64 c) {
    if (c >= calls || c + 2 < calls) return -1;
    const int b = (int)(c & 1);
    if (pend == b) flush();
    CK(cudaEventSynchronize(ev_done[b]));
    return b;
  }
  size_t levels_count() const { return 2 * cm[0].width - 1; }
  // positions' rows (row bytes each) and paths (depth digests each, leaf level first) of copy b, gathered on the device
  void open(int b, const u64* pos, int np, uint8_t* rows, uint8_t* paths);
};
// a leaf's row bytes: from the prefix-sum buffer, or (V31_LTW) from cuBLASLt's D
struct WordsPiece {
  const uint8_t* w;
  u32 row;
  DEV uint4 piece(u64 pos, u32 off) const { return *reinterpret_cast<const uint4*>(w + pos * row + off); }
};
struct LtPiece {
  LtRow r;
  DEV uint4 piece(u64 pos, u32 off) const { return r.src(pos).ld16(off); }
};
struct ServedPiece {
  ServedRow r;
  DEV uint4 piece(u64 pos, u32 off) const { return r.src(pos).ld16(off); }
};
// testing: the leaves' rows gathered into one buffer, a 16-byte piece a thread
template <class P>
__global__ void values_kernel(P src, u64 n, int row, uint8_t* __restrict__ out) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x, q = (u64)row / 16;
  if (t < n * q) *reinterpret_cast<uint4*>(out + 16 * t) = src.piece(t / q, (u32)(16 * (t % q)));
}
// a thread a (position, 16-byte piece) of the rows, then a (position, level) of the paths
template <class P>
__global__ void open_kernel(P src, const u64* __restrict__ levels, u64 width, int depth, int row, const u64* __restrict__ pos,
                            int np, uint8_t* __restrict__ rows, uint8_t* __restrict__ paths) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x, rq = (u64)np * (row / 16);
  if (t < rq) {
    const u64 p = t / (row / 16), c = t % (row / 16);
    *reinterpret_cast<uint4*>(rows + p * row + 16 * c) = src.piece(pos[p], (u32)(16 * c));
  } else if (t < rq + (u64)np * depth) {
    const u64 p = (t - rq) / depth;
    const int d = (int)((t - rq) % depth);
    const u64* src = levels + 8 * (2 * width - 2 * (width >> d)) + 8 * ((pos[p] >> d) ^ 1);
    for (int i = 0; i < 8; i++)   // the digest's big-endian bytes
      reinterpret_cast<u64*>(paths + 64 * (p * depth + d))[i] = (u64)bswap32((u32)src[i]) << 32 | bswap32((u32)(src[i] >> 32));
  }
}
void Fill::open(int b, const u64* pos, int np, uint8_t* rows, uint8_t* paths) {
  const int depth = cm[b].depth;
  const size_t need = (size_t)np * (row + 64 * depth);
  if (np > ocap) {
    cudaFree(opos); cudaFree(obuf);
    ocap = std::max(np, 64);
    opos = dalloc<u64>(ocap);
    obuf = dalloc<uint8_t>((size_t)ocap * (row + 64 * depth));
  }
  CK(cudaMemcpyAsync(opos, pos, 8 * (size_t)np, cudaMemcpyHostToDevice, so));
  const u64 th = (u64)np * (row / 16 + depth);
  const unsigned g = (unsigned)((th + 255) / 256);
  if (ltw)
    open_kernel<<<g, 256, 0, so>>>(LtPiece{ltrow(b)}, cm[b].levels, cm[b].width, depth, row, opos, np, obuf, obuf + (size_t)np * row);
  else if (o)
    open_kernel<<<g, 256, 0, so>>>(ServedPiece{servedrow(b)}, cm[b].levels, cm[b].width, depth, row, opos, np, obuf,
                                   obuf + (size_t)np * row);
  else
    open_kernel<<<g, 256, 0, so>>>(WordsPiece{reinterpret_cast<const uint8_t*>(words[b]), (u32)row}, cm[b].levels, cm[b].width,
                                   depth, row, opos, np, obuf, obuf + (size_t)np * row);
  CK(cudaMemcpyAsync(rows, obuf, (size_t)np * row, cudaMemcpyDeviceToHost, so));
  CK(cudaMemcpyAsync(paths, obuf + (size_t)np * row, need - (size_t)np * row, cudaMemcpyDeviceToHost, so));
  CK(cudaStreamSynchronize(so));
}
const uint8_t* Fill::values_of(int b) {
  if (!ltw && !o) return reinterpret_cast<const uint8_t*>(words[b]);
  if (!vbuf) vbuf = dalloc<uint8_t>(leaves * row);
  const u64 th = leaves * (row / 16);
  const unsigned g = (unsigned)((th + 255) / 256);
  if (ltw) values_kernel<<<g, 256, 0, so>>>(LtPiece{ltrow(b)}, leaves, row, vbuf);
  else values_kernel<<<g, 256, 0, so>>>(ServedPiece{servedrow(b)}, leaves, row, vbuf);
  CK(cudaStreamSynchronize(so));
  return vbuf;
}

static std::vector<uint8_t> readfile(const std::string& p, size_t want) {
  std::vector<uint8_t> v = readf(p);
  if (want && v.size() != want) { fprintf(stderr, "%s: %zu bytes, want %zu\n", p.c_str(), v.size(), want); exit(2); }
  return v;
}

struct KAttr { const char* name; const void* fn; };
// a kernel's registers, local and static shared bytes; the small path's (leaf_*, salt_*) also its CTAs an SM by resources
static void kernel_attr(FILE* f, const KAttr& k, bool first) {
  cudaFuncAttributes a;
  CK(cudaFuncGetAttributes(&a, k.fn));
  fprintf(f, "%s\"%s\": {\"regs\": %d, \"local\": %zu, \"smem_static\": %zu", first ? "" : ", ", k.name, a.numRegs,
          a.localSizeBytes, a.sharedSizeBytes);
  if (!strncmp(k.name, "leaf_", 5) || !strncmp(k.name, "salt_", 5)) {
    int nb = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k.fn, SMALL_THREADS, 0));
    fprintf(f, ", \"ctas_sm\": %d", nb);
  }
  fprintf(f, "}");
}
static void kernel_attrs(FILE* f) {
  const KAttr ks[] = {{"gemm_proto", (const void*)gemm_kernel<true, 1>}, {"gemm_plain", (const void*)gemm_kernel<false, 1>},
                      {"gemm_proto_wide", (const void*)gemm_kernel<true, G_NBW>},
                      {"gemm_plain_wide", (const void*)gemm_kernel<false, G_NBW>},
                      {"form_a", (const void*)form_a_kernel}, {"form_b", (const void*)form_b_kernel<0>},
                      {"form_b5", (const void*)form_b_kernel<1>}, {"form_b6", (const void*)form_b_kernel<2>},
                      {"wrap", (const void*)wrap_kernel}, {"wrap2_128", (const void*)wrap2_kernel<128>},
                      {"wrap2_256", (const void*)wrap2_kernel<256>}, {"wrap3_128", (const void*)wrap3_kernel<128>},
                      {"wrap3_256", (const void*)wrap3_kernel<256>}, {"wrap4_128", (const void*)wrap4_kernel<128>},
                      {"wrap4_256", (const void*)wrap4_kernel<256>}, {"wrap_a", (const void*)wrap_a_kernel<false>},
                      {"bits", (const void*)bits_kernel},
                      {"leaf_small", (const void*)leaf_small<V31Row, 0>}, {"salt_small", (const void*)salt_small<true>},
                      {"leaf_persist", (const void*)leaf_persist<V31Row>}, {"salt_persist", (const void*)salt_persist<false>},
                      {"salt_persist_ks", (const void*)salt_persist<true>},
                      {"salt_persist_m5", (const void*)salt_persist<false, SMALL_MINB>},
                      {"salt_persist_l1", (const void*)salt_persist<false, 4, 2>},
                      {"leaf_persist_nomem", (const void*)leaf_persist<NoMemRow>},
                      {"leaf_persist_m6", (const void*)leaf_persist<V31Row, 6>},
                      {"leaf_persist_lt", (const void*)leaf_persist<LtRow>}, {"leaf_persist_lt_m6", (const void*)leaf_persist<LtRow, 6>},
                      {"leaf_persist_served", (const void*)leaf_persist<ServedRow>},
                      {"leaf_persist_served_m6", (const void*)leaf_persist<ServedRow, 6>},
                      {"leaf_persist_served_m8", (const void*)leaf_persist<ServedRow, 8>}};
  fprintf(f, "{");
  for (size_t i = 0; i < sizeof ks / sizeof ks[0]; i++) kernel_attr(f, ks[i], i == 0);
#ifdef V31_WG
  const KAttr wk[] = {{"wg_256_c0", (const void*)wgp::gemm<256, 0>}, {"wg_256_c1", (const void*)wgp::gemm<256, 1>},
                      {"wg_128_c0", (const void*)wgp::gemm<128, 0>}, {"wg_128_c1", (const void*)wgp::gemm<128, 1>},
                      {"wg_256_r3", (const void*)wgp::gemm_r3<256>}, {"wg_128_r3", (const void*)wgp::gemm_r3<128>},
                      {"wg_256_r4", (const void*)wgp::gemm_r4<256, FusedLeaves>},
                      {"wg_128_r4", (const void*)wgp::gemm_r4<128, FusedLeaves>},
                      {"leaf_persist_fused", (const void*)leaf_persist<FusedRow>},
                      {"leaf_persist_fused_m4", (const void*)leaf_persist<FusedRow, 4>},
                      {"leaf_persist_served_m4", (const void*)leaf_persist<ServedRow, 4>}};
  for (const KAttr& w : wk) {
    cudaFuncAttributes a;
    CK(cudaFuncGetAttributes(&a, w.fn));
    fprintf(f, ", \"%s\": {\"regs\": %d, \"local\": %zu, \"smem_static\": %zu}", w.name, a.numRegs, a.localSizeBytes,
            a.sharedSizeBytes);
  }
  for (const int hr : wgp::R5_SPLITS)
    for (const int bn : {256, 128})
      for (const bool ts : {false, true}) {
        cudaFuncAttributes a;
        CK(cudaFuncGetAttributes(&a, bn == 256 ? ts ? Fill::r5_kernel_at<256, true>(hr) : Fill::r5_kernel_at<256, false>(hr)
                                     : ts      ? Fill::r5_kernel_at<128, true>(hr)
                                               : Fill::r5_kernel_at<128, false>(hr)));
        fprintf(f, ", \"wg_%d_r5_%d%s\": {\"regs\": %d, \"local\": %zu, \"smem_static\": %zu}", bn, hr, ts ? "_tst" : "",
                a.numRegs, a.localSizeBytes, a.sharedSizeBytes);
      }
#endif
  fprintf(f, "}");
}

// commit_gpu's ruler (commit_gpu.cu's sha_rate and sha_bench's launch) in this build: each thread chains reps compressions,
// its block from the last state; RULER_CTAS CTAs of 256 threads an SM
constexpr int RULER_CTAS = 8, RULER_CHAIN = 256;
__global__ void sha_ruler(u64* io, int reps) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 st[8], w[16];
  sha::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sha::compress(st, w);
  }
  if (st[0] == 0x0123456789abcdefull) io[t] = st[1];
}
// The SHA-512 compressions of a Fill commit's parts (compress_from and compress_kw count as one each) over n leaves of
// nblk chunks in a tree of width leaves (width >= SMALL_THREADS): the leaf phase, nblk + 3 a leaf (the row, h's two, the frame
// leaf's second) and two a node of the groups that hold data and of the levels above the groups; the salt phase, 6 a leaf
// (the PRF's three, the salt digest's two, the frame leaf's first; the mask is no compression), two a pad and a node of the
// groups of pads alone, which are the leaf phase's when its level kernels fold every group (after, Lanes::folds_after);
// the ruler, RULER_CHAIN a thread.
struct CommitCounts {
  double leaf, salt, sha;
};
static CommitCounts commit_counts(u64 n, u64 width, u32 nblk, int nsm, bool after) {
  const u64 ng = (n + SMALL_THREADS - 1) / SMALL_THREADS, wg = width / SMALL_THREADS;
  const double pad_groups = 2.0 * (SMALL_THREADS - 1) * (double)(wg - ng);
  return CommitCounts{(double)n * (nblk + 3) + 2.0 * ((double)ng * (SMALL_THREADS - 1) + (double)(wg - 1)) + (after ? pad_groups : 0),
                      6.0 * (double)n + 2.0 * (double)(width - n) + (after ? 0 : pad_groups),
                      (double)nsm * RULER_CTAS * 256 * RULER_CHAIN};
}

static u64 sum_counts(const int* d, int rows) {
  std::vector<int> h(rows);
  CK(cudaMemcpy(h.data(), d, rows * 4, cudaMemcpyDeviceToHost));
  u64 s = 0;
  for (int v : h) s += (u64)v;
  return s;
}

static int cmd_gate(const std::string& dir, int m, int n, int k, int lp, int t, int row, const std::string& cheat) {
  Ctx c;
  c.setup(m, n, k, lp, t, row, dir + "/params.bin", k, 1);
  const Cheat ch = parse_cheat(cheat);
  const auto a = readfile(dir + "/A.bin", (size_t)m * k), b = readfile(dir + "/B.bin", (size_t)n * k);
  const auto sd = readfile(dir + "/seeds.bin", 320);
  CK(cudaMemcpy(c.A, a.data(), a.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(c.B, b.data(), b.size(), cudaMemcpyHostToDevice));
  c.set_seeds(sd.data());
  const size_t wb = (size_t)c.tiles * c.S * 32768, ob = (size_t)m * n * 4, lb = c.levels_count() * 64;
  auto poison = [&](cudaStream_t s) {
    gk::poison(c.words[0], wb, s); gk::poison(c.O[0], ob, s); gk::poison(c.cm.levels, lb, s);
    gk::poison(c.X, (size_t)m * k, s); gk::poison(c.W, (size_t)n * k, s);
  };
  poison(c.s0);
  c.compute(c.s0, 0, ch);
  c.commit(c.s0, 0);
  CK(cudaStreamSynchronize(c.s0));
  CK(cudaGetLastError());
  gk::Dump d(dir);
  d.bytes("words", c.words[0], wb, c.s0);
  d.bytes("O", c.O[0], ob, c.s0);
  d.digests("levels", c.cm.levels, c.levels_count(), c.s0);
  d.bytes("X", c.X, (size_t)m * k, c.s0);
  d.bytes("Wp", c.W, (size_t)n * k, c.s0);
  const int fl = c.flags();
  const u64 wa = sum_counts(c.cntA, m), wbn = sum_counts(c.cntB, n);
  // the no-write control: the same buffers poisoned, nothing run
  poison(c.s0);
  CK(cudaStreamSynchronize(c.s0));
  gk::Dump dc(dir + "/control");
  dc.bytes("words", c.words[0], wb, c.s0);
  dc.bytes("O", c.O[0], ob, c.s0);
  dc.digests("levels", c.cm.levels, c.levels_count(), c.s0);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  FILE* f = fopen((dir + "/gpu.json").c_str(), "w");
  fprintf(f, "{\"gpu\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"lp\": %d, \"t\": %d, \"row\": %d, \"S\": %d, \"dz\": %d, \"dy\": %d, "
          "\"leaves\": %llu, \"flags\": %d, \"wraps_A\": %llu, \"wraps_B\": %llu, \"cheat\": \"%s\", \"kernels\": ",
          prop.name, m, n, k, lp, t, row, c.S, c.dz, c.dy, (unsigned long long)c.leaves, fl, (unsigned long long)wa,
          (unsigned long long)wbn, cheat.c_str());
  kernel_attrs(f);
  fprintf(f, "}\n");
  fclose(f);
  printf("gate %s: %dx%dx%d L' %d t %d S %d dz %d dy %d leaves %llu flags %d wraps A %llu B %llu cheat %s\n", dir.c_str(), m, n, k,
         lp, t, c.S, c.dz, c.dy, (unsigned long long)c.leaves, fl, (unsigned long long)wa, (unsigned long long)wbn, cheat.c_str());
  return fl ? 4 : 0;
}

// two sampled tiles of a bench shape's checked call (its values in buffer b), for `v31.py gpu-sample`: their A and B rows,
// prefix sums, O and leaves
static void dump_samples(Ctx& c, const std::string& dir, const std::string& pfile, u32 pick, int b = 0) {
  gk::Dump d(dir);
  std::vector<uint8_t> sd(320);
  for (int s = 0; s < 5; s++)
    for (int i = 0; i < 8; i++) sha::put_be64(sd.data() + 64 * s + 8 * i, c.seeds.w[s][i]);
  FILE* f = fopen((dir + "/seeds.bin").c_str(), "wb");
  fwrite(sd.data(), 1, 320, f);
  fclose(f);
  const auto p = readf(pfile);
  f = fopen((dir + "/params.bin").c_str(), "wb");
  fwrite(p.data(), 1, p.size(), f);
  fclose(f);
  const int picks[2] = {0, (int)(mix32(pick) % (u32)c.tiles)};
  f = fopen((dir + "/sample.json").c_str(), "w");
  fprintf(f, "{\"m\": %d, \"n\": %d, \"k\": %d, \"lp\": %d, \"t\": %d, \"row\": %d, \"tiles\": [%d, %d], \"root\": \"", c.m, c.n,
          c.k, c.lp, c.t, c.row, picks[0], picks[1]);
  const std::vector<u64> root = c.root();
  for (int i = 0; i < 64; i++) fprintf(f, "%02x", (unsigned)(root[i / 8] >> (56 - 8 * (i % 8))) & 0xFF);
  fprintf(f, "\"}\n");
  fclose(f);
  for (int i = 0; i < 2; i++) {
    const int tile = picks[i], tm = tile / c.tiles_n, tn = tile % c.tiles_n;
    const std::string s = "t" + std::to_string(i) + "_";
    d.bytes(s + "A", c.A + (size_t)tm * TILE * c.k, (size_t)TILE * c.k, c.s0);
    d.bytes(s + "B", c.B + (size_t)tn * TILE * c.k, (size_t)TILE * c.k, c.s0);
    d.bytes(s + "words", c.words[b] + (size_t)tile * c.S * TILE * TILE, (size_t)c.S * 32768, c.s0);
    std::vector<int> o((size_t)TILE * TILE), row(TILE);
    for (int r = 0; r < TILE; r++)
      CK(cudaMemcpy(o.data() + (size_t)r * TILE, c.O[b] + (size_t)(tm * TILE + r) * c.n + tn * TILE, TILE * 4, cudaMemcpyDeviceToHost));
    f = fopen((dir + "/" + s + "O.bin").c_str(), "wb");
    fwrite(o.data(), 4, o.size(), f);
    fclose(f);
    d.digests(s + "leaves", c.cm.levels + 8 * (size_t)tile * c.lpt, c.lpt, c.s0);
  }
}

static int cmd_bench(int reps, const std::string& out, const std::string& pdir, int lp, int t, int row,
                     const std::vector<std::string>& shapes) {
  gk::Nvml nv(0);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::mt19937_64 rng(std::random_device{}());
  for (const auto& sh : shapes) {
    char name[64];
    int m, n, k;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &n, &k) != 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    const std::string pfile = pdir + "/" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) + "-lp" +
                              std::to_string(lp) + "-t" + std::to_string(t) + "-r" + std::to_string(row) + ".bin";
    Ctx c;
    c.setup(m, n, k, lp, t, row, pfile, std::max(64, k / 8), 2);
    gen_kernel<<<m, 256, 0, c.s0>>>(c.A, k, (u32)rng(), true);
    gen_kernel<<<n, 256, 0, c.s0>>>(c.B, k, (u32)rng(), false);
    std::vector<uint8_t> sd(320);
    for (auto& x : sd) x = (uint8_t)rng();
    c.set_seeds(sd.data());
    Lt lt;
    lt_setup(lt, m, n, k, c.A, c.B, c.O[0], c.s0);
    // the checked call: poisoned outputs, then two sampled tiles to OUT/samples/NAME
    gk::poison(c.words[0], (size_t)c.tiles * c.S * 32768, c.s0);
    gk::poison(c.O[0], (size_t)m * n * 4, c.s0);
    gk::poison(c.cm.levels, c.levels_count() * 64, c.s0);
    c.compute(c.s0, 0);
    c.commit(c.s0, 0);
    CK(cudaStreamSynchronize(c.s0));
    CK(cudaGetLastError());
    // the capped commit's root against commit_gpu's small path on the same rows (gpu-sample checks only the sampled leaves)
    if (c.kn.ccta) {
      const std::vector<u64> r1 = c.root();
      const int cc = c.kn.ccta;
      c.kn.ccta = 0;
      c.commit(c.s0, 0);
      c.kn.ccta = cc;
      if (c.root() != r1) { fprintf(stderr, "%s: V31_CCTA %d's root isn't the small path's\n", name, cc); return 4; }
    }
    const int fl = c.flags();
    const u64 wa = sum_counts(c.cntA, m), wbn = sum_counts(c.cntB, n);
    dump_samples(c, out + "/samples/" + name, pfile, (u32)rng());
    if (fl) { fprintf(stderr, "%s: flags %d (1: a wrap list overflowed, 2: a top digit left int8)\n", name, fl); return 4; }
    // CUTLASS's fastest, held to cuBLASLt's D (in dO and O[0]: no arm's time depends on their values, and the overlapped
    // calls after the timing recompute both)
    Cut cut;
    cut_setup(cut, m, n, k, c.A, c.B, c.O[0], c.s0);
    if (cut.best) {
      lt_run(lt, c.A, c.B, c.dO, c.s0);
      cut_run(cut, c.s0);
      unsigned long long* ne = dalloc<unsigned long long>(1);
      unsigned long long hne = 0;
      CK(cudaMemsetAsync(ne, 0, 8, c.s0));
      ne_kernel<<<4 * c.nsm, 256, 0, c.s0>>>(c.dO, c.O[0], (size_t)m * n, ne);
      CK(cudaMemcpyAsync(&hne, ne, 8, cudaMemcpyDeviceToHost, c.s0));
      CK(cudaStreamSynchronize(c.s0));
      if (hne) { fprintf(stderr, "%s: CUTLASS's %s differs from cuBLASLt's D in %llu entries\n", name, cut.name.c_str(), hne); return 4; }
    }
    std::vector<gk::Arm> arms = {
        {"lt", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) lt_run(lt, c.A, c.B, c.O[0], s); }},
        {"plain", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.gemm(s, 0, false); }},
        {"coins", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.coins(s); }},
        {"form", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.form(s); }},
        {"wrap", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.wrap(s); }},
        {"gemm", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.gemm(s, 0, true); }},
        {"commit", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.commit(s, 0); }},
        {"call", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) { c.compute(s, 0); c.commit(s, 0); } }},
        {"ovl", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.call_ovl(s); c.ovl_drain(s); }},
        // the commit's parts: the salt phase, the leaves and nodes, and the leaves' compressions without their rows' loads
        {"salt", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.salts(s); }},
        {"leaf", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.leaves_nodes(s, 0); }},
        {"leaf_nomem", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.leaves_nodes_of(s, NoMemRow{(u32)c.row}); }},
    };
    if (cut.best) arms.push_back({"cutlass", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) cut_run(cut, s); }});
    // V31_DIAG: the protocol GEMM without its segment stores, its tail, its dO read, and all three (timing only)
    if (c.kn.diag)
      for (const auto& [nm, d] : {std::pair<const char*, int>{"gemm_nost", 1}, {"gemm_notail", 2}, {"gemm_nodo", 4}, {"gemm_bare", 7}})
        arms.push_back({nm, [&c, d = d](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.gemm(s, 0, true, d); }});
#ifdef V31_WG
    if (c.kn.gemm) arms.push_back({"wgl", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) c.wgl(s, 0); }});
#endif
    const gk::Timing tm = gk::timed_items(arms, reps, c.s0, nv);
    CK(cudaGetLastError());
    gk::summary(tm, arms, stderr);
    // the timed schedule's own values: two overlapped calls into poisoned buffers, the last one's tiles to NAME-ovl
    for (int b = 0; b < c.nb; b++) {
      gk::poison(c.words[b], (size_t)c.tiles * c.S * 32768, c.s0);
      gk::poison(c.O[b], (size_t)m * n * 4, c.s0);
    }
    gk::poison(c.cm.levels, c.levels_count() * 64, c.s0);
    c.call_ovl(c.s0);
    c.call_ovl(c.s0);
    c.ovl_drain(c.s0);
    CK(cudaStreamSynchronize(c.s0));
    CK(cudaGetLastError());
    dump_samples(c, out + "/samples/" + name + "-ovl", pfile, (u32)rng(), c.nb == 2 ? (c.ovl - 1) & 1 : 0);
    std::vector<double> med(arms.size()), mhz(arms.size());
    for (size_t a = 0; a < arms.size(); a++) {
      std::vector<double> v, f;
      for (const auto& it : tm.items)
        if (it.arm == (int)a) { v.push_back(it.ms / it.calls); if (it.sm_mhz()) f.push_back(it.sm_mhz()); }
      med[a] = gk::median(v);
      mhz[a] = gk::median(f);
    }
    printf("{\"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"lp\": %d, \"t\": %d, \"row\": %d, \"S\": %d, \"dz\": %d, \"dy\": %d, "
           "\"leaves\": %llu, \"gpu\": \"%s\", \"sms\": %d, \"grid\": %d, \"copies\": %d, \"wrap_rate_A\": %.6f, \"wrap_rate_B\": %.6f, "
           "\"lt\": {\"algo\": %d, \"tile\": %d, \"stages\": %d, \"splitk\": %d, \"candidates\": %d}, "
           "\"cutlass\": {\"kernel\": \"%s\", \"candidates\": %d}, \"steady\": %s, "
           "\"carve\": %d, \"dyn\": %d, \"cprio\": \"%s\", \"wrap\": %d, \"wcw\": %d, \"ccta\": %d, \"pf\": %d, \"maxnreg\": %d, "
           "\"wg_nreg\": %d, \"wg_stages\": %d, \"gemm_bn\": %d, \"sched\": %d, \"arms\": {",
           name, m, n, k, lp, t, row, c.S, c.dz, c.dy, (unsigned long long)c.leaves, prop.name, c.nsm, c.grid, c.nb,
           (double)wa / ((double)m * k), (double)wbn / ((double)n * k), lt.algo_id, lt.tile, lt.stages, lt.splitk, lt.ncand,
           cut.name.c_str(), cut.ncand, tm.steady ? "true" : "false", c.kn.carve, c.kn.dyn, c.kn.cprio.c_str(), c.kn.wrap, c.wcw,
           c.kn.ccta, c.kn.pf, V31_MAXNREG, G_WG_NREG, G_WG_STAGES, c.pair ? G_BN : TILE, c.nb == 2 ? c.kn.sched : 0);
    for (size_t a = 0; a < arms.size(); a++)
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f}", a ? ", " : "", arms[a].name.c_str(), med[a], mhz[a]);
    printf("}, \"gamma_call\": %.4f, \"gamma_ovl\": %.4f, \"gamma_gemm\": %.4f, \"plain_over_lt\": %.4f, ", med[7] / med[0],
           med[8] / med[0], med[5] / med[0], med[1] / med[0]);
    if (cut.best)   // arm 12: over CUTLASS's fastest, and the two baselines' ratio
      printf("\"gamma_call_cutlass\": %.4f, \"gamma_ovl_cutlass\": %.4f, \"gamma_gemm_cutlass\": %.4f, \"cutlass_over_lt\": %.4f, ",
             med[7] / med[12], med[8] / med[12], med[5] / med[12], med[12] / med[0]);
    for (size_t a = 0; a < arms.size(); a++)   // V31_GEMM 1: over wgmma.linear, the served GEMM
      if (arms[a].name == "wgl")
        printf("\"gemm_kind\": \"wgmma\", \"wgbn\": %d, \"cm\": %d, \"rp\": %d, \"wgg\": %d, \"gamma_call_wgl\": %.4f, \"gamma_ovl_wgl\": %.4f, "
               "\"gamma_gemm_wgl\": %.4f, \"plain_over_wgl\": %.4f, \"wgl_over_lt\": %.4f, ",
#ifdef V31_WG
               c.wbn,
#else
               0,
#endif
               c.kn.cm, c.kn.rp, c.kn.wgg, med[7] / med[a], med[8] / med[a], med[5] / med[a], med[1] / med[a], med[a] / med[0]);
    printf("\"kernels\": ");
    kernel_attrs(stdout);
    printf("}\n");
    fflush(stdout);
    FILE* items = fopen((out + "/items-" + name + ".jsonl").c_str(), "w");
    if (items) { gk::items_jsonl(tm, arms, items); fclose(items); }
    delete cut.best;
    CK(cudaDeviceSynchronize());
    CK(cudaDeviceReset());
  }
  return 0;
}

// One shape's call (compute, then commit) twice on synthetic operands, then V31_ONCE_OVL (default 0) overlapped calls,
// nothing timed: what a profiler watches.
static int cmd_once(const std::string& pdir, int lp, int t, int row, const std::string& sh) {
  char name[64];
  int m, n, k;
  if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &n, &k) != 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
  const int novl = env_int("V31_ONCE_OVL", 0);
  Ctx c;
  c.setup(m, n, k, lp, t, row, pdir + "/" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) + "-lp" +
          std::to_string(lp) + "-t" + std::to_string(t) + "-r" + std::to_string(row) + ".bin", std::max(64, k / 8),
          novl ? 2 : 1);
  gen_kernel<<<m, 256, 0, c.s0>>>(c.A, k, 1u, true);
  gen_kernel<<<n, 256, 0, c.s0>>>(c.B, k, 2u, false);
  std::vector<uint8_t> sd(320);
  for (size_t i = 0; i < sd.size(); i++) sd[i] = (uint8_t)mix32((u32)i);
  c.set_seeds(sd.data());
  for (int i = 0; i < 2; i++) {
    c.compute(c.s0, 0);
    c.commit(c.s0, 0);
  }
  // V31_ONCE_RANGE ovl: the overlapped calls inside cudaProfilerStart/Stop (ncu's range replay: every pipe over them, the
  // kernels concurrent as they run); wgl: as many wgmma.linear calls instead, the baseline's
  const std::string range = env_str("V31_ONCE_RANGE", "");
  if (!range.empty()) {
    CK(cudaDeviceSynchronize());
    CK(cudaProfilerStart());
  }
  if (range == "wgl") {
#ifdef V31_WG
    for (int i = 0; i < novl; i++) c.wgl(c.s0, 0);
#endif
  } else {
    for (int i = 0; i < novl; i++) c.call_ovl(c.s0);
    c.ovl_drain(c.s0);
  }
  CK(cudaStreamSynchronize(c.s0));
  if (!range.empty()) {
    CK(cudaDeviceSynchronize());
    CK(cudaProfilerStop());
  }
  CK(cudaGetLastError());
  const std::vector<u64> r = c.root();
  printf("%s: flags %d root %016llx\n", name, c.flags(), (unsigned long long)r[0]);
  return c.flags() ? 4 : 0;
}

// v31.py fill_params_name: "served-" (a served call, V31_SERVED=1) or "fill-"
static std::string fill_params_name(int m, int n, int k, int lp, int L, int row, bool o) {
  return std::string(o ? "served-" : "fill-") + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) + "-lp" +
         std::to_string(lp) + "-L" + std::to_string(L) + "-r" + std::to_string(row) + ".bin";
}
// One shape's Fill call twice, each committed before the next is issued, then V31_ONCE_OVL (default 6) back-to-back calls
// (each committed beside the next one's compute), nothing timed: what a profiler watches (run.sh's FILL_NSYS). V31_SERVED=1:
// served calls (A gen_kernel's activations)
static int cmd_fill_once(const std::string& pdir, int lp, int Lspec, int row, const std::string& sh) {
  char name[64];
  int m, n, k;
  if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &n, &k) != 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
  const int L = Lspec > 0 ? Lspec : k, novl = env_int("V31_ONCE_OVL", 6);
  Fill f;
  f.o = env_int("V31_SERVED", 0) != 0;
  f.setup(m, n, k, lp, L, row, read_params(pdir + "/" + fill_params_name(m, n, k, lp, L, row, f.o)), 0);
  int8_t* A = dalloc<int8_t>((size_t)m * k);
  gen_kernel<<<m, 256, 0, f.s0>>>(A, k, 1u, true);
  gen_kernel<<<n, 256, 0, f.s0>>>(f.B, k, 2u, false);
  std::vector<uint8_t> sd(384);
  for (size_t i = 0; i < sd.size(); i++) sd[i] = (uint8_t)mix32((u32)i);
  f.set_seeds(sd.data());
  CK(cudaStreamSynchronize(f.s0));
  for (int i = 0; i < 2; i++) {
    f.issue(f.calls * f.leaves, A);
    f.ready(f.calls - 1);
  }
  for (int i = 0; i < novl; i++) f.issue(f.calls * f.leaves, A, Cheat{}, true);
  f.flush();
  device_sync();
  CK(cudaGetLastError());
  printf("%s: %llu calls (2 serial, %d overlapped, weight coins a call), ltw %d, served %d, gsm %d (%d + %d SMs)\n", name,
         (unsigned long long)f.calls, novl, f.ltw ? 1 : 0, f.o ? 1 : 0, f.sp ? f.sp->n : 0, f.sp ? f.sp->sms[0] : f.nsm,
         f.sp ? f.sp->sms[1] : 0);
  cudaFree(A);
  f.release();
  return 0;
}

// Two sampled tiles of a call (copy b), for `v31.py fill-sample`: the seeds (dither order), the params (Ks with them),
// whether it's served and its salt offset, A's and B's rows of each tile, its values (the tile's leaf rows) and its leaves
// (level 0)
static void fill_samples(Fill& f, int b, const std::string& dir, const std::string& pfile, const std::vector<uint8_t>& sd,
                         const int8_t* A, u32 pick) {
  gk::Dump d(dir);
  writef(dir + "/seeds.bin", sd.data(), sd.size());
  const auto p = readf(pfile);
  writef(dir + "/params.bin", p.data(), p.size());
  const int picks[2] = {0, (int)(mix32(pick) % (u32)f.tiles)};
  std::vector<u64> root(8);
  CK(cudaMemcpy(root.data(), f.cm[b].levels + 8 * (f.levels_count() - 1), 64, cudaMemcpyDeviceToHost));
  FILE* fp = fopen((dir + "/sample.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"lp\": %d, \"L\": %d, \"row\": %d, \"o\": %s, \"off\": %llu, \"tiles\": [%d, %d], "
          "\"root\": \"", f.m, f.n, f.k, f.lp, f.L, f.row, f.o ? "true" : "false", (unsigned long long)f.off[b], picks[0], picks[1]);
  for (int i = 0; i < 64; i++) fprintf(fp, "%02x", (unsigned)(root[i / 8] >> (56 - 8 * (i % 8))) & 0xFF);
  fprintf(fp, "\"}\n");
  fclose(fp);
  const uint8_t* v = f.values_of(b);
  const size_t tb = (size_t)f.lpt * f.row;
  for (int i = 0; i < 2; i++) {
    const int tile = picks[i], tm = tile / f.tiles_n, tn = tile % f.tiles_n;
    const std::string s = "t" + std::to_string(i) + "_";
    d.bytes(s + "A", A + (size_t)tm * TILE * f.k, (size_t)TILE * f.k, f.s0);
    d.bytes(s + "B", f.B + (size_t)tn * TILE * f.k, (size_t)TILE * f.k, f.s0);
    d.bytes(s + "words", v + (size_t)tile * tb, tb, f.s0);
    d.digests(s + "leaves", f.cm[b].levels + 8 * (size_t)tile * f.lpt, f.lpt, f.s0);
  }
}

// Dither's no-fold call (Fill; V31_SERVED=1 a served call, else a filler's) per shape at segment length L (0: L = K), A a
// served operand (gen_kernel's activations), beside cuBLASLt and CUTLASS on the same A and B: one checked call into
// poisoned buffers with two sampled tiles dumped for `v31.py fill-sample`, then gpu_kit's interleaved arms: coins (d_A, V),
// coins_w (d_B, U, the permutations), pad (the filler's pad), form_a, form_b (served: a' and b' with their sums, then X and
// W' when W > 0), gemm (served: cuBLASLt's a' b'^T, then the protocol GEMM over k < W L), commit; call, one call serial
// with the weight side a period's (v3.2); ovl, back-to-back calls each committed beside the next call's compute, the
// weight side a period's; ovl_w, the same with the weight side every call (the run of record's per-call weight coins);
// ovl_pad, the filler's call (the pad and every coin a call); ovl_nomem, ovl with each leaf's bytes made in registers
// (NoMemRow: the commit's compressions without its loads); the commit's parts alone: salt (Ks and the salt phase), leaf
// (the leaf phase and the levels above it), leaf_nomem (leaf on NoMemRow); sha, this build's raw compression rate
// (sha_ruler, commit_gpu's ruler). Call c salts from c leaves. One JSON line a shape, with each part's compressions
// (commit_counts) and its compressions a SM-cycle at its median time and clock. The ovl arms run on V31_GSM's split when
// it's set. V31_GSMS "N ..." (or N_N_...): for each N, gemm_gN (the GEMM alone on a split's N SMs), commit_gN (the commit
// alone on its other SMs) and ovl_w_gN (a filler's ovl_pad_gN: the arm of record's calls on that split; N 0: on every SM,
// with no gemm_g0 or commit_g0), those calls V31_GCALLS (default 8) times an item's, so the pipeline's fill and drain are a
// share that small of each; each split's own two overlapped calls' tiles to NAME-ovl-gN. V31_ITEM_MS: gpu_kit's item
// length (default ITEM_MS).
static int cmd_fill_bench(int reps, const std::string& out, const std::string& pdir, int lp, int Lspec, int row,
                          const std::vector<std::string>& shapes) {
  gk::Nvml nv(0);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::mt19937_64 rng(std::random_device{}());
  const bool served = env_int("V31_SERVED", 0) != 0;
  const int gcalls = std::max(1, env_int("V31_GCALLS", 8));
  const double item_ms = env_int("V31_ITEM_MS", 0) > 0 ? env_int("V31_ITEM_MS", 0) : gk::ITEM_MS;
  std::vector<int> gsms;
  {
    std::string g = env_str("V31_GSMS", "");
    for (char& c : g)
      if (c == ',' || c == '_') c = ' ';
    std::istringstream is(g);
    for (int x; is >> x;)
      if (x >= 0 && std::find(gsms.begin(), gsms.end(), x) == gsms.end()) gsms.push_back(x);
  }
  for (const auto& sh : shapes) {
    char name[64];
    int m, n, k;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &n, &k) != 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    const int L = Lspec > 0 ? Lspec : k;
    const std::string pfile = pdir + "/" + fill_params_name(m, n, k, lp, L, row, served);
    Fill f;
    f.o = served;
    f.setup(m, n, k, lp, L, row, read_params(pfile), 0);
    int8_t* A = dalloc<int8_t>((size_t)m * k);
    int *D = dalloc<int>((size_t)m * n), *D2 = dalloc<int>((size_t)m * n);
    u64* rio = dalloc<u64>((size_t)f.nsm * RULER_CTAS * 256);
    gen_kernel<<<m, 256, 0, f.s0>>>(A, k, (u32)rng(), true);
    gen_kernel<<<n, 256, 0, f.s0>>>(f.B, k, (u32)rng(), false);
    std::vector<uint8_t> sd(384);
    for (auto& x : sd) x = (uint8_t)rng();
    if (served) std::fill(sd.begin(), sd.begin() + 64, 0);
    f.set_seeds(sd.data());
    Lt lt;
    lt_setup(lt, m, n, k, A, f.B, D, f.s0);
    f.poison(0, f.s0);
    gk::poison(f.cm[0].levels, f.levels_count() * 64, f.s0);
    CK(cudaStreamSynchronize(f.s0));
    const int b0 = f.issue(f.calls * f.leaves, A);
    f.ready(f.calls - 1);
    CK(cudaGetLastError());
    fill_samples(f, b0, out + "/samples/" + name, pfile, sd, A, (u32)rng());
    Cut cut;
    cut_setup(cut, m, n, k, A, f.B, D2, f.s0);
    if (cut.best) {
      lt_run(lt, A, f.B, D, f.s0);
      cut_run(cut, f.s0);
      unsigned long long* ne = dalloc<unsigned long long>(1);
      unsigned long long hne = 0;
      CK(cudaMemsetAsync(ne, 0, 8, f.s0));
      ne_kernel<<<4 * f.nsm, 256, 0, f.s0>>>(D, D2, (size_t)m * n, ne);
      CK(cudaMemcpyAsync(&hne, ne, 8, cudaMemcpyDeviceToHost, f.s0));
      CK(cudaStreamSynchronize(f.s0));
      if (hne) { fprintf(stderr, "%s: CUTLASS's %s differs from cuBLASLt's D in %llu entries\n", name, cut.name.c_str(), hne); return 4; }
    }
    // back-to-back calls on p (null: s0 and s1), after s's work so far, then s waits for every commit
    const Split* dflt = f.sp;
    auto ovl = [&f](cudaStream_t s, int r, const int8_t* a, bool wside, const Split* p) {
      f.use(p);
      if (p) split_fork(p, s);
      for (int i = 0; i < r; i++) f.issue(f.calls * f.leaves, a, Cheat{}, wside);
      f.flush();
      for (int b = 0; b < 2; b++) CK(cudaStreamWaitEvent(s, f.ev_done[b]));
    };
    std::vector<gk::Arm> arms = {
        {"lt", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) lt_run(lt, A, f.B, D, s); }},
        {"coins", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) f.coins(s, false, false); }},
        {"coins_w", [&](cudaStream_t s, int r) {
           for (int i = 0; i < r; i++) {
             const u64 bb = (u64)n * (f.kb64 / 64), cu = (u64)n * lp;
             bits_kernel<<<(unsigned)((bb + 127) / 128), 128, 0, s>>>(f.seeds, S_DB, bb, f.dB);
             if (!f.Wn) continue;
             noise_kernel<<<(unsigned)((cu / 16 + 127) / 128), 128, 0, s>>>(f.seeds, S_U, cu, f.U);
             perm_kernel<<<f.NT, PERM_THREADS(lp), lp * 6, s>>>(f.seeds, lp, f.perm);
           }
         }},
        {"form_a", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) f.form(s, A, false, 0); }},
        {"form_b", [&](cudaStream_t s, int r) {
           const u64 cb = (u64)n * k / 16;
           for (int i = 0; i < r; i++) {
             if (f.o) serve_rows_kernel<<<n, 256, 0, s>>>(f.B, f.dB, k, f.kb64, 1, f.Bp, f.rb[0], f.sb[0], f.bad + 2);
             if (f.Wn) fill_b_kernel<<<(unsigned)((cb + 255) / 256), 256, 0, s>>>(f.B, f.dB, f.U, f.perm, f.W, cb, k, lp, f.kb64);
           }
         }},
        {"gemm", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) f.gemm(s, 0); }},
        {"commit", [&](cudaStream_t s, int r) {
           for (int i = 0; i < r; i++) {
             f.set_ks(0, s);
             f.pl.salts(f.cm[0], f.kn.ccta, s, f.leaves);
             f.leaves_nodes(0, s);
           }
         }},
        {"salt", [&](cudaStream_t s, int r) {
           for (int i = 0; i < r; i++) {
             f.set_ks(0, s);
             f.pl.salts(f.cm[0], f.kn.ccta, s, f.leaves);
           }
         }},
        {"leaf", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) f.leaves_nodes(0, s); }},
        {"leaf_nomem", [&](cudaStream_t s, int r) { f.nomem = true; for (int i = 0; i < r; i++) f.leaves_nodes(0, s); f.nomem = false; }},
        {"sha", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) sha_ruler<<<f.nsm * RULER_CTAS, 256, 0, s>>>(rio, RULER_CHAIN); }},
        {"call", [&](cudaStream_t s, int r) {
           for (int i = 0; i < r; i++) {
             f.coins(s, false, false);
             f.form(s, A, false, 0);
             f.gemm(s, 0);
             f.set_ks(0, s);
             f.pl.salts(f.cm[0], f.kn.ccta, s, f.leaves);
             f.leaves_nodes(0, s);
           }
         }},
        {"ovl", [&](cudaStream_t s, int r) { ovl(s, r, A, false, dflt); }},
        {"ovl_w", [&](cudaStream_t s, int r) { ovl(s, r, A, true, dflt); }},
        {"ovl_nomem", [&](cudaStream_t s, int r) { f.nomem = true; ovl(s, r, A, false, dflt); f.nomem = false; }},
    };
    if (!served) {
      arms.push_back({"pad", [&](cudaStream_t s, int r) {
                        const u64 bp = (u64)m * k / 64;
                        for (int i = 0; i < r; i++)
                          bits_kernel<<<(unsigned)((bp + 127) / 128), 128, 0, s>>>(f.seeds, S_PAD, bp, reinterpret_cast<uint8_t*>(f.A));
                      }});
      arms.push_back({"ovl_pad", [&](cudaStream_t s, int r) { ovl(s, r, nullptr, true, dflt); }});
    }
    std::vector<std::pair<std::string, int>> per;   // arms whose item runs this many times its calls
    for (const int gn : gsms) {
      const Split* p = split_of(gn);
      const std::string g = std::to_string(gn);
      f.prep(p);
      if (p) {
        arms.push_back({"gemm_g" + g, [&f, p](cudaStream_t s, int r) {
                          split_fork(p, s);
                          for (int i = 0; i < r; i++) f.gemm(p->s[0], 0);
                          split_join(p, s);
                        }});
        arms.push_back({"commit_g" + g, [&f, p](cudaStream_t s, int r) {
                          split_fork(p, s);
                          for (int i = 0; i < r; i++) {
                            f.set_ks(0, p->s[1], p->cx[1]);
                            f.pl.salts(f.cm[0], f.kn.ccta, p->s[1], f.leaves);
                            f.leaves_nodes(0, p->s[1]);
                          }
                          split_join(p, s);
                        }});
      }
      const std::string rec = (served ? "ovl_w_g" : "ovl_pad_g") + g;
      arms.push_back({rec, [&, p](cudaStream_t s, int r) { ovl(s, r * gcalls, served ? A : nullptr, true, p); }});
      per.push_back({rec, gcalls});
    }
#ifdef V31_WG
    // wgmma.linear on A and B (bf16 values into D2, after CUTLASS's check of it)
    wgp::Args gw{};
    if (served) {
      gw = f.wg_args(A, f.B, 2);
      gw.O = D2;
      arms.push_back({"wgl", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) f.wg_run(s, gw); }});
    }
#endif
    if (const char* only = getenv("V31_ARMS")) {   // a comma list: time only these arms (the rest's numbers null)
      const std::string l = std::string(",") + only + ",";
      arms.erase(std::remove_if(arms.begin(), arms.end(), [&](const gk::Arm& a) { return l.find("," + a.name + ",") == std::string::npos; }),
                 arms.end());
    }
    const int ICUT = (int)arms.size();
    if (cut.best) arms.push_back({"cutlass", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) cut_run(cut, s); }});
    const gk::Timing tm = gk::timed_items(arms, reps, f.s0, nv, item_ms);
    CK(cudaGetLastError());
    gk::summary(tm, arms, stderr);
    // the timed schedule's own values: two overlapped calls into poisoned buffers (on split p), the last one's copy
    auto two = [&](const Split* p) {
      f.use(p);
      for (int b = 0; b < 2; b++) {
        f.poison(b, f.s0);
        gk::poison(f.cm[b].levels, f.levels_count() * 64, f.s0);
      }
      CK(cudaStreamSynchronize(f.s0));
      f.issue(f.calls * f.leaves, A);
      const int c = f.issue(f.calls * f.leaves, A);
      f.ready(f.calls - 2);
      f.ready(f.calls - 1);
      CK(cudaGetLastError());
      return c;
    };
    const int bl = two(dflt);
    // RP 4, 5: the leaves whose row the GEMMs' hash warps hashed whole, and the SHA-512 chunks they hashed, of the last
    // overlapped call (of every row: the prefix-sum rows too under r5w)
    long long fused = -1, fused_blocks = -1;
    const u32 nblk = (row + 17 + 127) / 128;
    if (f.xf[bl]) {
      unsigned long long* ne = dalloc<unsigned long long>(2);
      unsigned long long h[2] = {0, 0};
      CK(cudaMemset(ne, 0, 16));
      fused_count_kernel<<<4 * f.nsm, 256>>>(f.xf[bl], f.leaves, f.xep[bl], nblk, ne);
      CK(cudaMemcpy(h, ne, 16, cudaMemcpyDeviceToHost));
      cudaFree(ne);
      fused = (long long)h[0];
      fused_blocks = (long long)h[1];
    }
    fill_samples(f, bl, out + "/samples/" + name + "-ovl", pfile, sd, A, (u32)rng());
    for (const int gn : gsms) {
      const Split* p = split_of(gn);
      if (p == dflt) continue;
      const int c = two(p);
      fill_samples(f, c, out + "/samples/" + name + "-ovl-g" + std::to_string(gn), pfile, sd, A, (u32)rng());
    }
    f.use(dflt);
    std::vector<double> med(arms.size()), mhz(arms.size()), jc(arms.size()), w(arms.size());
    for (size_t a = 0; a < arms.size(); a++) {
      std::vector<double> v, fq, j, p;
      for (const auto& it : tm.items)
        if (it.arm == (int)a) {
          v.push_back(it.ms / it.calls);
          if (it.sm_mhz()) fq.push_back(it.sm_mhz());
          if (!std::isnan(it.mj())) { j.push_back(it.mj() / 1000 / it.calls); p.push_back(it.mj() / it.ms); }
        }
      med[a] = gk::median(v);
      mhz[a] = gk::median(fq);
      jc[a] = gk::median(j);
      w[a] = gk::median(p);
      for (const auto& x : per)
        if (x.first == arms[a].name) { med[a] /= x.second; jc[a] /= x.second; arms[a].calls *= x.second; }
    }
    auto at = [&](const char* nm) -> double {
      for (size_t a = 0; a < arms.size(); a++)
        if (arms[a].name == nm) return med[a];
      return NAN;
    };
    auto jn = [](double x) {   // a JSON number, or null (an arm this call doesn't run)
      char b[32];
      if (std::isnan(x)) return std::string("null");
      snprintf(b, sizeof b, "%.4f", x);
      return std::string(b);
    };
    printf("{\"fill\": true, \"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"lp\": %d, \"L\": %d, \"row\": %d, \"S\": %d, "
           "\"W\": %d, \"o\": %s, \"leaves\": %llu, \"gpu\": \"%s\", \"sms\": %d, \"grid\": %d, "
           "\"lt\": {\"algo\": %d, \"tile\": %d, \"stages\": %d, \"splitk\": %d, \"candidates\": %d}, "
           "\"cutlass\": {\"kernel\": \"%s\", \"candidates\": %d}, \"steady\": %s, "
           "\"carve\": %d, \"dyn\": %d, \"cprio\": \"%s\", \"ccta\": %d, \"pf\": %d, \"maxnreg\": %d, \"wg_nreg\": %d, "
           "\"wg_stages\": %d, \"gemm_bn\": %d, \"sched\": %d, \"ltw\": %d, \"wg\": %d, \"rp\": %d, \"wbn\": %d, "
           "\"hall\": %d, \"hchunk\": %d, \"r5h\": %d, \"r5w\": %d, \"tst\": %d, \"wgw\": %d, \"lminb\": %d, \"sahead\": %d, "
           "\"fused_leaves\": %lld, "
           "\"o_leaves\": %llu, \"fused_blocks\": %lld, \"row_blocks\": %u, \"gsm\": %d, \"gsm_sms\": [%d, %d], \"gcoins\": %d, "
           "\"gcalls\": %d, \"item_ms\": %.1f, \"gsms\": [",
           name, m, n, k, lp, L, row, f.S, f.Wn, f.o ? "true" : "false", (unsigned long long)f.leaves, prop.name, f.nsm, f.grid,
           lt.algo_id, lt.tile, lt.stages, lt.splitk, lt.ncand, cut.name.c_str(), cut.ncand, tm.steady ? "true" : "false",
           f.kn.carve, f.kn.dyn, f.kn.cprio.c_str(), f.kn.ccta, f.kn.pf, V31_MAXNREG, G_WG_NREG, G_WG_STAGES,
           f.pair ? G_BN : TILE, f.kn.sched, f.ltw ? 1 : 0, f.wg ? 1 : 0, f.kn.rp, f.wbn, env_int("V31_HALL", 0), f.hchunk,
           f.kn.rp == 5 ? f.r5h : 0, f.r5w ? 1 : 0, f.tst ? 1 : 0, f.wgw ? 1 : 0,
           f.kn.lminb, f.kn.sahead ? 1 : 0, fused, f.o ? (unsigned long long)f.tiles * (65536 / row) : 0ull, fused_blocks, nblk,
           dflt ? dflt->n : 0, dflt ? dflt->sms[0] : f.nsm, dflt ? dflt->sms[1] : f.nsm, f.kn.gcoins, gcalls, item_ms);
    for (size_t i = 0; i < gsms.size(); i++) {
      const Split* p = split_of(gsms[i]);
      printf("%s{\"n\": %d, \"sms\": [%d, %d]}", i ? ", " : "", gsms[i], p ? p->sms[0] : f.nsm, p ? p->sms[1] : f.nsm);
    }
    printf("], \"power_limit_w\": %s, \"arms\": {", jn(nv.power_limit_w()).c_str());
    for (size_t a = 0; a < arms.size(); a++)
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"j\": %s, \"w\": %s}", a ? ", " : "", arms[a].name.c_str(),
             med[a], mhz[a], arms[a].calls, jn(jc[a]).c_str(), jn(w[a]).c_str());
    const double l = at("lt");
    printf("}, \"gamma_ovl\": %s, \"gamma_ovl_w\": %s, \"gamma_ovl_pad\": %s, \"gamma_ovl_nomem\": %s, \"gamma_call\": %s, "
           "\"gamma_gemm\": %s, ",
           jn(at("ovl") / l).c_str(), jn(at("ovl_w") / l).c_str(), jn(at("ovl_pad") / l).c_str(), jn(at("ovl_nomem") / l).c_str(),
           jn(at("call") / l).c_str(), jn(at("gemm") / l).c_str());
    const double wl = at("wgl");
    printf("\"gamma_ovl_wgl\": %s, \"gamma_ovl_w_wgl\": %s, \"gamma_call_wgl\": %s, \"gamma_gemm_wgl\": %s, \"wgl_over_lt\": %s, ",
           jn(at("ovl") / wl).c_str(), jn(at("ovl_w") / wl).c_str(), jn(at("call") / wl).c_str(), jn(at("gemm") / wl).c_str(),
           jn(wl / l).c_str());
    if (cut.best)
      printf("\"gamma_ovl_cutlass\": %s, \"gamma_ovl_w_cutlass\": %s, \"gamma_ovl_pad_cutlass\": %s, "
             "\"gamma_gemm_cutlass\": %s, \"cutlass_over_lt\": %s, ",
             jn(at("ovl") / med[ICUT]).c_str(), jn(at("ovl_w") / med[ICUT]).c_str(), jn(at("ovl_pad") / med[ICUT]).c_str(),
             jn(at("gemm") / med[ICUT]).c_str(), jn(med[ICUT] / l).c_str());
    const CommitCounts cc = commit_counts(f.leaves, f.cm[0].width, nblk, f.nsm, f.pl.folds_after(f.kn.ccta));
    auto per_cycle = [&](const char* nm, double c) {   // compressions a SM-cycle at the arm's median time and clock
      for (size_t a = 0; a < arms.size(); a++)
        if (arms[a].name == nm) return c / (med[a] * 1e-3) / ((double)f.nsm * mhz[a] * 1e6);
      return (double)NAN;
    };
    const double rs = per_cycle("sha", cc.sha), rl = per_cycle("leaf", cc.leaf), rn = per_cycle("leaf_nomem", cc.leaf),
                 rt = per_cycle("salt", cc.salt), rc = per_cycle("commit", cc.leaf + cc.salt);
    printf("\"sminb\": %d, \"lfold\": %d, \"mask\": %d, \"full\": %u, \"comp\": {\"sha\": %.0f, \"leaf\": %.0f, \"salt\": %.0f}, "
           "\"per_sm_cycle\": {\"sha\": %s, \"leaf\": %s, \"leaf_nomem\": %s, \"salt\": %s, \"commit\": %s}, \"of_sha\": {\"leaf\": %s, "
           "\"leaf_nomem\": %s, \"salt\": %s, \"commit\": %s}, ",
           f.kn.sminb, f.kn.lfold, f.kn.mask, f.full_rows(), cc.sha, cc.leaf, cc.salt, jn(rs).c_str(), jn(rl).c_str(), jn(rn).c_str(),
           jn(rt).c_str(), jn(rc).c_str(), jn(rl / rs).c_str(), jn(rn / rs).c_str(), jn(rt / rs).c_str(), jn(rc / rs).c_str());
    printf("\"kernels\": ");
    kernel_attrs(stdout);
    printf("}\n");
    fflush(stdout);
    FILE* items = fopen((out + "/items-" + name + ".jsonl").c_str(), "w");
    if (items) { gk::items_jsonl(tm, arms, items); fclose(items); }
    delete cut.best;
    CK(cudaDeviceSynchronize());
    f.drop_splits();
    splits_free();
    CK(cudaDeviceReset());
  }
  return 0;
}

// A step's calls in served order: a Fill for each shape V31_STEP names, each in a domain slot of its own (the shapes share
// ROW, so C), every call's compute on one stream and every commit on another (a split's, on a split), so each call's
// commit runs beside the later calls' compute, as far behind as its own shape's two copies allow. V31_STEP
// "q,kv,q,up,down*52+head": a layer's calls by shape name, the layers (default 1), then the calls once a step. Each call
// forms its weight side (ovl_w's calls) on its own A. For each N of V31_GSMS (default 0, every SM), step_gN: a step an
// item; then on each split, one step into poisoned buffers with each shape's last call's two sampled tiles to
// NAME-step-gN. V31_ITEM_MS as fill-bench's. One JSON line.
static int cmd_fill_step(int reps, const std::string& out, const std::string& pdir, int lp, int Lspec, int row,
                         const std::vector<std::string>& shapes) {
  struct Shape {
    std::string name;
    int m, n, k;
  };
  std::vector<Shape> all, used;
  for (const auto& sh : shapes) {
    char name[64];
    int m, n, k;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &n, &k) != 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    all.push_back({name, m, n, k});
  }
  const std::string spec = env_str("V31_STEP", "");
  std::string layer = spec, tail;
  int layers = 1;
  if (const size_t p = layer.find('+'); p != std::string::npos) { tail = layer.substr(p + 1); layer = layer.substr(0, p); }
  if (const size_t p = layer.find('*'); p != std::string::npos) { layers = atoi(layer.c_str() + p + 1); layer = layer.substr(0, p); }
  auto seq_of = [&](const std::string& s, std::vector<int>& seq) {
    std::istringstream is(s);
    for (std::string nm; std::getline(is, nm, ',');) {
      if (nm.empty()) continue;
      int i = -1;
      for (size_t j = 0; j < used.size() && i < 0; j++)
        if (used[j].name == nm) i = (int)j;
      for (size_t j = 0; j < all.size() && i < 0; j++)
        if (all[j].name == nm) { used.push_back(all[j]); i = (int)used.size() - 1; }
      if (i < 0) { fprintf(stderr, "V31_STEP: no shape named %s among the arguments\n", nm.c_str()); return false; }
      seq.push_back(i);
    }
    return true;
  };
  std::vector<int> lseq, tseq;
  if (!seq_of(layer, lseq) || !seq_of(tail, tseq)) return 2;
  if (lseq.empty() || layers < 1) { fprintf(stderr, "V31_STEP: NAME,NAME...[*LAYERS][+NAME,...], not \"%s\"\n", spec.c_str()); return 2; }
  const size_t ns = used.size();
  if ((int)ns > COMMIT_GPU_DOMS) {
    fprintf(stderr, "V31_STEP: %zu shapes, %d domain slots (build with -DCOMMIT_GPU_DOMS=%zu)\n", ns, COMMIT_GPU_DOMS, ns);
    return 2;
  }
  std::vector<int> gsms;
  {
    std::string g = env_str("V31_GSMS", "0");
    for (char& c : g)
      if (c == ',' || c == '_') c = ' ';
    std::istringstream is(g);
    for (int x; is >> x;)
      if (x >= 0 && std::find(gsms.begin(), gsms.end(), x) == gsms.end()) gsms.push_back(x);
  }
  gk::Nvml nv(0);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::mt19937_64 rng(std::random_device{}());
  const bool served = env_int("V31_SERVED", 0) != 0;
  const double item_ms = env_int("V31_ITEM_MS", 0) > 0 ? env_int("V31_ITEM_MS", 0) : gk::ITEM_MS;
  const Knobs kn;
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t c0, c1;
  CK(cudaStreamCreateWithPriority(&c0, cudaStreamNonBlocking, kn.cprio == "lo" ? hi : lo));
  CK(cudaStreamCreateWithPriority(&c1, cudaStreamNonBlocking, kn.cprio == "hi" ? hi : lo));
  std::vector<Fill> fs(ns);
  std::vector<int8_t*> As(ns);
  std::vector<std::vector<uint8_t>> sds(ns, std::vector<uint8_t>(384));
  std::vector<std::string> pfiles(ns);
  std::vector<int> per_step(ns, 0);
  for (const int x : lseq) per_step[x] += layers;
  for (const int x : tseq) per_step[x]++;
  for (size_t i = 0; i < ns; i++) {
    const Shape& s = used[i];
    const int L = Lspec > 0 ? Lspec : s.k;
    pfiles[i] = pdir + "/" + fill_params_name(s.m, s.n, s.k, lp, L, row, served);
    Fill& f = fs[i];
    f.o = served;
    f.setup(s.m, s.n, s.k, lp, L, row, read_params(pfiles[i]), (int)i, c0, c1);
    As[i] = dalloc<int8_t>((size_t)s.m * s.k);
    gen_kernel<<<s.m, 256, 0, c0>>>(As[i], s.k, (u32)rng(), true);
    gen_kernel<<<s.n, 256, 0, c0>>>(f.B, s.k, (u32)rng(), false);
    for (auto& x : sds[i]) x = (uint8_t)rng();
    if (served) std::fill(sds[i].begin(), sds[i].begin() + 64, 0);
    f.set_seeds(sds[i].data());
  }
  // C is one a process: every shape's params must give the same one, Ks aside (each call sets its own)
  auto c_of = [](Consts c) {
    memset(c.ks, 0, sizeof c.ks);
    memset(c.ks_m8, 0, sizeof c.ks_m8);
    return c;
  };
  for (size_t i = 1; i < ns; i++) {
    const Consts a = c_of(fs[0].cm[0].hc), b = c_of(fs[i].cm[0].hc);
    if (memcmp(&a, &b, sizeof a)) {
      fprintf(stderr, "V31_STEP: %s's params give another C than %s's\n", used[i].name.c_str(), used[0].name.c_str());
      return 2;
    }
  }
  CK(cudaStreamSynchronize(c0));
  auto step = [&](cudaStream_t s, int r, const Split* p) {
    for (Fill& f : fs) f.use(p);
    if (p) split_fork(p, s);
    for (int i = 0; i < r; i++)
      for (int l = 0; l <= layers; l++)
        for (const int x : l < layers ? lseq : tseq) fs[x].issue(fs[x].calls * fs[x].leaves, As[x], Cheat{}, true);
    for (Fill& f : fs) {
      f.flush();
      for (int b = 0; b < 2; b++) CK(cudaStreamWaitEvent(s, f.ev_done[b]));
    }
  };
  std::vector<gk::Arm> arms;
  for (const int gn : gsms) {
    const Split* p = split_of(gn);
    for (Fill& f : fs) f.prep(p);
    arms.push_back({"step_g" + std::to_string(gn), [&step, p](cudaStream_t s, int r) { step(s, r, p); }});
  }
  const gk::Timing tm = gk::timed_items(arms, reps, c0, nv, item_ms);
  CK(cudaGetLastError());
  gk::summary(tm, arms, stderr);
  for (const int gn : gsms) {
    const Split* p = split_of(gn);
    for (Fill& f : fs) {
      f.use(p);
      for (int b = 0; b < 2; b++) {
        f.poison(b, c0);
        gk::poison(f.cm[b].levels, f.levels_count() * 64, c0);
      }
    }
    CK(cudaStreamSynchronize(c0));
    step(c0, 1, p);
    for (size_t i = 0; i < ns; i++) {
      const int b = fs[i].ready(fs[i].calls - 1);
      CK(cudaGetLastError());
      fill_samples(fs[i], b, out + "/samples/" + used[i].name + "-step-g" + std::to_string(gn), pfiles[i], sds[i], As[i],
                   (u32)rng());
    }
  }
  printf("{\"fill_step\": true, \"step\": \"%s\", \"layers\": %d, \"lp\": %d, \"L\": %d, \"row\": %d, \"o\": %s, \"gpu\": \"%s\", "
         "\"sms\": %d, \"cprio\": \"%s\", \"ccta\": %d, \"pf\": %d, \"sched\": %d, \"wg\": %d, \"wgw\": %d, \"rp\": %d, "
         "\"lminb\": %d, \"sahead\": %d, \"gcoins\": %d, \"item_ms\": %.1f, \"steady\": %s, \"shapes\": [",
         spec.c_str(), layers, lp, Lspec, row, served ? "true" : "false", prop.name, fs[0].nsm, kn.cprio.c_str(), kn.ccta, kn.pf,
         kn.sched, fs[0].wg ? 1 : 0, fs[0].wgw ? 1 : 0, kn.rp, kn.lminb, kn.sahead ? 1 : 0, kn.gcoins, item_ms,
         tm.steady ? "true" : "false");
  for (size_t i = 0; i < ns; i++)
    printf("%s{\"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"calls\": %d}", i ? ", " : "", used[i].name.c_str(), used[i].m,
           used[i].n, used[i].k, per_step[i]);
  printf("], \"gsms\": [");
  for (size_t i = 0; i < gsms.size(); i++) {
    const Split* p = split_of(gsms[i]);
    printf("%s{\"n\": %d, \"sms\": [%d, %d]}", i ? ", " : "", gsms[i], p ? p->sms[0] : fs[0].nsm, p ? p->sms[1] : fs[0].nsm);
  }
  auto jn = [](double x) {
    char b[32];
    if (std::isnan(x)) return std::string("null");
    snprintf(b, sizeof b, "%.4f", x);
    return std::string(b);
  };
  printf("], \"power_limit_w\": %s, \"arms\": {", jn(nv.power_limit_w()).c_str());
  for (size_t a = 0; a < arms.size(); a++) {
    std::vector<double> v, fq, j, p;
    for (const auto& it : tm.items)
      if (it.arm == (int)a) {
        v.push_back(it.ms / it.calls);
        if (it.sm_mhz()) fq.push_back(it.sm_mhz());
        if (!std::isnan(it.mj())) { j.push_back(it.mj() / 1000 / it.calls); p.push_back(it.mj() / it.ms); }
      }
    printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"j\": %s, \"w\": %s}", a ? ", " : "", arms[a].name.c_str(),
           gk::median(v), gk::median(fq), arms[a].calls, jn(gk::median(j)).c_str(), jn(gk::median(p)).c_str());
  }
  printf("}}\n");
  fflush(stdout);
  FILE* items = fopen((out + "/items-step.jsonl").c_str(), "w");
  if (items) { gk::items_jsonl(tm, arms, items); fclose(items); }
  CK(cudaDeviceSynchronize());
  for (Fill& f : fs) f.drop_splits();
  splits_free();
  CK(cudaDeviceReset());
  return 0;
}

#ifdef V31_LIB
// ---------------------------------------------------------------- the per-call binding (v31lib.py over ctypes)
// One process, one device: every context shares its compute and commit streams (V31_GSM, V31_GSM_BY: its split's, shared
// by the contexts on that split; calls run in issue order, each commit beside the next call's compute), a domain slot each
// (COMMIT_GPU_DOMS of them) and one ROW (commit_gpu's C fixes it). Every function returns 0, or -1 with the reason in
// v31_error() (the calling thread's); a CUDA failure exits the process (commit_gpu's CK). The thread that made the first
// live context drives every context, and any other is refused.
static thread_local std::string g_err;
static int g_dev = -1, g_row = 0, g_live = 0;
static std::thread::id g_thread;
static bool g_dom[COMMIT_GPU_DOMS] = {};
static cudaStream_t g_s0 = nullptr, g_s1 = nullptr;
struct LibCtx {
  Fill f;
  int dom = -1;
  int8_t* own_b = nullptr;   // the context's own weights buffer (v31_set_b's), where v31_point_b(null) points f.B back
};
static int fail(const std::string& why) {
  g_err = why;
  return -1;
}
static LibCtx* ctx_of(void* h) { return static_cast<LibCtx*>(h); }
static int foreign() { return g_live && std::this_thread::get_id() != g_thread ? fail("another thread made the live contexts") : 0; }
#define V31_FILL(h) if (foreign()) return -1; Fill& f = ctx_of(h)->f

extern "C" {
const char* v31_error() { return g_err.c_str(); }

// A context for calls of m x n x k at noise period lp (L'), segments of L and leaf rows of row bytes, served calls (O
// committed, T_S implied) or a filler's, committed under the CGP1 params file's domain and salted under its Ks
// (v31lib.Prover writes it), on CUDA device dev. Null on failure.
void* v31_create(int m, int n, int k, int lp, int L, int row, int served, const char* params_path, int dev) {
  if (foreign()) return nullptr;
  const std::string bad = Fill::bad_shape(m, n, k, lp, L, row);
  if (!bad.empty()) { fail(bad); return nullptr; }
  if (g_dev >= 0 && dev != g_dev) { fail("one device a process: contexts are on device " + std::to_string(g_dev)); return nullptr; }
  if (g_live && row != g_row) { fail("one ROW a process: live contexts use " + std::to_string(g_row)); return nullptr; }
  int dom = 0;
  while (dom < COMMIT_GPU_DOMS && g_dom[dom]) dom++;
  if (dom == COMMIT_GPU_DOMS) { fail("every domain slot is taken (COMMIT_GPU_DOMS " + std::to_string(COMMIT_GPU_DOMS) + ")"); return nullptr; }
  FILE* pf = fopen(params_path, "rb");
  if (!pf) { fail(std::string("cannot open ") + params_path); return nullptr; }
  fclose(pf);
  CK(cudaSetDevice(dev));
  if (!g_s0) {
    int lo, hi;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    const std::string cp = env_str("V31_CPRIO", "hi");
    CK(cudaStreamCreateWithPriority(&g_s0, cudaStreamNonBlocking, cp == "lo" ? hi : lo));
    CK(cudaStreamCreateWithPriority(&g_s1, cudaStreamNonBlocking, cp == "hi" ? hi : lo));
  }
  device_sync();   // Committer::init rewrites C, which a running salt phase reads
  LibCtx* c = new LibCtx;
  c->f.o = served != 0;
  c->f.setup(m, n, k, lp, L, row, read_params(params_path), dom, g_s0, g_s1);
  c->dom = dom;
  c->own_b = c->f.B;
  g_dom[dom] = true;
  g_dev = dev;
  g_row = row;
  if (!g_live++) g_thread = std::this_thread::get_id();
  return c;
}

// m n k lp L row S tiles lpt leaves depth calls W o, as u64 (W the committed prefix-sum segments: S - 1 served, S filler)
int v31_info(void* h, uint64_t* out) {
  V31_FILL(h);
  const uint64_t v[14] = {(u64)f.m, (u64)f.n, (u64)f.k, (u64)f.lp, (u64)f.L, (u64)f.row, (u64)f.S, (u64)f.tiles, (u64)f.lpt,
                          f.leaves, (u64)f.cm[0].depth, f.calls, (u64)f.Wn, (u64)f.o};
  memcpy(out, v, sizeof v);
  return 0;
}

// The weights (n x k int8, row-major) from host memory into the context's own buffer
int v31_set_b(void* h, const void* B) {
  V31_FILL(h);
  CK(cudaStreamSynchronize(f.cs()));
  f.B = ctx_of(h)->own_b;
  CK(cudaMemcpy(f.B, B, (size_t)f.n * f.k, cudaMemcpyHostToDevice));
  f.wformed = false;
  return 0;
}
// Or at device pointer B, no copy and no wait: a call reads the pointer it was issued with, so B stays unchanged until
// the calls issued after have formed (v31_sync). Null: the context's own buffer again, as v31_set_b last wrote it.
int v31_point_b(void* h, const void* B) {
  V31_FILL(h);
  f.B = B ? (int8_t*)B : ctx_of(h)->own_b;
  f.wformed = false;
  return 0;
}

// A (m x k int8, row-major) from host memory into this context's own operand buffer, in the compute stream's order (after
// every issued call's form); *dev gets the buffer's device pointer, for v31_prove's a_dev. The pad (a null a_dev)
// overwrites the same buffer.
int v31_load_a(void* h, const void* A, void** dev) {
  V31_FILL(h);
  CK(cudaMemcpyAsync(f.A, A, (size_t)f.m * f.k, cudaMemcpyHostToDevice, f.cs()));
  CK(cudaStreamSynchronize(f.cs()));
  *dev = f.A;
  return 0;
}

// One call: seeds (6 x 64 bytes, dither.STREAMS' order: pad, dA, dB, U, V, P; a served call's pad unused), its salt offset
// off (its leaf pos salted prf_salt(Ks, off + pos): off_c = c leaves for the c-th call under one Ks), and A (a device
// pointer to m x k int8, ready on the caller's side; null: the pad, a filler's only). Returns at once with the call's index
// in *call; the compute and commit run on the device. Waits first for call c - 2's commit, whose copy this call takes (its
// values can't be opened after). cheat (testing): "none", "skip:TAU" (noise tile TAU of X zeroed), "forge:S0[:FRAC]" (T_S0
// of every 1/FRAC-th tile overwritten; served, S0 0: O), "forgeO[:FRAC]" (forge:0) or "nodither" (d_A and d_B zero).
int v31_prove(void* h, const uint8_t* seeds, uint64_t off, const void* a_dev, const char* cheat, uint64_t* call) {
  V31_FILL(h);
  const std::string cs = cheat ? cheat : "none";
  if (cs != "none" && cs != "nodither" && cs.compare(0, 5, "skip:") && cs.compare(0, 6, "forge:") && cs.compare(0, 6, "forgeO"))
    return fail("cheat " + cs + ": none | skip:TAU | forge:S0[:FRAC] | forgeO[:FRAC] | nodither");
  const Cheat ch = parse_cheat(cs);
  if (ch.kind == 1 && (ch.tau < 0 || ch.tau >= f.NT)) return fail("skip: a noise tile 0 .. K / L' - 1");
  if (ch.kind == 2 && (ch.s0 < (f.o ? 0 : 1) || ch.s0 > f.Wn || ch.frac <= 0))
    return fail(f.o ? "forge: S0 in 0 (O) .. W, FRAC > 0" : "forge: S0 in 1 .. S, FRAC > 0");
  if (ch.kind == 3 && (!f.o || ch.frac <= 0)) return fail("forgeO: a served call's, FRAC > 0");
  if (f.o && !a_dev) return fail("a served call takes the caller's A");
  if (off + f.leaves < off) return fail("salt offset " + std::to_string(off) + ": off + leaves overflows 64 bits");
  f.set_seeds(seeds);
  *call = f.calls;
  f.issue(off, static_cast<const int8_t*>(a_dev), ch);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return fail("call " + std::to_string(*call) + ": a launch failed: " + cudaGetErrorString(e));
  return 0;
}

// A served call's operands out of what O's decomposition holds: a row of A + d_A or B + d_B with both -128 and 128 (O's
// int8 a' b'^T can't carry it); the call has no root and no openings
static int refused(const Fill& f, int b, uint64_t c) {
  if (!f.o) return 0;
  int bd[4];
  CK(cudaMemcpy(bd, f.bad, 16, cudaMemcpyDeviceToHost));
  if (bd[b]) return fail("call " + std::to_string(c) + ": a row of A + d_A spans -128 .. 128 (served O needs each row in 255 values)");
  if (bd[2 + b]) return fail("call " + std::to_string(c) + ": a row of B + d_B spans -128 .. 128 (served O needs each row in 255 values)");
  return 0;
}

// Call c's root (64 bytes), once its commit is done
int v31_root(void* h, uint64_t c, uint8_t* out) {
  V31_FILL(h);
  const int b = f.ready(c);
  if (b < 0) return fail("call " + std::to_string(c) + " isn't one of the last two calls");
  if (refused(f, b, c)) return -1;
  std::vector<u64> r(8);
  CK(cudaMemcpy(r.data(), f.cm[b].levels + 8 * (f.levels_count() - 1), 64, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 8; i++) sha::put_be64(out + 8 * i, r[i]);
  return 0;
}

// Call c's leaves at pos[0 .. np): their rows (np x row bytes) and paths (np x depth digests, the leaf's sibling first)
int v31_open(void* h, uint64_t c, const uint64_t* pos, int np, uint8_t* rows, uint8_t* paths) {
  V31_FILL(h);
  const int b = f.ready(c);
  if (b < 0) return fail("call " + std::to_string(c) + " isn't one of the last two calls");
  if (refused(f, b, c)) return -1;
  if (np <= 0) return 0;
  for (int i = 0; i < np; i++)
    if (pos[i] >= f.leaves) return fail("leaf " + std::to_string(pos[i]) + " of " + std::to_string(f.leaves));
  f.open(b, pos, np, rows, paths);
  return 0;
}

// Testing: call c's committed values (leaves x row bytes, dither.values_buffer: per tile its W prefix-sum segments, then its O
// rows when served), or the A operand its last call formed from (the pad, m x k bytes)
int v31_values(void* h, uint64_t c, uint8_t* out) {
  V31_FILL(h);
  const int b = f.ready(c);
  if (b < 0) return fail("call " + std::to_string(c) + " isn't one of the last two calls");
  CK(cudaMemcpy(out, f.values_of(b), (size_t)f.leaves * f.row, cudaMemcpyDeviceToHost));
  return 0;
}
// Call c's every tree level (leaves first, (2 width - 1) nodes of 8 native u64 words) into page-locked dst on the commit
// stream behind c's commit, at once: dst is whole after v31_sync, and call c + 2's commit, which reuses the copy's buffer,
// runs after the copy in that stream's order
int v31_levels_async(void* h, uint64_t c, void* dst) {
  V31_FILL(h);
  if (c >= f.calls || c + 2 < f.calls) return fail("call " + std::to_string(c) + " isn't one of the last two calls");
  if (f.pend == (int)(c & 1)) f.flush();
  CK(cudaMemcpyAsync(dst, f.cm[c & 1].levels, f.levels_count() * 64, cudaMemcpyDeviceToHost, f.ms()));
  return 0;
}
int v31_pad(void* h, uint8_t* out) {
  V31_FILL(h);
  CK(cudaStreamSynchronize(f.cs()));
  CK(cudaMemcpy(out, f.A, (size_t)f.m * f.k, cudaMemcpyDeviceToHost));
  return 0;
}

// Every issued call's compute and commit done
int v31_sync(void* h) {
  V31_FILL(h);
  f.flush();
  CK(cudaStreamSynchronize(f.cs()));
  CK(cudaStreamSynchronize(f.ms()));
  if (f.ss()) CK(cudaStreamSynchronize(f.ss()));
  return 0;
}

void v31_destroy(void* h) {
  LibCtx* c = ctx_of(h);
  if (!c || foreign()) return;
  c->f.flush();
  c->f.B = c->own_b;
  c->f.release();
  g_dom[c->dom] = false;
  if (--g_live == 0) g_dev = -1;
  delete c;
}

// cuBLASLt's D (m x n int32) = A (m x k int8) B^T (B n x k), row-major, the GEMM a V31_LTW call makes, from any thread:
// lt_setup autotuned at these pointers on stream st, then lt_run at the same shapes; lt_algo: algo, tile, stages, split-K
void* v31_lt_create(int m, int n, int k, const void* A, const void* B, void* D, void* st) {
  Lt* L = new Lt();
  lt_setup(*L, m, n, k, (const int8_t*)A, (const int8_t*)B, (int*)D, (cudaStream_t)st);
  return L;
}
void v31_lt_run(void* h, const void* A, const void* B, void* D, void* st) {
  lt_run(*(Lt*)h, (const int8_t*)A, (const int8_t*)B, (int*)D, (cudaStream_t)st);
}
void v31_lt_algo(void* h, int* out) {
  const Lt& L = *(Lt*)h;
  out[0] = L.algo_id, out[1] = L.tile, out[2] = L.stages, out[3] = L.splitk;
}
void v31_lt_destroy(void* h) { lt_free(*(Lt*)h); delete (Lt*)h; }
}   // extern "C"
#else
int main(int argc, char** argv) {
  if (argc >= 9 && !strcmp(argv[1], "fill-bench"))
    return cmd_fill_bench(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atoi(argv[7]),
                          std::vector<std::string>(argv + 8, argv + argc));
  if (argc >= 9 && !strcmp(argv[1], "fill-step"))
    return cmd_fill_step(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atoi(argv[7]),
                         std::vector<std::string>(argv + 8, argv + argc));
  if (argc == 7 && !strcmp(argv[1], "once")) return cmd_once(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6]);
  if (argc == 7 && !strcmp(argv[1], "fill-once"))
    return cmd_fill_once(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6]);
  if (argc >= 10 && !strcmp(argv[1], "gate"))
    return cmd_gate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), argv[9]);
  if (argc >= 9 && !strcmp(argv[1], "bench"))
    return cmd_bench(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atoi(argv[7]),
                     std::vector<std::string>(argv + 8, argv + argc));
  if (argc >= 2 && !strcmp(argv[1], "attrs")) {
    kernel_attrs(stdout);
    printf("\n");
    return 0;
  }
  fprintf(stderr, "usage: v31 gate DIR M N K LP T ROW CHEAT | bench REPS OUT PDIR LP T ROW name:M:N:K... | once PDIR LP T ROW "
                  "name:M:N:K | fill-bench REPS OUT PDIR LP L ROW name:M:N:K... (L 0: K) | fill-step REPS OUT PDIR LP L ROW "
                  "name:M:N:K... (V31_STEP) | fill-once PDIR LP L ROW name:M:N:K | attrs\n");
  return 2;
}
#endif
