// ACCEL v3.1's credited static INT8 matmul on the GPU (verity/core/protocols/pouw/accel/PROTOCOL.md on #4746), the
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
//            of L = t L' inner indices every thread stores its accumulators' low 16 bits (T_s) to the words buffer; then
//            P_A P_B^T's first two blocks (-Z_0 U^T - V Y_0^T) as more main stages, the tail stages subtracting the rest
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
//   v31 contend REPS name:M:N:K...        cuBLASLt's GEMM beside a filler kernel of one instruction mix (cmd_contend)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1
//        [-DV31_CUTLASS cutlass_2x.o] v31.cu -lcuda -lcublasLt (run.sh builds the harness's cutlass_2x.cu)
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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
// prg(seed, tag, idx) = SHA-512(seed || tag (8 bytes, zero padded) || be64(idx)): one block of 80 bytes
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

// d_A or d_B: block b = i ceil(K / 512) + k / 512 of row i, its 64 bytes in digest order (bit k % 512 LSB first)
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
// threads a tile and 6 L' bytes of shared memory
__host__ __device__ constexpr int PERM_THREADS(int lp) { return lp < 1024 ? lp : 1024; }
__global__ void perm_kernel(Seeds sd, int lp, uint16_t* __restrict__ perm) {
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
// (WRAP5: w_W to ww, N x K of -1, 0 or 1, wrap5's B operand; a parameter of its own, since a FormArgs field more costs
// form_a_kernel 72 registers)
template <bool WRAP5>
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
      if (WRAP5) ww[j * f.K + k] = (int8_t)w;
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
// group g) over the tile's 32-column blocks g, g + G, ... (G = blockDim / 128), from B~^T's row k of each wrap (k, s)
template <int CW>
DEV void wrap_a_side(const WrapArgs& a, size_t i0, size_t j0) {
  const int t = threadIdx.x, r = t & 127, g = t >> 7, G = blockDim.x >> 7;
  const int na = a.cntA[i0 + r];
  if (na == 0) return;
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
      int4 x = d[u];
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

// wrap5's A side: w_X B~^T added into dO's 128 x 128 tile a CTA, after the plain GEMM (X, w_W) has stored X w_W^T there
__global__ void __launch_bounds__(256) wrap_a_kernel(WrapArgs a) {
  wrap_a_side<TILE>(a, (size_t)blockIdx.y * TILE, (size_t)blockIdx.x * TILE);
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
  int* O;            // m x N, row-major (u32 mod 2^32); null: none (the filler commits only its words)
  int M, N, K, L, S, TT, group;
  int lp, dz;        // tail stage ts is block b = ts 128 / L' of P_A P_B^T: b 0 and 1 (-Z_0 U^T, -V Y_0^T) add as main
                     // stages do; then Z_{b - 1} U^T (b <= dz) or V Y_{b - dz}^T, at 128^digit, subtract
  u32* ctr;          // the tile counter (null: CTA c takes tiles c, c + grid, ...); a launch advances it by its tiles
  u32 base;          // ctr's value before this launch
  int diag;          // timing only (the bench's gemm_no* arms, wrong values): 1 no segment stores, 2 no tail (TT 0), 4 no dO
};
// 128 x 128 tiles, 8 warps of 64 x 32 (MI = 4 row blocks of 16, NI = 4 column blocks of 8), a 3-stage ring of 128 bytes of
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
  // for each row block), so the tail holds 16 more registers than the mainloop, not 48: the GEMM stays small enough for
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
// transpose the quad's 4 x 4 words, so lane q stores columns 8 q .. 8 q + 7 as one 16-byte store and a warp's store
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
template <bool PROTO>
__global__ void __maxnreg__(V31_MAXNREG) gemm_kernel(const GArgs a) {
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

// ---------------------------------------------------------------- cheats (a prover the verifier must refuse)
// forge: the T_s0 words (s0 >= 1) or (s0 = 0) the O values of every stride-th tile, overwritten with hash noise
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
// pf (hash_after_block's prefetch of the next 128-byte block, V31_PF): 1 into L1, 2 into L2
DEV void pf_to(const uint8_t* p, u32 pf) {
  if (pf == 2) asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
  else pf_l1(p);
}
// a row's 4-byte cell c (from c0) at base + (c >> 7) stride + (c & 127) 4: O's row h of a tile (the tile's first cell,
// stride N 4, c0 = h ROW / 4), or a words row (stride 512 and c0 = 0: contiguous)
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
// leaf pos: tile pos / LPT; j = pos % LPT < S ROW_W a words row (segment j / ROW_W), else O's row j - S ROW_W. Both kinds
// go through one CellRow, so a warp whose leaves straddle the two (most warps: LPT isn't a multiple of 32) hashes once,
// not once a kind.
struct V31Row {
  const uint8_t* words;
  const uint8_t* O;
  u64 lpt, srw, N, tiles_n;
  u32 row, S, pf;
  DEV void digest(u64 pos, u64* x) const {
    const u64 t = pos / lpt, j = pos % lpt;
    const bool w = j < srw;
    const uint8_t* base = w ? words + t * S * 32768ull + j * row : O + ((t / tiles_n) * TILE * N + (t % tiles_n) * TILE) * 4;
    hash_after_block(C.mid_P, CellRow{base, w ? 512ull : N * 4, w ? 0u : (u32)((j - srw) * row / 4), pf}, row, x, pf != 0);
  }
};
// timing only (the bench's leaf_nomem arm): a row's bytes made in registers, V31Row's compressions without its loads
struct NoMemSrc {
  u32 seed;
  DEV uint4 ld16(u32 off) const { return make_uint4(seed ^ off, seed + off, seed * 3u ^ off, off); }
  DEV void pf(u32) const {}
};
struct NoMemRow {
  u32 row;
  DEV void digest(u64 pos, u64* x) const { hash_after_block(C.mid_P, NoMemSrc{(u32)pos}, row, x); }
};
// A no-fold call's words rows (S = 1) read from cuBLASLt's int32 D = X W'^T (m x N row-major), its low halves: the row's
// 16-byte piece at off is cells c0 + off / 2 .. + 7 of the tile at base (one tile row's 8 adjacent cells, 32 bytes of D),
// the bytes V31Row reads from the tile-major words buffer
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
struct LtRow {
  const int* D;
  u64 lpt, N, tiles_n;
  u32 row, pf;
  DEV LtSrc src(u64 pos) const {
    const u64 t = pos / lpt, j = pos % lpt;
    return LtSrc{D + (t / tiles_n) * TILE * N + (t % tiles_n) * TILE, N, (u32)(j * row / 2), pf};
  }
  DEV void digest(u64 pos, u64* x) const { hash_after_block(C.mid_P, src(pos), row, x, pf != 0); }
};
// A served call's rows (dither.Shape(o=True)): per tile its W words rows (the words buffer, W tile-segments a tile), then
// its O rows: cells c0 + off / 4 .. + 3 of the tile, O = D + tau_j ra_i + sigma_i rb_j mod 2^32 (serve_rows_kernel: rb
// carries K tau_j; D = a' b'^T from cuBLASLt). One source for both kinds, so a warp whose leaves straddle them diverges only
// in its loads; the pointers are the tile's first row and column.
struct ServedSrc {
  const uint8_t* w;   // a words row, or null: an O row
  const int *D, *ra, *sa, *rb, *sb;
  u64 N;
  u32 c0, pfm;
  DEV const int* at(u32 off) const {
    const u32 c = c0 + off / 4;
    return D + (u64)(c >> 7) * N + (c & 127);
  }
  DEV uint4 ld16(u32 off) const {
    if (w) return ldg16(w + off);
    const u32 c = c0 + off / 4, r = c >> 7, col = c & 127;
    const uint4 d = __ldg(reinterpret_cast<const uint4*>(D + (u64)r * N + col));
    const uint4 b = __ldg(reinterpret_cast<const uint4*>(rb + col)), t = __ldg(reinterpret_cast<const uint4*>(sb + col));
    const u32 a = (u32)__ldg(ra + r), s = (u32)__ldg(sa + r);
    return make_uint4(d.x + t.x * a + s * b.x, d.y + t.y * a + s * b.y, d.z + t.z * a + s * b.z, d.w + t.w * a + s * b.w);
  }
  DEV void pf(u32 off) const { pf_to(w ? w + off : reinterpret_cast<const uint8_t*>(at(off)), pfm); }
};
struct ServedRow {
  const uint8_t* words;
  const int *D, *ra, *sa, *rb, *sb;
  u64 lpt, wrows, N, tiles_n;
  u32 row, W, pf;
  DEV ServedSrc src(u64 pos) const {
    const u64 t = pos / lpt, j = pos % lpt, tm = t / tiles_n, tn = t % tiles_n;
    const bool w = j < wrows;
    return ServedSrc{w ? words + t * W * 32768ull + j * row : nullptr, D + tm * TILE * N + tn * TILE, ra + tm * TILE,
                     sa + tm * TILE, rb + tn * TILE, sb + tn * TILE, N, w ? 0u : (u32)((j - wrows) * row / 4), pf};
  }
  DEV void digest(u64 pos, u64* x) const { hash_after_block(C.mid_P, src(pos), row, x, pf != 0); }
};

// ---------------------------------------------------------------- the commit in at most cap CTAs an SM
// commit_gpu's small path launches a CTA for every 128 leaves, and up to 5 of them sit on an SM, so a GEMM CTA launched
// beside the commit waits for an SM to drain while the commit's next CTAs take each freed slot. salt_persist and
// leaf_persist do the same work (salt_small's and leaf_small's, block by block) in at most cap CTAs an SM: a CTA claims
// one of its SM's slots (claim[smid] = epoch << 8 | count, epoch new each launch) or leaves at once, and the claimants take
// 128-leaf blocks from ctl[0] until none are left. The last CTA out zeroes the counters for the next launch; launches that
// share ctl run in one stream's order.
struct Persist {
  u32* ctl;     // [0] the next block, [1] the CTAs out, [2 + b] CTA b's current block
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
// the CTA's next block (nvb or more: none left), the same for every thread
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
// fold_cta for block vb
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
DEV int log2_cnt(u32 cnt) {
  int L = 0;
  while ((1u << L) < cnt) ++L;
  return L;
}
// salt_one<true, false, true> at the call's salt offset: S[pos] from prf_salt(Ks, off + pos) (dither.salts(Ks, off))
DEV void salt_one_at(u64 pos, u64 off, const uint8_t* T, u64* __restrict__ S) {
  u64 Y[24], c[8], my[8];
  u32 m[16];
  salt_prf(off + pos, Y);
  hm96_mask<false, true>(T, Y, m);
  salt_digest(Y, c);
#pragma unroll
  for (int i = 0; i < 8; ++i) my[i] = (u64)bswap32(m[2 * i]) << 32 | bswap32(m[2 * i + 1]);
  st8(S + 16 * pos, my);
  st8(S + 16 * pos + 8, c);
}
// salt_small<true> (slot 0) a block at a time: S and F for positions below n (salted at off + pos), the pads up to width,
// and the subtrees of blocks that hold only pads (nvb = ceil(width / 128))
__global__ void __launch_bounds__(SMALL_THREADS, SMALL_MINB)
    salt_persist(Persist p, int dom, const uint4* __restrict__ Tg, u64 n, u64 width, u64* __restrict__ S, u64* __restrict__ F,
                 u64* levels, u64 off) {
  const u32 cnt = (u32)(width < SMALL_THREADS ? width : SMALL_THREADS);
  if (persist_claim(p))
    for (u32 vb; (vb = persist_next(p)) < p.nvb;) {
      const u64 pos = (u64)vb * SMALL_THREADS + threadIdx.x;
      if (pos < n) {
        salt_one_at(pos, off, reinterpret_cast<const uint8_t*>(Tg), S);
        u64 f[8];
        frame_first(dom, pos, f);
        st8(F + 8 * pos, f);
      } else if (pos < width) {
        u64 o[8];
        pad(dom, pos, o);
        st8(levels + 8 * pos, o);
      }
      if ((u64)vb * cnt >= n) fold_vb(dom, levels, width, cnt, log2_cnt(cnt), vb);
    }
  persist_done(p);
}
// leaf_small (g = 1, no bc, no fused top) a block at a time (nvb = ceil(n / 128))
template <class Row>
__global__ void __launch_bounds__(SMALL_THREADS, SMALL_MINB)
    leaf_persist(Persist p, int dom, Row row, const u64* __restrict__ S, const u64* __restrict__ F, u64 n, u64 width,
                 u64* levels) {
  const u32 cnt = (u32)(width < SMALL_THREADS ? width : SMALL_THREADS);
  if (persist_claim(p))
    for (u32 vb; (vb = persist_next(p)) < p.nvb;) {
      const u64 pos = (u64)vb * SMALL_THREADS + threadIdx.x;
      if (pos < n) {
        u64 x[8], out[8];
        row.digest(pos, x);
        leaf_from_digest<0>(dom, pos, x, S, out, nullptr, F);
        st8(levels + 8 * pos, out);
      }
      fold_vb(dom, levels, width, cnt, log2_cnt(cnt), vb);
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
// heuristic's candidates (5 calls each)
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
  for (int i = 0; i < got; i++) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    bool ok = true;
    for (int w = 0; w < 2 && ok; w++)
      ok = cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, L.ws, L.wsz, st) == CUBLAS_STATUS_SUCCESS;
    if (!ok) continue;
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
// only: that entry alone (by name)
static void cut_setup(Cut& C, int m, int n, int k, const int8_t* A, const int8_t* B, int* D, cudaStream_t st,
                      const char* only = nullptr) {
#ifdef V31_CUTLASS
  int count = 0;
  const phc_entry* tab = phc_table_2x(&count);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float best = 1e30f;
  for (int i = 0; i < count; i++) {
    if (tab[i].kind != 7 || !tab[i].query || (only && strcmp(tab[i].name, only))) continue;
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
  (void)C; (void)m; (void)n; (void)k; (void)A; (void)B; (void)D; (void)st; (void)only;
#endif
}
static void cut_run(Cut& C, cudaStream_t st) {
  if (C.best->launch(st)) { fprintf(stderr, "CUTLASS's %s didn't launch\n", C.name.c_str()); exit(3); }
}
static void cut_free(Cut& C) {
  delete C.best;
  if (C.ws) CK(cudaFree(C.ws));
  C = Cut{};
}
__global__ void ne_kernel(const int* a, const int* b, size_t n, unsigned long long* out) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += a[i] != b[i];
  if (c) atomicAdd(out, c);
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
// driver's choice), so commit_gpu's small-path CTAs (no shared memory) can sit beside a GEMM CTA; V31_DYN: the GEMM takes
// its tiles from a counter; V31_CPRIO: the commit stream's priority against the compute stream's (hi, lo or eq); V31_WRAP:
// the wrap correction's kernel (1: wrap_kernel, 2: wrap2_kernel, 3: wrap3_kernel, 4: wrap4_kernel; 5: X w_W^T on the
// tensor cores, gemm_kernel<false> on X and w_W (form_b writes w_W), then wrap_a_kernel), V31_WCW its
// columns a CTA (256 where N allows, or 128; 1 to 4);
// V31_CCTA: the salt and leaf phases' CTAs an SM (salt_persist, leaf_persist; 0: commit_gpu's salt_small and leaf_small);
// V31_PF: each row's next 128-byte block prefetched as its compression starts (0 none, 1 into L1, 2 into L2); V31_DIAG 1:
// the bench's timing-only gemm_no* arms; V31_SCHED: the overlapped calls' order (Ctx::call_ovl).
struct Knobs {
  int carve = env_int("V31_CARVE", 100), dyn = env_int("V31_DYN", 1), wrap = env_int("V31_WRAP", 2), wcw = env_int("V31_WCW", 256),
      ccta = env_int("V31_CCTA", 1), pf = env_int("V31_PF", 0), diag = env_int("V31_DIAG", 0), sched = env_int("V31_SCHED", 0),
      ltw = env_int("V31_LTW", 0);   // Fill at S = 1: cuBLASLt's int32 X W'^T in place of the protocol GEMM, rows its low halves
  // Fill: copies of a call's values (2 or 3; call c's stay until call c + copies), and lstr 1: each copy's leaf phase on a
  // stream (and persistent counters) of its own, so the leaf phases of consecutive calls overlap; fstr 1: each call's coins
  // and form on a stream of their own into the other of two X (a') and W' (b') buffers, beside the previous call's GEMM,
  // that stream's priority fprio (hi or lo)
  int copies = env_int("V31_COPIES", 2), lstr = env_int("V31_LSTR", 0), fstr = env_int("V31_FSTR", 0);
  std::string cprio = env_str("V31_CPRIO", "hi"), fprio = env_str("V31_FPRIO", "hi");
};
static void set_carveout(int pct) {
  if (pct < 0) return;
  const void* ks[] = {(const void*)gemm_kernel<true>,  (const void*)gemm_kernel<false>,  (const void*)wrap_kernel,
                      (const void*)wrap2_kernel<128>,  (const void*)wrap2_kernel<256>,   (const void*)wrap3_kernel<128>,
                      (const void*)wrap3_kernel<256>,  (const void*)wrap4_kernel<128>,  (const void*)wrap4_kernel<256>,
                      (const void*)wrap_a_kernel,      (const void*)form_a_kernel,       (const void*)form_b_kernel<false>,
                      (const void*)form_b_kernel<true>, (const void*)bits_kernel,         (const void*)noise_kernel,
                      (const void*)perm_kernel,        (const void*)salt_small<true>,    (const void*)leaf_small<V31Row, 0>,
                      (const void*)level_kernel,       (const void*)top_small,           (const void*)salt_persist,
                      (const void*)leaf_persist<V31Row>, (const void*)leaf_small<NoMemRow, 0>, (const void*)leaf_persist<NoMemRow>,
                      (const void*)leaf_small<LtRow, 0>, (const void*)leaf_persist<LtRow>,  (const void*)fill_a_kernel,
                      (const void*)fill_b_kernel,      (const void*)leaf_small<ServedRow, 0>, (const void*)leaf_persist<ServedRow>,
                      (const void*)serve_rows_kernel};
  for (const void* k : ks) CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
}

// A committer's salt and leaf phases in at most ccta CTAs an SM (salt_persist, leaf_persist; ccta 0: commit_gpu's small
// path), then its levels above the leaves' CTAs. Launches that share ctl run in one stream's order.
struct Lanes {
  u32 *ctl = nullptr, *claim = nullptr, epoch = 0;
  int nsm = 1;
  void init(int nsm_) {
    nsm = nsm_;
    ctl = dalloc<u32>(2 + 2 * 8 * (size_t)nsm);
    claim = dalloc<u32>(1024);
    CK(cudaMemset(ctl, 0, 4 * (2 + 2 * 8 * (size_t)nsm)));
    CK(cudaMemset(claim, 0, 4096));
  }
  // a capped launch over nvb blocks: twice the CTAs the caps allow, so SMs whose first CTAs land late still get theirs
  Persist persist(u64 nvb, int ccta) {
    epoch = (epoch + 1) & 0xFFFFFF;
    return Persist{ctl, claim, epoch, (u32)ccta, (u32)nvb};
  }
  unsigned grid(u64 nvb, int ccta) const { return (unsigned)std::min<u64>(nvb, 2ull * ccta * nsm); }
  // leaf pos salted prf_salt(Ks, off + pos); commit_gpu's small path (ccta 0) salts at offset 0 only
  void salts(Committer& cm, int ccta, cudaStream_t st, u64 off = 0) {
    if (!ccta && !off) { cm.salts<true>(nullptr, st); return; }
    if (!ccta) ccta = 8;
    const u64 nvb = (cm.width + SMALL_THREADS - 1) / SMALL_THREADS;
    salt_persist<<<grid(nvb, ccta), SMALL_THREADS, 0, st>>>(persist(nvb, ccta), cm.dom, cm.Tn, cm.n, cm.width, cm.S, cm.F,
                                                             cm.levels, off);
  }
  template <class Row>
  void leaves_nodes(Committer& cm, const Row& rows, int ccta, cudaStream_t st) {
    if (!ccta) {
      cm.leaves(rows, st);
    } else {
      const u64 nvb = (cm.n + SMALL_THREADS - 1) / SMALL_THREADS;
      leaf_persist<Row><<<grid(nvb, ccta), SMALL_THREADS, 0, st>>>(persist(nvb, ccta), cm.dom, rows, cm.S, cm.F, cm.n,
                                                                   cm.width, cm.levels);
    }
    cm.nodes(st);
  }
};

// One shape's prover: every buffer of a call, NB copies of the committed values (words, O) for the overlapped arm.
struct Ctx {
  int m, n, k, lp, t, row, L, S, NT, kb64, dz, dy, TL, TT, tiles, tiles_n, lpt, grid, nsm, capA, capB, nb = 1;
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

  void setup(int m_, int n_, int k_, int lp_, int t_, int row_, const std::string& params_path, int cap, int want_nb) {
    m = m_; n = n_; k = k_; lp = lp_; t = t_; row = row_;
    if (m % TILE || n % TILE || k % BK || lp % BK || k % lp || 32768 % row || row % 16) {
      fprintf(stderr, "shape m %d n %d k %d L' %d ROW %d: m, N, K, L' multiples of 128, L' | K, ROW | 32768\n", m, n, k, lp, row);
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
    grid = std::min(tiles, nsm);
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
    if (kn.wrap == 5) {
      WW = dalloc<int8_t>(nk);
      const Maps mw{tmap(X, k, m, TILE), tmap(WW, k, n, TILE), tmap(PA, TL, m, TILE), tmap(PB, TL, n, TILE)};
      maps_dw = dalloc<Maps>(1);
      CK(cudaMemcpy(maps_dw, &mw, sizeof mw, cudaMemcpyHostToDevice));
    }
    CK(cudaFuncSetAttribute(gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Ring::SMEM));
    CK(cudaFuncSetAttribute(gemm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Ring::SMEM));
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
    if (WW) form_b_kernel<true><<<n / 16, lp, 16 * lp, st>>>(fb(), WW);
    else form_b_kernel<false><<<n / 16, lp, 16 * lp, st>>>(fb(), nullptr);
  }
  void wrap(cudaStream_t st) {
    const WrapArgs w{XT, BtT, listA, cntA, listB, cntB, dO, m, n, capA, capB, offB, lp, NT};
    if (kn.wrap == 5) {
      gemm_kernel<false><<<grid, Ring::NCT, Ring::SMEM, st>>>(
          GArgs{maps_dw, nullptr, nullptr, dO, m, n, k, L, S, 0, 8, lp, dz, kn.dyn ? gctr : nullptr, gbase, 0});
      gbase += (u32)tiles;
      wrap_a_kernel<<<dim3(n / TILE, m / TILE), 256, 0, st>>>(w);
    } else if (kn.wrap == 1) wrap_kernel<<<dim3(n / TILE, m / TILE), 256, TILE * 129 * 4, st>>>(w);
    else if (kn.wrap == 3 && wcw == 256) wrap3_kernel<256><<<dim3(n / 256, m / TILE), 256, (lp + 1) * 128, st>>>(w);
    else if (kn.wrap == 3) wrap3_kernel<128><<<dim3(n / 128, m / TILE), 256, (lp + 1) * 128, st>>>(w);
    else if (kn.wrap == 4 && wcw == 256) wrap4_kernel<256><<<dim3(n / 256, m / TILE), 256, (lp + 1) * 128, st>>>(w);
    else if (kn.wrap == 4) wrap4_kernel<128><<<dim3(n / 128, m / TILE), 256, (lp + 1) * 128, st>>>(w);
    else if (wcw == 256) wrap2_kernel<256><<<dim3(n / 256, m / TILE), 512, lp * 128, st>>>(w);
    else wrap2_kernel<128><<<dim3(n / 128, m / TILE), 256, lp * 128, st>>>(w);
  }
  // launches that share gctr run in one stream's order, so each one's base is the tiles launched before it
  void gemm(cudaStream_t st, int b, bool proto, int diag = 0) {
    if (proto) gemm_kernel<true><<<grid, Ring::NCT, Ring::SMEM, st>>>(gargs(b, true, diag));
    else gemm_kernel<false><<<grid, Ring::NCT, Ring::SMEM, st>>>(gargs(b, false));
    gbase += (u32)tiles;
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

// ---------------------------------------------------------------- Dither's no-fold call (circuit/dither.py, #4974, CMP-172)
// X and W' as fill_a_kernel and fill_b_kernel form them; T_s = (X W'^T over the first min(s L, K) inner indices) mod 2^16,
// s = 1 .. S = ceil(K / L), with no wrap correction, tail or binding. A filler call (A the "pad" stream: A[i, k] byte k of
// blocks i K / 64 ..) commits T_1 .. T_S; a served call (o; A the caller's activations) commits T_1 .. T_W, W = S - 1, from
// the protocol GEMM over k < W L, and O = (A + d_A)(B + d_B)^T mod 2^32 from cuBLASLt on serve_rows_kernel's a' and b' (the
// noise never enters O, so at S = 1 a served call forms no X or W'). Per tile its W (filler: S) words rows, then (served) its
// O rows (dither.values_buffer at ROW), in a tree under the params file's domain (dither.Shape.domain). V31_COPIES copies
// (values and committer, 2 or 3): call c's commit runs on s1 beside call c + 1's compute on s0, and call c's values stay
// until call c + copies is issued. Call c salts leaf pos prf_salt(Ks, off_c + pos) (dither.salts), Ks the params file's,
// set into C.ks on s1 ahead of its salt phase, so every Fill of a process salts on one s1; the leaf phase reads no Ks, so
// V31_LSTR 1 runs it on its copy's own stream. W' (and b') is formed again only when the weight-side seeds (d_B, U, P) or B
// change: v3.2 draws them once a period, the run of record once a call. V31_FSTR 1: the coins and form run on sf (else s0)
// into the slot of X (a') the GEMM before last read, and W' (b'), when formed, into the one its last formed W' didn't take;
// sf waits for the last GEMM that read the slot (ev_xfree, ev_wfree), and the GEMM on s0 for the form (ev_formed).
static void ks_consts(const uint8_t* ks, u64* out) {   // C.ks, then C.ks_m8 (Consts' adjacent fields)
  words_of(ks, out, 8);
  u64 blk[16] = {}, st[8];
  memcpy(blk, out, 64);
  sha::iv(st);
  sha::rounds_prefix(st, blk, 8, out + 8);
}
struct Fill {
  static constexpr int NBMAX = 3;
  int m = 0, n = 0, k = 0, lp = 0, L = 0, row = 0, S = 0, NT = 0, kb64 = 0, tiles = 0, tiles_n = 0, lpt = 0, grid = 0, nsm = 0;
  int Wn = 0;                         // committed word segments: S - 1 served, S filler
  int nb = 2;                         // copies (V31_COPIES)
  u64 leaves = 0;
  Seeds seeds{};
  int8_t *A = nullptr, *B = nullptr, *U = nullptr, *V = nullptr, *X = nullptr, *W = nullptr;
  uint8_t *dA = nullptr, *dB = nullptr;
  uint16_t* perm = nullptr;
  uint16_t* words[NBMAX] = {};
  int* D[NBMAX] = {};                 // served: cuBLASLt's a' b'^T; V31_LTW at S = 1: its X W'^T in place of words
  uint8_t* vbuf = nullptr;            // testing: a copy's values buffer
  Lt lt{};
  bool ltw = false;
  bool o = false;                     // a served call (set before setup)
  int8_t *Ap = nullptr, *Bp = nullptr;          // served: a' and b' (read by the compute alone)
  // X, W, Ap and Bp are the slots the last form took: one each, or (fstr) two, xs the X (a') slot and wsl the W' (b') one
  int8_t *Xs[2] = {}, *Ws[2] = {}, *Aps[2] = {}, *Bps[2] = {};
  int xs = 0, wsl = 0;
  int *ra[NBMAX] = {}, *sa[NBMAX] = {}, *rb[NBMAX] = {}, *sb[NBMAX] = {};   // their row sums and shifts, a copy each
  int* bad = nullptr;                 // bad[copy]: a row of A + d_A spans -128 .. 128; bad[NBMAX + copy]: one of B + d_B
  u64 off[NBMAX] = {};                // copy b's call's salt offset
  bool nomem = false;                 // timing only: the leaves hashed from NoMemRow's register bytes (wrong roots)
  Committer cm[NBMAX];
  Maps* maps = nullptr;
  cudaStream_t s0 = nullptr, s1 = nullptr, so = nullptr;   // compute, commit (salts; leaves too unless lstr), openings
  cudaStream_t sl[NBMAX] = {};                              // lstr: copy b's leaf phase
  cudaStream_t sf = nullptr;                                // the coins and form: fstr's own stream, else s0
  bool own = false, fown = false;                           // s0 and s1 are this Fill's; sf is
  cudaEvent_t ev_comp[NBMAX], ev_done[NBMAX], ev_salt[NBMAX], ev_formed, ev_xfree[2], ev_wfree[2], ev_join;
  u32* gctr = nullptr;
  u32 gbase = 0;
  Lanes pl;              // the salt phases' persistent counters (and the leaf phases' unless lstr)
  Lanes pll[NBMAX];      // lstr: copy b's leaf phase's
  Knobs kn;
  u64 calls = 0;         // calls issued; call c's values are in copy c % nb
  int pend = -1;         // V31_SCHED 1: the copy whose commit waits for the next call's form
  u64 wseed[3][8] = {};  // the d_B, U and P seeds W' was formed from
  bool wformed = false;
  u64* ksw[NBMAX] = {};  // pinned: Ks as C.ks, C.ks_m8 (a copy each, for the async copy)
  u64* opos = nullptr;                // openings: positions, then rows and paths
  uint8_t* obuf = nullptr;
  int ocap = 0;

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
             cudaStream_t c1 = nullptr, cudaStream_t cf = nullptr) {
    m = m_; n = n_; k = k_; lp = lp_; L = L_; row = row_;
    const std::string why = bad_shape(m, n, k, lp, L, row);
    if (!why.empty()) { fprintf(stderr, "%s\n", why.c_str()); exit(2); }
    S = (k + L - 1) / L; NT = k / lp; kb64 = (k + 511) / 512 * 64;
    Wn = o ? S - 1 : S;
    tiles_n = n / TILE; tiles = (m / TILE) * tiles_n;
    lpt = Wn * (32768 / row) + (o ? 65536 / row : 0);
    leaves = (u64)tiles * lpt;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    grid = std::min(tiles, nsm);
    ltw = kn.ltw && S == 1 && !o;
    if (kn.ccta < 0 || kn.ccta > 8) { fprintf(stderr, "V31_CCTA: 0 to 8\n"); exit(2); }
    if (kn.copies < 2 || kn.copies > NBMAX) { fprintf(stderr, "V31_COPIES: 2 or 3\n"); exit(2); }
    nb = kn.copies;
    own = !c0;
    if (own) {
      int lo, hi;
      CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
      CK(cudaStreamCreateWithPriority(&c0, cudaStreamNonBlocking, kn.cprio == "lo" ? hi : lo));
      CK(cudaStreamCreateWithPriority(&c1, cudaStreamNonBlocking, kn.cprio == "hi" ? hi : lo));
    }
    s0 = c0; s1 = c1;
    if (kn.fprio != "hi" && kn.fprio != "lo") { fprintf(stderr, "V31_FPRIO: hi or lo\n"); exit(2); }
    sf = !kn.fstr ? s0 : cf;
    fown = kn.fstr && !cf;
    if (fown) {
      int lo, hi;
      CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
      CK(cudaStreamCreateWithPriority(&sf, cudaStreamNonBlocking, kn.fprio == "hi" ? hi : lo));
    }
    CK(cudaStreamCreateWithFlags(&so, cudaStreamNonBlocking));
    set_carveout(kn.carve);
    gctr = dalloc<u32>(1);
    CK(cudaMemset(gctr, 0, 4));
    pl.init(nsm);
    for (int b = 0; b < nb; b++) {
      if (!kn.lstr) { sl[b] = s1; continue; }
      int pr;
      CK(cudaStreamGetPriority(s1, &pr));
      CK(cudaStreamCreateWithPriority(&sl[b], cudaStreamNonBlocking, pr));
      pll[b].init(nsm);
    }
    const size_t mk = (size_t)m * k, nk = (size_t)n * k;
    A = dalloc<int8_t>(mk); B = dalloc<int8_t>(nk);
    for (int e = 0; e < 2; e++) {
      Xs[e] = e && !kn.fstr ? Xs[0] : dalloc<int8_t>(mk);
      Ws[e] = e && !kn.fstr ? Ws[0] : dalloc<int8_t>(nk);
      if (o) { Aps[e] = e && !kn.fstr ? Aps[0] : dalloc<int8_t>(mk); Bps[e] = e && !kn.fstr ? Bps[0] : dalloc<int8_t>(nk); }
    }
    X = Xs[0]; W = Ws[0]; Ap = Aps[0]; Bp = Bps[0];
    U = dalloc<int8_t>((size_t)n * lp); V = dalloc<int8_t>((size_t)m * lp);
    dA = dalloc<uint8_t>((size_t)m * kb64); dB = dalloc<uint8_t>((size_t)n * kb64);
    perm = dalloc<uint16_t>((size_t)NT * lp);
    if (o) {
      for (int b = 0; b < nb; b++) { ra[b] = dalloc<int>(m); sa[b] = dalloc<int>(m); rb[b] = dalloc<int>(n); sb[b] = dalloc<int>(n); }
      bad = dalloc<int>(2 * NBMAX);
      CK(cudaMemset(bad, 0, 8 * NBMAX));
    }
    for (int b = 0; b < nb; b++) {
      if (ltw || o) D[b] = dalloc<int>((size_t)m * n);
      if (!ltw && Wn > 0) words[b] = dalloc<uint16_t>((size_t)tiles * Wn * TILE * TILE);
      cm[b].init(params, leaves, (u32)row, false, false, 0, dom, true, 1);
      CK(cudaMallocHost(&ksw[b], 128));
      ks_consts(params.ks, ksw[b]);
    }
    maps = dalloc<Maps>(4);   // the protocol GEMM's, X slot e >> 1 and W' slot e & 1
    for (int e = 0; e < 4; e++) {
      const Maps mp{tmap(Xs[e >> 1], k, m, TILE), tmap(Ws[e & 1], k, n, TILE), tmap(Xs[e >> 1], k, m, TILE),
                    tmap(Ws[e & 1], k, n, TILE)};
      CK(cudaMemcpy(maps + e, &mp, sizeof mp, cudaMemcpyHostToDevice));
    }
    CK(cudaFuncSetAttribute(gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Ring::SMEM));
    if (ltw) lt_setup(lt, m, n, k, X, W, D[0], s0);
    if (o) lt_setup(lt, m, n, k, Ap, Bp, D[0], s0);
    for (int b = 0; b < nb; b++) {
      CK(cudaEventCreateWithFlags(&ev_comp[b], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&ev_done[b], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&ev_salt[b], cudaEventDisableTiming));
      CK(cudaEventRecord(ev_done[b], s1));
    }
    for (cudaEvent_t* e : {&ev_formed, &ev_xfree[0], &ev_xfree[1], &ev_wfree[0], &ev_wfree[1], &ev_join})
      CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
  }
  void release() {
    CK(cudaDeviceSynchronize());
    for (void* p : {(void*)A, (void*)B, (void*)U, (void*)V, (void*)dA, (void*)dB, (void*)perm, (void*)vbuf, (void*)maps,
                    (void*)gctr, (void*)pl.ctl, (void*)pl.claim, (void*)opos, (void*)obuf, (void*)bad})
      cudaFree(p);
    for (int8_t** s : {Xs, Ws, Aps, Bps}) {
      cudaFree(s[0]);
      if (s[1] != s[0]) cudaFree(s[1]);
    }
    if (ltw || o) lt_free(lt);
    for (int b = 0; b < nb; b++) {
      for (void* p : {(void*)words[b], (void*)D[b], (void*)ra[b], (void*)sa[b], (void*)rb[b], (void*)sb[b], (void*)pll[b].ctl,
                      (void*)pll[b].claim})
        cudaFree(p);
      cm[b].release();
      cudaFreeHost(ksw[b]);
      cudaEventDestroy(ev_comp[b]);
      cudaEventDestroy(ev_done[b]);
      cudaEventDestroy(ev_salt[b]);
      if (kn.lstr) cudaStreamDestroy(sl[b]);
    }
    for (cudaEvent_t e : {ev_formed, ev_xfree[0], ev_xfree[1], ev_wfree[0], ev_wfree[1], ev_join}) cudaEventDestroy(e);
    cudaStreamDestroy(so);
    if (fown) cudaStreamDestroy(sf);
    if (own) { cudaStreamDestroy(s0); cudaStreamDestroy(s1); }
  }
  // fstr: sf waits for st's work so far (a timed arm's start, or the caller's stream before a call)
  void join(cudaStream_t st) {
    if (!kn.fstr || st == sf) return;
    CK(cudaEventRecord(ev_join, st));
    CK(cudaStreamWaitEvent(sf, ev_join));
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
  // copy b's operands. Served: a' (and, when wside, b') with their sums and shifts, else copy b takes the previous call's
  // copy's B side (the weights and their coins unchanged since); then X (and W') when a words segment is committed
  void form(cudaStream_t st, const int8_t* a, bool wside, int b) {
    if (o) {
      CK(cudaMemsetAsync(bad + b, 0, 4, st));
      serve_rows_kernel<<<m, 256, 0, st>>>(a, dA, k, kb64, 0, Ap, ra[b], sa[b], bad + b);
      if (wside) {
        CK(cudaMemsetAsync(bad + NBMAX + b, 0, 4, st));
        serve_rows_kernel<<<n, 256, 0, st>>>(B, dB, k, kb64, 1, Bp, rb[b], sb[b], bad + NBMAX + b);
      } else {
        const int p = (b + nb - 1) % nb;
        CK(cudaMemcpyAsync(rb[b], rb[p], 4 * (size_t)n, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(sb[b], sb[p], 4 * (size_t)n, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(bad + NBMAX + b, bad + NBMAX + p, 4, cudaMemcpyDeviceToDevice, st));
      }
    }
    if (!Wn) return;
    const u64 ca = (u64)m * k / 16, cb = (u64)n * k / 16;
    fill_a_kernel<<<(unsigned)((ca + 255) / 256), 256, 0, st>>>(a, dA, V, X, ca, k, lp, kb64);
    if (wside) fill_b_kernel<<<(unsigned)((cb + 255) / 256), 256, 0, st>>>(B, dB, U, perm, W, cb, k, lp, kb64);
  }
  // launches that share gctr run in one stream's order, so each one's base is the tiles launched before it. Served: D = a'
  // b'^T on cuBLASLt, then T_1 .. T_W from the protocol GEMM over k < W L (its K only counts its k-tiles)
  void gemm(cudaStream_t st, int b) {
    if (ltw) { lt_run(lt, X, W, D[b], st); return; }
    if (o) lt_run(lt, Ap, Bp, D[b], st);
    if (!Wn) return;
    gemm_kernel<true><<<grid, Ring::NCT, Ring::SMEM, st>>>(GArgs{maps + 2 * xs + wsl, words[b], nullptr, nullptr, m, n,
                                                                 o ? Wn * L : k, L, Wn, 0, 8, lp, 0, kn.dyn ? gctr : nullptr, gbase, 0});
    gbase += (u32)tiles;
  }
  V31Row rowsrc(int b) const {
    const uint8_t* w = reinterpret_cast<const uint8_t*>(words[b]);
    return V31Row{w, w, (u64)lpt, (u64)lpt, (u64)n, (u64)tiles_n, (u32)row, (u32)Wn, (u32)kn.pf};
  }
  LtRow ltrow(int b) const { return LtRow{D[b], (u64)lpt, (u64)n, (u64)tiles_n, (u32)row, (u32)kn.pf}; }
  ServedRow servedrow(int b) const {
    return ServedRow{reinterpret_cast<const uint8_t*>(words[b]), D[b], ra[b], sa[b], rb[b], sb[b], (u64)lpt,
                     (u64)Wn * (32768 / row), (u64)n, (u64)tiles_n, (u32)row, (u32)Wn, (u32)kn.pf};
  }
  void leaves_nodes(int b, cudaStream_t st, Lanes& ln) {
    if (nomem) ln.leaves_nodes(cm[b], NoMemRow{(u32)row}, kn.ccta, st);
    else if (ltw) ln.leaves_nodes(cm[b], ltrow(b), kn.ccta, st);
    else if (o) ln.leaves_nodes(cm[b], servedrow(b), kn.ccta, st);
    else ln.leaves_nodes(cm[b], rowsrc(b), kn.ccta, st);
  }
  void leaves_nodes(int b, cudaStream_t st) { leaves_nodes(b, st, pl); }
  // copy b's values poisoned (testing)
  void poison(int b, cudaStream_t st) {
    if (D[b]) gk::poison(D[b], (size_t)m * n * 4, st);
    if (words[b]) gk::poison(words[b], (size_t)tiles * Wn * 32768, st);
  }
  // copy b's committed values (testing): leaves x ROW bytes, leaf pos's row at pos ROW (dither.values_buffer); words[b]
  // itself, or the rows gathered into vbuf
  const uint8_t* values_of(int b);
  void set_ks(int b, cudaStream_t st) {
    CK(cudaMemcpyToSymbolAsync(C, ksw[b], 128, offsetof(Consts, ks), cudaMemcpyHostToDevice, st));
  }
  // copy b's commit: Ks and its call's salts on s1, then (after its compute) its leaves and levels on s1, or (lstr) on
  // sl[b] after the salts
  void commit_ovl(int b) {
    set_ks(b, s1);
    pl.salts(cm[b], kn.ccta, s1, off[b]);
    if (kn.lstr) {
      CK(cudaEventRecord(ev_salt[b], s1));
      CK(cudaStreamWaitEvent(sl[b], ev_salt[b]));
    }
    CK(cudaStreamWaitEvent(sl[b], ev_comp[b]));
    leaves_nodes(b, sl[b], kn.lstr ? pll[b] : pl);
    CK(cudaEventRecord(ev_done[b], sl[b]));
  }
  void flush() {
    if (pend >= 0) commit_ovl(pend);
    pend = -1;
  }
  // One call on the seeds set, its leaves salted from off (dither.salts(Ks, off): off_c = c leaves for the c-th call of a
  // tree's sequence), A at a (a device pointer to m x K int8, ready on the caller's side; null: the pad, a filler's only).
  // Waits on the host for call c - nb's commit, whose copy this call takes. wside: form W' (b') again even if its seeds are
  // unchanged (the per-call weight coins). Returns the copy.
  int issue(u64 off_, const int8_t* a, const Cheat& ch = Cheat{}, bool wside = false) {
    if (o && !a) { fprintf(stderr, "a served call takes the caller's A\n"); exit(2); }
    if ((ch.kind == 2 || ch.kind == 3) && (ch.kind == 3 ? 0 : ch.s0) > Wn) { fprintf(stderr, "forge: S0 above W\n"); exit(2); }
    if ((ch.kind == 3 || (ch.kind == 2 && !ch.s0)) && !D[0]) { fprintf(stderr, "forge of O: a served call's\n"); exit(2); }
    const int b = (int)(calls % nb);
    if (pend == b) flush();
    CK(cudaEventSynchronize(ev_done[b]));
    off[b] = off_;
    const bool ws = wside || weights_stale();
    if (kn.fstr) {
      xs ^= 1;
      if (ws) wsl ^= 1;
      X = Xs[xs]; W = Ws[wsl]; Ap = Aps[xs]; Bp = Bps[wsl];
      CK(cudaStreamWaitEvent(sf, ev_xfree[xs]));
      if (ws) CK(cudaStreamWaitEvent(sf, ev_wfree[wsl]));
    }
    coins(sf, !a, ws);
    if (ch.kind == 4) {   // nodither: d_A and d_B zero
      CK(cudaMemsetAsync(dA, 0, (size_t)m * kb64, sf));
      if (ws) CK(cudaMemsetAsync(dB, 0, (size_t)n * kb64, sf));
    }
    form(sf, a ? a : A, ws, b);
    if (ws) {
      memcpy(wseed[0], seeds.w[S_DB], 64); memcpy(wseed[1], seeds.w[S_U], 64); memcpy(wseed[2], seeds.w[S_P], 64);
      wformed = ch.kind != 4;
    }
    if (ch.kind == 1 && Wn) CK(cudaMemset2DAsync(X + (size_t)ch.tau * lp, k, 0, lp, m, sf));
    if (kn.sched == 1 || kn.fstr) CK(cudaEventRecord(ev_formed, sf));
    if (kn.fstr) CK(cudaStreamWaitEvent(s0, ev_formed));
    if (kn.sched == 1 && pend >= 0) {
      CK(cudaStreamWaitEvent(s1, ev_formed));
      commit_ovl(pend);
      pend = -1;
    }
    gemm(s0, b);
    if (kn.fstr) {
      CK(cudaEventRecord(ev_xfree[xs], s0));
      CK(cudaEventRecord(ev_wfree[wsl], s0));
    }
    if (ch.kind == 2 || ch.kind == 3) {
      const int stride = std::max(1, (int)std::lround(1.0 / ch.frac));
      const int s0f = ch.kind == 3 ? 0 : ch.s0;
      if (ltw || (o && s0f == 0))   // forge_kernel's O path into D: V31_LTW's T_1 (its low halves), or O
        forge_kernel<<<(tiles + stride - 1) / stride, 256, 0, s0>>>(nullptr, D[b], Wn, 0, stride, tiles_n, n, 0xF0F0u);
      else
        forge_kernel<<<(tiles + stride - 1) / stride, 256, 0, s0>>>(words[b], nullptr, Wn, s0f, stride, tiles_n, n, 0xF0F0u);
    }
    CK(cudaEventRecord(ev_comp[b], s0));
    if (kn.sched == 1) pend = b;
    else commit_ovl(b);
    calls++;
    return b;
  }
  // call c's copy once its commit is done, or -1 when c isn't one of the last nb calls
  int ready(u64 c) {
    if (c >= calls || c + nb < calls) return -1;
    const int b = (int)(c % nb);
    if (pend == b) flush();
    CK(cudaEventSynchronize(ev_done[b]));
    return b;
  }
  size_t levels_count() const { return 2 * cm[0].width - 1; }
  // positions' rows (row bytes each) and paths (depth digests each, leaf level first) of copy b, gathered on the device
  void open(int b, const u64* pos, int np, uint8_t* rows, uint8_t* paths);
};
// a leaf's row bytes: from the words buffer, or (V31_LTW) from cuBLASLt's D
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
static void kernel_attrs(FILE* f) {
  const KAttr ks[] = {{"gemm_proto", (const void*)gemm_kernel<true>}, {"gemm_plain", (const void*)gemm_kernel<false>},
                      {"form_a", (const void*)form_a_kernel}, {"form_b", (const void*)form_b_kernel<false>},
                      {"form_b5", (const void*)form_b_kernel<true>},
                      {"wrap", (const void*)wrap_kernel}, {"wrap2_128", (const void*)wrap2_kernel<128>},
                      {"wrap2_256", (const void*)wrap2_kernel<256>}, {"wrap3_128", (const void*)wrap3_kernel<128>},
                      {"wrap3_256", (const void*)wrap3_kernel<256>}, {"wrap4_128", (const void*)wrap4_kernel<128>},
                      {"wrap4_256", (const void*)wrap4_kernel<256>}, {"wrap_a", (const void*)wrap_a_kernel},
                      {"bits", (const void*)bits_kernel},
                      {"leaf_small", (const void*)leaf_small<V31Row, 0>}, {"salt_small", (const void*)salt_small<true>},
                      {"leaf_persist", (const void*)leaf_persist<V31Row>}, {"salt_persist", (const void*)salt_persist},
                      {"leaf_persist_nomem", (const void*)leaf_persist<NoMemRow>}};
  fprintf(f, "{");
  for (size_t i = 0; i < sizeof ks / sizeof ks[0]; i++) {
    cudaFuncAttributes a;
    CK(cudaFuncGetAttributes(&a, ks[i].fn));
    fprintf(f, "%s\"%s\": {\"regs\": %d, \"local\": %zu, \"smem_static\": %zu}", i ? ", " : "", ks[i].name, a.numRegs,
            a.localSizeBytes, a.sharedSizeBytes);
  }
  fprintf(f, "}");
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
// words, O and leaves
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
           "\"sched\": %d, \"arms\": {",
           name, m, n, k, lp, t, row, c.S, c.dz, c.dy, (unsigned long long)c.leaves, prop.name, c.nsm, c.grid, c.nb,
           (double)wa / ((double)m * k), (double)wbn / ((double)n * k), lt.algo_id, lt.tile, lt.stages, lt.splitk, lt.ncand,
           cut.name.c_str(), cut.ncand, tm.steady ? "true" : "false", c.kn.carve, c.kn.dyn, c.kn.cprio.c_str(), c.kn.wrap, c.wcw,
           c.kn.ccta, c.kn.pf, V31_MAXNREG, c.nb == 2 ? c.kn.sched : 0);
    for (size_t a = 0; a < arms.size(); a++)
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f}", a ? ", " : "", arms[a].name.c_str(), med[a], mhz[a]);
    printf("}, \"gamma_call\": %.4f, \"gamma_ovl\": %.4f, \"gamma_gemm\": %.4f, \"plain_over_lt\": %.4f, ", med[7] / med[0],
           med[8] / med[0], med[5] / med[0], med[1] / med[0]);
    if (cut.best)   // arm 12: over CUTLASS's fastest, and the two baselines' ratio
      printf("\"gamma_call_cutlass\": %.4f, \"gamma_ovl_cutlass\": %.4f, \"gamma_gemm_cutlass\": %.4f, \"cutlass_over_lt\": %.4f, ",
             med[7] / med[12], med[8] / med[12], med[5] / med[12], med[12] / med[0]);
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
  for (int i = 0; i < novl; i++) c.call_ovl(c.s0);
  c.ovl_drain(c.s0);
  CK(cudaStreamSynchronize(c.s0));
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
// served calls (A gen_kernel's activations). A shape is NAME:M:N:K, or NAME:M:N:K:LP[:L] at an L' (and L) of its own.
static int cmd_fill_once(const std::string& pdir, int lp0, int Lspec, int row, const std::string& sh) {
  char name[64];
  int m, n, k, lp = lp0, Ls = Lspec;
  const int got = sscanf(sh.c_str(), "%63[^:]:%d:%d:%d:%d:%d", name, &m, &n, &k, &lp, &Ls);
  if (got < 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
  const int L = Ls > 0 ? Ls : k, novl = env_int("V31_ONCE_OVL", 6);
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
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  printf("%s: %llu calls (2 serial, %d overlapped, weight coins a call), ltw %d, served %d\n", name, (unsigned long long)f.calls,
         novl, f.ltw ? 1 : 0, f.o ? 1 : 0);
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
// (NoMemRow: the commit's compressions without its loads). Call c salts from c leaves. One JSON line a shape (NAME:M:N:K, or
// NAME:M:N:K:LP[:L] at an L' (and L) of its own).
static int cmd_fill_bench(int reps, const std::string& out, const std::string& pdir, int lp0, int Lspec, int row,
                          const std::vector<std::string>& shapes) {
  gk::Nvml nv(0);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::mt19937_64 rng(std::random_device{}());
  const bool served = env_int("V31_SERVED", 0) != 0;
  for (const auto& sh : shapes) {
    char name[64];
    int m, n, k, lp = lp0, Ls = Lspec;
    const int got = sscanf(sh.c_str(), "%63[^:]:%d:%d:%d:%d:%d", name, &m, &n, &k, &lp, &Ls);
    if (got < 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    const int L = Ls > 0 ? Ls : k;
    const std::string pfile = pdir + "/" + fill_params_name(m, n, k, lp, L, row, served);
    Fill f;
    f.o = served;
    f.setup(m, n, k, lp, L, row, read_params(pfile), 0);
    int8_t* A = dalloc<int8_t>((size_t)m * k);
    int *D = dalloc<int>((size_t)m * n), *D2 = dalloc<int>((size_t)m * n);
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
    // back-to-back calls on s0 (their coins and form after s's work so far), then s0 waits for every commit
    auto ovl = [&f](cudaStream_t s, int r, const int8_t* a, bool wside) {
      f.join(s);
      for (int i = 0; i < r; i++) f.issue(f.calls * f.leaves, a, Cheat{}, wside);
      f.flush();
      for (int b = 0; b < f.nb; b++) CK(cudaStreamWaitEvent(s, f.ev_done[b]));
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
             if (f.o) serve_rows_kernel<<<n, 256, 0, s>>>(f.B, f.dB, k, f.kb64, 1, f.Bp, f.rb[0], f.sb[0], f.bad + Fill::NBMAX);
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
        {"ovl", [&](cudaStream_t s, int r) { ovl(s, r, A, false); }},
        {"ovl_w", [&](cudaStream_t s, int r) { ovl(s, r, A, true); }},
        {"ovl_nomem", [&](cudaStream_t s, int r) { f.nomem = true; ovl(s, r, A, false); f.nomem = false; }},
    };
    if (!served) {
      arms.push_back({"pad", [&](cudaStream_t s, int r) {
                        const u64 bp = (u64)m * k / 64;
                        for (int i = 0; i < r; i++)
                          bits_kernel<<<(unsigned)((bp + 127) / 128), 128, 0, s>>>(f.seeds, S_PAD, bp, reinterpret_cast<uint8_t*>(f.A));
                      }});
      arms.push_back({"ovl_pad", [&](cudaStream_t s, int r) { ovl(s, r, nullptr, true); }});
    }
    const int ICUT = (int)arms.size();
    if (cut.best) arms.push_back({"cutlass", [&](cudaStream_t s, int r) { for (int i = 0; i < r; i++) cut_run(cut, s); }});
    const gk::Timing tm = gk::timed_items(arms, reps, f.s0, nv);
    CK(cudaGetLastError());
    gk::summary(tm, arms, stderr);
    // the timed schedule's own values: nb overlapped calls into poisoned buffers, the last one's tiles to NAME-ovl
    for (int b = 0; b < f.nb; b++) {
      f.poison(b, f.s0);
      gk::poison(f.cm[b].levels, f.levels_count() * 64, f.s0);
    }
    CK(cudaStreamSynchronize(f.s0));
    int bl = -1;
    for (int i = 0; i < f.nb; i++) bl = f.issue(f.calls * f.leaves, A);
    for (int i = f.nb; i >= 1; i--) f.ready(f.calls - i);
    CK(cudaGetLastError());
    fill_samples(f, bl, out + "/samples/" + name + "-ovl", pfile, sd, A, (u32)rng());
    std::vector<double> med(arms.size()), mhz(arms.size());
    for (size_t a = 0; a < arms.size(); a++) {
      std::vector<double> v, fq;
      for (const auto& it : tm.items)
        if (it.arm == (int)a) { v.push_back(it.ms / it.calls); if (it.sm_mhz()) fq.push_back(it.sm_mhz()); }
      med[a] = gk::median(v);
      mhz[a] = gk::median(fq);
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
           "\"carve\": %d, \"dyn\": %d, \"cprio\": \"%s\", \"ccta\": %d, \"pf\": %d, \"maxnreg\": %d, \"sched\": %d, \"ltw\": %d, "
           "\"copies\": %d, \"lstr\": %d, \"fstr\": %d, \"fprio\": \"%s\", \"sha_rot\": %d, \"arms\": {",
           name, m, n, k, lp, L, row, f.S, f.Wn, f.o ? "true" : "false", (unsigned long long)f.leaves, prop.name, f.nsm, f.grid,
           lt.algo_id, lt.tile, lt.stages, lt.splitk, lt.ncand, cut.name.c_str(), cut.ncand, tm.steady ? "true" : "false",
           f.kn.carve, f.kn.dyn, f.kn.cprio.c_str(), f.kn.ccta, f.kn.pf, V31_MAXNREG, f.kn.sched, f.ltw ? 1 : 0, f.kn.copies,
           f.kn.lstr, f.kn.fstr, f.kn.fprio.c_str(), SHA_ROT);
    for (size_t a = 0; a < arms.size(); a++)
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f}", a ? ", " : "", arms[a].name.c_str(), med[a], mhz[a]);
    const double l = at("lt");
    printf("}, \"gamma_ovl\": %s, \"gamma_ovl_w\": %s, \"gamma_ovl_pad\": %s, \"gamma_ovl_nomem\": %s, \"gamma_call\": %s, "
           "\"gamma_gemm\": %s, ",
           jn(at("ovl") / l).c_str(), jn(at("ovl_w") / l).c_str(), jn(at("ovl_pad") / l).c_str(), jn(at("ovl_nomem") / l).c_str(),
           jn(at("call") / l).c_str(), jn(at("gemm") / l).c_str());
    if (cut.best)
      printf("\"gamma_ovl_cutlass\": %s, \"gamma_ovl_w_cutlass\": %s, \"gamma_ovl_pad_cutlass\": %s, "
             "\"gamma_gemm_cutlass\": %s, \"cutlass_over_lt\": %s, ",
             jn(at("ovl") / med[ICUT]).c_str(), jn(at("ovl_w") / med[ICUT]).c_str(), jn(at("ovl_pad") / med[ICUT]).c_str(),
             jn(at("gemm") / med[ICUT]).c_str(), jn(med[ICUT] / l).c_str());
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

#ifdef V31_LIB
// ---------------------------------------------------------------- the per-call binding (v31lib.py over ctypes)
// One process, one device: every context shares its compute and commit streams (calls run in issue order, each commit beside
// the next call's compute; V31_LSTR 1 moves each copy's leaf phase to a stream of its own, the salts staying on the shared
// one, which alone reads C.ks; V31_FSTR 1 runs every context's coins and form on one more shared stream, beside the GEMM
// before), a domain slot each (COMMIT_GPU_DOMS of them) and one ROW (commit_gpu's C fixes it). Every
// function returns 0, or -1 with the reason in v31_error() (the calling thread's); a CUDA failure exits the process
// (commit_gpu's CK). One thread drives every context: the thread that made the first live one, and any other is refused,
// since the salt phases read the Ks the last issued call set into C on the shared commit stream.
#include <thread>
static thread_local std::string g_err;
static int g_dev = -1, g_row = 0, g_live = 0;
static std::thread::id g_thread;
static bool g_dom[COMMIT_GPU_DOMS] = {};
static cudaStream_t g_s0 = nullptr, g_s1 = nullptr, g_sf = nullptr;
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
static bool foreign() { return g_live && std::this_thread::get_id() != g_thread; }
static int refuse_thread() { return fail("one thread drives every context: this isn't the thread that made the live ones"); }
#define V31_OWN_THREAD() \
  if (foreign()) return refuse_thread()

extern "C" {
const char* v31_error() { return g_err.c_str(); }

// A context for calls of m x n x k at noise period lp (L'), segments of L and leaf rows of row bytes, served calls (O
// committed, T_S implied) or a filler's, committed under the CGP1 params file's domain and salted under its Ks
// (v31lib.Prover writes it), on CUDA device dev. Null on failure.
void* v31_create(int m, int n, int k, int lp, int L, int row, int served, const char* params_path, int dev) {
  if (foreign()) { refuse_thread(); return nullptr; }
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
  if (!g_sf && env_int("V31_FSTR", 0)) {
    int lo, hi;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CK(cudaStreamCreateWithPriority(&g_sf, cudaStreamNonBlocking, env_str("V31_FPRIO", "hi") == "lo" ? lo : hi));
  }
  CK(cudaDeviceSynchronize());   // Committer::init rewrites C, which a running salt phase reads
  LibCtx* c = new LibCtx;
  c->f.o = served != 0;
  c->f.setup(m, n, k, lp, L, row, read_params(params_path), dom, g_s0, g_s1, g_sf);
  c->dom = dom;
  c->own_b = c->f.B;
  g_dom[dom] = true;
  g_dev = dev;
  g_row = row;
  if (!g_live++) g_thread = std::this_thread::get_id();
  return c;
}

// m n k lp L row S tiles lpt leaves depth calls W o, as u64 (W the committed word segments: S - 1 served, S filler)
int v31_info(void* h, uint64_t* out) {
  V31_OWN_THREAD();
  const Fill& f = ctx_of(h)->f;
  const uint64_t v[14] = {(u64)f.m, (u64)f.n, (u64)f.k, (u64)f.lp, (u64)f.L, (u64)f.row, (u64)f.S, (u64)f.tiles, (u64)f.lpt,
                          f.leaves, (u64)f.cm[0].depth, f.calls, (u64)f.Wn, (u64)f.o};
  memcpy(out, v, sizeof v);
  return 0;
}
// The context's copies (V31_COPIES at v31_create): call c's root, openings and values stay until call c + copies is issued
int v31_copies(void* h) {
  V31_OWN_THREAD();
  return ctx_of(h)->f.nb;
}

// The weights (n x k int8, row-major) from host memory, or from device memory when on_device, into the context's own
// buffer (after every issued call's form), which the calls then read
int v31_set_b(void* h, const void* B, int on_device) {
  V31_OWN_THREAD();
  LibCtx* c = ctx_of(h);
  Fill& f = c->f;
  CK(cudaStreamSynchronize(f.s0));
  CK(cudaStreamSynchronize(f.sf));
  f.B = c->own_b;
  CK(cudaMemcpy(f.B, B, (size_t)f.n * f.k, on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
  f.wformed = false;
  return 0;
}

// The weights by device pointer (n x k int8, row-major), no copy and no wait: the calls issued after read B where it is
// (the next one forms W' or b' again), so it stays unchanged until they have formed (v31_sync). Null: the context's own
// buffer again, as v31_set_b last wrote it.
int v31_point_b(void* h, const void* B) {
  V31_OWN_THREAD();
  LibCtx* c = ctx_of(h);
  c->f.B = B ? (int8_t*)B : c->own_b;
  c->f.wformed = false;
  return 0;
}

// A (m x k int8, row-major, from host memory or device memory when on_device) into this context's own operand buffer, in
// the form's stream order (after every issued call's form); *dev gets the buffer's device pointer, for v31_prove's
// a_dev. The pad (a null a_dev) overwrites the same buffer.
int v31_load_a(void* h, const void* A, int on_device, void** dev) {
  V31_OWN_THREAD();
  Fill& f = ctx_of(h)->f;
  CK(cudaMemcpyAsync(f.A, A, (size_t)f.m * f.k, on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, f.sf));
  CK(cudaStreamSynchronize(f.sf));
  *dev = f.A;
  return 0;
}

// One call: seeds (6 x 64 bytes, dither.STREAMS' order: pad, dA, dB, U, V, P; a served call's pad unused), its salt offset
// off (its leaf pos salted prf_salt(Ks, off + pos): off_c = c leaves for the c-th call under one Ks), and A (a device
// pointer to m x k int8, ready on the caller's side; null: the pad, a filler's only). Returns at once with the call's index
// in *call; the compute and commit run on the device. Waits first for call c - copies' commit, whose copy this call takes
// (its values can't be opened after). cheat (testing): "none", "skip:TAU" (noise tile TAU of X zeroed), "forge:S0[:FRAC]" (T_S0
// of every 1/FRAC-th tile overwritten; served, S0 0: O), "forgeO[:FRAC]" (forge:0) or "nodither" (d_A and d_B zero).
int v31_prove(void* h, const uint8_t* seeds, uint64_t off, const void* a_dev, const char* cheat, uint64_t* call) {
  V31_OWN_THREAD();
  Fill& f = ctx_of(h)->f;
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
  return 0;
}

// A served call's operands out of what O's decomposition holds: a row of A + d_A or B + d_B with both -128 and 128 (O's
// int8 a' b'^T can't carry it); the call has no root and no openings
static int refused(const Fill& f, int b, uint64_t c) {
  if (!f.o) return 0;
  int bd[2 * Fill::NBMAX];
  CK(cudaMemcpy(bd, f.bad, sizeof bd, cudaMemcpyDeviceToHost));
  if (bd[b]) return fail("call " + std::to_string(c) + ": a row of A + d_A spans -128 .. 128 (served O needs each row in 255 values)");
  if (bd[Fill::NBMAX + b])
    return fail("call " + std::to_string(c) + ": a row of B + d_B spans -128 .. 128 (served O needs each row in 255 values)");
  return 0;
}
static int stale(const Fill& f, uint64_t c) {
  return fail("call " + std::to_string(c) + " isn't one of the last " + std::to_string(f.nb) + " calls");
}

// Call c's root (64 bytes), once its commit is done
int v31_root(void* h, uint64_t c, uint8_t* out) {
  V31_OWN_THREAD();
  Fill& f = ctx_of(h)->f;
  const int b = f.ready(c);
  if (b < 0) return stale(f, c);
  if (refused(f, b, c)) return -1;
  std::vector<u64> r(8);
  CK(cudaMemcpy(r.data(), f.cm[b].levels + 8 * (f.levels_count() - 1), 64, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 8; i++) sha::put_be64(out + 8 * i, r[i]);
  return 0;
}

// Call c's leaves at pos[0 .. np): their rows (np x row bytes) and paths (np x depth digests, the leaf's sibling first)
int v31_open(void* h, uint64_t c, const uint64_t* pos, int np, uint8_t* rows, uint8_t* paths) {
  V31_OWN_THREAD();
  Fill& f = ctx_of(h)->f;
  const int b = f.ready(c);
  if (b < 0) return stale(f, c);
  if (refused(f, b, c)) return -1;
  if (np <= 0) return 0;
  for (int i = 0; i < np; i++)
    if (pos[i] >= f.leaves) return fail("leaf " + std::to_string(pos[i]) + " of " + std::to_string(f.leaves));
  f.open(b, pos, np, rows, paths);
  return 0;
}

// Testing: call c's committed values (leaves x row bytes, dither.values_buffer: per tile its W words segments, then its O
// rows when served), every tree level (leaves first, (2 width - 1) x 64 bytes), or the A operand its last call formed from
// (the pad, m x k bytes)
int v31_values(void* h, uint64_t c, uint8_t* out) {
  V31_OWN_THREAD();
  Fill& f = ctx_of(h)->f;
  const int b = f.ready(c);
  if (b < 0) return stale(f, c);
  CK(cudaMemcpy(out, f.values_of(b), (size_t)f.leaves * f.row, cudaMemcpyDeviceToHost));
  return 0;
}
int v31_levels(void* h, uint64_t c, uint8_t* out) {
  V31_OWN_THREAD();
  Fill& f = ctx_of(h)->f;
  const int b = f.ready(c);
  if (b < 0) return stale(f, c);
  std::vector<u64> lv(8 * f.levels_count());
  CK(cudaMemcpy(lv.data(), f.cm[b].levels, lv.size() * 8, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < lv.size(); i++) sha::put_be64(out + 8 * i, lv[i]);
  return 0;
}
// Call c's tree levels as v31_levels orders them, each node 8 native u64 words (not big-endian bytes), copied to dst
// (page-locked host memory, (2 width - 1) x 64 bytes) behind its commit without waiting: dst is whole once v31_sync
// returns, or once call c + copies is issued (which waits for it). A served call refused at its root (a row spanning -128
// .. 128) still copies; v31_root says so.
int v31_levels_async(void* h, uint64_t c, void* dst) {
  V31_OWN_THREAD();
  Fill& f = ctx_of(h)->f;
  if (c >= f.calls || c + f.nb < f.calls) return stale(f, c);
  const int b = (int)(c % f.nb);
  if (f.pend == b) f.flush();
  CK(cudaMemcpyAsync(dst, f.cm[b].levels, f.levels_count() * 64, cudaMemcpyDeviceToHost, f.sl[b]));
  CK(cudaEventRecord(f.ev_done[b], f.sl[b]));
  return 0;
}
int v31_pad(void* h, uint8_t* out) {
  V31_OWN_THREAD();
  Fill& f = ctx_of(h)->f;
  CK(cudaStreamSynchronize(f.sf));
  CK(cudaMemcpy(out, f.A, (size_t)f.m * f.k, cudaMemcpyDeviceToHost));
  return 0;
}

// Every issued call's compute and commit done
int v31_sync(void* h) {
  V31_OWN_THREAD();
  Fill& f = ctx_of(h)->f;
  f.flush();
  for (cudaStream_t s : {f.sf, f.s0, f.s1}) CK(cudaStreamSynchronize(s));
  for (int b = 0; b < f.nb; b++) CK(cudaStreamSynchronize(f.sl[b]));
  return 0;
}

// From another thread: nothing (v31_error says why)
void v31_destroy(void* h) {
  LibCtx* c = ctx_of(h);
  if (!c) return;
  if (foreign()) { refuse_thread(); return; }
  c->f.flush();
  c->f.B = c->own_b;
  c->f.release();
  g_dom[c->dom] = false;
  if (--g_live == 0) g_dev = -1;
  delete c;
}
}   // extern "C"
#else
// ---------------------------------------------------------------- contention probe
// What the commit's hashing costs cuBLASLt's GEMM when the two share the SMs: a filler kernel of cpb 128-thread CTAs an SM
// (the leaf kernels' shape and register cap) issuing one instruction mix beside the GEMM. sha: SHA-512 compressions on
// registers (the leaf hash without its loads); alu: funnel shifts, LOP3 and adds; fma: FP32 FMA; sleep: resident, issuing
// almost nothing.
template <int MODE>
__global__ void __launch_bounds__(SMALL_THREADS, SMALL_MINB) filler_kernel(u64 iters, u64* sink) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 acc = 0;
  if constexpr (MODE == 0) {
    u64 st[8], w[16];
#pragma unroll
    for (int i = 0; i < 8; i++) st[i] = t * 0x9e3779b97f4a7c15ull + i;
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = t ^ (0x5851f42d4c957f2dull * (i + 1));
    for (u64 r = 0; r < iters; r++) {
      sha::compress(st, w);
      w[0] ^= st[0];
    }
    acc = st[0] ^ st[7];
  } else if constexpr (MODE == 1) {
    u32 x[8];
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = (u32)t * (2 * i + 1) + i;
    for (u64 r = 0; r < iters; r++)
#pragma unroll
      for (int j = 0; j < 16; j++)
#pragma unroll
        for (int i = 0; i < 8; i++)
          x[i] = (__funnelshift_r(x[i], x[(i + 1) & 7], 13 + i) ^ (x[(i + 2) & 7] & x[(i + 3) & 7])) + x[(i + 5) & 7];
    acc = x[0] ^ x[3] ^ x[7];
  } else if constexpr (MODE == 2) {
    float f[8];
#pragma unroll
    for (int i = 0; i < 8; i++) f[i] = (float)(t & 1023) * 1e-3f + (float)i;
    for (u64 r = 0; r < iters; r++)
#pragma unroll
      for (int j = 0; j < 16; j++)
#pragma unroll
        for (int i = 0; i < 8; i++) f[i] = fmaf(f[i], 0.999f, f[(i + 1) & 7]);
    acc = __float_as_uint(f[0] + f[3] + f[7]);
  } else {
    for (u64 r = 0; r < iters; r++) __nanosleep(1000);
  }
  if (acc == 0x5eed5eedull) sink[0] = acc;
}

// v31 contend REPS name:M:N:K...: per shape, cuBLASLt's GEMM (cuBLASLt's pick for the served call's D, autotuned as
// lt_setup does) beside each filler mix at cpb 1, 2 and 3. g0, f0: the GEMM's and the filler's times alone; g1: the
// GEMM's beside a filler that outlasts it; f1: the filler's beside GEMMs that outlast it (the filler on the commit's
// high-priority stream, launched second). s_g = g1 / g0, s_f = f1 / f0, share = 1 / s_g + 1 / s_f: 1 when the two
// time-share the SM, 2 when each runs as if alone. Medians over REPS; valid is false when a run's outlasting failed.
// V31_CARVE (default 100, as Knobs) is the fillers' preferred shared-memory carveout, < 0 the driver's choice: an SM
// runs one carveout at a time, so a filler under a smaller one than the GEMM's 100 KiB keeps the GEMM's CTAs off it.
// V31_CGEMMS (default lt): the GEMMs, each in turn: lt (cuBLASLt's pick) or a CUTLASS entry's name (cutlass_2x.cu's,
// under -DV31_CUTLASS); V31_CMODES (default "sha alu fma sleep"): the fillers.
static int cmd_contend(int reps, const std::vector<std::string>& shapes) {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  const int nsm = prop.multiProcessorCount;
  int plo, phi;
  CK(cudaDeviceGetStreamPriorityRange(&plo, &phi));
  cudaStream_t sg, sf;
  CK(cudaStreamCreateWithPriority(&sg, cudaStreamNonBlocking, plo));
  CK(cudaStreamCreateWithPriority(&sf, cudaStreamNonBlocking, phi));
  u64* sink;
  CK(cudaMalloc(&sink, 64));
  cudaEvent_t g0e, g1e, f0e, f1e;
  for (cudaEvent_t* e : {&g0e, &g1e, &f0e, &f1e}) CK(cudaEventCreate(e));
  auto ms = [](cudaEvent_t a, cudaEvent_t b) {
    float v;
    CK(cudaEventElapsedTime(&v, a, b));
    return (double)v;
  };
  using FK = void (*)(u64, u64*);
  const FK fk[4] = {filler_kernel<0>, filler_kernel<1>, filler_kernel<2>, filler_kernel<3>};
  const char* fname[4] = {"sha", "alu", "fma", "sleep"};
  const u64 it0[4] = {16, 64, 64, 64};
  const int carve = env_int("V31_CARVE", 100);
  if (carve >= 0)
    for (FK k : fk) CK(cudaFuncSetAttribute((const void*)k, cudaFuncAttributePreferredSharedMemoryCarveout, carve));
  auto words = [](const std::string& s) {
    std::vector<std::string> v;
    std::string w;
    for (char ch : s + " ") {
      if (ch == ' ' || ch == ',') { if (!w.empty()) v.push_back(w); w.clear(); }
      else w += ch;
    }
    return v;
  };
  const std::vector<std::string> gemm_names = words(env_str("V31_CGEMMS", "lt"));
  const std::vector<std::string> modes = words(env_str("V31_CMODES", "sha alu fma sleep"));
  for (const std::string& sh : shapes) {
    char name[64];
    int m, n, k;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &n, &k) != 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    int8_t *A, *B;
    int* D;
    CK(cudaMalloc(&A, (size_t)m * k));
    CK(cudaMalloc(&B, (size_t)n * k));
    CK(cudaMalloc(&D, (size_t)m * n * 4));
    gen_kernel<<<m, 256, 0, sg>>>(A, k, 1u, true);
    gen_kernel<<<n, 256, 0, sg>>>(B, k, 2u, false);
    Lt lt{};
    lt_setup(lt, m, n, k, A, B, D, sg);
    for (const std::string& gn : gemm_names) {
      Cut cut;
      if (gn != "lt") {
        cut_setup(cut, m, n, k, A, B, D, sg, gn.c_str());
        if (!cut.best) { fprintf(stderr, "%s: no CUTLASS %s for it\n", name, gn.c_str()); continue; }
      }
      auto gemms = [&](int r) {
        CK(cudaEventRecord(g0e, sg));
        for (int i = 0; i < r; i++) {
          if (cut.best) cut_run(cut, sg);
          else lt_run(lt, A, B, D, sg);
        }
        CK(cudaEventRecord(g1e, sg));
      };
      auto fill = [&](int f, int cpb, u64 iters) {
        CK(cudaEventRecord(f0e, sf));
        fk[f]<<<nsm * cpb, SMALL_THREADS, 0, sf>>>(iters, sink);
        CK(cudaEventRecord(f1e, sf));
      };
      gemms(3);
      CK(cudaDeviceSynchronize());
      std::vector<double> gv;
      for (int r = 0; r < reps; r++) {
        gemms(4);
        CK(cudaDeviceSynchronize());
        gv.push_back(ms(g0e, g1e) / 4);
      }
      const double g0 = gk::median(gv);
      const int R = std::max(2, (int)std::ceil(20.0 / g0));
      for (int f = 0; f < 4; f++) {
        if (std::find(modes.begin(), modes.end(), fname[f]) == modes.end()) continue;
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, (const void*)fk[f]));
        for (int cpb = 1; cpb <= 3; cpb++) {
          // the filler alone at about 6 R g0 (twice: the first guess's time is mostly launch), so it outlasts R GEMMs
          // slowed up to 6x; f1's GEMMs alone take 6 f0, so they outlast a filler slowed up to 6x
          u64 iters = it0[f];
          for (int pass = 0; pass < 2; pass++) {
            fill(f, cpb, iters);
            CK(cudaDeviceSynchronize());
            iters = std::max<u64>(1, (u64)((double)iters * 6.0 * R * g0 / std::max(ms(f0e, f1e), 1e-3)));
          }
          std::vector<double> f0v, g1v, f1v;
          bool valid = true;
          for (int r = 0; r < reps; r++) {
            fill(f, cpb, iters);
            CK(cudaDeviceSynchronize());
            f0v.push_back(ms(f0e, f1e));
            fill(f, cpb, iters);
            gemms(R);
            CK(cudaDeviceSynchronize());
            g1v.push_back(ms(g0e, g1e) / R);
            valid = valid && ms(g1e, f1e) > 0;
            gemms(std::max(2, (int)std::ceil(6.0 * f0v.back() / g0)));
            fill(f, cpb, iters);
            CK(cudaDeviceSynchronize());
            f1v.push_back(ms(f0e, f1e));
            valid = valid && ms(f1e, g1e) > 0;
          }
          const double f0 = gk::median(f0v), g1 = gk::median(g1v), f1 = gk::median(f1v), sg_ = g1 / g0, sf_ = f1 / f0;
          printf("{\"contend\": true, \"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"gpu\": \"%s\", \"sms\": %d, "
                 "\"gemm\": \"%s\", \"lt\": {\"algo\": %d, \"tile\": %d, \"stages\": %d}, \"mode\": \"%s\", \"cpb\": %d, "
                 "\"regs\": %d, \"carve\": %d, \"iters\": %llu, \"R\": %d, \"g0\": %.6f, \"g1\": %.6f, \"f0\": %.6f, "
                 "\"f1\": %.6f, \"s_g\": %.4f, \"s_f\": %.4f, \"share\": %.4f, \"valid\": %s}\n",
                 name, m, n, k, prop.name, nsm, gn.c_str(), lt.algo_id, lt.tile, lt.stages, fname[f], cpb, fa.numRegs,
                 carve, (unsigned long long)iters, R, g0, g1, f0, f1, sg_, sf_, 1 / sg_ + 1 / sf_,
                 valid ? "true" : "false");
          fflush(stdout);
        }
      }
      cut_free(cut);
    }
    lt_free(lt);
    CK(cudaFree(A));
    CK(cudaFree(B));
    CK(cudaFree(D));
  }
  CK(cudaGetLastError());
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 9 && !strcmp(argv[1], "fill-bench"))
    return cmd_fill_bench(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atoi(argv[7]),
                          std::vector<std::string>(argv + 8, argv + argc));
  if (argc == 7 && !strcmp(argv[1], "once")) return cmd_once(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6]);
  if (argc == 7 && !strcmp(argv[1], "fill-once"))
    return cmd_fill_once(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6]);
  if (argc >= 10 && !strcmp(argv[1], "gate"))
    return cmd_gate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), argv[9]);
  if (argc >= 9 && !strcmp(argv[1], "bench"))
    return cmd_bench(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atoi(argv[7]),
                     std::vector<std::string>(argv + 8, argv + argc));
  if (argc >= 4 && !strcmp(argv[1], "contend"))
    return cmd_contend(atoi(argv[2]), std::vector<std::string>(argv + 3, argv + argc));
  if (argc >= 2 && !strcmp(argv[1], "attrs")) {
    kernel_attrs(stdout);
    printf("\n");
    return 0;
  }
  fprintf(stderr, "usage: v31 gate DIR M N K LP T ROW CHEAT | bench REPS OUT PDIR LP T ROW name:M:N:K... | once PDIR LP T ROW "
                  "name:M:N:K | fill-bench REPS OUT PDIR LP L ROW name:M:N:K... (L 0: K) | fill-once PDIR LP L ROW name:M:N:K "
                  "| contend REPS name:M:N:K... | attrs\n");
  return 2;
}
#endif
