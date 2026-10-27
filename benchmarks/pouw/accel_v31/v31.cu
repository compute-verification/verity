// ACCEL v3.1's credited static INT8 matmul on the GPU (verity/core/protocols/pouw/accel/PROTOCOL.md on #4746), the
// prover's side of one call, bit for bit v31.py's reference (its docstring has the semantics, the coin streams and the
// layout; this file follows it):
//
//   coins    d_A, d_B (bits), U, V (uniform on -80..80) and the noise tiles' permutations, from the challenger's five
//            512-bit seeds in SHA-512 counter mode (a stub: v31.py's StubChallenger issues the seeds)
//   form     X = wrap8(A + d_A + V) and W' = wrap8(B + d_B + U pi_tau) with their wrap lists (k << 1 | negative, a row
//            each), X^T and B~^T = (B + d_B + U pi)^T for the wrap correction, and the peel's tail rows
//            P_A = [Z_0 .. Z_{dz-1} | V x dy], P_B = [U x dz | Y_0 .. Y_{dy-1}] (Z_X and Y_W in balanced base-128 digits,
//            the last digit a full int8)
//   wrap     dO = w_X B~^T + X w_W^T, sparse, on the CUDA cores (a 128 x 128 tile a CTA)
//   gemm     X W'^T on the tensor cores (mma.sync s8, a 3-stage TMA ring, 128 x 128 tiles of 8 warps); after each segment
//            of L = t L' inner indices every thread stores its accumulators' low 16 bits (T_s) to the words buffer; then
//            the tail stages subtract P_A P_B^T stage by stage at their digit's weight (128^d), and the epilogue adds
//            256 dO: O = T_S + 256 dO - Z_X U^T - V Y_W^T = (A + d_A)(B + d_B)^T mod 2^32
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
//                                         each commit beside the next call's compute), one JSON line a shape
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1
//        v31.cu -lcuda -lcublasLt
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../commit_gpu/commit_gpu.cuh"
#include "../gpu_kit/bench.cuh"

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
enum { S_DA = 0, S_DB = 1, S_U = 2, S_V = 3, S_P = 4 };
struct Seeds { u64 w[5][8]; };   // dA, dB, U, V, P: each seed's 64 bytes as big-endian words
HD u64 tag2(char a, char b) { return (u64)(uint8_t)a << 56 | (u64)(uint8_t)b << 48; }
HD u64 tag_of(int s) {
  return s == S_DA ? tag2('d', 'A') : s == S_DB ? tag2('d', 'B') : s == S_U ? tag2('U', 0) : s == S_V ? tag2('V', 0) : tag2('P', 0);
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
// pi_tau: Fisher-Yates over L' with word tau L' + c of the P stream (c from L' - 1 down to 1), a CTA of L' threads a tile
__global__ void perm_kernel(Seeds sd, int lp, uint16_t* __restrict__ perm) {
  extern __shared__ u32 pw[];
  uint16_t* p = reinterpret_cast<uint16_t*>(pw + lp);
  const int tau = blockIdx.x, t = threadIdx.x;
  if (t < lp / 16) {
    u64 st[8];
    prg(sd.w[S_P], tag_of(S_P), (u64)tau * (lp / 16) + t, st);
#pragma unroll
    for (int i = 0; i < 16; ++i) pw[16 * t + i] = dword(st, i);
  }
  p[t] = (uint16_t)t;
  __syncthreads();
  if (t == 0)
    for (int c = lp - 1; c > 0; --c) {
      const int x = (int)(((u64)pw[c] * (u64)(c + 1)) >> 32);
      const uint16_t s = p[c]; p[c] = p[x]; p[x] = s;
    }
  __syncthreads();
  perm[(size_t)tau * lp + t] = p[t];
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
// A side: a CTA of L' threads takes 16 rows; thread c the column c of every noise tile. Z_X[i, pi_tau(c)] += A~[i, tau L' + c]
// through shared atomics; X^T in 16-byte columns of the 16 rows.
__global__ void form_a_kernel(FormArgs f) {
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
    const int k = tau * lp + c, pc = f.perm[(size_t)tau * lp + c];
    u32 xp[4] = {0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const size_t i = i0 + r;
      const int at = (int)f.Q[i * f.K + k] + ((f.bits[i * f.kb64 + (k >> 3)] >> (k & 7)) & 1) + v[r];
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
    for (int d = 0; d < f.dz; ++d) p[d * lp + c] = (int8_t)digit(z, d + 1 == f.dz, f.flag);
    for (int d = 0; d < f.dy; ++d) p[(f.dz + d) * lp + c] = (int8_t)v[r];
  }
}
// B side: thread c holds Y_W[j, c] for its 16 rows in registers; U's row gathered through pi from shared memory
__global__ void form_b_kernel(FormArgs f) {
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
    const int k = tau * lp + c, pc = f.perm[(size_t)tau * lp + c];
    u32 bp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const size_t j = j0 + r;
      const int bd = (int)f.Q[j * f.K + k] + ((f.bits[j * f.kb64 + (k >> 3)] >> (k & 7)) & 1), bt = bd + us[r * lp + pc];
      const int x = ((bt + 128) & 255) - 128, w = (bt - x) >> 8;
      f.X[j * f.K + k] = (int8_t)x;
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
    for (int d = 0; d < f.dz; ++d) p[d * lp + c] = us[r * lp + c];
    int z = y[r];
    for (int d = 0; d < f.dy; ++d) p[(f.dz + d) * lp + c] = (int8_t)digit(z, d + 1 == f.dy, f.flag);
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
  const int r = t & 127, g = t >> 7;
  const int na = a.cntA[i0 + r];
  if (na == 0) return;
  const int* LA = a.listA + (i0 + r) * a.capA;
#pragma unroll 1
  for (int half = 0; half < 2; ++half) {
    const size_t c0 = j0 + 64 * g + 32 * half;
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
  const int* dO;
  int* O;            // m x N, row-major (u32 mod 2^32)
  int M, N, K, L, S, TT, group;
  int lp, dz;        // tail stage ts is block ts 128 / L' of P_A P_B^T: Z_b U^T (b < dz) or V Y_{b - dz}^T, at 128^digit
  u32* ctr;          // the tile counter (null: CTA c takes tiles c, c + grid, ...); a launch advances it by its tiles
  u32 base;          // ctr's value before this launch
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

// T_s's low 16 bits, a pair of columns a 32-bit store
DEV void store_words(const int (&acc)[4][4][4], uint16_t* words, size_t tile, int S, int s, int wm, int wn, int g, int q) {
  u32* w = reinterpret_cast<u32*>(words + (tile * S + (size_t)(s - 1)) * (TILE * TILE));
  const int r0 = wm * 64 + g, c0 = wn * 32 + 2 * q;
#pragma unroll
  for (int mi = 0; mi < 4; mi++)
#pragma unroll
    for (int ni = 0; ni < 4; ni++) {
      const int p = (r0 + 16 * mi) * TILE + c0 + 8 * ni;
      w[p >> 1] = __byte_perm((u32)acc[mi][ni][0], (u32)acc[mi][ni][1], 0x5410);
      w[(p + 8 * TILE) >> 1] = __byte_perm((u32)acc[mi][ni][2], (u32)acc[mi][ni][3], 0x5410);
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
  const int KT = a.K / BK;
  auto seg = [&](const int (&acc)[4][4][4], size_t tile, int kt) {
    if constexpr (PROTO)
      if (((kt + 1) * BK) % a.L == 0 || kt + 1 == KT) store_words(acc, a.words, tile, a.S, ((kt + 1) * BK + a.L - 1) / a.L, wm, wn, g, q);
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
    if (KT == 1) {
      R.body<true, true>(acc, fa, fb);
      seg(acc, tile, 0);
    } else {
      R.body<true, false>(acc, fa, fb);
      seg(acc, tile, 0);
#pragma unroll 1
      for (int kt = 1; kt < KT - 1; kt++) {
        R.body<false, false>(acc, fa, fb);
        seg(acc, tile, kt);
      }
      R.body<false, true>(acc, fa, fb);
      seg(acc, tile, KT - 1);
    }
    const int row0 = tm * TILE + wm * 64 + g, col0 = tn * TILE + wn * 32 + 2 * q;
    if constexpr (PROTO) {
#pragma unroll 1
      for (int ts = 0; ts < a.TT; ts++) {
        const int blk = ts * BK / a.lp;
        R.tail(acc, 7 * (blk < a.dz ? blk : blk - a.dz));
      }
    }
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
      for (int ni = 0; ni < 4; ni++) {
        const size_t p = (size_t)(row0 + 16 * mi) * a.N + col0 + 8 * ni, p8 = p + (size_t)8 * a.N;
        int2 lo = make_int2(acc[mi][ni][0], acc[mi][ni][1]), hi = make_int2(acc[mi][ni][2], acc[mi][ni][3]);
        if constexpr (PROTO) {
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
// O's row h of a tile: cells [h ROW / 4, (h + 1) ROW / 4) in the tile's row-major order, read from the m x N output
struct ORow {
  const uint8_t* base;   // the tile's first cell
  u64 stride;            // N 4
  u32 c0;
  DEV uint4 ld16(u32 off) const {
    const u32 c = c0 + off / 4;
    return ldg16(base + (c >> 7) * stride + (c & 127) * 4);
  }
  DEV void pf(u32) const {}
};
// leaf pos: tile pos / LPT; j = pos % LPT < S ROW_W a words row (segment j / ROW_W), else O's row j - S ROW_W
struct V31Row {
  const uint8_t* words;
  const uint8_t* O;
  u64 lpt, srw, N, tiles_n;
  u32 row, S;
  DEV void digest(u64 pos, u64* x) const {
    const u64 t = pos / lpt, j = pos % lpt;
    if (j < srw) {
      hash_after_block(C.mid_P, Contig{words + t * S * 32768ull + j * row}, row, x);
    } else {
      const u64 tm = t / tiles_n, tn = t % tiles_n;
      hash_after_block(C.mid_P, ORow{O + (tm * TILE * N + tn * TILE) * 4, N * 4, (u32)((j - srw) * row / 4)}, row, x);
    }
  }
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
// salt_small<true> (slot 0) a block at a time: S and F for positions below n, the pads up to width, and the subtrees of
// blocks that hold only pads (nvb = ceil(width / 128))
__global__ void __launch_bounds__(SMALL_THREADS, SMALL_MINB)
    salt_persist(Persist p, int dom, const uint4* __restrict__ Tg, u64 n, u64 width, u64* __restrict__ S, u64* __restrict__ F,
                 u64* levels) {
  const u32 cnt = (u32)(width < SMALL_THREADS ? width : SMALL_THREADS);
  if (persist_claim(p))
    for (u32 vb; (vb = persist_next(p)) < p.nvb;) {
      const u64 pos = (u64)vb * SMALL_THREADS + threadIdx.x;
      if (pos < n) {
        salt_one<true, false, true>(pos, nullptr, reinterpret_cast<const uint8_t*>(Tg), S);
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

template <class T> static T* dalloc(size_t n) {
  T* p = nullptr;
  CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

struct Cheat {
  int kind = 0;   // 0 none, 1 skip, 2 forge (T_s0), 3 forgeO
  int tau = 0, s0 = 0;
  double frac = 1.0;
};
static Cheat parse_cheat(const std::string& s) {
  Cheat c;
  if (s == "none" || s.empty()) return c;
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
  fprintf(stderr, "cheat %s: none | skip:TAU | forge:S0[:FRAC] | forgeO[:FRAC]\n", s.c_str());
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
// the wrap correction's kernel (1: wrap_kernel, 2: wrap2_kernel), V31_WCW its columns a CTA (256 where N allows, or 128);
// V31_CCTA: the salt and leaf phases' CTAs an SM (salt_persist, leaf_persist; 0: commit_gpu's salt_small and leaf_small).
struct Knobs {
  int carve = env_int("V31_CARVE", 100), dyn = env_int("V31_DYN", 1), wrap = env_int("V31_WRAP", 2), wcw = env_int("V31_WCW", 256),
      ccta = env_int("V31_CCTA", 1);
  std::string cprio = env_str("V31_CPRIO", "hi");
};
static void set_carveout(int pct) {
  if (pct < 0) return;
  const void* ks[] = {(const void*)gemm_kernel<true>,  (const void*)gemm_kernel<false>,  (const void*)wrap_kernel,
                      (const void*)wrap2_kernel<128>,  (const void*)wrap2_kernel<256>,   (const void*)form_a_kernel,
                      (const void*)form_b_kernel,      (const void*)bits_kernel,         (const void*)noise_kernel,
                      (const void*)perm_kernel,        (const void*)salt_small<true>,    (const void*)leaf_small<V31Row, 0>,
                      (const void*)level_kernel,       (const void*)top_small,           (const void*)salt_persist,
                      (const void*)leaf_persist<V31Row>};
  for (const void* k : ks) CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
}

// One shape's prover: every buffer of a call, NB copies of the committed values (words, O) for the overlapped arm.
struct Ctx {
  int m, n, k, lp, t, row, L, S, NT, kb64, dz, dy, TL, TT, tiles, tiles_n, lpt, grid, nsm, capA, capB, nb = 1;
  u64 leaves;
  Seeds seeds;
  int8_t *A, *B, *U, *V, *X, *XT, *W, *PA, *PB;
  int16_t* BtT;
  uint8_t *dA, *dB;
  uint16_t* perm;
  int *listA, *cntA, *listB, *cntB, *offB, *dO, *flag;
  int wcw = 128;
  uint16_t* words[2] = {nullptr, nullptr};
  int* O[2] = {nullptr, nullptr};
  Maps *maps_proto, *maps_plain;
  Committer cm;
  Params params;
  cudaStream_t s0, s1;
  cudaEvent_t ev_comp[2], ev_done[2];
  int ovl = 0;
  Knobs kn;
  u32* gctr = nullptr;
  u32 gbase = 0;
  u32 *pctl = nullptr, *pclaim = nullptr, epoch = 0;

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
    pctl = dalloc<u32>(2 + 2 * 8 * (size_t)nsm);
    pclaim = dalloc<u32>(1024);
    CK(cudaMemset(pctl, 0, 4 * (2 + 2 * 8 * (size_t)nsm)));
    CK(cudaMemset(pclaim, 0, 4096));
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
    CK(cudaFuncSetAttribute(gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Ring::SMEM));
    CK(cudaFuncSetAttribute(gemm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Ring::SMEM));
    CK(cudaFuncSetAttribute(wrap_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TILE * 129 * 4));
    CK(cudaFuncSetAttribute(wrap2_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, lp * 128));
    CK(cudaFuncSetAttribute(wrap2_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, lp * 128));
    for (int b = 0; b < 2; b++) {
      CK(cudaEventCreateWithFlags(&ev_comp[b], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&ev_done[b], cudaEventDisableTiming));
      CK(cudaEventRecord(ev_done[b], s1));
    }
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
  GArgs gargs(int b, bool proto) const {
    return GArgs{proto ? maps_proto : maps_plain, words[b], dO, O[b], m, n, k, L, S, proto ? TT : 0, 8, lp, dz,
                 kn.dyn ? gctr : nullptr, gbase};
  }
  V31Row rowsrc(int b) const {
    return V31Row{reinterpret_cast<const uint8_t*>(words[b]), reinterpret_cast<const uint8_t*>(O[b]), (u64)lpt,
                  (u64)S * (32768 / row), (u64)n, (u64)tiles_n, (u32)row, (u32)S};
  }
  void coins(cudaStream_t st) {
    const u64 ba = (u64)m * (kb64 / 64), bb = (u64)n * (kb64 / 64);
    bits_kernel<<<(unsigned)((ba + 127) / 128), 128, 0, st>>>(seeds, S_DA, ba, dA);
    bits_kernel<<<(unsigned)((bb + 127) / 128), 128, 0, st>>>(seeds, S_DB, bb, dB);
    const u64 cu = (u64)n * lp, cv = (u64)m * lp;
    noise_kernel<<<(unsigned)((cu / 16 + 127) / 128), 128, 0, st>>>(seeds, S_U, cu, U);
    noise_kernel<<<(unsigned)((cv / 16 + 127) / 128), 128, 0, st>>>(seeds, S_V, cv, V);
    perm_kernel<<<NT, lp, lp * 6, st>>>(seeds, lp, perm);
  }
  void form(cudaStream_t st) {
    form_a_kernel<<<m / 16, lp, 16 * lp * 4, st>>>(fa());
    form_b_kernel<<<n / 16, lp, 16 * lp, st>>>(fb());
  }
  void wrap(cudaStream_t st) {
    const WrapArgs w{XT, BtT, listA, cntA, listB, cntB, dO, m, n, capA, capB, offB, lp, NT};
    if (kn.wrap == 1) wrap_kernel<<<dim3(n / TILE, m / TILE), 256, TILE * 129 * 4, st>>>(w);
    else if (wcw == 256) wrap2_kernel<256><<<dim3(n / 256, m / TILE), 512, lp * 128, st>>>(w);
    else wrap2_kernel<128><<<dim3(n / 128, m / TILE), 256, lp * 128, st>>>(w);
  }
  // launches that share gctr run in one stream's order, so each one's base is the tiles launched before it
  void gemm(cudaStream_t st, int b, bool proto) {
    if (proto) gemm_kernel<true><<<grid, Ring::NCT, Ring::SMEM, st>>>(gargs(b, true));
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
  // a capped launch over nvb blocks: twice the CTAs the caps allow, so SMs whose first CTAs land late still get theirs
  Persist persist(u64 nvb) {
    epoch = (epoch + 1) & 0xFFFFFF;
    return Persist{pctl, pclaim, epoch, (u32)kn.ccta, (u32)nvb};
  }
  unsigned persist_grid(u64 nvb) const { return (unsigned)std::min<u64>(nvb, 2ull * kn.ccta * nsm); }
  void salts(cudaStream_t st) {
    if (!kn.ccta) { cm.salts<true>(nullptr, st); return; }
    const u64 nvb = (cm.width + SMALL_THREADS - 1) / SMALL_THREADS;
    salt_persist<<<persist_grid(nvb), SMALL_THREADS, 0, st>>>(persist(nvb), cm.dom, cm.Tn, cm.n, cm.width, cm.S, cm.F, cm.levels);
  }
  void leaves_nodes(cudaStream_t st, int b) {
    if (!kn.ccta) {
      cm.leaves(rowsrc(b), st);
    } else {
      const u64 nvb = (cm.n + SMALL_THREADS - 1) / SMALL_THREADS;
      leaf_persist<V31Row><<<persist_grid(nvb), SMALL_THREADS, 0, st>>>(persist(nvb), cm.dom, rowsrc(b), cm.S, cm.F, cm.n,
                                                                        cm.width, cm.levels);
    }
    cm.nodes(st);
  }
  void commit(cudaStream_t st, int b) {
    salts(st);
    leaves_nodes(st, b);
  }
  // the overlapped call: compute on s0 once the commit that last read buffer b is done; its commit on s1, the salts
  // ahead of the compute they don't read
  void call_ovl(cudaStream_t st) {
    const int b = nb == 2 ? (ovl++ & 1) : 0;
    CK(cudaStreamWaitEvent(st, ev_done[b]));
    compute(st, b);
    CK(cudaEventRecord(ev_comp[b], st));
    salts(s1);
    CK(cudaStreamWaitEvent(s1, ev_comp[b]));
    leaves_nodes(s1, b);
    CK(cudaEventRecord(ev_done[b], s1));
  }
  void ovl_drain(cudaStream_t st) {
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

static std::vector<uint8_t> readfile(const std::string& p, size_t want) {
  std::vector<uint8_t> v = readf(p);
  if (want && v.size() != want) { fprintf(stderr, "%s: %zu bytes, want %zu\n", p.c_str(), v.size(), want); exit(2); }
  return v;
}

struct KAttr { const char* name; const void* fn; };
static void kernel_attrs(FILE* f) {
  const KAttr ks[] = {{"gemm_proto", (const void*)gemm_kernel<true>}, {"gemm_plain", (const void*)gemm_kernel<false>},
                      {"form_a", (const void*)form_a_kernel}, {"form_b", (const void*)form_b_kernel},
                      {"wrap", (const void*)wrap_kernel}, {"wrap2_128", (const void*)wrap2_kernel<128>},
                      {"wrap2_256", (const void*)wrap2_kernel<256>}, {"bits", (const void*)bits_kernel},
                      {"leaf_small", (const void*)leaf_small<V31Row, 0>}, {"salt_small", (const void*)salt_small<true>},
                      {"leaf_persist", (const void*)leaf_persist<V31Row>}, {"salt_persist", (const void*)salt_persist}};
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
    };
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
           "\"lt\": {\"algo\": %d, \"tile\": %d, \"stages\": %d, \"splitk\": %d, \"candidates\": %d}, \"steady\": %s, "
           "\"carve\": %d, \"dyn\": %d, \"cprio\": \"%s\", \"wrap\": %d, \"wcw\": %d, \"ccta\": %d, \"maxnreg\": %d, \"arms\": {",
           name, m, n, k, lp, t, row, c.S, c.dz, c.dy, (unsigned long long)c.leaves, prop.name, c.nsm, c.grid, c.nb,
           (double)wa / ((double)m * k), (double)wbn / ((double)n * k), lt.algo_id, lt.tile, lt.stages, lt.splitk, lt.ncand,
           tm.steady ? "true" : "false", c.kn.carve, c.kn.dyn, c.kn.cprio.c_str(), c.kn.wrap, c.wcw, c.kn.ccta, V31_MAXNREG);
    for (size_t a = 0; a < arms.size(); a++)
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f}", a ? ", " : "", arms[a].name.c_str(), med[a], mhz[a]);
    printf("}, \"gamma_call\": %.4f, \"gamma_ovl\": %.4f, \"gamma_gemm\": %.4f, \"plain_over_lt\": %.4f, \"kernels\": ", med[7] / med[0],
           med[8] / med[0], med[5] / med[0], med[1] / med[0]);
    kernel_attrs(stdout);
    printf("}\n");
    fflush(stdout);
    FILE* items = fopen((out + "/items-" + name + ".jsonl").c_str(), "w");
    if (items) { gk::items_jsonl(tm, arms, items); fclose(items); }
    CK(cudaDeviceSynchronize());
    CK(cudaDeviceReset());
  }
  return 0;
}

int main(int argc, char** argv) {
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
  fprintf(stderr, "usage: v31 gate DIR M N K LP T ROW CHEAT | bench REPS OUT PDIR LP T ROW name:M:N:K... | attrs\n");
  return 2;
}
