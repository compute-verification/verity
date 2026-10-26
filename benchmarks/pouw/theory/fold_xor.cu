// FOLD-XOR's device check (`fold_xor.py`): the 78-instruction IMAD-pipe program for z = x ^ y ^ K, as ptxas builds
// it for sm_100a, and bit-exact on the device. Every step is inline PTX mad.lo.u32 (IMAD) or mad.hi.u32 (IMAD.HI), and
// every multiplier and addend is in constant memory, so ptxas can't specialise a power-of-two multiplier into a shift
// (class ALU). Kernels:
//   - pairs: z[i] = xor78(x[i], y[i]) over random words and the edge words crossed, checked on the host;
//   - chain<N>: a, b = b, xor78(a, b) N times, fully unrolled, no load or address arithmetic a step, so the SASS of
//     chain<16> less chain<8>, over 8, is one XOR's instruction mix (`fold_xor.py --sass` counts it by census class);
//   - loop: the same step over a runtime count, not unrolled, so the loop body is one XOR in steady state.
// `main` checks pairs, both chains and the loop against the host's a ^ b ^ K, for K = 0 and one key, and exits 3 on a
// mismatch. SASS's IMAD.HI.U32 adds a 64-bit register pair, and ptxas lowers mad.hi.u32's addend c to a pair {0, c}
// it rebuilds at every use (MOV, HFMA2), unrolled and in the loop alike; `fold_xor.py`'s `sass_body` is the
// allocation that holds the pairs instead.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__constant__ uint32_t FM, FA, GM, GA;           // chunk 0: F = y*FM + FA, G = y*GM + GA, Q = x*F + G
__constant__ uint32_t SH[16], EA[16], HA[16];  // chunk j: shift 2^(32-2j), E's addend s*3*2^30, H's (c - s)*2^30
__constant__ uint32_t ONE, C31, C30;

__device__ __forceinline__ uint32_t mlo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ uint32_t mhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ uint32_t xor78(uint32_t x, uint32_t y) {
  uint32_t F = mlo(y, FM, FA), G = mlo(y, GM, GA), Q = mlo(x, F, G);
#pragma unroll
  for (int j = 1; j < 16; ++j) {
    uint32_t Bp = mhi(y, SH[j], ONE);
    uint32_t Sp = mhi(x, SH[j], Bp);
    uint32_t E = mlo(Bp, C31, EA[j]);
    uint32_t H = mhi(Q, C30, HA[j]);
    Q = mlo(Sp, E, H);
  }
  return Q;
}

__global__ void pairs(const uint32_t* x, const uint32_t* y, uint32_t* z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) z[i] = xor78(x[i], y[i]);
}

template <int N>
__global__ void chain(const uint32_t* x, const uint32_t* y, uint32_t* z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t a = x[i], b = y[i];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    uint32_t t = xor78(a, b);
    a = b;
    b = t;
  }
  z[i] = b;
}

template __global__ void chain<8>(const uint32_t*, const uint32_t*, uint32_t*, int);
template __global__ void chain<16>(const uint32_t*, const uint32_t*, uint32_t*, int);

// The served form: one XOR an iteration of a loop over a runtime count, so ptxas can hoist the constants out of it.
__global__ void loop(const uint32_t* x, const uint32_t* y, uint32_t* z, int n, int reps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t a = x[i], b = y[i];
#pragma unroll 1
  for (int r = 0; r < reps; ++r) {
    uint32_t t = xor78(a, b);
    a = b;
    b = t;
  }
  z[i] = b;
}

static const int AFF[4][2] = {{1, 0}, {-1, 1}, {1, 2}, {-1, 3}};  // u ^ k on 2 bits = s*u + c (mod 4)

static void keyed_constants(uint32_t K) {
  int s = AFF[K & 3][0], c = AFF[K & 3][1];
  uint32_t fm = (uint32_t)(-(int64_t)s * (1ll << 31)), fa = (uint32_t)((int64_t)s * (1ll << 30)), gm = fa,
           ga = (uint32_t)c << 30;
  uint32_t sh[16] = {0}, ea[16] = {0}, ha[16] = {0};
  for (int j = 1; j < 16; ++j) {
    s = AFF[K >> (2 * j) & 3][0];
    c = AFF[K >> (2 * j) & 3][1];
    sh[j] = (uint32_t)1 << (32 - 2 * j);
    ea[j] = (uint32_t)((int64_t)s * (3ll << 30));
    ha[j] = (uint32_t)((int64_t)(c - s) * (1ll << 30));
  }
  uint32_t one = 1, c31 = 1u << 31, c30 = 1u << 30;
  cudaMemcpyToSymbol(FM, &fm, 4);
  cudaMemcpyToSymbol(FA, &fa, 4);
  cudaMemcpyToSymbol(GM, &gm, 4);
  cudaMemcpyToSymbol(GA, &ga, 4);
  cudaMemcpyToSymbol(SH, sh, sizeof sh);
  cudaMemcpyToSymbol(EA, ea, sizeof ea);
  cudaMemcpyToSymbol(HA, ha, sizeof ha);
  cudaMemcpyToSymbol(ONE, &one, 4);
  cudaMemcpyToSymbol(C31, &c31, 4);
  cudaMemcpyToSymbol(C30, &c30, 4);
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t next32() {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (uint32_t)(rng >> 16);
}

int main() {
  const uint32_t edge[] = {0u, 1u, 0xFFFFFFFFu, 0xFFFFFFFEu, 0x80000000u, 0x7FFFFFFFu, 0x55555555u, 0xAAAAAAAAu,
                           0x0F0F0F0Fu, 0xF0F0F0F0u, 0x80000001u, 0x7FFFFFFEu};
  const int ne = sizeof edge / sizeof edge[0], nr = 1 << 20, n = ne * ne + nr;
  uint32_t *hx = (uint32_t*)malloc(4 * n), *hy = (uint32_t*)malloc(4 * n), *hz = (uint32_t*)malloc(4 * n);
  for (int a = 0; a < ne; ++a)
    for (int b = 0; b < ne; ++b) hx[a * ne + b] = edge[a], hy[a * ne + b] = edge[b];
  for (int i = ne * ne; i < n; ++i) hx[i] = next32(), hy[i] = next32();
  uint32_t *dx, *dy, *dz;
  cudaMalloc(&dx, 4 * n);
  cudaMalloc(&dy, 4 * n);
  cudaMalloc(&dz, 4 * n);
  cudaMemcpy(dx, hx, 4 * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy, 4 * n, cudaMemcpyHostToDevice);
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device %s sm_%d%d\n", p.name, p.major, p.minor);
  const uint32_t keys[] = {0u, 0x9E13B91Cu};
  long bad_total = 0;
  for (uint32_t K : keys) {
    keyed_constants(K);
    for (int kind = 0; kind < 4; ++kind) {
      int blocks = (n + 255) / 256, reps = kind == 1 ? 8 : kind == 2 ? 16 : 5;
      if (kind == 0) pairs<<<blocks, 256>>>(dx, dy, dz, n);
      if (kind == 1) chain<8><<<blocks, 256>>>(dx, dy, dz, n);
      if (kind == 2) chain<16><<<blocks, 256>>>(dx, dy, dz, n);
      if (kind == 3) loop<<<blocks, 256>>>(dx, dy, dz, n, reps);
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        printf("cuda error %s\n", cudaGetErrorString(e));
        return 2;
      }
      cudaMemcpy(hz, dz, 4 * n, cudaMemcpyDeviceToHost);
      long bad = 0;
      for (int i = 0; i < n; ++i) {
        uint32_t want;
        if (kind == 0) {
          want = hx[i] ^ hy[i] ^ K;
        } else {
          uint32_t a = hx[i], b = hy[i];
          for (int j = 0; j < reps; ++j) {
            uint32_t t = a ^ b ^ K;
            a = b;
            b = t;
          }
          want = b;
        }
        bad += hz[i] != want;
      }
      const char* name = kind == 0 ? "pairs" : kind == 1 ? "chain8" : kind == 2 ? "chain16" : "loop5";
      printf("bitexact key 0x%08x %s %d points %ld mismatches\n", K, name, n, bad);
      bad_total += bad;
    }
  }
  printf("bitexact total mismatches %ld\n", bad_total);
  return bad_total ? 3 : 0;
}
