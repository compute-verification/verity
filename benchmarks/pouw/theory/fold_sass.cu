// The three served receipt folds as per-lane word chains, for the SASS instruction mix FOLD-OPT prices on each device:
// fx1 on sm_120, li on the H200 (sm_90a) and li heavy (li + keyed rounds) on the B200 (sm_100a). Each kernel is the
// reference fold of `verity/core/protocols/pouw/circuit/segment.py` (`_fx1_absorb`, `_li_absorb`, and li's extra keyed
// round `y <- (y ^ k) * a + c`), one chain a lane, the words and keys in constant memory so ptxas cannot specialise the
// multiplier back into a shift. Built for sm_90a, sm_100a and sm_120a; `fold_sass.sh` dumps each kernel's SASS
// (`fold_sass_count.py` counts it by census class) and, under a lease, runs `main` on the device: every kernel is
// recomputed bit-exactly on the host from the same steps. clock64 around one warp's unrolled loop is that dependent
// chain's latency a word, not the pipe throughput a certificate prices (that is the honest kernels' measured rate).
//
// What the counts are to show, per device: fx1 is 2 IMAD + 2 ALU a word (4, none fused); li is one LOP3 + one IMAD a
// pair (1 a word); a keyed round is one LOP3 + one IMAD, as the designs' [fold] mixes price them.
#include <stdint.h>
#include <stdio.h>

#define L 128  // words a chain (fx1, li absorb this many per lane; 2L for li's pairs)

__constant__ uint32_t D[2 * L];   // the absorbed words (one lane's), d[j]
__constant__ uint32_t KM[2 * L];  // multipliers (odd), a per step
__constant__ uint32_t KX[2 * L];  // xor keys, k per step
__constant__ uint32_t KC[2 * L];  // added constants / fx1's d-multiplier, per step

__device__ __forceinline__ uint32_t rotl(uint32_t x, int r) { return __funnelshift_l(x, x, r); }
#define ROT(j) ((((j) * 7 + 5) % 31) + 1)

// fx1 (sm_120): d = p * m, u = y ^ d ^ kx, y' = u*a + z, z' = rotl(u, r) + y'. 2 IMAD + 2 ALU a word.
__device__ __forceinline__ void fx1_word(uint32_t &y, uint32_t &z, uint32_t p, uint32_t m, uint32_t a, uint32_t kx,
                                         int r) {
  uint32_t u = y ^ (p * m) ^ kx;
  y = u * a + z;
  z = rotl(u, r) + y;
}

// li (H200): a half step absorbs the pair (d0, d1): y <- (y ^ d0 ^ k) * a + d1. One LOP3 + one IMAD, 1 a word.
__device__ __forceinline__ uint32_t li_half(uint32_t y, uint32_t d0, uint32_t d1, uint32_t k, uint32_t a) {
  return (y ^ d0 ^ k) * a + d1;
}

// a keyed round (B200's heavy fold): y <- (y ^ k) * a + c. One LOP3 (XOR) + one IMAD, absorbs no word.
__device__ __forceinline__ uint32_t keyed_round(uint32_t y, uint32_t k, uint32_t a, uint32_t c) {
  return (y ^ k) * a + c;
}

extern "C" __global__ void fx1_chain(const uint32_t *__restrict__ st, uint32_t *__restrict__ out, long long *cyc) {
  uint32_t y = st[threadIdx.x], z = st[32 + threadIdx.x];
  long long t0 = clock64();
#pragma unroll
  for (int j = 0; j < L; ++j) fx1_word(y, z, D[j], KM[j], KC[j], KX[j], ROT(j));
  long long t1 = clock64();
  out[threadIdx.x] = y ^ rotl(z, 16);
  if (threadIdx.x == 0) *cyc = t1 - t0;
}

extern "C" __global__ void li_chain(const uint32_t *__restrict__ st, uint32_t *__restrict__ out, long long *cyc) {
  uint32_t y = st[threadIdx.x];
  long long t0 = clock64();
#pragma unroll
  for (int j = 0; j < L; ++j) y = li_half(y, D[2 * j], D[2 * j + 1], KX[j], KM[j]);
  long long t1 = clock64();
  out[threadIdx.x] = y;
  if (threadIdx.x == 0) *cyc = t1 - t0;
}

// li heavy: each pair's half step, then (3 of every 8 words) a keyed round, as the B200 design folds it.
extern "C" __global__ void li_heavy_chain(const uint32_t *__restrict__ st, uint32_t *__restrict__ out,
                                          long long *cyc) {
  uint32_t y = st[threadIdx.x];
  long long t0 = clock64();
#pragma unroll
  for (int j = 0; j < L; ++j) {
    y = li_half(y, D[2 * j], D[2 * j + 1], KX[j], KM[j]);
    if ((j & 3) != 0) y = keyed_round(y, KX[L + j], KM[L + j], KC[L + j]);  // 3 of every 4 pairs: 0.75 a word
  }
  long long t1 = clock64();
  out[threadIdx.x] = y;
  if (threadIdx.x == 0) *cyc = t1 - t0;
}

#ifndef __CUDA_ARCH__
#include <cstdlib>
#include <cuda_runtime.h>

static uint32_t h_rotl(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

int main() {
  const int NW = 2 * L;
  uint32_t d[NW], km[NW], kx[NW], kc[NW], st[64];
  srand(0xF01D);
  for (int i = 0; i < NW; ++i) {
    d[i] = (uint32_t)rand() * 2654435761u;
    km[i] = ((uint32_t)rand() * 40503u) | 1u;  // odd
    kx[i] = (uint32_t)rand() * 2246822519u;
    kc[i] = (uint32_t)rand() * 3266489917u;
  }
  for (int i = 0; i < 64; ++i) st[i] = (uint32_t)rand() * 668265263u;
  cudaMemcpyToSymbol(D, d, sizeof d);
  cudaMemcpyToSymbol(KM, km, sizeof km);
  cudaMemcpyToSymbol(KX, kx, sizeof kx);
  cudaMemcpyToSymbol(KC, kc, sizeof kc);
  uint32_t *d_st, *d_out;
  long long *d_cyc;
  cudaMalloc(&d_st, sizeof st);
  cudaMalloc(&d_out, 32 * sizeof(uint32_t));
  cudaMalloc(&d_cyc, sizeof(long long));
  cudaMemcpy(d_st, st, sizeof st, cudaMemcpyHostToDevice);

  int bad = 0;
  long long cyc;
  uint32_t out[32];

  // fx1
  fx1_chain<<<1, 32>>>(d_st, d_out, d_cyc);
  cudaMemcpy(out, d_out, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cyc, d_cyc, sizeof(long long), cudaMemcpyDeviceToHost);
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t y = st[lane], z = st[32 + lane];
    for (int j = 0; j < L; ++j) {
      uint32_t u = y ^ (d[j] * km[j]) ^ kx[j];
      y = u * kc[j] + z;
      z = h_rotl(u, ((j * 7 + 5) % 31) + 1) + y;
    }
    if (out[lane] != (y ^ h_rotl(z, 16))) bad++;
  }
  printf("fx1_chain  bitexact=%d lat_cyc_per_word=%.3f\n", bad == 0, (double)cyc / L);

  // li
  bad = 0;
  li_chain<<<1, 32>>>(d_st, d_out, d_cyc);
  cudaMemcpy(out, d_out, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cyc, d_cyc, sizeof(long long), cudaMemcpyDeviceToHost);
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t y = st[lane];
    for (int j = 0; j < L; ++j) y = (y ^ d[2 * j] ^ kx[j]) * km[j] + d[2 * j + 1];
    if (out[lane] != y) bad++;
  }
  printf("li_chain   bitexact=%d lat_cyc_per_word=%.3f\n", bad == 0, (double)cyc / (2 * L));

  // li heavy
  bad = 0;
  li_heavy_chain<<<1, 32>>>(d_st, d_out, d_cyc);
  cudaMemcpy(out, d_out, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cyc, d_cyc, sizeof(long long), cudaMemcpyDeviceToHost);
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t y = st[lane];
    for (int j = 0; j < L; ++j) {
      y = (y ^ d[2 * j] ^ kx[j]) * km[j] + d[2 * j + 1];
      if ((j & 3) != 0) y = (y ^ kx[L + j]) * km[L + j] + kc[L + j];
    }
    if (out[lane] != y) bad++;
  }
  printf("li_heavy_chain bitexact=%d lat_cyc_per_word=%.3f\n", bad == 0, (double)cyc / (2 * L));

  cudaFree(d_st);
  cudaFree(d_out);
  cudaFree(d_cyc);
  return 0;
}
#endif
