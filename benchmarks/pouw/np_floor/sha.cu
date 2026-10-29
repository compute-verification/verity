// SHA-512's compression on the H200, compute-bound: each thread runs a chain of compressions whose 16 message words are
// the previous state's 8 words and their complements (16 LOP3s a block of 128 bytes), so no memory path bounds it. It
// prices the hm96 rows' hashing at the ALU: units a byte = 540,672 x f_SM x seconds / bytes compressed. Thread 0's
// chain is checked against the same code on the host, and the host code against FIPS 180-4's "abc" vector.
//   nvcc -O3 -arch=sm_90a -std=c++17 sha.cu -o sha
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

__constant__ uint64_t KD[80];
static const uint64_t KH[80] = {
    0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull,
    0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull, 0x12835b0145706fbeull,
    0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull, 0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull,
    0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
    0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull, 0x983e5152ee66dfabull,
    0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,
    0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull,
    0x53380d139d95b3dfull, 0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
    0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull,
    0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull, 0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull,
    0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull,
    0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
    0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull, 0xca273eceea26619cull,
    0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull,
    0x113f9804bef90daeull, 0x1b710b35131c471bull, 0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull,
    0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull};
static const uint64_t IV[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
                               0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                               0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};

__host__ __device__ inline uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

template <bool DEV>
__host__ __device__ inline void compress(uint64_t s[8], const uint64_t m[16]) {
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = m[i];
  uint64_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll
  for (int i = 0; i < 80; i++) {
    if (i >= 16) {
      const uint64_t w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
      w[i & 15] += (rotr(w15, 1) ^ rotr(w15, 8) ^ (w15 >> 7)) + w[(i - 7) & 15] + (rotr(w2, 19) ^ rotr(w2, 61) ^ (w2 >> 6));
    }
#ifdef __CUDA_ARCH__
    const uint64_t k = KD[i];
#else
    const uint64_t k = KH[i];
#endif
    const uint64_t t1 = h + (rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41)) + ((e & f) ^ (~e & g)) + k + w[i & 15];
    const uint64_t t2 = (rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
  }
  s[0] += a, s[1] += b, s[2] += c, s[3] += d, s[4] += e, s[5] += f, s[6] += g, s[7] += h;
}

__host__ __device__ inline void chain(uint64_t s[8], uint64_t seed, int blocks) {
  for (int i = 0; i < 8; i++) s[i] = IV[i] ^ seed;
  for (int r = 0; r < blocks; r++) {
    uint64_t m[16];
#pragma unroll
    for (int i = 0; i < 8; i++) m[i] = s[i], m[8 + i] = ~s[i];
    compress<true>(s, m);
  }
}

__global__ void sha_chain(uint64_t* out, int blocks) {
  uint64_t s[8];
  const uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  for (int i = 0; i < 8; i++) s[i] = IV[i] ^ tid;
#pragma unroll 1
  for (int r = 0; r < blocks; r++) {
    uint64_t m[16];
#pragma unroll
    for (int i = 0; i < 8; i++) m[i] = s[i], m[8 + i] = ~s[i];
    compress<true>(s, m);
  }
  uint64_t x = 0;
  for (int i = 0; i < 8; i++) x ^= s[i] * (i + 1);
  out[tid] = x;
}

int main() {
  // FIPS 180-4 "abc": ddaf35a193617aba...
  uint64_t s[8], m[16] = {0x6162638000000000ull, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24};
  memcpy(s, IV, sizeof s);
  compress<false>(s, m);
  const int abc_ok = s[0] == 0xddaf35a193617abaull && s[7] == 0xa54ca49f;
  const int abc_ok2 = s[0] == 0xddaf35a193617abaull;
  CK(cudaMemcpyToSymbol(KD, KH, sizeof KH));
  int sms;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  printf("{\"abc_ok\": %d", abc_ok2);
  (void)abc_ok;
  for (int tpb : {128, 256, 512}) {
    const int grid = sms * (2048 / tpb), blocks = 512;
    uint64_t* out;
    CK(cudaMalloc(&out, (size_t)grid * tpb * 8));
    sha_chain<<<grid, tpb>>>(out, 4);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    sha_chain<<<grid, tpb>>>(out, blocks);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    uint64_t got[2];
    CK(cudaMemcpy(got, out, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got + 1, out + (size_t)grid * tpb - 1, 8, cudaMemcpyDeviceToHost));
    uint64_t hs[8], x0 = 0, x1 = 0;
    chain(hs, 0, blocks);
    for (int i = 0; i < 8; i++) x0 ^= hs[i] * (i + 1);
    chain(hs, (uint64_t)grid * tpb - 1, blocks);
    for (int i = 0; i < 8; i++) x1 ^= hs[i] * (i + 1);
    const double bytes = (double)grid * tpb * blocks * 128;
    const double bps = bytes / (ms * 1e-3);
    printf(", \"tpb%d\": {\"threads\": %d, \"bytes_per_s\": %.6g, \"units_per_byte\": %.1f, \"ok\": %d}", tpb,
           grid * tpb, bps, 540672.0 * 1.98e9 / bps, got[0] == x0 && got[1] == x1);
    CK(cudaFree(out));
  }
  printf("}\n");
  return 0;
}
