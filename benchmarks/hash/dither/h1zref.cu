// The gate's reference for the one-stream z (h1u.cu's kh_one; PouwH1Tile_v3): z = X W-hat^T mod 2^32, X = A + b
// (mod 2^8, the s8 the MMA reads) and W-hat = B + c, which v3's O' - sum_d 128^d D_d + 256 X (wb+ - wb-)^T equals for
// every int8 B. Computed another way: no peel, no digits, no wrap bits, no MMA. One warp a unit of the selected row
// blocks, lane r its row r; the noise from full SHA-512 compressions of NOISE_TAG || seed_j || u64be t (not the
// served kernel's midstate), the products by scalar dp4a over X B^T and X c^T.
// Also, per run: the wrap bits' counts (wb+ where B + c + U > 127, wb- where < -128, B's column against its own noise)
// and X's (A = 127 with b = 1, which X = A + b mod 2^8 makes -128); and unit 0's probe: X[r][0] for r < 32, then
// column 0's WB at k = 0 (+1, -1 or 0), for the flipped-wb fault.
#include "h1sha.cuh"

constexpr int RS = 516;  // a row's 512 bytes, padded so lane r's row starts in its own bank

__device__ __forceinline__ void block16(const u64* seed, u64 t, u32 (&h)[16]) {
  u64 msg[16] = {TAG0, TAG1, TAG2};
#pragma unroll
  for (int j = 0; j < 8; ++j) msg[3 + j] = seed[j];
  msg[11] = t;
  msg[12] = 0x8000000000000000ULL;
  msg[13] = 0;
  msg[14] = 0;
  msg[15] = 96 * 8;
  u64 s[8];
  sha512_words(msg, 1, s);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    h[2 * j] = (u32)(s[j] >> 32);
    h[2 * j + 1] = (u32)s[j];
  }
}

__device__ __forceinline__ int bit_of(const u32 (&h)[16], int k) { return (int)((h[k >> 5] >> (k & 31)) & 1u); }

__global__ void __launch_bounds__(32) hz_ref_k(const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned N,
                                               unsigned K, const u64* __restrict__ seeds, const uint8_t* __restrict__ rsel,
                                               int* __restrict__ zref, unsigned long long* __restrict__ counts,
                                               int* __restrict__ probe) {
  __shared__ __align__(16) int8_t Xs[32][RS];
  __shared__ __align__(16) int8_t Bs[32][RS];
  __shared__ u32 Cb[32][17];
  const unsigned ncb = N / 32, u = blockIdx.x, ri = u / ncb, cj = u % ncb, r = threadIdx.x;
  if (!rsel[ri]) return;
  const unsigned ks = K / 32, chunks = ks / 16, rb = chunks + 1, cb = chunks + 1;
  u64 seed[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) seed[j] = seeds[16 * (u64)u + j];
  int U[32];
  {
    u32 h[16];
    block16(seed, 32ull * rb + (u64)r * cb + chunks, h);
    for (int t = 0; t < 32; ++t) {
      const int x = (int)((h[t >> 2] >> (8 * (t & 3))) & 0xFFu);
      U[t] = (x >= 129 ? x - 129 : x) - 64;
    }
  }
  int acc[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) acc[j] = 0;
  unsigned long long wbp = 0, wbm = 0, xw = 0;
  const int8_t* arow = A + (u64)(32 * ri + r) * K;
  const int8_t* bcol = B + (u64)(32 * cj + r) * K;
  for (unsigned ch = 0; ch < chunks; ++ch) {
    u32 hr[16], hc[16];
    block16(seed, (u64)r * rb + ch, hr);
    block16(seed, 32ull * rb + (u64)r * cb + ch, hc);
    __syncwarp();
    for (int kk = 0; kk < 512; ++kk) {
      const int a = arow[512 * ch + kk], bb = bit_of(hr, kk);
      const int bw = bcol[512 * ch + kk], c = bit_of(hc, kk);
      Xs[r][kk] = (int8_t)(uint8_t)(a + bb);
      Bs[r][kk] = (int8_t)bw;
      xw += a == 127 && bb;
      const int s = bw + c + U[kk & 31];
      wbp += s > 127;
      wbm += s < -128;
      if (u == 0 && ch == 0 && kk == 0) {
        probe[r] = (int)(int8_t)(uint8_t)(a + bb);
        if (r == 0) probe[32] = s > 127 ? 1 : s < -128 ? -1 : 0;
      }
    }
#pragma unroll
    for (int w = 0; w < 16; ++w) Cb[r][w] = hc[w];
    __syncwarp();
    for (int k4 = 0; k4 < 128; ++k4) {
      const int x4 = *reinterpret_cast<const int*>(&Xs[r][4 * k4]);
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const u32 nib = (Cb[j][k4 >> 3] >> (4 * (k4 & 7))) & 15u;
        const int c4 = (int)((nib * 0x00204081u) & 0x01010101u);
        acc[j] = __dp4a(x4, *reinterpret_cast<const int*>(&Bs[j][4 * k4]), acc[j]);
        acc[j] = __dp4a(x4, c4, acc[j]);
      }
    }
  }
  int* zr = zref + (u64)(32 * ri + r) * N + 32 * cj;
#pragma unroll
  for (int j = 0; j < 32; ++j) zr[j] = acc[j];
  atomicAdd(counts + 0, wbp);
  atomicAdd(counts + 1, wbm);
  atomicAdd(counts + 2, xw);
}

// zref [rows, N] s32 (written at the selected row tiles only), counts [3] u64 (added to), probe [33] s32 (written when
// row tile 0 is selected); seeds as kh_seeds writes them, 16 words a unit
extern "C" int hz_ref(const void* A, const void* B, unsigned rows, unsigned N, unsigned K, const void* seeds,
                      const void* rsel, int* zref, void* counts, int* probe, void* stream) {
  if (K % 512 || rows % 32 || N % 32) return 2;
  const unsigned units = (rows / 32) * (N / 32);
  if (!units) return 0;
  hz_ref_k<<<units, 32, 0, (cudaStream_t)stream>>>((const int8_t*)A, (const int8_t*)B, N, K, (const u64*)seeds,
                                                   (const uint8_t*)rsel, zref, (unsigned long long*)counts, probe);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
