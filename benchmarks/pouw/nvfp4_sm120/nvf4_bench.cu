// Bench for mainloop_nvf4.cuh: operand generators, an independent reference (one warp per m16 x n8 block, every atom's
// fragments read from global memory), the header's GEMMs (FP32 words, and BF16 as CUTLASS's and cuBLASLt's plain NVFP4 GEMMs
// write it) and word counters, in one cubin:
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin -o nvf4_bench.cubin nvf4_bench.cu
//
// Every kernel takes one Args by value: p[] pointers, i[] integers.
#include "mainloop_nvf4.cuh"

struct Args {
  uint64_t p[16];
  int32_t i[16];
};

#define DEV __device__ __forceinline__

DEV uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// scale (r, c) of a rows x cols scale matrix (c = k / 16) in the 128 x 4 tiled layout CUTLASS and cuBLASLt read
DEV size_t sf_offset(int r, int c, int cols) {
  return ((size_t)(r / 128) * (cols / 4) + c / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + c % 4;
}

// random E2M1 codes, two a byte (every nibble is a finite E2M1 value): p0 out (u32 words), i0 words, i1 seed
extern "C" __global__ void nv_gen_codes(const Args a) {
  const size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= (size_t)(uint32_t)a.i[0]) return;
  reinterpret_cast<uint32_t*>(a.p[0])[w] = mix32((uint32_t)a.i[1] ^ mix32((uint32_t)w * 2u + 1u));
}

// random UE4M3 scale bytes, bit 7 never set and never 0x7F (NaN): p0 out (u32 words), i0 words, i1 seed, i2 range: 0 every
// code 0x00-0x7E (zero, subnormals, 2^-9 .. 448), 1 the 16 codes 0x30-0x3F (0.5 .. 1.875, as quantized weights' scales sit)
extern "C" __global__ void nv_gen_scales(const Args a) {
  const size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= (size_t)(uint32_t)a.i[0]) return;
  const uint32_t h = mix32((uint32_t)a.i[1] ^ mix32((uint32_t)w * 2u + 1u));
  uint32_t o = 0;
  for (int b = 0; b < 4; b++) {
    uint32_t c = (h >> (8 * b)) & 0xFF;
    c = a.i[2] == 1 ? 0x30 + (c & 0x0F) : (c & 0x7F) == 0x7F ? 0x7E : (c & 0x7F);
    o |= c << (8 * b);
  }
  reinterpret_cast<uint32_t*>(a.p[0])[w] = o;
}

// The reference: warp w computes the m16 x n8 block (w / (N / 8), w % (N / 8)) as the chain of K / 64 atoms from +0, each
// atom's A, B and scale registers read from global memory in the PTX fragment layout.  p0 A, p1 B, p2 SFA, p3 SFB, p4 C
// (FP32, M x N); i0 M, i1 N, i2 K
extern "C" __global__ void nv_ref_words(const Args a) {
  const int M = a.i[0], N = a.i[1], K = a.i[2], lane = threadIdx.x % 32;
  const size_t w = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int bn_count = N / 8;
  if (w >= (size_t)(M / 16) * bn_count) return;
  const int bm = (int)(w / bn_count), bn = (int)(w % bn_count), KB = K / 2, cols = K / 16, q = lane & 3;
  const uint8_t *A = reinterpret_cast<const uint8_t*>(a.p[0]), *B = reinterpret_cast<const uint8_t*>(a.p[1]);
  const uint8_t *SA = reinterpret_cast<const uint8_t*>(a.p[2]), *SB = reinterpret_cast<const uint8_t*>(a.p[3]);
  const uint8_t* a0 = A + (size_t)(16 * bm + lane / 4) * KB + 4 * q;
  const uint8_t* a1 = a0 + (size_t)8 * KB;
  const uint8_t* b0 = B + (size_t)(8 * bn + lane / 4) * KB + 4 * q;
  const int ra = 16 * bm + 8 * (lane & 1) + (lane >> 2), rb = 8 * bn + (lane >> 2);
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int s = 0; s < K / 64; s++) {
    const uint32_t fa[4] = {*reinterpret_cast<const uint32_t*>(a0 + 32 * s), *reinterpret_cast<const uint32_t*>(a1 + 32 * s),
                            *reinterpret_cast<const uint32_t*>(a0 + 32 * s + 16),
                            *reinterpret_cast<const uint32_t*>(a1 + 32 * s + 16)};
    const uint32_t fb[2] = {*reinterpret_cast<const uint32_t*>(b0 + 32 * s),
                            *reinterpret_cast<const uint32_t*>(b0 + 32 * s + 16)};
    const uint32_t sa = *reinterpret_cast<const uint32_t*>(SA + sf_offset(ra, 4 * s, cols));
    const uint32_t sb = *reinterpret_cast<const uint32_t*>(SB + sf_offset(rb, 4 * s, cols));
    nvf4::omma(d, fa, fb, sa, sb);
  }
  float* C = reinterpret_cast<float*>(a.p[4]);
  const size_t row = 16 * bm + lane / 4, col = 8 * bn + 2 * q;
  *reinterpret_cast<float2*>(C + row * N + col) = make_float2(d[0], d[1]);
  *reinterpret_cast<float2*>(C + (row + 8) * N + col) = make_float2(d[2], d[3]);
}

// FP32 words to BF16, round to nearest even (nvf4::bf16x2): p0 in, p1 out, i0 elements (even)
extern "C" __global__ void nv_to_bf16(const Args a) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (2 * i >= (size_t)(uint32_t)a.i[0]) return;
  const float2 v = reinterpret_cast<const float2*>(a.p[0])[i];
  reinterpret_cast<uint32_t*>(a.p[1])[i] = nvf4::bf16x2(v.x, v.y);
}

// how many of i0 elements of i1 bytes (4 or 2) differ between p0 and p1, added to the u64 at p2
extern "C" __global__ void nv_diff(const Args a) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  bool bad = false;
  if (i < (size_t)(uint32_t)a.i[0]) {
    if (a.i[1] == 4) bad = reinterpret_cast<const uint32_t*>(a.p[0])[i] != reinterpret_cast<const uint32_t*>(a.p[1])[i];
    else bad = reinterpret_cast<const uint16_t*>(a.p[0])[i] != reinterpret_cast<const uint16_t*>(a.p[1])[i];
  }
  const uint32_t n = __popc(__ballot_sync(0xffffffffu, bad));
  if ((threadIdx.x & 31) == 0 && n) atomicAdd(reinterpret_cast<unsigned long long*>(a.p[2]), (unsigned long long)n);
}

// C = A B^T by the mainloop, persistent over the TM x 128 tiles: p0 SFA, p1 SFB, p2 C (FP32 words, or BF16 if BF16), p10 the
// Maps; i0 M, i1 N, i2 K, i3 raster group
template <int TM, int ST, int RP, int BO, int BK, bool BF16>
DEV void nv_gemm_body(const Args& a) {
  using L = nvf4::Mainloop<TM, ST, RP, BO, BK>;
  extern __shared__ __align__(1024) uint8_t nv_smem[];
  L ml(nv_smem);
  const int M = a.i[0], N = a.i[1], K = a.i[2], group = a.i[3];
  ml.roles(reinterpret_cast<const nvf4::Maps*>(a.p[10]), reinterpret_cast<const uint8_t*>(a.p[0]),
           reinterpret_cast<const uint8_t*>(a.p[1]), M, N, K, group);
  const int tiles_m = M / TM, tiles_n = N / L::BN;
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    nvf4::raster(t, tiles_m, tiles_n, group, tm, tn);
    float acc[4][4][2][4];
    ml.tile(K, acc);
    if constexpr (BF16) ml.store_bf16(reinterpret_cast<uint16_t*>(a.p[2]), N, tm, tn, acc);
    else ml.store_f32(reinterpret_cast<float*>(a.p[2]), N, tm, tn, acc);
  }
}

#define NV_GEMM(NAME, TM, ST, RP, BO, BK, BF16)                                                                         \
  extern "C" __global__ void __launch_bounds__((nvf4::Mainloop<TM, ST, RP, BO, BK>::NT), 1) NAME(const Args a) {        \
    nv_gemm_body<TM, ST, RP, BO, BK, BF16>(a);                                                                          \
  }                                                                                                                     \
  extern "C" __global__ void NAME##_smem(int* out) { *out = nvf4::Mainloop<TM, ST, RP, BO, BK>::SMEM; }
#define NV_PAIR(NAME, TM, ST, RP, BO, BK) \
  NV_GEMM(NAME##_w, TM, ST, RP, BO, BK, false) NV_GEMM(NAME##_b, TM, ST, RP, BO, BK, true)

// nv256: 256 x 128 tiles, 128-deep stages, 3 of them, the producer at 40 registers (232 to each consumer), A-major MMA
// order; _o: B-major order (CUTLASS's); _k64: 64-deep stages, 7 of them
NV_PAIR(nv256, 256, 3, 40, 0, 128)
NV_PAIR(nv256_o, 256, 3, 40, 1, 128)
NV_PAIR(nv256_k64, 256, 7, 40, 0, 64)
NV_PAIR(nv256_k64_o, 256, 7, 40, 1, 64)
