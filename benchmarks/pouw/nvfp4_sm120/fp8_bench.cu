// Bench for mainloop_nvf4.cuh's FP8 mode (the plain E4M3 GEMM): an operand generator, an independent reference (one warp
// per m16 x n8 block, every atom's fragments read from global memory), the header's GEMMs (FP32 words, and BF16 as
// CUTLASS's and cuBLASLt's plain FP8 GEMMs write it) and word counters, in one cubin:
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin -o fp8_bench.cubin fp8_bench.cu
//
// Every kernel takes one Args by value: p[] pointers, i[] integers.  The helpers and the reference take nvf4_bench.cu's
// names and arguments, so that nvf4_bench.py's host side drives them (fp8_bench.py).
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

// random E4M3 codes, never a NaN (0x7F, 0xFF): p0 out (u32 words), i0 words, i1 seed, i2 range: 0 every other code (zero,
// subnormals, 2^-6 .. 448), 1 exponent fields 4 .. 10 (2^-3 .. 15, either sign)
extern "C" __global__ void f8_gen_codes(const Args a) {
  const size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= (size_t)(uint32_t)a.i[0]) return;
  const uint32_t h = mix32((uint32_t)a.i[1] ^ mix32((uint32_t)w * 2u + 1u));
  uint32_t o = 0;
  for (int b = 0; b < 4; b++) {
    uint32_t c = (h >> (8 * b)) & 0xFF;
    if (a.i[2] == 1) c = (c & 0x87) | ((4 + ((c >> 3) & 0xF) % 7) << 3);
    else if ((c & 0x7F) == 0x7F) c ^= 1;
    o |= c << (8 * b);
  }
  reinterpret_cast<uint32_t*>(a.p[0])[w] = o;
}

// The reference: warp w computes the m16 x n8 block (w / (N / 8), w % (N / 8)) as the chain of K / 32 atoms from +0, each
// atom's A and B registers read from global memory in the PTX fragment layout.  p0 A, p1 B, p4 C (FP32, M x N); i0 M,
// i1 N, i2 K (p2, p3 unused: nvf4_bench.py's scale buffers)
extern "C" __global__ void nv_ref_words(const Args a) {
  const int M = a.i[0], N = a.i[1], K = a.i[2], lane = threadIdx.x % 32;
  const size_t w = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int bn_count = N / 8;
  if (w >= (size_t)(M / 16) * bn_count) return;
  const int bm = (int)(w / bn_count), bn = (int)(w % bn_count), q = lane & 3;
  const uint8_t *A = reinterpret_cast<const uint8_t*>(a.p[0]), *B = reinterpret_cast<const uint8_t*>(a.p[1]);
  const uint8_t* a0 = A + (size_t)(16 * bm + lane / 4) * K + 4 * q;
  const uint8_t* a1 = a0 + (size_t)8 * K;
  const uint8_t* b0 = B + (size_t)(8 * bn + lane / 4) * K + 4 * q;
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int s = 0; s < K / 32; s++) {
    const uint32_t fa[4] = {*reinterpret_cast<const uint32_t*>(a0 + 32 * s), *reinterpret_cast<const uint32_t*>(a1 + 32 * s),
                            *reinterpret_cast<const uint32_t*>(a0 + 32 * s + 16),
                            *reinterpret_cast<const uint32_t*>(a1 + 32 * s + 16)};
    const uint32_t fb[2] = {*reinterpret_cast<const uint32_t*>(b0 + 32 * s),
                            *reinterpret_cast<const uint32_t*>(b0 + 32 * s + 16)};
    nvf4::qmma(d, fa, fb);
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

DEV uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// C = A B^T by the mainloop, persistent over the TM x 128 tiles: p2 C (FP32 words, or BF16 if BF16), p10 the Maps; i0 M,
// i1 N, i2 K, i3 raster group.  STAMP: as nvf4_bench.cu's (p3)
constexpr int F8_STAMP_TILES = 16;
template <int TM, int ST, int RP, int BO, int LATE, int WIDE, bool BF16, bool STAMP>
DEV void f8_gemm_body(const Args& a) {
  using L = nvf4::Mainloop<TM, ST, RP, BO, LATE, WIDE, 1>;
  extern __shared__ __align__(1024) uint8_t f8_smem[];
  L ml(f8_smem);
  const int M = a.i[0], N = a.i[1], K = a.i[2], group = a.i[3];
  ml.roles(reinterpret_cast<const nvf4::Maps*>(a.p[10]), nullptr, nullptr, M, N, K, group);
  const int tiles_m = M / TM, tiles_n = N / L::BN;
  uint64_t* stamp = STAMP ? reinterpret_cast<uint64_t*>(a.p[3]) + (size_t)(blockIdx.x * L::NCW + threadIdx.x / 32) * F8_STAMP_TILES * 4
                          : nullptr;
  const bool stamping = STAMP && threadIdx.x % 32 == 0;
  for (int t = blockIdx.x, j = 0; t < tiles_m * tiles_n; t += gridDim.x, j++) {
    int tm, tn;
    nvf4::raster(t, tiles_m, tiles_n, group, tm, tn);
    float acc[4][4][2][4];
    const bool st = stamping && j < F8_STAMP_TILES;
    if (st) stamp[4 * j] = clock64();
    ml.tile(K, acc, t + (int)gridDim.x < tiles_m * tiles_n);
    if (st) stamp[4 * j + 1] = clock64();
    if constexpr (BF16) ml.store_bf16(reinterpret_cast<uint16_t*>(a.p[2]), N, tm, tn, acc);
    else ml.store_f32(reinterpret_cast<float*>(a.p[2]), N, tm, tn, acc);
    if (st) {
      stamp[4 * j + 2] = clock64();
      stamp[4 * j + 3] = globaltimer();
    }
  }
}

#define F8_GEMM(NAME, TM, ST, RP, BO, LATE, WIDE, BF16, STAMP)                                                        \
  extern "C" __global__ void __launch_bounds__((nvf4::Mainloop<TM, ST, RP, BO, LATE, WIDE, 1>::NT), 1) NAME(const Args a) { \
    f8_gemm_body<TM, ST, RP, BO, LATE, WIDE, BF16, STAMP>(a);                                                              \
  }                                                                                                                        \
  extern "C" __global__ void NAME##_smem(int* out) { *out = nvf4::Mainloop<TM, ST, RP, BO, LATE, WIDE, 1>::SMEM; }
// NAME_w (FP32 words), NAME_b (BF16) and NAME_t (NAME_b with STAMP)
#define F8_PAIR(NAME, TM, ST, RP, BO, LATE, WIDE)                                                                          \
  F8_GEMM(NAME##_w, TM, ST, RP, BO, LATE, WIDE, false, false)                                                              \
  F8_GEMM(NAME##_b, TM, ST, RP, BO, LATE, WIDE, true, false) F8_GEMM(NAME##_t, TM, ST, RP, BO, LATE, WIDE, true, true)

// f8256: 256 x 128 tiles, 3 stages, the producer at 40 registers, A-major MMA order; _o B-major, _e LATE, _ew LATE and WIDE
// (nvf4_bench.cu's nv256 variants)
F8_PAIR(f8256, 256, 3, 40, 0, 0, 0)
F8_PAIR(f8256_o_e, 256, 3, 40, 1, 1, 0)
F8_PAIR(f8256_ew, 256, 3, 40, 0, 1, 1)
F8_PAIR(f8256_o_ew, 256, 3, 40, 1, 1, 1)
