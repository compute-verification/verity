// NOBITS probe: does an NVFP4/MXFP4 (kind::mxf4nvf4) MMA leave the committed s32 word's bits in a register
// for free on sm_120?  The fold's round-6 bound puts the FP4 route at >= 1.25 of
// honest under the premise NOBITS: "no MMA leaves the committed word's bits in a register without an op on the word".  The
// FP4 MMA accumulates in FP32; the honest int8 IMMA accumulates in s32 and its accumulator IS the committed word.
//
// This probe settles it directly.  One `mma.sync kind::mxf4nvf4` on integer-valued E2M1 inputs, with the accumulator C set
// to (1) 0 and (2) the binade constant 1.5*2^23 = 0x4B400000 (the trick NOBITS rules out).  For integer T within range it
// checks the FP32 accumulator holds the exact integer (unlike Hopper's FP8, reported to keep ~14 bits), then asks whether
// the accumulator's 32-bit pattern IS the two's-complement s32 word T.  It is not: plain FP32 holds IEEE754(T), and the
// binade form holds 0x4B400000 + T (low bits are T, top bits are the 0x4B4 binade).  So an op is needed on every word; the
// cheapest exact FP32->s32 conversions (one F2I, or one IADD3 that strips the binade) are compiled and SASS-counted.  The
// int8 IMMA runs the same integer products and its s32 accumulator equals T with no op, the honest baseline.
//
// The NVFP4/MXFP4 instruction and e2m1 fragment packing are copied from benchmarks/tc_probe_fp4/mma_fp4.cu (lane fp4-5090).
// Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -o nobits nobits.cu -lnvidia-ml

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <cstdlib>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s @ %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 3; } } while (0)

namespace {

inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// e2m1 magnitudes by code 0..7 (sign bit 0x8).  The integer-valued codes are 2(=1),4(=2),5(=3),6(=4),7(=6).
__host__ __device__ inline float e2m1_decode(uint8_t c) {
  static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = mag[c & 7];
  return (c & 8) ? -v : v;
}

constexpr uint32_t MAGIC = 0x4B400000u;  // 1.5 * 2^23 = 12582912.0f ; binade [2^23, 2^24) has ULP 1
constexpr uint8_t UE4M3_ONE = 0x38;      // UE4M3 (NVFP4) encoding of 1.0 (exp 0111, mantissa 000)
constexpr uint8_t UE8M0_ONE = 0x7F;      // UE8M0 (MXFP4) encoding of 1.0 (exponent 127, 2^(b-127))

// ---- NVFP4 / MXFP4 kind::mxf4nvf4 m16n8k64 (copied from mma_fp4.cu, lane fp4-5090) ----
__device__ __forceinline__ void mma_nvfp4(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4],
                                          uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {0, 0}, {%15}, {0, 0};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ void mma_mxfp4(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4],
                                          uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {0, 0}, {%15}, {0, 0};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa), "r"(sfb));
}

// int8 IMMA m16n8k32 (s32 accumulator): the honest baseline whose accumulator IS the committed word.
__device__ __forceinline__ void mma_s8(uint32_t d[4], const uint32_t a[4], const uint32_t b[2], const uint32_t c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__device__ __forceinline__ void pack_a_e2m1(uint32_t a[4], const uint8_t* A, int lane) {
  const int row0 = lane >> 2, kcol = 8 * (lane & 3);
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const int row = row0 + 8 * (r & 1), k0 = kcol + 32 * (r >> 1);
    uint32_t v = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) v |= (uint32_t)(A[row * 64 + k0 + j] & 0xF) << (4 * j);
    a[r] = v;
  }
}

__device__ __forceinline__ void pack_b_e2m1(uint32_t b[2], const uint8_t* Bt, int lane) {
  const int col = lane >> 2, kcol = 8 * (lane & 3);
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int k0 = kcol + 32 * r;
    uint32_t v = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) v |= (uint32_t)(Bt[col * 64 + k0 + j] & 0xF) << (4 * j);
    b[r] = v;
  }
}

__device__ __forceinline__ uint32_t pack_sfa(const uint8_t* SFA, int nblk, int lane) {
  const int row = 8 * (lane & 1) + (lane >> 2);
  uint32_t v = 0;
  for (int b = 0; b < nblk; b++) v |= (uint32_t)SFA[row * nblk + b] << (8 * b);
  return v;
}

__device__ __forceinline__ uint32_t pack_sfb(const uint8_t* SFB, int nblk, int lane) {
  const int col = lane >> 2;
  uint32_t v = 0;
  for (int b = 0; b < nblk; b++) v |= (uint32_t)SFB[col * nblk + b] << (8 * b);
  return v;
}

// int8 A/B pack (m16n8k32): reg r byte j -> row lane/4 + 8*(r&1), k = 4*(lane%4) + j + 16*(r>>1); B col = lane/4.
__device__ __forceinline__ void pack_a_s8(uint32_t a[4], const int8_t* A, int lane) {
  const int row0 = lane >> 2, kcol = 4 * (lane & 3);
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const int row = row0 + 8 * (r & 1), k0 = kcol + 16 * (r >> 1);
    uint32_t v = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) v |= (uint32_t)(uint8_t)A[row * 32 + k0 + j] << (8 * j);
    a[r] = v;
  }
}

__device__ __forceinline__ void pack_b_s8(uint32_t b[2], const int8_t* Bt, int lane) {
  const int col = lane >> 2, kcol = 4 * (lane & 3);
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int k0 = kcol + 16 * r;
    uint32_t v = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) v |= (uint32_t)(uint8_t)Bt[col * 32 + k0 + j] << (8 * j);
    b[r] = v;
  }
}

__device__ __forceinline__ void load_c(float cf[4], const uint32_t* c, int lane) {
  const int g = lane >> 2, t = lane & 3;
  cf[0] = __uint_as_float(c[g * 8 + 2 * t]);
  cf[1] = __uint_as_float(c[g * 8 + 2 * t + 1]);
  cf[2] = __uint_as_float(c[(g + 8) * 8 + 2 * t]);
  cf[3] = __uint_as_float(c[(g + 8) * 8 + 2 * t + 1]);
}

__device__ __forceinline__ void store_d(uint32_t* d, const float df[4], int lane) {
  const int g = lane >> 2, t = lane & 3;
  d[g * 8 + 2 * t] = __float_as_uint(df[0]);
  d[g * 8 + 2 * t + 1] = __float_as_uint(df[1]);
  d[(g + 8) * 8 + 2 * t] = __float_as_uint(df[2]);
  d[(g + 8) * 8 + 2 * t + 1] = __float_as_uint(df[3]);
}

__global__ void nvfp4_kernel(const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB,
                             const uint32_t* C, uint32_t* D, int ntiles, int mxf4) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= ntiles) return;
  const int lane = threadIdx.x & 31;
  uint32_t a[4], b[2];
  pack_a_e2m1(a, A + (size_t)warp * 16 * 64, lane);
  pack_b_e2m1(b, Bt + (size_t)warp * 8 * 64, lane);
  float cf[4], df[4];
  load_c(cf, C + (size_t)warp * 128, lane);
  if (mxf4) {
    const uint32_t sfa = pack_sfa(SFA + (size_t)warp * 16 * 2, 2, lane);
    const uint32_t sfb = pack_sfb(SFB + (size_t)warp * 8 * 2, 2, lane);
    mma_mxfp4(df, a, b, cf, sfa, sfb);
  } else {
    const uint32_t sfa = pack_sfa(SFA + (size_t)warp * 16 * 4, 4, lane);
    const uint32_t sfb = pack_sfb(SFB + (size_t)warp * 8 * 4, 4, lane);
    mma_nvfp4(df, a, b, cf, sfa, sfb);
  }
  store_d(D + (size_t)warp * 128, df, lane);
}

__global__ void s8_kernel(const int8_t* A, const int8_t* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= ntiles) return;
  const int lane = threadIdx.x & 31;
  uint32_t a[4], b[2], cf[4], df[4];
  pack_a_s8(a, A + (size_t)warp * 16 * 32, lane);
  pack_b_s8(b, Bt + (size_t)warp * 8 * 32, lane);
  const int g = lane >> 2, t = lane & 3;
  cf[0] = C[g * 8 + 2 * t]; cf[1] = C[g * 8 + 2 * t + 1];
  cf[2] = C[(g + 8) * 8 + 2 * t]; cf[3] = C[(g + 8) * 8 + 2 * t + 1];
  mma_s8(df, a, b, cf);
  uint32_t* Dw = D + (size_t)warp * 128;
  Dw[g * 8 + 2 * t] = df[0]; Dw[g * 8 + 2 * t + 1] = df[1];
  Dw[(g + 8) * 8 + 2 * t] = df[2]; Dw[(g + 8) * 8 + 2 * t + 1] = df[3];
}

// Cheapest exact FP32->s32 conversions, one per thread, for the SASS count.
__global__ void conv_f2i(const float* __restrict__ in, int32_t* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2int_rz(in[i]);  // -> F2I.S32.F32.TRUNC
}

__global__ void conv_binade(const uint32_t* __restrict__ in, int32_t* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (int32_t)(in[i] - MAGIC);  // -> one IADD3 with the -0x4b400000 immediate
}

}  // namespace

int main(int argc, char** argv) {
  int ntiles = argc > 1 ? atoi(argv[1]) : 64;
  uint64_t seed = argc > 2 ? strtoull(argv[2], nullptr, 0) : 0xB175C0DEULL;
  if (ntiles < 1) ntiles = 1;

  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("{\"event\":\"start\",\"gpu\":\"%s\",\"cc\":\"%d.%d\",\"ntiles\":%d,\"magic\":\"0x%08X\"}\n",
         prop.name, prop.major, prop.minor, ntiles, MAGIC);

  const int T = ntiles;
  std::vector<uint8_t> A((size_t)T * 16 * 64, 0), Bt((size_t)T * 8 * 64, 0);
  std::vector<int8_t> A8((size_t)T * 16 * 32, 0), Bt8((size_t)T * 8 * 32, 0);
  std::vector<uint8_t> SFA((size_t)T * 16 * 4, UE4M3_ONE), SFB((size_t)T * 8 * 4, UE4M3_ONE);
  // integer-valued e2m1 codes and their int8 values
  const uint8_t icode[5] = {2, 4, 5, 6, 7};  // 1,2,3,4,6
  const int8_t ival[5] = {1, 2, 3, 4, 6};
  auto rnd = [&](uint64_t& s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
  uint64_t s = seed ? seed : 1;
  // For the int8 contrast the k-range differs (32 vs 64), so build both from one logical value stream per (tile,row,col).
  // e2m1 fills 64 k-cols; int8 fills 32 k-cols with the first 32 logical values (same products are compared per kind, not
  // across kinds).  Each code carries a sign.
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < 16; i++)
      for (int k = 0; k < 64; k++) {
        uint64_t r = rnd(s); int idx = r % 5; uint8_t sgn = (r >> 8) & 1;
        A[((size_t)t * 16 + i) * 64 + k] = icode[idx] | (sgn ? 8 : 0);
        if (k < 32) A8[((size_t)t * 16 + i) * 32 + k] = sgn ? -ival[idx] : ival[idx];
      }
    for (int j = 0; j < 8; j++)
      for (int k = 0; k < 64; k++) {
        uint64_t r = rnd(s); int idx = r % 5; uint8_t sgn = (r >> 8) & 1;
        Bt[((size_t)t * 8 + j) * 64 + k] = icode[idx] | (sgn ? 8 : 0);
        if (k < 32) Bt8[((size_t)t * 8 + j) * 32 + k] = sgn ? -ival[idx] : ival[idx];
      }
  }
  // CPU int64 reference per kind.
  std::vector<int64_t> refNV((size_t)T * 128, 0), refS8((size_t)T * 128, 0);
  for (int t = 0; t < T; t++)
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 8; j++) {
        int64_t acc = 0;
        for (int k = 0; k < 64; k++)
          acc += (int64_t)llroundf(e2m1_decode(A[((size_t)t * 16 + i) * 64 + k])) *
                 (int64_t)llroundf(e2m1_decode(Bt[((size_t)t * 8 + j) * 64 + k]));
        refNV[((size_t)t * 16 + i) * 8 + j] = acc;
        int64_t a8 = 0;
        for (int k = 0; k < 32; k++)
          a8 += (int64_t)A8[((size_t)t * 16 + i) * 32 + k] * (int64_t)Bt8[((size_t)t * 8 + j) * 32 + k];
        refS8[((size_t)t * 16 + i) * 8 + j] = a8;
      }

  const size_t cbytes = (size_t)T * 128 * 4;
  uint8_t *dA, *dBt, *dSFA, *dSFB; int8_t *dA8, *dBt8; uint32_t *dC, *dD;
  CK(cudaMalloc(&dA, A.size())); CK(cudaMalloc(&dBt, Bt.size()));
  CK(cudaMalloc(&dA8, A8.size())); CK(cudaMalloc(&dBt8, Bt8.size()));
  CK(cudaMalloc(&dSFA, SFA.size())); CK(cudaMalloc(&dSFB, SFB.size()));
  CK(cudaMalloc(&dC, cbytes)); CK(cudaMalloc(&dD, cbytes));
  CK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBt, Bt.data(), Bt.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dA8, A8.data(), A8.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBt8, Bt8.data(), Bt8.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFA, SFA.data(), SFA.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFB, SFB.data(), SFB.size(), cudaMemcpyHostToDevice));
  const int wpb = 4, blocks = (T + wpb - 1) / wpb;
  std::vector<uint32_t> D((size_t)T * 128), C0((size_t)T * 128, 0), Cm((size_t)T * 128, MAGIC);

  // Run and analyze one kind / accumulator variant.  scale_byte is the format's encoding of 1.0 (UE4M3 for nvfp4, UE8M0
  // for mxfp4), so every block scale is exactly 1 and each product is the integer e2m1 * e2m1.
  auto run_nvfp4 = [&](const char* name, int mxf4, const std::vector<uint32_t>& Cinit, uint32_t cbias, uint8_t scale_byte) -> int {
    CK(cudaMemset(dSFA, scale_byte, SFA.size()));
    CK(cudaMemset(dSFB, scale_byte, SFB.size()));
    CK(cudaMemcpy(dC, Cinit.data(), cbytes, cudaMemcpyHostToDevice));
    CK(cudaMemset(dD, 0xA5, cbytes));
    nvfp4_kernel<<<blocks, 32 * wpb>>>(dA, dBt, dSFA, dSFB, dC, dD, T, mxf4);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(D.data(), dD, cbytes, cudaMemcpyDeviceToHost));
    long long acc_exact = 0, bits_eq_s32 = 0, n = 0, in_range = 0;
    uint32_t ex0 = 0, got0 = 0; int64_t t0 = 0; bool have_ex = false;
    for (size_t e = 0; e < (size_t)T * 128; e++) {
      int64_t Tv = refNV[e]; uint32_t got = D[e]; n++;
      bool rng = (cbias == MAGIC) ? (Tv > -(1 << 22) && Tv < (1 << 22)) : (Tv >= -(1 << 24) && Tv <= (1 << 24));
      if (!rng) continue; in_range++;
      // expected FP32 bit pattern of (bias_value + T)
      float want_f = (cbias == MAGIC) ? (u2f(MAGIC) + (float)Tv) : (float)Tv;
      uint32_t want = f2u(want_f);
      if (got == want) acc_exact++;
      uint32_t s32bits = (uint32_t)(int32_t)Tv;
      if (got == s32bits) bits_eq_s32++;
      if (!have_ex && Tv != 0) { ex0 = want; got0 = got; t0 = Tv; have_ex = true; }
    }
    printf("{\"event\":\"nvfp4\",\"kind\":\"%s\",\"acc\":\"%s\",\"n\":%lld,\"in_range\":%lld,"
           "\"accum_bits_exact\":%lld,\"bits_equal_s32_word\":%lld,"
           "\"example_T\":%lld,\"example_acc_bits\":\"0x%08X\",\"example_expected\":\"0x%08X\","
           "\"example_s32_bits\":\"0x%08X\"}\n",
           name, cbias == MAGIC ? "binade_1.5*2^23" : "zero", n, in_range, acc_exact, bits_eq_s32,
           (long long)t0, got0, ex0, (uint32_t)(int32_t)t0);
    return (acc_exact == in_range) ? 0 : 1;
  };

  int bad = 0;
  bad |= run_nvfp4("nvfp4", 0, C0, 0, UE4M3_ONE);
  bad |= run_nvfp4("nvfp4", 0, Cm, MAGIC, UE4M3_ONE);
  bad |= run_nvfp4("mxfp4", 1, C0, 0, UE8M0_ONE);
  bad |= run_nvfp4("mxfp4", 1, Cm, MAGIC, UE8M0_ONE);

  // int8 IMMA: accumulator s32, C = 0, so D's bits ARE the committed s32 word with no op.
  CK(cudaMemcpy(dC, C0.data(), cbytes, cudaMemcpyHostToDevice));
  CK(cudaMemset(dD, 0xA5, cbytes));
  s8_kernel<<<blocks, 32 * wpb>>>(dA8, dBt8, dC, dD, T);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(D.data(), dD, cbytes, cudaMemcpyDeviceToHost));
  {
    long long bits_eq_s32 = 0, n = 0; int64_t t0 = 0; uint32_t got0 = 0; bool have = false;
    for (size_t e = 0; e < (size_t)T * 128; e++) {
      int64_t Tv = refS8[e]; uint32_t got = D[e]; n++;
      if (got == (uint32_t)(int32_t)Tv) bits_eq_s32++;
      if (!have && Tv != 0) { t0 = Tv; got0 = got; have = true; }
    }
    printf("{\"event\":\"int8\",\"kind\":\"s8\",\"acc\":\"s32\",\"n\":%lld,\"bits_equal_s32_word\":%lld,"
           "\"example_T\":%lld,\"example_acc_bits\":\"0x%08X\",\"note\":\"accumulator IS the committed word, 0 ops\"}\n",
           n, bits_eq_s32, (long long)t0, got0);
    if (bits_eq_s32 != n) bad |= 1;
  }

  // Confirm both cheapest conversions recover T exactly, then leave the binary for the SASS count.
  const int ne = T * 128;
  int32_t* dOut; CK(cudaMalloc(&dOut, (size_t)ne * 4));
  std::vector<int32_t> outB(ne), outF(ne);
  CK(cudaMemset(dSFA, UE4M3_ONE, SFA.size()));  // nvfp4 scales back to 1.0
  CK(cudaMemset(dSFB, UE4M3_ONE, SFB.size()));
  // binade accumulator (C = magic): one IADD3 strips 0x4B400000.
  CK(cudaMemcpy(dC, Cm.data(), cbytes, cudaMemcpyHostToDevice));
  CK(cudaMemset(dD, 0xA5, cbytes));
  nvfp4_kernel<<<blocks, 32 * wpb>>>(dA, dBt, dSFA, dSFB, dC, dD, T, 0);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  conv_binade<<<(ne + 255) / 256, 256>>>(dD, dOut, ne);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(outB.data(), dOut, (size_t)ne * 4, cudaMemcpyDeviceToHost));
  // plain accumulator (C = 0): one F2I truncates the FP32 to s32.
  CK(cudaMemcpy(dC, C0.data(), cbytes, cudaMemcpyHostToDevice));
  CK(cudaMemset(dD, 0xA5, cbytes));
  nvfp4_kernel<<<blocks, 32 * wpb>>>(dA, dBt, dSFA, dSFB, dC, dD, T, 0);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  conv_f2i<<<(ne + 255) / 256, 256>>>((const float*)dD, dOut, ne);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(outF.data(), dOut, (size_t)ne * 4, cudaMemcpyDeviceToHost));
  long long okB = 0, okF = 0;
  for (int e = 0; e < ne; e++) {
    if ((int64_t)outB[e] == refNV[e]) okB++;
    if ((int64_t)outF[e] == refNV[e]) okF++;
  }
  printf("{\"event\":\"conversions\",\"n\":%d,\"binade_iadd_exact\":%lld,\"f2i_exact\":%lld}\n", ne, okB, okF);

  printf("{\"event\":\"verdict\",\"bad\":%d}\n", bad);
  return bad ? 5 : 0;
}
