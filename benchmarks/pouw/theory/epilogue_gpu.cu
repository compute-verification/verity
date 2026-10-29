// The stock int8 epilogue against exact_sweep.py's de-noising folds, on the device, bit for bit.
//
// Stock converts an int32 accumulator P with I2F, multiplies by the column's scale s_j, then the row's s_i, and packs
// bf16. The magic fold takes D = P + r_i + c_j and replaces I2F by IADD3(D, 0x4B400000 − r_i, −c_j) read as fp32 and
// the column FMUL by FFMA(v, s_j, −1.5·2²³·s_j); the column fold takes D = P + c_j through I2F and FFMA(v, s_j,
// −c_j·s_j). epi_stock, epi_magic and epi_ffma are the three as kernels (one output a thread) for the SASS diff;
// sweep() generates every operand on the device from a counter hash, runs all three, and counts outputs whose fp32
// value before the bf16 pack differs from stock's (equal fp32 implies equal bf16; bf16 alone would hide most errors):
//
//   magic   P uniform on [−2²², 2²²], r_i on [0, 2²⁰), c_j ∈ {0, 1}, s_j with 22-bit significands
//   ffma    P uniform on [−2²⁴, 2²⁴) (|P + c_j| ≤ 2²⁴), the same scales
//   raw     the magic fold with full 24-bit s_j (1.5·2²³·s_j not always exact): mismatches expected; also in bf16
//   edge    P = 2²² + 1 or −2²² − 1 through the magic and odd P past 2²⁵ through the column fold: mismatches expected
//
// Prints one JSON object. Build: nvcc -O3 -arch=sm_90a -o epilogue_gpu epilogue_gpu.cu; run: ./epilogue_gpu [BATCHES]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(2); } } while (0)

constexpr int MAGIC_BITS = 0x4B400000;
constexpr float MAGIC = 12582912.0f;

__device__ __forceinline__ unsigned short bf16_bits(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

extern "C" __global__ void epi_stock(const int* __restrict__ p, const float* __restrict__ sc,
                                     const float* __restrict__ sr, unsigned short* __restrict__ out, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
  if (j >= n) return;
  float v = __int2float_rn(p[i * n + j]);
  out[i * n + j] = bf16_bits(__fmul_rn(__fmul_rn(v, sc[j]), sr[i]));
}

extern "C" __global__ void epi_magic(const int* __restrict__ d, const int* __restrict__ row_term,
                                     const int* __restrict__ col_term, const float* __restrict__ sc,
                                     const float* __restrict__ neg_magic_sc, const float* __restrict__ sr,
                                     unsigned short* __restrict__ out, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
  if (j >= n) return;
  float v = __int_as_float(d[i * n + j] + row_term[i] + col_term[j]);
  out[i * n + j] = bf16_bits(__fmul_rn(__fmaf_rn(v, sc[j], neg_magic_sc[j]), sr[i]));
}

extern "C" __global__ void epi_ffma(const int* __restrict__ d, const float* __restrict__ sc,
                                    const float* __restrict__ neg_c_sc, const float* __restrict__ sr,
                                    unsigned short* __restrict__ out, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
  if (j >= n) return;
  float v = __int2float_rn(d[i * n + j]);
  out[i * n + j] = bf16_bits(__fmul_rn(__fmaf_rn(v, sc[j], neg_c_sc[j]), sr[i]));
}

__device__ __forceinline__ uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__device__ __forceinline__ float stock(int p, float sc, float sr) {
  return __fmul_rn(__fmul_rn(__int2float_rn(p), sc), sr);
}

__device__ __forceinline__ float magic(int d, int r, int c, float sc, float sr) {
  float v = __int_as_float(d + (MAGIC_BITS - r) - c);
  return __fmul_rn(__fmaf_rn(v, sc, __fmul_rn(-MAGIC, sc)), sr);
}

__device__ __forceinline__ float ffma(int d, int c, float sc, float sr) {
  return __fmul_rn(__fmaf_rn(__int2float_rn(d), sc, -(float)c * sc), sr);
}

__device__ __forceinline__ unsigned differs(float a, float b) { return __float_as_uint(a) != __float_as_uint(b); }

// counts: 0 magic, 1 ffma, 2 raw, 3 raw in bf16, 4 edge-magic, 5 edge-ffma (fp32 unless named), 6 raw max |Δ| in
// bf16 codes
__global__ void sweep(uint64_t seed, int m, int n, unsigned long long* counts) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
  if (j >= n) return;
  uint64_t hr = mix(seed ^ (0x100000000ull + i)), hc = mix(seed ^ (0x200000000ull + j));
  uint64_t h = mix(seed ^ ((uint64_t)i << 32 | (uint32_t)j) ^ 0x5bd1e995ull);
  int r = (int)(hr & 0xFFFFF), c = (int)(hc >> 40 & 1);
  unsigned exp_c = 113 + (unsigned)(hc % 7), exp_r = 117 + (unsigned)(hr >> 20) % 10;
  float sc = __uint_as_float(exp_c << 23 | (unsigned)(hc >> 8) & 0x7FFFFC);
  float sc_raw = __uint_as_float(exp_c << 23 | (unsigned)(hc >> 8) & 0x7FFFFF);
  float sr = __uint_as_float(exp_r << 23 | (unsigned)(hr >> 24) & 0x7FFFFF);
  int p = (int)(h % ((1u << 23) + 1)) - (1 << 22);
  int p24 = (int)(h >> 23 & 0x1FFFFFF) - (1 << 24);
  unsigned bad_magic = differs(magic(p + r + c, r, c, sc, sr), stock(p, sc, sr));
  unsigned bad_ffma = differs(ffma(p24 + c, c, sc, sr), stock(p24, sc, sr));
  float raw_m = magic(p + r + c, r, c, sc_raw, sr), raw_s = stock(p, sc_raw, sr);
  unsigned short a = bf16_bits(raw_m), b = bf16_bits(raw_s);
  unsigned bad_raw = differs(raw_m, raw_s), bad_raw_bf16 = a != b;
  int edge = (h >> 63) ? (1 << 22) + 1 : -(1 << 22) - 1;
  unsigned bad_edge_m = differs(magic(edge + r + c, r, c, sc, sr), stock(edge, sc, sr));
  int big = (1 << 25) + 1 + 2 * (int)(h >> 40 & 0x3FFFFF);
  unsigned bad_edge_f = differs(ffma(big + 1, 1, sc, sr), stock(big, sc, sr));
  unsigned long long codes = bad_raw_bf16 ? (unsigned long long)abs((int)a - (int)b) : 0ull;
  unsigned mask = __activemask();
  unsigned vals[6] = {bad_magic, bad_ffma, bad_raw, bad_raw_bf16, bad_edge_m, bad_edge_f};
  for (int q = 0; q < 6; q++) {
    unsigned cnt = __popc(__ballot_sync(mask, vals[q]));
    if ((threadIdx.x & 31) == __ffs(mask) - 1 && cnt) atomicAdd(&counts[q], (unsigned long long)cnt);
  }
  if (codes) atomicMax(&counts[6], codes);
}

int main(int argc, char** argv) {
  int batches = argc > 1 ? atoi(argv[1]) : 16, m = 8192, n = 8192;
  unsigned long long* dc;
  unsigned long long h[7] = {0};
  CK(cudaMalloc(&dc, sizeof h));
  CK(cudaMemset(dc, 0, sizeof h));
  dim3 blk(256), grd((n + 255) / 256, m);
  for (int b = 0; b < batches; b++) sweep<<<grd, blk>>>(0xC0FFEEull * (b + 1), m, n, dc);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, dc, sizeof h, cudaMemcpyDeviceToHost));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  double total = (double)batches * m * n;
  printf("{\"device\": \"%s\", \"sm\": \"%d%d\", \"outputs_per_case\": %.0f, \"magic_fp32_mismatches\": %llu, "
         "\"ffma_fp32_mismatches\": %llu, \"raw_scale_fp32_mismatches\": %llu, \"raw_scale_bf16_mismatches\": %llu, "
         "\"raw_scale_max_bf16_code_delta\": %llu, \"edge_magic_fp32_mismatches\": %llu, "
         "\"edge_ffma_fp32_mismatches\": %llu}\n",
         prop.name, prop.major, prop.minor, total, h[0], h[1], h[2], h[3], h[6], h[4], h[5]);
  return h[0] || h[1] || !h[2] || !h[4] || !h[5] ? 1 : 0;
}
