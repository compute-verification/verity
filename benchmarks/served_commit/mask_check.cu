// M_key·y three ways on the H200, each held to the committer's formula on the host: hm96_mask (the nibble table through L1),
// mask_roll<true> (the split form's byte table in shared memory, as rows_leaf_salt_kernel runs it) and mask_probe.cu's
// mask_unrolled. Then each one's time a salt with every SM busy, beside the salts' own generation alone. Prints one JSON object.
//
//     nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -I integrations/vllm/verity_vllm/commit/committer mask_check.cu
//
// The formula (native_rows.cu, hm96_mask): mask byte j is the xor over salt bytes p < 192 of byte p + j of T_(y_p), where
// T_v = T_(v & 15) ^ T_(16 (v >> 4)) and byte k of nibble row u is Tn[256 (k >> 4) + 16 u + (k & 15)] (the high nibble's rows
// 4 KB on); mask byte 4 w + b is byte b of m[w]. Any table: the check holds the forms to one linear map, whatever the key.
#define MASK_UNROLL 1
#include "mask_probe.cu"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

__host__ __device__ inline u64 mix(u64 z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// FORM 0 hm96_mask, 1 mask_roll<true>, 2 mask_unrolled, 3 none (the salts' generation alone). CHECK: every salt's mask to out,
// else their xor a thread.
template <int FORM, bool CHECK>
__global__ void __launch_bounds__(256, 2) masks(const uint8_t* __restrict__ Tn, u32* __restrict__ out, int n, u64 seed) {
  extern __shared__ uint4 T8[];
  if constexpr (FORM == 1 || FORM == 2) load_t8(Tn, T8);
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u32 acc = 0;
#pragma unroll 1
  for (int s = 0; s < n; ++s) {
    const u64 k = t * n + s;
    u64 Y[24];
#pragma unroll
    for (int i = 0; i < 24; ++i) Y[i] = mix(seed ^ (24 * k + i));
    u32 mk[16];
    if constexpr (FORM == 0) hm96_mask(Tn, Y, mk);
    else if constexpr (FORM == 1) mask_roll<true>(T8, Y, mk);
    else if constexpr (FORM == 2) mask_unrolled(T8, Y, mk);
    else {
#pragma unroll
      for (int w = 0; w < 16; ++w) mk[w] = (u32)(Y[w] ^ Y[w + 8] >> 32);
    }
    if constexpr (CHECK) {
#pragma unroll
      for (int w = 0; w < 16; ++w) out[16 * k + w] = mk[w];
    } else {
#pragma unroll
      for (int w = 0; w < 16; ++w) acc ^= mk[w];
    }
  }
  if constexpr (!CHECK) out[t] = acc;
}

static void ref_mask(const uint8_t* Tn, const u64* Y, u32* m) {
  uint8_t mb[64] = {0};
  for (int p = 0; p < 192; ++p) {
    const int v = (int)((Y[p >> 3] >> (56 - 8 * (p & 7))) & 0xFF);
    for (int j = 0; j < 64; ++j) {
      const int k = p + j, c = 256 * (k >> 4) + (k & 15);
      mb[j] ^= Tn[c + 16 * (v & 15)] ^ Tn[4096 + c + 16 * (v >> 4)];
    }
  }
  for (int w = 0; w < 16; ++w)
    m[w] = mb[4 * w] | mb[4 * w + 1] << 8 | mb[4 * w + 2] << 16 | (u32)mb[4 * w + 3] << 24;
}

template <int FORM, bool CHECK>
static float launch(int grid, const uint8_t* Tn, u32* out, int n, u64 seed, int reps) {
  const int shm = (FORM == 1 || FORM == 2) ? T8_BYTES : 0;
  cudaFuncSetAttribute(masks<FORM, CHECK>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
  masks<FORM, CHECK><<<grid, 256, shm>>>(Tn, out, n, seed);
  cudaDeviceSynchronize();
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  float best = 1e30f;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(a);
    masks<FORM, CHECK><<<grid, 256, shm>>>(Tn, out, n, seed);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    if (ms < best) best = ms;
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return best;
}

int main() {
  int sms = 0, khz = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
  std::mt19937_64 g(20261029);
  std::vector<uint8_t> tn(8192);
  for (auto& x : tn) x = (uint8_t)g();
  uint8_t* Tn;
  cudaMalloc(&Tn, tn.size());
  cudaMemcpy(Tn, tn.data(), tn.size(), cudaMemcpyHostToDevice);

  const int cgrid = 64, cn = 4;
  const u64 seed = g(), salts = (u64)cgrid * 256 * cn;
  std::vector<u32> want(16 * salts), got(16 * salts);
  for (u64 k = 0; k < salts; ++k) {
    u64 Y[24];
    for (int i = 0; i < 24; ++i) Y[i] = mix(seed ^ (24 * k + i));
    ref_mask(tn.data(), Y, &want[16 * k]);
  }
  u32* out;
  const int tgrid = sms * 2 * 4, tn_salts = 256;
  cudaMalloc(&out, sizeof(u32) * std::max<u64>(16 * salts, (u64)tgrid * 256));
  long bad[3];
  float ms[4];
  launch<0, true>(cgrid, Tn, out, cn, seed, 1);
  cudaMemcpy(got.data(), out, 4 * got.size(), cudaMemcpyDeviceToHost);
  bad[0] = 0;
  for (u64 i = 0; i < got.size(); ++i) bad[0] += got[i] != want[i];
  launch<1, true>(cgrid, Tn, out, cn, seed, 1);
  cudaMemcpy(got.data(), out, 4 * got.size(), cudaMemcpyDeviceToHost);
  bad[1] = 0;
  for (u64 i = 0; i < got.size(); ++i) bad[1] += got[i] != want[i];
  launch<2, true>(cgrid, Tn, out, cn, seed, 1);
  cudaMemcpy(got.data(), out, 4 * got.size(), cudaMemcpyDeviceToHost);
  bad[2] = 0;
  for (u64 i = 0; i < got.size(); ++i) bad[2] += got[i] != want[i];
  ms[0] = launch<0, false>(tgrid, Tn, out, tn_salts, seed, 5);
  ms[1] = launch<1, false>(tgrid, Tn, out, tn_salts, seed, 5);
  ms[2] = launch<2, false>(tgrid, Tn, out, tn_salts, seed, 5);
  ms[3] = launch<3, false>(tgrid, Tn, out, tn_salts, seed, 5);
  const cudaError_t e = cudaGetLastError();
  const double tsalts = (double)tgrid * 256 * tn_salts;
  const char* names[4] = {"hm96_mask", "mask_roll_byte", "mask_unrolled", "salts_alone"};
  printf("{\"sms\": %d, \"max_clock_khz\": %d, \"cuda_error\": \"%s\", \"checked_salts\": %llu, \"timed_salts\": %.0f, \"forms\": {\n",
         sms, khz, cudaGetErrorString(e), (unsigned long long)salts, tsalts);
  for (int f = 0; f < 4; ++f) {
    const double sm_cycles = ms[f] * 1e-3 * khz * 1e3 * sms / tsalts;
    printf("  \"%s\": {\"mismatched_words\": %ld, \"best_ms\": %.4f, \"ns_a_salt_gpu\": %.5f, \"sm_cycles_a_salt\": %.2f, "
           "\"mac_times_a_salt\": %.0f}%s\n",
           names[f], f < 3 ? bad[f] : 0L, ms[f], ms[f] * 1e6 / tsalts, sm_cycles, sm_cycles * 4096, f < 3 ? "," : "");
  }
  printf("}}\n");
  return e == cudaSuccess && !bad[0] && !bad[1] && !bad[2] ? 0 : 1;
}
