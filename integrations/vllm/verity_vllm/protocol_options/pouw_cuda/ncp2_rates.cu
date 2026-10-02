// Each hash's in-register rate on this GPU at a sweep of concurrencies, as JSON lines: from one thread (a leaf's latency: one
// serial chain of compressions or permutations) through the leaf counts one call has at 16 rows, to every lane busy.
//   nvcc -O3 -std=c++17 -fmad=false -gencode=arch=compute_89,code=sm_89 -o ncp2_rates ncp2_rates.cu && ./ncp2_rates
#include <stdio.h>

#include "ncp2_kernels.cu"

int main() {
  const char* names[4] = {"sha256", "sha512", "shake256", "turboshake128"};
  // (blocks, threads, chained compressions per thread): 1 leaf; 72 and 608 (qkv's and gate_up's tiles at 16 rows);
  // 19,008 (a 16-row forward's tiles); 3,072 x 256 (every lane, several waves)
  const int sweep[5][3] = {{1, 1, 2048}, {1, 72, 2048}, {19, 32, 2048}, {594, 32, 1024}, {3072, 256, 256}};
  int dev = 0;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 1;
  printf("{\"device\": \"%s\", \"sms\": %d, \"clock_khz\": %d}\n", prop.name, prop.multiProcessorCount, prop.clockRate);
  for (int kind = 0; kind < 4; ++kind)
    for (const auto& s : sweep) {
      double rate = 0;
      int rc = ncp2_ceiling(kind, s[0], s[1], s[2], &rate);
      double threads = (double)s[0] * s[1], ms = CEILING_BYTES[kind] * threads * s[2] / rate * 1e3;
      printf("{\"hash\": \"%s\", \"threads\": %.0f, \"chain\": %d, \"rc\": %d, \"bytes_per_s\": %.6g, \"us_per_step\": %.4g}\n",
             names[kind], threads, s[2], rc, rate, ms * 1e3 / s[2]);
    }
  return 0;
}
