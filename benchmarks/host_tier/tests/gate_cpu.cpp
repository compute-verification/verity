// The CPU half of loop.cu's gate, without CUDA: `gate_cpu KS SHA_NI` prints the gate JSON for group 0 of region 1000
// under z0 (no GPU receipts); `gate_cpu sha SHA_NI` prints "n sha256-hex" for the lengths test_receipt.py checks.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#include "receipt.h"

const uint8_t SEED[32] = {0x68, 0x6f, 0x73, 0x74, 0x2d, 0x74, 0x69, 0x65, 0x72, 0x2f, 0x76, 0x30, 1,  2,  3,  4,
                          5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17, 18, 19, 20};

static uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  sha_ni = atoi(argv[2]);
  if (!strcmp(argv[1], "sha")) {
    static uint8_t m[1000];
    for (size_t n : {0, 3, 55, 56, 63, 64, 65, 119, 120, 1000}) {
      for (size_t i = 0; i < n; ++i) m[i] = (uint8_t)(7 * i + 1);
      uint8_t h[32];
      sha256(m, n, h);
      printf("%zu ", n);
      for (int i = 0; i < 32; ++i) printf("%02x", h[i]);
      printf("\n");
    }
    return 0;
  }
  const int ks = atoi(argv[1]);
  uint64_t s;
  memcpy(&s, SEED, 8);
  uint32_t z[16];
  for (int t = 0; t < 16; ++t) z[t] = (uint32_t)splitmix64(s ^ (0x7a00000000ull + t));
  alignas(64) int8_t K[16 * 256];
  cpu_keys(K, ks);
  uint64_t* x = (uint64_t*)aligned_alloc(4096, 65536);
  for (int i = 0; i < 8192; ++i) x[i] = splitmix64(i ^ (1000ull << 48));
  alignas(64) uint8_t M[4096];
  cpu_mask(z, M);
  int32_t sk[16 * 16];
  uint8_t leaf[32];
  cpu_group((const uint8_t*)x, M, 0, 0, K, ks, leaf, sk);
  printf("{\"seed\": \"");
  for (int i = 0; i < 32; ++i) printf("%02x", SEED[i]);
  printf("\", \"ks\": %d, \"gpu_region\": 0, \"gpu_receipts\": [], \"cpu_region\": 1000, \"cpu_sketches\": [", ks);
  for (int i = 0; i < 16 * ks; ++i) printf("%s%d", i ? ", " : "", sk[i]);
  printf("], \"cpu_leaf\": \"");
  for (int i = 0; i < 32; ++i) printf("%02x", leaf[i]);
  printf("\"}\n");
  return 0;
}
