// Host check of sha512_rowv2.cuh: CASES 1 KiB rows through shav2::chunk, row byte k = 151 k + 7 + 13 c (sha512_check.py's
// v2_case), printed as "v2 CASE HEX" lines for sha512_check.py to compare with hashlib. The chunk is the device's
// source compiled for the host.
#include <cstdint>
#include <cstdio>

#include "sha512_rowv2.cuh"

static constexpr int CASES = 16;

int main() {
  for (int c = 0; c < CASES; ++c) {
    uint64_t v[128], st[8];
    for (int i = 0; i < 128; ++i) {
      uint64_t x = 0;
      for (int k = 0; k < 8; ++k) x = x << 8 | (uint8_t)(151 * (8 * i + k) + 7 + 13 * c);
      v[i] = x;
    }
    shav2::chunk<8, 8192>(v, st);
    printf("v2 %d ", c);
    for (int i = 0; i < 8; ++i) printf("%016llx", (unsigned long long)st[i]);
    printf("\n");
  }
  return 0;
}
