// Host check of sha512_l1.cuh: CASES L1 leaves through shal1::leaf, each from the case's domain id, rank, position
// and value (sha512_check.py's l1_case), printed as "l1 CASE HEX" lines for sha512_check.py to compare with hashlib.
// The leaf is the device's source compiled for the host.
#include <cstdint>
#include <cstdio>

#include "sha512_l1.cuh"

static constexpr int CASES = 16;

int main() {
  for (int c = 0; c < CASES; ++c) {
    uint8_t dom[64];
    for (int i = 0; i < 64; ++i) dom[i] = (uint8_t)(37 * i + 11 + 5 * c);
    uint64_t cw[9], v[shal1::VALUE_WORDS], st[8];
    shal1::call_words(dom, cw);
    for (int k = 0; k < shal1::VALUE_WORDS; ++k) {
      uint64_t x = 0;
      for (int j = 0; j < 8; ++j) x = x << 8 | (uint8_t)(151 * (8 * k + j) + 7 + 13 * c);
      v[k] = x;
    }
    shal1::leaf(cw, shal1::rank_word((uint8_t)(29 * c)), shal1::pos_word((uint8_t)(53 * c + 1)), v, st);
    printf("l1 %d ", c);
    for (int i = 0; i < 8; ++i) printf("%016llx", (unsigned long long)st[i]);
    printf("\n");
  }
  return 0;
}
