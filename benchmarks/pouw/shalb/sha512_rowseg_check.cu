// Host check of sha512_rowseg.cuh: CASES bound row-seg segments through shaseg::segment, each from the case's segment
// index, call (salt_w, f, j), row and segment bytes (sha512_check.py's seg_case), printed as "seg CASE HEX" lines for
// sha512_check.py to compare with hashlib. The segment is the device's source compiled for the host.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "sha512_rowseg.cuh"

static constexpr int CASES = 16;

static void be64(uint8_t* p, uint64_t x) {
  for (int k = 0; k < 8; ++k) p[k] = (uint8_t)(x >> (56 - 8 * k));
}

int main() {
  static const char tag[] = "verity/sha512-row-seg/v1/segment";
  for (int c = 0; c < CASES; ++c) {
    uint8_t b1[128] = {0}, seg[1024];
    size_t n = sizeof tag;
    memcpy(b1, tag, n);
    be64(b1 + n, 3 * c); be64(b1 + n + 8, 1024); be64(b1 + n + 16, 8192);
    n += 24;
    for (int i = 0; i < 32; ++i) b1[n + i] = (uint8_t)(41 * i + 3 + 7 * c);
    n += 32;
    be64(b1 + n, 1000 + c); be64(b1 + n + 8, 5 + c); be64(b1 + n + 16, 77 * c + 1);
    for (int k = 0; k < 1024; ++k) seg[k] = (uint8_t)(151 * k + 7 + 13 * c);
    uint64_t w1[16], v[shaseg::SEGMENT_WORDS], st[8];
    shaseg::prefix_words(b1, w1);
    shaseg::prefix_words(seg, v);
    for (int i = 1; i < 8; ++i) shaseg::prefix_words(seg + 128 * i, v + 16 * i);
    shaseg::segment(w1, v, st);
    printf("seg %d ", c);
    for (int i = 0; i < 8; ++i) printf("%016llx", (unsigned long long)st[i]);
    printf("\n");
  }
  return 0;
}
