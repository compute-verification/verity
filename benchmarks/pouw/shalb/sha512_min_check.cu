// Host check of sha512_min.cuh: SHA-512 of messages of every length 0..300 bytes through shamin::compress, and of the
// first 256 bytes through its padding-block form, printed as "TAG LEN HEX" lines for sha512_check.py to compare with
// hashlib. The compression is the device's source compiled for the host.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "sha512_min.cuh"

static void digest(const uint8_t* msg, size_t len, uint8_t* out) {
  uint64_t st[8];
  for (int i = 0; i < 8; ++i) st[i] = shamin::IVt(i);
  const size_t total = ((len + 17 + 127) / 128) * 128;
  for (size_t off = 0; off < total; off += 128) {
    uint8_t blk[128];
    for (int i = 0; i < 128; ++i) {
      const size_t j = off + i;
      blk[i] = j < len ? msg[j] : j == len ? 0x80 : 0;
    }
    if (off + 128 == total) {
      uint64_t bits = (uint64_t)len * 8;
      for (int i = 7; i >= 0; --i) { blk[120 + i] = (uint8_t)bits; bits >>= 8; }
    }
    uint64_t w[16];
    for (int i = 0; i < 16; ++i) {
      uint64_t v = 0;
      for (int k = 0; k < 8; ++k) v = v << 8 | blk[8 * i + k];
      w[i] = v;
    }
    if (off == 0) {
      shamin::compress<1>(st, w);
    } else {
      shamin::compress<0>(st, w);
    }
  }
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 8; ++k) out[8 * i + k] = (uint8_t)(st[i] >> (56 - 8 * k));
}

static void print(const char* tag, size_t len, const uint8_t* out) {
  printf("%s %zu ", tag, len);
  for (int i = 0; i < 64; ++i) printf("%02x", out[i]);
  printf("\n");
}

int main() {
  uint8_t msg[300];
  for (int i = 0; i < 300; ++i) msg[i] = (uint8_t)(i * 151 + 7);
  uint8_t out[64];
  for (size_t len = 0; len <= 300; ++len) {
    digest(msg, len, out);
    print("full", len, out);
  }
  // The first 256 bytes as an hm96 leaf hashes them: two message blocks, then the padding block from PadBlock(2048).
  uint64_t st[8], w[16];
  for (int i = 0; i < 8; ++i) st[i] = shamin::IVt(i);
  for (int blk = 0; blk < 2; ++blk) {
    for (int i = 0; i < 16; ++i) {
      uint64_t v = 0;
      for (int k = 0; k < 8; ++k) v = v << 8 | msg[128 * blk + 8 * i + k];
      w[i] = v;
    }
    if (blk == 0) {
      shamin::compress<1>(st, w);
    } else {
      shamin::compress<0>(st, w);
    }
  }
  shamin::compress<0, 2048>(st, nullptr);
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 8; ++k) out[8 * i + k] = (uint8_t)(st[i] >> (56 - 8 * k));
  print("pad", 256, out);
  return 0;
}
