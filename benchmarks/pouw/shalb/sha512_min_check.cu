// Host check of sha512_min.cuh: SHA-512 of messages of every length 0..300 bytes through shamin::compress, of longer
// ones through its chain form (pairs of blocks, CHAIN 1 then 2), and of the first 256 bytes through its padding-block
// form, printed as "TAG LEN HEX" lines for sha512_check.py to compare with hashlib. The compression is the device's
// source compiled for the host.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "sha512_min.cuh"

static constexpr size_t MSG_LEN = 2000;

static void block_words(const uint8_t* msg, size_t len, size_t off, size_t total, uint64_t* w) {
  uint8_t blk[128];
  for (int i = 0; i < 128; ++i) {
    const size_t j = off + i;
    blk[i] = j < len ? msg[j] : j == len ? 0x80 : 0;
  }
  if (off + 128 == total) {
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 7; i >= 0; --i) { blk[120 + i] = (uint8_t)bits; bits >>= 8; }
  }
  for (int i = 0; i < 16; ++i) {
    uint64_t v = 0;
    for (int k = 0; k < 8; ++k) v = v << 8 | blk[8 * i + k];
    w[i] = v;
  }
}

// CHAIN: every pair of blocks in chain form from the IV, an odd last block by the plain compression.
static void digest(const uint8_t* msg, size_t len, uint8_t* out, bool chain) {
  uint64_t st[8], w[16], w1[16], pend[2];
  for (int i = 0; i < 8; ++i) st[i] = shamin::IVt(i);
  const size_t total = ((len + 17 + 127) / 128) * 128;
  size_t off = 0;
  if (chain) {
    st[7] += shamin::Kt(0);
    for (; off + 256 <= total; off += 256) {
      block_words(msg, len, off, total, w);
      block_words(msg, len, off + 128, total, w1);
      shamin::compress<0, 0, 1>(st, w, pend);
      shamin::compress<0, 0, 2>(st, w1, pend);
    }
    st[7] -= shamin::Kt(0);
  }
  for (; off < total; off += 128) {
    block_words(msg, len, off, total, w);
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
  static uint8_t msg[MSG_LEN];
  for (size_t i = 0; i < MSG_LEN; ++i) msg[i] = (uint8_t)(i * 151 + 7);
  uint8_t out[64];
  for (size_t len = 0; len <= 300; ++len) {
    digest(msg, len, out, false);
    print("full", len, out);
  }
  constexpr size_t chain_lens[] = {0, 111, 112, 239, 240, 255, 256, 300, 367, 368, 500, 1000, 1023, 1024, 1500, 1999,
                                   2000};
  for (size_t len : chain_lens) {
    digest(msg, len, out, true);
    print("chain", len, out);
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
