// The challenger's stream generator (stream.py's S): AES-128-CTR with the 128-bit big-endian counter block i at block i,
// on the host.  Three implementations of the same keystream, so a run can say what the stream costs the host:
//   vaes     AVX-512 VAES, four blocks a register and eight registers in flight (32 blocks, 512 bytes, a step); vaes_nt
//            the same with non-temporal stores, for a stream bound for pinned memory and the copy engines
//   aesni    128-bit AES-NI, eight blocks in flight, and one at a time for the verifier's scattered blocks
//   openssl  libcrypto's EVP aes-128-ctr, loaded at run time (the node has the library but not its headers)
// Built by g++ (-O3 -mavx512f -mavx512bw -mvaes -maes), not nvcc, and linked into bwstream.

#include <dlfcn.h>
#include <immintrin.h>

#include <cstdint>
#include <cstring>

namespace {

template <int R>
inline __m128i expand_step(__m128i k) {
  __m128i t = _mm_aeskeygenassist_si128(k, R);
  t = _mm_shuffle_epi32(t, 0xff);
  k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
  k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
  k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
  return _mm_xor_si128(k, t);
}

inline __m128i bswap128(__m128i x) {
  return _mm_shuffle_epi8(x, _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
}

template <bool NT>
inline void ctr_vaes(const uint8_t rk[176], uint64_t first, uint8_t* out, uint64_t blocks) {
  __m512i K[11];
  for (int r = 0; r < 11; ++r) K[r] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)(rk + 16 * r)));
  const __m512i swap = _mm512_broadcast_i32x4(_mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
  const __m512i four = _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4);
  __m512i c = _mm512_set_epi64(0, first + 3, 0, first + 2, 0, first + 1, 0, first);
  for (uint64_t b = 0; b < blocks; b += 32) {
    __m512i x[8];
#pragma GCC unroll 8
    for (int j = 0; j < 8; ++j) {
      x[j] = _mm512_xor_si512(_mm512_shuffle_epi8(c, swap), K[0]);
      c = _mm512_add_epi64(c, four);
    }
#pragma GCC unroll 9
    for (int r = 1; r < 10; ++r)
#pragma GCC unroll 8
      for (int j = 0; j < 8; ++j) x[j] = _mm512_aesenc_epi128(x[j], K[r]);
#pragma GCC unroll 8
    for (int j = 0; j < 8; ++j) {
      __m512i y = _mm512_aesenclast_epi128(x[j], K[10]);
      if (NT)
        _mm512_stream_si512((__m512i*)(out + 64 * (b / 4 + j)), y);
      else
        _mm512_store_si512((__m512i*)(out + 64 * (b / 4 + j)), y);
    }
  }
  if (NT) _mm_sfence();
}

}  // namespace

extern "C" {

// The 11 round keys of AES-128, 176 bytes.
void sg_expand(const uint8_t key[16], uint8_t rk[176]) {
  __m128i k[11];
  k[0] = _mm_loadu_si128((const __m128i*)key);
  k[1] = expand_step<0x01>(k[0]);
  k[2] = expand_step<0x02>(k[1]);
  k[3] = expand_step<0x04>(k[2]);
  k[4] = expand_step<0x08>(k[3]);
  k[5] = expand_step<0x10>(k[4]);
  k[6] = expand_step<0x20>(k[5]);
  k[7] = expand_step<0x40>(k[6]);
  k[8] = expand_step<0x80>(k[7]);
  k[9] = expand_step<0x1b>(k[8]);
  k[10] = expand_step<0x36>(k[9]);
  for (int i = 0; i < 11; ++i) _mm_storeu_si128((__m128i*)(rk + 16 * i), k[i]);
}

// Blocks [first, first + blocks) of the keystream into out (64-byte aligned, blocks a multiple of 32).  The counter
// blocks are u128be(i) for i < 2^64.
void sg_ctr_vaes(const uint8_t rk[176], uint64_t first, uint8_t* out, uint64_t blocks) {
  ctr_vaes<false>(rk, first, out, blocks);
}

// The same with non-temporal stores (no read for ownership; the lines go to memory, not the cache), fenced at the end.
void sg_ctr_vaes_nt(const uint8_t rk[176], uint64_t first, uint8_t* out, uint64_t blocks) {
  ctr_vaes<true>(rk, first, out, blocks);
}

// The same keystream with 128-bit AES-NI, eight blocks in flight, any number of blocks.
void sg_ctr_aesni(const uint8_t rk[176], uint64_t first, uint8_t* out, uint64_t blocks) {
  __m128i K[11];
  for (int r = 0; r < 11; ++r) K[r] = _mm_loadu_si128((const __m128i*)(rk + 16 * r));
  uint64_t b = 0;
  for (; b + 8 <= blocks; b += 8) {
    __m128i x[8];
    for (int j = 0; j < 8; ++j) x[j] = _mm_xor_si128(bswap128(_mm_set_epi64x(0, (long long)(first + b + j))), K[0]);
    for (int r = 1; r < 10; ++r)
      for (int j = 0; j < 8; ++j) x[j] = _mm_aesenc_si128(x[j], K[r]);
    for (int j = 0; j < 8; ++j) _mm_storeu_si128((__m128i*)(out + 16 * (b + j)), _mm_aesenclast_si128(x[j], K[10]));
  }
  for (; b < blocks; ++b) {
    __m128i x = _mm_xor_si128(bswap128(_mm_set_epi64x(0, (long long)(first + b))), K[0]);
    for (int r = 1; r < 10; ++r) x = _mm_aesenc_si128(x, K[r]);
    _mm_storeu_si128((__m128i*)(out + 16 * b), _mm_aesenclast_si128(x, K[10]));
  }
}

// libcrypto's aes-128-ctr from counter block u128be(first): 0 on success, -1 if libcrypto.so.3 is not there.
int sg_ctr_openssl(const uint8_t key[16], uint64_t first, uint8_t* out, uint64_t blocks) {
  using New = void* (*)();
  using Cipher = const void* (*)();
  using Init = int (*)(void*, const void*, void*, const uint8_t*, const uint8_t*);
  using Update = int (*)(void*, uint8_t*, int*, const uint8_t*, int);
  using Free = void (*)(void*);
  static void* lib = dlopen("libcrypto.so.3", RTLD_NOW);
  if (!lib) return -1;
  static New ctx_new = (New)dlsym(lib, "EVP_CIPHER_CTX_new");
  static Cipher ctr = (Cipher)dlsym(lib, "EVP_aes_128_ctr");
  static Init init = (Init)dlsym(lib, "EVP_EncryptInit_ex");
  static Update update = (Update)dlsym(lib, "EVP_EncryptUpdate");
  static Free ctx_free = (Free)dlsym(lib, "EVP_CIPHER_CTX_free");
  if (!ctx_new || !ctr || !init || !update || !ctx_free) return -1;
  uint8_t iv[16] = {0};
  for (int i = 0; i < 8; ++i) iv[15 - i] = (uint8_t)(first >> (8 * i));
  void* ctx = ctx_new();
  if (!init(ctx, ctr(), nullptr, key, iv)) return ctx_free(ctx), -1;
  memset(out, 0, blocks * 16);
  const uint64_t step = 1 << 26;
  for (uint64_t off = 0; off < blocks * 16; off += step) {
    int n = (int)(blocks * 16 - off < step ? blocks * 16 - off : step), got = 0;
    if (!update(ctx, out + off, &got, out + off, n)) return ctx_free(ctx), -1;
  }
  ctx_free(ctx);
  return 0;
}

// Evicts p[0, n) from every cache level (clflush, a 64-byte line at a time), fenced.
void sg_flush(const uint8_t* p, uint64_t n) {
  for (uint64_t i = 0; i < n; i += 64) _mm_clflush(p + i);
  _mm_mfence();
}

// Writes p[0, n)'s dirty lines back to memory and leaves them cached and clean (clwb), fenced.
__attribute__((target("clwb"))) void sg_clwb(const uint8_t* p, uint64_t n) {
  for (uint64_t i = 0; i < n; i += 64) _mm_clwb(const_cast<uint8_t*>(p) + i);
  _mm_sfence();
}

}  // extern "C"
