// POUS native code for the serving process: batched SHAKE256 and Feistel-10 kernels (one warp per message) and the audit responder,
// a native thread that answers challenges from C in device memory over TCP, with no Python and no GIL on the path.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

__constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
__constant__ int KECCAK_RHO[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39,
                                   41, 45, 15, 21, 8, 18, 2, 61, 56, 14};

constexpr int RATE = 136;  // SHAKE256: 1088-bit rate
constexpr int RATE_WORDS = RATE / 8;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int MAX_TWEAK = 62;

// One warp holds one Keccak state: lane x + 5y holds word (x, y); lanes 25..31 shadow lane % 25 and are never read.
struct Lane {
  int t1, t2, t3, t4, dm, dp, rho, pi, c1, c2;
  bool first;
};

__device__ __forceinline__ Lane lane_of(int lane) {
  const int v = lane % 25, x = v % 5, y = v / 5;
  Lane l;
  l.t1 = x + 5 * ((y + 1) % 5);
  l.t2 = x + 5 * ((y + 2) % 5);
  l.t3 = x + 5 * ((y + 3) % 5);
  l.t4 = x + 5 * ((y + 4) % 5);
  l.dm = (x + 4) % 5 + 5 * y;
  l.dp = (x + 1) % 5 + 5 * y;
  l.rho = KECCAK_RHO[v];
  l.pi = (x + 3 * y) % 5 + 5 * x;  // pi: word (x, y) moves to (y, 2x + 3y), so lane (X, Y) takes ((X + 3Y) % 5, X)
  l.c1 = (x + 1) % 5 + 5 * y;
  l.c2 = (x + 2) % 5 + 5 * y;
  l.first = lane == 0;
  return l;
}

__device__ __forceinline__ uint64_t shfl64(uint64_t v, int src) { return __shfl_sync(0xffffffffu, v, src); }
__device__ __forceinline__ uint64_t rotlv(uint64_t v, int r) { return r ? (v << r) | (v >> (64 - r)) : v; }

__device__ __forceinline__ uint64_t keccak_f(uint64_t a, const Lane& l) {
#pragma unroll 1
  for (int r = 0; r < 24; ++r) {
    const uint64_t c = a ^ shfl64(a, l.t1) ^ shfl64(a, l.t2) ^ shfl64(a, l.t3) ^ shfl64(a, l.t4);
    const uint64_t d = shfl64(c, l.dm) ^ rotlv(shfl64(c, l.dp), 1);
    a = rotlv(a ^ d, l.rho);
    const uint64_t b = shfl64(a, l.pi);
    a = b ^ (~shfl64(b, l.c1) & shfl64(b, l.c2));
    if (l.first) a ^= KECCAK_RC[r];
  }
  return a;
}

// dst[0:olen] (^)= SHAKE256(pre[0:plen] || body[0:blen])[0:olen], computed by the calling warp.
__device__ void shake_warp(const uint8_t* pre, int plen, const uint8_t* body, int64_t blen, uint8_t* dst, int64_t olen,
                           bool xor_out, const Lane& l, int lane) {
  const int64_t len = plen + blen;
  const int64_t nblk = len / RATE + 1;
  uint64_t a = 0;
  for (int64_t b = 0; b < nblk; ++b) {
    if (lane < RATE_WORDS) {
      const int64_t base = b * RATE + lane * 8;
      uint64_t w = 0;
      if (base >= plen && base + 8 <= len) {
        const uint8_t* p = body + (base - plen);
#pragma unroll
        for (int k = 0; k < 8; ++k) w |= (uint64_t)p[k] << (8 * k);
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int64_t pos = base + k;
          uint8_t byte = 0;
          if (pos < plen) byte = pre[pos];
          else if (pos < len) byte = body[pos - plen];
          if (pos == len) byte ^= 0x1F;
          w |= (uint64_t)byte << (8 * k);
        }
        if (b == nblk - 1 && lane == RATE_WORDS - 1) w ^= 0x80ULL << 56;
      }
      a ^= w;
    }
    a = keccak_f(a, l);
  }
  for (int64_t o = 0; o < olen; o += RATE) {
    if (o > 0) a = keccak_f(a, l);
    if (lane < RATE_WORDS) {
      const int64_t base = o + lane * 8;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int64_t pos = base + k;
        if (pos < olen) {
          const uint8_t byte = (uint8_t)(a >> (8 * k));
          dst[pos] = xor_out ? (uint8_t)(dst[pos] ^ byte) : byte;
        }
      }
    }
  }
  __syncwarp();
}

// Row n: out[n, :olen] (^)= SHAKE256(prefix[n, :plen] || body[n, :blen])[0:olen]; one warp per row.
__global__ void shake256_kernel(const uint8_t* prefix, int plen, int64_t prefix_stride, const uint8_t* body, int64_t blen,
                                int64_t body_stride, uint8_t* out, int64_t olen, int64_t out_stride, int xor_out, int64_t rows) {
  const int lane = threadIdx.x % 32;
  const int64_t n = (int64_t)blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / 32;
  if (n >= rows) return;
  const Lane l = lane_of(lane);
  shake_warp(prefix + n * prefix_stride, plen, body + n * body_stride, blen, out + n * out_stride, olen, xor_out != 0, l, lane);
}

// Ten Feistel rounds in place on row n of x (halves A || B of `half` bytes): round i computes
// F_i = SHAKE256(domain || u8(i) || tweak[n] || src)[0:half] and xors it into the other half, odd rounds A ^= F_i(B),
// even rounds B ^= F_i(A); the inverse runs i = 10 .. 1.  That is (L, R) -> (R, L ^ F_i(R)) with the output left as A || B.
__global__ void feistel_kernel(uint8_t* x, int64_t x_stride, int64_t half, const uint8_t* tweak, int tlen, int64_t tweak_stride,
                               int domain, int inverse, int64_t rows) {
  __shared__ uint8_t prefix[WARPS_PER_BLOCK][MAX_TWEAK + 2];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int64_t n = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp;
  if (n >= rows) return;
  const Lane l = lane_of(lane);
  uint8_t* pre = prefix[warp];
  for (int j = lane; j < tlen; j += 32) pre[2 + j] = tweak[n * tweak_stride + j];
  if (lane == 0) pre[0] = (uint8_t)domain;
  uint8_t* a = x + n * x_stride;
  uint8_t* b = a + half;
  for (int step = 0; step < 10; ++step) {
    const int i = inverse ? 10 - step : step + 1;
    if (lane == 0) pre[1] = (uint8_t)i;
    __syncwarp();
    if (i % 2) shake_warp(pre, 2 + tlen, b, half, a, half, true, l, lane);
    else shake_warp(pre, 2 + tlen, a, half, b, half, true, l, lane);
  }
}

// ---- one thread per row (throughput): a whole Keccak state in one thread's registers ------------------------------------
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

__device__ __forceinline__ void keccak_f_thread(uint64_t st[25]) {
  const int rotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
  const int piln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
#pragma unroll 1
  for (int r = 0; r < 24; ++r) {
    uint64_t bc[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }
    uint64_t t = st[1];
#pragma unroll
    for (int i = 0; i < 24; ++i) {
      int j = piln[i];
      uint64_t b0 = st[j];
      st[j] = rotl64(t, rotc[i]);
      t = b0;
    }
#pragma unroll
    for (int j = 0; j < 25; j += 5) {
      uint64_t b[5];
#pragma unroll
      for (int i = 0; i < 5; ++i) b[i] = st[j + i];
#pragma unroll
      for (int i = 0; i < 5; ++i) st[j + i] ^= (~b[(i + 1) % 5]) & b[(i + 2) % 5];
    }
    st[0] ^= KECCAK_RC[r];
  }
}

// dst[0:olen] ^= SHAKE256(pre[0:plen] || body[0:blen])[0:olen], computed by the calling thread alone.
__device__ void shake_thread_xor(const uint8_t* pre, int plen, const uint8_t* body, int64_t blen, uint8_t* dst, int64_t olen) {
  const int64_t len = plen + blen;
  const int64_t nblk = len / RATE + 1;
  uint64_t st[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) st[i] = 0;
  for (int64_t b = 0; b < nblk; ++b) {
    const int64_t base = b * RATE;
    if (base >= plen && base + RATE <= len) {
      const uint8_t* p = body + (base - plen);
#pragma unroll
      for (int w = 0; w < RATE_WORDS; ++w) {
        uint64_t word = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) word |= (uint64_t)p[w * 8 + k] << (8 * k);
        st[w] ^= word;
      }
    } else {
#pragma unroll
      for (int w = 0; w < RATE_WORDS; ++w) {
        uint64_t word = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int64_t pos = base + w * 8 + k;
          uint8_t byte = 0;
          if (pos < plen) byte = pre[pos];
          else if (pos < len) byte = body[pos - plen];
          if (pos == len) byte ^= 0x1F;
          if (b == nblk - 1 && w == RATE_WORDS - 1 && k == 7) byte ^= 0x80;
          word |= (uint64_t)byte << (8 * k);
        }
        st[w] ^= word;
      }
    }
    keccak_f_thread(st);
  }
  for (int64_t o = 0; o < olen; o += RATE) {
    if (o > 0) keccak_f_thread(st);
#pragma unroll
    for (int w = 0; w < RATE_WORDS; ++w) {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int64_t i = o + w * 8 + k;
        if (i < olen) dst[i] ^= (uint8_t)(st[w] >> (8 * k));
      }
    }
  }
}

// feistel_kernel with one thread per row: many rows at once, where the warp kernel is bound by its shuffles.
__global__ void feistel_thread_kernel(uint8_t* x, int64_t x_stride, int64_t half, const uint8_t* tweak, int tlen,
                                      int64_t tweak_stride, int domain, int inverse, int64_t rows) {
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= rows) return;
  uint8_t pre[MAX_TWEAK + 2];
  pre[0] = (uint8_t)domain;
  for (int j = 0; j < tlen; ++j) pre[2 + j] = tweak[n * tweak_stride + j];
  uint8_t* a = x + n * x_stride;
  uint8_t* b = a + half;
  for (int step = 0; step < 10; ++step) {
    const int i = inverse ? 10 - step : step + 1;
    pre[1] = (uint8_t)i;
    if (i % 2) shake_thread_xor(pre, 2 + tlen, b, half, a, half);
    else shake_thread_xor(pre, 2 + tlen, a, half, b, half);
  }
}

// dst ^= SHAKE256(prefix || body)[0:olen] with 64-bit word I/O: `body` and `dst` 8-byte aligned, `blen` and `olen` multiples
// of 8, the prefix (`plen` < 8 bytes, little-endian in `pre`) shifting the body's words by `plen` bytes in the message.
__device__ void shake_thread_xor_words(uint64_t pre, int plen, const uint64_t* __restrict__ body, int64_t blen,
                                       uint64_t* __restrict__ dst, int64_t olen) {
  const int64_t len = plen + blen;
  const int64_t nblk = len / RATE + 1;
  const int sh = 8 * plen;
  uint64_t st[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) st[i] = 0;
  uint64_t prev = pre;  // message word j is (body[j] << sh) | (body[j - 1] >> (64 - sh)), body[-1] standing in for the prefix
  for (int64_t b = 0; b < nblk; ++b) {
    const int64_t base = b * RATE;
    if (base + RATE <= len && (base + RATE) / 8 <= blen / 8) {
#pragma unroll
      for (int w = 0; w < RATE_WORDS; ++w) {
        const uint64_t cur = body[b * RATE_WORDS + w];
        st[w] ^= sh ? (cur << sh) | (b == 0 && w == 0 ? prev : prev >> (64 - sh)) : cur;
        prev = cur;
      }
    } else {
      const uint8_t* bb = reinterpret_cast<const uint8_t*>(body);
#pragma unroll
      for (int w = 0; w < RATE_WORDS; ++w) {
        uint64_t word = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int64_t pos = base + w * 8 + k;
          uint8_t byte = 0;
          if (pos < plen) byte = (uint8_t)(pre >> (8 * pos));
          else if (pos < len) byte = bb[pos - plen];
          if (pos == len) byte ^= 0x1F;
          if (b == nblk - 1 && w == RATE_WORDS - 1 && k == 7) byte ^= 0x80;
          word |= (uint64_t)byte << (8 * k);
        }
        st[w] ^= word;
      }
    }
    keccak_f_thread(st);
  }
  const int64_t owords = olen / 8;
  for (int64_t o = 0; o < owords; o += RATE_WORDS) {
    if (o > 0) keccak_f_thread(st);
#pragma unroll
    for (int w = 0; w < RATE_WORDS; ++w)
      if (o + w < owords) dst[o + w] ^= st[w];
  }
}

// feistel_thread_kernel with word I/O, for untweaked permutations whose halves are whole words (Π and Π₂).
__global__ void feistel_words_kernel(uint8_t* x, int64_t x_stride, int64_t half, int domain, int inverse, int64_t rows) {
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= rows) return;
  uint64_t* a = reinterpret_cast<uint64_t*>(x + n * x_stride);
  uint64_t* b = a + half / 8;
  for (int step = 0; step < 10; ++step) {
    const int i = inverse ? 10 - step : step + 1;
    const uint64_t pre = (uint64_t)(uint8_t)domain | ((uint64_t)(uint8_t)i << 8);
    if (i % 2) shake_thread_xor_words(pre, 2, b, half, a, half);
    else shake_thread_xor_words(pre, 2, a, half, b, half);
  }
}

// Two independent Keccak-f[1600] permutations in one round loop, so their instructions interleave.
__device__ __forceinline__ void keccak_f_pair(uint64_t s[25], uint64_t t[25]) {
  const int rotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
  const int piln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
#pragma unroll 1
  for (int r = 0; r < 24; ++r) {
    uint64_t bs[5], bt[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      bs[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
      bt[i] = t[i] ^ t[i + 5] ^ t[i + 10] ^ t[i + 15] ^ t[i + 20];
    }
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const uint64_t ds = bs[(i + 4) % 5] ^ rotl64(bs[(i + 1) % 5], 1);
      const uint64_t dt = bt[(i + 4) % 5] ^ rotl64(bt[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) {
        s[j + i] ^= ds;
        t[j + i] ^= dt;
      }
    }
    uint64_t us = s[1], ut = t[1];
#pragma unroll
    for (int i = 0; i < 24; ++i) {
      const int j = piln[i];
      const uint64_t s0 = s[j], t0 = t[j];
      s[j] = rotl64(us, rotc[i]);
      t[j] = rotl64(ut, rotc[i]);
      us = s0;
      ut = t0;
    }
#pragma unroll
    for (int j = 0; j < 25; j += 5) {
      uint64_t a[5], c[5];
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        a[i] = s[j + i];
        c[i] = t[j + i];
      }
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        s[j + i] ^= (~a[(i + 1) % 5]) & a[(i + 2) % 5];
        t[j + i] ^= (~c[(i + 1) % 5]) & c[(i + 2) % 5];
      }
    }
    s[0] ^= KECCAK_RC[r];
    t[0] ^= KECCAK_RC[r];
  }
}

// The sponge absorbing prefix (2 bytes, `pre`) || src (`hw` words), padded: `st` is then ready to squeeze.
__device__ void absorb_words(uint64_t st[25], uint64_t pre, const uint64_t* __restrict__ src, int64_t hw) {
#pragma unroll
  for (int i = 0; i < 25; ++i) st[i] = 0;
  const int64_t len = 8 * hw + 2;
  const int64_t full = len / RATE;
  uint64_t carry = pre << 48;
  for (int64_t k = 0; k < full; ++k) {
#pragma unroll
    for (int w = 0; w < RATE_WORDS; ++w) {
      const uint64_t cur = src[k * RATE_WORDS + w];
      st[w] ^= (cur << 16) | (carry >> 48);
      carry = cur;
    }
    keccak_f_thread(st);
  }
  const int64_t j0 = full * RATE_WORDS;
  const int tail = (int)(len - full * RATE);
#pragma unroll
  for (int w = 0; w < RATE_WORDS; ++w) {
    const int64_t j = j0 + w;
    uint64_t m = 0;
    if (j < hw) {
      const uint64_t cur = src[j];
      m = (cur << 16) | (carry >> 48);
      carry = cur;
    } else if (j == hw) {
      m = carry >> 48;
    }
    if (w == tail / 8) m ^= (uint64_t)0x1F << (8 * (tail % 8));
    if (w == RATE_WORDS - 1) m ^= (uint64_t)0x80 << 56;
    st[w] ^= m;
  }
  keccak_f_thread(st);
}

// Feistel-10 forward with round r's squeeze fused to round r + 1's absorb: the words round r writes into its destination
// half are absorbed into round r + 1's sponge from registers, so each round reads and writes one half once.
__global__ void feistel_fused_kernel(uint8_t* x, int64_t x_stride, int64_t half, int domain, int64_t rows) {
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= rows) return;
  const int64_t hw = half / 8;
  uint64_t* a = reinterpret_cast<uint64_t*>(x + n * x_stride);
  uint64_t* b = a + hw;
  const int64_t len = 8 * hw + 2;
  const int64_t full = len / RATE;
  const int tail = (int)(len - full * RATE);
  const int64_t chunks = (hw + RATE_WORDS - 1) / RATE_WORDS;
  uint64_t s[25], t[25];
  absorb_words(s, (uint64_t)(uint8_t)domain | (1ull << 8), b, hw);
  for (int i = 1; i <= 10; ++i) {
    uint64_t* dst = (i % 2) ? a : b;
    const bool next = i < 10;
#pragma unroll
    for (int k = 0; k < 25; ++k) t[k] = 0;
    uint64_t carry = ((uint64_t)(uint8_t)domain | ((uint64_t)(i + 1) << 8)) << 48;
    for (int64_t c = 0; c < chunks; ++c) {
#pragma unroll
      for (int w = 0; w < RATE_WORDS; ++w) {
        const int64_t j = c * RATE_WORDS + w;
        if (j < hw) {
          const uint64_t out = dst[j] ^ s[w];
          dst[j] = out;
          if (c < full) t[w] ^= (out << 16) | (carry >> 48);
          carry = out;
        }
      }
      if (c + 1 < chunks) {
        if (next && c < full) keccak_f_pair(s, t);
        else keccak_f_thread(s);
      } else if (next && c < full) {
        keccak_f_thread(t);
      }
    }
    if (!next) break;
    const int64_t j0 = full * RATE_WORDS;
    const uint64_t* src = dst;
    uint64_t c2 = j0 > 0 ? src[j0 - 1] : ((uint64_t)(uint8_t)domain | ((uint64_t)(i + 1) << 8)) << 48;
#pragma unroll
    for (int w = 0; w < RATE_WORDS; ++w) {
      const int64_t j = j0 + w;
      uint64_t m = 0;
      if (j < hw) {
        const uint64_t cur = src[j];
        m = (cur << 16) | (c2 >> 48);
        c2 = cur;
      } else if (j == hw) {
        m = c2 >> 48;
      }
      if (w == tail / 8) m ^= (uint64_t)0x1F << (8 * (tail % 8));
      if (w == RATE_WORDS - 1) m ^= (uint64_t)0x80 << 56;
      t[w] ^= m;
    }
    keccak_f_thread(t);
#pragma unroll
    for (int k = 0; k < 25; ++k) s[k] = t[k];
  }
}

// feistel_fused_kernel with each row on a lane pair: in round r one lane squeezes round r's sponge and writes the destination
// half, and its partner absorbs the words it writes (sent by shuffle) into round r + 1's sponge; the roles swap each round.  Both
// lanes run the same Keccak code on their own state, so the pair does not diverge.
__global__ void feistel_pair_kernel(uint8_t* x, int64_t x_stride, int64_t half, int domain, int64_t rows) {
  const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t n = tid / 2;
  const int parity = (int)(tid & 1);
  const bool active = n < rows;
  const int64_t hw = half / 8;
  uint64_t* a = reinterpret_cast<uint64_t*>(x + (active ? n : 0) * x_stride);
  uint64_t* b = a + hw;
  const int64_t len = 8 * hw + 2;
  const int64_t full = len / RATE;
  const int tail = (int)(len - full * RATE);
  const int64_t chunks = (hw + RATE_WORDS - 1) / RATE_WORDS;
  uint64_t st[25];
#pragma unroll
  for (int k = 0; k < 25; ++k) st[k] = 0;
  if (parity == 1 && active) absorb_words(st, (uint64_t)(uint8_t)domain | (1ull << 8), b, hw);
  __syncwarp();
  for (int i = 1; i <= 10; ++i) {
    uint64_t* dst = (i % 2) ? a : b;
    const bool squeezer = parity == (i % 2);
    const bool next = i < 10;
    if (!squeezer) {
#pragma unroll
      for (int k = 0; k < 25; ++k) st[k] = 0;
    }
    uint64_t carry = ((uint64_t)(uint8_t)domain | ((uint64_t)(i + 1) << 8)) << 48;
    uint64_t nxt[RATE_WORDS];
#pragma unroll
    for (int w = 0; w < RATE_WORDS; ++w) nxt[w] = (squeezer && active && w < hw) ? dst[w] : 0;
    for (int64_t c = 0; c < chunks; ++c) {
      uint64_t cur[RATE_WORDS];
#pragma unroll
      for (int w = 0; w < RATE_WORDS; ++w) {
        cur[w] = nxt[w];
        const int64_t jn = (c + 1) * RATE_WORDS + w;
        nxt[w] = (squeezer && active && jn < hw) ? dst[jn] : 0;
      }
#pragma unroll
      for (int w = 0; w < RATE_WORDS; ++w) {
        const int64_t j = c * RATE_WORDS + w;
        uint64_t out = 0;
        if (j < hw && squeezer && active) {
          out = cur[w] ^ st[w];
          dst[j] = out;
        }
        const uint64_t got = __shfl_xor_sync(0xffffffffu, out, 1);
        if (j < hw && !squeezer && c < full) {
          st[w] ^= (got << 16) | (carry >> 48);
          carry = got;
        }
      }
      if (c + 1 < chunks || (next && c < full)) keccak_f_thread(st);
    }
    __syncwarp();
    if (!next) break;
    if (!squeezer && active) {
      const int64_t j0 = full * RATE_WORDS;
      uint64_t c2 = j0 > 0 ? dst[j0 - 1] : ((uint64_t)(uint8_t)domain | ((uint64_t)(i + 1) << 8)) << 48;
#pragma unroll
      for (int w = 0; w < RATE_WORDS; ++w) {
        const int64_t j = j0 + w;
        uint64_t m = 0;
        if (j < hw) {
          const uint64_t cur = dst[j];
          m = (cur << 16) | (c2 >> 48);
          c2 = cur;
        } else if (j == hw) {
          m = c2 >> 48;
        }
        if (w == tail / 8) m ^= (uint64_t)0x1F << (8 * (tail % 8));
        if (w == RATE_WORDS - 1) m ^= (uint64_t)0x80 << 56;
        st[w] ^= m;
      }
      keccak_f_thread(st);
    }
    __syncwarp();
  }
}

static void check_rows(const torch::Tensor& t, const char* name, int64_t rows) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kUInt8 && t.dim() == 2, name, " must be a 2-D uint8 CUDA tensor");
  TORCH_CHECK(t.size(0) == rows || (t.size(0) == 1 && t.stride(0) == 0) || t.size(1) == 0, name, " has the wrong row count");
  TORCH_CHECK(t.size(1) == 0 || t.stride(1) == 1, name, " rows must be contiguous");
}

static unsigned grid_of(int64_t rows) { return (unsigned)((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK); }

// out (^)= SHAKE256(prefix || body)[0:out.size(1)] row by row, on the current stream.  `prefix` may have stride(0) == 0.
void shake256_rows(torch::Tensor prefix, torch::Tensor body, torch::Tensor out, bool xor_out) {
  const int64_t rows = out.size(0);
  check_rows(prefix, "prefix", rows);
  check_rows(body, "body", rows);
  check_rows(out, "out", rows);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(out.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  shake256_kernel<<<grid_of(rows), 32 * WARPS_PER_BLOCK, 0, stream>>>(
      prefix.size(1) ? prefix.data_ptr<uint8_t>() : nullptr, (int)prefix.size(1), prefix.size(0) == 1 ? 0 : prefix.stride(0),
      body.data_ptr<uint8_t>(), body.size(1), body.stride(0), out.data_ptr<uint8_t>(), out.size(1), out.stride(0),
      xor_out ? 1 : 0, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Feistel-10 in place on the rows of x ([R, 2h]) with round functions SHAKE256(domain || u8(i) || tweak[r] || .): one warp
// per row below `thread_rows` rows (latency), one thread per row from there (throughput).
void feistel_rows(torch::Tensor x, torch::Tensor tweak, int64_t domain, bool inverse, int64_t thread_rows, bool word_io,
                  bool fused, bool pair) {
  const int64_t rows = x.size(0);
  check_rows(x, "x", rows);
  check_rows(tweak, "tweak", rows);
  TORCH_CHECK(x.size(1) % 2 == 0, "a balanced Feistel network needs an even width");
  TORCH_CHECK(tweak.size(1) <= MAX_TWEAK, "tweak longer than ", MAX_TWEAK, " bytes");
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  const bool words = word_io && tweak.size(1) == 0 && x.size(1) % 16 == 0 && x.stride(0) % 8 == 0 &&
                     reinterpret_cast<uintptr_t>(x.data_ptr<uint8_t>()) % 8 == 0;
  if (thread_rows > 0 && rows >= thread_rows && words && fused && pair && !inverse) {
    const int threads = 128;
    feistel_pair_kernel<<<(unsigned)((2 * rows + threads - 1) / threads), threads, 0, stream>>>(
        x.data_ptr<uint8_t>(), x.stride(0), x.size(1) / 2, (int)domain, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (thread_rows > 0 && rows >= thread_rows && words && fused && !inverse) {
    const int threads = 128;
    feistel_fused_kernel<<<(unsigned)((rows + threads - 1) / threads), threads, 0, stream>>>(
        x.data_ptr<uint8_t>(), x.stride(0), x.size(1) / 2, (int)domain, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (thread_rows > 0 && rows >= thread_rows && words) {
    const int threads = 128;
    feistel_words_kernel<<<(unsigned)((rows + threads - 1) / threads), threads, 0, stream>>>(
        x.data_ptr<uint8_t>(), x.stride(0), x.size(1) / 2, (int)domain, inverse ? 1 : 0, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (thread_rows > 0 && rows >= thread_rows) {
    const int threads = 128;
    feistel_thread_kernel<<<(unsigned)((rows + threads - 1) / threads), threads, 0, stream>>>(
        x.data_ptr<uint8_t>(), x.stride(0), x.size(1) / 2, tweak.size(1) ? tweak.data_ptr<uint8_t>() : nullptr,
        (int)tweak.size(1), tweak.size(0) == 1 ? 0 : tweak.stride(0), (int)domain, inverse ? 1 : 0, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  feistel_kernel<<<grid_of(rows), 32 * WARPS_PER_BLOCK, 0, stream>>>(
      x.data_ptr<uint8_t>(), x.stride(0), x.size(1) / 2, tweak.size(1) ? tweak.data_ptr<uint8_t>() : nullptr, (int)tweak.size(1),
      tweak.size(0) == 1 ? 0 : tweak.stride(0), (int)domain, inverse ? 1 : 0, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- the audit responder ----------------------------------------------------------------------------------------------
// Wire (little-endian): request = u64 op, u64 arg.  op 1 = challenge block `arg`, op 2 = ping (no device work), op 3 = close.
// Response = u64 status (0 ok, 1 index out of range, 2 CUDA error), u64 device_ns (copy start to end event), u64 server_ns
// (request received to response ready), then `block_bytes` payload bytes.

// The connection is non-blocking and polled: the thread never sleeps on the socket, so no wake-up sits on the answer's path.
static bool read_all(int fd, void* buf, size_t n, const std::atomic<bool>& running) {
  auto* p = static_cast<uint8_t*>(buf);
  while (n) {
    ssize_t r = ::recv(fd, p, n, MSG_DONTWAIT);
    if (r > 0) {
      p += r;
      n -= (size_t)r;
    } else if (r == 0 || (errno != EAGAIN && errno != EWOULDBLOCK) || !running.load(std::memory_order_relaxed)) {
      return false;
    }
  }
  return true;
}

static bool write_all(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const uint8_t*>(buf);
  while (n) {
    ssize_t r = ::send(fd, p, n, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (r > 0) {
      p += r;
      n -= (size_t)r;
    } else if (r == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
      return false;
    }
  }
  return true;
}

class Responder {
 public:
  Responder(int64_t base, int64_t blocks, int64_t block_bytes, int device, std::string host, int port, int cpu)
      : base_(reinterpret_cast<const uint8_t*>(base)), blocks_(blocks), block_bytes_(block_bytes), device_(device),
        host_(std::move(host)), port_(port), cpu_(cpu) {}
  ~Responder() { stop(); }

  int start() {
    TORCH_CHECK(!running_.load(), "responder already running");
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    TORCH_CHECK(listen_fd_ >= 0, "socket() failed");
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port_);
    TORCH_CHECK(::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) == 1, "bad host ", host_);
    TORCH_CHECK(::bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) == 0, "bind() failed on ", host_, ":", port_);
    TORCH_CHECK(::listen(listen_fd_, 4) == 0, "listen() failed");
    socklen_t len = sizeof(addr);
    ::getsockname(listen_fd_, (sockaddr*)&addr, &len);
    port_ = ntohs(addr.sin_port);
    running_.store(true);
    thread_ = std::thread([this] { serve(); });
    return port_;
  }

  void stop() {
    if (!running_.exchange(false)) return;
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    int c = conn_fd_.load();
    if (c >= 0) ::shutdown(c, SHUT_RDWR);
    if (thread_.joinable()) thread_.join();
  }

  int64_t answered() const { return answered_.load(); }
  int64_t pings() const { return pings_.load(); }
  int port() const { return port_; }

 private:
  void serve() {
    if (cpu_ >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu_, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    cudaSetDevice(device_);
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    uint8_t* host = nullptr;
    cudaHostAlloc((void**)&host, (size_t)block_bytes_ + 24, cudaHostAllocDefault);
    std::memset(host, 0, (size_t)block_bytes_ + 24);
    while (running_.load()) {
      int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) continue;
      int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
      conn_fd_.store(fd);
      uint64_t req[2];
      while (running_.load() && read_all(fd, req, sizeof(req), running_)) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t* head = reinterpret_cast<uint64_t*>(host);
        uint8_t* payload = host + 24;
        head[0] = 0;
        head[1] = 0;
        if (req[0] == 3) break;
        if (req[0] == 1) {
          if (req[1] >= (uint64_t)blocks_) {
            head[0] = 1;
          } else {
            cudaEventRecord(ev0, stream);
            cudaMemcpyAsync(payload, base_ + (size_t)req[1] * (size_t)block_bytes_, (size_t)block_bytes_,
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(ev1, stream);
            cudaError_t st;
            while ((st = cudaEventQuery(ev1)) == cudaErrorNotReady) {
            }
            if (st != cudaSuccess) {
              head[0] = 2;
            } else {
              float ms = 0.f;
              cudaEventElapsedTime(&ms, ev0, ev1);
              head[1] = (uint64_t)((double)ms * 1e6);
            }
            answered_.fetch_add(1);
          }
        } else {
          pings_.fetch_add(1);
        }
        head[2] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        if (!write_all(fd, host, (size_t)block_bytes_ + 24)) break;
      }
      conn_fd_.store(-1);
      ::close(fd);
    }
    cudaFreeHost(host);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
  }

  const uint8_t* base_;
  int64_t blocks_, block_bytes_;
  int device_;
  std::string host_;
  int port_;
  int cpu_;
  int listen_fd_ = -1;
  std::atomic<int> conn_fd_{-1};
  std::atomic<bool> running_{false};
  std::atomic<int64_t> answered_{0}, pings_{0};
  std::thread thread_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shake256_rows", &shake256_rows, "out (^)= SHAKE256(prefix || body)[0:out.size(1)], row by row");
  m.def("feistel_rows", &feistel_rows, "Feistel-10 in place, SHAKE256 round functions under a domain byte and per-row tweak",
        pybind11::arg("x"), pybind11::arg("tweak"), pybind11::arg("domain"), pybind11::arg("inverse"), pybind11::arg("thread_rows") = 0,
        pybind11::arg("word_io") = false, pybind11::arg("fused") = false, pybind11::arg("pair") = false);
  pybind11::class_<Responder>(m, "Responder")
      .def(pybind11::init<int64_t, int64_t, int64_t, int, std::string, int, int>())
      .def("start", &Responder::start)
      .def("stop", &Responder::stop)
      .def("answered", &Responder::answered)
      .def("pings", &Responder::pings)
      .def("port", &Responder::port);
}
