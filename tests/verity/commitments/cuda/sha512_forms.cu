// Every compression form of ../sha512.cuh as SHA-512 of whole messages, each chunk through one form, for test_sha512.py:
// built by a host compiler (`-x c++`) it runs them on the host, and built by nvcc also on the device, a thread a message.
#include <cstdio>
#include <utility>

#include "../sha512.cuh"

namespace {

// sx's pipe choices under test: all on the ALU, all mode 1, all mode 2, all mode 3, and two mixes
SX_HD constexpr int v_of(int j) {
  return j == 0 ? 0x00 : j == 1 ? 0x55 : j == 2 ? 0xaa : j == 3 ? 0xff : j == 4 ? 0x1b : 0xe4;
}
SX_HD constexpr int r0_of(int j) { return j == 0 ? 1 : j == 1 ? 8 : 15; }
constexpr int NV = 6, NR0 = 3;
// sx::compress<V, ROLL> (ROLL 0..2), sx::compress_kw<V, ROLL> over kw_of (ROLL 0..1), sha::compress, sha::compress_kw over
// schedule_kw, sha::compress_compact, sha::compress_from<R0> from rounds_prefix
constexpr int SX = 3 * NV, KW = 2 * NV, SHA = SX + KW, FORMS = SHA + 3 + NR0;

template <int I>
SX_HD void form(uint64_t* st, const uint64_t* blk) {
  uint64_t w[16];
  for (int t = 0; t < 16; ++t) w[t] = blk[t];
  if constexpr (I < SX) {
    sx::compress<v_of(I / 3), I % 3>(st, w);
  } else if constexpr (I < SX + KW) {
    constexpr int J = I - SX;
    uint64_t kw[80];
    sx::kw_of(w, kw);
    sx::compress_kw<v_of(J / 2), J % 2>(st, kw);
  } else if constexpr (I == SHA) {
    sha::compress(st, w);
  } else if constexpr (I == SHA + 1) {
    uint64_t kw[80];
    sha::schedule_kw(blk, kw);
    sha::compress_kw(st, kw);
  } else if constexpr (I == SHA + 2) {
    sha::compress_compact(st, blk);
  } else {
    constexpr int R0 = r0_of(I - SHA - 3);
    uint64_t m[8];
    sha::rounds_prefix(st, blk, R0, m);
    sha::compress_from<R0>(st, m, w);
  }
}

template <int I>
SX_HD void digest(const uint8_t* msg, uint64_t len, uint8_t* out) {
  uint64_t st[8], blk[16];
  sx::iv(st);
  const uint64_t total = (len + 17 + 127) / 128 * 128;
  for (uint64_t off = 0; off < total; off += 128) {
    for (int i = 0; i < 16; ++i) {
      uint64_t v = 0;
      for (int b = 0; b < 8; ++b) {
        const uint64_t j = off + 8 * i + b;
        v = v << 8 | (j < len ? msg[j] : j == len ? 0x80u : 0u);
      }
      blk[i] = v;
    }
    if (off + 128 == total) blk[15] = len * 8;
    form<I>(st, blk);
  }
  for (int i = 0; i < 8; ++i) sha::put_be64(out + 8 * i, st[i]);
}

using Digest = void (*)(const uint8_t*, uint64_t, uint8_t*);
template <int... I>
const Digest* host_digests(std::integer_sequence<int, I...>) {
  static const Digest d[] = {&digest<I>...};
  return d;
}

}  // namespace

extern "C" int sha512_forms() { return FORMS; }

extern "C" const char* sha512_form_name(int f) {
  static char s[64];
  if (f < SX) snprintf(s, sizeof s, "sx::compress<0x%02x, %d>", v_of(f / 3), f % 3);
  else if (f < SX + KW) snprintf(s, sizeof s, "sx::compress_kw<0x%02x, %d>", v_of((f - SX) / 2), (f - SX) % 2);
  else if (f == SHA) snprintf(s, sizeof s, "sha::compress");
  else if (f == SHA + 1) snprintf(s, sizeof s, "sha::compress_kw");
  else if (f == SHA + 2) snprintf(s, sizeof s, "sha::compress_compact");
  else snprintf(s, sizeof s, "sha::compress_from<%d>", r0_of(f - SHA - 3));
  return s;
}

// SHA-512 of msg with every chunk through form f; form -1 is sha::digest
extern "C" void sha512_digest(int f, const uint8_t* msg, uint64_t len, uint8_t* out) {
  if (f < 0) sha::digest(msg, len, out);
  else host_digests(std::make_integer_sequence<int, FORMS>{})[f](msg, len, out);
}

// 0 when kw_pad256's table is kw_of over the chunk that ends a 256-byte message and, in a host build, IV_DEV is iv's
extern "C" int sha512_tables() {
  uint64_t pad[16] = {0x8000000000000000ull}, kw[80];
  pad[15] = 2048;
  sx::kw_of(pad, kw);
  int bad = 0;
  for (int t = 0; t < 80; ++t) bad += kw[t] != sx::kw_pad256()[t];
#ifndef __CUDACC__
  uint64_t st[8];
  sx::iv(st);
  for (int i = 0; i < 8; ++i) bad += st[i] != sx::IV_DEV[i];
#endif
  return bad;
}

#ifdef __CUDACC__
namespace {

template <int I>
__global__ void digests(const uint8_t* msgs, const uint64_t* offs, const uint64_t* lens, int n, uint8_t* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) digest<I>(msgs + offs[i], lens[i], out + 64 * i);
}

using Kernel = void (*)(const uint8_t*, const uint64_t*, const uint64_t*, int, uint8_t*);
template <int... I>
const Kernel* kernels(std::integer_sequence<int, I...>) {
  static const Kernel k[] = {&digests<I>...};
  return k;
}

}  // namespace

// sha512_digest's on the device for n messages (msgs + offs[i], lens[i] bytes) at once: 0, or the CUDA error
extern "C" int sha512_digest_cuda(int f, const uint8_t* msgs, uint64_t nbytes, const uint64_t* offs, const uint64_t* lens,
                                  int n, uint8_t* out) {
  uint8_t *dm = nullptr, *dout = nullptr;
  uint64_t *doffs = nullptr, *dlens = nullptr;
  cudaError_t e = cudaMalloc(&dm, nbytes ? nbytes : 1);
  if (!e) e = cudaMalloc(&doffs, 8 * n);
  if (!e) e = cudaMalloc(&dlens, 8 * n);
  if (!e) e = cudaMalloc(&dout, 64 * n);
  if (!e) e = cudaMemcpy(dm, msgs, nbytes, cudaMemcpyHostToDevice);
  if (!e) e = cudaMemcpy(doffs, offs, 8 * n, cudaMemcpyHostToDevice);
  if (!e) e = cudaMemcpy(dlens, lens, 8 * n, cudaMemcpyHostToDevice);
  if (!e) {
    kernels(std::make_integer_sequence<int, FORMS>{})[f]<<<(n + 63) / 64, 64>>>(dm, doffs, dlens, n, dout);
    e = cudaGetLastError();
  }
  if (!e) e = cudaMemcpy(out, dout, 64 * n, cudaMemcpyDeviceToHost);
  cudaFree(dm); cudaFree(doffs); cudaFree(dlens); cudaFree(dout);
  return (int)e;
}
#endif
