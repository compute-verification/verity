// ffill's conformance on the H200 (sm_90a): the GPU's fma.rn.f32 and the filler chain (ffill.cuh), dumped for
// ffill_ref.py, which recomputes every word exactly (F32FmaV2: one rounding, subnormals kept, every NaN 0x7FFFFFFF).
//
//   ffill fma DIR N SEED          DIR/fma.bin: u32 quads (a, b, c, fma.rn.f32(a, b, c)) over the fixed cases (specials,
//                                 signed zeros, subnormals, ties, overflow, the double-rounding case) and N seeded ones
//                                 (every word; near binades, so cancellation; filler operands against ±s)
//   ffill chain DIR T SEGS SEED   DIR/chain.bin: T threads' chains over SEGS segments (3 steps each, FfChain::mma, the
//                                 kernel's path), coins derived on the device from (c0, c1, tile); the words are edge
//                                 cases on the first threads (0, ±(2^29 - 1), alternating, the exact range, any s32) and
//                                 seeded in ±2^27 on the rest
//   python3 ffill_ref.py check DIR
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a ffill.cu -o ffill   (ffill.sh; no --use_fast_math or
// -ftz, which ff_fma's explicit fma.rn.f32 ignores anyway)
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ffill.cuh"

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

__global__ void fma_kernel(const uint32_t* q, uint32_t* d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = __float_as_uint(ff_fma(__uint_as_float(q[3 * i]), __uint_as_float(q[3 * i + 1]),
                                           __uint_as_float(q[3 * i + 2])));
}

__global__ void chain_kernel(uint64_t c0, uint64_t c1, uint64_t tile, const int32_t* words, int threads, int segs,
                             uint32_t one, uint32_t* out, uint32_t* coins) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= threads) return;
  uint32_t c[4];
  ff_coins(c0, c1, tile, c);
  if (u == 0)
    for (int i = 0; i < 4; i++) coins[i] = c[i];
  FfChain ff(c, one);
  const uint32_t* w = reinterpret_cast<const uint32_t*>(words) + (size_t)u * FF_STEPS * segs;
  for (int m = 0; m < segs; m++, w += FF_STEPS) {
    if (m & 1) ff.mma<1>(w[0], w[1], w[2]);
    else ff.mma<0>(w[0], w[1], w[2]);
  }
  out[2 * u] = __float_as_uint(ff.x);
  out[2 * u + 1] = __float_as_uint(ff.y);
}

struct Rng {
  uint64_t k;
  uint64_t next() { return k = ff_smix(k); }
  uint32_t u32() { return (uint32_t)(next() >> 32); }
};

static void put(std::vector<uint32_t>& q, uint32_t a, uint32_t b, uint32_t c) {
  q.push_back(a);
  q.push_back(b);
  q.push_back(c);
}

static std::vector<uint32_t> fma_cases(int n, uint64_t seed) {
  std::vector<uint32_t> q;
  const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x00000001u, 0x80000001u, 0x007FFFFFu,
                         0x00800000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7F800001u,
                         0xFFC00001u, 0x7FFFFFFFu, 0x3F000000u, 0x4B400000u, 0x3F800001u, 0x33800000u, 0xB3800000u};
  for (uint32_t a : sp)
    for (uint32_t b : sp)
      for (uint32_t c : sp) put(q, a, b, c);
  put(q, 0x40400000u, 0x3F800001u, 0xA1800000u);  // ffill_ref.DOUBLE_ROUNDING: 0x40400001, not float64's 0x40400002
  put(q, 0x3F800001u, 0x3F800000u, 0x33800000u);  // 1 + 2^-23 + 2^-24: a tie, to even 0x3F800002
  put(q, 0x3F800000u, 0x3F800000u, 0x33800000u);  // 1 + 2^-24: a tie, to even 0x3F800000
  put(q, 0x00800000u, 0x3F000000u, 0x00000000u);  // 2^-127: subnormal, kept
  put(q, 0x00800000u, 0x3F7FFFFFu, 0x80000000u);  // just below the normal range
  put(q, 0x7F7FFFFFu, 0x40000000u, 0x00000000u);  // overflow to +inf
  put(q, 0x3F800000u, 0x3F800000u, 0xBF800000u);  // 1 - 1: +0
  put(q, 0x80000000u, 0x3F800000u, 0x80000000u);  // -0 + -0: -0
  Rng r{seed};
  for (int i = 0; i < n; i++) {
    const uint32_t a = r.u32(), b = r.u32(), c = r.u32();
    switch (i & 3) {
      case 0: put(q, a, b, c); break;
      case 1: {  // the product's binade and the addend's within a few of each other: cancellation, every rounding case
        const uint32_t ea = 96 + (a >> 26), eb = 96 + (b >> 26), ec = ea + eb - 130 + (c >> 29);
        put(q, (a & 0x807FFFFFu) | (ea << 23), (b & 0x807FFFFFu) | (eb << 23), (c & 0x807FFFFFu) | (ec << 23));
        break;
      }
      case 2: {  // results near and below 2^-126: subnormal outputs
        const uint32_t ea = 1 + (a >> 26), eb = 64 - (b >> 27) + (r.u32() & 31);
        put(q, (a & 0x807FFFFFu) | (ea << 23), (b & 0x807FFFFFu) | (eb << 23), c & 0x80FFFFFFu);
        break;
      }
      default: {  // a filler step: f = BIAS + w, b = ±s, c a chain value of f's size or a difference's
        const int32_t w = (int32_t)(a & 0x3FFFFFFFu) - (1 << 29);
        const uint32_t s = 0x3F800001u | (b & 0x7FFFFEu) | (b & 0x80000000u);
        const uint32_t cm = (c & 0x807FFFFFu) | ((127 + (c >> 23 & 31)) << 23);
        put(q, (uint32_t)w + FF_BIAS, s, cm);
      }
    }
  }
  return q;
}

static int write_file(const std::string& path, const void* p, size_t bytes) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(p, 1, bytes, f) != bytes) {
    fprintf(stderr, "ERR cannot write %s\n", path.c_str());
    return 2;
  }
  fclose(f);
  return 0;
}

static int run_fma(const std::string& dir, int n, uint64_t seed) {
  const std::vector<uint32_t> q = fma_cases(n, seed);
  const int m = (int)(q.size() / 3);
  uint32_t *dq, *dd;
  CK(cudaMalloc(&dq, q.size() * 4));
  CK(cudaMalloc(&dd, (size_t)m * 4));
  CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
  fma_kernel<<<(m + 255) / 256, 256>>>(dq, dd, m);
  CK(cudaGetLastError());
  std::vector<uint32_t> d(m), out(4 * (size_t)m);
  CK(cudaMemcpy(d.data(), dd, (size_t)m * 4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < m; i++) {
    out[4 * i] = q[3 * i];
    out[4 * i + 1] = q[3 * i + 1];
    out[4 * i + 2] = q[3 * i + 2];
    out[4 * i + 3] = d[i];
  }
  if (write_file(dir + "/fma.bin", out.data(), out.size() * 4)) return 2;
  printf("{\"mode\": \"fma\", \"cases\": %d, \"seed\": %llu}\n", m, (unsigned long long)seed);
  return 0;
}

static int run_chain(const std::string& dir, int threads, int segs, uint64_t seed) {
  const int steps = FF_STEPS * segs;
  const uint64_t c0 = ff_smix(seed), c1 = ff_smix(seed ^ 1), tile = ff_smix(seed ^ 2) & 0xFFFFF;
  std::vector<int32_t> w((size_t)threads * steps);
  Rng r{seed ^ 3};
  const int32_t big = (1 << 29) - 1;
  for (int u = 0; u < threads; u++)
    for (int n = 0; n < steps; n++) {
      int32_t v;
      switch (u) {
        case 0: v = 0; break;
        case 1: v = big; break;
        case 2: v = -big; break;
        case 3: v = n & 1 ? -big : big; break;
        case 4: v = (int32_t)(r.u32() & 0x7FFFFF) - (1 << 22); break;
        case 5: v = (int32_t)r.u32(); break;
        default: v = (int32_t)(r.u32() & 0x0FFFFFFF) - (1 << 27);
      }
      w[(size_t)u * steps + n] = v;
    }
  int32_t* dw;
  uint32_t *dout, *dco;
  CK(cudaMalloc(&dw, w.size() * 4));
  CK(cudaMalloc(&dout, (size_t)threads * 8));
  CK(cudaMalloc(&dco, 16));
  CK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  chain_kernel<<<(threads + 127) / 128, 128>>>(c0, c1, tile, dw, threads, segs, 1u, dout, dco);
  CK(cudaGetLastError());
  std::vector<uint32_t> out(2 * (size_t)threads), co(4);
  CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(co.data(), dco, 16, cudaMemcpyDeviceToHost));
  std::vector<unsigned char> f(48 + w.size() * 4 + out.size() * 4);
  const uint64_t h64[3] = {c0, c1, tile};
  const uint32_t h32[2] = {(uint32_t)threads, (uint32_t)steps};
  memcpy(f.data(), h64, 24);
  memcpy(f.data() + 24, h32, 8);
  memcpy(f.data() + 32, co.data(), 16);
  memcpy(f.data() + 48, w.data(), w.size() * 4);
  memcpy(f.data() + 48 + w.size() * 4, out.data(), out.size() * 4);
  if (write_file(dir + "/chain.bin", f.data(), f.size())) return 2;
  printf("{\"mode\": \"chain\", \"threads\": %d, \"steps\": %d, \"seed\": %llu}\n", threads, steps,
         (unsigned long long)seed);
  return 0;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (!((mode == "fma" && argc == 5) || (mode == "chain" && argc == 6))) {
    fprintf(stderr, "usage: ffill fma DIR N SEED | ffill chain DIR THREADS SEGS SEED\n");
    return 2;
  }
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  if (p.major != 9) {
    fprintf(stderr, "ERR device sm_%d%d: ffill is the H200's filler (sm_90a)\n", p.major, p.minor);
    return 2;
  }
  if (mode == "fma") return run_fma(argv[2], atoi(argv[3]), strtoull(argv[4], nullptr, 10));
  return run_chain(argv[2], atoi(argv[3]), atoi(argv[4]), strtoull(argv[5], nullptr, 10));
}
