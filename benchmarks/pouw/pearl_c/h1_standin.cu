// The decode GEMM's memory traffic, for timing -h1's hashing around it on sm_120 (h1_bench.py). It is a stand-in, not a GEMM:
// it reads the n x k FP8 weights once (16 bytes a load, a warp per weight row) and writes C~, U (m x n FP32 words) and y
// (m x n BF16), each a function of the weight row read and of A's rows, so that every output depends on both; and
// derive_rows, the next call's A (m x k BF16, k = n) from y, which makes the calls a dependent chain.
#include <cstdint>

struct Args { uint64_t p[16]; int32_t i[16]; };

// p0 W (n x k bytes), p1 A (m x k BF16), p2 C~ (m x n words), p3 U, p4 y (m x n BF16); i0 m, i1 n, i2 k
extern "C" __global__ void __launch_bounds__(256) standin_gemm(const Args a) {
  const int m = a.i[0], n = a.i[1], k = a.i[2];
  const int j = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (j >= n) return;
  const uint4* w = reinterpret_cast<const uint4*>(a.p[0] + (uint64_t)j * k);
  uint32_t acc = 0;
  for (int q = lane; q < k / 16; q += 32) {
    const uint4 v = w[q];
    acc ^= (v.x * 0x9E3779B1u + v.y) ^ (v.z * 0x85EBCA77u + v.w);
  }
  for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
  const uint32_t* A = reinterpret_cast<const uint32_t*>(a.p[1]);
  for (int r = lane; r < m; r += 32) {
    const uint32_t x = acc ^ A[(uint64_t)r * (k / 2) + (j % (k / 2))] ^ (uint32_t)r * 0x27D4EB2Du;
    reinterpret_cast<uint32_t*>(a.p[2])[(uint64_t)r * n + j] = x;
    reinterpret_cast<uint32_t*>(a.p[3])[(uint64_t)r * n + j] = x * 0x165667B1u;
    reinterpret_cast<uint16_t*>(a.p[4])[(uint64_t)r * n + j] = (uint16_t)(x >> 16);
  }
}

// p0 y (count BF16), p1 the next A (count BF16); i0 count
extern "C" __global__ void __launch_bounds__(256) derive_rows(const Args a) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (uint64_t)a.i[0]) return;
  reinterpret_cast<uint16_t*>(a.p[1])[t] = reinterpret_cast<const uint16_t*>(a.p[0])[t] ^ 0x3C00u;
}

// A read-only L2 flush: every thread sums its words of the buffer and writes one word only if the sum is a value it never is,
// so the flush leaves no dirty line.  p0 buffer, p1 sink; i0 words
extern "C" __global__ void __launch_bounds__(256) l2_flush(const Args a) {
  const uint32_t* b = reinterpret_cast<const uint32_t*>(a.p[0]);
  uint32_t s = 0;
  for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < (uint64_t)(uint32_t)a.i[0]; t += (uint64_t)gridDim.x * blockDim.x)
    s += b[t];
  if (s == 0x9E3779B9u) reinterpret_cast<uint32_t*>(a.p[1])[0] = s;
}
