// P2's second encoder as a library of its own for ctypes, libp2enc.so: p2enc.cuh's kernel behind p2v1.cu's p2v1_encode
// arguments. It is apart from libp2v1.so, whose sources and build stay as they are: a kernel added to p2v1.cu's
// translation unit changes the instructions nvcc gives decode_sq_kernel. p2sq.cuh ends in that kernel, so this
// library holds a copy of it that nothing calls.
// Build: nvcc -O3 -arch=sm_NN -std=c++17 -DP2SQ_KARA -Xcompiler -fPIC -shared -o libp2enc.so p2enc.cu
#include <algorithm>
#include <cstdint>

#include "p2enc.cuh"

// p2v1_encode on p2enc.cuh's kernel: the same rows, bytes and flags, (ea, en, er) being the exponent as gpu.exponent_split
// gives it. -1 when the exponent's steps don't fit a Chain, else CUDA's error
extern "C" int p2enc_encode(const void *m, void *c, const void *k, const void *t, uint32_t n, uint32_t c0, uint32_t ea,
                            uint32_t en, uint32_t er, void *flags, void *stream) {
  p2sq::Chain chain;
  if (!p2sq::chain_for(ea, en, er, chain)) return -1;
  if (n == 0) return 0;
  int dev = 0, sms = 0, occ = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void *)encode_sq_kernel, 32 * P2SQ_WARPS, 0);
  uint64_t full = ((uint64_t)n + P2SQ_WARPS - 1) / P2SQ_WARPS, cap = (uint64_t)(occ > 0 ? occ : 1) * (sms > 0 ? sms : 1);
  encode_sq_kernel<<<(uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(full, cap)), 32 * P2SQ_WARPS, 0, (cudaStream_t)stream>>>(
      chain, (const uint32_t *)m, (uint32_t *)c, (const uint32_t *)k, (const uint32_t *)t, n, c0, (uint32_t *)flags);
  return (int)cudaGetLastError();
}

// How many CTAs of the kernel one SM runs at once, by the driver
extern "C" int p2enc_ctas_per_sm() {
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void *)encode_sq_kernel, 32 * P2SQ_WARPS, 0);
  return occ;
}
