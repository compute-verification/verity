// Which CTA shapes ptxas builds for a hash warpgroup beside two wgmma s8 warpgroups: each kernel is its own nvcc run
// (-DPROBE=name), so one refusal leaves the others' verdicts. P producer registers, Y each MMA warpgroup's, H the hash's.
#include <cstdint>

#include "../s8_linear/s8_acc.h"

#define DEV __device__ __forceinline__
template <int R> DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
DEV uint32_t smem_at(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
template <int N> DEV void imma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  if constexpr (N == 256)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
                 ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 " ACC64_S
                 ", %64, %65, p;\n}" : ACC64_R : "l"(da), "l"(db), "r"(sd));
}

template <int N, int P, int Y, int H>
DEV void body(int32_t* out, int iters) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int wg = threadIdx.x / 128;
  if (wg == 0) {
    regs_dec<P>();
    asm volatile("exit;");
  }
  if (wg == 3) {
    if constexpr (H < 128) regs_dec<H>();
    uint64_t x = threadIdx.x;
    for (int i = 0; i < iters; ++i) x = (x << 7 | x >> 57) ^ (x * 0x9e3779b97f4a7c15ull);
    out[threadIdx.x] = (int32_t)x;
    asm volatile("exit;");
  }
  regs_inc<Y>();
  int32_t d[128];
  const uint32_t a0 = smem_at(smem) + (wg - 1) * 8192, b0 = smem_at(smem) + 16384;
  for (int i = 0; i < iters; ++i) {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    if constexpr (N == 256) {
      imma<256>(d, desc_sw128(a0), desc_sw128(b0), i > 0);
    } else {
      imma<128>(d, desc_sw128(a0), desc_sw128(b0), i > 0);
      imma<128>(d + 64, desc_sw128(a0), desc_sw128(b0 + 16384), i > 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
#pragma unroll
  for (int i = 0; i < 128; i++) asm volatile("" : "+r"(d[i])::"memory");
#pragma unroll
  for (int i = 0; i < 128; i++) out[i * 512 + threadIdx.x] = d[i];
}

#define K512(name, N, P, Y, H) \
  extern "C" __global__ void __launch_bounds__(512, 1) name(int32_t* out, int it) { body<N, P, Y, H>(out, it); }
#define K384(name, N, P, Y) \
  extern "C" __global__ void __launch_bounds__(384, 1) name(int32_t* out, int it) { body<N, P, Y, 128>(out, it); }

#if PROBE == 1
K384(c384_n256, 256, 40, 232)
#elif PROBE == 2
K512(w512_n256_h64, 256, 24, 208, 64)
#elif PROBE == 3
K512(w512_n128x2_h64, 128, 24, 208, 64)
#elif PROBE == 4
K512(w512_n256_h96, 256, 32, 192, 96)
#elif PROBE == 5
K512(w512_n128x2_h96, 128, 32, 192, 96)
#endif
