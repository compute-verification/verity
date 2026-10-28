// SHA-512 compressions for CTAs beside the GEMM: each thread chains reps compressions of sha512_rows' sx::compress<0, 1>,
// feeding its chaining value back into the next chunk, and each CTA stamps its start, end, clock64 and SM. Forms 0 and 1
// fit beside a GEMM CTA launched at 160 registers a thread (4,096 of the register file left): one warp at 128 registers or
// two at 64. Forms 2..4 (128, 256 and 512 threads at their launch bounds) are the hash alone.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "sha512x.cuh"

namespace {

using u64 = unsigned long long;
using u32 = uint32_t;

struct HStamp {
  u64 t0, t1, c0, c1;
  u32 sm, pad;
};
static_assert(sizeof(HStamp) == 40, "HStamp is bench.py's 5 u64 words");

__device__ __forceinline__ u64 gtimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ u32 smid() {
  u32 s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

__device__ __forceinline__ void chain(u64* io, int reps, HStamp* out) {
  if (threadIdx.x == 0) {
    out[blockIdx.x].t0 = gtimer();
    out[blockIdx.x].c0 = clock64();
  }
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 st[8], w[16];
  sx::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sx::compress<0, 1>(st, w);
  }
  if (st[0] == 0x0123456789abcdefull) io[t] = st[1];
  __syncthreads();
  if (threadIdx.x == 0) {
    out[blockIdx.x].c1 = clock64();
    out[blockIdx.x].t1 = gtimer();
    out[blockIdx.x].sm = smid();
  }
}

template <int T>
__global__ void __launch_bounds__(T, 1) hash_k(u64* io, int reps, HStamp* out) {
  chain(io, reps, out);
}

__global__ void __maxnreg__(128) hash_k32r128(u64* io, int reps, HStamp* out) { chain(io, reps, out); }
__global__ void __maxnreg__(64) hash_k64r64(u64* io, int reps, HStamp* out) { chain(io, reps, out); }

// spins until *counter reaches want: the hash stream's wait for every GEMM CTA to be resident
__global__ void wait_k(const volatile unsigned* counter, unsigned want) {
  while (*counter < want) __nanosleep(1000);
}

using HashFn = void (*)(u64*, int, HStamp*);

HashFn pick(int form, int* threads) {
  switch (form) {
    case 0: *threads = 32; return hash_k32r128;
    case 1: *threads = 64; return hash_k64r64;
    case 2: *threads = 128; return hash_k<128>;
    case 3: *threads = 256; return hash_k<256>;
    case 4: *threads = 512; return hash_k<512>;
    default: *threads = 0; return nullptr;
  }
}

}  // namespace

extern "C" {

// A hash form's threads, registers and local (spill) bytes as JSON; forms 0..4.
int u2_hash_info(int form, char* buf, int n) {
  int threads = 0;
  HashFn f = pick(form, &threads);
  if (!f) return -1;
  cudaFuncAttributes a{};
  cudaFuncGetAttributes(&a, f);
  return snprintf(buf, n, "{\"form\": %d, \"threads\": %d, \"regs\": %d, \"local_bytes\": %zu}", form, threads, a.numRegs,
                  a.localSizeBytes);
}

int u2_hash_run(int form, int blocks, int reps, void* io, void* stamps, void* stream) {
  int threads = 0;
  HashFn f = pick(form, &threads);
  if (!f) return -1;
  f<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<u64*>(io), reps, static_cast<HStamp*>(stamps));
  return (int)cudaGetLastError();
}

int u2_wait(void* counter, unsigned want, void* stream) {
  wait_k<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const volatile unsigned*>(counter), want);
  return (int)cudaGetLastError();
}

}  // extern "C"
