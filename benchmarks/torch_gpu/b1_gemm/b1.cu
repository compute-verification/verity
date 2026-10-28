// b1 (AND.POPC) warpgroup MMAs on the H100 and H200 (sm_90a): wgmma m64n256k256 .s32.b1.b1.and.popc, BGMMA.64x256x256,
// d += popc(a AND b) over 256 bits a row pair an instruction, at s8 n256's instruction rate (8x its MACs).
//
// b1_res<HW>: one warpgroup's BGMMAs on operands resident in shared memory: one 128-byte-deep stage of A (64 rows) and W
// (256 rows), placed as TMA's SWIZZLE_128B would (16-byte chunk c of row r at chunk c ^ r % 8), four atoms (k 1,024 bits)
// an iteration into one s32 accumulator, `iters` iterations; CTA 0 writes its accumulator (iters x popc(A_m AND W_n)) for
// the host to check.  HW > 0 warps beside the warpgroup chain SHA-512 compressions (sha512_rows' sx::compress) until it
// is done and add their count to `hn`.  Its shared memory request keeps a second probe CTA off an SM, so hash_k CTAs (no
// shared memory) are what shares an SM with it.
//
// hash_k: `reps` compressions a thread on CTAs of their own.  stamp_k: each SM's %clock64 and %globaltimer, one CTA an
// SM, once every CTA of the launch has arrived.  wait_k: holds a stream until a launch's CTAs have all started.
#include <cstddef>
#include <cstdint>
#include <cuda.h>

#include "../s8_linear/s8_acc.h"
#include "../sha512_rows/sha512x.cuh"

#define DEV __device__ __forceinline__

struct Stamp {
  unsigned long long t0, t1, c0, c1, sm, n;
};

DEV unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
DEV uint32_t smid() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
DEV uint32_t smem_at(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

DEV uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
// d = popc(A AND B) + (sd ? d : 0): one b1 atom (k = 256 bits, 32 bytes) of a 64 x 256 warpgroup tile; the accumulator's
// layout is s8 m64n256's
DEV void bgmma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k256.s32.b1.b1.and.popc " ACC128_S
               ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
}
template <int NA> DEV void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}

// accumulator i of thread x of a 64 x 256 tile: row 16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x %
// 4) + i % 2
DEV void store_acc(int32_t* o, int ld, const int32_t* d, int x) {
  int32_t* o0 = o + (size_t)(16 * (x / 32) + (x % 32) / 4) * ld + 2 * (x % 4);
  int32_t* o1 = o0 + (size_t)8 * ld;
#pragma unroll
  for (int j = 0; j < 32; j++) {
    *reinterpret_cast<int2*>(o0 + 8 * j) = make_int2(d[4 * j], d[4 * j + 1]);
    *reinterpret_cast<int2*>(o1 + 8 * j) = make_int2(d[4 * j + 2], d[4 * j + 3]);
  }
}

// one SHA-512 chain a thread: each block's first eight words xored with the chaining value
struct Chain {
  unsigned long long st[8], w[16];
  DEV explicit Chain(unsigned long long seed) {
    sx::iv(st);
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = seed * 16 + i;
  }
  DEV void step() {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sx::compress<0, 1>(st, w);
  }
  // written only for a state no input reaches, so the compressions stay live
  DEV void sink(unsigned long long* io, unsigned long long t) const {
    if (st[0] == 0x0123456789abcdefull) io[t] = st[1];
  }
};

// ---------------------------------------------------------------- resident operands
struct ResParams {
  const uint8_t* a;              // 64 rows of 128 bytes
  const uint8_t* b;              // 256 rows of 128 bytes
  int32_t* acc;                  // CTA 0's 64 x 256 accumulator, or null
  Stamp* st;                     // a stamp a CTA, or null
  unsigned long long* hn;        // hash warps' compressions, added up
  unsigned long long* io;        // their sink
  unsigned* started;             // counted up by each CTA as it starts, or null
  int iters;
};

template <int HW> DEV void res_body(const ResParams& p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ int stop;
  const int tid = threadIdx.x;
  unsigned long long t0 = 0, c0 = 0;
  if (tid == 0) {
    t0 = gtimer();
    c0 = clock64();
    stop = 0;
    if (p.started) atomicAdd(p.started, 1u);
  }
  const uint32_t s0 = smem_at(smem), base = (s0 + 1023) & ~1023u;
  uint8_t* sb = smem + (base - s0);
  for (int i = tid; i < 320 * 8; i += blockDim.x) {
    const int r = i / 8, c = i % 8;
    const uint8_t* src = r < 64 ? p.a + r * 128 + c * 16 : p.b + (r - 64) * 128 + c * 16;
    *reinterpret_cast<uint4*>(sb + r * 128 + (c ^ (r & 7)) * 16) = *reinterpret_cast<const uint4*>(src);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (tid < 128) {
    int32_t d[128] = {};
    const uint64_t da = desc_sw128(base), db = desc_sw128(base + 64 * 128);
    for (int it = 0; it < p.iters; it++) {
      WG_FENCE;
      bgmma(d, da, db, it > 0);
      bgmma(d, da + 2, db + 2, 1);  // the descriptor's address field counts 16 bytes: +2 is the next 32-byte atom
      bgmma(d, da + 4, db + 4, 1);
      bgmma(d, da + 6, db + 6, 1);
      WG_COMMIT;
      WG_WAIT(1);
    }
    WG_WAIT(0);
    acc_fence<128>(d);
    if (tid == 0) *(volatile int*)&stop = 1;
    if (blockIdx.x == 0 && p.acc) store_acc(p.acc, 256, d, tid);
  } else if constexpr (HW > 0) {
    const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + tid;
    Chain ch(t);
    unsigned long long n = 0;
    while (!*(volatile int*)&stop) {
      ch.step();
      n++;
    }
    ch.sink(p.io, t);
    atomicAdd(p.hn, n);
  }
  __syncthreads();
  if (tid == 0 && p.st)
    p.st[blockIdx.x] = Stamp{t0, gtimer(), c0, (unsigned long long)clock64(), smid(), (unsigned long long)p.iters};
}

// 168 registers a thread: b1_res leaves an SM's registers room for two hash_k CTAs, and b1_res_w8's 384 threads fit
constexpr int RES_SMEM = 120 * 1024;
extern "C" __global__ void __maxnreg__(168) b1_res(const __grid_constant__ ResParams p) { res_body<0>(p); }
extern "C" __global__ void __maxnreg__(168) b1_res_w4(const __grid_constant__ ResParams p) { res_body<4>(p); }
extern "C" __global__ void __maxnreg__(168) b1_res_w8(const __grid_constant__ ResParams p) { res_body<8>(p); }

// ---------------------------------------------------------------- beside
extern "C" __global__ void __maxnreg__(80) hash_k(unsigned long long* io, int reps, Stamp* out) {
  unsigned long long t0 = 0, c0 = 0;
  if (threadIdx.x == 0) { t0 = gtimer(); c0 = clock64(); }
  const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  Chain ch(t);
  for (int r = 0; r < reps; ++r) ch.step();
  ch.sink(io, t);
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = Stamp{t0, gtimer(), c0, (unsigned long long)clock64(), smid(), (unsigned long long)reps};
}

extern "C" __global__ void wait_k(volatile unsigned* started, unsigned want) {
  while (*started < want) {
  }
}

// after a millisecond without every CTA it stamps sm = ~0 (the harness refuses the segment)
extern "C" __global__ void stamp_k(Stamp* out, unsigned* arrive) {
  if (threadIdx.x != 0) return;
  atomicAdd(arrive, 1u);
  const unsigned long long t0 = gtimer();
  bool all = true;
  while (*(volatile unsigned*)arrive < gridDim.x)
    if (gtimer() - t0 > 1000000) { all = false; break; }
  out[blockIdx.x] = Stamp{gtimer(), 0, (unsigned long long)clock64(), 0, all ? smid() : ~0ull, 0};
}

extern "C" __global__ void b1_layout(int* out) {
  out[0] = sizeof(ResParams);
  out[1] = offsetof(ResParams, iters);
  out[2] = RES_SMEM;
}
