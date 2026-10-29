// Shared-memory loads as word writers on sm_90a, by width: LDS (1 register a lane), LDS.64 (2) and LDS.128 (4), on
// probe.cuh. 32 warps an SM, 8 loads an iteration. Each load of each iteration has its own run of consecutive lane
// entries round a 32 KB ring (the base moving U runs a warp and an iteration), so ptxas can't merge two loads and every
// phase reads 128 consecutive bytes: no bank conflict.
#include "../opcode_probe/probe.cuh"

constexpr int TPB = 1024, U = 8, RING_BYTES = 32768;

template <int B>
__host__ __device__ inline unsigned lds_off(int slot, int lane, int warp, int it) {
  return (((unsigned)lane + 32u * ((unsigned)slot + U * (unsigned)(warp + it))) * B) & (RING_BYTES - 1);
}

template <int B>
__device__ __forceinline__ void lds_probe(const OpArgs& a) {
  __shared__ __align__(128) unsigned char smem[RING_BYTES];
  for (int i = threadIdx.x; i < RING_BYTES / 4; i += blockDim.x)
    reinterpret_cast<unsigned*>(smem)[i] = i * 2654435761u + blockIdx.x + 1;
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  unsigned acc = threadIdx.x + 1;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const unsigned addr = sbase + lds_off<B>(u, lane, w, it);
      unsigned r0, r1 = 0, r2 = 0, r3 = 0;
      if constexpr (B == 4) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r0) : "r"(addr));
      if constexpr (B == 8) asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
      if constexpr (B == 16)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
      acc ^= r0 ^ r1 ^ r2 ^ r3;
    }
  }
  OP_END
  OP_SINK(acc ^ reinterpret_cast<unsigned*>(smem)[threadIdx.x])
}

OP_KERNEL(lds32, TPB) { lds_probe<4>(a); }
OP_KERNEL(lds64, TPB) { lds_probe<8>(a); }
OP_KERNEL(lds128w, TPB) { lds_probe<16>(a); }

static const OpProbe PROBES[] = {
    {.name = "lds32", .kernel = (const void*)lds32, .tpb = TPB, .expect = "LDS=8", .words = 32,
     .smem_ops = "LDS*8", .smem = lds_off<4>},
    {.name = "lds64", .kernel = (const void*)lds64, .tpb = TPB, .expect = "LDS.64=8", .words = 64,
     .smem_ops = "LDS.64*8", .smem = lds_off<8>},
    {.name = "lds128w", .kernel = (const void*)lds128w, .tpb = TPB, .expect = "LDS.128=8", .words = 128,
     .smem_ops = "LDS.128*8", .smem = lds_off<16>},
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
