// What a coin bit read by a lookup costs as the index widens (adaptive-dw): each lane indexes its own column of a table
// in shared memory with the next digit of its own coin word, so one LDS reads a whole digit and selects among 2^digit
// values. A trip computes one coin word a lane and reads every whole digit it holds. 32 warps an SM, as branch.cu; the
// price is 4096 x the block's cycles / (iters x 32 warps x lookups a trip) units a warp lookup, and a lane-bit is that
// over 32 x the digit's bits.
//   lut8    8-bit digits, 4 a trip: a 256 x 32-word table (branch.cu's lutl, for comparison in this binary)
//   lut10   10-bit digits, 3 a trip: a 1024 x 32-word table (128 KB)
//   lut12   12-bit digits, 2 a trip: a 4096 x 32-byte table (128 KB); entry i of lane l is byte i % 4 of word
//           (i / 4) x 32 + l, so every lane reads its own bank and no lookup conflicts
#include "../opcode_probe/probe.cuh"

constexpr int TPB = 1024;

__device__ __forceinline__ uint32_t coin(uint32_t x) {
  x *= 0x9E3779B1u;
  return x ^ (x >> 15);
}

template <int BITS>
__device__ __forceinline__ void body(const OpArgs& a) {
  extern __shared__ __align__(16) unsigned char sb[];
  constexpr int N = 32 / BITS;
  constexpr uint32_t MASK = (1u << BITS) - 1u;
  const uint32_t lane = threadIdx.x & 31, salt = blockIdx.x * 0x10001u;
  uint32_t* w = reinterpret_cast<uint32_t*>(sb);
  for (int i = threadIdx.x; i < (BITS == 12 ? 4096 * 8 : (1 << BITS) * 32); i += blockDim.x) w[i] = coin((uint32_t)i);
  __syncthreads();
  uint32_t x = threadIdx.x;
  OP_BEGIN
  OP_LOOP(it) {
    const uint32_t c = coin((uint32_t)it * 32u + lane + salt);
#pragma unroll
    for (int k = 0; k < N; k++) {
      const uint32_t i = (c >> (BITS * k)) & MASK;
      if constexpr (BITS == 12) x ^= sb[((i >> 2) * 32u + lane) * 4u + (i & 3u)];
      else x ^= w[i * 32u + lane];
    }
  }
  OP_END
  OP_SINK(x)
}

OP_KERNEL(lut8, TPB) { body<8>(a); }
OP_KERNEL(lut10, TPB) { body<10>(a); }
OP_KERNEL(lut12, TPB) { body<12>(a); }

#define P(NM, EXP, SMEM, NOTE) \
  {.name = #NM, .kernel = (const void*)NM, .tpb = TPB, .expect = EXP, .dyn_smem = SMEM, .iters = 4096, .note = NOTE}
static const OpProbe PROBES[] = {
    P(lut8, "LDS*=4,BRA=1", 256 * 32 * 4, "4 lane-column lookups, 8 coin bits each"),
    P(lut10, "LDS*=3,BRA=1", 1024 * 32 * 4, "3 lane-column lookups, 10 coin bits each"),
    P(lut12, "LDS*=2,BRA=1", 4096 * 32, "2 lane-column byte lookups, 12 coin bits each"),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
