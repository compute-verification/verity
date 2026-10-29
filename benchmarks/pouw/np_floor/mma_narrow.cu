// sm_90's narrow mma.sync forms as word writers, on probe.cuh: s4 (m8n8k32, m16n8k32, m16n8k64) and b1 with AND and XOR
// popc (m8n8k128, m16n8k128, m16n8k256), beside int8 m16n8k32. As ../opcode_probe/probes/mma.cu: 16 warps, 8
// independent accumulators a warp and one MMA each an iteration, so the rate is the tensor pipe's and not one chain's
// latency. A warp instruction writes 64 s32 registers (m8n8) or 128 (m16n8): `words`.
#include "../opcode_probe/probe.cuh"

constexpr int TPB = 512;

#define D2 "+r"(d[0]), "+r"(d[1])
#define D4 "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
#define A1 "r"(a[0])
#define A2 "r"(a[0]), "r"(a[1])
#define A4 "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
#define B1 "r"(b[0])
#define B2 "r"(b[0]), "r"(b[1])
#define FORM(NAME, ND, PTX, OUTS, ...)                                             \
  struct NAME {                                                                    \
    static constexpr int N = ND;                                                   \
    __device__ static void mma(int32_t* d, const uint32_t* a, const uint32_t* b) { \
      asm volatile(PTX : OUTS : __VA_ARGS__);                                      \
    }                                                                              \
  };
#define M16(SHAPE, TYPES, NA) \
  "mma.sync.aligned." SHAPE ".row.col.s32." TYPES " {%0,%1,%2,%3}, " NA ", {%0,%1,%2,%3};"
#define M8(SHAPE, TYPES) "mma.sync.aligned." SHAPE ".row.col.s32." TYPES " {%0,%1}, {%2}, {%3}, {%0,%1};"

FORM(S8k32, 4, M16("m16n8k32", "s8.s8.s32", "{%4,%5,%6,%7}, {%8,%9}"), D4, A4, B2)
FORM(S4k64, 4, M16("m16n8k64", "s4.s4.s32", "{%4,%5,%6,%7}, {%8,%9}"), D4, A4, B2)
FORM(S4k32, 4, M16("m16n8k32", "s4.s4.s32", "{%4,%5}, {%6}"), D4, A2, B1)
FORM(S4m8k32, 2, M8("m8n8k32", "s4.s4.s32"), D2, A1, B1)
FORM(AndK256, 4, M16("m16n8k256", "b1.b1.s32.and.popc", "{%4,%5,%6,%7}, {%8,%9}"), D4, A4, B2)
FORM(XorK256, 4, M16("m16n8k256", "b1.b1.s32.xor.popc", "{%4,%5,%6,%7}, {%8,%9}"), D4, A4, B2)
FORM(AndK128, 4, M16("m16n8k128", "b1.b1.s32.and.popc", "{%4,%5}, {%6}"), D4, A2, B1)
FORM(XorK128, 4, M16("m16n8k128", "b1.b1.s32.xor.popc", "{%4,%5}, {%6}"), D4, A2, B1)
FORM(AndM8k128, 2, M8("m8n8k128", "b1.b1.s32.and.popc"), D2, A1, B1)
FORM(XorM8k128, 2, M8("m8n8k128", "b1.b1.s32.xor.popc"), D2, A1, B1)

// Operands vary by lane, so no two lanes' products coincide; s4 nibbles and b1 bits are every pattern's valid values.
template <class F>
__device__ __forceinline__ void body(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t x[4] = {0x23572357u ^ lane, 0x75327532u, 0x35723572u ^ (lane << 8), 0x57235723u};
  const uint32_t y[2] = {0x22335577u ^ (lane >> 3), 0x77553322u ^ lane};
  int32_t d[8][F::N] = {};
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) F::mma(d[c], x, y);
  }
  OP_END
  uint32_t s = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)d[c][0] ^ (uint32_t)d[c][F::N - 1];
  OP_SINK(s)
}

OP_KERNEL(imma_s8_k32, TPB) { body<S8k32>(a); }
OP_KERNEL(imma_s4_k64, TPB) { body<S4k64>(a); }
OP_KERNEL(imma_s4_k32, TPB) { body<S4k32>(a); }
OP_KERNEL(imma_s4_m8k32, TPB) { body<S4m8k32>(a); }
OP_KERNEL(bmma_and_k256, TPB) { body<AndK256>(a); }
OP_KERNEL(bmma_xor_k256, TPB) { body<XorK256>(a); }
OP_KERNEL(bmma_and_k128, TPB) { body<AndK128>(a); }
OP_KERNEL(bmma_xor_k128, TPB) { body<XorK128>(a); }
OP_KERNEL(bmma_and_m8k128, TPB) { body<AndM8k128>(a); }
OP_KERNEL(bmma_xor_m8k128, TPB) { body<XorM8k128>(a); }

#ifndef EXPECT_S4K64
#define EXPECT_S4K64 "IMMA.16864.S4.S4=8"
#endif
#ifndef EXPECT_S4K32
#define EXPECT_S4K32 "IMMA.16832.S4.S4=8"
#endif
#ifndef EXPECT_S4M8
#define EXPECT_S4M8 "IMMA.8832.S4.S4=8"
#endif
#ifndef EXPECT_B256
#define EXPECT_B256(op) "BMMA.168256." op ".POPC=8"
#endif
#ifndef EXPECT_B128
#define EXPECT_B128(op) "BMMA.168128." op ".POPC=8"
#endif
#ifndef EXPECT_BM8
#define EXPECT_BM8(op) "BMMA.88128." op ".POPC=8"
#endif
static const OpProbe PROBES[] = {
    {.name = "imma_s8_k32", .kernel = (const void*)imma_s8_k32, .tpb = TPB, .expect = "IMMA.16832.S8.S8=8",
     .words = 128, .iters = 4096, .note = "int8 m16n8k32: 128 s32 registers a warp instruction"},
    {.name = "imma_s4_k64", .kernel = (const void*)imma_s4_k64, .tpb = TPB, .expect = EXPECT_S4K64, .words = 128,
     .iters = 4096, .note = "s4 m16n8k64: 128 s32 registers a warp instruction"},
    {.name = "imma_s4_k32", .kernel = (const void*)imma_s4_k32, .tpb = TPB, .expect = EXPECT_S4K32, .words = 128,
     .iters = 4096, .note = "s4 m16n8k32: 128 s32 registers a warp instruction"},
    {.name = "imma_s4_m8k32", .kernel = (const void*)imma_s4_m8k32, .tpb = TPB, .expect = EXPECT_S4M8, .words = 64,
     .iters = 4096, .note = "s4 m8n8k32: 64 s32 registers a warp instruction"},
    {.name = "bmma_and_k256", .kernel = (const void*)bmma_and_k256, .tpb = TPB, .expect = EXPECT_B256("AND"),
     .words = 128, .iters = 4096, .note = "b1 AND.POPC m16n8k256: 128 s32 registers a warp instruction"},
    {.name = "bmma_xor_k256", .kernel = (const void*)bmma_xor_k256, .tpb = TPB, .expect = EXPECT_B256("XOR"),
     .words = 128, .iters = 4096, .note = "b1 XOR.POPC m16n8k256: 128 s32 registers a warp instruction"},
    {.name = "bmma_and_k128", .kernel = (const void*)bmma_and_k128, .tpb = TPB, .expect = EXPECT_B128("AND"),
     .words = 128, .iters = 4096, .note = "b1 AND.POPC m16n8k128: 128 s32 registers a warp instruction"},
    {.name = "bmma_xor_k128", .kernel = (const void*)bmma_xor_k128, .tpb = TPB, .expect = EXPECT_B128("XOR"),
     .words = 128, .iters = 4096, .note = "b1 XOR.POPC m16n8k128: 128 s32 registers a warp instruction"},
    {.name = "bmma_and_m8k128", .kernel = (const void*)bmma_and_m8k128, .tpb = TPB, .expect = EXPECT_BM8("AND"),
     .words = 64, .iters = 4096, .note = "b1 AND.POPC m8n8k128: 64 s32 registers a warp instruction"},
    {.name = "bmma_xor_m8k128", .kernel = (const void*)bmma_xor_m8k128, .tpb = TPB, .expect = EXPECT_BM8("XOR"),
     .words = 64, .iters = 4096, .note = "b1 XOR.POPC m8n8k128: 64 s32 registers a warp instruction"},
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
