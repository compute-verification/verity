// Does a second MMA form run beside sm_120's IMMA? (M) and the JI3 rows charge every MMA form in one sum with IMMA's, so
// either is false on a run where both keep their alone rates. One probe times DMMA.8x8x4 (charged 8 SMSP-cycles), the
// other QMMA.16832 E4M3 (charged 16), the form FP8-g1's levels fit in one pass. This is sass_census/unpriced90.cu's
// H200 joint on sm_120a, with mma.sync IMMA m16n8k32 s8 as the tensor form; for each second form X:
//   s_imma        IMMA.16832 on all 32 warps: F1's and the machine model's reference
//   i_w8          IMMA on warps 0-7 of 16, alone in the joint's block: t's unit
//   s_x_w8        X on 8 warps (2 a SMSP, as in the joint): s's unit
//   xU_i          IMMA fixed on warps 0-7, X on warps 8-15 until it ends
//   iU_x          X fixed on warps 8-15, IMMA on warps 0-7 until it ends
// t + s near 1: X holds the tensor pipe; near 2: it runs beside, and mma_joint.py's (M) sum exceeds 1.
#include "../opcode_probe/probe.cuh"

#define RD __device__ __forceinline__

// Eight independent chains a thread for each form; chain s reads chain s+1's previous value, so nothing folds.
struct Imma {
  uint32_t r[8][4];
  uint32_t ia[4], ib[2];
  RD explicit Imma(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
      for (int j = 0; j < 4; j++) r[i][j] = ((tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u) * (j + 1);
#pragma unroll
    for (int j = 0; j < 4; j++) ia[j] = tid * 0x01010101u * (2 * j + 3);
    ib[0] = tid * 0x02040801u;
    ib[1] = tid * 0x80402011u;
  }
  RD void op(int i) {
    const int s = i & 7;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(r[s][0]), "+r"(r[s][1]), "+r"(r[s][2]), "+r"(r[s][3])
                 : "r"(ia[0]), "r"(ia[1]), "r"(ia[2]), "r"(ia[3]), "r"(ib[0]), "r"(ib[1]));
  }
  RD void eight() {
#pragma unroll
    for (int q = 0; q < 8; q++) op(q);
  }
  RD uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x += r[i][0] ^ r[i][1] ^ r[i][2] ^ r[i][3];
    return x;
  }
};

struct Dmma {
  double d[8][2];
  double da, db;
  RD explicit Dmma(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t h = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
      d[i][0] = (double)(h & 0xfffu);
      d[i][1] = (double)(h >> 20);
    }
    da = (double)(int8_t)(tid * 37u);
    db = (double)(int8_t)(tid * 91u);
  }
  RD void op(int i) {
    const int s = i & 7;
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+d"(d[s][0]), "+d"(d[s][1])
                 : "d"(da), "d"(db));
  }
  RD void eight() {
#pragma unroll
    for (int q = 0; q < 8; q++) op(q);
  }
  RD uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x += (uint32_t)__double_as_longlong(d[i][0]) ^ (uint32_t)__double_as_longlong(d[i][1]);
    return x;
  }
};

// Operand bytes are finite E4M3 (exponent fields 7 to 10, any mantissa: 1 to 15), so no NaN reaches the pipe.
struct Qmma {
  float c[8][4];
  uint32_t qa[4], qb[2];
  RD explicit Qmma(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
      for (int j = 0; j < 4; j++) c[i][j] = (float)(((tid * 0x9e3779b9u + i * 0x85ebca6bu) >> 20) * (j + 1));
#pragma unroll
    for (int j = 0; j < 4; j++) qa[j] = 0x38404850u ^ ((tid * (2 * j + 3)) & 0x07070707u);
    qb[0] = 0x50484038u ^ (tid & 0x07070707u);
    qb[1] = 0x40385048u ^ ((tid * 5u) & 0x07070707u);
  }
  RD void op(int i) {
    const int s = i & 7;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[s][0]), "+f"(c[s][1]), "+f"(c[s][2]), "+f"(c[s][3])
                 : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(qb[0]), "r"(qb[1]));
  }
  RD void eight() {
#pragma unroll
    for (int q = 0; q < 8; q++) op(q);
  }
  RD uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++)
      x += __float_as_uint(c[i][0]) ^ __float_as_uint(c[i][1]) ^ __float_as_uint(c[i][2]) ^ __float_as_uint(c[i][3]);
    return x;
  }
};

template <class C>
RD void alone(const OpArgs& a) {
  C c(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) { c.eight(); }
  OP_END
  OP_SINK(c.fold() ^ threadIdx.x)
}

RD uint32_t warp_u() { return __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0); }

// ROLE 0: IMMA alone on warps 0-7; 1: IMMA fixed, X until; 2: X fixed, IMMA until.
template <int ROLE, class X>
RD void joint(const OpArgs& a) {
  const uint32_t w = warp_u();
  uint32_t x = 0;
  OP_BEGIN_STOP
  if (w < 8) {
    Imma c(threadIdx.x ^ (uint32_t)a.mode);
    if constexpr (ROLE == 2) {
      OP_UNTIL(n) { c.eight(); }
      OP_TRIPS(n)
    } else {
      OP_LOOP(it) { c.eight(); }
      if constexpr (ROLE == 1) {
        OP_DONE
      }
    }
    x = c.fold();
  } else if constexpr (ROLE != 0) {
    X c(threadIdx.x ^ (uint32_t)a.mode);
    if constexpr (ROLE == 1) {
      OP_UNTIL(n) { c.eight(); }
      OP_TRIPS(n)
    } else {
      OP_LOOP(it) { c.eight(); }
      OP_DONE
    }
    x = c.fold();
  }
  OP_END_STOP
  OP_SINK(x ^ threadIdx.x)
}

OP_KERNEL(s_imma, 1024) { alone<Imma>(a); }
OP_KERNEL(i_w8, 512) { joint<0, Dmma>(a); }
OP_KERNEL(s_dmma_w8, 256) { alone<Dmma>(a); }
OP_KERNEL(dmmaU_i, 512) { joint<1, Dmma>(a); }
OP_KERNEL(iU_dmma, 512) { joint<2, Dmma>(a); }
OP_KERNEL(s_qmma_w8, 256) { alone<Qmma>(a); }
OP_KERNEL(qmmaU_i, 512) { joint<1, Qmma>(a); }
OP_KERNEL(iU_qmma, 512) { joint<2, Qmma>(a); }

#define I8 "IMMA.16832.S8.S8=8"
#define D4 "DMMA.8x8x4=8"
#define Q8 "QMMA.16832.F32.E4M3.E4M3=8"
static const OpProbe PROBES[] = {
    {.name = "s_imma", .kernel = (const void*)s_imma, .tpb = 1024, .expect = I8, .iters = 1024,
     .note = "IMMA.16832 s8 on 32 warps: 4096 MACs an instruction, the reference"},
    {.name = "i_w8", .kernel = (const void*)i_w8, .tpb = 512, .expect = I8, .warps = "8", .iters = 4096,
     .note = "IMMA.16832 alone on warps 0-7 of the joint's block: t's unit"},
    {.name = "s_dmma_w8", .kernel = (const void*)s_dmma_w8, .tpb = 256, .expect = D4, .iters = 256,
     .note = "DMMA.8x8x4 alone on the joint's 8 warps: s's unit"},
    {.name = "dmmaU_i", .kernel = (const void*)dmmaU_i, .tpb = 512, .expect = I8 ";" D4, .warps = "8;8", .iters = 8192,
     .note = "IMMA fixed, DMMA.8x8x4 until", .until = 2},
    {.name = "iU_dmma", .kernel = (const void*)iU_dmma, .tpb = 512, .expect = I8 ";" D4, .warps = "8;8", .iters = 256,
     .note = "DMMA.8x8x4 fixed, IMMA until", .until = 1},
    {.name = "s_qmma_w8", .kernel = (const void*)s_qmma_w8, .tpb = 256, .expect = Q8, .iters = 4096,
     .note = "QMMA.16832 E4M3 alone on the joint's 8 warps: s's unit"},
    {.name = "qmmaU_i", .kernel = (const void*)qmmaU_i, .tpb = 512, .expect = I8 ";" Q8, .warps = "8;8", .iters = 4096,
     .note = "IMMA fixed, QMMA.16832 E4M3 until", .until = 2},
    {.name = "iU_qmma", .kernel = (const void*)iU_qmma, .tpb = 512, .expect = I8 ";" Q8, .warps = "8;8", .iters = 4096,
     .note = "QMMA.16832 E4M3 fixed, IMMA until", .until = 1},
};
#undef I8
#undef D4
#undef Q8

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
