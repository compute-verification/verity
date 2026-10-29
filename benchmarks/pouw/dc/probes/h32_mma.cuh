// The H32 probes' legacy mma.sync machinery (sm90_h32_tc.cu, sm90_h32_sp.cu): a form is its D, A and B registers a
// thread and its PTX; ms_run issues R of it a trip on 8 accumulators, mm_run follows each with a round of h32.cuh's mix.
#pragma once
#include <type_traits>
#include "h32.cuh"
#include "wg_common.cuh"

// D's 32-bit registers a thread are ND * sizeof(Acc) / 4.
template <class ACC, class OP, int ND_, int NA_, int NB_, int TY>
struct MsShape {
  using Acc = ACC;
  using Op = OP;
  static constexpr int ND = ND_, NA = NA_, NB = NB_, TYPE = TY;
};
#define SIG_421 " {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
#define OPS_421(C, O) : "+" C(d[0]), "+" C(d[1]), "+" C(d[2]), "+" C(d[3]) : O(a[0]), O(a[1]), O(b[0])
#define SIG_442 " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
#define OPS_442(C, O) \
  : "+" C(d[0]), "+" C(d[1]), "+" C(d[2]), "+" C(d[3]) : O(a[0]), O(a[1]), O(a[2]), O(a[3]), O(b[0]), O(b[1])
#define SIG_221 " {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
#define OPS_221(C, O) : "+" C(d[0]), "+" C(d[1]) : O(a[0]), O(a[1]), O(b[0])
#define SIG_242 " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
#define OPS_242(C, O) : "+" C(d[0]), "+" C(d[1]) : O(a[0]), O(a[1]), O(a[2]), O(a[3]), O(b[0]), O(b[1])
#define SIG_211 " {%0,%1}, {%2}, {%3}, {%0,%1};"
#define OPS_211(C, O) : "+" C(d[0]), "+" C(d[1]) : O(a[0]), O(b[0])
#define SIG_822 " {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
#define OPS_822(C, O)                                                                                            \
  : "+" C(d[0]), "+" C(d[1]), "+" C(d[2]), "+" C(d[3]), "+" C(d[4]), "+" C(d[5]), "+" C(d[6]), "+" C(d[7])     \
  : O(a[0]), O(a[1]), O(b[0]), O(b[1])
#define SIG_422 " {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
#define OPS_422(C, O) : "+" C(d[0]), "+" C(d[1]), "+" C(d[2]), "+" C(d[3]) : O(a[0]), O(a[1]), O(b[0]), O(b[1])
// 2:4 sparse A (mma.sp): the metadata selects elements 0 and 1 of every group of four (0x4, valid for tf32's 1:2 too),
// sparsity selector 0; A holds half the registers of the dense shape's.
#define SIG_S422 " {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
#define OPS_S422(C, O) \
  : "+" C(d[0]), "+" C(d[1]), "+" C(d[2]), "+" C(d[3]) : O(a[0]), O(a[1]), O(b[0]), O(b[1]), "r"(0x44444444u)
#define SIG_S444 " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
#define OPS_S444(C, O)                                                                                           \
  : "+" C(d[0]), "+" C(d[1]), "+" C(d[2]), "+" C(d[3])                                                          \
  : O(a[0]), O(a[1]), O(a[2]), O(a[3]), O(b[0]), O(b[1]), O(b[2]), O(b[3]), "r"(0x44444444u)
#define SIG_S244 " {%0,%1}, {%2,%3,%4,%5}, {%6,%7,%8,%9}, {%0,%1}, %10, 0x0;"
#define OPS_S244(C, O) \
  : "+" C(d[0]), "+" C(d[1]) : O(a[0]), O(a[1]), O(a[2]), O(a[3]), O(b[0]), O(b[1]), O(b[2]), O(b[3]), "r"(0x44444444u)
#define MS_(INSN, NAME, SIG, PTX, ACC, C, OP, O, ND, NA, NB, TY)                                     \
  struct M_##NAME : MsShape<ACC, OP, ND, NA, NB, TY> {                                                \
    DC_DEV static void mma(ACC* d, const OP* a, const OP* b) {                                        \
      asm volatile(INSN PTX SIG_##SIG OPS_##SIG(C, O));                                               \
    }                                                                                                 \
  };
#define MS(...) MS_("mma.sync.aligned.", __VA_ARGS__)
#define MSP(...) MS_("mma.sp::ordered_metadata.sync.aligned.", __VA_ARGS__)

template <class M>
DC_DEV typename M::Op ms_operand(uint32_t i, int mode) {
  if constexpr (std::is_same_v<typename M::Op, double>) return (double)(int8_t)operand_word<0>(i, mode) * 0x1p-7;
  else return operand_word<M::TYPE>(i, mode);
}
template <class M>
struct MsState {
  typename M::Op x[M::NA], y[M::NB];
  typename M::Acc d[8][M::ND];
  DC_DEV MsState(uint32_t lane, int mode) {
#pragma unroll
    for (int i = 0; i < M::NA; i++) x[i] = ms_operand<M>(lane + 32 * i, mode);
#pragma unroll
    for (int i = 0; i < M::NB; i++) y[i] = ms_operand<M>(lane + 128 + 32 * i, mode);
#pragma unroll
    for (int c = 0; c < 8; c++)
#pragma unroll
      for (int e = 0; e < M::ND; e++) d[c][e] = typename M::Acc{};
  }
  DC_DEV void mma(int c) { M::mma(d[c & 7], x, y); }
  DC_DEV uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 8; c++)
#pragma unroll
      for (int e = 0; e < M::ND; e++) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&d[c][e]);
#pragma unroll
        for (int j = 0; j < (int)(sizeof(typename M::Acc) / 4); j++) s ^= w[j];
      }
    return s;
  }
};
template <class M, int R>
DC_DEV void ms_run(const OpArgs& a) {
  MsState<M> m(threadIdx.x & 31, a.mode);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < R; c++) m.mma(c);
  }
  OP_END
  OP_SINK(m.fold() ^ threadIdx.x)
}
template <class M, int R, int... X>
DC_DEV void mm_run(const OpArgs& a) {
  MsState<M> m(threadIdx.x & 31, a.mode);
  HLane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < R; c++) {
      m.mma(c);
      h32_round<X...>(l, c);
    }
  }
  OP_END
  OP_SINK(m.fold() ^ h32_fold<X...>(l) ^ threadIdx.x)
}
