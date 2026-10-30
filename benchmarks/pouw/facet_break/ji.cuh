// facet_break's joint-issue search (the JI3q rows: each MMA form beside every other vector-pipe instruction): the
// schedule dimensions the mix family held fixed. `generate.py --family ji` instantiates them.
//
//   FB_JMIX(name, Form, NA, U, L1, L2, L3, Q, P, D, QD, tpb)
//       every warp: 8U MMAs an iteration on NA accumulators (MMA c on c mod NA), Q lane ops a group of 8 MMAs (L1, L2,
//       L3 in turn) and QD loads of kind D a group (one line apart, mem.cuh's ring), placed P = 0 after each MMA, 1 after
//       the group, 2 half before and half after each MMA, 3 before each MMA
//   FB_JSPEC(name, Form, NA, U, L1, L2, L3, MW, LQ, tpb)
//       MW warps a SMSP: 8U MMAs a trip (OP_LOOP); the rest: LQ lane ops a trip (OP_UNTIL)
//   FB_JMIX2(name, Form1, Form2, NA, U, L1, L2, L3, Q, tpb)
//       every warp: 8U MMAs an iteration, the two forms in turn, each on its own NA accumulators, and Q lane ops a group
//       of 8 MMAs as jmix's P 0
#pragma once
#include "mem.cuh"

namespace fb {

template <class F, int NA>
struct MmaN {
  uint32_t a[4], b[2];
  typename F::Acc d[NA][4];
  __device__ explicit MmaN(uint32_t lane) {
    a[0] = F::W ^ (lane << 8); a[1] = F::W; a[2] = F::W ^ 0x00010000u; a[3] = F::W;
    b[0] = F::W; b[1] = F::W ^ (lane << 16);
#pragma unroll
    for (int c = 0; c < NA; c++)
#pragma unroll
      for (int j = 0; j < 4; j++) d[c][j] = 0;
  }
  __device__ __forceinline__ void go(int c) { F::mma(d[c % NA], a, b); }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < NA; c++) {
      typename F::Acc v = d[c][0];
      s ^= *reinterpret_cast<const uint32_t*>(&v);
    }
    return s;
  }
};

template <class F, int NA, int U, int L1, int L2, int L3, int Q, int P, int D, int QD>
__device__ __forceinline__ void jmix(const OpArgs& a) {
  FB_TABLE(D, D_NONE)
  MmaN<F, NA> m(lane);
  Lanes o(a);
  uint32_t acc = threadIdx.x + 1;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int g = 0; g < U; g++) {
#pragma unroll
      for (int c = 0; c < 8; c++) {
        const int q0 = g * Q + c * Q / 8, q1 = g * Q + (c + 1) * Q / 8, qm = (q0 + q1) / 2;
        if constexpr (P == 2) {
#pragma unroll
          for (int q = q0; q < qm; q++) o.template op3<L1, L2, L3>(q);
        }
        if constexpr (P == 3) {
#pragma unroll
          for (int q = q0; q < q1; q++) o.template op3<L1, L2, L3>(q);
        }
        m.go(g * 8 + c);
        if constexpr (P == 0) {
#pragma unroll
          for (int q = q0; q < q1; q++) o.template op3<L1, L2, L3>(q);
        }
        if constexpr (P == 2) {
#pragma unroll
          for (int q = qm; q < q1; q++) o.template op3<L1, L2, L3>(q);
        }
        if constexpr (D != D_NONE)
          acc ^= loads<D, D_NONE, 1, P_LINE, U * QD>(a, sbase, gb, gel, lane, w, it, g * QD + c * QD / 8,
                                                     g * QD + (c + 1) * QD / 8);
      }
      if constexpr (P == 1) {
#pragma unroll
        for (int q = g * Q; q < (g + 1) * Q; q++) o.template op3<L1, L2, L3>(q);
      }
    }
  }
  OP_END
  OP_SINK((lane ^ m.fold() ^ o.template fold3<L1, L2, L3>() ^ acc))
}

template <class F, int NA, int U, int L1, int L2, int L3, int MW, int LQ>
__device__ __forceinline__ void jspec(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  MmaN<F, NA> m(lane);
  Lanes o(a);
  OP_BEGIN_STOP
  if ((int)(threadIdx.x >> 5) < 4 * MW) {
    OP_LOOP(it) {
#pragma unroll
      for (int c = 0; c < 8 * U; c++) m.go(c);
    }
    OP_DONE
  } else {
    OP_UNTIL(n) {
#pragma unroll
      for (int q = 0; q < LQ; q++) o.template op3<L1, L2, L3>(q);
    }
    OP_TRIPS(n)
  }
  OP_END_STOP
  OP_SINK((lane ^ m.fold() ^ o.template fold3<L1, L2, L3>()))
}

// Two forms in turn: MMA c of group g is F1's (c even) or F2's (c odd), the form's (4g + c/2)-th on its own NA
// accumulators; lanes as jmix's P 0
template <class F1, class F2, int NA, int U, int L1, int L2, int L3, int Q>
__device__ __forceinline__ void jmix2(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  MmaN<F1, NA> m1(lane);
  MmaN<F2, NA> m2(lane);
  Lanes o(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int g = 0; g < U; g++) {
#pragma unroll
      for (int c = 0; c < 8; c++) {
        if (c & 1) m2.go(4 * g + (c >> 1));
        else m1.go(4 * g + (c >> 1));
#pragma unroll
        for (int q = g * Q + c * Q / 8; q < g * Q + (c + 1) * Q / 8; q++) o.template op3<L1, L2, L3>(q);
      }
    }
  }
  OP_END
  OP_SINK((lane ^ m1.fold() ^ m2.fold() ^ o.template fold3<L1, L2, L3>()))
}

}  // namespace fb

#define FB_JMIX2(name, F1, F2, NA, U, L1, L2, L3, Q, TPB) \
  OP_KERNEL(name, TPB) { fb::jmix2<fb::F1, fb::F2, NA, U, fb::L1, fb::L2, fb::L3, Q>(a); }
#define FB_JMIX(name, F, NA, U, L1, L2, L3, Q, P, D, QD, TPB) \
  OP_KERNEL(name, TPB) { fb::jmix<fb::F, NA, U, fb::L1, fb::L2, fb::L3, Q, P, fb::D, QD>(a); }
#define FB_JSPEC(name, F, NA, U, L1, L2, L3, MW, LQ, TPB) \
  OP_KERNEL(name, TPB) { fb::jspec<fb::F, NA, U, fb::L1, fb::L2, fb::L3, MW, LQ>(a); }
