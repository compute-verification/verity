// RT3-DC on the H200 (sm_90a): li's checkpoint by atomics beside the saturated wgmma (cpatom.cuh has the forms).
//
//   exact [U]       each form's checkpoints and final states against the host's reference, word for word (exact.jsonl)
//   probe ARGS      the opcode probes below (op_main: --list, --time NAME... --reps R): gated by probe.py
//
// wgcp_<form>: sm90_wgmma.cu's wgli loop (2 warpgroups, each warp its group of 4 wgmma s8 m64n256k32, then its fold, then
// wait_group 0), two trips an iteration so the checkpoint's buffers and words are static (cpatom.cuh's cp_trip): the
// block's cycles an iteration are 8 wgmma a warpgroup and 8 units a warp. wgli_redux is main's, for the same loop with
// LI-CP-W's 4 LOP3 + REDUX on the chains and no store. wgw_<form>: the same loop with 32 units an iteration (8 groups
// of 4 wgmma, 4 units a warp beside each and its wait), so CP_XT's batch is whole: wgw_xt against wgw_seg4 is the
// transpose's cost against ♭'s checkpoint beside the tensor. lr_*, lr8w_* and at_* are the lane-alone loops
// (cpatom.cuh).
#define main sm90_wgmma_main
#include "../dc/probes/sm90_wgmma.cu"
#undef main

#include "cpatom.cuh"

// group4 with every wgmma accumulating: the iteration's second group, so the first group's D is read (with group4's
// overwrite there, ptxas drops the first group as dead).
template <class F>
DC_DEV void group4_acc(typename F::Acc (&d)[F::NA], const Tiles& t) {
  wgmma_fence();
  F::mma(d, t, 1);
  F::mma(d, t, 1);
  F::mma(d, t, 1);
  F::mma(d, t, 1);
  wgmma_commit();
}

template <int CP>
DC_DEV void wg_cp(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  __shared__ __align__(16) uint32_t sh[8][8];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  int d[128] = {};
  Lane l(threadIdx.x);
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const uint32_t zm = (uint32_t)a.mode;
  if (lane < 8) sh[wp][lane] = 0;
  __syncwarp();
  const uint32_t a0 = smem_u32(&sh[wp][0]) + 4u * (lane & zm);
  uint32_t* wi = cp_words(a, 8, wp, lane);
  auto fold = [&] {
#pragma unroll
    for (int c = 0; c < 64; c++) {
      l.op<L_LOP3>(c);
      l.op<L_IMAD>(c);
    }
  };
  OP_BEGIN
  OP_LOOP(it) {
    group4<F_s8_n256>(d, t);
    cp_trip<CP>(l, fold, a0, wi);
    wgmma_wait<0>();
    group4_acc<F_s8_n256>(d, t);
    cp_trip<CP>(l, fold, a0 + 16u, wi + 4);
    wgmma_wait<0>();
    wi += 8;
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}
// wg_cp at 32 units an iteration (cpatom.cuh's cp_wide): four of its iterations' groups and units, then CP_XT's tail; its
// CP_SEG4 baseline in the same loop. CP_XT's buffers are dynamic (8 x XT_WORDS words).
template <int CP>
DC_DEV void wg_cpw(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  extern __shared__ __align__(16) uint32_t xdyn[];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  int d[128] = {};
  Lane l(threadIdx.x);
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const uint32_t zm = (uint32_t)a.mode;
  const uint32_t xb = smem_u32(xdyn) + 4u * XT_WORDS * wp;
  const uint32_t xs = xb + 4u * lane, xr = xb + 4u * XT_ROW * lane;
  uint32_t* wo = cp_wide(a, 8, wp) + (CP == CP_XT ? (uint32_t)lane : (lane & zm));
  auto fold = [&] {
#pragma unroll
    for (int c = 0; c < 64; c++) {
      l.op<L_LOP3>(c);
      l.op<L_IMAD>(c);
    }
  };
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      if (q == 0) group4<F_s8_n256>(d, t);
      else group4_acc<F_s8_n256>(d, t);
      cp_quad<CP>(l, fold, xs, 0u, wo, 8 * q);
      wgmma_wait<0>();
      group4_acc<F_s8_n256>(d, t);
      cp_quad<CP>(l, fold, xs, 0u, wo, 8 * q + 4);
      wgmma_wait<0>();
    }
    if constexpr (CP == CP_XT) xt_tail(xr, wo);
    wo += 32;
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}
OP_KERNEL(wgw_seg4, 256) { wg_cpw<CP_SEG4>(a); }
OP_KERNEL(wgw_xt, 256) { wg_cpw<CP_XT>(a); }

OP_KERNEL(wgcp_none, 256) { wg_cp<CK_NONE>(a); }
OP_KERNEL(wgcp_seg4, 256) { wg_cp<CP_SEG4>(a); }
OP_KERNEL(wgcp_ats, 256) { wg_cp<CP_ATS>(a); }
OP_KERNEL(wgcp_ats2, 256) { wg_cp<CP_ATS2>(a); }
OP_KERNEL(wgcp_ats8, 256) { wg_cp<CP_ATS8>(a); }
OP_KERNEL(wgcp_red, 256) { wg_cp<CP_RED>(a); }
OP_KERNEL(wgcp_red2, 256) { wg_cp<CP_RED2>(a); }

#define WG8 "IGMMA.64x256x32.S8.S8*=8,"
#define WG32 "IGMMA.64x256x32.S8.S8*=32,"
#define P_WGCP(F) {.name = "wgcp_" #F, .kernel = (const void*)wgcp_##F, .tpb = 256, .expect = WG8 CPA_CK_##F, .iters = 2048}
static const OpProbe RT3[] = {
    {.name = "wg_s8_n256_a1", .kernel = (const void*)wg_s8_n256_a1, .tpb = 256, .expect = "IGMMA.64x256x32.S8.S8*=4",
     .iters = 4096},
    {.name = "wgli_redux", .kernel = (const void*)wgli_redux, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4,LOP3.LUT=272,IMAD=256,REDUX*=4", .iters = 4096},
    P_WGCP(none), P_WGCP(seg4), P_WGCP(ats), P_WGCP(ats2), P_WGCP(ats8), P_WGCP(red), P_WGCP(red2),
    {.name = "wgw_seg4", .kernel = (const void*)wgw_seg4, .tpb = 256, .expect = WG32 CPA_CKW_seg4, .iters = 512},
    {.name = "wgw_xt", .kernel = (const void*)wgw_xt, .tpb = 256, .expect = WG32 CPA_CKW_xt,
     .dyn_smem = 8 * XT_WORDS * 4, .iters = 512},
    CPA_LR_PROBES, CPA_LRW_PROBES,
};

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "exact")) return run_exact(argc >= 3 ? atoi(argv[2]) : 64);
  if (argc >= 2 && !strcmp(argv[1], "probe")) return op_main(argc - 1, argv + 1, RT3);
  return op_main(argc, argv, RT3);
}
