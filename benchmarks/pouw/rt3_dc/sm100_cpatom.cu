// RT3-DC on the B200 (sm_100a): li's checkpoint by atomics beside the saturated tcgen05.mma (cpatom.cuh has the forms),
// and the SM clock under that load.
//
//   exact [U]               each form's checkpoints and final states against the host's reference (exact.jsonl)
//   clock LOAD ITERS N      N launches of one load, each block's clock64 over its own %globaltimer (clock.jsonl)
//   probe ARGS              the opcode probes below (op_main: --list, --time NAME... --reps R): gated by probe.py
//
// tcU_<form>: RT2-BLACKWELL's tcliU_floor shape (warp 0 fixed on 4 tcgen05.mma i8 128x256x32 an iteration, warps 4-15, 3
// a SMSP, folding only while it runs: OP_UNTIL), the words in registers (♭ credits no tcgen05.ld), two trips an
// iteration with the checkpoint in each form (cpatom.cuh's cp_trip). tcU_floor is tcliU_floor itself (♭'s 4 LOP3 +
// REDUX + 2 SHFL a segment, no store), 4 segments a trip. A loop's issue a SMSP-cycle beside the tensor is its
// instructions times trips over the stop's cycles, over 4.
#define main sm100_tcgen05_main
#include "../dc/probes/sm100_tcgen05.cu"
#undef main

#include "cpatom.cuh"

DC_DEV void tc_words(uint32_t (&r0)[16], uint32_t (&r1)[16]) {
#pragma unroll
  for (int c = 0; c < 16; c++) {
    r0[c] = dc_mix32(threadIdx.x * 32u + c);
    r1[c] = dc_mix32(threadIdx.x * 32u + 16u + c);
  }
}
DC_DEV void tc_mma4(Smem& s, uint32_t tmem, uint64_t da, uint64_t db) {
  constexpr uint32_t idesc = idesc_of<0>(128, 256);
  if (elect_one()) {
    umma_i8(tmem, da, db, idesc, 0);
    umma_i8(tmem, da, db, idesc, 1);
    umma_i8(tmem, da, db, idesc, 1);
    umma_i8(tmem, da, db, idesc, 1);
  }
  __syncwarp();
}

template <int CP>
DC_DEV void tc_cp(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  __shared__ __align__(16) uint32_t sh[16][8];
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  const uint32_t w = warp_u();
  const int lane = threadIdx.x & 31;
  const uint32_t zm = (uint32_t)a.mode;
  Lane l(threadIdx.x);
  uint32_t r0[16], r1[16];
  tc_words(r0, r1);
  if (lane < 8) sh[w][lane] = 0;
  __syncwarp();
  const uint32_t a0 = smem_u32(&sh[w][0]) + 4u * (lane & zm);
  uint32_t* wi = cp_words(a, 16, (int)w, lane);
  auto fold = [&] {
#pragma unroll
    for (int c = 0; c < 64; c++) {
      asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(l.r[c & 7]) : "r"(r0[c & 15]), "r"(0x85EBCA6Bu));
      asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(l.r[c & 7]) : "r"(0x9E3779B9u), "r"(r1[c & 15]));
    }
  };
  OP_BEGIN_STOP
  if (w == 0) {
    OP_LOOP(it) { tc_mma4(s, tmem, da, db); }
    OP_DONE
  } else if (w >= 4) {
    OP_UNTIL(n) {
      cp_trip<CP>(l, fold, a0, wi);
      cp_trip<CP>(l, fold, a0 + 16u, wi + 4);
      wi += 8;
    }
    OP_TRIPS(n)
  }
  OP_END_STOP
  if (w == 0) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t x = sink_ld(tmem, 0) ^ l.fold();
  teardown(tmem);
  OP_SINK(x ^ threadIdx.x)
}

// RT2-BLACKWELL's tcfloor_until_mma<4> (#2771), as it is there: ♭'s segment, 68 LOP3, 64 IMAD and REDUX + 2 SHFL.
DC_DEV void tc_floor(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  const uint32_t w = warp_u();
  Lane l(threadIdx.x);
  uint32_t r0[16], r1[16];
  tc_words(r0, r1);
  auto seg = [&] {
#pragma unroll
    for (int c = 0; c < 64; c++) {
      asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(l.r[c & 7]) : "r"(r0[c & 15]), "r"(0x85EBCA6Bu));
      asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(l.r[c & 7]) : "r"(0x9E3779B9u), "r"(r1[c & 15]));
    }
    l.op<L_LOP3>(0); l.op<L_LOP3>(2); l.op<L_LOP3>(4); l.op<L_LOP3>(6);
    l.op<L_REDUX>(0); l.op<L_SHFL>(1); l.op<L_SHFL>(3);
  };
  OP_BEGIN_STOP
  if (w == 0) {
    OP_LOOP(it) { tc_mma4(s, tmem, da, db); }
    OP_DONE
  } else if (w >= 4) {
    OP_UNTIL(n) {
#pragma unroll
      for (int k = 0; k < 4; k++) seg();
    }
    OP_TRIPS(n)
  }
  OP_END_STOP
  if (w == 0) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t x = sink_ld(tmem, 0) ^ l.fold();
  teardown(tmem);
  OP_SINK(x ^ threadIdx.x)
}

OP_KERNEL(tcU_floor, 512) { tc_floor(a); }
OP_KERNEL(tcU_none, 512) { tc_cp<CK_NONE>(a); }
OP_KERNEL(tcU_seg4, 512) { tc_cp<CP_SEG4>(a); }
OP_KERNEL(tcU_ats, 512) { tc_cp<CP_ATS>(a); }
OP_KERNEL(tcU_ats2, 512) { tc_cp<CP_ATS2>(a); }
OP_KERNEL(tcU_ats8, 512) { tc_cp<CP_ATS8>(a); }
OP_KERNEL(tcU_red, 512) { tc_cp<CP_RED>(a); }
OP_KERNEL(tcU_red2, 512) { tc_cp<CP_RED2>(a); }

#define P_TCU(F) {.name = "tcU_" #F, .kernel = (const void*)tcU_##F, .tpb = 512, .expect = UI ";" CPA_CK_##F, \
                  .warps = "1;12", .iters = 2048, .until = 2}
static const OpProbe RT3[] = {
    {.name = "tc_i8_m128_n256", .kernel = (const void*)tc_i8_m128_n256, .tpb = 128, .expect = UI, .warps = "1",
     .iters = 4096},
    {.name = "tcU_floor", .kernel = (const void*)tcU_floor, .tpb = 512,
     .expect = UI ";LOP3.LUT=272,IMAD=256,REDUX*=4,SHFL.BFLY=8", .warps = "1;12", .iters = 2048, .until = 2},
    P_TCU(none), P_TCU(seg4), P_TCU(ats), P_TCU(ats2), P_TCU(ats8), P_TCU(red), P_TCU(red2),
    CPA_LR_PROBES,
};

// ---- clock ---------------------------------------------------------------------------------------------------------
// RT2-HOPPER's clock (sm90_clock.cu) on the B200: each block's clock64 and %globaltimer at its start and end, so the
// clock is the block's cycles over its own nanoseconds (no launch or tail in it). Loads: the MMA alone (tc_alone, warp 0
// of 16), ♭'s floor beside it (tcU_floor: the honest mixed load), and the floor's fold alone (lr_seg4's loop, no MMA).
struct Stamp {
  unsigned long long g0, g1;
  long long c0, c1;
};
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
enum { CL_TC, CL_FLOOR, CL_LANE, CL_N };
static const char* const CL_NAMES[CL_N] = {"tc_i8_m128_n256", "tcU_floor", "lr_none"};
template <int LD>
__global__ void __launch_bounds__(512, 1) clk(OpArgs a, Stamp* st) {
  __syncthreads();
  const unsigned long long g0 = gtimer();
  const long long c0 = clock64();
  if constexpr (LD == CL_TC) tc_alone<0, 128, 256>(a);
  if constexpr (LD == CL_FLOOR) tc_floor(a);
  if constexpr (LD == CL_LANE) lr_body<CK_NONE, 16>(a);
  __syncthreads();
  if (threadIdx.x == 0) st[blockIdx.x] = {g0, gtimer(), c0, clock64()};
}
static int run_clock(int load, int iters, int launches) {
  if (load < 0 || load >= CL_N) return 2;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  OpArgs a{};
  cudaMalloc(&a.rec, sms * sizeof(OpClk));
  cudaMalloc(&a.sink, (size_t)sms * 512 * sizeof(unsigned));
  a.g = nullptr;  // no load here stores a checkpoint (lr_body<CK_NONE> writes no word)
  a.gbytes = 0;
  a.iters = iters;
  a.mode = 0;
  Stamp* dst;
  cudaMalloc(&dst, sms * sizeof(Stamp));
  std::vector<Stamp> st(sms);
  std::vector<OpClk> rec(sms);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  int bad = 0;
  for (int L = 0; L < launches; L++) {
    cudaEventRecord(e0);
    if (load == CL_TC) clk<CL_TC><<<sms, 512>>>(a, dst);
    if (load == CL_FLOOR) clk<CL_FLOOR><<<sms, 512>>>(a, dst);
    if (load == CL_LANE) clk<CL_LANE><<<sms, 512>>>(a, dst);
    cudaEventRecord(e1);
    const cudaError_t e = cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaMemcpy(st.data(), dst, sms * sizeof(Stamp), cudaMemcpyDeviceToHost);
    cudaMemcpy(rec.data(), a.rec, sms * sizeof(OpClk), cudaMemcpyDeviceToHost);
    std::vector<double> mhz(sms);
    long long cmax = 0;
    unsigned long long gmin = ~0ull, gmax = 0;
    for (int b = 0; b < sms; b++) {
      mhz[b] = 1e3 * (double)(st[b].c1 - st[b].c0) / (double)(st[b].g1 - st[b].g0);
      cmax = std::max(cmax, st[b].c1 - st[b].c0);
      gmin = std::min(gmin, st[b].g0);
      gmax = std::max(gmax, st[b].g1);
    }
    std::sort(mhz.begin(), mhz.end());
    std::vector<long long> trips(sms), stop(sms);
    for (int b = 0; b < sms; b++) trips[b] = rec[b].trips, stop[b] = rec[b].stop;
    std::sort(trips.begin(), trips.end());
    std::sort(stop.begin(), stop.end());
    printf("{\"probe\": \"clock\", \"load\": \"%s\", \"launch\": %d, \"iters\": %d, \"blocks\": %d, \"event_ms\": %.4f, "
           "\"span_ms\": %.4f, \"mhz_min\": %.2f, \"mhz_med\": %.2f, \"mhz_max\": %.2f, \"event_mhz\": %.2f, "
           "\"trips_med\": %lld, \"stop_med\": %lld, \"cuda\": \"%s\"}\n",
           CL_NAMES[load], L, iters, sms, ms, (gmax - gmin) / 1e6, mhz[0], mhz[sms / 2], mhz[sms - 1], 1e-3 * cmax / ms,
           trips[sms / 2], stop[sms / 2], cudaGetErrorString(e));
    fflush(stdout);
    bad += e != cudaSuccess;
  }
  return bad;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "exact")) return run_exact(argc >= 3 ? atoi(argv[2]) : 64);
  if (argc >= 5 && !strcmp(argv[1], "clock")) return run_clock(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  if (argc >= 2 && !strcmp(argv[1], "probe")) return op_main(argc - 1, argv + 1, RT3);
  return op_main(argc, argv, RT3);
}
