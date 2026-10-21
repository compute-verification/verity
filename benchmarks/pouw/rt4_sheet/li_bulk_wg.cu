// RT4-SHEET-21: li-bulk beside the IGMMA on sm_90a. One block a SM of 384 threads, wg_until's shape (dc/probes,
// sm90_wgmma.cu): warpgroups 0-1 issue wgmma s8 m64n256k32 flat out (group4, wait_group 1) for `iters` commit groups and
// the first warp to finish raises the stop; warpgroup 2 folds li-bulk units (li_bulk.cuh, BT 128, named barrier 1, its
// thread 256 the bulk issuer) until it sees the stop, the decision made uniform over the warpgroup by bar.red.or so no
// warp leaves the named barrier short. lane_on 0 leaves warpgroup 2 idle: the tensor alone in the same kernel, so the
// tensor fraction is the two wgmma spans' ratio. The lane's rate is its units over its own span (it overruns the stop by
// at most one unit). Not SASS-gated: h200_measure.sh keeps the SASS for the CPU-side count.
#include "../dc/probes/wg_common.cuh"
#include "li_bulk.cuh"

constexpr int WG_BT = 128, WG_LEAD = 256, WG_CPN = 4096;

// rec a block: [0] the wgmma span to the stop, [1] the lane warpgroup's span, [2] its units (each WG_BT thread-units)
__global__ void __launch_bounds__(384, 1) lbw(const uint32_t* st, uint32_t* sink, uint32_t* slots, uint32_t* cpw,
                                              long long* rec, int iters, int lane_on, uint32_t zm) {
  constexpr int BT = WG_BT, SLOT = BT * NCH, LEAD = WG_LEAD;
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  __shared__ __align__(128) uint32_t smem[3 * SLOT];
  __shared__ int stop;
  __shared__ long long tstop;
  if (threadIdx.x == 0) stop = 0, tstop = 0;
  const Tiles t = fill_tiles<0>(sm, 0);
  if (threadIdx.x >= LEAD) fill_kslab<BT, LEAD>(smem);
  asm volatile("fence.proxy.async;" ::: "memory");
  __syncthreads();
  const long long t0 = clock64();
  uint32_t s = threadIdx.x;
  if (threadIdx.x < LEAD) {
    int d[128] = {};
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
      group4<F_s8_n256>(d, t);
      wgmma_wait<1>();
    }
    wgmma_wait<0>();
    if ((threadIdx.x & 31) == 0 && atomicCAS(&stop, 0, 1) == 0) tstop = clock64() - t0;
    wgmma_hold(d);
    s ^= fold_acc(d);
  } else if (lane_on) {
    const int tid = threadIdx.x - LEAD, gtid = blockIdx.x * BT + tid;
    uint32_t* gblk = slots + (size_t)blockIdx.x * SLOT;
    uint32_t* gslot = gblk + NCH * tid;
    const uint32_t spb = (uint32_t)__cvta_generic_to_shared(smem) + 4u * NCH * tid;
    const uint32_t spblk = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t skslab = (uint32_t)__cvta_generic_to_shared(smem + SLOT);
    uint32_t y[8], pe0[8], po0[8], pe1[8], po1[8];
#pragma unroll
    for (int c = 0; c < 8; c++) {
      y[c] = st[(8 * gtid + c) & 255];
      pe0[c] = st[(gtid + 3 * c) & 255];
      po0[c] = st[(gtid + 5 * c) & 255];
      pe1[c] = st[(gtid + 7 * c) & 255];
      po1[c] = st[(gtid + 11 * c) & 255];
    }
    uint32_t* gw = cpw + (size_t)(gtid >> 5) * WG_CPN + (tid & zm);
    int n = 0;
#pragma unroll 1
    for (;; ++n) {
      int any;
      asm volatile(
          "{\n\t.reg .pred p, q;\n\tsetp.ne.s32 q, %1, 0;\n\tbar.red.or.pred p, 1, 128, q;\n\tselp.s32 %0, 1, 0, p;\n\t}"
          : "=r"(any)
          : "r"(*(volatile int*)&stop)
          : "memory");
      if (any) break;
#pragma unroll
      for (int q = 0; q < 4; q++) {
#pragma unroll
        for (int c = 0; c < 8; c++)
          asm volatile("" : "+r"(pe0[c]), "+r"(po0[c]), "+r"(pe1[c]), "+r"(po1[c]));
        li_half<BT, 0, LEAD>(y, pe0, po0, gslot, spb, gblk, spblk, skslab);
        li_half<BT, 1, LEAD>(y, pe1, po1, gslot, spb, gblk, spblk, skslab);
      }
      cp_red8(y, gw + (n & (WG_CPN - 1)));
    }
    const long long t1 = clock64();
    if (tid == 0) rec[3 * blockIdx.x + 1] = t1 - t0, rec[3 * blockIdx.x + 2] = n;
#pragma unroll
    for (int c = 0; c < 8; c++) s ^= y[c];
  }
  __syncthreads();
  if (threadIdx.x == 0) rec[3 * blockIdx.x] = tstop;
  sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int ITERS = 8192, blocks = sms;
  const double L2_FOLD = 8.0 * (16 + 32) * 4, L2_CP = 8.0 * 4;
  uint32_t *dst, *dsink, *dslots, *dcpw;
  long long* drec;
  std::vector<uint32_t> st(256);
  uint64_t seed = 0x5EED2027ull;
  for (auto& x : st) x = (uint32_t)((seed = seed * 6364136223846793005ull + 1442695040888963407ull) >> 32);
  cudaMalloc(&dst, 256 * 4);
  cudaMemcpy(dst, st.data(), 256 * 4, cudaMemcpyHostToDevice);
  cudaMalloc(&dsink, (size_t)blocks * 384 * 4);
  cudaMalloc(&dslots, (size_t)blocks * WG_BT * NCH * 4);
  cudaMalloc(&dcpw, (size_t)blocks * (WG_BT / 32) * WG_CPN * 4);
  cudaMalloc(&drec, (size_t)blocks * 3 * sizeof(long long));
  double span_alone = 0;
  int rc = 0;
  for (int rep = 0; rep < 2; rep++)
    for (int lane_on = 0; lane_on <= 1; lane_on++) {
      cudaMemset(dslots, 0, (size_t)blocks * WG_BT * NCH * 4);
      cudaMemset(drec, 0, (size_t)blocks * 3 * sizeof(long long));
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      cudaEventRecord(e0);
      lbw<<<blocks, 384>>>(dst, dsink, dslots, dcpw, drec, ITERS, lane_on, 0u);
      cudaEventRecord(e1);
      cudaError_t err = cudaGetLastError();
      if (err == cudaSuccess) err = cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      std::vector<long long> rec(3 * blocks);
      cudaMemcpy(rec.data(), drec, rec.size() * sizeof(long long), cudaMemcpyDeviceToHost);
      double tsum = 0, lsum = 0, nsum = 0, tmax = 0;
      for (int b = 0; b < blocks; b++) {
        tsum += rec[3 * b], lsum += rec[3 * b + 1], nsum += rec[3 * b + 2];
        tmax = rec[3 * b] > tmax ? rec[3 * b] : tmax;
      }
      const double tmean = tsum / blocks, lmean = lsum / blocks, nmean = nsum / blocks;
      if (!lane_on) span_alone = tmean;
      // a lane warp a SMSP: SMSP-cycles a thread-unit = span / (units x 32)
      const double smsp_cyc = lane_on && nmean > 0 ? lmean / (nmean * 32.0) : 0;
      const double units = nsum * WG_BT, sec = ms / 1e3;
      printf("lbw rep=%d lane_on=%d blocks=%d iters=%d ms=%.3f wgmma_span_mean=%.0f wgmma_span_max=%.0f "
             "tensor_fraction=%.5f lane_span_mean=%.0f lane_units_mean=%.1f smsp_cyc_per_unit=%.4f "
             "wgmma_cyc_per_group4=%.2f l2_GBps=%.1f err=%s\n",
             rep, lane_on, blocks, ITERS, ms, tmean, tmax, span_alone > 0 ? span_alone / tmean : 0, lmean, nmean,
             smsp_cyc, tmean / ITERS, (L2_FOLD + L2_CP) * units / sec / 1e9, cudaGetErrorString(err));
      rc |= err != cudaSuccess;
      cudaEventDestroy(e0);
      cudaEventDestroy(e1);
    }
  return rc;
}
