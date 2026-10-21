// RT4-SHEET-21: li's fold with its XOR-absorb moved onto the TMA bulk engine, built for the device and checked bit-exact
// against li's CPU reference (red_generic.cu's li_set / cp_seg<CP_RED8>). This is the "li-bulk" route RT4-SHEET-17
// counted from fbu_K4_BT128's sm_90a SASS: each lane writes its 8 chain states to its global slot (two STG.128) and the
// half-step's 8 even entries to a shared slot (two STS.128); one thread bulk-XORs the whole block's shared slot and a
// static li_k slab into the global slot (two cp.reduce.async.bulk.xor, UBLKRED); each lane reads its 8 states back (two
// LDG.128, ld.global.cg, the read-back fbw proved bit-exact on the device, r20261019-234605-a812) and finishes li's
// half-step with one IMAD a chain (u * li_a + odd entry). A unit is 128 words a lane (4 sets of 32, li's unit), folded in
// 8 half-steps; after each unit li's CP_RED8 checkpoint (8 generic reductions a lane, REDG.E.XOR). The lane keeps li's
// IMAD; the LOP3 that li issues to absorb a word is the work the bulk engine now does.
//
// `main` runs on the device it finds: lbe<BT> bit-exact (one block, every lane's U checkpoints and final states against
// the host), then lbl<BT> timed at one block a SM and at full occupancy (SMSP-cycles a unit, max over blocks), with the
// L2 bytes a unit the data movement moves and the achieved L2 bandwidth from the timed cycles. h200_measure.sh keeps
// lbl's SASS for its issue count; the device run answers whether li-bulk runs at that cost or is L2-bound. li_bulk_wg.cu
// runs the same half-steps beside the IGMMA.
#include "li_bulk.cuh"

// the bit-exact kernel: one block of BT lanes, U units from `words` (lane-major in a unit, 128 words a lane a unit), out
// holds U checkpoints (zeroed), then each lane's 8 final states, then the block's SLOT-word global slot (scratch).
template <int BT>
__global__ void __launch_bounds__(BT) lbe(const uint32_t* __restrict__ words, const uint32_t* st, uint32_t* out, int U,
                                          uint32_t zm) {
  constexpr int SLOT = BT * NCH;
  extern __shared__ __align__(128) uint32_t smem[];
  const int tid = threadIdx.x;
  uint32_t* sp = smem;
  uint32_t* gblk = out + U + SLOT;
  uint32_t* gslot = gblk + NCH * tid;
  const uint32_t spb = (uint32_t)__cvta_generic_to_shared(sp) + 4u * NCH * tid;
  const uint32_t spblk = (uint32_t)__cvta_generic_to_shared(sp);
  const uint32_t skslab = (uint32_t)__cvta_generic_to_shared(smem + SLOT);
  uint32_t y[8];
#pragma unroll
  for (int c = 0; c < 8; c++) y[c] = st[8 * tid + c];
  fill_kslab<BT>(smem);
  asm volatile("fence.proxy.async;" ::: "memory");
  __syncthreads();
  for (int u = 0; u < U; u++) {
#pragma unroll 1
    for (int s = 0; s < 4; s++) {
      const uint32_t* p = words + ((size_t)u * BT + tid) * 128 + 32 * s;
      uint32_t pe0[8], po0[8], pe1[8], po1[8];
#pragma unroll
      for (int c = 0; c < 8; c++) {
        pe0[c] = p[li_entry(c, 0)];
        po0[c] = p[li_entry(c, 1)];
        pe1[c] = p[li_entry(c, 2)];
        po1[c] = p[li_entry(c, 3)];
      }
      li_half<BT, 0>(y, pe0, po0, gslot, spb, gblk, spblk, skslab);
      li_half<BT, 1>(y, pe1, po1, gslot, spb, gblk, spblk, skslab);
    }
    cp_red8(y, out + (tid & zm) + u);
  }
#pragma unroll
  for (int c = 0; c < 8; c++) out[U + 8 * tid + c] = y[c];
}

// the timed loop: a unit's words are opaque registers (they stand for the committed tensor outputs), so the SASS counts
// the fold and checkpoint machinery alone. One unit an iteration; `cpw` holds the checkpoints, `cyc` each block's span.
template <int BT>
__global__ void __launch_bounds__(BT) lbl(const uint32_t* st, uint32_t* out, uint32_t* cpw, long long* cyc, int iters,
                                          uint32_t zm) {
  constexpr int SLOT = BT * NCH;
  extern __shared__ __align__(128) uint32_t smem[];
  const int tid = threadIdx.x, gtid = blockIdx.x * BT + tid;
  uint32_t* sp = smem;
  uint32_t* gblk = out + (size_t)blockIdx.x * SLOT;
  uint32_t* gslot = gblk + NCH * tid;
  const uint32_t spb = (uint32_t)__cvta_generic_to_shared(sp) + 4u * NCH * tid;
  const uint32_t spblk = (uint32_t)__cvta_generic_to_shared(sp);
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
  // li_loop's checkpoint words: one a warp a unit, every lane reducing into it (red_generic.cu)
  uint32_t* gw = cpw + (size_t)(gtid >> 5) * iters + (tid & zm);
  fill_kslab<BT>(smem);
  asm volatile("fence.proxy.async;" ::: "memory");
  __syncthreads();
  const long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int s = 0; s < 4; s++) {
#pragma unroll
      for (int c = 0; c < 8; c++)
        asm volatile("" : "+r"(pe0[c]), "+r"(po0[c]), "+r"(pe1[c]), "+r"(po1[c]));
      li_half<BT, 0>(y, pe0, po0, gslot, spb, gblk, spblk, skslab);
      li_half<BT, 1>(y, pe1, po1, gslot, spb, gblk, spblk, skslab);
    }
    cp_red8(y, gw + it);
  }
  __syncthreads();
  const long long t1 = clock64();
#pragma unroll
  for (int c = 0; c < 8; c++) out[8 * gtid + c] ^= y[c];
  if (tid == 0) cyc[blockIdx.x] = t1 - t0;
}

#define LBE_INST(BT) template __global__ void lbe<BT>(const uint32_t*, const uint32_t*, uint32_t*, int, uint32_t);
#define LBL_INST(BT) \
  template __global__ void lbl<BT>(const uint32_t*, uint32_t*, uint32_t*, long long*, int, uint32_t);
LBE_INST(128)
LBE_INST(256)
LBL_INST(128)
LBL_INST(256)

#ifndef __CUDA_ARCH__
#include <cuda_runtime.h>

static uint32_t lcg(uint64_t& s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(s >> 32);
}

typedef void (*Exact)(const uint32_t*, const uint32_t*, uint32_t*, int, uint32_t);
typedef void (*Loop)(const uint32_t*, uint32_t*, uint32_t*, long long*, int, uint32_t);

int main() {
  int dev = 0, sms = 0, maj = 0, mnr = 0, clk = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev);
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
  printf("device cc=%d.%d sms=%d clock_khz=%d\n", maj, mnr, sms, clk);

  // bit-exact: one block of BT lanes, U units of 128 words a lane, against the host li_set + CP_RED8
  const int U = 64;
  uint64_t seed = 0x5EED2026ull;
  int all = 1;
  struct { const char* name; int BT; Exact k; } exact[] = {{"lbe_bt128", 128, lbe<128>}, {"lbe_bt256", 256, lbe<256>}};
  for (auto& e : exact) {
    const int BT = e.BT, SLOT = BT * NCH, smem = (SLOT + 2 * SLOT) * 4;
    const size_t nw = (size_t)U * BT * 128;
    std::vector<uint32_t> w(nw), st(8 * BT);
    for (auto& x : w) x = lcg(seed);
    for (auto& x : st) x = lcg(seed);
    std::vector<uint32_t> cp(U), yh(st);
    for (int u = 0; u < U; u++) {
      uint32_t x = 0;
      for (int lane = 0; lane < BT; lane++) {
        uint32_t yy[8];
        for (int c = 0; c < 8; c++) yy[c] = yh[8 * lane + c];
        for (int s = 0; s < 4; s++) li_set(yy, &w[((size_t)u * BT + lane) * 128 + 32 * s]);
        for (int c = 0; c < 8; c++) yh[8 * lane + c] = yy[c], x ^= yy[c];
      }
      cp[u] = x;
    }
    uint32_t *dw, *dst, *dout;
    const size_t nout = (size_t)U + SLOT + SLOT;  // checkpoints, final states, the block's global slot
    cudaMalloc(&dw, nw * 4);
    cudaMalloc(&dst, st.size() * 4);
    cudaMalloc(&dout, nout * 4);
    cudaMemcpy(dw, w.data(), nw * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dst, st.data(), st.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, nout * 4);
    cudaFuncSetAttribute((const void*)e.k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    e.k<<<1, BT, smem>>>(dw, dst, dout, U, 0u);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    std::vector<uint32_t> got(U + SLOT);
    cudaMemcpy(got.data(), dout, (U + SLOT) * 4, cudaMemcpyDeviceToHost);
    int bad = err != cudaSuccess;
    for (int u = 0; u < U; u++) bad += got[u] != cp[u];
    for (int i = 0; i < SLOT; i++) bad += got[U + i] != yh[i];
    all &= bad == 0;
    printf("%s bitexact=%d lanes=%d units=%d words_checked=%d bad=%d err=%s\n", e.name, bad == 0, BT, U, U + SLOT, bad,
           cudaGetErrorString(err));
    cudaFree(dw);
    cudaFree(dst);
    cudaFree(dout);
  }
  printf("all_bitexact=%d\n", all);
  std::vector<uint32_t> st(256);
  for (auto& x : st) x = lcg(seed);
  uint32_t* dst;
  cudaMalloc(&dst, 256 * 4);
  cudaMemcpy(dst, st.data(), 256 * 4, cudaMemcpyHostToDevice);

  // timing: lbl at one block a SM and at full occupancy. L2 bytes a unit (8 half-steps a unit): STG 8 + LDG 8 words a
  // lane, and the two bulk reduces RMW the block's SLOT-word slot (read + write) = 4 SLOT words a block, SLOT = BT NCH.
  const int ITERS = 1024;
  struct { const char* name; int BT; Loop k; } timed[] = {{"lbl_bt128", 128, lbl<128>}, {"lbl_bt256", 256, lbl<256>}};
  // A thread-unit's L2 bytes: 8 half-steps, each the lane's STG 8 + LDG 8 words and its share of the two bulk reduces'
  // read-modify-write of the slot (2 x 2 x 8 words): 8 x 48 words = 1536 B; li's CP_RED8 adds 8 reductions (32 B).
  const double L2_FOLD = 8.0 * (16 + 32) * 4, L2_CP = 8.0 * 4;
  printf("l2_bytes_per_unit fold=%.0f cp=%.0f (128 words a unit: %.2f B a word)\n", L2_FOLD, L2_CP,
         (L2_FOLD + L2_CP) / 128.0);
  for (auto& e : timed) {
    const int SLOT = e.BT * NCH, smem = (SLOT + 2 * SLOT) * 4;
    cudaFuncSetAttribute((const void*)e.k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, e.k, e.BT, smem);
    occ = occ < 1 ? 1 : occ;
    for (int b = 1; b <= occ; b = (b < occ ? occ : occ + 1)) {
      const int blocks = sms * b;
      uint32_t *dout2, *dcpw;
      long long* d_cyc;
      std::vector<long long> cyc(blocks);
      cudaMalloc(&dout2, (size_t)blocks * SLOT * 4);
      cudaMalloc(&dcpw, (size_t)blocks * (e.BT / 32) * ITERS * 4);
      cudaMalloc(&d_cyc, blocks * sizeof(long long));
      cudaMemset(dout2, 0, (size_t)blocks * SLOT * 4);
      cudaEvent_t t0, t1;
      cudaEventCreate(&t0);
      cudaEventCreate(&t1);
      cudaEventRecord(t0);
      e.k<<<blocks, e.BT, smem>>>(dst, dout2, dcpw, d_cyc, ITERS, 0u);
      cudaEventRecord(t1);
      cudaError_t err = cudaGetLastError();
      if (err == cudaSuccess) err = cudaEventSynchronize(t1);
      float ms = 0;
      cudaEventElapsedTime(&ms, t0, t1);
      cudaMemcpy(cyc.data(), d_cyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
      long long mx = 0;
      for (long long c : cyc) mx = c > mx ? c : mx;
      // SMSP-cycles a unit: the slowest block's SM-cycles over the units its SM folded, times the 4 SMSPs that share it
      const double units_a_sm = (double)b * e.BT * ITERS;
      const double smsp_cyc_a_unit = (double)mx * 4.0 / units_a_sm;
      const double units = (double)blocks * e.BT * ITERS, sec = ms / 1e3;
      const double l2_GBs = (L2_FOLD + L2_CP) * units / sec / 1e9;
      const double clk_eff_mhz = (double)mx / sec / 1e6;
      printf("%s blocks_per_sm=%d occ=%d warps_per_smsp=%.2f ms=%.3f units=%.0f smsp_cyc_per_unit=%.4f "
             "l2_GBps=%.1f sm_clock_from_cycles_mhz=%.0f err=%s\n",
             e.name, b, occ, b * (e.BT / 32) / 4.0, ms, units, smsp_cyc_a_unit, l2_GBs, clk_eff_mhz,
             cudaGetErrorString(err));
      cudaEventDestroy(t0);
      cudaEventDestroy(t1);
      cudaFree(dout2);
      cudaFree(dcpw);
      cudaFree(d_cyc);
    }
  }
  return all ? 0 : 3;
}
#endif
