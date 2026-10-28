// Two probes on the H200 (sm_90a) in one binary.
//
// `exact`: li's fold (segment._li_absorb: y <- (y ^ d0 ^ LK[c][h]) * LA[c][h] + d1, 8 chains a lane, li_entries) over U
// units of 128 words a lane, with the warp checkpoint (the XOR of the warp's 8·32 chain states, redux.sync.xor) in four
// placements: CP_SEG one a 128-word segment, lane 0 storing it (the design's LI-CP-W); CP_SEG4 the same, four
// segments' words in one st.global.v4; CP_SET one a 32-word set (PouwWarpCheck_v1's spacing); CP_RED CP_SEG's word
// reduced in memory (red.global.xor, no REDUX), and CP_RED2 and CP_RED8 with 2 and 8 of them a lane, the lane's XOR
// mostly in memory. Each kernel's output is checked word for word against the host's
// reference (the same formulas; the run prints the reference's words, which ref_check.py recomputes through
// segment.py), and clock.sh counts the SASS of each kernel's unit loop (LOP3, IMAD, REDUX, RED, the uniform-to-vector
// move, STG), which is the cheapest bit-exact program found, a unit.
//
// `clock`: the SM clock under the honest mixed load, read inside the kernel: each block's clock64 and %globaltimer at
// its start and end, so the clock is the block's cycles over its own nanoseconds (no launch or tail in it, unlike the
// event clock), and each MMA's MACs over its cycles give the tensor's MACs a clock beside it. The loads
// are sm90_wgmma.cu's own loops: wg_s8_n256_a1's wgmma alone, wgli_redux (wgmma with LI and the redux checkpoint),
// wgli_fill (that and ffill's filler: the design's mixed load), and the LI loop with no wgmma.
//
// `roles`: sm90_wgmma.cu's wg_spec (the JP*90 rows' block-timed points) with each role's own time (a
// test on the H200): every warp records the cycle its fixed loop ends, and each MMA warp the iterations it had done when
// the first lane warp ended (one LDS an iteration), so the joint phase's tensor and lane rates are both exact.
//
// `lirate`: the fold with the lane alone on every SM, each checkpoint placement in one loop, its cycles a unit a
// warp-slot (whether a REDG costs one issue slot and nothing else, as 's REDG routes take it).
#define main sm90_wgmma_main
#include "../dc/probes/sm90_wgmma.cu"
#undef main

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---- exact ---------------------------------------------------------------------------------------------------------

__host__ __device__ constexpr uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
__host__ __device__ constexpr uint32_t li_a(int c, int h) { return mix32(0xA0000000u + 2u * (32 + 2 * c + h)) | 1u; }
__host__ __device__ constexpr uint32_t li_k(int c, int h) { return mix32(0xB0000000u + 2u * (32 + 2 * c + h) + 1u); }
__host__ __device__ constexpr int li_entry(int c, int i) { return 4 * (c / 4 + 2 * i) + c % 4; }

// One 32-word set into the 8 chains (p: the set's words).
__host__ __device__ inline void li_set(uint32_t (&y)[8], const uint32_t* p) {
#pragma unroll
  for (int c = 0; c < 8; c++) {
#pragma unroll
    for (int h = 0; h < 2; h++)
      y[c] = (y[c] ^ p[li_entry(c, 2 * h)] ^ li_k(c, h)) * li_a(c, h) + p[li_entry(c, 2 * h + 1)];
  }
}
__host__ __device__ inline uint32_t xor8(const uint32_t (&y)[8]) {
  return y[0] ^ y[1] ^ y[2] ^ y[3] ^ y[4] ^ y[5] ^ y[6] ^ y[7];
}

// CP_RED: CP_SEG's word with the warp's XOR done by the memory system: every lane's red.global.xor.b32 of its 8 chains'
// XOR into a zeroed word, one warp instruction (REDG) where the others take REDUX, the uniform-to-vector move and a
// store. ptxas turns a red to an address it can prove warp-uniform into REDUX and one lane's REDG, so the word's
// address is out + (lane & zmask), zmask 0 at run time. CP_RED2 and CP_RED8 move more of the lane's XOR to memory: 3
// LOP3 and 2 REDG (y0..y6's XOR, then y7), and 8 REDG with no LOP3 (each chain's state), so a checkpoint's lane XOR
// trades ALU for issue: 4 LOP3 + 1 REDG, 3 + 2, 0 + 8. CP_SEG4U is CP_SEG4 with the same trade and no atomics: 3 LOP3,
// a REDUX of y0..y6's XOR and one of y7, and their XOR, which on the uniform datapath (ULOP3) would be 2 ALU cycles
// fewer for 1 issue. ptxas 13.0 doesn't emit that: it moves both REDUX results to lanes (IMAD.U32) and XORs them with a
// LOP3, so as compiled a checkpoint is 4 LOP3 + 2 REDUX + 2 moves, the same function at no saving.
enum { CP_SEG, CP_SEG4, CP_SET, CP_RED, CP_RED2, CP_RED8, CP_SEG4U };
__device__ __forceinline__ void red_xor(uint32_t* p, uint32_t v) {
  asm volatile("red.relaxed.gpu.global.xor.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// One warp a block. words: U units x 32 lanes x 128 words (lane-major in a unit), st: 32 lanes x 8 starting states.
// out (zeroed): the checkpoints (CP_SET: 4U, the others U), then each lane's final 8 states.
template <int CP>
__global__ void __launch_bounds__(32, 1) li_exact(const uint4* __restrict__ words, const uint32_t* st, uint32_t* out,
                                                 int U, uint32_t zmask) {
  const int lane = threadIdx.x;
  uint32_t* const cpw = out + (lane & zmask);
  uint32_t y[8];
#pragma unroll
  for (int c = 0; c < 8; c++) y[c] = st[8 * lane + c];
  // four units a trip (U a multiple of 4), so every placement has the same loop and CP_SEG4 a static index
#pragma unroll 1
  for (int u4 = 0; u4 < U / 4; u4++) {
    uint32_t keep[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int u = 4 * u4 + j;
      const uint4* src = words + ((size_t)u * 32 + lane) * 32;
#pragma unroll
      for (int s = 0; s < 4; s++) {
        uint32_t p[32];
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const uint4 v = src[8 * s + q];
          p[4 * q] = v.x, p[4 * q + 1] = v.y, p[4 * q + 2] = v.z, p[4 * q + 3] = v.w;
        }
        li_set(y, p);
        if constexpr (CP == CP_SET) {
          const uint32_t r = __reduce_xor_sync(0xffffffffu, xor8(y));
          if (lane == 0) out[4 * u + s] = r;
        }
      }
      if constexpr (CP == CP_SEG) {
        const uint32_t r = __reduce_xor_sync(0xffffffffu, xor8(y));
        if (lane == 0) out[u] = r;
      }
      if constexpr (CP == CP_SEG4) keep[j] = __reduce_xor_sync(0xffffffffu, xor8(y));
      if constexpr (CP == CP_SEG4U)
        keep[j] = __reduce_xor_sync(0xffffffffu, y[0] ^ y[1] ^ y[2] ^ y[3] ^ y[4] ^ y[5] ^ y[6]) ^
                  __reduce_xor_sync(0xffffffffu, y[7]);
      if constexpr (CP == CP_RED) red_xor(cpw + u, xor8(y));
      if constexpr (CP == CP_RED2) {
        red_xor(cpw + u, y[0] ^ y[1] ^ y[2] ^ y[3] ^ y[4] ^ y[5] ^ y[6]);
        red_xor(cpw + u, y[7]);
      }
      if constexpr (CP == CP_RED8)
#pragma unroll
        for (int c = 0; c < 8; c++) red_xor(cpw + u, y[c]);
    }
    if constexpr (CP == CP_SEG4 || CP == CP_SEG4U)
      if (lane == 0) reinterpret_cast<uint4*>(out)[u4] = make_uint4(keep[0], keep[1], keep[2], keep[3]);
  }
  const int ncp = CP == CP_SET ? 4 * U : U;
#pragma unroll
  for (int c = 0; c < 8; c++) out[ncp + 8 * lane + c] = y[c];
}

static uint32_t lcg(uint64_t& s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(s >> 32);
}

static int run_exact(int U) {
  const size_t nw = (size_t)U * 32 * 128;
  std::vector<uint32_t> w(nw), st(256);
  uint64_t seed = 0x5EED2026ull;
  for (auto& x : w) x = lcg(seed);
  for (auto& x : st) x = lcg(seed);
  // reference: per lane, the same order
  std::vector<uint32_t> cp_seg(U), cp_set(4 * U), fin(256);
  std::vector<uint32_t> y(256);
  for (int i = 0; i < 256; i++) y[i] = st[i];
  for (int u = 0; u < U; u++) {
    for (int s = 0; s < 4; s++) {
      uint32_t x = 0;
      for (int lane = 0; lane < 32; lane++) {
        uint32_t yy[8];
        for (int c = 0; c < 8; c++) yy[c] = y[8 * lane + c];
        li_set(yy, &w[((size_t)u * 32 + lane) * 128 + 32 * s]);
        for (int c = 0; c < 8; c++) y[8 * lane + c] = yy[c];
        x ^= xor8(yy);
      }
      cp_set[4 * u + s] = x;
      if (s == 3) cp_seg[u] = x;
    }
  }
  uint32_t *dw, *dst, *dout;
  cudaMalloc(&dw, nw * 4);
  cudaMalloc(&dst, 256 * 4);
  cudaMalloc(&dout, (4 * U + 256) * 4);
  cudaMemcpy(dw, w.data(), nw * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dst, st.data(), 256 * 4, cudaMemcpyHostToDevice);
  // the reference's words, for ref_check.py to recompute through segment.py's own _li_absorb and warp_checkpoint
  auto list = [](const std::vector<uint32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
  };
  printf("{\"probe\": \"ref\", \"units\": %d, \"seed\": %llu, \"cp_seg\": %s, \"cp_set\": %s, \"final\": %s}\n", U,
         0x5EED2026ull, list(cp_seg).c_str(), list(cp_set).c_str(), list(y).c_str());
  int bad = 0;
  const char* names[7] = {"li_exact<CP_SEG>",  "li_exact<CP_SEG4>", "li_exact<CP_SET>",  "li_exact<CP_RED>",
                          "li_exact<CP_RED2>", "li_exact<CP_RED8>", "li_exact<CP_SEG4U>"};
  for (int cp = 0; cp < 7; cp++) {
    cudaMemset(dout, 0, (4 * U + 256) * 4);
    if (cp == 0) li_exact<CP_SEG><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    if (cp == 1) li_exact<CP_SEG4><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    if (cp == 2) li_exact<CP_SET><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    if (cp == 3) li_exact<CP_RED><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    if (cp == 4) li_exact<CP_RED2><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    if (cp == 5) li_exact<CP_RED8><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    if (cp == 6) li_exact<CP_SEG4U><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    cudaError_t e = cudaDeviceSynchronize();
    const int ncp = cp == 2 ? 4 * U : U;
    std::vector<uint32_t> got(ncp + 256);
    cudaMemcpy(got.data(), dout, (ncp + 256) * 4, cudaMemcpyDeviceToHost);
    int mism = 0;
    for (int i = 0; i < ncp; i++) mism += got[i] != (cp == 2 ? cp_set[i] : cp_seg[i]);
    for (int i = 0; i < 256; i++) mism += got[ncp + i] != y[i];
    printf("{\"probe\": \"exact\", \"kernel\": \"%s\", \"units\": %d, \"checkpoints\": %d, \"words_checked\": %d, "
           "\"mismatches\": %d, \"cuda\": \"%s\"}\n",
           names[cp], U, ncp, ncp + 256, mism, cudaGetErrorString(e));
    bad += mism != 0 || e != cudaSuccess;
  }
  cudaFree(dw);
  cudaFree(dst);
  cudaFree(dout);
  return bad;
}

// ---- clock ---------------------------------------------------------------------------------------------------------

struct Stamp {
  unsigned long long g0, g1;
  long long c0, c1;
};
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
DC_DEV void li_alone(const OpArgs& a) {
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) { li_fold<1>(l); }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}
enum { LD_WGMMA, LD_WGLI, LD_WGLI_FILL, LD_LI };
template <int LD>
__global__ void __launch_bounds__(256, 1) clk(OpArgs a, Stamp* st) {
  __syncthreads();
  const unsigned long long g0 = gtimer();
  const long long c0 = clock64();
  if constexpr (LD == LD_WGMMA) wg_alone<F_s8_n256, 1>(a);
  if constexpr (LD == LD_WGLI) wg_li<1>(a);
  if constexpr (LD == LD_WGLI_FILL) wg_li_fill(a);
  if constexpr (LD == LD_LI) li_alone(a);
  __syncthreads();
  if (threadIdx.x == 0) st[blockIdx.x] = {g0, gtimer(), c0, clock64()};
}

static int run_clock(int load, int iters, int launches) {
  int dev = 0, sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const char* names[4] = {"wg_s8_n256_a1", "wgli_redux", "wgli_fill", "li_alone"};
  // MACs a block-iteration: 2 warpgroups x 4 wgmma m64n256k32
  const double macs_it = load == LD_LI ? 0.0 : 2.0 * 4 * 64 * 256 * 32;
  OpArgs a{};
  cudaMalloc(&a.rec, sms * sizeof(OpClk));
  cudaMalloc(&a.sink, (size_t)sms * 256 * sizeof(unsigned));
  a.g = nullptr;
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
    if (load == LD_WGMMA) clk<LD_WGMMA><<<sms, 256>>>(a, dst);
    if (load == LD_WGLI) clk<LD_WGLI><<<sms, 256>>>(a, dst);
    if (load == LD_WGLI_FILL) clk<LD_WGLI_FILL><<<sms, 256>>>(a, dst);
    if (load == LD_LI) clk<LD_LI><<<sms, 256>>>(a, dst);
    cudaEventRecord(e1);
    cudaError_t e = cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaMemcpy(st.data(), dst, sms * sizeof(Stamp), cudaMemcpyDeviceToHost);
    cudaMemcpy(rec.data(), a.rec, sms * sizeof(OpClk), cudaMemcpyDeviceToHost);
    std::vector<double> mhz(sms), cyc_it(sms);
    long long cmax = 0;
    unsigned long long gmin = ~0ull, gmax = 0;
    for (int b = 0; b < sms; b++) {
      mhz[b] = 1e3 * (double)(st[b].c1 - st[b].c0) / (double)(st[b].g1 - st[b].g0);
      cyc_it[b] = (double)rec[b].cyc / iters;
      cmax = std::max(cmax, st[b].c1 - st[b].c0);
      gmin = std::min(gmin, st[b].g0);
      gmax = std::max(gmax, st[b].g1);
    }
    std::vector<double> sm = mhz, sc = cyc_it;
    std::sort(sm.begin(), sm.end());
    std::sort(sc.begin(), sc.end());
    const double med_cyc = sc[sms / 2];
    printf("{\"probe\": \"clock\", \"load\": \"%s\", \"launch\": %d, \"iters\": %d, \"blocks\": %d, \"event_ms\": %.4f, "
           "\"span_ms\": %.4f, \"mhz_min\": %.2f, \"mhz_med\": %.2f, \"mhz_max\": %.2f, \"event_mhz\": %.2f, "
           "\"cycles_per_iter_med\": %.3f, \"macs_per_clock_sm\": %.2f, \"cuda\": \"%s\"}\n",
           names[load], L, iters, sms, ms, (gmax - gmin) / 1e6, sm[0], sm[sms / 2], sm[sms - 1], 1e-3 * cmax / ms,
           med_cyc, macs_it / med_cyc, cudaGetErrorString(e));
    fflush(stdout);
    bad += e != cudaSuccess;
  }
  return bad;
}

// ---- roles ---------------------------------------------------------------------------------------------------------

// wg_spec<X, R> (warps 0-7 the two wgmma warpgroups, warps 8-11 one lane warp a SMSP), each warp's loop end recorded in
// ends[block][warp], and in at_lane[block][mma warp] the MMA iterations done when the first lane warp ended (-1: none
// ended first).
template <int X, int R>
__global__ void __launch_bounds__(384, 1) roles(OpArgs a, long long* ends, int* at_lane) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  __shared__ int lane_done;
  if (threadIdx.x == 0) lane_done = 0;
  const Tiles t = fill_tiles<0>(sm, a.mode);
  const bool mma_role = threadIdx.x < 256;
  int d[128] = {};
  Lane l(threadIdx.x);
  int seen = -1;
  OP_BEGIN
  if (mma_role) {
    OP_LOOP(it) {
      group4<F_s8_n256>(d, t);
      wgmma_wait<1>();
      if (seen < 0 && *(volatile int*)&lane_done) seen = it;
    }
    wgmma_wait<0>();
    wgmma_hold(d);
  } else {
    OP_LOOP(it) { lane_ops<X, R>(l); }
    if ((threadIdx.x & 31) == 0) atomicExch(&lane_done, 1);
  }
  const long long end = clock64() - op_t0_;
  if ((threadIdx.x & 31) == 0) {
    ends[blockIdx.x * 12 + (threadIdx.x >> 5)] = end;
    if (mma_role) at_lane[blockIdx.x * 8 + (threadIdx.x >> 5)] = seen;
  }
  OP_END
  OP_SINK((mma_role ? fold_acc(d) : l.fold()) ^ threadIdx.x)
}

template <int X, int R>
static int run_roles(const char* name, int iters, int reps) {
  int dev = 0, sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  OpArgs a{};
  cudaMalloc(&a.rec, sms * sizeof(OpClk));
  cudaMalloc(&a.sink, (size_t)sms * 384 * sizeof(unsigned));
  a.iters = iters;
  long long* dends;
  int* dat;
  cudaMalloc(&dends, sms * 12 * sizeof(long long));
  cudaMalloc(&dat, sms * 8 * sizeof(int));
  std::vector<long long> ends(sms * 12);
  std::vector<int> at(sms * 8);
  std::vector<OpClk> rec(sms);
  int bad = 0;
  for (int r = 0; r < reps; r++) {
    roles<X, R><<<sms, 384>>>(a, dends, dat);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(ends.data(), dends, ends.size() * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(at.data(), dat, at.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(rec.data(), a.rec, sms * sizeof(OpClk), cudaMemcpyDeviceToHost);
    printf("{\"probe\": \"roles\", \"name\": \"%s\", \"lane_ops_iter\": %d, \"rep\": %d, \"iters\": %d, \"blocks\": %d, "
           "\"cuda\": \"%s\", \"block_cyc\": [",
           name, R, r, iters, sms, cudaGetErrorString(e));
    for (int b = 0; b < sms; b++) printf("%s%lld", b ? ", " : "", rec[b].cyc);
    printf("], \"ends\": [");
    for (int i = 0; i < sms * 12; i++) printf("%s%lld", i ? ", " : "", ends[i]);
    printf("], \"mma_iters_at_lane_end\": [");
    for (int i = 0; i < sms * 8; i++) printf("%s%d", i ? ", " : "", at[i]);
    printf("]}\n");
    fflush(stdout);
    bad += e != cudaSuccess;
  }
  cudaFree(dends);
  cudaFree(dat);
  cudaFree(a.rec);
  cudaFree(a.sink);
  return bad;
}

static int run_all_roles(int iters, int reps) {
  int bad = 0;
  bad += run_roles<L_LOP3, 64>("roles_lop3_r64", iters, reps);
  bad += run_roles<L_LOP3, 128>("roles_lop3_r128", iters, reps);
  bad += run_roles<L_LOP3, 192>("roles_lop3_r192", iters, reps);
  bad += run_roles<L_LOP3, 256>("roles_lop3_r256", iters, reps);
  bad += run_roles<L_IMAD, 64>("roles_imad_r64", iters, reps);
  bad += run_roles<L_IMAD, 128>("roles_imad_r128", iters, reps);
  bad += run_roles<L_IMAD, 192>("roles_imad_r192", iters, reps);
  bad += run_roles<L_IMAD, 256>("roles_imad_r256", iters, reps);
  return bad;
}

// ---- lirate --------------------------------------------------------------------------------------------------------

// The fold's cost on the chip with the lane alone, every checkpoint in the same loop: W warps a block, one block a SM
// (W/4 a SMSP), fold 4 units an iteration (64 LOP3 + 64 IMAD a unit on the Lane's 8 chains, li_fold's ops) and
// checkpoint each unit into a ring of 1024 words a warp. CK_NONE has none (the chains' floor); CK_SEG is LI-CP-W's (4
// LOP3, REDUX, the uniform-to-vector move, lane 0's STG); CK_RED, CK_RED2 and CK_RED8 are li_exact's CP_RED, CP_RED2 and
// CP_RED8 (4 LOP3 + 1 REDG, 3 + 2, 0 + 8), all 32 lanes into the warp's one word (zm 0 at run time); CK_SEG4 is
// li_exact's CP_SEG4, CK_SEG with the trip's four words kept and lane 0 storing them in one STG.128; CK_SEG4U is
// CP_SEG4U (3 LOP3, two REDUX, their XOR), and CK_SEG4M three CK_SEG4 units and one CK_SEG4U a trip, the mix at which
// issue and the ALU pipe would balance with the XOR on the uniform datapath (134.5 and 135.5 a unit against CK_SEG4's
// 134.25 and 136). ptxas puts that XOR on lanes (CP_SEG4U above), so both time the compiled form. A unit a warp-slot
// is the block's cycles over iters x 4 units x W/4 warps a SMSP: on a loop bound by issue or the ALU pipe (LOP3 at half
// rate), max(issue, 2 x LOP3) a unit. A REDG that costs more than its issue slot (the L2's atomic unit taking a warp's 32
// lanes into one word) shows as CK_RED8 above that.
enum { CK_NONE, CK_SEG, CK_RED, CK_RED2, CK_RED8, CK_SEG4, CK_SEG4U, CK_SEG4M };
__device__ __forceinline__ uint32_t lop3x(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
// CK_SEG4's and CK_SEG4U's word, which its caller stores; 0 for the others
template <int K>
DC_DEV uint32_t li_ck(const uint32_t (&r)[8], uint32_t* w, int lane) {
  if constexpr (K == CK_SEG4U) {
    uint32_t x = lop3x(lop3x(lop3x(r[0], r[1], r[2]), r[3], r[4]), r[5], r[6]), y = r[7];
    asm volatile("redux.sync.xor.b32 %0,%0,0xffffffff;" : "+r"(x));
    asm volatile("redux.sync.xor.b32 %0,%0,0xffffffff;" : "+r"(y));
    return x ^ y;
  }
  if constexpr (K == CK_SEG || K == CK_RED || K == CK_SEG4) {
    uint32_t x = lop3x(lop3x(lop3x(r[0], r[1], r[2]), r[3], r[4]), r[5], r[6]);
    asm volatile("xor.b32 %0,%0,%1;" : "+r"(x) : "r"(r[7]));
    if constexpr (K == CK_RED) {
      red_xor(w, x);
    } else {
      asm volatile("redux.sync.xor.b32 %0,%0,0xffffffff;" : "+r"(x));
      if constexpr (K == CK_SEG4) return x;
      if (lane == 0) asm volatile("st.global.b32 [%0], %1;" ::"l"(w), "r"(x) : "memory");
    }
  }
  if constexpr (K == CK_RED2) {
    red_xor(w, lop3x(lop3x(lop3x(r[0], r[1], r[2]), r[3], r[4]), r[5], r[6]));
    red_xor(w, r[7]);
  }
  if constexpr (K == CK_RED8)
#pragma unroll
    for (int c = 0; c < 8; c++) red_xor(w, r[c]);
  return 0;
}
template <int K, int W>
__global__ void __launch_bounds__(W * 32, 1) lirate(OpArgs a, uint32_t* cp, uint32_t zm) {
  Lane l(threadIdx.x);
  const int lane = threadIdx.x & 31;
  uint32_t* const w = cp + ((size_t)blockIdx.x * W + (threadIdx.x >> 5)) * 1024 + (lane & zm);
  OP_BEGIN
  OP_LOOP(it) {
    uint32_t* const wi = w + ((it & 255) << 2);
    uint32_t keep[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
#pragma unroll
      for (int c = 0; c < 64; c++) {
        l.op<L_LOP3>(c);
        l.op<L_IMAD>(c);
      }
      if constexpr (K == CK_SEG4M)
        keep[u] = u == 3 ? li_ck<CK_SEG4U>(l.r, wi + u, lane) : li_ck<CK_SEG4>(l.r, wi + u, lane);
      else
        keep[u] = li_ck<K>(l.r, wi + u, lane);
    }
    if constexpr (K == CK_SEG4 || K == CK_SEG4U || K == CK_SEG4M)
      if (lane == 0)
        asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(wi), "r"(keep[0]), "r"(keep[1]), "r"(keep[2]),
                     "r"(keep[3])
                     : "memory");
  }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}

// lirate_mix<NR, W>: 8 units an iteration, the first NR checkpointed by CK_RED2 and the rest by CK_SEG, so NR/8 of the
// units move their lane XOR to memory. CK_RED2 throughout is bound by the atomic unit (its 32 lanes into one word); a
// cheater would use only its spare capacity, and this times that mix against NR = 0, LI-CP-W in the same loop.
template <int NR, int W>
__global__ void __launch_bounds__(W * 32, 1) lirate_mix(OpArgs a, uint32_t* cp, uint32_t zm) {
  Lane l(threadIdx.x);
  const int lane = threadIdx.x & 31;
  uint32_t* const w = cp + ((size_t)blockIdx.x * W + (threadIdx.x >> 5)) * 1024 + (lane & zm);
  OP_BEGIN
  OP_LOOP(it) {
    uint32_t* const wi = w + ((it & 127) << 3);
#pragma unroll
    for (int u = 0; u < 8; u++) {
#pragma unroll
      for (int c = 0; c < 64; c++) {
        l.op<L_LOP3>(c);
        l.op<L_IMAD>(c);
      }
      if (u < NR) li_ck<CK_RED2>(l.r, wi + u, lane);
      else li_ck<CK_SEG>(l.r, wi + u, lane);
    }
  }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}

template <int NR, int W>
static int run_lirate_mix(int iters, int reps) {
  int dev = 0, sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  OpArgs a{};
  cudaMalloc(&a.rec, sms * sizeof(OpClk));
  cudaMalloc(&a.sink, (size_t)sms * W * 32 * sizeof(unsigned));
  a.iters = iters;
  uint32_t* cp;
  cudaMalloc(&cp, (size_t)sms * W * 1024 * 4);
  cudaMemset(cp, 0, (size_t)sms * W * 1024 * 4);
  std::vector<OpClk> rec(sms);
  int bad = 0;
  for (int r = 0; r < reps; r++) {
    lirate_mix<NR, W><<<sms, W * 32>>>(a, cp, 0u);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(rec.data(), a.rec, sms * sizeof(OpClk), cudaMemcpyDeviceToHost);
    std::vector<double> u(sms);
    for (int b = 0; b < sms; b++) u[b] = (double)rec[b].cyc / iters / 8 / (W / 4);
    std::sort(u.begin(), u.end());
    printf("{\"probe\": \"lirate_mix\", \"red2_of_8\": %d, \"warps\": %d, \"rep\": %d, \"iters\": %d, \"blocks\": %d, "
           "\"cyc_unit_slot_min\": %.3f, \"cyc_unit_slot_med\": %.3f, \"cyc_unit_slot_max\": %.3f, \"cuda\": \"%s\"}\n",
           NR, W, r, iters, sms, u[0], u[sms / 2], u[sms - 1], cudaGetErrorString(e));
    fflush(stdout);
    bad += e != cudaSuccess;
  }
  cudaFree(cp);
  cudaFree(a.rec);
  cudaFree(a.sink);
  return bad;
}

template <int K, int W>
static int run_lirate(const char* name, int iters, int reps) {
  int dev = 0, sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  OpArgs a{};
  cudaMalloc(&a.rec, sms * sizeof(OpClk));
  cudaMalloc(&a.sink, (size_t)sms * W * 32 * sizeof(unsigned));
  a.iters = iters;
  uint32_t* cp;
  cudaMalloc(&cp, (size_t)sms * W * 1024 * 4);
  cudaMemset(cp, 0, (size_t)sms * W * 1024 * 4);
  std::vector<OpClk> rec(sms);
  int bad = 0;
  for (int r = 0; r < reps; r++) {
    lirate<K, W><<<sms, W * 32>>>(a, cp, 0u);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(rec.data(), a.rec, sms * sizeof(OpClk), cudaMemcpyDeviceToHost);
    std::vector<double> u(sms);
    for (int b = 0; b < sms; b++) u[b] = (double)rec[b].cyc / iters / 4 / (W / 4);
    std::sort(u.begin(), u.end());
    printf("{\"probe\": \"lirate\", \"name\": \"%s\", \"warps\": %d, \"rep\": %d, \"iters\": %d, \"blocks\": %d, "
           "\"cyc_unit_slot_min\": %.3f, \"cyc_unit_slot_med\": %.3f, \"cyc_unit_slot_max\": %.3f, \"cuda\": \"%s\"}\n",
           name, W, r, iters, sms, u[0], u[sms / 2], u[sms - 1], cudaGetErrorString(e));
    fflush(stdout);
    bad += e != cudaSuccess;
  }
  cudaFree(cp);
  cudaFree(a.rec);
  cudaFree(a.sink);
  return bad;
}

template <int W>
static int run_lirate_w(int iters, int reps) {
  int bad = 0;
  bad += run_lirate<CK_NONE, W>("ck_none", iters, reps);
  bad += run_lirate<CK_SEG, W>("ck_seg", iters, reps);
  bad += run_lirate<CK_RED, W>("ck_red", iters, reps);
  bad += run_lirate<CK_RED2, W>("ck_red2", iters, reps);
  bad += run_lirate<CK_RED8, W>("ck_red8", iters, reps);
  bad += run_lirate<CK_SEG4, W>("ck_seg4", iters, reps);
  bad += run_lirate<CK_SEG4U, W>("ck_seg4u", iters, reps);
  bad += run_lirate<CK_SEG4M, W>("ck_seg4m", iters, reps);
  return bad;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "exact")) return run_exact(argc >= 3 ? atoi(argv[2]) : 64) ? 1 : 0;
  if (argc >= 2 && !strcmp(argv[1], "lirate")) {
    const int iters = argc >= 3 ? atoi(argv[2]) : 20000, reps = argc >= 4 ? atoi(argv[3]) : 5;
    return run_lirate_w<8>(iters, reps) + run_lirate_w<16>(iters, reps) ? 1 : 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "lirmix")) {
    const int iters = argc >= 3 ? atoi(argv[2]) : 10000, reps = argc >= 4 ? atoi(argv[3]) : 7;
    int bad = 0;
    bad += run_lirate_mix<0, 16>(iters, reps);
    bad += run_lirate_mix<1, 16>(iters, reps);
    bad += run_lirate_mix<2, 16>(iters, reps);
    bad += run_lirate_mix<3, 16>(iters, reps);
    bad += run_lirate_mix<8, 16>(iters, reps);
    bad += run_lirate_mix<0, 16>(iters, reps);
    return bad ? 1 : 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "roles"))
    return run_all_roles(argc >= 3 ? atoi(argv[2]) : 4096, argc >= 4 ? atoi(argv[3]) : 3) ? 1 : 0;
  if (argc >= 5 && !strcmp(argv[1], "clock")) return run_clock(atoi(argv[2]), atoi(argv[3]), atoi(argv[4])) ? 1 : 0;
  // sm90_wgmma.cu's own probes by name (its op_main: --time NAME... [--reps R]), e.g. wgspec_lop3_r256 beside wgU_lop3
  if (argc >= 2 && !strcmp(argv[1], "probe")) return sm90_wgmma_main(argc - 1, argv + 1);
  fprintf(stderr, "usage: sm90_clock exact [U] | clock LOAD(0 wgmma, 1 wgli, 2 wgli_fill, 3 li) ITERS LAUNCHES | roles "
                  "[ITERS [REPS]] | lirate [ITERS [REPS]] | lirmix [ITERS [REPS]] | probe --time NAME...\n");
  return 2;
}
