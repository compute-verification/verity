// RT2-HOPPER (note:20261019T1140Z-draft-rt2-hopper) on the H200 (sm_90a), two probes in one binary.
//
// `exact`: li's fold (segment._li_absorb: y <- (y ^ d0 ^ LK[c][h]) * LA[c][h] + d1, 8 chains a lane, li_entries) over U
// units of 128 words a lane, with the warp checkpoint (the XOR of the warp's 8·32 chain states, redux.sync.xor) in four
// placements: CP_SEG one a 128-word segment, lane 0 storing it (the design's LI-CP-W); CP_SEG4 the same, four
// segments' words in one st.global.v4; CP_SET one a 32-word set (PouwWarpCheck_v1's spacing); CP_RED CP_SEG's word
// reduced in memory (red.global.xor, no REDUX). Each kernel's output is checked word for word against the host's
// reference (the same formulas; the run prints the reference's words, which ref_check.py recomputes through
// segment.py), and clock.sh counts the SASS of each kernel's unit loop (LOP3, IMAD, REDUX, RED, the uniform-to-vector
// move, STG), which is the cheapest bit-exact program found, a unit.
//
// `clock`: the SM clock under the honest mixed load, read inside the kernel: each block's clock64 and %globaltimer at
// its start and end, so the clock is the block's cycles over its own nanoseconds (no launch or tail in it, unlike the
// event clock, RT-H200-EMP-4), and each MMA's MACs over its cycles give the tensor's MACs a clock beside it. The loads
// are sm90_wgmma.cu's own loops: wg_s8_n256_a1's wgmma alone, wgli_redux (wgmma with LI and the redux checkpoint),
// wgli_fill (that and ffill's filler: the design's mixed load), and the LI loop with no wgmma.
//
// `roles`: sm90_wgmma.cu's wg_spec (the JP*90 rows' block-timed points) with each role's own time (RT-BLACKWELL-16's
// test on the H200): every warp records the cycle its fixed loop ends, and each MMA warp the iterations it had done when
// the first lane warp ended (one LDS an iteration), so the joint phase's tensor and lane rates are both exact.
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
// address is out + (lane & zmask), zmask 0 at run time.
enum { CP_SEG, CP_SEG4, CP_SET, CP_RED };

// One warp a block. words: U units x 32 lanes x 128 words (lane-major in a unit), st: 32 lanes x 8 starting states.
// out (zeroed): the checkpoints (CP_SEG, CP_SEG4, CP_RED: U; CP_SET: 4U), then each lane's final 8 states.
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
      if constexpr (CP == CP_RED)
        asm volatile("red.relaxed.gpu.global.xor.b32 [%0], %1;" ::"l"(cpw + u), "r"(xor8(y)) : "memory");
    }
    if constexpr (CP == CP_SEG4)
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
  const char* names[4] = {"li_exact<CP_SEG>", "li_exact<CP_SEG4>", "li_exact<CP_SET>", "li_exact<CP_RED>"};
  for (int cp = 0; cp < 4; cp++) {
    cudaMemset(dout, 0, (4 * U + 256) * 4);
    if (cp == 0) li_exact<CP_SEG><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    if (cp == 1) li_exact<CP_SEG4><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    if (cp == 2) li_exact<CP_SET><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    if (cp == 3) li_exact<CP_RED><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
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

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "exact")) return run_exact(argc >= 3 ? atoi(argv[2]) : 64) ? 1 : 0;
  if (argc >= 2 && !strcmp(argv[1], "roles"))
    return run_all_roles(argc >= 3 ? atoi(argv[2]) : 4096, argc >= 4 ? atoi(argv[3]) : 3) ? 1 : 0;
  if (argc >= 5 && !strcmp(argv[1], "clock")) return run_clock(atoi(argv[2]), atoi(argv[3]), atoi(argv[4])) ? 1 : 0;
  // sm90_wgmma.cu's own probes by name (its op_main: --time NAME... [--reps R]), e.g. wgspec_lop3_r256 beside wgU_lop3
  if (argc >= 2 && !strcmp(argv[1], "probe")) return sm90_wgmma_main(argc - 1, argv + 1);
  fprintf(stderr, "usage: sm90_clock exact [U] | clock LOAD(0 wgmma, 1 wgli, 2 wgli_fill, 3 li) ITERS LAUNCHES | roles "
                  "[ITERS [REPS]] | probe --time NAME...\n");
  return 2;
}
