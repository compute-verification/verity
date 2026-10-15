// Memory-side census (item 4): what the L2 and the TMA compute without the SM's pipes, and whether it competes with them.
//   red:   red.global / atom.global adds (f32, u32, u64, f64, v4.f32, bf16x2, v4.bf16x2, returning u32/f32) into an
//          L2-resident 32 MiB buffer and a DRAM-resident 4 GiB one, with three address patterns: coalesced (a warp's
//          lanes on consecutive slots), scattered (every lane on its own random 32-byte sector), hot (a warp's lanes all
//          on one address). Reports element adds, lane ops and warp instructions a second.
//   bulk:  TMA bulk loads (cp.async.bulk global->shared, mbarrier) and TMA bulk reductions (cp.reduce.async.bulk
//          .add.f32/.u32/.noftz.bf16) streaming through L2-resident and DRAM-resident buffers, bytes and adds a second.
//   mix:   warp-specialised co-runs in one block a SM (FFMA, QMMA, RED v4, L2 reads): each role's rate alone and beside
//          the other, timed by each role's own warps' clock64.
#include "common.cuh"

#include <cstring>

constexpr size_t L2_BYTES = 32ull << 20;
constexpr size_t DRAM_BYTES = 4ull << 30;

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}

// ---- red / atom -------------------------------------------------------------------------------------------------
#define DEF_RED(NAME, BYTES_, ELEMS_, CODE)                                                       \
  struct NAME {                                                                                   \
    static constexpr const char* name = #NAME;                                                    \
    static constexpr int BYTES = BYTES_, ELEMS = ELEMS_;                                          \
    __device__ __forceinline__ static void op(char* p, uint32_t v, uint32_t& acc) { CODE; }       \
  };
DEF_RED(red_f32, 4, 1, asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(1.0f) : "memory"))
DEF_RED(red_u32, 4, 1, asm volatile("red.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"))
DEF_RED(red_u64, 8, 1, asm volatile("red.global.add.u64 [%0], %1;" ::"l"(p), "l"((unsigned long long)v) : "memory"))
DEF_RED(red_f64, 8, 1, asm volatile("red.global.add.f64 [%0], %1;" ::"l"(p), "d"(1.0) : "memory"))
DEF_RED(red_v4_f32, 16, 4,
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(1.f), "f"(1.f), "f"(1.f), "f"(1.f)
                     : "memory"))
DEF_RED(red_bf16x2, 4, 2, asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" ::"l"(p), "r"(0x3f803f80u) : "memory"))
DEF_RED(red_v4_bf16x2, 16, 8,
        asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(0x3f803f80u), "r"(0x3f803f80u),
                     "r"(0x3f803f80u), "r"(0x3f803f80u)
                     : "memory"))
DEF_RED(atom_u32, 4, 1, uint32_t r; asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory");
        acc += r)
DEF_RED(atom_f32, 4, 1, float r; asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(r) : "l"(p), "f"(1.0f) : "memory");
        acc += __float_as_uint(r))

enum { PAT_COALESCED = 0, PAT_SCATTERED = 1, PAT_HOT = 2 };
static const char* PAT_NAME[] = {"coalesced", "scattered", "hot"};

template <class O, int PAT>
__global__ void k_red(char* buf, uint64_t slots, int iters, uint32_t* sink) {
  const uint32_t lane = threadIdx.x & 31;
  const uint64_t gw = (blockIdx.x * (uint64_t)blockDim.x + threadIdx.x) >> 5;
  const uint64_t nw = ((uint64_t)gridDim.x * blockDim.x) >> 5;
  const uint64_t mask = slots - 1;
  uint32_t acc = 0;
  for (int i = 0; i < iters; i++) {
    uint64_t slot;
    if (PAT == PAT_COALESCED) slot = (((uint64_t)i * nw + gw) * 32 + lane) & mask;
    else if (PAT == PAT_SCATTERED) slot = ((uint64_t)hash32((uint32_t)(gw * 32 + lane) * 0x9e3779b9u + i) * (32 / O::BYTES)) & mask;
    else slot = (((uint64_t)i * nw + gw) * (128 / O::BYTES)) & mask;
    O::op(buf + slot * O::BYTES, 1u, acc);
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

static char *g_l2buf, *g_drambuf;
static uint32_t* g_sink;
static BlockClk* g_rec;
static int g_sms;

template <class O, int PAT>
static void red_one(bool dram, const char* only) {
  char label[96];
  snprintf(label, sizeof label, "red:%s:%s:%s", O::name, PAT_NAME[PAT], dram ? "dram" : "l2");
  if (only && !strstr(label, only)) return;
  char* buf = dram ? g_drambuf : g_l2buf;
  uint64_t slots = (dram ? DRAM_BYTES : L2_BYTES) / O::BYTES;
  const int blocks = g_sms * 8, threads = 256;
  auto go = [&](int iters) { return best_ms([&] { k_red<O, PAT><<<blocks, threads>>>(buf, slots, iters, g_sink); }, 3); };
  double ms0 = go(64);
  int iters = std::max(64, (int)(64 * 20.0 / std::max(ms0, 1e-3)));
  iters = std::min(iters, 1 << 20);
  double ms = go(iters);
  double lane_ops = (double)blocks * threads * iters;
  printf("{\"probe\":\"red\",\"op\":\"%s\",\"pattern\":\"%s\",\"target\":\"%s\",\"bytes_per_lane\":%d,\"elems_per_lane\":%d,"
         "\"iters\":%d,\"ms\":%.4f,\"lane_ops_per_s\":%.4e,\"warp_instr_per_s\":%.4e,\"elem_adds_per_s\":%.4e,"
         "\"payload_bytes_per_s\":%.4e}\n",
         O::name, PAT_NAME[PAT], dram ? "dram" : "l2", O::BYTES, O::ELEMS, iters, ms, lane_ops / (ms * 1e-3),
         lane_ops / 32 / (ms * 1e-3), lane_ops * O::ELEMS / (ms * 1e-3), lane_ops * O::BYTES / (ms * 1e-3));
  fflush(stdout);
}

template <class O>
static void red_all(const char* only) {
  red_one<O, PAT_COALESCED>(false, only);
  red_one<O, PAT_SCATTERED>(false, only);
  red_one<O, PAT_HOT>(false, only);
  red_one<O, PAT_COALESCED>(true, only);
  red_one<O, PAT_SCATTERED>(true, only);
}

// ---- TMA bulk loads and bulk reductions ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  while (!ok)
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
}

constexpr int BULK_STAGES = 4;

// One elected thread a block streams `iters` chunks of `chunk` bytes (blocks interleaved) into a ring of shared stages.
__global__ void k_bulk_load(const char* src, uint64_t bytes, int chunk, int iters, uint32_t* sink) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) unsigned long long bars[BULK_STAGES];
  if (threadIdx.x == 0) {
    for (int s = 0; s < BULK_STAGES; s++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bars[s])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const uint64_t nch = bytes / chunk;
    for (int i = 0; i < iters; i++) {
      int s = i % BULK_STAGES;
      if (i >= BULK_STAGES) mbar_wait(smem_u32(&bars[s]), ((i / BULK_STAGES) - 1) & 1);
      const char* g = src + (((uint64_t)i * gridDim.x + blockIdx.x) % nch) * chunk;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bars[s])), "r"(chunk) : "memory");
      asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                       smem_u32(sm + s * chunk)),
                   "l"(g), "r"(chunk), "r"(smem_u32(&bars[s]))
                   : "memory");
    }
    for (int i = iters > BULK_STAGES ? iters - BULK_STAGES : 0; i < iters; i++)
      mbar_wait(smem_u32(&bars[i % BULK_STAGES]), (i / BULK_STAGES) & 1);
    sink[blockIdx.x] = sm[threadIdx.x];
  }
}

template <int OPK>
__global__ void k_bulk_red(char* dst, uint64_t bytes, int chunk, int iters) {
  extern __shared__ __align__(128) char sm[];
  for (int i = threadIdx.x; i < chunk / 4; i += blockDim.x) ((uint32_t*)sm)[i] = OPK == 2 ? 0x3f803f80u : (OPK == 0 ? 0x3f800000u : 1u);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint64_t nch = bytes / chunk;
    uint32_t s = smem_u32(sm);
    for (int i = 0; i < iters; i++) {
      char* g = dst + (((uint64_t)i * gridDim.x + blockIdx.x) % nch) * chunk;
      if (OPK == 0)
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" ::"l"(g), "r"(s), "r"(chunk) : "memory");
      else if (OPK == 1)
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" ::"l"(g), "r"(s), "r"(chunk) : "memory");
      else
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], %2;" ::"l"(g), "r"(s), "r"(chunk) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 8;" ::: "memory");
    }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
}

static void bulk_all(const char* only) {
  const int chunks[] = {4096, 16384};
  for (int dram = 0; dram < 2; dram++)
    for (int chunk : chunks)
      for (int bpsm : {1, 4}) {
        char label[64];
        snprintf(label, sizeof label, "bulk_load:%s:%d:%d", dram ? "dram" : "l2", chunk, bpsm);
        if (only && !strstr(label, only)) continue;
        int smem = chunk * BULK_STAGES;
        if (smem * bpsm > 96 * 1024) continue;
        CK(cudaFuncSetAttribute(k_bulk_load, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        const char* src = dram ? g_drambuf : g_l2buf;
        uint64_t bytes = dram ? DRAM_BYTES : L2_BYTES;
        int blocks = g_sms * bpsm;
        int iters = (int)std::min<uint64_t>(dram ? 4 * bytes / ((uint64_t)chunk * blocks) : 8192, 1 << 16);
        double ms = best_ms([&] { k_bulk_load<<<blocks, 32, smem>>>(src, bytes, chunk, iters, g_sink); }, 3);
        double moved = (double)blocks * iters * chunk;
        printf("{\"probe\":\"bulk_load\",\"target\":\"%s\",\"chunk\":%d,\"blocks_per_sm\":%d,\"iters\":%d,\"ms\":%.4f,"
               "\"bytes_per_s\":%.4e}\n",
               dram ? "dram" : "l2", chunk, bpsm, iters, ms, moved / (ms * 1e-3));
        fflush(stdout);
      }
  const char* opn[] = {"add.f32", "add.u32", "add.noftz.bf16"};
  const int elem_bytes[] = {4, 4, 2};
  for (int dram = 0; dram < 2; dram++)
    for (int opk = 0; opk < 3; opk++)
      for (int chunk : chunks) {
        char label[64];
        snprintf(label, sizeof label, "bulk_red:%s:%s:%d", opn[opk], dram ? "dram" : "l2", chunk);
        if (only && !strstr(label, only)) continue;
        char* dst = dram ? g_drambuf : g_l2buf;
        uint64_t bytes = dram ? DRAM_BYTES : L2_BYTES;
        int blocks = g_sms * 4;
        int iters = dram ? (int)std::min<uint64_t>(2 * bytes / ((uint64_t)chunk * blocks), 4096) : 1024;
        auto launch = [&] {
          if (opk == 0) k_bulk_red<0><<<blocks, 128, chunk>>>(dst, bytes, chunk, iters);
          else if (opk == 1) k_bulk_red<1><<<blocks, 128, chunk>>>(dst, bytes, chunk, iters);
          else k_bulk_red<2><<<blocks, 128, chunk>>>(dst, bytes, chunk, iters);
        };
        double ms = best_ms(launch, 3);
        double moved = (double)blocks * iters * chunk;
        printf("{\"probe\":\"bulk_red\",\"op\":\"%s\",\"target\":\"%s\",\"chunk\":%d,\"iters\":%d,\"ms\":%.4f,"
               "\"payload_bytes_per_s\":%.4e,\"elem_adds_per_s\":%.4e}\n",
               opn[opk], dram ? "dram" : "l2", chunk, iters, ms, moved / (ms * 1e-3), moved / elem_bytes[opk] / (ms * 1e-3));
        fflush(stdout);
      }
}

// ---- warp-specialised co-runs -------------------------------------------------------------------------------------
enum { ROLE_FFMA = 0, ROLE_QMMA = 1, ROLE_RED = 2, ROLE_LDG = 3, ROLE_NONE = 4 };
static const char* ROLE_NAME[] = {"ffma", "qmma_e4m3", "red_v4_f32", "ldg_l2", "none"};

struct WarpClk { long long t0, t1; };

template <int ROLE>
__device__ __forceinline__ void role_loop(int iters, char* buf, uint64_t mask16, uint32_t& acc) {
  const uint32_t lane = threadIdx.x & 31;
  if (ROLE == ROLE_FFMA) {
    float x[8];
#pragma unroll
    for (int k = 0; k < 8; k++) x[k] = (float)(threadIdx.x + k) * 1e-3f;
    const float a = 0.999f, b = 1e-4f;
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
#pragma unroll
      for (int u = 0; u < 8; u++)
#pragma unroll
        for (int k = 0; k < 8; k++) x[k] = fmaf(x[k], a, b);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) acc ^= __float_as_uint(x[k]);
  } else if (ROLE == ROLE_QMMA) {
    float d[8][4] = {};
    uint32_t a[4] = {0x38383838u ^ lane, 0x30303030u, 0x38303830u, 0x30383038u}, b[2] = {0x38383838u, 0x30303030u ^ lane};
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
#pragma unroll
      for (int c = 0; c < 8; c++)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
#pragma unroll
    for (int c = 0; c < 8; c++) acc ^= __float_as_uint(d[c][0] + d[c][1] + d[c][2] + d[c][3]);
  } else if (ROLE == ROLE_RED) {
    const uint64_t gw = ((uint64_t)blockIdx.x * 64 + (threadIdx.x >> 5));
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
      uint64_t slot = (((uint64_t)i * gridDim.x * 64 + gw) * 32 + lane) & mask16;
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(buf + slot * 16), "f"(1.f), "f"(1.f), "f"(1.f), "f"(1.f)
                   : "memory");
    }
  } else if (ROLE == ROLE_LDG) {
    const uint64_t gw = ((uint64_t)blockIdx.x * 64 + (threadIdx.x >> 5));
    uint4 s = make_uint4(0, 0, 0, 0);
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
#pragma unroll
      for (int u = 0; u < 4; u++) {
        uint64_t slot = ((((uint64_t)i * 4 + u) * gridDim.x * 64 + gw) * 32 + lane) & mask16;
        uint4 v = __ldcg((const uint4*)(buf + slot * 16));
        s.x ^= v.x; s.y ^= v.y; s.z ^= v.z; s.w ^= v.w;
      }
    }
    acc ^= s.x ^ s.y ^ s.z ^ s.w;
  }
}

template <int RA, int RB>
__global__ void __launch_bounds__(1024) k_mix(int wa, int ia, int ib, char* redbuf, char* ldgbuf, uint64_t mask16,
                                               WarpClk* rec, uint32_t* sink) {
  extern __shared__ uint32_t pad[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t acc = 0;
  __syncthreads();
  long long t0 = clock64();
  if (warp < wa) role_loop<RA>(ia, RA == ROLE_LDG ? ldgbuf : redbuf, mask16, acc);
  else role_loop<RB>(ib, RB == ROLE_LDG ? ldgbuf : redbuf, mask16, acc);
  long long t1 = clock64();
  if (lane == 0) rec[blockIdx.x * 32 + warp] = {t0, t1};
  if (acc == 0x12345678u) sink[0] = acc;
}

// Per-role work units a SM per SM clock: for each block, the role's span is its warps' latest end minus earliest start.
static void mix_report(const char* tag, int ra, int rb, int wa, int wb, int ia, int ib, double ms) {
  std::vector<WarpClk> h((size_t)g_sms * 32);
  CK(cudaMemcpy(h.data(), g_rec, h.size() * sizeof(WarpClk), cudaMemcpyDeviceToHost));
  std::vector<double> sa, sb;
  for (int b = 0; b < g_sms; b++) {
    long long a0 = LLONG_MAX, a1 = 0, b0 = LLONG_MAX, b1 = 0;
    for (int w = 0; w < wa + wb; w++) {
      WarpClk c = h[b * 32 + w];
      if (w < wa) { a0 = std::min(a0, c.t0); a1 = std::max(a1, c.t1); }
      else { b0 = std::min(b0, c.t0); b1 = std::max(b1, c.t1); }
    }
    if (wa) sa.push_back((double)(a1 - a0));
    if (wb) sb.push_back((double)(b1 - b0));
  }
  auto med = [](std::vector<double>& v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  double ca = med(sa), cb = med(sb);
  // units: ffma = FFMA lane ops, qmma = MACs, red = FP32 adds, ldg = bytes; each a SM
  auto units = [](int role, int warps, int iters) -> double {
    switch (role) {
      case ROLE_FFMA: return (double)warps * 32 * iters * 64;
      case ROLE_QMMA: return (double)warps * iters * 8 * 4096;
      case ROLE_RED: return (double)warps * 32 * iters * 4;
      case ROLE_LDG: return (double)warps * 32 * iters * 4 * 16;
    }
    return 0;
  };
  printf("{\"probe\":\"mix\",\"case\":\"%s\",\"role_a\":\"%s\",\"role_b\":\"%s\",\"warps_a\":%d,\"warps_b\":%d,\"ms\":%.4f,"
         "\"cyc_a\":%.0f,\"cyc_b\":%.0f,\"a_units_per_sm_per_clk\":%.4f,\"b_units_per_sm_per_clk\":%.4f}\n",
         tag, ROLE_NAME[ra], ROLE_NAME[rb], wa, wb, ms, ca, cb, wa ? units(ra, wa, ia) / ca : 0.0,
         wb ? units(rb, wb, ib) / cb : 0.0);
  fflush(stdout);
}

template <int RA, int RB>
static void mix_pair(int wa, int wb, int ia, int ib, const char* only) {
  char label[64];
  snprintf(label, sizeof label, "mix:%s+%s", ROLE_NAME[RA], ROLE_NAME[RB]);
  if (only && !strstr(label, only)) return;
  const uint64_t mask16 = L2_BYTES / 16 - 1;
  auto run = [&](const char* tag, int a, int b, auto kern) {
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
    double ms = best_ms([&] { kern<<<g_sms, (a + b) * 32, ONE_BLOCK_SMEM>>>(a, ia, ib, g_l2buf, g_drambuf, mask16,
                                                                            (WarpClk*)g_rec, g_sink); }, 3);
    mix_report(tag, RA, RB, a, b, ia, ib, ms);
  };
  run("a_alone", wa, 0, k_mix<RA, RB>);
  run("b_alone", 0, wb, k_mix<RA, RB>);
  run("both", wa, wb, k_mix<RA, RB>);
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  g_sms = num_sms();
  CK(cudaMalloc(&g_l2buf, L2_BYTES));
  CK(cudaMalloc(&g_drambuf, DRAM_BYTES));
  CK(cudaMemset(g_l2buf, 0, L2_BYTES));
  CK(cudaMemset(g_drambuf, 0, DRAM_BYTES));
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 64 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * 32 * sizeof(WarpClk)));
  red_all<red_f32>(only);
  red_all<red_u32>(only);
  red_all<red_u64>(only);
  red_all<red_f64>(only);
  red_all<red_v4_f32>(only);
  red_all<red_bf16x2>(only);
  red_all<red_v4_bf16x2>(only);
  red_all<atom_u32>(only);
  red_all<atom_f32>(only);
  bulk_all(only);
  // FFMA: 8 warps x 64 FFMA an iteration; QMMA: 8 warps x 8; RED: 8 warps, one v4 a lane an iteration; LDG: 4 x 16 B.
  mix_pair<ROLE_FFMA, ROLE_RED>(8, 8, 40000, 6000, only);
  mix_pair<ROLE_QMMA, ROLE_RED>(8, 8, 20000, 6000, only);
  mix_pair<ROLE_FFMA, ROLE_LDG>(8, 8, 40000, 3000, only);
  mix_pair<ROLE_QMMA, ROLE_LDG>(8, 8, 20000, 3000, only);
  mix_pair<ROLE_LDG, ROLE_RED>(8, 8, 3000, 6000, only);
  mix_pair<ROLE_QMMA, ROLE_FFMA>(8, 8, 20000, 40000, only);
  printf("{\"probe\":\"memside\",\"done\":true}\n");
  return 0;
}
