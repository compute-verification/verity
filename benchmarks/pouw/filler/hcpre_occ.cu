// HC-PREFILL's occupancy probe (cmp-13, approach pouw/filler-prefill): where the co filler's blocks land beside the
// scorer's prefill linears, and whether one filler block an SM fixes the chunk sizes that break the slowdown gate.
// dscore's co pump launches a chunk as one block an SM (grid = SMs) with no dynamic shared memory; at 88 registers a
// thread (W 4, NC 4) three filler blocks fit beside a prefill GEMM CTA (224 registers x 128 threads, 96 KiB of shared
// memory) and five on an SM the GEMM has left between launches, where a second filler warp on a sub-partition costs the
// GEMM's one warp ~1.4x (W 5 and W 6 screens). Per block this records its SM, its start and end (globaltimer) and the
// warp ranges it ran, for the map-1 tiles dfill_k runs (dtiles<NC, 1>, the same receipts):
//   MODE 0: dfill_k's static assignment (warp w of block b runs warp range b * W + w);
//   MODE 1: a block that finds another of the chunk's blocks on its SM exits at once, and the running blocks' warps take
//           warp ranges from a counter until the chunk's are done (an exit frees the SM, and the dispatcher puts the
//           next block there: r20261016-081635-dcc5 had 69 of 188 exit a chunk);
//   MODE 2: a block that finds another on its SM keeps its place asleep until that one is done, and the running warps
//           take SUB tiles of a warp range at a time from a counter (so no SM runs two filler blocks at once, and the
//           chunk's work ends evenly over the SMs that run it).
// The linears run on the highest-priority stream (dscore.cu's prefill workload, ABBA alone/arm over ROUNDS rounds), the
// probe's chunks on the lowest, three queued. Each pump's last chunk has 16 receipts checked against dfill_ref.
//
//   hcpre_occ ROUNDS K W NC S MODE [SUB]   one JSON line (map 1; DFILL_CHUNK_LOG2 sets the chunk as dfill_make does)
#include <sys/random.h>
#define main dcap_main
#include "../decode_capacity/dcap.cu"
#undef main

constexpr size_t MAX_BL = 786432;
constexpr int NLOG = 4096;  // chunks logged (a ring)

// dscore.cu's setup, as it is there (dscore.cu defines main, so it can't be included)
static int setup(Dev& d, bool weights, size_t kv_want) {
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&d.st));
  CK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, 0));
  if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); return 2; }
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
  if (nvmlDeviceGetHandleByPciBusId_v2(bus, &d.nv) != NVML_SUCCESS) { fprintf(stderr, "NVML handle failed\n"); return 2; }
  char uuid[96] = {0}, name[96] = {0};
  nvmlDeviceGetUUID(d.nv, uuid, sizeof uuid);
  nvmlDeviceGetName(d.nv, name, sizeof name);
  const char* want = getenv("GPU_LEASE_UUID");
  if (want && *want && strcmp(want, uuid) != 0) {
    fprintf(stderr, "device %s is not the leased %s\n", uuid, want);
    return 3;
  }
  fprintf(stderr, "device %s %s, %d SMs\n", name, uuid, d.sms);
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  CK(cudaMalloc(&g_ref, (size_t)256 * VOCAB * sizeof(int)));
  CK(cudaMalloc(&g_cnt, sizeof *g_cnt));
  if (weights) {
    for (int l = 0; l < LAYERS; l++)
      for (int j = 0; j < 4; j++) {
        const size_t n = (size_t)LN[j] * LK[j];
        CK(cudaMalloc(&d.W[l][j], n));
        fill_s8<<<1024, 256, 0, d.st>>>(d.W[l][j], n, 0x1000 + l * 4 + j);
      }
    CK(cudaMalloc(&d.Wlm, (size_t)VOCAB * HID));
    fill_s8<<<1024, 256, 0, d.st>>>(d.Wlm, (size_t)VOCAB * HID, 0x77);
  }
  CK(cudaMalloc(&d.X, (size_t)8192 * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(d.X, (size_t)8192 * FF, 0x55);
  CK(cudaMalloc(&d.D, (size_t)1 << 30));
  size_t fr, tot;
  CK(cudaMemGetInfo(&fr, &tot));
  d.kv_bytes = std::min({(size_t)48 << 30, fr - ((size_t)3 << 30), kv_want});
  CK(cudaMalloc(&d.kv, d.kv_bytes));
  fill_s8<<<4096, 256, 0, d.st>>>(d.kv, d.kv_bytes, 0x99);
  CK(cudaMalloc(&d.q, (size_t)256 * HQ * HD));
  fill_s8<<<64, 256, 0, d.st>>>(d.q, (size_t)256 * HQ * HD, 0x31);
  CK(cudaMalloc(&d.P, MAX_BL * HQ));
  fill_s8<<<1024, 256, 0, d.st>>>(d.P, MAX_BL * HQ, 0x32);
  CK(cudaMalloc(&d.S, MAX_BL * HQ * sizeof(int)));
  CK(cudaMalloc(&d.O, (size_t)256 * HQ * HD * sizeof(int)));
  CK(cudaMalloc(&d.sink, 64));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  return 0;
}

__device__ int occ_sm[1024];
__device__ unsigned long long occ_next;

// log entry a block: smid, start ns, end ns, (ran ? 1 : 0) | warp ranges run << 1
template <int NC, int MODE>
__global__ void __launch_bounds__(512) occ_k(uint64_t c0, uint64_t c1, unsigned long long tile0, int tiles, int steps,
                                             unsigned long long* __restrict__ rec, unsigned long long* lg, int sub) {
  unsigned smid;
  unsigned long long ts, te;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
  const int nw = blockDim.x >> 5, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned long long* e = lg + (unsigned long long)blockIdx.x * 4;
  unsigned long long runs = 0;
  if (MODE == 0) {
    dtiles<NC, 1>(c0, c1, tile0, tiles, steps, rec);
    runs = 1 | ((unsigned long long)nw << 1);
  } else if (MODE == 2) {
    if (threadIdx.x == 0)
      while (atomicCAS(&occ_sm[smid], 0, 1) != 0) __nanosleep(4000);
    __syncthreads();
    const int m = sub < tiles ? sub : tiles, per_w = tiles / m;
    const unsigned long long units = (unsigned long long)gridDim.x * nw * per_w;
    const unsigned long long wid = (unsigned long long)blockIdx.x * nw + w;
    int done = 0;
    for (;;) {
      unsigned long long u = 0;
      if (lane == 0) u = atomicAdd(&occ_next, 1ull);
      u = __shfl_sync(0xffffffffu, u, 0);
      if (u >= units) break;
      const unsigned long long first = (u / per_w) * tiles + (u % per_w) * m;
      const unsigned long long off = first - wid * m;
      dtiles<NC, 1>(c0, c1, tile0 + off, m, steps, rec + off);
      done++;
    }
    if (lane == 0 && done) atomicAdd(&e[3], (unsigned long long)done << 1);
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicOr(&e[3], (e[3] >> 1) ? 1ull : 0ull);
      atomicExch(&occ_sm[smid], 0);
    }
    runs = 0;
  } else {
    if (threadIdx.x == 0) e[3] = atomicAdd(&occ_sm[smid], 1) == 0 ? 1 : 0;
    __syncthreads();
    const bool run = *(volatile unsigned long long*)&e[3] != 0;
    int done = 0;
    if (run) {
      const unsigned long long units = (unsigned long long)gridDim.x * nw, wid = (unsigned long long)blockIdx.x * nw + w;
      for (;;) {
        unsigned long long u = 0;
        if (lane == 0) u = atomicAdd(&occ_next, 1ull);
        u = __shfl_sync(0xffffffffu, u, 0);
        if (u >= units) break;
        const unsigned long long off = (u - wid) * (unsigned long long)tiles;
        dtiles<NC, 1>(c0, c1, tile0 + off, tiles, steps, rec + off);
        done++;
      }
    }
    if (lane == 0 && done) atomicAdd(&e[3], (unsigned long long)done << 1);
    __syncthreads();
    if (threadIdx.x == 0) atomicSub(&occ_sm[smid], 1);
    runs = 0;
  }
  __syncthreads();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(te));
  if (threadIdx.x == 0) {
    e[0] = smid;
    e[1] = ts;
    e[2] = te;
    if (MODE == 0) e[3] = runs;
  }
}

struct Occ {
  DFill f;
  int mode, sub;
  unsigned long long* log;  // NLOG chunks x grid x 4
};
static void occ_launch(const Occ& o, long long k, cudaStream_t s) {
  const unsigned long long t0 = o.f.base + (unsigned long long)k * o.f.per();
  unsigned long long* lg = o.log + (unsigned long long)(k % NLOG) * o.f.grid * 4;
  const dim3 gr(o.f.grid), bl(32 * o.f.W);
  if (o.mode != 0) {
    CK(cudaMemsetAsync(lg, 0, (size_t)o.f.grid * 4 * sizeof(unsigned long long), s));
    void* p = nullptr;
    CK(cudaGetSymbolAddress(&p, occ_next));
    CK(cudaMemsetAsync(p, 0, sizeof(unsigned long long), s));
  }
#define OCC_CASE(NC_)                                                                                       \
  case NC_:                                                                                                 \
    if (o.mode == 0) occ_k<NC_, 0><<<gr, bl, 0, s>>>(o.f.c0, o.f.c1, t0, o.f.tiles, o.f.S, o.f.rec, lg, o.sub); \
    else if (o.mode == 1) occ_k<NC_, 1><<<gr, bl, 0, s>>>(o.f.c0, o.f.c1, t0, o.f.tiles, o.f.S, o.f.rec, lg, o.sub); \
    else occ_k<NC_, 2><<<gr, bl, 0, s>>>(o.f.c0, o.f.c1, t0, o.f.tiles, o.f.S, o.f.rec, lg, o.sub);         \
    break;
  switch (o.f.NC) {
    OCC_CASE(1) OCC_CASE(2) OCC_CASE(4)
    default: fprintf(stderr, "NC 1, 2 or 4\n"); exit(1);
  }
#undef OCC_CASE
}

struct Pumped {
  std::vector<double> t;  // host time each chunk was seen done
  long long first = 0, n = 0, bad = 0, checked = 0;
};
// chunks k0, k0 + 1, ... queued three deep on s until `on` drops; every chunk's receipts sampled against dfill_ref
static void occ_pump(const Occ& o, cudaStream_t s, long long k0, std::atomic<bool>& on, std::atomic<long long>& done,
                     Pumped& P) {
  cudaEvent_t ev[3];
  for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync));
  P.first = k0;
  long long k = 0;
  for (; on.load(); k++) {
    if (k >= 3) {
      CK(cudaEventSynchronize(ev[k % 3]));
      P.t.push_back(now_ms());
      done++;
    }
    occ_launch(o, k0 + k, s);
    CK(cudaEventRecord(ev[k % 3], s));
  }
  CK(cudaStreamSynchronize(s));
  P.n = k;
  P.checked += 16;
  P.bad += dfill_check(o.f, k0 + k - 1, 16);
  for (auto& e : ev) CK(cudaEventDestroy(e));
}

struct Stat {
  long long chunks = 0, blocks = 0, ran = 0, multi_sm_chunks = 0, sms_multi = 0, uncovered = 0, max_per_sm = 0;
  double dur_ms = 0, start_spread_us = 0, end_spread_us = 0;
};
static Stat occ_stat(const std::vector<unsigned long long>& L, int grid, long long k0, long long n, int sms) {
  Stat st;
  std::vector<int> per(sms > 1024 ? sms : 1024);
  for (long long k = k0; k < k0 + n; k++) {
    const unsigned long long* c = L.data() + (size_t)(k % NLOG) * grid * 4;
    std::fill(per.begin(), per.end(), 0);
    unsigned long long a0 = ~0ull, a1 = 0, b0 = ~0ull, b1 = 0;
    for (int b = 0; b < grid; b++) {
      const unsigned long long* e = c + b * 4;
      if (e[1] == 0) continue;
      st.blocks++;
      a0 = std::min(a0, e[1]);
      a1 = std::max(a1, e[1]);
      b0 = std::min(b0, e[2]);
      b1 = std::max(b1, e[2]);
      per[e[0] % per.size()]++;
      if (e[3] & 1) st.ran++;
    }
    int multi = 0, empty = 0, mx = 0;
    for (int s = 0; s < sms; s++) {
      multi += per[s] > 1;
      empty += per[s] == 0;
      mx = std::max(mx, per[s]);
    }
    st.chunks++;
    st.multi_sm_chunks += multi > 0;
    st.sms_multi += multi;
    st.uncovered += empty;
    st.max_per_sm = std::max<long long>(st.max_per_sm, mx);
    st.dur_ms += (b1 - a0) * 1e-6;
    st.start_spread_us += (a1 - a0) * 1e-3;
    st.end_spread_us += (b1 - b0) * 1e-3;
  }
  if (st.chunks) {
    st.dur_ms /= st.chunks;
    st.start_spread_us /= st.chunks;
    st.end_spread_us /= st.chunks;
  }
  return st;
}
static void print_stat(const char* key, const Stat& s, int sms) {
  printf("\"%s\": {\"chunks\": %lld, \"blocks\": %lld, \"ran\": %lld, \"chunks_with_a_doubled_sm\": %lld, "
         "\"doubled_sms_per_chunk\": %.3f, \"uncovered_sms_per_chunk\": %.3f, \"max_blocks_an_sm\": %lld, "
         "\"chunk_ms\": %.4f, \"start_spread_us\": %.2f, \"end_spread_us\": %.2f}",
         key, s.chunks, s.blocks, s.ran, s.multi_sm_chunks, s.chunks ? (double)s.sms_multi / s.chunks : 0,
         s.chunks ? (double)s.uncovered / s.chunks : 0, s.max_per_sm, s.dur_ms, s.start_spread_us, s.end_spread_us);
  (void)sms;
}

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "hcpre_occ ROUNDS K W NC S MODE [SUB]\n"); return 1; }
  const int rounds = atoi(argv[1]), K = atoi(argv[2]), W = atoi(argv[3]), NC = atoi(argv[4]), S = atoi(argv[5]),
            mode = atoi(argv[6]), sub = argc > 7 ? atoi(argv[7]) : 4;
  setenv("DFILL_MAP", "1", 1);
  Dev d;
  if (int rc = setup(d, true, (size_t)1 << 20)) return rc;
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t full, fs;
  CK(cudaStreamCreateWithPriority(&full, cudaStreamDefault, hi));
  CK(cudaStreamCreateWithPriority(&fs, cudaStreamNonBlocking, lo));
  uint64_t c[2];
  if (getrandom(c, sizeof c, 0) != (ssize_t)sizeof c) { fprintf(stderr, "getrandom\n"); return 2; }
  Occ o{dfill_make(d.sms, W, NC, S, c[0], c[1]), mode, sub, nullptr};
  if (mode == 2 && (sub < 1 || o.f.tiles % std::min(sub, o.f.tiles))) { fprintf(stderr, "SUB must divide tiles\n"); return 1; }
  o.f.base = 1ull << 40;
  const size_t lgn = (size_t)NLOG * o.f.grid * 4;
  CK(cudaMalloc(&o.log, lgn * sizeof(unsigned long long)));
  CK(cudaMemset(o.log, 0, lgn * sizeof(unsigned long long)));
  std::vector<unsigned long long> L(lgn);
  long long kn = 0;
  // alone: the probe's chunks with nothing beside them
  Pumped pa;
  {
    std::atomic<bool> on{true};
    std::atomic<long long> done{0};
    std::thread th([&] { occ_pump(o, fs, kn, on, done, pa); });
    while (done.load() < 12) {
    }
    on = false;
    th.join();
    kn += pa.n;
  }
  const double alone_rate = (pa.t.size() - 2) * o.f.macs() / ((pa.t.back() - pa.t[1]) * 1e-3);
  CK(cudaMemcpy(L.data(), o.log, lgn * sizeof L[0], cudaMemcpyDeviceToHost));
  const Stat sa = occ_stat(L, o.f.grid, pa.first + 1, pa.n - 2, d.sms);
  const std::vector<Lt> p = lin_plans(d, 8192);
  bool wrapped = false;
  d.st = full;
  auto work = [&] { enqueue_step(d, p, 0, 8192, 0, &wrapped); };
  std::vector<std::vector<Rep>> al(rounds), ar(rounds);
  std::vector<double> rate(rounds);
  std::vector<Stat> sb(rounds);
  long long bad = pa.bad, checked = pa.checked;
  for (int r = 0; r < rounds; r++) {
    auto arm = [&] {
      std::atomic<bool> on{true};
      std::atomic<long long> done{0};
      Pumped P;
      std::thread th([&] { occ_pump(o, fs, kn, on, done, P); });
      while (done.load() < 2) {
      }
      double t0 = 0, t1 = 0;
      ar[r] = time_work(d, work, K, &t0, &t1);
      on = false;
      th.join();
      kn += P.n;
      bad += P.bad;
      checked += P.checked;
      long long n = 0, i0 = -1, i1 = -1;
      double first = 0, last = 0;
      for (size_t i = 0; i < P.t.size(); i++)
        if (P.t[i] > t0 && P.t[i] <= t1) {
          if (!n++) { first = P.t[i]; i0 = (long long)i; }
          last = P.t[i];
          i1 = (long long)i;
        }
      rate[r] = n > 1 ? (n - 1) * o.f.macs() / ((last - first) * 1e-3) : 0;
      CK(cudaMemcpy(L.data(), o.log, lgn * sizeof L[0], cudaMemcpyDeviceToHost));
      // P.t[i] is chunk P.first + i done; the chunks inside the timed window
      if (i0 >= 0 && i1 > i0 && P.n < NLOG) sb[r] = occ_stat(L, o.f.grid, P.first + i0 + 1, i1 - i0, d.sms);
    };
    if (r % 2 == 0) {
      al[r] = time_work(d, work, K);
      arm();
    } else {
      arm();
      al[r] = time_work(d, work, K);
    }
  }
  printf("{\"mode\": \"occ\", \"occ_mode\": %d, \"sub\": %d, \"sms\": %d, \"W\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %d, \"grid\": %d, "
         "\"chunk_macs\": %.6e, \"receipts_checked\": %lld, \"receipts_bad\": %lld, \"alone_mac_per_s\": %.6e, ",
         mode, sub, d.sms, W, NC, S, o.f.tiles, o.f.grid, o.f.macs(), checked, bad, alone_rate);
  print_stat("alone_occ", sa, d.sms);
  printf(", \"rounds\": [");
  for (int r = 0; r < rounds; r++) {
    printf("%s{\"alone\": [", r ? ", " : "");
    for (size_t i = 0; i < al[r].size(); i++) printf("%s[%.4f, %.0f]", i ? ", " : "", al[r][i].ms, al[r][i].mhz);
    printf("], \"arm\": [");
    for (size_t i = 0; i < ar[r].size(); i++) printf("%s[%.4f, %.0f]", i ? ", " : "", ar[r][i].ms, ar[r][i].mhz);
    printf("], \"fill_mac_per_s\": %.6e, ", rate[r]);
    print_stat("occ", sb[r], d.sms);
    printf("}");
  }
  printf("]}\n");
  CK(cudaFree(o.log));
  return 0;
}
