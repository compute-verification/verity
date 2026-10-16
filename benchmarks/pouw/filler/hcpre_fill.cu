// HC-PREFILL's filler as a shared library (cmp-13, approach pouw/filler-prefill): dcap.cu's DRAM-free filler (dfill_k, its
// DFill, dpump and dfill_ref, unchanged) behind a C API, so that a Python process running vLLM's own prefill kernels (FA2,
// norms, rotary, activation, quantization, KV writes; hcpre_phase.py) can run the scorer's co placement beside them: the
// filler's chunks queued by a host thread on a lowest-priority stream of its own, serving on a higher-priority stream.
// Both runtimes bind the device's primary context, so the filler's stream and torch's share the GPU. HF_PUMPS=2 runs a
// second pump on a second lowest-priority stream, its tiles from the upper half of the round's range, so one stream's
// next chunk no longer waits for the slowest SM's last block.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -shared -Xcompiler -fPIC -o libhcpre_fill.so
//        hcpre_fill.cu -lcublasLt -lnvidia-ml -ldl
#define main dcap_main
#include "../decode_capacity/dcap.cu"
#undef main

#include <algorithm>

constexpr unsigned long long HF_HALF = 1ull << 31;
static int g_np = 1;
static DFill g_f[2];
static DPump* g_p[2] = {nullptr, nullptr};
static cudaStream_t g_fs[2] = {nullptr, nullptr};

extern "C" {
// the filler: `grid` blocks of W warps, NC chains, S steps, ~2^chunk_log2 MMAs an SM a chunk, map (DFILL_MAP's), on
// HF_PUMPS (1 or 2) streams at the device's lowest priority; returns the priority range's [lowest, highest] in prio
int hf_init(int grid, int W, int NC, int S, int chunk_log2, int map, int* prio) {
  CK(cudaSetDevice(0));
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  g_np = getenv("HF_PUMPS") ? std::min(2, std::max(1, atoi(getenv("HF_PUMPS")))) : 1;
  prio[0] = lo;
  prio[1] = hi;
  char buf[16];
  snprintf(buf, sizeof buf, "%d", chunk_log2);
  setenv("DFILL_CHUNK_LOG2", buf, 1);
  snprintf(buf, sizeof buf, "%d", map);
  setenv("DFILL_MAP", buf, 1);
  dfill_prep();
  for (int j = 0; j < g_np; j++) {
    CK(cudaStreamCreateWithPriority(&g_fs[j], cudaStreamNonBlocking, lo));
    g_f[j] = dfill_make(grid, W, NC, S, 0, 0);
  }
  return g_f[0].tiles;
}
int hf_pumps() { return g_np; }
double hf_chunk_macs() { return g_f[0].macs(); }
long long hf_per() { return g_f[0].per(); }
double hf_now_ms() { return now_ms(); }
// a round: its coins and the first tile of its public range; poisons the receipts (0xA5), then keeps 3 chunks queued on
// each pump (pump j's tiles from base + j·2^31)
void hf_start(uint64_t c0, uint64_t c1, unsigned long long base) {
  for (int j = 0; j < g_np; j++) {
    DFill& f = g_f[j];
    f.c0 = c0;
    f.c1 = c1;
    f.base = base + j * HF_HALF;
    CK(cudaMemsetAsync(f.rec, 0xA5, f.per() * sizeof(unsigned long long), g_fs[j]));
    CK(cudaStreamSynchronize(g_fs[j]));
    g_p[j] = new DPump;
    dpump_start(*g_p[j], f, g_fs[j], 3);
  }
}
// the fewest chunks any pump has seen done
long long hf_done() {
  long long n = -1;
  for (int j = 0; j < g_np; j++) {
    const long long d = g_p[j] ? g_p[j]->done.load() : 0;
    n = n < 0 ? d : std::min(n, d);
  }
  return n < 0 ? 0 : n;
}
// stops the pumps; returns pump 0's launched chunks (its tiles are [base, base + launched·per))
long long hf_stop() {
  for (int j = 0; j < g_np; j++) dpump_stop(*g_p[j]);
  return g_p[0]->launched.load();
}
long long hf_launched(int j) { return g_p[j] ? g_p[j]->launched.load() : 0; }
// the chunks (every pump's) seen done inside (t0, t1] and the span from the first to the last (dscore's count)
long long hf_count(double t0, double t1, double* span) {
  std::vector<double> t;
  for (int j = 0; j < g_np; j++)
    for (double x : g_p[j]->t)
      if (x > t0 && x <= t1) t.push_back(x);
  std::sort(t.begin(), t.end());
  *span = t.empty() ? 0 : t.back() - t.front();
  return (long long)t.size();
}
// n receipts of each stopped pump's last chunk against dfill_ref; their tile ids in `tiles` (n a pump); the mismatches
int hf_check(int n, unsigned long long* tiles) {
  int bad = 0;
  for (int j = 0; j < g_np; j++) {
    const DFill& f = g_f[j];
    const long long last = g_p[j]->launched.load() - 1;
    std::vector<unsigned long long> h(f.per());
    CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
      const long long idx = (long long)i * f.per() / n + (i * 7919) % std::max(1LL, f.per() / n);
      tiles[j * n + i] = f.base + (unsigned long long)last * f.per() + idx;
      bad += h[idx] != dfill_ref(f.c0, f.c1, tiles[j * n + i], f.NC, f.S, f.map);
    }
    delete g_p[j];
    g_p[j] = nullptr;
  }
  return bad;
}
uint64_t hf_seed(uint64_t c0, uint64_t c1, unsigned long long tile) { return dseed(c0, c1, tile); }
// NVML on the device CUDA calls 0 (by PCI bus id, as dscore's setup); the SM clock (MHz) and board power (W), -1 unread
static nvmlDevice_t g_nv;
int hf_nvml_init() {
  if (nvmlInit() != NVML_SUCCESS) return 2;
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
  return nvmlDeviceGetHandleByPciBusId_v2(bus, &g_nv) == NVML_SUCCESS ? 0 : 2;
}
double hf_mhz() {
  unsigned c = 0;
  return nvmlDeviceGetClockInfo(g_nv, NVML_CLOCK_SM, &c) == NVML_SUCCESS ? (double)c : -1;
}
double hf_watt() {
  unsigned p = 0;
  return nvmlDeviceGetPowerUsage(g_nv, &p) == NVML_SUCCESS ? p / 1000.0 : -1;
}
}
