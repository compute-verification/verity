// HC-PREFILL's filler as a shared library (cmp-13, approach pouw/filler-prefill): dcap.cu's DRAM-free filler (dfill_k, its
// DFill, dpump and dfill_ref, unchanged) behind a C API, so that a Python process running vLLM's own prefill kernels (FA2,
// norms, rotary, activation, quantization, KV writes; hcpre_phase.py) can run the scorer's co placement beside them: the
// filler's chunks queued by a host thread on a lowest-priority stream of its own, serving on a higher-priority stream.
// Both runtimes bind the device's primary context, so the filler's stream and torch's share the GPU.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -shared -Xcompiler -fPIC -o libhcpre_fill.so
//        hcpre_fill.cu -lcublasLt -lnvidia-ml -ldl
#define main dcap_main
#include "../decode_capacity/dcap.cu"
#undef main

static DFill g_f;
static DPump* g_p = nullptr;
static cudaStream_t g_fs = nullptr;

extern "C" {
// the filler: `grid` blocks of W warps, NC chains, S steps, ~2^chunk_log2 MMAs an SM a chunk, map (DFILL_MAP's), on a stream
// at the device's lowest priority; returns the priority range's [lowest, highest]
int hf_init(int grid, int W, int NC, int S, int chunk_log2, int map, int* prio) {
  CK(cudaSetDevice(0));
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  CK(cudaStreamCreateWithPriority(&g_fs, cudaStreamNonBlocking, lo));
  prio[0] = lo;
  prio[1] = hi;
  char buf[16];
  snprintf(buf, sizeof buf, "%d", chunk_log2);
  setenv("DFILL_CHUNK_LOG2", buf, 1);
  snprintf(buf, sizeof buf, "%d", map);
  setenv("DFILL_MAP", buf, 1);
  dfill_prep();
  g_f = dfill_make(grid, W, NC, S, 0, 0);
  return g_f.tiles;
}
double hf_chunk_macs() { return g_f.macs(); }
long long hf_per() { return g_f.per(); }
double hf_now_ms() { return now_ms(); }
// a round: its coins and the first tile of its public range; poisons the receipts (0xA5), then keeps 3 chunks queued
void hf_start(uint64_t c0, uint64_t c1, unsigned long long base) {
  g_f.c0 = c0;
  g_f.c1 = c1;
  g_f.base = base;
  CK(cudaMemsetAsync(g_f.rec, 0xA5, g_f.per() * sizeof(unsigned long long), g_fs));
  CK(cudaStreamSynchronize(g_fs));
  g_p = new DPump;
  dpump_start(*g_p, g_f, g_fs, 3);
}
long long hf_done() { return g_p ? g_p->done.load() : 0; }
// stops the pump; returns the chunks it launched (its tiles are [base, base + launched·per))
long long hf_stop() {
  dpump_stop(*g_p);
  return g_p->launched.load();
}
// the chunks seen done inside (t0, t1] and the span from the first to the last (dscore's count)
long long hf_count(double t0, double t1, double* span) {
  double first = 0, last = 0;
  long long n = 0;
  for (double x : g_p->t)
    if (x > t0 && x <= t1) {
      if (!n++) first = x;
      last = x;
    }
  *span = last - first;
  return n;
}
// n receipts of the stopped pump's last chunk against dfill_ref; their tile ids in `tiles`; returns the mismatches
int hf_check(int n, unsigned long long* tiles) {
  const long long last = g_p->launched.load() - 1;
  std::vector<unsigned long long> h(g_f.per());
  CK(cudaMemcpy(h.data(), g_f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; i++) {
    const long long idx = (long long)i * g_f.per() / n + (i * 7919) % std::max(1LL, g_f.per() / n);
    tiles[i] = g_f.base + (unsigned long long)last * g_f.per() + idx;
    bad += h[idx] != dfill_ref(g_f.c0, g_f.c1, tiles[i], g_f.NC, g_f.S, g_f.map);
  }
  delete g_p;
  g_p = nullptr;
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
