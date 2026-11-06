// Clock and power under load (eps_clock): does a program with a lighter
// tensor load run at a higher SM clock than the saturated honest kernel under the board's power cap? Every kernel runs
// one 512-thread block an SM; `--sustain NAME SECONDS` launches it back to back for SECONDS and prints a JSON line a
// launch: every block's clock64 cycles and %globaltimer ns over the timed loop (their ratio is the SM clock the block
// actually ran at, whatever NVML reports), the work done, and NVML's SM clock, power and clock-event reasons.
// `power.sh` runs it under one gpu-lease, at the node's lock and then at the default clocks. The table also gates every
// kernel's SASS through probe.py like any probe's (`--list`).
//   p_honest    8 IMMA + the duo fold of their 32 words a lane (honest.cuh), drawn operands
//   p_half      4 IMMA + the same fold (half the MMAs)
//   p_mma       8 IMMA, no fold (tensor only)
//   p_mma0      8 IMMA on all-zero operands, no fold (the lightest tensor load)
//   p_fold      the fold alone (scalar only, integer)
//   p_ffma      FFMA on 8 chains a thread (scalar only, FP32)
//   p_mixed     4 IMMA + the fold + 64 FFMA
//   p_honest16, p_mma16, p_mma16z: p_honest, p_mma and p_mma0 with m16n8k16 (the honest MMA at L = 16 on sm_89)
//   p_light     the fold alone on one warp an SM (the lightest load: the clock the board reaches unloaded)
#include <algorithm>
#include <chrono>
#include <dlfcn.h>
#include <string>
#include <thread>

#include "../probe.cuh"
#include "honest.cuh"

constexpr int TPB = 512;

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

struct Ffma {
  float f[8], c1, c2;
  __device__ explicit Ffma(int mode) {
    c1 = 0.99999994f + (float)mode;
    c2 = 1e-7f * (float)(mode + 1);
#pragma unroll
    for (int i = 0; i < 8; i++) f[i] = (float)(threadIdx.x + i);
  }
  template <int N>
  __device__ __forceinline__ void run() {
#pragma unroll
    for (int q = 0; q < N; q++) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[q & 7]) : "f"(c1), "f"(c2));
  }
  __device__ uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x ^= __float_as_uint(f[i]);
    return x;
  }
};

// The timed loop between OP_BEGIN and OP_END, its globaltimer span in rec.stop (ns) for the launch's clock.
template <int NM, bool FOLD, bool ZERO, int NF, bool K16 = false>
__device__ __forceinline__ void body(const OpArgs& a) {
  Honest h(a.mode, ZERO);
  Ffma f(a.mode);
  const unsigned long long t0 = gtime();
  OP_BEGIN
  OP_LOOP(it) {
    h.step<NM, FOLD, K16>();
    f.run<NF>();
  }
  OP_END
  const unsigned long long t1 = gtime();
  if (threadIdx.x == 0) {
    a.rec[blockIdx.x].stop = (long long)(t1 - t0);
    reinterpret_cast<unsigned long long*>(a.g)[2 * blockIdx.x] = t0;
    reinterpret_cast<unsigned long long*>(a.g)[2 * blockIdx.x + 1] = t1;
  }
  uint32_t s = h.fold() ^ f.fold();
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)h.d[c][0];
  OP_SINK(s)
}

OP_KERNEL(p_honest, TPB) { body<8, true, false, 0>(a); }
OP_KERNEL(p_half, TPB) { body<4, true, false, 0>(a); }
OP_KERNEL(p_mma, TPB) { body<8, false, false, 0>(a); }
OP_KERNEL(p_mma0, TPB) { body<8, false, true, 0>(a); }
OP_KERNEL(p_fold, TPB) { body<0, true, false, 0>(a); }
OP_KERNEL(p_ffma, TPB) { body<0, false, false, 64>(a); }
OP_KERNEL(p_mixed, TPB) { body<4, true, false, 64>(a); }
OP_KERNEL(p_honest16, TPB) { body<8, true, false, 0, true>(a); }
OP_KERNEL(p_mma16, TPB) { body<8, false, false, 0, true>(a); }
OP_KERNEL(p_mma16z, TPB) { body<8, false, true, 0, true>(a); }
OP_KERNEL(p_light, 32) { body<0, true, false, 0>(a); }

#define IMMA "IMMA.16832.S8.S8"
static const OpProbe PROBES[] = {
    {.name = "p_honest", .kernel = (const void*)p_honest, .tpb = TPB, .expect = IMMA "=8", .iters = 2048},
    {.name = "p_half", .kernel = (const void*)p_half, .tpb = TPB, .expect = IMMA "=4", .iters = 2048},
    {.name = "p_mma", .kernel = (const void*)p_mma, .tpb = TPB, .expect = IMMA "=8", .iters = 4096},
    {.name = "p_mma0", .kernel = (const void*)p_mma0, .tpb = TPB, .expect = IMMA "=8", .iters = 4096},
    {.name = "p_fold", .kernel = (const void*)p_fold, .tpb = TPB, .expect = "IMAD=32", .iters = 2048},
    {.name = "p_ffma", .kernel = (const void*)p_ffma, .tpb = TPB, .expect = "FFMA=64", .iters = 4096},
    {.name = "p_mixed", .kernel = (const void*)p_mixed, .tpb = TPB, .expect = IMMA "=4,FFMA=64", .iters = 2048},
    {.name = "p_honest16", .kernel = (const void*)p_honest16, .tpb = TPB, .expect = "IMMA.16816.S8.S8=8", .iters = 2048},
    {.name = "p_mma16", .kernel = (const void*)p_mma16, .tpb = TPB, .expect = "IMMA.16816.S8.S8=8", .iters = 4096},
    {.name = "p_mma16z", .kernel = (const void*)p_mma16z, .tpb = TPB, .expect = "IMMA.16816.S8.S8=8", .iters = 4096},
    {.name = "p_light", .kernel = (const void*)p_light, .tpb = 32, .expect = "IMAD=32", .iters = 2048},
};

// NVML's power reading beside bench.cuh's clock and reasons (dlopen'd, as there).
struct Power {
  void* lib = nullptr;
  void* h = nullptr;
  int (*power)(void*, unsigned*) = nullptr;
  int (*temp)(void*, int, unsigned*) = nullptr;
  explicit Power(const std::string& uuid) {
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) return;
    auto init = reinterpret_cast<int (*)()>(dlsym(lib, "nvmlInit_v2"));
    auto by_uuid = reinterpret_cast<int (*)(const char*, void**)>(dlsym(lib, "nvmlDeviceGetHandleByUUID"));
    power = reinterpret_cast<int (*)(void*, unsigned*)>(dlsym(lib, "nvmlDeviceGetPowerUsage"));
    temp = reinterpret_cast<int (*)(void*, int, unsigned*)>(dlsym(lib, "nvmlDeviceGetTemperature"));
    if (!init || !by_uuid || init() != 0 || by_uuid(uuid.c_str(), &h) != 0) h = nullptr;
  }
  unsigned mw() const {
    unsigned v = 0;
    return h && power && power(h, &v) == 0 ? v : 0;
  }
  unsigned celsius() const {
    unsigned v = 0;
    return h && temp && temp(h, 0, &v) == 0 ? v : 0;
  }
};

static int sustain(const std::string& name, double seconds, double scale) {
  const OpProbe* q = nullptr;
  for (const OpProbe& p : PROBES)
    if (name == p.name) q = &p;
  if (!q) {
    fprintf(stderr, "ERR no probe %s\n", name.c_str());
    return 2;
  }
  GK_CK(cudaSetDevice(0));
  cudaDeviceProp prop;
  GK_CK(cudaGetDeviceProperties(&prop, 0));
  gk::Nvml nv(0);
  Power pw(nv.uuid());
  const int blocks = prop.multiProcessorCount;
  OpClk* rec;
  unsigned* sink;
  unsigned char* g;
  GK_CK(cudaMalloc(&rec, sizeof(OpClk) * blocks));
  GK_CK(cudaMalloc(&sink, sizeof(unsigned) * blocks * q->tpb));
  GK_CK(cudaMalloc(&g, 1 << 20));
  OpArgs args{rec, sink, g, 1 << 20, std::max(1, (int)(q->iters * scale)), 0};
  void* kargs[] = {&args};
  cudaEvent_t e0, e1;
  GK_CK(cudaEventCreate(&e0));
  GK_CK(cudaEventCreate(&e1));
  std::vector<OpClk> h(blocks);
  const auto start = std::chrono::steady_clock::now();
  for (int launch = 0;; launch++) {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (t >= seconds) break;
    GK_CK(cudaEventRecord(e0));
    GK_CK(cudaLaunchKernel(q->kernel, dim3(blocks), dim3(q->tpb), kargs, 0, 0));
    GK_CK(cudaEventRecord(e1));
    const gk::Sample during = nv.sample();
    const unsigned mw = pw.mw();
    GK_CK(cudaEventSynchronize(e1));
    float ms = 0;
    GK_CK(cudaEventElapsedTime(&ms, e0, e1));
    GK_CK(cudaMemcpy(h.data(), rec, sizeof(OpClk) * blocks, cudaMemcpyDeviceToHost));
    std::vector<double> mhz, cyc;
    for (const OpClk& r : h) {
      cyc.push_back((double)r.cyc);
      if (r.stop > 0) mhz.push_back(1e3 * (double)r.cyc / (double)r.stop);
    }
    printf("{\"op\": \"sustain\", \"probe\": \"%s\", \"launch\": %d, \"t\": %.3f, \"ms\": %.4f, \"iters\": %d, "
           "\"blocks\": %d, \"cyc_median\": %.0f, \"clock_mhz_median\": %.2f, \"clock_mhz_min\": %.2f, "
           "\"nvml_sm_mhz\": %u, \"nvml_mw\": %u, \"temp_c\": %u, \"throttle\": \"0x%llx\"}\n",
           q->name, launch, t, ms, args.iters, blocks, gk::median(cyc), mhz.empty() ? 0.0 : gk::median(mhz),
           mhz.empty() ? 0.0 : *std::min_element(mhz.begin(), mhz.end()), during.sm_mhz, mw, pw.celsius(),
           during.throttle);
    fflush(stdout);
  }
  return 0;
}

// `--calib NAME SECONDS [REPS]`: REPS launches, each sized to last about SECONDS, timed three ways: every block's clock64
// cycles, its %globaltimer span, and the host's steady_clock around the launch and its synchronize. The longest block's
// cycles over the host span is a lower bound on the SM clock that reads no timer on the GPU (the host span adds the
// launch's latency, tens of microseconds); over the block's own %globaltimer span it is the clock power.cu reports.
static int calib(const std::string& name, double seconds, int reps) {
  const OpProbe* q = nullptr;
  for (const OpProbe& p : PROBES)
    if (name == p.name) q = &p;
  if (!q) {
    fprintf(stderr, "ERR no probe %s\n", name.c_str());
    return 2;
  }
  GK_CK(cudaSetDevice(0));
  cudaDeviceProp prop;
  GK_CK(cudaGetDeviceProperties(&prop, 0));
  gk::Nvml nv(0);
  Power pw(nv.uuid());
  const int blocks = prop.multiProcessorCount;
  OpClk* rec;
  unsigned* sink;
  unsigned char* g;
  GK_CK(cudaMalloc(&rec, sizeof(OpClk) * blocks));
  GK_CK(cudaMalloc(&sink, sizeof(unsigned) * blocks * q->tpb));
  GK_CK(cudaMalloc(&g, 1 << 20));
  OpArgs args{rec, sink, g, 1 << 20, q->iters * 16, 0};
  void* kargs[] = {&args};
  std::vector<OpClk> h(blocks);
  auto launch = [&]() {
    const auto t0 = std::chrono::steady_clock::now();
    GK_CK(cudaLaunchKernel(q->kernel, dim3(blocks), dim3(q->tpb), kargs, 0, 0));
    GK_CK(cudaDeviceSynchronize());
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
  };
  launch();
  const double pilot_ns = launch();
  args.iters = (int)std::min(2.0e9, args.iters * seconds * 1e9 / pilot_ns);
  for (int rep = 0; rep < reps; rep++) {
    const auto t0 = std::chrono::steady_clock::now();
    GK_CK(cudaLaunchKernel(q->kernel, dim3(blocks), dim3(q->tpb), kargs, 0, 0));
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds / 2));
    const gk::Sample during = nv.sample();
    const unsigned mw = pw.mw();
    GK_CK(cudaDeviceSynchronize());
    const double host_ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    GK_CK(cudaMemcpy(h.data(), rec, sizeof(OpClk) * blocks, cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> ts(2 * blocks);
    GK_CK(cudaMemcpy(ts.data(), g, sizeof(unsigned long long) * 2 * blocks, cudaMemcpyDeviceToHost));
    long long cyc_max = 0, cyc_min = h[0].cyc, gt_max = 0, gt_of_cyc_max = 0;
    unsigned long long t0_min = ts[0], t0_max = ts[0], t1_min = ts[1], t1_max = ts[1];
    std::vector<double> mhz;
    for (int b = 0; b < blocks; b++) {
      const OpClk& r = h[b];
      if (r.cyc > cyc_max) cyc_max = r.cyc, gt_of_cyc_max = r.stop;
      cyc_min = std::min(cyc_min, r.cyc);
      gt_max = std::max(gt_max, r.stop);
      if (r.stop > 0) mhz.push_back(1e3 * (double)r.cyc / (double)r.stop);
      t0_min = std::min(t0_min, ts[2 * b]), t0_max = std::max(t0_max, ts[2 * b]);
      t1_min = std::min(t1_min, ts[2 * b + 1]), t1_max = std::max(t1_max, ts[2 * b + 1]);
    }
    printf("{\"op\": \"calib\", \"probe\": \"%s\", \"rep\": %d, \"iters\": %d, \"blocks\": %d, \"host_ns\": %.0f, "
           "\"gt_ns_max\": %lld, \"gt_kernel_ns\": %llu, \"gt_start_spread_ns\": %llu, \"gt_end_spread_ns\": %llu, "
           "\"cyc_max\": %lld, \"cyc_min\": %lld, \"gt_ns_of_cyc_max\": %lld, \"clock_mhz_host\": %.3f, "
           "\"clock_mhz_gt_min\": %.3f, \"clock_mhz_gt_median\": %.3f, \"clock_mhz_gt_max\": %.3f, "
           "\"gt_over_host\": %.7f, \"nvml_sm_mhz\": %u, \"nvml_mw\": %u, \"temp_c\": %u, \"throttle\": \"0x%llx\"}\n",
           q->name, rep, args.iters, blocks, host_ns, gt_max, t1_max - t0_min, t0_max - t0_min, t1_max - t1_min, cyc_max,
           cyc_min, gt_of_cyc_max, 1e3 * (double)cyc_max / host_ns,
           mhz.empty() ? 0.0 : *std::min_element(mhz.begin(), mhz.end()), mhz.empty() ? 0.0 : gk::median(mhz),
           mhz.empty() ? 0.0 : *std::max_element(mhz.begin(), mhz.end()), (double)(t1_max - t0_min) / host_ns,
           during.sm_mhz, mw, pw.celsius(), during.throttle);
    fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 4 && std::string(argv[1]) == "--sustain")
    return sustain(argv[2], atof(argv[3]), argc >= 5 ? atof(argv[4]) : 1.0);
  if (argc >= 4 && std::string(argv[1]) == "--calib")
    return calib(argv[2], atof(argv[3]), argc >= 5 ? atoi(argv[4]) : 3);
  return op_main(argc, argv, PROBES);
}
