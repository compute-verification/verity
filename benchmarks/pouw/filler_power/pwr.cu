// The certified filler's board power beside decode (HC-POWER, cmp-13). FILL's DRAM-free chained-MMA filler (dcap.cu's
// dfill_k) and variants of it, alone and beside dcap's decode step on a green-context split, each arm sustained for
// seconds so that NVML's energy counter, instant power, SM clock and clock-event reasons describe the arm rather than the
// idle board around a 40 ms window. Clocks are read, never set.
//
// Modes (one JSON line per arm):
//   pgate                                         the variants' receipts against the host (int8 kinds), KIND 0 PAT 0 against
//                                                 FILL's dfill_ref, and each control
//   pidle SECS                                    the idle board
//   palone FILL_SMS KIND PAT W NC WARM MEAS       the filler alone on FILL_SMS SMs (the GPU's count: no split)
//   pfill B L FILL_SMS KIND PAT W NC WARM MEAS ROUNDS [CTL_PAT]
//                                                 per round: the step on all SMs (full), on its partition with cuBLASLt
//                                                 re-tuned there (split), the same beside the filler (split_fill), beside
//                                                 the CTL_PAT filler (split_ctl, when given), the filler alone (fill), and
//                                                 full again (full2)
// KIND: 0 int8 m16n8k32 s32 (FILL v1's step), 1 FP8 e4m3 m16n8k32 f32, 2 NVFP4 m16n8k64 f32 (ue4m3 block scales 1.0).
// Each step's D, read as the next A fragment in place (FILL's byte map); FP8 clears bit 3 of every byte (0xF7F7F7F7), so no
// A or B byte is the e4m3 NaN; e2m1 has no NaN. PAT, the operands from the coins: 0 as FILL (B_0, B_1 uniform), 1 B_1 = B_0,
// 2 B's bytes in {0, 1}, 3 B = 0 (a power control, not certified work: D = C every step), 4 B's bytes in [0, 16).
// PF_SLEEP_NS: each warp sleeps that long after each tile (duty cycling). The receipt is FILL's fold for every kind.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a pwr.cu -lcublasLt -lnvidia-ml -ldl
#define main dcap_main
#include "../decode_capacity/dcap.cu"
#undef main

#include <mutex>

// ---------------------------------------------------------------- NVML sampler
struct PSample {
  double t;
  float mhz, w_inst, w_avg;
  unsigned long long mj, why;
  int temp;
};
static float field_w(const nvmlFieldValue_t& f) {
  if (f.nvmlReturn != NVML_SUCCESS) return -1;
  switch (f.valueType) {
    case NVML_VALUE_TYPE_UNSIGNED_INT: return f.value.uiVal / 1000.0f;
    case NVML_VALUE_TYPE_UNSIGNED_LONG: return f.value.ulVal / 1000.0f;
    case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: return f.value.ullVal / 1000.0f;
    case NVML_VALUE_TYPE_SIGNED_INT: return f.value.siVal / 1000.0f;
    default: return -1;
  }
}
struct PSampler {
  nvmlDevice_t nv;
  std::thread th;
  std::atomic<bool> on{false};
  std::mutex mu;
  std::vector<PSample> v;
  PSample read() const {
    PSample s{};
    s.t = now_ms();
    unsigned c = 0, tc = 0;
    s.mhz = nvmlDeviceGetClockInfo(nv, NVML_CLOCK_SM, &c) == NVML_SUCCESS ? (float)c : -1;
    nvmlFieldValue_t f[2];
    memset(f, 0, sizeof f);
    f[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
    f[1].fieldId = NVML_FI_DEV_POWER_AVERAGE;
    if (nvmlDeviceGetFieldValues(nv, 2, f) == NVML_SUCCESS) {
      s.w_inst = field_w(f[0]);
      s.w_avg = field_w(f[1]);
    } else {
      s.w_inst = s.w_avg = -1;
    }
    if (nvmlDeviceGetTotalEnergyConsumption(nv, &s.mj) != NVML_SUCCESS) s.mj = 0;
    if (nvmlDeviceGetCurrentClocksEventReasons(nv, &s.why) != NVML_SUCCESS) s.why = ~0ull;
    s.temp = nvmlDeviceGetTemperature(nv, NVML_TEMPERATURE_GPU, &tc) == NVML_SUCCESS ? (int)tc : -1;
    return s;
  }
  void start(nvmlDevice_t n) {
    nv = n;
    on = true;
    th = std::thread([this] {
      while (on.load()) {
        const PSample s = read();
        {
          std::lock_guard<std::mutex> g(mu);
          v.push_back(s);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }
  void stop() {
    on = false;
    th.join();
  }
  std::vector<PSample> between(double t0, double t1) {
    std::lock_guard<std::mutex> g(mu);
    std::vector<PSample> o;
    for (const PSample& s : v)
      if (s.t >= t0 && s.t <= t1) o.push_back(s);
    return o;
  }
};
static PSampler g_samp;

// the window's power, clock and clock-event reasons as JSON fields
static void print_window(double t0, double t1) {
  const std::vector<PSample> s = g_samp.between(t0, t1);
  double mhz = 0, wi = 0, wa = 0, mhz_min = 1e9;
  int nm = 0, ni = 0, na = 0, n_pcap = 0, n_hws = 0, n_th = 0, n_brake = 0, n_idle = 0;
  unsigned long long why_or = 0;
  std::vector<float> clocks;
  for (const PSample& x : s) {
    if (x.mhz > 0) { mhz += x.mhz; nm++; mhz_min = std::min(mhz_min, (double)x.mhz); clocks.push_back(x.mhz); }
    if (x.w_inst > 0) { wi += x.w_inst; ni++; }
    if (x.w_avg > 0) { wa += x.w_avg; na++; }
    if (x.why != ~0ull) {
      why_or |= x.why;
      n_pcap += (x.why & 0x4) != 0;
      n_hws += (x.why & 0x8) != 0;
      n_th += (x.why & (0x20 | 0x40)) != 0;
      n_brake += (x.why & 0x80) != 0;
      n_idle += (x.why & 0x1) != 0;
    }
  }
  std::sort(clocks.begin(), clocks.end());
  // energy: between the first and last samples whose counter moved, so a stale first reading doesn't stretch the window
  double we = -1;
  if (s.size() >= 2) {
    size_t i0 = 0, i1 = s.size() - 1;
    while (i0 + 1 < s.size() && s[i0 + 1].mj == s[0].mj) i0++;
    while (i1 > i0 && s[i1 - 1].mj == s.back().mj) i1--;
    if (i1 > i0 && s[i1].mj > s[i0].mj) we = (double)(s[i1].mj - s[i0].mj) / (s[i1].t - s[i0].t);
  }
  const size_t n = s.size();
  printf("\"win_s\": %.3f, \"samples\": %zu, \"watt_energy\": %.1f, \"watt_inst\": %.1f, \"watt_avg1s\": %.1f, \"mhz\": %.1f, "
         "\"mhz_min\": %.0f, \"mhz_p10\": %.0f, \"frac_pcap\": %.3f, \"frac_hw_slowdown\": %.3f, \"frac_thermal\": %.3f, "
         "\"frac_brake\": %.3f, \"frac_idle\": %.3f, \"reasons_or\": \"0x%llx\", \"temp\": [%d, %d]",
         (t1 - t0) / 1e3, n, we, ni ? wi / ni : -1, na ? wa / na : -1, nm ? mhz / nm : -1, nm ? mhz_min : -1,
         clocks.empty() ? -1.0 : (double)clocks[clocks.size() / 10], n ? (double)n_pcap / n : -1,
         n ? (double)n_hws / n : -1, n ? (double)n_th / n : -1, n ? (double)n_brake / n : -1, n ? (double)n_idle / n : -1,
         why_or, n ? s.front().temp : -1, n ? s.back().temp : -1);
}

// ---------------------------------------------------------------- the filler's variants
struct PPat {
  uint32_t bmask;
  int bsame;
  int sleep_ns;
};
static PPat ppat(int pat) {
  PPat p{0xffffffffu, 0, 0};
  if (pat == 1) p.bsame = 1;
  if (pat == 2) p.bmask = 0x01010101u;
  if (pat == 3) p.bmask = 0;
  if (pat == 4) p.bmask = 0x0f0f0f0fu;
  if (const char* e = getenv("PF_SLEEP_NS")) p.sleep_ns = atoi(e);
  return p;
}
constexpr uint32_t FP8_MASK = 0xF7F7F7F7u;  // clears bit 3 of each byte: never 0x7F or 0xFF, e4m3's NaNs
constexpr uint32_t UE4M3_ONE = 0x38383838u;
template <int KIND>
__device__ __forceinline__ void pstep(uint32_t (&a)[4], const uint32_t (&b)[2], const uint32_t (&c)[4]) {
  if constexpr (KIND == 0) {
    dstep(a, b, c);
  } else {
    float d0, d1, d2, d3;
    if constexpr (KIND == 1)
      asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(__uint_as_float(c[0])),
            "f"(__uint_as_float(c[1])), "f"(__uint_as_float(c[2])), "f"(__uint_as_float(c[3])));
    else
      asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {0, 0}, {%15}, {0, 0};"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(__uint_as_float(c[0])),
            "f"(__uint_as_float(c[1])), "f"(__uint_as_float(c[2])), "f"(__uint_as_float(c[3])), "r"(UE4M3_ONE),
            "r"(UE4M3_ONE));
    const uint32_t m = KIND == 1 ? FP8_MASK : 0xffffffffu;
    a[0] = __float_as_uint(d0) & m;
    a[1] = __float_as_uint(d2) & m;
    a[2] = __float_as_uint(d3) & m;
    a[3] = __float_as_uint(d1) & m;
  }
}
// dfill_k with the operand pattern, the kind and the sleep; KIND 0 with PPat{~0, 0, 0} is dfill_k step for step
template <int KIND, int NC>
__global__ void __launch_bounds__(512) pfill_k(uint64_t c0, uint64_t c1, unsigned long long tile0, int tiles, int steps,
                                              PPat pat, unsigned long long* __restrict__ rec) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const unsigned long long wid = (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const uint32_t im = KIND == 1 ? FP8_MASK : 0xffffffffu;
  for (int i = 0; i < tiles; i++) {
    const unsigned long long idx = wid * tiles + i;
    const uint64_t seed = dseed(c0, c1, tile0 + idx);
    uint32_t b[2][2], c[4], a[NC][4];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int jj = pat.bsame ? 0 : j;
      b[j][0] = dword(seed, 1 + jj, g, t) & pat.bmask & im;
      b[j][1] = dword(seed, 1 + jj, g, 4 + t) & pat.bmask & im;
    }
    const int cv[4] = {dcval(seed, g, 2 * t), dcval(seed, g, 2 * t + 1), dcval(seed, g + 8, 2 * t),
                       dcval(seed, g + 8, 2 * t + 1)};
#pragma unroll
    for (int q = 0; q < 4; q++) c[q] = KIND == 0 ? (uint32_t)cv[q] : __float_as_uint((float)cv[q]);
#pragma unroll
    for (int ch = 0; ch < NC; ch++) {
      a[ch][0] = dword(seed, 16 + ch, g, t) & im;
      a[ch][1] = dword(seed, 16 + ch, g + 8, t) & im;
      a[ch][2] = dword(seed, 16 + ch, g, 4 + t) & im;
      a[ch][3] = dword(seed, 16 + ch, g + 8, 4 + t) & im;
    }
#pragma unroll 1
    for (int s = 0; s < steps; s += 8) {
#pragma unroll
      for (int u = 0; u < 8; u++)
#pragma unroll
        for (int ch = 0; ch < NC; ch++) pstep<KIND>(a[ch], b[u & 1], c);
    }
    uint64_t h = 0;
#pragma unroll
    for (int ch = 0; ch < NC; ch++)
      h ^= drec(seed, ch, g, t, a[ch][0]) ^ drec(seed, ch, g + 8, t, a[ch][1]) ^ drec(seed, ch, g, 4 + t, a[ch][2]) ^
           drec(seed, ch, g + 8, 4 + t, a[ch][3]);
#pragma unroll
    for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
    if (lane == 0) rec[idx] = h;
    if (pat.sleep_ns) __nanosleep(pat.sleep_ns);
  }
}
// dfill_ref with the pattern (int8 only)
static uint64_t pref(uint64_t c0, uint64_t c1, uint64_t tile, int nc, int steps, const PPat& p) {
  const uint64_t seed = dseed(c0, c1, tile);
  int8_t Bt[2][8][32];
  int32_t C[16][8];
  for (int b = 0; b < 2; b++)
    for (int n = 0; n < 8; n++)
      for (int w = 0; w < 8; w++) {
        const uint32_t x = dword(seed, 1 + (p.bsame ? 0 : b), n, w) & p.bmask;
        for (int q = 0; q < 4; q++) Bt[b][n][4 * w + q] = (int8_t)(x >> (8 * q));
      }
  for (int r = 0; r < 16; r++)
    for (int n = 0; n < 8; n++) C[r][n] = dcval(seed, r, n);
  uint64_t h = 0;
  for (int ch = 0; ch < nc; ch++) {
    int8_t A[16][32];
    for (int r = 0; r < 16; r++)
      for (int w = 0; w < 8; w++) {
        const uint32_t x = dword(seed, 16 + ch, r, w);
        for (int q = 0; q < 4; q++) A[r][4 * w + q] = (int8_t)(x >> (8 * q));
      }
    for (int s = 0; s < steps; s++) {
      int32_t D[16][8];
      for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++) {
          int32_t acc = C[r][n];
          for (int k = 0; k < 32; k++) acc += (int32_t)A[r][k] * (int32_t)Bt[s & 1][n][k];
          D[r][n] = acc;
        }
      auto byte = [](int32_t v, int q) { return (int8_t)((uint32_t)v >> (8 * q)); };
      for (int g = 0; g < 8; g++)
        for (int t = 0; t < 4; t++)
          for (int q = 0; q < 4; q++) {
            A[g][4 * t + q] = byte(D[g][2 * t], q);
            A[g + 8][4 * t + q] = byte(D[g + 8][2 * t], q);
            A[g][16 + 4 * t + q] = byte(D[g + 8][2 * t + 1], q);
            A[g + 8][16 + 4 * t + q] = byte(D[g][2 * t + 1], q);
          }
    }
    for (int r = 0; r < 16; r++)
      for (int w = 0; w < 8; w++) {
        uint32_t x = 0;
        for (int q = 0; q < 4; q++) x |= (uint32_t)(uint8_t)A[r][4 * w + q] << (8 * q);
        h ^= drec(seed, ch, r, w, x);
      }
  }
  return h;
}

struct PFill {
  int kind = 0, pat = 0, W = 8, NC = 4, S = 2048, tiles = 1, grid = 0;
  PPat p{0xffffffffu, 0, 0};
  uint64_t c0 = 0, c1 = 0;
  unsigned long long* rec = nullptr;
  long long per() const { return (long long)grid * W * tiles; }
  double mmas() const { return (double)per() * NC * S; }
  double macs() const { return mmas() * (kind == 2 ? 8192.0 : 4096.0); }
};
template <int KIND>
static void pfill_launch_k(const PFill& f, unsigned long long t0, cudaStream_t st) {
  const dim3 gr(f.grid), bl(32 * f.W);
  switch (f.NC) {
    case 1: pfill_k<KIND, 1><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.p, f.rec); break;
    case 2: pfill_k<KIND, 2><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.p, f.rec); break;
    case 4: pfill_k<KIND, 4><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.p, f.rec); break;
    default: fprintf(stderr, "NC must be 1, 2 or 4\n"); exit(1);
  }
}
static void pfill_launch(const PFill& f, long long chunk, cudaStream_t st) {
  const unsigned long long t0 = (unsigned long long)chunk * f.per();
  if (f.kind == 0) pfill_launch_k<0>(f, t0, st);
  else if (f.kind == 1) pfill_launch_k<1>(f, t0, st);
  else pfill_launch_k<2>(f, t0, st);
}
static PFill pfill_make(int grid, int kind, int pat, int W, int NC, int S) {
  PFill f;
  f.grid = grid; f.kind = kind; f.pat = pat; f.W = W; f.NC = NC; f.S = S; f.c0 = g_c0; f.c1 = g_c1; f.p = ppat(pat);
  const int lg = getenv("DFILL_CHUNK_LOG2") ? atoi(getenv("DFILL_CHUNK_LOG2")) : 19;
  f.tiles = std::max(1, (int)((1LL << lg) / ((long long)W * NC * S)));
  CK(cudaMalloc(&f.rec, f.per() * sizeof(unsigned long long)));
  return f;
}
// the host's receipts for n tiles of chunk `chunk` (int8 kinds; -1 for the others, which have no host reference here)
static int pfill_check(const PFill& f, long long chunk, int n) {
  if (f.kind != 0) return -1;
  std::vector<unsigned long long> h(f.per());
  CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; i++) {
    const long long idx = (long long)i * f.per() / n + (i * 7919) % std::max(1LL, f.per() / n);
    bad += h[idx] != pref(f.c0, f.c1, (unsigned long long)chunk * f.per() + idx, f.NC, f.S, f.p);
  }
  return bad;
}
// a host thread keeps `depth` chunks queued on `fs` and notes the host time each is seen done
struct PPump {
  std::thread th;
  std::atomic<bool> on{false};
  std::atomic<long long> done{0}, launched{0};
  std::vector<double> t;
};
static void ppump_start(PPump& p, const PFill& f, cudaStream_t fs, int depth) {
  p.on = true;
  p.done = 0;
  p.t.clear();
  p.th = std::thread([&p, &f, fs, depth] {
    std::vector<cudaEvent_t> ev(depth);
    for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync));
    long long k = 0;
    for (; p.on.load(); k++) {
      const int slot = (int)(k % depth);
      if (k >= depth) {
        CK(cudaEventSynchronize(ev[slot]));
        p.t.push_back(now_ms());
        p.done++;
      }
      pfill_launch(f, k, fs);
      CK(cudaEventRecord(ev[slot], fs));
    }
    p.launched = k;
    CK(cudaStreamSynchronize(fs));
    for (auto& e : ev) CK(cudaEventDestroy(e));
  });
}
static void ppump_stop(PPump& p) {
  p.on = false;
  p.th.join();
}
static long long chunks_in(const PPump& p, double t0, double t1) {
  long long n = 0;
  for (double x : p.t) n += x > t0 && x <= t1;
  return n;
}

// ---------------------------------------------------------------- sustained arms
static cudaGraphExec_t capture(Dev& d, cudaStream_t st, const std::function<void()>& work) {
  cudaGraph_t g;
  cudaGraphExec_t ge;
  d.st = st;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  work();
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaGraphDestroy(g));
  return ge;
}
// the graph back to back on `st` for `secs`, at most R reps in flight; each rep's event time into `ms`
static void step_loop(cudaGraphExec_t ge, cudaStream_t st, double secs, std::vector<float>* ms, double* t0, double* t1) {
  constexpr int R = 4;
  cudaEvent_t e[R][2];
  for (auto& x : e) {
    CK(cudaEventCreate(&x[0]));
    CK(cudaEventCreate(&x[1]));
  }
  const double start = now_ms();
  if (t0) *t0 = start;
  long long k = 0, harvested = 0;
  auto harvest = [&] {
    const int s = (int)(harvested % R);
    CK(cudaEventSynchronize(e[s][1]));
    float x;
    CK(cudaEventElapsedTime(&x, e[s][0], e[s][1]));
    if (ms) ms->push_back(x);
    harvested++;
  };
  while (now_ms() - start < secs * 1e3) {
    if (k - harvested == R) harvest();
    const int s = (int)(k % R);
    CK(cudaEventRecord(e[s][0], st));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(e[s][1], st));
    k++;
  }
  while (harvested < k) harvest();
  if (t1) *t1 = now_ms();
  for (auto& x : e) {
    CK(cudaEventDestroy(x[0]));
    CK(cudaEventDestroy(x[1]));
  }
}
static double med(std::vector<float> v) {
  if (v.empty()) return -1;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
static double mean(const std::vector<float>& v) {
  double s = 0;
  for (float x : v) s += x;
  return v.empty() ? -1 : s / v.size();
}
static void print_head(const char* mode, const char* arm, int round) {
  printf("{\"mode\": \"%s\", \"arm\": \"%s\", \"round\": %d, \"coins\": [\"%016llx\", \"%016llx\"], ", mode, arm, round,
         (unsigned long long)g_c0, (unsigned long long)g_c1);
}
static void print_fill(const PFill& f, int fill_sms) {
  printf("\"kind\": %d, \"pat\": %d, \"W\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %d, \"grid\": %d, \"fill_sms\": %d, "
         "\"sleep_ns\": %d, ",
         f.kind, f.pat, f.W, f.NC, f.S, f.tiles, f.grid, fill_sms, f.p.sleep_ns);
}
// one arm: the step graph on `ss` (if any) beside the filler on `fs` (if any), warm then measured; prints the arm's tail
static void arm(Dev& d, cudaGraphExec_t ge, cudaStream_t ss, const PFill* f, cudaStream_t fs, double warm, double meas,
                double step_macs) {
  PPump pump;
  if (f) {
    ppump_start(pump, *f, fs, 3);
    while (pump.done.load() < 2) {
    }
  }
  double t0, t1;
  std::vector<float> ms;
  if (ge) {
    step_loop(ge, ss, warm, nullptr, nullptr, nullptr);
    step_loop(ge, ss, meas, &ms, &t0, &t1);
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds((long long)(warm * 1e3)));
    t0 = now_ms();
    std::this_thread::sleep_for(std::chrono::milliseconds((long long)(meas * 1e3)));
    t1 = now_ms();
  }
  long long nch = 0;
  int bad = 0;
  if (f) {
    ppump_stop(pump);
    nch = chunks_in(pump, t0, t1);
    bad = pfill_check(*f, pump.launched.load() - 1, 8);
  }
  const double secs = (t1 - t0) * 1e-3;
  printf("\"step_reps\": %zu, \"step_ms\": %.4f, \"step_ms_mean\": %.4f, \"step_mac_per_s\": %.6e, \"fill_chunks\": %lld, "
         "\"fill_mma_per_s\": %.6e, \"fill_mac_per_s\": %.6e, \"rec_checked\": %d, \"rec_bad\": %d, ",
         ms.size(), med(ms), mean(ms), ge && !ms.empty() ? step_macs / (med(ms) * 1e-3) : 0.0, nch,
         f ? nch * f->mmas() / secs : 0.0, f ? nch * f->macs() / secs : 0.0, f && f->kind == 0 ? 8 : 0, bad);
  print_window(t0, t1);
  printf("}\n");
  fflush(stdout);
}

// ---------------------------------------------------------------- modes
static int pgate() {
  int fails = 0;
  for (int pat : {0, 1, 2, 3, 4})
    for (int nc : {2, 4})
      for (int S : {8, 2048}) {
        PFill f;
        f.grid = 3; f.W = 4; f.NC = nc; f.S = S; f.tiles = 2; f.c0 = g_c0; f.c1 = g_c1; f.kind = 0; f.pat = pat;
        f.p = ppat(pat);
        f.p.sleep_ns = 0;
        CK(cudaMalloc(&f.rec, f.per() * sizeof(unsigned long long)));
        CK(cudaMemset(f.rec, 0xA5, f.per() * sizeof(unsigned long long)));
        pfill_launch(f, 5, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        std::vector<unsigned long long> h(f.per());
        CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
        long bad = 0, ctl_equal = 0, v1_diff = 0;
        for (long long i = 0; i < f.per(); i++) {
          const unsigned long long tile = 5ull * f.per() + i;
          bad += h[i] != pref(f.c0, f.c1, tile, nc, S, f.p);
          ctl_equal += h[i] == pref(f.c0, f.c1, tile, nc, S - 8, f.p);
          if (pat == 0) v1_diff += h[i] != dfill_ref(f.c0, f.c1, tile, nc, S);
        }
        // PAT 3 (B = 0) freezes at D = C after one step, so S - 8 steps give the same receipt: it is the power control,
        // never certified work, and only its exactness is gated
        const bool ok = !bad && (pat == 3 || !ctl_equal) && !v1_diff;
        printf("{\"mode\": \"pgate\", \"kind\": 0, \"pat\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %lld, \"mismatches\": %ld, "
               "\"control_equal\": %ld, \"v1_diff\": %ld, \"certified\": %s, \"ok\": %s}\n",
               pat, nc, S, f.per(), bad, ctl_equal, v1_diff, pat == 3 ? "false" : "true", ok ? "true" : "false");
        fails += !ok;
        CK(cudaFree(f.rec));
      }
  // the float kinds: no host reference here; a receipt that isn't the 0xA5 fill and differs from S - 8's shows the chain ran
  for (int kind : {1, 2}) {
    PFill f;
    f.grid = 3; f.W = 4; f.NC = 4; f.S = 2048; f.tiles = 2; f.c0 = g_c0; f.c1 = g_c1; f.kind = kind; f.p = ppat(0);
    f.p.sleep_ns = 0;
    CK(cudaMalloc(&f.rec, f.per() * sizeof(unsigned long long)));
    std::vector<unsigned long long> h0(f.per()), h1(f.per());
    CK(cudaMemset(f.rec, 0xA5, f.per() * sizeof(unsigned long long)));
    pfill_launch(f, 5, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h0.data(), f.rec, h0.size() * 8, cudaMemcpyDeviceToHost));
    f.S = 2040;
    pfill_launch(f, 5, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    CK(cudaMemcpy(h1.data(), f.rec, h1.size() * 8, cudaMemcpyDeviceToHost));
    long poison = 0, same = 0;
    for (long long i = 0; i < f.per(); i++) {
      poison += h0[i] == 0xA5A5A5A5A5A5A5A5ull;
      same += h0[i] == h1[i];
    }
    printf("{\"mode\": \"pgate\", \"kind\": %d, \"pat\": 0, \"NC\": 4, \"S\": 2048, \"tiles\": %lld, \"poison_left\": %ld, "
           "\"equal_to_S_minus_8\": %ld, \"reference\": null}\n",
           kind, f.per(), poison, same);
    fails += poison != 0 || same != 0;
    CK(cudaFree(f.rec));
  }
  fflush(stdout);
  return fails;
}

static void run_pidle(Dev& d, double secs) {
  print_head("pidle", "idle", 0);
  arm(d, nullptr, nullptr, nullptr, nullptr, 0.5, secs, 0);
}

// the filler's stream on fill_sms SMs: the default stream when fill_sms is the whole GPU, else a green split's rest
static cudaStream_t fill_stream(Dev& d, int fill_sms, cudaStream_t* ss, int got[2]) {
  if (fill_sms >= d.sms) {
    got[0] = 0;
    got[1] = d.sms;
    *ss = nullptr;
    return d.st;
  }
  cudaStream_t fs = nullptr;
  const int e = gc_split(d.sms - fill_sms, ss, &fs, got);
  if (e || !fs) {
    fprintf(stderr, "gc_split(%d): %d\n", d.sms - fill_sms, e);
    exit(7);
  }
  return fs;
}

static void run_palone(Dev& d, int fill_sms, int kind, int pat, int W, int NC, double warm, double meas) {
  cudaStream_t ss;
  int got[2];
  cudaStream_t fs = fill_stream(d, fill_sms, &ss, got);
  PFill f = pfill_make(got[1], kind, pat, W, NC, 2048);
  print_head("palone", "fill", 0);
  print_fill(f, got[1]);
  arm(d, nullptr, nullptr, &f, fs, warm, meas, 0);
  CK(cudaFree(f.rec));
}

static void run_pfill(Dev& d, int B, int L, int fill_sms, int kind, int pat, int W, int NC, double warm, double meas,
                      int rounds, int ctl_pat) {
  const std::vector<Lt> p = lin_plans(d, B);
  cudaStream_t full = d.st, ss = nullptr;
  int got[2];
  cudaStream_t fs = fill_stream(d, fill_sms, &ss, got);
  d.st = ss;
  const std::vector<Lt> p2 = lin_plans(d, B);
  d.st = full;
  PFill f = pfill_make(got[1], kind, pat, W, NC, 2048);
  PFill fc;
  if (ctl_pat >= 0) fc = pfill_make(got[1], kind, ctl_pat, W, NC, 2048);
  bool wrapped = false;
  const cudaGraphExec_t g_full = capture(d, full, [&] { enqueue_step(d, p, 2, B, L, &wrapped); });
  const cudaGraphExec_t g_split = capture(d, ss, [&] { enqueue_step(d, p2, 2, B, L, &wrapped); });
  d.st = full;
  double step_macs = 0;
  for (int j = 0; j < 4; j++) step_macs += (double)LAYERS * B * LN[j] * LK[j];
  step_macs += (double)B * VOCAB * HID + 2.0 * LAYERS * B * HQ * HD * (double)L;
  auto head = [&](const char* name, int r, const PFill& ff) {
    print_head("pfill", name, r);
    printf("\"B\": %d, \"L\": %d, \"step_sms\": %d, \"kv_wrapped\": %s, \"step_macs\": %.6e, ", B, L, got[0],
           wrapped ? "true" : "false", step_macs);
    print_fill(ff, got[1]);
  };
  for (int r = 0; r < rounds; r++) {
    head("full", r, f);
    arm(d, g_full, full, nullptr, nullptr, warm, meas, step_macs);
    head("split", r, f);
    arm(d, g_split, ss, nullptr, nullptr, warm, meas, step_macs);
    head("split_fill", r, f);
    arm(d, g_split, ss, &f, fs, warm, meas, step_macs);
    if (ctl_pat >= 0) {
      head("split_ctl", r, fc);
      arm(d, g_split, ss, &fc, fs, warm, meas, step_macs);
    }
    head("fill", r, f);
    arm(d, nullptr, nullptr, &f, fs, warm, meas, step_macs);
    head("full2", r, f);
    arm(d, g_full, full, nullptr, nullptr, warm, meas, step_macs);
  }
  CK(cudaGraphExecDestroy(g_full));
  CK(cudaGraphExecDestroy(g_split));
  CK(cudaFree(f.rec));
  if (ctl_pat >= 0) CK(cudaFree(fc.rec));
}

// dcap's device setup: weights only for pfill, KV sized to B·L
static void setup(Dev& d, bool need_w, int B, int L) {
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&d.st));
  CK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, 0));
  if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); exit(2); }
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
  if (nvmlDeviceGetHandleByPciBusId_v2(bus, &d.nv) != NVML_SUCCESS) { fprintf(stderr, "NVML handle failed\n"); exit(2); }
  char uuid[96] = {0}, name[96] = {0};
  nvmlDeviceGetUUID(d.nv, uuid, sizeof uuid);
  nvmlDeviceGetName(d.nv, name, sizeof name);
  const char* want = getenv("GPU_LEASE_UUID");
  if (want && *want && strcmp(want, uuid) != 0) {
    fprintf(stderr, "device %s is not the leased %s\n", uuid, want);
    exit(3);
  }
  fprintf(stderr, "device %s %s, %d SMs\n", name, uuid, d.sms);
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (need_w) {
    for (int l = 0; l < LAYERS; l++)
      for (int j = 0; j < 4; j++) {
        const size_t n = (size_t)LN[j] * LK[j];
        CK(cudaMalloc(&d.W[l][j], n));
        fill_s8<<<1024, 256, 0, d.st>>>(d.W[l][j], n, 0x1000 + l * 4 + j);
      }
    CK(cudaMalloc(&d.Wlm, (size_t)VOCAB * HID));
    fill_s8<<<1024, 256, 0, d.st>>>(d.Wlm, (size_t)VOCAB * HID, 0x77);
  }
  const size_t maxBL = 786432;
  CK(cudaMalloc(&d.X, (size_t)8192 * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(d.X, (size_t)8192 * FF, 0x55);
  CK(cudaMalloc(&d.D, (size_t)1 << 30));
  d.kv_bytes = (size_t)1 << 20;
  if (B > 0) {
    size_t fr, tot;
    CK(cudaMemGetInfo(&fr, &tot));
    d.kv_bytes = std::min({(size_t)48 << 30, fr - ((size_t)3 << 30), (size_t)B * L * KV_TOKEN_BYTES + ((size_t)1 << 20)});
  }
  CK(cudaMalloc(&d.kv, d.kv_bytes));
  fill_s8<<<4096, 256, 0, d.st>>>(d.kv, d.kv_bytes, 0x99);
  CK(cudaMalloc(&d.q, (size_t)256 * HQ * HD));
  fill_s8<<<64, 256, 0, d.st>>>(d.q, (size_t)256 * HQ * HD, 0x31);
  CK(cudaMalloc(&d.P, maxBL * HQ));
  fill_s8<<<1024, 256, 0, d.st>>>(d.P, maxBL * HQ, 0x32);
  CK(cudaMalloc(&d.S, maxBL * HQ * sizeof(int)));
  CK(cudaMalloc(&d.O, (size_t)256 * HQ * HD * sizeof(int)));
  CK(cudaMalloc(&d.sink, 64));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  for (auto fn : {(const void*)pfill_k<0, 1>, (const void*)pfill_k<0, 2>, (const void*)pfill_k<0, 4>,
                  (const void*)pfill_k<1, 1>, (const void*)pfill_k<1, 2>, (const void*)pfill_k<1, 4>,
                  (const void*)pfill_k<2, 1>, (const void*)pfill_k<2, 2>, (const void*)pfill_k<2, 4>})
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: pwr pgate | pidle SECS | palone FILL_SMS KIND PAT W NC WARM MEAS\n"
                    "       pwr pfill B L FILL_SMS KIND PAT W NC WARM MEAS ROUNDS [CTL_PAT]\n"
                    "       (DFILL_COINS=<32 hex digits>: the round's coins; PF_SLEEP_NS: sleep a tile)\n");
    return 1;
  }
  const std::string mode = argv[1];
  if (const char* cs = getenv("DFILL_COINS")) {
    if (strlen(cs) != 32 || sscanf(cs, "%16llx%16llx", (unsigned long long*)&g_c0, (unsigned long long*)&g_c1) != 2) {
      fprintf(stderr, "DFILL_COINS must be 32 hex digits\n");
      return 1;
    }
  }
  Dev d{};
  int B = 0, L = 0;
  if (mode == "pfill") {
    if (argc < 12) { fprintf(stderr, "pfill B L FILL_SMS KIND PAT W NC WARM MEAS ROUNDS [CTL_PAT]\n"); return 1; }
    B = atoi(argv[2]);
    L = atoi(argv[3]);
    if ((size_t)B * L > 786432 || L % 512 || B > 256 || B < 1) { fprintf(stderr, "B*L <= 786432, L %% 512 == 0, B <= 256\n"); return 1; }
  }
  setup(d, mode == "pfill", B, L);
  g_samp.start(d.nv);
  int rc = 0;
  if (mode == "pgate") rc = pgate();
  else if (mode == "pidle") run_pidle(d, argc > 2 ? atof(argv[2]) : 3);
  else if (mode == "palone") {
    if (argc < 9) { fprintf(stderr, "palone FILL_SMS KIND PAT W NC WARM MEAS\n"); rc = 1; }
    else run_palone(d, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atof(argv[7]), atof(argv[8]));
  } else if (mode == "pfill") {
    const int fsm = atoi(argv[4]);
    if (fsm <= 0 || fsm >= d.sms) { fprintf(stderr, "0 < FILL_SMS < %d\n", d.sms); rc = 1; }
    else run_pfill(d, B, L, fsm, atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), atof(argv[9]), atof(argv[10]),
                   atoi(argv[11]), argc > 12 ? atoi(argv[12]) : -1);
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    rc = 1;
  }
  g_samp.stop();
  return rc;
}
