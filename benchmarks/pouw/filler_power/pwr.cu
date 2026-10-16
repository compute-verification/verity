// The certified filler's board power beside decode (HC-POWER, cmp-13). FILL's DRAM-free chained-MMA filler (dcap.cu's
// dfill_k) and variants of it, alone and beside dcap's decode step on a green-context split, each arm sustained for
// seconds so that NVML's energy counter, instant power, SM clock and clock-event reasons describe the arm rather than the
// idle board around a 40 ms window. Clocks are read, never set.
//
// Modes (one JSON line per arm):
//   pgate                                         the variants' receipts against the host (int8 kinds), KIND 0 PAT 0 against
//                                                 FILL's dfill_ref, and each control
//   pidle SECS                                    the idle board
//   pdone KIND PAT W NC                           three chunks on every SM, for ncu's DRAM counters
//   pbw                                           dcap's bw_read over 1 GiB, three launches: the DRAM counters' positive control
//   palone FILL_SMS KIND PAT W NC WARM MEAS       the filler alone on FILL_SMS SMs (the GPU's count: no split)
//   pfill B L FILL_SMS KIND PAT W NC WARM MEAS ROUNDS [CTL_PAT]
//                                                 per round: the step on all SMs (full), on its partition with cuBLASLt
//                                                 re-tuned there (split), the same beside the filler (split_fill), beside
//                                                 the CTL_PAT filler (split_ctl, when given), the filler alone (fill), and
//                                                 full again (full2); PW_CO=1 adds the step on all SMs beside the filler
//                                                 on all SMs (co_fill), PW_CO=2 also attention alone with and without it
//   pprefill M B L FILL_SMS KIND PAT W NC HEAT MEAS ROUNDS
//                                                 M-row prefill linears on all SMs right after HEAT s of the decode step on
//                                                 its partition, alone or beside the filler (run_pprefill)
//   ppsplit M FILL_SMS KIND PAT W NC WARM MEAS ROUNDS
//                                                 M-row prefill linears on all SMs, on the step partition alone and beside
//                                                 the filler, and on all SMs again (run_ppsplit)
// KIND: 0 int8 m16n8k32 s32 (FILL v1's step), 1 FP8 e4m3 m16n8k32 f32, 2 NVFP4 m16n8k64 f32 (ue4m3 block scales 1.0).
// Each step's D, read as the next A fragment in place (FILL's byte map); FP8 clears bit 3 of every byte (0xF7F7F7F7), so no
// A or B byte is the e4m3 NaN; e2m1 has no NaN. PAT, the operands from the coins: 0 as FILL (B_0, B_1 uniform), 1 B_1 = B_0,
// 2 B's bytes in {0, 1}, 3 B = 0 (a power control, not certified work: D = C every step), 4 B's bytes in [0, 16),
// 5 C in [-2^15, 2^15) (FILL-SEC's optional hardening), 6 A' = (d0, d1, d2, d3) (HC-KERN's map:direct), 7 both 5 and 6.
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
  int csmall;  // C in [-2^15, 2^15) instead of [-2^23, 2^23) (FILL-SEC's optional hardening)
  int map;     // 0: FILL's A' = (d0, d2, d3, d1); 1: A' = (d0, d1, d2, d3) (HC-KERN's map:direct)
};
static PPat ppat(int pat) {
  PPat p{0xffffffffu, 0, 0, 0, 0};
  if (pat == 1) p.bsame = 1;
  if (pat == 2) p.bmask = 0x01010101u;
  if (pat == 3) p.bmask = 0;
  if (pat == 4) p.bmask = 0x0f0f0f0fu;
  if (pat == 5 || pat == 7) p.csmall = 1;
  if (pat == 6 || pat == 7) p.map = 1;
  if (const char* e = getenv("PF_SLEEP_NS")) p.sleep_ns = atoi(e);
  return p;
}
__host__ __device__ inline int32_t pcval(uint64_t seed, int r, int n, int csmall) {
  return csmall ? (int32_t)(dword(seed, 3, r, n) & 0xffffu) - 0x8000 : dcval(seed, r, n);
}
constexpr uint32_t FP8_MASK = 0xF7F7F7F7u;  // clears bit 3 of each byte: never 0x7F or 0xFF, e4m3's NaNs
constexpr uint32_t UE4M3_ONE = 0x38383838u;
template <int KIND, int MAP>
__device__ __forceinline__ void pstep(uint32_t (&a)[4], const uint32_t (&b)[2], const uint32_t (&c)[4]) {
  if constexpr (KIND == 0 && MAP == 0) {
    dstep<0>(a, b, c);
  } else if constexpr (KIND == 0) {
    uint32_t d0, d1, d2, d3;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
    a[0] = d0;  // A'[g][4t..]       = D[g][2t]
    a[1] = d1;  // A'[g+8][4t..]     = D[g][2t+1]
    a[2] = d2;  // A'[g][16+4t..]    = D[g+8][2t]
    a[3] = d3;  // A'[g+8][16+4t..]  = D[g+8][2t+1]
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
    a[1] = __float_as_uint(MAP ? d1 : d2) & m;
    a[2] = __float_as_uint(MAP ? d2 : d3) & m;
    a[3] = __float_as_uint(MAP ? d3 : d1) & m;
  }
}
// dfill_k with the operand pattern, the kind and the sleep; KIND 0 with pattern 0 is dfill_k step for step
template <int KIND, int NC, int MAP>
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
    const int cv[4] = {pcval(seed, g, 2 * t, pat.csmall), pcval(seed, g, 2 * t + 1, pat.csmall),
                       pcval(seed, g + 8, 2 * t, pat.csmall), pcval(seed, g + 8, 2 * t + 1, pat.csmall)};
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
        for (int ch = 0; ch < NC; ch++) pstep<KIND, MAP>(a[ch], b[u & 1], c);
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
    for (int n = 0; n < 8; n++) C[r][n] = pcval(seed, r, n, p.csmall);
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
            A[g + 8][4 * t + q] = byte(p.map ? D[g][2 * t + 1] : D[g + 8][2 * t], q);
            A[g][16 + 4 * t + q] = byte(p.map ? D[g + 8][2 * t] : D[g + 8][2 * t + 1], q);
            A[g + 8][16 + 4 * t + q] = byte(p.map ? D[g + 8][2 * t + 1] : D[g][2 * t + 1], q);
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
  PPat p{0xffffffffu, 0, 0, 0, 0};
  uint64_t c0 = 0, c1 = 0;
  unsigned long long* rec = nullptr;
  long long per() const { return (long long)grid * W * tiles; }
  double mmas() const { return (double)per() * NC * S; }
  double macs() const { return mmas() * (kind == 2 ? 8192.0 : 4096.0); }
};
template <int KIND, int MAP>
static void pfill_launch_k(const PFill& f, unsigned long long t0, cudaStream_t st) {
  const dim3 gr(f.grid), bl(32 * f.W);
  switch (f.NC) {
    case 1: pfill_k<KIND, 1, MAP><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.p, f.rec); break;
    case 2: pfill_k<KIND, 2, MAP><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.p, f.rec); break;
    case 4: pfill_k<KIND, 4, MAP><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.p, f.rec); break;
    default: fprintf(stderr, "NC must be 1, 2 or 4\n"); exit(1);
  }
}
static void pfill_launch(const PFill& f, long long chunk, cudaStream_t st) {
  const unsigned long long t0 = (unsigned long long)chunk * f.per();
  if (f.p.map) {
    if (f.kind == 0) pfill_launch_k<0, 1>(f, t0, st);
    else if (f.kind == 1) pfill_launch_k<1, 1>(f, t0, st);
    else pfill_launch_k<2, 1>(f, t0, st);
  } else {
    if (f.kind == 0) pfill_launch_k<0, 0>(f, t0, st);
    else if (f.kind == 1) pfill_launch_k<1, 0>(f, t0, st);
    else pfill_launch_k<2, 0>(f, t0, st);
  }
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
         "\"sleep_ns\": %d, \"split_flags\": %d, ",
         f.kind, f.pat, f.W, f.NC, f.S, f.tiles, f.grid, fill_sms, f.p.sleep_ns,
         getenv("PW_SPLIT_FLAGS") ? atoi(getenv("PW_SPLIT_FLAGS")) : 0);
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
  for (int pat : {0, 1, 2, 3, 4, 5, 6, 7})
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

// dcap's gc_split with the split's flags (PW_SPLIT_FLAGS; 1 = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, finer
// partitions), and the SM resource's partition size and alignment on stderr
static int gc_split_flags(int n_step, cudaStream_t* s_step, cudaStream_t* s_fill, int got[2], unsigned flags) {
  auto dget = drv<decltype(&cuDeviceGet)>("cuDeviceGet");
  auto res = drv<decltype(&cuDeviceGetDevResource)>("cuDeviceGetDevResource");
  auto split = drv<decltype(&cuDevSmResourceSplitByCount)>("cuDevSmResourceSplitByCount");
  if (!dget || !res || !split) return 1000;
  CUdevice dev;
  CUdevResource all{}, part{}, rest{};
  unsigned nb = 1;
  int e;
  if ((e = dget(&dev, 0))) return 2000 + e;
  if ((e = res(dev, &all, CU_DEV_RESOURCE_TYPE_SM))) return 2000 + e;
  if ((e = split(&part, &nb, &all, &rest, flags, (unsigned)n_step))) return 3000 + e;
  fprintf(stderr, "split flags %u: requested %d, step %u (min %u, align %u), rest %u (min %u, align %u)\n", flags, n_step,
          part.sm.smCount, part.sm.minSmPartitionSize, part.sm.smCoscheduledAlignment, rest.sm.smCount,
          rest.sm.minSmPartitionSize, rest.sm.smCoscheduledAlignment);
  got[0] = (int)part.sm.smCount;
  got[1] = (int)rest.sm.smCount;
  CUstream a = nullptr, b = nullptr;
  if ((e = green_stream(&part, dev, &a))) return e;
  if (rest.sm.smCount && (e = green_stream(&rest, dev, &b))) return e;
  *s_step = (cudaStream_t)a;
  *s_fill = (cudaStream_t)b;
  return 0;
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
  const unsigned flags = getenv("PW_SPLIT_FLAGS") ? (unsigned)atoi(getenv("PW_SPLIT_FLAGS")) : 0u;
  const int e = gc_split_flags(d.sms - fill_sms, ss, &fs, got, flags);
  if (e || !fs) {
    fprintf(stderr, "gc_split(%d): %d\n", d.sms - fill_sms, e);
    exit(7);
  }
  return fs;
}

// three chunks on every SM, for ncu
static int run_pdone(Dev& d, int kind, int pat, int W, int NC) {
  PFill f = pfill_make(d.sms, kind, pat, W, NC, 2048);
  for (int k = 0; k < 3; k++) pfill_launch(f, k, d.st);
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  const int bad = pfill_check(f, 2, 4);
  printf("{\"mode\": \"pdone\", \"kind\": %d, \"pat\": %d, \"W\": %d, \"NC\": %d, \"chunk_mmas\": %.6e, \"rec_bad\": %d}\n", kind,
         pat, W, NC, f.mmas(), bad);
  CK(cudaFree(f.rec));
  return bad > 0;
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
  // PW_CO=1: an arm with the step on all SMs at the highest stream priority beside the filler on all SMs at the lowest
  // (FILL's co-residency), the power ceiling of any schedule that puts filler warps on the step's SMs
  // PW_CO=2: the same beside decode's attention alone (the step's attention kernels, no linears): the attention phase of
  // a schedule that fills every SM while attention runs (HC-SCHED's phase filler)
  const int co_mode = getenv("PW_CO") ? atoi(getenv("PW_CO")) : 0;
  PFill fco;
  PFill* co = nullptr;
  cudaStream_t co_step = nullptr, co_fs = nullptr;
  if (co_mode) {
    int lo, hi;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CK(cudaStreamCreateWithPriority(&co_step, cudaStreamNonBlocking, hi));
    CK(cudaStreamCreateWithPriority(&co_fs, cudaStreamNonBlocking, lo));
    fco = pfill_make(d.sms, kind, pat, W, NC, 2048);
    co = &fco;
  }
  bool wrapped = false;
  const cudaGraphExec_t g_full = capture(d, full, [&] { enqueue_step(d, p, 2, B, L, &wrapped); });
  const cudaGraphExec_t g_split = capture(d, ss, [&] { enqueue_step(d, p2, 2, B, L, &wrapped); });
  const cudaGraphExec_t g_att = co_mode == 2 ? capture(d, full, [&] { enqueue_step(d, p, 1, B, L, &wrapped); }) : nullptr;
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
    if (co) {
      head("co_fill", r, *co);
      arm(d, g_full, co_step, co, co_fs, warm, meas, step_macs);
    }
    if (g_att) {
      head("att", r, f);
      arm(d, g_att, co_step, nullptr, nullptr, warm, meas, step_macs);
      head("att_co_fill", r, *co);
      arm(d, g_att, co_step, co, co_fs, warm, meas, step_macs);
    }
    head("full2", r, f);
    arm(d, g_full, full, nullptr, nullptr, warm, meas, step_macs);
  }
  CK(cudaGraphExecDestroy(g_full));
  CK(cudaGraphExecDestroy(g_split));
  if (g_att) CK(cudaGraphExecDestroy(g_att));
  CK(cudaFree(f.rec));
  if (ctl_pat >= 0) CK(cudaFree(fc.rec));
  if (co) CK(cudaFree(fco.rec));
}

// Prefill after heat: per round, the decode step on its partition for HEAT s, alone (heat_split) or beside the filler
// (heat_fill), each followed at once by M-row prefill linears on all SMs for MEAS s (pre_after_split, pre_after_fill);
// the two orders alternate by round. Whether the filler's heat makes the power-capped prefill that follows run slower.
static void run_pprefill(Dev& d, int M, int B, int L, int fill_sms, int kind, int pat, int W, int NC, double heat,
                         double meas, int rounds) {
  const std::vector<Lt> pp = lin_plans(d, M);
  cudaStream_t full = d.st, ss = nullptr;
  int got[2];
  cudaStream_t fs = fill_stream(d, fill_sms, &ss, got);
  d.st = ss;
  const std::vector<Lt> p2 = lin_plans(d, B);
  d.st = full;
  PFill f = pfill_make(got[1], kind, pat, W, NC, 2048);
  bool wrapped = false;
  const cudaGraphExec_t g_pre = capture(d, full, [&] { enqueue_step(d, pp, 0, M, 0, &wrapped); });
  const cudaGraphExec_t g_split = capture(d, ss, [&] { enqueue_step(d, p2, 2, B, L, &wrapped); });
  d.st = full;
  double step_macs = 0, pre_macs = 0;
  for (int j = 0; j < 4; j++) {
    step_macs += (double)LAYERS * B * LN[j] * LK[j];
    pre_macs += (double)LAYERS * M * LN[j] * LK[j];
  }
  step_macs += (double)B * VOCAB * HID + 2.0 * LAYERS * B * HQ * HD * (double)L;
  auto head = [&](const char* name, int r) {
    print_head("pprefill", name, r);
    printf("\"M\": %d, \"B\": %d, \"L\": %d, \"step_sms\": %d, \"pre_macs\": %.6e, ", M, B, L, got[0], pre_macs);
    print_fill(f, got[1]);
  };
  for (int r = 0; r < rounds; r++)
    for (int c = 0; c < 2; c++) {
      const bool with = (c + r) % 2 == 1;
      head(with ? "heat_fill" : "heat_split", r);
      arm(d, g_split, ss, with ? &f : nullptr, fs, 0, heat, step_macs);
      head(with ? "pre_after_fill" : "pre_after_split", r);
      arm(d, g_pre, full, nullptr, nullptr, 0, meas, pre_macs);
    }
  CK(cudaGraphExecDestroy(g_pre));
  CK(cudaGraphExecDestroy(g_split));
  CK(cudaFree(f.rec));
}

// Prefill on a split: per round, M-row prefill linears on all SMs (pre_full), on the step partition with cuBLASLt
// re-tuned on it, alone (pre_split) and beside the filler on the rest (pre_split_fill), then on all SMs again (pre_full2).
// split ÷ full is the SM share's cost, split_fill ÷ split the filler's power at the cap.
static void run_ppsplit(Dev& d, int M, int fill_sms, int kind, int pat, int W, int NC, double warm, double meas,
                        int rounds) {
  const std::vector<Lt> pp = lin_plans(d, M);
  cudaStream_t full = d.st, ss = nullptr;
  int got[2];
  cudaStream_t fs = fill_stream(d, fill_sms, &ss, got);
  d.st = ss;
  const std::vector<Lt> p2 = lin_plans(d, M);
  d.st = full;
  PFill f = pfill_make(got[1], kind, pat, W, NC, 2048);
  bool wrapped = false;
  const cudaGraphExec_t g_full = capture(d, full, [&] { enqueue_step(d, pp, 0, M, 0, &wrapped); });
  const cudaGraphExec_t g_split = capture(d, ss, [&] { enqueue_step(d, p2, 0, M, 0, &wrapped); });
  d.st = full;
  double pre_macs = 0;
  for (int j = 0; j < 4; j++) pre_macs += (double)LAYERS * M * LN[j] * LK[j];
  auto head = [&](const char* name, int r) {
    print_head("ppsplit", name, r);
    printf("\"M\": %d, \"step_sms\": %d, \"pre_macs\": %.6e, ", M, got[0], pre_macs);
    print_fill(f, got[1]);
  };
  for (int r = 0; r < rounds; r++) {
    head("pre_full", r);
    arm(d, g_full, full, nullptr, nullptr, warm, meas, pre_macs);
    head("pre_split", r);
    arm(d, g_split, ss, nullptr, nullptr, warm, meas, pre_macs);
    head("pre_split_fill", r);
    arm(d, g_split, ss, &f, fs, warm, meas, pre_macs);
    head("pre_full2", r);
    arm(d, g_full, full, nullptr, nullptr, warm, meas, pre_macs);
  }
  CK(cudaGraphExecDestroy(g_full));
  CK(cudaGraphExecDestroy(g_split));
  CK(cudaFree(f.rec));
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
  CK(cudaMalloc(&g_ref, (size_t)256 * VOCAB * sizeof(int)));  // lt_make checks each candidate algorithm against these
  CK(cudaMalloc(&g_cnt, sizeof *g_cnt));
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
  d.kv_bytes = B < 0 ? ((size_t)1 << 30) + ((size_t)1 << 20) : (size_t)1 << 20;  // B < 0: pbw's 1 GiB stream
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
  for (auto fn : {(const void*)pfill_k<0, 1, 0>, (const void*)pfill_k<0, 2, 0>, (const void*)pfill_k<0, 4, 0>,
                  (const void*)pfill_k<1, 1, 0>, (const void*)pfill_k<1, 2, 0>, (const void*)pfill_k<1, 4, 0>,
                  (const void*)pfill_k<2, 1, 0>, (const void*)pfill_k<2, 2, 0>, (const void*)pfill_k<2, 4, 0>,
                  (const void*)pfill_k<0, 1, 1>, (const void*)pfill_k<0, 2, 1>, (const void*)pfill_k<0, 4, 1>,
                  (const void*)pfill_k<1, 1, 1>, (const void*)pfill_k<1, 2, 1>, (const void*)pfill_k<1, 4, 1>,
                  (const void*)pfill_k<2, 1, 1>, (const void*)pfill_k<2, 2, 1>, (const void*)pfill_k<2, 4, 1>})
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: pwr pgate | pidle SECS | palone FILL_SMS KIND PAT W NC WARM MEAS\n"
                    "       pwr pfill B L FILL_SMS KIND PAT W NC WARM MEAS ROUNDS [CTL_PAT]\n"
                    "       pwr pprefill M B L FILL_SMS KIND PAT W NC HEAT MEAS ROUNDS\n"
                    "       pwr ppsplit M FILL_SMS KIND PAT W NC WARM MEAS ROUNDS\n"
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
  if (mode == "pfill" || mode == "pprefill") {
    const int o = mode == "pprefill";
    if (argc < 12 + o) {
      fprintf(stderr, "pfill B L FILL_SMS KIND PAT W NC WARM MEAS ROUNDS [CTL_PAT]\n"
                      "pprefill M B L FILL_SMS KIND PAT W NC HEAT MEAS ROUNDS\n");
      return 1;
    }
    B = atoi(argv[2 + o]);
    L = atoi(argv[3 + o]);
    if ((size_t)B * L > 786432 || L % 512 || B > 256 || B < 1) { fprintf(stderr, "B*L <= 786432, L %% 512 == 0, B <= 256\n"); return 1; }
    if (o && (atoi(argv[2]) < 1 || atoi(argv[2]) > 8192)) { fprintf(stderr, "1 <= M <= 8192\n"); return 1; }
  }
  if (mode == "pbw") B = -1;
  setup(d, mode == "pfill" || mode == "pprefill" || mode == "ppsplit", B, L);
  g_samp.start(d.nv);
  int rc = 0;
  if (mode == "pgate") rc = pgate();
  else if (mode == "pbw") { g_graph = false; run_bw(d, (size_t)1 << 30, 3); }
  else if (mode == "pidle") run_pidle(d, argc > 2 ? atof(argv[2]) : 3);
  else if (mode == "pdone") {
    if (argc < 6) { fprintf(stderr, "pdone KIND PAT W NC\n"); rc = 1; }
    else rc = run_pdone(d, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  }
  else if (mode == "palone") {
    if (argc < 9) { fprintf(stderr, "palone FILL_SMS KIND PAT W NC WARM MEAS\n"); rc = 1; }
    else run_palone(d, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atof(argv[7]), atof(argv[8]));
  } else if (mode == "pfill") {
    const int fsm = atoi(argv[4]);
    if (fsm <= 0 || fsm >= d.sms) { fprintf(stderr, "0 < FILL_SMS < %d\n", d.sms); rc = 1; }
    else run_pfill(d, B, L, fsm, atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), atof(argv[9]), atof(argv[10]),
                   atoi(argv[11]), argc > 12 ? atoi(argv[12]) : -1);
  } else if (mode == "ppsplit") {
    const int fsm = argc > 9 ? atoi(argv[3]) : 0;
    if (argc < 10 || fsm <= 0 || fsm >= d.sms || atoi(argv[2]) < 1 || atoi(argv[2]) > 8192) {
      fprintf(stderr, "ppsplit M FILL_SMS KIND PAT W NC WARM MEAS ROUNDS (1 <= M <= 8192, 0 < FILL_SMS < %d)\n", d.sms);
      rc = 1;
    } else run_ppsplit(d, atoi(argv[2]), fsm, atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atof(argv[8]),
                       atof(argv[9]), atoi(argv[10]));
  } else if (mode == "pprefill") {
    const int fsm = atoi(argv[5]);
    if (fsm <= 0 || fsm >= d.sms) { fprintf(stderr, "0 < FILL_SMS < %d\n", d.sms); rc = 1; }
    else run_pprefill(d, atoi(argv[2]), B, L, fsm, atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), atoi(argv[9]),
                      atof(argv[10]), atof(argv[11]), atoi(argv[12]));
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    rc = 1;
  }
  g_samp.stop();
  return rc;
}
