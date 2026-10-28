// The filler scorer's round protocol and power window (SCORER, cmp-13), shared by dscore.cu and arm-side drivers that
// include dcap.cu (DFill, DPump, Dev, now_ms, nvml_mhz). Include it after dcap.cu.
//
// Anti-replay. Slot s of point POINT is round R = POINT·POINT_SLOTS + s, one count for every driver of a run, so no two
// points' rounds share an id (dscore's slots: 0 the filler-alone calibration, 1 + r round r's arm block, 1 + ROUNDS + r
// its standalone block, then the two power windows'). Its tiles are the public range [R·2^32, (R + 1)·2^32), fixed by the scorer before the round's coins,
// which are sampled from the OS once the range is fixed; every seed is dseed(the round's coins, tag | tile id), tag 0 for
// dfill_k (a filler whose kernel marks its ids with a high bit prints it as the round's tag).
#pragma once
#include <sys/random.h>

constexpr int RANGE_LOG2 = 32;
constexpr int POINT_SLOTS = 256;
constexpr int ALONE_CHUNKS = 200;  // the filler-alone calibration, ~0.2 s at 2^19 MMAs an SM a chunk
constexpr int SOLO_CHUNKS = 16;    // a standalone block's least chunks
struct Round {
  unsigned long long id = 0, lo = 0, hi = 0, end = 0, tag = 0;  // the round, its range [lo, hi), the tiles run [lo, end)
  int slot = 0;
  uint64_t c0 = 0, c1 = 0;
  std::vector<unsigned long long> checked;
};
static Round round_open(int point, int slot, unsigned long long tag = 0) {
  if (slot < 0 || slot >= POINT_SLOTS) {
    fprintf(stderr, "round slot %d outside [0, %d)\n", slot, POINT_SLOTS);
    exit(9);
  }
  Round r;
  r.slot = slot;
  r.tag = tag;
  r.id = (unsigned long long)point * POINT_SLOTS + slot;
  r.lo = r.id << RANGE_LOG2;
  r.hi = (r.id + 1) << RANGE_LOG2;
  uint64_t c[2];
  if (getrandom(c, sizeof c, 0) != (ssize_t)sizeof c) {
    fprintf(stderr, "getrandom failed\n");
    exit(1);
  }
  r.c0 = c[0];
  r.c1 = c[1];
  return r;
}
static Round round_open(DFill& f, int point, int slot) {
  Round r = round_open(point, slot);
  f.c0 = r.c0;
  f.c1 = r.c1;
  return r;
}
// a round whose launched tiles end at `end` (exclusive) fails the run past its range
static void round_end(Round& r, unsigned long long end) {
  r.end = end;
  if (r.end > r.hi) {
    fprintf(stderr, "round %llu ran past its tile range\n", r.id);
    exit(9);
  }
}

// the steps dfill_k runs a tile: f.S, unless the dishonest control (DSCORE_CHEAT_S) has its arm blocks run fewer
static int g_run_s = 0;
static int run_s(const DFill& f) { return g_run_s ? g_run_s : f.S; }
static void slaunch(const DFill& f, unsigned long long t0, cudaStream_t st) {
  const dim3 gr(f.grid), bl(32 * f.W);
  const int s = run_s(f);
  switch (f.NC) {
    case 1: dfill_k<1><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, s, f.rec); break;
    case 2: dfill_k<2><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, s, f.rec); break;
    case 4: dfill_k<4><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, s, f.rec); break;
    default: fprintf(stderr, "NC must be 1, 2 or 4\n"); exit(1);
  }
}
// dcap's dpump_start, with chunk k's tiles numbered from base + k·per
static void spump_start(DPump& p, const DFill& f, cudaStream_t fs, int depth, unsigned long long base) {
  p.on = true;
  p.done = 0;
  p.t.clear();
  p.th = std::thread([&p, &f, fs, depth, base] {
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
      slaunch(f, base + (unsigned long long)k * f.per(), fs);
      CK(cudaEventRecord(ev[slot], fs));
    }
    p.launched = k;
    CK(cudaStreamSynchronize(fs));
    for (auto& e : ev) CK(cudaEventDestroy(e));
  });
}
// the chunks a stopped pump finished inside (t0, t1], and the span from the first to the last
static void count_done(const DPump& p, double t0, double t1, long long* chunks, double* span) {
  double first = 0, last = 0;
  *chunks = 0;
  for (double x : p.t)
    if (x > t0 && x <= t1) {
      if (!(*chunks)++) first = x;
      last = x;
    }
  *span = last - first;
}
// after the round's pump stopped: the tiles it launched, and n receipts of its last chunk against dfill_ref (dcap's
// dfill_check at that chunk's first tile, at the steps the chunk ran); a round that ran past its range fails the run
static int round_close(Round& r, const DFill& f, const DPump& p, int n) {
  round_end(r, r.lo + (unsigned long long)p.launched.load() * f.per());
  const unsigned long long t0 = r.end - f.per();
  std::vector<unsigned long long> h(f.per());
  CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; i++) {
    const long long idx = (long long)i * f.per() / n + (i * 7919) % std::max(1LL, f.per() / n);
    r.checked.push_back(t0 + idx);
    bad += h[idx] != dfill_ref(f.c0, f.c1, t0 + idx, f.NC, run_s(f));
  }
  return bad;
}
static void print_round(const Round& r, const char* key = "round") {
  printf("\"%s\": {\"id\": %llu, \"slot\": %d, ", key, r.id, r.slot);
  if (r.tag) printf("\"tag\": \"%016llx\", ", r.tag);
  printf("\"range\": [%llu, %llu], \"tiles\": [%llu, %llu], \"coins\": [\"%016llx\", \"%016llx\"], \"seed0\": \"%016llx\", "
         "\"checked\": [",
         r.lo, r.hi, r.lo, r.end, (unsigned long long)r.c0, (unsigned long long)r.c1,
         (unsigned long long)dseed(r.c0, r.c1, r.tag | r.lo));
  for (size_t i = 0; i < r.checked.size(); i++) printf("%s%llu", i ? ", " : "", r.checked[i]);
  printf("]}");
}

// Board power over a settled window (HC-POWER, note:20261016T0435Z-draft-hc-power): NVML's power reading lags ~1 s,
// longer than a block's reps, so each point also runs serving alone, the arm and the filler alone sustained, PW_SETTLE_MS
// then PW_MEAS_MS with the energy counter sampled every 5 ms. The power is the counter's rise between the first sample at
// which it moved and the first at its last value, over their time (it advances in steps, so a window's ends are steps).
constexpr double PW_SETTLE_MS = 1000, PW_MEAS_MS = 1500;
struct Pwr {
  double watt = -1, mhz = -1, ms = 0;
  long samples = 0;
};
static cudaGraphExec_t graph_of(Dev& d, const std::function<void()>& work, cudaStream_t s, unsigned long long flags = 0) {
  const cudaStream_t keep = d.st;
  d.st = s;
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  work();
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, flags));
  CK(cudaGraphDestroy(g));
  d.st = keep;
  return ge;
}
// `ge` launched back to back on `s` (nullptr: nothing launched here, a filler's pump runs on its own) for the window
static Pwr power_window(const Dev& d, cudaGraphExec_t ge, cudaStream_t s) {
  const double ta = now_ms() + PW_SETTLE_MS, tb = ta + PW_MEAS_MS;
  std::vector<std::pair<double, unsigned long long>> e;
  double cs = 0;
  long nc = 0;
  std::thread smp([&] {
    while (now_ms() < ta) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    while (now_ms() < tb) {
      unsigned long long mj = 0;
      unsigned c = 0;
      if (nvmlDeviceGetTotalEnergyConsumption(d.nv, &mj) == NVML_SUCCESS) e.push_back({now_ms(), mj});
      if (nvmlDeviceGetClockInfo(d.nv, NVML_CLOCK_SM, &c) == NVML_SUCCESS) { cs += c; nc++; }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });
  if (ge) {
    cudaEvent_t ev[2];
    for (auto& x : ev) CK(cudaEventCreateWithFlags(&x, cudaEventDisableTiming));
    for (long k = 0; now_ms() < tb; k++) {
      if (k >= 2) CK(cudaEventSynchronize(ev[k % 2]));
      CK(cudaGraphLaunch(ge, s));
      CK(cudaEventRecord(ev[k % 2], s));
    }
    CK(cudaStreamSynchronize(s));
    for (auto& x : ev) CK(cudaEventDestroy(x));
  }
  smp.join();
  Pwr p;
  p.samples = (long)e.size();
  p.mhz = nc ? cs / nc : -1;
  size_t j0 = 0, j1 = 0;
  while (j0 < e.size() && e[j0].second == e.front().second) j0++;
  while (j1 < e.size() && e[j1].second != e.back().second) j1++;
  if (j0 < j1 && j1 < e.size()) {
    p.ms = e[j1].first - e[j0].first;
    p.watt = (double)(e[j1].second - e[j0].second) / p.ms;  // mJ / ms
  }
  return p;
}
static void print_pwr(const char* name, const Pwr& p, bool comma) {
  printf("\"%s\": {\"watt\": %.1f, \"mhz\": %.1f, \"ms\": %.1f, \"samples\": %ld}%s", name, p.watt, p.mhz, p.ms, p.samples,
         comma ? ", " : "");
}
