// The filler scorer's device driver (SCORER, cmp-13; note:20261016T0435Z-report-filler-spec). It times CAP's serving
// workload (dcap.cu's decode step and prefill linears, cuBLASLt int8 candidates gated bit for bit before use) beside FILL's
// DRAM-free filler (dcap.cu's dfill_k ABI: DFill, dfill_make, dfill_launch, dfill_check, dfill_ref) under an arm's
// placement, interleaved block by block with serving alone. dcap.cu is included unchanged; its main is renamed away.
//
// Modes (one JSON line each; DFILL_COINS = sgate's coins, DFILL_CHUNK_LOG2 = the arm's chunk size):
//   sgate W NC S   the arm's chunk shape on two blocks: receipts poisoned with 0xA5 and launched must all equal the host's
//                  dfill_ref; poisoned with nothing launched (the no-write control), none may; the previous chunk's
//                  receipts (stale work) and the chain at S - 8 (skipped steps) must differ from the reference everywhere.
//   dscore KIND B L PLACE FILL_SMS W NC S ROUNDS K POINT
//                  KIND step (the decode step at batch B, context L) or prefill (32 layers of linears at M = B rows);
//                  PLACE none | split (a green-context split, the filler on FILL_SMS SMs, cuBLASLt re-tuned on the
//                  serving partition) | co (the filler on every SM at the lowest stream priority, serving at the highest).
//                  ROUNDS rounds in ABBA order, each K reps of serving alone on the whole GPU and K reps under the
//                  placement (with PLACE none, serving alone again: the A/A control). Each placed block poisons the
//                  receipts, keeps the filler queued, counts the chunks finished inside the block's window and the span
//                  from the first to the last (the rate is completions over that span, free of the window's phase), checks 4
//                  receipts of its last chunk on the host. Every rep's time, SM clock and board power are printed.
//                  POINT (the point's index in the scorer's run) fixes each round's public tile range (`Round`).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a dscore.cu -lcublasLt -lnvidia-ml -ldl
#define main dcap_main
#include "../decode_capacity/dcap.cu"
#undef main

#include <sys/random.h>

enum { PL_NONE = 0, PL_SPLIT = 1, PL_CO = 2 };
static const char* PL_NAME[3] = {"none", "split", "co"};
constexpr size_t MAX_BL = 786432;  // dcap's bound on batch x context
constexpr int ALONE_CHUNKS = 200;  // the filler-alone calibration, ~0.2 s at 2^19 MMAs an SM a chunk

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

static int run_sgate(Dev& d, int W, int NC, int S) {
  dfill_prep();
  DFill f = dfill_make(2, W, NC, S, g_c0, g_c1);
  const long long chunk = 7;
  const size_t nb = f.per() * sizeof(unsigned long long);
  std::vector<unsigned long long> h(f.per()), ref(f.per()), skip(f.per());
  for (long long i = 0; i < f.per(); i++) {
    ref[i] = dfill_ref(f.c0, f.c1, (unsigned long long)(chunk * f.per() + i), NC, S);
    skip[i] = dfill_ref(f.c0, f.c1, (unsigned long long)(chunk * f.per() + i), NC, S - 8);
  }
  auto run = [&](long long k) {
    CK(cudaMemset(f.rec, 0xA5, nb));
    if (k >= 0) dfill_launch(f, k, d.st);
    CK(cudaStreamSynchronize(d.st));
    CK(cudaGetLastError());
    CK(cudaMemcpy(h.data(), f.rec, nb, cudaMemcpyDeviceToHost));
  };
  long bad = 0, skip_equal = 0, nowrite_equal = 0, stale_equal = 0;
  run(chunk);
  for (long long i = 0; i < f.per(); i++) {
    bad += h[i] != ref[i];
    skip_equal += h[i] == skip[i];
  }
  run(-1);
  for (long long i = 0; i < f.per(); i++) nowrite_equal += h[i] == ref[i];
  run(chunk - 1);
  for (long long i = 0; i < f.per(); i++) stale_equal += h[i] == ref[i];
  const bool ok = !bad && !skip_equal && !nowrite_equal && !stale_equal;
  printf("{\"mode\": \"sgate\", \"W\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %d, \"grid\": %d, \"chunk_macs\": %.6e, "
         "\"coins\": [\"%016llx\", \"%016llx\"], \"receipts\": %lld, \"mismatches\": %ld, \"nowrite_equal\": %ld, "
         "\"stale_equal\": %ld, \"skip_equal\": %ld, \"ok\": %s}\n",
         W, NC, S, f.tiles, f.grid, f.macs(), (unsigned long long)g_c0, (unsigned long long)g_c1, f.per(), bad,
         nowrite_equal, stale_equal, skip_equal, ok ? "true" : "false");
  fflush(stdout);
  CK(cudaFree(f.rec));
  return ok ? 0 : 5;
}

// Anti-replay. Slot s of point POINT (s = 0 the filler-alone calibration, 1 + r round r's arm block) is round
// R = POINT·(ROUNDS + 1) + s. Its tiles are the public range [R·2^32, (R + 1)·2^32), fixed by the scorer before the
// round's coins, which are drawn from the OS once the range is fixed; every seed is dseed(the round's coins, tile id).
constexpr int RANGE_LOG2 = 32;
struct Round {
  unsigned long long id = 0, lo = 0, hi = 0, end = 0;  // the round, its range [lo, hi), the tiles launched [lo, end)
  uint64_t c0 = 0, c1 = 0;
  std::vector<unsigned long long> checked;
};
static Round round_open(DFill& f, int point, int rounds, int slot) {
  Round r;
  r.id = (unsigned long long)point * (rounds + 1) + slot;
  r.lo = r.id << RANGE_LOG2;
  r.hi = (r.id + 1) << RANGE_LOG2;
  uint64_t c[2];
  if (getrandom(c, sizeof c, 0) != (ssize_t)sizeof c) {
    fprintf(stderr, "getrandom failed\n");
    exit(1);
  }
  f.c0 = r.c0 = c[0];
  f.c1 = r.c1 = c[1];
  return r;
}
static void slaunch(const DFill& f, unsigned long long t0, cudaStream_t st) {
  const dim3 gr(f.grid), bl(32 * f.W);
  switch (f.NC) {
    case 1: dfill_k<1><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 2: dfill_k<2><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
    case 4: dfill_k<4><<<gr, bl, 0, st>>>(f.c0, f.c1, t0, f.tiles, f.S, f.rec); break;
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
// after the round's pump stopped: the tiles it launched, and n receipts of its last chunk against dfill_ref (dcap's
// dfill_check at that chunk's first tile); a round that ran past its range fails the run
static int round_close(Round& r, const DFill& f, const DPump& p, int n) {
  r.end = r.lo + (unsigned long long)p.launched.load() * f.per();
  if (r.end > r.hi) {
    fprintf(stderr, "round %llu ran past its tile range\n", r.id);
    exit(9);
  }
  const unsigned long long t0 = r.end - f.per();
  std::vector<unsigned long long> h(f.per());
  CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; i++) {
    const long long idx = (long long)i * f.per() / n + (i * 7919) % std::max(1LL, f.per() / n);
    r.checked.push_back(t0 + idx);
    bad += h[idx] != dfill_ref(f.c0, f.c1, t0 + idx, f.NC, f.S);
  }
  return bad;
}
static void print_round(const Round& r) {
  printf("\"round\": {\"id\": %llu, \"range\": [%llu, %llu], \"tiles\": [%llu, %llu], \"coins\": [\"%016llx\", "
         "\"%016llx\"], \"seed0\": \"%016llx\", \"checked\": [",
         r.id, r.lo, r.hi, r.lo, r.end, (unsigned long long)r.c0, (unsigned long long)r.c1,
         (unsigned long long)dseed(r.c0, r.c1, r.lo));
  for (size_t i = 0; i < r.checked.size(); i++) printf("%s%llu", i ? ", " : "", r.checked[i]);
  printf("]}");
}

struct Block {
  std::vector<Rep> reps;
  double t0 = 0, t1 = 0, span = 0;  // span: from the first to the last chunk seen done inside (t0, t1]
  long long chunks = 0;
  Round rd;
};
static void print_block(const char* name, const Block& b, bool fill, bool comma) {
  printf("\"%s\": {\"window_ms\": %.3f, ", name, b.t1 - b.t0);
  if (fill) {
    printf("\"chunks\": %lld, \"span_ms\": %.4f, ", b.chunks, b.span);
    print_round(b.rd);
    printf(", ");
  }
  printf("\"reps\": [");
  for (size_t i = 0; i < b.reps.size(); i++)
    printf("%s[%.4f, %.0f, %.1f]", i ? ", " : "", b.reps[i].ms, b.reps[i].mhz, b.reps[i].watt);
  printf("]}%s", comma ? ", " : "");
}

static int run_dscore(Dev& d, int kind, int B, int L, int place, int fill_sms, int W, int NC, int S, int rounds, int K,
                      int point) {
  dfill_prep();
  const std::vector<Lt> p = lin_plans(d, B);
  bool wrapped = false;
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t full = d.st, ss = d.st, fs = nullptr;
  int got[2] = {d.sms, 0};
  std::vector<Lt> p2 = p;
  const double ts0 = now_ms();
  if (place == PL_SPLIT) {
    const int e = gc_split(d.sms - fill_sms, &ss, &fs, got);
    if (e || !fs) {
      fprintf(stderr, "gc_split(%d): %d (filler SMs %d)\n", d.sms - fill_sms, e, got[1]);
      return 7;
    }
    d.st = ss;
    p2 = lin_plans(d, B);
    d.st = full;
  } else if (place == PL_CO) {
    CK(cudaStreamCreateWithPriority(&full, cudaStreamDefault, hi));
    CK(cudaStreamCreateWithPriority(&fs, cudaStreamNonBlocking, lo));
    ss = full;
    got[1] = d.sms;
  }
  const double setup_ms = now_ms() - ts0;
  const bool fill = place != PL_NONE;
  DFill f;
  Round r0;
  double amhz = 0, awatt = 0, alone = 0;
  const int sp_step = sm_spread(d, ss, true);
  int sp_fill = 0, bad = 0, checked = 0;
  if (fill) {
    f = dfill_make(got[1], W, NC, S, 0, 0);
    sp_fill = sm_spread(d, fs, false);
    r0 = round_open(f, point, rounds, 0);
    DPump pump;
    spump_start(pump, f, fs, 3, r0.lo);
    double cs = 0, ps = 0;
    long long np = 0;
    while (pump.done.load() < ALONE_CHUNKS) {
      cs += nvml_mhz(d);
      ps += nvml_w(d);
      np++;
    }
    dpump_stop(pump);
    amhz = np ? cs / np : nvml_mhz(d);
    awatt = np ? ps / np : nvml_w(d);
    alone = (double)(pump.t.size() - 2) * f.macs() / ((pump.t.back() - pump.t[1]) * 1e-3);
    bad += round_close(r0, f, pump, 4);
    checked += 4;
  }
  auto w_full = [&] { enqueue_step(d, p, kind, B, L, &wrapped); };
  auto w_arm = [&] { enqueue_step(d, p2, kind, B, L, &wrapped); };
  auto alone_block = [&](Block& b) {
    d.st = full;
    b.reps = time_work(d, w_full, K, &b.t0, &b.t1);
  };
  auto arm_block = [&](Block& b, int r) {
    if (!fill) {
      alone_block(b);
      return;
    }
    b.rd = round_open(f, point, rounds, 1 + r);
    CK(cudaMemsetAsync(f.rec, 0xA5, f.per() * sizeof(unsigned long long), fs));
    CK(cudaStreamSynchronize(fs));
    DPump pump;
    spump_start(pump, f, fs, 3, b.rd.lo);
    while (pump.done.load() < 2) {
    }
    d.st = ss;
    b.reps = time_work(d, w_arm, K, &b.t0, &b.t1);
    dpump_stop(pump);
    double first = 0, last = 0;
    for (double x : pump.t)
      if (x > b.t0 && x <= b.t1) {
        if (!b.chunks++) first = x;
        last = x;
      }
    b.span = last - first;
    bad += round_close(b.rd, f, pump, 4);
    checked += 4;
  };
  std::vector<Block> a(rounds), m(rounds);
  for (int r = 0; r < rounds; r++) {
    if (r % 2 == 0) {
      alone_block(a[r]);
      arm_block(m[r], r);
    } else {
      arm_block(m[r], r);
      alone_block(a[r]);
    }
  }
  d.st = full;
  printf("{\"mode\": \"dscore\", \"kind\": \"%s\", \"B\": %d, \"L\": %d, \"place\": \"%s\", \"fill_sms_req\": %d, "
         "\"sms\": %d, \"step_sms\": %d, \"fill_sms\": %d, \"spread\": {\"step_graph\": %d, \"fill\": %d}, \"W\": %d, "
         "\"NC\": %d, \"S\": %d, \"tiles\": %d, \"grid\": %d, \"chunk_macs\": %.6e, \"point\": %d, "
         "\"prio\": [%d, %d], \"kv_wrapped\": %s, \"setup_ms\": %.3f, \"rounds\": %d, \"K\": %d, "
         "\"alone_fill\": {\"mac_per_s\": %.6e, \"mhz\": %.1f, \"watt\": %.1f",
         kind == 0 ? "prefill" : "step", B, L, PL_NAME[place], fill_sms, d.sms, fill ? got[0] : d.sms, fill ? got[1] : 0,
         sp_step, sp_fill, W, NC, S, fill ? f.tiles : 0, fill ? f.grid : 0, fill ? f.macs() : 0.0, point, lo, hi,
         wrapped ? "true" : "false", setup_ms, rounds, K, alone, amhz, awatt);
  if (fill) {
    printf(", ");
    print_round(r0);
  }
  printf("}, \"rec_checked\": %d, \"rec_bad\": %d, \"algos\": [", checked, bad);
  for (size_t j = 0; j < p.size(); j++) printf("%s[%d, %d]", j ? ", " : "", p[j].algo_id, p2[j].algo_id);
  printf("], \"blocks\": [");
  for (int r = 0; r < rounds; r++) {
    printf("%s{\"order\": \"%s\", ", r ? ", " : "", r % 2 == 0 ? "alone,arm" : "arm,alone");
    print_block("alone", a[r], false, true);
    print_block("arm", m[r], fill, false);
    printf("}");
  }
  printf("]}\n");
  fflush(stdout);
  if (fill) CK(cudaFree(f.rec));
  return bad ? 8 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: dscore sgate W NC S | dscore dscore step|prefill B L none|split|co FILL_SMS W NC S ROUNDS K POINT\n");
    return 1;
  }
  const std::string mode = argv[1];
  g_graph = !(getenv("DCAP_NOGRAPH") && atoi(getenv("DCAP_NOGRAPH")));
  if (const char* cs = getenv("DFILL_COINS")) {
    if (strlen(cs) != 32 || sscanf(cs, "%16llx%16llx", (unsigned long long*)&g_c0, (unsigned long long*)&g_c1) != 2) {
      fprintf(stderr, "DFILL_COINS must be 32 hex digits\n");
      return 1;
    }
  }
  Dev d;
  if (mode == "sgate") {
    if (argc < 5) { fprintf(stderr, "sgate W NC S\n"); return 1; }
    const int W = atoi(argv[2]), NC = atoi(argv[3]), S = atoi(argv[4]);
    if (W < 1 || W > 16 || S % 8 || S <= 8) { fprintf(stderr, "1 <= W <= 16, S a multiple of 8 above 8\n"); return 1; }
    if (int rc = setup(d, false, (size_t)1 << 20)) return rc;
    return run_sgate(d, W, NC, S);
  }
  if (mode != "dscore" || argc < 13) {
    fprintf(stderr, "dscore step|prefill B L none|split|co FILL_SMS W NC S ROUNDS K POINT\n");
    return 1;
  }
  const std::string k = argv[2], pl = argv[5];
  const int kind = k == "prefill" ? 0 : k == "step" ? 2 : -1;
  const int place = pl == "none" ? PL_NONE : pl == "split" ? PL_SPLIT : pl == "co" ? PL_CO : -1;
  const int B = atoi(argv[3]), L = atoi(argv[4]), fsm = atoi(argv[6]), W = atoi(argv[7]), NC = atoi(argv[8]),
            S = atoi(argv[9]), rounds = atoi(argv[10]), K = atoi(argv[11]), point = atoi(argv[12]);
  if (kind < 0 || place < 0 || rounds < 2 || K < 1 || W < 1 || W > 16 || S % 8 || S <= 0 || point < 0 ||
      (kind == 2 && ((size_t)B * L > MAX_BL || L % 512 || L <= 0 || B < 1 || B > 256)) ||
      (kind == 0 && (B < 1 || B > 8192))) {
    fprintf(stderr, "step: B*L <= %zu, L %% 512 == 0, 1 <= B <= 256; prefill: 1 <= B <= 8192; ROUNDS >= 2, K >= 1, "
                    "1 <= W <= 16, 8 | S, POINT >= 0\n", MAX_BL);
    return 1;
  }
  const size_t kv = kind == 2 ? (size_t)B * L * KV_TOKEN_BYTES + ((size_t)1 << 20) : (size_t)1 << 20;
  if (int rc = setup(d, true, kv)) return rc;
  if (place == PL_SPLIT && (fsm <= 0 || fsm >= d.sms)) { fprintf(stderr, "0 < FILL_SMS < %d\n", d.sms); return 1; }
  return run_dscore(d, kind, B, L, place, fsm, W, NC, S, rounds, K, point);
}
