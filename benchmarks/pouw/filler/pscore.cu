// The scheduled arm's driver for the filler scorer: dscore.cu's `dscore` protocol and
// row, with one addition to the arm block, the phase filler (dcap.cu's dfill_p) forked into each layer's attention on the
// step's own SMs beside FILL v1's split. filler-arm/v1's placements (none | split | co) can't name it, so this file and
// psched_score.py carry it until the scorer adopts it; its rows are dscore rows, which the locked `score.py report` scores.
//
// Modes (one JSON line each; DFILL_COINS = the round's coins, DFILL_CHUNK_LOG2 = the arm's chunk size, as dscore):
//   pscore B L FILL_SMS W NC S PW PNC PS PT ROUNDS K POINT
//        the decode step at batch B, context L as `dscore dscore step B L split FILL_SMS W NC S ROUNDS K POINT` (ROUNDS rounds in
//        ABBA order of K reps of serving alone on the whole GPU and K reps under the arm), the arm's step also forking PT
//        tiles a warp of dfill_p<PNC> (PW warps a block, one block a step SM, PS steps a tile) off each layer's qkv onto a
//        second stream of the step's green context, at the highest node priority, joined before o. PT is the arm's, chosen
//        before the run. An arm block's "chunks" is its filler work in FILL chunks: the split's chunks seen done inside the
//        block's window + the phase tiles of its K timed steps x tile MACs / chunk MACs; the device counts every tile, and
//        the count must equal K + 2 steps' worth (time_work's 2 warm-ups run the graph too). "alone_fill" is both fillers
//        alone on their own SMs: the split's (dfill_alone) + the phase filler's on the step's SMs.
//        Receipts: each arm block poisons both receipt buffers; 4 of the split's last chunk, and of the phase filler's last
//        step every slot's tile id and 8 receipts, are checked on the host.
//        filler-scorer/v2's row (rounds.cuh): each filler's tiles a round come from the scorer's range for its slot, with
//        the round's own coins (the phase filler's ids carry PBIT, printed as its rounds' tag); an arm block's split work
//        is its completions over their span and the phase filler's is `fixed_macs` over the window (`fixed_ms`); each
//        round's standalone block runs both fillers alone on their own SMs (the split pumped for the arm's window, the
//        phase filler's launch timed 5 times on the step's SMs meanwhile); then the settled power windows. Slots: 0 the
//        split's calibration, 1 the phase filler's, 2 + r and 2 + R + r round r's arm block, 2 + 2R + r and 2 + 3R + r
//        its standalone block, then four for the power windows (4R + 6 <= rounds.cuh's POINT_SLOTS).
//   pncu PW PNC PS PT
//        dfill_p on every SM, 4 launches of PT tiles a warp: ncu reads the phase filler's DRAM bytes from these.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a pscore.cu -lcublasLt -lnvidia-ml -ldl
#define main dcap_main
#include "dcap.cu"
#undef main
#include "rounds.cuh"

constexpr size_t MAX_BL = 786432;  // dcap's bound on batch x context

// dscore.cu's setup
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

static void pfill_prep() {
  CK(cudaFuncSetAttribute(dfill_p<1>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_p<2>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaFuncSetAttribute(dfill_p<4>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
}

// the phase filler's last step: every slot holds the tile id that step owed it, and `n` receipts, spread, equal the host's
static int pfill_last(const PFill& f, int n, int* checked, std::vector<unsigned long long>* ids) {
  const size_t slots = (size_t)LAYERS * f.per_layer();
  std::vector<unsigned long long> h(2 * slots);
  unsigned long long st[2];
  CK(cudaMemcpy(st, f.st, sizeof st, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
  const unsigned long long base = st[0] - slots;
  int bad = 0;
  for (size_t s = 0; s < slots; s++) bad += h[2 * s] != (PBIT | (base + s));
  for (int i = 0; i < n; i++) {
    const size_t s = ((size_t)i * slots / n + ((size_t)i * 7919) % std::max((size_t)1, slots / n)) % slots;
    bad += h[2 * s + 1] != dfill_ref(f.c0, f.c1, PBIT | (base + s), f.NC, f.S);
    ids->push_back(base + s);
    ++*checked;
  }
  return bad;
}
// the phase filler's last lone launch (layer 0's slots): every slot holds its tile id, and `n` receipts equal the host's
static int pfill_lone(const PFill& f, int n, int* checked, std::vector<unsigned long long>* ids) {
  const size_t slots = (size_t)f.per_layer();
  std::vector<unsigned long long> h(2 * slots);
  unsigned long long st[2];
  CK(cudaMemcpy(st, f.st, sizeof st, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.data(), f.rec, h.size() * sizeof h[0], cudaMemcpyDeviceToHost));
  const unsigned long long base = st[0] - slots;
  int bad = 0;
  for (size_t s = 0; s < slots; s++) bad += h[2 * s] != (PBIT | (base + s));
  for (int i = 0; i < n; i++) {
    const size_t s = ((size_t)i * slots / n + ((size_t)i * 7919) % std::max((size_t)1, slots / n)) % slots;
    bad += h[2 * s + 1] != dfill_ref(f.c0, f.c1, PBIT | (base + s), f.NC, f.S);
    ids->push_back(base + s);
    ++*checked;
  }
  return bad;
}
// a phase round: its coins on the filler, its range's start as the device's tile base
static Round phase_open(PFill& pf, int point, int slot) {
  Round r = round_open(point, slot, PBIT);
  pf.c0 = r.c0;
  pf.c1 = r.c1;
  const unsigned long long st[2] = {r.lo, 0};
  CK(cudaMemcpy(pf.st, st, sizeof st, cudaMemcpyHostToDevice));
  return r;
}
static unsigned long long pbase(const PFill& f) {
  unsigned long long h[2];
  CK(cudaMemcpy(h, f.st, sizeof h, cudaMemcpyDeviceToHost));
  return h[0];
}

struct Block {
  std::vector<Rep> reps;
  double t0 = 0, t1 = 0, span = 0, fixed_ms = 0;
  long long chunks = 0;
  unsigned long long phase_tiles = 0;
  double fixed_macs = 0;
  Round rd, prd;
};
static void print_block(const char* name, const Block& b, bool fill, bool comma) {
  printf("\"%s\": {\"window_ms\": %.3f, ", name, b.t1 - b.t0);
  if (fill) {
    printf("\"chunks\": %lld, \"span_ms\": %.4f, \"phase_tiles\": %llu, \"fixed_macs\": %.6e, \"fixed_ms\": %.4f, ",
           b.chunks, b.span, b.phase_tiles, b.fixed_macs, b.fixed_ms);
    print_round(b.rd);
    printf(", ");
    print_round(b.prd, "phase_round");
    printf(", ");
  }
  printf("\"reps\": [");
  for (size_t i = 0; i < b.reps.size(); i++)
    printf("%s[%.4f, %.0f, %.1f]", i ? ", " : "", b.reps[i].ms, b.reps[i].mhz, b.reps[i].watt);
  printf("]}%s", comma ? ", " : "");
}

static int run_pscore(Dev& d, int B, int L, int fill_sms, int W, int NC, int S, int PW, int PNC, int PS, int PT, int rounds,
                      int K, int point) {
  dfill_prep();
  pfill_prep();
  const std::vector<Lt> p = lin_plans(d, B);
  bool wrapped = false;
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  const cudaStream_t full = d.st;
  cudaStream_t ss = full, ss2 = nullptr, fs = nullptr;
  int got[2] = {d.sms, 0};
  const double ts0 = now_ms();
  const int e = gc_split2(d.sms - fill_sms, &ss, &ss2, &fs, got);
  if (e || !fs) {
    fprintf(stderr, "gc_split2(%d): %d (filler SMs %d)\n", d.sms - fill_sms, e, got[1]);
    return 7;
  }
  d.st = ss;
  const std::vector<Lt> p2 = lin_plans(d, B);
  d.st = full;
  const double setup_ms = now_ms() - ts0;
  const int sp_step = sm_spread(d, ss, true), sp_step2 = sm_spread(d, ss2, true);
  DFill f = dfill_make(got[1], W, NC, S, 0, 0);
  const int sp_fill = sm_spread(d, fs, false);
  int bad = 0, checked = 0;
  double amhz = 0, awatt = 0, alone_split = 0;
  Round r0 = round_open(f, point, 0);
  {
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
    alone_split = (double)(pump.t.size() - 2) * f.macs() / ((pump.t.back() - pump.t[1]) * 1e-3);
    bad += round_close(r0, f, pump, 4);
    checked += 4;
  }
  PFill pf;
  pf.mode = 1; pf.W = PW; pf.NC = PNC; pf.S = PS; pf.tiles = PT; pf.grid = got[0];
  pf.prio = hi; pf.join = true;
  pfill_alloc(pf);
  // the phase filler alone on the step's SMs: one layer's launch, advancing its tile base
  auto w_lone = [&] {
    plaunch(pf, 0, pf.tiles, nullptr, d.st);
    padv_k<<<1, 1, 0, d.st>>>(pf.st, (unsigned long long)pf.per_layer());
  };
  double alone_phase = 0, pmhz = 0;
  Round rp0 = phase_open(pf, point, 1);
  {
    d.st = ss;
    const Rep m = median(time_work(d, w_lone, 5));
    d.st = full;
    alone_phase = (double)pf.per_layer() * pf.tile_macs() / (m.ms * 1e-3);
    pmhz = m.mhz;
    round_end(rp0, pbase(pf));
    bad += pfill_lone(pf, 8, &checked, &rp0.checked);
  }
  cudaEvent_t ev[2];
  CK(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming));
  const long long per_step = (long long)LAYERS * pf.per_layer();
  const size_t pnb = 2 * (size_t)per_step * sizeof(unsigned long long);
  auto w_full = [&] { enqueue_step(d, p, 2, B, L, &wrapped); };
  auto w_arm = [&] { enqueue_step_phase(d, p2, B, L, pf, ss2, ev, &wrapped); };
  long long tiles_short = 0;
  auto alone_block = [&](Block& b) {
    d.st = full;
    b.reps = time_work(d, w_full, K, &b.t0, &b.t1);
  };
  auto poison = [&] {
    CK(cudaMemsetAsync(f.rec, 0xA5, f.per() * sizeof(unsigned long long), fs));
    CK(cudaStreamSynchronize(fs));
    CK(cudaMemsetAsync(pf.rec, 0xA5, pnb, ss));
    CK(cudaStreamSynchronize(ss));
  };
  auto arm_block = [&](Block& b, int r) {
    b.rd = round_open(f, point, 2 + r);
    b.prd = phase_open(pf, point, 2 + rounds + r);
    poison();
    const unsigned long long p0 = pdone(pf);
    DPump pump;
    spump_start(pump, f, fs, 3, b.rd.lo);
    while (pump.done.load() < 2) {
    }
    d.st = ss;
    g_inst_flags = cudaGraphInstantiateFlagUseNodePriority;
    b.reps = time_work(d, w_arm, K, &b.t0, &b.t1);
    g_inst_flags = 0;
    dpump_stop(pump);
    d.st = full;
    count_done(pump, b.t0, b.t1, &b.chunks, &b.span);
    tiles_short += (long long)(K + 2) * per_step - (long long)(pdone(pf) - p0);
    b.phase_tiles = (unsigned long long)K * per_step;
    b.fixed_macs = (double)b.phase_tiles * pf.tile_macs();
    b.fixed_ms = b.t1 - b.t0;
    bad += round_close(b.rd, f, pump, 4);
    checked += 4;
    round_end(b.prd, pbase(pf));
    bad += pfill_last(pf, 8, &checked, &b.prd.checked);
  };
  auto solo_block = [&](Block& b, int r, double win) {
    b.rd = round_open(f, point, 2 + 2 * rounds + r);
    b.prd = phase_open(pf, point, 2 + 3 * rounds + r);
    poison();
    DPump pump;
    spump_start(pump, f, fs, 3, b.rd.lo);
    while (pump.done.load() < 2) {
    }
    b.t0 = now_ms();
    d.st = ss;
    b.reps = time_work(d, w_lone, 5);
    d.st = full;
    while (now_ms() < b.t0 + win || pump.done.load() < 2 + SOLO_CHUNKS) {
    }
    b.t1 = now_ms();
    dpump_stop(pump);
    count_done(pump, b.t0, b.t1, &b.chunks, &b.span);
    b.phase_tiles = 5ull * pf.per_layer();
    b.fixed_macs = (double)b.phase_tiles * pf.tile_macs();
    for (const Rep& x : b.reps) b.fixed_ms += x.ms;
    bad += round_close(b.rd, f, pump, 4);
    checked += 4;
    round_end(b.prd, pbase(pf));
    bad += pfill_lone(pf, 8, &checked, &b.prd.checked);
  };
  std::vector<Block> a(rounds), m(rounds), so(rounds);
  for (int r = 0; r < rounds; r++) {
    if (r % 2 == 0) alone_block(a[r]);
    arm_block(m[r], r);
    solo_block(so[r], r, m[r].t1 - m[r].t0);
    if (r % 2 == 1) alone_block(a[r]);
  }
  Pwr pw_alone, pw_arm, pw_fill;
  {
    cudaGraphExec_t ge = graph_of(d, w_full, full);
    pw_alone = power_window(d, ge, full);
    CK(cudaGraphExecDestroy(ge));
  }
  for (int w = 0; w < 2; w++) {
    Round rd = round_open(f, point, 2 + 4 * rounds + 2 * w);
    Round prd = phase_open(pf, point, 3 + 4 * rounds + 2 * w);
    DPump pump;
    spump_start(pump, f, fs, 3, rd.lo);
    while (pump.done.load() < 2) {
    }
    cudaGraphExec_t ge = w == 0 ? graph_of(d, w_arm, ss, cudaGraphInstantiateFlagUseNodePriority) : graph_of(d, w_lone, ss);
    (w == 0 ? pw_arm : pw_fill) = power_window(d, ge, ss);
    CK(cudaGraphExecDestroy(ge));
    dpump_stop(pump);
    bad += round_close(rd, f, pump, 4);
    checked += 4;
    round_end(prd, pbase(pf));
  }
  d.st = full;
  unsigned smb[8];
  CK(cudaMemcpy(smb, pf.smbits, 32, cudaMemcpyDeviceToHost));
  int psms = 0;
  for (unsigned w : smb) psms += __builtin_popcount(w);
  if (tiles_short) fprintf(stderr, "the phase filler ran %lld tiles fewer than its steps owe\n", tiles_short);
  printf("{\"mode\": \"dscore\", \"driver\": \"pscore\", \"kind\": \"step\", \"B\": %d, \"L\": %d, \"place\": \"split\", "
         "\"fill_sms_req\": %d, \"sms\": %d, \"step_sms\": %d, \"fill_sms\": %d, \"spread\": {\"step_graph\": %d, "
         "\"fill\": %d, \"step2_graph\": %d, \"phase_sms\": %d}, \"W\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %d, "
         "\"grid\": %d, \"chunk_macs\": %.6e, \"fixed_sms\": %d, \"point\": %d, \"slots\": %d, \"cheat_S\": 0, "
         "\"prio\": [%d, %d], \"kv_wrapped\": %s, \"setup_ms\": %.3f, \"rounds\": %d, \"K\": %d, "
         "\"phase\": {\"W\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %d, \"grid\": %d, \"node_prio\": %d, "
         "\"tile_macs\": %.6e, \"tiles_a_step\": %lld, \"tiles_short\": %lld, \"alone_mac_per_s\": %.6e, \"alone_mhz\": %.1f}, "
         "\"alone_fill\": {\"mac_per_s\": %.6e, \"mhz\": %.1f, \"watt\": %.1f, \"split_mac_per_s\": %.6e, "
         "\"phase_mac_per_s\": %.6e, ",
         B, L, fill_sms, d.sms, got[0], got[1], sp_step, sp_fill, sp_step2, psms, W, NC, S, f.tiles, f.grid, f.macs(),
         got[0], point, POINT_SLOTS, lo, hi, wrapped ? "true" : "false", setup_ms, rounds, K, PW, PNC, PS, PT, pf.grid, pf.prio,
         pf.tile_macs(), per_step, tiles_short, alone_phase, pmhz, alone_split + alone_phase, amhz, awatt, alone_split,
         alone_phase);
  print_round(r0);
  printf(", ");
  print_round(rp0, "phase_round");
  printf("}, \"rec_checked\": %d, \"rec_bad\": %d, \"power\": {\"settle_ms\": %.0f, \"meas_ms\": %.0f, ", checked, bad,
         PW_SETTLE_MS, PW_MEAS_MS);
  print_pwr("alone", pw_alone, true);
  print_pwr("arm", pw_arm, true);
  print_pwr("filler_alone", pw_fill, false);
  printf("}, \"algos\": [");
  for (size_t j = 0; j < p.size(); j++) printf("%s[%d, %d]", j ? ", " : "", p[j].algo_id, p2[j].algo_id);
  printf("], \"blocks\": [");
  for (int r = 0; r < rounds; r++) {
    printf("%s{\"order\": \"%s\", ", r ? ", " : "", r % 2 == 0 ? "alone,arm,solo" : "arm,solo,alone");
    print_block("alone", a[r], false, true);
    print_block("arm", m[r], true, true);
    print_block("solo", so[r], true, false);
    printf("}");
  }
  printf("]}\n");
  fflush(stdout);
  pfill_free(pf);
  CK(cudaFree(f.rec));
  return bad || tiles_short ? 8 : 0;
}

static int run_pncu(Dev& d, int PW, int PNC, int PS, int PT) {
  pfill_prep();
  PFill pf;
  pf.mode = 1; pf.W = PW; pf.NC = PNC; pf.S = PS; pf.tiles = PT; pf.grid = d.sms; pf.c0 = g_c0; pf.c1 = g_c1;
  pfill_alloc(pf);
  const double t0 = now_ms();
  for (int i = 0; i < 4; i++) {
    plaunch(pf, 0, pf.tiles, nullptr, d.st);
    padv_k<<<1, 1, 0, d.st>>>(pf.st, (unsigned long long)pf.per_layer());
  }
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  int checked = 0, skipped = 0;
  const int bad = pfill_check(pf, 16, &checked, &skipped);
  printf("{\"mode\": \"pncu\", \"W\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %d, \"grid\": %d, \"launches\": 4, "
         "\"tiles_done\": %llu, \"wall_ms\": %.3f, \"rec_bad\": %d}\n",
         PW, PNC, PS, PT, pf.grid, pdone(pf), now_ms() - t0, bad);
  fflush(stdout);
  pfill_free(pf);
  return bad ? 8 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: pscore pscore B L FILL_SMS W NC S PW PNC PS PT ROUNDS K POINT | pscore pncu PW PNC PS PT\n");
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
  auto phase_ok = [](int PW, int PNC, int PS, int PT) {
    return PW >= 1 && PW <= 16 && (PNC == 1 || PNC == 2 || PNC == 4) && PS % 8 == 0 && PS > 8 && PT >= 1;
  };
  if (mode == "pncu") {
    if (argc < 6) { fprintf(stderr, "pncu PW PNC PS PT\n"); return 1; }
    const int PW = atoi(argv[2]), PNC = atoi(argv[3]), PS = atoi(argv[4]), PT = atoi(argv[5]);
    if (!phase_ok(PW, PNC, PS, PT)) { fprintf(stderr, "1 <= PW <= 16, PNC 1|2|4, 8 | PS > 8, PT >= 1\n"); return 1; }
    if (int rc = setup(d, false, (size_t)1 << 20)) return rc;
    return run_pncu(d, PW, PNC, PS, PT);
  }
  if (mode != "pscore" || argc < 15) {
    fprintf(stderr, "pscore B L FILL_SMS W NC S PW PNC PS PT ROUNDS K POINT\n");
    return 1;
  }
  const int B = atoi(argv[2]), L = atoi(argv[3]), fsm = atoi(argv[4]), W = atoi(argv[5]), NC = atoi(argv[6]),
            S = atoi(argv[7]), PW = atoi(argv[8]), PNC = atoi(argv[9]), PS = atoi(argv[10]), PT = atoi(argv[11]),
            rounds = atoi(argv[12]), K = atoi(argv[13]), point = atoi(argv[14]);
  if ((size_t)B * L > MAX_BL || L % 512 || L <= 0 || B < 1 || B > 256 || rounds < 2 || 4 * rounds + 6 > POINT_SLOTS ||
      K < 1 || W < 1 || W > 16 || point < 0 ||
      S % 8 || S <= 0 || !phase_ok(PW, PNC, PS, PT)) {
    fprintf(stderr, "B*L <= %zu, L %% 512 == 0, 1 <= B <= 256, ROUNDS >= 2, K >= 1, 1 <= W <= 16, 8 | S; "
                    "1 <= PW <= 16, PNC 1|2|4, 8 | PS > 8, PT >= 1\n", MAX_BL);
    return 1;
  }
  if (int rc = setup(d, true, (size_t)B * L * KV_TOKEN_BYTES + ((size_t)1 << 20))) return rc;
  if (fsm <= 0 || fsm >= d.sms) { fprintf(stderr, "0 < FILL_SMS < %d\n", d.sms); return 1; }
  return run_pscore(d, B, L, fsm, W, NC, S, PW, PNC, PS, PT, rounds, K, point);
}
