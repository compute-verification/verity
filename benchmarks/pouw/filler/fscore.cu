// FILL's arm-side driver for the filler scorer: dscore.cu's `dscore` protocol and
// row, with the arm block's serving step the fused one (dcap.cu's flin.cuh: the decode linears through flin_k, a
// warp-specialized GEMM whose filler warps run the DRAM-free chain while its consumers wait on memory, and each layer's
// attention blocks with filler warps), on the split's step SMs beside FILL v1's pump on the rest. filler-arm/v1's
// placements can't name it, so this file and fused_score.py carry it; its rows are dscore rows, which the locked
// `score.py report` scores.
//
// Modes (one JSON line each; DFILL_COINS = the gates' coins, DFILL_CHUNK_LOG2 = the pump's chunk size, as dscore):
//   fscore B L FILL_SMS W NC S FW FNC AFW ANC APS ALZ LT LM ROUNDS K POINT
//        the decode step at batch B, context L as `dscore dscore step B L split FILL_SMS W NC S ROUNDS K POINT` (ROUNDS
//        rounds in ABBA order of K reps of serving alone, dcap's cuBLASLt step on the whole GPU, and K reps under the arm),
//        the arm's step the fused one (enqueue_fstep): flin at the next tile height with FW filler warps of FNC chains a
//        block, but the linears whose bit is set in LT (1 qkv, 2 o, 4 gate_up, 8 down), which stay on cuBLASLt, and the
//        attention blocks with AFW filler warps of ANC chains (APS 1 persistent on the step's SMs; ALZ their lazy done
//        test); lm_head on cuBLASLt, or with LM 1 through flin. The filler warps stop when their block's serving work is
//        done, so a step's work is what the device counted: every slot's steps, its receipt the chain at that count.
//        Each fused round's tiles come from the scorer's range for its slot (rounds.cuh), tagged FBIT; the device keeps
//        the round's coins and tile base (flin.cuh's FState), so each launch of the captured step takes the next `per`
//        ids, and its receipts a row of a ring of K + 2 steps. An arm block's fixed work is its K timed steps' filler
//        MACs (`fixed_macs`) over the block's window (`fixed_ms`), on the step's SMs (`fixed_sms`); 16 receipts over its
//        K + 2 steps (warm-ups included) are checked against dchain_ref with the steps − 8 control, the device's step count
//        must be K + 2, and the last linear's output (down, when it runs through flin) and lm_head's (with LM 1) on every
//        row against the host's.
//        Any failure counts in rec_bad. Slots: dscore's (0 the pump's calibration, 1 + r and 1 + R + r round r's arm and
//        standalone blocks, 2R + 1 and 2R + 2 the pump's power windows), then 2R + 3 the fused step's eager warm-up,
//        2R + 4 + r round r's fused work, 3R + 4 the fused power window (3R + 5 <= POINT_SLOTS).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a fscore.cu -lcublasLt -lnvidia-ml -ldl
// -DFSCORE_PUMP_ONLY (a pump-only arm, every fused step FW = AFW = 0) builds fscore's flin and attention without filler
// warps, so the binary the scorer reads holds only the pump's filler; dcap's own build keeps them for its gates.
#ifdef FSCORE_PUMP_ONLY
#define FLIN_PUMP_ONLY
#endif
#define DCAP_FLIN
#define main dcap_main
#include "dcap.cu"
#undef main
#include "rounds.cuh"

constexpr size_t MAX_BL = 786432;  // dcap's bound on batch x context
constexpr unsigned long long FBIT = 1ull << 62;

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

struct Block {
  std::vector<Rep> reps;
  double t0 = 0, t1 = 0, span = 0, fixed_ms = 0, fixed_macs = 0;
  long long chunks = 0;
  Round rd, frd;
  FlinTally lin, att;
  bool fused = false;
};
static void print_block(const char* name, const Block& b, bool comma) {
  printf("\"%s\": {\"window_ms\": %.3f, \"chunks\": %lld, \"span_ms\": %.4f, ", name, b.t1 - b.t0, b.chunks, b.span);
  if (b.fused) {
    printf("\"fixed_macs\": %.6e, \"fixed_ms\": %.4f, \"fused_steps\": {\"lin\": %lld, \"att\": %lld}, "
           "\"fused_idle_slots\": %lld, ",
           b.fixed_macs, b.fixed_ms, b.lin.steps, b.att.steps, b.lin.idle + b.att.idle);
    print_round(b.frd, "fused_round");
    printf(", ");
  }
  print_round(b.rd);
  printf(", \"reps\": [");
  for (size_t i = 0; i < b.reps.size(); i++)
    printf("%s[%.4f, %.0f, %.1f]", i ? ", " : "", b.reps[i].ms, b.reps[i].mhz, b.reps[i].watt);
  printf("]}%s", comma ? ", " : "");
}
static void print_alone(const char* name, const Block& b, bool comma) {
  printf("\"%s\": {\"window_ms\": %.3f, \"reps\": [", name, b.t1 - b.t0);
  for (size_t i = 0; i < b.reps.size(); i++)
    printf("%s[%.4f, %.0f, %.1f]", i ? ", " : "", b.reps[i].ms, b.reps[i].mhz, b.reps[i].watt);
  printf("]}%s", comma ? ", " : "");
}

static bool flin_has(int M, int FW, int NC) {
  if (FW == 0) return flin_m_ok(M);
  const std::string s = std::string(", ") + flin_built(M) + ",", t = " " + std::to_string(FW) + ":" + std::to_string(NC) + ",";
  return s.find(t) != std::string::npos;
}

struct FusedCfg {
  int FW, FNC, AFW, ANC, APS, ALZ, LT, LM;
};

static int run_fscore(Dev& d, int B, int L, int fill_sms, int W, int NC, int S, const FusedCfg& c, int rounds, int K,
                      int point) {
  dfill_prep();
  g_lt_mask = c.LT;
  g_fatt_nc = c.ANC;
  g_fatt_per = c.APS;
  g_fatt_lazy = c.ALZ;
  const std::vector<int> seq = fstep_seq();
  const std::vector<Lt> p = lin_plans(d, B);
  bool wrapped = false;
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  const cudaStream_t full = d.st;
  cudaStream_t ss = full, fs = nullptr;
  int got[2] = {d.sms, 0};
  const double ts0 = now_ms();
  const int e = gc_split(d.sms - fill_sms, &ss, &fs, got);
  if (e || !fs) {
    fprintf(stderr, "gc_split(%d): %d (filler SMs %d)\n", d.sms - fill_sms, e, got[1]);
    return 7;
  }
  d.st = ss;
  const std::vector<Lt> p2 = lin_plans(d, B);
  Flin fl;
  flin_prepare(d, fl, flin_m_for(B), true);
  fl.grid = got[0];
  fl.thr = 0;
  fl.stop = g_flin_stop;
  fl.NC = c.FNC;
  fl.FW = c.FW;
  if (c.LM && !seq.empty()) flin_prepare_lm(d, fl);
  Fatt fa;
  fatt_prepare(fa, c.AFW, c.ANC, B, L, c.APS ? got[0] : 0);
  fa.lazy = c.ALZ;
  FRing ring;
  fl.ring = fa.ring = &ring;
  const unsigned long long per = fused_layout(fl, c.FW, fa);
  const bool fused = per != 0;  // with no filler warps (FW = AFW = 0) the pump alone: no fixed work, no fused rounds
  d.st = full;
  const double setup_ms = now_ms() - ts0;
  const int sp_step = sm_spread(d, ss, true);
  DFill f = dfill_make(got[1], W, NC, S, 0, 0);
  const int sp_fill = sm_spread(d, fs, false);
  int bad = 0, checked = 0;
  long long out_bad_total = 0, steps_bad = 0, ctl_equal = 0;
  double amhz = 0, awatt = 0, alone = 0;
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
    alone = (double)(pump.t.size() - 2) * f.macs() / ((pump.t.back() - pump.t[1]) * 1e-3);
    bad += round_close(r0, f, pump, 4);
    checked += 4;
  }
  auto fopen = [&](int slot, unsigned depth) {
    Round r = round_open(point, slot, FBIT);
    fring_open(ring, per, depth, r.c0, r.c1, FBIT | r.lo, ss);
    return r;
  };
  // the fused step's tiles over steps [s0, s1) of the open round: the filler MACs of [m0, s1), n receipts over all
  auto ftally = [&](Round& r, unsigned long long s0, unsigned long long s1, unsigned long long m0, int n, uint64_t salt,
                    FlinTally* tl, FlinTally* ta) {
    std::vector<FillLaunch> ll, la, cl;
    for (unsigned long long s = s0; s < s1; s++) {
      const auto vl = fused_launches(ring, &fl, flin_lin(fl, seq), nullptr, 0, s), va = fused_launches(ring, nullptr, {}, &fa, LAYERS, s);
      cl.insert(cl.end(), vl.begin(), vl.end());
      cl.insert(cl.end(), va.begin(), va.end());
      if (s >= m0) {
        ll.insert(ll.end(), vl.begin(), vl.end());
        la.insert(la.end(), va.begin(), va.end());
      }
    }
    *tl = fill_tally(ring, ll, 0, salt);
    *ta = fill_tally(ring, la, 0, salt);
    const FlinTally ck = fill_tally(ring, cl, n, salt, &r.checked, FBIT);
    bad += ck.bad + ck.ctl_equal;
    ctl_equal += ck.ctl_equal;
    checked += ck.checked;
  };
  // the last linear's output on every row, when down runs through flin (its last launch writes Db[(launches - 1) & 1]),
  // and lm_head's when it does (against a reference computed once)
  const int out_rows = c.LT & 8 || seq.empty() ? 0 : B;
  int* ref_lm = nullptr;
  if (fl.lm) {
    CK(cudaMalloc(&ref_lm, (size_t)B * VOCAB * sizeof(int)));
    ref_rows<<<(unsigned)(((size_t)B * VOCAB + 255) / 256), 256, 0, full>>>(d.X, d.Wlm, ref_lm, B, 1, VOCAB, HID);
  }
  long long lm_bad_total = 0;
  auto out_check = [&]() -> long long {
    unsigned long long nb = 0, nl = 0;
    if (out_rows) {
      const int N = LN[3], Kd = LK[3];
      ref_rows<<<(unsigned)(((size_t)B * N + 255) / 256), 256, 0, full>>>(d.X, d.W[LAYERS - 1][3], g_ref, B, 1, N, Kd);
      CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, full));
      count_neq_rows<<<(unsigned)(((size_t)B * N + 255) / 256), 256, 0, full>>>(fl.Db[(seq.size() - 1) & 1], g_ref, B, 1, N,
                                                                                g_cnt);
      CK(cudaMemcpyAsync(&nb, g_cnt, sizeof nb, cudaMemcpyDeviceToHost, full));
    }
    if (fl.lm) {
      CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, full));
      count_neq_rows<<<(unsigned)(((size_t)B * VOCAB + 255) / 256), 256, 0, full>>>(fl.Dlm, ref_lm, B, 1, VOCAB, g_cnt);
      CK(cudaMemcpyAsync(&nl, g_cnt, sizeof nl, cudaMemcpyDeviceToHost, full));
    }
    CK(cudaStreamSynchronize(full));
    lm_bad_total += (long long)nl;
    return (long long)(nb + nl);
  };
  auto w_full = [&] { enqueue_step(d, p, 2, B, L, &wrapped); };
  auto w_arm = [&] { enqueue_fstep(d, fl, fa, p2, B, L, &wrapped); };
  if (c.AFW && !fatt_with(c.AFW, c.ANC, [](auto) {})) {
    fprintf(stderr, "fatt: FW %d NC %d not built (%s)\n", c.AFW, c.ANC, fatt_built());
    return 1;
  }
  if (!seq.empty() && !flin_has(fl.M, c.FW, c.FNC)) {
    fprintf(stderr, "flin: M %d FW %d NC %d not built (%s)\n", fl.M, c.FW, c.FNC, flin_built(fl.M));
    return 1;
  }
  // the fused step once outside capture (its kernels' attributes), from its own round
  Round rw = fopen(3 + 2 * rounds, 1);
  {
    d.st = ss;
    w_arm();
    CK(cudaStreamSynchronize(ss));
    CK(cudaGetLastError());
    d.st = full;
    const FState st = fring_state(ring);
    steps_bad += st.step != 1;
    round_end(rw, rw.lo + st.step * per);
    FlinTally tl, ta;
    ftally(rw, 0, 1, 0, 8, 0xF5C0 + point, &tl, &ta);
    const long long ob = out_check();
    out_bad_total += ob;
    bad += ob != 0;
  }
  auto alone_block = [&](Block& b) {
    d.st = full;
    b.reps = time_work(d, w_full, K, &b.t0, &b.t1);
  };
  auto arm_block = [&](Block& b, int r) {
    b.rd = round_open(f, point, 1 + r);
    b.frd = fopen(4 + 2 * rounds + r, K + 2);
    b.fused = fused;
    CK(cudaMemsetAsync(f.rec, 0xA5, f.per() * sizeof(unsigned long long), fs));
    CK(cudaStreamSynchronize(fs));
    DPump pump;
    spump_start(pump, f, fs, 3, b.rd.lo);
    while (pump.done.load() < 2) {
    }
    d.st = ss;
    b.reps = time_work(d, w_arm, K, &b.t0, &b.t1);
    dpump_stop(pump);
    d.st = full;
    count_done(pump, b.t0, b.t1, &b.chunks, &b.span);
    bad += round_close(b.rd, f, pump, 4);
    checked += 4;
    const FState st = fring_state(ring);
    steps_bad += st.step != (unsigned long long)K + 2;
    round_end(b.frd, b.frd.lo + st.step * per);
    ftally(b.frd, 0, std::min<unsigned long long>(st.step, K + 2), 2, 16, 0xF5C1 + 64 * point + r, &b.lin, &b.att);
    b.fixed_macs = b.lin.macs + b.att.macs;
    b.fixed_ms = b.t1 - b.t0;
    const long long ob = out_check();
    out_bad_total += ob;
    bad += ob != 0;
  };
  auto solo_block = [&](Block& b, int r, double win) {
    b.rd = round_open(f, point, 1 + rounds + r);
    CK(cudaMemsetAsync(f.rec, 0xA5, f.per() * sizeof(unsigned long long), fs));
    CK(cudaStreamSynchronize(fs));
    DPump pump;
    spump_start(pump, f, fs, 3, b.rd.lo);
    while (pump.done.load() < 2) {
    }
    b.t0 = now_ms();
    while (now_ms() < b.t0 + win || pump.done.load() < 2 + SOLO_CHUNKS) {
    }
    b.t1 = now_ms();
    dpump_stop(pump);
    count_done(pump, b.t0, b.t1, &b.chunks, &b.span);
    bad += round_close(b.rd, f, pump, 4);
    checked += 4;
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
  Round rpw;
  for (int w = 0; w < 2; w++) {
    Round rd = round_open(f, point, 2 * rounds + 1 + w);
    DPump pump;
    spump_start(pump, f, fs, 3, rd.lo);
    while (pump.done.load() < 2) {
    }
    if (w == 0) {
      rpw = fopen(4 + 3 * rounds, K + 2);
      cudaGraphExec_t ge = graph_of(d, w_arm, ss);
      pw_arm = power_window(d, ge, ss);
      CK(cudaGraphExecDestroy(ge));
      round_end(rpw, rpw.lo + fring_state(ring).step * per);
    } else {
      pw_fill = power_window(d, nullptr, nullptr);
    }
    dpump_stop(pump);
    bad += round_close(rd, f, pump, 4);
    checked += 4;
  }
  d.st = full;
  bad += steps_bad != 0;
  double lin_macs = 0;
  for (int j = 0; j < 4; j++) lin_macs += (double)LAYERS * B * LN[j] * LK[j];
  printf("{\"mode\": \"dscore\", \"driver\": \"fscore\", \"kind\": \"step\", \"B\": %d, \"L\": %d, \"place\": \"split\", "
         "\"fill_sms_req\": %d, \"sms\": %d, \"step_sms\": %d, \"fill_sms\": %d, \"spread\": {\"step_graph\": %d, "
         "\"fill\": %d}, \"W\": %d, \"NC\": %d, \"S\": %d, \"tiles\": %d, \"grid\": %d, \"chunk_macs\": %.6e, "
         "\"fixed_sms\": %d, \"point\": %d, \"slots\": %d, \"cheat_S\": 0, \"prio\": [%d, %d], \"kv_wrapped\": %s, "
         "\"setup_ms\": %.3f, \"rounds\": %d, \"K\": %d, ",
         B, L, fill_sms, d.sms, got[0], got[1], sp_step, sp_fill, W, NC, S, f.tiles, f.grid, f.macs(), got[0], point,
         POINT_SLOTS, lo, hi, wrapped ? "true" : "false", setup_ms, rounds, K);
  printf("\"fused\": {\"M\": %d, \"FW\": %d, \"FNC\": %d, \"AFW\": %d, \"ANC\": %d, \"APS\": %d, \"ALZ\": %d, \"LT\": %d, "
         "\"LM\": %d, \"pdl\": %d, \"flin_launches\": %zu, \"att_grid\": [%zu, %zu], \"slots_a_step\": %llu, \"tag\": \"%016llx\", "
         "\"out_rows\": %d, \"out_mismatches\": %lld, \"lm_mismatches\": %lld, \"steps_bad\": %lld, \"ctl_equal\": %lld, "
         "\"lin_macs\": %.6e}, ",
         fl.M, c.FW, c.FNC, c.AFW, c.ANC, c.APS, c.ALZ, c.LT, (int)fl.lm, (int)g_pdl, flin_lin(fl, seq).size(), fa.pg[0],
         fa.pg[1], per, FBIT, out_rows, out_bad_total, lm_bad_total, steps_bad, ctl_equal, lin_macs);
  printf("\"alone_fill\": {\"mac_per_s\": %.6e, \"mhz\": %.1f, \"watt\": %.1f, ", alone, amhz, awatt);
  print_round(r0);
  if (fused) {
    printf(", ");
    print_round(rw, "fused_warm_round");
    printf(", ");
    print_round(rpw, "fused_power_round");
  }
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
    print_alone("alone", a[r], true);
    print_block("arm", m[r], true);
    print_block("solo", so[r], false);
    printf("}");
  }
  printf("]}\n");
  fflush(stdout);
  CK(cudaFree(f.rec));
  if (ref_lm) CK(cudaFree(ref_lm));
  return bad ? 8 : 0;
}

int main(int argc, char** argv) {
  const char* use = "fscore B L FILL_SMS W NC S FW FNC AFW ANC APS ALZ LT LM ROUNDS K POINT";
  if (argc < 2 || std::string(argv[1]) != "fscore" || argc < 19) {
    fprintf(stderr, "usage: fscore %s\n", use);
    return 1;
  }
  g_graph = true;
  g_pdl = !(getenv("FLIN_PDL") && !atoi(getenv("FLIN_PDL")));
  g_flin_stop = getenv("FLIN_STOP") ? atoi(getenv("FLIN_STOP")) : 0;
  if (const char* cs = getenv("DFILL_COINS")) {
    if (strlen(cs) != 32 || sscanf(cs, "%16llx%16llx", (unsigned long long*)&g_c0, (unsigned long long*)&g_c1) != 2) {
      fprintf(stderr, "DFILL_COINS must be 32 hex digits\n");
      return 1;
    }
  }
  int v[17];
  for (int i = 0; i < 17; i++) v[i] = atoi(argv[2 + i]);
  const int B = v[0], L = v[1], fsm = v[2], W = v[3], NC = v[4], S = v[5], rounds = v[14], K = v[15], point = v[16];
  const FusedCfg c{v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13]};
  if ((size_t)B * L > MAX_BL || L % 512 || L <= 0 || B < 1 || B > 256 || rounds < 2 || 3 * rounds + 5 > POINT_SLOTS ||
      K < 1 || W < 1 || W > 16 || S % 8 || S <= 0 || point < 0 || c.FW < 0 || c.AFW < 0 || (c.LT & ~15) ||
      !(c.ALZ == 0 || c.ALZ == 2 || c.ALZ == 4) || (c.APS != 0 && c.APS != 1) || (c.LM != 0 && c.LM != 1)) {
    fprintf(stderr, "%s: B*L <= %zu, L %% 512 == 0, 1 <= B <= 256, ROUNDS >= 2, 3 ROUNDS + 5 <= %d, K >= 1, "
                    "1 <= W <= 16, 8 | S, LT in [0, 16), ALZ 0|2|4, APS and LM 0|1\n", use, MAX_BL, POINT_SLOTS);
    return 1;
  }
  Dev d;
  if (int rc = setup(d, true, (size_t)B * L * KV_TOKEN_BYTES + ((size_t)1 << 20))) return rc;
  if (fsm <= 0 || fsm >= d.sms) { fprintf(stderr, "0 < FILL_SMS < %d\n", d.sms); return 1; }
  return run_fscore(d, B, L, fsm, W, NC, S, c, rounds, K, point);
}
