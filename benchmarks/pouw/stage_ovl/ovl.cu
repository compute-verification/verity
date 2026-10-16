// Deep overlap of per-stage live coins in prefill on one RTX PRO 6000 (sm_120), Llama-3.1-8B, int8 (STAGE-OVL, cmp-13;
// note:20261016T1037Z-draft-stage-ovl). k independent chunks (different sequences of M tokens) in flight, each through its
// own chain over 32 layers x 6 stages: commit the stage's operands (stage.cu's fast path of the hm96-sha512 row-seg
// frame-v3-sha512 tree, which stage.cu's gate holds byte-identical to verity.ml.commitments), wait DRAW_US for the draw,
// then the stage's int8 GEMM. Draw t = (stage s, chunk c) is numbered s * k + c, so T_{t-k} is the same chunk's previous
// stage and RS-D1E r2's release rule (iv_k) holds by each chunk's own order. stage.cu (with dcap.cu and hm96_rows.cu) is
// included unchanged, in namespace sc.
//
// Modes (one JSON line each; clocks are read, never set):
//   gate DIR M...           stage.cu's gate: every commitment file committed into 0xA5-poisoned buffers by both paths equals
//                           Python's leaves, b || c and root, a no-write control doesn't; fused attention and cuBLASLt
//                           against the host
//   ovl DIR M K... [rN]     for each k, the arms below as CUDA graphs timed interleaved rep by rep (after two warm-ups each),
//                           then every hashing arm run once more into 0xA5-poisoned trees, its roots compared with the serial
//                           path's, and nohash run the same way as the control that must differ
//   shathru                 SHA-512 compressions a second on the whole GPU (stage.cu's sha_chain at 4-32 CTAs a SM, by
//                           which rotations run on the FMA pipe; the commitments use -DFR_ROT's)
//   beside M K [rN]         plain's GEMMs for K sequences of M tokens against the same commitments' compressions as one
//                           small resident kernel (32-256 threads a SM): alone, and launched just before or after the GEMMs
// Arms (k chunks of M tokens; the QK^T and P.V stand-ins are stage.cu's, m = M, n = M/2, k = 4096, against a sequence's own
// distinct buffer a layer):
//   plain     the k sequences plain int8 on one stream: the linears over all k * M tokens in row blocks of PLAIN_RB (8192),
//             attention a strided batch over the k sequences. The divisor.
//   chunked   k streams, each one chunk's 192 GEMMs and nothing else
//   eq        k streams at one priority, per stage commit -> draw -> GEMM
//   prio      eq with chunk c's stream at priority class floor(c * levels / k), chunk 0 highest
//   hhi       eq with each chunk's commitments on a stream of its own at the highest priority
//   hprio     prio with the commitments at the highest priority
//   hlo       the commitments at the lowest priority, the GEMMs at the highest
//   rrW       hhi with the GEMMs in one static order (stage-major, chunk-minor), each after the GEMM W places before it
//   nodraw    hhi without the draws; nohash: eq without the commitments; hashonly: the k chunks' commitments alone
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a [-DFR_ROT=BITS] ovl.cu -lcublasLt -lnvidia-ml -ldl
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace sc {
#include "../stage_cost/stage.cu"
}
using namespace sc;

// the draw: one thread waits ns on the global timer and stamps when it started (its commitment done) and ended
__global__ void draw_stamp(unsigned long long ns, unsigned long long* o) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  t = t0;
  while (t - t0 < ns) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  o[0] = t0;
  o[1] = t;
}
__global__ void stamp(unsigned long long* o) {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  *o = t;
}

struct OChunk {
  uint8_t* op[NSTAGE];
  int8_t* A;
  int* D;
  Commit cm[NSTAGE];
  LtCtx lt;
};
struct Sched {
  std::string name;
  bool hash, draw, gemm;
  bool classes;  // chunk c's GEMMs at priority class floor(c * levels / k)
  int hp;        // the commitments' stream: 0 the chunk's own, 1 one of their own at the highest priority, -1 at the lowest
                 // with the GEMMs at the highest
  int rr;        // > 0: the GEMMs in one static order, each after the one rr places before it
};
static Sched sched_of(const std::string& n) {
  if (n == "chunked") return {n, false, false, true, false, 0, 0};
  if (n == "eq") return {n, true, true, true, false, 0, 0};
  if (n == "prio") return {n, true, true, true, true, 0, 0};
  if (n == "hhi") return {n, true, true, true, false, 1, 0};
  if (n == "hprio") return {n, true, true, true, true, 1, 0};
  if (n == "hlo") return {n, true, true, true, false, -1, 0};
  if (n == "nodraw") return {n, true, false, true, false, 1, 0};
  if (n == "nohash") return {n, false, true, true, false, 0, 0};
  if (n == "hashonly") return {n, true, false, false, false, 0, 0};
  if (n.rfind("rr", 0) == 0) return {n, true, true, true, false, 1, atoi(n.c_str() + 2)};
  fprintf(stderr, "unknown arm %s\n", n.c_str());
  exit(1);
}

// events for one capture, destroyed after it
struct EvPool {
  std::vector<cudaEvent_t> ev;
  cudaEvent_t get() {
    cudaEvent_t e;
    CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    ev.push_back(e);
    return e;
  }
  void join(cudaStream_t from, cudaStream_t to) {
    cudaEvent_t e = get();
    CK(cudaEventRecord(e, from));
    CK(cudaStreamWaitEvent(to, e, 0));
  }
  ~EvPool() {
    for (cudaEvent_t e : ev) cudaEventDestroy(e);
  }
};

static size_t mem_used() {
  size_t fr, tot;
  CK(cudaMemGetInfo(&fr, &tot));
  return tot - fr;
}

struct ArmRun {
  std::string name;
  Graph G;
  int k;
  bool hash, stamps;
  double macs;
  std::vector<Rep> reps;
  unsigned long long* st = nullptr;  // k x (2 * 192 + 1) stamps and one start stamp at the end
  std::vector<cudaStream_t> streams;
};

static int run_ovl(Dev& d, const std::string& dir, int M, const std::vector<int>& ks, int reps, const std::vector<std::string>& arm_names) {
  const int draw_us = getenv("DRAW_US") ? atoi(getenv("DRAW_US")) : 300;
  const int RB = getenv("PLAIN_RB") ? atoi(getenv("PLAIN_RB")) : 8192;
  const int K = *std::max_element(ks.begin(), ks.end());
  const int NS = LAYERS * NSTAGE, NSTAMP = 2 * NS + 1;
  std::vector<HostConst> hc;
  for (int t = 0; t < NSTAGE; t++) hc.push_back(load_const(dir, SK[t], M));
  const size_t mem0 = mem_used();
  // each sequence's attention stand-in a layer: (M/2) x 4096, K of them side by side
  const size_t kb_seq = (size_t)(M / 2) * HID;
  std::vector<int8_t*> Kb(LAYERS);
  for (int l = 0; l < LAYERS; l++) {
    CK(cudaMalloc(&Kb[l], kb_seq * K));
    fill_s8<<<1024, 256, 0, d.st>>>(Kb[l], kb_seq * K, 0x8000 + l);
  }
  const size_t mem_kb = mem_used();
  std::vector<OChunk> ch(K);
  for (int c = 0; c < K; c++) {
    for (int t = 0; t < NSTAGE; t++) {
      CK(cudaMalloc(&ch[c].op[t], (size_t)M * SK[t]));
      fill_s8<<<1024, 256, 0, d.st>>>((int8_t*)ch[c].op[t], (size_t)M * SK[t], 0x4000 + 16 * c + t);
    }
    CK(cudaMalloc(&ch[c].A, (size_t)M * FF));
    fill_s8<<<1024, 256, 0, d.st>>>(ch[c].A, (size_t)M * FF, 0x5000 + c);
    CK(cudaMalloc(&ch[c].D, (size_t)M * 2 * FF * sizeof(int)));
    ch[c].lt = lt_ctx();
  }
  CK(cudaStreamSynchronize(d.st));
  const size_t mem_act = mem_used();
  for (int c = 0; c < K; c++)
    for (int t = 0; t < NSTAGE; t++) ch[c].cm[t] = make_commit(hc[t]);
  CK(cudaDeviceSynchronize());
  const size_t mem_cm = mem_used();
  // plain's buffers: every token's A, one output block, one context
  const size_t plain_rows = (size_t)K * M;
  int8_t* Ap;
  int* Dp;
  CK(cudaMalloc(&Ap, plain_rows * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(Ap, plain_rows * FF, 0x5100);
  const size_t dp_bytes = std::max((size_t)std::min<size_t>(plain_rows, RB) * 2 * FF, (size_t)K * M * (M / 2)) * sizeof(int);
  CK(cudaMalloc(&Dp, dp_bytes));
  LtCtx plt = lt_ctx();
  CK(cudaStreamSynchronize(d.st));

  const int GN[NSTAGE] = {QKV, M / 2, M / 2, HID, 2 * FF, HID}, GK[NSTAGE] = {HID, HID, HID, HID, HID, FF};
  const int LJ[NSTAGE] = {0, -1, -1, 1, 2, 3};  // the stage's weight in d.W, or -1 for attention
  auto wt = [&](int l, int t, int c) -> const int8_t* {
    if (LJ[t] >= 0) return d.W[l][LJ[t]];
    return Kb[t == 1 ? l : (l + 1) % LAYERS] + kb_seq * c;
  };
  LtB g[NSTAGE];
  for (int t = 0; t < NSTAGE; t++) g[t] = plan(M, GN[t], GK[t], 1, 0, 0, 0, ch[0].A, wt(0, t, 0), ch[0].D, d.st);
  double macs_chunk = 0, gemm_alone_ms = 0;
  for (int t = 0; t < NSTAGE; t++) {
    macs_chunk += (double)LAYERS * M * GN[t] * GK[t];
    gemm_alone_ms += LAYERS * g[t].L.ms;
  }
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  const int levels = lo - hi + 1;
  cudaStream_t sm = prio_stream(lo);
  unsigned long long* stamps;
  const int narms = (int)arm_names.size();
  CK(cudaMalloc(&stamps, sizeof(unsigned long long) * ((size_t)K * NSTAMP + 1) * narms));

  // the serial path's roots, chunk by chunk, for the poisoned check
  auto roots = [&](int k) {
    std::vector<uint8_t> r((size_t)k * NSTAGE * 64);
    for (int c = 0; c < k; c++)
      for (int t = 0; t < NSTAGE; t++) {
        const Commit& m = ch[c].cm[t];
        CK(cudaMemcpy(r.data() + ((size_t)c * NSTAGE + t) * 64, m.tree + m.tree_bytes - 64, 64, cudaMemcpyDeviceToHost));
      }
    return r;
  };
  auto poison = [&](int k) {
    for (int c = 0; c < k; c++)
      for (int t = 0; t < NSTAGE; t++) {
        const Commit& m = ch[c].cm[t];
        CK(cudaMemset(m.tree, 0xA5, m.tree_bytes));
        CK(cudaMemset(m.fmid, 0xA5, (size_t)64 * (m.R + m.width - 1)));
      }
    CK(cudaDeviceSynchronize());
  };
  poison(K);
  for (int c = 0; c < K; c++)
    for (int t = 0; t < NSTAGE; t++) {
      commit_salt(ch[c].cm[t], LAYERS, d.st);
      commit_tree(ch[c].cm[t], LAYERS - 1, ch[c].op[t], d.st);
    }
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  const std::vector<uint8_t> ref = roots(K);

  printf("{\"mode\": \"ovl_setup\", \"M\": %d, \"k_max\": %d, \"draw_us\": %d, \"plain_rb\": %d, \"levels\": %d, "
         "\"fr_rot\": %d, \"mid_ahead\": %d, \"macs_chunk\": %.6e, \"gemm_alone_ms_chunk\": %.4f, \"mem_bytes\": {\"before\": "
         "%zu, \"att_standins\": %zu, \"activations\": %zu, \"commit_state\": %zu}, \"gemms\": [",
         M, K, draw_us, RB, levels, FR_ROT, g_mid, macs_chunk, gemm_alone_ms, mem0, mem_kb - mem0, mem_act - mem_kb,
         mem_cm - mem_act);
  for (int t = 0; t < NSTAGE; t++)
    printf("%s{\"stage\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"algo\": %d, \"ms_alone\": %.4f}", t ? ", " : "", SNAME[t], M,
           GN[t], GK[t], g[t].L.algo_id, g[t].L.ms);
  printf("]}\n");
  fflush(stdout);

  int fails = 0;
  for (int k : ks) {
    // plain's plans for this k
    const int rows = k * M, rb = std::min(rows, RB);
    LtB pl[NSTAGE];
    for (int t = 0; t < NSTAGE; t++)
      pl[t] = LJ[t] >= 0 ? plan(rb, GN[t], GK[t], 1, 0, 0, 0, Ap, d.W[0][LJ[t]], Dp, d.st)
                         : plan(M, M / 2, HID, k, (long long)M * HID, (long long)kb_seq, (long long)M * (M / 2), Ap, Kb[0], Dp, d.st);
    std::vector<ArmRun> arms;
    for (int ai = 0; ai < narms; ai++) {
      const std::string& nm = arm_names[ai];
      ArmRun a;
      a.name = nm;
      a.k = k;
      a.macs = macs_chunk * k;
      a.st = stamps + (size_t)ai * ((size_t)K * NSTAMP + 1);
      if (nm == "plain") {
        a.hash = a.stamps = false;
        a.G = capture(sm, [&] {
          for (int l = 0; l < LAYERS; l++)
            for (int t = 0; t < NSTAGE; t++) {
              if (LJ[t] >= 0)
                for (int r0 = 0; r0 < rows; r0 += rb) ltb_run(pl[t], plt, Ap + (size_t)r0 * GK[t], d.W[l][LJ[t]], Dp, sm);
              else
                ltb_run(pl[t], plt, Ap, Kb[t == 1 ? l : (l + 1) % LAYERS], Dp, sm);
            }
        });
        arms.push_back(a);
        continue;
      }
      const Sched S = sched_of(nm);
      if (k == 1 && (S.classes || S.hp || S.rr)) continue;  // one chunk: the same as eq or nodraw
      if (S.rr && S.rr >= k) continue;
      a.hash = S.hash;
      a.stamps = S.draw;
      std::vector<cudaStream_t> sg(k), sh(k), ss(k);
      for (int c = 0; c < k; c++) {
        const int pg = S.classes ? hi + std::min(levels - 1, c * levels / k) : S.hp < 0 ? hi : lo;
        sg[c] = prio_stream(pg);
        sh[c] = S.hp ? prio_stream(S.hp > 0 ? hi : lo) : sg[c];
        ss[c] = prio_stream(pg);
        a.streams.push_back(sg[c]);
        if (S.hp) a.streams.push_back(sh[c]);
        a.streams.push_back(ss[c]);
      }
      EvPool pool;
      std::vector<cudaEvent_t> gev(S.rr ? (size_t)k * NS : 0);
      for (auto& e : gev) e = pool.get();
      unsigned long long* st = a.st;
      a.G = capture(sm, [&] {
        stamp<<<1, 1, 0, sm>>>(st + (size_t)K * NSTAMP);
        std::vector<std::vector<cudaEvent_t>> salt_ev(k);
        for (int c = 0; c < k; c++) {
          pool.join(sm, sg[c]);
          if (!S.hash) continue;
          pool.join(sg[c], ss[c]);
          for (int t = 0; t < NSTAGE; t++) {
            commit_salt(ch[c].cm[t], LAYERS, ss[c]);
            salt_ev[c].push_back(pool.get());
            CK(cudaEventRecord(salt_ev[c].back(), ss[c]));
          }
        }
        for (int l = 0; l < LAYERS; l++)
          for (int t = 0; t < NSTAGE; t++)
            for (int c = 0; c < k; c++) {
              const int s = l * NSTAGE + t, idx = s * k + c;
              if (S.hash) {
                if (sh[c] != sg[c]) pool.join(sg[c], sh[c]);
                if (l == 0) CK(cudaStreamWaitEvent(sh[c], salt_ev[c][t], 0));
                commit_tree(ch[c].cm[t], l, ch[c].op[t], sh[c]);
                if (sh[c] != sg[c]) pool.join(sh[c], sg[c]);
              }
              if (S.draw) draw_stamp<<<1, 1, 0, sg[c]>>>((unsigned long long)draw_us * 1000ull, st + (size_t)c * NSTAMP + 2 * s);
              if (S.rr && idx >= S.rr) CK(cudaStreamWaitEvent(sg[c], gev[idx - S.rr], 0));
              if (S.gemm) ltb_run(g[t], ch[c].lt, ch[c].A, wt(l, t, c), ch[c].D, sg[c]);
              if (S.rr) CK(cudaEventRecord(gev[idx], sg[c]));
            }
        for (int c = 0; c < k; c++) {
          if (S.draw) stamp<<<1, 1, 0, sg[c]>>>(st + (size_t)c * NSTAMP + 2 * NS);
          pool.join(sg[c], sm);
        }
      });
      arms.push_back(a);
    }
    // two warm-ups each, then the reps interleaved arm by arm, each timed by events with the read SM clock and power
    for (ArmRun& a : arms)
      for (int i = 0; i < 2; i++) CK(cudaGraphLaunch(a.G.ge, sm));
    CK(cudaStreamSynchronize(sm));
    CK(cudaGetLastError());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int r = 0; r < reps; r++)
      for (ArmRun& a : arms) {
        CK(cudaEventRecord(e0, sm));
        CK(cudaGraphLaunch(a.G.ge, sm));
        CK(cudaEventRecord(e1, sm));
        double cs = 0, ps = 0;
        int nc = 0, np = 0;
        while (cudaEventQuery(e1) == cudaErrorNotReady) {
          const double cl = nvml_mhz(d), p = nvml_w(d);
          if (cl > 0) { cs += cl; nc++; }
          if (p > 0) { ps += p; np++; }
        }
        CK(cudaEventSynchronize(e1));
        if (nc == 0) { cs = nvml_mhz(d); nc = 1; }
        if (np == 0) { ps = nvml_w(d); np = 1; }
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        a.reps.push_back({ms, cs / nc, ps / np});
      }
    CK(cudaGetLastError());
    // the stamps of each arm's last rep
    std::vector<std::vector<unsigned long long>> sv(arms.size());
    for (size_t i = 0; i < arms.size(); i++) {
      if (!arms[i].stamps) continue;
      sv[i].resize((size_t)K * NSTAMP + 1);
      CK(cudaMemcpy(sv[i].data(), arms[i].st, sv[i].size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    }
    // poisoned: each hashing arm's roots equal the serial path's; nohash's (the control) don't
    std::vector<int> root_ok(arms.size(), -1);
    for (size_t i = 0; i < arms.size(); i++) {
      if (!arms[i].hash && arms[i].name != "nohash") continue;
      poison(k);
      CK(cudaGraphLaunch(arms[i].G.ge, sm));
      CK(cudaStreamSynchronize(sm));
      CK(cudaGetLastError());
      const std::vector<uint8_t> got = roots(k);
      root_ok[i] = memcmp(got.data(), ref.data(), got.size()) == 0;
      if (arms[i].hash ? !root_ok[i] : root_ok[i]) fails++;
    }
    printf("{\"mode\": \"ovl\", \"M\": %d, \"k\": %d, \"draw_us\": %d, \"plain_rb\": %d, \"macs\": %.6e, \"rejected\": %d, "
           "\"missing_algo\": %d, \"plain_gemms\": [",
           M, k, draw_us, rb, macs_chunk * k, g_rejected_total, g_missing);
    for (int t = 0; t < NSTAGE; t++)
      printf("%s{\"stage\": \"%s\", \"m\": %d, \"batch\": %d, \"algo\": %d, \"ms_alone\": %.4f}", t ? ", " : "", SNAME[t], pl[t].m,
             pl[t].batch, pl[t].L.algo_id, pl[t].L.ms);
    printf("], \"roots\": {");
    bool first = true;
    for (size_t i = 0; i < arms.size(); i++) {
      if (root_ok[i] < 0) continue;
      printf("%s\"%s\": %s", first ? "" : ", ", arms[i].name.c_str(), root_ok[i] ? "\"equal\"" : "\"differ\"");
      first = false;
    }
    // per chunk: its end (ms after the graph's start stamp), the mean commit + draw and GEMM + commit spans a stage
    printf("}, \"chunks\": {");
    first = true;
    for (size_t i = 0; i < arms.size(); i++) {
      if (!arms[i].stamps) continue;
      const std::vector<unsigned long long>& v = sv[i];
      const unsigned long long t0 = v[(size_t)K * NSTAMP];
      printf("%s\"%s\": {\"end_ms\": [", first ? "" : ", ", arms[i].name.c_str());
      first = false;
      for (int c = 0; c < k; c++) printf("%s%.3f", c ? ", " : "", (v[(size_t)c * NSTAMP + 2 * NS] - t0) * 1e-6);
      double draw_wait = 0, between = 0;
      for (int c = 0; c < k; c++)
        for (int s = 0; s < NS; s++) {
          const unsigned long long* x = &v[(size_t)c * NSTAMP + 2 * s];
          draw_wait += (x[1] - x[0]) * 1e-6;
          if (s) between += (x[0] - x[-1]) * 1e-6;
        }
      printf("], \"draw_ms_mean\": %.4f, \"gemm_commit_ms_mean\": %.4f, \"first_draw_ms\": %.4f}", draw_wait / (k * NS),
             between / (k * (NS - 1)), (v[0] - t0) * 1e-6);
    }
    printf("}, \"node_priorities\": {");
    for (size_t i = 0; i < arms.size(); i++) {
      printf("%s\"%s\": ", i ? ", " : "", arms[i].name.c_str());
      print_prios(arms[i].G);
    }
    printf("}, \"arms\": {");
    for (size_t i = 0; i < arms.size(); i++) print_reps(arms[i].name.c_str(), arms[i].reps, arms[i].macs, d.sms, i + 1 < arms.size());
    printf("}}\n");
    fflush(stdout);
    for (ArmRun& a : arms) {
      CK(cudaGraphExecDestroy(a.G.ge));
      for (cudaStream_t s : a.streams) cudaStreamDestroy(s);
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
  }
  return fails;
}

// the commitments' compressions as one small resident kernel: b CTAs of blockDim threads a SM, n chained compressions a
// thread; ts[0] / ts[1] take the earliest CTA start and the latest CTA end
__global__ void sha_beside(int n, uint64_t* out, unsigned long long* ts) {
  unsigned long long t;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    atomicMin(&ts[0], t);
  }
  uint64_t s[8], w[16];
  s512_iv(s);
  for (int j = 0; j < 16; j++) w[j] = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x + 1) * (j + 1);
  for (int i = 0; i < n; i++) {
    fr_compress(s, w);
    for (int j = 0; j < 8; j++) w[j] ^= s[j];
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s[0];
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    atomicMax(&ts[1], t);
  }
}
__global__ void ts_reset(unsigned long long* ts) {
  ts[0] = ~0ull;
  ts[1] = 0;
}

// beside: whether the commitments' hashing can hide next to the GEMMs if it is shaped to fit beside their CTAs. plain's
// GEMMs for k sequences of M tokens (as in ovl) against H = C x k x M x 32 compressions (C = BESIDE_C, 367: a token-layer's
// segment, row, leaf, node and salt compressions in stage.cu's tree) as one sha_beside launch of b CTAs of thr threads
// a SM: alone, launched just before the GEMMs (hfirst) and just after them (gfirst), on a stream at the highest priority
static int run_beside(Dev& d, int M, int k, int reps) {
  const int C = getenv("BESIDE_C") ? atoi(getenv("BESIDE_C")) : 367;
  const int RB = getenv("PLAIN_RB") ? atoi(getenv("PLAIN_RB")) : 8192;
  const size_t kb_seq = (size_t)(M / 2) * HID;
  std::vector<int8_t*> Kb(LAYERS);
  for (int l = 0; l < LAYERS; l++) {
    CK(cudaMalloc(&Kb[l], kb_seq * k));
    fill_s8<<<1024, 256, 0, d.st>>>(Kb[l], kb_seq * k, 0x8000 + l);
  }
  const int rows = k * M, rb = std::min(rows, RB);
  int8_t* Ap;
  int* Dp;
  CK(cudaMalloc(&Ap, (size_t)rows * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(Ap, (size_t)rows * FF, 0x5100);
  CK(cudaMalloc(&Dp, std::max((size_t)rb * 2 * FF, (size_t)k * M * (M / 2)) * sizeof(int)));
  LtCtx plt = lt_ctx();
  CK(cudaStreamSynchronize(d.st));
  const int GN[NSTAGE] = {QKV, M / 2, M / 2, HID, 2 * FF, HID}, GK[NSTAGE] = {HID, HID, HID, HID, HID, FF};
  const int LJ[NSTAGE] = {0, -1, -1, 1, 2, 3};
  LtB pl[NSTAGE];
  for (int t = 0; t < NSTAGE; t++)
    pl[t] = LJ[t] >= 0 ? plan(rb, GN[t], GK[t], 1, 0, 0, 0, Ap, d.W[0][LJ[t]], Dp, d.st)
                       : plan(M, M / 2, HID, k, (long long)M * HID, (long long)kb_seq, (long long)M * (M / 2), Ap, Kb[0], Dp, d.st);
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t sg = prio_stream(lo), sh = prio_stream(hi);
  auto gemms = [&](cudaStream_t s) {
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++) {
        if (LJ[t] >= 0)
          for (int r0 = 0; r0 < rows; r0 += rb) ltb_run(pl[t], plt, Ap + (size_t)r0 * GK[t], d.W[l][LJ[t]], Dp, s);
        else
          ltb_run(pl[t], plt, Ap, Kb[t == 1 ? l : (l + 1) % LAYERS], Dp, s);
      }
  };
  const double H = (double)C * k * M * LAYERS;
  const int cfg[][2] = {{32, 1}, {64, 1}, {128, 1}, {128, 2}, {256, 2}};
  uint64_t* out;
  CK(cudaMalloc(&out, (size_t)d.sms * 512 * sizeof(uint64_t)));
  unsigned long long* ts;
  CK(cudaMalloc(&ts, 4 * sizeof(unsigned long long)));
  cudaEvent_t e0, e1, ef, eh;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventCreateWithFlags(&ef, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&eh, cudaEventDisableTiming));
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  // one timed rep of `order`: 0 the GEMMs alone, 1 the hash alone, 2 hash then GEMMs, 3 GEMMs then hash
  struct Out { double ms, mhz, t_hs, t_he, t_ge; };
  auto rep = [&](int order, int thr, int b, int n) -> Out {
    stamp<<<1, 1, 0, sg>>>(ts + 2);
    ts_reset<<<1, 1, 0, sg>>>(ts);
    CK(cudaEventRecord(e0, sg));
    CK(cudaEventRecord(ef, sg));
    CK(cudaStreamWaitEvent(sh, ef, 0));
    if (order == 1 || order == 2) sha_beside<<<b * d.sms, thr, 0, sh>>>(n, out, ts);
    if (order != 1) gemms(sg);
    stamp<<<1, 1, 0, sg>>>(ts + 3);
    if (order == 3) sha_beside<<<b * d.sms, thr, 0, sh>>>(n, out, ts);
    CK(cudaEventRecord(eh, sh));
    CK(cudaStreamWaitEvent(sg, eh, 0));
    CK(cudaEventRecord(e1, sg));
    double cs = 0;
    int nc = 0;
    while (cudaEventQuery(e1) == cudaErrorNotReady) {
      const double cl = nvml_mhz(d);
      if (cl > 0) { cs += cl; nc++; }
    }
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    unsigned long long h[4];
    CK(cudaMemcpy(h, ts, sizeof h, cudaMemcpyDeviceToHost));
    const double z = (double)h[2];
    return {ms, nc ? cs / nc : nvml_mhz(d), order == 0 ? 0 : (h[0] - z) * 1e-6, order == 0 ? 0 : (h[1] - z) * 1e-6,
            order == 1 ? 0 : (h[3] - z) * 1e-6};
  };
  for (int i = 0; i < 2; i++) rep(0, 128, 1, 16);
  std::vector<double> gv, gm;
  for (int r = 0; r < reps; r++) {
    const Out o = rep(0, 0, 0, 0);
    gv.push_back(o.ms);
    gm.push_back(o.mhz);
  }
  const double g_ms = median(gv);
  printf("{\"mode\": \"beside\", \"M\": %d, \"k\": %d, \"C\": %d, \"H\": %.6e, \"plain_rb\": %d, \"gemm_ms\": %.3f, "
         "\"gemm_mhz\": %.0f, \"need_rate\": %.4e, \"configs\": [",
         M, k, C, H, rb, g_ms, median(gm), H / (g_ms * 1e-3));
  for (size_t ci = 0; ci < sizeof cfg / sizeof cfg[0]; ci++) {
    const int thr = cfg[ci][0], b = cfg[ci][1];
    const int n = (int)std::ceil(H / ((double)d.sms * b * thr));
    rep(1, thr, b, n);
    std::vector<Out> v[4];
    for (int r = 0; r < reps; r++)
      for (int o = 1; o <= 3; o++) v[o].push_back(rep(o, thr, b, n));
    printf("%s{\"thr\": %d, \"ctas_a_sm\": %d, \"n\": %d", ci ? ", " : "", thr, b, n);
    const char* nm[4] = {"", "alone", "hfirst", "gfirst"};
    for (int o = 1; o <= 3; o++) {
      std::sort(v[o].begin(), v[o].end(), [](const Out& a, const Out& c) { return a.ms < c.ms; });
      const Out& m = v[o][v[o].size() / 2];
      const double span = m.t_he - m.t_hs;
      printf(", \"%s\": {\"ms\": %.3f, \"mhz\": %.0f, \"%s\": %.4f, \"hash_start_ms\": %.3f, \"hash_end_ms\": %.3f, "
             "\"gemm_end_ms\": %.3f, \"hash_rate\": %.4e}",
             nm[o], m.ms, m.mhz, o == 1 ? "of_gemm" : "cost", o == 1 ? m.ms / g_ms : m.ms / g_ms - 1, m.t_hs, m.t_he, m.t_ge,
             (double)d.sms * b * thr * n / (span * 1e-3));
    }
    printf("}");
    fflush(stdout);
  }
  printf("]}\n");
  fflush(stdout);
  CK(cudaFree(out));
  CK(cudaFree(ts));
  CK(cudaFree(Ap));
  CK(cudaFree(Dp));
  for (int8_t* p : Kb) CK(cudaFree(p));
  return 0;
}

// SHA-512's throughput on the whole GPU: stage.cu's sha_chain<ROT> (n chained compressions a thread, fr_compress<ROT>
// rolled or s512_compress unrolled) over CTAs a SM x 128 threads, (n = 65) - (n = 1) over 64 compressions a thread. ROT's
// bits put S0, S1, s0, s1's rotations on the FMA pipe (stage.cu's FR_ROT; this binary's commitments use FR_ROT)
template <int ROT>
static void sha_thru_rot(Dev& d, uint64_t* out, cudaEvent_t e0, cudaEvent_t e1, bool& first) {
  for (int unrolled : {0, 1}) {
    if (unrolled && ROT) continue;
    for (int b : {4, 8, 16, 32}) {
      double ms[2];
      for (int k = 0; k < 2; k++) {
        const int n = k ? 65 : 1;
        std::vector<double> v;
        for (int r = 0; r < 5; r++) {
          CK(cudaEventRecord(e0, d.st));
          sha_chain<ROT><<<b * d.sms, 128, 0, d.st>>>(n, unrolled, out);
          CK(cudaEventRecord(e1, d.st));
          CK(cudaEventSynchronize(e1));
          float x;
          CK(cudaEventElapsedTime(&x, e0, e1));
          v.push_back(x);
        }
        std::sort(v.begin(), v.end());
        ms[k] = v[2];
      }
      printf("%s\"%s_rot%d_%dcta\": %.4e", first ? "" : ", ", unrolled ? "unrolled" : "rolled", ROT, b,
             (double)b * d.sms * 128 * 64 / ((ms[1] - ms[0]) * 1e-3));
      first = false;
    }
  }
}
static void sha_throughput(Dev& d) {
  uint64_t* out;
  CK(cudaMalloc(&out, (size_t)32 * d.sms * 128 * sizeof(uint64_t)));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  printf("{\"mode\": \"sha_throughput\", \"mhz\": %.0f, \"sms\": %d, \"fr_rot\": %d, \"compressions_a_s\": {", nvml_mhz(d),
         d.sms, FR_ROT);
  bool first = true;
  sha_thru_rot<0>(d, out, e0, e1, first);
  sha_thru_rot<1>(d, out, e0, e1, first);
  sha_thru_rot<2>(d, out, e0, e1, first);
  sha_thru_rot<3>(d, out, e0, e1, first);
  sha_thru_rot<4>(d, out, e0, e1, first);
  sha_thru_rot<8>(d, out, e0, e1, first);
  sha_thru_rot<12>(d, out, e0, e1, first);
  sha_thru_rot<15>(d, out, e0, e1, first);
  printf("}, \"mhz_after\": %.0f}\n", nvml_mhz(d));
  fflush(stdout);
  CK(cudaFree(out));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: ovl gate DIR M... | ovl ovl DIR M K... [rREPS] | ovl shathru | ovl beside M K [rREPS]   (ARMS: the arms, DRAW_US, PLAIN_RB)\n");
    return 1;
  }
  const std::string mode = argv[1];
  Dev d;
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
  if (want && *want && strncmp(want, uuid, strlen(uuid)) != 0) {
    fprintf(stderr, "device %s is not the leased %s\n", uuid, want);
    return 3;
  }
  fprintf(stderr, "device %s %s, %d SMs, CUDA_DEVICE_MAX_CONNECTIONS=%s\n", name, uuid, d.sms,
          getenv("CUDA_DEVICE_MAX_CONNECTIONS") ? getenv("CUDA_DEVICE_MAX_CONNECTIONS") : "(default)");
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  CK(cudaMalloc(&g_ref, (size_t)256 * VOCAB * sizeof(int)));
  CK(cudaMalloc(&g_cnt, sizeof *g_cnt));
  for (int l = 0; l < LAYERS; l++)
    for (int j = 0; j < 4; j++) {
      const size_t n = (size_t)LN[j] * LK[j];
      CK(cudaMalloc(&d.W[l][j], n));
      fill_s8<<<1024, 256, 0, d.st>>>(d.W[l][j], n, 0x1000 + l * 4 + j);
    }
  CK(cudaMalloc(&d.Wlm, (size_t)VOCAB * HID));
  fill_s8<<<1024, 256, 0, d.st>>>(d.Wlm, (size_t)VOCAB * HID, 0x77);
  CK(cudaMalloc(&d.X, (size_t)8192 * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(d.X, (size_t)8192 * FF, 0x55);
  CK(cudaMalloc(&d.D, (size_t)1 << 30));
  d.kv_bytes = (size_t)1 << 30;
  CK(cudaMalloc(&d.kv, d.kv_bytes));
  fill_s8<<<4096, 256, 0, d.st>>>(d.kv, d.kv_bytes, 0x99);
  CK(cudaMalloc(&d.q, (size_t)256 * HQ * HD));
  fill_s8<<<64, 256, 0, d.st>>>(d.q, (size_t)256 * HQ * HD, 0x31);
  const size_t maxBL = 786432;
  CK(cudaMalloc(&d.P, maxBL * HQ));
  fill_s8<<<1024, 256, 0, d.st>>>(d.P, maxBL * HQ, 0x32);
  CK(cudaMalloc(&d.S, maxBL * HQ * sizeof(int)));
  CK(cudaMalloc(&d.O, (size_t)256 * HQ * HD * sizeof(int)));
  CK(cudaMalloc(&d.sink, 64));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  h_init();
  link_init();
  g_mid = !(getenv("MID") && atoi(getenv("MID")) == 0);
  if (mode == "shathru") {
    sha_throughput(d);
    return 0;
  }
  if (mode == "beside") {
    if (argc < 4) return 1;
    return run_beside(d, atoi(argv[2]), atoi(argv[3]), argc > 4 && argv[4][0] == 'r' ? atoi(argv[4] + 1) : 3);
  }
  if (argc < 3) return 1;
  hm_load(argv[2]);
  if (mode == "gate") {
    std::vector<int> Ms;
    for (int i = 3; i < argc; i++) Ms.push_back(atoi(argv[i]));
    return run_gate(d, argv[2], Ms);
  }
  if (mode == "ovl") {
    if (argc < 5) return 1;
    std::vector<int> ks;
    int reps = 5;
    for (int i = 4; i < argc; i++) {
      if (argv[i][0] == 'r') reps = atoi(argv[i] + 1);
      else ks.push_back(atoi(argv[i]));
    }
    std::vector<std::string> arms;
    const char* an = getenv("ARMS");
    std::string s = an && *an ? an : "plain chunked eq prio hhi hprio hlo rr1 rr2 rr4 nodraw nohash hashonly";
    for (size_t p = 0; p < s.size();) {
      const size_t q = s.find(' ', p);
      const std::string w = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
      if (!w.empty()) arms.push_back(w);
      if (q == std::string::npos) break;
      p = q + 1;
    }
    return run_ovl(d, argv[2], atoi(argv[3]), ks, reps, arms);
  }
  fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 1;
}
