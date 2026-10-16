// Idle tensor time in tensor-parallel decode over PCIe, and the DRAM-free filler inside it (HC-COMM, cmp-13).
//
// A proxy of one Llama-3.1-70B decode step at tensor parallelism TP on one node, one process a rank with every leased GPU
// visible (rank r runs on device r), as vLLM's workers run. Per layer, in vLLM's order: norm + quantize, qkv (column-parallel,
// n = 10240/TP), RoPE + cache write, attention over the batch's context (the rank's 8/TP KV heads), quantize, o (row-parallel,
// k = 8192/TP), NCCL all-reduce of the B x 8192 BF16 hidden state, residual + norm + quantize, gate_up (n = 57344/TP),
// SiLU-and-mul + quantize, down (k = 28672/TP), all-reduce; one all-reduce for the vocab-parallel embedding first, and the
// lm_head shard (n = 128256/TP) with the all-gather of its BF16 logits last. The linears are int8 cuBLASLt GEMMs (dcap.cu's
// lt_make: the fastest of the heuristic's top 16 whose output matches the host), distinct weights a layer; attention is
// dcap.cu's int8 decode attention (QK and PV, dp4a, memory-bound) over B/TP sequences of dcap's 8 KV heads, which reads the
// same KV bytes as B sequences of the rank's 8/TP heads; the small kernels (norms, RoPE, quantize, SiLU) are one
// elementwise kernel each that reads and writes their bytes. Each arm is one CUDA graph a rank, NCCL captured inside it.
//
// Arms (each rank, in lockstep through the collectives): base (the step), nocomm (the step without its collectives), comm
// (the collectives alone), phase (the step with a filler beside every collective). Served work is the int8 MACs the step's
// linears and attention execute at 70B's shapes (QK and PV at 64/TP query heads); capacity is 1024 MACs/SM/clk at the SM
// clock NVML reads during each rep. Clocks are read, never set.
//
// The phase filler: dcap.cu's DRAM-free chained-MMA tiles (HC-KERN's map 2 by default, DFILL_MAP), launched on a side branch
// of the graph beside each collective, on SMs - TPC_RESERVE blocks so that NCCL's blocks find free SMs. Each warp runs up to
// TPC_TILES tiles from a public range fixed by (step epoch, collective, warp), and stops when the kernel after the
// collective raises the collective's flag: at a tile boundary (TPC_POLL 0) or within TPC_POLL steps (a read issued one poll
// earlier, so the flag's latency doesn't stall the MMAs), dropping the unfinished tile. A warp writes the receipts of the
// tiles it finished and their count; the host checks them against dfill_ref. TPC_POLL -1 ignores the flag (fixed work).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a tpcomm.cu -lcublasLt -lnvidia-ml -ldl -lnccl
#define main dcap_main
#include "../decode_capacity/dcap.cu"
#undef main

#include <nccl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <map>

#define NK(x)                                                                                     \
  do {                                                                                            \
    ncclResult_t r_ = (x);                                                                        \
    if (r_ != ncclSuccess) {                                                                      \
      fprintf(stderr, "NCCL %s at %s:%d: %s\n", #x, __FILE__, __LINE__, ncclGetErrorString(r_)); \
      exit(2);                                                                                    \
    }                                                                                             \
  } while (0)

constexpr int T_LAYERS = 80, T_HID = 8192, T_QKV = 10240, T_FF = 28672, T_VOCAB = 128256, T_HQ = 64, T_HKV = 8, T_HD = 128;

// ---------------------------------------------------------------- kernels
// one of the step's small kernels: reads nin and writes nout 16-byte words; the first thread raises `flag` (the phase filler's
// stop for the collective just before it)
__global__ void glue_k(const int4* __restrict__ in, size_t nin, int4* __restrict__ out, size_t nout, int* flag) {
  if (flag && blockIdx.x == 0 && threadIdx.x == 0) *(volatile int*)flag = 1;
  const size_t st = (size_t)gridDim.x * blockDim.x, n = nin > nout ? nin : nout;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += st) {
    int4 v = i < nin ? in[i] : make_int4(0, 0, 0, 0);
    if (i < nout) out[i] = make_int4(v.x ^ (int)i, v.y, v.z, v.w);
  }
}
// the step's first kernel when a phase filler runs: every collective's flag down, the epoch up
__global__ void prolog_k(int* flags, int n, unsigned long long* epoch) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) flags[i] = 0;
  if (threadIdx.x == 0) *epoch += 1;
}
__device__ __forceinline__ int ld_flag(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// Warp wid of collective `slot` runs tiles tile0 + wid·tiles + i, i < tiles, tile0 = (epoch·nslots + slot)·warps·maxcap (a
// public range a collective, never reused), the tiles and receipts of dcap.cu's dtiles. cnt[wid] is how many it finished; a
// tile the flag interrupts after s steps (a multiple of the poll) writes the receipt of its state after s steps at the next
// index, and part[wid] = s (0 if none), so the verifier recomputes exactly s steps. total[0] sums the steps run (finished
// tiles times S plus interrupted steps) over warps and launches, total[1] the finished tiles' steps alone.
template <int NC, int MAP>
__global__ void __launch_bounds__(512) cfill_k(uint64_t c0, uint64_t c1, const unsigned long long* __restrict__ epoch, int slot,
                                               int nslots, int tiles, int maxcap, int steps, int poll, const int* flag,
                                               unsigned long long* __restrict__ rec, int* __restrict__ cnt, int* __restrict__ part,
                                               unsigned long long* total) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const unsigned long long nw = (unsigned long long)gridDim.x * (blockDim.x >> 5);
  const unsigned long long wid = (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const unsigned long long tile0 = (*epoch * (unsigned long long)nslots + slot) * nw * maxcap;
  int done = 0, cut = 0;
  int fr = poll >= 0 ? ld_flag(flag) : 0;  // tested a poll later, so its latency overlaps MMAs
  for (int i = 0; i < tiles; i++) {
    if (poll >= 0) {
      if (__shfl_sync(0xffffffffu, fr, 0)) break;
      fr = ld_flag(flag);
    }
    const unsigned long long idx = wid * tiles + i;
    const DSeed seed = dseed_x(c0, c1, tile0 + idx, MAP);
    uint32_t b[2][2], c[4], a[NC][4];
    dgen<NC, MAP>(seed, g, t, b, c, a);
    bool stop = false;
    int left = poll;
#pragma unroll 1
    for (int s = 0; s < steps; s += 8) {
#pragma unroll
      for (int u = 0; u < 8; u++)
#pragma unroll
        for (int ch = 0; ch < NC; ch++) dstep<MAP>(a[ch], b[u & 1], c);
      if (poll > 0 && (left -= 8) == 0) {
        if (__shfl_sync(0xffffffffu, fr, 0) && s + 8 < steps) { stop = true; cut = s + 8; break; }
        fr = ld_flag(flag);
        left = poll;
      }
    }
    const uint64_t h = dhash<NC, MAP>(seed, g, t, a);
    if (lane == 0) rec[idx] = h;
    if (stop) break;
    done++;
  }
  if (lane == 0) {
    cnt[wid] = done;
    part[wid] = cut;
    atomicAdd(total, (unsigned long long)done * steps + cut);
    atomicAdd(total + 1, (unsigned long long)done * steps);
  }
}

// ---------------------------------------------------------------- one rank's state
struct PFill {  // the phase filler
  bool on = false;
  int W = 8, NC = 4, S = 256, tiles = 32, tiles_g = 256, poll = 16, grid = 0, map = 2;
  int nslots = 0;
  int* flags = nullptr;
  unsigned long long* epoch = nullptr;
  unsigned long long* total = nullptr;
  unsigned long long* rec = nullptr;  // nslots x warps x max(tiles, tiles_g)
  int* cnt = nullptr;                 // nslots x warps: finished tiles
  int* part = nullptr;                // nslots x warps: the interrupted tile's steps, 0 if none
  cudaStream_t side = nullptr;
  cudaEvent_t fork = nullptr, join = nullptr;
  long long warps() const { return (long long)grid * W; }
  int cap(int slot) const { return slot == nslots - 1 ? tiles_g : tiles; }
  int maxcap() const { return std::max(tiles, tiles_g); }
  long long stride() const { return warps() * maxcap(); }
  // the tile id of warp w's i-th tile at collective `slot` in step `ep`
  unsigned long long tile(unsigned long long ep, int slot, long long w, int i) const {
    return (ep * nslots + slot) * (unsigned long long)stride() + (unsigned long long)w * cap(slot) + i;
  }
};

struct Rank {
  int r = 0, tp = 1;
  Dev d;  // dcap's: st, nv and sms are what time_work reads
  ncclComm_t comm;
  int8_t* W[T_LAYERS][4];
  int8_t* Wlm;
  int8_t* X;
  int* D;
  uint16_t* ar;       // the BF16 hidden state the all-reduces sum
  uint16_t* logits;   // B x VOCAB/TP BF16
  uint16_t* gath;     // B x VOCAB BF16
  int8_t* kv;
  size_t kv_bytes;
  int8_t *q, *P;
  int *S, *O;
  int LN[4], LK[4];
  PFill pf;
};

static int8_t* kv_slot(const Rank& R, int Be, int L, int layer, bool v, bool* wrapped) {
  const size_t per = (size_t)Be * HKV * L * HD, lay = 2 * per;
  const size_t nfit = R.kv_bytes / lay;
  if (nfit < (size_t)T_LAYERS) *wrapped = true;
  const size_t slot = nfit ? (size_t)layer % nfit : 0;
  return R.kv + slot * lay + (v ? per : 0);
}

static void glue(Rank& R, const void* in, size_t in_bytes, void* out, size_t out_bytes, int* flag, cudaStream_t st) {
  const size_t nin = in_bytes / 16, nout = out_bytes / 16, n = std::max(nin, nout);
  const unsigned blocks = (unsigned)std::max<size_t>(1, std::min<size_t>(R.d.sms, (n + 255) / 256));
  glue_k<<<blocks, 256, 0, st>>>((const int4*)in, nin, (int4*)out, nout, flag);
}

static void cfill_launch(const PFill& f, int slot, uint64_t c0, uint64_t c1, cudaStream_t st) {
  const dim3 gr(f.grid), bl(32 * f.W);
  unsigned long long* rec = f.rec + (size_t)slot * f.stride();
  int* cnt = f.cnt + (size_t)slot * f.warps();
  int* part = f.part + (size_t)slot * f.warps();
  const int* flag = f.flags + slot;
  const int cap = f.cap(slot);
#define CF_CASE(NC_, MAP_)                                                                                                 \
  if (f.NC == NC_ && f.map == MAP_) {                                                                                      \
    cfill_k<NC_, MAP_><<<gr, bl, 0, st>>>(c0, c1, f.epoch, slot, f.nslots, cap, f.maxcap(), f.S, f.poll, flag, rec, cnt,   \
                                          part, f.total);                                                                       \
    return;                                                                                                                \
  }
  CF_CASE(1, 1) CF_CASE(2, 1) CF_CASE(4, 1) CF_CASE(1, 2) CF_CASE(2, 2) CF_CASE(4, 2)
#undef CF_CASE
  fprintf(stderr, "phase filler: NC 1, 2 or 4 and map 1 or 2\n");
  exit(1);
}

static uint64_t g_tc0 = 0x0123456789abcdefull, g_tc1 = 0xfedcba9876543210ull;  // the round's coins (DFILL_COINS)

// a collective at slot `slot`: the phase filler forks beside it when `pf`; `fn` enqueues the NCCL call on st
static void collective(Rank& R, bool pf, int slot, cudaStream_t st, const std::function<void()>& fn) {
  if (pf) {
    CK(cudaEventRecord(R.pf.fork, st));
    CK(cudaStreamWaitEvent(R.pf.side, R.pf.fork, 0));
    cfill_launch(R.pf, slot, g_tc0, g_tc1, R.pf.side);
  }
  fn();
}

// kind: 0 the step, 1 without collectives, 2 the collectives alone; pf: the phase filler beside each collective
static void enqueue_step(Rank& R, const std::vector<Lt>& p, const Lt& lm, int kind, bool pf, int B, int L, bool* wrapped) {
  cudaStream_t st = R.d.st;
  const size_t hb = (size_t)B * T_HID * 2;  // the BF16 hidden state
  const int Be = B / R.tp;
  const bool comp = kind != 2, comm = kind != 1;
  int slot = 0;
  int* flag = nullptr;  // the flag the next small kernel raises
  auto coll_ar = [&] {
    if (!comm) return;
    collective(R, pf, slot, st, [&] { NK(ncclAllReduce(R.ar, R.ar, (size_t)B * T_HID, ncclBfloat16, ncclSum, R.comm, st)); });
    flag = pf ? R.pf.flags + slot : nullptr;
    slot++;
  };
  auto g = [&](const void* in, size_t ib, void* out, size_t ob) {
    if (!comp) {
      if (flag) glue(R, R.ar, 16, R.D, 16, flag, st);  // the collectives alone still release their fillers
      flag = nullptr;
      return;
    }
    glue(R, in, ib, out, ob, flag, st);
    flag = nullptr;
  };
  if (pf) prolog_k<<<1, 256, 0, st>>>(R.pf.flags, R.pf.nslots, R.pf.epoch);
  coll_ar();  // the vocab-parallel embedding
  for (int l = 0; l < T_LAYERS; l++) {
    g(R.ar, hb, R.X, (size_t)B * T_HID);  // residual + RMSNorm + quantize
    if (comp) lt_run(p[0], R.X, R.W[l][0], R.D, st);
    g(R.D, (size_t)B * R.LN[0] * 4, R.q, (size_t)B * R.LN[0]);  // RoPE + cache write
    if (comp) {
      const int8_t* K = kv_slot(R, Be, L, l, false, wrapped);
      const int8_t* V = kv_slot(R, Be, L, l, true, wrapped);
      qk_dec<<<dim3((L + QK_W * QK_KW - 1) / (QK_W * QK_KW), HKV, Be), 256, 0, st>>>(K, R.q, R.S, L);
      CK(cudaMemsetAsync(R.O, 0, (size_t)Be * HQ * HD * sizeof(int), st));
      pv_dec<<<dim3((L / 4 + PV_W * PV_GW - 1) / (PV_W * PV_GW), HKV, Be), 256, 0, st>>>(V, R.P, R.O, L);
    }
    g(R.O, (size_t)B * R.LK[1] * 4, R.X, (size_t)B * R.LK[1]);  // quantize the attention output
    if (comp) lt_run(p[1], R.X, R.W[l][1], R.D, st);
    g(R.D, (size_t)B * T_HID * 4, R.ar, hb);  // the o shard's partial sum, BF16
    coll_ar();
    g(R.ar, 2 * hb, R.X, (size_t)B * T_HID);  // residual + RMSNorm + quantize
    if (comp) lt_run(p[2], R.X, R.W[l][2], R.D, st);
    g(R.D, (size_t)B * R.LN[2] * 4, R.X, (size_t)B * R.LK[3]);  // SiLU-and-mul + quantize
    if (comp) lt_run(p[3], R.X, R.W[l][3], R.D, st);
    g(R.D, (size_t)B * T_HID * 4, R.ar, hb);  // the down shard's partial sum, BF16
    coll_ar();
  }
  g(R.ar, hb, R.X, (size_t)B * T_HID);  // final norm + quantize
  if (comp) lt_run(lm, R.X, R.Wlm, R.D, st);
  g(R.D, (size_t)B * (T_VOCAB / R.tp) * 4, R.logits, (size_t)B * (T_VOCAB / R.tp) * 2);
  if (comm) {
    collective(R, pf, slot, st, [&] {
      NK(ncclAllGather(R.logits, R.gath, (size_t)B * (T_VOCAB / R.tp), ncclBfloat16, R.comm, st));
    });
    flag = pf ? R.pf.flags + slot : nullptr;
    slot++;
    glue(R, R.gath, 16, R.D, 16, flag, st);  // the sampler's first kernel
  }
  if (pf) {
    CK(cudaEventRecord(R.pf.join, R.pf.side));
    CK(cudaStreamWaitEvent(st, R.pf.join, 0));
  }
}

static double now_s() { return now_ms() * 1e-3; }

// every rank meets here (a one-element all-reduce, then the stream drained)
static void barrier(Rank& R) {
  NK(ncclAllReduce(R.ar, R.ar, 1, ncclBfloat16, ncclSum, R.comm, R.d.st));
  CK(cudaStreamSynchronize(R.d.st));
}

// The phase filler's gate on this rank: fixed-work launches checked tile for tile against dfill_ref (and the S - 8 control
// differing everywhere); a launch whose flag is already up finishes nothing and writes no receipt (0xA5 stays).
static int pf_gate(Rank& R) {
  PFill& f = R.pf;
  int fails = 0;
  const int keep_poll = f.poll, keep_tiles = f.tiles;
  for (int poll : {-1, 16}) {
    f.poll = poll;
    f.tiles = 2;
    const int slot = 3;
    CK(cudaMemsetAsync(f.rec + (size_t)slot * f.stride(), 0xA5, f.stride() * sizeof(unsigned long long), R.d.st));
    CK(cudaMemsetAsync(f.flags, 0, f.nslots * sizeof(int), R.d.st));
    if (poll >= 0) CK(cudaMemsetAsync(f.flags + slot, 1, 1, R.d.st));  // the flag already up
    cfill_launch(f, slot, g_tc0, g_tc1, R.d.st);
    CK(cudaStreamSynchronize(R.d.st));
    CK(cudaGetLastError());
    unsigned long long ep;
    CK(cudaMemcpy(&ep, f.epoch, sizeof ep, cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> h(f.stride());
    std::vector<int> c(f.warps()), pt(f.warps());
    CK(cudaMemcpy(h.data(), f.rec + (size_t)slot * f.stride(), h.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.data(), f.cnt + (size_t)slot * f.warps(), c.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pt.data(), f.part + (size_t)slot * f.warps(), pt.size() * 4, cudaMemcpyDeviceToHost));
    long bad = 0, ctl = 0, checked = 0, cnt_bad = 0, untouched_bad = 0;
    for (long long w = 0; w < f.warps(); w++) {
      const int want = poll < 0 ? f.tiles : 0;
      cnt_bad += c[w] != want || pt[w] != 0;
      for (int i = 0; i < f.tiles; i++) {
        const unsigned long long v = h[w * f.tiles + i];
        if (i >= c[w]) { untouched_bad += v != 0xA5A5A5A5A5A5A5A5ull; continue; }
        if ((w * f.tiles + i) % 97 != 0) continue;
        const unsigned long long tile = f.tile(ep, slot, w, i);
        bad += v != dfill_ref(g_tc0, g_tc1, tile, f.NC, f.S, f.map);
        ctl += v == dfill_ref(g_tc0, g_tc1, tile, f.NC, f.S - 8, f.map);
        checked++;
      }
    }
    const bool ok = !bad && !ctl && !cnt_bad && !untouched_bad && (poll >= 0 || checked > 0);
    printf("{\"mode\": \"pf_gate\", \"rank\": %d, \"poll\": %d, \"map\": %d, \"W\": %d, \"NC\": %d, \"S\": %d, \"grid\": %d, "
           "\"checked\": %ld, \"mismatches\": %ld, \"control_equal\": %ld, \"cnt_bad\": %ld, \"untouched_bad\": %ld, \"ok\": %s}\n",
           R.r, poll, f.map, f.W, f.NC, f.S, f.grid, checked, bad, ctl, cnt_bad, untouched_bad, ok ? "true" : "false");
    fflush(stdout);
    fails += !ok;
  }
  f.poll = keep_poll;
  f.tiles = keep_tiles;
  return fails;
}

// after an arm: `n` finished and `n` interrupted tiles of the last rep spread over the collectives, each checked against
// dfill_ref at the steps it ran; *cuts counts the last rep's interrupted tiles
static void pf_check(Rank& R, int n, long* checked, long* bad, long* cnt_over, long* cuts) {
  PFill& f = R.pf;
  unsigned long long ep;
  CK(cudaMemcpy(&ep, f.epoch, sizeof ep, cudaMemcpyDeviceToHost));
  std::vector<int> c((size_t)f.nslots * f.warps()), pt(c.size());
  CK(cudaMemcpy(c.data(), f.cnt, c.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(pt.data(), f.part, pt.size() * 4, cudaMemcpyDeviceToHost));
  *checked = *bad = *cnt_over = *cuts = 0;
  for (int s = 0; s < f.nslots; s++)
    for (long long w = 0; w < f.warps(); w++) {
      const size_t k = (size_t)s * f.warps() + w;
      *cnt_over += c[k] + (pt[k] > 0) > f.cap(s) || pt[k] < 0 || pt[k] >= f.S || pt[k] % 8;
      *cuts += pt[k] > 0;
    }
  std::mt19937_64 rng(ep * 7919 + R.r);
  long full = 0, cut = 0;
  for (int k = 0; k < 400 * n && (full < n || cut < n); k++) {
    const int s = (int)(rng() % f.nslots);
    const long long w = (long long)(rng() % f.warps());
    const size_t ix = (size_t)s * f.warps() + w;
    const bool want_cut = full >= n;
    int i, steps;
    if (want_cut) {
      if (pt[ix] <= 0) continue;
      i = c[ix];
      steps = pt[ix];
      cut++;
    } else {
      if (c[ix] <= 0) continue;
      i = (int)(rng() % c[ix]);
      steps = f.S;
      full++;
    }
    unsigned long long v;
    CK(cudaMemcpy(&v, f.rec + (size_t)s * f.stride() + w * f.cap(s) + i, 8, cudaMemcpyDeviceToHost));
    *bad += v != dfill_ref(g_tc0, g_tc1, f.tile(ep, s, w, i), f.NC, steps, f.map);
    (*checked)++;
  }
}

static int getenv_i(const char* k, int dflt) { return getenv(k) ? atoi(getenv(k)) : dflt; }

static void emit(Rank& R, const char* arm, int B, int L, const std::vector<Rep>& v, double served_macs, double fill_steps_rep,
                 const char* extra) {
  std::vector<Rep> s = v;
  std::sort(s.begin(), s.end(), [](const Rep& a, const Rep& b) { return a.ms < b.ms; });
  const Rep& m = s[s.size() / 2];
  const double peak = 1024.0 * R.d.sms * m.mhz * 1e6;
  const double fill_macs = fill_steps_rep * R.pf.NC * 4096.0;
  printf("{\"mode\": \"tp\", \"tp\": %d, \"rank\": %d, \"B\": %d, \"L\": %d, \"arm\": \"%s\", \"ms\": %.4f, \"ms_min\": %.4f, "
         "\"ms_max\": %.4f, \"mhz\": %.1f, \"watt\": %.1f, \"served_macs\": %.6e, \"served_share\": %.5f, \"fill_steps_rep\": %.1f, "
         "\"fill_macs_rep\": %.6e, \"fill_share\": %.5f%s, \"reps\": [",
         R.tp, R.r, B, L, arm, m.ms, s.front().ms, s.back().ms, m.mhz, m.watt, served_macs,
         served_macs / (m.ms * 1e-3) / peak, fill_steps_rep, fill_macs, fill_macs / (m.ms * 1e-3) / peak, extra);
  for (size_t i = 0; i < v.size(); i++) printf("%s[%.4f, %.0f, %.0f]", i ? ", " : "", v[i].ms, v[i].mhz, v[i].watt);
  printf("]}\n");
  fflush(stdout);
}

// one (B, L) point: plans, then arms in the order TPC_ARMS names (default base phase nocomm comm phase base; spaces or commas)
static int run_point(Rank& R, int B, int L, int reps) {
  if (B % R.tp) { fprintf(stderr, "B %d not a multiple of TP %d\n", B, R.tp); return 1; }
  std::vector<Lt> p(4);
  for (int j = 0; j < 4; j++) p[j] = lt_make(B, R.LN[j], R.LK[j], R.X, R.W[0][j], R.D, R.d.st);
  const Lt lm = lt_make(B, T_VOCAB / R.tp, T_HID, R.X, R.Wlm, R.D, R.d.st);
  for (int j = 0; j < 4; j++)
    if (p[j].algo_id == -2) return 6;
  if (lm.algo_id == -2) return 6;
  double lin = 0;
  for (int j = 0; j < 4; j++) lin += (double)T_LAYERS * B * R.LN[j] * R.LK[j];
  lin += (double)B * (T_VOCAB / R.tp) * T_HID;
  const double att = 2.0 * T_LAYERS * B * (T_HQ / R.tp) * T_HD * (double)L;
  const double served = lin + att;
  std::string arms = getenv("TPC_ARMS") ? getenv("TPC_ARMS") : (R.pf.on ? "base phase nocomm comm phase base" : "base nocomm comm base");
  bool wrapped = false;
  int rc = 0;
  size_t pos = 0;
  while (pos < arms.size()) {
    size_t e = arms.find_first_of(" ,", pos);
    if (e == std::string::npos) e = arms.size();
    const std::string a = arms.substr(pos, e - pos);
    pos = e + 1;
    if (a.empty()) continue;
    const int kind = a == "nocomm" ? 1 : a == "comm" || a == "commphase" ? 2 : 0;
    const bool pf = a == "phase" || a == "commphase";
    if (pf && !R.pf.on) continue;
    barrier(R);
    if (pf) CK(cudaMemset(R.pf.total, 0, 16));
    const std::vector<Rep> v = time_work(R.d, [&] { enqueue_step(R, p, lm, kind, pf, B, L, &wrapped); }, reps);
    double steps_rep = 0;
    char extra[256] = "";
    if (pf) {
      unsigned long long tot[2];
      CK(cudaMemcpy(tot, R.pf.total, 16, cudaMemcpyDeviceToHost));
      steps_rep = (double)tot[0] / (reps + 2);  // time_work's 2 warm-ups ran it too
      long checked, bad, over, cuts;
      pf_check(R, 8, &checked, &bad, &over, &cuts);
      snprintf(extra, sizeof extra, ", \"fill_steps_full_rep\": %.1f, \"rec_checked\": %ld, \"rec_bad\": %ld, "
               "\"cnt_over_cap\": %ld, \"cuts\": %ld, \"pf\": [%d, %d, %d, %d, %d, %d, %d]",
               (double)tot[1] / (reps + 2), checked, bad, over, cuts, R.pf.W, R.pf.NC, R.pf.S, R.pf.tiles, R.pf.poll,
               R.pf.grid, R.pf.map);
      if (bad || over) rc = 8;
    }
    emit(R, a.c_str(), B, L, v, kind == 2 ? 0 : served, steps_rep, extra);
  }
  printf("{\"mode\": \"point\", \"tp\": %d, \"rank\": %d, \"B\": %d, \"L\": %d, \"kv_wrapped\": %s, \"lin_macs\": %.6e, "
         "\"att_macs\": %.6e, \"algos\": [%d, %d, %d, %d, %d], \"lt_ms\": [%.4f, %.4f, %.4f, %.4f, %.4f]}\n",
         R.tp, R.r, B, L, wrapped ? "true" : "false", lin, att, p[0].algo_id, p[1].algo_id, p[2].algo_id, p[3].algo_id,
         lm.algo_id, p[0].ms, p[1].ms, p[2].ms, p[3].ms, lm.ms);
  fflush(stdout);
  return rc;
}

static int rank_main(int r, int tp, const ncclUniqueId& id, int argc, char** argv) {
  Rank R;
  R.r = r;
  R.tp = tp;
  CK(cudaSetDevice(r));
  CK(cudaStreamCreate(&R.d.st));
  CK(cudaDeviceGetAttribute(&R.d.sms, cudaDevAttrMultiProcessorCount, r));
  if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); return 2; }
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, r));
  if (nvmlDeviceGetHandleByPciBusId_v2(bus, &R.d.nv) != NVML_SUCCESS) { fprintf(stderr, "NVML handle failed\n"); return 2; }
  char uuid[96] = {0};
  nvmlDeviceGetUUID(R.d.nv, uuid, sizeof uuid);
  if (const char* want = getenv("GPU_LEASE_UUID")) {
    if (*want && !strstr(want, uuid)) { fprintf(stderr, "rank %d: device %s is not leased (%s)\n", r, uuid, want); return 3; }
  }
  fprintf(stderr, "rank %d: device %s bus %s, %d SMs\n", r, uuid, bus, R.d.sms);
  NK(ncclCommInitRank(&R.comm, tp, id, r));
  R.LN[0] = T_QKV / tp; R.LK[0] = T_HID;
  R.LN[1] = T_HID;      R.LK[1] = T_HID / tp;
  R.LN[2] = 2 * T_FF / tp; R.LK[2] = T_HID;
  R.LN[3] = T_HID;      R.LK[3] = T_FF / tp;
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  CK(cudaMalloc(&g_ref, (size_t)256 * VOCAB * sizeof(int)));
  CK(cudaMalloc(&g_cnt, sizeof *g_cnt));
  for (int l = 0; l < T_LAYERS; l++)
    for (int j = 0; j < 4; j++) {
      const size_t n = (size_t)R.LN[j] * R.LK[j];
      CK(cudaMalloc(&R.W[l][j], n));
      fill_s8<<<1024, 256, 0, R.d.st>>>(R.W[l][j], n, 0x1000 + l * 4 + j + 977 * r);
    }
  CK(cudaMalloc(&R.Wlm, (size_t)(T_VOCAB / tp) * T_HID));
  fill_s8<<<1024, 256, 0, R.d.st>>>(R.Wlm, (size_t)(T_VOCAB / tp) * T_HID, 0x77 + r);
  const int maxB = 256, maxL = getenv_i("TPC_MAXL", 8192);
  CK(cudaMalloc(&R.X, (size_t)maxB * T_FF));
  fill_s8<<<1024, 256, 0, R.d.st>>>(R.X, (size_t)maxB * T_FF, 0x55);
  CK(cudaMalloc(&R.D, (size_t)maxB * (T_VOCAB / tp + 64) * 4 + ((size_t)maxB * 2 * T_FF / tp) * 4));
  CK(cudaMalloc(&R.ar, (size_t)maxB * T_HID * 2));
  CK(cudaMemset(R.ar, 0, (size_t)maxB * T_HID * 2));
  CK(cudaMalloc(&R.logits, (size_t)maxB * (T_VOCAB / tp) * 2));
  CK(cudaMalloc(&R.gath, (size_t)maxB * T_VOCAB * 2));
  const size_t maxBe = maxB / tp;
  CK(cudaMalloc(&R.q, std::max<size_t>((size_t)maxB * T_QKV, maxBe * HQ * HD)));
  fill_s8<<<64, 256, 0, R.d.st>>>(R.q, maxBe * HQ * HD, 0x31);
  CK(cudaMalloc(&R.P, maxBe * HQ * maxL));
  fill_s8<<<1024, 256, 0, R.d.st>>>(R.P, maxBe * HQ * maxL, 0x32);
  CK(cudaMalloc(&R.S, maxBe * HQ * maxL * sizeof(int)));
  CK(cudaMalloc(&R.O, std::max<size_t>(maxBe * HQ * HD, (size_t)maxB * T_HID / tp) * sizeof(int)));
  CK(cudaMemset(R.O, 0, std::max<size_t>(maxBe * HQ * HD, (size_t)maxB * T_HID / tp) * sizeof(int)));
  // the phase filler
  PFill& f = R.pf;
  f.on = getenv_i("TPC_PHASE", 1) != 0;
  f.W = getenv_i("TPC_W", 8); f.NC = getenv_i("TPC_NC", 4); f.S = getenv_i("TPC_S", 256);
  f.tiles = getenv_i("TPC_TILES", 32); f.tiles_g = getenv_i("TPC_TILES_G", 256); f.poll = getenv_i("TPC_POLL", 16);
  f.map = dfill_map();
  f.grid = R.d.sms - getenv_i("TPC_RESERVE", 8);
  f.nslots = 2 * T_LAYERS + 2;
  if (f.on) {
    if (f.poll > 0 && f.poll % 8) { fprintf(stderr, "TPC_POLL must be a multiple of 8\n"); return 1; }
    CK(cudaMalloc(&f.flags, f.nslots * sizeof(int)));
    CK(cudaMemset(f.flags, 0, f.nslots * sizeof(int)));
    CK(cudaMalloc(&f.epoch, 8));
    CK(cudaMemset(f.epoch, 0, 8));
    CK(cudaMalloc(&f.total, 16));
    CK(cudaMalloc(&f.rec, (size_t)f.nslots * f.stride() * 8));
    CK(cudaMalloc(&f.cnt, (size_t)f.nslots * f.warps() * 4));
    CK(cudaMalloc(&f.part, (size_t)f.nslots * f.warps() * 4));
    CK(cudaStreamCreateWithFlags(&f.side, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&f.fork, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&f.join, cudaEventDisableTiming));
#define CF_ATTR(NC_, MAP_) \
  CK(cudaFuncSetAttribute(cfill_k<NC_, MAP_>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    CF_ATTR(1, 1) CF_ATTR(2, 1) CF_ATTR(4, 1) CF_ATTR(1, 2) CF_ATTR(2, 2) CF_ATTR(4, 2)
#undef CF_ATTR
  }
  // the KV pool: what is left, less room for cuBLASLt and NCCL
  size_t fr, tot;
  CK(cudaMemGetInfo(&fr, &tot));
  const size_t keep = (size_t)getenv_i("TPC_KEEP_GB", 4) << 30;
  R.kv_bytes = std::min((size_t)getenv_i("TPC_KV_GB", 40) << 30, fr > keep ? fr - keep : 0);
  CK(cudaMalloc(&R.kv, R.kv_bytes));
  fill_s8<<<4096, 256, 0, R.d.st>>>(R.kv, R.kv_bytes, 0x99);
  CK(cudaStreamSynchronize(R.d.st));
  CK(cudaGetLastError());
  fprintf(stderr, "rank %d: weights %.2f GB, KV pool %.2f GB\n", r,
          ((double)T_LAYERS * (R.LN[0] * (double)R.LK[0] + R.LN[1] * (double)R.LK[1] + R.LN[2] * (double)R.LK[2] +
                               R.LN[3] * (double)R.LK[3]) + (double)(T_VOCAB / tp) * T_HID) / 1e9,
          R.kv_bytes / 1e9);
  int rc = 0;
  if (f.on) {
    const int g = pf_gate(R);
    if (g) {
      fprintf(stderr, "rank %d: phase filler gate failed (%d)\n", r, g);
      rc = 5;
    }
  }
  // every rank agrees on the gate before anything is timed
  float ok = rc ? 1.f : 0.f, *dok;
  CK(cudaMalloc(&dok, 4));
  CK(cudaMemcpy(dok, &ok, 4, cudaMemcpyHostToDevice));
  NK(ncclAllReduce(dok, dok, 1, ncclFloat, ncclSum, R.comm, R.d.st));
  CK(cudaMemcpy(&ok, dok, 4, cudaMemcpyDeviceToHost));
  if (ok != 0.f) { ncclCommDestroy(R.comm); return rc ? rc : 5; }
  const int reps = argc > 3 ? atoi(argv[3]) : 9;
  std::string pts = argv[2];
  for (size_t pos = 0; pos < pts.size();) {
    size_t e = pts.find(' ', pos);
    if (e == std::string::npos) e = pts.size();
    const std::string bl = pts.substr(pos, e - pos);
    pos = e + 1;
    if (bl.empty()) continue;
    const int B = atoi(bl.c_str()), L = atoi(bl.substr(bl.find(':') + 1).c_str());
    if (L > maxL) { fprintf(stderr, "L %d over TPC_MAXL %d\n", L, maxL); rc = 1; continue; }
    const int prc = run_point(R, B, L, reps);
    if (prc) rc = prc;
  }
  NK(ncclCommDestroy(R.comm));
  return rc;
}

// The launcher: one rank a visible GPU (or TPC_TP of them), each this binary re-executed with TPC_RANK and the NCCL id,
// its stdout to $TPC_OUT/rank<r>.jsonl and stderr to rank<r>.err when TPC_OUT is set.
int main(int argc, char** argv) {
  if (argc < 3 || std::string(argv[1]) != "run") {
    fprintf(stderr, "usage: tpcomm run \"B:L ...\" [reps]   (TP = the visible GPUs, or TPC_TP; env: TPC_PHASE, TPC_W, TPC_NC,\n"
                    "       TPC_S, TPC_TILES, TPC_TILES_G, TPC_POLL, TPC_RESERVE, TPC_ARMS, TPC_MAXL, TPC_KV_GB, DFILL_MAP,\n"
                    "       DFILL_COINS)\n");
    return 1;
  }
  if (const char* cs = getenv("DFILL_COINS")) {
    if (strlen(cs) != 32 || sscanf(cs, "%16llx%16llx", (unsigned long long*)&g_tc0, (unsigned long long*)&g_tc1) != 2) {
      fprintf(stderr, "DFILL_COINS must be 32 hex digits\n");
      return 1;
    }
  }
  if (const char* rk = getenv("TPC_RANK")) {
    ncclUniqueId id;
    const char* hx = getenv("TPC_ID");
    if (!hx || strlen(hx) != 2 * sizeof id.internal) { fprintf(stderr, "TPC_ID missing\n"); return 1; }
    for (size_t i = 0; i < sizeof id.internal; i++) {
      unsigned v;
      sscanf(hx + 2 * i, "%2x", &v);
      id.internal[i] = (char)v;
    }
    return rank_main(atoi(rk), atoi(getenv("TPC_NRANKS")), id, argc, argv);
  }
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  const int tp = getenv_i("TPC_TP", ndev);
  if (tp < 1 || tp > ndev) { fprintf(stderr, "TP %d with %d visible GPUs\n", tp, ndev); return 1; }
  ncclUniqueId id;
  NK(ncclGetUniqueId(&id));
  char hx[2 * sizeof id.internal + 1];
  for (size_t i = 0; i < sizeof id.internal; i++) snprintf(hx + 2 * i, 3, "%02x", (unsigned char)id.internal[i]);
  const char* out = getenv("TPC_OUT");
  std::vector<pid_t> kids;
  for (int r = 0; r < tp; r++) {
    const pid_t pid = fork();
    if (pid == 0) {
      setenv("TPC_RANK", std::to_string(r).c_str(), 1);
      setenv("TPC_NRANKS", std::to_string(tp).c_str(), 1);
      setenv("TPC_ID", hx, 1);
      if (out) {
        const std::string o = std::string(out) + "/rank" + std::to_string(r);
        if (!freopen((o + ".jsonl").c_str(), "a", stdout) || !freopen((o + ".err").c_str(), "a", stderr)) _exit(9);
      }
      execv("/proc/self/exe", argv);
      _exit(10);
    }
    kids.push_back(pid);
  }
  int rc = 0;
  for (pid_t k : kids) {
    int st = 0;
    waitpid(k, &st, 0);
    const int c = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    if (c) rc = c;
  }
  return rc;
}
