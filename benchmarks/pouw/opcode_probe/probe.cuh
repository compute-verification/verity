// An opcode probe: a kernel that times one instruction class in one loop, and a table entry that says what the loop's
// SASS must hold. `probe.py` gates every probe's SASS (gpu_kit/sass_gate.py) before it times any, and turns the passed
// ones' cycles into price rows (rows.py). README.md has how a probe is written. Compile with -std=c++20.
//
//   OP_KERNEL(name, tpb) { setup; OP_BEGIN; OP_LOOP(it) { the timed instructions }; OP_END; OP_SINK(acc); }
//   static const OpProbe PROBES[] = {{.name = "name", .kernel = (const void*)name, ...}, ...};
//   int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
#pragma once
#include <string>
#include <vector>

#include "../gpu_kit/bench.cuh"

struct OpClk {
  long long cyc;  // the block's SM cycles from OP_BEGIN to OP_END (its slowest warp: both sides hold a barrier)
  int smid, pad;
};

struct OpArgs {
  OpClk* rec;          // one a block
  unsigned* sink;      // one a thread: OP_SINK's store
  unsigned char* g;    // a global buffer of gbytes, zeroed
  unsigned long long gbytes;
  int iters, mode;
};

#define OP_KERNEL(name, tpb) extern "C" __global__ void __launch_bounds__(tpb, 1) name(OpArgs a)
#define OP_BEGIN       \
  __syncthreads();     \
  const long long op_t0_ = clock64();
// One iteration a trip: ptxas adds no unrolled copy or remainder loop, so the gate counts one iteration's instructions.
#define OP_LOOP(it) _Pragma("unroll 1") for (int it = 0; it < a.iters; ++it)
#define OP_END                                                                   \
  __syncthreads();                                                               \
  if (threadIdx.x == 0) {                                                        \
    a.rec[blockIdx.x].cyc = clock64() - op_t0_;                                  \
    int s_;                                                                      \
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s_));                              \
    a.rec[blockIdx.x].smid = s_;                                                 \
  }
// Every thread stores what its timed instructions computed, unconditionally: the gate traces this store back to them.
#define OP_SINK(v) a.sink[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)(v);

// A lane's byte offset into shared memory for one access (`slot`, in the loop's order) at a warp and iteration: the
// kernel computes its addresses with the same function, so the gate's bank check reads the kernel's own addresses.
typedef unsigned (*OpSmemFn)(int slot, int lane, int warp, int it);

struct OpProbe {
  const char* name;            // the kernel's name (OP_KERNEL's): the probe's id
  const void* kernel;
  int tpb;
  const char* kind = "";       // the census kind (instruction class) its rate prices; "": a joint point, rows.jsonl only
  const char* row = "";        // the census row; "": ROW_<kind>
  const char* expect = "";     // the timed loops' opcode counts an iteration, "OP=N,OP=N" a loop, ';' between loops
  const char* counted = "";    // the expect pattern whose rate is the row's (default: the first)
  const char* warps = "";      // warps of a block that run each loop, ';' between loops ("": all)
  double words = 0;            // 4-byte words a warp instruction of `counted` moves, for MACs a word (0: none)
  const char* smem_ops = "";   // the loop's shared accesses in slot order, "OP*N,OP*N" (SASS names), "" for none
  OpSmemFn smem = nullptr;
  int dyn_smem = 0;            // dynamic shared bytes
  int cluster = 1;             // the cluster size its __cluster_dims__ declares: the grid is a multiple of it
  int mode = 0;
  int iters = 2048;
  const char* refuse = "";     // a known-invalid probe: the gate check that must refuse it (loops, smem, sink, forbid)
  const char* note = "";
};

namespace op {

inline std::string json_str(const char* s) {
  std::string o = "\"";
  for (const char* p = s; *p; ++p) o += (*p == '"' || *p == '\\') ? std::string("\\") + *p : std::string(1, *p);
  return o + "\"";
}

// The (warp, iteration) samples the bank check reads: warps 0, 1 and the last, at iterations 0, 1 and 7.
inline std::vector<std::pair<int, int>> samples(int tpb) {
  const int last = tpb / 32 - 1;
  return {{0, 0}, {1, 1}, {last, 7}, {0, 7}};
}

inline void list(const OpProbe* p, size_t n) {
  printf("[\n");
  for (size_t i = 0; i < n; i++) {
    const OpProbe& q = p[i];
    printf(" {\"name\": %s, \"tpb\": %d, \"kind\": %s, \"row\": %s, \"expect\": %s, \"counted\": %s, \"warps\": %s, "
           "\"words\": %.17g, \"smem_ops\": %s, \"dyn_smem\": %d, \"cluster\": %d, \"mode\": %d, \"iters\": %d, "
           "\"refuse\": %s, \"note\": %s",
           json_str(q.name).c_str(), q.tpb, json_str(q.kind).c_str(), json_str(q.row).c_str(),
           json_str(q.expect).c_str(), json_str(q.counted).c_str(), json_str(q.warps).c_str(), q.words,
           json_str(q.smem_ops).c_str(), q.dyn_smem, q.cluster, q.mode, q.iters, json_str(q.refuse).c_str(),
           json_str(q.note).c_str());
    printf(", \"smem\": [");
    if (q.smem && *q.smem_ops) {
      std::vector<std::string> ops;
      std::string spec = q.smem_ops;
      for (size_t at = 0; at <= spec.size();) {
        size_t end = spec.find(',', at);
        if (end == std::string::npos) end = spec.size();
        std::string e = spec.substr(at, end - at);
        size_t star = e.find('*');
        int times = star == std::string::npos ? 1 : atoi(e.c_str() + star + 1);
        for (int t = 0; t < times; t++) ops.push_back(e.substr(0, star));
        at = end + 1;
      }
      bool first = true;
      const auto ss = samples(q.tpb);
      for (size_t s = 0; s < ss.size(); s++)
        for (size_t slot = 0; slot < ops.size(); slot++) {
          printf("%s{\"sample\": %zu, \"warp\": %d, \"it\": %d, \"op\": %s, \"lanes\": [", first ? "" : ", ", s,
                 ss[s].first, ss[s].second, json_str(ops[slot].c_str()).c_str());
          for (int l = 0; l < 32; l++) printf("%s%u", l ? ", " : "", q.smem((int)slot, l, ss[s].first, ss[s].second));
          printf("]}");
          first = false;
        }
    }
    printf("]}%s\n", i + 1 < n ? "," : "");
  }
  printf("]\n");
}

// Time the named probes, interleaved rep by rep (gpu_kit's timed_items), after each one's sink passed the poisoned
// pass and its no-write control. A JSON line each: "gate" (the sink), then "time" a rep with every block's cycles.
inline int time(const OpProbe* all, size_t n, const std::vector<std::string>& names, int reps, double scale) {
  GK_CK(cudaSetDevice(0));
  cudaDeviceProp prop;
  GK_CK(cudaGetDeviceProperties(&prop, 0));
  gk::Nvml nv(0);
  cudaStream_t s;
  GK_CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const unsigned long long gbytes = 64ull << 20;
  unsigned char* g;
  GK_CK(cudaMalloc(&g, gbytes));
  GK_CK(cudaMemsetAsync(g, 0, gbytes, s));
  std::vector<const OpProbe*> ps;
  for (const std::string& nm : names) {
    const OpProbe* q = nullptr;
    for (size_t i = 0; i < n; i++)
      if (nm == all[i].name) q = &all[i];
    if (!q) {
      fprintf(stderr, "ERR no probe %s\n", nm.c_str());
      return 2;
    }
    ps.push_back(q);
  }
  struct Buf { OpClk* rec; unsigned* sink; int blocks; OpArgs args; };
  std::vector<Buf> bufs;
  std::vector<gk::Arm> arms;
  int failed = 0;
  for (const OpProbe* q : ps) {
    Buf b;
    b.blocks = prop.multiProcessorCount / q->cluster * q->cluster;
    GK_CK(cudaMalloc(&b.rec, sizeof(OpClk) * b.blocks));
    GK_CK(cudaMalloc(&b.sink, sizeof(unsigned) * b.blocks * q->tpb));
    if (q->dyn_smem)
      GK_CK(cudaFuncSetAttribute(q->kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, q->dyn_smem));
    b.args = OpArgs{b.rec, b.sink, g, gbytes, std::max(1, (int)(q->iters * scale)), q->mode};
    bufs.push_back(b);
  }
  for (size_t i = 0; i < ps.size(); i++) {
    const OpProbe* q = ps[i];
    Buf* b = &bufs[i];
    auto launch = [q, b](cudaStream_t st, int calls) {
      void* args[] = {&b->args};
      for (int c = 0; c < calls; c++)
        GK_CK(cudaLaunchKernel(q->kernel, dim3(b->blocks), dim3(q->tpb), args, q->dyn_smem, st));
    };
    const size_t sink_bytes = sizeof(unsigned) * b->blocks * q->tpb;
    const gk::PoisonGate pg = gk::poison_gate(
        b->sink, sink_bytes, s, [&](cudaStream_t st) { launch(st, 1); },
        [](const void* h, size_t bytes) { return gk::poisoned_words(h, bytes) == 0; });
    printf("{\"op\": \"gate\", \"probe\": %s, \"sink_written\": %s, \"control_rejected\": %s, \"left\": %zu}\n",
           json_str(q->name).c_str(), pg.accepted ? "true" : "false", pg.control_rejected ? "true" : "false", pg.left);
    if (!pg.ok()) {
      failed++;
      continue;
    }
    // the item's last launch wrote rec: every block's cycles, read once the item is done
    auto after = [q, b, s](const gk::Item& ti) {
      std::vector<OpClk> h(b->blocks);
      GK_CK(cudaMemcpyAsync(h.data(), b->rec, sizeof(OpClk) * b->blocks, cudaMemcpyDeviceToHost, s));
      GK_CK(cudaStreamSynchronize(s));
      printf("{\"op\": \"time\", \"probe\": %s, \"rep\": %d, \"calls\": %d, \"ms\": %.6f, \"sm_mhz\": {\"before\": %u, "
             "\"during\": %u, \"after\": %u}, \"throttle\": \"0x%llx\", \"iters\": %d, \"blocks\": %d, \"cyc\": [",
             json_str(q->name).c_str(), ti.rep, ti.calls, ti.ms, ti.before.sm_mhz, ti.during.sm_mhz, ti.after.sm_mhz,
             ti.before.throttle | ti.during.throttle | ti.after.throttle, b->args.iters, b->blocks);
      for (int i = 0; i < b->blocks; i++) printf("%s%lld", i ? ", " : "", h[i].cyc);
      printf("], \"smid\": [");
      for (int i = 0; i < b->blocks; i++) printf("%s%d", i ? ", " : "", h[i].smid);
      printf("]}\n");
    };
    arms.push_back({q->name, launch, after});
  }
  if (failed) return 3;
  gk::Timing t = gk::timed_items(arms, reps, s, nv);
  printf("{\"op\": \"timing\", \"uuid\": %s, \"nvml\": %s, \"steady\": %s, \"warmup_ms\": %.1f, \"sms\": %d}\n",
         json_str(nv.uuid().c_str()).c_str(), nv.ok() ? "true" : "false", t.steady ? "true" : "false", t.warmup_ms,
         prop.multiProcessorCount);
  gk::summary(t, arms, stderr);
  return 0;
}

}  // namespace op

template <size_t N>
int op_main(int argc, char** argv, const OpProbe (&probes)[N]) {
  std::vector<std::string> names;
  int reps = 10;
  double scale = 1;
  bool listing = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--list") listing = true;
    else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
    else if (a == "--scale" && i + 1 < argc) scale = atof(argv[++i]);
    else if (a == "--time") {
      while (i + 1 < argc && argv[i + 1][0] != '-') names.push_back(argv[++i]);
    } else {
      fprintf(stderr, "usage: %s --list | --time NAME... [--reps R] [--scale X]\n", argv[0]);
      return 2;
    }
  }
  if (listing) {
    op::list(probes, N);
    return 0;
  }
  if (names.empty()) {
    fprintf(stderr, "usage: %s --list | --time NAME... [--reps R] [--scale X]\n", argv[0]);
    return 2;
  }
  return op::time(probes, N, names, reps, scale);
}
