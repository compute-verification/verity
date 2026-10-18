// libvc's served-row commit (vc_commit_rows_launch: hm96-sha512/row/v2 leaves of 4 KiB rows, PRF salts, the frame-v3-sha512
// tree) interleaved rep by rep with the committers it replaces, on one die: commit_gpu's path, native_rows (dlopen'd) and
// sharaw's forms. This file is sharaw.cu (SHARAW_CU, fetched at its pinned commit by bench.sh) with its main renamed, so every
// replaced arm is sharaw's own code; it adds the libvc arms and a driver that compares every arm's levels with commit_gpu's on
// the device before anything is timed, then times each arm once a rep, in turn, with the SM clock (NVML) of each rep.
//
//   rows PARAMS N REPS NATIVE_SO     JSON lines on stdout (PARAMS: commit_gpu's reference.py params, rows of 32768 bits)
#define main sharaw_main
#include SHARAW_CU
#undef main

#include "../vc.h"

namespace {

struct Vc {
  std::vector<uint8_t> plan;
  void* tables = nullptr;
  uint8_t* levels = nullptr;
  void* scratch = nullptr;
  cudaGraphExec_t graph = nullptr;
  u64 n = 0;
  void init(const Params& p, u64 n_) {
    n = n_;
    if (vc_init()) { fprintf(stderr, "vc_init\n"); exit(2); }
    static const char schema[] = "hm96-sha512/row/v2";
    vc_spec s{};
    s.domain_id = p.domain_id;
    s.schema = schema;
    s.schema_len = 18;
    s.salt_prefix = p.salt_prefix;
    s.leaf_prefix = p.leaf_prefix;
    s.key = p.T.data() + 256;   // T_1 = the key
    s.row_scheme = VC_ROW_V2;
    s.row_bits = 8ull * sr::LB;
    s.row_prefix = p.P;
    s.salt_source = VC_SALT_PRF;
    s.salt_key = p.ks;
    plan.resize(vc_plan_bytes());
    if (vc_plan_build(&s, plan.data())) { fprintf(stderr, "vc_plan_build\n"); exit(2); }
    CK(cudaMalloc(&tables, vc_tables_bytes(plan.data())));
    CK(cudaMalloc(&levels, vc_tree_bytes(n)));
    CK(cudaMalloc(&scratch, vc_commit_scratch_bytes(n)));
    if (vc_tables_launch(plan.data(), tables, 0)) { fprintf(stderr, "vc_tables_launch\n"); exit(2); }
    CK(cudaDeviceSynchronize());
  }
  void run(const uint8_t* rows, int form, cudaStream_t st) {
    if (int e = vc_commit_rows_launch(plan.data(), tables, rows, sr::LB, nullptr, n, levels, nullptr, scratch, form, st)) {
      fprintf(stderr, "vc_commit_rows_launch %d\n", e);
      exit(3);
    }
  }
  void capture(const uint8_t* rows, int form, cudaStream_t st) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    run(rows, form, st);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&graph, g, 0));
  }
};

int rows_bench(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "rows PARAMS N REPS NATIVE_SO\n"); return 2; }
  const Params p = read_params(argv[2]);
  const u64 n = strtoull(argv[3], nullptr, 10);
  const int reps = atoi(argv[4]);
  const char* so = argv[5];
  const u64 bytes = n * sr::LB;
  int dev;
  cudaDeviceProp prop;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  uint8_t* src;
  CK(cudaMalloc(&src, bytes + 64));
  fill<<<4096, 256>>>(src, bytes + 64, 7u);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Committer cm;
  cm.init(p, n, sr::LB, false);
  sr::Mine me;
  me.init(p, n);
  Native nat;
  const bool have_nat = so && *so && nat.load(so);
  if (have_nat) nat.init(p, n, me.Tn);
  Vc vs, vf, vg;
  vs.init(p, n);
  vf.init(p, n);
  vg.init(p, n);
  vg.capture(src, VC_FORM_SPLIT, st);
  const u64 width = cm.width;
  const int depth = cm.depth;
  const double comps = n * (33 + 5 + 4.0) + 2.0 * (width - n) + 2.0 * (width - 1);
  auto cg = [&](cudaStream_t s) {
    cm.salts<true>(nullptr, s);
    cm.leaves<RowBytes>(RowBytes{src, sr::LB, sr::LB}, s);
    cm.nodes(s);
  };
  // which buffer each arm's levels are in, and in which word order (BSWAP: digest byte order)
  struct Out { const u64* lv; bool bswap; };
  std::vector<Arm> arms;
  std::vector<Out> outs;
  auto add = [&](const char* name, std::function<void(cudaStream_t)> f, Out o) { arms.push_back({name, true, f, comps}); outs.push_back(o); };
  add("commit_gpu", cg, {nullptr, false});
  if (have_nat) add("native_rows", [&](cudaStream_t s) { nat.run(src, s); }, {(const u64*)nat.lv, true});
  add("sharaw_v1", [&](cudaStream_t s) { me.salts(s); me.data<256, 2, false>(src, s); }, {me.levels, false});
  add("sharaw_v1n", [&](cudaStream_t s) { me.salts(s); me.data<256, 2, false, sr::FULL, 2>(src, s); }, {me.levels, false});
  add("sharaw_v4_128", [&](cudaStream_t s) { me.v4<256>(src, s); }, {me.levels, false});
  add("sharaw_v3s_cg", [&](cudaStream_t s) { me.v3s_cg(src, s); }, {me.levels, false});
  add("sharaw_lf0_cg", [&](cudaStream_t s) { me.leaf_cg<0>(src, s); }, {me.levels, false});
  add("libvc_split", [&](cudaStream_t s) { vs.run(src, VC_FORM_SPLIT, s); }, {(const u64*)vs.levels, true});
  add("libvc_fused", [&](cudaStream_t s) { vf.run(src, VC_FORM_FUSED, s); }, {(const u64*)vf.levels, true});
  add("libvc_auto", [&](cudaStream_t s) { vf.run(src, VC_FORM_AUTO, s); }, {(const u64*)vf.levels, true});
  add("libvc_split_graph", [&](cudaStream_t s) { CK(cudaGraphLaunch(vg.graph, s)); }, {(const u64*)vg.levels, true});
  u64* io;
  const int rt = 256, rb = prop.multiProcessorCount * 8, chain = 256;
  CK(cudaMalloc(&io, (size_t)rt * rb * 8));
  arms.push_back({"raw", false, [&](cudaStream_t s) { sha_rate<<<rb, rt, 0, s>>>(io, chain); }, (double)rt * rb * chain});
  outs.push_back({nullptr, false});

  // every arm's levels against commit_gpu's, on the device, before anything is timed
  unsigned long long* dbad;
  CK(cudaMalloc(&dbad, sizeof *dbad));
  const u64 words = 8 * (2 * width - 1);
  cg(st);
  CK(cudaStreamSynchronize(st));
  CK(cudaGetLastError());
  u64 root[8];
  CK(cudaMemcpy(root, cm.lv[depth], 64, cudaMemcpyDeviceToHost));
  printf("{\"device\": \"%s\", \"sms\": %d, \"n\": %llu, \"width\": %llu, \"depth\": %d, \"bytes\": %llu, \"compressions\": %.0f, \"root\": \"",
         prop.name, prop.multiProcessorCount, (unsigned long long)n, (unsigned long long)width, depth, (unsigned long long)bytes, comps);
  for (int i = 0; i < 8; ++i) printf("%016llx", (unsigned long long)root[i]);
  printf("\"}\n");
  bool all_ok = true;
  for (size_t i = 0; i < arms.size(); ++i) {
    if (!outs[i].lv) continue;
    CK(cudaMemset((void*)outs[i].lv, 0xA5, words * 8));
    for (int k = 0; k < 2; ++k) arms[i].run(st);
    CK(cudaMemsetAsync(dbad, 0, sizeof *dbad, st));
    if (outs[i].bswap) diff_kernel<true><<<1024, 256, 0, st>>>(cm.levels, outs[i].lv, words, dbad);
    else diff_kernel<false><<<1024, 256, 0, st>>>(cm.levels, outs[i].lv, words, dbad);
    unsigned long long bad = 0;
    CK(cudaMemcpyAsync(&bad, dbad, sizeof bad, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    printf("{\"same_levels\": \"%s\", \"against\": \"commit_gpu\", \"words\": %llu, \"differing\": %llu}\n", arms[i].name.c_str(),
           (unsigned long long)words, bad);
    if (bad) { arms[i].failed = true; all_ok = false; }
  }
  fflush(stdout);
  if (!all_ok) fprintf(stderr, "an arm's levels differ from commit_gpu's: it is left out of the timing\n");
  arms.erase(std::remove_if(arms.begin(), arms.end(), [](const Arm& a) { return a.failed; }), arms.end());

  Nv nv;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const long long SPIN = 1000000;
  auto timed = [&](Arm& a, unsigned& mhz) {
    spin<<<1, 1, 0, st>>>(SPIN);
    CK(cudaEventRecord(e0, st));
    a.run(st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    mhz = nv.sm();
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    return (double)t;
  };
  for (auto& a : arms) { unsigned m; timed(a, m); timed(a, m); }
  std::vector<std::vector<double>> t(arms.size()), mz(arms.size());
  for (int r = 0; r < reps; ++r)
    for (size_t i = 0; i < arms.size(); ++i) {
      unsigned m;
      t[i].push_back(timed(arms[i], m));
      mz[i].push_back(m);
    }
  for (size_t i = 0; i < arms.size(); ++i) {
    const auto& a = arms[i];
    const double tm = median(t[i]);
    printf("{\"arm\": \"%s\", \"verified\": %s, \"n\": %llu, \"ms\": %.4f, \"gb_per_s\": %.2f, \"gcomp_per_s\": %.4f, \"mhz\": %.0f, \"ms_all\": ",
           a.name.c_str(), a.verified ? "true" : "false", (unsigned long long)n, tm, a.name == "raw" ? 0.0 : bytes / tm * 1e-6,
           a.comps / tm * 1e-6, median(mz[i]));
    jlist(t[i]);
    printf(", \"mhz_all\": ");
    jlist(mz[i], "%.0f");
    printf("}\n");
  }
  fflush(stdout);
  return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "rows")) return rows_bench(argc, argv);
  return sharaw_main(argc, argv);
}
