// Red-team probe of the hash lane's sharaw (benchmarks/hash/sharaw/sharaw.cu on cursor/hash-sha-raw-16d5), which this file
// includes unedited (sharaw.sh fetches it at a pinned commit). Two things sharaw's own bench doesn't do:
//
//  gate   for each named arm: one call, then every buffer the call writes or reads back (the levels, the salt scratch S and F)
//         filled with 0xA5, then one more call, compared word for word with commit_gpu's levels; sharaw's bench poisons the
//         levels only, and only before the first of its two calls. Controls the gate must reject: no call at all, the data
//         kernel over half the positions, and a salt kernel launched over the pads only (S and F get no writes), which
//         sharaw's own gate (levels poisoned, S and F left from the arm before) would accept.
//  timed  raw SHA-512 (sharaw's sha_rate), commit_gpu and the named arms, interleaved rep by rep after a warm-up, the SM clock
//         read after every call; then the gate again, after the timed calls.
//
//   sharaw_gate PARAMS N REPS ARM[,ARM...] [TIMED_ARM,...]     JSON lines on stdout; exit 1 when a gate or control fails
#include <map>
#define main sharaw_main
#include SHARAW_CU
#undef main

static int probe(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "sharaw_gate PARAMS N REPS ARMS [TIMED_ARMS]\n"); return 2; }
  const Params p = read_params(argv[1]);
  const u64 n = strtoull(argv[2], nullptr, 10);
  const int reps = atoi(argv[3]);
  auto split = [](std::string s) {
    std::vector<std::string> v;
    size_t a = 0;
    while (a <= s.size()) {
      size_t b = s.find(',', a);
      if (b == std::string::npos) b = s.size();
      if (b > a) v.push_back(s.substr(a, b - a));
      a = b + 1;
    }
    return v;
  };
  const auto gate_arms = split(argv[4]);
  const auto timed_arms = argc >= 6 ? split(argv[5]) : std::vector<std::string>{};
  const u64 bytes = n * sr::LB;
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  uint8_t* src;
  CK(cudaMalloc(&src, bytes + 64));
  fill<<<4096, 256>>>(src, bytes + 64, 7u);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Committer cm;
  cm.init(p, n, sr::LB, false);
  sr::Mine me;
  me.init(p, n);
  const u64 width = cm.width;
  const int depth = cm.depth;
  const double comps = n * (33.0 + 5 + 4) + 2.0 * (width - n) + 2.0 * (width - 1);
  auto cg = [&](cudaStream_t s) {
    cm.salts<true>(nullptr, s);
    cm.leaves<RowBytes>(RowBytes{src, sr::LB, sr::LB}, s);
    cm.nodes(s);
  };
  std::map<std::string, std::function<void(cudaStream_t)>> arm = {
      {"commit_gpu", cg},
      {"sharaw_v1", [&](cudaStream_t s) { me.salts(s); me.data<256, 2, false>(src, s); }},
      {"lf0_cg", [&](cudaStream_t s) { me.leaf_cg<0>(src, s); }},
      {"v1_cg8", [&](cudaStream_t s) { me.v1_cg8(src, s); }},
      {"v1_cg", [&](cudaStream_t s) { me.v1_cg(src, s); }},
      {"v3s_cg", [&](cudaStream_t s) { me.v3s_cg(src, s); }},
      {"v4_128", [&](cudaStream_t s) { me.v4<256>(src, s); }},
      {"v1n", [&](cudaStream_t s) { me.salts(s); me.data<256, 2, false, sr::FULL, 2>(src, s); }},
  };
  const u32 cnt = (u32)std::min<u64>(width, 256);
  std::map<std::string, std::function<void(cudaStream_t)>> controls = {
      {"ctl_no_call", [&](cudaStream_t) {}},
      {"ctl_half_data",
       [&](cudaStream_t s) {
         sr::sr_salt3<<<(unsigned)((width + 127) / 128), 128, 0, s>>>(me.Tn, n, width, me.S, me.F, me.levels, 0, width);
         sr::sr_data3<false><<<(unsigned)((width / 2 + 255) / 256), 256, 0, s>>>(src, n, width, depth, me.S, me.F, me.levels,
                                                                               me.ctr, nullptr, 0, width / 2);
         me.nodes_cg(s);
       }},
      {"ctl_salt_nowrite",
       [&](cudaStream_t s) {
         if (width > n)
           sr::sr_salt3<<<(unsigned)((width - n + 127) / 128), 128, 0, s>>>(me.Tn, n, width, me.S, me.F, me.levels, n, width);
         sr::sr_data3<false><<<(unsigned)(width / cnt), cnt, 0, s>>>(src, n, width, depth, me.S, me.F, me.levels, me.ctr, nullptr,
                                                                     0, width);
         me.nodes_cg(s);
       }},
  };
  (void)cnt;
  cg(st);
  CK(cudaStreamSynchronize(st));
  CK(cudaGetLastError());
  u64 root_cg[8];
  CK(cudaMemcpy(root_cg, cm.lv[depth], 64, cudaMemcpyDeviceToHost));
  printf("{\"device\": \"%s\", \"sms\": %d, \"n\": %llu, \"width\": %llu, \"depth\": %d, \"bytes\": %llu, \"root\": \"", prop.name,
         prop.multiProcessorCount, (unsigned long long)n, (unsigned long long)width, depth, (unsigned long long)bytes);
  for (int i = 0; i < 8; ++i) printf("%016llx", (unsigned long long)root_cg[i]);
  printf("\", \"uuid\": \"%s\"}\n", getenv("GPU_LEASE_UUID") ? getenv("GPU_LEASE_UUID") : "");
  unsigned long long* dbad;
  CK(cudaMalloc(&dbad, sizeof *dbad));
  const u64 words = 8 * (2 * width - 1);
  auto poison = [&](bool salts_too) {
    CK(cudaMemsetAsync(me.levels, 0xA5, 2 * width * 64, st));
    if (salts_too) {
      CK(cudaMemsetAsync(me.S, 0xA5, n * 128, st));
      CK(cudaMemsetAsync(me.F, 0xA5, n * 64, st));
    }
  };
  auto differing = [&]() {
    CK(cudaMemsetAsync(dbad, 0, sizeof *dbad, st));
    diff_kernel<false><<<1024, 256, 0, st>>>(cm.levels, me.levels, words, dbad);
    unsigned long long bad = 0;
    CK(cudaMemcpyAsync(&bad, dbad, sizeof bad, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    return bad;
  };
  bool ok = true;
  auto strict = [&](const std::string& name, const std::function<void(cudaStream_t)>& run, const char* when) {
    run(st);
    poison(true);
    run(st);
    const auto bad = differing();
    printf("{\"gate\": \"strict\", \"arm\": \"%s\", \"when\": \"%s\", \"poisoned\": \"levels,S,F\", \"words\": %llu, \"differing\": %llu}\n",
           name.c_str(), when, (unsigned long long)words, bad);
    return bad;
  };
  for (const auto& a : gate_arms) {
    if (!arm.count(a) || a == "commit_gpu") { fprintf(stderr, "unknown arm %s\n", a.c_str()); return 2; }
    if (strict(a, arm[a], "before timing")) ok = false;
  }
  for (const auto& kv : controls) {
    const auto bad_strict = strict(kv.first, kv.second, "control");
    // sharaw's own gate on the same control: a full arm first (S and F valid), then the levels alone poisoned, two calls
    arm["v1_cg"](st);
    poison(false);
    kv.second(st);
    kv.second(st);
    const auto bad_theirs = differing();
    printf("{\"control\": \"%s\", \"strict_differing\": %llu, \"sharaw_gate_differing\": %llu, \"strict_rejects\": %s, "
           "\"sharaw_gate_rejects\": %s}\n",
           kv.first.c_str(), bad_strict, bad_theirs, bad_strict ? "true" : "false", bad_theirs ? "true" : "false");
    if (!bad_strict) ok = false;
  }
  fflush(stdout);
  if (!ok) { fprintf(stderr, "a gate or control failed: nothing is timed\n"); return 1; }
  if (timed_arms.empty()) return 0;

  struct T { std::string name; std::function<void(cudaStream_t)> run; double comps; };
  std::vector<T> ts;
  u64* io;
  const int rt = 256, rb = prop.multiProcessorCount * 8, chain = 256;
  CK(cudaMalloc(&io, (size_t)rt * rb * 8));
  ts.push_back({"raw", [&](cudaStream_t s) { sha_rate<<<rb, rt, 0, s>>>(io, chain); }, (double)rt * rb * chain});
  ts.push_back({"commit_gpu", cg, comps});
  for (const auto& a : timed_arms) {
    if (!arm.count(a)) { fprintf(stderr, "unknown arm %s\n", a.c_str()); return 2; }
    if (a != "commit_gpu") ts.push_back({a, arm[a], comps});
  }
  Nv nv;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const long long SPIN = 1000000;
  auto timed = [&](T& a, unsigned& mhz) {
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
  for (auto& a : ts) { unsigned m; timed(a, m); timed(a, m); }
  std::vector<std::vector<double>> t(ts.size()), mz(ts.size());
  for (int r = 0; r < reps; ++r)
    for (size_t i = 0; i < ts.size(); ++i) {   // rotate the order each rep
      const size_t j = (i + r) % ts.size();
      unsigned m;
      t[j].push_back(timed(ts[j], m));
      mz[j].push_back(m);
    }
  std::vector<double> raw_rate;
  for (double x : t[0]) raw_rate.push_back(ts[0].comps / x * 1e-6);
  for (size_t i = 0; i < ts.size(); ++i) {
    std::vector<double> pct;
    for (int r = 0; r < reps; ++r) pct.push_back(100.0 * (ts[i].comps / t[i][r] * 1e-6) / raw_rate[r]);
    const double tm = median(t[i]);
    printf("{\"arm\": \"%s\", \"ms\": %.4f, \"gb_per_s\": %.2f, \"gcomp_per_s\": %.4f, \"pct_raw\": %.2f, \"mhz\": %.0f, \"ms_all\": ",
           ts[i].name.c_str(), tm, i == 0 ? 0.0 : bytes / tm * 1e-6, ts[i].comps / tm * 1e-6, median(pct), median(mz[i]));
    jlist(t[i]);
    printf(", \"mhz_all\": ");
    jlist(mz[i], "%.0f");
    printf(", \"pct_raw_all\": ");
    jlist(pct, "%.2f");
    printf("}\n");
  }
  for (const auto& a : timed_arms)
    if (a != "commit_gpu" && strict(a, arm[a], "after timing")) ok = false;
  fflush(stdout);
  return ok ? 0 : 1;
}

int main(int argc, char** argv) { return probe(argc, argv); }
