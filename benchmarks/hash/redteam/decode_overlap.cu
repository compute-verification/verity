// Red-team probe of intpipe's decode-window overlap lever, built on its hm96lat.cu unedited (HM96LAT_CU, the file's path):
// its base commit (Setup, Bin) beside a real decode step instead of its stand-in. The step is Qwen3-8B at batch B (default
// 32) as one CUDA graph on stream A: 36 layers of cuBLASLt GEMMs (qkv, o, gate_up, down; each layer its own weights, every
// shape autotuned over the heuristic's algorithms with the layers rotating so no weight sits in L2), a KV read a layer
// (hm96lat's stand-in kernel over KV_MB, a flash-decode attention's bytes), silu(g) u, and lm_head; each layer's output is
// the next one's input. Controls in the same process: hm96lat's stand-in calibrated to the GEMM step's time, and to
// TARGET_MS (the claim's). For each window, rep by rep in rotating order over (step, pre|step salts): the step alone, the
// commit alone on a low-priority stream B, and both (the step's end and the root's), with the NVML SM clock. Before each
// call the window's leaves and root are poisoned (and the logits); after it the root must equal the commit's own root and
// the logits the step's own, bit for bit.
//   decode_overlap overlap PDIR REPS TARGET_MS N[,N]   (B, KV_MB from the environment; anything else goes to hm96lat's main)
#define main hm96lat_main
#include HM96LAT_CU
#undef main
#include <cublasLt.h>
#include <cuda_bf16.h>

#define LT(x)                                                                                 \
  do {                                                                                        \
    cublasStatus_t s_ = (x);                                                                  \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                        \
      fprintf(stderr, "cublasLt %d at %s:%d: %s\n", (int)s_, __FILE__, __LINE__, #x);         \
      exit(4);                                                                                \
    }                                                                                         \
  } while (0)

namespace dec {
using bf16 = __nv_bfloat16;
constexpr int L = 36, H = 4096, QKV = 6144, I = 12288, V = 151936;

__global__ void fillw(bf16* p, size_t n, u32 seed, float a) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    u32 x = (u32)i * 0x9E3779B1u ^ seed ^ (u32)(i >> 32) * 0x85EBCA77u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = __float2bfloat16(((float)(x >> 8) * (2.0f / 16777216.0f) - 1.0f) * a);
  }
}
__global__ void silu_mul(const bf16* __restrict__ gu, bf16* __restrict__ act, int m, int i) {
  for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < (size_t)m * i; j += (size_t)gridDim.x * blockDim.x) {
    const size_t r = j / i, c = j % i;
    const float g = __bfloat162float(gu[r * 2 * i + c]), u = __bfloat162float(gu[r * 2 * i + i + c]);
    act[j] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
  }
}
__global__ void ne16(const uint16_t* a, const uint16_t* b, size_t n, unsigned long long* bad, unsigned long long* nan) {
  unsigned long long c = 0, q = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    c += a[i] != b[i];
    q += (a[i] & 0x7F80u) == 0x7F80u;
  }
  if (c) atomicAdd(bad, c);
  if (q) atomicAdd(nan, q);
}

// Y[m, n] = X[m, k] W[n, k]^T (bf16, fp32 accumulate), X's rows ldx apart
struct Gemm {
  std::string name;
  cublasLtHandle_t h{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t la{}, lb{}, lc{};
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
  int best = 0, m = 0, n = 0, k = 0;
  void* ws = nullptr;
  size_t wsz = 32ull << 20;
  void init(const std::string& nm, cublasLtHandle_t h_, int m_, int n_, int k_, int ldx) {
    name = nm; h = h_; m = m_; n = n_; k = k_;
    LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, n, k));
    LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, m, ldx));
    LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, n, m, n));
    CK(cudaMalloc(&ws, wsz));
    cublasLtMatmulPreference_t pref;
    LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
    algos.resize(16);
    int got = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, lc, lc, pref, 16, algos.data(), &got));
    algos.resize(got);
    cublasLtMatmulPreferenceDestroy(pref);
  }
  cublasStatus_t run(int a, const bf16* X, const bf16* W, bf16* Y, cudaStream_t st) const {
    const float one = 1.0f, zero = 0.0f;
    return cublasLtMatmul(h, op, &one, W, la, X, lb, &zero, Y, lc, Y, lc, &algos[a].algo, ws, wsz, st);
  }
  // the fastest algorithm over 20 calls, call r on layer r % layers' weights (W + r * per)
  void tune(const bf16* X, const bf16* W, size_t per, int layers, bf16* Y, cudaStream_t st) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double bt = 1e30;
    for (int i = 0; i < (int)algos.size(); ++i) {
      if (run(i, X, W, Y, st) != CUBLAS_STATUS_SUCCESS) { printf("{\"tune\": \"%s\", \"algo\": %d, \"failed\": true}\n", name.c_str(), i); continue; }
      for (int r = 0; r < 3; ++r) LT(run(i, X, W + (size_t)(r % layers) * per, Y, st));
      CK(cudaEventRecord(a, st));
      for (int r = 0; r < 20; ++r) LT(run(i, X, W + (size_t)(r % layers) * per, Y, st));
      CK(cudaEventRecord(b, st));
      CK(cudaEventSynchronize(b));
      float ms;
      CK(cudaEventElapsedTime(&ms, a, b));
      if (ms / 20 < bt) { bt = ms / 20; best = i; }
    }
    printf("{\"tuned\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"best\": %d, \"of\": %zu, \"ms\": %.5f, \"gb_per_s\": %.1f}\n", name.c_str(),
           m, n, k, best, algos.size(), bt, 2.0 * n * k / bt * 1e-6);
    fflush(stdout);
  }
};

static double med(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

static int run(const std::string& pdir, int reps, double target, const std::vector<u64>& sizes) {
  const int B = getenv("B") ? atoi(getenv("B")) : 32;
  const double kv_mb = getenv("KV_MB") ? atof(getenv("KV_MB")) : 48.0;
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  int least, greatest;
  CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  cudaStream_t sa, sb;
  CK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithPriority(&sb, cudaStreamNonBlocking, least));
  cudaEvent_t e0, ea, eb;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
  Nv nv;
  unsigned long long* cnt;
  CK(cudaMallocManaged(&cnt, 16));

  // --- the decode step
  const size_t pq = (size_t)QKV * H, po = (size_t)H * H, pgu = (size_t)2 * I * H, pd = (size_t)H * I, plm = (size_t)V * H;
  bf16 *wq, *wo, *wgu, *wd, *wlm, *h0, *hb, *yqkv, *h2, *ygu, *act, *logits, *ref_logits;
  CK(cudaMalloc(&wq, L * pq * 2)); CK(cudaMalloc(&wo, L * po * 2)); CK(cudaMalloc(&wgu, L * pgu * 2)); CK(cudaMalloc(&wd, L * pd * 2));
  CK(cudaMalloc(&wlm, plm * 2));
  fillw<<<8192, 256>>>(wq, L * pq, 11, sqrtf(3.0f / H)); fillw<<<8192, 256>>>(wo, L * po, 12, sqrtf(3.0f / H));
  fillw<<<8192, 256>>>(wgu, L * pgu, 13, sqrtf(3.0f / H)); fillw<<<8192, 256>>>(wd, L * pd, 14, sqrtf(3.0f / I));
  fillw<<<8192, 256>>>(wlm, plm, 15, sqrtf(3.0f / H));
  CK(cudaMalloc(&h0, (size_t)B * H * 2)); CK(cudaMalloc(&hb, (size_t)B * H * 2)); CK(cudaMalloc(&yqkv, (size_t)B * QKV * 2));
  CK(cudaMalloc(&h2, (size_t)B * H * 2)); CK(cudaMalloc(&ygu, (size_t)B * 2 * I * 2)); CK(cudaMalloc(&act, (size_t)B * I * 2));
  CK(cudaMalloc(&logits, (size_t)B * V * 2)); CK(cudaMalloc(&ref_logits, (size_t)B * V * 2));
  fillw<<<1024, 256>>>(h0, (size_t)B * H, 16, 1.7f);
  const size_t kvper = std::max<size_t>(SCTA, (size_t)(kv_mb * 1048576.0) / SCTA * SCTA);
  uint4* kv;
  u32* sink;
  CK(cudaMalloc(&kv, L * kvper)); CK(cudaMemset(kv, 1, L * kvper));
  CK(cudaMalloc(&sink, 4096));
  CK(cudaDeviceSynchronize());
  cublasLtHandle_t lh;
  LT(cublasLtCreate(&lh));
  Gemm gq, go, ggu, gd, glm;
  gq.init("qkv", lh, B, QKV, H, H); go.init("o", lh, B, H, H, QKV); ggu.init("gate_up", lh, B, 2 * I, H, H);
  gd.init("down", lh, B, H, I, I); glm.init("lm_head", lh, B, V, H, H);
  gq.tune(h0, wq, pq, L, yqkv, sa); go.tune(yqkv, wo, po, L, h2, sa); ggu.tune(h2, wgu, pgu, L, ygu, sa);
  gd.tune(act, wd, pd, L, hb, sa); glm.tune(hb, wlm, 0, 1, logits, sa);
  const size_t wbytes = 2 * (L * (pq + po + pgu + pd) + plm), sbytes = wbytes + L * kvper;
  cudaGraphExec_t gstep = capture(sa, [&] {
    for (int l = 0; l < L; ++l) {
      LT(gq.run(gq.best, l ? hb : h0, wq + l * pq, yqkv, sa));
      standin<<<(unsigned)(kvper / SCTA), 256, 0, sa>>>(kv + (size_t)l * kvper / 16, sink);
      LT(go.run(go.best, yqkv, wo + l * po, h2, sa));
      LT(ggu.run(ggu.best, h2, wgu + l * pgu, ygu, sa));
      silu_mul<<<(unsigned)std::min<size_t>(4096, ((size_t)B * I + 255) / 256), 256, 0, sa>>>(ygu, act, B, I);
      LT(gd.run(gd.best, act, wd + l * pd, hb, sa));
    }
    LT(glm.run(glm.best, hb, wlm, logits, sa));
  });
  auto ms = [&](cudaEvent_t a, cudaEvent_t z) { float t; CK(cudaEventElapsedTime(&t, a, z)); return (double)t; };
  auto alone = [&](cudaGraphExec_t g) {
    spin<<<1, 1, 0, sa>>>(200000);
    CK(cudaEventRecord(e0, sa)); CK(cudaGraphLaunch(g, sa)); CK(cudaEventRecord(ea, sa)); CK(cudaEventSynchronize(ea));
    return ms(e0, ea);
  };
  auto logits_bad = [&](unsigned long long* nan) {
    cnt[0] = cnt[1] = 0;
    ne16<<<1024, 256, 0, sa>>>(reinterpret_cast<const uint16_t*>(logits), reinterpret_cast<const uint16_t*>(ref_logits), (size_t)B * V,
                               cnt, cnt + 1);
    CK(cudaStreamSynchronize(sa));
    if (nan) *nan = cnt[1];
    return cnt[0];
  };
  for (int i = 0; i < 3; ++i) alone(gstep);
  CK(cudaMemcpy(ref_logits, logits, (size_t)B * V * 2, cudaMemcpyDeviceToDevice));
  unsigned long long nan0 = 0;
  CK(cudaMemsetAsync(logits, 0xA5, (size_t)B * V * 2, sa));
  std::vector<double> sv;
  for (int i = 0; i < 5; ++i) sv.push_back(alone(gstep));
  const unsigned long long det = logits_bad(&nan0);
  const double tstep = med(sv);
  printf("{\"mode\": \"decode_step\", \"gpu\": \"%s\", \"sms\": %d, \"batch\": %d, \"layers\": %d, \"weight_bytes\": %zu, \"kv_bytes\": %zu, "
         "\"ms\": %.4f, \"gb_per_s\": %.1f, \"rerun_logits_differing\": %llu, \"logits_nan_or_inf\": %llu, \"prio_least\": %d}\n",
         prop.name, prop.multiProcessorCount, B, L, wbytes, L * kvper, tstep, sbytes / tstep * 1e-6, det, nan0, least);
  fflush(stdout);
  int fails = det != 0 || nan0 != 0;

  // --- hm96lat's stand-in, calibrated to a step time
  const size_t BUF = 4ull << 30;
  uint4* buf;
  CK(cudaMalloc(&buf, BUF));
  CK(cudaMemset(buf, 1, BUF));
  auto standin_graph = [&](size_t per) {
    return capture(sa, [&] {
      for (int j = 0; j < KS; ++j) {
        const size_t off = ((size_t)j * per) % (BUF - per) / SCTA * SCTA;
        standin<<<(unsigned)(per / SCTA), 256, 0, sa>>>(buf + off / 16, sink);
      }
    });
  };
  auto calibrate = [&](const char* name, double tgt) {
    size_t per = 64ull << 20;
    cudaGraphExec_t g = nullptr;
    double t = 0;
    for (int it = 0; it < 4; ++it) {
      if (g) CK(cudaGraphExecDestroy(g));
      g = standin_graph(per);
      alone(g);
      std::vector<double> v;
      for (int r = 0; r < 5; ++r) v.push_back(alone(g));
      t = med(v);
      if (it < 3) per = std::max<size_t>(SCTA, (size_t)(per * tgt / t) / SCTA * SCTA);
    }
    printf("{\"mode\": \"standin\", \"name\": \"%s\", \"kernels\": %d, \"bytes\": %zu, \"ms\": %.4f, \"gb_per_s\": %.1f, \"target_ms\": %.3f}\n",
           name, KS, per * KS, t, per * KS / t * 1e-6, tgt);
    fflush(stdout);
    return g;
  };
  struct Step { std::string name; cudaGraphExec_t g; bool logits; };
  std::vector<Step> steps = {{"decode_gemm", gstep, true}, {"standin_eq", calibrate("standin_eq", tstep), false},
                             {"standin_target", calibrate("standin_target", target), false}};

  for (u64 n : sizes) {
    Setup S(pdir + "/params-" + std::to_string(n) + ".bin", n, false, sa);
    Bin b{&S.a, 256, 0};
    u64* root = const_cast<u64*>(b.out_level());
    auto commit = [&](cudaStream_t st, int m) {
      S.a.st = st;
      if (m) b.salt();
      b.data(S.src);
    };
    commit(sb, 1);
    CK(cudaDeviceSynchronize());
    u64 want[8], got[8];
    CK(cudaMemcpy(want, root, 64, cudaMemcpyDeviceToHost));
    auto poison = [&] {
      CK(cudaMemsetAsync(S.a.lv, 0xA5, n * 64, sa));
      CK(cudaMemsetAsync(root, 0xA5, 64, sa));
    };
    auto root_ok = [&] {
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(got, root, 64, cudaMemcpyDeviceToHost));
      return !memcmp(got, want, 64);
    };
    auto commit_alone = [&](int m) {
      poison();
      CK(cudaStreamSynchronize(sa));
      spin<<<1, 1, 0, sb>>>(200000);
      CK(cudaEventRecord(e0, sb)); commit(sb, m); CK(cudaEventRecord(eb, sb)); CK(cudaEventSynchronize(eb));
      return ms(e0, eb);
    };
    auto both = [&](const Step& s, int m, double* tw, double* rt) {
      poison();
      if (s.logits) CK(cudaMemsetAsync(logits, 0xA5, (size_t)B * V * 2, sa));
      spin<<<1, 1, 0, sa>>>(200000);
      CK(cudaEventRecord(e0, sa));
      CK(cudaStreamWaitEvent(sb, e0, 0));
      CK(cudaGraphLaunch(s.g, sa));
      commit(sb, m);
      CK(cudaEventRecord(ea, sa)); CK(cudaEventRecord(eb, sb));
      CK(cudaEventSynchronize(ea)); CK(cudaEventSynchronize(eb));
      *tw = ms(e0, ea); *rt = ms(e0, eb);
    };
    for (int w = 0; w < 2; ++w)
      for (const auto& s : steps) { double x, y; both(s, 1, &x, &y); commit_alone(0); alone(s.g); }
    CK(cudaGetLastError());
    const int na = 2 * (int)steps.size();
    std::vector<std::vector<double>> va(na), vc(na), vw(na), vr(na), vd(na);
    std::vector<int> bad(na, 0);
    for (int r = 0; r < reps; ++r) {
      for (int j = 0; j < na; ++j) {
        const int c = (j + r) % na, m = c & 1;
        const Step& s = steps[c >> 1];
        if (s.logits) CK(cudaMemsetAsync(logits, 0xA5, (size_t)B * V * 2, sa));
        const double t0 = alone(s.g);
        const unsigned long long la = s.logits ? logits_bad(nullptr) : 0;
        const double c0 = commit_alone(m);
        const bool ra = root_ok();
        double tw, rt;
        both(s, m, &tw, &rt);
        const bool rb = root_ok();
        const unsigned long long lb = s.logits ? logits_bad(nullptr) : 0;
        const bool ok = la == 0 && lb == 0 && ra && rb;
        bad[c] += !ok;
        va[c].push_back(t0); vc[c].push_back(c0); vw[c].push_back(tw); vr[c].push_back(rt); vd[c].push_back(tw - t0);
        printf("{\"mode\": \"overlap\", \"step\": \"%s\", \"commit\": \"%s\", \"leaves\": %llu, \"bytes\": %llu, \"rep\": %d, \"step_alone_ms\": %.5f, "
               "\"commit_alone_ms\": %.5f, \"step_with_ms\": %.5f, \"root_ms\": %.5f, \"root_ok\": %s, \"logits_differing_alone\": %llu, "
               "\"logits_differing_with\": %llu, \"mhz\": %u}\n",
               s.name.c_str(), m ? "step" : "pre", (unsigned long long)n, (unsigned long long)(n * LEAF), r, t0, c0, tw, rt,
               ra && rb ? "true" : "false", la, lb, nv.sm());
      }
      fflush(stdout);
    }
    for (int c = 0; c < na; ++c) {
      const double add = med(vd[c]), ca = med(vc[c]);
      printf("{\"summary\": \"%s\", \"commit\": \"%s\", \"leaves\": %llu, \"step_alone_ms\": %.5f, \"commit_alone_ms\": %.5f, \"step_with_ms\": %.5f, "
             "\"root_ms\": %.5f, \"added_ms\": %.5f, \"hidden\": %.4f, \"added_pct_of_step\": %.3f, \"reps\": %d, \"failed_checks\": %d}\n",
             steps[c >> 1].name.c_str(), c & 1 ? "step" : "pre", (unsigned long long)n, med(va[c]), ca, med(vw[c]), med(vr[c]), add,
             1.0 - add / ca, 100.0 * add / med(va[c]), reps, bad[c]);
      fails += bad[c];
    }
    fflush(stdout);
  }
  printf("{\"decode_overlap_fails\": %d}\n", fails);
  return fails ? 1 : 0;
}
}  // namespace dec

int main(int argc, char** argv) {
  if (argc == 6 && !strcmp(argv[1], "overlap")) return dec::run(argv[2], atoi(argv[3]), atof(argv[4]), parse_sizes(argv[5]));
  return hm96lat_main(argc, argv);
}
