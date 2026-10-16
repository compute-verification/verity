// commit_gpu's command line: the selftest, `check` (every level and every leaf's b || c, for reference.py), `bench`
// (alone, as a CUDA graph, with the root on the host, and beside cuBLASLt's int8 GEMM) and the raw compression rate.
// The construction is commit_gpu.cuh's header.
//
//   commit_gpu selftest
//   commit_gpu check PARAMS OUT e2e|rows|seg N LBITS STRIDE ROWS SALTS|prf     (writes OUT/levels.bin, OUT/bc.bin)
//   commit_gpu check PARAMS OUT tile|tileseg M K R OPERAND SALTS|prf
//   commit_gpu bench PARAMS e2e|rows|seg N LBITS REPS [GEMM_M GEMM_N GEMM_K]     (JSON lines on stdout)
//   commit_gpu bench PARAMS tile|tileseg M K R REPS [GEMM_M GEMM_N GEMM_K]
//   commit_gpu sha REPS                                                      (raw compression rate)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a commit_gpu.cu -lcublasLt -lnvidia-ml [-DSHA_ADD=0|1] [-DSHA_ROT=0..15]
// [-DSHA_ROLL=0|1] (sha512.cuh; the defaults, 0, 0 and 1, are the fastest measured)
// (-DCOMMIT_GPU_DIAG_NOMASK: a timing-only build that skips M_key y, to measure its share; -DCOMMIT_GPU_DIAG_MERGED: one
// that skips h's two compressions (the frame leaf takes b ^ c), the timing of a format whose one leaf hash takes b || c and
// pos after a constant midstate (2 compressions where h and the frame leaf take 4); -DCOMMIT_GPU_DIAG_QUAD: one whose tree
// is 4-ary, 3 compressions a node, the timing of a 4-ary node format; the small path's -DCOMMIT_GPU_DIAG_NOOP (tile
// leaves' bytes from registers), -DCOMMIT_GPU_DIAG_NOS (S and F from registers) and -DCOMMIT_GPU_DIAG_NOTREE (a CTA's
// levels on registers, not through global memory): the same compressions without those loads and stores; all refuse
// `check`)
// COMMIT_GPU_SMALL=1: `check` takes the small path (no shared memory), from domain slot 1 and salt slot 1 of 2.
#include <cublasLt.h>
#include <nvml.h>

#include <chrono>

#define COMMIT_GPU_DOMS 2
#include "commit_gpu.cuh"

#define CL(x)                                                                         \
  do {                                                                                \
    cublasStatus_t s_ = (x);                                                          \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                \
      fprintf(stderr, "cuBLASLt %d at %s:%d: %s\n", (int)s_, __FILE__, __LINE__, #x); \
      exit(3);                                                                        \
    }                                                                                 \
  } while (0)

// The raw compression rate: each thread chains REPS compressions, its block from the last state.
__global__ void sha_rate(u64* io, int reps) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 st[8], w[16];
  sha::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sha::compress(st, w);
  }
  if (st[0] == 0x0123456789abcdefull) io[t] = st[1];
}

static int selftest() {
  // FIPS 180-4's "abc" and the two-block vector
  const char* in[2] = {"abc", "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"};
  const char* want[2] = {"ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
                         "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"};
  int bad = 0;
  for (int i = 0; i < 2; ++i) {
    uint8_t d[64];
    sha::digest((const uint8_t*)in[i], strlen(in[i]), d);
    char hex[129];
    for (int j = 0; j < 64; ++j) sprintf(hex + 2 * j, "%02x", d[j]);
    bad += strcmp(hex, want[i]) != 0;
  }
  printf("{\"selftest\": %s}\n", bad ? "false" : "true");
  return bad ? 1 : 0;
}

static std::string out_dir;

static int check(int argc, char** argv) {
  // check PARAMS OUT MODE N LBITS STRIDE ROWS SALTS|prf; the tile modes take M K R for N LBITS STRIDE, ROWS the operand
  if (argc < 10) { fprintf(stderr, "check PARAMS OUT e2e|rows|seg N LBITS STRIDE | tile|tileseg M K R  ROWS SALTS|prf\n"); return 2; }
  const Params p = read_params(argv[2]);
  out_dir = argv[3];
  const std::string mode = argv[4];
  const bool tile = mode == "tile" || mode == "tileseg", seg = mode == "seg" || mode == "tileseg";
  const u64 a = strtoull(argv[5], nullptr, 10), b = strtoull(argv[6], nullptr, 10), c = strtoull(argv[7], nullptr, 10);
  const u64 n = tile ? tile_leaves(a, b, c) : a, lbits = tile ? 256 * c : b, stride = tile ? 0 : c;
  const u32 lb = (u32)((lbits + 7) / 8);
  const bool prf = !strcmp(argv[9], "prf");
  if (!tile && mode != "e2e" && (stride % 16 || stride < lb)) { fprintf(stderr, "a row stride is a multiple of 16 bytes, at least the row\n"); return 2; }
  auto rows = readf(argv[8]);
  const u64 need = tile ? a * b : mode == "e2e" ? 32 * n : (n - 1) * stride + lb;
  if (rows.size() < need) { fprintf(stderr, "rows: %zu bytes is short\n", rows.size()); return 2; }
  rows.resize(rows.size() + 32, 0);
  uint8_t *drows, *dsalts = nullptr;
  CK(cudaMalloc(&drows, rows.size()));
  CK(cudaMemcpy(drows, rows.data(), rows.size(), cudaMemcpyHostToDevice));
  if (!prf) {
    const auto s = readf(argv[9]);
    if (s.size() != 192 * n) { fprintf(stderr, "salts: %zu bytes for %llu leaves\n", s.size(), (unsigned long long)n); return 2; }
    CK(cudaMalloc(&dsalts, s.size() + 16));
    CK(cudaMemcpy(dsalts, s.data(), s.size(), cudaMemcpyHostToDevice));
  }
  const char* sm = getenv("COMMIT_GPU_SMALL");
  const bool small = sm && !strcmp(sm, "1");
  const int slot = small ? 1 : 0;
  Committer cm;
  cm.init(p, n, mode == "e2e" ? 128 : lb, true, seg, lbits, small ? 1 : 0, small, small ? 2 : 1);
  prf ? cm.salts<true>(nullptr, 0) : cm.salts<false>(dsalts, 0);
  if (tile) {
    cm.tiles(drows, a, b, (u32)c, seg, 0, slot);
  } else if (mode == "seg") {
    cm.seg_leaves_from(SegRows{drows, stride}, 0, slot);
  } else if (mode == "e2e") {
    if (lbits != 1024) { fprintf(stderr, "an e2e leaf is 1024 bits\n"); return 2; }
    cm.leaves<RowE2E>(RowE2E{drows}, 0, slot);
  } else {
    cm.leaves<RowBytes>(RowBytes{drows, stride, lb}, 0, slot);
  }
  cm.nodes(0);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<u64> lv(16 * cm.width), bcv(16 * n);
  CK(cudaMemcpy(lv.data(), cm.levels, lv.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(bcv.data(), cm.bc, bcv.size() * 8, cudaMemcpyDeviceToHost));
  auto to_bytes = [](std::vector<u64>& v) {
    std::vector<uint8_t> b(v.size() * 8);
    for (size_t i = 0; i < v.size(); ++i) sha::put_be64(b.data() + 8 * i, v[i]);
    return b;
  };
  u64 used = 0;
  for (int d = 0; d <= cm.depth; ++d) used += cm.width >> d;
  lv.resize(8 * used);
  // the CPU tail over the GPU's level depth - k must give the GPU's levels above it (which the caller holds to the reference)
  const int k = std::min(cm.depth, 6), from = cm.depth - k;
  auto off = [&](int d) { return 8 * (2 * cm.width - 2 * (cm.width >> d)); };
  if (k) {
    std::vector<u64> cur(lv.begin() + off(from), lv.begin() + off(from + 1));
    std::vector<std::vector<u64>> tail;
    cm.host_tail(from, cur, &tail);
    for (int l = 0; l < k; ++l)
      if (!std::equal(tail[l].begin(), tail[l].end(), lv.begin() + off(from + 1 + l))) {
        fprintf(stderr, "the CPU tail's level %d differs from the GPU's\n", from + 1 + l);
        return 1;
      }
  }
  auto lb_ = to_bytes(lv), bb = to_bytes(bcv);
  writef(out_dir + "/levels.bin", lb_.data(), lb_.size());
  writef(out_dir + "/bc.bin", bb.data(), bb.size());
  printf("{\"check\": \"%s\", \"n\": %llu, \"depth\": %d, \"small\": %s}\n", mode.c_str(), (unsigned long long)n, cm.depth,
         small ? "true" : "false");
  cm.release();
  return 0;
}

// --- timing ------------------------------------------------------------------------------------------------------

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  unsigned sm() {
    unsigned c = 0;
    if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c);
    return c;
  }
};

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
static void json_list(const char* name, const std::vector<double>& v) {
  printf("\"%s\": [", name);
  for (size_t i = 0; i < v.size(); ++i) printf("%s%.5f", i ? ", " : "", v[i]);
  printf("]");
}

__global__ void fill(uint8_t* p, size_t n, u32 seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    u32 x = (u32)i * 0x9E3779B1u ^ seed;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = (uint8_t)x;
  }
}

struct Gemm {   // y (m x n, s32) = A (m x k, int8, row-major) B (n x k, int8, row-major)^T by cuBLASLt, autotuned
  cublasLtHandle_t lt{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t la{}, lb{}, lc{};
  cublasLtMatmulAlgo_t algo{};
  void* ws = nullptr;
  size_t wsz = 64 << 20;
  int8_t *A = nullptr, *B = nullptr;
  int32_t* Y = nullptr;
  int m, n, k, algo_index = -1;
  float best_ms = 0;
  void init(int m_, int n_, int k_, cudaStream_t st) {
    m = m_; n = n_; k = k_;
    CL(cublasLtCreate(&lt));
    CK(cudaMalloc(&ws, wsz));
    CK(cudaMalloc(&A, (size_t)m * k)); CK(cudaMalloc(&B, (size_t)n * k)); CK(cudaMalloc(&Y, (size_t)m * n * 4));
    fill<<<1024, 256>>>((uint8_t*)A, (size_t)m * k, 11u);
    fill<<<1024, 256>>>((uint8_t*)B, (size_t)n * k, 13u);
    CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    const cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N));
    CL(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, k, n, k));
    CL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, k, m, k));
    CL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, n, m, n));
    cublasLtMatmulPreference_t pref;
    CL(cublasLtMatmulPreferenceCreate(&pref));
    CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
    cublasLtMatmulHeuristicResult_t res[8];
    int got = 0;
    CL(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, res, &got));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < got; ++i) {
      algo = res[i].algo;
      std::vector<double> t;
      bool ok = true;
      for (int r = 0; r < 6 && ok; ++r) {
        CK(cudaEventRecord(e0, st));
        ok = run(st) == CUBLAS_STATUS_SUCCESS;
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (r) t.push_back(ms);
      }
      if (ok && (algo_index < 0 || median(t) < best_ms)) { best_ms = (float)median(t); algo_index = i; }
    }
    if (algo_index < 0) { fprintf(stderr, "no cuBLASLt int8 algorithm\n"); exit(3); }
    algo = res[algo_index].algo;
  }
  cublasStatus_t run(cudaStream_t st) {
    const int32_t one = 1, zero = 0;
    return cublasLtMatmul(lt, op, &one, B, la, A, lb, &zero, Y, lc, Y, lc, &algo, ws, wsz, st);
  }
};

// The operand's producer, for timing: what follows on its stream is enqueued while it runs, so an interval that starts at
// its end starts when the operand is complete, not when the host gets to a launch.
__global__ void spin(long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
}
constexpr long long SPIN = 400000;   // ~0.2 ms at 2 GHz

static int bench(int argc, char** argv) {
  // bench PARAMS MODE N LBITS REPS [GM GN GK]; the tile modes take M K R for N LBITS
  if (argc < 7) { fprintf(stderr, "bench PARAMS e2e|rows|seg N LBITS | tile|tileseg M K R  REPS [GEMM_M GEMM_N GEMM_K]\n"); return 2; }
  const Params p = read_params(argv[2]);
  const std::string mode = argv[3];
  const bool tile = mode == "tile" || mode == "tileseg", seg = mode == "seg" || mode == "tileseg";
  const int o = tile ? 1 : 0;
  const u64 a = strtoull(argv[4], nullptr, 10), b = strtoull(argv[5], nullptr, 10);
  const u32 R = tile ? (u32)atoi(argv[6]) : 0;
  const u64 n = tile ? tile_leaves(a, b, R) : a, lbits = tile ? 256ull * R : b;
  const int reps = atoi(argv[6 + o]);
  const bool gemm = argc >= 10 + o;
  const u32 lb = (u32)((lbits + 7) / 8);
  const u64 data = tile ? a * b : mode == "e2e" ? 32 * n : (u64)lb * n;   // operand bytes the commitment covers
  if (!tile && mode != "e2e" && lb % 16) { fprintf(stderr, "bench rows and seg: a multiple of 128 bits\n"); return 2; }
  uint8_t* src;
  CK(cudaMalloc(&src, data + 64));
  fill<<<1024, 256>>>(src, data + 64, 7u);
  cudaStream_t sc, sg, sp;
  CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&sg, cudaStreamNonBlocking));
  int prio_lo, prio_hi;
  CK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
  CK(cudaStreamCreateWithPriority(&sp, cudaStreamNonBlocking, prio_hi));
  Committer cm;
  cm.init(p, n, mode == "e2e" ? 128 : lb, false, seg, lbits);
  auto leaves = [&](cudaStream_t st) {
    if (tile) cm.tiles(src, a, b, R, seg, st);
    else if (mode == "e2e") cm.leaves<RowE2E>(RowE2E{src}, st);
    else if (mode == "seg") cm.seg_leaves(src, lb, st);
    else cm.leaves<RowBytes>(RowBytes{src, lb, lb}, st);
  };
  // the critical path is the data phase: the salt phase (salts()) ran before the operand existed
  auto commit = [&](cudaStream_t st, int top) { leaves(st); cm.nodes(st, top); };
  cudaEvent_t e[6];
  for (auto& x : e) CK(cudaEventCreate(&x));
  auto ms = [](cudaEvent_t a, cudaEvent_t b) { float t; CK(cudaEventElapsedTime(&t, a, b)); return (double)t; };
  Nv nv;
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  for (int w = 0; w < 2; ++w) { cm.salts<true>(nullptr, sc); commit(sc, -1); }
  CK(cudaStreamSynchronize(sc));
  CK(cudaGetLastError());
  // the data phase as one CUDA graph (a small commitment is a dozen short kernels, and their launch gaps count): to the
  // root, and to level depth - k for a CPU tail of k levels
  std::vector<int> ks;
  for (int k : {0, 3, 4, 5, 6, 8}) if (k <= cm.depth) ks.push_back(k);
  std::vector<cudaGraph_t> graphs(ks.size());
  std::vector<cudaGraphExec_t> gx(ks.size());
  for (size_t i = 0; i < ks.size(); ++i) {
    CK(cudaStreamBeginCapture(sc, cudaStreamCaptureModeThreadLocal));
    commit(sc, cm.depth - ks[i]);
    CK(cudaStreamEndCapture(sc, &graphs[i]));
    CK(cudaGraphInstantiate(&gx[i], graphs[i], 0));
  }
  auto commit_graph = [&](cudaStream_t st) { CK(cudaGraphLaunch(gx[0], st)); };
  commit_graph(sc);
  CK(cudaStreamSynchronize(sc));
  // alone: the salt phase, the data phase (its leaves, its nodes) and its graph from the producer's end (the latency from
  // the operand's last write to the root in device memory)
  std::vector<double> t_salt, t_all, t_leaf, t_graph, mhz;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e[0], sc));
    cm.salts<true>(nullptr, sc);
    CK(cudaEventRecord(e[1], sc));
    leaves(sc);
    CK(cudaEventRecord(e[2], sc));
    cm.nodes(sc);
    CK(cudaEventRecord(e[3], sc));
    spin<<<1, 1, 0, sc>>>(SPIN);
    CK(cudaEventRecord(e[5], sc));
    commit_graph(sc);
    CK(cudaEventRecord(e[4], sc));
    CK(cudaEventSynchronize(e[4]));
    t_salt.push_back(ms(e[0], e[1]));
    t_leaf.push_back(ms(e[1], e[2]));
    t_all.push_back(ms(e[1], e[3]));
    t_graph.push_back(ms(e[5], e[4]));
    mhz.push_back(nv.sm());
  }
  // the root on the host (host clock): the graph to level depth - k, that level's 2^k nodes to pinned memory, k levels on
  // the CPU; the root is checked against the GPU's
  u64* hbuf;
  CK(cudaMallocHost(&hbuf, 64ull << 8));
  u64 groot[8];
  CK(cudaMemcpy(groot, cm.lv[cm.depth], 64, cudaMemcpyDeviceToHost));
  std::vector<std::vector<double>> t_host(ks.size());
  for (int r = 0; r < reps; ++r)
    for (size_t i = 0; i < ks.size(); ++i) {
      const int from = cm.depth - ks[i];
      const auto t0 = std::chrono::steady_clock::now();
      CK(cudaGraphLaunch(gx[i], sc));
      CK(cudaMemcpyAsync(hbuf, cm.lv[from], 64ull << ks[i], cudaMemcpyDeviceToHost, sc));
      CK(cudaStreamSynchronize(sc));
      std::vector<u64> cur(hbuf, hbuf + (8ull << ks[i]));
      cm.host_tail(from, cur);
      t_host[i].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
#ifndef COMMIT_GPU_DIAG_QUAD   // the CPU tail is binary
      if (!std::equal(cur.begin(), cur.begin() + 8, groot)) { fprintf(stderr, "the CPU tail's root differs (k %d)\n", ks[i]); return 1; }
#endif
    }
  size_t kb = 0;
  for (size_t i = 1; i < ks.size(); ++i) if (median(t_host[i]) < median(t_host[kb])) kb = i;
  const double tc = median(t_all), tgr = median(t_graph), ts = median(t_salt);
  auto blocks = [](u64 bytes) { return (double)((bytes + 17 + 127) / 128); };   // after a first block the host folded
  const double row_comps = mode == "e2e" ? 2
                           : seg ? (cm.segs ? (cm.segs - 1) * blocks(1024) + blocks(cm.seg_last) : 0) + blocks(64ull * cm.segs)
                                 : blocks(lb);
  // the data phase: row, hm96 leaf (2), frame leaf (2) a leaf, and a node (2) a node (the pads' left out); the salt phase:
  // salt (3) and c (2) a leaf
  const double comps = n * (row_comps + 4.0) + 2.0 * (cm.width - 1), salt_comps = 5.0 * n;
  printf("{\"bench\": \"%s\", \"gpu\": \"%s\", \"sms\": %d, \"sha_add\": %d, \"sha_rot\": %d, \"sha_roll\": %d, \"leaf_threads\": %d, \"n\": %llu, \"lbits\": %llu, "
         "\"tile_m\": %llu, \"tile_k\": %llu, \"tile_r\": %u, \"graph_from\": \"producer\", "
         "\"depth\": %d, \"operand_bytes\": %llu, \"salt_ms\": %.5f, \"salt_compressions\": %.0f, \"commit_ms\": %.5f, "
         "\"leaf_ms\": %.5f, \"nodes_ms\": %.5f, \"graph_ms\": %.5f, \"mhz\": %.0f, \"compressions\": %.0f, \"gcomp_per_s\": %.3f, "
         "\"sha_gb_per_s\": %.2f, \"operand_gb_per_s\": %.2f, \"operand_gb_per_s_graph\": %.2f, \"root_on_host_best_k\": %d, "
         "\"root_on_host_best_ms\": %.5f, \"operand_gb_per_s_root_on_host\": %.2f, \"root_on_host_ms\": {",
         mode.c_str(), prop.name, prop.multiProcessorCount, SHA_ADD, SHA_ROT, SHA_ROLL, LEAF_THREADS, (unsigned long long)n,
         (unsigned long long)lbits, (unsigned long long)(tile ? a : 0), (unsigned long long)(tile ? b : 0), R, cm.depth, (unsigned long long)data, ts, salt_comps, tc, median(t_leaf), tc - median(t_leaf),
         tgr, median(mhz), comps, comps / tc * 1e-6, comps * 128 / tc * 1e-6, data / tc * 1e-6, data / tgr * 1e-6, ks[kb],
         median(t_host[kb]), data / median(t_host[kb]) * 1e-6);
  for (size_t i = 0; i < ks.size(); ++i) printf("%s\"%d\": %.5f", i ? ", " : "", ks[i], median(t_host[i]));
  printf("}, ");
  json_list("commit_ms_all", t_all);
  printf(", ");
  json_list("graph_ms_all", t_graph);
  printf("}\n");
  fflush(stdout);
  if (gemm) {
    Gemm G;
    G.init(atoi(argv[7 + o]), atoi(argv[8 + o]), atoi(argv[9 + o]), sg);
    // GEMM alone, then the same GEMMs with the commitment (its graph) beside them on another stream, then both on one
    // stream, then beside again with the commitment's stream at the device's highest priority (a graph launched into a
    // stream takes its priority), so its blocks are dispatched ahead of the GEMM's queued CTAs. Each starts at the end of
    // the operand's producer (spin).
    const int ng = std::max(1, (int)(tgr / G.best_ms + 0.5));
    std::vector<double> g_alone, both_wall, both_commit, both_gemm, serial_wall, mz, hi_wall, hi_commit, hi_gemm;
    for (int r = 0; r < reps; ++r) {
      spin<<<1, 1, 0, sg>>>(SPIN);
      CK(cudaEventRecord(e[0], sg));
      for (int i = 0; i < ng; ++i) CL(G.run(sg));
      CK(cudaEventRecord(e[1], sg));
      CK(cudaEventSynchronize(e[1]));
      g_alone.push_back(ms(e[0], e[1]) / ng);
      // concurrent: both streams wait on one start event
      spin<<<1, 1, 0, sg>>>(SPIN);
      CK(cudaEventRecord(e[0], sg));
      CK(cudaStreamWaitEvent(sc, e[0]));
      for (int i = 0; i < ng; ++i) CL(G.run(sg));
      CK(cudaEventRecord(e[1], sg));
      commit_graph(sc);
      CK(cudaEventRecord(e[2], sc));
      CK(cudaEventSynchronize(e[1]));
      CK(cudaEventSynchronize(e[2]));
      const double tg = ms(e[0], e[1]), tcm = ms(e[0], e[2]);
      both_gemm.push_back(tg);
      both_commit.push_back(tcm);
      both_wall.push_back(std::max(tg, tcm));
      mz.push_back(nv.sm());
      // serial on one stream
      spin<<<1, 1, 0, sg>>>(SPIN);
      CK(cudaEventRecord(e[3], sg));
      for (int i = 0; i < ng; ++i) CL(G.run(sg));
      commit_graph(sg);
      CK(cudaEventRecord(e[4], sg));
      CK(cudaEventSynchronize(e[4]));
      serial_wall.push_back(ms(e[3], e[4]));
      spin<<<1, 1, 0, sg>>>(SPIN);
      CK(cudaEventRecord(e[0], sg));
      CK(cudaStreamWaitEvent(sp, e[0]));
      for (int i = 0; i < ng; ++i) CL(G.run(sg));
      CK(cudaEventRecord(e[1], sg));
      commit_graph(sp);
      CK(cudaEventRecord(e[5], sp));
      CK(cudaEventSynchronize(e[1]));
      CK(cudaEventSynchronize(e[5]));
      hi_gemm.push_back(ms(e[0], e[1]));
      hi_commit.push_back(ms(e[0], e[5]));
      hi_wall.push_back(std::max(hi_gemm.back(), hi_commit.back()));
    }
    const double ga = median(g_alone), bw = median(both_wall), sw = median(serial_wall), hw = median(hi_wall);
    // the share of the shorter side that concurrency hides, against both on one stream measured in the same loop
    const double hidden = (sw - bw) / std::min(ng * ga, tgr), hi_hidden = (sw - hw) / std::min(ng * ga, tgr);
    printf("{\"beside_gemm\": {\"m\": %d, \"n\": %d, \"k\": %d, \"lt_algo_index\": %d, \"gemm_ms\": %.5f, \"gemms\": %d, "
           "\"graph\": true, \"commit_alone_ms\": %.5f, \"both_wall_ms\": %.5f, \"both_commit_ms\": %.5f, \"both_gemm_ms\": %.5f, "
           "\"serial_wall_ms\": %.5f, \"hidden_fraction\": %.4f, \"commit_over_gemm\": %.4f, \"commit_gb_per_s_beside\": %.2f, "
           "\"mhz\": %.0f, \"priority\": {\"stream_priority\": %d, \"wall_ms\": %.5f, \"commit_ms\": %.5f, \"gemm_ms\": %.5f, "
           "\"hidden_fraction\": %.4f, \"commit_gb_per_s_beside\": %.2f}, ",
           G.m, G.n, G.k, G.algo_index, ga, ng, tgr, bw, median(both_commit), median(both_gemm), sw, hidden, tgr / ga,
           data / median(both_commit) * 1e-6, median(mz), prio_hi, hw, median(hi_commit), median(hi_gemm), hi_hidden,
           data / median(hi_commit) * 1e-6);
    json_list("both_wall_all", both_wall);
    printf(", ");
    json_list("serial_wall_all", serial_wall);
    printf(", ");
    json_list("priority_wall_all", hi_wall);
    printf("}}\n");
  }
  for (size_t i = 0; i < ks.size(); ++i) { CK(cudaGraphExecDestroy(gx[i])); CK(cudaGraphDestroy(graphs[i])); }
  CK(cudaFreeHost(hbuf));
  cm.release();
  return 0;
}

static int sha_bench(int reps) {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  const int threads = 256, blocks = prop.multiProcessorCount * 8, chain = 256;
  u64* io; CK(cudaMalloc(&io, (size_t)threads * blocks * 8));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  Nv nv;
  std::vector<double> t, mz;
  for (int r = 0; r <= reps; ++r) {
    CK(cudaEventRecord(e0));
    sha_rate<<<blocks, threads>>>(io, chain);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    if (r) { t.push_back(ms); mz.push_back(nv.sm()); }
  }
  const double comps = (double)threads * blocks * chain, tm = median(t);
  printf("{\"sha\": {\"sha_add\": %d, \"sha_rot\": %d, \"sha_roll\": %d, \"compressions\": %.0f, \"ms\": %.5f, \"gcomp_per_s\": %.3f, \"gb_per_s\": %.2f, \"mhz\": %.0f}}\n",
         SHA_ADD, SHA_ROT, SHA_ROLL, comps, tm, comps / tm * 1e-6, comps * 128 / tm * 1e-6, median(mz));
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "selftest")) return selftest();
#if defined(COMMIT_GPU_DIAG_NOMASK) || defined(COMMIT_GPU_DIAG_MERGED) || defined(COMMIT_GPU_DIAG_QUAD) || \
    defined(COMMIT_GPU_DIAG_NOOP) || defined(COMMIT_GPU_DIAG_NOS) || defined(COMMIT_GPU_DIAG_NOTREE)
  if (argc >= 2 && !strcmp(argv[1], "check")) { fprintf(stderr, "a timing-only build (COMMIT_GPU_DIAG_*) has no check\n"); return 2; }
#endif
  if (argc >= 2 && !strcmp(argv[1], "check")) return check(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "bench")) return bench(argc, argv);
  if (argc >= 3 && !strcmp(argv[1], "sha")) return sha_bench(atoi(argv[2]));
  fprintf(stderr, "usage: commit_gpu selftest | check ... | bench ... | sha REPS\n");
  return 2;
}
