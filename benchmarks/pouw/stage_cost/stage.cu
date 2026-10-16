// What RS-D1E's stage coins and deferred edit cost on one RTX PRO 6000 (sm_120), at Llama-3.1-8B's layer shapes, int8
// (STAGE-COST, cmp-13; note:20261016T0750Z-draft-rs-d1e-stage-coins).
//
// The GEMMs are dcap.cu's: cuBLASLt int8 (s8 x s8 -> s32), the fastest of the heuristic's top 16 among those whose output
// equals the host's on sampled rows. The commitment is served_zk's hm96_rows.cu: a stage's operand rows (k int8 bytes a
// token) as hm96-sha512/row-seg/v1 leaves of a frame-v3-sha512 tree, its constants written by prep.py. The decode step beside
// the deferred pass is dcap.cu's (enqueue_step: 32 layers of linears and decode attention, B sequences of context L).
//
// Modes (one JSON line each; clocks are read, never set):
//   gate DIR            every commitment file in DIR committed into 0xA5-poisoned buffers equals Python's leaves, b || c and
//                       root, and a no-write control doesn't; cuBLASLt's int8 against the host (dcap's gate)
//   prefill DIR M       one chunk's forward (32 layers, 6 stages a layer) at M tokens, by arm:
//                         gemm     the stages' GEMMs alone
//                         hash     the stages' operand commitments alone
//                         serial   each stage's commitment, then its GEMM, on one stream
//                         conc     the GEMMs and the commitments on two streams with no dependency (the throughput cost
//                                  of sharing the GPU, with the dependency fully hidden); conc_lo with the hashes at low
//                                  priority
//                         green N  the commitments on a green-context partition of N SMs, the GEMMs on the rest
//                         pipe C   C chunks in flight, each on its own stream (chunk 0 highest priority): for every stage,
//                                  commit its operands, wait DRAW_US for the draw, then the GEMM; C chunks' makespan
//                                  against C x gemm
//                       Attention's QK^T and P.V are int8 GEMMs with the causal stage's MACs (m = M, n = M/2, k = 4096);
//                       P.V's commitment is the per-row normalizers (256 bytes a token), since P is derived.
//   deferred T COMP     the noised recompute of one layer's products batched over T tokens: QKV, o, gate_up, down (in row
//                       chunks of at most 8192), and per sequence QK^T twice (clean, to rebuild P; noised) and P.V, batched
//                       over the 8 KV heads with the 4 query heads of a group stacked. COMP seq500: T/512 sequences of
//                       512 tokens; b32: 32 sequences of T/32 tokens. Contexts spread over 1K..8K; keys = context + half the
//                       sub-round's tokens (the causal mean). Arms gemm (no forming) and form (every operand's noised copy
//                       formed by a pass over it, and S -> P quantized, before the GEMMs: an unfused upper bound).
//   beside T COMP B L   the deferred pass's layers kept queued beside the decode step: the step on the whole GPU, the
//                       step with the pass on a low-priority stream, the step on a green partition with the pass on
//                       FILL_SMS SMs, the step on the whole GPU again; the step's slowdown and the pass's rate in the window
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a stage.cu -lcublasLt -lnvidia-ml -ldl
#define main dcap_main
#include "../decode_capacity/dcap.cu"
#undef main
#include "../served_zk/hm96_rows.cu"

#include <fstream>
#include <map>
#include <tuple>

static const char* RUN_DRAW_US = "DRAW_US";

// ---------------------------------------------------------------- the commitment (hm96_rows.cu, prep.py's constants)
static uint8_t *g_tables, *g_hm_mid;
static std::vector<uint8_t> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static uint8_t* dput(const uint8_t* h, size_t n) {
  uint8_t* p;
  CK(cudaMalloc(&p, std::max<size_t>(n, 64)));
  if (n) CK(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice));
  return p;
}
static void hm_load(const std::string& dir) {
  const std::vector<uint8_t> b = read_file(dir + "/hm.bin");
  const size_t tb = (size_t)192 * 256 * 64;
  if (b.size() != tb + 256) { fprintf(stderr, "hm.bin: %zu bytes\n", b.size()); exit(2); }
  g_tables = dput(b.data(), tb);
  uint8_t* pre = dput(b.data() + tb, 256);
  CK(cudaMalloc(&g_hm_mid, 128));
  Args a{};
  a.p[0] = (unsigned long long)pre;
  a.p[1] = (unsigned long long)g_hm_mid;
  a.i[0] = 2;
  sz_mid<<<1, 32>>>(a);
  CK(cudaDeviceSynchronize());
}

// a constants file: the prefix blocks, heads, pads, and the gate's rows, salts and Python's references
struct HostConst {
  int k, R, s, last, width, nblk;
  std::vector<uint8_t> blocks, leaf_heads, node_heads, pads, rows, salts, hl, bc, root;
};
static HostConst load_const(const std::string& dir, int k, int R) {
  const std::vector<uint8_t> b = read_file(dir + "/c_" + std::to_string(k) + "_" + std::to_string(R) + ".bin");
  uint64_t h[8];
  memcpy(h, b.data(), 64);
  if (h[0] != 0x5354414745434F53ull || (int)h[1] != k || (int)h[2] != R) { fprintf(stderr, "bad constants k %d R %d\n", k, R); exit(2); }
  HostConst c;
  c.k = k; c.R = R; c.s = (int)h[3]; c.last = (int)h[4]; c.width = (int)h[5]; c.nblk = (int)h[6];
  size_t o = 64;
  auto take = [&](std::vector<uint8_t>& v, size_t n) {
    if (o + n > b.size()) { fprintf(stderr, "short constants file k %d R %d\n", k, R); exit(2); }
    v.assign(b.begin() + o, b.begin() + o + n);
    o += n;
  };
  take(c.blocks, (size_t)128 * c.nblk);
  take(c.leaf_heads, (size_t)SLOT * R);
  take(c.node_heads, (size_t)SLOT * std::max(c.width - 1, 1));
  take(c.pads, (size_t)64 * (c.width - R));
  take(c.rows, (size_t)R * k);
  take(c.salts, (size_t)192 * R);
  take(c.hl, (size_t)64 * R);
  take(c.bc, (size_t)128 * R);
  take(c.root, 64);
  return c;
}

// one commitment's device state: its own outputs, so commitments in flight at once never share a buffer
struct Commit {
  const HostConst* hc;
  int k, R, s, last, width;
  uint8_t *mids, *leaf_heads, *node_heads, *digests, *x, *bc, *tree, *salts;
  size_t tree_bytes;
};
static Commit make_commit(const HostConst& c) {
  Commit m;
  m.hc = &c; m.k = c.k; m.R = c.R; m.s = c.s; m.last = c.last; m.width = c.width;
  uint8_t* blk = dput(c.blocks.data(), c.blocks.size());
  CK(cudaMalloc(&m.mids, (size_t)64 * c.nblk));
  Args a{};
  a.p[0] = (unsigned long long)blk;
  a.p[1] = (unsigned long long)m.mids;
  a.i[0] = c.nblk;
  sz_mid<<<(c.nblk + 31) / 32, 32>>>(a);
  m.leaf_heads = dput(c.leaf_heads.data(), c.leaf_heads.size());
  m.node_heads = dput(c.node_heads.data(), c.node_heads.size());
  m.salts = dput(c.salts.data(), c.salts.size());
  CK(cudaMalloc(&m.digests, (size_t)64 * c.R * std::max(c.s, 1)));
  CK(cudaMalloc(&m.x, (size_t)64 * c.R));
  CK(cudaMalloc(&m.bc, (size_t)128 * c.R));
  m.tree_bytes = (size_t)64 * (2 * c.width - 1);
  CK(cudaMalloc(&m.tree, m.tree_bytes));
  if (!c.pads.empty()) CK(cudaMemcpy(m.tree + (size_t)64 * c.R, c.pads.data(), c.pads.size(), cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());
  CK(cudaFree(blk));
  return m;
}
// rows_bench.Commit.run(2, one_cta=False): segment digests, then x, b || c, the hm96 leaf and the frame leaf a thread a
// row, then the tree a level a launch
static void commit_run(const Commit& m, const uint8_t* data, cudaStream_t st) {
  if (m.s) {
    Args a{};
    a.p[0] = (unsigned long long)data;
    a.p[1] = (unsigned long long)m.mids;
    a.p[2] = (unsigned long long)m.digests;
    a.i[0] = m.R; a.i[1] = m.k; a.i[2] = m.s; a.i[3] = m.last;
    const long long total = (long long)m.R * m.s;
    sz_segments<<<(unsigned)((total + 255) / 256), 256, 8 * 32 * 33 * 4, st>>>(a);
  }
  Args f{};
  const unsigned long long ps[10] = {(unsigned long long)(m.s ? m.digests : data), (unsigned long long)(m.mids + 64 * m.s),
                                     (unsigned long long)g_tables, (unsigned long long)m.salts, (unsigned long long)g_hm_mid,
                                     (unsigned long long)m.leaf_heads, (unsigned long long)m.node_heads,
                                     (unsigned long long)m.x, (unsigned long long)m.bc, (unsigned long long)m.tree};
  for (int j = 0; j < 10; j++) f.p[j] = ps[j];
  f.i[0] = m.R; f.i[1] = m.s; f.i[2] = m.k; f.i[3] = 2; f.i[6] = m.width; f.i[7] = m.k;
  sz_finish<<<(unsigned)((m.R + 127) / 128), 128, 0, st>>>(f);
  long long w = m.width, off = 0;
  const uint8_t* heads = m.node_heads;
  while (w > 1) {
    Args l{};
    l.p[0] = (unsigned long long)m.tree;
    l.p[1] = (unsigned long long)heads;
    l.i[0] = (int)off; l.i[1] = (int)(w / 2);
    sz_level<<<(unsigned)((w / 2 + 127) / 128), 128, 0, st>>>(l);
    off += w;
    heads += (size_t)SLOT * (w / 2);
    w /= 2;
  }
}
// the commitment of the gate's rows into poisoned buffers equals Python's (run) or doesn't (the no-write control)
static bool commit_gate(const Commit& m, bool run) {
  const HostConst& c = *m.hc;
  uint8_t* rows = dput(c.rows.data(), c.rows.size());
  CK(cudaMemset(m.x, 0xA5, (size_t)64 * m.R));
  CK(cudaMemset(m.bc, 0xA5, (size_t)128 * m.R));
  CK(cudaMemset(m.tree, 0xA5, m.tree_bytes));
  if (!c.pads.empty()) CK(cudaMemcpy(m.tree + (size_t)64 * c.R, c.pads.data(), c.pads.size(), cudaMemcpyHostToDevice));
  if (run) commit_run(m, rows, 0);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint8_t> hl((size_t)64 * m.R), bc((size_t)128 * m.R), root(64);
  CK(cudaMemcpy(hl.data(), m.x, hl.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(bc.data(), m.bc, bc.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(root.data(), m.tree + m.tree_bytes - 64, 64, cudaMemcpyDeviceToHost));
  CK(cudaFree(rows));
  return hl == c.hl && bc == c.bc && root == c.root;
}

// ---------------------------------------------------------------- the draw: one thread waits on the global timer
__global__ void draw_wait(unsigned long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - t0 < ns);
}

// ---------------------------------------------------------------- forming: dst = src + noise, a pass over the operand
__device__ __forceinline__ uint32_t mix32(uint64_t x) {
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29; x *= 0x94D049BB133111EBull; x ^= x >> 32;
  return (uint32_t)x;
}
__global__ void form_add(const int4* __restrict__ src, int4* __restrict__ dst, size_t n16, uint64_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = __ldcs(src + i);
    const uint64_t h = (i + 1) * 0x9E3779B97F4A7C15ull ^ seed;
    v.x = (int)__vadd4((unsigned)v.x, mix32(h) & 0x01010101u);
    v.y = (int)__vadd4((unsigned)v.y, mix32(h + 1) & 0x01010101u);
    v.z = (int)__vadd4((unsigned)v.z, mix32(h + 2) & 0x01010101u);
    v.w = (int)__vadd4((unsigned)v.w, mix32(h + 3) & 0x01010101u);
    __stcs(dst + i, v);
  }
}
// P = the s8 quantization of S (a stand-in for exp-quantize's traffic: 4 bytes read, 1 written an entry)
__global__ void soft_q(const int4* __restrict__ S, int* __restrict__ P, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    const int4 s = __ldcs(S + i);
    const unsigned b = ((unsigned)(s.x >> 12) & 0xff) | (((unsigned)(s.y >> 12) & 0xff) << 8) |
                       (((unsigned)(s.z >> 12) & 0xff) << 16) | (((unsigned)(s.w >> 12) & 0xff) << 24);
    P[i] = (int)b;
  }
}
static void form(const void* src, void* dst, size_t bytes, uint64_t seed, cudaStream_t st, int sms) {
  form_add<<<sms * 8, 256, 0, st>>>((const int4*)src, (int4*)dst, bytes / 16, seed);
}

// ---------------------------------------------------------------- cuBLASLt int8, strided batched, verified on batch 0
struct LtB {
  Lt L;
  int m, n, k, batch;
};
static LtB lt_make_b(int m, int n, int k, int batch, long long sa, long long sw, long long sd, const void* A, const void* W,
                     void* D, cudaStream_t st) {
  LtB B;
  B.m = m; B.n = n; B.k = k; B.batch = batch;
  Lt& L = B.L;
  CL(cublasLtMatmulDescCreate(&L.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CL(cublasLtMatrixLayoutCreate(&L.la, CUDA_R_8I, k, n, k));
  CL(cublasLtMatrixLayoutCreate(&L.lb, CUDA_R_8I, k, m, k));
  CL(cublasLtMatrixLayoutCreate(&L.ld, CUDA_R_32I, n, m, n));
  if (batch > 1) {
    const int32_t bc = batch;
    CL(cublasLtMatrixLayoutSetAttribute(L.la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof bc));
    CL(cublasLtMatrixLayoutSetAttribute(L.lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof bc));
    CL(cublasLtMatrixLayoutSetAttribute(L.ld, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof bc));
    CL(cublasLtMatrixLayoutSetAttribute(L.la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sw, sizeof sw));
    CL(cublasLtMatrixLayoutSetAttribute(L.lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof sa));
    CL(cublasLtMatrixLayoutSetAttribute(L.ld, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sd, sizeof sd));
  }
  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_wsz, sizeof g_wsz));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  if (cublasLtMatmulAlgoGetHeuristic(g_lt, L.op, L.la, L.lb, L.ld, L.ld, pref, 16, res, &got) != CUBLAS_STATUS_SUCCESS) got = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  float best = 1e30f;
  int bi = -1;
  const int32_t one = 1, zero = 0;
  const int mr = std::min(m, 256), rs = m / mr;
  const size_t nchk = (size_t)mr * n;
  ref_rows<<<(unsigned)((nchk + 255) / 256), 256, 0, st>>>((const int8_t*)A, (const int8_t*)W, g_ref, mr, rs, n, k);
  for (int i = 0; i < got; i++) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    CK(cudaMemsetAsync(D, 0xA5, (size_t)m * n * sizeof(int), st));
    if (cublasLtMatmul(g_lt, L.op, &one, W, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, g_ws, g_wsz, st) !=
        CUBLAS_STATUS_SUCCESS)
      continue;
    CK(cudaMemsetAsync(g_cnt, 0, sizeof *g_cnt, st));
    count_neq_rows<<<(unsigned)((nchk + 255) / 256), 256, 0, st>>>((const int*)D, g_ref, mr, rs, n, g_cnt);
    unsigned long long bad = 0;
    CK(cudaMemcpyAsync(&bad, g_cnt, sizeof bad, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    if (bad) { L.rejected++; continue; }
    CK(cudaEventRecord(e0, st));
    for (int r = 0; r < 3; r++)
      cublasLtMatmul(g_lt, L.op, &one, W, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, g_ws, g_wsz, st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) { best = ms; bi = i; }
  }
  CK(cudaGetLastError());
  if (bi >= 0) {
    L.algo = res[bi].algo;
    L.ms = best / 3;
    size_t wr;
    cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_ID, &L.algo_id, sizeof L.algo_id, &wr);
  } else {
    fprintf(stderr, "cuBLASLt int8: no algorithm ran for m %d n %d k %d batch %d\n", m, n, k, batch);
  }
  CL(cublasLtMatmulPreferenceDestroy(pref));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return B;
}
// a handle and workspace per stream that may run a GEMM at the same time as another
struct LtCtx {
  cublasLtHandle_t h;
  void* ws;
};
static LtCtx lt_ctx() {
  LtCtx c;
  CL(cublasLtCreate(&c.h));
  CK(cudaMalloc(&c.ws, g_wsz));
  return c;
}
static void ltb_run(const LtB& B, const LtCtx& c, const void* A, const void* W, void* D, cudaStream_t st) {
  const int32_t one = 1, zero = 0;
  if (B.L.algo_id == -2) return;
  CL(cublasLtMatmul(c.h, B.L.op, &one, W, B.L.la, A, B.L.lb, &zero, D, B.L.ld, D, B.L.ld, &B.L.algo, c.ws, g_wsz, st));
}
static std::map<std::tuple<int, int, int, int>, LtB> g_plans;
static int g_rejected_total = 0, g_missing = 0;
static const LtB& plan(int m, int n, int k, int batch, long long sa, long long sw, long long sd, const void* A, const void* W,
                       void* D, cudaStream_t st) {
  const auto key = std::make_tuple(m, n, k, batch);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  LtB b = lt_make_b(m, n, k, batch, sa, sw, sd, A, W, D, st);
  g_rejected_total += b.L.rejected;
  g_missing += b.L.algo_id == -2;
  return g_plans.emplace(key, b).first->second;
}

// ---------------------------------------------------------------- timing helpers
struct Graph {
  cudaGraphExec_t ge = nullptr;
  std::map<int, int> prios;  // kernel nodes by the priority the capture recorded
};
// work captured from stream `st` (forks inside it join back before the end), run with each node's captured priority
static Graph capture(cudaStream_t st, const std::function<void()>& work) {
  cudaGraph_t g;
  Graph G;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  work();
  CK(cudaStreamEndCapture(st, &g));
  size_t n = 0;
  CK(cudaGraphGetNodes(g, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  CK(cudaGraphGetNodes(g, nodes.data(), &n));
  for (cudaGraphNode_t nd : nodes) {
    cudaGraphNodeType ty;
    CK(cudaGraphNodeGetType(nd, &ty));
    if (ty != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeAttrValue v{};
    if (cudaGraphKernelNodeGetAttribute(nd, cudaKernelNodeAttributePriority, &v) == cudaSuccess) G.prios[v.priority]++;
  }
  CK(cudaGraphInstantiateWithFlags(&G.ge, g, cudaGraphInstantiateFlagUseNodePriority));
  CK(cudaGraphDestroy(g));
  return G;
}
static void print_prios(const Graph& G) {
  printf("{");
  bool first = true;
  for (const auto& kv : G.prios) {
    printf("%s\"%d\": %d", first ? "" : ", ", kv.first, kv.second);
    first = false;
  }
  printf("}");
}
// reps of the graph on `st`, each timed by events, with the read SM clock and power
static std::vector<Rep> time_graph(Dev& d, const Graph& G, cudaStream_t st, int reps) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int i = 0; i < 2; i++) CK(cudaGraphLaunch(G.ge, st));
  CK(cudaStreamSynchronize(st));
  std::vector<Rep> out;
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(e0, st));
    CK(cudaGraphLaunch(G.ge, st));
    CK(cudaEventRecord(e1, st));
    double cs = 0, ps = 0;
    int nc = 0, np = 0;
    while (cudaEventQuery(e1) == cudaErrorNotReady) {
      const double c = nvml_mhz(d), p = nvml_w(d);
      if (c > 0) { cs += c; nc++; }
      if (p > 0) { ps += p; np++; }
    }
    CK(cudaEventSynchronize(e1));
    if (nc == 0) { cs = nvml_mhz(d); nc = 1; }
    if (np == 0) { ps = nvml_w(d); np = 1; }
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    out.push_back({ms, cs / nc, ps / np});
  }
  CK(cudaGetLastError());
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return out;
}
static double mean_ms(const std::vector<Rep>& v) {
  double s = 0;
  for (const Rep& r : v) s += r.ms;
  return v.empty() ? 0 : s / v.size();
}
static void print_reps(const char* name, const std::vector<Rep>& v, double macs, int sms, bool comma) {
  const Rep m = median(v);
  const double peak = 1024.0 * sms * m.mhz * 1e6;
  printf("\"%s\": {\"ms\": %.4f, \"mean_ms\": %.4f, \"mhz\": %.1f, \"watt\": %.1f, \"share_of_peak\": %.5f, \"reps_ms\": [", name,
         m.ms, mean_ms(v), m.mhz, m.watt, macs > 0 ? macs / (m.ms * 1e-3) / peak : 0.0);
  for (size_t i = 0; i < v.size(); i++) printf("%s%.4f", i ? ", " : "", v[i].ms);
  printf("]}%s", comma ? ", " : "");
}
static cudaStream_t prio_stream(int prio) {
  cudaStream_t s;
  CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, prio));
  return s;
}
static void fork(cudaStream_t from, cudaStream_t to) {
  cudaEvent_t e;
  CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  CK(cudaEventRecord(e, from));
  CK(cudaStreamWaitEvent(to, e, 0));
}

// ---------------------------------------------------------------- prefill: the six stages of 32 layers of one chunk
constexpr int NSTAGE = 6;
static const int SK[NSTAGE] = {HID, QKV, 256, HID, HID, FF};  // the bytes a token of each stage's committed operand
static const char* SNAME[NSTAGE] = {"qkv", "qk", "pv", "o", "gate_up", "down"};
struct Chunk {
  uint8_t* op[NSTAGE];  // each stage's operand rows, M x SK
  int8_t* A;            // the GEMMs' A, M x FF
  int* D;               // their output
  Commit cm[NSTAGE];
  LtCtx lt;
};
struct Prefill {
  int M;
  LtB g[NSTAGE];
  const int8_t* W[LAYERS][NSTAGE];
  int gk[NSTAGE];
};
static void stage_gemm(const Prefill& P, Chunk& c, int l, int t, cudaStream_t st) {
  ltb_run(P.g[t], c.lt, c.A, P.W[l][t], c.D, st);
}
static int run_prefill(Dev& d, const std::string& dir, int M, int reps) {
  const int draw_us = getenv(RUN_DRAW_US) ? atoi(getenv(RUN_DRAW_US)) : 300;
  std::vector<HostConst> hc;
  for (int t = 0; t < NSTAGE; t++) hc.push_back(load_const(dir, SK[t], M));
  Prefill P;
  P.M = M;
  const int GN[NSTAGE] = {QKV, M / 2, M / 2, HID, 2 * FF, HID}, GK[NSTAGE] = {HID, HID, HID, HID, HID, FF};
  for (int l = 0; l < LAYERS; l++) {
    const int8_t* w[NSTAGE] = {d.W[l][0], d.W[l][1], d.W[(l + 1) % LAYERS][1], d.W[l][1], d.W[l][2], d.W[l][3]};
    for (int t = 0; t < NSTAGE; t++) P.W[l][t] = w[t];
  }
  const int NC = 3;
  std::vector<Chunk> ch(NC);
  for (int c = 0; c < NC; c++) {
    for (int t = 0; t < NSTAGE; t++) {
      CK(cudaMalloc(&ch[c].op[t], (size_t)M * SK[t]));
      fill_s8<<<1024, 256, 0, d.st>>>((int8_t*)ch[c].op[t], (size_t)M * SK[t], 0x4000 + 16 * c + t);
      ch[c].cm[t] = make_commit(hc[t]);
    }
    CK(cudaMalloc(&ch[c].A, (size_t)M * FF));
    fill_s8<<<1024, 256, 0, d.st>>>(ch[c].A, (size_t)M * FF, 0x5000 + c);
    CK(cudaMalloc(&ch[c].D, (size_t)M * 2 * FF * sizeof(int)));
    ch[c].lt = lt_ctx();
  }
  CK(cudaStreamSynchronize(d.st));
  for (int t = 0; t < NSTAGE; t++) {
    P.gk[t] = GK[t];
    P.g[t] = plan(M, GN[t], GK[t], 1, 0, 0, 0, ch[0].A, P.W[0][t], ch[0].D, d.st);
  }
  double macs = 0;
  for (int t = 0; t < NSTAGE; t++) macs += (double)LAYERS * M * GN[t] * GK[t];
  int lo, hi;  // least and greatest priority; a lower number is a higher priority
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t s0 = prio_stream(lo), s1 = prio_stream(lo), sm = prio_stream(lo);
  std::vector<cudaStream_t> cs;
  for (int c = 0; c < NC; c++) cs.push_back(prio_stream(std::min(hi + c, lo)));
  auto gemms = [&](Chunk& c, cudaStream_t st) {
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++) stage_gemm(P, c, l, t, st);
  };
  auto hashes = [&](Chunk& c, cudaStream_t st) {
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++) commit_run(c.cm[t], c.op[t], st);
  };
  auto stage_seq = [&](Chunk& c, cudaStream_t st, bool hash, bool draw) {
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++) {
        if (hash) commit_run(c.cm[t], c.op[t], st);
        if (draw) draw_wait<<<1, 1, 0, st>>>((unsigned long long)draw_us * 1000ull);
        stage_gemm(P, c, l, t, st);
      }
  };
  std::vector<std::pair<std::string, std::vector<Rep>>> arms;
  std::vector<double> arm_macs;
  std::vector<Graph> arm_graphs;
  auto add = [&](const std::string& nm, cudaStream_t st, const std::function<void()>& w, double mm) {
    const Graph G = capture(st, w);
    arms.push_back({nm, time_graph(d, G, st, reps)});
    arm_macs.push_back(mm);
    CK(cudaGraphExecDestroy(G.ge));
    arm_graphs.push_back(G);
  };
  add("gemm", sm, [&] { gemms(ch[0], sm); }, macs);
  add("hash", sm, [&] { hashes(ch[0], sm); }, 0);
  add("serial", sm, [&] { stage_seq(ch[0], sm, true, false); }, macs);
  add("conc", sm, [&] {
    fork(sm, s0);
    gemms(ch[0], s0);
    fork(sm, s1);
    hashes(ch[1], s1);
    fork(s0, sm);
    fork(s1, sm);
  }, macs);
  cudaStream_t s1lo = prio_stream(lo), s0hi = prio_stream(hi);
  add("conc_lo", sm, [&] {
    fork(sm, s0hi);
    gemms(ch[0], s0hi);
    fork(sm, s1lo);
    hashes(ch[1], s1lo);
    fork(s0hi, sm);
    fork(s1lo, sm);
  }, macs);
  // hash latency a stage: 32 commitments of one stage's operands back to back
  std::vector<std::pair<std::string, std::vector<Rep>>> lat;
  for (int t = 0; t < NSTAGE; t++) {
    const Graph G = capture(sm, [&] { for (int l = 0; l < LAYERS; l++) commit_run(ch[0].cm[t], ch[0].op[t], sm); });
    lat.push_back({SNAME[t], time_graph(d, G, sm, reps)});
    CK(cudaGraphExecDestroy(G.ge));
  }
  // C chunks in flight, each hash -> draw -> GEMM a stage, chunk 0 at the highest priority
  for (int C = 1; C <= NC; C++) {
    add("pipe" + std::to_string(C), sm, [&] {
      for (int c = 0; c < C; c++) {
        fork(sm, cs[c]);
        stage_seq(ch[c], cs[c], true, true);
      }
      for (int c = 0; c < C; c++) fork(cs[c], sm);
    }, C * macs);
  }
  add("pipe1_nodraw_nohash", sm, [&] { stage_seq(ch[0], sm, false, true); }, macs);
  // the commitments on a green partition of N SMs, the GEMMs on the rest (no graph: the two contexts' streams)
  std::vector<std::tuple<int, int, int, double, double, double>> green;  // N asked, gemm SMs, hash SMs, gemm alone, both, hash alone
  for (int N : {16, 32}) {
    cudaStream_t sg = nullptr, sh = nullptr;
    int got[2] = {0, 0};
    const int e = gc_split(d.sms - N, &sg, &sh, got);
    if (e || !sh) { fprintf(stderr, "gc_split(%d): %d\n", d.sms - N, e); continue; }
    auto host_ms = [&](bool g, bool h) {
      std::vector<double> v;
      for (int r = 0; r < reps + 1; r++) {
        CK(cudaDeviceSynchronize());
        const double t0 = now_ms();
        if (g) gemms(ch[0], sg);
        if (h) hashes(ch[1], sh);
        CK(cudaStreamSynchronize(sg));
        CK(cudaStreamSynchronize(sh));
        const double t1 = now_ms();
        if (r) v.push_back(t1 - t0);
      }
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    const double ga = host_ms(true, false), both = host_ms(true, true), ha = host_ms(false, true);
    green.emplace_back(N, got[0], got[1], ga, both, ha);
  }
  printf("{\"mode\": \"prefill\", \"M\": %d, \"layers\": %d, \"draw_us\": %d, \"chunks_max\": %d, \"macs\": %.6e, "
         "\"op_bytes_token_layer\": %d, \"rejected\": %d, \"missing_algo\": %d, \"gemms\": [",
         M, LAYERS, draw_us, NC, macs, SK[0] + SK[1] + SK[2] + SK[3] + SK[4] + SK[5], g_rejected_total, g_missing);
  for (int t = 0; t < NSTAGE; t++)
    printf("%s{\"stage\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"algo\": %d, \"ms_alone\": %.4f}", t ? ", " : "", SNAME[t], M,
           GN[t], GK[t], P.g[t].L.algo_id, P.g[t].L.ms);
  printf("], \"hash_latency\": {");
  for (size_t i = 0; i < lat.size(); i++) print_reps(lat[i].first.c_str(), lat[i].second, 0, d.sms, i + 1 < lat.size());
  printf("}, \"green\": [");
  for (size_t i = 0; i < green.size(); i++) {
    const auto& g = green[i];
    printf("%s{\"N\": %d, \"gemm_sms\": %d, \"hash_sms\": %d, \"gemm_ms\": %.4f, \"both_ms\": %.4f, \"hash_ms\": %.4f}", i ? ", " : "",
           std::get<0>(g), std::get<1>(g), std::get<2>(g), std::get<3>(g), std::get<4>(g), std::get<5>(g));
  }
  printf("], \"node_priorities\": {");
  for (size_t i = 0; i < arms.size(); i++) {
    printf("%s\"%s\": ", i ? ", " : "", arms[i].first.c_str());
    print_prios(arm_graphs[i]);
  }
  printf("}, \"arms\": {");
  for (size_t i = 0; i < arms.size(); i++) print_reps(arms[i].first.c_str(), arms[i].second, arm_macs[i], d.sms, i + 1 < arms.size());
  printf("}}\n");
  fflush(stdout);
  return 0;
}

// ---------------------------------------------------------------- the deferred pass: one layer's noised products over T
struct Seq {
  int t, ctx, lk;
  int8_t *K, *V;  // [HKV][lk][HD] and [HKV][HD][lk]
};
struct Pass {
  int T;
  bool b32;
  std::vector<Seq> seqs;
  int8_t *X, *Q, *P, *Wf, *Xf, *KVf;
  int *D, *S, *O;
  size_t lin_rows_max;
  double lin_macs = 0, att_macs = 0, form_bytes = 0, stored_bytes_token_layer = 0;
  LtCtx lt;
};
static int ctx_of(int i, int n) {
  if (n <= 1) return 4608;
  return 1024 + (int)std::lround((8192.0 - 1024.0) * i / (n - 1) / 512.0) * 512;
}
static Pass make_pass(Dev& d, int T, bool b32) {
  Pass p;
  p.T = T;
  p.b32 = b32;
  const int nseq = b32 ? 32 : std::max(1, T / 512), t = T / nseq;
  size_t smax = 0, kvmax = 0;
  for (int i = 0; i < nseq; i++) {
    Seq s;
    s.t = t;
    s.ctx = ctx_of(i, nseq);
    s.lk = (s.ctx + t / 2 + 31) / 32 * 32;
    const size_t kv = (size_t)HKV * s.lk * HD;
    CK(cudaMalloc(&s.K, kv));
    CK(cudaMalloc(&s.V, kv));
    fill_s8<<<1024, 256, 0, d.st>>>(s.K, kv, 0x6000 + 2 * i);
    fill_s8<<<1024, 256, 0, d.st>>>(s.V, kv, 0x6001 + 2 * i);
    smax = std::max(smax, (size_t)HKV * 4 * t * s.lk);
    kvmax = std::max(kvmax, kv);
    p.seqs.push_back(s);
    p.att_macs += 3.0 * HKV * 4.0 * t * s.lk * HD;
  }
  p.lin_rows_max = std::min(T, 8192);
  CK(cudaMalloc(&p.X, p.lin_rows_max * FF));
  fill_s8<<<1024, 256, 0, d.st>>>(p.X, p.lin_rows_max * FF, 0x7000);
  CK(cudaMalloc(&p.Xf, p.lin_rows_max * FF));
  CK(cudaMalloc(&p.D, p.lin_rows_max * 2 * FF * sizeof(int)));
  CK(cudaMalloc(&p.Wf, (size_t)2 * FF * HID));
  CK(cudaMalloc(&p.KVf, kvmax));
  CK(cudaMalloc(&p.Q, (size_t)HKV * 4 * t * HD));
  fill_s8<<<1024, 256, 0, d.st>>>(p.Q, (size_t)HKV * 4 * t * HD, 0x7001);
  CK(cudaMalloc(&p.P, smax));
  fill_s8<<<1024, 256, 0, d.st>>>(p.P, smax, 0x7002);
  CK(cudaMalloc(&p.S, smax * sizeof(int)));
  CK(cudaMalloc(&p.O, (size_t)HKV * 4 * t * HD * sizeof(int)));
  for (int j = 0; j < 4; j++) p.lin_macs += (double)T * LN[j] * LK[j];
  // what the deferred edit keeps from the served pass, a token-layer: QKV's input, Q (K and V are the cache), attention's
  // output, gate_up's and down's inputs, and the row normalizers (m, Z) of the 32 query heads
  p.stored_bytes_token_layer = HID + HQ * HD + HID + HID + FF + HQ * 2 * 4;
  p.lt = lt_ctx();
  CK(cudaStreamSynchronize(d.st));
  return p;
}
// one layer of the pass on stream st; plans are made (and cached) on d.st beforehand by pass_plans
static void pass_layer(Dev& d, Pass& p, int layer, bool forming, cudaStream_t st, bool plan_only = false) {
  const int8_t* W[4] = {d.W[layer][0], d.W[layer][1], d.W[layer][2], d.W[layer][3]};
  p.form_bytes = 0;
  for (int j = 0; j < 4; j++) {
    if (forming && !plan_only) form(W[j], p.Wf, (size_t)LN[j] * LK[j], 0x100 + j, st, d.sms);
    p.form_bytes += 2.0 * LN[j] * LK[j];
    for (int r0 = 0; r0 < p.T; r0 += (int)p.lin_rows_max) {
      const int m = std::min((int)p.lin_rows_max, p.T - r0);
      if (forming && !plan_only) form(p.X, p.Xf, (size_t)m * LK[j], 0x200 + j, st, d.sms);
      p.form_bytes += 2.0 * m * LK[j];
      const LtB& B = plan(m, LN[j], LK[j], 1, 0, 0, 0, p.X, W[j], p.D, d.st);
      if (!plan_only) ltb_run(B, p.lt, forming ? p.Xf : p.X, forming ? p.Wf : W[j], p.D, st);
    }
  }
  for (const Seq& s : p.seqs) {
    const int m = 4 * s.t;
    const size_t kv = (size_t)HKV * s.lk * HD;
    const LtB& QK = plan(m, s.lk, HD, HKV, (long long)m * HD, (long long)s.lk * HD, (long long)m * s.lk, p.Q, s.K, p.S, d.st);
    const LtB& PV = plan(m, HD, s.lk, HKV, (long long)m * s.lk, (long long)HD * s.lk, (long long)m * HD, p.P, s.V, p.O, d.st);
    if (plan_only) continue;
    ltb_run(QK, p.lt, p.Q, s.K, p.S, st);  // clean, to rebuild P
    if (forming) {
      const size_t sn = (size_t)HKV * m * s.lk;
      soft_q<<<d.sms * 8, 256, 0, st>>>((const int4*)p.S, (int*)p.P, sn / 4);
      form(s.K, p.KVf, kv, 0x300, st, d.sms);
      form(s.V, p.KVf, kv, 0x301, st, d.sms);
      form(p.P, p.P, sn, 0x302, st, d.sms);
    }
    ltb_run(QK, p.lt, p.Q, forming ? p.KVf : s.K, p.S, st);
    ltb_run(PV, p.lt, p.P, forming ? p.KVf : s.V, p.O, st);
  }
  if (forming) {
    double att = 0;
    for (const Seq& s : p.seqs) att += 4.0 * HKV * s.lk * HD + (double)HKV * 4 * s.t * s.lk * 7;
    p.form_bytes += att;
  }
}
static const int PASS_LAYERS = 4;
static void print_pass(const Pass& p) {
  int nseq = (int)p.seqs.size();
  double ctx = 0;
  for (const Seq& s : p.seqs) ctx += s.ctx;
  printf("\"T\": %d, \"comp\": \"%s\", \"seqs\": %d, \"tokens_a_seq\": %d, \"ctx_mean\": %.0f, \"lin_macs\": %.6e, \"att_macs\": %.6e, "
         "\"stored_bytes_token_layer\": %.0f, \"stored_bytes_replica\": %.6e, \"form_bytes\": %.6e",
         p.T, p.b32 ? "b32" : "seq500", nseq, p.seqs[0].t, ctx / nseq, p.lin_macs, p.att_macs, p.stored_bytes_token_layer,
         p.stored_bytes_token_layer * LAYERS * p.T, p.form_bytes);
}
static int run_deferred(Dev& d, int T, bool b32, int reps) {
  Pass p = make_pass(d, T, b32);
  pass_layer(d, p, 0, true, d.st, true);
  cudaStream_t st = prio_stream(0);
  printf("{\"mode\": \"deferred\", \"layers_timed\": %d, ", PASS_LAYERS);
  print_pass(p);
  printf(", \"rejected\": %d, \"missing_algo\": %d, \"arms\": {", g_rejected_total, g_missing);
  const double macs = (p.lin_macs + p.att_macs) * PASS_LAYERS;
  int i = 0;
  for (bool forming : {false, true}) {
    const Graph G = capture(st, [&] { for (int l = 0; l < PASS_LAYERS; l++) pass_layer(d, p, l, forming, st); });
    print_reps(forming ? "form" : "gemm", time_graph(d, G, st, reps), macs, d.sms, i++ == 0);
    CK(cudaGraphExecDestroy(G.ge));
  }
  // the linears alone and attention alone
  {
    Pass q = p;
    q.seqs.clear();
    const Graph G = capture(st, [&] { for (int l = 0; l < PASS_LAYERS; l++) pass_layer(d, q, l, false, st); });
    printf(", ");
    print_reps("lin", time_graph(d, G, st, reps), p.lin_macs * PASS_LAYERS, d.sms, true);
    CK(cudaGraphExecDestroy(G.ge));
    const Graph A = capture(st, [&] {
      for (int l = 0; l < PASS_LAYERS; l++)
        for (const Seq& s : p.seqs) {
          const int m = 4 * s.t;
          const LtB& QK = plan(m, s.lk, HD, HKV, 0, 0, 0, nullptr, nullptr, nullptr, d.st);
          const LtB& PV = plan(m, HD, s.lk, HKV, 0, 0, 0, nullptr, nullptr, nullptr, d.st);
          ltb_run(QK, p.lt, p.Q, s.K, p.S, st);
          ltb_run(QK, p.lt, p.Q, s.K, p.S, st);
          ltb_run(PV, p.lt, p.P, s.V, p.O, st);
        }
    });
    print_reps("att", time_graph(d, A, st, reps), p.att_macs * PASS_LAYERS, d.sms, false);
    CK(cudaGraphExecDestroy(A.ge));
  }
  printf("}}\n");
  fflush(stdout);
  return 0;
}

// a host thread keeps `depth` launches of graph G queued on stream fs and notes when each is seen done
struct PumpG {
  std::thread th;
  std::atomic<bool> on{false};
  std::atomic<long long> done{0};
  std::vector<double> t;
};
static void pumpg_start(PumpG& p, const Graph& G, cudaStream_t fs, int depth) {
  p.on = true;
  p.done = 0;
  p.t.clear();
  p.th = std::thread([&p, &G, fs, depth] {
    std::vector<cudaEvent_t> ev(depth);
    for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync));
    for (long long k = 0; p.on.load(); k++) {
      const int slot = (int)(k % depth);
      if (k >= depth) {
        CK(cudaEventSynchronize(ev[slot]));
        p.t.push_back(now_ms());
        p.done++;
      }
      CK(cudaGraphLaunch(G.ge, fs));
      CK(cudaEventRecord(ev[slot], fs));
    }
    CK(cudaStreamSynchronize(fs));
    for (auto& e : ev) CK(cudaEventDestroy(e));
  });
}
static void pumpg_stop(PumpG& p) {
  p.on = false;
  p.th.join();
}
static long long items_in(const PumpG& p, double t0, double t1) {
  long long n = 0;
  for (double x : p.t) n += x > t0 && x <= t1;
  return n;
}
// the step on sst, and beside each step per_step of the pass's layers (credit carried across steps), so the pass gets exactly
// the token-layers decode produces; the mean step over whole periods, and how long the pass's stream takes to drain after
// the last step (a pass that can't keep up leaves a backlog that grows with the reps)
static std::vector<Rep> time_paced(Dev& d, const std::function<void()>& step, cudaStream_t sst, const Graph& Gp, cudaStream_t pst,
                                   double per_step, int reps, long long* launched, double* tail_ms) {
  d.st = sst;
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(sst, cudaStreamCaptureModeThreadLocal));
  step();
  CK(cudaStreamEndCapture(sst, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 2; i++) CK(cudaGraphLaunch(ge, sst));
  CK(cudaStreamSynchronize(sst));
  cudaEvent_t e0, e1, ep;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventCreate(&ep));
  std::vector<Rep> out;
  double credit = 0;
  long long n = 0;
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(e0, sst));
    CK(cudaGraphLaunch(ge, sst));
    CK(cudaEventRecord(e1, sst));
    credit += per_step;
    while (credit >= 1 - 1e-9) {
      CK(cudaGraphLaunch(Gp.ge, pst));
      credit -= 1;
      n++;
    }
    double cs = 0, ps = 0;
    int nc = 0, np = 0;
    while (cudaEventQuery(e1) == cudaErrorNotReady) {
      const double c = nvml_mhz(d), p = nvml_w(d);
      if (c > 0) { cs += c; nc++; }
      if (p > 0) { ps += p; np++; }
    }
    CK(cudaEventSynchronize(e1));
    if (nc == 0) { cs = nvml_mhz(d); nc = 1; }
    if (np == 0) { ps = nvml_w(d); np = 1; }
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    out.push_back({ms, cs / nc, ps / np});
  }
  const double t = now_ms();
  CK(cudaEventRecord(ep, pst));
  CK(cudaEventSynchronize(ep));
  *tail_ms = now_ms() - t;
  *launched = n;
  CK(cudaGetLastError());
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  CK(cudaEventDestroy(ep));
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  return out;
}
static int run_beside(Dev& d, int T, bool b32, int B, int L, int fill_sms, bool forming, int reps) {
  Pass p = make_pass(d, T, b32);
  pass_layer(d, p, 0, forming, d.st, true);
  const std::vector<Lt> lp = lin_plans(d, B);
  bool wrapped = false;
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t full = d.st, s_hi = prio_stream(hi), s_lo = prio_stream(lo);
  auto step = [&] { enqueue_step(d, lp, 2, B, L, &wrapped); };
  auto pass_graph = [&](cudaStream_t st) {
    return capture(st, [&] { pass_layer(d, p, 1, forming, st); });
  };
  const double pass_macs = p.lin_macs + p.att_macs;
  // the pass alone on the whole GPU
  const Graph Gp = pass_graph(s_lo);
  const std::vector<Rep> pa = time_graph(d, Gp, s_lo, std::max(3, reps / 3));
  d.st = full;
  const std::vector<Rep> a_full = time_work(d, step, reps);
  // the step at high priority, the pass kept queued at low priority
  d.st = s_hi;
  PumpG pump;
  pumpg_start(pump, Gp, s_lo, 2);
  while (pump.done.load() < 1) {
  }
  double t0 = 0, t1 = 0;
  const std::vector<Rep> a_prio = time_work(d, step, reps, &t0, &t1);
  pumpg_stop(pump);
  const long long n_prio = items_in(pump, t0, t1);
  const double w_prio = t1 - t0;
  const double per_step = (double)B * LAYERS / T;
  const int preps = std::min(256, std::max(reps, (int)std::ceil(4.0 / per_step)));
  long long pl_prio = 0, pl_green = 0;
  double tail_prio = 0, tail_green = 0;
  const std::vector<Rep> a_pprio = time_paced(d, step, s_hi, Gp, s_lo, per_step, preps, &pl_prio, &tail_prio);
  // a green split: the step on the rest, the pass on fill_sms SMs
  cudaStream_t ss = nullptr, fs = nullptr;
  int got[2] = {0, 0};
  const int e = gc_split(d.sms - fill_sms, &ss, &fs, got);
  std::vector<Rep> a_split, a_green, a_pgreen;
  long long n_green = 0;
  double w_green = 0;
  std::vector<Rep> pg;
  if (!e && fs) {
    d.st = ss;
    a_split = time_work(d, step, reps);
    const Graph Gg = pass_graph(fs);
    pg = time_graph(d, Gg, fs, std::max(3, reps / 3));
    pumpg_start(pump, Gg, fs, 2);
    while (pump.done.load() < 1) {
    }
    a_green = time_work(d, step, reps, &t0, &t1);
    pumpg_stop(pump);
    n_green = items_in(pump, t0, t1);
    w_green = t1 - t0;
    a_pgreen = time_paced(d, step, ss, Gg, fs, per_step, preps, &pl_green, &tail_green);
    CK(cudaGraphExecDestroy(Gg.ge));
  } else {
    fprintf(stderr, "gc_split(%d): %d\n", d.sms - fill_sms, e);
  }
  d.st = full;
  const std::vector<Rep> a_full2 = time_work(d, step, reps);
  double step_macs = 0;
  for (int j = 0; j < 4; j++) step_macs += (double)LAYERS * B * LN[j] * LK[j];
  step_macs += (double)B * VOCAB * HID + 2.0 * LAYERS * B * HQ * HD * (double)L;
  const double base = (median(a_full).ms + median(a_full2).ms) / 2;
  printf("{\"mode\": \"beside\", \"B\": %d, \"L\": %d, \"forming\": %s, \"fill_sms_req\": %d, \"step_sms\": %d, \"fill_sms\": %d, "
         "\"kv_wrapped\": %s, \"step_macs\": %.6e, \"pass_macs_layer\": %.6e, ",
         B, L, forming ? "true" : "false", fill_sms, got[0], got[1], wrapped ? "true" : "false", step_macs, pass_macs);
  print_pass(p);
  printf(", \"rejected\": %d, \"missing_algo\": %d", g_rejected_total, g_missing);
  printf(", \"prio\": {\"window_ms\": %.3f, \"pass_layers\": %lld, \"slowdown\": %.4f}", w_prio, n_prio, median(a_prio).ms / base);
  printf(", \"green\": {\"window_ms\": %.3f, \"pass_layers\": %lld, \"slowdown\": %.4f, \"slowdown_vs_split\": %.4f}", w_green, n_green,
         a_green.empty() ? 0.0 : median(a_green).ms / base,
         a_green.empty() ? 0.0 : median(a_green).ms / median(a_split).ms);
  const double base_mean = (mean_ms(a_full) + mean_ms(a_full2)) / 2;
  printf(", \"paced\": {\"per_step\": %.6f, \"reps\": %d, \"prio\": {\"pass_layers\": %lld, \"tail_ms\": %.3f, \"slowdown_mean\": %.4f}, "
         "\"green\": {\"pass_layers\": %lld, \"tail_ms\": %.3f, \"slowdown_mean\": %.4f}}",
         per_step, preps, pl_prio, tail_prio, mean_ms(a_pprio) / base_mean, pl_green, tail_green,
         a_pgreen.empty() ? 0.0 : mean_ms(a_pgreen) / base_mean);
  printf(", \"arms\": {");
  print_reps("pass_alone", pa, pass_macs, d.sms, true);
  if (!pg.empty()) print_reps("pass_alone_green", pg, pass_macs, d.sms, true);
  print_reps("full", a_full, step_macs, d.sms, true);
  print_reps("prio", a_prio, step_macs, d.sms, true);
  print_reps("paced_prio", a_pprio, step_macs, d.sms, true);
  if (!a_split.empty()) print_reps("split", a_split, step_macs, d.sms, true);
  if (!a_green.empty()) print_reps("green", a_green, step_macs, d.sms, true);
  if (!a_pgreen.empty()) print_reps("paced_green", a_pgreen, step_macs, d.sms, true);
  print_reps("full2", a_full2, step_macs, d.sms, false);
  printf("}}\n");
  fflush(stdout);
  CK(cudaGraphExecDestroy(Gp.ge));
  return 0;
}

// ---------------------------------------------------------------- the gate: commitments, then dcap's GEMM and attention gate
static int run_gate(Dev& d, const std::string& dir, const std::vector<int>& Ms) {
  int fails = 0;
  for (int M : Ms)
    for (int t = 0; t < NSTAGE; t++) {
      if (t == 4 || t == 3) continue;  // the same width as stage 0
      const HostConst c = load_const(dir, SK[t], M);
      const Commit m = make_commit(c);
      const bool ok = commit_gate(m, true), control = commit_gate(m, false);
      printf("{\"mode\": \"gate\", \"what\": \"hm96-sha512 row-seg frame-v3-sha512\", \"k\": %d, \"R\": %d, \"s\": %d, \"ok\": %s, "
             "\"control_rejected\": %s}\n",
             c.k, c.R, c.s, ok ? "true" : "false", control ? "false" : "true");
      fails += !ok || control;
    }
  fflush(stdout);
  return fails + gate(d);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: stage gate DIR | prefill DIR M | deferred T seq500|b32 | beside T seq500|b32 B L FILL_SMS form|gemm  [reps]\n");
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
  fprintf(stderr, "device %s %s, %d SMs\n", name, uuid, d.sms);
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
  const size_t maxBL = 786432;
  d.kv_bytes = (size_t)1 << 30;
  if (mode == "beside" && argc > 6) d.kv_bytes = (size_t)atoi(argv[4]) * atoi(argv[5]) * KV_TOKEN_BYTES + ((size_t)1 << 20);
  CK(cudaMalloc(&d.kv, d.kv_bytes));
  fill_s8<<<4096, 256, 0, d.st>>>(d.kv, d.kv_bytes, 0x99);
  CK(cudaMalloc(&d.q, (size_t)256 * HQ * HD));
  fill_s8<<<64, 256, 0, d.st>>>(d.q, (size_t)256 * HQ * HD, 0x31);
  CK(cudaMalloc(&d.P, maxBL * HQ));
  fill_s8<<<1024, 256, 0, d.st>>>(d.P, maxBL * HQ, 0x32);
  CK(cudaMalloc(&d.S, maxBL * HQ * sizeof(int)));
  CK(cudaMalloc(&d.O, (size_t)256 * HQ * HD * sizeof(int)));
  CK(cudaMalloc(&d.sink, 64));
  CK(cudaStreamSynchronize(d.st));
  CK(cudaGetLastError());
  int rc = 0;
  if (mode == "gate") {
    if (argc < 3) return 1;
    hm_load(argv[2]);
    std::vector<int> Ms;
    for (int i = 3; i < argc; i++) Ms.push_back(atoi(argv[i]));
    if (Ms.empty()) Ms = {2048, 8192};
    rc = run_gate(d, argv[2], Ms);
  } else if (mode == "prefill") {
    if (argc < 4) return 1;
    hm_load(argv[2]);
    rc = run_prefill(d, argv[2], atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 5);
  } else if (mode == "deferred") {
    if (argc < 4) return 1;
    rc = run_deferred(d, atoi(argv[2]), strcmp(argv[3], "b32") == 0, argc > 4 ? atoi(argv[4]) : 5);
  } else if (mode == "beside") {
    if (argc < 8) return 1;
    const int B = atoi(argv[4]), L = atoi(argv[5]);
    if ((size_t)B * L > maxBL || L % 512 || B > 256) { fprintf(stderr, "B*L <= %zu, L %% 512 == 0, B <= 256\n", maxBL); return 1; }
    rc = run_beside(d, atoi(argv[2]), strcmp(argv[3], "b32") == 0, B, L, atoi(argv[6]), strcmp(argv[7], "form") == 0,
                    argc > 8 ? atoi(argv[8]) : 21);
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }
  return rc;
}
