// The e2e slice's GPU engine: Llama-3.1-8B o_proj (m = 8192, n = k = 4096) served by lpgemm.cu's td route at LP_UW 64
// (A' = A + U W^T with U in {±1..±64} and A in [-63, 63], B' = B + F with F ±1, both SHA-512 of the call's sigma; duo receipts
// over A' B'^T; A B'^T served after the tail's peel), and a second stage on its own B (the two-stage slice), driven line by
// line on stdin by prover.py. STAGE (0 or 1, default 0) names the stage's problem, its B and its slots:
//   setup M N K BSEED BPATH [KIND [STAGE [SIDE]]]   the stage's B (n x k int8 in [-119, 119], seeded) on the device and to
//                                    BPATH; plans for every slot; KIND duo (a thread's 8 words over the whole of K, R
//                                    [tile][thread][8]) or duos (a word a thread and 32-step segment, R [tile][K / 32][thread],
//                                    fgemm.cu's DUOS). SIDE a (the default): a slot holds A, the peeled side; SIDE b: a slot
//                                    holds B, the served side, and the stage's fixed operand is A (m x k in [-63, 63], seeded,
//                                    to BPATH), so the per-call operand carries F (or, built -DLP_DITHER, the dither b)
//   opnds SLOT ASEED APATH [STAGE]   slot's operand (SIDE a: m x k int8 in [-63, 63]; SIDE b: n x k in [-119, 119], seeded) on
//                                    the device and to APATH, before its sigma
//   aload SLOT APATH [STAGE]         slot's operand from APATH
//   call SLOT S0 S1 S2 S3 RSEED RPATH FUSED [STAGE]   the edited GEMM on slot's A with sigma S0..S3 (hex u64, big-endian
//                                    words) and the receipts' key RSEED; receipts to RPATH; plain cuBLASLt on the same A, the
//                                    order alternating
//   tile SLOT TM TN PATH [STAGE]     the served product's tile (128 x 128 s32) to PATH
//   ydump SLOT PATH [STAGE]          the served product (m x n s32) to PATH
//   pa PATH [STAGE]                  P_A (m x 128, [U U U U]) of the stage's last call to PATH
// each answered by one JSON line on stdout. Other modes, outside the slice:
//   slice bench REPS SHAPE...        lpgemm.cu's interleaved timing (LP_ARMS)
//   slice commit REPS M K            the operand commitment's SHA-512 work per call on the GPU (commit_cost)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a slice.cu -lcuda -lcublasLt -lnvidia-ml; with -DLP_DITHER
// B's noise is decision 29's dither b = (1 - F) / 2 in {0, 1} (lpgemm.cu's fb_word), for SIDE b's served activations
#define LP_LIB
#define LP_UW 64
#include "../masked_gemm/lpgemm.cu"
#ifdef LP_DITHER
constexpr int LP_DITHER_ON = 1;
#else
constexpr int LP_DITHER_ON = 0;
#endif

#include <iostream>
#include <sstream>

constexpr int SLOTS = 8, STAGES_ = 2;

struct Stage {
  LProb P;
  bool on = false, duos = false, sb = false;   // sb: a slot holds B (SIDE b)
  size_t rb = 0, slot_bytes = 0;
  uint8_t* A[SLOTS] = {};   // the slots' operands, A or (sb) B
  int* Y[SLOTS] = {};
  uint32_t* R[SLOTS] = {};
  LtG plain[SLOTS];
  int* Yp = nullptr;
  uint8_t *A0 = nullptr, *B0 = nullptr;
  int* U0 = nullptr;
  uint32_t* R0 = nullptr;
};

struct Slice {
  Stage G[STAGES_];
  cudaEvent_t e[4];
  bool events = false;
  int calls = 0;
};

static void setup(Slice& S, Stage& G, int m, int n, int k, uint32_t bseed, const std::string& bpath, bool duos, bool sb) {
  LProb& P = G.P;
  lsetup(P, m, n, k, 1, bseed, 0, false);
  if (sb) sfill<I8><<<4096, 256, 0, P.st>>>(P.A, P.na, mix32(bseed ^ 0x2222u), -63, 127);
  else sfill<I8><<<4096, 256, 0, P.st>>>(P.B, P.nb, mix32(bseed ^ 0x2222u));
  G.duos = duos;
  G.sb = sb;
  G.slot_bytes = sb ? P.nb : P.na;
  G.rb = duos ? (size_t)(m / 128) * (n / 128) * CNT * (k / 32) * 4 : P.rb;
  CK(cudaMalloc(&G.Yp, P.ub));
  for (int s = 0; s < SLOTS; s++) {
    CK(cudaMalloc(&G.A[s], G.slot_bytes)); CK(cudaMalloc(&G.Y[s], P.ub)); CK(cudaMalloc(&G.R[s], G.rb));
    G.plain[s] = sb ? ltg(I8, m, n, k, k, k, n, P.A, G.A[s], G.Yp, 0, P.st, 1) : ltg(I8, m, n, k, k, k, n, G.A[s], P.B, G.Yp, 0, P.st, 1);
  }
  if (!S.events) for (auto& ev : S.e) CK(cudaEventCreate(&ev));
  S.events = true;
  CK(cudaStreamSynchronize(P.st));
  if (sb) dump("", bpath.c_str(), P.A, P.na, true);
  else dump("", bpath.c_str(), P.B, P.nb, true);
  G.A0 = P.A; G.B0 = P.B; G.U0 = P.U; G.R0 = P.R;
  G.on = true;
}

static double ms_between(cudaEvent_t a, cudaEvent_t b) {
  float ms;
  CK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

// the td route (lrun's R_T with duo 32, or DUOS) on slot s, with the given sigma and receipt key
static void edited(Stage& G, int s, const Sig& sg, uint32_t rseed, bool fused) {
  LProb& P = G.P;
  (G.sb ? P.B : P.A) = G.A[s];
  P.U = G.Y[s]; P.R = G.R[s]; P.seed = rseed;
  bside(P, fused, false, sg);
  formA(P, false);
  lgemm(P, G.duos ? (Kern)t_duos32 : (Kern)t_duo32, P.mT, sg);
}

static bool load(const std::string& path, void* dev, size_t bytes) {
  std::vector<uint8_t> h(bytes);
  FILE* f = fopen(path.c_str(), "rb");
  const bool ok = f && fread(h.data(), 1, bytes, f) == bytes;
  if (f) fclose(f);
  if (ok) CK(cudaMemcpy(dev, h.data(), bytes, cudaMemcpyHostToDevice));
  return ok;
}

// the optional trailing STAGE of a command, checked
static Stage* stage_of(Slice& S, std::istringstream& in, bool need_on = true) {
  int g = 0;
  if (!(in >> g)) g = 0;
  if (g < 0 || g >= STAGES_ || (need_on && !S.G[g].on)) return nullptr;
  return &S.G[g];
}

static void serve(Slice& S) {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    if (cmd == "quit") break;
    if (cmd == "setup") {
      int m, n, k; uint32_t bs; std::string bp, kind = "duo", side = "a";
      in >> m >> n >> k >> bs >> bp;
      if (!(in >> kind)) kind = "duo";
      Stage* G = stage_of(S, in, false);
      if (!(in >> side)) side = "a";
      if (!G || G->on || (kind != "duo" && kind != "duos") || (side != "a" && side != "b")) {
        printf("{\"error\": \"setup: a fresh stage, KIND duo or duos, SIDE a or b\"}\n"); fflush(stdout); continue;
      }
      setup(S, *G, m, n, k, bs, bp, kind == "duos", side == "b");
      int nsm; cudaDeviceProp prop;
      CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
      CK(cudaGetDeviceProperties(&prop, 0));
      printf("{\"ok\": \"setup\", \"m\": %d, \"n\": %d, \"k\": %d, \"grid\": %d, \"sms\": %d, \"gpu\": \"%s\", \"CNT\": %d, "
             "\"receipt_bytes\": %zu, \"kind\": \"%s\", \"side\": \"%s\", \"dither\": %d, \"lt_algo\": %d}\n", m, n, k, G->P.grid, nsm,
             prop.name, CNT, G->rb, kind.c_str(), side.c_str(), LP_DITHER_ON, G->plain[0].algo_id);
    } else if (cmd == "opnds") {
      int s; uint32_t seed; std::string ap;
      in >> s >> seed >> ap;
      Stage* G = stage_of(S, in);
      if (!G) { printf("{\"error\": \"opnds: no such stage\"}\n"); fflush(stdout); continue; }
      if (G->sb) sfill<I8><<<4096, 256, 0, G->P.st>>>(G->A[s], G->slot_bytes, mix32(seed ^ 0x1111u));
      else sfill<I8><<<4096, 256, 0, G->P.st>>>(G->A[s], G->slot_bytes, mix32(seed ^ 0x1111u), -63, 127);
      CK(cudaStreamSynchronize(G->P.st));
      dump("", ap.c_str(), G->A[s], G->slot_bytes, true);
      printf("{\"ok\": \"opnds\", \"slot\": %d}\n", s);
    } else if (cmd == "aload") {
      int s; std::string ap;
      in >> s >> ap;
      Stage* G = stage_of(S, in);
      if (!G || !load(ap, G->A[s], G->slot_bytes)) { printf("{\"error\": \"aload: no such stage or file\"}\n"); fflush(stdout); continue; }
      printf("{\"ok\": \"aload\", \"slot\": %d}\n", s);
    } else if (cmd == "call") {
      int s; std::string w[4]; uint32_t rseed; std::string rp; int fused = 0;
      in >> s >> w[0] >> w[1] >> w[2] >> w[3] >> rseed >> rp >> fused;
      Stage* G = stage_of(S, in);
      if (!G) { printf("{\"error\": \"call: no such stage\"}\n"); fflush(stdout); continue; }
      LProb& P = G->P;
      Sig sg;
      for (int i = 0; i < 4; i++) sg.w[i] = strtoull(w[i].c_str(), nullptr, 16);
      const bool plain_first = S.calls++ % 2;
      if (plain_first) { CK(cudaEventRecord(S.e[2], P.st)); ltg_run(G->plain[s], P.st); CK(cudaEventRecord(S.e[3], P.st)); }
      CK(cudaEventRecord(S.e[0], P.st));
      edited(*G, s, sg, rseed, fused);
      CK(cudaEventRecord(S.e[1], P.st));
      if (!plain_first) { CK(cudaEventRecord(S.e[2], P.st)); ltg_run(G->plain[s], P.st); CK(cudaEventRecord(S.e[3], P.st)); }
      CK(cudaStreamSynchronize(P.st));
      CK(cudaGetLastError());
      const double te = ms_between(S.e[0], S.e[1]), tp = ms_between(S.e[2], S.e[3]);
      dump("", rp.c_str(), G->R[s], G->rb, true);
      printf("{\"ok\": \"call\", \"slot\": %d, \"edited_ms\": %.6f, \"plain_ms\": %.6f, \"plain_first\": %d}\n", s, te, tp, plain_first);
    } else if (cmd == "tile") {
      int s, tm, tn; std::string p;
      in >> s >> tm >> tn >> p;
      Stage* G = stage_of(S, in);
      if (!G) { printf("{\"error\": \"tile: no such stage\"}\n"); fflush(stdout); continue; }
      std::vector<int> h((size_t)128 * 128);
      CK(cudaMemcpy2D(h.data(), 128 * 4, G->Y[s] + (size_t)tm * 128 * G->P.n + tn * 128, (size_t)G->P.n * 4, 128 * 4, 128,
                      cudaMemcpyDeviceToHost));
      dump("", p.c_str(), h.data(), h.size() * 4, false);
      printf("{\"ok\": \"tile\"}\n");
    } else if (cmd == "ydump") {
      int s; std::string p;
      in >> s >> p;
      Stage* G = stage_of(S, in);
      if (!G) { printf("{\"error\": \"ydump: no such stage\"}\n"); fflush(stdout); continue; }
      dump("", p.c_str(), G->Y[s], G->P.ub, true);
      printf("{\"ok\": \"ydump\"}\n");
    } else if (cmd == "pa") {
      std::string p;
      in >> p;
      Stage* G = stage_of(S, in);
      if (!G) { printf("{\"error\": \"pa: no such stage\"}\n"); fflush(stdout); continue; }
      dump("", p.c_str(), G->P.PA, (size_t)G->P.m * 128, true);
      printf("{\"ok\": \"pa\"}\n");
    } else {
      printf("{\"error\": \"unknown command %s\"}\n", cmd.c_str());
    }
    fflush(stdout);
  }
  for (Stage& G : S.G)
    if (G.on) { G.P.A = G.A0; G.P.B = G.B0; G.P.U = G.U0; G.P.R = G.R0; }
}

// The operand commitment's per-call SHA-512 work, as an hm96-sha512/row/v2 committer would do it (its work, not its
// bytes): a leaf a row i and step j, the 32 int8 at A[i, 32 j ..] as bf16 twice (128 bytes, the spot check's x row). Per
// leaf: the salt from a key (3 compressions, 192 bytes), the row digest (2), c (2, the salt prefix a midstate), b = x ^ M y
// by 192 lookups of 64-byte table rows (M's byte tables, 3 MB), the hm96 tree leaf (2), the frame-v3 leaf (2) and the
// leaf's share of the tree's nodes (2); 64 bytes out a leaf.
__global__ void __launch_bounds__(256) commit_cost(const uint8_t* __restrict__ A, int K, uint32_t leaves, const uint4* __restrict__ tab,
                                                   const __grid_constant__ Sig key, uint4* __restrict__ out) {
  const uint32_t leaf = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf >= leaves) return;
  const uint32_t steps = K / 32, i = leaf / steps, j = leaf % steps;
  const uint4* src = reinterpret_cast<const uint4*>(A + (size_t)i * K + 32 * j);
  const uint4 a0 = src[0], a1 = src[1];
  const uint32_t aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  uint64_t w[16], st[8], x[8], y[24], c[8];
#pragma unroll
  for (int q = 0; q < 8; q++) {   // 4 int8 -> 4 bf16 words = one u64, twice
    uint64_t v = 0;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      const float f = (float)(int8_t)(aw[q] >> (8 * t));
      v |= (uint64_t)(__float_as_uint(f) >> 16) << (16 * t);
    }
    w[q] = w[q + 8] = v;
  }
  rs_iv(st);
  rs_compress(st, w);
#pragma unroll
  for (int q = 0; q < 16; q++) w[q] = q == 0 ? 0x8000000000000000ull : q == 15 ? 1024 + 128 : 0;
  rs_compress(st, w);
#pragma unroll
  for (int q = 0; q < 8; q++) x[q] = st[q];
#pragma unroll
  for (int r = 0; r < 3; r++) {
    sha_block(key.w, (uint64_t)leaf << 2 | r, st);
#pragma unroll
    for (int q = 0; q < 8; q++) y[8 * r + q] = st[q];
  }
  rs_iv(st);
#pragma unroll
  for (int q = 0; q < 16; q++) w[q] = y[q];
  rs_compress(st, w);
#pragma unroll
  for (int q = 0; q < 16; q++) w[q] = q < 8 ? y[16 + q] : q == 8 ? 0x8000000000000000ull : q == 15 ? 8 * 320 : 0;
  rs_compress(st, w);
#pragma unroll
  for (int q = 0; q < 8; q++) c[q] = st[q];
  uint4 acc[4] = {};
#pragma unroll 8
  for (int p = 0; p < 192; p++) {
    const uint32_t v = (uint32_t)(y[p / 8] >> (56 - 8 * (p % 8))) & 0xFFu;
    const uint4* t = tab + ((size_t)p * 256 + v) * 4;
#pragma unroll
    for (int h = 0; h < 4; h++) {
      const uint4 u = __ldg(t + h);
      acc[h].x ^= u.x; acc[h].y ^= u.y; acc[h].z ^= u.z; acc[h].w ^= u.w;
    }
  }
#pragma unroll
  for (int h = 0; h < 4; h++) {
    x[2 * h] ^= (uint64_t)acc[h].y << 32 | acc[h].x;
    x[2 * h + 1] ^= (uint64_t)acc[h].w << 32 | acc[h].z;
  }
#pragma unroll
  for (int r = 0; r < 3; r++) {   // the tree leaf over b || c, the frame-v3 leaf, the node share: 2 compressions each
    rs_iv(st);
#pragma unroll
    for (int q = 0; q < 16; q++) w[q] = q < 8 ? x[q] : c[q - 8];
    rs_compress(st, w);
#pragma unroll
    for (int q = 0; q < 16; q++) w[q] = q == 0 ? 0x8000000000000000ull : q == 15 ? 8 * 256 : 0;
    rs_compress(st, w);
#pragma unroll
    for (int q = 0; q < 8; q++) x[q] = st[q];
  }
  uint4* o = out + (size_t)leaf * 4;
#pragma unroll
  for (int h = 0; h < 4; h++)
    o[h] = make_uint4((uint32_t)x[2 * h], (uint32_t)(x[2 * h] >> 32), (uint32_t)x[2 * h + 1], (uint32_t)(x[2 * h + 1] >> 32));
}

static int commit_bench(int reps, int m, int k) {
  const uint32_t leaves = (uint32_t)((size_t)m * k / 32);
  uint8_t* A; uint4 *tab, *out;
  CK(cudaMalloc(&A, (size_t)m * k)); CK(cudaMalloc(&tab, 192 * 256 * 64)); CK(cudaMalloc(&out, (size_t)leaves * 64));
  sfill<I8><<<4096, 256>>>(A, (size_t)m * k, 7u, -63, 127);
  sfill<I8><<<4096, 256>>>(reinterpret_cast<uint8_t*>(tab), 192 * 256 * 64, 9u);
  const Sig key = sigma(0xC0FFEEu, 1);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  Nv nv;
  std::vector<double> ms, mhz;
  for (int r = 0; r <= reps; r++) {
    CK(cudaEventRecord(e0));
    commit_cost<<<(leaves + 255) / 256, 256>>>(A, k, leaves, tab, key, out);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    if (r) { ms.push_back(ms_between(e0, e1)); mhz.push_back(nv.sm()); }
  }
  printf("{\"commit_cost\": {\"m\": %d, \"k\": %d, \"leaves\": %u, \"compressions_per_leaf\": 13, \"table_lookups_per_leaf\": 192, "
         "\"ms\": %.6f, \"mhz\": %.0f, \"ms_all\": [", m, k, leaves, median(ms), median(mhz));
  for (size_t i = 0; i < ms.size(); i++) printf("%s%.6f", i ? ", " : "", ms[i]);
  printf("]}}\n");
  return 0;
}

int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 2 && !strcmp(argv[1], "serve")) {
    Slice S;
    serve(S);
    return 0;
  }
  if (argc >= 4 && !strcmp(argv[1], "bench")) {
    std::vector<std::string> s(argv + 3, argv + argc);
    return lbench(atoi(argv[2]), s);
  }
  if (argc >= 5 && !strcmp(argv[1], "commit")) return commit_bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  fprintf(stderr, "usage: slice serve | bench REPS name:m:n:k[:batch]... | commit REPS M K\n");
  return 2;
}
