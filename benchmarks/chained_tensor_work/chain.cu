// A chain of int8 GEMMs as a proof of tensor-core work (note:20261015T1622Z-draft-chained-tensor-work), on sm_120a.
//
// Step s computes C_s = X_s W_s (s8 x s8 -> s32; X_s m x k row-major, W_s k x n with n = k / 4, stored as W_s^T, n x k
// row-major), and C_s's m x n s32 words, row-major, are X_{s+1}'s m x k bytes: step s writes its output into the buffer step
// s + 1 reads as its A operand, so the reinterpretation is no instruction at all. |C| <= k 128^2 = 2^27 at k = 8192, so s32
// never saturates. The GEMM is cuBLASLt's int8 GEMM (lt) or W13's mma.sync m16n8k32 TMA-ring kernel (fgemm.cu's plain
// arm, configs P: 128 x 128 tiles of 8 warps, W: of 16 warps).
//
// The commitment of X_j (Pearl-C's -h3 row tree, hash_h3.cuh and hash_fast.cuh, the verifier's bytes): each row's
// sha512/row-seg/v1 digest (s512_row_segs, s512_folds), its hm96-sha512 leaf under the prover's salt key with nonce j
// (hm_leaves, s512_frame_leaves), and the frame-v3-sha512 levels (s512_nodes, s512_tree_small) under X_j's domain id, which
// gen.py writes (`audit.rows_domain(binding(j), j, m, HM96)`). m is a power of two, so the tree has no pads.
//
// X_0 and W_s are splitmix64 bytes from the seed: a stand-in for their expansion from the live coins, which no arm times.
//
//   chain gate DIR KEYS M K S SEED     every step of every arm, poisoned first, to DIR for ref.py; the lt arm's roots too
//   chain bench KEYS REPS SHAPE...     interleaved timing, a JSON line an (shape, arm); SHAPE = name:m:k:S
//   chain one KEYS ARM M K S           one chain run of ARM after a warm-up (for ncu)
// Arms: G/C[o|h|g], G in {lt, kP, kW}, C the commit stride (0 none, else X_j committed when C divides j); o overlaps the
// commitments on a second stream of the same priority (four X buffers), h with the commitments' stream at the greatest
// priority, g with the GEMM's; hash (one commitment of an X), and its parts rowdig, leaves, tree.
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace fg {
#include "../floored_gemm/fgemm.cu"
}
#include "../pouw/pearl_c/hash_h3.cuh"
#include "../pouw/pearl_c_sm120/hash_fast.cuh"

static int cdiv(long a, long b) { return (int)((a + b - 1) / b); }

static std::vector<uint8_t> read_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(2); }
  std::vector<uint8_t> v;
  uint8_t buf[1 << 16];
  size_t r;
  while ((r = fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + r);
  fclose(f);
  return v;
}

struct Keys {   // gen.py's: hm96 constants, the salt key, X_j's domain ids (j = 0 .. count - 1)
  uint8_t *hm, *salt, *dom;
  int ndom;
  Keys(const std::string& dir, int m) {
    hm = fg::upload(read_file(dir + "/hm.bin"));
    salt = fg::upload(read_file(dir + "/salt.bin"));
    auto d = read_file(dir + "/dom_m" + std::to_string(m) + ".bin");
    ndom = (int)(d.size() / 64);
    dom = fg::upload(d);
  }
};

enum Gemm { LT = 0, KP = 1, KW = 2 };

struct Chain {
  int m, k, n, S, nbuf, grid[3];
  std::vector<int8_t*> X, W;
  int8_t *pa, *pb;
  uint32_t* R;
  fg::Lt lt;
  fg::Cfg cfg[3];
  std::vector<fg::Maps*> maps[3];   // [gemm][s nbuf + buffer]
  uint8_t *segdig, *rowdig, *tree;
  Keys* keys;
  cudaStream_t st, st2, sg, sh;   // st and st2 the default priority, sg and sh the greatest
  std::vector<cudaEvent_t> ev_out, ev_hash;
  std::vector<bool> hash_pending;
  uint8_t* roots = nullptr;   // the gate's: X_j's root copied to roots + 64 j after its commitment
};

static void setup(Chain& C, Keys* keys, int m, int k, int S, int nbuf, uint64_t seed) {
  C.m = m; C.k = k; C.n = k / 4; C.S = S; C.nbuf = nbuf; C.keys = keys;
  const int n = C.n;
  if (m % 128 || n % 256 || (m & (m - 1)) || k > 256 * RS_MAX_SEGS * 4) {
    fprintf(stderr, "shape m %d k %d: m a power of two >= 128, k a multiple of 1024\n", m, k);
    exit(2);
  }
  if (keys->ndom <= S) { fprintf(stderr, "keys have %d domain ids, the chain needs %d\n", keys->ndom, S + 1); exit(2); }
  CK(cudaStreamCreateWithFlags(&C.st, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&C.st2, cudaStreamNonBlocking));
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  CK(cudaStreamCreateWithPriority(&C.sg, cudaStreamNonBlocking, hi));
  CK(cudaStreamCreateWithPriority(&C.sh, cudaStreamNonBlocking, hi));
  fg::Rng r{seed * 0x100000001B3ull + 29};
  std::vector<int8_t> h((size_t)m * k);
  for (int b = 0; b < nbuf; b++) {
    int8_t* x;
    CK(cudaMalloc(&x, (size_t)m * k));
    C.X.push_back(x);
  }
  for (auto& v : h) v = (int8_t)(r.next() & 0xFF);
  CK(cudaMemcpy(C.X[0], h.data(), h.size(), cudaMemcpyHostToDevice));
  h.resize((size_t)n * k);
  for (int s = 0; s < S; s++) {
    for (auto& v : h) v = (int8_t)(r.next() & 0xFF);
    C.W.push_back(fg::upload(h));
  }
  CK(cudaMalloc(&C.pa, (size_t)m * 128)); CK(cudaMalloc(&C.pb, (size_t)n * 128));
  CK(cudaMalloc(&C.R, 1 << 20));
  C.cfg[KP] = fg::cfg_p(); C.cfg[KW] = fg::cfg_w();
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  for (int g = KP; g <= KW; g++) {
    const fg::Cfg& c = C.cfg[g];
    CK(cudaFuncSetAttribute(c.k[0], cudaFuncAttributeMaxDynamicSharedMemorySize, c.smem));
    C.grid[g] = std::min((m / c.TM) * (n / c.BN), nsm);
    for (int s = 0; s < S; s++)
      for (int b = 0; b < nbuf; b++) {
        fg::Maps mp{fg::tmap(C.X[b], k, m, c.TM), fg::tmap(C.W[s], k, n, c.BN), fg::tmap(C.pa, 128, m, c.TM), fg::tmap(C.pb, 128, n, c.BN)};
        C.maps[g].push_back(fg::upload_maps(mp));
      }
  }
  fg::lt_setup(C.lt, m, n, k, C.X[0], C.W[0], reinterpret_cast<int*>(C.X[1]), C.st);
  CK(cudaMalloc(&C.segdig, (size_t)m * (n / 256) * 64));
  CK(cudaMalloc(&C.rowdig, (size_t)m * 64));
  CK(cudaMalloc(&C.tree, (size_t)(2 * m - 1) * 64));
  C.ev_out.resize(nbuf); C.ev_hash.resize(nbuf); C.hash_pending.assign(nbuf, false);
  for (int b = 0; b < nbuf; b++) {
    CK(cudaEventCreateWithFlags(&C.ev_out[b], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&C.ev_hash[b], cudaEventDisableTiming));
  }
  s512_heads<<<cdiv(RS_HEAD_SEGS + RS_MAX_SEGS + 1, 128), 128, 0, C.st>>>(Args{});
  CK(cudaStreamSynchronize(C.st));
}

// X[bout] = X[bin] W_s
static void step(Chain& C, int g, int s, int bin, int bout, cudaStream_t st) {
  int* D = reinterpret_cast<int*>(C.X[bout]);
  if (g == LT) {
    fg::lt_run(C.lt, C.X[bin], C.W[s], D, st);
    return;
  }
  const fg::Cfg& c = C.cfg[g];
  fg::Args a{C.maps[g][(size_t)s * C.nbuf + bin], D, C.R, C.m, C.n, C.k, 8, 0};
  c.k[0]<<<C.grid[g], c.NCT, c.smem, st>>>(a);
}

static Args args(std::initializer_list<uint64_t> p, std::initializer_list<int32_t> i) {
  Args a{};
  int t = 0;
  for (uint64_t v : p) a.p[t++] = v;
  t = 0;
  for (int32_t v : i) a.i[t++] = v;
  return a;
}
// the parts of X[b]'s commitment as X_j: 1 the row digests, 2 the leaves, 4 the levels
static void commit(Chain& C, int b, int j, cudaStream_t st, int parts = 7) {
  const int m = C.m, segs = C.n / 256;
  const uint64_t tree = (uint64_t)C.tree, dom = (uint64_t)(C.keys->dom + 64 * j);
  if (parts & 1) {
    s512_row_segs<<<cdiv((long)m * segs, RS_FAST_NT), RS_FAST_NT, 0, st>>>(args({(uint64_t)C.X[b], (uint64_t)C.segdig}, {m, C.n}));
    s512_folds<<<cdiv(m, RS_FOLD_NT), RS_FOLD_NT, 0, st>>>(args({(uint64_t)C.segdig, (uint64_t)C.rowdig}, {m, segs, m, segs}));
  }
  if (parts & 2) {
    hm_leaves<<<cdiv(m, 64), 64, 0, st>>>(args({(uint64_t)C.rowdig, tree, (uint64_t)C.keys->salt, (uint64_t)C.keys->hm, 0, 0}, {m, j, 0}));
    s512_frame_leaves<<<cdiv(m, 64), 64, 0, st>>>(args({tree, dom}, {0, m}));
  }
  if (parts & 4) {
    int width = m, off = 0, level = 0;
    while (width > 128) {
      s512_nodes<<<cdiv(width / 2, 64), 64, 0, st>>>(args({tree + 64ull * off, tree + 64ull * (off + width), dom}, {level, width / 2}));
      off += width; width /= 2; level++;
    }
    if (width > 1) s512_tree_small<<<1, 64, 0, st>>>(args({tree + 64ull * off, dom}, {width, level}));
  }
  if (C.roots && parts == 7) CK(cudaMemcpyAsync(C.roots + 64ull * j, C.tree + 64ull * (2 * m - 2), 64, cudaMemcpyDeviceToDevice, st));
}
static uint8_t* root(Chain& C) { return C.tree + 64ull * (2 * C.m - 2); }

struct Arm {
  std::string name;
  int g = LT, c = 0, hash = 0;   // hash: 0 a chain, else the commitment parts alone
  bool overlap = false;
  char prio = 'o';
};
static Arm parse_arm(const std::string& s) {
  Arm a;
  a.name = s;
  if (s == "hash") { a.hash = 7; return a; }
  if (s == "rowdig") { a.hash = 1; return a; }
  if (s == "leaves") { a.hash = 2; return a; }
  if (s == "tree") { a.hash = 4; return a; }
  const size_t sl = s.find('/');
  const std::string g = s.substr(0, sl);
  a.g = g == "lt" ? LT : g == "kP" ? KP : g == "kW" ? KW : -1;
  if (a.g < 0 || sl == std::string::npos) { fprintf(stderr, "arm %s\n", s.c_str()); exit(2); }
  std::string c = s.substr(sl + 1);
  if (!c.empty() && (c.back() == 'o' || c.back() == 'h' || c.back() == 'g')) { a.overlap = true; a.prio = c.back(); c.pop_back(); }
  a.c = atoi(c.c_str());
  return a;
}

static cudaStream_t gemm_stream(Chain& C, const Arm& a) { return a.overlap && a.prio == 'g' ? C.sg : C.st; }
// S steps of the arm from X_0 (in buffer 0), the buffers rotating mod nbuf; with overlap, a step that overwrites a buffer
// waits for that buffer's commitment
static void run_chain(Chain& C, const Arm& a) {
  if (a.hash) {
    for (int s = 0; s < C.S; s++) commit(C, s % C.nbuf, s + 1, C.st, a.hash);
    return;
  }
  const cudaStream_t gs = gemm_stream(C, a), hs = a.prio == 'h' ? C.sh : C.st2;
  for (int s = 0; s < C.S; s++) {
    const int bin = s % C.nbuf, bout = (s + 1) % C.nbuf;
    if (a.overlap && C.hash_pending[bout]) { CK(cudaStreamWaitEvent(gs, C.ev_hash[bout], 0)); C.hash_pending[bout] = false; }
    step(C, a.g, s, bin, bout, gs);
    if (a.c && (s + 1) % a.c == 0) {
      if (a.overlap) {
        CK(cudaEventRecord(C.ev_out[bout], gs));
        CK(cudaStreamWaitEvent(hs, C.ev_out[bout], 0));
        commit(C, bout, s + 1, hs);
        CK(cudaEventRecord(C.ev_hash[bout], hs));
        C.hash_pending[bout] = true;
      } else {
        commit(C, bout, s + 1, gs);
      }
    }
  }
  if (a.overlap) {
    for (int b = 0; b < C.nbuf; b++)
      if (C.hash_pending[b]) { CK(cudaStreamWaitEvent(gs, C.ev_hash[b], 0)); C.hash_pending[b] = false; }
  }
}

static void dump(const std::string& path, const void* p, size_t bytes) {
  std::vector<uint8_t> h(bytes);
  CK(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(h.data(), 1, bytes, f) != bytes) { fprintf(stderr, "write %s failed\n", path.c_str()); exit(3); }
  fclose(f);
}

// every arm's X_1 .. X_S (each buffer filled with 0xA5 first), and the lt arm's row digests and roots of X_1 .. X_S; the
// control arm "none" skips step 0's GEMM, so its X_1 stays poisoned and ref.py must reject it. Then the timed overlapped
// arms on two buffers, so that a step overwrites a buffer whose commitment is in flight: their roots (poisoned first) and
// X_S, which ref.py compares with the serial ones
static const char* OVERLAP[] = {"lt/1h", "lt/1o", "lt/1g", "kP/1h", "kW/1h", "kW/2h"};
static int cmd_gate(const std::string& dir, const std::string& kdir, int m, int k, int S, uint64_t seed) {
  Keys keys(kdir, m);
  Chain C;
  setup(C, &keys, m, k, S, S + 1, seed);
  dump(dir + "/X0.bin", C.X[0], (size_t)m * k);
  for (int s = 0; s < S; s++) dump(dir + "/W" + std::to_string(s) + ".bin", C.W[s], (size_t)C.n * k);
  const char* names[4] = {"lt", "kP", "kW", "none"};
  for (int g = 0; g < 4; g++) {
    for (int b = 1; b <= S; b++) CK(cudaMemset(C.X[b], 0xA5, (size_t)m * k));
    CK(cudaMemset(C.rowdig, 0xA5, (size_t)m * 64)); CK(cudaMemset(C.tree, 0xA5, (size_t)(2 * m - 1) * 64));
    CK(cudaDeviceSynchronize());
    for (int s = 0; s < S; s++) {
      if (g < 3) step(C, g, s, s, s + 1, C.st);
      else if (s > 0) step(C, LT, s, s, s + 1, C.st);
      if (g == 0) {
        commit(C, s + 1, s + 1, C.st);
        CK(cudaStreamSynchronize(C.st));
        dump(dir + "/rowdig" + std::to_string(s + 1) + ".bin", C.rowdig, (size_t)m * 64);
        dump(dir + "/root" + std::to_string(s + 1) + ".bin", root(C), 64);
      }
    }
    CK(cudaStreamSynchronize(C.st));
    CK(cudaGetLastError());
    for (int s = 1; s <= S; s++) dump(dir + "/" + names[g] + "_X" + std::to_string(s) + ".bin", C.X[s], (size_t)m * k);
  }
  std::string ov;
  for (const char* name : OVERLAP) {
    Chain O;
    setup(O, &keys, m, k, S, 2, seed);
    CK(cudaMalloc(&O.roots, 64ull * (S + 1)));
    CK(cudaMemset(O.roots, 0xA5, 64ull * (S + 1)));
    for (int b = 1; b < O.nbuf; b++) CK(cudaMemset(O.X[b], 0xA5, (size_t)m * k));
    CK(cudaDeviceSynchronize());
    const Arm a = parse_arm(name);
    run_chain(O, a);
    CK(cudaStreamSynchronize(gemm_stream(O, a)));
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    std::string tag = name;
    std::replace(tag.begin(), tag.end(), '/', '_');
    dump(dir + "/ovl_" + tag + "_roots.bin", O.roots, 64ull * (S + 1));
    dump(dir + "/ovl_" + tag + "_XS.bin", O.X[S % O.nbuf], (size_t)m * k);
    ov += std::string(ov.empty() ? "" : ", ") + "\"" + tag + "\"";
  }
  FILE* f = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(f, "{\"m\": %d, \"k\": %d, \"S\": %d, \"seed\": %llu, \"arms\": [\"lt\", \"kP\", \"kW\"], \"control\": \"none\", \"overlap\": [%s]}\n",
          m, k, S, (unsigned long long)seed, ov.c_str());
  fclose(f);
  printf("gate dumped m %d k %d S %d to %s\n", m, k, S, dir.c_str());
  return 0;
}

static int cmd_bench(const std::string& kdir, int reps, const std::vector<std::string>& shapes, const std::vector<std::string>& arm_names) {
  fg::Nv nv;
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  std::vector<Arm> arms;
  for (auto& s : arm_names) arms.push_back(parse_arm(s));
  for (const std::string& sh : shapes) {
    char name[64];
    int m, k, S;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &k, &S) != 4) { fprintf(stderr, "shape %s\n", sh.c_str()); return 2; }
    Keys keys(kdir, m);
    Chain C;
    setup(C, &keys, m, k, S, 4, 0x5EED0000ull + m + k);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int na = (int)arms.size();
    std::vector<std::vector<double>> ms(na), clk(na);
    for (int rep = -2; rep < reps; rep++)
      for (int t = 0; t < na; t++) {
        const int ai = (t + (rep < 0 ? 0 : rep)) % na;
        const cudaStream_t gs = gemm_stream(C, arms[ai]);
        CK(cudaEventRecord(e0, gs));
        run_chain(C, arms[ai]);
        CK(cudaEventRecord(e1, gs));
        CK(cudaEventSynchronize(e1));
        const unsigned c = nv.sm();
        float x;
        CK(cudaEventElapsedTime(&x, e0, e1));
        if (rep >= 0) { ms[ai].push_back(x); clk[ai].push_back(c); }
      }
    CK(cudaGetLastError());
    const double macs = (double)m * C.n * k;
    for (int ai = 0; ai < na; ai++) {
      const double t = fg::median(ms[ai]) / S, mhz = fg::median(clk[ai]);
      const double peak = 1023.6 * nsm * mhz * 1e6;
      printf("{\"shape\": \"%s\", \"m\": %d, \"k\": %d, \"n\": %d, \"S\": %d, \"arm\": \"%s\", \"reps\": %d, \"ms_per_step\": %.6f, "
             "\"ms_min\": %.6f, \"ms_max\": %.6f, \"sm_mhz\": %.0f, \"mhz_min\": %.0f, \"mhz_max\": %.0f, \"eta\": %.4f, \"lt_algo\": %d, \"lt_tile\": %d, \"ms_reps\": [",
             name, m, k, C.n, S, arms[ai].name.c_str(), reps, t, *std::min_element(ms[ai].begin(), ms[ai].end()) / S,
             *std::max_element(ms[ai].begin(), ms[ai].end()) / S, mhz, *std::min_element(clk[ai].begin(), clk[ai].end()),
             *std::max_element(clk[ai].begin(), clk[ai].end()), arms[ai].hash ? 0.0 : macs / (t * 1e-3) / peak, C.lt.algo_id, C.lt.tile);
      for (size_t i = 0; i < ms[ai].size(); i++) printf("%s%.5f", i ? ", " : "", ms[ai][i]);
      printf("], \"mhz_reps\": [");
      for (size_t i = 0; i < clk[ai].size(); i++) printf("%s%.0f", i ? ", " : "", clk[ai][i]);
      printf("]}\n");
      fflush(stdout);
    }
  }
  return 0;
}

static int cmd_one(const std::string& kdir, const std::string& arm, int m, int k, int S) {
  Keys keys(kdir, m);
  Chain C;
  setup(C, &keys, m, k, S, 3, 0x5EED0000ull + m + k);
  const Arm a = parse_arm(arm);
  run_chain(C, a);
  CK(cudaStreamSynchronize(C.st));
  cudaProfilerStart();
  run_chain(C, a);
  CK(cudaStreamSynchronize(C.st));
  cudaProfilerStop();
  CK(cudaGetLastError());
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 8 && !strcmp(argv[1], "gate"))
    return cmd_gate(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), strtoull(argv[7], nullptr, 0));
  if (argc >= 5 && !strcmp(argv[1], "bench")) {
    std::vector<std::string> shapes;
    for (int i = 4; i < argc; i++) shapes.push_back(argv[i]);
    const char* env = getenv("ARMS");
    std::string s = env && *env ? env : "lt/0 kP/0 kW/0 lt/1 lt/2 lt/4 lt/1h lt/2h lt/4h kW/1h kW/2h kW/4h hash rowdig leaves tree";
    std::vector<std::string> arms;
    for (size_t p = 0; p < s.size();) {
      size_t q = s.find(' ', p);
      if (q == std::string::npos) q = s.size();
      if (q > p) arms.push_back(s.substr(p, q - p));
      p = q + 1;
    }
    return cmd_bench(argv[2], atoi(argv[3]), shapes, arms);
  }
  if (argc >= 7 && !strcmp(argv[1], "one")) return cmd_one(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
  fprintf(stderr, "usage: chain gate DIR KEYS M K S SEED | chain bench KEYS REPS SHAPE... | chain one KEYS ARM M K S\n");
  return 2;
}
