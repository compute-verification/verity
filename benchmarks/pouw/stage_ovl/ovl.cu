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
//                           small resident kernel (32-256 threads a SM): alone, launched just before or after the GEMMs,
//                           and spread (a slice after each GEMM, waiting on it, as each GEMM's output's commitment)
// Arms (k chunks of M tokens; the QK^T and P.V stand-ins are stage.cu's, m = M, n = M/2, k = 4096, against a sequence's own
// distinct buffer a layer):
//   plain     the k sequences plain int8 on one stream: the linears over all k * M tokens in row blocks of PLAIN_RB (8192),
//             attention a strided batch over the k sequences. The divisor.
//   chunked   k streams, each one chunk's 192 GEMMs and nothing else (chunkedrrW: in rrW's order); plain int8 too, so
//             report.py also costs against the fastest of plain and these
//   eq        k streams at one priority, per stage commit -> draw -> GEMM
//   prio      eq with chunk c's stream at priority class floor(c * levels / k), chunk 0 highest
//   hhi       eq with each chunk's commitments on a stream of its own at the highest priority
//   hprio     prio with the commitments at the highest priority
//   hlo       the commitments at the lowest priority, the GEMMs at the highest
//   rrW       hhi with the GEMMs in one static order (stage-major, chunk-minor), each after the GEMM W places before it
//   nodraw    hhi without the draws; nohash: eq without the commitments (nohashrrW: rrW without them); hashonly: the k
//             chunks' commitments alone
//   lite, liteprio, literrW, liteonly: hhi, hprio, rrW and hashonly with each commitment a stand-in of the same
//             compressions a row (LITE_C, 367 a token-layer) in 128-thread CTAs of LITE_N (8) chained compressions, no
//             shared memory: what a committer of that shape would cost (its latency understated, its roots not real)
//   litec, litecprio, litecrrW, liteconly: the same, the stand-in chained as the tree is (a thread a segment, a thread a
//             row, each CTA's 128-leaf levels, one CTA for the levels above: ~40 dependent compressions a stage at 2K)
//   cg, cgeq, cgrrW, cgonly: hhi, eq, rrW and hashonly with each commitment commit_gpu's small path (benchmarks/pouw/commit_gpu: CTAs of 128
//             threads, no shared memory) over the stage's operand as tile leaves of CG_R rows x 32 values (Decision 46),
//             sha512/row/v2 rows, hm96-sha512 leaves with derived salts in a frame-v3-sha512 tree of the stage's own domain
//             (the CGP1 files DIR/cg-M-R-t.bin); layer 0's salts with the frame-leaf first blocks and pads at the chunk's
//             start, layer l+1's salts for stage t on the chunk's salt stream after layer l's commitment of stage t.
//             Their poisoned check holds them to commit_gpu's serial roots, and CG_DUMP=DIR writes chunk 0's six
//             operands, levels and params for `reference.py verify` (benchmarks/pouw/commit_gpu) after the lease
//   cgh, cgheq, cghrrW, cghonly: the same with layers 1..31's S records made by host threads (commit_gpu's host_salt,
//             checked equal to the GPU's) and copied in by cudaMemcpyAsync where the cg arms run salt_small; layer 0's
//             salts, F and the pads stay on the GPU at the chunk's start
//   cgpre, cgpreeq, cgprerrW, cgpreonly, cgprelo, cgprelorrW (cgpre and cgprerrW with the commitments at the lowest
//             priority and the GEMMs at the highest, as hlo): the same with all 32 layers' salts, F and the pads made by
//             salt_small before the window (cg_pre, its time alone printed as mode salt_pre) into the committers' slots, so
//             the window only reads them; cg_pre runs again after each poisoning, so their poisoned check still covers
//             the salts
//   NAME-f, NAME-p, NAME-fp (NAME a cg, cgh or cgpre arm): the same with commit_gpu's fuse_top (f: the last leaf CTA of
//             each commitment folds the levels above the CTAs where top_small would be launched) and/or its prefetch (p:
//             each leaf's next 128-byte block into L1 as its compression starts); held to the serial roots (made without)
//   cgprend, cgprendeq: cgpre and cgpreeq without the draws
//   E:NAME, ER:NAME, ERL:NAME (NAME chunked, chunkedrrW, nohash or a cgpre arm): NAME with every linear (QKV, o_proj,
//             gate_up, down) EGEMM's edited GEMM (eg.cu: a call is qwe, the dithered quantizer writing the stage's operand
//             from bf16 and W' = W + U under the call's key, then the GEMM with windowed duo receipts at EG_S segments a
//             window, keyed by the call's seed, the draw's); ER also commits each GEMM's receipts (2 KiB rows, a window of a
//             tile each, every slot's salts made before the window) on a stream a chunk at the highest priority (ERL: the
//             lowest), the next layer's GEMM of that stage waiting for it. Their poisoned check holds the receipts' roots to
//             the serial path's (E's are the control), EG_DUMP=DIR writes chunk 0's last call of each linear (X, W', the
//             receipts, their levels) for eref.py and `reference.py verify`, and eg_gate holds each linear's Y to cuBLASLt's
//             X W^T and w_d's receipts to w_d4q's
//   EN:NAME   E:NAME without qwe: every layer's GEMM over the X and W' the serial reference left (the GEMMs' own cost)
//   Q:NAME    (NAME chunked or chunkedrrW) plain's GEMMs, each linear's after plain serving's activation quantizer
//             (egemm.cu's quant<false> from the same bf16 activations into a scratch operand)
//
// env: ARMS, DRAW_US (300), PLAIN_RB (8192), MID (1: stage.cu's frame heads ahead), CARVE (unset: the commitments' and draws'
// kernels at the driver's carveout; P: P% shared memory), BESIDE_C (367), CG_R (64), CG_DUMP, CG_SKIP (timing only: salts,
// leaves, top dropped from the cg arms' graphs; with commit_gpu's -DCOMMIT_GPU_DIAG_* builds, run.sh's cgdiag breakdown),
// EG_S (16), EG_KERN (unset: w_d or w_d4q, whichever is faster alone at each linear), EG_GRID (sms: persistent; else a CTA a
// tile), EG_DUMP
// Build: nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a [-DFR_ROT=BITS] ovl.cu eg.cu -lcuda
//   -lcublasLt -lnvidia-ml -ldl
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
// commit_gpu's committer (benchmarks/pouw/commit_gpu, its small path), a domain slot per stage
#define COMMIT_GPU_DOMS 6
namespace cg {
#include "../commit_gpu/commit_gpu.cuh"
}
#undef HD  // sha512.cuh's qualifier macro; HD below is stage.cu's head dimension
#include "eg.h"
// commit_gpu's timing-only builds (-DCOMMIT_GPU_DIAG_*), named in ovl_setup's cg_diag
static const char* const CG_DIAG = ""
#ifdef COMMIT_GPU_DIAG_NOOP
                                   "noop "
#endif
#ifdef COMMIT_GPU_DIAG_NOS
                                   "nos "
#endif
#ifdef COMMIT_GPU_DIAG_NOMASK
                                   "nomask "
#endif
#ifdef COMMIT_GPU_DIAG_NOTREE
                                   "notree "
#endif
    ;

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
// the chained stand-in's phases: `items` threads, n chained compressions each (out: cap slots)
__global__ void lite_items(long long items, int n, uint64_t* out, long long cap) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= items) return;
  uint64_t s[8], w[16];
  s512_iv(s);
  for (int j = 0; j < 16; j++) w[j] = (uint64_t)(i + 1) * (j + 1);
  for (int r = 0; r < n; r++) {
    fr_compress(s, w);
    for (int j = 0; j < 8; j++) w[j] ^= s[j];
  }
  out[i % cap] = s[0];
}
// a tree's levels over `leaves` nodes, blockDim leaves a CTA: each level half the threads, `per` chained compressions a
// node, a barrier between levels (the CTA's subtree, as fr_tree's shared-memory levels, without the memory)
__global__ void lite_levels(int leaves, int per, uint64_t* out) {
  uint64_t s[8], w[16];
  s512_iv(s);
  for (int j = 0; j < 16; j++) w[j] = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x + 1) * (j + 1);
  const int mine = min(leaves - (int)(blockIdx.x * blockDim.x), (int)blockDim.x);
  for (int width = mine / 2; width >= 1; width /= 2) {
    if ((int)threadIdx.x < width)
      for (int r = 0; r < per; r++) {
        fr_compress(s, w);
        for (int j = 0; j < 8; j++) w[j] ^= s[j];
      }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

// CARVE=P: the commitments' and the draws' kernels prefer a P% shared-memory carveout (100: the most, as the GEMMs), so
// an SM need not drain to switch its carveout between them and a GEMM's CTAs
static int g_carve = -1;
static void set_carve(int p) {
  const void* f[] = {(const void*)fr_salt, (const void*)fr_mid0, (const void*)fr_segments, (const void*)fr_tree,
                     (const void*)draw_stamp, (const void*)stamp, (const void*)sha_beside, (const void*)lite_items,
                     (const void*)lite_levels};
  for (const void* k : f) CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, p));
  const void* g[] = {(const void*)cg::salt_small<true>, (const void*)cg::leaf_small<cg::RowTile, 0>, (const void*)cg::level_kernel,
                     (const void*)cg::top_small};
  for (const void* k : g) CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, p));
  g_carve = p;
}

struct OChunk {
  uint8_t* op[NSTAGE];
  int8_t* A;
  int* D;
  Commit cm[NSTAGE];
  cg::Committer cgc[NSTAGE];
  LtCtx lt;
};
struct Sched {
  std::string name;
  bool hash, draw, gemm;
  bool classes;  // chunk c's GEMMs at priority class floor(c * levels / k)
  int hp;        // the commitments' stream: 0 the chunk's own, 1 one of their own at the highest priority, -1 at the lowest
                 // with the GEMMs at the highest
  int rr;        // > 0: the GEMMs in one static order, each after the one rr places before it
  int lite;      // each commitment a stand-in of its stage's compressions a row (LITE_C) in 128-thread CTAs, no shared
                 // memory: 1 flat (LITE_N chained a thread), 2 chained as the tree (segments, the row, CTA levels, the top);
                 // 3: commit_gpu's small path, the real commitment; 4: the same with layers 1..31's S records made by host
                 // threads and copied in (cudaMemcpyAsync on the chunk's salt stream) in place of salt_small; 5: the
                 // same with every slot's salts, F and the pads made before the graph (cg_pre), none in it
  bool fuse = false;  // a cg arm's NAME-f or NAME-fp: commit_gpu's fuse_top (the last leaf CTA folds the levels above the CTAs)
  bool pf = false;    // NAME-p or NAME-fp: commit_gpu's prefetch (each leaf's next block into L1)
  bool eg = false;    // E, ER, ERL, EN: the linears EGEMM's edited GEMM
  int rc = 0;         // ER: the receipts committed at the highest priority, ERL: at the lowest
  bool noqwe = false; // EN: no qwe in the graph (every layer's GEMM over the operands the serial reference left)
  bool quant = false; // Q: plain's GEMMs, each linear's after plain serving's activation quantizer
};
static Sched sched_of(const std::string& n) {
  for (const char* pre : {"ERL:", "ER:", "EN:", "E:", "Q:"}) {
    const size_t pl = strlen(pre);
    if (n.compare(0, pl, pre) != 0) continue;
    Sched s = sched_of(n.substr(pl));
    if ((s.hash && s.lite != 5) || !s.gemm || s.eg || s.quant || (pre[0] == 'Q' && s.hash)) {
      fprintf(stderr, "arm %s: E, ER, ERL and EN take chunked, chunkedrrW, nohash or a cgpre arm, Q chunked or chunkedrrW\n", n.c_str());
      exit(1);
    }
    s.name = n;
    if (pre[0] == 'Q') {
      s.quant = true;
      return s;
    }
    s.eg = true;
    s.rc = pre[1] != 'R' ? 0 : pre[2] == 'L' ? -1 : 1;
    s.noqwe = pre[1] == 'N';
    return s;
  }
  const size_t dash = n.rfind('-');
  if (dash != std::string::npos && dash + 1 < n.size() && n.find_first_not_of("fp", dash + 1) == std::string::npos) {
    Sched s = sched_of(n.substr(0, dash));
    if (s.lite < 3) {
      fprintf(stderr, "arm %s: -f, -p, -fp take a cg arm\n", n.c_str());
      exit(1);
    }
    s.name = n;
    s.fuse = n.find('f', dash) != std::string::npos;
    s.pf = n.find('p', dash) != std::string::npos;
    return s;
  }
  if (n == "chunked") return {n, false, false, true, false, 0, 0, 0};
  if (n.rfind("chunkedrr", 0) == 0) return {n, false, false, true, false, 0, atoi(n.c_str() + 9), 0};
  if (n == "eq") return {n, true, true, true, false, 0, 0, 0};
  if (n == "prio") return {n, true, true, true, true, 0, 0, 0};
  if (n == "hhi") return {n, true, true, true, false, 1, 0, 0};
  if (n == "hprio") return {n, true, true, true, true, 1, 0, 0};
  if (n == "hlo") return {n, true, true, true, false, -1, 0, 0};
  if (n == "nodraw") return {n, true, false, true, false, 1, 0, 0};
  if (n == "nohash") return {n, false, true, true, false, 0, 0, 0};
  if (n.rfind("nohashrr", 0) == 0) return {n, false, true, true, false, 0, atoi(n.c_str() + 8), 0};
  if (n == "hashonly") return {n, true, false, false, false, 0, 0, 0};
  if (n.rfind("rr", 0) == 0) return {n, true, true, true, false, 1, atoi(n.c_str() + 2), 0};
  if (n == "lite") return {n, true, true, true, false, 1, 0, 1};
  if (n == "liteprio") return {n, true, true, true, true, 1, 0, 1};
  if (n == "liteonly") return {n, true, false, false, false, 1, 0, 1};
  if (n.rfind("literr", 0) == 0) return {n, true, true, true, false, 1, atoi(n.c_str() + 6), 1};
  if (n == "litec") return {n, true, true, true, false, 1, 0, 2};
  if (n == "litecprio") return {n, true, true, true, true, 1, 0, 2};
  if (n == "liteconly") return {n, true, false, false, false, 1, 0, 2};
  if (n.rfind("litecrr", 0) == 0) return {n, true, true, true, false, 1, atoi(n.c_str() + 7), 2};
  if (n == "cgh") return {n, true, true, true, false, 1, 0, 4};
  if (n == "cgheq") return {n, true, true, true, false, 0, 0, 4};
  if (n == "cghonly") return {n, true, false, false, false, 1, 0, 4};
  if (n.rfind("cghrr", 0) == 0) return {n, true, true, true, false, 1, atoi(n.c_str() + 5), 4};
  if (n == "cgpre") return {n, true, true, true, false, 1, 0, 5};
  if (n == "cgprend") return {n, true, false, true, false, 1, 0, 5};
  if (n == "cgprendeq") return {n, true, false, true, false, 0, 0, 5};
  if (n.rfind("cgprelorr", 0) == 0) return {n, true, true, true, false, -1, atoi(n.c_str() + 9), 5};
  if (n == "cgprelo") return {n, true, true, true, false, -1, 0, 5};
  if (n == "cgpreeq") return {n, true, true, true, false, 0, 0, 5};
  if (n == "cgpreonly") return {n, true, false, false, false, 1, 0, 5};
  if (n.rfind("cgprerr", 0) == 0) return {n, true, true, true, false, 1, atoi(n.c_str() + 7), 5};
  if (n == "cg") return {n, true, true, true, false, 1, 0, 3};
  if (n == "cgeq") return {n, true, true, true, false, 0, 0, 3};
  if (n == "cgonly") return {n, true, false, false, false, 1, 0, 3};
  if (n.rfind("cgrr", 0) == 0) return {n, true, true, true, false, 1, atoi(n.c_str() + 4), 3};
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
  bool hash, stamps, cg = false, eg = false, rc = false;
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
  const int GN[NSTAGE] = {QKV, M / 2, M / 2, HID, 2 * FF, HID}, GK[NSTAGE] = {HID, HID, HID, HID, HID, FF};
  const int LJ[NSTAGE] = {0, -1, -1, 1, 2, 3};  // the stage's weight in d.W, or -1 for attention
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
  bool use_cg = false;
  bool use_cgh = false;
  for (const std::string& nm : arm_names) {
    use_cg |= nm != "plain" && sched_of(nm).lite >= 3;
    use_cgh |= nm != "plain" && sched_of(nm).lite == 4;
  }
  for (int c = 0; c < K; c++)
    for (int t = 0; t < NSTAGE; t++) ch[c].cm[t] = make_commit(hc[t]);
  CK(cudaDeviceSynchronize());
  const size_t mem_cm = mem_used();
  // commit_gpu's committers: stage t in domain slot t; every stage's Consts (P(256 R), the salt prefix, leaf_prefix, Ks)
  // must agree, as C holds one copy
  const int CGR = getenv("CG_R") && *getenv("CG_R") ? atoi(getenv("CG_R")) : 64;
  cg::u64 cg_n[NSTAGE] = {};
  std::vector<cg::Params> cgp;
  if (use_cg) {
    for (int t = 0; t < NSTAGE; t++) {
      cgp.push_back(cg::read_params(dir + "/cg-" + std::to_string(M) + "-" + std::to_string(CGR) + "-" + std::to_string(t) + ".bin"));
      const cg::Params& p = cgp.back();
      cg_n[t] = cg::tile_leaves(M, SK[t], CGR);
      for (int c = 0; c < K; c++) {
        ch[c].cgc[t].init(p, cg_n[t], 32 * CGR, false, false, 0, t, true, LAYERS);
        if (memcmp(&ch[c].cgc[t].hc, &ch[0].cgc[0].hc, sizeof(cg::Consts))) {
          fprintf(stderr, "commit_gpu: stage %d's constants differ from stage 0's (one Ks, key and row size for all)\n", t);
          exit(4);
        }
      }
    }
    CK(cudaDeviceSynchronize());
  }
  const size_t mem_cg = mem_used();
  auto cg_root = [&](const cg::Committer& m) { return m.levels + 8 * (2 * m.width - 2); };
  auto cg_roots = [&](int k) {
    std::vector<uint8_t> r((size_t)k * NSTAGE * 64);
    for (int c = 0; c < k; c++)
      for (int t = 0; t < NSTAGE; t++)
        CK(cudaMemcpy(r.data() + ((size_t)c * NSTAGE + t) * 64, cg_root(ch[c].cgc[t]), 64, cudaMemcpyDeviceToHost));
    return r;
  };
  auto cg_poison = [&](int k) {
    for (int c = 0; c < k; c++)
      for (int t = 0; t < NSTAGE; t++) {
        const cg::Committer& m = ch[c].cgc[t];
        CK(cudaMemset(m.levels, 0xA5, 2 * m.width * 64));
        CK(cudaMemset(m.S, 0xA5, m.n * 128 * m.slots));
        CK(cudaMemset(m.F, 0xA5, m.n * 64));
      }
    CK(cudaDeviceSynchronize());
  };
  // the cgpre arms' salts: every slot of k chunks' committers, F and the pads, alone on d.st; returns its time
  auto cg_pre = [&](int k) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, d.st));
    for (int c = 0; c < k; c++)
      for (int t = 0; t < NSTAGE; t++) ch[c].cgc[t].salts<true>(nullptr, d.st);
    CK(cudaEventRecord(b, d.st));
    CK(cudaEventSynchronize(b));
    CK(cudaGetLastError());
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return ms;
  };
  // the edited arms: a Lin a chunk's linear stage over its operand op[t] (qwe writes it), the weights in [-63, 63]; per-call
  // keys (the dither and U) and seeds (the receipts', the draw's) a function of (layer, stage, chunk). Layer 31's call
  // leaves op[t] and the receipts' tree, so the serial reference makes it, before commit_gpu's reference reads op[t]
  bool use_eg = false, use_rc = false;
  for (const std::string& nm : arm_names)
    if (nm != "plain") {
      const Sched s = sched_of(nm);
      use_eg |= s.eg || s.quant;
      use_rc |= s.eg && s.rc;
    }
  const int EGS = getenv("EG_S") && *getenv("EG_S") ? atoi(getenv("EG_S")) : 16;
  auto ekey = [](int l, int t, int c) { return (uint32_t)((l * NSTAGE + t) * 64 + c); };
  auto eseed = [](int l, int t, int c) { return 0x9E3779B9u * (uint32_t)((l * NSTAGE + t) * 64 + c + 1) ^ 0x5EEDu; };
  std::vector<std::vector<eg::Lin*>> lin(K, std::vector<eg::Lin*>(NSTAGE, nullptr));
  std::vector<uint8_t> rc_ref;
  int eg_fails = 0;
  auto rc_roots = [&](int k) {
    std::vector<uint8_t> r((size_t)k * NSTAGE * 64, 0);
    for (int c = 0; c < k; c++)
      for (int t = 0; t < NSTAGE; t++)
        if (lin[c][t]) eg::lin_root(lin[c][t], r.data() + ((size_t)c * NSTAGE + t) * 64);
    return r;
  };
  auto rc_poison_salt = [&](int k) {
    for (int c = 0; c < k; c++)
      for (int t = 0; t < NSTAGE; t++)
        if (lin[c][t]) eg::lin_rpoison(lin[c][t]);
    for (int c = 0; c < k; c++)
      for (int t = 0; t < NSTAGE; t++)
        if (lin[c][t]) eg::lin_rsalts(lin[c][t], d.st);
    CK(cudaStreamSynchronize(d.st));
    CK(cudaGetLastError());
  };
  if (use_eg) {
    eg::init();
    if (g_carve >= 0) eg::set_carve(g_carve);
    for (int l = 0; l < LAYERS; l++)
      for (int j = 0; j < 4; j++) eg::fill_w(d.W[l][j], (size_t)LN[j] * LK[j], 0x1000 + l * 4 + j);
    CK(cudaDeviceSynchronize());
    uint16_t* xs[NSTAGE] = {};
    int es[NSTAGE] = {};
    for (int t = 0; t < NSTAGE; t++) {
      if (LJ[t] < 0) continue;
      xs[t] = eg::make_x(M, GK[t], 0x7000 + t);
      for (es[t] = EGS; es[t] > 4 && (GK[t] / 32) % es[t]; es[t] -= 4) {}   // down at S = 128: 112 (448 segments)
    }
    for (int c = 0; c < K; c++)
      for (int t = 0; t < NSTAGE; t++)
        if (LJ[t] >= 0)
          lin[c][t] = eg::lin_new(M, GN[t], GK[t], es[t], reinterpret_cast<int8_t*>(ch[c].op[t]), xs[t], ch[c].D,
                                  dir + "/eg-" + std::to_string(M) + "-" + std::to_string(EGS) + "-" + std::to_string(t) + ".bin",
                                  LJ[t], LAYERS);
    // each linear's kernel: the faster alone (EG_KERN fixes it)
    double kt[NSTAGE][2] = {};
    int kern[NSTAGE] = {};
    for (int t = 0; t < NSTAGE; t++) {
      if (!lin[0][t]) continue;
      eg::lin_qwe(lin[0][t], d.W[0][LJ[t]], ekey(0, t, 0), d.st);
      for (int q = 0; q < 2; q++) kt[t][q] = eg::lin_time(lin[0][t], q, 9, d.st);
      kern[t] = getenv("EG_KERN") && *getenv("EG_KERN") ? atoi(getenv("EG_KERN")) : kt[t][1] < kt[t][0];
      for (int c = 0; c < K; c++) eg::lin_kernel(lin[c][t], kern[t]);
    }
    rc_poison_salt(K);
    // the gate on chunk 0's layer-31 call of each linear (and its dump for eref.py and reference.py verify)
    const char* dump = getenv("EG_DUMP");
    for (int t = 0; t < NSTAGE; t++) {
      if (!lin[0][t]) continue;
      const std::string w = dump && *dump ? std::string(dump) + "/M" + std::to_string(M) + "-S" + std::to_string(EGS) + "-" + SNAME[t] : "";
      bool ok = false;
      const uint32_t key = ekey(LAYERS - 1, t, 0), seed = eseed(LAYERS - 1, t, 0);
      const std::string g = eg::lin_gate(lin[0][t], d.W[LAYERS - 1][LJ[t]], key, seed, LAYERS - 1, w, ok);
      eg_fails += !ok;
      printf("{\"mode\": \"eg_gate\", \"M\": %d, \"stage\": \"%s\", \"gate\": %s}\n", M, SNAME[t], g.c_str());
      if (!w.empty()) {
        const std::string meta = "{\"m\": " + std::to_string(M) + ", \"n\": " + std::to_string(GN[t]) + ", \"k\": " + std::to_string(GK[t]) +
                                 ", \"S\": " + std::to_string(es[t]) + ", \"nw\": " + std::to_string(eg::lin_nw(lin[0][t])) +
                                 ", \"seed\": " + std::to_string(seed) + ", \"row_bytes\": 2048, \"tree\": \"R-" + SNAME[t] +
                                 "\", \"stage\": \"" + SNAME[t] + "\", \"layer_slot\": " + std::to_string(LAYERS - 1) + "}";
        cg::writef(w + "/dump.json", meta.data(), meta.size());
      }
    }
    // the serial reference: every chunk's layer-31 call of each linear and its receipts' tree
    for (int c = 0; c < K; c++)
      for (int t = 0; t < NSTAGE; t++) {
        if (!lin[c][t]) continue;
        eg::lin_qwe(lin[c][t], d.W[LAYERS - 1][LJ[t]], ekey(LAYERS - 1, t, c), d.st);
        eg::lin_gemm(lin[c][t], eseed(LAYERS - 1, t, c), d.st);
        eg::lin_rcommit(lin[c][t], LAYERS - 1, d.st);
      }
    CK(cudaStreamSynchronize(d.st));
    CK(cudaGetLastError());
    rc_ref = rc_roots(K);
    size_t leaves = 0, rbytes = 0;
    printf("{\"mode\": \"eg_setup\", \"M\": %d, \"S\": %d, \"grid\": \"%s\", \"stages\": [", M, EGS,
           getenv("EG_GRID") && !strcmp(getenv("EG_GRID"), "sms") ? "sms" : "tiles");
    bool first = true;
    for (int t = 0; t < NSTAGE; t++) {
      if (!lin[0][t]) continue;
      leaves += eg::lin_leaves(lin[0][t]);
      rbytes += eg::lin_rbytes(lin[0][t]);
      printf("%s{\"stage\": \"%s\", \"n\": %d, \"k\": %d, \"S\": %d, \"nw\": %d, \"leaves\": %zu, \"receipt_bytes\": %zu, "
             "\"ms_alone\": {\"w_d\": %.4f, \"w_d4q\": %.4f}, \"kernel\": \"%s\"}",
             first ? "" : ", ", SNAME[t], GN[t], GK[t], es[t], eg::lin_nw(lin[0][t]), eg::lin_leaves(lin[0][t]),
             eg::lin_rbytes(lin[0][t]), kt[t][0], kt[t][1], kern[t] ? "w_d4q" : "w_d");
      first = false;
    }
    printf("], \"receipt_leaves_a_layer\": %zu, \"receipt_bytes_a_layer\": %zu, \"x_bytes_a_layer\": %zu, \"fails\": %d}\n", leaves, rbytes,
           (size_t)M * (SK[0] + SK[3] + SK[4] + SK[5]), eg_fails);
    fflush(stdout);
  }
  std::vector<uint8_t> cg_ref;
  cg::u64* cg_hs[NSTAGE] = {};
  bool cg_host_equal = true;
  if (use_cg) {
    cg_poison(K);
    for (int c = 0; c < K; c++)
      for (int t = 0; t < NSTAGE; t++) {
        ch[c].cgc[t].salts<true>(nullptr, d.st);
        ch[c].cgc[t].tiles(ch[c].op[t], M, SK[t], CGR, false, d.st, LAYERS - 1);
        ch[c].cgc[t].nodes(d.st);
      }
    CK(cudaStreamSynchronize(d.st));
    CK(cudaGetLastError());
    cg_ref = cg_roots(K);
    if (use_cgh) {
      // the cgh arms' S records for layers 1..31 from host threads (cg::host_salt), a pinned buffer a stage (the bench's
      // stage-t salts are one function of pos for every chunk and layer), held to the GPU's slot 0 here and by the arms'
      // poisoned roots
      const unsigned nth = std::max(1u, std::thread::hardware_concurrency());
      auto t0 = std::chrono::steady_clock::now();
      for (int t = 0; t < NSTAGE; t++) {
        CK(cudaMallocHost(&cg_hs[t], cg_n[t] * 128));
        std::vector<std::thread> th;
        for (unsigned j = 0; j < nth; j++)
          th.emplace_back([&, t, j] {
            for (cg::u64 pos = j; pos < cg_n[t]; pos += nth) cg::host_salt(ch[0].cgc[t].hc, cgp[t].T.data(), pos, cg_hs[t] + 16 * pos);
          });
        for (auto& x : th) x.join();
      }
      const double fill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      const cg::u64 probe = std::min<cg::u64>(8192, cg_n[0]);
      cg::u64 tmp[16];
      volatile cg::u64 sink = 0;
      t0 = std::chrono::steady_clock::now();
      for (cg::u64 pos = 0; pos < probe; pos++) {
        cg::host_salt(ch[0].cgc[0].hc, cgp[0].T.data(), pos, tmp);
        sink = sink ^ tmp[0] ^ tmp[15];
      }
      const double ns_leaf = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / probe;
      for (int t = 0; t < NSTAGE; t++) {
        std::vector<cg::u64> g(cg_n[t] * 16);
        CK(cudaMemcpy(g.data(), ch[0].cgc[t].S, g.size() * 8, cudaMemcpyDeviceToHost));
        cg_host_equal &= memcmp(g.data(), cg_hs[t], g.size() * 8) == 0;
      }
      cg::u64 leaves = 0;
      for (int t = 0; t < NSTAGE; t++) leaves += cg_n[t];
      printf("{\"mode\": \"host_salt\", \"M\": %d, \"threads\": %u, \"leaves_a_layer\": %llu, \"fill_ms\": %.3f, \"ns_leaf_core\": %.1f, "
             "\"equal\": %s}\n",
             M, nth, (unsigned long long)leaves, fill_ms, ns_leaf, cg_host_equal ? "true" : "false");
      fflush(stdout);
    }
    // chunk 0's six commitments for reference.py verify: operand, every level, the CGP1 file
    const char* dump = getenv("CG_DUMP");
    if (dump && *dump) {
      for (int t = 0; t < NSTAGE; t++) {
        const cg::Committer& m = ch[0].cgc[t];
        const std::string w = std::string(dump) + "/M" + std::to_string(M) + "-R" + std::to_string(CGR) + "-" + SNAME[t];
        if (system(("mkdir -p " + w).c_str())) exit(4);
        std::vector<uint8_t> op((size_t)M * SK[t]);
        CK(cudaMemcpy(op.data(), ch[0].op[t], op.size(), cudaMemcpyDeviceToHost));
        cg::u64 used = 0;
        for (int l = 0; l <= m.depth; l++) used += m.width >> l;
        std::vector<cg::u64> lv(8 * used);
        CK(cudaMemcpy(lv.data(), m.levels, lv.size() * 8, cudaMemcpyDeviceToHost));
        std::vector<uint8_t> lb(lv.size() * 8);
        for (size_t i = 0; i < lv.size(); i++) cg::sha::put_be64(lb.data() + 8 * i, lv[i]);
        const std::vector<uint8_t> pf =
            cg::readf(dir + "/cg-" + std::to_string(M) + "-" + std::to_string(CGR) + "-" + std::to_string(t) + ".bin");
        cg::writef(w + "/operand.bin", op.data(), op.size());
        cg::writef(w + "/levels.bin", lb.data(), lb.size());
        cg::writef(w + "/params.bin", pf.data(), pf.size());
        const std::string meta = "{\"m\": " + std::to_string(M) + ", \"k\": " + std::to_string(SK[t]) + ", \"r\": " + std::to_string(CGR) +
                                 ", \"stage\": \"" + SNAME[t] + "\", \"layer_slot\": " + std::to_string(LAYERS - 1) + "}";
        cg::writef(w + "/dump.json", meta.data(), meta.size());
      }
    }
  }
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
  // the lite arms' stand-in commitments: a stage's compressions a row (segments, row, salt part, hm96 leaf, frame leaf,
  // node; 367 a token-layer in all) in 128-thread CTAs of lite_n chained compressions a thread
  const int LITE_C[NSTAGE] = {48, 67, 14, 48, 48, 143};
  const int lite_n = getenv("LITE_N") ? atoi(getenv("LITE_N")) : 8;
  int lite_grid[NSTAGE];
  for (int t = 0; t < NSTAGE; t++) lite_grid[t] = (int)(((long long)LITE_C[t] * M + 128LL * lite_n - 1) / (128LL * lite_n));
  // chained: a thread a segment (LITE_SEG chained), a thread a row (the rest less the nodes' 2), the CTAs' 128-leaf levels
  // and the levels above in one CTA, 2 compressions a node: about 40 dependent compressions a stage at 2K
  const int LITE_S[NSTAGE] = {4, 6, 1, 4, 4, 14}, LITE_SEG[NSTAGE] = {9, 9, 3, 9, 9, 9};
  const long long lite_cap = 1 << 20;
  uint64_t* lite_out;
  unsigned long long* lite_ts;
  CK(cudaMalloc(&lite_out, (size_t)std::max<long long>(lite_cap, (long long)*std::max_element(lite_grid, lite_grid + NSTAGE) * 128) *
                               sizeof(uint64_t)));
  CK(cudaMalloc(&lite_ts, 2 * sizeof(unsigned long long)));
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

  // CG_SKIP (timing only; the cg arms' roots then differ): any of salts, leaves, top drops those launches from the cg
  // arms' graphs, so a phase's share is the difference
  const std::string cg_skip = getenv("CG_SKIP") ? getenv("CG_SKIP") : "";
  const bool skip_salts = cg_skip.find("salts") != std::string::npos, skip_leaves = cg_skip.find("leaves") != std::string::npos,
             skip_top = cg_skip.find("top") != std::string::npos;
  printf("{\"mode\": \"ovl_setup\", \"M\": %d, \"k_max\": %d, \"draw_us\": %d, \"plain_rb\": %d, \"levels\": %d, \"cg_skip\": \"%s\", "
         "\"cg_diag\": \"%s\", "
         "\"fr_rot\": %d, \"carve\": %d, \"mid_ahead\": %d, \"lite_n\": %d, \"cg_r\": %d, \"cg_leaves\": [%llu, %llu, %llu, %llu, %llu, %llu], "
         "\"macs_chunk\": %.6e, \"gemm_alone_ms_chunk\": %.4f, \"mem_bytes\": {\"before\": "
         "%zu, \"att_standins\": %zu, \"activations\": %zu, \"commit_state\": %zu, \"cg_state\": %zu}, \"gemms\": [",
         M, K, draw_us, RB, levels, cg_skip.c_str(), CG_DIAG, FR_ROT, g_carve, g_mid, lite_n, use_cg ? CGR : 0, (unsigned long long)cg_n[0],
         (unsigned long long)cg_n[1], (unsigned long long)cg_n[2], (unsigned long long)cg_n[3], (unsigned long long)cg_n[4],
         (unsigned long long)cg_n[5], macs_chunk, gemm_alone_ms, mem0, mem_kb - mem0, mem_act - mem_kb, mem_cm - mem_act, mem_cg - mem_cm);
  for (int t = 0; t < NSTAGE; t++)
    printf("%s{\"stage\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"algo\": %d, \"ms_alone\": %.4f}", t ? ", " : "", SNAME[t], M,
           GN[t], GK[t], g[t].L.algo_id, g[t].L.ms);
  printf("]}\n");
  fflush(stdout);

  int fails = (cg_host_equal ? 0 : 1) + eg_fails;
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
      a.hash = S.hash && !S.lite;
      a.cg = S.hash && S.lite >= 3;
      a.stamps = S.draw;
      a.eg = S.eg;
      a.rc = S.eg && S.rc;
      if (a.cg)
        for (int c = 0; c < k; c++)
          for (int t = 0; t < NSTAGE; t++) {
            ch[c].cgc[t].fuse_top = S.fuse;
            ch[c].cgc[t].prefetch = S.pf;
          }
      if (a.rc)
        for (int c = 0; c < k; c++)
          for (int t = 0; t < NSTAGE; t++)
            if (lin[c][t]) eg::lin_flags(lin[c][t], S.fuse, S.pf);
      std::vector<cudaStream_t> sg(k), sh(k), ss(k), sr(k);
      for (int c = 0; c < k; c++) {
        const int pg = S.classes ? hi + std::min(levels - 1, c * levels / k) : S.hp < 0 ? hi : lo;
        sg[c] = prio_stream(pg);
        sh[c] = S.hp ? prio_stream(S.hp > 0 ? hi : lo) : sg[c];
        ss[c] = prio_stream(pg);
        a.streams.push_back(sg[c]);
        if (S.hp) a.streams.push_back(sh[c]);
        a.streams.push_back(ss[c]);
        if (a.rc) {
          sr[c] = prio_stream(S.rc > 0 ? hi : lo);
          a.streams.push_back(sr[c]);
        }
      }
      EvPool pool;
      std::vector<cudaEvent_t> gev(S.rr ? (size_t)k * NS : 0);
      for (auto& e : gev) e = pool.get();
      unsigned long long* st = a.st;
      a.G = capture(sm, [&] {
        stamp<<<1, 1, 0, sm>>>(st + (size_t)K * NSTAMP);
        // salt_ev[c][s]: stage.cu's salts for all layers at s < NSTAGE; commit_gpu's for stage s's own layer and stage
        std::vector<std::vector<cudaEvent_t>> salt_ev(k, std::vector<cudaEvent_t>(NS));
        for (int c = 0; c < k; c++) {
          pool.join(sm, sg[c]);
          if (!S.hash || (S.lite && S.lite < 3)) continue;
          pool.join(sg[c], ss[c]);
          for (int t = 0; t < NSTAGE; t++) {
            if (S.lite >= 3) {
              if (!skip_salts && S.lite != 5) ch[c].cgc[t].salts<true>(nullptr, ss[c], 0, 1);
            }
            else commit_salt(ch[c].cm[t], LAYERS, ss[c]);
            salt_ev[c][t] = pool.get();
            CK(cudaEventRecord(salt_ev[c][t], ss[c]));
          }
        }
        // the receipts' commitment of (c, t)'s last GEMM: that stage's next GEMM overwrites them after it
        std::vector<std::vector<cudaEvent_t>> rev(k, std::vector<cudaEvent_t>(NSTAGE, nullptr));
        for (int l = 0; l < LAYERS; l++)
          for (int t = 0; t < NSTAGE; t++)
            for (int c = 0; c < k; c++) {
              const int s = l * NSTAGE + t, idx = s * k + c;
              eg::Lin* const el = S.eg ? lin[c][t] : nullptr;
              if (el && !S.noqwe) eg::lin_qwe(el, d.W[l][LJ[t]], ekey(l, t, c), sg[c]);
              if (S.quant && lin[c][t]) eg::lin_quant(lin[c][t], sg[c]);
              if (S.hash) {
                if (sh[c] != sg[c]) pool.join(sg[c], sh[c]);
                if (S.lite == 1) {
                  sha_beside<<<lite_grid[t], 128, 0, sh[c]>>>(lite_n, lite_out, lite_ts);
                } else if (S.lite == 2) {
                  const long long segs = (long long)M * LITE_S[t];
                  lite_items<<<(unsigned)((segs + 127) / 128), 128, 0, sh[c]>>>(segs, LITE_SEG[t], lite_out, lite_cap);
                  lite_items<<<(unsigned)((M + 127) / 128), 128, 0, sh[c]>>>(M, LITE_C[t] - LITE_S[t] * LITE_SEG[t] - 2, lite_out,
                                                                           lite_cap);
                  lite_levels<<<(unsigned)((M + 127) / 128), 128, 0, sh[c]>>>(M, 2, lite_out);
                  if (M > 128) lite_levels<<<1, std::min(M / 128, 1024), 0, sh[c]>>>(M / 128, 2, lite_out);
                } else if (S.lite >= 3) {
                  // the next layer's salts for this stage on the chunk's salt stream, behind this commitment: they run
                  // beside the GEMMs, NSTAGE stages ahead of their use (cgh: the host's S records copied in instead)
                  CK(cudaStreamWaitEvent(sh[c], salt_ev[c][s], 0));
                  if (!skip_leaves) ch[c].cgc[t].tiles(ch[c].op[t], M, SK[t], CGR, false, sh[c], l);
                  if (!skip_top) ch[c].cgc[t].nodes(sh[c]);
                  if (l + 1 < LAYERS) {
                    pool.join(sh[c], ss[c]);
                    const cg::Committer& m = ch[c].cgc[t];
                    if (skip_salts || S.lite == 5) {
                    } else if (S.lite == 4) {
                      CK(cudaMemcpyAsync(m.S + 16 * m.n * (l + 1), cg_hs[t], m.n * 128, cudaMemcpyHostToDevice, ss[c]));
                    } else {
                      ch[c].cgc[t].salts<true>(nullptr, ss[c], l + 1, 1);
                    }
                    salt_ev[c][s + NSTAGE] = pool.get();
                    CK(cudaEventRecord(salt_ev[c][s + NSTAGE], ss[c]));
                  }
                } else {
                  if (l == 0) CK(cudaStreamWaitEvent(sh[c], salt_ev[c][t], 0));
                  commit_tree(ch[c].cm[t], l, ch[c].op[t], sh[c]);
                }
                if (sh[c] != sg[c]) pool.join(sh[c], sg[c]);
              }
              if (S.draw) draw_stamp<<<1, 1, 0, sg[c]>>>((unsigned long long)draw_us * 1000ull, st + (size_t)c * NSTAMP + 2 * s);
              if (S.rr && idx >= S.rr) CK(cudaStreamWaitEvent(sg[c], gev[idx - S.rr], 0));
              if (el) {
                if (rev[c][t]) CK(cudaStreamWaitEvent(sg[c], rev[c][t], 0));
                eg::lin_gemm(el, eseed(l, t, c), sg[c]);
              } else if (S.gemm) {
                ltb_run(g[t], ch[c].lt, ch[c].A, wt(l, t, c), ch[c].D, sg[c]);
              }
              if (S.rr) CK(cudaEventRecord(gev[idx], sg[c]));
              if (el && a.rc) {
                pool.join(sg[c], sr[c]);
                eg::lin_rcommit(el, l, sr[c]);
                rev[c][t] = pool.get();
                CK(cudaEventRecord(rev[c][t], sr[c]));
              }
            }
        for (int c = 0; c < k; c++) {
          if (S.draw) stamp<<<1, 1, 0, sg[c]>>>(st + (size_t)c * NSTAMP + 2 * NS);
          pool.join(sg[c], sm);
          if (a.rc) pool.join(sr[c], sm);
        }
      });
      arms.push_back(a);
    }
    bool any_pre = false;
    for (ArmRun& a : arms) any_pre |= a.cg && sched_of(a.name).lite == 5;
    if (any_pre) {
      cg_pre(k);
      const float pre_ms = cg_pre(k);
      cg::u64 leaves = 0;
      for (int t = 0; t < NSTAGE; t++) leaves += cg_n[t];
      printf("{\"mode\": \"salt_pre\", \"M\": %d, \"k\": %d, \"slots\": %d, \"leaves\": %llu, \"ms\": %.3f}\n", M, k, LAYERS,
             (unsigned long long)(leaves * LAYERS * k), pre_ms);
      fflush(stdout);
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
    // the same for commit_gpu's arms against its serial roots, with nohash's run as their control too
    std::vector<int> cg_ok(arms.size(), -1);
    for (size_t i = 0; use_cg && i < arms.size(); i++) {
      if (!arms[i].cg && arms[i].name.rfind("nohash", 0) != 0) continue;
      cg_poison(k);
      if (arms[i].cg && sched_of(arms[i].name).lite == 5) cg_pre(k);
      CK(cudaGraphLaunch(arms[i].G.ge, sm));
      CK(cudaStreamSynchronize(sm));
      CK(cudaGetLastError());
      const std::vector<uint8_t> got = cg_roots(k);
      cg_ok[i] = memcmp(got.data(), cg_ref.data(), got.size()) == 0;
      if (arms[i].cg ? !cg_ok[i] : cg_ok[i]) fails++;
    }
    // the receipts' roots of the edited arms against the serial path's (E's, which don't commit them, the control)
    std::vector<int> rc_ok(arms.size(), -1);
    for (size_t i = 0; use_rc && i < arms.size(); i++) {
      if (!arms[i].eg) continue;
      rc_poison_salt(k);
      CK(cudaGraphLaunch(arms[i].G.ge, sm));
      CK(cudaStreamSynchronize(sm));
      CK(cudaGetLastError());
      const std::vector<uint8_t> got = rc_roots(k);
      rc_ok[i] = memcmp(got.data(), rc_ref.data(), got.size()) == 0;
      if (arms[i].rc ? !rc_ok[i] : rc_ok[i]) fails++;
    }
    printf("{\"mode\": \"ovl\", \"M\": %d, \"k\": %d, \"eg_s\": %d, \"draw_us\": %d, \"plain_rb\": %d, \"macs\": %.6e, \"rejected\": %d, "
           "\"missing_algo\": %d, \"plain_gemms\": [",
           M, k, use_eg ? EGS : 0, draw_us, rb, macs_chunk * k, g_rejected_total, g_missing);
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
    printf("}, \"cg_roots\": {");
    first = true;
    for (size_t i = 0; i < arms.size(); i++) {
      if (cg_ok[i] < 0) continue;
      printf("%s\"%s\": %s", first ? "" : ", ", arms[i].name.c_str(), cg_ok[i] ? "\"equal\"" : "\"differ\"");
      first = false;
    }
    printf("}, \"eg_roots\": {");
    first = true;
    for (size_t i = 0; i < arms.size(); i++) {
      if (rc_ok[i] < 0) continue;
      printf("%s\"%s\": %s", first ? "" : ", ", arms[i].name.c_str(), rc_ok[i] ? "\"equal\"" : "\"differ\"");
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

// beside: whether the commitments' hashing can hide next to the GEMMs if it is shaped to fit beside their CTAs. plain's
// GEMMs for k sequences of M tokens (as in ovl) against H = C x k x M x 32 compressions (C = BESIDE_C, 367: a token-layer's
// segment, row, leaf, node and salt compressions in stage.cu's tree) in b CTAs of thr threads a SM on a stream at the
// highest priority: alone, as one launch just before the GEMMs (hfirst) or just after them (gfirst), or spread (a slice
// after each GEMM, waiting on it)
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
  // after(i) runs after the i-th GEMM launch
  auto gemms = [&](cudaStream_t s, const std::function<void(int)>& after) {
    int i = 0;
    for (int l = 0; l < LAYERS; l++)
      for (int t = 0; t < NSTAGE; t++) {
        if (LJ[t] >= 0)
          for (int r0 = 0; r0 < rows; r0 += rb) {
            ltb_run(pl[t], plt, Ap + (size_t)r0 * GK[t], d.W[l][LJ[t]], Dp, s);
            if (after) after(i);
            i++;
          }
        else {
          ltb_run(pl[t], plt, Ap, Kb[t == 1 ? l : (l + 1) % LAYERS], Dp, s);
          if (after) after(i);
          i++;
        }
      }
    return i;
  };
  const int nlaunch = LAYERS * (4 * ((rows + rb - 1) / rb) + 2);
  std::vector<cudaEvent_t> gev(nlaunch);
  for (auto& e : gev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  const double H = (double)C * k * M * LAYERS;
  // threads a CTA, CTAs a SM, the hash's preferred carveout (-1 the driver's default, 100 the GEMMs' most shared memory)
  const int cfg[][3] = {{32, 1, -1}, {64, 1, -1}, {128, 1, -1}, {128, 2, -1}, {256, 2, -1},
                        {32, 1, 100}, {128, 1, 100}, {128, 2, 100}, {256, 2, 100}};
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
  // one timed rep of `order`: 0 the GEMMs alone, 1 the hash alone, 2 hash then GEMMs, 3 GEMMs then hash, 4 spread: the
  // hash in one slice a GEMM, each after its GEMM (as a commitment of that GEMM's output)
  struct Out { double ms, mhz, t_hs, t_he, t_ge; };
  auto rep = [&](int order, int thr, int b, int n) -> Out {
    stamp<<<1, 1, 0, sg>>>(ts + 2);
    ts_reset<<<1, 1, 0, sg>>>(ts);
    CK(cudaEventRecord(e0, sg));
    CK(cudaEventRecord(ef, sg));
    CK(cudaStreamWaitEvent(sh, ef, 0));
    if (order == 1 || order == 2) sha_beside<<<b * d.sms, thr, 0, sh>>>(n, out, ts);
    const int ns = (n + nlaunch - 1) / nlaunch;
    if (order == 4)
      gemms(sg, [&](int i) {
        CK(cudaEventRecord(gev[i], sg));
        CK(cudaStreamWaitEvent(sh, gev[i], 0));
        sha_beside<<<b * d.sms, thr, 0, sh>>>(ns, out, ts);
      });
    else if (order != 1)
      gemms(sg, nullptr);
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
  rep(4, 128, 1, nlaunch);
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
    const int thr = cfg[ci][0], b = cfg[ci][1], carve = cfg[ci][2];
    CK(cudaFuncSetAttribute((const void*)sha_beside, cudaFuncAttributePreferredSharedMemoryCarveout, carve));
    const int n = (int)std::ceil(H / ((double)d.sms * b * thr));
    rep(1, thr, b, n);
    std::vector<Out> v[5];
    for (int r = 0; r < reps; r++)
      for (int o = 1; o <= 4; o++) v[o].push_back(rep(o, thr, b, n));
    printf("%s{\"thr\": %d, \"ctas_a_sm\": %d, \"carve\": %d, \"n\": %d, \"slices\": %d, \"n_slice\": %d", ci ? ", " : "", thr,
           b, carve, n, nlaunch, (n + nlaunch - 1) / nlaunch);
    const char* nm[5] = {"", "alone", "hfirst", "gfirst", "spread"};
    for (int o = 1; o <= 4; o++) {
      std::sort(v[o].begin(), v[o].end(), [](const Out& a, const Out& c) { return a.ms < c.ms; });
      const Out& m = v[o][v[o].size() / 2];
      const double span = m.t_he - m.t_hs;
      const double done = o == 4 ? (double)d.sms * b * thr * ((n + nlaunch - 1) / nlaunch) * nlaunch : (double)d.sms * b * thr * n;
      printf(", \"%s\": {\"ms\": %.3f, \"mhz\": %.0f, \"%s\": %.4f, \"hash_start_ms\": %.3f, \"hash_end_ms\": %.3f, "
             "\"gemm_end_ms\": %.3f, \"hash_rate\": %.4e}",
             nm[o], m.ms, m.mhz, o == 1 ? "of_gemm" : "cost", o == 1 ? m.ms / g_ms : m.ms / g_ms - 1, m.t_hs, m.t_he, m.t_ge,
             done / (span * 1e-3));
    }
    printf("}");
    fflush(stdout);
  }
  printf("]}\n");
  fflush(stdout);
  for (auto& e : gev) CK(cudaEventDestroy(e));
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
  if (getenv("CARVE") && *getenv("CARVE")) set_carve(atoi(getenv("CARVE")));
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
