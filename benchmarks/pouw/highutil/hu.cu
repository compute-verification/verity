// HIGHUTIL's period driver: an 8B model's linear layers back to back at full batch on one GPU, each call through
// V31-GPU's no-fold prover (v31.cu, vendored unmodified) or plain cuBLASLt, driven block by block by period.py over
// stdin and stdout.
//
//   hu CFG ROW LP T RMAX
//
// CFG holds shapes.txt (a line `NAME M N K NB` a layer, in call order), A-NAME.bin and B-NAME.bin (the layer's registered
// operands for NB transformer blocks, block-major: NB x M x K and NB x N x K int8) and params-NAME.bin (its CGP1 params).
// Both operands stay on the GPU; step g's calls read block g mod NB. RMAX bounds a block's steps (a step is one call of
// each layer).
//
// Commands, a line each:
//   block B G0 R ARM CHEAT SEEDS ROOTS   R steps from global step G0; ARM proto (v31's call, its commit beside the next
//                                        call's GEMMs) or plain (cuBLASLt); CHEAT none or v31's cheat, `@L` for layer L
//                                        only; SEEDS: R x layers records of 320 seed bytes (dA dB U V P) and the call's
//                                        64-byte salt key; ROOTS: the calls' value roots, 64 bytes each, written once
//                                        every tree is on the host. Answers `done B GPU_MS HOST_MS FLAGS SM_MHZ`.
//   paths B PICKS OUT                    PICKS: u32 (step in block, layer, leaf) triples; OUT: each one's siblings,
//                                        leaves first (64 bytes a level). Answers `paths B COUNT`.
//   dump LAYER TILE DIR                  the layer's last protocol call's tile: the A and B rows it read, its words, O
//                                        and leaves (period.py diag). Answers `dumped LAYER TILE call C`.
//   quit
// v31.cu's CUTLASS baseline names phc_prepared even when it isn't linked in; its header declares it
#include "../harness/native/cutlass_api.h"
#define main v31_main
#include "v31.cu"
#undef main

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

// a tree node as the device holds it (8 u64 words) to its digest's 64 bytes
static void put_node(FILE* f, const uint8_t* p) {
  uint8_t out[64];
  for (int w = 0; w < 8; w++) {
    u64 v;
    memcpy(&v, p + 8 * w, 8);
    sha::put_be64(out + 8 * w, v);
  }
  fwrite(out, 1, 64, f);
}

static Consts consts_with(const Consts& base, const uint8_t* ks) {
  Consts c = base;
  words_of(ks, c.ks, 8);
  u64 blk[16] = {}, st[8];
  memcpy(blk, c.ks, 64);
  sha::iv(st);
  sha::rounds_prefix(st, blk, 8, c.ks_m8);
  return c;
}

struct Layer {
  std::string name;
  int m, n, k, nb;
  Ctx c;
  Lt lt;
  int8_t *As = nullptr, *Bs = nullptr;   // the NB blocks' operands
  size_t lvb = 0;            // one call's levels, bytes
  int depth = 0;
  uint8_t* host = nullptr;   // pinned: RMAX calls' levels
  Dom* dom = nullptr;        // pinned: the layer's CD[0]
  u64 last_call = 0;         // the layer's last protocol call, and its step in its block
  int last_step = 0;
  cudaEvent_t ev_cm, ev_cp;
};

int main(int argc, char** argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: hu CFG ROW LP T RMAX\n");
    return 2;
  }
  const std::string cfg = argv[1];
  const int row = atoi(argv[2]), lp = atoi(argv[3]), t = atoi(argv[4]), rmax = atoi(argv[5]);
  std::vector<Layer*> L;
  {
    std::ifstream f(cfg + "/shapes.txt");
    std::string name;
    int m, n, k, nb;
    while (f >> name >> m >> n >> k >> nb) {
      Layer* l = new Layer;
      l->name = name; l->m = m; l->n = n; l->k = k; l->nb = nb;
      L.push_back(l);
    }
  }
  if (L.empty()) { fprintf(stderr, "%s/shapes.txt: no layers\n", cfg.c_str()); return 2; }
  const int NL = (int)L.size();
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t s0, s1, cs;   // compute (high), commit, tree copies
  CK(cudaStreamCreateWithPriority(&s0, cudaStreamNonBlocking, hi));
  CK(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, lo));
  CK(cudaStreamCreateWithPriority(&cs, cudaStreamNonBlocking, lo));
  void *symC, *symCD;
  CK(cudaGetSymbolAddress(&symC, C));
  CK(cudaGetSymbolAddress(&symCD, CD));
  for (Layer* l : L) {
    Ctx& c = l->c;
    c.setup(l->m, l->n, l->k, lp, t, row, cfg + "/params-" + l->name + ".bin", std::max(64, l->k / 8), 2);
    if (c.nb != 2) { fprintf(stderr, "%s: room for %d copies of the values, want 2\n", l->name.c_str(), c.nb); return 3; }
    CK(cudaStreamSynchronize(c.s1));
    c.s0 = s0; c.s1 = s1;
    for (auto [kind, rows, dst] : {std::tuple{"A", l->m, &l->As}, std::tuple{"B", l->n, &l->Bs}}) {
      const auto x = readfile(cfg + "/" + kind + "-" + l->name + ".bin", (size_t)l->nb * rows * l->k);
      *dst = dalloc<int8_t>(x.size());
      CK(cudaMemcpy(*dst, x.data(), x.size(), cudaMemcpyHostToDevice));
    }
    c.A = l->As; c.B = l->Bs;
    const Consts again = consts_with(c.cm.hc, c.params.ks);
    if (memcmp(&c.cm.hc, &again, sizeof(Consts))) {
      fprintf(stderr, "%s: consts_with doesn't rebuild the committer's constants\n", l->name.c_str());
      return 3;
    }
    l->lvb = c.levels_count() * 64;
    l->depth = c.cm.depth;
    CK(cudaHostAlloc((void**)&l->host, l->lvb * rmax, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**)&l->dom, sizeof(Dom), cudaHostAllocDefault));
    *l->dom = c.cm.hd;
    lt_setup(l->lt, l->m, l->n, l->k, c.A, c.B, c.O[0], s0);
    for (cudaEvent_t* e : {&l->ev_cm, &l->ev_cp}) {
      CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
      CK(cudaEventRecord(*e, s0));
    }
  }
  Consts* pc = nullptr;   // pinned: a block's calls' committer constants
  CK(cudaHostAlloc((void**)&pc, sizeof(Consts) * (size_t)rmax * NL, cudaHostAllocDefault));
  CK(cudaDeviceSynchronize());
  gk::Nvml nv(0);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  {
    size_t fr, tot;
    CK(cudaMemGetInfo(&fr, &tot));
    printf("ready layers %d depth", NL);
    for (Layer* l : L) printf(" %d", l->depth);
    printf(" free_gb %.1f\n", fr / 1e9);
    fflush(stdout);
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    if (cmd == "quit") break;
    if (cmd == "paths") {
      long long blk;
      std::string picks, out;
      in >> blk >> picks >> out;
      const auto p = readf(picks);
      FILE* f = fopen(out.c_str(), "wb");
      const size_t np = p.size() / 12;
      for (size_t i = 0; i < np; i++) {
        u32 r[3];
        memcpy(r, p.data() + 12 * i, 12);
        if ((int)r[0] >= rmax || (int)r[1] >= NL || r[2] >= L[r[1]]->c.leaves) { fprintf(stderr, "paths: bad pick %zu\n", i); return 2; }
        const Layer* l = L[r[1]];
        const uint8_t* lv = l->host + l->lvb * r[0];
        const u64 width = l->c.cm.width;
        u64 off = 0, pos = r[2];
        for (int d = 0; d < l->depth; d++) {
          put_node(f, lv + 64 * (off + (pos ^ 1)));
          off += width >> d;
          pos >>= 1;
        }
      }
      fclose(f);
      printf("paths %lld %zu\n", blk, np);
      fflush(stdout);
      continue;
    }
    if (cmd == "dump") {
      int li, tile;
      std::string dir;
      in >> li >> tile >> dir;
      Layer* l = L[li];
      Ctx& c = l->c;
      const int b = (c.ovl - 1) & 1, tm = tile / c.tiles_n, tn = tile % c.tiles_n;
      CK(cudaDeviceSynchronize());
      gk::Dump d(dir);
      d.bytes("A", c.A + (size_t)tm * TILE * l->k, (size_t)TILE * l->k, s0);
      d.bytes("B", c.B + (size_t)tn * TILE * l->k, (size_t)TILE * l->k, s0);
      d.bytes("words", c.words[b] + (size_t)tile * c.S * TILE * TILE, (size_t)c.S * 32768, s0);
      std::vector<int> o((size_t)TILE * TILE);
      for (int r = 0; r < TILE; r++)
        CK(cudaMemcpy(o.data() + (size_t)r * TILE, c.O[b] + (size_t)(tm * TILE + r) * c.n + tn * TILE, TILE * 4, cudaMemcpyDeviceToHost));
      FILE* f = fopen((dir + "/O.bin").c_str(), "wb");
      fwrite(o.data(), 4, o.size(), f);
      fclose(f);
      f = fopen((dir + "/leaves.bin").c_str(), "wb");
      for (int x = 0; x < c.lpt; x++) put_node(f, l->host + l->lvb * l->last_step + 64 * ((size_t)tile * c.lpt + x));
      fclose(f);
      printf("dumped %d %d call %llu\n", li, tile, (unsigned long long)l->last_call);
      fflush(stdout);
      continue;
    }
    if (cmd != "block") { fprintf(stderr, "unknown command %s\n", cmd.c_str()); return 2; }
    long long blk, g0;
    int R;
    std::string arm, cheat, seedsf, rootsf;
    in >> blk >> g0 >> R >> arm >> cheat >> seedsf >> rootsf;
    if (R < 1 || R > rmax || (arm != "proto" && arm != "plain")) { fprintf(stderr, "block: R 1..%d, ARM proto|plain\n", rmax); return 2; }
    int cheat_layer = -1;
    {
      const size_t at = cheat.find('@');
      if (at != std::string::npos) { cheat_layer = atoi(cheat.c_str() + at + 1); cheat = cheat.substr(0, at); }
    }
    const Cheat ch = parse_cheat(cheat);
    const bool proto = arm == "proto";
    std::vector<uint8_t> sd;
    if (proto) {
      sd = readfile(seedsf, (size_t)R * NL * 384);
      for (int s = 0; s < R; s++)
        for (int li = 0; li < NL; li++) pc[(size_t)s * NL + li] = consts_with(L[li]->c.cm.hc, sd.data() + ((size_t)s * NL + li) * 384 + 320);
    }
    const auto h0 = std::chrono::steady_clock::now();
    std::vector<unsigned> mhz;
    CK(cudaEventRecord(e0, s0));
    for (int s = 0; s < R; s++) {
      for (int li = 0; li < NL; li++) {
        Layer* l = L[li];
        Ctx& c = l->c;
        const u64 call = (u64)(g0 + s) * NL + li, blk_t = (u64)(g0 + s) % l->nb;
        c.A = l->As + blk_t * l->m * l->k;
        c.B = l->Bs + blk_t * l->n * l->k;
        if (proto) {
          const size_t j = (size_t)s * NL + li;
          CK(cudaStreamWaitEvent(s1, l->ev_cp));
          CK(cudaMemcpyAsync(symC, pc + j, sizeof(Consts), cudaMemcpyHostToDevice, s1));
          CK(cudaMemcpyAsync(symCD, l->dom, sizeof(Dom), cudaMemcpyHostToDevice, s1));
          c.set_seeds(sd.data() + j * 384);
          const int b = c.ovl++ & 1;
          CK(cudaStreamWaitEvent(s0, c.ev_done[b]));
          c.compute(s0, b, cheat_layer < 0 || cheat_layer == li ? ch : Cheat{});
          CK(cudaEventRecord(c.ev_comp[b], s0));
          c.commit_ovl(b);
          l->last_call = call;
          l->last_step = s;
          CK(cudaEventRecord(l->ev_cm, s1));
          CK(cudaStreamWaitEvent(cs, l->ev_cm));
          CK(cudaMemcpyAsync(l->host + l->lvb * s, c.cm.levels, l->lvb, cudaMemcpyDeviceToHost, cs));
          CK(cudaEventRecord(l->ev_cp, cs));
        } else {
          lt_run(l->lt, c.A, c.B, c.O[0], s0);
        }
      }
      const gk::Sample smp = nv.sample();
      if (smp.sm_mhz) mhz.push_back(smp.sm_mhz);
    }
    for (Layer* l : L) {
      CK(cudaStreamWaitEvent(s0, l->ev_cp));
      CK(cudaStreamWaitEvent(s0, l->ev_cm));
    }
    CK(cudaEventRecord(e1, s0));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    const double host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
    float gpu_ms = 0.f;
    CK(cudaEventElapsedTime(&gpu_ms, e0, e1));
    int flags = 0;
    for (Layer* l : L) flags |= l->c.flags();
    if (proto) {
      FILE* f = fopen(rootsf.c_str(), "wb");
      for (int s = 0; s < R; s++)
        for (Layer* l : L) put_node(f, l->host + l->lvb * s + l->lvb - 64);
      fclose(f);
    }
    std::sort(mhz.begin(), mhz.end());
    printf("done %lld %.3f %.3f %d %u\n", blk, gpu_ms, host_ms, flags, mhz.empty() ? 0u : mhz[mhz.size() / 2]);
    fflush(stdout);
  }
  return 0;
}
