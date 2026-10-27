// HIGHUTIL's period driver: an 8B model's linear layers back to back at full batch on one GPU, each call through
// V31-GPU's no-fold prover (v31.cu, vendored unmodified) or plain cuBLASLt, driven block by block by period.py over
// stdin and stdout.
//
//   hu CFG ROW LP T RMAX ACTKEY
//
// CFG holds shapes.txt (a line `NAME M N K` a layer, in call order), B-NAME.bin (the layer's registered N x K int8 weights)
// and params-NAME.bin (its CGP1 params). ACTKEY (16 hex digits) keys the activations: call c's A is act_value(ACTKEY, c,
// i K + k), generated on the GPU beside the other layers' GEMMs once the previous call of its layer has read its A
// (period.py's act_rows is its twin). RMAX bounds a block's steps (a step is one call of each layer).
//
// Commands, a line each:
//   block B G0 R ARM CHEAT SEEDS ROOTS   R steps from global step G0; ARM proto (v31's call, its commit beside the next
//                                        call's GEMMs) or plain (cuBLASLt); CHEAT none or v31's cheat, `@L` for layer L
//                                        only; SEEDS: R x layers records of 320 seed bytes (dA dB U V P) and the call's
//                                        64-byte salt key; ROOTS: the calls' value roots, 64 bytes each, written once
//                                        every tree is on the host. Answers `done B GPU_MS HOST_MS FLAGS SM_MHZ`.
//   paths B PICKS OUT                    PICKS: u32 (step in block, layer, leaf) triples; OUT: each one's siblings,
//                                        leaves first (64 bytes a level). Answers `paths B COUNT`.
//   quit
#define main v31_main
#include "v31.cu"
#undef main

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

HD u64 fmix64(u64 z) {
  z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull;
  z ^= z >> 27; z *= 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// entry e = i K + k of call c's A: a sum of four bytes, scaled (sd ~9, |a| <= 32), and one entry in 1024 a full-range
// outlier, so A's rows look like an absmax-quantized activation's and rarely wrap at V +-80
HD int act_value(u64 key, u64 call, u64 e) {
  const u64 h = fmix64(key ^ ((call << 32 | e) * 0x9E3779B97F4A7C15ull));
  const int s = (int)(h & 255) + (int)((h >> 8) & 255) + (int)((h >> 16) & 255) + (int)((h >> 24) & 255) - 510;
  if (((h >> 32) & 1023) == 0) { const int o = (int)(int8_t)(uint8_t)(h >> 40); return o < -127 ? -127 : o; }
  return s >> 4;
}
__global__ void act_kernel(int8_t* A, u64 total, u64 key, u64 call) {
  const u64 e0 = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 16;
  if (e0 >= total) return;
  alignas(16) int8_t v[16];
#pragma unroll
  for (int j = 0; j < 16; j++) v[j] = (int8_t)act_value(key, call, e0 + j);
  *reinterpret_cast<int4*>(A + e0) = *reinterpret_cast<const int4*>(v);
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
  int m, n, k;
  Ctx c;
  Lt lt;
  size_t lvb = 0;            // one call's levels, bytes
  int depth = 0;
  uint8_t* host = nullptr;   // pinned: RMAX calls' levels
  Dom* dom = nullptr;        // pinned: the layer's CD[0]
  u64 a_call = ~0ull;        // the call whose A the layer's buffer holds (or is being generated)
  cudaEvent_t ev_read, ev_gen, ev_cm, ev_cp;
};

int main(int argc, char** argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: hu CFG ROW LP T RMAX ACTKEY\n");
    return 2;
  }
  const std::string cfg = argv[1];
  const int row = atoi(argv[2]), lp = atoi(argv[3]), t = atoi(argv[4]), rmax = atoi(argv[5]);
  const u64 actkey = strtoull(argv[6], nullptr, 16);
  std::vector<Layer*> L;
  {
    std::ifstream f(cfg + "/shapes.txt");
    std::string name;
    int m, n, k;
    while (f >> name >> m >> n >> k) {
      Layer* l = new Layer;
      l->name = name; l->m = m; l->n = n; l->k = k;
      L.push_back(l);
    }
  }
  if (L.empty()) { fprintf(stderr, "%s/shapes.txt: no layers\n", cfg.c_str()); return 2; }
  const int NL = (int)L.size();
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t s0, s1, sg, cs;   // compute (high), commit, activations, tree copies
  CK(cudaStreamCreateWithPriority(&s0, cudaStreamNonBlocking, hi));
  CK(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, lo));
  CK(cudaStreamCreateWithPriority(&sg, cudaStreamNonBlocking, lo));
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
    const auto b = readfile(cfg + "/B-" + l->name + ".bin", (size_t)l->n * l->k);
    CK(cudaMemcpy(c.B, b.data(), b.size(), cudaMemcpyHostToDevice));
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
    for (cudaEvent_t* e : {&l->ev_read, &l->ev_gen, &l->ev_cm, &l->ev_cp}) {
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
  auto gen = [&](Layer* l, u64 call) {
    const u64 total = (u64)l->m * l->k;
    CK(cudaStreamWaitEvent(sg, l->ev_read));
    act_kernel<<<(unsigned)((total / 16 + 255) / 256), 256, 0, sg>>>(l->c.A, total, actkey, call);
    CK(cudaEventRecord(l->ev_gen, sg));
    l->a_call = call;
  };
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
          fwrite(lv + 64 * (off + (pos ^ 1)), 1, 64, f);
          off += width >> d;
          pos >>= 1;
        }
      }
      fclose(f);
      printf("paths %lld %zu\n", blk, np);
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
        const u64 call = (u64)(g0 + s) * NL + li;
        if (l->a_call != call) gen(l, call);
        CK(cudaStreamWaitEvent(s0, l->ev_gen));
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
          CK(cudaEventRecord(l->ev_read, s0));
          c.commit_ovl(b);
          CK(cudaEventRecord(l->ev_cm, s1));
          CK(cudaStreamWaitEvent(cs, l->ev_cm));
          CK(cudaMemcpyAsync(l->host + l->lvb * s, c.cm.levels, l->lvb, cudaMemcpyDeviceToHost, cs));
          CK(cudaEventRecord(l->ev_cp, cs));
        } else {
          lt_run(l->lt, c.A, c.B, c.O[0], s0);
          CK(cudaEventRecord(l->ev_read, s0));
        }
        gen(l, call + NL);
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
        for (Layer* l : L) fwrite(l->host + l->lvb * s + l->lvb - 64, 1, 64, f);
      fclose(f);
    }
    std::sort(mhz.begin(), mhz.end());
    printf("done %lld %.3f %.3f %d %u\n", blk, gpu_ms, host_ms, flags, mhz.empty() ? 0u : mhz[mhz.size() / 2]);
    fflush(stdout);
  }
  return 0;
}
