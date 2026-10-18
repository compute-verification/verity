// hm96lat: latency levers for full SHA-512 in hm96 format (hm96.cu's kernels and Arm, its base arm only; build with
// -DCAND_KIND=0) at decode-sized windows of the fill operand. Levers:
//   base       hm96.cu's tree: a launch a level while the level has more than 256 nodes, then one CTA of 256 threads
//   top512, top1024   the same with one CTA of 512 / 1024 threads from the level with that many nodes on
//   host4, host6, host8   the GPU up to H levels below the root, that level's 2^H nodes to pinned host memory, the last H
//              levels on the host (sha::digest, one thread)
//   k4, k8     FORMAT CHANGE (hypothetical): a 4-ary / 8-ary frame tree, node = H(frame "node" did uint(level) uint(index)
//              c_0 .. c_(k-1)), width k^d, pads and leaves as the binary tree's (hm96lat_ref.py)
//
//   hm96lat verify PARAMS N DIR    DIR/base: hm96.cu's dump (hm96_ref.py check); every binary lever's levels (top) or root
//                                  (host) against hm96.cu's own tree's, one JSON line each; DIR/k4, DIR/k8: the k-ary levels
//                                  (hm96lat_ref.py check)
//   hm96lat bench PDIR REPS N[,N]  JSON lines "lat": each lever, rep by rep in rotating order, salts precomputed (pre: the
//                                  data phase) and on the step (salt phase + data phase), each as one CUDA graph that ends
//                                  with the root (or the host's level) in pinned memory: "host" the host clock from graph
//                                  launch to the root's bytes on the host, "gpu" the graph queued behind a spin (events)
//   hm96lat overlap PDIR REPS TARGET_MS N[,N]   a memory-bound stand-in for a decode step on stream A (KS kernels of a
//                                  calibrated streaming read, one graph, TARGET_MS alone), base's commit launched on stream
//                                  B at default and at the highest priority: the stand-in alone and with the commit, the
//                                  commit alone, and when its root is written
#define HM96_NO_MAIN
#include "hm96.cu"

#include <chrono>
#include <functional>

using PS = Base;

// --- binary levers ----------------------------------------------------------------------------------------------------
template <int NT>
__global__ void __launch_bounds__(NT, 1) topn_kernel(u64* lv, u64 width, u64 n, int from, int to, int pads) {
  for (int l = from; l <= to; ++l) {
    const u64 c = cdiv_pow2(n, l), lo = pads ? c : 0, hi = pads ? (width >> l) : c;
    const u64* in = lv + level_off(width, l - 1);
    u64* out = lv + level_off(width, l);
    for (u64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      u64 o[8];
      node<PS, 0>(l - 1, i, in + 16 * i, in + 16 * i + 8, o);
      st8(out + 8 * i, o);
    }
    __syncthreads();
  }
}

struct Bin {
  Arm<PS, 0>* a;
  int nt = 256, host = 0;   // the top CTA's threads; levels on the host
  int stop() const { return a->depth - host; }
  void tree(bool pads) {
    const int to = stop();
    int l = 1;
    for (; l <= to && (a->width >> l) > (u64)nt; ++l) {
      const u64 c = cdiv_pow2(a->n, l), lo = pads ? c : 0, hi = pads ? (a->width >> l) : c;
      if (hi > lo)
        level_kernel<PS, 0><<<(unsigned)((hi - lo + THREADS - 1) / THREADS), THREADS, 0, a->st>>>(
            a->lv + level_off(a->width, l - 1), a->lv + level_off(a->width, l), lo, hi, l - 1);
    }
    if (l > to) return;
    const int p = pads ? 1 : 0;
    if (nt == 256) topn_kernel<256><<<1, 256, 0, a->st>>>(a->lv, a->width, a->n, l, to, p);
    else if (nt == 512) topn_kernel<512><<<1, 512, 0, a->st>>>(a->lv, a->width, a->n, l, to, p);
    else topn_kernel<1024><<<1, 1024, 0, a->st>>>(a->lv, a->width, a->n, l, to, p);
  }
  void salt() {
    salt_kernel<PS, 0><<<(unsigned)((a->n + THREADS - 1) / THREADS), THREADS, Arm<PS, 0>::TSMEM, a->st>>>(a->T, a->n, a->S, a->F);
    if (a->width > a->n)
      pad_kernel<PS, 0><<<(unsigned)((a->width - a->n + THREADS - 1) / THREADS), THREADS, 0, a->st>>>(a->n, a->width, a->lv);
    tree(true);
  }
  void data(const uint8_t* src) {
    const u64 segs = a->n * NSEG, grid = (segs + SEG_THREADS - 1) / SEG_THREADS;
    seg_kernel<PS, 0><<<(unsigned)grid, SEG_THREADS, Arm<PS, 0>::SEG_SMEM, a->st>>>(src, segs, a->D);
    leaf_kernel<PS, 0><<<(unsigned)((a->n + THREADS - 1) / THREADS), THREADS, 0, a->st>>>(a->n, a->D, a->S, a->F, a->lv, a->bc);
    tree(false);
  }
  const u64* out_level() const { return a->lv + level_off(a->width, stop()); }
  u64 out_nodes() const { return a->width >> stop(); }
};

// the host's levels: a node from hm96_ref's template (HP.node_tpl), digests as the device's words
static void host_node(const HP& hp, int level, u64 index, const u64* L, const u64* R, u64* out) {
  const int ni = index < 256 ? 1 : index < 65536 ? 2 : 3, len = 82 + 3 * 64 + ni;
  uint8_t m[8 * MAXW];
  for (int i = 0; i < MAXW; ++i) sha::put_be64(m + 8 * i, hp.node_tpl[ni][i]);
  m[57 + 64] = (uint8_t)level;
  for (int b = 0; b < ni; ++b) m[66 + 64 + b] = (uint8_t)(index >> (8 * (ni - 1 - b)));
  for (int i = 0; i < 8; ++i) {
    sha::put_be64(m + 74 + 64 + ni + 8 * i, L[i]);
    sha::put_be64(m + 82 + 128 + ni + 8 * i, R[i]);
  }
  uint8_t d[64];
  sha::digest(m, len, d);
  for (int i = 0; i < 8; ++i) out[i] = sha::be64(d + 8 * i);
}
// nodes: the 2^h nodes of level depth - h (words), in place; the root in nodes[0..8)
static void host_levels(const HP& hp, int depth, int h, u64* nodes) {
  for (int l = depth - h + 1; l <= depth; ++l) {
    const u64 m = 1ull << (depth - l);
    for (u64 i = 0; i < m; ++i) host_node(hp, l - 1, i, nodes + 16 * i, nodes + 16 * i + 8, nodes + 8 * i);
  }
}

// --- k-ary trees (FORMAT CHANGE) --------------------------------------------------------------------------------------
constexpr int KTW = 96;                 // words of the 8-ary node's longest message (709 bytes: 6 blocks)
__constant__ u64 KT[2][4][KTW];        // [k4, k8][len(uint(index))]: the whole message, variable fields zero
template <int K>
__host__ __device__ inline u64 offk(u64 width, int l) {
  u64 o = 0;
  for (int i = 0; i < l; ++i, width /= K) o += width;
  return 8 * o;
}
template <int K>
__host__ __device__ inline u64 cdivk(u64 n, int l) {
  u64 d = 1;
  for (int i = 0; i < l; ++i) d *= K;
  return (n + d - 1) / d;
}
template <int K, int J, int OFF>
DEV void put_children(u64* m, const u64* ch) {
  if constexpr (J < K) {
    or_dig<PS, OFF + J * 72>(m, ch + 8 * J);
    put_children<K, J + 1, OFF>(m, ch);
  }
}
template <int K, int NI>
DEV void nodek_ni(int level, u64 index, const u64* ch, u64* out) {
  constexpr int LEN = 66 + 8 * K + 64 * (1 + K) + NI, NW = 16 * PS::nblk(LEN);
  static_assert(NW <= KTW, "KT");
  u64 m[NW];
  ldc(m, KT[K == 8][NI], NW);
  or_be<PS, 66 + 64, NI>(m, index);
  put_children<K, 0, 74 + 64 + NI>(m, ch);
  ldc(out, HPC[0].node_mid[level], 8);   // the level is in the first block, as the binary node's
  finish<PS, LEN, 1>(out, m);
}
template <int K>
DEV void nodek(int level, u64 index, const u64* ch, u64* out) {
  switch (nbytes(index)) {
    case 1: nodek_ni<K, 1>(level, index, ch, out); break;
    case 2: nodek_ni<K, 2>(level, index, ch, out); break;
    default: nodek_ni<K, 3>(level, index, ch, out); break;
  }
}
template <int K>
__global__ void __launch_bounds__(THREADS) levelk_kernel(const u64* __restrict__ in, u64* __restrict__ out, u64 lo, u64 hi, int d) {
  const u64 i = lo + (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hi) return;
  u64 o[8];
  nodek<K>(d, i, in + 8 * K * i, o);
  st8(out + 8 * i, o);
}
template <int K>
__global__ void __launch_bounds__(THREADS, 1) topk_kernel(u64* lv, u64 width, u64 n, int from, int depth, int pads) {
  for (int l = from; l <= depth; ++l) {
    const u64 c = cdivk<K>(n, l), lo = pads ? c : 0, hi = pads ? cdivk<K>(width, l) : c;
    const u64* in = lv + offk<K>(width, l - 1);
    u64* out = lv + offk<K>(width, l);
    for (u64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      u64 o[8];
      nodek<K>(l - 1, i, in + 8 * K * i, o);
      st8(out + 8 * i, o);
    }
    __syncthreads();
  }
}
template <int K>
struct ArmK {
  Arm<PS, 0>* a;
  u64 width = 1, nodes = 0;
  int depth = 0;
  u64* lv = nullptr;
  void init(Arm<PS, 0>* a_) {
    a = a_;
    while (width < a->n) { width *= K; ++depth; }
    for (u64 w = width; w; w /= K) nodes += w;
    CK(cudaMalloc(&lv, nodes * 64));
  }
  void tree(bool pads) {
    for (int l = 1; l <= depth; ++l) {
      const u64 m = cdivk<K>(width, l), c = cdivk<K>(a->n, l), lo = pads ? c : 0, hi = pads ? m : c;
      if (m <= (u64)THREADS) {
        topk_kernel<K><<<1, THREADS, 0, a->st>>>(lv, width, a->n, l, depth, pads ? 1 : 0);
        return;
      }
      if (hi > lo)
        levelk_kernel<K><<<(unsigned)((hi - lo + THREADS - 1) / THREADS), THREADS, 0, a->st>>>(lv + offk<K>(width, l - 1),
                                                                                             lv + offk<K>(width, l), lo, hi, l - 1);
    }
  }
  void salt() {
    salt_kernel<PS, 0><<<(unsigned)((a->n + THREADS - 1) / THREADS), THREADS, Arm<PS, 0>::TSMEM, a->st>>>(a->T, a->n, a->S, a->F);
    if (width > a->n) pad_kernel<PS, 0><<<(unsigned)((width - a->n + THREADS - 1) / THREADS), THREADS, 0, a->st>>>(a->n, width, lv);
    tree(true);
  }
  void data(const uint8_t* src) {
    const u64 segs = a->n * NSEG, grid = (segs + SEG_THREADS - 1) / SEG_THREADS;
    seg_kernel<PS, 0><<<(unsigned)grid, SEG_THREADS, Arm<PS, 0>::SEG_SMEM, a->st>>>(src, segs, a->D);
    leaf_kernel<PS, 0><<<(unsigned)((a->n + THREADS - 1) / THREADS), THREADS, 0, a->st>>>(a->n, a->D, a->S, a->F, lv, a->bc);
    tree(false);
  }
  const u64* root() const { return lv + offk<K>(width, depth); }
};
// KT from the binary node's template: its bytes up to child 0's length field, then k times (u64be 64, 64 zero bytes)
static void load_kt(const HP& hp) {
  static u64 kt[2][4][KTW];
  memset(kt, 0, sizeof kt);
  for (int s = 0; s < 2; ++s) {
    const int K = s ? 8 : 4;
    for (int ni = 1; ni <= 3; ++ni) {
      uint8_t bin[8 * MAXW], m[8 * KTW];
      memset(m, 0, sizeof m);
      for (int i = 0; i < MAXW; ++i) sha::put_be64(bin + 8 * i, hp.node_tpl[ni][i]);
      const int head = 66 + 64 + ni;
      memcpy(m, bin, head);
      for (int j = 0; j < K; ++j) sha::put_be64(m + head + 72 * j, 64);
      for (int i = 0; i < KTW; ++i) kt[s][ni][i] = sha::be64(m + 8 * i);
    }
  }
  CK(cudaMemcpyToSymbol(KT, kt, sizeof kt));
}

// --- the levers as graphs ending with the output level in pinned memory --------------------------------------------------
struct Lever {
  std::string name;
  int host = 0, depth = 0;      // levels on the host; the binary depth
  u64 out_nodes = 1;
  cudaGraphExec_t g[2]{};       // pre, step
  u64* pin = nullptr;           // the output level's nodes
};

static cudaGraphExec_t capture(cudaStream_t s, const std::function<void()>& f) {
  cudaGraph_t g;
  cudaGraphExec_t x;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  f();
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&x, g, 0));
  CK(cudaGraphDestroy(g));
  return x;
}

struct Setup {
  Params p;
  const HP* hp;
  uint8_t* src;
  Arm<PS, 0> a;
  ArmK<4> k4;
  ArmK<8> k8;
  std::vector<Lever> levers;
  cudaStream_t s;
  Setup(const std::string& params, u64 n, bool with_bc, cudaStream_t s_) : s(s_) {
    p = load_params(params);
    hp = reinterpret_cast<const HP*>(p.raw.data() + 4);
    load_kt(*hp);
    src = operand(n);
    a.init(n, p.t0, with_bc);
    a.st = s;
    k4.init(&a);
    k8.init(&a);
    CK(cudaDeviceSynchronize());
  }
  void build() {
    auto bin = [&](const std::string& name, int nt, int host) {
      Lever L{name, host, a.depth};
      Bin b{&a, nt, host};
      L.out_nodes = b.out_nodes();
      CK(cudaMallocHost(&L.pin, L.out_nodes * 64));
      const u64* lvl = b.out_level();
      L.g[0] = capture(s, [&] { b.data(src); CK(cudaMemcpyAsync(L.pin, lvl, L.out_nodes * 64, cudaMemcpyDeviceToHost, s)); });
      L.g[1] = capture(s, [&] { b.salt(); b.data(src); CK(cudaMemcpyAsync(L.pin, lvl, L.out_nodes * 64, cudaMemcpyDeviceToHost, s)); });
      levers.push_back(L);
    };
    bin("base", 256, 0);
    bin("top512", 512, 0);
    bin("top1024", 1024, 0);
    for (int h : {4, 6, 8})
      if (h < a.depth) bin("host" + std::to_string(h), 256, h);
    auto kary = [&](const std::string& name, auto& k) {
      Lever L{name, 0, a.depth};
      CK(cudaMallocHost(&L.pin, 64));
      L.g[0] = capture(s, [&] { k.data(src); CK(cudaMemcpyAsync(L.pin, k.root(), 64, cudaMemcpyDeviceToHost, s)); });
      L.g[1] = capture(s, [&] { k.salt(); k.data(src); CK(cudaMemcpyAsync(L.pin, k.root(), 64, cudaMemcpyDeviceToHost, s)); });
      levers.push_back(L);
    };
    kary("k4", k4);
    kary("k8", k8);
  }
  // one run of a lever: the root on the host (the host's levels done), host ms
  double run_host(Lever& L, int m) {
    const auto t0 = std::chrono::steady_clock::now();
    CK(cudaGraphLaunch(L.g[m], s));
    CK(cudaStreamSynchronize(s));
    if (L.host) host_levels(*hp, L.depth, L.host, L.pin);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
};

static std::vector<u64> parse_sizes(const char* p) {
  std::vector<u64> v;
  while (*p) { v.push_back(strtoull(p, (char**)&p, 10)); if (*p == ',') ++p; }
  return v;
}
static std::string hex(const u64* w) {
  char b[129];
  for (int i = 0; i < 8; ++i) snprintf(b + 16 * i, 17, "%016llx", (unsigned long long)w[i]);
  return b;
}

static int verify(const std::string& params, u64 n, const std::string& dir) {
  dump(params, dir + "/base", n);
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  Setup S(params, n, false, s);
  S.build();
  const u64 nodes = 2 * S.a.width - 1;
  std::vector<u64> want(8 * nodes), got(8 * nodes);
  S.a.salt();
  S.a.data(S.src);
  CK(cudaStreamSynchronize(s));
  CK(cudaMemcpy(want.data(), S.a.lv, nodes * 64, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (auto& L : S.levers) {
    if (L.name == "k4" || L.name == "k8") continue;
    CK(cudaMemset(S.a.lv, 0, nodes * 64));
    CK(cudaDeviceSynchronize());
    S.run_host(L, 1);
    bool ok = !memcmp(L.pin, want.data() + 8 * (nodes - 1), 64);
    long diff = -1;
    if (!L.host) {
      CK(cudaMemcpy(got.data(), S.a.lv, nodes * 64, cudaMemcpyDeviceToHost));
      diff = 0;
      for (u64 i = 0; i < 8 * nodes; ++i) diff += got[i] != want[i];
      ok &= diff == 0;
    }
    bad += !ok;
    printf("{\"mode\": \"verify\", \"lever\": \"%s\", \"leaves\": %llu, \"depth\": %d, \"host_levels\": %d, \"words_differing\": %ld, "
           "\"root\": \"%s\", \"ok\": %s}\n",
           L.name.c_str(), (unsigned long long)n, S.a.depth, L.host, diff, hex(L.pin).c_str(), ok ? "true" : "false");
  }
  auto dumpk = [&](const std::string& name, int K, auto& k) {
    const std::string d = dir + "/" + name;
    k.salt(); k.data(S.src);
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    write_file(d + "/levels.bin", k.lv, k.nodes * 64);
    FILE* m = fopen((d + "/meta.json").c_str(), "w");
    fprintf(m, "{\"k\": %d, \"leaves\": %llu, \"leaf_bytes\": %d, \"seg_bytes\": %d, \"gpu\": \"%s\"}\n", K, (unsigned long long)n,
            LEAF, SEGB, prop.name);
    fclose(m);
  };
  dumpk("k4", 4, S.k4);
  dumpk("k8", 8, S.k8);
  fflush(stdout);
  return bad ? 1 : 0;
}

static int lat_bench(const std::string& pdir, int reps, const std::vector<u64>& sizes) {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  Nv nv;
  for (u64 n : sizes) {
    Setup S(pdir + "/params-" + std::to_string(n) + ".bin", n, false, s);
    S.build();
    auto gpu_ms = [&](Lever& L, int m) {
      spin<<<1, 1, 0, s>>>(200000);
      CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(L.g[m], s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
      float t; CK(cudaEventElapsedTime(&t, e0, e1));
      return (double)t;
    };
    for (int w = 0; w < 3; ++w)
      for (auto& L : S.levers) { S.run_host(L, 1); S.run_host(L, 0); gpu_ms(L, 0); }
    CK(cudaGetLastError());
    const int nl = (int)S.levers.size();
    for (int r = 0; r < reps; ++r) {
      for (int j = 0; j < nl; ++j) {
        Lever& L = S.levers[(j + r) % nl];
        double v[4];
        v[0] = S.run_host(L, 0); v[1] = S.run_host(L, 1); v[2] = gpu_ms(L, 0); v[3] = gpu_ms(L, 1);
        printf("{\"mode\": \"lat\", \"gpu\": \"%s\", \"sms\": %d, \"leaf_bytes\": %d, \"seg_bytes\": %d, \"lever\": \"%s\", \"rep\": %d, "
               "\"leaves\": %llu, \"bytes\": %llu, \"depth\": %d, \"host_levels\": %d, \"host_pre_ms\": %.5f, \"host_step_ms\": %.5f, "
               "\"gpu_pre_ms\": %.5f, \"gpu_step_ms\": %.5f, \"mhz\": %u}\n",
               prop.name, prop.multiProcessorCount, LEAF, SEGB, L.name.c_str(), r, (unsigned long long)n,
               (unsigned long long)(n * LEAF), L.depth, L.host, v[0], v[1], v[2], v[3], nv.sm());
      }
      fflush(stdout);
    }
  }
  return 0;
}

// --- overlap ----------------------------------------------------------------------------------------------------------
constexpr int KS = 252, SCTA = 256 * 8 * 16;   // stand-in kernels a step (36 layers x 7); bytes a CTA reads
__global__ void __launch_bounds__(256) standin(const uint4* __restrict__ p, u32* sink) {
  const uint4* q = p + (size_t)blockIdx.x * (256 * 8) + threadIdx.x;
  u32 acc = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint4 v = __ldcs(q + 256 * i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) sink[blockIdx.x & 1023] = acc;
}

static int overlap(const std::string& pdir, int reps, double target, const std::vector<u64>& sizes) {
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  int least, greatest;
  CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  cudaStream_t sa, sb[2];
  CK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithPriority(&sb[0], cudaStreamNonBlocking, least));
  CK(cudaStreamCreateWithPriority(&sb[1], cudaStreamNonBlocking, greatest));
  cudaEvent_t e0, ea, eb;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
  Nv nv;
  const size_t BUF = 4ull << 30;
  uint4* buf;
  u32* sink;
  CK(cudaMalloc(&buf, BUF));
  CK(cudaMemset(buf, 1, BUF));
  CK(cudaMalloc(&sink, 4096));
  auto ms = [&](cudaEvent_t a, cudaEvent_t z) { float t; CK(cudaEventElapsedTime(&t, a, z)); return (double)t; };
  auto standin_graph = [&](size_t per) {
    return capture(sa, [&] {
      for (int j = 0; j < KS; ++j) {
        const size_t off = ((size_t)j * per) % (BUF - per) / SCTA * SCTA;
        standin<<<(unsigned)(per / SCTA), 256, 0, sa>>>(buf + off / 16, sink);
      }
    });
  };
  auto alone = [&](cudaGraphExec_t g) {
    spin<<<1, 1, 0, sa>>>(200000);
    CK(cudaEventRecord(e0, sa)); CK(cudaGraphLaunch(g, sa)); CK(cudaEventRecord(ea, sa)); CK(cudaEventSynchronize(ea));
    return ms(e0, ea);
  };
  // calibrate the bytes a kernel reads so the step takes TARGET_MS alone
  size_t per = 64ull << 20;
  cudaGraphExec_t g = nullptr;
  double t = 0;
  for (int it = 0; it < 4; ++it) {
    if (g) CK(cudaGraphExecDestroy(g));
    g = standin_graph(per);
    alone(g);
    std::vector<double> v;
    for (int r = 0; r < 5; ++r) v.push_back(alone(g));
    std::sort(v.begin(), v.end());
    t = v[2];
    if (it < 3) per = std::max<size_t>(SCTA, (size_t)(per * target / t) / SCTA * SCTA);
  }
  printf("{\"mode\": \"standin\", \"gpu\": \"%s\", \"kernels\": %d, \"bytes_per_kernel\": %zu, \"bytes\": %zu, \"ms\": %.4f, "
         "\"gb_per_s\": %.1f, \"target_ms\": %.3f, \"prio_least\": %d, \"prio_greatest\": %d}\n",
         prop.name, KS, per, per * KS, t, per * KS / t * 1e-6, target, least, greatest);
  fflush(stdout);
  for (u64 n : sizes) {
    Setup S(pdir + "/params-" + std::to_string(n) + ".bin", n, false, sa);
    Bin b{&S.a, 256, 0};
    const u64* root = b.out_level();
    (void)root;
    auto commit = [&](cudaStream_t st, int m) {
      S.a.st = st;
      if (m) b.salt();
      b.data(S.src);
    };
    auto commit_alone = [&](cudaStream_t st, int m) {
      spin<<<1, 1, 0, st>>>(200000);
      CK(cudaEventRecord(e0, st)); commit(st, m); CK(cudaEventRecord(eb, st)); CK(cudaEventSynchronize(eb));
      return ms(e0, eb);
    };
    auto both = [&](cudaStream_t st, int m, double* tw, double* rt) {
      spin<<<1, 1, 0, sa>>>(200000);
      CK(cudaEventRecord(e0, sa));
      CK(cudaStreamWaitEvent(st, e0, 0));
      CK(cudaGraphLaunch(g, sa));
      commit(st, m);
      CK(cudaEventRecord(ea, sa)); CK(cudaEventRecord(eb, st));
      CK(cudaEventSynchronize(ea)); CK(cudaEventSynchronize(eb));
      *tw = ms(e0, ea); *rt = ms(e0, eb);
    };
    commit(sb[0], 1);
    CK(cudaDeviceSynchronize());
    for (int w = 0; w < 2; ++w) { double a_, b_; both(sb[1], 1, &a_, &b_); commit_alone(sb[0], 0); alone(g); }
    CK(cudaGetLastError());
    for (int r = 0; r < reps; ++r) {
      for (int k = 0; k < 4; ++k) {
        const int c = (k + r) % 4, m = c & 1, pr = c >> 1;
        const double t0 = alone(g), c0 = commit_alone(sb[pr], m);
        double tw, rt;
        both(sb[pr], m, &tw, &rt);
        printf("{\"mode\": \"overlap\", \"gpu\": \"%s\", \"leaf_bytes\": %d, \"leaves\": %llu, \"bytes\": %llu, \"rep\": %d, "
               "\"commit\": \"%s\", \"prio\": \"%s\", \"standin_alone_ms\": %.5f, \"commit_alone_ms\": %.5f, "
               "\"standin_with_ms\": %.5f, \"root_ms\": %.5f, \"mhz\": %u}\n",
               prop.name, LEAF, (unsigned long long)n, (unsigned long long)(n * LEAF), r, m ? "step" : "pre",
               pr ? "high" : "default", t0, c0, tw, rt, nv.sm());
      }
      fflush(stdout);
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 5 && !strcmp(argv[1], "verify")) return verify(argv[2], strtoull(argv[3], nullptr, 10), argv[4]);
  if (argc == 5 && !strcmp(argv[1], "bench")) return lat_bench(argv[2], atoi(argv[3]), parse_sizes(argv[4]));
  if (argc == 6 && !strcmp(argv[1], "overlap")) return overlap(argv[2], atoi(argv[3]), atof(argv[4]), parse_sizes(argv[5]));
  fprintf(stderr, "usage: hm96lat verify PARAMS N DIR | bench PDIR REPS N[,N] | overlap PDIR REPS TARGET_MS N[,N]\n");
  return 2;
}
