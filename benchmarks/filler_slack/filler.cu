// The receipted INT8 filler on sm_120 and its issue slack ν.
//
// The filler is A5a's best fused loop at L = 32 (`benchmarks/receipt_phi/phi.cu` on cursor/receipt-phi-ea24,
// `fused_k<DUO, 1, s8>`): each warp runs FT = 4 m16n8k32 s8 x s8 -> s32 tiles over salted register operands, and after
// every MMA round (32 k-steps) each lane absorbs its 16 accumulator words into four duo-φ chains. One kernel template
// carries the three ways of measuring what the filler leaves free:
//
//   XOP, X, UNR   X extra integer instructions of class XOP per loop trip of UNR rounds (4·UNR MMAs) inside the
//                 filler's own loop, reading no accumulator (method b: k = X / (4·UNR) per MMA)
//   POP           warps FW..FW+PW-1 of every block run a dependent-free stream of class POP, 256 a body, with an optional
//                 __nanosleep between bodies, until the block's filler warps finish (method a, co-resident probe)
//
// One block per SM (it reserves the SM's shared memory). Filler warps are 0..FW-1, so with FW and PW multiples of 4
// every SMSP holds FW/4 filler and PW/4 probe warps. Modes:
//
//   filler gate SEED KSEED           small instances of every kernel: "G" lines for gate.py to recompute
//   filler sweep ITERS REPS          every configuration, interleaved rep by rep: JSON lines, "V" lines for gate.py
//   filler one NAME FW PW SLEEP ITERS   one launch of one kernel (for ncu)
//   filler list                      the kernel names
//
// Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 filler.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(3);                                                                             \
    }                                                                                      \
  } while (0)

constexpr int NKEY = 32;
__constant__ uint32_t KA[NKEY], KK[NKEY];

__host__ __device__ __forceinline__ uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__host__ __device__ __forceinline__ uint32_t gen(uint32_t seed, uint32_t a, uint32_t b, uint32_t c) {
  return lowbias32(seed ^ lowbias32((a * 0x9E3779B1U) ^ lowbias32((b * 0x85EBCA77U) ^ lowbias32(c + 0x632BE5ABU))));
}

__device__ __forceinline__ uint32_t rotl(uint32_t x, int r) { return __funnelshift_l(x, x, r); }
__host__ __device__ constexpr int rot_of(int t) { return ((t * 7 + 5) % 31) + 1; }

struct St {
  uint32_t y, z;
};

// duo: u = y ^ p ^ k_t; y <- u·a_t + z; z <- rotl(u, r_t) + y. Receipt y ^ rotl(z, 16).
template <int T0>
__device__ __forceinline__ void duo4(St& s, const uint32_t (&p)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int t = T0 + i;
    const uint32_t u = s.y ^ p[i] ^ KK[t];
    s.y = u * KA[t] + s.z;
    s.z = rotl(u, rot_of(t)) + s.y;
  }
}
__device__ __forceinline__ uint32_t finish(const St& s) { return s.y ^ rotl(s.z, 16); }

__device__ __forceinline__ void mma_s8(uint32_t (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The extra (method b) and probe (method a) instruction classes. Each op updates register u = e mod 8 from itself and
// register e + 3 mod 8, last written five ops earlier, so a run of ops is dependency-free at the ALU's latency and no two
// ops fold into one. MIX alternates IADD3 and IMAD, one ALU-pipe and one FMA-pipe op.
enum Op { O_NONE = 0, O_IADD3, O_LOP3, O_IMAD, O_PRMT, O_MIX, NOPS };
static const char* OP_NAME[NOPS] = {"none", "iadd3", "lop3", "imad", "prmt", "mix"};

__device__ __forceinline__ uint32_t maj3(uint32_t a, uint32_t b, uint32_t c) { return (a & b) | (a & c) | (b & c); }

template <int OP>
__device__ __forceinline__ void xop(uint32_t (&y)[8], const uint32_t (&c)[4], int e) {
  const int u = e & 7, v = (e + 3) & 7;
  const int op = OP == O_MIX ? ((e & 1) ? O_IMAD : O_IADD3) : OP;
  if (op == O_IADD3) y[u] = y[u] + y[v] + c[e & 3];
  if (op == O_LOP3) y[u] = maj3(y[u], y[v], c[e & 3]);
  if (op == O_IMAD) y[u] = y[u] * c[e & 3] + y[v];
  if (op == O_PRMT) y[u] = __byte_perm(y[u], y[v], 0x5140);
}

constexpr int FT = 4, OUTW = 24, PBODY = 256, MAXW = 32;

// One loop trip: UNR rounds of FT MMAs, each followed by the four chains' absorbs, with the trip's X extras spread
// over the rounds (round q gets extras [q·X/UNR, (q+1)·X/UNR)).
template <int XOP, int X, int UNR>
__device__ __forceinline__ void trip(uint32_t (&acc)[FT][4], St (&s)[FT], const uint32_t (&a)[4],
                                     const uint32_t (&b)[2][2], uint32_t (&y)[8], const uint32_t (&c)[4]) {
#pragma unroll
  for (int q = 0; q < UNR; ++q) {
#pragma unroll
    for (int j = 0; j < FT; ++j) mma_s8(acc[j], a, b[j & 1]);
    duo4<0>(s[0], acc[0]);
    duo4<4>(s[1], acc[1]);
    duo4<8>(s[2], acc[2]);
    duo4<12>(s[3], acc[3]);
    if constexpr (X > 0) {
#pragma unroll
      for (int e = q * X / UNR; e < (q + 1) * X / UNR; ++e) xop<XOP>(y, c, e);
    }
  }
}

// out: OUTW words per filler thread (4 receipts, 16 accumulator words, the extras' fold). fclk: t0, t1 per filler warp
// slot (block · MAXW + warp). pinfo: bodies, t0, t1, fold per probe warp slot. smid: per block.
template <int POP, int XOP, int X, int UNR>
__global__ void __launch_bounds__(1024, 1)
    fk(uint32_t* out, long long* fclk, long long* pinfo, uint32_t* smid, int iters, int fw, int sleep_ns, uint32_t seed) {
  __shared__ int done;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    done = 0;
    uint32_t id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    smid[blockIdx.x] = id;
  }
  __syncthreads();
  if (warp < fw) {
    const uint32_t ftid = (blockIdx.x * fw + warp) * 32 + lane;
    uint32_t a[4], b[2][2], acc[FT][4], y[8], c[4];
    St s[FT];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = gen(seed, ftid, 7, i);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int k = 0; k < 2; ++k) b[i][k] = gen(seed, ftid, 8 + i, k);
#pragma unroll
    for (int j = 0; j < FT; ++j) {
      s[j].y = gen(seed, ftid, j, 100);
      s[j].z = gen(seed, ftid, j, 101);
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[j][r] = 0;
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) y[u] = gen(seed, ftid, 20 + u, 0);
#pragma unroll
    for (int r = 0; r < 4; ++r) c[r] = gen(seed, ftid, 30 + r, 0) | 1u;
    // Opaque to ptxas, so it keeps the operands in registers instead of recomputing gen() inside the timed loop.
#pragma unroll
    for (int i = 0; i < 4; ++i) asm volatile("" : "+r"(a[i]), "+r"(c[i]));
    asm volatile("" : "+r"(b[0][0]), "+r"(b[0][1]), "+r"(b[1][0]), "+r"(b[1][1]));
#pragma unroll
    for (int j = 0; j < FT; ++j) asm volatile("" : "+r"(s[j].y), "+r"(s[j].z));
#pragma unroll
    for (int u = 0; u < 8; ++u) asm volatile("" : "+r"(y[u]));
    __syncwarp();
    const long long t0 = clock64();
#pragma unroll 1
    for (int it = 0; it < iters; it += UNR) trip<XOP, X, UNR>(acc, s, a, b, y, c);
    const long long t1 = clock64();
    uint32_t* o = out + (size_t)ftid * OUTW;
#pragma unroll
    for (int j = 0; j < FT; ++j) {
      o[j] = finish(s[j]);
#pragma unroll
      for (int r = 0; r < 4; ++r) o[4 + 4 * j + r] = acc[j][r];
    }
    uint32_t fold = 0;
#pragma unroll
    for (int u = 0; u < 8; ++u) fold ^= y[u];
    o[20] = X > 0 ? fold : 0u;
    o[21] = o[22] = o[23] = 0;
    if (lane == 0) {
      fclk[2 * (blockIdx.x * MAXW + warp)] = t0;
      fclk[2 * (blockIdx.x * MAXW + warp) + 1] = t1;
    }
    __syncwarp();
    if (lane == 0) atomicAdd(&done, 1);
  } else if constexpr (POP != O_NONE) {
    const uint32_t ptid = 0x100000u + blockIdx.x * 1024 + threadIdx.x;
    uint32_t y[8], c[4];
#pragma unroll
    for (int u = 0; u < 8; ++u) y[u] = gen(seed, ptid, 20 + u, 0);
#pragma unroll
    for (int r = 0; r < 4; ++r) c[r] = gen(seed, ptid, 30 + r, 0) | 1u;
#pragma unroll
    for (int u = 0; u < 8; ++u) asm volatile("" : "+r"(y[u]));
#pragma unroll
    for (int r = 0; r < 4; ++r) asm volatile("" : "+r"(c[r]));
    const volatile int* vd = &done;
    long long n = 0;
    const long long t0 = clock64();
    while (*vd < fw) {
#pragma unroll
      for (int e = 0; e < PBODY; ++e) xop<POP>(y, c, e);
      ++n;
      if (sleep_ns > 0) __nanosleep(sleep_ns);
    }
    const long long t1 = clock64();
    uint32_t fold = 0;
#pragma unroll
    for (int u = 0; u < 8; ++u) fold ^= y[u];
    if (lane == 0) {
      long long* p = pinfo + 4 * (blockIdx.x * MAXW + warp);
      p[0] = n;
      p[1] = t0;
      p[2] = t1;
      p[3] = fold;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------- host
using KFn = void (*)(uint32_t*, long long*, long long*, uint32_t*, int, int, int, uint32_t);
struct Kern {
  std::string name;
  int pop, xop, x, unr;
  KFn fn;
};

#define BASE(U) {"base-u" #U, O_NONE, O_NONE, 0, U, fk<O_NONE, O_NONE, 0, U>}
#define XK(OP, X, U) {std::string("x-") + OP_NAME[OP] + "-u" #U "-x" #X, O_NONE, OP, X, U, fk<O_NONE, OP, X, U>}
#define XROW1(OP) XK(OP, 1, 1), XK(OP, 2, 1), XK(OP, 3, 1), XK(OP, 4, 1), XK(OP, 6, 1), XK(OP, 8, 1), XK(OP, 12, 1), XK(OP, 16, 1)
#define XROW4(OP) XK(OP, 1, 4), XK(OP, 2, 4), XK(OP, 4, 4)
#define PK(OP) {std::string("p-") + OP_NAME[OP], OP, O_NONE, 0, 1, fk<OP, O_NONE, 0, 1>}

static std::vector<Kern> kernels() {
  return {BASE(1),      BASE(4),      XROW1(O_IADD3), XROW1(O_LOP3), XROW1(O_IMAD), XROW1(O_PRMT),
          XROW1(O_MIX), XROW4(O_IADD3), XROW4(O_IMAD), XROW4(O_MIX), PK(O_IADD3),   PK(O_LOP3),
          PK(O_IMAD),   PK(O_PRMT),   PK(O_MIX)};
}

static void set_keys(uint32_t kseed) {
  uint32_t ka[NKEY], kk[NKEY];
  for (int i = 0; i < NKEY; ++i) {
    ka[i] = gen(kseed, 1, i, 0) | 1u;
    kk[i] = gen(kseed, 3, i, 0);
  }
  CK(cudaMemcpyToSymbol(KA, ka, sizeof ka));
  CK(cudaMemcpyToSymbol(KK, kk, sizeof kk));
}

struct Dev {
  int sms = 0, smem = 0;
  uint32_t *out = nullptr, *smid = nullptr;
  long long *fclk = nullptr, *pinfo = nullptr;
  cudaEvent_t e0, e1;
  size_t out_words = 0;
};

static Dev make_dev() {
  Dev d;
  int dev = 0, per_smp = 0, reserved = 0, optin = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaDeviceGetAttribute(&per_smp, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  CK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev));
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  d.smem = std::min(per_smp - reserved, optin) - 256;  // the static `done` word too
  d.out_words = (size_t)d.sms * MAXW * 32 * OUTW;
  CK(cudaMalloc(&d.out, d.out_words * sizeof(uint32_t)));
  CK(cudaMalloc(&d.fclk, (size_t)d.sms * MAXW * 2 * sizeof(long long)));
  CK(cudaMalloc(&d.pinfo, (size_t)d.sms * MAXW * 4 * sizeof(long long)));
  CK(cudaMalloc(&d.smid, (size_t)d.sms * sizeof(uint32_t)));
  CK(cudaEventCreate(&d.e0));
  CK(cudaEventCreate(&d.e1));
  for (const Kern& k : kernels()) {
    CK(cudaFuncSetAttribute(k.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, d.smem));
    CK(cudaFuncSetAttribute(k.fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  }
  return d;
}

struct Result {
  double ms = 0, mhz = 0, cyc_med = 0, rate_med = 0, rate_min = 0, probe_med = 0, probe_min = 0, sleep_frac = 0;
  int distinct = 0, blocks = 0;
};

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

// One launch over `blocks` blocks of FW + PW warps. rate: filler warp-MMAs per SM-clock over the block's filler span
// (first t0 to last t1 of its filler warps); probe: probe warp-instructions (bodies · 256) per SM-clock over the same span.
static Result launch(Dev& d, const Kern& k, int blocks, int fw, int pw, int sleep_ns, int iters, uint32_t seed) {
  const int threads = 32 * (fw + pw);
  CK(cudaMemset(d.out, 0xA5, d.out_words * sizeof(uint32_t)));
  CK(cudaMemset(d.fclk, 0, (size_t)d.sms * MAXW * 2 * sizeof(long long)));
  CK(cudaMemset(d.pinfo, 0, (size_t)d.sms * MAXW * 4 * sizeof(long long)));
  CK(cudaEventRecord(d.e0));
  k.fn<<<blocks, threads, d.smem>>>(d.out, d.fclk, d.pinfo, d.smid, iters, fw, sleep_ns, seed);
  CK(cudaEventRecord(d.e1));
  CK(cudaEventSynchronize(d.e1));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, d.e0, d.e1));
  std::vector<long long> fc((size_t)blocks * MAXW * 2), pi((size_t)blocks * MAXW * 4);
  std::vector<uint32_t> sm(blocks);
  CK(cudaMemcpy(fc.data(), d.fclk, fc.size() * sizeof(long long), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(pi.data(), d.pinfo, pi.size() * sizeof(long long), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sm.data(), d.smid, sm.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  Result r;
  r.ms = ms;
  r.blocks = blocks;
  std::vector<double> rate, probe, span;
  for (int bl = 0; bl < blocks; ++bl) {
    long long lo = 0, hi = 0;
    for (int w = 0; w < fw; ++w) {
      const long long t0 = fc[2 * (bl * MAXW + w)], t1 = fc[2 * (bl * MAXW + w) + 1];
      lo = w == 0 ? t0 : std::min(lo, t0);
      hi = w == 0 ? t1 : std::max(hi, t1);
    }
    const double sp = (double)(hi - lo);
    span.push_back(sp);
    rate.push_back((double)fw * iters * FT / sp);
    double ops = 0;
    for (int w = fw; w < fw + pw; ++w) ops += (double)pi[4 * (bl * MAXW + w)] * PBODY;
    probe.push_back(ops / sp);
  }
  std::vector<uint32_t> u(sm);
  std::sort(u.begin(), u.end());
  r.distinct = (int)(std::unique(u.begin(), u.end()) - u.begin());
  r.cyc_med = median(span);
  r.mhz = *std::max_element(span.begin(), span.end()) / (ms * 1e3);
  r.rate_med = median(rate);
  r.rate_min = *std::min_element(rate.begin(), rate.end());
  r.probe_med = median(probe);
  r.probe_min = *std::min_element(probe.begin(), probe.end());
  return r;
}

static const Kern& find(const std::vector<Kern>& ks, const std::string& name) {
  for (const Kern& k : ks)
    if (k.name == name) return k;
  fprintf(stderr, "no kernel %s\n", name.c_str());
  exit(2);
}

static void emit(const char* phase, const Kern& k, int fw, int pw, int sleep_ns, int iters, int rep, uint32_t seed,
                 const Result& r) {
  printf("{\"phase\":\"%s\",\"kernel\":\"%s\",\"pop\":\"%s\",\"xop\":\"%s\",\"x\":%d,\"unr\":%d,\"k\":%.4f,\"fw\":%d,"
         "\"pw\":%d,\"sleep_ns\":%d,\"iters\":%d,\"rep\":%d,\"seed\":%u,\"blocks\":%d,\"distinct_sms\":%d,\"ms\":%.5f,"
         "\"cycles_med\":%.0f,\"mhz\":%.2f,\"mma_per_sm_clk_med\":%.7f,\"mma_per_sm_clk_min\":%.7f,"
         "\"probe_inst_per_sm_clk_med\":%.6f,\"probe_inst_per_sm_clk_min\":%.6f}\n",
         phase, k.name.c_str(), OP_NAME[k.pop], OP_NAME[k.xop], k.x, k.unr, (double)k.x / (FT * k.unr), fw, pw,
         sleep_ns, iters, rep, seed, r.blocks, r.distinct, r.ms, r.cyc_med, r.mhz, r.rate_med, r.rate_min, r.probe_med,
         r.probe_min);
  fflush(stdout);
}

// "V KERNEL SEED KSEED ITERS FTID w0..w19": sampled filler threads of a timed launch, for gate.py.
static void emit_v(const Dev& d, const Kern& k, uint32_t seed, uint32_t kseed, int iters, FILE* vf) {
  static const uint32_t tids[3] = {0, 77, 5003};
  for (uint32_t t : tids) {
    uint32_t w[OUTW];
    CK(cudaMemcpy(w, d.out + (size_t)t * OUTW, sizeof w, cudaMemcpyDeviceToHost));
    fprintf(vf, "V %s %u %u %d %u", k.name.c_str(), seed, kseed, iters, t);
    for (int i = 0; i < 20; ++i) fprintf(vf, " %08x", w[i]);
    fprintf(vf, "\n");
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: filler gate|sweep|one|list ...\n");
    return 2;
  }
  const std::string mode = argv[1];
  const std::vector<Kern> ks = kernels();
  if (mode == "list") {
    for (const Kern& k : ks) printf("%s\n", k.name.c_str());
    return 0;
  }
  const uint32_t kseed0 = 0xC0FFEE11u;
  Dev d = make_dev();
  if (mode == "gate") {
    const uint32_t seed = argc > 2 ? (uint32_t)strtoul(argv[2], 0, 0) : 0x5EED1234u;
    const uint32_t kseed = argc > 3 ? (uint32_t)strtoul(argv[3], 0, 0) : kseed0;
    set_keys(kseed);
    const int fw = 4, pw = 4, blocks = 2;
    std::vector<uint32_t> h((size_t)blocks * fw * 32 * OUTW);
    for (const Kern& k : ks)
      for (int iters : {4, 12}) {
        launch(d, k, blocks, fw, k.pop ? pw : 0, 0, iters, seed);
        CK(cudaMemcpy(h.data(), d.out, h.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int t = 0; t < blocks * fw * 32; ++t) {
          printf("G %s %u %u %d %d", k.name.c_str(), seed, kseed, iters, t);
          for (int i = 0; i < 20; ++i) printf(" %08x", h[(size_t)t * OUTW + i]);
          printf("\n");
        }
      }
    return 0;
  }
  if (mode == "one") {
    if (argc < 7) return 2;
    set_keys(kseed0);
    const Kern& k = find(ks, argv[2]);
    launch(d, k, d.sms, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), 0x5EED1234u);
    return 0;
  }
  if (mode != "sweep") return 2;
  const int iters = (argc > 2 ? atoi(argv[2]) : 32000) & ~3, reps = argc > 3 ? atoi(argv[3]) : 7;
  FILE* vf = fopen(argc > 4 ? argv[4] : "verify.txt", "w");
  set_keys(kseed0);
  const Kern& b1 = find(ks, "base-u1");
  const Kern& b4 = find(ks, "base-u4");
  // Warm up.
  launch(d, b1, d.sms, 16, 0, 0, iters, 1);
  for (int rep = 0; rep < reps; ++rep) {
    const uint32_t seed = 0x5EED0000u + rep;
    // Occupancy: the filler alone at 1..8 warps per SMSP.
    for (int fw : {4, 8, 12, 16, 20, 24, 28, 32}) {
      Result r = launch(d, b1, d.sms, fw, 0, 0, iters, seed);
      emit("occupancy", b1, fw, 0, 0, iters, rep, seed, r);
      if (fw == 16) emit_v(d, b1, seed, kseed0, iters, vf);
    }
    for (int fw : {8, 16}) emit("occupancy", b4, fw, 0, 0, iters, rep, seed, launch(d, b4, d.sms, fw, 0, 0, iters, seed));
    // Method b: extras in the filler's loop, each beside its own base, at 2 and 4 filler warps per SMSP.
    for (int fw : {8, 16})
      for (const Kern& k : ks) {
        if (k.xop == O_NONE) continue;
        const Kern& b = k.unr == 1 ? b1 : b4;
        emit("inloop-base", b, fw, 0, 0, iters, rep, seed, launch(d, b, d.sms, fw, 0, 0, iters, seed));
        Result r = launch(d, k, d.sms, fw, 0, 0, iters, seed);
        emit("inloop", k, fw, 0, 0, iters, rep, seed, r);
        if (fw == 16) emit_v(d, k, seed, kseed0, iters, vf);
      }
    // Method a: probe warps beside the filler, each beside a filler-alone base.
    for (int fw : {8, 16})
      for (const Kern& k : ks) {
        if (k.pop == O_NONE) continue;
        for (int pw : {4, 8, 16}) {
          if (fw + pw > MAXW) continue;
          for (int sl : {0, 100, 400, 1600, 6400, 25600, 102400}) {
            emit("probe-base", b1, fw, 0, 0, iters, rep, seed, launch(d, b1, d.sms, fw, 0, 0, iters, seed));
            Result r = launch(d, k, d.sms, fw, pw, sl, iters, seed);
            emit("probe", k, fw, pw, sl, iters, rep, seed, r);
            if (fw == 16 && pw == 4 && sl == 0) emit_v(d, k, seed, kseed0, iters, vf);
          }
        }
      }
  }
  fclose(vf);
  return 0;
}
