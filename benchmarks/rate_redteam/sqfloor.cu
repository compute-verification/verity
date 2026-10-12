// Squarings mod p = 2^768 - 825 on the CUDA cores (sm_120), a thread a word: the codec lanes' column form beside a row
// form and a Karatsuba form, to bound how far their squaring sits above a tuned one (the rebuild's x is the
// adversary's squaring rate).
//
// Form 0 is ra-word-codec's `kv.cu` at KV_FIELD = 768 (`cursor/ra-word-codec-10bf`, acd61664f): column by column, each
// unordered pair once, a three-word accumulator (three instructions a product), then two folds of 2^768 = 825.
// Form 1 is the row form: each row's low halves in one carry chain and its high halves in a second, so a product costs
// two instructions and no third accumulator word; the doubling and the squares are one chain each; the fold is two
// chains and a single multiply-add (a fold that carries out leaves the value below 2^20, so the second never does).
// Form 2 is one level of Karatsuba over form 0's columns: x = x1 2^384 + x0, x^2 = C 2^768 + (A + C - D) 2^384 + A with
// A = x0^2, C = x1^2, D = (x0 - x1)^2, three 12-limb squares (234 products in place of 300), then form 0's fold.
// All keep x lazily below 2^768; the host reduces below p.
//
//   sqfloor gate FORM K N          N words from a fixed seed, K squarings: "x0 xK" in hex, canonical
//   sqfloor time FORM K TPB WAVES REPS   timed launches at the occupancy's blocks a SM times WAVES
//   sqfloor clock                  the SM clock off a timed clock64 spin
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sq {

using u32 = uint32_t;
using u64 = uint64_t;
constexpr int NL = 24;
constexpr u32 CF = 825;

__device__ __forceinline__ void mac(u32 *r, u32 a, u32 b) {
  asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(a), "r"(b));
}
__device__ __forceinline__ void add3(u32 *r, u32 a, u32 b) {
  asm("add.cc.u32 %0, %0, %3;\n\taddc.cc.u32 %1, %1, %4;\n\taddc.u32 %2, %2, 0;" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2])
      : "r"(a), "r"(b));
}

template <int N>
__device__ __forceinline__ void sqr_cols_n(const u32 (&a)[N], u32 (&t)[2 * N]) {
  u32 c0 = 0, c1 = 0;
#pragma unroll
  for (int m = 0; m < 2 * N - 1; ++m) {
    u32 r[3] = {0, 0, 0};
#pragma unroll
    for (int i = (m < N ? 0 : m - N + 1); 2 * i < m; ++i) mac(r, a[i], a[m - i]);
    r[2] = (r[2] << 1) | (r[1] >> 31);
    r[1] = (r[1] << 1) | (r[0] >> 31);
    r[0] <<= 1;
    if ((m & 1) == 0) mac(r, a[m / 2], a[m / 2]);
    add3(r, c0, c1);
    t[m] = r[0];
    c0 = r[1];
    c1 = r[2];
  }
  t[2 * N - 1] = c0;
}

__device__ __forceinline__ void sqr_cols(const u32 (&a)[NL], u32 (&t)[2 * NL]) { sqr_cols_n<NL>(a, t); }

__device__ __forceinline__ void fold_cols(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  u64 v = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    v += (u64)t[NL + k] * CF + t[k];
    x[k] = (u32)v;
    v >>= 32;
  }
  v *= CF;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    v += x[k];
    x[k] = (u32)v;
    v >>= 32;
  }
  if (v) {
    u64 z = CF;
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      z += x[k];
      x[k] = (u32)z;
      z >>= 32;
    }
  }
}

#define LO_CC(d, a, b) asm volatile("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define LOC_CC(d, a, b) asm volatile("madc.lo.cc.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define HI_CC(d, a, b) asm volatile("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define HIC_CC(d, a, b) asm volatile("madc.hi.cc.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define HIC(d, a, b) asm volatile("madc.hi.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define CARRY(d) asm volatile("addc.u32 %0, 0, 0;" : "=r"(d))
#define ADDC0(d) asm volatile("addc.u32 %0, %0, 0;" : "+r"(d))
#define ADDCC0(d) asm volatile("addc.cc.u32 %0, %0, 0;" : "+r"(d))
#define DBL(d) asm volatile("add.cc.u32 %0, %0, %0;" : "+r"(d))
#define DBLC(d) asm volatile("addc.cc.u32 %0, %0, %0;" : "+r"(d))
#define DBLC_END(d) asm volatile("addc.u32 %0, %0, %0;" : "+r"(d))

// Row i adds a_i (a_{i+1} … a_{NL-1}) at limb 2i + 1: the low halves as one chain ending in t[i + NL] (which holds at
// most the previous row's carry, so it can't overflow), then the high halves one limb up, whose carry is t[i + NL + 1]
// (no earlier row reaches it).
__device__ __forceinline__ void sqr_rows(const u32 (&a)[NL], u32 (&t)[2 * NL]) {
#pragma unroll
  for (int k = 0; k < 2 * NL; ++k) t[k] = 0;
#pragma unroll
  for (int i = 0; i < NL - 1; ++i) {
    LO_CC(t[2 * i + 1], a[i], a[i + 1]);
#pragma unroll
    for (int j = i + 2; j < NL; ++j) LOC_CC(t[i + j], a[i], a[j]);
    ADDC0(t[i + NL]);
    HI_CC(t[2 * i + 2], a[i], a[i + 1]);
#pragma unroll
    for (int j = i + 2; j < NL; ++j) HIC_CC(t[i + j + 1], a[i], a[j]);
    CARRY(t[i + NL + 1]);
  }
  DBL(t[1]);
#pragma unroll
  for (int k = 2; k < 2 * NL - 1; ++k) DBLC(t[k]);
  DBLC_END(t[2 * NL - 1]);
  LO_CC(t[0], a[0], a[0]);
  HIC_CC(t[1], a[0], a[0]);
#pragma unroll
  for (int i = 1; i < NL - 1; ++i) {
    LOC_CC(t[2 * i], a[i], a[i]);
    HIC_CC(t[2 * i + 1], a[i], a[i]);
  }
  LOC_CC(t[2 * NL - 2], a[NL - 1], a[NL - 1]);
  HIC(t[2 * NL - 1], a[NL - 1], a[NL - 1]);
}

__device__ __forceinline__ void fold_rows(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  const u32 c = CF;
  u32 e;
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = t[k];
  LO_CC(x[0], t[NL], c);
#pragma unroll
  for (int k = 1; k < NL; ++k) LOC_CC(x[k], t[NL + k], c);
  CARRY(e);
  HI_CC(x[1], t[NL], c);
#pragma unroll
  for (int k = 1; k < NL - 1; ++k) HIC_CC(x[k + 1], t[NL + k], c);
  HIC(e, t[2 * NL - 1], c);
  LO_CC(x[0], e, c);
#pragma unroll
  for (int k = 1; k < NL; ++k) ADDCC0(x[k]);
  CARRY(e);
  asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(x[0]) : "r"(e), "r"(c));
}

#define ADD_CC(d, a, b) asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b))
#define ADDC_CC(d, a, b) asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b))
#define SUB_CC(d, a, b) asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b))
#define SUBC_CC(d, a, b) asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b))
#define BORROW_MASK(d) asm volatile("subc.u32 %0, 0, 0;" : "=r"(d))

// M = A + C - D = 2 x0 x1 is below 2^769: 2H limbs and a top bit m.
__device__ __forceinline__ void sqr_kara(const u32 (&a)[NL], u32 (&t)[2 * NL]) {
  constexpr int H = NL / 2;
  u32 x0[H], x1[H], d[H], e[H], mask;
#pragma unroll
  for (int k = 0; k < H; ++k) {
    x0[k] = a[k];
    x1[k] = a[H + k];
  }
  SUB_CC(d[0], x0[0], x1[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) SUBC_CC(d[k], x0[k], x1[k]);
  BORROW_MASK(mask);
  SUB_CC(e[0], x1[0], x0[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) SUBC_CC(e[k], x1[k], x0[k]);
#pragma unroll
  for (int k = 0; k < H; ++k) d[k] = (d[k] & ~mask) | (e[k] & mask);
  u32 A[2 * H], C[2 * H], D[2 * H], M[2 * H], m;
  sqr_cols_n<H>(x0, A);
  sqr_cols_n<H>(x1, C);
  sqr_cols_n<H>(d, D);
  ADD_CC(M[0], A[0], C[0]);
#pragma unroll
  for (int k = 1; k < 2 * H; ++k) ADDC_CC(M[k], A[k], C[k]);
  CARRY(m);
  SUB_CC(M[0], M[0], D[0]);
#pragma unroll
  for (int k = 1; k < 2 * H; ++k) SUBC_CC(M[k], M[k], D[k]);
  asm volatile("subc.u32 %0, %0, 0;" : "+r"(m));
#pragma unroll
  for (int k = 0; k < 2 * H; ++k) {
    t[k] = A[k];
    t[2 * H + k] = C[k];
  }
  ADD_CC(t[H], t[H], M[0]);
#pragma unroll
  for (int k = 1; k < 2 * H; ++k) ADDC_CC(t[H + k], t[H + k], M[k]);
  ADDC_CC(t[3 * H], t[3 * H], m);
#pragma unroll
  for (int k = 3 * H + 1; k < 4 * H - 1; ++k) ADDCC0(t[k]);
  ADDC0(t[4 * H - 1]);
}

template <int FORM>
__global__ void __launch_bounds__(128) chain(u32 *xs, int n, int K) {
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= n) return;
  u32 x[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = xs[(size_t)k * n + w];
#pragma unroll 1
  for (int r = 0; r < K; ++r) {
    u32 t[2 * NL];
    if (FORM == 0) {
      sqr_cols(x, t);
      fold_cols(t, x);
    } else if (FORM == 1) {
      sqr_rows(x, t);
      fold_rows(t, x);
    } else {
      sqr_kara(x, t);
      fold_cols(t, x);
    }
  }
#pragma unroll
  for (int k = 0; k < NL; ++k) xs[(size_t)k * n + w] = x[k];
}

__global__ void spin(long long cycles, long long *out) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
  *out = clock64() - t0;
}

}  // namespace sq

static uint64_t splitmix(uint64_t &s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(3);                                                                        \
    }                                                                                 \
  } while (0)

static void fill(std::vector<uint32_t> &h, int n) {
  uint64_t s = 0x5eed768ull;
  h.assign((size_t)sq::NL * n, 0);
  for (int w = 0; w < n; ++w)
    for (int k = 0; k < sq::NL; ++k) h[(size_t)k * n + w] = (uint32_t)splitmix(s);
}

using Kernel = void (*)(uint32_t *, int, int);
static Kernel kernel(int form) { return form == 0 ? sq::chain<0> : form == 1 ? sq::chain<1> : sq::chain<2>; }

static void launch(int form, uint32_t *d, int n, int K, int tpb) {
  kernel(form)<<<(n + tpb - 1) / tpb, tpb>>>(d, n, K);
  CK(cudaGetLastError());
}

// x mod p for x < 2^768, as hex (most significant limb first).
static void print_canon(const std::vector<uint32_t> &h, int n, int w) {
  uint32_t x[sq::NL], z[sq::NL];
  for (int k = 0; k < sq::NL; ++k) x[k] = h[(size_t)k * n + w];
  uint64_t v = sq::CF;
  for (int k = 0; k < sq::NL; ++k) {
    v += x[k];
    z[k] = (uint32_t)v;
    v >>= 32;
  }
  const uint32_t *y = v ? z : x;
  for (int k = sq::NL - 1; k >= 0; --k) printf("%08x", y[k]);
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  if (!strcmp(argv[1], "clock")) {
    long long *d;
    CK(cudaMalloc(&d, sizeof(long long)));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    sq::spin<<<1, 1>>>(1LL << 26, d);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    sq::spin<<<1, 1>>>(1LL << 30, d);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    long long cyc;
    CK(cudaMemcpy(&cyc, d, sizeof cyc, cudaMemcpyDeviceToHost));
    printf("{\"clock_mhz\": %.1f}\n", cyc / (ms * 1e3));
    return 0;
  }
  if (!strcmp(argv[1], "gate") && argc == 5) {
    const int form = atoi(argv[2]), K = atoi(argv[3]), n = atoi(argv[4]);
    std::vector<uint32_t> h0, h;
    fill(h0, n);
    uint32_t *d;
    CK(cudaMalloc(&d, h0.size() * 4));
    CK(cudaMemcpy(d, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice));
    launch(form, d, n, K, 128);
    CK(cudaDeviceSynchronize());
    h.resize(h0.size());
    CK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
    for (int w = 0; w < n; ++w) {
      printf("%d %d ", form, K);
      print_canon(h0, n, w);
      printf(" ");
      print_canon(h, n, w);
      printf("\n");
    }
    return 0;
  }
  if (!strcmp(argv[1], "time") && argc == 7) {
    const int form = atoi(argv[2]), K = atoi(argv[3]), tpb = atoi(argv[4]), waves = atoi(argv[5]),
              reps = atoi(argv[6]);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel(form), tpb, 0));
    cudaFuncAttributes attr;
    CK(cudaFuncGetAttributes(&attr, kernel(form)));
    const int n = prop.multiProcessorCount * per_sm * tpb * waves;
    std::vector<uint32_t> h;
    fill(h, n);
    uint32_t *d;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    launch(form, d, n, K / 10 + 1, tpb);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> rate;
    for (int r = 0; r < reps; ++r) {
      CK(cudaEventRecord(a));
      launch(form, d, n, K, tpb);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms;
      CK(cudaEventElapsedTime(&ms, a, b));
      rate.push_back((double)n * K / (ms * 1e-3));
    }
    std::sort(rate.begin(), rate.end());
    printf("{\"form\": %d, \"K\": %d, \"tpb\": %d, \"waves\": %d, \"blocks_per_sm\": %d, \"regs\": %d, \"words\": %d, "
           "\"sms\": %d, \"sq_per_s_max\": %.4e, \"sq_per_s_median\": %.4e}\n",
           form, K, tpb, waves, per_sm, attr.numRegs, n, prop.multiProcessorCount, rate.back(), rate[rate.size() / 2]);
    return 0;
  }
  return 2;
}
