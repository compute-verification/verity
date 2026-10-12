// Squarings mod p = 2^512 - 569 on the CUDA cores (sm_120), a thread holding ILP words, for the adversary's floor on
// Sloth's encode (a root is (p+1)/4, about 510 squarings). Every form squares column by column (ra-word-codec's and
// ra-codec-tensor's `sqr_full`, the red team's form 0: each unordered pair once, doubled, a three-word accumulator) and
// differs in the fold 2^512 = 569:
//   FOLD 0  the red team's `fold_cols` (C u64 carries), ptxas's MOV / IADD.64 churn included: the baseline.
//   FOLD 1  `kv.cu`'s PTX carry chains: the low products' chain, then the high ones' a limb up.
//   FOLD 2  FOLD 1 with each multiply by 569 = 2^9 + 2^5 + 2^4 + 2^3 + 1 as shifts and adds on the integer ALU, so the
//           FMA-heavy pipe (the binding one, 84% busy at 512 bits) runs only the square's 136 IMAD.WIDE.
// ILP interleaves that many independent words a thread, to fill the pipe past the carry chains' latency.
// All keep x lazily below 2^512; the host reduces below p.
//
//   sq gate FOLD ILP K N           N words from a fixed seed, K squarings: "x0 xK" in hex, canonical
//   sq time FOLD ILP K TPB WAVES REPS   timed launches at the occupancy's blocks a SM times WAVES
//   sq clock                       the SM clock off a timed clock64 spin
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
constexpr int NL = 16;
constexpr u32 CF = 569;

__device__ __forceinline__ void mac(u32 *r, u32 a, u32 b) {
  asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(a), "r"(b));
}
__device__ __forceinline__ void add3(u32 *r, u32 a, u32 b) {
  asm("add.cc.u32 %0, %0, %3;\n\taddc.cc.u32 %1, %1, %4;\n\taddc.u32 %2, %2, 0;" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2])
      : "r"(a), "r"(b));
}

__device__ __forceinline__ void sqr_cols(const u32 (&a)[NL], u32 (&t)[2 * NL]) {
  u32 c0 = 0, c1 = 0;
#pragma unroll
  for (int m = 0; m < 2 * NL - 1; ++m) {
    u32 r[3] = {0, 0, 0};
#pragma unroll
    for (int i = (m < NL ? 0 : m - NL + 1); 2 * i < m; ++i) mac(r, a[i], a[m - i]);
    r[2] = (r[2] << 1) | (r[1] >> 31);
    r[1] = (r[1] << 1) | (r[0] >> 31);
    r[0] <<= 1;
    if ((m & 1) == 0) mac(r, a[m / 2], a[m / 2]);
    add3(r, c0, c1);
    t[m] = r[0];
    c0 = r[1];
    c1 = r[2];
  }
  t[2 * NL - 1] = c0;
}

// FOLD 0: the red team's sqfloor.cu `fold_cols`, verbatim.
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

// One PTX instruction each, so a carry chain is a run of them; volatile keeps anything that would clobber the carry
// flag out from between the links (`kv.cu`'s).
__device__ __forceinline__ u32 add_cc(u32 a, u32 b) { u32 r; asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 addc_cc(u32 a, u32 b) { u32 r; asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 addc(u32 a, u32 b) { u32 r; asm volatile("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 madlo_cc(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ u32 madclo_cc(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ u32 madhi_cc(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ u32 madchi_cc(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ u32 madchi(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}

// x += a (a < 2^32) into NL limbs; returns the carry out of 2^512. The carry leaves limb 0 with probability a / 2^32
// (every caller's a is below c^2 + c < 2^19), so the rest of the chain is a branch.
__device__ __forceinline__ u32 add_small(u32 (&x)[NL], u32 a) {
  x[0] += a;
  if (x[0] >= a) return 0u;
  x[1] = add_cc(x[1], 1u);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = addc_cc(x[k], 0u);
  return addc(0u, 0u);
}

// FOLD 1: `kv.cu`'s fold. x = lo + c hi: the low products' chain, then the high ones' a limb up, whose carry word
// h <= c; then h c (< 2^20) into the low limbs, and c once more if that carried out. x < 2^512.
__device__ __forceinline__ void fold_ptx(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  x[0] = madlo_cc(t[NL], CF, t[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = madclo_cc(t[NL + k], CF, t[k]);
  u32 h = addc(0u, 0u);
  x[1] = madhi_cc(t[NL], CF, x[1]);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = madchi_cc(t[NL + k - 1], CF, x[k]);
  h = madchi(t[2 * NL - 1], CF, h);
  if (add_small(x, h * CF)) add_small(x, CF);
}

// v 569 = v 2^9 + v 2^5 + v 2^4 + v 2^3 + v as (lo, hi), hi < 2^10, in shifts and adds only (no IMAD). The 64-bit
// sum of five shifted copies of v; volatile keeps ptxas from folding it back into a multiply.
__device__ __forceinline__ void mul569(u32 v, u32 &lo, u32 &hi) {
  u32 a, b;
  asm volatile(
      "{\n\t.reg .u32 s9l, s9h, s5l, s5h, s4l, s4h, s3l, s3h;\n\t"
      "shl.b32 s9l, %2, 9;\n\tshr.b32 s9h, %2, 23;\n\t"
      "shl.b32 s5l, %2, 5;\n\tshr.b32 s5h, %2, 27;\n\t"
      "shl.b32 s4l, %2, 4;\n\tshr.b32 s4h, %2, 28;\n\t"
      "shl.b32 s3l, %2, 3;\n\tshr.b32 s3h, %2, 29;\n\t"
      "add.cc.u32 %0, s9l, s5l;\n\taddc.u32 %1, s9h, s5h;\n\t"
      "add.cc.u32 %0, %0, s4l;\n\taddc.u32 %1, %1, s4h;\n\t"
      "add.cc.u32 %0, %0, s3l;\n\taddc.u32 %1, %1, s3h;\n\t"
      "add.cc.u32 %0, %0, %2;\n\taddc.u32 %1, %1, 0;\n\t}"
      : "=r"(a), "=r"(b) : "r"(v));
  lo = a;
  hi = b;
}

// FOLD 2: FOLD 1's arithmetic with the multiplies by c as shifts: x = lo + sum_k (P_k lo) 2^(32k) + (P_k hi) 2^(32(k+1)),
// P_k = t[NL + k] c. The low words' chain, then the high words' a limb up, then h c and c as in FOLD 1.
__device__ __forceinline__ void fold_shift(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  u32 pl[NL], ph[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) mul569(t[NL + k], pl[k], ph[k]);
  x[0] = add_cc(t[0], pl[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = addc_cc(t[k], pl[k]);
  u32 h = addc(0u, 0u);
  x[1] = add_cc(x[1], ph[0]);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = addc_cc(x[k], ph[k - 1]);
  h = addc(h, ph[NL - 1]);
  u32 hl, hh;
  mul569(h, hl, hh);
  if (add_small(x, hl)) add_small(x, CF);
}

template <int FOLD>
__device__ __forceinline__ void sqr(const u32 (&a)[NL], u32 (&x)[NL]) {
  u32 t[2 * NL];
  sqr_cols(a, t);
  if (FOLD == 0) fold_cols(t, x);
  else if (FOLD == 1) fold_ptx(t, x);
  else fold_shift(t, x);
}

template <int FOLD, int ILP>
__global__ void __launch_bounds__(128) chain(u32 *xs, int n, int K) {
  const int w0 = (blockIdx.x * blockDim.x + threadIdx.x) * ILP;
  if (w0 >= n) return;
  u32 x[ILP][NL];
#pragma unroll
  for (int j = 0; j < ILP; ++j)
#pragma unroll
    for (int k = 0; k < NL; ++k) x[j][k] = xs[(size_t)k * n + w0 + j];
#pragma unroll 1
  for (int r = 0; r < K; ++r) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      u32 y[NL];
      sqr<FOLD>(x[j], y);
#pragma unroll
      for (int k = 0; k < NL; ++k) x[j][k] = y[k];
    }
  }
#pragma unroll
  for (int j = 0; j < ILP; ++j)
#pragma unroll
    for (int k = 0; k < NL; ++k) xs[(size_t)k * n + w0 + j] = x[j][k];
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
  uint64_t s = 0x5eed512ull;
  h.assign((size_t)sq::NL * n, 0);
  for (int w = 0; w < n; ++w)
    for (int k = 0; k < sq::NL; ++k) h[(size_t)k * n + w] = (uint32_t)splitmix(s);
}

using Kernel = void (*)(uint32_t *, int, int);
static Kernel kernel(int fold, int ilp) {
#define K_(F, I) if (fold == F && ilp == I) return sq::chain<F, I>;
  K_(0, 1) K_(0, 2) K_(1, 1) K_(1, 2) K_(1, 4) K_(2, 1) K_(2, 2) K_(2, 4)
#undef K_
  fprintf(stderr, "no kernel for fold %d ilp %d\n", fold, ilp);
  exit(2);
}

static void launch(int fold, int ilp, uint32_t *d, int n, int K, int tpb) {
  const int threads = n / ilp;
  kernel(fold, ilp)<<<(threads + tpb - 1) / tpb, tpb>>>(d, n, K);
  CK(cudaGetLastError());
}

// x mod p for x < 2^512, as hex (most significant limb first).
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

#ifndef SQ_NO_MAIN
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
  if (!strcmp(argv[1], "gate") && argc == 6) {
    const int fold = atoi(argv[2]), ilp = atoi(argv[3]), K = atoi(argv[4]), n = atoi(argv[5]);
    std::vector<uint32_t> h0, h;
    fill(h0, n);
    uint32_t *d;
    CK(cudaMalloc(&d, h0.size() * 4));
    CK(cudaMemcpy(d, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice));
    launch(fold, ilp, d, n, K, 128);
    CK(cudaDeviceSynchronize());
    h.resize(h0.size());
    CK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
    for (int w = 0; w < n; ++w) {
      printf("%d %d 512 %u ", fold * 10 + ilp, K, sq::CF);
      print_canon(h0, n, w);
      printf(" ");
      print_canon(h, n, w);
      printf("\n");
    }
    return 0;
  }
  if (!strcmp(argv[1], "time") && argc == 8) {
    const int fold = atoi(argv[2]), ilp = atoi(argv[3]), K = atoi(argv[4]), tpb = atoi(argv[5]),
              waves = atoi(argv[6]), reps = atoi(argv[7]);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel(fold, ilp), tpb, 0));
    cudaFuncAttributes attr;
    CK(cudaFuncGetAttributes(&attr, kernel(fold, ilp)));
    const int n = prop.multiProcessorCount * per_sm * tpb * waves * ilp;
    std::vector<uint32_t> h;
    fill(h, n);
    uint32_t *d;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    launch(fold, ilp, d, n, K / 10 + 1, tpb);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> rate;
    for (int r = 0; r < reps; ++r) {
      CK(cudaEventRecord(a));
      launch(fold, ilp, d, n, K, tpb);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms;
      CK(cudaEventElapsedTime(&ms, a, b));
      rate.push_back((double)n * K / (ms * 1e-3));
    }
    std::sort(rate.begin(), rate.end());
    printf("{\"bits\": 512, \"fold\": %d, \"ilp\": %d, \"K\": %d, \"tpb\": %d, \"waves\": %d, \"blocks_per_sm\": %d, "
           "\"regs\": %d, \"words\": %d, \"sms\": %d, \"sq_per_s_max\": %.4e, \"sq_per_s_median\": %.4e}\n",
           fold, ilp, K, tpb, waves, per_sm, attr.numRegs, n, prop.multiProcessorCount, rate.back(),
           rate[rate.size() / 2]);
    return 0;
  }
  return 2;
}
#endif
