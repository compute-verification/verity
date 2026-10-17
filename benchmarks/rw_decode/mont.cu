// mont_core.h on one GPU (sm_120): Montgomery squarings mod an RSA modulus of -DMB_BITS bits, in FORM 0-5, and the
// fused RW-SMS decode (records in, payloads out), each gated against rw.py before run.sh times it.
//   mont clock                                  the SM clock off a timed clock64 spin
//   mont chain-gate FORM K params.txt N         N words below N, K squarings: "x0 xK" in hex
//   mont decode-gate FORM params.txt records.txt   each record's payload in hex
//   mont chain-time FORM K params.txt TPB WAVES REPS   squarings a second, the occupancy's blocks a SM times WAVES
//   mont decode-time FORM params.txt MB REPS     records of MB MiB of c (plus a tag byte each) decoded into payloads
//   mont bw MB REPS                             a read (uint4 loads, xor-folded) and a copy of MB MiB, GB/s
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

#include "mont_io.h"

namespace mb {

template <int FORM>
__global__ void __launch_bounds__(128) chain(u32 *xs, int n, int K) {
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= n) return;
  u32 x[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = xs[(size_t)k * n + w];
#pragma unroll 1
  for (int r = 0; r < K; ++r) {
    u32 y[NL];
    mont_sq<FORM>(x, y);
#pragma unroll
    for (int k = 0; k < NL; ++k) x[k] = y[k];
  }
#pragma unroll
  for (int k = 0; k < NL; ++k) xs[(size_t)k * n + w] = x[k];
}

template <int FORM>
__global__ void __launch_bounds__(128) dec(const u32 *cs, const uint8_t *tags, u32 *ms, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  u32 c[NL], m[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) c[k] = cs[(size_t)k * n + j];
  decode<FORM>(c, tags[j], (u32)j, m);
#pragma unroll
  for (int k = 0; k < NL; ++k) ms[(size_t)k * n + j] = m[k];
}

__global__ void rd(const uint4 *s, size_t n4, u32 *out) {
  u32 a = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = s[i];
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

__global__ void cp(const uint4 *s, uint4 *d, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) d[i] = s[i];
}

__global__ void spin(long long cycles, long long *out) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
  *out = clock64() - t0;
}

}  // namespace mb

using mb::NL;
using mb::u32;

#define CK(x)                                                                            \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) {                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(3);                                                                           \
    }                                                                                    \
  } while (0)

// -DMB_FORM=F builds form F alone (run.sh builds one binary a form, so their ptxas runs in parallel); else all six.
#ifndef MB_FORM
#define MB_FORM -1
#endif
using Chain = void (*)(u32 *, int, int);
using Dec = void (*)(const u32 *, const uint8_t *, u32 *, int);
#define MB_HAS(F) (MB_FORM < 0 || MB_FORM == F)
static Chain chain_k(int f) {
#if MB_HAS(0)
  if (f == 0) return mb::chain<0>;
#endif
#if MB_HAS(1)
  if (f == 1) return mb::chain<1>;
#endif
#if MB_HAS(2)
  if (f == 2) return mb::chain<2>;
#endif
#if MB_HAS(3)
  if (f == 3) return mb::chain<3>;
#endif
#if MB_HAS(4)
  if (f == 4) return mb::chain<4>;
#endif
#if MB_HAS(5)
  if (f == 5) return mb::chain<5>;
#endif
  fprintf(stderr, "form %d not built\n", f);
  exit(2);
}
static Dec dec_k(int f) {
#if MB_HAS(0)
  if (f == 0) return mb::dec<0>;
#endif
#if MB_HAS(1)
  if (f == 1) return mb::dec<1>;
#endif
#if MB_HAS(2)
  if (f == 2) return mb::dec<2>;
#endif
#if MB_HAS(3)
  if (f == 3) return mb::dec<3>;
#endif
#if MB_HAS(4)
  if (f == 4) return mb::dec<4>;
#endif
#if MB_HAS(5)
  if (f == 5) return mb::dec<5>;
#endif
  fprintf(stderr, "form %d not built\n", f);
  exit(2);
}

static mbio::Params load(const char *path) {
  mbio::Params p = mbio::read_params(path);
  CK(cudaMemcpyToSymbol(mb::cN, p.N, sizeof p.N));
  CK(cudaMemcpyToSymbol(mb::cNp, &p.np, sizeof p.np));
  CK(cudaMemcpyToSymbol(mb::cK, p.K, sizeof p.K));
  CK(cudaMemcpyToSymbol(mb::cT, p.T, sizeof p.T));
  CK(cudaMemcpyToSymbol(mb::cKj, p.Kj, sizeof p.Kj));
  return p;
}

static double ms_of(cudaEvent_t a, cudaEvent_t b) {
  float ms;
  CK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

template <class Kern>
static void attrs(Kern k, int tpb, int &per_sm, cudaFuncAttributes &attr) {
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, tpb, 0));
  CK(cudaFuncGetAttributes(&attr, k));
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *cmd = argv[1];
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  if (!strcmp(cmd, "clock")) {
    long long *d;
    CK(cudaMalloc(&d, sizeof(long long)));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    mb::spin<<<1, 1>>>(1LL << 26, d);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    mb::spin<<<1, 1>>>(1LL << 30, d);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    long long cyc;
    CK(cudaMemcpy(&cyc, d, sizeof cyc, cudaMemcpyDeviceToHost));
    printf("{\"clock_mhz\": %.1f, \"sms\": %d, \"name\": \"%s\"}\n", cyc / (ms_of(a, b) * 1e3), prop.multiProcessorCount,
           prop.name);
    return 0;
  }
  if (!strcmp(cmd, "chain-gate") && argc == 6) {
    const int f = atoi(argv[2]), K = atoi(argv[3]), n = atoi(argv[5]);
    mbio::Params p = load(argv[4]);
    std::vector<u32> h0, h;
    mbio::fill_below(p, h0, n, 0x5eedull);
    u32 *d;
    CK(cudaMalloc(&d, h0.size() * 4));
    CK(cudaMemcpy(d, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice));
    chain_k(f)<<<(n + 127) / 128, 128>>>(d, n, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    h.resize(h0.size());
    CK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
    for (int w = 0; w < n; ++w) {
      u32 x0[NL], x[NL];
      for (int k = 0; k < NL; ++k) {
        x0[k] = h0[(size_t)k * n + w];
        x[k] = h[(size_t)k * n + w];
      }
      mbio::print_hex(x0);
      printf(" ");
      mbio::print_hex(x);
      printf("\n");
    }
    return 0;
  }
  if (!strcmp(cmd, "decode-gate") && argc == 5) {
    const int f = atoi(argv[2]);
    load(argv[3]);
    mbio::Records r = mbio::read_records(argv[4]);
    const int n = r.n;
    u32 *dc, *dm;
    uint8_t *dt;
    CK(cudaMalloc(&dc, r.c.size() * 4));
    CK(cudaMalloc(&dm, r.c.size() * 4));
    CK(cudaMalloc(&dt, n));
    CK(cudaMemcpy(dc, r.c.data(), r.c.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dt, r.tag.data(), n, cudaMemcpyHostToDevice));
    dec_k(f)<<<(n + 127) / 128, 128>>>(dc, dt, dm, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<u32> m(r.c.size());
    CK(cudaMemcpy(m.data(), dm, m.size() * 4, cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; ++j) {
      u32 x[NL];
      for (int k = 0; k < NL; ++k) x[k] = m[(size_t)k * n + j];
      mbio::print_hex(x);
      printf("\n");
    }
    return 0;
  }
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  if (!strcmp(cmd, "chain-time") && argc == 8) {
    const int f = atoi(argv[2]), K = atoi(argv[3]), tpb = atoi(argv[5]), waves = atoi(argv[6]), reps = atoi(argv[7]);
    mbio::Params p = load(argv[4]);
    int per_sm;
    cudaFuncAttributes attr;
    attrs(chain_k(f), tpb, per_sm, attr);
    const int n = prop.multiProcessorCount * per_sm * tpb * waves;
    std::vector<u32> h;
    mbio::fill_below(p, h, n, 0x7177ull);
    u32 *d;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    chain_k(f)<<<n / tpb, tpb>>>(d, n, K / 10 + 1);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<double> rate;
    for (int r = 0; r < reps; ++r) {
      CK(cudaEventRecord(a));
      chain_k(f)<<<n / tpb, tpb>>>(d, n, K);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      rate.push_back((double)n * K / (ms_of(a, b) * 1e-3));
    }
    std::sort(rate.begin(), rate.end());
    printf("{\"bits\": %d, \"form\": %d, \"K\": %d, \"tpb\": %d, \"waves\": %d, \"blocks_per_sm\": %d, \"regs\": %d, "
           "\"local_bytes\": %zu, \"words\": %d, \"sms\": %d, \"sq_per_s_max\": %.4e, \"sq_per_s_median\": %.4e}\n",
           32 * NL, f, K, tpb, waves, per_sm, attr.numRegs, attr.localSizeBytes, n, prop.multiProcessorCount,
           rate.back(), rate[rate.size() / 2]);
    return 0;
  }
  if (!strcmp(cmd, "decode-time") && argc == 6) {
    const int f = atoi(argv[2]), mbytes = atoi(argv[4]), reps = atoi(argv[5]);
    mbio::Params p = load(argv[3]);
    const int n = (int)(((size_t)mbytes << 20) / (4 * NL) / 128 * 128);
    std::vector<u32> h;
    mbio::fill_below(p, h, n, 0xdec0deull);
    std::vector<uint8_t> tg(n);
    uint64_t s = 0x7a6ull;
    for (auto &t : tg) t = (uint8_t)(mbio::splitmix(s) & 0x0f);
    u32 *dc, *dm;
    uint8_t *dt;
    CK(cudaMalloc(&dc, h.size() * 4));
    CK(cudaMalloc(&dm, h.size() * 4));
    CK(cudaMalloc(&dt, n));
    CK(cudaMemcpy(dc, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dt, tg.data(), n, cudaMemcpyHostToDevice));
    int per_sm;
    cudaFuncAttributes attr;
    attrs(dec_k(f), 128, per_sm, attr);
    dec_k(f)<<<n / 128, 128>>>(dc, dt, dm, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<double> rec;
    for (int r = 0; r < reps; ++r) {
      CK(cudaEventRecord(a));
      dec_k(f)<<<n / 128, 128>>>(dc, dt, dm, n);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      rec.push_back((double)n / (ms_of(a, b) * 1e-3));
    }
    std::sort(rec.begin(), rec.end());
    const double rb = 4.0 * NL + 1, pb = 4.0 * NL - 1, med = rec[rec.size() / 2];
    printf("{\"bits\": %d, \"form\": %d, \"records\": %d, \"record_bytes\": %.0f, \"payload_bytes\": %.0f, "
           "\"blocks_per_sm\": %d, \"regs\": %d, \"local_bytes\": %zu, \"records_per_s_max\": %.4e, "
           "\"records_per_s_median\": %.4e, \"record_gb_per_s_median\": %.2f, \"payload_gb_per_s_median\": %.2f, "
           "\"traffic_gb_per_s_median\": %.2f}\n",
           32 * NL, f, n, rb, pb, per_sm, attr.numRegs, attr.localSizeBytes, rec.back(), med, med * rb / 1e9,
           med * pb / 1e9, med * (rb + 4.0 * NL) / 1e9);
    return 0;
  }
  if (!strcmp(cmd, "bw") && argc == 4) {
    const size_t bytes = (size_t)atoi(argv[2]) << 20, n4 = bytes / 16;
    const int reps = atoi(argv[3]);
    uint4 *s, *d;
    u32 *o;
    CK(cudaMalloc(&s, bytes));
    CK(cudaMalloc(&d, bytes));
    const int grid = prop.multiProcessorCount * 8, tpb = 256;
    CK(cudaMalloc(&o, (size_t)grid * tpb * 4));
    CK(cudaMemset(s, 0x5a, bytes));
    std::vector<double> rgb, cgb;
    for (int r = 0; r < reps + 1; ++r) {
      CK(cudaEventRecord(a));
      mb::rd<<<grid, tpb>>>(s, n4, o);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      if (r) rgb.push_back(bytes / (ms_of(a, b) * 1e-3) / 1e9);
      CK(cudaEventRecord(a));
      mb::cp<<<grid, tpb>>>(s, d, n4);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      if (r) cgb.push_back(2.0 * bytes / (ms_of(a, b) * 1e-3) / 1e9);
    }
    std::sort(rgb.begin(), rgb.end());
    std::sort(cgb.begin(), cgb.end());
    printf("{\"bytes\": %zu, \"read_gb_per_s_max\": %.1f, \"read_gb_per_s_median\": %.1f, \"copy_traffic_gb_per_s_max\": %.1f, "
           "\"copy_traffic_gb_per_s_median\": %.1f}\n",
           bytes, rgb.back(), rgb[rgb.size() / 2], cgb.back(), cgb[cgb.size() / 2]);
    return 0;
  }
  return 2;
}
