// The FFT part of one squaring mod 2^1257787 - 1 (the 1e6 public Sloth's decode), batched, on cuFFT in FP64: weight,
// forward D2Z, pointwise square, inverse Z2D, unweight and round with the round-off error. It leaves out the carry pass,
// so its rate bounds the real squaring's from above. Each word is N = 2^17 points of B_PT bits.
//
//   sloth_fft WORDS REPS     # one JSON line: words, points, bits a point, seconds a batch, word GB/s, max round-off
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef SL_LOG_N
#define SL_LOG_N 17
#endif
constexpr int N = 1 << SL_LOG_N;
constexpr double P_BITS = 1257787.0;

#define CK(x)                                                                         \
  do {                                                                                \
    auto e_ = (x);                                                                    \
    if (e_ != 0) {                                                                    \
      std::fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, int(e_));      \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

__global__ void weight(double* x, const double* w, int words) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < size_t(words) * N) x[i] *= w[i & (N - 1)];
}

__global__ void square(cufftDoubleComplex* z, size_t count) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < count) {
    cufftDoubleComplex a = z[i];
    z[i] = make_cuDoubleComplex(a.x * a.x - a.y * a.y, 2 * a.x * a.y);
  }
}

__global__ void unweight(double* x, const double* winv, int words, double* err) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  double e = 0;
  if (i < size_t(words) * N) {
    double v = x[i] * winv[i & (N - 1)];
    double r = rint(v);
    e = fabs(v - r);
    x[i] = r;
  }
  for (int o = 16; o; o >>= 1) e = fmax(e, __shfl_xor_sync(0xffffffffu, e, o));
  if ((threadIdx.x & 31) == 0 && e > 0) atomicMax(reinterpret_cast<unsigned long long*>(err), __double_as_longlong(e));
}

int main(int argc, char** argv) {
  const int words = argc > 1 ? std::atoi(argv[1]) : 64;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  const double bpt = P_BITS / N;
  std::vector<double> w(N), winv(N), host(size_t(words) * N);
  for (int j = 0; j < N; ++j) {
    const double frac = std::ceil(P_BITS * j / N) - P_BITS * j / N;
    w[j] = std::exp2(frac);
    winv[j] = std::exp2(-frac) / N;
  }
  unsigned long long s = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < host.size(); ++i) {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    const int j = int(i & (N - 1));
    const int bits = int(std::ceil(P_BITS * (j + 1) / N) - std::ceil(P_BITS * j / N));
    host[i] = double(s & ((1ull << bits) - 1)) - double(1ull << (bits - 1));
  }
  double *x, *dw, *dwinv, *err;
  cufftDoubleComplex* z;
  const size_t nc = size_t(words) * (N / 2 + 1);
  CK(cudaMalloc(&x, host.size() * sizeof(double)));
  CK(cudaMalloc(&z, nc * sizeof(cufftDoubleComplex)));
  CK(cudaMalloc(&dw, N * sizeof(double)));
  CK(cudaMalloc(&dwinv, N * sizeof(double)));
  CK(cudaMalloc(&err, sizeof(double)));
  CK(cudaMemcpy(dw, w.data(), N * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwinv, winv.data(), N * sizeof(double), cudaMemcpyHostToDevice));
  cufftHandle fwd, inv;
  int n = N;
  CK(cufftPlanMany(&fwd, 1, &n, nullptr, 1, N, nullptr, 1, N / 2 + 1, CUFFT_D2Z, words));
  CK(cufftPlanMany(&inv, 1, &n, nullptr, 1, N / 2 + 1, nullptr, 1, N, CUFFT_Z2D, words));
  const int tpb = 256;
  const unsigned gr = unsigned((host.size() + tpb - 1) / tpb), gc = unsigned((nc + tpb - 1) / tpb);
  auto once = [&] {
    weight<<<gr, tpb>>>(x, dw, words);
    CK(cufftExecD2Z(fwd, x, z));
    square<<<gc, tpb>>>(z, nc);
    CK(cufftExecZ2D(inv, z, x));
    unweight<<<gr, tpb>>>(x, dwinv, words, err);
  };
  CK(cudaMemcpy(x, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemset(err, 0, sizeof(double)));
  once();
  CK(cudaDeviceSynchronize());
  double maxerr;
  CK(cudaMemcpy(&maxerr, err, sizeof(double), cudaMemcpyDeviceToHost));
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  std::vector<float> ms(reps);
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemcpy(x, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaEventRecord(a));
    once();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms[r], a, b));
  }
  std::vector<float> sorted = ms;
  std::sort(sorted.begin(), sorted.end());
  const double med = sorted[reps / 2] * 1e-3, best = sorted[0] * 1e-3;
  const double word_bytes = P_BITS / 8;
  std::printf("{\"words\": %d, \"points\": %d, \"bits_per_point\": %.3f, \"s_median\": %.6e, \"s_best\": %.6e, "
              "\"word_gb_per_s_median\": %.2f, \"word_gb_per_s_best\": %.2f, \"max_roundoff\": %.4f}\n",
              words, N, bpt, med, best, words * word_bytes / med / 1e9, words * word_bytes / best / 1e9, maxerr);
  return 0;
}
