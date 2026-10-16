// The red team's deadline-slack probe (lane ra-rate-redteam). A rate audit's deadline must sit above the honest
// step's tail, and the adversary rebuilds until the deadline, not until the median. This replays a CUDA graph shaped
// like a decode step's weight reads (one streaming-read kernel a linear, 7 a layer) many times and records each
// step's time twice: between CUDA events on the GPU, and on the host from launch to the end of synchronization.
//
//   jitter LAYERS GIB ITERS OUT    e.g. 32 8 3000 (8B-like), 80 64 1000 (70B-like); JSON to OUT
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void fill(uint4* buf, size_t n16) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t z = i * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    buf[i] = make_uint4((uint32_t)z, (uint32_t)(z >> 32), (uint32_t)~z, (uint32_t)(i));
  }
}

__global__ void read_slice(const uint4* __restrict__ buf, size_t n16, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = __ldcs(buf + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) sink[0] = acc;
}

static double pct(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: jitter LAYERS GIB ITERS OUT\n");
    return 64;
  }
  const int layers = atoi(argv[1]), iters = atoi(argv[3]);
  const size_t gib = atoll(argv[2]), n16 = (gib << 30) / 16, kernels = (size_t)layers * 7, per = n16 / kernels;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  uint4* buf;
  uint32_t* sink;
  if (cudaMalloc(&buf, n16 * 16) || cudaMalloc(&sink, 4)) return 4;
  fill<<<4096, 256>>>(buf, n16);
  cudaDeviceSynchronize();
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  cudaGraph_t g;
  cudaGraphExec_t ge;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  for (size_t k = 0; k < kernels; ++k) read_slice<<<sms * 4, 512, 0, s>>>(buf + k * per, per, sink);
  cudaStreamEndCapture(s, &g);
  if (cudaGraphInstantiate(&ge, g, 0)) return 5;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int w = 0; w < 20; ++w) cudaGraphLaunch(ge, s);
  cudaStreamSynchronize(s);
  std::vector<double> gpu(iters), host(iters);
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    cudaEventRecord(a, s);
    cudaGraphLaunch(ge, s);
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    const auto t1 = std::chrono::steady_clock::now();
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    gpu[i] = ms;
    host[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  if (cudaGetLastError()) return 3;
  FILE* f = fopen(argv[4], "w");
  fprintf(f, "{\"layers\": %d, \"kernels\": %zu, \"bytes\": %zu, \"iters\": %d", layers, kernels, per * kernels * 16, iters);
  const char* names[2] = {"gpu_ms", "host_ms"};
  std::vector<double>* sets[2] = {&gpu, &host};
  for (int j = 0; j < 2; ++j) {
    const auto& v = *sets[j];
    fprintf(f, ", \"%s\": {\"p50\": %.4f, \"p90\": %.4f, \"p99\": %.4f, \"p999\": %.4f, \"max\": %.4f, \"min\": %.4f, \"samples\": [",
            names[j], pct(v, .5), pct(v, .9), pct(v, .99), pct(v, .999), *std::max_element(v.begin(), v.end()),
            *std::min_element(v.begin(), v.end()));
    for (int i = 0; i < iters; ++i) fprintf(f, "%s%.4f", i ? "," : "", v[i]);
    fprintf(f, "]}");
  }
  fprintf(f, "}\n");
  fclose(f);
  printf("%d layers, %zu GiB: gpu p50 %.3f p99 %.3f p999 %.3f max %.3f; host p50 %.3f p99 %.3f p999 %.3f max %.3f ms\n",
         layers, gib, pct(gpu, .5), pct(gpu, .99), pct(gpu, .999), *std::max_element(gpu.begin(), gpu.end()),
         pct(host, .5), pct(host, .99), pct(host, .999), *std::max_element(host.begin(), host.end()));
  return 0;
}
