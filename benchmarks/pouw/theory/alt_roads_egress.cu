// R1's GPU side: can one H200 source a 400 Gb/s (50 GB/s) stream of its words while it runs int8 GEMMs?
// Device-to-host copies into pinned memory stand in for a NIC's GPUDirect reads: both pull HBM over the GPU's PCIe
// link. Prints one JSON object: copies alone (both directions), cuBLAS int8 GEMMs alone at Llama-3.2-1B's shapes, and
// the same GEMMs under a device-to-host stream sized to outlast them. The host buffer is poisoned with 0xA5 and the
// device's with 0x3C, so a copy that moved nothing fails the check.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e = (x);                                                                         \
    if (e != cudaSuccess) {                                                                      \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e));          \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)
#define CB(x)                                                                                    \
  do {                                                                                           \
    cublasStatus_t s = (x);                                                                      \
    if (s != CUBLAS_STATUS_SUCCESS) {                                                            \
      fprintf(stderr, "%s:%d %s: cublas %d\n", __FILE__, __LINE__, #x, (int)s);                  \
      exit(3);                                                                                   \
    }                                                                                            \
  } while (0)

static const size_t BUF = size_t(2) << 30;
static const size_t CHUNK = size_t(64) << 20;

struct Gemm {
  int m, k, n;
};

static float copy_ms(void* dst, const void* src, size_t total, cudaMemcpyKind kind, cudaStream_t s, cudaEvent_t a,
                     cudaEvent_t b) {
  CK(cudaEventRecord(a, s));
  for (size_t off = 0; off < total; off += CHUNK) {
    size_t o = off % BUF;
    CK(cudaMemcpyAsync((char*)dst + o, (const char*)src + o, CHUNK, kind, s));
  }
  CK(cudaEventRecord(b, s));
  CK(cudaEventSynchronize(b));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

static void gemm(cublasHandle_t h, const Gemm& g, const int8_t* A, const int8_t* B, int32_t* C) {
  const int32_t one = 1, zero = 0;
  CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, g.n, g.m, g.k, &one, B, CUDA_R_8I, g.k, A, CUDA_R_8I, g.k, &zero, C,
                  CUDA_R_32I, g.n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
}

static float gemm_ms(cublasHandle_t h, const Gemm& g, int reps, const int8_t* A, const int8_t* B, int32_t* C,
                     cudaStream_t s, cudaEvent_t a, cudaEvent_t b) {
  CK(cudaEventRecord(a, s));
  for (int r = 0; r < reps; ++r) gemm(h, g, A, B, C);
  CK(cudaEventRecord(b, s));
  CK(cudaEventSynchronize(b));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

static bool host_holds(const unsigned char* h, unsigned char want) {
  for (size_t off = 0; off < BUF; off += CHUNK)
    if (h[off] != want || h[off + CHUNK - 1] != want) return false;
  return true;
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  unsigned char *dev, *host;
  CK(cudaMalloc(&dev, BUF));
  CK(cudaHostAlloc(&host, BUF, cudaHostAllocDefault));
  CK(cudaMemset(dev, 0x3C, BUF));
  memset(host, 0xA5, BUF);
  CK(cudaDeviceSynchronize());

  cudaStream_t s0, s1;
  CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  cudaEvent_t a0, b0, a1, b1;
  CK(cudaEventCreate(&a0));
  CK(cudaEventCreate(&b0));
  CK(cudaEventCreate(&a1));
  CK(cudaEventCreate(&b1));

  printf("{\"device\": \"%s\", \"sms\": %d, \"pci_bus\": %d, \"copy_engines\": %d,\n", p.name, p.multiProcessorCount,
         p.pciBusID, p.asyncEngineCount);

  const size_t total = size_t(8) << 30;
  copy_ms(host, dev, BUF, cudaMemcpyDeviceToHost, s1, a1, b1);
  std::vector<float> d2h, h2d;
  for (int r = 0; r < 3; ++r) d2h.push_back(copy_ms(host, dev, total, cudaMemcpyDeviceToHost, s1, a1, b1));
  bool d2h_moved = host_holds(host, 0x3C);
  memset(host, 0x5A, BUF);
  for (int r = 0; r < 3; ++r) h2d.push_back(copy_ms(dev, host, total, cudaMemcpyHostToDevice, s1, a1, b1));
  CK(cudaMemset(dev, 0x3C, BUF));
  memset(host, 0xA5, BUF);
  CK(cudaDeviceSynchronize());
  printf(" \"copy_bytes\": %zu, \"d2h_ms\": [%.3f, %.3f, %.3f], \"h2d_ms\": [%.3f, %.3f, %.3f], \"d2h_moved\": %s,\n",
         total, d2h[0], d2h[1], d2h[2], h2d[0], h2d[1], h2d[2], d2h_moved ? "true" : "false");

  cublasHandle_t h;
  CB(cublasCreate(&h));
  CB(cublasSetStream(h, s0));
  const std::vector<Gemm> shapes = {{64, 2048, 16384}, {512, 2048, 16384}, {8192, 2048, 16384}, {8192, 8192, 2048},
                                    {8192, 2048, 3072}};
  size_t maxA = 0, maxB = 0, maxC = 0;
  for (auto& g : shapes) {
    maxA = std::max(maxA, size_t(g.m) * g.k);
    maxB = std::max(maxB, size_t(g.k) * g.n);
    maxC = std::max(maxC, size_t(g.m) * g.n);
  }
  int8_t *A, *B;
  int32_t* C;
  CK(cudaMalloc(&A, maxA));
  CK(cudaMalloc(&B, maxB));
  CK(cudaMalloc(&C, maxC * 4));
  CK(cudaMemset(A, 1, maxA));
  CK(cudaMemset(B, 1, maxB));

  printf(" \"gemms\": [");
  for (size_t i = 0; i < shapes.size(); ++i) {
    const Gemm& g = shapes[i];
    double macs = double(g.m) * g.k * g.n;
    int reps = std::max(20, int(2e12 / macs));
    gemm_ms(h, g, reps, A, B, C, s0, a0, b0);
    float alone = gemm_ms(h, g, reps, A, B, C, s0, a0, b0);
    int32_t c00 = 0;
    CK(cudaMemcpy(&c00, C, 4, cudaMemcpyDeviceToHost));
    // The stream is sized to outlast the GEMMs at 60 GB/s, so every GEMM runs under it.
    size_t want = size_t(alone * 1e-3 * 60e9 * 1.5);
    size_t stream_bytes = std::max(size_t(4), (want + CHUNK - 1) / CHUNK) * CHUNK;
    memset(host, 0xA5, BUF);
    CK(cudaEventRecord(a1, s1));
    for (size_t off = 0; off < stream_bytes; off += CHUNK)
      CK(cudaMemcpyAsync(host + off % BUF, dev + off % BUF, CHUNK, cudaMemcpyDeviceToHost, s1));
    CK(cudaEventRecord(b1, s1));
    CK(cudaEventRecord(a0, s0));
    for (int r = 0; r < reps; ++r) gemm(h, g, A, B, C);
    CK(cudaEventRecord(b0, s0));
    CK(cudaEventSynchronize(b0));
    CK(cudaEventSynchronize(b1));
    float under = 0, stream = 0, lag = 0;
    CK(cudaEventElapsedTime(&under, a0, b0));
    CK(cudaEventElapsedTime(&stream, a1, b1));
    CK(cudaEventElapsedTime(&lag, a1, a0));
    bool moved = stream_bytes >= BUF ? host_holds(host, 0x3C) : (host[0] == 0x3C);
    printf("%s\n  {\"m\": %d, \"k\": %d, \"n\": %d, \"reps\": %d, \"alone_ms\": %.4f, \"under_stream_ms\": %.4f, "
           "\"stream_bytes\": %zu, \"stream_ms\": %.4f, \"gemm_start_after_stream_start_ms\": %.4f, \"c00\": %d, "
           "\"c00_want\": %d, \"stream_moved\": %s}",
           i ? "," : "", g.m, g.k, g.n, reps, alone, under, stream_bytes, stream, lag, c00, g.k, moved ? "true" : "false");
  }
  printf("\n ]}\n");
  CB(cublasDestroy(h));
  return 0;
}
