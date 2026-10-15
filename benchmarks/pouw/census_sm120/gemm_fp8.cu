// A dense FP8 GEMM to keep a GPU busy beside host_amx.sh's AMX runs: cuBLASLt D (BF16, n x n) = A . B^T with A and B
// n x n E4M3 (TN, as benchmarks/pouw/harness/native/lt.cu), FP32 compute, the heuristic's first algorithm, on CUDA
// device 0, back to back for SECONDS. Prints one JSON line a batch of BATCH GEMMs (CLOCK_REALTIME bounds, MAC/s by CUDA
// events) and a closing line with the checksum of D's first row. Waits block (no spinning host thread). Stops early
// once the file GEMM_STOP names exists.
// Usage: gemm_fp8 N SECONDS [BATCH [TAG]]
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <vector>

#define CK(x)                                                                \
  do {                                                                       \
    cudaError_t e_ = (x);                                                    \
    if (e_ != cudaSuccess) {                                                 \
      fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));               \
      return 1;                                                              \
    }                                                                        \
  } while (0)
#define LT(x)                                                                \
  do {                                                                       \
    cublasStatus_t s_ = (x);                                                 \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "%s: cublasStatus %d\n", #x, int(s_));                 \
      return 1;                                                              \
    }                                                                        \
  } while (0)

static double wall() {
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: gemm_fp8 N SECONDS [BATCH [TAG]]\n");
    return 2;
  }
  const int n = atoi(argv[1]);
  const double seconds = atof(argv[2]);
  const int batch = argc > 3 ? atoi(argv[3]) : 100;
  const char* tag = argc > 4 ? argv[4] : "";
  CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
  CK(cudaSetDevice(0));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const unsigned char* u = reinterpret_cast<const unsigned char*>(p.uuid.bytes);
  char uuid[48];
  snprintf(uuid, sizeof uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2],
           u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);

  const size_t elems = size_t(n) * n;
  std::vector<uint8_t> h(elems);
  uint64_t s = 0x9e3779b97f4a7c15ull;
  for (auto& b : h) {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    b = uint8_t(s >> 32) & 0xBF;  // E4M3 with exponent field <= 7: finite, |x| < 2
  }
  uint8_t *dA, *dB;
  uint16_t* dD;
  void* ws;
  size_t wsb = 64ull << 20;
  CK(cudaMalloc(&dA, elems));
  CK(cudaMalloc(&dB, elems));
  CK(cudaMalloc(&dD, elems * 2));
  CK(cudaMalloc(&ws, wsb));
  CK(cudaMemcpy(dA, h.data(), elems, cudaMemcpyHostToDevice));
  for (auto& b : h) b ^= 0x15;
  CK(cudaMemcpy(dB, h.data(), elems, cudaMemcpyHostToDevice));

  cublasLtHandle_t lt;
  LT(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op;
  LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  cublasLtMatrixLayout_t la, lb, ld;
  LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, n, n, n));
  LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, n, n, n));
  LT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, n, n, n));
  cublasLtMatmulPreference_t pref;
  LT(cublasLtMatmulPreferenceCreate(&pref));
  LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb));
  cublasLtMatmulHeuristicResult_t res;
  int nres = 0;
  LT(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1, &res, &nres));
  if (nres < 1) {
    fprintf(stderr, "cuBLASLt heuristic found no algorithm for E4M3 TN n=%d\n", n);
    return 1;
  }
  const float alpha = 1.f, beta = 0.f;
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  cudaEvent_t e0, e1;
  CK(cudaEventCreateWithFlags(&e0, cudaEventBlockingSync));
  CK(cudaEventCreateWithFlags(&e1, cudaEventBlockingSync));
  auto gemm = [&] {
    return cublasLtMatmul(lt, op, &alpha, dB, la, dA, lb, &beta, dD, ld, dD, ld, &res.algo, ws, wsb, st);
  };
  for (int i = 0; i < 3; i++) LT(gemm());
  CK(cudaStreamSynchronize(st));

  const double macs = double(n) * n * n * batch;
  const double T0 = wall();
  const char* stop = getenv("GEMM_STOP");
  int nb = 0;
  while (wall() - T0 < seconds && !(stop && access(stop, F_OK) == 0)) {
    const double t0 = wall();
    CK(cudaEventRecord(e0, st));
    for (int i = 0; i < batch; i++) LT(gemm());
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    const double t1 = wall();
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("{\"probe\":\"gemm_fp8\",\"tag\":\"%s\",\"gpu_uuid\":\"%s\",\"n\":%d,\"batch\":%d,\"index\":%d,\"t0_unix\":%.3f,"
           "\"t1_unix\":%.3f,\"gpu_s\":%.6f,\"macs_per_s\":%.4g}\n",
           tag, uuid, n, batch, nb++, t0, t1, ms * 1e-3, macs / (ms * 1e-3));
    fflush(stdout);
  }
  std::vector<uint16_t> row(n);
  CK(cudaMemcpy(row.data(), dD, size_t(n) * 2, cudaMemcpyDeviceToHost));
  double sum = 0;
  int finite = 1;
  for (uint16_t v : row) {
    uint32_t w = uint32_t(v) << 16;
    float f;
    memcpy(&f, &w, 4);
    finite &= std::isfinite(f);
    sum += f;
  }
  printf("{\"probe\":\"gemm_fp8.done\",\"tag\":\"%s\",\"gpu_uuid\":\"%s\",\"n\":%d,\"batches\":%d,\"seconds\":%.3f,"
         "\"row0_sum\":%.6g,\"row0_finite\":%s}\n",
         tag, uuid, n, nb, wall() - T0, sum, finite ? "true" : "false");
  return 0;
}
