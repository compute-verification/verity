// cuSPARSELt 2:4-structured sparse GEMM config sweeper for sm_120: find which (valueType, order, out type, scale mode, n)
// cuSPARSELt 0.9.1 accepts for E2M1 (NVFP4).  FP8 e4m3 is swept first to confirm the API flow is right, isolating FP4 as
// the variable.  Each config runs descriptors -> matmul desc -> scales -> alg selection -> plan -> compress -> matmul and
// reports the stage it reached and the status, non-fatally, plus a CPU int64 cross-check when it completes.
//
//   sp_smoke [n]
#include <cusparseLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));       \
      exit(3);                                                                                 \
    }                                                                                          \
  } while (0)

static int e2m1_code(int v) {
  int m = v < 0 ? -v : v, c;
  switch (m) {
    case 0: c = 0; break;
    case 1: c = 2; break;
    case 2: c = 4; break;
    case 3: c = 5; break;
    case 4: c = 6; break;
    case 6: c = 7; break;
    default: fprintf(stderr, "value %d not E2M1\n", v); exit(5);
  }
  return c | (v < 0 ? 8 : 0);
}
static uint8_t fp8_e4m3(int v) {  // small integers exactly: build via __nv_fp8 would need the header; use known codes.
  // e4m3: 1->0x38, 2->0x40, -1->0xB8, -2->0xC0, 0->0x00.  (sign|exp(bias7)|mant)
  switch (v) {
    case 0: return 0x00;
    case 1: return 0x38;
    case 2: return 0x40;
    case -1: return 0xB8;
    case -2: return 0xC0;
  }
  return 0x00;
}
static uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static cusparseLtHandle_t H;

struct Cfg {
  cudaDataType in;    // operand type
  cudaDataType out;   // C/D type
  cusparseComputeType comp;
  cusparseOrder_t order;
  cusparseLtMatmulMatrixScale_t scale;  // (cast) or 255 for NONE
  bool use_scale;
  int n;
};

// Returns a short description of the furthest stage reached and sets *st to the last status; *mism to CPU mismatches (-1 if
// not reached).  Builds A 2:4-structured (values in {-2,-1,1,2}) and B dense (+-1); alpha=1.
static const char* run_cfg(const Cfg& c, int* st, long* mism) {
  const int M = c.n, K = c.n, N = c.n;
  const bool fp4 = c.in == CUDA_R_4F_E2M1;
  *mism = -1;
  *st = 0;
  std::vector<int> hA(size_t(M) * K, 0), hB(size_t(K) * N);
  for (int i = 0; i < M; ++i)
    for (int g = 0; g < K; g += 4) {
      uint64_t r = mix64(0x5EEDull ^ (uint64_t(i) << 20) ^ g);
      int p0 = r & 3, p1 = (r >> 2) & 3;
      if (p1 == p0) p1 = (p1 + 1) & 3;
      static const int MAP[4] = {1, 2, -1, -2};
      hA[size_t(i) * K + g + p0] = MAP[(r >> 8) & 3];
      hA[size_t(i) * K + g + p1] = MAP[(r >> 16) & 3];
    }
  for (size_t i = 0; i < size_t(K) * N; ++i) hB[i] = (mix64(0xABCDull ^ i) & 1) ? 1 : -1;

  // Pack operands.
  auto packed_bytes = [&](size_t elems) { return fp4 ? elems / 2 : elems; };
  std::vector<uint8_t> pA(packed_bytes(size_t(M) * K)), pB(packed_bytes(size_t(K) * N));
  if (fp4) {
    for (size_t i = 0; i < size_t(M) * K / 2; ++i) pA[i] = e2m1_code(hA[2 * i]) | (e2m1_code(hA[2 * i + 1]) << 4);
    for (size_t i = 0; i < size_t(K) * N / 2; ++i) pB[i] = e2m1_code(hB[2 * i]) | (e2m1_code(hB[2 * i + 1]) << 4);
  } else {
    for (size_t i = 0; i < size_t(M) * K; ++i) pA[i] = fp8_e4m3(hA[i]);
    for (size_t i = 0; i < size_t(K) * N; ++i) pB[i] = fp8_e4m3(hB[i]);
  }
  uint8_t *dA, *dB;
  void* dD;
  size_t outsz = c.out == CUDA_R_32F ? 4 : 2;
  CK(cudaMalloc(&dA, pA.size()));
  CK(cudaMalloc(&dB, pB.size()));
  CK(cudaMalloc(&dD, size_t(M) * N * outsz));
  CK(cudaMemcpy(dA, pA.data(), pA.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, pB.data(), pB.size(), cudaMemcpyHostToDevice));

  cusparseLtMatDescriptor_t mA, mB, mC;
  const int AL = 16;
  const int64_t ldA = K, ldB = N, ldC = N;  // row-major
  cusparseStatus_t s;
  s = cusparseLtStructuredDescriptorInit(&H, &mA, M, K, ldA, AL, c.in, c.order, CUSPARSELT_SPARSITY_50_PERCENT);
  *st = s; if (s) return "structA";
  s = cusparseLtDenseDescriptorInit(&H, &mB, K, N, ldB, AL, c.in, c.order);
  *st = s; if (s) return "denseB";
  s = cusparseLtDenseDescriptorInit(&H, &mC, M, N, ldC, AL, c.out, c.order);
  *st = s; if (s) return "denseC";
  cusparseLtMatmulDescriptor_t mm;
  s = cusparseLtMatmulDescriptorInit(&H, &mm, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &mA, &mB,
                                     &mC, &mC, c.comp);
  *st = s; if (s) return "matmulDesc";
  uint8_t *dSA = nullptr, *dSB = nullptr;
  if (c.use_scale) {
    s = cusparseLtMatmulDescSetAttribute(&H, &mm, CUSPARSELT_MATMUL_A_SCALE_MODE, &c.scale, sizeof c.scale);
    *st = s; if (s) return "setAscaleMode";
    s = cusparseLtMatmulDescSetAttribute(&H, &mm, CUSPARSELT_MATMUL_B_SCALE_MODE, &c.scale, sizeof c.scale);
    *st = s; if (s) return "setBscaleMode";
    size_t sb = size_t(M) * K;  // generous; uniform so layout-independent
    CK(cudaMalloc(&dSA, sb));
    CK(cudaMalloc(&dSB, sb));
    if (c.scale == CUSPARSELT_MATMUL_MATRIX_SCALE_SCALAR_32F) {
      float one = 1.f;
      CK(cudaMemcpy(dSA, &one, 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dSB, &one, 4, cudaMemcpyHostToDevice));
    } else {
      CK(cudaMemset(dSA, 0x38, sb));  // UE4M3 1.0; UE8M0 1.0 is 127 (0x7F) but we test UE4M3 path for NVFP4
      CK(cudaMemset(dSB, 0x38, sb));
    }
    void *pSA = dSA, *pSB = dSB;
    s = cusparseLtMatmulDescSetAttribute(&H, &mm, CUSPARSELT_MATMUL_A_SCALE_POINTER, &pSA, sizeof pSA);
    *st = s; if (s) return "setAscalePtr";
    s = cusparseLtMatmulDescSetAttribute(&H, &mm, CUSPARSELT_MATMUL_B_SCALE_POINTER, &pSB, sizeof pSB);
    *st = s; if (s) return "setBscalePtr";
  }
  cusparseLtMatmulAlgSelection_t alg;
  s = cusparseLtMatmulAlgSelectionInit(&H, &alg, &mm, CUSPARSELT_MATMUL_ALG_DEFAULT);
  *st = s; if (s) return "algSelect";
  cusparseLtMatmulPlan_t plan;
  s = cusparseLtMatmulPlanInit(&H, &plan, &mm, &alg);
  *st = s; if (s) return "planInit";
  size_t cSize = 0, cBuf = 0;
  s = cusparseLtSpMMACompressedSize(&H, &plan, &cSize, &cBuf);
  *st = s; if (s) return "compressedSize";
  uint8_t *dAc, *dAcBuf;
  CK(cudaMalloc(&dAc, cSize ? cSize : 1));
  CK(cudaMalloc(&dAcBuf, cBuf ? cBuf : 1));
  s = cusparseLtSpMMACompress(&H, &plan, dA, dAc, dAcBuf, nullptr);
  *st = s; if (s) return "compress";
  size_t wsSize = 0;
  cusparseLtMatmulGetWorkspace(&H, &plan, &wsSize);
  void* dWs;
  CK(cudaMalloc(&dWs, wsSize ? wsSize : 1));
  float alpha = 1.f, beta = 0.f;
  cudaStream_t str = nullptr;
  CK(cudaMemset(dD, 0, size_t(M) * N * outsz));
  s = cusparseLtMatmul(&H, &plan, &alpha, dAc, dB, &beta, dD, dD, dWs, &str, 1);
  *st = s; if (s) return "matmul";
  CK(cudaDeviceSynchronize());
  std::vector<uint8_t> hD(size_t(M) * N * outsz);
  CK(cudaMemcpy(hD.data(), dD, hD.size(), cudaMemcpyDeviceToHost));
  long bad = 0;
  for (int i = 0; i < M && bad < 1000000; ++i)
    for (int j = 0; j < N; ++j) {
      int64_t sref = 0;
      for (int kk = 0; kk < K; ++kk) sref += int64_t(hA[size_t(i) * K + kk]) * hB[size_t(kk) * N + j];
      double got = c.out == CUDA_R_32F ? double(((float*)hD.data())[size_t(i) * N + j])
                                       : double(__half2float(((__half*)hD.data())[size_t(i) * N + j]));
      if (double(sref) != got) ++bad;
    }
  *mism = bad;
  cusparseLtMatmulPlanDestroy(&plan);
  cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(dAc); cudaFree(dAcBuf); cudaFree(dWs);
  if (dSA) cudaFree(dSA);
  if (dSB) cudaFree(dSB);
  return "DONE";
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 256;
  cusparseStatus_t is = cusparseLtInit(&H);
  if (is) { fprintf(stderr, "init %d\n", int(is)); return 4; }
  int ver = 0;
  cusparseLtGetVersion(&H, &ver);
  printf("{\"cusparseLt_version\":%d,\"n\":%d}\n", ver, n);

  auto SC = [](cusparseLtMatmulMatrixScale_t m) {
    switch (m) {
      case CUSPARSELT_MATMUL_SCALE_NONE: return "NONE";
      case CUSPARSELT_MATMUL_MATRIX_SCALE_SCALAR_32F: return "SCALAR32F";
      case CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3: return "VEC32_UE4M3";
      case CUSPARSELT_MATMUL_MATRIX_SCALE_VEC64_UE8M0: return "VEC64_UE8M0";
    }
    return "?";
  };
  struct Case { const char* tag; Cfg c; };
  std::vector<Case> cases;
  // FP8 e4m3 sanity (no scale, FP32 out), both orders.
  cases.push_back({"fp8", {CUDA_R_8F_E4M3, CUDA_R_32F, CUSPARSE_COMPUTE_32F, CUSPARSE_ORDER_ROW,
                           CUSPARSELT_MATMUL_SCALE_NONE, false, n}});
  cases.push_back({"fp8_col", {CUDA_R_8F_E4M3, CUDA_R_32F, CUSPARSE_COMPUTE_32F, CUSPARSE_ORDER_COL,
                               CUSPARSELT_MATMUL_SCALE_NONE, false, n}});
  // FP4 e2m1 sweep.
  for (cusparseOrder_t ord : {CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL})
    for (cudaDataType out : {CUDA_R_32F, CUDA_R_16F})
      for (auto sm : {CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3, CUSPARSELT_MATMUL_MATRIX_SCALE_SCALAR_32F,
                      CUSPARSELT_MATMUL_SCALE_NONE})
        cases.push_back({"fp4", {CUDA_R_4F_E2M1, out, CUSPARSE_COMPUTE_32F, ord, sm,
                                 sm != CUSPARSELT_MATMUL_SCALE_NONE, n}});

  for (auto& cs : cases) {
    int st = 0;
    long mism = -2;
    const char* stage = run_cfg(cs.c, &st, &mism);
    printf("{\"tag\":\"%s\",\"in\":\"%s\",\"order\":\"%s\",\"out\":\"%s\",\"scale\":\"%s\",\"stage\":\"%s\","
           "\"status\":%d,\"name\":\"%s\",\"mismatches\":%ld}\n",
           cs.tag, cs.c.in == CUDA_R_4F_E2M1 ? "e2m1" : "e4m3", cs.c.order == CUSPARSE_ORDER_ROW ? "ROW" : "COL",
           cs.c.out == CUDA_R_32F ? "f32" : "f16", SC(cs.c.scale), stage, st, cusparseLtGetErrorName((cusparseStatus_t)st),
           mism);
    fflush(stdout);
  }
  cusparseLtDestroy(&H);
  return 0;
}
