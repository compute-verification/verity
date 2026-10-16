// cuSPARSELt 2:4-structured NVFP4 (E2M1) GEMM smoke test on sm_120: does the library compute a bit-exact small-integer
// product D = A·B with A 2:4-sparse E2M1 and B dense E2M1, FP32 accumulate?  Resolves the conventions the full three-piece
// benchmark needs: FP4 nibble order, whether FP4 needs a block scale (SCALE_NONE vs VEC32_UE4M3), prune/compress, and that
// scalar alpha recovers an integer product.  Prints each cuSPARSELt status and a CPU int64 cross-check.
//
//   sp_smoke <n> <seed>     n a multiple of 256
#include <cusparseLt.h>
#include <cuda_runtime.h>

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
#define SP(x)                                                                                  \
  do {                                                                                         \
    cusparseStatus_t s_ = (x);                                                                 \
    if (s_ != CUSPARSE_STATUS_SUCCESS) {                                                       \
      fprintf(stderr, "%s:%d %s: cusparseStatus %d (%s)\n", __FILE__, __LINE__, #x, int(s_),   \
              cusparseGetErrorString(s_));                                                      \
      exit(4);                                                                                 \
    }                                                                                          \
  } while (0)

// E2M1 code of an integer at scale 1: magnitudes 0 1 2 3 4 6 -> codes 0 2 4 5 6 7; sign bit 8.
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

static uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 256;
  const uint64_t seed = argc > 2 ? strtoull(argv[2], nullptr, 0) : 0x5EEDull;
  const int M = n, K = n, N = n;

  // Host A (M x K row-major int, 2:4 structured along K, values in {-2,-1,1,2}) and B (K x N row-major, values +-1).
  std::vector<int> hA(size_t(M) * K, 0), hB(size_t(K) * N);
  for (int i = 0; i < M; ++i)
    for (int g = 0; g < K; g += 4) {
      uint64_t r = mix64(seed ^ (uint64_t(i) << 20) ^ g);
      int p0 = r & 3, p1 = (r >> 2) & 3;
      if (p1 == p0) p1 = (p1 + 1) & 3;
      int vals[2] = {int((r >> 8) & 3), int((r >> 16) & 3)};
      static const int MAP[4] = {1, 2, -1, -2};
      hA[size_t(i) * K + g + p0] = MAP[vals[0]];
      hA[size_t(i) * K + g + p1] = MAP[vals[1]];
    }
  for (size_t i = 0; i < size_t(K) * N; ++i) hB[i] = (mix64(seed ^ 0xABCDull ^ i) & 1) ? 1 : -1;

  // Pack to E2M1, 2 values a byte: low nibble = even column.
  std::vector<uint8_t> pA(size_t(M) * K / 2), pB(size_t(K) * N / 2);
  for (size_t i = 0; i < size_t(M) * K / 2; ++i) pA[i] = e2m1_code(hA[2 * i]) | (e2m1_code(hA[2 * i + 1]) << 4);
  for (size_t i = 0; i < size_t(K) * N / 2; ++i) pB[i] = e2m1_code(hB[2 * i]) | (e2m1_code(hB[2 * i + 1]) << 4);

  uint8_t *dA, *dB;
  float* dD;
  CK(cudaMalloc(&dA, pA.size()));
  CK(cudaMalloc(&dB, pB.size()));
  CK(cudaMalloc(&dD, size_t(M) * N * sizeof(float)));
  CK(cudaMemcpy(dA, pA.data(), pA.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, pB.data(), pB.size(), cudaMemcpyHostToDevice));

  cusparseLtHandle_t h;
  SP(cusparseLtInit(&h));
  int ver = 0;
  cusparseLtGetVersion(&h, &ver);
  printf("{\"cusparseLt_version\":%d}\n", ver);

  cusparseLtMatDescriptor_t mA, mB, mC;
  const int ALIGN = 16;
  SP(cusparseLtStructuredDescriptorInit(&h, &mA, M, K, K, ALIGN, CUDA_R_4F_E2M1, CUSPARSE_ORDER_ROW,
                                        CUSPARSELT_SPARSITY_50_PERCENT));
  SP(cusparseLtDenseDescriptorInit(&h, &mB, K, N, N, ALIGN, CUDA_R_4F_E2M1, CUSPARSE_ORDER_ROW));
  SP(cusparseLtDenseDescriptorInit(&h, &mC, M, N, N, ALIGN, CUDA_R_32F, CUSPARSE_ORDER_ROW));

  cusparseLtMatmulDescriptor_t mm;
  SP(cusparseLtMatmulDescriptorInit(&h, &mm, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &mA, &mB,
                                    &mC, &mC, CUSPARSE_COMPUTE_32F));

  // FP4 (E2M1) requires a block-scale mode.  A uniform scale of 1.0 (UE4M3 code 0x38) is layout-independent, so a
  // generously-sized all-0x38 array works and the scalar alpha carries any per-operand scale.
  cusparseLtMatmulMatrixScale_t smode = CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3;
  SP(cusparseLtMatmulDescSetAttribute(&h, &mm, CUSPARSELT_MATMUL_A_SCALE_MODE, &smode, sizeof smode));
  SP(cusparseLtMatmulDescSetAttribute(&h, &mm, CUSPARSELT_MATMUL_B_SCALE_MODE, &smode, sizeof smode));
  uint8_t *dSA, *dSB;
  size_t scaleBytes = size_t(M) * K / 16;  // generous (VEC32 needs K/32 per row); all 1.0
  CK(cudaMalloc(&dSA, scaleBytes));
  CK(cudaMalloc(&dSB, size_t(K) * N / 16));
  CK(cudaMemset(dSA, 0x38, scaleBytes));
  CK(cudaMemset(dSB, 0x38, size_t(K) * N / 16));
  void* pSA = dSA;
  void* pSB = dSB;
  SP(cusparseLtMatmulDescSetAttribute(&h, &mm, CUSPARSELT_MATMUL_A_SCALE_POINTER, &pSA, sizeof pSA));
  SP(cusparseLtMatmulDescSetAttribute(&h, &mm, CUSPARSELT_MATMUL_B_SCALE_POINTER, &pSB, sizeof pSB));

  cusparseLtMatmulAlgSelection_t alg;
  SP(cusparseLtMatmulAlgSelectionInit(&h, &alg, &mm, CUSPARSELT_MATMUL_ALG_DEFAULT));
  cusparseLtMatmulPlan_t plan;
  SP(cusparseLtMatmulPlanInit(&h, &plan, &mm, &alg));

  size_t cSize = 0, cBuf = 0;
  SP(cusparseLtSpMMACompressedSize(&h, &plan, &cSize, &cBuf));
  printf("{\"compressed_bytes\":%zu,\"compress_buf_bytes\":%zu}\n", cSize, cBuf);
  uint8_t *dAc, *dAcBuf;
  CK(cudaMalloc(&dAc, cSize));
  CK(cudaMalloc(&dAcBuf, cBuf ? cBuf : 1));

  // Validate the structure, then compress (no prune: the matrix is already 2:4 by construction).
  int valid = -1, *dValid;
  CK(cudaMalloc(&dValid, sizeof(int)));
  SP(cusparseLtSpMMAPruneCheck2(&h, &mA, 1, CUSPARSE_OPERATION_NON_TRANSPOSE, dA, dValid, nullptr));
  CK(cudaMemcpy(&valid, dValid, sizeof(int), cudaMemcpyDeviceToHost));
  printf("{\"prune_check_valid(0=ok)\":%d}\n", valid);
  SP(cusparseLtSpMMACompress(&h, &plan, dA, dAc, dAcBuf, nullptr));
  CK(cudaDeviceSynchronize());

  size_t wsSize = 0;
  SP(cusparseLtMatmulGetWorkspace(&h, &plan, &wsSize));
  void* dWs;
  CK(cudaMalloc(&dWs, wsSize ? wsSize : 1));

  float alpha = 1.f, beta = 0.f;
  cudaStream_t st = nullptr;
  CK(cudaMemset(dD, 0xA5, size_t(M) * N * sizeof(float)));
  cusparseStatus_t mret = cusparseLtMatmul(&h, &plan, &alpha, dAc, dB, &beta, dD, dD, dWs, &st, 1);
  printf("{\"matmul_status(0=ok)\":%d,\"name\":\"%s\"}\n", int(mret), cusparseLtGetErrorName(mret));
  CK(cudaDeviceSynchronize());

  std::vector<float> hD(size_t(M) * N);
  CK(cudaMemcpy(hD.data(), dD, size_t(M) * N * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU int64 reference D = A·B row-major; report mismatches and a sample.
  size_t bad = 0;
  int64_t sample_cpu = 0, sample_gpu = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int64_t s = 0;
      for (int kk = 0; kk < K; ++kk) s += int64_t(hA[size_t(i) * K + kk]) * hB[size_t(kk) * N + j];
      if (float(s) != hD[size_t(i) * N + j]) {
        if (bad < 4)
          fprintf(stderr, "mismatch (%d,%d) cpu=%lld gpu=%.1f\n", i, j, (long long)s, hD[size_t(i) * N + j]);
        ++bad;
      }
      if (i == 0 && j == 0) {
        sample_cpu = s;
        sample_gpu = int64_t(hD[0]);
      }
    }
  }
  printf("{\"n\":%d,\"checked\":%zu,\"mismatches\":%zu,\"sample00_cpu\":%lld,\"sample00_gpu\":%lld,\"PASS\":%s}\n", n,
         size_t(M) * N, bad, (long long)sample_cpu, (long long)sample_gpu, bad ? "false" : "true");

  cusparseLtMatmulPlanDestroy(&plan);
  cusparseLtMatDescriptorDestroy(&mA);
  cusparseLtMatDescriptorDestroy(&mB);
  cusparseLtMatDescriptorDestroy(&mC);
  cusparseLtDestroy(&h);
  return bad ? 1 : 0;
}
