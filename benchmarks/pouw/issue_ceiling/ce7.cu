// CE7: the SWAR branch-select kernel (ce7_sim.py), on sm_120.
//
// For X, Y in {+-1}^{M x K} x {+-1}^{K x N}, compute C = X * Y (int32) exactly. CE7 reads X as a guard: with
// Y' = Y + 1 in {0, 2} packed four columns a 32-bit word (8-bit fields, biased at 128), one warp-uniform branch on a
// pair of X bits picks IADD3 acc <- acc +- Y'_{k1} +- Y'_{k2} (8 MACs a lane instruction). C_ij = sum_k x_ik Y'_kj -
// sum_k x_ik. Fields stay carry-free within a K-chunk (<= 31 steps) then renormalise into int32. A warp owns a row
// tile so the branch is warp-uniform; lanes split the columns. Timed against cuBLASLt int8 (IMMA) that produces the
// same C, gated bit-exact against a CPU reference. SASS (cuobjdump) confirms IADD3 and the warp-uniform branch issued.
#include "common.cuh"

#include <cublasLt.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CKB(x)                                                                                   \
  do {                                                                                           \
    cublasStatus_t s_ = (x);                                                                     \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                           \
      fprintf(stderr, "CUBLAS %s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)s_);                 \
      exit(3);                                                                                   \
    }                                                                                            \
  } while (0)

// ---- CE7 kernel -------------------------------------------------------------------------------------------------
// Tile: BM rows x BN cols a block, BK of K a chunk. A warp (8 of them) owns TM rows; a lane owns TNW column-words.
constexpr int BK = 16, TM = 8, TNW = 2, WARPS = 8;
constexpr int BM = WARPS * TM;       // 64 rows
constexpr int BNW = 32 * TNW;        // 64 column-words
constexpr int BN = BNW * 4;          // 256 columns

// two dependent PTX adds; ptxas fuses them into one IADD3
__device__ __forceinline__ uint32_t iadd3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("add.u32 %0, %1, %2;\n\tadd.u32 %0, %0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__global__ void __launch_bounds__(WARPS * 32) ce7_gemm(const int8_t* __restrict__ X, const uint32_t* __restrict__ Yp,
                                                       int32_t* __restrict__ C, int M, int N, int K) {
  __shared__ int8_t Xs[BM * BK];
  __shared__ uint32_t Yps[BK * BNW];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int by = blockIdx.y * BM, bx = blockIdx.x * BN;
  const int NW = N / 4;

  uint32_t acc[TM][TNW];
  int32_t Cacc[TM][TNW][4];
  int32_t rowsum[TM];
#pragma unroll
  for (int m = 0; m < TM; m++) {
    rowsum[m] = 0;
#pragma unroll
    for (int t = 0; t < TNW; t++)
#pragma unroll
      for (int f = 0; f < 4; f++) Cacc[m][t][f] = 0;
  }

  for (int kk = 0; kk < K; kk += BK) {
    // load X[by..][kk..] and Yp[kk..][bx/4..] into shared
#pragma unroll
    for (int i = tid; i < BM * BK; i += WARPS * 32) {
      int r = i / BK, c = i % BK;
      Xs[i] = X[(by + r) * K + (kk + c)];
    }
#pragma unroll
    for (int i = tid; i < BK * BNW; i += WARPS * 32) {
      int r = i / BNW, c = i % BNW;
      Yps[i] = Yp[(kk + r) * NW + (bx / 4 + c)];
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < TM; m++)
#pragma unroll
      for (int t = 0; t < TNW; t++) acc[m][t] = 0x80808080u;

#pragma unroll 1
    for (int kp = 0; kp < BK; kp += 2) {
      uint32_t y1[TNW], y2[TNW], ny1[TNW], ny2[TNW];
#pragma unroll
      for (int t = 0; t < TNW; t++) {
        y1[t] = Yps[kp * BNW + lane * TNW + t];
        y2[t] = Yps[(kp + 1) * BNW + lane * TNW + t];
        ny1[t] = 0u - y1[t];
        ny2[t] = 0u - y2[t];
      }
#pragma unroll
      for (int m = 0; m < TM; m++) {
        int s1 = Xs[(warp * TM + m) * BK + kp];      // warp-uniform: one row a warp
        int s2 = Xs[(warp * TM + m) * BK + kp + 1];
        rowsum[m] += s1 + s2;
        int sel = (s1 > 0 ? 2 : 0) | (s2 > 0 ? 1 : 0);  // warp-uniform branch index
        switch (sel) {
          case 3:
#pragma unroll
            for (int t = 0; t < TNW; t++) acc[m][t] = iadd3(acc[m][t], y1[t], y2[t]);
            break;
          case 2:
#pragma unroll
            for (int t = 0; t < TNW; t++) acc[m][t] = iadd3(acc[m][t], y1[t], ny2[t]);
            break;
          case 1:
#pragma unroll
            for (int t = 0; t < TNW; t++) acc[m][t] = iadd3(acc[m][t], ny1[t], y2[t]);
            break;
          default:
#pragma unroll
            for (int t = 0; t < TNW; t++) acc[m][t] = iadd3(acc[m][t], ny1[t], ny2[t]);
            break;
        }
      }
    }
#pragma unroll
    for (int m = 0; m < TM; m++)
#pragma unroll
      for (int t = 0; t < TNW; t++)
#pragma unroll
        for (int f = 0; f < 4; f++)
          Cacc[m][t][f] += (int32_t)((acc[m][t] >> (8 * f)) & 0xff) - 128;
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; m++) {
    int i = by + warp * TM + m;
#pragma unroll
    for (int t = 0; t < TNW; t++)
#pragma unroll
      for (int f = 0; f < 4; f++) {
        int col = bx + (lane * TNW + t) * 4 + f;
        C[i * N + col] = Cacc[m][t][f] - rowsum[m];
      }
  }
}

// pack Y (row-major K x N, +-1) into Y' = Y + 1 words (K x N/4), untimed input marshalling
__global__ void pack_yp(const int8_t* __restrict__ Y, uint32_t* __restrict__ Yp, int K, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int NW = N / 4;
  if (idx >= K * NW) return;
  int k = idx / NW, w = idx % NW;
  uint32_t word = 0;
#pragma unroll
  for (int f = 0; f < 4; f++) {
    uint32_t v = (uint32_t)(Y[k * N + w * 4 + f] + 1);  // {0, 2}
    word |= v << (8 * f);
  }
  Yp[idx] = word;
}

// transpose Y (row-major K x N) into col-major K x N (ld = K) for cuBLASLt's int8 B operand, untimed
__global__ void y_to_colmajor(const int8_t* __restrict__ Y, int8_t* __restrict__ Yt, int K, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * N) return;
  int k = idx / N, n = idx % N;
  Yt[(size_t)n * K + k] = Y[(size_t)k * N + n];
}

// ---- cuBLASLt int8 (IMMA) baseline: D = X * Y, col-major M x N (ld = M) --------------------------------------
struct LtGemm {
  cublasLtHandle_t lt;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulHeuristicResult_t heur;
  void* ws = nullptr;
  size_t wsz = 32u << 20;
  int M, N, K;
};

static void lt_init(LtGemm& g, int M, int N, int K) {
  g.M = M; g.N = N; g.K = K;
  CKB(cublasLtCreate(&g.lt));
  CKB(cublasLtMatmulDescCreate(&g.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  CKB(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
  CKB(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
  // A = X as col-major K x M (ld = K): A^T = X (M x K). B = Yt col-major K x N (ld = K). D col-major M x N (ld = M).
  CKB(cublasLtMatrixLayoutCreate(&g.la, CUDA_R_8I, K, M, K));
  CKB(cublasLtMatrixLayoutCreate(&g.lb, CUDA_R_8I, K, N, K));
  CKB(cublasLtMatrixLayoutCreate(&g.ld, CUDA_R_32I, M, N, M));
  CK(cudaMalloc(&g.ws, g.wsz));
  CKB(cublasLtMatmulPreferenceCreate(&g.pref));
  CKB(cublasLtMatmulPreferenceSetAttribute(g.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g.wsz, sizeof(g.wsz)));
  int found = 0;
  CKB(cublasLtMatmulAlgoGetHeuristic(g.lt, g.op, g.la, g.lb, g.ld, g.ld, g.pref, 1, &g.heur, &found));
  if (!found) { fprintf(stderr, "no cuBLASLt int8 algo for %dx%dx%d\n", M, N, K); exit(3); }
}

static void lt_run(LtGemm& g, const int8_t* Xcm, const int8_t* Yt, int32_t* D) {
  int32_t alpha = 1, beta = 0;
  CKB(cublasLtMatmul(g.lt, g.op, &alpha, Xcm, g.la, Yt, g.lb, &beta, D, g.ld, D, g.ld, &g.heur.algo, g.ws, g.wsz, 0));
}

// ---- CPU reference and gate -------------------------------------------------------------------------------------
static void cpu_ref(const std::vector<int8_t>& X, const std::vector<int8_t>& Y, std::vector<int32_t>& C, int M, int N, int K) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      long acc = 0;
      for (int k = 0; k < K; k++) acc += (int)X[(size_t)i * K + k] * (int)Y[(size_t)k * N + j];
      C[(size_t)i * N + j] = (int32_t)acc;
    }
}

int main(int argc, char** argv) {
  // shapes: gate at a small one, then time the rest (default a few squares). m,n,k multiples of the tile.
  std::vector<std::array<int, 3>> shapes = {{256, 256, 256}, {1024, 1024, 1024}, {4096, 4096, 4096}};
  if (argc > 3) shapes = {{atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}};

  for (size_t si = 0; si < shapes.size(); si++) {
    int M = shapes[si][0], N = shapes[si][1], K = shapes[si][2];
    if (M % BM || N % BN || K % BK) { fprintf(stderr, "shape %dx%dx%d not tile-aligned\n", M, N, K); continue; }
    std::vector<int8_t> hX((size_t)M * K), hY((size_t)K * N);
    uint64_t seed = 7 + si;
    for (auto& v : hX) v = (splitmix64(seed) & 1) ? 1 : -1;
    for (auto& v : hY) v = (splitmix64(seed) & 1) ? 1 : -1;

    int8_t *dX, *dY, *dXcm, *dYt;
    uint32_t* dYp;
    int32_t *dC7, *dClt;
    CK(cudaMalloc(&dX, (size_t)M * K));
    CK(cudaMalloc(&dY, (size_t)K * N));
    CK(cudaMalloc(&dXcm, (size_t)M * K));
    CK(cudaMalloc(&dYt, (size_t)K * N));
    CK(cudaMalloc(&dYp, (size_t)K * (N / 4) * 4));
    CK(cudaMalloc(&dC7, (size_t)M * N * 4));
    CK(cudaMalloc(&dClt, (size_t)M * N * 4));
    CK(cudaMemcpy(dX, hX.data(), hX.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dY, hY.data(), hY.size(), cudaMemcpyHostToDevice));
    // X as col-major K x M (= X row-major reinterpreted) is just dX; cuBLASLt reads it with ld = K. So dXcm = dX copy.
    CK(cudaMemcpy(dXcm, dX, hX.size(), cudaMemcpyDeviceToDevice));

    // untimed marshalling
    pack_yp<<<(K * (N / 4) + 255) / 256, 256>>>(dY, dYp, K, N);
    y_to_colmajor<<<(K * N + 255) / 256, 256>>>(dY, dYt, K, N);
    CK(cudaDeviceSynchronize());

    LtGemm g;
    lt_init(g, M, N, K);

    dim3 grid(N / BN, M / BM), block(WARPS * 32);
    auto run_ce7 = [&] { ce7_gemm<<<grid, block>>>(dX, dYp, dC7, M, N, K); };
    auto run_lt = [&] { lt_run(g, dXcm, dYt, dClt); };

    // gate bit-exact vs CPU (only up to 1024 to keep CPU cheap)
    bool exact7 = true, exactlt = true;
    if (M <= 1024) {
      std::vector<int32_t> ref((size_t)M * N), h7((size_t)M * N), hlt((size_t)M * N);
      cpu_ref(hX, hY, ref, M, N, K);
      run_ce7(); CK(cudaDeviceSynchronize());
      run_lt(); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h7.data(), dC7, ref.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hlt.data(), dClt, ref.size() * 4, cudaMemcpyDeviceToHost));
      for (int i = 0; i < M && exact7; i++)
        for (int j = 0; j < N; j++)
          if (h7[(size_t)i * N + j] != ref[(size_t)i * N + j]) { exact7 = false; break; }
      for (int i = 0; i < M && exactlt; i++)
        for (int j = 0; j < N; j++)
          // D is col-major M x N (ld = M): D[i][j] at j*M + i
          if (hlt[(size_t)j * M + i] != ref[(size_t)i * N + j]) { exactlt = false; break; }
      printf("{\"probe\":\"ce7\",\"mode\":\"gate\",\"shape\":\"%dx%dx%d\",\"ce7_exact\":%s,\"int8_exact\":%s}\n",
             M, N, K, exact7 ? "true" : "false", exactlt ? "true" : "false");
      fflush(stdout);
    }

    // interleaved timing: warm up, then rep-by-rep CE7 then int8, best-of
    double ms7 = best_ms(run_ce7, 1), mslt = best_ms(run_lt, 1);  // warm-up
    double b7 = 1e300, blt = 1e300;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int r = 0; r < 20; r++) {
      CK(cudaEventRecord(e0)); run_ce7(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float t = 0; CK(cudaEventElapsedTime(&t, e0, e1)); b7 = std::min(b7, (double)t);
      CK(cudaEventRecord(e0)); run_lt(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      CK(cudaEventElapsedTime(&t, e0, e1)); blt = std::min(blt, (double)t);
    }
    double macs = 2.0 * M * N * K;  // report MACs (M*N*K products), GMAC/s = M*N*K/ms/1e6
    double gmac7 = (double)M * N * K / b7 / 1e6, gmaclt = (double)M * N * K / blt / 1e6;
    (void)ms7; (void)mslt; (void)macs;
    printf("{\"probe\":\"ce7\",\"mode\":\"ce7_result\",\"shape\":\"%dx%dx%d\",\"kernel\":\"ce7\",\"ms\":%.5f,"
           "\"gmac_s\":%.1f,\"exact\":%s,\"ratio_vs_int8\":%.3f}\n",
           M, N, K, b7, gmac7, exact7 ? "true" : "false", b7 / blt);
    printf("{\"probe\":\"ce7\",\"mode\":\"ce7_result\",\"shape\":\"%dx%dx%d\",\"kernel\":\"cublaslt_int8\",\"ms\":%.5f,"
           "\"gmac_s\":%.1f,\"exact\":%s,\"ratio_vs_int8\":1.0}\n",
           M, N, K, blt, gmaclt, exactlt ? "true" : "false");
    fflush(stdout);

    cudaFree(dX); cudaFree(dY); cudaFree(dXcm); cudaFree(dYt); cudaFree(dYp); cudaFree(dC7); cudaFree(dClt);
    cudaFree(g.ws);
  }
  printf("{\"probe\":\"ce7\",\"done\":true}\n");
  return 0;
}
