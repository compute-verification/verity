// BS-CSA: the guarded bit-sliced carry-save counter (csa_sim.py), on sm_120.  For X, Y in {+-1}^{MxK} x {+-1}^{KxN},
// compute C = X*Y (int32) exactly.
//
// x = (-1)^a, y = (-1)^b; C_ij = K - 2*#{k: a_ik != b_kj}.  Y's sign bits are packed as bit-planes: a lane word holds
// b_kj of 32 output columns (one plane per k).  A warp owns one output row i (so a_ik is warp-uniform); each of its 32
// lanes owns one 32-column plane word, so a warp covers 1024 columns.  Per row a Harley-Seal carry-save counter runs
// over k; a warp-uniform branch on (a_ik1, a_ik2) picks the first-level full adder's LOP3 immediates, so the XOR is
// free (2 LOP3 a raw term-plane).  The counter is reduced per CHUNK_TERMS-chunk (bs_csa_chunk.cuh, generated and
// verified bit-exact) and ripple-added into a persistent binary counter P[], converted to int once at the end.
//
// Timed against cuBLASLt int8 (IMMA) producing the same C, interleaved best-of; gated bit-exact against a CPU
// reference.  Reports end-to-end ms + ratio, and a clock64 hot-loop MAC/SM/clk.  SASS (cuobjdump) confirms the LOP3
// mix and the warp-uniform branch.
#include "common.cuh"
#include "bs_csa_chunk.cuh"

#include <cublasLt.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CKB(x)                                                                                   \
  do {                                                                                           \
    cublasStatus_t s_ = (x);                                                                     \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                           \
      fprintf(stderr, "CUBLAS %s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)s_);                 \
      exit(3);                                                                                   \
    }                                                                                            \
  } while (0)

// ---- BS-CSA kernel ----------------------------------------------------------------------------------------------
// A block owns WARPS rows x 1024 columns (32 column-blocks).  A warp owns one row; a lane owns one 32-column plane.
constexpr int WARPS = 8;
constexpr int CB = 32;                 // column-blocks a block (= 32 lanes); 1024 columns
constexpr int CT = BS_CSA_CHUNK_TERMS; // terms a chunk
constexpr int LP = BS_CSA_LEVELS_P;

__global__ void __launch_bounds__(WARPS * 32)
    bs_csa_gemm(const int8_t* __restrict__ X, const uint32_t* __restrict__ YP, int32_t* __restrict__ C, int M, int N,
                int K, long long* __restrict__ dbg) {
  __shared__ uint32_t Ys[CT * CB];  // lane-interleaved: Ys[t*CB + cb] (bank-conflict-free lane reads)
  __shared__ int8_t Xs[WARPS * CT]; // per-warp row: Xs[warp*CT + t]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int row = blockIdx.y * WARPS + warp;
  const int cb0 = blockIdx.x * CB;  // first column-block of this block
  const int NB = N >> 5;            // column-blocks a row

  uint32_t P[LP];
#pragma unroll
  for (int w = 0; w < LP; w++) P[w] = 0u;

  const bool clk = (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0);
  long long c0 = 0, g0 = 0;
  if (clk) { c0 = clock64(); g0 = gtimer(); }

  for (int base = 0; base < K; base += CT) {
#pragma unroll 1
    for (int i = tid; i < CT * CB; i += WARPS * 32) {
      int t = i / CB, cb = i % CB;  // Ys[t*CB + cb]
      Ys[i] = YP[(size_t)(base + t) * NB + (cb0 + cb)];
    }
#pragma unroll 1
    for (int i = tid; i < WARPS * CT; i += WARPS * 32) {
      int w = i / CT, t = i % CT;
      Xs[i] = X[(size_t)(blockIdx.y * WARPS + w) * K + base + t];
    }
    __syncthreads();
    bs_csa_chunk(P, &Ys[lane], &Xs[warp * CT]);  // yl = &Ys[lane]; term t at yl[t*CB]
    __syncthreads();
  }
  if (clk) { dbg[0] = clock64() - c0; dbg[1] = gtimer() - g0; }

  // convert P (per-column count bits) to int, write C_ij = K - 2*count
  const int colbase = (cb0 + lane) * 32;
#pragma unroll
  for (int j = 0; j < 32; j++) {
    int cnt = 0;
#pragma unroll
    for (int w = 0; w < LP; w++) cnt += (int)((P[w] >> j) & 1u) << w;
    C[(size_t)row * N + colbase + j] = K - 2 * cnt;
  }
}

// pack Y (row-major K x N, +-1) into bit-planes YP[k*NB + cb], bit j = (Y[k, cb*32 + j] < 0); untimed marshalling
__global__ void pack_y_planes(const int8_t* __restrict__ Y, uint32_t* __restrict__ YP, int K, int N) {
  int NB = N >> 5;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * NB) return;
  int k = idx / NB, cb = idx % NB;
  uint32_t word = 0;
#pragma unroll
  for (int j = 0; j < 32; j++)
    if (Y[(size_t)k * N + cb * 32 + j] < 0) word |= (1u << j);
  YP[idx] = word;
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
static void cpu_ref(const std::vector<int8_t>& X, const std::vector<int8_t>& Y, std::vector<int32_t>& C, int M, int N,
                    int K) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      long acc = 0;
      for (int k = 0; k < K; k++) acc += (int)X[(size_t)i * K + k] * (int)Y[(size_t)k * N + j];
      C[(size_t)i * N + j] = (int32_t)acc;
    }
}

int main(int argc, char** argv) {
  std::vector<std::array<int, 3>> shapes = {{256, 256, 256}, {1024, 1024, 1024}, {4096, 4096, 4096}};
  if (argc > 3) shapes = {{atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}};
  const int BM = WARPS, BN = CB * 32;  // rows / columns a block
  int nsm = num_sms();

  for (size_t si = 0; si < shapes.size(); si++) {
    int M = shapes[si][0], N = shapes[si][1], K = shapes[si][2];
    if (M % BM || N % BN || K % CT) { fprintf(stderr, "shape %dx%dx%d not tile-aligned\n", M, N, K); continue; }
    std::vector<int8_t> hX((size_t)M * K), hY((size_t)K * N);
    uint64_t seed = 7 + si;
    for (auto& v : hX) v = (splitmix64(seed) & 1) ? 1 : -1;
    for (auto& v : hY) v = (splitmix64(seed) & 1) ? 1 : -1;

    int8_t *dX, *dY, *dYt;
    uint32_t* dYP;
    int32_t *dCb, *dClt;
    long long* dDbg;
    CK(cudaMalloc(&dX, (size_t)M * K));
    CK(cudaMalloc(&dY, (size_t)K * N));
    CK(cudaMalloc(&dYt, (size_t)K * N));
    CK(cudaMalloc(&dYP, (size_t)K * (N / 32) * 4));
    CK(cudaMalloc(&dCb, (size_t)M * N * 4));
    CK(cudaMalloc(&dClt, (size_t)M * N * 4));
    CK(cudaMalloc(&dDbg, 2 * sizeof(long long)));
    CK(cudaMemcpy(dX, hX.data(), hX.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dY, hY.data(), hY.size(), cudaMemcpyHostToDevice));

    pack_y_planes<<<(K * (N / 32) + 255) / 256, 256>>>(dY, dYP, K, N);
    y_to_colmajor<<<(K * N + 255) / 256, 256>>>(dY, dYt, K, N);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    LtGemm g;
    lt_init(g, M, N, K);

    dim3 grid(N / BN, M / BM), block(WARPS * 32);
    auto run_bs = [&] { bs_csa_gemm<<<grid, block>>>(dX, dYP, dCb, M, N, K, dDbg); };
    auto run_lt = [&] { lt_run(g, dX, dYt, dClt); };

    bool exactb = true, exactlt = true;
    if (M <= 1024) {
      std::vector<int32_t> ref((size_t)M * N), hb((size_t)M * N), hlt((size_t)M * N);
      cpu_ref(hX, hY, ref, M, N, K);
      run_bs(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
      run_lt(); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hb.data(), dCb, ref.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hlt.data(), dClt, ref.size() * 4, cudaMemcpyDeviceToHost));
      for (int i = 0; i < M && exactb; i++)
        for (int j = 0; j < N; j++)
          if (hb[(size_t)i * N + j] != ref[(size_t)i * N + j]) { exactb = false; break; }
      for (int i = 0; i < M && exactlt; i++)
        for (int j = 0; j < N; j++)
          if (hlt[(size_t)j * M + i] != ref[(size_t)i * N + j]) { exactlt = false; break; }
      printf("{\"probe\":\"bs_csa\",\"mode\":\"gate\",\"shape\":\"%dx%dx%d\",\"bs_csa_exact\":%s,\"int8_exact\":%s}\n",
             M, N, K, exactb ? "true" : "false", exactlt ? "true" : "false");
      fflush(stdout);
    }

    // interleaved best-of timing: warm up, then rep-by-rep BS-CSA then int8
    best_ms(run_bs, 1); best_ms(run_lt, 1);
    double bb = 1e300, blt = 1e300;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int r = 0; r < 20; r++) {
      CK(cudaEventRecord(e0)); run_bs(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float t = 0; CK(cudaEventElapsedTime(&t, e0, e1)); bb = std::min(bb, (double)t);
      CK(cudaEventRecord(e0)); run_lt(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      CK(cudaEventElapsedTime(&t, e0, e1)); blt = std::min(blt, (double)t);
    }
    // hot-loop clock: block 0's chunk-loop cycles and the SM clock it ran at
    long long hdbg[2] = {0, 0};
    CK(cudaMemcpy(hdbg, dDbg, sizeof(hdbg), cudaMemcpyDeviceToHost));
    double sm_hz = hdbg[1] > 0 ? (double)hdbg[0] / ((double)hdbg[1] * 1e-9) : 0.0;
    double gmacb = (double)M * N * K / bb / 1e6, gmaclt = (double)M * N * K / blt / 1e6;
    // achieved MAC/SM/clk end-to-end (compute-bound at large K): total MACs / (SMs * wall_cycles)
    double macsmclk_b = sm_hz > 0 ? (double)M * N * K / (nsm * (bb * 1e-3) * sm_hz) : 0.0;
    double macsmclk_lt = sm_hz > 0 ? (double)M * N * K / (nsm * (blt * 1e-3) * sm_hz) : 0.0;

    printf("{\"probe\":\"bs_csa\",\"mode\":\"result\",\"shape\":\"%dx%dx%d\",\"kernel\":\"bs_csa\",\"chunk\":%d,"
           "\"ms\":%.5f,\"gmac_s\":%.1f,\"exact\":%s,\"ratio_vs_int8\":%.3f,\"sm_hz\":%.3e,"
           "\"mac_sm_clk\":%.1f,\"hot_cyc\":%lld}\n",
           M, N, K, CT, bb, gmacb, exactb ? "true" : "false", bb / blt, sm_hz, macsmclk_b, hdbg[0]);
    printf("{\"probe\":\"bs_csa\",\"mode\":\"result\",\"shape\":\"%dx%dx%d\",\"kernel\":\"cublaslt_int8\",\"chunk\":0,"
           "\"ms\":%.5f,\"gmac_s\":%.1f,\"exact\":%s,\"ratio_vs_int8\":1.0,\"sm_hz\":%.3e,\"mac_sm_clk\":%.1f}\n",
           M, N, K, blt, gmaclt, exactlt ? "true" : "false", sm_hz, macsmclk_lt);
    fflush(stdout);

    cudaFree(dX); cudaFree(dY); cudaFree(dYt); cudaFree(dYP); cudaFree(dCb); cudaFree(dClt); cudaFree(dDbg);
    cudaFree(g.ws);
  }
  printf("{\"probe\":\"bs_csa\",\"done\":true}\n");
  return 0;
}
