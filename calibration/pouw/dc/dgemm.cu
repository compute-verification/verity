// The honest Dither GEMM with its duo fold on Hopper (sm_90a, wgmma) and datacenter Blackwell (sm_100a, tcgen05).
//
// C = X W^T with X (m x k) the dithered served operand and W (n x k) the edited weights, both s8 codes (the forming
// kernels, quant and wedit, are sm_120's edited_gemm ones: CUDA-core code, not ported here). Every 32 steps of k the s32
// running sum T_s = X[:, :32 s] W[:, :32 s]^T is committed: each of its words is absorbed by duo (3 ops a word; dc.cuh).
//
// The receipt (one definition for both kernels, so one reference checks both): output tile (tm, tn) is BM x BN; chain
// (i, tn, c), c in 0..15, owns the words of row i whose column j in the tile has j % 16 == c, starts at
// duo_init(seed, (i tiles_n + tn) 16 + c), and absorbs at each boundary s = 1..k/32, in ascending j, word T_s[i, j] at key
// position t = (j / 16) % 16. Its receipt word R[(i tiles_n + tn) 16 + c] is duo_finish. dgemm_ref.py is the twin.
//
// Kernels (each a correctness-first single-stage mainloop: the tile is loaded, fenced, multiplied and folded a step at
// a time; the pipelined TMA version is the next attempt, and its receipts must equal these):
//   dg90<BN>   sm_90a: one warpgroup a 64 x BN tile, wgmma m64nBNk32 s8 (accumulators in registers), 8 chains a thread
//   dg100      sm_100a: four warps a 128 x 256 tile, tcgen05.mma kind::i8 128x256x32 into TMEM, tcgen05.ld.32x32b.x16
//              to registers, 16 chains a thread (its row); C leaves TMEM in one more pass after the last step
// FOLD = false is the same kernel without receipts (the hash-free arm). WRITE = false stores nothing (the control).
//
// Modes:
//   dgemm cpuref DIR M N K SEED        host reference only (no GPU): X, W, C and R to DIR for dgemm_ref.py
//   dgemm gate DIR M N K SEED          poison C and R with 0xA5, run the fold kernel, compare with the host reference,
//                                      run the no-write control (must be rejected); X, W, C, R and meta.json to DIR. A
//                                      mismatch is rerun with LBO and SBO exchanged and reported (lbo_sbo_swapped)
//   dgemm bench REPS M N K             fold, no-fold and (with -DDG_CUBLASLT) cuBLASLt int8 interleaved, one JSON line
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a dgemm.cu -o dgemm_sm90   (or 100a; build.sh)
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dc.cuh"
#include "wgmma.cuh"
#ifdef DG_CUBLASLT
#include <cublasLt.h>
#endif

#define CK(x)                                                                                          \
  do {                                                                                                 \
    cudaError_t e_ = (x);                                                                              \
    if (e_ != cudaSuccess) {                                                                           \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));           \
      exit(2);                                                                                         \
    }                                                                                                  \
  } while (0)

constexpr int L = 32;     // commit spacing: one wgmma / tcgen05.mma of K = 32
constexpr int NCH = 16;   // chains a (row, tile): column mod 16

// Copies a rows x 32-byte K slice (row-major source with row stride k) into the K-major layout. All threads.
DC_DEV void load_slice(unsigned char* dst, const int8_t* src, int rows, int k, int k0) {
  for (int c = threadIdx.x; c < rows * 2; c += blockDim.x) {
    const int r = c >> 1, kc = c & 1;
    const uint4 v = *reinterpret_cast<const uint4*>(src + (size_t)r * k + k0 + 16 * kc);
    *reinterpret_cast<uint4*>(dst + kmaj_off(r, 16 * kc, L)) = v;
  }
}

// ---- Hopper: wgmma, accumulators in registers ----------------------------------------------------------------------------
// Thread (warp w, lane = 4 g + q) holds D[16 w + g + 8 (e >> 1)][8 ni + 2 q + (e & 1)] in d[4 ni + e] (wgmma's D fragment).
// Its chains: h = 4 (e >> 1) + 2 (ni % 2) + (e & 1), i.e. row 16 w + g + 8 (h >> 2), c = 8 ((h >> 1) & 1) + 2 q + (h & 1).
template <int NI, int END>
DC_DEV void absorb90(DuoSt (&ch)[8], const int* d) {
  if constexpr (NI < END) {
#pragma unroll
    for (int e = 0; e < 4; e++) duo<(NI / 2) % 16>(ch[4 * (e >> 1) + 2 * (NI % 2) + (e & 1)], (uint32_t)d[4 * NI + e]);
    absorb90<NI + 1, END>(ch, d);
  }
}

template <int BN, bool FOLD, bool WRITE>
__global__ void __launch_bounds__(128) dg90(const int8_t* X, const int8_t* W, int m, int n, int k, uint32_t seed,
                                            int32_t* C, uint32_t* R, uint32_t lbo, uint32_t sbo) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __shared__ __align__(1024) unsigned char sA[64 * L];
  __shared__ __align__(1024) unsigned char sB[BN * L];
  const int tn = blockIdx.x, tm = blockIdx.y, tiles_n = gridDim.x;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const uint64_t da = wgmma_desc(smem_u32(sA), lbo, sbo);
  const uint64_t db = wgmma_desc(smem_u32(sB), lbo, sbo);
  int d[BN / 2];
#pragma unroll
  for (int i = 0; i < BN / 2; i++) d[i] = 0;
  DuoSt ch[8];
  auto row_of = [&](int h) { return tm * 64 + 16 * w + g + 8 * (h >> 2); };
  auto col_of = [&](int h) { return 8 * ((h >> 1) & 1) + 2 * q + (h & 1); };
  if constexpr (FOLD) {
#pragma unroll
    for (int h = 0; h < 8; h++) ch[h] = duo_init(seed, (uint32_t)((row_of(h) * tiles_n + tn) * NCH + col_of(h)));
  }
  for (int s = 0; s < k / L; s++) {
    load_slice(sA, X + (size_t)tm * 64 * k, 64, k, s * L);
    load_slice(sB, W + (size_t)tn * BN * k, BN, k, s * L);
    fence_async_smem();
    __syncthreads();
    wgmma_fence();
    if constexpr (BN == 128) wgmma_s8_n128(d, da, db, s > 0);
    else wgmma_s8_n256(d, da, db, s > 0);
    wgmma_commit();
    wgmma_wait<0>();
    wgmma_hold(d);
    if constexpr (FOLD) absorb90<0, BN / 8>(ch, d);
    __syncthreads();
  }
  if constexpr (WRITE) {
#pragma unroll
    for (int ni = 0; ni < BN / 8; ni++)
#pragma unroll
      for (int e = 0; e < 4; e++)
        C[(size_t)(tm * 64 + 16 * w + g + 8 * (e >> 1)) * n + tn * BN + 8 * ni + 2 * q + (e & 1)] = d[4 * ni + e];
    if constexpr (FOLD) {
#pragma unroll
      for (int h = 0; h < 8; h++) R[(size_t)(row_of(h) * tiles_n + tn) * NCH + col_of(h)] = duo_finish(ch[h]);
    }
  }
#endif
}

// ---- Blackwell: tcgen05.mma into tensor memory, tcgen05.ld to registers ---------------------------------------------------
// Thread (warp w, lane) owns row 32 w + lane of the 128 x 256 tile; chunk u of 16 columns lands in r[0..15] (c = 0..15).
template <int U, int END, bool FOLD, bool STORE>
DC_DEV void absorb100(DuoSt (&ch)[NCH], uint32_t trow, int32_t* crow) {
  if constexpr (U < END) {
    uint32_t r[16];
    tmem_ld16(trow + 16 * U, r);
    tmem_wait_ld();
    if constexpr (FOLD) {
#pragma unroll
      for (int c = 0; c < 16; c++) duo<U % 16>(ch[c], r[c]);
    }
    if constexpr (STORE) {
#pragma unroll
      for (int c = 0; c < 16; c += 4)
        *reinterpret_cast<int4*>(crow + 16 * U + c) = make_int4((int)r[c], (int)r[c + 1], (int)r[c + 2], (int)r[c + 3]);
    }
    absorb100<U + 1, END, FOLD, STORE>(ch, trow, crow);
  }
}

template <bool FOLD, bool WRITE>
__global__ void __launch_bounds__(128) dg100(const int8_t* X, const int8_t* W, int m, int n, int k, uint32_t seed,
                                             int32_t* C, uint32_t* R, uint32_t lbo, uint32_t sbo) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  constexpr int BM = 128, BN = 256;
  __shared__ __align__(1024) unsigned char sA[BM * L];
  __shared__ __align__(1024) unsigned char sB[BN * L];
  __shared__ uint64_t bar;
  __shared__ uint32_t tslot;
  const int tn = blockIdx.x, tm = blockIdx.y, tiles_n = gridDim.x;
  const uint32_t w = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
  const int row = tm * BM + 32 * w + lane;
  if (threadIdx.x == 0) {
    mbar_init(&bar, 1);
    fence_mbar_init();
  }
  if (w == 0) tmem_alloc(&tslot, BN);
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = tslot, trow = tmem + ((32u * w) << 16);
  const uint64_t da = umma_desc(smem_u32(sA), lbo, sbo);
  const uint64_t db = umma_desc(smem_u32(sB), lbo, sbo);
  constexpr uint32_t idesc = umma_idesc(2, 1, 1, BM, BN);
  DuoSt ch[NCH];
  if constexpr (FOLD) {
#pragma unroll
    for (int c = 0; c < NCH; c++) ch[c] = duo_init(seed, (uint32_t)((row * tiles_n + tn) * NCH + c));
  }
  int32_t* crow = C + (size_t)row * n + tn * BN;
  const int steps = k / L;
  for (int s = 0; s < steps; s++) {
    load_slice(sA, X + (size_t)tm * BM * k, BM, k, s * L);
    load_slice(sB, W + (size_t)tn * BN * k, BN, k, s * L);
    fence_async_smem();
    tc_before_sync();
    __syncthreads();
    tc_after_sync();
    if (w == 0) {
      if (elect_one()) {
        umma_i8(tmem, da, db, idesc, s > 0);
        umma_commit(&bar);
      }
      __syncwarp();
    }
    mbar_wait(&bar, (uint32_t)(s & 1));
    tc_after_sync();
    if constexpr (FOLD) absorb100<0, BN / 16, true, false>(ch, trow, crow);
    tc_before_sync();
    __syncthreads();
  }
  tc_after_sync();
  if constexpr (WRITE) {
    absorb100<0, BN / 16, false, true>(ch, trow, crow);
    if constexpr (FOLD) {
#pragma unroll
      for (int c = 0; c < NCH; c++) R[(size_t)(row * tiles_n + tn) * NCH + c] = duo_finish(ch[c]);
    }
  }
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (w == 0) tmem_dealloc(tmem, BN);
#endif
}

// ---- host ------------------------------------------------------------------------------------------------------------------
struct Shape {
  int m, n, k, bm, bn;
};
static int8_t code(uint32_t seed, uint32_t which, uint32_t i) {
  return (int8_t)((int)(dc_mix32(dc_mix32(seed ^ (which << 28)) + i) % 255u) - 127);
}
static void make_inputs(const Shape& sh, uint32_t seed, std::vector<int8_t>& X, std::vector<int8_t>& W) {
  X.resize((size_t)sh.m * sh.k);
  W.resize((size_t)sh.n * sh.k);
  for (size_t i = 0; i < X.size(); i++) X[i] = code(seed, 1, (uint32_t)i);
  for (size_t i = 0; i < W.size(); i++) W[i] = code(seed, 2, (uint32_t)i);
}
// The receipt by its definition (header): no layout, one chain at a time.
static void host_ref(const Shape& sh, uint32_t seed, const std::vector<int8_t>& X, const std::vector<int8_t>& W,
                     std::vector<int32_t>& C, std::vector<uint32_t>& R) {
  const int tiles_n = sh.n / sh.bn;
  C.assign((size_t)sh.m * sh.n, 0);
  R.assign((size_t)sh.m * tiles_n * NCH, 0);
  std::vector<int32_t> T(sh.n);
  for (int i = 0; i < sh.m; i++) {
    std::vector<DuoSt> ch(tiles_n * NCH);
    for (int tn = 0; tn < tiles_n; tn++)
      for (int c = 0; c < NCH; c++) ch[tn * NCH + c] = duo_init(seed, (uint32_t)((i * tiles_n + tn) * NCH + c));
    std::fill(T.begin(), T.end(), 0);
    for (int s = 0; s < sh.k / L; s++) {
      for (int j = 0; j < sh.n; j++) {
        int32_t acc = 0;
        for (int kk = s * L; kk < (s + 1) * L; kk++) acc += (int32_t)X[(size_t)i * sh.k + kk] * W[(size_t)j * sh.k + kk];
        T[j] = (int32_t)((uint32_t)T[j] + (uint32_t)acc);
      }
      for (int tn = 0; tn < tiles_n; tn++)
        for (int jl = 0; jl < sh.bn; jl++) {
          DuoSt& st = ch[tn * NCH + jl % NCH];
          const uint32_t p = (uint32_t)T[tn * sh.bn + jl];
          const int t = (jl / 16) % 16;
          const uint32_t u = st.y ^ p ^ duo_xor(t);
          st.y = u * duo_mul(t) + st.z;
          st.z = dc_rotl(u, duo_rot(t)) + st.y;
        }
    }
    for (int j = 0; j < sh.n; j++) C[(size_t)i * sh.n + j] = T[j];
    for (int q = 0; q < tiles_n * NCH; q++) R[(size_t)i * tiles_n * NCH + q] = duo_finish(ch[q]);
  }
}
template <class T>
static void dump(const std::string& dir, const char* name, const std::vector<T>& v) {
  FILE* f = fopen((dir + "/" + name).c_str(), "wb");
  if (!f) { fprintf(stderr, "ERR open %s/%s\n", dir.c_str(), name); exit(2); }
  fwrite(v.data(), sizeof(T), v.size(), f);
  fclose(f);
}
static void meta(const std::string& dir, const Shape& sh, uint32_t seed, const char* arch, const char* extra) {
  FILE* f = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(f, "{\"m\": %d, \"n\": %d, \"k\": %d, \"bm\": %d, \"bn\": %d, \"seed\": %u, \"L\": %d, \"nch\": %d, \"arch\": \"%s\"%s}\n",
          sh.m, sh.n, sh.k, sh.bm, sh.bn, seed, L, NCH, arch, extra);
  fclose(f);
}

static int g_sm = 0;  // 90 or 100: the binary's kernel family (from the device, or the build for cpuref)
static Shape shape_for(int m, int n, int k) {
  const Shape sh = g_sm == 100 ? Shape{m, n, k, 128, 256} : Shape{m, n, k, 64, 128};
  if (m % sh.bm || n % sh.bn || k % L) {
    fprintf(stderr, "ERR shape %dx%dx%d: m a multiple of %d, n of %d, k of %d\n", m, n, k, sh.bm, sh.bn, L);
    exit(2);
  }
  return sh;
}
template <bool FOLD, bool WRITE>
static void launch(const Shape& sh, const int8_t* X, const int8_t* W, uint32_t seed, int32_t* C, uint32_t* R,
                   cudaStream_t st, bool swap = false) {
  const dim3 grid(sh.n / sh.bn, sh.m / sh.bm);
  const uint32_t lbo = swap ? kmaj_sbo(L) : kmaj_lbo(), sbo = swap ? kmaj_lbo() : kmaj_sbo(L);
  if (g_sm == 100) dg100<FOLD, WRITE><<<grid, 128, 0, st>>>(X, W, sh.m, sh.n, sh.k, seed, C, R, lbo, sbo);
  else dg90<128, FOLD, WRITE><<<grid, 128, 0, st>>>(X, W, sh.m, sh.n, sh.k, seed, C, R, lbo, sbo);
  CK(cudaGetLastError());
}

static int run_gate(const std::string& dir, int m, int n, int k, uint32_t seed) {
  const Shape sh = shape_for(m, n, k);
  std::vector<int8_t> X, W;
  make_inputs(sh, seed, X, W);
  std::vector<int32_t> Cr;
  std::vector<uint32_t> Rr;
  host_ref(sh, seed, X, W, Cr, Rr);
  int8_t *dX, *dW;
  int32_t* dC;
  uint32_t* dR;
  CK(cudaMalloc(&dX, X.size()));
  CK(cudaMalloc(&dW, W.size()));
  CK(cudaMalloc(&dC, Cr.size() * 4));
  CK(cudaMalloc(&dR, Rr.size() * 4));
  CK(cudaMemcpy(dX, X.data(), X.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));
  std::vector<int32_t> C(Cr.size());
  std::vector<uint32_t> R(Rr.size());
  auto call = [&](bool write, bool swap) {
    CK(cudaMemset(dC, 0xA5, Cr.size() * 4));
    CK(cudaMemset(dR, 0xA5, Rr.size() * 4));
    if (write) launch<true, true>(sh, dX, dW, seed, dC, dR, 0, swap);
    else launch<true, false>(sh, dX, dW, seed, dC, dR, 0, swap);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(R.data(), dR, R.size() * 4, cudaMemcpyDeviceToHost));
    size_t bc = 0, br = 0;
    for (size_t i = 0; i < C.size(); i++) bc += C[i] != Cr[i];
    for (size_t i = 0; i < R.size(); i++) br += R[i] != Rr[i];
    return std::make_pair(bc, br);
  };
  const auto ctl = call(false, false);
  const auto got = call(true, false);
  dump(dir, "X.bin", X);
  dump(dir, "W.bin", W);
  dump(dir, "C.bin", C);
  dump(dir, "R.bin", R);
  const bool ok = got.first == 0 && got.second == 0, ctl_rejected = ctl.first > 0 && ctl.second > 0;
  // A diagnosis only: the layout of record stays dc.cuh's until a commit changes it and the gate passes.
  std::string swapped = "null";
  if (!ok) {
    const auto sw = call(true, true);
    swapped = sw.first == 0 && sw.second == 0 ? "\"exact\"" : "\"mismatch\"";
  }
  char extra[256];
  snprintf(extra, sizeof extra, ", \"source\": \"gpu\", \"c_bad\": %zu, \"r_bad\": %zu, \"control_c_bad\": %zu, \"control_r_bad\": %zu",
           got.first, got.second, ctl.first, ctl.second);
  meta(dir, sh, seed, g_sm == 100 ? "sm_100a" : "sm_90a", extra);
  printf("{\"mode\": \"gate\", \"arch\": \"sm_%da\", \"m\": %d, \"n\": %d, \"k\": %d, \"exact\": %s, \"c_bad\": %zu, "
         "\"r_bad\": %zu, \"control_rejected\": %s, \"lbo_sbo_swapped\": %s, \"words\": %zu, \"receipts\": %zu}\n",
         g_sm, m, n, k, ok ? "true" : "false", got.first, got.second, ctl_rejected ? "true" : "false", swapped.c_str(),
         C.size(), R.size());
  return ok && ctl_rejected ? 0 : 4;
}

static int run_cpuref(const std::string& dir, int m, int n, int k, uint32_t seed) {
  const Shape sh = shape_for(m, n, k);
  std::vector<int8_t> X, W;
  make_inputs(sh, seed, X, W);
  std::vector<int32_t> C;
  std::vector<uint32_t> R;
  host_ref(sh, seed, X, W, C, R);
  dump(dir, "X.bin", X);
  dump(dir, "W.bin", W);
  dump(dir, "C.bin", C);
  dump(dir, "R.bin", R);
  meta(dir, sh, seed, g_sm == 100 ? "sm_100a" : "sm_90a", ", \"source\": \"host\"");
  printf("{\"mode\": \"cpuref\", \"arch\": \"sm_%da\", \"m\": %d, \"n\": %d, \"k\": %d, \"words\": %zu, \"receipts\": %zu}\n",
         g_sm, m, n, k, C.size(), R.size());
  return 0;
}

#ifdef DG_CUBLASLT
struct Lt {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
  void* ws;
  size_t wsb = 64 << 20;
  bool ok = false;
  Lt(const Shape& sh) {
    cublasLtCreate(&h);
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    const cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N);
    // column-major view: C^T (n x m) = W (k x n)^T X^T (k x m)
    cublasLtMatrixLayoutCreate(&a, CUDA_R_8I, sh.k, sh.n, sh.k);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_8I, sh.k, sh.m, sh.k);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_32I, sh.n, sh.m, sh.n);
    cudaMalloc(&ws, wsb);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb);
    cublasLtMatmulHeuristicResult_t r;
    int got = 0;
    if (cublasLtMatmulAlgoGetHeuristic(h, op, a, b, c, c, pref, 1, &r, &got) == CUBLAS_STATUS_SUCCESS && got) {
      algo = r.algo;
      ok = true;
    }
  }
  void run(const int8_t* X, const int8_t* W, int32_t* C, cudaStream_t st) {
    const int32_t one = 1, zero = 0;
    cublasLtMatmul(h, op, &one, W, a, X, b, &zero, C, c, C, c, &algo, ws, wsb, st);
  }
};
#endif

static int run_bench(int reps, int m, int n, int k) {
  const Shape sh = shape_for(m, n, k);
  std::vector<int8_t> X, W;
  make_inputs(sh, 7, X, W);
  int8_t *dX, *dW;
  int32_t* dC;
  uint32_t* dR;
  CK(cudaMalloc(&dX, X.size()));
  CK(cudaMalloc(&dW, W.size()));
  CK(cudaMalloc(&dC, (size_t)m * n * 4));
  CK(cudaMalloc(&dR, (size_t)m * (n / sh.bn) * NCH * 4));
  CK(cudaMemcpy(dX, X.data(), X.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<std::string> names = {"fold", "nofold"};
  std::vector<std::vector<float>> ms(3);
#ifdef DG_CUBLASLT
  Lt lt(sh);
  if (lt.ok) names.push_back("cublaslt");
#endif
  auto arm = [&](int a) {
    if (a == 0) launch<true, true>(sh, dX, dW, 1, dC, dR, 0);
    else if (a == 1) launch<false, true>(sh, dX, dW, 1, dC, dR, 0);
#ifdef DG_CUBLASLT
    else lt.run(dX, dW, dC, 0);
#endif
  };
  for (int a = 0; a < (int)names.size(); a++) arm(a);
  CK(cudaDeviceSynchronize());
  for (int r = 0; r < reps; r++)
    for (int a = 0; a < (int)names.size(); a++) {
      CK(cudaEventRecord(e0));
      arm(a);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float t = 0;
      CK(cudaEventElapsedTime(&t, e0, e1));
      ms[a].push_back(t);
    }
  printf("{\"mode\": \"bench\", \"arch\": \"sm_%da\", \"m\": %d, \"n\": %d, \"k\": %d, \"reps\": %d", g_sm, m, n, k, reps);
  for (int a = 0; a < (int)names.size(); a++) {
    std::sort(ms[a].begin(), ms[a].end());
    printf(", \"%s_ms\": %.6f", names[a].c_str(), ms[a][ms[a].size() / 2]);
  }
  printf("}\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: dgemm cpuref|gate DIR M N K SEED | bench REPS M N K\n");
    return 2;
  }
  const std::string mode = argv[1];
  if (mode == "cpuref") {
    if (argc != 8) { fprintf(stderr, "usage: dgemm cpuref DIR M N K SEED 90|100\n"); return 2; }
    g_sm = atoi(argv[7]);
    return run_cpuref(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 10));
  }
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_sm = p.major == 10 ? 100 : p.major == 9 ? 90 : 0;
  if (!g_sm) { fprintf(stderr, "ERR device sm_%d%d: this binary runs on sm_90a or sm_100a\n", p.major, p.minor); return 2; }
  if (mode == "gate" && argc == 7)
    return run_gate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 10));
  if (mode == "bench" && argc == 6) return run_bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: dgemm cpuref DIR M N K SEED 90|100 | gate DIR M N K SEED | bench REPS M N K\n");
  return 2;
}
