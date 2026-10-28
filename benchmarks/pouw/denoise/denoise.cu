// The honest prover's noise costs in a Dither call (verity/core/protocols/pouw/dither/PROTOCOL.md), each part beside
// cuBLASLt's plain int8 GEMM of the same shape, on accel_v31's kernels (v31.cu included whole, its main renamed away) and its
// cuBLASLt setup. A call computes O = (A + d_A)(B + d_B)^T from X = wrap8(A + d_A + V) and W' = wrap8(B + d_B + U pi_tau),
// U and V periodic with period L'. Per shape, the arms (each timed as gpu_kit's interleaved items; arm 0 the baseline):
//
//   plain      cuBLASLt O = A B^T, A and B v31's `llm` operands (Student-t(3) and Gaussian rows quantized to +-127)
//   noised     the same GEMM on X and W' (L' 128's)
//   bits_a, bits_b   d_A's and d_B's SHA-512 coin streams (v31's bits_kernel)
//   form_x, form_w   fill_a_kernel and fill_b_kernel: X, and W', alone (what the no-fold call forms)
//   recompute  fresh noise's plain recompute: serve_rows_kernel's a' = A + d_A - sigma, then cuBLASLt a' b'^T (b' formed once
//              a period, outside the arm; the rank-1 term v31 adds as O's rows are hashed is left out)
//   rows_a     serve_rows_kernel alone
//   fresh3k    fresh noise's de-noise by GEMMs, fused: cuBLASLt at 3K, [X | X | V][W' | -U | -B']^T, the noised GEMM and its
//              two full-K corrections before their wrap digits
// and for each L' of 32, 128 and 512 (suffix -LP):
//   noise_v, coins_w   V (m x L'); U (N x L') and the noise tiles' permutations
//   form_a, form_b     v3.1's forming: X or W', with X^T or B~^T, Z_X or Y_W as the peel's tail rows P_A, P_B in balanced
//                      base-128 digits (dz, dy of them), and the wrap lists
//   ext        the periodic de-noise fused as more K: cuBLASLt [X | P_A][W' | P_B]^T at K + (dz + dy) L', the tensor work of
//              v31's tail stages (each digit at weight 1 here)
//   sep        the de-noise unfused: cuBLASLt P_A P_B^T at (dz + dy) L' into a second buffer, then O += it
//   wrap       wrap2_kernel's dO = w_X B~^T + X w_W^T, sparse on the CUDA cores, over max(m, 128) rows (its tiles are 128)
//   seg        the noised GEMM as S = K / L chunks accumulating in cuBLASLt (beta 1), L = min(8 L', K / 2): the prefix sums
//              T_s left in D by a library GEMM rather than stored from the kernel's registers
// and fresh noise's A side: noise_vk (V, m x K) and form_xk (X from it). The served call (PROTOCOL.md's no-fold call: T_s
// committed for s < S, O recomputed plain), at the word interval Lw (DENOISE_L, default 1,024, the static pick):
//   ts_srv     the noised GEMM over k < (Sw - 1) Lw as Sw - 1 chunks, the first at beta 0 and each next one reading the last
//              one's T_s (C) into the other of two buffers (D): every T_s left whole in DRAM for a hash to read
//   served-LP  the whole call's GPU work in one arm: d_A's bits, V (m x L'), X (all of K), ts_srv, a', and O = a' b'^T
//   nop        an empty kernel: this loop's launch floor
//   plain_i8, plain_bf16   the plain GEMM writing int8 or bf16 (scale float32) instead of int32, where cuBLASLt has one
//
//   denoise REPS name:M:N:K ...     one JSON line a shape on stdout, gpu_kit's summary lines on stderr
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 denoise.cu -lcuda -lcublasLt
#define main v31_main_unused
#include "../accel_v31/v31.cu"
#undef main

namespace {

constexpr int LPS[3] = {32, 128, 512};

__global__ void add_kernel(int* __restrict__ o, const int* __restrict__ d, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    int4 a = reinterpret_cast<int4*>(o)[i];
    const int4 b = reinterpret_cast<const int4*>(d)[i];
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    reinterpret_cast<int4*>(o)[i] = a;
  }
}

__global__ void nop_kernel() {}

// lt_setup's D (m x n) = A B^T + beta C, with A and B at row stride ldk (a K-chunk of wider rows), autotuned the same way
struct LtS {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatmulAlgo_t algo;
  void* ws;
  size_t wsz = 64u << 20;
  int32_t beta = 1;
  int algo_id = -1;
};
void lts_setup(LtS& L, int m, int n, int k, int ldk, const int8_t* A, const int8_t* B, int* D, cudaStream_t st,
               int32_t beta = 1) {
  L.beta = beta;
  CL(cublasLtCreate(&L.h));
  CL(cublasLtMatmulDescCreate(&L.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CL(cublasLtMatrixLayoutCreate(&L.la, CUDA_R_8I, k, n, ldk));
  CL(cublasLtMatrixLayoutCreate(&L.lb, CUDA_R_8I, k, m, ldk));
  CL(cublasLtMatrixLayoutCreate(&L.ld, CUDA_R_32I, n, m, n));
  CK(cudaMalloc(&L.ws, L.wsz));
  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &L.wsz, sizeof L.wsz));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  CL(cublasLtMatmulAlgoGetHeuristic(L.h, L.op, L.la, L.lb, L.ld, L.ld, pref, 16, res, &got));
  if (got == 0) { fprintf(stderr, "cuBLASLt (strided, beta 1): no algorithm\n"); exit(3); }
  const int32_t one = 1;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float best = 1e30f;
  int bi = -1;
  for (int i = 0; i < got; i++) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    bool ok = true;
    for (int w = 0; w < 2 && ok; w++)
      ok = cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &L.beta, D, L.ld, D, L.ld, &res[i].algo, L.ws, L.wsz, st) ==
           CUBLAS_STATUS_SUCCESS;
    if (!ok) continue;
    CK(cudaEventRecord(e0, st));
    for (int r = 0; r < 5; r++) cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &L.beta, D, L.ld, D, L.ld, &res[i].algo, L.ws, L.wsz, st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) { best = ms; bi = i; }
  }
  if (bi < 0) { fprintf(stderr, "cuBLASLt (strided, beta 1): no algorithm ran\n"); exit(3); }
  L.algo = res[bi].algo;
  size_t w;
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_ID, &L.algo_id, sizeof L.algo_id, &w);
  CL(cublasLtMatmulPreferenceDestroy(pref));
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
}
void lts_run(LtS& L, const int8_t* A, const int8_t* B, int* D, cudaStream_t st) {
  const int32_t one = 1;
  CL(cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &L.beta, D, L.ld, D, L.ld, &L.algo, L.ws, L.wsz, st));
}
void lts_run2(LtS& L, const int8_t* A, const int8_t* B, const int* C, int* D, cudaStream_t st) {
  const int32_t one = 1;
  CL(cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &L.beta, C, L.ld, D, L.ld, &L.algo, L.ws, L.wsz, st));
}

// lt_setup's plain GEMM with D of type dt and a float32 scale: false where cuBLASLt has no algorithm for it
struct LtX {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatmulAlgo_t algo;
  void* ws = nullptr;
  size_t wsz = 64u << 20;
  int algo_id = -1;
};
bool ltx_setup(LtX& L, int m, int n, int k, cudaDataType dt, const int8_t* A, const int8_t* B, void* D, cudaStream_t st) {
  CL(cublasLtCreate(&L.h));
  CL(cublasLtMatmulDescCreate(&L.op, CUBLAS_COMPUTE_32I, CUDA_R_32F));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CL(cublasLtMatrixLayoutCreate(&L.la, CUDA_R_8I, k, n, k));
  CL(cublasLtMatrixLayoutCreate(&L.lb, CUDA_R_8I, k, m, k));
  CL(cublasLtMatrixLayoutCreate(&L.ld, dt, n, m, n));
  CK(cudaMalloc(&L.ws, L.wsz));
  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &L.wsz, sizeof L.wsz));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  if (cublasLtMatmulAlgoGetHeuristic(L.h, L.op, L.la, L.lb, L.ld, L.ld, pref, 16, res, &got) != CUBLAS_STATUS_SUCCESS) got = 0;
  CL(cublasLtMatmulPreferenceDestroy(pref));
  const float one = 1.f, zero = 0.f;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float best = 1e30f;
  int bi = -1;
  for (int i = 0; i < got; i++) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    bool ok = true;
    for (int w = 0; w < 2 && ok; w++)
      ok = cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, L.ws, L.wsz, st) ==
           CUBLAS_STATUS_SUCCESS;
    if (!ok) continue;
    CK(cudaEventRecord(e0, st));
    for (int r = 0; r < 5; r++) cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &res[i].algo, L.ws, L.wsz, st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) { best = ms; bi = i; }
  }
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  CK(cudaGetLastError());
  if (bi < 0) return false;
  L.algo = res[bi].algo;
  size_t w;
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_ID, &L.algo_id, sizeof L.algo_id, &w);
  return true;
}
void ltx_run(LtX& L, const int8_t* A, const int8_t* B, void* D, cudaStream_t st) {
  const float one = 1.f, zero = 0.f;
  CL(cublasLtMatmul(L.h, L.op, &one, B, L.la, A, L.lb, &zero, D, L.ld, D, L.ld, &L.algo, L.ws, L.wsz, st));
}

// One L': its coins, both forms' outputs, the extended operands and the GEMMs that read them
struct Per {
  int lp = 0, NT = 0, dz = 0, dy = 0, TL = 0, L = 0, S = 0, cap = 0, flag_h = 0;
  double wrap_a = 0, wrap_b = 0;
  int8_t *U, *V, *X, *XT, *W, *PA, *PB, *Xe, *We;
  int16_t* BtT;
  uint16_t* perm;
  int *listA, *cntA, *listB, *cntB, *offB, *flag;
  Lt ext, sep;
  LtS seg;
};

int run_shape(int reps, const std::string& sh, std::mt19937_64& rng, const gk::Nvml& nv) {
  char name[64];
  int m, n, k;
  if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &n, &k) != 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
  if (m % 16 || n % TILE || k % 1024) { fprintf(stderr, "%s: m a multiple of 16, N of 128, K of 1024\n", name); return 2; }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  const int mw = std::max(m, TILE), kb64 = (k + 511) / 512 * 64;
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const size_t mk = (size_t)mw * k, nk = (size_t)n * k, mn = (size_t)m * n;
  int8_t *A = dalloc<int8_t>(mk), *B = dalloc<int8_t>(nk), *Ap = dalloc<int8_t>(mk), *Bp = dalloc<int8_t>(nk);
  int8_t *Xf = dalloc<int8_t>((size_t)m * 3 * k), *Wf = dalloc<int8_t>((size_t)n * 3 * k), *Vk = dalloc<int8_t>((size_t)m * k);
  uint8_t *dA = dalloc<uint8_t>((size_t)mw * kb64), *dB = dalloc<uint8_t>((size_t)n * kb64);
  int *O = dalloc<int>(mn), *D2 = dalloc<int>(mn), *dO = dalloc<int>((size_t)mw * n);
  int *rsA = dalloc<int>(mw), *shA = dalloc<int>(mw), *rsB = dalloc<int>(n), *shB = dalloc<int>(n), *bad = dalloc<int>(1);
  CK(cudaMemsetAsync(bad, 0, 4, s));
  gen_kernel<<<mw, 256, 0, s>>>(A, k, (u32)rng(), true);
  gen_kernel<<<n, 256, 0, s>>>(B, k, (u32)rng(), false);
  Seeds sd;
  for (auto& row : sd.w)
    for (auto& x : row) x = rng();
  const u64 ba = (u64)m * (kb64 / 64), bb = (u64)n * (kb64 / 64), baw = (u64)mw * (kb64 / 64);
  bits_kernel<<<(unsigned)((baw + 127) / 128), 128, 0, s>>>(sd, S_DA, baw, dA);
  bits_kernel<<<(unsigned)((bb + 127) / 128), 128, 0, s>>>(sd, S_DB, bb, dB);
  serve_rows_kernel<<<n, 256, 0, s>>>(B, dB, k, kb64, 0, Bp, rsB, shB, bad);
  serve_rows_kernel<<<m, 256, 0, s>>>(A, dA, k, kb64, 0, Ap, rsA, shA, bad);
  for (const int cw : {128, 256}) {
    CK(cudaFuncSetAttribute(cw == 128 ? (const void*)wrap2_kernel<128> : (const void*)wrap2_kernel<256>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * 128));
  }
  const int wcw = n % 256 == 0 ? 256 : 128;
  Per P[3];
  for (int q = 0; q < 3; q++) {
    Per& p = P[q];
    p.lp = LPS[q]; p.NT = k / p.lp;
    p.dz = 208LL * p.NT < 16384 ? 2 : 3;
    p.dy = 128LL * p.NT < 16384 ? 2 : 3;
    p.TL = (p.dz + p.dy) * p.lp;
    p.L = std::min(8 * p.lp, k / 2); p.S = k / p.L;
    p.cap = std::max(64, k / 4);
    p.U = dalloc<int8_t>((size_t)n * p.lp); p.V = dalloc<int8_t>((size_t)mw * p.lp);
    p.X = dalloc<int8_t>(mk); p.XT = dalloc<int8_t>(mk); p.W = dalloc<int8_t>(nk); p.BtT = dalloc<int16_t>(nk);
    p.PA = dalloc<int8_t>((size_t)mw * p.TL); p.PB = dalloc<int8_t>((size_t)n * p.TL);
    p.Xe = dalloc<int8_t>((size_t)m * (k + p.TL)); p.We = dalloc<int8_t>((size_t)n * (k + p.TL));
    p.perm = dalloc<uint16_t>((size_t)p.NT * p.lp);
    p.listA = dalloc<int>((size_t)mw * p.cap); p.cntA = dalloc<int>(mw);
    p.listB = dalloc<int>((size_t)n * p.cap); p.cntB = dalloc<int>(n);
    p.offB = dalloc<int>((size_t)n * (p.NT + 1)); p.flag = dalloc<int>(1);
    CK(cudaMemsetAsync(p.flag, 0, 4, s));
    noise_kernel<<<(unsigned)(((u64)n * p.lp / 16 + 127) / 128), 128, 0, s>>>(sd, S_U, (u64)n * p.lp, p.U);
    noise_kernel<<<(unsigned)(((u64)mw * p.lp / 16 + 127) / 128), 128, 0, s>>>(sd, S_V, (u64)mw * p.lp, p.V);
    perm_kernel<<<p.NT, PERM_THREADS(p.lp), p.lp * 6, s>>>(sd, p.lp, p.perm);
    const FormArgs fa{A, dA, p.V, p.perm, p.X, p.XT, p.listA, p.cntA, p.PA, p.flag, mw, k, p.lp, p.NT, kb64, p.dz, p.dy, p.cap, p.TL, nullptr};
    const FormArgs fb{B, dB, p.U, p.perm, p.W, p.BtT, p.listB, p.cntB, p.PB, p.flag, n, k, p.lp, p.NT, kb64, p.dz, p.dy, p.cap, p.TL, p.offB};
    form_a_kernel<<<mw / 16, p.lp, 16 * p.lp * 4, s>>>(fa);
    form_b_kernel<0><<<n / 16, p.lp, 16 * p.lp, s>>>(fb, nullptr);
    CK(cudaMemcpy2DAsync(p.Xe, k + p.TL, p.X, k, k, m, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpy2DAsync(p.Xe + k, k + p.TL, p.PA, p.TL, p.TL, m, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpy2DAsync(p.We, k + p.TL, p.W, k, k, n, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpy2DAsync(p.We + k, k + p.TL, p.PB, p.TL, p.TL, n, cudaMemcpyDeviceToDevice, s));
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    CK(cudaMemcpy(&p.flag_h, p.flag, 4, cudaMemcpyDeviceToHost));
    p.wrap_a = (double)sum_counts(p.cntA, mw) / ((double)mw * k);
    p.wrap_b = (double)sum_counts(p.cntB, n) / ((double)n * k);
    lt_setup(p.ext, m, n, k + p.TL, p.Xe, p.We, O, s);
    lt_setup(p.sep, m, n, p.TL, p.PA, p.PB, D2, s);
    lts_setup(p.seg, m, n, p.L, k, p.X, p.W, O, s);
  }
  for (int c = 0; c < 3; c++) {
    CK(cudaMemcpy2DAsync(Xf + (size_t)c * k, 3 * k, P[1].X, k, k, m, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpy2DAsync(Wf + (size_t)c * k, 3 * k, P[1].W, k, k, n, cudaMemcpyDeviceToDevice, s));
  }
  Lt plain, f3;
  lt_setup(plain, m, n, k, A, B, O, s);
  lt_setup(f3, m, n, 3 * k, Xf, Wf, O, s);
  const char* le = getenv("DENOISE_L");
  const int Lw = std::min(le ? atoi(le) : 1024, k), Sw = Lw > 0 ? k / Lw : 0;
  if (Lw <= 0 || k % Lw) { fprintf(stderr, "%s: DENOISE_L must divide K\n", name); return 2; }
  int* Ts[2] = {dalloc<int>(mn), dalloc<int>(mn)};
  LtS t0, t1;
  if (Sw > 1) lts_setup(t0, m, n, Lw, k, P[1].X, P[1].W, Ts[0], s, 0);
  if (Sw > 2) lts_setup(t1, m, n, Lw, k, P[1].X, P[1].W, Ts[1], s, 1);
  auto ts_chain = [&](const Per* p, cudaStream_t st) {
    if (Sw < 2) return;
    lts_run2(t0, p->X, p->W, Ts[0], Ts[0], st);
    for (int c = 1; c < Sw - 1; c++)
      lts_run2(t1, p->X + (size_t)c * Lw, p->W + (size_t)c * Lw, Ts[(c - 1) & 1], Ts[c & 1], st);
  };
  int8_t* O8 = dalloc<int8_t>(mn);
  uint16_t* Obf = dalloc<uint16_t>(mn);
  LtX xi8, xbf;
  const bool has_i8 = ltx_setup(xi8, m, n, k, CUDA_R_8I, A, B, O8, s);
  const bool has_bf = ltx_setup(xbf, m, n, k, CUDA_R_16BF, A, B, Obf, s);
  CK(cudaStreamSynchronize(s));
  const Per& p1 = P[1];
  const u64 cx = (u64)m * k / 16, cwb = (u64)n * k / 16;
  std::vector<gk::Arm> arms = {
      {"plain", [&](cudaStream_t st, int r) { for (int i = 0; i < r; i++) lt_run(plain, A, B, O, st); }},
      {"noised", [&](cudaStream_t st, int r) { for (int i = 0; i < r; i++) lt_run(plain, p1.X, p1.W, O, st); }},
      {"bits_a", [&](cudaStream_t st, int r) {
         for (int i = 0; i < r; i++) bits_kernel<<<(unsigned)((ba + 127) / 128), 128, 0, st>>>(sd, S_DA, ba, dA);
       }},
      {"bits_b", [&](cudaStream_t st, int r) {
         for (int i = 0; i < r; i++) bits_kernel<<<(unsigned)((bb + 127) / 128), 128, 0, st>>>(sd, S_DB, bb, dB);
       }},
      {"form_x", [&](cudaStream_t st, int r) {
         for (int i = 0; i < r; i++)
           fill_a_kernel<<<(unsigned)((cx + 255) / 256), 256, 0, st>>>(A, dA, p1.V, p1.X, cx, k, p1.lp, kb64);
       }},
      {"form_w", [&](cudaStream_t st, int r) {
         for (int i = 0; i < r; i++)
           fill_b_kernel<<<(unsigned)((cwb + 255) / 256), 256, 0, st>>>(B, dB, p1.U, p1.perm, p1.W, cwb, k, p1.lp, kb64);
       }},
      {"rows_a", [&](cudaStream_t st, int r) {
         for (int i = 0; i < r; i++) serve_rows_kernel<<<m, 256, 0, st>>>(A, dA, k, kb64, 0, Ap, rsA, shA, bad);
       }},
      {"recompute", [&](cudaStream_t st, int r) {
         for (int i = 0; i < r; i++) {
           serve_rows_kernel<<<m, 256, 0, st>>>(A, dA, k, kb64, 0, Ap, rsA, shA, bad);
           lt_run(plain, Ap, Bp, D2, st);
         }
       }},
      {"fresh3k", [&](cudaStream_t st, int r) { for (int i = 0; i < r; i++) lt_run(f3, Xf, Wf, O, st); }},
  };
  for (int q = 0; q < 3; q++) {
    Per* p = &P[q];
    const std::string sfx = "-" + std::to_string(p->lp);
    const u64 cv = (u64)m * p->lp, cu = (u64)n * p->lp;
    arms.push_back({"noise_v" + sfx, [=](cudaStream_t st, int r) {
                      for (int i = 0; i < r; i++) noise_kernel<<<(unsigned)((cv / 16 + 127) / 128), 128, 0, st>>>(sd, S_V, cv, p->V);
                    }});
    arms.push_back({"coins_w" + sfx, [=](cudaStream_t st, int r) {
                      for (int i = 0; i < r; i++) {
                        noise_kernel<<<(unsigned)((cu / 16 + 127) / 128), 128, 0, st>>>(sd, S_U, cu, p->U);
                        perm_kernel<<<p->NT, PERM_THREADS(p->lp), p->lp * 6, st>>>(sd, p->lp, p->perm);
                      }
                    }});
    arms.push_back({"form_a" + sfx, [=](cudaStream_t st, int r) {
                      const FormArgs fa{A, dA, p->V, p->perm, p->X, p->XT, p->listA, p->cntA, p->PA, p->flag, m, k, p->lp,
                                        p->NT, kb64, p->dz, p->dy, p->cap, p->TL, nullptr};
                      for (int i = 0; i < r; i++) form_a_kernel<<<m / 16, p->lp, 16 * p->lp * 4, st>>>(fa);
                    }});
    arms.push_back({"form_b" + sfx, [=](cudaStream_t st, int r) {
                      const FormArgs fb{B, dB, p->U, p->perm, p->W, p->BtT, p->listB, p->cntB, p->PB, p->flag, n, k, p->lp,
                                        p->NT, kb64, p->dz, p->dy, p->cap, p->TL, p->offB};
                      for (int i = 0; i < r; i++) form_b_kernel<0><<<n / 16, p->lp, 16 * p->lp, st>>>(fb, nullptr);
                    }});
    arms.push_back({"ext" + sfx, [=](cudaStream_t st, int r) { for (int i = 0; i < r; i++) lt_run(p->ext, p->Xe, p->We, O, st); }});
    arms.push_back({"sep" + sfx, [=](cudaStream_t st, int r) {
                      for (int i = 0; i < r; i++) {
                        lt_run(p->sep, p->PA, p->PB, D2, st);
                        add_kernel<<<4 * 132, 256, 0, st>>>(O, D2, mn / 4);
                      }
                    }});
    arms.push_back({"wrap" + sfx, [=](cudaStream_t st, int r) {
                      const WrapArgs w{p->XT, p->BtT, p->listA, p->cntA, p->listB, p->cntB, dO, mw, n, p->cap, p->cap, p->offB,
                                       p->lp, p->NT};
                      for (int i = 0; i < r; i++) {
                        if (wcw == 256) wrap2_kernel<256><<<dim3(n / 256, mw / TILE), 512, p->lp * 128, st>>>(w);
                        else wrap2_kernel<128><<<dim3(n / 128, mw / TILE), 256, p->lp * 128, st>>>(w);
                      }
                    }});
    arms.push_back({"seg" + sfx, [=](cudaStream_t st, int r) {
                      for (int i = 0; i < r; i++)
                        for (int c = 0; c < p->S; c++) lts_run(p->seg, p->X + (size_t)c * p->L, p->W + (size_t)c * p->L, O, st);
                    }});
  }
  const u64 cvk = (u64)m * k;
  int8_t* X1 = P[1].X;
  arms.push_back({"noise_vk", [=](cudaStream_t st, int r) {
                    for (int i = 0; i < r; i++) noise_kernel<<<(unsigned)((cvk / 16 + 127) / 128), 128, 0, st>>>(sd, S_V, cvk, Vk);
                  }});
  arms.push_back({"form_xk", [=](cudaStream_t st, int r) {
                    for (int i = 0; i < r; i++)
                      fill_a_kernel<<<(unsigned)((cx + 255) / 256), 256, 0, st>>>(A, dA, Vk, X1, cx, k, k, kb64);
                  }});
  arms.push_back({"ts_srv", [&](cudaStream_t st, int r) { for (int i = 0; i < r; i++) ts_chain(&P[1], st); }});
  for (int q = 0; q < 3; q++) {
    const Per* p = &P[q];
    const u64 cv = (u64)m * p->lp;
    arms.push_back({"served-" + std::to_string(p->lp), [&, p, cv](cudaStream_t st, int r) {
                      for (int i = 0; i < r; i++) {
                        bits_kernel<<<(unsigned)((ba + 127) / 128), 128, 0, st>>>(sd, S_DA, ba, dA);
                        noise_kernel<<<(unsigned)((cv / 16 + 127) / 128), 128, 0, st>>>(sd, S_V, cv, p->V);
                        fill_a_kernel<<<(unsigned)((cx + 255) / 256), 256, 0, st>>>(A, dA, p->V, p->X, cx, k, p->lp, kb64);
                        ts_chain(p, st);
                        serve_rows_kernel<<<m, 256, 0, st>>>(A, dA, k, kb64, 0, Ap, rsA, shA, bad);
                        lt_run(plain, Ap, Bp, D2, st);
                      }
                    }});
  }
  arms.push_back({"nop", [](cudaStream_t st, int r) { for (int i = 0; i < r; i++) nop_kernel<<<1, 32, 0, st>>>(); }});
  if (has_i8) arms.push_back({"plain_i8", [&](cudaStream_t st, int r) { for (int i = 0; i < r; i++) ltx_run(xi8, A, B, O8, st); }});
  if (has_bf)
    arms.push_back({"plain_bf16", [&](cudaStream_t st, int r) { for (int i = 0; i < r; i++) ltx_run(xbf, A, B, Obf, st); }});
  const gk::Timing tm = gk::timed_items(arms, reps, s, nv);
  CK(cudaStreamSynchronize(s));
  CK(cudaGetLastError());
  gk::summary(tm, arms, stderr);
  int hbad = 0;
  CK(cudaMemcpy(&hbad, bad, 4, cudaMemcpyDeviceToHost));
  printf("{\"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"gpu\": \"%s\", \"uuid\": \"%s\", \"sms\": %d, \"steady\": %s, "
         "\"warmup_ms\": %.1f, \"served_rows_bad\": %d, \"lt_algo\": %d, \"Lw\": %d, \"Sw\": %d, \"ts_algo\": [%d, %d], "
         "\"i8_algo\": %d, \"bf16_algo\": %d, \"lp\": [",
         name, m, n, k, prop.name, nv.uuid().c_str(), prop.multiProcessorCount, tm.steady ? "true" : "false", tm.warmup_ms, hbad,
         plain.algo_id, Lw, Sw, t0.algo_id, t1.algo_id, has_i8 ? xi8.algo_id : -1, has_bf ? xbf.algo_id : -1);
  for (int q = 0; q < 3; q++) {
    const Per& p = P[q];
    printf("%s{\"lp\": %d, \"NT\": %d, \"dz\": %d, \"dy\": %d, \"TL\": %d, \"L\": %d, \"S\": %d, \"cap\": %d, \"flag\": %d, "
           "\"wrap_rate_A\": %.6f, \"wrap_rate_B\": %.6f, \"ext_algo\": %d, \"sep_algo\": %d, \"seg_algo\": %d}",
           q ? ", " : "", p.lp, p.NT, p.dz, p.dy, p.TL, p.L, p.S, p.cap, p.flag_h, p.wrap_a, p.wrap_b, p.ext.algo_id,
           p.sep.algo_id, p.seg.algo_id);
  }
  printf("], \"arms\": {");
  for (size_t a = 0; a < arms.size(); a++) {
    std::vector<double> v, f;
    bool thr = false;
    for (const auto& it : tm.items)
      if (it.arm == (int)a) {
        v.push_back(it.ms / it.calls);
        if (it.sm_mhz()) f.push_back(it.sm_mhz());
        thr |= ((it.before.throttle | it.during.throttle | it.after.throttle) & gk::BAD_THROTTLE) != 0;
      }
    printf("%s\"%s\": {\"ms\": %.6f, \"min_ms\": %.6f, \"mhz\": %.0f, \"throttled\": %s}", a ? ", " : "", arms[a].name.c_str(),
           gk::median(v), *std::min_element(v.begin(), v.end()), gk::median(f), thr ? "true" : "false");
  }
  printf("}}\n");
  fflush(stdout);
  CK(cudaDeviceSynchronize());
  CK(cudaDeviceReset());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: denoise REPS name:M:N:K ...\n");
    return 2;
  }
  gk::Nvml nv(0);
  std::mt19937_64 rng(0x6d64656e6f697365ull);
  for (int i = 2; i < argc; i++)
    if (const int rc = run_shape(atoi(argv[1]), argv[i], rng, nv)) return rc;
  return 0;
}
