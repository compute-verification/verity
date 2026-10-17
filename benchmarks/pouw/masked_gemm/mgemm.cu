// The honest cost of masked matmul PoUW constructions on sm_120a (RTX PRO 6000): A B^T computed from masked operands, as
// one GEMM over stacked operands, int8 (s8 -> s32) or E4M3's exact integer grid (f32 accumulation), against one plain GEMM.
//
// Full rank: A~ = A + R, B~ = B + S (R, S hashed from the seed per call), and
//   A B^T = A~ B~^T - R B~^T - A S^T = [A~ | R | A] [B~ | -B~ | -S]^T,   K = 3k,
// so the masked product's running sums are the first k columns' (receipts go there), and the peel is the next 2k.
// Low rank r: A~ = A + U V^T, B~ = B + W Z^T, U (m x r), W (n x r) in {-1, 0, 1}, V and Z (k x r) two signed permutations
// of each chunk of r columns of k (fgemm.cu's F for r = 32), and with P = B~ V (n x r), Q = A~ Z - U G, G = V^T Z,
//   A B^T = A~ B~^T - U P^T - Q W^T = [A~ | -w_l U ... | -Q_l ...] [B~ | P_l ... | w_l W ...]^T,   K = k + 2 l r,
// P and Q in l digits with weights w_l on the mask factor (int8: 1, 127, 127, ...; E4M3: 1, 16, 256, 256, ...), l from
// the worst case over all inputs. Forming on CUDA cores (one warp a row, V and Z read as descriptors), or with thin GEMMs
// on the tensor cores (cuBLASLt: U V^T and W Z^T with inner dimension r, then A~ Z and B~ V with r outputs).
// The GEMM: cuBLASLt (int8 -> s32, E4M3 -> f32), or fgemm.cu's mainloop (128 x 128 tiles, 16 warps) with receipts duo or
// TB4C at L = 32 or 64 over the first k columns. Every arm forms both sides per call (fresh masks per call).
//   mgemm gate DIR M N K SEED        every arm at one small shape, inputs and outputs to DIR for mref.py
//   mgemm bench REPS SHAPE...        interleaved timing, one JSON line a shape (SHAPE = name:m:n:k); MG_ARMS selects arms
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a mgemm.cu -lcuda -lcublasLt -lnvidia-ml
#ifndef FG_LIB
#define FG_LIB
#endif
#include "../floored_gemm/fgemm.cu"

#include <cuda_fp8.h>

#include <map>

enum Fmt { I8 = 0, E4 = 1 };
static const char* FMTN[2] = {"i8", "e4"};
static const int RANKS[5] = {8, 16, 32, 64, 128};
constexpr int NR = 5;

template <int FMT> DEV int dec(uint32_t b) {
  if constexpr (FMT == I8) return (int)(int8_t)(b & 0xFF);
  else return (int)__half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(b & 0xFF), __NV_E4M3)));
}
template <int FMT> DEV uint32_t enc(int v) {
  if constexpr (FMT == I8) return (uint32_t)(v & 0xFF);
  else return (uint32_t)__nv_cvt_float_to_fp8((float)v, __NV_SATFINITE, __NV_E4M3);
}
static uint8_t enc_e4m3(int v) {   // integers with at most 4 significant bits, |v| <= 448
  if (v == 0) return 0;
  const uint8_t s = v < 0 ? 0x80 : 0;
  int a = std::abs(v), e = 0;
  while ((a >> e) >= 2) e++;
  const int man = ((a << 3) >> e) & 7;
  if (((man | 8) << e) >> 3 != a || e > 8) { fprintf(stderr, "%d not on the E4M3 grid\n", v); exit(3); }
  return s | (uint8_t)((e + 7) << 3) | (uint8_t)man;
}
static uint8_t enc_h(int fmt, int v) { return fmt == I8 ? (uint8_t)(int8_t)v : enc_e4m3(v); }

// ---------------------------------------------------------------- forming, full rank
// 16 bytes a thread; the mask bytes are mix32(seed ^ mix32(row k/4 + col/4)), int8 (b & 127) - 64, E4M3 (b & 15) - 8
template <int FMT, bool SIDE_A>
__global__ void __launch_bounds__(256) full_form(const uint8_t* __restrict__ X, uint8_t* __restrict__ Xp, int rows, int k,
                                                 int Kp, uint32_t seed) {
  const int per = k / 16;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)rows * per) return;
  const int row = (int)(idx / per), c = (int)(idx % per) * 16;
  const uint4 x4 = *reinterpret_cast<const uint4*>(X + (size_t)row * k + c);
  const uint32_t xs[4] = {x4.x, x4.y, x4.z, x4.w};
  uint32_t o0[4], o1[4], o2[4];
#pragma unroll
  for (int w = 0; w < 4; w++) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)row * (uint32_t)(k / 4) + (uint32_t)(c / 4 + w)));
    o0[w] = o1[w] = o2[w] = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int x = dec<FMT>(xs[w] >> (8 * i)), hb = (h >> (8 * i)) & 0xFF;
      const int r = FMT == I8 ? (hb & 0x7F) - 64 : (hb & 0xF) - 8;
      const int a = SIDE_A ? x + r : x + r, b = SIDE_A ? r : -(x + r), cc = SIDE_A ? x : -r;
      o0[w] |= enc<FMT>(a) << (8 * i); o1[w] |= enc<FMT>(b) << (8 * i); o2[w] |= enc<FMT>(cc) << (8 * i);
    }
  }
  uint8_t* p = Xp + (size_t)row * Kp + c;
  *reinterpret_cast<uint4*>(p) = make_uint4(o0[0], o0[1], o0[2], o0[3]);
  *reinterpret_cast<uint4*>(p + k) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
  *reinterpret_cast<uint4*>(p + 2 * k) = make_uint4(o2[0], o2[1], o2[2], o2[3]);
}

// ---------------------------------------------------------------- forming, low rank
DEV int fdiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
// write column c's tail entries for a wide-factor value y (P on side B, Q on side A) and mask entry u (U or W): side A
// [ -w_l u (l < L) | -digit_l(y) ], side B [ digit_l(y) | w_l u ]; *flag |= 1 if y doesn't fit L digits
template <int FMT, bool SIDE_A>
DEV void tail_col(uint8_t* t, int R, int L, int c, int y, int u, int* flag) {
  int q, d;
  for (int l = 0; l < L; l++) {
    int wl;
    if constexpr (FMT == I8) {
      if (l == 0) { d = y - 127 * fdiv(y + 63, 127); q = (y - d) / 127; }
      else { d = max(-127, min(127, q)); q -= d; }
      wl = l == 0 ? 1 : 127;
    } else {
      if (l == 0) { d = y - 16 * fdiv(y + 8, 16); q = (y - d) / 16; }
      else if (l == 1) { d = q - 16 * fdiv(q + 8, 16); q = (q - d) / 16; }
      else { d = max(-16, min(16, q)); q -= d; }
      wl = l == 0 ? 1 : l == 1 ? 16 : 256;
    }
    if (SIDE_A) { t[l * R + c] = (uint8_t)enc<FMT>(-wl * u); t[(L + l) * R + c] = (uint8_t)enc<FMT>(-d); }
    else { t[l * R + c] = (uint8_t)enc<FMT>(d); t[(L + l) * R + c] = (uint8_t)enc<FMT>(wl * u); }
  }
  if (q != 0) atomicOr(flag, 1);
}
// one warp a row: X~ = X + U_row (descriptor own) and Y = X~ (descriptor inv) [- U_row G on side A], then the tail
template <int FMT, bool SIDE_A, int R>
__global__ void __launch_bounds__(256) low_form(const uint8_t* __restrict__ X, const int8_t* __restrict__ Um,
                                                const uint32_t* __restrict__ own, const uint32_t* __restrict__ inv,
                                                const int* __restrict__ G, uint8_t* __restrict__ Xp, int rows, int k, int Kp,
                                                int L, int* flag) {
  constexpr int CPL = R >= 32 ? R / 32 : 1, NSUB = R >= 32 ? 1 : 32 / R, NBK = 128 / R;
  __shared__ int ush[8][R];
  __shared__ int buf[8][128];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, row = blockIdx.x * 8 + w;
  if (row >= rows) return;
  for (int i = lane; i < R; i += 32) ush[w][i] = Um[(size_t)row * R + i];
  __syncwarp();
  const int c_l = R >= 32 ? lane : lane % R, sub = R >= 32 ? 0 : lane / R;
  int y[CPL];
#pragma unroll
  for (int i = 0; i < CPL; i++) y[i] = 0;
  const uint8_t* x = X + (size_t)row * k;
  uint8_t* xp = Xp + (size_t)row * Kp;
  for (int c0 = 0; c0 < k; c0 += 128) {
    const uint32_t qw = *reinterpret_cast<const uint32_t*>(x + c0 + 4 * lane);
    const uint4 d4 = *reinterpret_cast<const uint4*>(own + c0 + 4 * lane);
    const uint32_t d[4] = {d4.x, d4.y, d4.z, d4.w};
    uint32_t xw = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int e1 = ush[w][d[i] & 0xFF], e2 = ush[w][(d[i] >> 8) & 0xFF];
      const int v = dec<FMT>(qw >> (8 * i)) + ((d[i] >> 16) & 1 ? -e1 : e1) + ((d[i] >> 17) & 1 ? -e2 : e2);
      xw |= enc<FMT>(v) << (8 * i);
      buf[w][4 * lane + i] = v;
    }
    *reinterpret_cast<uint32_t*>(xp + c0 + 4 * lane) = xw;
    __syncwarp();
#pragma unroll
    for (int ch = sub; ch < NBK; ch += NSUB) {
      const int s = c0 / R + ch;
#pragma unroll
      for (int i = 0; i < CPL; i++) {
        const uint32_t di = inv[s * R + c_l + 32 * i];
        const int b1 = buf[w][ch * R + (di & 0xFF)], b2 = buf[w][ch * R + ((di >> 8) & 0xFF)];
        y[i] += ((di >> 16) & 1 ? -b1 : b1) + ((di >> 17) & 1 ? -b2 : b2);
      }
    }
    __syncwarp();
  }
  // the padding past the tail, up to the row stride: another layout's bytes are left there (E4M3 NaNs among them)
  for (int j = k + 2 * L * R + lane; j < Kp; j += 32) xp[j] = 0;
  if constexpr (R < 32) {
#pragma unroll
    for (int off = R; off < 32; off <<= 1) y[0] += __shfl_xor_sync(0xFFFFFFFFu, y[0], off);
    if (sub != 0) return;
  }
#pragma unroll
  for (int i = 0; i < CPL; i++) {
    const int c = c_l + 32 * i;
    if constexpr (SIDE_A)
      for (int j = 0; j < R; j++) y[i] -= ush[w][j] * G[j * R + c];
    tail_col<FMT, SIDE_A>(xp + k, R, L, c, y[i], ush[w][c], flag);
  }
}
// the tensor-core forming's elementwise step: X~ = X + D, D the thin GEMM U V^T (s32, or f32 for E4M3); 4 a thread
template <int FMT>
__global__ void __launch_bounds__(256) add_form(const uint8_t* __restrict__ X, const void* __restrict__ D,
                                                uint8_t* __restrict__ Xp, int rows, int k, int Kp) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x, per = k / 4;
  if (idx >= (size_t)rows * per) return;
  const int row = (int)(idx / per), c = (int)(idx % per) * 4;
  const uint32_t xw = *reinterpret_cast<const uint32_t*>(X + (size_t)row * k + c);
  int dv[4];
  if constexpr (FMT == I8) {
    const int4 v = *reinterpret_cast<const int4*>(reinterpret_cast<const int*>(D) + (size_t)row * k + c);
    dv[0] = v.x; dv[1] = v.y; dv[2] = v.z; dv[3] = v.w;
  } else {
    const float4 v = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(D) + (size_t)row * k + c);
    dv[0] = (int)v.x; dv[1] = (int)v.y; dv[2] = (int)v.z; dv[3] = (int)v.w;
  }
  uint32_t o = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) o |= enc<FMT>(dec<FMT>(xw >> (8 * i)) + dv[i]) << (8 * i);
  *reinterpret_cast<uint32_t*>(Xp + (size_t)row * Kp + c) = o;
}
// the tensor-core forming's tail: Y (rows x ldy, the thin GEMM X~ Z or B~ V) [- U G], digits, the tail entries
template <int FMT, bool SIDE_A>
__global__ void __launch_bounds__(256) tail_form(const void* __restrict__ Y, int ldy, const int8_t* __restrict__ Um,
                                                 const int* __restrict__ G, uint8_t* __restrict__ Xp, int rows, int R, int k,
                                                 int Kp, int L, int* flag) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)rows * R) return;
  const int row = (int)(idx / R), c = (int)(idx % R);
  int y = FMT == I8 ? reinterpret_cast<const int*>(Y)[(size_t)row * ldy + c]
                    : (int)reinterpret_cast<const float*>(Y)[(size_t)row * ldy + c];
  const int8_t* u = Um + (size_t)row * R;
  if (SIDE_A)
    for (int j = 0; j < R; j++) y -= u[j] * G[j * R + c];
  tail_col<FMT, SIDE_A>(Xp + (size_t)row * Kp + k, R, L, c, y, u[c], flag);
  if (c == 0)
    for (int j = k + 2 * L * R; j < Kp; j++) Xp[(size_t)row * Kp + j] = 0;
}

// ---------------------------------------------------------------- the receipt GEMMs (fgemm.cu's W mainloop, no tail)
#define GEMMF(NAME, TM, BN, WM, ST, RS, KIND)                                                                   \
  __global__ void __launch_bounds__(ML<TM, BN, WM, ST, true>::NT, 1) NAME(const Args a) {                       \
    gemm_body<TM, BN, WM, ST, false, RS, KIND, true>(a);                                                        \
  }
GEMM(mi_plain, 128, 128, 32, 3, false, 0, NONE)
GEMM(mi_duo32, 128, 128, 32, 3, false, 32, DUO)
GEMM(mi_duo64, 128, 128, 32, 3, false, 64, DUO)
GEMM(mi_t4c32, 128, 128, 32, 3, false, 32, TB4C)
GEMM(mi_t4c64, 128, 128, 32, 3, false, 64, TB4C)
GEMMF(mf_plain, 128, 128, 32, 3, 0, NONE)
GEMMF(mf_duo32, 128, 128, 32, 3, 32, DUO)
GEMMF(mf_duo64, 128, 128, 32, 3, 64, DUO)
GEMMF(mf_t4c32, 128, 128, 32, 3, 32, TB4C)
GEMMF(mf_t4c64, 128, 128, 32, 3, 64, TB4C)
constexpr int NRC = 5;
static const char* RCN[NRC] = {"cg", "cduo32", "cduo64", "ct4c32", "ct4c64"};
static const int RCRS[NRC] = {0, 32, 64, 32, 64};
static const int RCKIND[NRC] = {NONE, DUO, DUO, TB4C, TB4C};
static Kern RCK[2][NRC] = {{mi_plain, mi_duo32, mi_duo64, mi_t4c32, mi_t4c64}, {mf_plain, mf_duo32, mf_duo64, mf_t4c32, mf_t4c64}};
constexpr int CSMEM = ML<128, 128, 32, 3>::SMEM, CNT = 512;

// ---------------------------------------------------------------- cuBLASLt, int8 -> s32 or E4M3 -> f32
// D (m x n, row-major, ldd) = A (m x kk, lda) B^T (B n x kk, ldb) [+ D]: cuBLASLt's column-major (n, m, kk), its A = our B
struct LtG {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatmulAlgo_t algo;
  const void *A, *B;
  void* D;
  int fmt, beta, algo_id = -1;
  float ms = 0;   // the chosen algorithm's time a call at setup
};
static cublasLtHandle_t g_lt;
static void* g_ws;
static const size_t g_wsz = 64u << 20;
static void ltg_run(const LtG& L, cudaStream_t st) {
  if (L.algo_id == -2) return;
  const int32_t i1 = 1, ib = L.beta;
  const float f1 = 1.f, fb = (float)L.beta;
  const void* al = L.fmt == I8 ? (const void*)&i1 : (const void*)&f1;
  const void* be = L.fmt == I8 ? (const void*)&ib : (const void*)&fb;
  CL(cublasLtMatmul(g_lt, L.op, al, L.B, L.la, L.A, L.lb, be, L.D, L.ld, L.D, L.ld, &L.algo, g_ws, g_wsz, st));
}
static LtG ltg(int fmt, int m, int n, int kk, int lda, int ldb, int ldd, const void* A, const void* B, void* D, int beta,
               cudaStream_t st, int batch = 1) {
  LtG L;
  L.A = A; L.B = B; L.D = D; L.fmt = fmt; L.beta = beta;
  const cudaDataType_t in = fmt == I8 ? CUDA_R_8I : CUDA_R_8F_E4M3, out = fmt == I8 ? CUDA_R_32I : CUDA_R_32F;
  CL(cublasLtMatmulDescCreate(&L.op, fmt == I8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F, fmt == I8 ? CUDA_R_32I : CUDA_R_32F));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CL(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CL(cublasLtMatrixLayoutCreate(&L.la, in, kk, n, ldb));
  CL(cublasLtMatrixLayoutCreate(&L.lb, in, kk, m, lda));
  CL(cublasLtMatrixLayoutCreate(&L.ld, out, n, m, ldd));
  if (batch > 1) {   // problems stacked in rows, as fgemm.cu's Args batch
    const int64_t sa = (int64_t)n * ldb, sb = (int64_t)m * lda, sd = (int64_t)m * ldd;
    for (auto [lay, stride] : {std::pair{L.la, sa}, std::pair{L.lb, sb}, std::pair{L.ld, sd}}) {
      CL(cublasLtMatrixLayoutSetAttribute(lay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof batch));
      CL(cublasLtMatrixLayoutSetAttribute(lay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof stride));
    }
  }
  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_wsz, sizeof g_wsz));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  if (cublasLtMatmulAlgoGetHeuristic(g_lt, L.op, L.la, L.lb, L.ld, L.ld, pref, 16, res, &got) != CUBLAS_STATUS_SUCCESS) got = 0;
  if (got == 0) {
    fprintf(stderr, "cuBLASLt %s: no algorithm for %d %d %d (lda %d ldb %d); its arms are skipped\n", FMTN[fmt], m, n, kk, lda, ldb);
    CL(cublasLtMatmulPreferenceDestroy(pref));
    L.algo_id = -2;
    return L;
  }
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float best = 1e30f;
  int bi = -1;
  for (int i = 0; i < got; i++) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    L.algo = res[i].algo;
    const int32_t i1 = 1, i0 = 0;
    const float f1 = 1.f, f0 = 0.f;
    const void* al = fmt == I8 ? (const void*)&i1 : (const void*)&f1;
    const void* be = fmt == I8 ? (const void*)&i0 : (const void*)&f0;
    if (cublasLtMatmul(g_lt, L.op, al, B, L.la, A, L.lb, be, D, L.ld, D, L.ld, &L.algo, g_ws, g_wsz, st) != CUBLAS_STATUS_SUCCESS)
      continue;
    CK(cudaEventRecord(e0, st));
    for (int r = 0; r < 3; r++) cublasLtMatmul(g_lt, L.op, al, B, L.la, A, L.lb, be, D, L.ld, D, L.ld, &L.algo, g_ws, g_wsz, st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) { best = ms; bi = i; }
  }
  if (bi < 0) {
    fprintf(stderr, "cuBLASLt %s: no algorithm ran for %d %d %d; its arms are skipped\n", FMTN[fmt], m, n, kk);
    CL(cublasLtMatmulPreferenceDestroy(pref));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    L.algo_id = -2;
    return L;
  }
  L.algo = res[bi].algo;
  L.ms = best / 3;
  size_t w;
  cublasLtMatmulAlgoConfigGetAttribute(&L.algo, CUBLASLT_ALGO_CONFIG_ID, &L.algo_id, sizeof L.algo_id, &w);
  CL(cublasLtMatmulPreferenceDestroy(pref));
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return L;
}

// ---------------------------------------------------------------- one shape's problem
static int round_up(int x, int a) { return (x + a - 1) / a * a; }
// the digits a wide factor needs for every input: |A~ Z| <= amax 2k/r and |U G| <= 4k/r
static int limbs_for(int fmt, int k, int r) {
  const long b = (long)((fmt == I8 ? 127 : 16) + 2) * 2 * k / r;
  for (int L = 1; L < 200; L++) {
    const long cap = fmt == I8 ? 63 + 127L * 127 * (L - 1) : L == 1 ? 7 : 7 + 16 * 7 + 256L * 16 * (L - 2);
    if (cap >= b) return L;
  }
  return -1;
}
// two signed permutations of each chunk of r columns: own[j] = p1 | p2 << 8 | s1 << 16 | s2 << 17 (row j of the factor);
// inv[s r + c] = c1 | c2 << 8 | s1 << 16 | s2 << 17 (column c's two entries in chunk s), and the dense factor
struct Fac {
  std::vector<uint32_t> own, inv;
  std::vector<int> dense;   // k x r
  Fac(int k, int r, Rng& g) : own(k), inv(k), dense((size_t)k * r, 0) {
    std::vector<int> p1(r), p2(r), c1(r), c2(r);
    std::vector<uint32_t> s1(r), s2(r);
    for (int s = 0; s < k / r; s++) {
      for (int i = 0; i < r; i++) p1[i] = p2[i] = i;
      for (int i = r - 1; i > 0; i--) { std::swap(p1[i], p1[g.next() % (i + 1)]); std::swap(p2[i], p2[g.next() % (i + 1)]); }
      for (int c = 0; c < r; c++) {
        s1[c] = g.next() & 1; s2[c] = g.next() & 1;
        own[s * r + c] = p1[c] | p2[c] << 8 | s1[c] << 16 | s2[c] << 17;
        c1[p1[c]] = c; c2[p2[c]] = c;
        dense[(size_t)(s * r + c) * r + p1[c]] += s1[c] ? -1 : 1;
        dense[(size_t)(s * r + c) * r + p2[c]] += s2[c] ? -1 : 1;
      }
      for (int j = 0; j < r; j++) inv[s * r + j] = c1[j] | c2[j] << 8 | s1[c1[j]] << 16 | s2[c2[j]] << 17;
    }
  }
};
struct Low {   // one rank's masks and buffers
  int r, rp, L[2], Kc[2], Klt[2], Kmain[2];   // Kmain: the stacked GEMM's depth on cuBLASLt, Klt or Kc, the faster
  int8_t *UA, *UB;               // mask factors U (m x r), W (n x r), values
  uint32_t *ownV, *invV, *ownZ, *invZ;
  int* G;
  uint8_t *tUA[2], *tUB[2], *tV[2], *tZ[2], *tVt[2], *tZt[2];   // the thin GEMMs' operands per format, r padded to rp
  LtG main_[2], sepk[2], sepp[2], uv[2], wz[2], az[2], bv[2];
  std::vector<int8_t> hUA, hUB;
  Fac *V, *Z;
  std::vector<int> hG;
};
struct MProb {
  int m, n, k, grid;
  uint8_t *A[2], *B[2];   // codes per format
  uint8_t *XA, *XB;       // formed operands, m x 3k and n x 3k
  int* U;                 // m x n s32 (or f32 bits)
  uint32_t* R;
  void *DA, *DB, *YA, *YB;   // the tensor-core forming's intermediates
  int* flag;
  Low low[NR];
  LtG plain[2], full[2];
  Maps* cmaps_full[2];
  Maps* cmaps_low[NR][2];
  uint32_t seed;
  cudaStream_t st;
  std::vector<int8_t> hA[2], hB[2];   // values
};

static void fill_vals(std::vector<int8_t>& v, Rng& g, int lo, int hi) { for (auto& x : v) x = (int8_t)g.range(lo, hi); }
static uint8_t* upload_codes(int fmt, const std::vector<int8_t>& v) {
  std::vector<uint8_t> c(v.size());
  for (size_t i = 0; i < v.size(); i++) c[i] = enc_h(fmt, v[i]);
  return upload(c);
}
static uint8_t* upload_dense(int fmt, const std::vector<int>& d, int rows, int r, int rp, bool transpose) {
  std::vector<uint8_t> c((size_t)rows * rp, 0);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < r; j++) {
      const uint8_t e = enc_h(fmt, d[(size_t)i * r + j]);
      if (transpose) c[(size_t)j * rows + i] = e; else c[(size_t)i * rp + j] = e;
    }
  return upload(c);
}

static void msetup(MProb& P, int m, int n, int k, uint32_t seed, bool gate) {
  P.m = m; P.n = n; P.k = k; P.seed = seed;
  if (m % 128 || n % 128 || k % 128 || k < 256) { fprintf(stderr, "shape %d %d %d not tiled\n", m, n, k); exit(2); }
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  P.grid = std::min((m / 128) * (n / 128), nsm);
  CK(cudaStreamCreate(&P.st));
  Rng g{seed * 0x100000001B3ull + 29};
  // full rank: int8 |A| <= 63 (masks [-64, 63]), E4M3 |A| <= 8 (masks [-8, 7]); low rank: int8 |A| <= 125, E4M3 <= 14
  // (masks |U V^T| <= 2). One set of values a format serves both: the narrower range.
  for (int f = 0; f < 2; f++) {
    P.hA[f].resize((size_t)m * k); P.hB[f].resize((size_t)n * k);
    const int a = f == I8 ? 63 : 8;
    fill_vals(P.hA[f], g, -a, a); fill_vals(P.hB[f], g, -a, a);
    P.A[f] = upload_codes(f, P.hA[f]); P.B[f] = upload_codes(f, P.hB[f]);
  }
  CK(cudaMalloc(&P.XA, (size_t)m * 3 * k)); CK(cudaMalloc(&P.XB, (size_t)n * 3 * k));
  CK(cudaMemset(P.XA, 0, (size_t)m * 3 * k)); CK(cudaMemset(P.XB, 0, (size_t)n * 3 * k));
  CK(cudaMalloc(&P.U, (size_t)m * n * 4));
  CK(cudaMalloc(&P.R, (size_t)(m / 128) * (n / 128) * CNT * 8 * 4));
  CK(cudaMalloc(&P.DA, (size_t)m * k * 4)); CK(cudaMalloc(&P.DB, (size_t)n * k * 4));
  CK(cudaMalloc(&P.YA, (size_t)m * 128 * 4)); CK(cudaMalloc(&P.YB, (size_t)n * 128 * 4));
  CK(cudaMalloc(&P.flag, 4)); CK(cudaMemset(P.flag, 0, 4));
  for (int f = 0; f < 2; f++) {
    P.plain[f] = ltg(f, m, n, k, k, k, n, P.A[f], P.B[f], P.U, 0, P.st);
    P.full[f] = ltg(f, m, n, 3 * k, 3 * k, 3 * k, n, P.XA, P.XB, P.U, 0, P.st);
    Maps mp{tmap(P.XA, 3 * k, m, 128), tmap(P.XB, 3 * k, n, 128), tmap(P.XA, 3 * k, m, 128), tmap(P.XB, 3 * k, n, 128)};
    P.cmaps_full[f] = upload_maps(mp);
  }
  for (int i = 0; i < NR; i++) {
    Low& lw = P.low[i];
    const int r = RANKS[i];
    lw.r = r; lw.rp = std::max(r, 16);
    lw.hUA.resize((size_t)m * r); lw.hUB.resize((size_t)n * r);
    fill_vals(lw.hUA, g, -1, 1); fill_vals(lw.hUB, g, -1, 1);
    lw.UA = upload(lw.hUA); lw.UB = upload(lw.hUB);
    lw.V = new Fac(k, r, g); lw.Z = new Fac(k, r, g);
    lw.ownV = upload(lw.V->own); lw.invV = upload(lw.V->inv); lw.ownZ = upload(lw.Z->own); lw.invZ = upload(lw.Z->inv);
    lw.hG.assign((size_t)r * r, 0);
    for (int j = 0; j < k; j++)
      for (int a = 0; a < r; a++)
        if (lw.V->dense[(size_t)j * r + a])
          for (int b = 0; b < r; b++) lw.hG[a * r + b] += lw.V->dense[(size_t)j * r + a] * lw.Z->dense[(size_t)j * r + b];
    lw.G = upload(lw.hG);
    std::vector<int> ua(lw.hUA.begin(), lw.hUA.end()), ub(lw.hUB.begin(), lw.hUB.end());
    for (int f = 0; f < 2; f++) {
      lw.L[f] = limbs_for(f, k, r);
      const int used = k + 2 * lw.L[f] * r;
      lw.Kc[f] = round_up(used, 128);
      lw.Klt[f] = round_up(used, 32);
      if (lw.Kc[f] > 3 * k) { fprintf(stderr, "r %d: K' %d past 3k\n", r, lw.Kc[f]); exit(2); }
      lw.tUA[f] = upload_dense(f, ua, m, r, lw.rp, false); lw.tUB[f] = upload_dense(f, ub, n, r, lw.rp, false);
      lw.tV[f] = upload_dense(f, lw.V->dense, k, r, lw.rp, false); lw.tZ[f] = upload_dense(f, lw.Z->dense, k, r, lw.rp, false);
      lw.tVt[f] = upload_dense(f, lw.V->dense, k, r, lw.rp, true); lw.tZt[f] = upload_dense(f, lw.Z->dense, k, r, lw.rp, true);
      const int Kp = lw.Kc[f];
      // the padding up to Kc is zero, so the stacked GEMM may run to Klt or Kc: cuBLASLt's speed depends on which
      auto faster = [](LtG a, LtG b) { return a.algo_id == -2 || (b.algo_id != -2 && b.ms < a.ms) ? b : a; };
      lw.main_[f] = ltg(f, m, n, lw.Klt[f], Kp, Kp, n, P.XA, P.XB, P.U, 0, P.st);
      lw.sepp[f] = ltg(f, m, n, lw.Klt[f] - k, Kp, Kp, n, P.XA + k, P.XB + k, P.U, 1, P.st);
      lw.Kmain[f] = lw.Klt[f];
      if (Kp != lw.Klt[f]) {
        const LtG alt = ltg(f, m, n, Kp, Kp, Kp, n, P.XA, P.XB, P.U, 0, P.st);
        if (alt.algo_id != -2 && (lw.main_[f].algo_id == -2 || alt.ms < lw.main_[f].ms)) { lw.main_[f] = alt; lw.Kmain[f] = Kp; }
        lw.sepp[f] = faster(lw.sepp[f], ltg(f, m, n, Kp - k, Kp, Kp, n, P.XA + k, P.XB + k, P.U, 1, P.st));
      }
      lw.sepk[f] = ltg(f, m, n, k, Kp, Kp, n, P.XA, P.XB, P.U, 0, P.st);
      lw.uv[f] = ltg(f, m, k, lw.rp, lw.rp, lw.rp, k, lw.tUA[f], lw.tV[f], P.DA, 0, P.st);
      lw.wz[f] = ltg(f, n, k, lw.rp, lw.rp, lw.rp, k, lw.tUB[f], lw.tZ[f], P.DB, 0, P.st);
      lw.az[f] = ltg(f, m, lw.rp, k, Kp, k, lw.rp, P.XA, lw.tZt[f], P.YA, 0, P.st);
      lw.bv[f] = ltg(f, n, lw.rp, k, Kp, k, lw.rp, P.XB, lw.tVt[f], P.YB, 0, P.st);
      Maps mp{tmap(P.XA, Kp, m, 128), tmap(P.XB, Kp, n, 128), tmap(P.XA, Kp, m, 128), tmap(P.XB, Kp, n, 128)};
      P.cmaps_low[i][f] = upload_maps(mp);
    }
  }
  for (int f = 0; f < 2; f++)
    for (int j = 0; j < NRC; j++) CK(cudaFuncSetAttribute(RCK[f][j], cudaFuncAttributeMaxDynamicSharedMemorySize, CSMEM));
  (void)gate;
}

// ---------------------------------------------------------------- the arms
template <int FMT> static void form_full_t(MProb& P) {
  const size_t ta = (size_t)P.m * (P.k / 16), tb = (size_t)P.n * (P.k / 16);
  full_form<FMT, true><<<(unsigned)((ta + 255) / 256), 256, 0, P.st>>>(P.A[FMT], P.XA, P.m, P.k, 3 * P.k, P.seed);
  full_form<FMT, false><<<(unsigned)((tb + 255) / 256), 256, 0, P.st>>>(P.B[FMT], P.XB, P.n, P.k, 3 * P.k, P.seed ^ 0x5A5A5A5Au);
}
template <int FMT, int R> static void form_low_r(MProb& P, Low& lw) {
  const int Kp = lw.Kc[FMT];
  low_form<FMT, true, R><<<(P.m + 7) / 8, 256, 0, P.st>>>(P.A[FMT], lw.UA, lw.ownV, lw.invZ, lw.G, P.XA, P.m, P.k, Kp, lw.L[FMT], P.flag);
  low_form<FMT, false, R><<<(P.n + 7) / 8, 256, 0, P.st>>>(P.B[FMT], lw.UB, lw.ownZ, lw.invV, nullptr, P.XB, P.n, P.k, Kp, lw.L[FMT], P.flag);
}
template <int FMT> static void form_low_t(MProb& P, int i) {
  Low& lw = P.low[i];
  switch (lw.r) {
    case 8: form_low_r<FMT, 8>(P, lw); break;
    case 16: form_low_r<FMT, 16>(P, lw); break;
    case 32: form_low_r<FMT, 32>(P, lw); break;
    case 64: form_low_r<FMT, 64>(P, lw); break;
    default: form_low_r<FMT, 128>(P, lw); break;
  }
}
template <int FMT> static void form_tc_t(MProb& P, int i) {
  Low& lw = P.low[i];
  const int Kp = lw.Kc[FMT], k = P.k;
  ltg_run(lw.uv[FMT], P.st);
  add_form<FMT><<<(unsigned)(((size_t)P.m * k / 4 + 255) / 256), 256, 0, P.st>>>(P.A[FMT], P.DA, P.XA, P.m, k, Kp);
  ltg_run(lw.wz[FMT], P.st);
  add_form<FMT><<<(unsigned)(((size_t)P.n * k / 4 + 255) / 256), 256, 0, P.st>>>(P.B[FMT], P.DB, P.XB, P.n, k, Kp);
  ltg_run(lw.az[FMT], P.st);
  tail_form<FMT, true><<<(unsigned)(((size_t)P.m * lw.r + 255) / 256), 256, 0, P.st>>>(P.YA, lw.rp, lw.UA, lw.G, P.XA, P.m, lw.r, k, Kp, lw.L[FMT], P.flag);
  ltg_run(lw.bv[FMT], P.st);
  tail_form<FMT, false><<<(unsigned)(((size_t)P.n * lw.r + 255) / 256), 256, 0, P.st>>>(P.YB, lw.rp, lw.UB, nullptr, P.XB, P.n, lw.r, k, Kp, lw.L[FMT], P.flag);
}
static void form_full(MProb& P, int f) { f == I8 ? form_full_t<I8>(P) : form_full_t<E4>(P); }
static void form_low(MProb& P, int f, int i) { f == I8 ? form_low_t<I8>(P, i) : form_low_t<E4>(P, i); }
static void form_tc(MProb& P, int f, int i) { f == I8 ? form_tc_t<I8>(P, i) : form_tc_t<E4>(P, i); }
static void cgemm(MProb& P, int f, int rc, Maps* maps, int K) {
  Args a{maps, P.U, P.R, P.m, P.n, K, 8, P.seed, P.k};
  RCK[f][rc]<<<P.grid, CNT, CSMEM, P.st>>>(a);
}

// an arm: what it runs, by name. Full rank: full (form + GEMM), fullform; low rank r: low, lowform, lowsep (the peel as a
// second GEMM adding into C), lowtc (tensor-core forming + GEMM), lowtcform; receipts on the custom mainloop:
// <rc>_full, <rc>_low (cg = no receipt, cduo32, cduo64, ct4c32, ct4c64), each with its forming
struct Arm { std::string name; int fmt, kind, r, rc; bool gemm; };
enum { A_PLAIN, A_FULL, A_FULLFORM, A_LOW, A_LOWFORM, A_LOWSEP, A_LOWTC, A_LOWTCFORM, A_CFULL, A_CLOW };
static std::vector<Arm> all_arms() {
  std::vector<Arm> v;
  for (int f = 0; f < 2; f++) {
    const std::string F = FMTN[f];
    v.push_back({"plain_" + F, f, A_PLAIN, 0, 0, true});
    v.push_back({"full_" + F, f, A_FULL, 0, 0, true});
    v.push_back({"fullform_" + F, f, A_FULLFORM, 0, 0, false});
    for (int j = 0; j < NRC; j++) v.push_back({std::string(RCN[j]) + "_full_" + F, f, A_CFULL, 0, j, true});
    for (int i = 0; i < NR; i++) {
      const std::string R = std::to_string(RANKS[i]);
      v.push_back({"low" + R + "_" + F, f, A_LOW, i, 0, true});
      v.push_back({"lowform" + R + "_" + F, f, A_LOWFORM, i, 0, false});
      v.push_back({"lowsep" + R + "_" + F, f, A_LOWSEP, i, 0, true});
      v.push_back({"lowtc" + R + "_" + F, f, A_LOWTC, i, 0, true});
      v.push_back({"lowtcform" + R + "_" + F, f, A_LOWTCFORM, i, 0, false});
      for (int j = 0; j < NRC; j++) v.push_back({std::string(RCN[j]) + "_low" + R + "_" + F, f, A_CLOW, i, j, true});
    }
  }
  return v;
}
// whether every cuBLASLt GEMM the arm runs has an algorithm
static bool arm_ok(MProb& P, const Arm& a) {
  const int f = a.fmt;
  const Low& lw = P.low[a.r];
  auto ok = [](const LtG& L) { return L.algo_id != -2; };
  switch (a.kind) {
    case A_PLAIN: return ok(P.plain[f]);
    case A_FULL: return ok(P.full[f]);
    case A_LOW: return ok(lw.main_[f]);
    case A_LOWSEP: return ok(lw.sepk[f]) && ok(lw.sepp[f]);
    case A_LOWTC: return ok(lw.main_[f]) && ok(lw.uv[f]) && ok(lw.wz[f]) && ok(lw.az[f]) && ok(lw.bv[f]);
    case A_LOWTCFORM: return ok(lw.uv[f]) && ok(lw.wz[f]) && ok(lw.az[f]) && ok(lw.bv[f]);
    default: return true;
  }
}
static void mrun_arm(MProb& P, const Arm& a) {
  const int f = a.fmt;
  Low& lw = P.low[a.r];
  switch (a.kind) {
    case A_PLAIN: ltg_run(P.plain[f], P.st); break;
    case A_FULL: form_full(P, f); ltg_run(P.full[f], P.st); break;
    case A_FULLFORM: form_full(P, f); break;
    case A_LOW: form_low(P, f, a.r); ltg_run(lw.main_[f], P.st); break;
    case A_LOWFORM: form_low(P, f, a.r); break;
    case A_LOWSEP: form_low(P, f, a.r); ltg_run(lw.sepk[f], P.st); ltg_run(lw.sepp[f], P.st); break;
    case A_LOWTC: form_tc(P, f, a.r); ltg_run(lw.main_[f], P.st); break;
    case A_LOWTCFORM: form_tc(P, f, a.r); break;
    case A_CFULL: form_full(P, f); cgemm(P, f, a.rc, P.cmaps_full[f], 3 * P.k); break;
    case A_CLOW: form_low(P, f, a.r); cgemm(P, f, a.rc, P.cmaps_low[a.r][f], lw.Kc[f]); break;
  }
}

static int mgate(const std::string& dir, int m, int n, int k, uint32_t seed) {
  MProb P;
  msetup(P, m, n, k, seed, true);
  const size_t ub = (size_t)m * n * 4, rb = (size_t)(m / 128) * (n / 128) * CNT * 8 * 4;
  for (int f = 0; f < 2; f++) {
    dump(dir, (std::string("A_") + FMTN[f] + ".bin").c_str(), P.hA[f].data(), P.hA[f].size(), false);
    dump(dir, (std::string("B_") + FMTN[f] + ".bin").c_str(), P.hB[f].data(), P.hB[f].size(), false);
  }
  for (int i = 0; i < NR; i++) {
    const std::string R = std::to_string(RANKS[i]);
    Low& lw = P.low[i];
    dump(dir, ("UA" + R + ".bin").c_str(), lw.hUA.data(), lw.hUA.size(), false);
    dump(dir, ("UB" + R + ".bin").c_str(), lw.hUB.data(), lw.hUB.size(), false);
    dump(dir, ("V" + R + ".bin").c_str(), lw.V->dense.data(), lw.V->dense.size() * 4, false);
    dump(dir, ("Z" + R + ".bin").c_str(), lw.Z->dense.data(), lw.Z->dense.size() * 4, false);
  }
  std::string arms_json, skipped;
  for (const Arm& a : all_arms()) {
    if (!arm_ok(P, a)) { skipped += (skipped.empty() ? "\"" : ", \"") + a.name + "\""; continue; }
    CK(cudaMemset(P.U, 0xA5, ub)); CK(cudaMemset(P.R, 0xA5, rb));
    mrun_arm(P, a);
    CK(cudaStreamSynchronize(P.st));
    CK(cudaGetLastError());
    if (a.kind == A_FULLFORM || a.kind == A_LOWFORM || a.kind == A_LOWTCFORM) {
      const int Kp = a.kind == A_FULLFORM ? 3 * k : P.low[a.r].Kc[a.fmt];
      dump(dir, ("XA_" + a.name + ".bin").c_str(), P.XA, (size_t)m * Kp, true);
      dump(dir, ("XB_" + a.name + ".bin").c_str(), P.XB, (size_t)n * Kp, true);
    } else {
      dump(dir, ("U_" + a.name + ".bin").c_str(), P.U, ub, true);
      if (a.kind == A_CFULL || a.kind == A_CLOW) dump(dir, ("R_" + a.name + ".bin").c_str(), P.R, rb, true);
    }
    arms_json += (arms_json.empty() ? "\"" : ", \"") + a.name + "\"";
  }
  int flag;
  CK(cudaMemcpy(&flag, P.flag, 4, cudaMemcpyDeviceToHost));
  FILE* fp = fopen((dir + "/meta.json").c_str(), "w");
  fprintf(fp, "{\"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"grid\": %d, \"TM\": 128, \"BN\": 128, \"WM\": 32, \"NCT\": %d, "
              "\"digit_overflow\": %d, \"low\": {", m, n, k, seed, P.grid, CNT, flag);
  for (int i = 0; i < NR; i++) {
    const Low& lw = P.low[i];
    fprintf(fp, "%s\"%d\": {\"L\": [%d, %d], \"Kc\": [%d, %d], \"Klt\": [%d, %d], \"Kmain\": [%d, %d]}", i ? ", " : "", lw.r,
            lw.L[0], lw.L[1], lw.Kc[0], lw.Kc[1], lw.Klt[0], lw.Klt[1], lw.Kmain[0], lw.Kmain[1]);
  }
  fprintf(fp, "}, \"arms\": [%s], \"skipped\": [%s]}\n", arms_json.c_str(), skipped.c_str());
  fclose(fp);
  printf("gate dump %s: m %d n %d k %d grid %d flag %d\n", dir.c_str(), m, n, k, P.grid, flag);
  return 0;
}

static int mbench(int reps, const std::vector<std::string>& shapes) {
  Nv nv;
  int nsm;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::vector<Arm> all;
  const char* fa = getenv("MG_ARMS");
  for (const Arm& a : all_arms()) {
    const std::string want = fa ? std::string(",") + fa + "," : "";
    if (!fa || want.find("," + a.name + ",") != std::string::npos) all.push_back(a);
  }
  for (const auto& sh : shapes) {
    char name[64];
    int m, n, k;
    if (sscanf(sh.c_str(), "%63[^:]:%d:%d:%d", name, &m, &n, &k) < 4) { fprintf(stderr, "bad shape %s\n", sh.c_str()); return 2; }
    MProb P;
    msetup(P, m, n, k, 0x5EED0000u + m + n + k, false);
    std::vector<Arm> sel;
    for (const Arm& a : all) if (arm_ok(P, a)) sel.push_back(a);
    const int NS = (int)sel.size();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time_arm = [&](const Arm& a, int calls) {
      CK(cudaEventRecord(e0, P.st));
      for (int c = 0; c < calls; c++) mrun_arm(P, a);
      CK(cudaEventRecord(e1, P.st));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      return (double)ms / calls;
    };
    std::vector<int> calls(NS);
    for (int j = 0; j < NS; j++) {
      time_arm(sel[j], 2);
      const double t = time_arm(sel[j], 3);
      calls[j] = std::max(2, std::min(400, (int)(20.0 / std::max(t, 1e-4))));
    }
    CK(cudaGetLastError());
    std::vector<std::vector<double>> ms(NS), mhz(NS);
    for (int r = 0; r < reps; r++)
      for (int j = 0; j < NS; j++) {
        const int a = (j + r) % NS;
        ms[a].push_back(time_arm(sel[a], calls[a]));
        mhz[a].push_back(nv.sm());
      }
    int flag;
    CK(cudaMemcpy(&flag, P.flag, 4, cudaMemcpyDeviceToHost));
    printf("{\"shape\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"sms\": %d, \"gpu\": \"%s\", \"grid\": %d, \"digit_overflow\": %d, \"low\": {",
           name, m, n, k, nsm, prop.name, P.grid, flag);
    for (int i = 0; i < NR; i++) {
      const Low& lw = P.low[i];
      fprintf(stdout, "%s\"%d\": {\"L\": [%d, %d], \"Kc\": [%d, %d], \"Klt\": [%d, %d], \"Kmain\": [%d, %d]}", i ? ", " : "",
              lw.r, lw.L[0], lw.L[1], lw.Kc[0], lw.Kc[1], lw.Klt[0], lw.Klt[1], lw.Kmain[0], lw.Kmain[1]);
    }
    std::string skipped;
    for (const Arm& a : all) if (!arm_ok(P, a)) skipped += (skipped.empty() ? "\"" : ", \"") + a.name + "\"";
    printf("}, \"skipped\": [%s], \"lt_algo\": [%d, %d], \"arms\": {", skipped.c_str(), P.plain[0].algo_id, P.plain[1].algo_id);
    for (int j = 0; j < NS; j++) {
      printf("%s\"%s\": {\"ms\": %.6f, \"mhz\": %.0f, \"calls\": %d, \"ms_all\": [", j ? ", " : "", sel[j].name.c_str(), median(ms[j]),
             median(mhz[j]), calls[j]);
      for (size_t i = 0; i < ms[j].size(); i++) printf("%s%.6f", i ? ", " : "", ms[j][i]);
      printf("]}");
    }
    printf("}}\n");
    fflush(stdout);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    CK(cudaDeviceSynchronize());
    for (void* p : {(void*)P.XA, (void*)P.XB, (void*)P.U, (void*)P.R, P.DA, P.DB, P.YA, P.YB}) CK(cudaFree(p));
  }
  return 0;
}

#ifndef MG_LIB
int main(int argc, char** argv) {
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  if (argc >= 7 && !strcmp(argv[1], "gate")) return mgate(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], 0, 0));
  if (argc >= 4 && !strcmp(argv[1], "bench")) {
    std::vector<std::string> s(argv + 3, argv + argc);
    return mbench(atoi(argv[2]), s);
  }
  fprintf(stderr, "usage: mgemm gate DIR M N K SEED | bench REPS name:m:n:k...\n");
  return 2;
}
#endif
