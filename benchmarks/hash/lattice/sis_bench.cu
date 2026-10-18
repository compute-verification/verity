// sis_bench: the SIS hash h(x) = A x mod q on the integer tensor cores (cuBLASLt int8 GEMMs), timed rep by rep beside
// SHA-512 on the same die. SHA-512 is commit_gpu's compression (benchmarks/pouw/commit_gpu/sha512.cuh, included read-only).
//
// The hash (README.md has it in full). A row is m bytes x_0..x_{m-1}. Its digits s_k are the bytes as int8 (enc q32, q16)
// or its nibbles, low first (enc q8n: s_2i = x_i & 15, s_2i+1 = x_i >> 4). A is L limbs of n x K int8 (K = m or 2m), read
// row-major from the SHA-512 counter stream of the tag "verity/hash-lattice/sis/v0|ENC|n|K": limb l, row j, column k is
// stream byte (l n + j) K + k. Then h_j = sum_l 2^(8l) sum_k A_l[j][k] s_k mod 2^(8L), stored little-endian (L bytes a
// word). Uniform int8 limbs make each entry of A uniform mod q.
//
//   sis_bench ENC M_BYTES N_WORDS BATCH_MIB REPS OUTDIR [tree]
//
// Order: build A on the host; fill a batch of rows; hash it on the GPU; check sampled rows against the host reference
// (exit 4 on any mismatch, before anything is timed); dump the sampled rows and digests for reference.py; then REPS
// interleaved reps of: the SIS hash (GEMM + limb combine), SHA-512 over every row, SHA-512 over every digest, and the raw
// compression rate, each with the SM clock. With `tree`, an all-SIS tree (each level's digests are the next level's rows)
// is checked on sampled rows of every level and timed in the same reps.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../pouw/commit_gpu/sha512.cuh"

typedef uint64_t u64;
typedef uint32_t u32;

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) {                                                                         \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x);     \
      exit(3);                                                                                       \
    }                                                                                                \
  } while (0)
#define CL(x)                                                                         \
  do {                                                                                \
    cublasStatus_t s_ = (x);                                                          \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                \
      fprintf(stderr, "cuBLASLt %d at %s:%d: %s\n", (int)s_, __FILE__, __LINE__, #x); \
      exit(3);                                                                        \
    }                                                                                 \
  } while (0)

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  unsigned sm() {
    unsigned c = 0;
    if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c);
    return c;
  }
};

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
static std::string jlist(const std::vector<double>& v) {
  std::string s = "[";
  char b[32];
  for (size_t i = 0; i < v.size(); ++i) { snprintf(b, sizeof b, "%s%.5f", i ? ", " : "", v[i]); s += b; }
  return s + "]";
}

// --- the hash's parameters -----------------------------------------------------------------------------------------

struct Enc {
  std::string name;
  int L;       // limbs of A (q = 2^(8L))
  int split;   // 1: digits are bytes as int8; 2: nibbles
};
static Enc enc_of(const char* s) {
  if (!strcmp(s, "q32")) return {"q32", 4, 1};
  if (!strcmp(s, "q16")) return {"q16", 2, 1};
  if (!strcmp(s, "q8n")) return {"q8n", 1, 2};
  fprintf(stderr, "ENC is q32, q16 or q8n\n");
  exit(2);
}

// A's limbs, stacked: (L n) x K int8, row-major, from the SHA-512 counter stream of the tag.
static std::vector<int8_t> derive_A(const Enc& e, int n, int K) {
  char tag[128];
  snprintf(tag, sizeof tag, "verity/hash-lattice/sis/v0|%s|%d|%d", e.name.c_str(), n, K);
  const size_t len = (size_t)e.L * n * K, tl = strlen(tag);
  std::vector<int8_t> A(len);
  std::vector<uint8_t> msg(tl + 8);
  memcpy(msg.data(), tag, tl);
  uint8_t d[64];
  for (size_t c = 0; c * 64 < len; ++c) {
    sha::put_be64(msg.data() + tl, (u64)c);
    sha::digest(msg.data(), msg.size(), d);
    memcpy(A.data() + c * 64, d, std::min<size_t>(64, len - c * 64));
  }
  return A;
}

static void digits_of(const Enc& e, const uint8_t* x, int m, std::vector<int>& s) {
  s.resize((size_t)m * e.split);
  for (int i = 0; i < m; ++i) {
    if (e.split == 1) s[i] = (int8_t)x[i];
    else { s[2 * i] = x[i] & 15; s[2 * i + 1] = x[i] >> 4; }
  }
}

// The host reference: the digest of one row, L bytes a word, little-endian.
static void ref_digest(const Enc& e, const std::vector<int8_t>& A, int n, const uint8_t* x, int m, uint8_t* out) {
  std::vector<int> s;
  digits_of(e, x, m, s);
  const int K = m * e.split;
  for (int j = 0; j < n; ++j) {
    u64 h = 0;
    for (int l = 0; l < e.L; ++l) {
      const int8_t* a = A.data() + ((size_t)l * n + j) * K;
      long long acc = 0;
      for (int k = 0; k < K; ++k) acc += (long long)a[k] * s[k];
      h += (u64)acc << (8 * l);
    }
    for (int b = 0; b < e.L; ++b) out[(size_t)j * e.L + b] = (uint8_t)(h >> (8 * b));
  }
}

// --- device kernels ------------------------------------------------------------------------------------------------

__global__ void fill(uint8_t* p, size_t n, u32 seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    u32 x = (u32)i * 0x9E3779B1u ^ seed ^ (u32)(i >> 32) * 0x85EBCA77u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = (uint8_t)x;
  }
}

// bytes -> nibbles, low first: 4 bytes in, 8 digits out
__global__ void split_nibbles(const u32* x, uint2* y, size_t words) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
    const u32 v = x[i], lo = v & 0x0F0F0F0Fu, hi = (v >> 4) & 0x0F0F0F0Fu;
    y[i] = make_uint2(__byte_perm(lo, hi, 0x5140), __byte_perm(lo, hi, 0x7362));
  }
}

// Y: rows x (L n) s32 limb products -> D: rows x n words of L bytes, h_j = sum_l Y[l n + j] << 8l mod 2^(8L)
template <int L>
__global__ void combine(const int32_t* Y, uint8_t* D, size_t rows, int n) {
  const size_t total = rows * n;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / n, j = i % n;
    const int32_t* y = Y + r * (size_t)L * n + j;
    u32 h = 0;
#pragma unroll
    for (int l = 0; l < L; ++l) h += (u32)y[(size_t)l * n] << (8 * l);
    if (L == 4) ((u32*)D)[i] = h;
    else if (L == 2) ((uint16_t*)D)[i] = (uint16_t)h;
    else D[i] = (uint8_t)h;
  }
}

__device__ __forceinline__ u64 bswap64(u64 v) {
  const u32 lo = (u32)v, hi = (u32)(v >> 32);
  return ((u64)__byte_perm(lo, 0, 0x0123) << 32) | __byte_perm(hi, 0, 0x0123);
}

// SHA-512 of each row (m bytes, a multiple of 128): one thread a row, m/128 blocks and the padding block.
__global__ void row_sha(const uint8_t* X, size_t rows, int m, uint8_t* out) {
  const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const uint4* p = (const uint4*)(X + r * (size_t)m);
  u64 st[8], w[16];
  sha::iv(st);
  for (int b = 0; b < m / 128; ++b) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint4 v = p[b * 8 + i];
      w[2 * i] = bswap64((u64)v.y << 32 | v.x);
      w[2 * i + 1] = bswap64((u64)v.w << 32 | v.z);
    }
    sha::compress(st, w);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = 0x8000000000000000ull;
  w[15] = (u64)m * 8;
  sha::compress(st, w);
  u64* o = (u64*)(out + r * 64);
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = bswap64(st[i]);
}

// commit_gpu's raw rate kernel (its sha_rate): each thread chains `reps` compressions.
__global__ void sha_rate(u64* io, int reps) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 st[8], w[16];
  sha::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sha::compress(st, w);
  }
  if (st[0] == 0x0123456789abcdefull) io[t] = st[1];
}

// --- the fused kernel ----------------------------------------------------------------------------------------------
//
// One pass: rows of bytes in, digests out. mma.sync m16n8k32 s8 (every arch from sm_80), CTA tile 128 rows x 128 columns of
// W, 8 warps of 64 x 32, a 64-digit k-step double-buffered through registers. W is A with its rows interleaved (row
// j L + l is limb l of word j), so a thread's accumulators hold adjacent limbs of one word and the epilogue combines them
// in registers (and across the lane pair t, t^1 when L = 4). With SPLIT = 2 each byte's two nibbles are split on the way
// into shared memory. Rows a multiple of 128, L n a multiple of 128, K a multiple of 64.
namespace fz {
constexpr int BM = 128, BN = 128, BK = 64, LD = BK + 16;  // LD: a smem row's stride in bytes (padding against conflicts)

__device__ __forceinline__ void ldsm_x4(u32* r, const void* p) {
  const u32 a = (u32)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma(int* c, const u32* a, const u32* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint4 nib_lo(uint4 v) {  // bytes 0..7 of 16 -> 16 digits
  const u32 l0 = v.x & 0x0F0F0F0Fu, h0 = (v.x >> 4) & 0x0F0F0F0Fu, l1 = v.y & 0x0F0F0F0Fu, h1 = (v.y >> 4) & 0x0F0F0F0Fu;
  return make_uint4(__byte_perm(l0, h0, 0x5140), __byte_perm(l0, h0, 0x7362), __byte_perm(l1, h1, 0x5140),
                    __byte_perm(l1, h1, 0x7362));
}
__device__ __forceinline__ uint4 nib_hi(uint4 v) { return nib_lo(make_uint4(v.z, v.w, 0, 0)); }

template <int L, int SPLIT>
__global__ void __launch_bounds__(256) sis_fused(const uint8_t* __restrict__ X, const int8_t* __restrict__ W,
                                                 uint8_t* __restrict__ D, int m, int NW) {
  __shared__ __align__(16) uint8_t sX[2][BM * LD], sW[2][BN * LD];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
  const int n0 = blockIdx.x * BN;
  const size_t r0 = (size_t)blockIdx.y * BM;
  const int K = m * SPLIT, steps = K / BK, xb = BK / SPLIT;  // xb: data bytes a row a step
  int acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0;

  uint4 rx[2], rw[2];
  auto load = [&](int s) {
    const int k0 = s * BK;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + i * 256, row = idx >> 2, ch = idx & 3;
      rw[i] = *(const uint4*)(W + (size_t)(n0 + row) * K + k0 + ch * 16);
    }
    if (SPLIT == 1) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int idx = tid + i * 256, row = idx >> 2, ch = idx & 3;
        rx[i] = *(const uint4*)(X + (r0 + row) * m + (size_t)s * xb + ch * 16);
      }
    } else {
      const int row = tid >> 1, ch = tid & 1;
      rx[0] = *(const uint4*)(X + (r0 + row) * m + (size_t)s * xb + ch * 16);
    }
  };
  auto store = [&](int b) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + i * 256, row = idx >> 2, ch = idx & 3;
      *(uint4*)(&sW[b][row * LD + ch * 16]) = rw[i];
    }
    if (SPLIT == 1) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int idx = tid + i * 256, row = idx >> 2, ch = idx & 3;
        *(uint4*)(&sX[b][row * LD + ch * 16]) = rx[i];
      }
    } else {
      const int row = tid >> 1, ch = tid & 1;
      *(uint4*)(&sX[b][row * LD + ch * 32]) = nib_lo(rx[0]);
      *(uint4*)(&sX[b][row * LD + ch * 32 + 16]) = nib_hi(rx[0]);
    }
  };

  load(0);
  store(0);
  __syncthreads();
  for (int s = 0; s < steps; ++s) {
    const int b = s & 1;
    if (s + 1 < steps) load(s + 1);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 32) {
      u32 af[4][4], bf[2][4];
      const int mi = lane >> 3, r = lane & 7;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ldsm_x4(af[i], &sX[b][(wm * 64 + i * 16 + (mi & 1) * 8 + r) * LD + kk + (mi >> 1) * 16]);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        ldsm_x4(bf[j], &sW[b][(wn * 32 + j * 16 + (mi >> 1) * 8 + r) * LD + kk + (mi & 1) * 16]);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) mma(acc[i][j], af[i], &bf[j >> 1][(j & 1) * 2]);
    }
    if (s + 1 < steps) {
      store(b ^ 1);
      __syncthreads();
    }
  }

  // epilogue: thread holds C[g][2t], C[g][2t+1], C[g+8][2t], C[g+8][2t+1] of each 16 x 8 tile
  const int g = lane >> 2, t = lane & 3, n = NW / L;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const size_t row = r0 + wm * 64 + i * 16 + g;
      const int col = n0 + wn * 32 + j * 8 + 2 * t;  // a W row: word col / L, limb col % L
      const int* c = acc[i][j];
      if (L == 1) {
        *(uint16_t*)(D + row * n + col) = (uint16_t)((c[0] & 255) | (c[1] & 255) << 8);
        *(uint16_t*)(D + (row + 8) * n + col) = (uint16_t)((c[2] & 255) | (c[3] & 255) << 8);
      } else if (L == 2) {
        ((uint16_t*)D)[row * n + col / 2] = (uint16_t)((u32)c[0] + ((u32)c[1] << 8));
        ((uint16_t*)D)[(row + 8) * n + col / 2] = (uint16_t)((u32)c[2] + ((u32)c[3] << 8));
      } else {
        const int sh = (t & 1) * 16;
        const u32 pg = ((u32)c[0] << sh) + ((u32)c[1] << (sh + 8)), pg8 = ((u32)c[2] << sh) + ((u32)c[3] << (sh + 8));
        const u32 fg = pg + __shfl_xor_sync(0xffffffffu, pg, 1), fg8 = pg8 + __shfl_xor_sync(0xffffffffu, pg8, 1);
        if (!(t & 1)) ((u32*)D)[row * n + col / 4] = fg;
        else ((u32*)D)[(row + 8) * n + col / 4] = fg8;
      }
    }
}
}  // namespace fz

// --- cuBLASLt ------------------------------------------------------------------------------------------------------

// Y (M x N, s32, row-major) = X (M x K, int8, row-major) W (N x K, int8, row-major)^T, autotuned over the heuristic's top 8.
struct Gemm {
  cublasLtHandle_t lt{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t la{}, lb{}, lc{};
  cublasLtMatmulAlgo_t algo{};
  void* ws = nullptr;
  size_t wsz = 64 << 20;
  const int8_t *X = nullptr, *W = nullptr;
  int32_t* Y = nullptr;
  size_t M;
  int N, K, algo_index = -1, algos = 0;
  float best_ms = 0;
  void init(cublasLtHandle_t h, const int8_t* X_, const int8_t* W_, int32_t* Y_, size_t M_, int N_, int K_, cudaStream_t st) {
    lt = h; X = X_; W = W_; Y = Y_; M = M_; N = N_; K = K_;
    CK(cudaMalloc(&ws, wsz));
    CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    const cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof Nn));
    CL(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, K, N, K));
    CL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, K, M, K));
    CL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, N, M, N));
    cublasLtMatmulPreference_t pref;
    CL(cublasLtMatmulPreferenceCreate(&pref));
    CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
    cublasLtMatmulHeuristicResult_t res[8];
    int got = 0;
    CL(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, res, &got));
    algos = got;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < got; ++i) {
      algo = res[i].algo;
      std::vector<double> t;
      bool ok = true;
      for (int r = 0; r < 4 && ok; ++r) {
        CK(cudaEventRecord(e0, st));
        ok = run(st) == CUBLAS_STATUS_SUCCESS;
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (r) t.push_back(ms);
      }
      if (ok && (algo_index < 0 || median(t) < best_ms)) { best_ms = (float)median(t); algo_index = i; }
    }
    if (algo_index < 0) { fprintf(stderr, "no cuBLASLt int8 algorithm for %zu x %d x %d\n", M, N, K); exit(3); }
    algo = res[algo_index].algo;
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  }
  cublasStatus_t run(cudaStream_t st) {
    const int32_t one = 1, zero = 0;
    return cublasLtMatmul(lt, op, &one, W, la, X, lb, &zero, Y, lc, Y, lc, &algo, ws, wsz, st);
  }
};

// One SIS level: rows of m bytes in X -> digests in D (rows x n L bytes).
struct Sis {
  Enc e;
  int m, n, K;
  size_t rows;
  const uint8_t* X;
  int8_t* Xs = nullptr;  // the nibble digits (q8n)
  int8_t* dA = nullptr;
  int32_t* Y = nullptr;
  uint8_t* D = nullptr;
  Gemm g;
  void init(cublasLtHandle_t h, const Enc& e_, const int8_t* dA_, const uint8_t* X_, size_t rows_, int m_, int n_,
            cudaStream_t st) {
    e = e_; m = m_; n = n_; rows = rows_; X = X_; K = m * e.split; dA = (int8_t*)dA_;
    if (e.split == 2) CK(cudaMalloc(&Xs, rows * (size_t)K));
    CK(cudaMalloc(&Y, rows * (size_t)e.L * n * 4));
    CK(cudaMalloc(&D, rows * (size_t)n * e.L));
    split(st);
    g.init(h, e.split == 2 ? Xs : (const int8_t*)X, dA, Y, rows, e.L * n, K, st);
  }
  void split(cudaStream_t st) {
    if (e.split == 2) split_nibbles<<<2048, 256, 0, st>>>((const u32*)X, (uint2*)Xs, rows * (size_t)m / 4);
  }
  void run(cudaStream_t st) {
    split(st);
    CL(g.run(st));
    const int blocks = 4096;
    if (e.L == 4) combine<4><<<blocks, 256, 0, st>>>(Y, D, rows, n);
    else if (e.L == 2) combine<2><<<blocks, 256, 0, st>>>(Y, D, rows, n);
    else combine<1><<<blocks, 256, 0, st>>>(Y, D, rows, n);
  }
  size_t dbytes() const { return (size_t)n * e.L; }
};

// Check `samples` rows of a level against the host reference; dump them. Returns mismatching rows.
static int check_level(const Sis& s, const std::vector<int8_t>& A, int samples, const char* outdir, const char* name,
                       int level) {
  const size_t db = s.dbytes();
  std::vector<uint8_t> x(s.m), d(db), want(db);
  std::string base = std::string(outdir) + "/" + name + "-L" + std::to_string(level);
  FILE* fr = fopen((base + "-rows.bin").c_str(), "wb");
  FILE* fd = fopen((base + "-digests.bin").c_str(), "wb");
  FILE* fi = fopen((base + "-index.txt").c_str(), "w");
  int bad = 0;
  const size_t ns = std::min<size_t>(samples, s.rows);
  for (size_t i = 0; i < ns; ++i) {
    const size_t r = ns == s.rows ? i : (i * 2654435761ull + 12345) % s.rows;
    CK(cudaMemcpy(x.data(), s.X + r * s.m, s.m, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(d.data(), s.D + r * db, db, cudaMemcpyDeviceToHost));
    ref_digest(s.e, A, s.n, x.data(), s.m, want.data());
    bad += memcmp(d.data(), want.data(), db) != 0;
    if (fr) fwrite(x.data(), 1, s.m, fr);
    if (fd) fwrite(d.data(), 1, db, fd);
    if (fi) fprintf(fi, "%zu\n", r);
  }
  if (fr) fclose(fr);
  if (fd) fclose(fd);
  if (fi) fclose(fi);
  return bad;
}

static int check_sha(const uint8_t* X, const uint8_t* dig, size_t rows, int m, int samples) {
  std::vector<uint8_t> x(m), d(64), want(64);
  int bad = 0;
  for (int i = 0; i < samples; ++i) {
    const size_t r = (i * 2654435761ull + 777) % rows;
    CK(cudaMemcpy(x.data(), X + r * m, m, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(d.data(), dig + r * 64, 64, cudaMemcpyDeviceToHost));
    sha::digest(x.data(), m, want.data());
    bad += memcmp(d.data(), want.data(), 64) != 0;
  }
  return bad;
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: sis_bench q32|q16|q8n M_BYTES N_WORDS BATCH_MIB REPS OUTDIR [tree]\n");
    return 2;
  }
  const Enc e = enc_of(argv[1]);
  const int m = atoi(argv[2]), n = atoi(argv[3]), reps = atoi(argv[5]);
  const size_t batch = (size_t)atoi(argv[4]) << 20;
  const char* outdir = argv[6];
  bool tree = false, fused = false;
  for (int i = 7; i < argc; ++i) {
    tree |= !strcmp(argv[i], "tree");
    fused |= !strcmp(argv[i], "fused");
  }
  const size_t rows = batch / m;
  const int K = m * e.split;
  char name[96];
  snprintf(name, sizeof name, "%s-m%d-n%d", e.name.c_str(), m, n);

  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cublasLtHandle_t lt; CL(cublasLtCreate(&lt));
  Nv nv;

  std::vector<int8_t> A = derive_A(e, n, K);
  int8_t* dA; CK(cudaMalloc(&dA, A.size()));
  CK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
  uint8_t* X; CK(cudaMalloc(&X, batch));
  fill<<<4096, 256, 0, st>>>(X, batch, 0xC0FFEEu ^ (u32)m);

  Sis s0; s0.init(lt, e, dA, X, rows, m, n, st);
  s0.run(st);
  CK(cudaStreamSynchronize(st));
  int bad = check_level(s0, A, 24, outdir, name, 0);

  // the all-SIS tree: level i's digests, read as rows of m bytes, are level i+1's rows (bytes encodings, m / dbytes a
  // power of two)
  std::vector<Sis> levels;
  std::vector<std::vector<int8_t>> keepA;
  if (tree && e.split == 1 && m % s0.dbytes() == 0) {
    const uint8_t* in = s0.D;
    size_t r = rows * s0.dbytes() / m;
    while (r >= 1) {
      levels.emplace_back();
      levels.back().init(lt, e, dA, in, r, m, n, st);
      levels.back().run(st);
      in = levels.back().D;
      if (r == 1) break;
      r = r * levels.back().dbytes() / m;
    }
    CK(cudaStreamSynchronize(st));
    for (size_t i = 0; i < levels.size(); ++i) bad += check_level(levels[i], A, 8, outdir, name, (int)i + 1);
  }

  // SHA-512 over every row, and over every digest
  uint8_t *shaX, *shaD;
  CK(cudaMalloc(&shaX, rows * 64)); CK(cudaMalloc(&shaD, rows * 64));
  const int tb = 128;
  auto run_shaX = [&]() { row_sha<<<(rows + tb - 1) / tb, tb, 0, st>>>(X, rows, m, shaX); };
  auto run_shaD = [&]() { row_sha<<<(rows + tb - 1) / tb, tb, 0, st>>>(s0.D, rows, (int)s0.dbytes(), shaD); };
  const bool dig_whole = s0.dbytes() % 128 == 0;
  run_shaX();
  if (dig_whole) run_shaD();
  CK(cudaStreamSynchronize(st));
  const int bad_sha = check_sha(X, shaX, rows, m, 16) + (dig_whole ? check_sha(s0.D, shaD, rows, (int)s0.dbytes(), 16) : 0);

  // the fused kernel, held to the cuBLASLt path's digests over the whole batch
  int8_t* dWi = nullptr;
  uint8_t* Df = nullptr;
  long long fused_bad = -1;
  const int NW = e.L * n;
  fused = fused && rows % fz::BM == 0 && NW % fz::BN == 0 && K % fz::BK == 0;
  auto run_fused = [&]() {
    const dim3 grid(NW / fz::BN, (unsigned)(rows / fz::BM));
    if (e.L == 4) fz::sis_fused<4, 1><<<grid, 256, 0, st>>>(X, dWi, Df, m, NW);
    else if (e.L == 2) fz::sis_fused<2, 1><<<grid, 256, 0, st>>>(X, dWi, Df, m, NW);
    else fz::sis_fused<1, 2><<<grid, 256, 0, st>>>(X, dWi, Df, m, NW);
  };
  if (fused) {
    std::vector<int8_t> Wi(A.size());
    for (int l = 0; l < e.L; ++l)
      for (int j = 0; j < n; ++j)
        memcpy(Wi.data() + ((size_t)j * e.L + l) * K, A.data() + ((size_t)l * n + j) * K, K);
    CK(cudaMalloc(&dWi, Wi.size()));
    CK(cudaMemcpy(dWi, Wi.data(), Wi.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Df, rows * s0.dbytes()));
    CK(cudaMemset(Df, 0xA5, rows * s0.dbytes()));
    run_fused();
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(st));
    std::vector<uint8_t> a(rows * s0.dbytes()), b(rows * s0.dbytes());
    CK(cudaMemcpy(a.data(), Df, a.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), s0.D, b.size(), cudaMemcpyDeviceToHost));
    fused_bad = 0;
    for (size_t i = 0; i < a.size(); ++i) fused_bad += a[i] != b[i];
  }

  printf("{\"check\": {\"name\": \"%s\", \"gpu\": \"%s\", \"sm\": %d.%d, \"sis_rows_bad\": %d, \"sha_rows_bad\": %d, "
         "\"tree_levels\": %zu, \"fused_bytes_bad\": %lld}}\n",
         name, prop.name, prop.major, prop.minor, bad, bad_sha, levels.size(), fused_bad);
  fflush(stdout);
  if (bad || bad_sha || fused_bad > 0) return 4;

  const int rthreads = 256, rblocks = prop.multiProcessorCount * 8, chain = 64;
  u64* io; CK(cudaMalloc(&io, (size_t)rthreads * rblocks * 8));
  cudaEvent_t ev[2]; CK(cudaEventCreate(&ev[0])); CK(cudaEventCreate(&ev[1]));
  auto timed = [&](auto f, std::vector<double>& t, std::vector<double>& mhz) {
    CK(cudaEventRecord(ev[0], st));
    f();
    CK(cudaEventRecord(ev[1], st));
    CK(cudaEventSynchronize(ev[1]));
    float ms; CK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
    t.push_back(ms);
    mhz.push_back(nv.sm());
  };
  std::vector<double> t_sis, t_gemm, t_tree, t_shaX, t_shaD, t_rate, mz_sis, mz_gemm, mz_tree, mz_shaX, mz_shaD, mz_rate;
  std::vector<double> t_fused, mz_fused;
  for (int r = 0; r <= reps; ++r) {
    std::vector<double> a, b, c, d, f, g, a1, b1, c1, d1, f1, g1, h, h1;
    if (fused) timed(run_fused, h, h1);
    if (r && fused) { t_fused.push_back(h[0]); mz_fused.push_back(h1[0]); }
    timed([&] { s0.run(st); }, a, a1);
    timed([&] { CL(s0.g.run(st)); }, b, b1);
    if (!levels.empty()) timed([&] { s0.run(st); for (auto& l : levels) l.run(st); }, c, c1);
    timed(run_shaX, d, d1);
    if (dig_whole) timed(run_shaD, f, f1);
    timed([&] { sha_rate<<<rblocks, rthreads, 0, st>>>(io, chain); }, g, g1);
    if (!r) continue;  // warm-up
    t_sis.push_back(a[0]); mz_sis.push_back(a1[0]);
    t_gemm.push_back(b[0]); mz_gemm.push_back(b1[0]);
    if (!c.empty()) { t_tree.push_back(c[0]); mz_tree.push_back(c1[0]); }
    t_shaX.push_back(d[0]); mz_shaX.push_back(d1[0]);
    if (!f.empty()) { t_shaD.push_back(f[0]); mz_shaD.push_back(f1[0]); }
    t_rate.push_back(g[0]); mz_rate.push_back(g1[0]);
  }
  const double gb = (double)rows * m * 1e-6;  // GB in, over ms
  const double macs = (double)rows * K * e.L * n;
  const double comps = (double)rthreads * rblocks * chain;
  printf("{\"bench\": {\"name\": \"%s\", \"gpu\": \"%s\", \"enc\": \"%s\", \"m\": %d, \"n\": %d, \"L\": %d, \"K\": %d, "
         "\"rows\": %zu, \"digest_bytes\": %zu, \"batch_bytes\": %zu, \"gemm_algo\": %d, \"gemm_algos\": %d, "
         "\"sis_ms\": %.5f, \"sis_gb_s\": %.2f, \"gemm_ms\": %.5f, \"gemm_gb_s\": %.2f, \"gemm_tmac_s\": %.1f, "
         "\"tree_ms\": %.5f, \"tree_levels\": %zu, \"sha_rows_ms\": %.5f, \"sha_rows_gb_s\": %.2f, "
         "\"sha_digests_ms\": %.5f, \"rate_gcomp_s\": %.3f, \"rate_gb_s\": %.2f, \"x_sha_rows\": %.3f, "
         "\"mhz_sis\": %.0f, \"mhz_sha\": %.0f, \"mhz_rate\": %.0f, \"fused_ms\": %.5f, \"fused_gb_s\": %.2f, "
         "\"fused_tmac_s\": %.1f, \"mhz_fused\": %.0f, ",
         name, prop.name, e.name.c_str(), m, n, e.L, K, rows, s0.dbytes(), batch, s0.g.algo_index, s0.g.algos,
         median(t_sis), gb / median(t_sis), median(t_gemm), gb / median(t_gemm), macs / median(t_gemm) * 1e-9,
         t_tree.empty() ? 0.0 : median(t_tree), levels.size(), median(t_shaX), gb / median(t_shaX),
         t_shaD.empty() ? 0.0 : median(t_shaD), comps / median(t_rate) * 1e-6, comps * 128 / median(t_rate) * 1e-6,
         median(t_shaX) / median(t_sis), median(mz_sis), median(mz_shaX), median(mz_rate),
         t_fused.empty() ? 0.0 : median(t_fused), t_fused.empty() ? 0.0 : gb / median(t_fused),
         t_fused.empty() ? 0.0 : macs / median(t_fused) * 1e-9, t_fused.empty() ? 0.0 : median(mz_fused));
  printf("\"reps\": {\"fused_ms\": %s, \"mhz_fused\": %s, \"sis_ms\": %s, \"gemm_ms\": %s, \"tree_ms\": %s, \"sha_rows_ms\": %s, \"sha_digests_ms\": %s, "
         "\"rate_ms\": %s, \"mhz_sis\": %s, \"mhz_gemm\": %s, \"mhz_tree\": %s, \"mhz_sha_rows\": %s, \"mhz_sha_digests\": %s, "
         "\"mhz_rate\": %s}}}\n",
         jlist(t_fused).c_str(), jlist(mz_fused).c_str(), jlist(t_sis).c_str(), jlist(t_gemm).c_str(), jlist(t_tree).c_str(), jlist(t_shaX).c_str(), jlist(t_shaD).c_str(),
         jlist(t_rate).c_str(), jlist(mz_sis).c_str(), jlist(mz_gemm).c_str(), jlist(mz_tree).c_str(),
         jlist(mz_shaX).c_str(), jlist(mz_shaD).c_str(), jlist(mz_rate).c_str());
  return 0;
}
