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
  const bool tree = argc > 7 && !strcmp(argv[7], "tree");
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

  printf("{\"check\": {\"name\": \"%s\", \"gpu\": \"%s\", \"sm\": %d.%d, \"sis_rows_bad\": %d, \"sha_rows_bad\": %d, "
         "\"tree_levels\": %zu}}\n",
         name, prop.name, prop.major, prop.minor, bad, bad_sha, levels.size());
  fflush(stdout);
  if (bad || bad_sha) return 4;

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
  for (int r = 0; r <= reps; ++r) {
    std::vector<double> a, b, c, d, f, g, a1, b1, c1, d1, f1, g1;
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
         "\"mhz_sis\": %.0f, \"mhz_sha\": %.0f, \"mhz_rate\": %.0f, ",
         name, prop.name, e.name.c_str(), m, n, e.L, K, rows, s0.dbytes(), batch, s0.g.algo_index, s0.g.algos,
         median(t_sis), gb / median(t_sis), median(t_gemm), gb / median(t_gemm), macs / median(t_gemm) * 1e-9,
         t_tree.empty() ? 0.0 : median(t_tree), levels.size(), median(t_shaX), gb / median(t_shaX),
         t_shaD.empty() ? 0.0 : median(t_shaD), comps / median(t_rate) * 1e-6, comps * 128 / median(t_rate) * 1e-6,
         median(t_shaX) / median(t_sis), median(mz_sis), median(mz_shaX), median(mz_rate));
  printf("\"reps\": {\"sis_ms\": %s, \"gemm_ms\": %s, \"tree_ms\": %s, \"sha_rows_ms\": %s, \"sha_digests_ms\": %s, "
         "\"rate_ms\": %s, \"mhz_sis\": %s, \"mhz_gemm\": %s, \"mhz_tree\": %s, \"mhz_sha_rows\": %s, \"mhz_sha_digests\": %s, "
         "\"mhz_rate\": %s}}}\n",
         jlist(t_sis).c_str(), jlist(t_gemm).c_str(), jlist(t_tree).c_str(), jlist(t_shaX).c_str(), jlist(t_shaD).c_str(),
         jlist(t_rate).c_str(), jlist(mz_sis).c_str(), jlist(mz_gemm).c_str(), jlist(mz_tree).c_str(),
         jlist(mz_shaX).c_str(), jlist(mz_shaD).c_str(), jlist(mz_rate).c_str());
  return 0;
}
