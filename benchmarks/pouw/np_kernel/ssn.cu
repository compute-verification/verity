// ssn_1l on one H200 (sm_90a): the served sparse-noise GEMM with every k32 accumulator word exported and hm96-hashed.
//
// Layout (K data entries, S = 1 + ceil((K - 28) / 31) k32 steps; lanes are a step's 32 K bytes):
//   A'  step 1: lanes 0-27 A[i, 0..27], lanes 28-31 a1[i, 0..3] in {0,1}; step t >= 2: lanes 0-30 the next 31 entries of A,
//       lane 31 a_t[i] in {0,1}; data lanes past K are 0. The data lanes are written by A's producer with the noise lanes 0;
//       the forming pass (form_k) ORs the noise bits into the noise registers (one 32-bit register a row a step).
//   B'  step 1: lanes 0-27 B[0..27, j], lanes 28-31 the public code C[0..3, j] (base-13 digits of c_j minus 6, c_j = j + 1
//       below 14,280 and j + 2 from there); step t >= 2: lanes 0-30 the next 31 rows of B, lane 31 = 1. Formed once at load.
//   The accumulator starts at -beta_j, beta_j = rint((sum_e C[e, j] + S - 1) / 2) (half to even), so word (i, j, t) is
//   D_t(i, j) - beta_j + sum_e a1[i, e] C[e, j] + sum_{2 <= t' <= t} a_t'[i], and the last is ssn-1l-b1-v1's P'.
// Coins: a row's S + 3 bits are bits 0..S+2 of its H = ceil((S + 3) / 32) coin words (a1[i, e] bit e, a_t[i] bit t + 2);
// row i's words are coin words i H .. i H + H - 1 of the call's stream, whose 16-word block c is SHA-512(seed || be64 c),
// seed = SHA-512 of the call's 224-byte seed message (tag, salt, forward, position, root(A_j), root(B_j)).
// Operands are tiled: A' tile (64-row block, step) and B' tile (BN-column block, step) are contiguous, in dc.cuh's K-major
// layout, so a step's tiles are one linear copy. Exported words: a block (row tile, column tile, step) of 64 x BN s32 in
// wgmma's fragment order (uint4 j * 128 + thread holds the thread's d[4 j .. 4 j + 3]); the hash pass (Hasher) hashes them
// as the served committer's rows or as served_zk's row-seg rows.
//
//   ssn check                 every exported word, the final outputs and the formed registers against the host reference
//                             on small shapes (random, extreme and zero data; columns past the code's skip; S = 1 and a
//                             partial last step)
//   ssn time REPS M [BUDGET_GB [ROW_BYTES ...]]
//                             Llama-3.2-1B's fused linears at M rows, one JSON line a timing (see time_all), the hash pass
//                             by each ROW_BYTES (0: row-seg; default 0 65536 16384)
// Build (from the tree's root): nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a benchmarks/pouw/np_kernel/ssn.cu
//   integrations/vllm/verity_vllm/commit/committer/native_rows.cu -o ssn
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../dc/dc.cuh"
#include "../dc/wgmma.cuh"
#include "../served_zk/hm96_rows.cu"

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

typedef uint32_t u32;
typedef uint64_t u64;

constexpr int KB = 32, TA = 64 * KB, STAGES = 4;

__host__ __device__ inline int steps_of(int k) { return k <= 28 ? 1 : 1 + (k - 28 + 30) / 31; }
__host__ __device__ inline int coin_words(int s) { return (s + 3 + 31) / 32; }

struct Clk {
  long long c0, c1;
  u64 t0, t1;
};
__device__ __forceinline__ u64 gtime() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void cp16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(s)), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template <int BN> __device__ __forceinline__ void mma(int (&d)[BN / 2], u64 da, u64 db);
template <> __device__ __forceinline__ void mma<256>(int (&d)[128], u64 da, u64 db) { wgmma_s8_n256(d, da, db, 1); }
template <> __device__ __forceinline__ void mma<128>(int (&d)[64], u64 da, u64 db) { wgmma_s8_n128(d, da, db, 1); }

// One CTA (a warpgroup) a 64 x BN output tile over S steps, STAGES-deep cp.async ring. EXPORT: wait for each step's wgmma
// and store its 64 x BN words (block (rt, ct, t) of W); otherwise one wgmma group stays in flight (the plain GEMM).
template <int BN, bool EXPORT>
__global__ void __launch_bounds__(128) ssn_k(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                             const int* __restrict__ beta, int S, int n, uint4* __restrict__ W,
                                             int* __restrict__ out, Clk* clk) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int TB = BN * KB, ST = TA + TB;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid = threadIdx.x, ct = blockIdx.x, rt = blockIdx.y, nct = gridDim.x;
  const uint8_t* a0 = A + (size_t)rt * S * TA;
  const uint8_t* b0 = B + (size_t)ct * S * TB;
  auto load = [&](int t) {
    if (t < S) {
      unsigned char* st = sm + (t % STAGES) * ST;
      cp16(st + 16 * tid, a0 + (size_t)t * TA + 16 * tid);
#pragma unroll
      for (int j = 0; j < TB / 16 / 128; j++) cp16(st + TA + 16 * (tid + 128 * j), b0 + (size_t)t * TB + 16 * (tid + 128 * j));
    }
    cp_commit();
  };
#pragma unroll
  for (int t = 0; t < STAGES - 1; t++) load(t);
  const int w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  int d[BN / 2];
#pragma unroll
  for (int ni = 0; ni < BN / 8; ni++)
#pragma unroll
    for (int e = 0; e < 4; e++) d[4 * ni + e] = beta ? -beta[ct * BN + 8 * ni + 2 * q + (e & 1)] : 0;
  wgmma_hold(d);
  const long long c0 = clock64();
  const u64 t0 = gtime();
  uint4* wb = W + (size_t)(rt * nct + ct) * S * (BN * 16);
#pragma unroll 1
  for (int t = 0; t < S; t++) {
    cp_wait<STAGES - 2>();
    fence_async_smem();
    __syncthreads();
    unsigned char* st = sm + (t % STAGES) * ST;
    const u64 da = wgmma_desc(smem_u32(st), kmaj_lbo(), kmaj_sbo(KB));
    const u64 db = wgmma_desc(smem_u32(st + TA), kmaj_lbo(), kmaj_sbo(KB));
    wgmma_fence();
    mma<BN>(d, da, db);
    wgmma_commit();
    if constexpr (EXPORT) {
      wgmma_wait<0>();
      wgmma_hold(d);
      uint4* r = wb + (size_t)t * (BN * 16);
#pragma unroll
      for (int j = 0; j < BN / 8; j++) r[j * 128 + tid] = make_uint4(d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
    } else {
      wgmma_wait<1>();
    }
    load(t + STAGES - 1);
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  if (clk && rt == 0 && ct == 0 && tid == 0) *clk = Clk{c0, clock64(), t0, gtime()};
  if (out)
#pragma unroll
    for (int ni = 0; ni < BN / 8; ni++)
#pragma unroll
      for (int e = 0; e < 4; e++)
        out[(size_t)(64 * rt + 16 * w + g + 8 * (e >> 1)) * n + ct * BN + 8 * ni + 2 * q + (e & 1)] = d[4 * ni + e];
#endif
}

// The call's coin stream: thread b derives the seed from the 256-byte seed message (two blocks; the second carries the
// padding) and writes block b = SHA-512(seed || be64 b), 16 coin words (word 2 j the high half of state word j).
__global__ void coins_k(const u64* __restrict__ msg, u32* __restrict__ coins, long long blocks) {
  const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  u64 st[8], w[16];
  s512_iv(st);
  for (int j = 0; j < 16; j++) w[j] = msg[j];
  s512_compress(st, w);
  for (int j = 0; j < 16; j++) w[j] = msg[16 + j];
  s512_compress(st, w);
  u64 seed[8];
  for (int j = 0; j < 8; j++) seed[j] = st[j];
  s512_iv(st);
  for (int j = 0; j < 8; j++) w[j] = seed[j];
  w[8] = (u64)b;
  w[9] = 0x8000000000000000ull;
  for (int j = 10; j < 15; j++) w[j] = 0;
  w[15] = 72 * 8;
  s512_compress(st, w);
  uint4* o = reinterpret_cast<uint4*>(coins + 16 * b);
  for (int j = 0; j < 4; j++)
    o[j] = make_uint4((u32)(st[2 * j] >> 32), (u32)st[2 * j], (u32)(st[2 * j + 1] >> 32), (u32)st[2 * j + 1]);
}

// The forming pass: one thread a noise register (row i, step t), rows fastest. Step 1's register is a1 spread over its four
// bytes (one IMAD and one LOP3); step t's ORs bit t + 2 of the row's stream into byte 3 (a funnel shift and one LOP3).
__global__ void form_k(uint8_t* __restrict__ A, const u32* __restrict__ coins, int m, int S) {
  const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= (long long)m * S) return;
  const int i = (int)(x % m), t = (int)(x / m), H = coin_words(S);
  u32* reg = reinterpret_cast<u32*>(A + ((size_t)(i >> 6) * S + t) * TA + kmaj_off(i & 63, 28, KB));
  const u32* h = coins + (size_t)i * H;
  if (t == 0) {
    *reg = ((h[0] & 0xFu) * 0x00204081u) & 0x01010101u;
  } else {
    const int p = t + 3;
    *reg |= __funnelshift_r(h[p >> 5], h[p >> 5], (p + 8) & 31) & 0x01000000u;
  }
}

__global__ void clock_k(Clk* clk, long long cycles) {
  const long long c0 = clock64();
  const u64 t0 = gtime();
  while (clock64() - c0 < cycles) {
  }
  *clk = Clk{c0, clock64(), t0, gtime()};
}

// ---- host ------------------------------------------------------------------------------------------------------------------
namespace hsha {
static const u64 K[80] = {
    0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull,
    0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull, 0x12835b0145706fbeull,
    0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull, 0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull,
    0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
    0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull, 0x983e5152ee66dfabull,
    0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,
    0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull,
    0x53380d139d95b3dfull, 0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
    0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull,
    0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull, 0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull,
    0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull,
    0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
    0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull, 0xca273eceea26619cull,
    0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull,
    0x113f9804bef90daeull, 0x1b710b35131c471bull, 0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull,
    0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull};
static u64 rotr(u64 x, int n) { return (x >> n) | (x << (64 - n)); }
static void iv(u64 h[8]) {
  const u64 v[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                    0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  for (int i = 0; i < 8; i++) h[i] = v[i];
}
static void compress(u64 h[8], const u64 m[16]) {
  u64 w[80], a[8];
  for (int i = 0; i < 16; i++) w[i] = m[i];
  for (int i = 16; i < 80; i++) {
    const u64 s0 = rotr(w[i - 15], 1) ^ rotr(w[i - 15], 8) ^ (w[i - 15] >> 7);
    const u64 s1 = rotr(w[i - 2], 19) ^ rotr(w[i - 2], 61) ^ (w[i - 2] >> 6);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  for (int i = 0; i < 8; i++) a[i] = h[i];
  for (int i = 0; i < 80; i++) {
    const u64 t1 = a[7] + (rotr(a[4], 14) ^ rotr(a[4], 18) ^ rotr(a[4], 41)) + ((a[4] & a[5]) ^ (~a[4] & a[6])) + K[i] + w[i];
    const u64 t2 = (rotr(a[0], 28) ^ rotr(a[0], 34) ^ rotr(a[0], 39)) + ((a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]));
    for (int j = 7; j > 0; j--) a[j] = a[j - 1];
    a[4] += t1;
    a[0] = t1 + t2;
  }
  for (int i = 0; i < 8; i++) h[i] += a[i];
}
}  // namespace hsha

static u64 mix64(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
static int code_of(int j, int e) {
  long long c = j + 1 < 14280 ? j + 1 : j + 2;
  for (int r = 0; r < e; r++) c /= 13;
  return (int)(c % 13) - 6;
}
static int beta_of(int j, int S) {
  int s = S - 1;
  for (int e = 0; e < 4; e++) s += code_of(j, e);
  return (int)std::nearbyint(s / 2.0);
}
// data entry of step t (0-based), lane l: its K index, or -1 for a noise or padding lane
static int kidx(int t, int l, int K) {
  const int k = t == 0 ? (l < 28 ? l : -1) : (l < 31 ? 28 + 31 * (t - 1) + l : -1);
  return k >= 0 && k < K ? k : -1;
}

struct Lin {
  int m, n, K, S, BN;
  std::vector<int8_t> A, B;  // A[i * K + k], B[j * K + k] (B transposed: a column's K entries contiguous)
};
static void fill(Lin& L, int kind, u64 seed) {
  L.A.resize((size_t)L.m * L.K);
  L.B.resize((size_t)L.n * L.K);
  auto v = [&](u64 x) -> int8_t {
    if (kind == 1) return 127;
    if (kind == 2) return -127;
    if (kind == 3) return 0;
    return (int8_t)((int)(mix64(x) % 255) - 127);
  };
  for (size_t i = 0; i < L.A.size(); i++) L.A[i] = kind == 2 ? (int8_t)-127 : v(seed * 1000003 + i);
  for (size_t i = 0; i < L.B.size(); i++) L.B[i] = kind == 2 ? (int8_t)127 : v(~seed * 7919 + i);
}
// A' (data lanes, noise lanes 0) and B' in the tiled layout
static std::vector<uint8_t> tile_a(const Lin& L) {
  std::vector<uint8_t> o((size_t)L.m * L.S * KB, 0);
  for (int i = 0; i < L.m; i++)
    for (int t = 0; t < L.S; t++)
      for (int l = 0; l < KB; l++) {
        const int k = kidx(t, l, L.K);
        if (k >= 0) o[((size_t)(i >> 6) * L.S + t) * TA + kmaj_off(i & 63, l, KB)] = (uint8_t)L.A[(size_t)i * L.K + k];
      }
  return o;
}
static std::vector<uint8_t> tile_b(const Lin& L) {
  std::vector<uint8_t> o((size_t)L.n * L.S * KB, 0);
  const size_t TB = (size_t)L.BN * KB;
  for (int j = 0; j < L.n; j++)
    for (int t = 0; t < L.S; t++)
      for (int l = 0; l < KB; l++) {
        const int k = kidx(t, l, L.K);
        int v = k >= 0 ? L.B[(size_t)j * L.K + k] : 0;
        if (t == 0 && l >= 28) v = code_of(j, l - 28);
        if (t > 0 && l == 31) v = 1;
        o[((size_t)(j / L.BN) * L.S + t) * TB + kmaj_off(j % L.BN, l, KB)] = (uint8_t)(int8_t)v;
      }
  return o;
}
// the stock-shaped operands: 32 data lanes a step, no noise (the plain GEMM at the linear's own K)
static std::vector<uint8_t> tile_plain(const std::vector<int8_t>& X, int rows, int K, int R) {
  const int S0 = K / KB;
  std::vector<uint8_t> o((size_t)rows * K, 0);
  for (int i = 0; i < rows; i++)
    for (int t = 0; t < S0; t++)
      for (int l = 0; l < KB; l++)
        o[((size_t)(i / R) * S0 + t) * R * KB + kmaj_off(i % R, l, KB)] = (uint8_t)X[(size_t)i * K + KB * t + l];
  return o;
}
template <class T>
static T* up(const std::vector<T>& v) {
  T* p;
  CK(cudaMalloc(&p, v.size() * sizeof(T)));
  CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
static std::vector<u64> seed_msg(u64 call) {
  std::vector<u64> m(32, 0);
  for (int j = 0; j < 28; j++) m[j] = mix64(call * 64 + j);  // tag, salt, forward, position, root(A_j), root(B_j): 224 bytes
  m[28] = 0x8000000000000000ull;
  m[31] = 224 * 8;
  return m;
}
static long long coin_blocks(int m, int S) { return ((long long)m * coin_words(S) + 15) / 16; }

template <int BN, bool EXPORT>
static void gemm_bn(const uint8_t* A, const uint8_t* B, const int* beta, int m, int n, int S, uint4* W, int* out, Clk* clk,
                    cudaStream_t st) {
  constexpr int smem = STAGES * (TA + BN * KB);
  static bool init = false;
  if (!init) {
    CK(cudaFuncSetAttribute(ssn_k<BN, EXPORT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    init = true;
  }
  ssn_k<BN, EXPORT><<<dim3(n / BN, m / 64), 128, smem, st>>>(A, B, beta, S, n, W, out, clk);
  CK(cudaGetLastError());
}
static void gemm(int BN, bool ex, const uint8_t* A, const uint8_t* B, const int* beta, int m, int n, int S, uint4* W, int* out,
                 Clk* clk, cudaStream_t st) {
  if (BN == 256 && ex) gemm_bn<256, true>(A, B, beta, m, n, S, W, out, clk, st);
  else if (BN == 256) gemm_bn<256, false>(A, B, beta, m, n, S, W, out, clk, st);
  else if (ex) gemm_bn<128, true>(A, B, beta, m, n, S, W, out, clk, st);
  else gemm_bn<128, false>(A, B, beta, m, n, S, W, out, clk, st);
}
static void coins_form(const u64* msg, u32* coins, uint8_t* A, int m, int S, cudaStream_t st, bool coins_on = true,
                       bool form_on = true) {
  const long long nb = coin_blocks(m, S);
  if (coins_on) coins_k<<<(unsigned)((nb + 127) / 128), 128, 0, st>>>(msg, coins, nb);
  const long long r = (long long)m * S;
  if (form_on) form_k<<<(unsigned)((r + 255) / 256), 256, 0, st>>>(A, coins, m, S);
  CK(cudaGetLastError());
}

// native_rows.cu (the served committer, linked as its own translation unit): hm96-sha512/row/v2 leaves of fixed-size rows
extern "C" int verity_rows_params_bytes();
extern "C" int verity_rows_build(const uint8_t* domain_id, const uint8_t* ks, const uint8_t* row_prefix, const uint8_t* salt_prefix,
                                 const uint8_t* leaf_prefix, const char* schema, int schema_len, uint32_t lb, void* out);
extern "C" int verity_rows_leaves_sf_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows,
                                            const uint8_t* salts, const uint8_t* tn, uint64_t first, uint64_t m, uint64_t n_total,
                                            uint8_t* out, uint8_t* bc, uint8_t* hs, uint8_t* scratch, uint64_t max_blocks,
                                            cudaStream_t st);
extern "C" int verity_rows_levels_launch(const void* params, const uint8_t* in, uint8_t* out, uint64_t n_in, int level,
                                         uint64_t base, int n_levels, cudaStream_t st);

// The hash pass over a run of exported words. rb 0: served_zk's row-seg rows (sz_segments), one row an exported block: its
// 1 KiB segments, then its segment digests. rb > 0: the served committer's hm96-sha512/row/v2 leaves of rb-byte rows over
// the words (salts derived, the split form), one thread a row. The constants (prefixes, key, nibble table) are stand-ins of
// the scheme's lengths: they set no compression count.
struct Hasher {
  int rb = 0;
  u64* mids = nullptr;
  uint8_t *dig = nullptr, *dig2 = nullptr, *tn = nullptr, *leaves = nullptr, *scratch = nullptr, *lv = nullptr;
  std::vector<uint8_t> params;
  void init(size_t max_bytes, int row_bytes) {
    rb = row_bytes;
    if (rb) {
      std::vector<uint8_t> c(1024);
      for (size_t i = 0; i < c.size(); i++) c[i] = (uint8_t)mix64(i + 99);
      params.resize(verity_rows_params_bytes());
      const char* schema = "hm96-sha512/row/v2";
      if (verity_rows_build(&c[0], &c[64], &c[128], &c[256], &c[384], schema, (int)strlen(schema), (uint32_t)rb, params.data())) {
        fprintf(stderr, "ERR verity_rows_build\n");
        exit(2);
      }
      std::vector<uint8_t> t(256 * 17 * 16);
      for (size_t i = 0; i < t.size(); i++) t[i] = (uint8_t)mix64(i + 5);
      tn = up(t);
      const size_t rows = max_bytes / rb + 2;
      CK(cudaMalloc(&leaves, rows * 64));
      CK(cudaMalloc(&lv, rows * 64));
      CK(cudaMalloc(&scratch, rows * 192));
      return;
    }
    std::vector<uint32_t> pre(64 * 32);
    for (size_t i = 0; i < pre.size(); i++) pre[i] = (u32)mix64(i + 77);
    uint32_t* dpre = up(pre);
    CK(cudaMalloc(&mids, 64 * 8 * sizeof(u64)));
    Args a{};
    a.p[0] = (unsigned long long)dpre;
    a.p[1] = (unsigned long long)mids;
    a.i[0] = 64;
    sz_mid<<<1, 64>>>(a);
    CK(cudaGetLastError());
    CK(cudaMalloc(&dig, max_bytes / 16 + 64));
    CK(cudaMalloc(&dig2, max_bytes / 256 + 64));
    CK(cudaFuncSetAttribute(sz_segments, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 32 * 33 * 4));
  }
  void seg(const void* data, long long rows, int stride, int s, uint8_t* o, cudaStream_t st) {
    Args a{};
    a.p[0] = (unsigned long long)data;
    a.p[1] = (unsigned long long)mids;
    a.p[2] = (unsigned long long)o;
    a.i[0] = (int)rows;
    a.i[1] = stride;
    a.i[2] = s;
    a.i[3] = 1024;
    const long long total = rows * s;
    sz_segments<<<(unsigned)((total + 255) / 256), 256, 8 * 32 * 33 * 4, st>>>(a);
    CK(cudaGetLastError());
  }
  void run(const uint4* W, long long blocks, int block_bytes, cudaStream_t st, u64 cap = 0) {
    if (rb) {
      const u64 n = ((u64)blocks * block_bytes + rb - 1) / rb;
      const int rc = verity_rows_leaves_sf_launch(params.data(), (const uint8_t*)W, rb, n, nullptr, tn, 0, n, n, leaves, nullptr,
                                                  nullptr, scratch, cap, st);
      if (rc) {
        fprintf(stderr, "ERR verity_rows_leaves_sf_launch %d\n", rc);
        exit(2);
      }
      return;
    }
    const int s = block_bytes / 1024;
    seg(W, blocks, block_bytes, s, dig, st);
    seg(dig, blocks, s * 64, s * 64 / 1024, dig2, st);
  }
  // A root over `bytes` (a multiple of 64 KiB). rb > 0: the leaves, then the frame tree's nodes above them, up to ten levels
  // a launch (the leaf count a power of two, so no pads). Row-seg: its segment and block digests, with no tree above them.
  void root(const uint8_t* data, u64 bytes, cudaStream_t st) {
    run((const uint4*)data, (long long)(bytes / 65536), 65536, st);
    if (!rb) return;
    u64 n = bytes / rb;
    if (n & (n - 1)) {
      fprintf(stderr, "ERR root over %llu leaves: not a power of two\n", (unsigned long long)n);
      exit(2);
    }
    const uint8_t* in = leaves;
    uint8_t* out = lv;
    for (int level = 0; n > 1;) {
      int L = 0;
      while (L < 10 && (n >> L) > 1) L++;
      const int rc = verity_rows_levels_launch(params.data(), in, out, n, level, 0, L, st);
      if (rc) {
        fprintf(stderr, "ERR verity_rows_levels_launch %d\n", rc);
        exit(2);
      }
      in = out + 64 * (n - (n >> (L - 1)));
      out += 64 * (n - (n >> L));
      n >>= L;
      level += L;
    }
  }
};

// word (i, j, t) of an exported run: block (i / 64, j / BN, t), thread 32 w + 4 g + q, register 4 ni + e
static size_t word_at(int i, int j, int t, int S, int n, int BN) {
  const int r = i & 63, c = j % BN, w = r >> 4, g = r & 7, eh = (r >> 3) & 1, ni = c >> 3, q = (c & 7) >> 1, el = c & 1;
  const size_t blk = ((size_t)(i >> 6) * (n / BN) + j / BN) * S + t;
  return blk * (64 * (size_t)BN) + ((size_t)ni * 128 + 32 * w + 4 * g + q) * 4 + 2 * eh + el;
}

static int check_one(int m, int n, int K, int BN, int kind, u64 seed) {
  Lin L{m, n, K, steps_of(K), BN, {}, {}};
  fill(L, kind, seed);
  const int S = L.S, H = coin_words(S);
  std::vector<int> beta(n);
  for (int j = 0; j < n; j++) beta[j] = beta_of(j, S);
  const auto ta = tile_a(L);
  uint8_t* dA = up(ta);
  uint8_t* dB = up(tile_b(L));
  int* dbeta = up(beta);
  const std::vector<u64> msg = seed_msg(seed);
  u64* dmsg = up(msg);
  u32* dcoins;
  const long long nb = coin_blocks(m, S);
  CK(cudaMalloc(&dcoins, nb * 64));
  const size_t words = (size_t)m * n * S;
  uint4* dW;
  int* dout;
  CK(cudaMalloc(&dW, words * 4));
  CK(cudaMalloc(&dout, (size_t)m * n * 4));
  CK(cudaMemset(dW, 0xA5, words * 4));
  CK(cudaMemset(dout, 0xA5, (size_t)m * n * 4));
  coins_form(dmsg, dcoins, dA, m, S, 0);
  gemm(BN, true, dA, dB, dbeta, m, n, S, dW, dout, nullptr, 0);
  CK(cudaDeviceSynchronize());
  std::vector<u32> coins(nb * 16);
  std::vector<int> W(words), out((size_t)m * n);
  std::vector<uint8_t> fa(ta.size());
  CK(cudaMemcpy(coins.data(), dcoins, nb * 64, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(W.data(), dW, words * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(out.data(), dout, (size_t)m * n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(fa.data(), dA, ta.size(), cudaMemcpyDeviceToHost));
  auto bit = [&](int i, int p) { return (int)((coins[(size_t)i * H + (p >> 5)] >> (p & 31)) & 1u); };
  // every coin block against a host SHA-512: seed = SHA-512(seed message), block b = SHA-512(seed || be64 b)
  size_t bad_coin = 0;
  {
    u64 sd[8], st[8], w[16];
    hsha::iv(sd);
    hsha::compress(sd, &msg[0]);
    hsha::compress(sd, &msg[16]);
    for (long long b = 0; b < nb; b++) {
      for (int j = 0; j < 8; j++) w[j] = sd[j];
      w[8] = (u64)b;
      w[9] = 0x8000000000000000ull;
      for (int j = 10; j < 15; j++) w[j] = 0;
      w[15] = 72 * 8;
      hsha::iv(st);
      hsha::compress(st, w);
      for (int j = 0; j < 8; j++)
        bad_coin += (coins[16 * b + 2 * j] != (u32)(st[j] >> 32)) + (coins[16 * b + 2 * j + 1] != (u32)st[j]);
    }
  }
  // the formed A' against the spec: data lanes unchanged, noise lanes the row's bits
  size_t bad_form = 0;
  for (int i = 0; i < m; i++)
    for (int t = 0; t < S; t++)
      for (int l = 0; l < KB; l++) {
        const int k = kidx(t, l, K);
        int v = k >= 0 ? L.A[(size_t)i * K + k] : 0;
        if (t == 0 && l >= 28) v = bit(i, l - 28);
        if (t > 0 && l == 31) v = bit(i, t + 3);
        bad_form += (int8_t)fa[((size_t)(i >> 6) * S + t) * TA + kmaj_off(i & 63, l, KB)] != v;
      }
  // every word against word (i, j, t) = D_t(i, j) - beta_j + sum_e a1[i, e] C[e, j] + sum_{2 <= t' <= t} a_t'[i]
  size_t bad_w = 0, bad_o = 0, ones = 0;
  long long maxabs = 0;
  std::vector<int> code(4 * (size_t)n);
  for (int j = 0; j < n; j++)
    for (int e = 0; e < 4; e++) code[4 * j + e] = code_of(j, e);
  for (int i = 0; i < m; i++) {
    const int8_t* a = &L.A[(size_t)i * K];
    for (int j = 0; j < n; j++) {
      const int8_t* b = &L.B[(size_t)j * K];
      long long acc = -beta[j];
      for (int e = 0; e < 4; e++) acc += bit(i, e) * code[4 * j + e];
      for (int t = 0; t < S; t++) {
        const int lo = t == 0 ? 0 : 28 + 31 * (t - 1), hi = std::min(K, t == 0 ? 28 : lo + 31);
        for (int k = lo; k < hi; k++) acc += (int)a[k] * (int)b[k];
        if (t > 0) acc += bit(i, t + 3);
        maxabs = std::max(maxabs, acc < 0 ? -acc : acc);
        bad_w += W[word_at(i, j, t, S, n, BN)] != acc;
      }
      bad_o += out[(size_t)i * n + j] != (int)acc;
    }
    for (int t = 1; t < S; t++) ones += bit(i, t + 3);
  }
  printf("{\"check\": \"ssn_1l\", \"m\": %d, \"n\": %d, \"K\": %d, \"S\": %d, \"BN\": %d, \"data\": \"%s\", \"words\": %zu, "
         "\"word_mismatches\": %zu, \"final_mismatches\": %zu, \"formed_mismatches\": %zu, \"max_abs_word\": %lld, "
         "\"step_bits_set\": %zu, \"step_bits\": %zu, \"coin_blocks\": %lld, \"coin_word_mismatches\": %zu}\n",
         m, n, K, S, BN, kind == 0 ? "random" : kind == 1 ? "max" : kind == 2 ? "min" : "zero", words, bad_w, bad_o, bad_form,
         maxabs, ones, (size_t)m * (S - 1), nb, bad_coin);
  fflush(stdout);
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dbeta)); CK(cudaFree(dmsg)); CK(cudaFree(dcoins));
  CK(cudaFree(dW)); CK(cudaFree(dout));
  return bad_w || bad_o || bad_form || bad_coin;
}

static int check() {
  {
    u64 h[8], m[16] = {0x6162638000000000ull, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24};
    hsha::iv(h);
    hsha::compress(h, m);
    if (h[0] != 0xddaf35a193617abaull || h[7] != 0x2a9ac94fa54ca49full) {
      fprintf(stderr, "ERR host SHA-512 of \"abc\"\n");
      return 1;
    }
  }
  struct C { int m, n, K, BN, kind; };
  const C cs[] = {{128, 512, 2048, 256, 0}, {128, 512, 2048, 128, 0}, {64, 256, 8192, 256, 0}, {64, 512, 100, 256, 0},
                  {64, 256, 28, 256, 0},    {64, 256, 59, 128, 0},    {64, 16384, 90, 256, 0}, {64, 256, 8192, 256, 1},
                  {64, 256, 8192, 256, 2},  {128, 256, 2048, 256, 3}, {192, 768, 2048, 256, 0}};
  int bad = 0;
  u64 seed = 1;
  for (const C& c : cs) bad += check_one(c.m, c.n, c.K, c.BN, c.kind, seed++);
  return bad ? 1 : 0;
}

// ---- timing ----------------------------------------------------------------------------------------------------------------
static int g_sms = 132;
struct Timer {
  cudaEvent_t a, b;
  Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
  template <class F> double ms(int reps, F f, cudaStream_t st = 0) {
    f();
    CK(cudaDeviceSynchronize());
    std::vector<double> v;
    for (int r = 0; r < reps; r++) {
      CK(cudaEventRecord(a, st));
      f();
      CK(cudaEventRecord(b, st));
      CK(cudaEventSynchronize(b));
      float t = 0;
      CK(cudaEventElapsedTime(&t, a, b));
      v.push_back(t);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  }
};
static double sm_mhz(Clk* dclk) {
  clock_k<<<1, 1>>>(dclk, 20000000);
  CK(cudaDeviceSynchronize());
  Clk c;
  CK(cudaMemcpy(&c, dclk, sizeof c, cudaMemcpyDeviceToHost));
  return (double)(c.c1 - c.c0) / (double)(c.t1 - c.t0) * 1e3;
}
static double units(double ms, double mhz) { return ms * 1e-3 * mhz * 1e6 * g_sms * 4096.0; }

struct LinDev {
  const char* name;
  int n, K, S, BN, chunk;  // chunk: rows a hash chunk
  uint8_t *A, *B, *A0, *B0;
  int* beta;
  u64* msg;
  u32* coins;
};

// One JSON line a measurement, at M rows: per linear the plain GEMM at its own K, ssn_1l's GEMM without and with export,
// the hash pass over the exported words, coins, forming and coins + forming as one call's latency; then the forward's
// 16 layers end to end, serial (each chunk's GEMM with export, then its hash, one stream) and beside (chunk c's hash on
// buffer c mod B's stream while the GEMM stream runs on), with coins and forming before each linear's GEMM.
static int time_all(int reps, int M, double budget_gb, const std::vector<int>& rbs) {
  struct Spec { const char* name; int n, K; };
  const Spec specs[] = {{"qkv", 3072, 2048}, {"o", 2048, 2048}, {"gate_up", 16384, 2048}, {"down", 2048, 8192}};
  Clk* dclk;
  CK(cudaMalloc(&dclk, sizeof(Clk)));
  std::vector<LinDev> lins;
  size_t max_chunk_words = 0;
  const size_t budget = (size_t)(budget_gb * 1e9);
  for (const Spec& s : specs) {
    Lin L{M, s.n, s.K, steps_of(s.K), ((M / 64) * (s.n / 256) < g_sms) ? 128 : 256, {}, {}};
    fill(L, 0, 11 + s.n + s.K);
    LinDev d{s.name, s.n, s.K, L.S, L.BN, 0};
    d.A = up(tile_a(L));
    d.B = up(tile_b(L));
    d.A0 = up(tile_plain(L.A, M, s.K, 64));
    d.B0 = up(tile_plain(L.B, s.n, s.K, L.BN));
    std::vector<int> beta(s.n);
    for (int j = 0; j < s.n; j++) beta[j] = beta_of(j, L.S);
    d.beta = up(beta);
    d.msg = up(seed_msg(s.n));
    CK(cudaMalloc(&d.coins, coin_blocks(M, L.S) * 64));
    const size_t row_bytes = (size_t)s.n * L.S * 4;
    d.chunk = (int)std::min<size_t>(M, std::max<size_t>(64, budget / row_bytes / 64 * 64));
    max_chunk_words = std::max(max_chunk_words, (size_t)d.chunk * s.n * L.S);
    lins.push_back(d);
  }
  // beside's buffers, each with its own hash stream: as many as fit in 48 GB, 2 to 8
  const int nbuf = (int)std::max<size_t>(2, std::min<size_t>(8, (size_t)48e9 / (max_chunk_words * 4)));
  std::vector<uint4*> Wb(nbuf);
  for (auto& w : Wb) CK(cudaMalloc(&w, max_chunk_words * 4 + 2 * 65536));
  std::vector<Hasher> hsv(nbuf * rbs.size());
  for (size_t h = 0; h < hsv.size(); h++) hsv[h].init(max_chunk_words * 4, rbs[h / nbuf]);
  int hi = 0;  // the hasher in use: hsv[nbuf hi + b] for buffer b
  // beside's cap on a row hash launch's blocks (HASH_CAP; 0 none): the committer's max_blocks, which leaves SM slots to the
  // GEMM stream's export
  const u64 hash_cap = getenv("HASH_CAP") ? strtoull(getenv("HASH_CAP"), nullptr, 10) : 0;
  // root(A_j), which each call's seed reads, over A_j's M·K bytes (Hasher::root: the leaves and the tree above them), on
  // Hashers of its own so it never shares buffers with the words' hash streams. Timed a call by each of ROOT_A_LIST's
  // schemes (row bytes, 0 row-seg; default the words'). ROOT_A=1 puts it in the forward, ahead of the coins, by the words'
  // scheme, or by ROOT_A_ROWS's when that is set.
  const int root_a = getenv("ROOT_A") ? atoi(getenv("ROOT_A")) : 0;
  const int root_a_rows = getenv("ROOT_A_ROWS") ? atoi(getenv("ROOT_A_ROWS")) : -1;
  std::vector<int> ras = rbs;
  if (const char* e = getenv("ROOT_A_LIST")) {
    ras.clear();
    for (char* p = (char*)e; *p;) {
      char* q;
      const long v = strtol(p, &q, 10);
      if (q == p) break;
      ras.push_back((int)v);
      p = q;
    }
  }
  std::map<int, Hasher> hra;
  for (int r : ras) hra[r];
  for (int r : rbs) hra[r];
  if (root_a_rows >= 0) hra[root_a_rows];
  for (auto& [r, h] : hra) h.init((size_t)M * 8192, r);
  auto fwd_root_a = [&](const LinDev& d, cudaStream_t st) {
    if (root_a) hra[root_a_rows >= 0 ? root_a_rows : rbs[hi]].root(d.A0, (u64)M * d.K, st);
  };
  auto rname = [](int rb) { return rb ? "rows" + std::to_string(rb) : std::string("rowseg"); };
  auto hname = [&](int h) { return h < 0 ? std::string("none") : rname(rbs[h]); };
  int* dout;
  CK(cudaMalloc(&dout, (size_t)M * 16384 * 4));
  Timer T;
  cudaStream_t s1;
  std::vector<cudaStream_t> s2(nbuf);
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  for (auto& s : s2) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const double f0 = sm_mhz(dclk);
  auto line = [&](const char* what, const LinDev* d, double ms, double macs, double words, double extra, std::string hn = "") {
    const double u = units(ms, f0);
    if (hn.empty()) hn = std::strstr(what, "hash") || std::strstr(what, "fwd_s") || std::strstr(what, "fwd_b") ? hname(hi) : "none";
    printf("{\"what\": \"%s\", \"hasher\": \"%s\", \"linear\": \"%s\", \"M\": %d, \"n\": %d, \"K\": %d, \"S\": %d, \"BN\": %d, \"ms\": %.6f, "
           "\"mhz\": %.1f, \"units\": %.6e, \"stock_macs\": %.6e, \"units_per_stock_mac\": %.6f, \"words\": %.6e, "
           "\"units_per_word\": %.3f, \"units_per_byte\": %.3f, \"extra\": %.6g}\n",
           what, hn.c_str(),
           d ? d->name : "all", M, d ? d->n : 0, d ? d->K : 0, d ? d->S : 0, d ? d->BN : 0, ms, f0, u, macs,
           macs > 0 ? u / macs : 0, words, words > 0 ? u / words : 0, words > 0 ? u / (4 * words) : 0, extra);
    fflush(stdout);
  };
  double fwd_macs = 0;
  for (const LinDev& d : lins) fwd_macs += 16.0 * M * d.n * (double)d.K;
  for (const LinDev& d : lins) {
    const double macs = (double)M * d.n * d.K, words = (double)M * d.n * d.S;
    line("plain_gemm", &d, T.ms(reps, [&] { gemm(d.BN, false, d.A0, d.B0, nullptr, M, d.n, d.K / KB, nullptr, nullptr, nullptr, 0); }),
         macs, 0, 0);
    line("ssn_gemm", &d, T.ms(reps, [&] { gemm(d.BN, false, d.A, d.B, d.beta, M, d.n, d.S, nullptr, nullptr, nullptr, 0); }),
         macs, 0, 0);
    double ex = 0;
    std::vector<double> hsh(rbs.size(), 0.0);
    for (int r0 = 0; r0 < M; r0 += d.chunk) {
      const int mc = std::min(d.chunk, M - r0);
      const uint8_t* A = d.A + (size_t)(r0 / 64) * d.S * TA;
      ex += T.ms(reps, [&] { gemm(d.BN, true, A, d.B, d.beta, mc, d.n, d.S, Wb[0], nullptr, nullptr, 0); });
      const long long blocks = (long long)(mc / 64) * (d.n / d.BN) * d.S;
      for (size_t h = 0; h < rbs.size(); h++) hsh[h] += T.ms(reps, [&] { hsv[nbuf * h].run(Wb[0], blocks, 64 * d.BN * 4, 0); });
    }
    line("ssn_gemm_export", &d, ex, macs, words, d.chunk);
    for (hi = 0; hi < (int)rbs.size(); hi++) line("hash_pass", &d, hsh[hi], macs, words, d.chunk);
    hi = -1;
    line("coins", &d, T.ms(std::max(reps, 51), [&] { coins_form(d.msg, d.coins, d.A, M, d.S, 0, true, false); }), macs,
         words, (double)coin_blocks(M, d.S));
    line("forming", &d, T.ms(std::max(reps, 51), [&] { coins_form(d.msg, d.coins, d.A, M, d.S, 0, false, true); }), macs,
         words, (double)M * d.S);
    line("coins_forming", &d, T.ms(std::max(reps, 51), [&] { coins_form(d.msg, d.coins, d.A, M, d.S, 0); }), macs, words,
         (double)M * d.S);
    const u64 abytes = (u64)M * d.K;
    for (int r : ras) {
      Hasher& h = hra[r];
      line("root_a_hash", &d, T.ms(std::max(reps, 51), [&] { h.run((const uint4*)d.A0, (long long)(abytes / 65536), 65536, 0); }),
           macs, abytes / 4.0, (double)abytes, rname(r));
      line("root_a_root", &d, T.ms(std::max(reps, 51), [&] { h.root(d.A0, abytes, 0); }), macs, abytes / 4.0, (double)abytes,
           rname(r));
    }
  }
  double fwd_words = 0;
  for (const LinDev& d : lins) fwd_words += 16.0 * M * d.n * d.S;
  // forward: plain, ssn without export, serial, beside
  auto fwd_plain = [&] {
    for (int l = 0; l < 16; l++)
      for (const LinDev& d : lins) gemm(d.BN, false, d.A0, d.B0, nullptr, M, d.n, d.K / KB, nullptr, nullptr, nullptr, s1);
  };
  line("fwd_plain_gemm", nullptr, T.ms(reps, fwd_plain, s1), fwd_macs, 0, 0);
  auto fwd_serial = [&] {
    for (int l = 0; l < 16; l++)
      for (const LinDev& d : lins) {
        fwd_root_a(d, s1);
        coins_form(d.msg, d.coins, d.A, M, d.S, s1);
        for (int r0 = 0; r0 < M; r0 += d.chunk) {
          const int mc = std::min(d.chunk, M - r0);
          gemm(d.BN, true, d.A + (size_t)(r0 / 64) * d.S * TA, d.B, d.beta, mc, d.n, d.S, Wb[0], nullptr, nullptr, s1);
          hsv[nbuf * hi].run(Wb[0], (long long)(mc / 64) * (d.n / d.BN) * d.S, 64 * d.BN * 4, s1);
        }
      }
  };
  std::vector<cudaEvent_t> done(nbuf), full(nbuf);
  for (int b = 0; b < nbuf; b++) {
    CK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&full[b], cudaEventDisableTiming));
  }
  auto fwd_beside = [&] {
    int c = 0;
    for (int b = 0; b < nbuf; b++) CK(cudaEventRecord(done[b], s2[b]));
    for (int l = 0; l < 16; l++)
      for (const LinDev& d : lins) {
        fwd_root_a(d, s1);
        coins_form(d.msg, d.coins, d.A, M, d.S, s1);
        for (int r0 = 0; r0 < M; r0 += d.chunk, c++) {
          const int mc = std::min(d.chunk, M - r0), b = c % nbuf;
          CK(cudaStreamWaitEvent(s1, done[b]));
          gemm(d.BN, true, d.A + (size_t)(r0 / 64) * d.S * TA, d.B, d.beta, mc, d.n, d.S, Wb[b], nullptr, nullptr, s1);
          CK(cudaEventRecord(full[b], s1));
          CK(cudaStreamWaitEvent(s2[b], full[b]));
          hsv[nbuf * hi + b].run(Wb[b], (long long)(mc / 64) * (d.n / d.BN) * d.S, 64 * d.BN * 4, s2[b], hash_cap);
          CK(cudaEventRecord(done[b], s2[b]));
        }
      }
    for (int b = 0; b < nbuf; b++) CK(cudaStreamWaitEvent(s1, done[b]));
  };
  for (hi = 0; hi < (int)rbs.size(); hi++) {
    line("fwd_serial", nullptr, T.ms(reps, fwd_serial, s1), fwd_macs, fwd_words, 0);
    line("fwd_beside", nullptr, T.ms(reps, fwd_beside, s1), fwd_macs, fwd_words, 0);
  }
  printf("{\"what\": \"clock\", \"mhz_start\": %.1f, \"mhz_end\": %.1f, \"beside_buffers\": %d, \"hash_cap\": %llu, "
         "\"root_a_in_forward\": %d, \"root_a_rows\": %d, \"root_a_tree\": 1}\n", f0, sm_mhz(dclk), nbuf,
         (unsigned long long)hash_cap, root_a, root_a_rows);
  return 0;
}

int main(int argc, char** argv) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_sms = p.multiProcessorCount;
  if (p.major != 9) {
    fprintf(stderr, "ERR device sm_%d%d: this binary runs on sm_90a\n", p.major, p.minor);
    return 2;
  }
  if (argc == 2 && !strcmp(argv[1], "check")) return check();
  if (argc >= 4 && !strcmp(argv[1], "time")) {
    std::vector<int> rbs;
    for (int a = 5; a < argc; a++) rbs.push_back(atoi(argv[a]));
    if (rbs.empty()) rbs = {0, 65536, 16384};
    return time_all(atoi(argv[2]), atoi(argv[3]), argc >= 5 ? atof(argv[4]) : 16.0, rbs);
  }
  fprintf(stderr, "usage: ssn check | time REPS M [BUDGET_GB [ROW_BYTES ...]]   (ROW_BYTES 0: row-seg)\n");
  return 2;
}
