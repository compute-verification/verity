// A standalone sm_120 probe for the band decode's Keccak-f[1600] (no torch): integer pipe throughput, and the round code's
// variants, each checked bit for bit against pous_native.cu's keccak_f_pair, at the thread counts a decode launches.
//
//   nvcc -O3 -arch=sm_120 -o keccak_probe keccak_probe.cu && ./keccak_probe > probe.json
//
// Pipes: eight independent chains per thread of one instruction (LOP3, SHF, IMAD, IMAD.HI, IMAD.WIDE) or a 1:1 mix, at full
// occupancy, reported as warp instructions per SM clock (clock64 inside the kernel).  Keccak: each thread permutes two
// interleaved states (`pair`, as the band chain does) or one (`single`, a lane-pair split's share), `iters` times, reported
// as Keccak-f per second and per SM clock.  A variant whose states differ from the base's fails the probe (exit 1).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));        \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

__constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
__constant__ uint32_t MUL[33];  // 1 << n, set at run time so ptxas cannot strength-reduce a multiply back to a shift
__constant__ uint32_t MUL2[33];  // the same values again, so ptxas cannot merge a mul.lo and a mad.hi into one IMAD.WIDE

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

// ---- the base: pous_native.cu's keccak_rho_pi_chi and keccak_f_pair, verbatim ----------------------------------------
__device__ __forceinline__ void keccak_rho_pi_chi(uint64_t st[25]) {
  constexpr int PI_DEST[25] = {0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4};
  constexpr int RHO[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
  uint64_t b[25];
#pragma unroll
  for (int s = 0; s < 25; ++s) b[PI_DEST[s]] = RHO[s] ? rotl64(st[s], RHO[s]) : st[s];
#pragma unroll
  for (int j = 0; j < 25; j += 5) {
#pragma unroll
    for (int i = 0; i < 5; ++i) st[j + i] = b[j + i] ^ ((~b[j + (i + 1) % 5]) & b[j + (i + 2) % 5]);
  }
}

__device__ __forceinline__ void keccak_f_thread(uint64_t st[25]) {
#pragma unroll 1
  for (int r = 0; r < 24; ++r) {
    uint64_t bc[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }
    keccak_rho_pi_chi(st);
    st[0] ^= KECCAK_RC[r];
  }
}

__device__ __forceinline__ void keccak_f_pair(uint64_t s[25], uint64_t t[25]) {
#pragma unroll 1
  for (int r = 0; r < 24; ++r) {
    uint64_t bs[5], bt[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      bs[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
      bt[i] = t[i] ^ t[i + 5] ^ t[i + 10] ^ t[i + 15] ^ t[i + 20];
    }
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const uint64_t ds = bs[(i + 4) % 5] ^ rotl64(bs[(i + 1) % 5], 1);
      const uint64_t dt = bt[(i + 4) % 5] ^ rotl64(bt[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) {
        s[j + i] ^= ds;
        t[j + i] ^= dt;
      }
    }
    keccak_rho_pi_chi(s);
    keccak_rho_pi_chi(t);
    s[0] ^= KECCAK_RC[r];
    t[0] ^= KECCAK_RC[r];
  }
}

// ---- variants ------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t lo32(uint64_t x) { return (uint32_t)x; }
__device__ __forceinline__ uint32_t hi32(uint64_t x) { return (uint32_t)(x >> 32); }
__device__ __forceinline__ uint64_t mk64(uint32_t lo, uint32_t hi) { return ((uint64_t)hi << 32) | lo; }
__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ uint64_t xor3_64(uint64_t a, uint64_t b, uint64_t c) {
  return mk64(xor3(lo32(a), lo32(b), lo32(c)), xor3(hi32(a), hi32(b), hi32(c)));
}
__device__ __forceinline__ uint32_t mul_lo(uint32_t a, uint32_t m) {
  uint32_t d;
  asm("mul.lo.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(m));
  return d;
}
__device__ __forceinline__ uint32_t mad_hi(uint32_t a, uint32_t m, uint32_t c) {
  uint32_t d;
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(m), "r"(c));
  return d;
}
// rotl64 on the multiply pipe: (hi, lo) <<< n is hi' = (hi << n) + (lo >> (32 - n)), lo' = (lo << n) + (hi >> (32 - n)) (the
// two parts of each word share no bit, so + is |), each a mul.lo and a mad.hi by 2^n; n > 32 swaps the halves first.
// SPLIT_M takes the mad.hi's multiplier from MUL2, so each rotation stays two IMAD and two IMAD.HI.
template <bool SPLIT_M>
__device__ __forceinline__ uint64_t rotl64_imad(uint64_t x, int n) {
  uint32_t lo = lo32(x), hi = hi32(x);
  if (n > 32) {
    const uint32_t t = lo;
    lo = hi;
    hi = t;
    n -= 32;
  }
  if (n == 0 || n == 32) return n ? mk64(hi, lo) : mk64(lo, hi);
  const uint32_t m = MUL[n], m2 = SPLIT_M ? MUL2[n] : MUL[n];
  return mk64(mad_hi(hi, m2, mul_lo(lo, m)), mad_hi(lo, m2, mul_lo(hi, m)));
}

// FD: theta's D folded into the column update (st ^= C[x-1] ^ rotl(C[x+1], 1) as one 3-input LOP3 per half).
// IM: bit s of IM puts rho's rotation of lane s on the multiply pipe.  DIM: theta's five rotations by one there too.
// SM: those rotations as two IMAD and two IMAD.HI (MUL2), else as ptxas merges them (IMAD.WIDE).  U2: the round loop
// unrolled twice.
template <bool FD, uint32_t IM, bool DIM, bool SM>
__device__ __forceinline__ void theta(uint64_t st[25]) {
  uint64_t bc[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    const uint64_t r1 = DIM ? rotl64_imad<SM>(bc[(i + 1) % 5], 1) : rotl64(bc[(i + 1) % 5], 1);
    if (FD) {
#pragma unroll
      for (int j = 0; j < 25; j += 5) st[j + i] = xor3_64(st[j + i], bc[(i + 4) % 5], r1);
    } else {
      const uint64_t t = bc[(i + 4) % 5] ^ r1;
#pragma unroll
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }
  }
}

template <uint32_t IM, bool SM>
__device__ __forceinline__ void rho_pi_chi(uint64_t st[25]) {
  constexpr int PI_DEST[25] = {0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4};
  constexpr int RHO[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
  uint64_t b[25];
#pragma unroll
  for (int s = 0; s < 25; ++s)
    b[PI_DEST[s]] = RHO[s] == 0 ? st[s] : (((IM >> s) & 1) ? rotl64_imad<SM>(st[s], RHO[s]) : rotl64(st[s], RHO[s]));
#pragma unroll
  for (int j = 0; j < 25; j += 5) {
#pragma unroll
    for (int i = 0; i < 5; ++i) st[j + i] = b[j + i] ^ ((~b[j + (i + 1) % 5]) & b[j + (i + 2) % 5]);
  }
}

template <bool FD, uint32_t IM, bool DIM, bool SM>
__device__ __forceinline__ void pair_round(uint64_t s[25], uint64_t t[25], int r) {
  theta<FD, IM, DIM, SM>(s);
  theta<FD, IM, DIM, SM>(t);
  rho_pi_chi<IM, SM>(s);
  rho_pi_chi<IM, SM>(t);
  s[0] ^= KECCAK_RC[r];
  t[0] ^= KECCAK_RC[r];
}

template <bool FD, uint32_t IM, bool DIM, bool SM, bool U2>
__device__ __forceinline__ void keccak_f_pair_v(uint64_t s[25], uint64_t t[25]) {
  if (U2) {
#pragma unroll 1
    for (int r = 0; r < 24; r += 2) {
      pair_round<FD, IM, DIM, SM>(s, t, r);
      pair_round<FD, IM, DIM, SM>(s, t, r + 1);
    }
  } else {
#pragma unroll 1
    for (int r = 0; r < 24; ++r) pair_round<FD, IM, DIM, SM>(s, t, r);
  }
}

template <bool FD, uint32_t IM, bool DIM, bool SM, bool U2>
__device__ __forceinline__ void keccak_f_single_v(uint64_t s[25]) {
#pragma unroll 1
  for (int r = 0; r < 24; r += U2 ? 2 : 1) {
    theta<FD, IM, DIM, SM>(s);
    rho_pi_chi<IM, SM>(s);
    s[0] ^= KECCAK_RC[r];
    if (U2) {
      theta<FD, IM, DIM, SM>(s);
      rho_pi_chi<IM, SM>(s);
      s[0] ^= KECCAK_RC[r + 1];
    }
  }
}

// ---- benches ------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void seed(uint64_t s[25], uint64_t t[25], int64_t n) {
#pragma unroll
  for (int i = 0; i < 25; ++i) {
    s[i] = (uint64_t)n * 0x9E3779B97F4A7C15ull + (uint64_t)i;
    t[i] = s[i] ^ 0xC2B2AE3D27D4EB4Full;
  }
}

// variant: 0 base, then the template list in VARIANTS below; pair: two states per thread, else one
template <int V>
__device__ __forceinline__ void run_pair(uint64_t s[25], uint64_t t[25]);
template <int V>
__device__ __forceinline__ void run_single(uint64_t s[25]);

constexpr uint32_t ALL = 0x1FFFFFEu;  // lanes 1..24
constexpr uint32_t HALF = 0x0AAAAAAu;  // lanes 1, 3, .., 23
constexpr uint32_t TWO3 = 0x16DB6DAu;  // two lanes in three: 1, 3, 4, 6, 7, 9, ...

#define VARIANT(V, FD, IM, DIM, SM, U2)                                                                    \
  template <>                                                                                              \
  __device__ __forceinline__ void run_pair<V>(uint64_t s[25], uint64_t t[25]) {                            \
    keccak_f_pair_v<FD, IM, DIM, SM, U2>(s, t);                                                            \
  }                                                                                                        \
  template <>                                                                                              \
  __device__ __forceinline__ void run_single<V>(uint64_t s[25]) {                                          \
    keccak_f_single_v<FD, IM, DIM, SM, U2>(s);                                                             \
  }
template <>
__device__ __forceinline__ void run_pair<0>(uint64_t s[25], uint64_t t[25]) { keccak_f_pair(s, t); }
template <>
__device__ __forceinline__ void run_single<0>(uint64_t s[25]) { keccak_f_thread(s); }
VARIANT(1, true, 0u, false, false, false)
VARIANT(2, true, 0u, false, false, true)
VARIANT(3, true, ALL, false, false, false)
VARIANT(4, true, ALL, false, true, false)
VARIANT(5, true, HALF, false, true, false)
VARIANT(6, true, TWO3, false, true, false)
VARIANT(7, true, ALL, true, true, false)
VARIANT(8, true, HALF, false, true, true)
VARIANT(9, false, 0u, false, false, true)
const char* VARIANT_NAMES[] = {"base", "fd", "fd_u2", "fd_imadw_rho", "fd_imad_rho", "fd_imad_half", "fd_imad_two3",
                               "fd_imad_rho_d", "fd_imad_half_u2", "base_u2"};
constexpr int NVAR = 10;

template <int V>
__global__ void keccak_kernel(uint64_t* out, int64_t threads, int64_t iters, int pair, long long* cycles) {
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const long long c0 = clock64();
  if (n < threads) {
    uint64_t s[25], t[25];
    seed(s, t, n);
    if (pair) {
      for (int64_t it = 0; it < iters; ++it) run_pair<V>(s, t);
    } else {
      for (int64_t it = 0; it < iters; ++it) run_single<V>(s);
    }
#pragma unroll
    for (int i = 0; i < 25; ++i) out[n * 50 + i] = s[i];
#pragma unroll
    for (int i = 0; i < 25; ++i) out[n * 50 + 25 + i] = pair ? t[i] : 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) cycles[blockIdx.x] = clock64() - c0;
}

// ---- pipes ------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t op_lop3(uint32_t x, uint32_t y, uint32_t z) { return xor3(x, y, z); }
__device__ __forceinline__ uint32_t op_shf(uint32_t x, uint32_t y) {
  uint32_t d;
  asm("shf.l.wrap.b32 %0, %1, %2, 7;" : "=r"(d) : "r"(x), "r"(y));
  return d;
}
__device__ __forceinline__ uint32_t op_imad(uint32_t x, uint32_t m, uint32_t c) {
  uint32_t d;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(x), "r"(m), "r"(c));
  return d;
}
__device__ __forceinline__ uint64_t op_wide(uint32_t x, uint32_t m, uint64_t c) {
  uint64_t d;
  asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(d) : "r"(x), "r"(m), "l"(c));
  return d;
}

// OP: 0 LOP3, 1 SHF, 2 IMAD, 3 IMAD.HI, 4 IMAD.WIDE, 5 LOP3 + IMAD, 6 SHF + IMAD, 7 LOP3 + SHF, 8 LOP3 + IMAD.HI,
// 9 LOP3 + SHF + 2 IMAD (one Keccak rotation's mix)
constexpr int UNROLL = 16;
template <int OP>
__global__ void pipe_kernel(uint32_t* out, int iters, long long* cycles) {
  uint32_t x[8];
  uint64_t w[8];
  const uint32_t m = MUL[7], y = threadIdx.x * 0x9E3779B9u, z = blockIdx.x * 0x85EBCA6Bu;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    x[k] = threadIdx.x + k * 0x1234567u;
    w[k] = x[k];
  }
  __syncthreads();
  const long long c0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if (OP == 0) x[k] = op_lop3(x[k], y, z);
        if (OP == 1) x[k] = op_shf(x[k], y);
        if (OP == 2) x[k] = op_imad(x[k], m, z);
        if (OP == 3) x[k] = mad_hi(x[k], m, z);
        if (OP == 4) w[k] = op_wide((uint32_t)w[k], m, w[k]);
        if (OP == 5) x[k] = (k & 1) ? op_imad(x[k], m, z) : op_lop3(x[k], y, z);
        if (OP == 6) x[k] = (k & 1) ? op_imad(x[k], m, z) : op_shf(x[k], y);
        if (OP == 7) x[k] = (k & 1) ? op_shf(x[k], y) : op_lop3(x[k], y, z);
        if (OP == 8) x[k] = (k & 1) ? mad_hi(x[k], m, z) : op_lop3(x[k], y, z);
        if (OP == 9) x[k] = (k & 3) == 0 ? op_lop3(x[k], y, z) : (k & 3) == 1 ? op_shf(x[k], y) : op_imad(x[k], m, z);
      }
    }
  }
  __syncthreads();
  const long long c1 = clock64();
  uint32_t acc = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) acc ^= x[k] ^ (uint32_t)w[k] ^ (uint32_t)(w[k] >> 32);
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
  if (threadIdx.x == 0) cycles[blockIdx.x] = c1 - c0;
}

template <int OP>
static double pipe_rate(int sms, int iters, uint32_t* dout, long long* dcyc) {
  const int threads = 256, blocks = sms * 6;  // 48 warps per SM
  pipe_kernel<OP><<<blocks, threads>>>(dout, 4, dcyc);
  CK(cudaDeviceSynchronize());
  pipe_kernel<OP><<<blocks, threads>>>(dout, iters, dcyc);
  CK(cudaDeviceSynchronize());
  std::vector<long long> cyc(blocks);
  CK(cudaMemcpy(cyc.data(), dcyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost));
  double mean = 0;
  for (auto c : cyc) mean += (double)c;
  mean /= blocks;
  const double warp_instr_per_sm = (double)iters * UNROLL * 8 * (threads / 32) * 6;
  return warp_instr_per_sm / mean;  // warp instructions per SM clock
}

template <int V>
static void keccak_run(int64_t threads, int64_t iters, int pair, int block, uint64_t* dout, long long* dcyc, float* ms,
                       double* cycles_max) {
  const unsigned blocks = (unsigned)((threads + block - 1) / block);
  keccak_kernel<V><<<blocks, block>>>(dout, threads, 2, pair, dcyc);
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  keccak_kernel<V><<<blocks, block>>>(dout, threads, iters, pair, dcyc);
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  CK(cudaEventElapsedTime(ms, a, b));
  std::vector<long long> cyc(blocks);
  CK(cudaMemcpy(cyc.data(), dcyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost));
  long long mx = 0;
  for (auto c : cyc) mx = c > mx ? c : mx;
  *cycles_max = (double)mx;
}

typedef void (*KRun)(int64_t, int64_t, int, int, uint64_t*, long long*, float*, double*);
static KRun KRUNS[NVAR] = {keccak_run<0>, keccak_run<1>, keccak_run<2>, keccak_run<3>, keccak_run<4>,
                           keccak_run<5>, keccak_run<6>, keccak_run<7>, keccak_run<8>, keccak_run<9>};

int main(int argc, char** argv) {
  int64_t iters = argc > 1 ? atoll(argv[1]) : 2000;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  uint32_t mul[33];
  for (int n = 0; n < 33; ++n) mul[n] = n < 32 ? (1u << n) : 0u;
  CK(cudaMemcpyToSymbol(MUL, mul, sizeof(mul)));
  CK(cudaMemcpyToSymbol(MUL2, mul, sizeof(mul)));
  const int sms = p.multiProcessorCount;
  const int64_t maxthreads = 4 * (int64_t)sms * 32 * 2 * 2;  // two warps per scheduler, twice
  uint64_t* dout;
  long long* dcyc;
  uint32_t* pout;
  CK(cudaMalloc(&dout, maxthreads * 50 * sizeof(uint64_t)));
  CK(cudaMalloc(&dcyc, 65536 * sizeof(long long)));
  CK(cudaMalloc(&pout, (size_t)sms * 6 * 256 * sizeof(uint32_t)));
  printf("{\"device\": \"%s\", \"sms\": %d, \"cc\": \"%d.%d\", \"pipes_warp_instr_per_sm_clk\": {", p.name, sms, p.major, p.minor);
  const char* pn[] = {"lop3", "shf", "imad", "imad_hi", "imad_wide", "lop3+imad", "shf+imad", "lop3+shf", "lop3+imad_hi",
                      "lop3+shf+2imad"};
  double pr[10] = {pipe_rate<0>(sms, 2000, pout, dcyc), pipe_rate<1>(sms, 2000, pout, dcyc), pipe_rate<2>(sms, 2000, pout, dcyc),
                   pipe_rate<3>(sms, 2000, pout, dcyc), pipe_rate<4>(sms, 2000, pout, dcyc), pipe_rate<5>(sms, 2000, pout, dcyc),
                   pipe_rate<6>(sms, 2000, pout, dcyc), pipe_rate<7>(sms, 2000, pout, dcyc), pipe_rate<8>(sms, 2000, pout, dcyc),
                   pipe_rate<9>(sms, 2000, pout, dcyc)};
  for (int i = 0; i < 10; ++i) printf("%s\"%s\": %.4f", i ? ", " : "", pn[i], pr[i]);
  printf("}, \"keccak\": [");
  // thread counts: the decode's rows (15,360 = 30 segments x 512), one warp per scheduler, a lane-pair split's lanes, two
  // warps per scheduler
  const int64_t sched = 4 * (int64_t)sms * 32;
  struct Cfg { int64_t threads; int pair; int block; };
  std::vector<Cfg> cfgs = {{15360, 1, 64}, {sched, 1, 128}, {2 * sched, 1, 128}, {30720, 0, 64}, {sched, 0, 128}, {2 * sched, 0, 128}};
  std::vector<uint64_t> ref, got;
  bool first = true, exact = true;
  for (auto& c : cfgs) {
    for (int v = 0; v < NVAR; ++v) {
      float ms = 0;
      double cyc = 0;
      KRUNS[v](c.threads, iters, c.pair, c.block, dout, dcyc, &ms, &cyc);
      got.resize(c.threads * 50);
      CK(cudaMemcpy(got.data(), dout, c.threads * 50 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      bool same = true;
      if (v == 0) ref = got;
      else same = memcmp(ref.data(), got.data(), got.size() * sizeof(uint64_t)) == 0;
      exact = exact && same;
      const double perms = (double)c.threads * iters * (c.pair ? 2 : 1);
      printf("%s\n {\"variant\": \"%s\", \"threads\": %lld, \"pair\": %d, \"block\": %d, \"iters\": %lld, \"ms\": %.4f, "
             "\"keccak_f_per_s\": %.6e, \"block_cycles_max\": %.0f, \"cycles_per_keccak_f_per_thread\": %.1f, \"exact\": %s}",
             first ? "" : ",", VARIANT_NAMES[v], (long long)c.threads, c.pair, c.block, (long long)iters, ms, perms / (ms / 1e3),
             cyc, cyc / (iters * (c.pair ? 2 : 1)), same ? "true" : "false");
      first = false;
    }
  }
  printf("\n], \"exact\": %s}\n", exact ? "true" : "false");
  return exact ? 0 : 1;
}
