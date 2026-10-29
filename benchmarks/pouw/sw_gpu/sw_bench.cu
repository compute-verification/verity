// Forming the noised operands and expanding their coins, whole calls on one H200 (sm_90a), timed in time and in SM-clocks.
//
// Every kernel is grid-stride over its operand and reads the SM clock twice (block 0 samples clock64 against %globaltimer,
// so a call's SM clock is its own), and its loop is one iteration: run.sh's SASS gate pins each loop's opcodes. Operand
// bytes are offset-binary (route U's u8 domain, ncp.py): a = A + 64 for int7 A, a coin byte c gives E = (c & 63) - 31, so
// a + (c & 63) + 33 is X + beta at beta 128 with no carry between bytes, and one 32-bit add forms four entries.
//
//   copy        X = A                                   (the HBM reference: 1 B read and 1 B written an entry)
//   form1       X = A + D (+ beta), D bytes given       (one copy; D already in range: one IADD3 a 4-entry word)
//   form1s      the same as a byte-wise mod-256 add     (SWAR: no range assumption, 4 to 5 ALU ops a word)
//   formncp     NCP's three copies from A and coin bytes C: E1 = C & 63, E2 = E1 shifted by 24 entries (ncp-v1-shift24's R,
//               a word move), X1 = A + E1, X2 = A + E2, X3 = -(A + E1 + E2), each + beta
//   coins       SHA-512 in counter mode, SHA-512(seed || ctr) with the 64-byte seed fixed (compress_from<8>), 64 noise bytes
//               a compression, written out
//   coinsform1  coins and form1 fused: 64 entries of X a compression, A read, D never written
//   coinsncp    coins and formncp fused: E2's shift crosses into the next lane's compression (one SHFL a word)
//
//   sw_bench check                 every kernel's output against the host on a small domain (exit 1 on a mismatch)
//   sw_bench run REPS ENTRIES...   one JSON line an (item, size): ms a call (median of REPS), its SM clock, bytes, and the
//                                  MAC-times it occupies (ms x f x SMs x 4,096: MH100's unit, one dense int8 MAC = 1)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a [-DSHA_ROT=N] sw_bench.cu -o sw_bench
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../../integrations/vllm/verity_vllm/commit/committer/sha512.cuh"

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
#define DEV __device__ __forceinline__

struct Clk {
  long long c0, c1;
  u64 t0, t1;
};
__device__ Clk g_clk;

DEV u64 gtime() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define T_BEGIN                     \
  const long long c0_ = clock64(); \
  const u64 t0_ = gtime();
#define T_END                                                                            \
  {                                                                                      \
    const long long c1_ = clock64();                                                     \
    const u64 t1_ = gtime();                                                             \
    if (blockIdx.x == 0 && threadIdx.x == 0) g_clk = Clk{c0_, c1_, t0_, t1_};             \
  }

constexpr u32 MASK6 = 0x3f3f3f3fu;   // a coin byte's low six bits
constexpr u32 K33 = 0x21212121u;     // + 33 a byte: (A + 64) + (E + 31) + 33 = X + 128
constexpr u32 K254 = 0xfefefefeu;    // 254 - (A + 64) - (E1 + 31) - (E2 + 31) = -(A + E1 + E2) + 128

DEV u32 swar_add(u32 a, u32 d) {
  const u32 s = (a & 0x7f7f7f7fu) + (d & 0x7f7f7f7fu);
  return s ^ ((a ^ d) & 0x80808080u);
}
DEV uint4 f1(uint4 a, uint4 d) { return make_uint4(a.x + d.x + K33, a.y + d.y + K33, a.z + d.z + K33, a.w + d.w + K33); }
DEV uint4 f1s(uint4 a, uint4 d) { return make_uint4(swar_add(a.x, d.x), swar_add(a.y, d.y), swar_add(a.z, d.z), swar_add(a.w, d.w)); }

// One 4-entry word of NCP's three copies from a, and coin words c (E1) and c2 (E2's source).
DEV void ncp_word(u32 a, u32 c, u32 c2, u32& x1, u32& x2, u32& x3) {
  const u32 e1 = c & MASK6, e2 = c2 & MASK6;
  x1 = a + e1 + K33;
  x2 = a + e2 + K33;
  x3 = K254 - a - e1 - e2;
}

__global__ void __launch_bounds__(256) copy_k(const uint4* __restrict__ A, uint4* __restrict__ X, size_t n16) {
  T_BEGIN
#pragma unroll 1
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) X[i] = A[i];
  T_END
}

template <bool SWAR>
__global__ void __launch_bounds__(256) form1_k(const uint4* __restrict__ A, const uint4* __restrict__ D, uint4* __restrict__ X,
                                               size_t n16) {
  T_BEGIN
#pragma unroll 1
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
    X[i] = SWAR ? f1s(A[i], D[i]) : f1(A[i], D[i]);
  T_END
}

// E2 = E1 shifted by 24 entries: the coin stream read again 24 bytes on (two 8-byte loads, L1/L2-resident).
__global__ void __launch_bounds__(256) formncp_k(const uint4* __restrict__ A, const unsigned char* __restrict__ C,
                                                 uint4* __restrict__ X, size_t n16) {
  T_BEGIN
  const uint4* C4 = reinterpret_cast<const uint4*>(C);
#pragma unroll 1
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 a = A[i], c = C4[i];
    const uint2 lo = *reinterpret_cast<const uint2*>(C + 16 * i + 24), hi = *reinterpret_cast<const uint2*>(C + 16 * i + 32);
    uint4 x1, x2, x3;
    ncp_word(a.x, c.x, lo.x, x1.x, x2.x, x3.x);
    ncp_word(a.y, c.y, lo.y, x1.y, x2.y, x3.y);
    ncp_word(a.z, c.z, hi.x, x1.z, x2.z, x3.z);
    ncp_word(a.w, c.w, hi.y, x1.w, x2.w, x3.w);
    X[i] = x1;
    X[n16 + i] = x2;
    X[2 * n16 + i] = x3;
  }
  T_END
}

// ---- coins ---------------------------------------------------------------------------------------------------------------
// SHA-512(seed || be64(ctr)): a 72-byte message, one block (words 0..7 the seed, 8 the counter, 9 the pad bit, 15 the
// length 576). Rounds 0..7 read only the seed, so the host runs them once (sha::rounds_prefix) and every compression starts
// at round 8 (sha::compress_from<8>), as the committer's leaves do.
struct Seed {
  u64 w[8];    // the seed's words
  u64 mid[8];  // the state after rounds 0..7
};

DEV void coin_block(const Seed& s, u64 ctr, u64 (&cv)[8]) {
  u64 w[16];
#pragma unroll
  for (int j = 0; j < 8; j++) w[j] = s.w[j];
  w[8] = ctr;
  w[9] = 0x8000000000000000ull;
#pragma unroll
  for (int j = 10; j < 15; j++) w[j] = 0;
  w[15] = 576;
  sha::iv(cv);
  sha::compress_from<8>(cv, s.mid, w);
}
// The digest's bytes in order (big-endian words) as four 16-byte vectors.
DEV uint4 dig16(const u64 (&cv)[8], int q) {
  const u64 a = cv[2 * q], b = cv[2 * q + 1];
  return make_uint4(__byte_perm((u32)(a >> 32), 0, 0x0123), __byte_perm((u32)a, 0, 0x0123), __byte_perm((u32)(b >> 32), 0, 0x0123),
                    __byte_perm((u32)b, 0, 0x0123));
}

__global__ void __launch_bounds__(256, 2) coins_k(const __grid_constant__ Seed s, uint4* __restrict__ out, u64 n_comp) {
  T_BEGIN
#pragma unroll 1
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n_comp; i += (u64)gridDim.x * blockDim.x) {
    u64 cv[8];
    coin_block(s, i, cv);
#pragma unroll
    for (int q = 0; q < 4; q++) out[4 * i + q] = dig16(cv, q);
  }
  T_END
}

__global__ void __launch_bounds__(256, 2) coinsform1_k(const __grid_constant__ Seed s, const uint4* __restrict__ A,
                                                       uint4* __restrict__ X, u64 n_comp) {
  T_BEGIN
#pragma unroll 1
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n_comp; i += (u64)gridDim.x * blockDim.x) {
    u64 cv[8];
    coin_block(s, i, cv);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint4 c = dig16(cv, q);
      X[4 * i + q] = f1(A[4 * i + q], make_uint4(c.x & MASK6, c.y & MASK6, c.z & MASK6, c.w & MASK6));
    }
  }
  T_END
}

// A lane's 64 entries are words 0..15 of its digest; E2's word j is word j + 6 of the stream: its own for j < 10, the next
// lane's word j - 6 for j >= 10 (lane 31 takes lane 0's: a warp's stream wraps, which moves no cost).
__global__ void __launch_bounds__(256, 2) coinsncp_k(const __grid_constant__ Seed s, const uint4* __restrict__ A,
                                                     uint4* __restrict__ X, u64 n_comp) {
  T_BEGIN
  const u64 n16 = 4 * n_comp;
#pragma unroll 1
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n_comp; i += (u64)gridDim.x * blockDim.x) {
    u64 cv[8];
    coin_block(s, i, cv);
    u32 c[16], nx[6];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint4 v = dig16(cv, q);
      c[4 * q] = v.x; c[4 * q + 1] = v.y; c[4 * q + 2] = v.z; c[4 * q + 3] = v.w;
    }
#pragma unroll
    for (int j = 0; j < 6; j++) nx[j] = __shfl_sync(0xffffffffu, c[j], (threadIdx.x + 1) & 31);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint4 a = A[4 * i + q];
      const u32 av[4] = {a.x, a.y, a.z, a.w};
      u32 x1[4], x2[4], x3[4];
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int j = 4 * q + r;
        ncp_word(av[r], c[j], j < 10 ? c[j + 6] : nx[j - 10], x1[r], x2[r], x3[r]);
      }
      X[4 * i + q] = make_uint4(x1[0], x1[1], x1[2], x1[3]);
      X[n16 + 4 * i + q] = make_uint4(x2[0], x2[1], x2[2], x2[3]);
      X[2 * n16 + 4 * i + q] = make_uint4(x3[0], x3[1], x3[2], x3[3]);
    }
  }
  T_END
}

// ---- host ------------------------------------------------------------------------------------------------------------------
static u64 mix64(u64 x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
  return x;
}
static Seed make_seed(u64 tag) {
  Seed s;
  for (int j = 0; j < 8; j++) s.w[j] = mix64(tag * 8 + j + 1);
  u64 cv[8], blk[16] = {};
  for (int j = 0; j < 8; j++) blk[j] = s.w[j];
  sha::iv(cv);
  sha::rounds_prefix(cv, blk, 8, s.mid);
  return s;
}
static void host_coins(const Seed& s, u64 ctr, unsigned char* out64) {
  unsigned char msg[72];
  for (int j = 0; j < 8; j++) sha::put_be64(msg + 8 * j, s.w[j]);
  sha::put_be64(msg + 64, ctr);
  sha::digest(msg, 72, out64);
}
static unsigned char hb_x(unsigned char a, unsigned char e) { return (unsigned char)(a + e + 33); }

static int g_sms = 132;
struct Bufs {
  size_t entries = 0;
  unsigned char *A = nullptr, *D = nullptr, *X = nullptr;
};
// A in offset-binary int7 (1..127) and D as E + 31 in 0..63, so A + D + 33 can't carry between a word's bytes (formncp
// masks its coin bytes to 0..63 itself). Filled on the device (the per-forward operands are gigabytes); check() reads them
// back.
__global__ void fill_k(unsigned char* A, unsigned char* D, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n + 64; i += (size_t)gridDim.x * blockDim.x) {
    u64 x = i + 0x9e3779b97f4a7c15ull;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    if (i < n) A[i] = (unsigned char)(1 + (x >> 8) % 127);
    D[i] = (unsigned char)(x & 63);
  }
}
static void alloc(Bufs& b, size_t entries) {
  b.entries = entries;
  CK(cudaMalloc(&b.A, entries));
  CK(cudaMalloc(&b.D, entries + 64));
  CK(cudaMalloc(&b.X, 3 * entries));
  fill_k<<<g_sms * 8, 256>>>(b.A, b.D, entries);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
}
static void release(Bufs& b) {
  cudaFree(b.A); cudaFree(b.D); cudaFree(b.X);
  b = Bufs{};
}

enum Item { COPY, FORM1, FORM1S, FORMNCP, COINS, COINSFORM1, COINSNCP, N_ITEMS };
static const char* NAMES[N_ITEMS] = {"copy", "form1", "form1s", "formncp", "coins", "coinsform1", "coinsncp"};
// bytes read and written an entry (HBM), and ALU work is from the SASS (run.sh)
static const double RW[N_ITEMS][2] = {{1, 1}, {2, 1}, {2, 1}, {2, 3}, {0, 1}, {1, 1}, {1, 3}};

static void launch(int it, Bufs& b, const Seed& s, cudaStream_t st) {
  const size_t n16 = b.entries / 16;
  const int mem_grid = g_sms * 8;
  const u64 n_comp = b.entries / 64;
  const int sha_grid = (int)std::min<u64>((n_comp + 255) / 256, (u64)g_sms * 2 * 16);
  switch (it) {
    case COPY: copy_k<<<mem_grid, 256, 0, st>>>((const uint4*)b.A, (uint4*)b.X, n16); break;
    case FORM1: form1_k<false><<<mem_grid, 256, 0, st>>>((const uint4*)b.A, (const uint4*)b.D, (uint4*)b.X, n16); break;
    case FORM1S: form1_k<true><<<mem_grid, 256, 0, st>>>((const uint4*)b.A, (const uint4*)b.D, (uint4*)b.X, n16); break;
    case FORMNCP: formncp_k<<<mem_grid, 256, 0, st>>>((const uint4*)b.A, b.D, (uint4*)b.X, n16); break;
    case COINS: coins_k<<<sha_grid, 256, 0, st>>>(s, (uint4*)b.X, n_comp); break;
    case COINSFORM1: coinsform1_k<<<sha_grid, 256, 0, st>>>(s, (const uint4*)b.A, (uint4*)b.X, n_comp); break;
    case COINSNCP: coinsncp_k<<<sha_grid, 256, 0, st>>>(s, (const uint4*)b.A, (uint4*)b.X, n_comp); break;
  }
}

static int check() {
  const size_t E = 1 << 20;  // 16,384 compressions: whole warps, so coinsncp's lane-31 wrap is inside each warp
  Bufs b;
  alloc(b, E);
  const Seed s = make_seed(7);
  std::vector<unsigned char> A(E), D(E + 64), X(3 * E), C(E);
  CK(cudaMemcpy(A.data(), b.A, E, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(D.data(), b.D, E + 64, cudaMemcpyDeviceToHost));
  for (size_t c = 0; c < E / 64; c++) host_coins(s, c, C.data() + 64 * c);
  int bad_all = 0;
  for (int it = 0; it < N_ITEMS; it++) {
    CK(cudaMemset(b.X, 0xA5, 3 * E));
    launch(it, b, s, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(X.data(), b.X, 3 * E, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < E; i++) {
      unsigned char w1 = 0, w2 = 0, w3 = 0;
      int copies = 1;
      switch (it) {
        case COPY: w1 = A[i]; break;
        case FORM1: w1 = hb_x(A[i], D[i]); break;
        case FORM1S: w1 = (unsigned char)(A[i] + D[i]); break;
        case FORMNCP:
        case COINSNCP: {
          copies = 3;
          unsigned char c1, c2;
          if (it == FORMNCP) {
            c1 = D[i]; c2 = D[i + 24];
          } else {
            const size_t comp = i / 64, off = i % 64, warp0 = comp / 32 * 32;
            c1 = C[i];
            const size_t o2 = off + 24, comp2 = warp0 + (comp - warp0 + o2 / 64) % 32;
            c2 = C[64 * comp2 + o2 % 64];
          }
          const unsigned char e1 = c1 & 63, e2 = c2 & 63;
          w1 = (unsigned char)(A[i] + e1 + 33);
          w2 = (unsigned char)(A[i] + e2 + 33);
          w3 = (unsigned char)(254 - A[i] - e1 - e2);
          break;
        }
        case COINS: w1 = C[i]; break;
        case COINSFORM1: w1 = hb_x(A[i], C[i] & 63); break;
      }
      bad += X[i] != w1;
      if (copies == 3) bad += (X[E + i] != w2) + (X[2 * E + i] != w3);
    }
    printf("{\"check\": \"%s\", \"entries\": %zu, \"mismatches\": %zu}\n", NAMES[it], E, bad);
    bad_all += bad != 0;
  }
  release(b);
  return bad_all ? 1 : 0;
}

static int run(int reps, const std::vector<size_t>& sizes) {
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const Seed s = make_seed(11);
  for (size_t entries : sizes) {
    Bufs b;
    alloc(b, entries);
    for (int it = 0; it < N_ITEMS; it++) {
      launch(it, b, s, st);
      CK(cudaStreamSynchronize(st));
      // enough launches a batch for 2 ms or more
      CK(cudaEventRecord(e0, st));
      launch(it, b, s, st);
      CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1));
      float one = 0;
      CK(cudaEventElapsedTime(&one, e0, e1));
      const int inner = std::max(1, std::min(2000, (int)(2.0f / std::max(one, 1e-4f))));
      std::vector<double> ms(reps), mhz(reps);
      for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(e0, st));
        for (int j = 0; j < inner; j++) launch(it, b, s, st);
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float t = 0;
        CK(cudaEventElapsedTime(&t, e0, e1));
        ms[r] = t / inner;
        Clk c;
        CK(cudaMemcpyFromSymbol(&c, g_clk, sizeof c));
        mhz[r] = c.t1 > c.t0 ? (double)(c.c1 - c.c0) / (double)(c.t1 - c.t0) * 1e3 : 0;
      }
      std::vector<double> sm = ms, sh = mhz;
      std::sort(sm.begin(), sm.end());
      std::sort(sh.begin(), sh.end());
      const double t = sm[reps / 2], f = sh[reps / 2];
      const double units = t * 1e-3 * f * 1e6 * g_sms * 4096.0;
      printf("{\"item\": \"%s\", \"entries\": %zu, \"ms\": %.6f, \"ms_min\": %.6f, \"ms_max\": %.6f, \"inner\": %d, \"mhz\": %.1f, "
             "\"read_b\": %.0f, \"write_b\": %.0f, \"gbps\": %.2f, \"units\": %.6e, \"units_per_entry\": %.4f}\n",
             NAMES[it], entries, t, sm[0], sm[reps - 1], inner, f, RW[it][0] * entries, RW[it][1] * entries,
             (RW[it][0] + RW[it][1]) * entries / (t * 1e-3) * 1e-9, units, units / entries);
      fflush(stdout);
    }
    release(b);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: sw_bench check | run REPS ENTRIES...\n");
    return 2;
  }
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_sms = p.multiProcessorCount;
  if (!strcmp(argv[1], "check")) return check();
  if (!strcmp(argv[1], "run") && argc >= 4) {
    std::vector<size_t> sizes;
    for (int i = 3; i < argc; i++) {
      const size_t e = strtoull(argv[i], nullptr, 10);
      if (e % 2048) {
        fprintf(stderr, "entries a multiple of 2,048 (a whole warp's compressions)\n");
        return 2;
      }
      sizes.push_back(e);
    }
    return run(atoi(argv[2]), sizes);
  }
  fprintf(stderr, "usage: sw_bench check | run REPS ENTRIES...\n");
  return 2;
}
