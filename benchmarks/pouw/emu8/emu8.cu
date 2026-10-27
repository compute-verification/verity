// EMU8: Dither's int8 product C = Ã B̃ᵀ on sm_120's E4M3 tensor cores, bit-exactly, beside native IMMA.
//
// Every int8 x is balanced base-16 limbs x = 16 h + l, l in [-8, 7], h in [-8, 8], m = h + l in [-15, 14]; all three are
// E4M3 integers, and so are 4h, 16h and -l. Two emulations of each k32 step's 32 MACs:
//   k3 (Karatsuba, 3 QMMAs): xy = 240 hh' + 16 mm' - 15 ll'. Planes A = [4h, m, -l], B = [4h', m', l']; acc1 = QMMA(4h, 4h',
//      C1) then QMMA(-l, l', acc1) = C1 + T1, T1 = 16 HH - LL; acc2 = QMMA(m, m', C2) = C2 + T2, T2 = MM; the step adds
//      15 T1 + 16 T2 to the word.
//   k4 (schoolbook, 4 QMMAs): planes A = [16h, l], B = [16h', l'], the four pairings chained into one accumulator from C0.
// Every accumulator starts each k32 step from a constant word (the MMA's C input) in a binade wide enough for the step:
// k4's C0 = 0x4B400000 (1.5 2^23, ulp 1; |T| <= 2^19), k3's C1 = 0x4B7BAB6C (ulp 1; |T1| <= 32768) and C2 = 0x49E07A56
// (1838922.75, ulp 1/8; -6720 <= T2 <= 7200), each with room to spare. In a binade the FP32 word is C + T exactly,
// so T is the bits' difference (times the ulp's inverse), and since 15 C1 + 2 C2 = 0 mod 2^32,
// k3's increment is 15 bits(acc1) + 2 bits(acc2) mod 2^32: two integer ops a word, no constant. The int32 prefix word of
// every k32 step is rebuilt in registers from those bits, never through memory; it is the word IMMA's s32 accumulator
// holds after the same step.
//
// Arms on one mainloop (128 x 128 CTA tile, 8 warps of 64 x 32, cp.async stages, ldmatrix): imma (IMMA.16832.S8.S8),
// k3, k4; each with FOLD (every k32 word folded into 4 sinks a thread, s = s 0x9E3779B1 + w, the receipt's read of every
// word) or without (the final word only; the k32 words are still built, since the final word is their sum). DUMP mode
// stores every k32 word (the gate).
//
// Modes:
//   emu8 gate DIR            the adversarial distributions at K up to 16384: each arm's every prefix word under 0xA5
//                            poisoning against the host's int64 words, the no-write control, emu == imma byte for byte;
//                            operands and words to DIR for emu8.py's recheck
//   emu8 time REPS M N K...  per shape (triples): the final-word gate of every arm against cuBLASLt int8 (poisoned, with
//                            control), then interleaved items: cuBLASLt int8 (arm 0), cuBLASLt FP8, imma, k3, k4 (and
//                            _fold), the limb split
//   emu8 hostdraw D S R K    the generator on the host (no GPU): distribution D, side S, R x K bytes to stdout
// Build: nvcc -std=c++20 -O3 -gencode arch=compute_120a,code=sm_120a emu8.cu -lcublasLt -o emu8
#include <cublasLt.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../gpu_kit/bench.cuh"

#define DEV __device__ __forceinline__

constexpr uint32_t C0_BITS = 0x4B400000u, C1_BITS = 0x4B7BAB6Cu, C2_BITS = 0x49E07A56u;
static_assert(15u * C1_BITS + 2u * C2_BITS == 0u, "k3's increment carries no constant");
constexpr uint32_t FOLD_MUL = 0x9E3779B1u;
constexpr int IMMA = 0, K3 = 1, K4 = 2;
constexpr int BM = 128, BN = 128, THREADS = 256;

// ---- operands: a counter hash a (side, row, k); emu8.py's `draw` is the host's copy ---------------------------------

__host__ __device__ inline uint64_t smix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__host__ __device__ inline uint64_t hkey(uint64_t seed, int side, uint32_t r, uint32_t k) {
  return smix(seed * 0x9E3779B97F4A7C15ull ^ ((uint64_t)side << 62) ^ ((uint64_t)r << 24) ^ k);
}
// The distributions (emu8.py's DISTS, in this order)
enum Dist { UNIFORM, MAXLIMB, SAMESIGN, RUNS, MIXED, DITHER, ZERO_A, NDIST };
__host__ __device__ inline const int8_t* maxlimb_set() {
  // |h| = 8 (±127, 120, -121, -128), |l| = 8 (-120: m = -15, 104, -104 + ... ), m = 14 (119), and the small limbs' edges
  static constexpr int8_t S[16] = {127, -127, 120, -121, 119, -120, -128, 104, -104, 8, -8, -9, 7, 112, -113, 121};
  return S;
}
__host__ __device__ inline int8_t draw(int dist, uint64_t seed, int side, uint32_t r, uint32_t k) {
  const uint64_t v = hkey(seed, side, r, k);
  switch (dist) {
    case UNIFORM: return (int8_t)((int)(v % 255) - 127);
    case MAXLIMB: return maxlimb_set()[v & 15];
    case SAMESIGN: {  // every product positive and large: both operands from {-120, -127, -128} (A) / {-120, -127} (B)
      static constexpr int8_t S[4] = {-120, -127, -128, -120};
      return side ? S[v & 1] : S[(v & 3)];
    }
    case RUNS: {  // max-limb magnitudes, one sign per 1024-step run (per row), so runs of 1024 same-sign products
      const uint64_t s = hkey(seed ^ 0x5151, side, r, k >> 10);
      const int m = (int)(v & 7);
      static constexpr int8_t S[8] = {127, 120, 119, 104, 121, 112, 127, 120};
      return (int8_t)((s & 1) ? -S[m] : S[m]);
    }
    case MIXED: {  // max magnitude, independent signs
      static constexpr int8_t S[4] = {127, 120, 119, 104};
      return (int8_t)((v >> 8 & 1) ? -S[v & 3] : S[v & 3]);
    }
    case DITHER: {  // Ã = clamp(a, -127, 126) + R; B̃ = b + U[j, k mod 32], |b| <= 63, U in ±1..±64
      if (side == 0) {
        int a = (int)(v % 254) - 127;  // [-127, 126]
        return (int8_t)(a + (int)(v >> 40 & 1));
      }
      const int b = (int)(v % 127) - 63;
      const uint64_t u = hkey(seed ^ 0x7777, 2, r, k & 31);
      const int um = (int)(u & 63) + 1;
      return (int8_t)(b + ((u >> 6 & 1) ? -um : um));
    }
    case ZERO_A: {  // the cheater's operands: A = 0, B = 0, so Ã = R and B̃ = S
      if (side == 0) return (int8_t)(v >> 40 & 1);
      const uint64_t u = hkey(seed ^ 0x7777, 2, r, k & 31);
      const int um = (int)(u & 63) + 1;
      return (int8_t)((u >> 6 & 1) ? -um : um);
    }
  }
  return 0;
}
__global__ void gen(int8_t* X, int rows, int K, int dist, uint64_t seed, int side) {
  const size_t n = (size_t)rows * K;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    X[i] = draw(dist, seed, side, (uint32_t)(i / K), (uint32_t)(i % K));
}

// ---- the limb split: int8 rows to the k3 and k4 planes, [row][k / 32][plane][32] E4M3 bytes -------------------------

__host__ __device__ inline void limbs(int x, int& h, int& l) {
  h = (x + 8) >> 4;  // floor
  l = x - 16 * h;
}
DEV uint8_t e4m3(int v) { return __nv_cvt_float_to_fp8((float)v, __NV_SATFINITE, __NV_E4M3); }
// side 0 (A) or 1 (B): k3's A planes are [4h, m, -l], B's [4h', m', l']; k4's [16h, l] on both sides
__global__ void split(const int8_t* X, uint8_t* P3, uint8_t* P4, int rows, int K, int side) {
  const size_t n = (size_t)rows * K;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / K, k = i % K, kb = k >> 5, e = k & 31;
    int h, l;
    limbs(X[i], h, l);
    uint8_t* p3 = P3 + r * (size_t)K * 3 + kb * 96 + e;
    p3[0] = e4m3(4 * h);
    p3[32] = e4m3(h + l);
    p3[64] = e4m3(side ? l : -l);
    uint8_t* p4 = P4 + r * (size_t)K * 2 + kb * 64 + e;
    p4[0] = e4m3(16 * h);
    p4[32] = e4m3(l);
  }
}

// ---- the mainloop -----------------------------------------------------------------------------------------------------

template <int ARM> struct Cfg {
  static constexpr int P = ARM == IMMA ? 1 : ARM == K3 ? 3 : 2;  // planes a k32 block
  static constexpr int S = ARM == IMMA ? 2 : 1;                   // k32 steps a stage
  static constexpr int SB = 32 * P * S;                           // bytes a row a stage
  static constexpr int PITCH = SB + 16;                           // PITCH / 16 odd: ldmatrix's 8 rows on distinct banks
  static constexpr int CPR = SB / 16;                             // 16-byte chunks a row a stage
  static constexpr int NS = ARM == K3 ? 3 : 4;                    // stages
  static constexpr int STAGE = (BM + BN) * PITCH;
  static constexpr int SMEM = NS * STAGE;
  static_assert((PITCH / 16) % 2 == 1, "pitch");
  static_assert((BM + BN) * CPR % THREADS == 0, "loads");
};

DEV void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g)); }
DEV void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> DEV void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
DEV void ldsm4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
DEV void imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// d = a b + c, c a separate input (k3 and k4 start each k32 step's accumulator from C0)
DEV void qmma(float* d, const uint32_t* a, const uint32_t* b, const float* c) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%10,%11,%12,%13};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
                 "f"(c[3]));
}
DEV uint32_t clk() {
  uint32_t c;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
  return c;
}

// A: [M][K P] bytes, B: [N][K P] (planes interleaved a k32 block); out: [M][N] the final word; sinks: 4 a thread a tile
// (FOLD); words: [K / 32][M][N] every k32 word (DUMP); cycles: a tile's mainloop clocks (the SASS gate's timed region)
template <int ARM, bool FOLD, bool DUMP>
__global__ void __launch_bounds__(THREADS, 1)
    gemm(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, int M, int N, int K, int32_t* __restrict__ out,
         uint32_t* __restrict__ sinks, int32_t* __restrict__ words, uint32_t* __restrict__ cycles) {
  using C = Cfg<ARM>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;  // 2 x 4 warps of 64 x 32
  // grouped raster: 8 M tiles a group
  const int tiles_n = N / BN, tiles_m = M / BM;
  const int t = blockIdx.x, group = t / (8 * tiles_n), first_m = group * 8,
            gm = min(tiles_m - first_m, 8), tm = first_m + (t % (8 * tiles_n)) % gm, tn = (t % (8 * tiles_n)) / gm;
  const size_t rowb = (size_t)K * C::P;
  const uint8_t* gA = A + (size_t)tm * BM * rowb;
  const uint8_t* gB = B + (size_t)tn * BN * rowb;
  const int T = K / (32 * C::S);

  auto load = [&](int st, int slot) {
    const uint32_t s0 = sbase + slot * C::STAGE;
#pragma unroll
    for (int i = 0; i < (BM + BN) * C::CPR / THREADS; i++) {
      const int c = tid + i * THREADS, row = c / C::CPR, col = c % C::CPR;
      const uint8_t* g = (row < BM ? gA + (size_t)row * rowb : gB + (size_t)(row - BM) * rowb) + (size_t)st * C::SB + col * 16;
      cp16(s0 + row * C::PITCH + col * 16, g);
    }
  };

  int32_t acc[4][4][4];  // imma: the s32 accumulator; k3, k4: the int32 word rebuilt every k32 step
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][j][e] = 0;
  uint32_t sink[4] = {0, 0, 0, 0};
  const float c0[4] = {__uint_as_float(C0_BITS), __uint_as_float(C0_BITS), __uint_as_float(C0_BITS),
                       __uint_as_float(C0_BITS)};
  const float c1[4] = {__uint_as_float(C1_BITS), __uint_as_float(C1_BITS), __uint_as_float(C1_BITS),
                       __uint_as_float(C1_BITS)};
  const float c2[4] = {__uint_as_float(C2_BITS), __uint_as_float(C2_BITS), __uint_as_float(C2_BITS),
                       __uint_as_float(C2_BITS)};

#pragma unroll
  for (int s = 0; s < C::NS - 1; s++) {
    if (s < T) load(s, s);
    cp_commit();
  }
  // ldmatrix lanes: A rows (lane & 7) + 8 (lane >> 3 & 1), bytes 16 (lane >> 4); B rows (lane & 7) + 8 (lane >> 4),
  // bytes 16 (lane >> 3 & 1)
  const uint32_t a_off = (wm * 64 + (lane & 7) + 8 * (lane >> 3 & 1)) * C::PITCH + 16 * (lane >> 4);
  const uint32_t b_off = (BM + wn * 32 + (lane & 7) + 8 * (lane >> 4)) * C::PITCH + 16 * (lane >> 3 & 1);
  const int g = lane >> 2, q = lane & 3;
  const uint32_t t0 = clk();
#pragma unroll 1
  for (int it = 0; it < T; it++) {
    cp_wait<C::NS - 2>();
    __syncthreads();
    if (it + C::NS - 1 < T) load(it + C::NS - 1, (it + C::NS - 1) % C::NS);
    cp_commit();
    const uint32_t s0 = sbase + (it % C::NS) * C::STAGE;
#pragma unroll
    for (int s = 0; s < C::S; s++) {
      uint32_t af[C::P][4][4], bf[C::P][4][2];
#pragma unroll
      for (int p = 0; p < C::P; p++) {
        const uint32_t kb = (s * C::P + p) * 32;
#pragma unroll
        for (int i = 0; i < 4; i++) ldsm4(af[p][i], s0 + a_off + i * 16 * C::PITCH + kb);
#pragma unroll
        for (int j = 0; j < 2; j++) {
          uint32_t r[4];
          ldsm4(r, s0 + b_off + j * 16 * C::PITCH + kb);
          bf[p][2 * j][0] = r[0]; bf[p][2 * j][1] = r[1]; bf[p][2 * j + 1][0] = r[2]; bf[p][2 * j + 1][1] = r[3];
        }
      }
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
          if constexpr (ARM == IMMA) {
            imma(acc[i][j], af[0][i], bf[0][j]);
          } else if constexpr (ARM == K3) {
            float f1[4], f2[4];
            qmma(f1, af[0][i], bf[0][j], c1);
            qmma(f1, af[2][i], bf[2][j], f1);
            qmma(f2, af[1][i], bf[1][j], c2);
#pragma unroll
            for (int e = 0; e < 4; e++)
              asm("mad.lo.u32 %0, %1, 15, %0;\n\tmad.lo.u32 %0, %2, 2, %0;"
                  : "+r"(acc[i][j][e]) : "r"(__float_as_uint(f1[e])), "r"(__float_as_uint(f2[e])));
          } else {
            float f[4];
            qmma(f, af[0][i], bf[0][j], c0);
            qmma(f, af[0][i], bf[1][j], f);
            qmma(f, af[1][i], bf[0][j], f);
            qmma(f, af[1][i], bf[1][j], f);
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] += (int32_t)(__float_as_uint(f[e]) - C0_BITS);
          }
        }
      if constexpr (FOLD) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) sink[e] = sink[e] * FOLD_MUL + (uint32_t)acc[i][j][e];
      }
      if constexpr (DUMP) {
        int32_t* w = words + (size_t)(it * C::S + s) * M * N;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
              const int row = tm * BM + wm * 64 + i * 16 + g + 8 * (e >> 1), col = tn * BN + wn * 32 + j * 8 + 2 * q + (e & 1);
              w[(size_t)row * N + col] = acc[i][j][e];
            }
      }
    }
  }
  const uint32_t t1 = clk();
  cp_wait<0>();
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int row = tm * BM + wm * 64 + i * 16 + g + 8 * (e >> 1), col = tn * BN + wn * 32 + j * 8 + 2 * q + (e & 1);
        out[(size_t)row * N + col] = acc[i][j][e];
      }
  if constexpr (FOLD) {
#pragma unroll
    for (int e = 0; e < 4; e++) sinks[((size_t)t * THREADS + tid) * 4 + e] = sink[e];
  }
  if (tid == 0 && cycles) cycles[t] = t1 - t0;
}

// ---- host ---------------------------------------------------------------------------------------------------------------

#define CKL(x)                                                                         \
  do {                                                                                 \
    cublasStatus_t s_ = (x);                                                           \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                 \
      fprintf(stderr, "ERR %s:%d %s: cublas %d\n", __FILE__, __LINE__, #x, (int)s_);   \
      exit(2);                                                                         \
    }                                                                                  \
  } while (0)

template <class T> T* dalloc(size_t n) {
  T* p;
  GK_CK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

struct Ops {
  int M, N, K;
  int8_t *A, *B;
  uint8_t *A3, *B3, *A4, *B4;
  Ops(int M_, int N_, int K_) : M(M_), N(N_), K(K_) {
    A = dalloc<int8_t>((size_t)M * K); B = dalloc<int8_t>((size_t)N * K);
    A3 = dalloc<uint8_t>((size_t)M * K * 3); B3 = dalloc<uint8_t>((size_t)N * K * 3);
    A4 = dalloc<uint8_t>((size_t)M * K * 2); B4 = dalloc<uint8_t>((size_t)N * K * 2);
  }
  ~Ops() { cudaFree(A); cudaFree(B); cudaFree(A3); cudaFree(B3); cudaFree(A4); cudaFree(B4); }
  void draw(int dist, uint64_t seed, cudaStream_t s) {
    gen<<<1024, 256, 0, s>>>(A, M, K, dist, seed, 0);
    gen<<<1024, 256, 0, s>>>(B, N, K, dist, seed, 1);
    split_all(s);
  }
  void split_all(cudaStream_t s) {
    split<<<2048, 256, 0, s>>>(A, A3, A4, M, K, 0);
    split<<<2048, 256, 0, s>>>(B, B3, B4, N, K, 1);
  }
};

template <int ARM, bool FOLD, bool DUMP> void set_smem() {
  GK_CK(cudaFuncSetAttribute(gemm<ARM, FOLD, DUMP>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<ARM>::SMEM));
}
template <int ARM, bool FOLD, bool DUMP>
void launch(const Ops& o, int32_t* out, uint32_t* sinks, int32_t* words, uint32_t* cycles, cudaStream_t s) {
  const uint8_t* a = ARM == IMMA ? (const uint8_t*)o.A : ARM == K3 ? o.A3 : o.A4;
  const uint8_t* b = ARM == IMMA ? (const uint8_t*)o.B : ARM == K3 ? o.B3 : o.B4;
  gemm<ARM, FOLD, DUMP><<<(o.M / BM) * (o.N / BN), THREADS, Cfg<ARM>::SMEM, s>>>(a, b, o.M, o.N, o.K, out, sinks, words, cycles);
}
void set_all_smem() {
  set_smem<IMMA, false, false>(); set_smem<IMMA, true, false>(); set_smem<IMMA, false, true>();
  set_smem<K3, false, false>(); set_smem<K3, true, false>(); set_smem<K3, false, true>();
  set_smem<K4, false, false>(); set_smem<K4, true, false>(); set_smem<K4, false, true>();
}

// cuBLASLt: D (M x N row-major) = A (M x K row) B (N x K row)ᵀ, as column-major Dᵀ = Bᵀᵀ... (TN: opA = T on B, opB = N on A)
struct Lt {
  cublasLtHandle_t h{};
  cublasLtMatmulDesc_t d{};
  cublasLtMatrixLayout_t la{}, lb{}, ld{};
  cublasLtMatmulAlgo_t algo{};
  void* ws = nullptr;
  size_t wsb = 64 << 20;
  bool ok = false;
  bool fp8 = false;
  Lt(int M, int N, int K, bool fp8_) : fp8(fp8_) {
    CKL(cublasLtCreate(&h));
    ws = dalloc<uint8_t>(wsb);
    const cudaDataType_t in = fp8 ? CUDA_R_8F_E4M3 : CUDA_R_8I, outt = fp8 ? CUDA_R_16BF : CUDA_R_32I;
    CKL(cublasLtMatmulDescCreate(&d, fp8 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32I, fp8 ? CUDA_R_32F : CUDA_R_32I));
    const cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CKL(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof Nn));
    CKL(cublasLtMatrixLayoutCreate(&la, in, K, N, K));
    CKL(cublasLtMatrixLayoutCreate(&lb, in, K, M, K));
    CKL(cublasLtMatrixLayoutCreate(&ld, outt, N, M, N));
    cublasLtMatmulPreference_t p;
    CKL(cublasLtMatmulPreferenceCreate(&p));
    CKL(cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb));
    cublasLtMatmulHeuristicResult_t r[8];
    int n = 0;
    if (cublasLtMatmulAlgoGetHeuristic(h, d, la, lb, ld, ld, p, 8, r, &n) == CUBLAS_STATUS_SUCCESS && n > 0) {
      algos.assign(r, r + n);
      algo = r[0].algo;
      ok = true;
    }
    cublasLtMatmulPreferenceDestroy(p);
  }
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
  void run(const void* A, const void* B, void* D, cudaStream_t s) const {
    if (fp8) {
      const float one = 1, zero = 0;
      CKL(cublasLtMatmul(h, d, &one, B, la, A, lb, &zero, D, ld, D, ld, &algo, ws, wsb, s));
    } else {
      const int32_t one = 1, zero = 0;
      CKL(cublasLtMatmul(h, d, &one, B, la, A, lb, &zero, D, ld, D, ld, &algo, ws, wsb, s));
    }
  }
};

// The host's prefix words: words[l][i][j] = sum over k < 32 (l + 1) of A[i][k] B[j][k], in int64, as s32
std::vector<int32_t> host_words(const std::vector<int8_t>& A, const std::vector<int8_t>& B, int M, int N, int K) {
  const int L = K / 32;
  std::vector<int32_t> w((size_t)L * M * N);
  std::vector<int64_t> run((size_t)M * N, 0);
  for (int l = 0; l < L; l++)
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++) {
        int64_t s = 0;
        const int8_t* a = &A[(size_t)i * K + 32 * l];
        const int8_t* b = &B[(size_t)j * K + 32 * l];
        for (int m = 0; m < 32; m++) s += (int64_t)a[m] * b[m];
        run[(size_t)i * N + j] += s;
        w[((size_t)l * M + i) * N + j] = (int32_t)(uint32_t)(uint64_t)run[(size_t)i * N + j];
      }
  return w;
}

std::vector<unsigned char> fetch(const void* d, size_t n) {
  std::vector<unsigned char> h(n);
  GK_CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost));
  return h;
}
void write_file(const std::string& p, const void* h, size_t n) {
  FILE* f = fopen(p.c_str(), "wb");
  if (!f || fwrite(h, 1, n, f) != n) { fprintf(stderr, "ERR write %s\n", p.c_str()); exit(2); }
  fclose(f);
}

const char* DIST_NAMES[NDIST] = {"uniform", "maxlimb", "samesign", "runs", "mixed", "dither", "zero_a"};
const char* ARM_NAMES[3] = {"imma", "k3", "k4"};

int gate(const std::string& dir) {
  set_all_smem();
  cudaStream_t s;
  GK_CK(cudaStreamCreate(&s));
  struct Case { int dist, M, N, K; };
  std::vector<Case> cases;
  for (int d = 0; d < NDIST; d++) cases.push_back({d, 128, 128, 16384});
  cases.push_back({UNIFORM, 256, 384, 2048});
  cases.push_back({DITHER, 384, 256, 4096});
  const uint64_t seed = 0xE8A5u;
  int bad = 0;
  std::filesystem::create_directories(dir);
  for (size_t c = 0; c < cases.size(); c++) {
    const Case& cs = cases[c];
    Ops o(cs.M, cs.N, cs.K);
    o.draw(cs.dist, seed, s);
    GK_CK(cudaStreamSynchronize(s));
    const std::vector<unsigned char> ha = fetch(o.A, (size_t)cs.M * cs.K), hb = fetch(o.B, (size_t)cs.N * cs.K);
    std::vector<int8_t> A(ha.begin(), ha.end()), B(hb.begin(), hb.end());
    const std::vector<int32_t> ref = host_words(A, B, cs.M, cs.N, cs.K);
    const size_t wbytes = ref.size() * 4;
    int32_t* words = dalloc<int32_t>(ref.size());
    int32_t* out = dalloc<int32_t>((size_t)cs.M * cs.N);
    char name[64];
    snprintf(name, sizeof name, "c%02zu-%s-%dx%dx%d", c, DIST_NAMES[cs.dist], cs.M, cs.N, cs.K);
    const std::string cdir = dir + "/" + name;
    std::filesystem::create_directories(cdir);
    write_file(cdir + "/A.bin", ha.data(), ha.size());
    write_file(cdir + "/B.bin", hb.data(), hb.size());
    std::vector<unsigned char> imma_words;
    for (int arm = 0; arm < 3; arm++) {
      auto run = [&](cudaStream_t st) {
        if (arm == IMMA) launch<IMMA, false, true>(o, out, nullptr, words, nullptr, st);
        else if (arm == K3) launch<K3, false, true>(o, out, nullptr, words, nullptr, st);
        else launch<K4, false, true>(o, out, nullptr, words, nullptr, st);
        GK_CK(cudaGetLastError());
      };
      const gk::PoisonGate g = gk::poison_gate(words, wbytes, s, run, [&](const void* h, size_t n) {
        return n == wbytes && std::memcmp(h, ref.data(), n) == 0;
      });
      run(s);
      const std::vector<unsigned char> hw = fetch(words, wbytes);
      size_t diff = 0;
      for (size_t i = 0; i < ref.size(); i++) diff += std::memcmp(&hw[4 * i], &ref[i], 4) != 0;
      bool eq_imma = true;
      if (arm == IMMA) imma_words = hw;
      else eq_imma = hw == imma_words;
      write_file(cdir + "/" + ARM_NAMES[arm] + ".bin", hw.data(), hw.size());
      const bool ok = g.ok() && diff == 0 && eq_imma && g.left == 0;
      bad += !ok;
      printf("{\"emu8\": \"gate\", \"case\": \"%s\", \"dist\": \"%s\", \"M\": %d, \"N\": %d, \"K\": %d, \"arm\": \"%s\", "
             "\"words\": %zu, \"accepted\": %s, \"control_rejected\": %s, \"left\": %zu, \"diff\": %zu, "
             "\"equal_imma\": %s, \"ok\": %s}\n",
             name, DIST_NAMES[cs.dist], cs.M, cs.N, cs.K, ARM_NAMES[arm], ref.size(), g.accepted ? "true" : "false",
             g.control_rejected ? "true" : "false", g.left, diff, eq_imma ? "true" : "false", ok ? "true" : "false");
      fflush(stdout);
    }
    cudaFree(words);
    cudaFree(out);
  }
  printf("GATE %s (%d bad)\n", bad ? "FAIL" : "PASS", bad);
  return bad ? 3 : 0;
}

__global__ void count_diff(const int32_t* a, const int32_t* b, size_t n, unsigned long long* c) {
  unsigned long long k = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) k += a[i] != b[i];
  if (k) atomicAdd(c, k);
}

int time_shapes(int reps, const std::vector<int>& dims) {
  set_all_smem();
  cudaStream_t s;
  GK_CK(cudaStreamCreate(&s));
  gk::Nvml nv(0);
  printf("{\"emu8\": \"device\", \"uuid\": \"%s\", \"nvml\": %s}\n", nv.uuid().c_str(), nv.ok() ? "true" : "false");
  int bad = 0;
  for (size_t d = 0; d + 2 < dims.size(); d += 3) {
    const int M = dims[d], N = dims[d + 1], K = dims[d + 2];
    if (M % BM || N % BN || K % 64) { fprintf(stderr, "ERR shape %dx%dx%d: M, N multiples of 128, K of 64\n", M, N, K); return 2; }
    Ops o(M, N, K);
    o.draw(DITHER, 0xD17Eu + d, s);
    const size_t mn = (size_t)M * N, ntiles = (size_t)(M / BM) * (N / BN);
    int32_t* ref = dalloc<int32_t>(mn);
    int32_t* out = dalloc<int32_t>(mn);
    uint32_t* sinks = dalloc<uint32_t>(ntiles * THREADS * 4);
    uint32_t* sinks_ref = dalloc<uint32_t>(ntiles * THREADS * 4);
    void* d_bf16 = dalloc<uint16_t>(mn);
    unsigned long long* cnt = dalloc<unsigned long long>(1);
    Lt lt(M, N, K, false), lt8(M, N, K, true);
    if (!lt.ok) { fprintf(stderr, "ERR cuBLASLt int8 has no algorithm at %dx%dx%d\n", M, N, K); return 2; }
    lt.run(o.A, o.B, ref, s);
    // imma's fold sinks are the reference for every arm's (the same words in the same order)
    launch<IMMA, true, false>(o, out, sinks_ref, nullptr, nullptr, s);
    GK_CK(cudaStreamSynchronize(s));
    const std::vector<unsigned char> href = fetch(ref, mn * 4), hsref = fetch(sinks_ref, ntiles * THREADS * 16);
    struct G { const char* name; std::function<void(cudaStream_t)> run; bool fold; };
    std::vector<G> gates = {
        {"imma", [&](cudaStream_t st) { launch<IMMA, false, false>(o, out, nullptr, nullptr, nullptr, st); }, false},
        {"k3", [&](cudaStream_t st) { launch<K3, false, false>(o, out, nullptr, nullptr, nullptr, st); }, false},
        {"k4", [&](cudaStream_t st) { launch<K4, false, false>(o, out, nullptr, nullptr, nullptr, st); }, false},
        {"imma_fold", [&](cudaStream_t st) { launch<IMMA, true, false>(o, out, sinks, nullptr, nullptr, st); }, true},
        {"k3_fold", [&](cudaStream_t st) { launch<K3, true, false>(o, out, sinks, nullptr, nullptr, st); }, true},
        {"k4_fold", [&](cudaStream_t st) { launch<K4, true, false>(o, out, sinks, nullptr, nullptr, st); }, true},
    };
    bool shape_ok = true;
    for (G& g : gates) {
      const gk::PoisonGate pg = gk::poison_gate(out, mn * 4, s, g.run, [&](const void* h, size_t n) {
        return std::memcmp(h, href.data(), n) == 0;
      });
      bool sinks_ok = true;
      if (g.fold) {
        const gk::PoisonGate ps = gk::poison_gate(sinks, ntiles * THREADS * 16, s, g.run, [&](const void* h, size_t n) {
          return std::memcmp(h, hsref.data(), n) == 0;
        });
        sinks_ok = ps.ok();
      }
      const bool ok = pg.ok() && sinks_ok;
      shape_ok &= ok;
      printf("{\"emu8\": \"final_gate\", \"M\": %d, \"N\": %d, \"K\": %d, \"arm\": \"%s\", \"accepted\": %s, "
             "\"control_rejected\": %s, \"left\": %zu, \"sinks_ok\": %s, \"ok\": %s}\n",
             M, N, K, g.name, pg.accepted ? "true" : "false", pg.control_rejected ? "true" : "false", pg.left,
             sinks_ok ? "true" : "false", ok ? "true" : "false");
      fflush(stdout);
    }
    if (!shape_ok) { bad++; printf("{\"emu8\": \"skip_timing\", \"M\": %d, \"N\": %d, \"K\": %d}\n", M, N, K); continue; }
    std::vector<gk::Arm> arms = {
        {"cublaslt_i8", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) lt.run(o.A, o.B, ref, st); }},
        {"imma", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) launch<IMMA, false, false>(o, out, nullptr, nullptr, nullptr, st); }},
        {"imma_fold", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) launch<IMMA, true, false>(o, out, sinks, nullptr, nullptr, st); }},
        {"k3", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) launch<K3, false, false>(o, out, nullptr, nullptr, nullptr, st); }},
        {"k3_fold", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) launch<K3, true, false>(o, out, sinks, nullptr, nullptr, st); }},
        {"k4", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) launch<K4, false, false>(o, out, nullptr, nullptr, nullptr, st); }},
        {"k4_fold", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) launch<K4, true, false>(o, out, sinks, nullptr, nullptr, st); }},
        {"split", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) o.split_all(st); }},
    };
    if (lt8.ok)
      arms.push_back({"cublaslt_fp8", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) lt8.run(o.A4, o.B4, d_bf16, st); }});
    const gk::Timing tm = gk::timed_items(arms, reps, s, nv);
    printf("{\"emu8\": \"shape\", \"M\": %d, \"N\": %d, \"K\": %d, \"steady\": %s, \"warmup_ms\": %.1f, \"fp8_lt\": %s}\n", M, N,
           K, tm.steady ? "true" : "false", tm.warmup_ms, lt8.ok ? "true" : "false");
    gk::summary(tm, arms);
    gk::items_jsonl(tm, arms, stdout);
    // the final words still match after the timed items (the last call of each arm wrote `out`)
    launch<K3, false, false>(o, out, nullptr, nullptr, nullptr, s);
    GK_CK(cudaMemsetAsync(cnt, 0, 8, s));
    count_diff<<<1024, 256, 0, s>>>(out, ref, mn, cnt);
    unsigned long long h = 0;
    GK_CK(cudaMemcpyAsync(&h, cnt, 8, cudaMemcpyDeviceToHost, s));
    GK_CK(cudaStreamSynchronize(s));
    printf("{\"emu8\": \"after\", \"M\": %d, \"N\": %d, \"K\": %d, \"k3_diff_vs_cublaslt\": %llu}\n", M, N, K, h);
    bad += h != 0;
    cudaFree(ref); cudaFree(out); cudaFree(sinks); cudaFree(sinks_ref); cudaFree(d_bf16); cudaFree(cnt);
  }
  return bad ? 3 : 0;
}

int main(int argc, char** argv) {
  if (argc >= 3 && std::string(argv[1]) == "gate") return gate(argv[2]);
  if (argc == 6 && std::string(argv[1]) == "hostdraw") {  // the generator on the host, no GPU: DIST SIDE ROWS K, bytes
    const int dist = atoi(argv[2]), side = atoi(argv[3]), rows = atoi(argv[4]), K = atoi(argv[5]);
    std::vector<int8_t> x((size_t)rows * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = draw(dist, 0xE8A5u, side, (uint32_t)(i / K), (uint32_t)(i % K));
    return fwrite(x.data(), 1, x.size(), stdout) == x.size() ? 0 : 2;
  }
  if (argc >= 6 && std::string(argv[1]) == "time") {
    std::vector<int> dims;
    for (int i = 3; i < argc; i++) dims.push_back(atoi(argv[i]));
    return time_shapes(atoi(argv[2]), dims);
  }
  fprintf(stderr, "usage: emu8 gate DIR | emu8 time REPS M N K [M N K ...]\n");
  return 2;
}
