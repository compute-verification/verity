// Is B200's kind::mxf4nvf4 exact when a row's four k16 blocks
// carry different integer UE4M3 scales, from +0 and from the 1.5 2^23 binade? lowbit_dc's fpx.cu filled every scale byte
// alike, so a block's scale factored out of the whole MMA. With mixed scales the four scaled block partials of one MMA
// differ in magnitude, and a narrow internal adder would drop the small ones; uniform scales can't show that.
//
// sm_100a: tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X, E2M1 (k64, packed), UE4M3 scales, M = 128, N = 32,
// FP32 accumulators in tensor memory. Every TMEM lane and column of a side's scale region holds the same 32-bit word of
// four scale bytes, so the one unknown of the layout is which byte a block reads. The first launch calibrates it: row r
// of A (column j of B) is nonzero in block r % 4 (j % 4) alone, and the output names that block's byte. The second runs
// the chains with that map, each output against the int64 sum the same thread computes:
//
//   chain   nsteps MMAs into one accumulator from C0 (+0, or 1.5 2^(23 + e) whose ulp is the unit 2^e), E2M1 operands
//           from a counter hash; at K = 64 2^j the CTA counts outputs whose MMA-boundary partials were all FP32-
//           representable (rep), those among them that differ from the exact sum (bad_rep), and every output that
//           differs (inexact).
//   mixwin  one MMA: row r's big block r % 4 near E2M1's top (15 products of 6 x 6 and one of 6 x 0.5), every other
//           block one product of 0.5 x 0.5, each at its own scale product: the small blocks must survive the big one's
//           alignment (at scale products to 225 the result stays under 2^24 quarter units from +0 and from the binade,
//           so FP32 holds it exactly).
//
// A control issues no MMA (the accumulator keeps C0); the host requires that it count as wrong.
//
//   nvf4_mixed run OUT   calibrate, then every config: OUT/calibration.json, OUT/records.jsonl, OUT/summary.json
//   nvf4_mixed list      the configs (no GPU)
// Exit 0: calibrated, every exactness verdict as stated (no bad_rep, mixwin exact, controls caught); 3: the scale layout
// isn't a block-to-byte map; 4: a bad_rep, an inexact mixwin or a missed control; 2: no device.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a nvf4_mixed.cu -o nvf4_mixed
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../dc/dc.cuh"

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

enum Dist { D_UNI = 0, D_MAX = 1, D_SIGN = 2, D_CANCEL = 3, D_MIXWIN = 4, D_CALA = 5, D_CALB = 6 };
static const char* DISTS[] = {"uniform", "max", "max_signs", "cancel", "mixwin", "cal_a", "cal_b"};

constexpr int BM = 128;   // M of every MMA
constexpr int TN = 32;    // N of every MMA
constexpr int KBY = 32;   // operand bytes a row a MMA: E2M1 k64, packed
constexpr int KS = 64;    // k a MMA
constexpr int MAXCK = 16; // checkpoints a config

struct Cfg {
  int dist, grid, c0mode, nsteps, control;
  uint32_t seed;
  uint32_t sfa, sfb;  // the scale words, bytes 0..3 (the same in every lane and column)
  int prod[4];        // block b's scale product, from the calibrated map (0 in the calibration launch)
  int e;              // the unit 2^e: -2 for E2M1 halves, 0 when every value is an integer
  uint32_t c0bits;
  long long c0u;      // C0 / 2^e, exact
};
struct Rec {
  long long K;
  int n, rep, bad_rep, inexact;
  float maxerr;
};

__host__ __device__ inline uint32_t mix32(uint32_t x) { return dc_mix32(x); }
__host__ __device__ inline uint32_t hsh(uint32_t seed, int step, int side, int row, int k) {
  return mix32(mix32(mix32(seed ^ ((uint32_t)step * 0x9E3779B1u)) + (uint32_t)side * 0x85EBCA77u + (uint32_t)row) *
                   0xC2B2AE3Du +
               (uint32_t)k);
}
// An operand in half units (E2M1 x 2): grid 0 every E2M1 value, grid 1 its integers.
__host__ __device__ inline int grid_val(int grid, uint32_t r) {
  const int x2[15] = {-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12};
  const int in[11] = {-12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12};
  return grid == 0 ? x2[r % 15u] : in[r % 11u];
}
__host__ __device__ inline int val2(const Cfg& c, int step, int side, int row, int k) {
  const int blk = k / 16;
  switch (c.dist) {
    case D_UNI: return grid_val(c.grid, hsh(c.seed, step, side, row, k));
    case D_MAX: return 12;
    case D_SIGN: return (hsh(c.seed, step, side, row, k) & 1u) ? 12 : -12;
    case D_CANCEL: {
      const int v = grid_val(c.grid, hsh(c.seed, step & ~1, side, row, k));
      return side == 1 && (step & 1) ? -v : v;
    }
    case D_MIXWIN: {
      // B is 6 but 0.5 at each block's last k; A's row r is 6 across its big block r % 4 and 0.5 at the other blocks'
      // last k: 15 x 36 + 3 in the big block, 0.25 in each small one.
      const bool last = k % 16 == 15;
      if (side == 1) return last ? 1 : 12;
      return blk == row % 4 ? 12 : (last ? 1 : 0);
    }
    case D_CALA: return side == 0 ? (blk == row % 4 && k % 16 == 0 ? 2 : 0) : 2;
    case D_CALB: return side == 1 ? (blk == row % 4 && k % 16 == 0 ? 2 : 0) : 2;
    default: return 0;
  }
}
__host__ __device__ inline uint8_t enc_e2m1(int v2, int* bad) {
  const int t[8] = {0, 1, 2, 3, 4, 6, 8, 12};  // E2M1 magnitudes in half units
  const int m = v2 < 0 ? -v2 : v2;
  for (int i = 0; i < 8; i++)
    if (t[i] == m) return (uint8_t)((v2 < 0 ? 8 : 0) | i);
  *bad += 1;
  return 0;
}
__host__ __device__ inline bool rep24(long long P) {
  if (P == 0) return true;
  unsigned long long a = P < 0 ? (unsigned long long)(-P) : (unsigned long long)P;
  while (!(a & 1ull)) a >>= 1;
  return a < (1ull << 24);
}

struct Tally {
  int rep, bad_rep, inexact;
  float maxerr;
};

#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
DC_DEV void umma_nvf4(uint32_t d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n}\n" ::"r"(d),
      "l"(da), "l"(db), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc)
      : "memory");
}
DC_DEV void tmem_fill(uint32_t taddr, uint32_t v, int cols) {
  uint32_t r[16];
#pragma unroll
  for (int i = 0; i < 16; i++) r[i] = v;
  for (int u = 0; u < cols / 16; u++) tmem_st16(taddr + 16 * u, r);
}
#endif
// Block-scaled instruction descriptor: A and B E2M1 (1) at [7,10) and [10,13), N >> 3 at [17,23), UE4M3 scales (0) at
// 23, M >> 4 at [24,29), both scale-factor ids 0 (fpx.cu's idesc_bs).
DC_HD constexpr uint32_t idesc_nvf4(uint32_t m, uint32_t n) {
  return (1u << 7) | (1u << 10) | ((n >> 3) << 17) | ((m >> 4) << 24);
}

__global__ void __launch_bounds__(128) nvf4mix(const Cfg* cfgs, Rec* recs, uint32_t* dump, int* encbad_out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  const Cfg c = cfgs[blockIdx.x];
  __shared__ __align__(1024) uint8_t sA[BM * KBY];
  __shared__ __align__(1024) uint8_t sB[TN * KBY];
  __shared__ int8_t vA[BM][KS];
  __shared__ int8_t vB[TN][KS];
  __shared__ uint64_t bar;
  __shared__ uint32_t tslot;
  __shared__ Tally tally;
  __shared__ int encbad;
  const uint32_t w = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0);
  const int row = threadIdx.x;
  if (threadIdx.x == 0) {
    mbar_init(&bar, 1);
    fence_mbar_init();
    encbad = 0;
  }
  if (w == 0) tmem_alloc(&tslot, 128);
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = tslot, trow = tmem + ((32u * w) << 16);
  // cols 0-31 the accumulators, 64-95 A's scale words, 96-127 B's
  tmem_fill(trow + 64, c.sfa, 32);
  tmem_fill(trow + 96, c.sfb, 32);
  tmem_fill(trow, c.c0bits, 32);
  tmem_wait_st();
  long long P[TN];
  int rep[TN];
  for (int j = 0; j < TN; j++) P[j] = c.c0u, rep[j] = 1;
  int ck = 0, myenc = 0;
  const uint32_t idesc = idesc_nvf4(BM, TN);
  for (int s = 0; s < c.nsteps; s++) {
    for (int side = 0; side < 2; side++) {
      const int rows = side ? TN : BM;
      uint8_t* base = side ? sB : sA;
      for (int idx = threadIdx.x; idx < rows * KBY; idx += blockDim.x) {
        const int r = idx / KBY, b = idx % KBY;
        const int v0 = val2(c, s, side, r, 2 * b), v1 = val2(c, s, side, r, 2 * b + 1);
        (side ? vB : vA)[r][2 * b] = (int8_t)v0;
        (side ? vB : vA)[r][2 * b + 1] = (int8_t)v1;
        base[kmaj_off(r, b, KBY)] = enc_e2m1(v0, &myenc) | (uint8_t)(enc_e2m1(v1, &myenc) << 4);
      }
    }
    fence_async_smem();
    tc_before_sync();
    __syncthreads();
    tc_after_sync();
    if (w == 0) {
      if (elect_one()) {
        if (c.control) {
          asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(smem_u32(&bar))
                       : "memory");
        } else {
          umma_nvf4(tmem, umma_desc(smem_u32(sA), kmaj_lbo(), kmaj_sbo(KBY)),
                    umma_desc(smem_u32(sB), kmaj_lbo(), kmaj_sbo(KBY)), idesc, tmem + 64, tmem + 96, 1);
          umma_commit(&bar);
        }
      }
      __syncwarp();
    }
    mbar_wait(&bar, s & 1);
    tc_after_sync();
    // the step's exact sum in units 2^e: half-unit operands make quarter-unit products, times each block's scale
    for (int j = 0; j < TN; j++) {
      long long tq = 0;
      for (int b = 0; b < 4; b++) {
        int d = 0;
        for (int k = 16 * b; k < 16 * b + 16; k++) d += (int)vA[row][k] * (int)vB[j][k];
        tq += (long long)c.prod[b] * d;
      }
      P[j] += c.e == 0 ? tq / 4 : tq;
      if (!rep24(P[j])) rep[j] = 0;
    }
    const bool at_ck = ((s + 1) & s) == 0 || s == c.nsteps - 1;
    if (at_ck) {
      Tally t{};
      for (int h = 0; h < 2; h++) {
        uint32_t g[16];
        tmem_ld16(trow + 16 * h, g);
        tmem_wait_ld();
        for (int q = 0; q < 16; q++) {
          const int j = 16 * h + q;
          const double want = ldexp((double)P[j], c.e);
          const float gf = __uint_as_float(g[q]);
          const bool ok = (double)gf == want && !(want == 0 && g[q] != 0u);
          t.rep += rep[j];
          t.bad_rep += rep[j] && !ok;
          t.inexact += !ok;
          const float err = (float)fabs((double)gf - want);
          if (err > t.maxerr) t.maxerr = err;
          if (s == c.nsteps - 1) dump[((size_t)blockIdx.x * BM + row) * TN + j] = g[q];
        }
      }
      if (threadIdx.x == 0) tally = Tally{};
      __syncthreads();
      atomicAdd(&tally.rep, t.rep);
      atomicAdd(&tally.bad_rep, t.bad_rep);
      atomicAdd(&tally.inexact, t.inexact);
      atomicMax((int*)&tally.maxerr, __float_as_int(t.maxerr));
      __syncthreads();
      if (threadIdx.x == 0 && ck < MAXCK) {
        Rec* r = &recs[blockIdx.x * MAXCK + ck];
        r->K = (long long)(s + 1) * KS;
        r->n = BM * TN;
        r->rep = tally.rep;
        r->bad_rep = tally.bad_rep;
        r->inexact = tally.inexact;
        r->maxerr = tally.maxerr;
      }
      ck++;
    }
    tc_before_sync();
    __syncthreads();
    tc_after_sync();
  }
  atomicAdd(&encbad, myenc);
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (threadIdx.x == 0) atomicAdd(encbad_out, encbad);
  if (w == 0) tmem_dealloc(tmem, 128);
#endif
}

// ---- host ----------------------------------------------------------------------------------------------------------------
static double ue4m3(int b) {
  const int e = (b >> 3) & 15, m = b & 7;
  return e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
}
static int enc_ue4m3(int s) {  // an integer 1..448 that UE4M3 holds
  for (int b = 0; b < 127; b++)
    if (ue4m3(b) == s) return b;
  fprintf(stderr, "ERR UE4M3 has no %d\n", s);
  exit(2);
}
static uint32_t word(const int (&s)[4]) {
  uint32_t w = 0;
  for (int i = 0; i < 4; i++) w |= (uint32_t)enc_ue4m3(s[i]) << (8 * i);
  return w;
}
struct Pattern {
  const char* name;
  int a[4], b[4];  // the integer scales of bytes 0..3, A's and B's
};
// mixed integer scales to 10 (nvi10's), to 15 (nvi's), odd ones (nv's), one uniform pattern as the fpx.cu control
static const Pattern PATTERNS[] = {
    {"uniform1", {1, 1, 1, 1}, {1, 1, 1, 1}},       {"int10a", {1, 10, 3, 7}, {9, 2, 5, 1}},
    {"int10b", {10, 1, 10, 1}, {10, 10, 1, 1}},     {"int10c", {8, 6, 4, 2}, {2, 4, 6, 8}},
    {"int15", {15, 1, 1, 15}, {15, 1, 15, 1}},      {"odd15", {15, 13, 11, 9}, {1, 3, 5, 7}},
    {"pow2mix", {1, 2, 4, 8}, {8, 4, 2, 1}},
};
constexpr int NPAT = sizeof(PATTERNS) / sizeof(PATTERNS[0]);

static Cfg make(int dist, int grid, int c0mode, int nsteps, uint32_t seed, const Pattern& p, const int (&ma)[4],
                const int (&mb)[4]) {
  Cfg c{};
  c.dist = dist; c.grid = grid; c.c0mode = c0mode; c.nsteps = nsteps; c.seed = seed;
  c.sfa = word(p.a);
  c.sfb = word(p.b);
  for (int b = 0; b < 4; b++) c.prod[b] = ma[0] < 0 ? 0 : p.a[ma[b]] * p.b[mb[b]];
  c.e = grid == 0 ? -2 : 0;
  if (c0mode == 1) {
    float f = std::ldexp(1.5f, 23 + c.e);
    memcpy(&c.c0bits, &f, 4);
    c.c0u = 12582912;
  }
  return c;
}
static std::vector<Cfg> configs(const int (&ma)[4], const int (&mb)[4]) {
  std::vector<Cfg> v;
  const int chain_d[] = {D_UNI, D_UNI, D_UNI, D_MAX, D_SIGN, D_CANCEL};
  uint32_t seed = 1;
  for (int p = 0; p < NPAT; p++)
    for (int g = 0; g < 2; g++)
      for (int c0 = 0; c0 < 2; c0++)
        for (int d : chain_d) v.push_back(make(d, g, c0, c0 == 0 ? 4096 : 2048, seed++, PATTERNS[p], ma, mb));
  for (int p = 0; p < NPAT; p++)
    for (int c0 = 0; c0 < 2; c0++) v.push_back(make(D_MIXWIN, 0, c0, 1, seed++, PATTERNS[p], ma, mb));
  Cfg k = make(D_UNI, 0, 1, 64, seed++, PATTERNS[1], ma, mb);
  k.control = 1;
  v.push_back(k);
  return v;
}
static void cfg_json(FILE* f, int i, const Cfg& c, const char* pat) {
  fprintf(f,
          "\"cfg\": %d, \"dist\": \"%s\", \"pattern\": \"%s\", \"grid\": %d, \"c0\": \"%s\", \"c0bits\": \"0x%08X\", "
          "\"sfa\": \"0x%08X\", \"sfb\": \"0x%08X\", \"prod\": [%d, %d, %d, %d], \"unit_e\": %d, \"nsteps\": %d, "
          "\"control\": %d",
          i, DISTS[c.dist], pat, c.grid, c.c0mode ? "binade" : "zero", c.c0bits, c.sfa, c.sfb, c.prod[0], c.prod[1],
          c.prod[2], c.prod[3], c.e, c.nsteps, c.control);
}
// One launch; returns the records, the last step's outputs and the encoding errors.
static void launch(const std::vector<Cfg>& cfgs, std::vector<Rec>& R, std::vector<uint32_t>& D, int& encbad, float& ms) {
  const int n = (int)cfgs.size();
  Cfg* dC; Rec* dR; uint32_t* dD; int* dE;
  CK(cudaMalloc(&dC, n * sizeof(Cfg)));
  CK(cudaMalloc(&dR, (size_t)n * MAXCK * sizeof(Rec)));
  CK(cudaMalloc(&dD, (size_t)n * BM * TN * 4));
  CK(cudaMalloc(&dE, 4));
  CK(cudaMemcpy(dC, cfgs.data(), n * sizeof(Cfg), cudaMemcpyHostToDevice));
  CK(cudaMemset(dR, 0xA5, (size_t)n * MAXCK * sizeof(Rec)));
  CK(cudaMemset(dD, 0xA5, (size_t)n * BM * TN * 4));
  CK(cudaMemset(dE, 0, 4));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  nvf4mix<<<n, 128>>>(dC, dR, dD, dE);
  CK(cudaGetLastError());
  CK(cudaEventRecord(e1));
  CK(cudaDeviceSynchronize());
  CK(cudaEventElapsedTime(&ms, e0, e1));
  R.assign((size_t)n * MAXCK, Rec{});
  D.assign((size_t)n * BM * TN, 0);
  CK(cudaMemcpy(R.data(), dR, R.size() * sizeof(Rec), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&encbad, dE, 4, cudaMemcpyDeviceToHost));
  CK(cudaFree(dC)); CK(cudaFree(dR)); CK(cudaFree(dD)); CK(cudaFree(dE));
}

int main(int argc, char** argv) {
  const std::string mode = argc >= 2 ? argv[1] : "";
  const int ident[4] = {0, 1, 2, 3};
  if (mode == "list") {
    printf("{\"configs\": %zu, \"patterns\": %d}\n", configs(ident, ident).size(), NPAT);
    return 0;
  }
  if (mode != "run" || argc != 3) { fprintf(stderr, "usage: nvf4_mixed run OUT | nvf4_mixed list\n"); return 2; }
  const std::string out = argv[2];
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { fprintf(stderr, "ERR no device\n"); return 2; }
  if (p.major != 10) { fprintf(stderr, "ERR sm_%d%d: nvf4_mixed is for sm_100a and sm_103a\n", p.major, p.minor); return 2; }

  // calibration: byte i holds the integer CAL[i]; row r % 4 (A) or column j % 4 (B) is the only nonzero block, one
  // product of 1 x 1 there, so output (r, j) is that block's A scale times its B scale
  const int CAL[4] = {1, 2, 3, 5}, none[4] = {-1, -1, -1, -1};
  const Pattern ca{"cal", {1, 2, 3, 5}, {1, 1, 1, 1}}, cb{"cal", {1, 1, 1, 1}, {1, 2, 3, 5}};
  std::vector<Cfg> cal = {make(D_CALA, 1, 0, 1, 0, ca, none, none), make(D_CALB, 1, 0, 1, 0, cb, none, none)};
  std::vector<Rec> R;
  std::vector<uint32_t> D;
  int encbad = 0, encbad2 = 0;
  float ms0 = 0, ms = 0;
  launch(cal, R, D, encbad, ms0);
  int ma[4] = {-1, -1, -1, -1}, mb[4] = {-1, -1, -1, -1}, consistent = 1;
  for (int side = 0; side < 2; side++)
    for (int r = 0; r < BM; r++)
      for (int j = 0; j < TN; j++) {
        float gf;
        memcpy(&gf, &D[((size_t)side * BM + r) * TN + j], 4);
        const int blk = side == 0 ? r % 4 : j % 4;
        int byte = -1;
        for (int i = 0; i < 4; i++)
          if ((double)gf == CAL[i]) byte = i;
        int* m = side == 0 ? ma : mb;
        if (byte < 0 || (m[blk] >= 0 && m[blk] != byte)) consistent = 0;
        else m[blk] = byte;
      }
  for (int b = 0; b < 4; b++) consistent &= ma[b] >= 0 && mb[b] >= 0;
  FILE* fc = fopen((out + "/calibration.json").c_str(), "w");
  if (!fc) { fprintf(stderr, "ERR open %s\n", out.c_str()); return 2; }
  fprintf(fc, "{\"device\": \"%s\", \"cc\": \"%d.%d\", \"a_block_byte\": [%d, %d, %d, %d], \"b_block_byte\": [%d, %d, %d, %d], "
              "\"consistent\": %s, \"encoding_bad\": %d}\n",
          p.name, p.major, p.minor, ma[0], ma[1], ma[2], ma[3], mb[0], mb[1], mb[2], mb[3], consistent ? "true" : "false",
          encbad);
  fclose(fc);
  if (!consistent) { fprintf(stderr, "nvf4_mixed: the scale layout is not one block-to-byte map; see calibration.json\n"); return 3; }

  auto cfgs = configs(ma, mb);
  launch(cfgs, R, D, encbad2, ms);
  encbad += encbad2;
  FILE* fr = fopen((out + "/records.jsonl").c_str(), "w");
  long bad_rep = 0, mix_bad = 0, missed = 0, rep_checked = 0;
  const int npat_cfgs = NPAT * 2 * 2 * 6;
  for (int i = 0; i < (int)cfgs.size(); i++) {
    const Cfg& c = cfgs[i];
    const char* pat = c.control ? "control" : PATTERNS[i < npat_cfgs ? i / 24 : (i - npat_cfgs) / 2].name;
    const int nck = [&] { int k = 0; for (int s = 0; s < c.nsteps; s++) k += ((s + 1) & s) == 0 || s == c.nsteps - 1; return k; }();
    long cfg_bad = 0;
    for (int k = 0; k < nck && k < MAXCK; k++) {
      const Rec& r = R[(size_t)i * MAXCK + k];
      fprintf(fr, "{");
      cfg_json(fr, i, c, pat);
      fprintf(fr, ", \"K\": %lld, \"n\": %d, \"rep\": %d, \"bad_rep\": %d, \"inexact\": %d, \"maxerr\": %.9g}\n", r.K, r.n,
              r.rep, r.bad_rep, r.inexact, r.maxerr);
      cfg_bad += r.bad_rep + r.inexact;
      if (!c.control) {
        bad_rep += r.bad_rep;
        rep_checked += r.rep;
        if (c.dist == D_MIXWIN) mix_bad += r.inexact;
      }
    }
    if (c.control && cfg_bad == 0) missed++;
  }
  fclose(fr);
  const bool ok = bad_rep == 0 && mix_bad == 0 && missed == 0 && encbad == 0;
  FILE* fs = fopen((out + "/summary.json").c_str(), "w");
  fprintf(fs,
          "{\"device\": \"%s\", \"cc\": \"%d.%d\", \"configs\": %zu, \"kernel_ms\": %.3f, \"calibration_ms\": %.3f, "
          "\"a_block_byte\": [%d, %d, %d, %d], \"b_block_byte\": [%d, %d, %d, %d], \"encoding_bad\": %d, "
          "\"rep_checked\": %ld, \"bad_rep\": %ld, \"mixwin_inexact\": %ld, \"missed_controls\": %ld, \"as_stated\": %s}\n",
          p.name, p.major, p.minor, cfgs.size(), ms, ms0, ma[0], ma[1], ma[2], ma[3], mb[0], mb[1], mb[2], mb[3], encbad,
          rep_checked, bad_rep, mix_bad, missed, ok ? "true" : "false");
  fclose(fs);
  printf("{\"device\": \"%s\", \"configs\": %zu, \"bad_rep\": %ld, \"rep_checked\": %ld, \"mixwin_inexact\": %ld, "
         "\"missed_controls\": %ld, \"encoding_bad\": %d}\n",
         p.name, cfgs.size(), bad_rep, rep_checked, mix_bad, missed, encbad);
  return ok ? 0 : 4;
}
