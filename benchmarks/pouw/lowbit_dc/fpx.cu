// Are the datacenter FP8 and FP4 tensor kinds exact integer arithmetic, and do emu8's int8 routes hold there?
//
// sm_100a (B200): tcgen05.mma kind::f8f6f4 E4M3 (k32), kind::mxf4.block_scale.scale_vec::2X (E2M1, UE8M0, k64) and
// kind::mxf4nvf4.block_scale.scale_vec::4X (E2M1, UE4M3, k64), accumulators in tensor memory. sm_90a (H200): wgmma
// m64n32k32 E4M3, accumulators in registers. One CTA a config; each config is one of:
//
//   chain   nsteps MMAs into one accumulator from C0 (zero, the binade whose ulp is the product unit, NOBITS' 0x4B400000,
//           the binade above), operands from a counter hash over an exact grid; at K = kstep 2^j the CTA counts outputs
//           whose MMA-boundary partials were all FP32-representable (rep), those among them that differ from the exact
//           sum (bad_rep), every output that differs (inexact), and the outputs equal to a chain of one FP32 add of the
//           exact step sum per MMA, rounded toward zero (rz) or to nearest (rn). The exact sum is an int64 the same
//           thread computes from the same operands.
//   win     one MMA: product 2^16 beside product 2^(16 - g), g = 0..34 over the tile: the products' alignment window.
//   accwin  one MMA from C0 = 2^c (c = row mod 32) with one product 2^p: the window relative to the accumulator.
//   k3b k4b emu8's int8 routes, every k32 prefix word: balanced base-16 limbs through E4M3, acc restarted each step from
//           the binade constants (k3: C1 = 0x4B7BAB6C, C2 = 0x49E07A56; k4: C0 = 0x4B400000), word += 15 bits(acc1) +
//           2 bits(acc2) (k3) or bits(acc) - C0 (k4); bad_acc counts accumulators that differ from C + T, bad_word words
//           that differ from the s32 prefix of the int8 dot product.
//   k3z k4z the same with the accumulators restarted from +0 (tcgen05 enable_input_d = 0, no constant store), the word from
//           one FFMA a limb accumulator into the binade: bits(fma(16, acc2, fma(15, acc1, 1.5 2^23))) - 0x4B400000.
//
// A control config issues no MMA (the accumulator keeps C0); the host requires that it count as wrong.
//
//   fpx run OUT      every config of this device's family: OUT/records.jsonl, OUT/windows.jsonl, OUT/summary.json
//   fpx list         the configs (no GPU)
// Exit 0: every exactness verdict as stated (controls caught); 4: a bad_rep, a bad word or a missed control; 2: no device.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a (or 103a for B300, or 90a) fpx.cu -o fpx
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

enum Fmt { F_E4M3 = 0, F_MXF4 = 1, F_NVF4 = 2, F_K3B = 3, F_K4B = 4, F_K3Z = 5, F_K4Z = 6, F_K3H = 7, F_E4R = 8 };
enum Dist { D_UNI = 0, D_MAX = 1, D_SIGN = 2, D_EDGE = 3, D_CANCEL = 4, D_WIN = 5, D_ACCWIN = 6, D_LMIN = 7 };
static const char* FMTS[] = {"e4m3", "mxf4", "nvf4", "k3b", "k4b", "k3z", "k4z", "k3h", "e4r"};
static const char* DISTS[] = {"uniform", "max", "max_signs", "edge", "cancel", "win", "accwin", "l_min_h_max"};
static const char* C0S[] = {"zero", "binade", "nobits", "above"};

constexpr int TN = 32;    // N of every MMA
constexpr int KBY = 32;   // operand bytes a row a MMA: E4M3 k32, E2M1 k64 packed
constexpr int MAXCK = 16; // checkpoints a config
constexpr uint32_t C0B = 0x4B400000u, C1B = 0x4B7BAB6Cu, C2B = 0x49E07A56u;

struct Cfg {
  int fmt, grid, dist, c0mode, nsteps, control, dump;
  int sa, sb;       // FP4 scale bytes, one for every block of the side
  uint32_t seed;
  int s;            // operand value = v / s
  int num, e;       // product unit with the scales: num 2^e, num odd
  uint32_t c0bits;  // the starting accumulator (chain)
  long long c0u;    // C0 / 2^e, exact
};
struct Rec {
  long long K;
  int n, rep, bad_rep, inexact, rz, rn, negzero, bad_word, bad_acc;
  float maxerr;
};

__host__ __device__ inline bool is_fp4(int f) { return f == F_MXF4 || f == F_NVF4; }
__host__ __device__ inline bool is_emu(int f) { return f >= F_K3B && f <= F_K3H; }
__host__ __device__ inline bool is_word(int f) { return is_emu(f) || f == F_E4R; }  // a word checked every step
__host__ __device__ inline bool is_k3(int f) { return f == F_K3B || f == F_K3Z || f == F_K3H; }
__host__ __device__ inline int kstep_of(int f) { return is_fp4(f) ? 64 : 32; }

__host__ __device__ inline uint32_t hsh(uint32_t seed, int step, int side, int row, int k) {
  return dc_mix32(dc_mix32(dc_mix32(seed ^ ((uint32_t)step * 0x9E3779B1u)) + (uint32_t)side * 0x85EBCA77u + (uint32_t)row) *
                      0xC2B2AE3Du +
                  (uint32_t)k);
}
// Grid values in units of 1/s: E4M3 the integers -16..16 (s = 1, 2, 4: the integers, halves to 8, quarters to 4); E2M1 in
// half units (s = 2): grid 0 every E2M1 value, grid 1 its integer-valued subset.
__host__ __device__ inline int grid_val(int fmt, int grid, uint32_t r) {
  if (!is_fp4(fmt)) return (int)(r % 33u) - 16;
  const int x2[15] = {-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12};
  const int in[11] = {-12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12};
  return grid == 0 ? x2[r % 15u] : in[r % 11u];
}
__host__ __device__ inline int grid_max(int fmt) { return is_fp4(fmt) ? 12 : 16; }
__host__ __device__ inline int grid_min(int fmt, int grid) { return is_fp4(fmt) && grid == 1 ? 2 : 1; }

// The real operand value of a chain, win or accwin config (side 0: A row, side 1: B row, both K-major).
__host__ __device__ inline double real_val(const Cfg& c, int step, int side, int row, int k) {
  switch (c.dist) {
    case D_WIN:
      if (step) return 0;
      if (k == 0) return 256.0;
      if (k == 1) return ldexp(1.0, 8 - row % 18);
      return 0;
    case D_ACCWIN:
      if (step || k) return 0;
      if (side == 0) return 1.0;
      return is_fp4(c.fmt) ? ldexp(1.0, row % 3) : ldexp(1.0, row % 9);
    default: break;
  }
  int v = 0;
  const int vmax = grid_max(c.fmt);
  switch (c.dist) {
    case D_UNI: v = grid_val(c.fmt, c.grid, hsh(c.seed, step, side, row, k)); break;
    case D_MAX: v = vmax; break;
    case D_SIGN: v = (hsh(c.seed, step, side, row, k) & 1u) ? vmax : -vmax; break;
    case D_EDGE: v = k == 0 ? grid_min(c.fmt, c.grid) : vmax; break;
    case D_CANCEL:
      v = grid_val(c.fmt, c.grid, hsh(c.seed, step & ~1, side, row, k));
      if (side == 1 && (step & 1)) v = -v;
      break;
    default: break;
  }
  return (double)v / c.s;
}
// emu8's int8 draws (fp8acc's classes plus Dither-shaped uniform).
__host__ __device__ inline int emu_draw(const Cfg& c, int step, int side, int row, int k) {
  const uint32_t r = hsh(c.seed, step, side, row, k);
  switch (c.dist) {
    case D_MAX: return 127;
    case D_SIGN: return (r & 1u) ? 127 : -128;
    case D_EDGE: return -128;
    case D_LMIN: return (r & 1u) ? 120 : -120;
    default: return (int)(int8_t)(r & 0xFFu);
  }
}

__host__ __device__ inline uint8_t enc_e4m3(double a, int* bad) {
  if (a == 0) return 0;
  const uint8_t sg = a < 0 ? 0x80 : 0;
  const double m = fabs(a);
  int ex;
  const double fr = frexp(m, &ex);  // m = fr 2^ex, fr in [0.5, 1)
  const int e = ex - 1;
  if (e >= -6 && e <= 8) {
    const double man = (fr * 2 - 1) * 8;
    const int mi = (int)man;
    if (mi != man || mi > 7 || (e == 8 && mi == 7)) { *bad += 1; return 0; }
    return sg | (uint8_t)((e + 7) << 3) | (uint8_t)mi;
  }
  const double sub = m * 512.0;
  const int si = (int)sub;
  if (si != sub || si < 1 || si > 7) { *bad += 1; return 0; }
  return sg | (uint8_t)si;
}
__host__ __device__ inline uint8_t enc_e2m1(double a, int* bad) {
  const double t[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  const double m = fabs(a);
  for (int i = 0; i < 8; i++)
    if (t[i] == m) return (uint8_t)((a < 0 ? 8 : 0) | i);
  *bad += 1;
  return 0;
}
__host__ __device__ inline uint32_t f2b(float f) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(f);
#else
  uint32_t b;
  memcpy(&b, &f, 4);
  return b;
#endif
}
__host__ __device__ inline float b2f(uint32_t b) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(b);
#else
  float f;
  memcpy(&f, &b, 4);
  return f;
#endif
}
// Every partial an FP32 can hold: |P| without its trailing zeros fits 24 bits (the exponent range is never reached here).
__host__ __device__ inline bool rep24(long long P) {
  if (P == 0) return true;
  unsigned long long a = P < 0 ? (unsigned long long)(-P) : (unsigned long long)P;
  while (!(a & 1ull)) a >>= 1;
  return a < (1ull << 24);
}
// emu8's limb planes of an int8 x: k3 A [4h, m, -l], B [4h', m', l']; k4 [16h, l] both sides.
__host__ __device__ inline int plane_val(int fmt, int p, int x, int bside) {
  const int h = (x + 8) >> 4, l = x - 16 * h, m = h + l;
  if (fmt == F_K3B || fmt == F_K3Z) return p == 0 ? 4 * h : p == 1 ? m : (bside ? l : -l);
  if (fmt == F_K3H) return p == 0 ? h : p == 1 ? m : l;
  return p == 0 ? 16 * h : l;
}
__host__ __device__ inline int nplanes(int fmt) { return is_k3(fmt) ? 3 : is_emu(fmt) ? 2 : 1; }

// ---- shared device state -------------------------------------------------------------------------------------------------
struct Tally {
  int rep, bad_rep, inexact, rz, rn, negzero, bad_word, bad_acc;
  float maxerr;
};
__device__ inline void tally_add(Tally* t, const Tally& l) {
  atomicAdd(&t->rep, l.rep);
  atomicAdd(&t->bad_rep, l.bad_rep);
  atomicAdd(&t->inexact, l.inexact);
  atomicAdd(&t->rz, l.rz);
  atomicAdd(&t->rn, l.rn);
  atomicAdd(&t->negzero, l.negzero);
  atomicAdd(&t->bad_word, l.bad_word);
  atomicAdd(&t->bad_acc, l.bad_acc);
  atomicMax((int*)&t->maxerr, __float_as_int(l.maxerr));  // non-negative floats order as ints
}
__device__ inline void write_rec(Rec* r, long long K, int n, const Tally& t) {
  r->K = K;
  r->n = n;
  r->rep = t.rep;
  r->bad_rep = t.bad_rep;
  r->inexact = t.inexact;
  r->rz = t.rz;
  r->rn = t.rn;
  r->negzero = t.negzero;
  r->bad_word = t.bad_word;
  r->bad_acc = t.bad_acc;
  r->maxerr = t.maxerr;
}
// Operands of one step into the shared tiles: rows x KBY bytes a plane, K-major (dc.cuh's kmaj_off), and the integer
// values (chain: v in units 1/s; emu: the int8 x) for the reference.
__device__ void fill_tiles(const Cfg& c, int step, int rowsA, uint8_t* sA, uint8_t* sB, int8_t (*vA)[64],
                           int8_t (*vB)[64], int* encbad) {
  const int np = nplanes(c.fmt);
  for (int side = 0; side < 2; side++) {
    const int rows = side ? TN : rowsA, stride = rows * KBY;
    uint8_t* base = side ? sB : sA;
    for (int idx = threadIdx.x; idx < rows * KBY; idx += blockDim.x) {
      const int r = idx / KBY, b = idx % KBY;
      if (is_emu(c.fmt)) {
        const int x = emu_draw(c, step, side, r, b);
        (side ? vB : vA)[r][b] = (int8_t)x;
        for (int p = 0; p < np; p++)
          base[p * stride + kmaj_off(r, b, KBY)] = enc_e4m3((double)plane_val(c.fmt, p, x, side), encbad);
      } else if (is_fp4(c.fmt)) {
        const double r0 = real_val(c, step, side, r, 2 * b), r1 = real_val(c, step, side, r, 2 * b + 1);
        (side ? vB : vA)[r][2 * b] = (int8_t)lrint(r0 * c.s);
        (side ? vB : vA)[r][2 * b + 1] = (int8_t)lrint(r1 * c.s);
        base[kmaj_off(r, b, KBY)] = enc_e2m1(r0, encbad) | (uint8_t)(enc_e2m1(r1, encbad) << 4);
      } else {
        const double rv = real_val(c, step, side, r, b);
        (side ? vB : vA)[r][b] = (int8_t)lrint(rv * c.s);
        base[kmaj_off(r, b, KBY)] = enc_e4m3(rv, encbad);
      }
    }
  }
}
// The thread's view of one output after a step. chain: P, mrz, mrn, rep are the thread's running state for (row, col).
struct OutState {
  long long P;
  float mrz, mrn;
  int rep;
  int32_t word, dot;
};
__device__ inline void chain_step(const Cfg& c, OutState& o, int T) {
  const long long tu = (long long)T * c.num;
  o.P += tu;
  const float sv = ldexpf((float)tu, c.e);
  o.mrz = __fadd_rz(o.mrz, sv);
  o.mrn = __fadd_rn(o.mrn, sv);
  if (!rep24(o.P)) o.rep = 0;
}
__device__ inline void chain_check(const Cfg& c, const OutState& o, uint32_t got, Tally& t) {
  const double want = ldexp((double)o.P, c.e);
  const float gf = __uint_as_float(got);
  const bool ok = (double)gf == want && !(want == 0 && got != 0u);
  t.rep += o.rep;
  t.bad_rep += o.rep && !ok;
  t.inexact += !ok;
  t.rz += got == __float_as_uint(o.mrz);
  t.rn += got == __float_as_uint(o.mrn);
  t.negzero += got == 0x80000000u;
  const float err = (float)fabs((double)gf - want);
  if (err > t.maxerr) t.maxerr = err;
}
// emu: the accumulators' exact values and the word increment from what the tensor core wrote.
// k3h: three accumulators from +0 on unscaled limbs, Σhh', Σmm', Σll' (each |T| < 2^13), word += 240 Σhh' + 16 Σmm' -
// 15 Σll' by three FFMAs into the binade.
__device__ inline void emu_step(const Cfg& c, OutState& o, const int8_t* xa, const int8_t* xb, uint32_t g0, uint32_t g1,
                                uint32_t g2, Tally& t) {
  int T0 = 0, T1 = 0, T2 = 0, dot = 0;
  for (int k = 0; k < 32; k++) {
    const int x = xa[k], y = xb[k];
    dot += x * y;
    if (c.fmt == F_K3H) {
      T0 += plane_val(c.fmt, 0, x, 0) * plane_val(c.fmt, 0, y, 1);
      T1 += plane_val(c.fmt, 1, x, 0) * plane_val(c.fmt, 1, y, 1);
      T2 += plane_val(c.fmt, 2, x, 0) * plane_val(c.fmt, 2, y, 1);
    } else if (is_k3(c.fmt)) {
      T0 += plane_val(c.fmt, 0, x, 0) * plane_val(c.fmt, 0, y, 1) + plane_val(c.fmt, 2, x, 0) * plane_val(c.fmt, 2, y, 1);
      T1 += plane_val(c.fmt, 1, x, 0) * plane_val(c.fmt, 1, y, 1);
    } else {
      T0 += x * y;
    }
  }
  const bool k3 = is_k3(c.fmt), binade = c.fmt == F_K3B || c.fmt == F_K4B;
  const float c0 = binade ? __uint_as_float(k3 ? C1B : C0B) : 0.f, c1 = binade ? __uint_as_float(C2B) : 0.f;
  const float w0 = c0 + (float)T0, w1 = c1 + (float)T1;  // exact: |T| < 2^19 inside each binade
  t.bad_acc += __float_as_uint(w0) != g0;
  if (k3) t.bad_acc += __float_as_uint(w1) != g1;
  if (c.fmt == F_K3H) t.bad_acc += __float_as_uint((float)T2) != g2;
  uint32_t inc;
  if (c.fmt == F_K3H) {
    const float r = __fmaf_rn(-15.f, __uint_as_float(g2),
                              __fmaf_rn(16.f, __uint_as_float(g1), __fmaf_rn(240.f, __uint_as_float(g0), 12582912.f)));
    inc = __float_as_uint(r) - C0B;
  } else if (binade) {
    inc = k3 ? 15u * g0 + 2u * g1 : g0 - C0B;
  } else {
    const float r = k3 ? __fmaf_rn(16.f, __uint_as_float(g1), __fmaf_rn(15.f, __uint_as_float(g0), 12582912.f))
                       : __fadd_rn(__uint_as_float(g0), 12582912.f);
    inc = __float_as_uint(r) - C0B;
  }
  o.word += (int32_t)inc;
  o.dot += dot;
  t.bad_word += o.word != o.dot;
}
// e4r: one MMA a step from +0 on a grid (units 2^e, num 1); the accumulator must be the step's exact sum, and the word in
// units grows by one FFMA into the binade and one IADD: the honest route where a narrow adder forbids a binade start.
__device__ inline void e4r_step(const Cfg& c, OutState& o, int T, uint32_t g, Tally& t) {
  t.bad_acc += __float_as_uint(ldexpf((float)T, c.e)) != g;
  const float r = __fmaf_rn(__uint_as_float(g), ldexpf(1.f, -c.e), 12582912.f);
  o.word += (int32_t)(__float_as_uint(r) - C0B);
  o.dot += T;
  t.bad_word += o.word != o.dot;
}

// ---- sm_100a: tcgen05 ----------------------------------------------------------------------------------------------------
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
DC_DEV void umma_mxf4(uint32_t d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;\n}\n" ::"r"(d),
      "l"(da), "l"(db), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc)
      : "memory");
}
DC_DEV void umma_nvf4(uint32_t d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n}\n" ::"r"(d),
      "l"(da), "l"(db), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc)
      : "memory");
}
DC_DEV void mbar_arrive(uint64_t* bar) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(smem_u32(bar)) : "memory");
}
DC_DEV void tmem_fill(uint32_t taddr, uint32_t v, int cols) {
  uint32_t r[16];
#pragma unroll
  for (int i = 0; i < 16; i++) r[i] = v;
  for (int u = 0; u < cols / 16; u++) tmem_st16(taddr + 16 * u, r);
}
#endif
// Block-scaled instruction descriptor (kind::mxf4, mxf4nvf4): A and B E2M1 (1) at [7,10) and [10,13), N >> 3 at [17,23),
// scale type at 23 (UE4M3 0, UE8M0 1), M >> 4 at [24,29), both scale-factor ids 0. D is FP32 by the kind.
DC_HD constexpr uint32_t idesc_bs(uint32_t m, uint32_t n, uint32_t ue8m0) {
  return (1u << 7) | (1u << 10) | ((n >> 3) << 17) | (ue8m0 << 23) | ((m >> 4) << 24);
}

__global__ void __launch_bounds__(128) fpx100(const Cfg* cfgs, Rec* recs, uint32_t* dump, int* encbad_out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  constexpr int BM = 128;
  const Cfg c = cfgs[blockIdx.x];
  __shared__ __align__(1024) uint8_t sA[3][BM * KBY];
  __shared__ __align__(1024) uint8_t sB[3][TN * KBY];
  __shared__ int8_t vA[BM][64];
  __shared__ int8_t vB[TN][64];
  __shared__ uint64_t bar;
  __shared__ uint32_t tslot;
  __shared__ Tally tally;
  __shared__ int encbad;
  const uint32_t w = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
  const int row = threadIdx.x, ks = kstep_of(c.fmt), np = nplanes(c.fmt);
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
  // cols 0-31 acc0, 32-63 acc1, 64-95 A's scales, 96-127 B's scales (uniform, so their TMEM layout does not matter)
  tmem_fill(trow + 64, 0x01010101u * (uint32_t)c.sa, 32);
  tmem_fill(trow + 96, 0x01010101u * (uint32_t)c.sb, 32);
  const uint32_t c0 = c.dist == D_ACCWIN ? __float_as_uint(ldexpf(1.f, row % 32)) : c.c0bits;
  if (!is_word(c.fmt)) tmem_fill(trow, c0, 32);
  tmem_wait_st();
  OutState o[TN];
  for (int j = 0; j < TN; j++) o[j] = OutState{c.c0u, __uint_as_float(c0), __uint_as_float(c0), 1, 0, 0};
  Tally loc{};
  int ck = 0, myenc = 0;
  const uint32_t idesc = c.fmt == F_MXF4 ? idesc_bs(BM, TN, 1) : c.fmt == F_NVF4 ? idesc_bs(BM, TN, 0)
                                                                                  : umma_idesc(1, 0, 0, BM, TN);
  for (int s = 0; s < c.nsteps; s++) {
    fill_tiles(c, s, BM, &sA[0][0], &sB[0][0], vA, vB, &myenc);
    if (c.fmt == F_K3B || c.fmt == F_K4B) {
      tmem_fill(trow, c.fmt == F_K3B ? C1B : C0B, 32);
      tmem_fill(trow + 32, C2B, 32);
      tmem_wait_st();
    }
    fence_async_smem();
    tc_before_sync();
    __syncthreads();
    tc_after_sync();
    if (w == 0) {
      if (elect_one()) {
        if (c.control) {
          mbar_arrive(&bar);
        } else {
          auto da = [&](int p) { return umma_desc(smem_u32(sA[p]), kmaj_lbo(), kmaj_sbo(KBY)); };
          auto db = [&](int p) { return umma_desc(smem_u32(sB[p]), kmaj_lbo(), kmaj_sbo(KBY)); };
          const uint32_t z = (c.fmt == F_K3Z || c.fmt == F_K4Z) ? 0u : 1u;
          switch (c.fmt) {
            case F_E4M3: umma_f8f6f4(tmem, da(0), db(0), idesc, 1); break;
            case F_MXF4: umma_mxf4(tmem, da(0), db(0), idesc, tmem + 64, tmem + 96, 1); break;
            case F_NVF4: umma_nvf4(tmem, da(0), db(0), idesc, tmem + 64, tmem + 96, 1); break;
            case F_E4R: umma_f8f6f4(tmem, da(0), db(0), idesc, 0); break;
            case F_K3H:  // acc2 in the A-scale columns: no scales on this route
              umma_f8f6f4(tmem, da(0), db(0), idesc, 0);
              umma_f8f6f4(tmem + 32, da(1), db(1), idesc, 0);
              umma_f8f6f4(tmem + 64, da(2), db(2), idesc, 0);
              break;
            case F_K3B: case F_K3Z:
              umma_f8f6f4(tmem, da(0), db(0), idesc, z);
              umma_f8f6f4(tmem, da(2), db(2), idesc, 1);
              umma_f8f6f4(tmem + 32, da(1), db(1), idesc, z);
              break;
            default:
              umma_f8f6f4(tmem, da(0), db(0), idesc, z);
              umma_f8f6f4(tmem, da(0), db(1), idesc, 1);
              umma_f8f6f4(tmem, da(1), db(0), idesc, 1);
              umma_f8f6f4(tmem, da(1), db(1), idesc, 1);
              break;
          }
          umma_commit(&bar);
        }
      }
      __syncwarp();
    }
    mbar_wait(&bar, s & 1);
    tc_after_sync();
    const bool at_ck = ((s + 1) & s) == 0 || s == c.nsteps - 1;
    if (is_emu(c.fmt)) {
      uint32_t g0[16], g1[16], g2[16];
      for (int h = 0; h < 2; h++) {
        tmem_ld16(trow + 16 * h, g0);
        tmem_ld16(trow + 32 + 16 * h, g1);
        tmem_ld16(trow + 64 + 16 * h, g2);
        tmem_wait_ld();
        for (int q = 0; q < 16; q++) emu_step(c, o[16 * h + q], vA[row], vB[16 * h + q], g0[q], g1[q], g2[q], loc);
      }
    } else if (c.fmt == F_E4R) {
      uint32_t g[16];
      for (int h = 0; h < 2; h++) {
        tmem_ld16(trow + 16 * h, g);
        tmem_wait_ld();
        for (int q = 0; q < 16; q++) {
          int T = 0;
          for (int k = 0; k < 32; k++) T += (int)vA[row][k] * (int)vB[16 * h + q][k];
          e4r_step(c, o[16 * h + q], T, g[q], loc);
        }
      }
    } else {
      for (int j = 0; j < TN; j++) {
        int T = 0;
        for (int k = 0; k < ks; k++) T += (int)vA[row][k] * (int)vB[j][k];
        chain_step(c, o[j], T);
      }
    }
    if (at_ck) {
      Tally t = is_word(c.fmt) ? loc : Tally{};
      if (!is_word(c.fmt)) {
        for (int h = 0; h < 2; h++) {
          uint32_t g[16];
          tmem_ld16(trow + 16 * h, g);
          tmem_wait_ld();
          for (int q = 0; q < 16; q++) {
            chain_check(c, o[16 * h + q], g[q], t);
            if (c.dump && s == c.nsteps - 1) dump[((size_t)blockIdx.x * BM + row) * TN + 16 * h + q] = g[q];
          }
        }
      }
      if (threadIdx.x == 0) tally = Tally{};
      __syncthreads();
      tally_add(&tally, t);
      __syncthreads();
      if (threadIdx.x == 0 && ck < MAXCK)
        write_rec(&recs[blockIdx.x * MAXCK + ck], (long long)(s + 1) * ks, BM * TN, tally);
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
  (void)lane;
#endif
}

// ---- sm_90a: wgmma -------------------------------------------------------------------------------------------------------
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
DC_DEV void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DC_DEV void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
DC_DEV void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
// m64n32k32 E4M3 x E4M3 -> FP32, 16 accumulator registers a thread; scale_d 0 overwrites D.
DC_DEV void wg_e4m3_n32(float (&d)[16], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15}, %16, %17, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "r"(scale_d)
      : "memory");
}
template <int N> DC_DEV void hold(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; i++) asm volatile("" : "+f"(d[i])::"memory");
}
#endif

__global__ void __launch_bounds__(128) fpx90(const Cfg* cfgs, Rec* recs, uint32_t* dump, int* encbad_out) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int BM = 64;
  const Cfg c = cfgs[blockIdx.x];
  __shared__ __align__(1024) uint8_t sA[3][BM * KBY];
  __shared__ __align__(1024) uint8_t sB[3][TN * KBY];
  __shared__ int8_t vA[BM][64];
  __shared__ int8_t vB[TN][64];
  __shared__ Tally tally;
  __shared__ int encbad;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) encbad = 0;
  __syncthreads();
  int rowq[16], colq[16];
  for (int q = 0; q < 16; q++) {
    rowq[q] = 16 * warp + (lane >> 2) + 8 * ((q >> 1) & 1);
    colq[q] = 8 * (q >> 2) + 2 * (lane & 3) + (q & 1);
  }
  float d0[16], d1[16], d2[16];
  OutState o[16];
  for (int q = 0; q < 16; q++) {
    const uint32_t c0 = c.dist == D_ACCWIN ? __float_as_uint(ldexpf(1.f, rowq[q] % 32)) : c.c0bits;
    d0[q] = __uint_as_float(c0);
    d1[q] = 0.f;
    d2[q] = 0.f;
    o[q] = OutState{c.c0u, d0[q], d0[q], 1, 0, 0};
  }
  Tally loc{};
  int ck = 0, myenc = 0;
  for (int s = 0; s < c.nsteps; s++) {
    fill_tiles(c, s, BM, &sA[0][0], &sB[0][0], vA, vB, &myenc);
    if (c.fmt == F_K3B || c.fmt == F_K4B) {
      for (int q = 0; q < 16; q++) {
        d0[q] = __uint_as_float(c.fmt == F_K3B ? C1B : C0B);
        d1[q] = __uint_as_float(C2B);
      }
    }
    fence_async_smem();
    __syncthreads();
    if (!c.control) {
      auto da = [&](int p) { return wgmma_desc(smem_u32(sA[p]), kmaj_lbo(), kmaj_sbo(KBY)); };
      auto db = [&](int p) { return wgmma_desc(smem_u32(sB[p]), kmaj_lbo(), kmaj_sbo(KBY)); };
      const int z = (c.fmt == F_K3Z || c.fmt == F_K4Z) ? 0 : 1;
      wg_fence();
      switch (c.fmt) {
        case F_E4M3: wg_e4m3_n32(d0, da(0), db(0), 1); break;
        case F_E4R: wg_e4m3_n32(d0, da(0), db(0), 0); break;
        case F_K3H:
          wg_e4m3_n32(d0, da(0), db(0), 0);
          wg_e4m3_n32(d1, da(1), db(1), 0);
          wg_e4m3_n32(d2, da(2), db(2), 0);
          break;
        case F_K3B: case F_K3Z:
          wg_e4m3_n32(d0, da(0), db(0), z);
          wg_e4m3_n32(d0, da(2), db(2), 1);
          wg_e4m3_n32(d1, da(1), db(1), z);
          break;
        default:
          wg_e4m3_n32(d0, da(0), db(0), z);
          wg_e4m3_n32(d0, da(0), db(1), 1);
          wg_e4m3_n32(d0, da(1), db(0), 1);
          wg_e4m3_n32(d0, da(1), db(1), 1);
          break;
      }
      wg_commit();
      wg_wait0();
      hold(d0);
      hold(d1);
      hold(d2);
    }
    const bool at_ck = ((s + 1) & s) == 0 || s == c.nsteps - 1;
    if (is_emu(c.fmt)) {
      for (int q = 0; q < 16; q++)
        emu_step(c, o[q], vA[rowq[q]], vB[colq[q]], __float_as_uint(d0[q]), __float_as_uint(d1[q]),
                 __float_as_uint(d2[q]), loc);
    } else if (c.fmt == F_E4R) {
      for (int q = 0; q < 16; q++) {
        int T = 0;
        for (int k = 0; k < 32; k++) T += (int)vA[rowq[q]][k] * (int)vB[colq[q]][k];
        e4r_step(c, o[q], T, __float_as_uint(d0[q]), loc);
      }
    } else {
      for (int q = 0; q < 16; q++) {
        int T = 0;
        for (int k = 0; k < 32; k++) T += (int)vA[rowq[q]][k] * (int)vB[colq[q]][k];
        chain_step(c, o[q], T);
      }
    }
    if (at_ck) {
      Tally t = is_word(c.fmt) ? loc : Tally{};
      if (!is_word(c.fmt))
        for (int q = 0; q < 16; q++) {
          chain_check(c, o[q], __float_as_uint(d0[q]), t);
          if (c.dump && s == c.nsteps - 1)
            dump[((size_t)blockIdx.x * 128 + rowq[q]) * TN + colq[q]] = __float_as_uint(d0[q]);
        }
      if (threadIdx.x == 0) tally = Tally{};
      __syncthreads();
      tally_add(&tally, t);
      __syncthreads();
      if (threadIdx.x == 0 && ck < MAXCK) write_rec(&recs[blockIdx.x * MAXCK + ck], (long long)(s + 1) * 32, BM * TN, tally);
      ck++;
    }
    __syncthreads();
  }
  atomicAdd(&encbad, myenc);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(encbad_out, encbad);
#endif
}

// ---- host ----------------------------------------------------------------------------------------------------------------
static double ue4m3(int b) {
  const int e = (b >> 3) & 15, m = b & 7;
  return e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
}
static double scale_val(int fmt, int b) { return fmt == F_MXF4 ? std::ldexp(1.0, b - 127) : ue4m3(b); }
static void odd_split(double v, int* num, int* e) {  // v = num 2^e, num odd
  int ee = 0;
  while (v != std::floor(v)) { v *= 2; ee--; }
  long long n = (long long)v;
  while (n && !(n & 1)) { n >>= 1; ee++; }
  *num = (int)n;
  *e = ee;
}
static Cfg make(int fmt, int grid, int dist, int c0mode, int nsteps, uint32_t seed, int sa = 0, int sb = 0) {
  Cfg c{};
  c.fmt = fmt; c.grid = grid; c.dist = dist; c.c0mode = c0mode; c.nsteps = nsteps; c.seed = seed;
  c.sa = sa; c.sb = sb;
  if (is_emu(fmt)) { c.s = 1; c.num = 1; c.e = 0; return c; }
  if (fmt == F_E4R) c0mode = 0;
  double unit;
  if (is_fp4(fmt)) { c.s = 2; unit = scale_val(fmt, sa) * scale_val(fmt, sb) / 4.0; }
  else { c.s = grid == 0 ? 1 : grid == 1 ? 2 : 4; unit = 1.0 / (c.s * c.s); }
  if (dist == D_WIN || dist == D_ACCWIN) unit = std::ldexp(1.0, -18);
  odd_split(unit, &c.num, &c.e);
  auto bits = [](double v) { float f = (float)v; uint32_t b; memcpy(&b, &f, 4); return b; };
  switch (c0mode) {
    case 1: c.c0bits = bits(std::ldexp(1.5, 23 + c.e)); c.c0u = 12582912; break;
    case 2: c.c0bits = C0B; c.c0u = c.e <= 0 ? (12582912ll << -c.e) : 0; break;
    case 3: c.c0bits = bits(std::ldexp(1.5, 24 + c.e)); c.c0u = 25165824; break;
    default: c.c0bits = 0; c.c0u = 0; break;
  }
  return c;
}
static std::vector<Cfg> configs(int family) {
  std::vector<Cfg> v;
  const int chain_d[] = {D_UNI, D_UNI, D_UNI, D_MAX, D_SIGN, D_EDGE, D_CANCEL};
  uint32_t seed = 1;
  for (int g = 0; g < 3; g++)
    for (int c0 = 0; c0 < 4; c0++)
      for (int d : chain_d) v.push_back(make(F_E4M3, g, d, c0, c0 == 0 ? 4096 : 2048, seed++));
  if (family == 10) {
    const int mx[][2] = {{127, 127}, {124, 124}, {130, 130}, {124, 130}};
    const int nv[][2] = {{0x38, 0x38}, {0x44, 0x44}, {0x57, 0x57}, {0x38, 0x57}};
    for (int f = F_MXF4; f <= F_NVF4; f++)
      for (int g = 0; g < 2; g++)
        for (int sc = 0; sc < 4; sc++)
          for (int c0 = 0; c0 < 4; c0++) {
            const int sa = f == F_MXF4 ? mx[sc][0] : nv[sc][0], sb = f == F_MXF4 ? mx[sc][1] : nv[sc][1];
            Cfg probe = make(f, g, D_UNI, c0, 1, 0, sa, sb);
            if (c0 == 2 && probe.e > 0) continue;  // NOBITS' 0x4B400000 can't hold a product unit above 1
            for (int d : chain_d) v.push_back(make(f, g, d, c0, c0 == 0 ? 4096 : 2048, seed++, sa, sb));
          }
  }
  const int emu_d[] = {D_UNI, D_UNI, D_UNI, D_MAX, D_SIGN, D_EDGE, D_LMIN};
  for (int f = F_K3B; f <= F_K3H; f++)
    for (int d : emu_d) v.push_back(make(f, 0, d, 0, 512, seed++));
  for (int g = 0; g < 3; g++)
    for (int d : chain_d) v.push_back(make(F_E4R, g, d, 0, 512, seed++));
  Cfg w = make(F_E4M3, 0, D_WIN, 0, 1, seed++); w.dump = 1; v.push_back(w);
  Cfg a = make(F_E4M3, 0, D_ACCWIN, 0, 1, seed++); a.dump = 1; v.push_back(a);
  if (family == 10) {
    Cfg m = make(F_MXF4, 0, D_ACCWIN, 0, 1, seed++, 127, 127); m.dump = 1; v.push_back(m);
    Cfg n = make(F_NVF4, 0, D_ACCWIN, 0, 1, seed++, 0x38, 0x38); n.dump = 1; v.push_back(n);
  }
  // controls: no MMA issued, so the accumulator keeps C0 and every check must see it
  Cfg k1 = make(F_E4M3, 0, D_UNI, 1, 64, seed++); k1.control = 1; v.push_back(k1);
  Cfg k2 = make(F_K3B, 0, D_UNI, 0, 64, seed++); k2.control = 1; v.push_back(k2);
  Cfg k3 = make(F_K3Z, 0, D_UNI, 0, 64, seed++); k3.control = 1; v.push_back(k3);
  Cfg k6 = make(F_K3H, 0, D_UNI, 0, 64, seed++); k6.control = 1; v.push_back(k6);
  Cfg k7 = make(F_E4R, 0, D_UNI, 0, 64, seed++); k7.control = 1; v.push_back(k7);
  if (family == 10) {
    Cfg k4 = make(F_MXF4, 0, D_UNI, 1, 64, seed++, 127, 127); k4.control = 1; v.push_back(k4);
    Cfg k5 = make(F_NVF4, 0, D_UNI, 1, 64, seed++, 0x38, 0x38); k5.control = 1; v.push_back(k5);
  }
  return v;
}
static float rz_f32(double x) {
  float f = (float)x;
  if (std::fabs((double)f) > std::fabs(x)) f = std::nextafter(f, 0.0f);
  return f;
}
static void cfg_json(FILE* f, int i, const Cfg& c) {
  fprintf(f,
          "\"cfg\": %d, \"fmt\": \"%s\", \"grid\": %d, \"s\": %d, \"dist\": \"%s\", \"c0\": \"%s\", \"c0bits\": \"0x%08X\", "
          "\"sa\": %d, \"sb\": %d, \"unit_num\": %d, \"unit_e\": %d, \"nsteps\": %d, \"control\": %d",
          i, FMTS[c.fmt], c.grid, c.s, DISTS[c.dist], C0S[c.c0mode], c.c0bits, c.sa, c.sb, c.num, c.e, c.nsteps, c.control);
}

int main(int argc, char** argv) {
  const std::string mode = argc >= 2 ? argv[1] : "";
  if (mode == "list") {
    for (int fam : {9, 10}) {
      auto v = configs(fam);
      printf("{\"family\": %d, \"configs\": %zu}\n", fam, v.size());
    }
    return 0;
  }
  if (mode != "run" || argc != 3) { fprintf(stderr, "usage: fpx run OUT | fpx list\n"); return 2; }
  const std::string out = argv[2];
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { fprintf(stderr, "ERR no device\n"); return 2; }
  const int fam = p.major;
  if (fam != 9 && fam != 10) { fprintf(stderr, "ERR sm_%d%d: fpx is for sm_90a, sm_100a and sm_103a\n", p.major, p.minor); return 2; }
  auto cfgs = configs(fam);
  const int n = (int)cfgs.size();
  Cfg* dC; Rec* dR; uint32_t* dD; int* dE;
  CK(cudaMalloc(&dC, n * sizeof(Cfg)));
  CK(cudaMalloc(&dR, (size_t)n * MAXCK * sizeof(Rec)));
  CK(cudaMalloc(&dD, (size_t)n * 128 * TN * 4));
  CK(cudaMalloc(&dE, 4));
  CK(cudaMemcpy(dC, cfgs.data(), n * sizeof(Cfg), cudaMemcpyHostToDevice));
  CK(cudaMemset(dR, 0xA5, (size_t)n * MAXCK * sizeof(Rec)));
  CK(cudaMemset(dD, 0xA5, (size_t)n * 128 * TN * 4));
  CK(cudaMemset(dE, 0, 4));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  if (fam == 10) fpx100<<<n, 128>>>(dC, dR, dD, dE);
  else fpx90<<<n, 128>>>(dC, dR, dD, dE);
  CK(cudaGetLastError());
  CK(cudaEventRecord(e1));
  CK(cudaDeviceSynchronize());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  std::vector<Rec> R((size_t)n * MAXCK);
  std::vector<uint32_t> D((size_t)n * 128 * TN);
  int encbad = 0;
  CK(cudaMemcpy(R.data(), dR, R.size() * sizeof(Rec), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&encbad, dE, 4, cudaMemcpyDeviceToHost));

  FILE* fr = fopen((out + "/records.jsonl").c_str(), "w");
  FILE* fw = fopen((out + "/windows.jsonl").c_str(), "w");
  if (!fr || !fw) { fprintf(stderr, "ERR open %s\n", out.c_str()); return 2; }
  long bad_rep = 0, bad_word = 0, bad_acc = 0, missed_controls = 0, configs_run = 0;
  for (int i = 0; i < n; i++) {
    const Cfg& c = cfgs[i];
    if (c.dist == D_WIN || c.dist == D_ACCWIN) continue;
    const int nck = [&] { int k = 0; for (int s = 0; s < c.nsteps; s++) k += ((s + 1) & s) == 0 || s == c.nsteps - 1; return k; }();
    long cfg_bad = 0;
    for (int k = 0; k < nck && k < MAXCK; k++) {
      const Rec& r = R[(size_t)i * MAXCK + k];
      fprintf(fr, "{");
      cfg_json(fr, i, c);
      fprintf(fr,
              ", \"K\": %lld, \"n\": %d, \"rep\": %d, \"bad_rep\": %d, \"inexact\": %d, \"rz\": %d, \"rn\": %d, "
              "\"negzero\": %d, \"bad_word\": %d, \"bad_acc\": %d, \"maxerr\": %.9g}\n",
              r.K, r.n, r.rep, r.bad_rep, r.inexact, r.rz, r.rn, r.negzero, r.bad_word, r.bad_acc, r.maxerr);
      cfg_bad += r.bad_rep + r.bad_word + r.bad_acc + (is_word(c.fmt) ? 0 : r.inexact);
      if (!c.control) {
        if (k == nck - 1 || k == MAXCK - 1) { bad_word += r.bad_word; bad_acc += r.bad_acc; }
        bad_rep += r.bad_rep;
      }
    }
    if (c.control && cfg_bad == 0) missed_controls++;
    configs_run++;
  }
  // windows: classify every output of the win and accwin configs against the exact sum
  for (int i = 0; i < n; i++) {
    const Cfg& c = cfgs[i];
    if (c.dist != D_WIN && c.dist != D_ACCWIN) continue;
    const int rows = fam == 10 ? 128 : 64;
    const int G = 64;
    long cnt[G][6] = {};
    for (int r = 0; r < rows; r++)
      for (int j = 0; j < TN; j++) {
        const uint32_t gb = D[((size_t)i * 128 + r) * TN + j];
        float gf; memcpy(&gf, &gb, 4);
        double big, small; int g;
        if (c.dist == D_WIN) {
          g = r % 18 + j % 18; big = 65536.0; small = std::ldexp(1.0, 16 - g);
        } else {
          const int cc = r % 32, pp = is_fp4(c.fmt) ? j % 3 : j % 9;
          g = cc - pp + 16; big = std::ldexp(1.0, cc); small = std::ldexp(1.0, pp);
        }
        if (g < 0 || g >= G) continue;
        const double ex = big + small;
        const int cls = (double)gf == ex ? 0 : gf == rz_f32(ex) ? 1 : gf == (float)ex ? 2 : (double)gf == big ? 3 : 4;
        cnt[g][cls]++;
        cnt[g][5]++;
      }
    for (int g = 0; g < G; g++) {
      if (!cnt[g][5]) continue;
      fprintf(fw, "{");
      cfg_json(fw, i, c);
      fprintf(fw, ", \"test\": \"%s\", \"gap\": %d, \"n\": %ld, \"exact\": %ld, \"rz\": %ld, \"rn\": %ld, \"dropped\": %ld, "
                  "\"other\": %ld}\n",
              DISTS[c.dist], c.dist == D_WIN ? g : g - 16, cnt[g][5], cnt[g][0], cnt[g][1], cnt[g][2], cnt[g][3], cnt[g][4]);
    }
  }
  fclose(fr);
  fclose(fw);
  const bool ok = bad_rep == 0 && bad_word == 0 && bad_acc == 0 && missed_controls == 0 && encbad == 0;
  FILE* fs = fopen((out + "/summary.json").c_str(), "w");
  fprintf(fs,
          "{\"device\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"configs\": %ld, \"kernel_ms\": %.3f, \"encoding_bad\": %d, "
          "\"bad_rep\": %ld, \"emu_bad_word\": %ld, \"emu_bad_acc\": %ld, \"missed_controls\": %ld, \"as_stated\": %s}\n",
          p.name, p.major, p.minor, p.multiProcessorCount, configs_run, ms, encbad, bad_rep, bad_word, bad_acc,
          missed_controls, ok ? "true" : "false");
  fclose(fs);
  printf("{\"device\": \"%s\", \"configs\": %ld, \"kernel_ms\": %.1f, \"encoding_bad\": %d, \"bad_rep\": %ld, "
         "\"emu_bad_word\": %ld, \"emu_bad_acc\": %ld, \"missed_controls\": %ld}\n",
         p.name, configs_run, ms, encbad, bad_rep, bad_word, bad_acc, missed_controls);
  return ok ? 0 : 4;
}
