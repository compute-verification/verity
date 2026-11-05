// The numeric pin of the SASS gate (sass_gate.py; sass_pins.json "numeric"). ptxas builds div.rn, sqrt.rn and rcp.rn from
// .FTZ instructions and MUFU seeds, and integer division from a MUFU.RCP seed, in every build; the gate exempts them only as a
// pinned ptxas builds them, and this is the evidence that each is exact there, subnormals included; its builds are where
// sass_pins.json's pins come from. Each check runs a sequence as this file compiles it (ieee_pin.py gates the binary and
// requires every sequence and every body its toolkit pins) and compares it on the card with a reference that doesn't use it:
//   rcp    __frcp_rn(x) on all 2^32 x, against 1 / x in double rounded once (innocuous double rounding: 53 >= 2*24 + 2)
//   sqrt   sqrtf(x) on all 2^32 x, against sqrt in double rounded once
//   div    a / b on 2^32 counter-hashed pairs weighted to subnormal operands and results, short mantissas (exact quotients and
//          ties) and the overflow edge, against a / b in double rounded once
//   cdiv   3.0f / b on all 2^32 b (cdiv3) beside a / b on weighted a (cdiv), in one kernel, so the two share their slow path
//          and the constant is an immediate of the fast path, as ptxas builds it outside a function with one division
//   u16    div.u16 on all 2^32 pairs in a loop, and again (u16_call) one pair per thread from memory, the two call sites for
//          which 13.0 builds __cuda_sm20_div_u16 differently; u32 a / b, s32 a / b and a % b, u32n a / n for a uniform n (64
//          divisors), u64 a / b and s64 a % b on weighted pairs: exact when q * b + r == a with |r| < |b| and r of a's sign,
//          in integer arithmetic
//   host_* the device's rcp, sqrt, div and 3 / b on every subnormal input and every input whose reciprocal is subnormal
//          (2^24 - 2 + 2^25), each with a weighted divisor, against the host's double arithmetic: a reference off the card
//   units_* on an sm_90 card, the slow-path bodies ptxas builds for sm_90a in kernels' units.cu, from copies of its kernels
//          (units_checks): sqrt on all 2^32 x; rmsnorm's division on 2^32 weighted pairs and its one / mean; softmax's,
//          whose numerator its body loads, at 1.0 on all 2^32 divisors and at 1024 weighted numerators on weighted divisors
// Division by zero and INT_MIN / -1 are outside the domain, counted as skipped. NaN results compare equal whatever their
// payload; everything else bit for bit. Prints one JSON line for the device, then one per check.
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a --no-compress [FLAGS] ieee_pin.cu -o ieee_pin
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define EACH(i, n) for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (uint64_t)gridDim.x * blockDim.x)

struct Tally {
  unsigned long long cases, skipped, sub_in, sub_out, bad, first[4];
  int got;
};

__host__ __device__ inline uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
__host__ __device__ inline float flt(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
__host__ __device__ inline bool same(float h, float r) { return bits(h) == bits(r) || (h != h && r != r); }
__host__ __device__ inline bool subn(float x) { uint32_t u = bits(x); return (u & 0x7F800000u) == 0 && (u & 0x7FFFFFu); }

__host__ __device__ inline uint32_t mix(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
  return (uint32_t)x;
}

// An FP32 operand: 1/8 subnormal, 1/8 just above, 1/8 near overflow, 1/8 around 1, the rest any exponent; 1/8 of each with a
// short mantissa (exact quotients and ties).
__host__ __device__ inline uint32_t pick(uint64_t i, uint32_t salt) {
  uint32_t r = mix(i * 2 + salt), cls = r & 7, man = mix(i * 7 + salt + 99) & 0x7FFFFF, e;
  if (cls == 0) e = 0;
  else if (cls == 1) e = 1 + (r >> 8) % 8;
  else if (cls == 2) e = 247 + (r >> 8) % 8;
  else if (cls == 3) e = 120 + (r >> 8) % 16;
  else e = 1 + (r >> 8) % 254;
  if (((r >> 16) & 7) == 0) man &= 0x7F0000;
  return (r & 0x80000000u) | (e << 23) | man;
}

// Integer operands: small, 16-bit, powers of two and their neighbours, the top of the range, a shifted value, or any.
__host__ __device__ inline uint32_t pick32(uint64_t i, uint32_t salt) {
  uint32_t r = mix(i * 2 + salt), v = mix(i * 7 + salt + 99), k = (r >> 8) & 31;
  switch (r & 7) {
    case 0: return v & 0xff;
    case 1: return v & 0xffff;
    case 2: return 1u << k;
    case 3: return (1u << k) - 1u;
    case 4: return (1u << k) + 1u;
    case 5: return ~(v & 0xff);
    case 6: return v >> k;
    default: return v;
  }
}

// As pick32 at 64 bits; 1/8 fit in 32 bits, the path ptxas inlines instead of calling __cuda_sm20_div_u64.
__host__ __device__ inline uint64_t pick64(uint64_t i, uint32_t salt) {
  uint32_t r = mix(i * 2 + salt), k = (r >> 8) & 63;
  uint64_t v = (uint64_t)mix(i * 7 + salt + 99) << 32 | mix(i * 11 + salt + 7);
  switch (r & 7) {
    case 0: return v & 0xffffffffull;
    case 1: return v & 0xffff;
    case 2: return 1ull << k;
    case 3: return (1ull << k) - 1;
    case 4: return (1ull << k) + 1;
    case 5: return ~(v & 0xff);
    case 6: return v >> k;
    default: return v;
  }
}

__device__ void add(Tally* t, unsigned long long c, unsigned long long s, unsigned long long si, unsigned long long so,
                    unsigned long long b) {
  atomicAdd(&t->cases, c); atomicAdd(&t->skipped, s); atomicAdd(&t->sub_in, si); atomicAdd(&t->sub_out, so);
  atomicAdd(&t->bad, b);
}

__device__ void first(Tally* t, uint64_t a, uint64_t b, uint64_t h, uint64_t r) {
  if (atomicCAS(&t->got, 0, 1) == 0) { t->first[0] = a; t->first[1] = b; t->first[2] = h; t->first[3] = r; }
}

__global__ void rcp_all(Tally* t) {
  unsigned long long c = 0, si = 0, so = 0, nb = 0;
  EACH(i, 1ull << 32) {
    float x = flt((uint32_t)i), h = __frcp_rn(x), r = __double2float_rn(1.0 / (double)x);
    c++; si += subn(x); so += subn(r);
    if (!same(h, r)) { nb++; first(t, bits(x), 0, bits(h), bits(r)); }
  }
  add(t, c, 0, si, so, nb);
}

__global__ void sqrt_all(Tally* t) {
  unsigned long long c = 0, si = 0, so = 0, nb = 0;
  EACH(i, 1ull << 32) {
    float x = flt((uint32_t)i), h = sqrtf(x), r = __double2float_rn(sqrt((double)x));
    c++; si += subn(x); so += subn(r);
    if (!same(h, r)) { nb++; first(t, bits(x), 0, bits(h), bits(r)); }
  }
  add(t, c, 0, si, so, nb);
}

__global__ void div_all(Tally* t) {
  unsigned long long c = 0, si = 0, so = 0, nb = 0;
  EACH(i, 1ull << 32) {
    float a = flt(pick(i, 1)), b = flt(pick(i, 2)), h = a / b, r = __double2float_rn((double)a / (double)b);
    c++; si += subn(a) || subn(b); so += subn(r);
    if (!same(h, r)) { nb++; first(t, bits(a), bits(b), bits(h), bits(r)); }
  }
  add(t, c, 0, si, so, nb);
}

__global__ void cdiv_all(Tally* t) {
  unsigned long long c = 0, si[2] = {0, 0}, so[2] = {0, 0}, nb[2] = {0, 0};
  EACH(i, 1ull << 32) {
    float b = flt((uint32_t)i), a = flt(pick(i, 3)), h3 = 3.0f / b, ha = a / b;
    float r3 = __double2float_rn(3.0 / (double)b), ra = __double2float_rn((double)a / (double)b);
    c++; si[0] += subn(b); so[0] += subn(r3); si[1] += subn(a) || subn(b); so[1] += subn(ra);
    if (!same(h3, r3)) { nb[0]++; first(t, bits(b), 0, bits(h3), bits(r3)); }
    if (!same(ha, ra)) { nb[1]++; first(t + 1, bits(a), bits(b), bits(ha), bits(ra)); }
  }
  add(t, c, 0, si[0], so[0], nb[0]);
  add(t + 1, c, 0, si[1], so[1], nb[1]);
}

__global__ void u16_all(Tally* t) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, 1ull << 32) {
    uint16_t a = (uint16_t)i, b = (uint16_t)(i >> 16), q;
    if (b == 0) { s++; continue; }
    asm("div.u16 %0, %1, %2;" : "=h"(q) : "h"(a), "h"(b));
    uint32_t p = (uint32_t)q * b;
    c++;
    if (!(p <= a && a - p < b)) { nb++; first(t, a, b, q, 0); }
  }
  add(t, c, s, 0, 0, nb);
}

// ptxas specializes __cuda_sm20_div_u16 to its call sites: with the operands loaded straight into the call, as in
// ftz_probe.cu's udiv16_k, 13.0 converts the dividend as a u32 (I2FP.F32.U32.RZ) instead of as a u16 (I2F.U16.RZ, as in the
// loop above). u16_call divides N pairs that u16_fill writes from `base`, and u16_check tallies them.
__global__ void u16_fill(uint16_t* a, uint16_t* b, uint64_t base, uint64_t n) {
  EACH(j, n) { a[j] = (uint16_t)(base + j); b[j] = (uint16_t)((base + j) >> 16); }
}

__global__ void u16_call(uint16_t* y, const uint16_t* a, const uint16_t* b) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  uint16_t r;
  asm("div.u16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  y[i] = r;
}

__global__ void u16_check(Tally* t, const uint16_t* y, const uint16_t* a, const uint16_t* b, uint64_t n) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(j, n) {
    if (b[j] == 0) { s++; continue; }
    uint32_t p = (uint32_t)y[j] * b[j];
    c++;
    if (!(p <= a[j] && a[j] - p < b[j])) { nb++; first(t, a[j], b[j], y[j], 0); }
  }
  add(t, c, s, 0, 0, nb);
}

__global__ void u32_all(Tally* t) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, 1ull << 32) {
    uint32_t a = pick32(i, 1), b = pick32(i, 2), r2 = mix(i * 5 + 3);
    if ((r2 & 7) == 0) b = (a >> ((r2 >> 3) & 31)) + ((r2 >> 8) & 3) - 1;
    if (b == 0) { s++; continue; }
    uint32_t q = a / b;
    uint64_t p = (uint64_t)q * b;
    c++;
    if (!(p <= a && a - p < b)) { nb++; first(t, a, b, q, 0); }
  }
  add(t, c, s, 0, 0, nb);
}

__global__ void s32_all(Tally* t) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, 1ull << 32) {
    uint32_t r2 = mix(i * 5 + 4);
    int32_t a = (int32_t)pick32(i, 3), b = (int32_t)pick32(i, 4);
    if (r2 & 1) a = (int32_t)(0u - (uint32_t)a);
    if (r2 & 2) b = (int32_t)(0u - (uint32_t)b);
    if (b == 0 || (a == INT32_MIN && b == -1)) { s++; continue; }
    int32_t q = a / b, r = a % b;
    int64_t ar = r < 0 ? -(int64_t)r : r, ab = b < 0 ? -(int64_t)b : b;
    c++;
    if (!((int64_t)q * b + r == a && ar < ab && (r == 0 || (r < 0) == (a < 0)))) {
      nb++; first(t, (uint32_t)a, (uint32_t)b, (uint32_t)q, (uint32_t)r);
    }
  }
  add(t, c, s, 0, 0, nb);
}

__global__ void u32n_all(Tally* t, uint32_t n, uint64_t count, uint32_t salt) {
  unsigned long long c = 0, nb = 0;
  EACH(i, count) {
    uint32_t a = pick32(i, salt), q = a / n;
    uint64_t p = (uint64_t)q * n;
    c++;
    if (!(p <= a && a - p < n)) { nb++; first(t, a, n, q, 0); }
  }
  add(t, c, 0, 0, 0, nb);
}

__global__ void u64_all(Tally* t, uint64_t count) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, count) {
    uint64_t a = pick64(i, 5), b = pick64(i, 6);
    if (b == 0) { s++; continue; }
    uint64_t q = a / b, lo = q * b, hi = __umul64hi(q, b);
    c++;
    if (!(hi == 0 && lo <= a && a - lo < b)) { nb++; first(t, a, b, q, 0); }
  }
  add(t, c, s, 0, 0, nb);
}

__global__ void s64_all(Tally* t, uint64_t count) {
  unsigned long long c = 0, s = 0, nb = 0;
  EACH(i, count) {
    uint32_t r2 = mix(i * 13 + 1);
    int64_t a = (int64_t)pick64(i, 7), b = (int64_t)pick64(i, 8);
    if (r2 & 1) a = (int64_t)(0 - (uint64_t)a);
    if (r2 & 2) b = (int64_t)(0 - (uint64_t)b);
    if (b == 0 || (a == INT64_MIN && b == -1)) { s++; continue; }
    int64_t r = a % b;
    uint64_t ua = a < 0 ? 0 - (uint64_t)a : (uint64_t)a, ub = b < 0 ? 0 - (uint64_t)b : (uint64_t)b;
    uint64_t uq = ua / ub, lo = uq * ub, hi = __umul64hi(uq, ub), ur = ua - lo;
    int64_t want = a < 0 ? (int64_t)(0 - ur) : (int64_t)ur;
    c++;
    if (!(hi == 0 && lo <= ua && ur < ub && r == want)) { nb++; first(t, (uint64_t)a, (uint64_t)b, (uint64_t)r, (uint64_t)want); }
  }
  add(t, c, s, 0, 0, nb);
}

// ---- units.cu as ptxas builds it for sm_90a -------------------------------------------------------------------------------
// ptxas builds a function's slow-path bodies around its calls, so these three are copies of units.cu's kernels (its helpers,
// unchanged) that build the bodies its rmsnorm and softmax take: units_sqrt the sqrt.rn one, units_rms the div.rn one of
// rmsnorm's two divisions and units_soft softmax's, which loads its numerator (the parameter one_word, at the same offset)
// from the constant bank itself. Their kernel names are their own; only an sm_90a build has their bodies.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 900
#define UNITS 1
#else
#define UNITS 0
#endif
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#define D __device__ __forceinline__
D float fadd(float a, float b) { return __fadd_rn(a, b); }
D float fsub(float a, float b) { return __fsub_rn(a, b); }
D float fmul(float a, float b) { return __fmul_rn(a, b); }
D float fdiv(float a, float b) { return __fdiv_rn(a, b); }
D float fsqrt(float a) { return __fsqrt_rn(a); }
D float f32(u32 w) { return __uint_as_float(w); }
D float widen(u32 h) { return __uint_as_float((h & 0xFFFFu) << 16); }
D u16 bf16(float f) { u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
D float ex2(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
D int order16(u32 v) { int m = (int)(v & 0x7FFFu); return (v & 0x8000u) ? -m : m; }
D bool isnan16(u32 v) { return (v & 0x7F80u) == 0x7F80u && (v & 0x7Fu); }
#define RT 128
#define RQ 2048
D int swz(int q, bool sw) { return sw ? q ^ ((q >> 4) & 15) : q; }
D u32 rms_pair(u32 x2, u32 w2, float rs) {
  const u32 lo = bf16(fmul(widen(w2), widen(bf16(fmul(widen(x2), rs)))));
  const u32 hi = bf16(fmul(widen(w2 >> 16), widen(bf16(fmul(widen(x2 >> 16), rs)))));
  return lo | (hi << 16);
}
#define LMAX 32
#define SR 128
D int sswz(int q) { return q ^ ((q >> 3) & 3); }

// y[j] = sqrt(base + j), one word a thread
extern "C" __global__ void __launch_bounds__(128) units_sqrt(u32* y, u64 base, u64 n) {
#if UNITS
  const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) y[j] = __float_as_uint(fsqrt(f32((u32)(base + j))));
#endif
}

// units_rmsnorm with row's total tots[row] and count ns[row]: groups[row] = tot / n and rs_out[row] = one / (mean + eps)
extern "C" __global__ void __launch_bounds__(RT) units_rms(const u32* __restrict__ tots, const u32* __restrict__ ns,
                                                           const u16* __restrict__ x, const u16* __restrict__ w,
                                                           u16* __restrict__ out, u32* __restrict__ chunks,
                                                           u32* __restrict__ groups, u32* __restrict__ rs_out, int N, int C,
                                                           int nc, int G, int ng, u32 n_word, u32 eps_word, u32 one_word) {
#if UNITS
  __shared__ uint4 xs[RQ];
  __shared__ float cs[RT];
  __shared__ float gs[RT];
  __shared__ float rs_s;
  const int t = threadIdx.x, n8 = N >> 3;
  const bool sw = C == 128;
  const size_t row = blockIdx.x;
  const uint4* xp = reinterpret_cast<const uint4*>(x + row * (size_t)N);
#pragma unroll
  for (int b = 0; b < RQ / RT; b += 8) {
    uint4 v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int q = t + RT * (b + i);
      v[i] = q < n8 ? xp[q] : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int q = t + RT * (b + i);
      if (q < n8) xs[swz(q, sw)] = v[i];
    }
  }
  __syncthreads();
  if (t < nc) {
    const int K = C >> 3;
    float acc = 0.0f;
#pragma unroll 16
    for (int k = 0; k < K; ++k) {
      const uint4 v = xs[swz(t * K + k, sw)];
      const u32 q[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const float a = widen(q[h]), b = widen(q[h] >> 16);
        acc = fadd(acc, fmul(a, a));
        acc = fadd(acc, fmul(b, b));
      }
    }
    cs[t] = acc;
    if (chunks) chunks[row * nc + t] = __float_as_uint(acc);
  }
  __syncthreads();
  if (t < ng) {
    float acc = 0.0f;
#pragma unroll 16
    for (int k = 0; k < G; ++k) acc = fadd(acc, cs[t * G + k]);
    gs[t] = acc;
    if (groups) groups[row * ng + t] = __float_as_uint(acc);
  }
  __syncthreads();
  if (t == 0) {
    float total = 0.0f;
    total = f32(tots[row]);
    const float mean = fdiv(total, f32(ns[row]));
    if (groups) groups[row] = __float_as_uint(mean);
    const float rs = fdiv(f32(one_word), fadd(mean, f32(eps_word)));
    rs_s = rs;
    if (rs_out) rs_out[row] = __float_as_uint(rs);
  }
  __syncthreads();
  const float rs = rs_s;
  const uint4* wp = reinterpret_cast<const uint4*>(w);
  uint4* op = reinterpret_cast<uint4*>(out + row * (size_t)N);
#pragma unroll 4
  for (int q = t; q < n8; q += RT) {
    const uint4 a = xs[swz(q, sw)], b = wp[q];
    op[q] = make_uint4(rms_pair(a.x, b.x, rs), rms_pair(a.y, b.y, rs), rms_pair(a.z, b.z, rs), rms_pair(a.w, b.w, rs));
  }
#endif
}

// units_softmax with the divisor s k + sv[row]: at rows of one key (s = 1) and k = -0, r_out[row] = one / sv[row]
extern "C" __global__ void __launch_bounds__(SR) units_soft(const u16* __restrict__ x, const int* __restrict__ lens,
                                                            u16* __restrict__ p, u16* __restrict__ m_out,
                                                            u32* __restrict__ r_out, int heads, u32 one_word, u32 k_word,
                                                            const u32* __restrict__ sv) {
#if UNITS
  __shared__ uint4 buf[SR * 4];
  const int t = threadIdx.x, h0 = blockIdx.x * SR, rows = min(SR, heads - h0), nq = rows * 4;
  const size_t row0 = (size_t)blockIdx.y * heads + h0;
  const uint4* xp = reinterpret_cast<const uint4*>(x + row0 * LMAX);
  {
    uint4 a[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q = t + SR * i;
      a[i] = q < nq ? xp[q] : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q = t + SR * i;
      if (q < nq) buf[sswz(q)] = a[i];
    }
  }
  __syncthreads();
  int n = lens[blockIdx.y];
  n = n < 1 ? 1 : (n > LMAX ? LMAX : n);
  u32 pw[LMAX / 2] = {};
  if (t < rows) {
    u32 v[LMAX];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint4 b = buf[sswz(t * 4 + k)];
      const u32 q[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const int j = 8 * k + 2 * h;
        v[j] = j < n ? (q[h] & 0xFFFFu) : 0u;
        v[j + 1] = j + 1 < n ? (q[h] >> 16) : 0u;
      }
    }
    u32 m = v[0];
    int best = order16(v[0]);
    bool nan = isnan16(v[0]);
#pragma unroll
    for (int j = 1; j < LMAX; ++j) {
      if (j < n) {
        const int o = order16(v[j]);
        if (o > best) { best = o; m = v[j]; }
        nan = nan || isnan16(v[j]);
      }
    }
    if (nan) m = 0x7FFFu;
    const float mf = widen(m), log2e = f32(0x3FB8AA3Bu);
    float e[LMAX];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < LMAX; ++j) {
      e[j] = 0.0f;
      if (j < n) {
        e[j] = ex2(fmul(fsub(widen(v[j]), mf), log2e));
        s = fadd(s, e[j]);
      }
    }
    const float r = fdiv(f32(one_word), fadd(fmul(s, f32(k_word)), f32(sv[row0 + t])));
#pragma unroll
    for (int j = 0; j < LMAX; j += 2)
      pw[j / 2] = (j < n ? (u32)bf16(fmul(e[j], r)) : 0u) | ((j + 1 < n ? (u32)bf16(fmul(e[j + 1], r)) : 0u) << 16);
    if (m_out) m_out[row0 + t] = (u16)m;
    if (r_out) r_out[row0 + t] = __float_as_uint(r);
  }
  __syncthreads();
  if (t < rows) {
#pragma unroll
    for (int k = 0; k < 4; ++k) buf[sswz(t * 4 + k)] = make_uint4(pw[4 * k], pw[4 * k + 1], pw[4 * k + 2], pw[4 * k + 3]);
  }
  __syncthreads();
  uint4* pp = reinterpret_cast<uint4*>(p + row0 * LMAX);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = t + SR * i;
    if (q < nq) pp[q] = buf[sswz(q)];
  }
#endif
}

// the operands: a[j] = base + j (every word, when `all`) or pick(base + j, sa), b[j] = pick(base + j, sb)
__global__ void units_fill(uint32_t* a, uint32_t* b, uint64_t base, uint64_t n, bool all, uint32_t sa, uint32_t sb) {
  EACH(j, n) {
    if (a) a[j] = all ? (uint32_t)(base + j) : pick(base + j, sa);
    if (b) b[j] = pick(base + j, sb);
  }
}

__global__ void units_sqrt_check(Tally* t, const uint32_t* y, uint64_t base, uint64_t n) {
  unsigned long long c = 0, si = 0, so = 0, nb = 0;
  EACH(j, n) {
    float x = flt((uint32_t)(base + j)), h = flt(y[j]), r = __double2float_rn(sqrt((double)x));
    c++; si += subn(x); so += subn(r);
    if (!same(h, r)) { nb++; first(t, bits(x), 0, bits(h), bits(r)); }
  }
  add(t, c, 0, si, so, nb);
}

// t[0]: num[j] / den[j] against q[j]; t[1], with `second`: one / (q's reference + eps) against q2[j]
__global__ void units_div_check(Tally* t, const uint32_t* num, uint32_t one, const uint32_t* den, const uint32_t* q,
                                const uint32_t* q2, uint32_t eps, uint64_t n) {
  unsigned long long c = 0, si[2] = {0, 0}, so[2] = {0, 0}, nb[2] = {0, 0};
  EACH(j, n) {
    float a = flt(num ? num[j] : one), b = flt(den[j]), h = flt(q[j]), r = __double2float_rn((double)a / (double)b);
    c++; si[0] += subn(a) || subn(b); so[0] += subn(r);
    if (!same(h, r)) { nb[0]++; first(t, bits(a), bits(b), bits(h), bits(r)); }
    if (q2) {
      float d = __fadd_rn(r, flt(eps)), h2 = flt(q2[j]), r2 = __double2float_rn((double)flt(one) / (double)d);
      si[1] += subn(flt(one)) || subn(d); so[1] += subn(r2);
      if (!same(h2, r2)) { nb[1]++; first(t + 1, one, bits(d), bits(h2), bits(r2)); }
    }
  }
  add(t, c, 0, si[0], so[0], nb[0]);
  if (q2) add(t + 1, c, 0, si[1], so[1], nb[1]);
}

__global__ void apply(const uint32_t* x, const uint32_t* y, uint32_t* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = flt(x[i]), b = flt(y[i]);
    out[4 * i] = bits(__frcp_rn(a)); out[4 * i + 1] = bits(sqrtf(a)); out[4 * i + 2] = bits(a / b);
    out[4 * i + 3] = bits(3.0f / b);
  }
}

static void print(const char* check, const Tally& t) {
  printf("{\"check\":\"%s\",\"cases\":%llu,\"skipped\":%llu,\"subnormal_in\":%llu,\"subnormal_out\":%llu,\"bad\":%llu,"
         "\"first\":[%llu,%llu,%llu,%llu]}\n", check, t.cases, t.skipped, t.sub_in, t.sub_out, t.bad, t.first[0], t.first[1],
         t.first[2], t.first[3]);
  fflush(stdout);
}

// units.cu's bodies: units_sqrt on all 2^32 words; units_rms on 2^32 weighted (total, count) pairs, rows of 8 words, with
// one = pick(chunk) and eps = -0, so its second division is one / mean; units_soft at rows of one key with k = -0, one = 1.0
// on all 2^32 divisors, then 2^32 weighted divisors at 1024 numerators pick(chunk)
static void units_checks(int grid, Tally* t) {
  const uint64_t n = 1ull << 24, all = 1ull << 32;
  const uint32_t neg0 = 0x80000000u, one = 0x3f800000u;
  uint32_t *a, *b, *y, *y2;
  uint16_t *x, *w, *out;
  int* lens;
  CK(cudaMalloc(&a, n * 4)); CK(cudaMalloc(&b, n * 4)); CK(cudaMalloc(&y, n * 4)); CK(cudaMalloc(&y2, n * 4));
  CK(cudaMalloc(&x, n * 64)); CK(cudaMalloc(&out, n * 64)); CK(cudaMalloc(&w, 16)); CK(cudaMalloc(&lens, 4));
  CK(cudaMemset(x, 0, n * 64)); CK(cudaMemset(w, 0, 16));
  const int one_key = 1;
  CK(cudaMemcpy(lens, &one_key, 4, cudaMemcpyHostToDevice));
  auto done = [&] { CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); };
  memset(t, 0, 2 * sizeof(Tally));
  for (uint64_t base = 0; base < all; base += n) {
    units_sqrt<<<(unsigned)(n / 128), 128>>>(y, base, n);
    units_sqrt_check<<<grid, 256>>>(t, y, base, n);
  }
  done(); print("units_sqrt", t[0]);
  memset(t, 0, 2 * sizeof(Tally));
  for (uint64_t base = 0, k = 0; base < all; base += n, k++) {
    const uint32_t num = pick(k, 11);
    units_fill<<<grid, 256>>>(a, b, base, n, false, 9, 10);
    units_rms<<<(unsigned)n, RT>>>(a, b, x, w, out, nullptr, y, y2, 8, 8, 1, 1, 1, 0u, neg0, num);
    units_div_check<<<grid, 256>>>(t, a, num, b, y, y2, neg0, n);
  }
  done(); print("units_rms_mean", t[0]); print("units_rms_rs", t[1]);
  const uint64_t m = 1ull << 22;
  memset(t, 0, 2 * sizeof(Tally));
  for (uint64_t base = 0; base < all; base += m) {
    units_fill<<<grid, 256>>>(b, nullptr, base, m, true, 0, 0);
    units_soft<<<dim3((unsigned)(m / SR), 1), SR>>>(x, lens, out, nullptr, y, (int)m, one, neg0, b);
    units_div_check<<<grid, 256>>>(t, nullptr, one, b, y, nullptr, 0u, m);
  }
  done(); print("units_soft_one", t[0]);
  memset(t, 0, 2 * sizeof(Tally));
  for (uint64_t base = 0, k = 0; base < all; base += m, k++) {
    const uint32_t num = pick(k, 12);
    units_fill<<<grid, 256>>>(nullptr, b, base, m, false, 0, 13);
    units_soft<<<dim3((unsigned)(m / SR), 1), SR>>>(x, lens, out, nullptr, y, (int)m, num, neg0, b);
    units_div_check<<<grid, 256>>>(t, nullptr, num, b, y, nullptr, 0u, m);
  }
  done(); print("units_soft_div", t[0]);
  CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(y)); CK(cudaFree(y2)); CK(cudaFree(x)); CK(cudaFree(out)); CK(cudaFree(w));
  CK(cudaFree(lens));
}

static void host_check() {
  std::vector<uint32_t> x, y;
  for (uint32_t s = 0; s < 2; s++)
    for (uint32_t m = 1; m < (1u << 23); m++) x.push_back(s << 31 | m);
  for (uint32_t s = 0; s < 2; s++)
    for (uint32_t e = 253; e <= 254; e++)
      for (uint32_t m = 0; m < (1u << 23); m++) x.push_back(s << 31 | e << 23 | m);
  for (size_t i = 0; i < x.size(); i++) y.push_back(pick(i, 2));
  size_t n = x.size();
  uint32_t *dx, *dy, *dout;
  CK(cudaMalloc(&dx, n * 4)); CK(cudaMalloc(&dy, n * 4)); CK(cudaMalloc(&dout, n * 16));
  CK(cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dy, y.data(), n * 4, cudaMemcpyHostToDevice));
  apply<<<(unsigned)((n + 255) / 256), 256>>>(dx, dy, dout, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<uint32_t> out(n * 4);
  CK(cudaMemcpy(out.data(), dout, n * 16, cudaMemcpyDeviceToHost));
  CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(dout));
  Tally t[4] = {};
  for (size_t i = 0; i < n; i++) {
    float a = flt(x[i]), b = flt(y[i]);
    float ref[4] = {(float)(1.0 / (double)a), (float)std::sqrt((double)a), (float)((double)a / (double)b),
                    (float)(3.0 / (double)b)};
    bool in[4] = {subn(a), subn(a), subn(a) || subn(b), subn(b)};
    for (int k = 0; k < 4; k++) {
      float h = flt(out[4 * i + k]);
      t[k].cases++; t[k].sub_in += in[k]; t[k].sub_out += subn(ref[k]);
      if (!same(h, ref[k])) {
        if (!t[k].bad) { t[k].first[0] = x[i]; t[k].first[1] = y[i]; t[k].first[2] = bits(h); t[k].first[3] = bits(ref[k]); }
        t[k].bad++;
      }
    }
  }
  print("host_rcp", t[0]); print("host_sqrt", t[1]); print("host_div", t[2]); print("host_cdiv3", t[3]);
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const unsigned char* u = (const unsigned char*)p.uuid.bytes;
  printf("{\"device\":\"%s\",\"uuid\":\"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\","
         "\"sm\":%d%d,\"sms\":%d}\n", p.name, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12],
         u[13], u[14], u[15], p.major, p.minor, p.multiProcessorCount);
  const int grid = p.multiProcessorCount * 8;
  Tally* t;
  CK(cudaMallocManaged(&t, 2 * sizeof(Tally)));
  auto run = [&](const char* check, auto launch) {
    memset(t, 0, 2 * sizeof(Tally));
    launch();
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    print(check, t[0]);
  };
  run("rcp", [&] { rcp_all<<<grid, 256>>>(t); });
  run("sqrt", [&] { sqrt_all<<<grid, 256>>>(t); });
  run("div", [&] { div_all<<<grid, 256>>>(t); });
  memset(t, 0, 2 * sizeof(Tally));
  cdiv_all<<<grid, 256>>>(t);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  print("cdiv3", t[0]); print("cdiv", t[1]);
  run("u16", [&] { u16_all<<<grid, 256>>>(t); });
  run("u16_call", [&] {
    const uint64_t n = 1ull << 28;
    uint16_t *a, *b, *y;
    CK(cudaMalloc(&a, n * 2)); CK(cudaMalloc(&b, n * 2)); CK(cudaMalloc(&y, n * 2));
    for (uint64_t base = 0; base < (1ull << 32); base += n) {
      u16_fill<<<grid, 256>>>(a, b, base, n);
      u16_call<<<(unsigned)(n / 256), 256>>>(y, a, b);
      u16_check<<<grid, 256>>>(t, y, a, b, n);
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(y));
  });
  run("u32", [&] { u32_all<<<grid, 256>>>(t); });
  run("s32", [&] { s32_all<<<grid, 256>>>(t); });
  run("u32n", [&] {
    const uint32_t fixed[] = {1u, 2u, 3u, 5u, 7u, 10u, 255u, 256u, 257u, 0xffffu, 0x10000u, 0x10001u, 0x7fffffffu,
                              0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu};
    for (uint32_t k = 0; k < 64; k++) {
      uint32_t n = k < 17 ? fixed[k] : (mix(k) >> (mix(k + 1000) & 31)) | 1u;
      u32n_all<<<grid, 256>>>(t, n, 1ull << 26, k + 1);
    }
  });
  run("u64", [&] { u64_all<<<grid, 256>>>(t, 1ull << 30); });
  run("s64", [&] { s64_all<<<grid, 256>>>(t, 1ull << 30); });
  if (p.major == 9) units_checks(grid, t);
  host_check();
  return 0;
}
