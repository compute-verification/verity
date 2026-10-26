// The frozen decoder's non-matmul units on sm_90a, each word its twin's (words.py and the unit modules' OPS docstrings):
// RMSNorm, the per-row s8 quantizer, softmax over a QK^T row, argmax over the head, ReLU, and units_sweep, which runs the
// helpers below on every operand word so check.py can hold them to the twins' instructions exhaustively.
//
// Every f32 operation is a named IEEE round-to-nearest-even instruction (__fadd_rn, __fsub_rn, __fmul_rn, __fdiv_rn,
// __fsqrt_rn), built with -fmad=false and no FTZ flag, so nothing is contracted or flushed; the one approximate operation
// is ex2.approx.ftz.f32 (MUFU.EX2, `mufu.MufuEx2Ftz`), in softmax only.  Every reduction is sequential in a fixed order
// from +0 by one thread; threads meet only through shared memory and __syncthreads.  No atomics.
//
// Every division's operands are runtime values (the constants 1 and 127 come in as kernel arguments): ptxas folds a
// constant numerator into a copy of div.rn's slow path that the harness SASS gate's pins don't cover.  No kernel divides
// integers (the host passes the counts), so the integer kernels hold no floating-point instruction at all.
#include <cstdint>

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
D u32 s8(float x) { int r; asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(r) : "f"(x)); return (u32)r & 0xFFu; }
D int order16(u32 v) { int m = (int)(v & 0x7FFFu); return (v & 0x8000u) ? -m : m; }
D bool isnan16(u32 v) { return (v & 0x7F80u) == 0x7F80u && (v & 0x7Fu); }
D u32 relu16(u32 v) { v &= 0xFFFFu; return (!isnan16(v) && order16(v) < 0) ? 0u : v; }
// RN(amax / 127) when 1 <= f32(amax) <= +inf as signed words, else 1.0; f127 is 127.0's word
D u32 quant_scale(u32 amax, u32 f127) {
  const u32 a = (amax & 0xFFFFu) << 16;
  return ((int)a >= 1 && (int)a <= 0x7F800000) ? __float_as_uint(fdiv(f32(a), f32(f127))) : 0x3F800000u;
}

// ---- RMSNorm: one block per row.  N = nc C, nc = ng G, C % 8 == 0, nc <= blockDim.x <= 1024 -------------------------
D u32 rms_pair(u32 x2, u32 w2, float rs) {
  const u32 lo = bf16(fmul(widen(w2), widen(bf16(fmul(widen(x2), rs)))));
  const u32 hi = bf16(fmul(widen(w2 >> 16), widen(bf16(fmul(widen(x2 >> 16), rs)))));
  return lo | (hi << 16);
}

extern "C" __global__ void units_rmsnorm(const u16* __restrict__ x, const u16* __restrict__ w, u16* __restrict__ out,
                                         u32* __restrict__ chunks, u32* __restrict__ groups, u32* __restrict__ rs_out,
                                         int N, int C, int nc, int G, int ng, u32 n_word, u32 eps_word, u32 one_word) {
  __shared__ float cs[1024];
  __shared__ float gs[1024];
  __shared__ float rs_s;
  const int t = threadIdx.x;
  const size_t row = blockIdx.x;
  const u16* xr = x + row * (size_t)N;
  if (t < nc) {
    const uint4* p = reinterpret_cast<const uint4*>(xr + (size_t)t * C);
    float acc = 0.0f;
    for (int k = 0; k < (C >> 3); ++k) {
      const uint4 v = p[k];
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
    for (int k = 0; k < G; ++k) acc = fadd(acc, cs[t * G + k]);
    gs[t] = acc;
    if (groups) groups[row * ng + t] = __float_as_uint(acc);
  }
  __syncthreads();
  if (t == 0) {
    float total = 0.0f;
    for (int k = 0; k < ng; ++k) total = fadd(total, gs[k]);
    const float mean = fdiv(total, f32(n_word));
    const float rs = fdiv(f32(one_word), fsqrt(fadd(mean, f32(eps_word))));
    rs_s = rs;
    if (rs_out) rs_out[row] = __float_as_uint(rs);
  }
  __syncthreads();
  const float rs = rs_s;
  const uint4* xp = reinterpret_cast<const uint4*>(xr);
  const uint4* wp = reinterpret_cast<const uint4*>(w);
  uint4* op = reinterpret_cast<uint4*>(out + row * (size_t)N);
  for (int i = t; i < (N >> 3); i += blockDim.x) {
    const uint4 a = xp[i], b = wp[i];
    op[i] = make_uint4(rms_pair(a.x, b.x, rs), rms_pair(a.y, b.y, rs), rms_pair(a.z, b.z, rs), rms_pair(a.w, b.w, rs));
  }
}

// ---- the s8 quantizer: one block per row.  The absmax tree's levels take chunks of min(128, width) words, level 1 over
// the (relu'd) words with their sign bits cleared; tree gets every level's words in order, the root last.  N % 8 == 0,
// every width >= 128 a multiple of 128, and level 1's width <= blockDim.x <= 1024 -----------------------------------------
D u32 quant2(u32 v2, float s, int relu) {
  const u32 a = relu ? relu16(v2) : (v2 & 0xFFFFu), b = relu ? relu16(v2 >> 16) : (v2 >> 16);
  return s8(fdiv(widen(a), s)) | (s8(fdiv(widen(b), s)) << 8);
}

D u32 amax2(u32 best, u32 v2, int relu) {
  const u32 a = (relu ? relu16(v2) : (v2 & 0xFFFFu)) & 0x7FFFu, b = (relu ? relu16(v2 >> 16) : (v2 >> 16)) & 0x7FFFu;
  return max(best, max(a, b));
}

extern "C" __global__ void units_quant(const u16* __restrict__ x, uint8_t* __restrict__ codes, u32* __restrict__ scale_out,
                                       u16* __restrict__ tree, int N, int T, int relu, u32 f127) {
  __shared__ u16 lv[2][1024];
  __shared__ u32 sc;
  const int t = threadIdx.x;
  const size_t row = blockIdx.x;
  const u16* xr = x + row * (size_t)N;
  u16* tr = tree ? tree + row * (size_t)T : nullptr;
  int a = N < 128 ? N : 128, width = N < 128 ? 1 : N >> 7;
  if (t < width) {
    const uint4* p = reinterpret_cast<const uint4*>(xr + (size_t)t * a);
    u32 best = 0;
    for (int k = 0; k < (a >> 3); ++k) {
      const uint4 v = p[k];
      best = amax2(amax2(amax2(amax2(best, v.x, relu), v.y, relu), v.z, relu), v.w, relu);
    }
    lv[0][t] = (u16)best;
    if (tr) tr[t] = (u16)best;
  }
  __syncthreads();
  int cur = 0, off = width;
  while (width > 1) {
    a = width < 128 ? width : 128;
    const int next = width < 128 ? 1 : width >> 7;
    if (t < next) {
      u32 best = 0;
      for (int k = 0; k < a; ++k) best = max(best, (u32)lv[cur][t * a + k]);
      lv[cur ^ 1][t] = (u16)best;
      if (tr) tr[off + t] = (u16)best;
    }
    __syncthreads();
    cur ^= 1;
    off += next;
    width = next;
  }
  if (t == 0) {
    const u32 s = quant_scale(lv[cur][0], f127);
    sc = s;
    if (scale_out) scale_out[row] = s;
  }
  __syncthreads();
  const float s = f32(sc);
  const uint4* xp = reinterpret_cast<const uint4*>(xr);
  uint2* cp = reinterpret_cast<uint2*>(codes + row * (size_t)N);
  for (int i = t; i < (N >> 3); i += blockDim.x) {
    const uint4 v = xp[i];
    cp[i] = make_uint2(quant2(v.x, s, relu) | (quant2(v.y, s, relu) << 16), quant2(v.z, s, relu) | (quant2(v.w, s, relu) << 16));
  }
}

// ---- softmax over one QK^T row: one thread per row, row b heads + h of sequence b = blockIdx.y at n = lens[b] keys
// (clamped to 1..L, L <= 32) ----------------------------------------------------------------------------------------------
#define LMAX 32
extern "C" __global__ void units_softmax(const u16* __restrict__ x, const int* __restrict__ lens, u16* __restrict__ p,
                                         u16* __restrict__ m_out, u32* __restrict__ r_out, int heads, int L, u32 one_word) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= heads) return;
  const size_t row = (size_t)blockIdx.y * heads + h;
  int n = lens[blockIdx.y];
  n = n < 1 ? 1 : (n > L ? L : n);
  const u16* xr = x + row * L;
  u32 v[LMAX];
#pragma unroll
  for (int j = 0; j < LMAX; ++j) v[j] = j < n ? (u32)xr[j] : 0u;
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
  const float r = fdiv(f32(one_word), s);
  u16* pr = p + row * L;
#pragma unroll
  for (int j = 0; j < LMAX; ++j)
    if (j < L) pr[j] = j < n ? bf16(fmul(e[j], r)) : (u16)0;
  if (m_out) m_out[row] = (u16)m;
  if (r_out) r_out[row] = __float_as_uint(r);
}

// ---- argmax over a row of V logits in nch chunks of C (the last may be short): one thread per row ------------------------
extern "C" __global__ void units_argmax(const u16* __restrict__ x, u16* __restrict__ best_out, u16* __restrict__ cidx_out,
                                        u16* __restrict__ out, int R, int V, int C, int nch) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= R) return;
  const u16* xr = x + (size_t)row * V;
  u32 run = xr[0];
  int run_i = 0;
  for (int c = 0; c < nch; ++c) {
    u32 b = 0xFF80u;
    int bi = 0;
    const int lo = c * C, hi = min(V, lo + C);
    for (int i = lo; i < hi; ++i) {
      const u32 v = xr[i];
      if (!isnan16(v) && order16(v) > order16(b)) { b = v; bi = i - lo; }
    }
    if (best_out) best_out[(size_t)row * nch + c] = (u16)b;
    if (cidx_out) cidx_out[(size_t)row * nch + c] = (u16)bi;
    if (!isnan16(b) && !isnan16(run) && order16(b) > order16(run)) { run = b; run_i = lo + bi; }
  }
  out[row] = (u16)(run_i & 0xFFFF);
}

// ---- ReLU on bf16 words ---------------------------------------------------------------------------------------------------
extern "C" __global__ void units_relu(const u16* __restrict__ x, u16* __restrict__ y, u64 n) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) y[i] = (u16)relu16(x[i]);
}

// ---- the sweep: block b of 1024 threads covers operands i in [b 2^24, (b + 1) 2^24) below n, and writes
// sum_i splitmix64((i << 32) | op(i)) mod 2^64 (sweep.py) --------------------------------------------------------------
D u64 splitmix64(u64 z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

D u32 sweep_op(int op, u32 i, u32 arg) {
  switch (op) {
    case 0: return __float_as_uint(ex2(f32(i)));
    case 1: return s8(f32(i));
    case 2: return __float_as_uint(fsqrt(f32(i)));
    case 3: return __float_as_uint(fdiv(f32(arg), f32(i)));
    case 4: return __float_as_uint(fdiv(f32(i), f32(arg)));
    case 5: return __float_as_uint(fdiv(widen(i), f32(quant_scale(i >> 16, arg))));
    case 6: return bf16(f32(i));
    case 7: return quant_scale(i, arg);
    default: return 0xFFFFFFFFu;
  }
}

extern "C" __global__ void __launch_bounds__(1024) units_sweep(int op, u64 n, u32 arg, u64* __restrict__ partial) {
  __shared__ u64 ws[32];
  const u64 base = (u64)blockIdx.x << 24;
  u64 acc = 0;
  for (u32 k = 0; k < (1u << 14); ++k) {
    const u64 i = base | ((u64)k << 10) | threadIdx.x;
    if (i >= n) break;
    acc += splitmix64((i << 32) | sweep_op(op, (u32)i, arg));
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, o);
  if ((threadIdx.x & 31) == 0) ws[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    u64 total = 0;
    for (int k = 0; k < 32; ++k) total += ws[k];
    partial[blockIdx.x] = total;
  }
}
